# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests for ask_and_check.py ("the client"), against a fake server and a fake check script.

No proof is made or verified here. Run them from integrations/ with: pytest tests
"""

import contextlib
import errno
import http.server
import io
import json
import os
import posix
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types

import pytest

import ask_and_check
import check_proof
import proof_middleware
import spool

COMMITMENT_ID, PROOF_ID = "ab" * 32, "12" * 32
JOB = "/plugins/proof/jobs/" + PROOF_ID
PROMPT, REPLY, END = [1, 2, 3], [4, 5, 7], 7
# A parameter file (policy.txt) for exact match and request files of 2^4 tokens.
POLICY = "lt 4\nV 200\nC 256\ntheta 256\ncap 0\ns 15\nsh 8\nsT 4\nF 24\n"
# Replaces check_proof.py. It writes what it was given to seen.json, then prints and exits as
# Client.check_exits_with() set (code -N: it sends itself signal N). START:END in that output
# becomes the request file's range.
FAKE_CHECK = """import json, os, signal, struct, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(sys.argv[2], "rb") as f:
    request = f.read()
start_end = "%d:%d" % struct.unpack_from("<2I", request, 4 * 16)
request = request.hex()
with open(os.path.join(here, "seen.json"), "w") as f:
    names = ("VLLM_API_KEY", "CHECK_PROOF_FETCH_KEY")
    keys = {name: os.environ[name] for name in names if name in os.environ}
    seen = {"argv": sys.argv[1:], "request": request, "keys": keys, "isolated": sys.flags.isolated}
    seen["0 and 2"] = [os.readlink("/proc/self/fd/%d" % fd) for fd in (0, 2)]
    with open(os.path.join(sys.argv[1], "proof.bin"), "rb") as g:
        seen["proof"] = g.read().hex()
    json.dump(seen, f)
with open(os.path.join(here, "status")) as f:
    code, lines = f.read().split("\\n", 1)
sys.stdout.write(lines.replace("START:END", start_end))
sys.stdout.flush()
if int(code) < 0:
    signal.raise_signal(-int(code))
sys.exit(int(code))
"""


class QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """Ignore a client that hung up; report any other error."""
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class FakeServer:
    """A fake vLLM server with the routes of proof_middleware.py that the client calls.

    To make it misbehave, change its attributes, or set instead[method, path] to (status, headers,
    body); a header whose value is None is not sent. 'calls' lists the requests, each as (method,
    path, headers, body).
    """

    def __init__(self):
        self.commitment_id = COMMITMENT_ID
        self.prompt, self.reply, self.text = PROMPT, REPLY, "some text"
        self.job = {"state": "proved"}
        self.files = {"y.bin": b"y" * 8, "proof.bin": b"p" * 40}
        self.instead, self.calls = {}, []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def respond(self):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                headers = {name.lower(): value for name, value in self.headers.items()}
                fake.calls.append((self.command, self.path, headers, body))
                planned = fake.instead.get((self.command, self.path))
                status, headers, data = planned or fake.response(self.command, self.path)
                self.send_response(status)
                for name, value in {"content-length": str(len(data)), **headers}.items():
                    if value is not None:
                        self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = respond

            def log_message(self, *args):
                pass

        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, args=(0.01,), daemon=True).start()

    def response(self, method, path):
        ids = {"x-proof-id": PROOF_ID}
        if path == "/plugins/proof/policies":
            return (
                200,
                {},
                json.dumps({"policies": {"exact": {"commitment_id": self.commitment_id}}}).encode(),
            )
        if path == "/v1/completions":
            return (
                200,
                ids,
                json.dumps(
                    {
                        "choices": [
                            {
                                "text": self.text,
                                "token_ids": self.reply,
                                "prompt_token_ids": self.prompt,
                            }
                        ]
                    }
                ).encode(),
            )
        if path == "/v1/chat/completions":
            choice = {"message": {"content": self.text}, "token_ids": self.reply}
            return (
                200,
                ids,
                json.dumps({"prompt_token_ids": self.prompt, "choices": [choice]}).encode(),
            )
        if path == JOB:
            return 200, {}, json.dumps(self.job if method == "GET" else {}).encode()
        if path.startswith(JOB + "/") and path[len(JOB) + 1 :] in self.files:
            return 200, {}, self.files[path[len(JOB) + 1 :]]
        return 404, {}, b'{"error": {"message": "not found"}}'

    def routes(self):
        return [call[0] + " " + call[1] for call in self.calls]


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.httpd.shutdown()
    fake.httpd.server_close()


WITHIN_LIMITS = "check_proof: within limits: fake"  # a last line of check_proof.py at exit code 0
# check_proof.py's line before the last. PASS omits some fields.
PASS = "result: outcome=%%s commitment_id=%s range=START:END claim=%%s leaf=0" % COMMITMENT_ID
RESULTS = {
    0: PASS % ("verified", "exact-match"),
    1: "result: outcome=rejected",
    2: "result: outcome=error",
    14: "result: outcome=not-a-verification",
}
WITHIN_LIMITS_RESULT = PASS % ("within-limits", "policy")
# outcome= in the last "result:" line on stdout, by the client's exit code.
OUTCOMES = {
    0: ("verified", "within-limits"),
    1: ("rejected", "no-proof"),
    2: ("error", None),  # None: no such line (a usage error)
    14: ("not-a-verification",),
    24: ("server-chose-prompt",),
    75: ("try-again", "stored"),  # "stored" with --ask-only
}


def outcome(out):
    """Return outcome= of the last stdout line that begins with "result:", or None."""
    lines = [line for line in out.splitlines() if line.startswith("result:")]
    return lines[-1].split()[1].removeprefix("outcome=") if lines else None


class Client:
    """Run ask_and_check.main() with a complete command line.

    Keyword arguments of run() override options; None removes one. Every run() also compares the
    exit code with the last "result:" line (OUTCOMES).
    """

    def __init__(self, tmp_path, server, capfd):
        self.dir, self.server, self.capfd, self.out_dir = tmp_path, server, capfd, tmp_path / "out"
        (tmp_path / "key").mkdir()
        (tmp_path / "policy.txt").write_text(POLICY)
        (tmp_path / "KL.bin").write_bytes(b"scales")
        (tmp_path / "check.py").write_text(FAKE_CHECK)
        self.check_exits_with(0)

    def check_exits_with(self, code, last_line=None, result=None, end="\n"):
        """Set the fake check script's exit code and output. result: the line before the last."""
        if last_line is None:
            last_line = "check_proof: %s: fake" % ("rejected" if code else "verified")
        if result is None:
            result = RESULTS.get(code)
        lines = ["some output", *([result] if result else []), last_line]
        (self.dir / "status").write_text("%d\n%s%s" % (code, "\n".join(lines), end))

    def seen_by_check(self):
        return json.loads((self.dir / "seen.json").read_text())

    def checked(self):
        return (self.dir / "seen.json").exists()

    def argv(self, *flags, after=(), **options):
        given = {
            "server": self.server.url,
            "model": "m",
            "policy_name": "exact",
            "commitment_id": COMMITMENT_ID,
            "public_key_dir": self.dir / "key",
            "parameter_file": self.dir / "policy.txt",
            "scales_file": self.dir / "KL.bin",
            "out_dir": self.out_dir,
            "end_token": END,
            "prompt_ids": "1,2,3",
            "max_tokens": 5,
            "wait": 0,
            **options,
        }
        argv = [
            arg
            for name, value in given.items()
            if value is not None
            for arg in ("--" + name.replace("_", "-"), str(value))
        ]
        return [*argv, *flags, *(["--", *after] if after else [])]

    def run(self, *flags, after=(), **options):
        try:
            status = ask_and_check.main(self.argv(*flags, after=after, **options))
        except SystemExit as exit:
            status = exit.code
        out, err = self.capfd.readouterr()
        assert "Traceback" not in err
        assert outcome(out) in OUTCOMES[status], (status, out)
        assert (outcome(out) is None) == err.startswith("usage: ")
        return status, out, err


@pytest.fixture
def client(tmp_path, server, capfd, monkeypatch):
    monkeypatch.setattr(ask_and_check, "CHECK_SCRIPT", str(tmp_path / "check.py"))
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    # Reach the fake server directly, even if http_proxy is set.
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1")
    return Client(tmp_path, server, capfd)


# Options for Client.run(): a text prompt instead of token IDs.
TEXT = {"prompt_ids": None, "prompt": "hello"}


# ---- A proof that passes ----


def test_verified_with_prompt_ids(client, server):
    status, out, err = client.run("--show-checked")
    assert (status, err) == (0, "")
    assert "checked: the reply's token IDs 4,5,7 after a prompt of 3 token IDs; SHA-256" in out
    assert out.splitlines()[-3:] == [
        "some output",
        RESULTS[0].replace("START:END", "2:5"),
        "check_proof: verified: fake",
    ]
    assert sorted(os.listdir(client.out_dir)) == ["proof", "request.bin", "served.json"]
    assert (client.out_dir / "proof" / "proof.bin").read_bytes() == server.files["proof.bin"]
    assert (client.out_dir / "proof" / "y.bin").read_bytes() == server.files["y.bin"]
    assert client.out_dir.stat().st_mode & 0o777 == 0o700
    assert (client.out_dir / "served.json").stat().st_mode & 0o777 == 0o600
    assert server.routes()[-1] == "DELETE " + JOB


def test_check_script_gets_the_callers_values(client, server):
    server.job["request"] = {"total_tokens": 1024, "prompt_tokens": 1, "reply_tokens": 5}
    assert client.run()[0] == 0
    seen = client.seen_by_check()
    assert all(re.fullmatch("/dev/fd/[0-9]+", name) for name in seen["argv"][:2])
    assert seen["argv"][2:] == [
        "--fetch-from",
        server.url + "/plugins/proof/published/exact",
        "--commitment-id",
        COMMITMENT_ID,
        "--public-key-dir",
        str(client.dir / "key"),
        "--parameter-file",
        str(client.dir / "policy.txt"),
        "--scales-file",
        str(client.dir / "KL.bin"),
        "--reply",
        "3:3",
        "--end-token",
        "7",
    ]
    assert bytes.fromhex(seen["request"]) == spool.whole_reply(PROMPT, REPLY, 16, fill=0)
    assert (client.out_dir / "request.bin").read_bytes() == bytes.fromhex(seen["request"])
    assert bytes.fromhex(seen["proof"]) == server.files["proof.bin"]
    assert seen["keys"] == {}
    assert seen["isolated"] == 1


def test_api_key_goes_with_every_call_to_the_server(client, server, monkeypatch):
    routes = [
        "GET /plugins/proof/policies",
        "POST /v1/completions",
        "GET " + JOB,
        "GET " + JOB + "/y.bin",
        "GET " + JOB + "/proof.bin",
        "DELETE " + JOB,
    ]
    for number, (key, header) in enumerate(((None, None), ("secret", "Bearer secret"))):
        server.calls.clear()
        if key:
            monkeypatch.setenv("VLLM_API_KEY", key)
        assert client.run(out_dir=client.dir / str(number))[0] == 0
        assert server.routes() == routes
        assert [headers.get("authorization") for _, _, headers, _ in server.calls] == [header] * 6


def test_api_key_over_plain_http_goes_through_no_proxy(client, server, monkeypatch):
    proxy = FakeServer()
    try:
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name)
        for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(name, proxy.url)
        monkeypatch.setenv("VLLM_API_KEY", "secret")
        assert client.run()[0] == 0
        assert proxy.calls == [] and len(server.calls) == 6
    finally:
        proxy.httpd.shutdown()
        proxy.httpd.server_close()


def test_check_script_gets_the_api_key_only_for_fetches_from_the_server(client, monkeypatch):
    monkeypatch.setenv("CHECK_PROOF_FETCH_KEY", "another")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    assert client.run()[0] == 0
    assert client.seen_by_check()["keys"] == {"CHECK_PROOF_FETCH_KEY": "secret"}
    assert client.run("--resume", after=["--fetch-from", "http://other.example"])[0] == 0
    assert client.seen_by_check()["keys"] == {}


def test_api_key_is_not_a_command_line_option(client, server):
    """Other users can see a command line with ps."""
    status, _, err = client.run("--api-key", "secret")
    assert status == 2 and err.startswith("usage: ") and server.calls == []


@pytest.mark.parametrize(
    "url, sent",
    [
        ("https://gpu-box:8000", True),
        ("http://127.0.0.1:8000", True),
        ("http://[::1]:8000", True),
        ("http://gpu-box:8000", False),
        ("HTTP://GPU-BOX", False),
        ("http://127.0.0.1.example:8000", False),
        ("http://127.0.0.1.evil.example", False),
        # A user name, an escape or a second colon: URL parsers disagree on where the host ends.
        ("http://localhost:x.evil.example:80", False),
        ("http://evil.example%2F@127.0.0.1", False),
        ("http://u@localhost", False),
        ("http://localhost:80:80", False),
        # localhost can resolve to 127.0.0.1 or ::1; another local user may listen on the free one.
        ("http://localhost:8000", False),
        ("http://localhost", False),
    ],
)
def test_api_key_goes_only_over_https_or_to_this_machine(client, capfd, monkeypatch, url, sent):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    argv = client.argv(server=url)
    if sent:
        assert ask_and_check.parse_args(argv)[0].api_key == "secret"
        return
    with pytest.raises(SystemExit) as usage_error:
        ask_and_check.parse_args(argv)
    err = capfd.readouterr().err
    assert usage_error.value.code == 2 and len(err.splitlines()) == 2
    assert "must be 127.0.0.1 or [::1]" in err and "secret" not in err


def test_plain_http_to_another_machine_is_allowed_without_an_api_key(client, monkeypatch):
    argv = client.argv(server="http://gpu-box")
    assert ask_and_check.parse_args(argv)[0].api_key is None
    monkeypatch.setenv("VLLM_API_KEY", "")
    assert ask_and_check.parse_args(argv)[0].api_key == ""


@pytest.mark.parametrize(
    "key", ["se\ncret", "secret\r", "secret\n", "se\x1bcret", "se\x7fcret", "s\u00e9cret"]
)
def test_api_key_that_is_not_printable_ascii_is_a_usage_error_and_is_not_shown(
    client, server, monkeypatch, key
):
    monkeypatch.setenv("VLLM_API_KEY", key)
    status, _, err = client.run()
    assert status == 2 and len(err.splitlines()) == 2 and server.calls == []
    assert "a character that is not printable ASCII" in err and "cret" not in err


def test_check_script_gets_the_right_request_even_if_request_bin_is_swapped(client, monkeypatch):
    """A user who can rename --out-dir or a directory above it can swap request.bin."""
    popen = subprocess.Popen

    def swap_first(*args, **options):
        (client.out_dir / "request.bin").unlink()
        (client.out_dir / "request.bin").write_bytes(b"another request")
        return popen(*args, **options)

    monkeypatch.setattr(subprocess, "Popen", swap_first)
    assert client.run()[0] == 0
    request = bytes.fromhex(client.seen_by_check()["request"])
    assert request == spool.whole_reply(PROMPT, REPLY, 16, fill=0)


def test_request_body_and_headers(client, server, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    client.run("--ask-only")
    method, path, headers, body = server.calls[-1]
    assert (method, path) == ("POST", "/v1/completions")
    assert (headers["x-proof-policy"], headers["authorization"]) == ("exact", "Bearer secret")
    assert json.loads(body) == {
        "model": "m",
        "prompt": PROMPT,
        "max_tokens": 5,
        "temperature": 0,
        "return_token_ids": True,
    }
    client.run("--ask-only", "--chat", out_dir=client.dir / "chat", **TEXT)
    assert server.calls[-1][1] == "/v1/chat/completions"
    assert json.loads(server.calls[-1][3]) == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 5,
        "temperature": 0,
        "return_token_ids": True,
    }


@pytest.mark.parametrize("flags", [(), ("--chat",)])
def test_text_prompt_is_not_a_full_verification(client, server, flags):
    for number, lines in enumerate(((None, None), (WITHIN_LIMITS, WITHIN_LIMITS_RESULT))):
        client.check_exits_with(0, *lines)
        status, out, _ = client.run(*flags, out_dir=client.dir / str(number), **TEXT)
        assert status == 24
        shown = (lines[1] or RESULTS[0]).replace("START:END", "2:5")
        assert out.splitlines()[-4] == shown
        # The warning comes first: check_proof.py's output may include "verified".
        assert out.splitlines()[-6:-4] == [ask_and_check.SERVER_CHOSE_PROMPT_WARNING, "some output"]
        assert out.splitlines()[-2] == "result: outcome=server-chose-prompt"
        assert out.splitlines()[-1].startswith(
            "ask_and_check.py: not a full verification (exit code 24)"
        )
        assert server.routes()[-1] == "DELETE " + JOB


@pytest.mark.parametrize(
    "rests_on, strict",
    [
        (" commitment_assumed_rounds=49 record=1 work=20", []),
        (" commitment_assumed_rounds=49 record=1 work=0", ["--no-proof-of-work"]),
    ],
)
def test_within_limits_is_a_pass(client, server, rests_on, strict):
    (client.dir / "policy.txt").write_text(POLICY.replace("theta 256", "theta 300"))  # clipped gap
    client.check_exits_with(0, WITHIN_LIMITS, WITHIN_LIMITS_RESULT + rests_on)
    after = ["--policy", "clipped-gap", "--show-all", "--C", "5", "--s=15", *strict]
    status, out, err = client.run(after=after)
    shown = [(WITHIN_LIMITS_RESULT + rests_on).replace("START:END", "2:5"), WITHIN_LIMITS]
    assert (status, err, out.splitlines()[-2:]) == (0, "", shown)
    assert client.seen_by_check()["argv"][2 : 2 + len(after)] == after
    assert server.routes()[-1] == "DELETE " + JOB


def test_reply_of_exactly_max_tokens_needs_no_end_token(client, server):
    server.reply = [4, 5, 6, 8, 9]
    assert client.run()[0] == 0
    assert client.seen_by_check()["argv"][-4:] == ["--reply", "3:5", "--end-token", "none"]


def test_reply_longer_than_max_tokens_passes_if_it_ends_with_an_end_token(client, server):
    server.reply = [4, 5, 6, 8, 9, 10, END]
    assert client.run()[0] == 0


def test_policy_options_are_passed_on(client):
    after = ["--policy", "clipped-gap", "--C=5", "--fetch-from", "/some/dir"]
    assert client.run(after=after)[0] == 0
    assert client.seen_by_check()["argv"][2:7] == after


def test_verifier_is_passed_on(client):
    assert client.run(verifier="/some/cp_verify")[0] == 0
    argv = client.seen_by_check()["argv"]
    assert argv[argv.index("--verifier") + 1] == "/some/cp_verify"


# ---- No pass, or no proof ----


@pytest.mark.parametrize(
    "check_status, status",
    [(1, 1), (2, 2), (14, 14), (24, 2), (5, 2), (-9, 2)],
)
def test_exit_code_of_the_check_script(client, server, check_status, status):
    """24 and 5 are not exit codes of check_proof.py; -9: a signal kills the check script."""
    client.check_exits_with(check_status)
    got, _, err = client.run(**TEXT)
    assert got == status
    assert "DELETE " + JOB not in server.routes() and "was not asked to delete its copy" in err
    assert ("was killed by signal 9" in err) == (check_status == -9)


@pytest.mark.parametrize(
    "code, result",
    [
        *((code, "") for code in RESULTS if code != 2),  # no result line
        *((code, RESULTS[other]) for code in RESULTS for other in RESULTS if other != code),
        (1, WITHIN_LIMITS_RESULT),
        (0, PASS % ("no-proof", "exact-match")),
        (0, PASS % ("other", "exact-match")),
        (0, PASS % ("verified", "")),
        (0, RESULTS[0].replace(" claim=exact-match", "")),
        (0, PASS % ("verified", "next-token")),
        (0, PASS % ("verified", "policy")),
        (0, PASS % ("within-limits", "exact-match")),
        (0, RESULTS[0].replace(COMMITMENT_ID, "cd" * 32)),
        (0, RESULTS[0].replace(" commitment_id=" + COMMITMENT_ID, "")),
        (0, RESULTS[0].replace("START:END", "2:4")),
        (0, RESULTS[0].replace(" range=START:END", "")),
        (0, "result: x"),
        (0, "result: outcome=verified"),
        (0, "result: outcome=verified claim=exact-match"),
        (0, RESULTS[0].replace("outcome=verified ", "")),
        (0, " " + RESULTS[0]),
        (0, RESULTS[0] + "\nanother line"),  # not directly before the last line
        (0, RESULTS[0] + " outcome=verified"),  # a repeated field
        (0, RESULTS[0] + " range=START:END"),
        (0, RESULTS[0].replace("result:", "result: outcome=rejected")),
        # outcome= is not the first field
        (0, RESULTS[0].replace("outcome=verified ", "") + " outcome=verified"),
        (1, "result: outcome=other"),
    ],
)
def test_exit_code_and_result_line_must_agree(client, server, code, result):
    client.check_exits_with(code, result=result)
    status, out, err = client.run()
    assert status == 2 and "disagree" in err and "DELETE " + JOB not in server.routes()
    assert out.splitlines()[-1] == "result: outcome=error"


def test_pass_line_beside_an_unknown_exit_code_is_an_error(client, server):
    client.check_exits_with(10, result=RESULTS[0])
    status, out, err = client.run()
    assert status == 2 and "unexpected code 10" in err and "DELETE " + JOB not in server.routes()
    assert out.splitlines()[-1] == "result: outcome=error"


def test_usage_error_of_the_check_script_is_an_error(client):
    client.check_exits_with(2, result="")
    status, out, err = client.run()
    assert status == 2 and outcome(out) == "error" and "disagree" not in err


@pytest.mark.parametrize(
    "code, options",
    [
        (2, {}),
        (2, {"result": "", "last_line": RESULTS[0]}),  # a pass's line is the last line
        (2, {"result": RESULTS[0] + "\none line"}),  # or the third line from the end
        (2, {"end": ""}),  # the output does not end with a newline
        (0, {"result": "", "last_line": RESULTS[0], "end": ""}),
        (1, {"last_line": RESULTS[0]}),
    ],
)
def test_exit_code_2_always_ends_with_the_clients_result_line(client, code, options):
    client.check_exits_with(code, **options)
    status, out, _ = client.run()
    assert status == 2 and out.splitlines()[-1] == "result: outcome=error"


def test_no_proof_of_the_check_script_is_passed_through(client):
    client.check_exits_with(1, result="result: outcome=no-proof")
    status, out, _ = client.run()
    assert status == 1 and outcome(out) == "no-proof"


def test_help_lists_every_exit_code():
    rows = re.findall("^ +([0-9]+)  ", ask_and_check.EXIT_CODES, re.M)
    assert sorted(map(int, rows)) == sorted({*ask_and_check.OUTCOMES, 24, 75})


def test_client_and_check_script_agree_on_exit_codes_and_outcomes():
    theirs = {code: [word] for code, word in check_proof.OUTCOMES.items()}
    theirs[check_proof.EXIT_VERIFIED].append(check_proof.WITHIN_LIMITS_OUTCOME)
    theirs[check_proof.EXIT_REJECTED].append(check_proof.NO_PROOF_OUTCOME)
    assert {code: list(words) for code, words in ask_and_check.OUTCOMES.items()} == theirs
    assert list(ask_and_check.CLAIMS) == theirs[check_proof.EXIT_VERIFIED]
    assert ask_and_check.LARGEST_FILE == check_proof.MAX_FETCHED_BYTES
    assert ask_and_check.others_can_change is check_proof.others_can_change


@pytest.mark.parametrize("policy_words, outcome", [([], "verified"), (["x"], "within-limits")])
def test_client_and_check_script_agree_on_claims(monkeypatch, policy_words, outcome):
    monkeypatch.setattr(check_proof, "named_policy", lambda *_: (policy_words, None))
    request = types.SimpleNamespace(checked=1)
    assert check_proof.claim(None, {}, request) == ask_and_check.CLAIMS[outcome]


# The client and the check script must also agree on which options take a value (nargs 0: none).
def test_check_script_has_the_options_that_may_follow_the_dashes():
    actions = check_proof.build_parser()._actions
    takes_value = {name: action.nargs != 0 for action in actions for name in action.option_strings}
    assert set(ask_and_check.WITHOUT_VALUE) < set(ask_and_check.PASSED_ON)
    for name in ask_and_check.PASSED_ON:
        assert takes_value[name] == (name not in ask_and_check.WITHOUT_VALUE)


def test_last_line_of_the_check_script_is_ignored(client):
    client.check_exits_with(0, "any words")
    assert client.run()[0] == 0
    client.check_exits_with(1, "check_proof: verified: fake")
    assert client.run("--resume")[0] == 1


def test_sigchld_is_reset_to_default_before_the_check_script_starts(client, monkeypatch):
    """Python's subprocess module reports exit code 0 for every child while SIGCHLD is ignored."""
    events, set_signal, popen = [], signal.signal, subprocess.Popen

    def record_signal(number, handler):
        events.append((number, handler))
        return set_signal(number, handler)

    def record_child(*args, **options):
        events.append("child")
        return popen(*args, **options)

    monkeypatch.setattr(signal, "signal", record_signal)
    monkeypatch.setattr(subprocess, "Popen", record_child)
    assert client.run()[0] == 0
    assert events == [(signal.SIGCHLD, signal.SIG_DFL), "child"]


@pytest.mark.parametrize("state", ["no_proof", "not_provable", "expired"])
def test_no_proof(client, server, state):
    server.job = {"state": state, "detail": "the prover\nissued none"}
    status, out, err = client.run()
    assert (status, err) == (
        1,
        "error: there is no proof; the server's reason: the prover\\nissued none\n",
    )
    assert server.routes()[-1] == "DELETE " + JOB and not client.checked()
    assert out.splitlines()[-1] == "result: outcome=no-proof"
    server.calls.clear()
    assert client.run("--resume")[::2] == (1, err) and server.calls == []  # read from no_proof.txt


def test_proof_not_ready_then_resume(client, server):
    server.job = {"state": "waiting"}
    status, out, err = client.run()
    assert status == 75 and "--resume" in err and outcome(out) == "try-again"
    server.instead["GET", JOB] = (503, {}, b"busy")
    assert client.run("--resume")[0] == 75
    del server.instead["GET", JOB]
    server.job = {"state": "proved"}
    server.calls.clear()
    assert client.run("--resume")[0] == 0
    assert not any(route.startswith("POST") for route in server.routes())


def test_ask_only_then_resume(client, server):
    for number, options in enumerate(({}, TEXT)):
        status, out, _ = client.run("--ask-only", out_dir=client.dir / str(number), **options)
        assert status == 75 and not client.checked()
        assert out.splitlines()[-2] == "result: outcome=stored"
        assert out.splitlines()[-1].startswith("ask_and_check.py: reply stored; not checked yet (")
    assert client.run("--ask-only")[0] == 75
    assert os.listdir(client.out_dir) == ["served.json"]
    stored = json.loads((client.out_dir / "served.json").read_text())
    assert (
        stored["proof_id"],
        stored["prompt_token_ids"],
        stored["reply_token_ids"],
        stored["text"],
    ) == (PROOF_ID, PROMPT, REPLY, "some text")
    assert client.run("--resume")[0] == 0
    flags = set(client.seen_by_check()["argv"]) & set(ask_and_check.WITHOUT_VALUE)
    assert not flags  # a resumed run adds no option of its own either
    server.calls.clear()
    assert client.run("--resume")[0] == 0
    assert server.routes() == ["DELETE " + JOB]


def test_resume_after_a_rejection_checks_again_without_a_download(client, server):
    client.check_exits_with(1)
    assert client.run()[0] == 1
    server.calls.clear()
    (client.dir / "seen.json").unlink()
    assert client.run("--resume")[0] == 1 and client.checked() and server.calls == []


# ---- A server that misbehaves ----


@pytest.mark.parametrize("theirs", ["cd" * 32, "", None, 5])
def test_rejects_wrong_commitment_id(client, server, theirs):
    server.commitment_id = theirs
    status, _, err = client.run()
    assert status == 1 and "commitment ID" in err
    assert err.endswith("with where the commitment is published)\n") == (theirs == "cd" * 32)
    assert "cd" * 32 not in err
    assert server.routes() == ["GET /plugins/proof/policies"]
    assert os.listdir(client.out_dir) == []


def test_closed_policy_takes_no_new_request_but_a_run_can_be_resumed(client, server):
    entry = {"commitment_id": COMMITMENT_ID, "closed": True}
    closed = (200, {}, json.dumps({"policies": {"exact": entry}}).encode())
    server.instead["GET", "/plugins/proof/policies"] = closed
    status, _, err = client.run()
    assert (status, err) == (2, "error: the policy exact is closed: it takes no new requests\n")
    assert server.routes() == ["GET /plugins/proof/policies"]
    assert os.listdir(client.out_dir) == []
    server.instead.clear()
    assert client.run("--ask-only")[0] == 75
    server.instead["GET", "/plugins/proof/policies"] = closed
    assert client.run("--resume")[0] == 0


def test_unknown_policy_is_an_error(client, server):
    status, _, err = client.run(policy_name="other")
    assert (status, err) == (2, "error: the server has no policy named other (it has: exact)\n")


def test_rejects_prompt_ids_that_differ_from_those_sent(client, server):
    server.prompt = [1, 2, 3, 4]
    assert client.run()[0] == 1
    assert os.listdir(client.out_dir) == [] and not client.checked()


@pytest.mark.parametrize("reply", [[4, 7, 5, 7], [7, 4, 5, 6, 8], [4, 5, 6], [4, 5, 6, 8, 9, 10]])
def test_rejects_reply_that_breaks_the_end_token_rule(client, server, reply):
    """An end token before the last token; or no end token, and not exactly --max-tokens tokens."""
    server.reply = reply
    for number, (flags, options) in enumerate((((), {}), ((), TEXT), (("--ask-only",), {}))):
        status, _, err = client.run(*flags, out_dir=client.dir / ("out-%d" % number), **options)
        assert status == 1 and "end token" in err and not client.checked()
    assert "GET " + JOB not in server.routes()


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": []},
        {"reply": []},
        {"reply": [1, -1]},
        {"reply": [1 << 32]},
        {"reply": [True]},
        {"reply": "47"},
        {"text": 5},
    ],
)
def test_unusable_token_ids_or_text_are_an_error(client, server, change):
    vars(server).update(change)
    assert client.run(**TEXT)[0] == 2
    assert os.listdir(client.out_dir) == []


@pytest.mark.parametrize("proof_id", [None, "12" * 31, "AB" * 32, "../" * 21 + "x"])
def test_malformed_proof_id_is_an_error(client, server, proof_id):
    body = server.response("POST", "/v1/completions")[2]
    server.instead["POST", "/v1/completions"] = (200, {"x-proof-id": proof_id}, body)
    status, _, err = client.run()
    assert status == 2 and "x-proof-id" in err


@pytest.mark.parametrize(
    "redirect, method, path, status",
    [
        (307, "GET", "/plugins/proof/policies", 2),
        (301, "POST", "/v1/completions", 2),
        (302, "POST", "/v1/completions", 2),
        (303, "POST", "/v1/completions", 2),
        (307, "POST", "/v1/completions", 2),
        (308, "POST", "/v1/completions", 2),
        (307, "GET", JOB, 2),
        (307, "GET", JOB + "/proof.bin", 2),
        # The proof is already checked; a failed DELETE changes nothing.
        (307, "DELETE", JOB, 0),
    ],
)
def test_follows_no_redirect(client, server, monkeypatch, redirect, method, path, status):
    other = FakeServer()
    try:
        server.instead[method, path] = (redirect, {"location": other.url + path}, b"")
        monkeypatch.setenv("VLLM_API_KEY", "secret")
        got, _, err = client.run()
        assert got == status and ("returned a redirect (status" in err) == (status == 2)
        assert other.calls == []
    finally:
        other.httpd.shutdown()
        other.httpd.server_close()


def test_stores_no_more_than_the_announced_length(client, server):
    server.instead["GET", JOB + "/proof.bin"] = (200, {"content-length": "5"}, b"p" * 40)
    assert client.run()[0] == 0
    assert (client.out_dir / "proof" / "proof.bin").read_bytes() == b"p" * 5


def test_server_strings_are_escaped_and_truncated(client, server):
    forged = "\ncheck_proof: verified: all\x1b[2J\u202e" + " " * 50 + "x" * 5000
    server.text = forged
    server.job = {"state": "no_proof", "detail": forged}
    _, out, err = client.run()
    for stream in (out, err):
        assert "\x1b" not in stream and "\u202e" not in stream and "  " not in stream
        assert not any(line.startswith("check_proof:") for line in stream.splitlines())
    assert len(err) < 500 and len(out) < 4500


def test_tls_to_a_plain_port_is_an_error(client, server):
    status, _, err = client.run(server=server.url.replace("http://", "https://"))
    assert status == 2 and err.startswith("error: cannot reach https://")


def test_status_4xx_is_an_error_and_5xx_means_try_again(client, server):
    server.instead["POST", "/v1/completions"] = (401, {}, b'{"error": "Unauthorized"}')
    status, _, err = client.run()
    assert status == 2 and err.endswith("(check VLLM_API_KEY)\n")
    server.instead.clear()
    server.instead["GET", JOB] = (401, {}, b'{"error": "Unauthorized"}')
    status, _, err = client.run(out_dir=client.dir / "job")
    assert status == 2 and err.endswith("(check VLLM_API_KEY)\n")
    server.instead.clear()
    server.instead["GET", "/plugins/proof/policies"] = (502, {}, b"bad gateway")
    status, _, err = client.run()
    assert status == 75 and err.endswith(": bad gateway (run the same command again)\n")
    assert os.listdir(client.out_dir) == []


def test_rejects_prompt_and_reply_that_do_not_fit_the_request_file(client, server):
    server.reply = list(range(10, 23)) + [END]  # 17 tokens with the prompt; the request file has 16
    assert client.run()[0] == 1 and not client.checked()


def test_unexpected_exception_is_an_error_without_a_traceback(client, monkeypatch):
    def broken(args):
        raise RuntimeError("some\nwords")

    monkeypatch.setattr(ask_and_check, "ask", broken)
    status, _, err = client.run()
    assert status == 2 and err.endswith("RuntimeError: some\\nwords\n")


@pytest.mark.parametrize(
    "text",
    [
        None,
        "(",
        "raise ValueError('a\\nb')",
        "\0",
        "raise SystemExit(0)",
        "raise KeyboardInterrupt",
    ],
)
def test_failed_import_is_an_error(tmp_path, text):
    """check_proof.py is missing from the client's directory, or raises an exception as it loads."""
    shutil.copy(ask_and_check.__file__, tmp_path)
    if text is not None:
        (tmp_path / "check_proof.py").write_text(text)
    command = [sys.executable, "-B", tmp_path / "ask_and_check.py", "--help"]
    got = subprocess.run(command, capture_output=True, text=True)
    assert got.returncode == 2 and got.stdout == "result: outcome=error\n"
    if text == "raise KeyboardInterrupt":
        assert got.stderr == "error: interrupted (run the same command again)\n"
    else:
        assert got.stderr.startswith("error: cannot import check_proof.py and spool.py (")
        assert got.stderr.endswith("): they must be in this script's directory, unchanged\n")
        assert len(got.stderr.splitlines()) == 1


def test_error_lines_hide_the_proof_id(client, server):
    body = json.dumps({"error": {"message": "job %s is broken" % PROOF_ID}}).encode()
    server.instead["GET", JOB] = (500, {}, body)
    status, out, err = client.run()
    assert status == 75 and PROOF_ID not in out + err
    assert (
        err == "error: the server returned status 500 for GET /plugins/proof/jobs/<proof ID>: job"
        " <proof ID> is broken (run the same command again with --resume)\n"
    )
    server.instead["GET", JOB] = (200, {}, json.dumps({"state": "x", "id": PROOF_ID}).encode())
    status, out, err = client.run("--resume")
    assert status == 2 and "is unusable" in err and PROOF_ID not in out + err
    del server.instead["GET", JOB]
    server.job = {"state": "no_proof", "detail": "job %s failed" % PROOF_ID}
    status, out, err = client.run("--resume")
    assert status == 1 and PROOF_ID not in out + err
    assert PROOF_ID not in (client.out_dir / "no_proof.txt").read_text()


def test_busy_server_during_a_download_means_try_again(client, server):
    server.instead["GET", JOB + "/proof.bin"] = (503, {}, b"busy")
    assert client.run()[0] == 75
    server.instead.clear()
    assert client.run("--resume")[0] == 0


def test_downloads_only_the_two_known_files(client, server):
    client.run()
    assert [route for route in server.routes() if route.startswith("GET " + JOB + "/")] == [
        "GET %s/%s" % (JOB, name) for name in ask_and_check.DOWNLOADS
    ]


@pytest.mark.parametrize(
    "headers, body, status",
    [
        ({"content-length": "40"}, b"p" * 10, 75),  # it ended early, so a retry can help
        ({"content-length": None}, b"p" * 40, 2),
        ({"content-length": "x"}, b"p", 2),
        ({"content-length": str(256 << 20)}, b"p", 75),  # the largest length that is taken
        ({"content-length": str((256 << 20) + 1)}, b"p", 2),
    ],
)
def test_download_without_its_full_length_is_not_used(client, server, headers, body, status):
    server.instead["GET", JOB + "/proof.bin"] = (200, headers, body)
    got, _, err = client.run()
    assert got == status and ("or one over 256 MiB, for proof.bin" in err) == (status == 2)
    assert (
        not (client.out_dir / "proof" / "proof.bin").exists()
        and not client.checked()
        and "DELETE " + JOB not in server.routes()
    )
    del server.instead["GET", JOB + "/proof.bin"]
    assert client.run("--resume")[0] == 0


# ---- The output directory ----


def test_failed_first_run_leaves_the_directory_empty(client, server):
    server.instead["POST", "/v1/completions"] = (500, {}, b"down")
    status, _, err = client.run()
    assert status == 75 and err.endswith(": down (run the same command again)\n")
    assert os.listdir(client.out_dir) == []
    server.instead.clear()
    assert client.run()[0] == 0


def create_user_file(path):
    """Create a file at 'path', as the user might. Return a function: is the file unchanged?"""
    path.parent.mkdir(mode=0o700, exist_ok=True)
    path.write_bytes(b"the user's own")
    os.utime(path, (1, 1))
    before = path.stat()

    def untouched():
        now = path.lstat()
        same = (now.st_ino, now.st_mtime, now.st_nlink) == (before.st_ino, 1, 1)
        return same and path.read_bytes() == b"the user's own"

    return untouched


# Names that the client uses in --out-dir (RANDOM: any suffix); the last two it never uses.
NAMES = [
    "served.json",
    "served.json.part",
    "served.json.part.link",
    "request.bin",
    "request.bin.part.RANDOM",
    "no_proof.txt",
    "proof/y.bin.part.RANDOM",
    "proof/proof.bin.part.RANDOM",
    "proof/y.bin.part",
    "notes.txt",
]


@pytest.mark.parametrize("name", NAMES)
def test_first_run_touches_nothing_in_a_directory_that_is_not_empty(client, server, name):
    client.out_dir.mkdir(mode=0o700)
    untouched = create_user_file(client.out_dir / name)
    status, _, err = client.run()
    assert status == 2 and "is not empty" in err and server.calls == []
    assert untouched() and os.listdir(client.out_dir) == [name.split("/")[0]]


def make_the_run_end_with(ending, client, server):
    """Set up how a resumed run ends. Return the expected exit code."""
    if ending == "server fault":
        server.instead["GET", JOB] = (500, {}, b"down")
    elif ending == "no proof":
        server.job = {"state": "no_proof", "detail": "none"}
    elif ending == "download cut":
        server.instead["GET", JOB + "/proof.bin"] = (200, {"content-length": "40"}, b"p" * 10)
    elif ending == "download refused":
        server.instead["GET", JOB + "/y.bin"] = (403, {}, b"no")
    else:
        client.check_exits_with(ending)
    codes = {"server fault": 75, "no proof": 1, "download cut": 75, "download refused": 2}
    return codes.get(ending, ending if ending in (0, 1, 14) else 2)


ENDINGS = ["server fault", "no proof", "download cut", "download refused", 0, 1, 2, 14, 5, -9]


@pytest.mark.parametrize("ending", ENDINGS)
def test_resumed_run_keeps_the_users_files(client, server, ending):
    assert client.run("--ask-only")[0] == 75
    names = [name for name in NAMES if name not in ("served.json", "no_proof.txt")]
    user_files = {name: create_user_file(client.out_dir / name) for name in names}
    expected = make_the_run_end_with(ending, client, server)
    status, _, err = client.run("--resume")
    assert status == expected
    assert [name for name, untouched in user_files.items() if not untouched()] == []
    # The run's own temporary files are gone.
    left = {str(path.relative_to(client.out_dir)) for path in client.out_dir.rglob("*")}
    final = {"served.json", "no_proof.txt", "proof", "proof/y.bin", "proof/proof.bin"}
    assert left - set(names) <= final
    assert ("is not the request that was checked" in err) == client.checked()


@pytest.mark.parametrize("name", ["request.bin", "proof/y.bin", "proof/proof.bin"])
def test_existing_file_at_a_temporary_name_is_an_error_and_is_kept(client, monkeypatch, name):
    assert client.run("--ask-only")[0] == 75
    monkeypatch.setattr(os, "urandom", lambda size: bytes(size))  # the random suffix is all zeros
    untouched = create_user_file(client.out_dir / (name + ".part." + "00" * 8))
    status, _, err = client.run("--resume")
    assert status == 2 and err.endswith(": File exists\n") and untouched()


def test_messages_after_a_killed_first_run_say_what_to_do(client, server):
    client.out_dir.mkdir(mode=0o700)
    # as a killed run would leave it
    untouched = create_user_file(client.out_dir / "served.json.part")
    assert client.run()[2].endswith("or choose a new directory)\n")
    assert client.run("--resume")[2].endswith("(choose a new --out-dir, without --resume)\n")
    assert untouched() and server.calls == []


def test_existing_no_proof_file_is_read_and_kept(client, server):
    assert client.run("--ask-only")[0] == 75
    untouched = create_user_file(client.out_dir / "no_proof.txt")
    status, _, err = client.run("--resume")
    assert status == 1 and "the user's own" in err and untouched()


@pytest.mark.parametrize("name", ["served.json", "no_proof.txt"])
@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_resume_reads_only_regular_files(client, server, name, kind):
    """A plain open() would follow the symlink, and block on the FIFO."""
    assert client.run("--ask-only")[0] == 75
    server.calls.clear()
    path, moved = client.out_dir / name, client.dir / "moved"
    if path.exists():
        path.rename(moved)
    else:
        moved.write_text("a reason")
    {
        "symlink": lambda: path.symlink_to(moved),
        "fifo": lambda: os.mkfifo(path),
        "directory": path.mkdir,
    }[kind]()
    status, _, err = client.run("--resume")
    assert (status, err) == (2, "error: %s: not a regular file (choose a new --out-dir)\n" % path)
    assert server.calls == [] and not client.checked()


@pytest.mark.parametrize("kind", ["symlink", "file", "fifo", "shared"])
def test_proof_must_be_a_private_directory(client, server, kind):
    assert client.run("--ask-only")[0] == 75
    server.calls.clear()
    path, elsewhere = client.out_dir / "proof", client.dir / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    if kind == "symlink":
        path.symlink_to(elsewhere)
    elif kind == "file":
        path.write_text("mine")
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
        path.chmod(0o770)
    status, _, err = client.run("--resume")
    assert status == 2 and err.startswith("error: %s cannot be used: " % path)
    assert err.endswith("(choose a new --out-dir)\n")
    assert server.calls == [] and os.listdir(elsewhere) == [] and not client.checked()


def test_request_bin_is_written_once(client):
    assert client.run()[0] == 0
    before = (client.out_dir / "request.bin").stat()
    status, _, err = client.run("--resume")
    assert (status, err) == (0, "")
    now = (client.out_dir / "request.bin").stat()
    assert (now.st_ino, now.st_mtime_ns, now.st_nlink) == (before.st_ino, before.st_mtime_ns, 1)


@pytest.mark.parametrize("record", ["closed", "other bytes", "a link"])
def test_note_on_an_earlier_record_says_whether_it_was_compared(client, monkeypatch, record):
    assert client.run()[0] == 0
    words = "was not compared with the request that was checked in this run, because this account"
    words += " cannot read it as a regular file"
    if record == "closed":
        fail_to_open(monkeypatch, client.out_dir / "request.bin", errno.EACCES)
    else:
        (client.out_dir / "request.bin").rename(client.dir / "the record")
    if record == "other bytes":
        (client.out_dir / "request.bin").write_bytes(b"another")
        words = "is not the request that was checked in this run"
    elif record == "a link":  # to the very request
        (client.out_dir / "request.bin").symlink_to(client.dir / "the record")
    note = "note: %s %s; it was left as it is\n" % (client.out_dir / "request.bin", words)
    assert client.run("--resume")[::2] == (0, note)


@pytest.mark.parametrize("taken", ["served.json.part", "served.json.part.link"])
def test_concurrent_second_run_deletes_nothing(client, server, monkeypatch, taken):
    """The other run's file appears after this run has found the directory empty."""
    client.out_dir.mkdir(mode=0o700)
    untouched = create_user_file(client.out_dir / taken)
    with monkeypatch.context() as patch:
        patch.setattr(os, "listdir", lambda path: [])
        status, _, err = client.run()
    assert status == 2 and "another run is using the directory" in err and server.calls == []
    assert untouched() and os.listdir(client.out_dir) == [taken]


def test_file_system_without_hard_links_is_detected_before_the_prompt_is_sent(
    client, server, monkeypatch
):
    def no_link(*names, **where):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    status, _, err = client.run()
    assert status == 2 and "without hard links" in err and server.calls == []
    assert os.listdir(client.out_dir) == []


def test_local_write_failure_is_an_environment_error(client, monkeypatch):
    link = os.link

    def full(source, target, **where):
        if target == "served.json":
            raise OSError(errno.ENOSPC, "No space left on device", source, None, target)
        link(source, target, **where)

    monkeypatch.setattr(os, "link", full)
    status, _, err = client.run()
    served_json = client.out_dir / "served.json"
    assert (status, err) == (2, "error: %s: No space left on device\n" % served_json)
    assert os.listdir(client.out_dir) == []


@pytest.mark.parametrize(
    "name, advice",
    [("out", "run the command again"), ("proof", "run the same command again with --resume")],
)
def test_directory_that_disappears_before_it_is_opened_is_an_error(
    client, monkeypatch, name, advice
):
    real = ask_and_check.Directory.make_directory

    def made_and_removed(self, new):
        real(self, new)
        if new == name:
            os.rmdir(new, dir_fd=self.fd)

    monkeypatch.setattr(ask_and_check.Directory, "make_directory", made_and_removed)
    status, _, err = client.run()
    path = client.out_dir / "proof" if name == "proof" else client.out_dir
    start = "" if name == "proof" else "--out-dir "
    assert status == 2
    assert err == "error: %s%s disappeared before it could be opened (%s)\n" % (start, path, advice)


# ---- An interrupt (Ctrl-C) ----


def interrupt_after(monkeypatch, function, name, then=lambda: None):
    """Patch os.link or os.unlink: the first call on 'name' does its work, calls then() and raises
    KeyboardInterrupt. For os.link, 'name' is the new name."""
    real, done = getattr(os, function), []

    def interrupted(*names, **where):
        real(*names, **where)
        if names[-1] == name and not done:
            done.append(name)
            then()
            raise KeyboardInterrupt

    monkeypatch.setattr(os, function, interrupted)


def test_interrupt_before_the_reply_is_stored_leaves_the_directory_empty(client, monkeypatch):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(ask_and_check, "ask", interrupted)
    status, out, err = client.run()
    assert (status, err) == (2, "error: interrupted (run the same command again)\n")
    assert out.splitlines()[-1] == "result: outcome=error" and os.listdir(client.out_dir) == []


@pytest.mark.parametrize(
    "function, name, left",
    [("link", "served.json", ["served.json.part"]), ("unlink", "served.json.part", [])],
)
def test_interrupt_after_the_reply_is_stored_keeps_it(client, monkeypatch, function, name, left):
    with monkeypatch.context() as patch:
        interrupt_after(patch, function, name)
        status, _, err = client.run()
    assert (status, err) == (2, "error: interrupted (run the same command again with --resume)\n")
    assert sorted(os.listdir(client.out_dir)) == ["served.json", *left]
    assert client.run("--resume")[0] == 0


@pytest.mark.parametrize("name", ["served.json.part.link", "served.json.part"])
def test_interrupt_keeps_a_file_that_another_process_created_under_a_temporary_name(
    client, monkeypatch, name
):
    path, untouched = client.out_dir / name, []
    interrupt_after(monkeypatch, "unlink", name, lambda: untouched.append(create_user_file(path)))
    assert client.run()[0] == 2 and untouched[0]()


def test_interrupt_while_the_server_deletes_its_copy_changes_nothing(client, monkeypatch):
    new_opener = ask_and_check.new_opener

    class Interrupted:
        def __init__(self, args):
            self.opener = new_opener(args)

        def open(self, request, timeout):
            if request.get_method() == "DELETE":
                raise KeyboardInterrupt
            return self.opener.open(request, timeout=timeout)

    monkeypatch.setattr(ask_and_check, "new_opener", Interrupted)
    assert client.run()[::2] == (0, "")


RUN_CLIENT = """import sys
import ask_and_check
ask_and_check.CHECK_SCRIPT = sys.argv.pop(1)
sys.exit(ask_and_check.main())
"""
# A check script that, like check_proof.py, ignores a second SIGINT and takes a moment to clean up.
# Its stderr is redirected, so that the test's stderr pipe closes when the client exits. It ends
# once the client, its parent, is gone, or after about 60 seconds.
SLOW_CHECK = """import os, signal, sys, time
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
def stop(*_):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    time.sleep(1)
    open(__file__ + ".stopped", "w").close()
    sys.exit(2)
signal.signal(signal.SIGINT, stop)
client = os.getppid()
print("the check has started", flush=True)
for _ in range(600):
    if os.getppid() != client:
        break
    time.sleep(0.1)
"""


@pytest.mark.parametrize("whole_group", [True, False])
def test_interrupt_during_the_check(client, whole_group):
    """Ctrl-C in a terminal reaches the client and the check script; kill -INT only the client."""
    (client.dir / "run_client.py").write_text(RUN_CLIENT)
    (client.dir / "slow_check.py").write_text(SLOW_CHECK)
    # If SIGCHLD were ignored, the kernel would reap the client at its exit and free its process ID.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    command = [sys.executable, "-B", client.dir / "run_client.py", client.dir / "slow_check.py"]
    child = subprocess.Popen(
        [*command, *client.argv()],
        env={**os.environ, "PYTHONPATH": os.path.dirname(ask_and_check.__file__)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # a process group with only the client and the check script
    )
    try:
        for line in child.stdout:
            if line == "the check has started\n":
                break
        if whole_group:
            os.killpg(child.pid, signal.SIGINT)
        else:
            child.send_signal(signal.SIGINT)
        out, err = child.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        if child.poll() is None:  # not reaped, so its process group ID cannot be reused
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    assert child.returncode == 2 and out.splitlines()[-1] == "result: outcome=error"
    assert err == "error: interrupted (run the same command again with --resume)\n"
    # The check script got the interrupt, and the client waited for it.
    assert (client.dir / "slow_check.py.stopped").exists()


# ---- The connection, --out-dir and --resume ----


def test_failed_connection_means_try_again(client, server, monkeypatch):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        url = "http://127.0.0.1:%d" % unused.getsockname()[1]
    # The port is closed again: connection refused.
    status, _, err = client.run(server=url)
    assert status == 75 and err.startswith("error: cannot reach %s for GET " % url)
    # The connection breaks inside a JSON body.
    server.instead["GET", "/plugins/proof/policies"] = (200, {"content-length": "500"}, b'{"pol')
    status, _, err = client.run()
    assert status == 75 and "ended early (run the same command again)" in err

    class TimesOut:
        def open(self, request, timeout):
            raise TimeoutError("timed out")

    monkeypatch.setattr(ask_and_check, "new_opener", lambda args: TimesOut())
    assert client.run()[0] == 75

    class Broken:
        def read(self, size):
            raise TimeoutError("timed out")

    with pytest.raises(ask_and_check.TryAgain, match="the connection to the server failed"):
        ask_and_check.read(Broken(), 10)


def test_rejects_directory_that_is_not_empty(client, server):
    client.out_dir.mkdir(mode=0o700)
    (client.out_dir / "notes.txt").write_text("mine")
    status, _, err = client.run()
    assert status == 2 and "is not empty" in err and server.calls == []
    assert os.listdir(client.out_dir) == ["notes.txt"]
    other = client.dir / "other"
    assert client.run(out_dir=other)[0] == 0
    assert client.run(out_dir=other)[0] == 2


@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_rejects_directory_that_others_can_write_or_a_symlink(client, server, flags):
    client.run("--ask-only")
    server.calls.clear()
    advice = (
        "(choose a new --out-dir, without --resume)\n" if flags else "(choose a new directory)\n"
    )
    client.out_dir.chmod(0o770)
    status, _, err = client.run(*flags)
    assert status == 2 and "can write to it (mode 0770) " + advice in err
    client.out_dir.chmod(0o700)
    (client.dir / "link").symlink_to(client.out_dir)
    for spelling in ("link", "link/", "link/.", "./link//"):
        status, _, err = client.run(*flags, out_dir=str(client.dir) + "/" + spelling)
        assert status == 2 and server.calls == []
        assert err.endswith("it is a symlink (name the directory itself, not a link to it)\n")


@pytest.mark.parametrize(
    "flags, advice",
    [((), "choose a new directory"), (("--resume",), "choose a new --out-dir, without --resume")],
)
def test_rejects_a_file_as_out_dir(client, server, flags, advice):
    client.out_dir.write_text("mine")
    status, _, err = client.run(*flags)
    words = "error: --out-dir %s cannot be used: it is not a directory (%s)\n"
    assert (status, err) == (2, words % (client.out_dir, advice)) and server.calls == []
    assert client.out_dir.read_text() == "mine"


# The advice at --resume when --out-dir is another account's: a new directory has nothing to resume.
AS_IT_BEGAN = (
    "resume as the account that started this run: with sudo if it was started with sudo, without"
    " sudo if not"
)


@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_rejects_directory_of_another_owner(client, server, monkeypatch, flags):
    client.run("--ask-only")
    server.calls.clear()
    mine, trusted = os.geteuid(), ask_and_check.trusted_owners()
    monkeypatch.setattr(os, "geteuid", lambda: mine + 1)
    # The two looks at the directories above --out-dir depend on os.geteuid() too. Unless they are
    # replaced, they report this account's directories as another's (except under root), and
    # --out-dir itself is never judged.
    monkeypatch.setattr(ask_and_check, "others_can_change", lambda folder: None)
    monkeypatch.setattr(ask_and_check, "trusted_owners", lambda: trusted)
    status, _, err = client.run(*flags)
    assert status == 2 and server.calls == []
    assert "its owner is uid %d, and this script runs as uid %d" % (mine, mine + 1) in err
    assert err.endswith(" (%s)\n" % (AS_IT_BEGAN if flags else "choose a new directory"))


def is_meant(target, path, where):
    """Return whether a call with the first argument 'path' and the keywords 'where' names the
    pathlib.Path 'target': by its last name, with dir_fd open on the directory that holds it. An
    entry of the same name higher up is not meant."""
    if path != target.name or "dir_fd" not in where:
        return False
    above, holds = os.fstat(where["dir_fd"]), target.parent.stat()
    return (above.st_dev, above.st_ino) == (holds.st_dev, holds.st_ino)


def fail_to_open(monkeypatch, target, number):
    """Make os.open() fail with the error 'number' for 'target': see is_meant()."""
    real = os.open

    def failing(path, *rest, **where):
        if is_meant(target, path, where):
            raise OSError(number, os.strerror(number), path)
        return real(path, *rest, **where)

    monkeypatch.setattr(os, "open", failing)


def fail_to_look_at(monkeypatch, target, number):
    """Make os.lstat() fail with the error 'number' for 'target': see is_meant()."""
    real = os.lstat

    def failing(path, **where):
        if is_meant(target, path, where):
            raise OSError(number, os.strerror(number), path)
        return real(path, **where)

    monkeypatch.setattr(os, "lstat", failing)


@pytest.mark.parametrize("theirs", [True, False])
@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_out_dir_that_may_not_be_opened_is_an_error(client, server, monkeypatch, flags, theirs):
    """theirs: stands in for an --out-dir that a run under sudo made (root's, mode 0700), opened
    later without sudo. Otherwise the caller has taken its own permission away."""
    client.run("--ask-only")
    server.calls.clear()
    fail_to_open(monkeypatch, client.out_dir, errno.EACCES)
    mine, trusted = os.geteuid(), ask_and_check.trusted_owners()
    reason, advice = "Permission denied", "choose a new directory"
    if flags:
        advice = AS_IT_BEGAN if theirs else "choose a new --out-dir, without --resume"
    if theirs:  # as in test_rejects_directory_of_another_owner
        monkeypatch.setattr(os, "geteuid", lambda: mine + 1)
        monkeypatch.setattr(ask_and_check, "others_can_change", lambda folder: None)
        monkeypatch.setattr(ask_and_check, "trusted_owners", lambda: trusted)
        reason = "its owner is uid %d, and this script runs as uid %d" % (mine, mine + 1)
    status, _, err = client.run(*flags)
    words = "error: --out-dir %s cannot be used: %s (%s)\n"
    assert (status, err) == (2, words % (client.out_dir, reason, advice)) and server.calls == []


@pytest.mark.parametrize(
    "flags, advice", [((), "choose a new directory"), (("--resume",), AS_IT_BEGAN)]
)
def test_out_dir_in_a_directory_closed_to_the_caller(client, server, monkeypatch, flags, advice):
    """Stands in for a run under sudo below a closed directory of root's, such as "sudo mktemp -d"
    gives, resumed without sudo: not even the owner of --out-dir can be looked up. Without --resume
    it stands in only for the directory that holds --out-dir being closed to the caller after the
    client's mkdir() has run."""
    client.run("--ask-only")
    server.calls.clear()
    fail_to_open(monkeypatch, client.out_dir, errno.EACCES)
    fail_to_look_at(monkeypatch, client.out_dir, errno.EACCES)
    status, _, err = client.run(*flags)
    words = (
        "error: --out-dir %s cannot be used: Permission denied: it may be another account's (%s)\n"
    )
    assert (status, err) == (2, words % (client.out_dir, advice)) and server.calls == []


def test_out_dir_that_disappears_after_a_refused_opening(client, server, monkeypatch):
    """Only a refused lstat() is taken for another account's directory."""
    client.run("--ask-only")
    server.calls.clear()
    fail_to_open(monkeypatch, client.out_dir, errno.EACCES)
    fail_to_look_at(monkeypatch, client.out_dir, errno.ENOENT)
    status, _, err = client.run("--resume")
    words = "does not exist, so there is nothing to resume (run the command without --resume)\n"
    assert (status, err) == (2, "error: --out-dir %s %s" % (client.out_dir, words))
    assert server.calls == []


def test_another_failure_to_open_out_dir_gets_no_advice(client, server, monkeypatch):
    fail_to_open(monkeypatch, client.out_dir, errno.EMFILE)
    status, _, err = client.run()
    assert (status, err) == (2, "error: %s: Too many open files\n" % client.out_dir)
    assert server.calls == []


OPEN_PLACE = (
    "ask_and_check.py: the --out-dir %s lies below %s, which another account may be able to change"
    " (its group or others can write in it, and it has no sticky bit): "
)


# levels: 1 puts --out-dir directly in the group-writable directory, 3 two directories deeper.
@pytest.mark.parametrize("flags", [(), ("--resume",)])
@pytest.mark.parametrize("levels", [1, 3])
def test_rejects_out_dir_below_a_directory_that_others_can_change(client, server, flags, levels):
    open_place = client.dir / "open"
    out_dir = open_place.joinpath(*["below"] * (levels - 1), "out")
    out_dir.parent.mkdir(parents=True)
    for directory in (out_dir.parent, *out_dir.parent.parents[: levels - 1]):
        directory.chmod(0o700)  # whatever the umask
    if flags:
        assert client.run("--ask-only", out_dir=out_dir)[0] == 75
    before = sorted(out_dir.parent.rglob("*"))
    server.calls.clear()
    open_place.chmod(0o775)
    status, out, err = client.run(*flags, out_dir=out_dir)
    assert (status, out) == (2, "result: outcome=error\n") and len(err.splitlines()) == 1
    assert err.startswith(OPEN_PLACE % (out_dir, open_place))
    assert server.calls == [] and not client.checked()
    assert sorted(out_dir.parent.rglob("*")) == before


def test_rejects_out_dir_below_a_link_to_a_directory_that_others_can_change(client, server):
    (client.dir / "open").mkdir()
    (client.dir / "open").chmod(0o777)
    (client.dir / "link").symlink_to("open")
    status, _, err = client.run(out_dir=client.dir / "link" / "out")
    assert status == 2 and err.startswith(
        OPEN_PLACE % (client.dir / "link/out", client.dir / "open")
    )
    assert server.calls == [] and os.listdir(client.dir / "open") == []


def sudo_account():
    """Return the account to which a test that runs as root gives its directory "theirs".

    Under sudo, pytest's directory may lie below one of the sudo caller's own directories, such as
    its home. The tests must leave SUDO_UID naming that account, or the client would refuse that
    directory too. So it is that account, or 54321 if SUDO_UID names no account other than root.
    """
    sudo = os.environ.get("SUDO_UID", "")
    return int(sudo) if re.fullmatch("[0-9]{1,10}", sudo) and 0 < int(sudo) < 2**32 - 1 else 54321


@pytest.mark.parametrize("look", ["by path", "on descriptors"])
@pytest.mark.parametrize("sudo_uid, status", [(None, 2), ("theirs", 0), ("other", 2)])
def test_out_dir_below_another_accounts_directory(
    client, server, monkeypatch, sudo_uid, status, look
):
    """Refused, unless the client runs as root and SUDO_UID names that account.

    The client looks twice, so each look is tried with the other one switched off.
    """
    if look == "by path":
        monkeypatch.setattr(ask_and_check, "trusted_owners", lambda: range(2**32))
    else:
        monkeypatch.setattr(ask_and_check, "others_can_change", lambda folder: None)
    theirs = client.dir / "theirs"
    theirs.mkdir(mode=0o700)
    if os.geteuid() == 0:
        os.chown(theirs, sudo_account(), -1)
    else:
        # Root for trusted_owners() alone, for which the tests' own account is then another. What
        # the client makes still belongs to that account.
        mine, asks = os.geteuid(), lambda: sys._getframe(2).f_code.co_name
        monkeypatch.setattr(os, "geteuid", lambda: 0 if asks() == "trusted_owners" else mine)
    owner = theirs.stat().st_uid
    monkeypatch.delenv("SUDO_UID", raising=False)
    if sudo_uid:
        monkeypatch.setenv("SUDO_UID", str(owner if sudo_uid == "theirs" else owner + 1))
    got, _, err = client.run(out_dir=theirs / "out")
    assert got == status and (server.calls == []) == (status == 2)
    assert ("(owned by another account)" in err) == (status == 2)
    assert (theirs / "out").exists() == (status == 0)


def test_sudo_uid_counts_only_for_root(client, server, monkeypatch):
    """When the caller is not root, the account that SUDO_UID names is still another account."""
    mine, account = os.geteuid(), sudo_account()
    caller = mine + 1 if account != mine + 1 else mine + 2  # neither root nor that account
    monkeypatch.setattr(os, "geteuid", lambda: caller)
    monkeypatch.setenv("SUDO_UID", str(mine))
    theirs = client.dir / "theirs"
    theirs.mkdir(mode=0o700)
    # trusted_owners() always includes root, so under root the directory needs another owner.
    if mine == 0:
        os.chown(theirs, account, -1)
        monkeypatch.setenv("SUDO_UID", str(account))
    status, _, err = client.run(out_dir=theirs / "out")
    assert status == 2 and "(owned by another account)" in err and server.calls == []


def test_out_dir_may_be_in_a_sticky_directory(client):
    (client.dir / "shared").mkdir()
    (client.dir / "shared").chmod(0o1777)
    assert client.run(out_dir=client.dir / "shared" / "out")[0] == 0


def test_links_above_out_dir_are_resolved_once(client):
    (client.dir / "real").mkdir(mode=0o700)
    (client.dir / "link").symlink_to("real")
    status, out, _ = client.run(out_dir=client.dir / "link" / "out")
    assert status == 0 and "stored in %s\n" % (client.dir / "real" / "out" / "served.json") in out
    assert str(client.dir / "link") not in " ".join(client.seen_by_check()["argv"])


@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_out_dir_needs_an_existing_parent(client, server, flags):
    status, out, err = client.run(*flags, out_dir=client.dir / "no\x1b\nne" / "out")
    missing = "%s/no\\x1b\\nne" % client.dir
    message = "error: --out-dir %s/out cannot be used: %s does not exist\n"
    assert (status, err) == (2, message % (missing, missing))
    assert out == "result: outcome=error\n" and server.calls == []


def put_in_its_place(path, other):
    """Move the directory 'path' aside and put a link to 'other', or 'other' itself, in its place,
    as an account that owns the directory above could."""
    path.rename(path.with_name(path.name + ".aside"))
    if other.name == "link":
        path.symlink_to(other.resolve())
    else:
        other.rename(path)


def before(monkeypatch, function, action):
    """Patch a function of the client: its first call runs action() first."""
    real, done = getattr(ask_and_check, function), []

    def wrapped(*args):
        if not done:
            done.append(action())
        return real(*args)

    monkeypatch.setattr(ask_and_check, function, wrapped)


@pytest.mark.parametrize(
    "function, swapped",
    [
        ("compare_commitment_id", "out"),
        ("ask", "out"),
        ("wait_for_proof", "out"),
        ("download_proof", "out"),
        ("check", "out"),
        ("check", "out/proof"),
    ],
)
def test_directory_swapped_during_the_run_leads_nothing_elsewhere(
    client, server, monkeypatch, function, swapped
):
    """Another directory under the name gets no file of the run, and none of its files is used."""
    planted = client.dir / "planted"
    (planted / "proof").mkdir(parents=True)
    for name in ("served.json", "request.bin", "proof/y.bin", "proof/proof.bin"):
        (planted / name).write_bytes(b"planted")
    (client.dir / "link").symlink_to("planted/proof" if swapped == "out/proof" else "planted")
    before(
        monkeypatch, function, lambda: put_in_its_place(client.dir / swapped, client.dir / "link")
    )
    status, _, err = client.run()
    assert (status, err) == (0, "")
    files = [path for path in planted.rglob("*") if path.is_file()]
    assert len(files) == 4 and all(path.read_bytes() == b"planted" for path in files)
    seen = client.seen_by_check()
    assert bytes.fromhex(seen["proof"]) == server.files["proof.bin"]
    assert bytes.fromhex(seen["request"]) == spool.whole_reply(PROMPT, REPLY, 16, fill=0)
    held = client.dir / "out.aside" if swapped == "out" else client.out_dir
    names = {"served.json", "request.bin", "proof.aside" if swapped == "out/proof" else "proof"}
    assert names <= set(os.listdir(held))


@pytest.mark.parametrize(
    "kind, words",
    [
        (
            "link",
            "error: --out-dir %s/out cannot be used: %s has changed since it was looked at: it is"
            " now a link or not a directory (run the command again, or choose a new --out-dir)\n",
        ),
        ("open", OPEN_PLACE.replace("%s", "%s/out", 1)),
    ],
)
@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_directory_above_swapped_after_the_look_is_refused(
    client, server, monkeypatch, flags, kind, words
):
    """Each directory above --out-dir is judged again once it is open."""
    above = client.dir / "above"
    above.mkdir(mode=0o700)
    if flags:
        assert client.run("--ask-only", out_dir=above / "out")[0] == 75
        server.calls.clear()
    (client.dir / "open").mkdir()
    (client.dir / "open").chmod(0o777)
    (client.dir / "link").symlink_to("open")
    before(monkeypatch, "open_above", lambda: put_in_its_place(above, client.dir / kind))
    status, out, err = client.run(*flags, out_dir=above / "out")
    assert status == 2 and err.startswith(words % (above, above)) and err.count("\n") == 1
    assert out == "result: outcome=error\n" and server.calls == []
    assert os.listdir(above) == []


@pytest.mark.parametrize("flags", [(), ("--resume",)])
def test_link_laid_above_right_after_the_first_look_is_refused(client, server, monkeypatch, flags):
    """Even one that leads to a directory that would pass both looks."""
    above, other = client.dir / "above", client.dir / "other"
    above.mkdir(mode=0o700)
    other.mkdir(mode=0o700)
    if flags:
        assert client.run("--ask-only", out_dir=above / "out")[0] == 75
        server.calls.clear()
    (client.dir / "link").symlink_to("other")
    look = ask_and_check.others_can_change

    def swapped_afterwards(folder):
        found = look(folder)
        put_in_its_place(above, client.dir / "link")
        return found

    monkeypatch.setattr(ask_and_check, "others_can_change", swapped_afterwards)
    status, _, err = client.run(*flags, out_dir=above / "out")
    assert status == 2 and "%s has changed since it was looked at: it is now a link" % above in err
    assert server.calls == [] and os.listdir(other) == []


def test_directory_above_that_another_account_owns_after_the_look_is_refused(
    client, server, monkeypatch
):
    above = client.dir / "above"
    above.mkdir(mode=0o700)

    def give_away():
        if os.geteuid() == 0:
            os.chown(above, sudo_account() + 1, -1)
        else:
            monkeypatch.setattr(ask_and_check, "trusted_owners", lambda: {0})

    before(monkeypatch, "open_above", give_away)
    status, _, err = client.run(out_dir=above / "out")
    assert status == 2 and "(owned by another account)" in err and err.count("\n") == 1
    assert server.calls == [] and os.listdir(above) == []


def test_directory_swapped_while_those_above_are_opened_leads_nothing_elsewhere(
    client, monkeypatch
):
    """Each directory is opened relative to the one before, not by its path."""
    first, other = client.dir / "first", client.dir / "other"
    for directory in (first, first / "second", other, other / "second"):
        directory.mkdir(mode=0o700)
    (client.dir / "link").symlink_to("other")
    real, done = os.open, []

    def swapped_first(path, *flags, **where):
        if not done and is_meant(first / "second", path, where):
            done.append(put_in_its_place(first, client.dir / "link"))
        return real(path, *flags, **where)

    before(monkeypatch, "open_above", lambda: monkeypatch.setattr(os, "open", swapped_first))
    assert client.run(out_dir=first / "second" / "out")[0] == 0 and done
    assert os.listdir(other / "second") == []
    assert os.listdir(client.dir / "first.aside" / "second") == ["out"]


def test_directory_above_swapped_once_it_is_open_leads_nothing_elsewhere(client, monkeypatch):
    above, other = client.dir / "above", client.dir / "other"
    above.mkdir(mode=0o700)
    other.mkdir(mode=0o700)
    (client.dir / "link").symlink_to("other")
    before(monkeypatch, "create_or_open", lambda: put_in_its_place(above, client.dir / "link"))
    assert client.run(out_dir=above / "out")[0] == 0
    assert os.listdir(other) == [] and os.listdir(client.dir / "above.aside") == ["out"]


def test_directory_above_that_cannot_be_opened_is_an_error(client, server, monkeypatch):
    real = os.open

    def denied(path, *flags, **where):
        if path == client.dir.name:
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return real(path, *flags, **where)

    before(monkeypatch, "open_above", lambda: monkeypatch.setattr(os, "open", denied))
    status, _, err = client.run()
    message = "error: --out-dir %s cannot be used: %s cannot be opened: Permission denied (choose"
    assert status == 2 and err.startswith(message % (client.out_dir, client.dir))
    assert server.calls == [] and not client.out_dir.exists()


def test_link_gives_a_symlink_itself_a_second_name(tmp_path):
    (tmp_path / "file").write_text("x")
    (tmp_path / "symlink").symlink_to("file")
    with ask_and_check.Directory(os.open(tmp_path, os.O_RDONLY), str(tmp_path)) as directory:
        directory.link("symlink", "second")
    assert (tmp_path / "second").is_symlink()


def test_needs_linux(client, server, monkeypatch):
    monkeypatch.delattr(os, "O_PATH")
    status, out, err = client.run()
    assert (status, out) == (2, "result: outcome=error\n")
    assert err == "error: this script needs Linux: it opens directories with O_PATH\n"
    assert server.calls == [] and not client.out_dir.exists()


def test_new_out_dir_is_judged_too(client, server, monkeypatch):
    """As on a file system that shows a new directory under another owner."""
    real, mine = os.mkdir, os.geteuid()

    def made_for_another(*name_and_mode, **where):
        real(*name_and_mode, **where)
        monkeypatch.setattr(os, "geteuid", lambda: mine + 1)

    monkeypatch.setattr(os, "mkdir", made_for_another)
    status, _, err = client.run()
    assert status == 2 and server.calls == [] and os.listdir(client.out_dir) == []
    assert "its owner is uid %d, and this script runs as uid %d" % (mine, mine + 1) in err


def test_what_is_judged_is_the_open_directory(client, server, monkeypatch):
    """Another directory that gets --out-dir's name right after the open does not pass for it."""
    client.out_dir.mkdir()
    client.out_dir.chmod(0o777)
    (client.dir / "private").mkdir(mode=0o700)
    real = os.open

    def swapped_afterwards(path, flags, *mode, **where):
        fd = real(path, flags, *mode, **where)
        if path == "out" and not flags & os.O_PATH:
            put_in_its_place(client.out_dir, client.dir / "private")
        return fd

    monkeypatch.setattr(os, "open", swapped_afterwards)
    status, _, err = client.run()
    assert status == 2 and "(mode 0777)" in err and server.calls == []
    assert os.listdir(client.out_dir) == os.listdir(client.dir / "out.aside") == []


@pytest.mark.parametrize(
    "road, status",
    [
        ("a pass", 0),
        ("the check fails", 2),
        ("no proof yet", 75),
        ("--out-dir is open to others", 2),
        ("the directory above is open to others", 2),
    ],
)
def test_no_descriptor_stays_open(client, server, monkeypatch, road, status):
    def open_files():
        """Return the files and directories that the process holds open, by descriptor."""
        found = {}
        for fd in os.listdir("/proc/self/fd"):
            with contextlib.suppress(OSError):
                found[fd] = os.readlink("/proc/self/fd/" + fd)
        return {fd: name for fd, name in found.items() if name[0] == "/" and "/proc/" not in name}

    above = client.dir / "above"
    above.mkdir(mode=0o700)
    if road == "the check fails":
        client.check_exits_with(2)
    if road == "no proof yet":
        server.job["state"] = "waiting"
    if road == "--out-dir is open to others":
        (above / "out").mkdir()
        (above / "out").chmod(0o777)
    if road == "the directory above is open to others":
        above.chmod(0o777)
        monkeypatch.setattr(ask_and_check, "others_can_change", lambda folder: None)
    held = open_files()
    assert client.run(wait=0, out_dir=above / "out")[0] == status and open_files() == held


def test_resume_needs_an_earlier_run(client, server):
    assert client.run("--resume")[0] == 2
    client.out_dir.mkdir(mode=0o700)
    assert client.run("--resume")[0] == 2 and server.calls == []


@pytest.mark.parametrize(
    "options", [{"prompt_ids": "1,2,4"}, {"max_tokens": 3}, {"model": "n"}, TEXT]
)
def test_resume_with_another_request_is_an_error(client, server, options):
    client.run("--ask-only")
    status, _, err = client.run("--resume", **options)
    assert status == 2 and "another request" in err


@pytest.mark.parametrize("text", ["", "{", "[]", '{"sent": 1}', "[" * 100000])
def test_resume_from_a_damaged_file_is_an_error(client, text):
    client.run("--ask-only")
    (client.out_dir / "served.json").write_text(text)
    status, _, err = client.run("--resume")
    assert status == 2 and "damaged" in err


def test_resume_rejects_stored_prompt_ids_that_differ(client):
    client.run("--ask-only")
    stored = json.loads((client.out_dir / "served.json").read_text())
    (client.out_dir / "served.json").write_text(
        json.dumps({**stored, "prompt_token_ids": [9, 9, 9]})
    )
    assert client.run("--resume")[0] == 2 and not client.checked()


def test_resume_completes_an_interrupted_download(client, server):
    client.run("--ask-only")
    proof_dir = client.out_dir / "proof"
    proof_dir.mkdir(mode=0o700)
    (proof_dir / "y.bin").write_bytes(server.files["y.bin"])  # complete, so not downloaded again
    (proof_dir / "proof.bin.part.0123456789abcdef").write_bytes(b"half")
    (proof_dir / "notes.txt").write_text("mine")
    assert client.run("--resume")[0] == 0
    kept = ["notes.txt", "proof.bin", "proof.bin.part.0123456789abcdef", "y.bin"]
    assert sorted(os.listdir(proof_dir)) == kept  # files that this run did not create are kept
    assert (proof_dir / "proof.bin").read_bytes() == server.files["proof.bin"]
    assert "GET " + JOB + "/y.bin" not in server.routes()


# ---- A prompt on standard input ----


@pytest.fixture
def put_on_stdin(monkeypatch):
    """Return a function that makes a file with the given bytes Client.run()'s standard input."""
    with contextlib.ExitStack() as files:

        def put(data):
            f = files.enter_context(tempfile.TemporaryFile())
            f.write(data)
            f.seek(0)
            monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(f))

        yield put


@pytest.mark.parametrize("data", [b"1,2,3", b"1,2,3\n"])
def test_prompt_ids_on_standard_input(client, server, put_on_stdin, data):
    put_on_stdin(data)
    assert client.run("--ask-only", prompt_ids="-")[0] == 75
    assert json.loads(server.calls[-1][3])["prompt"] == PROMPT
    # The run is stored like one with the IDs on the command line, which can therefore resume it.
    assert client.run("--resume")[0] == 0
    put_on_stdin(data)
    assert client.run("--resume", prompt_ids="-")[0] == 0
    put_on_stdin(b"1,2,4")
    assert "the reply to another request" in client.run("--resume", prompt_ids="-")[2]


@pytest.mark.parametrize("flags", [(), ("--chat",)])
@pytest.mark.parametrize(
    "data, prompt",
    [
        (b"hello", "hello"),
        (b"hello\n", "hello"),
        ("two\nlines \u00e9\n\n".encode(), "two\nlines \u00e9\n"),
    ],
)
def test_text_prompt_on_standard_input(client, server, put_on_stdin, flags, data, prompt):
    """Only one last line ending is removed."""
    put_on_stdin(data)
    assert client.run("--ask-only", *flags, prompt_ids=None, prompt="-")[0] == 75
    sent = json.loads(server.calls[-1][3])
    assert (sent["messages"][0]["content"] if flags else sent["prompt"]) == prompt


def test_prompt_on_standard_input_reaches_the_restarted_script(client, server):
    """The script, run by path, starts again in isolated mode before it reads standard input."""
    command = [sys.executable, "-B", ask_and_check.__file__]
    command += client.argv("--ask-only", prompt_ids="-")
    got = subprocess.run(command, input="1,2,3\n", capture_output=True, text=True)
    assert (got.returncode, got.stderr) == (75, "")
    assert json.loads(server.calls[-1][3])["prompt"] == PROMPT


LARGEST = b"1," * ((1 << 19) - 1) + b"11"  # 1 MiB


@pytest.mark.parametrize(
    "data, options, words",
    [
        (b"", {}, "the prompt on standard input is empty"),
        (b"\n", {}, "the prompt on standard input is empty"),
        (b"", TEXT, "the prompt on standard input is empty"),
        (b"1,2,x", {}, "--prompt-ids on standard input: expected whole numbers below 2^32"),
        (b"1, 2", {}, "--prompt-ids on standard input: expected whole numbers below 2^32"),
        (b"1,2\n\n", {}, "--prompt-ids on standard input: expected whole numbers below 2^32"),
        (b"-", {}, "--prompt-ids on standard input: expected whole numbers below 2^32"),
        (b"1,\xff", {}, "the prompt on standard input is not UTF-8"),
        (b"\xff", TEXT, "the prompt on standard input is not UTF-8"),
        (LARGEST + b"1", {}, "the prompt on standard input is longer than 1 MiB"),
        (LARGEST + b"\n", TEXT, "the prompt on standard input is longer than 1 MiB"),
    ],
)
def test_unusable_standard_input_is_an_error(client, server, put_on_stdin, data, options, words):
    put_on_stdin(data)
    name = "prompt" if options else "prompt_ids"
    status, _, err = client.run(**{**options, name: "-"})
    assert status == 2 and err.startswith("error: " + words) and err.count("\n") == 1
    assert server.calls == [] and not client.out_dir.exists()


def test_largest_prompt_on_standard_input_is_sent(client, server, put_on_stdin):
    put_on_stdin(LARGEST)
    status, _, err = client.run(prompt_ids="-")
    assert status == 1 and "the server returned other prompt token IDs than were sent" in err
    assert json.loads(server.calls[-1][3])["prompt"] == [1] * ((1 << 19) - 1) + [11]


def test_prompt_over_the_limit_is_not_cut_short_when_it_arrives_in_pieces(client, server):
    """The last byte comes when the client has read exactly 1 MiB: it must read on, and refuse."""
    command = [sys.executable, "-B", ask_and_check.__file__, *client.argv(prompt_ids="-")]
    pipes = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    with subprocess.Popen(command, **pipes) as got:
        got.stdin.write(LARGEST)  # returns once the client has read all but what the pipe holds
        got.stdin.flush()
        time.sleep(0.5)  # the client now has 1 MiB, and waits for more or for the end
        _, err = got.communicate(b"1")
    assert got.returncode == 2 and server.calls == [] and not client.out_dir.exists()
    assert err == b"error: the prompt on standard input is longer than 1 MiB\n"


@pytest.mark.parametrize("written, late", [(b"", False), (b"1,2,3", False), (b"1,2,3", True)])
def test_non_blocking_standard_input_is_an_error(client, server, monkeypatch, written, late):
    """There a read returns what has been written so far, which may be a part of the prompt.

    late: another process that shares the pipe sets the mode while the client is reading.
    """
    reader, writer = os.pipe()

    def switch_and_write():
        os.set_blocking(reader, False)
        os.write(writer, written)  # the pipe stays open: more may come

    other = threading.Timer(0.2 * late, switch_and_write)
    other.start()
    if not late:
        other.join()
    with open(reader) as stdin, open(writer, "wb"):
        monkeypatch.setattr(sys, "stdin", stdin)
        status, _, err = client.run(prompt_ids="-")
        other.join()
    words = (
        "error: cannot read the prompt from standard input: it is in non-blocking mode (give the"
        " prompt from a file: < FILE)\n"
    )
    assert (status, err) == (2, words) and server.calls == [] and not client.out_dir.exists()


def test_prompt_that_arrives_in_pieces_is_read_whole(client, server, monkeypatch):
    reader, writer = os.pipe()

    def write_slowly():
        with open(writer, "wb", buffering=0) as pipe:
            for piece in (b"1", b",2", b",", b"3", b"\n"):
                time.sleep(0.05)
                pipe.write(piece)

    other = threading.Thread(target=write_slowly)
    other.start()
    with open(reader) as stdin:
        monkeypatch.setattr(sys, "stdin", stdin)
        assert client.run("--ask-only", prompt_ids="-")[0] == 75
    other.join()
    assert json.loads(server.calls[-1][3])["prompt"] == PROMPT


@contextlib.contextmanager
def stdin_that_raises(monkeypatch, error):
    """Give Client.run() a standard input that raises 'error' when it is read."""
    real = os.read

    def read(fd, size):
        if fd == null.fileno():
            raise error
        return real(fd, size)

    with open(os.devnull) as null, monkeypatch.context() as patch:
        patch.setattr(sys, "stdin", null)
        patch.setattr(os, "read", read)
        yield


@pytest.mark.parametrize(
    "error, words",
    [
        (
            OSError(errno.EIO, os.strerror(errno.EIO)),
            "cannot read the prompt from standard input: Input/output error",
        ),
        (KeyboardInterrupt(), "interrupted (run the same command again)"),
    ],
)
def test_standard_input_that_cannot_be_read_is_an_error(client, server, monkeypatch, error, words):
    with stdin_that_raises(monkeypatch, error):
        status, _, err = client.run(prompt_ids="-")
    assert (status, err) == (2, "error: %s\n" % words)
    assert server.calls == [] and not client.out_dir.exists()


def test_standard_input_without_a_descriptor_is_an_error(client, server, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("1,2,3"))  # as a program that calls main() may
    status, _, err = client.run(prompt_ids="-")
    words = "error: cannot read the prompt from standard input: it has no file descriptor\n"
    assert (status, err) == (2, words)
    assert server.calls == [] and not client.out_dir.exists()


def test_closed_standard_input_is_an_error(client, server, monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)  # as Python sets it at its start
    status, _, err = client.run(prompt_ids="-")
    assert (status, err) == (2, "error: cannot read the prompt from standard input: it is closed\n")
    assert server.calls == [] and not client.out_dir.exists()


def test_standard_input_is_left_alone_without_a_dash(client, monkeypatch):
    with stdin_that_raises(monkeypatch, AssertionError("read")):
        assert client.run()[0] == 0
        assert client.run(out_dir=client.dir / "text", **TEXT)[0] == 24


# ---- Standard descriptors that the caller has closed ----


def run_with(client, redirections, **options):
    """Run the client as a program, with the fake check script, under a shell's redirections."""
    (client.dir / "run_client.py").write_text(RUN_CLIENT)
    command = [sys.executable, "-B", client.dir / "run_client.py", client.dir / "check.py"]
    command = " ".join(shlex.quote(str(word)) for word in [*command, *client.argv(**options)])
    return subprocess.run(
        ["/bin/sh", "-c", "exec %s %s" % (command, redirections)],
        env={**os.environ, "PYTHONPATH": os.path.dirname(ask_and_check.__file__)},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("deeper", [False, True])
@pytest.mark.parametrize("closed", ["<&-", "2>&-", "<&- 2>&-"])
def test_run_without_standard_input_or_standard_error_passes(client, closed, deeper):
    """No directory that the client holds open may become the check script's standard input or
    standard error: with a directory as its standard input, Python does not even start.

    deeper: on its way down to --out-dir the client opens and closes directories by turns, so the
    path's depth decides which of them would get a free number.
    """
    (client.dir / "a").mkdir(mode=0o700)
    got = run_with(client, closed, out_dir=client.dir / "a" / "out" if deeper else client.out_dir)
    assert (got.returncode, got.stderr) == (0, "")
    assert got.stdout.splitlines()[-1] == "check_proof: verified: fake"
    for fd, redirection in ((0, "<&-"), (2, "2>&-")):
        # What the check script opens must not get such a number either.
        if redirection in closed:
            assert client.seen_by_check()["0 and 2"][fd // 2] == os.devnull


@pytest.mark.parametrize("closed, said", [(">&-", True), ("<&- >&- 2>&-", False)])
def test_run_without_standard_output_is_an_error(client, server, closed, said):
    got = run_with(client, closed)
    words = (
        "error: standard output is closed, but the result is printed there (run the command with"
        " standard output open)\n"
    )
    assert (got.returncode, got.stdout, got.stderr) == (2, "", words * said)
    assert server.calls == [] and not client.out_dir.exists()


def test_dev_null_that_cannot_be_opened_is_an_error(client, server, monkeypatch):
    real = os.open

    def out_of_descriptors(path, *rest, **options):
        if path == os.devnull:
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE), path)
        return real(path, *rest, **options)

    monkeypatch.setattr(os, "open", out_of_descriptors)
    status, _, err = client.run()
    assert (status, err) == (2, "error: cannot open /dev/null: Too many open files\n")
    assert server.calls == [] and not client.out_dir.exists()


def test_messages_stay_off_standard_output_without_standard_error(client, monkeypatch):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(ask_and_check, "ask", interrupted)
    monkeypatch.setattr(sys, "stderr", None)  # as Python sets it at its start
    status, out, _ = client.run()
    assert (status, out) == (2, "result: outcome=error\n")


# ---- The command line ----


@pytest.mark.parametrize(
    "option",
    [
        "--commitment-id",
        "--commitment-id=" + "cd" * 32,
        "--commit",
        "--public-key-dir",
        "--parameter-file",
        "--scales-file",
        "--reply",
        "--range",
        "--end-token",
        "--verifier",
        "--next-token",
        "--bits",
        "other-proof",
    ],
)
def test_options_that_the_script_sets_cannot_follow_the_dashes(client, server, option):
    status, _, err = client.run(after=[option, "x"])
    assert status == 2 and 'cannot follow "--"' in err and server.calls == []


@pytest.mark.parametrize(
    "flags, options, after",
    [
        (("--end-token", "8"), {}, ()),
        ((), {}, ("--C", "1", "--C=2")),
        (("--chat",), {}, ()),
        (("--prompt", "x"), {}, ()),
        (("--max", "3"), {}, ()),
        ((), {"end_token": "7, 8"}, ()),
        ((), {"end_token": "x"}, ()),
        ((), {"prompt_ids": str(1 << 32)}, ()),
        ((), {"commitment_id": "ab"}, ()),
        ((), {"policy_name": "a\nb"}, ()),
        ((), {"policy_name": "a/b"}, ()),
        ((), {"max_tokens": "2.5"}, ()),
        ((), {"out_dir": ""}, ()),
        ((), {"server": "localhost:8000"}, ()),
        ((), {"server": "xyz://localhost"}, ()),
        ((), {"server": "http://"}, ()),
        ((), {"scales_file": "/nonexistent"}, ()),
        ((), {"model": None}, ()),
        ((), {"max_tokens": 0}, ()),
        ((), {"wait": -1}, ()),
        ((), {"wait": "nan"}, ()),
        ((), {"wait": "inf"}, ()),
        ((), {}, ("--C",)),
        ((), {}, ("--C", "--theta")),
        ((), {}, ("--C", "--show-all")),
        ((), {}, ("--show-all=1",)),
        ((), {}, ("--show-all", "1")),
        ((), {}, ("--show-all", "--show-all")),
        ((), {}, ("--no-proof-of-work=1",)),
        ((), {}, ("--no-proof-of-work", "1")),
        (("--no-proof-of-work",), {}, ()),
        ((), {}, ("--no-such-option",)),
        (("-h",), {}, ()),
        (("-hh",), {}, ()),
        (("-hx",), {}, ()),
        (("--help",), {}, ()),
        (("--help=x",), {}, ()),
        ((), {"max_tokens": "abc"}, ()),
        (("extra",), {}, ()),
    ],
)
def test_usage_errors(client, server, flags, options, after):
    status, _, err = client.run(*flags, after=after, **options)
    assert (
        status == 2
        and err.startswith("usage: ")
        and len(err.splitlines()) == 2
        and server.calls == []
    )


@pytest.mark.parametrize(
    "text, words",
    [
        ("", "does not end with a newline"),
        (POLICY[:-1], "does not end with a newline"),
        *((POLICY.replace("lt 4", line), "not a name and a plain") for line in ("lt 04", "lt x")),
        *(
            (POLICY.replace("lt 4", line), 'needs a line "lt N"')
            for line in ("lt 3", "lt 17", "x 4")
        ),
        ("lt 4\n", "check_proof.py: it has no line C\n"),
        (POLICY + "temp 256\n", 'check_proof.py: it has a line "temp"\n'),
        (POLICY.replace("sh 8", "sh 9"), "it has sh, sT or F other than 8, 4 and 24\n"),
    ],
)
def test_rejects_parameter_file_that_cannot_be_used(client, server, text, words):
    (client.dir / "policy.txt").write_text(text)
    status, _, err = client.run()
    assert status == 2 and "parameter file" in err and words in err
    assert server.calls == [] and not client.out_dir.exists()


def test_run_from_a_checkout_leaves_no_bytecode_there(tmp_path):
    for name in ("ask_and_check", "check_proof", "spool"):
        shutil.copy(os.path.join(os.path.dirname(ask_and_check.__file__), name + ".py"), tmp_path)
    names = ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")
    environment = {name: value for name, value in os.environ.items() if name not in names}
    command = [sys.executable, tmp_path / "ask_and_check.py", "--help"]
    assert subprocess.run(command, env=environment, capture_output=True).returncode == 0
    assert len(os.listdir(tmp_path)) == 3


def test_client_and_middleware_agree_on_the_protocol():
    for name in ("PREFIX", "POLICY_HEADER", "ID_HEADER", "PADDING"):
        assert getattr(ask_and_check, name) == getattr(proof_middleware, name)
    assert ask_and_check.POLICY_NAME == proof_middleware.POLICY_NAME.pattern
    assert ask_and_check.DOWNLOADS == tuple(proof_middleware.DOWNLOADS)


def test_help_is_short():
    got = subprocess.run(
        [sys.executable, "-B", ask_and_check.__file__, "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "80"},
    )
    assert (
        got.returncode == 0 and len(got.stdout.splitlines()) <= 60 and "exit codes:" in got.stdout
    )
    words = "Take --commitment-id from where the commitment is published, never from this server."
    assert words in " ".join(got.stdout.split())


@pytest.mark.parametrize("name", ["others_can_change", "MAX_FETCHED_BYTES"])
def test_neighbour_without_a_needed_name_is_an_error(tmp_path, name):
    """check_proof.py loads but, like a copy from another version, lacks a name that the client
    imports. The client must exit: nothing may stand in for the missing name."""
    for module in ("ask_and_check", "spool", "check_proof"):
        shutil.copy(os.path.join(os.path.dirname(ask_and_check.__file__), module + ".py"), tmp_path)
    with open(tmp_path / "check_proof.py", "a") as f:
        f.write("\ndel %s\n" % name)
    command = [sys.executable, "-B", tmp_path / "ask_and_check.py", "--help"]
    got = subprocess.run(command, capture_output=True, text=True)
    assert got.returncode == 2 and got.stdout == "result: outcome=error\n"
    assert got.stderr.startswith("error: cannot import check_proof.py and spool.py (")


# Only the client is copied. Without site-packages (-S) and PYTHONPATH, Python finds no
# check_proof.py; with PYTHONSAFEPATH set, it would not search the current directory for the client.
def test_failed_import_as_a_module_is_an_error(tmp_path):
    shutil.copy(ask_and_check.__file__, tmp_path)
    names = ("PYTHONPATH", "PYTHONSAFEPATH")
    environment = {name: value for name, value in os.environ.items() if name not in names}
    command = [sys.executable, "-B", "-S", "-c", "import ask_and_check"]
    got = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True)
    assert got.returncode == 2 and got.stdout == "result: outcome=error\n"
    assert got.stderr.startswith("error: cannot import check_proof.py and spool.py (")
    assert got.stderr.endswith("): install them with pip install ./integrations\n")


# ---- Isolated mode ----

# A false module has a real module's name. If imported, it appends that name to the file that the
# environment variable SEEN names, and ends the program. It imports posix, which is built in,
# because os can itself be a false module.
FALSE_MODULE = """import posix
open(posix.environ[b"SEEN"], "a").write(__name__)
raise SystemExit(99)
"""


@pytest.mark.parametrize("flags", [[], ["-B"], ["-S"], ["-I"]])
@pytest.mark.parametrize("start", ["by path", "through a symlink", "as a module"])
def test_false_modules_are_not_imported(tmp_path, flags, start):
    """False modules are on PYTHONPATH, in the current directory, beside the symlink and (with
    standard modules' names only) beside the client."""
    if start == "as a module" and "-I" in flags:
        pytest.skip("python3 -I -m finds no module in the current directory")
    checkout, false = tmp_path / "checkout", tmp_path / "false"
    checkout.mkdir()
    for name in ("ask_and_check", "check_proof", "spool"):
        shutil.copy(os.path.join(os.path.dirname(ask_and_check.__file__), name + ".py"), checkout)
    false.mkdir()
    (false / "client").symlink_to(checkout / "ask_and_check.py")
    (checkout / "loaded.py").write_text("import sys\nprint(*sys.modules)\n")
    words, directory = {
        "by path": ([checkout / "ask_and_check.py"], false),
        "through a symlink": ([false / "client"], false),
        "as a module": (["-m", "ask_and_check"], checkout),
    }[start]
    environment = {**os.environ, "PYTHONPATH": str(false), "SEEN": str(tmp_path / "seen")}
    # Python itself may import some standard modules while it starts (for site, .pth files or -m).
    # The client cannot prevent that, so only modules that the probe did not load get a false copy.
    probe = ["-m", "loaded"] if start == "as a module" else [checkout / "loaded.py"]
    command = [sys.executable, *flags, *probe]
    probe = subprocess.run(command, env=environment, cwd=directory, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    # Of the three scripts only spool.py imports hmac. If spool.py's imports searched the client's
    # directory, they would find the false hmac.py there.
    standard = {"os", "json", "argparse", "struct", "ssl", "hmac"} - set(probe.stdout.split())
    assert standard
    for target, names in ((false, ("check_proof", "spool", *standard)), (checkout, standard)):
        for name in names:
            (target / (name + ".py")).write_text(FALSE_MODULE)
    command = [sys.executable, *flags, *words, "--help"]
    got = subprocess.run(command, env=environment, cwd=directory, capture_output=True, text=True)
    assert got.returncode == 0 and "exit codes:" in got.stdout and got.stderr == ""
    assert not (tmp_path / "seen").exists()


# kept: the options that follow -I in the restart; None: already in isolated mode, so no restart.
@pytest.mark.parametrize(
    "isolated, no_bytecode, no_site, kept",
    [(0, 0, 0, []), (0, 1, 0, ["-B"]), (0, 0, 1, ["-S"]), (0, 1, 1, ["-B", "-S"]), (1, 1, 1, None)],
)
def test_restart_in_isolated_mode_keeps_arguments_and_flags(
    monkeypatch, isolated, no_bytecode, no_site, kept
):
    calls = []
    flags = {name: getattr(sys.flags, name) for name in dir(sys.flags) if name[0] != "_"}
    flags.update(isolated=isolated, dont_write_bytecode=no_bytecode, no_site=no_site)
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(**flags))
    monkeypatch.setattr(posix, "execv", lambda *call: calls.append(call))
    monkeypatch.setattr(sys, "argv", ["x", "a b", "--c"])
    ask_and_check.isolate()
    if kept is None:
        assert calls == []
    else:
        words = [sys.executable, "-I", *kept, ask_and_check.__file__, "a b", "--c"]
        assert calls == [(sys.executable, words)]


@pytest.mark.skipif(bool(sys.flags.isolated), reason="pytest itself runs in isolated mode")
@pytest.mark.parametrize("failure", [OSError(errno.ENOENT, "No such file"), ValueError("empty")])
def test_failed_restart_is_an_error(monkeypatch, capsys, failure):
    def fail(*_):
        raise failure

    monkeypatch.setattr(posix, "execv", fail)
    with pytest.raises(SystemExit) as ended:
        ask_and_check.isolate()
    out, err = capsys.readouterr()
    assert (ended.value.code, out) == (2, "result: outcome=error\n")
    assert err.startswith("error: cannot start %r in isolated mode: " % sys.executable)
    assert err.count("\n") == 1
