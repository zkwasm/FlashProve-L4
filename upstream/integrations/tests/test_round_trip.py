# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests that run the example client, the middleware, check_proof.py and end_to_end.py together, on
stand-ins for vLLM, cp_prove and cp_verify. No proof is made or verified."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

import ask_and_check
import dummy_service
import dummy_vllm
import end_to_end
import fake_cp_verify
import proof_middleware

CHILD_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}  # pytest's import path
API_KEY = "a-key-for-this-test"
# TOKENS: tokens per request; must equal 2**lt (PARAMETERS).
TOKENS, END_TOKEN = 16, 40
PARAMETERS = "version H821\nrecord 1\nlt 4\nV 512\nC %d\ntheta %d\ncap %d\ns 15\nsh 8\nsT 4\nF 24\n"
EXACT, CLIPPED_GAP = (256, 256, 0), (1000, 300, 1)  # C, theta, cap
WORK_LINE = "work positions 20 28 32\n"
# The client's exit codes. 75: stored, not checked (--ask-only).
PASSED, REJECTED, STORED = 0, 1, 75

# A wrapper for fake_cp_verify.py, which checks nothing in a proof. It makes the fake reject a
# proof without the request's SHA-256 (dummy_service.dummy_proof() includes it), report counted work
# if the parameter file has a line "work positions", and removes fake_cp_verify.MARK from each
# line's end, so that check_proof.py reads cp_verify's lines.
VERIFIER = """#!%s
import hashlib, json, os, subprocess, sys

arguments, changes = sys.argv[1:], {}
if arguments[0] != "-plan":
    with open(os.path.join(arguments[0], "policy.txt")) as parameters:
        if "\\nwork positions " in parameters.read():
            changes = {"work": True}
    with open(arguments[2], "rb") as request, open(arguments[4], "rb") as proof:
        if hashlib.sha256(request.read()).hexdigest().encode() not in proof.read():
            changes = {"status": 1, "last_line": "NO: another-request"}
environment = {**os.environ, "FAKE_CP_VERIFY": json.dumps(changes)}
command = [sys.executable, "-S", "-B", %r, *arguments]
done = subprocess.run(command, env=environment, stdout=subprocess.PIPE, text=True)
sys.stdout.write(done.stdout.replace(%r, "\\n"))
sys.exit(done.returncode)
""" % (sys.executable, fake_cp_verify.__file__, fake_cp_verify.MARK + "\n")
# dummy_service.py accepts only a spool directory below TMPDIR.
CP_PROVE = """#!/bin/sh
echo "setting: a stand-in; built for tests alone"  # like dummy_service.py's line on the cause
while [ $# -gt 0 ]; do
    case "$1" in
    -isk | -batch) shift 2 ;;
    -spool) spool=$2; shift 2 ;;
    -end-when-empty) shift ;;
    *) echo "cp_prove: unknown option $1"; echo "p258 end 1 usage"; exit 1 ;;
    esac
done
TMPDIR=$(dirname "$spool") exec "%s" -B "%s" "$spool" --tokens %d --end-when-empty
""" % (sys.executable, dummy_service.__file__, TOKENS)
VLLM = '#!/bin/sh\nexec "%s" -B "%s" "$@"\n' % (sys.executable, dummy_vllm.__file__)


def write_program(path, text):
    path.write_text(text)
    path.chmod(0o700)


class Files:
    """A provider's and a client's files before the first request."""

    def __init__(self, top):
        self.top = top
        for name in ("configuration", "public_key/leaf", "secret_key", "model", "build", "client"):
            (top / name).mkdir(parents=True)
        (top / "configuration" / "KL.bin").write_bytes(bytes(104))
        (top / "configuration" / "layout.txt").write_text("1 1 1 M 4 0\n")
        key = b"a stand-in for key.bin"
        (top / "public_key" / "key.bin").write_bytes(key)
        (top / "client" / "key.bin").write_bytes(key)  # the leaf file comes from the server
        (top / "public_key" / "leaf" / "0.bin").write_bytes(b"a stand-in for a leaf file")
        write_program(top / "cp_verify", VERIFIER)
        write_program(top / "build" / "cp_prove", CP_PROVE)
        write_program(top / "vllm", VLLM)
        self.set_policy(EXACT)
        self.set_model(step=1)

    def set_policy(self, numbers, counted_work=False):
        text = PARAMETERS % numbers + WORK_LINE * counted_work
        (self.top / "configuration" / "policy.txt").write_text(text)
        key = (self.top / "public_key" / "key.bin").read_bytes()
        self.commitment_id = fake_cp_verify.commitment_id(text.encode(), key)

    def set_model(self, step):
        """Set the step by which dummy_vllm.py's replies count up. dummy_service.py expects 1."""
        config = {"eos_token_id": END_TOKEN, "step": step}
        (self.top / "model" / "config.json").write_text(json.dumps(config))

    def check_options(self):
        return [
            *("--commitment-id", self.commitment_id),
            *("--parameter-file", str(self.top / "configuration" / "policy.txt")),
            *("--scales-file", str(self.top / "configuration" / "KL.bin")),
            *("--end-token", str(END_TOKEN)),
            *("--verifier", str(self.top / "cp_verify")),
        ]


@pytest.fixture(autouse=True)
def with_the_default_sigchld():
    """Under an ignored SIGCHLD, which can be inherited, the kernel reaps a test's children as soon
    as they exit: their exit codes are lost, and a PID that a test still holds can pass to another
    process."""
    set_handler = signal.signal
    before = set_handler(signal.SIGCHLD, signal.SIG_DFL)
    yield
    set_handler(signal.SIGCHLD, before)


@pytest.fixture
def files(tmp_path):
    return Files(tmp_path)


# ---- Client, middleware, service ----


class Provider:
    """A running dummy_vllm.py with the middleware."""

    def __init__(self, files):
        self.files, self.spool_dir = files, files.top / "spool"
        policy = {
            "served_model": "demo",
            "commitment_id": files.commitment_id,
            "parameter_file": str(files.top / "configuration" / "policy.txt"),
            "spool_dir": str(self.spool_dir),
            "public_key_dir": str(files.top / "public_key"),
            "verifier": str(files.top / "cp_verify"),
        }
        config = {"jobs_dir": str(files.top / "jobs"), "policies": {"demo": policy}}
        (files.top / "middleware.json").write_text(json.dumps(config))
        (files.top / "middleware.json").chmod(0o600)  # whatever the umask
        own = {
            "VLLM_API_KEY": API_KEY,
            proof_middleware.CONFIG_VARIABLE: str(files.top / "middleware.json"),
        }
        argv = [str(files.top / "vllm"), "serve", str(files.top / "model")]
        argv += ["--middleware", "proof_middleware.ProofMiddleware"]
        self.server = subprocess.Popen(
            argv, env={**CHILD_ENV, **own}, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.url = "http://127.0.0.1:%d" % int(self.server.stdout.readline().split()[-1])

    def prove(self):
        argv = [str(self.files.top / "build" / "cp_prove"), "-spool", str(self.spool_dir)]
        return subprocess.run(
            argv, env=CHILD_ENV, check=True, capture_output=True, text=True, timeout=60
        ).stdout

    def client(self, *more, prompt="1,2,3", api_key=API_KEY):
        """Run the example client. Return its exit code, its last outcome= field and its output."""
        argv = [sys.executable, "-B", ask_and_check.__file__, "--server", self.url]
        argv += ["--model", "demo", "--policy-name", "demo", "--prompt-ids", prompt]
        argv += ["--max-tokens", "4", "--out-dir", str(self.files.top / "out")]
        argv += ["--public-key-dir", str(self.files.top / "client"), *self.files.check_options()]
        environment = {**CHILD_ENV, "VLLM_API_KEY": api_key, "TMPDIR": str(self.files.top)}
        done = subprocess.run(
            [*argv, *more], env=environment, capture_output=True, text=True, timeout=60
        )
        results = [line for line in done.stdout.splitlines() if line.startswith("result: ")]
        return done.returncode, results[-1].split()[1], done.stdout + done.stderr

    def job_state(self):
        served = json.loads((self.files.top / "out" / "served.json").read_text())
        route = self.url + proof_middleware.PREFIX + "jobs/" + served["proof_id"]
        request = urllib.request.Request(route, headers={"Authorization": "Bearer " + API_KEY})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy
        with opener.open(request, timeout=60) as response:
            return json.load(response)["state"]


@pytest.fixture
def start_provider(files):
    """Return a function that starts the server, so that a test can choose the model first."""
    started = []

    def start():
        started.append(Provider(files))
        return started[-1]

    yield start
    for one in started:
        one.server.kill()
        one.server.communicate()


@pytest.mark.parametrize("umask", [0o022, 0o002])  # 002: many accounts' default
@pytest.mark.parametrize("prompt, reply", [("1,2,3", [4, 5, 6, 7]), ("38", [39, END_TOKEN])])
def test_a_served_reply_gets_a_proof_that_passes(start_provider, files, prompt, reply, umask):
    os.umask(umask)  # for every program from here on; the fixture in conftest.py puts it back
    provider = start_provider()
    assert provider.client("--ask-only", prompt=prompt)[:2] == (STORED, "outcome=stored")
    served = json.loads((files.top / "out" / "served.json").read_text())
    assert served["reply_token_ids"] == reply
    assert provider.job_state() == "waiting"
    assert "failed" not in provider.prove()
    code, outcome, printed = provider.client("--resume", prompt=prompt)
    assert (code, outcome) == (PASSED, "outcome=verified"), printed
    assert (files.top / "client" / "leaf" / "0.bin").exists()


def test_a_reply_that_was_changed_after_it_was_served_is_rejected(start_provider, files):
    provider = start_provider()
    assert provider.client("--ask-only")[0] == STORED
    provider.prove()
    served_json = files.top / "out" / "served.json"
    served = json.loads(served_json.read_text())
    served["reply_token_ids"][-1] += 1
    served_json.write_text(json.dumps(served))
    code, outcome, printed = provider.client("--resume")
    assert (code, outcome) == (REJECTED, "outcome=rejected"), printed
    assert "check_proof: rejected: cp_verify printed NO: another-request" in printed
    assert provider.job_state() == "proved"  # the server keeps its copy of the proof


def test_a_reply_of_another_model_gets_no_proof(start_provider, files):
    files.set_model(step=2)
    provider = start_provider()
    assert provider.client("--ask-only")[0] == STORED
    # 26: the tokens do not meet the policy
    assert ": failed: cause 26" in provider.prove()
    code, outcome, printed = provider.client("--resume")
    assert (code, outcome) == (REJECTED, "outcome=no-proof"), printed
    assert (files.top / "out" / "no_proof.txt").read_text() == proof_middleware.DECLINED
    assert not (files.top / "out" / "proof").exists()


def test_the_client_waits_for_a_proof_that_is_made_while_it_runs(start_provider, files):
    provider = start_provider()
    argv = [sys.executable, "-B", dummy_service.__file__, str(provider.spool_dir)]
    argv += ["--tokens", str(TOKENS)]
    environment = {**CHILD_ENV, "TMPDIR": str(files.top)}
    service = subprocess.Popen(argv, env=environment)
    try:
        code, outcome, printed = provider.client()
    finally:
        service.kill()
        service.wait()
    assert (code, outcome) == (PASSED, "outcome=verified"), printed


def test_the_stand_ins_help_works_under_python_oo():
    argv = [sys.executable, "-OO", "-B", dummy_vllm.__file__, "--help"]
    got = subprocess.run(argv, capture_output=True, text=True, env=CHILD_ENV)
    assert (got.returncode, got.stderr) == (0, "") and got.stdout.startswith("usage: ")


def test_the_server_requires_the_api_key_for_a_reply(start_provider):
    provider = start_provider()
    code, outcome, printed = provider.client(api_key="another-key")
    assert (code, outcome) == (2, "outcome=error")
    assert "status 401" in printed


# ---- end_to_end.py ----


needs_python_3_11 = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="end_to_end.py needs Python 3.11 or later"
)


@pytest.fixture
def demo(files, monkeypatch):
    """Return a function that runs end_to_end.main() on the stand-ins, with its options for your own
    commitment. Then restore the signal handlers, umask and environment, which main() changes."""
    if not re.fullmatch("[A-Za-z0-9/._-]+", str(files.top)):
        pytest.skip(
            "the script takes a WORK_DIR of letters, digits and / . _ - alone, and the path of "
            "pytest's temporary directory has another character: %r. Choose one whose path has "
            "none: pytest takes it from TMPDIR; if you passed --basetemp, give it the name of a "
            "new directory (pytest removes an existing one)" % str(files.top)
        )
    monkeypatch.setattr(end_to_end, "check_prerequisites", lambda args: [])
    monkeypatch.setattr(end_to_end, "vllm_program", lambda: str(files.top / "vllm"))
    numbers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGCHLD)
    handlers = {number: signal.getsignal(number) for number in numbers}
    mask, environment = os.umask(0o077), dict(os.environ)
    os.environ.update(PYTHONPATH=CHILD_ENV["PYTHONPATH"], TMPDIR=str(files.top))
    (files.top / "prompts.txt").write_text("1,2,3\n38\n7\n")

    def run(*more):
        argv = [
            str(files.top / "work"),
            *files.check_options(),
            *("--public-key-dir", str(files.top / "public_key")),
            *("--secret-key-dir", str(files.top / "secret_key")),
            *("--model-dir", str(files.top / "model")),
            *("--prompts-file", str(files.top / "prompts.txt")),
            *("--build-dir", str(files.top / "build")),
            *("--max-tokens", "4"),
        ]
        return end_to_end.main([*argv, *(more or ("--serve", "integer"))])

    yield run
    os.umask(mask)
    os.environ.clear()
    os.environ.update(environment)
    for number, handler in handlers.items():
        signal.signal(number, handler)


@needs_python_3_11
def test_the_steps_run_in_order_and_give_the_expected_result(demo, files, capfd):
    assert demo() == 0
    lines = capfd.readouterr().out.splitlines()
    assert [line[:2] for line in lines if line[1:3] == ". "] == ["%d." % n for n in range(1, 8)]
    checks = [line for line in lines if line.startswith("   reply ") and "check_proof: " in line]
    assert len(checks) == 3 and all("; proof of work: none counted; " in line for line in checks)
    for result in (
        "   3 of 3 proofs passed",
        "   the proving service: no proof, because the tokens do not meet the policy, as expected",
        "   check_proof.py, with the same proof and the changed reply: rejected, as expected",
    ):
        assert result in lines
    assert lines[-3] == (
        "Note: cp_prove in %s is not the build that is meant for serving: by its log it is "
        '"built for tests alone". The build meant for serving is made by prover/build.sh run '
        "without an argument." % (files.top / "build")
    )
    assert lines[-2].startswith("Note: The masking is built in and on by default; ")
    assert lines[-2].endswith('not proved for the programs. See docs/limits.md, "Weight hiding".')
    assert lines[-1] == "RESULT: every step gave the expected result"


@needs_python_3_11
def test_the_steps_pass_with_counted_work_and_each_check_shows_what_the_commitment_rests_on(
    demo, files, capfd
):
    files.set_policy(EXACT, counted_work=True)
    assert demo() == 0
    lines = capfd.readouterr().out.splitlines()
    footing = fake_cp_verify.RECORD_CLAUSE + "; " + fake_cp_verify.WORK
    checks = [line for line in lines if line.startswith("   reply ") and "check_proof: " in line]
    assert len(checks) == 3 and all(footing in line for line in checks)
    assert "   3 of 3 proofs passed" in lines
    assert lines[-1] == "RESULT: every step gave the expected result"


@needs_python_3_11
def test_the_steps_pass_under_a_clipped_gap_policy(demo, files, capfd):
    files.set_policy(CLIPPED_GAP)
    assert demo("--serve", "float", "--limits", "%d,%d,%d,15" % CLIPPED_GAP) == 0
    lines = capfd.readouterr().out.splitlines()
    assert "   3 of 3 proofs passed" in lines
    assert lines[-4].startswith("Note: the clipped-gap policy bounds a mean and a count ")
    assert lines[-1].endswith("gave the expected result; not checked: the negative control")


@needs_python_3_11
def test_stop_vllm_to_prove_stops_vllm_whenever_cp_prove_runs(demo, monkeypatch, capfd):
    events, start = [], end_to_end.Programs.start

    def recording_start(self, name, *args, **kwargs):
        events.append("%s, with %d running" % (re.sub(r"_\d+$", "", name), len(self.running)))
        return start(self, name, *args, **kwargs)

    monkeypatch.setattr(end_to_end.Programs, "start", recording_start)
    assert demo("--serve", "integer", "--stop-vllm-to-prove") == 0
    assert events == [
        "vllm, with 0 running",
        *["ask, with 1 running"] * 3,
        "cp_prove, with 0 running",
        "vllm_again, with 0 running",
        *["check, with 1 running"] * 3,
        "cp_prove_control, with 0 running",
        "control_unchanged, with 0 running",
        "control_changed, with 0 running",
    ]
    lines = capfd.readouterr().out.splitlines()
    assert "7. vLLM is already stopped." in lines
    assert lines[-1] == "RESULT: every step gave the expected result"


@needs_python_3_11
def test_an_interrupt_during_the_last_step_gives_the_signals_exit_code(demo, monkeypatch, capfd):
    killpg, interrupts = os.killpg, [signal.SIGINT]

    def interrupted_killpg(group, number):
        killpg(group, number)  # the script's own call, passed on
        if number == signal.SIGTERM and interrupts:  # first SIGTERM (step 7): simulate Ctrl-C
            signal.getsignal(signal.SIGINT)(interrupts.pop(), None)

    monkeypatch.setattr(os, "killpg", interrupted_killpg)
    # main() installs its SIGINT handler only if SIGINT is not ignored.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    assert demo() == 130
    lines = capfd.readouterr().out.splitlines()
    assert "7. Stop vLLM." in lines and not any(line.startswith("RESULT") for line in lines)


@needs_python_3_11
def test_the_negative_control_runs_check_proof_isolated(demo, monkeypatch):
    commands, start = {}, end_to_end.Programs.start

    def recording_start(self, name, argv, *args, **kwargs):
        commands[name] = argv
        return start(self, name, argv, *args, **kwargs)

    monkeypatch.setattr(end_to_end.Programs, "start", recording_start)
    assert demo() == 0
    for name in ("control_unchanged", "control_changed"):
        assert commands[name][:3] == [sys.executable, "-I", end_to_end.check_proof.__file__]


@needs_python_3_11
def test_a_server_of_another_model_fails_the_run(demo, files, capfd):
    files.set_model(step=2)
    assert demo() == 1
    printed = capfd.readouterr()
    assert "   0 of 3 proofs passed" in printed.out.splitlines()
    assert printed.out.splitlines()[-1] == "RESULT: failed"


@needs_python_3_11
def test_a_client_that_is_told_to_try_again_fails_the_run_and_both_logs_are_named(
    demo, files, capfd
):
    (files.top / "public_key").chmod(0o777)  # the middleware refuses such a configuration
    assert demo() == 1
    printed, logs = capfd.readouterr(), files.top / "work" / "logs"
    assert "result: outcome=try-again" in (logs / "ask_0.log").read_text().splitlines()
    assert printed.err.splitlines() == [
        "error: the client stored no reply to prompt 0; see %s and, for the server's side, %s"
        % (logs / "ask_0.log", logs / "vllm.log")
    ]
    assert printed.out.splitlines()[-1] == "RESULT: failed"
    assert sorted(os.listdir(logs)) == ["ask_0.log", "vllm.log"]  # no further prompt was sent


@needs_python_3_11
def test_another_users_listener_gets_neither_the_key_nor_a_prompt(demo, files, monkeypatch, capfd):
    monkeypatch.setattr(end_to_end, "listener_uid", lambda port: os.geteuid() + 1)
    assert demo() == 1
    assert "error: no socket of this user listens on 127.0.0.1:" in capfd.readouterr().err
    assert sorted(os.listdir(files.top / "work" / "logs")) == ["vllm.log"]  # no client has run


@needs_python_3_11
def test_no_prompt_is_sent_after_the_server_has_exited(demo, files, monkeypatch, capfd):
    run = end_to_end.Programs.run

    def run_and_stop_the_server(self, name, *args, **kwargs):
        code = run(self, name, *args, **kwargs)
        if name == "ask_0":
            self.stop_all()
        return code

    monkeypatch.setattr(end_to_end.Programs, "run", run_and_stop_the_server)
    assert demo() == 1
    assert "or vLLM has exited (see " in capfd.readouterr().err
    assert sorted(os.listdir(files.top / "work" / "logs")) == ["ask_0.log", "vllm.log"]


def test_a_hold_that_the_demo_lays_keeps_the_server_from_handing_out_a_proof(start_provider, files):
    """After end_to_end.put_on_hold() the real middleware hands out no proof, not even one that is
    ready: the client gets status 503 and is told to try again."""
    provider = start_provider()
    assert provider.client("--ask-only")[0] == STORED
    provider.prove()
    for _ in range(200):  # the check of its batch runs beside the calls
        if provider.job_state() == "proved":
            break
        time.sleep(0.1)
    assert provider.job_state() == "proved"

    class Arguments:  # all that put_on_hold() reads of the demo's arguments
        work_dir = str(files.top)

    end_to_end.put_on_hold(Arguments)
    assert (provider.spool_dir / "hold").exists()
    code, outcome, printed = provider.client("--resume")
    assert (code, outcome) == (ask_and_check.TRY_AGAIN, "outcome=try-again"), printed
    assert "status 503" in printed and proof_middleware.ON_HOLD in printed
