# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests for integrations/check_proof.py.

Most tests run or import a copy of the script in a small repository under tmp_path, in which
bin/cp_verify runs integrations/tests/fake_cp_verify.py. The fake checks nothing: these tests show
what the script does with a verifier's output, not that any proof is valid.
"""

import errno
import functools
import http.server
import importlib.util
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import check_proof
import fake_cp_verify
import pytest

HERE = Path(__file__).resolve().parent

PYTHON = [sys.executable, "-S", "-B"]
FAKE = '"%s"' % '" "'.join(PYTHON + [fake_cp_verify.__file__])
# What outcome() returns: the script's exit code and the words in its last line.
VERIFIED, REJECTED, ERROR = (0, "verified"), (1, "rejected"), (2, "error")
IN_LIMITS = (0, "within limits")
# W and NOT_W are exit codes of cp_verify, not of the script.
W = check_proof.STATUS_WITHIN_LIMITS  # with "WITHIN LIMITS: ..."
NOT_W = 86 if W == 0 else 0  # not a pass with that line
NOT_A_VERIFICATION = (14, "not a verification")
TOKENS = [10, 11, 12, 13, 14, 15]
OTHER_ID = "0" * 64
# The clipped-gap policy: a token's cost is its logit gap to the model's greedy choice plus 256. The
# costs, each clipped to C, must average at most theta, and at most cap tokens may cost more than C.
# s is the provider's statement of a gap's unit: 2^-s of a logit of the model.
# Below: lines of a parameter file (policy.txt), a caller's options that accept them, and
# cp_verify's last line for such a pass.
CLIPPED_GAP = {"C": 1000, "theta": 300, "cap": 1}
ACCEPT_CLIPPED_GAP = {"policy": "clipped-gap", "s": 15, **CLIPPED_GAP}
WITHIN_LIMITS = "WITHIN LIMITS: clipped-gap C=1000 theta=300 cap=1 s=15 sh=8"


def pack(*words):
    return struct.pack("<%dI" % len(words), *words)


# Returns a parameter file with exact match (every gap is 0). A keyword sets a line; None omits it.
def parameter_file(**changes):
    values = {"version": "H821", "record": 1, "L": 2, "MF": 44, "C": 256, "theta": 256, "cap": 0}
    values.update({"s": 15, "sh": 8, "sT": 4, "F": 24, **changes})
    lines = ["%s %s\n" % item for item in values.items() if item[1] is not None]
    return "".join(lines + ["work positions 0 0 0\n", "call R16a R16 G U\n"]).encode()


def write_program(path, *shell_lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["#!/bin/sh", *shell_lines, ""]))
    path.chmod(0o755)


class Client:
    """A small repository with the script and a fake verifier, and a good call's arguments.

    The methods' argument `fake` sets keys of the environment variable FAKE_CP_VERIFY, which change
    the fake's answers; fake_cp_verify.py lists them.
    """

    def __init__(self, top):
        self.top, self.tmp, self.log = top, top / "tmp", top / "calls"
        self.script = top / "repo" / "integrations" / "check_proof.py"
        self.verifier = top / "repo" / "bin" / "cp_verify"
        self.config, self.public, self.response = top / "config", top / "public", top / "response"
        self.request, self.leaf_file = top / "request.bin", self.public / "leaf" / "0.bin"
        for name in ("tmp", "repo/integrations", "config", "public/leaf", "response"):
            (top / name).mkdir(parents=True)
        shutil.copy(check_proof.__file__, self.script)
        write_program(
            self.verifier,
            "# The fake marks every line that it prints. The mark is taken off here, so that the",
            "# script reads the lines as cp_verify prints them. The exit status is the fake's.",
            "exec 3>&1",
            's=$({ { %s "$@" 3>&- 4>&-; echo $? >&4; } | LC_ALL=C "%s" "s/%s$//" >&3; } 4>&1)'
            % (FAKE, shutil.which("sed"), fake_cp_verify.MARK),  # a test may set another PATH
            'exit "$s"',
        )
        (self.config / "KL.bin").write_bytes(bytes(range(104)))
        (self.config / "layout.txt").write_text("1 1 1 M 4 0\n")
        (self.public / "key.bin").write_bytes(b"a key".ljust(56, b"k"))
        self.leaf_file.write_bytes(b"a leaf file".ljust(48, b"c"))
        (self.response / "proof.bin").write_bytes(b"CPPFH821" + bytes(4) + b"the rest of a proof")
        self.options = {
            "public_key_dir": self.public,
            "parameter_file": self.config / "policy.txt",
            "scales_file": self.config / "KL.bin",
            "end_token": 15,
        }
        self.set_parameters()
        self.set_request(TOKENS, 2, 5)

    def set_parameters(self, **changes):
        """Write the parameter file; later calls use its commitment ID."""
        data = parameter_file(**changes)
        (self.config / "policy.txt").write_bytes(data)
        self.id = fake_cp_verify.commitment_id(data, (self.public / "key.bin").read_bytes())
        self.options["commitment_id"] = self.id

    def set_request(self, tokens, start, end, next_token=None):
        """Write the request file and the response's token file; later calls use their values.

        The range start:end checks tokens start + 1 to end, counted from 0. Token len(tokens) is
        next_token, the one after the request.
        """
        self.request.write_bytes(pack(*tokens, start, end))
        (self.response / "y.bin").write_bytes(pack(next_token or 0))
        self.options.update(range="%d:%d" % (start, end), next_token=next_token)

    def words(self, *flags, **options):
        """Return a call's arguments. `options` override the defaults; None leaves an option out."""
        words = [str(self.response), str(self.request), *flags]
        for name, value in {**self.options, **options}.items():
            if value is not None:
                words += ["--" + name.replace("_", "-"), str(value)]
        return words

    def environment(self, fake=None):
        changes = json.dumps({"log": str(self.log), **(fake or {})})
        own = {"TMPDIR": str(self.tmp), "FAKE_CP_VERIFY": changes, "COLUMNS": "80"}
        return {**{k: v for k, v in os.environ.items() if k not in check_proof.API_KEYS}, **own}

    def run_script(self, words, fake=None, **more):
        more = {"env": self.environment(fake), "cwd": self.top, "timeout": 120, **more}
        words = [*PYTHON, str(self.script), *words]
        done = subprocess.run(words, capture_output=True, text=True, **more)
        assert os.listdir(self.tmp) == []  # no run leaves anything in TMPDIR
        assert "Traceback" not in done.stderr
        return done

    def run(self, *flags, fake=None, **options):
        return self.run_script(self.words(*flags, **options), fake)

    def printed(self, done):
        """Return a run's standard output without this directory's path, which can hold anything."""
        return done.stdout.replace(str(self.top), "").replace(os.path.realpath(self.top), "")

    def calls(self, plans=False):
        """Return the fake verifier's logged calls: the verification calls, or the "-plan" calls."""
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        calls = [json.loads(line) for line in lines]
        return [call for call in calls if (call["arguments"][0] == "-plan") == plans]


@pytest.fixture
def client(tmp_path):
    return Client(tmp_path)


def last_line(done):
    return done.stdout.splitlines()[-1]


def load_copy(client, monkeypatch):
    """Import the client's copy of the script, and set the client's environment in this process."""
    spec = importlib.util.spec_from_file_location("check_proof_copy", client.script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in client.environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", None)  # so that tempfile reads TMPDIR again
    return module


def outcome(done):
    """Return (exit code, the words in the last line), after checking the "result:" line."""
    line = last_line(done)
    words = [
        words
        for words in (*check_proof.LAST_LINE.values(), check_proof.WITHIN_LIMITS_WORDS)
        if line.startswith("check_proof: %s: " % words)
    ]
    assert len(words) == 1, line
    results = [line for line in done.stdout.splitlines() if line.startswith("result: ")]
    assert results == done.stdout.splitlines()[-2:-1]
    fields = check_proof.parse_result(results[0])
    expected = {check_proof.OUTCOMES[done.returncode]}
    if done.returncode == 0:  # the words agree with claim= and outcome=
        within = words[0] == check_proof.WITHIN_LIMITS_WORDS
        assert within == (fields["claim"] == "policy")
        if within:
            expected = {check_proof.WITHIN_LIMITS_OUTCOME}
    if done.returncode == 1:
        expected.add(check_proof.NO_PROOF_OUTCOME)
    assert fields["outcome"] in expected
    assert (len(fields) > 1) == (done.returncode == 0)  # only a pass has further fields
    return done.returncode, words[0]


def result(done):
    (line,) = [line for line in done.stdout.splitlines() if line.startswith("result: ")]
    return check_proof.parse_result(line)


def names(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


# ---- a proof that verifies ----


def test_verifies_a_good_proof(client):
    done = client.run()
    assert outcome(done) == VERIFIED and done.stderr == ""
    assert "exact match" in last_line(done) and "end token 15" in last_line(done)
    expected = (
        "result: outcome=verified commitment_id=%s range=2:5 checked_tokens=3 end_token=15 "
        "claim=exact-match policy=exact next_token=none C=256 theta=256 cap=0 s=15 limits=exact "
        "request_tokens=6 leaf=0 commitment_assumed_rounds=49 record=1 work=0 least_round=128.02 "
        "least_round_without_work=128.02 build=%s"
    )
    assert done.stdout.splitlines()[-2] == expected % (client.id, fake_cp_verify.BUILD)
    assert done.stdout.splitlines()[0] == "verifier: %s" % client.verifier


def test_verifier_gets_the_callers_commitment_id_and_private_copies(client):
    (client.public / "leaf" / "1.bin").write_bytes(b"another leaf file")
    client.run(threads=2)
    (call,) = client.calls()
    files = ["config", "ivk", "request.bin", "y.bin", "proof.bin"]
    assert call["arguments"] == files + ["-id", client.id, "-bits", "128", "-j", "2", "-q"]
    assert Path(call["directory"]).parent == client.tmp
    assert call["files"] == {
        "config/policy.txt": parameter_file().hex(),
        "config/KL.bin": (client.config / "KL.bin").read_bytes().hex(),
        "config/layout.txt": (client.config / "layout.txt").read_bytes().hex(),
        "request.bin": client.request.read_bytes().hex(),
        "y.bin": bytes(4).hex(),
        "proof.bin": (client.response / "proof.bin").read_bytes().hex(),
        "ivk/key.bin": "symlink",
        "ivk/leaf/0.bin": "symlink",
    }


def test_checks_the_next_token_when_the_range_reaches_the_end(client):
    client.set_request(TOKENS, 2, 6, next_token=7)
    done = client.run(end_token="7,8")
    assert outcome(done) == VERIFIED and "next token, 7," in last_line(done)
    assert (result(done)["next_token"], result(done)["end_token"]) == ("7", "7")
    assert client.calls()[0]["files"]["y.bin"] == pack(7).hex()


def test_reply_lengths_give_the_same_range(client):
    assert result(client.run(range=None, reply="3:3"))["range"] == "2:5"
    assert outcome(client.run(range=None, reply="3:2")) == REJECTED


def test_response_can_be_named_by_its_proof_file(client):
    shutil.copy(client.response / "proof.bin", client.top / "abc.proof")
    shutil.copy(client.response / "y.bin", client.top / "abc.y")
    words = [str(client.top / "abc.proof")] + client.words()[1:]
    assert outcome(client.run_script(words)) == VERIFIED


def test_accepts_relative_paths(client):
    options = client.words(
        public_key_dir="public",
        parameter_file="config/policy.txt",
        scales_file="config/KL.bin",
        verifier="repo/bin/cp_verify",
    )[2:]
    assert outcome(client.run_script(["response", "request.bin"] + options)) == VERIFIED


def test_verifier_is_not_searched_for_in_path(client):
    env = {**client.environment(), "PATH": "%s:%s" % (client.verifier.parent, os.environ["PATH"])}
    done = client.run_script(client.words(verifier="cp_verify"), env=env)
    assert is_short_usage_error(client, done) and "--verifier needs a path" in done.stderr


def shadow(directory, name="argparse"):
    """Write a module that fakes a pass if it is imported as `name`."""
    text = 'print("result: outcome=verified\\ncheck_proof: verified: fake")\nraise SystemExit(0)\n'
    (directory / (name + ".py")).write_text(text)


@pytest.mark.parametrize("flags", [[], ["-B"], ["-S", "-B"], ["-I"]])
def test_shadow_modules_on_pythonpath_or_beside_the_script_are_not_imported(client, flags):
    (client.top / "elsewhere").mkdir()
    for directory in (client.top / "elsewhere", client.script.parent, client.top):
        shadow(directory)
    env = {**client.environment(), "PYTHONPATH": str(client.top / "elsewhere")}
    words = [sys.executable, *flags, str(client.script), *client.words(range="2:4")]
    done = subprocess.run(words, capture_output=True, text=True, env=env, cwd=client.top)
    assert outcome(done) == REJECTED and "fake" not in client.printed(done) and done.stderr == ""


def test_nothing_on_pythonpath_is_imported_before_the_restart(client):
    """With -S, Python up to 3.10 has not loaded os when the script starts."""
    shadow(client.top, "os")
    env = {**client.environment(), "PYTHONPATH": str(client.top)}
    words = [sys.executable, "-S", "-B", str(client.script), "--help"]
    done = subprocess.run(words, capture_output=True, text=True, env=env, cwd=client.tmp)
    assert done.returncode == 0 and done.stdout.startswith("usage: ") and "fake" not in done.stdout


def test_restart_in_isolated_mode_keeps_arguments_and_flags(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    calls = []
    monkeypatch.setattr(module.posix, "execv", lambda *call: calls.append(call))
    monkeypatch.setattr(module.sys, "argv", ["x", "a b", "--c"])
    module.isolate()
    kept = ["-B"] * sys.flags.dont_write_bytecode + ["-S"] * sys.flags.no_site
    words = [sys.executable, "-I", *kept, str(client.script), "a b", "--c"]
    assert calls == ([] if sys.flags.isolated else [(sys.executable, words)])


@pytest.mark.parametrize("failure", [OSError(errno.ENOENT, "No such file"), ValueError("empty")])
def test_failed_restart_is_an_error_and_no_rejection(client, monkeypatch, capsys, failure):
    module = load_copy(client, monkeypatch)

    def fail(*_):
        raise failure

    monkeypatch.setattr(module.posix, "execv", fail)
    if sys.flags.isolated:
        pytest.skip("pytest itself runs in isolated mode")
    with pytest.raises(SystemExit) as ended:
        module.isolate()
    assert ended.value.code == module.EXIT_ERROR
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "result: outcome=error" and lines[1].startswith("check_proof: error: ")


def test_end_token_none_skips_the_end_token_rule(client):
    done = client.run(end_token="none")
    assert outcome(done) == VERIFIED and "not checked" in last_line(done)
    assert result(done)["end_token"] == "not_checked"


# ---- rejected before the verifier runs ----


def rejected_without_a_call(client, done, *texts):
    assert all(text in last_line(done) for text in texts)
    return outcome(done) == REJECTED and client.calls() == []


def test_rejects_wrong_commitment_id(client):
    done = client.run(commitment_id=OTHER_ID)
    assert rejected_without_a_call(client, done, "a commitment ID other than the one you gave")
    assert client.id not in done.stdout + done.stderr  # the files' ID is not shown


def test_rejects_another_range(client):
    assert rejected_without_a_call(client, client.run(range="2:4"), "2:5", "2:4")


def test_rejects_another_next_token(client):
    client.set_request(TOKENS, 2, 6, next_token=7)
    done = client.run(next_token=8, end_token="none")
    assert rejected_without_a_call(client, done, "another next token than the 8 that you expect")
    assert "7" not in last_line(done)  # the response's token is not shown


@pytest.mark.parametrize("end, next_token", [(5, 7), (6, None), (6, "none")])
def test_next_token_is_given_exactly_when_the_range_reaches_the_end(client, end, next_token):
    client.set_request(TOKENS, 2, end)
    done = client.run(next_token=next_token, end_token="none")
    assert outcome(done) == ERROR and client.calls() == []


def test_rejects_a_last_token_that_is_not_an_end_token(client):
    assert rejected_without_a_call(client, client.run(end_token="14,16"), "15")


def test_rejects_an_early_end_token(client):
    assert rejected_without_a_call(client, client.run(end_token="15,13"), "token 3")


def test_allows_an_end_token_in_the_prompt(client):
    assert outcome(client.run(end_token="15,12")) == VERIFIED  # token 2 ends the prompt


@pytest.mark.parametrize("header", [b"CPPGH821", b"CPPFH822", b"H821CPPF", b"CPPF"])
def test_rejects_a_proof_file_with_another_header(client, header):
    (client.response / "proof.bin").write_bytes(header + bytes(20))
    assert rejected_without_a_call(client, client.run(), "header")


def test_parameter_file_without_a_version_is_rejected(client):
    client.set_parameters(version=None)
    assert rejected_without_a_call(client, client.run(), "has no line version")


@pytest.mark.parametrize(
    "parameters, options, expected",
    [
        ({}, {}, VERIFIED),
        ({}, ACCEPT_CLIPPED_GAP, VERIFIED),  # exact match is within every clipped-gap limit
        (CLIPPED_GAP, {}, REJECTED),
        (CLIPPED_GAP, ACCEPT_CLIPPED_GAP, IN_LIMITS),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "theta": 301, "cap": 2}, IN_LIMITS),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "theta": 299}, REJECTED),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "theta": None}, REJECTED),  # the default is 256
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "cap": 0}, REJECTED),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "cap": None}, REJECTED),  # the default is 0
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "C": 999}, REJECTED),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "C": 1001}, REJECTED),
        (CLIPPED_GAP, {**ACCEPT_CLIPPED_GAP, "s": 14}, REJECTED),
        ({"C": 1000}, {}, REJECTED),  # allows only zero gaps, but is not cp_verify's exact match
        ({"sh": 9}, {}, REJECTED),
        ({"sT": 15}, {}, REJECTED),
        ({"F": 23}, {}, REJECTED),
        ({"cap": None}, {}, REJECTED),  # no such line
    ],
)
def test_compares_the_policy_with_the_callers_limits(client, parameters, options, expected):
    client.set_parameters(**parameters)
    assert outcome(client.run(**options)) == expected
    assert len(client.calls()) == (expected in (VERIFIED, IN_LIMITS))


def test_rejection_by_the_limits_shows_none_of_the_files_numbers(client):
    client.set_parameters(C=1001, theta=301, cap=2, s=14)
    done = client.run(**ACCEPT_CLIPPED_GAP)
    assert outcome(done) == REJECTED and client.calls() == []
    faults = "(C differs; theta is above your limit; cap is above your limit; s differs): C=1000 "
    assert faults in last_line(done) and "never take limits from the file" in last_line(done)
    assert not any(number in last_line(done) for number in ("1001", "301", "14"))
    client.set_parameters(C=1000, theta=301, cap=1)
    assert "accept (theta is above your limit): " in last_line(client.run(**ACCEPT_CLIPPED_GAP))


@pytest.mark.parametrize(
    "parameters, options",
    [({}, {}), (CLIPPED_GAP, ACCEPT_CLIPPED_GAP), ({}, {"range": "5:6", "next_token": 7})],
)
def test_parameter_file_with_a_line_temp_is_rejected(client, parameters, options):
    if "range" in options:
        client.set_request(TOKENS, 5, 6, next_token=7)
    client.set_parameters(temp=256, **parameters)
    done = client.run(end_token="none", **options)
    assert outcome(done) == REJECTED and client.calls() == []
    assert 'has a line "temp": this script does not support that' in last_line(done)


@pytest.mark.parametrize(
    "parameters, fault", [({"cap": None}, "has no line cap"), ({"sT": 5}, "has sh, sT or F other")]
)
def test_unsupported_policy_lines_are_named(client, parameters, fault):
    client.set_parameters(**parameters)
    done = client.run()
    assert outcome(done) == REJECTED and fault in last_line(done) and client.calls() == []


def test_clipped_gap_file_without_the_callers_policy_is_rejected(client):
    client.set_parameters(**CLIPPED_GAP)
    done = client.run()
    assert outcome(done) == REJECTED and "you asked for exact match" in last_line(done)
    assert not any(c.isdigit() for c in last_line(done))  # none of the file's numbers to copy
    assert client.calls() == []


def test_files_with_another_commitment_id_are_rejected_and_their_id_is_not_shown(client):
    done = client.run(fake={"plan_id": OTHER_ID})
    assert rejected_without_a_call(client, done, "a commitment ID other than the one you gave")
    assert OTHER_ID not in done.stdout + done.stderr


# cp_verify's planning call "-plan id" exits with 4 and prints one commitment ID.
@pytest.mark.parametrize(
    "fake",
    [
        {"plan_status": 7},
        {"plan_status": 0},
        {"plan_second_id": True},
    ],
    ids=str,
)
def test_planning_call_that_fails_or_prints_two_ids_is_an_error(client, fake):
    done = client.run(fake=fake)
    assert outcome(done) == ERROR and "cp_verify -plan id exited with status" in last_line(done)
    assert client.calls() == []


def test_verifier_killed_by_a_signal_is_an_error(client):
    write_program(
        client.top / "killed",
        'case "$1" in -plan) exec "%s" "$@";; esac' % client.verifier,
        "kill -9 $$",
    )
    done = client.run(verifier=client.top / "killed")
    assert outcome(done) == ERROR and "was killed by signal 9" in last_line(done)


def test_verifier_gets_the_callers_environment_without_the_api_keys(client):
    keys = {check_proof.FETCH_KEY: "k-1", "VLLM_API_KEY": "k-2", "SOMETHING_ELSE": "kept"}
    done = client.run_script(client.words(), env={**client.environment(), **keys})
    calls = client.calls(plans=True) + client.calls()
    assert outcome(done) == VERIFIED and len(calls) == 2
    for call in calls:
        assert "SOMETHING_ELSE" in call["environment"]
        assert not set(call["environment"]) & {check_proof.FETCH_KEY, "VLLM_API_KEY"}


@pytest.mark.parametrize(
    "option, path",
    [
        ("public_key_dir", "public"),
        ("parameter_file", "config/policy.txt"),
        ("scales_file", "config/KL.bin"),
        ("layout_file", "config/layout.txt"),
        ("verifier", "repo/bin/cp_verify"),
    ],
)
def test_only_fetch_from_can_be_a_url(client, option, path):
    """Elsewhere, what looks like a URL is a relative path like any other."""
    (client.top / "https:" / "provider.example").mkdir(parents=True)
    (client.top / "https:" / "provider.example" / "x").symlink_to(client.top / path)
    layout = {} if option == "layout_file" else {"layout_file": client.config / "layout.txt"}
    done = client.run(**{option: "https://provider.example/x"}, **layout)
    assert outcome(done) == VERIFIED


def test_missing_layout_file_is_an_error(client):
    (client.config / "layout.txt").unlink()
    done = client.run()
    assert outcome(done) == ERROR and "layout.txt does not exist" in last_line(done)
    assert client.calls(plans=True) == []


@pytest.mark.parametrize("name", ["proof.bin", "y.bin"])
def test_symbolic_link_in_the_response_is_an_error(client, name):
    (client.response / name).rename(client.top / name)
    (client.response / name).symlink_to(client.top / name)
    done = client.run()
    assert outcome(done) == ERROR and "a symbolic link does not count" in last_line(done)
    assert client.calls() == []


@pytest.mark.parametrize("name", ["proof.bin", "y.bin"])
def test_response_file_that_is_no_regular_file_is_an_error(client, name):
    (client.response / name).unlink()
    os.mkfifo(client.response / name)  # no process writes to it: the check must not wait
    done = client.run()
    assert outcome(done) == ERROR and "is not a regular file" in last_line(done)
    assert client.calls() == []


def test_proof_that_is_too_long_is_an_error(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    monkeypatch.setattr(module, "MAX_FETCHED_BYTES", 20)
    assert module.main(client.words()) == module.EXIT_ERROR and client.calls() == []
    assert "proof.bin: longer than the 20 bytes" in capsys.readouterr().out.splitlines()[-1]
    assert os.listdir(client.tmp) == []


def test_commitment_id_can_be_typed_in_capitals(client):
    done = client.run(commitment_id=client.id.upper())
    assert outcome(done) == VERIFIED and result(done)["commitment_id"] == client.id


def test_exact_match_can_be_requested_explicitly(client):
    assert outcome(client.run(policy="exact")) == VERIFIED
    client.set_parameters(**CLIPPED_GAP)
    assert outcome(client.run(policy="exact")) == REJECTED
    assert len(client.calls()) == 1  # the second run stopped before the verifier ran


def test_exact_match_accepts_no_limit_options(client):
    assert is_short_usage_error(client, client.run(policy="exact", C=1000))


def test_verifier_gets_no_policy_that_the_caller_did_not_request(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    # As if an edit had removed the check.
    monkeypatch.setattr(module, "check_limits", lambda *_: None)
    client.set_parameters(**CLIPPED_GAP)
    assert module.main(client.words()) == module.EXIT_REJECTED
    assert "cp_verify printed NO: policy" in capsys.readouterr().out.splitlines()[-1]
    assert "-policy" not in client.calls()[0]["arguments"]


def test_key_file_rejected_by_the_plan_call(client):
    done = client.run(fake={"plan_rejects": "NO: key-head"})
    assert outcome(done) == REJECTED and client.calls() == []
    assert "cp_verify rejected key.bin or a configuration file: NO: key-head" in last_line(done)


def test_names_and_reports_a_clipped_gap_policy(client):
    client.set_parameters(**CLIPPED_GAP)
    done = client.run(**{**ACCEPT_CLIPPED_GAP, "theta": 400})
    assert client.calls()[0]["arguments"][-2:] == ["-policy", "clipped-gap,1000,300,1,15,8"]
    assert (result(done)["policy"], result(done)["theta"]) == ("clipped-gap", "300")
    assert result(done)["claim"] == "policy" and done.returncode == 0
    assert result(done)["limits"] == "C:1000,theta:400,cap:1,s:15"
    assert "clipped to 744, add up to at most 132" in last_line(done)  # 1000 - 256; 3 * (300 - 256)
    assert "(C and theta count costs: a cost is a gap plus 256)" in last_line(done)


@pytest.mark.parametrize(
    "parameters, text",
    [
        ({"C": 1000}, "add up to at most 0,"),  # theta = 256 forces every gap to zero
        ({"cap": 1}, "up to 1 of the checked tokens"),
    ],
)
def test_describes_what_a_clipped_gap_policy_shows(client, parameters, text):
    client.set_parameters(**parameters)
    done = client.run(**{"policy": "clipped-gap", "C": 256, "s": 15, **parameters})
    assert outcome(done) == IN_LIMITS and text in last_line(done)
    assert result(done)["policy"] == "clipped-gap"


def test_reports_whether_the_end_token_is_shown_to_be_the_greedy_choice(client):
    doubt = "does not show that it is the model's greedy choice"
    assert doubt not in last_line(client.run())
    client.set_parameters(C=1000)
    assert doubt not in last_line(client.run(policy="clipped-gap", C=1000, s=15))
    client.set_parameters(**CLIPPED_GAP)
    assert doubt in last_line(client.run(**ACCEPT_CLIPPED_GAP))


def test_compares_no_limits_when_only_the_next_token_is_checked(client):
    client.set_parameters(**CLIPPED_GAP)
    client.set_request(TOKENS, 5, 6, next_token=7)
    done = client.run(end_token="none")
    assert outcome(done) == VERIFIED and "-policy" not in client.calls()[0]["arguments"]
    assert (result(done)["policy"], result(done)["limits"]) == ("none", "not_compared")


@pytest.mark.parametrize("alone", [False, True])
def test_no_proof_is_a_rejection(client, alone):
    (client.response / "failed").write_text("")
    if alone:
        (client.response / "proof.bin").unlink()
        (client.response / "y.bin").unlink()
    done = client.run()
    assert outcome(done) == REJECTED and "made no proof" in last_line(done) and client.calls() == []
    assert result(done) == {"outcome": "no-proof"}  # an invalid proof gives "rejected"


@pytest.mark.parametrize(
    "fake, word",
    [
        ({"last_line": "NO: batch-1", "status": 1}, "rejected"),
        ({"last_line": "NOT-A-VERIFICATION", "status": 14}, "not-a-verification"),
        ({"last_line": "YES", "status": 7}, "error"),
    ],
)
def test_failed_check_prints_only_its_outcome(client, fake, word):
    done = client.run(fake=fake)
    outcome(done)
    assert done.stdout.splitlines()[-2] == "result: outcome=" + word


def test_nothing_follows_a_pass_whose_lines_cannot_be_printed(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    printed = []

    def print_and_fail_at_the_result(*words, **more):
        printed.append(" ".join(words))
        if words and words[0].startswith("result: "):
            raise OSError("no space left on device")

    monkeypatch.setattr(module, "print", print_and_fail_at_the_result, raising=False)
    monkeypatch.setattr(module.os, "dup2", lambda *_: None)  # do not touch pytest's output
    assert module.main(client.words()) == module.EXIT_ERROR
    assert printed[-1].startswith("result: outcome=verified ")
    assert sum(text.startswith("result: ") for text in printed) == 1


@pytest.mark.parametrize(
    "url, this_machine",
    [
        ("http://127.0.0.1:8000/x", True),
        ("http://127.0.0.1", True),
        ("HTTP://[::1]:8000", True),
        ("https://127.0.0.1:8443/a%20b?c@d:e#f", True),
        ("http://localhost", False),  # may resolve to another user's listener
        ("http://localhost:8000/x", False),
        ("http://LOCALHOST:1/", False),
        ("http://user@localhost:1/", False),
        ("http://u@localhost", False),
        ("http://localhost:x.evil.example:80", False),
        ("HTTP://localhost:x.evil.example:80", False),
        ("http://127.0.0.1:.evil.example:8000", False),
        ("http://[::1]:x.evil.example:80", False),
        ("http://evil.example%2F@127.0.0.1", False),
        ("http://localhost:80:81", False),
        ("http://localhost:123456", False),
        ("http://localhost?x", False),
        ("http://localhost#x", False),
        ("http://localhost.", False),
        ("http://127.1", False),
        ("http://localhost:80\n", False),
        ("http://127.0.0.1.example.org", False),
        ("http://localhost.example.org", False),
        ("http://files.example/127.0.0.1", False),
        ("http://127.0.0.1@files.example/", False),
        ("http://files.example\\@127.0.0.1/", False),
        ("http://[", False),
        ("http://127.0.0.1 /", False),
        ("http://127x0x0x1", False),
        ("http://localho\u017ft", False),  # U+017F, which IDNA turns into "s"
        ("http\u017f://127.0.0.1", False),
    ],
)
def test_is_this_machine(url, this_machine):
    assert check_proof.is_this_machine(url) == this_machine


def test_fetch_itself_sends_no_key_over_plain_http_to_another_machine(tmp_path, monkeypatch):
    monkeypatch.setenv(check_proof.FETCH_KEY, "k-1")
    monkeypatch.delattr(socket, "socket")  # if the rule fails, the test must not reach the network
    monkeypatch.delattr(socket, "getaddrinfo")
    for source in (
        "http://files.example",
        "http://localhost:x.evil.example:80",
        "http://localhost",
    ):
        with pytest.raises(OSError, match="plain http may go only to 127.0.0.1 or"):
            check_proof.fetch(source, "key.bin", 10, tmp_path / "key.bin")


def test_pass_names_its_outcome_first(client):
    assert (
        client.run().stdout.splitlines()[-2].startswith("result: outcome=verified commitment_id=")
    )
    assert "bits" not in result(client.run())


@pytest.mark.parametrize("change", ["no proof", "no token", "long token"])
def test_incomplete_response_is_an_error(client, change):
    if change == "long token":
        (client.response / "y.bin").write_bytes(bytes(8))
    else:
        (client.response / {"no proof": "proof.bin", "no token": "y.bin"}[change]).unlink()
    assert outcome(client.run()) == ERROR and client.calls() == []


@pytest.mark.parametrize("data", [b"", pack(1, 2), bytes(14), pack(1, 2, 1, 1), pack(1, 2, 0, 3)])
def test_malformed_request_file_is_an_error(client, data):
    client.request.write_bytes(data)
    assert outcome(client.run()) == ERROR and client.calls() == []


# ---- what is compared after the verifier passes ----


# The start and end of cp_verify's "result-v1" line, which test names call the line for programs.
LINE = "result-v1 status=0"
END = "dw=8 build=%s\n" % fake_cp_verify.BUILD


@pytest.mark.parametrize(
    "fake",
    [
        {"replace": [[LINE, "result-v2 status=0"]]},  # no such line
        {"replace": [[LINE, " " + LINE]]},  # not at a line's start
        {"replace": [[LINE, "result-v1 x\n" + LINE]]},  # two lines
        {"copy_result_v1": True},  # the same line twice
        {"replace": [["---- the call ----\n", ""], ["\nYES", "\n---- the call ----"]]},
        {"replace": [[LINE, LINE + " status=0"]]},
        {"replace": [[END, END + "more\n"]]},  # not directly before the last line
        {"replace": [[END, END.rstrip() + " more\n"]]},
        {"replace": [[END, END.rstrip() + "\r\n"]]},
        {"replace": [[" stages=key,leaf,request", ""]]},
        {"replace": [[" r=2 e=5 T=6 tokens", " e=5 r=2 T=6 tokens"]]},
    ],
    ids=str,
)
def test_a_yes_with_unexpected_output_is_an_error(client, fake):
    done = client.run(fake=fake)
    assert outcome(done) == ERROR and "status 0" in last_line(done)


FIELDS = (
    "status claim id bits work work-rounds least-round least-round-without-work "
    "commitment-assumed-rounds r e T tokens y policy C theta cap s sh sT F above-clip leaf stages "
    "dw build"
)


@pytest.mark.parametrize("field", FIELDS.split())
def test_each_field_of_the_line_for_programs_is_compared(client, field):
    done = client.run(fake={"result_v1": {field: "9"}})
    assert outcome(done) == ERROR and "status 0" in last_line(done)


@pytest.mark.parametrize("word", ["policy", "next-token", "exact", "exact-match-"])
def test_another_claim_in_the_line_for_programs_is_an_error(client, word):
    assert outcome(client.run(fake={"result_v1": {"claim": word}})) == ERROR


def test_the_other_lines_of_a_yes_are_not_compared(client):
    changes = [["STAGE PASSED: key", "stage key: passed"], ["covered: r=2", "covered: r=3"]]
    assert outcome(client.run(fake={"replace": changes})) == VERIFIED


@pytest.mark.parametrize(
    "limits",
    [
        {"C": 255, "theta": 255},
        {"C": 1000, "theta": 255},
        {"C": 1000, "theta": 1001},
        {"C": 1000, "theta": 300, "cap": 7},  # the request has 6 tokens
        {"C": 1000, "theta": 1000, "cap": 3},  # 3 tokens are checked
        {"C": 256, "cap": 3},
    ],
    ids=str,
)
def test_caller_limits_that_say_nothing_are_an_error(client, limits):
    done = client.run(policy="clipped-gap", s=15, **limits)  # even if the file has exact match
    assert outcome(done) == ERROR and "cannot be used: limits need " in last_line(done)
    assert client.calls() == []


def test_limits_when_only_the_next_token_is_checked(client):
    client.set_request(TOKENS, 5, 6, next_token=7)
    options = {"policy": "clipped-gap", "s": 15, "end_token": "none"}
    done = client.run(C=256, **options)  # theta = C, cap = 0: accepted
    assert outcome(done) == VERIFIED and result(done)["claim"] == "next-token"
    done = client.run(C=1000, theta=1001, **options)
    assert outcome(done) == ERROR and "cap <= 6" in last_line(done)
    assert "number of checked tokens" not in last_line(done)


def test_widest_limits_that_still_constrain_are_accepted(client):
    assert outcome(client.run(policy="clipped-gap", s=15, C=1000, theta=1000, cap=2)) == VERIFIED
    assert outcome(client.run(policy="clipped-gap", s=15, C=1000, theta=999, cap=6)) == VERIFIED


# ---- the verifier's possible results ----


@pytest.mark.parametrize(
    "line, status, expected",
    [
        ("YES", 0, VERIFIED),
        ("NO: batch-1", 1, REJECTED),
        ("NOT-A-VERIFICATION", 14, NOT_A_VERIFICATION),
        ("YES", 1, ERROR),
        ("YES", 14, ERROR),
        ("YES", 86, ERROR),
        ("YES ", 0, ERROR),
        ("YES: all good", 0, ERROR),
        ("", 0, ERROR),
        ("NO: batch-1", 0, ERROR),
        ("NO: batch-1", 14, ERROR),
        ("NO:", 1, ERROR),
        ("NOT-A-VERIFICATION", 0, ERROR),
        ("NOT-A-VERIFICATION", 1, ERROR),
        ("YES-WITH-WORK: 96.02 + 32", 10, ERROR),
        (
            "WITHIN LIMITS: clipped-gap C=256 theta=256 cap=0 s=15 sh=8",
            W,
            ERROR,
        ),  # no named policy
    ],
)
def test_a_status_counts_only_with_its_last_line(client, line, status, expected):
    assert outcome(client.run(fake={"last_line": line, "status": status})) == expected


@pytest.mark.parametrize(
    "line, status, expected",
    [
        (WITHIN_LIMITS, W, IN_LIMITS),
        (WITHIN_LIMITS.replace("theta=300", "theta=299"), W, ERROR),
        (WITHIN_LIMITS, NOT_W, ERROR),
        ("YES", 0, ERROR),
        ("YES", 86, ERROR),
        ("YES-WITH-WORK: 96.02 + 32", 10, ERROR),
    ],
)
def test_named_policy_passes_only_with_its_line_and_status(client, line, status, expected):
    client.set_parameters(**CLIPPED_GAP)
    done = client.run(fake={"last_line": line, "status": status}, **ACCEPT_CLIPPED_GAP)
    assert outcome(done) == expected


def test_fake_verifier_uses_the_status_that_the_script_expects():
    assert fake_cp_verify.STATUS_WITHIN_LIMITS == W


@pytest.mark.parametrize(
    "line, status",
    [
        ("input-unreadable-proof: cannot read proof.bin: No such file or directory", 2),
        ("PART PASSED: key", 3),
        ("PLAN ONLY: no proof was checked", 4),
        ("NOT CHECKED: internal-result-line", 5),
    ],
)
def test_other_results_of_cp_verify_are_errors(client, line, status):
    assert outcome(client.run(fake={"last_line": line, "status": status})) == ERROR


@pytest.mark.parametrize("line", ["YES: x", "YES (almost)"])
def test_nothing_is_shown_of_a_wrong_last_line_at_the_passing_status(client, line):
    done = client.run(fake={"last_line": line, "status": 0})
    assert outcome(done) == ERROR and "cp_verify: " not in done.stdout
    assert "its last line does not go with this call" in last_line(done)


def test_cp_verifys_lines_show_no_other_commitment_id_unless_its_run_is_a_pass(client):
    shown = {"replace": [["id: named by the caller and equal", "id: " + OTHER_ID]]}
    shown["stderr"] = "a warning about " + OTHER_ID
    done = client.run(fake=shown)
    assert "cp_verify: id: " + OTHER_ID in done.stdout and OTHER_ID in done.stderr
    for ending in (
        {"last_line": "NOT CHECKED: internal-stage", "status": 5},
        {"last_line": "NOT-A-VERIFICATION", "status": 14},
    ):
        done = client.run(fake={**shown, **ending})
        assert done.returncode != 0 and OTHER_ID not in done.stdout + done.stderr
        assert "cp_verify: id: " + check_proof.HIDDEN_ID in done.stdout
        assert "cp_verify: commitment ID: " + client.id in done.stdout
        assert done.stderr == "a warning about %s\n" % check_proof.HIDDEN_ID
    shown["replace"] = [["refused by the stage request", "refused: " + OTHER_ID]]
    for line in ("NO: batch-1", "NO: " + OTHER_ID):
        done = client.run(fake={**shown, "last_line": line, "status": 1})
        assert outcome(done) == REJECTED and OTHER_ID not in done.stdout + done.stderr
        assert "cp_verify: refused: " + check_proof.HIDDEN_ID in done.stdout
        if line.endswith(OTHER_ID):
            assert last_line(done).endswith("cp_verify printed NO: " + check_proof.HIDDEN_ID)
    done = client.run(fake={"plan_rejects": "NO: " + OTHER_ID})
    assert outcome(done) == REJECTED and OTHER_ID not in done.stdout
    assert last_line(done).endswith("NO: " + check_proof.HIDDEN_ID)


@pytest.mark.parametrize(
    "change, code",
    [
        ({"result_v1": {"r": "3"}}, 2),
        ({"result_v1": {"y": "yes"}}, 2),
        ({"test_build": True}, 14),
        ({"status": 15}, 2),  # a pass's full output, but 15 means that cp_verify could not write it
        ({"status": 7}, 2),
    ],
)
def test_nothing_is_shown_of_a_pass_that_the_script_does_not_accept(client, change, code):
    shown = {"replace": [["id: named by the caller and equal", "id: " + OTHER_ID]]}
    done = client.run(fake={**shown, **change, "stderr": "a warning about " + OTHER_ID})
    assert done.returncode == code and OTHER_ID not in done.stdout + done.stderr
    assert "cp_verify: " not in done.stdout and "YES" not in client.printed(done)


def test_silent_verifier_is_an_error(client):
    write_program(client.top / "silent", "exit 0")
    assert outcome(client.run(verifier=client.top / "silent")) == ERROR


@pytest.mark.parametrize(
    "ending",
    [
        {"last_line": "NOT-A-VERIFICATION", "status": 14},
        {"last_line": "NOT CHECKED: internal-stage", "status": 5},
    ],
    ids=str,
)
def test_no_failure_shows_the_files_policy_numbers(client, ending):
    client.set_request(TOKENS, 5, 6, next_token=7)  # only the next token: no limits are compared
    client.set_parameters(C=5000, theta=4000, cap=3)
    done = client.run(end_token="none", fake=ending)
    assert done.returncode != 0 and "cp_verify: id: named by the caller and equal" in done.stdout
    assert not any(number in client.printed(done) for number in ("5000", "4000", "cap=3"))


@pytest.mark.parametrize("why", ["", ": its link does not fit the key"])
def test_rejection_by_the_stage_leaf_blames_the_leaf_file(client, why):
    fake = {"last_line": "NO: entry-refused", "status": 1}
    assert "leaf file" not in last_line(client.run(fake=fake))
    fake["replace"] = [["refused by the stage request", "refused by the stage leaf" + why]]
    done = client.run(fake=fake)
    assert outcome(done) == REJECTED
    assert "NO: entry-refused: it rejected the leaf file leaf/0.bin, not the proof" in last_line(
        done
    )


@pytest.mark.parametrize("status", [15, 7, 14])
def test_nothing_is_shown_of_an_output_that_is_cut_short(client, status):
    cut = {"replace": [["proof of work: none counted\n", "proof of work: none counted\n\x00"]]}
    done = client.run("--show-all", fake={**cut, "cut_at": "\x00", "status": status})
    assert outcome(done) == ERROR and "cp_verify: " not in done.stdout


def test_line_on_a_file_that_cp_verify_cannot_read_is_shown(client):
    line = "input-unreadable-proof: the proof file cannot be read: Permission denied"
    (client.top / "output").write_text(line + "\n")
    done = client.run(fake={"output": str(client.top / "output"), "status": 2})
    assert outcome(done) == ERROR and "cp_verify: " + line in done.stdout.splitlines()


def test_pass_tells_that_lines_were_left_out_and_how_to_see_them(client):
    done = client.run()
    assert outcome(done) == VERIFIED and "cp_verify: covered: r=2 e=5 " in done.stdout
    assert "accounting:" not in done.stdout and "result-v1" not in done.stdout
    note = "cp_verify: (2 more lines are not shown: the verifier's own account of what its figures"
    assert done.stdout.splitlines()[-3].startswith(note)
    end = " with its caveats, and its line for programs; to see all, pass --show-all)"
    assert done.stdout.splitlines()[-3].endswith(end)
    done = client.run("--show-all")
    assert outcome(done) == VERIFIED and "not shown" not in done.stdout
    assert "cp_verify: accounting: the largest round error" in done.stdout
    assert "cp_verify: result-v1 status=0 " in done.stdout and "cp_verify: YES" not in done.stdout


def test_show_all_shows_no_more_of_a_run_that_did_not_pass(client):
    client.set_parameters(**CLIPPED_GAP)
    fake = {"last_line": "NOT-A-VERIFICATION", "status": 14}
    done = client.run(fake=fake, **ACCEPT_CLIPPED_GAP)
    note = "cp_verify: (1 more line is not shown: the verifier's own account of what its figures "
    assert (
        done.stdout.splitlines()[-3]
        == note + "rest on, with its caveats; to see all, pass --show-all)"
    )
    done = client.run("--show-all", fake=fake, **ACCEPT_CLIPPED_GAP)
    assert outcome(done) == NOT_A_VERIFICATION and "cp_verify: accounting: " in done.stdout
    assert "theta=" not in client.printed(done) and "result-v1" not in done.stdout
    assert "not shown" not in done.stdout
    done = client.run("--show-all", fake={"status": 15}, **ACCEPT_CLIPPED_GAP)
    assert outcome(done) == ERROR and "cp_verify: " not in done.stdout


def test_pass_tells_how_much_proof_of_work_was_counted(client):
    done = client.run()
    end = (
        "; proof of work: none counted; the least round has 128.02 bits (cp_verify's own figure, "
        "not a security level; --show-all has its caveats)"
    )
    assert result(done)["work"] == "0" and last_line(done).endswith(end)
    assert "cp_verify: proof of work: none counted" in done.stdout.splitlines()
    done = client.run(fake={"work": True})
    sentence = fake_cp_verify.WORK
    assert outcome(done) == VERIFIED and result(done)["work"] == "32"
    figures = (result(done)["least_round"], result(done)["least_round_without_work"])
    assert figures == ("128.02", "96.02")  # both numbers, as cp_verify's line for programs has them
    end = " (cp_verify's own figures, not a security level; --show-all has its caveats)"
    assert last_line(done).endswith("; " + sentence + end)
    assert "cp_verify: " + sentence in done.stdout.splitlines()


def test_figures_are_those_of_the_verifiers_line_for_programs(client):
    # A parameter file can declare work that no round needs: least-round= then still counts it.
    declared = {"least-round": "136.25", "least-round-without-work": "128.25"}
    done = client.run(fake={"result_v1": declared})
    figures = (result(done)["least_round"], result(done)["least_round_without_work"])
    assert outcome(done) == VERIFIED and figures == ("136.25", "128.25")
    end = "; proof of work: none counted; the least round has 128.25 bits (cp_verify's own figure, "
    assert end in last_line(done) and "136.25" not in last_line(done)
    # Where work is counted, both figures of cp_verify's sentence must be the line's.
    replaced = [["has 128.02 bits, without it 96.02", "has 130.51 bits, without it 97.75"]]
    other = {"least-round": "130.51", "least-round-without-work": "97.75"}
    done = client.run(fake={"work": True, "result_v1": other, "replace": replaced})
    figures = (result(done)["least_round"], result(done)["least_round_without_work"])
    assert outcome(done) == VERIFIED and figures == ("130.51", "97.75")
    assert "has 130.51 bits, without it 97.75 bits (cp_verify's own figures, " in last_line(done)
    for name in other:
        done = client.run(fake={"work": True, "result_v1": {name: other[name]}})
        assert outcome(done) == ERROR, name


@pytest.mark.parametrize(
    "fake",
    [
        {"result_v1": {"work": "32"}},
        {"work": True, "result_v1": {"work": "0"}},
        {"work": True, "result_v1": {"work": "31"}},
        {"work": True, "result_v1": {"work-rounds": "13"}},
        {"work": True, "result_v1": {"least-round-without-work": "96.03"}},
        {
            "work": True,
            "result_v1": {"work-rounds": "1"},
            "replace": [["counted in 14 rounds;", "counted in 1 rounds;"]],
        },
        {"replace": [["proof of work: none counted\n", ""]]},
        {"replace": [["proof of work: none counted", "proof of work: none counted."]]},
        {
            "replace": [
                ["proof of work: none counted\n", "proof of work: none counted\nmeaning: x\n"]
            ]
        },
    ],
    ids=str,
)
def test_line_on_proof_of_work_must_go_with_the_line_for_programs(client, fake):
    assert outcome(client.run(fake=fake)) == ERROR


# cp_verify's line on the commitment's one-time files: without its closing clause, and whole.
LINE_WITHOUT_CLAUSE = fake_cp_verify.on_commitment(fake_cp_verify.ASSUMED)
WHOLE_LINE = LINE_WITHOUT_CLAUSE + fake_cp_verify.RECORD_CLAUSE
SOURCE = fake_cp_verify.SOURCE_WORDS
ONE_ROUNDS = "1 closeness rounds of the one-time files rest"  # cp_verify prints no such words
# What follows cp_verify's sentence on the commitment in the script's last line if no work counts.
NO_WORK_COUNTED = "; proof of work: none counted; the least round has "


@pytest.mark.parametrize(
    "fake, rounds", [({}, 49), ({"assumed": 1}, 1), ({"assumed": 37}, 37)], ids=str
)
def test_pass_tells_what_the_commitment_rests_on(client, fake, rounds):
    sentence = fake_cp_verify.on_commitment(rounds) + fake_cp_verify.RECORD_CLAUSE
    done = client.run(fake=fake)
    assert outcome(done) == VERIFIED and result(done)["commitment_assumed_rounds"] == str(rounds)
    assert " commitment_assumed_rounds=%d record=1 work=0 " % rounds in done.stdout.splitlines()[-2]
    assert "; " + sentence + NO_WORK_COUNTED in last_line(done)
    assert "cp_verify: " + sentence in done.stdout.splitlines()


def test_sentence_on_the_commitment_is_repeated_in_the_words_that_cp_verify_prints(client):
    start, _, printed = fake_cp_verify.RECORD_CLAUSE.partition(": ")
    other = 'whatever cp_verify prints here; such as "A_1/b.2" = [c]'
    done = client.run(fake={"replace": [[printed, other]]})
    assert outcome(done) == VERIFIED and result(done)["record"] == "1"
    assert LINE_WITHOUT_CLAUSE + start + ": " + other + NO_WORK_COUNTED in last_line(done)
    assert printed not in done.stdout


def test_pass_under_a_policy_with_fetched_files_tells_what_it_rests_on(client, published):
    client.set_parameters(**CLIPPED_GAP)
    done = client.run(fetch_from=published, fake={"assumed": 3}, **ACCEPT_CLIPPED_GAP)
    assert outcome(done) == IN_LIMITS and result(done)["commitment_assumed_rounds"] == "3"
    assert result(done)["record"] == "1"
    assert names(client.public) == ["key.bin", "leaf", "leaf/0.bin"]


def test_pass_with_counted_work_tells_what_it_rests_on(client):
    fake = {"work": True}
    done = client.run("--no-proof-of-work", fake=fake)
    assert outcome(done) == NOT_A_VERIFICATION and last_line(done).endswith("--no-proof-of-work")
    done = client.run(fake=fake)
    assert outcome(done) == VERIFIED
    assert (result(done)["record"], result(done)["work"]) == ("1", "32")
    assert result(done)["commitment_assumed_rounds"] == "49"
    words = "; " + WHOLE_LINE + "; " + fake_cp_verify.WORK
    assert words + " (cp_verify's own figures, " in last_line(done)


@pytest.mark.parametrize(
    "fake, expected",
    [
        ({"test_build": True}, NOT_A_VERIFICATION),
        ({"status": 1, "last_line": "NO: request"}, REJECTED),
        ({"status": 14, "last_line": "NOT-A-VERIFICATION"}, NOT_A_VERIFICATION),
    ],
    ids=str,
)
def test_only_a_pass_has_the_field_record(client, fake, expected):
    for flags in ((), ("--no-proof-of-work",)):
        done = client.run(*flags, fake=fake)
        assert outcome(done) == expected and "record=" not in done.stdout.splitlines()[-2]


@pytest.mark.parametrize(
    "fake",
    [
        {"assumed": 0},
        {"result_v1": {"commitment-assumed-rounds": "1"}},
        {"assumed": 37, "result_v1": {"commitment-assumed-rounds": "0"}},
        {"assumed": 37, "result_v1": {"commitment-assumed-rounds": "36"}},
        {"assumed": 7, "replace": [["commitment: 7 closeness", "commitment: 07 closeness"]]},
        {"assumed": 7, "replace": [[SOURCE, " on " + OTHER_ID]]},
        {"assumed": 7, "replace": [[SOURCE, " on theta=300 cap=1"]]},
        {"assumed": 7, "replace": [["commitment: 7 closeness", "commitment: closeness"]]},
        {"assumed": 7, "replace": [[SOURCE + ";", SOURCE + "s;"]]},
        {"assumed": 7, "replace": [[SOURCE, " on the theorem on codes"]]},
        {"assumed": 1, "replace": [["1 closeness round of", "1 closeness rounds of"]]},
        {"assumed": 1, "replace": [["1 closeness round of the one-time files rests", ONE_ROUNDS]]},
        {"assumed": 7, "replace": [[fake_cp_verify.RECORD_CLAUSE, ""]]},
        {"assumed": 7, "replace": [[" record=1", ""]]},
        {"assumed": 7, "replace": [[fake_cp_verify.RECORD_CLAUSE, ""], [" record=1", ""]]},
        {"assumed": 7, "replace": [["Theorem 4.6)", "Theorem\t4.6)"]]},
        {"assumed": 7, "replace": [["Theorem 4.6)", "Th\u00e9or\u00e8me 4.6)"]]},
        {"assumed": 7, "replace": [["Theorem 4.6)", "Theorem 4.6)" + "x" * 300]]},
        {"replace": [[fake_cp_verify.RECORD_CLAUSE.partition(": ")[2], ""]]},
        {"assumed": 7, "replace": [[" record=1 r=", " r="], [" stages=", " record=1 stages="]]},
        {"work": True, "replace": [[" record=1", ""]]},
        {"replace": [[" record=1", " record=0"]]},
        {"replace": [[" record=1", " record=2"]]},
        {"replace": [[" record=1", " record=1 record=1"]]},
        {"replace": [[fake_cp_verify.RECORD_CLAUSE, fake_cp_verify.RECORD_CLAUSE * 2]]},
        {"replace": [[WHOLE_LINE + "\n", ""]]},
        {"replace": [[LINE_WITHOUT_CLAUSE, LINE_WITHOUT_CLAUSE + "."]]},
        {
            "replace": [
                [
                    WHOLE_LINE + "\nproof of work: none counted",
                    "proof of work: none counted\n" + WHOLE_LINE,
                ]
            ]
        },
    ],
    ids=str,
)
def test_line_on_the_commitment_must_go_with_the_line_for_programs(client, fake):
    done = client.run(fake=fake)
    assert outcome(done) == ERROR and "cp_verify exited with status 0, but " in last_line(done)
    assert outcome(client.run("--no-proof-of-work", fake=fake)) == ERROR  # not that option's 14


@pytest.mark.parametrize("line", [None, 0, 2], ids=str)
def test_parameter_file_with_an_unexpected_line_record_gives_no_pass(client, line):
    client.set_parameters(record=line)
    done = client.run()
    assert outcome(done) == ERROR and "are not those of a pass under this para" in last_line(done)


def test_caller_can_refuse_a_pass_that_counts_proof_of_work(client, published):
    done = client.run("--no-proof-of-work", fetch_from=published, fake={"work": True})
    assert outcome(done) == NOT_A_VERIFICATION and "cp_verify: " not in done.stdout
    text = "cp_verify passed, but printed '%s', and you passed --no-proof-of-work"
    assert last_line(done).endswith(text % fake_cp_verify.WORK)
    assert names(client.public) == ["key.bin", "leaf"]  # the fetched leaf file is not saved
    done = client.run("--no-proof-of-work", fetch_from=published)
    assert outcome(done) == VERIFIED and result(done)["work"] == "0"


def test_caller_can_refuse_counted_work_under_a_policy_too(client):
    client.set_parameters(**CLIPPED_GAP)
    done = client.run("--no-proof-of-work", fake={"work": True}, **ACCEPT_CLIPPED_GAP)
    assert outcome(done) == NOT_A_VERIFICATION and "theta=" not in client.printed(done)
    assert outcome(client.run("--no-proof-of-work", **ACCEPT_CLIPPED_GAP)) == IN_LIMITS


def test_one_round_of_proof_of_work_is_in_the_singular(client):
    fake = {"work": True, "result_v1": {"work-rounds": "1"}}
    fake["replace"] = [["counted in 14 rounds;", "counted in 1 round;"]]
    assert outcome(client.run(fake=fake)) == VERIFIED


def test_test_build_is_not_a_verification(client):
    done = client.run(fake={"test_build": True})
    assert outcome(done) == NOT_A_VERIFICATION and "test build" in last_line(done)


def test_missing_leaf_file_is_an_error(client):
    client.leaf_file.unlink()
    done = client.run()
    assert outcome(done) == ERROR and "leaf/0.bin, which the proof needs, is missing" in last_line(
        done
    )
    assert client.calls() == []


def test_verifier_output_cannot_imitate_the_last_line(client):
    done = client.run(fake={"last_line": "NO: x\rcheck_proof: verified: y", "status": 1})
    assert outcome(done) == REJECTED and "\r" not in done.stdout


# ---- the parameter file ----


def test_parses_a_parameter_file():
    values = check_proof.parse_policy(parameter_file(cap=4294967295))
    assert (values["C"], values["cap"], "call" in values) == ("256", "4294967295", False)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"C 256",
        b"C 256\r\n",
        b"C 256\n\n",
        b"C  256\n",
        b"C 256 \n",
        b"C\t256\n",
        b"C 256 1\n",
        b"C 256\nC 256\n",
        b"C +1\n",
        b"C 01\n",
        b"C 0x10\n",
        b"C 4294967296\n",
        b"C \xd9\xa1\n",  # an Arabic-Indic digit
        b"C 1\x00\n",
        b"call A\n",
        b"call A  B C\n",
        b"call A B C\ncall A B D\n",
        b"work positions\n",
        b"version H821\nversion H821\n",
        b"version 5\n",
        b"version 12345\n",
        b"version H821 1\n",
        b"version H82-\n",
    ],
)
def test_parser_rejects_what_two_parsers_could_read_differently(data):
    with pytest.raises(check_proof.PolicyError):
        check_proof.parse_policy(data)


def test_bad_parameter_file_is_an_error_before_any_call(client):
    (client.config / "policy.txt").write_bytes(parameter_file() + b"C 1000\n")
    assert outcome(client.run()) == ERROR and client.calls() == client.calls(plans=True) == []


def test_layout_file_can_be_named(client):
    (client.config / "layout.txt").rename(client.top / "other.txt")
    assert outcome(client.run(layout_file=client.top / "other.txt")) == VERIFIED
    assert "config/layout.txt" in client.calls()[0]["files"]
    assert outcome(client.run(layout_file=client.top / "nothing_here")) == ERROR


# ---- --fetch-from ----


@pytest.fixture
def published(client):
    """Move the leaf file from the public key directory to where it can be fetched."""
    (client.top / "published" / "leaf").mkdir(parents=True)
    client.leaf_file.rename(client.top / "published" / "leaf" / "0.bin")
    return client.top / "published"


@pytest.mark.parametrize("name", ["key.bin", "leaf/0.bin", "policy.txt"])
def test_nothing_is_fetched_where_an_entry_exists(client, published, name):
    """A dangling link is neither followed nor replaced."""
    directory = client.config if name == "policy.txt" else client.public
    source = directory / name
    if source.exists():
        source.rename(published / name)
    source.symlink_to(client.top / "nothing_here")
    done = client.run(fetch_from=published)
    assert outcome(done) == ERROR and client.calls() == []
    assert source.is_symlink() and not (client.top / "nothing_here").exists()


def test_named_layout_file_is_not_fetched(client, published):
    (client.config / "layout.txt").rename(published / "layout.txt")
    named = client.config / "layout.txt"
    done = client.run(fetch_from=published, layout_file=named)
    assert outcome(done) == ERROR and "layout.txt does not exist" in last_line(done)
    assert not named.exists() and outcome(client.run(fetch_from=published)) == VERIFIED


def test_temporary_directory_that_was_replaced_is_not_used(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    other = client.top / "somebody_elses"
    other.mkdir()
    (other / "a_file").write_text("kept")
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_: str(other))
    assert module.main(client.words()) == module.EXIT_ERROR and client.calls(plans=True) == []
    assert "was replaced" in capsys.readouterr().out.splitlines()[-1]
    assert names(other) == ["a_file"]


def test_clean_up_removes_its_directory_where_it_lies_and_no_other_of_that_name(
    client, monkeypatch
):
    module = load_copy(client, monkeypatch)
    monkeypatch.chdir(client.top)
    with module.PrivateDirectory() as work:
        name = Path(work.path).name
        client.tmp.rename(client.top / "moved")  # as if another user had replaced TMPDIR
        (client.tmp / name).mkdir(parents=True)
    assert names(client.tmp) == [name] and names(client.top / "moved") == []


def test_clean_up_leaves_another_directory_that_has_taken_its_name(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    monkeypatch.chdir(client.top)
    with module.PrivateDirectory() as work:
        Path(work.path).rename(client.tmp / "moved")
        Path(work.path).mkdir()
    assert names(client.tmp) == [Path(work.path).name, "moved"]


@pytest.mark.parametrize("open_place", ["the directory", "its parent"])
def test_fetched_file_is_not_saved_where_another_account_could_change_a_directory(
    client, published, open_place
):
    place = client.public if open_place == "the directory" else client.public.parent
    place.chmod(0o777)
    done = client.run(fetch_from=published)
    place.chmod(0o755)
    assert outcome(done) == VERIFIED and "saved the fetched file" not in done.stdout
    warning = "warning: did not save %s: another account may be able to change %s (its group or"
    assert warning % (client.leaf_file, place) in done.stdout
    # chmod go-w is advised to the directory's owner, never to root: see check_proof.cure().
    cure = "no sticky bit): if it is yours, run chmod go-w on it, and then look at what is in it"
    advised = cure + "\n" in done.stdout
    assert advised == (os.geteuid() != 0)
    assert names(client.public) == ["key.bin", "leaf"]


def test_fetched_file_is_not_saved_in_a_leaf_directory_that_others_can_change(client, published):
    (client.public / "leaf").chmod(0o777)
    done = client.run(fetch_from=published)
    (client.public / "leaf").chmod(0o755)
    assert outcome(done) == VERIFIED and names(client.public) == ["key.bin", "leaf"]
    text = "warning: did not save %s: the directory leaf can be written by its group or by other "
    assert text % client.leaf_file in done.stdout
    # The cure is named to the directory's owner, never to root.
    cure = "other accounts: if it is yours, run chmod go-w on it, and then look at what is in it\n"
    assert (cure in done.stdout) == (os.geteuid() != 0)


def test_fetched_file_is_not_saved_in_a_leaf_directory_of_another_account(
    client, monkeypatch, capsys
):
    module = load_copy(client, monkeypatch)
    client.leaf_file.unlink()
    monkeypatch.setattr(module, "others_can_change", lambda folder: None)  # no check above leaf/
    monkeypatch.setattr(
        module, "trusted_owners", set
    )  # no owner is trusted, so leaf/ counts as another account's
    module.save_fetched(client.public / "key.bin", str(client.leaf_file), True)
    text = (
        "warning: did not save %s: the directory leaf belongs to another account: name another "
        "--public-key-dir\n"
    )
    assert text % client.leaf_file in capsys.readouterr().out
    assert names(client.public) == ["key.bin", "leaf"]


def test_endless_sudo_uid_costs_the_saving_and_not_the_verdict(client, monkeypatch, capsys):
    """As root, int() refuses a SUDO_UID of more digits than Python converts (3.11 and later)."""
    module = load_copy(client, monkeypatch)
    client.leaf_file.unlink()

    def endless(folder):
        raise ValueError("Exceeds the limit (4300 digits) for integer string conversion")

    monkeypatch.setattr(module, "others_can_change", endless)
    module.save_fetched(client.public / "key.bin", str(client.leaf_file), True)
    text = (
        "warning: did not save %s: the environment variable SUDO_UID is not a user ID: unset it\n"
    )
    assert capsys.readouterr().out == text % client.leaf_file
    assert names(client.public / "leaf") == []


def test_directory_that_the_caller_names_leaf_is_like_any_other(client, published):
    (client.public / "key.bin").rename(published / "key.bin")
    named = client.top / "elsewhere" / "leaf"  # it does not exist, and is not created
    (client.top / "elsewhere").mkdir()
    done = client.run(fetch_from=published, public_key_dir=named)
    assert outcome(done) == VERIFIED and names(client.top / "elsewhere") == []
    text = (
        "warning: did not save %s: the directory %s is missing (it does not exist): create it "
        "(mkdir -m 755) to keep fetched files"
    )
    assert text % (named / "key.bin", named) in done.stdout


def test_fetched_file_is_not_saved_in_a_directory_that_does_not_exist(client, published):
    for name in ("key.bin", "leaf"):
        (client.public / name).rename(published / name) if name == "key.bin" else None
    (client.public / "leaf").rmdir()
    client.public.rmdir()
    done = client.run(fetch_from=published)
    assert outcome(done) == VERIFIED and not client.public.exists()
    text = (
        "warning: did not save %s: the directory %s is missing (it does not exist): create it "
        "(mkdir -m 755) to keep fetched files"
    )
    assert text % (client.public / "key.bin", client.public) in done.stdout


def test_emptied_temporary_directory_stays_if_the_one_above_cannot_be_opened(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    monkeypatch.chdir(client.top)
    real = os.open

    def open_but_not_the_parent(path, *more, **others):
        if path == os.pardir:
            raise OSError(errno.EMFILE, "Too many open files")
        return real(path, *more, **others)

    monkeypatch.setattr(module.os, "open", open_but_not_the_parent)
    with module.PrivateDirectory() as work:
        Path(work.new_file("a_file")).write_text("this run's")
    assert names(client.tmp) == [Path(work.path).name] and Path.cwd() == client.top


def test_clean_up_deletes_only_in_its_own_directory(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    other = client.top / "somebody_elses"
    (other / "config").mkdir(parents=True)
    (other / "a_file").write_text("kept")
    monkeypatch.chdir(client.top)
    with module.PrivateDirectory() as work:
        Path(work.new_file("a_file")).write_text("this run's")
        os.chdir(other)  # as if something had changed the working directory meanwhile
    assert names(other) == ["a_file", "config"] and Path.cwd() == client.top


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_not_followed_and_the_key_goes_nowhere_else(client, published, status):
    seen = []

    class Second(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_error(404)

        def log_message(self, *_):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Second) as second:
        target = "http://127.0.0.1:%d/leaf/0.bin" % second.server_address[1]

        class First(Second):
            def do_GET(self):
                self.send_response(status)
                self.send_header("Location", target)
                self.end_headers()

        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), First) as first:
            for server in (first, second):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                env = {**client.environment(), check_proof.FETCH_KEY: "k-1"}
                source = "http://127.0.0.1:%d" % first.server_address[1]
                done = client.run_script(client.words(fetch_from=source), env=env)
            finally:
                first.shutdown()
                second.shutdown()
    assert outcome(done) == ERROR and "HTTP status %d" % status in done.stdout and seen == []


def test_fetch_from_a_url_sends_the_key_from_its_environment_variable(client, published):
    seen = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            if seen[-1] != "Bearer k-1":
                return self.send_error(401)
            return super().do_GET()

        def log_message(self, *_):
            pass

    handler = functools.partial(Handler, directory=str(published))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        words = client.words(fetch_from="http://127.0.0.1:%d" % server.server_address[1])
        env = {**client.environment(), "no_proxy": "*", "VLLM_API_KEY": "another program's"}
        done = client.run_script(words, env=env)
        assert outcome(done) == ERROR and seen == [None]
        assert "warning: could not fetch leaf/0.bin: HTTP status 401" in done.stdout
        # With the key, a proxy from the environment is bypassed: it would see the key.
        env.update(
            {check_proof.FETCH_KEY: "k-1", "no_proxy": "", "http_proxy": "http://127.0.0.1:9"}
        )
        assert outcome(client.run_script(words, env=env)) == VERIFIED
        server.shutdown()
    assert client.leaf_file.read_bytes() == (published / "leaf" / "0.bin").read_bytes()


def test_key_is_not_sent_in_clear_to_another_machine(client, published):
    env = {**client.environment(), check_proof.FETCH_KEY: "k-1"}
    for source in ("http://files.example/x", "http://localhost:8000"):
        done = client.run_script(client.words(fetch_from=source), env=env)
        assert is_short_usage_error(client, done)
        assert "must be https, or http://127.0.0.1" in done.stderr
    assert outcome(client.run_script(client.words(fetch_from=published), env=env)) == VERIFIED


def test_saves_a_fetched_leaf_file_once_verified(client, published):
    assert outcome(client.run(fetch_from=published)) == VERIFIED
    data = (published / "leaf" / "0.bin").read_bytes()
    assert client.calls()[0]["files"]["ivk/leaf/0.bin"] == data.hex()
    assert names(client.public) == ["key.bin", "leaf", "leaf/0.bin"]
    assert client.leaf_file.read_bytes() == data


@pytest.mark.parametrize(
    "fake",
    [
        {"last_line": "NO: batch-1", "status": 1},
        {"test_build": True},
        {"result_v1": {"id": OTHER_ID}},
        {"result_v1": {"leaf": "1"}},
        {"result_v1": {"stages": "key,request"}},
    ],
    ids=str,
)
def test_saves_no_fetched_file_unless_verified(client, published, fake):
    assert outcome(client.run(fetch_from=published, fake=fake)) != VERIFIED
    assert names(client.public) == ["key.bin", "leaf"]


def test_never_overwrites_a_file_that_appeared_meanwhile(client, published):
    done = client.run(fetch_from=published, fake={"create": {str(client.leaf_file): "not ours"}})
    assert outcome(done) == VERIFIED and "warning" in done.stdout
    assert names(client.public) == ["key.bin", "leaf", "leaf/0.bin"]
    assert client.leaf_file.read_text() == "not ours"


def test_saving_does_not_touch_the_target_of_a_swapped_in_symlink(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    victim = client.top / "victim"
    victim.write_text("the caller's own")
    victim.chmod(0o600)
    client.leaf_file.unlink()
    copyfileobj = shutil.copyfileobj

    def copy_and_swap(source, to):  # another user swaps the unfinished file for a symlink
        copyfileobj(source, to)
        (part,) = client.leaf_file.parent.iterdir()
        part.unlink()
        part.symlink_to(victim)

    monkeypatch.setattr(module.shutil, "copyfileobj", copy_and_swap)
    module.save_fetched(client.public / "key.bin", str(client.leaf_file), True)
    assert (victim.stat().st_mode & 0o777, victim.read_text()) == (0o600, "the caller's own")
    out = capsys.readouterr().out
    assert "warning: did not save" in out and "saved the fetched file" not in out


def test_saves_nothing_through_a_leaf_directory_that_became_a_symlink(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    (client.top / "victim").mkdir()
    client.leaf_file.unlink()

    def swap(name, _, dir_fd):  # another user swaps leaf/ for a symlink, as late as possible
        os.rmdir(name, dir_fd=dir_fd)
        os.symlink(client.top / "victim", name, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", swap)
    module.save_fetched(client.public / "key.bin", str(client.leaf_file), True)
    assert "warning: did not save" in capsys.readouterr().out
    assert names(client.top / "victim") == []


def test_saves_into_a_public_key_directory_that_the_caller_names_by_a_symlink(client, published):
    (client.public / "key.bin").rename(published / "key.bin")
    (client.top / "link").symlink_to(client.public)
    assert outcome(client.run(fetch_from=published, public_key_dir=client.top / "link")) == VERIFIED
    assert names(client.public) == ["key.bin", "leaf", "leaf/0.bin"]
    assert (client.public / "key.bin").stat().st_mode & 0o777 == 0o644


def test_does_not_fetch_what_the_caller_has(client, published):
    client.leaf_file.write_bytes(b"the caller's own")
    assert outcome(client.run(fetch_from=published)) == VERIFIED
    assert client.calls()[0]["files"]["ivk/leaf/0.bin"] == "symlink"
    assert client.leaf_file.read_bytes() == b"the caller's own"


def test_size_of_a_fetched_file_is_capped(client, published, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    monkeypatch.setattr(module, "MAX_FETCHED_BYTES", 10)
    assert module.main(client.words(fetch_from=published)) == module.EXIT_ERROR
    assert "longer than the 10 bytes" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["policy.txt", "request"])
def test_file_that_is_too_long_is_not_copied(client, monkeypatch, capsys, name):
    module = load_copy(client, monkeypatch)
    monkeypatch.setattr(module, "MAX_SMALL_FILE_BYTES", 5300)
    assert module.main(client.words()) == module.EXIT_VERIFIED
    path = client.request if name == "request" else client.config / name
    os.truncate(path, 5301)
    client.log.unlink()
    assert module.main(client.words()) == module.EXIT_ERROR and client.calls() == []
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == "check_proof: error: %s: longer than the 5300 bytes that it can have" % path


def test_fetches_the_key_and_the_configuration_files(client, published):
    (client.public / "key.bin").rename(published / "key.bin")
    for name in ("policy.txt", "KL.bin", "layout.txt"):
        (client.config / name).rename(published / name)
    assert outcome(client.run(fetch_from=published)) == VERIFIED
    assert names(client.public) == ["key.bin", "leaf", "leaf/0.bin"]
    assert names(client.config) == ["KL.bin", "layout.txt", "policy.txt"]


def test_missing_file_without_a_source_is_an_error(client):
    (client.public / "key.bin").unlink()
    assert outcome(client.run()) == ERROR


# ---- the command line ----


def test_main_returns_the_exit_code_and_restores_the_working_directory(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    monkeypatch.chdir(client.config)
    assert module.main(client.words()) == module.EXIT_VERIFIED
    assert module.main(["--help"]) == 0 and module.main([]) == 2
    assert module.main(client.words(range="2:4")) == module.EXIT_REJECTED
    assert Path.cwd() == client.config and os.listdir(client.tmp) == []
    assert capsys.readouterr().out.splitlines()[-1].startswith("check_proof: rejected: ")
    client.set_parameters(**CLIPPED_GAP)
    assert module.main(client.words(**ACCEPT_CLIPPED_GAP)) == module.EXIT_VERIFIED == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("check_proof: within limits: ")


def test_stopped_by_a_signal_it_cleans_up_and_reports_an_error(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    real = module.run

    def run_and_be_stopped(words, **more):
        if words[1] == "-plan":
            return real(words, **more)
        return module.stop(signal.SIGTERM, None)  # as if it arrived during the verifier's run

    monkeypatch.setattr(module, "run", run_and_be_stopped)
    assert module.main(client.words()) == module.EXIT_ERROR
    assert capsys.readouterr().out.splitlines()[-1] == "check_proof: error: stopped by SIGTERM"
    assert os.listdir(client.tmp) == []


def test_second_signal_does_not_interrupt_the_clean_up(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    module.ENDING[0] = False
    with pytest.raises(module.Stopped):
        module.stop(signal.SIGINT, None)
    assert module.stop(signal.SIGINT, None) is None


def test_late_signal_changes_nothing(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    assert module.main(client.words()) == module.EXIT_VERIFIED
    assert module.stop(signal.SIGTERM, None) is None  # the result is printed: nothing is raised
    assert module.main([]) == module.EXIT_ERROR and module.ENDING[0]
    ending, run = [], module.run
    monkeypatch.setattr(
        module, "run", lambda *a, **k: ending.append(module.ENDING[0]) or run(*a, **k)
    )
    assert module.main(client.words()) == module.EXIT_VERIFIED
    assert ending and not any(ending)  # each main() call lets a signal stop it again
    real = module.PrivateDirectory.__exit__

    def exit_and_be_stopped(self, *more):
        real(self, *more)
        module.stop(signal.SIGTERM, None)  # as if it arrived at the end of the clean-up

    monkeypatch.setattr(module.PrivateDirectory, "__exit__", exit_and_be_stopped)
    assert module.main(client.words()) == module.EXIT_VERIFIED
    assert capsys.readouterr().out.splitlines()[-1].startswith("check_proof: verified: ")
    assert os.listdir(client.tmp) == []


def test_command_catches_a_stop_that_main_does_not(client, monkeypatch, capsys):
    module = load_copy(client, monkeypatch)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)  # do not touch pytest's handlers

    def stopped_early():
        module.ENDING[0] = False
        module.stop(signal.SIGHUP, None)

    monkeypatch.setattr(module, "main", stopped_early)
    with pytest.raises(SystemExit) as ended:
        module.command()
    assert ended.value.code == module.EXIT_ERROR and module.ENDING[0]
    lines = ["", "result: outcome=error", "check_proof: error: stopped by SIGHUP"]
    assert capsys.readouterr().out.splitlines()[-3:] == lines


def test_command_sets_the_signal_handlers(client, monkeypatch):
    module = load_copy(client, monkeypatch)
    calls = []
    monkeypatch.setattr(module.signal, "signal", lambda *call: calls.append(call))
    monkeypatch.setattr(module, "main", lambda: calls.append("main") or 0)
    with pytest.raises(SystemExit):
        module.command()
    assert calls == [
        (signal.SIGCHLD, signal.SIG_DFL),  # see command()
        (signal.SIGTERM, module.stop),
        (signal.SIGHUP, module.stop),
        (signal.SIGINT, module.stop),
        "main",
    ]


def is_short_usage_error(client, done):
    lines = done.stderr.splitlines()
    assert lines[0].startswith("usage: check_proof.py ") and len(lines) < 15
    assert lines[-1].startswith("check_proof.py: error: ")
    return (done.returncode, done.stdout) == (2, "") and client.calls(plans=True) == []


@pytest.mark.parametrize(
    "more",
    [
        "--range 2:5",
        "--reply 3:3",
        "--ms " + OTHER_ID,
        "one_more_word",
        "--theta 300",
        "--policy clipped-gap --C 1000",
        "--policy logprob",
        "--bits 128",
        "--threads 0",
        "--help",  # accepted only as the sole argument
        "-h",
    ],
)
def test_options_that_do_not_fit_are_a_short_usage_error(client, more):
    assert is_short_usage_error(client, client.run(*more.split()))


@pytest.mark.parametrize(
    "verifier, fault",
    [
        (None, "bin/cp_verify does not exist or is not executable"),
        ("nothing_here/cp_verify", "nothing_here/cp_verify does not exist or is not executable"),
    ],
)
def test_missing_verifier_is_an_error_of_the_check_that_names_the_build_command(
    client, verifier, fault
):
    client.verifier.unlink()
    done = client.run(verifier=verifier)
    assert outcome(done) == ERROR and done.stderr == ""
    assert fault + " (to build it, run verifier/build.sh in the repo" in last_line(done)
    assert "; or pass --verifier PATH)" in last_line(done)


def test_no_arguments_or_an_abbreviated_option_is_a_short_usage_error(client):
    assert is_short_usage_error(client, client.run_script([]))
    words = ["--commitment" if word == "--commitment-id" else word for word in client.words()]
    assert is_short_usage_error(client, client.run_script(words))


NEEDED = "commitment_id range end_token public_key_dir parameter_file scales_file".split()


@pytest.mark.parametrize("option", NEEDED)
def test_needed_option_cannot_be_left_out(client, option):
    assert is_short_usage_error(client, client.run(**{option: None}))


@pytest.mark.parametrize(
    "option, value",
    [
        ("commitment_id", "0" * 63),
        ("range", "2-5"),
        ("range", "-1:5"),
        ("end_token", "15,"),
        ("end_token", "any"),
        ("next_token", "4294967296"),
        ("threads", "1e2"),
        ("public_key_dir", ""),
        ("fetch_from", ""),
    ],
)
def test_malformed_value_is_a_short_usage_error(client, option, value):
    assert is_short_usage_error(client, client.run(**{option: value}))


def test_unexpected_error_is_not_a_rejection(client, tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    shell = (
        'cd "$1" && rmdir "$1" && shift && exec "$@"'  # a working directory that no longer exists
    )
    words = ["sh", "-c", shell, "sh", str(gone), *PYTHON, str(client.script)]
    words += ["response", "request.bin"] + client.words()[2:]
    done = subprocess.run(words, capture_output=True, text=True, env=client.environment())
    assert done.returncode == 2 and last_line(done).startswith("check_proof: error: ")
    assert "Traceback" not in done.stderr


@pytest.mark.parametrize("redirect", [">/dev/full", ">&-"])
def test_full_or_closed_standard_output(client, redirect):
    words = [*PYTHON, str(client.script), *client.words()]
    shell = 'exec "$@" ' + redirect
    done = subprocess.run(["sh", "-c", shell, "sh", *words], env=client.environment())
    assert done.returncode == (2 if redirect == ">/dev/full" else 0)


def test_help_is_short(client):
    done = client.run_script(["--help"])
    assert done.returncode == 0 and len(done.stdout.splitlines()) <= 100
    assert all(word in done.stdout for word in ("--commitment-id", "--end-token", "exit codes:"))
    assert not any(line.startswith("check_proof: ") for line in done.stdout.splitlines())
    for code, words in check_proof.LAST_LINE.items():
        assert '  %-3d "%s"' % (code, words) in done.stdout
    assert '  0   "%s"' % check_proof.WITHIN_LIMITS_WORDS in done.stdout
    assert max(map(len, done.stdout.splitlines())) <= 80
    assert "SIGTERM, SIGHUP and SIGINT stop the check (exit code 2)," in done.stdout
    assert ": so where work=0, read least_round_without_work=.\n" in done.stdout
    assert "Both are rounded down to two decimals for printing" in done.stdout
    assert "Do not compare them with a target: cp_verify makes that comparison\n" in done.stdout


# ---- the fake verifier itself ----


def run_fake(client, *words, **environment):
    words = [*PYTHON, fake_cp_verify.__file__, *map(str, words)]
    return subprocess.run(words, capture_output=True, text=True, env=environment, cwd=client.top)


@pytest.mark.parametrize("environment", [{}, {"FAKE_CP_VERIFY": ""}], ids=["unset", "empty"])
def test_fake_verifier_runs_only_for_a_test(client, environment):
    done = run_fake(client, "-plan", "id", client.config, client.public / "key.bin", **environment)
    assert (done.returncode, done.stdout) == (2, "") and "it checks nothing" in done.stderr


def test_fake_verifier_marks_every_line(client):
    files = [client.request, client.response / "y.bin", client.response / "proof.bin"]
    calls = [
        ("-plan", "id", client.config, client.public / "key.bin"),
        (client.config, client.public, *files, "-id", client.id, "-bits", 128, "-q"),
    ]
    for call in calls:
        lines = run_fake(client, *call, FAKE_CP_VERIFY="{}").stdout.splitlines()
        assert lines and all(line.endswith(" (fake)") for line in lines)
    assert lines[-1] == "YES (fake)"


def test_script_does_not_accept_the_fake_verifiers_marked_lines(client):
    write_program(client.verifier, 'exec %s "$@"' % FAKE)
    assert outcome(client.run()) == ERROR
    assert client.calls(plans=True)


@pytest.mark.parametrize(
    "fake",
    [
        {"result_v1": {"least-round": "128x02"}},
        {"result_v1": {"least-round-without-work": "96x02"}},
        {"replace": [["without it 96.02 bits", "without it 96.02 bits."]]},
        {
            "result_v1": {"least-round": "128.020"},
            "replace": [["has 128.02 bits,", "has 128.020 bits,"]],
        },
        {  # a sentence with one figure alone
            "replace": [
                [
                    "with it the least round has 128.02 bits, without it 96.02 bits",
                    "without it the least round has 96.02 bits",
                ]
            ],
        },
    ],
    ids=str,
)
def test_counted_work_told_in_another_form_is_an_error(client, fake):
    assert outcome(client.run(fake={"work": True})) == VERIFIED
    assert outcome(client.run(fake={"work": True, **fake})) == ERROR


@pytest.mark.parametrize(
    "words, expected",
    [
        ("x" * 180, VERIFIED),
        ("x" * 181, ERROR),
        ("a\x7fb", ERROR),
        ("a\rb", ERROR),
        (" ", ERROR),
        ("   ", ERROR),
        (" x", ERROR),
        ("x ", ERROR),
        ("a  b", ERROR),
        ("; the table has a record:", ERROR),
    ],
    ids=repr,
)
def test_words_after_the_records_marker_are_at_most_180_characters_with_single_blanks(
    client, words, expected
):
    printed = fake_cp_verify.RECORD_CLAUSE.partition(": ")[2]
    done = client.run(fake={"replace": [[printed, words]]})
    assert outcome(done) == expected
    if expected == VERIFIED:
        assert "; the table has a record: %s; proof of work: " % words in last_line(done)


def test_longest_last_line_of_a_pass_is_not_cut():
    top = 2**32 - 1
    values = {"C": top, "theta": top, "cap": top, "s": 31, "sh": 8, "sT": 4, "F": 24}
    args = type("Args", (), {"next_token": top, "end_token": {top}})
    head = fake_cp_verify.on_commitment(top)
    work = "proof of work: up to %d bits counted in %d rounds; with it the " % (top, top)
    work += "least round has %d.00 bits, without it %d.00 bits" % (top, top)
    for pattern, line in ((check_proof.ON_THE_COMMITMENT, head), (check_proof.WORK, work)):
        assert check_proof.re.fullmatch(pattern, line)  # lines that the script takes
    words = "x" * check_proof.MAX_RECORD_WORDS
    rests_on = "; ".join([head + check_proof.RECORD_CLAUSE + words, work])
    rests_on += " " + check_proof.BOTH_FIGURES
    for reaches_end in (False, True):
        fields = {"start": top - 10**9, "checked": 10**9, "reaches_end": reaches_end, "last": top}
        text = check_proof.describe(args, values, type("Request", (), fields)) + "; " + rests_on
        assert text.startswith("clipped gap: the logit gaps ") and " gaps can be above " in text
        assert 900 < len(text) <= 1100 and check_proof.one_line(text, 1100) == text


def test_stand_in_has_the_clause_on_a_record_of_cp_verifys_source():
    source = HERE.parent.parent / "verifier" / "src" / "vf_common.h"
    if not source.parent.exists():
        pytest.skip("%s is not here: this is only a part of the repository" % source.parent)
    start = '#define RECORDCLAUSE "'
    lines = [line for line in source.read_text().splitlines() if line.startswith(start)]
    found = [line[len(start) :].partition('"')[0] for line in lines]
    assert found == [fake_cp_verify.RECORD_CLAUSE], "looked for %r in %s" % (start, source)


def test_tests_do_not_hand_the_tester_s_own_api_keys_to_the_script(client, monkeypatch):
    # A tester's own key would make a test that awaits no key fail, and pytest would print it.
    for name in check_proof.API_KEYS:
        monkeypatch.setenv(name, "the tester's own")
    monkeypatch.setenv("SOMETHING_ELSE", "kept")
    environment = client.environment()
    assert not set(environment) & set(check_proof.API_KEYS)
    assert environment["SOMETHING_ELSE"] == "kept"


def test_exact_match_at_another_offset_is_rejected_under_every_policy(client):
    # cp_verify also counts C = theta = 128, sT 3 at sh 7 as exact match; this script does not.
    client.set_parameters(C=128, theta=128, cap=0, sh=7, sT=3)
    for options in ({}, ACCEPT_CLIPPED_GAP):
        done = client.run(**options)
        assert outcome(done) == REJECTED and client.calls() == []
        assert "has sh, sT or F other than 8, 4 and 24" in last_line(done)
    client.set_request(TOKENS, 5, 6, next_token=7)  # a range that checks the next token alone
    done = client.run(end_token="none", range="5:6", next_token=7)
    assert outcome(done) == REJECTED and client.calls() == []
    assert "has sh, sT or F other than 8, 4 and 24" in last_line(done)
