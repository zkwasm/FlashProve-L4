# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests for integrations/vllm/end_to_end/end_to_end.py that need no GPU, vLLM or built program."""

import argparse
import contextlib
import errno
import itertools
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

import pytest

if sys.version_info < (3, 11):
    pytest.skip("end_to_end.py needs Python 3.11 or later", allow_module_level=True)

import ask_and_check
import check_proof
import end_to_end
import spool

SCRIPT = end_to_end.__file__
has_float_pair = end_to_end.has_float_pair  # saved before with_the_float_pair replaces it
CHILD_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}  # pytest's import path
ID = "ab" * 32
EXAMPLE = os.path.join(end_to_end.REPO, "prover", "examples", end_to_end.DEMO_CONFIGURATION)
# Skipped only in a copy of integrations/ alone: in a repository a missing example is a failure.
needs_the_example = pytest.mark.skipif(
    not os.path.isdir(os.path.dirname(EXAMPLE)), reason=os.path.dirname(EXAMPLE) + " is missing"
)
# For the tests that run as root: the account in SUDO_UID, and one that nothing trusts. Under sudo,
# pytest's own directories may be below the first one's home, so the tests keep that account.
SUDO_UID = max(end_to_end.trusted_owners() - {0, os.geteuid()}, default=65534)
OTHER_UID = 65533 if SUDO_UID != 65533 else 65534
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
IGNORES_SIGTERM = [
    sys.executable,
    "-c",
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); "
    "time.sleep(60)",
]


# The eight options for your own commitment, then "--serve integer", which [:-2] leaves out.
def own_options(directory):
    def at(name):
        return str(directory / name)

    return [
        "--commitment-id",
        ID,
        "--public-key-dir",
        at("pub"),
        "--secret-key-dir",
        at("sec"),
        "--parameter-file",
        at("policy.txt"),
        "--scales-file",
        at("KL.bin"),
        "--model-dir",
        at("model"),
        "--prompts-file",
        at("prompts.txt"),
        "--end-token",
        "7",
        "--serve",
        "integer",
    ]


only_root = pytest.mark.skipif(
    os.geteuid() != 0, reason="only root can give a directory to another user"
)


def shown(path):
    return spool.printable(str(path))


def usage_error(capsys, *argv):
    with pytest.raises(SystemExit) as stopped:
        end_to_end.parse_args(list(argv))
    assert stopped.value.code == 2
    return capsys.readouterr().err


def wait_for(condition, seconds=20):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def is_running(pid):
    """Return whether a process exists and is not a zombie."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.fixture(autouse=True)
def with_the_default_sigchld():
    """Under an ignored SIGCHLD, which can be inherited, the kernel reaps a test's children at once:
    their exit codes are lost, and a PID that a test still holds can become another process's."""
    set_handler = signal.signal  # a test may replace signal.signal until after this fixture ends
    before = set_handler(signal.SIGCHLD, signal.SIG_DFL)
    yield
    set_handler(signal.SIGCHLD, before)


@pytest.fixture(autouse=True)
def with_the_float_pair(monkeypatch):
    """Make the tests independent of whether integer_model_vllm.float_pair is installed."""
    monkeypatch.setattr(end_to_end, "has_float_pair", lambda: True)


@pytest.fixture(autouse=True)
def in_an_empty_directory(tmp_path_factory, monkeypatch):
    """Keep relative paths such as "w" clear of pytest's start directory.

    If the path of pytest's temporary directory has a character that the script takes in no
    WORK_DIR, a test is skipped where the script refuses its WORK_DIR for that; the others run."""
    directory = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(directory)
    if re.fullmatch("[A-Za-z0-9/._-]+", str(directory)):
        return
    error = argparse.ArgumentParser.error

    def skip_or_error(parser, message):
        if message.startswith("WORK_DIR's full path, "):
            pytest.skip(
                "the script takes a WORK_DIR of letters, digits and / . _ - alone, and the path of "
                "pytest's temporary directory has another character: %r. Choose one whose path has "
                "none: pytest takes it from TMPDIR; if you passed --basetemp, give it the name of "
                "a new directory (pytest removes an existing one)" % str(directory)
            )
        error(parser, message)

    monkeypatch.setattr(argparse.ArgumentParser, "error", skip_or_error)


@pytest.fixture
def programs(tmp_path):
    """Return a Programs object whose start() also appends its grace_s to the list `graces`."""
    before = signal.getsignal(signal.SIGCHLD)
    started = end_to_end.Programs(str(tmp_path))
    started.graces, start = [], started.start

    def recording_start(name, argv, env=None, grace_s=None):
        started.graces.append(grace_s)
        return start(name, argv, env, grace_s)

    started.start = recording_start
    yield started
    started.kill_now = True
    started.stop_all()
    signal.signal(signal.SIGCHLD, before)


# ---- Options ----


@needs_the_example
def test_the_demos_configuration_is_in_the_repository():
    assert {"policy.txt", "KL.bin", "layout.txt"} <= set(os.listdir(EXAMPLE))


def test_the_programs_are_looked_for_where_the_build_scripts_put_them():
    args = end_to_end.parse_args(["w"])
    assert args.build_dir == os.path.join(end_to_end.REPO, "prover/build/hc")  # prover/build.sh
    assert args.verifier == os.path.join(end_to_end.REPO, "bin/cp_verify")  # verifier/build.sh


def test_each_way_of_serving_has_its_default_policy():
    assert end_to_end.parse_args(["w", "--serve", "integer"]).policy == "exact"
    assert end_to_end.parse_args(["w", "--serve", "float"]).policy == "clipped-gap"
    args = end_to_end.parse_args(["w", "--serve", "integer", "--policy", "clipped-gap"])
    assert args.policy == "clipped-gap"


def test_ordinary_serving_is_the_default(tmp_path):
    args = end_to_end.parse_args(["w"])
    assert (args.serve, args.policy, args.fell_back) == ("float", "clipped-gap", False)
    own = own_options(tmp_path)[:-2]
    args = end_to_end.parse_args(["w", *own, "--limits", "1,2,3,4"])
    assert (args.serve, args.policy, args.fell_back) == ("float", "clipped-gap", False)


def test_without_the_float_pair_only_a_call_without_serve_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(end_to_end, "has_float_pair", lambda: False)
    args = end_to_end.parse_args(["w"])
    assert (args.serve, args.policy, args.fell_back) == ("integer", "exact", True)
    args = end_to_end.parse_args(["w", "--serve", "float"])
    assert (args.serve, args.policy, args.fell_back) == ("float", "clipped-gap", False)
    assert not end_to_end.parse_args(["w", "--serve", "integer"]).fell_back
    # Your own commitment needs no float pair.
    own = own_options(tmp_path)[:-2]
    args = end_to_end.parse_args(["w", *own, "--limits", "1,2,3,4"])
    assert (args.serve, args.fell_back) == ("float", False)


@pytest.mark.parametrize("there", [True, False])
def test_exact_match_requires_an_explicit_serve_option(tmp_path, monkeypatch, capsys, there):
    monkeypatch.setattr(end_to_end, "has_float_pair", lambda: there)
    message = "with --policy exact, also pass --serve integer"
    assert message in usage_error(capsys, "w", "--policy", "exact")
    assert message in usage_error(capsys, "w", *own_options(tmp_path)[:-2], "--policy", "exact")
    own = [*own_options(tmp_path)[:-2], "--policy", "exact", "--serve", "float"]
    assert end_to_end.parse_args(["w", *own]).serve == "float"


def test_own_commitment_without_serve_reports_the_missing_limits(tmp_path, capsys):
    message = "--policy clipped-gap (the default with --serve float) needs --limits"
    assert message in usage_error(capsys, "w", *own_options(tmp_path)[:-2])


def test_clipped_gap_without_serve_and_without_the_float_pair_is_an_error(tmp_path, monkeypatch):
    class Plugin:
        name = "integer_model"

    monkeypatch.setattr(end_to_end.importlib.metadata, "entry_points", lambda group: [Plugin])
    monkeypatch.setattr(end_to_end, "has_float_pair", lambda: False)
    args = end_to_end.parse_args([str(tmp_path / "w"), "--policy", "clipped-gap"])
    assert (args.serve, args.policy) == ("integer", "clipped-gap")
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert "has no integer_model_vllm.float_pair" in errors


@pytest.mark.parametrize("there", [True, False])
def test_a_run_that_fell_back_prints_a_note_first(tmp_path, monkeypatch, capsys, there):
    def stop_here(args, programs):
        raise end_to_end.StepError("far enough")

    monkeypatch.setattr(end_to_end, "has_float_pair", lambda: there)
    monkeypatch.setattr(end_to_end, "prepare_demo", stop_here)
    with pytest.raises(end_to_end.StepError, match="far enough"):
        end_to_end.run_steps(end_to_end.parse_args([str(tmp_path / "w")]), None)
    note = "Note: the installed integer model package has no integer_model_vllm.float_pair, which "
    assert capsys.readouterr().out.startswith(note) is not there


def test_a_run_ends_with_its_limits(tmp_path):
    synthetic, clipped_gap, hiding = end_to_end.closing_notes(end_to_end.parse_args(["w"]))
    assert synthetic == (
        "the demo's model is synthetic: a pass says nothing about any model that anyone serves, "
        "and nothing about weight hiding."
    )
    assert clipped_gap.startswith("the clipped-gap policy bounds a mean and a count ")
    assert hiding == (
        "The masking is built in and on by default; that it is enough to hide the weights is not "
        'proved for the programs. See docs/limits.md, "Weight hiding".'
    )
    own = ["w", *own_options(tmp_path)[:-2], "--serve", "integer"]
    assert end_to_end.closing_notes(end_to_end.parse_args(own)) == [hiding]


def test_options_after_two_dashes_are_for_vllm():
    args = end_to_end.parse_args(["w", "--serve", "float", "--", "--max-num-seqs", "2"])
    assert (args.serve, args.vllm_options) == ("float", ["--max-num-seqs", "2"])
    assert end_to_end.parse_args(["w"]).vllm_options == []


def test_an_option_after_two_dashes_is_not_parsed_by_the_script(tmp_path):
    args = end_to_end.parse_args(["w", "--", "--batch", "--serve", "integer"])
    assert (args.serve, args.demo) == ("float", True)
    assert args.vllm_options == ["--batch", "--serve", "integer"]


@pytest.mark.parametrize("option", ["--port", "--host=0.0.0.0", "--middleware", "--max-model-len"])
def test_rejects_vllm_options_that_the_script_sets(capsys, option):
    assert "cannot follow --: the script sets it" in usage_error(capsys, "w", "--", option, "1")


def test_paths_are_resolved_as_the_system_resolves_them(tmp_path, monkeypatch, capsys):
    (tmp_path / "real" / "inner").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real" / "inner")
    monkeypatch.chdir(tmp_path)
    assert end_to_end.parse_args(["w"]).work_dir == str(tmp_path / "w")
    assert end_to_end.parse_args(["real/inner/../w"]).work_dir == str(tmp_path / "real" / "w")
    build = end_to_end.parse_args(["w", "--build-dir", "link/../b"]).build_dir
    assert build == str(tmp_path / "real" / "b")
    assert "WORK_DIR leads through a symbolic link (%s): name" % shown(
        tmp_path / "link"
    ) in usage_error(capsys, "link/../w")


def test_own_commitment_needs_all_eight_options(capsys, tmp_path):
    own = own_options(tmp_path)
    assert own[-4:] == ["--end-token", "7", "--serve", "integer"]
    assert "also needed: --end-token" in usage_error(capsys, "w", *own[:-4])


def test_clipped_gap_on_own_commitment_needs_limits(capsys, tmp_path):
    argv = ["w", *own_options(tmp_path), "--policy", "clipped-gap"]
    assert "needs --limits" in usage_error(capsys, *argv)
    assert "needs --limits" in usage_error(capsys, *argv, "--limits", "1,2,3")
    assert end_to_end.parse_args([*argv, "--limits", "1,2,3,15"]).limits == "1,2,3,15"


@pytest.mark.parametrize(
    "argv, message",
    [
        ([""], "WORK_DIR must not be empty"),
        (["w", "--build-dir", ""], "--build-dir must not be empty"),
        (["w", "--max-tokens", "-5"], "not a positive integer"),
        (["w", "--batch", "0"], "not a positive integer"),
        (["w", "--start-timeout", "x"], "not a positive integer"),
        (["w", "--limits", "1,2,3,4"], "applies only with your own commitment"),
        (["w", "--serve", "float", "--policy", "exact"], "needs --policy clipped-gap"),
    ],
)
def test_rejects_bad_values(capsys, argv, message):
    assert message in usage_error(capsys, *argv)


def test_rejects_bad_values_for_own_commitment(capsys, tmp_path):
    own = own_options(tmp_path)
    assert "64 hex digits" in usage_error(capsys, "w", "--commitment-id", "ab", *own[2:])
    assert "separated by commas" in usage_error(capsys, "w", *own, "--end-token", "7,x")
    assert "only with --policy clipped-gap" in usage_error(capsys, "w", *own, "--limits", "1,2,3,4")
    assert end_to_end.parse_args(["w", "--commitment-id", ID.upper(), *own[2:]]).commitment_id == ID


# ---- Prerequisites ----


def test_prerequisites_create_nothing_and_list_what_is_missing(tmp_path):
    args = end_to_end.parse_args(
        [
            str(tmp_path / "w"),
            "--build-dir",
            str(tmp_path / "build"),
            "--verifier",
            str(tmp_path / "cp_verify"),
        ]
    )
    errors = "\n".join(end_to_end.check_prerequisites(args))
    for path in (
        tmp_path / "cp_verify",
        tmp_path / "build" / "cp_commit",
        tmp_path / "build" / "cp_prove",
        tmp_path / "build" / "cp_verify",
    ):
        assert "%s is missing or not executable" % path in errors
    assert list(tmp_path.iterdir()) == []


def test_your_own_commitment_needs_no_cp_commit_and_no_cp_verify_beside_cp_prove(tmp_path):
    own = own_options(tmp_path)
    args = end_to_end.parse_args([str(tmp_path / "w"), "--build-dir", str(tmp_path / "build"), *own])
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert "%s is missing or not executable" % (tmp_path / "build" / "cp_prove") in errors
    assert str(tmp_path / "build" / "cp_commit") not in errors
    assert str(tmp_path / "build" / "cp_verify") not in errors


def test_a_program_that_is_not_executable_is_reported(tmp_path):
    (tmp_path / "cp_verify").write_text("")
    args = end_to_end.parse_args([str(tmp_path / "w"), "--verifier", str(tmp_path / "cp_verify")])
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert "%s is missing or not executable" % (tmp_path / "cp_verify") in errors
    (tmp_path / "cp_verify").chmod(0o700)
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert str(tmp_path / "cp_verify") not in errors


def test_rejects_a_work_dir_that_exists(capsys, tmp_path):
    assert "%s already exists" % shown(tmp_path) in usage_error(capsys, str(tmp_path))
    (tmp_path / "link").symlink_to(tmp_path / "nowhere")
    assert "%s already exists" % shown(tmp_path / "link") in usage_error(
        capsys, str(tmp_path / "link")
    )


def test_own_directories_must_contain_what_the_programs_need(tmp_path):
    (tmp_path / "sec").mkdir()
    (tmp_path / "model").mkdir()
    args = end_to_end.parse_args([str(tmp_path / "w"), *own_options(tmp_path)])
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert "%s has no isk.txt, so it is not a secret key directory" % (tmp_path / "sec") in errors
    assert "%s has no tokenizer.json" % (tmp_path / "model") in errors
    assert "layout.txt" not in errors  # the parameter file itself is reported as missing
    (tmp_path / "policy.txt").write_text("")
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert "%s is missing: put the layout.txt" % (tmp_path / "layout.txt") in errors
    (tmp_path / "layout.txt").write_text("")
    (tmp_path / "sec" / "isk.txt").write_text("")
    (tmp_path / "model" / "tokenizer.json").write_text("{}")
    errors = "\n".join(end_to_end.check_prerequisites(args))
    assert not any(name in errors for name in ("isk.txt", "tokenizer.json", "layout.txt"))


@pytest.mark.parametrize(
    "argv, needs_it",
    [
        (["--serve", "float"], True),
        (["--serve", "integer", "--policy", "clipped-gap"], True),
        (["--serve", "integer"], False),
    ],
)
def test_the_demo_under_clipped_gap_needs_the_float_pair(tmp_path, monkeypatch, argv, needs_it):
    class Plugin:
        name = "integer_model"

    monkeypatch.setattr(end_to_end.importlib.metadata, "entry_points", lambda group: [Plugin])
    args = end_to_end.parse_args([str(tmp_path / "w"), *argv])
    for there in (True, False):
        monkeypatch.setattr(end_to_end, "has_float_pair", lambda: there)
        errors = "\n".join(end_to_end.check_prerequisites(args))
        assert ("has no integer_model_vllm.float_pair" in errors) == (needs_it and not there)


@pytest.mark.parametrize("there", [True, False])
def test_has_float_pair_looks_into_the_installed_package(tmp_path, monkeypatch, there):
    (tmp_path / "integer_model_vllm").mkdir()
    (tmp_path / "integer_model_vllm" / "__init__.py").write_text("")
    if there:
        (tmp_path / "integer_model_vllm" / "float_pair.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "integer_model_vllm", raising=False)
    monkeypatch.delitem(sys.modules, "integer_model_vllm.float_pair", raising=False)
    try:
        assert has_float_pair() is there
    finally:
        sys.modules.pop("integer_model_vllm", None)


def test_has_float_pair_is_false_if_the_package_cannot_be_imported(monkeypatch):
    def not_found(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(end_to_end.importlib.util, "find_spec", not_found)
    assert has_float_pair() is False


def test_help_works_under_python_oo():
    argv = [sys.executable, "-OO", "-B", SCRIPT, "--help"]
    got = subprocess.run(argv, capture_output=True, text=True, env=CHILD_ENV)
    assert (got.returncode, got.stderr) == (0, "") and got.stdout.startswith("usage: ")


def test_help_works_before_the_modules_are_installed(tmp_path):
    # -S, no PYTHONPATH: only the standard library
    argv = [sys.executable, "-S", "-B", SCRIPT]
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    got = subprocess.run([*argv, "--help"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert (got.returncode, got.stderr) == (0, "") and got.stdout.startswith("usage: ")
    got = subprocess.run([*argv, "w"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert got.returncode == 2
    assert got.stderr.startswith("error: check_proof is not installed (pip install ")
    assert os.listdir(tmp_path) == []


def test_missing_prerequisites_exit_with_code_2(tmp_path):
    argv = [sys.executable, "-B", SCRIPT, str(tmp_path / "w"), "--build-dir", str(tmp_path)]
    end_to_end.parse_args(argv[3:])  # in this process too: see in_an_empty_directory()
    got = subprocess.run(argv, capture_output=True, text=True, env=CHILD_ENV)
    assert got.returncode == 2
    assert got.stderr.startswith("error: %s/cp_prove is missing" % tmp_path)
    assert not (tmp_path / "w").exists()


# ---- Programs ----


def test_run_returns_the_exit_code_and_rejects_an_unexpected_one(programs, tmp_path):
    exit_3 = [sys.executable, "-c", "print('hello'); raise SystemExit(3)"]
    assert programs.run("a", exit_3, 60, expected=(3,)) == 3
    assert (tmp_path / "a.log").read_text() == "hello\n"
    with pytest.raises(end_to_end.StepError, match="b exited with code 3, expected 0"):
        programs.run("b", exit_3, 60)
    assert programs.running == []


@pytest.mark.parametrize("kind", ["file", "link", "dangling link"])
def test_a_log_is_always_a_new_file(programs, tmp_path, kind):
    (tmp_path / "target").write_text("kept")
    if kind == "file":
        (tmp_path / "a.log").write_text("kept")
    else:
        (tmp_path / "a.log").symlink_to(tmp_path / ("target" if kind == "link" else "nowhere"))
    with pytest.raises(
        end_to_end.StepError, match="the log file %s already exists" % (tmp_path / "a.log")
    ):
        programs.run("a", [sys.executable, "-c", "print('new')"], 60)
    assert (tmp_path / "target").read_text() == "kept" and not (tmp_path / "nowhere").exists()
    assert programs.running == []


def test_a_directory_put_in_the_place_of_the_log_directory_gets_no_log(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "target").write_text("kept")
    programs = end_to_end.Programs(str(tmp_path / "logs"))
    (tmp_path / "logs").rename(tmp_path / "moved")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.log").symlink_to(tmp_path / "target")
    assert programs.run("a", [sys.executable, "-c", "print('new')"], 60) == 0
    assert (tmp_path / "moved" / "a.log").read_text() == "new\n"
    assert (tmp_path / "target").read_text() == "kept"
    assert end_to_end.last_line(programs.log_to_read("a")) == "new"


def test_run_reports_a_program_that_cannot_be_started(programs, tmp_path):
    with pytest.raises(end_to_end.StepError, match="cannot start"):
        programs.run("none", [str(tmp_path / "none")], 60)
    assert programs.running == []


def test_run_stops_a_program_that_takes_too_long(programs):
    with pytest.raises(end_to_end.StepError, match="did not finish within 1 s;"):
        programs.run("slow", SLEEP, 1)
    assert programs.running == []


def start_with_child(programs, tmp_path, child, then):
    """Start a parent that starts a child, reads a line from it, prints its PID and runs `then`."""
    code = (
        "import subprocess, sys, time; p = subprocess.Popen(%r, stdout=subprocess.PIPE); "
        "p.stdout.readline(); print(p.pid, flush=True); %s" % (child, then)
    )
    process = programs.start("parent", [sys.executable, "-c", code])
    wait_for(lambda: (tmp_path / "parent.log").read_text().strip())
    return process, int((tmp_path / "parent.log").read_text())


def test_stop_all_kills_a_child_that_outlives_its_parent(programs, tmp_path, monkeypatch):
    monkeypatch.setattr(end_to_end, "STOP_GRACE_S", 1)
    process, child = start_with_child(programs, tmp_path, IGNORES_SIGTERM, "time.sleep(60)")
    programs.stop_all()
    assert process.returncode == -signal.SIGTERM
    wait_for(lambda: not is_running(child))


def test_reap_kills_a_child_that_an_exited_program_left_behind(programs, tmp_path):
    process, child = start_with_child(programs, tmp_path, IGNORES_SIGTERM, "pass")
    wait_for(lambda: programs.exited(process))
    assert is_running(child)
    assert programs.reap(process) == 0
    wait_for(lambda: not is_running(child))


def test_stop_all_kills_a_program_that_ignores_sigterm(programs, tmp_path, monkeypatch):
    monkeypatch.setattr(end_to_end, "STOP_GRACE_S", 1)
    process = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    programs.stop_all()
    assert process.returncode == -signal.SIGKILL


def test_stop_all_gives_a_program_the_grace_that_it_was_started_with(
    programs, tmp_path, monkeypatch, capsys
):
    slow = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), sys.exit(5))); "
        "print('up', flush=True); time.sleep(60)"
    )
    monkeypatch.setattr(end_to_end, "STOP_GRACE_S", 0.1)
    process = programs.start("slow", [sys.executable, "-c", slow], grace_s=30)
    wait_for(lambda: (tmp_path / "slow.log").read_text())
    programs.stop_all()
    assert process.returncode == 5
    assert capsys.readouterr().out == (
        "   Still waiting for %s to exit after SIGTERM, for up to 30 s. Ctrl-C, twice, kills them "
        "at once, and then they wipe and remove nothing.\n" % os.path.basename(sys.executable)
    )


def test_stop_all_waits_for_a_child_that_exits_after_its_parent(programs, tmp_path):
    slow_child = [
        sys.executable,
        "-c",
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), open(%r, 'x'), sys.exit())); "
        "print('up', flush=True); time.sleep(60)" % str(tmp_path / "exited by itself"),
    ]
    process, child = start_with_child(programs, tmp_path, slow_child, "time.sleep(60)")
    programs.stop_all()
    assert process.returncode == -signal.SIGTERM
    assert (tmp_path / "exited by itself").exists() and not is_running(child)


def test_living_groups_reads_any_process_name(programs, tmp_path, monkeypatch):
    # The kernel cuts a name to 15 bytes, even in the middle of a character; it can hold ")" too.
    for pid, name, state, group in (
        (7, b"a" * 14 + b"\xc3", b"S", 7),
        (8, b"a) Z 1 99 (", b"R", 8),
        (9, b"exited", b"Z", 9),
        (10, b"\xff\xfe", b"D", 10),
    ):
        (tmp_path / str(pid)).mkdir()
        stat_line = b"%d (%s) %s 1 %d %d 0 -1 4194304\n" % (pid, name, state, group, group)
        (tmp_path / str(pid) / "stat").write_bytes(stat_line)
    (tmp_path / "11").mkdir()  # it exited between the listing and the reading
    (tmp_path / "12").mkdir()
    (tmp_path / "12" / "stat").write_bytes(b"not such a line")
    (tmp_path / "self").mkdir()
    monkeypatch.setattr(end_to_end, "PROC", str(tmp_path))
    assert programs.living_groups() == {7, 8, 10}


def test_a_closed_standard_output_does_not_end_a_stop(programs, tmp_path, monkeypatch):
    def closed(text):
        raise BrokenPipeError("standard output is closed")

    monkeypatch.setattr(end_to_end, "say", closed)
    monkeypatch.setattr(end_to_end, "STOP_GRACE_S", 0.1)
    process = programs.start("stubborn", IGNORES_SIGTERM, grace_s=1)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    programs.stop_all()
    assert (process.returncode, programs.running) == (-signal.SIGKILL, [])


def test_a_program_that_proc_does_not_show_still_counts_as_alive(programs, tmp_path, monkeypatch):
    process = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    monkeypatch.setattr(programs, "living_groups", set)
    assert programs.alive([process]) == [process]
    programs.kill_now = True
    programs.stop_all()
    assert process.returncode == -signal.SIGKILL


def test_an_inherited_ignored_sigchld_does_not_let_the_kernel_reap_a_program(tmp_path):
    before = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        programs = end_to_end.Programs(str(tmp_path))
        assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
        process = programs.start("quick", [sys.executable, "-c", "raise SystemExit(5)"])
        wait_for(lambda: programs.exited(process))
        # It has exited and is not reaped, so its PID is still its own.
        assert os.path.exists("/proc/%d" % process.pid) and not is_running(process.pid)
        assert programs.reap(process) == 5
    finally:
        signal.signal(signal.SIGCHLD, before)


def test_without_proc_a_stop_still_ends_a_program_and_what_it_left_behind(
    programs, tmp_path, monkeypatch
):
    monkeypatch.setattr(end_to_end, "STOP_GRACE_S", 1)
    stubborn = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    parent, child = start_with_child(programs, tmp_path, IGNORES_SIGTERM, "pass")
    wait_for(lambda: programs.exited(parent))
    monkeypatch.setattr(end_to_end, "PROC", str(tmp_path / "no such directory"))
    assert programs.living_groups() is None
    programs.stop_all()
    assert (stubborn.returncode, parent.returncode, programs.running) == (-signal.SIGKILL, 0, [])
    wait_for(lambda: not is_running(child))


def test_no_signal_goes_to_a_group_that_has_exited(programs, monkeypatch):
    process = programs.start("quick", [sys.executable, "-c", "pass"])
    wait_for(lambda: programs.exited(process))
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("a signal was sent"))
    programs.stop_all()
    assert (process.returncode, programs.running) == (0, [])


def test_run_lets_what_a_program_left_behind_exit_by_itself(programs, tmp_path):
    slow_child = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), open(%r, 'x'), sys.exit())); "
        "print('up', flush=True); time.sleep(60)" % str(tmp_path / "exited by itself")
    )
    parent = (
        "import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', %r], "
        "stdout=subprocess.PIPE); p.stdout.readline()" % slow_child
    )
    assert programs.run("parent", [sys.executable, "-c", parent], 60) == 0
    assert (tmp_path / "exited by itself").exists()


@pytest.mark.parametrize(
    "arriving",
    [
        # As if a service manager sent SIGTERM and SIGHUP, and the user then pressed Ctrl-C.
        {2: signal.SIGTERM, 3: signal.SIGHUP, 4: signal.SIGTERM, 6: signal.SIGINT},
        {3: signal.SIGINT, 6: signal.SIGINT},  # one Ctrl-C alone does not end the wait
    ],
)
def test_only_ctrl_c_after_an_earlier_signal_ends_the_wait(
    programs, tmp_path, monkeypatch, arriving
):
    programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    sleep, killpg, passes, sent = time.sleep, os.killpg, [], []

    def signal_at_some_passes(seconds):
        passes.append(seconds)
        if len(passes) in arriving:
            programs.on_signal(arriving[len(passes)], None)
        sleep(0.01)

    def recording_killpg(group, number):
        sent.append((len(passes), number))
        killpg(group, number)

    monkeypatch.setattr(time, "sleep", signal_at_some_passes)
    monkeypatch.setattr(os, "killpg", recording_killpg)
    with pytest.raises(end_to_end.Interrupted) as interrupt:
        programs.stop_all()
    assert sent == [(0, signal.SIGTERM), (6, signal.SIGKILL)]
    assert interrupt.value.args == (arriving[min(arriving)],)  # the first signal


def test_a_program_leaves_the_list_before_it_is_reaped(programs):
    # Otherwise an interrupt between the two could lead to a signal for a PID that is free again.
    process = programs.start("quick", [sys.executable, "-c", "pass"])
    wait_for(lambda: programs.exited(process))
    wait, listed = process.wait, []

    def recording_wait(timeout):
        listed.append(process in programs.running)
        return wait(timeout)

    process.wait = recording_wait
    assert programs.reap(process) == 0 and listed == [False]


def test_an_interrupt_while_run_waits_for_what_was_left_behind_keeps_the_program_tracked(
    programs, monkeypatch
):
    parent = (
        "import subprocess; p = subprocess.Popen(%r, stdout=subprocess.PIPE); p.stdout.readline()"
        % (IGNORES_SIGTERM,)
    )
    wait_for_groups = programs.wait_for_groups

    def interrupted_wait(processes):
        monkeypatch.setattr(time, "sleep", lambda s: programs.on_signal(signal.SIGINT, None))
        wait_for_groups(processes)

    monkeypatch.setattr(programs, "wait_for_groups", interrupted_wait)
    with pytest.raises(end_to_end.Interrupted):
        programs.run("parent", [sys.executable, "-c", parent], 60)
    assert len(programs.running) == 1  # so that main()'s stop_all() still ends its group


def test_cp_prove_gets_the_long_grace(programs, tmp_path):
    args = with_fake_cp_prove(tmp_path, "exit 0")
    end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    assert programs.graces == [end_to_end.GPU_GRACE_S]


def test_stop_and_wait_waits_for_a_slow_exit(programs, tmp_path):
    slow = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), sys.exit(5))); "
        "print('up', flush=True); time.sleep(60)"
    )
    process = programs.start("slow", [sys.executable, "-c", slow])
    wait_for(lambda: (tmp_path / "slow.log").read_text())
    programs.stop_and_wait(60)
    assert (process.returncode, programs.running) == (5, [])


def test_stop_and_wait_waits_for_a_child_that_exits_after_its_parent(programs, tmp_path):
    slow_child = [
        sys.executable,
        "-c",
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(1), open(%r, 'x'), sys.exit())); "
        "print('up', flush=True); time.sleep(60)" % str(tmp_path / "exited by itself"),
    ]
    process, child = start_with_child(programs, tmp_path, slow_child, "time.sleep(60)")
    programs.stop_and_wait(60)
    assert process.returncode == -signal.SIGTERM
    assert (tmp_path / "exited by itself").exists() and not is_running(child)


def test_stop_and_wait_gives_up_on_a_program_that_does_not_exit(programs, tmp_path):
    process = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    with pytest.raises(end_to_end.StepError, match="did not exit within 1 s after SIGTERM"):
        programs.stop_and_wait(1)
    assert programs.running == [process]  # so that main()'s stop_all() still kills it


def test_kill_now_skips_the_grace_period(programs, tmp_path):
    process = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    programs.kill_now = True
    began = time.monotonic()
    programs.stop_all()
    assert process.returncode == -signal.SIGKILL
    assert time.monotonic() - began < end_to_end.STOP_GRACE_S / 2


def test_an_interrupt_during_start_is_held_back_until_the_program_is_tracked(programs, monkeypatch):
    popen = subprocess.Popen

    def interrupted_popen(*args, **kwargs):
        process = popen(*args, **kwargs)
        programs.on_signal(signal.SIGTERM, None)  # as if two signals arrived here
        programs.on_signal(signal.SIGHUP, None)
        return process

    monkeypatch.setattr(subprocess, "Popen", interrupted_popen)
    with pytest.raises(end_to_end.Interrupted) as interrupt:
        programs.start("sleep", SLEEP)
    assert len(programs.running) == 1
    assert interrupt.value.args == (signal.SIGTERM,)  # the first signal


def test_ctrl_c_twice_while_stop_all_waits_ends_the_wait_and_is_raised_afterwards(
    programs, tmp_path, monkeypatch
):
    process = programs.start("stubborn", IGNORES_SIGTERM)
    wait_for(lambda: (tmp_path / "stubborn.log").read_text())
    # as if Ctrl-C arrived at each sleep of the wait: the second one sets kill_now
    monkeypatch.setattr(time, "sleep", lambda s: programs.on_signal(signal.SIGINT, None))
    began = time.monotonic()
    with pytest.raises(end_to_end.Interrupted):
        programs.stop_all()
    assert time.monotonic() - began < end_to_end.STOP_GRACE_S / 2
    assert (process.returncode, programs.running) == (-signal.SIGKILL, [])
    programs.stop_all()  # main()'s last call does not raise again


def test_only_the_first_interrupt_raises(programs):
    with pytest.raises(end_to_end.Interrupted):
        programs.on_signal(signal.SIGINT, None)
    programs.on_signal(signal.SIGINT, None)
    assert programs.kill_now


# ---- main() ----


@pytest.fixture
def ready(tmp_path, monkeypatch):
    """Turn off main()'s check of the prerequisites, which includes the early look at the
    directories above WORK_DIR; return main()'s arguments. Afterwards undo main()'s changes."""
    monkeypatch.setattr(end_to_end, "check_prerequisites", lambda args: [])
    handlers = {
        n: signal.getsignal(n)
        for n in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGCHLD)
    }
    mask, environment = os.umask(0o077), dict(os.environ)
    yield [str(tmp_path / "w")]
    os.umask(mask)
    os.environ.clear()
    os.environ.update(environment)
    for number, handler in handlers.items():
        signal.signal(number, handler)


@pytest.mark.parametrize(
    "outcome, code, last",
    [
        ((True, []), 0, "RESULT: every step gave the expected result"),
        (
            (True, ["the negative control"]),
            0,
            "RESULT: every step that ran gave the expected result; "
            "not checked: the negative control",
        ),
        ((False, []), 1, "RESULT: failed"),
    ],
)
def test_main_reports_the_result(ready, monkeypatch, capsys, outcome, code, last):
    monkeypatch.setattr(end_to_end, "run_steps", lambda args, programs: outcome)
    assert end_to_end.main(ready) == code
    assert capsys.readouterr().out.splitlines()[-1] == last


@pytest.mark.parametrize(
    "error, text",
    [
        (end_to_end.StepError("a step failed"), "error: a step failed\n"),
        (KeyError("lt"), "error: KeyError: 'lt'\n"),
    ],
)
def test_main_turns_an_error_into_exit_code_1(ready, monkeypatch, capsys, error, text):
    def fail(args, programs):
        raise error

    monkeypatch.setattr(end_to_end, "run_steps", fail)
    assert end_to_end.main(ready) == 1
    assert capsys.readouterr().err == text


def test_an_interrupt_stops_the_programs_and_exits_with_130(ready, monkeypatch):
    started = []

    def interrupted(args, programs):
        started.append(programs.start("sleep", SLEEP))
        os.kill(os.getpid(), signal.SIGINT)  # this process itself
        time.sleep(30)

    monkeypatch.setattr(end_to_end, "run_steps", interrupted)
    assert end_to_end.main(ready) == 130
    assert started[0].returncode == -signal.SIGTERM


def test_an_error_that_takes_an_interrupts_place_does_not_change_the_exit_code(ready, monkeypatch):
    def interrupted_then_failing(args, programs):
        try:
            programs.on_signal(signal.SIGTERM, None)
        finally:
            raise end_to_end.StepError("a step failed while the interrupt was passing")

    monkeypatch.setattr(end_to_end, "run_steps", interrupted_then_failing)
    assert end_to_end.main(ready) == 128 + signal.SIGTERM


def test_an_interrupt_during_the_last_stop_gives_the_signals_exit_code(ready, monkeypatch):
    def fail(args, programs):
        programs.start("stubborn", IGNORES_SIGTERM)
        wait_for(lambda: os.path.getsize(programs.log_path("stubborn")))
        # as if SIGTERM, and then Ctrl-C, arrived during main()'s last stop
        arriving = itertools.chain([signal.SIGTERM], itertools.repeat(signal.SIGINT))
        monkeypatch.setattr(time, "sleep", lambda s: programs.on_signal(next(arriving), None))
        raise end_to_end.StepError("a step failed")

    monkeypatch.setattr(end_to_end, "run_steps", fail)
    assert end_to_end.main(ready) == 128 + signal.SIGTERM


def test_main_sets_sigchld_to_its_default_before_the_steps(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(signal, "signal", lambda number, handler: calls.append((number, handler)))

    def is_default(args, programs):
        return (signal.SIGCHLD, signal.SIG_DFL) in calls, []

    monkeypatch.setattr(end_to_end, "run_steps", is_default)
    assert end_to_end.main(ready) == 0


def test_a_work_dir_that_cannot_be_created_exits_with_code_2(ready, tmp_path, capsys):
    (tmp_path / "file").write_text("")
    assert end_to_end.main([str(tmp_path / "file" / "w")]) == 2
    assert capsys.readouterr().err.startswith("error: cannot create ")


@pytest.mark.parametrize(
    "mode, refused",
    [(0o777, True), (0o775, True), (0o2777, True), (0o1777, False), (0o755, False)],
)
def test_a_directory_above_that_another_user_can_change_is_an_error(tmp_path, mode, refused):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared").chmod(mode)
    args = end_to_end.parse_args([str(tmp_path / "shared" / "w")])
    error = "WORK_DIR cannot be used: %s can be written by its group or by others (mode %04o)" % (
        tmp_path / "shared",
        mode,
    )
    errors = end_to_end.check_prerequisites(args)
    assert any(text.startswith(error) for text in errors) is refused
    assert any(text.startswith("WORK_DIR cannot be used") for text in errors) is refused
    why = ". So that no other user can lead this run's files elsewhere, choose a new name in an "
    assert any(why in text for text in errors) is refused
    example = 'For example: W="$(mktemp -d -p /tmp)", and then "${W:?}/run" as WORK_DIR.'
    assert any(text.endswith(example) for text in errors) is refused
    assert os.listdir(tmp_path / "shared") == []


@pytest.mark.root_only
@only_root
def test_a_directory_above_that_another_user_owns_is_found(tmp_path):
    (tmp_path / "theirs").mkdir(mode=0o755)
    os.chown(tmp_path / "theirs", OTHER_UID, -1)
    fault = "%s belongs to another user (uid %d)" % (tmp_path / "theirs", OTHER_UID)
    assert end_to_end.fault_above(str(tmp_path / "theirs" / "w")) == fault


@pytest.mark.parametrize(
    "euid, sudo_uid, trusted",
    [
        (0, "1000", {0, 1000}),  # under sudo
        (0, None, {0}),
        (0, "", {0}),
        (0, "1000 ", {0}),
        (0, "1000\n", {0}),
        (0, "-1", {0}),
        (0, "01000", {0}),
        (0, "0", {0}),
        (0, "4294967294", {0, 4294967294}),
        (0, "4294967295", {0}),  # no user's ID
        (0, "1" * 11, {0}),
        (0, "9" * 5000, {0}),
        (0, "\u0661\u0660\u0660\u0660", {0}),  # digits, but not ASCII
        (1001, "1000", {0, 1001}),  # only root's SUDO_UID counts
    ],
)
def test_root_also_trusts_the_account_that_ran_sudo(monkeypatch, euid, sudo_uid, trusted):
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    monkeypatch.setattr(os, "getuid", lambda: euid)
    monkeypatch.delenv("SUDO_UID", raising=False)
    if sudo_uid is not None:
        monkeypatch.setenv("SUDO_UID", sudo_uid)
    assert end_to_end.trusted_owners() == trusted


@pytest.mark.parametrize(
    "uid, euid, trusted",
    [
        (1001, 0, {0}),  # as under a set-user-ID interpreter
        (0, 1001, {0, 1001}),
    ],
)
def test_sudo_uid_counts_only_for_a_caller_that_is_root_by_both_its_ids(
    monkeypatch, uid, euid, trusted
):
    monkeypatch.setattr(os, "getuid", lambda: uid)
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    monkeypatch.setenv("SUDO_UID", "1000")
    assert end_to_end.trusted_owners() == trusted


@pytest.mark.root_only
@only_root
def test_under_sudo_the_work_dir_can_be_below_the_users_own_directory(
    ready, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(end_to_end, "run_steps", lambda args, programs: (True, []))
    (tmp_path / "home").mkdir(mode=0o755)
    os.chown(tmp_path / "home", SUDO_UID, -1)
    monkeypatch.delenv("SUDO_UID", raising=False)
    assert end_to_end.main([str(tmp_path / "home" / "refused")]) == 2
    assert "Note: " not in capsys.readouterr().out
    monkeypatch.setenv("SUDO_UID", str(SUDO_UID))
    assert end_to_end.fault_above(str(tmp_path / "home" / "w")) is None
    assert end_to_end.main([str(tmp_path / "home" / "w")]) == 0
    assert os.listdir(tmp_path / "home") == ["w"]
    above = reversed((tmp_path / "home" / "w").parents)  # from "/" down
    assert capsys.readouterr().out.startswith(
        "Note: %s belongs to the account in SUDO_UID (uid %d), and this run trusts that account "
        "like root: it could lead the run's files elsewhere. To rule that out, choose a WORK_DIR "
        "with only root's directories above it. For example: "
        'W="$(sudo mktemp -d -p /tmp)", and then "${W:?}/run" as WORK_DIR.\n'
        % (next(path for path in above if path.stat().st_uid == SUDO_UID), SUDO_UID)
    )


def test_the_note_on_sudo_rests_on_one_look(ready, monkeypatch, capsys):
    monkeypatch.setattr(end_to_end, "run_steps", lambda args, programs: (True, []))
    answers = iter([("/home/user", 1000), None])  # as if the directory changed hands in between
    monkeypatch.setattr(end_to_end, "trusted_like_root", lambda path: next(answers))
    assert end_to_end.main(ready) == 0
    assert capsys.readouterr().out.startswith("Note: /home/user belongs to the account in SUDO_UID")


def test_a_run_below_the_users_and_roots_directories_prints_no_note_on_sudo(
    ready, monkeypatch, capsys
):
    if end_to_end.trusted_like_root(ready[0]):
        pytest.skip("pytest's temporary directory is below a directory of the account in SUDO_UID")
    monkeypatch.setattr(end_to_end, "run_steps", lambda args, programs: (True, []))
    monkeypatch.setenv("SUDO_UID", str(SUDO_UID))
    assert end_to_end.main(ready) == 0
    assert "SUDO_UID" not in capsys.readouterr().out


@pytest.mark.root_only
@only_root
@pytest.mark.parametrize(
    "owner, mode, fault",
    [
        (OTHER_UID, 0o755, " belongs to another user (uid %d)" % OTHER_UID),  # a third account
        (
            SUDO_UID,
            0o775,
            " can be written by its group or by others (mode 0775) and is not sticky",
        ),
        (
            SUDO_UID,
            0o777,
            " can be written by its group or by others (mode 0777) and is not sticky",
        ),
    ],
)
def test_what_the_trust_in_the_account_that_ran_sudo_does_not_open(
    ready, monkeypatch, tmp_path, capsys, owner, mode, fault
):
    (tmp_path / "home").mkdir()
    (tmp_path / "home").chmod(mode)
    os.chown(tmp_path / "home", owner, -1)
    monkeypatch.setenv("SUDO_UID", str(SUDO_UID))
    assert end_to_end.fault_above(str(tmp_path / "home" / "w")) == str(tmp_path / "home") + fault
    assert end_to_end.main([str(tmp_path / "home" / "w")]) == 2  # `ready` turns off the early look
    assert str(tmp_path / "home") + fault in capsys.readouterr().err
    assert os.listdir(tmp_path / "home") == []


def test_a_link_in_the_place_of_a_directory_above_is_found(tmp_path):
    (tmp_path / "elsewhere").mkdir(mode=0o755)
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere")
    fault = "%s is a link or not a directory" % (tmp_path / "link")
    assert end_to_end.fault_above(str(tmp_path / "link" / "w")) == fault
    assert end_to_end.fault_above(str(tmp_path / "elsewhere" / "w")) is None


@pytest.mark.parametrize(
    "planted, reason",
    [
        ("link", ".*/planted is a link or not a directory"),
        (
            "open directory",
            ".*/planted can be written by its group or by others .mode 0777. and .*",
        ),
        pytest.param(
            "their directory",
            ".*/planted belongs to another user .uid %d." % OTHER_UID,
            marks=[pytest.mark.root_only, only_root],
        ),
    ],
)
def test_what_is_planted_above_the_work_dir_after_the_early_look_stops_the_run(
    ready, monkeypatch, tmp_path, capsys, planted, reason
):
    parse_args = end_to_end.parse_args
    (tmp_path / "elsewhere").mkdir()

    def parse_then_plant(argv):  # `ready` turns off `check_prerequisites`, the early look
        args = parse_args(argv)
        if planted == "link":
            (tmp_path / "planted").symlink_to(tmp_path / "elsewhere")
        elif planted == "open directory":
            (tmp_path / "planted").mkdir()
            (tmp_path / "planted").chmod(0o777)
        else:
            (tmp_path / "planted").mkdir(mode=0o755)
            os.chown(tmp_path / "planted", OTHER_UID, -1)
        return args

    monkeypatch.setattr(end_to_end, "parse_args", parse_then_plant)
    work_dir = tmp_path / "planted" / "deeper" / "w"
    assert end_to_end.main([str(work_dir)]) == 2
    assert re.fullmatch(
        "error: cannot create %s: (%s)\n" % (work_dir, reason), capsys.readouterr().err
    )
    assert os.listdir(tmp_path / "planted") == os.listdir(tmp_path / "elsewhere") == []


def test_the_walk_looks_at_the_work_dirs_own_parent(ready, tmp_path, capsys):
    (tmp_path / "open").mkdir()
    (tmp_path / "open").chmod(0o777)
    assert end_to_end.main([str(tmp_path / "open" / "w")]) == 2  # `ready` turns off the early look
    assert "%s can be written by its group or by others" % (tmp_path / "open") in (
        capsys.readouterr().err
    )
    assert os.listdir(tmp_path / "open") == []


def test_the_walk_refuses_a_directory_that_cannot_be_opened(ready, monkeypatch, tmp_path, capsys):
    (tmp_path / "closed").mkdir()
    real_open, holds = os.open, tmp_path.stat()

    def closed_to_this_user(path, flags, *more, **keywords):
        if path == "closed":  # the one in tmp_path, not a directory of that name higher up
            above = os.fstat(keywords["dir_fd"])
            if (above.st_dev, above.st_ino) == (holds.st_dev, holds.st_ino):
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))
        return real_open(path, flags, *more, **keywords)

    monkeypatch.setattr(os, "open", closed_to_this_user)
    work_dir = tmp_path / "closed" / "w"
    assert end_to_end.main([str(work_dir)]) == 2
    assert capsys.readouterr().err == "error: cannot create %s: %s: Permission denied\n" % (
        work_dir,
        tmp_path / "closed",
    )
    assert os.listdir(tmp_path / "closed") == []


def test_a_file_in_the_place_of_a_directory_above_gives_a_plain_error(tmp_path, capsys, request):
    (tmp_path / "file").touch()
    work_dir = str(tmp_path / "file" / "deeper" / "w")
    errors = end_to_end.check_prerequisites(end_to_end.parse_args([work_dir]))
    fault = "WORK_DIR cannot be used: %s is a link or not a directory. " % (tmp_path / "file")
    assert any(text.startswith(fault) for text in errors)
    assert not any("another user" in text for text in errors)
    request.getfixturevalue("ready")  # from here on without the early look
    assert end_to_end.main([work_dir]) == 2
    assert (
        capsys.readouterr().err
        == "error: cannot create %s: %s is a link or not a directory\n"
        % (
            work_dir,
            tmp_path / "file",
        )
    )


def test_rejects_a_work_dir_that_leads_through_a_link(capsys, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere")
    error = usage_error(capsys, str(tmp_path) + "//link/w")
    assert (
        "WORK_DIR leads through a symbolic link (%s): name a place that has no link on its way"
        % shown(tmp_path / "link")
    ) in error
    assert "elsewhere" not in error  # a person in a hurry would copy the resolved path
    real = str(tmp_path / "elsewhere" / "w")
    assert end_to_end.parse_args(["/" + real]).work_dir == real  # "//" at the start is no link


def test_rejects_a_secret_key_dir_that_leads_through_a_link(capsys, tmp_path):
    # cp_prove refuses a link on the way to -isk, so the script must not pass it the resolved path.
    (tmp_path / "real" / "sec").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    error = usage_error(capsys, str(tmp_path / "w"), *own_options(tmp_path / "link"))
    assert (
        "--secret-key-dir leads through a symbolic link (%s): name the directory by its own path, "
        "with no link on the way; do not copy or move it for this" % shown(tmp_path / "link")
    ) in error
    assert str(tmp_path / "real") not in error  # a person in a hurry would copy the resolved path
    args = end_to_end.parse_args([str(tmp_path / "w"), *own_options(tmp_path / "real")])
    assert args.secret_key_dir == str(tmp_path / "real" / "sec")


def test_rejects_a_relative_secret_key_dir(capsys, tmp_path, monkeypatch):
    # After "cd link", os.getcwd() gives the resolved path: a relative name would hide the link.
    (tmp_path / "real" / "sec").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    monkeypatch.chdir(tmp_path / "link")
    monkeypatch.setenv("PWD", str(tmp_path / "link"))
    own = own_options(tmp_path / "real")
    own[own.index("--secret-key-dir") + 1] = "sec"
    error = usage_error(capsys, str(tmp_path / "w"), *own)
    assert '--secret-key-dir must be a full path from /, for example "$PWD/isk"' in error
    own[own.index("--secret-key-dir") + 1] = os.environ["PWD"] + "/sec"  # the advice, obeyed
    assert "--secret-key-dir leads through a symbolic link (%s): name" % shown(
        tmp_path / "link"
    ) in usage_error(capsys, str(tmp_path / "w"), *own)


def test_a_work_dir_that_appears_before_it_is_made_stops_the_run(ready, monkeypatch, capsys):
    parse_args = end_to_end.parse_args

    def parse_then_plant(argv):
        args = parse_args(argv)
        os.mkdir(args.work_dir, 0o700)
        return args

    monkeypatch.setattr(end_to_end, "parse_args", parse_then_plant)
    assert end_to_end.main(ready) == 2
    assert capsys.readouterr().err == "error: cannot create %s: File exists\n" % ready[0]
    assert os.listdir(ready[0]) == []


@pytest.mark.parametrize("name", ["with a space", "a+b", "caf\u00e9", "a\nb"])
def test_rejects_a_work_dir_whose_path_cp_prove_would_not_accept(capsys, tmp_path, name):
    assert "may contain only letters" in usage_error(capsys, str(tmp_path / name))


def test_what_the_caller_typed_is_shown_without_control_characters(capsys, tmp_path):
    (tmp_path / "a\x1b[2Jb").mkdir()
    for words, argv in (
        ("already exists", [str(tmp_path / "a\x1b[2Jb")]),
        ("may contain only letters", [str(tmp_path / "new\x1b[2Jb")]),
        ("cannot contain", ["w", "--cp-prove-options=-isk=\x1b[2J"]),
    ):
        error = usage_error(capsys, *argv)
        assert words in error and "\x1b" not in error and "\\x1b[2J" in error


def test_the_work_dirs_path_has_at_most_380_characters(capsys, tmp_path):
    top = os.path.realpath(tmp_path)
    if len(top) > 370:
        pytest.skip("the path of pytest's temporary directory is too long")
    rest = ["d"] * (380 - len(top) - 1)
    rest[100:-1:101] = "/" * len(rest[100:-1:101])  # no name has more than 101 characters
    longest = top + "/" + "".join(rest)
    assert len(longest) == 380
    assert end_to_end.parse_args([longest]).work_dir == longest
    assert "at most 380 characters" in usage_error(capsys, longest + "w")


def test_the_provers_programs_get_a_home_and_a_tmpdir_below_the_work_dir(
    ready, monkeypatch, tmp_path
):
    seen = {}

    def look(args, programs):
        environment = end_to_end.prover_env(args)
        seen.update({name: environment[name] for name in ("HOME", "TMPDIR")})
        return True, []

    monkeypatch.setattr(end_to_end, "run_steps", look)
    assert end_to_end.main([str(tmp_path / "w")]) == 0
    assert seen == {"HOME": str(tmp_path / "w" / "home"), "TMPDIR": str(tmp_path / "w" / "tmp")}
    assert all(os.path.isdir(path) for path in seen.values())


def test_main_sets_the_umask_to_077_before_the_steps(ready, monkeypatch):
    masks = []

    def look(args, programs):
        masks.append(os.umask(0o077))
        return True, []

    monkeypatch.setattr(end_to_end, "run_steps", look)
    os.umask(0o022)
    assert end_to_end.main(ready) == 0 and masks == [0o077]


@pytest.mark.parametrize(
    "change",
    [
        lambda path: os.chmod(path, 0o750),
        pytest.param(
            lambda path: os.chown(path, 65534, -1), marks=[pytest.mark.root_only, only_root]
        ),
    ],
)
def test_main_stops_if_the_new_work_dir_is_not_the_users_own(ready, monkeypatch, capsys, change):
    make_work_dir = end_to_end.make_work_dir

    def make_then_change(path):
        fd = make_work_dir(path)
        change(path)
        return fd

    monkeypatch.setattr(end_to_end, "make_work_dir", make_then_change)
    assert end_to_end.main(ready) == 2
    assert capsys.readouterr().err == (
        "error: cannot create %s: it is not this user's private directory\n" % ready[0]
    )


def test_the_log_directory_is_not_opened_through_a_link(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "logs").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError):
        end_to_end.Programs(str(tmp_path / "logs"))


def test_a_missing_directory_above_the_work_dir_is_not_made(tmp_path, capsys, request):
    work_dir = str(tmp_path / "a" / "b" / "w")
    errors = end_to_end.check_prerequisites(end_to_end.parse_args([work_dir]))
    fault = "WORK_DIR cannot be used: %s does not exist. " % (tmp_path / "a")
    assert any(text.startswith(fault) for text in errors)
    request.getfixturevalue("ready")  # from here on without the early look
    assert end_to_end.main([work_dir]) == 2
    assert capsys.readouterr().err == "error: cannot create %s: %s does not exist\n" % (
        work_dir,
        tmp_path / "a",
    )
    assert os.listdir(tmp_path) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root may look at every directory")
def test_a_directory_above_that_cannot_be_looked_at_is_refused(tmp_path):
    (tmp_path / "closed" / "inner").mkdir(parents=True)
    (tmp_path / "closed").chmod(0o000)
    try:
        fault = end_to_end.fault_above(str(tmp_path / "closed" / "inner" / "w"))
        assert fault == "%s: Permission denied" % (tmp_path / "closed" / "inner")
    finally:
        (tmp_path / "closed").chmod(0o700)


def test_the_early_look_refuses_a_directory_that_gives_an_error(tmp_path, monkeypatch):
    (tmp_path / "closed").mkdir()
    real_lstat = os.lstat

    def closed_to_this_user(path, **keywords):
        if path == str(tmp_path / "closed"):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))
        return real_lstat(path, **keywords)

    monkeypatch.setattr(os, "lstat", closed_to_this_user)
    fault = end_to_end.fault_above(str(tmp_path / "closed" / "w"))
    assert fault == "%s: Permission denied" % (tmp_path / "closed")


def open_descriptors():
    found = set()
    for fd in map(int, os.listdir("/proc/self/fd")):
        with contextlib.suppress(OSError):  # the one that listed the directory is closed again
            os.fstat(fd)
            found.add(fd)
    return found


def test_the_walk_leaves_no_descriptor_open_but_the_one_that_it_returns(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "open").mkdir()
    (tmp_path / "open").chmod(0o777)
    before = open_descriptors()
    for refused in ("a/b/missing/w", "open/w", "a/b"):
        with pytest.raises(OSError):
            end_to_end.make_work_dir(str(tmp_path / refused))
        assert open_descriptors() == before
    fd = end_to_end.make_work_dir(str(tmp_path / "a" / "b" / "w"))
    assert open_descriptors() == before | {fd}
    os.close(fd)


def test_main_keeps_only_the_log_directorys_descriptor(ready, monkeypatch):
    seen = []

    def look(args, programs):
        seen.extend([open_descriptors(), programs.log_fd])
        return True, []

    monkeypatch.setattr(end_to_end, "run_steps", look)
    before = open_descriptors()
    assert end_to_end.main(ready) == 0
    assert seen[0] == before | {seen[1]}
    assert open_descriptors() == before


# ---- Results ----


VERIFIED, WITHIN_LIMITS = "verified", "within-limits"


@pytest.mark.parametrize(
    "more, passing",
    [
        ([], [(0, VERIFIED)]),
        (["--policy", "clipped-gap", "--limits", "1,2,3,4"], [(0, WITHIN_LIMITS)]),
    ],
)
def test_a_check_passes_only_if_code_and_outcome_fit_the_run(tmp_path, more, passing):
    args = end_to_end.parse_args(["w", *own_options(tmp_path), *more])
    for code in (0, 1, 2, 10, 14, 86):
        for outcome in (VERIFIED, WITHIN_LIMITS, "rejected", "no-proof", "error", ""):
            assert end_to_end.has_passed(args, code, outcome) is ((code, outcome) in passing)


def test_outcome_is_read_from_the_last_result_line(tmp_path):
    log = tmp_path / "log"
    log.write_text("result: outcome=verified claim=x\ncheck_proof: verified: x\n")
    assert end_to_end.read_outcome(str(log)) == "verified"
    log.write_text(log.read_text() + "result: outcome=error\nerror: y\n")
    assert end_to_end.read_outcome(str(log)) == "error"
    for text in ("check_proof: verified: x\n a result: outcome=verified\n", "result: verified\n"):
        log.write_text(text)
        assert end_to_end.read_outcome(str(log)) == ""


def test_last_line(tmp_path):
    (tmp_path / "log").write_text("check_proof: verified: x\nother\ncheck_proof: rejected: y\nz\n")
    log = str(tmp_path / "log")
    assert end_to_end.last_line(log, "check_proof: ") == "check_proof: rejected: y"
    assert end_to_end.last_line(log) == "z"
    assert end_to_end.last_line(log, "none") == ""


@pytest.mark.parametrize(
    "code, line, rejected",
    [
        (1, "check_proof: rejected: cp_verify printed NO: proof-refused", True),
        (
            1,
            "check_proof: rejected: cp_verify printed NO: link: it rejected the leaf file "
            "leaf/0.bin, not the proof",
            False,
        ),
        (1, "check_proof: rejected: the commitment ID differs", False),
        (2, "check_proof: error: the leaf file leaf/0.bin is missing", False),
        (0, "check_proof: verified: exact match", False),
        (2, "check_proof: rejected: cp_verify printed NO: proof-refused", False),
        (1, "", False),
    ],
)
def test_only_the_verifiers_no_to_the_proof_counts_as_rejected(code, line, rejected):
    assert end_to_end.verifier_rejected(code, line) is rejected


def test_only_a_failure_caused_by_the_policy_counts_as_declined():
    failed = spool.Response(ID, True, 1, {}, [])
    # Only a cp_prove built with test options prints this line; 26 is end_to_end.POLICY_CAUSE.
    by_policy = "# seat 0, request %s: failed: cause 26, flags 00000040\n" % ID
    assert end_to_end.prover_declined(failed, "x\n" + by_policy)[0] is True
    for silent in ("", "another line\n", by_policy.replace(ID, "cd" * 32)):
        assert end_to_end.prover_declined(failed, silent)[0] is None
    other = by_policy.replace("26", "2")
    assert end_to_end.prover_declined(failed, other)[0] is False
    assert end_to_end.prover_declined(failed, by_policy + other)[0] is False
    assert end_to_end.prover_declined(spool.Response(ID, False, 1, {}, []), by_policy)[0] is False
    faulty = spool.Response(ID, True, 1, {}, ["no tag"])
    assert end_to_end.prover_declined(faulty, by_policy)[0] is False


def test_changed_reply_differs_and_stays_in_the_vocabulary():
    assert end_to_end.changed_reply("exact", [1, 2, 9], 10) == [1, 2, 0]
    assert end_to_end.changed_reply("clipped-gap", [1, 2, 9], 10) == [2, 3, 0]


# ---- Files ----


def test_read_prompts(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("1,2,3\n\n4\n")
    assert end_to_end.read_prompts(str(path), 8, 10) == [[1, 2, 3], [4]]
    for text in ("", "1,x\n", "1,,2\n", "10\n", "1,1,1,1,1,1,1,1\n"):
        path.write_text(text)
        with pytest.raises(end_to_end.StepError):
            end_to_end.read_prompts(str(path), 8, 10)


def test_read_parameters(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("lt 6\nV 512\n")
    assert end_to_end.read_parameters(str(path)) == (64, 512)
    for text in ("lt 6\n", "lt 6\nV 512", "lt  6\nV 512\n"):
        path.write_text(text)
        with pytest.raises(end_to_end.StepError, match="cannot use the parameter file"):
            end_to_end.read_parameters(str(path))


def test_the_configuration_files_are_copied_together_and_private_to_the_user(tmp_path):
    """The sources are group-writable, as in a checkout under umask 002."""
    for name in ("p", "k", "l"):
        (tmp_path / name).write_text(name)
        (tmp_path / name).chmod(0o664)
    args = end_to_end.parse_args([str(tmp_path / "w")])
    (tmp_path / "w").mkdir()
    mask = os.umask(0o077)
    try:
        directory = end_to_end.copy_configuration(args, *(str(tmp_path / n) for n in "pkl"))
    finally:
        os.umask(mask)
    assert directory == str(tmp_path / "w" / "configuration")
    for name, content in (("policy.txt", "p"), ("KL.bin", "k"), ("layout.txt", "l")):
        path = tmp_path / "w" / "configuration" / name
        assert (path.read_text(), path.stat().st_mode & 0o777) == (content, 0o600)


def test_a_copy_never_goes_to_an_existing_file_or_through_a_link(tmp_path):
    (tmp_path / "source").write_text("new")
    (tmp_path / "target").write_text("kept")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    for name in ("target", "link"):
        with pytest.raises(FileExistsError):
            end_to_end.copy_to_new_file(str(tmp_path / "source"), str(tmp_path / name))
    assert (tmp_path / "target").read_text() == "kept"


def test_a_configuration_file_that_is_missing_is_named(tmp_path):
    (tmp_path / "p").write_text("p")
    (tmp_path / "k").write_text("k")
    args = end_to_end.parse_args([str(tmp_path / "w")])
    (tmp_path / "w").mkdir()
    message = re.escape("cannot copy %s to %s" % (tmp_path / "layout.txt", tmp_path / "w"))
    with pytest.raises(end_to_end.StepError, match=message):
        end_to_end.copy_configuration(
            args, str(tmp_path / "p"), str(tmp_path / "k"), str(tmp_path / "layout.txt")
        )


def test_with_own_commitment_the_run_uses_the_copies(tmp_path, monkeypatch):
    (tmp_path / "policy.txt").write_text("lt 6\nV 512\n")
    (tmp_path / "KL.bin").write_text("scales")
    (tmp_path / "layout.txt").write_text("layout")
    (tmp_path / "prompts.txt").write_text("1,2,3\n")
    seen = {}

    def stop_here(args, own, programs, config, tokens_per_request, name):
        seen.update(own=own, config=config, id_checked_first="id" in seen)
        raise end_to_end.StepError("far enough")

    monkeypatch.setattr(end_to_end, "start_vllm", stop_here)
    monkeypatch.setattr(end_to_end, "check_the_id", lambda *_: seen.update(id=True))
    args = end_to_end.parse_args([str(tmp_path / "w"), *own_options(tmp_path)])
    (tmp_path / "w").mkdir()
    with pytest.raises(end_to_end.StepError, match="far enough"):
        end_to_end.run_steps(args, None)
    assert seen["id_checked_first"]
    copies = tmp_path / "w" / "configuration"
    assert sorted(os.listdir(copies)) == ["KL.bin", "layout.txt", "policy.txt"]
    assert seen["own"].parameter_file == str(copies / "policy.txt")
    assert seen["own"].scales_file == str(copies / "KL.bin")
    with open(seen["config"]) as f:
        policy = json.load(f)["policies"][end_to_end.POLICY_NAME]
    assert policy["parameter_file"] == str(copies / "policy.txt")


def test_the_layout_file_is_expected_next_to_the_path_as_given(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "p.txt").write_text("")
    (tmp_path / "policy.txt").symlink_to(tmp_path / "store" / "p.txt")
    args = end_to_end.parse_args([str(tmp_path / "w"), *own_options(tmp_path)])
    assert args.layout_file == str(tmp_path / "layout.txt")
    assert args.parameter_file == str(tmp_path / "store" / "p.txt")


# ---- Command lines ----


def commitment(tmp_path, limits=()):
    names = ("pub", "sec", "policy.txt", "KL.bin", "model", "prompts.txt")
    return end_to_end.Commitment(ID, *(str(tmp_path / name) for name in names), "7", limits)


def test_the_default_verifier_is_check_proofs():
    assert os.path.join(end_to_end.REPO, end_to_end.VERIFIER) == check_proof.VERIFIER


@pytest.mark.parametrize("more", [[], ["--policy", "clipped-gap", "--limits", "4096,300,2,15"]])
def test_the_client_and_check_proof_accept_the_scripts_options(tmp_path, more):
    """Their parsers read the argument lists, so a renamed option fails here."""
    for name in ("pub", "policy.txt", "KL.bin", "cp_verify"):
        (tmp_path / name).touch(mode=0o700)
    verifier = ["--verifier", str(tmp_path / "cp_verify")]
    argv = [str(tmp_path / "w"), *own_options(tmp_path), *verifier, *more]
    args = end_to_end.parse_args(argv)
    (tmp_path / "w" / "client" / "public_key").mkdir(parents=True)
    own = commitment(tmp_path, tuple((args.limits or "").split(",")))
    policy = end_to_end.policy_options(args, own)
    options = end_to_end.client_options(args, own, "http://127.0.0.1:1", 0, [1, 2], 4)
    ask_and_check.parse_args([*options, "--ask-only"])
    ask_and_check.parse_args([*options, "--resume", "--", *policy])
    check = ["proof", "request.bin", "--end-token", "none", "--reply", "2:4"]
    check_proof.parse_args([*check, *end_to_end.check_options(args, own), *policy])


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    """Replace vLLM by a script that records its arguments and five env variables, then exits 1."""
    fake = tmp_path / "vllm"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$0.argv"\nenv | grep -E "^(VLLM_API_KEY|PROOF_MIDDLEWARE_CONFIG|VLLM_USE_V2_MODEL_RUNNER|VLLM_PLUGINS|VLLM_ENGINE_READY_TIMEOUT_S)=" > "$0.env"\nexit 1\n')
    fake.chmod(0o700)
    monkeypatch.setattr(end_to_end, "vllm_program", lambda: str(fake))
    return fake


@pytest.mark.parametrize("serve", ["integer", "float"])
def test_start_vllm_passes_what_each_way_of_serving_needs(programs, tmp_path, fake_vllm, serve):
    args = end_to_end.parse_args(
        [str(tmp_path / "w"), "--serve", serve, "--", "--max-num-seqs", "2"]
    )
    with pytest.raises(end_to_end.StepError, match="vLLM exited during start-up"):
        end_to_end.start_vllm(args, commitment(tmp_path), programs, "config.json", 64)
    assert programs.graces == [end_to_end.GPU_GRACE_S]
    argv = (tmp_path / "vllm.argv").read_text().split("\n")
    env = (tmp_path / "vllm.env").read_text().split("\n")
    assert argv[:2] == ["serve", str(tmp_path / "model")]
    assert argv[-3:] == ["--max-num-seqs", "2", ""]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--disable-uvicorn-access-log" in argv
    (key,) = [line.partition("=")[2] for line in env if line.startswith("VLLM_API_KEY=")]
    assert re.fullmatch("[0-9a-f]{32}", key) and key not in "\n".join(argv)
    assert "--skip-tokenizer-init" not in argv
    assert "PROOF_MIDDLEWARE_CONFIG=config.json" in env
    assert ("--load-format" in argv, "VLLM_USE_V2_MODEL_RUNNER=1" in env) == (
        serve == "integer",
    ) * 2
    assert ("--moe-backend" in argv, "VLLM_PLUGINS=" in env) == (serve == "float",) * 2
    assert ("--kv-cache-memory-bytes" in argv) == (serve == "float")
    assert "--gpu-memory-utilization" in argv
    assert "VLLM_ENGINE_READY_TIMEOUT_S=3600" in env  # --start-timeout's default


def test_a_vllm_that_does_not_become_ready_in_time_stops_the_run(
    programs, tmp_path, fake_vllm, monkeypatch
):
    loaded = time.monotonic() + 5  # as if it were still loading the model when the limit comes
    monkeypatch.setattr(programs, "exited", lambda process: time.monotonic() > loaded)
    args = end_to_end.parse_args([str(tmp_path / "w"), "--start-timeout", "1"])
    with pytest.raises(end_to_end.StepError) as error:
        end_to_end.start_vllm(args, commitment(tmp_path), programs, "config.json", 64)
    assert str(error.value) == (
        "vLLM did not answer /health within 1 s; see %s. If the log shows that it was still "
        "loading the model, pass a larger --start-timeout." % (tmp_path / "vllm.log")
    )


def test_with_own_commitment_the_float_model_gets_vllms_own_memory_settings(
    programs, tmp_path, fake_vllm
):
    limits = ["--serve", "float", "--limits", "4096,300,2,15"]
    args = end_to_end.parse_args([str(tmp_path / "w"), *own_options(tmp_path), *limits])
    with pytest.raises(end_to_end.StepError, match="vLLM exited during start-up"):
        end_to_end.start_vllm(args, commitment(tmp_path), programs, "config.json", 64)
    argv = (tmp_path / "vllm.argv").read_text().split("\n")
    assert "--kv-cache-memory-bytes" not in argv and "--gpu-memory-utilization" not in argv


def test_vllms_own_limit_for_its_engine_follows_start_timeout(programs, tmp_path, fake_vllm):
    args = end_to_end.parse_args([str(tmp_path / "w"), "--start-timeout", "77"])
    with pytest.raises(end_to_end.StepError, match="vLLM exited during start-up"):
        end_to_end.start_vllm(args, commitment(tmp_path), programs, "config.json", 64)
    assert "VLLM_ENGINE_READY_TIMEOUT_S=77" in (tmp_path / "vllm.env").read_text().split("\n")


def test_listener_uid_finds_only_a_socket_that_listens():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        assert end_to_end.listener_uid(port) is None
        s.listen()
        assert end_to_end.listener_uid(port) == os.geteuid()
    assert end_to_end.listener_uid(port) is None


@pytest.mark.root_only
@pytest.mark.skipif(os.geteuid() != 0, reason="only root can open a socket as another user")
def test_listener_uid_tells_another_users_socket_from_ours():
    other = 65534  # "nobody" on most systems
    try:
        os.seteuid(other)
        s = socket.socket()
    except OSError:
        pytest.skip("the user ID %d is not mapped in this container" % other)
    finally:
        os.seteuid(0)
    with s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert end_to_end.listener_uid(s.getsockname()[1]) == other


def with_fake_cp_prove(tmp_path, script):
    (tmp_path / "build").mkdir()
    fake = tmp_path / "build" / "cp_prove"
    fake.write_text("#!/bin/sh\n%s\n" % script)
    fake.chmod(0o700)
    args = end_to_end.parse_args([str(tmp_path / "w"), "--build-dir", str(tmp_path / "build")])
    # prove() may create "hold" here. Made only now: parse_args() rejects a WORK_DIR that exists.
    (tmp_path / "w" / "spool").mkdir(parents=True)
    return args


@pytest.mark.parametrize(
    "output, message",
    [
        ("refused (gpu-in-use): more than 2 GiB", "Give them different GPUs with --vllm-gpu"),
        ("p258 end 1", 'exited with code 1, expected 0; its last line: "p258 end 1"; see '),
        ("a\033[31m" + "b" * 300, 'its last line: "a\\?\\[31m%s"; see ' % ("b" * 194)),
        (
            "",
            "exited with code 1, expected 0; it printed nothing; see .*[.]log[.] If that gives no ",
        ),
    ],
)
def test_prove_says_why_cp_prove_failed_as_far_as_its_log_does(programs, tmp_path, output, message):
    args = with_fake_cp_prove(tmp_path, "printf '%%s' '%s'\nexit 1" % output)
    advice = "or pass --stop-vllm-to-prove"
    with pytest.raises(end_to_end.StepError, match=message) as failed:
        end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    assert advice in str(failed.value)
    # With vLLM stopped, or with another --prover-gpu than --vllm-gpu: it ends at the log's path.
    if "GPUs" in message:
        message = "Give it a GPU that nothing else uses"
    else:
        message = message.partition(" .*[.]log[.] If that")[0].rstrip() + " .*[.]log$"
    args.stop_vllm_to_prove = True
    with pytest.raises(end_to_end.StepError, match=message) as failed:
        end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove_alone")
    assert advice not in str(failed.value)
    args.stop_vllm_to_prove, args.prover_gpu = False, "1"
    with pytest.raises(end_to_end.StepError, match=message) as failed:
        end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove_again")
    assert advice not in str(failed.value)


@pytest.mark.parametrize("script, held", [("exit 0", False), ("exit 9", True), ("sleep 30", True)])
def test_proofs_are_put_on_hold_unless_cp_prove_exits_with_code_0(
    programs, tmp_path, capsys, script, held
):
    args = with_fake_cp_prove(tmp_path, script)
    args.proof_timeout = 1
    for name in ("first", "second"):  # twice: an existing "hold" file is fine
        with contextlib.nullcontext() if not held else pytest.raises(end_to_end.StepError):
            end_to_end.prove(args, commitment(tmp_path), programs, name)
        assert os.listdir(tmp_path / "w" / "spool") == ["hold"] * held
        assert ("proofs are on hold while" in capsys.readouterr().out) is held


def test_proofs_are_put_on_hold_if_the_run_is_interrupted_while_cp_prove_runs(
    programs, tmp_path, monkeypatch
):
    def interrupted(*args, **more):
        raise end_to_end.Interrupted(signal.SIGINT)

    args = with_fake_cp_prove(tmp_path, "exit 0")
    monkeypatch.setattr(programs, "run", interrupted)
    with pytest.raises(end_to_end.Interrupted):
        end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    assert os.listdir(tmp_path / "w" / "spool") == ["hold"]


def test_a_hold_file_that_cannot_be_created_does_not_hide_why_cp_prove_failed(
    programs, tmp_path, capsys
):
    args = with_fake_cp_prove(tmp_path, "echo 'p258 end 9 own-test-slot-pads'\nexit 9")
    (tmp_path / "w" / "spool").rmdir()
    with pytest.raises(end_to_end.StepError, match='its last line: "p258 end 9 own-test-slot-pads"'):
        end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    assert "cannot create %s: No such file" % (tmp_path / "w" / "spool" / "hold") in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "printed, status, passes",
    [(ID, 4, True), ("cd" * 32, 4, False), (ID, 0, False), (ID, 1, False)],
)
def test_your_commitment_id_is_held_against_your_files(tmp_path, printed, status, passes):
    verifier = tmp_path / "cp_verify"
    verifier.write_text("#!/bin/sh\necho 'commitment ID: %s'\nexit %d\n" % (printed, status))
    verifier.chmod(0o700)
    (tmp_path / "config").mkdir()
    own = commitment(tmp_path)
    own.commitment_id, own.parameter_file = ID, str(tmp_path / "config" / "policy.txt")
    args = argparse.Namespace(verifier=str(verifier))
    with contextlib.nullcontext() if passes else pytest.raises(end_to_end.StepError):
        end_to_end.check_the_id(args, own)


def test_cp_prove_gets_the_options_that_the_caller_adds(programs, tmp_path, capsys):
    args = with_fake_cp_prove(tmp_path, 'printf "%s\\n" "$@" > "$0.argv"')
    assert args.cp_prove_options == []
    args.cp_prove_options = end_to_end.parse_args(
        ["w", "--cp-prove-options=-memory lean -name 'two words'"]
    ).cp_prove_options
    end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    argv = (tmp_path / "build" / "cp_prove.argv").read_text().split("\n")
    own = commitment(tmp_path)
    assert argv == [
        *("-isk", own.secret_key_dir, "-spool", str(tmp_path / "w" / "spool")),
        *("-batch", str(args.batch), "-end-when-empty"),
        *("-memory", "lean", "-name", "two words", ""),
    ]
    message = "--cp-prove-options cannot contain -spool: the script sets it"
    assert message in usage_error(capsys, "w", "--cp-prove-options=-memory lean -spool x")
    assert "cannot contain --isk=ab:" in usage_error(capsys, "w", "--cp-prove-options=--isk=ab")
    assert end_to_end.parse_args(["w", "--cp-prove-options=-name isk"]).cp_prove_options[1] == "isk"
    assert "--cp-prove-options: No closing quotation" in usage_error(
        capsys, "w", "--cp-prove-options='"
    )


@needs_the_example
def test_the_demo_commits_to_synthetic_weights_with_two_leaf_tables(programs, tmp_path):
    (tmp_path / "build").mkdir()
    fake = tmp_path / "build" / "cp_commit"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$0.argv"\nexit 1\n')
    fake.chmod(0o700)
    build = ["--serve", "integer", "--build-dir", str(tmp_path / "build")]
    args = end_to_end.parse_args([str(tmp_path / "w"), *build])
    (tmp_path / "w").mkdir()
    with pytest.raises(end_to_end.StepError, match="cp_commit exited with code 1"):
        end_to_end.prepare_demo(args, programs)
    assert programs.graces == [end_to_end.GPU_GRACE_S]
    argv = (tmp_path / "build" / "cp_commit.argv").read_text().split("\n")
    work = tmp_path / "w"
    assert argv == [
        *("-weights", "made-up-blocks", "-config", str(work / "configuration")),
        *("-ivk", str(work / "public_key"), "-isk", str(work / "secret" / "key"), "-M", "2", ""),
    ]


def with_cp_commit_that_prints(tmp_path, line):
    (tmp_path / "build").mkdir()
    fake = tmp_path / "build" / "cp_commit"
    fake.write_text('#!/bin/sh\necho "another line"\necho "%s"\n' % line)
    fake.chmod(0o700)
    build = ["--serve", "integer", "--build-dir", str(tmp_path / "build")]
    args = end_to_end.parse_args([str(tmp_path / "w"), *build])
    (tmp_path / "w").mkdir()
    return args


@needs_the_example
@pytest.mark.parametrize(
    "printed",
    ["", "cp_commit: commitment ID: " + ID, "commitment ID: ab", "commitment ID: %s x" % ID],
)
def test_the_demo_names_the_expected_line_if_cp_commit_prints_no_valid_id(
    programs, tmp_path, printed
):
    args = with_cp_commit_that_prints(tmp_path, printed)
    message = re.escape('no commitment ID (expected a line "commitment ID: " and 64 lowercase hex')
    with pytest.raises(end_to_end.StepError, match=message):
        end_to_end.prepare_demo(args, programs)


@needs_the_example
def test_the_demo_reads_the_commitment_id_that_cp_commit_prints(
    programs, tmp_path, monkeypatch, capsys
):
    def stop_here(*args):
        raise end_to_end.StepError("far enough")

    monkeypatch.setattr(end_to_end, "write_demo_model", stop_here)
    args = with_cp_commit_that_prints(tmp_path, "commitment ID: " + ID)
    with pytest.raises(end_to_end.StepError, match="far enough"):
        end_to_end.prepare_demo(args, programs)
    assert "   commitment ID %s\n" % ID in capsys.readouterr().out


def test_prove_returns_if_cp_prove_exits_with_code_0(programs, tmp_path):
    args = with_fake_cp_prove(tmp_path, "echo 'refused (gpu-in-use): in an earlier line'\nexit 0")
    end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")


@pytest.mark.parametrize(
    "line, said",
    [
        # the two lines as cp_prove prints them, shortened in the middle
        ("setting: field degree 3; version word 825374792; hb 128; a serving build", None),
        (
            "setting: field degree 3; version word 825374792; hb 128; built for tests alone",
            'is not the build that is meant for serving: by its log it is "built for tests alone".',
        ),
        (
            "setting: x; \x1b[2J" + "y" * 200,  # shown without control characters, and cut
            'is not the build that is meant for serving: by its log it is "\\x1b[2J%s".'
            % ("y" * 96),
        ),
        ("another line", 'did not say what kind of build it is (its log has no "setting:" line).'),
    ],
)
def test_a_run_says_so_if_cp_prove_is_not_the_serving_build(programs, tmp_path, line, said):
    args = with_fake_cp_prove(tmp_path, "echo '%s'\necho 'a later line'" % line)
    before = end_to_end.closing_notes(args)
    end_to_end.prove(args, commitment(tmp_path), programs, "cp_prove")
    new = [text for text in end_to_end.closing_notes(args) if text not in before]
    note = (
        "cp_prove in %s %s The build meant for serving is made by prover/build.sh run without an "
        "argument."
    )
    assert new == ([note % (tmp_path / "build", said)] if said else [])


@needs_the_example
@pytest.mark.parametrize("serve", ["integer", "float"])
def test_the_demos_model_directory(tmp_path, serve):
    """The integer model package writes the files; a changed signature fails here."""
    pytest.importorskip(
        "integer_model_vllm" if serve == "integer" else "integer_model_vllm.float_pair"
    )
    (tmp_path / "w").mkdir()
    args = end_to_end.parse_args([str(tmp_path / "new"), "--serve", serve])
    args.work_dir = str(tmp_path / "w")
    configuration = end_to_end.demo_configuration(args)
    assert sorted(os.listdir(configuration))[:1] == ["KL.bin"]
    assert {"policy.txt", "layout.txt"} <= set(os.listdir(configuration))
    model_dir, end_token = end_to_end.write_demo_model(args, configuration, [1])
    assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(os.listdir(model_dir))
    assert ("model.safetensors" in os.listdir(model_dir)) == (serve == "float")
    assert 0 <= end_token < end_to_end.read_parameters(os.path.join(configuration, "policy.txt"))[1]
