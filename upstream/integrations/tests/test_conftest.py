# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests for conftest.py, which run pytest on a small copy of this directory's layout.

conftest.py skips this directory's tests ("stand aside" in names) if another account could change
the directory in which pytest makes its temporary directory, or one above it. A run with no other
tests and no error then exits with code 77. One more test compares the block of routines that
conftest.py uses from spool.py, and its copy in check_proof.py, with an expected SHA-256 digest.
"""

import hashlib
import os
import shutil
import subprocess
import sys

import pytest

import spool

HERE = os.path.dirname(os.path.abspath(__file__))
# Start of the last line of these tests' runs with code 77.
STOOD_ASIDE = "stood aside: below an open place: %s (its group or others can write in it, and it "
BELOW = " below directories that only you or root can change"


@pytest.fixture
def run_pytest(tmp_path, monkeypatch):
    """Return run(ground, *arguments), which runs pytest on the copy with TMPDIR = <ground>/tmp.

    `ground` is "own" or "open" (mode 0777, no sticky bit), the only unsafe case tried.
    """
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)  # it would outrank TMPDIR
    (tmp_path / "tree" / "tests").mkdir(parents=True)
    (tmp_path / "tree" / "other").mkdir()
    (tmp_path / "tree" / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    shutil.copy(spool.__file__, tmp_path / "tree" / "spool.py")
    shutil.copy(os.path.join(HERE, "conftest.py"), tmp_path / "tree" / "tests" / "conftest.py")
    (tmp_path / "tree" / "tests" / "test_ours.py").write_text(
        "def test_ours(tmp_path):\n    assert tmp_path.is_dir()\n"
    )
    (tmp_path / "tree" / "other" / "test_theirs.py").write_text("def test_theirs():\n    pass\n")
    for ground, mode in (("own", 0o755), ("open", 0o777)):
        (tmp_path / ground / "tmp").mkdir(parents=True)
        (tmp_path / ground).chmod(mode)

    def run(ground, *arguments):
        env = {**os.environ, "TMPDIR": str(tmp_path / ground / "tmp")}
        env.pop("PYTHONPATH", None)
        got = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *arguments],
            cwd=tmp_path / "tree",
            env={**env, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=120,
        )
        return got.returncode, (got.stdout + got.stderr).strip().split("\n")[-1]

    return run


@pytest.mark.parametrize("name", ["spool.py", "check_proof.py"])
def test_the_copies_of_the_look_at_the_directories_above_are_unchanged(name):
    """The block says "change nothing in it": a fix belongs in run_suite.py and in every copy."""
    with open(os.path.join(os.path.dirname(HERE), name), encoding="utf-8") as f:
        text = f.read()
    start = text.index("# The look that run_suite.py makes before it works below a folder")
    block = text[start : text.index("# fmt: on", start)].rstrip("\n") + "\n"
    assert hashlib.sha256(block.encode()).hexdigest() == (
        "074f31044e834d727586a54594240539bdb4428bb287a6afebb91e5e20f5f2ab"
    )


def test_on_ground_of_ones_own_the_tests_run(run_pytest):
    code, last = run_pytest("own", "tests")
    assert code == 0 and last.startswith("1 passed")


def test_below_an_open_place_the_tests_stand_aside(run_pytest, tmp_path):
    code, last = run_pytest("open", "tests")
    assert code == 77 and last.startswith(STOOD_ASIDE % spool.printable(str(tmp_path / "open")))
    assert last.endswith(
        "has no sticky bit): set TMPDIR to an existing directory that only you can write," + BELOW
    )
    assert "--basetemp" not in last  # pytest removes the directory that this option names
    assert os.listdir(tmp_path / "open" / "tmp") == []


def test_the_ground_is_the_one_that_pytest_takes(run_pytest, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path / "open" / "tmp"))
    code, last = run_pytest("own", "tests")
    assert code == 77 and last.startswith(STOOD_ASIDE % spool.printable(str(tmp_path / "open")))
    assert last.endswith(
        "): set PYTEST_DEBUG_TEMPROOT to an existing directory that only you can write," + BELOW
    )


def test_the_advice_on_basetemp_warns_that_pytest_removes_the_directory(run_pytest, tmp_path):
    code, last = run_pytest("own", "tests", "--basetemp", str(tmp_path / "open" / "tmp" / "new"))
    assert code == 77 and last.startswith(STOOD_ASIDE % spool.printable(str(tmp_path / "open")))
    assert last.endswith(
        "): give --basetemp the name of a new directory (pytest removes an existing one)" + BELOW
    )


def test_on_ground_that_is_missing_no_test_is_run(run_pytest, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(tmp_path / "own" / "missing"))
    code, last = run_pytest("own", "tests", "other")
    assert (code, last) == (
        4,
        "ERROR: no test was run: pytest's temporary directory would be below %s (it does not "
        "exist): set PYTEST_DEBUG_TEMPROOT to an existing directory that only you can write,%s"
        % (spool.printable(str(tmp_path / "own" / "missing")), BELOW),
    )


def test_a_run_that_fails_for_another_reason_keeps_its_status(run_pytest, tmp_path):
    (tmp_path / "tree" / "tests" / "test_broken.py").write_text("import no_such_module\n")
    # An error in collection: pytest's exit code 2.
    for ground in ("own", "open"):
        code, last = run_pytest(ground, "tests")
        assert code == 2 and "stood aside" not in last


def test_another_directorys_tests_are_left_alone(run_pytest):
    code, last = run_pytest("open", "tests", "other")
    assert code == 0 and last.startswith("1 passed, 1 skipped")


def test_only_the_chosen_tests_are_counted(run_pytest, tmp_path):
    code, last = run_pytest("open", "tests", "other", "-k", "ours")
    assert code == 77 and last.startswith(STOOD_ASIDE % spool.printable(str(tmp_path / "open")))
    # No test is chosen: pytest's exit code 5. Tests are only listed: 0.
    assert run_pytest("open", "tests", "-k", "matches_nothing")[0] == 5
    assert run_pytest("open", "tests", "--collect-only")[0] == 0


def test_the_marker_root_only_is_registered(run_pytest, tmp_path):
    (tmp_path / "tree" / "tests" / "test_ours.py").write_text(
        "import pytest\n\n\n@pytest.mark.root_only\ndef test_marked():\n    pass\n\n\n"
        "def test_plain():\n    pass\n"
    )
    code, last = run_pytest("own", "tests", "--strict-markers", "-m", "root_only")
    assert code == 0 and last.startswith("1 passed, 1 deselected")
