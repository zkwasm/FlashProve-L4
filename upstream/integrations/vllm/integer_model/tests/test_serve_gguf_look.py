# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""examples/serve_gguf.sh's check of the folder for temporary files, before it makes its working
directory there, and that it writes nothing there. They use test_examples.py's helpers and its
fixture programs_that_print (imported, so ruff's F811 is off in this file)."""

# ruff: noqa: F811
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from test_examples import PACKAGE, example, printed, programs_that_print, run  # noqa: F401


def serve_gguf_in(top, environment, tmp_mode, tmpdir=None):
    """Runs serve_gguf.sh in top, with model/, reference/ and tmp/ (of tmp_mode) made there, and
    TMPDIR tmpdir, or top/tmp."""
    for folder in (top / "model", top / "reference", top / "tmp"):
        folder.mkdir(parents=True)
    (top / "tmp").chmod(tmp_mode)
    for name in example.REFERENCE_FILES:
        (top / "reference" / name).write_text(name)
    variables = {**environment, "VLLM_API_KEY": "k", "TMPDIR": str(tmpdir or top / "tmp")}
    return run(
        *(PACKAGE / "examples" / "serve_gguf.sh", "reference", "model", "name"),
        env=variables,
        cwd=top,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.parametrize(
    "mode, status", [(0o700, 0), (0o755, 0), (0o1777, 0), (0o777, 2), (0o770, 2)]
)
def test_the_folder_for_temporary_files_is_looked_at(programs_that_print, tmp_path, mode, status):
    """Below a folder that group or others can write in, without the sticky bit, serve_gguf.sh
    stops with status 2 before it draws its folder; a private and a sticky one are taken."""
    top = tmp_path / "t"
    done = serve_gguf_in(top, programs_that_print, mode)
    assert done.returncode == status, done.stderr
    if status:
        assert "Set TMPDIR to such a folder." in done.stderr
        assert list((top / "tmp").iterdir()) == []
    else:
        _, variables = printed(done)
        assert pathlib.Path(variables["TMPDIR"]).parent == top / "tmp"


def test_under_sudo_the_sudoing_account_is_not_trusted_with_that_folder(
    programs_that_print, tmp_path
):
    """As root with SUDO_UID set, a TMPDIR below a folder of that account is refused: root's server
    would run there. Without root the scene cannot be made, and the test checks only that
    the same folder, open to others, is refused; it never skips, so that the suite's count of
    skips does not grow."""
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    environment = dict(programs_that_print)
    if os.geteuid() == 0:
        os.chown(theirs, 54321, 54321)
        environment["SUDO_UID"] = "54321"
    else:
        theirs.chmod(0o777)
    done = serve_gguf_in(theirs / "t", environment, 0o700)
    assert done.returncode == 2, done.stderr
    assert list((theirs / "t" / "tmp").iterdir()) == []


def test_a_folder_that_does_not_stand_is_refused_too(programs_that_print, tmp_path):
    """The look always runs: a TMPDIR that does not stand is refused by the routine, so that it
    cannot appear between a test and mktemp, and nothing is made."""
    top = tmp_path / "t"
    done = serve_gguf_in(top, programs_that_print, 0o700, top / "tmp" / "gone")
    assert done.returncode == 2, done.stderr
    assert "does not exist" in done.stderr
    assert list((top / "tmp").iterdir()) == []


def test_the_look_writes_no_bytecode_into_the_checkout(programs_that_print, tmp_path):
    """python3 -B: even with bytecode allowed, the look leaves no __pycache__ in the checkout. The
    script runs from a fresh copy of the package, so that no earlier run has left bytecode there."""
    copy = tmp_path / "checkout" / "integrations" / "vllm" / "integer_model"
    for part in ("examples", "integer_model_vllm"):
        shutil.copytree(PACKAGE / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
    top = tmp_path / "t"
    for folder in (top / "model", top / "reference", top / "tmp"):
        folder.mkdir(parents=True)
    (top / "tmp").chmod(0o777)
    for name in example.REFERENCE_FILES:
        (top / "reference" / name).write_text(name)
    # run() hands on exactly this environment: without the variable, Python would write bytecode.
    environment = {k: v for k, v in programs_that_print.items() if k != "PYTHONDONTWRITEBYTECODE"}
    environment |= {"VLLM_API_KEY": "k", "TMPDIR": str(top / "tmp")}
    done = run(
        copy / "examples" / "serve_gguf.sh",
        "reference",
        "model",
        "name",
        env=environment,
        cwd=top,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 2, done.stderr  # refused, after the look
    assert list(copy.rglob("__pycache__")) == []


def test_the_look_takes_no_python_setting_of_the_caller(programs_that_print, tmp_path):
    """python3 -I: a user site under HOME (as under sudo with HOME kept), a sitecustomize.py on
    PYTHONPATH, and PYTHONINSPECT (which would turn the refusal into status 0) all leave the look as
    it is: below a 0777 folder, status 2, nothing made, and neither customizing file ran."""
    home, fake = tmp_path / "home", tmp_path / "fake"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = home / ".local" / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    marker = tmp_path / "usercustomize-ran"
    (site / "usercustomize.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    fake.mkdir()
    (fake / "sitecustomize.py").write_text(f"open({str(marker)!r} + '-site', 'w').close()\n")
    for setting in (
        {"HOME": str(home)},
        {"PYTHONPATH": str(fake)},
        {"PYTHONINSPECT": "1"},
    ):
        top = tmp_path / next(iter(setting))
        done = serve_gguf_in(top, {**programs_that_print, **setting}, 0o777)
        assert done.returncode == 2, (setting, done.stderr)
        assert list((top / "tmp").iterdir()) == []
    assert not marker.exists() and not pathlib.Path(f"{marker}-site").exists()


def test_the_working_directory_is_made_by_the_real_path(programs_that_print, tmp_path):
    """TMPDIR by a link of the caller's own to a private folder: the check looks at both paths, and
    mktemp draws in the real one, which vLLM then gets."""
    top = tmp_path / "t"
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(real)
    done = serve_gguf_in(top, programs_that_print, 0o700, tmp_path / "link")
    _, variables = printed(done)
    work = pathlib.Path(variables["TMPDIR"])
    assert work.parent == real
    assert work.stat().st_mode & 0o777 == 0o700 and list(work.iterdir()) == []


def test_a_newline_at_the_end_of_the_path_is_not_lost(programs_that_print, tmp_path):
    """TMPDIR a private folder whose name ends in a newline, beside an open folder of the same name
    without it: the check refuses the control character, so the script draws in neither, and never
    in the one that it did not look at."""
    top = tmp_path / "t"
    (tmp_path / "nl").mkdir()
    (tmp_path / "nl").chmod(0o777)
    (tmp_path / "nl\n").mkdir(mode=0o700)
    done = serve_gguf_in(top, programs_that_print, 0o700, tmp_path / "nl\n")
    assert done.returncode == 2, done.stderr
    assert list((tmp_path / "nl").iterdir()) == [] and list((tmp_path / "nl\n").iterdir()) == []


def test_a_missing_folder_for_temporary_files_is_named_plainly(programs_that_print, tmp_path):
    top = tmp_path / "t"
    done = serve_gguf_in(top, programs_that_print, 0o700, top / "tmp" / "gone")
    assert done.returncode == 2, done.stderr
    assert f"TMPDIR, {top / 'tmp' / 'gone'}, does not exist." in done.stderr


def test_the_script_writes_nothing_into_its_working_directory(programs_that_print, tmp_path):
    """A stand-in mktemp lays a link to another file in the new directory, and in a second scene a
    plain file. The script writes nothing there: the server is started, nothing else lies there,
    and the link, the other file and the plain file stay as they were."""
    victim = tmp_path / "victim"
    victim.write_text("not the digests")
    real_mktemp = shutil.which("mktemp")
    fake = tmp_path / "bin"  # the fixture's folder of stand-ins, first on its PATH
    lines = [f'd=$({real_mktemp} "$@") || exit 1', f'ln -s {victim} "$d/sha256.json"', 'echo "$d"']
    (fake / "mktemp").write_text("#!/bin/sh\n" + "\n".join(lines) + "\n")
    (fake / "mktemp").chmod(0o755)
    done = serve_gguf_in(tmp_path / "t", programs_that_print, 0o700)
    printed(done)  # asserts status 0: the stand-in for vLLM ran
    (work,) = (tmp_path / "t" / "tmp").iterdir()
    assert [path.name for path in work.iterdir()] == ["sha256.json"]
    assert (work / "sha256.json").is_symlink() and victim.read_text() == "not the digests"
    # A plain file at the name instead.
    lines[1] = 'echo "not the digests" > "$d/sha256.json"'
    (fake / "mktemp").write_text("#!/bin/sh\n" + "\n".join(lines) + "\n")
    printed(serve_gguf_in(tmp_path / "u", programs_that_print, 0o700))
    (work,) = (tmp_path / "u" / "tmp").iterdir()
    assert [path.name for path in work.iterdir()] == ["sha256.json"]
    assert (work / "sha256.json").read_text() == "not the digests\n"
