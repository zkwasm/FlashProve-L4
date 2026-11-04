# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""What the package's scripts do where vllm, or a Python package that they need, is not found:
examples/find_or_give_root_up.sh, and the lines at the scripts' heads that start it.

Everything runs in a made-up checkout below tmp_path. No test skips as root: there a test checks
that the helper stops, in a checkout of root's own and in one that belongs to another uid, and that
nothing of the environment beside the package is started.
"""

import ast
import itertools
import os
import pathlib
import pwd
import shutil
import subprocess
import sys

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
HELPER = "find_or_give_root_up.sh"
SHELL_SCRIPTS = ("serve.sh", "serve_gguf.sh")
PYTHON_SCRIPTS = (
    "examples/score_reply.py",
    "examples/fit_limits.py",
    "examples/try_limits.py",
    "integer_model_vllm/example.py",
)
ROOT = os.geteuid() == 0
PYTHON = sys._base_executable  # an interpreter without an active virtual environment
NO_VLLM = "vllm was not found on PATH"
NO_PACKAGE = f"the Python package no_such_package was not found by {PYTHON}"
MAKE = (
    "Nothing was made. Make the virtual environment as integrations/vllm/integer_model/README.md "
    "says, or activate yours."
)


def no_account() -> int:
    """A uid that has no account on this machine."""
    for uid in itertools.count(54321):
        try:
            pwd.getpwuid(uid)
        except KeyError:
            return uid


def as_root(missing: str) -> str:
    """What root is told, whoever owns the checkout."""
    return (
        f"started as root, and {missing} (sudo may drop an activated environment's PATH), so it "
        "stops here. Run the same command as the checkout's owner, without sudo. If it must be "
        "root, activate an environment of root's own, in a checkout of root's own; if you activate "
        "another account's environment in a root shell instead, root runs whatever that "
        'environment holds. See integrations/vllm/integer_model/OPERATING.md, "As root or under '
        'sudo".\n'
    )


def give_away(package: pathlib.Path) -> None:
    """As root: gives examples/ and its files to a uid that has no account."""
    uid = no_account()
    for path in (package / "examples", *(package / "examples").iterdir()):
        os.chown(path, uid, uid)


def head_of(script: str) -> str:
    """The lines at a Python script's head that look for its packages, without the call."""
    text = (PACKAGE / script).read_text()
    start = text.index('if __name__ == "__main__":\n    # Where no virtual environment is active')
    end = '            os.execv("/bin/sh", ["/bin/sh", helper, *words])\n'
    return text[start : text.index(end, start) + len(end)]


@pytest.fixture
def run(tmp_path):
    """Runs a command with a PATH that holds four tools and no vllm, wherever one is installed."""
    (tmp_path / "tools").mkdir()
    for tool in ("sh", "realpath", "id", "tr"):
        (tmp_path / "tools" / tool).symlink_to(shutil.which(tool))

    def run(*command, cwd, stdin=None, **variables):
        return subprocess.run(
            [str(word) for word in command],
            env={"PATH": f"{tmp_path}/tools", "HOME": "/nonexistent", **variables},
            cwd=cwd,
            stdin=stdin,
            capture_output=True,
            text=True,
            timeout=60,
        )

    return run


def checkout(top: pathlib.Path) -> pathlib.Path:
    """Makes a checkout's package folder below top, with the helper and the two shell scripts in
    examples/. Every mode that a head compares is set, whatever the umask.
    """
    package = top / "integrations" / "vllm" / "integer_model"
    (package / "examples").mkdir(parents=True)
    (package / "examples").chmod(0o755)
    for name, mode in ((HELPER, 0o644), *((script, 0o755) for script in SHELL_SCRIPTS)):
        shutil.copy(PACKAGE / "examples" / name, package / "examples")
        (package / "examples" / name).chmod(mode)
    return package


@pytest.fixture
def package(tmp_path):
    """A made-up checkout's package folder, with examples/toy.py, which has score_reply.py's head,
    asks for json and no_such_package, and prints "went on".
    """
    package = checkout(tmp_path)
    call = '\n    find_or_hand_over("json", "no_such_package")\n\nprint("went on")\n'
    (package / "examples" / "toy.py").write_text(head_of("examples/score_reply.py") + call)
    (package / "examples" / "toy.py").chmod(0o644)
    return package


def stand_in(path: pathlib.Path, words: str = '"$@" "${PATH%%:*}"') -> None:
    """Makes a program that prints its name and, unless told otherwise, its arguments and PATH's
    first entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nprintf "<%s>" "${{0##*/}}" {words}\n')
    path.chmod(0o755)


def test_the_four_python_heads_are_the_same():
    assert len({head_of(script) for script in PYTHON_SCRIPTS}) == 1
    for script in PYTHON_SCRIPTS:  # ahead of every import but that of __future__
        before = (PACKAGE / script).read_text().split(head_of(script))[0]
        imports = [n for n in ast.parse(before).body if isinstance(n, ast.Import | ast.ImportFrom)]
        assert [n.module for n in imports] in ([], ["__future__"]), script


def test_the_two_shell_heads_are_the_same():
    heads = set()
    for script in SHELL_SCRIPTS:
        text = (PACKAGE / "examples" / script).read_text()
        heads.add(text[text.index("# Where vllm is not on PATH") : text.index("\nfi\n")])
    assert len(heads) == 1 and HELPER in heads.pop()


def test_the_vllm_package_imports_nothing_at_its_top():
    """python3 -m integer_model_vllm.example runs __init__.py ahead of example.py's head."""
    tree = ast.parse((PACKAGE / "integer_model_vllm" / "__init__.py").read_text())
    assert not [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]


def test_serve_sh_keeps_example_py_from_handing_over():
    text = (PACKAGE / "examples" / "serve.sh").read_text()
    marked = "INTEGER_MODEL_ENVIRONMENT_TRIED=1 PYTHONPATH=$package \\\n    python3 -B -s -m "
    assert text.count(marked) == 1 and text.count("python3 -B") == 1


def test_a_python_script_goes_on_where_its_packages_are_found(package, run):
    """A file beside the script counts as found, as it does for Python's import; it is not run."""
    (package / "examples" / "no_such_package.py").write_text('print("the planted file ran")\n')
    (package / "examples" / HELPER).write_text("echo the helper ran\n")
    done = run(PYTHON, "-B", package / "examples" / "toy.py", cwd=package)
    assert (done.returncode, done.stdout, done.stderr) == (0, "went on\n", "")


def test_a_package_in_the_working_directory_counts_as_found(package, run):
    """python3 -m puts the working directory first on the import path; the head leaves it there."""
    (package / "no_such_package").mkdir()
    (package / "no_such_package" / "__init__.py").write_text('print("the package was run")\n')
    # -S: without the machine's site packages, one of which may be named examples too
    done = run(PYTHON, "-S", "-B", "-m", "examples.toy", cwd=package)
    assert (done.returncode, done.stdout, done.stderr) == (0, "went on\n", "")


def test_a_python_script_is_started_again_by_the_environment_beside_the_package(
    package, tmp_path, run
):
    """As root the environment is not looked for, in root's own checkout or another's."""
    stand_in(package / ".venv" / "bin" / "python3", '"$@" "$INTEGER_MODEL_ENVIRONMENT_TRIED"')
    typed = "integrations/vllm/integer_model/examples/toy.py"
    done = run(PYTHON, "-B", typed, "a b", "$(c)", cwd=tmp_path)
    if ROOT:
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr == f"{typed}: {as_root(NO_PACKAGE)}"
        give_away(package)
        assert run(PYTHON, "-B", typed, "a b", "$(c)", cwd=tmp_path).__dict__ == done.__dict__
        return
    assert (done.returncode, done.stdout) == (0, f"<python3><-B><{typed}><a b><$(c)><1>")
    again = f"starting again with {package}/.venv/bin/python3."
    assert done.stderr == f"{typed}: {NO_PACKAGE}: {again}\n"


def test_a_python_script_stops_where_there_is_no_environment(package, run):
    """As root: in root's own checkout, and in another's."""
    toy = package / "examples" / "toy.py"

    def stopped():
        done = run(PYTHON, "-B", toy, cwd=package)
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr.startswith(f"{toy}: ")
        return done.stderr.removeprefix(f"{toy}: ")

    if ROOT:
        assert stopped() == as_root(NO_PACKAGE)
        give_away(package)
        assert stopped() == as_root(NO_PACKAGE)
        return
    none = f"{NO_PACKAGE}, and {package}/.venv/bin has no python3. {MAKE}"
    assert stopped() == f"{none}\n"
    (package / ".venv" / "bin" / "python3").mkdir(parents=True)  # a directory is no program
    assert stopped() == f"{none}\n"
    # No head comes back to the helper; if one did, the helper would stop.
    words = [package / "examples" / HELPER, "python3", "no_such_package", PYTHON, toy]
    done = run("/bin/sh", *words, cwd=package, INTEGER_MODEL_ENVIRONMENT_TRIED="1")
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr == f"{toy}: {NO_PACKAGE} either. {MAKE}\n"


def test_a_python_head_does_nothing_where_it_should_not_hand_over(package, tmp_path, run):
    """With an environment active; when the variable says that one was tried; fed on standard input;
    outside a checkout; and where someone who cannot change the script could change the helper.
    """
    toy, helper = package / "examples" / "toy.py", package / "examples" / HELPER
    helper.write_text("echo the helper ran\n")
    # cwd: python3 -m looks for modules in the working directory first.
    make = [PYTHON, "-B", "-m", "venv", "--without-pip", tmp_path / "active"]
    subprocess.run(make, cwd=tmp_path, check=True)

    def hands_over(*command, **more):
        done = run(*command, cwd=package / "examples", **more)
        assert (done.stdout, done.stderr) in (("the helper ran\n", ""), ("went on\n", "")), done
        return done.stdout == "the helper ran\n"

    assert hands_over(PYTHON, "-B", toy)
    assert not hands_over(tmp_path / "active" / "bin" / "python3", "-B", toy)
    assert not hands_over(PYTHON, "-B", toy, INTEGER_MODEL_ENVIRONMENT_TRIED="1")
    assert hands_over(PYTHON, "-B", toy, INTEGER_MODEL_ENVIRONMENT_TRIED="")
    with toy.open() as file:
        assert not hands_over(PYTHON, "-B", stdin=file)
    # Run by another program, which says that it is toy.py: the program that was started is not.
    names = f"{{'__name__': '__main__', '__file__': {str(toy)!r}}}"
    assert not hands_over(PYTHON, "-B", "-c", f"exec(open({str(toy)!r}).read(), {names})")
    helper.chmod(0o664)  # beside a script that its group may not write
    assert not hands_over(PYTHON, "-B", toy)
    toy.chmod(0o664)  # as in a checkout made under umask 0002
    assert hands_over(PYTHON, "-B", toy)
    for open_to_others in (package / "examples", toy, helper):
        mode = open_to_others.stat().st_mode
        open_to_others.chmod(mode | 0o002)
        assert not hands_over(PYTHON, "-B", toy), open_to_others
        open_to_others.chmod(mode)
    if ROOT:  # another owner's, and another group's
        for uid, gid in ((no_account(), 0), (0, no_account()), (0, 0)):
            os.chown(helper, uid, gid)
            assert hands_over(PYTHON, "-B", toy) == ((uid, gid) == (0, 0))
    helper.rename(tmp_path / "aside")
    helper.symlink_to(tmp_path / "aside")
    assert not hands_over(PYTHON, "-B", toy)
    helper.unlink()
    assert not hands_over(PYTHON, "-B", toy)
    # A copy outside a checkout, with a file of the helper's name where the head would look.
    for folder in ("y", "examples"):
        (tmp_path / "G" / folder).mkdir(parents=True)
    shutil.copy(toy, tmp_path / "G" / "y")
    (tmp_path / "G" / "examples" / HELPER).write_text("echo the helper ran\n")
    assert not hands_over(PYTHON, "-B", tmp_path / "G" / "y" / "toy.py")


def test_example_py_asks_only_for_the_float_pair(package, run):
    """Started by python3 -m; its helper lies in the sister folder examples/."""
    (package / "integer_model_vllm").mkdir()
    (package / "integer_model_vllm").chmod(0o755)
    (package / "integer_model_vllm" / "__init__.py").touch()
    text = (PACKAGE / "integer_model_vllm" / "example.py").read_text()
    end = "    del find_or_hand_over\n"
    head = text[: text.index(end) + len(end)].replace('"torch", "safetensors"', '"no_such_package"')
    (package / "integer_model_vllm" / "example.py").write_text(head + '\nprint("went on")\n')
    (package / "integer_model_vllm" / "example.py").chmod(0o644)
    (package / "examples" / HELPER).write_text('echo "the helper ran: $2"\n')

    def printed(*words):
        done = run(PYTHON, "-B", "-m", "integer_model_vllm.example", *words, cwd=package)
        assert done.stderr == ""
        return done.stdout

    assert printed("small_float_pair", "--quiet") == "the helper ran: no_such_package\n"
    for words in (
        ["small_synthetic"],
        ["vllm_files", "small_float_pair", "7"],
        ["--directory", "small_float_pair"],
        *(["small_float_pair", help] for help in ("-h", "--help", "--hel", "--h")),
    ):
        assert printed(*words) == "went on\n", words
    # examples/ must be no more open than example.py's own folder.
    (package / "examples").chmod(0o775)
    assert printed("small_float_pair") == "went on\n"
    (package / "integer_model_vllm").chmod(0o775)
    assert printed("small_float_pair") == "the helper ran: no_such_package\n"


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_a_shell_script_takes_vllm_from_the_environment_beside_the_package(
    package, tmp_path, run, name
):
    """As root the environment is not looked for, in root's own checkout or another's."""
    script = package / "examples" / name
    done = run(script, cwd=tmp_path)
    assert (done.returncode, done.stdout) == (2, "")
    if ROOT:
        assert done.stderr == f"{script}: {as_root(NO_VLLM)}"
        stand_in(package / ".venv" / "bin" / "vllm")
        give_away(package)
        assert run(script, cwd=tmp_path).__dict__ == done.__dict__
        return
    assert done.stderr == f"{script}: {NO_VLLM}, nor in {package}/.venv/bin. {MAKE}\n"
    stand_in(package / ".venv" / "bin" / "vllm")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "link.sh").symlink_to(script)
    (tmp_path / "other" / HELPER).write_text("echo a planted helper ran\n")
    for typed in (f"../integrations/vllm/integer_model/examples/{name}", "./link.sh"):
        done = run("/bin/sh", typed, cwd=tmp_path / "other")
        first, second = done.stderr.splitlines()[:2]
        assert first == f"{typed}: {NO_VLLM}: {package}/.venv/bin is now first on PATH."
        assert second.startswith("usage: ") and f" {script} " in second  # it went on, to its usage
        assert (done.returncode, done.stdout) == (2, "")
    # The arguments come through as they are. An empty PATH counts as none: no empty entry, which
    # would mean the working directory.
    (tmp_path / "other" / "caller.sh").write_text('printf "<%s>" "$PATH" "$@"\n')
    words = [package / "examples" / HELPER, "vllm", "x", tmp_path / "other" / "caller.sh"]
    done = run("/bin/sh", *words, "a b", "-c", "$(d)", "", cwd=tmp_path, PATH="")
    path = f"{package}/.venv/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    assert done.stdout == f"<{path}><a b><-c><$(d)><>"


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_a_shell_head_does_not_hand_over_where_it_should_not(package, tmp_path, run, name):
    """With vllm on PATH the script goes on. Where someone who cannot change the script could change
    the helper, it stops, whoever runs it.
    """
    script, helper = package / "examples" / name, package / "examples" / HELPER
    helper.write_text("echo the helper ran\n")
    stand_in(tmp_path / "bin" / "vllm")
    stopped = (
        f"{script}: vllm was not found on PATH. {HELPER} beside this script, which takes over in "
        "that case, is missing, is a link, or it or its folder fails the check of owner, group and "
        "permissions. Nothing was made. See "
        'integrations/vllm/integer_model/README.md, "Install and quick start".\n'
    )

    def hands_over():
        done = run(script, cwd=tmp_path)
        outcome = (done.returncode, done.stdout, done.stderr)
        assert outcome in ((0, "the helper ran\n", ""), (2, "", stopped)), done
        return done.returncode == 0

    assert hands_over()
    done = run(script, cwd=tmp_path, PATH=f"{tmp_path}/bin:{tmp_path}/tools")
    assert (done.returncode, done.stdout) == (2, "") and done.stderr.startswith("usage: ")
    helper.chmod(0o664)  # beside a script that its group may not write
    assert not hands_over()
    script.chmod(0o775)  # as in a checkout made under umask 0002
    (package / "examples").chmod(0o775)
    assert hands_over()
    (package / "examples").chmod(0o757)  # others could exchange the helper
    assert not hands_over()
    (package / "examples").chmod(0o755)
    if ROOT:  # another owner's, and another group's
        for uid, gid in ((no_account(), 0), (0, no_account()), (0, 0)):
            os.chown(helper, uid, gid)
            assert hands_over() == ((uid, gid) == (0, 0))
    helper.rename(tmp_path / "aside")
    helper.symlink_to(tmp_path / "aside")
    assert not hands_over()
    helper.unlink()
    assert not hands_over()
    # A backslash in the path by which the script is called is printed as it is.
    (tmp_path / "a\\cb").symlink_to(package / "examples")
    typed = tmp_path / "a\\cb" / name
    done = run(typed, cwd=tmp_path)
    outcome = (done.returncode, done.stdout, done.stderr)
    assert outcome == (2, "", stopped.replace(str(script), str(typed)))


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_a_shell_script_must_be_started_by_a_name_that_tells_its_place(
    package, tmp_path, run, name
):
    """Sourced or read from standard input, "$0" is the shell's name; after "sh NAME" it is NAME.
    Whatever lies at a name without a slash in the working directory, the script stops, with and
    without vllm on PATH.
    """
    script = package / "examples" / name
    (package / "examples" / HELPER).write_text("echo the helper ran\n")
    (tmp_path / HELPER).write_text("echo a planted helper ran\n")
    other = checkout(tmp_path / "other")  # another checkout, with a helper of its own
    (other / "examples" / HELPER).write_text("echo a planted helper ran\n")
    stand_in(tmp_path / "bin" / "vllm")
    said = (
        f": start {name} by a path with a slash, such as ./{name} or its full path. It cannot be "
        "sourced or read from standard input.\n"
    )
    for lay in (lambda path: None, pathlib.Path.touch, pathlib.Path.unlink, None):
        if lay is None:
            (tmp_path / "sh").symlink_to(other / "examples" / name)
        else:
            lay(tmp_path / "sh")
        for path in (f"{tmp_path}/tools", f"{tmp_path}/bin:{tmp_path}/tools"):
            with script.open() as file:
                fed = run("sh", stdin=file, cwd=tmp_path, PATH=path)
            for done in (fed, run("sh", "-c", f". {script}", cwd=tmp_path, PATH=path)):
                assert (done.returncode, done.stdout, done.stderr) == (2, "", "sh" + said)
    # The name is shown in printable ASCII, and a backslash in it is not acted on.
    done = run("sh", "-c", f". {script}", "o\x1bd\\cd\nx", cwd=tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (2, "", "o?d\\cd?x" + said)
    # A copy outside a checkout stops too.
    shutil.copy(script, tmp_path / name)
    done = run(tmp_path / name, cwd=tmp_path)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr.startswith(
        f"{tmp_path / name}: run this script by its path inside a checkout"
    )
    # The script's own bare name is refused too: inside examples/, and as a link elsewhere. With a
    # slash before it, the same name is taken.
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / name).symlink_to(script)
    for folder in (package / "examples", tmp_path / "mine"):
        for path in (f"{tmp_path}/tools", f"{tmp_path}/bin:{tmp_path}/tools"):
            done = run("sh", name, cwd=folder, PATH=path)
            assert (done.returncode, done.stdout, done.stderr) == (2, "", name + said)
        assert run("sh", f"./{name}", cwd=folder).stdout == "the helper ran\n"


def test_the_caller_is_started_again_only_once(tmp_path, run):
    """A colon in the path breaks PATH's new entry, so the script comes back to the helper. Root
    is stopped before that.
    """
    package = checkout(tmp_path / "a:b")
    stand_in(package / ".venv" / "bin" / "vllm")
    script = package / "examples" / "serve_gguf.sh"
    done = run(script, cwd=tmp_path)
    assert (done.returncode, done.stdout) == (2, "")
    twice = (
        f"{script}: {NO_VLLM}: {package}/.venv/bin is now first on PATH.\n"
        f"{script}: vllm was still not found with {package}/.venv/bin first on PATH. {MAKE}\n"
    )
    assert done.stderr == (f"{script}: {as_root(NO_VLLM)}" if ROOT else twice)


def test_the_lines_show_printable_ascii_alone(tmp_path, run):
    package = checkout(tmp_path / "o\x1bdd\nñ")
    done = run(package / "examples" / "serve_gguf.sh", cwd=tmp_path)
    shown = f"{tmp_path}/o?dd???/integrations/vllm/integer_model"
    said = as_root(NO_VLLM) if ROOT else f"{NO_VLLM}, nor in {shown}/.venv/bin. {MAKE}\n"
    assert done.stderr == f"{shown}/examples/serve_gguf.sh: {said}"


def test_the_helper_is_not_meant_to_be_run_by_hand(package, tmp_path, run):
    helper = package / "examples" / HELPER
    said = ": the scripts of the package start this file; it is not meant to be run by hand\n"
    for words in ([], ["vllm", "x"], ["python3", "a", "b"], ["other", "a", "b", "c", "d"]):
        done = run("/bin/sh", helper, *words, cwd=package)
        assert (done.returncode, done.stdout, done.stderr) == (2, "", f"{helper}{said}")
    # Read from standard input, "$0" is the shell's name: whatever lies at that name in the working
    # directory, here a link into another checkout with a python3 of its own, is not looked at.
    other = checkout(tmp_path / "other")
    stand_in(other / ".venv" / "bin" / "python3")
    (tmp_path / "sh").symlink_to(other / "examples" / HELPER)
    for words in (["python3", "a", "b", "c"], ["x/y:python3", "a", "b", "c"], ["vllm", "a", "b"]):
        with helper.open() as file:
            done = run("sh", "-s", *words, stdin=file, cwd=tmp_path)
        assert (done.returncode, done.stdout, done.stderr) == (2, "", f"sh{said}")
