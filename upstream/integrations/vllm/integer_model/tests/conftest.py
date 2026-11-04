# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Shared set-up for the tests: the skips that the markers imply, and the fixtures."""

import importlib
import os
import pathlib
import re
import shutil
import stat
import subprocess

import pytest
import torch

import vllm_tools
from block_tools import FILE_MODEL, PACKAGE, SMALL_EXAMPLE, TREE
from integer_model.block_weights import BlockWeights
from integer_model.parameters import Parameters

# Set when pytest's base temporary directory lies below a place that another account may be
# able to change: example.py and try_limits.py refuse to work there, so no test runs.
OPEN_GROUND = pytest.StashKey[str]()


# fmt: off
# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")
# fmt: on


def pytest_sessionstart(session):
    """Looks at pytest's base temporary directory with the routine above."""
    base = session.config._tmp_path_factory.getbasetemp()  # creates it, as every tmp_path would
    found = others_can_change(str(base))
    if found:
        place, why = found
        line = f"stood aside: below an open place: {printable(place)} ({whose(place, why)})"
        session.config.stash[OPEN_GROUND] = line


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if OPEN_GROUND in session.config.stash:
        return True  # run no test
    return None


def pytest_sessionfinish(session, exitstatus):
    if OPEN_GROUND in session.config.stash:
        session.exitstatus = 77  # the test runner's code for a row that stood aside


def pytest_unconfigure(config):
    if OPEN_GROUND in config.stash:
        print(config.stash[OPEN_GROUND], flush=True)  # after pytest's own last line


def pytest_configure(config):
    # The tests make folders with mkdir. Under a umask of 002, common where each user has a private
    # group, their group could write in them, and example.py and try_limits.py refuse to work below
    # such a folder. So this session's umask keeps group and others from writing, as run_suite.py's
    # does.
    os.umask(os.umask(0) | 0o022)
    # By default PyTorch starts one thread per CPU, which makes these small models several times
    # slower. Setting OMP_NUM_THREADS as well passes the same limit to the scripts that the tests
    # start, which set_num_threads does not reach.
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())


def reasons_to_skip() -> dict[str, str]:
    """For each marker of pyproject.toml whose tests cannot run here, the reason."""
    reasons = {}
    hidden = vllm_tools.HAS_GPU and os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
    if not vllm_tools.HAS_GPU:
        reasons["gpu"] = "needs an NVIDIA GPU"
    elif hidden:
        reasons["gpu"] = "CUDA_VISIBLE_DEVICES hides the GPU; unset it"
    if vllm_tools.installed_version("vllm") != "0.29.0":
        reasons["vllm"] = "needs vLLM 0.29.0"
    elif hidden:  # vLLM would still try to use the GPU that nvidia-smi lists
        reasons["vllm"] = "CUDA_VISIBLE_DEVICES hides the GPU; unset it"
    if not os.environ.get("INTEGER_MODEL_PROVER_BUILD"):
        reasons["prover"] = (
            "INTEGER_MODEL_PROVER_BUILD does not name the directory with the prover's programs "
            "(tests/README.md)"
        )
    return reasons


def pytest_collection_modifyitems(config, items):
    reasons = reasons_to_skip()
    for item in items:
        for marker, reason in reasons.items():
            if item.get_closest_marker(marker):
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def device() -> str:
    """The PyTorch device on which the tests run the model."""
    return os.environ.get("INTEGER_MODEL_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session")
def compiler(tmp_path_factory) -> None:
    """Skips the test if g++ is missing, or if pytest's temporary directory does not let programs
    run. The version of g++ is not checked.
    """
    if shutil.which("g++") is None:
        pytest.skip("needs g++ 13 or later")
    probe = tmp_path_factory.mktemp("probe") / "probe"
    probe.write_text("#!/bin/sh\n")
    probe.chmod(0o700)
    try:
        subprocess.run([probe], check=True)
    except PermissionError:
        pytest.skip(
            f"{probe.parent} does not let programs run (mounted noexec?). First make a folder on a "
            "file system that does: dir=$(mktemp -d -p DIR), where DIR is a folder of yours there. "
            "If that succeeds, run pytest again "
            'with your other arguments and --basetemp="${dir:?}/t" (pytest empties the folder that '
            "--basetemp names)."
        )


@pytest.fixture(scope="session")
def programs(compiler, tmp_path_factory) -> pathlib.Path:
    """The directory with block_reference and make_model, built once per run."""
    folder = tmp_path_factory.mktemp("block_programs")
    for source in (
        PACKAGE / "tests" / "block_reference.cpp",
        TREE / "prover/tests/weights_blocks/make_model.cpp",
    ):
        # -fno-ipa-modref guards against a bug in g++ 13.3.0 that can silently drop the stores that
        # fill an object passed by reference.
        subprocess.run(
            ["g++", "-O2", "-fno-ipa-modref", "-std=c++17", "-pthread", "-I", TREE / "prover"]
            + ["-o", folder / source.stem, source],
            check=True,
        )
    return folder


@pytest.fixture(scope="session")
def file_model(programs, tmp_path_factory) -> pathlib.Path:
    """A small model: a directory with policy.txt, KL.bin and the subdirectory model (two GGUF
    files, wsums.txt, fold.txt).
    """
    folder = tmp_path_factory.mktemp("file_model")
    lines = [line.split() for line in (SMALL_EXAMPLE / "policy.txt").read_text().splitlines()]
    replaced = [
        [line[0], str(FILE_MODEL[line[0]])] if line[0] in FILE_MODEL else line for line in lines
    ]
    (folder / "policy.txt").write_text("".join(" ".join(line) + "\n" for line in replaced))
    shutil.copyfile(SMALL_EXAMPLE / "KL.bin", folder / "KL.bin")  # writable, whatever KL.bin's mode
    shape = [FILE_MODEL[word] for word in ("L", "H", "G", "f", "fe", "NE", "V")]
    seed = 30
    subprocess.run(
        [programs / "make_model", folder / "model", *map(str, shape), str(seed)],
        check=True,
        capture_output=True,
    )
    return folder


@pytest.fixture(scope="session")
def file_weights(file_model, device) -> BlockWeights:
    shards = sorted(
        str(path.relative_to(file_model)) for path in (file_model / "model").glob("*.gguf")
    )
    return BlockWeights(
        Parameters.read(file_model / "policy.txt"),
        device,
        folder=file_model,
        blocks="gguf",
        setup_folder="model",
        shards=shards,
        hash_integers=True,
    )


@pytest.fixture(scope="session")
def synthetic_weights(device) -> BlockWeights:
    """The synthetic model in blocks for prover/examples/target_small_f2/policy.txt, loaded."""
    return BlockWeights(
        Parameters.read(SMALL_EXAMPLE / "policy.txt"),
        device,
        blocks="synthetic",
        hash_integers=True,
    )


@pytest.fixture(scope="session")
def installed_plugin(tmp_path_factory):
    """Installs the distribution into a temporary directory and sets those of vLLM's environment
    variables that are not set already, both for the length of the run.

    Each of vLLM's processes then finds the plug-in through its entry point, and the stand-in
    (stand_in_integer_model.py) beside this file. Returns the directory with the installed
    packages.
    """
    site = vllm_tools.install(str(tmp_path_factory.mktemp("installed_plugin")))
    environment = dict(vllm_tools.ENVIRONMENT)
    if vllm_tools.HAS_GPU:
        del environment["VLLM_TARGET_DEVICE"]
        # An earlier test may have initialized CUDA in this process: a forked engine would fail.
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    else:
        environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the engine runs in the test's process
    with pytest.MonkeyPatch.context() as patch:
        for name, value in environment.items():
            if name not in os.environ:
                patch.setenv(name, value)
        patch.setenv("PYTHONPATH", os.pathsep.join([site, vllm_tools.HERE]), prepend=os.pathsep)
        patch.syspath_prepend(site)
        yield site


@pytest.fixture(scope="session")
def vllm(installed_plugin):
    """The module vllm. It reads its environment when it is imported, so the tests take it from
    here.
    """
    return importlib.import_module("vllm")
