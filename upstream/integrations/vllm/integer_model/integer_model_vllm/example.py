# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Creates the files of an example that vLLM can serve, with synthetic weights. Needs no vLLM.

usage: python3 -m integer_model_vllm.example NAME
       python3 -m integer_model_vllm.example vllm_files MODEL_DIRECTORY END_TOKEN_IDS

    small_synthetic   a small model whose weights are computed from a seed
    full_size         Llama 4 Maverick's shape, synthetic quantized blocks; 238 GB of free host
                      memory
    small_gguf        a small model in GGUF files, for OPERATING.md, "Serve from GGUF files"
                      (needs g++)
    small_float_pair  a small model and, in float_model/, the same model in floating point (needs
                      PyTorch and safetensors)
    vllm_files        writes config.json and a word-list tokenizer into your model directory

An example prints the new directory's path on standard output and, unless --quiet is given, the
command that serves it on standard error. A directory below a folder that another account may be
able to change is refused with exit code 2, before anything is made; so is such a model directory
for vllm_files. examples/serve.sh creates an example and serves it.
"""

from __future__ import annotations

if __name__ == "__main__":
    # Where no virtual environment is active and a package named below is not found,
    # examples/find_or_give_root_up.sh takes over: it starts this program again with
    # .venv/bin/python3 beside the package, or, as root, stops and says what works.
    import os
    import sys

    def find_or_hand_over(*names: str) -> None:
        """Asks the finders that Python has loaded, which runs none of the packages' code. Starts
        the helper only if this file, at its place in a checkout, is the program that was started,
        and only if the helper and its folder have the owner and group of this file and its folder
        and no write permission for the group that these lack, and others may write in none of them.
        """
        finders = [finder for finder in sys.meta_path if hasattr(finder, "find_spec")]
        missing = [n for n in names if not any(f.find_spec(n, None) for f in finders)]
        tried = os.environ.get("INTEGER_MODEL_ENVIRONMENT_TRIED")
        typed = getattr(sys, "orig_argv", None)  # since Python 3.10
        if not missing or tried or not typed or sys.prefix != sys.base_prefix:
            return
        this = os.path.realpath(globals().get("__file__") or os.devnull)
        folder = os.path.dirname(this)
        package, name = os.path.split(folder)
        examples = os.path.join(package, "examples")
        helper = os.path.join(examples, "find_or_give_root_up.sh")
        placed = package.endswith("/integrations/vllm/integer_model")
        placed = placed and name in ("examples", "integer_model_vllm")
        if not placed or os.path.realpath(sys.argv[0]) != this:
            return
        try:
            pairs = [(os.stat(this), os.lstat(helper)), (os.stat(folder), os.lstat(examples))]
        except OSError:
            return
        regular = pairs[0][1].st_mode & 0o170000 == 0o100000  # a file, not a link to one
        if regular and all(
            (mine.st_uid, mine.st_gid) == (its.st_uid, its.st_gid)
            and not (mine.st_mode | its.st_mode) & 0o002
            and not its.st_mode & 0o020 & ~mine.st_mode
            for mine, its in pairs
        ):
            words = ["python3", missing[0], sys.executable, sys.argv[0], *typed[1:]]
            os.execv("/bin/sh", ["/bin/sh", helper, *words])

    # Only this example needs them; -h, --help and its abbreviations (--h, --he, --hel) do not.
    if sys.argv[1:2] == ["small_float_pair"] and not [
        option for option in sys.argv if option.startswith(("-h", "--h"))
    ]:
        find_or_hand_over("torch", "safetensors")
    del find_or_hand_over

import argparse
import contextlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from . import ARCHITECTURE, LOAD_FORMAT

# name: the directory under prover/examples that holds policy.txt and KL.bin, and the content of
# integer_model.json.
SYNTHETIC = {
    "small_synthetic": ("target_small_f2", {"weights": "synthetic"}),
    "full_size": (
        "target_full_f2",
        {"weights": "blocks", "weight_settings": {"blocks": "synthetic"}},
    ),
}
# small_gguf: the shape lines of its policy.txt (make_model fixes d and dh). Its other lines and its
# KL.bin, the quantization scales, are small_synthetic's (prover/examples/target_small_f2), so it
# has 8 layers.
GGUF_SHAPE = {"L": 8, "d": 5120, "H": 2, "G": 1, "dh": 128, "f": 512, "fe": 256, "NE": 2, "V": 512}
GGUF_SEED = 30  # examples/request.sh's reply for small_gguf depends on it
# serve_gguf.sh's five small files; used by the tests.
REFERENCE_FILES = ("integer_model.json", "policy.txt", "KL.bin", "wsums.txt", "fold.txt")

ENVIRONMENT = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "PYTHONSAFEPATH": "1",
}


def write_vllm_files(
    directory: str | os.PathLike,
    vocabulary_size: int,
    window_length: int,
    end_token: int | list[int],
) -> None:
    """Writes what vLLM itself reads from a model directory: config.json and a tokenizer.

    The directory must exist, and an existing file is an error. config.json must name a model type
    that transformers knows. `architectures` selects the model class, which reads vocab_size, and
    eos_token_id if there is no generation_config.json, but none of the other fields. vLLM accepts
    no --max-model-len above max_position_embeddings.

    In the tokenizer the word "w<n>" is token n, but the end token (the first, if several) is
    "<end>"; any other word is token 0. The last token's word is an added token; only then does vLLM
    reject the id vocabulary_size in a prompt (check_tokenizer() in model.py).
    """
    first_end = end_token if isinstance(end_token, int) else end_token[0]
    words = ["<end>" if i == first_end else f"w{i}" for i in range(vocabulary_size)]
    last = vocabulary_size - 1
    files = {
        "config.json": {
            "architectures": [ARCHITECTURE],
            "model_type": "llama",
            "vocab_size": vocabulary_size,
            "eos_token_id": end_token,
            "max_position_embeddings": window_length,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
        },
        "tokenizer.json": {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": last,
                    "content": words[last],
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": last == first_end,
                }
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {word: i for i, word in enumerate(words[:last])},
                "unk_token": words[0],
            },
        },
        "tokenizer_config.json": {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": words[0],
            "eos_token": "<end>",
        },
    }
    for name, value in files.items():
        with open(Path(directory, name), "x") as f:
            json.dump(value, f)


def new_file(path: Path, data: bytes | str) -> None:
    """Writes data into a new file at path, made with O_CREAT | O_EXCL, which refuses a file or a
    link that stands at that name: nothing is overwritten, also if something is put there after this
    script's own mkdir. A link put in place of a directory above it is still followed, and the new
    file is then made there. The program that g++ writes for small_gguf is built and run in a
    folder of its own that only this account can write in (build_folder()).
    """
    with open(path, "xb" if isinstance(data, bytes) else "x") as file:
        file.write(data)


def copy_new(source: Path, folder: Path) -> None:
    """Copies the file source into a new file of the same name in folder (see new_file)."""
    new_file(folder / source.name, source.read_bytes())


def sizes(parameter_file: Path) -> tuple[int, int]:
    """The vocabulary size V and the window length T = 2^lt that a parameter file states."""
    lines = dict(line.split(maxsplit=1) for line in parameter_file.read_text().splitlines())
    return int(lines["V"]), 1 << int(lines["lt"])


def find_repository(given: str | None) -> Path:
    """The top of the FlashProve checkout that has prover/examples/target_small_f2/policy.txt.

    Without a given path, it is a directory above this file's resolved path whose
    integrations/vllm/integer_model is also above it. The current directory is not searched:
    small_gguf compiles and runs a program from the checkout, so it must not be one that someone
    else planted.
    """
    here = Path(__file__).resolve()
    package = Path("integrations", "vllm", "integer_model")
    tops = (
        [Path(given)] if given else [top for top in here.parents if (top / package) in here.parents]
    )
    for top in tops:
        if (top / "prover" / "examples" / "target_small_f2" / "policy.txt").is_file():
            return top.resolve()
    if given:
        sys.exit(
            f"{printable(given)} has no prover/examples/target_small_f2/policy.txt: it is not the "
            "top of the FlashProve checkout."
        )
    sys.exit(
        "This package is not inside a checkout of FlashProve. Pass the checkout's top as "
        "--repository."
    )


def make_synthetic(name: str, top: Path, repository: Path) -> list[str]:
    """Creates model/ for small_synthetic or full_size. Returns the command that serves it."""
    configuration, weights = SYNTHETIC[name]
    model = top / "model"
    model.mkdir()
    for file in ("policy.txt", "KL.bin"):
        copy_new(repository / "prover" / "examples" / configuration / file, model)
    new_file(model / "integer_model.json", json.dumps(weights) + "\n")
    vocabulary_size, window_length = sizes(model / "policy.txt")
    write_vllm_files(model, vocabulary_size, window_length, end_token=vocabulary_size - 1)
    return [
        *(f"{variable}={value}" for variable, value in ENVIRONMENT.items()),
        *("vllm", "serve", str(model), "--host", "127.0.0.1", "--served-model-name", name),
        *(
            "--load-format",
            LOAD_FORMAT,
            "--max-num-seqs",
            "1",
            "--max-model-len",
            str(window_length),
        ),
        *("--no-enable-prefix-caching", "--enforce-eager", "--generation-config", "vllm"),
    ]


def others_can_change_under_sudo(folder: str) -> tuple[str, str] | None:
    """others_can_change(folder), but trusting no account that ran sudo: for a folder in which root
    writes or runs something that that account must not be able to swap."""
    sudo_uid = os.environ.pop("SUDO_UID", None)
    try:
        return others_can_change(folder)
    finally:
        if sudo_uid is not None:
            os.environ["SUDO_UID"] = sudo_uid


def or_unset_it() -> str:
    """The words ", or unset it" where TMPDIR is set, for sentences that say how to change it."""
    return ", or unset it" if os.environ.get("TMPDIR") else ""


def folder_for_temporary_files() -> str:
    """TMPDIR, or /tmp if it is unset or empty, as mktemp takes it: not tempfile.gettempdir(),
    which makes a probe file in it before anything has looked, and passes over a TMPDIR that does
    not exist to the next place that works (/tmp, /var/tmp, the working directory)."""
    return os.path.abspath(os.environ.get("TMPDIR") or "/tmp")


def build_folder() -> Path:
    """A new folder for building and running make_model, made by tempfile.mkdtemp() (mode 700).

    The folder for temporary files that will hold it is looked at first, with the routine below,
    which here does not trust the account that ran sudo: under sudo, that account could otherwise
    change a folder above the new one and swap the program that root then runs. If anyone but this
    account or root could change that folder, or one above it, the run stops with status 2 before
    anything is made. (Whoever can change the checkout, under sudo too, chooses what root runs
    anyway: make_model's source and every module of this package come from it.)
    """
    # The folder for temporary files is looked at first, as given and by its real path, so that a
    # refusal leaves nothing; the new folder is then made in that real path.
    parent = folder_for_temporary_files()
    real = os.path.realpath(parent)
    shown = printable(parent)
    found = others_can_change_under_sudo(parent) or others_can_change_under_sudo(real)
    if not found and os.path.realpath(real) != real:  # checked from the top: no link can come now
        print(f"{shown} changed while it was looked at. Nothing was built.", file=sys.stderr)
        sys.exit(2)
    if found and found[1] == "does not exist":
        words = f"TMPDIR, {shown}, does not exist. Set TMPDIR to a directory of yours"
        words += or_unset_it()
        print(f"integer_model_vllm.example: {words}. Nothing was made.", file=sys.stderr)
        sys.exit(2)
    if found:
        words = refusal("folder for temporary files", parent, *found)
        words += (
            ". Set TMPDIR to a directory that only this account can write in" + or_unset_it() + "."
        )
        print("integer_model_vllm.example: " + words, file=sys.stderr)
        sys.exit(2)
    try:
        return Path(tempfile.mkdtemp(prefix="make_model.", dir=real))
    except OSError as error:  # a TMPDIR that this account cannot write in, or that is a file
        words = f"cannot make a folder in TMPDIR, {shown}: {error.strerror}. Set TMPDIR to a"
        words += " directory that you can write in" + or_unset_it()
        print(f"integer_model_vllm.example: {words}. Nothing was made.", file=sys.stderr)
        sys.exit(2)


def make_gguf(top: Path, repository: Path, build: Path) -> list[str]:
    """Creates the files of small_gguf. Returns the command that serves it.

    weights/    the GGUF files, 70 MB, with wsums.txt and fold.txt (the set-up directory)
    reference/  your reference copies of the five small files, for the digests
    model/      the model directory

    The program that writes the GGUF files, make_model, is built from prover/tests/weights_blocks
    and run in build, a new folder of its own that only this account can write in (build_folder()),
    and removed afterwards (the folder, if nothing else lies in it), so that nobody can put another
    program in its place.
    """
    prover = repository / "prover"
    weights, reference, model = top / "weights", top / "reference", top / "model"

    # make_model writes weights/. In real use those files come with a commitment, as do policy.txt
    # and KL.bin; this example computes none. -fno-ipa-modref turns off a pass of g++ 13.3.0 that
    # can treat memory as unread when it is read.
    program = build / "make_model"
    source = prover / "tests" / "weights_blocks" / "make_model.cpp"
    # g++ and make_model get PATH alone of the caller's environment: variables such as CPATH,
    # COMPILER_PATH or DEPENDENCIES_OUTPUT would lead the compiler (the last makes it append to a
    # file that it names), and g++'s own temporary files go into the closed folder too.
    small_environment = {"PATH": os.environ.get("PATH", os.defpath), "TMPDIR": str(build)}
    small_environment["LC_ALL"] = "C"
    try:
        subprocess.run(
            [
                "g++",
                "-O2",
                "-fno-ipa-modref",
                "-std=c++17",
                "-pthread",
                "-I",
                prover,
                "-o",
                program,
                source,
            ],
            check=True,
            cwd=build,
            env=small_environment,
        )
        shape = [str(GGUF_SHAPE[name]) for name in ("L", "H", "G", "f", "fe", "NE", "V")]
        try:
            subprocess.run(
                [program, weights, *shape, str(GGUF_SEED)],
                check=True,
                stdout=subprocess.DEVNULL,
                env=small_environment,
            )
        except PermissionError:
            sys.exit(
                f"Cannot run make_model in {printable(str(build.parent))}: its file system may not "
                "let programs run (mounted noexec?). Set TMPDIR to a folder on one that does. The "
                f"example is incomplete. What was written is in {printable(str(top))}."
            )
    finally:  # the two things that this function made, each by its own name
        if program.exists():
            os.unlink(program)
        with contextlib.suppress(OSError):  # so as not to hide the error on its way, if any
            os.rmdir(build)
    configuration = prover / "examples" / SYNTHETIC["small_synthetic"][0]
    lines = [line.split() for line in (configuration / "policy.txt").read_text().splitlines()]
    policy = "".join(f"{line[0]} {GGUF_SHAPE.get(line[0], ' '.join(line[1:]))}\n" for line in lines)

    # Steps 1 and 2 of OPERATING.md, "Serve from GGUF files": reference copies, in a directory
    # only you can access.
    reference.mkdir(mode=0o700)
    new_file(reference / "policy.txt", policy)
    copy_new(configuration / "KL.bin", reference)
    for file in ("wsums.txt", "fold.txt"):
        copy_new(weights / file, reference)
    settings = {
        "blocks": "gguf",
        "setup_folder": str(weights),
        "shards": sorted(str(path) for path in weights.glob("*.gguf")),
        "verify_sums": True,
    }
    description = {"weights": "blocks", "weight_settings": settings}
    new_file(reference / "integer_model.json", json.dumps(description, indent=2) + "\n")

    # Step 3: the model directory, from the copies.
    model.mkdir()
    for file in ("policy.txt", "KL.bin", "integer_model.json"):
        copy_new(reference / file, model)
    vocabulary_size, window_length = sizes(model / "policy.txt")
    write_vllm_files(model, vocabulary_size, window_length, end_token=vocabulary_size - 1)

    # Step 4: the script starts vLLM with the path of the directory of reference copies.
    script = repository / "integrations" / "vllm" / "integer_model" / "examples" / "serve_gguf.sh"
    return [
        "VLLM_API_KEY=example",
        str(script),
        str(reference),
        str(model),
        "small_gguf",
        str(window_length),
    ]


def add_vllm_files(words: list[str]) -> None:
    """vllm_files: writes write_vllm_files()'s files into a model directory that has policy.txt,
    or nothing if one of them exists.
    """
    if len(words) != 2 or not re.fullmatch("[0-9]+(,[0-9]+)*", words[1]):
        sys.exit(
            "usage: python3 -m integer_model_vllm.example vllm_files MODEL_DIRECTORY END_TOKEN_IDS"
            "\nEND_TOKEN_IDS as in 7,9: eos_token_id in the publisher's generation_config.json"
        )
    model, ends = Path(words[0]), [int(token) for token in words[1].split(",")]
    # Another account must not be able to change the model directory or a folder above it: looked at
    # as given and by its real path, by which everything below is then read and written, once sure
    # that it did not turn. The look comes first, so that nothing is read by the path as given.
    real, shown = os.path.realpath(model), printable(str(model))
    for place in (str(model), real):
        found = others_can_change(place)
        if found and found[1] == "does not exist":
            missing = "" if found[0] == str(model) else f": {printable(found[0])} is missing"
            print(
                f"integer_model_vllm.example: the model directory {shown} does not exist{missing}.",
                file=sys.stderr,
            )
            sys.exit(2)
        if found:
            said = refusal("model directory", str(model), *found)
            said += (
                ". Here the directory must hold the model's files: move them, or copy them, into"
            )
            said += " a new directory below a folder that only you can write in, and name that one."
            print("integer_model_vllm.example: " + said, file=sys.stderr)
            sys.exit(2)
    if os.path.realpath(real) != real:  # checked from the top: no link can come now
        print(f"{shown} changed while it was looked at. Nothing was written.", file=sys.stderr)
        sys.exit(2)
    model, shown = Path(real), printable(real)
    try:
        vocabulary_size, window_length = sizes(model / "policy.txt")
    except OSError as error:
        sys.exit(f"Cannot read the parameter file {shown}/policy.txt: {error.strerror}.")
    except (KeyError, ValueError):
        sys.exit(f"{shown}/policy.txt is not a parameter file with the lines V and lt.")
    if vocabulary_size < 2:
        sys.exit(f"{shown}/policy.txt: the vocabulary size V must be at least 2.")
    if max(ends) >= vocabulary_size:
        sys.exit(f"An end token id must be below the vocabulary size, {vocabulary_size}.")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json")
    # lexists(): a link whose target is missing counts as there too; write_vllm_files() refuses it.
    if found := [name for name in names if os.path.lexists(model / name)]:
        sys.exit(
            f"{shown} has {', '.join(found)} already. Nothing was written. This command is for a "
            "model directory that has none of the three: leave these files as they are, and run it "
            "on a new directory that holds only policy.txt, KL.bin and integer_model.json."
        )
    try:
        write_vllm_files(model, vocabulary_size, window_length, ends[0] if len(ends) == 1 else ends)
    except FileExistsError as error:  # one of the names came to stand after the look above
        sys.exit(
            f"{printable(str(error.filename))} exists already; nothing is overwritten. The files "
            f"written before it stay in {shown}."
        )
    print(
        f"Wrote {', '.join(names)} into {shown}: vocab_size {vocabulary_size}, "
        f"max_position_embeddings {window_length}, eos_token_id {words[1]}."
    )


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


def main(arguments: list[str] | None = None) -> None:
    """Creates an example's files in a new directory, or runs vllm_files. Overwrites nothing."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments[:1] == ["vllm_files"]:
        return add_vllm_files(arguments[1:])
    parser = argparse.ArgumentParser(
        prog="python3 -m integer_model_vllm.example",
        description="Creates the files of an example that vLLM can serve, with synthetic weights.",
        epilog="The module's docstring describes the four examples, and the command vllm_files.",
    )
    parser.add_argument("example", choices=[*SYNTHETIC, "small_gguf", "small_float_pair"])
    parser.add_argument(
        "--directory",
        help="the new directory; it must not exist yet (default: a new temporary directory)",
    )
    parser.add_argument(
        "--repository", help="the top of the FlashProve checkout (default: the one with this file)"
    )
    parser.add_argument("--quiet", action="store_true", help="print only the new directory's path")
    given = parser.parse_args(arguments)
    repository = find_repository(given.repository)
    if given.example == "small_gguf" and shutil.which("g++") is None:
        sys.exit(
            "small_gguf needs g++, to build the program that writes the GGUF files "
            "(prover/tests/weights_blocks)."
        )
    if given.example == "small_float_pair":
        missing = [
            name for name in ("torch", "safetensors") if importlib.util.find_spec(name) is None
        ]
        if missing:
            sys.exit(
                f"small_float_pair needs the {'package' if len(missing) == 1 else 'packages'} "
                f"{' and '.join(missing)}, to write the floating-point model. "
                f"{'It comes' if len(missing) == 1 else 'They come'} with vLLM's pinned packages. "
                "In integrations/vllm/integer_model: "
                "pip install --require-hashes -r ../requirements.txt"
            )
    # The directory's parent: another account must not be able to change it or a folder above it.
    # The look is always made: the routine refuses a missing parent too, which another account
    # could otherwise make between a test and the mkdir. The folder for temporary files is looked at
    # without trust in the account that ran sudo, as examples/serve_gguf.sh looks at it: under sudo
    # root makes the example there.
    chosen = given.directory is not None
    new = os.path.abspath(given.directory if chosen else folder_for_temporary_files() + "/x")
    parent = os.path.dirname(new)
    look = others_can_change if chosen else others_can_change_under_sudo
    # The real path is looked at too, and the directory made there: a link on the way could
    # otherwise lead the mkdir elsewhere than the look went.
    real, shown = os.path.realpath(parent), printable(parent)
    found = look(parent) or look(real)
    if not found and os.path.realpath(real) != real:  # checked from the top: no link can come now
        print(f"{shown} changed while it was looked at. Nothing was made.", file=sys.stderr)
        sys.exit(2)
    if found and found[1] == "does not exist":  # the routine's own words fit this case badly
        if chosen:
            asked, missing = printable(given.directory), printable(found[0])
            words = f"Cannot create {asked}: {missing} does not exist"
        else:
            words = f"TMPDIR, {shown}, does not exist. Set TMPDIR to a directory of yours"
            words += or_unset_it()
        print(f"integer_model_vllm.example: {words}. Nothing was made.", file=sys.stderr)
        sys.exit(2)
    if found:
        if chosen:
            words = refusal("directory", new, *found)
        else:
            words = refusal("folder for temporary files", parent, *found)
            words += ". Set TMPDIR to such a folder, or give --directory."
        print("integer_model_vllm.example: " + words, file=sys.stderr)
        sys.exit(2)
    parent = real
    # small_gguf's folder for make_model is made and looked at before the example's directory, so
    # that its refusal leaves nothing behind.
    build = build_folder() if given.example == "small_gguf" else None
    try:
        if not chosen:
            top = Path(tempfile.mkdtemp(prefix=f"integer-model-{given.example}-", dir=parent))
        else:
            # Group and others may not write in it, whatever the umask, so that the next script's
            # look at it (examples/try_limits.py) finds nothing open.
            os.mkdir(os.path.join(parent, os.path.basename(new)), 0o755)
            top = Path(parent, os.path.basename(new))
    except OSError as error:
        if build is not None:
            os.rmdir(build)
        if chosen:
            sys.exit(f"Cannot create {printable(given.directory)}: {error.strerror}.")
        words = f"cannot make a folder in TMPDIR, {printable(parent)}: {error.strerror}. Set TMPDIR"
        words += " to a directory that you can write in" + or_unset_it()
        print(f"integer_model_vllm.example: {words}. Nothing was made.", file=sys.stderr)
        sys.exit(2)
    top = top.resolve()
    try:
        if given.example == "small_gguf":
            command = make_gguf(top, repository, build)
        elif given.example == "small_float_pair":
            from .float_pair import make_float_pair  # needs PyTorch and safetensors

            command = make_float_pair(top, repository)
        else:
            command = make_synthetic(given.example, top, repository)
    except subprocess.CalledProcessError as error:
        sys.exit(
            f"{printable(Path(error.cmd[0]).name)} failed. The example is incomplete. What was "
            f"written is in {printable(str(top))}."
        )
    except FileExistsError as error:
        # float_export.py raises it with a message alone, so that filename is None there.
        name = error.filename or str(error).removesuffix(" exists already.")
        sys.exit(
            f"{printable(str(name))} exists already; nothing is overwritten. The example is "
            f"incomplete. What was written is in {printable(str(top))}."
        )
    except OSError as error:
        sys.exit(
            f"{printable(str(error.filename))}: {error.strerror}. The example is incomplete. What "
            f"was written is in {printable(str(top))}."
        )
    print(top, flush=True)
    if not given.quiet:
        print("To serve it:\n  " + shlex.join(command), file=sys.stderr)


if __name__ == "__main__":
    main()
