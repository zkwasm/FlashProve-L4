# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Runs README.md's section "A request from the model's own reply": its two shell blocks, as
printed, on the CPU.

The blocks are cut out of the page by the section's heading and run in a fresh bash, from the
directory that the page names, in a small copy of the checkout below pytest's temporary directory:
so whatever they leave behind shows. What the tests compare with comes from the page itself: the
prompt, the number of reply tokens, the last id, the sum n + m and the request file's SHA-256.
"""

import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

import integer_model

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
HEADING = "\n## A request from the model's own reply\n"
UNSET = "W: first run the block above, or set W to the path that it printed"


def paths_below(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


def found(what, pattern, text) -> re.Match:
    match = re.search(pattern, text, re.M)
    assert match, f"{what} is not there as this test expects it (searched for: {pattern})"
    return match


class Shell:
    """A fresh bash for each call, in the package directory of a small copy of the checkout."""

    def __init__(self, top: pathlib.Path, package: pathlib.Path, environment: dict[str, str]):
        self.top, self.package, self.environment = top, package, environment

    def __call__(self, script: str, directory: pathlib.Path | None = None, **variables: str):
        # The script comes from standard input, as if typed. (With -c, bash 5.2 ends "${W:?...}"
        # with status 127, not 1.)
        return subprocess.run(
            ["bash", "--norc", "--noprofile", "-s"],
            input=script,
            cwd=directory or self.package,
            env={**self.environment, **variables},
            capture_output=True,
            text=True,
        )


@pytest.fixture(scope="module")
def page():
    """The section's text, its two blocks, and what its prose and its command line state."""
    text = (PACKAGE / "README.md").read_text()
    assert text.count(HEADING) == 1, "README.md must have the section that this file runs, once"
    section = text.split(HEADING)[1].split("\n## ")[0]
    blocks = re.findall(r"^```sh\n(.*?)^```$", section, re.S | re.M)
    assert len(blocks) == 2, f"the section should have two sh blocks; it has {len(blocks)}"
    # No here-document: a shell may write one to a temporary file in TMPDIR, which another account
    # could replace (python3 -c takes the program text as an argument instead).
    assert not re.search(r"<<-?\s*['\"]?\w", "".join(blocks)), "a block has a here-document"
    # Nothing is run unless the first block begins by making a new folder of its own, which mktemp
    # puts below TMPDIR. Only this beginning is checked: the lines after it are run as they stand.
    beginning = 'W=$(mktemp -d) && echo "W=$W" && E=../../../'
    assert blocks[0].startswith(beginning), (
        f"the first block should begin with {beginning!r}; it begins with "
        f"{blocks[0].splitlines()[0]!r}. Nothing was run"
    )
    prose = " ".join(re.sub(r"^```sh\n.*?^```$", "", section, flags=re.S | re.M).split())
    # The call's arguments follow the program text of python3 -c, on the block's last line.
    call = r'^  "\$W" ([\d,]+) (\d+) "\$W/reply\.json"$'
    arguments = found("README.md: the first block's call of python3", call, blocks[0])
    return SimpleNamespace(
        blocks=blocks,
        configuration=found(
            "README.md: the first block's E=", r"E=\.\./\.\./\.\./(\S+) &&", blocks[0]
        )[1],
        prompt=[int(word) for word in arguments[1].split(",")],
        number=int(arguments[2]),
        stated_number=int(
            found("README.md: the number of ids", r"print the (\d+) ids of the reply", prose)[1]
        ),
        last_ids=re.findall(r"the last of which is (\d+)|so y is (\d+)", prose),
        total=[
            int(w)
            for w in found("README.md: n + m", r"here (\d+) \+ (\d+) = (\d+),", prose).groups()
        ],
        sha256=found("README.md: the sha256", r"whose sha256 is ([0-9a-f]{64})\.", prose)[1],
    )


@pytest.fixture(scope="module")
def shell(page, tmp_path_factory):
    """HOME and TMPDIR are below pytest's temporary directory, and python3 is this Python."""
    if not (REPOSITORY / page.configuration / "policy.txt").is_file():
        pytest.skip(f"the package stands alone: there is no {page.configuration} above it")
    top = tmp_path_factory.mktemp("readme_request")
    package = top / "checkout" / "integrations" / "vllm" / "integer_model"
    without_bytecode = shutil.ignore_patterns("__pycache__")
    shutil.copytree(PACKAGE / "integer_model", package / "integer_model", ignore=without_bytecode)
    copy = top / "checkout" / page.configuration  # copyfile: writable, whatever the files' modes
    shutil.copytree(REPOSITORY / page.configuration, copy, copy_function=shutil.copyfile)
    for name in ("home", "tmp", "bin"):
        (top / name).mkdir()
    (top / "bin" / "python3").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    (top / "bin" / "python3").chmod(0o755)
    environment = {
        "PATH": f"{top / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(top / "home"),
        "TMPDIR": str(top / "tmp"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }
    try:
        subprocess.run([top / "bin" / "python3", "-c", ""], check=True)
    except PermissionError:
        pytest.skip(
            f"{top} does not let programs run (mounted noexec?). First make a folder on a file "
            "system that does: dir=$(mktemp -d -p DIR), where DIR is a folder of yours there. If "
            "that succeeds, run pytest again with your other arguments and "
            '--basetemp="${dir:?}/t" (pytest empties the folder that --basetemp names).'
        )
    return Shell(top, package, environment)


@pytest.fixture(scope="module")
def outcome(page, shell):
    """Both blocks, one after the other, in one shell, as the page says."""
    checkout = shell.top / "checkout"
    before = paths_below(checkout)
    done = shell(page.blocks[0] + page.blocks[1])
    assert done.returncode == 0, done.stderr[-2000:]
    printed = done.stdout.splitlines()
    assert len(printed) == 2 and printed[0].startswith("W="), (
        "the two blocks should print two lines, W=<the new folder> and the reply's ids; they "
        f"printed {len(printed)}: {done.stdout[:500]!r}"
    )
    said, ids = printed
    return SimpleNamespace(
        folder=pathlib.Path(said.removeprefix("W=")),
        reply=[int(word) for word in ids.split(",")],
        new_in_the_checkout=sorted(set(paths_below(checkout)) - set(before)),
    )


def test_the_folder_is_new_and_holds_five_files(outcome, shell):
    assert outcome.folder.parent == shell.top / "tmp"
    assert paths_below(shell.top / "tmp") == [outcome.folder.name] + [
        f"{outcome.folder.name}/{name}"
        for name in ("KL.bin", "integer_model.json", "policy.txt", "reply.json", "reply.req")
    ]
    assert paths_below(shell.top / "home") == []


def test_nothing_is_left_in_the_checkout(outcome):
    """No bytecode either: the copy had none, and no variable switches its writing off."""
    assert outcome.new_in_the_checkout == []


def test_the_reply_is_the_one_that_the_page_tells(page, outcome):
    assert len(outcome.reply) == page.number == page.stated_number
    assert len(page.last_ids) == 2  # the last id, and y
    assert {int(first or second) for first, second in page.last_ids} == {outcome.reply[-1]}
    assert json.loads((outcome.folder / "reply.json").read_text()) == {
        "prompt_token_ids": page.prompt,
        "reply_token_ids": outcome.reply,
    }


def test_the_request_file_is_the_one_that_the_page_tells(page, outcome):
    data = (outcome.folder / "reply.req").read_bytes()
    assert hashlib.sha256(data).hexdigest() == page.sha256
    lt = int(
        found("policy.txt's line lt", r"^lt (\d+)$", (outcome.folder / "policy.txt").read_text())[1]
    )
    T, n, m = 1 << lt, len(page.prompt), len(outcome.reply)
    assert page.total == [n, m, n + m] and n + m == T + 1
    assert len(data) == 4 * (T + 2)
    *window, r, e = struct.unpack(f"<{T + 2}I", data)
    assert window == (page.prompt + outcome.reply)[:T]
    assert (r, e) == (n - 1, n + m - 1)


def test_every_covered_token_is_the_first_choice_in_one_pass_over_the_window(page, outcome):
    """The blocks decode token by token. A proof is about one pass over the whole window."""
    model = integer_model.load_from_folder(outcome.folder, "cpu")
    sequence = model.new_sequence()
    choices = model.extend(sequence, (page.prompt + outcome.reply)[: model.parameters.T])
    model.free(sequence)
    assert choices[len(page.prompt) - 1 :].tolist() == outcome.reply


def test_the_second_block_alone_stops_with_its_sentence(page, outcome, shell):
    before = paths_below(shell.top)
    for variables in ({}, {"W": ""}):
        done = shell(page.blocks[1], **variables)
        assert (done.returncode, done.stdout) == (1, "")
        assert done.stderr.strip().endswith(UNSET)
    assert paths_below(shell.top) == before


def test_the_second_block_writes_only_a_new_file(page, outcome, shell):
    before = (outcome.folder / "reply.req").read_bytes()
    done = shell(page.blocks[1], W=str(outcome.folder))
    assert done.returncode != 0
    assert "FileExistsError" in done.stderr
    assert (outcome.folder / "reply.req").read_bytes() == before


@pytest.mark.parametrize("kept, written", [(10, True), (1, True), (0, False), (50, False)])
def test_the_second_block_on_other_lengths(kept, written, page, outcome, shell, tmp_path):
    """A shorter reply is padded with token 0. No reply at all, and a prompt and a reply of more
    than T + 1 tokens together, are refused, and no file is written.
    """
    reply = (outcome.reply + [1])[:kept]
    shutil.copy(outcome.folder / "policy.txt", tmp_path)
    one = {"prompt_token_ids": page.prompt, "reply_token_ids": reply}
    (tmp_path / "reply.json").write_text(json.dumps(one))
    done = shell(page.blocks[1], W=str(tmp_path))
    assert (tmp_path / "reply.req").exists() == written == (done.returncode == 0)
    if written:
        n, T = len(page.prompt), len(page.prompt) + len(outcome.reply) - 1
        *window, r, e = struct.unpack(f"<{T + 2}I", (tmp_path / "reply.req").read_bytes())
        assert window == page.prompt + reply + [0] * (T - n - kept)
        assert (r, e) == (n - 1, n + kept - 1)
    else:
        assert "together at most T + 1" in done.stderr


def test_the_second_block_takes_no_module_from_the_callers_directory(
    page, outcome, shell, tmp_path
):
    """python3 -I: files named like the modules that the function imports lie where it is called."""
    (tmp_path / "here").mkdir()
    for name in ("json.py", "struct.py"):
        (tmp_path / "here" / name).write_text("raise SystemExit('taken from the directory')\n")
    (tmp_path / "W").mkdir()
    for name in ("policy.txt", "reply.json"):
        shutil.copy(outcome.folder / name, tmp_path / "W")
    done = shell(page.blocks[1], directory=tmp_path / "here", W=str(tmp_path / "W"))
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert (tmp_path / "W" / "reply.req").read_bytes() == (
        outcome.folder / "reply.req"
    ).read_bytes()
    assert paths_below(tmp_path / "here") == ["json.py", "struct.py"]


def test_the_first_block_writes_only_a_new_file(page, outcome, shell):
    """As printed, the block always works in a new folder. Here a stand-in for mktemp names the
    folder that is there already.
    """
    before = (outcome.folder / "reply.json").read_bytes()
    done = shell('mktemp() { echo "$OLD"; }\n' + page.blocks[0], OLD=str(outcome.folder))
    assert done.returncode != 0
    assert "FileExistsError" in done.stderr
    assert (outcome.folder / "reply.json").read_bytes() == before
