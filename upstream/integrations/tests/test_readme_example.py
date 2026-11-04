# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The commands that integrations/README.md prints for two shipped examples fit their files.

The page prints the commitment ID and the next token as literal values, so that no reader learns to
take them from files that come with a proof. This test keeps the page and the files together.
"""

import pathlib
import re
import struct

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPOSITORY / "integrations" / "README.md"
SMALL = "accepted-proof"
# examples/full-size/README.md has this sentence if the release ships that example's three large
# files as a separate archive and not in the repository.
NOT_IN_REPOSITORY = "Three files of this example are not in the repository"
LARGE = ("example.proof", "ivk/key.bin", "ivk/leaf/0.bin")


def directory(name):
    return REPOSITORY / "examples" / name


def here(name):
    """Return the example's name as a test parameter that is skipped where its files are missing."""
    missing = not (directory(name) / "ID.txt").is_file()
    skip = pytest.mark.skipif(missing, reason="examples/%s/ is not here" % name)
    return pytest.param(name, marks=skip)


each_example = pytest.mark.parametrize("name", [here(SMALL), here("full-size")])


def printed_options(name):
    """Return the options of the page's one command on the example, as a dictionary."""
    blocks = re.findall(r"```sh\n(.*?)```", PAGE.read_text(), re.S)
    (block,) = [text for text in blocks if "examples/%s/example.proof" % name in text]
    words = block.replace("\\\n", " ").split()
    return {word: words[place + 1] for place, word in enumerate(words) if word.startswith("--")}


@each_example
def test_values_are_literal(name):
    options = printed_options(name)
    assert re.fullmatch("[0-9a-f]{64}", options["--commitment-id"])
    assert re.fullmatch("[0-9]+", options["--next-token"])


def test_every_value_on_the_page_is_literal():
    for name in (SMALL, "full-size"):
        if not (directory(name) / "ID.txt").is_file():
            pytest.skip("examples/%s/ is not here" % name)
    page = PAGE.read_text()
    for value in re.findall(r"--commitment-id (\S+)", page):
        assert re.fullmatch("[0-9a-f]{64}", value), value
    for value in re.findall(r"--next-token (\S+)", page):
        assert re.fullmatch("[0-9]+", value), value


@each_example
def test_commitment_id_is_the_examples(name):
    stored = (directory(name) / "ID.txt").read_text().strip()
    assert printed_options(name)["--commitment-id"] == stored


@each_example
def test_next_token_is_the_examples(name):
    (token,) = struct.unpack("<I", (directory(name) / "example.y").read_bytes())
    assert printed_options(name)["--next-token"] == str(token)


@each_example
def test_range_is_the_requests(name):
    start, end = struct.unpack("<II", (directory(name) / "example.req").read_bytes()[-8:])
    assert printed_options(name)["--range"] == "%d:%d" % (start, end)


@each_example
def test_named_files_exist(name):
    folder = directory(name)
    if name == "full-size" and not any((folder / path).exists() for path in LARGE):
        if NOT_IN_REPOSITORY in (folder / "README.md").read_text():
            pytest.skip("full-size: not run: the set is not in this tree")
    options = printed_options(name)
    named = [options[word] for word in ("--public-key-dir", "--parameter-file", "--scales-file")]
    named += ["examples/%s/example.proof" % name, "examples/%s/example.req" % name]
    for path in named:
        assert (REPOSITORY / path).exists(), path


@pytest.mark.parametrize("name", [here(SMALL)])
def test_prose_has_the_commands_range(name):
    request = (directory(name) / "example.req").read_bytes()
    start, end = struct.unpack("<II", request[-8:])
    page = PAGE.read_text()
    assert "`%d:%d` on %d tokens" % (start, end, len(request) // 4 - 2) in page
    assert "`--reply %d:%d` is `--range %d:%d`" % (start + 1, end - start, start, end) in page
    assert "range=%d:%d " % (start, end) in page


@pytest.mark.parametrize("name", [here("full-size")])
def test_prose_has_the_full_size_range(name):
    request = (directory(name) / "example.req").read_bytes()
    start, end = struct.unpack("<II", request[-8:])
    length = len(request) // 4 - 2
    words = "The request has {:,} tokens, so `{}:{}` checks ".format(length, start, end)
    if end - start == 1:
        words += "one token"
    else:
        words += "tokens {} to {} and the next token: {:,} tokens in one proof".format(
            start + 1, end - 1, end - start
        )
    assert start < end == length and words in PAGE.read_text()
