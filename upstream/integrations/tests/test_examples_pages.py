# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Check the README.md ("page" in the tests' names) of each example in FOLDERS against its files.

examples/<name>/README.md must state the same values as the files stored beside it: the commitment
ID and the request's checked range and, in some examples, the next token (example.y), the
acceptance policy's parameters, the bound above-clip that follows from them, and the one token that
README.md calls the model's second choice. One test checks SHA256SUMS instead, and one that
README.md has no unfilled placeholder (@NAME@). Each test is skipped if the example's files are
missing.
"""

import hashlib
import pathlib
import re
import struct

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
FOLDERS = ["accepted-proof", "accepted-proof-clipped-gap", "full-size", "full-size-clipped-gap"]


@pytest.fixture(params=FOLDERS)
def folder(request):
    path = EXAMPLES / request.param
    if not (path / "example.proof").is_file():
        pytest.skip("%s/example.proof is missing" % path)
    return path


def parameter_lines(folder):
    words = (line.split() for line in (folder / "config" / "policy.txt").read_text().splitlines())
    return {word[0]: word[1] for word in words if len(word) == 2}


def test_the_page_prints_the_commitment_id(folder):
    # typed after -id in a command, and shown in the sample output's line result-v1
    page = (folder / "README.md").read_text()
    typed, shown = (re.findall(p + r"([0-9a-f]{64})\b", page) for p in (r"-id ", r"\bid="))
    assert typed and shown and set(typed + shown) == {(folder / "ID.txt").read_text().strip()}


def test_the_page_has_every_value_filled_in(folder):
    assert re.findall(r"@[A-Z_-]+@", (folder / "README.md").read_text()) == []


def test_the_list_of_digests_fits_the_files(folder):
    listed = dict(
        reversed(line.split("  ", 1)) for line in (folder / "SHA256SUMS").read_text().splitlines()
    )
    stored = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()} - {
        "SHA256SUMS",
        "README.md",
    }
    assert set(listed) == stored
    for name, digest in listed.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name


# example.req contains 2^lt token IDs (`lt`: a line of policy.txt), then r and e; the covered
# positions are r to e - 1.
def test_the_page_describes_the_request(folder):
    request = (folder / "example.req").read_bytes()
    tokens = 1 << int(parameter_lines(folder)["lt"])
    assert len(request) == 4 * (tokens + 2)
    r, e = struct.unpack("<2I", request[-8:])
    assert "r = %d and e = %d" % (r, e) in (folder / "README.md").read_text()


def test_the_page_prints_the_token_after_the_request():
    folder = EXAMPLES / "accepted-proof"
    if not (folder / "example.y").is_file():
        pytest.skip("%s/example.y is missing" % folder)
    (token,) = struct.unpack("<I", (folder / "example.y").read_bytes())
    assert "the one after the request: %d |" % token in (folder / "README.md").read_text()


@pytest.mark.parametrize("name", ["full-size", "full-size-clipped-gap"])
def test_the_full_size_page_prints_the_token_after_the_request(name):
    folder = EXAMPLES / name
    if not (folder / "example.y").is_file():
        pytest.skip("%s/example.y is missing" % folder)
    (token,) = struct.unpack("<I", (folder / "example.y").read_bytes())
    assert (
        "the token after the request, %d (in `example.y`)" % token
        in (folder / "README.md").read_text()
    )


@pytest.mark.parametrize("name", ["accepted-proof-clipped-gap", "full-size-clipped-gap"])
def test_the_page_prints_the_policy(name):
    folder = EXAMPLES / name
    if not (folder / "config" / "policy.txt").is_file():
        pytest.skip("%s/config/policy.txt is missing" % folder)
    line = parameter_lines(folder)
    page = (folder / "README.md").read_text()
    assert (
        "-policy clipped-gap,"
        + ",".join(line[key] for key in ("C", "theta", "cap", "s", "sh"))
        + " "
        in page
    )
    for key in ("C", "theta", "cap"):
        assert "| `%s %s` |" % (key, line[key]) in page


# The checked tokens are tokens r + 1 to min(e, 2^lt - 1) (verifier/docs/output.md).
def checked_tokens(folder):
    r, e = struct.unpack("<2I", (folder / "example.req").read_bytes()[-8:])
    return r + 1, min(e, (1 << int(parameter_lines(folder)["lt"])) - 1)


def test_the_full_size_clipped_gap_page_names_the_token_that_is_not_a_first_choice():
    folder = EXAMPLES / "full-size-clipped-gap"
    if not (folder / "example.req").is_file():
        pytest.skip("%s/example.req is missing" % folder)
    found = re.findall(
        r"Token (\d+) is the model's second choice at its place: token ID (\d+),",
        (folder / "README.md").read_text(),
    )
    assert len(found) == 1
    place, token = map(int, found[0])
    first, last = checked_tokens(folder)
    assert first <= place <= last
    request = (folder / "example.req").read_bytes()
    assert struct.unpack_from("<I", request, 4 * place) == (token,)


# above-clip: how many checked tokens the limits let cost more than C ("The field above-clip=" in
# verifier/docs/output.md gives this formula).
@pytest.mark.parametrize("name", ["accepted-proof-clipped-gap", "full-size-clipped-gap"])
def test_the_page_prints_the_bound_that_follows_from_the_policy(name):
    folder = EXAMPLES / name
    if not (folder / "example.req").is_file():
        pytest.skip("%s/example.req is missing" % folder)
    line = {key: int(value) for key, value in parameter_lines(folder).items() if value.isdigit()}
    first, last = checked_tokens(folder)
    checked = last - first + 1
    first_choice = 1 << line["sh"]
    bound = min(line["cap"], checked)
    if line["C"] > first_choice:
        room = max(line["theta"] - first_choice, 0)
        bound = min(bound, checked * room // (line["C"] - first_choice))
    shown = re.findall(r"above-clip=(\d+)", (folder / "README.md").read_text())
    assert shown and set(shown) == {str(bound)}
