# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Licence copies match the repository's. Each is listed in integrations/pyproject.toml."""

import pathlib
import re

import pytest

INTEGRATIONS = pathlib.Path(__file__).resolve().parent.parent
REPOSITORY = INTEGRATIONS.parent
IN_REPOSITORY = (REPOSITORY / "THIRD_PARTY.md").is_file()


def patterns():
    # Read with regular expressions because Python 3.10 has no tomllib.
    text = (INTEGRATIONS / "pyproject.toml").read_text()
    (value,) = re.findall(r"^license-files = \[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    return re.findall(r'"([^"]+)"', value)


def listed_files():
    found = [path for pattern in patterns() for path in sorted(INTEGRATIONS.glob(pattern))]
    return [path.relative_to(INTEGRATIONS).as_posix() for path in found]


# True if integrations/ has no licence copy at all: then only test_license_and_notice_are_listed()
# runs. If only some copies are missing, test_pattern_matches_files_only() fails.
NONE_YET = not listed_files() and not (INTEGRATIONS / "licenses").exists()
unless_none_yet = pytest.mark.skipif(
    NONE_YET,
    reason="none of the licence files that pyproject.toml lists is here: "
    "copy them from the repository's top directory before you build a wheel",
)


def test_license_and_notice_are_listed():
    assert {"LICENSE", "NOTICE"} <= set(patterns())


# A file must be listed by its exact path: a glob pattern that matches it does not count.
@unless_none_yet
def test_every_file_in_licenses_is_listed():
    files = [path for path in (INTEGRATIONS / "licenses").iterdir() if path.is_file()]
    found = {"licenses/" + path.name for path in files if not path.name.startswith(".")}
    unlisted = found - set(patterns())
    assert not unlisted, "not in license-files: %s" % ", ".join(sorted(unlisted))


@unless_none_yet
@pytest.mark.parametrize("pattern", patterns())
def test_pattern_matches_files_only(pattern):
    found = list(INTEGRATIONS.glob(pattern))
    assert found and all(path.is_file() for path in found)


@pytest.mark.skipif(
    not IN_REPOSITORY,
    reason="integrations/ is not inside the repository: nothing to compare the copies with",
)
@pytest.mark.parametrize("name", listed_files())
def test_listed_file_is_a_copy_of_the_repositorys_file(name):
    original = REPOSITORY / name
    assert original.is_file(), "the repository has no %s" % name
    same = (INTEGRATIONS / name).read_bytes() == original.read_bytes()
    assert same, "the copy differs from the repository's %s: copy that file over it" % name
