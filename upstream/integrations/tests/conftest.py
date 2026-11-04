# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Let this directory's tests stand aside where the scripts under test would refuse to work.

The scripts refuse a directory below a place that another account can change. If pytest's temporary
directory is below such a place, every test of this directory is skipped, also the few that reach no
such script. If the run has no other tests and no error, it ends with status 77 (skipped) and a last
line that says why. Other directories' tests run as usual, and then the status is theirs. If a
directory on the way to that place is missing or cannot be examined, no test is run (status 4).

Each test also starts under a umask of 022, so that its own plain mkdir() makes no such place.

The marker root_only is registered here. A test that only root can run has it beside its skipif, so
that -m root_only chooses just these tests and -m "not root_only" leaves them out.
"""

import os
import tempfile

import pytest

import spool

HERE = os.path.dirname(os.path.abspath(__file__))
STOOD_ASIDE = []  # the last line, if all the run's tests are this directory's and were skipped


def pytest_configure(config):
    config.addinivalue_line("markers", "root_only: only root can run this test; others skip it")


@pytest.fixture(autouse=True)
def with_a_umask_of_022():
    """Under a umask of 002, many accounts' default, a test's plain mkdir() makes a directory that
    its group can write, and the scripts rightly refuse to work below it."""
    before = os.umask(0o022)
    yield
    os.umask(before)


@pytest.hookimpl(trylast=True)  # behind -k and -m, so that only the chosen tests are counted
def pytest_collection_modifyitems(config, items):
    base = config.option.basetemp
    # As pytest chooses the place for its temporary directories.
    root = os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir()
    ground = os.path.dirname(os.path.abspath(base)) if base else root
    found = spool.others_can_change(ground)
    if not found:
        return
    # The advice names what this run took the place from. It never brings --basetemp up by itself:
    # pytest removes the directory that this option names, with all that is in it.
    if base:
        cure = "give --basetemp the name of a new directory (pytest removes an existing one)"
    elif os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        cure = "set PYTEST_DEBUG_TEMPROOT to an existing directory that only you can write,"
    else:
        cure = "set TMPDIR to an existing directory that only you can write,"
    cure += " below directories that only you or root can change"
    place = "%s (%s)" % (spool.printable(found[0]), spool.whose(*found))
    if found[1] in (spool.GONE, spool.UNSEEN):
        # No open place, and no safe one: in a sticky directory such as /tmp, another account could
        # make the missing directory before pytest looks for it.
        raise pytest.UsageError(
            "no test was run: pytest's temporary directory would be below %s: %s" % (place, cure)
        )
    why = "stood aside: below an open place: %s: %s" % (place, cure)
    ours = [item for item in items if str(item.path).startswith(HERE + os.sep)]
    for item in ours:
        item.add_marker(pytest.mark.skip(reason=why))
    if len(ours) == len(items) and not config.option.collectonly:
        STOOD_ASIDE.append(why)


def pytest_sessionfinish(session, exitstatus):
    if STOOD_ASIDE and exitstatus == 0:
        session.exitstatus = 77
    else:  # an error in collection, an interrupt, no test chosen: that status stays
        STOOD_ASIDE.clear()


def pytest_unconfigure():
    if STOOD_ASIDE:  # printed after pytest's own last line
        print(STOOD_ASIDE[0])
