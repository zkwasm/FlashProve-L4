#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Tests that prover/tools/pack_layout.py refuses what can't give a layout, in plain words.

Each case calls the tool on a small list of arrays, or on a file that is no such list. The tool must
end with status 2, print nothing on standard output, and say on standard error what is wrong: one
line that begins "pack_layout.py: " and holds the case's words. Seven calls that must pass stand
beside them, so that a tool that refuses everything fails here too.

A fault here could take all the machine's memory, or never end. So every call runs under a limit of
500 MB of address space, 20 seconds of CPU time and 10 MB a file, and is stopped after 30 seconds,
and such an end counts as "not as expected".

usage: pack_layout_refusals_test.py <new work folder>
Exit status 0: all as expected. 1: not all. 2: a wrong call. The script makes <new work folder> and
missing parents, writes only there and deletes nothing.
"""

import os
import resource
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "..", "tools", "pack_layout.py")
MEMORY, CPU_SECONDS, WALL_SECONDS, FILE_BYTES = 500_000_000, 20, 30, 10_000_000
UTF8 = dict(os.environ, PYTHONUTF8="1")  # so that the cases read alike under any locale
SUPERSCRIPT_TWO = "\u00b2".encode()  # a digit to Python's isdigit(), and none to int()


def capped():
    """Lowers the soft limits alone: a hard limit of the caller's may lie below them."""
    for kind, most in ((resource.RLIMIT_AS, MEMORY), (resource.RLIMIT_CPU, CPU_SECONDS), (resource.RLIMIT_FSIZE, FILE_BYTES)):
        soft, hard = resource.getrlimit(kind)
        resource.setrlimit(kind, (most if hard == resource.RLIM_INFINITY else min(most, hard), hard))


def tool(work, *words, stdin=None):
    """Returns (status, stdout, stderr); status is None if the tool had to be stopped. Its output goes to files, under a
    limit on their size, so that a tool that prints without end fills neither this test's memory nor the disk."""
    paths = os.path.join(work, "stdout"), os.path.join(work, "stderr")
    with open(paths[0], "wb") as out, open(paths[1], "wb") as err:
        try:
            status = subprocess.run([sys.executable, "-B", PACK, *words], input=stdin, stdout=out, stderr=err, timeout=WALL_SECONDS, preexec_fn=capped, check=False, env=UTF8).returncode
        except subprocess.TimeoutExpired:
            status = None
    with open(paths[0], "rb") as out, open(paths[1], "rb") as err:
        return status, out.read(), err.read()


def array(index, rows, columns, group="1", link="1", moment="1", life="0", kind="N"):
    return "array: %s %s %s %s %s %s %s %s r a%s\n" % (link, moment, group, kind, life, index, rows, columns, index)


SMALL = array(0, 3, 5)
FIFTEEN = "".join(array(i, 1, 1) for i in range(15))
MOST_ROWS, PLACES = (1 << 32) - 1, 252  # at L 8 and free 4


def cases(work):
    """The cases: name, the list or None, the tool's words with LIST for the list's path, words that the message holds."""
    folder = os.path.join(work, "a-folder")
    os.mkdir(folder)
    no_text = os.path.join(work, "no-text")
    with open(no_text, "wb") as f:
        f.write(b"array: \xff\xfe\n")
    full = "leave no room in a row"
    nan = "is not a whole number"
    sides = "at least 1 row and 1 column"
    return [
        ("free equal to the row", SMALL, ["--arrays", "LIST", "2", "4"], full),
        ("free above the row, few elements", FIFTEEN, ["--arrays", "LIST", "4", "32"], full),
        ("free above the row, many elements", array(0, 8, 8), ["--arrays", "LIST", "1", "4"], full),
        ("a row of one entry", SMALL, ["--arrays", "LIST", "0", "1"], full),
        ("a row too long", SMALL, ["--arrays", "LIST", "32", "4"], "is too long"),
        ("no row in an array", array(0, 0, 5), ["--arrays", "LIST", "8", "4"], sides),
        ("no column in an array", array(0, 3, 0), ["--arrays", "LIST", "8", "4"], sides),
        ("too many rows in an array", array(0, 1 << 48, 1), ["--arrays", "LIST", "31", "4"], "fewer than 2^48"),
        ("too many columns in an array", array(0, 1, 1 << 48), ["--arrays", "LIST", "31", "4"], "fewer than 2^48"),
        ("a negative row count", array(0, -1000, 1), ["--arrays", "LIST", "8", "4"], "rows '-1000' " + nan),
        ("a negative column count", array(0, 1, -1000), ["--arrays", "LIST", "8", "4"], "columns '-1000' " + nan),
        ("rows that are no number", array(0, "x", 1), ["--arrays", "LIST", "8", "4"], "rows 'x' " + nan),
        ("a digit that is not ASCII in the list", array(0, "\u00b2", 1), ["--arrays", "LIST", "8", "4"], nan),
        ("a number of 20 digits", array(0, "1" * 20, 1), ["--arrays", "LIST", "8", "4"], nan),
        ("a group that is no number", array(0, 1, 1, group="x"), ["--arrays", "LIST", "8", "4"], "group 'x' " + nan),
        ("a link that is no number", array(0, 1, 1, link="x"), ["--arrays", "LIST", "8", "4"], "link 'x' " + nan),
        ("a moment that is no number", array(0, 1, 1, moment="x"), ["--arrays", "LIST", "8", "4"], "moment 'x' " + nan),
        ("a life that is no number", array(0, 1, 1, life="x"), ["--arrays", "LIST", "8", "4"], "life 'x' " + nan),
        ("an index that is no number", array("x", 1, 1), ["--arrays", "LIST", "8", "4"], "index 'x' " + nan),
        ("a line cut off", SMALL + "array: 1 1 1 N\n", ["--arrays", "LIST", "8", "4"], "has 5 fields"),
        ("a line with a field too many", SMALL.rstrip() + " more\n", ["--arrays", "LIST", "8", "4"], "has 12 fields"),
        ("a group above the limit", array(0, MOST_ROWS + 1, PLACES), ["--arrays", "LIST", "8", "4"], "fewer than 2^32 rows"),
        ("a group above the limit, not the last", array(0, MOST_ROWS + 1, PLACES) + array(1, 1, 1, group="2"), ["--arrays", "LIST", "8", "4"], "fewer than 2^32 rows"),
        ("a group above the limit, not the first", SMALL + array(1, MOST_ROWS + 1, PLACES, group="2"), ["--arrays", "LIST", "8", "4"], "fewer than 2^32 rows"),
        ("arrays of one group in two classes", SMALL + array(1, 1, 1, kind="W"), ["--arrays", "LIST", "8", "4"], "differ in class or life"),
        ("a file that does not exist", None, ["--arrays", os.path.join(work, "none"), "8", "4"], "can't be read"),
        ("a folder as the file", None, ["--arrays", folder, "8", "4"], "can't be read"),
        ("bytes that are no text", None, ["--arrays", no_text, "8", "4"], "is not text"),
        ("no line of arrays", "nothing here\n", ["--arrays", "LIST", "8", "4"], "no line 'array:'"),
        ("a source that sends no line feed", "x" * (1 << 21), ["--arrays", "LIST", "8", "4"], "characters or more"),
        ("a digit that is not ASCII in the length", SMALL, ["--arrays", "LIST", SUPERSCRIPT_TWO, "4"], "usage:"),
        ("a length with too many digits", SMALL, ["--arrays", "LIST", "9" * 5000, "4"], "usage:"),
        ("free that is no number", SMALL, ["--arrays", "LIST", "8", "x"], "usage:"),
        ("a digit that is not ASCII in free", SMALL, ["--arrays", "LIST", "8", SUPERSCRIPT_TWO], "usage:"),
        ("free that is no power of two", SMALL, ["--arrays", "LIST", "8", "3"], "usage:"),
        ("free 0", SMALL, ["--arrays", "LIST", "8", "0"], "usage:"),
        ("a word too few", SMALL, ["--arrays", "LIST", "8"], "usage:"),
        ("no --arrays", SMALL, ["LIST", "8", "4"], "usage:"),
    ]


def play(work):
    """Returns the number of cases and of those not as expected."""
    bad = 0
    all_cases = cases(work)
    for number, (name, text, words, said) in enumerate(all_cases):
        path = os.path.join(work, "list.%d" % number)
        if text is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        status, out, err = tool(work, *[path if w == "LIST" else w for w in words])
        lines = err.decode("utf-8", "replace").splitlines()
        if status != 2 or out or len(lines) != 1 or not lines[0].startswith("pack_layout.py: ") or said not in lines[0]:
            bad += 1
            print("NOT AS EXPECTED: %s: status %s, %d bytes on standard output, standard error: %s" % (name, status, len(out), ascii((lines or ["nothing"])[-1][:200])))
    def listed(name, text):
        path = os.path.join(work, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    passes = [
        ("a good list", tool(work, "--arrays", listed("good", SMALL), "8", "4"), b"1 rows over F"),
        ("a good list on standard input", tool(work, "--arrays", "-", "8", "4", stdin=SMALL.encode()), b"1 rows over F"),
        ("a byte-order mark before the first line", tool(work, "--arrays", listed("mark", "\ufeff" + SMALL + array(1, 1, 1)), "8", "4"), b"5 blocks"),
        ("free kept free: a row just full", tool(work, "--arrays", listed("full", array(0, 1, PLACES)), "8", "4"), b"1 rows over F"),
        ("free kept free: one entry more", tool(work, "--arrays", listed("more", array(0, 1, PLACES + 1)), "8", "4"), b"2 rows over F"),
        ("a group at the limit", tool(work, "--arrays", listed("most", array(0, MOST_ROWS, PLACES)), "8", "4"), b"%d rows over F" % MOST_ROWS),
        ("the longest row", tool(work, "--arrays", listed("long", SMALL), "31", "4"), b"1 rows over F"),
    ]
    for name, (status, out, err), said in passes:
        if status != 0 or not out.startswith(b"# a group is an oracle message") or said not in err:
            bad += 1
            print("NOT AS EXPECTED: %s: status %s, standard error: %s" % (name, status, ascii(err[-200:])))
    return len(all_cases) + len(passes), bad


def main():
    a = sys.argv[1:]
    if len(a) != 1 or os.path.lexists(os.path.abspath(a[0])):
        print(__doc__[__doc__.index("usage: ") : __doc__.index("Exit status")].rstrip(), file=sys.stderr)
        return 2
    work = os.path.abspath(a[0])
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    count, bad = play(work)
    print("pack_layout_refusals_test.py: %d cases: %s" % (count, "all as expected" if not bad else "%d not as expected" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
