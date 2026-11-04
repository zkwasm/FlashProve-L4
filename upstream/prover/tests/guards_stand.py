#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Checks, by text only, that listed guard lines of cp_commit and cp_prove are still in the sources.

Needs no GPU and no build, and writes nothing in the tree; it takes about twenty seconds. Prints
ALL AS EXPECTED and exits 0, or prints what differs and exits 1.

A guard is a line of the committer program cp_commit or the prover program cp_prove, such as a
self-check of the program, a zeroing or a fill before a use, or a wait for a side stream (an extra
CUDA stream that works for a request). Without it an honest run writes the same files as before, so
no run of the honest programs shows that it is missing. Two checks:

  pins     Each text in prover/tests/guards_stand.tsv (a "pin") occurs exactly once in its file,
           outside comments and under no preprocessor condition other than those in FENCES.
           Otherwise the test fails and prints, from the table, what the text guards. If a guard
           has only been reworded, update the table. For this check comments are blanked in the
           file and in the pin; the controls replace the pin's exact text, so a comment inside a
           pin must match the file as well.
  streams  The calls under prover/ that create a CUDA stream (outside tests/, examples/, docs/ and
           the build directories) are exactly those in STREAMS, which records whether the next line
           registers the stream with side_streams::enter() ("entered" in the output). save() and
           end() in prover/prover/member_state.cuh make their thread's stream wait for the entered
           streams before they save a request's failure bits (a request with a bit set gets no
           proof). prover/common/side_streams.h explains why the others are not entered. When
           adding a stream, decide whether to enter it.

Controls, in every run. The files that the table and STREAMS name are copied to a temporary
directory; no copy is compiled. There one pinned text at a time is weakened in each way listed in
WEAKENINGS and the copy is checked in the same way: that pin must be reported, and nothing else.
Likewise, in prover/attn/front.cuh only, for an added call that creates a stream and for a removed
or commented-out side_streams::enter() line. In the output's last line but one, each control on a
pin is a "tooth", which "bites" if it passes.

Not shown: that a guard works, is reached or has its result acted on, or that it is at the point of
a run that the table describes. This test runs no program. For the rows on prover/engine/tree.cuh
the table's fourth cell names a GPU test, prover/engine/tests/tree_arrays_cleared_test.cu, which
shows in part that those guards work; for the other rows no test in this tree shows these things.
A pin still passes if its text follows an if (0) or a return, is wrapped in a macro or a
string literal, or is in a file that no build compiles. The table does not list every guard, and
nothing here checks its number of rows. Which streams carry a request's work was decided by reading
the sources. A stream created outside prover/ is not found: the programs also compile files of
blocks/ and verifier/src/, which are not read.
"""

import os
import re
import shutil
import sys
import tempfile

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guards_stand.tsv")
# (file, from prover/; the variable that receives the stream): does the next line enter the stream?
STREAMS = {("attn/front.cuh", "sxs"): True, ("hiding/product_test.cuh", "cs[q]"): True, ("hiding/product_test.cuh", "made"): True,
           ("prover/weight_stream.cuh", "feed"): False, ("prover/fork_memory.h", "zeroing_stream"): False,
           ("hash_commitment/hf_commit_windows.cuh", "copies"): False, ("hash_commitment/hf_range_windows.cuh", "copies_"): False,
           ("hash_commitment/hf_commit_weights.cuh", "g.copies"): False}  # fmt: skip
IN_WORDS = {None: "not made", True: "made and entered", False: "made and not entered"}
FENCES = {"#ifdef __CUDACC__"}  # the conditions under which a pinned text may occur: true in every build of the GPU programs
# How the controls weaken a pin: (words for the message, pinned text -> what replaces it).
# prover/build.sh compiles the production builds of cp_commit and cp_prove with -DSERVING_BUILD=1,
# so a text under #ifndef SERVING_BUILD is missing from them.
WEAKENINGS = [
    ("removed", lambda text: ""),
    ("turned into line comments", lambda text: "".join("// " + line for line in text.splitlines(True))),
    ("placed after a // comment that ends in a backslash", lambda text: "// \\\n" + text),
    ("placed in a block comment", lambda text: "/* " + text.replace("*/", "* /") + " */"),
    ("disabled by #if 0", lambda text: "\n#if 0\n" + text + "\n#endif\n"),
    ("placed under #ifndef SERVING_BUILD", lambda text: "\n#ifndef SERVING_BUILD\n" + text + "\n#endif\n"),
    ("placed in the #else of #ifdef __CUDACC__", lambda text: "\n#ifdef __CUDACC__\n#else\n" + text + "\n#endif\n"),
]
# What is done | in which file | how | the stream that must be reported.
STREAM_CONTROLS = [
    ("a call that makes a stream is planted", "attn/front.cuh", lambda whole: whole + "CK(cudaStreamCreate(&planted));\n", "planted"),
    ("the entry of an entered stream is struck", "attn/front.cuh", lambda whole: whole.replace("CK(side_streams::enter(sxs));", ""), "sxs"),
    ("the entry of an entered stream is made a comment", "attn/front.cuh", lambda whole: whole.replace("CK(side_streams::enter(sxs));", "// CK(side_streams::enter(sxs));"), "sxs"),
]


def read(path):
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(text)


def blank(text):
    """Returns text with each C or C++ comment replaced by spaces; the comment's newlines stay.

    Comment marks inside string, raw-string and character literals do not count as comments.
    """
    def one(m):
        s = m.group(0)
        return s if s[0] in "\"'R" else re.sub(r"[^\n]", " ", s)
    return re.sub(r'R"([^\s()\\]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//(?:\\\n|[^\n])*|/\*.*?\*/', one, text, flags=re.S)


def fences(text):
    """Returns the preprocessor conditions still open at the end of text, outermost first.

    Comments in text must already be blanked. A condition whose #else or #elif has passed has that
    line appended, so the #else part of a condition in FENCES is not itself allowed.
    """
    open_ones = []
    for line in (l.strip() for l in text.split("\n")):
        if re.match(r"#\s*if", line):
            open_ones.append(re.sub(r"\s+", " ", line))
        elif re.match(r"#\s*(else|elif)", line) and open_ones:
            open_ones[-1] += " .. " + re.sub(r"\s+", " ", line)
        elif re.match(r"#\s*endif\b", line) and open_ones:
            open_ones.pop()
    return open_ones


def pins():
    """Returns the table's rows: (file from prover/, text, what it guards, a note not used here)."""
    rows = [line.split("\t") for line in read(TABLE).split("\n") if line.strip() and not line.startswith("#")]
    for row in rows:
        if len(row) != 4 or not all(cell.strip() for cell in row):
            print("DIFFERS   guards_stand.tsv: this row does not have four filled cells: %s\n1 CHECKS DIFFER" % " | ".join(row))
            sys.exit(1)
    # In the text cell \n, \t and \\ mean a newline, a tab and a backslash; other escapes are kept.
    plain = {"n": "\n", "t": "\t", "\\": "\\"}
    return [(file, re.sub(r"\\(.)", lambda m: plain.get(m.group(1), m.group(0)), text), guard, play) for file, text, guard, play in rows]


def look(prover, table):
    """Returns what differs under prover (prover/ or the controls' copy), as ((file, x), message).

    table is the result of pins(). x is a pinned text, the variable of a stream, or None for a file
    whose calls that create a stream could not all be read.
    """
    said, found = [], {}
    for file, text, guard, _ in table:
        whole = blank(read(os.path.join(prover, file))) if os.path.isfile(os.path.join(prover, file)) else ""
        times = whole.count(blank(text))
        under = [x for x in fences(whole[:whole.index(blank(text))]) if x not in FENCES] if times == 1 else []
        if times != 1 or under:
            said.append(((file, text), "%s holds this text %s:\n%s\n          What it guards (tests/guards_stand.tsv, where a row may go on from the row above it): %s"
                         % (file, "under a condition of the preprocessor that FENCES (tests/guards_stand.py) does not allow: " + ", ".join(under) if under else "%d times outside comments, not once" % times, text.rstrip("\n"), guard)))  # fmt: skip
    for folder, folders, files in os.walk(prover):
        folders[:] = [x for x in folders if x not in ("tests", "examples", "docs") and not x.startswith("build")]
        for name in files:
            if name.endswith((".cu", ".cuh", ".h", ".inc", ".cpp")):
                file, text = os.path.relpath(os.path.join(folder, name), prover), read(os.path.join(folder, name))
                here = re.findall(r"\bcu(?:da)?StreamCreate\w*\(\s*&([^,)]+)[^\n]*\n([^\n]*)", text)
                # A call in another form (not "&name"), or a second call for one name, is reported.
                # Comments are not blanked for this check: a call inside a comment counts too.
                calls = len(re.findall(r"\bcu(?:da)?StreamCreate\w*\s*\(", text))
                if calls != len({stream for stream, _ in here}):
                    said.append(((file, None), "%s: %d calls that make a stream, of which %d different ones are read" % (file, calls, len({stream for stream, _ in here}))))
                for stream, line in here:
                    found[file, stream] = line.strip().startswith("CK(side_streams::enter(%s));" % stream)
    for k in sorted(k for k in {*found, *STREAMS} if found.get(k) != STREAMS.get(k)):
        said.append((k, "the stream %s of %s: by the list STREAMS %s, in the sources %s" % (k[1], k[0], IN_WORDS[STREAMS.get(k)], IN_WORDS[found.get(k)])))
    return said


def main():
    prover, table = os.path.dirname(os.path.dirname(os.path.abspath(__file__))), pins()
    said = look(prover, table)
    for _, sentence in said:
        print("DIFFERS   " + sentence)
    differ, bite = len(said), {"pin": 0, "stream": 0}
    controls = [("pin", "the pinned text is " + how, file, lambda whole, text=text, weaken=weaken: whole.replace(text, weaken(text)), text)
                for file, text, _, _ in table for how, weaken in WEAKENINGS]  # fmt: skip
    with tempfile.TemporaryDirectory() as copy:
        for file in sorted({row[0] for row in table} | {file for file, _ in STREAMS}):
            if os.path.isfile(os.path.join(prover, file)):
                os.makedirs(os.path.dirname(os.path.join(copy, file)), exist_ok=True)
                shutil.copyfile(os.path.join(prover, file), os.path.join(copy, file))
        for kind, what, file, change, name in controls + [("stream", *control) for control in STREAM_CONTROLS]:
            if (file, name) in [x[0] for x in said] or not os.path.isfile(os.path.join(copy, file)):
                continue  # already reported for the sources, or the file is missing: no control
            whole = read(os.path.join(copy, file))
            write(os.path.join(copy, file), change(whole))
            seen = [x[0] for x in look(copy, table) if x not in said]
            write(os.path.join(copy, file), whole)
            if seen == [(file, name)]:
                bite[kind] += 1
            else:
                differ += 1
                print("DIFFERS   control: %s: %s (%s): that alone was to be reported, and this was: %s"
                      % (file, what, name.strip().split("\n")[0], "; ".join("%s: %s" % (f, str(n).strip().split("\n")[0]) for f, n in seen) or "nothing"))  # fmt: skip
    print("%d guards pinned in %d files; %d of %d teeth bite; %d places that make a stream, and %d of %d controls on them bite"
          % (len(table), len({row[0] for row in table}), bite["pin"], len(table) * len(WEAKENINGS), len(STREAMS), bite["stream"], len(STREAM_CONTROLS)))  # fmt: skip
    print("ALL AS EXPECTED" if not differ else "%d CHECKS DIFFER" % differ)
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
