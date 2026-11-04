#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs cp_plan on files and checks its exit status and the lines that a person would quote.

usage: cp_plan_files.py <new work folder>

cp_plan (blocks/numbers/cp_plan.cpp) computes round errors and sizes from a text file.
This test builds it with blocks/Makefile's default flags and runs it on
blocks/numbers/settings/full-goldilocks.txt, on copies of that file with one or two texts replaced,
and on a text of codes and groups with no protocol (both written to the work folder).
No figure is checked but the target and the field's p: whether an error is computed correctly is
tested by blocks/numbers/tests/numbers_test.cpp and blocks/numbers/tests/holds_test.cpp.
Exit status 0: all as expected. 1: not. 2: wrong arguments (the work folder must not exist yet), a
text that does not occur exactly once, or cp_plan does not build.
"""

import os
import re
import subprocess
import sys

NUMBERS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICT = r"the plan's own verdict: "
LOC = "  loc group=seg trees=P1,P2,P3 where=segments\n"
FULL = "full-goldilocks.txt"
WORK = "work positions request=20 shared=28 once=32\nwork closeness request=20 shared=20 once=20\n"
# A setting, codes and groups: no protocol, so no round.
CODES = """field goldilocks
target 128
work positions request=0 shared=0 once=0
code long log_n=23 ell=2^22
code short log_n=19 ell=2^16
code table log_n=33 ell=402653184 r=44040191
group long main=long
group short main=short
group table main=table scope=shared
group table-eighth main=table regime=delta:1/8 scope=once
"""
# Each case: name; file of blocks/numbers/settings/, or None for the text CODES; pairs (a text that
# occurs there once, its replacement); cp_plan's exit status; patterns, each matched at a line's
# start, in output order.
CASES = (
    ("full", FULL, (), 0, (VERDICT + "holds: .*one distance$", "accounting: target: 128 bits in every round, proof of work counted; ", "accounting: THE ROUNDS: HOLD WITH WORK: ", "size: ")),
    # No `work` lines: what the accounting prints without proof of work, at a target low enough for
    # the codes' random tails (`r`; printed as "random coordinates") to meet every hiding budget.
    ("no-work", FULL, ((WORK, ""), ("target 128\n", "target 96\n")), 0, (VERDICT + "holds: .*one distance$", "accounting: target: 96 bits in every round; no round has proof of work; ", "accounting: THE ROUNDS: HOLD: ", "size: ")),
    ("no-round", None, (), 0, ("code: ", VERDICT + "none: the file lays no round with an error$", "accounting: none was formed: ")),
    ("no-line-field", FULL, (("field goldilocks\n", ""),), 0, (r"setting: field p = 2\^64 - 2\^32 \+ 1, ", VERDICT + "holds: .*one distance$")),
    ("another-field", FULL, (("field goldilocks\n", "field m61\n"),), 2, ("cp_plan: line [0-9]+ ",)),
    # A second group on the two codes of group `seg`, at the next smaller distance (`m:4143`;
    # cp_plan chooses m = 4144 for `seg`), so that the two distances are the only fault: every
    # round holds by itself and every hiding budget is met.
    ("two-distances", FULL, ((LOC, LOC.replace("=seg ", "=seg3 ")), ("group last ", "group seg3  main=long  mask=mask  k=7 h=0 cube=31 rows=2^16 regime=m:4143\ngroup last ")), 3,
     (VERDICT + r"NO: round [0-9]+, LOC, .*\(asked of the whole file as one plan", "accounting: THE ROUNDS: NO: by the plan's own verdict.*: round [0-9]+, LOC, ", "size: ")),
    ("a-product-that-wraps", FULL, (("  rng columns=2^16 where=segments\n", "  rng columns=2^62 where=segments\n"),), 3, (VERDICT + "NO: ", "accounting: THE PARAMETERS ARE NOT ADMITTED: ", "accounting: THE ROUNDS: NO: the parameters are not admitted$")),
    ("a-target-that-is-missed", FULL, (("target 128\n", "target 170\n"),), 3, (VERDICT + "NO: round [0-9]+, .*above the target", "accounting: THE ROUNDS: NO: round [0-9]+, ")),
    ("a-hiding-budget-that-is-missed", FULL, ((LOC, LOC.replace(" where=", " opens=1000 where=")),), 3, (VERDICT + "holds: .*BUT A HIDING BUDGET IS NOT MET", "accounting: THE ROUNDS: HOLD WITH WORK: ", "hiding budget: NOT MET")),
    ("an-unknown-word", FULL, (("target 128\n", "targets 128\n"),), 2, ("cp_plan: line [0-9]+ ",)),
    ("a-table-without-answers", FULL, (("table W code=table v=11 delta_once=1/8\n", "table W code=table v=11 delta_once=1/8 answers=0\n"),), 2, ("cp_plan: line [0-9]+ .*answers must be from 1 to 8",)),
)


def main():
    if len(sys.argv) != 2 or os.path.lexists(os.path.abspath(sys.argv[1])):
        print(__doc__.split("\n")[2], file=sys.stderr)
        return 2
    work = os.path.abspath(sys.argv[1])
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    program = os.path.join(work, "cp_plan")
    built = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-fno-ipa-modref", "-DBLK_FIELD=2", "-DBLK_HASH=1", "-I", os.path.dirname(NUMBERS), os.path.join(NUMBERS, "cp_plan.cpp"), "-o", program],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False)  # fmt: skip
    if built.returncode:
        print("cp_plan_files.py: cp_plan did not build: " + built.stdout.strip()[-300:])
        return 2
    bad = 0
    for name, file, turns, status, patterns in CASES:
        path = os.path.join(NUMBERS, "settings", file or FULL)
        if turns or file is None:
            with open(path, encoding="utf-8") as f:
                text = CODES if file is None else f.read()
            for old, new in turns:
                if text.count(old) != 1:
                    print("cp_plan_files.py: %s: the text %r occurs %d times in %s, not once" % (name, old, text.count(old), file))
                    return 2
                text = text.replace(old, new)
            path = os.path.join(work, name + ".txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        ran = subprocess.run([program, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False)
        lines, at, missed = ran.stdout.split("\n"), 0, None
        for pattern in patterns:
            hit = next((n for n in range(at, len(lines)) if re.match(pattern, lines[n])), None)
            if hit is None:
                missed = pattern
                break
            at = hit + 1
        ok = ran.returncode == status and missed is None
        bad += not ok
        print("%s | %s | status %d%s%s" % ("as expected" if ok else "NOT AS EXPECTED", name, ran.returncode, "" if ran.returncode == status else ", wanted %d" % status,
              "" if missed is None else "; no line after the earlier patterns' lines begins with: " + missed))  # fmt: skip
    print("cp_plan_files.py: %d files: %s" % (len(CASES), "all as expected" if not bad else "%d not as expected" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
