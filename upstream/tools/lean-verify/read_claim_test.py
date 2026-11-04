#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Holds read_claim.py against what cp_verify printed for the stored examples.

usage: python3 -B tools/lean-verify/read_claim_test.py [TREE]

For each stored example that has a request file, a token file and an ID file, read_claim.py reads the
claim, and its commitment ID, r, e, the range of the checked tokens, whether the next token is
claimed, and the policy's seven numbers are compared with the fields id, r, e, tokens, y, C, theta,
cap, s, sh, sT and F of the line "result-v1" that the example's README.md shows: a line that the C++
verifier printed for these files. T is taken from that line, and the numbers of the prompt's, the
output's and the further tokens must add up to it. The tokens themselves are not in that line: for
them the test reads the two files once more, by slices of bytes, and compares.

A pass shows that two readings of four stored claims agree. It runs no verifier and no Lean, and it
says nothing about Lean's own reading of the claim. It writes no file.
Exit code 0: the last line ends "all as expected". 1: a difference. 2: a wrong call, or an example
or its line is missing.
"""
import os
import re
import subprocess
import sys

EXAMPLES = ["accepted-proof", "accepted-proof-clipped-gap", "full-size", "full-size-clipped-gap"]


def main(argv):
    if len(argv) > 2:
        print("usage: read_claim_test.py [TREE]", file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    tree = argv[1] if len(argv) == 2 else os.path.join(here, "..", "..")
    off = 0
    for name in EXAMPLES:
        d = os.path.join(tree, "examples", name)
        try:
            with open(os.path.join(d, "README.md"), encoding="utf-8") as f:
                shown = [l for l in f.read().split("\n") if l.startswith("result-v1 ")]
            with open(os.path.join(d, "example.req"), "rb") as f:
                req = f.read()
            with open(os.path.join(d, "example.y"), "rb") as f:
                tok = f.read()
        except OSError as e:
            print("read_claim_test.py: %s: %s" % (name, e.strerror), file=sys.stderr)
            return 2
        if len(shown) != 1:
            print("read_claim_test.py: %s: README.md does not show exactly one line result-v1" % name, file=sys.stderr)
            return 2
        want = dict(p.split("=", 1) for p in shown[0].split()[1:] if "=" in p)
        if not want.get("T", "").isdigit():
            print("read_claim_test.py: %s: README.md's line result-v1 has no field T" % name, file=sys.stderr)
            return 2
        run = subprocess.run(
            [sys.executable, "-I", "-B", os.path.join(here, "read_claim.py"), os.path.join(d, "config"),
             os.path.join(d, "example.req"), os.path.join(d, "example.y"), os.path.join(d, "ID.txt")],
            stdout=subprocess.PIPE, check=False)
        m = re.fullmatch(
            r"claim: commitment ID: ([0-9a-f]{64})\n"
            r"claim: checked positions: from r = (\d+) up to, not including, e = (\d+)\n"
            r"claim: prompt, (\d+) tokens: \[([0-9, ]*)\]\n"
            r"claim: output, (\d+) tokens: \[([0-9, ]*)\]\n"
            r"not part of the claim: tokens of the request behind position e, which go to the verifier "
            r"with the proof: (\d+)\n"
            r"public files: the acceptance policy, by the parameter file's own lines: "
            r"\[C (\d+), theta (\d+), cap (\d+), s (\d+), sh (\d+), sT (\d+), F (\d+)\]\n",
            run.stdout.decode("ascii", "replace"))
        if run.returncode != 0 or m is None:
            print("NOT AS EXPECTED: %s: read_claim.py ended with %d, or its six lines have another form" % (name, run.returncode))
            off += 1
            continue
        ident, r, e, np, prompt, no, output, nf, C, theta, cap, s, sh, sT, F = m.groups()
        r, e, T = int(r), int(e), int(want["T"])
        got = {"id": ident, "r": str(r), "e": str(e), "tokens": "%d..%d" % (r + 1, min(e, T - 1)),
               "y": "yes" if e == T else "no", "C": C, "theta": theta, "cap": cap, "s": s, "sh": sh, "sT": sT, "F": F}
        for k, v in got.items():
            if want.get(k) != v:
                print("NOT AS EXPECTED: %s: %s: read_claim.py has %s, the stored line %s" % (name, k, v, want.get(k)))
                off += 1
        word = lambda b, i: int.from_bytes(b[4 * i : 4 * i + 4], "little")
        first = [word(req, i) for i in range(r + 1)]
        then = [word(req, i) for i in range(r + 1, min(e, T - 1) + 1)] + ([word(tok, 0)] if e == T else [])
        nums = lambda s: [int(x) for x in s.split(", ")] if s else []
        if nums(prompt) != first or nums(output) != then or int(np) != len(first) or int(no) != len(then):
            print("NOT AS EXPECTED: %s: the tokens are not those of the two files" % name)
            off += 1
        if int(np) + int(no) + int(nf) != T + (1 if e == T else 0):
            print("NOT AS EXPECTED: %s: prompt, output and further tokens do not add up" % name)
            off += 1
    if off:
        print("read_claim_test.py: %d stored examples: NOT AS EXPECTED: %d" % (len(EXAMPLES), off))
        return 1
    print("read_claim_test.py: %d stored examples: all as expected" % len(EXAMPLES))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
