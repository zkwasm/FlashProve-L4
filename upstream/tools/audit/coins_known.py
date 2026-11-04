#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Self-test of tools/audit/coins.py, which derives the committer's and prover's private random
# values (`coins` in names).
# Argument: the repository's top directory.
# Exit status 0: all as the known answers. 1: something differs, or the argument is not this
# repository. 77: python3 lacks numpy.
# First `Schedule`'s keys and draw_copy(), against prover/tests/coin_keys_values.txt. The
# synthetic inputs, not the functions, are copied from prover/tests/coin_keys_values.py, which
# prints that file's contents.
import os, sys
sys.dont_write_bytecode = True
# format_rules.py is imported only from the repository that this file is in.
tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if len(sys.argv) != 2 or os.path.realpath(sys.argv[1]) != tree: sys.exit("coins_known.py: the argument must be the repository that this file is in: " + tree)
try:
    import numpy  # coins.py imports it
except ImportError:
    print("not run: python3 lacks numpy here")
    sys.exit(77)
sys.path.insert(0, tree + "/docs/format")
import format_rules as R
import coins as C
def pattern(a, b, n=32): return bytes((a * i + b) & 255 for i in range(n))
want = dict(l.split() for l in open(tree + "/prover/tests/coin_keys_values.txt") if not l.startswith("#"))
S = C.Schedule(R.H, C.words8(pattern(7, 1)))
A, B, c, ID = pattern(5, 3), pattern(11, 9), pattern(3, 2), pattern(13, 4)
SHAPE = [33, 2**32 + 5, 44040191, 11, 0, 40]; NAME, ANS = pattern(9, 1, 112), pattern(17, 6, 24)
first = S.batch_first(B, c, 5, 2, [[1, 2, 3, 0xFFFFFFFF], None, [7, 9]])
got = dict(first_table=S.first_table(c, SHAPE), copy_table=S.copy_table(3, A, c, SHAPE), key_file=S.key_file(B, ID), entry=S.entry(3, A, ID, NAME), entry_with_answers=S.entry(3, A, ID, NAME, ANS),
           batch_first=first, batch_process=S.batch_process(first, NAME))
bad = 0
import struct
for n, k in got.items():
    ok = struct.pack("<8I", *k).hex() == want[n]; bad += not ok
    print("%-20s %s" % (n, "as the known answer" if ok else "DIFFERS"))
ok = str(C.draw_copy(first, 1000)) == want["draw"]; bad += not ok
# Then the generator: context_key() and Context.elements(), against
# blocks/coins/tests/coin_values.txt (printed by blocks/coins/tests/coin_values.py, which shares
# no code with blocks/coins/coins.h). Each element is compared for both values of `field`: 2 is
# p = 2^64 - 2^32 + 1 (Goldilocks), the only field that blocks/base.h compiles for; 1 is
# the modulus 2^61 - 1 (`C.P[1]`). Four elements at `field` 2 are first retries after a rejected
# word (`refused` below; chance 2^-32 per word). At `field` 1 the retry is not tested: elements()
# rejects only a word whose low 61 bits are all ones (chance 2^-61), and no such word is known.
lines = [l.split() for l in open(tree + "/blocks/coins/tests/coin_values.txt") if not l.startswith("#")]
process = [int(x, 16) for x in next(l for l in lines if l[0] == "process")[1:]]
keys = coins = refused = 0
for l in lines:
    if l[0] == "context":
        keys += 1; bad += [int(x, 16) for x in l[5:]] != [int(x) for x in C.context_key(process, *(int(x, 16) for x in l[1:5]))]
second = [int(x, 16) for x in [l for l in lines if l[0] == "context"][1][1:5]]
for l in lines:
    if l[0] == "coin":
        stream, number = int(l[1], 16), int(l[2], 16); coins += 1; refused += int(l[7]) > 0
        for field, want_e in ((1, l[4]), (2, l[5])): bad += C.Context(process, *second, field, "known answers").elements(stream, number, 1, "a known answer") != [int(want_e, 16)]
print("%-20s %s" % ("the generator", "%d contexts' keys, %d coins at both fields, %d of them behind a refused word: %s" % (keys, coins, refused, "as the known answers" if not bad else "SOMETHING DIFFERS")))
print("%-20s %s" % ("draw", "as the known answer" if ok else "DIFFERS"))
sys.exit(1 if bad else 0)
