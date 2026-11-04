#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The hash functions C, H and TH of verifier/HASHES.md in Python, built on the F of blake2b_f.py.

The module imports no code of the programs, and of scales/policy_tables.py it takes one number, DW. The F of blake2b_f.py has the
same outside source as the programs' F (that file's head). The known answers of their unit tests are computed again and checked here.

A digest has DW = 8 words of 32 bits. No environment variable is read.

    format_hash.py                checks that H, with the high half of F's counter forced to 0, is
                                  hashlib's BLAKE2b-512 of the list's bytes, for lists of 0 to 300
                                  words; that the domain-separation number (`num`) and the list's
                                  length, which go into that half, change the result; TH's edge
                                  cases; and the vector file, verifier/tests/hash_vectors.txt: each
                                  line's value, and that the lines are those that --vectors prints
    format_hash.py --hold <file>  checks lines "H <number> <L> <state>" and "TH <L> <digest>" (words
                                  of 8 hex digits) on the test lists: word k of a list of L words is
                                  ((k + 1) * 0x89ABCDEF) mod 2^32, k = 0 .. L - 1
    format_hash.py --vectors      prints the vector file (H_LENGTHS explains its lists' lengths)

The script exits with code 1 if anything differs, and with 2 on a wrong command line.
"""

import hashlib
import os
import struct
import sys

from blake2b_f import START_512, f, list_bytes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scales"))
from policy_tables import DW  # noqa: E402  words of a digest: 8


def arguments():
    """The calling script's arguments."""
    return sys.argv[1:]


SW, BLKW = 16, 32  # the sizes of a state and a block, in 32-bit words
assert DW == 8
THPW = 16 * DW  # the size of a piece of TH


def wide(words):
    """Two 32-bit words make a 64-bit word of F, the low word first."""
    return [words[i] | words[i + 1] << 32 for i in range(0, len(words), 2)]


def narrow(words):
    return [x for w in words for x in (w & 0xFFFFFFFF, w >> 32)]


# The first state is BLAKE2b-512's (RFC 7693, 3.3: IV with h[0] xor 0x01010040).
FIRST = narrow(START_512)


def c(cv, m, counter, length, num, last):
    """C(cv, m, c, l, num, last) = F(cv, m, t, f) with t mod 2^64 = c, t >> 64 = num + 2^32 l,
    and f set iff last."""
    assert len(cv) == SW and len(m) == BLKW and all(0 <= x < 2**32 for x in (*cv, *m, length, num)) and 0 <= counter < 2**64
    return narrow(f(wide(cv), wide(m), counter | (num | length << 32) << 64, last))


def h(num, w, high=True):
    """H(num; w): the state. With high=False the high half of t is forced to 0, which gives BLAKE2b-512."""
    length = len(w)
    blocks = max(1, -(-length // BLKW))
    filled = list(w) + [0] * (blocks * BLKW - length)
    cv = FIRST
    for b in range(blocks):
        last = b == blocks - 1
        counter = 4 * length if last else 4 * BLKW * (b + 1)
        cv = c(cv, filled[BLKW * b : BLKW * (b + 1)], counter, length if high else 0, num if high else 0, last)
    return cv


def digest(num, w):
    return h(num, w)[:DW]


def tree_f(num, t):
    """The number that c() gets for a call in Merkle tree t: `num` in the low 16 bits, t above."""
    assert 0 <= num < 2**16 and 0 <= t < 2**16
    return num + (t << 16)


def hp(num, t, x, w):
    """H with a place, H(num, t, x; w): the state. As h(), but c() gets the number tree_f(num, t),
    and bits 32 to 63 of its counter hold x, the heap index (root 1) of the vertex computed."""
    length = len(w)
    assert 1 <= x < 2**32 and length < 2**30
    blocks = max(1, -(-length // BLKW))
    filled = list(w) + [0] * (blocks * BLKW - length)
    cv = FIRST
    for b in range(blocks):
        last = b == blocks - 1
        c0 = 4 * length if last else 4 * BLKW * (b + 1)
        cv = c(cv, filled[BLKW * b : BLKW * (b + 1)], c0 | x << 32, length, tree_f(num, t), last)
    return cv


def digest_at(num, t, x, w):
    return hp(num, t, x, w)[:DW]


def th(w):
    """TH, on a list of at least DW words: hashes its pieces of THPW words under the number 10,
    then each level of digests in the same way under 11, until one digest is left."""
    assert len(w) >= DW
    number = 10
    while True:
        w = [x for i in range(0, len(w), THPW) for x in digest(number, w[i : i + THPW])]
        if len(w) == DW:
            return w
        number = 11


def made(length):
    return list(struct.unpack("<%dI" % length, list_bytes(4 * length)))


def hexes(words):
    return "".join("%08x" % x for x in words)


# Lengths, in words, of the vector file's lists, which made() builds. H: 0, 1; around a digest, a
# block, two blocks and a piece of TH; one long list; vectors() adds two more lines. TH: one piece
# (four lengths); two (three); a full second level, then one word, one digest more; 3 to 5 levels.
H_LENGTHS = (0, 1, DW - 1, DW, DW + 1, 31, 32, 33, 63, 64, 65, THPW, THPW + 1, 1000)
TH_LENGTHS = (DW, DW + 1, THPW - 1, THPW, THPW + 1, 17 * DW, 2 * THPW, 16 * THPW, 16 * THPW + 1, 257 * DW, 160 * THPW, 256 * THPW + 1, 1000001)


def vectors():
    out = ["H %d %d %s" % (num, n, hexes(h(num, made(n)))) for n in H_LENGTHS for num in (3, 10, 107)]
    out += ["H %d %d %s" % (num, n, hexes(h(num, made(n)))) for num, n in ((2147483651, 33), (107, 70001))]
    return out + ["TH %d %s" % (n, hexes(th(made(n)))) for n in TH_LENGTHS]


def main():
    bad = 0

    def check(good, what):
        nonlocal bad
        bad += not good
        print("%-9s %s" % ("matches" if good else "differs", what))

    def hold(path):
        """Checks the file's lines. Returns how many of each kind it has."""
        nonlocal bad
        kinds = {"H": 0, "TH": 0}
        with open(path) as lines:
            for line in lines:
                x = line.split()
                mine = hexes(h(int(x[1]), made(int(x[2])))) if x[0] == "H" else hexes(th(made(int(x[1]))))
                kinds[x[0]] += 1
                if mine != x[-1]:
                    bad += 1
                    print("differs   " + " ".join(x[:-1]))
        print("%d lines checked, %d with a different answer" % (sum(kinds.values()), bad))
        return kinds

    args = arguments()
    if len(args) == 2 and args[0] == "--hold":
        hold(args[1])
        return 1 if bad else 0
    if args == ["--vectors"]:
        print("\n".join(vectors()))
        return 0
    if args:
        print("usage: format_hash.py [--hold <file> | --vectors]", file=sys.stderr)
        return 2
    check(all(struct.pack("<%dI" % SW, *h(0, made(n), high=False)) == hashlib.blake2b(list_bytes(4 * n)).digest() for n in range(301)),
          "with the high half of t forced to 0, H is BLAKE2b-512 on the list's bytes, L = 0 .. 300 (hashlib)")
    w = made(40)
    check(len({hexes(x) for x in (h(3, w), h(10, w), h(3, w, high=False), h(3, w + [0]), h(3, w[:-1]))}) == 5,
          "a different number, the plain routine, a list longer by a zero word, a shorter list: five distinct states")
    check(all(th(made(n)) == digest(10, made(n)) for n in (DW, DW + 1, THPW)), "TH: %d, %d and %d words: one call of H under 10" % (DW, DW + 1, THPW))
    w = made(THPW + 1)
    check(th(w) == digest(11, digest(10, w[:THPW]) + digest(10, w[THPW:])), "TH: %d words: two pieces under 10, the list of their digests under 11" % (THPW + 1))
    check(th(w) != th(digest(10, w[:THPW]) + digest(10, w[THPW:])), "TH: a list and the list of the digests of its pieces have different values")
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "hash_vectors.txt")
    kinds = hold(path)
    check(kinds == {"H": 44, "TH": 13}, "the vector file of digests of %d words has 44 lines of H and 13 of TH" % DW)
    with open(path) as lines:
        heads = [tuple(x.split()[:-1]) for x in lines]
    wanted = [("H", str(num), str(n)) for n in H_LENGTHS for num in (3, 10, 107)] + [("H", "2147483651", "33"), ("H", "107", "70001")]
    check(heads == wanted + [("TH", str(n)) for n in TH_LENGTHS], "its lines are those that --vectors prints, in their order")
    print("all as expected" if not bad else "%d differ" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
