#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Known answers for the Reed-Solomon code of the weight table's rows: nine symbols of one long row.

prover/hash_commitment/tests/hf_code_test.cu contains the nine values (--hold checks that) and
computes them again with the factors that hf_f_values() in prover/hash_commitment/hf_code.cuh gives.

The row is over E = F[X] / (X^3 - 2), where F is the field with p = 2^64 - 2^32 + 1 elements
(docs/format/FORMAT.md R1.4). It has 446,693,375 coefficients, as many as a row of the
full-size example configuration (a random tail of 44,040,191 entries, then 402,653,184 message
entries: docs/format/09-key-and-leaf-files.md R12.11). The symbol at position x of a code of
length 2^a is the sum of c_i z_x^i, where c_i = coefficient(pt, i) with pt = made(0), .., made(28)
and z_x = point(a, x). Before it prints, main() compares symbol() with the sum taken term by term,
at every position of four small codes.

How far the values are independent of hf_code.cuh: this script shares no line with it, but takes
the definition of the coefficients from hf_code_test.cu. Six positions are below 2^32 and have
R12.11's points; three of these are in the code with a = 33, and R12.11 covers only a <= 32. The
three from 2^32 on rest on the rule of the next coset (see point()), which no numbered rule of the
format gives (R12.11 only reports it from the programs' source) and which is taken from the
sources, hf_code.cuh among them: there the agreement shows arithmetic, not the rule. No example
configuration (prover/examples/) has a = 33.

usage: code_test_values.py [--hold <file>]
Prints one line for each case: a, x, and the low and high 32 bits of the symbol's e_0, e_1, e_2.
With --hold, each case must also be in <file>, written {a, x, {its six words as 0x.. constants}}:
exit code 1 if one is missing. With any other command line it only prints.
"""
import re
import sys

P = (1 << 64) - (1 << 32) + 1
ROOT_32 = pow(7, 5 * ((1 << 32) - 1), P)  # of order 2^32
assert ROOT_32 == 0xE2AF70635744FD26
ONE, ZERO = (1, 0, 0), (0, 0, 0)


def mul(a, b):  # in E: X^3 = 2
    return ((a[0] * b[0] + 2 * (a[1] * b[2] + a[2] * b[1])) % P,
            (a[0] * b[1] + a[1] * b[0] + 2 * a[2] * b[2]) % P,
            (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % P)


def add(a, b):
    return tuple((x + y) % P for x, y in zip(a, b))


def sub(a, b):
    return tuple((x - y) % P for x, y in zip(a, b))


def scale(a, s):
    return tuple(x * s % P for x in a)


# An arbitrary fixed element of E.
def made(v):
    x = (1001 + 3 * v, 1002 + 3 * v, 1003 + 3 * v)
    for _ in range(6):
        x = mul(x, x)
    return x


# The point z_x: R12.11's for x < 2^32. For a = 33 a position from 2^32 on is on the next coset:
# z_x = 7^(1 + (x div 2^32)) times R12.11's product of roots of unity over the low 32 bits of x.
def point(a, x):
    assert 0 <= x < 1 << a
    z = pow(7, 1 + (x >> 32), P)
    for i in range(min(a, 32)):
        if x >> i & 1:
            z = z * pow(ROOT_32, 1 << (31 - i), P) % P
    return z


# The product over v < n of pt[v] if bit n - 1 - v of i is 1, and of 1 - pt[v] if it is 0.
def coefficient(pt, i):
    n, c = len(pt), ONE
    for v in range(n):
        c = mul(c, pt[v] if i >> (n - 1 - v) & 1 else sub(ONE, pt[v]))
    return c


def symbol_term_by_term(pt, z, count):
    s = ZERO
    for i in range(count):
        s = add(s, scale(coefficient(pt, i), pow(z, i, P)))
    return s


def symbol(pt, z, count):
    """The sum over i < count of coefficient(pt, i) z^i, split at the top bit of i at each step."""
    n = len(pt)
    bit_0 = [sub(ONE, pt[n - 1 - b]) for b in range(n)]                    # the factor of bit b of the index if it is 0
    bit_1 = [scale(pt[n - 1 - b], pow(z, 1 << b, P)) for b in range(n)]    # and if it is 1, with its power of z
    every = [ONE]                                                          # every[k]: the sum over all 2^k values of the low k bits
    for b in range(n):
        every.append(mul(every[b], add(bit_0[b], bit_1[b])))

    def below(k, count):  # the sum over the indices below count, count <= 2^k, of the low k bits' factors
        if count == 0:
            return ZERO
        if count == 1 << k:
            return every[k]
        half = 1 << (k - 1)
        if count <= half:
            return mul(bit_0[k - 1], below(k - 1, count))
        return add(mul(bit_0[k - 1], every[k - 1]), mul(bit_1[k - 1], below(k - 1, count - half)))

    return below(n, count)


def main():
    for a, count, n in ((6, 21, 5), (5, 13, 4), (7, 63, 6), (4, 8, 3)):
        pt = [made(v) for v in range(n)]
        for x in range(1 << a):
            assert symbol(pt, point(a, x), count) == symbol_term_by_term(pt, point(a, x), count)
    pt = [made(v) for v in range(29)]
    held = None
    if sys.argv[1:2] == ["--hold"] and len(sys.argv) == 3:
        with open(sys.argv[2], encoding="utf-8") as f:
            held = f.read()
    missing = cases = 0
    for a, positions in ((33, (0, 1, 2**32 - 1, 2**32, 2**32 + 1, 2**33 - 1)), (32, (2**31 - 1, 2**31, 2**31 + 1))):
        for x in positions:
            e = symbol(pt, point(a, x), 446693375)
            print(a, x, " ".join("%08x %08x" % (c & 0xFFFFFFFF, c >> 32) for c in e))
            cases += 1
            words = r",\s*".join(r"0x%08x,\s*0x%08x" % (c & 0xFFFFFFFF, c >> 32) for c in e)
            if held is not None and not re.search(r"\{%d, %d(ull)?, \{%s\}\}" % (a, x, words), held):
                missing += 1
                print("  NOT IN THE FILE: the case a = %d, x = %d with these six words" % (a, x))
    if held is not None:
        print("code_test_values.py: %d cases; %d of them do not stand in %s" % (cases, missing, sys.argv[2]))
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
