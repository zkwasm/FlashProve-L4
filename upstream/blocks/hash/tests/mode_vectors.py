#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Prints known answers for blocks/hash/hash.h, each from one call of Python's hashlib.blake2b.

Columns: domain number (`Domain` in blocks/hash/domains.h), tree (under number 162, a
Fiat-Shamir hash chain's label), index, start (BLAKE2b's `salt` and `person` parameters:
64 hex digits, or - for none), W, s, n, digest bytes (32 or 64), digest in hex.
A Merkle leaf (numbers 164, 173, 174) has W 8-byte symbols and s salt bytes; otherwise both are 0.

n bytes follow H()'s 16-byte label; byte i is (7 i + 3) mod 251. A leaf has 8 W such bytes; then
zeros up to the next multiple of 128 bytes (BLAKE2b's block; label included) if the salt would
cross one; then the salt (hashed data, not BLAKE2b's `salt`), whose byte k is (5 k + 1) mod 251.

Usage: mode_vectors.py > <a new file>, then compare it with blocks/hash/tests/mode_vectors.txt,
which blocks/hash/tests/mode_test.cpp reads. No test runs this script. Redirecting into that file
erases the stored answers: the shell empties it before the script runs.
"""
import hashlib
import struct


def H(number, tree, index, start, W, s, data, length):
    start = start or bytes(32)
    label = struct.pack('<QQ', number + (tree << 16) + (W << 32) + ((s // 8) << 56), index)
    return hashlib.blake2b(label + data, digest_size=length, salt=start[:16], person=start[16:]).digest()


def main():
    print('# Known answers of blocks/hash/hash.h, written by mode_vectors.py: each is one call of hashlib.blake2b.')
    start = bytes((11 * i + 5) % 256 for i in range(32))
    cases = [(168, 0, 0, None, 0, 0, n, 32) for n in (0, 1, 7, 8, 31, 32, 63, 64, 65, 111, 112, 113, 127, 128, 129, 191, 192, 193, 239, 240, 241, 255, 256, 257, 367, 368, 369, 1000)]
    cases += [(166, 0, 0, start, 0, 0, 64, 32), (178, 0, 0, start, 0, 0, 64, 32), (172, 0, 3, None, 0, 0, 64, 32), (176, 0, 0, None, 0, 0, 500, 32),
              (160, 0, 1, None, 0, 0, 96, 32), (179, 0, 0, None, 0, 0, 152, 32), (169, 0, 2, None, 0, 0, 72, 32), (171, 0, 7, start, 0, 0, 0, 32), (171, 0, 2**40 + 1, start, 0, 0, 0, 32)]
    cases += [(162, c, r, start, 0, 0, n, 64)
              for c, r, n in ((0, 1, 16), (1, 12, 24), (2, 12, 64), (2, 12, 112), (2, 12, 120), (2, 12, 128), (2, 12, 136), (2**32, 12, 144), (2**48 - 1, 2**40, 272))]
    cases += [(number, tree, index, start, W, s, None, 32)
              for number, tree, index, W, s in ((164, 3, 2**20 + 77, 23, 40), (164, 65535, 2**63 - 1, 0, 40), (164, 0, 5, 9, 40), (164, 0, 5, 10, 40), (164, 0, 5, 11, 40),
                                                (164, 0, 5, 14, 40), (164, 0, 5, 10, 48), (164, 9, 2**33 + 5, 7, 64), (164, 9, 2**33 + 5, 8, 64), (164, 1, 6, 300, 56),
                                                (173, 0, 2**5 + 9, 4, 40), (174, 0, 2**33 + 2**32 + 1, 2048, 40))]
    for number, tree, index, st, W, s, n, length in cases:
        if n is None:
            data = bytes((7 * i + 3) % 251 for i in range(8 * W))
            if (16 + len(data)) % 128 + s > 128:
                data += bytes(-(16 + len(data)) % 128)
            data += bytes((5 * k + 1) % 251 for k in range(s))
        else:
            data = bytes((7 * i + 3) % 251 for i in range(n))
        print(number, tree, index, st.hex() if st else '-', W, s, len(data), length, H(number, tree, index, st, W, s, data, length).hex())


main()
