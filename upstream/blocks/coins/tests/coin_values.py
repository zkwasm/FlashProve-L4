#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Known answers of blocks/coins/coins.h (private randomness), computed without any of that code.

`f` implements ChaCha20's block function from RFC 8439, sections 2.1 to 2.3, and is checked
against the RFC's test vector of 2.3.2 before anything is printed. `key`, `take`, `lane`
and `element` restate coin_key(), coin_key_take(), coin_lane() and coin_element() of
blocks/coins/coins.h; `admitted` with `field` = 2 restates F_from_word() (blocks/field/field.h).

Usage: coin_values.py > <a new file>, then compare it with blocks/coins/tests/coin_values.txt,
which blocks/coins/tests/coins_test.cpp reads. Not "> coin_values.txt": the shell empties it first.
No test in SUITE.md runs this script.

The comparison in blocks/coins/tests/coins_test.cpp checks how keys are derived, which words go
into and come out of each call of the block function, and the retry after a rejected ("refused")
word. It uses only the values for p = P[2], the field of blocks/field/field.h, not those for P[1].
It cannot show a fault common to `f` and chacha20_block() (blocks/coins/chacha.h), which both
compute the RFC's block function. The RFC's vector checks that function, here and in the test's
rfc_vector().
"""
M32 = 2**32 - 1
P = {1: 2**61 - 1, 2: 2**64 - 2**32 + 1}


def rotl(x, n):
    return (x << n | x >> (32 - n)) & M32


def quarter(s, a, b, c, d):
    s[a] = (s[a] + s[b]) & M32; s[d] = rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & M32; s[b] = rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & M32; s[d] = rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & M32; s[b] = rotl(s[b] ^ s[c], 7)


def f(key, x0, x1, x2, x3):
    first = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] + list(key) + [x0, x1, x2, x3]
    s = list(first)
    for _ in range(10):
        for q in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
            quarter(s, *q)
    return [(a + b) & M32 for a, b in zip(s, first)]


def key(K, a, b, c, d):
    return f(K, a, b, c, d)[:8]


def take(K, words):
    for i in range(0, len(words), 4):
        K = key(K, *words[i:i + 4])
    return K


def context_key(process, counter, version, obj, member):
    return key(key(key(process, counter & M32, counter >> 32, 0, 0), 0, 0, version, obj), member, 0, 0, 0)


def lane(K, stream, call, retry, q):
    o = f(K, call & M32, stream, retry, call >> 32)
    return o[2 * q] | o[2 * q + 1] << 32


def admitted(field, w):
    if field == 1:
        return None if (w & P[1]) == P[1] else w & P[1]
    return None if w >= P[2] else w


def element(field, K, stream, i):
    """Element i, and the number of rejected ("refused") words before it."""
    w, k = lane(K, stream, i // 4, 0, i % 4), 0
    while admitted(field, w) is None:
        k += 1
        w = lane(K, stream, i // 4, 8 * k + i % 4, 0)
    return admitted(field, w), k


def main():
    rfc = f([0x03020100 + 0x04040404 * i for i in range(8)], 1, 0x09000000, 0x4a000000, 0)
    assert rfc == [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
                   0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2]
    print('# Known answers of blocks/coins/coins.h, written by coin_values.py, which shares no code with it. Numbers are hexadecimal.')
    seed = [0x01010101 * (i + 1) for i in range(8)]
    child = key(seed, 0x44535043, 6, 0, 0)
    fresh = [0x10203040 + i for i in range(8)]
    inputs = [0x0A0B0C0D * (i + 1) & M32 for i in range(8)]
    process = take(take(child, fresh), inputs)
    print('# seed word i = 01010101 (i + 1); child = seed_child(seed, 6, 0); fresh word i = 10203040 + i; inputs word i = 0a0b0c0d (i + 1)')
    print('child', ' '.join('%08x' % w for w in child))
    print('process', ' '.join('%08x' % w for w in process))
    print('# the secret of a table\'s making: the same child and fresh words, then the table\'s digest, word i = 0f0e0d0c (i + 1)')
    print('making', ' '.join('%08x' % w for w in take(take(child, fresh), [0x0F0E0D0C * (i + 1) & M32 for i in range(8)])))
    print('# context: counter, version, object, member, then the key')
    contexts = [(0, 0, 0, 0), (5, 0x81902, 1, 3), (2**40 + 7, 0x81902, 6, 0)]
    for c in contexts:
        print('context', '%x %x %x %x' % c, ' '.join('%08x' % w for w in context_key(process, *c)))
    K = context_key(process, *contexts[1])
    print('# under the second context: stream, number of the word, the word, the element with p = 2^61 - 1, the element with')
    print('# p = 2^64 - 2^32 + 1, and the refused words before each of the two')
    places = [(16384, i) for i in range(9)] + [(16385, 0), (16384 + 2**20, 3), (16390, 2**32 + 1), (16390, 2**34 + 2), (2**30 + 16383, 2**45 + 3)]
    places += REFUSED
    for stream, i in places:
        e1, k1 = element(1, K, stream, i)
        e2, k2 = element(2, K, stream, i)
        print('coin', '%x %x %016x %016x %016x %d %d' % (stream, i, lane(K, stream, i // 4, 0, i % 4), e1, e2, k1, k2))


# (stream, word number) of words rejected with p = P[2], under main()'s second context;
# they were found by a search over the word numbers.
REFUSED = [(16400, 362584566), (16400, 1327292691), (16400, 6312099836), (16400, 16942013412)]

main()
