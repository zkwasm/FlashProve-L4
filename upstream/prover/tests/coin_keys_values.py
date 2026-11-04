# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

#
# Terms: docs/glossary.md.
"""Known answers for the keys of the programs' private randomness (prover/common/coin_keys.h).

A third implementation of that header's table of keys, besides
prover/second_writing/schedule_keys.h, on hashlib's BLAKE2b and ChaCha20's block function as
specified in RFC 8439, 2.1 to 2.3. prover/tests/coin_keys_test.cpp checks both headers and
prover/common/cp_keys_own.h against the output, prover/tests/coin_keys_values.txt, on the same
synthetic inputs; a shared fault does not show.

A line: a key's name, its 32 bytes in hex (eight words, lowest byte first). The last line:
what draw_leaf() (prover/common/cp_files.h) returns for LIVE live leaf tables (weight tables;
`copy` in names) under the key `batch_first`.

Usage, from prover/: python3 -B tests/coin_keys_values.py > <a new file>, then compare it with
tests/coin_keys_values.txt (not "> tests/coin_keys_values.txt": the shell empties it first).
"""
import hashlib
import struct


def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


# RFC 8439, 2.1: the quarter round.
def quarter(s, a, b, c, d):
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = rotl(s[b] ^ s[c], 7)


# x0 replaces the block counter, x1 to x3 the nonce. Not run on the RFC's test vectors.
def f(key, x0, x1, x2, x3):
    init = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574] + list(key) + [x0, x1, x2, x3]
    s = list(init)
    for _ in range(10):
        quarter(s, 0, 4, 8, 12); quarter(s, 1, 5, 9, 13); quarter(s, 2, 6, 10, 14); quarter(s, 3, 7, 11, 15)
        quarter(s, 0, 5, 10, 15); quarter(s, 1, 6, 11, 12); quarter(s, 2, 7, 8, 13); quarter(s, 3, 4, 9, 14)
    return [(a + b) & 0xFFFFFFFF for a, b in zip(s, init)]


def key(K, a, b, c, d):
    return f(K, a, b, c, d)[:8]


def words(b):
    return list(struct.unpack('<8I', b))


def take(K, b):
    w = words(b)
    return key(key(K, *w[:4]), *w[4:])


def q(x):
    return struct.pack('<Q', x)


# 176 is `COIN_INPUTS` in blocks/hash/domains.h.
def digest(items):
    s = q(176) + q(0)
    for name, b in items:
        n = name.encode()
        s += q(len(n)) + n + q(len(b)) + b
    return hashlib.blake2b(s, digest_size=32).digest()


# 0x44535043, 0x4B425043, 0x464C5043: "CPSD", "CPBK", "CPLF" in little-endian ASCII.
def form(seed, kind, number, thirty_two, items):
    return take(take(key(seed, 0x44535043, kind, number, 0), thirty_two), digest(items))


def pattern(a, b, n=32):
    return bytes((a * i + b) % 256 for i in range(n))


SEED = words(pattern(7, 1))
SYSTEM_A, SYSTEM_B = pattern(5, 3), pattern(11, 9)
C, ID = pattern(3, 2), pattern(13, 4)
SHAPE = [33, 2**32 + 5, 44040191, 11, 0, 40]
NAME, ANSWERS = pattern(9, 1, 112), pattern(17, 6, 24)
SLOTS = [[1, 2, 3, 0xFFFFFFFF], None, [7, 9]]
COPIES, FIRST_LIVE, K = 5, 2, 3
LIVE = 1000


def shape(s):
    return ('the shape', b''.join(q(x) for x in s))


def table(name, answers):
    return [("the copy's name", name), ("the copy's answers", answers)]


def main():
    out = {}
    out['first_table'] = form(SEED, 1, 0, bytes(32), [("a stored table's making", b''), ('the configuration', C), shape(SHAPE), ('the first table', b'')])
    out['copy_table'] = form(SEED, 7, K, SYSTEM_A, [("a stored table's making", b''), ('the configuration', C), shape(SHAPE), ('copy', q(K))])
    out['key_file'] = form(SEED, 8, 0, SYSTEM_B, [("the key file's making", b''), ('the published ID', ID)])
    out['entry'] = form(SEED, 5, K, SYSTEM_A, [("an entry's making", b''), ('the published ID', ID)] + table(NAME, b''))
    out['entry_with_answers'] = form(SEED, 5, K, SYSTEM_A, [("an entry's making", b''), ('the published ID', ID)] + table(NAME, ANSWERS))
    slots = [('request', b''.join(struct.pack('<I', w) for w in s)) if s is not None else ('no request', b'') for s in SLOTS]
    first = form(SEED, 6, 0, SYSTEM_B, [('a batch', b''), ('the configuration', C), ("the prover's revision", q(1)), ('copies', q(COPIES)), ('the first live copy', q(FIRST_LIVE)), ('slots', q(len(SLOTS)))] + slots)
    out['batch_first'] = first
    out['batch_process'] = take(key(first, 0x4B425043, 0, 0, 0), digest([("the batch's stored copy", b'')] + table(NAME, b'')))
    print('# Known answers of common/coin_keys.h, written by tests/coin_keys_values.py: made-up inputs; hashlib, and the ChaCha20 block function as specified in RFC 8439, 2.1 to 2.3.')
    for name, k in out.items():
        print(name, struct.pack('<8I', *k).hex())
    t = 0
    while True:
        block = f(first, 0x464C5043, t, 0, 0)
        v = block[0] | block[1] << 32
        if v < 2**64 // LIVE * LIVE:
            break
        t += 1
    print('draw', v % LIVE)


main()
