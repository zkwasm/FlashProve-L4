#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""The committer's and prover's private randomness (`coins` in names), derived again for the audit.

The audit (tools/audit/README.md) compares it with the random values in published files.
Derived from docs/format/COIN-KEYS.md (sections 1, 2, 2a), cited below by section, unless a
comment names another source. f() is ChaCha20's block function as specified in RFC 8439, 2.1 to 2.3.
H(176, ..) follows docs/format/FORMAT.md, rules R2.x, through H() of
docs/format/format_rules.py. A key is eight 32-bit words.

A second implementation of the same written rules as the programs' code; the audit cannot see a
fault that both share. Agreement tests the key schedule, the order of calls to the block function
f() and the bytes that go in and come out. This file does not count as separate from
blocks/coins/coins.h, part of the programs' code. Both implement section 2's formula for a key and
section 2a's steps from a process key to bytes and field elements. Where one of these rules is
misunderstood, the two files err alike, and comparing their values cannot show it. Nor does f()
count as separate from the programs' block functions: these all compute one published function,
and f() is adapted from the reference code that _block() names, as are the rounds of
blocks/coins/chacha.h.
No one has looked into whether this file is separate from anything else in the programs' files,
of whatever kind: for all of it the answer is unknown. own_test(), which level1.run() calls,
compares f() with RFC 8439's test vector for one block (2.3.2)."""
import struct

import numpy as np

M32 = 0xFFFFFFFF
P = {1: (1 << 61) - 1, 2: (1 << 64) - (1 << 32) + 1}
# The sixteen characters that D. J. Bernstein's chacha-merged.c names sigma, as four words: the
# first character of each is its lowest byte. The copy of that file that OpenSSH 9.6p1 carries as
# chacha.c reads them that way, with its macro U8TO32_LITTLE.
CONST = struct.unpack("<4I", b"expand 32-byte k")


def _w(s):  # four ASCII characters as one word, the first lowest
    return struct.unpack("<I", s.encode("ascii"))[0]


# Labels, each passed to f() as `a` (sections 2, 2a). CPSD: a child key of the seed (child()).
# CPLF: the draw of a batch's leaf table (draw_copy()). CPBK: the first step from a batch's first
# key to its process key (Schedule.batch_process()).
CPSD, CPLF, CPBK = _w("CPSD"), _w("CPLF"), _w("CPBK")


# The four functions below are the macros ROTATE and PLUS of D. J. Bernstein's chacha-merged.c, each
# written twice. They follow the copy that OpenSSH 9.6p1 carries as chacha.c, where ROTATE
# stands for that copy's macro ROTL32, and both cut to 32 bits with its macro U32V. Python's numbers
# have no width, so the mask M32 does for them what U32V does. In numpy's arrays of uint32 the type
# does it.
def _rotate(v, c):
    return ((v << c) & M32) | (v >> (32 - c))


def _plus(v, w):
    return (v + w) & M32


def _rotate_many(v, c):
    return (v << np.uint32(c)) | (v >> np.uint32(32 - c))


def _plus_many(v, w):
    return v + w


def _block(j, plus, rotate):
    """ChaCha20's block function on j, a list of sixteen words: twenty rounds, then j is added.

    Pass _plus and _rotate for words that are numbers, _plus_many and _rotate_many for words that
    are arrays of uint32. Returns a list of sixteen words and leaves j as it is."""

    # Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version 20080118, public domain).
    # RFC 8439, section 2.1, specifies the same function.
    def quarter_round(a, b, c, d):
        a = plus(a, b)
        d = rotate(d ^ a, 16)
        c = plus(c, d)
        b = rotate(b ^ c, 12)
        a = plus(a, b)
        d = rotate(d ^ a, 8)
        c = plus(c, d)
        b = rotate(b ^ c, 7)
        return a, b, c, d

    # As in that file, the words x0 to x15 start as j's words, and j's words are added to them last.
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = j
    # The order of the eight quarter rounds is that of the round loop in D. J. Bernstein's
    # chacha-merged.c.
    for _ in range(20, 0, -2):
        x0, x4, x8, x12 = quarter_round(x0, x4, x8, x12)
        x1, x5, x9, x13 = quarter_round(x1, x5, x9, x13)
        x2, x6, x10, x14 = quarter_round(x2, x6, x10, x14)
        x3, x7, x11, x15 = quarter_round(x3, x7, x11, x15)
        x0, x5, x10, x15 = quarter_round(x0, x5, x10, x15)
        x1, x6, x11, x12 = quarter_round(x1, x6, x11, x12)
        x2, x7, x8, x13 = quarter_round(x2, x7, x8, x13)
        x3, x4, x9, x14 = quarter_round(x3, x4, x9, x14)
    x = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
    return [plus(x[i], j[i]) for i in range(16)]


def f(K, a, b, c, d):
    """The whole block function: sixteen 32-bit words.

    a replaces the block counter; b, c, d replace the nonce."""
    if len(K) != 8:
        raise ValueError("the key must have eight 32-bit words; this one has %d" % len(K))
    # Word i of j stands where input[i] stands in the file that _block() names. OpenSSH's copy fills
    # the last four from a two-word counter and a two-word nonce; f()'s division is RFC 8439's.
    j = [0] * 16
    j[0:4] = CONST
    j[4:12] = K
    j[12:16] = a & M32, b & M32, c & M32, d & M32
    return _block(j, _plus, _rotate)


def f_many(K, a, b, c, d):
    """f() for n calls under one key: each of a .. d is a number or an array of n.

    Returns n x 16 words (uint32)."""
    xs = [np.atleast_1d(np.asarray(x, dtype=np.uint64)) for x in (a, b, c, d)]
    n = max(len(x) for x in xs)
    first = [np.full(n, w, dtype=np.uint32) for w in CONST + tuple(K)] + [np.broadcast_to((x & np.uint64(M32)).astype(np.uint32), (n,)).copy() for x in xs]
    with np.errstate(over="ignore"):
        out = np.stack(_block(first, _plus_many, _rotate_many), axis=1)
    # f() and f_many() share _block(). In rows 0 and n - 1 this compares their forms of plus and
    # rotate and the state's first twelve words. own_test() tests the rounds and the last addition.
    for i in {0, n - 1}:
        assert out[i].tolist() == f(K, *(int(first[12 + j][i]) for j in range(4))), "f_many and f differ"
    return out


# key(), take(), child(), item(): section 2.
def key(K, a, b, c, d):
    return f(K, a, b, c, d)[:8]


def words8(b):
    assert len(b) == 32
    return list(struct.unpack("<8I", b))


def take(K, w):
    assert len(w) == 8
    return key(key(K, *w[:4]), *w[4:])


def child(seed, kind, number):
    return key(seed, CPSD, kind, number, 0)


def u64(x):
    return struct.pack("<Q", x)


def item(name, data=b""):
    n = name.encode("ascii")
    return u64(len(n)) + n + u64(len(data)) + data


class Schedule:
    """The six keys of section 2's table.

    H: as format_rules.H. seed: eight words. fresh, rho: 32 bytes from the system. c: configuration
    digest; ID: commitment ID; S: a weight table's shape words; name: its root, c and S; answers:
    its record's stored answers (section 1)."""

    def __init__(self, H, seed):
        self.H, self.seed = H, seed

    def digest(self, items):
        return words8(self.H(176, 0, 0, None, b"".join(items)))

    def K(self, kind, number, fresh, items):
        return take(take(child(self.seed, kind, number), words8(fresh)), self.digest(items))

    @staticmethod
    def shape_bytes(S):
        return b"".join(u64(x) for x in S)

    def first_table(self, c, S):
        return self.K(1, 0, bytes(32), [item("a stored table's making"), item("the configuration", c), item("the shape", self.shape_bytes(S)), item("the first table")])

    def copy_table(self, k, fresh, c, S):
        return self.K(7, k, fresh, [item("a stored table's making"), item("the configuration", c), item("the shape", self.shape_bytes(S)), item("copy", u64(k))])

    def key_file(self, fresh, ID):
        return self.K(8, 0, fresh, [item("the key file's making"), item("the published ID", ID)])

    def entry(self, k, fresh, ID, name, answers=b""):
        return self.K(5, k, fresh, [item("an entry's making"), item("the published ID", ID), item("the copy's name", name), item("the copy's answers", answers)])

    def batch_first(self, rho, c, copies, first_live, slots):
        """slots: for each slot the request's words (a list of numbers below 2^32), or None."""
        it = [item("a batch"), item("the configuration", c), item("the prover's revision", u64(1)), item("copies", u64(copies)), item("the first live copy", u64(first_live)), item("slots", u64(len(slots)))]
        for s in slots:
            it.append(item("no request") if s is None else item("request", struct.pack("<%dI" % len(s), *s)))
        return self.K(6, 0, rho, it)

    def batch_process(self, first_key, name, answers=b""):
        return take(key(first_key, CPBK, 0, 0, 0), self.digest([item("the batch's stored copy"), item("the copy's name", name), item("the copy's answers", answers)]))


def draw_copy(first_key, live):
    """Section 2a: draws a batch's leaf table (`copy` in names) by rejection sampling.

    Returns its index among the `live` live ones."""
    most = (1 << 64) // live * live
    t = 0
    while True:
        o = f(first_key, CPLF, t, 0, 0)
        v = o[0] | o[1] << 32
        if v < most:
            return v % live
        t += 1


# K_ctx of section 2a.
def context_key(process, counter, version, obj, member):
    k1 = key(process, counter & M32, counter >> 32, 0, 0)
    k2 = key(k1, 0, 0, version, obj)
    return key(k2, member, 0, 0, 0)


def admit(word, field):
    """A 64-bit word as a field element, or None: a word not below p is rejected (section 2a).

    At `field` 1 (`P[1]` = 2^61 - 1) only the low 61 bits count (docs/format/FORMAT.md, R6.3,
    as element_of_word() of docs/format/format_rules.py does). The programs can be built for
    `field` 2 alone, p = 2^64 - 2^32 + 1 (blocks/base.h)."""
    if field == 1:
        word &= P[1]
        return None if word == P[1] else word
    return None if word >= P[2] else word


class Context:
    """The random values of one context, such as one request's proof (section 2a).

    Each range of bytes or elements returned is recorded under `place`, for settle()."""

    def __init__(self, process, counter, version, obj, member, field, name):
        self.k, self.field, self.name = context_key(process, counter, version, obj, member), field, name
        self.noted = {}  # stream -> list of (first, count, kind, place)

    def note(self, stream, first, count, kind, place):
        self.noted.setdefault(stream, []).append((first, count, kind, place))

    def words(self, stream, first, count):
        """64-bit words first .. first + count - 1 of a stream, at retry 0.

        A call gives four, its lanes: word i is lane i mod 4 of call i div 4."""
        c0, c1 = first // 4, (first + count - 1) // 4
        calls = np.arange(c0, c1 + 1, dtype=np.uint64)
        o = f_many(self.k, calls & np.uint64(M32), stream, 0, calls >> np.uint64(32))
        lanes = (o[:, 0:8:2].astype(np.uint64) | (o[:, 1:8:2].astype(np.uint64) << np.uint64(32))).reshape(-1)
        return lanes[first - 4 * c0:first - 4 * c0 + count]

    def bytes(self, stream, first, count, place):
        """Bytes first .. of a stream: the words as they come, lowest byte first."""
        self.note(stream, first, count, "bytes", place)
        w0, w1 = first // 8, (first + count - 1) // 8
        b = self.words(stream, w0, w1 - w0 + 1).astype("<u8").tobytes()
        return b[first - 8 * w0:first - 8 * w0 + count]

    def elements(self, stream, first, count, place):
        """Elements first .. of a stream, as a list of numbers.

        A rejected word is read again as lane 0 of its call at retry 8 k + its lane, k = 1, 2, .."""
        self.note(stream, first, count, "elements", place)
        w = self.words(stream, first, count)
        if self.field == 1:
            w = w & np.uint64(P[1])
            bad = np.nonzero(w == np.uint64(P[1]))[0]
        else:
            bad = np.nonzero(w >= np.uint64(P[2]))[0]
        out = w.tolist()
        for j in bad.tolist():
            i = first + j
            k = 1
            while True:
                o = f(self.k, (i // 4) & M32, stream, 8 * k + i % 4, (i // 4) >> 32)
                x = admit(o[0] | o[1] << 32, self.field)
                if x is not None:
                    out[j] = x
                    break
                k += 1
        return out

    def first3_many(self, streams, places):
        """Elements 0, 1, 2 of many streams at once: a list of triples."""
        if not streams: return []
        for st, pl in zip(streams, places): self.note(st, 0, 3, "elements", pl)
        o = f_many(self.k, 0, np.array(streams, dtype=np.uint64), 0, 0)
        w = o[:, 0:6:2].astype(np.uint64) | (o[:, 1:6:2].astype(np.uint64) << np.uint64(32))
        if self.field == 1:
            w = w & np.uint64(P[1]); bad = (w == np.uint64(P[1])).any(axis=1)
        else:
            bad = (w >= np.uint64(P[2])).any(axis=1)
        out = [tuple(x) for x in w.tolist()]
        for j in np.nonzero(bad)[0].tolist():
            self.noted[streams[j]].pop()
            out[j] = tuple(self.elements(streams[j], 0, 3, places[j]))
        return out

    def settle(self):
        """[(stream, place, place)] for places given overlapping ranges of one stream.

        Also lists a stream read as elements and as bytes. Overlaps within one place are not
        reported."""
        clash = []
        for stream, got in self.noted.items():
            got = sorted(set(got))
            if len({g[2] for g in got}) > 1: clash.append((stream, got[0][3], 'one stream as elements and as bytes'))
            reach = {}  # place -> its farthest end so far, to catch overlaps of non-adjacent ranges
            for first, count, kind, place in got:
                for other, end in reach.items():
                    if other != place and end > first: clash.append((stream, other, place))
                reach[place] = max(reach.get(place, 0), first + count)
        return clash


# ---- below a table's making key ----
def _be(s):
    return struct.unpack(">I", s.encode("ascii"))[0]


class TableCoins:
    """Values taken straight from a table's making key ("commit-run key" in sections 2 and 2a).

    Only salts: nothing in tools/audit/ uses `MASK` (a table's masks).
    word: how the letters MASK and SALT make a word: 'le' (the first letter lowest, as CPSD) or
    'be'. The programs use 'be' (prover/hash_commitment/hf_table_format.h); section 2 states this
    from the programs' source. The default here is 'le'."""

    def __init__(self, K, field, word="le"):
        self.K, self.field = K, field
        self.MASK, self.SALT = (_w("MASK"), _w("SALT")) if word == "le" else (_be("MASK"), _be("SALT"))


    # Tree leaf I's salt, 40 bytes.
    def salt(self, I):
        o = f(self.K, I & M32, I >> 32, 0, self.SALT)
        p = P[self.field]
        return b"".join(u64((o[2 * i] | o[2 * i + 1] << 32) % p) for i in range(5))


def own_test():
    k = [0x03020100 + 0x04040404 * i for i in range(8)]
    want = [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2]
    assert f(k, 1, 0x09000000, 0x4a000000, 0) == want, "RFC 8439, 2.3.2"
    assert f_many(k, [1, 1], 0x09000000, 0x4a000000, 0)[1].tolist() == want
    return True


if __name__ == "__main__":
    own_test()
    print("coins.py: RFC 8439's block (2.3.2): as the RFC prints it")
