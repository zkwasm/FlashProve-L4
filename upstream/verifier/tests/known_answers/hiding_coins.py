#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""A Python implementation of the committer's and prover's generator of private random values.

From a key it derives field elements (`coins`, `cell` in names), the masks of a weight table's rows
and the salts of its tree leaves. It also derives keys from a seed and a batch's 32 random bytes
(rho), but not by the programs' rule: see "keys" below. Every value is for the one field,
p = 2^64 - 2^32 + 1.

It follows RFC 8439, sections 2.1 to 2.3 (the quarter round, the block function).
It has the block function twice: block(), one call in Python's integers, line by line after the RFC, and blocks(), many
calls at once on arrays. Every call of blocks() computes its first and last block again by block() and stops if they
differ.

It imports no code of the three programs or of the tree's other Python implementations of the
generator (toy/mask_rules.py and tools/audit/coins.py among them). It is not independent of the
programs in its rules. On four of them this script rests on the same understanding of the rule as
the programs: the first call of a key's derivation, a value's label (the sixteen words that lead
from a process key to the value's block), the masks, the salts. prover/common/coin_chacha.h is one
file concerned. So a match with a program's values excludes a slip in one implementation's code. It
does not exclude a wrong reading of a rule that this script and the programs share, or an error in
the rule itself. Whether this script is independent of any file of the programs beyond
prover/common/coin_chacha.h has not been examined: of all the rest nothing is known, one way or the
other. block() follows the same specification as the programs' block functions; that it computes
the RFC's function rests only on the RFC's test vectors (2.2.1, 2.3.2) in main().

One test, which is not part of this repository, imports the part on masks (words_of(),
anchor_mask_key(), leaf_mask_key(), NO_FRESH, mask_cells(), P). The keys from rho, link_key(),
once_key(), the labels (`Key`, coins(), cell()) and leaf_salts() are called only by main() or by
nothing: nothing compares them with a file of any program. The audit's derivation,
tools/audit/coins.py, does not import this script.

A limit of every comparison by value: it cannot see a word of a label that is zero, or equal to
another word, in every run that is compared: a program that left such a word out, or put one in the
other's place, would write the same files. So the branch for a second try in mask_cells() and
mask_row() (a chance of about 2^-29 a block) has been met by no comparison.

usage: hiding_coins.py      runs the module's own known answers (RFC 8439, 2.3.2; rules on small made-up keys) and prints
one line. Exit status 0: all hold. 1: one does not. 77: python3 lacks numpy.
"""

import sys

try:
    import numpy
except ImportError:
    print("not run: python3 lacks numpy here")
    sys.exit(77)

M32 = 0xFFFFFFFF
P = 2**64 - 2**32 + 1
U32, U64 = numpy.uint32, numpy.uint64
CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)  # RFC 8439, 2.3: "expand 32-byte k"
CPSD, CPLF, CPBK = 0x44535043, 0x464C5043, 0x4B425043  # their names in ASCII, little-endian
MASK, SALT_WORD = 0x4D41534B, 0x53414C54  # "MASK", "SALT", big-endian: x3 of their blocks
RETRYQ = 8  # a cell that is p or more is drawn again from the block whose third word is 8 x (the try's number) + (its place)
SALT_CELLS = 5


class Fault(Exception):
    """A fault of this module or of its caller: never a difference between a file and the derivation."""


# ---- RFC 8439 ----
def quarter(s, a, b, c, d):
    """The quarter round that RFC 8439, section 2.1, specifies, on the words a, b, c and d of the
    state s."""
    for x, y, z, n in ((a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)):
        s[x] = (s[x] + s[y]) & M32
        t = s[z] ^ s[x]
        s[z] = (t << n | t >> 32 - n) & M32


ROUNDS = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))  # fmt: skip


def block(key, x0, x1, x2, x3):
    """f(K; x0, x1, x2, x3): 2.3 with x0 where the RFC has its block counter and x1 to x3 where it has its nonce."""
    first = [*CONSTANTS, *key, x0, x1, x2, x3]
    if len(first) != 16 or any(not 0 <= w <= M32 for w in first):
        raise Fault("block() wants a key of eight words and four words")
    s = list(first)
    for _ in range(10):
        for q in ROUNDS:
            quarter(s, *q)
    return [(a + b) & M32 for a, b in zip(s, first)]


def blocks(key, x0, x1, x2, x3):
    """The same for n calls under one key: each x is a number or an array of n numbers. Returns an array of n x 16 words."""
    xs = [numpy.atleast_1d(numpy.asarray(x, dtype=U64)) for x in (x0, x1, x2, x3)]
    n = max(len(x) for x in xs)
    if any(len(x) not in (1, n) for x in xs) or any(int(x.max()) > M32 for x in xs):
        raise Fault("blocks() wants words, and arrays of one length")
    first = [numpy.full(n, w, dtype=U32) for w in (*CONSTANTS, *key)] + [numpy.broadcast_to(x.astype(U32), (n,)).copy() for x in xs]
    s = [a.copy() for a in first]
    for _ in range(10):
        for a, b, c, d in ROUNDS:
            for x, y, z, r in ((a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)):
                s[x] += s[y]
                t = s[z] ^ s[x]
                s[z] = t << U32(r) | t >> U32(32 - r)
    out = numpy.stack([a + b for a, b in zip(s, first)], axis=1)
    for i in {0, n - 1}:
        if out[i].tolist() != block(key, *(int(first[12 + j][i]) for j in range(4))):
            raise Fault("blocks() and block() give different words")
    return out


def sixty_four(words):
    """v_sigma = o_(2 sigma) + 2^32 o_(2 sigma + 1), for an array of n x 16 words: n x 8 numbers."""
    return words[:, 0::2].astype(U64) | words[:, 1::2].astype(U64) << U64(32)


# ---- keys ----
# The anchor is a commitment's first weight table; leaf table k (`leaf` in names) is a further one.
# A mask key gives a table's masks and salts; a process key gives one run's other values (`Key`).
# anchor_mask_key(), leaf_mask_key(), once_key(), link_key(), hedged_key() and hedged_batch_key()
# are not the programs' derivations (prover/common/coin_keys.h): the first call of key() is the
# same, the words taken in after it differ. So what is derived here from a seed is not what
# `cp_commit` or `cp_prove` derives from it.
def key(k, a, b, c, d):
    return block(k, a, b, c, d)[:8]


def words_of(b):
    """A key's 32 bytes are its eight words, little-endian."""
    if len(b) != 32:
        raise Fault("a seed or rho has 32 bytes")
    return [int.from_bytes(b[4 * i : 4 * i + 4], "little") for i in range(8)]


def anchor_mask_key(seed):
    return key(seed, CPSD, 1, 0, 0)


def leaf_mask_key(seed, k, fresh):
    """The mask key for one creation of leaf table k. fresh: the eight words of the 32 bytes drawn
    for that creation, which the committer keeps (fresh_of()); zeros under a test seed."""
    if len(fresh) != 8:
        raise Fault("leaf_mask_key() wants eight words")
    return key(key(key(seed, CPSD, 7, k, 0), *fresh[:4]), *fresh[4:])


NO_FRESH = (0,) * 8


def fresh_of(isk, k):
    """The eight words of <isk>/leaf/<k>/fresh.bin."""
    with open("%s/leaf/%d/fresh.bin" % (isk, k), "rb") as f:
        return words_of(f.read())


def once_key(seed, h_0, version):
    """The process key for creating the key file. h_0: a digest of the anchor's commitment record
    (see once_key_of())."""
    if len(h_0) != 8:
        raise Fault("once_key() wants a digest of 8 words")
    return cascade(key(seed, CPSD, 8, 0, 0), [*h_0, version])


def once_key_of(seed, key_file):
    """once_key() from a key file's words: two header words, then the anchor's record: nine header
    words, of which word 1 is the version word, and four digests. h_0 is the tree hash of the record
    without its last digest. The key file's layout in blocks/frame/FORMAT.md, "key", is another."""
    from format_hash import DW, th

    record = [int(x) for x in key_file[2 : 2 + 9 + 4 * DW]]
    return once_key(seed, th(record[: 9 + 3 * DW]), record[1])


def link_key(seed, k, h_card, h_0, fresh=(0,) * 8):
    """The process key for one creation of leaf table k's LINK proof (`link` in names), which ties
    it to the anchor. h_card: a digest of the anchor's record; h_0: one of the leaf table's record;
    fresh: zeros under a test seed."""
    w = [*h_card, *h_0, *fresh]
    if len(h_card) != 8 or len(h_0) != 8 or len(fresh) != 8:
        raise Fault("link_key() wants two digests of 8 words and eight words")
    out = key(seed, CPSD, 5, k, 0)
    for i in range(0, len(w), 4):
        out = key(out, *w[i : i + 4])
    return out



def hedge_key(seed):
    return key(seed, CPSD, 6, 0, 0)


def cascade(k, words):
    """K = key(K; four words) over the words in their order, the last step filled with zero words."""
    w = list(words) + [0] * (-len(words) % 4)
    for i in range(0, len(w), 4):
        k = key(k, *w[i : i + 4])
    return k


def batch_digest(version, h_par, m, first, slots):
    """D = TH("CPHD" | the version word | h_par | M | first | n | the slots). In the hash a slot is
    1 | the request's T + 2 words, or 0 for None (the slot's file is no request). TH is the format's
    tree hash (format_hash.py)."""
    from format_hash import th

    w = [0x44485043, version, *h_par, m, first, len(slots)]
    for slot in slots:
        w += [0] if slot is None else [1, *slot]
    return th(w)


def hedged_key(seed, rho, d):
    """Kbat, from which a batch's leaf table and process key are derived. d: batch_digest()."""
    return cascade(hedge_key(seed), [*rho, *d])


def hedged_batch_key(kbat, h_card):
    """The batch's process key. h_card: the digest of the drawn leaf table's record."""
    return cascade(key(kbat, CPBK, 0, 0, 0), h_card)


def batch_leaf(kbat, m, first=0):
    """The number k of the batch's leaf table: v = words 0 (low) and 1 (high) of
    f(Kbat; "CPLF", t, 0, 0), t = 0, 1, ..; the first v that is at most 2^64 - 1 - (2^64 mod L),
    L = M - first (the live ones of M leaf tables), is taken; k = first + (v mod L)."""
    live = m - first
    rem = 2**64 % live
    t = 0
    while True:
        o = block(kbat, CPLF, t, 0, 0)
        v = o[0] + (o[1] << 32)
        if v <= 2**64 - 1 - rem:
            return first + v % live
        t += 1


# ---- the ledger ----
# Each function that takes `place`, the caller's name for where it puts the values, notes that place
# with the key's or context's name, the stream and the cells. `Key`, mask_row() and leaf_salts()
# also note the key's words. settle() and same_keys() test the rules as the calling scripts read
# them (no label serves two places), not a program's values. A ledger lives for one run of one
# script: it does not see what another script derives.
class Ledger:
    def __init__(self):
        self.notes = {}  # (key or context name, stream) -> [(first cell, one past last, place)]
        self.keys = {}  # a key's name -> the keys (eight words each) that were noted under it

    def key(self, name, words):
        """Notes a key under its name."""
        self.keys.setdefault(name, set()).add(tuple(words))

    def same_keys(self):
        """Returns a list of texts on keys that were noted under two different names."""
        named, same = {}, []
        for name, some in sorted(self.keys.items()):
            for words in sorted(some):
                if words in named:
                    same.append("one key has two names: %s and %s" % (named[words], name))
                named.setdefault(words, name)
        return same

    def note(self, whose, stream, first, n, place):
        if place is None:
            raise Fault("every draw names its place")
        if n:
            self.notes.setdefault((whose, stream), []).append((first, first + n, place))

    def settle(self):
        """Returns (the number of cells noted, texts on cells noted for two different places)."""
        cells, twice = 0, []
        for (whose, stream), runs in sorted(self.notes.items(), key=str):
            runs = sorted(set(runs))
            reach, held = 0, None
            for a, b, place in runs:
                if a < reach and place != held:
                    twice.append("%s stream %s: cells %d to %d serve %s and %s" % (whose, stream, a, min(b, reach) - 1, held, place))
                cells += max(0, b - max(a, reach))
                if b > reach:
                    reach, held = b, place
        return cells, twice


LEDGER = Ledger()


# ---- a coin ----
class Key:
    """A context under a process key. Of a label's sixteen words, 0 to 3 select a block under the
    context (cell()), 4 to 12 select the context, as in coin_context_key() in
    prover/common/coin_chacha.h, and 13 to 15 are zero. name: the process key's ledger name."""

    def __init__(self, process, name, ctr, version, obj, mem, nonce=(0, 0, 0, 0)):
        self.name = "%s, counter %d, object %d, member %d, word %d" % (name, ctr, obj, mem, version)
        LEDGER.key(name, process)
        k1 = key(process, ctr & M32, ctr >> 32, nonce[0], nonce[1])
        k2 = key(k1, nonce[2], nonce[3], version, obj)
        self.context = key(k2, mem, 0, 0, 0)


def cells_a_block(s):
    """A block gives four cells, words 0 to 7. In a stream of salts of a request's row commitments
    (numbered from 2^30, kind 2 in bits 14 and 15: stream() in prover/hiding/long_rows_words.h) it
    gives one salt, five cells."""
    return SALT_CELLS if s >= 2**30 and (s >> 14) & 3 == 2 else 4


def cell(k, s, i):
    """Cell i of a stream, in Python's integers. It notes nothing in the ledger."""
    per = cells_a_block(s)
    b, q = divmod(i, per)
    o = block(k.context, b & M32, s, 0, b >> 32)
    c, t = o[2 * q] + (o[2 * q + 1] << 32), 0
    while c >= P:
        t += 1
        o = block(k.context, b & M32, s, RETRYQ * t + q, b >> 32)
        c = o[0] + (o[1] << 32)
    return c


def coins(k, s, n, first=0, place=None, array=False):
    """The n cells of a stream from the cell numbered first on."""
    LEDGER.note(k.name, s, first, n, place)
    if not n:
        return numpy.zeros(0, dtype=U64) if array else []
    per = cells_a_block(s)
    b = numpy.arange(first // per, -(-(first + n) // per), dtype=U64)
    v = sixty_four(blocks(k.context, b & U64(M32), s, 0, b >> U64(32)))[:, :per]
    out = v.reshape(-1)[first % per :][:n].copy()
    for i in numpy.nonzero(out >= U64(P))[0].tolist():
        out[i] = cell(k, s, first + i)
    if any(int(out[i]) != cell(k, s, first + i) for i in (0, n // 2, n - 1)):
        raise Fault("coins() and cell() give different cells in stream %d" % s)
    return out if array else out.tolist()


def raw_words(k, s, n, first=0, place=None):
    """Whole blocks, for callers whose rule divides a block differently. Returns the n blocks from
    block first on, n x 16 words, and notes four cells for each."""
    LEDGER.note(k.name, s, 4 * first, 4 * n, place)
    b = numpy.arange(first, first + n, dtype=U64)
    return blocks(k.context, b & U64(M32), s, 0, b >> U64(32))


# ---- masks and salts of a table ----
# The blocks' input words are those of hf_mask8() and hf_salt5() in
# prover/hash_commitment/hf_mask.cuh.
def mask_row(km, name, r, nm, place=None):
    """The nm masks of row r: coefficient 8 g + sigma is v_sigma mod p of f(kM; r, g, try, "MASK") at the least try at which
    all eight v are below p, the greatest multiple of p that fits into 64 bits."""
    LEDGER.key(name, km)
    LEDGER.note(name, "MASK row %d" % r, 0, nm, place)
    g = numpy.arange(-(-nm // 8), dtype=U64)
    v = sixty_four(blocks(km, r, g, 0, MASK))
    for i in numpy.nonzero((v >= U64(P)).any(axis=1))[0].tolist():
        t = 1
        while True:
            o = block(km, r, i, t, MASK)
            w = [o[2 * j] + (o[2 * j + 1] << 32) for j in range(8)]
            if max(w) < P:
                break
            t += 1
        v[i] = w
    return (v % U64(P)).reshape(-1)[:nm]


def mask_cells(km, name, rows, first, n, nm, place=None):
    """The masks first .. first + n - 1 of each of these rows, by mask_row()'s rule: an array of
    len(rows) x n."""
    if not rows or not 0 <= first < first + n <= nm:
        raise Fault("mask_cells() wants rows, and a window inside a row's masks")
    for r in rows:
        LEDGER.note(name, "MASK row %d" % r, first, n, place)
    g0, g1 = first // 8, -(-(first + n) // 8)
    r = numpy.repeat(numpy.asarray(rows, dtype=U64), g1 - g0)
    g = numpy.tile(numpy.arange(g0, g1, dtype=U64), len(rows))
    v = sixty_four(blocks(km, r, g, 0, MASK))
    for i in numpy.nonzero((v >= U64(P)).any(axis=1))[0].tolist():
        t = 1
        while True:
            o = block(km, int(r[i]), int(g[i]), t, MASK)
            w = [o[2 * j] + (o[2 * j + 1] << 32) for j in range(8)]
            if max(w) < P:
                break
            t += 1
        v[i] = w
    return (v % U64(P)).reshape(len(rows), -1)[:, first - 8 * g0 :][:, :n]


def leaf_salts(km, name, places, place=None):
    """The salts of the tree leaves I in places: five symbols, v_sigma mod p for sigma < 5 of
    f(kM; I mod 2^32, I div 2^32, 0, "SALT"). An array of n x 5."""
    LEDGER.key(name, km)
    at = numpy.asarray(places, dtype=U64)
    for i in at.tolist() if len(at) < 4096 else ():
        LEDGER.note(name, "SALT", i, 1, place)
    if len(at) >= 4096:
        if at.tolist() != list(range(int(at[0]), int(at[0]) + len(at))):
            raise Fault("a long list of leaves is a run")
        LEDGER.note(name, "SALT", int(at[0]), len(at), place)
    return sixty_four(blocks(km, at & U64(M32), at >> U64(32), 0, SALT_WORD))[:, :SALT_CELLS] % U64(P)


def main():
    bad = []
    want = "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
    got = b"".join(w.to_bytes(4, "little") for w in block(words_of(bytes(range(32))), 1, 0x09000000, 0x4A000000, 0)).hex()
    if got != want:
        bad.append("RFC 8439, 2.3.2")
    s = [0x879531E0, 0xC5ECF37D, 0x516461B1, 0xC9A62F8A, 0x44C20EF3, 0x3390AF7F, 0xD9FC690B, 0x2A5F714C,
         0x53372767, 0xB00A5631, 0x974C541A, 0x359E9963, 0x5C971061, 0x3D631689, 0x2098D9D6, 0x91DBD320]  # fmt: skip
    quarter(s, 2, 7, 8, 13)
    if (s[2], s[7], s[8], s[13]) != (0xBDB886DC, 0xCFACAFD2, 0xE46BEA80, 0xCCC07C79):
        bad.append("RFC 8439, 2.2.1")
    k = Key(list(range(1, 9)), "a made-up key", 2**33 + 1, 7, 1, 3)
    for st in (4 * 4096, 2**30 + 5 * 2**16 + 2 * 2**14):
        some = coins(k, st, 23, 2**34 + 3, place="here")
        if some != [cell(k, st, 2**34 + 3 + i) for i in range(23)]:
            bad.append("coins() against cell() in stream %d" % st)
    coins(k, 4 * 4096, 4, 2**34 + 20, place="there")
    if len(LEDGER.settle()[1]) != 1:
        bad.append("the ledger does not see cells that serve two places")
    whole = [mask_row(list(range(2, 10)), "a made-up mask key", r, 61, place="a row") for r in (0, 5, 2**31 + 7)]
    some = mask_cells(list(range(2, 10)), "a made-up mask key", [0, 5, 2**31 + 7], 13, 41, 61, place="a row")
    if some.shape != (3, 41) or any(some[i].tolist() != whole[i][13:54].tolist() for i in range(3)):
        bad.append("mask_cells() against mask_row()")
    if batch_leaf(list(range(8)), 1) != 0 or not 2 <= batch_leaf(list(range(8)), 7, 2) < 7:
        bad.append("batch_leaf()")

    def accepted(call, digest):
        try:
            call(digest)
        except Fault:
            return False
        return True

    eight = list(range(8))
    for name, call in (
        ("once_key()", lambda d: once_key(eight, d, 7)),
        ("link_key()", lambda d: link_key(eight, 0, d, d)),
        ("link_key(), as h_card alone,", lambda d: link_key(eight, 0, d, eight)),
        ("link_key(), as h_0 alone,", lambda d: link_key(eight, 0, eight, d)),
    ):
        if not accepted(call, eight) or accepted(call, list(range(10))):
            bad.append(name + " must accept a digest of 8 words and reject one of 10")
    print("hiding_coins.py: %d of its own known answers do not hold%s" % (len(bad), ": " + "; ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
