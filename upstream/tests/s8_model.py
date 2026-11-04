#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The hash rules of the proof of work of a transcript that no program has and of the folded leaf, as a model that
shares no code with the programs, except that BLAKE2b's compression function F has the same outside
source as theirs (the comment above rotr64): a comparison with them can show an error made in
adapting that source, not an error of the source.

But for F, written from texts alone.
C and H are defined in verifier/HASHES.md, "F and C" and "H(tag; x)".
No program has this transcript: its absorb (hash number 3) and its test of a nonce (127) stand in no
program's source: this model states the rules. verifier/HASHES.md tells that test, in "The numbers that the sources pass".
The leaf's calls have the hash numbers 103, 104 and 133 (`LEAF_CHAIN`, `LEAF_END` and `LEAF_FOLD` in
prover/hiding/long_rows_words.h). That header defines these calls; verifier/HASHES.md does not.
The proof files' format has neither rule: see docs/format/FORMAT.md, R5.5 (work), R7.4 (leaf).

  s8_model.py              test plain BLAKE2b against hashlib, then print tests/s8_known_answers.tsv
  s8_model.py -write FILE  the same, into the new file FILE

As a module: H(), test_value(), passes(), least_nonce(), failing_nonce(), absorb(), leaf().
"""
import hashlib
import struct
import sys

M64 = (1 << 64) - 1
P61 = (1 << 61) - 1
# BLAKE2b's compression function F, which RFC 7693, section 3.2, specifies. The text from rotr64 to the end of F is adapted from the
# BLAKE2 team's reference implementation in C. The head of its file blake2b-ref.c, which its two headers have too:
# /*
#    BLAKE2 reference source code package - reference C implementations
#
#    Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
#    terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
#    your option.  The terms of these licenses can be found at:
#
#    - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
#    - OpenSSL license   : https://www.openssl.org/source/license.html
#    - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
#
#    More information about the BLAKE2 hash function can be found at
#    https://blake2.net.
# */
# Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
# source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
# It is a translation into Python, under the reference's names: rotr64, blake2b_IV, blake2b_sigma, blake2b_init_param with what
# blake2b_init does before it calls it, the macros G and ROUND, and blake2b_compress, which is F here.
# What differs from the reference:
# - A Python integer has no fixed width, so every sum and every rotation is cut to 64 bits with M64.
# - G and ROUND are functions, not macros: G also takes m and returns the four new words, and ROUND also takes v and m and
#   writes into v. The twelve lines ROUND( 0 ) to ROUND( 11 ) are a loop.
# - F has no state structure S, no buffer and no load64. What the reference keeps in S, and the block, are arguments: h is
#   S->h, t0 is S->t[0], t1 is S->t[1], last is whether the last-block flag S->f[0] is set, m is the block as sixteen words.
#   The arguments are not written into, and the new chaining value is returned.
# - In the reference S->t[1] is the upper half of a count of bytes; here t1 is any word, and C (below) puts the format's
#   additional words there. S->f[1], which marks a last node in tree hashing, is always 0 here.
# - blake2b_init_param takes the two lengths, not a filled parameter block, and returns S->h.


def rotr64(w, c):
    return (w >> c | w << (64 - c)) & M64


blake2b_IV = (
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
)  # fmt: skip

blake2b_sigma = (
    ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15),
    (14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3),
    (11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4),
    ( 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8),
    ( 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13),
    ( 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9),
    (12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11),
    (13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10),
    ( 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5),
    (10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0),
    ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15),
    (14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3),
)  # fmt: skip


def blake2b_init_param(outlen, keylen):
    """Returns the initial chaining value, the reference's S->h after blake2b_init (keylen = 0) or
    blake2b_init_key: blake2b_IV XOR the parameter block P, read as little-endian words (load64)."""
    # digest_length, key_length, fanout, depth; all later fields are 0
    P = bytes([outlen, keylen, 1, 1]) + bytes(60)
    return [blake2b_IV[i] ^ int.from_bytes(P[8 * i : 8 * i + 8], "little") for i in range(8)]


def G(r, i, a, b, c, d, m):
    a = (a + b + m[blake2b_sigma[r][2 * i + 0]]) & M64
    d = rotr64(d ^ a, 32)
    c = (c + d) & M64
    b = rotr64(b ^ c, 24)
    a = (a + b + m[blake2b_sigma[r][2 * i + 1]]) & M64
    d = rotr64(d ^ a, 16)
    c = (c + d) & M64
    b = rotr64(b ^ c, 63)
    return a, b, c, d


def ROUND(r, v, m):
    v[0], v[4], v[8], v[12] = G(r, 0, v[0], v[4], v[8], v[12], m)
    v[1], v[5], v[9], v[13] = G(r, 1, v[1], v[5], v[9], v[13], m)
    v[2], v[6], v[10], v[14] = G(r, 2, v[2], v[6], v[10], v[14], m)
    v[3], v[7], v[11], v[15] = G(r, 3, v[3], v[7], v[11], v[15], m)
    v[0], v[5], v[10], v[15] = G(r, 4, v[0], v[5], v[10], v[15], m)
    v[1], v[6], v[11], v[12] = G(r, 5, v[1], v[6], v[11], v[12], m)
    v[2], v[7], v[8], v[13] = G(r, 6, v[2], v[7], v[8], v[13], m)
    v[3], v[4], v[9], v[14] = G(r, 7, v[3], v[4], v[9], v[14], m)


def F(h, m, t0, t1, last):
    """BLAKE2b's compression function (RFC 7693, section 3.2; its IV: section 2.6, SIGMA: 2.7): eight long words, sixteen long words, two
    counters, a flag."""
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]
    v[8] = blake2b_IV[0]
    v[9] = blake2b_IV[1]
    v[10] = blake2b_IV[2]
    v[11] = blake2b_IV[3]
    v[12] = blake2b_IV[4] ^ t0  # S->t[0]
    v[13] = blake2b_IV[5] ^ t1  # S->t[1]

    # S->f[0]: blake2b_set_lastblock sets it to (uint64_t)-1
    v[14] = blake2b_IV[6] ^ (M64 if last else 0)
    v[15] = blake2b_IV[7]  # S->f[1] is 0
    for r in range(12):
        ROUND(r, v, m)
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def longs(words):
    return [words[2 * i] | (words[2 * i + 1] << 32) for i in range(len(words) // 2)]


def words_of(ls):
    out = []
    for x in ls:
        out += [x & 0xffffffff, x >> 32]
    return out


IVP = words_of(blake2b_init_param(64, 0))     # IV', the state BLAKE2b-512 starts from


def C(cv, block, c, l, f, last):
    """One call: state (16 words), block (32), counter, length, hash number, last-block flag."""
    assert len(cv) == 16 and len(block) == 32
    return words_of(F(longs(cv), longs(block), c, f | (l << 32), last))


def bare(cv, block, c0, c1, f):
    return C(cv, block, c0 | (c1 << 32), 0, f, False)


def H(tag, x, t1_zero=False):
    """The state (16 words) of H(tag; x). t1_zero makes it plain BLAKE2b-512, for the self-test."""
    L = len(x)
    nb = max(1, (L + 31) // 32)
    cv = IVP
    for k in range(nb):
        blk = list(x[32 * k:32 * k + 32])
        blk += [0] * (32 - len(blk))
        lastb = k == nb - 1
        cnt = 4 * L if lastb else 128 * (k + 1)
        cv = C(cv, blk, cnt, 0 if t1_zero else L, 0 if t1_zero else tag, lastb)
    return cv


# ---- the proof of work (`work` in the table) -------------------------------------------------

POW_NUMBER = 127


def test_value(dig, n):
    """d = H(127; dig(s), n_lo, n_hi), all 16 words. dig(s): the digest of the state s."""
    return H(POW_NUMBER, list(dig) + [n & 0xffffffff, n >> 32])


def passes(dig, n, g):
    return (test_value(dig, n)[0] & ((1 << g) - 1)) == 0


def least_nonce(dig, g, start=0):
    n = start
    while not passes(dig, n, g):
        n += 1
    return n


def failing_nonce(dig, g, start=0):
    assert g > 0
    n = start
    while passes(dig, n, g):
        n += 1
    return n


def absorb(dig, w):
    """The state after an absorb of the list w: H(3; dig(s), w)."""
    return H(3, list(dig) + list(w))


# ---- the leaf of a request's row commitment ---------------------------------------------------

def sym(x):
    return [x & 0xffffffff, x >> 32]


def leaf(symbols, salt, ln, NUM, I, lsalt, DW):
    """symbols: the Rr real symbols of column I; salt: five symbols. Returns (digest, calls, folded).
    ln: log2 of the code length; NUM: the tree's number; DW: words in a digest.
    With lsalt = 1, a leaf is folded if at most 9 symbols fall into its last block of 16: the salt
    shares that block (hash number 133). Otherwise the salt goes into a separate call (104).
    The word f of every call is the hash number + 2^16 NUM."""
    Rr = len(symbols)
    assert Rr >= 1 and len(salt) == 5
    R = (Rr + 7) // 8 * 8
    sl = (R + 15) // 16
    rl = Rr - 16 * (sl - 1)
    assert 1 <= rl <= 16
    folded = lsalt == 1 and rl <= 9
    col = list(symbols) + [0] * (16 * sl - Rr)
    saltw = [w for s in salt for w in sym(s)]
    st, tree = IVP, NUM << 16
    for b in range(sl - 1 if folded else sl):
        st = bare(st, [w for s in col[16 * b:16 * b + 16] for w in sym(s)], b, 0, 103 + tree)
    if folded:
        B = [w for s in symbols[16 * (sl - 1):] for w in sym(s)]
        B += [0] * (18 - len(B)) + saltw + [R, ln, NUM, I]
        return bare(st, B, sl - 1, 0, 133 + tree)[:DW], sl, True
    B = saltw + [R, ln, NUM, I] + [0] * 18
    return bare(st, B, sl, 0, 104 + tree)[:DW], sl + 1, False


# ---- the inputs of the known answers ----------------------------------------------------------

def ka_state(DW):
    return [0x01010101 * (i + 1) for i in range(DW)]


KA_LN, KA_NUM, KA_I = 20, 7, 5


# `P61` (2^61 - 1) is not the prime of the programs' field, 2^64 - 2^32 + 1. It is only the modulus
# of these symbols (leaf() does no field arithmetic), so no known answer has one of 2^61 or more.
def ka_symbol(r):
    return (r + 1) * 0x0123456789ABCDF % P61


def ka_salt(j):
    return (j + 1) * 0x0FEDCBA98765432 % P61


def hexs(ws):
    return " ".join("%08x" % w for w in ws)


def self_test():
    for L in (0, 1, 8, 31, 32, 33, 64, 100):
        x = [(0x9e3779b9 * (i + 1)) & 0xffffffff for i in range(L)]
        want = hashlib.blake2b(struct.pack("<%dI" % L, *x)).digest()
        got = struct.pack("<16I", *H(0, x, t1_zero=True))
        assert got == want, L


def table():
    out = ["# Known answers of the proof of work of a transcript that no program has and of the folded leaf, printed by tests/s8_model.py (a model in Python; it imports no code of the programs).",
           "# Columns are tab-separated; words are hexadecimal, eight digits, in the order of the list.",
           "# work: dig(s) = words 0x01010101 * (i + 1), i = 0 .. DW - 1. least = the least nonce that passes at g bits; next = the next one;",
           "#   d = H(127; dig(s), n_lo, n_hi) at the least nonce, words 0 .. DW - 1; s1 = H(3; dig(s), n_lo, n_hi) at the least nonce, all 16 words.",
           "# nonce: the same dig(s) and a nonce with n_hi not 0; g = 8: it passes at 8 bits; g = 0: bit 0 of d's word 0 is 1, so it fails at every g above 0. d and s1 as above.",
           "# leaf: symbol of row r = (r + 1) * 0x0123456789ABCDF mod p, salt symbol j = (j + 1) * 0x0FEDCBA98765432 mod p, p = 2^61 - 1;",
           "#   ln = %d, NUM = %d, I = %d. calls = compression calls of this leaf." % (KA_LN, KA_NUM, KA_I),
           "kind\tDW\tg\tleast\tnext\td\ts1"]
    for DW in (8, 10):
        dig = ka_state(DW)
        for g in (1, 3, 4, 5, 8, 12, 16):
            n = least_nonce(dig, g)
            n2 = least_nonce(dig, g, n + 1)
            out.append("work\t%d\t%d\t%d\t%d\t%s\t%s" % (
                DW, g, n, n2, hexs(test_value(dig, n)[:DW]), hexs(absorb(dig, [n & 0xffffffff, n >> 32]))))
    out.append("kind\tDW\tg\tn_lo\tn_hi\td\ts1")
    for DW in (8, 10):  # nonces with n_hi not 0: the least from 2^32 on that passes at 8 bits, and the least that fails at 1 bit
        dig = ka_state(DW)
        for g, x in ((8, least_nonce(dig, 8, 1 << 32)), (0, failing_nonce(dig, 1, 1 << 32))):
            out.append("nonce\t%d\t%d\t%d\t%d\t%s\t%s" % (DW, g, x & 0xffffffff, x >> 32, hexs(test_value(dig, x)[:DW]), hexs(absorb(dig, [x & 0xffffffff, x >> 32]))))
    out.append("kind\tDW\tlsalt\tRr\tR\tfolded\tcalls\tleaf")
    for DW in (8, 10):
        for lsalt in (0, 1):
            for Rr in (1, 8, 9, 10, 16, 17, 25, 26):
                d, calls, fo = leaf([ka_symbol(r) for r in range(Rr)], [ka_salt(j) for j in range(5)],
                                    KA_LN, KA_NUM, KA_I, lsalt, DW)
                out.append("leaf\t%d\t%d\t%d\t%d\t%d\t%d\t%s" % (DW, lsalt, Rr, (Rr + 7) // 8 * 8, int(fo), calls, hexs(d)))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    self_test()
    t = table()
    if len(sys.argv) == 3 and sys.argv[1] == "-write":
        open(sys.argv[2], "x").write(t)  # fails if the name is taken: nothing is overwritten
    else:
        sys.stdout.write(t)
