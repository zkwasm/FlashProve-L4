#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""BLAKE2b's compression function F (RFC 7693).

f() is F, and blake2b() is BLAKE2b built on f(). format_hash.py, in this directory, chains f()
block by block itself; hashlib does not expose F.

    blake2b_f.py            the self-test: checks the RFC's values for "abc" (Appendix A: the work
                            vector at i = 0, 1 and 12, the digest), then compares blake2b() with
                            hashlib.blake2b, with and without a key. Exits with code 1 if anything
                            differs
    blake2b_f.py <n> <seed> made_up(): prints n lines "h m t last -> f(h, m, t, last)" in hex, on
                            pseudorandom arguments, to compare other implementations of F with f()

In the self-test t is always below 2**64, so it does not test the line of f() that takes in the high
half of t. format_hash.py's check of verifier/tests/hash_vectors.txt does, because there that half
is never 0.

The RFC is M-J. Saarinen, J-P. Aumasson: "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", November
2015; https://www.rfc-editor.org/rfc/rfc7693. The sections that describe what is here are 2.1 (w = 64, r = 12, bb = 128;
the rotations 32, 24, 16, 63), 2.6 (IV), 2.7 (SIGMA), 3.1 (G), 3.2 (F) and 3.3 (BLAKE2). The text is adapted from the
BLAKE2 reference implementation:
    /*
       BLAKE2 reference source code package - reference C implementations

       Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
       terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
       your option.  The terms of these licenses can be found at:

       - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
       - OpenSSL license   : https://www.openssl.org/source/license.html
       - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0

       More information about the BLAKE2 hash function can be found at
       https://blake2.net.
    */
The fourteen indented lines are the head of blake2b-ref.c; the two headers that it includes,
blake2.h and blake2-impl.h, begin with the same lines. The comment above M64 says what is adapted
and all that differs.
This module imports only Python's standard library. The programs' versions of F (verifier/src/vf_earlier_hash_f.h,
blocks/hash/blake2b_f.h, prover/common/blake2b_f.h) are adapted from the same reference
implementation, so a comparison with f() would not show that implementation's errors. The other
Python versions of F in this tree (such as tests/s8_model.py, scales/policy_tables.py and
docs/format/format_rules.py) are the same translation under other names, so a comparison
among them can show only an error made in copying.
"""

import struct
import sys

# Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
# source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
# From here to the end of f, and the body of blake2b: a translation into Python, in the order of
# blake2b-ref.c and under its names: two of the three constants (M64 is ours), rotr64, blake2b_IV
# and blake2b_sigma, blake2b_init_param with what blake2b_init and blake2b_init_key do before they
# call it, the macros G and ROUND, blake2b_compress, which is f here, and blake2b with
# blake2b_update and blake2b_final.
# What differs from the reference:
# - A Python integer has no fixed width, so every sum and every rotation is cut to 64 bits with M64.
# - G and ROUND are macros that read m and write v in the function around them. Here they are
#   functions: G also takes m and returns the four new words, and ROUND also takes v and m and
#   writes into v. The twelve lines ROUND( 0 ) to ROUND( 11 ) are a loop.
# - f has no state structure S, no buffer and no load64. What the reference keeps in S, and the
#   block, are arguments: h is the chaining value S->h; t is the whole counter, S->t[0] + 2^64
#   S->t[1]; last is whether the last-block flag S->f[0] is set; m is the block as sixteen words. So
#   a caller can do the chaining itself (format_hash.py does). The arguments are not written into,
#   and the new chaining value is returned.
# - S->f[1], which marks a last node in tree hashing, is always 0 here.
# - The high half of t, the reference's S->t[1], is 0 in plain BLAKE2b for every input below 2^64
#   bytes. format_hash.py puts the format's additional words there, and made_up() sets all 128 bits.
# - The list seen is ours: the reference does not expose its array v.
# - IV (a second name of blake2b_IV, for the modules that import it), START_512 and the assert in f
#   are ours too.
# - blake2b_init_param takes the two lengths, not a filled parameter block, and returns S->h.
# - The comment in blake2b says what differs there.

M64 = 2**64 - 1
BLAKE2B_BLOCKBYTES = 128
BLAKE2B_OUTBYTES = 64


def rotr64(w, c):
    return (w >> c | w << (64 - c)) & M64


blake2b_IV = (
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
)  # fmt: skip
IV = blake2b_IV

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


# BLAKE2b-512's initial chaining value, without a key.
START_512 = blake2b_init_param(BLAKE2B_OUTBYTES, 0)


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


def f(h, m, t, last, seen=None):
    """The compression function F, specified in RFC 7693, section 3.2. Takes the chaining value h
    (eight 64-bit words) and one block m (sixteen 64-bit words); returns the new chaining value as a
    list.

    In BLAKE2b t, below 2**128, is the number of bytes passed to f() so far, including this block.
    All 128 bytes of a key's block count; the zeros after the end of the message do not. `last` (the RFC's
    argument f) is true only for the last block. f() checks t only for its range and does not check
    `last`, so made_up() can pass pseudorandom values. `seen`, if given, is a list to which f()
    appends a copy of the array v before each round and after the last round (the RFC's Appendix A
    prints these)."""
    assert (
        len(h) == 8 and len(m) == 16 and 0 <= t < 2**128 and all(0 <= x < 2**64 for x in (*h, *m))
    )
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]
    v[8] = blake2b_IV[0]
    v[9] = blake2b_IV[1]
    v[10] = blake2b_IV[2]
    v[11] = blake2b_IV[3]
    v[12] = blake2b_IV[4] ^ (t & M64)  # S->t[0]
    v[13] = blake2b_IV[5] ^ (t >> 64)  # S->t[1]

    # S->f[0]: blake2b_set_lastblock sets it to (uint64_t)-1
    v[14] = blake2b_IV[6] ^ (M64 if last else 0)
    v[15] = blake2b_IV[7]  # S->f[1] is 0
    for r in range(12):
        if seen is not None:
            seen.append(list(v))
        ROUND(r, v, m)
    if seen is not None:
        seen.append(list(v))
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def words(block):
    """A block of 128 bytes as sixteen 64-bit words, each little-endian (specified in RFC 7693,
    section 2.4)."""
    return [int.from_bytes(block[8 * i : 8 * i + 8], "little") for i in range(16)]


def blake2b(data, nn=64, key=b"", seen=None):
    """BLAKE2b, specified in RFC 7693, section 3.3: returns the nn-byte digest of data, with an
    optional key. Only the self-test and made_up() call it. `seen` goes to the last call of f()."""
    # Adapted from blake2b, blake2b_init, blake2b_init_key, blake2b_update and blake2b_final of
    # blake2b-ref.c. What differs: the whole message is at hand, so there is no streaming state
    # (no S->buf, no S->buflen); h is S->h and t is S->t as one integer, whose carry does what
    # blake2b_increment_counter does for t[1]; the checks of the arguments and secure_zero_memory
    # are left out.
    h = blake2b_init_param(nn, len(key))
    t = 0
    # blake2b_init_key: the key, zero-padded to one block, is hashed before the message.
    if len(key) > 0:
        data = key + bytes(BLAKE2B_BLOCKBYTES - len(key)) + data
    # blake2b_update: a block is compressed only when more bytes follow it.
    at, inlen = 0, len(data)
    while inlen > BLAKE2B_BLOCKBYTES:
        t += BLAKE2B_BLOCKBYTES  # blake2b_increment_counter
        h = f(h, words(data[at : at + BLAKE2B_BLOCKBYTES]), t, False)
        at += BLAKE2B_BLOCKBYTES
        inlen -= BLAKE2B_BLOCKBYTES
    # blake2b_final: the remaining 0 to 128 bytes, zero-padded, with the last-block flag set.
    t += inlen
    h = f(h, words(data[at:] + bytes(BLAKE2B_BLOCKBYTES - inlen)), t, True, seen)
    buffer = b"".join(x.to_bytes(8, "little") for x in h)  # store64
    return buffer[:nn]


def list_bytes(count, skip=0):
    """Returns count bytes of a fixed test pattern: the 32-bit words ((k + 1) * 0x89ABCDEF) mod 2^32
    for k = skip, skip + 1, ..., each little-endian."""
    n = -(-count // 4)
    return struct.pack(
        "<%dI" % n, *((k + 1) * 0x89ABCDEF % 2**32 for k in range(skip, skip + n))
    )[:count]


# Test values that RFC 7693 prints in its Appendix A, for BLAKE2b-512 of "abc": the work vector at i = 0, 1 and 12, and the digest.
A_V0 = """6A09E667F2BDC948 BB67AE8584CAA73B 3C6EF372FE94F82B A54FF53A5F1D36F1 510E527FADE682D1 9B05688C2B3E6C1F 1F83D9ABFB41BD6B
5BE0CD19137E2179 6A09E667F3BCC908 BB67AE8584CAA73B 3C6EF372FE94F82B A54FF53A5F1D36F1 510E527FADE682D2 9B05688C2B3E6C1F
E07C265404BE4294 5BE0CD19137E2179"""
A_V1 = """86B7C1568029BB79 C12CBCC809FF59F3 C6A5214CC0EACA8E 0C87CD524C14CC5D 44EE6039BD86A9F7 A447C850AA694A7E DE080F1BB1C0F84B
595CB8A9A1ACA66C BEC3AE837EAC4887 6267FC79DF9D6AD1 FA87B01273FA6DBE 521A715C63E08D8A E02D0975B8D37A83 1C7B754F08B7D193
8F885A76B6E578FE 2318A24E2140FC64"""
A_V12 = """12EF8A641EC4F6D6 BCED5DE977C9FAF5 733CA476C5148639 97DF596B0610F6FC F42C16519AD5AFA7 AA5AC1888E10467E 217D930AA51787F3
906A6FF19E573942 75AB709BD3DCBF24 EE7CE1F345947AA4 F8960D6C2FAF5F5E E332538A36B6D246 885BEF040EF6AA0B A4939A417BFB78A3
646CBB7AF6DCE980 E813A23C60AF3B82"""
A_DIGEST = """BA 80 A5 3F 98 1C 4D 0D 6A 27 97 B6 9F 12 F6 E9 4C 21 2F 14 68 5A C4 B7 4B 12 BB 6F DB FF A2 D1
7D 87 C5 39 2A AB 79 2D C2 52 D5 DE 45 33 CC 95 18 D3 8A A8 DB F1 92 5A B9 23 86 ED D4 00 99 23"""


# The self-test: returns 1 if any check fails.
def anchors():
    bad = 0

    def check(good, what):
        nonlocal bad
        bad += not good
        print("%-9s %s" % ("matches" if good else "differs", what))

    import hashlib

    abc = bytes.fromhex(A_DIGEST.replace("\n", " "))
    seen = []
    digest = blake2b(b"abc", seen=seen)
    for i, page in ((0, A_V0), (1, A_V1), (12, A_V12)):
        check(seen[i] == [int(x, 16) for x in page.split()], "Appendix A: the work vector at i = %d" % i)
    check(digest == abc, 'Appendix A: BLAKE2b-512("abc")')
    check(hashlib.blake2b(b"abc").digest() == abc, 'hashlib.blake2b gives the same value for "abc"')
    # The inputs and the key of the reference implementation's known-answer tests.
    ramp = bytes(range(256))
    same = all(
        blake2b(ramp[:n], 64, key) == hashlib.blake2b(ramp[:n], key=key).digest()
        for n in range(256)
        for key in (b"", ramp[:64])
    )
    check(
        same,
        "256 inputs of 0 to 255 bytes (byte i is i), each without a key and under the key 00 01 .."
        " 3f: equal to Python's hashlib.blake2b",
    )
    # Lengths at the edges of a block (128 bytes), of a digest and of a key (64 bytes each), and some between.
    cases = [
        (n, nn, kk)
        for n in (0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000)
        for nn in (1, 31, 32, 33, 40, 63, 64)
        for kk in (0, 1, 31, 32, 63, 64)
    ]
    same = all(
        blake2b(list_bytes(n), nn, list_bytes(kk, 300))
        == hashlib.blake2b(list_bytes(n), digest_size=nn, key=list_bytes(kk, 300)).digest()
        for n, nn, kk in cases
    )  # fmt: skip
    check(
        same,
        "%d further inputs, with and without a key: equal to Python's hashlib.blake2b" % len(cases),
    )
    print("all as expected" if not bad else "%d differ" % bad)
    return 1 if bad else 0


def made_up(n, seed):
    """Prints n calls of f(), one per line, on arguments that blake2b() derives from the seed."""
    for i in range(n):
        # 256 bytes: h 64, m 128, t 16, last 1; the other 47 are unused
        raw = b"".join(blake2b(b"%d %d %d" % (seed, i, j)) for j in range(4))
        h, m = (
            [int.from_bytes(raw[8 * k : 8 * k + 8], "little") for k in range(8)],
            words(raw[64:192]),
        )
        t, last = int.from_bytes(raw[192:208], "little"), raw[208] & 1
        out = f(h, m, t, bool(last))
        print(
            "%s %s %032x %d -> %s"
            % (
                "".join("%016x" % x for x in h),
                "".join("%016x" % x for x in m),
                t,
                last,
                "".join("%016x" % x for x in out),
            )
        )  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) == 3:
        made_up(int(sys.argv[1]), int(sys.argv[2]))
    else:
        sys.exit(anchors())
