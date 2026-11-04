# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The five public tables of the integer model, and the digests that pin them.

The model reads two rotary tables, a softmax table, a SiLU table and a sigmoid table.
docs/model.md, "4 Public tables", defines each entry as a fixed sequence of float64 operations; this
module carries them out with Python's floats and the math module. The names C, S, EX, u0, Ts, Sig
and rha below are that section's.

The result could depend on the platform's math library, so the parameter file (`policy.txt`) has
four digests: one for the two rotary tables together and one for each other table. make_tables()
raises ParameterError if a table computed here has a different digest. After that check the tables
equal those from which the parameter file's digests were made, and the model only indexes them.
"""

import math
import struct
from dataclasses import dataclass

import torch

from .parameters import ParameterError, Parameters

# The SiLU and sigmoid tables have signed keys. A table over the 16-bit keys -32,768 .. 32,767
# has key c's entry at index c + KEY_OFFSET_16, and one over the 15-bit keys -16,384 .. 16,383 at
# index c + KEY_OFFSET_15.
KEY_OFFSET_16 = 1 << 15
KEY_OFFSET_15 = 1 << 14


def _round_half_away(v: float) -> int:
    """rha(v): floor(v + 0.5) for v >= 0, else -floor(-v + 0.5), with a float64 addition: ties go
    away from zero."""
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def rotary_tables(T: int, dh: int, cs: int, theta0: int) -> tuple[list[int], list[int]]:
    """C(t, i) and S(t, i) for t < T and i < dh / 2. Entry (t, i) is at index t * (dh / 2) + i."""
    # The angles follow modeling_llama4.py and configuration_llama4.py of Hugging Face transformers
    # (NOTICE).
    angles = [
        float(t) * math.pow(float(theta0), (-2.0 * i) / dh)
        for t in range(T)
        for i in range(dh // 2)
    ]
    scale = float(1 << cs)
    return (
        [_round_half_away(scale * math.cos(x)) for x in angles],
        [_round_half_away(scale * math.sin(x)) for x in angles],
    )


def softmax_table(expS: int, rx: int) -> list[int]:
    """EX[u] for u = 0 .. u0 - 1, where u0 is the smallest u with EX[u] = 0. Python's round() sends
    ties to even."""
    table = []
    while True:
        entry = round(float(1 << expS) * math.exp(-len(table) / float(1 << rx)))
        if entry == 0:
            return table
        table.append(entry)


def silu_table(eg: int, smax: int) -> list[int]:
    """Ts[c] for the keys c = -32,768 .. 32,767."""
    table = []
    for c in range(-KEY_OFFSET_16, KEY_OFFSET_16):
        z = min(max(c, -(64 << eg)), (64 << eg) - 1) / float(1 << eg)
        s = z / (1.0 + math.exp(-z))
        table.append(min(max(_round_half_away(64.0 * s), -smax), smax))
    return table


def sigmoid_table() -> list[int]:
    """Sig[c] for the keys c = -32,768 .. 32,767."""
    return [
        16 * _round_half_away(4096.0 / (1.0 + math.exp(-c / 256.0)))
        for c in range(-KEY_OFFSET_16, KEY_OFFSET_16)
    ]


_TABLE_NUMBER = 72  # the number under which a public table is hashed
DW, SW, BLKW = 8, 16, 32  # 32-bit words of a digest, of BLAKE2b's 64-byte result and of its block

# The hash of a public table is built on the compression function of BLAKE2b. "RFC 7693" below is
# "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", by M-J. Saarinen and
# J-P. Aumasson (https://www.rfc-editor.org/rfc/rfc7693), which specifies BLAKE2b.
#
# fmt: off
# The text from _rotr64 to the end of _compress is adapted from the BLAKE2 team's reference implementation in C. The head of its
# file blake2b-ref.c, which its two headers have too:
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
# It is a translation into Python of rotr64, blake2b_IV, blake2b_sigma, blake2b_init_param with what blake2b_init does before it
# calls it, the macros G and ROUND, and blake2b_compress. The names are the reference's behind an underscore, which marks them
# as private to this module, except two: blake2b_IV is _IV and blake2b_compress is _compress.
# What differs from the reference:
# - A Python integer has no fixed width, so every sum and every rotation is cut to 64 bits with _MASK64.
# - G and ROUND are functions, not macros: _G also takes m and returns the four new words, and _ROUND also takes v and m and
#   writes into v. The twelve lines ROUND( 0 ) to ROUND( 11 ) are a loop.
# - _compress has no state structure S, no buffer and no load64. What the reference keeps in S, and the block, are arguments:
#   h is S->h, t_low is S->t[0], t_high is S->t[1], last is whether the last-block flag S->f[0] is set, m is the block as
#   sixteen words. The arguments are not written into, and the new chaining value is returned.
# - In the reference S->t[1] is the upper half of a count of bytes; here t_high is any word, and hash_words puts the format's
#   additional words there. S->f[1], which marks a last node in tree hashing, is always 0 here.
# - _blake2b_init_param takes the two lengths, not a filled parameter block, and returns S->h.

_MASK64 = 2**64 - 1


def _rotr64(w, c):
    return (w >> c | w << (64 - c)) & _MASK64


_IV = (
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
)  # fmt: skip

_blake2b_sigma = (
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


def _blake2b_init_param(outlen, keylen):
    """Returns the initial chaining value, the reference's S->h after blake2b_init (keylen = 0) or
    blake2b_init_key: blake2b_IV XOR the parameter block P, read as little-endian words (load64)."""
    # digest_length, key_length, fanout, depth; all later fields are 0
    P = bytes([outlen, keylen, 1, 1]) + bytes(60)
    return [_IV[i] ^ int.from_bytes(P[8 * i : 8 * i + 8], "little") for i in range(8)]


def _G(r, i, a, b, c, d, m):
    a = (a + b + m[_blake2b_sigma[r][2 * i + 0]]) & _MASK64
    d = _rotr64(d ^ a, 32)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 24)
    a = (a + b + m[_blake2b_sigma[r][2 * i + 1]]) & _MASK64
    d = _rotr64(d ^ a, 16)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 63)
    return a, b, c, d


def _ROUND(r, v, m):
    v[0], v[4], v[8], v[12] = _G(r, 0, v[0], v[4], v[8], v[12], m)
    v[1], v[5], v[9], v[13] = _G(r, 1, v[1], v[5], v[9], v[13], m)
    v[2], v[6], v[10], v[14] = _G(r, 2, v[2], v[6], v[10], v[14], m)
    v[3], v[7], v[11], v[15] = _G(r, 3, v[3], v[7], v[11], v[15], m)
    v[0], v[5], v[10], v[15] = _G(r, 4, v[0], v[5], v[10], v[15], m)
    v[1], v[6], v[11], v[12] = _G(r, 5, v[1], v[6], v[11], v[12], m)
    v[2], v[7], v[8], v[13] = _G(r, 6, v[2], v[7], v[8], v[13], m)
    v[3], v[4], v[9], v[14] = _G(r, 7, v[3], v[4], v[9], v[14], m)


def _compress(h, m, t_low, t_high, last):
    """BLAKE2b's compression function F (RFC 7693, section 3.2; its IV: section 2.6, SIGMA: 2.7, G: 3.1), on 64-bit integers.

    h is the chaining value (8 integers), m the block (16 integers), t_low and t_high the two halves of the RFC's
    counter t, and last its flag f. The result is the new chaining value.
    """
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]
    v[8] = _IV[0]
    v[9] = _IV[1]
    v[10] = _IV[2]
    v[11] = _IV[3]
    v[12] = _IV[4] ^ t_low  # S->t[0]
    v[13] = _IV[5] ^ t_high  # S->t[1]

    # S->f[0]: blake2b_set_lastblock sets it to (uint64_t)-1
    v[14] = _IV[6] ^ (_MASK64 if last else 0)
    v[15] = _IV[7]  # S->f[1] is 0
    for r in range(12):
        _ROUND(r, v, m)
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def hash_words(num, words):
    """The digest H(num; words), a list of DW words.

    num is the domain-separation number and words a list of 32-bit words, possibly empty. H follows BLAKE2b-512 without
    a key (RFC 7693, section 3.3) over the words as little-endian bytes, with one difference: in every call of the
    compression function the high half of the counter, which in BLAKE2b is 0 for an input below 2^64 bytes, is
    num + 2^32 * len(words). hashlib.blake2b has no argument for the counter, so the hash is written
    out in this module. The digest is the first DW words of the 64-byte result.
    """
    count = len(words)
    if not 0 <= num < 2**32 or count >= 2**32 or not all(0 <= word < 2**32 for word in words):
        raise ValueError("H is defined for a number, a list length and words from 0 to 2^32 - 1")
    blocks = max(1, -(-count // BLKW))
    high = num + (count << 32)
    h = _blake2b_init_param(64, 0)  # an output of 64 bytes, no key
    for k in range(blocks):
        block = words[BLKW * k : BLKW * (k + 1)]
        m = struct.unpack("<16Q", struct.pack("<%dI" % len(block), *block).ljust(4 * BLKW, b"\0"))
        last = k == blocks - 1
        h = _compress(h, m, 4 * count if last else 4 * BLKW * (k + 1), high, last)
    return list(struct.unpack("<%dI" % SW, struct.pack("<8Q", *h))[:DW])
# fmt: on


def table_digest(words: list[int]) -> list[int]:
    """The digest of a public table, as eight 32-bit words: hash_words() under the number 72
    (docs/model.md, "4 Public tables").

    A word outside 0 .. 2^32 - 1 raises ValueError. No table of a parameter file that
    Parameters.read accepts has one.
    """
    return hash_words(_TABLE_NUMBER, words)


@dataclass(frozen=True)
class Tables:
    """The public tables as int64 tensors on the model's device."""

    cos: torch.Tensor  # C(t, i), shape (T, dh / 2)
    sin: torch.Tensor  # S(t, i), shape (T, dh / 2)
    # EX[0 .. u0 - 1], then one more entry, 0, which is used for every key from u0 on.
    softmax: torch.Tensor
    silu: torch.Tensor  # Ts; key c is at index c + 32,768
    sigmoid: torch.Tensor  # Sig for the keys -16,384 .. 16,383; key c is at index c + 16,384


def make_tables(p: Parameters, device: torch.device) -> Tables:
    """Computes the tables for the parameters p. Raises ParameterError if a table does not match its
    digest in the parameter file.

    docs/model.md, "4 Public tables", lists which words are hashed. Tables.sigmoid contains Sig only
    for the keys -16,384 .. 16,383, the only keys at which the model reads it (the same section).
    """
    cos, sin = rotary_tables(p.T, p.dh, p.cs, p.theta0)
    softmax = softmax_table(p.expS, p.rx)
    silu = silu_table(p.eg, p.smax)
    sigmoid = sigmoid_table()
    router_sigmoid = sigmoid[KEY_OFFSET_16 - KEY_OFFSET_15 : KEY_OFFSET_16 + KEY_OFFSET_15]
    if p.gate_clamp:
        # A proof's lookup table for the gate's clamp, hashed after Sig; the model never reads it.
        clamp_rows = [
            min(max(key, -127), 127) + 128 for key in range(-KEY_OFFSET_15, KEY_OFFSET_15)
        ]
        sigmoid_words = router_sigmoid + clamp_rows
    else:
        sigmoid_words = sigmoid
    hashed = {
        "tsilu": ("SiLU table", [entry + 4096 for entry in silu]),
        "tsig": ("sigmoid table", sigmoid_words),
        "trope": ("pair of rotary tables", [entry + 4096 for entry in cos + sin]),
        "texp": ("softmax table", softmax),
    }
    for name, (table, words) in hashed.items():
        if table_digest(words) != p.table_digests[name]:
            raise ParameterError(
                f"The {table} computed on this machine does not match its digest in the parameter "
                f"file (lines {name}0 to {name}7), so this would not be the proved model. "
                "Possible causes: a parameter that the table depends on was edited after the "
                "digests were made; the digests were made with another hash function; or this "
                "machine's math library does not round exp, cos, sin or pow correctly, in which "
                "case the model cannot run here."
            )

    def tensor(entries: list[int]) -> torch.Tensor:
        return torch.tensor(entries, dtype=torch.int64, device=device)

    return Tables(
        cos=tensor(cos).view(p.T, p.dh // 2),
        sin=tensor(sin).view(p.T, p.dh // 2),
        softmax=tensor(softmax + [0]),
        silu=tensor(silu),
        sigmoid=tensor(router_sigmoid),
    )
