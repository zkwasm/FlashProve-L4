#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Computes the digests of the four public tables of the integer (fixed-point) model and prints a parameter file with them.

usage: policy_tables.py [--check | --strip] <policy.txt>

The integer model reads four public tables: the SiLU table of the feed-forward gate, the sigmoid table of the router, the
rotary tables (cosines and sines of the positions) and the softmax table of attention. A parameter file (policy.txt) can
state a digest of each table, one line for each of its words: 32 lines, tsilu0 to texp7. A verifier or prover that reads such
a file computes the tables itself and stops if one of its digests differs from the lines.

Without --check or --strip, the program prints the parameter file with the lines computed from the file's other lines. It takes
out every digest line that the file has, wherever it is, and puts the new lines where the format requires them: directly
before the first call line. Every other line is copied unchanged. Run the program again after a change of one
of the lines eg, smax, gcl, lt, dh, cs, theta0_lo, theta0_hi, S or rx.
With --check, the program compares the file's digest lines, in the order of the file, with the computed ones and prints
what it finds. It does not test where the lines are; the verifier does.
With --strip, the program prints the file without its digest lines and interprets no other line.

To compute the lines, the program needs a file with "prof 20" and NE above 0, because only such a
model has all four tables. It rejects cs above 12 and S above 31, where an entry would not fit a
32-bit word. It does not test whether the verifier accepts the file's values. Time and memory grow
with the rotary tables (2^lt * dh entries) and the softmax table (in proportion to 2^rx).

Exit status:
  0  Done. With --check, the file has exactly the computed lines, in order.
  1  With --check, the file does not have exactly these lines. Standard output says what differs, with one line for each
     table concerned.
  2  The command line has none of the three forms, or the program rejects the file. Standard error has a line that says
     why, and standard output is empty.

The tables and the words that are hashed are defined in docs/model.md, "4 Public tables" (which counts five tables: the
rotary tables C and S as two), and the hash in verifier/HASHES.md, "H(tag; x)". This file shares no source file with the
verifier or the provers: it imports only Python's standard library. So a test that compares its lines with a
program's own tables compares two implementations of the same written definition.
One part has a common outside source: the text of BLAKE2b's compression function here and in the programs is adapted from the
BLAKE2 reference implementation (the comment at the hash H), so such a test can show an error made in adapting that text, not an
error of the text.
Tests in other directories import digests(), table_lines() and the format's widths from here, and so
run no source file of the program under test for them.
"""

import math
import re
import struct
import sys

# ---------------------------------------------------------------------------------------------------------------------
# Widths of the format. The unit is a 32-bit word unless a comment says otherwise. This file's
# functions use DW, SW, BLKW and TBLW. The others are for the scripts and tests that import them.
# prover/common/setting.h has and explains every name except SALT_CELLS and those with HC_;
# verifier/src/vf_widths_and_bounds.h has the four that the functions use, THPW and HEXW.

# A digest has 8 words. The programs have the width as a constant of their build (DIGEST_WORDS), and
# scripts import it from this file.
DW = 8  # a digest
SW = 16  # a hash value before it is cut to a digest (a chaining value)
BLKW = 32  # a block, the input of one call of the compression function
THPW = 16 * DW  # the tree hash TH cuts its input into parts of this length
HEXW = 8 * DW  # hexadecimal digits of a printed digest
SALTW = 10  # the salt of a leaf of a row commitment
SALT_CELLS = SALTW // 2  # cells that this salt takes in a column
SESSW = 10  # the salt of a session
LAMW = DW  # the salt of a leaf of a batch's tree
NONCEW = 4  # the nonce of an attempt
RETRYQ = 8  # a prover labels a random value that it draws again with RETRYQ * (number of the retry) + its place in the block
FILLC = 8  # a column of a format-B tree is filled to a multiple of this many cells
SLABR = BLKW // 2  # rows that one call of the running hash of a format-A leaf takes
SH2_DIGB = 32  # bytes that a prover's choice of a tree's shape counts for one digest of a path
# Bits: the verifier's target. No run that requests fewer is a verification. In the verifier's own
# error accounting every round must reach the requested bits. BAR is no security level.
BAR = 128

# A weight table's commitment record (prover/hash_commitment/hf_table_format.h). The record is nine
# words and four digests, the last of which is h_once. h_0 is the tree hash of its first HC_REC0W
# words and h_card that of all HC_RECW. The file wcard.bin is the record, then T_F, the root RT_F.
HC_REC0W = 9 + 3 * DW
HC_RECW = HC_REC0W + DW
HC_TWW = DW
HC_CARDW = HC_RECW + HC_TWW  # the length of wcard.bin
TBLW = 4 * DW  # the digest lines of a parameter file

# ---------------------------------------------------------------------------------------------------------------------
# The hash H (defined in verifier/HASHES.md, "H(tag; x)"). It uses the compression function of
# BLAKE2b. "RFC 7693" below is "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", by M-J. Saarinen and
# J-P. Aumasson, which specifies BLAKE2b.
#
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
    num + 2^32 * len(words). Python's hashlib.blake2b has no argument for the counter, so the chain is written out here.
    The digest is the first DW words of the 64-byte result.
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


# ---------------------------------------------------------------------------------------------------------------------
# The four tables (defined in docs/model.md, "4 Public tables", which counts five: the rotary tables C and S as two).
#
# The tables are defined by procedure: every +, * and / below is one IEEE 754 binary64 operation, in the stated order,
# which is what Python's float arithmetic does. A multiplication or division by a power of two is exact. Nothing here
# may be evaluated in higher precision: the SiLU table at eg = 8 and smax = 4,095, for one, differs from the correctly
# rounded function at 1,744 keys.
#
# The definition asks for correctly rounded exp, cos, sin and pow. Here they are those of the
# platform's C library, through the module math. In the standard parameter form (defined in
# verifier/docs/parameter-file.md, "The standard parameter form": without it cp_verify ends at best
# with NOT-A-VERIFICATION) no entry changes if all results of that library are moved by 1, 16 or
# 1,024 units in the last place, given binary64 arithmetic in the stated order: computed for every
# entry (scales/tests/table_ties.py). That no smaller error changes an entry is argued there from
# monotonic steps and not run, and nothing of this is proved; with 80-bit intermediate results some
# entries differ. Outside that form an entry can lie so close to a tie that
# two libraries round it differently: at theta0 = 3,675,548,936, dh = 128 and cs = 12 the cosine of
# pair 1 at position 689 is -548 with correctly rounded functions and -547 with a common library.
# The digest lines then match the tables of programs that use the same library as this one.
# Source for the question whether an entry lies so close to a tie that two libraries round it
# differently, and for answering it by a search over all arguments: Lefèvre, Muller, Tisserand,
# "The Table Maker's Dilemma", Research Report RR-1998-12, Laboratoire de l'Informatique du
# Parallélisme, École Normale Supérieure de Lyon, 1998 (HAL, hal-02101765), the introduction and
# the section "Exhaustive tests". There the arguments and results are floating-point numbers; here
# the arguments are the keys of a finite table and the results are integers.
#
# Reading an activation function from a table with fixed-point keys and values is common in fixed-point inference (L. Lai
# et al., "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs", arXiv:1801.06601, section 4.5.2) and in
# proofs of inference, where a lookup argument proves each read (H. Sun et al., "zkLLM: Zero Knowledge Proofs for Large
# Language Models", arXiv:2404.16109, section 4 on tlookup and section 5 on zkAttn).

KEYS = range(-32768, 32768)  # the keys of the SiLU table and of the sigmoid table, in the order of the tables


def _round_half_away(v):
    """The binary64 number v rounded to the nearest integer, ties away from zero; the addition is a binary64 addition."""
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def silu_table(eg=8, smax=4095):
    """The SiLU table Ts, a list over KEYS. A key is in units of 2^-eg and a value in units of 2^-6,
    clamped to -smax .. smax.

    SiLU(z) = z * sigmoid(z) (S. Elfwing et al., "Sigmoid-Weighted Linear Units for Neural Network Function Approximation
    in Reinforcement Learning", arXiv:1702.03118). The table is constant below the key -64 * 2^eg and from 64 * 2^eg - 1 on.
    """
    table = []
    one = 2**eg  # the key of z = 1
    for c in KEYS:
        z = min(max(c, -64 * one), 64 * one - 1) / one
        e = math.exp(-z)
        g = 1 + e
        s = z / g
        table.append(min(max(_round_half_away(64 * s), -smax), smax))
    return table


def sigmoid_table():
    """The sigmoid table Sig, a list over KEYS. Keys are in units of 2^-8, values of 2^-16."""
    table = []
    for c in KEYS:
        e = math.exp(-c / 256)
        g = 1 + e
        q = 4096 / g
        table.append(16 * _round_half_away(q))
    return table


def rotary_tables(positions, dh, cs, theta0):
    """The rotary tables C and S, two lists in the order position t major, pair i minor. A value is
    in units of 2^-cs. There are dh // 2 pairs.

    The rotary position encoding is that of J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding",
    arXiv:2104.09864. Its section 3.2.2 sets the angles theta_i = 10000^(-2 (i - 1) / d) for i from 1 to d / 2.
    The angle of pair i at position t is t * theta0^(-2 i / dh), as in the library transformers, version 5.17.0
    (github.com/huggingface/transformers), src/transformers/models/llama4/modeling_llama4.py, line 214:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)).
    The code below keeps the order of docs/model.md, "4 Public tables", item 1 (one call of pow with
    a negative exponent), and not the order of that line: a reciprocal would round once more.
    """
    cosines, sines = [], []
    one = 2**cs  # the value that stands for 1
    for t in range(positions):
        for i in range(dh // 2):
            b = (-2 * i) / dh
            w = math.pow(theta0, b)
            x = t * w
            cosines.append(_round_half_away(one * math.cos(x)))
            sines.append(_round_half_away(one * math.sin(x)))
    return cosines, sines


def softmax_table(exp_s, rx):
    """The softmax table EX: round(2^exp_s * exp(-u / 2^rx)), ties to even, for u = 0, 1, .. up to the last nonzero entry."""
    table = []
    while True:
        e = math.exp(-len(table) / 2**rx)
        entry = round(2**exp_s * e)
        if entry == 0:
            return table
        table.append(entry)


# ---------------------------------------------------------------------------------------------------------------------
# The parameter file (described in verifier/docs/parameter-file.md).

TABLES = ("tsilu", "tsig", "trope", "texp")  # the names of the four groups of digest lines, in the order of the file
_TABLE_NUMBER = 72  # the domain-separation number of a public table's digest (verifier/HASHES.md)
_OFFSET = 4096  # added to the signed entries of the SiLU and rotary tables, which makes them words

_REQUIRED = ("prof", "NE", "lt", "dh", "cs", "theta0_lo", "theta0_hi", "S", "rx")
_OPTIONAL = ("eg", "smax", "gcl")
_MOST_DIGITS = 20  # the most digits that a value may have after its leading zeros; the format's values have at most 10
# The largest values at which every hashed entry is a 32-bit word: a rotary entry plus _OFFSET is not negative, and the
# first softmax entry, 2^S, is below 2^32.
_LARGEST = {"cs": 12, "S": 31}


def _parameters(lines):
    """The values of the lines that determine the tables, as a dictionary from name to integer.

    A line counts if it is exactly a name, one space and decimal digits. Of several such lines with one name the first
    counts. Any other line with that name is passed over, as if it were absent. These two rules matter only for files that
    the verifier rejects: it accepts each name once, at its place, and only in the exact form.
    """
    found = {}
    for line in lines:
        name, _, value = line.partition(" ")
        if name in _REQUIRED + _OPTIONAL and name not in found and re.fullmatch("[0-9]+", value):
            if len(value.lstrip("0")) > _MOST_DIGITS:
                raise ValueError("the value of the line %s has more than %d digits" % (name, _MOST_DIGITS))
            found[name] = int(value.lstrip("0") or "0")
    for name in _REQUIRED:
        if name not in found:
            raise ValueError('there is no line of the exact form "%s <number>": the name, one space and decimal digits, and '
                             "nothing after them, not even a carriage return" % name)  # fmt: skip
    if found["prof"] != 20 or found["NE"] == 0:
        raise ValueError('the file does not have "prof 20" and NE above 0, so its model lacks one of the four tables')
    for name, largest in _LARGEST.items():
        if found[name] > largest:
            raise ValueError("the line %s is above %d, the largest value that this program handles" % (name, largest))
    found["theta0"] = found["theta0_lo"] + 2**32 * found["theta0_hi"]
    if found["theta0"] == 0:
        raise ValueError("the rotary base, theta0_lo + 2^32 * theta0_hi, is 0")
    return found


def digests(lines):
    """The digests of the four tables, in the order of TABLES, each a list of DW words.

    lines are the lines of a parameter file without their line feeds. Digest lines among them are ignored. The words
    that are hashed are those of docs/model.md, "4 Public tables", "Digests". If the lines do not
    determine the four tables, the function raises a ValueError whose text says why.
    """
    p = _parameters(lines)
    sigmoid = sigmoid_table()
    if "gcl" in p:
        # With the gate's clamp, the table that is hashed has the sigmoid's middle keys in its lower half, and in its
        # upper half the clamp to -127 .. 127, shifted by 128.
        sigmoid = sigmoid[16384:49152] + [min(max(t - 49152, -127), 127) + 128 for t in range(32768, 65536)]
    hashed = (
        [entry + _OFFSET for entry in silu_table(p.get("eg", 8), p.get("smax", 4095))],
        sigmoid,
        [entry + _OFFSET for table in rotary_tables(2 ** p["lt"], p["dh"], p["cs"], p["theta0"]) for entry in table],
        softmax_table(p["S"], p["rx"]),
    )
    return [hash_words(_TABLE_NUMBER, words) for words in hashed]


def table_lines(d):
    """The digest lines, without line feeds, for the result d of digests."""
    return ["%s%d %d" % (table, i, word) for table, digest in zip(TABLES, d) for i, word in enumerate(digest)]


def _table_of(line):
    """The group of digest lines to which a line belongs by its name (the text before the first space), or None."""
    name = re.fullmatch("(%s)[0-9]+" % "|".join(TABLES), line.partition(" ")[0])
    return name[1] if name else None


def is_table_line(line):
    """Whether a line of a parameter file is a digest line: its name is that of a group, followed by decimal digits.

    So a stray line such as "tsilu10 5", which the verifier rejects, is also taken out, both by
    --strip and when the lines are computed.
    """
    return _table_of(line) is not None


# ---------------------------------------------------------------------------------------------------------------------
# The command line.

USAGE = "usage: policy_tables.py [--check | --strip] <policy.txt>"
DIFFERS = 1  # the exit status of --check if the file's digest lines are not the computed ones
REJECTED = 2  # the exit status of a wrong command line and of a file that the program rejects
_AGAIN = "run policy_tables.py without --check to compute them"


def _read(path):
    """The lines of the file, without their line feeds."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        hint = "; " + USAGE if path.startswith("-") else ""  # for someone who tried an option that does not exist
        raise ValueError("the file cannot be read (%s)%s" % (e.strerror, hint)) from e
    if not data.isascii():
        raise ValueError("the file is not ASCII text")
    if not data.endswith(b"\n"):
        raise ValueError("the file is empty, or its last line does not end with a line feed")
    return data.decode("ascii")[:-1].split("\n")


def _with_tables(lines):
    """The lines with the digest lines taken out and the computed ones directly before the first call line."""
    rest = [line for line in lines if not is_table_line(line)]
    at = [i for i, line in enumerate(rest) if line.startswith("call ")][:1]
    if not at:
        raise ValueError('the file has no line that begins with "call "; the digest lines go directly before the first of them')
    return rest[: at[0]] + table_lines(digests(rest)) + rest[at[0] :]


def _check(lines):
    """Compares the digest lines, in the order of the file, with the computed ones. Returns what differs, as messages."""
    computed = table_lines(digests(lines))
    stated = [line for line in lines if is_table_line(line)]
    if not stated:
        return ["the file has no digest lines; " + _AGAIN]
    findings = [
        "the lines %s0 to %s%d do not match the table that the file's other lines define; %s" % (table, table, DW - 1, _AGAIN)
        for k, table in enumerate(TABLES)
        if [line for line in stated if _table_of(line) == table] != computed[DW * k : DW * (k + 1)]
    ]
    if not findings and stated != computed:
        findings = ["the digest lines are right but not in the order %s; %s" % (", ".join(TABLES), _AGAIN)]
    return findings


def main(arguments):
    """Runs the program on the command line's arguments. Returns the exit status."""
    mode = arguments[0] if arguments and arguments[0] in ("--check", "--strip") else ""
    paths = arguments[1:] if mode else arguments
    if len(paths) != 1:
        print(USAGE, file=sys.stderr)
        return REJECTED
    try:
        lines = _read(paths[0])
        if mode == "--check":
            findings = _check(lines)
        elif mode == "--strip":
            new = [line for line in lines if not is_table_line(line)]
        else:
            new = _with_tables(lines)
    except (ValueError, MemoryError) as e:
        print("%s: %s" % (paths[0], str(e) or "the tables for the file's values do not fit in memory"), file=sys.stderr)
        return REJECTED
    if mode == "--check":
        for finding in findings or ["the %d digest lines match the tables" % TBLW]:
            print(finding)
        return DIFFERS if findings else 0
    sys.stdout.buffer.write("".join(line + "\n" for line in new).encode("ascii"))  # as bytes: no line end is translated
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
