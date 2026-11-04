# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Masks and salts of a weight table in Python, with known answers, and a checker for call logs.

A weight table (the committed weights, as a matrix) has R rows, each with nm masks (random values
encoded with the row), and n Merkle leaves, each with a salt. Both come from one key of 32 bytes
for each table, kM (derived from the commitment's seed: prover/common/coin_keys.h), and f, the
block function of ChaCha20 (RFC 8439, section 2.3: sixteen output words o_0 .. o_15, and all four
last words x12 .. x15 of the initial state are inputs). With
v_sigma = o_(2 sigma) + 2^32 o_(2 sigma + 1):

    masks   blk(r, g, try) = f(kM; r, g, try, 0x4D41534B),   r < R, g < ceil(nm / 8), try = 0, 1, ..
            a block is good if all eight v_sigma are below the threshold L, the greatest multiple
            of the field's prime p that is at most 2^64: p itself at p = 2^64 - 2^32 + 1 (field 2,
            Goldilocks, the programs' field), 8 p = 2^64 - 8 at p = 2^61 - 1 (field 1: see PRIMES)
            try* = the least try number whose block is good
            mask[r][8 g + sigma] = v_sigma mod p, from blk(r, g, try*)
            the values of a row's last block with 8 g + sigma >= nm are dropped
    salts   salt_I = (v_0 mod p, .., v_4 mod p) of one block, I < n:
            f(kM; I mod 2^32, floor(I / 2^32), 0, 0x53414C54); no threshold and no second try

docs/format/COIN-KEYS.md, "Below a table's commit-run key", states the same rule; it calls kM
a table's commit-run key.

This file shares no code with the programs cp_commit and cp_prove: block() is typed from the RFC
(RFC 8439, sections 2.1 to 2.3), and the rest is typed from the lines above. It has the
generator only: nothing of a row's encoding, of a leaf's hash or of the tree.

Run as a script (python3 -B toy/mask_rules.py: the test `toy.mask_rules` of SUITE.md), it recomputes
the RFC's test vector of section 2.3.2 and the known answers below and prints whether they come
out; the exit code is 0 only if all do. With --make or --header (see KNOWN2) it prints rows and
exits with 0. Nothing in this tree imports it: other files use only its table KNOWN2, and row(),
dropped() and the log checker are never run.
"""

import collections
import re
import struct
import sys

# The argument `field` of the functions below is 1 or 2, a key of PRIMES. The programs can be built
# for field 2 alone; this file also has field 1, with its known answers (KNOWN). The default is 2 in
# threshold(), mask_block(), row(), dropped(), salt() and findings(), but 1 in known(),
# known_answers_hold() and known_inputs().
P = 2**61 - 1  # field 1
PRIMES = {1: P, 2: 2**64 - 2**32 + 1}
MASK_WORD = 0x4D41534B  # the fourth input word of every block that gives masks ("MASK")
SALT_WORD = 0x53414C54  # the fourth input word of every block that gives a salt ("SALT")
SALT_SYMBOLS = 5  # symbols (field elements) of a salt: the first of its block's eight numbers
CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)  # RFC 8439, 2.3
M32 = 0xFFFFFFFF
# RFC 8439, 2.3: a double round is four column rounds, then four diagonal rounds.
ROUNDS = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))  # fmt: skip


def block(key, x12, x13, x14, x15):
    """f: the sixteen output words of the block function under the key's eight words, with
    x12 .. x15 as the last four words of the initial state. Every word must be below 2^32, which is
    not checked."""
    first = list(CONSTANTS) + list(key) + [x12, x13, x14, x15]
    w = list(first)
    for _ in range(10):
        for a, b, c, d in ROUNDS:  # the quarter round of RFC 8439, 2.1
            wa, wb, wc, wd = w[a], w[b], w[c], w[d]
            wa = (wa + wb) & M32
            wd ^= wa
            wd = ((wd << 16) | (wd >> 16)) & M32
            wc = (wc + wd) & M32
            wb ^= wc
            wb = ((wb << 12) | (wb >> 20)) & M32
            wa = (wa + wb) & M32
            wd ^= wa
            wd = ((wd << 8) | (wd >> 24)) & M32
            wc = (wc + wd) & M32
            wb ^= wc
            wb = ((wb << 7) | (wb >> 25)) & M32
            w[a], w[b], w[c], w[d] = wa, wb, wc, wd
    return [(x + y) & M32 for x, y in zip(w, first)]


def seed_words(data):
    """The eight words of a key's 32 bytes, each word with its low byte first, as RFC 8439, 2.3,
    reads a key."""
    if len(data) != 32:
        raise ValueError("a key has 32 bytes")
    return list(struct.unpack("<8I", data))


def values(o):
    """v_0 .. v_7 of a block's sixteen words."""
    return [o[2 * s] + (o[2 * s + 1] << 32) for s in range(8)]


def blocks(nm):
    """ceil(nm / 8): the number of blocks for a row's nm masks."""
    return -(-nm // 8)


def threshold(k=8, field=2):
    """The threshold that the digit k names: at 8 the rule's L; at 1 to 7 a test's lower one,
    k floor(L / 8), which is k p in field 1.

    A block is good with probability about (k / 8)^8: 0.34 at k = 7, 0.10 at k = 6, 2^-8 at k = 4,
    2^-24 at k = 1. So a whole table can be made at 6 and 7, and no table at 1. At k = 8 a block is
    rejected with probability about 8 (2^64 - L) / 2^64, that is 2^-58 in field 1, which no run
    meets, and 2^-29 in field 2: about 21 times in a table of the full size
    (prover/examples/target_full_f2). The generator of
    prover/hash_commitment/hf_mask.cuh takes a lower threshold only if it is compiled with the macro
    HC_MASK_LOW_LIMIT nonzero; mask_limit() then reads k from the test variable H_HC_MASK_LIMIT."""
    p = PRIMES[field]
    full = 2**64 // p * p
    if k == 8:
        return full
    if k not in range(1, 8):
        raise ValueError("a threshold's digit is 1 to 8")
    return k * (full // 8)


def mask_block(key, r, g, k=8, field=2):
    """(the eight numbers v_sigma mod p of blk(r, g, try*); try*), at the threshold of the digit k.
    They are mask[r][8 g + sigma] as far as 8 g + sigma < nm."""
    p, limit, t = PRIMES[field], threshold(k, field), 0
    while True:
        v = values(block(key, r, g, t, MASK_WORD))
        if max(v) < limit:
            return [x % p for x in v], t
        t += 1


def row(key, r, nm, k=8, field=2):
    """(the nm masks of row r; the try number taken for each g < ceil(nm / 8))."""
    masks, tries = [], []
    for g in range(blocks(nm)):
        mu, t = mask_block(key, r, g, k, field)
        masks += mu
        tries.append(t)
    return masks[:nm], tries


def dropped(key, r, nm, k=8, field=2):
    """The values of row r's last block that are no masks: those with 8 g + sigma >= nm. No test uses
    it to check that a prover writes them nowhere."""
    return mask_block(key, r, blocks(nm) - 1, k, field)[0][nm - 8 * (blocks(nm) - 1) :]


def salt(key, I, field=2):
    """The symbols of the salt of leaf I."""
    return [x % PRIMES[field] for x in values(block(key, I % 2**32, I >> 32, 0, SALT_WORD))[:SALT_SYMBOLS]]


def rfc_vector_holds():
    """Whether block() gives the output of RFC 8439, 2.3.2: the key 00 01 .. 1f, the block
    counter 1, the nonce 00 00 00 09 00 00 00 4a 00 00 00 00."""
    got = block(seed_words(bytes(range(32))), 1, 0x09000000, 0x4A000000, 0)
    return got == [0xE4E7F110, 0x15593BD1, 0x1FDD0F50, 0xC47120A3, 0xC7F4D1C7, 0x0368C033, 0x9AAA2204, 0x4E6CD4C3,
                   0x466482D2, 0x09AA9F07, 0x05D7C214, 0xA2028BD9, 0xD19C12B5, 0xB94E16DE, 0xE883D0CB, 0x4E3C50A2]  # fmt: skip


def mix64(z):
    """One step of SplitMix64, used here only to make test keys: the same function as hc_mix64() of
    prover/hash_commitment/hc_format.h. Public source: splitmix64.c by Sebastiano Vigna,
    prng.di.unimi.it/splitmix64.c (public domain), which has the step with the same three constants.
    The increment and the mix (as mix64variant13) are printed in G. Steele, D. Lea, C. Flood,
    "Fast Splittable Pseudorandom Number Generators" (OOPSLA 2014)."""
    z = (z + 0x9E3779B97F4A7C15) % 2**64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 % 2**64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB % 2**64
    return z ^ (z >> 31)


def test_seed(n):
    """The eight words of test key n, 0 to 255 (not checked): the same as hc_test_seed_word() of
    prover/hash_commitment/hc_format.h. Test key 0 is the bytes 00 01 .. 1f, the key of RFC 8439,
    2.3.2."""
    return [(mix64(0x484B4559 + 16 * n + i) >> 16) % 2**32 if n else 0x03020100 + 0x04040404 * i for i in range(8)]


# Known answers for field 1, made by this file's functions, in rows of the form place: answer.
#   ("MASK", the test key's number, r, g, k): (try*, the eight numbers v_sigma mod p in hex)
#   ("SALT", the test key's number, I): the five symbols in hex
# No test in this tree compares these rows with another implementation: the C++ files named at
# KNOWN2 are for field 2 alone. Running this file recomputes them, and --make takes the places of
# KNOWN2's rows from them.
# Some rows have indices that small configurations rarely or never reach, to catch a truncated
# index: r above 255, g above 65,535, I from 2^32 on.
# (2047, 5505023) is the last block of the last row at the full size (R = 2048, nm = 44,040,191);
# its eighth number is the dropped value.
KNOWN = {
    ("MASK", 0, 0, 0, 8): (0, "1fb30dce10cc818b 1e8b7f2bf388be0d 02486e03f6a91ede 15dba716074d26eb 13b31db7b2a70c1e 0729952061d0fa8d 046d8ef053f50182 06272ca3ba8cd6fb"),
    ("MASK", 0, 1, 0, 8): (0, "1d9e469cc24680dc 1f7de042cba3a833 1da6a959784b24bb 14b8b2bbe840e17e 123224fba62c28a0 1eafb7553af55566 1ea25b02e66900c0 0a0877ca0fa93496"),
    ("MASK", 0, 0, 1, 8): (0, "0af3d8e0acee02ea 03662a5a19bdcb5b 191a3f2191486d97 1c59d6101717d436 1a00221ce0f1076f 0f1654db02c244f5 009338032cecde57 160b69b390730edf"),
    ("MASK", 0, 262143, 4194303, 8): (0, "0abcbeacd144e750 1437826087310637 19fb19ef442fd9e1 1dea8287e02c14ff 06bc1365158eaf62 040062fef2331411 0fb961cf4a9b9e23 0968ef36ce5cfadc"),
    ("MASK", 0, 0, 0, 7): (0, "1fb30dce10cc818b 1e8b7f2bf388be0d 02486e03f6a91ede 15dba716074d26eb 13b31db7b2a70c1e 0729952061d0fa8d 046d8ef053f50182 06272ca3ba8cd6fb"),
    ("MASK", 0, 1, 0, 7): (3, "0d61561d722fdab2 0749d4241fb0d2ae 1a151e5ef275a409 1308668100c166ea 071366dd429aafc6 0be057b0b73cced2 0a783567b8511333 129befb36d003191"),
    ("MASK", 0, 2, 0, 7): (2, "0cb53b70fd5dfafd 12084982f9ab51d8 0ff62b6a00f7c0af 041b4b636dd1bf9d 0f4057cf542cb784 09cbe1218ea1070c 08e93f1493144173 07fa2a087ef41e49"),
    ("MASK", 0, 3, 0, 7): (1, "15e9cb3a8f5f0cd7 16d39a200d90c53f 0789fc8e0f0b6ce2 1498b777c4a2637f 169ff2f8dcbb09dd 1dce843835bb5575 07636dce27e1c06b 1f2ac92d1b0e7e43"),
    ("MASK", 0, 6, 0, 6): (3, "114b2c290b5df88b 11b9e902bd9eb876 0f1ef85cacade090 0b56d1f16da3e6f4 1a53f370c8538a93 1770854edbfc6ad5 0343fe75ff78e67b 1654b47ef10db827"),
    ("MASK", 255, 0, 0, 8): (0, "1c0d53fb99c4a7ce 08a0477d1ae578e1 01d8a3673be830e8 1f668bb656628cc6 0741bd7af6076dec 1a490e9ca7ceec18 146a6296fa07534b 0f749f8150df3e37"),
    ("MASK", 1, 0, 0, 8): (0, "1b93ea1f2ddfbe17 1c413591cd609bb9 19130c1fc945431d 16fb6028d67ded61 15d52faafcd67aad 038d3e192adc331e 148d6f5712a56acc 0b841877d1c7f752"),
    ("MASK", 1, 1, 0, 8): (0, "18fe32c354f6bc6b 10e5084635b7cf41 0fbf482a4c60053f 042d7fc67ed00470 151159080902bf14 1aa684af8970be7c 02ef16f732b29341 1cb1269ccf8b7515"),
    ("MASK", 1, 0, 1, 8): (0, "195a1b0a16dbf255 06c44772985251ef 07191c8da328d847 1e849103685df526 1ffa51baf417921b 0e79380ed900cf62 039dc2909d602c68 0b287ace818fb406"),
    ("MASK", 1, 256, 0, 8): (0, "10c1adf4389383f1 09f67ba2283b7467 06ee1b39b1387a01 16d9b5b1a8a260b0 15ef88e6f3949289 162340b66b435b79 006d7eb4e9e5dd05 00984e3b74acc1b9"),
    ("MASK", 1, 0, 65536, 8): (0, "0dcee7b17551dc55 16bec09f3a5f1d13 07b4e50cf3633f34 1ce45cb0e02ea33d 19aee156ff5f153d 00c6ae1d0eaf34aa 0f63514181fcb678 1c7d14187367ab2c"),
    ("MASK", 1, 65536, 0, 8): (0, "0e30bdfba5c6ff88 0d6bc5891f560fe3 0b020ec15e709cd9 0ed9ee13d23860f6 02895c580a6b5f8f 09e22736a0b11176 05d5f0ed442ed21d 15f0ddcaab00c1fa"),
    ("MASK", 1, 2047, 5505023, 8): (0, "11ada3751f23c3a0 1b21c1e689eb37a0 1721f9efbad82ada 1da0608b4f4e5b77 18bab20fec2227ce 0b0f1a5ada2dee6d 19b6f3d997f18149 14c907247d86f195"),
    ("MASK", 1, 8191, 5505023, 8): (0, "0d8ca6767ff6eb99 0f2ff42402b33bb4 01b9e630bfcbc9ad 0c4a75ddd053f81e 123dfbc14b084ae1 01964ec3b13cab03 0573a228df42c5df 17472bcbc9b00421"),
    ("MASK", 1, 2047, 5505022, 7): (2, "072824a6af8fd700 0ed7f5849866b4da 13558cf130b73206 1e5e0fce298bddc1 096cf46f5642f94a 0293335265e29055 0323fef3d1d100d2 077ea8062df015ed"),
    ("MASK", 1, 2047, 5505020, 7): (1, "0d6f84c5c6c8de1b 0ea1c63c318c8814 1c8be78537c3e6e2 0ad0fe133ceb6ffb 0e2c440af3bc315a 00ec3f70471013eb 1b2ca073d43a3ac7 0e16b256d13cd65d"),
    ("SALT", 0, 0): "1bf90a4d1108f19e 09ba481bb13890be 1f5ba572f66a3cf1 136142df94e8d0dd 1e7b3d06bbdee6bd",
    ("SALT", 1, 0): "1ca55308f7250e55 1092d81a9d4057ed 072c8c7d77a022d1 1d3053c098f71ad1 169f496996f6ad14",
    ("SALT", 1, 1): "0cfce91e068aaf43 0f6d5a1ac3c0087d 076dd89629266fb3 0c915950efd0079e 057dc2197cb59ed8",
    ("SALT", 1, 4294967295): "05caa89f5acbff7f 09ebc82dbaf5735e 03ee12134b3238b0 0550de77b742d4f3 1f3d896c3560901e",
    ("SALT", 1, 4294967296): "0a595a5156bc9541 12a051050c88d011 087ab2a255705480 0a542c27a4431d9f 147a59b846ce3ecf",
    ("SALT", 1, 4294967297): "0f3267028bb1b2c1 01b85a65aa4cd909 0905f0c015ae3df8 0b94a9b7075463bf 0c824b880a5693df",
    ("SALT", 1, 8589934591): "199a80af389c5733 03f1262c45c13061 1ddca460e6bc79e1 0cc81c88bdf27e18 1e6e1d8215e27f50",
    ("SALT", 255, 8589934591): "1a0d70b8ad579bb7 13281ca189d66bad 094732a5f64b3a34 0e5a8dc71542db42 0937a769e0ee674b",
}  # fmt: skip


# The same for field 2: the places of KNOWN, and three more MASK rows at the full threshold whose
# first block is rejected (one of its eight numbers is p or more), so that try* = 1. A search over
# the rows 0 .. 399 and all block numbers of the full size under test key 1 found these three and
# no other; the program that searched used block() of prover/hash_commitment/hf_mask_ref.h and is
# not in this repository. To regenerate the rows:
# python3 -B toy/mask_rules.py --make 235:3890687 292:5264827 391:1987782
# prover/tests/masking_cells_test.cpp reads this table from this file's text: it finds the table by
# its first line, and every row must stay on one line, in the form given at KNOWN. It compares the
# SALT rows and the MASK rows at k = 8, and no others, with hf_salt5() and hf_mask8() of
# prover/hash_commitment/hf_mask.cuh, and every row with prover/hash_commitment/hf_mask_ref.h.
# prover/hash_commitment/hf_mask_known.h must have exactly these rows, in the same order. The option
# --header 2 prints that file's tables, from which the programs' self-test of their generator,
# generator_own_test() (prover/hash_commitment/hf_mask_own.cuh), takes its rows.
# What depends on the field here, in prover/hash_commitment/hf_mask_ref.h and in
# prover/second_writing/mask_generator.h was written together, from the rule's wording: for field 2
# these three are separate from prover/hash_commitment/hf_mask.cuh, not from one another.
KNOWN2 = {
    ("MASK", 0, 0, 0, 8): (0, "7fb30dce10cc8188 de8b7f2bf388be07 22486e03f6a91edd 55dba716074d26e9 b3b31db7b2a70c19 c729952061d0fa87 246d8ef053f50181 a6272ca3ba8cd6f6"),
    ("MASK", 0, 1, 0, 8): (0, "bd9e469cc24680d7 3f7de042cba3a832 1da6a959784b24bb 94b8b2bbe840e17a 323224fba62c289f 1eafb7553af55566 9ea25b02e66900bc ea0877ca0fa9348f"),
    ("MASK", 0, 0, 1, 8): (0, "eaf3d8e0acee02e3 a3662a5a19bdcb56 f91a3f2191486d90 dc59d6101717d430 3a00221ce0f1076e 4f1654db02c244f3 a09338032cecde52 760b69b390730edc"),
    ("MASK", 0, 262143, 4194303, 8): (0, "0abcbeacd144e750 5437826087310635 79fb19ef442fd9de bdea8287e02c14fa 06bc1365158eaf62 240062fef2331410 6fb961cf4a9b9e20 2968ef36ce5cfadb"),
    ("MASK", 0, 0, 0, 7): (0, "7fb30dce10cc8188 de8b7f2bf388be07 22486e03f6a91edd 55dba716074d26e9 b3b31db7b2a70c19 c729952061d0fa87 246d8ef053f50181 a6272ca3ba8cd6f6"),
    ("MASK", 0, 1, 0, 7): (3, "0d61561d722fdab2 6749d4241fb0d2ab ba151e5ef275a404 b308668100c166e5 071366dd429aafc6 cbe057b0b73ccecc 4a783567b8511331 929befb36d00318d"),
    ("MASK", 0, 2, 0, 7): (2, "0cb53b70fd5dfafd 72084982f9ab51d5 8ff62b6a00f7c0ab 641b4b636dd1bf9a 8f4057cf542cb780 49cbe1218ea1070a 48e93f1493144171 67fa2a087ef41e46"),
    ("MASK", 0, 3, 0, 7): (1, "95e9cb3a8f5f0cd3 76d39a200d90c53c a789fc8e0f0b6cdd 1498b777c4a2637f 569ff2f8dcbb09db 7dce843835bb5572 47636dce27e1c069 5f2ac92d1b0e7e41"),
    ("MASK", 0, 6, 0, 6): (3, "914b2c290b5df887 71b9e902bd9eb873 af1ef85cacade08b ab56d1f16da3e6ef 1a53f370c8538a93 b770854edbfc6ad0 2343fe75ff78e67a 5654b47ef10db825"),
    ("MASK", 255, 0, 0, 8): (0, "7c0d53fb99c4a7cb a8a0477d1ae578dc e1d8a3673be830e1 3f668bb656628cc5 2741bd7af6076deb ba490e9ca7ceec13 346a6296fa07534a 8f749f8150df3e33"),
    ("MASK", 1, 0, 0, 8): (0, "7b93ea1f2ddfbe14 7c413591cd609bb6 59130c1fc945431b f6fb6028d67ded5a 95d52faafcd67aa9 e38d3e192adc3317 748d6f5712a56ac9 eb841877d1c7f74b"),
    ("MASK", 1, 1, 0, 8): (0, "b8fe32c354f6bc66 d0e5084635b7cf3b 8fbf482a4c60053b 242d7fc67ed0046f d51159080902bf0e 1aa684af8970be7c c2ef16f732b2933b 3cb1269ccf8b7514"),
    ("MASK", 1, 0, 1, 8): (0, "795a1b0a16dbf252 66c44772985251ec a7191c8da328d842 9e849103685df522 bffa51baf4179216 ce79380ed900cf5c e39dc2909d602c61 cb287ace818fb400"),
    ("MASK", 1, 256, 0, 8): (0, "f0c1adf4389383ea 09f67ba2283b7467 66ee1b39b13879fe 96d9b5b1a8a260ac f5ef88e6f3949282 762340b66b435b76 806d7eb4e9e5dd01 00984e3b74acc1b9"),
    ("MASK", 1, 0, 65536, 8): (0, "4dcee7b17551dc53 b6bec09f3a5f1d0e c7b4e50cf3633f2e 9ce45cb0e02ea339 79aee156ff5f153a a0c6ae1d0eaf34a5 8f63514181fcb674 7c7d14187367ab29"),
    ("MASK", 1, 65536, 0, 8): (0, "8e30bdfba5c6ff84 6d6bc5891f560fe0 8b020ec15e709cd5 ced9ee13d23860f0 82895c580a6b5f8b c9e22736a0b11170 65d5f0ed442ed21a f5f0ddcaab00c1f3"),
    ("MASK", 1, 2047, 5505023, 8): (0, "31ada3751f23c39f 1b21c1e689eb37a0 1721f9efbad82ada 7da0608b4f4e5b74 d8bab20fec2227c8 4b0f1a5ada2dee6b 59b6f3d997f18147 74c907247d86f192"),
    ("MASK", 1, 8191, 5505023, 8): (0, "ad8ca6767ff6eb94 2f2ff42402b33bb3 21b9e630bfcbc9ac 8c4a75ddd053f81a 723dfbc14b084ade 01964ec3b13cab03 4573a228df42c5dd 57472bcbc9b0041f"),
    ("MASK", 1, 2047, 5505022, 7): (2, "a72824a6af8fd6fb 4ed7f5849866b4d8 d3558cf130b73200 9e5e0fce298bddbd 696cf46f5642f947 a293335265e29050 c323fef3d1d100cc c77ea8062df015e7"),
    ("MASK", 1, 2047, 5505020, 7): (1, "8d6f84c5c6c8de17 0ea1c63c318c8814 9c8be78537c3e6de 4ad0fe133ceb6ff9 0e2c440af3bc315a c0ec3f70471013e5 5b2ca073d43a3ac5 0e16b256d13cd65d"),
    ("MASK", 1, 235, 3890687, 8): (1, "e88fa190e8b72544 d93feb442c403780 d5fbf391c9e619a8 b0fe467a7a828c18 d64b78ff709e1812 c3738c0375c94958 376964af723caba4 198da622a8cbad32"),
    ("MASK", 1, 292, 5264827, 8): (1, "cddcfb68434c1081 bba4070d5e67d90b f288c70b1c4e7f23 4924fa59a91259e6 ef2c39855d42af32 dd5fc9f67000bea2 836a441e16f67a79 bc87f80414391420"),
    ("MASK", 1, 391, 1987782, 8): (1, "634023c494c9b60d 332ae5cfafcd67e6 de9455fe623215c1 267ed6297bfd349c 92c46a5297dff9cd ea39516316f4bc09 a821fff66fb92f20 b47d0a0e0973eada"),
    ("SALT", 0, 0): "dbf90a4d1108f198 69ba481bb13890bb 9f5ba572f66a3ced b36142df94e8d0d8 fe7b3d06bbdee6b6",
    ("SALT", 1, 0): "dca55308f7250e4f 9092d81a9d4057e9 a72c8c7d77a022cc bd3053c098f71acc 969f496996f6ad10",
    ("SALT", 1, 1): "2cfce91e068aaf42 4f6d5a1ac3c0087b 476dd89629266fb1 ec915950efd00797 e57dc2197cb59ed1",
    ("SALT", 1, 4294967295): "a5caa89f5acbff7a e9ebc82dbaf57357 a3ee12134b3238ab e550de77b742d4ec df3d896c35609018",
    ("SALT", 1, 4294967296): "ea595a5156bc953a 92a051050c88d00d 087ab2a255705480 8a542c27a4431d9b 947a59b846ce3ecb",
    ("SALT", 1, 4294967297): "6f3267028bb1b2be 61b85a65aa4cd906 8905f0c015ae3df4 8b94a9b7075463bb ec824b880a5693d8",
    ("SALT", 1, 8589934591): "999a80af389c572f e3f1262c45c1305a 5ddca460e6bc79df 0cc81c88bdf27e18 9e6e1d8215e27f4c",
    ("SALT", 255, 8589934591): "da0d70b8ad579bb1 f3281ca189d66ba6 e94732a5f64b3a2d 2e5a8dc71542db41 2937a769e0ee674a",
}  # fmt: skip


def known(field=1):
    return {1: KNOWN, 2: KNOWN2}[field]


def known_answers_hold(field=1):
    """The rows of a field's known answers that do not come out of this file's functions, as a list
    of their places; empty if all come out."""
    fails = []
    for at, want in known(field).items():
        if at[0] == "MASK":
            got = mask_block(test_seed(at[1]), at[2], at[3], at[4], field) == ([int(x, 16) for x in want[1].split()], want[0])
        else:
            got = salt(test_seed(at[1]), at[2], field) == [int(x, 16) for x in want.split()]
        if not got:
            fails.append(at)
    return fails


def made_rows(places, field):
    """The source text of rows of known answers at the given places, in the form of KNOWN's rows."""
    lines = []
    for at in places:
        if at[0] == "MASK":
            mu, t = mask_block(test_seed(at[1]), at[2], at[3], at[4], field)
            lines.append('    %r: (%d, "%s"),' % (at, t, " ".join("%016x" % x for x in mu)))
        else:
            lines.append('    %r: "%s",' % (at, " ".join("%016x" % x for x in salt(test_seed(at[1]), at[2], field))))
    return "\n".join(lines).replace("'", '"')


def header_tables(field):
    """A field's known answers in the form of the two tables of
    prover/hash_commitment/hf_mask_known.h, which has those of field 2."""
    hexes = lambda text: ", ".join("0x%sull" % x for x in text.split())
    masks = ["    {%d, %du, %du, %d, %du, {%s}}," % (at[1], at[2], at[3], at[4], want[0], hexes(want[1]))
             for at, want in known(field).items() if at[0] == "MASK"]  # fmt: skip
    salts = ["    {%d, %dull, {%s}}," % (at[1], at[2], hexes(want)) for at, want in known(field).items() if at[0] == "SALT"]
    return "\n".join(["constexpr HfKnownMasks HF_KNOWN_MASKS[] = {"] + masks + ["};", "constexpr HfKnownSalt HF_KNOWN_SALTS[] = {"] + salts + ["};"])


# ---- checking a log of a program's calls of f

# The rule as this file has it: a program calls f under kM on the two families of inputs at the top
# of this file and on no other. A table has a third family, "PADS"
# (docs/format/COIN-KEYS.md, section 2a), which this file does not have. findings() checks a log of
# the calls (its error message says "count") against the rule as this file has it.
# Nothing in this tree writes such a log.
# A line of a log: kind, key (hex_of() of its eight words, which is not the hex of its 32 bytes),
# x13, x14, x15, the first x12, one past the last x12, times: f was called `times` times at each
# x12 of that range. Kind "block": by the program's generator; "again": by its second function for
# the same values, which its own tests use.
LINE = re.compile(r"(block|again) ([0-9a-f]{64}) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")


def hex_of(words):
    return "".join("%08x" % w for w in words)


def read(text, kind="block"):
    """(the lines of one kind, as tuples of LINE's fields after the kind; the lines that do not
    match LINE)."""
    calls, odd = [], []
    for line in text.splitlines():
        m = LINE.fullmatch(line)
        if not m:
            odd.append(line)
        elif m[1] == kind:
            calls.append((m[2],) + tuple(int(x) for x in m.groups()[2:]))
    return calls, odd


# The inverse of read(): tuples of one kind as a log's lines.
def text_of(calls, kind="block"):
    return "".join(kind + " %s %d %d %d %d %d %d\n" % c for c in calls)


def known_inputs(field=1):
    """{a key's words in hex: the inputs (x12, x13, x14, x15) at which a program's self-test of its
    generator may call f under that key}: RFC 8439's vector under test key 0, and the rows of the
    field's known answers, each with the tries up to its try*. These keys (0, 1, 255) are public."""
    found = {hex_of(test_seed(0)): {(1, 0x09000000, 0x4A000000, 0)}}
    for at, want in known(field).items():
        inputs = found.setdefault(hex_of(test_seed(at[1])), set())
        if at[0] == "MASK":
            inputs |= {(at[2], at[3], t, MASK_WORD) for t in range(want[0] + 1)}
        else:
            inputs.add((at[2] % 2**32, at[2] >> 32, 0, SALT_WORD))
    return found


def findings(text, key, R, nm, n, masks, salts, k=8, reckoned=None, kind="block", field=2):
    """The failures of a log, as a list of strings.

    key: the run's key, eight words; test keys 0, 1 and 255 raise ValueError. R, nm, n: the table's
    sizes. masks: how often the run has to make each block (r, g), r < R, g < ceil(nm / 8): one
    number for all g, or {g: times}, where a g that is missing means never. salts: {I: times}, the
    leaves whose salts the run has to make; a range or list means once each. k: the digit of the
    run's threshold. field: the run's field. reckoned: the g at which try* is recomputed here, for
    all r, by mask_block(); None for all g. kind: which lines are checked. Lines of the program's
    self-test of its generator are skipped first: those under test key 0, 1 or 255 whose inputs are
    all among that key's in known_inputs(). Each failure begins with its rule's name:
      line    a line that does not match LINE
      form    an input that is neither (r, g, try, "MASK") with r < R and g < ceil(nm / 8) nor
              (I mod 2^32, floor(I / 2^32), 0, "SALT") with I < n, under any key
      key     a call under other key words than the run's key
      masks   try 0 is not made at all R rows of every block number g, each as often as `masks` says
      salts   the salts made are not those of exactly the leaves in `salts`, each as often as said
      tries   at a block (r, g) the tries made are not 0, 1, .., T without a gap, each as often as
              try 0
      least   at a g in `reckoned`, T is not try*: the program went on after a good block, or
              stopped at a bad one"""
    calls, odd = read(text, kind)
    G = blocks(nm)
    want_masks = dict(masks) if isinstance(masks, dict) else dict.fromkeys(range(G), masks)
    want_salts = dict(salts) if isinstance(salts, dict) else dict.fromkeys(salts, 1)
    fails = ["line: a line of no known form: " + x[:80] for x in odd]
    mine, made, salted = hex_of(key), {}, collections.Counter()  # made[g][try] = Counter {r: times}
    aside = known_inputs(field)
    if mine in aside:
        raise ValueError("a count is judged under a key other than test keys 0, 1 and 255")
    for s, x13, x14, word, first, behind, times in calls:
        if s in aside and behind - first <= len(aside[s]) and all((x12, x13, x14, word) in aside[s] for x12 in range(first, behind)):
            continue
        where = "f at (%d .. %d, %d, %d, 0x%08X), %d times" % (first, behind - 1, x13, x14, word, times)
        is_mask = word == MASK_WORD and behind <= R and x13 < G
        is_salt = word == SALT_WORD and x14 == 0 and behind - 1 + (x13 << 32) < n
        if not is_mask and not is_salt:
            fails.append("form: " + where)
        elif s != mine:
            fails.append("key: %s under the key words %s.." % (where, s[:16]))
        elif is_mask:
            at = made.setdefault(x13, {}).setdefault(x14, collections.Counter())
            for r in range(first, behind):
                at[r] += times
        else:
            for low in range(first, behind):
                salted[low + (x13 << 32)] += times
    for g in sorted(set(made) | set(want_masks)):
        want, have = want_masks.get(g, 0), made.get(g, {}).get(0, {})
        if dict(have) != (dict.fromkeys(range(R), want) if want else {}):
            fails.append("masks: block number %d: try 0 is expected %d times at each of the %d rows; seen at %d rows, %s times"
                         % (g, want, R, len(have), sorted(set(have.values()))[:3]))  # fmt: skip
            continue
        last, gaps = [0] * R, 0
        for t in sorted(made.get(g, {})):
            for r, times in made[g][t].items():
                if t and (last[r] != t - 1 or times != want):
                    gaps += 1
                    fails.append("tries: row %d, block number %d: try %d is made %d times, after try %d" % (r, g, t, times, last[r]))
                last[r] = t
        if want and not gaps and (reckoned is None or g in reckoned):
            least = [mask_block(key, r, g, k, field)[1] for r in range(R)]
            unlike = [r for r in range(R) if last[r] != least[r]]
            if unlike:
                fails.append("least: block number %d: at %d rows the last try is not the least good one; the first: row %d, try %d and not %d"
                             % (g, len(unlike), unlike[0], last[unlike[0]], least[unlike[0]]))  # fmt: skip
    unlike = sorted(I for I in set(salted) | set(want_salts) if salted.get(I, 0) != want_salts.get(I, 0))
    if unlike:
        fails.append("salts: at %d leaves the salt is not made as often as expected; the first: leaf %d, %d times and not %d"
                     % (len(unlike), unlike[0], salted.get(unlike[0], 0), want_salts.get(unlike[0], 0)))  # fmt: skip
    return fails


if __name__ == "__main__":
    good = rfc_vector_holds()
    print("mask_rules.py: the block of RFC 8439, 2.3.2, " + ("comes out" if good else "does NOT come out"))
    if sys.argv[1:2] == ["--make"]:  # field-2 rows at KNOWN's places; r:g adds MASK (1, r, g, 8)
        more = [("MASK", 1) + tuple(int(x) for x in word.split(":")) + (8,) for word in sys.argv[2:]]
        masks = [at for at in KNOWN if at[0] == "MASK"]
        print(made_rows(masks + more + [at for at in KNOWN if at[0] == "SALT"], 2))
        sys.exit(0)
    if sys.argv[1:2] == ["--header"]:  # the tables of hf_mask_known.h for one field
        print(header_tables(int(sys.argv[2])))
        sys.exit(0)
    for field in (1, 2):
        print("mask_rules.py: field %d: of the %d known answers %d do not come out" % (field, len(known(field)), len(known_answers_hold(field))))
    second = [at for at, want in KNOWN2.items() if at[0] == "MASK" and at[4] == 8 and want[0] > 0]
    print("mask_rules.py: field 2: %d known answers at the full threshold take a second try" % len(second))
    sys.exit(0 if good and not known_answers_hold(1) and not known_answers_hold(2) and len(second) >= 2 else 1)
