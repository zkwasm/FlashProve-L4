#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Expected values for the shape of a lookup forest, stored in verifier/tests/forest_values.txt and
verifier/tests/forest_weights.txt.

usage: forest_values.py                            prints the lines of forest_values.txt
       forest_values.py --weights                  prints the lines of forest_weights.txt
       forest_values.py [--weights] --hold <file>  regenerates the lines and compares them with the
                                                   file's; exit code 0 if equal, 1 if not
Other arguments give a usage message and exit code 2.

A lookup call is to show that each of its keys occurs in a public table. The calls of one group (a
sub-proof's, or an approximate range check's) run as one lookup: their keys are cut into trees of
equal size, and the trees of all calls are numbered. These trees are the forest
(verifier/src/vf_lookup_argument.h). Its shape is what the list of calls determines before any part of a
proof is read.

The formulas ("the clauses" in a message below) and the numbers stored in F1 come from a written
specification that is not part of this repository. The script shares no code with the verifier or
the prover.

Each file begins with a comment line ("#": --hold and the tests skip such lines). Every other line
begins with its kind (SHAPE, COUNTS, ..), which is explained at the function or loop below that
makes it. Two unit tests compare the verifier's functions with the files: for APUB and AREFUSE
verifier/tests/hc_forest_attention_test.cpp ("the attention test" below), for the rest a test that
is not part of this repository ("the shape test"). Limit: no test compares COUNTS, ARRAY or
LEDGER with anything that the verifier computes, and PLACE is compared only with a function that
the shape test itself defines.

After a change here, regenerate both files: the test `verifier.forest_values` of SUITE.md runs both
--hold forms. Both unit tests also expect a fixed number of lines of each kind that they read.
"""

import math
import sys

# P is the Goldilocks prime. LE is lg |E| rounded up, for the extension field E of emul(): three
# coordinates of 64 bits.
P = 2**64 - 2**32 + 1
LE = 192
ONE = (1, 0, 0)


def emul(a, b):
    """The product in E = F[x] / (x^3 - 2), where F is the integers mod P."""
    return (
        (a[0] * b[0] + 2 * (a[1] * b[2] + a[2] * b[1])) % P,
        (a[0] * b[1] + a[1] * b[0] + 2 * a[2] * b[2]) % P,
        (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % P,
    )


def esub(a, b):
    return tuple((x - y) % P for x, y in zip(a, b))


def syn(k):
    """Synthetic input number k."""
    return ((k + 1) * 0x9E3779B97F4A7C15 % 2**64) % P


# A list of synthetic elements of E.
def made_e(seed, count):
    return [(syn(seed + 3 * k), syn(seed + 3 * k + 1), syn(seed + 3 * k + 2)) for k in range(count)]


def eq(pt, j):
    """eq(pt, j): coordinate 0 of pt goes with the most significant of the len(pt) bits of j."""
    b, r = len(pt), ONE
    for t in range(b):
        x = pt[b - 1 - t]
        r = emul(r, x if j >> t & 1 else esub(ONE, x))
    return r


def he(a):
    """An element of E as text: its three coefficients, 16 hex digits each, joined by ":"."""
    return ":".join("%016x" % x for x in a)


def clog2(v):
    """The smallest r with 2^r >= v; clog2(1) = 0."""
    return (v - 1).bit_length()


# SHAPE: the expected result of forestShape(). Call c of the list has 2^n_c keys and a table of
# 2^k_c rows. lid is lidb, the line of the parameter file (policy.txt) that caps dT:
# dT = min(lidb, min n_c). kT = max k_c. Call c has I_c = 2^iota_c trees of 2^dT keys, where
# iota_c = n_c - dT. NT = sum I_c is the number of trees, and rK = clog2(NT). The trees are numbered
# call by call, the calls with the most trees first, ties in the list's order. first_c is call c's
# first tree number, and pre_c = first_c / I_c. Tree number y = first_c + b is tree b of call c:
# callOf[y] = c and treeOf[y] = b. The numbers NT .. 2^rK - 1 are empty: they belong to no call.
# iota, first and pre are in the list's order.
def shape(lid, n, k):
    d_t, k_t = min([lid] + n), max(k)
    iota = [x - d_t for x in n]
    size = [1 << i for i in iota]
    order = sorted(range(len(n)), key=lambda c: -size[c])  # Python's sort is stable
    first, at = [0] * len(n), 0
    for c in order:
        first[c] = at
        at += size[c]
    call_of, tree_of = [0] * at, [0] * at
    for c in range(len(n)):
        assert first[c] % size[c] == 0  # I_c divides first_c
        for b in range(size[c]):
            call_of[first[c] + b], tree_of[first[c] + b] = c, b
    return dict(dT=d_t, kT=k_t, NT=at, rK=clog2(at), iota=iota, first=first, pre=[first[c] // size[c] for c in range(len(n))],
                callOf=call_of, treeOf=tree_of)  # fmt: skip


# COUNTS: the specification's counts for one group. The verifier's counts, forestElems() and
# forestLins(), follow other formulas, which the shape test states itself.
# roots, key and table count message slots (masked elements that the prover sends): those of the
# trees' roots and of the product layers over the keys and over the tables' rows. A layer of mu
# rounds has 2 mu + 2 slots; the last two layers in table are the leaf layer and the bit layer.
# arrays and rootRounds are the slots of the root test, the specification's test on the trees'
# roots. open1, rel1 and lin1 count the claims of the types OPEN, REL and LIN before that test,
# openRoot, relRoot and linRoot those in it. slots1 is all slots but the arrays, with one more for
# each OPEN and two for each REL; slots2 adds the arrays. rounds counts sumcheck rounds, singles
# the single challenges (see LEDGER in weights_lines()).
def counts(s):
    d_t, k_t, nt, r_k = s["dT"], s["kT"], s["NT"], s["rK"]
    roots = 33 * nt
    key = sum(2 * mu + 2 for mu in range(r_k, r_k + d_t))
    table = sum(2 * mu + 2 for mu in range(r_k + 5, r_k + 5 + k_t)) + 2 * (2 * (r_k + 5 + k_t) + 2)
    e_f = roots + key + table
    rounds = sum(range(r_k, r_k + d_t)) + sum(r_k + 5 + l for l in range(k_t)) + 2 * (k_t + r_k + 5) + r_k
    assert e_f + 2 * r_k == 33 * nt + 2 * rounds + 2 * (d_t + k_t + 2)  # a cross-check
    open1, rel1 = 2 * (d_t + k_t + 2), d_t + k_t + 2
    slots1 = e_f + 2 * r_k + open1 + 93 + 2 * (rel1 + 1)
    return dict(roots=roots, key=key, table=table, eF=e_f, rootRounds=2 * r_k, arrays=62 * nt, open1=open1, rel1=rel1, lin1=2,
                openRoot=93, relRoot=1, linRoot=1, rounds=rounds, singles=4 * r_k + 10 + 2 * k_t + d_t + 2, slots1=slots1,
                slots2=slots1 + 62 * nt)  # fmt: skip


# ORDER: the expected result of orderForestRange(), the order in which the last product layer over
# the keys binds its mu = rK + dT - 1 index bits (bit positions 0 .. mu - 1). Bits rK and above come
# first, the rK bits of the tree number last. The formula requires dT >= 10: not checked here.
def forest_order(mu, r_k):
    low = [r_k + v for v in (2, 1, 0, 5, 4, 3, 8, 7, 6)]
    return low + list(range(mu - 1, r_k + 8, -1)) + list(range(r_k - 1, -1, -1))


# ORDER.WRONG: the order for a single lookup call at the same mu, the expected result of
# orderRange() (verifier/src/vf_common.h). A program must not produce it for a forest with rK > 0.
def range_order(mu):
    return [2, 1, 0, 5, 4, 3, 8, 7, 6] + list(range(mu - 1, 8, -1))


# PUBRUN: the expected result of forestPubRun() for a call whose padding keys are one run: those
# with the indices [tot, 2^n). Padding keys are fill; the multiplicities (how many keys hit each
# table row) do not count them. Tree b has the keys [b 2^dT, (b + 1) 2^dT). Returns the number of
# padding keys in each tree. PUBRUN and PUBROWS test forestPubRun() by itself; of the numbers that
# the verifier derives with it for a group, the files have only attention's (APUB).
def pub_run(n, d_t, tot):
    return [max(0, min((b + 1) << d_t, 1 << n) - max(b << d_t, tot)) for b in range(1 << (n - d_t))]


# F1: rows copied from table F1 of the specification, as printed there, one for each group of five
# configurations: configuration, group, lidb of its parameter file (22 if absent), n_c, k_c;
# then the expected dT, kT, NT, rK, first_c, roots, key, table, rootRounds, eF + rootRounds,
# open1 + openRoot, rel1 + relRoot, slots1, rounds. lines() exits if shape() and counts() give
# other numbers. The group "feed-forward" is the layer sub-proof's and "head" the output-head
# sub-proof's; a group whose name begins with "section-9" is a range check's.
FF16 = [16, 16, 16, 8, 16, 16, 16, 16]
HD16 = [16, 16, 8, 16, 16, 16]
F1 = (
    ("toy", "feed-forward", 22, [13, 14, 14, 15, 13, 13, 13, 13], FF16, 13, 16, 13, 4, [8, 4, 6, 0, 9, 10, 11, 12], 429, 286, 664, 8, 1387, 155, 32, 1606, 448),
    ("toy", "section-9", 22, [21, 21], [10, 8], 21, 10, 2, 1, [0, 1], 66, 504, 298, 2, 870, 159, 34, 1097, 369),
    ("toy", "head", 22, [13] * 6, HD16, 13, 16, 6, 3, [0, 1, 2, 3, 4, 5], 198, 260, 628, 6, 1092, 155, 32, 1311, 416),
    ("profile4", "feed-forward", 22, [13, 14, 14, 15, 13, 13, 13, 13], FF16, 13, 16, 13, 4, [8, 4, 6, 0, 9, 10, 11, 12], 429, 286, 664, 8, 1387, 155, 32, 1606, 448),
    ("profile4", "section-9", 22, [21, 21], [10, 8], 21, 10, 2, 1, [0, 1], 66, 504, 298, 2, 870, 159, 34, 1097, 369),
    ("profile4", "head", 22, [13] * 6, HD16, 13, 16, 6, 3, [0, 1, 2, 3, 4, 5], 198, 260, 628, 6, 1092, 155, 32, 1311, 416),
    ("profile4", "attention", 22, [13] * 4, [16, 13, 16, 16], 13, 16, 4, 2, [0, 1, 2, 3], 132, 234, 592, 4, 962, 155, 32, 1181, 384),
    ("profile4", "section-9-attention", 22, [20, 20], [10, 8], 20, 10, 2, 1, [0, 1], 66, 460, 298, 2, 826, 157, 33, 1049, 348),
    ("profile4-12", "feed-forward", 22, [13, 14, 14, 15, 13, 12, 12, 12], FF16, 12, 16, 23, 5, [16, 8, 12, 0, 18, 20, 21, 22], 759, 276, 700, 10, 1745, 153, 31, 1960, 463),
    ("profile4-12", "section-9-feed-forward", 22, [22, 22], [10, 8], 22, 10, 2, 1, [0, 1], 66, 550, 298, 2, 916, 161, 35, 1147, 391),
    ("profile4-12", "section-9-head", 22, [21, 21], [10, 8], 21, 10, 2, 1, [0, 1], 66, 504, 298, 2, 870, 159, 34, 1097, 369),
    ("profile4-12", "head", 22, [13, 12, 13, 12, 12, 12], HD16, 12, 16, 8, 3, [0, 4, 2, 5, 6, 7], 264, 228, 628, 6, 1126, 153, 31, 1341, 401),
    ("profile4-12", "attention", 22, [12, 12, 13, 12], [16, 13, 16, 16], 12, 16, 5, 3, [2, 3, 0, 4], 165, 228, 628, 6, 1027, 153, 31, 1242, 401),
    ("profile4-12", "section-9-attention", 22, [20, 20], [10, 8], 20, 10, 2, 1, [0, 1], 66, 460, 298, 2, 826, 157, 33, 1049, 348),
    ("small_both", "feed-forward", 24, [20, 20, 20, 21, 20, 20, 20, 20], FF16, 20, 16, 9, 4, [2, 3, 4, 0, 5, 6, 7, 8], 297, 580, 664, 8, 1549, 169, 39, 1796, 588),
    ("small_both", "section-9", 24, [24, 24], [10, 8], 24, 10, 2, 1, [0, 1], 66, 648, 298, 2, 1014, 165, 37, 1253, 438),
    ("small_both", "head", 24, [20] * 6, HD16, 20, 16, 6, 3, [0, 1, 2, 3, 4, 5], 198, 540, 628, 6, 1372, 169, 39, 1619, 549),
    ("small_both", "attention", 24, [20] * 4, [16] * 4, 20, 16, 4, 2, [0, 1, 2, 3], 132, 500, 592, 4, 1228, 169, 39, 1475, 510),
    ("small_both", "section-9-attention", 24, [23, 23], [10, 8], 23, 10, 2, 1, [0, 1], 66, 598, 298, 2, 964, 163, 36, 1199, 414),
    ("full_prompt", "feed-forward", 28, [27, 27, 27, 28, 27, 27, 26, 26], FF16, 26, 16, 16, 4, [4, 6, 8, 0, 10, 12, 14, 15], 528, 910, 664, 8, 2110, 181, 45, 2381, 747),
    ("full_prompt", "section-9", 28, [27, 27], [10, 8], 27, 10, 2, 1, [0, 1], 66, 810, 298, 2, 1176, 171, 40, 1427, 516),
    ("full_prompt", "head", 28, [28, 28, 28, 27, 27, 27], HD16, 27, 16, 9, 4, [0, 2, 4, 6, 7, 8], 297, 972, 664, 8, 1941, 183, 46, 2216, 777),
    ("full_prompt", "attention", 28, [27, 27, 26, 25], [16] * 4, 25, 16, 11, 4, [0, 4, 8, 10], 363, 850, 664, 8, 1885, 179, 44, 2152, 718),
    ("full_prompt", "section-9-attention", 28, [26, 26], [10, 8], 26, 10, 2, 1, [0, 1], 66, 754, 298, 2, 1120, 169, 39, 1367, 489),
)  # fmt: skip

# Synthetic lists: name, lidb, n_c, k_c. forestShape() must reject three, each with the rejection
# name in quotation marks on its line. The SHAPE lines do not have the names: the shape test derives
# the expected name from a line's number of calls and its NT.
MADE_UP = (
    ("one-call", 22, [17], [8]),  # NT = 1, rK = 0
    ("one-call-of-eight-trees", 22, [25], [16]),
    ("all-alike", 22, [15] * 5, [5, 16, 9, 16, 7]),  # ties throughout: the list's order stays; three empty numbers
    ("rising", 30, [10, 11, 12, 13, 14], [8] * 5),  # the sort reverses the list; the smallest depth, dT = 10
    ("ties-among-the-large", 22, [14, 16, 15, 16, 14, 15, 16], [16, 8, 16, 8, 16, 8, 16]),
    ("cap-below-all", 23, [30, 28, 29], [16, 16, 16]),  # lidb sets dT: no call has one tree
    ("cap-above-all", 30, [24, 26], [10, 8]),
    ("sixteen-calls", 22, [12] * 16, [16] * 16),  # 16 calls, the maximum; no empty number
    ("seventeen-calls", 22, [12] * 17, [16] * 17),  # rejected: "forest-calls"
    ("trees-256", 22, [29, 28, 27, 26, 25, 24, 23, 22, 22], [16] * 9),  # NT = 256, the maximum
    ("trees-257", 22, [29, 28, 27, 26, 25, 24, 23, 22, 22, 22], [16] * 10),  # rejected: "forest-trees"
    ("trees-256-alike", 22, [26] * 16, [16] * 16),
    ("widest", 30, [62, 55], [16, 5]),  # n_c = 62, the maximum; 2^32 + 2^25 trees: "forest-trees"
)


# A list as text: its items joined by commas, or "-" if it is empty.
def comma(v):
    return ",".join(map(str, v)) if len(v) else "-"


def pairs(d, names):
    return " ".join("%s=%s" % (x, comma(d[x]) if isinstance(d[x], list) else d[x]) for x in names)


# The lines of forest_values.txt. A line is "<kind> <name>=<value> .. -> <answer>". The first, FORM,
# has dw=0: no line depends on a digest's width. The shape test requires this value (knownForm() in
# verifier/tests/flat_known.h).
def lines():
    out = ["FORM dw=0 -> no line hangs on a digest's width: one file for both builds"]
    groups = []
    for r in F1:
        name, lid, n, k = r[0] + "." + r[1], r[2], r[3], r[4]
        s = shape(lid, n, k)
        c = counts(s)
        mine = (s["dT"], s["kT"], s["NT"], s["rK"], s["first"], c["roots"], c["key"], c["table"], c["rootRounds"], c["eF"] + c["rootRounds"],
                c["open1"] + c["openRoot"], c["rel1"] + c["relRoot"], c["slots1"], c["rounds"])  # fmt: skip
        if mine != tuple(r[5:]):
            sys.exit("forest_values.py: the clauses do not give F1's row %s: %s against %s" % (name, mine, r[5:]))
        groups.append((name, lid, n, k))
    groups += MADE_UP
    for name, lid, n, k in groups:
        head = "name=%s lid=%d n=%s k=%s" % (name, lid, comma(n), comma(k))
        d_t = min([lid] + n)
        if sum(1 << (x - d_t) for x in n) > 1 << 12:  # too many trees to list: only the numbers
            nt = sum(1 << (x - d_t) for x in n)
            out.append("SHAPE %s -> dT=%d kT=%d NT=%d rK=%d iota=%s first=- pre=- callOf=- treeOf=-" % (head, d_t, max(k), nt, clog2(nt), comma([x - d_t for x in n])))
            continue
        s = shape(lid, n, k)
        out.append("SHAPE %s -> %s" % (head, pairs(s, ("dT", "kT", "NT", "rK", "iota", "first", "pre", "callOf", "treeOf"))))
        out.append("COUNTS %s -> %s" % (head, pairs(counts(s), ("roots", "key", "table", "eF", "rootRounds", "arrays", "open1", "rel1", "lin1", "openRoot",
                                                                 "relRoot", "linRoot", "rounds", "singles", "slots1", "slots2"))))  # fmt: skip
    # ORDER and ORDER.WRONG. The cases (dT, rK) include the smallest depth, six of F1's shapes,
    # rK = 0 and rK = 8, the largest.
    for d_t, r_k in ((10, 0), (10, 1), (10, 3), (11, 2), (13, 4), (12, 5), (21, 1), (26, 4), (27, 4), (25, 4), (30, 8), (22, 8)):
        mu = r_k + d_t - 1
        order = forest_order(mu, r_k)
        assert sorted(order) == list(range(mu))  # a permutation of the layer's bit positions
        out.append("ORDER mu=%d rK=%d -> %s" % (mu, r_k, comma(order)))
        assert (order == range_order(mu)) == (r_k == 0)
        out.append("ORDER.WRONG mu=%d rK=%d -> %s" % (mu, r_k, comma(range_order(mu))))
    # ARRAY: the positions of entry y of the root test's arrays Q_j and P_j, j < 31, among their
    # 62 NT slots.
    for nt in (1, 2, 5, 13, 16, 256):
        seen = set()
        for j in range(31):
            for y in range(nt):
                seen |= {2 * (30 - j) * nt + y, (2 * (30 - j) + 1) * nt + y}
        assert seen == set(range(62 * nt))  # the positions tile the 62 NT slots
        for j, y in ((30, 0), (30, nt - 1), (29, 0), (7, nt // 2), (0, 0), (0, nt - 1)):
            out.append("ARRAY NT=%d j=%d y=%d -> Q=%d P=%d len=%d" % (nt, j, y, 2 * (30 - j) * nt + y, (2 * (30 - j) + 1) * nt + y, 62 * nt))
    # PLANES, PLACE: the committed matrix of the multiplicities' bits has 2^lm rows, where
    # 5 <= lm <= kT + 5. PLANES gives its number of columns, the expected result of
    # forestPlaneCols(). PLACE gives the storage index S, the row and the column of bit j < 32 of
    # the multiplicity of table row t in multiplicity vector y. The specification has one such
    # vector for each tree, the verifier one for each call: the shape test passes NT as the number
    # of calls, and y is a call's number.
    # The first twelve cases have an expected number of columns, copied from the same specification.
    # NT is F1's for feed-forward, attention and head of profile4, profile4-12, small_both and
    # full_prompt, and lm is the parameter line lmb, lmbA or lmbH. The last four are synthetic, at
    # both ends of lm's range, with no stored number (0).
    for k_t, nt, lm, printed in ((16, 13, 12, 6656), (16, 4, 11, 4096), (16, 6, 12, 3072), (16, 23, 13, 5888), (16, 5, 11, 5120), (16, 8, 12, 4096),
                                 (16, 9, 13, 2304), (16, 4, 13, 1024), (16, 6, 13, 1536), (16, 16, 16, 512), (16, 11, 14, 1408), (16, 9, 15, 576),
                                 (10, 2, 5, 0), (10, 2, 15, 0), (5, 1, 10, 0), (16, 256, 21, 0)):  # fmt: skip
        cols = nt << (k_t + 5 - lm)
        assert not printed or cols == printed, (k_t, nt, lm, cols, printed)
        out.append("PLANES kT=%d NT=%d lm=%d -> cols=%d" % (k_t, nt, lm, cols))
        for t, y, j in ((0, 0, 0), (0, 0, 31), (1, 0, 0), ((1 << k_t) - 1, 0, 5), (0, nt - 1, 0), ((1 << k_t) - 1, nt - 1, 31), ((1 << k_t) // 3, nt // 2, 17)):
            s = ((y << k_t) + t) * 32 + j
            row, col = s % (1 << lm), s >> lm
            # a second formula for the same row and column
            assert row == (t % (1 << (lm - 5))) * 32 + j and col == (y << (k_t + 5 - lm)) + (t >> (lm - 5))
            out.append("PLACE kT=%d NT=%d lm=%d t=%d y=%d j=%d -> S=%d row=%d col=%d" % (k_t, nt, lm, t, y, j, s, row, col))
    # PUBRUN. The first case is attention's call KR at full size: tot = 2.8125 x 2^25 keys in four
    # trees of 2^25, of which the last has only padding keys.
    for n, d_t, tot in ((27, 25, 94371840), (13, 12, 5000), (13, 12, 4096), (13, 12, 4095), (13, 12, 8192), (13, 12, 1), (12, 12, 4000), (12, 12, 4096),
                        (15, 12, 12288), (15, 12, 12289), (15, 12, 20000), (16, 10, 1025)):  # fmt: skip
        p = pub_run(n, d_t, tot)
        assert sum(p) == (1 << n) - tot
        out.append("PUBRUN n=%d dT=%d tot=%d -> %s" % (n, d_t, tot, comma(p)))
    return out


def pub_by_places(s, d_t, pads):
    """PUBROWS, by the definition itself: for y = first_c + b, pub_y is the number of call c's
    padding keys whose index i has floor(i / 2^dT) = b, counted one by one. pads[c] lists the
    indices of call c's padding keys, none twice."""
    pub = [0] * s["NT"]
    for c, places in pads.items():
        for i in places:
            pub[s["first"][c] + (i >> d_t)] += 1
    return pub


def attention_lines():
    """The APUB lines, then the AREFUSE lines: what aforkShapeHid() (verifier/src/vf_attention_check.h) must
    derive from a configuration of the attention sub-proof, or the name with which it must reject.

    APUB: each lookup segment (a named list of keys) has HA pieces of 2^bb keys for every layer. A
    call of kind KR or KFD becomes one call for each shift value, in rising order, with the pieces
    of the layers that have this shift. tot is the number of keys in a call's pieces,
    n_c = max(clog2(tot), nmin), and the keys from index tot on are padding keys. pub_y is the
    number of padding keys in tree y."""
    out, in_f1 = [], 0
    # A case's fields are the parameter lines of these names, except: lgk = log2 G, ldh = log2 dh,
    # ha = HA = H / G (query heads for each key-value head), lid is lidb, and sh is not the line
    # `sh` but the attention shift of each layer of the sub-proof (stored in `KL.bin`). The
    # attention test accepts only a power of two as the number of layers. other is the lines rx,
    # lmS, la, lmbA, lmh, lmo: only lmbA is used here; the attention test passes all six to
    # aforkShapeHid(), whose other checks fail without them. calls is the `callA` lines: name,
    # kind, lookup segments.
    # The seven segments. A piece has the keys of G heads; for each head, with T = 2^lt tokens:
    #   RM, FM  one key for each of the T^2 / 2 places into which the score matrix's tiles
    #           (2^lB x 2^lB) below the diagonal are packed; RM's table has the kind KR, FM's KFD
    #   RD, FD  the same for the T 2^lB places of the diagonal tiles
    #   WL      two keys for each of the T dh output entries, each a rounded quotient: the low
    #           16 bits of the remainder and of (divisor - 1 - remainder)
    #   WH      one key for each output entry: the high byte of each of these two numbers
    #   IX      one key for each token i: 2^(16 - lt) (i - position of row i's largest score)
    reference = (("KR", "KR", ("RM", "RD")), ("KFD", "KFD", ("FM", "FD")), ("R16", "R16", ("WL", "IX")), ("WH", "WH", ("WH",)))
    small, full = (5, 7, 6, 11, 7, 8), (8, 16, 16, 14, 13, 16)
    for name, lt, l_b, lgk, ldh, ha, nmin, lid, kfd, sh, other, calls in (
        ("profile4", 4, 2, 1, 4, 2, 13, 22, 13, (8, 8), small, reference),
        ("profile4-12", 4, 2, 1, 4, 2, 12, 22, 13, (8, 8), small, reference),
        ("profile4-11", 4, 2, 1, 4, 2, 11, 22, 13, (8, 8), small, reference),  # WH's tot is 2^11 = 2^nmin: no padding key in the call
        ("small_both", 6, 4, 1, 7, 2, 20, 24, 16, (8, 8, 8, 8), (8, 11, 8, 13, 12, 13), reference),
        ("full_prompt", 10, 6, 3, 7, 5, 20, 28, 16, (8, 8, 8, 8), full, reference),
        # Synthetic. Two shift values give six calls: KR8 and KFD8 have the keys of the one layer
        # with the shift 8, KR9 and KFD9 those of the three layers with 9.
        ("two-shifts", 10, 6, 3, 7, 5, 20, 28, 16, (9, 8, 9, 9), full, reference),
        # Full size with lidb = 24, which is below every n_c and so determines dT.
        ("lidb-binds", 10, 6, 3, 7, 5, 20, 24, 16, (8, 8, 8, 8), full, reference),
        # IX in a separate call, with nmin raised to 24 to keep the number of trees small. The calls
        # are not listed in the order of their trees.
        ("IX-by-itself", 10, 6, 3, 7, 5, 24, 28, 16, (8, 8, 8, 8), full,
         (("X", "R16", ("IX",)), ("KR", "KR", ("RM", "RD")), ("WH", "WH", ("WH",)), ("KFD", "KFD", ("FD", "FM")), ("R16", "R16", ("WL",)))),
    ):  # fmt: skip
        bb = dict(RM=lgk + 2 * lt - 1, FM=lgk + 2 * lt - 1, RD=lgk + lt + l_b, FD=lgk + lt + l_b, WL=1 + lgk + lt + ldh, WH=lgk + lt + ldh,
                  IX=lgk + lt)  # fmt: skip
        names, tot, k = [], [], []
        for line, kind, segments in calls:
            shifted = kind in ("KR", "KFD")
            for value in sorted(set(sh)) if shifted else (None,):
                layers = sum(1 for x in sh if x == value) if shifted else len(sh)
                names.append(line + (str(value) if shifted and len(set(sh)) > 1 else ""))
                tot.append(sum(layers * ha << bb[s] for s in segments))
                k.append(kfd if kind == "KFD" else 16)
        n = [max(clog2(t), nmin) for t in tot]
        s = shape(lid, n, k)
        pub = [0] * s["NT"]
        for c in range(len(n)):
            for b, v in enumerate(pub_run(n[c], s["dT"], tot[c])):
                pub[s["first"][c] + b] = v
        if name == "full_prompt":  # the specification's numbers at full size
            assert tot == [94371840, 94371840, 42106880, 20971520] and n == [27, 27, 26, 25] and s["first"] == [0, 4, 8, 10]
            assert pub == [0, 0, 6291456, 33554432, 0, 0, 6291456, 33554432, 0, 25001984, 12582912] and (s["dT"], s["NT"], s["rK"]) == (25, 11, 4)
        for r in F1:  # F1's rows for attention
            if (r[0].split(" ")[0], r[1]) == (name, "attention"):
                assert n == r[3] and k == r[4], (name, n, k)
                in_f1 += 1
        # columns: those of bA, attention's matrix of the multiplicities' bits (see PLANES), with
        # lm = lmbA. The group's lookup has one table (blocks/iop/lkp.h): its row (t | c) is row
        # t < 2^kT of call c's table and has a multiplicity of 32 bits. Only the rows of the NC
        # calls are stored, which makes NC 2^(kT + 5 - lm) columns.
        cols = len(names) << (s["kT"] + 5 - other[3])
        # printed: an expected number of columns (None: the specification has no number by call) and
        # the parameter line iotaA_0 = 32 + dT.
        printed = {"profile4": (None, 45), "profile4-12": (None, 44), "small_both": (None, 52), "full_prompt": (None, 57), "profile4-11": (None, 43)}
        if name in printed:
            assert printed[name][0] in (None, cols) and printed[name][1] == 32 + s["dT"], (name, cols, s["dT"])
        lines = ";".join("%s:%s:%s" % (line, kind, "+".join(segments)) for line, kind, segments in calls)
        out.append("APUB name=%s lt=%d lB=%d lgk=%d ldh=%d HA=%d nmin=%d lid=%d kFD=%d sh=%s other=%s list=%s -> calls=%s tot=%s n=%s k=%s dT=%d kT=%d NT=%d first=%s columns=%d pub=%s" % (
            name, lt, l_b, lgk, ldh, ha, nmin, lid, kfd, comma(sh), comma(other), lines, ",".join(names), comma(tot), comma(n), comma(k), s["dT"], s["kT"], s["NT"], comma(s["first"]), cols, comma(pub)))  # fmt: skip
    assert in_f1 == 4  # all four of F1's rows for attention were compared
    # AREFUSE: configurations that aforkShapeHid() must reject, with the rejection name.
    alone = (("KR", "KR", ("RM", "RD")), ("KFD", "KFD", ("FM", "FD")), ("R16", "R16", ("WL",)), ("X", "R16", ("IX",)), ("WH", "WH", ("WH",)))
    for name, lt, l_b, lgk, ldh, ha, nmin, lid, kfd, sh, other, calls, refusal in (
        # IX in a separate call has n_c = nmin = 20, so dT = 20 and there are
        # 128 + 128 + 64 + 1 + 32 = 353 trees.
        ("353-trees", 10, 6, 3, 7, 5, 20, 28, 16, (8, 8, 8, 8), full, alone, "forest-trees"),
        # dT = 10 is below lt + ldh = 11.
        ("a-tree-shorter-than-a-row-of-a-head", 4, 2, 0, 7, 1, 10, 22, 16, (8, 8), (5, 6, 6, 13, 7, 8), reference, "forest-depth-A"),
        # The valid range is 5 <= lmbA <= kT + 5 = 21.
        ("lmbA-4", 4, 2, 1, 4, 2, 13, 22, 13, (8, 8), (5, 7, 6, 4, 7, 8), reference, "forest-planes"),
        ("lmbA-22", 4, 2, 1, 4, 2, 13, 22, 13, (8, 8), (5, 7, 6, 22, 7, 8), reference, "forest-planes"),
        # Seven shift values give 16 calls: the forest's limit of sixteen allows them, attention's
        # limit of twelve does not, and it has a separate name. verifier/src/cp_verify.cpp has the
        # same limit under a third name, "policy-call-list-A".
        ("sixteen-calls", 10, 6, 3, 7, 5, 20, 28, 16, (4, 5, 6, 7, 8, 9, 10, 10), full, reference, "af-policy-call-list"),
    ):  # fmt: skip
        lines = ";".join("%s:%s:%s" % (line, kind, "+".join(segments)) for line, kind, segments in calls)
        out.append("AREFUSE name=%s lt=%d lB=%d lgk=%d ldh=%d HA=%d nmin=%d lid=%d kFD=%d sh=%s other=%s list=%s -> %s" % (
            name, lt, l_b, lgk, ldh, ha, nmin, lid, kfd, comma(sh), comma(other), lines, refusal))  # fmt: skip
    return out


# The lines of forest_weights.txt. After the comment line, the first is the same FORM line.
def weights_lines():
    out = ["FORM dw=0 -> no line hangs on a digest's width: one file for both builds"]
    # PHIB: the expected result of forestPhiB(),
    #   phi_e(row) = eq((last lm - 5 coordinates of zt | zj), row),
    # the weight of a row of PLANES' matrix M in the evaluation table
    # TB_e[col] = sum over the rows of phi_e(row) M[row][col]. The point zb_e = (zt | zy | zj) has
    # kT, rK and 5 coordinates, for t, y and j, and is synthetic here. The verifier has two such
    # points, e = 0 and 1, and the shape test checks each line at both. As in PLANES, y is a call's
    # number for the shape test. The cases: all 32 values of j at the first, a middle and the last
    # value of the row's other bits; lm at both ends of its range and between.
    for k_t, r_k, lm, seed in ((6, 2, 5, 7500000), (6, 2, 8, 7501000), (6, 2, 11, 7502000), (5, 0, 7, 7503000), (16, 4, 12, 7504000)):
        zb = made_e(seed, k_t + r_k + 5)
        zt, zj = zb[:k_t], zb[k_t + r_k :]
        point = zt[k_t - (lm - 5) :] + zj if lm > 5 else zj
        assert len(point) == lm
        top = (1 << (lm - 5)) - 1
        for low in sorted({0, top // 3, top}):
            for j in range(32):
                out.append("PHIB kT=%d rK=%d lm=%d seed=%d row=%d -> %s" % (k_t, r_k, lm, seed, low * 32 + j, he(eq(point, low * 32 + j))))
    # PHIP: the expected result of forestPhiP(), the same for the table TP: 2^(row mod 32).
    for lm in (5, 6, 8):
        out.append("PHIP lm=%d -> %s" % (lm, comma([1 << (row % 32) for row in range(1 << lm)])))
    # PUBROWS: the expected result of forestPubRun() called run by run. A case: name, lidb, n_c,
    # k_c, then for each named call: start, stride, length, number, tail. The call's padding keys
    # are "number" runs of "length" keys, run r beginning at index start + r stride, and all keys
    # from index "tail" on. The range check's padding keys have this form: the end of every row of
    # every piece, then all keys from the end of the last piece on.
    for name, lid, n, k, runs in (
        # The calls are not listed in the order of their trees (first = 24, 0, 16). Rows of 51 and
        # of 100 keys do not divide a tree, so runs cross from one tree into the next: three in
        # call 0, two in call 2.
        ("list-order-is-not-the-trees", 10, [12, 14, 13], [8, 8, 8], {0: (37, 51, 30, 65, 3400), 2: (5, 100, 60, 30, 8000)}),
        # Two calls of one size, so first_1 = 2^iota, as in the range check. The padding keys are
        # the end of every row of 64 keys, the same in both calls.
        ("two-calls-alike", 10, [13, 13], [10, 8], {0: (50, 64, 14, 96, 6144), 1: (50, 64, 14, 96, 6144)}),
        ("a-run-over-three-trees", 10, [11, 12], [8, 8], {1: (500, 4096, 2000, 1, 4000), 0: (1000, 512, 100, 2, 2048)}),
        ("no-tail-and-no-run", 10, [11, 11], [8, 8], {0: (0, 1, 0, 0, 2048), 1: (1023, 1, 2, 1, 2048)}),
        ("every-place", 10, [10, 11], [8, 8], {0: (0, 1, 0, 0, 0)}),
    ):
        s = shape(lid, n, k)
        pads, fields = {}, []
        for c in sorted(runs):
            start, stride, length, number, tail = runs[c]
            places = [start + r * stride + x for r in range(number) for x in range(length)] + list(range(tail, 1 << n[c]))
            assert len(set(places)) == len(places) and all(0 <= i < 1 << n[c] for i in places)  # no index twice
            pads[c] = places
            fields.append((c, start, stride, length, number, tail))
        out.append("PUBROWS name=%s lid=%d n=%s k=%s call=%s start=%s stride=%s length=%s number=%s tail=%s -> %s" % (
            (name, lid, comma(n), comma(k)) + tuple(comma([x[i] for x in fields]) for i in range(6)) + (comma(pub_by_places(s, s["dT"], pads)),)))  # fmt: skip
    out += attention_lines()
    # LEDGER, for every group that is not rejected: the specification's two error terms of a forest,
    # in bits, with LE = 192 for lg |E|. identity = LE - dT. singles = LE - log2(number of single
    # challenges), each of which errs with probability 1 / |E|.
    for name, lid, n, k in [(r[0] + "." + r[1], r[2], r[3], r[4]) for r in F1] + list(MADE_UP):
        if len(n) > 16 or sum(1 << (x - min([lid] + n)) for x in n) > 256:
            continue  # rejected: no shape, no error terms
        s = shape(lid, n, k)
        out.append("LEDGER name=%s lid=%d n=%s k=%s -> identity=%d singles=%.9f" % (name, lid, comma(n), comma(k), LE - s["dT"],
                                                                                  LE - math.log2(counts(s)["singles"])))  # fmt: skip
    return out


def main():
    args = sys.argv[1:]
    weights = args[:1] == ["--weights"]
    if weights:
        args = args[1:]
    mine = weights_lines() if weights else lines()
    if len(args) == 2 and args[0] == "--hold":
        with open(args[1]) as f:
            theirs = [x.rstrip("\n") for x in f if not x.startswith("#")]
        bad = sum(a != b for a, b in zip(mine, theirs)) + abs(len(mine) - len(theirs))
        print("forest_values.py: %d lines made again, %d differ from the file" % (len(mine), bad))
        return 1 if bad else 0
    if args:
        print("usage: forest_values.py [--weights] [--hold <file>]", file=sys.stderr)
        return 2
    print("# Generated by verifier/tests/known_answers/forest_values.py. To compare this file with what the script generates: "
          "forest_values.py %s--hold <this file>" % ("--weights " if weights else ""))  # fmt: skip
    print("\n".join(mine))
    return 0


if __name__ == "__main__":
    sys.exit(main())
