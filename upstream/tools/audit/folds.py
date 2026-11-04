#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Recomputes what a fold and a code switch derive from private randomness and challenges only.

level1.py compares the results with published values. A fold (FOLD) combines the rows of a group
of oracle messages. A code switch (blocks/iop/cs.h: CS^sigma and CS) re-encodes folded rows as a
new oracle message, part of which cs.h calls "the rest".
Functions follow the comments of the blocks/iop/ headers named here; no library function is copied.
`Ex`: a field.E; elements of E are triples.
"""
import cut


def triples(flat): return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]


def weights(Ex, g):
    """Row b's weight in a fold at g (blocks/iop/fold_algebra.h):

    w[b] = prod_j (g_j if bit j of b else 1 - g_j).
    """
    w = [Ex.one]
    for x in g:
        no = Ex.sub(Ex.one, x)
        w = [Ex.mul(a, no) for a in w] + [Ex.mul(a, x) for a in w]
    return w


def combine(Ex, w, rows, length):
    """sum_b w[b] rows[b]; rows: {b: list of elements of E, or of numbers (a row over F)}."""
    p = Ex.p
    acc = [[0, 0, 0] for _ in range(length)]
    for b, row in rows.items():
        wb = w[b]
        if row and not isinstance(row[0], tuple):
            for i, c in enumerate(row):
                a = acc[i]; a[0] += wb[0] * c; a[1] += wb[1] * c; a[2] += wb[2] * c
        else:
            for i, e in enumerate(row):
                m = Ex.mul(wb, e); a = acc[i]; a[0] += m[0]; a[1] += m[1]; a[2] += m[2]
    return [(a[0] % p, a[1] % p, a[2] % p) for a in acc]


def blocks(sizes):
    """sizes: [(key, real rows)] in sending order. Returns ({key: first row}, rows used).

    An oracle message with K real rows takes an aligned block of 2^ceil_log2(K) rows. Blocks are
    placed from row 0 by decreasing size, ties in sending order; the remaining rows are zero
    (blocks/iop/relation.h).
    """
    order = sorted(range(len(sizes)), key=lambda i: (-cut.up2(sizes[i][1]), i))
    at, out = 0, {}
    for i in order: out[sizes[i][0]] = at; at += cut.up2(sizes[i][1])
    return out, at


# p.deg is 1 for rows over F.
def main_tail_rows(p, r):
    a = p.tail.elements()
    if p.deg == 1: return [a[r * b:r * (b + 1)] for b in range(p.main_rows)]
    return [triples(a[3 * r * b:3 * r * (b + 1)]) for b in range(p.main_rows)]


def mask_tail_rows(p, r):
    a = p.mask_tail.elements()
    return [triples(a[3 * r * b:3 * r * (b + 1)]) for b in range(p.mask_rows)]


# fold() and lay() follow the zero-knowledge sumcheck IOR and the zero-knowledge code-switching IOR of
# the paper that this protocol builds on (see THIRD_PARTY.md).
def fold(Ex, members, fresh, g, k, h, code, mask_code):
    """Returns "the rest" as elements of E.

    Random tails of the group's main rows are folded at all k challenges in g; messages and random
    tails of its mask rows (old) and of `fresh` (the fold's mask) at the first h
    (blocks/iop/single_rows.h). Kept main rows (read again later) are not folded and get no block.
    Order: messages of the old and fresh mask rows; random tail of the folded main row, then of
    each kept row; random tails of the two folded mask rows; old pads of the kept oracle messages
    (a pad masks a sample's answer).
    members: the group, [(part, kept)] in sending order; a part is a cut.Part.
    level1.py sets a part's `tail`, `mask_tail`, `pads` and `msg_rows` (mask rows' messages, lists
    of elements of E).
    code, mask_code: (log2 length, message entries, tail entries).
    """
    ell_zk, r_zk, r = mask_code[1], mask_code[2], code[2]
    mains, n = blocks([(id(p), p.main_rows) for p, kept in members if p.main_rows and not kept]); assert n <= 1 << k, 'the main blocks do not fit 2^k rows'
    masks, n = blocks([(id(p), p.mask_rows) for p, kept in members if p.mask_rows]); assert n <= 1 << h, 'the mask blocks do not fit 2^h rows'
    wk, wh = weights(Ex, g[:k]), weights(Ex, g[:h])
    tails, msg, mtail = {}, {}, {}
    for p, kept in members:
        if p.main_rows and not kept:
            for b, row in enumerate(main_tail_rows(p, r)): tails[mains[id(p)] + b] = row
        if p.mask_rows:
            for b, row in enumerate(p.msg_rows): msg[masks[id(p)] + b] = row
            for b, row in enumerate(mask_tail_rows(p, r_zk)): mtail[masks[id(p)] + b] = row
    rest = combine(Ex, wh, msg, ell_zk) + combine(Ex, wh, dict(enumerate(fresh.msg_rows)), ell_zk)
    main_tail = combine(Ex, wk, tails, r)
    rest += main_tail
    for p, kept in members:
        if kept:
            for row in main_tail_rows(p, r): rest += [(c, 0, 0) for c in row] if p.deg == 1 else row
    rest += combine(Ex, wh, mtail, r_zk) + combine(Ex, wh, dict(enumerate(mask_tail_rows(fresh, r_zk))), r_zk)
    for p, kept in members:
        if kept: rest += [(d.elements()[0], 0, 0) for d in p.pads] if p.deg == 1 else [d.e() for d in p.pads]
    return rest


def lay(Ex, rest, rows, ell, pads):
    """Returns a block of `rows` rows (a power of two; not checked) of length ell.

    `rest` fills it row index first, skipping the pads, which end the last row
    (blocks/iop/send.h). Free entries stay zero, as in CS^sigma; CS fills them with uniform values,
    which the caller appends to `rest`.
    """
    n = len(pads)
    table = [[Ex.zero] * ell for _ in range(rows)]
    skip = {(rows - 1) + (ell - n + i) * rows for i in range(n)}
    j = 0
    for e in rest:
        while j in skip: j += 1
        table[j % rows][j // rows] = e; j += 1
    assert j <= rows * ell
    for i, e in enumerate(pads): table[rows - 1][ell - n + i] = e
    return table
