#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""A second verifier, beside the C++ check_entry(), of a leaf file ("entry" in the programs' names).

The file ties a leaf table ("the further table", "the copy") to its commitment's anchor ("the first
table"): its hash chain runs LINK, which is to show that their rows agree in every message entry
before the cut. Channel, hashes, draws, Merkle path checks: from tests/new/second/toy_check.py and
docs/format/format_rules.py. LINK's algebra: from part 10 of
docs/format/toys/3-key-and-entry.md, not word for word as it stands; the proximity test: as in parts 3 and 4
(docs/format/toys/1-product.md) and 5 (docs/format/toys/2-links.md). It calls no code
of blocks/. The lines on the record (stored answers to samples) run on the leaf files that
tests/new/second/run_honest.sh makes.

This check is not independent of the C++ programs on these points, where an error in understanding
a rule would be in both and comparing the two would not show it: the order of the file's parts and
the chain's messages (blocks/frame/check_entry.h, blocks/frame/tests/check_entry_test.cpp); a toy
LINK's shape and plan (the first 120 lines of blocks/iop/tests/link_world.h); the first query's
items (some 25 lines at link_bind(), blocks/iop/link.h) and their bytes (some ten lines each at
check_entry_test.cpp's `world()`, check_entry.h's entry_fixed() and link_world.h's link_commit());
and the points in the docstring of tests/new/second/toy_check.py, which has the same author as
this check. Nor is it independent of the C++ programs in parse(), table_item(), first_table_of(),
`statement`, the record of one value at each tree vertex (toy_check.py's `Run.opening()`) or the
notes (intermediate values; no decision reads them): of the C++ verifier's 28, this check's first
writing had seven; the other 21 are from that verifier's list, under its names, in its order;
all 28 were equal at the first comparison; two more, on the records' first rows, are from that list in
the same way and were equal at their first comparison. Whether
this check is independent of the following has not been examined, so nothing is known either way:
the rest of blocks/iop/link.h and of the other files of which only some lines are named here;
blocks/iop/foldeq.h; blocks/frame/link_frame.h; every other file of the C++ and CUDA programs.

usage: entry_check.py <directory of format_rules.py> <directory of the files> <run>
Reads <run>.parameters (as given: no soundness error is computed), <run>.statement (for the
anchor's item, and to compare) and <run>.file.
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
from toy_check import ZERO, ONE, Refused, add, sub, mul, scale, emb, inner, eq_table, log2, point, powers  # noqa: E402

R = tc.R
tc.KIND.update({"LINK point": 1, "LINK": 4, "LINK phi": 1})
X = (ONE, (0, 1, 0), (0, 0, 1))
LABEL = 0x4c4e4b01


def clog2(n):
    return (n - 1).bit_length()


def parse(raw):
    b, out = tc.Bytes(raw), {}
    assert b.text() == "the format's version" and b.word() == 8 and b.take(8) == b"H821" + bytes(4)
    out["head"] = [b.word() for _ in range(8)]
    out["rounds"] = []
    for _ in range(out["head"][7]):
        name = b.text()
        out["rounds"].append(dict(zip(tc.ENTRY, [b.word() for _ in range(11)]), name=name))
    items = []
    while b.at < len(raw):
        name = b.text()
        items.append((name, b.take(b.word())))
    out["items"] = items
    assert [n for n, _ in items] == ["configuration", "codes", "the link"]
    v = tc.Bytes(items[1][1])
    out["base"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    out["mask"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    v = tc.Bytes(items[2][1])
    # This item has each public number once; the cut (`plain` in verify()) and the test's first
    # stage are derived.
    code, vv, s, salt = tuple(v.word() for _ in range(3)), v.word(), v.word(), v.word()
    nu0, nuj, gc, gm, Rr, high, cls, ahead = (v.word() for _ in range(8))
    later = [tuple(v.word() for _ in range(4)) for _ in range(v.word())]
    ride = v.word()
    assert v.at == len(items[2][1])
    mk = lambda nu: dict(code=code, v=vv, s=s, shape=(gc, (1 << 32) - 1, Rr, 3, 0, out["head"][5]), rows_high=high, leaf_class=cls, columns=nu)  # noqa: E731
    out["link"] = {"first": mk(nu0), "further": mk(nuj), "cut": code[1] - 3 * s, "mask_code": gm, "masks_ahead": ahead, "test": [(gc, gm, 1 + clog2(Rr), 0)] + later,
                   "choice": (out["head"][5], ride), "salt": salt}
    return out


def table_item(root, y):
    """A table's item in the statement.

    The root as a string of bytes, then the list of its record's answers.
    """
    return R.u64(32) + root + R.u64(len(y)) + b"".join(R.ext(e) for e in y)


def first_table_of(statement, l, rows, s):
    """The anchor's root and record's answers, from the statement's first item.

    A full verifier takes them from a key file that it has checked.
    """
    b = tc.Bytes(statement)
    assert b.text() == "the first table"
    v = tc.Bytes(b.take(b.word()))
    assert v.word() == 32
    return v.take(32), [tc.ext_of(v.take(24)) for _ in range(v.word())]


# Follows the verifier's side of the zero-knowledge IOPP for constrained codes of
# the paper that this protocol builds on (see THIRD_PARTY.md).
def iopp(ch, st, Cs, Zs, c, d, tau, first_tree, first_reader, trees, notes):
    """The proximity test after BAT (which batches the claims).

    Every mask is sent in its own stage's tree. first_reader: maps an opened column of the first
    stage's tree to (2^k main, 2^h old mask, 2^h fresh mask symbols).
    """
    own = first_tree
    for s in range(len(st) - 1):
        k, h = st[s][2], st[s][3]
        k2, h2 = st[s + 1][2], st[s + 1][3]
        g, c0, d0, d1, tau = tc.fold_block(ch, k, h, c, d, tau, Zs[s][1])
        notes.append(("FOLD: target", tau))
        l, r, lz, rz = Cs[s][1], Cs[s][2], Zs[s][1], Zs[s][2]
        assert Cs[s + 1][1] << k2 == l and (Zs[s + 1][1] << h2) >= 2 * lz + r + 2 * rz + 1
        lastly = s + 2 == len(st)
        trees.append((Cs[s + 1][0], (3 << k2) + (3 << h2) + (6 if lastly else (3 << h2)), ch.recv_root("CS: the root of the next stage's tree")))
        ch.close("SEND seed")
        bits_main, bits_mask = log2(l), h2 + log2(Zs[s + 1][1])
        z, beta = tc.seed(ch, max(bits_main, bits_mask) + 1)
        alpha, = ch.recv("CS: the answer to the sample")
        ch.close("CS")
        draws = ch.positions()
        theta = ch.element()
        col = ch.opening(own, trees[own][0], trees[own][1], trees[own][2], draws)
        assert ch.entry["rows"] == 2 + 3 * len(draws)
        A1, T, T0, T1 = lz, 2 * lz, 2 * lz + r, 2 * lz + r + rz
        nc, nd = list(c0), [ZERO] * (Zs[s + 1][1] << h2)
        for j in range(lz):
            nd[j], nd[A1 + j] = d0[j], d1[j]
        u, w = tc.sample_row(z, beta, bits_main, bits_mask)
        f = theta
        nc = [add(a_, mul(f, b_)) for a_, b_ in zip(nc, u)]
        nd = [add(a_, mul(f, b_)) for a_, b_ in zip(nd, w)]
        ntau = add(tau, mul(f, alpha))
        eg, egh = eq_table(g), eq_table(g[:h])
        for n, xx in enumerate(draws):
            cc = col[xx]
            if s == 0:
                main, oldm, fresh = first_reader(cc)
            else:
                main = [tuple(cc[3 * b:3 * b + 3]) for b in range(1 << k)]
                oldm = [tuple(cc[(3 << k) + 3 * b:(3 << k) + 3 * b + 3]) for b in range(1 << h)]
                fresh = [tuple(cc[(3 << k) + (3 << h) + 3 * b:(3 << k) + (3 << h) + 3 * b + 3]) for b in range(1 << h)]
            ys = (inner(eg, main), inner(egh, oldm), inner(egh, fresh))
            if n == 0:
                notes += [("CS: y_main at the first draw", ys[0]), ("CS: y_0 at the first draw", ys[1]), ("CS: y_1 at the first draw", ys[2])]
            pw = powers(point(xx), max(r + l, rz + lz))
            f = mul(f, theta)
            for j in range(l):
                nc[j] = add(nc[j], scale(pw[r + j], f))
            for j in range(r):
                nd[T + j] = add(nd[T + j], scale(pw[j], f))
            ntau = add(ntau, mul(f, ys[0]))
            for which, (A, TT) in enumerate(((0, T0), (A1, T1))):
                f = mul(f, theta)
                for j in range(lz):
                    nd[A + j] = add(nd[A + j], scale(pw[rz + j], f))
                for j in range(rz):
                    nd[TT + j] = add(nd[TT + j], scale(pw[j], f))
                ntau = add(ntau, mul(f, ys[1 + which]))
        c, d, tau, own = nc, nd, ntau, len(trees) - 1
        notes += [("CS: target", tau), ("CS: c'[0]", c[0]), ("CS: d'[0]", d[0]), ("CS: d's last entry", d[-1])]
    l, r, lz, rz = Cs[-1][1], Cs[-1][2], Zs[-1][1], Zs[-1][2]
    tstar, = ch.recv("BASE: the masking word's target")
    ch.close("BASE lambda")
    lam = ch.element()
    M0, M1 = ch.recv("BASE: the masked main message", l), ch.recv("BASE: the masked mask message", lz)
    R0, R1 = ch.recv("BASE: the masked main tail", r), ch.recv("BASE: the masked mask tail", rz)
    left, right = add(inner(c, M0), inner(d, M1)), add(tstar, mul(lam, tau))
    notes += [("BASE: left side of the linear check", left), ("BASE: right side of the linear check", right)]
    if left != right:
        raise Refused("BASE: the linear check fails")
    ch.close("BASE")
    draws = ch.positions()
    col = ch.opening(own, trees[own][0], trees[own][1], trees[own][2], draws)
    for xx in sorted(set(draws)):
        g3m, gsm, g3z, gsz = (tuple(col[xx][3 * i:3 * i + 3]) for i in range(4))
        if tc.symbol(Cs[-1], xx, M0, R0) != add(gsm, mul(lam, g3m)) or tc.symbol(Zs[-1], xx, M1, R1) != add(gsz, mul(lam, g3z)):
            raise Refused("BASE: a check at a position fails")


# (1) to (7): the steps of part 10, section 4.
# notes: receives intermediate values, which no decision reads.
def verify(par, raw, given_statement, file, notes):
    q = par["link"]
    F0, Fj = q["first"], q["further"]
    assert F0["code"] == Fj["code"] and F0["v"] == Fj["v"] and F0["s"] == Fj["s"] and F0["shape"] == Fj["shape"] and q["masks_ahead"] == 1 and q["choice"] == (1, 1)
    (n_log, l, r), v, s = F0["code"], F0["v"], F0["s"]
    plain, a = l - 3 * s, clog2(l)
    conf, salt = par["items"][0][1], par["head"][2]
    f = tc.Bytes(file)
    if f.take(4) != b"CPEN":
        raise Refused("a file of another kind")
    if f.take(4) != b"H821":
        raise Refused("another version")
    root_j = f.take(32)
    y_j = [tc.ext_of(f.take(24)) for _ in range(s << v)]
    salt = q["salt"]   # stored tables' trees all have number 0: only roots tell tables apart
    root_0, y_0 = first_table_of(given_statement, l, 1 << v, s)
    seeds = [R.record_seeds(R.table_name(root, conf, n_log, l, r, v, s, salt), a, s, 2) for root in (root_0, root_j)]
    statement = R.item("the first table", table_item(root_0, y_0)) + R.item("the further table", table_item(root_j, y_j))
    notes.append(("the statement's bytes, formed by the reader, are those that the C++ Binder holds", statement == given_statement))
    ch = tc.Run(par, raw, None, file[f.at:], label=LABEL, statement=statement)
    # (1) the fold point
    ch.close("LINK point")
    gamma = [ch.element() for _ in range(v)]
    w = eq_table(gamma)
    # (2) to (4)
    Cu = par["base"][F0["shape"][0]]
    Rr, lu = F0["shape"][2], Cu[1]
    ar, ap = clog2(Rr), log2(lu)
    trees = [(Cu[0], 3 * 2 * Rr + 3, ch.recv_root("LINK: the root of the tree of both folded rows"))]
    ch.close("SEND seed")
    z, beta = tc.seed(ch, ar + ap)
    alphas = ch.recv("LINK: the two answers", 2)
    ch.close("SEND answer")
    # (5), (6)
    ch.close("LINK")
    draws = ch.positions()
    lists = (draws[:F0["columns"]], draws[F0["columns"]:])
    assert len(lists[1]) == Fj["columns"]
    cols, ch.openings_at = [], [f.at + ch.f.at]   # file offsets bounding the two tables' openings
    for which, (t, rt, xs) in enumerate(((0, root_0, lists[0]), (0, root_j, lists[1]))):
        try:
            cols.append(ch.opening(t, n_log, 1 << v, rt, xs, stored=True))
        except Refused as why:
            raise Refused("%s [in %s]" % (why, ("the first table's opening", "the copy's opening")[which]))
        ch.openings_at.append(f.at + ch.f.at)
    # (7) the point of comparison
    ch.close("LINK phi")
    phi = [ch.element() for _ in range(a)]
    # The linear claims (`rows`): part 10, section 5.
    # at(i, e): where entry e of folded row i is in the group's vector.
    st = q["test"]
    k = st[0][2]
    assert k == 1 + ar and st[0][3] == 0
    t0 = -(-l // (1 << clog2(r))) << clog2(r)

    def at(i, e):
        row, place = (e >> ap, e & (lu - 1)) if F0["rows_high"] else (e & ((1 << ar) - 1), e >> ar)
        return ((i << ar) + row) + (place << k)

    rows = []
    for i, ys in ((0, y_0), (1, y_j)):
        for qn in range(s):
            zz, bb = seeds[i][qn]
            e = eq_table(zz)
            form = {at(i, j): e[j] for j in range(plain)}
            for t in range(3):
                form[at(i, plain + 3 * qn + t)] = mul(bb, X[t])
            rows.append((form, inner(w, [ys[b * s + qn] for b in range(1 << v)])))
    e = eq_table(phi)
    form = {}
    for j in range(q["cut"]):
        form[at(0, j)], form[at(1, j)] = e[j], sub(ZERO, e[j])
    rows.append((form, ZERO))
    e = eq_table(z)
    for i in range(2):   # sample claims; j = row + 2^ar place; the pad: row R - 1, place l_u - 1
        form = {((i << ar) + (j & ((1 << ar) - 1))) + ((j >> ar) << k): e[j] for j in range(len(e))}
        form[((i << ar) + Rr - 1) + ((lu - 1) << k)] = beta
        rows.append((form, alphas[i]))
    firsts = []   # noted only: each table's linear claim at its first draw
    for i in range(2):
        for n_, x in enumerate(lists[i]):
            pw = powers(point(x), r + l)
            form = {at(i, j): emb(pw[r + j]) for j in range(l)}
            for j in range(r):
                form[at(i, t0 + j)] = emb(pw[j])
            rows.append((form, inner(w, [emb(cb) for cb in cols[i][x]])))
            if n_ == 0:
                firsts += [("FOLDEQ: (Col_1)'s target", rows[-1][1]), ("FOLDEQ: (Col_1) on G_u<0>", form[at(i, 0)]), ("FOLDEQ: (Col_1) on G_u<B>", form[at(i, t0)])]
    notes += [("LINK: the target of the first table's record's first row", rows[0][1]), ("LINK: the target of the further table's record's first row", rows[s][1])]
    notes += firsts
    notes += [("LINK: the fold of the first table's column at its first draw", firsts[0][1]), ("LINK: the fold of the further table's column at its first draw", firsts[3][1]),
              ("LINK: the equality row's coefficient at entry 0 of the first table's folded row", eq_table(phi)[0])]
    ch.close("BAT")
    assert ch.entry["rows"] == len(rows), (ch.entry, len(rows))
    x = ch.element()
    Cs, Zs = [par["base"][t[0]] for t in st], [par["mask"][t[1]] for t in st]
    c, tau, fct = [ZERO] * (lu << k), ZERO, ONE
    for form, target in rows:
        for i, cf in form.items():
            c[i] = add(c[i], mul(fct, cf))
        tau, fct = add(tau, mul(fct, target)), mul(fct, x)
    notes.append(("BAT: target", tau))

    def reader(cc):
        main = [ZERO] * (1 << k)
        for i in range(2):
            for b in range(Rr):
                main[(i << ar) + b] = tuple(cc[3 * (i * Rr + b):3 * (i * Rr + b) + 3])
        return main, [ZERO], [tuple(cc[6 * Rr:6 * Rr + 3])]

    try:
        iopp(ch, st, Cs, Zs, c, [ZERO] * Zs[0][1], tau, 0, reader, trees, notes)
    except Refused as why:
        raise Refused("%s [in the link's chain, its proximity test]" % why)
    ch.finish()
    ch.trees = trees
    return ch


def main():
    folder, run = sys.argv[2], sys.argv[3]
    read = lambda tail: open("%s/%s.%s" % (folder, run, tail), "rb").read()  # noqa: E731
    raw, notes = read("parameters"), []
    par = parse(raw)
    try:
        ch = verify(par, raw, read("statement"), read("file"), notes)
    except Refused as why:
        print("%s: REFUSED by the second verifier: %s" % (run, why))
        return 1
    print("%s: ACCEPTED by the second verifier (%d bytes, %d rounds)" % (run, len(read("file")), len(ch.rounds)))
    for a, b in notes:
        print("  %s: %s" % (a, b if isinstance(b, bool) else tc.dec(b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
