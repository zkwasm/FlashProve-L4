#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""The verifier's side of LIN, the test of a request's linear claim (`Row` in identifiers) on a
committed table.

A second implementation, imported by tests/new/second/key_check.py and
tests/new/second/proof_check.py.
Sections and letters: Part 9 of docs/format/toys/3-key-and-entry.md.
R-numbers: rules of docs/format/FORMAT.md.

It calls no code of blocks/. It is not independent of the C++ programs there on these points, where
both follow the same text or rule, so that an error in understanding it would be in both and
comparing them would not show it: what blocks/iop/tests/lin_world.h contains (a toy test fixture, a
session of LIN, and the order in which one request's verifier calls LIN's functions), and what
tests/new/second/toy_check.py lists for itself. Its formulas are from Part 9, not word for
word as it stands; this is not claimed of table_of() and the start of class Lin, which are not of the first writing. Part 9
describes the code of blocks/iop/lin.h, blocks/iop/foldeq.h, blocks/iop/split.h and
blocks/iop/index_cpu.h, so this check follows the same rules as those four files: a rule understood
wrongly is wrong in both, and the check does not show it. Whether this check is independent of the
rest of the files that tests/new/second/toy_check.py lists only in part, or of any other file of the
programs, in blocks/ or outside it, has not been examined: about all of these files nothing is
known, either way.

The lines on the table's record (stored answers to samples) run on the toy's key files and proofs
that tests/new/second/run_honest.sh makes.

As a script it reads one request's three hash chains and path, dumped from LIN's toy tests by a
program outside this repository, so nothing here runs it that way.
usage: lin_check.py <directory of format_rules.py> <directory> <stem>
reads <stem>.first, .path, .shared, .second, the three .parameters, .first.statement, .claim
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
import zkpy  # noqa: E402
from toy_check import ZERO, ONE, Refused, add, sub, mul, emb, inner, eq_table, log2, point, powers  # noqa: E402
from zkpy import V, Row, clog2, NONE, X  # noqa: E402

R = tc.R
# Round name -> kind of the verifier's message (R5.2).
tc.KIND.update({"LIN point": 1, "LIN zeta": 1, "LOC": 4, "SPLIT": 7})


# Parses one chain's parameters. A code is (log2 n, l, r).
# "S", if present, is LinShape of section 1: the table has 2^v rows, each with `a` stored answers
# to samples (the record).
# Below, P names what parse() returns: P1, P2, P3 for the first, shared and second chain.
def parse(raw, with_shape=True):
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
    d = dict(items)
    if "codes" in d:
        v = tc.Bytes(d["codes"])
        out["base"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
        out["mask"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
        v = tc.Bytes(d["the test on the table"])
        six = lambda: tuple(v.word() for _ in range(6))  # noqa: E731
        S = dict(code=tuple(v.word() for _ in range(3)), v=v.word(), a=v.word(), salt=v.word(), Gu=six(), u_high=v.word(), u_class=v.word(), nu=v.word(), k_lo=v.word(), m=v.word(), Om=six(), W=six(),
                 om_high=v.word(), w_high=v.word(), split_shared=six(), split_own=six(), classes=tuple(v.word() for _ in range(4)), shared_mask=v.word(), own_mask=v.word(), ahead=v.word())
        S["tests"] = [[tuple(v.word() for _ in range(4)) for _ in range(v.word())] for _ in range(4)]
        S["choice"] = (out["head"][5], v.word())
        assert v.at == len(d["the test on the table"])
        out["S"] = S
    return out


def lay(o, L, high):
    """Entry j of a vector -> its index in oracle message O's main table: O<j> in section 0."""
    b = clog2(o[2])
    return (lambda j: (j // L) + ((j % L) << b)) if high else (lambda j: j)


def items_of(b):
    out, v = [], tc.Bytes(b)
    while v.at < len(b):
        name = v.text()
        out.append((name, v.take(v.word())))
    return out


def table_of(value):
    """The item "table": the root as a string of bytes, then the list of the record's answers."""
    v = tc.Bytes(value)
    assert v.word() == 32
    return v.take(32), [tc.ext_of(v.take(24)) for _ in range(v.word())]


class Lin:
    """What is public about the test on a table, from a chain's parameters and the item "table".

    The table's name (R4.10) and the record's seeds (R4.8) are computed.
    The table holds a matrix M: M[y][j + 2^k_lo r] is entry j + 2^k_lo y of row r's message.
    Honest oracle messages: W = g^T M; G_u: the table's rows folded at gamma; Omega = M eq(star).
    """

    def __init__(self, P, table_item):
        self.S, self.base, self.mask, self.table_item = P["S"], P["base"], P["mask"], table_item
        S = self.S
        root, self.y = table_of(table_item)
        (self.nT, self.lT, self.rT), self.v, self.a = S["code"], S["v"], S["a"]
        self.conf = P["items"][0][1]
        self.name = R.table_name(root, self.conf, self.nT, self.lT, self.rT, self.v, self.a, S["salt"])
        self.seeds = R.record_seeds(self.name, clog2(self.lT), self.a, 2)
        if len(self.y) != self.a << self.v:
            raise Refused("the record has another number of answers")
        self.plain, self.m, self.k_lo = self.lT - 3 * self.a, S["m"], S["k_lo"]
        self.k_col, self.k_row = self.k_lo + self.v, clog2(self.m)
        self.shared_split, self.own_split = S["Gu"][0] != S["Om"][0], S["W"][0] != S["Om"][0]
        self.ride, self.ahead = S["choice"][1], S["ahead"]
        self.Wat = lay(S["W"], self.base[S["W"][0]][1], S["w_high"])
        self.Gat = lay(S["Gu"], self.base[S["Gu"][0]][1], S["u_high"])
        self.Oat = lay(S["Om"], self.base[S["Om"][0]][1], S["om_high"])


# The notice (R12.3) of the mask message for the first fold of a proximity test
# (`iopp` in identifiers), if sent ahead.
def fold_mask(test, cls):
    return (NONE, test[0][1], 0, 1, 1 << test[0][3], 0, cls)


def lin_first(A, L, g, psi, masks, tau):
    """Section 4: LIN's part of a request's first chain.

    The claim: g^T M eq(psi) + mask terms = tau; a mask term is ((oracle message, entry),
    coefficient). Not checked: that the chain absorbs what fixes the claim before its seal.
    """
    S = L.S
    cl_om, cl_ss, cl_w, cl_os = S["classes"]
    ids = A.send([S["W"] + (cl_w,)] + ([S["split_own"] + (cl_os,)] if L.own_split else []) + ([fold_mask(S["tests"][2], cl_w)] if L.ahead else []))
    F = dict(W=ids[0], K=ids[1] if L.own_split else None, mask=ids[-1] if L.ahead else None, g=g)
    F["OWN"] = A.new_group(S["W"][0], S["own_mask"])
    for i in sorted({o for (o, _), _ in masks}) + [F["W"]] + ([F["K"]] if L.own_split else []):
        A.join(F["OWN"], i)
    e = eq_table(psi)
    forms = {(F["W"], "M"): {L.Wat(j): e[j] for j in range(1 << L.k_col)}}
    for (o, index), cf in masks:
        forms.setdefault((o, "Z"), {})
        forms[(o, "Z")][index] = add(forms[(o, "Z")].get(index, ZERO), cf)
    A.groups[F["OWN"]]["plain"].append(Row(forms, tau))
    A.note("LIN: (Eval)'s target", tau), A.note("LIN: (Eval) on W<1>", e[1])
    return F


# Section 5: the root of the binding tree (over a batch's first chains' seals), from a path p.
# Nothing here compares it: lin_shared() puts it into the shared chain's first query.
def binding_root(conf, d, salt_bytes, p, seal):
    place = p.word()
    if place >> d:
        raise Refused("the path: no such place")
    hsh = R.binding_leaf(d, place, conf, seal, p.take(salt_bytes))
    for t in range(d):
        s = p.take(32)
        hsh = R.H(178, 0, 0, conf, s + hsh if place >> t & 1 else hsh + s)
    return hsh


def lin_shared(L, P2, raw2, f, root, label=0x4c494e01):
    """Section 6: the entire shared chain. f: its bytes, or a tc.Bytes at them.

    Not compared here: P2's codes and shape with L's.
    """
    S, base, mask = L.S, L.base, L.mask
    cl_om, cl_ss, cl_w, cl_os = S["classes"]
    st2 = R.item("table", L.table_item) + R.item("the binding tree's root", root)
    ch = tc.Run(P2, raw2, None, f, label=label, statement=st2, random=root)
    B = V(ch, base, mask)
    ch.close("LIN point")
    star = [ch.element() for _ in range(L.k_col)]
    psi_lo, gamma = star[:L.k_lo], star[L.k_lo:]
    U = B.new_group(S["Gu"][0], S["shared_mask"])
    OM = B.new_group(S["Om"][0], S["shared_mask"]) if L.shared_split else U
    er = eq_table(gamma)
    B.note("FOLDEQ: e_0", er[0])
    ids = B.send([S["Gu"] + (S["u_class"],)] + ([fold_mask(S["tests"][0], S["u_class"])] if L.ahead else []))
    Gu, u_mask = ids[0], ids[1] if L.ahead else None
    B.join(U, Gu)
    Gat, a, lT, rT, v = L.Gat, L.a, L.lT, L.rT, L.v
    for i in range(a):
        zz, bb = L.seeds[i]
        ee = eq_table(zz)
        form = {Gat(pp): ee[pp] for pp in range(L.plain)}
        for t in range(3):
            form[Gat(lT - 3 * a + 3 * i + t)] = mul(bb, X[t])
        B.groups[U]["plain"].append(Row({(Gu, "M"): form}, inner(er, [L.y[r * a + i] for r in range(1 << v)])))
        if i == 0:
            B.note("FOLDEQ: (Rec_0)'s target", B.groups[U]["plain"][-1].target)
    ch.close("LOC")
    draws = ch.positions()
    assert len(draws) == S["nu"]
    T = zkpy.Tree()
    T.stored, T.number, T.depth, T.width, T.start, T.root = True, 0, L.nT, 1 << v, L.conf, L.name[:32]
    B.table_opening_at = ch.f.at
    cols = zkpy.opening(ch, T, draws)
    Bt = -(-lT // (1 << clog2(rT))) << clog2(rT)
    for x in draws:
        pw = powers(point(x), rT + lT)
        form = {Gat(pp): emb(pw[rT + pp]) for pp in range(lT)}
        for q in range(rT):
            form[Gat(Bt + q)] = emb(pw[q])
        B.groups[U]["gens"].append(Row({(Gu, "M"): form}, inner(er, [emb(s) for s in cols[x]])))
        if len(B.groups[U]["gens"]) == 1:
            B.note("FOLDEQ: (Col_1)'s target", B.groups[U]["gens"][0].target), B.note("FOLDEQ: (Col_1) on G_u<0>", form[Gat(0)]), B.note("FOLDEQ: (Col_1) on G_u<B>", form[Gat(Bt)])
    ids = B.send([S["Om"] + (cl_om,)] + ([S["split_shared"] + (cl_ss,)] if L.shared_split else []))
    Om = ids[0]
    assert B.oracles[Om].tree.parts == [Om], "something shares Omega's tree"
    B.join(OM, Om)
    if L.shared_split:
        B.join(OM, ids[1]), B.join(U, ids[1])
    ch.close("LIN zeta")
    zeta = [ch.element() for _ in range(L.k_row)]
    ez, el = eq_table(zeta), eq_table(psi_lo)
    row = Row({(Om, "M"): {L.Oat(yy): ez[yy] for yy in range(L.m)},
               (Gu, "M"): {Gat(j + (yy << L.k_lo)): sub(ZERO, mul(el[j], ez[yy])) for yy in range(L.m) for j in range(1 << L.k_lo)}}, ZERO)
    B.note("LIN: (Fold) on Omega<0>", row.forms[(Om, "M")][L.Oat(0)]), B.note("LIN: (Fold) on G_u<1>", row.forms[(Gu, "M")][Gat(1)])
    if L.shared_split:
        B.split(row, OM, U, ids[1])
    else:
        B.groups[U]["plain"].append(row)
    B.iopp(U, S["tests"][0], L.ride, u_mask, "the shared chain, the test on the folded row's group")
    if L.shared_split:
        B.iopp(OM, S["tests"][1], L.ride, None, "the shared chain, the test on Omega's group")
    seal2 = ch.seal()
    return B, Om, star, seal2, ch


def lin_second(A, B, L, F, Om, star, P3, raw3, f, seal1, seal2, label=0x4c494e02):
    """Section 7: the request's second chain, which continues the first (A) and its count of trees.

    Omega's tree, taken over from the shared chain (B), keeps its own number and start (R7.13).
    """
    S = L.S
    st3 = R.item("the seal of an earlier chain", seal1) + R.item("the seal of an earlier chain", seal2)
    ch = tc.Run(P3, raw3, None, f, label=label, statement=st3)
    A.ch = ch
    o = zkpy.Oracle(len(A.oracles), S["Om"] + (S["classes"][0],))
    o.tree = B.oracles[Om].tree
    o.tree.main_at[o.id], o.tree.mask_at[o.id] = o.tree.main_at[Om], o.tree.mask_at[Om]
    o.tree.cols = {}
    A.oracles.append(o)
    A.trees.append(o.tree)
    A.number += 1   # the tree that is taken over uses up a place of this chain's count (R7.13)
    A.samples[o.id] = Row({(o.id, part): form for (_, part), form in B.samples[Om].forms.items()}, B.samples[Om].target)
    OWN = F["OWN"]
    if L.own_split:
        AR = A.new_group(S["Om"][0], S["own_mask"])
        A.join(AR, o.id), A.join(AR, F["K"])
    else:
        AR = OWN
        A.join(OWN, o.id)
    es = eq_table(star)
    row = Row({(o.id, "M"): {L.Oat(yy): F["g"][yy] for yy in range(L.m)}, (F["W"], "M"): {L.Wat(j): sub(ZERO, es[j]) for j in range(1 << L.k_col)}}, ZERO)
    A.note("LIN: (Dot) on Omega<0>", row.forms[(o.id, "M")][L.Oat(0)]), A.note("LIN: (Dot) on W<1>", row.forms[(F["W"], "M")][L.Wat(1)])
    if L.own_split:
        A.split(row, AR, OWN, F["K"])
    else:
        A.groups[OWN]["plain"].append(row)
    A.iopp(OWN, S["tests"][2], L.ride, F["mask"], "the second chain, the test on the own rows' group")
    if L.own_split:
        A.iopp(AR, S["tests"][3], L.ride, None, "the second chain, the test on Omega's group")
    return ch


def verify(read):
    zkpy.new_session()
    P1, P2, P3 = parse(read("first.parameters")), parse(read("shared.parameters")), parse(read("second.parameters"))
    st1 = read("first.statement")
    L = Lin(P1, dict(items_of(st1))["table"])
    c = tc.Bytes(read("claim"))
    target = tc.ext_of(c.take(24))
    psi = [tc.ext_of(c.take(24)) for _ in range(c.word())]
    g = [tc.ext_of(c.take(24)) for _ in range(c.word())]
    ch = tc.Run(P1, read("first.parameters"), None, read("first"), label=1, statement=st1)
    A = V(ch, L.base, L.mask)
    inp, = A.send([struct.unpack("<7Q", dict(P1["items"])["a request's mask message"])])
    tau, = ch.recv("the claim's target")
    if tau != target:
        raise Refused("the proof is about another target")
    F = lin_first(A, L, g, psi, [((inp, A.take(inp)), ONE)], tau)
    seal1 = ch.seal()
    ch.finish()
    d, salt_bytes = struct.unpack("<2Q", dict(P1["items"])["the binding tree"][:16])
    p = tc.Bytes(read("path"))
    root = binding_root(L.conf, d, salt_bytes, p, seal1)
    if p.at != len(p.b):
        raise Refused("the path: bytes are left over")
    B, Om, star, seal2, ch2 = lin_shared(L, P2, read("shared.parameters"), read("shared"), root)
    ch2.finish()
    lin_second(A, B, L, F, Om, star, P3, read("second.parameters"), read("second"), seal1, seal2).finish()
    return A, B


def main():
    folder, stem = sys.argv[2], sys.argv[3]
    read = lambda tail: open("%s/%s.%s" % (folder, stem, tail), "rb").read()  # noqa: E731
    try:
        verify(read)
        print("%s: ACCEPTED by the second verifier" % stem)
        return 0
    except Refused as why:
        print("%s: REFUSED by the second verifier: %s" % (stem, why))
        return 1


if __name__ == "__main__":
    sys.exit(main())
