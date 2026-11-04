#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by tests/new/second/run_honest.sh, which the runner starts; by hand: below a directory of your own.
"""A second verifier, beside the C++ one, for the toy proof "links": three segments (`link` in
names here), each an entrywise product c = x * b, where a later x is the c before (a kept message),
in the frame of blocks/iop/req.h (the order of segments, closings and tests around a request's
sub-proofs).
It calls no code of blocks/.
The channel (`Run`) and the field are those of tests/new/second/toy_check.py, which imports
docs/format/format_rules.py; the protocol blocks' formulas are as in toy_check.py.
The frame's formulas and letters are defined in part 5 of
docs/format/toys/2-links.md.
It takes <run>.parameters as given and computes no soundness error.
With --write it writes four lists of what it hashed and read (<run>.*.tsv) beside its input files,
replacing files of these names: toy_check.py's report() does so, under new_file()'s conditions.
Exit code 0 if all agrees, otherwise 1.
It is not independent of the C++ programs on the following points. On each, both follow the same
understanding of a text or rule: an error in that understanding would be in both, and comparing them
would not show it. The points: all that blocks/toy/segments.h fixes (the toy's messages, sub-proof,
layout, calls into the frame); every point and file that toy_check.py's docstring lists for itself;
some 25 lines of blocks/iop/req.h around a malicious prover's switch (the switch and the code under
it are not among these sources; the lines around them are: before class ReqProver and in
ReqProver::close()). Otherwise the frame's formulas rest on written descriptions, from which part 5
differs in wording. One author wrote this program and toy_check.py.
Not of the program's first writing: the parser, two names of
values, main(), and one formula: the kept message's old sample claim, which stands by itself among
the next segment's claims. This formula was taken from a written description.
Its independence from any file of the C++ programs that is named neither above nor in toy_check.py's
docstring, and from the rest of any file named only in part (such as blocks/iop/req.h), has not
been examined: about all of these nothing is known, either way.

usage: links_check.py <directory of format_rules.py> <directory of the files> <links|tiny-links>
                      [--write]
       links_check.py <directory of format_rules.py> <directory of the files> <run>
                      --cheats <directory of malicious provers' proofs> <list of expected checks>
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
from toy_check import ZERO, ONE, Refused, add, sub, mul, scale, emb, inner, eq_table, fold, log2, point, powers, prod  # noqa: E402

R = tc.R
tc.KIND.update({"LOC": 4, "BAT with FOLD epsilon": 2})
# the basis 1, X, X^2 of E over F
X = (ONE, (0, 1, 0), (0, 0, 1))


# Returns each code as (log2 of its length, message entries l, random tail entries r).
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
    assert [n for n, _ in items] == ["configuration", "codes", "the chain of links"]
    v = tc.Bytes(items[1][1])
    out["base"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    out["mask"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    out["req"] = parse_links(items[2][1])
    return out


def parse_links(value):
    """Reads the item "the chain of links" (part 5, section 11).

    A message is six words, a stage four (base code, mask code, k, h: 2^k main rows, 2^h mask
    rows), the choices one. The item omits the last proximity test's first stage, (code, mask
    code, 1, 0): added here."""
    v = tc.Bytes(value)
    q = {"code": v.word(), "mask_code": v.word(), "link": []}
    for _ in range(v.word()):
        plain, messages = v.word(), []
        for _ in range(v.word()):
            main_rows, deg, mask_rows, leaf_class, moment, life = (v.word() for _ in range(6))
            messages.append(((q["code"] if main_rows else tc_none(), q["mask_code"] if mask_rows else tc_none(), main_rows, deg, mask_rows, None, leaf_class), moment, life))
        q["link"].append((plain, messages))
    assert v.word() == 0  # nothing rides at the start
    q["answers"] = v.word()
    q["last"] = [(q["code"], q["mask_code"], 1, 0)] + [tuple(v.word() for _ in range(4)) for _ in range(v.word())]
    q["choice"] = (q["answers"], v.word())
    assert v.at == len(value)
    return q


# The code number for rows that a message does not have:
# docs/format/09-key-and-leaf-files.md, R12.3.
def tc_none():
    return (1 << 32) - 1


def axpy(acc, f, form):
    """acc += f * form; a form is a list (dense) or a dict (index -> coefficient)."""
    if isinstance(form, dict):
        for i, c in form.items():
            acc[i] = add(acc[i], mul(f, c))
    else:
        for i, c in enumerate(form):
            if c != ZERO:
                acc[i] = add(acc[i], mul(f, c))


def psc(ch, V):
    """ZERO (challenges zeta), then PSC (a sumcheck with a hiding polynomial), on V variables.

    Defined in part 2 of docs/format/toys/1-product.md."""
    ch.close("ZERO")
    zeta = [ch.element() for _ in range(V)]
    mu, = ch.recv("PSC: the hiding polynomial's sum")
    ch.close("PSC epsilon")
    eps = ch.element()
    claim, gamma = mu, []
    for j in range(V):
        q1, q2 = ch.recv("PSC: round %d" % (j + 1), 2)
        ch.close("PSC round")
        x = ch.element()
        q0 = sub(claim, mul(zeta[j], add(q1, q2)))
        claim = add(q0, mul(x, add(q1, mul(x, q2))))
        gamma.append(x)
    p1, p2 = ch.recv("PSC: the round in Y", 2)
    ch.close("PSC Y")
    by = ch.element(True)
    claim = add(claim, mul(by, add(p1, mul(by, p2))))
    v1, v2 = ch.recv("PSC: the two end values", 2)
    ch.close("PSC values")
    return eps, gamma, by, claim, v1, v2


def rounds_of_fold(ch, k, claim):
    g = []
    for j in range(k):
        q0, q2 = ch.recv("FOLD: round %d" % (j + 1), 2)
        ch.close("FOLD round")
        x = ch.element()
        q1 = sub(sub(claim, scale(2, q0)), q2)
        claim = add(q0, mul(x, add(q1, mul(x, q2))))
        g.append(x)
    return g, claim


# The linear form of an out-of-domain sample on a message without mask rows. Its pad, meant to mask
# the answer, is the last entry (rows over E) or the last three (over F).
def sample_one_table(z, beta, bits, over_E):
    u = eq_table(z[:bits])
    if over_E:
        u[-1] = beta
    else:
        for t in range(3):
            u[len(u) - 3 + t] = mul(beta, X[t])
    return u


# Tests no equation before BASE, the last step: until then only the channel (`tc.Run`) rejects
# (raises `Refused`).
# `notes`: values that tc.report() compares with <run>.targets.tsv.
# A closing is BAT (batches the group's linear claims, `rows` in a plan entry, into one), then FOLD
# (folds the group's rows). A segment's group is its oracle messages, by key: "G" carries the folded
# rows of the closing before; "B" x_1 and b_1, then only b_m; "Bz" its mask row; "P" the product
# c_m, a kept row if m < 3: the next segment sends it again as "Pp". U (in the column layouts
# below): FOLD's fresh mask.
# These shapes are the toy's: of q["link"] only the length is read.
def verify(par, raw, conf, proof, notes):
    ch = tc.Run(par, raw, conf, proof, label=2)
    q = par["req"]
    C, Z = par["base"][q["code"]], par["mask"][q["mask_code"]]
    assert C[0] == Z[0] and len(q["link"]) == 3 and q["answers"] == 1
    n_log, l, r = C
    _, lz, rz = Z
    ll, V = log2(l), log2(l) - 1
    # `LOC`: a round whose challenges are positions only. The 9 counts the kept message's old sample
    # claim separately.
    batched = [e["rows"] for e in par["rounds"] if e["name"] == "BAT"]
    t_loc = [e["positions"] for e in par["rounds"] if e["name"] == "LOC"][0]
    assert batched[1] == batched[2] == 9 + 4 * t_loc and batched[0] == 5 and batched[3] == 2 + 3 * t_loc
    trees = []      # (depth, width, root), by number
    old = None      # what the closing of the segment before has left
    for m in range(1, 5):  # segments 1, 2, 3; 4: the last group
        last = m == 4
        root = ch.recv_root("the root of the tree of prover message %d" % m)
        ch.close("SEND seed")
        rows_B = 2 if m == 1 else 1
        bits_B = (1 if m == 1 else 0) + ll
        a = max(bits_B, log2(lz)) + 1 if m == 1 else 1 + ll
        z, beta = tc.seed(ch, a)
        sampled = (["G"] if m > 1 else []) + (["Pp"] if 1 < m < 4 else []) + ([] if last else ["B", "P"])
        answers = dict(zip(sampled, ch.recv("answers to the sample", len(sampled))))
        size = {"G": 2 * l, "Pp": l, "B": rows_B * l, "P": l, "Bz": lz}
        plain, gens = [], []
        if m == 1:
            ch.close("SEND answer")
            trees.append((n_log, 3 + 6, root))
        else:
            ch.close("LOC")
            draws = ch.positions()
            trees.append((n_log, 6 + 3 if last else 6 + 3 + 6, root))
            col = ch.opening(m - 2, trees[m - 2][0], trees[m - 2][1], trees[m - 2][2], draws)
            kept = old["kept"]
            # Offsets in row 1 of G: folded mask row (0), folded fresh mask (A1); random tails of
            # the folded main row (T), the kept row (TK), the two masks (T0, T1); the old pad (OP).
            A1, T = lz, 2 * lz
            TK = T + r
            T0 = TK + (r if kept else 0)
            T1 = T0 + rz
            OP = T1 + rz
            # the carried claim
            G, Pp = {}, {}
            for j in range(l):
                G[2 * j] = old["c"][j]
            for j in range(lz):
                G[1 + 2 * j], G[1 + 2 * (A1 + j)] = old["d0"][j], old["d1"][j]
            if kept:
                for j in range(l - 3):
                    Pp[j] = old["K"][j]
                for t in range(3):
                    G[1 + 2 * (OP + t)] = old["K"][l - 3 + t]
            plain.append(({"G": G, "Pp": Pp}, old["tau"]))
            if kept:   # the kept message's old sample claim, read on Pp, and in G at the old pad
                u_old, y_old = old["sample"]
                plain.append(({"Pp": {j: u_old[j] for j in range(l - 3)}, "G": {1 + 2 * (OP + t): u_old[l - 3 + t] for t in range(3)}}, y_old))
            eg = eq_table(old["gamma"])
            for n, x in enumerate(draws):
                cc = col[x]
                if m == 2:    # tree 1: x_1, b_1 | c_1 | B_1's mask row | U_1
                    main, v, at = [emb(cc[0]), emb(cc[1])], emb(cc[2]), 3
                else:         # G (2 rows over E) | Pp | b | c | the mask row | U
                    g0, g1 = tuple(cc[0:3]), tuple(cc[3:6])
                    if m == 3:
                        main, v = [g0, g1, emb(cc[6]), emb(cc[7])], emb(cc[8])
                    else:
                        main, v = [g0, g1, emb(cc[6]), emb(cc[7]), emb(cc[8]), ZERO, ZERO, ZERO], None
                    at = 9
                ys = (inner(eg, main), tuple(cc[at:at + 3]), tuple(cc[at + 3:at + 6]))
                if n == 0:
                    notes += [("CS: y_main at the first draw", ys[0]), ("CS: y_0 at the first draw", ys[1]), ("CS: y_1 at the first draw", ys[2])]
                gens.append((x, ys, v))
        # The sub-proof's three claims: hiding polynomial and c; x at gamma; b at gamma.
        if not last:
            eps, gamma, by, claim, v1, v2 = psc(ch, V)
            e = eq_table(gamma)
            # x: message, row, row bits; b: row, row bits
            xo, xrow, xrb = ("B", 0, 1) if m == 1 else ("Pp", 0, 0)
            brow, brb = (1, 1) if m == 1 else (0, 0)
            dU = {}
            for t in range(V):
                dU[2 * t], dU[2 * t + 1] = gamma[t], mul(gamma[t], gamma[t])
            dU[2 * V], dU[2 * V + 1], dU[2 * V + 2] = ONE, by, mul(by, by)
            plain.append(({"P": {i: sub(ZERO, mul(eps, e[i])) for i in range(l // 2)}, "Bz": dU}, sub(claim, mul(eps, mul(v1, v2)))))
            plain.append(({xo: {xrow + (i << xrb): e[i] for i in range(l // 2)}, "Bz": {2 * V + 3: by}}, v1))
            plain.append(({"B": {brow + (i << brb): e[i] for i in range(l // 2)}, "Bz": {2 * V + 4: by}}, v2))
        samples = []
        for o in sampled:
            if o == "B":
                u, w = tc.sample_row(z[:max(bits_B, log2(lz)) + 1], beta, bits_B, log2(lz))
                samples.append(({"B": u, "Bz": w}, answers[o]))
            else:
                samples.append(({o: sample_one_table(z, beta, 1 + ll if o == "G" else ll, o == "G")}, answers[o]))
        ch.close("BAT")
        theta = ch.element()
        n_rows = len(plain) + len(samples) + len(gens) * (4 if (old and old["kept"]) else 3)
        assert ch.entry["rows"] == n_rows, (ch.entry, n_rows)
        acc = {o: [ZERO] * s for o, s in size.items()}
        tau, f = ZERO, ONE
        for forms, target in plain + samples:
            for o, form in forms.items():
                axpy(acc[o], f, form)
            tau, f = add(tau, mul(f, target)), mul(f, theta)
        # For each draw, 3 or 4 claims: the rows in G (and Pp), encoded, match the opening.
        for x, ys, v in gens:
            pw = powers(point(x), max(r + l, rz + lz))
            G = acc["G"]
            for j in range(l):
                G[2 * j] = add(G[2 * j], scale(pw[r + j], f))
            for j in range(r):
                G[1 + 2 * (T + j)] = add(G[1 + 2 * (T + j)], scale(pw[j], f))
            tau, f = add(tau, mul(f, ys[0])), mul(f, theta)
            if old["kept"]:
                Pp = acc["Pp"]
                for j in range(l - 3):
                    Pp[j] = add(Pp[j], scale(pw[r + j], f))
                for t in range(3):
                    G[1 + 2 * (OP + t)] = add(G[1 + 2 * (OP + t)], scale(pw[r + l - 3 + t], f))
                for j in range(r):
                    G[1 + 2 * (TK + j)] = add(G[1 + 2 * (TK + j)], scale(pw[j], f))
                tau, f = add(tau, mul(f, v)), mul(f, theta)
            for which, (A, TT) in enumerate(((0, T0), (A1, T1))):
                for j in range(lz):
                    G[1 + 2 * (A + j)] = add(G[1 + 2 * (A + j)], scale(pw[rz + j], f))
                for j in range(rz):
                    G[1 + 2 * (TT + j)] = add(G[1 + 2 * (TT + j)], scale(pw[j], f))
                tau, f = add(tau, mul(f, ys[1 + which])), mul(f, theta)
        # onto the group's main table: (message, first row, row bits); a kept P has no row
        k = {1: 1, 2: 2, 3: 3, 4: 1}[m]
        c = [ZERO] * (l << k)
        blocks = {1: [("B", 0, 1)], 2: [("G", 0, 1), ("Pp", 2, 0), ("B", 3, 0)], 3: [("G", 0, 1), ("Pp", 2, 0), ("B", 3, 0), ("P", 4, 0)], 4: [("G", 0, 1)]}[m]
        for o, first, rb in blocks:
            for i, cf in enumerate(acc[o]):
                if cf != ZERO:
                    c[first + (i & ((1 << rb) - 1)) + ((i >> rb) << k)] = cf
        d = acc["Bz"]
        if last:
            break
        notes.append(("REQ: the group row's target", tau))
        sigma, = ch.recv("FOLD: the hiding polynomial's sum")
        ch.close("FOLD epsilon")
        feps = ch.element()
        g, claim = rounds_of_fold(ch, k, add(sigma, mul(feps, tau)))
        notes.append(("REQ: FOLD's target", claim))
        kept = m < 3
        fk = mul(feps, prod([sub(ONE, x) for x in g]))
        fd = mul(feps, prod(g))
        # FOLD leaves one claim: forms on the folded main row (c), the kept row (K), the folded mask
        # row (d0) and the fresh mask (d1).
        old = dict(c=[mul(feps, v) for v in fold(c, g)], K=[mul(fk, v) for v in acc["P"]] if kept else None, d0=[mul(fd, v) for v in d],
                   d1=tc.mask_weights(g, k, 0, lz), tau=claim, gamma=g, kept=kept, sample=[(f_["P"], y_) for f_, y_ in samples if "P" in f_][0])
    # the last proximity test, on the group of G_4
    # `ride`: if set, later masks share the tree of the CS (the code switch) before them
    notes.append(("BAT: target", tau))
    st, ride = q["last"], q["choice"][1]
    Cs, Zs = [par["base"][s[0]] for s in st], [par["mask"][s[1]] for s in st]
    assert len(st) == 3 and Cs[0] == C and Zs[0] == Z and st[0][2:4] == (1, 0)
    own = [3]            # numbers of the trees with the present stage's rows and its fresh mask
    for s in range(2):
        k, h = st[s][2], st[s][3]
        k2, h2 = st[s + 1][2], st[s + 1][3]
        if s > 0 and not ride:
            trees.append((Zs[s][0], 3 << h, ch.recv_root("FOLD: the root of the fresh mask's own tree")))
            own.append(len(trees) - 1)
        g, c0, d0, d1, tau = tc.fold_block(ch, k, h, c, d, tau, Zs[s][1])
        notes.append(("FOLD: target", tau))
        l, r, lz, rz = Cs[s][1], Cs[s][2], Zs[s][1], Zs[s][2]
        assert Cs[s + 1][1] << k2 == l and (Zs[s + 1][1] << h2) >= 2 * lz + r + 2 * rz + 1
        width = (3 << k2) + (3 << h2) + (((3 << h2) if s == 0 else 6) if ride else 0)
        trees.append((Cs[s + 1][0], width, ch.recv_root("CS: the root of the next stage's tree")))
        nxt = [len(trees) - 1]
        ch.close("SEND seed")
        bits_main, bits_mask = log2(l), h2 + log2(Zs[s + 1][1])
        z, beta = tc.seed(ch, max(bits_main, bits_mask) + 1)
        alpha, = ch.recv("CS: the answer to the sample")
        ch.close("CS")
        draws = ch.positions()
        theta = ch.element()
        cols = [ch.opening(t, trees[t][0], trees[t][1], trees[t][2], draws) for t in own]
        assert ch.entry["rows"] == 2 + 3 * len(draws)
        room = Zs[s + 1][1] << h2
        A1, T, T0, T1 = lz, 2 * lz, 2 * lz + r, 2 * lz + r + rz
        nc, nd = list(c0), [ZERO] * room
        for j in range(lz):
            nd[j], nd[A1 + j] = d0[j], d1[j]
        ntau = tau
        u, w = tc.sample_row(z, beta, bits_main, bits_mask)
        f = theta
        nc = [add(a_, mul(f, b_)) for a_, b_ in zip(nc, u)]
        nd = [add(a_, mul(f, b_)) for a_, b_ in zip(nd, w)]
        ntau = add(ntau, mul(f, alpha))
        eg, egh = eq_table(g), eq_table(g[:h])
        for n, xx in enumerate(draws):
            cc = cols[0][xx]
            main = [tuple(cc[3 * b:3 * b + 3]) for b in range(1 << k)]
            at = 3 << k
            if s == 0:     # the group of G_4 has no mask row: read as zero
                oldm = [ZERO]
            else:
                oldm = [tuple(cc[at + 3 * b:at + 3 * b + 3]) for b in range(1 << h)]
                at += 3 << h
            src, at = (cc, at) if len(own) == 1 else (cols[1][xx], 0)
            fresh = [tuple(src[at + 3 * b:at + 3 * b + 3]) for b in range(1 << h)]
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
        c, d, tau, own = nc, nd, ntau, nxt
        notes += [("CS: target", tau), ("CS: c'[0]", c[0]), ("CS: d'[0]", d[0]), ("CS: d's last entry", d[-1])]
    l, r, lz, rz = Cs[2][1], Cs[2][2], Zs[2][1], Zs[2][2]
    if not ride:
        trees.append((Cs[2][0], 6, ch.recv_root("BASE: the root of the masking word's own tree")))
        own.append(len(trees) - 1)
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
    cols = [ch.opening(t, trees[t][0], trees[t][1], trees[t][2], draws) for t in own]
    for xx in sorted(set(draws)):
        if ride:
            g3m, gsm, g3z, gsz = (tuple(cols[0][xx][3 * i:3 * i + 3]) for i in range(4))
        else:
            g3m, g3z = (tuple(cols[0][xx][3 * i:3 * i + 3]) for i in range(2))
            gsm, gsz = (tuple(cols[1][xx][3 * i:3 * i + 3]) for i in range(2))
        if tc.symbol(Cs[2], xx, M0, R0) != add(gsm, mul(lam, g3m)) or tc.symbol(Zs[2], xx, M1, R1) != add(gsz, mul(lam, g3z)):
            raise Refused("BASE: a check at a position fails")
    ch.finish()
    return ch


def main():
    folder, run = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "links"
    if len(sys.argv) > 5 and sys.argv[4] == "--cheats":
        return tc.cheats(folder, run, sys.argv[5], sys.argv[6], parse, verify)
    return tc.report(folder, run, parse, verify)


if __name__ == "__main__":
    sys.exit(main())
