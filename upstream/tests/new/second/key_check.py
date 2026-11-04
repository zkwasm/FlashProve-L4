#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""A second verifier of a key file: a committed table's root and the committer's one-time proofs
about that table. It calls no code of blocks/. tests/new/second/files_check.py compares its
verdicts and notes with the C++ verifier's (check_key() in blocks/frame/check_key.h).

The file: header ("head" in `marks`), table's root, record's answers (stored answers to samples),
one-time hash chain, place (leaf number and salt in the binding tree), shared chain, second chain.
This verifier follows docs/format/toys/3-key-and-entry.md, not word for word as that page stands: parts 6 (the
key file; WF, the committer's range proof), 7 (LKP, the lookup), 8 (BATT, which batches WF's claims)
and 9 (LIN, the test on the committed table: tests/new/second/lin_check.py). Their "row" is a linear
claim (`Row` in tests/new/second/zkpy.py).

This check is not independent of the C++ programs on these points, where both rest on one reading
of the same text or rule, so that the comparison cannot show an error in that reading: those that
the docstring of tests/new/second/toy_check.py names; the order of the calls
(blocks/frame/check_key.h and its test); the first lines of blocks/iop/tests/wf_world.h; and, for
the false key files, lines of blocks/iop/wf.h (`WfLookup`, `WfMatrix`, the start of `WfPublic`),
blocks/iop/lkp.h (the malicious provers' globals), blocks/iop/tests/wf_test.cpp (last_key_level())
and one switch in blocks/iop/base_case.h (globals, routine and switch are not among these sources). Nor
is it independent of them in the parsers, the names of chain and test in a `Refused` message, the
record of one value at each tree vertex (zkpy.opening()), or the notes (A.note(); no decision reads
them): "LKP: a part's key claim" differed from the C++ programs' list at the first comparison and
was then redefined to match part 7, 4.8; thirteen notes are under that list's names, and
"FOLDEQ: (Rec_0)'s target" in tests/new/second/lin_check.py too. No one
has examined whether this check is independent of the other parts of the C++ files named here or of
any other file of the C++ programs: of all of these nothing is known, either way.

usage: key_check.py <directory of format_rules.py> <directory> <stem>
Reads <stem>.file, .one-time.parameters, .shared.parameters, .second.parameters and
.one-time.statement; the comparison with the last is only printed: the exit code ignores it.
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
import zkpy  # noqa: E402
import lin_check as lc  # noqa: E402
from toy_check import ZERO, ONE, Refused, add, sub, mul, scale, emb, inner, eq_table, prod  # noqa: E402
from zkpy import V, Row, clog2, NONE  # noqa: E402

# R: docs/format/format_rules.py. Kind 1: the challenge is elements of E, the cubic extension.
R = tc.R
tc.KIND.update({"LKP": 1, "JOIN": 1, "LINE": 1, "BAT-sigma": 1})


def power(x, n):
    out = ONE
    while n:
        if n & 1:
            out = mul(out, x)
        x, n = mul(x, x), n >> 1
    return out


# The parameter item `ranges`; part 6, 5.1 and 5.2, define its fields.
def parse_ranges(value):
    v = tc.Bytes(value)
    words = lambda: [v.word() for _ in range(v.word())]  # noqa: E731
    out = dict(lookups=[])   # the matrix's size is in the item "the test on the table", not here
    for _ in range(v.word()):
        q = dict(d=v.word(), mu=v.word(), live=v.word(), digit_weight=words(), offset=v.word(), padding_key=v.word(), parts=[])
        for _ in range(v.word()):
            q["parts"].append((v.word(), [tuple(v.word() for _ in range(4)) for _ in range(v.word())]))
        q["first_bound"] = words()
        out["lookups"].append(q)
    out["notice"], out["number"] = tuple(v.word() for _ in range(7)), v.word()
    out["stages"] = [tuple(v.word() for _ in range(4)) for _ in range(v.word())]
    out["choice"] = (1, v.word())
    assert v.at == len(value)
    return out


# masks: mask terms, {(oracle message, index in its mask rows): coefficient}; more: as `Row.forms`.
def as_row(masks, target, more=None):
    forms = dict(more or {})
    for (o, index), cf in masks.items():
        forms.setdefault((o, "Z"), {})[index] = cf
    return Row(forms, target)


def psc(A, zs, ds, L_in, T, s, D, take):
    """PSC, the sumcheck with a hiding polynomial (part 7, 1.1). take(n_w, n_c) returns the fresh
    mask entries. Checks no equation: the caller makes row (U) from the result. U: that row's mask
    terms, without the kept map's term."""
    ch, k = A.ch, len(zs)
    n_w = 1 + sum(d - 1 for d in ds) + D
    fresh = take(n_w, s)
    mu, = ch.recv("PSC: the hiding polynomial's sum")
    ch.close("PSC epsilon")
    eps = ch.element()
    Rj, gamma, U, at = add(mu, mul(eps, T)), [], {}, 0
    for j in range(k):
        q = ch.recv("PSC: a round's coefficients", ds[j] - 1)
        ch.close("PSC round")
        x = ch.element()
        tot = ZERO
        for c in q:
            tot = add(tot, c)
        val, pw = sub(Rj, mul(zs[j], tot)), ONE
        if s == 0 and j == k - 1:      # only the last variable of the list has a constant
            U[fresh[at]] = ONE
            at += 1
        for c in q:
            pw = mul(pw, x)
            val = add(val, mul(c, pw))
            U[fresh[at]] = pw
            at += 1
        Rj = val
        gamma.append(x)
    beta, v, cs = None, [], []
    if s:
        p = ch.recv("PSC: the round in Y", D)
        ch.close("PSC Y")
        beta = ch.element(True)
        pw = ONE
        U[fresh[at]] = ONE
        at += 1
        for c in p:
            pw = mul(pw, beta)
            Rj = add(Rj, mul(c, pw))
            U[fresh[at]] = pw
            at += 1
        v = ch.recv("PSC: the end values", s)
        ch.close("PSC values")
        cs = fresh[at:at + s]
        at += s
    assert at == len(fresh)
    for key, cf in L_in.items():
        U[key] = add(U.get(key, ZERO), mul(eps, cf))
    return dict(eps=eps, gamma=gamma, beta=beta, R=Rj, v=v, c=cs, U=U)


def lkp(A, g, q, parts, pad_count, pad_value, own, key, planes):
    """LKP (part 7, sections 4 and 5) as WF uses it: one function, no shift, a range table
    0 .. 2^d - 1. Oracle message `key` gives the last two mask entries of each part's last level,
    `own` all others. Returns a key claim (z, mask terms, value) for each part, the padding's share
    still in the value."""
    ch, d, mu = A.ch, q["d"], q["mu"]
    n_planes, w, k = q["live"] or (1 << mu), 1 << mu, len(parts)
    N = sum(1 << v for v in parts)
    rows = A.groups[g]["plain"]
    from_own = lambda n_w, n_c: [(own, A.take(own)) for _ in range(n_w + n_c)]  # noqa: E731
    ch.close("LKP")
    alpha = ch.element()
    a = power(sub(alpha, emb(pad_value)), pad_count)
    A.note("LKP: alpha", alpha), A.note("LKP: the public factor a", a)
    # 4.1: the tie
    r = psc(A, [], [], {}, ZERO, 3, 3, from_own)
    rows.append(as_row(r["U"], sub(r["R"], mul(r["eps"], sub(r["v"][0], mul(a, mul(mul(r["v"][1], r["v"][1]), r["v"][2])))))))
    claim = lambda r_, i: ({r_["c"][i]: r_["beta"]}, r_["v"][i])  # noqa: E731
    Z, P, G = claim(r, 0), claim(r, 1), [claim(r, 2)]
    for j in range(1, n_planes - 1):
        r = psc(A, [], [], P[0], P[1], 2, 3, from_own)
        rows.append(as_row(r["U"], sub(r["R"], mul(r["eps"], mul(mul(r["v"][0], r["v"][0]), r["v"][1])))))
        P = claim(r, 0)
        G.append(claim(r, 1))
    G.append(P)
    # 4.1b: the parts' roots
    roots, Q = [], Z
    for i in range(k - 1):
        r = psc(A, [], [], Q[0], Q[1], 2, 2, from_own)
        rows.append(as_row(r["U"], sub(r["R"], mul(r["eps"], mul(r["v"][0], r["v"][1])))))
        roots.append(claim(r, 0))
        Q = claim(r, 1)
    roots.append(Q)
    ch.close("JOIN")
    rr = [ch.element() for _ in range(mu)]
    e = eq_table(rr)
    value, form = ZERO, {}
    for j in range(w):
        value = add(value, mul(e[j], G[j][1]) if j < n_planes else e[j])
        if j < n_planes:
            for key_, cf in G[j][0].items():
                form[key_] = add(form.get(key_, ZERO), mul(e[j], cf))
    A.note("LKP: the table side's first claim", value)

    # 4.3, 4.4: one PSC and one LINE for each level of a product tree
    def levels(pi, L, y, n_levels, first_vars, last_order, last_from_key):
        for l in range(n_levels):
            n = first_vars + l
            order = list(range(n - 1, -1, -1))
            lastly = l == n_levels - 1
            if lastly and last_order is not None:
                order = list(last_order) + [t for t in order if t not in last_order]
            take = (lambda n_w, n_c: [(own, A.take(own)) for _ in range(n_w)] + [(key, A.take(key)) for _ in range(n_c)]) if lastly and last_from_key else from_own
            r_ = psc(A, [pi[t] for t in order], [3] * n, L, y, 2, 2, take)
            rows.append(as_row(r_["U"], sub(r_["R"], mul(r_["eps"], mul(r_["v"][0], r_["v"][1])))))
            back = [None] * n
            for j, t in enumerate(order):
                back[t] = r_["gamma"][j]
            ch.close("LINE")
            rho = ch.element()
            pi = back + [rho]
            L = {r_["c"][0]: mul(sub(ONE, rho), r_["beta"]), r_["c"][1]: mul(rho, r_["beta"])}
            y = add(mul(sub(ONE, rho), r_["v"][0]), mul(rho, r_["v"][1]))
        return pi, L, y

    ends = [levels([], roots[p][0], roots[p][1], parts[p], 0, q["first_bound"], True) for p in range(k)]
    pi, L, y = levels(rr, form, value, d, mu, None, False)
    # 4.5: the leaves
    n = mu + d
    order = list(range(n - 1, -1, -1))
    r = psc(A, [pi[t] for t in order], [3 if t >= mu else 2 for t in order], L, y, 0, 0, from_own)
    back = [None] * n
    for j, t in enumerate(order):
        back[t] = r["gamma"][j]
    tau = ZERO
    for j in range(d):
        tau = add(tau, scale(1 << j, back[mu + j]))
    A1 = sub(sub(alpha, ONE), tau)
    A.note("LKP: the leaves' factor", A1)
    f = mul(r["eps"], A1)
    rows.append(as_row(r["U"], sub(r["R"], r["eps"]), planes([mul(f, x) for x in eq_table(back)])))
    # 4.6: the bits are bits
    ch.close("ZERO")
    zeta = [ch.element() for _ in range(n)]
    r = psc(A, [zeta[t] for t in order], [3] * n, {}, ZERO, 1, 2, from_own)
    for j, t in enumerate(order):
        back[t] = r["gamma"][j]
    rows.append(as_row(r["U"], sub(r["R"], mul(r["eps"], sub(mul(r["v"][0], r["v"][0]), r["v"][0])))))
    rows.append(as_row({r["c"][0]: r["beta"]}, r["v"][0], planes(eq_table(back))))
    rows.append(as_row({}, emb(N - pad_count), planes([emb(1 << (i & (w - 1))) for i in range(1 << n)])))
    out = []
    for z, L_, y_ in ends:
        out.append((z, {key_: sub(ZERO, cf) for key_, cf in L_.items()}, sub(alpha, y_)))
    return out


def wf(A, W, masks0):
    """WF (part 6, section 5): the multiplicities' oracle message, the lookups, WF's own proximity
    test (`iopp`). Returns the pending claims (g, psi, mask terms, target); only these tie the
    lookups' keys to the committed matrix."""
    ride, k_col = W["choice"][1], W["k_col"]
    st = W["stages"]
    ids = A.send([W["notice"]] + ([(NONE, st[0][1], 0, 1, 1 << st[0][3], 0, W["notice"][6])] if ride else []))
    o1 = ids[0]
    assert o1 == W["number"]
    l, rb = A.base[W["notice"][0]][1], clog2(W["notice"][2])
    # 5.3: the layout of the multiplicities' bits in the message
    blocks = []
    for n, q in enumerate(W["lookups"]):
        live, first = q["live"] or (1 << q["mu"]), 0
        while first < live:
            b = 0
            while first + (2 << b) <= live and first % (2 << b) == 0:
                b += 1
            blocks.append(dict(lookup=n, first=first, b=b, size=1 << (b + q["d"])))
            first += 1 << b
    row, place = 0, 0
    for blk in sorted(blocks, key=lambda x: -x["size"]):
        if place + blk["size"] > l:
            row, place = row + 1, 0
        blk["row"], blk["place"] = row, place
        place += blk["size"]
    g = A.new_group(W["notice"][0], W["notice"][1])
    A.join(g, o1)
    pending = []
    for n, q in enumerate(W["lookups"]):
        mu, d = q["mu"], q["d"]

        def planes(vec, n=n, mu=mu, d=d):
            form = {}
            for blk in blocks:
                if blk["lookup"] == n:
                    for j in range(1 << blk["b"]):
                        for t in range(1 << d):
                            form[blk["row"] + ((blk["place"] + j + (t << blk["b"])) << rb)] = vec[blk["first"] + j + (t << mu)]
            return {(o1, "M"): form}

        parts = [lp + k_col for lp, _ in q["parts"]]
        covered_places = [sum(r[2] for r in runs) for _, runs in q["parts"]]
        pad = sum(((1 << lp) - c) << k_col for (lp, _), c in zip(q["parts"], covered_places))
        for (lp, runs), (z, L, value) in zip(q["parts"], lkp(A, g, q, parts, pad, q["padding_key"], o1, masks0, planes)):
            e = eq_table(z[k_col:])
            gw, covered = [ZERO] * W["matrix_rows"], ZERO
            for first_place, first_row, places, stride in runs:
                for p in range(places):
                    covered = add(covered, e[first_place + p])
                    for qq, dw in enumerate(q["digit_weight"]):
                        at = first_row + p + qq * stride
                        gw[at] = add(gw[at], scale(dw, e[first_place + p]))
            handed = sub(value, scale(q["padding_key"], sub(ONE, covered)))   # part 7, 4.8: LKP's target, the padding's share taken out: the places of padding have the weight 1 - covered
            A.note("LKP: a part's key claim", handed)
            target = sub(handed, scale(q["offset"], covered))
            pending.append((gw, z[:k_col], L, target))
    A.iopp(g, st, ride, ids[1] if ride else None, "the one-time chain, WF's own test")
    return pending


def batt(A, rows, ledger, K):
    """BATT (part 8, section 2): batches the claims `rows`, each (g, psi, mask terms, target), into
    one of the same form. K: len(psi). Its mask entries come from oracle message `ledger`."""
    ch = A.ch
    ch.close("BAT-sigma")
    assert ch.entry["rows"] == len(rows)
    x = ch.element()
    chi = [power(x, j) for j in range(len(rows))]
    T = ZERO
    for c, r in zip(chi, rows):
        T = add(T, mul(c, r[3]))
    w = [(ledger, A.take(ledger)) for _ in range(1 + 2 * K)]
    mu, = ch.recv("BATT: the hiding polynomial's sum")
    ch.close("PSC epsilon")
    eps = ch.element()
    Rt, gamma = add(mu, mul(eps, T)), []
    for t in range(K):
        q0, q2 = ch.recv("BATT: a round's constant and quadratic coefficient", 2)
        ch.close("PSC round")
        gm = ch.element()
        q1 = sub(sub(Rt, scale(2, q0)), q2)
        Rt = add(q0, mul(gm, add(q1, mul(gm, q2))))
        gamma.append(gm)
    P = prod(gamma)
    masks = {}
    for l in range(K - 1):
        G = prod(gamma[l + 1:])
        masks[w[2 * l]], masks[w[2 * l + 1]] = mul(gamma[l], G), mul(mul(gamma[l], gamma[l]), G)
    masks[w[2 * K - 2]], masks[w[2 * K - 1]], masks[w[2 * K]] = ONE, gamma[K - 1], mul(gamma[K - 1], gamma[K - 1])
    g = [ZERO] * len(rows[0][0])
    for c, (gj, psi, mj, _) in zip(chi, rows):
        eqv = ONE
        for zt, xt in zip(psi, gamma):
            eqv = mul(eqv, add(mul(zt, xt), mul(sub(ONE, zt), sub(ONE, xt))))
        aj = mul(mul(eps, c), eqv)
        g = [add(u, mul(aj, v)) for u, v in zip(g, gj)]
        for key, cf in mj.items():
            masks[key] = add(masks.get(key, ZERO), mul(mul(mul(eps, P), c), cf))
    A.note("BATT: target", Rt), A.note("BATT: gamma_0", gamma[0])
    A.note("BATT: coefficient of its first mask entry", masks[w[0]])
    last = list(rows[-1][2].items())[-1]
    A.note("BATT: coefficient of the last mask term", mul(mul(mul(eps, P), chi[-1]), last[1]))
    return g, gamma, masks, Rt


# The parameters that read() returns (plans, shapes, configuration digest) are taken as given;
# main() in tests/new/second/files_check.py compares the digest with the public files. Returns the
# one-time and second chains' `V`, the shared chain's, and whether the statement formed here is the
# one given.
# tests/new/second/run_honest.sh runs it on the toy's key files (one stored answer for each row: a = 1).
# Its altered copy of the record's first answer is refused at the first proof of work, because the
# answers stand in the chain's first query: that copy does not reach the rows on the record.
def verify(read):
    zkpy.new_session()
    raws = [read(c + ".parameters") for c in ("first", "shared", "second")]
    P1, P2, P3 = (lc.parse(r) for r in raws)
    d1 = dict(P1["items"])
    S, conf, salt = P1["S"], P1["items"][0][1], P1["head"][2]
    f = tc.Bytes(read("file"))
    if f.take(4) != b"CPIK":
        raise Refused("a file of another kind")
    if f.take(4) != b"H821":
        raise Refused("another version")
    root = f.take(32)
    (nT, lT, rT), v, a = S["code"], S["v"], S["a"]
    y = [tc.ext_of(f.take(24)) for _ in range(a << v)]
    table_item = R.u64(32) + root + R.u64(len(y)) + b"".join(R.ext(e) for e in y)   # the root as a string of bytes, the list of answers; name and seeds are computed where they are needed (lin_check.Lin)
    given = read("first.statement")
    st1 = R.item("table", table_item)
    same = st1 == given
    L = lc.Lin(P1, table_item)
    marks = [("head", 0), ("the table's root", 8), ("the record's answers", 40), ("the one-time chain", f.at)]
    ch = tc.Run(P1, raws[0], None, f, label=0x46524d02, statement=st1)
    A = V(ch, L.base, L.mask)
    masks0, = A.send([struct.unpack("<7Q", d1["index-mask message"])])
    W = parse_ranges(d1["ranges"])
    W["k_col"], W["matrix_rows"] = L.k_col, L.m
    pending = wf(A, W, masks0)
    g, psi, masks, tau = batt(A, pending, masks0, L.k_col) if len(pending) > 1 else pending[0]
    A.note("member: the claim's target", tau)
    F = lc.lin_first(A, L, g, psi, list(masks.items()), tau)
    seal1 = ch.seal()
    ch.finish(last=False)
    A.first_notes = len(A.notes)
    # a key file's binding tree has one leaf
    dd, salt_bytes = struct.unpack("<2Q", d1["binding tree"][:16])
    assert dd == 0
    marks.append(("the place", f.at))
    broot = lc.binding_root(conf, dd, salt_bytes, f, seal1)
    marks.append(("the shared chain", f.at))
    B, Om, star, seal2, ch2 = lc.lin_shared(L, P2, raws[1], f, broot)
    ch2.finish(last=False)
    marks.append(("the second chain", f.at))
    A.marks = marks + [("the end", len(f.b))]
    ch3 = lc.lin_second(A, B, L, F, Om, star, P3, raws[2], f, seal1, seal2)
    ch3.finish()
    A.runs, A.seals, A.binding_root = [("key.one-time", ch), ("key.shared", ch2), ("key.second", ch3)], (seal1, seal2), broot
    return A, B, same


def main():
    folder, stem = sys.argv[2], sys.argv[3]
    names = {"first.parameters": "one-time.parameters", "first.statement": "one-time.statement"}   # as the tool blocks/frame/handover_files.cpp names its files
    read = lambda tail: open("%s/%s.%s" % (folder, stem, names.get(tail, tail)), "rb").read()  # noqa: E731
    try:
        A, B, same = verify(read)
        print("%s: ACCEPTED by the second verifier (%d bytes); the one-time chain's statement, formed by the reader, is the C++ Binder's: %s" % (stem, len(read("file")), same))
        return 0
    except Refused as why:
        print("%s: REFUSED by the second verifier: %s" % (stem, why))
        return 1


if __name__ == "__main__":
    sys.exit(main())
