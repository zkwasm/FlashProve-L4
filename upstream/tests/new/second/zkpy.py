#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Code shared by second verifiers in this directory, which re-implement verifiers of blocks/.

For runs with several oracle messages (encoded rows, sent as Merkle roots), trees, groups and hash
chains: which oracle messages share a tree (R7.12, R7.13), one value at a vertex (R7.14), sampling
(SEND), groups, batching (BAT) and splitting (SPLIT) of linear claims (`Row`), and the proximity
test (`iopp`). Mask codes may be shorter than base codes. R-numbers: rules of
docs/format/FORMAT.md. Goldilocks only.

Written from the texts in docs/format/toys/ (parts 1, 3, 4, 5, 9), not word for word as these pages stand, with
their writers' written answers to questions, and from those rules. It calls no code of blocks/.
Field arithmetic, codes' points and the channel (`Run`) are tests/new/second/toy_check.py's.
"""
import toy_check as tc
from toy_check import ZERO, ONE, Refused, add, sub, mul, scale, emb, inner, eq_table, fold, log2, point, powers, prod

R = tc.R
# in a notice (R12.3): the code of a part without rows
NONE = (1 << 32) - 1
# X[t] is X^t in E = F[X]/(X^3 - 2)
X = (ONE, (0, 1, 0), (0, 0, 1))


def clog2(n):
    return (n - 1).bit_length() if n > 0 else 0


# An oracle message, made from its notice (R12.3). `deg`: 1 for main rows over F, 3 over E.
# An index into a part is row + 2^rb * entry (mask part: 2^zb).
class Oracle:
    def __init__(self, oid, notice):
        self.id = oid
        self.code, self.mask_code, self.main_rows, self.deg, self.mask_rows, self.answers, self.leaf_class = notice
        self.rb, self.zb, self.next = clog2(self.main_rows), clog2(self.mask_rows), 0


# V.receive() sets the fields; for a stored tree (made once, read by many runs) the caller sets
# those that opening() reads.
class Tree:
    pass


# A linear claim. `forms`: {(oracle id, "M" main or "Z" mask part): {index: coefficient}}.
class Row:
    def __init__(self, forms, target):
        self.forms, self.target = forms, target


class V:
    """A verifier's state. To continue it in a later chain, a caller replaces the channel `ch`.

    `base`, `mask`: lists of codes, each (log2 of the length, message entries, random tail's
    entries).
    """

    def __init__(self, ch, base, mask):
        self.ch, self.base, self.mask = ch, base, mask
        self.oracles, self.trees, self.groups, self.samples, self.number, self.notes = [], [], [], {}, 0, []

    # ---- trees ----
    def receive(self, notices):
        ids, fresh = [], []
        for n in notices:
            self.oracles.append(Oracle(len(self.oracles), n))
            ids.append(len(self.oracles) - 1)
        for with_main in (True, False):
            for i in ids:
                o = self.oracles[i]
                if (o.main_rows > 0) != with_main:
                    continue
                for t in fresh:
                    if t.leaf_class == o.leaf_class and (not with_main or (t.has_main and t.depth == self.base[o.code][0])):
                        break
                else:
                    t = Tree()
                    t.leaf_class, t.has_main, t.parts, t.cols, t.stored = o.leaf_class, with_main, [], {}, False
                    t.depth = self.base[o.code][0] if with_main else self.mask[o.mask_code][0]
                    fresh.append(t)
                t.parts.append(i)
                o.tree = t
        for t in fresh:
            t.parts.sort()
            t.number, t.start = self.number, self.ch.run
            self.number += 1
            t.root = self.ch.recv_root("the root of tree %d" % t.number)
            at = 0
            t.main_at, t.mask_at = {}, {}
            for i in t.parts:
                t.main_at[i] = at
                at += self.oracles[i].main_rows * self.oracles[i].deg
            for i in t.parts:
                t.mask_at[i] = at
                at += self.oracles[i].mask_rows * 3
            t.width = at
            self.trees.append(t)
        return ids

    def open(self, ids, draws):
        """Opens these messages' trees in the order of `self.trees`.

        A tree of depth b is opened at the draws mod 2^b.
        """
        for t in self.trees:
            if any(self.oracles[i].tree is t for i in ids):
                got = opening(self.ch, t, [x & ((1 << t.depth) - 1) for x in draws])
                t.cols.update(got)   # a column opened again has the same leaf digest (`opening`)

    def main_symbols(self, i, x):
        o = self.oracles[i]
        c, at = o.tree.cols[x], o.tree.main_at[i]
        return [emb(c[at + b]) if o.deg == 1 else tuple(c[at + 3 * b:at + 3 * b + 3]) for b in range(o.main_rows)]

    def mask_symbols(self, i, x):
        o = self.oracles[i]
        c, at = o.tree.cols[x & ((1 << o.tree.depth) - 1)], o.tree.mask_at[i]
        return [tuple(c[at + 3 * b:at + 3 * b + 3]) for b in range(o.mask_rows)]

    # ---- SEND ----
    def bits(self, o):
        return (o.rb + log2(self.base[o.code][1]) if o.main_rows else None, o.zb + log2(self.mask[o.mask_code][1]) if o.mask_rows else None)

    def cube(self, o):
        m, z = self.bits(o)
        return max(m, z) + 1 if m is not None and z is not None else (m if z is None else z)

    def pad(self, o):
        """o's pad as [(part, index, factor)]: its sample's coefficient there is beta * factor.

        The last entry of the last mask row or, without mask rows, of the last main row (over F: the
        last three entries).
        """
        if o.mask_rows:
            return [("Z", (o.mask_rows - 1) + ((self.mask[o.mask_code][1] - 1) << o.zb), ONE)]
        l = self.base[o.code][1]
        if o.deg == 3:
            return [("M", (o.main_rows - 1) + ((l - 1) << o.rb), ONE)]
        return [("M", (o.main_rows - 1) + ((l - 3 + t) << o.rb), X[t]) for t in range(3)]

    # SEND: receives oracle messages and samples those with an answer at one seed (z, beta).
    # If own_round is false, the answers are hashed in the caller's next round.
    def send(self, notices, own_round=True):
        ids = self.receive(notices)
        self.ch.close("SEND seed")
        sampled = [self.oracles[i] for i in ids if self.oracles[i].answers]
        assert all(o.answers == 1 for o in sampled)
        z, beta = tc.seed(self.ch, max(self.cube(o) for o in sampled))
        answers = self.ch.recv("answers to the sample", len(sampled))
        if own_round:
            self.ch.close("SEND answer")
        for o, y in zip(sampled, answers):
            m, zz = self.bits(o)
            forms = {}
            if m is not None and zz is not None:
                u, w = tc.sample_row(z[:max(m, zz) + 1], beta, m, zz)   # w[-1] = beta: undone below
                w[-1] = mul(mul(z[max(m, zz)], prod([sub(ONE, z[t]) for t in range(zz, max(m, zz))])), eq_table(z[:zz])[-1])
                forms[(o.id, "M")], forms[(o.id, "Z")] = dict(enumerate(u)), dict(enumerate(w))
            elif zz is None:
                forms[(o.id, "M")] = dict(enumerate(eq_table(z[:m])))
            else:
                forms[(o.id, "Z")] = dict(enumerate(eq_table(z[:zz])))
            for part, index, factor in self.pad(o):
                forms[(o.id, part)][index] = mul(beta, factor)
            self.samples[o.id] = Row(forms, y)
        return ids

    def take(self, i):
        """The next unused mask entry of message i. Without mask rows the loop does not end."""
        o = self.oracles[i]
        while (o.next & ((1 << o.zb) - 1)) >= o.mask_rows:
            o.next += 1
        assert o.next < self.pad(o)[0][1]
        o.next += 1
        return o.next - 1

    # ---- groups ----
    # `plain`, `gens`: claims batched before and after the members' sample claims
    def new_group(self, code, mask_code):
        self.groups.append(dict(code=code, mask_code=mask_code, members=[], plain=[], gens=[]))
        return len(self.groups) - 1

    def join(self, g, i):
        self.groups[g]["members"] = sorted(set(self.groups[g]["members"] + [i]))

    def layout(self, g):
        out = {}
        for part, rows, rb in (("M", "main_rows", "rb"), ("Z", "mask_rows", "zb")):
            blocks = sorted([(-(1 << getattr(self.oracles[i], rb)), n, i) for n, i in enumerate(self.groups[g]["members"]) if getattr(self.oracles[i], rows)])
            at = 0
            for size, _, i in blocks:
                out[(i, part)] = at
                at -= size
            out[part] = at
        h = clog2(out["Z"])
        return max(clog2(out["M"]), h, 1), h, out

    def combine(self, g, theta):
        G = self.groups[g]
        k, h, first = self.layout(g)
        l, lz = self.base[G["code"]][1], self.mask[G["mask_code"]][1]
        c, d, tau, f = [ZERO] * (l << k), [ZERO] * (lz << h), ZERO, ONE
        rows = G["plain"] + [self.samples[i] for i in G["members"] if i in self.samples] + G["gens"]
        assert self.ch.entry["rows"] == len(rows), (self.ch.entry, len(rows))
        for row in rows:
            for (i, part), form in row.forms.items():
                assert i in G["members"], "a row on an oracle message that is no member of the group"
                o = self.oracles[i]
                rb, kk, into = (o.rb, k, c) if part == "M" else (o.zb, h, d)
                for index, cf in form.items():
                    if cf != ZERO:
                        at = first[(i, part)] + (index & ((1 << rb) - 1)) + ((index >> rb) << kk)
                        into[at] = add(into[at], mul(f, cf))
            tau, f = add(tau, mul(f, row.target)), mul(f, theta)
        return k, h, first, c, d, tau

    # SPLIT of a claim on g1 and g2, with a mask entry of K, a member of both.
    def split(self, row, g1, g2, K):
        kappa = self.take(K)
        sigma, = self.ch.recv("SPLIT: the share")
        self.ch.close("SPLIT")
        one = {key: form for key, form in row.forms.items() if key[0] in self.groups[g1]["members"]}
        two = {key: form for key, form in row.forms.items() if key[0] not in self.groups[g1]["members"]}
        assert all(key[0] in self.groups[g2]["members"] for key in two) and (K, "Z") not in row.forms
        one[(K, "Z")], two[(K, "Z")] = {kappa: ONE}, {kappa: sub(ZERO, ONE)}
        self.groups[g1]["plain"].append(Row(one, sigma))
        self.groups[g2]["plain"].append(Row(two, sub(row.target, sigma)))

    def note(self, what, value):
        self.notes.append((what, value))

    # ---- the proximity test ----
    def iopp(self, g, st, ride, mask=None, name="the proximity test"):
        """st: the stages, each (base code, mask code, k, h), for 2^k main and 2^h mask rows.

        ride: if true, the fresh mask (R12.10) for a later stage's FOLD or BASE comes with the code
        switch (CS) that makes the stage; if false, at the stage's start.
        mask: the first stage's fresh mask, if sent already.
        """
        try:
            return self.iopp_(g, st, ride, mask)
        except Refused as why:
            raise Refused("%s [in %s]" % (why, name))

    def iopp_(self, g, st, ride, mask=None):
        ch = self.ch
        ch.close("BAT")
        theta = ch.element()
        k, h, first, c, d, tau = self.combine(g, theta)
        self.note("BAT: target", tau)
        G = self.groups[g]
        assert (G["code"], G["mask_code"], k, h) == tuple(st[0][:4]), ((G["code"], G["mask_code"], k, h), st[0])
        members = list(G["members"])

        def read(x):
            main, old = [ZERO] * (1 << k), [ZERO] * (1 << h)
            for i in members:
                o = self.oracles[i]
                if o.main_rows:
                    main[first[(i, "M")]:first[(i, "M")] + o.main_rows] = self.main_symbols(i, x)
                if o.mask_rows:
                    old[first[(i, "Z")]:first[(i, "Z")] + o.mask_rows] = self.mask_symbols(i, x)
            return main, old

        for s in range(len(st) - 1):
            C, Z, k, h = self.base[st[s][0]], self.mask[st[s][1]], st[s][2], st[s][3]
            C2, Z2, k2, h2 = self.base[st[s + 1][0]], self.mask[st[s + 1][1]], st[s + 1][2], st[s + 1][3]
            if mask is None:
                mask, = self.receive([(NONE, st[s][1], 0, 1, 1 << h, 0, 0)])
            gm, c0, d0, d1, tau = tc.fold_block(ch, k, h, c, d, tau, Z[1])
            self.note("FOLD: target", tau)
            l, r, lz, rz = C[1], C[2], Z[1], Z[2]
            assert C2[1] << k2 == l and (Z2[1] << h2) >= 2 * lz + r + 2 * rz + 1
            lastly = s + 2 == len(st)
            sent = [(st[s + 1][0], st[s + 1][1], 1 << k2, 3, 1 << h2, 1, 0)]
            if ride:
                sent.append((st[s + 1][0], st[s + 1][1], 1, 3, 1, 0, 0) if lastly else (NONE, st[s + 1][1], 0, 1, 1 << h2, 0, 0))
            ids = self.receive(sent)
            ch.close("SEND seed")
            bits_main, bits_mask = log2(l), h2 + log2(Z2[1])
            z, beta = tc.seed(ch, max(bits_main, bits_mask) + 1)
            alpha, = ch.recv("CS: the answer to the sample")
            ch.close("CS")
            draws = ch.positions()
            theta = ch.element()
            self.open(members + [mask], draws)
            assert ch.entry["rows"] == 2 + 3 * len(draws)
            # The new mask part: the folded mask rows' messages at 0 and A1; random tails at T (main
            # row), T0 and T1.
            A1, T, T0, T1 = lz, 2 * lz, 2 * lz + r, 2 * lz + r + rz
            nc, nd = list(c0), [ZERO] * (Z2[1] << h2)
            for j in range(lz):
                nd[j], nd[A1 + j] = d0[j], d1[j]
            u, w = tc.sample_row(z, beta, bits_main, bits_mask)
            f = theta
            nc = [add(a_, mul(f, b_)) for a_, b_ in zip(nc, u)]
            nd = [add(a_, mul(f, b_)) for a_, b_ in zip(nd, w)]
            ntau = add(tau, mul(f, alpha))
            eg, egh = eq_table(gm), eq_table(gm[:h])
            for n, xx in enumerate(draws):
                main, old = read(xx)
                ys = (inner(eg, main), inner(egh, old), inner(egh, self.mask_symbols(mask, xx)))
                if n == 0:
                    self.note("CS: y_main at the first draw", ys[0]), self.note("CS: y_0 at the first draw", ys[1]), self.note("CS: y_1 at the first draw", ys[2])
                pw, pz = powers(point(xx), r + l), powers(point(xx & ((1 << Z[0]) - 1)), rz + lz)
                f = mul(f, theta)
                for j in range(l):
                    nc[j] = add(nc[j], scale(pw[r + j], f))
                for j in range(r):
                    nd[T + j] = add(nd[T + j], scale(pw[j], f))
                ntau = add(ntau, mul(f, ys[0]))
                for which, (A, TT) in enumerate(((0, T0), (A1, T1))):
                    f = mul(f, theta)
                    for j in range(lz):
                        nd[A + j] = add(nd[A + j], scale(pz[rz + j], f))
                    for j in range(rz):
                        nd[TT + j] = add(nd[TT + j], scale(pz[j], f))
                    ntau = add(ntau, mul(f, ys[1 + which]))
            c, d, tau = nc, nd, ntau
            self.note("CS: target", tau), self.note("CS: c'[0]", c[0]), self.note("CS: d'[0]", d[0]), self.note("CS: d's last entry", d[-1])
            members, mask = [ids[0]], ids[1] if ride else None
            first = {(ids[0], "M"): 0, (ids[0], "Z"): 0}
            k, h = k2, h2
        C, Z = self.base[st[-1][0]], self.mask[st[-1][1]]
        if mask is None:
            mask, = self.receive([(st[-1][0], st[-1][1], 1, 3, 1, 0, 0)])
        tstar, = ch.recv("BASE: the masking word's target")
        ch.close("BASE lambda")
        lam = ch.element()
        M0, M1 = ch.recv("BASE: the masked main message", C[1]), ch.recv("BASE: the masked mask message", Z[1])
        R0, R1 = ch.recv("BASE: the masked main tail", C[2]), ch.recv("BASE: the masked mask tail", Z[2])
        left, right = add(inner(c, M0), inner(d, M1)), add(tstar, mul(lam, tau))
        self.note("BASE: left side of the linear check", left), self.note("BASE: right side of the linear check", right)
        if left != right:
            raise Refused("BASE: the linear check fails")
        ch.close("BASE")
        draws = ch.positions()
        self.open(members + [mask], draws)
        for xx in sorted(set(draws)):
            xz = xx & ((1 << Z[0]) - 1)
            if tc.symbol(C, xx, M0, R0) != add(self.main_symbols(mask, xx)[0], mul(lam, self.main_symbols(members[0], xx)[0])):
                raise Refused("BASE: a check at a position fails")
            if tc.symbol(Z, xz, M1, R1) != add(self.mask_symbols(mask, xx)[0], mul(lam, self.mask_symbols(members[0], xx)[0])):
                raise Refused("BASE: a check at a position fails")


SESSION = {}   # root -> {2^depth + place: value}: what the chains of one verdict share


# Call it before each verdict.
def new_session():
    SESSION.clear()


def opening(ch, t, draws):
    """Reads and checks an opening of t (R7.8 to R7.10).

    t may come from another chain of the session: its leaves are hashed with `t.start`.
    """
    xs, salt, columns, digests = sorted(set(draws)), ch.par["head"][2], {}, []
    number = 174 if t.stored else 164
    for x in xs:
        col = [tc.base_of(ch.f.take(8)) for _ in range(t.width)]
        s = ch.f.take(salt)
        columns[x] = col
        digests.append(R.leaf(number, t.number, t.depth, x, t.start, col, s))
    # R7.14, one value at a vertex (leaves included): one record for each root (its 32 bytes) read
    # in a session (`SESSION`), whichever tree, chain or file it came with. Only what an accepted
    # opening shows is entered.
    record, fresh = SESSION.setdefault(t.root, {}), {}

    def enter(d, i, value):
        if record.get((1 << d) + i, value) != value or fresh.get((1 << d) + i, value) != value:
            raise Refused("a vertex of a tree gets two values under one root")
        fresh[(1 << d) + i] = value

    if R.walk(t.depth, ch.conf, xs, digests, lambda d, i: ch.f.take(32), enter) != t.root:
        raise Refused("an opening does not lead to its root")
    record.update(fresh)
    return columns
