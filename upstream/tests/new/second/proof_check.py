#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by tests/new/second/run_honest.sh, which the runner starts; by hand: below a directory of your own.
"""Second verifier, in Python, of the toy proof files that a C++ tool writes.

The tool is blocks/frame/handover_proof.cpp; the first verifier is check_proof()
(blocks/frame/check.h). run_honest.sh runs this program to compare the two on honest and altered
files; --cheats compares them on false proofs, which run_honest.sh does not make, and no script of
this repository runs this program with --cheats. A proof file is a header (`head` in strings), the
first chain, the place (the request's leaf number in its batch's binding tree, with salt and path),
the shared chain and the second chain. A chain is one Fiat-Shamir transcript; the binding tree is a
Merkle tree over the seals of the batch's first chains. In the toy, the first chain starts with a
sequence of segments (`link` in identifiers and messages) that make claims on the committed matrix.
This program calls no code of blocks/. It uses modules of this directory: toy_check.py (field,
channel), zkpy.py (trees, oracle messages, groups, the proximity test: `iopp`), lin_check.py (LIN,
the test on the committed table), key_check.py (BATT, which batches the pending claims on that
table), links_check.py (a toy segment's sumcheck with a hiding polynomial: `PSC` in round names,
psc()) and, in write() alone, files_check.py.
"ACCEPTED" means only "passes under the given parameters": verify() lists what is not checked.
Nothing is hidden in the toy files: the tool's private randomness comes from small fixed numbers.

How independent it is: less than the other verifiers in this directory. One author wrote them, this
program and the C++ tool that writes the files. It is not independent of that tool, nor of the C++
programs on these points, where it can share a misreading with them (paths under blocks/):
- frame/check.h; frame/tests/check_test.cpp and seam_test.cpp (both sides of the toy segments);
  frame/handover_files.cpp (the same author's); frame/tests/toy_file.h; iop/batt_shown.h;
- iop/req.h: its top comment, the shape's structs and bytes, req_groups() (what is where in the
  carrying message), ReqFrame, ReqVerifier and, for a segment's further messages, req_plan() and
  ReqProver::later();
- iop/batt.h: BattClaim, batt_claim(), batt_row() (chi, the target, the incoming masks, the plain
  sum with three coefficients, a_j); iop/psc.h: its top comment, psc_verify(); iop/hp.h (weights,
  the hiding polynomial's form); iop/member.h: member_first_verify();
  iop/tests/member_world.h: member_row_form(); iop/tests/lin_world.h: lin_weights(), lin_w();
  iop/tensor.h: Tensor, add_to(); iop/lin.h: LinBindingShape, LinBinding and a few more lines;
  chain/public.h: Public, Sealed, Binder;
- frame/heads.h (labels), numbers/round_names.h, frame/check_key.h, frame/parameters.h,
  frame/files.h, frame/configure.h (`slots`); a program outside this repository that dumps a claim;
- the guards, rejection messages (the channel's ten among them) and names of notes (A.note()) of
  iop/ (send.h, cs.h, fold.h, bat.h, base_case.h, single_rows.h, foldeq.h, relation.h), of tree/
  and of chain/rounds.h;
- every point and file that the other verifiers' docstrings here name.
No one has examined this program's independence of the unnamed parts of these files or of any
other file of the C++ programs, under blocks/ or elsewhere: about all of these nothing is known
either way, and no independence from them is claimed.
claims_leave() is written from the C++ headers' own comments and code, not from a document. links()
applies links_check.py's formulas, which follow
docs/format/toys/2-links.md, not word for word as that page stands, to zkpy.py's oracle messages.
At its first run this program accepted the honest proof. Not in the program at that run: in links(), six
places that read a toy segment's further messages (one in a leaf class of its own,
one at a later commit point); no formula of a row is among them, and proofs with such messages were
accepted at their first run. Nor were the lists that --write writes, the modes, the reading of
the layout and, in verify(): the rule on `number`, the comparisons among the given files that its
docstring names under (1), the configuration's digest (at that run it was checked outside verify(),
after the word ACCEPTED) and rejections in place of assertions.

usage: proof_check.py RULES BATCH <n> TABLE [--number k] [--write]
       proof_check.py RULES BATCH 0 TABLE --altered <the other of the two TABLE files>
       proof_check.py RULES TOP --all
       proof_check.py RULES TOP --more-altered
       proof_check.py RULES TOP --cheats <directory of false proofs> <the list of what is expected>
RULES: the directory of format_rules.py. TOP: the tool's output directory. BATCH and TABLE:
TOP/on-the-first-table and TOP/key.file, or TOP/on-the-copy and TOP/entry.file.
--number: the number of entry.file's table (default 1). --all: for both batches, the three honest
files with --write, then --altered. For the other modes see altered(), more_altered(), cheats().
--write: writes into BATCH, for each of <n>.first, shared and <n>.second, .chain.tsv, .digests.tsv
and .items.tsv; shared.statement and <n>.second.statement as this program forms them;
<n>.openings.tsv (the trees that each round of positions opens, and where in the file);
<n>.layout.tsv; <n>.targets.second-verifier.tsv (intermediate values; verify() only records them).
In every mode an output file replaces a file or symbolic link of its name, and the program exits
rather than write into a directory that is a symbolic link, belongs to another account, or is
writable by its group or others (new_file() in toy_check.py).
"""
import os
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
import zkpy  # noqa: E402
import lin_check as lc  # noqa: E402
import key_check as kc  # noqa: E402
import links_check as lk  # noqa: E402
from toy_check import ZERO, ONE, Refused, add, sub, mul, scale, emb, inner, eq_table, fold, log2, point, powers, prod  # noqa: E402
from zkpy import V, Row, clog2, NONE  # noqa: E402

# R: the module format_rules. Kind 7: the verifier's message in the round `REPLY` is empty.
R = tc.R
tc.KIND.update({"REPLY": 7})


def parse_shape(value):
    """Parses the item «request's shape».

    `plain`, a segment's first word, is 1 if the reply is sent right before its first message.
    A segment's message is six words. `moment` is its commit point: the number, from 0, of the
    segment's prover message that carries it. `life` is the number of later segments that read
    its main rows. `at_start`: oracle messages in the very first prover message that belong to no
    segment's group; seven words each, in the order of zkpy.Oracle's fields. A stage of the last
    proximity test is four words; its first stage is not in the bytes.
    """
    v = tc.Bytes(value)
    q = {"code": v.word(), "mask_code": v.word(), "link": []}
    for _ in range(v.word()):
        plain, messages = v.word(), []
        for _ in range(v.word()):
            messages.append(tuple(v.word() for _ in range(6)))   # main rows, degree, mask rows, leaf class, moment, life
        q["link"].append((plain, messages))
    q["at_start"] = [tuple(v.word() for _ in range(7)) for _ in range(v.word())]
    q["answers"] = v.word()
    q["last"] = [(q["code"], q["mask_code"], 1, 0)] + [tuple(v.word() for _ in range(4)) for _ in range(v.word())]
    q["ride"] = v.word()
    assert v.at == len(value)
    return q


def parse_layout(value):
    """Parses the item «layout»: for each segment a list of arrays.

    An array is its name, the number of its oracle message within the segment, its rows and columns,
    and its blocks; a block is nine words, which write() names. The toy's sub-proofs do not go
    through arrays, so verify() checks only the number of lists.
    """
    v = tc.Bytes(value)
    out = [[(v.text(), v.word(), v.word(), v.word(), [tuple(v.word() for _ in range(9)) for _ in range(v.word())]) for _ in range(v.word())] for _ in range(v.word())]
    assert v.at == len(value)
    return out


# <n>.claims: for each segment, its claims' forms as pairs (psi, g): column point, row weights.
def read_claims(raw):
    v = tc.Bytes(raw)
    links, claims = v.word(), v.word()
    several = lambda: [tc.ext_of(v.take(24)) for _ in range(v.word())]  # noqa: E731
    out = [[(several(), several()) for _ in range(claims)] for _ in range(links)]   # (psi, g)
    assert v.at == len(raw)
    return out


def claims_leave(A, rows, own, index, K):
    """BATT with its final value sent (blocks/iop/batt_shown.h): carries claims out of a segment.

    rows: claims on the matrix as tuples (g, psi, masks, tau): row weights, a column point of K
    coordinates, {mask entry: coefficient}, the claimed value (`target` in messages). Their mask
    entries are in the mask row of `own`, a message of the segment. Returns claim (U), a Row on
    entries of that mask row only, and claim (E), such a tuple at a fresh point, whose one mask
    entry is in `index`, the index-mask message (the request's masks for its claims on the matrix).
    """
    ch = A.ch
    ch.close("BAT-sigma")
    assert ch.entry["rows"] == len(rows)
    x = ch.element()
    chi = [kc.power(x, j) for j in range(len(rows))]
    T, incoming = ZERO, {}
    for c, (_, _, mj, tau) in zip(chi, rows):
        T = add(T, mul(c, tau))
        for key, cf in mj.items():
            incoming[key] = add(incoming.get(key, ZERO), mul(c, cf))
    w = [(own, A.take(own)) for _ in range(2 + 2 * K)]
    c_in = (index, A.take(index))
    mu, = ch.recv("claims leave a link: the hiding polynomial's sum")
    ch.close("PSC epsilon")
    eps = ch.element()
    running, gamma = add(mu, mul(eps, T)), []
    for _ in range(K):     # q(0) + q(1) = 2 q0 + q1 + q2 must be `running`, so q1 is not sent
        q0, q2 = ch.recv("claims leave a link: a round's constant and quadratic coefficient", 2)
        ch.close("PSC round")
        gm = ch.element()
        q1 = sub(sub(running, scale(2, q0)), q2)
        running = add(q0, mul(gm, add(q1, mul(gm, q2))))
        gamma.append(gm)
    p1, = ch.recv("claims leave a link: the round in Y")     # p(Y) = `running` + p1 Y
    ch.close("PSC Y")
    beta = ch.element(True)
    running = add(running, mul(beta, p1))
    v, = ch.recv("claims leave a link: the end value")
    ch.close("PSC values")
    masks, tail = {w[2 * K]: ONE, w[2 * K + 1]: beta}, ONE
    for l in range(K - 1, -1, -1):
        masks[w[2 * l]], masks[w[2 * l + 1]] = mul(tail, gamma[l]), mul(tail, mul(gamma[l], gamma[l]))
        tail = mul(tail, gamma[l])
    for key, cf in incoming.items():
        masks[key] = add(masks.get(key, ZERO), mul(mul(eps, tail), cf))
    g = [ZERO] * len(rows[0][0])
    for c, (gj, psi, _, _) in zip(chi, rows):
        eqv = ONE
        for zt, xt in zip(psi, gamma):
            eqv = mul(eqv, add(mul(zt, xt), mul(sub(ONE, zt), sub(ONE, xt))))
        aj = mul(c, eqv)
        g = [add(a_, mul(aj, b_)) for a_, b_ in zip(g, gj)]
    A.note("claims leave a link: (U)'s target", sub(running, mul(eps, v))), A.note("claims leave a link: the end value", v)
    return kc.as_row(masks, sub(running, mul(eps, v))), (g, gamma, {c_in: beta}, v)


def links(A, q, claims, k_col):
    """Checks a request's segment chain, through its last proximity test, with the toy's sub-proofs.

    Only the toy's pattern of messages is accepted: in each segment b and c (c = x * b), then any
    messages of one row over F of which the sub-proof claims nothing. Returns the index-mask
    message's number, the pending claims (what the segments leave for LIN; tuples as in
    claims_leave()) and the reply: one scalar, which enters no claim.
    """
    ch = A.ch
    C, Z = A.base[q["code"]], A.mask[q["mask_code"]]
    assert C[0] == Z[0] and q["answers"] == 1 and len(q["at_start"]) == 1
    _, l, r = C
    _, lz, rz = Z
    Vn, n_links = log2(l) - 1, len(q["link"])
    for i, (plain, messages) in enumerate(q["link"]):   # the formulas below assume this pattern
        need(messages[:2] == [(2 if i == 0 else 1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 1 if i + 1 < n_links else 0)] and plain == (1 if i + 1 == n_links else 0)
             and all(m_[:3] == (1, 1, 0) and m_[5] == 0 for m_ in messages[2:]) and [m_[4] for m_ in messages] == sorted(m_[4] for m_ in messages) and all(m_[4] <= 1 for m_ in messages),
             "the links are not the toy's: b and c, then messages of one row over F with no later reader, by moments 0 and 1")
    carried, fresh_mask = (q["code"], NONE, 2, 3, 0, 1, 0), (NONE, q["mask_code"], 0, 1, 1, 0, 0)
    old, index, pending, reply = None, None, [], None
    for m in range(n_links + 1):
        last = m == n_links
        if not last and q["link"][m][0]:
            reply, = ch.recv("the reply")
            ch.close("REPLY")
        kept_before = bool(old and old["kept"] is not None)
        sent = ([carried] if m else []) + ([(q["code"], NONE, 1, 1, 0, 1, 0)] if kept_before else [])
        if not last:
            sent += [(q["code"], q["mask_code"], 2 if m == 0 else 1, 1, 1, 1, 0), (q["code"], NONE, 1, 1, 0, 1, 0)]
            sent += [(q["code"], NONE, 1, 1, 0, 1, m_[3]) for m_ in q["link"][m][1][2:] if m_[4] == 0]
        sent += [fresh_mask] + (q["at_start"] if m == 0 else [])
        ids = list(A.send(sent, own_round=m == 0))
        G = ids.pop(0) if m else None
        Pp = ids.pop(0) if kept_before else None
        B, P = (ids.pop(0), ids.pop(0)) if not last else (None, None)
        further = [ids.pop(0) for m_ in q["link"][m][1][2:] if m_[4] == 0] if not last else []
        U = ids.pop(0)
        if m == 0:
            index, = ids
        g = A.new_group(q["code"], q["mask_code"])
        plain, gens = A.groups[g]["plain"], A.groups[g]["gens"]
        if m:
            ch.close("LOC")
            draws = ch.positions()
            A.open(old["members"] + ([old["kept"]] if kept_before else []) + [old["U"]], draws)
            A1, T = lz, 2 * lz
            TK = T + r
            T0 = TK + (r if kept_before else 0)
            T1 = T0 + rz
            OP = T1 + rz
            on_G = {2 * j: old["c"][j] for j in range(l)}
            for j in range(lz):
                on_G[1 + 2 * j], on_G[1 + 2 * (A1 + j)] = old["d0"][j], old["d1"][j]
            forms = {(G, "M"): on_G}
            if kept_before:
                forms[(Pp, "M")] = {j: old["K"][j] for j in range(l - 3)}
                for t in range(3):
                    on_G[1 + 2 * (OP + t)] = old["K"][l - 3 + t]
            plain.append(Row(forms, old["tau"]))
            if kept_before:   # the kept message's earlier sample claim, as a claim by itself
                u_old, y_old = old["sample"]
                plain.append(Row({(Pp, "M"): {j: u_old[j] for j in range(l - 3)}, (G, "M"): {1 + 2 * (OP + t): u_old[l - 3 + t] for t in range(3)}}, y_old))
            eg = eq_table(old["gamma"])
            for n, x in enumerate(draws):
                main = [ZERO] * (1 << old["k"])
                for i in old["members"]:
                    o = A.oracles[i]
                    main[old["first"][(i, "M")]:old["first"][(i, "M")] + o.main_rows] = A.main_symbols(i, x)
                y0, y1, y2 = inner(eg, main), A.mask_symbols(old["B"], x)[0], A.mask_symbols(old["U"], x)[0]
                if n == 0:
                    A.note("CS: y_main at the first draw", y0), A.note("CS: y_0 at the first draw", y1), A.note("CS: y_1 at the first draw", y2)
                pw = [emb(p_) for p_ in powers(point(x), max(r + l, rz + lz))]
                f0 = {2 * j: pw[r + j] for j in range(l)}
                f0.update({1 + 2 * (T + j): pw[j] for j in range(r)})
                gens.append(Row({(G, "M"): f0}, y0))
                if kept_before:
                    fk = {1 + 2 * (OP + t): pw[r + l - 3 + t] for t in range(3)}
                    fk.update({1 + 2 * (TK + j): pw[j] for j in range(r)})
                    gens.append(Row({(Pp, "M"): {j: pw[r + j] for j in range(l - 3)}, (G, "M"): fk}, A.main_symbols(old["kept"], x)[0]))
                for (a_, tt), y in (((0, T0), y1), ((A1, T1), y2)):
                    fm = {1 + 2 * (a_ + j): pw[rz + j] for j in range(lz)}
                    fm.update({1 + 2 * (tt + j): pw[j] for j in range(rz)})
                    gens.append(Row({(G, "M"): fm}, y))
            A.join(g, G)
            if kept_before:
                A.join(g, Pp)
        if last:
            A.iopp(g, q["last"], q["ride"], U, "the first chain, the links' last test")
            return index, pending, reply
        # ---- the toy segment's sub-proof: c = x * b, entry by entry ----
        eps, gamma, by, claim, v1, v2 = lk.psc(ch, Vn)
        e = eq_table(gamma)
        w = [A.take(B) for _ in range(2 * Vn + 5)]
        dU = {}
        for t in range(Vn):
            dU[w[2 * t]], dU[w[2 * t + 1]] = gamma[t], mul(gamma[t], gamma[t])
        dU[w[2 * Vn]], dU[w[2 * Vn + 1]], dU[w[2 * Vn + 2]] = ONE, by, mul(by, by)
        xo, xrow, xrb = (B, 0, 1) if m == 0 else (Pp, 0, 0)
        brow, brb = (1, 1) if m == 0 else (0, 0)
        plain.append(Row({(P, "M"): {i: sub(ZERO, mul(eps, e[i])) for i in range(l // 2)}, (B, "Z"): dU}, sub(claim, mul(eps, mul(v1, v2)))))
        plain.append(Row({(xo, "M"): {xrow + (i << xrb): e[i] for i in range(l // 2)}, (B, "Z"): {w[2 * Vn + 3]: by}}, v1))
        if xo == B:   # two forms on one message's main rows
            plain.append(Row({(B, "M"): {brow + (i << brb): e[i] for i in range(l // 2)}, (B, "Z"): {w[2 * Vn + 4]: by}}, v2))
        else:
            plain.append(Row({(B, "M"): {brow + (i << brb): e[i] for i in range(l // 2)}, (B, "Z"): {w[2 * Vn + 4]: by}}, v2))
        # ---- commit point (`moment`) 1: further messages, sent amid the sub-proof ----
        late = [(q["code"], NONE, 1, 1, 0, 1, m_[3]) for m_ in q["link"][m][1][2:] if m_[4] == 1]
        if late:
            further += list(A.send(late))
        # ---- its claims on the committed matrix, masked by entries of its own mask row ----
        rows = []
        for psi, gj in claims[m]:
            tau, = ch.recv("a link's claim on the table: its target")
            rows.append((gj, psi, {(B, A.take(B)): ONE}, tau))
        row_U, row_E = claims_leave(A, rows, B, index, k_col)
        plain.append(row_U)
        pending.append(row_E)
        # ---- closing: BAT, FOLD reduce the segment's claims to one; kept rows are not folded ----
        kept = P if q["link"][m][1][1][5] else None
        A.join(g, B)
        if kept is None:
            A.join(g, P)
        for i in further:
            A.join(g, i)
        members = list(A.groups[g]["members"])
        ch.close("BAT")
        theta = ch.element()
        k, h, first = A.layout(g)
        assert h == 0
        c, d, K, tau, f = [ZERO] * (l << k), [ZERO] * lz, [ZERO] * l, ZERO, ONE
        batched = plain + [A.samples[i] for i in sorted(members + ([kept] if kept is not None else [])) if i in A.samples] + gens
        assert ch.entry["rows"] == len(batched), (ch.entry, len(batched))
        for row in batched:
            for (i, part), form in row.forms.items():
                o = A.oracles[i]
                for at, cf in form.items():
                    if cf == ZERO:
                        continue
                    if part == "Z":
                        assert i == B
                        d[at] = add(d[at], mul(f, cf))
                    elif i == kept:
                        K[at] = add(K[at], mul(f, cf))
                    else:
                        to = first[(i, "M")] + (at & ((1 << o.rb) - 1)) + ((at >> o.rb) << k)
                        c[to] = add(c[to], mul(f, cf))
            tau, f = add(tau, mul(f, row.target)), mul(f, theta)
        A.note("REQ: the group row's target", tau)
        sigma, = ch.recv("FOLD: the hiding polynomial's sum")
        ch.close("FOLD epsilon")
        feps = ch.element()
        gm, claim = lk.rounds_of_fold(ch, k, add(sigma, mul(feps, tau)))
        A.note("REQ: FOLD's target", claim)
        fk, fd = mul(feps, prod([sub(ONE, x_) for x_ in gm])), mul(feps, prod(gm))
        sample = None
        if kept is not None:
            form = A.samples[kept].forms[(kept, "M")]
            sample = ([form.get(j, ZERO) for j in range(l)], A.samples[kept].target)
        old = dict(c=[mul(feps, v_) for v_ in fold(c, gm)], K=[mul(fk, v_) for v_ in K] if kept is not None else None, d0=[mul(fd, v_) for v_ in d], d1=tc.mask_weights(gm, k, 0, lz), tau=claim, gamma=gm,
                   kept=kept, sample=sample, members=members, first=first, k=k, B=B, U=U)


def need(what, words):
    """Rejects the given files (parameters, claims, table) unless `what` is true.

    The message's prefix tells this rejection from a rejection of the proof.
    """
    if not what:
        raise Refused("the given files: " + words)


# The first chain's parameter items, in order. The shared chain has the first three, the second
# chain the first.
FIRST_ITEMS = ["configuration", "codes", "the test on the table", "request's shape", "layout", "links' identifiers", "pending rows", "binding tree"]


def verify(read, number):
    """Checks one proof file. Returns the zkpy.V of the request's two chains and the shared chain's.

    Raises Refused if the proof or a given file fails; some malformed parameters fail an assertion
    instead. read(tail) returns the bytes of `proof`, `first.parameters`, `shared.parameters`,
    `second.parameters`, `request`, `reply`, `claims`, `table` (the key file, or the leaf file:
    `entry` in file names and messages), `public-1`, `public-2`.

    Four inputs are taken as given; the first is the largest assumption.
    (1) The three .parameters files, the bytes of the C++ class `Binder` (blocks/chain/public.h).
        They contain all that the C++ verifier derives from the two public files by configure()
        (blocks/frame/configure.h): the three plans (each round's name, bytes, bits of proof of
        work, number of positions, draws, claims), the codes, LIN's shape (the item «the test on the
        table»), the request's shape, layout, identifiers, the number of pending claims, the binding
        tree. This program derives none of it. It does not check whether a round's error meets the
        128-bit target, whether the conditions for hiding are met, or whether the parameters are
        acceptable: given parameters with one position per round and a proof made under them, it
        accepts.
        The only tie to the public files is the item «configuration», which must be their digest in
        all three chains. Among the given files it checks the items' names and order; that the
        shared chain's codes and LIN's shape are the first chain's, byte for byte; that each segment
        has one identifier and one list of arrays; and the form of `at_start`.
        Three limits of this program, not rules of the format: it rejects parameters with further
        items after «binding tree» (a caller's, which the format allows), a plan with listed
        positions (which the format has), and a plan with a stored table's RECORD.
    (2) The claims' forms (<n>.claims). In the toy they stand in for a model and are a public
        function of the request's bytes, which this program does not compute.
    (3) The table's root: 32 bytes of a key file (the anchor's root; messages say "the first table")
        or of a leaf file (a leaf table's root; messages say "copy"). Beyond its tag and
        version that file is not checked here: key_check.py and entry_check.py do that, and
        files_check.py calls both.
    (4) `number`, under which the caller keeps the table, as for check_proof(). Unlike
        check_proof(), this program requires 0 with a key file and a number above 0 with a leaf
        file, as the tool numbers its tables.
        The programs cp_commit, cp_prove and cp_verify number leaf tables from 0 (ivk/leaf/0.bin),
        so this program would reject a proof of theirs on leaf table 0 (a reading, not a run: this
        program reads toy segments only).
    """
    zkpy.new_session()
    raws = [read(c + ".parameters") for c in ("first", "shared", "second")]
    P1, P2, P3 = (lc.parse(r_) for r_ in raws)
    d1, d2 = dict(P1["items"]), dict(P2["items"])
    need([n_ for n_, _ in P1["items"]] == FIRST_ITEMS and [n_ for n_, _ in P2["items"]] == FIRST_ITEMS[:3] and [n_ for n_, _ in P3["items"]] == FIRST_ITEMS[:1], "a chain's parameters have other items than this program reads (the format lets a caller's items stand behind «binding tree»), or in another order")
    conf = R.configuration_digest([read("public-1"), read("public-2")])
    need(all(P["items"][0][1] == conf for P in (P1, P2, P3)), "a chain's item «configuration» is not the digest of the two public files")
    need(d2["codes"] == d1["codes"] and d2["the test on the table"] == d1["the test on the table"], "the shared chain's codes or its shape of the test on the table are not the first chain's, byte for byte")
    need(all(P["head"][4] == 0 and all(e["name"] != "RECORD" for e in P["rounds"]) for P in (P1, P2, P3)), "a plan with listed positions, or with a stored table's RECORD: this program does not read these")
    table = read("table")
    record = table[40:40 + 24 * (P1["S"]["a"] << P1["S"]["v"])]   # R4.6: the record's answers stand behind the root, in a key file as in a leaf file
    need(len(table) >= 40 + 24 * (P1["S"]["a"] << P1["S"]["v"]) and table[:4] in (b"CPIK", b"CPEN") and table[4:8] == b"H821", "the table's file is no key and no entry of this version")
    if table[:4] == b"CPIK" and number != 0:
        raise Refused("the first table, whose root stands in a key, is held under the number 0")
    if table[:4] == b"CPEN" and number == 0:
        raise Refused("a copy of the table, whose root stands in an entry, is held under a number above 0")
    f = tc.Bytes(read("proof"))
    if f.take(4) != b"CPPF":
        raise Refused("a file of another kind")
    if f.take(4) != b"H821":
        raise Refused("another version")
    made_on, = struct.unpack("<I", f.take(4))
    if made_on != number:
        raise Refused("the proof was made on another stored copy of the table")
    table_item = R.u64(32) + table[8:40] + R.u64(len(record) // 24) + record
    request = read("request")
    st1 = R.proof_statement(table[8:40], record, made_on, request, request)  # R14.3; in the toy a request is its own file
    L = lc.Lin(P1, table_item)
    marks = [("head", 0), ("the first chain", f.at)]
    ch = tc.Run(P1, raws[0], None, f, label=0x46524d01, statement=st1)
    A = V(ch, L.base, L.mask)
    q = parse_shape(d1["request's shape"])
    A.arrays = parse_layout(d1["layout"])   # only its number of lists is checked
    identifiers = struct.unpack("<%dQ" % (len(d1["links' identifiers"]) // 8), d1["links' identifiers"])
    need(len(A.arrays) == len(q["link"]) and identifiers[0] == len(identifiers) - 1 == len(q["link"]), "not one list of arrays and one identifier for each link")
    need(len(q["at_start"]) == 1 and q["at_start"][0][0] == NONE and q["at_start"][0][1] == L.S["own_mask"] and q["at_start"][0][2] == 0 and q["at_start"][0][4] >= 1
         and q["at_start"][0][6] > max(m_[3] for _, messages in q["link"] for m_ in messages),
         "what rides at the start is not one message of mask rows alone, under the own group's mask code, in a class of leaves above the links'")
    claims = read_claims(read("claims"))
    need(len(claims) == len(q["link"]) and all(len(psi) == L.k_col and len(g_) == L.m for link in claims for psi, g_ in link), "the claims' forms are not one list a link, each with a point of k_col coordinates and a weight for each row of the matrix")
    index, pending, reply = links(A, q, claims, L.k_col)
    if reply != tuple(int(x_) for x_ in read("reply").split()):
        raise Refused("the proof is about another reply")
    need(len(pending) == struct.unpack("<Q", d1["pending rows"])[0], "the links leave another number of rows on the table than the first query says")
    g, psi, masks, tau = kc.batt(A, pending, index, L.k_col) if len(pending) > 1 else pending[0]
    A.note("member: the claim's target", tau)
    F = lc.lin_first(A, L, g, psi, list(masks.items()), tau)
    seal1 = ch.seal()
    ch.finish(last=False)
    depth, salt_bytes = struct.unpack("<2Q", d1["binding tree"][:16])
    marks.append(("the place", f.at))
    A.place = int.from_bytes(f.b[f.at:f.at + 8], "little")
    broot = lc.binding_root(conf, depth, salt_bytes, f, seal1)
    marks.append(("the shared chain", f.at))
    Bv, Om, star, seal2, ch2 = lc.lin_shared(L, P2, raws[1], f, broot, label=0x46524d03)
    ch2.finish(last=False)
    marks.append(("the second chain", f.at))
    A.marks = marks + [("the end", len(f.b))]
    ch3 = lc.lin_second(A, Bv, L, F, Om, star, P3, raws[2], f, seal1, seal2, label=0x46524d04)
    ch3.finish()
    A.with_work = any(e["work"] for P in (P1, P2, P3) for e in P["rounds"])
    A.shared_notes = Bv.notes
    A.runs, A.seals, A.binding_root, A.statement = [("first", ch), ("shared", ch2), ("second", ch3)], (seal1, seal2), broot, st1
    ch.raw, ch2.raw, ch3.raw = raws
    A.statements = {"first": st1, "shared": R.item("table", table_item) + R.item("the binding tree's root", broot),
                    "second": R.item("the seal of an earlier chain", seal1) + R.item("the seal of an earlier chain", seal2)}
    return A, Bv


# A tuple's fields are the columns of <n>.openings.tsv in write(): [8] root, [11] first byte in the
# file, [12] bytes.
OPENINGS = []   # every opening of a verdict, in the file's order


def listen():
    """Wraps zkpy.opening() so that OPENINGS records each accepted opening. No verdict changes."""
    plain = zkpy.opening

    def opening(ch, t, draws):
        at = ch.f.at
        got = plain(ch, t, draws)
        OPENINGS.append((ch, ch.chain.round, ch.rounds[-1]["name"], t.number, "stored" if t.stored else "of a run", t.depth, t.width, t.start, t.root, len(draws), len(set(draws)), at, ch.f.at - at))
        return got

    zkpy.opening = opening


def items_list(raw_parameters, statement):
    """Names and lengths of a first query's items, in order."""
    out, b = [], tc.Bytes(raw_parameters)
    name = b.text()
    out.append(("parameter", name, b.word()))
    b.take(out[-1][2])
    at = b.at
    head = [b.word() for _ in range(8)]
    for _ in range(head[7]):
        b.text()
        b.take(88)
    out.append(("parameter", "(the plan: eight words, then %d rounds, each its name and eleven words; no name and no length of its own)" % head[7], b.at - at))
    while b.at < len(raw_parameters):
        name = b.text()
        out.append(("parameter", name, b.word()))
        b.take(out[-1][2])
    for name, value in lc.items_of(statement):
        out.append(("statement", name, len(value)))
    return out


def write(folder, n, A, statements):
    import files_check as fc
    seal1, seal2 = A.seals
    more = {"first": [("the chain's seal", seal1.hex())], "shared": [("the binding tree's root", A.binding_root.hex()), ("the chain's seal", seal2.hex())], "second": []}
    for name, ch in A.runs:
        stem = "shared" if name == "shared" else "%s.%s" % (n, name)
        before = None
        if name == "shared":   # shared by the batch: a later proof must reproduce these two files
            try:
                before = [open("%s/shared.%s.tsv" % (folder, x), "rb").read() for x in ("chain", "digests")]
            except FileNotFoundError:
                pass
        fc.write_chain(folder, stem, ch, more[name])
        if before is not None and before != [open("%s/shared.%s.tsv" % (folder, x), "rb").read() for x in ("chain", "digests")]:
            raise SystemExit("the shared chain of %s.proof is not the one that another file of the batch has given" % n)
        with tc.new_file("%s/%s.items.tsv" % (folder, stem)) as o:
            o.write("part of the first query\tthe item's name\tbytes of its value\n")
            for row in items_list(ch.raw, statements[name]):
                o.write("%s\t%s\t%d\n" % row)
        if name != "first":
            with tc.new_file("%s/%s.statement" % (folder, stem), "wb") as o:
                o.write(statements[name])
    names = {id(ch): name for name, ch in A.runs}
    with tc.new_file("%s/%s.openings.tsv" % (folder, n)) as o:
        o.write("chain\tround\tthe round's name\ttree's number\tkind of tree\tdepth\twidth: elements of F in a column\tits leaves start from\troot\tpositions drawn\tdistinct columns opened\tfirst byte in the file\tbytes\n")
        for ch, rnd, rname, number, kind, depth, width, start, root, drawn, distinct, at, size in OPENINGS:
            o.write("\t".join([names[id(ch)], str(rnd), rname, str(number), kind, str(depth), str(width), start.hex(), root.hex(), str(drawn), str(distinct), str(at), str(size)]) + "\n")
    with tc.new_file("%s/%s.layout.tsv" % (folder, n)) as o:
        o.write("link\tarray\toracle message of the link\trows\tcolumns\tblock\tfirst row in the array\tfirst column in the array\tfirst row in the table\tfirst place in the table\tbits of rows in the array\t"
                "bits of columns in the array\tbits of rows in the table\tbits of places in the table\tthe array's row index stands in the low bits\n")
        for l, arrays in enumerate(A.arrays):
            for name, message, rows, columns, blocks in arrays:
                for b, block in enumerate(blocks):
                    o.write("\t".join(map(str, (l + 1, name, message, rows, columns, b + 1) + block)) + "\n")
    with tc.new_file("%s/%s.targets.second-verifier.tsv" % (folder, n)) as o:
        o.write("what\tthe element's coordinates\n")
        for what, value in A.notes + [("(shared chain) " + w_, v_) for w_, v_ in A.shared_notes]:
            o.write("%s\t%s\n" % (what, tc.dec(value)))


def reader_of(folder, n, table, proof=None, other=None):
    """Returns read(tail) for verify(). other: {tail: bytes to return instead of the file's}."""
    whole, other = {"shared.parameters", "second.parameters"}, other or {}

    def read(tail):
        if tail in other:
            return other[tail]
        if tail in ("public-1", "public-2"):   # beside the table's file
            path = "%s/public-file-%s.txt" % (table.rsplit("/", 1)[0], tail[-1])
        else:
            path = table if tail == "table" else proof if tail == "proof" and proof else "%s/%s" % (folder, tail) if tail in whole else "%s/%s.%s" % (folder, n, tail)
        return open(path, "rb").read()
    return read


def number_of(table, given=None):
    """The table's number for verify(): 0 for a key file; else `given`, or 1 as in the tool."""
    return 0 if open(table, "rb").read(4) == b"CPIK" else 1 if given is None else given


def honest(folder, n, table, writes, number=None):
    """Checks one honest proof file and prints the result.

    Returns 0 if it is accepted and agrees with the tool's <n>.first.statement, <n>.parts.tsv,
    <n>.place.txt and <n>.targets.tsv, else 1. verify() reads none of these four files.
    """
    del OPENINGS[:]
    read = reader_of(folder, n, table)
    try:
        A, _ = verify(read, number_of(table, number))
    except Refused as why:
        print("%s/%s.proof: REFUSED by the second verifier: %s" % (folder, n, why))
        return 1
    same = A.statement == read("first.statement")
    theirs = [x.rstrip("\n").split("\t") for x in open("%s/%s.parts.tsv" % (folder, n))][1:]
    borders = [int(x[1]) for x in theirs] + [int(theirs[-1][1]) + int(theirs[-1][2])] == [a_ for _, a_ in A.marks]
    said = open("%s/%s.place.txt" % (folder, n)).read().split()
    place = A.place == int(said[1]) and number_of(table, number) == int(said[-1])
    # <n>.targets.tsv: check_proof()'s intermediate values in the first chain, in run order. The
    # i-th value under a name is compared with this verifier's i-th under it; if there is none, it
    # is only counted.
    ours, seen, equal, differ, unknown = {}, {}, 0, 0, {}
    for what, value in A.notes:
        ours.setdefault(what, []).append(tc.dec(value))
    values = [x.rstrip("\n").split("\t") for x in open("%s/%s.targets.tsv" % (folder, n))][1:]
    for what, value in values:
        i = seen.get(what, 0)
        seen[what] = i + 1
        if what not in ours or i >= len(ours[what]):
            unknown[what] = unknown.get(what, 0) + 1
        elif ours[what][i] == value:
            equal += 1
        else:
            differ += 1
    print("%s/%s.proof: ACCEPTED by the second verifier, under the given parameters%s (%d bytes; place %d); the first chain's statement, formed by the reader, is the C++ Binder's: %s; place and number are the tool's: %s; "
          "the five parts begin where the prover's list has them: %s; of check_proof()'s %d values on the way %d are equal to the second verifier's, %d differ, %d it does not note%s"
          % (folder, n, ", with work counted" if A.with_work else "", len(read("proof")), A.place, same, place, borders, len(values), equal, differ, sum(unknown.values()),
             " (%s)" % "; ".join(sorted(unknown)) if unknown else ""))
    if writes:
        write(folder, n, A, A.statements)
    return 0 if same and borders and place and not differ and equal else 1


# check_proof()'s message -> the start of this verifier's message for the same kind of check
KINDS = {"file-of-another-kind": "a file of another kind", "version-unknown": "another version", "the proof was made on another stored copy of the table": "the proof was made on another stored copy of the table",
         "a nonce does not give the proof of work that its round asks for": "a nonce has not the work of its round", "an opening does not lead to its root": "an opening does not lead to its root", "the opened columns of the table do not lead to its root": "an opening does not lead to its root",
         "an opening is cut short or holds a symbol that is not in canonical form": "the file ends before its last item", "bytes are left over at the end of the proof": "bytes are left over",
         "the proof is about another reply": "the proof is about another reply", "the request's place in its batch": "the path: no such place", "a vertex of a tree gets two values under one root": "a vertex of a tree gets two values under one root",
         "a value that the prover sends is missing or not in canonical form": "eight bytes that are no element's spelling", "BASE: the masked message in the proof does not satisfy the linear claim": "BASE: the linear check fails"}


def altered(folder, table, other_table):
    """Runs verify() on every line of altered.tsv and writes altered.both-verifiers.tsv.

    A line is an altered copy of 0.proof, or a call on the honest file with another input, with
    check_proof()'s message. Returns (lines, lines that one of the two verifiers accepted, lines
    whose two messages are neither of the same kind of check, by KINDS, nor a pair in APART).
    """
    n, lines, bad, other_kind = "0", [x.rstrip("\n").split("\t") for x in open(folder + "/altered.tsv")][1:], 0, 0
    number = number_of(table)
    calls = {"the honest file, asked about request 1's bytes": (dict(request=open(folder + "/1.request", "rb").read(), claims=open(folder + "/1.claims", "rb").read()), number),
             "the honest file, asked about another reply": (dict(reply=open(folder + "/1.reply", "rb").read()), number),
             "the honest file, asked about another stored copy's number": ({}, number + 1),
             "the honest file, checked on the other table": (dict(table=open(other_table, "rb").read()), number)}
    with tc.new_file(folder + "/altered.both-verifiers.tsv") as o:
        o.write("copy\twhat was changed\tat byte\tcheck_proof()'s words\tthe second verifier's words\tthe same kind of check\n")
        for copy, what, at, theirs in lines:
            other, asked = calls[what] if copy == "-" else ({}, number ^ 1 if what == "the head's third word, and the call asks for that number" else number)
            try:
                verify(reader_of(folder, n, table, None if copy == "-" else "%s/%s.altered-%s.proof" % (folder, n, copy), other), asked)
                mine = "ACCEPTED"
            except Refused as why:
                mine = str(why)
            kind = KINDS.get(theirs) == mine.partition(" [in ")[0] or (theirs, mine.partition(" [in ")[0]) in APART
            bad, other_kind = bad + (mine == "ACCEPTED" or theirs == "ACCEPTED"), other_kind + (not kind)
            o.write("\t".join([copy, what, at, theirs, mine, "yes" if KINDS.get(theirs) == mine.partition(" [in ")[0] else "NO, AND KNOWN: " + APART[(theirs, mine.partition(" [in ")[0])] if kind else "NO"]) + "\n")
    return len(lines), bad, other_kind


# (check_proof()'s message, this verifier's) -> why the two reject at different checks, where known
OWN_RULE = "the second verifier holds the number and the kind of the file with the root together (0: a key; above 0: an entry) and refuses before it reads the proof; check_proof() takes table and number from its caller, and the file falls at the next check"
APART = {(theirs, mine): OWN_RULE for mine in ("the first table, whose root stands in a key, is held under the number 0", "a copy of the table, whose root stands in an entry, is held under a number above 0")
         for theirs in ("a nonce does not give the proof of work that its round asks for", "the proof was made on another stored copy of the table",
                        "an opening does not lead to its root", "the opened columns of the table do not lead to its root")}   # the last two: a work test of a few bits passed by luck, then the opening after it


def more_altered(top):
    """Writes two altered copies of 0.proof per batch, at offsets that only parsing the file gives.

    Prints for each: batch, the copy's number, what was changed, at which byte. It checks none of
    them: run_honest.sh gives them to the tool (mode `judge`) and adds them to altered.tsv.
    """
    for where, table in (("on-the-first-table", "key.file"), ("on-the-copy", "entry.file")):
        folder = "%s/%s" % (top, where)
        del OPENINGS[:]
        verify(reader_of(folder, "0", "%s/%s" % (top, table)), number_of("%s/%s" % (top, table)))
        good = open(folder + "/0.proof", "rb").read()
        last = max(int(x.split("\t")[0]) for x in list(open(folder + "/altered.tsv"))[1:] if x[0] != "-")
        twice = [o for o in OPENINGS if sum(1 for p in OPENINGS if p[8] == o[8]) > 1]          # by the root
        second = [o for i, o in enumerate(twice) if any(p[8] == o[8] for p in twice[:i])][-1]   # the last opening of a tree that was opened before: Omega's, in the second chain
        for what, at in (("the last digest that is given in the second opening of a tree that is opened twice (a vertex near the root, which the first opening has shown)", second[11] + second[12] - 1),
                         ("the first symbol of the first column in that second opening", second[11])):
            last += 1
            bent = bytearray(good)
            bent[at] ^= 1
            with tc.new_file("%s/0.altered-%02d.proof" % (folder, last), "wb") as o:
                o.write(bytes(bent))
            print("%s\t%02d\t%s\t%d" % (where, last, what, at))
    return 0


WHOLE_BATCH_FALSE = ("SEAM_OTHER_MATRIX",)   # the shared chain itself runs on another table: no request of that batch is honest


def cheats(top, false_top, list_path):
    """Checks each false proof (a malicious prover's) against its line of `list_path`.

    A line's tab-separated columns 1, 4 and 5 are read: the proof's name, whether check_proof()
    must reject in the first chain, and the message expected of this verifier.
    This verifier's message must be the expected one, word for word; check_proof()'s message must
    map to the same check (KINDS); and this verifier, the tool's .txt file and the list must agree
    on whether the first chain alone rejects. The honest proof of request 1 of the same batch must
    be accepted. A name in WHOLE_BATCH_FALSE has no such proof, and the file must not exist.
    """
    folder, bad, n = top + "/on-the-copy", 0, 0
    for l in open(list_path, encoding="utf-8").read().splitlines():
        if l.startswith("#") or l.startswith("name\t") or not l.strip():
            continue
        name, _, _, theirs_expected, expected = l.split("\t")[:5]
        stem = "%s/on-the-copy/0.cheat-%s" % (false_top, name)
        words, first_alone = open(stem + ".txt").read().strip().split("\n")
        try:
            verify(reader_of(folder, "0", top + "/entry.file", stem + ".proof"), 1)
            mine = "ACCEPTED"
        except Refused as why:
            mine = str(why)
        beside = "%s/on-the-copy/1.honest-neighbour-of-%s.proof" % (false_top, name)
        if name in WHOLE_BATCH_FALSE:
            neighbour = "accepted" if not os.path.exists(beside) else "A FILE LIES THERE, AND THE WHOLE BATCH IS FALSE"
        else:
            try:
                verify(reader_of(folder, "1", top + "/entry.file", beside), 1)
                neighbour = "accepted"
            except Refused as why:
                neighbour = "REFUSED: %s" % why
        in_first = "the first chain" in mine
        chains = first_alone == "the first chain alone: " + ("refuses" if in_first else "passes") and theirs_expected.startswith("refused in the first chain" if in_first else "the first chain passes")
        ok = mine == expected and KINDS.get(words) == mine.partition(" [in ")[0] and chains and neighbour == "accepted"
        n, bad = n + 1, bad + (not ok)
        print("proof\t%s\texpected beforehand: %s\tthe second verifier: %s\tcheck_proof(): %s; %s\tits honest neighbour by the second verifier: %s\t%s" % (name, expected, mine, words, first_alone, neighbour if name not in WHOLE_BATCH_FALSE or neighbour != "accepted" else "none: the whole batch is false",
                                                                                                                                                     "as expected" if ok else "NOT AS EXPECTED"))
    print("%d false proofs, %s" % (n, "ALL REFUSED AS EXPECTED" if not bad else "%d NOT AS EXPECTED" % bad))
    return 1 if bad else 0


def main():
    listen()
    if "--all" in sys.argv:     # both batches: honest files with --write, then altered.tsv
        top, bad, copies = sys.argv[2], 0, 0
        for where, table, other in (("on-the-first-table", "key.file", "entry.file"), ("on-the-copy", "entry.file", "key.file")):
            for n in "012":
                bad += honest("%s/%s" % (top, where), n, "%s/%s" % (top, table), True)
            lines, accepted, other_kind = altered("%s/%s" % (top, where), "%s/%s" % (top, table), "%s/%s" % (top, other))
            print("%s/%s: %d altered copies and calls; accepted by one of the two verifiers: %d; refused by the two at different kinds of check: %d" % (top, where, lines, accepted, other_kind))
            bad, copies = bad + accepted + other_kind, copies + lines
        print("proofs: 6 honest files, %d altered copies and calls: %s" % (copies, "ALL AGREES" if not bad else "%d THINGS DIFFER" % bad))
        return 1 if bad else 0
    if "--more-altered" in sys.argv:
        return more_altered(sys.argv[2])
    if "--cheats" in sys.argv:
        i = sys.argv.index("--cheats")
        return cheats(sys.argv[2], sys.argv[i + 1], sys.argv[i + 2])
    folder, n, table = sys.argv[2], sys.argv[3], sys.argv[4]
    if "--altered" in sys.argv:
        lines, accepted, other_kind = altered(folder, table, sys.argv[sys.argv.index("--altered") + 1])
        print("%s: %d altered copies and calls; accepted by one of the two verifiers: %d; refused by the two at different kinds of check: %d" % (folder, lines, accepted, other_kind))
        return 1 if accepted or other_kind else 0
    return honest(folder, n, table, "--write" in sys.argv, int(sys.argv[sys.argv.index("--number") + 1]) if "--number" in sys.argv else None)


if __name__ == "__main__":
    sys.exit(main())
