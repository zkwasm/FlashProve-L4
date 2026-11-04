#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Level 1 of the audit for the proof files of one batch; tools/audit/level1.py describes the method,
# its sources and its limits. Each script_*() repeats one generator's draws of private random values
# (`coins` in identifiers) in the order of the prover's sources (a fault shared with them goes
# unseen); run() compares what follows from the draws with the files' bytes.
# In names, `link` is a segment (one sub-proof's part of a request's first chain), `copy` a leaf
# table, `rider` the request's index-mask message. Omega and W are oracle messages of LIN, the test
# on the weight table (blocks/iop/lin.h). Each message x of a segment is six words: main rows,
# degree, mask rows, leaf class, commit point (`moment`), life f (R12.12). R12.12 and R12.16 are
# rules of docs/format/09-key-and-leaf-files.md, "9.9 A chain of segments".
import struct

import coins as CO
import cut
from level1 import Gen, fold_mask_coins, fresh_mask, send, test, shared_part, w_part, hold_ledger, masks_in_order, flat, es


# The batch's own generator, in the order of `BatchOpening` (prover/prover/batch_opening.h).
def script_batch(G, C, file, pr):
    run = C.runs[(file, 'shared')]; codes, salt, samples = run['codes'], run['eight'][2], run['eight'][5]
    n = 1 << pr['depth']
    out = {'binding-salts': G.bytes(40 * n, 'the salts of the binding tree'), 'free-leaves': [G.bytes(32, 'the free leaf %d of the binding tree' % i) for i in range(n)]}
    free = list(range(n))
    for i in range(n, 1, -1):  # batch_opening.h: a random arrangement
        most = ((1 << 64) - 1) - ((1 << 64) - 1) % i
        while True:
            x = struct.unpack('<Q', G.bytes(8, 'the arrangement of the places, step %d' % i).bytes())[0]
            if x < most: break
        j = x % i
        free[i - 1], free[j] = free[j], free[i - 1]
    out['free'] = free
    msgs = iter([m for _, m in C.msgs[(file, 'shared')]])
    out['omega'] = shared_part(G, msgs, pr['T'], codes, salt, samples, (file, 'shared')); assert next(msgs, None) is None
    return out


def script_seat(G, C, file, pr, entries, omega):
    """One request's generator. entries: for each segment, how many mask entries its sub-proof uses
    (scalars_of_links()). omega: Omega's part, from script_batch().
    Groups of oracle messages (R12.16): one per segment, then the last proximity test's."""
    run = C.runs[(file, 'first')]; codes, salt, samples = run['codes'], run['eight'][2], run['eight'][5]
    Q, T = pr['Q'], pr['T']; L = len(Q['links']); ell = codes[1][Q['mask']][1]
    out = {'chain-bytes': {'first': G.bytes(32, 'the random bytes of the first chain')}}
    msgs = iter([m for _, m in C.msgs[(file, 'first')]])
    base, mask = codes
    born, group, before = {}, None, None  # before: the previous group's (members, fold mask)
    def close(m_, members, fm, target):
        G.events.append(dict(what='fold', chain=(file, 'first'), members=members, fresh=fm, k=fold_k(Q, m_), h=0, code=base[Q['code']], mask_code=mask[Q['mask']], target=target, codes=codes))
    def kept_in(j, i, m_): return j + Q['links'][j - 1][1][i][5] > m_  # R12.16: a kept message counts in group j + f alone
    def members_of(m_, parts):
        out = []
        for p in parts:
            if p.kind == 'carrying': out.append((p, False))
            elif p.kind in ('kept', 'link'): out.append((p, kept_in(p.born[0], p.born[1], m_)))
        return out
    def name_born(m_, m):
        kept_here = [(j, i) for j in range(1, m_) for i, x in enumerate(Q['links'][j - 1][1]) if x[5] > 0 and j + x[5] >= m_]
        for p, ji in zip([p for p in m if p.kind == 'kept'], kept_here): p.born = ji
    for m_ in range(1, L + 1):
        m = next(msgs)
        name_born(m_, m)
        if before is not None: close(m_ - 1, before[0], before[1], next(p for p in m if p.kind == 'carrying'))
        if m_ == 1:
            for p in m:
                if p.kind == 'rider': fresh_mask(G, p, codes); rider = p
        own = [p for p in m if p.kind == 'link']
        row = next(p for p in own if p.mask_rows)
        row.entries = entries[m_ - 1]
        row.rest = [G.e('entry %d of the mask row of link %d' % (j, m_)) for j in range(row.entries, ell)]
        for p in own:
            if p.mask_rows: p.dropped = G.fill(3 * p.mask_rows * ell, 'a mask table of %s that is drawn and not used' % p.name)
        fm = next(p for p in m if p.kind == 'fold-mask')
        k = fold_k(Q, m_)
        fold_mask_coins(G, fm, k)
        send(G, m, codes, salt)
        parts = list(m)
        for t in range(1, 1 + max(x[4] for x in Q['links'][m_ - 1][1])):
            m2 = next(msgs); send(G, m2, codes, salt); parts += m2
        number = 0
        order = sorted(range(len(Q['links'][m_ - 1][1])), key=lambda i: (Q['links'][m_ - 1][1][i][4], i))  # the segment's messages by moment: the order of sending
        for p, i in zip([p for p in parts if p.kind == 'link'], order): p.born = (m_, i)
        before = (members_of(m_, parts), fm)
    m = next(msgs)
    name_born(L + 1, m)
    close(L, before[0], before[1], next(p for p in m if p.kind == 'carrying'))
    fm = next(p for p in m if p.kind == 'fold-mask')
    fold_mask_coins(G, fm, fold_k(Q, L + 1))
    send(G, m, codes, salt)
    last = (Q['code'], Q['mask'], fold_k(Q, L + 1), 0)
    test(G, msgs, [last] + Q['later'], Q['ride'], fm, codes, salt, samples, (file, 'first'), members_of(L + 1, m))
    mw = next(msgs); w_part(G, mw, T, codes, salt)
    assert next(msgs, None) is None
    out['chain-bytes']['second'] = G.bytes(32, 'the random bytes of the second chain')
    msgs = iter([m for _, m in C.msgs[(file, 'second')]])
    test(G, msgs, T['tests'][2], T['ride'], mw[1] if T['ahead'] else None, codes, salt, samples, (file, 'second'), [(rider, False), (mw[0], False), (omega, False)]); assert next(msgs, None) is None
    return out


def fold_k(q, m):
    """k of group m (its FOLD's rounds): R12.16."""
    L = len(q['links'])
    own = q['links'][m - 1][1] if m <= L else []
    main = (2 if m > 1 else 0) + sum(cut.up2(x[0]) for j in range(1, m) for x in q['links'][j - 1][1] if x[5] > 0 and j + x[5] == m) + sum(cut.up2(x[0]) for x in own if x[5] == 0)
    h = cut.log_up(sum(cut.up2(x[2]) for x in own))
    return max(cut.log_up(main), h, 1)


def scalars_of_links(C, file, L):
    """For each segment, how many elements of E its sub-proof sends: the records of the four classes
    below, from the round with the segment's mask row up to the next round `BAT`. Sets their
    `link`."""
    out = [0] * L
    sent = [n for n, m in C.msgs[(file, 'first')] if any(p.kind == 'link' and p.mask_rows for p in m)]
    bats = [i + 1 for i, e in enumerate(C.runs[(file, 'first')]['entries']) if e[0] == 'BAT']  # a round BAT has an empty message, so no record: its number comes from the plan
    ends = [min(b for b in bats if b > s) for s in sent]
    for r in C.recs:
        if r.file != file or r.chain != 'first' or r.rnd is None: continue
        for link in range(L):
            if sent[link] <= r.rnd < ends[link] and r.cls in ('hiding-sum', 'sumcheck-coefficient', 'end-value', 'subproof-scalar'): out[link] += r.n // 24; r.link = link
    return out


def link_psc_sums(J, C, f, pr):
    """Checks the hiding sums (sums of hiding polynomials) of the sumchecks (PSC) in each segment's
    lookups (for its lookup calls and its approximate range check's digits): 711 of a proof's
    747 at the small example configurations. Of the other 36, level1.Judge.batt_sum() checks one.
    The sub-proofs' own sumchecks are not checked: their weights and points are specific to each
    sub-proof. They number 35 a proof at the small sizes and 185 at full size, according to
    tools/audit/README.md, "What a batch's YES does not cover".

    The rule is that of level1.Judge.key_psc_sums(), on the segment's own mask entries.
    A sumcheck's scalars and its mask entries have the same numbers (prover/hiding/psc_device.cuh),
    so its first entry's number is the count of scalars that the segment sent before the hiding sum.

    The k rounds of the last level of `keys` follow orderForestRange() (verifier/src/vf_lookup_argument.h),
    which needs rK, the number of bits of a tree's number. rK is public but in no item that the
    audit parses: it is found by trial, at most k - 8 values (range(0, k - 8) below) against one
    element of E.

    Needs `link` from scalars_of_links() and each mask row's `row` from run()."""
    Ex = J.Ex; chain = (f, 'first'); ent = C.runs[chain]['entries']; L = len(pr['Q']['links'])
    sent = [n for n, m in C.msgs[chain] if any(p.kind == 'link' and p.mask_rows for p in m)]
    rows = [p for n, m in C.msgs[chain] for p in m if p.kind == 'link' and p.mask_rows]
    bats = [i + 1 for i, e in enumerate(ent) if e[0] == 'BAT']; ends = [min(b for b in bats if b > s) for s in sent]
    recs = {}
    for r in C.recs:
        if r.file == f and r.chain == 'first' and r.link is not None: recs.setdefault(r.rnd, []).append(r)
    held, found = 0, []
    for link in range(L):
        row = rows[link].row; e = lambda i: tuple(row[3 * i:3 * i + 3])
        slot, i = 0, sent[link]; p, q, zero = [], [], None
        while i < ends[link]:
            nm, where, drew = C.draws[chain + (i,)]; here = sum(r.n // 24 for r in recs.get(i, []))
            if nm == 'LKP': p = []
            elif nm == 'JOIN': q = list(drew)
            elif nm == 'ZERO': zero = list(drew)
            sec = where.rsplit(', ', 1)[-1]
            if nm == 'PSC epsilon' and sec in ('tie', 'keys', 'table', 'leaves', 'bits'):
                j = i + 1; g, nc = [], []
                while C.draws[chain + (j,)][0] == 'PSC round': g.append(C.draws[chain + (j,)][2][0]); nc.append(ent[j - 1][3] // 24); j += 1
                has_y = C.draws[chain + (j,)][0] == 'PSC Y'; k = len(g); jj = j + (2 if has_y else 0)
                point = {'tie': [], 'keys': p, 'table': q, 'leaves': q, 'bits': zero}[sec]; assert len(point) == k, where
                hs = [r for r in recs[i] if r.cls == 'hiding-sum']; assert len(hs) == 1
                s0 = slot + here - 1  # single values may precede the sum in this round's message
                def total(order):
                    at, acc = s0, Ex.zero
                    for t, x in enumerate(order):
                        m = nc[t] + (1 if not has_y and t == k - 1 else 0); blk = [e(at + u) for u in range(m)]; at += m
                        if not has_y and t == k - 1: acc = Ex.add(acc, blk[0]); blk = blk[1:]
                        sm = Ex.zero
                        for y in blk: sm = Ex.add(sm, y)
                        acc = Ex.add(acc, Ex.mul(point[x], sm))
                    return struct.pack('<3Q', *(Ex.add(acc, e(at)) if has_y else acc))
                order = list(range(k - 1, -1, -1))
                after = C.draws[chain + (jj + 1,)] if sec == 'keys' else None
                last_level = sec == 'keys' and not (after[0] == 'PSC epsilon' and after[1].endswith(', keys'))
                if last_level:
                    for rK in range(0, k - 8):
                        o = [rK + v for v in (2, 1, 0, 5, 4, 3, 8, 7, 6)] + list(range(k - 1, rK + 8, -1)) + list(range(rK - 1, -1, -1))
                        if total(o) == J.raw(hs[0]): order = o; found.append(rK); break
                J.verdict(hs[0], total(order) == J.raw(hs[0]), 'the hiding sum of a sum-check in «%s»' % where); held += 1
                gamma = [None] * k
                for x, c in zip(order, g): gamma[x] = c
                if sec in ('keys', 'table'):
                    assert C.draws[chain + (jj,)][0] == 'LINE'; new = gamma + [C.draws[chain + (jj,)][2][0]]
                    if sec == 'table': q = new
                    else: p = new
            slot += here; i += 1
        assert slot == rows[link].entries
    return held, found


# Called by level1.run(). W: audit.py's dictionary. Assumes one batch; without --batch also first
# live leaf table 0, no empty slot, no failed request, and slots in the sorted order of the proofs'
# names.
def run(W, J, S, version, ID, name_of, listing):
    C, R, field, rd, conf, say = W['C'], W['R'], W['field'], W['rd'], W['conf'], J.say
    names = sorted(W['proofs'])
    # slots (under --batch from batch.read()): (name, the request's words, or None if the file is no
    # well-formed request), in slot order. A request's contexts have its slot's number as counter
    # and member, also where earlier slots are empty or failed (prover/prover/isk.cuh); defined in
    # docs/format/COIN-KEYS.md, "2a. From a key to a random value".
    slots = W.get('slots') or [(n, list(struct.unpack('<%dI' % (len(rd('in/%s.req' % n)) // 4), rd('in/%s.req' % n)))) for n in names]
    slot_of = {n: j for j, (n, _) in enumerate(slots)}
    rho = bytes(W['batch']) if W.get('batch') else rd('coins.test')[:32]
    copies, first_live = W.get('copies') or (len(W['entries']), 0)
    first = S.batch_first(rho, conf, copies, first_live, [w for _, w in slots])
    copy = first_live + CO.draw_copy(first, copies - first_live)
    for n in names:
        for r in J.by_cls[(n + '.proof', 'head/copy-number')]: J.verdict(r, struct.pack('<I', copy) == J.raw(r), 'the drawn copy')
    # `answers`: the stored answers of the drawn leaf table's record (bytes).
    process = S.batch_process(first, name_of(W['entries'][copy]['root']), W['entries'][copy]['answers'])
    gens = []
    f0 = names[0] + '.proof'
    GB = Gen(CO.Context(process, 0, version, 4, 0, field, "the batch's own coins")); gens.append(GB)
    outB = script_batch(GB, C, f0, W['proofs'][names[0]])
    hold_ledger(J, "the batch's own", GB, listing, 'batch', version, 4, 0, 0)
    if 'batch' in listing and len(listing['batch']) != 1 + len(names): J.cries["(L) the prover's listing has %d generators, not one for the batch and one a seat" % len(listing['batch'])] += 1
    # All the batch's files repeat the shared chain; the first file's `Part`s (cut.py) get its
    # draws.
    J.events(GB)
    J.chain_parts(f0, ('shared',)); J.answers(f0, ('shared',)); J.fold_sums(f0, 'shared', masks_in_order(C, f0, 'shared'))
    # Each other file's `Part`s and trees get the same draws, so its own records are checked
    # too, also where its second chain opens Omega's tree.
    span = lambda f: [r for r in C.recs if r.file == f and r.chain == 'shared']
    a = span(f0)
    for n in names[1:]:
        f = n + '.proof'; b = span(f)
        same = len(a) == len(b) and J.data[f0][a[0].off:a[-1].off + a[-1].n] == J.data[f][b[0].off:b[-1].off + b[-1].n]
        say('the shared chain in %s: %s' % (f, 'the same bytes as in ' + f0 if same else 'OTHER BYTES THAN IN ' + f0))
        if not same: J.cries['%s | the shared chain differs from the first file\'s' % f] += 1
        for (_, m0), (_, m1) in zip(C.msgs[(f0, 'shared')], C.msgs[(f, 'shared')]):
            assert len(m0) == len(m1)
            for p0, p1 in zip(m0, m1):
                assert p0.name == p1.name
                for k_ in ('coins', 'k', 'w', 'table', 'main_table', 'pads', 'mask_tail', 'tail', 'msg_rows', 'room', 'main_room', 'rest_len'):
                    if hasattr(p0, k_): setattr(p1, k_, getattr(p0, k_))
                p1.tree.salts = p0.tree.salts
        J.chain_parts(f, ('shared',)); J.answers(f, ('shared',)); J.fold_sums(f, 'shared', masks_in_order(C, f, 'shared'))
        J.base_again(GB, f0, f)
    # The whole binding tree (over the first chains' seals). Requests take their places from the end
    # of `free`.
    depth = W['proofs'][names[0]]['depth']; n_leaves = 1 << depth
    leaves = [d.bytes() for d in outB['free-leaves']]
    well_formed = sum(w is not None for _, w in slots)
    taken = outB['free'][::-1][:well_formed]
    got = [W['proofs'][n]['place'] for n in names]
    # A well-formed request without a proof: if it reached its seal, that seal is a leaf of the
    # binding tree, and no file has it. `lost`: the places that such requests may have taken. A
    # sibling above one of them cannot be derived: it is not compared and stays at level 0.
    lost = [p for p in taken if p not in got] if well_formed > len(names) else []
    if lost: say('places: the %d proofs\' places %s among the %d that the replayed arrangement gives out first' % (len(names), 'are' if set(got) <= set(taken) else 'ARE NOT', well_formed))
    else: say('places: the %d proofs\' places %s those that the replayed arrangement gives out first%s' % (len(names), 'are' if sorted(taken) == sorted(got) else 'ARE NOT', '' if taken == got or sorted(taken) != sorted(got) else ' (in another order than the slots\')'))
    for n in names:
        pr = W['proofs'][n]; f = n + '.proof'
        for r in J.by_cls[(f, 'place-word')]: J.verdict(r, struct.unpack('<Q', J.raw(r))[0] in taken, 'the place is one that the arrangement gives out to this batch')
        for r in J.by_cls[(f, 'binding-salt')]: J.verdict(r, outB['binding-salts'].bytes(40 * pr['place'], 40) == J.raw(r), "the binding leaf's salt")
        leaves[pr['place']] = R.binding_leaf(depth, pr['place'], conf, pr['seal1'], outB['binding-salts'].bytes(40 * pr['place'], 40))
    if len(set(got)) != len(got): J.cries['two requests at one place'] += 1
    level = {depth: leaves}
    for d in range(depth - 1, -1, -1): level[d] = [R.binding_vertex(conf, level[d + 1][2 * i], level[d + 1][2 * i + 1]) for i in range(1 << d)]
    for n in names:
        pr = W['proofs'][n]
        for r in J.by_cls[(n + '.proof', 'binding-sibling')]:
            if any(p >> r.x == (pr['place'] >> r.x) ^ 1 for p in lost): r.why = "a failed request's seal is in no file"; W['siblings not compared'] = W.get('siblings not compared', 0) + 1
            else: J.verdict(r, level[depth - r.x][(pr['place'] >> r.x) ^ 1] == J.raw(r), 'a sibling in the binding tree, made anew from random values and the batch\'s seals')
    if lost: say('the binding tree: %d well-formed request(s) have no proof: %d siblings above the places that they may have taken are NOT COMPARED: a failed request\'s seal is in no file' % (len(lost), W.get('siblings not compared', 0)))
    # The requests that have a proof; j is a request's slot.
    ids = cut.words_of(W['proofs'][names[0]]['names']["links' identifiers"])[1:]
    for n in names:
        f = n + '.proof'; pr = W['proofs'][n]; j = slot_of[n]
        G = Gen(CO.Context(process, j, version, 2, j, field, 'the seat %d' % j)); gens.append(G)
        entries = scalars_of_links(C, f, len(pr['Q']['links']))
        if n == names[0]: W.setdefault('notes', []).append('mask entries that the sub-proofs use, by the audit\'s count of sent scalars: %s' % entries)
        out = script_seat(G, C, f, pr, entries, outB['omega'])
        hold_ledger(J, 'seat %d' % j, G, listing, 'batch', version, 2, j, j)
        J.chain_bytes(f, out['chain-bytes'])
        # A segment's mask row: slot pads (the entries that its sub-proof uses), then `rest`, with
        # SEND's pads last. The slot pads' label follows
        # prover/second_writing/request_coins.h and prover/second_writing/coin_streams.h.
        old = CO.Context(process, j, version, 1, j, field, 'the seat %d, the older library' % j); gens.append(Gen(old))
        codes = C.runs[(f, 'first')]['codes']; ell = codes[1][pr['Q']['mask']][1]
        link = 0
        for _, m in C.msgs[(f, 'first')]:
            for p in m:
                if p.kind == 'link' and p.mask_rows:
                    ns = p.entries
                    cells = old.elements(4 * (4096 + 64 * ids[link]) + 1, 0, 3 * ns, 'the slot pads of link %d' % (link + 1))
                    row = [c for y in range(ns) for c in (cells[y], cells[ns + y], cells[2 * ns + y])] + flat(es(p.rest))
                    for i, e in enumerate(es(p.pads)):
                        at = ell - p.answers + i; row[3 * at:3 * at + 3] = list(e)
                    p.row = row
                    link += 1
        J.events(G)
        J.chain_parts(f, ('first', 'second')); J.answers(f, ('first',))
        J.batt_sum(f, 'first', next(p for p in C.msgs[(f, 'first')][0][1] if p.kind == 'rider'), cut.words_of(pr['names']['pending rows'])[0])
        held, found = link_psc_sums(J, C, f, pr)
        if n == names[0]:
            say("%s: the links' look-ups: %d hiding sums held against the links' own mask entries" % (f, held))
            W.setdefault('notes', []).append("the bits of a tree's number in the links' last key levels, found by trial: %s" % found)
        fm = masks_in_order(C, f, 'first'); ahead = pr['T']['ahead']
        J.fold_sums(f, 'first', fm[:-1] if ahead else fm)
        J.fold_sums(f, 'second', (fm[-1:] if ahead else []) + masks_in_order(C, f, 'second'))
    return gens
