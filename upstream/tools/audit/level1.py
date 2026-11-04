#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Level 1 of the audit: recomputes published values from the secrets and compares them by value.

run() covers the values in the key file and the leaf files (`entry` in identifiers) that depend
only on the provider's secrets (not the weights) and on public data, except those that its comments
name; under --drawn or --batch, only the salts of the weight tables' opened leaves. It calls
level1_proof.py for proofs (not under --drawn), returns the failures (`cries`) and prints only
counts, names and verdicts. The audit: tools/audit/README.md.

coins.py derives the private random values (`coins`) as docs/format/COIN-KEYS.md specifies.
Which value goes where follows from the order of the prover's draws. On that order this check is
not independent of the programs: no document specifies it, and "the scripts" below follow the
provers' sources. So the comparison misses a fault that the scripts share with the programs.
Against such faults this file has only:
(H1) the replay gives no range of a stream to two differently named places (coins.Context.settle());
(H2) every row has at least as many random tail entries as distinct positions opened in the files;
(H3) no two opened tree leaves have the same salt."""
import collections
import os
import struct

import coins as CO
import cut
import field as FD
import folds as FO


# One draw of a Gen: one stream of `n` field elements or bytes.
class Draw:
    __slots__ = ('g', 'stream', 'n', 'kind', 'place')

    def __init__(s, g, stream, n, kind, place): s.g, s.stream, s.n, s.kind, s.place = g, stream, n, kind, place
    def elements(s, first=0, count=None):
        count = s.n - first if count is None else count
        assert s.kind == 'elements' and first + count <= s.n
        return s.g.ctx.elements(s.stream, first, count, s.place)
    def bytes(s, first=0, count=None):
        count = s.n - first if count is None else count
        assert s.kind == 'bytes' and first + count <= s.n
        return s.g.ctx.bytes(s.stream, first, count, s.place)
    def e(s): return tuple(s.elements(0, 3))


class Gen:
    """One generator (a coins.Context): streams are numbered in the order of the draws, from 16,384
    up (`COIN_FIRST_STREAM` in blocks/coins/coins.h)."""

    def __init__(s, ctx): s.ctx, s.ledger, s.events = ctx, [], []
    def _open(s, n, kind, place):
        d = Draw(s, 16384 + len(s.ledger), n, kind, place); s.ledger.append(d); return d
    def fill(s, n, place): return s._open(n, 'elements', place)
    def bytes(s, n, place): return s._open(n, 'bytes', place)
    def e(s, place): return s.fill(3, place)
    def f(s, place): return s.fill(1, place)


def es(draws):
    """The elements of E (the field's cubic extension) of many draws of three elements, at once.

    All draws must be of one Gen.
    """
    if not draws: return []
    return draws[0].g.ctx.first3_many([d.stream for d in draws], [d.place for d in draws])


# ---- the scripts ----
# Each function repeats some prover functions' draws, in their order, and attaches each draw to
# the cut.Part (one oracle message) or cut.Tree that it fills; the other draws it returns. These
# scripts and level1_proof.py's are not independent of the provers' sources that they follow:
# make_key_under() in blocks/frame/check_key.h;
# blocks/iop/{link,lin,wf,req,iopp,fold,cs,base_case,send}.h; send_oracles() in
# blocks/iop/context.h; Engine::commit() in blocks/engine/engine.h;
# prover/prover/{chain_frame.cuh,chain_run.cuh,batch_opening.h}.
# This file is not independent either of any file of the programs that it cites elsewhere, such as
# the two headers at fold_mask_rows(): the code at each such place follows the cited file.
# Whether this file is independent of any other file of the programs, of whatever kind, nobody has
# examined: of all the rest nothing is known, either way.
# A chain is a hash chain, the Fiat-Shamir transcript of one protocol run.
# codes: (base codes, mask codes); a code is (log2 of length, message entries, random tail entries).
# A stage of a proximity test is (base code, mask code, k, h): room for 2^k main, 2^h mask rows.
# W, Omega and the folded row are oracle messages of LIN, the test on the weight table
# (blocks/iop/lin.h).

def fold_mask_coins(G, p, k):
    p.coins = [G.e('coefficient %d of %s' % (i, p.name)) for i in range(2 * k + 1)]; p.k = k


def fresh_mask(G, p, codes):
    p.table = G.fill(3 * p.mask_rows * codes[1][p.mask_code][1], 'mask table of ' + p.name)


def send(G, msg, codes, salt_bytes, sampled=True):
    """send_prove() (the pads, which mask the samples' answers; only if sampled), then
    send_oracles(): mask rows' random tails by part; then, tree by tree: main rows' random tails by
    part, salts."""
    base, mask = codes
    if sampled:
        for p in msg:
            if not p.answers: continue
            if p.mask_rows or p.deg == 3: p.pads = [G.e('pad %d of %s' % (i, p.name)) for i in range(p.answers)]
            else: p.pads = [G.f('pad %d of %s' % (i, p.name)) for i in range(3 * p.answers)]
    for p in msg:
        if p.mask_rows: p.mask_tail = G.fill(3 * p.mask_rows * mask[p.mask_code][2], 'mask tail of ' + p.name)
    seen = []
    for p in sorted(msg, key=lambda p: (p.tree.number, p.order)):
        if p.tree not in seen: seen.append(p.tree)
    for t in seen:
        for p in t.parts:
            if p.main_rows: p.tail = G.fill(p.main_rows * base[p.code][2] * p.deg, 'main tail of ' + p.name)
        t.salts = G.bytes((1 << t.depth) * salt_bytes, 'salts of tree %d of %s (%s)' % (t.number, t.who, t.sent[0]))


def test(G, msgs, stages, ride, first_sent, codes, salt_bytes, samples, chain=None, group=None):
    """iopp_prove(): one proximity test.

    msgs: iterator over the test's prover messages that contain roots. ride: if true, each stage's
    message also carries the next fold's (FOLD's) mask or, at the last stage, the mask M*
    (`masking_word`, `masking-word` in names) of BASE, the test's last step. first_sent: the first
    fold's mask, if it was sent before, else None. samples: pads in each stage's message.
    chain: (file, chain title). group: the first stage's oracle messages, [(part, kept)].
    """
    base, mask = codes
    sent = first_sent
    for i in range(len(stages) - 1):
        st, nx = stages[i], stages[i + 1]; to_last = i + 2 == len(stages)
        if not sent:
            m = next(msgs); assert len(m) == 1 and m[0].kind == 'fold-mask'
            fold_mask_coins(G, m[0], st[2]); send(G, m, codes, salt_bytes, False); sent = m[0]
        fresh, sent = sent, None
        m = next(msgs); stage = m[0]; assert stage.kind == 'stage'
        G.events.append(dict(what='fold', chain=chain, members=group, fresh=fresh, k=st[2], h=st[3], code=base[st[0]], mask_code=mask[st[1]], target=stage, codes=codes))
        group = [(stage, False)]
        if ride and not to_last: fold_mask_coins(G, m[1], nx[2]); sent = m[1]
        word = None
        if to_last:
            word = m[1] if ride else None
            W = word if word is not None else cut.masking_word(nx, 'held')
            W.main_table = G.fill(3 * base[nx[0]][1], 'M* (main) of ' + W.name); W.table = G.fill(3 * mask[nx[1]][1], 'M* (mask) of ' + W.name)
            held = W
        # switch_drafts() in blocks/iop/cs.h, with `uniform_room`, fills unused entries with random
        # values: first after `rest` (folds.fold()), then after the folded row.
        rest = 2 * mask[st[1]][1] + base[st[0]][2] + 2 * mask[st[1]][2]
        places = (mask[nx[1]][1] << nx[3]) - samples
        stage.rest_len, stage.room = rest, [G.e('free room %d behind the rest in %s' % (j, stage.name)) for j in range(places - rest)]
        ell, room = base[st[0]][1], base[nx[0]][1] << nx[2]
        stage.main_room = [G.e('free room %d behind the row in %s' % (j, stage.name)) for j in range(room - ell)]
        send(G, m, codes, salt_bytes)
        if to_last and not ride:
            m = next(msgs); m[0].main_table, m[0].table = held.main_table, held.table; send(G, m, codes, salt_bytes, False); held = m[0]
        if to_last: G.events.append(dict(what='base', chain=chain, stage=stage, word=held, codes=codes))


def script_entry(G, C, name, e):
    run = C.runs[(name, 'link')]; codes, salt, samples = run['codes'], run['eight'][2], run['eight'][5]
    L = e['link']; msgs = iter([m for _, m in C.msgs[(name, 'link')]])
    out = {'chain-bytes': {'link': G.bytes(32, 'the random bytes of the chain link')}}
    st0 = (L['code'], L['mcode'], 1 + cut.log_up(L['R']), 0)
    m = next(msgs)
    if L['ahead']: fold_mask_coins(G, m[2], st0[2])
    send(G, m, codes, salt)
    test(G, msgs, [st0] + L['later'], L['ride'], m[2] if L['ahead'] else None, codes, salt, samples, (name, 'link'), [(m[0], False), (m[1], False)])
    assert next(msgs, None) is None
    return out


def w_part(G, m, T, codes, salt):
    if m[0].mask_rows: fresh_mask(G, m[0], codes)  # blocks/iop/lin.h: W's draft draws its mask table first if its shape has mask rows (full-size configurations)
    if T['ahead']: fold_mask_coins(G, m[1], T['tests'][2][0][2])
    send(G, m, codes, salt)


def shared_part(G, msgs, T, codes, salt, samples, chain):
    """LIN's shared chain. Returns Omega's part."""
    mu = next(msgs)
    if T['ahead']: fold_mask_coins(G, mu[1], T['tests'][0][0][2])
    send(G, mu, codes, salt)
    m = next(msgs)
    for p in m:
        if p.mask_rows: fresh_mask(G, p, codes)
    send(G, m, codes, salt)
    split = T['u'][0] != T['omega'][0]
    assert split, 'one test on the folded row and Omega together: not written for'
    test(G, msgs, T['tests'][0], T['ride'], mu[1] if T['ahead'] else None, codes, salt, samples, chain, [(mu[0], False), (m[1], False)])
    test(G, msgs, T['tests'][1], T['ride'], None, codes, salt, samples, chain, [(m[0], False), (m[1], False)])
    return m[0]


def script_key(G, C, k):
    name = 'key.bin'; run = C.runs[(name, 'one-time')]; codes, salt, samples = run['codes'], run['eight'][2], run['eight'][5]
    T, R = k['T'], k['G']
    out = {'chain-bytes': {'one-time': G.bytes(32, 'the random bytes of the one-time chain')}}
    msgs = iter([m for _, m in C.msgs[(name, 'one-time')]])
    m = next(msgs); fresh_mask(G, m[0], codes); send(G, m, codes, salt); rider = m[0]
    m = next(msgs); fresh_mask(G, m[0], codes)
    if R['ride']: fold_mask_coins(G, m[1], R['stages'][0][2])
    send(G, m, codes, salt)
    test(G, msgs, R['stages'], R['ride'], m[1] if R['ride'] else None, codes, salt, samples, (name, 'one-time'), [(m[0], False)])
    mw = next(msgs); w_part(G, mw, T, codes, salt)
    assert next(msgs, None) is None
    out['binding-salts'] = G.bytes(40, 'the salts of the binding tree'); out['free-leaves'] = [G.bytes(32, 'the free leaf 0 of the binding tree')]
    msgs = iter([m for _, m in C.msgs[(name, 'shared')]])
    omega = shared_part(G, msgs, T, codes, salt, samples, (name, 'shared')); assert next(msgs, None) is None
    out['chain-bytes']['second'] = G.bytes(32, 'the random bytes of the second chain')
    msgs = iter([m for _, m in C.msgs[(name, 'second')]])
    test(G, msgs, T['tests'][2], T['ride'], mw[1] if T['ahead'] else None, codes, salt, samples, (name, 'second'), [(rider, False), (mw[0], False), (omega, False)]); assert next(msgs, None) is None
    return out


def read_listing(path):
    out, cur = [], None
    for l in open(path):
        w = l.split()
        if w[0] == 'context': cur = dict(key=int(w[2]), counter=int(w[4]), version=int(w[6]), object=int(w[8]), member=int(w[10]), draws=[]); out.append(cur)
        else: cur['draws'].append((int(w[0]), int(w[1]), w[2]))
    return out


def hold_ledger(J, what, G, listing, which, version, obj, counter, member, nth=None):
    """Compares G's draws with the list that a test build wrote to the file named in COINS_CONTEXTS.

    By stream, length and kind. Reported as failures: a difference, a generator missing from the
    list or, without nth, listed twice, another version word, and a missing list, which under
    --commitment or --batch is only printed. nth: which of the listed generators with this (object,
    counter, member). No other check covers draws whose values no comparison reads: the pads that
    send_prove() (blocks/iop/send.h) puts into main rows, and values that are drawn and not used.
    """
    say = J.say
    if ('commitment' in J.W['opts'] or 'batch' in J.W['opts']) and which not in listing:
        say('ledger  %s: no listing (a build without test options writes none): the order of the draws, and the length of a draw that enters a compared value whole, are checked through these values alone' % what); return False
    if which not in listing:
        say('(L) %s: the world has no listing %s.contexts.txt: the order and lengths of the draws are checked against nothing' % (what, which)); J.cries['(L) %s: no listing to hold the ledger against' % what] += 1; return False
    found = [g for g in listing[which] if (g['object'], g['counter'], g['member']) == (obj, counter, member)]
    if nth is None and len(found) != 1 or nth is not None and nth >= len(found):
        say('(L) %s: the listing has %d generators under (counter %d, object %d, member %d)' % (what, len(found), counter, obj, member)); J.cries['(L) %s: not one generator of this context in the listing' % what] += 1; return False
    g = found[nth or 0]
    if g['version'] != version: J.cries['(L) %s: the listing has another version word' % what] += 1
    listed = g['draws']
    mine = [(d.stream, d.n, d.kind) for d in G.ledger]
    if mine == listed:
        say('ledger  %s: the audit\'s %d draws equal the program\'s listing, line for line' % (what, len(mine))); return True
    i = next((i for i, (a, b) in enumerate(zip(mine, listed)) if a != b), min(len(mine), len(listed)))
    J.cries['(L) %s: the audit\'s list of draws and the program\'s listing differ from draw %d on (stream %d: %s)' % (what, i, 16384 + i, G.ledger[i].place if i < len(mine) else 'behind the audit\'s last')] += 1
    say('(L) %s: the audit has %d draws, the listing %d; they part at draw %d: the audit\'s %s (%s), the listing\'s %s; before it: %s' % (
        what, len(mine), len(listed), i, mine[i] if i < len(mine) else None, G.ledger[i].place if i < len(mine) else '-', listed[i] if i < len(listed) else None, G.ledger[i - 1].place if 0 < i <= len(mine) else '-'))
    say('        the listing goes on: %s' % (listed[i:i + 12],)); say('        the audit\'s goes on: %s' % ([(d.n, d.kind[0], d.place[:40]) for d in G.ledger[i:i + 8]],))
    return False


# ---- from coins to rows ----
def fold_mask_rows(Ex, p, ell):
    """The messages of a fold mask's 2^h mask rows, each a list of ell elements of E.

    A second implementation of fold_mask_table(), as blocks/iop/fold_algebra.h describes it.
    Coefficients are in the order of hp_place() (blocks/iop/hp.h): by variable, then by rising
    degree; the last variable has degrees 0, 1, 2, the others 1, 2.
    This function is not independent of these two headers: it follows the same descriptions as their
    code, so an error in understanding them would be in both and the comparison would not show it.
    """
    k, rows = p.k, p.mask_rows; h = cut.log_up(rows)
    w = p.w = getattr(p, 'w', None) or es(p.coins)
    co = lambda l, t: w[2 * (l - 1) + (t if l == k else t - 1)]  # l counted from 1
    out = []
    for b in range(rows):
        ones = lambda l: all((b >> (j - 1)) & 1 for j in range(l, h + 1))
        row = [Ex.zero] * ell
        row[0] = co(h + 1, 1) if h < k else co(k, 0)
        for l in range(1, h + 1):
            if ones(l): row[0] = Ex.add(row[0], co(l, 1)); row[l] = co(l, 2)
        col = h + 1
        for l in range(h + 1, k + 1):
            for t in ((2,) if l == h + 1 else (1, 2)): row[col] = co(l, t); col += 1
        if h < k: row[col] = co(k, 0)
        out.append(row)
    return out


def fold_mask_sum(Ex, p):
    """The sum over {0,1}^k of U = sum_l w_l(X_l) prod_{j > l} X_j (blocks/iop/fold_algebra.h).

    U is FOLD's hiding polynomial; this sum is its "hiding sum".
    In term l the variables before l are free (2^(l-1) points); those after l must be 1.
    """
    k = p.k; w = p.w = getattr(p, 'w', None) or es(p.coins)
    co = lambda l, t: (w[2 * (l - 1) + (t if l == k else t - 1)] if (l == k or t >= 1) else Ex.zero)
    s = Ex.zero
    for l in range(1, k + 1):
        one = Ex.add(Ex.add(co(l, 0), co(l, 0)), Ex.add(co(l, 1), co(l, 2)))
        s = Ex.add(s, Ex.mul(((1 << (l - 1)) % Ex.p, 0, 0), one))
    return s


def table_rows(p, ell):
    """A drawn mask table's rows, as flat lists of 3 ell numbers.

    Layout (blocks/engine/table.h): a[(row * ell + place) * 3 + coordinate]. The pads replace the
    last entries of the last row (Send::pad() in blocks/iop/send.h).
    """
    a = p.table.elements()
    rows = [a[3 * ell * b:3 * ell * (b + 1)] for b in range(p.mask_rows)]
    for i, e in enumerate(es(getattr(p, 'pads', []))):
        at = ell - p.answers + i
        rows[-1][3 * at:3 * at + 3] = list(e)
    return rows


def flat(row): return [c for e in row for c in e]


# Compares records (cut.Rec: a byte range of one file, with its class) with recomputed values.
class Judge:
    def __init__(s, W):
        s.W, s.C, s.field = W, W['C'], W['field']; s.Ex = FD.E(s.field)
        s.data = {}
        s.by_part, s.by_tree_salt, s.by_cls = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        s.by_tree_digest, s.by_root, s.made = collections.defaultdict(list), collections.defaultdict(list), {}
        for r in s.C.recs:
            if r.part is not None: s.by_part[(id(r.part), r.cls)].append(r)
            elif r.cls == 'salt': s.by_tree_salt[id(r.tree)].append(r)
            elif r.cls == 'given-digest': s.by_tree_digest[id(r.tree)].append(r)
            elif r.cls == 'root': s.by_root[(r.file, s.as_cut(r))].append(r)
            s.by_cls[(r.file, r.cls)].append(r)
        s.cries, s.lines, s.chal = collections.Counter(), [], {}

    def say(s, x): s.lines.append(x); print(x)
    def es(s, draws): return es(draws)
    def raw(s, r): return s.data[r.file][r.off:r.off + r.n]

    def as_cut(s, r):
        """The record's bytes as cut.py read them; under --flip the copy in `data` differs."""
        return s.W['rd'](('out/' if r.file.endswith('.proof') else 'ivk/') + r.file)[r.off:r.off + r.n]

    def verdict(s, r, ok, what, level=1):
        r.level = level
        if not ok: r.bad = 1; s.cries['%s | %s%s | %s' % (r.file, r.cls, '/' + r.sub if r.sub else '', what)] += 1

    def salts(s, t):
        for r in s.by_tree_salt[id(t)]: s.verdict(r, t.salts.bytes(r.x * t.salt, t.salt) == s.raw(r), 'salts of tree %d of %s' % (t.number, t.who))

    def rows(s, p, cls, code, deg, messages, tails, nrows, sub=None, book=True, level=1):
        """Compares part p's records of class cls with the rows' codewords.

        messages, tails: one flat list per row. Returns how many records differ; with book = False
        it records no verdict.
        """
        recs = [r for r in s.by_part[(id(p), cls)] if sub is None or r.sub == sub]
        if not recs: return 0
        xs = [r.x & ((1 << code[0]) - 1) for r in recs]
        got = [FD.row_symbols(s.field, code, deg, messages[b], tails[b], xs) for b in range(nrows)]
        differ = 0
        for i, r in enumerate(recs):
            want = struct.pack('<%dQ' % (nrows * deg), *[c for b in range(nrows) for c in got[b][i]])
            if book: s.verdict(r, want == s.raw(r), p.name, level)
            differ += want != s.raw(r)
        return differ

    def remake(s, t, file, codes):
        """Rebuilds a tree whose parts have only mask rows: it consists of random values only.

        Every leaf is recomputed from the derived rows and salts, and every vertex above them
        (leaf() and build() in docs/format/format_rules.py); the root and every given digest
        of the tree's openings are compared. So every leaf is checked, opened or not: a wrong salt
        or symbol at an unopened leaf changes the root.
        (`leafno` 164: a tree made within a chain; 174: a stored tree, that of a weight table.)
        """
        if t.leafno != 164 or any(p.main_rows for p in t.parts): return
        if any(s.msg_rows(p, codes) is None for p in t.parts): return
        if t.root not in s.made:  # (every proof of a batch repeats the shared chain's trees)
            n = 1 << t.depth; xs = list(range(n)); cols = [[] for _ in xs]
            for p in t.parts:
                code = codes[1][p.mask_code]; r = code[2]; tail = p.mask_tail.elements()
                assert code[0] == t.depth, 'a tree of mask rows alone has as many leaves as its rows have symbols'
                for b in range(p.mask_rows):
                    for x, y in enumerate(FD.row_symbols(s.field, code, 3, flat(p.msg_rows[b]), tail[3 * r * b:3 * r * (b + 1)], xs)): cols[x] += y
            salts = t.salts.bytes(0, n * t.salt); R = s.C.R
            s.made[t.root] = R.build(t.depth, s.C.conf, [R.leaf(t.leafno, t.number, t.depth, x, t.start, cols[x], salts[x * t.salt:(x + 1) * t.salt]) for x in xs])
        value = s.made[t.root]; t.remade = True; what = 'tree %d of %s, made again whole' % (t.number, t.who)
        for r in s.by_root[(file, t.root)]: s.verdict(r, value[0][0] == s.raw(r), 'the root of ' + what)
        for r in s.by_tree_digest[id(t)]: s.verdict(r, value[r.x[0]][r.x[1]] == s.raw(r), 'a given digest of ' + what)

    def mask_part(s, p, codes):
        code = codes[1][p.mask_code]; ell, r = code[1], code[2]
        if s.msg_rows(p, codes) is None: return
        msg = [flat(x) for x in p.msg_rows]
        t = p.mask_tail.elements()
        s.rows(p, 'mask-symbol', code, 3, msg, [t[3 * r * b:3 * r * (b + 1)] for b in range(p.mask_rows)], p.mask_rows)

    def msg_rows(s, p, codes):
        """The messages of a part's mask rows, as lists of elements of E; None if not derived."""
        if not hasattr(p, 'msg_rows'):
            ell = codes[1][p.mask_code][1]
            if p.kind == 'fold-mask': p.msg_rows = fold_mask_rows(s.Ex, p, ell)
            elif hasattr(p, 'row'): p.msg_rows = [FO.triples(p.row)]
            elif hasattr(p, 'table'): p.msg_rows = [FO.triples(x) for x in table_rows(p, ell)]
            else: return None
        return p.msg_rows

    def challenges(s, chain):
        if chain not in s.chal:
            fo, la = [], []
            n = len(s.C.runs[chain]['entries'])
            for i in range(1, n + 1):
                name, where, drew = s.C.draws[chain + (i,)]
                if name == 'FOLD epsilon': fo.append([])
                elif name == 'FOLD round': fo[-1].append(drew[0])
                elif name == 'BASE lambda': la.append(drew[0])
            s.chal[chain] = (iter(fo), iter(la))
        return s.chal[chain]

    def events(s, G):
        """The folds, code switches and BASE steps of a generator's chains, in their order."""
        Ex = s.Ex
        for ev in G.events:
            codes = ev['codes']
            if ev['what'] == 'fold':
                g = ev['g'] = next(s.challenges(ev['chain'])[0]); assert len(g) == ev['k'], 'a fold of %d rounds, and k = %d' % (len(g), ev['k'])
                for p, kept in ev['members']:
                    if p.mask_rows: assert s.msg_rows(p, codes) is not None, p.name
                s.msg_rows(ev['fresh'], codes)
                rest = FO.fold(Ex, ev['members'], ev['fresh'], g, ev['k'], ev['h'], ev['code'], ev['mask_code'])
                t = ev['target']; pads = es(t.pads)
                if t.kind == 'stage':
                    assert len(rest) == t.rest_len
                    t.msg_rows = FO.lay(Ex, rest + es(t.room), t.mask_rows, codes[1][t.mask_code][1], pads)
                else:
                    t.rest_row = FO.lay(Ex, rest, 1, codes[0][t.code][1], pads)[0]
                    r = codes[0][t.code][2]; tail = t.tail.elements()
                    s.rows(t, 'main-symbol', codes[0][t.code], 3, [flat(t.rest_row)], [tail[3 * r:6 * r]], 1, sub='carrying-rest')
            else:
                lam = next(s.challenges(ev['chain'])[1]); ev['lambda'] = lam
                s.base(ev['stage'], ev['word'], lam, codes)

    def base_again(s, G, f0, f):
        """The BASE steps of the shared chain once more, on another file's own records."""
        twin = {}
        for (_, m0), (_, m1) in zip(s.C.msgs[(f0, 'shared')], s.C.msgs[(f, 'shared')]):
            for p0, p1 in zip(m0, m1): twin[id(p0)] = p1
        for ev in G.events:
            if ev['what'] == 'base': s.base(twin[id(ev['stage'])], twin[id(ev['word'])], ev['lambda'], ev['codes'])

    # BASE sends M' = M* + lambda M (blocks/iop/base_case.h); w holds the mask M*.
    def base(s, st, w, lam, codes):
        Ex = s.Ex
        if True:
            if True:
                cb, cm = codes[0][st.code], codes[1][st.mask_code]
                recs = {r.sub: r for r in s.by_part[(id(w), 'base-word')]}
                masked = lambda star, x: struct.pack('<%dQ' % (3 * len(x)), *flat([Ex.add(a, Ex.mul(lam, b)) for a, b in zip(star, x)]))
                main_tail = FO.main_tail_rows(st, cb[2])[0]
                s.verdict(recs['mask-message'], masked(FO.triples(w.table.elements()), st.msg_rows[0]) == s.raw(recs['mask-message']), "BASE's masked mask message of " + w.name)
                s.verdict(recs['main-tail'], masked(FO.triples(w.tail.elements()), main_tail) == s.raw(recs['main-tail']), "BASE's masked main tail of " + w.name)
                s.verdict(recs['mask-tail'], masked(FO.triples(w.mask_tail.elements()), FO.mask_tail_rows(st, cm[2])[0]) == s.raw(recs['mask-tail']), "BASE's masked mask tail of " + w.name)
                # The main message is not recomputed: (sent - M*) / lambda must be the last stage's
                # committed row, checked at its opened positions under the derived random tail.
                got = FO.triples(list(struct.unpack('<%dQ' % (3 * cb[1]), s.raw(recs['main-message'])))); inv = Ex.inv(lam)
                M = [Ex.mul(inv, Ex.sub(a, b)) for a, b in zip(got, FO.triples(w.main_table.elements()))]
                # (the last stage's main symbols depend on the witness: read, not derived; level 0)
                differ = s.rows(st, 'main-symbol', cb, 3, [flat(M)], [flat(main_tail)], 1, book=False)
                if not s.by_part[(id(st), 'main-symbol')]: differ = 1  # nothing opened: nothing compared
                s.verdict(recs['main-message'], differ == 0, "BASE's masked main message of %s: less M*, over lambda, it is not the committed last row" % w.name)

    def answers(s, file, titles):
        """The answers to the samples of oracle messages that have only mask rows.

        Send::form() in blocks/iop/send.h: eq(index, z) on every entry that is not a pad, beta on
        the answer's pad. An entry's index is row + (place << row bits).
        """
        Ex = s.Ex
        for title in titles:
            codes = s.C.runs[(file, title)]['codes']
            for number, m in s.C.msgs[(file, title)]:
                name, where, seeds = s.C.draws[(file, title, number)]
                if name != 'SEND seed': continue
                for p in m:
                    if p.main_rows or not p.answers or s.msg_rows(p, codes) is None: continue
                    assert p.answers == 1 and len(seeds) == 1
                    z, beta = seeds[0]; ell = codes[1][p.mask_code][1]; rb = cut.log_up(p.mask_rows); bits = rb + cut.log_up(ell)
                    w = FO.weights(Ex, z[:bits]); acc = Ex.zero
                    for b, row in enumerate(p.msg_rows):
                        for c, e in enumerate(row):
                            if b == p.mask_rows - 1 and c == ell - 1: acc = Ex.add(acc, Ex.mul(beta, e))
                            else: acc = Ex.add(acc, Ex.mul(w[b + (c << rb)], e))
                    for r in s.by_part[(id(p), 'sample-answer')]: s.verdict(r, struct.pack('<3Q', *acc) == s.raw(r), 'the answer to the sample of ' + p.name)

    def word_part(s, p, codes):
        code = codes[0][p.code]
        s.rows(p, 'masking-word-symbol', code, 3, [p.main_table.elements()], [p.tail.elements()], 1)

    def chain_parts(s, file, titles):
        for title in titles:
            codes = s.C.runs[(file, title)]['codes']
            for number, m in s.C.msgs[(file, title)]:
                trees = []
                for p in m:
                    if p.tree not in trees: trees.append(p.tree)
                    if p.mask_rows: s.mask_part(p, codes)
                    if p.kind == 'masking-word': s.word_part(p, codes)
                for t in trees: s.salts(t); s.remake(t, file, codes)
        return

    def fold_sums(s, file, title, masks):
        """masks: the chain's fold masks, in the order of its rounds `FOLD epsilon`."""
        recs = [r for r in s.by_cls[(file, 'hiding-sum')] if r.chain == title and r.rname == 'FOLD epsilon']
        assert len(recs) == len(masks), '%s %s: %d rounds FOLD epsilon, %d fold masks' % (file, title, len(recs), len(masks))
        for r, p in zip(recs, masks): s.verdict(r, struct.pack('<3Q', *fold_mask_sum(s.Ex, p)) == s.raw(r), 'the sum of U of ' + p.name)

    def batt_sum(s, file, title, rider, first):
        """BATT's hiding sum (`where` in the plan: «rows on the table»).

        BATT (blocks/iop/batt.h) takes 1 + 2 k_col entries of the index-mask message `rider`, after
        the `first` entries that the pending claims' masks take. Its hiding polynomial has k_col
        plainly summed variables of degree 2 and no Y: the form of FOLD's U
        (blocks/iop/hp.h: psi = X for a plain sum). A match also shows which entries BATT takes.
        """
        n_w = s.C.msgs[(file, title)][-1][0]; k = 0
        while s.C.draws[(file, title, n_w - 1 - k)][0] == 'PSC round': k += 1
        at = n_w - 1 - k; assert s.C.draws[(file, title, at)][0] == 'PSC epsilon' and s.C.draws[(file, title, at - 1)][0] == 'BAT-sigma'
        # `rider` has Z mask rows. Its entries are taken by rising index e, as in key_psc_sums():
        # row e mod 2^h, place e div 2^h, where 2^h is the least power of two not below Z. An index
        # whose row is Z or more names no entry and is skipped. With one row, e is the place.
        Z, h = rider.mask_rows, cut.log_up(rider.mask_rows); taken, e = [], 0
        assert first + 2 * k + 1 <= Z * (len(rider.msg_rows[0]) - rider.answers) + Z - 1, "BATT's hiding sum wants more entries than the index-mask message has before its pads"  # as `Send::free_mask_entries` in blocks/iop/send.h: the pads are the last entries of the last row, and no entry at or after the first pad's index is taken
        while len(taken) < first + 2 * k + 1:
            if e & ((1 << h) - 1) < Z: taken.append(rider.msg_rows[e & ((1 << h) - 1)][e >> h])
            e += 1
        class U: pass
        u = U(); u.k = k; u.w = taken[first:]; u.coins = []
        for r in s.by_cls[(file, 'hiding-sum')]:
            if r.chain == title and r.rnd == at: s.verdict(r, struct.pack('<3Q', *fold_mask_sum(s.Ex, u)) == s.raw(r), "BATT's hiding sum, from the index-mask entries %d .. %d" % (first, first + 2 * k))

    def key_psc_sums(s, K):
        """The hiding sums of the key file's lookups, from mask entries and public points only.

        They are the messages of the rounds `PSC epsilon` with `where` «ranges: ..» in the plan
        (PSC: the sumcheck with a hiding polynomial U).
          sum of U = (the last variable's constant) + sum_l z_l (w_l,1 + w_l,2 + ..)
        (this file's own derivation) from:
        blocks/iop/hp.h: U = sum_l w_l(X_l) prod_{j>l} psi_j(X_j); only the last variable has a
          constant. Equality weights: phi(g) = (1 - z) g(0) + z g(1), psi = 1. Y: phi(g) = g(0).
        blocks/iop/psc.h: mask entries used = scalars sent: w in hp_place()'s order, then the c_j
          (one per shown map).
        blocks/iop/mask.h: entries are taken by rising index, the row in the low bits; rows beyond
          the real ones are skipped.
        blocks/iop/lkp.h: the steps and the order of rounds. A level's claim is at the previous
          level's point plus LINE's coordinate; the table side starts at the point of `JOIN`, the
          bits' sumcheck at that of `ZERO`.
          The two inner masks of a part's last key level come from the index-mask message.
        """
        C, Ex = s.C, s.Ex; chain = ('key.bin', 'one-time'); ent = C.runs[chain]['entries']; n = len(ent)
        part = C.msgs[chain][1][1][0]; assert part.kind == 'multiplicities'
        rows, lr = part.mask_rows, cut.log_up(part.mask_rows); nxt = [0]
        def take(m):
            out = []
            for _ in range(m):
                while (nxt[0] & ((1 << lr) - 1)) >= rows: nxt[0] += 1
                out.append(part.msg_rows[nxt[0] & ((1 << lr) - 1)][nxt[0] >> lr]); nxt[0] += 1
            return out
        sums = {r.rnd: r for r in s.by_cls[('key.bin', 'hiding-sum')] if r.chain == 'one-time'}
        k_col = K['T']['k_lo'] + K['shape'][3]
        look, prt, p, q, zero, i, held = -1, 0, [], [], None, 1, 0
        while i <= n:
            name, where, drew = C.draws[chain + (i,)]
            if name == 'LKP': look += 1; prt = 0; p = []
            elif name == 'JOIN': q = list(drew)
            elif name == 'ZERO': zero = list(drew)
            elif name == 'PSC epsilon' and where.startswith('ranges: ') and ', ' in where:  # («ranges: rows on the table» is BATT's: `batt_sum`)
                sec = where.rsplit(', ', 1)[1]; j = i + 1; g, nc = [], []
                while C.draws[chain + (j,)][0] == 'PSC round': g.append(C.draws[chain + (j,)][2][0]); nc.append(ent[j - 1][3] // 24); j += 1
                deg = shown = 0; has_y = C.draws[chain + (j,)][0] == 'PSC Y'
                if has_y: deg = ent[j - 1][3] // 24; shown = ent[j][3] // 24; assert C.draws[chain + (j + 1,)][0] == 'PSC values'; j += 2
                k = len(g); L = K['G']['lookups'][look]
                point = {'tie': [], 'parts': [], 'keys': p, 'table': q, 'leaves': q, 'bits': zero}[sec]; assert len(point) == k, '%s: a sum-check of %d rounds on a claim at %d coordinates' % (where, k, len(point))
                last_level = sec == 'keys' and k == L['parts'][prt][0] + k_col - 1
                order = (list(L['bounds']) + [x for x in range(k - 1, -1, -1) if x not in L['bounds']]) if last_level else list(range(k - 1, -1, -1))
                acc = Ex.zero
                for t, x in enumerate(order):
                    e = take(nc[t] + (1 if not has_y and t == k - 1 else 0))
                    if not has_y and t == k - 1: acc = Ex.add(acc, e[0]); e = e[1:]
                    sm = Ex.zero
                    for y in e: sm = Ex.add(sm, y)
                    acc = Ex.add(acc, Ex.mul(point[x], sm))
                if has_y: acc = Ex.add(acc, take(deg + 1)[0])
                if not last_level: take(shown)
                s.verdict(sums[i], struct.pack('<3Q', *acc) == s.raw(sums[i]), 'the hiding sum of a sum-check in «%s»' % where); held += 1
                gamma = [None] * k
                for x, c in zip(order, g): gamma[x] = c
                if sec in ('keys', 'table'):
                    assert C.draws[chain + (j,)][0] == 'LINE'; new = gamma + [C.draws[chain + (j,)][2][0]]
                    if sec == 'table': q = new
                    elif last_level: prt += 1; p = []
                    else: p = new
                i = j - 1
            i += 1
        s.say("the key's look-ups: %d hiding sums held against the multiplicities' mask entries (%d entries handed out of %d)" % (held, nxt[0] - (nxt[0] >> lr) * ((1 << lr) - rows), rows * len(part.msg_rows[0]) - 1))

    def chain_bytes(s, file, draws):
        for r in s.by_cls[(file, 'chain-bytes')]: s.verdict(r, draws[r.chain].bytes() == s.raw(r), 'the random bytes of the chain ' + r.chain)


def masks_in_order(C, file, title):
    return [p for _, m in C.msgs[(file, title)] for p in m if p.kind == 'fold-mask']


# W: the dictionary that audit.py builds.
def run(W):
    C, R, field, rd, conf = W['C'], W['R'], W['field'], W['rd'], W['conf']
    CO.own_test()
    J = Judge(W); say = J.say
    J.data['key.bin'] = rd('ivk/key.bin')
    for k in W['entries']: J.data['leaf/%d.bin' % k] = rd('ivk/leaf/%d.bin' % k)
    for n in W['proofs']: J.data[n + '.proof'] = rd('out/%s.proof' % n)
    flipped = []
    if 'flip' in W['opts']:  # a self-test: bits flip in `J.data`, which the comparisons read; cut.py parsed the unchanged bytes
        first, last = {}, {}
        kind = lambda f: 'key' if f == 'key.bin' else 'entry' if f.startswith('leaf/') else 'proof'
        for r in C.recs:
            k_ = (kind(r.file), r.chain, r.cls, r.sub); first.setdefault(k_, r); last[k_] = r
        small = [r for r in C.recs if r.cls in ('end-value', 'sample-answer', 'hiding-sum', 'split-value', 'base-target')]  # only some records of these classes are checked, so all get a flipped bit
        # Hashes pick the further records and each record's bit: this program draws no randomness.
        import hashlib
        pick = lambda *a: int.from_bytes(hashlib.sha256(repr(a).encode()).digest()[:8], 'little')
        of = collections.defaultdict(list)
        for r in C.recs:
            if r.cls in ('salt', 'stored-salt', 'binding-salt', 'binding-sibling', 'chain-bytes', 'mask-symbol', 'masking-word-symbol', 'base-word', 'given-digest', 'root') or (r.cls == 'main-symbol' and r.sub == 'carrying-rest'): of[(kind(r.file), r.chain, r.cls, r.sub)].append(r)
        more = [v[pick(k_, i) % len(v)] for k_, v in of.items() for i in range(16)]
        flipped = list({id(r): r for r in list(first.values()) + list(last.values()) + small + more}.values())
        for f in J.data: J.data[f] = bytearray(J.data[f])
        for r in flipped:
            b = pick(r.file, r.off) % (8 * r.n); J.data[r.file][r.off + b // 8] ^= 1 << (b % 8)
        for f in J.data: J.data[f] = bytes(J.data[f])
        say('FLIP: one bit is turned, in the comparisons\' copy alone, in %d records: the first and the last of each (kind of file, chain, class), sixteen more of each class of coins alone, and every end value, sample answer, hiding sum, split value and BASE target; which bit of a record: picked over its whole length' % len(flipped))
    world = W['world']
    drawn = 'drawn' in W['opts']
    kept = W.get('kept')  # --commitment: the 32 fresh bytes of the key file and of each leaf file (kept by `cp_commit -keep-for-the-audit`)
    batch = W.get('batch')  # --batch: the 32 fresh bytes of one batch (kept by `cp_prove -keep-for-the-audit`)
    seed = rd('isk/seed.bin') if drawn or kept or batch else rd('seed.test')
    if drawn and b'seed test' in rd('isk/isk.txt'): say('--drawn on files that were made under a test\'s seed: audited as if the makings\' and the batches\' bytes were kept nowhere')
    S = CO.Schedule(R.H, CO.words8(seed))
    version = struct.unpack('<I', C.version)[0]
    shape = W['key']['shape']
    name_of = lambda root: R.table_name(root, conf, *shape)
    seeds = [([R.ext(x) for x in z], R.ext(beta)) for z, beta in R.record_seeds(name_of(W['key']['root']), (shape[1] - 1).bit_length(), shape[4], field)]
    ID = R.published_id(conf, name_of(W['key']['root']), seeds, [W['key']['answers'][i:i + 24] for i in range(0, len(W['key']['answers']), 24)])  # docs/format/FORMAT.md, R4.11: the commitment ID's hash includes the record's seeds and the stored sample answers, 24 bytes each, of the key file's weight table
    W['ID'] = ID.hex() if isinstance(ID, (bytes, bytearray)) else struct.pack('<%dI' % len(ID), *ID).hex()
    listing = {}
    for what in ('commit', 'batch'):
        p = W['at'](what + '.contexts.txt')
        if os.path.exists(p): listing[what] = read_listing(p)
    zeros = bytes(32)
    gens = []

    # ---- the key file ----
    if drawn: say('WITH --drawn: the 32 bytes of the system that went into the key file, into each entry and into each batch are not read (they are kept only under -keep-for-the-audit, for --commitment and --batch): what follows from them is not recomputed; compared: the stored tables\' salts%s' % (' and columns' if W['opts']['level'] == '2' else ''))
    if batch: say('A BATCH\'S AUDIT: the 32 random bytes that went into the key file and into each entry are kept nowhere by now: in these files only the stored tables\' salts are recomputed; the proofs are recomputed as far as level 1 reaches')
    if not drawn and not batch:
        G = Gen(CO.Context(S.key_file(bytes(kept['key']) if kept else zeros, ID), 0, version, 5, 0, field, 'the key file\'s making')); gens.append(G)
        out = script_key(G, C, W['key'])
        hold_ledger(J, 'key file', G, listing, 'commit', version, 5, 0, 0)
        if 'commit' in listing and len(listing['commit']) != 1 + len(W['entries']): J.cries['(L) the committer\'s listing has %d generators, not one for the key file and one an entry' % len(listing['commit'])] += 1
        J.chain_bytes('key.bin', out['chain-bytes'])
        for r in J.by_cls[('key.bin', 'binding-salt')]: J.verdict(r, out['binding-salts'].bytes() == J.raw(r), 'the binding leaf\'s salt')
        J.events(G)
        J.chain_parts('key.bin', ('one-time', 'shared', 'second')); J.answers('key.bin', ('one-time', 'shared'))
        J.batt_sum('key.bin', 'one-time', C.msgs[('key.bin', 'one-time')][0][1][0], 2 * sum(len(L['parts']) for L in W['key']['G']['lookups']))
        J.key_psc_sums(W['key'])
        # With `ahead`, the one-time chain's last fold mask (sent with W) is for the second chain's
        # first fold.
        ot = masks_in_order(C, 'key.bin', 'one-time'); J.fold_sums('key.bin', 'one-time', ot[:-1] if W['key']['T']['ahead'] else ot)
        J.fold_sums('key.bin', 'shared', masks_in_order(C, 'key.bin', 'shared'))
        J.fold_sums('key.bin', 'second', (ot[-1:] if W['key']['T']['ahead'] else []) + masks_in_order(C, 'key.bin', 'second'))

        # ---- the leaf files ----
        for k, e in W['entries'].items():
            f = 'leaf/%d.bin' % k
            G = Gen(CO.Context(S.entry(k, bytes(kept[k]) if kept else zeros, ID, name_of(e['root']), e['answers']), 0, version, 10, 0, field, 'entry %d\'s making' % k)); gens.append(G)
            out = script_entry(G, C, f, e)
            hold_ledger(J, 'entry %d' % k, G, listing, 'commit', version, 10, 0, 0, nth=k)
            J.chain_bytes(f, out['chain-bytes'])
            J.events(G)
            J.chain_parts(f, ('link',))
            J.fold_sums(f, 'link', masks_in_order(C, f, 'link'))

    # ---- the stored tables' salts ----
    fresh = {}
    for k in W['entries']:
        p = 'isk/leaf/%d/fresh.bin' % k
        if batch and k not in W['live'] and not os.path.exists(W['at'](p)): continue  # a retired leaf table (`copy` in output) whose fresh.bin is gone, which cp_prove allows (`read_isk` in prover/prover/isk.cuh): its salts, which its leaf file opens, cannot be derived
        if (drawn or kept) and not os.path.exists(W['at'](p)): print('audit.py: refused: %s is missing; without it copy %d\'s coins cannot be derived' % (p, k)); raise SystemExit(2)
        fresh[k] = rd(p) if batch or os.path.exists(W['at'](p)) else zeros  # (--batch: if the file has vanished since it was seen, `rd` raises; audit.py ends with NOT WHOLE and, with --erase-kept-files, calls `KeptBatch.erase` in kept.py on the batch's kept file)
    # MASK and SALT are read as words with the first letter highest: 0x4D41534B and 0x53414C54 in
    # prover/hash_commitment/hf_table_format.h and, taken from the programs' source, in
    # docs/format/COIN-KEYS.md, "2. The schedule".
    # The other reading is not tried: that would evaluate the generator, under a table's commit-run
    # key, at a label that no program uses; this audit recomputes only what the programs compute.
    tc = {'key': CO.TableCoins(S.first_table(conf, shape), field, 'be'), **{k: CO.TableCoins(S.copy_table(k, fresh[k], conf, shape), field, 'be') for k in fresh}}
    roots = {W['key']['root']: 'key', **{e['root']: k for k, e in W['entries'].items()}}
    for r in C.recs:
        if r.cls == 'stored-salt' and roots[r.tree.root] not in tc: r.why = "a retired copy's fresh.bin is gone"; W['salts of retired copies'] = W.get('salts of retired copies', 0) + 1
        elif r.cls == 'stored-salt': J.verdict(r, tc[roots[r.tree.root]].salt(r.x) == J.raw(r), 'salts of the stored table %s' % roots[r.tree.root])
    if W.get('salts of retired copies'): say('retired copies without fresh.bin: %d salts of their tables, opened in their entries, are NOT COMPARED' % W['salts of retired copies'])
    W['table coins'] = tc

    if W['proofs'] and not drawn:
        import level1_proof
        gens += level1_proof.run(W, J, S, version, ID, name_of, listing)


    # Mode 'x': an existing --ledger file is never overwritten.
    if 'ledger' in W['opts']:
        with open(W['opts']['ledger'], 'x') as f:
            for G in gens:
                for d in G.ledger: f.write('%s\t%d\t%d\t%s\t%s\n' % (G.ctx.name, d.stream, d.n, d.kind, d.place))

    # ---- every record of random values only must have been checked ----
    # Under --drawn, and under --batch outside the proofs, only the stored tables' salts must.
    pure = lambda r: r.cls in ('salt', 'stored-salt', 'binding-salt', 'chain-bytes', 'mask-symbol', 'masking-word-symbol', 'head/copy-number', 'binding-sibling') or (r.cls == 'main-symbol' and r.sub == 'carrying-rest') or \
        (r.cls == 'place-word' and r.file != 'key.bin') or (r.cls == 'base-word') or (r.cls == 'hiding-sum' and (r.rname == 'FOLD epsilon' or (r.where or '').rsplit(', ', 1)[-1] in ('tie', 'parts', 'keys', 'table', 'leaves', 'bits') or (r.where or '').endswith('rows on the table')))
    for r in C.recs:
        if r.why: continue  # `why`: the record cannot be derived; this file (above) or level1_proof.py has printed the reason and the count
        if (r.cls == 'stored-salt' if drawn or (batch and not r.file.endswith('.proof')) else pure(r)) and r.level == 0: J.cries['(H0) %s | %s%s: a word of coins alone (or under a mask that level 1 derives) was held against nothing' % (r.file, r.cls, '/' + r.sub if r.sub else '')] += 1
    # remake() and answers() silently skip what is not theirs. So here every root and given digest
    # of a tree whose parts have only mask rows, and every answer to a sample of such a part
    # (kind rider or split-mask), must have been checked; under --batch, only those in proofs;
    # under --drawn, none.
    # play.sh checks the printed counts, except the third.
    # Not required: the tree of a BASE mask that is sent alone. Its one main row is random too, but
    # level 1 does not rebuild it: opened symbols are compared, an unopened leaf's salt is not.
    if not drawn:
        alone = {t.root for t in C.trees if t.leafno == 164 and not any(p.main_rows for p in t.parts) and not (batch and not t.sent[0].endswith('.proof'))}
        theirs = [r for r in C.recs if not (batch and not r.file.endswith('.proof')) and ((r.cls == 'given-digest' and r.tree.root in alone) or (r.cls == 'root' and J.as_cut(r) in alone) or (r.cls == 'sample-answer' and not r.part.main_rows))]
        for r in theirs:
            if r.level == 0: J.cries['(H0) %s | %s%s: of a tree or a message of mask rows alone, and held against nothing' % (r.file, r.cls, '/' + r.sub if r.sub else '')] += 1
        say('(H0) different trees of random values alone: %d, made again whole: %d; their roots and given digests held, in all files: %d; answers to samples of such messages held: %d' % (
            len(alone), len({t.root for t in C.trees if t.root in alone and getattr(t, 'remade', False)}), sum(r.level >= 1 for r in theirs if r.cls != 'sample-answer'), sum(r.level >= 1 for r in theirs if r.cls == 'sample-answer')))



    # ---- the audit's own conditions ----
    # (H1), with the check that no two contexts have the same key, is about this audit's own
    # bookkeeping only. What the program used it cannot see: a value that the program uses at two
    # places shows as a mismatch, or not at all.
    for G in gens:
        for c in G.ctx.settle(): J.cries['(H1) one cell for two places: %s: stream %d: %s | %s' % (G.ctx.name, c[0], c[1], c[2])] += 1
    keys = collections.Counter(tuple(int(x) for x in G.ctx.k) for G in gens) if not kept and not batch else {}  # (skipped under --commitment and --batch: there a key enters only the generator's block function)
    if keys and max(keys.values()) > 1: J.cries['(H1) two contexts of this world have one key'] += 1
    # one leaf can be opened in several files or lists: the same (tree's root, position) counts once
    where, met = collections.defaultdict(set), {}
    for r in C.recs:
        if r.cls in ('salt', 'stored-salt'): where[J.raw(r)].add((r.tree.root, r.x)); met.setdefault(J.raw(r), r.file)
        elif r.cls == 'binding-salt': where[J.raw(r)].add(('binding', r.file)); met.setdefault(J.raw(r), r.file)
    for b, v in where.items():
        if len(v) > 1: J.cries['(H3) one salt at two leaves: first met in %s' % met[b]] += 1
    leaves = len({x for v in where.values() for x in v})
    say('(H3) %d opened leaves: %s' % (leaves, 'no salt twice' if len(where) == leaves else '%d DIFFERENT SALTS ONLY' % len(where)))
    short = 0
    lists_of = collections.defaultdict(list)  # one tree can be in several files (the shared chain's) and be read in several chains: its lists are joined by its root
    for t in C.trees: lists_of[t.root] += t.lists
    done = set()
    for t in C.trees:
        if not t.lists or t.root in done: continue
        done.add(t.root)
        for p in t.parts:
            for rows, code in ((p.main_rows, None if p.kind == 'stored' else ('b', p.code)), (p.mask_rows, ('m', p.mask_code))):
                if not rows or code is None: continue
                cd = C.runs[(t.sent[0], t.sent[1])]['codes'][0 if code[0] == 'b' else 1][code[1]]
                opened = {x & ((1 << cd[0]) - 1) for l in lists_of[t.root] for x in l[3]}
                hides = cd[2] - (1 if field == 1 and cd[2] and cd[2] % 2 == 0 else 0)
                if len(opened) > hides: short += 1; J.cries['(H2) %s: %d positions opened, %d random entries' % (p.name, len(opened), cd[2])] += 1
    say('(H2) fresh rows opened, in these files, at more positions than their random entries hide: %d' % short)
    cols, drawn_cols = collections.defaultdict(set), collections.Counter()
    for t in C.trees:
        if t.leafno == 174:
            for l in t.lists: cols[t.root] |= set(l[3]); drawn_cols[t.root] += l[4]  # count.bin counts the positions drawn, repeats included
    hides = shape[2] - (1 if field == 1 and shape[2] % 2 == 0 else 0)
    for root, xs in cols.items():
        say('(H2) the stored table %s: %d different columns opened in these files; its rows\' %d random entries hide %d' % (roots[root], len(xs), shape[2], hides))
        if len(xs) > hides: J.cries['(H2) the stored table %s is opened at more columns than its random entries hide' % roots[root]] += 1
        cf = W['at']('isk/anchor/count.bin' if roots[root] == 'key' else 'isk/leaf/%s/count.bin' % roots[root])
        if os.path.exists(cf):
            n = os.path.getsize(cf)
            # The count is the length of count.bin (prover/common/cp_files.h); an opening that was
            # begun and not finished is counted too.
            say('     counted in its count.bin: %d (%s the %d different columns seen)' % (n, 'at least' if n >= len(xs) else 'BELOW', len(xs)))
            if n < len(xs): J.cries['(H2) the stored table %s: more columns are opened in these files than its lasting count has counted' % roots[root]] += 1
            # Under --commitment no proof may have been made before the audit's YES: the count must
            # equal what the key file and the leaf files drew.
            if kept and n != drawn_cols[root]: J.cries['(H2) the stored table %s: its count.bin has counted %d columns, the key file and the entries draw %d: a prover has run on this folder, or an opening was lost' % (roots[root], n, drawn_cols[root])] += 1
        elif kept: J.cries['(H2) the stored table %s has no count.bin' % roots[root]] += 1
    if flipped:
        held = [r for r in flipped if r.level >= 1]; quiet = [r for r in held if not r.bad]; ids = {id(r) for r in flipped}
        others = [r for r in C.recs if r.bad and id(r) not in ids]
        say('FLIP: of %d records with a bit turned %d are held at a level above 0: %d of these cried, %d stayed quiet; records without a turned bit that cried: %d' % (len(flipped), len(held), len(held) - len(quiet), len(quiet), len(others)))
        for r in quiet[:20]: say('FLIP: QUIET: %s | %s | %s%s' % (r.file, r.chain, r.cls, '/' + r.sub if r.sub else ''))
        for k_, n in collections.Counter((r.file, r.chain, r.cls, r.sub) for r in others).most_common(12): say('FLIP: cried without a turned bit: %d x %s' % (n, k_))
        say('FLIP: %s' % ('EVERY HELD CLASS IS ALIVE, AND NOTHING ELSE MOVED' if not quiet and not others else 'FAILED: %d checked records stayed quiet, %d others cried' % (len(quiet), len(others))))
    return ['%d x %s' % (n, w) for w, n in sorted(J.cries.items())]
