#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Level 0 of the audit (tools/audit/README.md): parses a key, leaf (`entry`) or proof file.

Every byte is recorded (`Rec`) under one class, every hash chain (Fiat-Shamir transcript) is
replayed, and every opening is hashed to its root; a file that does not fit raises an exception,
mostly by `assert`: Python must run without -O (tools/audit/audit.py rejects -O).
Beyond that, none of the protocol's checks is made: run the verifier `cp_verify` too.
Numbers such as R7.12 are rules of docs/format/FORMAT.md. The hash, chains, leaves,
Merkle paths and the item `request` (R15.x) come from docs/format/format_rules.py (`R`).
lay() and the lists of a proof file's messages with roots follow docs/format/read_proof.py.
Public data from tools/audit/pubtool.cpp, which runs the verifier's configuration code, is taken as
given: the parameter bytes of the seven first queries (a chain's first hash input), and each chain's
plan (its rounds and messages, with their names). The replay checks the bytes only indirectly: under
other parameters the first proof of work or the first opening fails. The names (`what`, `where`)
are not checked: `what` chooses a record's class."""
import struct

# Replaces a code's number where none applies (R12.2).
NONE = (1 << 32) - 1
# KIND: a round's name -> its kind (R5.2)
ONE = ('BAT', 'BAT-sigma', 'PSC round', 'PSC Y', 'ZERO', 'LINE', 'LKP shift', 'JOIN', 'LKP', 'FOLD round', 'BASE lambda', 'LIN point', 'LIN zeta', 'LINK point', 'LINK phi')  # R14.11
KIND = dict([(n, 1) for n in ONE] + [('PSC epsilon', 2), ('FOLD epsilon', 2), ('SEND seed', 3), ('LOC', 4), ('CS', 4), ('BASE', 4), ('LINK', 4), ('RNG', 5), ('SEAL', 6), ('SEND answer', 7), ('SPLIT', 7),
                                     ('PSC values', 7), ('REPLY', 7), ('LAST', 7)])
# the plan's name of a piece of a prover's message -> the class
PIECE = {'root': 'root', 'sample answers': 'sample-answer', 'hiding sum': 'hiding-sum', 'round polynomial in Y': 'sumcheck-coefficient', 'values at the last point': 'end-value', 'split target': 'split-value',
         'masked target': 'base-target', 'masked message and randomness': 'base-word', 'plain scalars': 'reply'}


# Scalars that sub-proofs send: the names given to plan_scalars() in prover/prover/plan_rounds.h,
# prover/attn/plan_rounds.h, prover/head/plan_rounds.h and prover/hiding/token_plan.h.
# An unknown name gets the class `unknown-scalar`, which tools/audit/audit.py reports: a new kind
# of published value.
SCALARS = ('claimed sum', 'token sum', 'sums of limb products', 'row products', "the embedding's value", 'plane values', "plane values and the scores' sum", 'sums, index bits, remainders', "the value product's two sums", 'the maxima at the point')


class Part:
    """One oracle message: main and mask rows in one Merkle tree. n: R12.3's seven words.

    deg: 1 for main rows over F, 3 over E. cls: leaf class."""

    def __init__(s, name, n, kind):
        s.name, s.code, s.mask_code, s.main_rows, s.deg, s.mask_rows, s.answers, s.cls = (name,) + tuple(n)
        s.kind = kind  # what its rows are made of, by the protocol: stored | link | kept | carrying | stage | fold-mask | masking-word | rider | split-mask | multiplicities | folded-row | Omega | W
        s.order = None


# number, start, leafno: inputs to its leaves' hashes (R7.13, R3.1); leafno 174: a stored tree.
class Tree:
    def __init__(s, root, number, start, who, depth, parts, leafno=164, salt=None, sent=None):
        s.root, s.number, s.start, s.who, s.depth, s.parts, s.leafno, s.salt, s.sent = root, number, start, who, depth, parts, leafno, salt, sent
        s.width = sum(p.main_rows * p.deg for p in parts) + sum(3 * p.mask_rows for p in parts)
        s.lists = []  # every list at which it was opened: (file, chain, round, the different positions, how many were drawn)
        s.uid = None


# x: a position, a given digest's (depth, index), a binding sibling's number, or a record answer's
# (row, index). level1.py sets level and bad.
class Rec:
    __slots__ = ('file', 'off', 'n', 'cls', 'sub', 'chain', 'rnd', 'rname', 'where', 'tree', 'part', 'x', 'level', 'bad', 'why', 'link')

    def __init__(s, **k):
        s.sub = s.chain = s.rnd = s.rname = s.where = s.tree = s.part = s.x = s.why = s.link = None
        s.level, s.bad = 0, 0
        for a, b in k.items():
            setattr(s, a, b)


def words_of(b):
    return list(struct.unpack('<%dQ' % (len(b) // 8), b))


class Take:
    def __init__(s, w): s.w, s.p = w, 0
    def one(s):
        s.p += 1; return s.w[s.p - 1]
    def some(s, n): return tuple(s.one() for _ in range(n))
    def lst(s, f): return [f() for _ in range(s.one())]
    def done(s): return s.p == len(s.w)


def up2(r): return 0 if r == 0 else 1 << (r - 1).bit_length()
def log_up(r): return 0 if r <= 1 else (r - 1).bit_length()


# stage: base code, mask code, k, h, for 2^k main rows and 2^h mask rows (R12.4).
def fold_mask(stage, name, cls=0):  # R12.10
    return Part('fold mask of ' + name, (NONE, stage[1], 0, 1, 1 << stage[3], 0, cls), 'fold-mask')


def masking_word(stage, name):
    return Part('masking word of ' + name, (stage[0], stage[1], 1, 3, 1, 0, 0), 'masking-word')


def test_messages(stages, first_own, ride, name, s):
    """R13.19, R12.10: the prover's messages of a proximity test that contain roots, in order, each
    a list of Parts.

    first_own: the first fold mask is a message of its own. ride: later fold masks and the masking
    word go with their stage's rows."""
    out = []
    for i, st in enumerate(stages[:-1]):
        if (first_own if i == 0 else not ride): out.append([fold_mask(st, '%s, stage %d' % (name, i + 1))])
        n = stages[i + 1]; last = i + 2 == len(stages)
        m = [Part('%s, rows of stage %d' % (name, i + 2), (n[0], n[1], 1 << n[2], 3, 1 << n[3], s, 0), 'stage')]
        if ride: m.append(masking_word(n, name) if last else fold_mask(n, '%s, stage %d' % (name, i + 2)))
        out.append(m)
    if not ride: out.append([masking_word(stages[-1], name)])
    return out


def lay(codes, parts):
    """R7.12, R7.1: the trees of one prover's message, each as (depth, its parts in the order of
    sending)."""
    base, mask = codes
    for i, p in enumerate(parts): p.order = i
    trees = []  # [class, log2 of the main rows' length or None, parts]
    for p in parts:
        if p.main_rows:
            t = next((t for t in trees if t[0] == p.cls and t[1] == base[p.code][0]), None)
            if t is None: t = [p.cls, base[p.code][0], []]; trees.append(t)
            t[2].append(p)
    for p in parts:
        if not p.main_rows:
            t = next((t for t in trees if t[0] == p.cls), None)
            if t is None: t = [p.cls, None, []]; trees.append(t)
            t[2].append(p)
    return [(t[1] if t[1] is not None else mask[t[2][0].mask_code][0], sorted(t[2], key=lambda p: p.order)) for t in trees]


# A segment's message (in `links`): main rows, deg, mask rows, leaf class, commit point (`moment`),
# life (R12.12).
def shape_of_request(b):  # R12.14
    t = Take(words_of(b)); code, mask_code = t.some(2)
    links = t.lst(lambda: (t.one(), t.lst(lambda: t.some(6))))
    o = dict(code=code, mask=mask_code, links=links, riders=t.lst(lambda: t.some(7)), s=t.one(), later=t.lst(lambda: t.some(4)), ride=t.one())
    assert t.done(), "request's shape: words are left"
    return o


def table_test(b):  # R13.13
    t = Take(words_of(b)); o = {}
    o['table'] = t.some(6); o['u'] = t.some(6); o['u high'] = t.one(); o['u class'] = t.one(); o['columns'], o['k_lo'], o['matrix rows'] = t.some(3); o['omega'] = t.some(6); o['w'] = t.some(6); o['omega high'], o['w high'] = t.some(2)
    o['shared split'] = t.some(6); o['own split'] = t.some(6); o['classes'] = t.some(4); o['mask codes'] = t.some(2); o['ahead'] = t.one()
    o['tests'] = [t.lst(lambda: t.some(4)) for _ in range(4)]; o['ride'] = t.one()
    assert t.done(), 'the test on the table: words are left'
    return o


def first_chain_messages(q, T):
    """R14.4 with R12.16 to R12.18: the first chain's messages with roots (chain()'s `due`)."""
    L, out, groups = len(q['links']), [], []
    for m in range(1, L + 2):
        own = list(enumerate(q['links'][m - 1][1])) if m <= L else []
        kept_here = [(j, i, x) for j in range(1, m) for i, x in enumerate(q['links'][j - 1][1]) if x[5] > 0 and j + x[5] >= m]
        main = (2 if m > 1 else 0) + sum(up2(x[0]) for j in range(1, m) for x in q['links'][j - 1][1] if x[5] > 0 and j + x[5] == m) + sum(up2(x[0]) for _, x in own if x[5] == 0)
        masks = sum(up2(x[2]) for _, x in own)
        h = log_up(masks); k = max(log_up(main), h, 1)
        groups.append((q['code'], q['mask'], k, h))
        note = lambda i, x: Part('link %d, message %d (moment %d)' % (m, i, x[4]), (q['code'] if x[0] else NONE, q['mask'] if x[2] else NONE, x[0], x[1], x[2], q['s'], x[3]), 'link')
        first = ([Part('carrying message into group %d' % m, (q['code'], NONE, 2, 3, 0, q['s'], 0), 'carrying')] if m > 1 else [])
        first += [Part('link %d, message %d, sent anew in group %d' % (j, i, m), (q['code'], NONE, x[0], x[1], 0, q['s'], x[3]), 'kept') for j, i, x in kept_here]
        first += [note(i, x) for i, x in own if x[4] == 0] + [fold_mask(groups[-1], 'group %d' % m)] + ([Part('rider: the index-mask message', n, 'rider') for n in q['riders']] if m == 1 else [])
        out.append(first)
        for t in range(1, 1 + max([x[4] for _, x in own], default=0)): out.append([note(i, x) for i, x in own if x[4] == t])
    out += test_messages([groups[-1]] + q['later'], False, q['ride'], "the request's last test", q['s'])
    out.append(w_message(T))
    return out


def w_message(T):
    return [Part('W', T['w'] + (T['classes'][2],), 'W')] + ([fold_mask(T['tests'][2][0], "the test on a request's own rows, stage 1", T['classes'][2])] if T['ahead'] else [])


def shared_chain_messages(T, s):
    split = T['u'][0] != T['omega'][0]
    out = [[Part('the folded row', T['u'] + (T['u class'],), 'folded-row')] + ([fold_mask(T['tests'][0][0], 'the test on the folded row, stage 1', T['u class'])] if T['ahead'] else [])]
    out.append([Part('Omega', T['omega'] + (T['classes'][0],), 'Omega')] + ([Part("the shared split's mask message", T['shared split'] + (T['classes'][1],), 'split-mask')] if split else []))
    out += test_messages(T['tests'][0], not T['ahead'], T['ride'], 'the test on the folded row', s)
    if split: out += test_messages(T['tests'][1], True, T['ride'], 'the test on Omega', s)
    return out


def ranges_item(b):  # R13.14
    t = Take(words_of(b))
    def run(): return t.some(4)
    def part(): return (t.one(), t.lst(run))
    def lookup(): return dict(d=t.one(), mu=t.one(), live=t.one(), weights=t.lst(t.one), offset=t.one(), padding=t.one(), parts=t.lst(part), bounds=t.lst(t.one))
    o = dict(lookups=t.lst(lookup), notice=t.some(7), number=t.one(), stages=t.lst(lambda: t.some(4)), ride=t.one())
    assert t.done(), 'ranges: words are left'
    return o


def one_time_messages(index_mask, G, T, s):  # R13.18, R13.19
    out = [[Part('the index-mask message', index_mask, 'rider')]]
    out.append([Part('the multiplicities', G['notice'], 'multiplicities')] + ([fold_mask(G['stages'][0], 'the test of ranges, stage 1', G['notice'][6])] if G['ride'] else []))
    out += test_messages(G['stages'], not G['ride'], G['ride'], 'the test of ranges', s)
    out.append(w_message(T))
    return out


class Cutter:
    def __init__(s, R, field, conf, pub):
        """conf: configuration digest. pub(name): a file's bytes, in pubtool's output directory."""
        s.R, s.field, s.conf, s.pub = R, field, conf, pub
        s.version = {1: b'H811', 2: b'H821'}[field]
        s.recs, s.trees, s.log = [], [], []
        s.msgs = {}  # (file, chain title) -> its messages with roots, in order: (round, [Parts])
        s.runs = {}  # (file, chain title) -> dict(eight, entries, names, codes)
        s.to_answer, s.last_word = [], None
        s.draws = {}  # (file, chain title, round number) -> what the verifier drew: for the levels above

    # ---- the file being parsed ----
    def open(s, name, data):
        s.name, s.f, s.at = name, data, 0

    def take(s, n, cls, **k):
        assert s.at + n <= len(s.f), '%s ends early at byte %d (%s)' % (s.name, s.at, cls)
        b = s.f[s.at:s.at + n]
        if n: s.recs.append(Rec(file=s.name, off=s.at, n=n, cls=cls, **k))
        s.at += n
        return b

    def close(s):
        assert s.at == len(s.f), '%s: %d bytes are left' % (s.name, len(s.f) - s.at)

    # R4.3: the plan's eight words, its entries, the items after `configuration`.
    def parse(s, par):
        R = s.R
        pos = len(R.item("the format's version", s.version + bytes(4)))
        eight = struct.unpack('<8Q', par[pos:pos + 64]); pos += 64
        entries = []
        for _ in range(eight[7]):
            n = struct.unpack('<Q', par[pos:pos + 8])[0]
            entries.append((par[pos + 8:pos + 8 + n].decode('ascii'),) + struct.unpack('<11Q', par[pos + 8 + n:pos + 96 + n])); pos += 96 + n
        head = R.parameters_head(s.field, R.plan_bytes(s.field, *eight[2:7], entries)) + R.item('configuration', s.conf)
        assert par.startswith(head), 'the head of the parameters, spelled anew'
        names = {}; p = len(head)
        while p < len(par):
            n = struct.unpack('<Q', par[p:p + 8])[0]; name = par[p + 8:p + 8 + n].decode(); p += 8 + n
            m = struct.unpack('<Q', par[p:p + 8])[0]; names[name] = par[p + 8:p + 8 + m]; p += 8 + m
        return eight, entries, names

    def pieces(s, stem):
        """The plan's lists: for each round (name, where, [(what, where, bytes) of the pieces of its
        prover's message]).

        Other columns, from 0. rounds.tsv: 3 count, 5 message bytes. messages.tsv: 4 count,
        5 absorbed by the chain."""
        rounds = [l.split('\t') for l in s.pub(stem + '.rounds.tsv').decode().split('\n') if l]
        msgs = [l.split('\t') for l in s.pub(stem + '.messages.tsv').decode().split('\n') if l]
        inch = [(m[0], m[1], int(m[3])) for m in msgs if m[5] == '1']
        assert all(m[4] == '1' for m in msgs) and all(r[3] == '1' for r in rounds), 'a plan with repeats'
        out, at = [], 0
        for r in rounds:
            need, got = int(r[5]), []
            while need:
                assert at < len(inch) and inch[at][2] <= need, 'the plan\'s messages do not fill round %s (%s)' % (r[0], r[1])
                got.append(inch[at]); need -= inch[at][2]; at += 1
            out.append((r[0], r[1], got))
        assert at == len(inch), "pieces of the plan's messages are left"
        return out

    def open_tree(s, t, k, draws, ctx):
        """R7.8, R7.15: reads tree t's opening at the current offset, or returns 0 and restores it.

        In a carrying message (the code switch, blocks/iop/cs.h) row 0 takes the folded main row's
        message, row 1 the rest: private randomness and challenges only."""
        R, keep, nrec = s.R, s.at, len(s.recs)
        xs = sorted({R.short_position(x, t.depth) for x in draws})
        try:
            leaves = []
            for x in xs:
                col = []
                for p in t.parts:
                    if p.main_rows and p.kind == 'carrying':
                        col += struct.unpack('<3Q', s.take(24, 'main-symbol', sub='carrying', tree=t, part=p, x=x, **ctx)) + struct.unpack('<3Q', s.take(24, 'main-symbol', sub='carrying-rest', tree=t, part=p, x=x, **ctx))
                    elif p.main_rows:
                        cls = {'masking-word': 'masking-word-symbol', 'stored': 'stored-symbol'}.get(p.kind, 'main-symbol')
                        col += struct.unpack('<%dQ' % (p.main_rows * p.deg), s.take(8 * p.main_rows * p.deg, cls, sub=p.kind, tree=t, part=p, x=x, **ctx))
                for p in t.parts:
                    if p.mask_rows: col += struct.unpack('<%dQ' % (3 * p.mask_rows), s.take(24 * p.mask_rows, 'mask-symbol', sub=p.kind, tree=t, part=p, x=x, **ctx))
                if any(y >= R.PRIME[s.field] for y in col): raise AssertionError('a symbol that is no element')
                salt = s.take(t.salt, 'stored-salt' if t.leafno == 174 else 'salt', tree=t, x=x, **ctx)
                leaves.append(R.leaf(t.leafno, t.number, t.depth, x, t.start, col, salt))
            if R.walk(t.depth, s.conf, xs, leaves, lambda d, i: s.take(32, 'stored-given-digest' if t.leafno == 174 else 'given-digest', tree=t, x=(d, i), **ctx)) != t.root: raise AssertionError('another root')
        except AssertionError:
            s.at = keep; del s.recs[nrec:]
            return 0
        t.lists.append((s.name, ctx['chain'], ctx['rnd'], xs, len(draws)))
        return len(xs)

    def chain(s, title, stem, label, statement, random_bytes, book, stored, due, codes, split_draws=None, record=None):
        """Replays one hash chain from the current offset.

        stored: the trees of the weight tables ("stored table" in strings) that LOC or LINK opens,
        in order. due: the messages that contain roots, each a list of Parts.

        random_bytes: None if the file has them (R4.5). book: the trees that the chain can read
        (R7.13); it grows. split_draws: R7.19. record: the entry RECORD with which the plan must
        begin (record_entry()); None if it must have none. Which trees of `book` a round of
        positions opens is found by trial; one opening is enough, so a missing further opening goes
        unnoticed. Returns (seal, run's value)."""
        R = s.R
        par = s.pub(stem + '.parameters')
        eight, entries, names = s.parse(par); pd = R.parameters_digest(par)
        pieces = s.pieces(stem)
        due = list(due)
        sent = s.msgs[(s.name, title)] = []
        salt_bytes, nonce_bytes = eight[2], eight[3]
        assert eight[4] == 0, 'the plan lists positions: not written for'
        rb = s.take(32, 'chain-bytes', chain=title) if random_bytes is None else random_bytes
        run = R.run_value(label, pd, rb)
        c = R.Chain(R.first_value(label, pd, statement, run), s.field, label)
        seal, number = None, 0
        # R13.15: the plan of a
        # chain that follows a table's root in its file (a key file's one-time chain, a leaf file's
        # link chain) begins with one entry RECORD. It is not a round, so the replay skips it.
        live = [e for e in entries if e[0] != 'RECORD']
        assert [e for e in entries if e[0] == 'RECORD'] == ([record] if record else []) and (not record or entries[0] == record), "the plan's entry RECORD is not what R13.15 says for this chain"
        pieces = [p for p in pieces if p[0] != 'RECORD']
        s.runs[(s.name, title)] = dict(eight=eight, entries=live, names=names, codes=codes)  # the chain's rounds, by their numbers: without the entry RECORD
        assert [e[0] for e in live] == [p[0] for p in pieces], "the plan's list of rounds and the first query's differ"
        for e, (pname, where, cut) in zip(live, pieces):
            name, count, w, mb, nroots, answers, nE, nz, npos, k, ntrits, rows = e
            assert count == 1
            number += 1
            ctx = dict(chain=title, rnd=number, rname=name, where=where)
            start = s.at
            got_roots = []
            for what, wh, n in cut:
                cls = PIECE.get(what)
                if what == 'round polynomial': cls = 'fold-coefficient' if name == 'FOLD round' else 'sumcheck-coefficient'
                if cls is None: cls = 'subproof-scalar' if what in SCALARS else 'unknown-scalar'
                if cls == 'hiding-sum': what = 'FOLD' if name == 'FOLD epsilon' else 'PSC'
                if cls == 'base-word':  # R12.9: the masked main message; the masked mask message; the masked main tail; the masked mask tail
                    lw = s.last_word; cb, cm = codes[0][lw.code], codes[1][lw.mask_code]
                    assert n == 24 * (cb[1] + cm[1] + cb[2] + cm[2]), "BASE's message has not the last stage's lengths"
                    for sub, m in (('main-message', cb[1]), ('mask-message', cm[1]), ('main-tail', cb[2]), ('mask-tail', cm[2])): s.take(24 * m, cls, sub=sub, part=lw, **ctx)
                    continue
                if cls == 'sample-answer':  # R8.3: message by message in the order sent
                    assert n == 24 * sum(p.answers for p in s.to_answer), 'the answers are not those that the seeds ask for'
                    for p in s.to_answer: s.take(24 * p.answers, cls, sub=p.kind, part=p, **ctx)
                    s.to_answer = []
                    continue
                b = s.take(n, cls, sub=what, **ctx)
                if cls == 'root': got_roots.append(b)
            msg = s.f[start:s.at]
            assert len(msg) == mb and len(got_roots) == nroots
            assert msg[:32 * nroots] == b''.join(got_roots), 'R8.6, R13.11: the roots stand first'
            if nroots:
                assert due, 'round %d %s holds a root, and by the rules no message with a root is due' % (number, name)
                sent.append((number, due[0]))
                if KIND[name] == 3: s.to_answer = [p for p in due[0] if p.answers]
                for p in due[0]:
                    if p.kind == 'masking-word': s.last_word = p
                laid = lay(codes, due.pop(0))
                assert len(laid) == nroots, 'round %d %s (%s) holds %d roots; by R7.12 its oracle messages make %d trees' % (number, name, where, nroots, len(laid))
                for j, (depth, parts) in enumerate(laid):
                    t = Tree(got_roots[j], len(book), run, title, depth, parts, 164, salt_bytes, (s.name, title, number))
                    t.uid = len(s.trees); s.trees.append(t); book.append(t)
                    for p in parts: p.tree = t
            nonce = int.from_bytes(s.take(nonce_bytes, 'nonce', **ctx), 'little') if w else 0
            assert c.query(msg, KIND[name], w, nonce), '%s, %s: work in round %d %s' % (s.name, title, number, name)
            drew = None
            if KIND[name] == 3:
                v = nE // nz - 1
                drew = [([c.element() for _ in range(v)], c.element(True)) for _ in range(nz)]
            elif KIND[name] == 4:
                draws = [c.position(k) for _ in range(npos)]; drew = (draws, [c.element() for _ in range(nE)])
                got = []
                if name == 'LINK':
                    n0 = split_draws[0]; assert npos == sum(split_draws)
                    for t, part in zip(stored, (draws[:n0], draws[n0:])):
                        assert s.open_tree(t, k, part, ctx), 'a stored table'; got.append(t)
                else:
                    if name == 'LOC' and stored:
                        for t in stored:
                            assert s.open_tree(t, k, draws, ctx), 'the stored table'; got.append(t)
                    for t in book:
                        if t.depth <= k and s.open_tree(t, k, draws, ctx): got.append(t)
                assert got, 'round %d %s: nothing was opened' % (number, name)
                s.log.append('%s | %s | round %d %s (%s): %d draws in [0, 2^%d): %s' % (s.name, title, number, name, where, npos, k, '; '.join('tree %d of %s [%s] %d columns' % (t.number, t.who, ' + '.join(p.name for p in t.parts), len(t.lists[-1][3])) for t in got)))
            elif KIND[name] == 5:
                drew = c.trits(ntrits)
            elif KIND[name] == 6:
                seal = c.seal_digest()
            else:
                drew = [c.element(nz == nE and nz > 0) for _ in range(nE)]
            s.draws[(s.name, title, number)] = (name, where, drew)
        assert not due, 'by the rules %d more messages with roots were due' % len(due)
        return seal, run

    # ---- a stored table's record ----
    # shape (R4.10): log2 of a row's length, message entries, tail entries, log2 of the row count,
    # stored answers for each row, salt bytes.
    @staticmethod
    def record_entry(shape):
        """R13.15: the plan's entry RECORD for a table of this shape."""
        s_, c = shape[4], log_up(shape[1])
        return ('RECORD', 1, 0, 0, 0, 0, s_ * (c + 1), s_, 0, 0, 0, 0)

    def answers(s, shape, who):
        """R13.2, R13.3, R13.8: reads the answers of a table's record, which follow its root: by
        row, within a row by answer, 24 bytes each.

        They stay at level 0: an answer is a weighted sum over the row's real entries plus a public
        factor beta times its pad (a private random value), so without the weights nothing can be
        compared with it: a committer that leaves the pads out gets the audit's YES. Details:
        tools/audit/README.md, "What a YES does not cover". Returns their bytes."""
        return b''.join(s.take(24, 'record-answer', sub=who, x=(r, i)) for r in range(1 << shape[3]) for i in range(shape[4]))

    def table_item(s, name, root, answers):
        """R4.6: a table's item in a chain's statement: its root, then the list of its record's
        answers."""
        return s.R.item(name, s.R.string(root) + s.R.u64(len(answers) // 24) + answers)

    # ---- the three kinds of file ----
    def stored_tree(s, root, shape, who):
        p = Part('the stored table ' + who, (NONE, NONE, 1 << shape[3], 1, 0, 0, 0), 'stored')
        t = Tree(root, 0, s.conf, who, shape[0], [p], 174, shape[5]); t.uid = len(s.trees); s.trees.append(t)
        return t

    def codes_of(s, b):
        tc = Take(words_of(b)); c = (tc.lst(lambda: tc.some(3)), tc.lst(lambda: tc.some(3)))
        assert tc.done()
        return c

    # R13.2
    def key(s, name, data):
        R = s.R
        s.open(name, data)
        assert s.take(8, 'head') == b'CPIK' + s.version, 'the head'
        eight, _, names = s.parse(s.pub('key.one-time.parameters'))
        T = table_test(names['the test on the table']); shape = T['table']; samples = eight[5]
        root = s.take(32, 'stored-root'); answers = s.answers(shape, 'the first table')
        table = s.stored_tree(root, shape, 'of the key'); codes = s.codes_of(names['codes'])
        st_table = s.table_item('table', root, answers)
        G = ranges_item(names['ranges']); im = struct.unpack('<7Q', names['index-mask message'])
        depth, leaf_salt = struct.unpack('<2Q', names['binding tree']); assert depth == 0
        A = []
        seal1, run1 = s.chain('one-time', 'key.one-time', 0x46524d02, st_table, None, A, None, one_time_messages(im, G, T, samples), codes, record=s.record_entry(shape))
        assert struct.unpack('<Q', s.take(8, 'place-word'))[0] == 0
        salt = s.take(leaf_salt, 'binding-salt')
        broot = R.binding_leaf(0, 0, s.conf, seal1, salt)
        B = []
        seal2, run2 = s.chain('shared', 'key.shared', 0x4c494e01, st_table + R.item("the binding tree's root", broot), broot, B, [table], shared_chain_messages(T, samples), codes)
        omega = [t for t in B if any(p.kind == 'Omega' for p in t.parts)]; assert len(omega) == 1  # R13.7, R14.10
        st3 = R.item('the seal of an earlier chain', seal1) + R.item('the seal of an earlier chain', seal2)
        s.chain('second', 'key.second', 0x4c494e02, st3, None, A + omega, None, test_messages(T['tests'][2], not T['ahead'], T['ride'], "the test on a request's own rows", samples), codes)
        s.close()
        return dict(root=root, answers=answers, shape=shape, T=T, G=G, codes=codes, table=table, names=names)

    # R13.3, R13.9, R13.10. key_root, key_answers: `root` and `answers` of key()'s result.
    def entry(s, name, data, key_root, k, key_answers):
        R = s.R
        s.open(name, data)
        assert s.take(8, 'head') == b'CPEN' + s.version, 'the head'
        eight, _, names = s.parse(s.pub('entry.parameters')); samples = eight[5]
        t = Take(words_of(names['the link'])); shape = t.some(6); nu0, nuj, code, mcode, Rr, high, cls, ahead = t.some(8); later = t.lst(lambda: t.some(4)); ride = t.one(); assert t.done()
        root = s.take(32, 'stored-root'); answers = s.answers(shape, 'copy %d' % k)
        codes = s.codes_of(names['codes'])
        first, further = s.stored_tree(key_root, shape, 'of the key (read in entry %d)' % k), s.stored_tree(root, shape, 'copy %d' % k)
        st0 = (code, mcode, 1 + log_up(Rr), 0)
        m0 = [Part("the first table's folded row", (code, NONE, Rr, 3, 0, samples, cls), 'folded-row'), Part("the copy's folded row", (code, NONE, Rr, 3, 0, samples, cls), 'folded-row')]
        if ahead: m0.append(fold_mask(st0, 'the test on the two folded rows, stage 1', cls))
        due = [m0] + test_messages([st0] + later, not ahead, ride, 'the test on the two folded rows', samples)
        s.chain('link', 'entry', 0x4c4e4b01, s.table_item('the first table', key_root, key_answers) + s.table_item('the further table', root, answers), None, [], [first, further], due, codes, (nu0, nuj), record=s.record_entry(shape))
        s.close()
        return dict(root=root, answers=answers, shape=shape, table=further, first=first, names=names, codes=codes, link=dict(nu0=nu0, nuj=nuj, code=code, mcode=mcode, R=Rr, high=high, cls=cls, ahead=ahead, later=later, ride=ride))

    def proof(s, name, data, roots, policy, kl, request, answers):
        """R14.1. roots: a leaf table's number (`copy`) -> its root. answers: that number -> the
        bytes of its record's answers, as its leaf file has them (R14.3)."""
        R = s.R
        s.open(name, data)
        assert s.take(8, 'head') == b'CPPF' + s.version, 'the head'
        copy = struct.unpack('<I', s.take(4, 'head/copy-number'))[0]
        eight, _, names = s.parse(s.pub('proof.first.parameters')); samples = eight[5]
        T = table_test(names['the test on the table']); shape = T['table']
        Q = shape_of_request(names["request's shape"]); codes = s.codes_of(names['codes'])
        root = roots[copy]
        table = s.stored_tree(root, shape, 'copy %d (read in %s)' % (copy, name))
        st_table = s.table_item('table', root, answers[copy])
        depth, leaf_salt = struct.unpack('<2Q', names['binding tree'])
        st1 = st_table + R.item('stored copy', R.u64(copy)) + R.item('request', R.string(R.statement_words(policy, kl, request))) + R.item("the request file's words", R.string(request))
        A = []
        seal1, run1 = s.chain('first', 'proof.first', 0x46524d01, st1, None, A, None, first_chain_messages(Q, T), codes)
        place = struct.unpack('<Q', s.take(8, 'place-word'))[0]; assert place >> depth == 0
        salt = s.take(leaf_salt, 'binding-salt')
        sib = [s.take(32, 'binding-sibling', x=i) for i in range(depth)]
        node = R.binding_root(depth, place, s.conf, seal1, salt, sib)
        B = []
        seal2, run2 = s.chain('shared', 'proof.shared', 0x46524d03, st_table + R.item("the binding tree's root", node), node, B, [table], shared_chain_messages(T, samples), codes)
        omega = [t for t in B if any(p.kind == 'Omega' for p in t.parts)]; assert len(omega) == 1
        st3 = R.item('the seal of an earlier chain', seal1) + R.item('the seal of an earlier chain', seal2)
        s.chain('second', 'proof.second', 0x46524d04, st3, None, A + omega, None, test_messages(T['tests'][2], not T['ahead'], T['ride'], "the test on a request's own rows", samples), codes)
        s.close()
        return dict(copy=copy, place=place, depth=depth, seal1=seal1, salt=salt, siblings=sib, binding_root=node, T=T, Q=Q, codes=codes, table=table, names=names, A=A, B=B)
