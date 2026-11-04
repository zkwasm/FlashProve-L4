#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Turns a key file, a leaf file and a proof file into parsed files (PARSED-FORM.md).

usage: parse_files.py TABLES_DIR CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE PROOF_FILE OUT_DIR [--lines]
  TABLES_DIR      the output directory of tools/audit/pubtool, run on this CONFIG_DIR
  CONFIG_DIR      policy.txt, layout.txt, KL.bin
  PUBLIC_KEY_DIR  key.bin and leaf/<k>.bin
  OUT_DIR         an existing, empty directory that only you can write in (mkdir -m 700)
It parses key.bin, the proof file, and the one leaf file that the proof file's head names, and writes
key.parsed.bin, leaf-<k>.parsed.bin and proof.parsed.bin, each with two small files beside it
(.order, .seeds; for the leaf file also .seeds0). With --lines it also writes every item of each
file as one line of JSON (key.lines.jsonl and so on), for people. PARSED-FORM.md describes all of them.

This parser is untrusted: it counts as the prover's last step. README.md says what that means, what
must not pass through it, and why it has to replay the hash chains. It makes only the checks that it
needs to cut the files (PARSED-FORM.md, section 7). That it parses a file says nothing about whether
the file proves anything.

It imports, unchanged, tools/audit/cut.py (class Cutter: records every byte of a file under one
class) and docs/format/format_rules.py (hash, chains, leaves, paths) from the repository that it is
in. Numbers such as R7.9 are rules of docs/format/FORMAT.md.

Exit code 0 and a last line "PARSED: 3 files". A key file, leaf file or proof file that does not fit
or cannot be read: exit code 1 and a last line that begins "REFUSED: ". A wrong call, among them one
of the user's own files that cannot be read and an OUT_DIR at or below a place that another account
could change: exit code 2. A fault of this script: exit code 3. The files are written last, when all
three have been parsed, so nothing is written before that; only a signal or a fault during the
writing itself, such as a full disk, can leave some of them."""
import hashlib
import json
import os
import re
import stat
import struct
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'docs', 'format'), os.path.join(HERE, '..', 'audit')]
import cut  # noqa: E402
import format_rules  # noqa: E402

# A chain's label (R13.4, R14.3, R14.9, R14.10) -> the name that cut.py gives the chain. A proof
# file's shared and second chains have labels of their own, or the key file's: both are listed.
TITLE = {0x46524d02: 'one-time', 0x4c494e01: 'shared', 0x4c494e02: 'second', 0x4c4e4b01: 'link', 0x46524d01: 'first', 0x46524d03: 'shared', 0x46524d04: 'second'}


CAP = 1 << 28  # No file is read beyond this many bytes. A full-size proof file has about 19 MB.
NO_MEMORY = 'REFUSED: the files need more memory than this machine gives'


class Refused(Exception):
    """A file does not fit. Anything else that goes wrong after the cutting is a fault of this script."""


class WrongCall(Exception):
    """One of the user's own files (a public file, a table, the request file) cannot be read."""


def ask(ok, why):
    if not ok: raise Refused(why)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_bytes(path):
    """The bytes of a regular file of at most CAP bytes. Its kind and length are asked before it is
    read: a FIFO would block, and a link to /dev/zero would never end."""
    with open(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as f:
        about = os.fstat(f.fileno())
        ask(stat.S_ISREG(about.st_mode), 'it is not a regular file')
        ask(about.st_size <= CAP, 'it has more than the %d bytes that this parser reads' % CAP)
        b = f.read(about.st_size + 1)
    ask(len(b) == about.st_size, 'its length changed while it was read')
    return b


class Forget(dict):
    """Takes the place of the cutter's table of all drawn values, which this parser does not read."""

    def __setitem__(self, key, value):
        pass


class Rules:
    """format_rules.py as the cutter sees it. It notes each value that comes out of the hash.

    A note is (how many records the cutter had at that moment, the note itself). keep_draws: also
    note what each round draws (for --lines)."""

    def __init__(self, keep_draws):
        self.notes, self.cutter = [], None
        rules = self

        class Chain(format_rules.Chain):
            def query(self, message, kind, w=0, nonce=0):
                ok = format_rules.Chain.query(self, message, kind, w, nonce)
                if ok:
                    self.now = dict(name='round', chain=TITLE[self.label], round=self.round, rkind=kind, work=w, value=self.rho.hex(), elements=[], draws=[], trits=[])
                    rules.note(self.now)
                return ok

            def element(self, nonzero=False):
                e = format_rules.Chain.element(self, nonzero)
                if keep_draws: self.now['elements'].append(list(e))
                return e

            def position(self, k):
                x = format_rules.Chain.position(self, k)
                if keep_draws: self.now['draws'].append(x)
                return x

            def trits(self, n):
                t = format_rules.Chain.trits(self, n)
                if keep_draws: self.now['trits'] = t
                return t

            def seal_digest(self):
                d = format_rules.Chain.seal_digest(self)
                self.now['seal'] = d.hex()
                return d
        self.Chain = Chain

    def __getattr__(self, name):
        return getattr(format_rules, name)

    def note(self, d):
        self.notes.append((len(self.cutter.recs), d))

    def run_value(self, label, parameters, random_bytes):
        v = format_rules.run_value(label, parameters, random_bytes)
        self.note(dict(name='parameters digest', chain=TITLE[label], label=label, value=parameters.hex()))
        self.note(dict(name="run's value", chain=TITLE[label], label=label, value=v.hex()))
        return v

    def first_value(self, label, parameters, statement, run, record=b''):
        v = format_rules.first_value(label, parameters, statement, run, record)
        self.note(dict(name='first value', chain=TITLE[label], label=label, value=v.hex()))
        return v

    def binding_leaf(self, depth, place, configuration, seal, salt):
        v = format_rules.binding_leaf(depth, place, configuration, seal, salt)
        self.note(dict(name='binding root', chain='slot', depth=depth, place=place, value=v.hex()))  # called for a key file only, whose binding tree is one leaf
        return v

    def binding_root(self, depth, place, configuration, seal, salt, siblings):
        v = format_rules.binding_root(depth, place, configuration, seal, salt, siblings)
        self.note(dict(name='binding root', chain='slot', depth=depth, place=place, value=v.hex()))
        return v


# ---- the cutter's records -> items, in the file's order ----
SYMBOL = ('main-symbol', 'masking-word-symbol', 'stored-symbol', 'mask-symbol')
OUTSIDE = {'head': 'head', 'head/copy-number': 'head', 'stored-root': 'table', 'record-answer': 'table', 'place-word': 'slot', 'binding-salt': 'slot', 'binding-sibling': 'slot'}
OUTSIDE_KIND = {'head/copy-number': 'u32', 'stored-root': 'root', 'place-word': 'word', 'binding-salt': 'salt', 'binding-sibling': 'digest'}


class Parsed:
    """One file as cut: its bytes, its stretch of the cutter's records, and its notes. beside: the
    SHA-256 of each other file that was read to cut it. table: the computed lines on its weight
    tables."""

    def __init__(self, kind, name, data, cutter, rules, first_rec, first_note, beside, table):
        self.kind, self.name, self.data, self.cutter, self.beside, self.table = kind, name, data, cutter, beside, table
        self.recs = range(first_rec, len(cutter.recs))
        self.notes = rules.notes[first_note:]
        self.sent = [t.uid for t in cutter.trees if t.sent and t.sent[0] == name]  # its own trees, in the order sent

    def tree(self, t, chain):
        """What an opening's items say of their tree. order: its index among all roots that the file
        sends. number, place: R7.13's two."""
        stored = int(t.leafno == 174)
        adopted = not stored and t.who == 'shared' and chain == 'second'  # R13.7, R14.10: its place follows the trees of the chain that the second chain continues
        place = sum(1 for u in self.sent if self.cutter.trees[u].who in ('one-time', 'first')) if adopted else t.number
        return dict(stored=stored, order=None if stored else self.sent.index(t.uid), place=place, number=t.number, sender=t.who, depth=t.depth, width=t.width, salt_bytes=t.salt, uid=t.uid,
                    parts=[dict(msg=p.name, sub=p.kind, main=p.main_rows, deg=p.deg, mask=p.mask_rows) for p in t.parts])

    def items(self):
        """Yields the head line, then every item of the file (len > 0; the items tile the file) and,
        at its place, every computed value (len 0). An item's bytes are under 'raw'."""
        f, recs, name = self.data, self.cutter.recs, self.name
        yield dict(form='lines-0', file=self.kind, name=name, bytes=len(f), sha256=sha(f), beside=self.beside)
        yield from self.table
        count, ni = {}, 0
        at_round, opening, tree, past_columns = None, -1, None, False

        def item(off, n, chain, rnd, rname, part, kind, **more):
            key = (chain, rnd, part, kind, more.get('open'))
            count[key] = count.get(key, -1) + 1
            d = dict(off=off, len=n, chain=chain, round=rnd, rname=rname, part=part, kind=kind, i=count[key], raw=f[off:off + n])
            if kind in ('word', 'u32', 'nonce'): d['v'] = int.from_bytes(d['raw'], 'little')
            d.update({k: v for k, v in more.items() if v is not None})
            return d

        def computed(d):
            d = dict(d)
            what, chain = d.pop('name'), d.pop('chain')
            if what != 'round':
                yield dict(len=0, kind='derived', name=what, chain=chain, **d)
                return
            base = dict(len=0, kind='derived', chain=chain, round=d['round'], rname=self.cutter.runs[(name, chain)]['entries'][d['round'] - 1][0])
            yield dict(base, name="chain's value", value=d['value'], rkind=d['rkind'], work=d['work'])
            for k in ('elements', 'draws', 'trits'):
                if d[k]: yield dict(base, name=k, v=d[k])
            if 'seal' in d: yield dict(base, name='seal', value=d['seal'])

        j = self.recs.start
        while j < self.recs.stop:
            r = recs[j]
            while ni < len(self.notes) and self.notes[ni][0] <= j:
                yield from computed(self.notes[ni][1])
                ni += 1
            more = dict(cls=r.cls, sub=r.sub, where=r.where, msg=r.part.name if r.part else None)
            if r.cls == 'head':
                yield item(r.off, 4, 'head', 0, '', '', 'magic', cls=r.cls)
                yield item(r.off + 4, 4, 'head', 0, '', '', 'version', cls=r.cls)
            elif r.cls == 'record-answer': yield item(r.off, r.n, 'table', 0, '', '', 'answer', row=r.x[0], answer=r.x[1], **more)
            elif r.cls in OUTSIDE: yield item(r.off, r.n, OUTSIDE[r.cls], 0, '', '', OUTSIDE_KIND[r.cls], **more)
            elif r.cls == 'chain-bytes': yield item(r.off, r.n, r.chain, 0, '', '', 'random', cls=r.cls)
            elif r.cls == 'nonce': yield item(r.off, r.n, r.chain, r.rnd, r.rname, 'nonce', 'nonce', **more)
            elif r.tree is not None:  # an opening: for each position a column and its salt, then the given digests
                if at_round != (r.chain, r.rnd): at_round, opening, tree = (r.chain, r.rnd), -1, None
                if tree is not r.tree or (r.cls in SYMBOL and past_columns):
                    opening, tree, past_columns = opening + 1, r.tree, False
                    xs = next(l[3] for l in r.tree.lists if l[:3] == (name, r.chain, r.rnd))
                    yield dict(len=0, kind='derived', name='list', chain=r.chain, round=r.rnd, rname=r.rname, open=opening, tree=self.tree(r.tree, r.chain), v=xs)
                here = dict(open=opening, tree=r.tree.uid)
                if r.cls in SYMBOL:  # the cutter has one record for each oracle message's share of a column
                    k, pieces = j, []
                    while k < self.recs.stop and recs[k].cls in SYMBOL and recs[k].tree is r.tree and (recs[k].x, recs[k].rnd, recs[k].chain) == (r.x, r.rnd, r.chain):
                        pieces.append(dict(msg=recs[k].part.name, cls=recs[k].cls, sub=recs[k].sub, words=recs[k].n // 8))
                        k += 1
                    n = recs[k - 1].off + recs[k - 1].n - r.off
                    assert n == 8 * r.tree.width, "a column that has not its tree's width"
                    yield item(r.off, n, r.chain, r.rnd, r.rname, 'opening', 'column', pos=r.x, where=r.where, pieces=pieces, **here)
                    j = k
                    continue
                if r.cls in ('salt', 'stored-salt'): yield item(r.off, r.n, r.chain, r.rnd, r.rname, 'opening', 'salt', pos=r.x, cls=r.cls, **here)
                else:
                    past_columns = True
                    yield item(r.off, r.n, r.chain, r.rnd, r.rname, 'opening', 'digest', level=r.x[0], node=r.x[1], cls=r.cls, **here)
            elif r.cls == 'root': yield item(r.off, r.n, r.chain, r.rnd, r.rname, 'message', 'root', **more)
            else:  # any other piece of a message: elements of the extension field, 24 bytes each
                assert r.n % 24 == 0, 'a piece of a message that is no whole number of elements: %s' % r.cls
                for a in range(r.off, r.off + r.n, 24):
                    yield item(a, 24, r.chain, r.rnd, r.rname, 'message', 'answer' if r.cls == 'sample-answer' else 'E', **more)
            j += 1
        for _, d in self.notes[ni:]:
            yield from computed(d)

    def lines(self):
        """The items as lines of JSON: 'raw' becomes 'hex', and elements get their words under 'v'."""
        for d in self.items():
            raw = d.pop('raw', None)
            if raw is not None:
                d['hex'] = raw.hex()
                if d['kind'] in ('answer', 'E', 'column'): d['v'] = list(struct.unpack('<%dQ' % (len(raw) // 8), raw))
            yield json.dumps(d, separators=(',', ':')) + '\n'

    def tiles(self):
        """Asks that the items tile the file: no gap, no overlap, and their joined bytes are the file's."""
        at = 0
        for d in self.items():
            if d.get('len'):
                assert d['off'] == at and d['raw'] == self.data[at:at + d['len']] and len(d['raw']) == d['len'], '%s: the items do not tile the file at byte %d' % (self.name, at)
                at += d['len']
        assert at == len(self.data), '%s: the items end at byte %d' % (self.name, at)


# ---- items -> the parsed form (PARSED-FORM.md) ----
def u64(x):
    return struct.pack('<Q', x)


def a_list(xs):
    return u64(len(xs)) + b''.join(xs)


def some_bytes(b):
    return u64(len(b)) + b


def element(b):
    """An element of E as it is in a file, 24 bytes. Asks its canonical form (R1.3, R1.4)."""
    ask(all(w < format_rules.PRIME[2] for w in struct.unpack('<3Q', b)), 'a word of an element is not below p')
    return b


KIND = {'root': 0, 'E': 3, 'answer': 4}  # 1 (an element of the base field) and 2 (a word) are never written: no message holds one
MAGIC = {'key': b'FPKEYC01', 'leaf': b'FPENTC01', 'proof': b'FPPRFC01'}


def table_in_front(p):
    """The root and the stored answers at the front of a key file or a leaf file."""
    got = []
    for d in p.items():
        if d.get('len') and d['chain'] == 'table': got.append(d['raw'])
        elif d.get('len') and d['chain'] != 'head': break
    return got[0], got[1:]


def parsed_form(p, other, configuration):
    """p: a Parsed. other: for a proof the Parsed of its leaf file, for a leaf file that of the key
    file, which hold the root and the answers that p's own file lacks. Returns (the parsed file, its
    order file, a line of numbers)."""
    # chains: their names in order. By a chain's name: runs: its run's value; rounds: {round: [items,
    # nonce, name, has a nonce]}; extra: its label, random bytes and nonce's length, for the order file.
    chains, runs, rounds, extra = [], {}, {}, {}
    root = bind = place = places = slot_salt = None
    answers, roots, copy, slot_path = [], [], 0, []
    # rows: ('s' | 't', index) -> {leaf: (column, salt, whole path)}. sender: a tree's index -> its
    # chain's number. order: the order file's openings. stored_seen: the weight tables, as first met.
    rows, sender, order, stored_seen = {}, {}, [], []
    cur = None                                          # the opening that is being read
    if p.kind == 'proof': root, answers = table_in_front(other)
    if p.kind == 'leaf': root0, answers0 = table_in_front(other)

    def a_round(chain, d):
        return rounds[chain].setdefault(d['round'], [[], 0, d['rname'], 0])

    def close():
        """The opening is whole: walks it (R7.9) with every vertex kept, and enters its rows."""
        nonlocal cur
        if cur is None: return
        t, xs = cur['tree'], cur['xs']
        ask(xs and all(a < b for a, b in zip(xs, xs[1:])) and xs[-1] < 1 << t['depth'], 'a list of positions that does not increase below 2^depth')
        ask([x for x, _, _ in cur['cols']] == xs and all(s is not None for _, _, s in cur['cols']), "an opening whose columns are not its list's")
        for x, c, s in cur['cols']: ask(len(c) == 8 * t['width'] and len(s) == t['salt_bytes'], "a column or a salt that has not its tree's length")
        start = configuration if t['stored'] else runs[t['sender']]
        leaves = [format_rules.leaf(174 if t['stored'] else 164, t['number'], t['depth'], x, start, list(struct.unpack('<%dQ' % t['width'], c)), s) for x, c, s in cur['cols']]
        met, given = {}, iter(cur['given'])
        top = format_rules.walk(t['depth'], configuration, xs, leaves, lambda d, i: next(given), lambda d, i, v: met.__setitem__((d, i), v))
        ask(next(given, None) is None, 'a given digest too many')
        if t['stored'] and t['uid'] not in stored_seen: stored_seen.append(t['uid'])
        which = stored_seen.index(t['uid']) if t['stored'] else t['order']   # a leaf file opens two weight tables: the anchor, then its own
        ask(top == (((root0, root) if p.kind == 'leaf' else (root,))[which] if t['stored'] else roots[which]), 'an opening that does not lead to its root')
        if not t['stored']: sender[which] = chains.index(t['sender'])
        table = rows.setdefault(('s' if t['stored'] else 't', which), {})
        for x, c, s in cur['cols']:
            if x in table: ask(table[x][:2] == (c, s), 'one leaf with two columns or two salts')   # R7.14: written once
            else: table[x] = (c, s, [met[(t['depth'] - k, (x >> k) ^ 1)] for k in range(t['depth'])])
        order.append(u64(chains.index(cur['chain'])) + u64(cur['round']) + u64(t['stored']) + u64(which) + a_list([u64(x) for x in xs]))
        cur = None

    for d in p.items():
        if 'form' in d: continue
        chain = d['chain']
        if d['len'] == 0:
            what = d['name']
            if what == "run's value":
                chains.append(chain)
                runs[chain], rounds[chain] = bytes.fromhex(d['value']), {}
                extra.setdefault(chain, {})['label'] = d['label']
            elif what == "chain's value":
                close()
                a_round(chain, d)
            elif what == 'binding root': bind, places = bytes.fromhex(d['value']), 1 << d['depth']
            elif what == 'list':
                close()
                cur = dict(tree=d['tree'], xs=d['v'], cols=[], given=[], chain=chain, round=d['round'])
            continue
        b, kind = d['raw'], d['kind']
        if d['part'] == 'opening':
            if kind == 'column': cur['cols'].append([d['pos'], b, None])
            elif kind == 'salt':
                assert cur['cols'][-1][0] == d['pos']
                cur['cols'][-1][2] = b
            else: cur['given'].append(b)
            continue
        close()
        if chain == 'head':
            if kind == 'u32': copy = d['v']
        elif chain == 'table':
            if kind == 'root': root = b
            else: answers.append(element(b))
        elif chain == 'slot':
            if kind == 'word': place = d['v']
            elif kind == 'salt': slot_salt = b
            else: slot_path.append(b)
        elif kind == 'random': extra.setdefault(chain, {})['random'] = b
        elif d['part'] == 'message':
            if kind == 'root': roots.append(b)
            a_round(chain, d)[0].append(u64(KIND[kind]) + (b if kind == 'root' else element(b)))
        else:
            assert d['part'] == 'nonce'
            r = a_round(chain, d)
            r[1], r[3] = d['v'], 1
            assert extra[chain].setdefault('nonce bytes', d['len']) == d['len']
    close()
    if p.kind == 'leaf': assert len(chains) == 1 and root is not None and bind is None and place is None
    else: assert len(chains) == 3 and None not in (root, bind, place, slot_salt) and len(slot_path) == places.bit_length() - 1 and place < places
    for c in chains: assert sorted(rounds[c]) == list(range(1, len(rounds[c]) + 1)), 'a round is lacking in the chain ' + c

    def in_order(c): return [rounds[c][i] for i in sorted(rounds[c])]
    def the_rounds(c): return a_list([a_list(r[0]) + u64(r[1]) for r in in_order(c)])
    def the_rows(t): return a_list([u64(x) + u64(len(t[x][0]) // 8) + t[x][0] + some_bytes(t[x][1]) + a_list(t[x][2]) for x in sorted(t)])
    trees = [u64(o) + u64(sender[o]) + the_rows(rows[('t', o)]) for o in sorted(o for s, o in rows if s == 't')]
    if p.kind == 'leaf':
        out = [MAGIC['leaf'], root0, a_list(answers0), root, a_list(answers), runs[chains[0]], the_rounds(chains[0]), a_list(trees), the_rows(rows.get(('s', 0), {})), the_rows(rows.get(('s', 1), {}))]
    else:
        out = [MAGIC[p.kind], root, a_list(answers), runs[chains[0]], u64(copy), the_rounds(chains[0]), runs[chains[1]], the_rounds(chains[1]), runs[chains[2]], the_rounds(chains[2]),
               a_list(trees), the_rows(rows.get(('s', 0), {})), bind, u64(place), u64(places), some_bytes(slot_salt), a_list(slot_path)]
    side = b'FPORDER1' + u64({'key': 1, 'leaf': 2, 'proof': 3}[p.kind]) + u64(p.kind != 'proof')
    side += a_list([u64(extra[c]['label']) + u64(extra[c].get('nonce bytes', 0)) + u64('random' in extra[c]) + extra[c].get('random', bytes(32))
                    + a_list([some_bytes(r[2].encode('ascii')) for r in in_order(c)]) + a_list([u64(r[3]) for r in in_order(c)]) for c in chains])
    side += a_list(order)
    numbers = 'rounds %s; %d stored answers; %d roots sent; %d trees opened at %d leaves; weight tables opened at %s leaves; %d openings' % (
        ' | '.join(str(len(rounds[c])) for c in chains), len(answers), len(roots), len(trees), sum(len(rows[k]) for k in rows if k[0] == 't'), ' + '.join(str(len(rows[k])) for k in sorted(rows) if k[0] == 's'), len(order))
    return b''.join(out), side, numbers


def seeds_of(root, configuration, shape):
    """R4.8: the seeds of the record of the weight table with this root, drawn from the hash at the
    table's name (R4.10). Returns (the .seeds file, the two computed values for the lines)."""
    name = format_rules.table_name(root, configuration, *shape)
    seeds = format_rules.record_seeds(name, (shape[1] - 1).bit_length(), shape[4], 2)
    e = lambda x: struct.pack('<3Q', *x)
    return a_list([a_list([e(x) for x in z]) + e(beta) for z, beta in seeds]), [
        dict(len=0, kind='derived', name="table's name", chain='table', value=name.hex(), shape=dict(zip(('a', 'message', 'tail', 'log_rows', 'answers', 'salt_bytes'), shape))),
        dict(len=0, kind='derived', name='record seeds', chain='table', v=[dict(point=[list(x) for x in z], beta=list(beta)) for z, beta in seeds])]


# ---- the look before anything is written into a directory that the caller names ----
# From here to cure(): run_suite.py's own constants and functions, copied whole and unchanged.
# tests/folder_look_copies_test.py compares every copy in the repository. They need os, re and stat.
ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")


def main(argv):
    with_lines = '--lines' in argv
    args = [a for a in argv if a != '--lines']
    if len(args) != 6 or any(a.startswith('--') for a in args):
        print(__doc__)
        return 2
    if not __debug__:
        print('parse_files.py: run Python without -O: the cutter refuses a file by assert', file=sys.stderr)
        return 2
    tables, config, ivk, request_file, proof_file, out = args
    found = others_can_change(out)  # before anything is read or written
    if found:
        print('parse_files.py: ' + ('%s does not exist: OUT_DIR must be an existing, empty directory' % printable(found[0]) if found[1] == GONE else refusal('output directory', out, *found)), file=sys.stderr)
        return 2
    out = os.path.realpath(out)  # the path that was looked at, with no link left in it
    if not os.path.isdir(out) or os.listdir(out):
        print('parse_files.py: %s is not an existing, empty directory' % printable(out), file=sys.stderr)
        return 2
    sums, seen = {}, []  # every file read: its name here -> its sha256; the names in the order read

    def read(path, name, users=False, hint=''):
        """users: the file is the user's own, so that a failure is a wrong call and not a refusal."""
        try:
            b = file_bytes(path)
        except (OSError, Refused) as x:
            raise (WrongCall if users else Refused)('%s%s cannot be read: %s' % (name, hint, x.strerror if isinstance(x, OSError) else x))
        sums[name] = sha(b)
        seen.append(name)
        return b

    cutter = None
    try:
        public = [read(os.path.join(config, n), 'config/' + n, True) for n in ('policy.txt', 'layout.txt', 'KL.bin')]
        always = list(seen)
        configuration = format_rules.configuration_digest(public)
        rules = Rules(with_lines)
        cutter = cut.Cutter(rules, 2, configuration, lambda n: read(os.path.join(tables, n), 'tables/' + n, True))
        rules.cutter, cutter.draws, cutter.name, cutter.at = cutter, Forget(), '', 0
        conf_line = dict(len=0, kind='derived', name="configuration's digest", chain='', value=configuration.hex())

        def cut_one(kind, name, data, call, keep, roots):
            """Cuts one file. keep: the files read before that it also depends on. roots: of the
            weight tables whose names and seeds its lines give."""
            del seen[:]
            at = len(cutter.recs), len(rules.notes)
            got = call()  # reads its chains' tables
            table = [conf_line] + [dict(d, which=which) for which, r in roots(got) for d in seeds_of(r, configuration, got['shape'])[1]]
            return got, Parsed(kind, name, data, cutter, rules, at[0], at[1], {n: sums[n] for n in always + list(keep) + seen}, table)

        proof = read(proof_file, 'the proof file')
        ask(len(proof) >= 12, 'the proof file ends inside its head')
        k = struct.unpack('<I', proof[8:12])[0]
        key = read(os.path.join(ivk, 'key.bin'), 'key.bin')
        K, pk = cut_one('key', 'key.bin', key, lambda: cutter.key('key.bin', key), (), lambda g: [("this file's table", g['root'])])
        leaf_name = 'leaf/%d.bin' % k
        leaf = read(os.path.join(ivk, 'leaf', '%d.bin' % k), leaf_name, hint=" (the number comes from the proof file's head)")
        L, pl = cut_one('leaf', leaf_name, leaf, lambda: cutter.entry(leaf_name, leaf, K['root'], k, K['answers']), ('key.bin',), lambda g: [('the anchor, read in the key file', K['root']), ("this file's table", g['root'])])
        ask(tuple(L['shape']) == tuple(K['shape']), 'the key file and the leaf file name two shapes of a weight table')
        request = read(request_file, 'the request file', True)
        _, pp = cut_one('proof', 'the proof file', proof, lambda: cutter.proof('the proof file', proof, {k: L['root']}, public[0], public[2], request, {k: L['answers']}), (leaf_name, 'the request file'), lambda g: [])
    except WrongCall as x:
        print('parse_files.py: %s' % x, file=sys.stderr)
        return 2
    except UnicodeError:  # only the public files and the tables are read as text
        print('parse_files.py: a file of CONFIG_DIR or TABLES_DIR is not the text that it should be', file=sys.stderr)
        return 2
    except struct.error:  # the cutter asks a file's length before it unpacks its bytes, so this is a table
        print('parse_files.py: a file of TABLES_DIR ends early: make TABLES_DIR anew, with pubtool, from this CONFIG_DIR', file=sys.stderr)
        return 2
    except Refused as x:
        print('REFUSED: %s' % x)
        return 1
    except MemoryError:
        print(NO_MEMORY)
        return 1
    except AssertionError as x:  # how the cutter refuses a file
        name = getattr(cutter, 'name', '')
        why = str(x) or 'it does not fit the plan' + (', or the request file does not fit the public files' if name == 'the proof file' else '')
        if name and why.startswith((name + ', ', name + ': ')): why = why[len(name) + 2:]  # some of the cutter's words begin with the file's name
        print('REFUSED: %s at byte %d: %s' % (name or 'the public files', getattr(cutter, 'at', 0), why))
        return 1
    todo, said = [], []  # (file name in OUT_DIR, bytes or a Parsed whose lines are written); what is printed once they are written
    try:
        for p, other, stem in ((pk, None, 'key'), (pl, pk, 'leaf-%d' % k), (pp, pl, 'proof')):
            p.tiles()
            body, side, numbers = parsed_form(p, other, configuration)
            todo += [(stem + '.parsed.bin', body), (stem + '.parsed.bin.order', side)]
            if p.kind == 'leaf': todo.append((stem + '.parsed.bin.seeds0', seeds_of(K['root'], configuration, K['shape'])[0]))
            todo.append((stem + '.parsed.bin.seeds', seeds_of(K['root'] if p.kind == 'key' else L['root'], configuration, K['shape'])[0]))
            said.append('%s: %d bytes -> %s.parsed.bin: %d bytes, sha256 %s; %s' % (p.name, len(p.data), stem, len(body), sha(body), numbers))
            if with_lines: todo.append((stem + '.lines.jsonl', p))
    except Refused as x:
        print('REFUSED: %s: %s' % (p.name, x))
        return 1
    except MemoryError:
        print(NO_MEMORY)
        return 1
    # Everything has been cut, checked and built: only now is anything written.
    for name, what in todo:
        with open(os.path.join(out, name), 'xb') as f:
            if isinstance(what, bytes): f.write(what)
            else:
                for line in what.lines(): f.write(line.encode('ascii'))
    print('\n'.join(said))
    print('PARSED: 3 files')
    return 0


if __name__ == '__main__':
    try:
        status = main(sys.argv[1:])
    except Exception as x:  # not a verdict on the files
        print('parse_files.py: AN ERROR OF THE SCRIPT ITSELF: %s: %s' % (type(x).__name__, x))
        status = 3
    sys.exit(status)
