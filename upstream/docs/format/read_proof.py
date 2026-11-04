#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Walks a proof file by the rules R2.x to R8.x and R12.x to R15.x of FORMAT.md, with
format_rules.py.

From the repository's top:
    python3 docs/format/read_proof.py TOP BATCH N
with, after N, any of: [FILE] [--public=a,b,c] [--table=NAME] [--statement=PARAMETERS,SCALES]
[--field=1] [--trial=1 [--widest=W]]. It is NOT a verifier: it makes none of the tests on content.
  TOP          holds public-file-1.txt, public-file-2.txt, key.file, entry.file (a leaf file).
  TOP/BATCH    holds N.proof, N.request, N.first.parameters, shared.parameters and
               second.parameters, as blocks/frame/handover_proof writes them (field 2).
  FILE         another file of TOP/BATCH instead of N.proof.
  --public     the public files' names in TOP, in the configuration's order (R4.9).
  --table      the file in TOP whose root is the table's (for a proof that cp_prove wrote: the
               leaf file of the proof's leaf table).
  --statement  the names in TOP of the parameter file and the scale file: N.request is then the
               request file, and the item "request" is formed by R15.5 to R15.7 (a proof that
               cp_prove wrote). Without it N.request holds the item's bytes. In both cases
               N.request is also the item "the request file's words" (R14.3, item 4).
  --field      1 for field 1; without it, field 2.
  --trial      depths and widths are found by trial too, as read_key.py does (depths down to five
               bits below the list's); slow where columns are wide.
  --widest     the widest column that is tried, in elements of F (96).

What it does. It reads the head, encodes the three chains' statements and the heads of their
parameters again, forms every first value, makes every round's query, tests the proof of work,
samples, goes from the first chain's seal up the binding tree by the slot's salt and siblings, and
asks that no byte is left. It knows all names of the format's rounds (R14.11). How deep and how
wide each tree is it derives from the items of the first query (R7.12, R12.10, R12.16 to R12.18,
R13.19, R14.4), and it stops where a prover's message holds another number of roots than these
rules give. It stops where positions are sampled and no tree opens after them, and at the file's
end it asks that every tree of the three chains has been opened at least once.

What it does not do.
- It makes NONE of the protocol's own checks: that it reads a file whole says nothing on whether
  the file proves anything.
- It derives no plan: it takes the bytes of each chain's parameters from the files named above.
- Of the sampled values only the positions and the seals show in anything that it asks. A sampled
  value moves nothing in a chain (the next round's value is fixed when the query is made), so the
  elements and signs that it samples are made and dropped, and a reader that left them out would
  print the same.
- Which trees are opened after a round that samples positions it finds by trial: it tries every
  tree that the chain can read, by increasing place. It does NOT know which trees a given round
  must open: it reads a file whole even if the second opening of a tree that is opened twice has
  been taken out of it.
- Without --table it reads a proof whose head has the number 0 against the table of key.file, and
  any other proof against that of entry.file. That is the numbering of the tool that makes the toy
  files, and not what R14.2 says.
- It cannot read a file in which the sampled positions are listed.
- It has been run only on files with toy segments, on honest proofs that cp_prove wrote in field 2
  at small configurations (verifier/tests/walker_on_a_world_test.py runs it at two of the tests'),
  and on one in field 1 that programs of other sources made (the programs of this repository
  can be built for field 2 alone). The one exception: it has read one proof at full size whole, the stored example
  examples/full-size (by the account of those who ran it; no record of the run is in this
  repository).
"""
import os
import struct
import sys


sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_rules as R
words = [a for a in sys.argv[1:] if not a.startswith('--')]
if len(words) < 3:
    sys.exit("usage: python3 docs/format/read_proof.py TOP BATCH N [FILE] [--public=a,b,c] [--table=NAME] [--statement=PARAMETERS,SCALES] [--field=1] [--trial=1 [--widest=W]] (run from the repository's top folder; the comment at the top of the script explains each argument)")
opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
top, folder, which = words[0], os.path.join(words[0], words[1]), words[2]
FIELD = int(opts.get('field', 2)); VERSION = {1: b'H811', 2: b'H821'}[FIELD]  # R4.3
rd = lambda n: open('%s/%s' % (folder, n), 'rb').read()
ONE = ('BAT', 'BAT-sigma', 'PSC round', 'PSC Y', 'ZERO', 'LINE', 'LKP shift', 'JOIN', 'LKP', 'FOLD round', 'BASE lambda', 'LIN point', 'LIN zeta', 'LINK point', 'LINK phi')  # R14.11
KIND = dict([(n, 1) for n in ONE] + [('PSC epsilon', 2), ('FOLD epsilon', 2), ('SEND seed', 3), ('LOC', 4), ('CS', 4), ('BASE', 4), ('LINK', 4), ('RNG', 5), ('SEAL', 6), ('SEND answer', 7), ('SPLIT', 7), ('PSC values', 7), ('REPLY', 7), ('LAST', 7)])
up = lambda n: open('%s/%s' % (top, n), 'rb').read()
conf = R.configuration_digest([up(n) for n in opts.get('public', 'public-file-1.txt,public-file-2.txt').split(',')])
f = rd(words[3] if len(words) > 3 else which + '.proof'); at = 0
def take(n):
    global at
    assert at + n <= len(f), 'the file ends early'
    b = f[at:at + n]; at += n; return b
def parse(par):
    pos = len(R.item("the format's version", VERSION + bytes(4)))
    eight = struct.unpack('<8Q', par[pos:pos + 64]); pos += 64
    entries = []
    for _ in range(eight[7]):
        n = struct.unpack('<Q', par[pos:pos + 8])[0]
        entries.append((par[pos + 8:pos + 8 + n].decode('ascii'),) + struct.unpack('<11Q', par[pos + 8 + n:pos + 96 + n])); pos += 96 + n
    head = R.parameters_head(FIELD, R.plan_bytes(FIELD, *eight[2:7], entries)) + R.item('configuration', conf)
    assert par.startswith(head), 'the head of the parameters, encoded again'
    names = {}; p = len(head)
    while p < len(par):
        n = struct.unpack('<Q', par[p:p + 8])[0]; name = par[p + 8:p + 8 + n].decode(); p += 8 + n
        m = struct.unpack('<Q', par[p:p + 8])[0]; names[name] = par[p + 8:p + 8 + m]; p += 8 + m
    return eight, entries, names
class Tree:
    def __init__(s, root, number, start, who, depth=None, width=None): s.root, s.number, s.start, s.who, s.depth, s.width, s.lists = root, number, start, who, depth, width, 0
NONE = (1 << 32) - 1
def words_of(b): return list(struct.unpack('<%dQ' % (len(b) // 8), b))
class Take:
    def __init__(s, w): s.w, s.p = w, 0
    def one(s):
        s.p += 1; return s.w[s.p - 1]
    def some(s, n): return tuple(s.one() for _ in range(n))
    def lst(s, f): return [f() for _ in range(s.one())]
def lay(codes, notices):
    """R7.12, R7.1: the trees of one prover's message as (depth, width in elements of F). A notice:
    R12.3's seven words."""
    base, mask = codes
    trees = []  # [class, length of the main rows or None, parts]
    for n in notices:
        if n[2]:
            t = next((t for t in trees if t[0] == n[6] and t[1] == base[n[0]][0]), None)
            if t is None: t = [n[6], base[n[0]][0], []]; trees.append(t)
            t[2].append(n)
    for n in notices:
        if not n[2]:
            t = next((t for t in trees if t[0] == n[6]), None)
            if t is None: t = [n[6], None, []]; trees.append(t)
            t[2].append(n)
    return [(t[1] if t[1] is not None else mask[t[2][0][1]][0], sum(n[2] * n[3] + 3 * n[4] for n in t[2])) for t in trees]
def up2(r): return 0 if r == 0 else 1 << (r - 1).bit_length()  # R12.16's up(r)
def log_up(r): return 0 if r <= 1 else (r - 1).bit_length()
def fresh_mask(stage, cls=0): return (NONE, stage[1], 0, 1, 1 << stage[3], 0, cls)  # R12.10
def test_messages(stages, first_own, ride):
    """R13.19, R12.10: the prover's messages of a proximity test that hold roots, in order, each as
    a list of notices."""
    out = []
    for i, s in enumerate(stages[:-1]):
        if (first_own if i == 0 else not ride): out.append([fresh_mask(s)])
        n = stages[i + 1]; last = i + 2 == len(stages)
        m = [(n[0], n[1], 1 << n[2], 3, 1 << n[3], 0, 0)]
        if ride: m.append((n[0], n[1], 1, 3, 1, 0, 0) if last else fresh_mask(n))
        out.append(m)
    if not ride: out.append([(stages[-1][0], stages[-1][1], 1, 3, 1, 0, 0)])
    return out
def shape_of_request(b):
    t = Take(words_of(b)); code, mask_code = t.some(2)
    links = t.lst(lambda: (t.one(), t.lst(lambda: t.some(6))))
    return dict(code=code, mask=mask_code, links=links, riders=t.lst(lambda: t.some(7)), s=t.one(), later=t.lst(lambda: t.some(4)), ride=t.one())
def table_test(b):
    t = Take(words_of(b)); o = {}
    o['table'] = t.some(6); o['u'] = t.some(6); t.one(); o['u class'] = t.one(); t.some(3); o['omega'] = t.some(6); o['w'] = t.some(6); t.some(2)
    o['shared split'] = t.some(6); o['own split'] = t.some(6); o['classes'] = t.some(4); t.some(2); o['ahead'] = t.one()
    o['tests'] = [t.lst(lambda: t.some(4)) for _ in range(4)]; o['ride'] = t.one()
    return o
def first_chain_messages(q, T):
    """R14.4 with R12.16 to R12.18: the messages of a proof's first chain that hold roots."""
    L, out, groups = len(q['links']), [], []
    for m in range(1, L + 2):  # R12.16
        own = q['links'][m - 1][1] if m <= L else []
        kept_here = [x for j in range(1, m) for x in q['links'][j - 1][1] if x[5] > 0 and j + x[5] >= m]  # sent again in groups j + 1 to j + f
        main = (2 if m > 1 else 0) + sum(up2(x[0]) for j in range(1, m) for x in q['links'][j - 1][1] if x[5] > 0 and j + x[5] == m) + sum(up2(x[0]) for x in own if x[5] == 0)
        masks = sum(up2(x[2]) for x in own)
        h = log_up(masks); k = max(log_up(main), h, 1)
        groups.append((q['code'], q['mask'], k, h))
        note = lambda x: (q['code'] if x[0] else NONE, q['mask'] if x[2] else NONE, x[0], x[1], x[2], q['s'], x[3])
        first = ([(q['code'], NONE, 2, 3, 0, q['s'], 0)] if m > 1 else []) + [(q['code'], NONE, x[0], x[1], 0, q['s'], x[3]) for x in kept_here]
        first += [note(x) for x in own if x[4] == 0] + [fresh_mask(groups[-1])] + (list(q['riders']) if m == 1 else [])
        out.append(first)
        for t in range(1, 1 + max([x[4] for x in own], default=0)): out.append([note(x) for x in own if x[4] == t])
    out += test_messages([groups[-1]] + q['later'], False, q['ride'])
    out.append([T['w'] + (T['classes'][2],)] + ([fresh_mask(T['tests'][2][0], T['classes'][2])] if T['ahead'] else []))
    return out
def shared_chain_messages(T):
    split = T['u'][0] != T['omega'][0]
    out = [[T['u'] + (T['u class'],)] + ([fresh_mask(T['tests'][0][0], T['u class'])] if T['ahead'] else [])]
    out.append([T['omega'] + (T['classes'][0],)] + ([T['shared split'] + (T['classes'][1],)] if split else []))
    out += test_messages(T['tests'][0], not T['ahead'], T['ride'])
    if split: out += test_messages(T['tests'][1], True, T['ride'])
    return out
def try_open(t, k, draws, salt_bytes, leafno=164):
    """Tries tree t at the current position in the file. Returns the number of opened columns and
    moves on, or returns 0 and stays."""
    global at
    keep = at
    for d in ([t.depth] if t.depth else range(k, max(k - 6, 0), -1)):
        xs = sorted({R.short_position(x, d) for x in draws})
        for W in ([t.width] if t.width else range(1, int(opts.get('widest', 96)) + 1)):
            at = keep
            try:
                cols = []
                for x in xs:
                    col = struct.unpack('<%dQ' % W, take(8 * W))
                    if any(y >= R.PRIME[FIELD] for y in col): raise AssertionError
                    cols.append((col, take(salt_bytes)))
                leaves = [R.leaf(leafno, t.number, d, x, t.start, list(col), salt) for x, (col, salt) in zip(xs, cols)]
                if R.walk(d, conf, xs, leaves, lambda dd, i: take(32)) == t.root:
                    t.depth, t.width, t.lists = d, W, t.lists + 1
                    return len(xs)
            except AssertionError:
                pass
    at = keep
    return 0
def chain(title, label, par, statement, random_bytes, book, table, due=None):
    """due: the messages that hold roots, in order, each as a list of notices, which lay() places
    into trees by the codes; None: depths and widths are found by trial."""
    eight, entries, names = parse(par); pd = R.parameters_digest(par)
    due = list(due) if due is not None else None
    salt_bytes, nonce_bytes = eight[2], eight[3]
    rb = take(32) if random_bytes is None else random_bytes
    run = R.run_value(label, pd, rb)
    c = R.Chain(R.first_value(label, pd, statement, run), FIELD, label)
    print('%s: label %#x, %d rounds, items after the plan: %s' % (title, label, sum(e[0] != 'RECORD' for e in entries), ', '.join(['configuration'] + list(names))))
    seal = None
    number = 0
    for e in entries:
        name, count, w, mb, nroots, answers, nE, nz, npos, k, ntrits, rows = e
        if name == 'RECORD':  # R4.3: a plan's entry that is not a round; it is passed over
            continue
        number += 1
        msg = take(mb)
        laid = [(None, None)] * nroots
        if nroots and due is not None:
            assert due, 'round %d %s holds a root, and by the rules no message with a root is due' % (number, name)
            laid = lay(CODES, due.pop(0))
            assert len(laid) == nroots, 'round %d %s holds %d roots; by R7.12 its oracle messages make %d trees' % (number, name, nroots, len(laid))
        for j in range(nroots): book.append(Tree(msg[32 * j:32 * j + 32], len(book), run, title, *laid[j]))
        nonce = int.from_bytes(take(nonce_bytes), 'little') if w else 0
        assert c.query(msg, KIND[name], w, nonce), 'proof of work in round %d' % number
        if KIND[name] == 3:
            v = nE // nz - 1
            for _ in range(nz):
                [c.element() for _ in range(v)]; c.element(True)
        elif KIND[name] == 4:
            draws = [c.position(k) for _ in range(npos)]; [c.element() for _ in range(nE)]
            got = []
            if table and name == 'LOC':
                n = try_open(table, k, draws, table.salt, 174); assert n, 'the stored table'; got.append('the stored table (depth %d, width %d, %d columns)' % (table.depth, table.width, n))
            i = 0
            while i < len(book):
                began = at
                n = try_open(book[i], k, draws, salt_bytes)
                if n: got.append('place %d (number %d of %s; depth %d, width %d, %d columns; bytes %d to %d)' % (i, book[i].number, book[i].who, book[i].depth, book[i].width, n, began, at))
                i += 1
            print('  round %d %s: %d draws in [0, 2^%d): %s' % (number, name, npos, k, '; '.join(got) or 'NOTHING OPENED'))
            assert got, 'round %d %s: positions are drawn and no tree opens behind them' % (number, name)
        elif KIND[name] == 5:
            c.trits(ntrits)
        elif KIND[name] == 6:
            seal = c.seal_digest()
        else:
            [c.element(nz == nE and nz > 0) for _ in range(nE)]
    assert not due, 'by the rules %d more messages with roots were due' % len(due)
    return seal, run
assert take(8) == b'CPPF' + VERSION, 'the head'
copy = struct.unpack('<I', take(4))[0]
first_par = rd(which + '.first.parameters')
names = parse(first_par)[2]
shape = struct.unpack('<6Q', names['the test on the table'][:48])  # R13.13 begins with the tables' shape
# the default is the numbering of the tool that makes the toy files, not R14.2's: see the head
stored = up(opts['table'] if 'table' in opts else 'key.file' if copy == 0 else 'entry.file')
assert stored[:8] in (b'CPIK' + VERSION, b'CPEN' + VERSION)
root = stored[8:40]; answers = stored[40:40 + (24 * shape[4] << shape[3])]
trial = 'trial' in opts
table = Tree(root, 0, conf, 'the stored copy %d' % copy, *((None, None) if trial else (shape[0], 1 << shape[3]))); table.salt = shape[5]
tc = Take(words_of(names['codes'])); CODES = (tc.lst(lambda: tc.some(3)), tc.lst(lambda: tc.some(3)))  # R12.2
Q, TT = shape_of_request(names["request's shape"]), table_test(names['the test on the table'])
st_table = R.item('table', R.string(root) + R.u64(len(answers) // 24) + answers)
depth, leaf_salt = struct.unpack('<2Q', names['binding tree'])  # R14.6
asked = file_words = rd(which + '.request')
# R15.5 to R15.7: N.request is the request file; with the parameter file and the scale file it
# gives the item
if 'statement' in opts:
    asked = R.statement_words(*[up(n) for n in opts['statement'].split(',')], asked)
st1 = st_table + R.item('stored copy', R.u64(copy)) + R.item('request', R.string(asked)) + R.item("the request file's words", R.string(file_words))  # R14.3
bookA = []
seal1, run1 = chain('the first chain', 0x46524d01, first_par, st1, None, bookA, None, None if trial else first_chain_messages(Q, TT))
place = struct.unpack('<Q', take(8))[0]; assert place >> depth == 0, 'the slot'
salt = take(leaf_salt)
node = R.binding_root(depth, place, conf, seal1, salt, [take(32) for _ in range(depth)])  # R7.18, R14.7
print('slot %d of 2^%d; the binding tree\'s root %s' % (place, depth, node.hex()))
bookB = []
seal2, run2 = chain('the shared chain', 0x46524d03, rd('shared.parameters'), st_table + R.item("the binding tree's root", node), node, bookB, table, None if trial else shared_chain_messages(TT))
st3 = R.item('the seal of an earlier chain', seal1) + R.item('the seal of an earlier chain', seal2)
keep = at
# how often each tree has been opened up to here: an attempt below that fails leaves no count
opened = [(x, x.lists) for x in bookA + bookB]
for j, t in enumerate(bookB):
    at = keep
    for x, n in opened: x.lists = n
    bookC = list(bookA) + [t]
    try:
        chain('the second chain', 0x46524d04, rd('second.parameters'), st3, None, bookC, None, None if trial else test_messages(TT['tests'][2], not TT['ahead'], TT['ride']))
        if at == len(f):
            never = ['number %d of %s' % (x.number, x.who) for x in bookC + [y for y in bookB if y is not t] if not x.lists]
            assert not never, 'the file ends, and %d of its trees were never opened: %s' % (len(never), ', '.join(never))
            print('adopted: place %d of the shared chain; the whole file is read: %d bytes' % (j, len(f))); break
    except AssertionError as ex:
        print('  (with place %d of the shared chain: %s)' % (j, ex))
else:
    sys.exit('no choice of a tree to adopt reads the file to its end')
