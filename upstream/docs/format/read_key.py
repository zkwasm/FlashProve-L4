#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Walks a key file by the rules R2.x to R8.x and R13.x of FORMAT.md, with format_rules.py.

From the repository's top:
    python3 docs/format/read_key.py DIRECTORY
DIRECTORY holds public-file-1.txt, public-file-2.txt, key.file, key.one-time.parameters,
key.shared.parameters and key.second.parameters, as blocks/frame/handover_files writes them
(field 2). It is NOT a verifier: it makes none of the tests on content.

What it does. It reads the head and the root, encodes the three chains' statements and the heads
of their parameters again, forms every first value, makes every round's query, tests the proof of
work, samples, forms the binding tree's root from the first seal, and asks that no byte is left.

What it does not do.
- It makes NONE of the protocol's own checks (no equation, no code word): that it reads a file
  whole says nothing on whether the file proves anything.
- It derives no plan: it takes the bytes of each chain's parameters from the files named above.
- Of the protocol it knows the kinds of the rounds' names (R12.8, R13.11) and nothing of the
  groups. Which trees are opened after a round that samples positions, how deep and how wide they
  are, and which tree the second chain adopts, it finds by trial: it tries every tree that the
  chain can read, by increasing place, at the list or at a short list of up to five bits fewer,
  at every width up to 96, until the walk ends in that tree's root. So what it prints is read off
  the bytes.
- It does NOT know which trees a given round must open, and of a chain's own trees it does not ask
  that one is opened at all: it reads a file whole even if the second opening of a tree that is
  opened twice, or the only opening of another tree, has been taken out of it. With the adopted
  tree's opening in the second chain taken out, it prints another tree as the adopted one. (Tried on
  one toy key file.)
- It serves field 2 and a configuration of two public files only, takes the salt in the slot as
  40 bytes, and cannot read a file in which the sampled positions are listed.
- It has been run on the toy's key file alone; no run on a key file of cp_commit is known.
"""
import os
import struct
import sys


sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_rules as R
if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
    sys.exit("usage: python3 docs/format/read_key.py DIRECTORY (run from the repository's top folder; the comment at the top of the script says what DIRECTORY must hold)")
folder = sys.argv[1]
rd = lambda n: open('%s/%s' % (folder, n), 'rb').read()
ONE = ('LKP', 'JOIN', 'LINE', 'ZERO', 'BAT', 'BAT-sigma', 'PSC round', 'PSC Y', 'FOLD round', 'BASE lambda', 'LIN point', 'LIN zeta')
KIND = dict([(n, 1) for n in ONE] + [('PSC epsilon', 2), ('FOLD epsilon', 2), ('SEND seed', 3), ('LOC', 4), ('CS', 4), ('BASE', 4), ('SEAL', 6), ('SEND answer', 7), ('SPLIT', 7), ('PSC values', 7)])
conf = R.configuration_digest([rd('public-file-1.txt'), rd('public-file-2.txt')])
f = rd('key.file'); at = 0
def take(n):
    global at
    assert at + n <= len(f), 'the file ends early'
    b = f[at:at + n]; at += n; return b
def parse(par):
    pos = len(R.item("the format's version", b'H821' + bytes(4)))
    eight = struct.unpack('<8Q', par[pos:pos + 64]); pos += 64
    entries = []
    for _ in range(eight[7]):
        n = struct.unpack('<Q', par[pos:pos + 8])[0]
        entries.append((par[pos + 8:pos + 8 + n].decode('ascii'),) + struct.unpack('<11Q', par[pos + 8 + n:pos + 96 + n])); pos += 96 + n
    head = R.parameters_head(2, R.plan_bytes(2, *eight[2:7], entries)) + R.item('configuration', conf)
    assert par.startswith(head), 'the head of the parameters, encoded again'
    names = {}; p = len(head)
    while p < len(par):
        n = struct.unpack('<Q', par[p:p + 8])[0]; name = par[p + 8:p + 8 + n].decode(); p += 8 + n
        m = struct.unpack('<Q', par[p:p + 8])[0]; names[name] = par[p + 8:p + 8 + m]; p += 8 + m
    return eight, entries, names
class Tree:
    def __init__(s, root, number, start, who): s.root, s.number, s.start, s.who, s.depth, s.width, s.lists = root, number, start, who, None, None, 0
def try_open(t, k, draws, salt_bytes, leafno=164):
    """Tries tree t at the current position in the file. Returns the number of opened columns and
    moves on, or returns 0 and stays."""
    global at
    keep = at
    for d in ([t.depth] if t.depth else range(k, max(k - 6, 0), -1)):
        xs = sorted({R.short_position(x, d) for x in draws})
        for W in ([t.width] if t.width else range(1, 97)):
            at = keep
            try:
                cols = []
                for x in xs:
                    col = struct.unpack('<%dQ' % W, take(8 * W))
                    if any(y >= R.PRIME[2] for y in col): raise AssertionError
                    cols.append((col, take(salt_bytes)))
                leaves = [R.leaf(leafno, t.number, d, x, t.start, list(col), salt) for x, (col, salt) in zip(xs, cols)]
                if R.walk(d, conf, xs, leaves, lambda dd, i: take(32)) == t.root:
                    t.depth, t.width, t.lists = d, W, t.lists + 1
                    return len(xs)
            except AssertionError:
                pass
    at = keep
    return 0
def chain(title, label, par, statement, random_bytes, book, table):
    eight, entries, names = parse(par); pd = R.parameters_digest(par)
    salt_bytes, nonce_bytes = eight[2], eight[3]
    rb = take(32) if random_bytes is None else random_bytes
    run = R.run_value(label, pd, rb)
    c = R.Chain(R.first_value(label, pd, statement, run), 2, label)
    print('%s: label %#x, %d rounds, items after the plan: %s' % (title, label, sum(e[0] != 'RECORD' for e in entries), ', '.join(['configuration'] + list(names))))
    seal = None
    number = 0
    for e in entries:
        name, count, w, mb, nroots, answers, nE, nz, npos, k, ntrits, rows = e
        if name == 'RECORD':  # R4.3: a plan's entry that is not a round; it is passed over
            continue
        number += 1
        msg = take(mb)
        for j in range(nroots): book.append(Tree(msg[32 * j:32 * j + 32], len(book), run, title))
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
                n = try_open(book[i], k, draws, salt_bytes)
                if n: got.append('place %d (number %d of %s; depth %d, width %d, %d columns)' % (i, book[i].number, book[i].who, book[i].depth, book[i].width, n))
                i += 1
            print('  round %d %s: %d draws in [0, 2^%d): %s' % (number, name, npos, k, '; '.join(got) or 'NOTHING OPENED'))
        elif KIND[name] == 6:
            seal = c.seal_digest()
        else:
            [c.element(nz == nE and nz > 0) for _ in range(nE)]
    return seal, run
assert take(8) == b'CPIKH821', 'the head'
root = take(32)
shape = struct.unpack('<6Q', parse(rd('key.one-time.parameters'))[2]['the test on the table'][:48])  # R13.13 begins with the tables' shape
answers = take(24 * shape[4] << shape[3])
table = Tree(root, 0, conf, 'the key'); table.salt = shape[5]
st_table = R.item('table', R.string(root) + R.u64(len(answers) // 24) + answers)
bookA = []
seal1, run1 = chain('the one-time chain', 0x46524d02, rd('key.one-time.parameters'), st_table, None, bookA, None)
place = struct.unpack('<Q', take(8))[0]; salt = take(40); assert place == 0
broot = R.binding_leaf(0, 0, conf, seal1, salt)
bookB = []
st2 = st_table + R.item("the binding tree's root", broot)
seal2, run2 = chain('the shared chain', 0x4c494e01, rd('key.shared.parameters'), st2, broot, bookB, table)
st3 = R.item('the seal of an earlier chain', seal1) + R.item('the seal of an earlier chain', seal2)
print('the second chain is tried with each tree of the shared chain adopted in turn')
keep = at
for j, t in enumerate(bookB):
    at = keep
    book = list(bookA) + [t]
    try:
        chain('the second chain', 0x4c494e02, rd('key.second.parameters'), st3, None, book, None)
        if at == len(f):
            print('adopted: place %d of the shared chain; the whole file is read: %d bytes' % (j, len(f))); break
    except AssertionError as ex:
        print('  (with place %d of the shared chain: %s)' % (j, ex))
else:
    sys.exit('no choice of a tree to adopt reads the file to its end')
