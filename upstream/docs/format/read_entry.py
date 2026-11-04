#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Walks a leaf file by the rules R2.x to R8.x and R13.x of FORMAT.md, with format_rules.py.

From the repository's top:
    python3 docs/format/read_entry.py DIRECTORY
DIRECTORY holds public-file-1.txt, public-file-2.txt, key.file, entry.file (the leaf file) and
entry.parameters, as blocks/frame/handover_files writes them (field 2). It is NOT a verifier: it
makes none of the tests on content.

What it does. It reads the head, the leaf table's root and its record's answers, takes the
anchor's root and answers from the key file, encodes the statement and the head of the parameters
again, reads the item "the link" (R13.9), forms the first value, makes every round's query, tests
the proof of work, samples, divides the draws of the round LINK into their two parts (R7.19),
follows the openings of both stored tables and of the chain's own trees up to their roots, with
the widths of R13.10, and asks that no byte is left.

What it does not do.
- It makes NONE of the protocol's own checks (no equation, no code word): that it reads a file
  whole says nothing on whether the file proves anything.
- It derives no plan: it takes the bytes of the chain's parameters from entry.parameters.
- It serves field 2 and a configuration of two public files only, and cannot read a file in which
  the sampled positions are listed.
- It reads the word of "the link" on a mask that is sent ahead and does not use it, and it does
  not read the choices' word: its widths are those of a file in which both words are 1.
"""
import os
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_rules as R
if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
    sys.exit("usage: python3 docs/format/read_entry.py DIRECTORY (run from the repository's top folder; the comment at the top of the script says what DIRECTORY must hold)")
folder = sys.argv[1]
rd = lambda n: open('%s/%s' % (folder, n), 'rb').read()
KIND = {'SEND seed': 3, 'SEND answer': 7, 'LINK point': 1, 'LINK': 4, 'LINK phi': 1, 'BAT': 1, 'FOLD epsilon': 2, 'FOLD round': 1, 'CS': 4, 'BASE lambda': 1, 'BASE': 4}
LABEL = 0x4c4e4b01
conf = R.configuration_digest([rd('public-file-1.txt'), rd('public-file-2.txt')])
par = rd('entry.parameters'); pd = R.parameters_digest(par)
pos = len(R.item("the format's version", b'H821' + bytes(4)))
eight = struct.unpack('<8Q', par[pos:pos + 64]); pos += 64
entries = []
for _ in range(eight[7]):
    n = struct.unpack('<Q', par[pos:pos + 8])[0]
    entries.append((par[pos + 8:pos + 8 + n].decode('ascii'),) + struct.unpack('<11Q', par[pos + 8 + n:pos + 96 + n])); pos += 96 + n
head = R.parameters_head(2, R.plan_bytes(2, *eight[2:7], entries)) + R.item('configuration', conf)
assert par.startswith(head), 'the head of the parameters, encoded again'
items = {}; p = len(head)
while p < len(par):
    n = struct.unpack('<Q', par[p:p + 8])[0]; name = par[p + 8:p + 8 + n].decode(); p += 8 + n
    m = struct.unpack('<Q', par[p:p + 8])[0]; items[name] = par[p + 8:p + 8 + m]; p += 8 + m
link = struct.unpack('<%dQ' % (len(items['the link']) // 8), items['the link'])
log_n, l, r, v, s, tsalt = link[:6]; rest = link[6:]
nu0, nuj, code, mcode, Rr, rows_high, leaf_class, masks_ahead = rest[:8]
nst = rest[8]; stages = [rest[9 + 4 * i:13 + 4 * i] for i in range(nst)]
salt_bytes, nonce_bytes = eight[2], eight[3]
f = rd('entry.file'); at = 0
def take(n):
    global at
    assert at + n <= len(f), 'the file ends early'
    b = f[at:at + n]; at += n; return b
assert take(8) == b'CPENH821', 'the head'
root_j = take(32); answers_j = take(24 * s << v)
key = rd('key.file'); assert key[:8] == b'CPIKH821'; root_0 = key[8:40]; answers_0 = key[40:40 + (24 * s << v)]
table = lambda name, root, ans: R.item(name, R.string(root) + R.u64(len(ans) // 24) + ans)
statement = table('the first table', root_0, answers_0) + table('the further table', root_j, answers_j)
rb = take(32); run_value = R.run_value(LABEL, pd, rb)
c = R.Chain(R.first_value(LABEL, pd, statement, run_value), 2, LABEL)
h0 = 0; k0 = 1 + (Rr - 1).bit_length()
widths = [3 * 2 * Rr + 3 * (1 << h0)] + [3 * (1 << st[2]) + 6 * (1 << st[3]) for st in stages[:-1]] + [12]
roots, opened = [], 0
def open_tree(number, tree, depth, W, xs, start, root, sb):
    cols = []
    for x in xs:
        col = struct.unpack('<%dQ' % W, take(8 * W)); assert all(y < R.PRIME[2] for y in col); cols.append((col, take(sb)))
    leaves = [R.leaf(number, tree, depth, x, start, list(col), salt) for x, (col, salt) in zip(xs, cols)]
    n = [0]
    def given(d, i):
        n[0] += 1; return take(32)
    assert R.walk(depth, conf, xs, leaves, given) == root, 'an opening does not lead to its root'
    return n[0]
number = 0
for e in entries:
    name, count, w, mb, nroots, answers, nE, nz, npos, k, ntrits, rows = e
    if name == 'RECORD':  # R4.3: a plan's entry that is not a round; it is passed over
        continue
    number += 1
    msg = take(mb)
    if nroots: roots.append(msg[:32])
    nonce = int.from_bytes(take(nonce_bytes), 'little') if w else 0
    assert c.query(msg, KIND[name], w, nonce), 'proof of work in round %d' % number
    if KIND[name] == 3:
        vv = nE // nz - 1
        for _ in range(nz):
            [c.element() for _ in range(vv)]; c.element(True)
    elif name == 'LINK':
        draws = [c.position(k) for _ in range(npos)]
        assert npos == nu0 + nuj and k == log_n
        for part, root, what in ((draws[:nu0], root_0, 'the anchor'), (draws[nu0:], root_j, 'the leaf table')):
            xs = sorted(set(part)); g = open_tree(174, 0, log_n, 1 << v, xs, conf, root, tsalt)
            print('round %d: %s: %d draws, %d positions, width %d, %d given digests: root reached' % (number, what, len(part), len(xs), 1 << v, g))
    elif KIND[name] == 4:
        draws = [c.position(k) for _ in range(npos)]; [c.element() for _ in range(nE)]
        xs = sorted(set(draws)); g = open_tree(164, opened, k, widths[opened], xs, run_value, roots[opened], salt_bytes)
        print('round %d: %d draws, %d positions, tree %d of depth %d, width %d, %d given digests: root reached' % (number, npos, len(xs), opened, k, widths[opened], g)); opened += 1
    else:
        [c.element(nz == nE and nz > 0) for _ in range(nE)]
assert at == len(f), 'bytes are left: %d' % (len(f) - at)
print('leaf file: %d bytes, %d rounds: the whole file is read; last chain value %s' % (len(f), number, c.rho.hex()[:24]))
