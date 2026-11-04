#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
"""Reads a file that parse_files.py wrote and tests it. The form is in PARSED-FORM.md.

usage: read_parsed.py PARSED_FILE [--config=CONFIG_DIR] [--file=ORIGINAL_FILE]

This is a second reader. It shares with parse_files.py only docs/format/format_rules.py, from which
both take the hash, so that an error in one of the two shows as a difference. It reads PARSED_FILE
without any hash and without the plan. It also reads the seeds files beside it: PARSED_FILE.seeds,
and for a leaf file PARSED_FILE.seeds0 too. It then makes these tests:
  1. The form: a known first eight bytes; item kinds 0, 3 and 4 only; every word of an element, an
     answer, a column or a seed is below p (R1.3); no byte is left over, in a seeds file either.
  2. The shape: trees by increasing number; every tree and weight table has a row; leaves strictly
     increasing; in one tree all columns, all salts and all paths have one length, and one that
     the format admits (R2.3, R2.7, R7.3); each leaf is below 2^depth; each tree was sent by the
     chain that the file names for it; the slot fits its path's length; "copy" is 0 in a key's
     file. In a seeds file all points have one length, and its table has one stored answer for
     each seed and each word of a column (R4.10).
  3. With --config: every opened leaf, hashed as docs/format/FORMAT.md says (R7.4), leads by its
     whole path to its tree's root.
  4. With --file: PARSED_FILE and PARSED_FILE.order together give ORIGINAL_FILE again, byte for byte.
What it does not test:
  - The values of the seeds. Computing them takes the table's shape (R4.10), which no parsed file
    holds.
  - bindRoot; whether an element's kind is 3 or 4; the nonce of a round without proof of work; the
    stored answers that come from another file (a proof's "answers", a leaf file's "answers0");
    the run's value of a chain none of whose trees is opened; in the order file the labels, the
    names, and "random" where hasRandom is 0. None of them enters a path or the original file.
  - Whether every opening that the protocol asks for is there.
  - Anything of the protocol: a file that passes may still prove nothing.
Exit code 0: one line that begins "READ: ". 1: a line that begins "REFUSED: "; so ends also any file
that is no regular file or has more than 2^28 bytes. 2: a wrong call, or ORIGINAL_FILE or a file of
CONFIG_DIR cannot be opened."""
import os
import stat
import struct
import sys

sys.dont_write_bytecode = True
P = (1 << 64) - (1 << 32) + 1  # R1.2, field 2: the field of the three heads below
CAP = 1 << 28  # bytes: ten times the largest parsed file of the full-size example
HEADS = {1: b'CPIKH821', 2: b'CPENH821', 3: b'CPPFH821'}  # R13.1, R14.2


class Refused(Exception):
    pass


def ask(condition, why):
    if not condition:
        raise Refused(why)


class Reader:
    def __init__(self, data, what):
        """what: how a refusal names the file."""
        self.data, self.what, self.at = data, what, 0

    def raw(self, n):
        ask(self.at + n <= len(self.data), '%s ends early, at byte %d' % (self.what, self.at))
        self.at += n
        return self.data[self.at - n:self.at]

    def word(self):
        return struct.unpack('<Q', self.raw(8))[0]

    def flag(self):
        v = self.word()
        ask(v <= 1, '%s: the flag at byte %d is neither 0 nor 1' % (self.what, self.at - 8))
        return v

    def element(self):
        v = self.word()
        ask(v < P, '%s: the word at byte %d is not below p, so it is no element of the field' % (self.what, self.at - 8))
        return v

    def many(self, one):
        n = self.word()
        ask(n <= len(self.data), '%s: the list at byte %d says that it has %d items, more than the file can hold' % (self.what, self.at - 8, n))
        return [one() for _ in range(n)]

    def string(self):
        return self.raw(self.word())

    def digest(self):
        return self.raw(32)

    def ext(self):
        for _ in range(3):
            self.element()
        return self.data[self.at - 24:self.at]

    def end(self):
        ask(self.at == len(self.data), '%s has bytes after its last field (%d)' % (self.what, len(self.data) - self.at))


def read_parsed(data):
    """Returns the fields by name. A leaf file's one chain is returned as the first of three."""
    r = Reader(data, 'the parsed file')

    def item():
        kind = r.word()
        ask(kind in (0, 3, 4), 'the parsed file: the item at byte %d has kind %d; only 0, 3 and 4 are ever written' % (r.at - 8, kind))
        return kind, r.ext() if kind else r.digest()

    def one_round():
        return r.many(item), r.word()

    def row():
        return dict(leaf=r.word(), column=r.many(r.element), salt=r.string(), path=r.many(r.digest))

    def tree():
        return dict(tree=r.word(), chain=r.word(), rows=r.many(row))

    p = dict(magic=r.raw(8))
    if p['magic'] == b'FPENTC01':
        p.update(root0=r.digest(), answers0=r.many(r.ext), root=r.digest(), answers=r.many(r.ext))
        p.update(runs=[r.digest()], chains=[r.many(one_round)], trees=r.many(tree))
        p.update(stored=[r.many(row), r.many(row)], roots=[p['root0'], p['root']])
        p.update(copy=0, place=0, places=1, slot_salt=b'', slot_path=[])
    else:
        ask(p['magic'] in (b'FPKEYC01', b'FPPRFC01'), 'the parsed file begins with none of FPKEYC01, FPENTC01, FPPRFC01')
        p.update(root=r.digest(), answers=r.many(r.ext), runs=[], chains=[])
        p['runs'].append(r.digest())
        p['copy'] = r.word()
        p['chains'].append(r.many(one_round))
        for _ in range(2):
            p['runs'].append(r.digest())
            p['chains'].append(r.many(one_round))
        p.update(trees=r.many(tree), stored=[r.many(row)], roots=[p['root']], bind_root=r.digest())
        p.update(place=r.word(), places=r.word(), slot_salt=r.string(), slot_path=r.many(r.digest))
    r.end()
    return p


def read_seeds(data, what):
    """Returns the seeds, each as (the point z, beta)."""
    r = Reader(data, what)
    seeds = r.many(lambda: (r.many(r.ext), r.ext()))
    r.end()
    return seeds


def check_seeds(seeds, answers, rows, who):
    """seeds, answers, rows: of one weight table. It has one seed for each stored answer of a row,
    and a column has one word for each row (R4.10)."""
    ask(len({len(z) for z, _ in seeds}) <= 1, who + ' has points of two lengths')
    ask(len(answers) == len(seeds) * len(rows[0]['column']), 'the number of seeds in %s (%d) does not fit the number of stored answers (%d)' % (who, len(seeds), len(answers)))


def check_shape(p):
    """Returns the roots that the chains sent, in order, each as (chain, root)."""
    sent = [(c, v) for c, rounds in enumerate(p['chains']) for items, _ in rounds for k, v in items if k == 0]
    numbers = [t['tree'] for t in p['trees']]
    ask(all(a < b for a, b in zip(numbers, numbers[1:])), 'the trees are not in increasing order')
    named = [(t['rows'], 'tree %d' % t['tree']) for t in p['trees']]
    named += [(rows, 'weight table %d' % i) for i, rows in enumerate(p['stored'])]
    for rows, who in named:
        ask(rows, who + ' has no row')
        leaves = [x['leaf'] for x in rows]
        ask(all(a < b for a, b in zip(leaves, leaves[1:])), who + ' has leaves that do not increase strictly')
        for field in ('column', 'salt', 'path'):
            ask(len({len(x[field]) for x in rows}) == 1, '%s has two lengths of %s' % (who, field))
        ask(leaves[-1] >> len(rows[0]['path']) == 0, who + ' has a leaf that is not below 2^depth')
        width, salt, depth = (len(rows[0][field]) for field in ('column', 'salt', 'path'))
        ask(0 < width < 1 << 24 and salt in (40, 48, 56, 64) and depth <= 62, who + ' has a width, a salt or a depth that the format does not admit')  # R2.3, R2.7, R7.3
    for t in p['trees']:
        ask(t['tree'] < len(sent) and sent[t['tree']][0] == t['chain'], 'tree %d was not sent by chain %d' % (t['tree'], t['chain']))
    ask(p['places'] == 1 << len(p['slot_path']) and p['place'] < p['places'], "the slot does not fit its path's length")
    ask(p['magic'] == b'FPPRFC01' or p['copy'] == 0, '"copy" is not 0 in a file that is no proof')
    return sent


def check_paths(p, sent, rules, configuration):
    """Returns the number of paths followed."""
    def climb(value, place, path):
        for sibling in path:
            value = rules.vertex(configuration, sibling, value) if place & 1 else rules.vertex(configuration, value, sibling)
            place >>= 1
        return value

    in_chain = [sum(1 for c, _ in sent if c == i) for i in range(3)]
    followed = 0
    for t in p['trees']:
        c, j = t['chain'], t['tree'] - sum(in_chain[:t['chain']])
        # R7.13: the second chain's list begins with the first chain's trees and one adopted tree.
        number = j if c < 2 else in_chain[0] + 1 + j
        for x in t['rows']:
            leaf = rules.leaf(164, number, len(x['path']), x['leaf'], p['runs'][c], x['column'], x['salt'])
            ask(climb(leaf, x['leaf'], x['path']) == sent[t['tree']][1], 'tree %d, leaf %d: the path does not lead to the root' % (t['tree'], x['leaf']))
            followed += 1
    for i, (rows, root) in enumerate(zip(p['stored'], p['roots'])):
        for x in rows:
            leaf = rules.leaf(174, 0, len(x['path']), x['leaf'], configuration, x['column'], x['salt'])
            ask(climb(leaf, x['leaf'], x['path']) == root, 'weight table %d, leaf %d: the path does not lead to the root' % (i, x['leaf']))
            followed += 1
    return followed


def read_order(data):
    r = Reader(data, 'the order file')
    ask(r.raw(8) == b'FPORDER1', 'the order file does not begin with FPORDER1')

    def chain():
        return dict(label=r.word(), nonce_bytes=r.word(), has_random=r.flag(), random=r.digest(), names=r.many(r.string), has_nonce=r.many(r.flag))

    def opening():
        return dict(chain=r.word(), round=r.word(), stored=r.flag(), tree=r.word(), at=r.many(r.word))

    o = dict(kind=r.word(), table_here=r.flag(), chains=r.many(chain), openings=r.many(opening))
    r.end()
    ask(all(c['nonce_bytes'] <= 8 for c in o['chains']), 'the order file has a nonce of more than 8 bytes')
    for x in o['openings']:  # so a list is no longer than its tree has rows
        ask(x['at'] and all(a < b for a, b in zip(x['at'], x['at'][1:])), 'the order file has an opening whose leaves are none or do not increase strictly')
    return o


def given_places(depth, leaves):
    """R7.9 without values: (depth, place) of each given digest, in the file's order."""
    out, level = [], list(leaves)
    for d in range(depth, 0, -1):
        above, v = [], 0
        while v < len(level):
            i = level[v]
            if i % 2 == 0 and v + 1 < len(level) and level[v + 1] == i + 1:
                v += 1
            else:
                out.append((d, i ^ 1))
            above.append(i >> 1)
            v += 1
        level = above
    return out


def original_again(p, o, limit):
    """limit: the original file's length. The order file can name one opening any number of times,
    so the openings' bytes are counted against it."""
    ask(o['kind'] in HEADS and len(o['chains']) == len(p['chains']), 'the order file is of another kind of file')
    out = [HEADS[o['kind']]]
    if o['kind'] == 3:
        out.append(struct.pack('<I', p['copy']))
    if o['table_here']:
        out += [p['root']] + p['answers']
    own = {t['tree']: {x['leaf']: x for x in t['rows']} for t in p['trees']}
    stored = {i: {x['leaf']: x for x in rows} for i, rows in enumerate(p['stored'])}
    after, room = {}, limit
    for x in o['openings']:
        after.setdefault((x['chain'], x['round']), []).append(x)
    for c, (extra, rounds) in enumerate(zip(o['chains'], p['chains'])):
        ask(len(extra['has_nonce']) == len(rounds), 'the order file has another number of rounds')
        if c == 1:
            out += [struct.pack('<Q', p['place']), p['slot_salt']] + p['slot_path']
        if extra['has_random']:
            out.append(extra['random'])
        for n, ((items, nonce), has) in enumerate(zip(rounds, extra['has_nonce']), 1):
            out += [v for _, v in items]
            if has:
                out.append(nonce.to_bytes(extra['nonce_bytes'], 'little'))
            for x in after.get((c, n), []):
                table = (stored if x['stored'] else own).get(x['tree'], {})
                ask(all(leaf in table for leaf in x['at']), 'the order file names a row of %s %d that the parsed file does not have' % ('weight table' if x['stored'] else 'tree', x['tree']))
                rows = [table[leaf] for leaf in x['at']]
                depth = len(rows[0]['path'])
                given = given_places(depth, x['at'])
                room -= len(rows) * (8 * len(rows[0]['column']) + len(rows[0]['salt'])) + 32 * len(given)
                ask(room >= 0, 'the openings have more bytes than the original file')
                out += [struct.pack('<%dQ' % len(y['column']), *y['column']) + y['salt'] for y in rows]
                value = {(depth - j, (y['leaf'] >> j) ^ 1): v for y in rows for j, v in enumerate(y['path'])}
                out += [value[place] for place in given]
    return b''.join(out)


def load(path):
    """The bytes of a regular file of at most CAP bytes. O_NONBLOCK: opening a FIFO does not wait."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    found = os.fstat(fd)
    if not stat.S_ISREG(found.st_mode) or found.st_size > CAP:
        os.close(fd)
        raise Refused('%s is no regular file, or has more than 2^28 bytes' % path)
    with os.fdopen(fd, 'rb') as f:
        data = f.read(found.st_size + 1)
    ask(len(data) == found.st_size, '%s changed its length while it was read' % path)
    return data


def load_beside(path):
    """A file that came with the parsed file: if it cannot be read, that is a refusal and no wrong call."""
    try:
        return load(path)
    except OSError as e:
        raise Refused('%s cannot be read: %s' % (path, e.strerror))


def main():
    names = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    if len(names) != 1 or set(opts) - {'config', 'file'} or len(names) + len(opts) != len(sys.argv) - 1:
        print(__doc__)
        return 2
    if not __debug__:
        print('read_parsed.py: run Python without -O: docs/format/format_rules.py refuses a value by assert', file=sys.stderr)
        return 2
    try:
        data = load_beside(names[0])
        p = read_parsed(data)
        sent = check_shape(p)
        tables = [('.seeds0', p['answers0']), ('.seeds', p['answers'])] if p['magic'] == b'FPENTC01' else [('.seeds', p['answers'])]
        seeds = [read_seeds(load_beside(names[0] + ending), 'the seeds file ' + ending) for ending, _ in tables]
        for (ending, answers), found, rows in zip(tables, seeds, p['stored']):
            check_seeds(found, answers, rows, 'the seeds file ' + ending)
        said = ['%s, %d bytes' % (p['magic'].decode('ascii'), len(data)), 'rounds ' + ' | '.join(str(len(c)) for c in p['chains']), '%d roots sent' % len(sent),
                '%d trees opened at %d leaves' % (len(p['trees']), sum(len(t['rows']) for t in p['trees'])), 'weight tables opened at %s leaves' % ' | '.join(str(len(s)) for s in p['stored']), 'seeds ' + ' | '.join(str(len(s)) for s in seeds)]
        if 'config' in opts:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'format'))
            import format_rules
            configuration = format_rules.configuration_digest([load(os.path.join(opts['config'], n)) for n in ('policy.txt', 'layout.txt', 'KL.bin')])
            said.append('%d whole paths lead to their roots' % check_paths(p, sent, format_rules, configuration))
        if 'file' in opts:
            original = load(opts['file'])
            ask(original_again(p, read_order(load_beside(names[0] + '.order')), len(original)) == original, 'the parsed file and its order file do not give the original file again')
            said.append('with its order file it gives the original again, %d bytes' % len(original))
    except Refused as e:
        print('REFUSED: %s' % e)
        return 1
    except (KeyError, OverflowError, struct.error, AssertionError):  # AssertionError: how format_rules.py refuses a value
        print('REFUSED: the parsed file and the files beside it do not fit each other or the format')
        return 1
    except MemoryError:
        print('REFUSED: the files need more memory than this machine gives')
        return 1
    except OSError as e:
        print('read_parsed.py: %s cannot be read: %s' % (e.filename, e.strerror), file=sys.stderr)
        return 2
    print('READ: ' + '; '.join(said))
    return 0


if __name__ == '__main__':
    sys.exit(main())
