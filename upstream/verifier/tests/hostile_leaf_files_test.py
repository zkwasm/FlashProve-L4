#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Gives cp_verify, the verifier, leaf files that a hostile sender could make from an honest one.

usage: hostile_leaf_files_test.py <the tree's folder> <unit.o> <cp_verify> <set> <new work folder> [<calls at a time>]
(each word, what stays in the work folder, and the exit status: changed_sets.py, beside this file)

A leaf file may come from anyone (verifier/docs/parameter-file.md). Each case changes the set's
ivk/leaf/0.bin in one place:
  a word that encodes no element of the field, p, p + 1 or 2^64 - 1, written over the first word and
  over the last word of each kind of message, of the record's stored answers, and of each opening's
  columns; and p - 1, the largest word that does encode one, as the control;
  the words p, 0, 0 over the first and the last element of each round's message and of the stored
  answers (there they are zero of the extension field, written in another way), and over the first
  three words of each opening's first and middle column;
  the salt of each opening's first and last column one byte shorter, or one zero byte longer, with
  the bytes after it moved;
  a whole opening cut out.
cp_verify gets each file twice, in the part-call -leaf and in a whole call, both under the set's ID.
Each call must end with status 1 and the last line "NO: entry-refused".
A word at or above p must be refused AS SUCH, and not only because it changes what a chain or a leaf
hashes: the line before the last, in which cp_verify says why, must hold "canonical" or, for the
stored answers, "the record's answers"; for the control it must not.
The control on the calls: the unchanged set must end with status 3 (-leaf) and 0.
Not tested: other places than the first and last of each kind; other parameter files."""
import struct
import sys

sys.dont_write_bytecode = True  # before the import: no __pycache__ beside the tests
import changed_sets
from changed_sets import LEAF

s = changed_sets.begin(__doc__)
whole, records = s.whole[LEAF], s.records[LEAF]
WORDS = (("p", s.prime), ("p-plus-1", s.prime + 1), ("all-ones", 2**64 - 1), ("p-minus-1", s.prime - 1))
cases = []


def case(name, why, at, new, taken=None):
    """new is written at byte `at`, over as many bytes as it has, or over `taken` bytes."""
    cases.append((LEAF, name, lambda: whole[:at] + new + whole[at + (len(new) if taken is None else taken) :], why))


def words(name, why, first, last):
    for place, at in (("first-word", first.off), ("last-word", last.off + last.n - 8)):
        for word, value in WORDS:
            case("%s.%s.%s" % (name, place, word), ("not " if value < s.prime else "") + why, at, struct.pack("<Q", value))


P_0_0 = struct.pack("<3Q", s.prime, 0, 0)
sent = [r for r in records if r.tree is None and r.cls not in ("head", "stored-root", "root", "nonce", "chain-bytes")]  # the five: a leaf file's records outside openings that hold no field elements
assert sent and all(r.n % 24 == 0 for r in sent), "a message's record is no list of elements of the extension field"
told = lambda r: "the record's answers" if r.rnd is None else "canonical"  # noqa: E731
kinds = list(dict.fromkeys((r.cls, r.sub) for r in sent))
for i, kind in enumerate(kinds):
    of_it = [r for r in sent if (r.cls, r.sub) == kind]
    words("message-%02d.%s" % (i, kind[0]), told(of_it[0]), of_it[0], of_it[-1])
for in_round in dict.fromkeys(r.rnd for r in sent):
    of_it = [r for r in sent if r.rnd == in_round]
    name, first, last = "stored-answers" if in_round is None else "round-%02d" % in_round, of_it[0].off, of_it[-1].off + of_it[-1].n - 24
    case(name + ".first-element.p-0-0", told(of_it[0]), first, P_0_0)
    if last != first:
        case(name + ".last-element.p-0-0", told(of_it[0]), last, P_0_0)
for i, opening in enumerate(changed_sets.openings(records)):
    columns, salts = [r for r in opening if r.cls.endswith("symbol")], [r for r in opening if r.cls in ("salt", "stored-salt")]
    words("opening-%d" % i, "canonical", columns[0], columns[-1])
    starts = list(dict((r.x, r.off) for r in reversed(columns)).values())[::-1]  # each column's first byte, in the file's order
    assert salts[0].off - starts[0] >= 24, "a column has fewer than three words"
    for place, at in (("first-column", starts[0]), ("middle-column", starts[len(starts) // 2])):
        case("opening-%d.%s.p-0-0" % (i, place), "canonical", at, P_0_0)
    for place, salt in (("first-column", salts[0]), ("last-column", salts[-1])):
        case("opening-%d.%s.salt-one-byte-shorter" % (i, place), None, salt.off + salt.n - 1, b"", 1)
        case("opening-%d.%s.salt-one-byte-longer" % (i, place), None, salt.off + salt.n, b"\0", 0)
    case("opening-%d.cut-out" % i, None, opening[0].off, b"", opening[-1].off + opening[-1].n - opening[0].off)
print("%s: %d bytes, %d kinds of message (the stored answers among them), %d openings" % (LEAF, len(whole), len(kinds), i + 1))
changed_sets.play(s, cases, True)
