#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Gives cp_verify, the verifier, honest files from which a whole opening of a Merkle tree is left out.

usage: openings_left_out_test.py <the tree's folder> <unit.o> <cp_verify> <set> <new work folder> [<calls at a time>]
(each word, what stays in the work folder, and the exit status: changed_sets.py, beside this file)

An opening (columns, salts and the digests of the paths to a tree's root, at the positions that a
round has sampled) enters no query of a hash chain. So no later challenge changes if one is missing:
only the verifier's own reading of the file, opening by opening, can notice it.
For each of key file, leaf file and proof file, and each opening in it, cp_verify gets the set with
that file changed:
  left out          the opening's bytes are gone, and the bytes after them follow directly;
  laid at the end   the same, with the opening's bytes appended, so that the length stays
                    (not for an opening that ends the file: that copy would be the honest file);
and once with all openings left out but the first of each round, if that leaves out several.
Each call is a whole call under the set's ID. It must end with status 1 and the last line
"NO: key-refused", "NO: entry-refused" or "NO: proof-refused", by the file that was changed.
The control: the unchanged set must end with status 0.
Not tested: a part of an opening left out or changed (proof_fields_sweep_test.py and
hostile_leaf_files_test.py change single places); files under another parameter file than the set's; that cut.py finds every opening (it asks that no byte of a
file is left over, and the count of openings is printed)."""
import sys

sys.dont_write_bytecode = True  # before the import: no __pycache__ beside the tests
import changed_sets

s = changed_sets.begin(__doc__)


def without(path, ranges, at_the_end=False):
    whole, kept, upto = s.whole[path], b"", 0
    for a, b in ranges:
        kept, upto = kept + whole[upto:a], b
    return kept + whole[upto:] + (b"".join(whole[a:b] for a, b in ranges) if at_the_end else b"")


cases = []
for path in s.whole:
    found = [(o[0].off, o[-1].off + o[-1].n, (o[0].chain, o[0].rnd)) for o in changed_sets.openings(s.records[path])]
    further = [(a, b) for i, (a, b, in_round) in enumerate(found) if i and found[i - 1][2] == in_round]
    short = path.split("/")[-1]
    for i, (a, b, _) in enumerate(found):
        cases.append((path, "%s.%03d.left-out" % (short, i), lambda path=path, a=a, b=b: without(path, [(a, b)]), None))
        if b < len(s.whole[path]):
            cases.append((path, "%s.%03d.laid-at-the-end" % (short, i), lambda path=path, a=a, b=b: without(path, [(a, b)], True), None))
    if len(further) > 1:
        cases.append((path, "%s.all-but-each-rounds-first.left-out" % short, lambda path=path, further=further: without(path, further), None))
    print("%s: %d bytes, %d openings, %d of them not the first of their round" % (path, len(s.whole[path]), len(found), len(further)))
changed_sets.play(s, cases)
