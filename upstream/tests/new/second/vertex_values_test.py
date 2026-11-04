#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests that three of this directory's verifiers reject two values at one Merkle-tree vertex.

No input file can break only this rule (docs/format/FORMAT.md, R7.14): it would contain a
hash collision. So, before a verifier's first opening, the test writes 32 zero bytes into
`toy_check.Run.values`, the record of vertex values in all three, at that opening's first leaf.
The honest file's rejection message must start with `WORDS`. No C++ verifier is tested.

In the key file some trees are opened twice, so there the rule compares values:
tests/new/second/files_check.py --spoil-vertex-record overwrites the record of two of them
between openings.

usage: vertex_values_test.py <directory of format_rules.py> <files>
<directory of format_rules.py>: appended to sys.path by `toy_check`
<files>: honest-and-altered/, filled by tests/new/second/run_honest.sh
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
import links_check as lk  # noqa: E402
import entry_check as ec  # noqa: E402

WORDS = "a vertex of a tree gets two values under one root"


def play(name, run):
    honest, planted = tc.Run.opening, []

    def opening(self, tree, depth, width, root, draws, stored=False):
        if not planted:
            planted.append(True)
            self.values.setdefault(root, {})[(1 << depth) + min(draws)] = bytes(32)
        return honest(self, tree, depth, width, root, draws, stored)

    tc.Run.opening = opening
    try:
        run()
        said = "ACCEPTED"
    except tc.Refused as why:
        said = str(why)
    tc.Run.opening = honest
    print("%s, with another value planted at the first leaf that is opened: %s" % (name, said))
    return said.startswith(WORDS)


def main():
    top = sys.argv[2]
    rd = lambda p: open("%s/%s" % (top, p), "rb").read()  # noqa: E731
    ok = play("product, tiny", lambda: tc.verify(tc.parse_parameters(rd("product/tiny.parameters")), rd("product/tiny.parameters"), rd("product/tiny.configuration"), rd("product/tiny.proof"), []))
    ok &= play("links, tiny", lambda: lk.verify(lk.parse(rd("links/tiny-links.parameters")), rd("links/tiny-links.parameters"), rd("links/tiny-links.configuration"), rd("links/tiny-links.proof"), []))
    ok &= play("the entry", lambda: ec.verify(ec.parse(rd("key-and-entry/entry.parameters")), rd("key-and-entry/entry.parameters"), rd("key-and-entry/entry.statement"), rd("key-and-entry/entry.file"), []))
    print("THE RECORD OF VALUES AT VERTICES IS KEPT AND READ" if ok else "THE RECORD OF VALUES AT VERTICES IS NOT READ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
