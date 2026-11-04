#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Compares fieldlib.cpp's row_symbols() at field 2 (Goldilocks), on synthetic base-field rows only,
# with format_rules.symbol() (docs/format/FORMAT.md, R12.11).
# Also checks field.E.inv() on one triple. Needs libfield.so (field.lib()).
import sys, os, random
sys.dont_write_bytecode = True
# format_rules.py is imported only from the repository that this file is in.
tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if len(sys.argv) != 2 or os.path.realpath(sys.argv[1]) != tree: sys.exit('test_field.py: the argument must be the repository that this file is in: ' + tree)
sys.path.insert(0, tree + '/docs/format'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_rules as R, field as FD
random.seed(5); bad = 0
for a, ell, r in ((4, 5, 3), (6, 16, 7), (10, 300, 44), (13, 64, 156)):
    m = [random.randrange(FD.P[2]) for _ in range(ell)]; t = [random.randrange(FD.P[2]) for _ in range(r)]
    xs = [random.randrange(1 << a) for _ in range(9)] + [0, 1, (1 << a) - 1]
    got = FD.row_symbols(2, (a, ell, r), 1, m, t, xs)
    bad += sum(g[0] != R.symbol(a, m, t, x) for g, x in zip(got, xs))
print('fieldlib at field 2 against format_rules.symbol: %s' % ('the same at all positions' if not bad else '%d DIFFER' % bad))
E = FD.E(2); x = (3, 5, 7); assert E.mul(x, E.inv(x)) == E.one
E = FD.E(1); assert E.mul(x, E.inv(x)) == E.one
sys.exit(1 if bad else 0)
