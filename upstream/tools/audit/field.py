#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Arithmetic in E, and codeword symbols computed by tools/audit/fieldlib.cpp, for the audit
# (tools/audit/README.md).
# F = Z / P[field], E = F[X] / (X^3 - NONRES[field]) (docs/format/FORMAT.md, R1.4).
import ctypes
import os

import numpy as np

P = {1: (1 << 61) - 1, 2: (1 << 64) - (1 << 32) + 1}
NONRES = {1: 5, 2: 2}
_lib = None


def lib():
    global _lib
    if _lib is None:
        here = os.path.dirname(os.path.abspath(__file__))
        so = os.environ.get('AUDIT_LIB', os.path.join(here, 'libfield.so'))
        _lib = ctypes.CDLL(so)
        _lib.row_symbols.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint] + [ctypes.c_void_p] * 3 + [ctypes.c_size_t, ctypes.c_void_p]
    return _lib


def row_symbols(field, code, deg, msg, tail, xs):
    """Returns one row's codeword symbol, deg numbers, for each position in xs.

    code is (a, ell, r): length 2^a, ell message entries, r random-tail entries.
    msg and tail are flat, each entry's deg coordinates (3 for E) together; None means zeros.
    """
    a, ell, r = code
    m = None if msg is None else np.ascontiguousarray(np.array(msg, dtype=np.uint64))
    t = None if tail is None else np.ascontiguousarray(np.array(tail, dtype=np.uint64))
    assert m is None or len(m) == ell * deg
    assert t is None or len(t) == r * deg
    x = np.ascontiguousarray(np.array(xs, dtype=np.uint64)); out = np.zeros(len(xs) * deg, dtype=np.uint64)
    lib().row_symbols(field, a, ell, r, deg, None if m is None else m.ctypes.data, None if t is None else t.ctypes.data, x.ctypes.data, len(xs), out.ctypes.data)
    return out.reshape(len(xs), deg).tolist()


class E:
    """Arithmetic on coefficient triples, constant term first."""

    def __init__(s, field):
        s.p, s.c = P[field], NONRES[field]
        s.zero, s.one = (0, 0, 0), (1, 0, 0)

    def add(s, a, b): return ((a[0] + b[0]) % s.p, (a[1] + b[1]) % s.p, (a[2] + b[2]) % s.p)
    def sub(s, a, b): return ((a[0] - b[0]) % s.p, (a[1] - b[1]) % s.p, (a[2] - b[2]) % s.p)

    def mul(s, a, b):
        p, c = s.p, s.c
        return ((a[0] * b[0] + c * (a[1] * b[2] + a[2] * b[1])) % p, (a[0] * b[1] + a[1] * b[0] + c * a[2] * b[2]) % p, (a[0] * b[2] + a[1] * b[1] + a[2] * b[0]) % p)

    def inv(s, a):
        # a^(p^3 - 2); zero gives zero
        r, e, b = s.one, s.p ** 3 - 2, a
        while e:
            if e & 1: r = s.mul(r, b)
            b = s.mul(b, b); e >>= 1
        return r
