#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Lines iota_0 .. iota_7 that a parameter file must state, for tests that change a model's shape.

A layer sub-proof (a proof has one per group of 2^ll layers) makes one lookup call per `call` line:
each shows that every key in its segments (named lists) occurs in a public table. The calls' keys
are split into trees of 2^dT keys, the lookup forest. Call q has 2^n_q keys in 2^iota_q trees;
dT is the smallest of the n_q and `lidb` (22 if absent). The verifier `cp_verify` requires
iota_q = n_q - dT, or 0 if there is no call q, and rejects other values as `shape-call-<name>` or
`header-absent-call`. It does so in every stage (key, leaf, request), in layout()
(verifier/src/cp_verify.cpp), after scanning the file. So a test that changes a `call` line
or f, NE, d, H, G, dh, xl, ll, lt, nmin or lidb must recompute these lines to reach a later check.

This file imports nothing from the repository: it restates layout()'s segment lengths, so a change
there must be repeated here. It covers the layer sub-proof only: the output head's lines iotaH_q and
attention's iotaA_0 depend on some of the same values (`lidb` for one) and are left to the test.

usage: forest_lines.py [<top directory>]
Compares layer() with the parameter files of the list FILES, without running `cp_verify`. The list
is written out, so that a file that goes missing is reported and not passed over. Every other
policy*.txt under prover/examples/, prover/tests/, verifier/tests/ or the integer model's tests is
compared too. blocks/frame/examples/ is not walked: its files end
before the lines on a model and have none of the lines that layer() is about.
It also compares layer() with the typed lines of SHAPES. The file SMALL belongs to the tests of
integrations/vllm/integer_model. Do not copy or edit it. A case that needs another value sets it
here.
As a module: calls_of(text), keys(L, calls), layer(L, calls). L: dict, a line's name -> its integer.
Exit code 0: every listed file was read and neither a file nor a typed shape differs. 1: one
differs or a listed file is missing.
"""

import glob
import os
import sys

SMALL = "integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/policy.txt"
FILES = (
    SMALL,
    "prover/examples/target_full_f2/policy.txt",
    "prover/examples/target_small_f2/policy.txt",
    "prover/examples/target_small_f2_gap/policy.txt",
    "prover/examples/target_small_f2_roomy/policy.txt",
    "prover/examples/target_small_file_f2/policy.txt",
    "prover/tests/configurations/pinned/policy.txt",
    "prover/tests/configurations/pinned_no_proof_of_work/policy.txt",
    "prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt",
)


# SMALL's lines with another nmin, and iota_0 .. iota_7 as typed here. No parameter file has either shape.
# At nmin 11, iota_6 and iota_7 are above 0, as in no parameter file: n_q is 13, 14, 14, 15, 13, 11, 12, 12
# and the depth dT is 11. The shape at nmin 12 also stands in verifier/tests/forest_values.txt, which
# verifier/tests/known_answers/forest_values.py writes without this file.
SHAPES = ((11, (2, 3, 3, 4, 2, 0, 1, 1)), (12, (1, 2, 2, 3, 1, 0, 0, 0)))


def bits_for(x):
    """The smallest e with 2^e >= x, for x >= 1."""
    return (x - 1).bit_length()


def calls_of(text):
    """The `call` lines of a parameter file's text, in order, each as (name, [segment names])."""
    return [(w[1], w[3:]) for w in (x.split(" ") for x in text.split("\n")) if w[0] == "call"]


def length(segment, L):
    """The number of columns (keys per layer and token) that a segment takes, as in layout()."""
    dkv = L["G"] * L["dh"]
    nqk = L["H"] * L["dh"] + dkv
    if segment == "ROW":
        return 2 * (99 if L.get("xl") == 5 else 74)  # 2 norms' rows: 99 digits at 5 limbs, 74 at 4
    if segment == "MR":
        return 1
    for names, n in (("G U UT GH UH G0 F CG", L["f"]), ("QK QKH CQK", nqk), ("N1 N1H N2 N2H O OH D DH XR CN1 CN2", L["d"]), ("V VH CV", dkv),
                     ("LR LT DD LH SEL L3 CLR", L["NE"])):  # fmt: skip
        if segment in names.split(" "):
            return n
    raise ValueError("no such segment: " + segment)


def keys(L, calls):
    """n_q per call: 2^ll layers x 2^lt tokens x J columns make 2^n_q keys.

    J: the call's segments' columns, summed, rounded up to a power of two, then raised until
    n_q >= nmin.
    """
    return [max(L["ll"] + bits_for(sum(length(s, L) for s in segments)) + L["lt"], L["nmin"]) for _, segments in calls]


def layer(L, calls):
    """The eight lines as a dict: iota_q = n_q - dT for each call in `calls`, 0 for the rest.

    Calls after the eighth get no line.
    """
    n = keys(L, calls)
    depth = min([L.get("lidb", 22)] + n)
    return {"iota_%d" % q: n[q] - depth if q < len(n) else 0 for q in range(8)}


def values_of(text):
    """The lines of a parameter file's text of the form name and number, as a dict."""
    return {w[0]: int(w[1]) for w in (x.split(" ") for x in text.split("\n")) if len(w) == 2 and w[1].isdigit()}


def main():
    tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    files = sorted(f for top in ("integrations/vllm/integer_model/tests", "prover/examples", "prover/tests", "verifier/tests") for f in glob.glob(tree + "/" + top + "/**/policy*.txt", recursive=True))
    missing = sorted(set(FILES) - {os.path.relpath(f, tree) for f in files})
    for f in missing:
        print("%s: on this script's list, and no such file" % f)
    print("forest_lines.py: %d of the %d listed files read, and %d others" % (len(FILES) - len(missing), len(FILES), len(files) - len(FILES) + len(missing)))
    differ = 0
    for path in files:
        with open(path, encoding="ascii") as f:
            text = f.read()
        L = values_of(text)
        mine = layer(L, calls_of(text))
        other = {k: (L[k], v) for k, v in mine.items() if L[k] != v}
        differ += bool(other)
        if other:
            print("%s: %s" % (os.path.relpath(path, tree), "; ".join("%s is %d, by the rule %d" % (k, a, b) for k, (a, b) in other.items())))
    compared = wrong = 0
    if SMALL not in missing:
        with open(os.path.join(tree, SMALL), encoding="ascii") as f:
            text = f.read()
        for nmin, typed in SHAPES:
            mine = tuple(layer(dict(values_of(text), nmin=nmin), calls_of(text))["iota_%d" % q] for q in range(8))
            compared += 1
            wrong += mine != typed
            if mine != typed:
                print("forest_lines.py: at nmin %d layer() gives %s, and SHAPES has %s" % (nmin, mine, typed))
    print("forest_lines.py: %d typed shapes, %d differ" % (compared, wrong))
    print("forest_lines.py: %d parameter files, %d differ" % (len(files), differ))
    return 1 if differ or wrong or missing else 0


if __name__ == "__main__":
    sys.exit(main())
