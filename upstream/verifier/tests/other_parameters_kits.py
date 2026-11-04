#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by verifier/tests/other_parameters_worlds.sh, which the runner starts; by hand: below a directory of your own.
"""Writes variants of a configuration for other_parameters_test.py. Each changes one number;
`target-lower` changes two.

usage: other_parameters_kits.py <config folder: policy.txt, KL.bin> <folder to write into>

Each variant ("sister" in the test's messages) gets a folder <name>/ with policy.txt and KL.bin,
and no layout.txt: other_parameters_worlds.sh makes that file from the two (cp_verify -plan arrays,
then prover/tools/pack_layout.py). If <name>/ exists already, the script exits and overwrites
nothing. The programs (cp_verify, cp_commit, cp_prove) reject some variants of some configurations
when they read the files: for example, a target that some round misses, or a code too short for its
random tail. other_parameters_worlds.sh leaves these out and prints how many. The test is given the
least number of variants that must have been made. SUITE.md states both numbers.

`norm-constant`, `last-norm-constant` and `scale-of-one-layer` change the function that is proved,
the others a parameter of the proof system."""
import os
import stat
import struct
import sys

# FRAME, FUNCTION: (name, start of a line of policy.txt, index of the number to change, the change).
# A `work` line: bits of proof of work in a request's own rounds, a batch's shared rounds, the
# committer's rounds. Where they are 0, as in prover/tests/configurations/pinned_no_proof_of_work,
# the `work-*-other` variants have four bits more.
# A `group` line: index 1 is log2 of its code length; index 2 of a
# `stages` line: the same for the next stage of its proximity test.
# Outside the standard parameter form, `cp_verify` ends at best with `NOT-A-VERIFICATION`.
other_work = lambda v: v - 4 if v >= 4 else v + 4  # four bits fewer; below four bits, four more
FRAME = [
    ("work-positions-other", "work positions", 0, other_work),
    ("work-positions-more", "work positions", 0, lambda v: v + 2),
    ("work-shared-other", "work positions", 1, other_work),
    ("work-committer-other", "work positions", 2, other_work),
    ("work-sample", "work sample", 0, lambda v: v + 4),
    ("slots-half", "slots", 0, lambda v: v // 2),
    ("salt-longer", "salt", 0, lambda v: v + 8),
    ("link-rate", "group link", 1, lambda v: v + 1),
    ("link-schedule", "stages link", 2, lambda v: v + 1),
    ("folded-rate", "group folded", 1, lambda v: v + 1),
    ("own-schedule", "stages own", 2, lambda v: v + 1),
    ("once-folded-rate", "group once-folded", 1, lambda v: v + 1),
    ("ranges-part", "ranges-part", 0, lambda v: v - 1),
]
FUNCTION = [
    ("norm-constant", "eps1_lo", 0, lambda v: v + 1),
    ("last-norm-constant", "epsF_lo", 0, lambda v: v + 1),
]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    config, out = sys.argv[1], os.path.abspath(sys.argv[2])
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.mkdir(out, 0o700)  # a new folder is closed to other users
    except FileExistsError:
        found = os.lstat(out)  # its permission bits are not looked at: the script that starts this one has just made it
        if not stat.S_ISDIR(found.st_mode) or found.st_uid != os.geteuid():
            sys.exit("other_parameters_kits: nothing is written into %s: it is a link, or not a folder, or belongs to another account: name another folder" % out)
    lines = open(os.path.join(config, "policy.txt")).read().split("\n")
    scales = open(os.path.join(config, "KL.bin"), "rb").read()

    def lay(name, text, words):
        try:
            os.mkdir(os.path.join(out, name), 0o700)  # fails if the name is taken: no file of an earlier call is overwritten
        except FileExistsError:
            sys.exit("other_parameters_kits: %s exists already and is not overwritten: name another folder to write into" % os.path.join(out, name))
        open(os.path.join(out, name, "policy.txt"), "w").write(text)
        open(os.path.join(out, name, "KL.bin"), "wb").write(words)
        print(name)

    def changed(text, head, at, change):
        found = [i for i, l in enumerate(text) if l.startswith(head + " ")]
        if len(found) != 1:
            sys.exit("other_parameters_kits: the parameter file has %d lines '%s'" % (len(found), head))
        numbers = text[found[0]][len(head) + 1:].split()
        numbers[at] = str(change(int(numbers[at])))
        return text[:found[0]] + [head + " " + " ".join(numbers)] + text[found[0] + 1:]

    for name, head, at, change in FRAME + FUNCTION:
        lay(name, "\n".join(changed(lines, head, at, change)), scales)
    # `tau`, the number of combinations in the approximate range check, must be the target (`sec`)
    # less that check's bits of proof of work (`work smallness`), and a multiple of 32.
    lay("target-lower", "\n".join(changed(changed(lines, "sec", 0, lambda v: v - 32), "tau", 0, lambda v: v - 32)), scales)
    # The fifth number of the line table is the constant 0. This variant sets another value; the
    # planning call rejects it, so the set-ups leave it out.
    table = [int(x) for l in lines if l.startswith("table ") for x in l.split()[1:]]
    lay("table-radius", "\n".join(changed(lines, "table", 4, lambda v: ((1 << table[1]) - table[2] - table[3]) // 3 + 2000)), scales)
    # `KL.bin` starts with `kq`, the first layer's right shift after the query projection. In the
    # example configurations the line `hq` allows `kq` + 1.
    first = struct.unpack_from("<I", scales)[0]
    lay("scale-of-one-layer", "\n".join(lines), struct.pack("<I", first + 1) + scales[4:])


main()
