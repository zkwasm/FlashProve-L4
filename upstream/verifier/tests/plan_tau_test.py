#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Tests that cp_verify rejects a `tau` that differs from the plan's number.

The parameter file (`policy.txt`) fixes in two ways how many random combinations the approximate
range check (`smallness` in printed lines) uses in a sub-proof. The line `tau` sizes the sub-proof's
arrays and random matrix. In the plan (the list of rounds) the matrix's round has error 2^-N,
N = rng_combinations() (blocks/numbers/blocks.h): in cp_verify, the target (line `sec`) minus that
round's bits of proof of work (first number of `work smallness`). If N and `tau` differed, the error
would be computed for more combinations than are drawn, or for fewer: planRounds() in
verifier/src/vf_smallness_test_members.inc rejects such a file (`plan-tau`). Checks before it reject a `tau` other
than 32, 64, 96, 128, 160, 192 or 256 (`hid-tau`) or below `sec` (`policy-tau-hb`).

Each case is a planning call (`cp_verify -plan accounting`) on a copy of the given configuration
(`kit` in identifiers and messages) with `tau`, `work smallness`, both or neither altered; no proof
is read. A `tau` 32 or 64 above `sec` changes the arrays' sizes, so the test makes a fitting
`layout.txt`: the given one would be rejected before planRounds() (`plan-layout`).

usage: plan_tau_test.py <cp_verify> <kit folder> <new work folder>
Exit status 0: all as expected. 1: not all, or no `layout.txt` could be made. 2: a wrong call
or configuration. The script makes <new work folder> and missing parents, writes only there and
deletes nothing.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "..", "..", "prover", "tools", "pack_layout.py")


def call(words):
    return subprocess.run(words, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False)


def the_cases(verifier, kit, work):
    """Makes the work folder and, in it, a copy of the configuration for each case.

    Returns sec, the field's number and the cases, or None for an unsuitable configuration. A case:
    name, what is altered, the copy's path, the expected rejection name or "planned", N or None."""
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    with open(os.path.join(kit, "policy.txt"), encoding="ascii") as f:
        good = f.read().split("\n")
    one = {x.split(" ")[0]: x.split(" ")[1] for x in good if len(x.split(" ")) == 2}
    sec, tau = int(one["sec"]), int(one["tau"])
    if sec != tau or "work smallness 0 0 0" not in good:
        print("plan_tau_test.py: the kit must have tau = sec and no work on smallness", file=sys.stderr)
        return None
    # log2 of a row's message length, for pack_layout.py
    message = [x.split(" ")[4] for x in good if x.startswith("group link ")][0]
    field = one["version"][2]

    def bent(name, work_line=None, new_tau=None, own_layout=False):
        """Copies the configuration's three files, with the given lines of policy.txt replaced.
        With own_layout, layout.txt is made by prover/tools/pack_layout.py from this verifier's list
        of arrays."""
        folder = os.path.join(work, name)
        os.makedirs(folder)
        lines = ["work smallness " + work_line if work_line and x.startswith("work smallness ") else "tau %d" % new_tau if new_tau and x.startswith("tau ") else x for x in good]
        with open(os.path.join(folder, "policy.txt"), "w", encoding="ascii") as f:
            f.write("\n".join(lines))
        shutil.copy(os.path.join(kit, "KL.bin"), folder)
        if not own_layout:
            shutil.copy(os.path.join(kit, "layout.txt"), folder)
            return folder
        r = call([verifier, "-plan", "arrays", folder, str(sec)])
        if r.returncode != 4:
            sys.exit("plan_tau_test.py: %s: the planning call on the arrays ended with status %d" % (name, r.returncode))
        with open(folder + ".arrays.txt", "w", encoding="ascii") as f:
            f.write(r.stdout)
        made = subprocess.run([sys.executable, "-B", PACK, "--arrays", folder + ".arrays.txt", message, "4"], capture_output=True, text=True, check=False)
        if made.returncode:
            sys.exit("plan_tau_test.py: pack_layout.py: " + made.stderr.strip()[-200:])
        with open(os.path.join(folder, "layout.txt"), "w", encoding="ascii") as f:
            f.write(made.stdout)
        return folder

    cases = [("kit", "nothing", bent("kit"), "planned", sec)]
    for w in ("1 1 1", "4 4 4", "32 32 32", "4 0 0"):
        cases.append(("work-" + w.replace(" ", "-"), "work smallness %s, tau %d" % (w, tau), bent("work-" + w.replace(" ", "-"), w), "plan-tau", sec - int(w.split(" ")[0])))
    cases.append(("work-0-4-4", "work smallness 0 4 4: work in the rounds of a batch and of the committer, which have no smallness test", bent("work-0-4-4", "0 4 4"), "planned", sec))
    for t in (sec + 32, sec + 64):
        cases.append(("tau-%d" % t, "tau %d, with a layout file made for it" % t, bent("tau-%d" % t, None, t, True), "plan-tau", sec))
    cases.append(("tau-%d" % (sec - 32), "tau %d" % (sec - 32), bent("tau-%d" % (sec - 32), None, sec - 32), "policy-tau-hb", None))
    for t in (sec - 1, sec + 1):
        cases.append(("tau-%d" % t, "tau %d" % t, bent("tau-%d" % t, None, t), "hid-tau", None))
    cases.append(("work-4-tau-%d" % (sec - 4), "work smallness 4 4 4, tau %d" % (sec - 4), bent("work-4-tau", "4 4 4", sec - 4), "hid-tau", None))
    cases.append(("work-32-tau-%d" % (sec - 32), "work smallness 32 32 32, tau %d" % (sec - 32), bent("work-32-tau", "32 32 32", sec - 32), "policy-tau-hb", None))
    return sec, field, cases


def play(program, sec, cases, who=""):
    """Makes each case's planning call and prints the outcome after `who`. Returns how many did
    not end as expected (exit status 4: planned; 1: rejected)."""
    bad = 0
    for name, what, folder, want, number in cases:
        r = call([program, "-plan", "accounting", folder, str(sec)])
        no = [x[4:].split(" ")[0] for x in r.stdout.split("\n") if x.startswith("NO: ")]
        said = re.findall(r"^smallness test: (\d+) combinations$", r.stdout, re.M)
        if want == "planned":
            ok = r.returncode == 4 and not no and said == [str(number)]
        else:
            ok = r.returncode == 1 and no == [want]
        bad += not ok
        print("%s%s: %s: status %d, %s; expected: %s: %s" % (who, name, what, r.returncode,
              "NO: " + ", ".join(no) if no else "planned with %s combinations" % ", ".join(said), want if want != "planned" else "planned with %d combinations" % number,
              "as expected" if ok else "not as expected"))  # fmt: skip
    return bad


def main():
    a = sys.argv[1:]
    if len(a) != 3 or os.path.lexists(os.path.abspath(a[2])):
        print(__doc__[__doc__.index("usage: ") : __doc__.index("Exit status")].rstrip(), file=sys.stderr)
        return 2
    verifier = os.path.abspath(a[0])
    laid = the_cases(verifier, a[1], os.path.abspath(a[2]))
    if laid is None:
        return 2
    sec, _, cases = laid
    bad = play(verifier, sec, cases)
    print("plan_tau_test.py: %d cases: %s" % (len(cases), "all as expected" if not bad else "%d not as expected" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
