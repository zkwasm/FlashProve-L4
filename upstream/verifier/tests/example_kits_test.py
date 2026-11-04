#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Checks cp_verify's planning call on prover/examples' parameter files and one of the tests' own.

usage: example_kits_test.py <cp_verify> <new folder> [<the build's field, 1 or 2; without it 1>]
status 0: all as expected; 1: not; 2: wrong arguments, or <new folder> exists

The tests' own is prover/tests/configurations/pinned_no_proof_of_work/policy.txt (no proof of work).
Each parameter file prover/examples/*/policy*.txt (`policy` in identifiers) is written, unchanged or
with lines changed or added, as policy.txt of a copy of its directory, and the script runs
cp_verify "-plan accounting <copy> 128", which prints the error accounting of a verification call
with these files (128 requested bits; `asked` in code and output). No proof is generated or checked.
A completed planning call ("PLAN ONLY") exits with status 4, a rejection ("NO: <name>"; "refused"
in this script's output) with status 1. The script creates <new folder> (mode 0700), with missing
parent directories, and leaves it in place; it removes only these copies, which it makes there.
The lines named in steps 1 to 3 are defined in blocks/frame/parameters.h, those of step 6 in
verifier/docs/parameter-file.md, "8. Output head and acceptance policy".
1. A file's kind is the first whose rule applies; after the colon, the unchanged file's expected
   outcome:
   field     the first line is not "version H8<field>1": NO: version-unknown (policy).
             The only field is number 2, p = 2^64 - 2^32 + 1 (blocks/base.h): pass 2 as <field>, the
             third argument
   bare      the directory has no layout.txt, which prover/tools/pack_layout.py generates before
             use: NO: plan-layout
   work      a line "work .." has a number above 0, which asks for proof of work: status 4, one
             line "work: <pair> (.." and one "accounting: THE ROUNDS: HOLD WITH WORK: <pair>: .."
             with the same pair "<bits> + <work>", whose <bits> is below 128
   yes       every other file: status 4, no line "work:", one "accounting: THE ROUNDS: HOLD: .."
   A file whose first line does not start with "version H8" gets no kind: it is reported as a fault,
   because cp_commit, cp_prove and cp_verify reject it (read_parameters() in
   blocks/frame/parameters.h).
   Steps 2 to 6 use only files of kind work or yes. Unchanged, these must get no line on the form,
   "plan: not the standard parameter form (<conditions missed>): ..", which states that no run with
   these files can be a verification. Two lines, "accounting: one-time files: the
   plan of a key | an entry: ..; the table has a record", must say on which bounds the proximity
   rounds (`closeness` in output) of the committer's files rest.
2. The lines reading and rowcost have one value. With reading 2, and with rowcost 2, the planning
   call must end NO: parameter-range (policy).
   At least one plan of the run must have printed the two lines of step 1.
3. On the unchanged file, a planning call that requests one bit more than the file's target (the
   line "sec") must end NO: security-bits and print one line with both numbers. (Other calls accept
   at most 128 requested bits, these files' target, so here only a planning call reaches this
   rejection.)
4. A lookup call's name (line "call <name> ..") has at most 8 characters, because cp_verify prints
   it at the end of rejection names and the file's author chose it. With a name of 9 characters in
   the first such line, the planning call must end NO: policy-call-line; with one of 8 characters,
   in any other way (the directory's layout.txt may not know the new name).
5. A parameter file of another format must be rejected. With "magic 1295397120", such a file's first
   line, added at the top, the planning call must end NO: parameter-line-order (policy).
6. No verification call ("whole call" in output) passes with a parameter file whose line sT is not
   sh - 4 (cpPolicyHeld() in verifier/src/cp_verify.cpp), and a planning call must say so. With sT
   set to the value of the line s (the acceptance policy's kind is then log-probability) and to one
   less (another-table), each time with C 4096, theta 300 and cap 1, a planning call that ends with
   status 4 must print exactly one line "plan: the policy's kind is <kind>: every whole call under
   this parameter file ends NO: policy"; with any other status it must print none. At least one
   such call must end with status 4. The unchanged files must get no such line.
7. On the tests' own file, a value out of range: with the line record at 0 and at 2, the planning
   call must end NO: parameter-range (policy); with the first number of each line "group" at 0 and
   at 2, NO: parameter-group-shape (policy); with the fifth number of the line table at 1, NO:
   parameter-table-shape (policy). One line for each field says so.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent.parent / "prover" / "examples"
OWN = [pathlib.Path(__file__).resolve().parent.parent.parent / "prover" / "tests" / "configurations" / "pinned_no_proof_of_work" / "policy.txt"]
FORM = "plan: not the standard parameter form ("
ONE_TIME_LINE = r"^accounting: one-time files: the plan of (a key|an entry): (\d+ closeness rounds rest on the theorem on codes \(ePrint 2025/2055, Thm 4\.6\), taken as given, \d+ on the bound within a third of the distance); the table has a record$"
RECORD = "a record"


def rests(out):
    """What a planning call prints for (the key file's plan, the leaf file's): RECORD for each. () if
    it prints no such line. None if it prints some but not exactly one for each, or one in another
    form than ONE_TIME_LINE."""
    found = [m.group(1) for m in (re.match(ONE_TIME_LINE, l) for l in out) if m]
    if len(found) != sum(l.startswith("accounting: one-time files:") for l in out):
        return None  # a line with this beginning in another form
    if not found:
        return ()
    return (RECORD, RECORD) if sorted(found) == ["a key", "an entry"] else None


def main():
    if len(sys.argv) not in (3, 4) or os.path.lexists(os.path.abspath(sys.argv[2])) or sys.argv[3:] not in ([], ["1"], ["2"]):
        print(__doc__.split("\n")[2].strip(), file=sys.stderr)
        return 2
    # `verifier` is made absolute so that a bare file name is not looked up on PATH.
    verifier, work, own = str(pathlib.Path(sys.argv[1]).absolute()), pathlib.Path(os.path.abspath(sys.argv[2])), "version H8%s1" % (sys.argv[3:] or ["1"])[0]
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(mode=0o700)  # fails if the name is taken; closed to other users
    bad, count, played, above = [], {}, 0, 0
    named = 0  # the files whose first lookup call was tried under other names
    warned = 0  # the plans that said that no whole call can pass
    recorded = 0  # the plans that printed the two lines
    WARNS = "plan: the policy's kind is "
    formats = 0  # the files that were tried with the line "magic 1295397120" put first

    def plan(policy, text, name, asked="128"):
        kit = work / name
        shutil.copytree(policy.parent, kit)
        (kit / "policy.txt").write_text(text)
        r = subprocess.run([verifier, "-plan", "accounting", str(kit), asked], capture_output=True, text=True, errors="replace", check=False)
        shutil.rmtree(kit)
        return r.returncode, r.stdout.splitlines() or [""]

    # The kind (step 1) whose expected outcome matches (status, out), or a description if none does.
    def said(status, out):
        if status == 1:
            return {"NO: version-unknown (policy)": "field", "NO: plan-layout": "bare"}.get(out[-1], "status 1, " + out[-1])
        line = [m for m in (re.match(r"^work: (\d+\.\d\d \+ \d+) \(", l) for l in out) if m]
        word = [m for m in (re.match(r"^accounting: THE ROUNDS: (HOLD|HOLD WITH WORK: (\d+\.\d\d \+ \d+)): ", l) for l in out) if m]
        if status != 4 or len(word) != 1 or len(line) > 1:
            return "status %d, %d lines THE ROUNDS, %d lines work:, last line %r" % (status, len(word), len(line), out[-1])
        if not line and word[0][1] == "HOLD":
            return "yes"
        if line and word[0][2] == line[0][1] and float(line[0][1].split()[0]) < 128:
            return "work"
        return "the line work: and the line THE ROUNDS disagree, or the pair's first number is not below 128"

    for n, policy in enumerate(sorted(EXAMPLES.glob("*/policy*.txt")) + OWN):
        name, text = "%s/%s" % (policy.parent.name, policy.name), policy.read_text()
        lines = text.splitlines()
        if not lines or not lines[0].startswith("version H8"):
            bad.append("%s: its first line is not \"version H8..\": cp_commit, cp_prove and cp_verify refuse such a file" % name)
            continue
        if lines[0] != own:
            kind = "field"
        elif not (policy.parent / "layout.txt").exists():
            kind = "bare"
        else:
            kind = "work" if any(re.match(r"^work \w+ (?!0 0 0$)\d+ \d+ \d+$", l) for l in lines) else "yes"
        count[kind] = count.get(kind, 0) + 1
        status, out = plan(policy, text, "kit%d" % n)
        if said(status, out) != kind:
            bad.append("%s: expected %s, got %s" % (name, kind, said(status, out)))
        if kind not in ("work", "yes"):
            continue
        if any(l.startswith(FORM) for l in out):
            bad.append("%s: the standard parameter form is missed" % name)
        if any(l.startswith(WARNS) for l in out):
            bad.append("%s: the plan warns of the policy of an unchanged file" % name)
        recorded += rests(out) == (RECORD, RECORD)
        if rests(out) != (RECORD, RECORD):
            bad.append("%s: the lines on what the committer's files rest are owed; the plan prints %s" % (name, rests(out)))
        targets = [l.split()[1] for l in lines if re.match(r"^sec \d+$", l)]
        if len(targets) != 1:
            bad.append("%s: the file does not have exactly one line sec <bits>" % name)
        else:
            above += 1
            status, out = plan(policy, text, "kit%d.asked" % n, str(int(targets[0]) + 1))
            why = [l for l in out if l.startswith("accounting: asked of this call: ")]
            if status != 1 or out[-1] != "NO: security-bits" or why != ["accounting: asked of this call: %d bits, above the parameter file's target of %s bits" % (int(targets[0]) + 1, targets[0])]:
                bad.append("%s asked for one bit above its target: status %d, %s; on the asked bits: %s" % (name, status, out[-1][:60], why[0][:110] if why else "nothing"))
        formats += 1
        status, out = plan(policy, "magic 1295397120\n" + text, "kit%d.format" % n)
        if (status, out[-1]) != (1, "NO: parameter-line-order (policy)"):
            bad.append("%s with the line \"magic 1295397120\" put first was not refused as parameter-line-order: status %d, %s" % (name, status, out[-1][:60]))
        for line, value in (("reading", 2), ("rowcost", 2)):
            changed, times = re.subn(r"(?m)^%s 1$" % line, "%s %d" % (line, value), text)
            if times != 1:
                bad.append("%s: the file does not have exactly one line %s 1" % (name, line))
                continue
            played += 1
            status, out = plan(policy, changed, "kit%d.%s" % (n, line))
            if (status, out[-1]) != (1, "NO: parameter-range (policy)"):
                bad.append("%s with %s %d: status %d, %s" % (name, line, value, status, out[-1][:60]))
        scale = next((int(l.split()[1]) for l in lines if re.match(r"^s \d+$", l)), None)
        for called, sT in (("log-probability", scale), ("another-table", scale - 1)) if scale else ():
            other = text
            for line, value in (("sT", sT), ("C", 4096), ("theta", 300), ("cap", 1)):
                other = re.sub(r"(?m)^%s \d+$" % line, "%s %d" % (line, value), other)
            status, out = plan(policy, other, "kit%d.%s" % (n, called))
            says = [l for l in out if l.startswith(WARNS)]
            if status == 4 and says == [WARNS + called + ": every whole call under this parameter file ends NO: policy"]:
                warned += 1
            elif status == 4 or says:
                bad.append("%s with sT %d: status %d; on the policy: %s" % (name, sT, status, says[0][:110] if says else "nothing"))
        if policy in OWN:
            groups = [l for l in lines if l.startswith("group ")]
            table = next(l for l in lines if l.startswith("table "))
            words = table.split(" ")
            fields = [("the line record", "parameter-range", [("record 1", "record %d" % v) for v in (0, 2)]),
                      ("a group's first number", "parameter-group-shape", [(l, " ".join(l.split(" ")[:2] + [str(v)] + l.split(" ")[3:])) for l in groups for v in (0, 2)]),
                      ("the table's fifth number", "parameter-table-shape", [(table, " ".join(words[:5] + ["1"] + words[6:]))])]  # fmt: skip
            for field, refusal, turns in fields:
                held = 0
                for i, (old, new) in enumerate(turns):
                    status, out = plan(policy, "\n".join(new if l == old else l for l in lines) + "\n", "kit%d.range%d" % (n, i))
                    held += (status, out[-1]) == (1, "NO: %s (policy)" % refusal)
                    if (status, out[-1]) != (1, "NO: %s (policy)" % refusal):
                        bad.append("%s with %s: status %d, %s" % (name, new, status, out[-1][:60]))
                print("out of range: %s: %d of %d plays end NO: %s (policy)" % (field, held, len(turns), refusal))
        first = next((l for l in lines if l.startswith("call ")), None)
        if first is None:
            bad.append("%s: the file has no line call" % name)
        named += first is not None
        for length in (9, 8) if first else ():
            renamed = " ".join(first.split(" ")[:1] + ["N" * length] + first.split(" ")[2:])
            status, out = plan(policy, "\n".join(renamed if l == first else l for l in lines) + "\n", "kit%d.name%d" % (n, length))
            if ((status, out[-1]) == (1, "NO: policy-call-line")) != (length == 9):
                bad.append("%s with a call's name of %d characters: status %d, %s" % (name, length, status, out[-1][:60]))
    if not count.get("work", 0) + count.get("yes", 0):
        bad.append("no file is of this build's format and field")
    for b in bad:
        print("NOT AS EXPECTED:", b)
    print("asked for one bit above the file's target: %d files" % above)
    print("a lookup call under a name of 9 and of 8 characters: %d files" % named)
    if not warned:
        bad.append("no plan under a policy that no whole call takes ended PLAN ONLY")
        print("NOT AS EXPECTED:", bad[-1])
    print("plans that say that no whole call can pass: %d" % warned)
    if not recorded:
        bad.append("no plan printed the two lines on what the committer's files rest")
        print("NOT AS EXPECTED:", bad[-1])
    print("plans with the two lines on the committer's files: %d" % recorded)
    print("with the line \"magic 1295397120\" put first: %d files" % formats)
    print("example_kits_test.py: %d parameter files (%s), %d plays with a line changed; %d not as expected"
          % (sum(count.values()), ", ".join("%d %s" % (count[k], k) for k in sorted(count)), played, len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
