#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Checks the "accounting:" and "covered:" lines of a call of cp_verify, the verifier.

usage: call_accounting_test.py <cp_verify> <config folder> <asked bits> <file with the output of a whole call under that folder, made without -q> [<that call's request file>]
In a verification call (the usage line's "whole call") the child process of the stage `request`
passes the call's accounting of soundness errors to the parent process as text
(verifier/src/vf_accounting_record.h), and the parent prints it. A planning call, `-plan accounting`,
computes the same accounting from the same configuration and prints it with the same function,
blk::print_accounting(). So the lines "accounting:" and "work:" after the line
"---- the call ----" must be the planning call's, in the same order. A match does not show that the
numbers are right: both calls use blk::account(). Not compared (`OWN`), because the two calls
differ in them: the verifier's own hash calls (counted in one, planned in the other), the
compiled bound, which depends on their number, at three values of t (the prover's hash calls), the
bits requested of the call (printed: "asked"), and the two lines "accounting:
one-time files:", which a verification call prints earlier, in the stages `key` and `leaf`. The
counted hash calls must be above 0 and at most the number in the plan's line "planned for chains
and openings". No rule says that they are: the count also includes hashing that is not planned (the
line says which). On the stored sets it holds by a margin, because the plan counts every opened
path whole, and paths share vertices.
With the request file, the output must contain exactly one line "covered:", identical to what
covered_line() computes from it and the configuration's policy.txt. That line tells a caller which
positions of the request (r to e - 1) a yes is about (docs/verifying.md, "When to accept").
covered_line(), policy_kind() and above_clip() follow the rules that verifier/docs/output.md gives
for the same fields of the line "result-v1" and, for `exact-match`,
verifier/docs/parameter-file.md, "The standard parameter form". How far they are independent of
cp_verify.cpp has not been checked. They are taken from a module of test helpers that is not part
of this repository: covered_line()'s format and above_clip()'s formula unchanged, the condition for
`exact-match` inlined in policy_kind(), policy.txt read as text.
Not tested: any request but the one given. SUITE.md's row verifier.call_accounting.examples gives it
the requests of examples/accepted-proof and examples/full-size: both ask for several tokens up to
the one after the request, under the policy `exact-match`.
Exit status 0: the same. 1: not; each difference has a line. 2: could not run; but an argument that
names a missing file gives a traceback and status 1."""
import os, re, struct, subprocess, sys

OWN = ("accounting: hash calls:", "accounting:   planned for chains and openings:", "accounting:   at t = 2^", "accounting: asked of this call:", "accounting: one-time files:")


def values(policy):
    return dict(l.split() for l in policy.splitlines() if len(l.split()) == 2)


def policy_kind(policy):
    """The field policy= of "covered:" and "result-v1": exact-match (C = theta = 2^sh, cap = 0,
    sT = sh - 4), clipped-gap (the step table, sT = sh - 4), log-probability (sT = s) or
    another-table."""
    v = values(policy)
    C, theta, cap, sh, sT = (int(v[n]) for n in ("C", "theta", "cap", "sh", "sT"))
    exact = C == theta == 1 << sh and cap == 0 and sT + 4 == sh
    return "exact-match" if exact else "clipped-gap" if sT + 4 == sh else "log-probability" if v["sT"] == v["s"] else "another-table"


def above_clip(n, policy):
    """How many of n tokens may cost more than C: the smallest of cap, n and, if C is above 2^sh,
    n max(theta - 2^sh, 0) / (C - 2^sh) rounded down."""
    v = values(policy)
    C, theta, cap, least = int(v["C"]), int(v["theta"]), int(v["cap"]), 1 << int(v["sh"])
    return min(cap, n) if C <= least else min(cap, n, n * max(theta - least, 0) // (C - least))


# request: little-endian 32-bit words: T token ids, r, e.
def covered_line(request, policy):
    w = struct.unpack("<%dI" % (len(request) // 4), request)
    T, r, e = len(w) - 2, w[-2], w[-1]
    v = values(policy)
    return "covered: r=%d e=%d T=%d C=%s theta=%s cap=%s s=%s sh=%s positions=%d..%d tokens=%s y=%s next-token=%s policy=%s above-clip=%d (places in the request, counted from 0; y=yes: the answer is about y too; it is about nothing else; above-clip: so many of the tokens named by tokens= may cost more than C, and such a token may be any token. It is no limit on the tokens that may be any token: one that costs C or less may be any token too where the committed model's logits lie close together, and no verifier sees that. All costs, each counted as at most C, add up to at most theta times the number of these tokens)" % (
        r, e, T, v["C"], v["theta"], v["cap"], v["s"], v["sh"], r, e - 1, "none" if r == T - 1 else "%d..%d" % (r + 1, min(e, T - 1)), "yes" if e == T else "no", "yes" if (r, e) == (T - 1, T) else "no", policy_kind(policy), above_clip(min(e, T - 1) - r, policy))  # fmt: skip


# Not called here: verifier/tests/what_a_yes_covers_test.py imports the next four functions.
def result_pattern(request, policy, proof, status, name, bits=128):
    """A regular expression for a verification call's line for programs, "result-v1 .."
    (verifier/README.md, "The line for programs"). Groups 1 to 6 are work=, work-rounds=,
    least-round=, least-round-without-work=, commitment-assumed-rounds= and build=, matched by form
    only. The field record=1 must follow group 5. Every other field is
    matched literally: a constant, the status, or what follows from the caller's own files and
    options. request, proof: the files' bytes (a proof's third 32-bit word is leaf=); policy: the
    text of policy.txt; name: the commitment ID passed with -id, or None."""
    w = struct.unpack("<%dI" % (len(request) // 4), request)
    T, r, e = len(w) - 2, w[-2], w[-1]
    v = values(policy)
    kind, last = policy_kind(policy), min(e, T - 1)
    claim = "next-token" if r == T - 1 else "exact-match" if kind == "exact-match" else "policy"
    head = "result-v1 status=%d claim=%s id=%s bits=%d " % (status, claim, name or "not-shown-without-id", bits)
    tail = " r=%d e=%d T=%d tokens=%s y=%s policy=%s C=%s theta=%s cap=%s s=%s sh=%s sT=%s F=%s above-clip=%d leaf=%d stages=key,leaf,request dw=8 build=" % (
        r, e, T, "none" if r == T - 1 else "%d..%d" % (r + 1, last), "yes" if e == T else "no", kind, v["C"], v["theta"], v["cap"], v["s"], v["sh"], v["sT"], v["F"], above_clip(last - r, policy),
        struct.unpack("<I", proof[8:12])[0])  # fmt: skip
    figures = r"work=(\d+) work-rounds=(\d+) least-round=(\d+\.\d\d) least-round-without-work=(\d+\.\d\d) commitment-assumed-rounds=(0|[1-9]\d*) record=1"
    return re.escape(head) + figures + re.escape(tail) + "([0-9a-f]{64})"


def work_agrees(line, fields):
    """True if the call's line "proof of work: .." agrees with work=, work-rounds=, least-round= and
    least-round-without-work= in fields, a match of result_pattern()."""
    if line == "proof of work: none counted":
        return fields[1] == fields[2] == "0"
    told = re.fullmatch(r"proof of work: up to (\d+) bits counted in (\d+) rounds?; with it the least round has (\d+\.\d\d) bits, without it (\d+\.\d\d) bits", line)
    return told is not None and told.groups() == (fields[1], fields[2], fields[3], fields[4]) and 1 <= int(fields[1]) <= 32 and int(fields[2]) >= 1


def commitment_agrees(line, fields):
    """True if the call's line "commitment: .." agrees with commitment-assumed-rounds= in fields, a
    match of result_pattern(), and ends with the clause on a record (RECORDCLAUSE in
    verifier/src/vf_common.h)."""
    n = int(fields[5])
    clause = "; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given"
    return line == "commitment: %d closeness round%s of the one-time files rest%s on another source than the bound within a third of the distance" % (n, "" if n == 1 else "s", "s" if n == 1 else "") + clause


def figures_agree(lines, fields, stages_printed=True):
    """True if two of fields, a match of result_pattern(), agree with lines, the call's standard
    output: least-round= with the only line "accounting: THE LARGEST ROUND ERROR" after
    "---- the call ----", and build= with the line "result:" of the stage `request`.
    stages_printed: the call's ending is one before which cp_verify prints the stages' output; if
    false, that line may be missing."""
    closing = lines[lines.index("---- the call ----"):] if "---- the call ----" in lines else []
    largest = re.findall(r"^accounting: THE LARGEST ROUND ERROR: 2\^-(\d+\.\d\d),", "\n".join(closing), re.M)
    builds = set(re.findall(r"^result:.* build=([0-9a-f]{64}) ", "\n".join(lines), re.M))
    return largest == [fields[3]] and builds <= {fields[6]} and (bool(builds) or not stages_printed)


def lines_of(text):
    return [l for l in text.splitlines() if (l.startswith("accounting:") or l.startswith("work: ")) and not l.startswith(OWN)]


def main():
    if len(sys.argv) not in (5, 6):
        print(__doc__.split("\n\n")[1].splitlines()[0], file=sys.stderr)
        return 2
    verifier, folder, asked, log = sys.argv[1:5]
    verifier = os.path.abspath(verifier)  # absolute, because subprocess looks a bare name up on PATH, not in the current directory
    plan = subprocess.run([verifier, "-plan", "accounting", folder, asked], capture_output=True, text=True, check=False)
    if plan.returncode != 4:
        print("call_accounting_test.py: the planning call ended with status %d" % plan.returncode)
        return 2
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if text.count("---- the call ----\n") != 1:
        print("call_accounting_test.py: the file does not hold exactly one line ---- the call ----")
        return 2
    call, planned = lines_of(text.split("---- the call ----\n")[1]), lines_of(plan.stdout)
    differ = 0
    for i in range(max(len(call), len(planned))):
        a, b = (x[i] if i < len(x) else "(none)" for x in (call, planned))
        if a != b:
            differ += 1
            print("differ:\n  the call: %s\n  the plan: %s" % (a, b))
    counted = [l for l in text.split("---- the call ----\n")[1].splitlines() if l.startswith("accounting: hash calls:") and "in this run: " in l]
    if len(counted) != 1 or not counted[0].rsplit(" ", 1)[1].isdigit() or int(counted[0].rsplit(" ", 1)[1]) == 0:
        differ += 1
        print("the call prints no count of its own hash calls")
    else:
        most = [l.split(": ", 2)[2].split(" ")[0] for l in plan.stdout.splitlines() if l.startswith("accounting:   planned for chains and openings: ")]
        if len(most) != 1 or not most[0].isdigit() or int(counted[0].rsplit(" ", 1)[1]) > int(most[0]):
            differ += 1
            print("the call's count of hash calls, %s, is above the planned number, %s" % (counted[0].rsplit(" ", 1)[1], most))
    if len(sys.argv) == 6:
        with open(sys.argv[5], "rb") as f, open(folder + "/policy.txt", encoding="ascii") as g:
            mine = covered_line(f.read(), g.read())
        said = [l for l in text.splitlines() if l.startswith("covered:")]
        if said == [mine]:
            print("the line covered: is, letter for letter, what the page's rules give: %s (..)" % mine.split(" (")[0])
        else:
            differ += 1
            print("the line covered: differs, or stands %d times:\n  the call: %s\n  the page's rules: %s" % (len(said), said, mine))
    print("call_accounting_test.py: %d lines of the plan's accounting, %d not as in the call" % (len(planned), differ))
    return 1 if differ or not planned else 0


if __name__ == "__main__":
    sys.exit(main())
