#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py.)
# The work folder goes there, since the test does not check the directories above it.
"""Checks that the verifier, cp_verify, rejects an honest proof file with one leading field altered.

usage: proof_fields_sweep_test.py <cp_verify> <config folder> <ivk folder> <request file> <y file> <proof file> <id> <new work folder> [<workers>]

<config folder> contains `policy.txt`, `KL.bin` and `layout.txt`; <ivk folder> is the commitment's
public key directory, <y file> contains the next token, <id> is the commitment ID.
`cp_verify -plan request .. -fields` lists every field of a proof file with its place (byte offset),
length and class. The length of a Merkle opening depends on the drawn positions (codeword indexes),
so places are exact only up to the first opening's first field, "opened symbols".
Only fields up to there are altered: the header, the salt of the first hash chain
(a Fiat-Shamir transcript), its whole first segment (the part for one sub-proof), the start of the
second, and that field. Of fields whose names differ only in numbers, the first and the last are
taken. By class:
  elements   in the first and the last coordinate (eight bytes each): (a) "second spelling": the
             prime p is added, which gives a non-canonical encoding of the same element; skipped if
             the sum reaches 2^64: for p = 2^64 - 2^32 + 1, at every value from 2^32 - 1 up;
             (b) "plus one"; (c) "not canonical": eight bytes ff, above the prime. After (a) and
             (c) an output line "proof: .." or "chain: .." must contain "canonical".
  others     (digest, salt, nonce, words) one bit of the last byte is flipped.
"opened symbols" is not stored as listed. The list gives an opening's symbols, then its salts, then its paths,
each at its largest possible length. The file has, for each distinct drawn position in order, its symbols and then
its salt (docs/format/FORMAT.md, R7.8; blocks/tree/opening.h, write_opening()), so only the first position is
where the list says. At that position the test alters the first and the last symbol, as elements, and one bit of
the salt. Any bytes can be a salt, so after this last change a "proof: .." or "chain: .." line must contain
"an opening does not lead to its root" and none "canonical". The field's last listed coordinate is not taken:
depending on the number of positions it falls on a symbol or a salt, and the checks for elements fit only a symbol.
Also, the file is shortened and lengthened by one byte and by eight bytes.
The unaltered proof must end with `YES` or `NOT-A-VERIFICATION` (a test build's best result; the
cause is not checked), and every altered copy with exit status 1 and a last line `NO: <name>`
that begins with none of `NOT_THE_PROOF`: those names report a failure of cp_verify itself
(`internal-`) or a fault in another input. If every check passes, the test's last line ends
"all refused" (rejected) and its exit status is 0. If cp_verify finds no fault in an altered copy
(`PASSES`), the test prints "WAS NOT REFUSED: .." with the copy's path, keeps the copy in the work
folder and ends with exit status 3; if only other checks fail, with 1.
If the work folder exists, the test only prints its usage (exit status 2). It creates the folder,
closed to other users, and any missing directory above it, with the default mode, and removes
neither. It deletes each "altered-<n>.proof" after its call unless kept as above;
`length.proof`, the last copy of another length, stays.
"""

import concurrent.futures
import os
import re
import subprocess
import sys


def main():
    if len(sys.argv) not in (9, 10) or os.path.lexists(sys.argv[8]):
        print(__doc__.split("\n")[2], file=sys.stderr)
        return 2
    program, config, ivk, request, y, proof, name, work = sys.argv[1:9]
    program, work = os.path.abspath(program), os.path.abspath(work)  # absolute, because subprocess looks a bare name up on PATH, not in the current directory; the work folder too, so that os.path.dirname() of a one-word name is not empty
    workers = int(sys.argv[9]) if len(sys.argv) == 10 else 4
    plan = subprocess.run([program, "-plan", "request", os.path.join(config, "policy.txt"), os.path.join(config, "KL.bin"), "128", "-layout",
                           os.path.join(config, "layout.txt"), "-prompt", request, "-reply", y, "-fields", "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fields, salts = [], None  # salts: the listed length, in bytes, of the first opening's salts
    for l in plan.stdout.split("\n"):
        m = re.match(r"^field: proof at ([0-9]+) ([0-9]+) ([a-z]+): (.*)$", l)
        if m:
            fields.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
        elif l.startswith("field: proof") and fields:
            m = re.match(r"^field: proof at-most [0-9]+ ([0-9]+) salt: .*salts$", l)
            salts = int(m.group(1)) if m else None
            break  # the first field whose place is not exact
    # 4: a planning call's exit status
    if plan.returncode != 4 or len(fields) < 100:
        print("FAILED: the planning call gives %d fields with exact places, status %d" % (len(fields), plan.returncode))
        return 1
    version = [l.split()[1] for l in open(os.path.join(config, "policy.txt")) if l.startswith("version ")][0]
    salt = int([l.split()[1] for l in open(os.path.join(config, "policy.txt")) if l.startswith("salt ")][0])  # a salt's bytes
    if [f[3].endswith("opened symbols") for f in fields].count(True) != 1 or not fields[-1][3].endswith("opened symbols") or not salts or salts % salt or fields[-1][1] % (salts // salt):
        print("FAILED: the list's fields with exact places do not end with one opening's symbols, followed by its salts")
        return 1
    column = fields[-1][1] // (salts // salt)  # the bytes of one position's symbols
    prime = (1 << 61) - 1 if version[2] == "1" else (1 << 64) - (1 << 32) + 1  # the version's third character is the field's number
    honest = open(proof, "rb").read()
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    groups = {}
    for f in fields:
        groups.setdefault((f[2], re.sub(r"[0-9]+", "#", f[3])), []).append(f)
    chosen = sorted({g[0] for g in groups.values()} | {g[-1] for g in groups.values()})

    cases = []
    for at, size, cls, what in chosen:
        if cls == "elements":
            for k in sorted({0, (column if what.endswith("opened symbols") else size) // 8 - 1}):  # its first and its last coordinate; of an opening's symbols: the first position's
                v = int.from_bytes(honest[at + 8 * k : at + 8 * k + 8], "little")
                if v + prime < 1 << 64:
                    cases.append(("second spelling", at, what, at + 8 * k, (v + prime).to_bytes(8, "little")))
                cases.append(("plus one", at, what, at + 8 * k, ((v + 1) % prime).to_bytes(8, "little")))
                cases.append(("not canonical", at, what, at + 8 * k, b"\xff" * 8))
        else:
            cases.append(("a bit of a " + cls, at, what, at + size - 1, bytes([honest[at + size - 1] ^ 0x10])))
    at, what = fields[-1][0], fields[-1][3]
    cases.append(("a bit of the first position's salt", at, what, at + column + salt - 1, bytes([honest[at + column + salt - 1] ^ 0x10])))

    # -show-stages: without it a rejecting call prints no "proof: .." or "chain: .." line
    def call(path):
        x = subprocess.run([program, config, ivk, request, y, path, "-id", name, "-q", "-show-stages"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = x.stdout.strip().split("\n")
        return x.returncode, lines[-1], [l for l in lines if l.startswith(("proof: ", "chain: "))]

    # Exit statuses taken as "no fault found". cp_verify's table has 0, 3 and 14 for that, and no
    # 10 or 11: verifier/docs/output.md, "Exit codes".
    PASSES = (0, 3, 10, 11, 14)
    # Beginnings of a last line that says nothing about the altered proof. `NO: internal-..` reports
    # a failure of cp_verify itself. The others are rejections of the call's other inputs: the
    # commitment ID (model-statement), the acceptance policy (policy), a line of the parameter file,
    # policy.txt (policy-..), the request file or the y file (statement-..). Most of these
    # rejections come before cp_verify reads more of the proof than its header.
    NOT_THE_PROOF = ("NO: internal-", "NO: model-statement", "NO: policy", "NO: statement-")

    def one(i):
        kind, at, what, where, new = cases[i]
        path = os.path.join(work, "altered-%d.proof" % i)
        open(path, "wb").write(honest[:where] + new + honest[where + len(new):])
        status, last, why = call(path)
        if status in PASSES:  # None stands for "not refused"; the copy stays
            return None, "%s, field at %d (%s): status %d: %s; the copy is kept: %s" % (kind, at, what, status, last, path)
        os.remove(path)
        good = status == 1 and last.startswith("NO: ") and not last.startswith(NOT_THE_PROOF)
        if kind in ("second spelling", "not canonical"):
            good = good and any("canonical" in l for l in why)
        if kind == "a bit of the first position's salt":
            good = good and any("an opening does not lead to its root" in l for l in why) and not any("canonical" in l for l in why)
        return good, "%s, field at %d (%s): status %d: %s%s" % (kind, at, what, status, last, " | " + why[-1] if why else "")

    status, last, _ = call(proof)
    passes = (status == 0 and last == "YES") or (status == 14 and last == "NOT-A-VERIFICATION")
    print("the unaltered proof: status %d: %s" % (status, last))
    bad = 0 if passes else 1
    counts = {}
    accepted = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, (good, text) in enumerate(pool.map(one, range(len(cases)))):
            counts[cases[i][0]] = counts.get(cases[i][0], 0) + 1
            if not good:
                bad += 1
                accepted += good is None
                print(("WAS NOT REFUSED: " if good is None else "NOT AS ASKED: ") + text)
    for kind, data in (("one byte shorter", honest[:-1]), ("eight bytes shorter", honest[:-8]), ("one byte longer", honest + b"\0"), ("eight bytes longer", honest + bytes(8))):
        path = os.path.join(work, "length.proof")
        open(path, "wb").write(data)
        status, last, why = call(path)
        good = status == 1 and last.startswith("NO: ") and not last.startswith(NOT_THE_PROOF)
        print("%s: status %d: %s%s%s" % (kind, status, last, " | " + why[-1] if why else "", "" if good else "   <-- NOT AS ASKED"))
        bad += not good
        if status in PASSES:
            accepted += 1
            kept = os.path.join(work, "not-refused.%s.proof" % kind.replace(" ", "-"))
            os.replace(path, kept)  # the next length is written to the same path
            print("WAS NOT REFUSED: %s: the copy is kept: %s" % (kind, kept))
    print("%d fields with exact places, %d kinds of field, %d chosen; altered copies: %s" % (len(fields), len(groups), len(chosen), ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items()))))
    print("proof_fields_sweep_test: %d altered copies and 4 lengths: %s" % (len(cases), "%d NOT AS ASKED" % bad if bad else "all refused"))
    return 3 if accepted else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
