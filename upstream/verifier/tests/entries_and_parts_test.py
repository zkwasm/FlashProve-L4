#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Tests the verifier program cp_verify: leaf table numbers, part-calls, bits above the target.

usage: entries_and_parts_test.py <cp_verify> <world> <new work folder> <honest status: 0, or 14 for a test build> <a part-call's honest status: 3, or 14>

The fixture (`world` in names and printed lines): a directory worlds/<name> of honest files
made by verifier/tests/other_parameters_worlds.sh (config/, ivk/key.bin, ivk/leaf/, in/example.req,
out/example.y, out/example.proof, id.txt). key.bin is an 8-byte header, the 32-byte root of its
table, under "record 1" in config/policy.txt the record's answers (24 bytes for each answer and
row), then proofs. For cases 9 to 13, config/ must be a small example configuration with the lines
"sec 128" and "tau 128".
The honest status is cp_verify's on the unaltered files (0: YES; 14: NOT-A-VERIFICATION).
The work directory must not exist yet. The test removes only the case directories it made, and only
if every case goes as expected.

A commitment has several tables of the same weights: the key file's (the anchor), and the numbered
leaf tables (printed: copy) against which proofs are made. A leaf table's leaf file (`entry` in
names and printed lines) is ivk/leaf/<number>.bin.
The number. A proof's header gives its leaf table's number, and cp_verify reads the leaf file with
that number. A leaf file contains no number. The proof is tied to the number by the statement
hashed at the start of its first hash chain, which contains the number.
`own`: the header's number; `other`: another number.
  1  Control: the proof's leaf file under both numbers, the proof unaltered: the honest status.
  2  The same, with the header's number changed to `other`: the proof's own leaf file is read, so
     only the number differs. Expected: NO: proof-refused or NO: chain-refused.
  3  The header's number changed to `other`, no leaf file under it: NO: cp-entry-missing.
  8  If the fixture has a second leaf table (worlds/true has): its honest leaf file under `own`, the
     proof unaltered. The stage leaf (cp_verify's check of the leaf file) must pass (a rejecting
     call prints its stages' output only with -show-stages). Expected as in 2, since the proof was
     made against another table.
  14 A false leaf file (the proof's, last byte altered; control: -leaf gives NO: entry-refused)
     under `other`, next to the true one: the honest status, since only the leaf file that the proof
     names counts. A second call gives the same status and last line, and both calls leave the
     case's files and bytes unchanged.
The part-calls (-key, -leaf: each checks one file). -leaf takes the key file's table from its header,
root and record's answers (form; with -id, their ID) and skips its proofs. These honest, then zeros:
  4  -leaf passes as a part-call and prints "key: NOT CHECKED BY THIS CALL ...".
  5  -key: NO: key-refused.   6  The verification call (printed: the whole call): NO: key-refused.
     With -show-stages, the two passing stages' output (leaf, request) comes before the stage key's.
  7  Control: -leaf on the honest key file prints the same line: it is about the call, not the file.
  15 -key on the honest key file: the part-call's honest status and that status's last line
     (3: PART PASSED: key; 14: NOT-A-VERIFICATION).   16  See before_the_proofs(), below main().
The target. Each stage of cp_verify (key, leaf, request) rejects with NO: security-bits if the
requested bits (-bits; printed: asked) exceed the parameter file's target (line `sec`). A part-call
or verification call accepts at most 128 requested bits, the fixture's target, so the test makes a
configuration with a lower one: `sec` and `tau` (random combinations in the approximate range
check; tied to `sec`) set to 96 and layout.txt regenerated, by the recipe in
prover/examples/TARGET-KITS.md (cp_verify -plan arrays, prover/tools/pack_layout.py with the small
configurations' arguments). The other files are the fixture's, since a file without a key file's
form is rejected before that check. No -id: the new configuration has another commitment ID.
  9  Control: the planning call at 96 requested bits exits with status 4 (PLAN ONLY), so the
     configuration itself is accepted.
  10 -key, 11 -leaf, 12 the verification call, at the default of 128 requested bits, with
     -show-stages: NO: security-bits, and a stage's printed line gives both numbers.
  13 -key with -bits 96 passes that check; the key file, made under another configuration, is
     rejected: NO: key-refused.
Exit status 0: every case went as expected. 1: some case did not; also wrong arguments or an
existing work directory. 3: a call that cp_verify must reject (`MUST_REFUSE`) ended with a status
for no fault found (`PASSES`): the test prints a line with the words WAS NOT REFUSED, exits at once
and removes nothing."""
import os
import re
import shutil
import struct
import subprocess
import sys


def main():
    if len(sys.argv) != 6 or not sys.argv[3]:
        sys.exit(__doc__)
    if os.path.lexists(sys.argv[3]):
        sys.exit("entries_and_parts_test.py: %s exists already: name a new work folder" % sys.argv[3])
    program, world, work, honest, part = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])  # absolute: subprocess looks up a bare program name on PATH, not in the current directory, and `os.path.dirname` of a bare directory name is empty
    at = lambda *p: os.path.join(world, *p)
    config, request, reply = at("config"), at("in", "example.req"), at("out", "example.y")
    named = open(at("id.txt")).read().strip()
    proof, key = (open(p, "rb").read() for p in (at("out", "example.proof"), at("ivk", "key.bin")))
    own = struct.unpack_from("<I", proof, 8)[0]  # the proof's leaf table: the prover chooses it
    other = 1 if own == 0 else 0                 # another number
    entry = open(at("ivk", "leaf", "%d.bin" % own), "rb").read()
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    bad = 0

    def lay(name, key_bytes, entries, proof_bytes, entry=entry):
        folder = os.path.join(work, name)
        os.makedirs(folder)  # fails if it exists: the final removal takes only what `lay` made
        os.makedirs(os.path.join(folder, "ivk", "leaf"))
        open(os.path.join(folder, "ivk", "key.bin"), "wb").write(key_bytes)
        for number in entries:
            open(os.path.join(folder, "ivk", "leaf", "%d.bin" % number), "wb").write(entry)
        open(os.path.join(folder, "proof"), "wb").write(proof_bytes)
        return folder

    def run(*words, named=named):
        ran = subprocess.run([program] + list(words) + (["-id", named] if named else []), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return ran.returncode, [l for l in ran.stdout.decode("utf-8", "replace").split("\n") if l]

    # cp_verify's exit statuses for a call that found no fault. 10 and 11 are not in its table:
    # verifier/docs/output.md, "Exit codes".
    PASSES = (0, 3, 10, 11, 14)
    # Name prefixes (`what`) of the cases in which cp_verify must reject.
    MUST_REFUSE = ("2 ", "3 ", "5 ", "6 ", "8 ", "10 ", "11 ", "12 ", "13 ", "14 control", "16 ")

    def ask(what, good, status, lines):
        nonlocal bad
        if what.startswith(MUST_REFUSE) and status in PASSES:
            print("WAS NOT REFUSED: %s: status %d: %s; the cases' folders are kept in %s" % (what, status, lines[-1][:90] if lines else "(no line)", work))
            sys.exit(3)
        bad += not good
        print("%s: status %d: %s%s" % (what, status, lines[-1][:90] if lines else "(no line)", "" if good else "   NOT AS FORETOLD"))

    whole = lambda f, *more: run(config, os.path.join(f, "ivk"), request, reply, os.path.join(f, "proof"), *more)
    turned = proof[:8] + struct.pack("<I", other) + proof[12:]
    NOT_CHECKED = "key: NOT CHECKED BY THIS CALL"
    status, lines = whole(lay("1", key, (own, other), proof))
    ask("1 two entries, the proof as made", status == honest, status, lines)
    status, lines = whole(lay("2", key, (own, other), turned))
    ask("2 the head turned to another copy's number, under which the proof's own entry lies", status == 1 and lines[-1] in ("NO: proof-refused", "NO: chain-refused"), status, lines)
    status, lines = whole(lay("3", key, (own,), turned))
    ask("3 the head turned to another copy's number, no such entry", status == 1 and lines[-1] == "NO: cp-entry-missing", status, lines)
    hollow = lay("4", key[: proofs_begin(config)] + bytes(len(key) - proofs_begin(config)), (own,), proof)
    k, e = os.path.join(hollow, "ivk", "key.bin"), os.path.join(hollow, "ivk", "leaf", "%d.bin" % own)
    status, lines = run("-leaf", config, k, e)
    ask("4 -leaf on a key that is hollow behind its root", status == part and any(l.startswith(NOT_CHECKED) for l in lines), status, lines)
    status, lines = run("-key", config, k)
    ask("5 -key on it", status == 1 and lines[-1] == "NO: key-refused", status, lines)
    status, lines = whole(hollow)
    ask("6 the whole call on it", status == 1 and lines[-1] == "NO: key-refused", status, lines)
    status, lines = whole(hollow, "-show-stages")
    passed = [i for i, l in enumerate(lines) if l.startswith("STAGE PASSED") or l == "NOT-A-VERIFICATION"]  # the second: how a stage of a test build ends
    ask("6 the same with -show-stages: the two stages that passed stand before the one that refused", status == 1 and len(passed) == 2 and passed[-1] < min([i for i, l in enumerate(lines) if l.startswith("---- stage key ")], default=-1), status, lines)
    status, lines = run("-leaf", config, at("ivk", "key.bin"), at("ivk", "leaf", "%d.bin" % own))
    ask("7 -leaf on the honest key", status == part and any(l.startswith(NOT_CHECKED) for l in lines), status, lines)
    status, lines = run("-key", config, at("ivk", "key.bin"))
    ask("15 -key on the honest key", status == part and bool(lines) and lines[-1] == {3: "PART PASSED: key", 14: "NOT-A-VERIFICATION"}.get(part), status, lines)
    beside = lay("14", key, (own,), proof)
    planted = os.path.join(beside, "ivk", "leaf", "%d.bin" % other)
    open(planted, "wb").write(entry[:-1] + bytes([entry[-1] ^ 1]))
    status, lines = run("-leaf", config, at("ivk", "key.bin"), planted)
    ask("14 control: -leaf on the false entry", status == 1 and lines[-1:] == ["NO: entry-refused"], status, lines)
    state = lambda: sorted((os.path.relpath(os.path.join(r, f), beside), open(os.path.join(r, f), "rb").read()) for r, _, fs in os.walk(beside) for f in fs)
    before = state()
    status, lines = whole(beside)
    ask("14 the false entry beside the one that the proof names", status == honest, status, lines)
    again, lines_again = whole(beside)
    ask("14 the same call again: the same status and last line", (again, lines_again[-1:]) == (status, lines[-1:]), again, lines_again)
    ask("14 the case's folder after both calls: the same files and bytes", state() == before, again, lines_again)
    made = ["1", "2", "3", "4", "14"]
    if os.path.isfile(at("ivk", "leaf", "%d.bin" % other)):
        status, lines = whole(lay("8", key, (own,), proof, open(at("ivk", "leaf", "%d.bin" % other), "rb").read()), "-show-stages")  # the option: a rejecting call then prints its stages' output too
        ask("8 the honest entry of another copy under the proof's number", status == 1 and lines[-1] in ("NO: proof-refused", "NO: chain-refused") and
            any(l.startswith("STAGE PASSED") and l.endswith(": leaf") for l in lines), status, lines)
        made.append("8")
    else:
        print("8 not played: the world has one stored copy")
    nine = lay("9", key, (own,), proof)
    made.append("9")
    low = os.path.join(nine, "config")
    os.makedirs(low)
    lower, times = re.subn(r"(?m)^(sec|tau) 128$", r"\1 96", open(at("config", "policy.txt")).read())
    open(os.path.join(low, "policy.txt"), "w").write(lower)
    shutil.copy(at("config", "KL.bin"), low)
    arrays = os.path.join(nine, "arrays.txt")
    open(arrays, "wb").write(subprocess.run([program, "-plan", "arrays", low, "96"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout)
    pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prover", "tools", "pack_layout.py")
    open(os.path.join(low, "layout.txt"), "wb").write(subprocess.run([sys.executable, "-B", pack, "--arrays", arrays, "17", "4"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout)
    status, lines = run("-plan", "accounting", low, "96", named=None)
    ask("9 a folder with the target 96, planned at 96 asked bits", times == 2 and status == 4, status, lines)
    k, e = os.path.join(nine, "ivk", "key.bin"), os.path.join(nine, "ivk", "leaf", "%d.bin" % own)
    WHY = "accounting: asked of this call: 128 bits, above the parameter file's target of 96 bits"
    for what, words in (("10 -key", ("-key", low, k)), ("11 -leaf", ("-leaf", low, k, e)), ("12 the whole call", (low, os.path.join(nine, "ivk"), request, reply, os.path.join(nine, "proof")))):
        status, lines = run(*words, "-show-stages", named=None)
        ask(what + " under it, 128 bits asked", status == 1 and lines[-1] == "NO: security-bits" and WHY in lines, status, lines)
    status, lines = run("-key", low, k, "-bits", "96", named=None)
    ask("13 -key under it, 96 bits asked", status == 1 and lines[-1] == "NO: key-refused", status, lines)
    made.append(before_the_proofs(config, key, own, proof, lay, whole, run, ask))
    if not bad:
        for name in made:
            shutil.rmtree(os.path.join(work, name))  # only directories that `lay` made
    print("entries_and_parts_test: " + ("ok" if not bad else "%d NOT AS FORETOLD" % bad))
    sys.exit(1 if bad else 0)


def before_the_proofs(config, key, own, proof, lay, whole, run, ask):
    """Case 16: one bit turned in the last 8 bytes before the key file's proofs: in the
    last answer's last coordinate. These bytes enter the commitment ID and the first
    query of the leaf file's LINK proof. Expected: the verification call, with -id: NO: model-statement;
    -leaf without -id: NO: entry-refused, its reason ending "(before the test)": the chain itself has
    turned, before the proximity test. Returns the name of the case's directory."""
    at = proofs_begin(config) - 8
    bent = lay("16", key[:at] + bytes([key[at] ^ 1]) + key[at + 1 :], (own,), proof)
    status, lines = whole(bent)
    ask("16 a bit turned just before the key's proofs, the whole call", status == 1 and lines[-1] == "NO: model-statement", status, lines)
    status, lines = run("-leaf", config, os.path.join(bent, "ivk", "key.bin"), os.path.join(bent, "ivk", "leaf", "%d.bin" % own), named=None)
    ask("16 the same key, -leaf without -id", status == 1 and lines[-1] == "NO: entry-refused" and lines[-2].endswith("(before the test)"), status, lines)
    return "16"


def proofs_begin(config):
    """Returns where a key file's proofs begin (blocks/frame/check_key.h, "Layout"): behind the 8-byte
    header, the 32-byte root and one element of E (24 bytes) for each answer and
    each of the table's 2^v rows (policy.txt: answers, the first number of table)."""
    said = {w[0]: w[1:] for w in (l.split() for l in open(os.path.join(config, "policy.txt"))) if w}
    return 40 + 24 * (int(said["answers"][0]) << int(said["table"][0]))


main()
