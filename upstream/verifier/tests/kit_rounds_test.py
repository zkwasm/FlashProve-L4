#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Compares the planned rounds (error, work, count) of parameter files with stored lines.

usage: kit_rounds_test.py <cp_verify> <new folder> [2]      the word 2, the field's number, may follow and changes nothing
status 0: all as stored; 1: not; 2: a wrong call
`<new folder>` must not exist (else status 2); the script creates it (mode 0700), with any missing
parent directories (default mode), and removes nothing.

Prover, verifier (cp_verify) and planning calls compute a round's error (the programs' own figure
for the soundness error of one round of verifier challenges) with the same functions
(blocks/numbers/blocks.h), so no comparison between them shows a wrong figure, and most rounds meet
the target (128 bits) with so many bits to spare that a wrong figure would not change the verdict.
This test makes cp_verify's planning call `-plan request` with `-shapes` (expected status: 4) for
the parameter files that main() lists (each a `kit` in identifiers and output): files of
prover/examples/ (what each of them is: prover/examples/TARGET-KITS.md) and one of the tests' own,
prover/tests/configurations/pinned_no_proof_of_work, with no proof of work.
For each it compares the printed "round:" lines, in order, with a stored file in
verifier/tests/known_answers/kit_rounds/, some 400 lines each. Each line (a "kind" in the output)
counts the rounds that share a name (numbers printed as #), bits of proof of work and error in
bits, rounded down to two decimals.
cp_verify itself printed the stored lines. They are no second computation: a difference shows that
a figure has changed, not which of the two is right.
verifier/tests/kit_rounds_second.py computes the errors a second time, but only for the proximity
tests' rounds. Its list KITS says for which of the files here; its head says what it leaves out.
After a deliberate change to an error bound, a size or a parameter file, regenerate the stored
files (the script prints a command for each that differs; it writes `<stored file>.new`, beside the
stored file, and touches no stored file) and read the difference. For the full-size file the printed
command fails as it stands: it reads `request` in `<new folder>`, which by then holds the small
size's, so cp_verify rejects the call and the new file is empty. First write a full-size one there.
Only planning calls are made: no proof is made or checked."""
import difflib, os, pathlib, shlex, struct, subprocess, sys

TREE = pathlib.Path(__file__).resolve().parent.parent.parent
STORED = TREE / "verifier" / "tests" / "known_answers" / "kit_rounds"


def main():
    if len(sys.argv) not in (3, 4) or os.path.lexists(os.path.abspath(sys.argv[2])) or sys.argv[3:] not in ([], ["2"]):
        print(__doc__.split("\n")[2].strip(), file=sys.stderr)
        return 2
    verifier, work = str(pathlib.Path(sys.argv[1]).absolute()), pathlib.Path(os.path.abspath(sys.argv[2]))
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(mode=0o700)  # fails if the name is taken; closed to other users
    # Inputs, 32-bit words: `y`, the next token; `request`, 2^`lt` token ids, then the range r, e.
    (work / "y").write_bytes(struct.pack("<I", 0))
    bad = kinds = 0
    kits = [TREE / "prover" / "examples" / "target_full_f2", TREE / "prover" / "examples" / "target_small_f2", TREE / "prover" / "tests" / "configurations" / "pinned_no_proof_of_work"]
    for at in kits:
        kit, policy = at.name, "policy.txt"  # the stored file has the directory's name
        tokens = 1 << int(next(l.split()[1] for l in (at / policy).read_text().splitlines() if l.startswith("lt ")))
        (work / "request").write_bytes(struct.pack("<%dI" % (tokens + 2), *([0] * tokens + [tokens - 1, tokens])))
        call = [verifier, "-plan", "request", str(at / policy), str(at / "KL.bin"), "128", "-layout", str(at / "layout.txt"), "-prompt", str(work / "request"), "-reply", str(work / "y"), "-shapes"]
        r = subprocess.run(call, capture_output=True, text=True, errors="replace", check=False)
        said = [l for l in r.stdout.splitlines() if l.startswith("round:")]
        stored = (STORED / (kit + ".txt")).read_text().splitlines()
        kinds += len(stored)
        if r.returncode != 4 or not said:
            bad += 1
            print("NOT AS STORED: %s: the planning call ended with status %d and %d lines round:" % (kit, r.returncode, len(said)))
        elif said != stored:
            moved = [l for l in difflib.unified_diff(stored, said, "stored", "printed", lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
            bad += 1
            print("NOT AS STORED: %s: %d lines differ (- stored, + printed); the first twelve:" % (kit, len(moved)))
            for l in moved[:12]:
                print("  " + l)
            stored_at = str(STORED / (kit + ".txt"))
            print("  to make the print anew, beside the stored file: %s | grep '^round:' > %s" % (shlex.join(map(str, call)), shlex.quote(stored_at + ".new")))
            print("  Read the new file; if it is what you mean, put it in the place of %s" % shlex.quote(stored_at))
    print("kit_rounds_test.py: %d kits, %d kinds of rounds; %d kits not as stored" % (len(kits), kinds, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
