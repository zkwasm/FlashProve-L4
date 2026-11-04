#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs the verifier cp_verify on synthetic key, leaf and proof files; needs no committer or prover.

usage: made_up_files_test.py <cp_verify> <config folder> <new work folder> [<threads>]

Exit status: 0 if every check passed, 1 if not, 2 for bad usage, e.g. an existing work folder.
3 if a call found no fault in these files: the test prints a line with the words WAS NOT REFUSED
and ends at once. The test removes nothing at any ending.

Each file is a valid header, then zero bytes or bytes ff up to the maximum length that the planning
call `-plan accounting` prints. In the output `entry` is the leaf file and `whole` the verification
call (stages key, leaf, request).
  - With either fill, `-key`, `-leaf` and `whole` must exit with status 1 and print no "internal-",
    which marks a fault of program or machine.
  - Zero bytes, without `-show-stages`: each call must print three lines and no more (three()).
    The stages' output, which holds values read from the rejected files, comes only with
    `-show-stages`; the stages' reasons and IDs below are taken from it.
  - Zero bytes: key file and proof file must be rejected at their first proof of work or, if
    `policy.txt` asks for none, first Merkle opening: their checkers and plans of rounds agree that
    far. The leaf file's reason is not compared. `whole` must run all stages, print
    `read: sub-proof G_0` (the proof's first sub-proof was read to its end) and end
    `NO: key-refused`, the first stage's.
  - Bytes ff are not a canonical field element. No reason is compared, so in `whole` a leaf or
    request stage killed by a signal goes unnoticed. `-leaf` also gets a key file of zero bytes.
  - `-plan id` must print the commitment ID, 64 hex digits, once. In `whole` with `-id` and that ID
    all three stages must print it; the stages of `-key` and `-leaf` must print no other. `whole`
    without `-id` must show `not-shown-without-id` instead. `-key` with `-id` and that ID ends as
    without it. With another ID `-key`, `-leaf` and `whole`, with or without `-show-stages`, must
    print only two fixed `id:` lines and `NO: model-statement`, so no ID. Beyond the files' headers
    cp_verify compares the ID first and checks nothing else in such files. Another root in `key.bin`
    gives another ID. On a `key.bin` that is a header alone `-plan id` must end with status 1 and
    `NO: ..` and print neither an ID nor `h_par=`, the digest of the config folder's files.
"""

import os
import re
import struct
import subprocess
import sys

# cp_verify's exit statuses for a call that found no fault: 0, 3 and 14, listed in
# verifier/docs/output.md, "Exit codes". No call ends with 10 or 11.
PASSES = (0, 3, 10, 11, 14)


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__.split("\n")[2], file=sys.stderr)
        return 2
    # Absolute paths: subprocess looks a bare program name up on PATH, not in the current directory,
    # and os.path.dirname() of a one-word `work`, taken below, would be empty.
    program, config, work = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
    threads = sys.argv[4] if len(sys.argv) == 5 else "4"
    policy = open(os.path.join(config, "policy.txt")).read()
    version = re.search(r"^version (\S{4})$", policy, re.M).group(1).encode()
    tokens = 1 << int(re.search(r"^lt (\d+)$", policy, re.M).group(1))

    def run(*words):
        x = subprocess.run([program] + list(words), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        # Every file here is synthetic, so no call that checks one may pass.
        if words[0] != "-plan" and x.returncode in PASSES:
            print("WAS NOT REFUSED: cp_verify %s: status %d: %s; the files are kept in %s" % (" ".join(words), x.returncode, x.stdout.strip().split("\n")[-1], work))
            sys.exit(3)
        return x.returncode, x.stdout.strip().split("\n")

    status, plan = run("-plan", "accounting", config, "128")
    most = {m.group(1): int(m.group(2)) for m in (re.match(r"^length: (key|entry|proof): at most (\d+) bytes", l) for l in plan) if m}
    if status != 4 or len(most) != 3:
        print("FAILED: the planning call: status %d, %d lengths: %s" % (status, len(most), plan[-1]))
        return 1
    # The work folder must be new: ivk/key.bin and ivk/leaf/0.bin, written below, are the names of a
    # client's real key file and leaf file.
    try:
        os.makedirs(os.path.dirname(work), exist_ok=True)
        os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    except OSError as fault:
        print("made_up_files_test.py: cannot create the work folder, which must not exist yet: %s" % fault, file=sys.stderr)
        return 2
    os.makedirs(os.path.join(work, "ivk", "leaf"))
    key, entry, proof = os.path.join(work, "ivk", "key.bin"), os.path.join(work, "ivk", "leaf", "0.bin"), os.path.join(work, "proof")
    request, y = os.path.join(work, "request"), os.path.join(work, "y")
    open(request, "wb").write(struct.pack("<%dI" % (tokens + 2), *([0] * tokens + [tokens - 1, tokens])))
    open(y, "wb").write(struct.pack("<I", 0))

    def lay(fill, root=b""):
        open(key, "wb").write(b"CPIK" + version + root + fill * (most["key"] - 8 - len(root)))
        open(entry, "wb").write(b"CPEN" + version + fill * (most["entry"] - 8))
        open(proof, "wb").write(b"CPPF" + version + bytes(4) + fill * (most["proof"] - 12))

    calls = {"-key": ["-key", config, key], "-leaf": ["-leaf", config, key, entry], "whole": [config, os.path.join(work, "ivk"), request, y, proof]}
    tail = ["-bits", "128", "-j", threads]
    bad = 0

    def ask(what, good, text):
        nonlocal bad
        print("%s: %s%s" % (what, text, "" if good else "   <-- NOT AS ASKED"))
        bad += not good

    def ids(lines):
        """The distinct IDs in `lines`, printed by a stage, a passing `whole` or `-plan id`."""
        return sorted({l.split("commitment ID: ", 1)[1] for l in lines if l.startswith(("commitment ID: ", "computed commitment ID: "))})

    def three(lines, stage, last, reason=""):
        """Whether `lines` are a rejecting call's three lines and no more. The second names the
        stage and, after a colon, the stage's reason if it gave one. `reason`: how that reason must
        begin."""
        named = "refused by the stage " + stage
        return len(lines) == 3 and lines[0] == "---- the call ----" and (lines[1].startswith(named + ": " + reason) or (not reason and lines[1] == named)) and lines[2] == last

    asks_for_work = any(int(x) for line in re.findall(r"^work \w+ (\d+) (\d+) (\d+)$", policy, re.M) for x in line)
    FIRST = "a nonce does not give the proof of work that its round asks for" if asks_for_work else "an opening does not lead to its root"
    lay(b"\0")
    seen = set()
    for name, stage, want, reason in (("-key", "key", "NO: key-refused", "key: " + FIRST), ("-leaf", "leaf", "NO: entry-refused", "entry: "), ("whole", "key", "NO: key-refused", "proof: " + FIRST)):
        status, lines = run(*calls[name], *tail)
        ask("zero bytes, " + name, status == 1 and three(lines, stage, want, FIRST if stage == "key" else ""), "status %d: %d lines: %s" % (status, len(lines), lines[-1]))
        status, lines = run(*calls[name], "-show-stages", *tail)
        ask("zero bytes, %s -show-stages" % name, status == 1 and three(lines[-3:], stage, want) and any(l.startswith(reason) for l in lines) and not any("internal-" in l for l in lines),
            "status %d: %s" % (status, lines[-1]))
        if name == "whole":
            ask("zero bytes, whole: all three stages ran", all(any(l.startswith(w) for l in lines) for w in ("key: ", "entry: ", "proof: ", "read: sub-proof G_0")), "%d lines" % len(lines))
            ask("zero bytes, whole, no -id: no ID is shown", ids(lines) == ["not-shown-without-id"], ", ".join(ids(lines)))
        else:
            seen.update(ids(lines))
    status, lines = run("-plan", "id", config, key)
    seen.update(ids(lines))
    ask("the planning call id", status == 4 and lines[-1] == "PLAN ONLY: no proof was checked" and any(l.startswith("h_par=") for l in lines) and sum(l.startswith("commitment ID: ") for l in lines) == 1,
        "status %d: %s" % (status, lines[-1]))
    own = next(iter(seen), "0" * 64)
    status, lines = run(*calls["whole"], "-id", own, "-show-stages", *tail)
    seen.update(ids(lines))
    ask("the whole call with the files' own ID: each stage computes it", status == 1 and lines[-1] == "NO: key-refused" and sum(l == "computed commitment ID: " + own for l in lines) == 3,
        "status %d: %s" % (status, lines[-1]))
    ask("one ID in all four calls", len(seen) == 1 and re.fullmatch("[0-9a-f]{64}", own) is not None, ", ".join(sorted(seen)))
    status, lines = run(*calls["-key"], "-id", own, *tail)
    ask("-key with the files' own ID", status == 1 and lines[-1] == "NO: key-refused", "status %d: %s" % (status, lines[-1]))
    other = ("1" if own[0] != "1" else "2") + own[1:]
    ANOTHER = ["id: these are not the files of the commitment that you named; nothing in them was checked",
               "id: take the commitment ID from your own record of where it was published, never from these files", "NO: model-statement"]  # fmt: skip
    for name in ("-key", "-leaf", "whole"):
        for more in ([], ["-show-stages"]):
            status, lines = run(*calls[name], "-id", other, *more, *tail)
            ask(" ".join([name] + more) + " with another ID", status == 1 and lines == ANOTHER, "status %d: %s" % (status, lines[-1]))
    lay(b"\0", root=b"\1" + bytes(31))
    status, lines = run("-plan", "id", config, key)
    ask("another root, another ID", status == 4 and len(ids(lines)) == 1 and ids(lines)[0] != own, "status %d" % status)
    open(key, "wb").write(b"CPIK" + version)
    status, lines = run("-plan", "id", config, key)
    ask("the planning call id on a key.bin that is a head alone: refused, and no digest is printed", status == 1 and lines[-1].startswith("NO: ") and not ids(lines) and not any(l.startswith("h_par=") for l in lines),
        "status %d: %s" % (status, lines[-1]))
    lay(b"\xff")
    # `-leaf` also reads the record's answers in key.bin. Bytes ff are not canonical field elements,
    # so it rejects the key file first.
    for name, want in (("-key", "NO: key-refused"), ("-leaf", "NO: key-refused"), ("whole", "NO: key-refused")):
        status, lines = run(*calls[name], "-show-stages", *tail)
        ask("bytes ff, " + name, status == 1 and lines[-1] == want and not any("internal-" in l for l in lines), "status %d: %s | %s" % (status, lines[-1], next((l for l in lines if l.startswith(("key: ", "entry: "))), "-")))
    # A key file of zero bytes, which `-leaf` reads without complaint, with the
    # leaf file of bytes ff: the call must get as far as the leaf file and reject that.
    open(key, "wb").write(b"CPIK" + version + bytes(most["key"] - 8))
    status, lines = run(*calls["-leaf"], "-show-stages", *tail)
    ask("a key file of zero bytes, a leaf file of bytes ff, -leaf", status == 1 and lines[-1] == "NO: entry-refused" and not any("internal-" in l for l in lines),
        "status %d: %s | %s" % (status, lines[-1], next((l for l in lines if l.startswith(("key: ", "entry: "))), "-")))
    print("made_up_files_test: %s" % ("%d NOT AS ASKED" % bad if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
