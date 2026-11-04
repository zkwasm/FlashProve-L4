#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Checks that cp_verify refuses altered public files and proofs under a commitment.

usage: record_altered_sets_test.py <cp_verify> <config folder> <ivk folder> <request file> <y file> <proof file> <commitment ID> <new work folder>
  The first seven name an honest set on synthetic weights (the config folder holds policy.txt, KL.bin
  and layout.txt; the ivk folder key.bin and leaf/<k>.bin) whose parameter file, policy.txt, counts
  proof of work (a line "work positions" with a number above 0), such as
  the set in the folder worlds/true that verifier/tests/other_parameters_worlds.sh makes from
  prover/tests/configurations/pinned. They are only read.
  new work folder   must not exist: the test creates it, with mode 0700 (and any missing folder above
                    it, with the caller's umask), writes the altered copies and cp_verify's output
                    there, and leaves it in place. It removes nothing. Each altered set is a whole
                    copy of the ivk folder or of the proof: 40 to 80 MB in all at the small
                    configurations, about 200 MB at full size.
No GPU is needed: cp_verify is the only program that the test starts.

A key file and a leaf file hold, after the head (8 bytes) and the table's root (32
bytes), the record's stored answers, 24 bytes each: 2^b rows (b: the first number of the parameter
line "table") times the number in the line "answers" (docs/format/FORMAT.md, R13.2, R13.3). The leaf
file that the test alters is the one that the proof's head names.

The controls, on the honest set:
  the whole call under the commitment ID ends with YES, and its line result-v1 has that ID, bits=128,
  record=1 and counted proof of work (work= and work-rounds= above 0);
  the same call with -no-proof-of-work ends with NOT-A-VERIFICATION: by cp_verify's own count some
  round reaches the parameter file's target only with its proof of work;
  the part-calls -key and -leaf end with PART PASSED;
  an unaltered copy of the ivk folder and of the proof, written as the altered copies are, gets the
  same answers from the whole call and the two part-calls: so copying alone makes no refusal.
The altered sets: each differs from the honest set in what its line says and in nothing else, which
the test checks on the written files, byte by byte, before it calls cp_verify. Each must end with
status 1 and the last line "NO: <name>" with the name in the list CASES below. No line before it may
begin with a word with which a call, a part-call or a stage says that no check failed (a refusing
call without -show-stages prints no stage's own lines), and no line result-v1, the line for programs,
may have status 0, 3 or 14. The altered sets:
  one bit of the first and of the last stored answer of the key file's record and of the leaf file's;
  the two files' records exchanged; the leaf file alone with the key file's record;
  one bit of each file's root, of the first byte after its record, of the middle of what follows the
  record, and of its last byte;
  one bit in the proof's middle; the proof without its last byte; the proof with one more byte.
The commitment ID is a hash over the first table's root and its record's answers, among others
(FORMAT.md, R4.11), so under the ID a changed root or answer of the key file is refused as
model-statement, and of the exchanged records only the key file's half is judged. The altered records
are therefore also given to the part-calls -key and -leaf, which the test calls without -id: there
the stage itself must refuse them (key-refused, entry-refused). Root and answers are in the statement
of every hash chain that speaks of the table (FORMAT.md, R4.6). In the key file the two names also
mark the record's end: its last byte falls under model-statement, the next under key-refused. The
leaf file has no such mark: every change to it is refused as entry-refused.

A pass shows that cp_verify notices these changes to these files, and no more: the copies are honest
files with a few bytes changed, not the work of a cheating committer or prover. A refusal's name tells
the stage that refused, not the check that failed. No case aims at a nonce of a proof of work.

Exit status 0: all as expected. 1: something is not. 2: a wrong call, or a set that is not of this
kind. 3: cp_verify did not refuse an altered set: its line has the words WAS NOT REFUSED; keep the
work folder as it is and report it (SECURITY.md)."""
import hashlib
import os
import re
import struct
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)  # each line shows as it is printed, into a file or a pipe too: a run that is ended early leaves what it had
ME = "record_altered_sets_test.py"
PASSED = (0, 3, 14)  # the exit codes of a call in which no check failed: accepted; a part passed; not a verification
# A line that says so: one that begins with a passing word, or the line for programs with such a status.
PASSING = re.compile(r"^[ \t]*(?:(?:YES|WITHIN LIMITS|PART PASSED|STAGE PASSED|NOT-A-VERIFICATION)(?:\W.*)?|result-v1 .*\bstatus=(?:0|3|14)\b.*)$", re.M)


def stop(words):
    print("%s: %s" % (ME, words), file=sys.stderr)
    sys.exit(2)


if len(sys.argv) != 9:
    print(__doc__)
    sys.exit(2)
verifier, config, ivk, request, y, proof_file, work = (os.path.abspath(x) for x in sys.argv[1:7] + sys.argv[8:])
commitment = sys.argv[7]
if not re.fullmatch("[0-9a-f]{64}", commitment):
    stop("the commitment ID is not 64 hexadecimal digits in lower case")
if not os.path.isfile(verifier) or not os.access(verifier, os.X_OK):
    stop("%s is not a program that can be started" % verifier)
if os.path.lexists(work):
    stop("%s exists already: name a new work folder" % work)
if any(os.path.realpath(work).startswith(os.path.realpath(x) + "/") for x in (config, ivk)):
    stop("the work folder would lie inside the honest set: %s" % work)


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        stop("cannot read %s: %s" % (path, e.strerror))


# The parameter lines that say where the record lies, and that the set is of the kind in the usage.
lines = [l.split() for l in read(config + "/policy.txt").decode("ascii", "replace").split("\n")]


def numbers(*name):
    found = [l[len(name) :] for l in lines if tuple(l[: len(name)]) == name]
    if len(found) != 1 or not found[0] or not all(re.fullmatch("[0-9]{1,9}", x) for x in found[0]):
        stop('policy.txt has not one line "%s" with numbers' % " ".join(name))
    return [int(x) for x in found[0]]


if not any(numbers("work", "positions")):
    stop('policy.txt counts no proof of work (the line "work positions"): this test is for a set that does')
rows, answers = numbers("table")[0], numbers("answers")[0]
if not 1 <= rows <= 32 or not 1 <= answers <= 9:
    stop('the line "table" begins with %d and the line "answers" has %d' % (rows, answers))
FIRST = 40  # the head and the root
AFTER = FIRST + 24 * answers * (1 << rows)  # the first byte after the record

proof = read(proof_file)
if len(proof) < 12:
    stop("the proof is shorter than its head")
LEAF = "leaf/%d.bin" % struct.unpack("<3I", proof[:12])[2]  # the head's third word: the leaf table's number
try:
    names = ["key.bin"] + sorted("leaf/" + n for n in os.listdir(ivk + "/leaf"))
    in_config = sorted(n for n in os.listdir(config) if os.path.isfile(config + "/" + n))
except OSError as e:
    stop("cannot list %s: %s" % (e.filename, e.strerror))
honest = {n: read(ivk + "/" + n) for n in names}
if LEAF not in honest:
    stop("the proof's head names %s, and the ivk folder has no such file" % LEAF)
key, leaf = honest["key.bin"], honest[LEAF]
if min(len(key), len(leaf)) <= AFTER:
    stop("a file ends within its record: key.bin has %d bytes, %s %d, and the record ends at byte %d" % (len(key), LEAF, len(leaf), AFTER - 1))


def flip(data, at):
    return data[:at] + bytes([data[at] ^ 1]) + data[at + 1 :]


def with_record_of(data, other):
    return data[:FIRST] + other[FIRST:AFTER] + data[AFTER:]


# Each case: its name; the call (whole, key or leaf); the files that differ from the honest set, as
# {name: new bytes}; what must differ, as {name: a list of bytes, or a word}; the refusal's name.
CASES = []


def bit(call, name, file, at, refusal):
    data = proof if file == "proof" else honest[file]
    CASES.append(("%s.%s" % (call, name), call, {file: flip(data, at)}, {file: [at]}, refusal))


exchanged = {"key.bin": with_record_of(key, leaf), LEAF: with_record_of(leaf, key)}
for call, of_key, of_leaf in (("whole", "model-statement", "entry-refused"), ("key", "key-refused", None), ("leaf", None, "entry-refused")):
    if of_key:
        bit(call, "key-first-answer", "key.bin", FIRST, of_key)
        bit(call, "key-last-answer", "key.bin", AFTER - 1, of_key)
    if of_leaf:
        bit(call, "leaf-first-answer", LEAF, FIRST, of_leaf)
        bit(call, "leaf-last-answer", LEAF, AFTER - 1, of_leaf)
    # The call -key reads the key file alone; the call -leaf is given the honest key file.
    files = exchanged if call == "whole" else {n: exchanged[n] for n in (("key.bin",) if call == "key" else (LEAF,))}
    CASES.append((call + ".records-exchanged", call, files, {n: "the other file's record" for n in files}, of_key or of_leaf))
CASES.append(("whole.leaf-with-the-key-files-record", "whole", {LEAF: exchanged[LEAF]}, {LEAF: "the other file's record"}, "entry-refused"))
bit("whole", "key-root", "key.bin", 8, "model-statement")
bit("whole", "key-after-the-record", "key.bin", AFTER, "key-refused")
bit("whole", "key-middle", "key.bin", (AFTER + len(key)) // 2, "key-refused")
bit("whole", "key-last-byte", "key.bin", len(key) - 1, "key-refused")
bit("whole", "leaf-root", LEAF, 8, "entry-refused")
bit("whole", "leaf-after-the-record", LEAF, AFTER, "entry-refused")
bit("whole", "leaf-middle", LEAF, (AFTER + len(leaf)) // 2, "entry-refused")
bit("whole", "leaf-last-byte", LEAF, len(leaf) - 1, "entry-refused")
bit("whole", "proof-middle", "proof", len(proof) // 2, "proof-refused")
CASES.append(("whole.proof-cut", "whole", {"proof": proof[:-1]}, {"proof": "one byte shorter"}, "proof-refused"))
CASES.append(("whole.proof-longer", "whole", {"proof": proof + b"\0"}, {"proof": "one byte longer"}, "proof-refused"))



def sums():
    return [hashlib.sha256(read(p)).hexdigest() for p in [config + "/" + n for n in in_config] + [request, y, proof_file] + [ivk + "/" + n for n in names]]


before = sums()  # also: every file of the set can be read, before anything is made
try:
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
except OSError as e:
    stop("cannot make %s: %s" % (work, e.strerror))


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "xb") as f:  # x: fails if anything stands at the name
        f.write(data)


def differs_as_said(old, new, other, said):
    """Whether the written bytes differ from the honest ones in what the case says and nothing else."""
    if said == "one byte shorter":
        return new == old[:-1]
    if said == "one byte longer":
        return len(new) == len(old) + 1 and new[:-1] == old
    if said == "the other file's record":
        return len(new) == len(old) and new[:FIRST] == old[:FIRST] and new[FIRST:AFTER] == other[FIRST:AFTER] != old[FIRST:AFTER] and new[AFTER:] == old[AFTER:]
    return len(new) == len(old) and [i for i, (a, b) in enumerate(zip(old, new)) if a != b] == said


def call(name, kind, folder, proof_path, *more):
    """Run cp_verify; keep its output in <work>/<name>.txt; return its status, last line and output."""
    if kind == "whole":
        words = [config, folder, request, y, proof_path, "-id", commitment]
    elif kind == "key":
        words = ["-key", config, folder + "/key.bin", "-bits", "128"]
    else:
        words = ["-leaf", config, folder + "/key.bin", folder + "/" + LEAF, "-bits", "128"]
    r = subprocess.run([verifier] + words + list(more), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    called.append(name)
    write("%s/%s.txt" % (work, name), r.stdout)
    text = r.stdout.decode("utf-8", "replace")
    printed = [l for l in text.split("\n") if l.strip()]
    return r.returncode, printed[-1] if printed else "", text


bad, accepted, called = 0, 0, []


def say(ok, words):
    global bad
    bad += not ok
    print("%s %s" % ("as expected    " if ok else "NOT AS EXPECTED", words))


print("the record: %d stored answers of 24 bytes: bytes %d to %d of key.bin (%d bytes) and of %s (%d bytes); the proof: %d bytes" % (answers << rows, FIRST, AFTER - 1, len(key), LEAF, len(leaf), len(proof)))
same = sum(key[i : i + 24] == leaf[i : i + 24] for i in range(FIRST, AFTER, 24))
zeros = sum(f[i : i + 24] == bytes(24) for f in (key, leaf) for i in range(FIRST, AFTER, 24))
say(same == 0 and zeros == 0 and key[8:40] != leaf[8:40], "the two records: %d answers are the same in both files, %d are zero; the two roots %s" % (same, zeros, "differ" if key[8:40] != leaf[8:40] else "are the same"))

# The controls.
status, last, text = call("honest.whole", "whole", ivk, proof_file)
say(status == 0 and last == "YES", "the honest set, the whole call: status %d: %s" % (status, last[:120]))
result = re.findall(r"^result-v1 (.*)$", text, re.M)
fields = dict(w.split("=", 1) for w in result[0].split(" ") if "=" in w) if len(result) == 1 else {}
shown = " ".join("%s=%s" % (n, fields.get(n, "(missing)")) for n in ("status", "bits", "work", "work-rounds", "record"))
say(fields.get("status") == "0" and fields.get("id") == commitment and fields.get("bits") == "128" and fields.get("record") == "1" and all(re.fullmatch("[1-9][0-9]*", fields.get(n, "")) for n in ("work", "work-rounds")),
    "its line result-v1 (%d such line(s)): the ID %s; %s" % (len(result), "is the call's" if fields.get("id") == commitment else "IS NOT THE CALL'S", shown))  # fmt: skip
status, last, text = call("honest.no-proof-of-work", "whole", ivk, proof_file, "-no-proof-of-work")
say(status == 14 and last == "NOT-A-VERIFICATION", "the honest set, the whole call with -no-proof-of-work: status %d: %s" % (status, last[:120]))
for n in names:
    write("%s/unaltered/ivk/%s" % (work, n), honest[n])
write(work + "/unaltered/copy.proof", proof)
say(all(read("%s/unaltered/ivk/%s" % (work, n)) == honest[n] for n in names) and read(work + "/unaltered/copy.proof") == proof, "the unaltered copy has the honest set's bytes")
status, last, text = call("unaltered.whole", "whole", work + "/unaltered/ivk", work + "/unaltered/copy.proof")
say(status == 0 and last == "YES", "the unaltered copy, the whole call: status %d: %s" % (status, last[:120]))
for which, folder in (("honest set", ivk), ("unaltered copy", work + "/unaltered/ivk")):
    for kind in ("key", "leaf"):
        status, last, text = call("%s.%s" % (which.split(" ")[0], kind), kind, folder, proof_file)
        say(status == 3 and last == "PART PASSED: " + kind, "the %s, the part-call -%s: status %d: %s" % (which, kind, status, last[:120]))

# The altered sets.
for name, kind, files, said, refusal in CASES:
    folder, proof_path = "%s/%s/ivk" % (work, name), proof_file
    if "proof" in files:
        folder, proof_path = ivk, "%s/%s/altered.proof" % (work, name)
        write(proof_path, files["proof"])
        as_said = differs_as_said(proof, read(proof_path), None, said["proof"])
    else:
        for n in names:
            write(folder + "/" + n, files.get(n, honest[n]))
        as_said = all(differs_as_said(honest[n], read(folder + "/" + n), leaf if n == "key.bin" else key, said[n]) if n in said else read(folder + "/" + n) == honest[n] for n in names)
    what = "; ".join("%s: %s" % (n, "bit 0 of byte %d" % s[0] if isinstance(s, list) else s) for n, s in sorted(said.items()))
    if not as_said:
        say(False, "%s: the written copy does not differ from the honest set in just this: %s; cp_verify is not called" % (name, what))
        continue
    status, last, text = call(name, kind, folder, proof_path)
    passing = PASSING.search(text)
    if status in PASSED or passing:
        accepted += 1
        print("WAS NOT REFUSED: %s (%s): status %d: %s%s; the files are kept in %s/%s" % (name, what, status, last[:120], "; a line of its output: " + passing[0].strip()[:120] if passing and passing[0].strip() != last else "", work, name))
        continue
    say(status == 1 and last == "NO: " + refusal, "%s (%s): status %d: %s%s" % (name, what, status, last[:120], "" if last == "NO: " + refusal else " (expected: NO: %s)" % refusal))

say(sums() == before, "the honest set's files have the same digests after all calls as before")
if accepted:
    print("%s: %d of %d altered sets WERE NOT REFUSED: keep %s as it is and report it" % (ME, accepted, len(CASES), work))
    sys.exit(3)
print("%s: %d calls on the honest set and its unaltered copy, %d altered sets: %s" % (ME, sum(n.startswith(("honest.", "unaltered.")) for n in called), len(CASES), "all as expected" if not bad else "%d NOT AS EXPECTED" % bad))
sys.exit(1 if bad else 0)
