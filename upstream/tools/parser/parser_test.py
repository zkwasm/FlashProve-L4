#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (The runner is run_suite.py. By
# hand, name a work folder in a directory that no other account can write to.)
"""Tests tools/parser/ on a stored example: parse_files.py writes, read_parsed.py reads back.

usage: parser_test.py <the tree's folder> <unit.o> <example> <new work folder>
  unit.o    verifier/src/cp_verify.cpp compiled with -DCP_VERIFY_AS_A_UNIT (the test
            build.verifier.unit makes it)
  example   a folder below examples/ with config/, ivk/, example.req and example.proof
  new work folder   must not exist, and the folder above it must: the test creates it, with mode
            0700, and leaves it in place

Checks: (1) On the honest files parse_files.py exits with 0 and writes three parsed files, each with
its order file and its seeds.
(2) read_parsed.py reads each of the three whole, with its seeds, follows every path to its root, and
makes the original file again from the parsed file and its order file.
(3) What a parsed file takes from another original file is what that file's own parsed file has: the
leaf file's root0 and answers0 are the key file's root and answers, and the proof's root and answers
are the leaf file's.
(4) With --lines parse_files.py writes the same ten files again, and three files of lines. In each of
these the first line has the original file's length and SHA-256, and the joined "hex" of the lines
with "len" above 0 is the original file.
(5) Eight damaged copies of each original file (one bit changed in the first byte, at each quarter
and in the last byte; the last byte cut off; a byte added; the empty file): on each, parse_files.py
exits with 1, its last line begins "REFUSED: ", and it writes nothing.
(6) Damaged copies of each parsed file, order file and seeds file (one bit changed in the first byte,
and for a parsed file also in the last; the last byte cut off; a byte added): on each, read_parsed.py
exits with 1 and its last line begins "REFUSED: ".
The test shows that the two scripts agree on the form and refuse these damaged files. What it does
not test: the values of the seeds (read_parsed.py does not compute them); an original file from which
a whole opening has been taken out (parse_files.py can parse such a file: README.md, Limits); a leaf
that two openings share (the small stored examples have none); the other fields of the lines; whether a parsed
file proves anything (neither script makes the protocol's checks).
Exit code 0: all as expected. 1: something is not. 2: a wrong call."""
import hashlib
import json
import os
import struct
import subprocess
import sys

if len(sys.argv) != 5:
    print(__doc__)
    sys.exit(2)
if os.path.lexists(sys.argv[4]) or not os.path.isdir(os.path.dirname(os.path.abspath(sys.argv[4]))):
    print("parser_test.py: %s exists already, or the folder above it does not: name a new work folder in a folder that exists" % sys.argv[4], file=sys.stderr)
    sys.exit(2)
tree, unit, example = (os.path.abspath(x) for x in sys.argv[1:4])
work = os.path.abspath(sys.argv[4])
os.umask(0o077)  # parse_files.py refuses an output folder that another account can write to, as one made under umask 002 is
os.mkdir(work, 0o700)  # new, and this account's alone
bad = 0


def run(*words):
    r = subprocess.run(list(words), cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in r.stdout.split("\n") if l.strip()]
    return r.returncode, lines[-1] if lines else ""


def say(ok, what):
    global bad
    bad += not ok
    print("%s %s" % ("as expected    " if ok else "NOT AS EXPECTED", what))


def load(path):
    with open(path, "rb") as f:
        return f.read()


def store(path, data):
    with open(path, "xb") as f:  # x: every file of this test is new
        f.write(data)


def flipped(data, at):
    return data[:at] + bytes([data[at] ^ 1]) + data[at + 1:]


def table(data, at):
    """A root and its list of answers, at byte `at` of a parsed file (PARSED-FORM.md, sections 2
    and 3), and where they end."""
    end = at + 40 + 24 * struct.unpack_from("<Q", data, at + 32)[0]
    return data[at:end], end


status, last = run("g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-w", "-pthread", "-I" + tree + "/verifier/src", "-I" + tree + "/blocks", "-I" + tree + "/prover", "-DBLK_FIELD=2", "-DDIGEST_WORDS=8",
                   tree + "/tools/audit/pubtool.cpp", unit, "-o", work + "/pubtool")
if status:
    sys.exit("parser_test.py: tools/audit/pubtool.cpp does not build: " + last[:300])
os.mkdir(work + "/tables")
status, last = run(work + "/pubtool", example + "/config", "128", work + "/tables")  # 128: the requested bits
if status:
    sys.exit("parser_test.py: pubtool ends with status %d: %s" % (status, last[:300]))

number = struct.unpack("<I", load(example + "/example.proof")[8:12])[0]  # the proof's leaf table (R14.2)
originals = {"key": "ivk/key.bin", "leaf-%d" % number: "ivk/leaf/%d.bin" % number, "proof": "example.proof"}
parse = [sys.executable, "-B", tree + "/tools/parser/parse_files.py", work + "/tables", example + "/config"]
read = [sys.executable, "-B", tree + "/tools/parser/read_parsed.py"]

os.mkdir(work + "/parsed")
status, last = run(*parse, example + "/ivk", example + "/example.req", example + "/example.proof", work + "/parsed")
wrote = sorted(os.listdir(work + "/parsed"))
want = sorted(s + ".parsed.bin" + e for s in originals for e in ("", ".order", ".seeds")) + ["leaf-%d.parsed.bin.seeds0" % number]
say(status == 0 and last == "PARSED: 3 files" and wrote == sorted(want), "the honest files: status %d: %s; %d files written" % (status, last[:120], len(wrote)))
if bad:
    sys.exit("parser_test.py: the honest files are not parsed: nothing more can be tested")
for stem, path in originals.items():
    status, last = run(*read, "%s/parsed/%s.parsed.bin" % (work, stem), "--config=" + example + "/config", "--file=%s/%s" % (example, path))
    say(status == 0 and last.startswith("READ: ") and "; seeds " in last and "lead to their roots" in last and "gives the original again" in last, "%s read back: status %d: %s" % (stem, status, last[:400]))

of_key, of_leaf, of_proof = (load("%s/parsed/%s.parsed.bin" % (work, stem)) for stem in originals)
anchor, own_at = table(of_leaf, 8)
say(anchor == table(of_key, 8)[0] and table(of_leaf, own_at)[0] == table(of_proof, 8)[0], "the leaf file's root0 and answers0 are the key file's root and answers; the proof's root and answers are the leaf file's")

os.mkdir(work + "/with-lines")
status, last = run(*parse, example + "/ivk", example + "/example.req", example + "/example.proof", work + "/with-lines", "--lines")
lined = status == 0 and sorted(os.listdir(work + "/with-lines")) == sorted(want + [s + ".lines.jsonl" for s in originals]) and all(load(work + "/with-lines/" + n) == load(work + "/parsed/" + n) for n in want)
say(lined, "with --lines: status %d: %s; the ten files again, and three files of lines" % (status, last[:120]))
for stem, path in originals.items() if lined else ():
    whole = load(example + "/" + path)
    lines = [json.loads(l) for l in load("%s/with-lines/%s.lines.jsonl" % (work, stem)).decode("ascii").split("\n") if l]
    say((lines[0].get("bytes"), lines[0].get("sha256")) == (len(whole), hashlib.sha256(whole).hexdigest()) and bytes.fromhex("".join(d["hex"] for d in lines[1:] if d["len"])) == whole,
        "%s.lines.jsonl: %d lines; the first has the original's length and SHA-256, and the joined hex is the original" % (stem, len(lines)))

case = 0
for stem, path in originals.items():
    whole = load(example + "/" + path)
    n = len(whole)
    for name, data in [("bit at byte %d" % at, flipped(whole, at)) for at in (0, n // 4, n // 2, 3 * n // 4, n - 1)] + [("last byte cut off", whole[:-1]), ("a byte added", whole + b"\0"), ("empty", b"")]:
        case += 1
        here = "%s/case-%d" % (work, case)
        os.makedirs(here + "/out")
        if path == "example.proof":  # the damaged proof with the example's own key and leaf files
            store(here + "/example.proof", data)
            status, last = run(*parse, example + "/ivk", example + "/example.req", here + "/example.proof", here + "/out")
        else:  # the key file and the leaf file again, one of them damaged, with the example's own proof
            os.makedirs(here + "/ivk/leaf")
            for other in list(originals.values())[:2]:
                store(here + "/" + other, data if other == path else load(example + "/" + other))
            status, last = run(*parse, here + "/ivk", example + "/example.req", example + "/example.proof", here + "/out")
        say(status == 1 and last.startswith("REFUSED: ") and not os.listdir(here + "/out"), "%s, %s: status %d: %s" % (path, name, status, last[:120]))
damaged_originals = case
for stem, path in originals.items():
    endings = [e for e in ("", ".order", ".seeds0", ".seeds") if stem + ".parsed.bin" + e in wrote]
    for hit in endings:
        whole = load("%s/parsed/%s.parsed.bin%s" % (work, stem, hit))
        for name, data in [("bit in the first byte", flipped(whole, 0))] + [("bit in the last byte", flipped(whole, len(whole) - 1))] * (hit == "") + [("last byte cut off", whole[:-1]), ("a byte added", whole + b"\0")]:
            case += 1
            here = "%s/case-%d" % (work, case)  # the parsed file and the files beside it again, one of them damaged
            os.mkdir(here)
            for e in endings:
                store(here + "/x.parsed.bin" + e, data if e == hit else load("%s/parsed/%s.parsed.bin%s" % (work, stem, e)))
            status, last = run(*read, here + "/x.parsed.bin", "--config=" + example + "/config", "--file=%s/%s" % (example, path))
            say(status == 1 and last.startswith("REFUSED: "), "%s.parsed.bin%s, %s: status %d: %s" % (stem, hit, name, status, last[:120]))
print("parser_test.py: %s: 3 files, %d damaged originals, %d damaged parsed, order and seeds files: %s" % (os.path.basename(example), damaged_originals, case - damaged_originals, "all as expected" if not bad else "%d NOT AS EXPECTED" % bad))
sys.exit(1 if bad else 0)
