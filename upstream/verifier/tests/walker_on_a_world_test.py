#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs the format document's walker (a proof-file parser) on a proof from the prover, cp_prove.

usage: walker_on_a_world_test.py <the tree's folder> <unit.o> <world> <new work folder> <the field's number: 2>
  unit.o   verifier/src/cp_verify.cpp (the verifier program) compiled with -DCP_VERIFY_AS_A_UNIT
           (the test build.verifier.unit makes it)
  world    an honest small world: config/ (policy.txt, layout.txt, KL.bin), ivk/leaf/<k>.bin, in/example.req, out/example.proof
  new work folder   must not exist: the test creates it, with mode 0700, and leaves it in place
  the field's number   2 is p = 2^64 - 2^32 + 1 (Goldilocks), the programs' only field

The walker, docs/format/read_proof.py, reads a proof file by the rules of
docs/format/FORMAT.md only, except that the functions for its rules R15.x (they form the first
hash chain's item `request` from policy.txt, KL.bin and the request file) were first written from
cp_verify's source. It recomputes the proof's three hash chains (Fiat-Shamir transcripts), tests
every nonce of a proof of work, and opens Merkle trees at the drawn positions, at the depth and
width that the document's rules give. It fails unless a tree opens at every draw of positions, every
tree of the three chains is opened at least once and no byte is left. Apart from positions and seals
(digests that end a chain on which another builds), nothing that it draws from a chain affects the
result. It makes none of the protocol's checks: that it reads a file to the end shows that the
document describes the file's layout, not that the file proves anything. It takes as given the
parameter bytes of each chain's first query (the hash input that starts the chain): the plan, which
lists the rounds, and the items that the programs derive from the public files.
verifier/tests/first_queries_tool.cpp writes these bytes from the configuration that the programs'
own code computes. No other test runs the walker; on the toy proofs that
blocks/frame/handover_proof.cpp writes it has been run by hand. Which files it has read:
docs/format/FORMAT.md, "17.3 The scripts".

Checks: (1) The proof's header (printed: `head`) names a leaf table (one of a commitment's copies of
the weight table, each under its own masks; printed: `copy`). With that table's leaf file
(ivk/leaf/<k>.bin; printed: `entry`) the walker exits with status 0, and its last line says that the
whole file is read and gives the file's length.
(2) Negative control, if ivk/leaf/ has another leaf file: with that one the walker fails, because
the first chain's statement contains the table's root, so that every value of the chain changes.
It fails with "work in round" if policy.txt asks for proof of work, and otherwise with
"no tree opens behind them".
(3) Three copies of the proof with openings removed (the last; the first; all): on each the walker
exits with status 1. Openings enter no query, so only the walker's two conditions on openings can
detect the removal. The leaf table's own opening is never removed.
(4) The walker's limit: without the second opening of a tree that is opened twice, the file is still
read to the end, since the walker finds by trial which trees a draw of positions opens and does not
know which trees it must open.
Exit status 0: all as expected. 1: something is not. 2: a wrong call."""
import os
import re
import shutil
import struct
import subprocess
import sys

if len(sys.argv) != 6 or sys.argv[5] != "2":
    print(__doc__)
    sys.exit(2)
if os.path.lexists(sys.argv[4]):
    print("walker_on_a_world_test.py: %s exists already: name a new work folder" % sys.argv[4], file=sys.stderr)
    sys.exit(2)
tree, unit, world, work = (os.path.abspath(x) for x in sys.argv[1:5])
field = sys.argv[5]
os.makedirs(os.path.dirname(work), exist_ok=True)
os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
os.makedirs(work + "/top/batch")
tool = work + "/first_queries_tool"
build = subprocess.run(["g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-w", "-pthread", "-I.", "-I../../blocks", "-I../../prover", "-DBLK_FIELD=" + field, "-DDIGEST_WORDS=8",
                        "../tests/first_queries_tool.cpp", unit, "-o", tool], cwd=tree + "/verifier/src", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
if build.returncode:
    print(build.stdout[-2000:])
    sys.exit("walker_on_a_world_test.py: the tool does not build")
for name in ("policy.txt", "layout.txt", "KL.bin"):
    shutil.copy(world + "/config/" + name, work + "/top/" + name)
shutil.copy(world + "/out/example.proof", work + "/top/batch/example.proof")
shutil.copy(world + "/in/example.req", work + "/top/batch/example.request")
copies = sorted(int(n[:-4]) for n in os.listdir(world + "/ivk/leaf") if n.endswith(".bin") and n[:-4].isdigit())
for k in copies:
    shutil.copy("%s/ivk/leaf/%d.bin" % (world, k), "%s/top/leaf-%d.bin" % (work, k))
# 128: the tool's argument <asked bits>.
wrote = subprocess.run([tool, work + "/top", "128", work + "/top/batch", "example"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(wrote.stdout, end="")
if wrote.returncode:
    sys.exit("walker_on_a_world_test.py: the tool ends with status %d" % wrote.returncode)
with open(work + "/top/batch/example.proof", "rb") as f:
    own = struct.unpack("<3I", f.read(12))[2]  # the header's third word: the leaf table's number
length = os.path.getsize(work + "/top/batch/example.proof")


# Runs the walker with leaf file k, on top/batch/<file> if given, else on the proof.
def walk(k, file=None):
    r = subprocess.run([sys.executable, "-B", tree + "/docs/format/read_proof.py", "top", "batch", "example"] + ([file] if file else []) + ["--field=" + field, "--public=policy.txt,layout.txt,KL.bin",
                        "--table=leaf-%d.bin" % k, "--statement=policy.txt,KL.bin"], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open("%s/walker.%s.txt" % (work, file or "leaf-%d" % k), "w", encoding="utf-8") as f:
        f.write(r.stdout)
    lines = [l for l in r.stdout.split("\n") if l.strip()]
    return r.returncode, lines[-1] if lines else ""


bad = 0
if own not in copies:
    sys.exit("walker_on_a_world_test.py: the proof's head names the copy %d, and the world has no entry of that number" % own)
status, last = walk(own)
ok = status == 0 and last.endswith("the whole file is read: %d bytes" % length)
bad += not ok
print("%s under its own entry (copy %d): status %d: %s" % ("as expected    " if ok else "NOT AS EXPECTED", own, status, last[:160]))
others = [k for k in copies if k != own]
with open(world + "/config/policy.txt", encoding="ascii") as f:
    asks_for_work = any(int(x) for l in f if l.startswith("work ") for x in l.split()[2:])
for k in others[:1]:
    status, last = walk(k)
    ok = status != 0 and ("work in round" if asks_for_work else "positions are drawn and no tree opens behind them") in last
    bad += not ok
    print("%s under another copy's entry (copy %d): status %d: %s" % ("as expected    " if ok else "NOT AS EXPECTED", k, status, last[:160]))
# The openings of trees whose roots are in the proof, as (first byte, end, tree), from the walker's
# output in check (1). set() drops repeats: a failed try of the second chain prints some again.
with open("%s/walker.leaf-%d.txt" % (work, own), encoding="utf-8") as f:
    openings = sorted(set((int(a), int(b), who) for who, a, b in re.findall(r"\((number \d+ of [a-z ]+); depth \d+, width \d+, \d+ columns; bytes (\d+) to (\d+)\)", f.read())))
with open(work + "/top/batch/example.proof", "rb") as f:
    proof = f.read()
again = [o for i, o in enumerate(openings) if o[2] in [p[2] for p in openings[:i]]]  # the later openings of trees that are opened more than once


def without(cut):
    kept, at = b"", 0
    for a, b, _ in cut:
        kept, at = kept + proof[at:a], b
    return kept + proof[at:]


# `sound`: at least three printed openings, disjoint, the last ending the file, and some tree opened
# twice. If not, the test fails and skips checks (3) and (4).
sound = len(openings) > 2 and all(a < b for a, b, _ in openings) and all(x[1] <= y[0] for x, y in zip(openings, openings[1:])) and openings[-1][1] == length and len(again) > 0
bad += not sound
print("%s %d openings of %d trees in the walker's print, apart from each other, the last at the file's end; %d of them a tree's second" % ("as expected    " if sound else "NOT AS EXPECTED", len(openings), len(set(o[2] for o in openings)), len(again)))
for name, cut, whole in (("the-last-opening-cut-off", openings[-1:], False), ("the-first-opening-taken-out", openings[:1], False), ("all-openings-taken-out", openings, False), ("a-second-opening-taken-out", again[:1], True)) if sound else ():
    with open("%s/top/batch/%s" % (work, name), "wb") as f:
        f.write(without(cut))
    status, last = walk(own, name)
    ok = (status == 0 and last.endswith("the whole file is read: %d bytes" % len(without(cut)))) == whole and (whole or status == 1)
    bad += not ok
    print("%s %s (%d bytes)%s: status %d: %s" % ("as expected    " if ok else "NOT AS EXPECTED", name, len(without(cut)), ": THE LIMIT: read whole" if whole else "", status, last[:160]))
print("walker_on_a_world_test.py: field %s: a proof of %d bytes, %d control(s): %s" % (field, length, len(others[:1]), "all as expected" if not bad else "%d NOT AS EXPECTED" % bad))
sys.exit(1 if bad else 0)
