#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs the verifier program cp_verify on honest files, then on copies altered to drive it out of
an array or into undefined arithmetic.

No call on a copy may pass or crash. Apart from what the last list below reports, any rejection will
do, even a usage error (status 2). The test is meant for a cp_verify built with the compiler's
sanitizers ("verifier/tests/sanitized_walk_test.sh --world" runs it on one). On the ordinary build
it detects everything listed below except what only a sanitizer reports.

usage: extreme_proof_files_test.py <cp_verify> <new work folder> <fixture> [--jobs <n>]
       [--places <n>]
<fixture>  honest files made from synthetic weights (`world` in the code and the printed usage;
           verifier/tests/other_parameters_worlds.sh makes such fixtures): config/,
           ivk/key.bin and ivk/leaf/<n>.bin (the commitment's public files), in/example.req,
           out/example.y, out/example.proof, id.txt
--jobs     calls at a time (default 8)
--places   32-bit words, evenly spread, altered in each file besides its first 32 and last 2
           (default 32)
The work folder must not exist, not even as a broken symbolic link. It is created closed to other
users, after any missing parent folders (these with the default mode). Altered copies are written to
it, and only those of reported calls stay. The script does not remove the folder.

One file is altered at a time, in the call that reads it, always with -id (the commitment ID, from
id.txt):
  key            ivk/key.bin, the key file: "cp_verify -key"
  `entry`        the leaf file ivk/leaf/<n>.bin, n from the proof's header: "cp_verify -leaf"
  proof, req, y  proof file, request, token file: the verification call

Every file is truncated (to 8 bytes or fewer, to each sixteenth of its length, just before its end),
extended by 1 to 2^20 bytes of 00 or of ff, and has each chosen word set to 0, 2^31 and 2^32 - 1.

The other cases take offsets ("places" in messages) from the planning call
"cp_verify -plan request .. -fields", which lists the fields of key, entry and proof. Offsets are
exact only up to and including a file's first field whose length depends on how many drawn positions
are distinct (an opening of a Merkle tree, or the list of positions before it), and only fields with
exact offsets are used: in a proof, the header, the first hash chain's salt (random bytes), the
first segment (the first sub-proof's part) and what follows up to the first opening. Fields whose
names differ only in numbers form a group, of which the first and the last field are taken. In the
proof, the first eight bytes of each taken field are set to 00 and to ff.

In all three files, fields of the class "elements" get non-canonical coordinates. A coordinate is
eight bytes and must be below the prime p = 2^64 - 2^32 + 1. By the end of a case's name:
  second-spelling  the field's first or last coordinate plus p: a second encoding of the same
                   element; only if the sum fits eight bytes (few do)
  ff               the same coordinates set to eight bytes ff, above p
  p, p-plus-1      the first coordinate of a group's first field set to p or p + 1: second encodings
                   of 0 and 1, the smallest non-canonical values
  p-minus-1        the control: the same coordinate set to the largest canonical value
All but the control must be rejected as non-canonical: a line of the output contains "canonical".
The control must be rejected by a later check: no line contains "canonical".
Two fields are an exception: the stored answers of a table's record, "table: record" in the key and
"further table: record" in the entry. The list has them only under the parameter line "record 1".
The code that reads them (blocks/frame/check_key.h, check_entry.h) rejects with the reason
"the record's answers", for a non-canonical answer as for a file that ends too early, and does not
say "canonical". So for the cases in these two fields, and for no others, a line of the output must
contain that reason instead. In each of the two files they are: ff in the field's first and last
coordinate, p and p + 1 in its first, and a second-spelling where one fits. Their control must be
rejected by a later check: no line contains that reason or "canonical". The closing lines count
these cases apart. Under "record 1" the test stops with status 2 if the list lacks either field.
In a field "opened symbols", the coordinates are the first and the last of its first position.
The -fields list gives an opening's symbols and then its salts, each at its largest size, but the
file has, for each distinct drawn position, the symbols and then the salt
(docs/format/FORMAT.md, R7.8), so only the first position is at the listed offset.

A call is reported on one line, under the first of these that applies; the counts come at the end:
  slow         no end within 600 s; cp_verify is then killed, but not its child processes, which may
               run on
  fault        a signal, a status that is not cp_verify's own (a sanitizer's 97 among them), its
               status 15 (the answer could not be written), or output with a sanitizer's line or a
               failed assertion of the C++ library
  not-checked  status 5 (NOT CHECKED: cp_verify itself failed, as when a signal ends one of its
               child processes), or a last line "NO: <name>" whose name has "internal-" at its
               start or after a sub-proof's prefix
  honest       a call on the honest files did not pass; status 14 (NOT-A-VERIFICATION) is no pass
  accepted     a call on an altered file passed, or ended with status 14 (no check that ran
               failed); printed as "WAS NOT REFUSED"
  unsaid       the required words ("canonical"; in a record's stored answers "the record's answers")
               are missing, or the control's output has them
Exit status 0: no such line. 1: some. 2: wrong arguments, or an unexpected list of fields.
3: an altered file WAS NOT REFUSED.
A run without a fault shows that these inputs led the program to no fault that this build can
detect. It shows nothing about other inputs, and nothing about soundness.
"""

import argparse
import concurrent.futures
import os
import re
import struct
import subprocess
import sys

# OWN leaves out 15 (the answer could not be written): this test reads the output to its end, so a
# 15 would be a fault.
OWN = (0, 1, 2, 3, 4, 5, 14)  # cp_verify's own statuses (verifier/docs/output.md, "Exit codes")
# Statuses that a call on an altered file must not have.
PASSES = (0, 3, 14)
WORDS = (0, 0x80000000, 0xFFFFFFFF)
LIMIT = 600
# The field of each file that holds a record's stored answers, as the planning call's list names it,
# and the one reason with which the routines that read it reject.
RECORD = {"key": "table: record", "entry": "further table: record"}
RECORD_REASON = "the record's answers"


def cuts(n):
    return sorted(x for x in {0, 1, 3, 4, 5, 8, n - 4, n - 1} | {n * i // 16 for i in range(1, 16)} if 0 <= x < n)


def main():
    ap = argparse.ArgumentParser(usage="extreme_proof_files_test.py <cp_verify> <new work folder> <world> [--jobs <n>] [--places <n>]")
    ap.add_argument("program"), ap.add_argument("work"), ap.add_argument("world")
    ap.add_argument("--jobs", type=int, default=8), ap.add_argument("--places", type=int, default=32)
    a = ap.parse_args()
    if os.path.lexists(os.path.abspath(a.work)) or a.jobs < 1 or a.places < 0 or not os.access(a.program, os.X_OK):
        ap.print_usage(sys.stderr)
        return 2
    program, work, world = os.path.abspath(a.program), os.path.abspath(a.work), os.path.abspath(a.world)
    config = os.path.join(world, "config")
    with open(os.path.join(world, "id.txt"), encoding="ascii") as f:
        named = f.read().strip()
    paths = {"proof": os.path.join(world, "out", "example.proof"), "req": os.path.join(world, "in", "example.req"), "y": os.path.join(world, "out", "example.y"),
             "key": os.path.join(world, "ivk", "key.bin")}  # fmt: skip
    own = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            own[name] = f.read()
    number = struct.unpack("<I", own["proof"][8:12])[0]  # third word of a proof's header: the number of its leaf file
    paths["entry"] = os.path.join(world, "ivk", "leaf", "%d.bin" % number)
    with open(paths["entry"], "rb") as f:
        own["entry"] = f.read()

    def words_of(which, copy):
        """Arguments of the call that reads the file `which`, with `copy`, if any, in its place.

        -show-stages: if a stage rejects, the call prints the stages' output too, with the reason
        (unless the files' ID differs from the one given with -id).
        """
        at = lambda n: copy if copy and n == which else paths[n]
        if which == "key":
            return ["-key", config, at("key"), "-id", named, "-q", "-show-stages"]
        if which == "entry":
            return ["-leaf", config, at("key"), at("entry"), "-id", named, "-q", "-show-stages"]
        return [config, os.path.join(world, "ivk"), at("req"), at("y"), at("proof"), "-id", named, "-q", "-show-stages"]

    say, hush = {}, {}  # case name -> words that the call's output must contain; a tuple of words that it must not contain
    in_record = set()  # the names of the cases in a record's stored answers
    cases = [("honest." + w, w, None) for w in ("key", "entry", "proof")]
    for which in ("key", "entry", "proof", "req", "y"):
        b = own[which]
        for x in cuts(len(b)):
            cases.append(("%s.cut-%d" % (which, x), which, b[:x]))
        for more in (1, 4, 4096, 1 << 20):
            cases.append(("%s.more-%d" % (which, more), which, b + bytes(more)))
            cases.append(("%s.more-ff-%d" % (which, more), which, b + b"\xff" * more))
        n = len(b) // 4
        for i in sorted(x for x in set(range(32)) | {n - 2, n - 1} | {n * k // (a.places + 1) for k in range(1, a.places + 1)} if 0 <= x < n):
            for w in WORDS:
                new = struct.pack("<I", w)
                if b[4 * i : 4 * i + 4] != new:
                    cases.append(("%s.word-%d=%d" % (which, i, w), which, b[: 4 * i] + new + b[4 * i + 4 :]))
    plan = subprocess.run([program, "-plan", "request", os.path.join(config, "policy.txt"), os.path.join(config, "KL.bin"), "128", "-layout", os.path.join(config, "layout.txt"),
                           "-prompt", paths["req"], "-reply", paths["y"], "-fields", "-q"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False)  # fmt: skip
    groups, elements = {}, {"key": {}, "entry": {}, "proof": {}}
    salts, before = {}, None  # {(file, offset of a field "opened symbols"): the listed bytes of its salts}; the line before
    for l in plan.stdout.split("\n"):  # fields after a file's first one of variable length are printed with "at-most"
        m = re.match(r"^field: (key|entry|proof) at-most [0-9]+ ([0-9]+) salt: .*salts$", l)
        if m and before and before.group(1) == m.group(1) and before.group(5).endswith("opened symbols"):
            salts[m.group(1), int(before.group(2))] = int(m.group(2))
        m = before = re.match(r"^field: (key|entry|proof) at ([0-9]+) ([0-9]+) ([a-z]+): (.*)$", l)
        if m and m.group(1) == "proof":
            groups.setdefault((m.group(4), re.sub(r"[0-9]+", "#", m.group(5))), []).append((int(m.group(2)), int(m.group(3))))
        if m and m.group(4) == "elements":
            elements[m.group(1)].setdefault(re.sub(r"[0-9]+", "#", m.group(5)), []).append((int(m.group(2)), int(m.group(3))))
    if plan.returncode != 4 or len(groups) < 10:
        print("extreme_proof_files_test.py: the planning call gives %d kinds of fields with exact places, status %d" % (len(groups), plan.returncode), file=sys.stderr)
        return 2
    if any(RECORD[w] not in elements[w] for w in RECORD):
        print("extreme_proof_files_test.py: the planning call's list lacks the field of a record's stored answers in the key or the entry", file=sys.stderr)
        return 2
    b = own["proof"]
    for at, size in sorted({g[0] for g in groups.values()} | {g[-1] for g in groups.values()}):
        for name, fill in (("00", b"\0"), ("ff", b"\xff")):
            new = fill * min(8, size)
            if b[at : at + len(new)] != new:
                cases.append(("proof.field-at-%d=%s" % (at, name), "proof", b[:at] + new + b[at + len(new) :]))
    with open(os.path.join(config, "policy.txt"), encoding="ascii") as f:
        # cp_verify accepts only the field number 2 (blocks/frame/version.h): p = 2^64 - 2^32 + 1.
        prime = (1 << 61) - 1 if f.readline().split()[1][2] == "1" else (1 << 64) - (1 << 32) + 1  # the version word's third character is the field's number
        salt = int([l.split()[1] for l in f if l.startswith("salt ")][0])  # a salt's bytes
    for which, kinds in elements.items():
        b = own[which]
        for at, size, opening, record in sorted({(*g[i], k.endswith("opened symbols"), k == RECORD.get(which)) for k, g in kinds.items() for i in (0, -1)}):
            if opening:  # the bytes of its first position's symbols
                if salts.get((which, at), 0) < salt or salts[which, at] % salt or size % (salts[which, at] // salt):
                    print("extreme_proof_files_test.py: the planning call's list has no salts behind the opened symbols at %d of the %s" % (at, which), file=sys.stderr)
                    return 2
                size //= salts[which, at] // salt
            for x in sorted({at, at + size - 8}):  # its first and its last coordinate
                v = int.from_bytes(b[x : x + 8], "little")
                for name, new in (("second-spelling", v + prime), ("ff", (1 << 64) - 1)):
                    if new < 1 << 64:
                        cases.append(("%s.element-at-%d.%s" % (which, x, name), which, b[:x] + new.to_bytes(8, "little") + b[x + 8 :]))
                        say[cases[-1][0]] = RECORD_REASON if record else "canonical"
                        in_record |= {cases[-1][0]} if record else set()
        for at, record in sorted({(g[0][0], k == RECORD.get(which)) for k, g in kinds.items()}):
            for name, new in (("p", prime), ("p-plus-1", prime + 1), ("p-minus-1", prime - 1)):
                if b[at : at + 8] != new.to_bytes(8, "little"):
                    cases.append(("%s.element-at-%d.%s" % (which, at, name), which, b[:at] + new.to_bytes(8, "little") + b[at + 8 :]))
                    if name == "p-minus-1":
                        hush[cases[-1][0]] = ("canonical", RECORD_REASON) if record else ("canonical",)
                    else:
                        say[cases[-1][0]] = RECORD_REASON if record else "canonical"
                    in_record |= {cases[-1][0]} if record else set()

    # Runs one case; the result "status-<n>" is not reported.
    def one(numbered):
        n, (name, which, altered) = numbered
        copy = None
        if altered is not None:
            copy = os.path.join(work, "%05d.%s" % (n, which))
            with open(copy, "wb") as f:
                f.write(altered)
        try:
            r = subprocess.run([program] + words_of(which, copy), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False, timeout=LIMIT)
        except subprocess.TimeoutExpired:
            return name, "slow", "no end within %d s" % LIMIT
        lines = r.stdout.strip().split("\n")
        sign = next((l for l in lines if "runtime error:" in l or "ERROR: AddressSanitizer" in l or ("Assertion '" in l and "' failed" in l)), None)
        if r.returncode not in OWN or sign:
            kind = "fault"
        elif r.returncode == 5 or re.match(r"NO: (\w+:)?internal-", lines[-1]):
            kind = "not-checked"
        elif altered is None:
            kind = "status-%d" % r.returncode if r.returncode in (0, 3) else "honest"
        elif r.returncode in PASSES:
            kind = "accepted"
        elif (name in say and not any(say[name] in l for l in lines)) or (name in hush and any(w in l for w in hush[name] for l in lines)):
            kind = "unsaid"
        else:
            kind = "status-%d" % r.returncode
        if copy and kind.startswith("status-"):
            os.remove(copy)  # the file of a reported call stays
        return name, kind, "status %d | %s" % (r.returncode, (sign or lines[-1])[:200])

    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    counts, ended = {}, {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        for name, kind, said in pool.map(one, enumerate(cases)):
            counts[kind] = counts.get(kind, 0) + 1
            if name in say or name in hush:
                under = (name.split(".")[0] + ("'s record" if name in in_record else ""), name in hush, said.split(" | ", 1)[1])
                ended[under] = ended.get(under, 0) + 1
            if not kind.startswith("status-"):
                print("%s | %s | %s" % ("WAS NOT REFUSED" if kind == "accepted" else kind, name, said), flush=True)
    bad = {k: counts.get(k, 0) for k in ("fault", "not-checked", "slow", "honest", "accepted", "unsaid")}
    for w in ("key", "entry", "proof") + (("key's record", "entry's record") if in_record else ()):
        mine = lambda n: n.startswith(w.split("'")[0] + ".") and (n in in_record) == w.endswith("record")
        print("second encodings in the %s: %d: %s" % (w, sum(1 for n in say if mine(n)), ", ".join("%s %d" % (l, n) for (x, c, l), n in sorted(ended.items()) if x == w and not c)))
        print("the control p - 1 in the %s: %d: %s" % (w, sum(1 for n in hush if mine(n)), ", ".join("%s %d" % (l, n) for (x, c, l), n in sorted(ended.items()) if x == w and c)))
    print("extreme_proof_files_test.py: 3 honest calls and %d altered files: refused %d, usage %d; fault %d, not-checked %d, slow %d, honest calls not as expected %d, not refused %d, "
          "second encodings or their control not as asked %d"
          % (len(cases) - 3, counts.get("status-1", 0), counts.get("status-2", 0), bad["fault"], bad["not-checked"], bad["slow"], bad["honest"], bad["accepted"], bad["unsaid"]))  # fmt: skip
    return 3 if bad["accepted"] else 1 if any(bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
