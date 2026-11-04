# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by openings_left_out_test.py or hostile_leaf_files_test.py, which the runner starts; by hand: below a directory of your own.
"""What openings_left_out_test.py and hostile_leaf_files_test.py share: both change one file of an
honest set at places that tools/audit/cut.py names, and ask that cp_verify, the verifier, refuses it.

The tests' command line:
  <the tree's folder> <unit.o> <cp_verify> <set> <new work folder> [<calls at a time>]
  unit.o   verifier/src/cp_verify.cpp compiled with -DCP_VERIFY_AS_A_UNIT (the test
           build.verifier.unit makes it)
  set      an honest set as under examples/: config/, ivk/key.bin, ivk/leaf/0.bin, example.req,
           example.y, example.proof, ID.txt
  new work folder   must not exist: the test creates it, with mode 0700, and folders that are
           missing above it, with the default mode; it leaves it in place
  calls at a time   1 to 64; 4 if not given
cut.py parses key file, leaf file and proof file into records, by the plans that
tools/audit/pubtool.cpp writes from the configuration; begin() builds that tool against unit.o.
The files of a changed set that was refused as expected are removed again; all others are kept.
The tool, its tables and the cases' empty folders stay too.
The tests' exit status: 0: all as expected. 3: a changed set WAS NOT REFUSED. 1: something else is
not as expected. 2: a wrong call."""
import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import sys

KEY, LEAF, PROOF = "ivk/key.bin", "ivk/leaf/0.bin", "example.proof"
REFUSAL = {KEY: "NO: key-refused", LEAF: "NO: entry-refused", PROOF: "NO: proof-refused"}  # the last line of a call on a set with that file changed


class Set:
    """verifier, honest (the set's folder), work, at_a_time, ID; whole[path]: a file's bytes;
    records[path]: cut.py's records of it, in the file's order; prime: the field's p."""


def begin(usage):
    a = sys.argv
    if len(a) not in (6, 7) or a[6:] and a[6] not in map(str, range(1, 65)) or not os.path.isfile(a[1] + "/tools/audit/cut.py") or sys.flags.optimize:
        print(usage + "\n" + __doc__ + "\n(Not under python3 -O: cut.py works by its assertions.)")
        sys.exit(2)
    if os.path.lexists(a[5]):
        print("%s exists already: name a new work folder" % a[5], file=sys.stderr)
        sys.exit(2)
    s = Set()
    tree, unit, s.verifier, s.honest, s.work = (os.path.abspath(x) for x in a[1:6])
    s.at_a_time = int(a[6]) if a[6:] else 4
    os.makedirs(os.path.dirname(s.work), exist_ok=True)
    os.mkdir(s.work, 0o700)  # fails if the name is taken; closed to other users
    os.mkdir(s.work + "/public")
    for what, words, at in (("tools/audit/pubtool.cpp does not build", ["g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-w", "-pthread", "-I.", "-I../../blocks", "-I../../prover", "-DBLK_FIELD=2", "-DDIGEST_WORDS=8",
                                                                        "../../tools/audit/pubtool.cpp", unit, "-o", s.work + "/pubtool"], tree + "/verifier/src"),
                            ("pubtool fails on the set's config/", [s.work + "/pubtool", s.honest + "/config", "128", s.work + "/public"], None)):
        r = subprocess.run(words, cwd=at, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            print(r.stdout[-2000:])
            sys.exit("%s (status %d)" % (what, r.returncode))

    def rd(name, folder=s.honest):
        with open(os.path.join(folder, name), "rb") as f:
            return f.read()

    sys.path[:0] = [tree + "/docs/format", tree + "/tools/audit"]
    import cut
    import format_rules as R

    s.ID, s.prime, s.whole = rd("ID.txt").decode("ascii").strip(), R.PRIME[2], {path: rd(path) for path in (KEY, LEAF, PROOF)}
    policy, scales = rd("config/policy.txt"), rd("config/KL.bin")
    C = cut.Cutter(R, 2, R.configuration_digest([policy, rd("config/layout.txt"), scales]), lambda n: rd(n, s.work + "/public"))
    key = C.key(KEY, s.whole[KEY])
    leaf = C.entry(LEAF, s.whole[LEAF], key["root"], 0, key["answers"])
    C.proof(PROOF, s.whole[PROOF], {0: leaf["root"]}, policy, scales, rd("example.req"), {0: leaf["answers"]})
    s.records = {path: [r for r in C.recs if r.file == path] for path in s.whole}
    for path, recs in s.records.items():
        assert sum(r.n for r in recs) == len(s.whole[path]) and all(x.off + x.n == y.off for x, y in zip(recs, recs[1:])), "the records do not cover " + path
    return s


def openings(records):
    """The openings in a file: for each, its records: those of one tree after one round."""
    runs = []
    for r in records:
        if r.tree is not None:
            which = (r.chain, r.rnd, id(r.tree))
            if runs and runs[-1][0] == which and runs[-1][1][-1].off + runs[-1][1][-1].n == r.off:
                runs[-1][1].append(r)
            else:
                runs.append((which, [r]))
    assert runs and len({which for which, _ in runs}) == len(runs), "no opening, or the records of one do not lie together"
    return [recs for _, recs in runs]


def play(s, cases, part=False):
    """Gives cp_verify the unchanged set and then each case: (the changed file's path; a name for the
    case; a function that makes the file's new bytes; a text, or none). Each gets a whole call under
    the set's ID and, if `part` (every case changes the leaf file), the part-call -leaf before it.
    Every call on a case must end with status 1 and the file's line of REFUSAL, and the line before
    it, in which cp_verify says why, must hold the case's text, or, if that begins with "not ", must
    not hold the rest of it. Ends the process."""

    def one(path, name, make, why):
        at, ivk, proof, wrote, data = os.path.join(s.work, name), s.honest + "/ivk", s.honest + "/" + PROOF, [path], b""
        if path == PROOF:
            os.mkdir(at)
            proof = at + "/" + PROOF
        elif path:  # cp_verify takes key file and leaf file from one folder: the other of the two is copied
            os.makedirs(at + "/ivk/leaf")
            ivk, other = at + "/ivk", LEAF if path == KEY else KEY
            shutil.copy(os.path.join(s.honest, other), os.path.join(at, other))
            wrote.append(other)
        if path:
            data = make()
            with open(os.path.join(at, path), "wb") as f:
                f.write(data)
        calls = [[s.verifier, "-leaf", s.honest + "/config", ivk + "/key.bin", ivk + "/leaf/0.bin", "-id", s.ID, "-q"]] if part else []
        calls.append([s.verifier, s.honest + "/config", ivk, s.honest + "/example.req", s.honest + "/example.y", proof, "-id", s.ID, "-q"])
        said = []
        for words in calls:
            r = subprocess.run(words, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            said.append((r.returncode, ["", ""] + [l[:150] for l in r.stdout.decode("utf-8", "replace").split("\n") if l.strip()]))
        if not path:
            end = "as expected" if [st for st, _ in said] == ([3, 0] if part else [0]) else "NOT AS EXPECTED"
        elif all(st == 1 and l[-1] == REFUSAL[path] and (not why or (why[4:] not in l[-2] if why.startswith("not ") else why in l[-2])) for st, l in said):
            end = "as expected"
            for file in wrote:
                os.remove(os.path.join(at, file))
        else:
            end = "WAS NOT REFUSED" if any(st in (0, 3, 14) for st, _ in said) else "NOT AS EXPECTED"
        if end != "as expected":  # said at once: a fault of the test itself in a later case must not hide it
            print("%s: %s: %s" % (end, name, "; ".join("status %d: %s | %s" % (st, l[-2], l[-1]) for st, l in said)), flush=True)
        return end, hashlib.sha256(data).digest(), [l[-2] for _, l in said]

    with concurrent.futures.ThreadPoolExecutor(s.at_a_time) as pool:
        out = list(pool.map(lambda c: one(*c), [(None, "the unchanged set", None, None)] + cases))
    assert len({digest for _, digest, _ in out[1:]} | {hashlib.sha256(x).digest() for x in s.whole.values()}) == len(cases) + 3, "two cases are the same file, or one is the honest file"
    for why in sorted({x for o in out[1:] for x in o[2]}):
        print("%4d x %s" % (sum(x == why for o in out[1:] for x in o[2]), why))
    ends = [end for end, _, _ in out[1:]]
    print("%s: %d changed sets, %d calls: %d refused, %d NOT REFUSED, %d otherwise; the unchanged set %s" % (
        os.path.basename(sys.argv[0]), len(ends), sum(len(o[2]) for o in out[1:]), ends.count("as expected"), ends.count("WAS NOT REFUSED"), ends.count("NOT AS EXPECTED"), "passes" if out[0][0] == "as expected" else "DOES NOT PASS"))
    sys.exit(3 if "WAS NOT REFUSED" in ends else 1 if "NOT AS EXPECTED" in ends or out[0][0] != "as expected" else 0)
