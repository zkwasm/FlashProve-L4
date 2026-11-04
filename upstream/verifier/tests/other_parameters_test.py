#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Files that the honest programs made under one configuration are rejected under another.

usage: other_parameters_test.py <cp_verify> <work folder> <calls side by side> <honest status> <least number of sisters> <true world> <sister world> .. <the control's world>
<work folder>   a new folder, or an empty folder of your own that no other account can write

A test fixture (`world` in usage and output) is a folder with the files of one complete honest run
(cp_commit, then cp_prove), made by other_parameters_worlds.sh:
    config/ (the configuration: policy.txt, KL.bin, layout.txt)
    ivk/key.bin   ivk/leaf/0.bin (`ivk`: the public key directory)
    in/example.req   out/example.y   out/example.proof   id.txt (the commitment ID, 64 hex digits)
All fixtures have the same weights, committed under the same test seed, and the same request (only
the request is checked here). A sister fixture's configuration differs from the true one's in one
thing (other_parameters_kits.py): a parameter of the proof system or a constant of the function
that is proved. other_parameters_worlds.sh leaves out, with a printed line, each sister that a
program rejects.

Tests with altered bytes do not show that a proof is tied to the parameters under which it is
checked: a proof made with fewer bits of proof of work or at a lower target (`sec` in policy.txt:
bits required in every round) is a well-formed, honest file. The harm would be a verifier that
accepts it as made under its own parameters. At the configuration of this test's row in SUITE.md
the sister with the lower target (`target-lower`) is left out, so that row gives cp_verify no file
made at a lower target.

For each sister, the verifier cp_verify is called with the true fixture as judge and the
sister as the other, then the reverse. A call uses the judge's configuration, request and files,
except that these files are the other's:
    proof        proof file and token file (out/example.y; `reply` in identifiers)
    key          key file (ivk/key.bin), about the anchor: the commitment's first weight table
    entry        leaf file (ivk/leaf/<k>.bin; `entry` in identifiers), about leaf table k: a further
                 weight table
    key-entry    key file and leaf file
    all          all four
Each mix is called three times: -id names the judge's ID, -id names the other's ID, no -id.

A call on a fixture's own files must end with <honest status>: at best 14, NOT-A-VERIFICATION,
because every call passes -bits 0 (fewer than 128 bits requested). Every other call, except the
control's, must end with status 1, print no line with "internal-", and have the last line that
`FORETOLD` gives for it.
Every hash in a weight table's Merkle tree starts from the configuration digest (a hash of the
configuration's files), so fixtures with different configurations share no root. The commitment ID
is a hash of the configuration digest, the anchor's name and its record's seeds and answers, so an
ID matches only its own key file under its own configuration: a call that names the other's ID, or
the judge's ID with the other's key file, ends with `NO: model-statement` before any proof is
checked. Otherwise the first stage of cp_verify (key, leaf, request, in this order) that gets a file
of the other's rejects it. For a proof file the test accepts two rejection names: chain-refused,
raised where reading a hash chain (a Fiat-Shamir transcript) fails inside a sub-proof, for example
at an unexpected proof-of-work nonce, and proof-refused.
Exactly one sister, the control, must have the true fixture's three configuration files. It has
the same commitment, and a proof from a second run of the prover; this script only prints
whether the two proofs differ. The same thirty calls with the control must each end with
<honest status>: mixing files does not by itself get a call rejected.
A call on another configuration's files that ends with <honest status> or a status in `PASSES` is
printed with `WAS NOT REFUSED`, and the test ends with status 3. Otherwise, if any call does not
end as required, the test ends with status 1. Every such call is printed, and its copies of the
key file and the leaf file are kept.
This script makes <work folder> and any missing parent folders. It writes and removes files only
inside <work folder>."""
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


# Statuses that count as not rejected. Of these, cp_verify returns only 0, 3 and 14, each for a
# call that found no fault (verifier/docs/output.md, "Exit codes").
PASSES = (0, 3, 10, 11, 14)
FORETOLD = {  # (mix, whose ID is named): the expected last line
    ("proof", "the judge's ID"): "NO: proof-refused or NO: chain-refused", ("proof", "no ID"): "NO: proof-refused or NO: chain-refused",
    ("entry", "the judge's ID"): "NO: entry-refused", ("entry", "no ID"): "NO: entry-refused",
    ("key", "the judge's ID"): "NO: model-statement", ("key", "no ID"): "NO: key-refused",
    ("key-entry", "the judge's ID"): "NO: model-statement", ("key-entry", "no ID"): "NO: key-refused",
    ("all", "the judge's ID"): "NO: model-statement", ("all", "no ID"): "NO: key-refused",
}
for what in ("proof", "key", "entry", "key-entry", "all"):
    FORETOLD[(what, "the other's ID")] = "NO: model-statement"


class World:
    def __init__(self, folder):
        self.folder, self.name = folder, os.path.basename(os.path.normpath(folder))
        at = lambda *p: os.path.join(folder, *p)
        self.config, self.key, self.entry = at("config"), at("ivk", "key.bin"), at("ivk", "leaf", "0.bin")
        self.request, self.reply, self.proof = at("in", "example.req"), at("out", "example.y"), at("out", "example.proof")
        self.id = open(at("id.txt")).read().strip()
        if len(self.id) != 64:
            sys.exit("other_parameters_test: %s: id.txt does not hold 64 hex digits" % folder)
        for f in (self.key, self.entry, self.request, self.reply, self.proof, at("config", "policy.txt"), at("config", "KL.bin"), at("config", "layout.txt")):
            if not os.path.isfile(f):
                sys.exit("other_parameters_test: no file " + f)


def same(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


def main():
    if len(sys.argv) < 8:
        sys.exit(__doc__)
    program, work, side, honest, least = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])  # absolute, because subprocess looks a bare name up on PATH, not in the current directory
    true, sisters = World(sys.argv[6]), [World(f) for f in sys.argv[7:]]
    os.makedirs(os.path.dirname(work), exist_ok=True)
    # The calls below copy files into folders inside `work` and remove those folders, so `work`
    # must not be writable by another account: it is created here, or it is an empty folder that
    # this account owns and that group and others cannot write.
    try:
        os.mkdir(work, 0o700)  # fails if anything, even a symbolic link, already has this name
    except FileExistsError:
        if os.path.islink(work) or not os.path.isdir(work) or os.lstat(work).st_uid != os.geteuid() or os.lstat(work).st_mode & 0o022 or os.listdir(work):
            sys.exit("other_parameters_test: %s exists and is not an empty folder of your own that only you can write: name a new folder" % work)
    for s in sisters:
        if not same(s.request, true.request):
            sys.exit("other_parameters_test: %s has another request" % s.name)
        s.control = all(same(os.path.join(s.config, f), os.path.join(true.config, f)) for f in ("policy.txt", "KL.bin", "layout.txt"))
    if sum(s.control for s in sisters) != 1 or len(sisters) - 1 < least:
        sys.exit("other_parameters_test: %d sisters and %d controls; at least %d and exactly one are wanted" % (sum(not s.control for s in sisters), sum(s.control for s in sisters), least))

    def call(label, judge, key, entry, reply, proof, named):
        ivk = os.path.join(work, label.replace(" ", "_").replace("|", "").replace("/", "_"))
        os.makedirs(os.path.join(ivk, "leaf"))  # must not exist yet: each call has its own label (two fixtures with one folder name would clash here)
        shutil.copyfile(key, os.path.join(ivk, "key.bin"))
        # cp_verify reads the leaf file whose number is the proof file's third 32-bit word, which
        # names the leaf table (`copy` in identifiers) that the proof was made against. A fixture
        # may have several leaf files: the one of that number is copied if it exists, else `entry`.
        with open(proof, "rb") as f:
            number = struct.unpack_from("<I", f.read(12), 8)[0]
        of_that_copy = os.path.join(os.path.dirname(entry), "%d.bin" % number)
        shutil.copyfile(of_that_copy if os.path.isfile(of_that_copy) else entry, os.path.join(ivk, "leaf", "%d.bin" % number))
        words = [program, judge.config, ivk, judge.request, reply, proof, "-bits", "0", "-j", "1", "-show-stages"] + (["-id", named] if named else [])  # -show-stages: print the stages' output also if a stage rejects (not with model-statement), for the "internal-" check below
        run = subprocess.run(words, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        lines = run.stdout.decode("utf-8", "replace").split("\n")
        lines = [l for l in lines if l]
        return label, run.returncode, lines, ivk

    jobs = [("%s | its own files" % w.name, w, w.key, w.entry, w.reply, w.proof, w.id) for w in [true] + sisters]
    for s in sisters:
        for judge, other in ((true, s), (s, true)):
            for what, key, entry, reply, proof in (
                ("proof", judge.key, judge.entry, other.reply, other.proof),
                ("key", other.key, judge.entry, judge.reply, judge.proof),
                ("entry", judge.key, other.entry, judge.reply, judge.proof),
                ("key-entry", other.key, other.entry, judge.reply, judge.proof),
                ("all", other.key, other.entry, other.reply, other.proof),
            ):
                for whose, named in (("the judge's ID", judge.id), ("the other's ID", other.id), ("no ID", "")):
                    jobs.append(("%s judges %s | %s | %s%s" % (judge.name, other.name, what, whose, " | control" if s.control else ""), judge, key, entry, reply, proof, named))
    with ThreadPoolExecutor(max_workers=side) as pool:
        done = list(pool.map(lambda j: call(*j), jobs))

    bad, accepted, controls, tally = 0, 0, 0, {}
    for label, status, lines, ivk in done:
        last = lines[-1] if lines else "(no line)"
        if label.endswith("its own files"):
            ok = status == honest
            print("%s: status %d: %s%s" % (label, status, last[:100], "" if ok else "   NOT THE HONEST STATUS"))
        else:
            part = label.split(" | ")
            if len(part) == 4:  # a call on the control: its label ends with " | control"
                ok = status == honest
                controls += ok
                if not ok:
                    print("%s: status %d: %s   THE CONTROL IS NOT TAKEN" % (label, status, last[:160]))
                    bad += 1
                else:
                    shutil.rmtree(ivk)
                continue
            ok = status == 1 and last in FORETOLD[(part[1], part[2])].split(" or ") and not any("internal-" in l for l in lines)
            tally.setdefault((part[1], part[2], last[:60]), []).append(part[0])
            if not ok:
                passed = status in PASSES or status == honest
                accepted += passed
                print("%s: status %d: %s   %s" % (label, status, last[:160], "WAS NOT REFUSED; its key and entry are kept in %s; request, reply and proof are the named worlds' own" % ivk if passed else "NOT AS FORETOLD"))
        if ok:
            shutil.rmtree(ivk)  # a folder of two copied files, in this run's work folder
        else:
            bad += 1
    for (what, whose, names), who in sorted(tally.items()):
        print("%-9s | %-14s | %3d calls | %s" % (what, whose, len(who), names))
    print("replies that differ from the true world's: %s" % (", ".join(s.name for s in sisters if not same(s.reply, true.reply)) or "none"))
    print("the control's proof differs from the true world's: %s" % ("yes" if not all(same(s.proof, true.proof) for s in sisters if s.control) else "NO"))
    print("keys with the true world's first 72 bytes (head and root): %s" % (", ".join(s.name for s in sisters if not s.control and open(s.key, "rb").read(72) == open(true.key, "rb").read(72)) or "none"))
    print("the control: %d of 30 calls end with the honest status" % controls)
    print("%d worlds, %d calls on files made under another configuration, %d calls in all not as foretold" % (1 + len(sisters), len(done) - 1 - len(sisters) - 30, bad))
    print("other_parameters_test: " + ("ok" if not bad else "FAILED"))
    sys.exit(3 if accepted else 1 if bad else 0)


main()
