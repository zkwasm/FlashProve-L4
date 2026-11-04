#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Checks that cp_verify, the verifier, forms the commitment ID that docs/format/FORMAT.md defines.

usage: commitment_id_test.py <cp_verify> <folder> [<folder> ..]
Each folder holds config/{policy.txt,layout.txt,KL.bin} and ivk/key.bin, as the folders under
examples/ do. The test writes no file.

For each folder it forms the ID with docs/format/format_rules.py (rules R4.8 to R4.11), from the
three configuration files and the root and the stored answers at the head of key.bin. Then:
1. `cp_verify -plan id` must print that ID.
2. `cp_verify -key .. -id <that ID>` must pass (status 3, "PART PASSED ..").
3. `cp_verify -key .. -id <the shorter ID>` must end with status 1 and the last line
   "NO: model-statement". The shorter ID is the hash of the same input without the seeds' list and
   without the closing word 0 (docs/format/10-proof-file.md, under R14.3, names this case).
Not tested: cp_commit, which forms the ID with the same routine of blocks/frame/heads.h.
Exit status 0: all as expected. 3: the shorter ID WAS NOT REFUSED. 1: something else is not as
expected; each difference has a line. 2: wrong usage."""
import os, subprocess, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "format"))
import format_rules as R


def ids(folder):
    """The folder's commitment ID, and the shorter ID of the same files."""
    rd = lambda n: open(os.path.join(folder, n), "rb").read()  # noqa: E731
    policy, key = rd("config/policy.txt"), rd("ivk/key.bin")
    line = {}
    for x in (y.split(" ") for y in policy.decode("ascii").split("\n") if y):
        line.setdefault(x[0], x[1:])
    rows, length, message, tail = (int(x) for x in line["table"][:4])
    answers = int(line["answers"][0])
    conf = R.configuration_digest([policy, rd("config/layout.txt"), rd("config/KL.bin")])
    name = R.table_name(key[8:40], conf, length, message, tail, rows, answers, int(line["salt"][0]))
    seeds = [([R.ext(x) for x in z], R.ext(beta)) for z, beta in R.record_seeds(name, (message - 1).bit_length(), answers, 2)]
    stored = key[40 : 40 + 24 * (answers << rows)]
    shorter = R.H(179, 0, 0, None, conf + R.string(name) + R.u64(len(stored) // 24) + stored)
    return R.published_id(conf, name, seeds, [stored[i : i + 24] for i in range(0, len(stored), 24)]).hex(), shorter.hex()


def main():
    if len(sys.argv) < 3:
        print(__doc__.split("\n")[2])
        return 2
    verifier, wrong, accepted = sys.argv[1], 0, 0
    for folder in sys.argv[2:]:
        mine, shorter = ids(folder)
        config, key = os.path.join(folder, "config"), os.path.join(folder, "ivk", "key.bin")
        call = lambda *words: subprocess.run([verifier, *words], capture_output=True, text=True, check=False)  # noqa: E731
        r = call("-plan", "id", config, key)
        said = [l[15:] for l in r.stdout.splitlines() if l.startswith("commitment ID: ")]
        if r.returncode != 4 or said != [mine]:
            wrong += 1
            print("%s: cp_verify -plan id: status %d, %s; the format's rules give %s" % (folder, r.returncode, said, mine))
        r = call("-key", config, key, "-id", mine)
        if r.returncode != 3 or not r.stdout.splitlines()[-1:] or not r.stdout.splitlines()[-1].startswith("PART PASSED"):
            wrong += 1
            print("%s: cp_verify -key under the ID: status %d, last line %s" % (folder, r.returncode, r.stdout.splitlines()[-1:]))
        r = call("-key", config, key, "-id", shorter)
        if r.returncode in (0, 3, 14):
            accepted += 1
            print("%s: WAS NOT REFUSED: the shorter ID: status %d, last line %s" % (folder, r.returncode, r.stdout.splitlines()[-1:]))
        elif r.returncode != 1 or r.stdout.splitlines()[-1:] != ["NO: model-statement"]:
            wrong += 1
            print("%s: cp_verify -key under the shorter ID: status %d, last line %s" % (folder, r.returncode, r.stdout.splitlines()[-1:]))
        print("%s: the ID %s" % (folder, mine))
    print("commitment_id_test.py: %s" % ("all as expected" if not wrong + accepted else "%d not as expected" % (wrong + accepted)))
    return 3 if accepted else 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
