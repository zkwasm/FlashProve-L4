#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Gives each stored file in this folder to its checker: every false file must be refused, every honest one accepted.

usage: stored_false_test.py <folder that holds handover_files.bin and handover_proof.bin> <a new folder>

Started by the runner; by hand: below a directory of your own. (run_suite.py runs it as the row
examples.protocol_toy_false_files of SUITE.md. By hand, give as the new folder a path that does not exist yet, inside a
directory of your own; the script makes that folder and writes only in it.)

The files are key files, leaf files (`entry` in names) and proofs of the toy instances on which the library in blocks/
tests itself. They are made from fixed small numbers, so nothing in them is secret. Those in false/ were made by
builds of the library's own tools that lie on purpose, except one, an honest entry whose bytes a script rearranged;
all twelve carry `cheat` in their names; those in honest/ are controls. LIST.tsv lists them: see read_list(). The code
that lies is not in this repository, so the eleven files that such builds made cannot be made anew from it.

Build the two programs first, for example with
  make -C blocks OUT=<folder> SETTINGS=f2h1 <folder>/f2h1/frame/handover_files.bin <folder>/f2h1/frame/handover_proof.bin
and pass <folder>/f2h1/frame. Both rebuild the toy instance from their own sources and read only the
file that they check (`judge` in their commands).

The test first checks every listed file's SHA-256 and runs no checker if one is missing or
differs. It writes only below the new folder and deletes nothing. Each file gets its own folder
there, because the C++ checkers write their verdict beside the file. Each checker may run for 300
seconds.
Exit status: 0 if every file got the verdict and the words in LIST.tsv; 3 if any false file was
accepted; otherwise 1 if a listed file is missing or its sum differs, a checker gave no verdict, an
honest file was refused, or the words differ; 2 if the call or LIST.tsv is wrong.
A pass shows that these checkers refuse these files, not that they refuse every false file.
"""

import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(os.path.dirname(HERE))
SECOND = os.path.join(TOP, "tests/new/second")
PAGE = os.path.dirname(os.path.join(TOP, "docs/format/format_rules.py"))  # the folder of format_rules.py, which the Python checkers import
CHECKERS = ("key", "entry", "proof-0", "proof-1", "product", "links")
LIMIT = 300


def usage(why):
    print("stored_false_test.py: " + why, file=sys.stderr)
    print("usage: stored_false_test.py <folder of handover_files.bin and handover_proof.bin> <a new folder>", file=sys.stderr)
    sys.exit(2)


# The six cells of a line of LIST.tsv, separated by tabs:
#   file      the path, relative to this folder
#   sha256    the file's SHA-256
#   checker   key, entry: the library's check_key(), check_entry() (handover_files.bin judge-key or
#             judge-entry);
#             proof-0, proof-1: its check_proof() (handover_proof.bin judge on-the-copy), for
#             request 0 or 1 of the toy batch on the leaf table (`copy` in names);
#             product, links: the second verifier in Python (tests/new/second/toy_check.py or
#             links_check.py), which also reads <run>.parameters and <run>.configuration from
#             honest/product or honest/links (their sums are not checked)
#   must      refuse or accept
#   words     the checker's verdict as printed: the line that a C++ checker writes beside the file,
#             or the cell after "the second verifier: " in a Python checker's output
#   what      what is false in the file, or what the honest file is
def read_list():
    rows = []
    with open(os.path.join(HERE, "LIST.tsv"), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.rstrip("\n").split("\t")
            if cells[0] == "file":
                continue
            if len(cells) != 6 or cells[2] not in CHECKERS or cells[3] not in ("refuse", "accept") or not cells[4]:
                usage("LIST.tsv, line %d: six cells are needed; the checker must be one of %s, and the cell \"must\" is refuse or accept" % (n, ", ".join(CHECKERS)))
            if cells[2] in ("product", "links") and (".cheat-" not in os.path.basename(cells[0]) or not cells[0].endswith(".proof")):
                usage("LIST.tsv, line %d: a Python checker reads files named <run>.cheat-<name>.proof" % n)
            rows.append([n] + cells)
    if not rows:
        usage("LIST.tsv names no file")
    return rows


def run(command, own):
    """Runs a checker; returns (status, what it printed), or (None, why) if it did not end in time."""
    try:
        done = subprocess.run(command, cwd=own, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", timeout=LIMIT, env=dict(os.environ, TMPDIR=own))
    except subprocess.TimeoutExpired:
        return None, "it did not end within %d seconds" % LIMIT
    return done.returncode, done.stdout


def judge_cpp(bins, checker, source, own):
    """The library's checkers: status 0 means accepted, 1 refused, anything else no verdict. Returns (verdict, words, note)."""
    copy = os.path.join(own, os.path.basename(source))
    shutil.copyfile(source, copy)
    if checker in ("key", "entry"):
        command = [os.path.join(bins, "handover_files.bin"), own, "judge-" + checker, copy]
    else:
        command = [os.path.join(bins, "handover_proof.bin"), own, "judge", "on-the-copy", checker[-1], copy]
    status, printed = run(command, own)
    verdict_file = os.path.splitext(copy)[0] + ".txt"
    if status not in (0, 1) or not os.path.isfile(verdict_file):
        return None, "", "no verdict: " + (printed if status is None else "status %s; %s" % (status, printed.strip()[-160:]))
    with open(verdict_file, encoding="utf-8", errors="replace") as f:
        words = f.read().rstrip("\n")
    if (words == "ACCEPTED") != (status == 0):
        return None, words, "no verdict: status %d does not match the words" % status
    return ("accept" if status == 0 else "refuse"), words, ""


def judge_python(checker, source, own, words):
    """Runs a second verifier in its mode --cheats, which checks every <run>.cheat-*.proof in a
    folder against a list of the checks expected to fail: here one proof and the first word of
    `words` (such as D4). Returns the same as judge_cpp()."""
    name = os.path.basename(source)
    run_name, tag = name[:name.index(".cheat-")], name[name.index(".cheat-") + 7:-len(".proof")]
    base, _, place = tag.rpartition("-") if tag.rpartition("-")[2].isdigit() else (tag, "", "-")  # a name ends in -<place> where the prover cheats at one of several places
    honest, where = os.path.join(own, "honest"), os.path.join(own, "false")
    os.mkdir(honest)
    os.mkdir(where)
    for tail in ("parameters", "configuration"):
        shutil.copyfile(os.path.join(HERE, "honest", checker, "%s.%s" % (run_name, tail)), os.path.join(honest, "%s.%s" % (run_name, tail)))
    shutil.copyfile(source, os.path.join(where, name))
    # The mode also compares the C++ verifier's words, in a .txt file beside each proof, with the
    # expected check. No C++ tool checks these files here, so the mode prints NOT AS EXPECTED and
    # exits with status 1; this test ignores both.
    with open(os.path.join(where, name[:-len(".proof")] + ".txt"), "w", encoding="utf-8") as f:
        f.write("not judged by a C++ tool in this test\n")
    listed = os.path.join(own, "expected.tsv")
    with open(listed, "w", encoding="utf-8") as f:
        f.write("%s\t%s\t-\t-\trefused at %s\t-\n" % (base, place, words.split(" ")[0]))
    script = os.path.join(SECOND, "toy_check.py" if checker == "product" else "links_check.py")
    status, printed = run([sys.executable, "-B", script, PAGE, honest, run_name, "--cheats", where, listed], own)
    for line in (printed or "").splitlines():
        cells = line.split("\t")
        if cells[0] == tag:
            said = [c for c in cells if c.startswith("the second verifier: ")]
            if said:
                words = said[0][len("the second verifier: "):]
                return ("accept" if words == "ACCEPTED" else "refuse"), words, ""
    return None, "", "no verdict: " + (printed if status is None else "status %s; %s" % (status, (printed or "").strip()[-160:]))


def main():
    if len(sys.argv) != 3 or not sys.argv[1] or not sys.argv[2]:
        usage("two arguments are needed")
    bins, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    for program in ("handover_files.bin", "handover_proof.bin"):
        if not os.access(os.path.join(bins, program), os.X_OK):
            usage("%s is not a program in %s" % (program, bins))
    if os.path.lexists(work):
        usage("%s exists already: name a new folder" % work)
    rows = read_list()
    wrong = 0
    for n, name, want_sum, checker, must, words, what in rows:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path) or os.path.islink(path):
            print("%s: not there, or not a plain file" % name)
            wrong += 1
            continue
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != want_sum:
                print("%s: its SHA-256 differs from the one in LIST.tsv" % name)
                wrong += 1
    if wrong:
        print("stored_false_test.py: missing or changed files: %d; nothing was judged" % wrong)
        return 1
    os.mkdir(work, 0o700)
    accepted = other = 0
    for n, name, want_sum, checker, must, words, what in rows:
        own = os.path.join(work, str(n))
        os.mkdir(own, 0o700)
        source = os.path.join(HERE, name)
        if checker in ("product", "links"):
            verdict, said, note = judge_python(checker, source, own, words)
        else:
            verdict, said, note = judge_cpp(bins, checker, source, own)
        if verdict is None:
            print("%s: %s" % (name, note))
            other += 1
        elif verdict == "accept" and must == "refuse":
            print("%s: WAS NOT REFUSED, but it is false: %s" % (name, what))
            accepted += 1
        elif verdict != must:
            print("%s: refused, but it is an honest file: %s" % (name, said))
            other += 1
        elif said != words:
            print("%s: %s as it must be, but with other words: %r, where LIST.tsv has %r; if the checker's words changed on purpose, change LIST.tsv" % (name, "refused" if must == "refuse" else "accepted", said, words))
            other += 1
        else:
            print("%s: %s: %s" % (name, "refused" if must == "refuse" else "accepted", said))
    false_files = sum(1 for r in rows if r[4] == "refuse")
    if accepted or other:
        print("stored_false_test.py: false files accepted: %d of %d; other faults: %d" % (accepted, false_files, other))
    else:
        print("stored_false_test.py: all %d false files refused and all %d honest files accepted, each with the words in LIST.tsv" % (false_files, len(rows) - false_files))
    return 3 if accepted else 1 if other else 0


if __name__ == "__main__":
    sys.exit(main())
