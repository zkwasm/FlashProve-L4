#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs the CPU tests of the library blocks/: honest runs and damaged proofs.

usage: run_new.py <tree> <new work folder> [--jobs n] [--settings f2h1] [--only stage[,stage..]] [--bits n] [--stride n] [--tests list]
       [--whole-tree] [--doors table] [--channels list] [--left-out list] [--teeth table] [--known-limits folder]
<tree> contains blocks/ and tests/new/ (the stage `second` also reads docs/format/).
What the stages build goes under the work folder, and run_new.py sets TMPDIR to <work folder>/tmp
for every program that a stage runs, compilers too. It passes on the rest of your environment: with
DEPENDENCIES_OUTPUT or SUNPRO_DEPENDENCIES set, g++ also appends to the file that the variable names,
wherever that is, so unset both before you call this script by hand. Nothing in the tree changes.
The work folder must be outside the tree and must not exist yet, and its path must not contain
"..". This call creates it, so that nothing from an earlier call is mistaken for this call's.
Its parent directory must exist. That directory, and every directory and symbolic link on the way
to it, must belong to you or root, and none of the directories may be writable by its group or by
others unless it has the sticky bit, as /tmp has: an account that can change one of these could
put another directory in the work folder's place.
Needed: g++ (C++17), make, python3; for the stage `second` also bash, GNU xargs and GNU coreutils
(sha256sum, realpath, stat, readlink). No stage needs a GPU.
--tests <list>    The stage `tests` runs only the test sources that this file lists, one a line, as
                  paths from blocks/ ("#" starts a comment). blocks/tests/TESTS is such a list.
Without it, every test source runs. Paths are relative to the current directory, not to <tree>.
Two tables, two lists and a folder, which this repository lacks, are named by the caller; a stage
reads one only when it is given (--help says what each is):
--doors, --channels, --left-out  a table and two lists that the stage `table` also checks.
--teeth <table>   the negative controls of the stage `teeth`.
--known-limits <folder>  the programs that the stage `fences` runs as known limits.
--whole-tree      Asks for them: then a
                  missing --doors or --channels is a fault of the stage `table`, a missing
                  --known-limits or blocks/tests/must_not_build/ one of `fences` and a missing
                  --teeth one of `teeth`. Leaving out --left-out is no fault in itself.
In a call without --whole-tree, the stage `table` reports two things in its line instead of failing
on them: names in fence.h that no source holds; and, unless --left-out is given, test sources that
blocks/tests/TESTS does not list.

STAGES, in this order. Each prints one line "stage | passed or FAILED | what was seen | seconds"
(a skipped stage prints why instead) and, after FAILED, its first 60 faults, one a line:
  table    Compares lists about the tests with the sources; builds nothing.
           A function that only tests may call, according to a table of such functions, may be
           named in blocks/, prover/ and verifier/, comments apart, only in the file that defines
           it, in the files that its line allows and below directories tests/. Each test source
           must be listed exactly once, in blocks/tests/TESTS (the list for --tests) or in a list of
           the sources left out (--left-out). The stage also checks fence.h's list of switches and
           a list of the tests that construct a verifier's channel themselves against the sources.
           BLK_CHEAT_<NAME> is the form of name of the compile-time switches for malicious provers
           ("cheat" in messages), and BLK_CHEAT_BUILD the mark that a build with one must bear.
           These sources have none, and fence.h lists none, so a whole name of that form in a source
           of blocks/, prover/ or verifier/ (fence.h apart), even in a comment, is a fault. No
           Makefile, CMakeLists.txt, *.sh or *.mk file in blocks/ may name one in a compiler's
           option -D; one outside blocks/ may do so only in a command that also sets
           BLK_CHEAT_BUILD.
           A table or list that the call does not give is skipped, and the stage's line names it;
           fence.h and the list of sources left out are the exceptions, skipped without a word.
           With --whole-tree, each input that the line would name as not given is a fault instead.
  fences   Checks blocks/fence.h, which is to keep tests' switches out of production builds
           (SERVING_BUILD), by compiling at the first setting (a setting is a field and a hash:
           f2h1 gives BLK_FIELD=2, BLK_HASH=1; blocks/ compiles with no other): SERVING_BUILD with
           a test's switch must not compile. A program built with a test's switch must contain the
           file's marker string, which begins BLK-NOT-FOR-SERVING; one built without one must not.
           Then blocks/tests/must_not_build/, if present: the compiler must reject ("refuse" in
           names and messages) each source with the message that the source names, and accept it
           with -DCONTROL.
           Then the folder of known limits, if given (--known-limits): each source is built and run
           at every setting and must exit with status 0 and print the words that it names. A pass
           there is no rejection: the library's functions, driven by a test harness, accept a false
           claim, as expected, so a limit that the tree documents stands.
  tests    make test: the unit tests and honest runs (*.cpp in blocks/<directory>/tests/,
           blocks/tests/ and blocks/tests/hiding/), at every setting; with --tests, only those
           listed.
  teeth    Negative controls, listed in a table of their own, if given (--teeth). A row replaces
           one text in the library, or a few, in a scratch copy of blocks/ under the work folder;
           the honest test that it names must then fail on the copy, with exit status 1, at every
           setting. A row that names switches asks for builds that these sources cannot give: it is
           counted and not played; a table without a row that can be played is a fault. In a call
           with neither --teeth nor --whole-tree the stage is skipped, which the runner reports;
           with --whole-tree and no --teeth, that is a fault.
  toy      The complete runs of blocks/toy/ (*_run.cpp) at the target of 128 bits a round (by the
           library's own error accounting; in rounds that draw positions, 20 of the 128 are bits of
           proof of work), at every setting: the verdict must be ACCEPT.
  bytes    Damaged copies (blocks/tests/damage.h), at the first setting: each must be rejected. On
           complete toy proofs at a target of 32 bits a round, every_byte_test flips --bits of each
           byte's bits (default 8), writes a word at or above the field's modulus at every byte,
           tries every shorter length and appends bytes. Then check_test, check_key_test and
           check_entry_test (blocks/frame/tests/) do the same, with one bit flipped, at every
           --stride-th byte (default 7; 1 takes some 40 minutes on 48 cores) of proof, key and leaf
           files (`entry` in names) of 0.1 to 0.7 MB each. Each of the three has files at the
           128-bit target.
           (`bytes` runs programs built by the stage `tests`: --only must name `tests` too, and
           --tests, which may leave these programs out, cannot be given.)
  second   tests/new/second/run_honest.sh generates toy files: honest files, altered copies and two
           leaf files ("bent entries" in its output) made by copying or swapping bytes within an
           honest one; it makes no false proof. Each file is checked by the C++ verifiers and by the
           second verifiers, in Python, in tests/new/second/; the files' SHA-256 digests are
           compared with those stored there. The script builds at f2h1, whatever --settings says.
           Without the script the stage is skipped, which the runner reports.

All weights and vectors are synthetic.

An unexpected accept is a line with the word ACCEPTED or the words WAS NOT REFUSED (from a program
that failed: any form of "accept"). The runner goes on, deletes nothing, repeats these lines at its
end after the word LOOK, and exits with status 3. Keep the work folder and its files as they are:
they show what was accepted.
Exit status 0: every stage that ran passed; the last line names them, and the line before it names
any that did not run. 1: a stage failed otherwise. 2: could not run, or no stage ran.
"""
import argparse
import collections
import glob
import itertools
import os
import re
import stat
import subprocess
import sys
import time

# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")

sys.dont_write_bytecode = True
STAGES = ["table", "fences", "tests", "teeth", "toy", "bytes", "second"]
ALARM = re.compile(r"ACCEPTED|WAS NOT REFUSED")
# The output of a program that failed (exit status not 0) is searched for any form of "accept":
# failing tests have their own wordings ("accepted with byte 7 altered", "A WEAKENED PLAN HOLDS").
# A program that exits 0 prints "accepted" for honest proofs and expected controls, so its output is
# searched with the narrower ALARM.
ALARM_OF_A_FAILURE = re.compile(r"(?i)accept|NOT REFUSED|WEAKENED PLAN HOLDS")
LOOKS = []  # alarm lines in full, in the order found; what `told` returns of a fault is shortened


def told(head, said, failed):
    """Returns one fault line, shortened; alarm lines of `said` are added to LOOKS in full."""
    net = ALARM_OF_A_FAILURE if failed else ALARM
    LOOKS.extend("%s: %s" % (head, l) for l in said if net.search(l))
    text = " | ".join(said)
    return "%s: %s" % (head, text if len(text) <= 400 else text[:400] + " ..[+%d characters: the stage's log]" % (len(text) - 400))


# Runs the command `words` and appends it, its output (stderr included) and its status to the file
# `log`. Returns (status, output).
def sh(words, log, cwd=None, env=None):
    r = subprocess.run(words, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False, cwd=cwd, env=dict(os.environ, **(env or {})))
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ %s\n%s(status %d)\n" % (" ".join(words), r.stdout, r.returncode))
    return r.returncode, r.stdout


# Each stage is a function stage_<name>(o) that returns (fault lines, a one-line summary); None in
# place of the list means that the stage had nothing to run. `o`: the parsed arguments, with
# o.blocks = <tree>/blocks, o.out = <work folder>/blocks-out (make's OUT), o.settings as a list.
def stage_table(o):
    faults = []
    absent = []  # tables and lists not given, and blocks/tests/TESTS if missing; the stage names them
    notes = []  # what a copy without the further inputs lacks, said in the stage's line instead of counted as a fault
    # fence.h lists the switches of tests: it guards only those listed. It lists no BLK_CHEAT_<NAME>, so any such name in a source is a fault.
    fence = os.path.join(o.blocks, "fence.h")
    if os.path.exists(fence):
        with open(fence, encoding="utf-8") as f:
            fenced = set(re.findall(r"defined\((BLK_[A-Z0-9_]+)\)", f.read()))
        TEST_SWITCH = r"\bBLK_(?:TEST_[A-Z0-9_]+|[A-Z0-9_]+_FOR_A_TEST|[A-Z0-9_]+_FOR_TESTS_ONLY|[A-Z0-9_]+_HIDE_NOTHING)\b"
        met, words = set(), set()
        for top in (o.blocks, os.path.join(o.tree, "prover"), os.path.join(o.tree, "verifier")):  # where programs are compiled with the headers of blocks/
            for path in glob.glob(os.path.join(top, "**", "*"), recursive=True):
                if os.path.isfile(path) and path.endswith((".h", ".cuh", ".cpp", ".cu", ".inc")) and path != fence:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    met |= {"BLK_CHEAT_" + x for x in re.findall(r"\bBLK_CHEAT_([A-Z0-9_]+[A-Z0-9])\b", text)} | set(re.findall(TEST_SWITCH, text))
                    words |= set(re.findall(r"\bBLK_[A-Z0-9_]+\b", text))
        met -= {"BLK_TEST_JOBS", "BLK_TEST_DAMAGE"}  # two variables of the environment
        faults += ["%s stands in a source and not in fence.h: it is not fenced" % x for x in sorted(met - fenced)]
        if all(os.path.isdir(os.path.join(o.tree, "prover", x)) for x in ("engine", "hash_commitment")):  # the list also has the GPU side's switches: not checked in a tree without these directories
            unsourced = sorted(fenced - words - {"BLK_NOT_FOR_SERVING"})
            if o.whole_tree:  # with --whole-tree every name of fence.h must stand in a source
                faults += ["fence.h lists %s, which stands in no source" % x for x in unsourced]
            elif unsourced:  # a copy may lack the sources of some names
                notes.append("fence.h lists %d name(s) that stand in no source here: %s" % (len(unsourced), ", ".join(unsourced)))
    # Tests should run verifiers through run_verifier() (blocks/chain/closed.h), which passes the
    # verifier function its channel and handles to the inputs written into the hash chain's first
    # query, not the test's inputs themselves. A file under tests/ that constructs a verifier's
    # channel (`VerifierRounds`) itself must be listed, with a reason, in the list `channels`, if
    # given (--channels).
    kept, channels = {}, o.channels
    if channels:
        with open(channels, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    kept[line.split("\t")[0]] = line.rstrip("\n").split("\t")[1:]
    makes = set()
    for path in glob.glob(os.path.join(o.blocks, "**", "tests", "**", "*"), recursive=True):
        if os.path.isfile(path) and path.endswith((".h", ".cpp")):
            with open(path, encoding="utf-8") as f:
                if re.search(r"\bVerifierRounds\s+\w+\s*[({]|\bVerifierRounds\s*[({]|make_unique<\s*VerifierRounds\b|new\s+VerifierRounds\b", f.read()):
                    makes.add(os.path.relpath(path, o.blocks))
    if not channels:
        absent.append("the list of tests that make a verifier's channel themselves (--channels)")
    else:
        faults += ["%s makes a verifier's channel itself and is not listed in %s" % (p, channels) for p in sorted(makes - set(kept))]
    faults += ["%s lists %s, which makes no verifier's channel: take the line out" % (channels, p) for p in sorted(set(kept) - makes)]
    faults += ["%s: %s has no reason" % (channels, p) for p, why in sorted(kept.items()) if not why or not why[0].strip()]
    # blocks/tests/TESTS is the list for --tests. A second list, if given (--left-out; this
    # repository has none), has the other test sources, each followed by two spaces and the reason.
    # Each source that test_sources() returns must be in exactly one of the two, once, so that a new
    # test is placed deliberately.
    listing = os.path.join(o.blocks, "tests", "TESTS")
    if os.path.exists(listing):
        named = {}
        for name, path in (("tests/TESTS", listing), (o.left_out, o.left_out)):
            if path and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    lines = [l.rstrip("\n") for l in f if l.split("#")[0].strip()]
                named[name] = [l.split("#")[0].split("  ")[0].strip() if name != "tests/TESTS" else l.split("#")[0].strip() for l in lines]
                if name != "tests/TESTS":
                    faults += ["%s: the line «%s» gives no reason behind two blanks" % (path, l) for l in lines if len(l.split("  ", 1)) < 2 or not l.split("  ", 1)[1].strip()]
        on, out = set(named.get("tests/TESTS", [])), set(named.get(o.left_out, []))
        sources = set(test_sources(o.blocks))
        unlisted = sorted(sources - on - out)
        if o.left_out or o.whole_tree:  # with the list of sources left out, or with --whole-tree, each source is on one list
            faults += ["the test source %s stands neither in tests/TESTS nor in the list of sources left out (--left-out)" % x for x in unlisted]
        elif unlisted:
            notes.append("%d test source(s) in neither tests/TESTS nor a list given with --left-out: %s" % (len(unlisted), ", ".join(unlisted)))
        faults += ["%s stands in both tests/TESTS and the list of sources left out (--left-out)" % x for x in sorted(on & out)]
        faults += ["%s names %s, which is not a test source of the Makefile's list" % (n, x) for n, xs in sorted(named.items()) for x in sorted(set(xs) - sources)]
        faults += ["%s names %s twice" % (n, x) for n, xs in sorted(named.items()) for x, k in sorted(collections.Counter(xs).items()) if k > 1]
    else:
        absent.append("blocks/tests/TESTS")
    # Functions that bypass a rule of the library, for tests only (`doors` in names), listed in a
    # table, if given (--doors): name; defining file, from blocks/; other files that
    # may use it, from the tree's top ("-": none); why. A use is the name followed by "(", outside
    # comments.
    doors = o.doors
    if not doors:
        absent.append("the table of names that only tests may write (--doors)")
    if doors:
        with open(doors, encoding="utf-8") as f:
            kept_doors = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")][1:]  # the first is the columns' names
        faults += ["%s: a line without four full cells: %s" % (doors, "\t".join(c)[:80]) for c in kept_doors if len(c) != 4 or not all(x.strip() for x in c)]
        kept_doors = [c for c in kept_doors if len(c) == 4 and all(x.strip() for x in c)]
        written = {c[0]: [] for c in kept_doors}
        for top in (o.blocks, os.path.join(o.tree, "prover"), os.path.join(o.tree, "verifier")):
            for path in glob.glob(os.path.join(top, "**", "*"), recursive=True):
                if os.path.isfile(path) and path.endswith((".h", ".cuh", ".cpp", ".cu", ".inc")):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), f.read(), flags=re.S)
                    for n, line in enumerate(text.split("\n"), 1):
                        for name in written:
                            if re.search(r"\b%s\s*\(" % re.escape(name), line.split("//")[0]):
                                written[name].append((os.path.relpath(path, o.tree), n))
        for name, home, also, _ in kept_doors:
            home = os.path.join("blocks", home)
            free = {home} | ({x.strip() for x in also.split(",")} - {"-"})
            if not any(p == home for p, _ in written[name]):
                faults.append("%s says that %s defines %s, and it does not write the name" % (doors, home, name))
            faults += ["%s lets %s write %s, and it does not: take it out" % (doors, p, name) for p in sorted(free - {home} - {p for p, _ in written[name]})]
            faults += ["%s:%d writes %s, which only tests may write (%s)" % (p, n, name, doors) for p, n in written[name] if p not in free and "tests" not in p.split(os.sep)[:-1]]
    for path in glob.glob(os.path.join(o.tree, "**", "*"), recursive=True):  # a cheat must not be part of a build that somebody could take for the programs
        name = os.path.basename(path)
        if os.path.isfile(path) and (name in ("Makefile", "CMakeLists.txt") or name.endswith((".sh", ".mk"))):
            with open(path, encoding="utf-8", errors="replace") as f:
                commands = re.sub(r"\\\n", " ", "\n".join(l.rstrip("\n").split("#")[0] for l in f)).split("\n")  # a command continued after a backslash is one line
            for command in commands:
                if re.search(r"-D\s*BLK_CHEAT_(?!BUILD\b)[A-Z]", command) and (os.path.relpath(path, o.tree).startswith("blocks" + os.sep) or not re.search(r"-D\s*BLK_CHEAT_BUILD\b", command)):
                    faults.append("%s sets a cheat's switch by name%s" % (os.path.relpath(path, o.tree), "" if os.path.relpath(path, o.tree).startswith("blocks" + os.sep) else " in a command without the mark BLK_CHEAT_BUILD"))
                    break
    said = "%d files of tests make a verifier's channel themselves" % len(makes)
    said += "".join("; " + x for x in notes)
    if absent and o.whole_tree:  # with --whole-tree a missing table is a fault
        faults += ["%s is missing, and --whole-tree was given" % x for x in absent]
    return faults, said + ("; not in this copy of the tree, or not given, so not checked: %s" % ", ".join(absent) if absent else "")


def stage_fences(o):
    """Checks fence.h and must_not_build/ by compiling, and the known limits, if given, by building and running."""
    import concurrent.futures

    fence = os.path.join(o.blocks, "fence.h")
    if not os.path.exists(fence):
        return ["there is no blocks/fence.h"], ""
    with open(fence, encoding="utf-8") as f:
        text = f.read()
    tests = sorted(set(re.findall(r"defined\((BLK_[A-Z0-9_]+)\)", text)) - {"BLK_NOT_FOR_SERVING"})
    errors = re.findall(r'#error "([^"]+)"', text)
    string = re.search(r'"(BLK-NOT-FOR-SERVING[^"]*)"', text)
    if len(errors) != 1 or not string:
        return ["fence.h: expected exactly one line #error (found %d) and a quoted string that begins BLK-NOT-FOR-SERVING (%s)" % (len(errors), "found" if string else "not found")], ""
    if not tests:
        return ["fence.h: its list names no switch"], ""
    work = os.path.join(o.work, "fences")
    os.makedirs(work)
    source, log = os.path.join(work, "one_line.cpp"), os.path.join(work, "fences.log")
    with open(source, "w", encoding="utf-8") as f:
        f.write('#include "base.h"\nint main() { return 0; }\n')
    s = o.settings[0]
    # -fno-ipa-modref, here and below: as in CXXFLAGS of blocks/Makefile.
    base = ["nice", "g++", "-std=c++17", "-fno-ipa-modref", "-DBLK_FIELD=" + s[1], "-DBLK_HASH=" + s[3], "-I", o.blocks]

    def refused(words, why):  # "" if compiling with these switches fails with the message `why`
        st, said = sh(base + ["-fsyntax-only"] + ["-D" + w for w in words] + [source], log)
        return "" if st != 0 and why in said else "%s: %s" % (" ".join("-D" + w for w in words), "COMPILES" if st == 0 else "does not compile, but not by the fence's words")

    plays = [([x, "SERVING_BUILD"], errors[0]) for x in tests]
    valued = "BLK_GL64_SUBGROUP_LOG" in tests  # a switch with a value: 32 is the programs' own, every other value is a test's
    if valued:
        plays += [(["BLK_GL64_SUBGROUP_LOG=4", "SERVING_BUILD"], errors[0])]
    with concurrent.futures.ThreadPoolExecutor(o.jobs) as pool:
        faults = [x for x in pool.map(lambda p: refused(*p), plays) if x]
    if valued and sh(base + ["-fsyntax-only", "-DBLK_GL64_SUBGROUP_LOG=32", "-DSERVING_BUILD", source], log)[0] != 0:
        faults.append("-DBLK_GL64_SUBGROUP_LOG=32 -DSERVING_BUILD: does not compile, and should: 32 is the programs' own value")
    held = {}
    for name, words in (("honest", []), ("serving", ["SERVING_BUILD"])) + tuple(("test-" + x, [x]) for x in tests):
        out = os.path.join(work, name + ".bin")
        st, _ = sh(base + ["-O2"] + ["-D" + w for w in words] + [source, "-o", out], log)
        if st:
            faults.append("%s: does not compile, and should" % (" ".join("-D" + w for w in words) or "no switch"))
            continue
        with open(out, "rb") as f:
            held[name] = string.group(1).encode() in f.read()
    faults += ["the program built %s the fence's string" % ("without a switch HOLDS" if k in ("honest", "serving") else "as «%s» LACKS" % k) for k, v in held.items() if v != (k not in ("honest", "serving"))]
    # Sources that must not compile, except with -DCONTROL (blocks/tests/must_not_build/README.md).
    folder = os.path.join(o.blocks, "tests", "must_not_build")
    sources = sorted(x for x in os.listdir(folder) if x.endswith(".cpp")) if os.path.isdir(folder) else []
    must_not_build_here = os.path.isdir(folder)  # may be absent; if present, must contain sources
    if o.whole_tree and not must_not_build_here:  # --whole-tree asks for every folder
        faults.append("blocks/tests/must_not_build/ is missing, and --whole-tree was given")
    if must_not_build_here and not sources:
        faults.append("blocks/tests/must_not_build/ holds no source")

    def walked_round(name):  # "" if it fails with the message that it names and compiles with -DCONTROL
        path = os.path.join(folder, name)
        with open(path, encoding="utf-8") as f:
            head = re.search(r"^// MUST NOT BUILD(?: with (\S+))?, by the compiler's words «([^»]+)»\. With -DCONTROL it builds\.$", f.read(), re.M)
        if not head:
            return "must_not_build/%s: no line «// MUST NOT BUILD .., by the compiler's words «..». With -DCONTROL it builds.»" % name
        st, said = sh(base + ["-fsyntax-only"] + ([head.group(1)] if head.group(1) else []) + [path], log, env={"LC_ALL": "C"})  # the sources name the compiler's English messages
        if st == 0 or not re.search(head.group(2), said):
            return "must_not_build/%s: %s" % (name, "BUILDS" if st == 0 else "does not build, but not by the words that its head names")
        return "" if sh(base + ["-fsyntax-only", "-DCONTROL", path], log)[0] == 0 else "must_not_build/%s: its control does not build" % name

    with concurrent.futures.ThreadPoolExecutor(o.jobs) as pool:
        faults += [x for x in pool.map(walked_round, sources) if x]
    # Known limits (the folder given with --known-limits): programs in which the library's
    # functions, driven by a test harness, are expected to accept a false claim. A pass is no
    # rejection: it shows that a documented limit stands. The limit is stated in
    # docs/limits.md, "The library under the programs can be misused".
    folder = o.known_limits  # the folder comes from the caller, like the further tables
    limits = sorted(x for x in os.listdir(folder) if x.endswith(".cpp")) if folder else []
    known_limits_here = bool(folder)
    if o.whole_tree and not known_limits_here:
        faults.append("the folder of known limits (--known-limits) is missing, and --whole-tree was given")

    def still_stands(play):  # "" if it builds with the switches that it names and exits 0 with the words that it names
        name, s = play
        path, out = os.path.join(folder, name), os.path.join(work, "%s.%s.bin" % (name[:-4], s))
        with open(path, encoding="utf-8") as f:
            head = re.search(r"^// KNOWN LIMIT: built with ((?:-D\S+ )+)it ENDS 0 WITH THE WORDS «([^»]+)»\.$", f.read(), re.M)
        if not head:
            return "%s: no line «// KNOWN LIMIT: built with .. it ENDS 0 WITH THE WORDS «..».»" % path
        if sh(["nice", "g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-DBLK_FIELD=" + s[1], "-DBLK_HASH=" + s[3], "-I", o.blocks] + head.group(1).split() + [path, "-o", out], log)[0] != 0:
            return "%s at %s: does not build" % (path, s)
        st, said = sh([out], log)
        return "" if st == 0 and head.group(2) in said else "%s at %s: THE KNOWN LIMIT DOES NOT STAND AS THIS PROGRAM SAYS (status %d): the file, its README's entry and the tree's sentence must agree" % (path, s, st)

    with concurrent.futures.ThreadPoolExecutor(o.jobs) as pool:
        faults += [x for x in pool.map(still_stands, [(x, s) for x in limits for s in o.settings]) if x]
    # `MayLay` (blocks/numbers/blocks.h) lists as friends the functions that may add rounds to a
    # `Plan`. blocks/numbers/tests/plan_types_test.cpp has an array with one such function's address
    # per line. The two counts must agree: a friend that the library does not define could be
    # defined by any source.
    with open(os.path.join(o.blocks, "numbers", "blocks.h"), encoding="utf-8") as f:
        may_lay = re.search(r"^class MayLay final \{\n(.*?)^\};", f.read(), re.M | re.S)
        friends = len(re.findall(r"^  friend ", may_lay.group(1), re.M)) if may_lay else 0
    with open(os.path.join(o.blocks, "numbers", "tests", "plan_types_test.cpp"), encoding="utf-8") as f:
        addresses = len(re.findall(r"^    \(const void \*\)", f.read(), re.M))
    if friends != addresses or not friends:
        faults.append("numbers/blocks.h: class MayLay names %d friends, numbers/tests/plan_types_test.cpp takes %d addresses" % (friends, addresses))
    said = "%d switches of tests do not compile with SERVING_BUILD; the fence's string stands in %d programs built with one of them and in neither program built without; " % (
        len(tests), len([k for k, v in held.items() if v]))
    said += ("%d sources that must not build do not, each by its own words, and their controls build; " % len(sources)) if must_not_build_here else "blocks/tests/must_not_build/ is not in this copy of the tree; "
    said += ("KNOWN LIMITS THAT STAND (a false claim ACCEPTED in a harness, as foretold; no refusal): %d" % len(limits)) if known_limits_here else \
        "KNOWN LIMITS: no folder of known limits was given (--known-limits), so none was run; the limits that stand are named in blocks/README.md («Not held»)"
    return faults, said


def make(o, targets, log, extra=()):
    return sh(["nice", "make", "-k", "-j%d" % o.jobs, "-C", o.blocks, "OUT=" + o.out, "SETTINGS=" + " ".join(o.settings)] + list(extra) + list(targets), log)


def test_sources(blocks):
    """Returns the test sources, as paths from blocks/: the patterns are blocks/Makefile's TESTS."""
    return sorted(os.path.relpath(x, blocks) for x in glob.glob(os.path.join(blocks, "*", "tests", "*.cpp")) + glob.glob(os.path.join(blocks, "tests", "*.cpp")) + glob.glob(os.path.join(blocks, "tests", "hiding", "*.cpp")))


def listed_tests(o):
    """Returns (the sources that the stage `tests` runs, faults): all that test_sources() returns
    or, with --tests, that file's lines, each of which must be one of them."""
    sources = test_sources(o.blocks)
    if not o.tests:
        return sources, []
    with open(o.tests, encoding="utf-8") as f:
        named = [l.split("#")[0].strip() for l in f]
    named = [x for x in named if x]
    faults = ["%s: %s is not a test source of blocks/ (the Makefile's list)" % (o.tests, x) for x in named if x not in sources]
    faults += ["%s: %s is listed twice" % (o.tests, x) for x, n in sorted(collections.Counter(named).items()) if n > 1]
    if not named:
        faults.append("%s names no test source" % o.tests)
    return sorted(set(named) & set(sources)), faults


def said_by(target):
    """Returns (a Makefile target's output lines, whether it exited with 0); (None, False) if it did
    not run. The recipe writes the output to <target>.said and creates <target>.ran after status 0;
    make's own output is not read, because -j mixes the recipes' lines."""
    try:
        with open(target + ".said", encoding="utf-8", errors="replace") as f:
            return f.read().rstrip("\n").split("\n"), os.path.exists(target + ".ran")
    except FileNotFoundError:
        return None, False


def strays(o, known):
    """Faults for programs that make ran and `known` omits: the Makefile's list of sources and this
    file's differ."""
    faults = []
    for s in o.settings:
        for path in sorted(glob.glob(os.path.join(o.out, s, "**", "*.said"), recursive=True)):
            target = path[: -len(".said")]
            if target not in known:
                said, well = said_by(target)
                faults.append(told("%s: make ran it, and this runner's list does not have it" % target, said, not well))
    return faults


def stage_tests(o):
    sources, faults = listed_tests(o)
    if faults:
        return faults, "no test ran: the list is refused"
    status, _ = make(o, ["test"], os.path.join(o.work, "tests.log"), ["TESTS=" + " ".join(sources)] if o.tests else [])
    ran, read = {}, set()
    for s, source in itertools.product(o.settings, sources):
        name = source[: -len(".cpp")]
        target = os.path.join(o.out, s, name)
        read.add(target)
        said, well = said_by(target)
        if said is None:
            faults.append("%s %s: does not build (tests.log)" % (s, name))
            continue
        ran[s, name] = said
        if not well:
            faults.append(told("%s %s" % (s, name), said, True))
        elif any(ALARM.search(l) for l in said):
            faults.append(told("%s %s" % (s, name), said, False))
    faults += strays(o, read)
    if status and not faults:
        faults = ["make test ended with status %d" % status]
    return faults, "%d tests ran (%d sources%s, %d settings)" % (len(ran), len(sources), ", as listed in %s" % o.tests if o.tests else "", len(o.settings))


def stage_toy(o):
    status, out = make(o, ["toy"], os.path.join(o.work, "toy.log"))
    faults, ran = ([] if status == 0 else ["make toy ended with status %d" % status]), 0
    for s in o.settings:
        for path in sorted(glob.glob(os.path.join(o.out, s, "toy", "*_run.bin"))):  # the complete runs; other programs of toy/ need arguments and are only built
            st, said = sh(["nice", path, "20", "5"], os.path.join(o.work, "toy.log"))  # 20 bits of proof of work on rounds that draw positions; a fixed key for the witness and the private randomness ("coins" in blocks/toy/), under which nothing is hidden
            ran += 1
            if st != 0 or not re.search(r"(?m)^verdict: ACCEPT$", said):
                faults.append("%s %s: status %d: %s" % (s, os.path.basename(path), st, (said.strip().splitlines() or [""])[-1][:200]))
    if not ran:
        faults.append("no program of toy/ was built")
    return faults, "%d whole runs at the real target accepted" % (ran - len([f for f in faults if ": status " in f]))


def stage_bytes(o):
    s = o.settings[0]
    program = os.path.join(o.out, s, "tests", "every_byte_test.bin")
    if not os.path.isfile(program):
        return ["%s is not built (stage tests builds it)" % program], ""
    st, said = sh(["nice", program, str(o.bits), "1"], os.path.join(o.work, "bytes.log"), env={"BLK_TEST_JOBS": str(o.jobs)})
    lines = said.strip().splitlines()
    faults, seen = ([] if st == 0 else lines[:20] or ["status %d" % st]), [lines[-1] if lines else ""]
    for name in ("check_test", "check_key_test", "check_entry_test"):
        st, said = sh(["nice", os.path.join(o.out, s, "frame", "tests", name + ".bin")], os.path.join(o.work, "bytes.log"), cwd=o.blocks, env={"BLK_TEST_JOBS": str(o.jobs), "BLK_TEST_DAMAGE": "1 %d" % o.stride})
        lines = said.strip().splitlines()
        if st:
            faults.append(told("frame/tests/%s" % name, lines or ["status %d" % st], True))
        flipped = [int(x) for x in re.findall(r"refused: (\d+) with one bit flipped \(1 of 8 bits of every %d-th byte\)" % o.stride, said)]
        if not flipped:
            faults.append("frame/tests/%s says nothing of damage at every %d-th byte" % (name, o.stride))
        seen.append("%s: %d files, %d places (every %d-th byte)" % (name, len(flipped), sum(flipped), o.stride))
    return faults, "%s: %s" % (s, "; ".join(seen))


def stage_teeth(o):
    """Negative controls: one text of the library, or a few, is replaced in a scratch copy; the named honest
    test must then fail."""
    import concurrent.futures
    import shutil

    jobs, faults, left = [], [], 0
    if not o.teeth:
        if o.whole_tree:
            return ["the table of weakened copies (--teeth) is missing, and --whole-tree was given"], "no weakened copy was made"
        return None, "no table of weakened copies was given (--teeth)"
    with open(o.teeth, encoding="utf-8") as f:
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
    for n, row in enumerate(rows):
        tooth, names, olds, news, program, switches = row[:6]
        if switches.strip() != "-":  # a row that names switches asks for builds that these sources cannot give
            if len(row) > 6 and row[6].strip() == "the honest test fails":
                faults.append("%s: the seventh cell says 'the honest test fails', so the switches' cell must be '-' alone" % tooth)
            else:
                left += 1
            continue
        if not re.fullmatch(r"[A-Za-z0-9-]+", tooth):  # the cell becomes part of a directory's name
            faults.append("%s: the first cell %r may hold only letters, digits and '-'" % (o.teeth, tooth))
            continue
        if len(row) != 7 or row[6].strip() != "the honest test fails":
            faults.append("%s: the switches' cell is '-', so a seventh cell, the last, must say 'the honest test fails'" % tooth)
            continue
        parts = list(zip(names.split(" ;; "), olds.split(" ;; "), news.split(" ;; ")))
        if not len(parts) == len(names.split(" ;; ")) == len(olds.split(" ;; ")) == len(news.split(" ;; ")):
            faults.append("%s: as many files as texts and replacements" % tooth)
            continue
        # The files that a row weakens and its program must be relative paths inside blocks/, so that this
        # stage writes, compiles and runs only files in its scratch copy. The replacement texts are still code that is compiled and run:
        # a table is to be trusted like the tree's own code.
        top = os.path.realpath(o.blocks) + os.sep
        outside = [x for x in names.split(" ;; ") + [program] if os.path.isabs(x) or ".." in x.split("/") or not os.path.realpath(os.path.join(o.blocks, x)).startswith(top)]
        if outside:
            faults.append("%s: the files to weaken and the program must be relative paths inside blocks/, with no \"..\" and no link that leads out of it: %s" % (tooth, ", ".join(outside)))
            continue
        copy, fits = os.path.join(o.work, "teeth", "%d-%s" % (n, tooth), "blocks"), True
        shutil.copytree(o.blocks, copy)
        for name, old, new in parts:
            with open(os.path.join(copy, name), encoding="utf-8") as f:
                text = f.read()
            if text.count(old) != 1:
                faults.append("%s: in %s the text stands %d times: %s" % (tooth, name, text.count(old), old[:70]))
                fits = False
                break
            with open(os.path.join(copy, name), "w", encoding="utf-8") as f:
                f.write(text.replace(old, new + " /* WEAKENED */"))
        if fits:
            jobs += [(tooth, copy, program, s) for s in o.settings]
    if not jobs and not faults:
        faults.append("%s has no row that can be played: none whose switches' cell is '-'" % o.teeth)

    def play(job):
        tooth, copy, program, s = job
        out, log = os.path.join(os.path.dirname(copy), "%s.%s.bin" % (os.path.basename(program), s)), os.path.join(os.path.dirname(copy), "teeth.%s.log" % s)
        tooth = "%s, %s" % (s, tooth)
        st, said = sh(["nice", "g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-DBLK_FIELD=" + s[1], "-DBLK_HASH=" + s[3], "-I", copy, os.path.join(copy, program), "-o", out], log)
        if st:
            return "%s, %s: the weakened copy does not build" % (tooth, program)
        st, said = sh([out], log)
        seen = " | ".join(said.strip().splitlines())[-300:]
        # The honest test must fail by a check of its own (status 1; death by a signal does not count).
        return "" if st == 1 else "TEETH NOT AS FORETOLD: on the copy weakened by %s the honest test %s %s: %s" % (tooth, program, "passes: it does not notice" if st == 0 else "ends with status %d, which is no failed check" % st, seen)

    with concurrent.futures.ThreadPoolExecutor(o.jobs) as pool:
        faults += [x for x in pool.map(play, jobs) if x]
    good = len(jobs) - len([x for x in faults if "NOT AS FORETOLD" in x or "does not build" in x])
    return faults, "%d plays as foretold on %d weakened copies at %d settings: in each an honest test had to fail; %d rows of the table name switches and were not played" % (good, len(rows) - left, len(o.settings), left)


def stage_second(o):
    script = os.path.join(o.tree, "tests", "new", "second", "run_honest.sh")  # makes honest files, altered copies and two bent entries, and no false proof
    if not os.path.isfile(script):
        return None, "this tree holds no tests/new/second/run_honest.sh"
    st, said = sh(["nice", "bash", script, os.path.join(o.work, "second"), str(o.jobs)], os.path.join(o.work, "second.log"))
    lines = said.strip().splitlines() or [""]
    if st == 0 and lines[-1].startswith("ALL AS EXPECTED: "):
        return [], lines[-1]
    # The script repeats each program's last line after a prefix. Honest programs say "accepted"
    # there, so these lines get a narrower search than `told` makes.
    repeated = [l for l in lines if re.match(r"(C\+\+|Python|bytes): ", l)]
    LOOKS.extend("second: %s" % l for l in repeated if re.search(r"ACCEPTED|ACCEPTS|NOT REFUSED", l))
    return [told("status %d" % st, [l for l in lines if l not in repeated] or lines[-1:], True)], ""


def main():
    a = argparse.ArgumentParser(usage=__doc__)
    a.add_argument("tree")
    a.add_argument("work")
    a.add_argument("--jobs", type=int, default=4)
    a.add_argument("--settings", default="f2h1")
    a.add_argument("--only", default="")
    a.add_argument("--bits", type=int, default=8)
    a.add_argument("--stride", type=int, default=7)
    a.add_argument("--tests", default="")
    a.add_argument("--whole-tree", action="store_true", help="ask for the tables, lists and folders that this repository lacks (the usage says what is then a fault)")
    # Further tables and lists, which this repository lacks, are named by the caller: a stage reads one only when it is given.
    for option, what in (("--teeth", "the table of weakened copies for the stage teeth"), ("--doors", "the table of names that only tests may write"), ("--channels", "the list of tests that make a verifier's channel themselves"), ("--left-out", "the list of test sources that blocks/tests/TESTS leaves out")):
        a.add_argument(option, default="", help=what)
    a.add_argument("--known-limits", default="", help="the folder of programs whose expected end is a false claim accepted in a harness (a limit that stands), for the stage fences")
    o = a.parse_args()
    if ".." in o.work.split(os.sep):
        print('run_new.py: the work folder\'s path contains "..": give a path without it', file=sys.stderr)
        return 2
    # The work folder is checked for existence under this path; it is created and used under the
    # same path with its parent directory resolved (below). abspath() keeps two leading slashes and
    # make's `abspath` does not, so they become one here: targets' paths are built here from o.out
    # and in blocks/Makefile from $(abspath $(OUT)).
    o.tree, o.work = os.path.realpath(o.tree), os.path.abspath(o.work).replace("//", "/")
    o.blocks, o.out, o.settings = os.path.join(o.tree, "blocks"), os.path.join(o.work, "blocks-out"), o.settings.split()
    wanted = o.only.split(",") if o.only else list(STAGES)
    if not os.path.isdir(o.blocks) or set(wanted) - set(STAGES) or (os.path.realpath(o.work) + os.sep).startswith(o.tree + os.sep):
        print("run_new.py: no blocks/ in the tree, an unknown stage, or a work folder inside the tree", file=sys.stderr)
        return 2
    if "tests" not in wanted and "bytes" in wanted:
        print("run_new.py: the stage bytes runs programs that the stage tests builds: name tests with it", file=sys.stderr)
        return 2
    if o.tests and "bytes" in wanted:
        print("run_new.py: --tests may leave out programs that the stage bytes runs: name the stages with --only, without bytes", file=sys.stderr)
        return 2
    if o.tests and ("tests" not in wanted or not os.path.isfile(o.tests)):
        print("run_new.py: --tests %s: %s" % (o.tests, "no such file" if "tests" in wanted else "the stage tests is not in --only, so this option would do nothing: add the stage or leave the option out"), file=sys.stderr)
        return 2
    if o.known_limits and not os.path.isdir(o.known_limits):
        print("run_new.py: --known-limits %s: no such folder" % o.known_limits, file=sys.stderr)
        return 2
    o.known_limits = o.known_limits and os.path.realpath(o.known_limits)
    for option in ("teeth", "doors", "channels", "left_out"):
        path = getattr(o, option)
        if path and not os.path.isfile(path):
            print("run_new.py: --%s %s: no such file, or not a file" % (option.replace("_", "-"), path), file=sys.stderr)
            return 2
        setattr(o, option, path and os.path.realpath(path))
    o.tests = o.tests and os.path.realpath(o.tests)
    if os.path.lexists(o.work):
        print("run_new.py: %s exists already: name a new work folder" % o.work, file=sys.stderr)
        return 2
    # The work folder's parent directory must exist, and others_can_change() must find nothing: no
    # directory or symbolic link at or above it that another account could change or that could not
    # be examined. From here on o.work and o.out begin with the parent's resolved path.
    found = others_can_change(os.path.dirname(o.work))
    if found and found[1] == GONE:
        print("run_new.py: %s, on the way to the work folder, does not exist: make it first; this program makes only the work folder itself" % printable(found[0]), file=sys.stderr)
        return 2
    if found:
        print("run_new.py: " + refusal("work folder", o.work, *found), file=sys.stderr)
        return 2
    o.work = os.path.join(os.path.realpath(os.path.dirname(o.work)), os.path.basename(o.work))
    for what, path in (("tree", o.tree), ("work folder", o.work)):  # the stages find files by glob patterns built on these paths
        if any(c in path for c in "[]*?"):
            print("run_new.py: the %s's path %s holds one of [ ] * ?, which this program's file patterns would read as wildcards: use a path without them" % (what, path), file=sys.stderr)
            return 2
    o.out = os.path.join(o.work, "blocks-out")
    os.mkdir(o.work, 0o700)  # fails if the name is taken; closed to other users
    # Every program that a stage runs, compilers included, keeps its temporary files below the work folder, not in the
    # caller's TMPDIR, which nothing here examines.
    os.mkdir(os.path.join(o.work, "tmp"), 0o700)
    os.environ["TMPDIR"] = os.path.join(o.work, "tmp")
    status, look, idle = 0, [], {}
    for stage in STAGES:
        if stage not in wanted:
            continue
        t0 = time.time()
        faults, seen = globals()["stage_" + stage](o)
        if faults is None:  # the call gives the stage nothing to do: neither passed nor failed
            print("%s | not played | %s" % (stage, seen), flush=True)
            idle[stage] = seen
            continue
        print("%s | %s | %s | %.0f s" % (stage, "FAILED" if faults else "passed", seen, time.time() - t0), flush=True)
        for f in faults[:60]:
            print("    " + f, flush=True)
        if len(faults) > 60:
            print("    .. and %d more (the stage's log)" % (len(faults) - 60))
        look += [f for f in faults if ALARM.search(f) and not any(l.split(": ")[0] == f.split(": ")[0] for l in LOOKS)]
        status = max(status, 1 if faults else 0)
    look = LOOKS + look
    if look:
        print("LOOK: accepted, and nobody foretold it. Keep %s as it is: its files show what was accepted:" % o.work)
        for f in look:
            print("LOOK " + f)
        status = 3
    left = [s + (" (%s)" % idle[s] if s in idle else "") for s in STAGES if s not in wanted or s in idle]
    if left:
        print("not played: %s" % ", ".join(left))
    ran = [s for s in STAGES if s in wanted and s not in idle]
    if not ran:
        print("run_new.py: no stage ran")
        return 2
    print("run_new.py: %s" % {0: "passed: " + ", ".join(ran), 1: "a stage failed", 3: "SOMETHING WAS ACCEPTED THAT HAD TO BE REFUSED"}[status])
    return status


if __name__ == "__main__":
    sys.exit(main())
