#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Alter files of a stored example one at a time and show that the verifier rejects each set.

usage: try_altered_copies.py CP_VERIFY NEW_DIRECTORY [TABLE] [--example NAME]

CP_VERIFY is the path to the built verifier. NEW_DIRECTORY must not exist yet. Every directory
and link on the way to it must belong to you or root, and a directory there that other accounts
can write in must also have the sticky bit, as /tmp has. The script creates NEW_DIRECTORY, writes
the altered copies there and leaves them for you to look at. It writes nowhere else and deletes
nothing. TABLE lists the altered sets, one row each; without it the script takes the example's own
table in examples/rejected-proofs/. The first lines of altered_copies.tsv explain the columns.

--example NAME chooses the example: accepted-proof (the default) or accepted-proof-clipped-gap,
whose calls name its policy with -policy.

The script first checks the unaltered files: that call must end in YES, or for the clipped-gap
example in WITHIN LIMITS and its policy. Each later call differs from it in one file or in the
commitment ID. Exit code 0: the unaltered files passed and every altered set was rejected by the
expected check. 3: the verifier accepted an altered set. 2: a wrong call, a faulty table, or a
missing file that the script needs. 1: anything else.
"""

import argparse
import functools
import os
import pathlib
import re
import stat
import struct
import subprocess
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
TABLES = {
    "accepted-proof": "altered_copies.tsv",
    "accepted-proof-clipped-gap": "altered_copies_clipped_gap.tsv",
}
EXAMPLE = HERE.parent / "accepted-proof"  # main() puts the example that --example names here
CONFIGURATION = ["config/policy.txt", "config/KL.bin", "config/layout.txt"]
FILES = [
    "example.proof",
    "example.req",
    "example.y",
    "ivk/key.bin",
    "ivk/leaf/0.bin",
    *CONFIGURATION,
]


class FaultyTable(Exception):
    pass


def read(name):
    return (EXAMPLE / name).read_bytes()


def run(command):
    """Return a command's exit code and printed lines, or 126 and why it could not be started."""
    try:
        done = subprocess.run(
            [str(word) for word in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="backslashreplace",
            check=False,
        )
    except OSError as fault:
        return 126, ["could not be started: %s" % fault]
    return done.returncode, done.stdout.strip().split("\n")


@functools.cache
def exact_places(verifier):
    """Return {name: [places]} for the proof's fields whose place the verifier calls exact."""
    configuration = [EXAMPLE / "config/policy.txt", EXAMPLE / "config/KL.bin", "128"]
    layout = ["-layout", EXAMPLE / "config/layout.txt"]
    request = ["-prompt", EXAMPLE / "example.req", "-reply", EXAMPLE / "example.y"]
    command = [verifier, "-plan", "request", *configuration, *layout, *request, "-fields", "-q"]
    code, lines = run(command)
    if code != 4:  # the exit code of a planning call
        raise FaultyTable("the verifier could not list the proof's fields (exit code %d)" % code)
    places = {}
    for line in lines:
        found = re.fullmatch(r"field: proof at ([0-9]+) [0-9]+ [a-z]+: (.*)", line)
        if found:
            places.setdefault(found.group(2), []).append(int(found.group(1)))
    return places


def place_of_field(verifier, what):
    """Turn "NAME" or "NAME #K" into the place of that field, or of the K-th field of that name."""
    name, which = re.fullmatch(r"(.*?)(?: #([0-9]+))?", what).groups()
    places = exact_places(verifier).get(name, [])
    if not places:
        raise FaultyTable("the verifier lists no field '%s' with an exact place" % name)
    if which is None and len(places) > 1:
        raise FaultyTable("'%s' names %d fields: add which, as in '#2'" % (name, len(places)))
    if not 1 <= int(which or 1) <= len(places):
        raise FaultyTable(
            "there is no #%s among the %d of the name '%s'" % (which, len(places), name)
        )
    return places[int(which or 1) - 1]


def offset(verifier, file, place, length):
    """Turn a row's place into a number of bytes (in ID.txt: of digits) from the start."""
    kind, _, what = place.partition(" ")
    if place in ("first", "middle", "last", "end"):
        return {"first": 0, "middle": length // 2, "last": length - 1, "end": length}[place]
    if kind in ("byte", "token") and re.fullmatch("[0-9]+", what):
        return int(what) * (4 if kind == "token" else 1)
    if place == "token first-checked" and file == "example.req":
        # A request is T tokens, then r and e, as 32-bit little-endian words. Token r, counted from
        # 0, is the prompt's last, so the first checked token is r + 1.
        tokens = length // 4 - 2
        first_checked = struct.unpack("<I", read(file)[4 * tokens : 4 * tokens + 4])[0] + 1
        if first_checked >= tokens:
            raise FaultyTable("the request has no checked token")
        return 4 * first_checked
    if kind == "field" and file == "example.proof":
        return place_of_field(verifier, what)
    raise FaultyTable("'%s' is not a place in %s" % (place, file))


def altered(data, at, change):
    """Apply a row's change to a file's bytes."""
    words = change.split(" ")
    if change == "other-digit":
        raise FaultyTable("other-digit is for ID.txt only")
    try:
        if change == "flip" and at < len(data):
            return data[:at] + bytes([data[at] ^ 1]) + data[at + 1 :]
        if change == "cut" and at < len(data):
            return data[:at]
        if words[0] == "append" and len(words) == 2 and at == len(data) and words[1]:
            return data + bytes.fromhex(words[1])
        if words[0] == "put" and (len(words) == 2 or (len(words) == 4 and words[2] == "over")):
            new, there = bytes.fromhex(words[1]), data[at : at + len(words[1]) // 2]
            if len(words) == 4 and len(bytes.fromhex(words[3])) != len(new):
                raise FaultyTable("put and over must name the same number of bytes")
            if len(words) == 4 and there != bytes.fromhex(words[3]):
                raise FaultyTable(
                    "the bytes there are not %s: the row is for another file" % words[3]
                )
            if new and new == there:
                raise FaultyTable("'%s' does not change anything at byte %d" % (change, at))
            if new and len(there) == len(new):
                return data[:at] + new + data[at + len(new) :]
    except ValueError:
        pass
    raise FaultyTable("'%s' cannot be done at byte %d of %d" % (change, at, len(data)))


def cases(verifier, table):
    """Yield (description, {file name: new bytes}, commitment ID, expected last line) per row."""
    # For this demonstration only. In a real check the commitment ID comes from where the commitment
    # is published, never from files that come with a proof.
    true_id = read("ID.txt").decode().strip()
    try:
        text = table.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise FaultyTable("%s is not a text file in UTF-8" % table) from None
    for number, row in enumerate(text.splitlines(), 1):
        if not row.strip() or row.startswith("#"):
            continue
        try:
            columns = row.split("\t")
            if len(columns) != 5:
                raise FaultyTable("%d columns, not 5" % len(columns))
            description, file, place, change, expected = columns
            if not description:
                raise FaultyTable("the description is empty")
            if not expected.startswith("NO: "):
                raise FaultyTable("the last column must begin with 'NO: '")
            if file == "ID.txt" and change != "other-digit":
                raise FaultyTable("the only change in ID.txt is other-digit")
            if file == "ID.txt":
                at = offset(verifier, file, place, len(true_id))
                if not at < len(true_id):
                    raise FaultyTable("the commitment ID has no digit %d" % at)
                digit = "0" if true_id[at] != "0" else "1"
                yield description, {}, true_id[:at] + digit + true_id[at + 1 :], expected
            elif file in FILES:
                data = read(file)
                at = offset(verifier, file, place, len(data))
                yield description, {file: altered(data, at, change)}, true_id, expected
            else:
                raise FaultyTable("'%s' is not a file of the example" % file)
        except FaultyTable as fault:
            raise FaultyTable("%s, line %d: %s" % (table, number, fault)) from None


@functools.cache
def stated_policy():
    """Return the words that name the example's policy in a call, and a passing call's last line."""
    if EXAMPLE.name == "accepted-proof":
        return [], "YES"  # its policy is exact match, which a call need not name
    # For this demonstration only. In a real check you fix the limits that you accept before the
    # proof is made, and never take them from files that come with a proof.
    lines = [line.split() for line in read("config/policy.txt").decode().splitlines()]
    values = dict(line for line in lines if len(line) == 2)
    limits = tuple(values.get(name, "missing") for name in ("C", "theta", "cap", "s", "sh"))
    return (
        ["-policy", "clipped-gap,%s,%s,%s,%s,%s" % limits],
        "WITHIN LIMITS: clipped-gap C=%s theta=%s cap=%s s=%s sh=%s" % limits,
    )


def copy_bytes(source, target):
    """Copy a directory's files by their bytes alone: copies of read-only files can be written."""
    target.mkdir()
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            copy_bytes(entry, target / entry.name)
        else:
            (target / entry.name).write_bytes(entry.read_bytes())


def check(verifier, directory, changed, commitment_id):
    """Run the verifier on the example's files, after writing the changed ones into the directory.

    Return its exit code and printed lines.
    """

    def path(name):
        return directory / name if name in changed else EXAMPLE / name

    # The verifier takes config/ and ivk/ as directories, so a changed file there needs its
    # neighbours beside it.
    for part in ("config", "ivk"):
        if any(name.startswith(part + "/") for name in changed):
            copy_bytes(EXAMPLE / part, directory / part)
    for name, data in changed.items():
        (directory / name).write_bytes(data)
    parts = [
        directory / part if (directory / part).is_dir() else EXAMPLE / part
        for part in ("config", "ivk")
    ]
    files = [path("example.req"), path("example.y"), path("example.proof")]
    words = [*parts, *files, "-id", commitment_id, *stated_policy()[0], "-q"]
    return run([verifier, *words])


def both_numbers(lines):
    """Return what an accepting call printed on proof of work, as lines for the screen."""
    work = [line for line in lines if line.startswith("proof of work: ")]
    fields = dict(
        word.split("=", 1)
        for line in lines
        if line.startswith("result-v1 ")
        for word in line.split()[1:]
        if "=" in word
    )
    if len(work) != 1 or not {"least-round", "least-round-without-work"} <= set(fields):
        return ["    (the verifier's lines on proof of work could not be read)"]
    return [
        "    " + work[0],
        "    by the verifier's own count, the weakest round has %s bits with proof of work counted"
        " and the weakest without it has %s; no figure here is a security level"
        % (fields["least-round"], fields["least-round-without-work"]),
    ]


# A check that no other account can change the path to a directory. The constants and functions
# from here to cure() have the same code as in run_suite.py. A script that creates a directory
# whose path it was given, or works below one, copies this block whole and changes nothing in it.
# The block needs `import os`, `import re` and `import stat`. Its printable() stays even where the
# script has one of its own.
#
# Call others_can_change() on the existing parent of the directory to be created, or on the
# directory itself if it exists. If it returns (place, why), print
# "<script>: " + refusal(<what>, <path>, place, why) on standard error and exit with 2.
# run_suite.py prints the same message, except for a place that is missing or cannot be examined.
# If others_can_change() returns None, use the parent's resolved path from then on.
#
# run_suite.py also rejects the directory that its environment variable BUILD names if that has a
# default access control list. That check is not in this block, so nothing here returns `ACL`.

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


def wrong_call(words):
    print("try_altered_copies.py: " + words, file=sys.stderr)
    return 2


def main():
    global EXAMPLE
    usage = __doc__[__doc__.index("usage: ") + len("usage: ") :].split("\n")[0]
    parser = argparse.ArgumentParser(usage=usage, add_help=False, allow_abbrev=False)
    parser.add_argument("verifier", type=pathlib.Path, metavar="CP_VERIFY")
    parser.add_argument("work", type=pathlib.Path, metavar="NEW_DIRECTORY")
    parser.add_argument("table", type=pathlib.Path, metavar="TABLE", nargs="?")
    parser.add_argument("--example", choices=list(TABLES), default=EXAMPLE.name)
    call = parser.parse_args()  # a wrong call ends the script here, with exit code 2
    EXAMPLE = HERE.parent / call.example
    verifier, work = call.verifier.resolve(), call.work
    table = call.table or HERE / TABLES[call.example]
    if not verifier.is_file():
        return wrong_call("%s is missing: build the verifier first (verifier/build.sh)" % verifier)
    for needed in (*(EXAMPLE / name for name in [*FILES, "ID.txt"]), table):
        if not needed.is_file():
            return wrong_call("%s is missing or is not a file" % needed)
    found = others_can_change(str(work.parent))  # before CP_VERIFY is started for the first time
    if found:  # any place returned stops the script; a missing one gets the plainer message
        if found[1] == GONE:
            return wrong_call("the directory above %s is missing" % work)
        return wrong_call(refusal("directory", str(work), *found))
    work = work.parent.resolve() / work.name  # the path that was looked at, with no link left in it
    try:
        rows = sum(1 for _ in cases(verifier, table))  # a faulty table stops the script here
    except FaultyTable as fault:
        return wrong_call(str(fault))
    if not rows:
        return wrong_call("%s has no row" % table)
    try:
        work.mkdir(mode=0o700)  # closed to other accounts, so that none can put a link in the way
    except FileExistsError:
        return wrong_call("%s exists already: name a new directory" % work)
    except FileNotFoundError:
        return wrong_call("the directory above %s is missing" % work)

    print("The altered copies are written to %s" % work.resolve())
    code, lines = check(verifier, work, {}, read("ID.txt").decode().strip())
    print("%-38s exit code %-3d %s" % ("the files as they are", code, lines[-1]))
    if (code, lines[-1]) != (0, stated_policy()[1]):
        print("The unaltered files did not pass, so the altered ones show nothing.")
        return 1
    print("\n".join(both_numbers(lines)))
    faults = accepted = 0
    try:
        for number, row in enumerate(cases(verifier, table), 1):
            description, changed, commitment_id, expected = row
            directory = work / ("%02d" % number)
            directory.mkdir(mode=0o700)
            code, lines = check(verifier, directory, changed, commitment_id)
            accepted += code == 0
            last = lines[-1]
            if code == 0:
                note = "   <-- WAS ACCEPTED"
            elif code != 1 or not last.startswith("NO: "):
                note = "   <-- NOT REJECTED"
            elif last != expected:
                note = "   <-- rejected, but not by the expected check (%s)" % expected
            else:
                note = ""
            print("%-38s exit code %-3d %s%s" % (description, code, last, note))
            faults += bool(note)
        if faults:
            words = "%d altered sets: %d NOT AS EXPECTED, %d of them ACCEPTED"
            print(words % (rows, faults, accepted))
            return 3 if accepted else 1
    except Exception:
        if not accepted:
            raise
        traceback.print_exc()  # whatever went wrong later, an accepted set decides the exit code
        return 3
    print("%d altered sets: all rejected" % rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
