#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Runs the GPU route of docs/quickstart.md as printed and checks that it ends as the page says.

usage: quick_start_test.py --gpu <new directory>

The page's section "3. GPU: a small model" has three fenced blocks of commands. The two that name a
shell script are the route, five lines in all: one block builds the verifier and the two GPU
programs, the other runs the example. So the test needs one GPU and the CUDA compiler. The block
that names no script only inspects the machine. It is not run, because the route itself fails on a
machine that lacks what that block looks for.

The test runs the route in a copy of this repository that leaves out the build directories, so
nearly all of the time goes to the build. It runs the lines under "bash -e", which stops at the
first line that fails, as the page tells a reader to do. Every line is run as printed, with two
exceptions:
- The line that makes a temporary directory and sets WORK, the example's new work directory, is
  replaced: the test sets WORK to a path below <new directory>, so that it knows what to remove.
- The line "unset OUT BUILD GPU_RUN" is left out. The test takes OUT and BUILD out of the
  environment itself. It keeps GPU_RUN, a command prefix for the programs that use the GPU
  (prover/examples/run_target_example.sh), which a test runner may set.

Before the run, the test checks that the three scripts which the route names exist and are
executable. After it, the test checks that the route printed the commitment ID once and ended as the
section promises in this sentence, with the three lines as the last three of standard output:
"A successful run ends with the lines `status <n>`, `<line on proof of work>` and `<line>`, and exit code <n>."

So the test fails when the programs and the page disagree. It does not show that what the page says
about the meaning of that ending is right.

<new directory> must not exist yet, and must lie below directories that only the caller or root can
change (others_can_change(), below); otherwise nothing is run. If the test passes, it removes the
copy and the example's work directory. It never removes anything else: <new directory> stays, with
the route's output in route.txt. Missing parent directories are created and stay. Exit code: 0 if
all is as the page says; 1 otherwise; 2 if the test could not start or the page does not have the
expected form.
"""

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = "docs/quickstart.md"
SECTION = "3. GPU: a small model"
NEW_WORK = 'W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"'  # the line that sets WORK
CLEARS = "unset OUT BUILD GPU_RUN"
# The promise, with any white space between its words: the page's lines are wrapped.
ENDING = r"A successful run ends with the lines `status ([0-9]+)`, `(proof of work: [^`]+)` and `([^`]+)`, and exit code ([0-9]+)\.".replace(" ", r"\s+")
tally = {True: 0, False: 0}


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


def say(what, good, seen):
    tally[bool(good)] += 1
    print(f"{what}: {seen}, {'as expected' if good else 'NOT as expected'}", flush=True)


def stop(why):
    print("quick_start_test.py: " + why, file=sys.stderr)
    sys.exit(2)


def route_of(page):
    """Returns the route's lines, the scripts that they name, the promised exit code and the
    promised last three lines."""
    section = re.search(r"^## " + re.escape(SECTION) + r"\n(.*?)^## ", page, re.M | re.S)
    if not section:
        stop(f"{PAGE} has no section {SECTION!r} that another section follows")
    fenced = re.findall(r"^```(?:sh)?\n(.*?)^```$", section[1], re.M | re.S)
    blocks = [b for b in fenced if re.search(r"[\w./-]+\.sh\b", b)]
    if len(re.findall(r"^```", section[1], re.M)) != 6 or len(fenced) != 3 or len(blocks) != 2:
        stop(f"the section {SECTION!r} of {PAGE} has {len(fenced)} blocks of commands, {len(blocks)} of which name a shell script, not 3 and 2")
    lines = "".join(blocks).splitlines()
    if len(lines) != 5:
        stop(f"the route in {PAGE} has {len(lines)} lines, not 5")
    for line in (NEW_WORK, CLEARS):
        if lines.count(line) != 1:
            stop(f"the route in {PAGE} has the line '{line}' {lines.count(line)} times, not once")
    uses = [use for x in lines if x != NEW_WORK for use in re.findall(r'"?\$\{?W(?:ORK)?\b:?\??', x)]
    if uses != ['"${WORK:?'] or lines[3] != NEW_WORK or not re.match(r'prover/examples/[\w.-]+\.sh .*"\$\{WORK:\?', lines[4]):
        stop(f"the route in {PAGE} must use the work directory once, as \"${{WORK:?...}}\", in its last line, which runs the example's script right after the line "
             f"that sets WORK, and W nowhere else; it has: {' '.join(uses) or 'no use'}")
    due = re.findall(ENDING, section[1])
    if len(due) != 1 or due[0][0] != due[0][3] or not due[0][2].startswith("YES") or re.fullmatch(r"proof\s+of\s+work:\s*\u2026", due[0][1]):
        stop(f"the section {SECTION!r} of {PAGE} must promise one ending, in the form 'A successful run ends with the lines `status N`, `proof of work: ...` "
             f"and `YES...`, and exit code N.', with the same N twice and more than an ellipsis in the second; found: {due!a}")
    return lines, sorted(set(re.findall(r"[\w./-]+\.sh\b", "\n".join(lines)))), int(due[0][0]), [f"status {due[0][0]}", re.sub(r"\s+", " ", due[0][1]), re.sub(r"\s+", " ", due[0][2])]


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--gpu" or not sys.argv[2] or os.path.lexists(os.path.abspath(sys.argv[2])):
        print("usage: quick_start_test.py --gpu <a directory that does not exist yet>", file=sys.stderr)
        return 2
    work = os.path.abspath(sys.argv[2])
    if not os.path.isfile(os.path.join(TREE, PAGE)):
        stop(f"{PAGE} is not in this tree")
    with open(os.path.join(TREE, PAGE), encoding="utf-8") as f:
        lines, named, due_status, due_lines = route_of(f.read())
    absent = [n for n in named if not os.access(os.path.join(TREE, n), os.X_OK)]
    say("the-page-names-files-that-are-there", len(named) == 3 and not absent,
        f"{len(named)} scripts in the route's commands (3 are expected): {' '.join(named)}; not there or not executable: {' '.join(absent) or 'none'}")
    above = os.path.dirname(work)
    while not os.path.lexists(above):  # the deepest directory above the work directory that exists
        above = os.path.dirname(above)
    found = others_can_change(above)  # before anything is made
    if found:
        stop(refusal("work directory", work, *found))
    mask = os.umask(0o077)
    try:
        os.umask(mask | 0o022)  # neither group nor others get write permission on the parents that are created
        os.makedirs(os.path.dirname(work), exist_ok=True)
    finally:
        os.umask(mask)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    found = others_can_change(work)  # once more, before anything is written: a directory that was missing above may have been made by another account
    if found:
        stop(refusal("work directory", work, *found))  # what this run has made so far stays: empty directories
    work = os.path.realpath(work)
    commands = "\n".join("WORK=" + shlex.quote(work + "/example") if x == NEW_WORK else x for x in lines if x != CLEARS)
    copy = work + "/tree"
    # Left out of the route's environment: names that begin with H_ or MB_, as some test settings
    # do; OUT and BUILD, by which the scripts take other directories; W and WORK, which the page's
    # own line sets.
    env = {k: v for k, v in os.environ.items() if not re.match(r"H_|MB_|(OUT|BUILD|W|WORK)$", k)}
    shutil.copytree(TREE, copy, symlinks=True,
                    ignore=lambda folder, names: [n for n in names if (folder == TREE and n in ("bin", "out", ".git"))
                                                  or n in ("build", "__pycache__")
                                                  # Leaves out `work` itself if it is inside TREE.
                                                  or (n == os.path.basename(work) and os.path.samefile(os.path.join(folder, n), work))])
    done = subprocess.run(["bash", "-e", "-c", commands], cwd=copy, env=env, capture_output=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL)
    with open(f"{work}/route.txt", "w", encoding="utf-8") as f:
        f.write(done.stdout + "\n--- standard error\n" + done.stderr)
    status, said = done.returncode, done.stdout.rstrip("\n").split("\n")
    # The example's script indents its status line, so blanks around that line are ignored; the
    # other two are compared exactly. If the promised second line ends with an ellipsis (U+2026),
    # the printed line need only begin with the text before it.
    last_lines = [x.strip() if i == 0 else x for i, x in enumerate(said[-3:])]
    compared = list(last_lines)
    if due_lines[1].endswith("\u2026") and len(compared) == 3 and compared[1].startswith(due_lines[1][:-1]):
        compared[1] = due_lines[1]
    ids = [x for x in said if re.fullmatch(r" *the commitment's ID, to be published: [0-9a-f]{64}", x)]
    say("the-route-ends-as-the-page-promises",
        status == due_status and len(ids) == 1 and compared == due_lines,
        f"status {status} ({due_status} is promised); lines with the commitment's ID: {len(ids)}; "
        f"the last three lines {last_lines!a} ({due_lines!a} are promised)")
    if not tally[False]:
        shutil.rmtree(copy)
        shutil.rmtree(work + "/example")
    print(f"quick_start_test.py: {tally[True]} as expected, {tally[False]} otherwise; the logs are in {work}")
    return 1 if tally[False] else 0


if __name__ == "__main__":
    sys.exit(main())
