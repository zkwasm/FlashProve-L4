#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests tools/confine.py, which confines a command with Linux's Landlock. On any kernel: the level
(what it confines) that it gives for six Landlock ABIs, and that it runs nothing without Landlock.
Where this kernel gives a level: harmful commands, run confined, must leave their decoys intact.

usage: confine_test.py <new folder>
Started by the runner; by hand: below a directory of your own (the new folder is made there).
The runner is run_suite.py.

  levels      confine.level() for ABI 0, 1, 2, 3, 5 and 6. 0: no level, and a reason. 1:
              `confine.FILES` (files; of signals, only those sent to every process at once) and a
              note; the Landlock rights REFER (link or rename across directories) and TRUNCATE are
              not handled. 2: REFER is handled, TRUNCATE is not. 3: `confine.FILES`; both are
              handled. 5: `confine.FILES`. 6: `confine.BOTH` (files and signals)
  a wrong call
              confine.py on a directory that does not exist: exit code 2, and the command is not run
  no Landlock confine.main() with kernel_abi() made to return 0: exit code 125, its NOT CONFINED
              line, and the command is not run
  its TMPDIR  where this kernel gives a level: confine.py makes <folder>/tmp with mode 700, or takes
              one that exists only if it is a directory of this account's own that group and others
              cannot write in. One of mode 777, and a symbolic link in its place, each end the call
              with exit code 127 and its sentence, and the command is not run (for mode 777, the
              sentence offers "chmod go-w" to an ordinary account, never to root); one of mode 700 is
              taken
  a set-up that fails half way
              confine.main() in a stand-in whose Landlock calls fail at one step: the ruleset's
              creation, the second rule's addition, or the last, restricting call. Each time:
              exit code 125, its NOT CONFINED line naming the step, and the command is not run
The other checks run commands from tools/confine_plants.txt (`plant` in identifiers). The harmful
ones run only confined, each aimed at a decoy this test made, except «as root, /proc's system places»
and «as root, pseudo-terminals», which open real files for writing and close them unwritten:
  a removal   confined to <folder>/run, tries to remove <folder>/decoy recursively, through "..":
              this fails (the command prints exit code 1) and the decoy's file remains
  an overwriting, a renaming, an emptying, a making, an empty folder's removal
              overwrite, rename or truncate the decoy's file; create a directory, a symbolic link, a
              hard link or a fifo next to it; remove the decoy's empty directory. Every step fails
              with exit code 1; after each command the decoy contains only its file, with its
              original text, and its empty directory. "an emptying" runs only from Landlock ABI 3
              on: earlier ABIs do not handle truncate()
  a home      with HOME=<folder>/home: a write to $HOME creates nothing; one to <folder>/run works
  a signal    only at the level "files and signals" (as root, followed by what root's commands get): sends SIGKILL to every process whose command
              line contains the unique name of this test's own sleeping child. The child survives.
              The command's own shell matches too and dies by SIGKILL, so the signal was sent
  its own /proc
              writes a name into its own /proc/<pid>/comm, as CUDA does to name its threads. Without
              --gpu the write fails. With --gpu the output equals that of a bash started without
              confine.py. Where this test itself runs confined without --gpu, as it does by default
              under run_suite.py, the write fails in all three, and that comparison shows nothing. As
              root where confine.py refuses --gpu (no mount namespace, and places of /proc writable),
              only the call without --gpu is made, and the other is named as not run
  what is ignored
              the confined command ignores the same signals as a bash that this test starts by its
              path without confine.py, as it starts the confined side's Python (Python ignores
              SIGPIPE and SIGXFSZ, and an exec passes that on; confine.py resets both)
  the signals' filter
              the seccomp filter that confine.py installs where Landlock cannot confine signals
              (the level `confine.FILES`, which printed lines call "the second level"). Signal 0,
              which sends nothing, to -1 (every process that the sender may signal) is rejected;
              signals to a process group that the command created, to its own group and to its
              child are allowed. Run with CONFINE_SIGNALS_FILTER=always, then without it through
              confine.main() with kernel_abi() made to return 1, 2, 3 and 5 (none above this
              kernel's ABI), where confine.py must install the filter unasked
  the option --also
              confined to <folder>/run with --also <folder>/also: a write to the second folder
              works, and one to the decoy next to them still fails
  the rule for /dev/shm
              a regular file of a name unique to this run can be made in /dev/shm and removed
              again, but not a folder (the test removes both by those names afterwards, if they
              stand). Not run where /dev/shm is not a folder. Where this test itself runs confined,
              as under run_suite.py, its own confinement already refuses the folder, so only a
              run by hand shows that half
  a device node
              the confined command makes, with Python's os.mknod, in the confined folder a character
              device 0:0 (the inert "whiteout" node), a block device 0:0 and a character device 1:3
              (a twin of /dev/null): each fails with EACCES (13), the error of Landlock's refusal,
              and no node is left; the test removes by name any node that was made. Where the kernel
              lets the node 0:0 by and Landlock refuses the other two, the test prints a line on
              that node and reports that part as not run: the comment at the check says more
  the mount namespace only for root
              the confined command's mount namespace (/proc/self/ns/mnt) is this test's own, unless
              this test runs as root with CAP_SYS_ADMIN and not already inside a row that confine.py
              confined as root (its /dev/shm is then a tmpfs named confine.py-row, read at this
              test's start): then it differs. Not run where confine.py sets up no namespace because
              nothing needs one, or because the namespace is refused and root may not plainly mount
              (CAP_SYS_ADMIN, no seccomp filter, no AppArmor or SELinux profile, and / a mount point).
              Where root may plainly mount, a refused namespace is a failure
  multiprocessing
              the confined command makes a Lock and a Queue of Python's multiprocessing and starts a
              child by "spawn": it prints "mp ok"
  as root, /proc's system places
              only as root: confined with --gpu, the command opens for writing, and closes at once
              without writing, /proc/sys/kernel/domainname, /proc/sysrq-trigger,
              /proc/irq/default_smp_affinity and one device's file below /proc/bus/pci (those that
              exist), and /proc/sys/fs/binfmt_misc/register where binfmt_misc itself is mounted (not a
              mere automount): each open fails with EROFS (30), and the mount lines at or below /proc
              and /dev/shm, read by this test outside the row, are as before. Not run where confine.py
              refuses --gpu here (but where root may plainly mount, a refusal is a failure)
  as root, pseudo-terminals
              only as root: the confined command cannot open /dev/ptmx for writing (EACCES,
              13), which as root would reach every account's terminals
  as root, another account's file in /dev/shm
              only as root, not inside a row that confine.py confined as root, and not where confine.py
              cannot set up a mount namespace and root may not plainly mount: the test makes a decoy
              /dev/shm/confine-test-theirs-<process id>-<clock reading>, hands it to uid 65534 and puts
              known bytes in it. The confined command cannot list it (ls exits 2), and what it
              appends, removes or makes there stays in its own /dev/shm: afterwards the decoy keeps its
              bytes and owner, <decoy>.mine does not exist, and /dev/shm's device and mount line are as
              before, as are the mount lines at or below /proc and /dev/shm. The test removes both
              names afterwards
  as root, the order of the calls (a stand-in)
  as root, a failed or unexpected step (stand-ins: nothing is run)
  as root, nothing to mount, or the namespace refused without --gpu (stand-ins: the command runs)
  as root, mount_setattr refused (a stand-in: each place remounted read-only)
  as root, the level (stand-ins)
  as root, pseudo-terminals (a stand-in)
              confine.main() and confine.level_here() with root faked: geteuid() returns 0, unshare,
              mount and mount_setattr are recorded and not made, /proc/self/mountinfo and the
              namespace's name come from tables that the test writes, and the Landlock calls are real.
              The command, "touch <marker>" (with --gpu unless said), must run only after unshare, then
              "/" made private, then, for each of the five /proc places that exist here, in any order,
              a bind over itself followed at once by the call that makes it read-only, then a tmpfs
              named confine.py-row on /dev/shm. Only a refused unshare may still run, and then
              --gpu only where the places are read-only already; nothing may run where any later step
              fails, refused or not, where unshare fails otherwise, where the namespace did not change,
              where a result is not as expected, or where the setup differs from the one that
              run_suite.py found at its start and passed in CONFINE_AS_ROOT. Inside a command that confine.py already confined as root, the command runs
              and nothing is mounted; where the namespace is refused, confine.py prints one line saying
              that /dev/shm is shared when started without CONFINE_AS_ROOT (by hand), and no such line
              when run_suite.py starts it; with standard error full, closed before confine.py starts, a
              pipe whose reader has exited, or a pipe that is full and unread, the command still runs
              and nothing reaches standard output. Where mount_setattr is refused
              or missing, each place is remounted read-only instead. The level names what root's
              commands get. With root faked, the command cannot open /dev/ptmx for writing; an ordinary account,
              confined, can (compared only when this test does not run as root). These show the
              order and the refusals, not that a hole is closed on a machine: «as root, /proc's system
              places», «as root, pseudo-terminals» and «as root, another account's file in /dev/shm»
              do that

Creates <new folder> and does not remove it. Prints a line for each check, then "confine_test: <n>
checks as expected, <m> otherwise" and what was not run. Exit code 0 if all are as expected; 1 if
not; 77 if none failed but some were not run (no Landlock, a Landlock that cannot confine signals,
a Landlock that does not refuse the node 0:0, no folder /dev/shm, no /dev/ptmx, or, as root, a
check was skipped because confine.py sets up no mount namespace here, refuses --gpu here or fails
to set up (where root may not plainly mount), or because this test already runs inside a command
that confine.py confined as root); when this test does not run as root, the three checks for root
alone are named as not applicable and do not keep it from 0;
2 for a wrong call, which includes a path that exists or whose absolute
form has a character other than letters, digits and / . _ -.
"""

import errno
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # no __pycache__ in the tree
import confine  # noqa: E402

CONFINE = os.path.join(HERE, "confine.py")


# Fakes root for the checks whose names begin "as root" and contain "stand-in": geteuid() returns 0; unshare, mount and
# mount_setattr are written to `record` and return 0, or -1 with plan["fail"]'s errno for their step; mountinfo comes from
# plan ("host", then "private", "bound", "hardened", "laid" as the calls go). Landlock's calls are real, so the command runs confined.
ROOT_STAND_IN = r"""import ctypes, json, os, struct, sys
sys.path.insert(0, %(here)r); sys.dont_write_bytecode = True; import confine
plan, record = %(plan)r, %(record)r
os.geteuid = lambda: 0
real = ctypes.CDLL


def plain(v):
    if isinstance(v, ctypes.Array):
        return v.raw.hex()
    if isinstance(v, bytes):
        return v.decode("latin-1")
    return v.value if hasattr(v, "value") else v


def calls():
    try:
        with open(record) as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []


def answer(step, *call):
    with open(record, "a") as f:
        f.write(json.dumps([plain(x) for x in call]) + "\n")
    fault = plan["fail"].get(step)
    if fault:
        ctypes.set_errno(fault)
        return -1
    return 0


class Libc:
    def __init__(s, *a, **k):
        s.libc = real(*a, **k)

    def __getattr__(s, n):  # an allowlist: any other call of the C library would be made for real
        if n != "prctl":
            raise AttributeError("the stand-in does not make " + n)
        return getattr(s.libc, n)

    def unshare(s, flags):
        return answer("unshare", "unshare", flags)

    def mount(s, source, target, kind, flags, data):
        f = plain(flags)
        step = ("private" if plain(target) == "/" else "tmpfs" if plain(kind) == "tmpfs" else "remount" if f & 32 else "bind")
        return answer(step, "mount", source, target, kind, flags, data)

    def syscall(s, nr, *a):
        if nr == 442:
            return answer("setattr", "mount_setattr", *a)
        if nr not in (confine.CREATE, confine.ADD_RULE, confine.RESTRICT):  # only Landlock's calls go through
            raise RuntimeError("the stand-in does not make system call %%d" %% nr)
        if nr == confine.ADD_RULE:  # the path of each rule, for the check of pseudo-terminals
            with open(record, "a") as f:
                f.write(json.dumps(["rule", os.readlink("/proc/self/fd/%%d" %% struct.unpack("=Qi", a[2].raw[:12])[1])]) + "\n")
        return s.libc.syscall(nr, *a)


def mountinfo():
    made = [c for c in calls() if c[0] == "mount" and not plan["fail"].get("private" if c[2] == "/" else "tmpfs" if c[3] == "tmpfs" else "bind")]
    if any(c[3] == "tmpfs" for c in made):
        return plan["laid"]
    bound = [c[2] for c in made if c[2] not in ("/", None) and c[2] == c[1]]  # places bound over themselves
    done = [c[2] for c in calls() if c[0] == "mount_setattr" and not plan["fail"].get("setattr")] + [
        c[2] for c in made if c[1] is None and c[4] & 32]  # made read-only: mount_setattr, or a remount (MS_REMOUNT)
    if bound and set(bound) <= set(done):
        return plan["hardened"]
    if bound:
        return plan["bound"]
    return plan["private"] if any(c[2] == "/" for c in made) else plan["host"]


def namespace():
    taken = any(c[0] == "unshare" for c in calls()) and not plan["fail"].get("unshare") and not plan.get("same")
    return "mnt:[4026532000]" if taken else "mnt:[4026531840]"


ctypes.CDLL = Libc
confine.read_mountinfo, confine.mount_ns = mountinfo, namespace
confine.read_only_here = lambda place: plan.get("read_back", True)  # statvfs would read this machine's own mounts
os.unshare = os.setns = None  # never made for real
os.environ.update(plan.get("env", {}))
if plan.get("level"):
    print(json.dumps(confine.level_here()))
    sys.exit(0)
sys.exit(confine.main(plan["words"]))
"""


# Reads confine_plants.txt into {check's name: command}. A line: name, tab, command in backquotes.
def plants():
    with open(os.path.join(HERE, "confine_plants.txt")) as f:
        return {n: c.strip().strip("`") for n, c in (line.rstrip("\n").split("\t", 1) for line in f if line.strip() and not line.startswith("#"))}


def main(words):
    if len(words) != 1 or os.path.lexists(os.path.abspath(words[0])) or re.search(r"[^A-Za-z0-9_./-]", os.path.abspath(words[0])):
        print(__doc__.split("\n")[4], file=sys.stderr)
        return 2
    work = os.path.abspath(words[0])
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    tally = [0, 0]
    # Taken at the start, apart from the code under test: whether this test itself runs inside a row that confine.py
    # confined as root (its /dev/shm is then a tmpfs named confine.py-row), its mount namespace, and /dev/shm's line.
    dev_shm = os.path.realpath("/dev/shm")
    with open("/proc/self/mountinfo", errors="surrogateescape") as f:
        table = f.read().split("\n")
    shm_lines = [line for line in table if line.split(" ")[4:5] == [dev_shm]]
    inside = any(" - tmpfs confine.py-row " in line for line in shm_lines)
    binfmt = any(line.split(" ")[4:5] == ["/proc/sys/fs/binfmt_misc"] and " - binfmt_misc " in line for line in table)
    own_ns = os.readlink("/proc/self/ns/mnt")

    def our_mounts():  # the lines of this process's mount table at or below /proc and /dev/shm, the places that confine.py changes as root
        with open("/proc/self/mountinfo", errors="surrogateescape") as f:
            return sorted(line for line in f.read().split("\n")
                          if line.split(" ")[4:5] and re.match(r"(/proc|%s)(/|$)" % re.escape(dev_shm), line.split(" ")[4]))  # fmt: skip

    with open("/proc/self/status") as f:
        status = f.read().split("\n")
    admin = any(line.startswith("CapEff:") and int(line.split()[1], 16) >> 21 & 1 for line in status)  # CAP_SYS_ADMIN
    try:  # a security module's label for this process: "unconfined" (AppArmor) or none means no module stops mount
        with open("/proc/self/attr/current") as f:
            label = f.read().strip("\x00\n ")
    except OSError:
        label = ""
    # Root that may plainly mount: CAP_SYS_ADMIN, no seccomp filter, no security module's profile. There confine.py must set
    # up the namespace, so a check that it did not is a failure, not "not run".
    may_mount = (admin and "Seccomp:\t0" in status and label in ("", "unconfined")
                 and any(line.split(" ")[3:5] == ["/", "/"] for line in table))  # and not in a chroot whose / is not a mount

    def check(ok, what, seen=""):
        tally[0 if ok else 1] += 1
        print("%s: %s%s" % (what, "as expected" if ok else "NOT as expected", "" if ok else ": " + seen), flush=True)

    got = {abi: confine.level(abi) for abi in (0, 1, 2, 3, 5, 6)}
    check(got[0][0] is None and "no Landlock" in got[0][1]
          and got[1][0].startswith(confine.FILES + "; ") and not got[1][1] & (confine.REFER | confine.TRUNCATE) and not got[1][2]
          and got[2][1] & confine.REFER and not got[2][1] & confine.TRUNCATE
          and got[3][0] == confine.FILES and got[3][1] & confine.REFER and got[3][1] & confine.TRUNCATE and not got[3][2]
          and got[5][0] == confine.FILES and not got[5][2]
          and got[6][0] == confine.BOTH and got[6][2] == confine.SCOPE_SIGNAL, "levels", repr(got))  # fmt: skip
    wrong = subprocess.run([sys.executable, "-B", CONFINE, os.path.join(work, "none"), "touch", os.path.join(work, "ran")],
                           capture_output=True, text=True)  # fmt: skip
    check(wrong.returncode == 2 and not os.path.lexists(os.path.join(work, "ran")), "a wrong call", repr(wrong))
    os.mkdir(os.path.join(work, "none"))
    stand_in = ("import sys; sys.path.insert(0, %r); sys.dont_write_bytecode = True; import confine; "
                "confine.kernel_abi = lambda: 0; sys.exit(confine.main(%r))") % (HERE, [os.path.join(work, "none"), "touch", os.path.join(work, "ran")])
    none = subprocess.run([sys.executable, "-I", "-B", "-c", stand_in], capture_output=True, text=True)  # -I: keeps the current directory off sys.path
    check(none.returncode == confine.NOT_RUN and none.stderr.startswith("confine.py: NOT CONFINED, the command is not run: ")
          and not os.path.lexists(os.path.join(work, "ran")), "no Landlock", repr(none))  # fmt: skip
    here, why, _ = confine.level_here()
    if not here:  # where Landlock works and root may plainly mount, a failed setup as root is a failure, not a skip
        broke = os.geteuid() == 0 and confine.level_here(probe_root=False)[0] and may_mount
        print("confine_test: %d checks as expected, %d otherwise; %s: %s" % (tally[0], tally[1] + bool(broke),
              "NOT as expected, as root the setup failed" if broke else "not run: the planted commands", why))  # fmt: skip
        return 1 if tally[1] or broke else 77
    run, decoy, home = (os.path.join(work, n) for n in ("run", "decoy", "home"))
    for folder in (run, decoy, home):
        os.mkdir(folder)
    with open(os.path.join(decoy, "keep.txt"), "w") as f:
        f.write("keep\n")
    os.mkdir(os.path.join(decoy, "empty"))

    # Every command from confine_plants.txt runs without the caller's BASH_ENV, ENV, SHELLOPTS and
    # BASHOPTS: a start-up file or a shell option that they name could turn on job control (set -m)
    # or change more. Nor with CONFINE_AS_ROOT, which only run_suite.py sets, for confine.py.
    plain = {n: v for n, v in os.environ.items() if n not in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "CONFINE_AS_ROOT")}

    # Runs `command` in bash under confine.py, confined to `run`; `before`: confine.py's options.
    def confined(command, *before, **env):
        return subprocess.run([sys.executable, "-B", CONFINE, run, *before, "bash", "-c", command], capture_output=True, text=True,
                              env=dict(plain, **env))  # fmt: skip

    # its TMPDIR: <folder>/tmp, made by confine.py or taken only if it is ours and closed. Three new folders of this test's
    # own, each with a tmp made beforehand: one that group and others can write in, one of mode 700, and a link.
    tmp_runs = {kind: os.path.join(work, "tmp-" + kind) for kind in ("open", "closed", "link")}
    for folder in tmp_runs.values():
        os.mkdir(folder)
    mask = os.umask(0)
    os.mkdir(os.path.join(tmp_runs["open"], "tmp"), 0o777)
    os.umask(mask)
    os.mkdir(os.path.join(tmp_runs["closed"], "tmp"), 0o700)
    os.symlink(decoy, os.path.join(tmp_runs["link"], "tmp"))
    open_tmp, closed_tmp, linked_tmp = (subprocess.run([sys.executable, "-B", CONFINE, tmp_runs[kind], "bash", "-c", "echo started"],
                                                       capture_output=True, text=True, env=plain) for kind in ("open", "closed", "link"))  # fmt: skip
    cure_told = "chmod go-w it" in open_tmp.stderr  # offered to an ordinary account, never to root
    check(open_tmp.returncode == 127 and "started" not in open_tmp.stdout and "group or others can write in it" in open_tmp.stderr
          and cure_told == (os.geteuid() != 0)
          and linked_tmp.returncode == 127 and "started" not in linked_tmp.stdout and "not a directory of this account's own" in linked_tmp.stderr
          and closed_tmp.returncode == 0 and closed_tmp.stdout == "started\n", "its TMPDIR", repr((open_tmp, closed_tmp, linked_tmp)))  # fmt: skip

    plant, not_run, not_applicable = plants(), [], []  # not applicable: checks for root, when this test does not run as root
    ran = confined(plant["a removal"].replace("{run}", run))
    check(os.path.isfile(os.path.join(decoy, "keep.txt")) and "removal 1" in ran.stdout, "a removal", repr(ran))
    for name, says in (("an overwriting", "appending 1 overwriting 1"), ("a renaming", "within 1 out 1"), ("an emptying", "emptying 1"),
                       ("a making", "folder 1 link 1 name 1 fifo 1"), ("an empty folder's removal", "rmdir 1")):  # fmt: skip
        if name == "an emptying" and confine.kernel_abi() < 3:
            not_run.append("an emptying (this kernel's Landlock does not govern truncate())")
            continue
        ran = confined(plant[name].replace("{run}", run))
        text = open(os.path.join(decoy, "keep.txt")).read() if os.path.isfile(os.path.join(decoy, "keep.txt")) else None
        check(ran.stdout.split() == says.split() and sorted(os.listdir(decoy)) == ["empty", "keep.txt"] and text == "keep\n", name, repr((ran, os.listdir(decoy), text)))
    ran = confined(plant["a home"].replace("{run}", run), HOME=home)
    check(not os.path.lexists(os.path.join(home, "planted.txt")) and os.path.isfile(os.path.join(run, "mine.txt")),
          "a home", repr(ran))  # fmt: skip
    if not here.startswith(confine.BOTH):  # as root, the level's words are followed by what root's commands get
        not_run.append("a signal")
    else:
        name = "confine-test-decoy-%d-%d" % (os.getpid(), time.monotonic_ns())
        child = subprocess.Popen(["bash", "-c", 'exec -a "$0" sleep 60', name])
        time.sleep(0.3)  # time for its exec: sleep then runs with the name as its argv[0]
        ran = confined(plant["a signal"].replace("{name}", name))
        lives = child.poll() is None
        child.kill()  # the test's own child, by its handle
        child.wait()
        check(lives and ran.returncode == -9, "a signal", repr(ran))  # pkill -f also kills the command's own shell
    without = confined(plant["its own /proc"])
    if os.geteuid() == 0 and confine.root_tried()[0] == confine.UNMADE and confine.root_tried()[2]:  # confine.py refuses --gpu here, rightly
        not_run.append("its own /proc with --gpu (as root, confine.py refuses --gpu here: %s)" % confine.root_tried()[1])
        check(without.stdout.split() == ["proc", "1"], "its own /proc", repr(without))
    else:
        with_it = confined(plant["its own /proc"], "--gpu")
        mine = subprocess.run(["bash", "-c", plant["its own /proc"]], capture_output=True, text=True, env=plain)
        check(without.stdout.split() == ["proc", "1"] and with_it.stdout == mine.stdout != "", "its own /proc", repr((without, with_it, mine)))
    # The bare side starts bash by its path, as the confined side starts Python by its path: here Python's subprocess starts a
    # program named by a path through posix_spawn, and a bare name through fork. Likely, where clone3 is refused, glibc's
    # posix_spawn leaves its two internal signals, 32 and 33, ignored in the child, which confine.py does not do (seen under a filter that refuses clone3).
    bare = [shutil.which("bash", path=plain.get("PATH")) or "bash", "-c"]
    ran, mine = confined(plant["what is ignored"]), subprocess.run(bare + [plant["what is ignored"]], capture_output=True, text=True, env=plain)
    check(ran.stdout.startswith("SigIgn:") and ran.stdout == mine.stdout, "what is ignored", repr((ran, mine)))
    filtered = ["group", "0", "all", "1", "own", "0", "child", "0"]
    ran = confined(plant["the signals' filter"], CONFINE_SIGNALS_FILTER="always")
    check(ran.stdout.split() == filtered, "the signals' filter", repr(ran))
    for abi in (1, 2, 3, 5):
        if abi > confine.kernel_abi():  # a kernel rejects rules on rights that it does not know
            not_run.append("the second level with the kernel's answer made %d" % abi)
            continue
        second = ("import sys; sys.path.insert(0, %r); sys.dont_write_bytecode = True; import confine; "
                  "confine.kernel_abi = lambda: %d; sys.exit(confine.main(%r))") % (HERE, abi, [run, "bash", "-c", plant["the signals' filter"]])
        ran = subprocess.run([sys.executable, "-I", "-B", "-c", second], capture_output=True, text=True, env={n: v for n, v in plain.items() if n != "CONFINE_SIGNALS_FILTER"})
        check(ran.stdout.split() == filtered, "the signals' filter, laid by the second level itself (the kernel's answer made %d)" % abi, repr(ran))
    also = os.path.join(work, "also")
    os.mkdir(also)
    ran = confined(plant["the option --also"].replace("{also}", also).replace("{run}", run), "--also", also)
    check(ran.stdout.split() == ["also", "0", "decoy", "1"] and os.path.isfile(os.path.join(also, "made.txt"))
          and sorted(os.listdir(decoy)) == ["empty", "keep.txt"], "the option --also", repr((ran, os.listdir(decoy))))  # fmt: skip
    if not os.path.isdir("/dev/shm"):
        not_run.append("the rule for /dev/shm (no folder /dev/shm here)")
    else:
        shm = "confine-test-shm-%d-%d" % (os.getpid(), time.monotonic_ns())
        try:
            ran = confined(plant["the rule for /dev/shm"].replace("{name}", shm))
            check(ran.stdout.split() == ["file", "0", "removal", "0", "folder", "1"], "the rule for /dev/shm", repr(ran))
        finally:  # this run's own two names, should a step have gone otherwise
            if os.path.lexists("/dev/shm/" + shm):
                os.remove("/dev/shm/" + shm)
            if os.path.isdir("/dev/shm/" + shm + ".d") and not os.path.islink("/dev/shm/" + shm + ".d"):
                os.rmdir("/dev/shm/" + shm + ".d")
    # a device node: refused in the confined folder for every account (as root a device file there would reach a disk).
    # Without the refusal any account may make the node 0:0 on ext4, xfs or tmpfs, and an account other than root gets EPERM
    # for the block device and for 1:3, for want of CAP_MKNOD; so EACCES for these two shows that both refusals are in force.
    # On some kernels Landlock counts the making of the node 0:0 as the making of a regular file, which confine.py allows in
    # the given folders, and not as the making of a character device (seen on one 6.18 kernel, whose Landlock reports one
    # more erratum than the others; whether that is the cause is not known; not seen on another 6.18 kernel or on 6.12).
    # There the node is made, or the file system refuses it with EPERM (overlayfs does).
    nodes = [os.path.join(run, n) for n in ("zero-node", "block-node", "null-twin")]
    try:
        three = plant["a device node"].replace("[:3 if os.geteuid() == 0 else 2]", "")  # all three nodes, whatever the account
        ran = confined(three.replace("{run}", run))
        says = ["13"] * 3  # EACCES: Landlock's refusal
        got, spared = ran.stdout.split(), []
        if got[:1] in (["0"], ["1"]) and got[1:] == says[1:]:  # only the inert node 0:0 was not refused by Landlock: made, or EPERM
            told = "the character device 0:0 (%s; the block device and the character device 1:3 were refused with EACCES: on this kernel Landlock's refusal of character devices does not reach the inert node 0:0)" % (
                "it was made in the confined folder; this test removes it" if got[0] == "0" else "the file system refused it with EPERM")  # fmt: skip
            print("a device node: not run for " + told, flush=True)
            not_run.append("a device node, for " + told)
            got[0], spared = "13", nodes[:1]  # "finally" below removes the node
        check(three != plant["a device node"] and got == says and not any(os.path.lexists(n) for n in nodes if n not in spared), "a device node", repr(ran))
    finally:  # remove this check's files in case any was made
        for n in nodes:
            if os.path.lexists(n) and not os.path.isdir(n):
                os.remove(n)
    # the mount namespace: root's command gets one of its own, unless this test already runs inside a command that confine.py
    # confined as root
    ran = confined(plant["the mount namespace"])
    seen = ran.stdout.strip()
    if os.geteuid() == 0 and not inside and admin and confine.root_tried()[0] in (confine.UNMADE, confine.ALREADY) and not (
            may_mount and confine.root_tried()[0] == confine.UNMADE):
        not_run.append("the mount namespace only for root (as root with CAP_SYS_ADMIN, but confine.py sets up no namespace here: %s)"
                       % (confine.root_tried()[1] or "there is no /dev/shm, and /proc's system settings are read-only already"))  # fmt: skip
    else:
        differs = os.geteuid() == 0 and not inside and admin
        check(ran.returncode == 0 and seen.startswith("mnt:[") and (seen != own_ns) == differs, "the mount namespace only for root",
              repr((ran, own_ns, os.geteuid(), inside, admin)))  # fmt: skip
    if not os.path.isdir("/dev/shm"):
        not_run.append("multiprocessing (no folder /dev/shm here)")
    else:
        ran = confined(plant["multiprocessing"])
        check(ran.stdout.strip() == "mp ok", "multiprocessing", repr(ran))
    # as root, /proc's system places: opened for writing and closed at once, never written; each must fail with EROFS (30)
    if os.geteuid() != 0:
        not_applicable.append("as root, /proc's system places and pseudo-terminals")
    else:
        paths = [p for p in ("/proc/sys/kernel/domainname", "/proc/sysrq-trigger", "/proc/irq/default_smp_affinity") if os.path.exists(p)]
        paths += sorted(p for p in glob.glob("/proc/bus/pci/*/*") if os.path.isfile(p))[:1]  # a device's configuration space
        paths += ["/proc/sys/fs/binfmt_misc/register"] if binfmt else []  # only where binfmt_misc is mounted, not a mere automount
        before = our_mounts()
        ran = confined(plant["as root, /proc's system places"].replace("{paths}", " ".join(paths)), "--gpu")
        if ran.returncode == confine.NOT_RUN and "as root, with --gpu: " in ran.stderr and not may_mount:
            not_run.append("as root, /proc's system places (confine.py refuses --gpu here: %s)" % ran.stderr.strip())
        else:
            check(paths and ran.stdout.split() == ["30"] * len(paths) and our_mounts() == before, "as root, /proc's system places",
                  repr((paths, ran, before, our_mounts())))  # fmt: skip
        terminal = confined(plant["as root, /proc's system places"].replace("{paths}", "/dev/ptmx"))
        if not os.path.exists("/dev/ptmx"):
            not_run.append("as root, pseudo-terminals (no /dev/ptmx here)")
        else:
            check(terminal.stdout.split() == ["13"], "as root, pseudo-terminals", repr(terminal))
    # as root, another account's file in /dev/shm: a decoy of this test's own, handed to uid 65534, must stay out of reach
    if os.geteuid() != 0:
        not_applicable.append("as root, another account's file in /dev/shm")
    elif inside:
        not_run.append("as root, another account's file in /dev/shm (this test already runs inside a row that confine.py confined as root, "
                       "so its /dev/shm is that row's own)")  # fmt: skip
    elif confine.root_tried()[0] == confine.UNMADE and not may_mount:
        not_run.append("as root, another account's file in /dev/shm (confine.py cannot set up a mount namespace here, so /dev/shm is shared: %s)"
                       % confine.root_tried()[1])  # fmt: skip
    elif not os.path.isdir("/dev/shm"):
        not_run.append("as root, another account's file in /dev/shm (no folder /dev/shm here)")
    else:
        theirs = "confine-test-theirs-%d-%d" % (os.getpid(), time.monotonic_ns())
        decoy_path = os.path.join("/dev/shm", theirs)

        def shm_state():  # seen from outside the row: /dev/shm's device, and the mount lines at or below /proc and /dev/shm
            return os.stat("/dev/shm").st_dev, our_mounts()

        ours = False
        try:
            made = os.open(decoy_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
            ours = True
            os.write(made, b"theirs\n")
            os.close(made)
            os.chown(decoy_path, 65534, 65534, follow_symlinks=False)
            before = shm_state()
            ran = confined(plant["as root, another account's file in /dev/shm"].replace("{name}", theirs))
            with open(decoy_path, "rb") as f:
                intact = f.read() == b"theirs\n" and os.lstat(decoy_path).st_uid == 65534
            check(ran.stdout.split() == ["ls", "2", "append", "0", "rm", "0", "mine", "0"] and intact and not os.path.lexists(decoy_path + ".mine")
                  and shm_state() == before, "as root, another account's file in /dev/shm", repr(ran))  # fmt: skip
        except OSError as fault:
            check(False, "as root, another account's file in /dev/shm", repr(fault))
        finally:  # remove this check's files, once it has made the decoy
            for n in (decoy_path, decoy_path + ".mine") if ours else ():
                if os.path.lexists(n) and not os.path.isdir(n):
                    os.remove(n)
    # as root, the order of the calls, and as root, when the namespace can't be made: the stand-in for root above
    places = [p for p in ("/proc/sys", "/proc/sysrq-trigger", "/proc/irq", "/proc/bus", "/proc/fs") if os.path.lexists(p)]
    head = ("1 0 0:1 / / rw,relatime shared:1 - ext4 /dev/root rw\n2 1 0:2 / /proc rw,nosuid,nodev,noexec shared:2 - proc proc rw\n"
            "4 2 0:4 / /proc/sys/fs/binfmt_misc rw,relatime shared:4 - autofs systemd-1 rw\n"  # hidden below a bind of /proc/sys
            + ("3 1 0:3 / %s rw,nosuid,nodev shared:3 - tmpfs tmpfs rw\n" % dev_shm if os.path.isdir(dev_shm) else ""))  # fmt: skip
    private = re.sub(r" shared:[0-9]+", "", head)
    binds = "".join("%d 2 0:2 /%s %s ro,nosuid,nodev,noexec - proc proc rw\n" % (10 + i, os.path.basename(p), p) for i, p in enumerate(places))
    copy = "30 10 0:4 / /proc/sys/fs/binfmt_misc %s,relatime - autofs systemd-1 rw\n" if places[:1] == ["/proc/sys"] else ""  # the bind's own copy
    own_shm = "20 3 0:20 / %s rw,nosuid,nodev - tmpfs confine.py-row rw\n" % dev_shm if os.path.isdir(dev_shm) else ""
    laid = private + binds + copy.replace("%s", "ro") + own_shm
    bound = private + binds.replace(" ro,", " rw,")  # bound over themselves, not yet read-only
    runc = private + binds + copy.replace("%s", "ro")  # read-only already, as container runtimes leave them
    expected = [["unshare", 0x20000], ["mount", None, "/", None, 0x44000, None]]  # CLONE_NEWNS; MS_REC | MS_PRIVATE
    for p in places:  # MS_BIND | MS_REC; then mount_setattr(AT_FDCWD, p, AT_RECURSIVE, {attr_set: MOUNT_ATTR_RDONLY}, 32)
        expected += [["mount", p, p, None, 0x5000, None], ["mount_setattr", -100, p, 0x8000, "01" + "00" * 31, 32]]

    def stand_in(tag, **plan):
        folder, record = os.path.join(work, "root-" + tag), os.path.join(work, "root-" + tag + ".record")
        os.mkdir(folder)
        marker = os.path.join(folder, "ran")
        gpu = plan.pop("gpu", True)
        stderr_as = plan.pop("stderr_as", None)  # "full", "closed" (before the program starts), "broken" (a pipe whose reader has gone) or "full-pipe" (a full pipe that nobody reads)
        plan = dict({"fail": {}, "host": head, "private": private, "bound": bound + copy.replace("%s", "rw"), "hardened": runc, "laid": laid,
                     "words": [folder] + (["--gpu"] if gpu else []) + ["touch", marker]}, **plan)  # fmt: skip
        where, closing = subprocess.PIPE, None
        if stderr_as == "full":
            where = open("/dev/full", "w")
        elif stderr_as == "closed":
            where, closing = subprocess.DEVNULL, lambda: os.close(2)
        elif stderr_as == "broken":
            reader, where = os.pipe()
            os.close(reader)
        elif stderr_as == "full-pipe":  # open, never read, and filled to the brim, then blocking again
            reader, where = os.pipe()
            os.set_blocking(where, False)
            try:
                while True:
                    os.write(where, b"x" * 65536)
            except BlockingIOError:
                pass
            os.set_blocking(where, True)
        try:
            done = subprocess.run([sys.executable, "-I", "-B", "-c", ROOT_STAND_IN % {"here": HERE, "plan": plan, "record": record}],
                                  stdout=subprocess.PIPE, stderr=where, text=True, env=plain, preexec_fn=closing, timeout=60)  # fmt: skip
        except subprocess.TimeoutExpired as stuck:  # no end within 60 s, most likely a write waiting on the full pipe, so the command never started
            done = subprocess.CompletedProcess(stuck.cmd, None, (stuck.stdout or b"").decode(errors="replace"), None)
        finally:
            if stderr_as == "full":
                where.close()
            elif stderr_as in ("broken", "full-pipe"):
                os.close(where)
            if stderr_as == "full-pipe":
                os.close(reader)
        calls = [c for c in (json.loads(line) for line in open(record)) if c[0] != "rule"] if os.path.exists(record) else []  # the mount calls
        done.args = "the stand-in for root (%s)" % tag  # so that a failure shows this label, not the whole stand-in program
        return done, calls, os.path.lexists(marker)

    def tmpfs_call(c):
        return c[:4] == ["mount", "confine.py-row", dev_shm, "tmpfs"] and c[4] & 0x6 == 0x6 and not c[4] & ~0xE and c[5].startswith("mode=1777")

    done, calls, ran_it = stand_in("order")
    pairs = [calls[i:i + 2] for i in range(2, len(expected), 2)]  # the places may come in any order, each bind right before its mount_setattr
    check(done.returncode == 0 and ran_it and calls[:2] == expected[:2] and sorted(pairs) == sorted(expected[i:i + 2] for i in range(2, len(expected), 2))
          and (len(calls) == len(expected) + 1 and tmpfs_call(calls[-1]) if os.path.isdir(dev_shm) else len(calls) == len(expected)),
          "as root, the order of the calls (a stand-in)", repr((done, calls)))  # fmt: skip
    refused = "confine.py: NOT CONFINED, the command is not run: as root"
    cases = {"the mounts are not made private": (dict(fail={"private": errno.EINVAL}, gpu=False), expected[:2]),
             "the mounts are not made private, where the places are read-only already": (dict(fail={"private": errno.EACCES}, host=runc, private=runc, gpu=False), expected[:2]),
             "a mount still shared after that": (dict(private=head), expected[:2]),
             "the namespace did not change": (dict(same=True), expected[:1]),
             "a place left writable": (dict(laid=laid.replace(" ro,nosuid", " rw,nosuid", 1) if places else laid.replace("confine.py-row", "tmpfs")), None),
             "no namespace, and /proc writable (--gpu)": (dict(fail={"unshare": errno.EPERM}), expected[:1]),
             "unshare fails for want of memory": (dict(fail={"unshare": errno.ENOMEM}, host=runc), expected[:1]),
             "no namespace, where the runner's first line promised one": (dict(fail={"unshare": errno.EPERM}, host=runc, env={"CONFINE_AS_ROOT": "laid"}), expected[:1]),
             "mount_setattr refused, with an automount below": (dict(fail={"setattr": errno.ENOSYS}), None),
             "a bind refused": (dict(fail={"bind": errno.EPERM}, gpu=False), None),
             "a bind fails for want of memory": (dict(fail={"bind": errno.ENOMEM}, gpu=False), None),
             "the tmpfs refused, where the runner's first line promised it": (dict(fail={"tmpfs": errno.EACCES}, env={"CONFINE_AS_ROOT": "laid"}, gpu=False), None),
             "the tmpfs fails for want of memory": (dict(fail={"tmpfs": errno.ENOMEM}, gpu=False), None),
             "the tmpfs refused": (dict(fail={"tmpfs": errno.EACCES}, gpu=False), None),
             "laid, where the runner's first line told of none": (dict(env={"CONFINE_AS_ROOT": "unmade"}, gpu=False), None),
             "a place does not read back as read-only": (dict(read_back=False, gpu=False), None),
             "mount_setattr fails otherwise": (dict(fail={"setattr": errno.EINVAL}, gpu=False), None),
             "the remount fails": (dict(fail={"setattr": errno.ENOSYS, "remount": errno.EPERM}, bound=bound, gpu=False), None)}
    if not os.path.isdir(dev_shm):  # no /dev/shm, so no tmpfs is mounted: the three cases of a failing tmpfs cannot happen
        cases = {case: got for case, got in cases.items() if not case.startswith("the tmpfs")}
    seen = {}
    for case, (plan, calls_then) in cases.items():
        done, calls, ran_it = stand_in(re.sub(r"[^a-z]+", "-", case), **plan)
        if not (done.returncode == confine.NOT_RUN and done.stderr.startswith(refused) and not ran_it and (calls_then is None or calls == calls_then)):
            seen[case] = (done, calls, ran_it)
    check(not seen, "as root, a failed or unexpected step (stand-ins: nothing is run)", repr(seen))
    try:
        full_device = stat.S_ISCHR(os.stat("/dev/full").st_mode)
    except OSError:
        full_device = False
    runs = {"no-namespace-read-only": (dict(fail={"unshare": errno.EPERM}, host=runc), expected[:1]),
            "no-namespace-writable-no-gpu": (dict(fail={"unshare": errno.EPERM}, gpu=False), expected[:1]),
            "nested": (dict(host=laid), []),
            "unmade-as-the-runner-told": (dict(fail={"unshare": errno.EPERM}, host=runc, env={"CONFINE_AS_ROOT": "unmade"}), expected[:1]),
            "laid-as-the-runner-told": (dict(env={"CONFINE_AS_ROOT": "laid"}), None),
            **({"stderr-full": (dict(fail={"unshare": errno.EPERM}, host=runc, stderr_as="full"), expected[:1])}
               if os.path.isdir(dev_shm) and full_device else {}),  # never opened unless it is the device: as root, that name would make a file
            **({"stderr-closed": (dict(fail={"unshare": errno.EPERM}, host=runc, stderr_as="closed"), expected[:1]),
                "stderr-broken": (dict(fail={"unshare": errno.EPERM}, host=runc, stderr_as="broken"), expected[:1]),
                "stderr-full-pipe": (dict(fail={"unshare": errno.EPERM}, host=runc, stderr_as="full-pipe"), expected[:1])}
               if os.path.isdir(dev_shm) else {})}  # without /dev/shm no notice is due, so these would show nothing
    seen = {tag: stand_in(tag, **plan) + (calls_then,) for tag, (plan, calls_then) in runs.items()}
    seen = {tag: got for tag, got in seen.items() if not (got[0].returncode == 0 and got[2] and (got[3] is None or got[1] == got[3])
                                                           and (not tag.startswith("no-namespace") or not os.path.isdir(dev_shm)
                                                                or "shares /dev/shm with whatever else runs here" in got[0].stderr)
                                                           and not (tag.startswith("unmade-as") and "shares /dev/shm" in got[0].stderr)
                                                           and not (tag.startswith("stderr-") and got[0].stdout))}  # fmt: skip
    check(not seen, "as root, nothing to mount, or the namespace refused without --gpu (stand-ins: the command runs)", repr(seen))
    if not os.path.isdir(dev_shm):
        not_run.append("as root, the notice with standard error full, closed or a broken pipe (no folder /dev/shm here)")
    elif not full_device:
        not_run.append("as root, the notice with standard error on /dev/full (no such device here)")
    # Linux before 5.12, or a filter that refuses mount_setattr: each place is remounted read-only (MS_REMOUNT | MS_BIND |
    # MS_RDONLY, 0x1021, with the bind's nosuid, nodev and noexec, 0xE), right after its bind and the refused call.
    got = {fault: stand_in("remounted-%d" % fault, fail={"setattr": fault}, bound=bound) for fault in (errno.ENOSYS, errno.EPERM)}
    threes = {fault: [calls[i:i + 3] for i in range(2, 2 + 3 * len(places), 3)] for fault, (_, calls, _) in got.items()}
    check(all(done.returncode == 0 and ran_it and calls[:2] == expected[:2]
              and sorted(threes[fault]) == sorted(expected[i:i + 2] + [["mount", None, expected[i][2], None, 0x102F, None]] for i in range(2, len(expected), 2))
              for fault, (done, calls, ran_it) in got.items()),
          "as root, mount_setattr refused (a stand-in: each place remounted read-only)", repr(got))  # fmt: skip
    said, refusal = stand_in("level", level=True), stand_in("level-refused", level=True, same=True)
    words_of = json.loads(said[0].stdout or "[null]")[0] or ""
    check(words_of.endswith("; as root: each command gets /proc's system settings (/proc/sys and four more places) read-only"
                            + (" and an empty /dev/shm of its own" if os.path.isdir(dev_shm) else ""))
          and json.loads(refusal[0].stdout or "[1]")[0] is None and (json.loads(refusal[0].stdout or "[0, 0]")[1] or "").startswith("as root, "),
          "as root, the level (stand-ins)", repr((said, refusal)))  # fmt: skip
    # As root no pseudo-terminal may be written: the rules are real, so this holds of the stand-in's own command too.
    if not os.path.exists("/dev/ptmx"):
        not_run.append("as root, pseudo-terminals (a stand-in; no /dev/ptmx here)")
    else:
        opens = "exec 3<>/dev/ptmx && echo opened || echo refused"
        mine = confined(opens)  # this account, confined: may open it
        folder = os.path.join(work, "root-terminals")
        os.mkdir(folder)
        plan = {"fail": {}, "host": laid, "private": laid, "bound": laid, "hardened": laid, "laid": laid, "words": [folder, "bash", "-c", opens]}
        done = subprocess.run([sys.executable, "-I", "-B", "-c", ROOT_STAND_IN % {"here": HERE, "plan": plan, "record": folder + ".record"}],
                              capture_output=True, text=True, env=plain)  # fmt: skip
        done.args = "the stand-in for root (terminals)"
        rules = [json.loads(line)[1] for line in open(folder + ".record") if line.startswith('["rule"')] if os.path.exists(folder + ".record") else []
        control = os.geteuid() == 0 or mine.stdout.strip() == "opened"  # this account, confined, may open it (root may not)
        check(control and done.stdout.strip() == "refused" and rules and not any(
            r in ("/dev/ptmx", "/dev/pts") or r.startswith("/dev/pts/") for r in rules), "as root, pseudo-terminals (a stand-in)",
            repr((mine, done, rules)))  # fmt: skip
    # a set-up that fails half way: libc.syscall fails with EPERM at one Landlock step; confine.py must refuse and run nothing
    failed = {}
    for step, fails in (("creation", "nr == confine.CREATE and a[2] == 0"), ("a rule", "nr == confine.ADD_RULE and s.rules == 1"),
                        ("restriction", "nr == confine.RESTRICT")):  # fmt: skip
        marker = os.path.join(run, "ran-" + step.replace(" ", "-"))  # inside run: a command run confined after all could make it
        stand_in = ("import ctypes, errno, sys; sys.path.insert(0, %r); sys.dont_write_bytecode = True; import confine\n"
                    "real = ctypes.CDLL\n"
                    "class Libc:\n"
                    "    def __init__(s, *a, **k): s.libc, s.rules = real(*a, **k), 0\n"
                    "    def __getattr__(s, n): return getattr(s.libc, n)\n"
                    "    def syscall(s, nr, *a):\n"
                    "        if %s:\n"
                    "            ctypes.set_errno(errno.EPERM); return -1\n"
                    "        s.rules += nr == confine.ADD_RULE\n"
                    "        return s.libc.syscall(nr, *a)\n"
                    "ctypes.CDLL = Libc\n"
                    "confine.root_walls = lambda libc: ('laid', '', [])\n"  # as root, no real namespace: this check is about Landlock's steps
                    "sys.exit(confine.main(%r))") % (HERE, fails, [run, "touch", marker])
        failed[step] = subprocess.run([sys.executable, "-I", "-B", "-c", stand_in], capture_output=True, text=True, env=plain)
        failed[step] = (failed[step], os.path.lexists(marker))
    says = {"creation": "the kernel refused the rules", "a rule": "the rule for ", "restriction": "the confinement could not be laid"}
    check(all(r.returncode == confine.NOT_RUN and r.stderr.startswith("confine.py: NOT CONFINED, the command is not run: " + says[step])
              and not made for step, (r, made) in failed.items()), "a set-up that fails half way", repr(failed))  # fmt: skip
    print("confine_test: %d checks as expected, %d otherwise%s%s (this kernel: %s)" % (
        tally[0], tally[1], "; not run: " + "; ".join(not_run) if not_run else "",
        "; for root alone, so not applicable here: " + "; ".join(not_applicable) if not_applicable else "", here))  # fmt: skip
    return 1 if tally[1] else 77 if not_run else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
