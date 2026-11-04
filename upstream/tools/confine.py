#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Runs a command in a Landlock sandbox: it and its descendants can write, create, rename and remove
files only below the given folders (exceptions below) and, from Linux 6.12, signal only each other.

usage: confine.py <folder> [--also <folder>] ... [--gpu] <command> [<argument> ...]
       confine.py --level          prints the level that this kernel gives, and nothing else
Started by the runner; by hand: below a directory of your own (<folder>, in which it makes <folder>/tmp).

run_suite.py runs each test under confine.py (unless given --unconfined). The sandbox is there to
limit what a faulty test can damage by accident. It is not a barrier against a test written to do
harm, and when the suite runs as root it cannot be one. Run as root, a test can still change the
owner, mode or times of any file on the machine (chmod, chown, touch), which the sandbox does not
limit at all, and it keeps root's other powers, such as the network and the machine itself ("Not
confined" below).
In the given folders the command cannot make device files, whatever the account (mknod there fails
with EACCES; of the inert character node 0:0 nothing is promised). Outside them it can also write:
  - /dev/null, /dev/zero, /dev/full and, except as root, pseudo-terminals (/dev/ptmx, /dev/pts): as
    root these would reach every account's terminals, so none may be opened for writing by its name,
    not even the command's own terminal (reading one, and ioctl on one, stay possible: see "Not
    confined");
  - regular files in /dev/shm and in every folder below it, which it can also create and remove:
    Python's multiprocessing creates its semaphores there. So it can remove files that other
    processes of its user account have there (under run_suite.py, other tests' files);
  - with --gpu, the GPU device files that exist at its start (/dev/nvidia*) and the files of /proc.
    CUDA names its threads by writing /proc/self/task/<id>/comm, and cuInit fails if it cannot. So
    the command can write whatever else the account may write in /proc, such as the oom_score_adj of
    any of its processes.
As root, confine.py first gives the command a mount namespace of its own. It makes every mount in
it private, so that nothing mounted there reaches the rest of the machine; then, in every mount of
the whole proc file system, it makes /proc/sys, /proc/sysrq-trigger, /proc/irq, /proc/bus and
/proc/fs read-only, binding each one that is not read-only already over itself; then it mounts an
empty tmpfs of the command's own on /dev/shm (the size of the machine's /dev/shm when that is a
tmpfs mounted at /dev/shm with a size limit, otherwise half of the memory; so commands side by side may together hold more
there than the machine's one /dev/shm allowed); and it checks the result. So a command run as root
cannot write the kernel's settings in those five places by accident, nor reach other accounts'
files in /dev/shm (for calls made to undo this, see "Not confined"). If a step fails, or its result is not as expected, nothing is run, with one exception.
Where the machine refuses the first step, the new namespace itself (root without CAP_SYS_ADMIN, as
in Docker's and Kubernetes' containers by default, or a filter that refuses unshare), a command without --gpu runs all the same,
since it cannot write /proc, and a command with --gpu runs only if those five places are read-only
already; /dev/shm is then shared, and the command can remove, overwrite or empty any account's files
there. When confine.py is started by hand rather than by run_suite.py (which prints its own line about
this), it says so in one line on standard error just before the command starts; if standard error
refuses writes (/dev/full, a full disk), is closed, or is a pipe that is full or whose reader has gone
when confine.py looks, the line is lost and the command still runs. Any later
failure, refused or not, runs nothing: so in a chroot whose / is not a mount point, or in a
container where root may make the namespace but not mount (Docker with CAP_SYS_ADMIN under its
AppArmor profile, say), nothing runs as root, and run_suite.py's --unconfined is the way to run the
tests without the sandbox, on purpose. A confine.py started
inside a command that confine.py confined as root sees that all this is set up and mounts nothing
more (the Landlock rules are still laid). "As root" here means effective user id 0.
The command's TMPDIR is <folder>/tmp, under the first folder. SIGPIPE and SIGXFSZ, which Python
ignores, are reset to their defaults for the command.
Not confined:
  - reading and executing files;
  - a file's mode, owner, times and extended attributes: chmod, chown and touch -d work on any file
    of the account, as root on any file;
  - on Landlock ABI 1 and 2 (ABI: the kernel's Landlock version), truncate() on a file that the
    account may write (--level reports this);
  - a descriptor that was already open: if standard error is a file outside the folders, ">&2"
    works and "> /dev/stderr" fails;
  - locks on files, and ioctl on devices, that the command can open;
  - other processes' limits and niceness (prlimit, renice);
  - System V shared memory, semaphores and message queues; the network;
  - how much disk and memory and how many processes it uses;
  - what an unconfined process does on its behalf;
  - a device file that is already in a given folder: the command cannot make one there, but it
    can open, read and write one that exists (of the inert character node 0:0 nothing is promised);
  - as root, root's other powers: chmod, chown and extended attributes on any file (a program's
    file capabilities among them); ioctl on devices that it may open; the network, and other
    programs' sockets (systemd's or Docker's, through which root can stop services or restart the
    machine); modules, reboot and the clock; with --gpu, files of /proc outside the five places above that root may write
    (the oom_score_adj of any process, for example);
  - as root, calls made to undo the read-only places: the command keeps CAP_SYS_ADMIN, and Landlock
    does not refuse mount_setattr(2) or open_tree(2). A root command with --gpu (without it, Landlock
    still refuses writes to /proc) has been seen to make /proc's system settings writable again in
    two ways: mount_setattr clearing read-only on /proc/sys, and open_tree of /proc. The read-only
    places stop a plain open by mistake, not a command that means to write.
A side effect, as root: while the command runs, the machine cannot fully let go of a file system
that is unmounted meanwhile (its device stays busy), because the command's namespace keeps its own
copy of every mount.

On a kernel whose Landlock cannot confine signals (ABI 1 to 5: Linux 5.13 to 6.11), a seccomp filter
(a filter on system calls) blocks a signal to -1, which would reach every process that the account
may signal. A signal to another process group or to an unrelated process still passes: the filter
cannot tell their ids from the command's own. As root, an unrelated process can be any process on
the machine, so on such a kernel a test run as root that ends processes by name ends every matching
process on the machine. The two levels, as --level prints them:
"files and signals" (ABI 6 or higher: Linux 6.12 or later) and
"files; of signals only those to -1 are refused (...)".
As root, --level adds what root's commands get, found by trying the same steps in a child process
that then exits: "; as root: each command gets /proc's system settings (/proc/sys and four more
places) read-only and an empty /dev/shm of its own", or, where the machine refuses the namespace, what each command gets instead. Where there is no
level, or as root the setup fails in any way other than a refused namespace, --level prints "NOT
CONFINED: <why>" and exits with 125.
Everything printed after the command starts is its own; just before, as root where the namespace was
refused and run_suite.py did not start it, confine.py prints its one line that /dev/shm is shared.
Exit codes:
  125  the confinement could not be set up (no Landlock, for example, or, as root, the command's own
       mount namespace is needed and could not be set up); standard error has
       "confine.py: NOT CONFINED, the command is not run: <why>".
       Not 3, which run_suite.py takes to mean that something false was accepted.
  2    a wrong call or a missing folder.
  127  confined, but the command could not be started (not found, for example, or TMPDIR could not
       be created, or exists but is not a directory of this account's own that group and others cannot
       write in).
  Otherwise the command's own.
"""

import ctypes
import errno
import glob
import os
import platform
import re
import select
import signal
import stat
import struct
import sys

# System calls landlock_create_ruleset, landlock_add_rule and landlock_restrict_self. WRITE_FILE to
# SCOPE_SIGNAL: LANDLOCK_ACCESS_FS_* and LANDLOCK_SCOPE_* bits ("ABI n": first ABI with the bit).
CREATE, ADD_RULE, RESTRICT = 444, 445, 446  # the same numbers on x86_64 and aarch64
PR_SET_NO_NEW_PRIVS = 38
WRITE_FILE, REMOVE_DIR, REMOVE_FILE = 1 << 1, 1 << 4, 1 << 5
MAKE = sum(1 << b for b in range(6, 13))  # bits MAKE_CHAR, _DIR, _REG, _SOCK, _FIFO, _BLOCK, _SYM
REFER, TRUNCATE = 1 << 13, 1 << 14  # ABI 2, ABI 3
SCOPE_SIGNAL = 1 << 1  # ABI 6
FILE_RIGHTS = WRITE_FILE | TRUNCATE
BOTH = "files and signals"
FILES = "files; of signals only those to -1 are refused (this kernel cannot confine signals)"


def level(abi):
    """Returns (level, rights to restrict, scope) for a Landlock ABI; for 0, (None, reason, 0)."""
    if abi < 1:
        return None, "this kernel gives no Landlock (Linux 5.13 or later, with Landlock among its security modules)", 0
    handled = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE | (REFER if abi >= 2 else 0) | (TRUNCATE if abi >= 3 else 0)
    if abi >= 6:
        return BOTH, handled, SCOPE_SIGNAL
    return FILES + ("" if abi >= 3 else "; a file may still be emptied by truncate() on this kernel (Landlock ABI below 3)"), handled, 0


# The seccomp filter for Landlock ABI 1 to 5: kill() and rt_sigqueueinfo() to -1 fail with EPERM. A
# loop over process ids, as in "pkill -u", can still end every process of the account.
# ARCH: (AUDIT_ARCH_*, kill, rt_sigqueueinfo) for a machine's native system calls and for its 32-bit
# ones. On x86_64 every x32 system call (bit `X32` of the call number) fails with EPERM: otherwise
# kill() could pass under its x32 number.
# LD's offsets in struct seccomp_data: 0 the call number, 4 architecture, 16 first argument.
ARCH = {"x86_64": ((0xC000003E, 62, 129), (0x40000003, 37, 178)), "aarch64": ((0xC00000B7, 129, 138), (0x40000028, 37, 178))}
PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 22, 2
LD, JEQ, JSET, RET = 0x20, 0x15, 0x45, 0x06  # BPF_LD|BPF_W|BPF_ABS, BPF_JMP|BPF_JEQ|BPF_K, BPF_JMP|BPF_JSET|BPF_K, BPF_RET|BPF_K
ALLOW, EPERM = 0x7FFF0000, 0x00050000 | 1  # SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO | EPERM
X32 = 0x40000000


def signals_filter():
    """Returns this machine's filter as packed struct sock_filter entries, or None if the machine is
    not in ARCH. A system call of an architecture other than its two fails with EPERM."""
    if platform.machine() not in ARCH:
        return None
    code, fix = [], []  # fix: (index, label of jt, label of jf) to be resolved

    def op(c, k, jt=None, jf=None):
        fix.append((len(code), jt, jf))
        code.append([c, 0, 0, k])

    labels = {}
    native, compat = ARCH[platform.machine()]
    op(LD, 4)
    op(JEQ, native[0], "native", None)
    op(JEQ, compat[0], "compat", "refuse")
    for name, (_, kill, queue) in (("native", native), ("compat", compat)):
        labels[name] = len(code)
        op(LD, 0)
        if name == "native" and platform.machine() == "x86_64":
            op(JSET, X32, "refuse", None)
        op(JEQ, kill, "judge", None)
        op(JEQ, queue, "judge", "allow")
    labels["judge"] = len(code)
    op(LD, 16)  # the process id's low word
    op(JEQ, 0xFFFFFFFF, "refuse", "allow")  # -1: every process of the account
    labels["refuse"] = len(code)
    op(RET, EPERM)
    labels["allow"] = len(code)
    op(RET, ALLOW)
    for at, jt, jf in fix:
        code[at][1] = labels[jt] - at - 1 if jt else 0
        code[at][2] = labels[jf] - at - 1 if jf else 0
    return b"".join(struct.pack("=HBBI", *c) for c in code)


# Installs the filter that signals_filter() returns. Call it after prctl(PR_SET_NO_NEW_PRIVS):
# without that, installing a filter needs CAP_SYS_ADMIN.
def lay_signals_filter(libc):
    program = signals_filter()
    if program is None:
        refuse("no signals' filter for a %s machine" % platform.machine())

    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_char_p)]

    prog = Prog(len(program) // 8, program)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) < 0:
        refuse("the signals' filter could not be laid (%s)" % os.strerror(ctypes.get_errno()))


# Used only as root (euid 0), to give the command its own mount namespace (the usage text, "As root"). Values from the
# Linux headers: sched.h CLONE_NEWNS; mount.h MS_*; mount_setattr(2) is system call 442 on x86_64 and aarch64 (Linux 5.12).
CLONE_NEWNS = 0x20000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT = 1, 2, 4, 8, 32
MS_NOSYMFOLLOW, MS_NOATIME, MS_NODIRATIME, MS_BIND, MS_REC, MS_PRIVATE = 256, 1 << 10, 1 << 11, 1 << 12, 1 << 14, 1 << 18
MS_RELATIME, MS_STRICTATIME = 1 << 21, 1 << 24
MOUNT_SETATTR, AT_FDCWD, AT_RECURSIVE, MOUNT_ATTR_RDONLY = 442, -100, 0x8000, 1
MAKE_CHAR, MAKE_BLOCK = 1 << 6, 1 << 11  # Landlock's rights to make character and block devices: checked everywhere, granted nowhere
SYSTEM_PLACES = ("sys", "sysrq-trigger", "irq", "bus", "fs")  # below each proc mount: those that container runtimes (runc) make read-only
ROW_SHM = "confine.py-row"  # the source name of a command's own /dev/shm, by which a confine.py started inside knows it
LAID, NESTED, ALREADY, UNMADE = "laid", "nested", "already", "unmade"  # what root_walls() found or did
SETUP_WORDS = {LAID: "a mount namespace of its own", NESTED: "no setup of its own, since it already runs inside a confined command",
               ALREADY: "no setup of its own, since /proc's system settings are already read-only and there is no /dev/shm",
               UNMADE: "no mount namespace of its own, since the machine refused one"}  # fmt: skip


def read_mountinfo():
    with open("/proc/self/mountinfo", errors="surrogateescape") as f:  # a mount point need not be UTF-8
        return f.read()


def mount_ns():
    return os.readlink("/proc/self/ns/mnt")


def mounts(text):
    """Parses /proc/self/mountinfo into a list, in the file's order, of dicts: id, parent, root (the folder of its file
    system that it shows), point (the mount point),
    options (the set of per-mount options, field 6), tags (the optional fields, such as shared:1), type and source.
    Raises ValueError on a line that it cannot read."""
    found = []
    for line in text.split("\n"):  # not splitlines(), which also breaks at characters that a mount point may hold unescaped
        if not line:
            continue
        fields = line.split(" ")
        try:
            cut = fields.index("-", 6)
            found.append({"id": int(fields[0]), "parent": int(fields[1]), "root": unescape(fields[3]), "point": unescape(fields[4]), "options": set(fields[5].split(",")),
                          "tags": fields[6:cut], "type": fields[cut + 1], "source": unescape(fields[cut + 2])})  # fmt: skip
        except (ValueError, IndexError):
            raise ValueError("a line of /proc/self/mountinfo could not be read: %r" % line)
    return found


def unescape(field):
    """A mountinfo field with its escapes undone (the kernel writes space, tab, newline and backslash as \\ and three octal digits)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def below(path, top):
    """Whether path is top or lies below it."""
    return path == top or path.startswith(top.rstrip("/") + "/")


def reachable(table):
    """The mounts of table that a path can reach. A mount is hidden by another mount of the same parent at a folder above
    its point, or by a mount at its own point that comes later in mountinfo (so the host's binfmt_misc stays listed below a
    bind of /proc/sys, but cannot be reached); a mount with a child at its own point is covered by that child. Where the
    order of lines misleads, read_only_here() is the second reading."""
    ids = {m["id"] for m in table}
    kids = {}
    for at, m in enumerate(table):
        kids.setdefault(m["parent"] if m["parent"] in ids and m["parent"] != m["id"] else None, []).append((at, m))

    def visible(children):
        points, shown, later = {m["point"] for _, m in children}, [], set()
        for _, m in reversed(children):
            point, above = m["point"], set()
            while point not in ("/", ""):
                point = os.path.dirname(point)
                above.add(point)
            if not above & points and m["point"] not in later:
                shown.append(m)
            later.add(m["point"])
        return shown[::-1]

    found, todo, seen = [], visible(kids.get(None, [])), set()
    while todo:
        m = todo.pop()
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        inner = visible(kids.get(m["id"], []))
        if not any(c["point"] == m["point"] for c in inner):
            found.append(m)
        todo += inner
    return found


def writable_places(table):
    """The system places (SYSTEM_PLACES below each whole proc mount that a path reaches; not below a bind of a part of
    one, such as /proc/sys) where the mount that holds the place, or a reachable mount below it, lacks the per-mount
    option ro, or that statvfs() finds writable. [] if every one is read-only. Touches no path below a place, so that
    no automount there (binfmt_misc) is set off."""
    seen = reachable(table)
    places = sorted({os.path.join(m["point"], name) for m in seen if m["type"] == "proc" and m["root"] == "/" for name in SYSTEM_PLACES
                     if os.path.lexists(os.path.join(m["point"], name))})  # fmt: skip
    bad = []
    for place in places:
        holder = max((m for m in seen if below(place, m["point"])), key=lambda m: len(m["point"]), default=None)
        if (holder is None or any("ro" not in m["options"] for m in [holder] + [m for m in seen if below(m["point"], place)])
                or not read_only_here(place)):  # fmt: skip
            bad.append(place)
    return bad


def read_only_here(place):
    """Whether the mount that place reaches is read-only, by statvfs(). This is checked beside mountinfo because the
    order of mountinfo's lines need not be the order in which mounts cover each other. statvfs() on these five places
    sets off no automount."""
    try:
        return bool(os.statvfs(place).f_flag & os.ST_RDONLY)
    except OSError:
        return False


def shm_is_rows(table):
    """Whether the mount that a path to /dev/shm reaches is a command's own, laid by an outer confine.py (a tmpfs named
    ROW_SHM, at /dev/shm itself)."""
    shm = os.path.realpath("/dev/shm")
    holder = max((m for m in reachable(table) if below(shm, m["point"])), key=lambda m: len(m["point"]), default=None)
    return bool(holder) and holder["point"] == shm and holder["type"] == "tmpfs" and holder["source"] == ROW_SHM


def root_walls(libc):
    """As root: gives this process its own private mount namespace, with /proc's system places read-only and an empty
    tmpfs on /dev/shm, and checks the result. Returns (kind, why, writable): NESTED or ALREADY when nothing is needed,
    LAID when done, UNMADE when the machine refused the namespace (writable: places still writable), or None when
    nothing may run; why says what failed."""
    try:
        table = mounts(read_mountinfo())
        shm = os.path.realpath("/dev/shm")
        if not writable_places(table) and (shm_is_rows(table) or not os.path.isdir(shm)):
            return (NESTED if os.path.isdir(shm) else ALREADY), "", []
        before = mount_ns()
    except (OSError, ValueError) as fault:
        return None, "this process's mounts could not be read (%s)" % fault, []

    if libc.unshare(CLONE_NEWNS) != 0:  # only a refusal of this first step still lets the command run; --level reports it
        fault = ctypes.get_errno()
        why = "unshare: " + os.strerror(fault)
        if fault not in (errno.EPERM, errno.EACCES, errno.ENOSYS):  # no right to it, or a filter; anything else is unexpected
            return None, why, []
        try:
            return UNMADE, why, writable_places(mounts(read_mountinfo()))
        except (OSError, ValueError) as seen:
            return None, "%s; and this process's mounts could not be read (%s)" % (why, seen), []

    def failed(step, hint=""):  # any later failure, refused or not, runs nothing
        return None, "%s: %s%s" % (step, os.strerror(ctypes.get_errno()), hint), []

    try:
        if mount_ns() == before:
            return None, "unshare reported success, but the mount namespace did not change", []
        if libc.mount(None, b"/", None, ctypes.c_ulong(MS_REC | MS_PRIVATE), None) != 0:
            return failed("making the mounts private", "; in a chroot whose / is not a mount point?" if ctypes.get_errno() == errno.EINVAL else "")
        table = mounts(read_mountinfo())
        if any(tag.startswith("shared:") for m in table for tag in m["tags"]):
            return None, "a mount still shares its changes with the machine after the mounts were made private", []
        for place in writable_places(table):
            said = read_only(libc, place)
            if said:
                return failed(said)
            if not read_only_here(place):
                return None, "%s did not become read-only" % place, []
        if os.path.isdir(shm):
            about = os.statvfs(shm)
            flags = MS_NOSUID | MS_NODEV | (MS_NOEXEC if about.f_flag & os.ST_NOEXEC else 0)
            holder = max((m for m in reachable(table) if below(shm, m["point"])), key=lambda m: len(m["point"]), default=None)
            # as large as the machine's /dev/shm where that is a tmpfs mounted at /dev/shm with a size limit; otherwise tmpfs's
            # default, half of the memory
            sized = holder and holder["point"] == shm and holder["type"] == "tmpfs" and about.f_blocks > 0
            data = b"mode=1777" + (b",size=%d" % (about.f_blocks * about.f_frsize) if sized else b"")
            if libc.mount(ROW_SHM.encode(), os.fsencode(shm), b"tmpfs", ctypes.c_ulong(flags), data) != 0:
                return failed("mounting a /dev/shm of its own")
        table = mounts(read_mountinfo())
        left = writable_places(table)
        if left:
            return None, "%s stayed writable" % ", ".join(left), []
        if os.path.isdir(shm) and not shm_is_rows(table):
            return None, "after mounting a new /dev/shm, /dev/shm is still another file system", []
    except (OSError, ValueError) as fault:
        return None, "this process's mounts, or /dev/shm, could not be read (%s)" % fault, []
    return LAID, "", []


def read_only(libc, place):
    """Binds place over itself and makes it, and every mount below it, read-only. Returns None, or the step that failed
    in words, with errno set to that step's error (EPERM where an automount stops the remount)."""
    path = os.fsencode(place)
    if libc.mount(path, path, None, ctypes.c_ulong(MS_BIND | MS_REC), None) != 0:
        return "binding %s over itself" % place
    attr = ctypes.create_string_buffer(struct.pack("=QQQQ", MOUNT_ATTR_RDONLY, 0, 0, 0), 32)  # struct mount_attr: set, clear, propagation, userns_fd
    if libc.syscall(MOUNT_SETATTR, ctypes.c_int(AT_FDCWD), path, ctypes.c_uint(AT_RECURSIVE), attr, ctypes.c_size_t(32)) == 0:
        return None
    if ctypes.get_errno() not in (errno.ENOSYS, errno.EPERM):
        return "making %s read-only" % place
    # Linux before 5.12, or a filter that refuses the call: each mount at or below the place is remounted read-only,
    # keeping its other per-mount options. A remount would set off an automount there, so where there is one, the place is
    # refused instead.
    inner = sorted((m for m in reachable(mounts(read_mountinfo())) if below(m["point"], place)), key=lambda m: len(m["point"]))
    if any(m["type"] == "autofs" for m in inner):
        ctypes.set_errno(errno.EPERM)  # refused here, before anything is mounted below the place
        return "making %s read-only (mount_setattr is refused or missing, and remounting would set off the automount below it)" % place
    keep = {"nosuid": MS_NOSUID, "nodev": MS_NODEV, "noexec": MS_NOEXEC, "noatime": MS_NOATIME, "nodiratime": MS_NODIRATIME,
            "relatime": MS_RELATIME, "strictatime": MS_STRICTATIME, "nosymfollow": MS_NOSYMFOLLOW}  # fmt: skip
    for m in inner:
        flags = MS_REMOUNT | MS_BIND | MS_RDONLY | sum(bit for name, bit in keep.items() if name in m["options"])
        if libc.mount(None, os.fsencode(m["point"]), None, ctypes.c_ulong(flags), None) != 0:
            return "making %s read-only" % m["point"]
    return None


_tried = []  # root_tried()'s one answer


def root_tried():
    """As root: what root_walls() returns, tried in a forked child that then exits, so that this process's own mounts stay
    as they are. Computed once per process."""
    if not _tried:
        import json

        read, write = os.pipe()
        child = os.fork()
        if child == 0:
            try:
                os.close(read)
                try:
                    said = root_walls(ctypes.CDLL(None, use_errno=True))
                except Exception as fault:  # any fault is a refusal, never a silence
                    said = (None, "the trial of root's setup failed (%r)" % fault, [])
                os.write(write, json.dumps(said).encode())
            finally:
                os._exit(0)
        os.close(write)
        with os.fdopen(read, "rb") as f:
            text = f.read()
        try:
            os.waitpid(child, 0)
        except ChildProcessError:  # SIGCHLD ignored by whoever started this process: the pipe has said all
            pass
        try:
            _tried.append(tuple(json.loads(text)))
        except ValueError:
            _tried.append((None, "the trial of root's setup, in a child process, ended without an answer", []))
    return _tried[0]


def rows_as_root():
    """As root: what root_walls() reports for the commands that confine.py will start (root_tried()). None for any other
    account."""
    return root_tried() if os.geteuid() == 0 else None


def as_root(kind, why, writable):
    """The words that the level gets for root, or None if nothing may run (root_refusal(why) then says why)."""
    shm = os.path.isdir("/dev/shm")
    if kind == LAID:
        return "; as root: each command gets /proc's system settings (/proc/sys and four more places) read-only" + (
            " and an empty /dev/shm of its own" if shm else "")
    if kind == NESTED:
        return "; as root: inside a command that confine.py already confined as root, so nothing more is set up"
    if kind == ALREADY:
        return "; as root: /proc's system settings are read-only here already, and there is no /dev/shm"
    shared = "; /dev/shm is shared with whatever else runs here, so a command can remove, overwrite or empty any account's files there" if shm else ""
    if kind == UNMADE and not writable:
        return "; as root: the machine refused each command a mount namespace of its own (%s): /proc's system settings are read-only here already%s" % (why, shared)
    if kind == UNMADE:
        return ("; as root: the machine refused each command a mount namespace of its own (%s): %s %s writable here, so commands that use "
                "the GPU (--gpu; under run_suite.py, the GPU rows) are refused%s" % (why, ", ".join(writable), "is" if len(writable) == 1 else "are", shared))  # fmt: skip
    return None


def root_refusal(why):
    return ("as root, each command must get a mount namespace of its own, with /proc's system settings read-only and, if there is a "
            "/dev/shm, an empty one. That failed at this step: %s" % why)  # fmt: skip


def level_here(probe_root=True):
    """Returns level() for this kernel, or (None, reason, 0) if the kernel cannot confine signals
    and ARCH has no entry for this machine. As root, and with probe_root, the words also say what root's commands get
    (as_root()), or it returns (None, reason, 0) if they cannot be confined."""
    words_of, handled, scoped = level(kernel_abi())
    if words_of and not scoped and signals_filter() is None:
        return None, "this kernel does not confine signals, and there is no signals' filter for a %s machine" % platform.machine(), 0
    if words_of and probe_root and os.geteuid() == 0:
        kind, why, writable = root_tried()
        more = as_root(kind, why, writable)
        if more is None:
            return None, root_refusal(why), 0
        words_of += more
    return words_of, handled, scoped


def kernel_abi():
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(CREATE, None, 0, 1)  # LANDLOCK_CREATE_RULESET_VERSION
    return abi if abi > 0 else 0


NOT_RUN = 125


def refuse(why):
    print("confine.py: NOT CONFINED, the command is not run: " + why, file=sys.stderr, flush=True)
    sys.exit(NOT_RUN)


def main(words):
    if words == ["--level"]:
        words_of, why, _ = level_here()
        print(words_of or "NOT CONFINED: " + why)
        return 0 if words_of else NOT_RUN
    folders = []
    while words and not words[0].startswith("-") and not folders or len(words) > 2 and words[0] == "--also":
        if words[0] == "--also":
            words = words[1:]
        folders.append(words[0])
        words = words[1:]
    gpu = words[:1] == ["--gpu"]
    if gpu:
        words = words[1:]
    if not folders or not words or words[0] in ("--also", "--gpu") or not all(os.path.isdir(f) for f in folders):
        print(__doc__.split("\n")[3], file=sys.stderr)
        return 2
    words_of, handled, scoped = level_here(probe_root=False)
    if not words_of:
        refuse(handled)
    libc = ctypes.CDLL(None, use_errno=True)
    promised = os.environ.pop("CONFINE_AS_ROOT", None)  # set by run_suite.py: root's setup as its first line reported it; not for the command
    tell_shared = False
    if os.geteuid() == 0:  # before the Landlock rules, so that the rule for /dev/shm applies to the command's new tmpfs
        in_shm = [f for f in folders if below(os.path.realpath(f), os.path.realpath("/dev/shm"))]
        kind, why, writable = root_walls(libc)
        if kind is None:
            refuse(root_refusal(why) + (". When the run started, run_suite.py found that each row would get %s; for this row the machine answered "
                                        "otherwise. Start the run again; if it happens again, report it with run_suite.py's first line" % SETUP_WORDS.get(promised, promised) if promised else
                                        ". Run it as an account other than root, or, to run it without the sandbox on purpose, run it without "
                                        "confine.py (under run_suite.py: give --unconfined)"))  # fmt: skip
        if promised and kind != promised:
            refuse("as root, run_suite.py found at its start that each row would get %s, but this row gets %s%s. Start the run again; "
                   "if this happens again, this row sees the machine differently from run_suite.py's check at the start: report it with "
                   "run_suite.py's first line" % (SETUP_WORDS.get(promised, promised), SETUP_WORDS.get(kind, kind), " (the machine said: %s)" % why if why else ""))  # fmt: skip
        if kind == LAID and in_shm:
            refuse("as root, the command gets an empty /dev/shm of its own, which hides %s; give a folder outside /dev/shm" % ", ".join(in_shm))
        if kind == UNMADE and gpu and writable:
            refuse("as root, with --gpu: the machine refused this command a mount namespace of its own (%s), so /proc's system settings in "
                   "%s stay writable, and --gpu would let the command change them. Run it as an account other than root, or, to run it without "
                   "the sandbox on purpose, run it without confine.py" % (why, ", ".join(writable)))  # fmt: skip
        # only when started by hand: run_suite.py prints its own line saying this
        tell_shared = kind == UNMADE and not promised and os.path.isdir("/dev/shm")

    # struct landlock_ruleset_attr: handled_access_fs, handled_access_net, scoped
    class Ruleset(ctypes.Structure):
        _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]

    # The calls below take the same steps as the Linux kernel's sample samples/landlock/sandboxer.c
    # (BSD-3-Clause).
    attr = Ruleset(handled, 0, scoped)
    ruleset = libc.syscall(CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        refuse("the kernel refused the rules (%s)" % os.strerror(ctypes.get_errno()))
    places = [(os.path.realpath(f), handled & ~(MAKE_CHAR | MAKE_BLOCK)) for f in folders]
    terminals = ["/dev/ptmx", "/dev/pts"] if os.geteuid() != 0 else []  # as root, /dev/pts would hold every account's terminals
    places += [(p, FILE_RIGHTS & handled) for p in ["/dev/null", "/dev/zero", "/dev/full"] + terminals if os.path.exists(p)]
    if gpu:  # /proc: CUDA names its threads by writing /proc/self/task/<id>/comm
        places += [(p, FILE_RIGHTS & handled) for p in sorted(glob.glob("/dev/nvidia*")) + ["/proc"]]
    if os.path.isdir("/dev/shm"):  # no environment variable moves multiprocessing's semaphores
        places.append(("/dev/shm", (FILE_RIGHTS | REMOVE_FILE | (1 << 8)) & handled))  # 1 << 8: MAKE_REG; no folders
    for path, allowed in places:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError as fault:
            refuse("%s cannot be opened for its rule (%s)" % (path, fault.strerror))
        if not os.path.isdir(path):
            allowed &= FILE_RIGHTS  # a file takes only a file's rights
        rule = ctypes.create_string_buffer(struct.pack("=Qi", allowed, fd))  # struct landlock_path_beneath_attr, packed
        if libc.syscall(ADD_RULE, ruleset, 1, rule, 0) < 0:
            refuse("the rule for %s was refused (%s)" % (path, os.strerror(ctypes.get_errno())))
        os.close(fd)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 or libc.syscall(RESTRICT, ruleset, 0) < 0:
        refuse("the confinement could not be laid (%s)" % os.strerror(ctypes.get_errno()))
    os.close(ruleset)
    if not scoped or os.environ.get("CONFINE_SIGNALS_FILTER") == "always":  # level `FILES`; "always" is for tests
        lay_signals_filter(libc)
    os.environ["TMPDIR"] = os.path.join(os.path.realpath(folders[0]), "tmp")
    try:
        try:  # created with mode 0700; an existing one must be this account's directory, not a link, not writable by group or others
            os.mkdir(os.environ["TMPDIR"], 0o700)
        except FileExistsError:
            about = os.lstat(os.environ["TMPDIR"])
            if not stat.S_ISDIR(about.st_mode) or about.st_uid != os.geteuid():
                raise OSError(errno.EEXIST, "its TMPDIR exists, but is not a directory of this account's own (give confine.py another "
                              "folder)", os.environ["TMPDIR"])  # fmt: skip
            if about.st_mode & 0o022:
                raise OSError(errno.EEXIST, "its TMPDIR is this account's own, but group or others can write in it (give confine.py "
                              "another folder%s)" % ("" if os.geteuid() == 0  # the advice not for root: root owns shared folders such as /tmp, and chmod go-w on one would shut every other account out
                                                     else "; or, if nothing of another's can be in it, chmod go-w it"),
                              os.environ["TMPDIR"])  # fmt: skip
        if tell_shared and sys.stderr is not None:  # last, so that only a failed exec (status 127) can follow it; best effort: never stop the command
            try:  # os.write, not print, which falls back to standard output where there is no standard error; SIGPIPE and SIGXFSZ are still ignored (reset below), so a pipe whose reader has gone loses the notice, not the command
                ready = select.poll()  # write only if standard error can take it now (a ready pipe has room for at least PIPE_BUF bytes, more than this line); not O_NONBLOCK, which would change standard error for every process sharing it
                ready.register(sys.stderr.fileno(), select.POLLOUT)
                if any(events & select.POLLOUT for _, events in ready.poll(0)):
                    os.write(sys.stderr.fileno(), b"confine.py: running as root, and the machine refused this command a mount namespace of its own, "
                             b"so it shares /dev/shm with whatever else runs here and can remove, overwrite or empty any account's files there; "
                             b"run it as another account to avoid this\n")  # fmt: skip
            except (OSError, ValueError):  # closed, a device or disk that refuses the write, or a pipe whose reader has gone
                pass
        for ignored in (signal.SIGPIPE, signal.SIGXFSZ):  # exec would keep them ignored, and bash cannot reset what was ignored at its start
            signal.signal(ignored, signal.SIG_DFL)
        os.execvp(words[0], words)
    except OSError as fault:  # already confined, so nothing ran unconfined; no traceback
        print("confine.py: confined, but the command could not be started: %s (%s)" % (fault.strerror, fault.filename or words[0]), file=sys.stderr, flush=True)
        return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
