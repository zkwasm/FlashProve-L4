#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Checks that a caller of cp_verify that is killed with SIGKILL leaves no process running.

    prover/tests/gate_children.py <gate_children_parent> <work directory>
    prover/tests/gate_children.py --service <name> <seconds> <command> [<argument> ...]

cp_commit checks its key file and leaf files: judge_says() (prover/common/judge.h) runs cp_verify
(`judge` in identifiers), which forks a child for each of its stages. A killed cp_commit cannot end
that run, so judge_says() starts it under a supervisor process, which kills the run when its caller
dies. The supervisor starts only the cp_verify in its own program's directory. cp_commit calls that
function through a wrapper of its own (judge_says() in prover/hash_commitment/hf_commit_making.h):
the cases of as_cp_commit().
First form: the cases of first_form(), on a small parent program (prover/tests/unit.sh build
tests/gate_children_parent.cpp <directory>) and a shell script in place of cp_verify. The test creates
<work directory>, which must not exist, and any missing directory above it. It writes nothing else
outside <work directory> and deletes nothing. No GPU; about a minute. Second form: second_form().
Linux only, with a /proc that shows all processes (no hidepid). Prints
ALL AS EXPECTED and exits with code 0; 1: a case failed, or SIGTERM or SIGHUP came; 2: wrong usage.

The test sends a signal only to a process that it has started itself (stop()). What that process
leaves running is only searched for in /proc (in_session()), never signalled: the script's processes
exit after a few seconds. In one case the script kills itself.

Not covered: cp_commit itself and, in the first form, cp_verify;
judge_says() under a supervisor without a Running list; a calling thread that exits while its process lives on; a supervisor
that is killed itself; a process stuck in a system call; the supervisor's PR_SET_DUMPABLE (no case fails without it);
a cp_verify that is removed while its caller runs; a symbolic link in the path given to the supervisor.
"""
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

WITHIN = 2.0  # seconds
LIMIT_MS = 60000  # the parent's time limit where a case does not need another
EXIT_CODES = (0, 1, 2, 3, 4, 5, 10, 14, 15, 77, 255)
DELAYS = (0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
SUPERVISE = "--supervise-cp_verify"  # judge::SUPERVISE
OTHER = "another_program"  # a second copy of the script, which no supervisor may start
# The script that is run in place of cp_verify. Arguments:
# sleeps <seconds before the children> <the children's seconds> <seconds after them> | exits <code> |
# kills-itself <the children's seconds> | reports. With "sleeps" it is a main process and three
# sleeping children, and it marks each step for the test by creating an empty file next to itself.
STAND_IN = r"""#!/bin/sh
here=${0%/*}
case $1 in
sleeps)
  : >"$here/started"
  /bin/sleep "$2"
  /bin/sleep "$3" &
  /bin/sleep "$3" &
  /bin/sleep "$3" &
  : >"$here/children-started"
  wait
  : >"$here/children-exited"
  /bin/sleep "$4"
  exit 0
  ;;
exits) exit "$2" ;;
kills-itself)
  /bin/sleep "$2" &
  /bin/sleep "$2" &
  kill -9 $$
  ;;
reports)
  for f in /proc/$$/fd/*; do echo "descriptor ${f##*/}"; done
  echo "inputs: $(/bin/cat /dev/stdin) $(/bin/cat /dev/fd/3) $(/bin/cat /dev/fd/4)"
  export -p
  while read -r a b; do
    case $a in SigBlk: | SigIgn:) echo "$a $b" ;; esac
  done </proc/$$/status
  exit 0
  ;;
esac
exit 9
"""
counts = {True: 0, False: 0}


def report(good, text, detail=""):
    print(("as expected   " if good else "NOT AS EXPECTED   ") + text, flush=True)
    if detail:
        print("              (%s)" % detail, flush=True)
    counts[bool(good)] += 1


def in_session(session):
    """[(process ID, name)] of the processes of this session that have not exited. Only reads.

    The test starts each parent in a new session. A process stays in its parent's session whichever
    process group it joins, and after its parent has died, so the session contains the parent and
    everything that it has started, and once it is empty it stays empty. Not seen: a process that
    starts a new session. A zombie counts as exited."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry, "rb") as f:
                stat = f.read().decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError):  # it has exited since the listing
            continue
        # proc(5): process ID, (name), state, parent, group, session, ... The name may contain spaces and parentheses.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 3 and int(fields[3]) == session and fields[0] not in ("Z", "X"):
            found.append((int(entry), stat[stat.find("(") + 1 : stat.rfind(")")]))
    return sorted(found)


def names(processes):
    return " ".join(name for _, name in processes) or "none"


def command_line(process_id):
    """The argument vector of a process; [] if it has exited."""
    try:
        with open("/proc/%d/cmdline" % process_id, "rb") as f:
            return f.read().decode("utf-8", "replace").split("\0")[:-1]
    except (FileNotFoundError, ProcessLookupError):
        return []


def start(arguments, blocked=False, **more):
    """Starts a process in a new session, whose ID is its process ID. blocked: it inherits SIGTERM and SIGALRM blocked."""
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGALRM} if blocked else set())
    try:
        parent = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, start_new_session=True, **more)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    if parent.pid <= 1:
        sys.exit("the started process has the process ID %d" % parent.pid)
    return parent


def stop(parent):
    """Sends SIGKILL to a process from start(), through its Popen object, and waits for it. Returns the time of the signal;
    or None, and sends nothing, if the process has exited already. main() sets SIGCHLD to its default, so the process ID
    stays the child's until poll() or wait() has seen the exit."""
    if parent.poll() is not None:
        return None
    sent = time.monotonic()
    parent.kill()
    parent.wait()
    return sent


def empty_after(session, seconds, begun=None):
    """(the seconds after which the session was empty, []) or, after `seconds`, (None, the processes
    found last); counted from `begun` or from now. One pass over /proc is not atomic: a process that
    forks and exits during it can hide its child. So two passes in a row must find nothing."""
    begun = begun or time.monotonic()
    while True:
        there = in_session(session) or in_session(session)
        now = time.monotonic() - begun
        if not there:
            return now, []
        if now >= seconds:
            return None, there
        time.sleep(0.01)


def new_stand_in(program, work, case):
    """Creates a directory for one case with a copy of the parent program and the script under two
    names, cp_verify and another_program: the supervisor looks for cp_verify in its program's
    directory. Returns the paths of the parent's copy and of cp_verify."""
    directory = os.path.join(work, case)
    os.makedirs(directory)
    for name in ("cp_verify", OTHER):
        with open(os.path.join(directory, name), "w", encoding="ascii") as f:
            f.write(STAND_IN)
        os.chmod(os.path.join(directory, name), 0o700)
    return shutil.copy(program, directory), os.path.join(directory, "cp_verify")


def parent_killed(program, work, case, title, mode, sleeps, step=None, delay=0.0, control=False, blocked=False, arguments=False, limit_ms=LIMIT_MS):
    """One case: kills the parent once the script has created the file `step`, or `delay` seconds
    after the parent has printed "calls". Two seconds after the signal no process may be left;
    in the control, the script's four must be. arguments: also checks the supervisor's arguments;
    its value is the time limit expected among them, in seconds."""
    program, path = new_stand_in(program, work, case)
    script_arguments = [path, "sleeps"] + [str(s) for s in sleeps]
    parent = start([program, mode, str(limit_ms)] + script_arguments, blocked, stdout=subprocess.PIPE)
    try:
        printed = parent.stdout.readline() if select.select([parent.stdout], [], [], 20)[0] else b""
        waited = time.monotonic()
        reached = lambda: not step or os.path.exists(os.path.join(os.path.dirname(path), step))
        while not reached() and parent.poll() is None and time.monotonic() - waited < 20:
            time.sleep(0.002)
        time.sleep(delay)
        before = in_session(parent.pid) if step else []  # a pass over /proc takes milliseconds, which the short delays cannot spare
        vectors = [command_line(p) for p, _ in before] if arguments else []
        if printed != b"calls\n" or not reached() or not (sent := stop(parent)):
            report(False, "%s: the parent or the script did not get that far" % title, "the parent printed %r; the file %s exists: %s; the parent's exit code: %s" % (printed, step, reached(), parent.poll()))
            return
        died = time.monotonic() - sent
        after, left = empty_after(parent.pid, WITHIN, sent)
        if left or control:
            report(control and len(left) == 4, "%s: %d processes still run two seconds later (%s)" % (title, len(left), names(left)))
        else:
            report(True, "%s: no process is left" % title, ("%d processes at the signal: %s; " % (len(before), names(before)) if step else "") + "the parent was gone %.3f s after the signal, all %.3f s after it" % (died, after))
        if arguments:
            # The caller's process ID; the caller's limit in seconds plus 10; three inputs, so descriptors 0 to 4 are inherited.
            wanted = ["/proc/self/exe", SUPERVISE, str(parent.pid), arguments, "5", path] + script_arguments
            report(wanted in vectors, "%s: judge_says() gives the supervisor the expected arguments" % title, "" if wanted in vectors else "expected %s; found %s" % (wanted, vectors))
    finally:
        stop(parent)
        parent.stdout.close()
        _, left = empty_after(parent.pid, sum(sleeps) + 10)  # whatever still runs exits without help
        if left:
            report(False, "%s: still running long after the script's sleeps: %s" % (title, left))


def output(program, work, case, mode, limit_ms, arguments, blocked, named="cp_verify"):
    """The output of a parent that is not killed, or a sentence on what went wrong. No process may be left after it.
    named: the file in the case's directory that the parent is told to run."""
    program, path = new_stand_in(program, work, case + "." + mode)
    parent = start([program, mode, str(limit_ms), os.path.join(os.path.dirname(path), named)] + arguments, blocked, stdout=subprocess.PIPE)
    try:
        printed = parent.communicate(timeout=60)[0].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        printed = "no output within a minute"
    finally:
        stop(parent)
    _, left = empty_after(parent.pid, WITHIN)
    if left:
        empty_after(parent.pid, 10)  # they exit without help
    return printed if not left else "%s; still running: %s" % (printed, names(left))


def both_blocked(reported):
    """Whether the script's line SigBlk, a hexadecimal mask in which bit n - 1 is signal n, has SIGTERM and SIGALRM."""
    both = 1 << (signal.SIGTERM - 1) | 1 << (signal.SIGALRM - 1)
    return any(line.startswith("SigBlk: ") and int(line[8:], 16) & both == both for line in reported.split("\n"))


def supervisor_by_the_test(program, work, case, title, caller, seconds, check, leads=True, bystander=False, named="cp_verify"):
    """One case with the supervisor started by this test and not by judge_says().

    check(parent, reached, title) returns report()'s arguments; reached(step): whether the script has created that file.
    Every case goes through a shell that leads a new session, so a supervisor that wrongly signalled its group would
    kill that shell and not this test. leads: the shell replaces itself by the supervisor (exec), which therefore leads
    the group and has this test as its parent. Otherwise the supervisor is the shell's child and does not lead the group.
    caller: the caller's process ID for the supervisor; "": the shell's ($$), which with leads is the supervisor's own and
    otherwise its parent's. bystander: the shell first starts a sleep of three seconds, which stays in its group.
    seconds: the supervisor's time limit; "0": none.
    named: the path of the program for the supervisor to start, relative to the case's directory."""
    program, path = new_stand_in(program, work, case)
    script = 'program=$0 option=$1 caller=${2:-$$}; shift 2; %s"$program" "$option" "$caller" "$@"; exit $?' % ("exec " if leads else "")
    if bystander:
        script = "/bin/sleep 3 & " + script
    parent = start(["/bin/sh", "-c", script, program, SUPERVISE, caller, seconds, "3", os.path.dirname(path) + "/" + named, path, "sleeps", "0", "6", "0"], stdout=subprocess.DEVNULL)
    try:
        report(*check(parent, lambda step: os.path.exists(os.path.join(os.path.dirname(path), step)), "the supervisor started by the test: " + title))
    finally:
        stop(parent)
        _, left = empty_after(parent.pid, 16)
        if left:
            report(False, "the supervisor started by the test: %s: still running long after the script's sleeps: %s" % (title, left))


def code_within(parent, seconds):
    """The exit code (below 0: minus the signal that killed it), or None if it still runs after that time."""
    try:
        return parent.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return None


def the_supervisor(program, work):
    me = str(os.getpid())

    def at_its_limit(parent, reached, title):
        begun = time.monotonic()
        code, took, left = code_within(parent, 5), time.monotonic() - begun, empty_after(parent.pid, WITHIN)[1]
        seen = "exit code %s after %.1f s; children started: %s; still running: %s" % (code, took, reached("children-started"), names(left))
        return code == -9 and 0.5 <= took <= 3 and reached("children-started") and not left, title, seen

    def without_a_limit(parent, reached, title):
        time.sleep(3)
        there, code = in_session(parent.pid), code_within(parent, 10)
        seen = "after three seconds: %s; exit code %s; children exited: %s" % (names(there), code, reached("children-exited"))
        return len(there) == 5 and code == 0 and reached("children-exited"), title, seen

    def at_once(wanted, running):
        def check(parent, reached, title):
            code, there = code_within(parent, WITHIN), in_session(parent.pid)
            return code == wanted and not reached("started") and len(there) == running, title, "exit code %s; script started: %s; still running: %s" % (code, reached("started"), names(there))

        return check

    supervisor_by_the_test(program, work, "limit", "a caller that stays alive and does nothing: the group is killed at the supervisor's limit of one second", me, "1", at_its_limit)
    supervisor_by_the_test(program, work, "no-limit", "control: without a limit all five processes run after three seconds, and the script's exit code arrives", me, "0", without_a_limit)
    # A caller that has died before the supervisor asks for the parent-death signal: the short delays of first_form() hit that
    # moment only by chance. The sleep is in the supervisor's group but was not started by it.
    supervisor_by_the_test(program, work, "other-caller", "the caller's process ID is not its parent's: nothing is started, and only it dies, by SIGKILL, at once", "", "0", at_once(-9, 1), bystander=True)
    supervisor_by_the_test(program, work, "no-leader", "not the leader of its process group: nothing is started, no signal is sent, exit code 2", "", "0", at_once(2, 0), leads=False)
    # As "no-limit" but for the program's path, and with the sleep of "other-caller" in the group.
    # The script would create its file "started" in the same directory under either name.
    supervisor_by_the_test(program, work, "other-program", "another program's path: nothing is started, no signal is sent, exit code 2", me, "0", at_once(2, 1), bystander=True, named=OTHER)
    supervisor_by_the_test(program, work, "other-spelling", "cp_verify's path in another spelling: nothing is started, no signal is sent, exit code 2", me, "0", at_once(2, 1), bystander=True, named="./cp_verify")


def first_form(program, work):
    """The control shows that the search finds what a killed parent leaves: a faulty search would
    pass every case that expects no process. The delays aim at the supervisor's first steps; a case
    also passes if its parent dies before it has started the supervisor."""
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    parent_killed(program, work, "control", "control: a parent that starts the script directly, killed while the children run", "direct", (0, 6, 0), "children-started", control=True)
    parent_killed(program, work, "before", "parent killed before the script has its three children", "supervised", (3, 3, 0), "started")
    parent_killed(program, work, "during", "parent killed while the children run", "supervised", (0, 6, 0), "children-started", arguments=str(LIMIT_MS // 1000 + 10))
    parent_killed(program, work, "after", "parent killed after the three children have exited", "supervised", (0, 0.2, 6), "children-exited")
    parent_killed(program, work, "blocked", "parent killed while the children run, SIGTERM and SIGALRM blocked", "supervised", (0, 6, 0), "children-started", blocked=True)
    for n, delay in enumerate(DELAYS):
        parent_killed(program, work, "delay-%d" % n, 'parent killed %g s after it printed "calls"' % delay, "supervised", (0, 4, 0), delay=delay)
    # A parent that is not killed prints the same with and without the supervisor. -3: judge::LATE.
    # What the script inherits holds no variable of its caller's environment: these three are set for that.
    os.environ.update(HOME="/nowhere", TMPDIR="/nowhere", LD_NOT_A_LOADERS_WORD="1")
    for title, case, limit_ms, arguments, wanted in (
        [("exit code %d" % n, "exits-%d" % n, LIMIT_MS, ["exits", str(n)], "calls\nstatus %d\n" % n) for n in EXIT_CODES]
        + [
            ("a script that exceeds the time limit", "late", 300, ["sleeps", "0", "6", "0"], "calls\nstatus -3\n"),
            ("a script that a signal kills", "signal", LIMIT_MS, ["kills-itself", "6"], "calls\nstatus -1\n"),
            ("what the script inherits", "reports", LIMIT_MS, ["reports"], None),
            ("what the script inherits, SIGTERM and SIGALRM blocked", "reports-blocked", LIMIT_MS, ["reports"], None),
        ]
    ):
        blocked = case == "reports-blocked"
        supervised, direct = (output(program, work, case, mode, limit_ms, arguments, blocked) for mode in ("supervised", "direct"))
        good = supervised == direct and (
            supervised == wanted
            if wanted
            else supervised.startswith("calls\nstatus 0\ndescriptor ")
            and "inputs: first second third\n" in supervised
            and both_blocked(supervised) == blocked
            and not re.search(r"\b(HOME|TMPDIR|LD_\w*)=", supervised)
        )
        report(good, "a parent that is not killed: %s: the expected output, supervised and direct" % title)
        if not good:
            print("supervised:\n%s\ndirect:\n%s" % (supervised, direct), flush=True)
    # The script itself would exit with code 0; 2 is the supervisor's exit code for another path.
    printed = output(program, work, "other-program-by-the-parent", "supervised", LIMIT_MS, ["exits", "0"], False, OTHER)
    report(printed == "calls\nstatus 2\n", "a parent that is told to run another program, supervised: it is not started, and judge_says() returns 2", "printed: %r" % printed)
    the_supervisor(program, work)
    as_cp_commit(program, work)


def as_cp_commit(program, work):
    """The parent calls hf::making::judge_says(), as cp_commit does for a key file or a leaf file. The script then writes
    to the parent's standard output itself, before the parent's line "status". Started directly, it has no time limit."""
    killed = lambda case, title, *more, **still_more: parent_killed(program, work, "commit-" + case, "as cp_commit: " + title, *more, **still_more)
    killed("control", "control: a parent that starts the script directly, killed while the children run", "cp_commit-direct", (0, 6, 0), "children-started", control=True)
    killed("before", "parent killed before the script has its three children", "cp_commit", (3, 3, 0), "started")
    # Time limit 0: the parent leaves cp_commit's limit as compiled, 1800 seconds; judge_says() gives the supervisor ten more.
    killed("during", "parent killed while the children run", "cp_commit", (0, 6, 0), "children-started", arguments="1810", limit_ms=0)
    killed("after", "parent killed after the three children have exited", "cp_commit", (0, 0.2, 6), "children-exited")
    killed("blocked", "parent killed while the children run, SIGTERM and SIGALRM blocked", "cp_commit", (0, 6, 0), "children-started", blocked=True)
    for n, delay in enumerate(DELAYS):
        killed("delay-%d" % n, 'parent killed %g s after it printed "calls"' % delay, "cp_commit", (0, 4, 0), delay=delay)
    for title, case, arguments, wanted in (
        [("exit code %d" % n, "exits-%d" % n, ["exits", str(n)], "calls\nstatus %d\n" % n) for n in EXIT_CODES]
        + [("a script that a signal kills", "signal", ["kills-itself", "6"], "calls\nstatus -1\n"), ("what the script inherits", "reports", ["reports"], None)]
    ):
        supervised, direct = (output(program, work, "commit-" + case, mode, LIMIT_MS, arguments, False) for mode in ("cp_commit", "cp_commit-direct"))
        good = supervised == direct and (
            supervised == wanted if wanted else supervised.startswith("calls\ndescriptor ") and "inputs: first second third\n" in supervised and supervised.endswith("\nstatus 0\n")
        )
        report(good, "as cp_commit: a parent that is not killed: %s: the expected output, supervised and direct" % title)
        if not good:
            print("supervised:\n%s\ndirect:\n%s" % (supervised, direct), flush=True)
    printed = output(program, work, "commit-late", "cp_commit", 300, ["sleeps", "0", "6", "0"], False)
    report(printed == "calls\nstatus -3\n", "as cp_commit: a script that exceeds the time limit is killed with its children, and judge_says() returns -3", "printed: %r" % printed)
    printed = output(program, work, "commit-other-program", "cp_commit", LIMIT_MS, ["exits", "0"], False, OTHER)
    report(printed == "calls\nstatus -1\n", "as cp_commit: a parent that is told to run another program: it is not started, and judge_says() returns -1", "printed: %r" % printed)


def second_form(name, seconds, command):
    """The same check around any command that starts the real cp_verify. Waits, up to an hour, until
    a process named `name`, for example cp_verify, runs under it and still does `seconds` later:
    this skips shorter runs. Then kills the command's process and searches. What still runs a minute
    later is listed and left running."""
    parent = start(command)
    named = lambda processes: {p for p, n in processes if n == name[:15] and p != parent.pid}  # the kernel keeps 15 characters of a name
    try:
        waited, before, lasting = time.monotonic(), [], set()
        while not lasting and parent.poll() is None and time.monotonic() - waited < 3600:
            time.sleep(0.02)
            first = named(in_session(parent.pid))
            if first:
                time.sleep(seconds)
                before = in_session(parent.pid)
                lasting = first & named(before)
        if not lasting or not (sent := stop(parent)):
            report(False, "no process named %s ran for %g s under the command" % (name, seconds), "running at the last search: %s" % names(before))
            return
        died = time.monotonic() - sent
        after, left = empty_after(parent.pid, WITHIN, sent)
        seen = "the command's process was gone %.3f s after the signal" % died + ("" if left else ", all %.3f s after it" % after)
        report(not left, "%d processes at the signal (%s); two seconds after the signal: %s" % (len(before), names(before), names(left)), seen)
    finally:
        stop(parent)
        _, left = empty_after(parent.pid, 60)
        if left:
            print("still running a minute later, and not signalled by this test: %s" % left, flush=True)


def main(arguments):
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # if it were inherited ignored, the kernel would reap the children itself
    for number in (signal.SIGTERM, signal.SIGHUP):  # so that a "finally" still kills the process that the test has started
        if signal.getsignal(number) != signal.SIG_IGN:  # as SIGHUP is under nohup
            signal.signal(number, lambda number, _: sys.exit("gate_children.py: ended by signal %d" % number))
    if len(arguments) >= 4 and arguments[0] == "--service" and arguments[2].replace(".", "", 1).isdigit():
        second_form(arguments[1], float(arguments[2]), arguments[3:])
    elif len(arguments) == 2 and os.access(arguments[0], os.X_OK) and os.path.isfile(arguments[0]) and not os.path.lexists(arguments[1]):
        # realpath: the supervisor builds cp_verify's path from its own program's, which /proc gives without symbolic links
        first_form(os.path.abspath(arguments[0]), os.path.realpath(arguments[1]))
    else:
        print("usage:\n%s\n<gate_children_parent> must be a program, and <work directory> must not exist" % __doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    print("cases as expected: %d; not: %d" % (counts[True], counts[False]))
    print("ALL AS EXPECTED" if not counts[False] else "NOT ALL AS EXPECTED")
    return 1 if counts[False] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
