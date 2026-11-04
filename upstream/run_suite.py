#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Runs the tests that SUITE.md lists and reports which of them passed.
This program itself removes no file. It sends signals only to the tests' own process groups, and to
a test on the GPU only while that test still waits for the GPU.

usage: run_suite.py <cpu|gpu|all|long> <new work folder> [options]
       run_suite.py --help-more        (every option, every rule, and what each cell of SUITE.md means)

What to type. From this folder (the repository's top), each time with a new work folder outside it,
in your home folder or in /tmp (its path may contain only letters, digits and / . _ -):
  python3 run_suite.py cpu ~/suite-work
      the tests that need no GPU; start here
  python3 run_suite.py gpu ~/suite-work-gpu
      the GPU programs' builds and their tests (they need nvcc 13.2.86, and one H100)
  python3 run_suite.py cpu ~/suite-work-2 --only verifier
      the tests whose names match the regular expression "verifier", and the rows that they need

What it does. SUITE.md is this folder's table of tests: one row for each test, or for each build
step that tests need. The runner (this program) runs each row's command after the rows that it
needs, and prints one line for each row as it ends: "passed"; "in part" (status 77: it ran, but its
last line says what it left out); "FAILED"; "not run" (a row that it needs did not pass, or its last
line says what is missing here, such as a Python module or g++); or "ACCEPT" (status 3: something
false was accepted, in the suite's tests by the proof verifier, cp_verify). Each row has a new
folder of its own for its files and is stopped at its time limit. The exception is a test on the
GPU: it is named and left running, and is never killed while a GPU program of its own may be
running; it is ended only if it is still waiting for the GPU. Rows run confined by Linux's Landlock,
to limit what a faulty test can damage by accident; the confinement is not a barrier against a test
written to do harm, and as root it leaves chmod, chown and touch on any file unlimited. Each row can write only in its own folder and in a few places that --help-more lists (among
them this folder's bin/ and files in /dev/shm; for tests on the GPU also /proc). As root, each row
also gets /proc's system settings read-only and a /dev/shm of its own, where the machine lets it
(not in Docker's and Kubernetes' containers by default). The first line of output says how far the kernel confines them. Where it cannot confine them enough (Linux older than 5.13, or
some containers), nothing is run (status 2) unless --unconfined is given, which runs the rows with
all the rights of your account. Mode cpu runs no row that needs the CUDA compiler or a GPU: a test
that needs such a row is left out, and a line names it; mode all runs it. Rows run side by side as
far as the job slots allow (tests on the GPU one at a time; with GPU_PREFIX set, rows of the kind
gpu-by-run run side by side; see --help-more, «gpu»): a slot stands for one core and about 2 GiB of
memory (--jobs; by default as many as the cores and the memory allow).

Where things go. The runner creates the work folder, closed to other accounts. Each test's whole
output is in <work>/screens/<row>.txt: read it for any row that did not pass. A line that begins
"LOOK" repeats an alarm that a test printed (something false may have been accepted, or a check may
not show what it should): read it, even if the test passed. The last line counts the rows by how
they ended, and the rows that were left out because what they test does not exist under this
folder's version of the file formats (the "version word"; you do not need to do anything about
them).

The work folder's place. It must be below folders that only you or root can change, such as your
home folder, or in /tmp, whose sticky bit keeps others from replacing your folder. If others could
change a folder on the way, nothing is run, and one line names that folder and says what to do (for
a folder of your own: chmod go-w). On a machine where you trust every account,
--i-trust-every-account-on-this-machine accepts such a place, though a test that starts the prover
(cp_prove) may still fail there with the prover's own message (see --help-more).

Exit status: 3 if a test found that something false was accepted, whatever else happened: keep the
work folder and this folder's bin/ as they are, and report it (SECURITY.md in this folder says how).
Otherwise 2 for a wrong call, or if nothing was run; otherwise 1 if a row failed or was not run for
a reason of the run's own (a row that it needs failed, a GPU program of an earlier test still runs
on the card, the work folder was replaced), a LOOK line was printed, a test on the GPU was left
running at its time limit, a signal stopped the run, or the work folder was replaced or removed
while the rows ran; otherwise 77 if a row did not run and said why in its own last line (one that
begins "not run", such as for a compiler that this machine lacks), or did not run only because a row
that it needs did so; otherwise 0: every chosen row passed, in part or whole.
"""

import concurrent.futures
import errno
import fcntl
import io
import os
import re
import shlex
import signal
import site
import stat
import subprocess
import sys
import threading
import time
import traceback

DETAILS = """Runs the tests in this folder that SUITE.md lists, and reports which of them passed.

usage: run_suite.py <cpu|gpu|all|long> <new work folder> [--jobs <n>] [--only <regular expression>] [--share <i>/<n>]
                    [--taken <regular expression>] [--shared-from <folder>] [--list] [--unconfined | --confined]
                    [--i-trust-every-account-on-this-machine]

Every row runs through tools/confine.py (Linux's Landlock), unless --unconfined is given: a row may write, make and remove files only below
its own folder <work>/<row> (its $W is new inside it), in the run's shared folder $S and in the tree's bin/ (the runner makes it, and $B, if they are missing),
a row that builds also in $B (the first line names it), and FILES anywhere in /dev/shm (Python's
multiprocessing needs them: a hole), and a row of a GPU kind the GPU's device files and the files of /proc (CUDA's thread
names: a hole). It may make no device file anywhere; of the inert character node 0:0 the sandbox promises nothing. Where the kernel can, it may signal only its own processes, and
elsewhere a seccomp filter refuses it a signal to -1 (as root there, any other process of the machine stays in reach). The
sandbox is there to limit what a faulty test can damage by accident. It is not a barrier against a test written to do harm,
and when the suite runs as root it cannot be one. Run as root, a test can still change the owner, mode or times of any file on
the machine (chmod, chown, touch), which the sandbox does not limit at all, and it keeps root's other powers, such as the
network and the machine itself. As root, a row may not write pseudo-terminals. As root, each row gets a
mount namespace of its own, in which /proc/sys, /proc/sysrq-trigger, /proc/irq, /proc/bus and /proc/fs are read-only and
/dev/shm is an empty folder of the row's own: the two holes above then reach neither those five places nor other accounts'
files in /dev/shm, unless a row undoes it on purpose (other files of /proc that root may write stay open: tools/confine.py,
"Not confined"). Each row's /dev/shm is the size of the machine's /dev/shm when that is a tmpfs mounted there, and otherwise
half of the memory, so rows side by side may together hold more there than the machine's one /dev/shm allowed. Where root may
not make such a namespace at all (as in Docker's and Kubernetes' containers by default), the first line says so: /dev/shm is then shared, and where those
places of /proc are writable a run that chooses a row on the GPU stops before it starts anything (status 2), unless
--unconfined is given. Where the namespace can be made but a later step fails (a chroot whose / is not a mount point, or a
container where root may make it but not mount), nothing is run as root (status 2); --unconfined runs the rows without the
sandbox, on purpose. Where each row gets a /dev/shm of its own, the runner refuses a card's lock there (GPU_LOCK, or a path
that GPU_PREFIX names, also one that does not exist yet): each row would make and lock its own copy, and the lock would not
stop two rows from using the card at the same time. It also refuses this folder, BUILD, PYTHONUSERBASE,
VIRTUAL_ENV, CONDA_PREFIX, CUDA_HOME, CUDA_MPS_PIPE_DIRECTORY, your Python user base (~/.local) or a folder of PATH, PYTHONPATH or
LD_LIBRARY_PATH in /dev/shm: the rows would not see them. Anything else that a row would read
from /dev/shm is hidden from it there too. What confine.py does not confine (a file's mode and owner, another process's limits, the network, …) is
in tools/confine.py's own usage text. GPU_PREFIX runs inside the confinement (after the card's lock, which the runner makes): a lock file that it names must exist already (the confined flock cannot make it
outside the row's own places; where each row gets a /dev/shm of its own, it must also lie outside /dev/shm), and a prefix that hands the row to another machine or process leaves that part unconfined (the first line names the prefix). A confined row's
TMPDIR, TMP and TEMP are <work>/<row>/tmp, and its CUDA_CACHE_PATH <work>/<row>/cuda-cache. The first line says the level, and so does
the last line, also when SIGTERM, SIGINT, SIGQUIT or SIGHUP ends the run once its rows have begun; where the kernel gives no Landlock, or does not confine signals and
confine.py has no filter for the machine, nothing is run (status 2) unless --unconfined is given. --unconfined runs every
row as it stands, but for its temporary files (its TMPDIR, TMP and TEMP are <work>/tmp/<row>, and its CUDA_CACHE_PATH
<work>/tmp/<row>/cuda-cache), and the first and the last line say so (the handler's line too, after a signal); --confined is the default
said aloud; both at once are refused. As root (under sudo, say), what the runner makes is root's: the work folder and, if the tree had none, bin/ with the programs built there. In a confined run (the default), where root makes bin/ in a tree that another account owns, a note says so and how to go on, and a later run that is not root, which cannot build into that bin/, prints the same note.

cpu   The rows that need no GPU: checks of the sources, unit tests built with g++, the Python tools' tests, and
      runs of cp_verify on files and altered copies of them. Up to <n> job
      slots are in use at once, and a row takes as many as its column "slots" says (all of them, if it says
      more). The column counts memory too: a slot stands for one core and about 2 GiB. The default is the
      smallest of: the cores that this program may run on, its control groups' CPU quota, and the memory (its
      control groups' limit, or the machine's) divided by 2 GiB. A row that signal 9 ended is reported with a
      note on memory, as the system ends a program that runs out of it. A row that needs, at any depth, a row of
      the kind nvcc, gpu or gpu-by-run (a CUDA build, or a world that the GPU programs make) is left out of this
      mode, in a line of its own, and counted apart in the last line; mode all runs it (mode gpu chooses no row
      of the kind cpu).
gpu   The rows that need nvcc or one H100: the GPU programs' builds, in parallel, and then their tests, one at a time
      (but see GPU_PREFIX). THE RUNNER ALONE SAYS WHICH LOCK IS THE CARD's: it puts «flock <lock>» before every command that uses the
      GPU (so the lock is held until the command has ended, and with it every process that it has left running and that has inherited the open lock file; Python's subprocess does not hand the open file on), with <lock> the file that GPU_LOCK names (one file for all runs that share a card; the runner makes it) or else
      <work>/gpu.lock (so two runs in two work folders on one card, without GPU_LOCK, hold two locks and meet on the card);
      GPU_LOCK's path may hold letters, digits and / . _ - only, and lies outside the work folder; it is not passed to the
      rows. Its folder gets the same look as the work folder's (the option takes it alike): a folder that only you or root
      can change, or one with the sticky bit such as /tmp. A lock in a folder that every account can write in (/tmp) may
      serve the runs of one account only: where the kernel protects such folders (fs.protected_regular, set by default
      under systemd), another account cannot open that file. Sharing one card's lock between accounts needs a lock file that
      each of them can open: a folder of root's that only their group can write in, with the sticky bit (mode 1770), where
      root makes the lock file first if fs.protected_regular is 2; or a group folder without the sticky bit (mode 770),
      which this runner's look takes only with --i-trust-every-account-on-this-machine. No script that a row runs takes a
      lock on the card of its own, and GPU_PREFIX must not name the lock (refused). For a row of the kind gpu the lock is
      put before the whole row. A row of the kind gpu-by-run gets it as GPU_RUN and puts it before each of its GPU programs itself, so that
      it holds no GPU while it waits or calls the verifier. If GPU_PREFIX is set, its words follow the lock's; with it set,
      rows of the kind gpu-by-run run side by side, as far as the job slots allow. Without it they run one at a time. A row's wait for
      the lock counts against its time limit, so with GPU_PREFIX or GPU_LOCK set the limit of a row on the GPU is at least
      an hour. A short row that waits longer behind long ones still fails at its limit.
all   Both.
long  The rows of the kind long alone: rows on the CPU that take long (hours), which no other mode chooses. Name them
      with --only, or run them all. The rows that they need are added, whatever their kind.

--only takes the rows in whose names the expression matches somewhere. --share i/n takes the i-th of n shares of them,
for n machines. The tests are dealt out by their minutes, the longest first, so that the shares take about equally long.
In either case the rows that a chosen row needs are added, so each machine makes the builds that its own tests need.
--taken is for a machine that already holds what some set-up rows make, for example the built GPU programs in the folder
that BUILD names. A row that another row to be run needs, and in whose name the expression matches, is not run, even if
the call chose it too. It is printed as "taken", counts as passed for the rows that need it, and is counted apart in the
last line. The rows that only taken rows need are not run either. They are not printed, and the last line counts them.
What a row leaves in S is never there, because the work folder is new, and neither is what it leaves in B unless BUILD
names a folder that outlives the run. So take only rows that build into a B that BUILD names, or whose product the rows
that need it find by a variable that you set, or in S when --shared-from lays it there.

--shared-from copies the plain files and folders below a folder of yours into S before any row starts. The folders above
it get the same look as BUILD (the option takes them alike). The folder itself and everything below it are copied,
so there a folder or file of another account, or one that its group or others can write in (with the sticky bit or
not), stops the run before any row starts (status 2), since they could change what the rows run while it is copied. A copied file
gets its source's mode without the write bits for group and others and without the set-ID bits (so at most 755), less
what the run's umask takes away. A symbolic link or any other kind of file below it stops the run before
any row starts (status 2). A line before the rows names the folder and how many files were laid.
Some rows that others need are tests themselves, so a loose expression leaves tests out. Whether what a taken row would
have made is there, and is made from these sources, is the caller's matter.
--list prints the rows and runs none. A valid call with it ends with status 0, also if there is no row to print.

The work folder must not exist yet, must be outside this folder, and its path may contain only letters, digits and
/ . _ -; the folder above it must exist. The runner makes the work folder and nothing above it. It is closed to other accounts
(mode 700), and so is every folder that the runner makes in it; the runner's umask, which the rows inherit, takes away
the write bit of group and others (a stricter one stays for them; the owner's bits are never taken away). Each row's
output goes to <work>/screens/<row>.txt, a new file, and each row gets its own new folder, <work>/<row>.
If another account may be able to change the work folder's parent, or a folder above it or above BUILD (a folder that
its group or others can write in without the sticky bit, or one that belongs to an account other than yours or root, or
one that does not exist or could not be examined; for BUILD, also a default access control list on BUILD itself),
nothing is run (status 2): that account could put another folder in its place while rows run. If that folder is yours
and you are not root, close it with chmod go-w, and a further such folder below it is named in turn (under a private
group and a umask of 002, new folders are 775); root works below a closed folder of its own. (BUILD itself must in any
case be a folder of yours that only you can write in, as below, with the option too.) A folder that only you or root can
change, such as one in your home folder, or /tmp, with its sticky bit, will do. On a machine where you trust every
account (a container of your own, for example), --i-trust-every-account-on-this-machine takes such a folder all the
same, and the first and last lines say so; rows that start a program or script carrying the same look (such as cp_prove,
or prover/examples/run_target_example.sh) may still fail there with that program's refusal, since those take no such
option: run the suite below a closed place for those. A place that does not exist or could not be examined is refused
whatever the option. A row that ends with status 77 and a last line beginning "stood aside: below an open place:" counts
as passed in part, and the closing line counts such rows apart and says where to run them (one whose last line begins
"not run" counts as not run). For the tree that the run reads and runs (the folder that holds this program), one line at
the start, on standard error, names the first such place above it (from its parent up, so that a clone of mode 775 under
a private group raises nothing), or the tree itself where every account may write in it, and says to keep it below
folders that only you or root can change; the run goes on. If at the end the path no longer names the folder that the
runner made, it says so in capitals and ends with status 1. It also looks just before each row starts (after its wait
for a job slot), and starts no row once it has seen that the path leads to another folder. What it does not see: a row
that is already running, or starting in the moment between that look and its start, goes on by the path; and a swap,
between the runner's mkdir and its first look, for another empty folder of the caller's own that is closed to others is
taken as the work folder.
BUILD, if set and not empty, must be an absolute path of letters, digits and / . _ - with no part "." or "..", at
least two levels below /, naming a folder (not a symbolic link) that you own, or root, or under sudo the account that ran
sudo, and that neither group nor others can write, or a new name inside such a folder; otherwise nothing is run.

SUITE.md's tables have one row per test or set-up step, with eight cells. Further tables, where they stand, are read as
part of the same table, in the same form: a further table is a file SUITE-<word>.md at the tree's top whose first line
is «<!-- a further table of tests for run_suite.py -->»; they are read in sorted order after SUITE.md, a row of one may
need rows of SUITE.md and of the tables read before it (never the reverse), and a name stands once in them all. A
SUITE-<word>.md without that first line is a page and is passed over, unless it holds a table of tests (a line that
begins «| name | kind |»): then nothing is run (status 2), lest its rows be left out unseen. A cell may be in backquotes, and the cells «passes»
and «command» must be; no cell contains a vertical bar:
  name     letters, digits, . _ -
  kind     cpu, nvcc (a build that needs the CUDA compiler and no GPU), gpu, gpu-by-run, or long (a row on the CPU
           that takes long, which only mode long chooses)
  needs    the rows that must have passed before, separated by commas; - for none
  word     any; or a version word (letters and digits): the row is left out unless this tree's version word is that one
  slots    how many job slots the row takes
  minutes  the time that it took on one machine. It sets the order of the rows and the time limit (five times that, at
           least ten minutes; for a row on the GPU with GPU_PREFIX or GPU_LOCK set, at least an hour)
  passes   the exit statuses that count as passed, separated by commas. After a blank, a regular expression can follow
           that the last line of output must match. More expressions can follow it, each after " ;; " (a blank, two
           semicolons, a blank): each must match some line of the output. By this folder's convention, a test ends
           with status 77 to say "some of this cannot be run on this machine". Where a row lets status 77 pass, read
           the row's last line. If that line begins with "not run", nothing of the row ran, and the row is counted as
           not run, whatever its status. Status 3 may not stand in this cell (see "Status 3" below)
  command  run by bash from this folder, with these variables set: T this folder, W the row's own folder (not made yet;
           confined, it lies inside a folder that the runner makes for the row), S <work>/shared (made; for what
           set-up rows leave for others), B the GPU programs' build folder (BUILD, or
           <work>/build, which the runner makes; refused if it is the root or holds this folder or the work folder, links
           followed), WORD the version word

The version word is that of prover/examples/target_small_f2/policy.txt.

Status 3. A test that SUITE.md lists ends with status 3 when cp_verify as it is built from these sources, or the client's script,
accepted a file or a statement that it should have refused, and the test did not expect that; and, in verifier.ws_build_fuzz, when
the verifier's layout routine gave a layout whose blocks do not stay inside the matrix. In the first case the test also prints a
line with the words WAS NOT REFUSED and deletes nothing (exceptions: SUITE.md, "What a test's exit status means"). No row can let
status 3 pass. A row that ends with status 3 is reported as ACCEPT, and those of its lines that ALARM_WORDS matches are printed
after the word LOOK. Keep the work folder and bin/ as they are, and report it (SECURITY.md in this folder says how): the files that
were accepted and the program that accepted them are there. A status 3 from anything else, such as pytest's internal error, is
shown in the same way: read the row's output. "Cannot be run here" has a number of its own, 77, so that the two never share one.

The rows get the user's environment without the variables by which the tests control programs and scripts (the list
STEERS in this program's source). One of these in the user's shell would make rows test another configuration or build than the table says. The
variables that were dropped are printed at the start. A row that needs such a variable sets it in its own command.
BASH_ENV, ENV, CDPATH, SHELLOPTS, BASHOPTS, LD_PRELOAD, LD_AUDIT, XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS,
SSH_AUTH_SOCK, DOCKER_HOST, PYTHONOPTIMIZE and exported bash functions are not passed on either, and are not printed:
they change what every bash of a row runs first, where its cd goes or how it reads a script, load a library of the
user's into every program, are handles of the user's own services, or (PYTHONOPTIMIZE) take out the assert statements
by which Python tests check. LD_LIBRARY_PATH is passed on. Each row gets its own HOME, a folder named home in the folder
that the runner makes for the row (for an unconfined row, inside the row's TMPDIR), with XDG_CACHE_HOME, XDG_CONFIG_HOME,
XDG_DATA_HOME and XDG_STATE_HOME below it ($HOME/.cache, .config, .local/share and .local/state), so that what the
programs that it starts keep in a home stays in the row's folder, which the sandbox lets it write. Unless you set
PYTHONUSERBASE to a non-empty value yourself, the runner sets it to your ~/.local when that folder exists, so packages
that you installed with pip --user still load.
Other variables that you set to name a place in your home (PIP_CACHE_DIR, HF_HOME, TORCH_HOME, TRITON_CACHE_DIR, for
example) are passed on as they are.

Exit status: 3 if a row ended with status 3, whatever else happened; otherwise 1 if a row failed, or was not run for a
reason of the run's own, if a LOOK line was printed, or if the work folder was replaced or removed during the run;
otherwise 77 if a row did not run and said why in its own last line (one that begins "not run"), or did not run only
because a row that it needs did so; 0 if every row that was to run passed (rows that passed in part, were left out
under the version word or in mode cpu because they need nvcc or a GPU, or were taken do not prevent it); 2 if
nothing was run (--list apart). Nothing is run, and no work folder is made, if the call is wrong or if no row is to run:
none is of the kind asked for and matches --only, or the version word leaves all of these out, or the share holds none
of them. A row is not
run if a row that it needs failed, or if it reported itself that something is missing here (an environment variable, a
Python module). At the end, the rows' output files are
searched for lines by which a test says that something false was accepted, that a proof, a reply, a key or a printed
value came out of a run that should have given none, or that an honest run does not hide what it should (ALARM_WORDS).
Each is printed after the word LOOK.

Each row runs in its own process group. At its time limit every process that is still in that group gets SIGTERM (but on
the GPU, as below), so that a test can still remove what must not outlive it (a weakened copy of the verifier,
say). Those that are left twenty seconds later get SIGKILL. A row of the kind gpu or gpu-by-run gets no
signal at all, at its limit or for what it leaves: a program killed while a kernel of its own runs can make the card's
driver take the whole machine out of use, and SIGTERM is not kinder there (cp_commit's handler ends it at once, whatever
kernel is running: prover/hash_commitment/hf_commit_making.h). One case is safe, and is ended: a row of the kind gpu
is started as «flock <lock> …» (the runner's own lock in every run: GPU_LOCK's file, or <work>/gpu.lock), so where
the runner can take that same lock file within ten seconds while the row's flock has not ended, flock is still waiting
and nothing of the row has run: it is ended as a row off the GPU is, while the runner holds the lock; it fails. A row
of the kind gpu-by-run takes the lock only for the programs before which its scripts put GPU_RUN, so it is never ended
this way. Otherwise, if its group still lives, the runner names it in capitals (row, process group), leaves it
running (its programs, and what it would remove at its end, are left until it ends by itself), starts no further row on
the GPU, finishes the others and exits with status 1 (or 3 if a row accepted something false; a row left at its limit
that has ended with status 3 by the end of the run is reported so, in capitals). For a row that is not on the GPU, a process that has left the group is reached only if its parent passes the
signal on: GNU timeout makes a group of its own unless it is given --foreground, so such a row that is ended at its
limit can leave «timeout .. <program>» behind it until that timeout's own end. Processes that such a row left running in
its group when it ended by itself are ended in the same way before the next row gets its cores; but not where the row's
status was collected by something else before this program read it (the row then fails, and its group's number may
already be another's).

If this program itself gets SIGTERM, SIGINT, SIGQUIT or SIGHUP, it ends every running row in the same way, with three
seconds between the two signals; a row on the GPU again gets no signal unless it is of the kind gpu and the runner can
take the card's lock (as above), and is otherwise named and left running while its group lives (its programs, and what it would remove at its
end, are left until it ends by itself). It starts no further row,
prints the LOOK lines of the output files as they then are, and exits with status 1, or with 3 if a row had ended with
status 3. SIGHUP counts only if it was not ignored at the start, as it is under nohup. The other three count even if
they were ignored.

Three things cannot be caught by any program:
  - If a row is started at the very moment at which such a signal arrives, it inherits this program's ignoring of a
    second signal, gets only the SIGKILL and cannot clean up.
  - If this program is ended by SIGKILL, its running rows live on in their own groups until they end by themselves,
    without a time limit and without a report.
  - What a row that got SIGKILL had built stays in <work>/<row>.
Nothing here removes a work folder: it stays until you remove it.
"""  # printed by --help-more: every option, every rule, every cell

HERE = os.path.dirname(os.path.abspath(__file__))
CONFINE = os.path.join(HERE, "tools/confine.py")
FOLDER_HELD = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC  # a folder that the runner holds open
# The note for a bin/ of root's in a tree that another account owns: the same note as verifier/build.sh and
# weight_sums/build.sh print, except for how to go on, since run_suite.py has no option to build elsewhere. It names paths
# and an account's uid, and bids nobody act as root.
ROOT_MADE_BIN = ("run_suite.py: %(folder)s belongs to root, in %(top)s, which belongs to the account with uid %(uid)d. A build that does "
                 "not run as root cannot write into it. To go on without root, start again in a new copy of the repository. That the "
                 "directory is root's does not protect it: whoever owns %(top)s can replace it, cp_verify included.")  # fmt: skip
NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC  # a new file, never an existing one or a link
# Regular expressions for the output lines by which tests that SUITE.md lists report that something
# false was accepted, that a proof, a reply, a key or a printed value came out of a run that should
# have given none, or that an honest run does not hide what it should. look() prints each matching
# line again after the word LOOK. The expressions follow messages in test scripts. For the scripts
# that it lists, the check "alarms" in run_suite_test.py fails if a script no longer contains the
# words, so whoever rewords such a message changes it here and there too. The list is not complete,
# and some of these lines are also printed where a test failed for another reason: see
# SUITE.md, "What a test's exit status means".
ALARM_WORDS = (
    # Something false was accepted.
    r"(?<![,;:] 0 )WAS NOT REFUSED",  # but not a closing count of none (", 0 WAS NOT REFUSED")
    r"^LOOK[ :]",  # the lines that a test of many cases repeats at its own end
    r"\b(IS|WAS) ACCEPTED\b",  # unit tests of the verifier's routines, and two closing lines
    r"PASSES FOR THIS LINK'S|THE PLANTED FAULT WAS NOT STOPPED",  # a check of the prover's own let a fault through
    # Something came out of a run that should have withheld it. In most of these tests a test build is told to lose or spoil
    # a random value, and the program's own checks must end the run before a file is released.
    r"AND A FILE LIES UNDER -out",
    r"WANTED: no file of any kind",
    r"IT SPOKE AND (FILES ARE|A PROOF IS) THERE",
    r"IN A RUN THAT HAD TO BE REFUSED",
    r"IF PROOF AND y STAND THERE WITH NO LINE OF A SELF-CHECK",
    r"wanted no regular file under -out|a file is left staged",
    r"NOT AS EXPECTED: (commit|prove)-planted: (an ID was printed|\d+ names are left|wanted no file of any kind)",
    r"NOT AS EXPECTED: (commit|prove)-planted: (expected a nonzero status, no ID|wanted a status other than 0 and no file)",
    # A finding on hiding in honest files.
    r"^held_look: \w+: a line of hull's says: ",  # an honest key file tells two sets of weights apart
)
ALARMS = re.compile("|".join("(?:%s)" % words for words in ALARM_WORDS))
# Not passed on to the rows: BASH_ENV and ENV name a file of the caller's that a shell runs when it
# starts, CDPATH changes where a script's cd goes, SHELLOPTS and BASHOPTS change how bash reads a
# script, LD_PRELOAD and LD_AUDIT load a library of the caller's into every program (before a
# sanitizer's runtime, which must come first), PYTHONOPTIMIZE takes out the assert statements by
# which Python tests check, and the rest give access to the caller's own services. main() also drops each exported bash function (BASH_FUNC_<name>%%), which would replace
# the command of that name, even flock or cd, in every bash of a row.
UNSET = ("BASH_ENV", "ENV", "CDPATH", "SHELLOPTS", "BASHOPTS", "LD_PRELOAD", "LD_AUDIT", "XDG_RUNTIME_DIR",
         "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "DOCKER_HOST", "PYTHONOPTIMIZE")  # fmt: skip
# Variables by which tests control programs and scripts; the caller's values are not passed on.
# A name matches if it begins with an entry of the first group or equals one of the second.
STEERS = re.compile(
    r"(TOY_|H_|MB_|COMMIT_LIE_|PLANTED_|DUMMY|CHAIN_|FORMF_|ROWS_|COINS_|LENT_|TEST_|RANGE_|CP_TESTS_|"
    r"LAUNCH_COUNT_|STAND_IN_)|"
    r"(EIDB|ETAU|LIE|LIE_SEAT|PLANT|LAYER_LIE|COMMIT_LIE|WEIGHT_FEED|OUT|UNIT|CHANGED_COPIES_JOBS|"
    r"REFERENCE_PROVER|GL|ISK|IVK|CONFIG|BITS|LEAF|PROOF|ROADS|SCENES|BEFORE|TRUE_VERIFIER|MAKE_MODEL|SMALL_MODEL|"
    r"COIN_RECORD|DEAD_REQUEST|OPENING_PEAK_MB|GPU_RUN|GPU_LOCK|TOOLS|VLLM_USE_V2_MODEL_RUNNER|"
    r"VLLM_ENABLE_V1_MULTIPROCESSING|VLLM_TARGET_DEVICE|CW|KITS|TREE|COMMIT_CHEAT_WORD|HELD_WORD|DOOR_CHEAT|"
    r"PADDING_TOLD|SLOT_COUNT|NVCC_APPEND_FLAGS|NVCC_PREPEND_FLAGS|FIELD|HONEST|LIAR|TWIN|CHEAT|KIT|LAUNCHES|"
    r"OF|"  # OF: read, beside WORD, by a build script that is not part of this repository; main() sets WORD for every row
    r"DEPENDENCIES_OUTPUT|SUNPRO_DEPENDENCIES)$"  # make gcc (and nvcc, through gcc) write a dependency file where the caller says
)


def stop(why):
    print("run_suite.py: " + why, file=sys.stderr)
    sys.exit(2)


# str.isprintable() is true for these characters, but they show as blanks.
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def limits(cgroup, root, v2, v1, read):
    """Returns the limits that read() finds in the files of this process's control groups and of
    every group above them: v2 names the cgroup v2 files, v1 the cgroup v1 controller, then its
    files. Calls read(words) with the words of one group's files, in order; it returns a limit or
    None. A group whose files cannot be read or parsed is skipped."""
    try:
        with open(cgroup, errors="surrogateescape") as f:
            lines = [line.rstrip("\n").split(":", 2) for line in f]
    except OSError:
        lines = []
    found = []
    for hierarchy, controllers, own in (x for x in lines if len(x) == 3):
        if hierarchy == "0" and controllers == "":  # cgroup v2
            where, files = root, v2
        elif v1[0] in controllers.split(","):  # cgroup v1
            where, files = root + "/" + controllers, v1[1:]
        else:
            continue
        # without its own cgroup namespace a container sees the host's path, which is missing here;
        # the loop below reaches the container's own group at /
        while True:
            try:
                words = []
                for name in files:
                    with open(where + own.rstrip("/") + "/" + name, errors="surrogateescape") as f:
                        words += f.read().split()
                limit = read(words)
                if limit is not None:
                    found.append(limit)
            except (OSError, ValueError, IndexError, ZeroDivisionError):
                pass
            if own in ("", "/"):
                break
            own = os.path.dirname(own.rstrip("/"))
    return found


# "The usage text" in these comments is DETAILS above, which --help-more prints.
SLOT_BYTES = 2 << 30  # one job slot: one core and about 2 GiB (the usage text, "cpu")


def memory_limit(cgroup="/proc/self/cgroup", root="/sys/fs/cgroup", memory=None):
    """Returns the smallest of this process's control-group memory limits and the machine's memory,
    in bytes, or None if none is known. cgroup, root and memory (None: the machine's) are for
    tests."""
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if memory is None else memory
    except (AttributeError, OSError, ValueError):
        memory = None

    def limit(words):  # v2: bytes or "max"; v1: bytes
        value = None if words[0] == "max" else int(words[0])
        return None if value is None or value < 0 else value

    found = limits(cgroup, root, ("memory.max",), ("memory", "memory.limit_in_bytes"), limit)
    found += [] if memory is None or memory <= 0 else [memory]
    return min(found) if found else None


def default_jobs(cgroup="/proc/self/cgroup", root="/sys/fs/cgroup", cores=None, memory=None):
    """Returns the number of job slots when --jobs is not given (the usage text, "cpu"). cgroup,
    root, cores (None: those that this process may run on) and memory (None: the machine's) are for
    tests."""
    # the cores that this process may run on: in a container os.cpu_count() counts the host's
    try:
        cores = len(os.sched_getaffinity(0)) if cores is None else cores
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1

    def cpu_quota(words):  # v2: "<quota> <period>" or "max <period>"; v1: the quota (-1: none), then the period
        quota = None if words[0] == "max" else int(words[0])
        return None if quota is None or quota < 0 else quota // int(words[1])

    quotas = limits(cgroup, root, ("cpu.max",), ("cpu", "cpu.cfs_quota_us", "cpu.cfs_period_us"), cpu_quota)
    memory = memory_limit(cgroup, root, memory)
    return max(1, min([cores] + quotas + ([] if memory is None else [memory // SLOT_BYTES])))


def new_work_folder(named):
    """Returns the work folder as an absolute path; exits with status 2 if it may not be used. It
    creates nothing."""
    if not named:
        stop("the work folder's name is empty: name a folder that does not exist yet")
    if any(part.startswith("-") for part in named.split("/")):  # as a bare name it is an option
        stop('the work folder\'s name has a part that begins with "-": name another')
    try:  # a relative name is taken from the working folder, which may be gone
        work = os.path.abspath(named)
    except OSError as fault:
        stop("the work folder's path cannot be worked out (%s): call from a folder that exists" % fault.strerror)
    # someone will type the path again to look at the folder or remove it, and an unquoted ~, * or blank then names
    # another place; a control character, such as a line end, would change the name on the way through a shell
    inside = (work + "/").startswith(HERE + "/") or (os.path.realpath(os.path.dirname(work)) + "/").startswith(os.path.realpath(HERE) + "/")
    if re.search(r"[^A-Za-z0-9_./-]", work) or inside:  # inside also through a link above the work folder
        stop("the work folder must be new, lie outside this folder, and have a path of letters, digits and / . _ - only")
    if os.path.lexists(work):  # anything counts, also a link that leads nowhere
        stop("%s exists already: name a folder that does not exist yet" % work)
    # abspath() takes "link/.." away as text, while the system follows the link first, so the name
    # as typed and the path could name two places: such a name is rejected, as is a last part ".."
    typed = named.rstrip("/")
    while typed.endswith("/."):
        typed = typed[:-2].rstrip("/")
    to_the_system = os.path.join(os.path.realpath(os.path.dirname(typed) or "."), os.path.basename(typed))
    if typed in ("", ".", "..") or to_the_system != os.path.join(os.path.realpath(os.path.dirname(work)), os.path.basename(work)):
        stop('the work folder\'s name goes through a link and "..", or ends in "..": name it without ".."')
    return work


TRUST_EVERY_ACCOUNT = "--i-trust-every-account-on-this-machine"
ASIDE = "stood aside: below an open place:"  # begins a test's last line: see the usage text
TRUST_CLAUSE = (", or, if you trust every account on this machine, say so with %s (rows that start a program or script "
                "that makes the same look may still refuse such a place on their own)" % TRUST_EVERY_ACCOUNT)  # fmt: skip


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")  # fmt: skip


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))  # fmt: skip


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


def unseen(what, path, place, why):
    """Returns this program's message for a folder below a place that does not exist or could not be
    examined, or None if `why` is another reason. refusal(), of which other scripts such as
    driver/spool.py hold unchanged copies, would blame another account there. No option accepts
    such a place."""
    if why not in (GONE, UNSEEN):
        return None
    return ("the %s %s lies below %s, which %s: a place that cannot be looked at is not taken on trust. Name a new %s "
            "below folders that only you or root can change (such as one in your home folder, or in /tmp, whose sticky "
            "bit keeps others from replacing it); nothing was run"
            % (what, printable(path), printable(place), "does not exist" if why == GONE else "could not be examined",
               what))  # fmt: skip


def check_room(work, vouched=False):
    """Exits with status 2 if the folder above the work folder does not exist or is not a folder, or
    if others_can_change() finds a place at or above it: another account could then put another
    folder in the work folder's place while rows run, and the rows reach their files by path. With
    `vouched` (TRUST_EVERY_ACCOUNT was given), a place that exists and could be examined is returned
    as (place, why) instead. Otherwise returns None."""
    found = others_can_change(os.path.dirname(work))  # first, so that a folder shut to the caller reads as such
    if found and found[1] == GONE:
        stop("the folder above the work folder, %s, does not exist: make it first; this program makes only the work "
             "folder itself" % printable(os.path.dirname(work)))  # fmt: skip
    if not found and not os.path.isdir(os.path.dirname(work)):
        stop("the folder above the work folder, %s, is not a folder: name a work folder inside a folder"
             % printable(os.path.dirname(work)))  # fmt: skip
    if found and vouched and found[1] not in (GONE, UNSEEN):  # the option trusts accounts, not a place that cannot be looked at
        return found
    if found:
        place, why = found
        stop(unseen("work folder", work, place, why) or refusal("work folder", work, place, why, TRUST_CLAUSE))
    return None


def build_note(vouched=False):
    """Checks BUILD, if it is set and not empty, and exits with status 2 if it may not be used (the
    usage text has the conditions). It also exits if others_can_change() finds a place at or above
    BUILD (another account could put another folder in its place, with programs of its own that the
    rows would then run) or BUILD (or, while BUILD does not exist yet, the folder above it) has a
    default access control list. With `vouched` (TRUST_EVERY_ACCOUNT was given), such a find is
    returned as (place, why) instead, unless the place does not exist or could not be examined.
    Otherwise returns None."""
    build = os.environ.get("BUILD")
    if not build:
        return None  # <work>/build, which this run makes in its own closed folder
    if not os.path.isabs(build) or re.search(r"[^A-Za-z0-9_./-]", build) or any(p in (".", "..") for p in build.split("/")):
        stop('BUILD, when set, must be an absolute path of letters, digits and / . _ - only, without "." or ".." as a '
             "part; it was %r" % build)  # fmt: skip
    build = build.rstrip("/") or "/"
    # The rows build into BUILD. A partial guard against a run as root that builds over another
    # account's programs: BUILD may not be / or a folder right below it, also not through a doubled
    # slash or a symbolic link.
    if os.path.realpath(build).count("/") < 2:
        stop("BUILD, when set, must lie at least two folders below /, for example ~/build; it was %r" % build)
    there = build if os.path.lexists(build) else os.path.dirname(build)
    about = os.lstat(there) if os.path.lexists(there) else None
    why = ("does not exist or could not be examined" if about is None else "is a symbolic link" if stat.S_ISLNK(about.st_mode)
           else "is not a folder" if not stat.S_ISDIR(about.st_mode)
           else "belongs to another account" if about.st_uid not in trusted_owners()
           else "can be written by its group or other accounts; make a new folder of your own first (for example: mkdir -m 700 ~/build) and give that as BUILD"
           if about.st_mode & 0o022 else "")  # fmt: skip
    if why:
        named = there if there == build else "the folder above it, " + there + ","
        stop("BUILD, when set, must be a folder that you (or root) own and only its owner can write in, or a new name inside one; %s %s"
             % (named, why))  # fmt: skip
    # what the rows create in BUILD gets a default ACL's rights, whatever the umask
    found = others_can_change(there) or ((there, ACL) if default_acl(there) else None)
    if found and (not vouched or found[1] in (GONE, UNSEEN)):
        place, why = found
        stop(unseen("build folder (BUILD)", build, place, why) or refusal("build folder (BUILD)", build, place, why,
                                                                           TRUST_CLAUSE))  # fmt: skip
    return found


def shared_note(source, vouched=False):
    """Returns None, or, where the caller gave TRUST_EVERY_ACCOUNT, (place, why) for a folder given with --shared-from that
    another account could change: the folder, a folder above it, or a folder or file below it. The rows run the copies
    that the runner lays in $S, so such an account could put programs of its own into S while they are copied. Exits
    with status 2 otherwise, as for BUILD, and for a folder above it that does not exist or could not be examined."""
    if os.path.islink(source):  # a folder given as a symbolic link is refused here; nothing is copied
        stop("--shared-from: %s is a symbolic link: name the folder itself; nothing was run" % printable(source))
    found = others_can_change(os.path.dirname(source))  # the folders above it; a sticky one passes, as for BUILD
    if found and (not vouched or found[1] in (GONE, UNSEEN)):
        place, why = found
        stop(unseen("folder for --shared-from", source, place, why)
             or refusal("folder for --shared-from", source, place, why, TRUST_CLAUSE))  # fmt: skip
    if found:
        return found
    # the folder itself and everything below it: whatever lies there is copied into S, so a sticky folder does not pass
    # (others could add a name of their own between this look and the copy)
    for where in [source] + [os.path.join(top, n) for top, folders, files in os.walk(source) for n in folders + files]:
        try:
            about = os.lstat(where)
        except OSError as fault:  # the walk in main() found it: gone or unreadable now, it is refused
            stop("--shared-from: %s could not be examined (%s), though it was there a moment before: something moved or "
                 "removed it while the run started. Nothing was run; run again once nothing is changing this folder"
                 % (printable(where), fault.strerror))  # fmt: skip
        why = (THEIRS if about.st_uid not in trusted_owners()
               else WRITABLE if (stat.S_ISDIR(about.st_mode) or stat.S_ISREG(about.st_mode)) and about.st_mode & 0o022
               else None)  # fmt: skip
        if why and not vouched:
            stop("--shared-from: %s%s %s: another account could change what the rows run while it is copied. %sive a "
                 "folder whose files and folders only you or root can change%s; nothing was run"
                 % (printable(where), "" if where == source else ", below the folder " + printable(source) + ",",
                    "belongs to another account" if why == THEIRS else "can be written by its group or by other accounts",
                    # one place at a time, as for the work folder: each chmod names a place the caller has just been shown, where a
                    # recursive one would also change, unseen, a file elsewhere that has a second name (a hard link) in the folder
                    "If %s is yours, close it with chmod go-w %s (a further such file or folder in %s, if any, is named in "
                    "turn); otherwise g" % (printable(where), shlex.quote(printable(where)), printable(source))
                    if why != THEIRS and os.geteuid() != 0 and about.st_uid == os.geteuid() else "G", TRUST_CLAUSE))  # fmt: skip
        if why:
            return where, why
    return None


def on_shared_memory(folder):
    """Whether `folder` lies in /dev/shm, in which tools/confine.py lets every confined test make, write and remove files.
    Where /dev/shm is a file system of its own, any folder on it counts (also one reached by another name, such as a bind
    mount or /run/shm); where it is a plain folder, only a folder whose real path lies under it."""
    try:
        if not os.path.isdir("/dev/shm"):
            return False
        if os.stat("/dev/shm").st_dev != os.stat("/dev").st_dev:  # a mount of its own
            return os.stat(folder).st_dev == os.stat("/dev/shm").st_dev
        return (os.path.realpath(folder) + "/").startswith(os.path.realpath("/dev/shm") + "/")
    except OSError:  # check_room() has already refused a folder that cannot be examined
        return False


def nearest_folder(path):
    """The real path of `path` if it is a folder, else of the nearest folder that exists above it (through links, as the
    rows' programs follow them)."""
    place = os.path.realpath(path)
    while not os.path.isdir(place) and place != os.path.dirname(place):
        place = os.path.dirname(place)
    return place


def make_work(work):
    """Creates the work folder, closed to other accounts, and its folders screens, shared and tmp
    (the temporary files of rows that run unconfined). Returns the work folder's (device, inode) and
    {name: an open descriptor} of the work folder ("work") and of screens and tmp. Exits with status
    2 if the folder cannot be created, changed before it was opened, or shows permissions for other
    accounts."""
    umask = os.umask(0o077)  # closed to other accounts from its first moment
    try:
        # a plain mkdir, which fails if anything has appeared there since new_work_folder() looked; it makes nothing
        # above the work folder. The mode also limits a default ACL, which the umask does not.
        os.mkdir(work, 0o700)
    except OSError as fault:
        stop("%s was not made (%s): name a folder that does not exist yet, in a folder that exists" % (work, fault.strerror))
    finally:
        os.umask(umask)
    try:
        here = os.open(work, FOLDER_HELD)
    except OSError:  # replaced by a link, or removed, since the mkdir
        stop("%s changed between its making and its opening: another account may write above it; nothing was run" % work)
    made = os.fstat(here)
    # opened by its path, so check that it is the folder just created: ours and empty here, closed
    # to other accounts below
    if made.st_uid != os.geteuid() or os.listdir(here):
        stop("%s changed between its making and its opening: another account may write above it; nothing was run" % work)
    if stat.S_IMODE(made.st_mode) & 0o077:  # ours and empty, but open: a file system that shows fixed modes
        stop("%s was made with mode 700 but shows mode %o: its file system may not keep modes (vfat, some network or FUSE "
             "file systems), so other accounts could read or write in it; name a work folder on one that keeps them, such "
             "as one in your home folder; nothing was run" % (work, stat.S_IMODE(made.st_mode)))  # fmt: skip
    # the runner's own folders and files are created relative to the open work folder, so that they
    # stay in it even if the path comes to name another folder
    for folder in ("screens", "shared", "tmp"):
        os.mkdir(folder, 0o700, dir_fd=here)
    opened = {name: os.open(name, FOLDER_HELD, dir_fd=here) for name in ("screens", "tmp")}
    opened["work"] = here
    return (made.st_dev, made.st_ino), opened


def lay_folder(source, into, vouched=False):
    """Copies the plain files and folders below the folder `source` into the folder whose open
    descriptor is `into`, and returns the number of files copied. A file gets its source's mode, cut
    to at most 0o755 and by the umask; a folder gets 0o700. Below `source` no link is followed and
    nothing is opened by its whole path: each level is opened relative to the one above. Exits with
    status 2 at a symbolic link or any other kind of file, at a name that already exists in `into`,
    or if `source` cannot be opened as a folder. Unless `vouched`, it also exits if a folder or file
    that it has open belongs to an account that trusted_owners() does not name or can be written by
    its group or others: what it copies is what it judged, even if something was swapped in after
    shared_note() looked."""
    try:
        held = os.open(source, FOLDER_HELD)
    except OSError as fault:
        stop("--shared-from: %s cannot be opened as a folder (%s)" % (printable(source), fault.strerror))
    try:
        held_as_judged(held, source, vouched)
        return lay_level(held, into, source, vouched)
    finally:
        os.close(held)


def held_as_judged(fd, shown, vouched):
    """Exits with status 2 if the folder or file open as `fd` belongs to an account that
    trusted_owners() does not name, or can be written by its group or others, unless `vouched`."""
    about = os.fstat(fd)
    if not vouched and (about.st_uid not in trusted_owners() or about.st_mode & 0o022):
        stop("--shared-from: %s changed after the runner checked it (it now %s); another account may have swapped it in. "
             "Nothing was run. Make sure that only you or root can change this folder and everything in it, then run again "
             "with a new work folder"
             % (printable(shown), "belongs to another account" if about.st_uid not in trusted_owners()
                else "can be written by its group or by other accounts"))  # fmt: skip


# One level of lay_folder(): `held` and `into` are open folders; `shown` is the path of `held`.
def lay_level(held, into, shown, vouched=False):
    laid = 0
    try:
        names = sorted(os.listdir(held))
    except OSError as fault:
        stop("--shared-from: %s could not be laid in $S (%s): nothing was run" % (printable(shown), fault.strerror))
    for name in names:
        where = printable(os.path.join(shown, name))
        try:
            about = os.stat(name, dir_fd=held, follow_symlinks=False)
        except OSError as fault:
            stop("--shared-from: %s could not be laid in $S (%s): nothing was run" % (where, fault.strerror))
        try:
            if stat.S_ISDIR(about.st_mode):
                os.mkdir(name, 0o700, dir_fd=into)
                below, made = os.open(name, FOLDER_HELD, dir_fd=held), os.open(name, FOLDER_HELD, dir_fd=into)
                try:
                    held_as_judged(below, os.path.join(shown, name), vouched)
                    laid += lay_level(below, made, os.path.join(shown, name), vouched)
                finally:
                    os.close(below), os.close(made)
            elif stat.S_ISREG(about.st_mode):
                source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=held)
                try:
                    if not stat.S_ISREG(os.fstat(source).st_mode):  # it changed kind since the check above
                        stop("--shared-from: %s is no longer a plain file: nothing was run" % where)
                    held_as_judged(source, os.path.join(shown, name), vouched)
                    target = os.open(name, NEW_FILE, stat.S_IMODE(about.st_mode) & 0o755, dir_fd=into)
                    try:
                        while os.sendfile(target, source, None, 1 << 20):
                            pass
                    finally:
                        os.close(target)
                finally:
                    os.close(source)
                laid += 1
            else:
                stop("--shared-from: %s is neither a plain file nor a folder (a link, for example): nothing was run" % where)
        except OSError as fault:
            stop("--shared-from: %s could not be laid in $S (%s): nothing was run" % (where, fault.strerror))
    return laid


def make_lock(path, at=None):
    """Creates the GPU lock file ("the card's lock" in messages and in the usage text) at `path`
    (relative to the open folder `at`, if given) unless a regular file is there already. Exits with
    status 2 if something else is there, such as a symbolic link, or if it cannot be created."""
    try:
        # opened for reading, as flock opens it, so that another account's lock file, which all runs
        # on one GPU share, can be used; O_NONBLOCK, so that a FIFO put in its place does not block
        held = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o644, dir_fd=at)
        try:
            about = os.fstat(held)
        finally:
            os.close(held)
    except OSError as fault:
        stop("the card's lock %s cannot be made or opened as a file: %s" % (printable(path), fault.strerror))
    if not stat.S_ISREG(about.st_mode):
        stop("the card's lock %s is not a regular file" % printable(path))


def read_screen(opened, name):
    """Returns the text of the output file of the row `name`. Raises OSError if it is not a regular
    file or cannot be read; a FIFO put in its place does not block."""
    screen = os.open(name + ".txt", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=opened["screens"])
    with os.fdopen(screen, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        return f.read().decode("utf-8", "replace")


def say_replaced(work, shown):
    """Prints the alarm for a work folder that the path `work` no longer names; `shown`: where it is
    now, or None."""
    print("run_suite.py: THE WORK FOLDER WAS REPLACED OR REMOVED WHILE THE RUN WENT ON: %s is no longer the folder "
          "that this run made, %s; look at who could write above it" % (work, "which is now at %s; keep both as they are"
          % shown if shown else "which was removed"), flush=True)  # fmt: skip


def where_now(work, made, opened):
    """Returns whether the path `work` no longer names the folder that this run created (`made`: its
    device and inode), and where that folder is now: `work`, another path, or None if it was removed
    or cannot be found."""
    try:
        now = os.lstat(work)
        if (now.st_dev, now.st_ino) == made:
            return False, work
    except OSError:
        pass
    try:
        where = os.readlink("/proc/self/fd/%d" % opened["work"])
    except OSError:
        where = None
    if where is None or where.endswith(" (deleted)"):
        return True, None
    # another account may have chosen that name, so anything but a plain path is shown escaped
    return True, ascii(where) if re.search(r"[^A-Za-z0-9_./-]", where) else where


# The reasons that others_can_change() and build_note() give; note() chooses its words by them.
ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
WRITABLE = "can be written by its group or by other accounts"  # shared_note(): at or below --shared-from, sticky or not
UNSEEN = "could not be examined"
GONE = "does not exist"


def note(found):
    """If `found`, a list of (place, why), is not empty, prints one line on standard error that
    names each place once, in order. A place that has gone since is named without its mode or
    owner."""
    said = []
    for place, why in dict.fromkeys(found):  # each once, in order
        try:
            about = os.lstat(place)
        except OSError:  # gone since it was found: the reason alone
            about = None
        shown = printable(place)  # the path as given can hold any character
        if why == ACL:
            said.append("%s (a default access control list)" % shown)
        elif why == UNSEEN:
            said.append("%s (it could not be examined)" % shown)
        elif why == GONE:
            said.append("%s (it does not exist)" % shown)
        elif why == THEIRS:
            said.append("%s (%s another account%s)" % (shown, "a symbolic link of" if about and stat.S_ISLNK(about.st_mode) else "owned by",
                                                        "" if about is None else " (uid %d)" % about.st_uid))  # fmt: skip
        else:
            said.append("%s (%sno sticky bit)" % (shown, "" if about is None else "mode %o, " % stat.S_IMODE(about.st_mode)))
    if said:
        print("run_suite.py: note: another account may be able to change %s, under which this run works, and so could change what "
              "the rows run or read; for a run that no other account can touch, keep the tested tree (%s) below folders that "
              "only you or root can change"
              % (said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1], printable(HERE)),
              file=sys.stderr, flush=True)  # fmt: skip


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def default_acl(folder):
    """Whether `folder` has a default access control list, which gives what is created in it rights
    that the umask does not take away."""
    try:
        return bool(os.getxattr(folder, "system.posix_acl_default", follow_symlinks=False))
    except (OSError, AttributeError):  # none, or a system without such lists
        return False


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


def card_free(lock, then, wait=10, ident=None):
    """Tries for `wait` seconds to take the GPU lock (the file `lock`), calls then() while holding
    it, and returns whether then() returned a true value. Returns False, without calling then(), if
    the lock cannot be opened or taken, or if `ident` is given and `lock` is not the file with that
    (device, inode). A row of kind gpu is started as "flock <lock> ...". So while this program holds
    the lock and the row's leader (flock, or the shell that waits for it) still runs, flock is still
    waiting and nothing of the row has run: then() may end the row. A row whose flock has ended may
    have left a program on the GPU that holds no lock, because a process started by Python's
    subprocess does not inherit the open lock file. So then() must check that the leader still runs
    (end_of(leader, 0) is None) before it ends anything."""
    try:
        fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return False
    try:
        about = os.fstat(fd)
        if ident is not None and (about.st_dev, about.st_ino) != ident:  # not the file that the rows' flock holds
            return False
        until = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    return False
                time.sleep(0.2)
            except OSError:  # another fault of flock (a file system without locks, say): not free, as far as we know
                return False
        return bool(then())
    finally:
        os.close(fd)  # the lock goes with it


def end_if_waiting(leader, grace):
    """The then() of card_free(), called while this program holds the GPU lock. If the leader of a
    row of kind gpu still runs, so that its flock is still waiting and nothing of the row has run on
    the GPU, ends the row with end_the_groups() and returns True. Otherwise ends nothing and returns
    False."""
    if end_of(leader, 0) is not None:
        return False
    end_the_groups([leader], grace)
    return True


def live_groups():
    """The process groups that have a member that is not a zombie, read from /proc in one scan."""
    groups = set()
    for pid in os.listdir("/proc"):
        try:
            with open("/proc/%s/stat" % pid, errors="replace") as f:  # a process's name need not be UTF-8
                fields = f.read().rsplit(")", 1)[1].split()  # state, parent, process group
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[0] != "Z" and fields[2].isdigit():
            groups.add(int(fields[2]))
    return groups


LOST = -1000  # in place of a status that could not be read (see `end_of`); no row can let it pass


def end_of(leader, seconds):
    """Waits up to `seconds` for a row's leader to end and returns its status, or None if it still
    runs. The leader is the Popen handle of the row's first process, whose process number is also
    its group's. end_of() does not reap it: until it is reaped, the number cannot go to another
    process, so end_the_groups() reaches only this row's processes. play() reaps it at the row's
    end, unless this program has got a signal or the row is a GPU row that is left running. If
    something else has reaped it, its status is lost: end_of() returns LOST and sets the handle's
    returncode to LOST, so that nothing signals the number and no row passes on it."""
    until = time.monotonic() + seconds
    while True:
        try:
            info = os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:  # reaped elsewhere; `main` resets SIGCHLD to prevent one cause
            leader.returncode = LOST  # otherwise `Popen.wait` would report status 0
            return LOST
        if info is not None:
            return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status  # negative: ended by that signal
        if time.monotonic() >= until:
            return None
        time.sleep(0.02)


def end_the_groups(leaders, grace):
    """Ends rows: sends SIGTERM to each leader's process group; once no group has a live member, or
    after `grace` seconds, sends SIGKILL to each, and waits up to `grace` seconds more. Skips a
    leader whose handle already has a status (see end_of()), or that something else has reaped: its
    number may belong to another process. Also skips a leader that has ended and whose group has no
    live member. Only end_if_waiting() passes a row on the GPU, while this program holds the GPU
    lock and the row still waits for it; any other row on the GPU gets no signal at all (the usage
    text gives the reason)."""
    for sign in (signal.SIGTERM, signal.SIGKILL):
        # Each signal starts again from every leader that is not reaped, not from the groups that
        # the wait below still found alive: a scan of /proc can miss a member.
        left = [leader for leader in leaders if leader.returncode is None]
        for leader in left:
            # The leader's number names its group only while the leader is this program's child and
            # is not reaped, so ask first, without reaping it.
            try:
                ended = os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                leader.returncode = LOST
                continue
            if ended and leader.pid not in live_groups():
                continue
            try:
                os.killpg(leader.pid, sign)
            except ProcessLookupError:
                pass
        until = time.monotonic() + grace
        while True:
            live = live_groups()
            left = [leader for leader in left if leader.pid in live]
            if not left or time.monotonic() >= until:
                break
            time.sleep(0.1)


def rows_of(path, public=None):
    """Returns {name: row} from the tables in `path`, in order; the usage text describes the cells.
    `public`: when `path` is a further table (see tables()), the rows read before it, which are
    SUITE.md's and earlier further tables': its rows may need them and may not share a name with
    them. Exits with status 2 at a malformed row, but does not check "slots" and "minutes": a bad
    one raises ValueError."""
    rows, table = {}, os.path.basename(path)
    # the messages below name a table by its file's name
    public = public or {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.startswith("|") or line.startswith(("| name |", "|---")):
                continue
            raw = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cell.strip("`") for cell in raw]
            if len(cells) != 8 or not re.fullmatch(r"[A-Za-z0-9._-]+", cells[0]) or cells[0] in rows or cells[0] in public:
                stop("%s, line %d: the row does not have eight cells, or its name is malformed or already used" % (table, n))
            if any(len(cell) < 2 or cell[0] != "`" or cell[-1] != "`" for cell in raw[6:]):
                stop("%s, line %d: the cells 'passes' and 'command' must each stand between two backticks" % (table, n))
            name, kind, needs, word, slots, minutes, passes, command = cells
            statuses, _, last = passes.partition(" ")
            last, *lines = last.split(" ;; ")
            if kind not in ("cpu", "nvcc", "gpu", "gpu-by-run", "long") or not re.fullmatch(r"[A-Za-z0-9]+", word):
                stop("%s, line %d: the cell 'kind' must be cpu, nvcc, gpu, gpu-by-run or long, and the cell 'word' must be any or a version word (letters and digits)" % (table, n))
            if not re.fullmatch(r"\d+(,\d+)*", statuses):
                stop("%s, line %d: the cell 'passes' must begin with statuses in digits, separated by commas" % (table, n))
            statuses = [int(x) for x in statuses.split(",")]
            if 3 in statuses:
                stop("%s, line %d: status 3 means that something false was accepted, and no row may let it pass: "
                     "take 3 out of the cell 'passes'" % (table, n))  # fmt: skip
            for expression in [last] + lines:
                try:
                    re.compile(expression)
                except re.error as fault:
                    stop("%s, line %d: in the cell 'passes', %r is not a regular expression: %s" % (table, n, expression, fault))
            rows[name] = dict(name=name, kind=kind, needs=[] if needs == "-" else needs.split(","), word=word,
                              slots=int(slots), minutes=float(minutes), statuses=statuses,
                              last=last, lines=lines, command=command)  # fmt: skip
    for row in rows.values():
        for need in row["needs"]:
            if need not in rows and need not in public:  # SUITE.md's rows need none of a further table's
                stop("%s: %s needs %s, which is not a row%s" % (table, row["name"], need, "" if public else
                     " of it (a row of SUITE.md may not need a row of a further table)"))  # fmt: skip
    return rows


FURTHER = "<!-- a further table of tests for run_suite.py -->"  # the first line of a further table


def tables(folder=HERE):
    """Returns the rows of SUITE.md and of each further table, as one table. A further table is a
    file SUITE-<word>.md in `folder` whose first line is FURTHER. They are
    read in sorted order, and a row of one may need rows of SUITE.md and of the tables read before
    it, but not the other way round. Any other SUITE-<word>.md is skipped as a page, unless it
    cannot be read or has the head line of a table of tests ("| name | kind |"): then exits with
    status 2, so that no rows are left out unnoticed. Also exits with status 2 if a row of another
    kind needs a row of kind long, which only mode long runs."""
    rows = rows_of(os.path.join(folder, "SUITE.md"))
    for name in sorted(os.listdir(folder)):
        if not re.fullmatch(r"SUITE-[A-Za-z0-9_-]+\.md", name):
            continue
        try:
            with open(os.path.join(folder, name), errors="replace") as f:
                first, rest = f.readline().rstrip("\n"), f.read()
        except OSError as fault:  # a page that cannot be read could hold rows: not passed over unseen
            stop("%s cannot be read (%s): a table of tests there would be left out; make it readable, or move it away"
                 % (name, fault.strerror))  # fmt: skip
        if first == FURTHER:
            rows.update(rows_of(os.path.join(folder, name), public=rows))
        elif re.search(r"^\| name \| kind \|", first + "\n" + rest, re.M):  # a table of tests without its first line would be left out unseen
            stop("%s holds a table of tests, but its first line is not %s: add that line, or its rows are not run" % (name, FURTHER))
    for row in rows.values():
        for need in row["needs"]:
            if rows[need]["kind"] == "long" and row["kind"] != "long":
                stop("%s needs %s, a row of the kind long, which only mode long runs: make %s long too, or drop the need"
                     % (row["name"], need, row["name"]))  # fmt: skip
    return rows


class Slots:
    """A count of job slots, and the one GPU. take() waits until n slots are free (all of them, if n
    is larger) and, for a row that must have the GPU to itself, until no other such row runs. A row
    that waits holds neither."""

    def __init__(self, n):
        self.free, self.most, self.gpu_in_use, self.changed = n, n, False, threading.Condition()

    def take(self, n, alone):
        n = min(n, self.most)
        with self.changed:
            self.changed.wait_for(lambda: self.free >= n and not (alone and self.gpu_in_use))
            self.free -= n
            self.gpu_in_use = self.gpu_in_use or alone
        return n

    def give(self, n, alone):
        with self.changed:
            self.free += n
            self.gpu_in_use = self.gpu_in_use and not alone
            self.changed.notify_all()


def main():
    # If this program's parent ignored SIGCHLD, this program would inherit that: the kernel would
    # reap the rows' leaders itself, and their numbers could go to other processes before
    # end_the_groups() signals them.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    # What this program and the rows create is not writable by group or others; a stricter umask
    # that the caller set is kept. The owner's bits are cleared from the umask, or this program
    # could not write in the folders that it creates.
    os.umask(os.umask(0o077) & 0o077 | 0o022)
    argv, jobs, only, listing, share, taken_as, confined = sys.argv[1:], default_jobs(), "", False, (1, 1), None, True
    shared_from, vouched = None, False
    if "--confined" in argv and "--unconfined" in argv:
        stop("--confined and --unconfined: give one of them, or neither (then the rows run confined)")
    while len(argv) > 2 or (argv and argv[-1] in ("--list", "--confined", "--unconfined", TRUST_EVERY_ACCOUNT)):
        if argv[-1] == "--list":
            listing = True
            argv = argv[:-1]
        elif argv[-1] == TRUST_EVERY_ACCOUNT:
            vouched, argv = True, argv[:-1]
        elif argv[-1] in ("--confined", "--unconfined"):
            confined = argv[-1] == "--confined"
            argv = argv[:-1]
        elif len(argv) >= 4 and argv[-2] == "--jobs" and argv[-1].isdigit() and int(argv[-1]):
            jobs, argv = int(argv[-1]), argv[:-2]
        elif len(argv) >= 4 and argv[-2] == "--only":
            only, argv = argv[-1], argv[:-2]
        elif len(argv) >= 4 and argv[-2] == "--taken":
            taken_as, argv = argv[-1], argv[:-2]
        elif len(argv) >= 4 and argv[-2] == "--shared-from":
            shared_from, argv = argv[-1], argv[:-2]
        elif len(argv) >= 4 and argv[-2] == "--share" and re.fullmatch(r"[1-9]\d*/[1-9]\d*", argv[-1]):
            share, argv = tuple(map(int, argv[-1].split("/"))), argv[:-2]
        else:
            break
    if argv == ["--help-more"]:
        print(DETAILS)
        return 0
    if len(argv) != 2 or argv[0] not in ("cpu", "gpu", "all", "long"):
        print(__doc__)
        return 2
    on_the_gpu = ("gpu", "gpu-by-run")
    off_the_cpu = ("nvcc",) + on_the_gpu  # the kinds that mode gpu chooses: they need nvcc or a GPU
    kinds = {"cpu": ("cpu",), "gpu": off_the_cpu, "all": ("cpu",) + off_the_cpu, "long": ("long",)}[argv[0]]
    work = os.path.abspath(argv[1])
    for option, expression in (("--only", only), ("--taken", taken_as or "")):
        try:
            re.compile(expression)
        except re.error as fault:
            stop("%s: %r is not a regular expression: %s" % (option, expression, fault))
    if taken_as == "":
        stop("--taken: the expression is empty, so it would match every row that another needs")
    with open(os.path.join(HERE, "prover/examples/target_small_f2/policy.txt")) as f:
        word = re.search(r"^version ([A-Za-z0-9]+)$", f.read(), re.M)[1]
    rows = tables()
    chosen = {n for n, r in rows.items() if r["kind"] in kinds and re.search(only, n)}
    out = {n for n, r in rows.items() if r["word"] not in ("any", word)}
    while True:  # a row that needs a row that is left out is left out too
        more = {n for n, r in rows.items() if n not in out and out.intersection(r["needs"])}
        if not more:
            break
        out |= more
    left_out, matched = chosen & out, len(chosen)
    chosen -= out
    # needs_the_gpu: a chosen row -> the first row found, among those that it needs at any depth,
    # whose kind is in off_the_cpu. Mode cpu leaves such chosen rows out (the usage text, "cpu").
    needs_the_gpu = {}
    if argv[0] == "cpu":
        for n in sorted(chosen):
            seen, todo = set(), list(rows[n]["needs"])
            while todo:
                m = todo.pop(0)
                if m in seen:
                    continue
                seen.add(m)
                if taken_as is not None and re.search(taken_as, m):  # taken: assumed present; what it needs is not run
                    continue
                if rows[m]["kind"] in off_the_cpu:
                    needs_the_gpu[n] = m
                    break
                todo += rows[m]["needs"]
        chosen -= set(needs_the_gpu)
    only_the_gpus = bool(needs_the_gpu) and not chosen  # before --share: every chosen row needs nvcc or a GPU
    if share[0] > share[1]:
        stop("--share i/n: i must be from 1 to n")
    # Each row that no other chosen row needs, the longest first, goes to the share with the fewest
    # minutes so far.
    needed = {need for n in chosen for need in rows[n]["needs"]}
    minutes, mine = [0.0] * share[1], set()
    for n in sorted(chosen - needed, key=lambda n: (-rows[n]["minutes"], n)):
        least = minutes.index(min(minutes))
        minutes[least] += rows[n]["minutes"]
        if least == share[0] - 1:
            mine.add(n)
    chosen, taken, full = mine, set(), set(mine)
    grown = True
    while grown:  # the rows that the chosen rows need are added, whatever their kind; what a taken row needs is not
        more = {need for n in chosen - taken for need in rows[n]["needs"]} - chosen
        taken |= {n for n in more if taken_as is not None and re.search(taken_as, n)}
        grown, chosen = bool(more), chosen | more
    while taken:  # what would have been run without --taken
        more = {need for n in full for need in rows[n]["needs"]} - full
        if not more:
            break
        full |= more
    gone = len(full - chosen)  # the rows that only taken rows need
    if listing:
        for n in rows:
            if n in chosen:
                kind = "taken" if n in taken else rows[n]["kind"]
                print("%-44s %-10s %6.1f min  %s" % (n, kind, rows[n]["minutes"], rows[n]["command"][:110]))
        total = sum(rows[n]["minutes"] for n in chosen - taken)
        print("%d rows, %d of them taken (--taken), %.0f minutes one after the other; %d rows that only taken rows need are not "
              "listed; version word %s leaves out %d rows" % (len(chosen), len(taken), total, gone, word, len(left_out)))  # fmt: skip
        for name in sorted(needs_the_gpu):
            print("left out %s: it needs a row that needs nvcc or a GPU (%s); mode all runs it" % (name, needs_the_gpu[name]))
        return 0
    if only_the_gpus:
        stop("nothing was run: in mode cpu each chosen row needs a row that needs nvcc or a GPU (%s); mode all runs them"
             % ", ".join("%s needs %s" % (n, needs_the_gpu[n]) for n in sorted(needs_the_gpu)))  # fmt: skip
    if not chosen:
        stop("nothing was run, because no row was chosen: %d rows of the table belong to '%s' and match --only; "
             "version word %s leaves out %d of them; share %d/%d holds none of the others"
             % (matched, argv[0], word, len(left_out), *share))  # fmt: skip
    work = new_work_folder(argv[1])
    open_above_work = check_room(work, vouched)
    work = os.path.join(os.path.realpath(os.path.dirname(work)), os.path.basename(work))  # checked as typed and resolved; used resolved from here on
    if confined and on_shared_memory(os.path.dirname(work)):
        stop("a work folder in /dev/shm can't be kept apart by the sandbox, which lets every test write files there; name a "
             "folder elsewhere (%s lies in /dev/shm); nothing was run" % printable(os.path.dirname(work)))  # fmt: skip
    if re.search(r"[^A-Za-z0-9_./-]", work):
        stop("the work folder's path leads through a link to %s, whose name has characters other than letters, digits and "
             "/ . _ -: name a work folder whose real path has none" % printable(work))  # fmt: skip
    build = os.path.abspath(os.environ.get("BUILD") or os.path.join(work, "build"))  # "or": an empty BUILD would otherwise mean the current folder
    if any((os.path.realpath(held) + "/").startswith(os.path.realpath(build).rstrip("/") + "/") for held in (HERE, work)):  # with links resolved, as confine.py does
        stop("BUILD names %s, which is the root or holds the tests' folder (%s) or the work folder: the rows that build may write and remove there" % (build, HERE))
    open_above_build = build_note(vouched)
    build = os.path.join(os.path.realpath(os.path.dirname(build)), os.path.basename(build))  # likewise used resolved
    if re.search(r"[^A-Za-z0-9_./-]", build):
        stop("BUILD leads through a link to %s, whose name has characters other than letters, digits and / . _ -: give a "
             "BUILD whose real path has none" % printable(build))  # fmt: skip
    level = None
    if confined:
        sys.path.insert(0, os.path.dirname(CONFINE))
        sys.dont_write_bytecode = True  # no __pycache__ in the tree
        import confine  # noqa: E402

        level, why, _ = confine.level_here()
        if not level:
            root_failed = os.geteuid() == 0 and confine.level_here(probe_root=False)[0]  # Landlock is there: root's own setup failed
            stop("NOT CONFINED, nothing is run: " + printable(why) + ". " + ("Run as an account other than root, or, to run the rows unconfined "
                 "all the same, give --unconfined: each row may then write and remove any file on the machine and signal any process"
                 if root_failed else "To run the rows unconfined all the same, give --unconfined: each row may then write and remove "
                 + ("any file on the machine and signal any process" if os.geteuid() == 0 else
                    "wherever this account can, and signal this account's other processes")))  # fmt: skip
    on_gpu = any(rows[n]["kind"] in ("gpu", "gpu-by-run") for n in chosen)
    the_lock = os.path.abspath(os.environ.get("GPU_LOCK") or work + "/gpu.lock")
    if on_gpu and os.path.realpath(the_lock) in {os.path.realpath(w) for w in os.environ.get("GPU_PREFIX", "").split()}:  # also through a link
        stop("GPU_PREFIX names the card's lock %s, which the runner puts before every row itself: each row would wait on its own hold. "
             "Leave the lock out of GPU_PREFIX" % the_lock)  # fmt: skip
    tried = confine.rows_as_root() if level else None  # what root's rows get (tools/confine.py), as the first line says
    if tried:
        kind, why, writable = tried
        hidden = [("this folder (the tests' tree) " + printable(HERE), HERE), ("BUILD " + printable(build), build)]
        hidden += [("%s's %s" % (name, printable(os.environ[name])), os.environ[name])
                   for name in ("PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX", "CUDA_HOME", "CUDA_MPS_PIPE_DIRECTORY") if os.environ.get(name)]  # fmt: skip
        hidden += [("your Python user base " + printable(site.getuserbase()), site.getuserbase())] if os.path.isdir(site.getuserbase() or "") else []
        hidden += [("%s's %s" % (name, printable(p)), p) for name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
                   for p in os.environ.get(name, "").split(":") if p.startswith("/") and os.path.isdir(p)]  # fmt: skip
        for what, path in hidden if kind == confine.LAID else ():
            if on_shared_memory(nearest_folder(path)):
                listed = what.split("'s ")[0] if what.startswith(("PATH's", "PYTHONPATH's", "LD_LIBRARY_PATH's")) else ""
                name = what.split("'s ")[0] if "'s " in what else ""
                cure = ("Take it out of %s, or move it off /dev/shm" % listed if listed else "Move the tests' tree off /dev/shm" if what.startswith("this folder")
                        else "Set PYTHONUSERBASE to a folder off /dev/shm" if what.startswith("your Python user base") else "Set %s to a folder off /dev/shm" % (name or what.split(" ")[0]))  # fmt: skip
                stop("nothing is run: %s is on /dev/shm's file system, but as root each row gets an empty /dev/shm of its own and would not see it. "
                     "%s, or run as an account other than root" % (what, cure))  # fmt: skip
        if kind == confine.UNMADE and writable and on_gpu:
            stop("NOT CONFINED, nothing is run: as root, the rows' own mount namespace could not be set up here (%s), and rows on the GPU "
                 "may write /proc, where %s %s writable. Run mode cpu, or run as another account, or give --unconfined to run the rows "
                 "without the sandbox: each row may then write and remove any file on the machine and signal any process"
                 % (printable(why), printable(", ".join(writable)), "is" if len(writable) == 1 else "are"))  # fmt: skip
        named = ([("the card's lock (GPU_LOCK) " + printable(the_lock), the_lock)] if os.environ.get("GPU_LOCK") else []) + [
            ("GPU_PREFIX's " + printable(w), os.path.join(HERE, w.split("=")[-1])) for w in os.environ.get("GPU_PREFIX", "").split() if "/" in w]
        for what, path in named if kind == confine.LAID and on_gpu else ():
            if on_shared_memory(nearest_folder(path)):  # also a lock that does not exist yet: flock makes it there
                stop("%s is in /dev/shm, but as root each row gets its own empty /dev/shm, so each row would make and lock its own copy, "
                     "and the lock would not stop two rows from using the card at the same time. Name a lock outside /dev/shm; nothing was run" % what)  # fmt: skip
    open_above_lock = None
    if os.environ.get("GPU_LOCK") and on_gpu:
        if re.search(r"[^A-Za-z0-9_./-]", the_lock):
            stop("GPU_LOCK: the lock's path may hold letters, digits and / . _ - only (scripts put it before programs unquoted)")
        if the_lock == work or the_lock.startswith(work + "/"):
            stop("GPU_LOCK: the lock may not lie in the work folder, which must be new")
        # The lock's folder is checked like the work folder's parent, before anything is created
        # there: another account that could change it could put its own file, or a link, in the
        # lock's place.
        found = others_can_change(os.path.dirname(the_lock))
        if found and (not vouched or found[1] in (GONE, UNSEEN)):
            place, why = found
            if why == GONE:
                stop("the card's lock (GPU_LOCK) %s lies below %s, which does not exist: make that folder first, or name "
                     "a lock elsewhere; nothing was run" % (printable(the_lock), printable(place)))  # fmt: skip
            if why == UNSEEN:
                stop("the card's lock (GPU_LOCK) %s lies below %s, which could not be examined: name a lock in a folder "
                     "that you can look at; nothing was run" % (printable(the_lock), printable(place)))  # fmt: skip
            stop("the card's lock (GPU_LOCK) %s lies below %s, which another account may be able to change (%s): it could "
                 "put a file of its own in the lock's place, and two runs would then meet on the card. Name a lock in a "
                 "folder that only you or root can change, or in one with the sticky bit, such as /tmp (a lock there may "
                 "serve the runs of one account only), or, if you trust every account on this machine, say so with %s; "
                 "nothing was run" % (printable(the_lock), printable(place), whose(place, why), TRUST_EVERY_ACCOUNT))  # fmt: skip
        open_above_lock = found  # the option accepted it: the first and the last line say so
        make_lock(the_lock)  # before any folder is made, so that a wrong GPU_LOCK makes none
    # The rows start in the repository and run its programs by path. The check starts at its parent,
    # so that a clone of mode 775 under a private group causes no warning.
    open_at_tree = others_can_change(os.path.dirname(HERE))
    if not open_at_tree and os.stat(HERE).st_mode & 0o002 and not os.stat(HERE).st_mode & stat.S_ISVTX:
        open_at_tree = HERE, OPEN  # the tree itself, where every account may write (a group's write bit alone is quiet)
    real_work = os.path.join(os.path.realpath(os.path.dirname(work)), os.path.basename(work))
    if shared_from is not None and (real_work + "/").startswith(os.path.realpath(shared_from).rstrip("/") + "/"):
        stop("--shared-from: %s holds the work folder: name a folder outside it" % printable(os.path.abspath(shared_from)))
    def unlisted(fault):  # a folder that cannot be listed is rejected before the work folder exists
        stop("--shared-from: %s could not be looked at (%s): nothing was run" % (printable(fault.filename or ""), fault.strerror))

    for top, folders, files in os.walk(shared_from, onerror=unlisted) if shared_from is not None else ():  # before the work folder is made
        for where in (os.path.join(top, n) for n in folders + files):
            try:
                kind = os.lstat(where).st_mode
            except OSError as fault:  # gone or closed since it was listed
                stop("--shared-from: %s could not be looked at (%s): nothing was run" % (printable(where), fault.strerror))
            if not (stat.S_ISREG(kind) or stat.S_ISDIR(kind)):
                stop("--shared-from: %s is neither a plain file nor a folder (a link, for example): nothing was run"
                     % printable(where))  # fmt: skip
    open_at_shared = shared_note(os.path.abspath(shared_from), vouched) if shared_from is not None else None
    made, opened = make_work(work)
    if not os.environ.get("BUILD"):  # the GPU programs' build folder, in the work folder, so that what they build stays out of the tree's prover/build
        os.mkdir("build", 0o700, dir_fd=opened["work"])
    # For the first and the last line of output: the level of confinement, where rows that build may
    # also write, GPU_PREFIX, which may pass a row to another machine or process, and each open
    # place that TRUST_EVERY_ACCOUNT accepted, or else the fact that the option was given but not
    # needed.
    ran_how = (("confined: %s; rows that build may also write in %s" % (printable(level), build) if level else "UNCONFINED (--unconfined)")
               + ("; GPU_PREFIX: " + os.environ["GPU_PREFIX"] if on_gpu and os.environ.get("GPU_PREFIX") else "")
               + "".join("; %s lies below %s, which other accounts could change: the caller said that every account on this "
                         "machine is trusted (%s)" % (what, printable(found[0]), TRUST_EVERY_ACCOUNT)
                         for what, found in (("the work folder", open_above_work), ("BUILD", open_above_build),
                                             ("the card's lock (GPU_LOCK)", open_above_lock)) if found)
               + ("; the folder given with --shared-from holds or lies below %s, which other accounts could change: the caller "
                  "said that every account on this machine is trusted (%s)" % (printable(open_at_shared[0]), TRUST_EVERY_ACCOUNT)
                  if open_at_shared else "")
               + ("; %s was given, though no place on the way needed it" % TRUST_EVERY_ACCOUNT
                  if vouched and not (open_above_work or open_above_build or open_above_lock or open_at_shared) else ""))  # fmt: skip
    print("run_suite.py: the rows run " + ran_how, flush=True)  # fmt: skip
    if tried and tried[0] == confine.UNMADE and os.path.isdir("/dev/shm"):  # the first line says it too, in the middle of much else
        print("run_suite.py: running as root, and the machine refused the rows a mount namespace of their own, so every row shares "
              "/dev/shm with whatever else runs here and can remove, overwrite or empty any account's files there", flush=True)  # fmt: skip
    note([found for found in (open_at_tree,) if found])
    dropped = sorted(n for n in os.environ if STEERS.match(n))
    if dropped:
        print("run_suite.py: these environment variables are not passed on to the rows: " + " ".join(dropped), flush=True)
    if shared_from is not None:
        into = os.open("shared", FOLDER_HELD, dir_fd=opened["work"])
        try:
            laid = lay_folder(os.path.abspath(shared_from), into, vouched)
        finally:
            os.close(into)
        print("run_suite.py: %d file%s laid in $S from %s (--shared-from)" % (laid, "" if laid == 1 else "s", printable(os.path.abspath(shared_from))), flush=True)  # fmt: skip
    env = dict({n: v for n, v in os.environ.items() if n not in dropped and n not in UNSET and not n.startswith("BASH_FUNC_")}, T=HERE, S=work + "/shared", WORD=word)
    env["B"] = build
    env.pop("CONFINE_AS_ROOT", None)
    if tried and tried[0]:  # tools/confine.py refuses a row whose setup as root differs from what the first line reported
        env["CONFINE_AS_ROOT"] = tried[0]
    # where the user's own Python packages are (~/.local, by the caller's HOME), kept in reach when a row gets a HOME of its own
    user_base = env.get("PYTHONUSERBASE") or (site.getuserbase() if os.path.isdir(site.getuserbase() or "") else None)
    # The usage text lists where a confined row may write. bin/ is inside the repository: a gap in
    # the confinement, which builds need: they write their output there.
    bins = [os.path.join(HERE, "bin")]  # verifier/build.sh's default output folder
    if confined:
        unmade = []
        for folder in [env["B"]] + bins:
            try:
                own_bin(folder) if folder in bins else os.makedirs(folder, exist_ok=True)
            except OSError:  # a tree that cannot be written: the rows that build there fail by themselves
                unmade.append(folder)
        if unmade:
            print("run_suite.py: these folders cannot be made, so no row may write there (a tree that cannot be written?): "
                  + " ".join(unmade), flush=True)  # fmt: skip
    prefix = os.environ.get("GPU_PREFIX", "")
    # The GPU lock (the usage text, "gpu"). Created here, because a confined row's flock cannot
    # create it.
    lock = the_lock
    if on_gpu and not os.environ.get("GPU_LOCK"):
        make_lock("gpu.lock", opened["work"])  # GPU_LOCK's file was created, or rejected, above
    gpu_words = "flock " + lock + (" " + prefix if prefix else "")  # unquoted: GPU_RUN is expanded unquoted in scripts
    try:  # the file that the rows' flock will open, by device and inode: `card_free` takes no other
        lock_ident = (lambda about: (about.st_dev, about.st_ino))(os.stat(lock)) if on_gpu else None
    except OSError:
        lock_ident = None
    if on_gpu and lock_ident is None:
        lock = None  # no lock file to rely on: `card_free` is not called; no GPU row gets a signal

    # done: row's name -> (result, seconds, note); the result is one of passed, in part, FAILED,
    # not run, ACCEPT, taken. said: held while a result is printed. ended: one event for each row,
    # set once `done` has its result; the rows that need it wait for that.
    slots, done, said = Slots(jobs), {}, threading.Lock()
    lacking = {}  # rows that did not run only because a test said it could not run here: row -> the rows that said so
    memory = memory_limit()  # for the note on a row that SIGKILL ended
    ended = {n: threading.Event() for n in chosen}
    began = time.monotonic()
    running, starting, over = {}, threading.Lock(), []  # running: leader -> row's name; over: not empty after a signal
    on_card = []  # (row, group) of each GPU row left running at its end or limit, without a signal
    at_limit = {}  # row -> handle of a GPU row left running at its limit: its status is read later
    ended_free = set()  # GPU rows ended at their limit while they still waited for the GPU lock

    def still_runs(name, group):
        print("run_suite.py: A GPU PROGRAM STILL RUNS: row %s, process group %d. It got no signal, because ending a program "
              "while its GPU kernel runs can make the card's driver take the whole machine out of use. It holds the card, "
              "and what it would remove at its end stays, until it ends by itself; nvidia-smi shows it." % (name, group),
              flush=True)  # fmt: skip

    def look():
        """Prints the lines of the chosen rows' output files that ALARMS matches, each after the word LOOK. Returns their
        number. An output file that exists but cannot be read counts as one such line; a missing one is skipped."""
        looks = 0
        for name in sorted(chosen):
            path = "%s/screens/%s.txt" % (work, name)
            try:
                text = read_screen(opened, name)
            except FileNotFoundError:
                continue
            except OSError as fault:  # counted, so that the run cannot pass
                looks += 1
                print("LOOK     %s: cannot be read as a file: %s" % (path, fault.strerror))
                continue
            for n, line in enumerate(io.StringIO(text, newline=None), 1):  # split as in text mode
                if ALARMS.search(line):
                    looks += 1
                    print("LOOK     %s:%d: %s" % (path, n, printable(line.strip()[:160])))
        return looks

    def accepts():
        return sum(d[0] == "ACCEPT" for d in list(done.values()))

    def loud():
        """Prints what a status 3 means and what to do, if a row has ended with it."""
        if accepts():
            print("run_suite.py: SOMETHING FALSE WAS ACCEPTED: rows that ended with status 3 (ACCEPT above): %d. Keep %s and this "
                  "folder's bin/ as they are, and report it: SECURITY.md in this folder says how." % (accepts(), work), flush=True)  # fmt: skip

    heard = [s for s in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGHUP)
             if s != signal.SIGHUP or signal.getsignal(s) != signal.SIG_IGN]  # fmt: skip

    # GPU rows left running at their time limit that had ended with status 3 when a signal came
    late_at_stop = []

    def ended_from_outside(sign, _frame):
        for s in heard:  # a second signal must not interrupt this
            signal.signal(s, signal.SIG_IGN)
        try:
            with starting:
                over.append(sign)
                names = dict(running)
                leaders = list(names)
            on_gpu = [x for x in leaders if rows[names[x]]["kind"] in on_the_gpu]
            end_the_groups([x for x in leaders if x not in on_gpu], 3)
            # A row of kind gpu that still only waits for the GPU lock is ended (see card_free()).
            # Any other GPU row gets no signal here either: it is named and left, if still alive.
            freed = [x for x in on_gpu if rows[names[x]]["kind"] == "gpu" and lock and card_free(
                lock, lambda x=x: end_if_waiting(x, 3), ident=lock_ident)]  # fmt: skip
            live = live_groups()
            left = [x for x in on_gpu if x not in freed and x.returncode is None and x.pid in live]
            for leader in left:
                still_runs(names[leader], leader.pid)
            for leader in leaders:  # a status 3 that is not recorded yet: ask the unreaped leader
                if end_of(leader, 0) == 3:
                    done.setdefault(names[leader], ("ACCEPT", 0.0, "status 3, seen when the signal came"))
            for name, run in list(at_limit.items()):  # a GPU row left at its time limit that has since ended with status 3
                if end_of(run, 0) == 3:
                    late_at_stop.append(name)
                    print("run_suite.py: SOMETHING FALSE WAS ACCEPTED: row %s, left running at its time limit, has since ended "
                          "with status 3. Keep %s and this folder's bin/ as they are, and report it: SECURITY.md in this folder says how." % (name, work), flush=True)  # fmt: skip
            looks = look()
            loud()
            print("run_suite.py: ended by signal %d; %d running rows were ended with it and %d on the GPU were left running; "
                  "%d rows had ended with status 3; %d lines to look at; the rows ran %s"
                  % (sign, len(leaders) - len(left), len(left), accepts(), looks, ran_how), flush=True)  # fmt: skip
        finally:  # also if the print fails, as on a pipe whose reader the same signal has ended
            os._exit(3 if accepts() or late_at_stop else 1)

    for s in heard:
        signal.signal(s, ended_from_outside)

    # Runs one row, in its own thread: waits for the rows that it needs, then for its slots, and
    # records and prints the result.
    def play(name):
        row = rows[name]
        for need in row["needs"] if name not in taken else []:
            ended[need].wait()
        if name in taken:
            done[name] = ("taken", 0.0, "not run: --taken")
        elif any(done[need][0] not in ("passed", "in part", "taken") for need in row["needs"]):
            unmet = [need for need in row["needs"] if done[need][0] not in ("passed", "in part", "taken")]
            if all(need in lacking for need in unmet):  # those rows said why they did not run: this one cannot run either
                lacking[name] = sorted({said for need in unmet for said in lacking[need]})
                named = [said if said in unmet else said + " (through another row)" for said in lacking[name]]
                one = len(named) == 1
                done[name] = ("not run", 0.0, "not run: it needs %s, which did not run and %s why in %s last line%s"
                              % (" and ".join(named), "says" if one else "say", "its" if one else "their", "" if one else "s"))  # fmt: skip
            else:
                done[name] = ("not run", 0.0, "a row that it needs did not pass")
        elif where_now(work, made, opened)[0]:  # rows go by path: not started if it leads elsewhere
            done[name] = ("not run", 0.0, "the work folder was replaced or removed before it started")
        elif on_card and row["kind"] in on_the_gpu:  # that program still holds the GPU and its lock
            done[name] = ("not run", 0.0, "the GPU program of %s still runs on the card" % on_card[0][0])
        else:
            alone = row["kind"] == "gpu" or (row["kind"] == "gpu-by-run" and not prefix)
            command = row["command"] if row["kind"] != "gpu" else gpu_words + " bash -c " + quoted(row["command"])
            own = dict(env, W="%s/%s" % (work, name), **({"GPU_RUN": gpu_words} if row["kind"] == "gpu-by-run" else {}))
            words = ["bash", "-c", command]
            if confined:  # before `slots.take`, so that an exception here leaves no slots taken
                os.mkdir(name, 0o700, dir_fd=opened["work"])  # own["W"], made in the folder that this run holds
                builds = row["kind"] == "nvcc" or name.startswith("build.")
                also = [env["S"]] + bins + ([env["B"]] if builds else [])  # $S, which many rows fill; bin/, for rows that build there
                also = [a for a in also if os.path.isdir(a)]  # confine.py rejects a missing folder
                own["CUDA_CACHE_PATH"] = os.path.join(own["W"], "cuda-cache")  # not the account's ~/.nv
                own["TMP"] = own["TEMP"] = os.path.join(os.path.realpath(own["W"]), "tmp")  # the TMPDIR that tools/confine.py gives the row
                os.mkdir(name + "/home", 0o700, dir_fd=opened["work"])  # a HOME of its own, which the sandbox lets it write
                own["HOME"] = os.path.join(os.path.realpath(own["W"]), "home")
                words = [sys.executable, "-B", CONFINE, own["W"]] + [x for a in also for x in ("--also", a)] + (["--gpu"] if row["kind"] in on_the_gpu else []) + words
                own["W"] += "/w"
            else:  # not the caller's TMPDIR, where another account may write, nor the account's ~/.nv
                os.mkdir(name, 0o700, dir_fd=opened["tmp"])
                own["TMPDIR"] = own["TMP"] = own["TEMP"] = "%s/tmp/%s" % (work, name)
                own["CUDA_CACHE_PATH"] = own["TMPDIR"] + "/cuda-cache"
                os.mkdir(name + "/home", 0o700, dir_fd=opened["tmp"])
                own["HOME"] = own["TMPDIR"] + "/home"
            own["XDG_CACHE_HOME"], own["XDG_CONFIG_HOME"] = own["HOME"] + "/.cache", own["HOME"] + "/.config"
            own["XDG_DATA_HOME"], own["XDG_STATE_HOME"] = own["HOME"] + "/.local/share", own["HOME"] + "/.local/state"
            if user_base and not own.get("PYTHONUSERBASE"):  # an empty one counts as unset, as for Python
                own["PYTHONUSERBASE"] = user_base
            held = slots.take(row["slots"], alone)
            with starting:  # the last check, after the wait for slots: a swap or a GPU program left running meanwhile counts
                late = None if over else (
                    "the work folder was replaced or removed before it started" if where_now(work, made, opened)[0]
                    else "the GPU program of %s still runs on the card" % on_card[0][0] if on_card and row["kind"] in on_the_gpu
                    else None)  # fmt: skip
            if late:
                slots.give(held, alone)
                done[name] = ("not run", 0.0, late)
            else:
                start = time.monotonic()
                try:
                    with os.fdopen(os.open(name + ".txt", NEW_FILE, 0o600, dir_fd=opened["screens"]), "wb") as f:
                        with starting:
                            run = None if over else subprocess.Popen(
                                words, cwd=HERE, env=own, stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
                            if run:
                                running[run] = name
                        if not run:
                            threading.Event().wait()  # a signal has arrived: the program is about to exit
                        least = 60 if (prefix or os.environ.get("GPU_LOCK")) and row["kind"] in on_the_gpu else 10
                        status = end_of(run, 60 * max(least, 5 * row["minutes"]))
                        if status == 3:  # at once: a signal may come before the result is recorded
                            done[name] = ("ACCEPT", time.monotonic() - start, "status 3")
                        still = []
                        if run.returncode is None:  # not reaped, so its group's number is still this row's
                            if row["kind"] == "gpu" and status is None and lock and card_free(
                                    lock, lambda: end_if_waiting(run, 20), ident=lock_ident):
                                ended_free.add(name)  # nothing of the row had run on the GPU
                            elif row["kind"] in on_the_gpu:
                                # Any other row on the GPU at its time limit, and what a row on the
                                # GPU left running in its group, gets no signal (the usage text
                                # gives the reason). A group that is still alive after ten seconds
                                # is reported and left running.
                                until = time.monotonic() + 10  # a process that is already on its way out gets a moment
                                while run.pid in live_groups() and time.monotonic() < until:
                                    time.sleep(0.2)
                                still = [run] if run.pid in live_groups() else []
                                if status is None and still:
                                    at_limit[name] = run  # its status is read again later
                            else:
                                end_the_groups([run], 20)
                        with starting:
                            reap = not over and not still  # after a signal the handler may still signal this group: it stays unreaped
                            if reap or still:
                                running.pop(run, None)
                            if still:
                                on_card.append((name, run.pid))
                        if reap:
                            run.wait()
                finally:
                    slots.give(held, alone)
                try:
                    text, unreadable = read_screen(opened, name), None
                except OSError as fault:  # something else was put in its place: the row cannot pass
                    text, unreadable = "", fault.strerror
                shown = [x for x in text.splitlines() if x.strip()] or [""]
                last = shown[-1]
                unmet = [wanted for wanted in row["lines"] if not any(re.search(wanted, x) for x in shown)]
                good = status in row["statuses"] and (not row["last"] or re.search(row["last"], last)) and not unmet and not unreadable
                what = "ACCEPT" if status == 3 else "FAILED" if not good else "passed"
                if good and (status == 77 or last.startswith("not run")):
                    what = "not run" if last.startswith("not run") else "in part"
                seen = (("stopped at its time limit, while nothing of it was on the card (the runner held the card's lock)"
                          if name in ended_free
                          else "at its time limit, left running (no signal is sent on the GPU); its own status is not known yet"
                          if row["kind"] in on_the_gpu else "stopped at its time limit")
                        if status is None else "status %d; %s" % (status, last[:100]))  # fmt: skip
                if any(n == name for n, _ in on_card):  # an ACCEPT stays one: what was accepted must not read as a failure
                    good, what = False, "ACCEPT" if what == "ACCEPT" else "FAILED"
                    seen += "; ITS GPU PROGRAM STILL RUNS (process group %d): see the last lines" % dict(on_card)[name]
                if status == LOST:
                    seen = ("how it ended is not known: its exit status was collected before run_suite.py read it. It counts as "
                            "failed; it may have been status 3, so read its screen; what it left running was not ended; ") + last[:100]
                if status in row["statuses"] and row["last"] and not re.search(row["last"], last):
                    seen += "; the last line does not match " + row["last"]
                seen += "; no line matches " + unmet[0] if unmet else ""
                seen += "; its output file cannot be read as a file: " + unreadable if unreadable else ""
                if what == "FAILED" and (status == -signal.SIGKILL or (status == 128 + signal.SIGKILL and status not in row["statuses"])):
                    # SIGKILL ended the row or (bash's status 137) its last program, as the system ends one that runs out of
                    # memory
                    seen += ("; it was killed by signal 9" if status < 0 else "; status 137 usually means that a program in it was "
                             "killed by signal 9") + ", perhaps by the system for want of memory" + (
                        ": this machine or its control group has %.1f GiB for all the rows that run at once, and this row asks for %d "
                        "job slot%s, about %d GiB; run it alone (--only) or on a machine with more memory"
                        % (memory / (1 << 30), row["slots"], "" if row["slots"] == 1 else "s", 2 * row["slots"])
                        if memory else "")  # fmt: skip
                if what == "not run":  # the row itself says why it did not run (and was not turned FAILED above)
                    lacking[name] = [name]
                done[name] = (what, time.monotonic() - start, seen)
        with said:
            print("%-8s %-44s %7.1f s  %s" % (done[name][0], name, done[name][1], printable(done[name][2])), flush=True)
        ended[name].set()

    def played(name):
        try:
            play(name)
        except Exception as fault:  # this program's fault: the row has failed, and rows that wait for it must not hang
            if done.get(name, ("",))[0] != "ACCEPT":  # for example, its line could not be printed
                done[name] = ("FAILED", 0.0, "an error in run_suite.py itself: %r" % fault)
                lacking.pop(name, None)  # a failed row did not only say why it could not run
            try:
                with said:
                    print("%-8s %-44s %7.1f s  %s" % (done[name][0], name, done[name][1], printable(done[name][2])), flush=True)
            finally:
                ended[name].set()

    try:
        # The longest first, so that the last row to end is a short one.
        order = sorted(chosen, key=lambda n: -rows[n]["minutes"])
        with concurrent.futures.ThreadPoolExecutor(len(order)) as pool:
            list(pool.map(played, order))

        replaced, shown_at = where_now(work, made, opened)
        if replaced:
            say_replaced(work, shown_at)
        looks = look()
        late = []  # a GPU row left at its time limit (counted as failed) that has since ended with status 3
        for name, run in at_limit.items():
            if end_of(run, 0) == 3:
                late.append(name)
                print("run_suite.py: SOMETHING FALSE WAS ACCEPTED: row %s, left running at its time limit, has since ended with "
                      "status 3. Keep %s and this folder's bin/ as they are, and report it: SECURITY.md in this folder says how." % (name, work), flush=True)  # fmt: skip
        count = {k: sum(d[0] == k for d in done.values()) for k in ("passed", "in part", "FAILED", "not run", "ACCEPT", "taken")}
        # the rows that passed in part and whose last line begins with ASIDE (see the usage text)
        aside = sum(d[0] == "in part" and d[2].split("; ", 1)[-1].startswith(ASIDE) for d in done.values())
        for name in sorted(left_out):
            print("left out %-44s what it tests does not exist under version word %s" % (name, word))
        for name in sorted(needs_the_gpu):
            print("left out %s: it needs a row that needs nvcc or a GPU (%s); mode all runs it" % (name, needs_the_gpu[name]))
        live = live_groups()
        for name, group in on_card:  # named again only while it still runs
            if group in live:
                still_runs(name, group)
        loud()
        print("run_suite.py: %d rows passed, %d passed in part (status 77: their last lines say what was not run%s), %d failed, "
              "%d not run%s, %d left out under version word %s, %d left out in mode cpu because they need nvcc or a GPU, "
              "%d taken and not run (--taken) and %d that only these need; "
              "%d ended with status 3; %d lines to look at; %.0f s; the screens are in %s/screens; the rows ran %s; "
              "the call chose %d of the table's %d rows"
              % (count["passed"], count["in part"], "; %d of them stood aside below an open place: name a work folder below your "
                 "home folder or /tmp to run them" % aside if aside else "", count["FAILED"], count["not run"],
                 " (%d of them only because a test said it could not run here)" % len(lacking) if lacking else "", len(left_out), word, len(needs_the_gpu), count["taken"], gone,
                 count["ACCEPT"], looks, time.monotonic() - began, shown_at or work, ran_how,
                 len(chosen), len(rows)), flush=True)  # fmt: skip
        not_run_otherwise = any(d[0] == "not run" and n not in lacking for n, d in done.items())
        return 3 if count["ACCEPT"] or late else 1 if count["FAILED"] or not_run_otherwise or looks or replaced else 77 if lacking else 0

    except Exception:
        if not accepts():
            raise
        # This program failed (for example, its standard output was closed) and a row ended with status 3: the status
        # must still be 3, and what the run has printed must not be lost.
        try:
            try:
                sys.stdout.flush()
            except Exception:  # standard output may be what failed
                pass
            sys.stderr.write(traceback.format_exc())
            print("run_suite.py: a row ended with status 3, which means that something false was accepted, and this program "
                  "also failed (the error is above). Keep %s and this folder's bin/ as they are, and report it: SECURITY.md in this folder says how." % work,
                  file=sys.stderr, flush=True)  # fmt: skip
        finally:
            os._exit(3)


def quoted(text):
    return "'" + text.replace("'", "'\\''") + "'"


def own_bin(folder):
    """Makes the tree's bin/ if it is missing, as os.makedirs(folder, exist_ok=True) does; root changes no owner. The note
    ROOT_MADE_BIN is printed where bin/ is a folder of root's in a tree whose top folder another account owns (as under
    sudo, in that account's own copy of the repository): as root, when this call has just made it and no group or
    other account can write in it; not as root, when this account cannot write in it."""
    made = not os.path.isdir(folder)
    if made:
        os.makedirs(folder, exist_ok=True)
    top = os.path.dirname(folder)
    try:
        owner, about = os.stat(top), os.lstat(folder)
        if owner.st_uid == 0 or not stat.S_ISDIR(about.st_mode) or about.st_uid != 0:
            return
        if os.geteuid() == 0:
            # root cannot ask os.access for another account, so: only a new folder that gives its group and others no write
            # (where the folder has an access list, its group bits are that list's mask)
            if not made or about.st_mode & 0o022:
                return
        elif os.access(folder, os.W_OK):
            return
    except OSError:  # the note is only a help
        return
    try:
        print(ROOT_MADE_BIN % {"folder": printable(folder), "top": printable(top), "uid": owner.st_uid}, flush=True)  # fmt: skip
    except (UnicodeEncodeError, OSError):  # a path that standard output's encoding cannot hold, or standard output gone
        pass


if __name__ == "__main__":
    sys.exit(main())
