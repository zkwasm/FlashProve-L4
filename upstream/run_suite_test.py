#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests run_suite.py, the program that runs the tests listed in SUITE.md, mostly by running a copy
of it on small SUITE.md tables that this test writes. No row of those tables needs a GPU or nvcc,
whatever its kind. The checks are on what users of run_suite.py rely on: which environment variables
reach a test, when a test counts as passed, that neither a call that runs nothing nor a run with a
status 3 ends like a passed run, the GPU lock, that a row on the GPU is sent no signal while it may
run on the card, the time limit, the confinement, and who can write in a run's folders. Give it a folder that does not exist
yet, outside this repository and not below a folder that another account can change: it creates the
folder, with mode 700, and never removes it. It sends signals only to processes that it started. In
the folder, tree/ ("the copy's tree") holds the copy, tools/confine.py and one parameter file,
policy.txt.
A row is one line of a SUITE.md table: a test or a set-up step. A row on the GPU is one of kind gpu
or gpu-by-run. <work> is the work folder that a call of the copy names. A confined row runs under
tools/confine.py, a sandbox that uses Linux's Landlock. The top of run_suite.py explains the cells,
options, variables and output lines below.

usage: run_suite_test.py <new folder outside this folder>
Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The folder: as described above.)

  parsed           bash -n reads the command of every row of this repository's SUITE.md, and of
                   further tables next to it if there are any (see "two tables"), without an error.
                   One unmatched quotation mark would make bash reject a whole command, and
                   run_suite.py --list would not notice. No command is run

run_suite.py drops from the rows' environment the variables by which tests steer programs and
scripts (those that its regular expression STEERS matches), so that the user's shell cannot change
what a row tests. A row that needs one sets it in its command.

  by text          every variable that one of the commands of "parsed" sets before a program (SET in
                   this file) matches STEERS, unless it is in PASSED_ON or OWN in this file; nothing
                   in PASSED_ON matches
  the programs' list
                   STEERS matches every string in the GPU programs' list of test variables
                   (test_word_in_the_environment(), prover/common/test_words.h, read as text). A
                   prefix that ends in "_" is tried with an X added; the prefix DEAD_REQUEST only as
                   it stands
  dropped          the copy runs two rows while the caller sets every dropped variable (those that
                   "by text" found, and ALSO in this file), PASSED_ON's, except PATH and C, and
                   CALLERS_OWN's. CALLERS_OWN, in this file, has variables that change what a shell
                   does, such as BASH_ENV, LD_PRELOAD, one exported bash function, PYTHONOPTIMIZE
                   (which takes out Python's assert statements), and variables that give access to
                   the caller's services, such as SSH_AUTH_SOCK. The first row
                   gets none of the dropped ones, none of CALLERS_OWN's, and PASSED_ON's as the
                   caller set them, except TMPDIR, which for an unconfined row is the row's own
                   folder below <work> (see "temporary files")
  said             that run's second line lists exactly the dropped variables
  in the command   the second row sets a dropped variable in its command, and its program gets that
                   value

A cell "passes" can hold, after its expression for the last line of output, further expressions,
each after " ;; ". Each must match some line of the output.

  some line        of four rows that print the same three lines, the one whose expressions all match
                   passes. Each of the others requires one thing that is not so (a line with another
                   number; an earlier line as the last; a second further line that is not printed)
                   and fails. For the first two, the row's line in the output names the expression
                   that did not match

A valid call that chooses no row must not end with status 0, which a script would take for a passed
run. Below, "rejected" means: status 2, nothing on standard output, one line on standard error, and
no work folder. Each of the next six checks runs the copy on those four rows, which are of kind cpu,
and expects it to be rejected with a line that begins "run_suite.py: nothing was run", unless its
entry says otherwise.

  empty only       --only with an expression that matches no row's name
  empty share      --share 5/5: the four rows go to the first four shares. As a control, --share 1/5
                   runs one row, which passes, with status 0; that run's first and last lines say
                   UNCONFINED, and the last that the call chose 1 of the table's 4 rows
  empty kind       the argument gpu, which chooses none of the four rows
  no expression    --only with an unmatched bracket; the line begins "run_suite.py: --only: "
  empty list       --list with an expression that matches no row: status 0, no work folder, and one
                   printed line, which counts 0 rows
  all left out     the rows' cell "word" is 2, and the copy's policy.txt has version word 1, under
                   which such rows are left out: the line counts four rows as left out

For a test that SUITE.md lists, status 3 means that something false was accepted, and status 77
that something cannot be run on this machine (run_suite.py, "passes" and "Status 3").

  no row with 3    a one-row table whose cell "passes" allows status 3 (written 3 or 03), has a
                   malformed list of statuses, or has an invalid regular expression: rejected, and
                   the line names the table's line
  an accept        seven rows. Two end with status 3, one of them after a line with the words WAS
                   NOT REFUSED: both are reported as ACCEPT, one line reports that something false
                   was accepted in two rows, the run ends with status 3, and the folder of the row
                   that printed the words is still there. A row that needs one of the two is not
                   run. A row that allows status 77 and whose last line begins "not run" counts as
                   not run, whether it ends with 77 or 0. The line with WAS NOT REFUSED, and a
                   closing count of 10 such lines, are printed after the word LOOK; a count of 0 is
                   not
  not run, said why
                   a row whose last line is "not run: …" (status 77), a row that needs it and a row
                   that needs that one: all three
                   not run, the closing line counts them as not run only because a test said it
                   could not run here, each row that needs it, directly or through another row,
                   names the row that said so, and the run ends with status 77; with a failed row
                   beside them, with status 1
  alarms           ALARMS in run_suite.py, which joins the regular expressions of ALARM_WORDS, finds
                   the lines that are printed after LOOK. For each entry of ALARM_LINES in this
                   file, the named file contains the entry's pieces of text, and the pieces put
                   together match ALARMS. ALARMS matches no line of QUIET in this file. So the check
                   fails if such a message is reworded in a test, after which run_suite.py would no
                   longer find it. Further alarm tables, if there are any (tests/alarm_lines.<word>.tsv
                   whose first line is ALARM_FURTHER), add entries; a file of that name without that
                   line fails the check. Only with such a table must each expression of ALARM_WORDS
                   match at least one entry: some match none of ALARM_LINES. Nothing is run
  an accept, then a signal
                   one row ends with status 3 while another sleeps. After the first row's line the
                   copy gets SIGTERM: it still reports that something false was accepted, and ends
                   with status 3
  a lost status    a child that ended with status 0 is reaped before end_of() in run_suite.py asks
                   for its status, as happens when SIGCHLD is ignored. end_of() returns LOST, which
                   is negative, so that no cell "passes" can allow it, and stores it in the child's
                   handle (a Popen object); end_the_groups() then sends no signal (os.killpg is
                   replaced by a recorder). Not shown: that run_suite.py reports such a row as
                   FAILED

  taken            three rows: the second needs the first, and the third the second. --taken names
                   the second. With --only for the third, or without --only, the second is printed
                   as taken, only the third runs, and the output counts one row as taken and one
                   that only it needs; in the first of these calls no line may report that something
                   false was accepted. With --only for the second, the first two run, since no row
                   to be run needs the second. An empty --taken is rejected
  side by side     commands that use the GPU run under "flock <lock>"; <lock> is the GPU lock ("the
                   card's lock" in messages). A row of kind gpu-by-run gets these words, then
                   GPU_PREFIX's, as GPU_RUN, to put before its GPU programs. Two such rows run with
                   two job slots; one waits for a file that the other creates. With GPU_PREFIX set
                   it waits up to 60 seconds: both pass, so they ran side by side (or the other row
                   ran first, which is not told apart), and the first line of output names the
                   prefix. Without it they run one at a time: the waiting row fails after five
                   seconds or, if the other row ran first, passes and has recorded that the file was
                   there when it began. To pass, a row must also have got GPU_RUN as described
  the card's lock  a row of kind gpu prints its parent's command line, which must begin
                   "flock <lock> bash", and a row of kind gpu-by-run puts $GPU_RUN before a program,
                   unquoted, as scripts do. Both pass, and run_suite.py has created <lock>: first
                   the run's own gpu.lock, then the file that GPU_LOCK names. The second run is
                   confined where rows can be confined (tools/confine.py --level), so that its flock
                   could not create the file. Rejected: a GPU_LOCK in a folder that does not exist
                   or with a blank in its path, and a GPU_PREFIX that names the lock, directly or
                   through a symbolic link
  as root, a card's lock in /dev/shm
                   in a stand-in for root's rows (tools/confine.py is made to say that each row gets
                   a /dev/shm of its own, and nothing is mounted), a call of kind gpu with GPU_LOCK in
                   /dev/shm, with GPU_LOCK below a symbolic link to /dev/shm, with GPU_PREFIX
                   "flock /dev/shm", and with GPU_PREFIX naming a lock in /dev/shm that does not exist
                   yet: each rejected, with a line that says the lock would not stop two rows from
                   using the card at the same time, and no lock is made. Also rejected, with a line
                   that says root's rows would not see the folder and what to change: PYTHONUSERBASE or
                   VIRTUAL_ENV naming a missing folder in /dev/shm, and a PYTHONPATH of /dev/shm. Two
                   more cases in the same stand-in: where the machine refuses rows a mount
                   namespace of their own and /proc/sys is writable, a run that picks a row of kind gpu
                   is refused before anything is made; and where the namespace is set up, a row's command
                   sees no CONFINE_AS_ROOT, though the caller sets it and the runner sets it for confine.py;
                   and where the namespace is refused, the runner prints a line of its own saying that
                   every row shares /dev/shm (these last two only when the test does not run as root).
                   Where rows cannot be confined or there is no folder /dev/shm: not checked, and
                   printed as expected
  a backtick lost  two checks: a one-row table whose cell "passes", or "command", lacks its closing
                   backtick. Status 2, nothing on standard output, no work folder, and standard
                   error names the table's line
  what a row leaves
                   a row leaves a child (sleep 60) in its process group, writes the child's process
                   ID into its folder and ends. After the run that child is gone, because
                   run_suite.py ends the group of a row that is not on the GPU when the row ends
  the time limit   end_of() returns None for a child that outlives a limit of one second, and
                   end_the_groups() then ends the child's group with SIGTERM. Not shown: how long a
                   row's limit is (five times its minutes, and at least ten minutes, or an hour for
                   a row on the GPU if GPU_PREFIX or GPU_LOCK is set)
  confined         a default call, with neither --confined nor --unconfined, on one row that finds
                   no $W yet (a confined row's W is <work>/<row>/w), creates it, and tries to create
                   a file beside <work>/<row>. The row passes, the file is not created, the row's
                   HOME is <work>/<row>/home with XDG_CACHE_HOME, XDG_CONFIG_HOME, XDG_DATA_HOME and
                   XDG_STATE_HOME below it, and HOME and the cache can be written,
                   and the first and last lines of output state the level of confinement. Where rows cannot
                   be confined (a kernel without Landlock, for example): rejected, and the line
                   begins "run_suite.py: NOT CONFINED"
  a work folder on shared memory
                   a default (confined) call whose work folder lies in /dev/shm: rejected with
                   status 2 («a work folder in /dev/shm can't be kept apart …»), and no work folder
                   is made; with --unconfined it is taken. Where no folder can be made in /dev/shm
                   (as inside a confined run) or there is no confinement: not checked, and printed
                   as expected
  shared memory as a plain folder
                   on_shared_memory() where /dev/shm is stood in for as a plain folder on the work
                   folder's own file system: the test's own work folder does not count, /dev/shm
                   itself does
  a HOME of its own
                   an unconfined row, called with HOME set to a folder that holds .local and an
                   empty PYTHONUSERBASE: the row's HOME is <work>/tmp/<row>/home, and its
                   PYTHONUSERBASE is the caller's <HOME>/.local
  a link laid during the look
                   others_can_change() on a path through a symbolic link that it reads, and through
                   a link that lies below another link (its typed name is not the name the look
                   read): None. Each of these is refused with the place and the value UNSEEN ("it
                   could not be examined"): a link swapped for another trusted link right after the
                   look reads it (a stand-in readlink swaps it); a link swapped for one that a
                   stand-in lstat shows with the identity of the link that was read; a link that is
                   given a second name (a hard link) right after the look reads it, which changes
                   only its change time; and a name that is missing when the look first looks at
                   it and is then laid as a link to a folder of the test's own that only it can
                   write in (the stand-in racing_lstat lays it at once). The second-name case is
                   not checked where the file system's change times do not move within 20 ms on
                   two probes in a row; one line says so, and the check still counts as expected
  a tree that cannot be written
                   the copy's tree has a file named bin, so run_suite.py cannot create bin/. A
                   confined run reports that in a line, and its row passes: confine.py got only
                   folders that exist
  confined, no Landlock
                   a default call with a copy of tools/confine.py altered to find no Landlock:
                   rejected as in "confined"
  the build folder
                   with BUILD set but empty, a row's $B is <work>/build, as with BUILD unset.
                   run_suite.py has created it with mode 700 and nothing at the copy's prover/build,
                   and a confined run's first line names it. Rejected: BUILD=/, the folder above the
                   copy's tree, directly or through a symbolic link, and the work folder itself
  the GPU's places
                   a confined run of a row of kind cpu and a row of kind gpu. Each writes its
                   process's name through /proc, as CUDA does for its threads. In the first the
                   write fails. In the second it ends as in a bash that this test starts itself: it
                   succeeds, unless this test is itself a confined row of kind cpu
  closed folders   the work folder of "what a row leaves" and its folders screens, shared and tmp
                   have mode 700, and the row's output file has mode 600, though this test's umask
                   is 022
  a work folder that exists
                   the call names an empty folder that exists: status 2, nothing on standard output,
                   one line on standard error, and the folder stays empty
  a swapped work folder
                   a row moves the work folder away and creates a new one in its place. The row
                   passes, its output file is in the moved folder, the new folder stays empty, the
                   output says in capitals that the work folder was replaced or removed, and the run
                   ends with status 1. With one job slot, the row that needs it is not run, and a
                   third row either ran before it, in the folder that was then moved, or is not run
  the note         a folder of mode 777 without the sticky bit. A work folder in it is rejected: the
                   line names both folders and the reason, and advises chmod go-w unless the caller
                   is root. So is a BUILD in it. With --i-trust-every-account-on-this-machine both
                   runs pass, and the first line says that the caller trusts every account; for the
                   work folder the last line does too. note(), by which run_suite.py warns about the
                   tree that it runs from, names such a folder and its mode on standard error and
                   says to keep the tree below folders that only the caller or root can change. Not
                   shown: that a run calls note()
  a link as the card's lock
                   rejected, with a line that begins "run_suite.py: the card's lock": a GPU_LOCK
                   that is a symbolic link to nothing (the link's target is not created) or to a
                   regular file, and GPU_LOCK=/dev/null, of which the line says "is not a regular
                   file"
  BUILD refused    rejected, each with its own line: a relative BUILD, a folder that its group and
                   others can write in, and a name right below /
  temporary files  an unconfined row's TMPDIR, TMP and TEMP are <work>/tmp/<row>, and that folder exists, and its
                   CUDA_CACHE_PATH is <work>/tmp/<row>/cuda-cache, whatever the caller's
  the last line says how
                   a confined run with BUILD and GPU_PREFIX set: its first and last lines both state
                   the level of confinement, the folder where rows that build may also write, and
                   GPU_PREFIX's words
  a row on the GPU's work
                   six rows: one of kind gpu, one of kind nvcc, three of kind cpu that need one of
                   these two, directly or through another row, and one of kind cpu that needs none.
                   In mode cpu (the argument cpu) the row that needs none passes, the three are left
                   out, each in a line of its own, and counted apart, and the run ends with status
                   0. With --only for one of the three: rejected, and the line names the row that it
                   needs. In mode all, all six pass
  odd work folder names
                   a name with a part that begins with "-", a name that goes through a symbolic link
                   and "..", and the name of a symbolic link to nothing: each status 2, nothing on
                   standard output, its own line on standard error, and nothing created
  under a set-group-ID folder
                   a work folder created in a folder with the set-group-ID bit inherits the bit: the
                   run passes, and the work folder's other mode bits are 700
  a loose umask    the copy is called under umask 000: the folder and the file that an unconfined
                   row creates have modes 755 and 644, so that group and others cannot write in
                   them. Under umask 277, which takes away the owner's write bit, the same run still
                   passes
  a long row       a row of kind long needs a row of kind cpu: modes cpu and all run only the
                   second, and mode long runs both. Rejected: a table in which a row of kind cpu
                   needs one of kind long
  printed as escapes
                   a row's last line, and a line of its output that is printed after LOOK, contain
                   terminal control sequences: the copy prints their escape character as a Python
                   escape, never raw
  a folder above that does not exist
                   the work folder's parent does not exist: status 2, and the parent is not created
  a GPU row taken in mode cpu
                   a row of kind cpu needs a row of kind gpu that --taken names: in mode cpu the
                   second is printed as taken, and the first runs and passes
  shared from      --shared-from names a folder that holds one file, of mode 6755, in a folder of
                   its own. The copy copies it into $S with mode 755 (no set-ID bits) and reports one
                   file, and a row reads it there. Rejected: a folder that contains the work folder.
                   With a symbolic link in the folder: status 2, its own line, no work folder, and
                   no row is run
  shared from, open
                   a file of mode 666 below the folder (its group and others can write in it):
                   status 2, its line, no work folder; with --i-trust-every-account-on-this-machine
                   the run goes on, its first line names the file, and the copy in $S has mode 644
                   (no write bits for group and others). The folder itself of mode 1777 (sticky, as
                   /tmp): status 2. Given by a symbolic link: status 2, even with that option
  shared from, as copied
                   lay_folder() on a folder holding a file of mode 664, as if swapped in after the
                   look, on one holding a sub-folder of mode 775, and on a folder that is itself of
                   mode 775; as root also on one holding a file, and on one holding a sub-folder, of
                   another account's, and on a folder that is itself another account's: each status 2,
                   "changed after the runner checked it"; with the option that trusts every
                   account each is copied. shared_note() on a folder that is missing at its look:
                   status 2, "could not be examined …, though it was there a moment before"
  shared from, its cure
                   a file of mode 664 below the folder given with --shared-from, the caller's own,
                   looked at as an account that is not root: status 2, and the cure names
                   «chmod go-w» of that one file (its name quoted for the shell), not of the whole folder
  a stubborn row is killed
                   end_the_groups(), by which run_suite.py ends rows that are not on the GPU, on a
                   process group that ignores SIGTERM, with a grace of one second: SIGKILL ends it
  a GPU row at a stop
                   the copy gets SIGTERM while a row of kind gpu runs: the copy ends with status 1
                   and reports in capitals that a GPU program still runs, with the row's name and
                   its process group, and the copy's last line says the rows ran unconfined. Then that
                   group still has a member, and the row's trap has recorded no SIGTERM. This test then
                   creates the file the row waits for, so that the row ends by itself (lives_then_go())
  no GPU row after one left running
                   a row of kind gpu ends with status 3 and leaves a program in its group. The copy
                   reports it as in "a GPU row at a stop" and leaves it running, with no SIGTERM.
                   The row is still reported as ACCEPT, a later row of kind gpu is not run because
                   that program still runs on the card, and the run ends with status 3. This test
                   then lets the program end by itself
  a failed look    others_can_change() in run_suite.py looks along a path for a folder or link that
                   another account can change. A folder that it cannot examine must not pass: for a
                   path through a folder that does not exist, and for one through a name of 300
                   characters, it returns that folder and the reason (GONE, UNSEEN). For a work
                   folder below either, or below a missing folder inside an open one, the copy,
                   though called with --i-trust-every-account-on-this-machine, ends with status 2
                   and gives the reason, and it does not create the missing folder.
                   A row that ends with status 77 after a last line that begins "stood aside: below
                   an open place:" is reported as passed in part, the last line of output counts it
                   apart, and the run ends with status 0
  the card's lock  card_free() in run_suite.py takes the card's lock file only when it is free, and
                   only the file that the rows' flock holds; end_if_waiting() then ends a row of kind
                   gpu only if its leader still runs, so that its flock still waits. Tried here with a
                   free lock, a lock that another process holds, a leader that has ended, a flock
                   that waits, and another file
  the lock's folder
                   GPU_LOCK in a folder that others can write in, without the sticky bit, ends the
                   copy with status 2 before the lock file or the work folder is made; a sticky
                   folder, as /tmp, passes the look
  two tables       the copy runs SUITE.md alone, then with a further table next to it (a
                   SUITE-<word>.md whose first line is run_suite.FURTHER) and a page named
                   SUITE-<word>.md: both tables' rows pass and the page is passed over. Rejected,
                   with status 2: a name that is in both tables, a row of SUITE.md that needs one of
                   the further table's, and a table of tests without that first line
  an accept, and no standard output
                   a row ends with status 3 after the reader of the copy's standard output has gone,
                   so that every later print fails: the copy still ends with status 3, and standard
                   error has the words "status 3"
  bin under sudo   own_bin() of run_suite.py, called twice on a new name in a folder of this test's own, makes one
                   empty folder, this account's, also when the folder above is reached through a link; one that exists is
                   left as it is, and a file at the name is an error. As root, in a folder that another account (uid 4242)
                   owns, the new folder stays root's and one note, whole, names it, the folder above (escaped) and the uid;
                   none for a bin/ that account made, one with group write, root's tree, or a path stdout cannot hold. A run
                   played as not root prints it where bin/ is root's. Not shown: a whole run under sudo

Not shown: that a command that bash can read does what it should; that STEERS is complete (a
variable that a test reads and that no SUITE.md command sets is seen only if it is in ALSO or in the
list of "the programs' list"); that LD_AUDIT is not passed on (a value that is not an audit library
would make the loader print a line in every program); that run_suite.py does not write a row's
output into a file or a link that already has the output file's name. Where rows cannot be confined,
"a tree that cannot be written", "the GPU's places" and "the last line says how" check nothing and
are printed as expected; as root where run_suite.py refuses rows on the GPU (no mount namespace, and
places of /proc writable), "the GPU's places", "the last line says how" and the confined run of "the
card's lock" say "not checked here", and the checks are printed as expected; where the file system's
change times are too coarse, one case of "a link laid during the look" is not checked, and that check
is printed as expected.
Prints one line for each check, then how many were as expected and how many were not. Exits 0 if all
were as expected, 1 if not, and 2 if it could not start: after printing this text, for a wrong call
or a folder that it cannot create, or after one line, for a folder below one that another account
may be able to change.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True  # before run_suite is imported: no __pycache__ in the tree, also when called without -B
import run_suite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Names that run_suite.py must not drop: STEERS must not match them, and a SUITE.md command may set
# them. BUILD names the GPU programs' build folder. C is a path that some commands set for their
# test script; run_suite.py does not drop a caller's C. The others set numbers of threads, where
# programs are found, whether Python writes bytecode, or where temporary files go (each row gets a
# TMPDIR of its own, not the caller's).
PASSED_ON = {"BUILD", "C", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PATH", "PYTHONDONTWRITEBYTECODE", "SUITE_JOBS", "TMPDIR"}
# run_suite.py sets B and WORD (the version word) for every row, so a caller's values never arrive.
# A row may set them again, in its own command, for a script that it calls. No row of SUITE.md
# does.
OWN = {"B", "WORD"}
# Names that run_suite.py must drop and that "by text" cannot find, because no SUITE.md command sets
# them. The first three are vLLM's, NVCC_PREPEND_FLAGS is nvcc's, DEPENDENCIES_OUTPUT and SUNPRO_DEPENDENCIES
# make gcc (and nvcc, through gcc) write a dependency file where the caller says, and GPU_LOCK is run_suite.py's own
# input. Tests set some of the others for the programs, for test builds or for altered copies of the
# programs (test_word_in_the_environment() in prover/common/test_words.h has the GPU programs' own
# list); the rest, such as TRUE_VERIFIER, are here because STEERS has them.
ALSO = {
    "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_TARGET_DEVICE", "CP_TESTS_GATE",
    "H_HC_MASK_LIMIT", "COINS_REFUSED_DRAW", "CHAIN_PARTS", "DEAD_REQUEST", "LAUNCH_COUNT_FILE", "LAUNCH_COUNT_APART",
    "STAND_IN_TEST_WORD", "TRUE_VERIFIER", "REFERENCE_PROVER", "COIN_RECORD", "GPU_LOCK", "BEFORE", "NVCC_PREPEND_FLAGS",
    "ETAU", "LIE", "LIE_SEAT", "PLANT", "LAYER_LIE", "COMMIT_LIE", "COMMIT_LIE_LOOKUP", "WEIGHT_FEED",
    "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES",
}  # fmt: skip
# Names that run_suite.py passes on to no row and does not list as dropped: UNSET in run_suite.py,
# except LD_AUDIT, and one exported bash function. BASH_ENV and ENV name a file that a shell may run
# when it starts, CDPATH changes where a script's cd goes, SHELLOPTS and BASHOPTS change how bash
# reads a script, LD_PRELOAD loads a library of the caller's into every program, an exported
# function replaces the command of its name, PYTHONOPTIMIZE takes out Python's assert statements,
# and the rest give access to the caller's services.
CALLERS_OWN = ["BASH_ENV", "ENV", "CDPATH", "SHELLOPTS", "BASHOPTS", "LD_PRELOAD", "XDG_RUNTIME_DIR",
               "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "DOCKER_HOST", "PYTHONOPTIMIZE", "BASH_FUNC_flock%%"]  # the last: an exported bash function
ALARM_FURTHER = "# further rows of run_suite_test.py's alarm table"  # the first line of a further alarm table

# For "alarms": a file of this repository and pieces of its text. Put together, the pieces stand for
# a line that the file's test prints, without the names, numbers and words between them.
ALARM_LINES = (
    ("verifier/tests/extreme_proof_files_test.py", ("WAS NOT REFUSED", " | ")),
    ("tests/new/run_new.py", ("LOOK: accepted, and ",)),
    ("tests/new/run_new.py", ("LOOK ", "ACCEPTED")),
    ("tests/new/run_new.py", ("run_new.py: ", "SOMETHING WAS ACCEPTED THAT HAD TO BE REFUSED")),
    ("prover/tests/batch_opening_test.cpp", ("A VIEW IS ACCEPTED UNDER ANOTHER REQUEST'S ROWS: ",)),
    ("verifier/tests/link_door_test.cpp", ("A FALSE VALUE IS ACCEPTED: ",)),
    ("verifier/tests/req_chain_test.cpp", ("A LINK IS LEFT WITHOUT ITS CLOSE, AND THE RUN IS ACCEPTED: ",)),
    ("verifier/tests/req_chain_test.cpp", ("A MOMENT IS LEFT OUT, AND THE RUN IS ACCEPTED: ",)),
    ("verifier/tests/req_chain_test.cpp", ("A FALSE TARGET IS ACCEPTED: ",)),
    ("verifier/tests/req_chain_test.cpp", ("ANOTHER WEIGHT IS ACCEPTED: ",)),
)
# For "alarms": lines that rows print when nothing is wrong, or when they fail for another reason.
QUIET = (
    "5 rows as expected, 0 WAS NOT REFUSED",
    "as expected: a judge that is sound: status 0, 0 lines LOOK, 0 copies kept",
    "honest: 12 of 12: ACCEPTED (every combination stayed in the window)",
    "no_pad: hull: still: 0 entries are moved by no coin and are the same for both witnesses; 6 are moved by no coin for one witness at least and tell them apart: TOLD APART: ",
    "held_look: no_pad: no line of hull's says: ^hull: still: ",
    "prove-007 | request 0 | elements 3 | 59800 draws | host | first stream 16386 | VERIFIES",
    "   NOT AS EXPECTED: prove-planted: the own test's line is missing",
    "is accepted: -in a -out b",
    "AS EXPECTED: the honest copy was accepted, and the copy with one bit flipped was rejected. The last line of each check:",
    "OTHERWISE   every play ends in one of the three ways OWN TEST, REFUSED, VERIFIES",
    "   NOT AS EXPECTED: prove-control: wanted status 0, proof and y",
    "NOT AS EXPECTED: hf_open_side_test: racecheck did not end within 30 minutes, so it says nothing",
    "all seven files at level 1: as expected",
    "  WANTED: the mark failed and no other file, staged or released",
    "OTHERWISE: one bit: the new files are not the request's empty mark alone",
)
KIT = os.path.join("prover", "examples", "target_small_f2")  # its policy.txt has the version word
# Finds NAME=value before a program: NAME= at the start or after ; & ( "then", "do" or a blank, a
# value without blanks, any further NAME=value words, then a word that begins with a-z . / $ or ".
SET = re.compile(
    r"(?:^|[;&(]\s*|then\s+|do\s+|\s)([A-Z][A-Z0-9_]*)=(?=\S*\s+(?:[A-Z][A-Z0-9_]*=\S*\s+)*[a-z./$\"])"
)


def main():
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # an inherited SIG_IGN loses children's statuses
    os.umask(0o022)  # under umask 077 "closed folders" would pass even if run_suite.py set no modes
    if len(sys.argv) != 2 or (os.path.abspath(sys.argv[1]) + "/").startswith(HERE + "/"):
        print(__doc__)
        return 2
    if run_suite.others_can_change(os.path.dirname(os.path.abspath(sys.argv[1]))):
        print("run_suite_test.py: %s lies below a folder that another account may be able to change, where run_suite.py refuses to "
              "work: name a folder in your home folder or in /tmp" % os.path.abspath(sys.argv[1]))  # fmt: skip
        return 2
    try:  # made at once, closed to other accounts, and only if it does not exist yet
        os.mkdir(os.path.abspath(sys.argv[1]), 0o700)
    except OSError:  # it exists already, or the folder above it does not
        print(__doc__)
        return 2
    tally = [0, 0]

    # Counts and prints one check. `what` is the check's name in the text at the top of this file,
    # which says what is expected; `seen` is printed only if the check fails.
    def check(ok, what, seen=""):
        tally[0 if ok else 1] += 1
        print("%s: %s%s" % (what, "as expected" if ok else "NOT as expected", "" if ok else ": " + seen), flush=True)

    rows = run_suite.tables(HERE)  # SUITE.md, and the further tables next to it if there are any
    bash = ["bash", "-n", "-c"]
    # The caller's environment without what run_suite.py passes on to no row: even "bash -n" would
    # read the file that a caller's BASH_ENV names.
    plain = {n: v for n, v in os.environ.items() if n not in run_suite.UNSET and not n.startswith("BASH_FUNC_")}
    read = {n: subprocess.run(bash + [r["command"]], capture_output=True, text=True, env=plain) for n, r in rows.items()}
    unread = ["%s (%s)" % (n, r.stderr.strip().split("\n")[0]) for n, r in read.items() if r.returncode or r.stderr]
    check(rows and not unread, "parsed", "of %d rows bash cannot read these: %s" % (len(rows), "; ".join(unread)))
    named = {m[1] for row in rows.values() for m in SET.finditer(row["command"])}
    kept = sorted(n for n in named - PASSED_ON - OWN if not run_suite.STEERS.match(n))
    gone = sorted(n for n in PASSED_ON if run_suite.STEERS.match(n))
    seen = "set by a row and passed on: %s; in PASSED_ON and dropped: %s" % (kept, gone)
    check(named and not kept and not gone, "by text", seen)

    def text_of(*path):
        try:
            with open(os.path.join(HERE, *path)) as f:
                return f.read()
        except OSError:
            return None

    listed = re.search(r"for \(const char \*head : \{([^}]*)\}", text_of("prover", "common", "test_words.h") or "")
    heads = re.findall(r'"([A-Z0-9_]+)=?"', listed[1]) if listed else []
    kept = [n for n in heads if not run_suite.STEERS.match(n + ("X" if n.endswith("_") else ""))]
    check(heads and not kept, "the programs' list", "names found in the list of prover/common/test_words.h: %d; passed on: %s" % (len(heads), kept))
    gone, silent, rung = [], [], set()
    # Further alarm tables, tests/alarm_lines.<word>.tsv whose first line is ALARM_FURTHER, add
    # entries in the form of ALARM_LINES: on each line a path, a tab and the pieces as a JSON list.
    # Whether each expression of ALARM_WORDS matches an entry is checked only with such a table:
    # some match none of ALARM_LINES. A file of that name without that first line fails the check.
    lines, further, unmarked = list(ALARM_LINES), [], []
    for name in sorted(os.listdir(os.path.join(HERE, "tests"))):
        if re.fullmatch(r"alarm_lines\.[A-Za-z0-9_-]+\.tsv", name):
            with open(os.path.join(HERE, "tests", name), encoding="utf-8") as f:
                if f.readline().rstrip("\n") != ALARM_FURTHER:  # an alarm table that lost its first line: not passed over
                    unmarked.append("tests/%s" % name)
                    continue
                further.append(name)
                lines += [(s, tuple(json.loads(p))) for s, p in (x.rstrip("\n").split("\t") for x in f if x.strip() and not x.startswith("#"))]
    for script, pieces in lines:
        text = text_of(script)
        gone += ["%s: not there" % script] if text is None else ["%s: %r" % (script, piece) for piece in pieces if piece not in text]
        by = [words for words in run_suite.ALARM_WORDS if re.search(words, "".join(pieces) + "\n")]
        rung.update(by)
        if not by or not run_suite.ALARMS.search("".join(pieces) + "\n"):
            silent.append("".join(pieces))
    idle = [words for words in run_suite.ALARM_WORDS if words not in rung] if further else []
    loud = [line for line in QUIET if run_suite.ALARMS.search(line + "\n")]
    check(not gone and not silent and not idle and not loud and not unmarked, "alarms",
          "no longer in the script: %s; not matched: %s; wordings that match no line: %s; matched, and should not be: %s; "
          "alarm tables without their first line: %s" % (gone, silent, idle, loud, unmarked))  # fmt: skip

    work = os.path.abspath(sys.argv[1])
    os.makedirs(work + "/tree/" + KIT)
    shutil.copy(os.path.join(HERE, "run_suite.py"), work + "/tree")
    os.makedirs(work + "/tree/tools")
    shutil.copy(os.path.join(HERE, "tools/confine.py"), work + "/tree/tools")
    shutil.copy(os.path.join(HERE, KIT, "policy.txt"), work + "/tree/" + KIT)
    dropped, passed = sorted((named - PASSED_ON - OWN) | ALSO), sorted(PASSED_ON - {"PATH", "C"})
    head = "| name | kind | needs | word | slots | minutes | passes | command |\n|---|---|---|---|---|---|---|---|\n"
    with open(work + "/tree/SUITE.md", "w") as f:
        cells = "| %s | cpu | - | any | 1 | 0.1 | `0` | `%s` |\n"
        f.write(head + cells % ("first", 'for n in %s; do echo "$n=$(printenv $n)"; done' % " ".join(dropped + passed + CALLERS_OWN)))
        f.write(cells % ("second", "%s=in-the-command printenv %s" % (dropped[0], dropped[0])))
    env = dict(os.environ, **{n: "from-the-caller" for n in dropped + passed + CALLERS_OWN})
    # BUILD gets a folder that only the caller can write in: run_suite.py rejects "from-the-caller".
    os.mkdir(work + "/callers-build", 0o700)
    env["BUILD"] = work + "/callers-build"
    # A well-formed exported function, which bash imports and exports again: if run_suite.py passed
    # it on, the row's printenv would show it.
    env["BASH_FUNC_flock%%"] = "() {  echo a function of the caller\n}"
    env["LD_PRELOAD"] = "libc.so.6"  # a library that every program loads anyway, so the programs started here print nothing
    run = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/run", "--unconfined"], env=env,
                         capture_output=True, text=True)  # fmt: skip

    # The lines of the output file, <work>/screens/<row>.txt, of a row of the run above, or [].
    def screen(name):
        try:
            with open("%s/run/screens/%s.txt" % (work, name)) as f:
                return f.read().split("\n")[:-1]
        except OSError:
            return []

    # an unconfined row's TMPDIR is its own folder in the work folder, whatever the caller's is
    want = [n + "=" for n in dropped] + [n + "=" + (work + "/run/tmp/first" if n == "TMPDIR" else env[n]) for n in passed] + [n + "=" for n in CALLERS_OWN]
    seen = "status %d; %s" % (run.returncode, screen("first"))
    check(run.returncode == 0 and screen("first") == want, "dropped", seen)
    first = run.stdout.split("\n")[1]
    check(first.endswith(": " + " ".join(dropped)), "said", first)
    check(screen("second") == ["in-the-command"], "in the command", str(screen("second")))

    with open(work + "/tree/SUITE.md", "w") as f:
        cells = "| %s | cpu | - | any | 1 | 0.1 | `0 %s` | `echo Ran 3 tests; echo in 2 files; echo OK` |\n"
        f.write(head + cells % ("all-there", "^OK$ ;; ^Ran 3 tests$ ;; ^in 2 files$"))
        f.write(cells % ("another-number", "^OK$ ;; ^Ran 4 tests$"))
        f.write(cells % ("another-last-line", "^in 2 files$ ;; ^Ran 3 tests$"))
        f.write(cells % ("the-second-is-missing", "^OK$ ;; ^Ran 3 tests$ ;; ^in 3 files$"))
    run = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/run2", "--unconfined"],
                         capture_output=True, text=True)  # fmt: skip
    ended = sorted(tuple(x.split()[:2]) for x in run.stdout.split("\n") if x.startswith(("passed", "FAILED")))
    want = [("FAILED", "another-last-line"), ("FAILED", "another-number"), ("FAILED", "the-second-is-missing"),
            ("passed", "all-there")]  # fmt: skip
    reasons = ("status 0; OK; no line matches ^Ran 4 tests$", "status 0; OK; the last line does not match ^in 2 files$")
    told = [x for x in run.stdout.split("\n") if x.endswith(reasons)]
    check(run.returncode == 1 and ended == want and len(told) == 2, "some line",
          "status %d; %s; %s" % (run.returncode, ended, told))  # fmt: skip

    def gpu_refused():  # True where run_suite.py refuses rows on the GPU: as root, no mount namespace, and a place of /proc writable
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import confine  # noqa: E402
        tried = confine.rows_as_root() if confine.level_here()[0] else None
        return bool(tried and tried[0] == confine.UNMADE and tried[2])

    def not_here(name):
        print("%s: not checked here, and counted as expected (as root, run_suite.py refuses rows on the GPU here: the machine refused a "
              "mount namespace, and places of /proc are writable)" % name, flush=True)  # fmt: skip
        check(True, name)

    # Runs the copy on a work folder inside this test's folder; returns the run and a description.
    def suite(folder, *more, kind="cpu", env=None, bare=False):
        call = [sys.executable, "-B", work + "/tree/run_suite.py", kind, work + "/" + folder, *more]
        if not bare:  # bare: the default call, which confines the rows; other checks run unconfined
            call.append("--unconfined")
        run = subprocess.run(call, capture_output=True, text=True, env=env)
        made = "a work folder" if os.path.lexists(work + "/" + folder) else "no work folder"
        return run, "status %d; %s; printed %r; standard error %r" % (run.returncode, made, run.stdout, run.stderr)

    # Whether the run was "rejected" (see the top of this file). Its line on standard error must
    # start with `begins` and contain `says`.
    def nothing(run, seen, begins="run_suite.py: nothing was run", says=""):
        one_line = run.stderr.startswith(begins) and run.stderr.count("\n") == 1 and says in run.stderr
        return run.returncode == 2 and not run.stdout and one_line and "; no work folder; " in seen

    run, seen = suite("run3", "--only", "^no-such-row$")
    check(nothing(run, seen), "empty only", seen)
    run, seen = suite("run4", "--share", "5/5")
    control, seen_too = suite("run5", "--share", "1/5")
    ran_one = control.returncode == 0 and all(x in control.stdout for x in (": 1 rows passed, ", "passed   all-there "))
    ran_one = ran_one and control.stdout.startswith("run_suite.py: the rows run UNCONFINED (--unconfined)\n")
    ran_one = ran_one and control.stdout.rstrip().endswith("; the rows ran UNCONFINED (--unconfined); the call chose 1 of the table's 4 rows")
    check(nothing(run, seen) and ran_one, "empty share", seen + "; the control: " + seen_too)
    run, seen = suite("run6", kind="gpu")
    check(nothing(run, seen), "empty kind", seen)
    run, seen = suite("run7", "--only", "(")
    check(nothing(run, seen, "run_suite.py: --only: "), "no expression", seen)
    run, seen = suite("run8", "--only", "^no-such-row$", "--list")
    listed = run.stdout.startswith("0 rows, ") and run.stdout.count("\n") == 1
    check(run.returncode == 0 and listed and not run.stderr and "; no work folder; " in seen, "empty list", seen)
    with open(work + "/tree/SUITE.md") as f:
        page = f.read()
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(page.replace(" | any | ", " | 2 | "))
    with open(work + "/tree/" + KIT + "/policy.txt", "w") as f:
        f.write("version 1\n")
    run, seen = suite("run9")
    check(nothing(run, seen, says="; version word 1 leaves out 4 of them; "), "all left out", seen)

    cells = "| %s | cpu | %s | any | 1 | 0.1 | `%s` | `%s` |\n"

    def ends(run):
        """Returns the rows' lines of a run as sorted (outcome, row name) pairs."""
        found = (re.match(r"(ACCEPT|not run|in part|taken|passed|FAILED) +(\S+) ", x) for x in run.stdout.split("\n"))
        return sorted((m[1], m[2]) for m in found if m)

    refused = []
    for cell, begins in (("0,3", "status 3 "), ("0,03", "status 3 "), ("0,", "the cell 'passes' must "), ("0 (", "in the cell 'passes', '(' is not ")):
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + cells % ("a-bad-cell", "-", cell, "true"))
        run, seen = suite("run10")
        refused.append(nothing(run, seen, "run_suite.py: SUITE.md, line 3: " + begins))
    check(all(refused), "no row with 3", seen + "; %s" % refused)
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + cells % ("accepts", "-", "0", "mkdir -p $W; echo WAS NOT REFUSED: a made-up line of this test; exit 3"))
        f.write(cells % ("needs-it", "accepts", "0", "true"))
        f.write(cells % ("not-here", "-", "0,77 ^not run: ", "echo not run: a made-up reason; exit 77"))
        f.write(cells % ("says-nothing", "-", "0", "exit 3"))
        f.write(cells % ("not-here-with-0", "-", "0,77", "echo not run: a made-up reason"))
        f.write(cells % ("counts-none", "-", "0", "echo 5 rows as expected, 0 WAS NOT REFUSED"))
        f.write(cells % ("counts-ten", "-", "0", "echo 5 rows as expected, 10 WAS NOT REFUSED"))
    run, seen = suite("run11")
    want = [("ACCEPT", "accepts"), ("ACCEPT", "says-nothing"), ("not run", "needs-it"), ("not run", "not-here"),
            ("not run", "not-here-with-0"), ("passed", "counts-none"), ("passed", "counts-ten")]  # fmt: skip
    said = run.stdout.split("\n")
    looked = [x for x in said if x.startswith("LOOK ") and x.endswith((": WAS NOT REFUSED: a made-up line of this test", ", 10 WAS NOT REFUSED"))]
    # A LOOK line for the count of 0 would add nine entries, so that len(looked) == 2 fails.
    looked += [x for x in said if x.startswith("LOOK ") and x.endswith(", 0 WAS NOT REFUSED")] * 9
    loud = [x for x in said if x.startswith("run_suite.py: SOMETHING FALSE WAS ACCEPTED: ") and "(ACCEPT above): 2. Keep " in x]
    counted = ": 2 rows passed, 0 passed in part " in said[-2] and " 0 failed, 3 not run (2 of them only because a test said it could not run here), " in said[-2] and "; 2 ended with status 3; " in said[-2]
    check(run.returncode == 3 and ends(run) == want and len(looked) == 2 and len(loud) == 1 and counted
          and os.path.isdir(work + "/run11/accepts"), "an accept", seen)  # fmt: skip

    # A row that did not run and said why (its own «not run» line, as for a missing compiler), and a row that needs only
    # it, end the run with 77, not 1; with a failed row beside them, it is 1.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + cells % ("no-tool", "-", "0,77 ^not run: ", "echo not run: there is no made-up-tool on this machine; exit 77"))
        f.write(cells % ("needs-the-tool", "no-tool", "0", "true") + cells % ("plain", "-", "0", "true"))
        f.write(cells % ("chain", "needs-the-tool", "0", "true"))
    why, seen_why = suite("run11w")
    with open(work + "/tree/SUITE.md", "a") as f:
        f.write(cells % ("fails", "-", "0", "exit 1"))
    worse, seen_worse = suite("run11x")
    check(why.returncode == 77
          and ends(why) == [("not run", "chain"), ("not run", "needs-the-tool"), ("not run", "no-tool"), ("passed", "plain")]
          and " 0 failed, 3 not run (3 of them only because a test said it could not run here), " in why.stdout
          and "not run: it needs no-tool, which did not run and says why in its last line" in why.stdout
          and "not run: it needs no-tool (through another row), which did not run and says why in its last line" in why.stdout
          and worse.returncode == 1, "not run, said why", "%s; then with a failed row: %s" % (seen_why, seen_worse))  # fmt: skip

    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + cells % ("accepts", "-", "0", "exit 3") + cells % ("goes-on", "-", "0", "sleep 60"))
    call = [sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/run11s", "--jobs", "2", "--unconfined"]
    with subprocess.Popen(call, stdout=subprocess.PIPE, text=True) as run:
        run.stdout.readline()  # how the rows run
        said = [run.stdout.readline()]  # the first row to end is the one that ends at once
        run.send_signal(signal.SIGTERM)
        said += run.stdout.read().split("\n")
    loud = [x for x in said if x.startswith("run_suite.py: SOMETHING FALSE WAS ACCEPTED: ")]
    check(run.returncode == 3 and said[0].startswith("ACCEPT ") and len(loud) == 1, "an accept, then a signal",
          "status %d; %s" % (run.returncode, said))  # fmt: skip

    lost = subprocess.Popen(["true"])
    os.waitpid(lost.pid, 0)  # reaped here, so the Popen object `lost` cannot get the status, 0
    got = run_suite.end_of(lost, 5)
    sent, killpg = [], os.killpg
    os.killpg = lambda pid, number: sent.append((pid, number))  # records, sends nothing
    try:
        run_suite.end_the_groups([lost], 0)
    finally:
        os.killpg = killpg
    check(got == run_suite.LOST < 0 and lost.returncode == run_suite.LOST and not sent, "a lost status",
          "end_of gave %r, the handle holds %r, end_the_groups() sent %r" % (got, lost.returncode, sent))  # fmt: skip

    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + cells % ("one", "-", "0", "mkdir $W") + cells % ("two", "one", "0", "mkdir $W"))
        f.write(cells % ("three", "two", "0", "mkdir $W"))

    def made(folder):
        return [x for x in ("one", "two", "three") if os.path.isdir("%s/%s/%s" % (work, folder, x))]

    run, seen = suite("run12", "--only", "^three$", "--taken", "^two$")
    good = [run.returncode == 0 and ends(run) == [("passed", "three"), ("taken", "two")] and made("run12") == ["three"]
            and ", 1 taken and not run (--taken) and 1 that only these need; " in run.stdout
            and "SOMETHING FALSE" not in run.stdout]  # fmt: skip
    run, more = suite("run13", "--only", "^two$", "--taken", "^two$")
    good.append(run.returncode == 0 and ends(run) == [("passed", "one"), ("passed", "two")])
    seen += "; the second call: " + more
    run, more = suite("run14", "--taken", "^two$")
    good.append(run.returncode == 0 and ends(run) == [("passed", "three"), ("taken", "two")] and made("run14") == ["three"]
                and ", 1 taken and not run (--taken) and 1 that only these need; " in run.stdout)  # fmt: skip
    seen += "; the third call: " + more
    run, more = suite("run15", "--taken", "")
    good.append(nothing(run, more, "run_suite.py: --taken: "))
    check(all(good), "taken", seen + "; the fourth call: " + more + "; %s" % good)

    # The row "waits" has the larger cell "minutes", so run_suite.py takes it up first, but "makes"
    # can still get the GPU before it. The variable `bare` is this test's environment without
    # GPU_PREFIX and GPU_LOCK; suite() has a parameter of the same name.
    by_run = "| %s | gpu-by-run | - | any | 1 | %s | `0 ^got:%s$` | `%s && echo got:$GPU_RUN` |\n"
    # "waits" creates $S/early if the file is there when it begins: "makes" then ran before it, not
    # beside it. It waits up to 60 s side by side (a loaded machine may start "makes" late), and 5 s
    # one at a time.
    wait = "if [ -e $S/made ]; then touch $S/early; fi; for i in $(seq %d); do if [ -e $S/made ]; then break; fi; sleep 0.1; done; test -e $S/made"
    bare, good = {n: v for n, v in os.environ.items() if n not in ("GPU_PREFIX", "GPU_LOCK")}, []
    for folder, words in (("run16", "env"), ("run17", "")):
        with open(work + "/tree/SUITE.md", "w") as f:
            got = "flock \\S+" + (" " + words if words else "")  # the GPU lock, GPU_PREFIX's words
            f.write(head + by_run % ("waits", "2", got, wait % (600 if words else 50)) + by_run % ("makes", "1", got, "touch $S/made"))
        run, more = suite(folder, "--jobs", "2", kind="gpu", env=dict(bare, GPU_PREFIX=words) if words else bare)
        if words:
            ok = run.returncode == 0 and ends(run) == [("passed", "makes"), ("passed", "waits")]
        else:  # one at a time: "waits" fails, or it found the file at its start ("makes" ran first)
            early = os.path.exists(work + "/" + folder + "/shared/early")
            ok = (run.returncode == 1 and ends(run) == [("FAILED", "waits"), ("passed", "makes")]) or (
                run.returncode == 0 and ends(run) == [("passed", "makes"), ("passed", "waits")] and early)
        good.append(ok and run.stdout.split("\n")[0].endswith("; GPU_PREFIX: env") == bool(words))
        seen = more if words else seen + "; without GPU_PREFIX: " + more
    check(all(good), "side by side", seen)
    # The GPU lock: the file that GPU_LOCK names, or else <work>/gpu.lock; run_suite.py creates it.
    shown = ("| parent | gpu | - | any | 1 | 0.1 | `0 ^flock \\S*%s bash ` | `tr '\\0' ' ' < /proc/$PPID/cmdline` |\n"
             "| by-run | gpu-by-run | - | any | 1 | 0.1 | `0 ^ok$` | `$GPU_RUN true && echo ok` |\n")  # as a script puts it: unquoted
    good, seen = [], ""
    refused = gpu_refused()
    if refused:  # then the run with GPU_LOCK also runs unconfined, like run24, and its confined form is not checked
        print("the card's lock: its confined run not checked here (as root, run_suite.py refuses rows on the GPU here: the machine refused "
              "a mount namespace, and places of /proc are writable)", flush=True)  # fmt: skip
    for folder, lock in (("run24", ""), ("run25", work + "/card.lock")):
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + shown % (re.escape(os.path.basename(lock or "gpu.lock"))))
        landlock = subprocess.run([sys.executable, "-B", os.path.join(HERE, "tools/confine.py"), "--level"], capture_output=True).returncode == 0
        run, more = suite(folder, kind="gpu", env=dict(bare, **({"GPU_LOCK": lock} if lock else {})), bare=bool(lock) and landlock and not refused)  # the run with GPU_LOCK is confined where rows on the GPU can be
        good.append(run.returncode == 0 and "passed   parent " in run.stdout and "passed   by-run " in run.stdout
                    and os.path.isfile(lock or work + "/" + folder + "/gpu.lock"))  # fmt: skip
        seen += more + "; "
    run, more = suite("run26", kind="gpu", env=dict(bare, GPU_LOCK=work + "/no-such-folder/card.lock"))
    odd, more_odd = suite("run27", kind="gpu", env=dict(bare, GPU_LOCK=work + "/a b.lock"))
    twice, more_twice = suite("run28", kind="gpu", env=dict(bare, GPU_LOCK=work + "/card.lock", GPU_PREFIX="flock " + work + "/card.lock"))
    os.symlink(work + "/card.lock", work + "/card.link")
    linked, more_linked = suite("run28l", kind="gpu", env=dict(bare, GPU_LOCK=work + "/card.lock", GPU_PREFIX="flock " + work + "/card.link"))
    check(all(good) and nothing(run, more, "run_suite.py: the card's lock ") and nothing(odd, more_odd, "run_suite.py: GPU_LOCK: ")
          and nothing(twice, more_twice, "run_suite.py: GPU_PREFIX names the card's lock") and nothing(linked, more_linked, "run_suite.py: GPU_PREFIX names the card's lock"), "the card's lock",
          seen + "a lock that cannot be made: " + more + "; odd signs: " + more_odd + "; the lock in GPU_PREFIX too: " + more_twice + "; through a link: " + more_linked)  # fmt: skip
    # As root, a card's lock in /dev/shm: a stand-in for root's rows (tools/confine.py's rows_as_root() and root_tried() say
    # "laid", so nothing is mounted). Refused before anything is made: directly, through a link, and in GPU_PREFIX, also
    # where the lock does not exist yet.
    sys.path.insert(0, os.path.join(HERE, "tools"))
    sys.dont_write_bytecode = True
    import confine  # noqa: E402

    landlock = confine.level(confine.kernel_abi())[0] is not None  # Landlock alone: as root, --level would also try the namespace
    if not (landlock and os.path.isdir("/dev/shm")):
        print("as root, a card's lock in /dev/shm: not checked here (no confinement, or no folder /dev/shm)", flush=True)
        check(True, "as root, a card's lock in /dev/shm")
    else:
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + shown % re.escape("gpu.lock"))
        os.symlink("/dev/shm", work + "/shm-link")
        in_shm = "/dev/shm/run-suite-test-%d-%d.lock" % (os.getpid(), time.monotonic_ns())
        as_root = ("import sys; sys.path[:0] = [%r, %r]; sys.dont_write_bytecode = True; import confine, run_suite; "
                   "confine.rows_as_root = confine.root_tried = lambda: ('laid', '', []); sys.argv = ['run_suite.py', 'gpu', %%r]; sys.exit(run_suite.main())"
                   % (work + "/tree/tools", work + "/tree"))  # fmt: skip
        said = {}
        names = [in_shm, in_shm + ".linked", in_shm + ".prefix"]  # this check's own, should a runner make them
        for folder, more in (("run28r", {"GPU_LOCK": in_shm}), ("run28rl", {"GPU_LOCK": work + "/shm-link/" + os.path.basename(names[1])}),
                             ("run28rp", {"GPU_PREFIX": "flock /dev/shm"}), ("run28rn", {"GPU_PREFIX": "flock " + names[2]})):  # fmt: skip
            done = subprocess.run([sys.executable, "-I", "-B", "-c", as_root % (work + "/" + folder)], capture_output=True, text=True, env=dict(bare, **more))
            said[folder] = (done, "status %d; %s; printed %r; standard error %r" % (done.returncode, "a work folder" if os.path.lexists(work + "/" + folder) else "no work folder", done.stdout, done.stderr))
        # Paths in /dev/shm that root's rows would not see. The two missing folders are still refused, because run_suite.py
        # checks the nearest existing folder at or above each path.
        hidden = {}
        for folder, more, cure in (("run28rs", {"PYTHONUSERBASE": in_shm + ".base"}, "Set PYTHONUSERBASE to a folder off /dev/shm"),
                                   ("run28rv", {"VIRTUAL_ENV": in_shm + ".venv"}, "Set VIRTUAL_ENV to a folder off /dev/shm"),
                                   ("run28ry", {"PYTHONPATH": "/dev/shm"}, "Take it out of PYTHONPATH, or move it off /dev/shm")):  # fmt: skip
            done = subprocess.run([sys.executable, "-I", "-B", "-c", as_root % (work + "/" + folder)], capture_output=True, text=True, env=dict(bare, **more))
            hidden[folder] = (done, cure, "%s in /dev/shm: status %d; %s; printed %r; standard error %r" % (
                list(more)[0], done.returncode, "a work folder" if os.path.lexists(work + "/" + folder) else "no work folder", done.stdout, done.stderr))  # fmt: skip
        # Root's GPU stop with no namespace and /proc/sys writable; and CONFINE_AS_ROOT kept from the row.
        unmade = as_root.replace("lambda: ('laid', '', [])", "lambda: ('unmade', 'unshare: Operation not permitted', ['/proc/sys'])")
        done = subprocess.run([sys.executable, "-I", "-B", "-c", unmade % (work + "/run28ru")], capture_output=True, text=True, env=bare)
        stops = {"run28ru": (done, "a row on the GPU, no namespace and /proc/sys writable: status %d; %s; printed %r; standard error %r" % (
            done.returncode, "a work folder" if os.path.lexists(work + "/run28ru") else "no work folder", done.stdout, done.stderr))}  # fmt: skip
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + '| pop | cpu | - | any | 1 | 0.1 | `0` | `test -z "${CONFINE_AS_ROOT-}"` |\n')
        popped = None if os.geteuid() == 0 else subprocess.run(  # as root the row's real confine.py would find its own setup
            [sys.executable, "-I", "-B", "-c", as_root.replace("'gpu'", "'cpu'") % (work + "/run28rc")], capture_output=True, text=True,
            env=dict(bare, CONFINE_AS_ROOT="from-the-caller"))  # fmt: skip
        shared = None if os.geteuid() == 0 else subprocess.run(  # where the namespace is refused: a line of its own says /dev/shm is shared
            [sys.executable, "-I", "-B", "-c", as_root.replace("'gpu'", "'cpu'").replace("lambda: ('laid', '', [])", "lambda: ('unmade', 'unshare: Operation not permitted', [])")
             % (work + "/run28rh")], capture_output=True, text=True, env=bare)  # fmt: skip
        made = [n for n in names if os.path.lexists(n)]
        for n in made:
            if not os.path.isdir(n):
                os.remove(n)
        check(not made and all(nothing(done, seen, "run_suite.py: ", "the lock would not stop two rows from using the card at the same time") for done, seen in said.values())
              and all(nothing(done, seen, "run_suite.py: nothing is run: ", "would not see it. " + cure) for done, cure, seen in hidden.values())
              and nothing(*stops["run28ru"], "run_suite.py: NOT CONFINED, nothing is run: as root, the rows' own mount namespace could not be set up here (unshare: Operation not permitted)")
              and (popped is None or popped.returncode == 0 and ": 1 rows passed, " in popped.stdout)
              and (shared is None or shared.returncode == 0 and (shared.stdout.split("\n") + ["", ""])[1].startswith(
                  "run_suite.py: running as root, and the machine refused the rows a mount namespace of their own, so every row shares")),
              "as root, a card's lock in /dev/shm", " | ".join([seen for _, seen in said.values()] + [seen for _, _, seen in hidden.values()]
                                                          + [seen for _, seen in stops.values()] + ["a row that must not see CONFINE_AS_ROOT: " + ("not checked as root" if popped is None else "status %d; last 300 characters printed %r; last 300 of standard error %r" % (popped.returncode, popped.stdout[-300:], popped.stderr[-300:]))]
                                                          + ["the line that /dev/shm is shared: " + ("not checked as root" if shared is None else "status %d; first 400 characters printed %r" % (shared.returncode, shared.stdout[:400]))])
              + (" | made in /dev/shm: %s" % made if made else ""))  # fmt: skip
    for cell, row in (("passes", "| lost | cpu | - | any | 1 | 0.1 | `0 ^x$ | `echo x` |\n"),
                      ("command", "| lost | cpu | - | any | 1 | 0.1 | `0 ^x$` | `echo x |\n")):
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + row)
        run, seen = suite("run18-" + cell, "--only", "^lost$")
        check(run.returncode == 2 and "line 3" in run.stderr and "backticks" in run.stderr and not run.stdout
              and "; no work folder; " in seen, "a backtick lost (%s)" % cell, seen)  # fmt: skip

    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| leaves | cpu | - | any | 1 | 0.1 | `0 ^done$` | `mkdir -p $W && { sleep 60 & echo $! > $W/left; } && echo done` |\n")
    began = time.monotonic()
    run, seen = suite("run19")
    try:
        with open(work + "/run19/leaves/left") as f:
            left = int(f.read())
    except (OSError, ValueError):
        left = None
    try:  # if a sleep with that process ID is alive, run_suite.py did not end the row's group
        with open("/proc/%d/cmdline" % left, "rb") as f:
            lives = f.read().split(b"\0")[0].endswith(b"sleep")
    except (OSError, TypeError):
        lives = False
    late = time.monotonic() - began  # after 60 s the child has ended by itself: nothing is shown
    check(run.returncode == 0 and left is not None and not lives and late < 55, "what a row leaves",
          "%s; the child's number %r, still sleeping: %s; looked after %.1f s" % (seen, left, lives, late))  # fmt: skip

    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    got = run_suite.end_of(child, 1)
    if got is None:
        run_suite.end_the_groups([child], 5)
    try:
        status = child.wait(10)
    except subprocess.TimeoutExpired:  # `end_the_groups` failed: this test ends its own child
        status = None
        child.kill()
        child.wait()
    check(got is None and status == -signal.SIGTERM, "the time limit", "end_of gave %r; the child ended with %r" % (got, status))
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| w | cpu | - | any | 1 | 0.1 | `0 ^done$` | `test ! -e $W && mkdir $W && touch $W/ok && "
                "{ touch $W/../../beside 2>/dev/null && echo LEAKED; true; } && "
                "{ [ \"$HOME\" = %s ] && [ \"$XDG_CACHE_HOME\" = \"$HOME/.cache\" ] "
                "&& [ \"$XDG_CONFIG_HOME\" = \"$HOME/.config\" ] && [ \"$XDG_DATA_HOME\" = \"$HOME/.local/share\" ] "
                "&& [ \"$XDG_STATE_HOME\" = \"$HOME/.local/state\" ] "
                "&& mkdir -p \"$XDG_CACHE_HOME/probe\" && touch \"$HOME/probe\" && echo home-ok; true; } && echo done` |\n"
                % shlex.quote(os.path.realpath(work) + "/run20/w/home"))  # fmt: skip
    run, seen = suite("run20", bare=True)
    sys.path.insert(0, os.path.join(HERE, "tools"))
    sys.dont_write_bytecode = True
    import confine  # noqa: E402

    if confine.level_here()[0]:
        lines = run.stdout.strip().split("\n")
        shown = []
        try:
            with open(work + "/run20/screens/w.txt") as f:
                shown = f.read().split("\n")
        except OSError:
            pass
        good = (run.returncode == 0 and lines[0].startswith("run_suite.py: the rows run confined: ")
                and "; the rows ran confined: " in lines[-1] and "passed   w " in run.stdout and "done" in shown
                and "LEAKED" not in shown and not os.path.lexists(work + "/run20/beside")
                and "home-ok" in shown)  # fmt: skip
        seen += "; its screen %r" % shown
    else:
        good = nothing(run, seen, "run_suite.py: NOT CONFINED")
    check(good, "confined", seen)
    # A work folder on /dev/shm's file system is refused when the rows are to be confined; unconfined, it is taken.
    try:  # inside a confined run the sandbox forbids folders in /dev/shm, whatever its mode says
        shm_top = tempfile.mkdtemp(prefix="run-suite-test-", dir="/dev/shm") if os.path.isdir("/dev/shm") and confine.level_here()[0] else None
    except OSError:
        shm_top = None
    if shm_top:
        try:  # mkdtemp made it with mode 700
            on_shm = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", shm_top + "/w"],
                                    capture_output=True, text=True)  # fmt: skip
            unconfined = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", shm_top + "/u", "--unconfined"],
                                        capture_output=True, text=True)  # fmt: skip
            shm_ok = (on_shm.returncode == 2 and "a work folder in /dev/shm can't be kept apart by the sandbox" in on_shm.stderr
                      and not os.path.lexists(shm_top + "/w") and unconfined.returncode == 0 and os.path.isdir(shm_top + "/u"))  # fmt: skip
        finally:
            shutil.rmtree(shm_top, ignore_errors=True)
        check(shm_ok, "a work folder on shared memory", "confined: status %d, %r; unconfined: status %d, %r"
              % (on_shm.returncode, on_shm.stderr, unconfined.returncode, unconfined.stderr))  # fmt: skip
    else:
        print("a work folder on shared memory: not checked here (no folder can be made in /dev/shm, or no confinement)", flush=True)
        check(True, "a work folder on shared memory", "")
    # where /dev/shm is a plain folder on the work folder's own file system (stood in for: /dev and /dev/shm get the work
    # folder's own stat), only a path under it counts: the work folder must not
    plain_shm = subprocess.run([sys.executable, "-I", "-B", "-c", "import os, sys; sys.path.insert(0, %r); import run_suite; "
                                "real = os.stat; here = real(%r); "
                                "os.stat = lambda p, *a, **k: here if p in ('/dev', '/dev/shm') else real(p, *a, **k); "
                                "print(run_suite.on_shared_memory(%r), run_suite.on_shared_memory('/dev/shm'))" % (work + "/tree", work, work)],
                               capture_output=True, text=True)  # fmt: skip
    check(plain_shm.stdout.strip() == ("False True" if os.path.isdir("/dev/shm") else "False False"),
          "shared memory as a plain folder", repr(plain_shm))
    # The caller's own Python packages stay in reach from the row's HOME of its own: PYTHONUSERBASE is the caller's ~/.local.
    os.makedirs(work + "/caller-home/.local")
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| u | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo \"base=$PYTHONUSERBASE home=$HOME\"; echo done` |\n")
    base_run, base_seen = suite("run20u", env=dict(os.environ, HOME=work + "/caller-home", PYTHONUSERBASE=""))
    with open(work + "/run20u/screens/u.txt") as f:
        base_said = f.read()
    check(base_run.returncode == 0 and ("base=%s/caller-home/.local " % work) in base_said
          and ("home=%s/run20u/tmp/u/home" % work) in base_said, "a HOME of its own", base_seen + "; its screen %r" % base_said)
    # A file named bin stops run_suite.py from creating bin/, also when it runs as root.
    if confine.level_here()[0]:
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + "| w | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo done` |\n")
        if os.path.isdir(work + "/tree/bin"):
            os.rmdir(work + "/tree/bin")  # empty: this test's earlier confined runs created it
        open(work + "/tree/bin", "w").close()
        run, more = suite("run29", bare=True)
        os.remove(work + "/tree/bin")
        check(run.returncode == 0 and "these folders cannot be made" in run.stdout and "passed   w " in run.stdout, "a tree that cannot be written", more)
    else:
        check(True, "a tree that cannot be written")  # rows cannot be confined: nothing is checked
    with open(work + "/tree/tools/confine.py") as f:
        text = f.read()
    with open(work + "/tree/tools/confine.py", "w") as f:  # altered to find no Landlock
        f.write(text.replace('\nif __name__ == "__main__":', '\nkernel_abi = lambda: 0  # run_suite_test.py\n\nif __name__ == "__main__":'))
    run, seen = suite("run21", bare=True)
    check(nothing(run, seen, "run_suite.py: NOT CONFINED"), "confined, no Landlock", seen)
    with open(work + "/tree/tools/confine.py", "w") as f:  # restores the copy's confine.py
        f.write(text)
    # Every row that builds may write in BUILD's folder, so too wide a BUILD is rejected.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| b | cpu | - | any | 1 | 0.1 | `0 ^B=%s$` | `echo B=$B` |\n" % re.escape(work + "/run30/build"))
    landlock = bool(confine.level_here()[0])
    run, seen = suite("run30", env=dict(os.environ, BUILD=""), bare=landlock)
    try:
        made_b = oct(os.stat(work + "/run30/build").st_mode & 0o777)
    except OSError as fault:
        made_b = fault.strerror
    good = (run.returncode == 0 and "passed   b " in run.stdout and made_b == "0o700" and not os.path.lexists(work + "/tree/prover/build")
            and (not landlock or run.stdout.split("\n")[0].endswith("; rows that build may also write in " + work + "/run30/build")))  # fmt: skip
    seen += "; the build folder's mode: %s" % made_b
    os.symlink(work, work + "/above.link")
    for folder, names in (("run31", "/"), ("run32", work), ("run33", work + "/run33"), ("run35", work + "/above.link")):
        wide, more = suite(folder, env=dict(os.environ, BUILD=names), bare=landlock)
        good = good and nothing(wide, more, "run_suite.py: BUILD names ")
        seen += "; BUILD=%s: %s" % (names, more)
    check(good, "the build folder", seen)
    if landlock and not gpu_refused():
        names_itself = "echo row > /proc/$$/comm; echo said $?"
        mine = subprocess.run(["bash", "-c", names_itself], capture_output=True, text=True).stdout.strip()  # "said 0", or "said 1" if this test is itself a confined row of kind cpu
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + "| on-the-cpu | cpu | - | any | 1 | 0.1 | `0 ^said 1$` | `%s` |\n| on-the-gpu | gpu | - | any | 1 | 0.1 | `0 ^%s$` | `%s` |\n" % (names_itself, mine, names_itself))
        run, seen = suite("run34", kind="all", env=bare, bare=True)
        check(run.returncode == 0 and "passed   on-the-cpu " in run.stdout and "passed   on-the-gpu " in run.stdout, "the GPU's places", seen)
    elif landlock:
        not_here("the GPU's places")
    else:
        check(True, "the GPU's places")  # rows cannot be confined: nothing is checked
    # Closed to other accounts: what run_suite.py created in run19, the run of "what a row leaves".
    modes = {}
    for p in ("", "screens", "shared", "tmp", "screens/leaves.txt"):
        try:
            modes[p] = oct(os.stat(work + "/run19/" + p).st_mode & 0o777)
        except OSError as fault:
            modes[p] = fault.strerror
    check(list(modes.values()) == ["0o700"] * 4 + ["0o600"], "closed folders", str(modes))
    os.mkdir(work + "/run40", 0o700)
    run, seen = suite("run40")
    check(run.returncode == 2 and not run.stdout and run.stderr.startswith("run_suite.py: ") and run.stderr.count("\n") == 1
          and os.listdir(work + "/run40") == [], "a work folder that exists", seen)  # fmt: skip
    # run_suite.py holds the work folder open, so it goes on writing in the moved folder. It starts
    # no further row, because rows reach their files by path.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| swap | cpu | - | any | 1 | 0.1 | `0 ^done$` | `w=$(dirname $S) && mv $w $w.moved && mkdir $w && echo done` |\n"
                "| after | cpu | swap | any | 1 | 0.1 | `0` | `mkdir -p $W && touch $W/ran` |\n"
                "| late | cpu | - | any | 1 | 0.01 | `0` | `mkdir -p $W && touch $W/ran` |\n")  # fmt: skip
    run, seen = suite("run41", "--jobs", "1")
    try:
        with open(work + "/run41.moved/screens/swap.txt") as f:
            moved = f.read().split("\n")
    except OSError:
        moved = []
    check(run.returncode == 1 and "passed   swap " in run.stdout and "THE WORK FOLDER WAS REPLACED OR REMOVED" in run.stdout
          and "not run  after " in run.stdout and "done" in moved and os.listdir(work + "/run41") == []
          # "late" got the one job slot first and ran in the folder that then moved, or was not run
          and ("not run  late " in run.stdout or os.path.exists(work + "/run41.moved/late/ran")), "a swapped work folder", seen)  # fmt: skip
    # Another account could replace what lies in work/open, so run_suite.py rejects a work folder or
    # a BUILD there unless the caller gives the option. For its own tree it only prints a note.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| fine | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo done` |\n")
    os.mkdir(work + "/open")
    os.chmod(work + "/open", 0o777)  # its group and others may write, and it is not sticky
    refused, more_refused = suite("open/run42")
    vouched, more_vouched = suite("open/run42c", "--i-trust-every-account-on-this-machine")
    said_so = "; the work folder lies below %s/open, which other accounts could change: the caller said that every " \
              "account on this machine is trusted (--i-trust-every-account-on-this-machine)" % work
    lines_vouched = vouched.stdout.rstrip().split("\n")
    unneeded, more_unneeded = suite("run42e", "--i-trust-every-account-on-this-machine")  # given, with no open place on the way
    lines_unneeded = unneeded.stdout.rstrip().split("\n")
    more_refused += "; the option given with no open place: " + more_unneeded
    given = "; --i-trust-every-account-on-this-machine was given, though no place on the way needed it"
    os.mkdir(work + "/open/b", 0o755)
    build_refused, more_build = suite("run42b", env=dict(os.environ, BUILD=work + "/open/b"))
    build_vouched, seen = suite("run42d", "--i-trust-every-account-on-this-machine", env=dict(os.environ, BUILD=work + "/open/b"))
    build_said = "; BUILD lies below %s/open, which other accounts could change: the caller said that every account " \
                 "on this machine is trusted (--i-trust-every-account-on-this-machine)" % work
    # note() is called directly: no run of this test has the copy's tree below such a folder.
    # --shared-from: what is copied is judged on the descriptors that the copy holds, and a source missing at the look stops
    os.makedirs(work + "/as-copied/src")
    with open(work + "/as-copied/src/a.txt", "w") as f:
        f.write("a")
    os.chmod(work + "/as-copied/src/a.txt", 0o664)  # as if swapped in, open, after the look
    os.makedirs(work + "/as-copied/into-1")
    os.makedirs(work + "/as-copied/into-2")
    copy_call = ("import os, sys; sys.path.insert(0, %r); import run_suite; "
                 "into = os.open(%r, os.O_RDONLY | os.O_DIRECTORY); print(run_suite.lay_folder(%r, into, %s))")
    as_judged = subprocess.run([sys.executable, "-I", "-B", "-c", copy_call % (work + "/tree", work + "/as-copied/into-1",
                                work + "/as-copied/src", "False")], capture_output=True, text=True)  # fmt: skip
    vouched_copy = subprocess.run([sys.executable, "-I", "-B", "-c", copy_call % (work + "/tree", work + "/as-copied/into-2",
                                   work + "/as-copied/src", "True")], capture_output=True, text=True)  # fmt: skip
    gone_source = subprocess.run([sys.executable, "-I", "-B", "-c", "import sys; sys.path.insert(0, %r); import run_suite; "
                                  "run_suite.shared_note(%r)" % (work + "/tree", work + "/as-copied/no-such-folder")],
                                 capture_output=True, text=True)  # fmt: skip
    # a sub-folder of mode 775; as root also a file and a sub-folder of another account's: each refused, each copied with the option
    more = {"open-sub": ("sub", 0o775, None), "open-top": (None, 0o775, None)}  # None: the source folder itself
    if os.geteuid() == 0:
        more.update({"their-file": ("b.txt", 0o644, 4242), "their-sub": ("sub", 0o755, 4242), "their-top": (None, 0o755, 4242)})
    refused_more, copied_more = {}, {}
    for case, (name, mode, owner) in more.items():
        src = work + "/as-copied/" + case
        os.makedirs(src)
        if name == "sub":
            os.mkdir(src + "/sub")
            with open(src + "/sub/c.txt", "w") as f:
                f.write("c")
        else:
            with open(src + "/" + (name or "d.txt"), "w") as f:
                f.write("b")
        target = src + "/" + name if name else src
        os.chmod(target, mode)
        if owner is not None:
            os.chown(target, owner, -1)
        for vouch in ("False", "True"):
            into = "%s/as-copied/into-%s-%s" % (work, case, vouch)  # beside the source, never inside it
            os.makedirs(into)
            got = subprocess.run([sys.executable, "-I", "-B", "-c", copy_call % (work + "/tree", into, src, vouch)],
                                 capture_output=True, text=True)  # fmt: skip
            (refused_more if vouch == "False" else copied_more)[case] = got
    refused_ok = all(r.returncode == 2 and "changed after the runner checked it" in r.stderr
                     and ("belongs to another account" if case.startswith("their") else "can be written by its group") in r.stderr
                     for case, r in refused_more.items())  # fmt: skip
    copied_ok = all(r.returncode == 0 for r in copied_more.values())
    check(as_judged.returncode == 2 and "changed after the runner checked it (it now can be written by its group or by other accounts)" in as_judged.stderr
          and refused_ok and copied_ok
          and vouched_copy.returncode == 0 and vouched_copy.stdout.strip() == "1"
          and gone_source.returncode == 2 and "could not be examined" in gone_source.stderr and "a moment before" in gone_source.stderr,
          "shared from, as copied", "judged: %r; vouched: %r; a missing source: %r; refused: %r; copied with the option: %r"
          % (as_judged, vouched_copy, gone_source, refused_more, copied_more))  # fmt: skip
    # --shared-from's cure, as a caller who is not root sees it (whoever runs this test): chmod go-w of the one open file, quoted
    os.makedirs(work + "/cure-open/inner")
    with open(work + "/cure-open/inner/a b.txt", "w") as f:
        f.write("a")
    os.chmod(work + "/cure-open/inner/a b.txt", 0o664)
    if os.geteuid() == 0:  # the open file must be the caller's own: as root, give it to the stand-in account
        os.chown(work + "/cure-open/inner/a b.txt", 4242, -1)
    cure = subprocess.run([sys.executable, "-I", "-B", "-c", "import os, sys; sys.path.insert(0, %r); import run_suite; "
                           "t = run_suite.trusted_owners(); me = os.getuid() or 4242; os.geteuid = lambda: me; "
                           "run_suite.trusted_owners = lambda: t | {me}; "
                           "run_suite.shared_note(%r)" % (work + "/tree", work + "/cure-open")],
                          capture_output=True, text=True)  # fmt: skip
    check(cure.returncode == 2 and ("If %s is yours, close it with chmod go-w %s (a further such file or folder in %s, if any, is "
                                    "named in turn); otherwise give "
                                    % (work + "/cure-open/inner/a b.txt", shlex.quote(work + "/cure-open/inner/a b.txt"), work + "/cure-open"))
                                   in cure.stderr and "chmod -R" not in cure.stderr,
          "shared from, its cure", "status %d; %s" % (cure.returncode, cure.stderr.strip()))
    heard = subprocess.run([sys.executable, "-I", "-B", "-c", "import sys; sys.path.insert(0, %r); import run_suite; "
                            "run_suite.note([(%r, run_suite.OPEN)])" % (work + "/tree", work + "/open")],
                           capture_output=True, text=True)  # fmt: skip
    check(nothing(refused, more_refused, "run_suite.py: the work folder %s/open/run42 lies below %s/open, which another "
                  "account may be able to change (its group or others can write in it, and it has no sticky bit)" % (work, work),
                  "If %s/open is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); otherwise name a new work folder" % work
                  if os.geteuid() != 0 else "")
          and ("chmod go-w" in refused.stderr) == (os.geteuid() != 0)  # root is never told to chmod (a shared disk is «its own»)
          and vouched.returncode == 0 and lines_vouched[0].endswith(said_so) and said_so in lines_vouched[-1]
          and unneeded.returncode == 0 and lines_unneeded[0].endswith(given) and given in lines_unneeded[-1]
          and nothing(build_refused, more_build, "run_suite.py: the build folder (BUILD) %s/open/b lies below %s/open" % (work, work))
          and build_vouched.returncode == 0 and build_said in build_vouched.stdout.split("\n")[0]
          and heard.stderr.startswith("run_suite.py: note: another account may be able to change %s/open (mode 777, no sticky bit)" % work)
          and heard.stderr.rstrip().endswith("keep the tested tree (%s) below folders that only you or root can change" % (work + "/tree")),
          "the note", seen + "; the work folder below it: " + more_refused + "; with the option: " + more_vouched
          + "; BUILD below it: " + more_build + "; the note: " + heard.stderr)  # fmt: skip
    # The GPU lock must be a regular file, not a symbolic link.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| card | gpu | - | any | 1 | 0.1 | `0` | `true` |\n")
    os.symlink(work + "/the-links-target", work + "/link.lock")
    run, seen = suite("run43", kind="gpu", env=dict(bare, GPU_LOCK=work + "/link.lock"))
    os.symlink(work + "/card.lock", work + "/link-to-a-file.lock")  # card.lock: a regular file that an earlier check made
    to_a_file, more = suite("run43f", kind="gpu", env=dict(bare, GPU_LOCK=work + "/link-to-a-file.lock"))
    device, more_too = suite("run43d", kind="gpu", env=dict(bare, GPU_LOCK="/dev/null"))
    check(nothing(run, seen, "run_suite.py: the card's lock ") and not os.path.lexists(work + "/the-links-target")
          and nothing(to_a_file, more, "run_suite.py: the card's lock ") and nothing(device, more_too, "run_suite.py: the card's lock ", says="is not a regular file"),
          "a link as the card's lock", seen + "; a link to a file: " + more + "; /dev/null: " + more_too)  # fmt: skip
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| fine | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo done` |\n")
    builds = []
    for build, begins in (("build", "run_suite.py: BUILD, when set, must be an absolute path"),
                          (work + "/open", "run_suite.py: BUILD, when set, must be a folder that you (or root) own"),
                          ("/no-such-folder", "run_suite.py: BUILD, when set, must lie at least two folders below /")):
        run, more = suite("run44", env=dict(os.environ, BUILD=build))
        builds.append(nothing(run, more, begins))
        seen = more
    check(all(builds), "BUILD refused", seen + "; %s" % builds)
    # An unconfined row's temporary files lie in the work folder, not in the caller's TMPDIR.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| temp | cpu | - | any | 1 | 0.1 | `0 ^exists$` | `echo $TMPDIR; echo $TMP $TEMP $CUDA_CACHE_PATH; test -d $TMPDIR && echo exists` |\n")
    run, seen = suite("run45", env=dict(os.environ, TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", CUDA_CACHE_PATH="/tmp/nv"))
    try:
        with open(work + "/run45/screens/temp.txt") as f:
            said = f.read().split("\n")
    except OSError:
        said = []
    check(run.returncode == 0 and said[:3] == [work + "/run45/tmp/temp", ("%s " * 2 + "%s/cuda-cache") % ((work + "/run45/tmp/temp",) * 3), "exists"], "temporary files", seen)
    # The last line repeats what the first says: where rows that build may write, and GPU_PREFIX.
    if confine.level_here()[0] and not gpu_refused():
        with open(work + "/tree/SUITE.md", "w") as f:
            f.write(head + "| card | gpu | - | any | 1 | 0.1 | `0` | `true` |\n")
        run, seen = suite("run47", kind="gpu", env=dict(bare, BUILD=work + "/callers-build", GPU_PREFIX="env"), bare=True)
        lines = run.stdout.rstrip().split("\n")
        how = "confined: %s; rows that build may also write in %s; GPU_PREFIX: env" % (confine.level_here()[0], work + "/callers-build")
        check(run.returncode == 0 and lines[0] == "run_suite.py: the rows run " + how and ("; the rows ran %s; " % how) in lines[-1],
              "the last line says how", seen)  # fmt: skip
    elif confine.level_here()[0]:
        not_here("the last line says how")
    else:
        check(True, "the last line says how")  # rows cannot be confined: nothing is checked
    # Mode cpu leaves out a row that needs a row of kind nvcc, gpu or gpu-by-run, reports it, and
    # does not count it against the run.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| world | gpu | - | any | 1 | 0.1 | `0` | `true` |\n| reads | cpu | world | any | 1 | 0.1 | `0` | `true` |\n"
                "| alone | cpu | - | any | 1 | 0.1 | `0` | `true` |\n| later | cpu | reads | any | 1 | 0.1 | `0` | `true` |\n"
                "| built | nvcc | - | any | 1 | 0.1 | `0` | `true` |\n| uses | cpu | built | any | 1 | 0.1 | `0` | `true` |\n")  # fmt: skip
    run, seen = suite("run48", env=bare)
    cpu_ok = (run.returncode == 0 and ends(run) == [("passed", "alone")]
              and "left out reads: it needs a row that needs nvcc or a GPU (world); mode all runs it" in run.stdout.split("\n")
              and "left out later: it needs a row that needs nvcc or a GPU (world); mode all runs it" in run.stdout.split("\n")
              and "left out uses: it needs a row that needs nvcc or a GPU (built); mode all runs it" in run.stdout.split("\n")
              and ", 3 left out in mode cpu because they need nvcc or a GPU, " in run.stdout)  # fmt: skip
    named, more = suite("run49", "--only", "^reads$", env=bare)
    whole, more_too = suite("run50", kind="all", env=bare)
    check(cpu_ok and nothing(named, more, "run_suite.py: nothing was run: in mode cpu each chosen row needs a row that needs nvcc or a GPU", says="reads needs world")
          and whole.returncode == 0 and ends(whole) == [("passed", "alone"), ("passed", "built"), ("passed", "later"), ("passed", "reads"), ("passed", "uses"), ("passed", "world")],
          "a row on the GPU's work", seen + "; --only for it: " + more + "; mode all: " + more_too)  # fmt: skip
    # Work folder names that are rejected before anything is created.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| fine | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo done` |\n")
    os.symlink(work + "/tree/tools", work + "/to-the-tools")  # through it, ".." is work/tree; as text it is work
    os.symlink(work + "/nowhere", work + "/run52")
    named = []
    for folder, begins in (("-run51", "run_suite.py: the work folder's name has a part that begins with"),
                           ("to-the-tools/../run53", "run_suite.py: the work folder's name goes through a link"),
                           ("run52", "run_suite.py: %s/run52 exists already" % work)):
        run, more = suite(folder)
        named.append(run.returncode == 2 and not run.stdout and run.stderr.startswith(begins) and run.stderr.count("\n") == 1
                     and not os.path.lexists(work + "/nowhere") and not os.path.lexists(work + "/run53")
                     and not os.path.lexists(work + "/tree/run53"))  # fmt: skip
        seen = more
    check(all(named), "odd work folder names", seen + "; %s" % named)
    # make_work() in run_suite.py accepts the inherited set-group-ID bit: of the new folder's mode
    # it checks only that group and others have no rights.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| fine | cpu | - | any | 1 | 0.1 | `0 ^done$` | `echo done` |\n")
    os.mkdir(work + "/sgid", 0o700)
    os.chmod(work + "/sgid", 0o2700)  # the set-group-ID bit, which new folders below it inherit
    run, seen = suite("sgid/run58")
    try:
        inherited = os.stat(work + "/sgid/run58").st_mode & 0o7777
    except OSError:
        inherited = None
    check(run.returncode == 0 and "passed   fine " in run.stdout and inherited is not None and inherited & 0o2000 and inherited & 0o777 == 0o700,
          "under a set-group-ID folder", "%s; the work folder's mode: %s" % (seen, None if inherited is None else oct(inherited)))  # fmt: skip
    # run_suite.py sets a umask, which the rows inherit: the caller's, with the write bits of group
    # and others added and the owner's bits removed.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| loose | cpu | - | any | 1 | 0.1 | `0 ^done$` | `mkdir -p $W/d && echo x > $W/d/f && echo done` |\n")
    loose = subprocess.run(["bash", "-c", 'umask 000; exec "$@"', "-", sys.executable, "-B", work + "/tree/run_suite.py", "cpu",
                            work + "/run59", "--unconfined"], capture_output=True, text=True)  # fmt: skip
    try:
        made_modes = [oct(os.stat(work + "/run59/loose/" + p).st_mode & 0o777) for p in ("d", "d/f")]
    except OSError as fault:
        made_modes = [fault.strerror]
    # Umask 277, kept as it is, would leave new folders without the owner's write bit.
    strict = subprocess.run(["bash", "-c", 'umask 277; exec "$@"', "-", sys.executable, "-B", work + "/tree/run_suite.py", "cpu",
                             work + "/run59s", "--unconfined"], capture_output=True, text=True)  # fmt: skip
    check(loose.returncode == 0 and made_modes == ["0o755", "0o644"] and strict.returncode == 0, "a loose umask",
          "status %d; the row's folder and file: %s; %r; under umask 277: status %d, %r"
          % (loose.returncode, made_modes, loose.stdout[-300:], strict.returncode, (strict.stdout + strict.stderr)[-300:]))  # fmt: skip
    # Only mode long runs a row of kind long, so a row of another kind must not need one.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| quick | cpu | - | any | 1 | 0.1 | `0` | `true` |\n| slow | long | quick | any | 1 | 0.1 | `0` | `true` |\n")
    cpu_run, seen = suite("run60")
    all_run, more = suite("run61", kind="all")
    long_run, more_long = suite("run62", kind="long")
    seen += "; all: " + more + "; long: " + more_long
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| slow | long | - | any | 1 | 0.1 | `0` | `true` |\n| needs-slow | cpu | slow | any | 1 | 0.1 | `0` | `true` |\n")
    wrong, more_wrong = suite("run63")
    check(ends(cpu_run) == [("passed", "quick")] and ends(all_run) == [("passed", "quick")]
          and ends(long_run) == [("passed", "quick"), ("passed", "slow")]
          and nothing(wrong, more_wrong, "run_suite.py: needs-slow needs slow, a row of the kind long"), "a long row",
          seen + "; a cpu row that needs a long one: " + more_wrong)  # fmt: skip
    # run_suite.py escapes what it prints of a row's output: a row must not steer the terminal.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| esc | cpu | - | any | 1 | 0.1 | `0` | `printf 'WAS NOT REFUSED: \\033[2J\\nlast \\033[31m\\n'` |\n")
    run, seen = suite("run64")
    check("\x1b" not in run.stdout and "\\x1b[31m" in run.stdout and "\\x1b[2J" in run.stdout, "printed as escapes", seen)
    # run_suite.py creates the work folder and nothing above it.
    run, seen = suite("no-such-folder/run65")
    check(run.returncode == 2 and not os.path.lexists(work + "/no-such-folder"), "a folder above that does not exist", seen)
    # --taken says that the needed row's product is there already, so mode cpu runs the row that
    # needs it.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| world | gpu | - | any | 1 | 0.1 | `0` | `true` |\n| reads | cpu | world | any | 1 | 0.1 | `0` | `true` |\n")
    run, seen = suite("run66", "--taken", "^world$", env=bare)
    check(run.returncode == 0 and ends(run) == [("passed", "reads"), ("taken", "world")], "a GPU row taken in mode cpu", seen)
    # --shared-from: what run_suite.py copies into $S before any row starts.
    os.makedirs(work + "/laid/inner")
    with open(work + "/laid/inner/world.txt", "w") as f:
        f.write("a world\n")
    os.chmod(work + "/laid/inner/world.txt", 0o6755)  # set-ID bits, to go
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| reads | cpu | - | any | 1 | 0.1 | `0 ^a world$` | `cat $S/inner/world.txt` |\n")
    run, seen = suite("run67", "--shared-from", work + "/laid")
    try:
        laid_mode = oct(os.stat(work + "/run67/shared/inner/world.txt").st_mode & 0o7777)
    except OSError as fault:
        laid_mode = fault.strerror
    holds, more_holds = suite("laid/run69", "--shared-from", work + "/laid")
    os.symlink(work + "/laid/inner/world.txt", work + "/laid/a-link")
    linked, more = suite("run68", "--shared-from", work + "/laid")
    # a file below the folder that group and others can write in: refused, and taken only with the option
    os.makedirs(work + "/laid-open/inner")
    with open(work + "/laid-open/inner/world.txt", "w") as f:
        f.write("a world\n")
    os.chmod(work + "/laid-open/inner/world.txt", 0o666)
    open_file, more_open = suite("run68b", "--shared-from", work + "/laid-open")
    open_vouched, more_ov = suite("run68c", "--shared-from", work + "/laid-open", "--i-trust-every-account-on-this-machine")
    os.mkdir(work + "/laid-sticky")
    os.chmod(work + "/laid-sticky", 0o1777)  # sticky, as /tmp: others could still add a name of their own before the copy
    sticky_shared, more_sticky = suite("run68d", "--shared-from", work + "/laid-sticky")
    os.symlink(work + "/laid-open", work + "/laid-link")  # the folder given by a link: refused, and nothing is copied
    link_shared, more_link = suite("run68e", "--shared-from", work + "/laid-link", "--i-trust-every-account-on-this-machine")
    try:  # taken on the option, the copy still drops the write bits of group and others
        open_laid_mode = oct(os.stat(work + "/run68c/shared/inner/world.txt").st_mode & 0o7777)
    except OSError as fault:
        open_laid_mode = fault.strerror
    check(open_file.returncode == 2 and open_file.stderr.startswith("run_suite.py: --shared-from: %s/laid-open/inner/world.txt, below the "
                                                                    "folder %s/laid-open, can be written by its group or by other "
                                                                    "accounts" % (work, work))
          and sticky_shared.returncode == 2 and sticky_shared.stderr.startswith("run_suite.py: --shared-from: %s/laid-sticky can be "
                                                                                  "written by its group or by other accounts" % work)
          and link_shared.returncode == 2 and link_shared.stderr.startswith("run_suite.py: --shared-from: %s/laid-link is a symbolic "
                                                                              "link" % work) and not os.path.exists(work + "/run68e")
          and not os.path.exists(work + "/run68b") and not ends(open_file)
          and open_vouched.returncode == 0 and ends(open_vouched) == [("passed", "reads")] and open_laid_mode == "0o644"
          and "; the folder given with --shared-from holds or lies below %s/laid-open/inner/world.txt" % work in open_vouched.stdout.split("\n")[0],
          "shared from, open", "refused: %s; with the option: %s; its copy's mode %s; sticky: %s; a link: %s"
          % (more_open, more_ov, open_laid_mode, more_sticky, more_link))  # fmt: skip
    check(run.returncode == 0 and ends(run) == [("passed", "reads")] and laid_mode == "0o755"
          and "run_suite.py: 1 file laid in $S from %s/laid (--shared-from)" % work in run.stdout.split("\n")
          and linked.returncode == 2 and linked.stderr.startswith("run_suite.py: --shared-from: %s/laid/a-link is neither" % work)
          and not os.path.exists(work + "/run68")  # rejected before the work folder is created
          and not ends(linked) and nothing(holds, more_holds, "run_suite.py: --shared-from: %s/laid holds the work folder" % work),
          "shared from", seen + "; laid mode %s; with a link: %s; holding the work folder: %s" % (laid_mode, more, more_holds))  # fmt: skip
    # end_the_groups() ends rows that are not on the GPU: a group that ignores SIGTERM gets SIGKILL.
    stubborn = subprocess.Popen(["bash", "-c", "trap '' TERM; touch %s/stubborn-ready; exec sleep 30" % work], start_new_session=True)
    for _ in range(200):  # up to 10 s, until its trap is set
        if os.path.exists(work + "/stubborn-ready"):
            break
        time.sleep(0.05)
    run_suite.end_the_groups([stubborn], 1)
    check(stubborn.wait() == -signal.SIGKILL, "a stubborn row is killed",
          "status at the end %r" % stubborn.returncode)  # fmt: skip
    def lives_then_go(group, shared):
        """For a row of this test whose process group the copy has left running: finds whether the
        group still has a member and got no SIGTERM, lets it end by itself, and returns the finding.
        `group`, the number that the copy printed, must be the one that the row wrote into the file
        group in `shared`. On SIGTERM a trap in the row's command creates the file signalled there,
        and its loop ends once the file go is there, or after about three minutes. This test sends
        the group no signal."""
        try:
            with open(shared + "/group") as f:
                written = int(f.read())
        except (OSError, ValueError):
            written = -1
        lives = False
        if group > 1 and group != os.getpgrp() and group == written:
            try:
                os.killpg(group, 0)  # signal 0: asks only whether the group has a member
                lives = True
            except OSError:
                pass
        time.sleep(0.5)  # a SIGTERM, had one come, is written down at the end of the row's current fifth of a second
        signalled = os.path.exists(shared + "/signalled")
        open(shared + "/go", "w").close()  # the row's loop ends within a fifth of a second
        return lives and not signalled

    # run_suite.py sends a row on the GPU no signal while it may run on the card, not even when a
    # signal ends run_suite.py itself: ending a program while its GPU kernel runs can make the driver
    # take the whole machine out of use. (A row of kind gpu that only waits for the card's lock is
    # ended: see «the card's lock».)
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| spins | gpu | - | any | 1 | 1 | `0` | `cut -d' ' -f5 /proc/$$/stat > $S/group; trap 'touch $S/signalled' TERM; touch $S/up; for i in $(seq 900); do [ -e $S/go ] && break; sleep 0.2; done` |\n")  # fmt: skip
    call = [sys.executable, "-B", work + "/tree/run_suite.py", "gpu", work + "/run70", "--jobs", "1", "--unconfined"]
    bare_env = {n: v for n, v in os.environ.items() if n not in ("GPU_PREFIX", "GPU_LOCK")}
    with subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=bare_env) as stopped:
        for _ in range(300):
            if os.path.exists(work + "/run70/shared/up"):
                break
            time.sleep(0.1)
        stopped.send_signal(signal.SIGTERM)
        out, err = stopped.communicate(timeout=120)
    named = re.search(r"A GPU PROGRAM STILL RUNS: row spins, process group (\d+)\.", out)
    members = lives_then_go(int(named[1]) if named else 0, work + "/run70/shared")
    check(stopped.returncode == 1 and members and out.rstrip().endswith("the rows ran UNCONFINED (--unconfined)"), "a GPU row at a stop",
          "status %r; members %r; %r; %r" % (stopped.returncode, members, out[-600:], err[-300:]))  # fmt: skip
    # The code of run_suite.py that deals with what "leaves" leaves running also deals with a row on
    # the GPU at its time limit, which is ten minutes or more and is not tried here. "later" needs
    # "waits", which ends 45 s after "leaves": past the 10 s that run_suite.py waits for the group
    # to end.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| leaves | gpu | - | any | 1 | 1 | `0` | `cut -d' ' -f5 /proc/$$/stat > $S/group; ( trap 'touch $S/signalled' TERM; for i in $(seq 900); do [ -e $S/go ] && break; sleep 0.2; done ) & touch $S/up; exit 3` |\n"
                "| waits | cpu | - | any | 1 | 0.1 | `0` | `for i in $(seq 300); do [ -e $S/up ] && break; sleep 0.1; done; sleep 45` |\n"
                "| later | gpu | waits | any | 1 | 0.1 | `0` | `touch $S/later` |\n")  # fmt: skip
    left, seen_left = suite("run71", "--jobs", "2", kind="all")
    named = re.search(r"A GPU PROGRAM STILL RUNS: row leaves, process group (\d+)\.", left.stdout)
    stand_ins = lives_then_go(int(named[1]) if named else 0, work + "/run71/shared")
    # A status 3 stays an ACCEPT, though the row also left a program running.
    check(left.returncode == 3 and stand_ins and ("ACCEPT", "leaves") in ends(left) and ("not run", "later") in ends(left)
          and "the GPU program of leaves still runs on the card" in left.stdout
          and not os.path.exists(work + "/run71/shared/later"), "no GPU row after one left running",
          seen_left + "; stand-ins %r" % stand_ins)  # fmt: skip
    # A symbolic link that the look read and followed passes; one laid where the look found no name, after it first
    # looked at that name and before it examined it (a race, stood in for by racing_lstat, which lays the link), is never
    # a pass; nor is one swapped or given a second name after the look read it.
    os.makedirs(work + "/race/real/inner")
    os.symlink(work + "/race/real", work + "/race/link")
    read_link = run_suite.others_can_change(work + "/race/link/inner")
    # a link below a link: the typed name of the inner one is not the name under which the look read it
    os.makedirs(work + "/race2/data/a")
    os.makedirs(work + "/race2/x")
    os.symlink(work + "/race2/data/a", work + "/race2/a")
    os.symlink(work + "/race2/x", work + "/race2/data/a/b")
    below = run_suite.others_can_change(work + "/race2/a/b")
    lstat, readlink, laid, swapped, hard = os.lstat, os.readlink, [], [], []

    def from_the_look():  # stand-ins act only on the look's own calls
        return sys._getframe(2).f_code.co_name == "looked_in"

    def racing_lstat(path, *args, **kwargs):
        if path == work + "/race/late" and not laid and from_the_look():  # not there when looked at: then the other lays it
            laid.append(True)
            try:
                return lstat(path, *args, **kwargs)
            finally:
                os.symlink(work + "/race/real", path)
        return lstat(path, *args, **kwargs)

    def swapping_readlink(path, *args, **kwargs):
        if path == work + "/race/hard" and not hard and sys._getframe(1).f_code.co_name == "looked_in":
            hard.append(True)  # read, then given a second name: the same inode and target, a new change time
            try:
                return readlink(path, *args, **kwargs)
            finally:
                time.sleep(0.02)  # past one clock tick of the file system's change times
                os.link(path, work + "/race/hard-too", follow_symlinks=False)
        if path == work + "/race/swap" and not swapped and sys._getframe(1).f_code.co_name == "looked_in":  # read, then swapped for another trusted link
            swapped.append(True)
            try:
                return readlink(path, *args, **kwargs)
            finally:
                os.unlink(path)
                os.symlink(work + "/race/other", path)
        return readlink(path, *args, **kwargs)

    # Whether this file system's change times move within a short while: where they are coarse (one second), giving
    # a link a second name may leave its change time as it was, and that one case cannot be checked here.
    fine_times = True
    for probe in ("1", "2"):  # twice in a row: one probe can cross a second boundary on a one-second file system
        os.symlink(".", work + "/race/tick" + probe)
        tick_before = os.lstat(work + "/race/tick" + probe).st_ctime_ns
        time.sleep(0.02)
        os.link(work + "/race/tick" + probe, work + "/race/tick-too" + probe, follow_symlinks=False)
        fine_times = fine_times and os.lstat(work + "/race/tick" + probe).st_ctime_ns != tick_before
        os.unlink(work + "/race/tick" + probe)
        os.unlink(work + "/race/tick-too" + probe)
    if not fine_times:
        print("a link laid during the look: one case not checked here: this file system's change times are too coarse", flush=True)
    os.makedirs(work + "/race/other/inner")
    os.symlink(work + "/race/real", work + "/race/swap")
    os.symlink(work + "/race/real", work + "/race/same")
    os.symlink(work + "/race/real", work + "/race/hard")
    same_as_read = []

    def same_identity_lstat(path, *args, **kwargs):  # every lstat of it shows the first link's identity (inode and times)
        if path == work + "/race/same":
            if not same_as_read:
                same_as_read.append(lstat(path, *args, **kwargs))
            return same_as_read[0]
        return racing_lstat(path, *args, **kwargs)

    def same_identity_readlink(path, *args, **kwargs):  # read, then swapped for a link elsewhere that looks the same
        if path == work + "/race/same" and from_the_look():
            try:
                return readlink(path, *args, **kwargs)
            finally:
                os.unlink(path)
                os.symlink(work + "/race/other", path)
        return swapping_readlink(path, *args, **kwargs)

    os.lstat, os.readlink = racing_lstat, swapping_readlink
    try:
        raced = run_suite.others_can_change(work + "/race/late/inner")
        swap = run_suite.others_can_change(work + "/race/swap/inner")
        hard_linked = run_suite.others_can_change(work + "/race/hard/inner")
        os.lstat, os.readlink = same_identity_lstat, same_identity_readlink
        same = run_suite.others_can_change(work + "/race/same/inner")
    finally:
        os.lstat, os.readlink = lstat, readlink
    check(read_link is None and below is None and laid and raced == (work + "/race/late", run_suite.UNSEEN)
          and swapped and swap == (work + "/race/swap", run_suite.UNSEEN) and same == (work + "/race/same", run_suite.UNSEEN)
          and (hard and hard_linked == (work + "/race/hard", run_suite.UNSEEN) or not fine_times),
          "a link laid during the look", "a link that was read: %r; a link below a link: %r; laid: %r; a link laid after its "
          "name was looked at: %r; swapped: %r; a link swapped after it was read: %r; one swapped with the same identity: %r; "
          "a second name given: %r; then: %r" % (read_link, below, laid, raced, swapped, swap, same, hard, hard_linked))  # fmt: skip
    # A name of 300 characters is longer than a file system allows, so it cannot be examined.
    missing = run_suite.others_can_change(work + "/no-such-place/below")
    too_long = run_suite.others_can_change(work + "/" + "x" * 300 + "/below")
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| aside | cpu | - | any | 1 | 0.1 | `0,77` | `echo '%s' $W; exit 77` |\n" % run_suite.ASIDE)
    stood, seen_stood = suite("run72")
    gone_run = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/no-such-place/w",
                               run_suite.TRUST_EVERY_ACCOUNT], capture_output=True, text=True)  # fmt: skip
    unseen_run = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/" + "x" * 300 + "/w",
                                 run_suite.TRUST_EVERY_ACCOUNT], capture_output=True, text=True)  # fmt: skip
    os.mkdir(work + "/open-place", 0o777)
    os.chmod(work + "/open-place", 0o777)  # open, not sticky: the option takes it, but not the missing folder below it
    below_open = subprocess.run([sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/open-place/missing/w",
                                 run_suite.TRUST_EVERY_ACCOUNT], capture_output=True, text=True)  # fmt: skip
    check(gone_run.returncode == 2 and "does not exist" in gone_run.stderr and not os.path.exists(work + "/no-such-place")
          and below_open.returncode == 2 and os.listdir(work + "/open-place") == []
          and unseen_run.returncode == 2 and "which could not be examined" in unseen_run.stderr
          and missing == (work + "/no-such-place", run_suite.GONE) and too_long == (work + "/" + "x" * 300, run_suite.UNSEEN)
          and stood.returncode == 0 and ("in part", "aside") in ends(stood)
          and "1 passed in part (status 77: their last lines say what was not run; 1 of them stood aside below an open place: " in stood.stdout,
          "a failed look", "missing: %r; too long: %r; %s" % (missing, (too_long or ("",))[0][-20:], seen_stood))  # fmt: skip
    # The card's lock, taken by the runner before it ends a row on the GPU: where it is free the row is ended while the
    # runner holds it; where another holds it, nothing is ended.
    card = work + "/card.lock"
    open(card, "w").close()
    ended_free = []
    free = run_suite.card_free(card, lambda: ended_free.append(1) or True, wait=1)
    holder = subprocess.Popen(["flock", card, "sleep", "5"])
    for _ in range(100):  # until the holder has the lock
        if subprocess.run(["flock", "-n", card, "true"]).returncode != 0:
            break
        time.sleep(0.05)
    ended_held = []
    held_free = run_suite.card_free(card, lambda: ended_held.append(1), wait=1)
    holder.wait()
    # A row whose flock has ended may have left a program that holds no lock: the lock is free, but the row's leader has
    # ended, so the runner's then() ends nothing (the left program ends by itself a moment later).
    leader = subprocess.Popen(["flock", card, sys.executable, "-I", "-c", "import subprocess; subprocess.Popen(['sleep', '2'])"],
                              start_new_session=True)  # fmt: skip
    while run_suite.end_of(leader, 0.1) is None:
        pass
    left_alone = not run_suite.card_free(card, lambda: run_suite.end_if_waiting(leader, 1), wait=1)
    # and a row whose flock still waits (another holds the lock) is ended by end_if_waiting()
    holder = subprocess.Popen(["flock", card, "sleep", "2"])
    for _ in range(100):
        if subprocess.run(["flock", "-n", card, "true"]).returncode != 0:
            break
        time.sleep(0.05)
    waiter = subprocess.Popen(["flock", card, "sleep", "30"], start_new_session=True)
    time.sleep(0.3)
    waited_ended = run_suite.end_if_waiting(waiter, 1)
    waiter_status = waiter.wait(timeout=30)
    holder.wait()
    stranger = run_suite.card_free(card, lambda: True, wait=1, ident=(0, 0))  # another file than the rows' lock: not taken
    leader.wait()
    check(free and ended_free == [1] and not held_free and not ended_held and left_alone and not stranger and waited_ended
          and waiter_status == -signal.SIGTERM, "the card's lock",
          "free: %r, called %r; held by another: %r, called %r; after the leader ended: left %r; another file: %r; "
          "a waiter: ended %r, status %r" % (free, ended_free, held_free, ended_held, left_alone, stranger, waited_ended,
                                             waiter_status))  # fmt: skip
    # others_can_change() checks GPU_LOCK's folder before the lock file is made there: below an open folder, refused
    # (status 2) and no file made; a sticky one, as /tmp, passes the check.
    os.mkdir(work + "/open-lock", 0o777)
    os.chmod(work + "/open-lock", 0o777)
    with open(work + "/tree/SUITE.md") as f:
        table_before = f.read()
    with open(work + "/tree/SUITE.md", "w") as f:  # one row on the GPU, which the refusal keeps from running
        f.write(head + "| lockrow | gpu | - | any | 1 | 0.1 | `0 ^x$` | `echo x` |\n")
    lock_open, more_lock = suite("run-lock1", kind="gpu", env=dict(bare, GPU_LOCK=work + "/open-lock/gpu.lock"))
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(table_before)
    os.mkdir(work + "/sticky-lock", 0o777)
    os.chmod(work + "/sticky-lock", 0o1777)
    sticky_found = run_suite.others_can_change(work + "/sticky-lock")
    check(nothing(lock_open, more_lock, "run_suite.py: the card's lock (GPU_LOCK) ") and os.listdir(work + "/open-lock") == []
          and sticky_found is None, "the lock's folder", "open: %s; sticky: %r" % (more_lock, sticky_found))  # fmt: skip
    # run_suite.py reads further tables next to SUITE.md (a SUITE-<word>.md whose first line is
    # run_suite.FURTHER) as part of the same table. Their rows may need SUITE.md's, not the reverse,
    # and no name may be in two. Any other SUITE-<word>.md is passed over, unless it holds a table.
    further = work + "/tree/SUITE-OTHER.md"
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| public | cpu | - | any | 1 | 0.1 | `0 ^p$` | `echo p` |\n")
    alone, seen = suite("run54")
    with open(further, "w") as f:
        f.write(run_suite.FURTHER + "\n" + head + "| other | cpu | public | any | 1 | 0.1 | `0 ^i$` | `echo i` |\n")
    with open(work + "/tree/SUITE-PAGE.md", "w") as f:  # a page whose name begins with SUITE-, a list of names, no table
        f.write("# Rows struck from SUITE.md\n\n- page\n| page | a line of a page, not of a table of tests |\n")
    both, more = suite("run55")
    seen += "; both: " + more
    with open(further, "w") as f:
        f.write(run_suite.FURTHER + "\n" + head + "| public | cpu | - | any | 1 | 0.1 | `0` | `true` |\n")
    twice, more_twice = suite("run56")
    seen += "; a name in both: " + more_twice
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| public | cpu | other | any | 1 | 0.1 | `0` | `true` |\n")
    with open(further, "w") as f:
        f.write(run_suite.FURTHER + "\n" + head + "| other | cpu | - | any | 1 | 0.1 | `0` | `true` |\n")
    backwards, more = suite("run57")
    seen += "; a public row that needs a further table's: " + more
    os.remove(further)
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| public | cpu | - | any | 1 | 0.1 | `0 ^p$` | `echo p` |\n")
    with open(work + "/tree/SUITE-PAGE.md", "w") as f:  # a table of tests that lost its first line: refused, not passed over
        f.write(head + "| lost-line | cpu | - | any | 1 | 0.1 | `0` | `true` |\n")
    unmarked, more_unmarked = suite("run57u")
    seen += "; a table without its first line: " + more_unmarked
    os.remove(work + "/tree/SUITE-PAGE.md")
    check(alone.returncode == 0 and ends(alone) == [("passed", "public")] and both.returncode == 0
          and ends(both) == [("passed", "other"), ("passed", "public")]
          and nothing(twice, more_twice, "run_suite.py: SUITE-OTHER.md, line 4: the row does not have eight cells, or its name is malformed or already used")
          and nothing(backwards, more, "run_suite.py: SUITE.md: public needs other, which is not a row")
          and nothing(unmarked, more_unmarked, "run_suite.py: SUITE-PAGE.md holds a table of tests, but its first line is not"),
          "two tables", seen)  # fmt: skip
    # A status 3 must not be lost when run_suite.py can no longer print.
    with open(work + "/tree/SUITE.md", "w") as f:
        f.write(head + "| accepts | cpu | - | any | 1 | 0.1 | `0` | `sleep 1; exit 3` |\n")
    call = [sys.executable, "-B", work + "/tree/run_suite.py", "cpu", work + "/run46", "--unconfined"]
    with subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as run:
        first = run.stdout.readline()  # the first line, printed before the row starts
        run.stdout.close()  # no reader is left: the copy's next print gets a broken pipe
        said = run.stderr.read()
    check(run.returncode == 3 and first.startswith("run_suite.py: ") and "status 3" in said, "an accept, and no standard output",
          "status %d; first line %r; standard error %r" % (run.returncode, first, said[-300:]))  # fmt: skip
    # own_bin(): root changes no owner and prints no command; where bin/ is root's in a tree that another account owns, one note
    me, seen_bins = (os.geteuid(), os.getegid()), {}
    root = os.geteuid() == 0
    # twists, run in the child before own_bin(): it plays an ordinary account (4242) that cannot write in bin/; its first look
    # at bin/ says «missing»; its standard output holds only ASCII; its umask gives the group write
    ordinary = "import os\nos.geteuid = lambda: 4242\nos.access = lambda *a, **k: False\n"
    meanwhile = ("import os.path\nreal_isdir, looks = os.path.isdir, []\ndef isdir(path):\n    if str(path).endswith('/bin') and not looks:\n"
                 "        looks.append(path)\n        return False\n    return real_isdir(path)\nos.path.isdir = isdir\n")  # fmt: skip
    twists = {"later, without root": ordinary, "later, in root's tree": ordinary, "made meanwhile": meanwhile,
              "a path ascii cannot hold": "import sys\nsys.stdout.reconfigure(encoding='ascii')\n", "open to others": "import os\nos.umask(0o002)\n"}
    places = {"an odd path": "own-bin-odd\x1b[2K", "a path ascii cannot hold": "own-bin-r\u00e9po"}
    cases = [("plain", None), ("there", "folder"), ("a file", "file"), ("through a link", "link")]
    cases += [("another's", None), ("an odd path", None), ("later, without root", "root's bin"), ("later, in root's tree", "root's bin"),
              ("made meanwhile", "folder"), ("a path ascii cannot hold", None), ("open to others", None)] if root else []
    for case, lies in cases:
        place = work + "/" + places.get(case, "own-bin-" + case.replace(" ", "-").replace("'", "").replace(",", ""))
        os.mkdir(place, 0o700)
        top = place
        if lies in ("folder", "root's bin"):
            os.mkdir(place + "/bin")
        elif lies == "file":
            open(place + "/bin", "w").close()
        elif lies == "link":  # the tree's top reached through a link, as a user may call run_suite.py
            os.mkdir(place + "/real", 0o700)
            os.symlink("real", place + "/top")
            top = place + "/top"
        if root and case not in ("plain", "there", "a file", "through a link", "later, in root's tree"):  # the top belongs to another account
            os.chown(place, 4242, 4243)
        if case == "made meanwhile":  # as if that account had made bin/ right after own_bin() looked
            os.chown(place + "/bin", 4242, 4243)
        call = subprocess.run([sys.executable, "-I", "-B", "-c", twists.get(case, "") + "import sys; sys.path.insert(0, %r); import run_suite; "
                               "run_suite.own_bin(%r); run_suite.own_bin(%r)" % (work + "/tree", top + "/bin", top + "/bin")],
                              capture_output=True, text=True, env={n: v for n, v in os.environ.items() if not n.startswith("SUDO_")})  # fmt: skip
        said = [line for line in call.stdout.splitlines() if line.startswith("run_suite.py: ")]
        about = os.lstat(top + "/bin")
        seen_bins[case] = (call.returncode, call.stderr.strip()[-200:], os.path.isdir(top + "/bin") and not os.path.islink(top + "/bin"),
                           (about.st_uid, about.st_gid), said, place)  # fmt: skip
    # the note's words, written out here so that a change to them in run_suite.py fails this check
    words = ("run_suite.py: %(folder)s belongs to root, in %(top)s, which belongs to the account with uid %(uid)d. A build that does not run "
             "as root cannot write into it. To go on without root, start again in a new copy of the repository. That the directory is "
             "root's does not protect it: whoever owns %(top)s can replace it, cp_verify included.")  # fmt: skip
    def note(case, times):  # the note, whole, as many times as own_bin() should print it
        place = run_suite.printable(seen_bins[case][5])
        return seen_bins[case][4] == [words % {"folder": place + "/bin", "top": place, "uid": 4242}] * times
    check(all(seen_bins[c][0] == 0 and seen_bins[c][2] and seen_bins[c][3] == me and not seen_bins[c][4] for c in ("plain", "there", "through a link"))
          and seen_bins["a file"][0] == 1 and "FileExistsError" in seen_bins["a file"][1] and not seen_bins["a file"][2] and not seen_bins["a file"][4]
          and (not root or (all(seen_bins[c][0] == 0 and not seen_bins[c][1] and seen_bins[c][2] for c in ("another's", "an odd path", "later, without root", "later, in root's tree", "made meanwhile", "a path ascii cannot hold", "open to others"))
                            and all(seen_bins[c][3] == (0, 0) for c in ("another's", "an odd path", "later, without root", "later, in root's tree", "a path ascii cannot hold", "open to others"))
                            and note("another's", 1) and note("an odd path", 1) and "\\x1b[2K" in seen_bins["an odd path"][4][0] and note("later, without root", 2)
                            and seen_bins["made meanwhile"][3] == (4242, 4243)
                            and not any(seen_bins[c][4] for c in ("later, in root's tree", "made meanwhile", "a path ascii cannot hold", "open to others")))),
          "bin under sudo", repr(seen_bins))  # fmt: skip
    print("run_suite_test.py: %d checks as expected, %d otherwise" % tuple(tally))
    return 1 if tally[1] else 0


if __name__ == "__main__":
    sys.exit(main())
