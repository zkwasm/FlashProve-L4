# tests

This folder holds a few tests that check the repository as a whole, a Python model of two hash rules that the proof files' format does not use, and the runner of the tests of the shared library in `blocks/`. The tests of the programs lie mostly in `prover/tests/` and `verifier/tests/`. [../SUITE.md](../SUITE.md) lists the tests that `../run_suite.py` runs, with what each needs and its command; the library's runner and the second verifiers below can also be run by hand. Start at [README.md](../README.md).

Everything runs on synthetic weights.

## Run tests through the suite

Type the commands in the top folder, one level above tests/. [../SUITE.md](../SUITE.md), "Run the tests", says what the suite needs, where it builds and writes, how to choose tests and what its exit status means. The command below runs one short test, which runs docs/format/format_rules.py (SUITE.md gives it 0.1 minutes). In zsh, type them without the words from `#` on, or run `setopt interactivecomments` first, because by default, at its prompt zsh takes those words as part of the command line:

    D=$(mktemp -d)    # a new folder, by its absolute path; ${D:?} below stops a command if D is not set
    ./run_suite.py cpu "${D:?}/work" --only '^sources\.format_vectors$'

Each test gets one line with its state, its name, its seconds, its status and its own last line, and the run's last line counts them; each test's whole output is in <D>/work/screens/<test>.txt.

## Which programs to use

prover/build.sh builds the programs without test options into hc/ of its output folder. With the word tests it builds instead cp_prove and cp_commit with the test options into hc/test_build/, under the same names, and these refuse a model's files (the head of prover/build.sh says so; no test in SUITE.md runs that refusal); it also builds make_model, which writes the tests' small synthetic models.

If make_model fails, it removes the files that the run has created through a descriptor of the folder that it made, whatever has happened to that folder's path since, and the folder itself only if it is empty and its path still names it. Two narrow stretches stay open, by the source's own comments: another folder of the same user's that is put at the path between the making and the opening is not noticed (nothing in it is overwritten, and only the run's own names are removed), and something else could take the path between the last look and the removal, which removes nothing but an empty folder. So name a new folder below one that only you can write.

build.sh builds no cheating program. The programs with the test options also read test-only environment variables, which can make them print internal values, change a value or fail on purpose, and only on synthetic weights with the option -test-seed (cp_commit) or -test-coins (cp_prove) (prover/common/test_words.h). Some of these variables name where a test build writes files. `CHAIN_PUT_VECTORS` names a folder, in which cp_prove writes `link_<N>.vectors` and `link_<N>.values`: both are created, never written over: if anything already has such a name, a symbolic link included, the run ends with an internal error. `CHAIN_ROWS_OUT`, `COINS_LEDGER` and `COINS_CONTEXTS` each name a file, which is created or, if it exists, added to at its end. `CHAIN_HELD_STEM` (`<stem>.bin`, `<stem>.list`, created, never written over: the process ends if anything already has one of the two names, a symbolic link included) is read only by a test build of cp_commit that build.sh does not make. `ROWS_DUMP` is read only by routines that no program of this repository calls (prover/weights_commitment/cert_doors.h, whose head says that cp_commit calls only once_key_fault() and cp_prove calls none), so setting it writes nothing. A symbolic link at one of the three names that are added to is followed, and so is one at the folder that `CHAIN_PUT_VECTORS` names or at a folder above any of these names, so give each run a new folder of your own.

Use only the programs that are directly in a folder hc (hc/cp_commit, hc/cp_prove, hc/cp_verify), never one from hc/test_build/ or below it; put no other program of these names on your PATH, and copy none out.

## Run one script alone

Type the commands in tests/, with `..` for the top folder. Each script gives its command line in a usage text near its top; SUITE.md's row for it gives the command that the suite runs. The sandbox around each test (tools/confine.py) comes from the runner: a script that you start yourself, or a row's command that you type yourself, runs outside it, with all the rights of your account. dangerous_calls.tsv, below, counts file by file the lines that delete recursively or end processes, with a reason given for each; it does not cover single removals, renames or emptying a file.

## The files

| file | what it is |
|---|---|
| dangerous_calls_test.py, dangerous_calls.tsv | reads the repository's scripts and source files without their comments, and the parts of pages that can be run: it refuses five forms that aim a signal or a removal at the wrong target, and counts, per file and kind, the lines that end processes or delete recursively against the table of reasons in dangerous_calls.tsv; one line more or fewer than the table says turns the test red. It matches text: a form it doesn't know goes unnoticed (the test sources.dangerous_calls) |
| folder_look_copies_test.py | reads the repository's shell and Python files and checks that each copy of the routine that refuses a work folder below a place that another account can change (open_above() in the shell scripts, others_can_change() in the Python ones) has the agreed text, and likewise each copy of available_memory(), which reads how much memory may be used, so that a fix made to one copy cannot leave another behind. It runs no script and writes nothing (the test sources.folder_look_copies) |
| s8_model.py, s8_known_answers.tsv | a model in Python of two hash rules that the proof files' format does not use (the proof-of-work test and the folded leaf of a hash tree), which shares no code with the programs except BLAKE2b's compression function, adapted, as theirs is, from the BLAKE2 reference code in C; and the known answers that a unit test of the library reads, blocks/hash/tests/older_test.cpp (the test t-s8_model) |
| new/run_new.py | runs the shared library's tests in stages: lists about the tests compared with the sources, the fence that keeps tests' switches out of serving builds and the sources that must not compile, its honest unit tests, copies of the library with one piece of text, or a few, replaced (in which an honest test must then fail; here the stage makes no copy and says so), whole toy runs, damaged files, and the second verifiers. Without --whole-tree, its stage `table` reports two things in its one-line result instead of failing on them (names in blocks/fence.h that no source holds, and, unless --left-out is given, test sources that blocks/tests/TESTS does not list); with --whole-tree each is a fault. Called by hand, it hands every program, the compilers too, your environment with TMPDIR set below the work folder: with `DEPENDENCIES_OUTPUT` or `SUNPRO_DEPENDENCIES` set, g++ appends to the file that the variable names, wherever that is; the script's head says so too. Unset both before such a call; the suite's runner does so itself (read in the code). |
| new/second/ | second verifiers, written in Python, of the shared library's toy files ([new/second/README.md](new/second/README.md)) |
