# blocks/tests/: tests over whole proofs

Each block's own tests are in its folder (`<folder>/tests/`). This folder holds what concerns whole proofs and what concerns all blocks together. [`../../tests/new/run_new.py`](../../tests/new/run_new.py) runs
all of these tests (its stage `bytes` also asks for a program that this repository does not have: the table below).

| file | what |
|---|---|
| `whole_runs.h` | the whole runs of `toy/` behind one interface: plan from free choices, prove from a key for the coins, check bytes |
| `damage.h` | damage to the bytes of a proof or a file, for any checker: bits flipped, a word at or above p (w + p where 64 bits hold it) in the place of eight bytes, every shorter length, bytes after the end; the work is shared among child processes. Also used by `frame/tests/check_test.cpp` and `check_entry_test.cpp` on the frame's files at the real target: a sample in `make test`, every 7th byte in the runner's stage `bytes`, every byte with `--stride 1`. That stage also asks for a third program, `check_key_test`, whose source is not in this repository, so here it reports a fault for it |
| `every_byte_test.cpp` | that damage to the whole toy proofs: all rejected. As `make test` runs it, it flips one bit of every 13th byte; every bit of every byte only in the runner's stage `bytes` (tests/new/run_new.py) |
| `other_parameters_test.cpp` | a proof made at one set of parameters, given to a verifier at another: rejected both ways, by a plain no; 28 variations, and a public file with one byte more, from which the same run follows |
| `proof_file.cpp` | each whole run's honest proof, made and checked |
| `TESTS` | the library's honest tests: the list for the runner's `--tests`. It lists all 89 sources that `make test` builds and runs, so the suite's row `blocks.tests` runs every one of them |

## How to run them

In `blocks/`, `make OUT=<a new folder> test` builds the library and runs its tests; `make OUT=<a new folder>` with no target does what the target `all` does: `test`, `toy` and `programs`, so it runs all 89 tests too ([`../README.md`](../README.md), "Build and test", says what one of them costs). `prover/build.sh` builds cp_commit, cp_prove and cp_verify.

The runner's stage `table` holds two lists against the sources. A macro that a source holds and whose name marks it as test-only (`fence.h` says which names do) must be in `fence.h`'s list, or the stage fails. A name in that list that no source holds, and a test source that `TESTS` does not list,
are only named in the stage's line; the second is a fault in a call with `--left-out`, both in a call with `--whole-tree`. With `--whole-tree` the stage fails here in any case, because it then asks for a table and a list that this repository does not have.

## Cheating provers

No test of this folder or of a block's folder is built with a cheating switch. A cheating switch is a compile-time switch under which a test build does something wrong on purpose: most make a prover or a committer cheat, some make public inputs contradict what runs, and some plant a fault that breaks hiding. The code under such switches is not in this repository. Tests' own switches are another kind (`fence.h` lists them): some of them let a test disable checks, choose the private randomness or call functions that only a test may call; one keeps what the library's prover sends out of its own hash chain, for tests of hiding. Verifiers, which are unchanged, are designed to reject such a proof, because the verifier derives its challenges from the prover's messages through its own hash chain; no test here gives a verifier such a proof. Some tests of this folder set one of these switches in their own source. Some tests here give a check a false claim or a false value, which the test's own code makes.

This library's tests with cheating switches do not ship, and a pass of one rests on a coarse comparison of the reasons of refusal that a program printed with those that a table named for it.

## Teeth

The stage `teeth` of `../../tests/new/run_new.py` asks whether the honest tests bite: each row of a table replaces one piece of text in the library, or a few, and the honest test that it names must then fail. No such table ships: the stage reads a table only if the call names one (`--teeth`), so in this repository it says that none was given and makes no copy, and no run of it on these sources is reported here. It works in scratch copies under the runner's work folder and nowhere else, and it refuses a row whose files or program lead out of `blocks/`. A table's replacement texts are compiled and run, so give `--teeth` no table that you would not run as a program.

## What these tests do not show

Most blocks' verifiers check nothing themselves: they pass rows with public targets on, and a false row is rejected at the end of the proximity test, by
one of BASE's two checks. Such a rejection does not say **which** row was false: only a block's own test can ask that.

Where something is bound in **two ways**, each of which is enough, no test of the real sources can see the loss of one. The public file's digest is bound like this (an item of the first query's parameters; the start of the vertices' hashes), and the first of these ways is itself double: the parameters' digest enters the first query as it is and once more inside the run's value. On test builds at toy size, not on the released programs, `other_parameters_test.cpp`'s proofs were still refused with the digest taken out of either of the two ways, and accepted with it taken out of all three places. No test in this repository repeats that.

Both sides of every run here are compiled from one set of basic files: arithmetic, hashing, the chain, the codes, the plan. If one of these is wrong, both sides are wrong alike, and no test of this folder notices; the known
answers and the second implementations in the tests of the folders that hold these files stand against that. A rejection at toy size with a low target says nothing about the size of an error
term: that is a matter for the accounting (`numbers/`).
