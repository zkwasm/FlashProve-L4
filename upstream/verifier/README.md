# cp_verify

`cp_verify` is the verifier: a C++ program that checks one proof per call. It runs on a CPU and
needs no GPU. This page is a reference for building and calling it;
[docs/verifying.md](../docs/verifying.md) explains how to check a proof safely and when to accept
the result, and [docs/output.md](docs/output.md) is the reference for its output: the closing lines,
the exit codes and the rejection names. To read the source, start at
[docs/source-map.md](docs/source-map.md): it says what each file holds and what the words and short
names in the code mean.

This is research code. It is tested, not verified: no theorem is about this program, and no security
level is claimed for it. What Lean proves is about the protocol and a verifier written in Lean
([guide/what-is-proved.md](../guide/what-is-proved.md)). See [Status and limits](#status-and-limits).

## Build

From the root of the repository, run:

```
verifier/build.sh
```

This compiles `cp_verify` and puts it in `bin/`. The exit code is 0, and the last line of output
begins `THE VERIFIER IS BUILT:` and gives the program's path. A second build also prints a notice
that it overwrote the first. [docs/quickstart.md](../docs/quickstart.md#install) lists what to
install first. Before it compiles, the script prints `build=` and 64 hex digits. That digest covers
the source code, not the compiler or its flags.

Two arguments change the build (a third, `--field 2`, is accepted and changes nothing):

- `- DIRECTORY` builds into another directory. A relative path is resolved against your working
  directory. If the directory does not exist, the script creates only the last path component.
- `--clang - DIRECTORY` compiles with clang++. By report, only version 18.1.3 has been tried†.

**Directories that another account can change.** The script builds nothing (exit code 2) if another
account could tamper with a directory that you name, or with a directory above it, because that
account could replace the program that you are about to run. The message names the directory and the
cause. A new directory inside one created by `mktemp -d` passes. There is one exception, for the
default directory. If another account could tamper with the root of the checkout, or with a
directory above it, as in a clone made under a umask of 002, the script prints a note and builds
anyway. After all, an account with that access can change the script and its sources too. On a
shared machine, do not rely on that build.

**As root.** If the script runs as root in a checkout that belongs to another account, as under
`sudo`, a `bin/` that it makes is root's, and it says so in a note. A later build there that does
not run as root builds nothing (exit code 2) and prints the same words. To go on without root, name
another output directory; the tests and the repository's other scripts look for `cp_verify` in
`bin/`, so for them start again in a new copy of the repository. That `bin/` is root's does not
protect it: whoever owns the checkout can replace it.

**A killed build.** Its work directory, `cp_verify.building.XXXXXX`, can stay in the output
directory, and no later build removes it. If root made it, your own later builds there stop with
"could not be searched" (under that exception, only a note): name another output directory.

**Your environment.** The script does not empty your environment: `PATH` chooses the compiler, and
variables such as `COMPILER_PATH`, `GCC_EXEC_PREFIX` and `CPATH` choose its parts and add headers.
Build from a shell whose environment you know.

**The flags.**

- With g++, `build.sh` passes `-fno-ipa-modref`, because g++ 13.3.0 can silently optimize away the
  stores that fill an object passed by reference. `verifier/tests/compiler_pass_control.sh`
  reproduces the fault on a short program. If you build these sources with g++ some other way, pass
  that option too.
- It passes the usual hardening flags. With them, indexing a `std::vector` past its end with `v[i]`
  kills the program with a signal, and a stage that dies that way counts as not passed. These index
  checks do not cover memory that is reached through a plain pointer or a C array, as the sources do
  in many places; the inner loop of `eqTab()` in `verifier/src/vf_common.h` uses plain pointers to
  skip the checks. A reading† of `eqTab()` and each of its calls found no overrun; nothing was run
  for this, and the other such places have not been gone through one by one. Two tests build the
  verifier with the address sanitizer, and a third builds one of its routines
  (`verifier.sanitized_files`, `verifier.sanitized_walk`, `verifier.ws_build_fuzz` in `SUITE.md`). On
  the files that they try, the sanitizer would report an access past the end of a block of memory,
  also through a plain pointer, but not one that stays inside a vector's own memory. We have not
  measured how much the checks slow this program down.
- It passes `-DSERVING_BUILD=1`, which makes the build fail if a test-only switch listed in
  `blocks/fence.h` is defined on the compiler's command line. It does not catch a switch that is
  missing from that list, or one that a source file defines later.

<a id="run"></a>

## The call

The full [acceptance rule](../docs/verifying.md#when-to-accept) applies to every call. For a call on
a stored proof that you can copy and run, and a rejection, see [the quick
start](../docs/quickstart.md#1-cpu-check-a-proof).

```
bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID
```

- `CONFIG_DIR` holds `policy.txt` (the parameter file), `KL.bin` and `layout.txt`. The program's
  usage text calls it `<config folder>`.
- `PUBLIC_KEY_DIR` holds `key.bin` and the leaf files `leaf/<k>.bin`. The usage text calls it
  `<ivk folder>`.
- `REQUEST_FILE` holds T tokens (the prompt, the reply and filler tokens) and ends with two more
  32-bit words, r and e. They mark the range to check: tokens r + 1 to min(e, T - 1), counted
  from 0. If e = T, the call also checks y, the token that comes after the request.
- `TOKEN_FILE` holds one 32-bit little-endian word: y, or 0 if e is less than T. Write it yourself.
- `PROOF_FILE` is the proof the provider sent.

[docs/verifying.md](../docs/verifying.md#what-you-need-and-where-it-must-come-from) says where each
of these must come from, and how to build the request file.

A whole call runs three stages on the same files. The `key` stage checks the committer's proofs in
`key.bin`. The `leaf` stage ties the leaf file to `key.bin`. The `request` stage checks the proof.

- An option that comes before the paths is read as a path, so put every option after the five paths.
- Always pass `-id`, because a call without `-id` is never accepted.
- Under a clipped gap ([Acceptance policies](../docs/verifying.md#acceptance-policies)), add
  `-policy clipped-gap,C,THETA,CAP,S,SH`. Every value must equal the corresponding line of
  `policy.txt`. If the parameter file sets a clipped gap, a call without `-policy` that checks a
  token inside the window ends in `NO: policy`, with exit code 1.
- `-j N` sets the number of threads for the request stage (1 to 256; the default is 1). The two
  shorter stages run at the same time, each with an eighth as many threads, but at least one.
- `-q` suppresses most progress lines.
- `-show-stages` prints the stages' output even when the call cannot be accepted, except for
  `NO: model-statement` ([when it is left out](docs/output.md#the-closing-lines)). Use it only to
  find out which check failed. After a rejection, that output contains values read from the rejected
  files, so do not trust or copy anything in it. A few checks run after it is printed, so
  `STAGE PASSED` can appear above a final `NOT CHECKED`.
- `-next-token` is a safeguard for a request that asks only about y. With any other request, the
  call ends in `NOT-A-VERIFICATION` at best. "Next token" is the program's name for y. In the
  `covered:` line, `y=yes` means y was checked (e = T), and `next-token=yes` means only y was
  checked.
- `-no-proof-of-work` makes the call end in `NOT-A-VERIFICATION` at best if any round would miss the
  parameter file's target without its proof of work. It does not change the setting, and the
  part-calls take it too.
- Leave `-bits` at its default, 128. It is the minimum you require of every round, not a measure of
  the security you get, and not the width of the weights. A call that asks for less ends in
  `NOT-A-VERIFICATION` at best. A call that asks for more is rejected, because `cp_verify` follows
  the usual convention of counting a 256-bit digest as 128 bits. That convention is not a proved
  bound.

## The output

A whole call that exits with code 0 or 14 ends with four lines: the `commitment:` line, the line on
proof of work, the `result-v1` line and the verdict. See
[docs/output.md](docs/output.md#the-closing-lines) for how points 4 to 8 of the acceptance rule
apply to these lines.

<a id="exit-codes"></a><a id="read-the-result"></a>

## Last line and exit status

Treat the reply as unproved after any exit code other than 0. Do not accept just because the exit
code is 0. The program also returns 0 for `WITHIN LIMITS`, which is not a `YES`, and for a `YES`
that covers only the next token, y. The full [acceptance rule](../docs/verifying.md#when-to-accept)
applies.

[docs/output.md](docs/output.md#exit-codes) has the table of all exit codes and last lines, the
causes of code 14 and the common rejection names, and
[docs/verifying.md](../docs/verifying.md#exit-codes) says what a client does after each.
`verifier/src/vf_rejection_names.txt` lists the names that a stage or a call rejects with, but not
the names that the parent process itself gives a `NOT CHECKED`, such as `internal-stage`.

When a call ends in `NOT CHECKED:`, the name that follows contains `internal-`, as in
`internal-stage` or `G0:internal-...`.

Run it again where the name is `internal-stage` or `internal-exception-parFor`, because the machine
may have run out of memory. The fault may also be in the program itself, and `cp_verify` cannot tell
which, so report the name if it comes back. Any other name will recur with the same files, so please
report it ([SECURITY.md](../SECURITY.md) says how, and what a provider must not send). Rarely, a
valid proof also gets `NOT CHECKED`, because `cp_verify` cannot handle certain challenge values that
the protocol allows. Even then, treat the reply as unproved. A malicious provider may be able to
turn a `NO` into `NOT CHECKED`, so it says nothing in the provider's favour
([more](docs/output.md#exit-codes)).

### Planning calls

A planning call (one with `-plan`) does not check a proof. If it succeeds, its last line is
`PLAN ONLY: no proof was checked`, and it exits with code 4. It can also end in `NO: <name>` (exit
code 1) or with exit code 2, 5 or 15. There are four: `-plan accounting`, `-plan request`,
`-plan arrays` and `-plan id`. [docs/plan-data.md](docs/plan-data.md) describes what they print.

You can try one without a GPU or a proof. This call prints the error accounting for the small
example at the default setting:

```
bin/cp_verify -plan accounting prover/examples/target_small_f2 128
```

A [part-call](../docs/glossary.md#part-call) (`-key`, `-leaf`) checks only one stage. At best it
ends in `PART PASSED: <stage>`, with exit code 3. A `-leaf` result counts only if a `-key` call also
passed on the same files.

`-plan id` prints the commitment ID of the files it is given. It is for the party that publishes the
ID. A part-call that passes or has `-show-stages` prints the ID too, on the line
`computed commitment ID:`, even without `-id`. **Never pass as `-id` an ID that was computed from
files that someone sent you.** It matches those files by construction and says nothing about who
published it, so you would get a `YES` about those files and no others.

<a id="the-result-v1-line"></a>

## The line for programs

The third of the four closing lines begins `result-v1` and is meant for programs. The table in
[docs/output.md](docs/output.md#the-line-for-programs) lists every field and what to compare it
with. The program's usage text names this README for that list. docs/output.md also explains the
line on proof of work and the `above-clip=` field.

## The error accounting

At the default setting every closeness round rests on the theorem on codes, which the programs'
accounting takes as given, and every example configuration counts proof of work, which rests on a
step that is assumed for the programs' hash chains. That our parameters meet the theorem's conditions
is our own reckoning, with no bits to spare, and a `YES` rests both the request's truth and the
binding of the weights on that theorem.

`cp_verify` does its own error accounting for each round (one step of a check), and the verdict
depends on it. Each round's error, with any proof of work counted, must be at most 2^-target, where
the target is the `sec` line of the parameter file. The output lines that begin `accounting:` show
the result. The printed bits are for display only, because the verdict comes from an exact
comparison of every round's error with 2^-`sec`.

This accounting is the program's own calculation, not a security level. `docs/limits.md` explains
[what the figures are and what they are worth](../docs/limits.md#the-soundness-figure), [what they
rest on](../docs/limits.md#what-the-theorem-assumes) and [what the calculation does not
settle](../docs/limits.md#how-the-programs-count).

The provider's parameter file decides how much proof of work is counted and, through its numbers, on which source some closeness rounds rest. Up to a cap of 32 bits a round, unless the caller passes `-no-proof-of-work`, `cp_verify` does not test how much work is counted, and it never tests the other choice, for any closeness round: a request's, a batch's, or the key file's and the leaf file's. The last line does not show either; the work and accounting lines do.

**Every example configuration under `prover/examples/` counts proof of work.** A plain `YES` does
not tell you this, and neither does `WITHIN LIMITS` or a part-call's `PART PASSED`. To find out,
read the output line on proof of work, which `-q` does not suppress. To accept no counted proof of
work, pass `-no-proof-of-work`. A `YES` under it says only that no round leans on work by the
programs' own count. On each of their parameter files the call then ends at best in
`NOT-A-VERIFICATION`, by the programs' own count of rounds that need their work ([point 6 of the
acceptance rule](../docs/verifying.md#when-to-accept)). Counting proof of work rests on a step that
is assumed for the programs' hash chains. ePrint 2021/582, Theorem 6, proves the same step, but in a
different setting. The program prints this caveat too.

The program bases the error bound of each [closeness round](../docs/glossary.md#closeness-round) on
one of two sources:

1. a lemma of the paper that this protocol builds on (A. Chiesa, G. Fenzi, G. Weissenberg,
   "Zero-Knowledge IOPPs for Constrained Interleaved Codes", revised version, 2026; an earlier version:
   ePrint 2026/391): "[the bound within a third of the distance](../docs/glossary.md#proved-lemma)";
2. ePrint 2025/2055, Theorem 4.6 ("the theorem on codes"; [what it is
   worth](../docs/limits.md#what-the-theorem-assumes)), which the programs' accounting takes as
   given. A statement on codes held to follow from that theorem, in its case of two words, is proved
   in this repository's Lean library, by way of another formalization's theorem and a comparison of
   bounds; the paper's own form of the bound is not proved there, and that the library's statement
   follows from the paper's, and that these programs' parameters meet the theorem's conditions, rest
   on reading, not on a proof ([lean/LIMITS.md](../lean/LIMITS.md), L15).

Its output says how many rounds rest on each source, for the key file, the leaf file and the proof
together (the `accounting:` lines). A planning call (`-plan accounting`) also prints these counts
for each of the three files (`footing:`), and `group:` lines that name the source for each group of
rows. Of up to two steps it prints "a bound argued on paper for this protocol": a written argument,
not machine-checked for these programs. At the default setting every closeness round rests on the
theorem on codes (the planner counts none on the bound within a third of the distance): a request's and a batch's, and the
key file's and the leaf files' ([how many, and what else the bound rests
on](../docs/limits.md#what-the-theorem-assumes)); that our parameters meet the theorem's conditions
is our own reckoning, with no bits to spare.

## What a YES does not say

[docs/verifying.md](../docs/verifying.md#what-a-yes-does-not-say) lists what an accepted proof does
not tell a client. A `YES` does not say which source the closeness rounds rest on or, unless you
passed `-no-proof-of-work`, whether proof of work was counted ([The error
accounting](#the-error-accounting) says where to find both). No verifier checks whether the provider
masked the proof or the public files.

Nor does a `YES` say how closely the integer model tracks the floating-point model. The provider
fits the quantization scales (`KL.bin` and eight lines of `policy.txt`), and the programs check only
that they are within the allowed ranges. Scales within those ranges can round every layer's
contribution to zero for ordinary-sized weights.
Scales that shrink the logits make a limit on gaps easier to meet
([SO-6.2](../docs/limits-further.md#so-6.2)). So anyone who publishes a commitment ID, or relies on
one, should read these values or evaluate the integer model once.

**On what it rests.** The committed table has out-of-domain (OOD) answers (masked sums of the
weights; the programs call them a record: line `record 1`). A `YES` rests both the request's truth and the
binding of the weights on the theorem on codes; at the default setting it does so with no margin to spare. The written proof of that
binding, which is not in this repository, is for a commitment ID whose hashed string does not take
the OOD seeds. The programs' ID takes them, so this page claims no binding of the weights for the
programs' own commitment ([binding](../docs/glossary.md#binding)). What Lean does prove — of
`verify`, the verifier written in Lean, that for any commitment, whoever made it, one set of weights
or none is settled before any proof is made; and, for a committer written in Lean, that those are
the weights put in and that the commitment cannot be opened to others — is in
[guide/what-is-proved.md](../guide/what-is-proved.md) and
[lean/THEOREM.md, "What the commitment gives, as proved"](../lean/THEOREM.md). The written proof of
soundness, which is not in this repository either, does not cover three things that the programs do:
what a proof's first hash takes in, the labels of two of a proof's chains, and how the commitment ID
is formed.
[docs/limits.md](../docs/limits.md#what-the-theorem-assumes) says where counted proof of work comes
in. [docs/output.md](docs/output.md#the-closing-lines) shows which lines of the output say that the
table "has a record".

## Status and limits

- **It accepts a wider range of parameters than our own tools produce, and that wider range is
  untested.** No honest files exist for it, so no test with an altered file or a malicious prover
  has been run there.
- **The range check's own code is only partly tested.** A unit test
  (`verifier/tests/tail_twin_test.cpp`, which no row of `SUITE.md` runs) checks it claim by claim, at
  toy lengths. It does not run the step that hands the sub-proof's equation to the proof's chain, the
  segment's closing or the proximity tests (row T2.3 of `docs/limits-all-rows.tsv`).
- **It does not cap sizes, so a file that is far too large can make the call exit with code 5 or get
  the process killed by the system.** It reads `key.bin`, the leaf file and the proof file in full
  before it checks their lengths ([more](../docs/limits.md#hostile-files)).
- **Core dumps should be off, but we have not tested that.** On Linux, the program marks itself
  non-dumpable before it reads any file. This matters because `cp_commit` and a provider's check run this program on files that are not
  public yet, and a response that `cp_verify` rejects can contain unmasked hidden values.
- **Memory and floating point.** The three stages are in memory at the same time, so leave headroom
  when you set a memory limit ([what a full-size check needs](../guide/performance.md)). The four
  public tables are computed with the platform's floating-point library, so on a platform whose
  library rounds differently, an honest proof ends in `NO: table-digest-<name>`.

<a id="tests"></a>

## What is in this directory

- `src/` holds the source code, `build.sh` is the build script, and `tests/` holds the tests.
- `docs/` holds the reference for [what `cp_verify` prints](docs/output.md) and, for developers,
  [the parameter file](docs/parameter-file.md), [the output of planning calls](docs/plan-data.md)
  and [a map of the source code](docs/source-map.md).

† Not on the released programs. What it ran on, and how far that speaks for them: [what was run on these programs, and what was not](../docs/limits.md#what-was-run-on-these-programs).
