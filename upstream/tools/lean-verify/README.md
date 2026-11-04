# tools/lean-verify: run the Lean verifier on a real proof

`lean_verify.sh` runs `verify`, the Lean verifier that the soundness theorem `sound_of_verify_at` is
about ([lean/THEOREM.md](../../lean/THEOREM.md)), on a full-size proof as the C++ prover wrote it,
and prints ACCEPT or REJECT. (Terms: [docs/glossary.md](../../docs/glossary.md).)

**Why it exists.** `cp_verify`, the C++ program, is the verifier that users run: fast, tested, not
formally verified. This command is the slow cross-check. Two Lean theorems stand behind it: `verify`
is sound at exact match, and the function that this command evaluates gives the value of `verify`.
That Lean, the C compiler and this script carry that function out faithfully is trusted, not proved:
[What an accept means](#what-an-accept-means).

**What it asks of you.** It runs the verifier inside Lean; it is not a program of its own. Lean and
its packages must be installed and about a third of the Lean library built, once: half an hour on a
large machine. So that a run takes minutes and not hours, the first call compiles most of the Lean
code that it needs into a shared library (the helper library). A call takes 19 to 39 minutes (the
longest was a first call) and 9 to 16 GB of memory; a first call, while it compiles, took up to 30
GB on 96 cores ([What it costs](#what-it-costs)). It takes full-size files only, on Linux x86-64.

## Which files are what

`lean_verify.sh` is the command. `LeanVerify.lean` hands its arguments to `PV.Program.run` and turns
the answer into Lean's exit status. `read_claim.py` is a second reader of the claim. Before Lean
runs, `tools/parser/parse_files.sh` (the parser) cuts the prover's key, leaf and proof files into
the typed files that Lean reads. The rest belongs to [the helper library's
list](#the-helper-librarys-list) and to two tests. What the command evaluates lies under `lean/` and
is built and audited with the library: `lean/Johnson/PVR/Program.lean`, `ProgramTotal.lean` and
`ProgramGuard.lean`. This folder lies outside `lean/`: the library's own check does not reach it.

## Before you run it

- **No sudo.** Run everything as one ordinary user. Under sudo Lake would leave root's files in
  `lean/.lake`, which then break your own builds; both scripts stop if `lean/.lake` belongs to
  another account.
- **A fresh unpack, no links.** The script itself removes, moves and overwrites nothing; it writes
  below a fresh work directory from `mktemp -d`. But Lake, which it starts, writes below
  `lean/.lake` and can replace or delete files there ([lean/CHECKING.md](../../lean/CHECKING.md),
  "Before you start"). The script stops if `lean/`, `lean/.lake`, its `build` or `packages`, or a
  package's folder is a symbolic link. Do not link this folder, `tools/` or `lean/` into another
  tree: the scripts work in the tree that really holds them, and print it (`tree: ..`).
- **Disk.** About 26 GB below `lean/.lake`. Every call leaves a work directory of about 300 MB in
  `TMPDIR` (default `/tmp`, which on many systems is memory): delete them when you are done.
- **Memory and time.** See step 1 and [What it costs](#what-it-costs). Set no tight `ulimit -v`:
  Lean's thread reserves a stack of 4 GB. The script sets no bound on memory or time; with
  `timeout -k 10 2h` in front, the codes 124 and 137 are no accept, whatever line stands beside
  them.
- **One call at a time,** of either script: two would share `lean/.lake`.
- **A stray process.** Look for one after a signal. At TERM, HUP or Ctrl-C the script stops what it
  has started and ends with `NO VERDICT`; it cannot pass on a KILL: what it has started then runs
  on. About one signal in a hundred, sent within milliseconds of a start, missed a process, which
  then ran to its own end: up to half an hour and 16 GB. Look with `pgrep -af LeanVerify`,
  `pgrep -af parse_files` and `pgrep -af 'lake build' | cut -c1-200`, and end processes by number,
  never by a pattern.
- **What it needs:** `lake` on the PATH, bash, GNU coreutils, diffutils, grep and sed, `prlimit`,
  `python3`, and `g++` 13 or later, with which the parser compiles a program from this repository's
  C++ and runs it. The scripts call no network tool; Lake or elan may fetch if a package or the Lean
  version is not the one that `lean/` names.
- **The first call writes the helper library's objects below `lean/.lake`,** most of them into the
  packages' folders. Run the library's full check in a tree in which this command has not been
  called.
- **The tests.** `lean_verify_test.sh` tests both scripts' refusals and the command's reading of
  Lean's status and lines, with stand-ins for Lean and the parser: a few minutes, and a pass ends
  `all as expected`. It says nothing about the real `lake`, `lean` and parser. What it does in its
  work directory: its opening comment. `python3 -B tools/lean-verify/read_claim_test.py` compares
  `read_claim.py` with what `cp_verify` printed for the four stored examples.

## What to type

1. Once: get Lean and the packages ([lean/CHECKING.md](../../lean/CHECKING.md), "The commands",
   steps 1 to 3). Then, in `lean/`, check that the packages include their `.c` files (if not: [If a
   .c file is missing](#if-a-c-file-is-missing)), and build what the command asks for:

   ```sh
   ls .lake/packages/mathlib/.lake/build/ir/Mathlib/Logic/Basic.c
   LEAN_ABORT_ON_PANIC=1 lake build Johnson.PVR.ProgramGuard
   ```

   **This build can strain a machine.** Lake starts one process per core; on 96 cores they peaked at
   20 GB of their own memory. If memory runs short, put `LEAN_NUM_THREADS=8` in front. A whole build
   of the library also works.

2. The stored full-size example (synthetic weights), from the repository's top directory. Expect
   exit code 0 and `VERDICT: ACCEPT (exact match)`. Type steps 2 to 4 in one shell.

   ```sh
   E=examples/full-size
   tools/lean-verify/lean_verify.sh $E/config $E/ivk $E/example.req $E/example.y $E/example.proof \
     $E/ID.txt
   ```

3. A reject that comes early: claim another commitment ID, here 64 zeros. Expect exit code 1,
   `VERDICT: REJECT`, and `idOk no` in the line of steps. (`$D` is a small directory that stays:
   delete it when you are done.)

   ```sh
   D=$(mktemp -d) && printf '%064d\n' 0 > "$D/other-id.txt"
   tools/lean-verify/lean_verify.sh $E/config $E/ivk $E/example.req $E/example.y $E/example.proof \
     "$D/other-id.txt"
   ```

4. A reject that comes late: claim another last token of the output, here token 1. Expect exit code
   1 and `VERDICT: REJECT`, with `armB no` alone, after about as long as step 2.

   ```sh
   printf '\001\000\000\000' > "${D:?}/other.y"
   tools/lean-verify/lean_verify.sh $E/config $E/ivk $E/example.req "${D:?}/other.y" \
     $E/example.proof $E/ID.txt
   ```

**A second stored set,** `examples/full-size-clipped-gap/`, has the same kinds of file under another
acceptance policy (`C` 25856, `theta` 512, `cap` 1), which tolerates tokens other than the model's
first choice. Set `E=examples/full-size-clipped-gap` and type step 2's command. Expect exit code 0
and the second of the two ACCEPT lines below.

**These steps check no theorem.** A build audits neither `sorry` nor axioms, and step 1 builds only
what the command asks for. What checks the theorems that give an ACCEPT its meaning:
[lean/CHECKING.md](../../lean/CHECKING.md).

In general:

```
tools/lean-verify/lean_verify.sh [--no-native] CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE \
  PROOF_FILE ID_FILE
```

The first five are `cp_verify`'s arguments of the same names ([AGENTS.md](../../AGENTS.md)).
`CONFIG_DIR` holds the three public files: the parameter file `policy.txt`, the layout file
`layout.txt` and the scale file `KL.bin`. `ID_FILE` holds the commitment ID: 64 hex digits in lower
case, then at most a newline. On the small examples the command gives no verdict: it wants a request
file of 4,104 bytes. `--no-native` runs everything in Lean's interpreter, without the helper
library.

## Verdict and exit code

Standard output holds one line; everything else goes to the error stream. **The verdict is the exit
code together with that one line, and it fails closed.** No other line is a verdict: Lean's own
lines, shown on the error stream behind `lean: `, can hold `VERDICT: ACCEPT` in a call that then
ends with `NO VERDICT`, so do not search a log of both streams for ACCEPT. Compare the whole line: a
search for `VERDICT: ACCEPT` or for `exact match` finds both accepts. After any other code, or an
end by a signal, treat the reply as unproved.

- Code 0: accepted, under the policy that the line names. Where the parameter file's lines `C`,
  `theta` and `cap` read 256, 256 and 0 (and `sh` and `sT` 8 and 4), the first line; else the
  second, with the file's three numbers.

  ```
  VERDICT: ACCEPT (exact match)
  VERDICT: ACCEPT (under the files' policy C=25856 theta=512 cap=1; NOT exact match: see "What an accept means")
  ```
- Code 1 and `VERDICT: REJECT`: rejected. An honest proof can be rejected, for instance if a file is
  damaged.
- Code 2 and `NO VERDICT`: wrong arguments, a missing or malformed file, a proof file that the
  parser refuses, a library that is not built, a failed tool, a signal, or a run that did not end as
  it should. The error stream says which. It says nothing about the proof.

## What the printed lines mean

The error stream shows, among others:

```
lean: claim: commitment ID: 7a285919b43961861d9839530c26725accf26aa6d74e3c2afa31b9dc6bcd18fe
lean: claim: checked positions: from r = 23 up to, not including, e = 1024
lean: claim: prompt, 24 tokens: [4099, 129262, ..]
lean: claim: output, 1001 tokens: [12664, 32096, ..]
lean: public files: the acceptance policy, by the parameter file's own lines: [C 256, theta 256, cap 0, s 15, sh 8, sT 4, F 24]
lean: steps: buildOk yes; encOk yes; idOk yes; laidT yes; laidL yes; laidB yes; num yes; seed0 yes; seedJ yes; armT yes; armL yes; armB yes
```

**Read the claim.** The four `claim:` lines are what is judged: the commitment ID, the prompt (the
request's tokens 0 to r) and the output (its tokens r + 1 to e, where the token in `TOKEN_FILE`
comes after the request's last; `docs/format/FORMAT.md`, 12.2). Compare them with what you sent,
what you received and the ID that you held beforehand.

**The twelve steps.** Accept: all twelve say yes. A later step says no where an earlier one that it
needs has failed. Only the verdict is tied to `verify` by a theorem; what a single step tests is
told here from the Lean text (`lean/Johnson/Top/Fire/Code/VerifyVFastStepsD.lean`).

| step | it says yes if |
|---|---|
| `buildOk` | an instance is built from prompt, output and the request's further tokens |
| `encOk` | the public key can be encoded |
| `idOk` | the hash of that encoding is the commitment ID |
| `laidT`, `laidL`, `laidB` | the three typed proofs are built from the parsed key file, leaf file and proof file |
| `num` | the leaf table's number that comes with the proof is the number in the plain part of the proof file's first round (`docs/format/FORMAT.md`, R14.2) |
| `seed0`, `seedJ` | the seeds that the parser wrote next to the parsed key file and leaf file are the ones that the verifier works out itself |
| `armT`, `armL`, `armB` | the key file, the leaf file and the proof file verify |

With public files outside `verify`'s scope the verdict is REJECT, and Lean prints
`the run stopped before the steps:` in place of the steps.

## What an accept means

### Proved in Lean

- `verify` is sound at exact match, the policy, set in the parameter file, under which every output
  token must be the model's first choice ([lean/THEOREM.md](../../lean/THEOREM.md)). With the hash
  function modelled as random and the public files fixed independently of it, a false claim is
  accepted with probability at most about q/2^124 + 2^-101, where q is the sum of a bound on the
  committer's hash queries and a bound on the prover's. That is for one proof; for N proofs handed
  in at once under one commitment, N times that, with q counted over the whole list; for proofs
  checked one after another no theorem states a bound.
- The function that the command evaluates gives `verify`'s verdict. As written, `verify` cannot be
  executed; what runs is an executable form. `PV.Program.verdictOf` is the pure function from the
  files' bytes to the verdict; `run` calls its three parts in turn. Wherever the claim can be read,
  `verdictOf` gives the value of `verify` at exactly that claim, at the format's hash function;
  where it cannot, it gives no verdict (`verdictOf_map`, `verdictOf_eq`, `verdictOf_is_verify`, in
  `lean/Johnson/PVR/`). These are theorems on Lean terms, not on the code that runs.
- The parser need not be trusted for soundness: the theorems hold for every value of the parsed
  files. A wrong parsed file can cost an honest proof its accept, and otherwise gives a cheating
  prover nothing that it does not already have.

### Tested

The calls described here were made over Lean files and sources of the parser that differ from this
repository's. The reference run over this repository's Lean files is told in
[lean/CHECK-RESULTS.md](../../lean/CHECK-RESULTS.md).

At full size, on synthetic weights, over a build from scratch of Lean files that differ from this
repository's, with the parser built from sources that differ from this repository's and a
`cp_verify` that printed, as `build=`, the source digest that `verifier/build.sh` computed for them,
the command accepted all 47 honest proofs given to it: 42 by one prover built from [other
sources](../../docs/glossary.md#other-sources) (not the released one: [which kinds of result these
are](../../docs/limits.md#on-other-programs); 41 at exact match, 1 under the second stored set's
policy) and 5 others (the two stored examples, the second under its own policy, and 3 honest proofs
from the test programs). It accepted none of 57 lying proofs, 24 of a false reply and 33 of the true
reply with a lie inside the proof: it rejected 54, and at 3 the parser refused the file before Lean
ran (`NO VERDICT`). It rejected 4 honest proofs paired with another reply, and the stored example
paired with another commitment ID. `cp_verify` agreed wherever both gave a verdict. The conditions:
little variety (7 commitments and 39 requests; the 41 at 3 commitments, set up on one set of
weights; the lying proofs at 1 commitment and 3 requests); the 41 at exact match are a part of 363
that were chosen, by a rule fixed beforehand, out of 723 made, as reported; in the run's own record
`cp_verify` had accepted each honest proof beforehand; the command itself had judged every one of
these files before; every lying proof was at exact match; Lean's C compiler was GCC 15.3.0, not the
clang that comes with Lean; and the packages were compiled beforehand. By a comparison of checksum
lists, in packages fetched once by `lean/CHECKING.md`'s step 3, through a local mirror and not from
the public servers, every source file, every file that Lean loads when a module is imported and
every `.c` file has the same SHA-256 checksum as in that compiled copy; that the mirror serves the
files that the public servers serve is not shown. A proof file does not show which program made it:
that 42 of the honest proofs came from that prover, and the lying ones from test programs, is how
the runs were set up, not something a verifier can check. Of all this, steps 2 to 4 and the second
stored set are what you can repeat from this repository.

### What you must check yourself

- *Exact match* is a premise on the parameter file, not a test that `verify` makes: files with
  another policy are in scope, and some put no condition on the counted tokens. The accept's line
  takes its numbers from the `public files:` line, which `read_claim.py` and Lean must print alike,
  to the byte. That line shows the parameter file's lines as they stand there, not what the
  verifier's own reader made of them; `verify`'s scope test (`InScope`, `lean/Ladder.lean`) ties the
  two as the definitions read, but no lemma says so. So read `policy.txt` yourself
  (`docs/format/FORMAT.md`, 12.5). The error stream also says whether your three public files are,
  byte for byte, those of `examples/full-size/config`, which ask for exact match.
- *An accept under another policy says less.* The main theorem does not speak there. A second
  theorem does: the accepted reply meets the files' policy at the weights that the protocol's
  extractor returns, except with probability at most its stated bound. Under a policy like the
  second stored set's that does not show that any one token of the output is the model's first
  choice.
- *Which model.* "The model" is the extracted weights, run at the public scale and parameter files,
  which you must hold and read. No theorem says that these weights are the ones that the release's
  committer put in, and scale files in scope can switch off any of the model's blocks, up to all of
  them. More on this bullet and the one before: [lean/THEOREM.md](../../lean/THEOREM.md).

### What is trusted

None of this is proved.

- *No theorem covers how the claim is read from your files.* Three short functions
  (`PV.Program.claimOf`, `PV.Front.idOf`, `PV.Front.userOf`) turn your three files into commitment
  ID, prompt and output. A fault there would be an accept about another claim than yours. The
  guards: Lean prints the claim that it judges, in full; `read_claim.py`, written from the format
  page, must read the same claim off the same bytes, to the byte, or there is no verdict; and on the
  four stored examples that reader agrees with what `cp_verify` printed.
- *What runs is compiled code.* The 5,056 modules of `native-modules.txt` run as the C compiler made
  them, Lean's own modules as the program `lean` holds them, and Lean interprets the rest of this
  library. All are meant to compute the definitions that the kernel checked, except the replaced
  implementations below; nothing here proves it. So you trust Lean's kernel, compiler, interpreter,
  loader and runtime, Lake, the C compiler and linker, and the operating system.
- *Replaced implementations.* At each build `ProgramGuard.lean` lists, by the kernel's terms and not
  the compiler's output, the constants that `PV.Program.run` uses whose compiled code is not made
  from the definition that the kernel checked. The build fails if one is this library's. Some 180 of
  Lean's core and Std, of Mathlib and of the packages are listed and not judged, and the compiler's
  replacement rules are followed, not checked.
- *Where the compiled files come from.* A run reads compiled files below `lean/.lake`, not the
  sources. For this library the script asks Lake at every call whether they fit the sources (if not:
  `NO VERDICT`, and back to step 1), links the helper library afresh, and stops at an object that is
  not newer than what it was compiled from: Lake's word and file times, no proof. The packages'
  compiled files, and the `.c` files from which 3,386 of the objects are made, are used as fetched.
- *Also under no theorem:* the part of `Program.lean` that reads files and prints,
  `LeanVerify.lean`, this script and `read_claim.py`; and your shell's environment (`PATH` picks
  `lake` and every tool; `ELAN_TOOLCHAIN`, `LD_PRELOAD` and the like reach them).
- *The hash function.* The theorem draws it at random, and assumes that a second, fixed hash
  function, built on BLAKE2b's compression function, does not interact with it; a run puts one fixed
  function, BLAKE2b, in the random one's place, as is usual for such proofs.

### Not covered

An accept says nothing of `cp_verify` or of the prover. That `verify` accepts honest proofs is
tested, not proved; a gap there could only cost a visible rejection.

## What it costs

Measured on Linux x86-64 servers with 64 to 128 cores, as `nproc` counts them. None of these figures
is an upper limit, and none was measured on the released files: the measurements were made over Lean
files and sources of the parser that differ from this repository's. They were not all measured on
the same text of the command and of the library: for one, the builds on 48 and 32 threads lacked
thirteen small modules.

Lean's C compiler was GCC 15.3.0, not the clang that comes with Lean: with that one the compile of
the helper library is untried, and its time, size and memory may differ.

| what | time | memory and disk |
|---|---|---|
| step 1, the build, once | 25 to 26 minutes on 96 cores, 33 on 48 threads, 49 on 32; 12.8 hours of processor time on 48 threads, 11.5 on 32 | on 96 cores 20 GB of the processes' own memory, 196 GB if the compiled files that each process maps are counted for each (counted so, 142 GB on 32 threads); on 48 and 32 threads the largest process 10 GB, and about 15 GB of disk, beside about 11 GB of packages (fetched ones may take nearer 7 GB) |
| the helper library, at the first call | 5 to 9 minutes, by the C compiler alone | 8.3 GB on 16 threads, for the C compiler's processes together; on 96 cores 23 to 24 GB of own memory, for all the call's processes (29 to 30 GB if the files that each process maps are counted for each); 0.4 GB of disk |
| a call to an accept, or to step 4's late reject | 19 to 39 minutes, on about one core; the longest was a first call, which also compiled the helper library; calls that found it compiled were shorter | a peak of 9 to 15 GB for the largest process (of the accepts); 15 to 16 GB summed over the call's processes (both count the compiled files that a process maps) |
| step 3's early reject | 4.5 to 11 minutes | 11 GB, summed and counted likewise (one run) |
| `check_list.sh` | under a minute | 9 GB for the largest process, counted likewise |
| `--no-native` | 9 h 18 min (one run, its commands typed by hand; the script's option itself has run only with stand-ins for Lean) | 9.5 GB |

How many measurements stand behind each figure in the table:

- One: every figure of the builds on 48 and on 32 threads (one of each build); the helper library's
  8.3 GB and 0.4 GB; the early reject's 4.5 minutes and 11 GB; `--no-native`.
- Two: every figure on 96 cores (two runs); the 15 to 16 GB (two calls).
- The packages' 11 GB: one copy; the 7 GB is a forecast, not a measurement.
- The helper library's 5 to 9 minutes: five compiles.
- The early reject's 11 minutes: the longest of six calls.
- `check_list.sh`: seven runs.
- A call to an accept or a late reject: the times in 21 calls to an accept and six to a late reject;
  the 9 to 15 GB in the 21; "about one core" in six of the 21, made side by side, each in its own
  copy of the tree, and in calls on one other machine. Several hundred more such calls were made
  many at a time for tests, likewise each in its own copy (those under "Tested" among them); no
  figure here is theirs.

Whether the records behind the 142 GB, the 8.3 GB and the 9.5 GB counted in powers of 1,000 or of
1,024 bytes is not known: they may be up to 7 percent too low.

How much memory a call takes varies, and why is not known: on one shared, busy machine the same
command line on the same proof, with the same helper library, peaked at 9.9 GB and at 13.5 GB (the
largest process, counted as in the table), and in two more runs, one near the higher and the other
somewhat below the lower of the two. `lean_verify.sh` was not the same in all four runs, and
`read_claim.py` and four Lean files were not the same in the first two. But on that machine steps 2
and 4, run one right after the other with no file changed between the two, differed by about as
much; what the two were given differs in one file of 4 bytes alone. At every call the helper library
is linked anew, in 9 to 38 s (once, on that same machine, 125 s; why is not known).

## If an object is refused

`not newer than its .c and its .olean` means that a compiled object may come from an older version
of its source, so the script stops before the verifier runs. The usual causes: a built tree was
copied, or file times are kept to whole seconds only (by an archive or a file system), so that an
object and its source, written within the same second, tie. Take a fresh unpack where finer times
are kept, fetch and build there, and call the script there; or use `--no-native`.

## If a .c file is missing

Lake makes each native object from a `.c` file that Lean wrote when it compiled the module; the
packages as `lean/CHECKING.md`'s step 3 fetched them once, through a local mirror, include these
files; a fetch from the public servers is untried. If yours lack them, Lake would have to compile
the packages' Lean sources, Mathlib's among them: a long build, which this script does not start.
Use `--no-native`, or start that build yourself, in `lean/`, with `lake build` and the targets
`+<module>:c.o.export` for the modules of `native-modules.txt`; step 1's caution on memory holds for
it too. Neither way has been run on packages that lack these files.

## The helper library's list

`native-modules.txt` names the modules whose compiled code goes into the helper library: 1,670 of
this library, the others Mathlib's and the other packages'. `LeanVerifyWalk.lean` made it; its
opening comment says how, and why seven modules and all that import them are left out and
interpreted. To make the list again and compare, after step 1, type
`tools/lean-verify/check_list.sh`. It prints `native-modules.txt: the same as made anew` (exit code
0), `native-modules.txt: NOT the same as made anew` (1) or `native-modules.txt: not checked` (2). It
judges no proof and builds nothing. A wrong list can cost time, or cost a run its verdict.
