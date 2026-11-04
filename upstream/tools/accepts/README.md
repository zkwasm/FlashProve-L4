# tools/accepts: run Lean's checks on a stored proof with one command

(The terms for what Lean proves: [lean/TERMS.md](../../lean/TERMS.md).)

```
tools/accepts/accepts.sh small [honest|controls|all]     # default: all
```

The command takes the files of examples/accepted-proof as the C++ prover wrote them, converts them
with [tools/parser](../parser/README.md), and runs the Lean files that `sheets.tsv` names on the
result: first unchanged (`honest`), then with one deliberate change at a time (`controls`) to a file
or to the claim (commitment ID, request, reply, public files). The unchanged files should get a yes,
and each control should end as its Lean file expects: in the files of this repository, refused. `full`
(examples/full-size) is not in `sheets.tsv`, so the command ends without testing anything,
exit code 2.

## What a yes does not show

A yes is the result of a test on stored files. It is not a verification and not a theorem.

- It shows that the checks run and accept what an honest prover wrote. It does not show that they
  ask enough to stop a dishonest one. No test of this kind can: that takes a proof of soundness. Lean
  has one for its own verifier, `verify` (the next item); no stand-alone verifier program is proved sound.
- It is not `verify`, the Lean verifier of the soundness theorem `sound_of_verify_at` ([lean/THEOREM.md](../../lean/THEOREM.md) says what is proved of it, and on which conditions).
  What runs here is single parts' checks on the parsed files, part of them written to follow the
  C++ verifier. To run the Lean verifier's executable form on a proof: the command `lean-verify`,
  under `tools/lean-verify/`. lean/THEOREM.md also tells what has been run with that form and what such
  a run trusts ("What has been run"). It also tells that altered proof files were accepted: parsed proof
  files with bits changed in Merkle-path entries that the verifier's walk does not take (by reading
  and by runs, not proved); on some such files the executable form, given the parsed file, and `cp_verify`,
  given the raw proof, part; in those runs the parser refused those proofs (not the parser as released) (lean/THEOREM.md, "What has been run", says over which files). No claim that is false by the run's own policy was accepted in any run.
- A refused control shows that this one change is caught, and nothing about other changes.
- The examples have synthetic weights, and the checks judge under exact match alone.

The whole list is [What a yes does not mean](../../lean/Ask/README.md#what-a-yes-does-not-mean) in
lean/Ask/README.md. Read it before you quote a result.

## Before you run it

- Run it as an ordinary user, not with sudo. As root, under sudo or not, its check that no other account can swap the work directory also trusts directories of the account that `SUDO_UID` names. Below such a directory that account (under sudo, one that holds sudo already; otherwise any account) could swap the work directory during a run, and root's writes, and its load of a library from there, would follow. As root, Lake's new files below `lean/.lake` are root's too.
- Build `lean/` first, with its packages in place: steps 3 and 4 of "The commands" in
  [lean/CHECKING.md](../../lean/CHECKING.md). It also needs `lake` on the PATH, `g++` 13 or later,
  `python3`, `bash`, `awk`, and GNU coreutils, findutils, grep and sed.
- It calls no network tool, and it stops if `lean/.lake/packages` is missing. Lake or elan may still
  fetch if a package or the Lean version is not the one that `lean/` names.
- `small` took 31 minutes from start to `END:`, with one core busy on average, and about 10 GiB of
  memory at its peak, counting mapped files. Measured once, on one machine, not with this script and `lean/` as this repository has them,
  with `lean/` built beforehand. Neither figure is an upper limit. With `ACCEPTS_JOBS` above 1, up to that many calls of Lean run side by side, each with some 10 GiB. The script sets no upper bound on `ACCEPTS_JOBS`; `sheets.tsv` has four calls, so no more than four run at once.
- This repository's script and sheet, by their checksums, ran at small size alone over this
  repository's Lean files: all 4 calls of Lean ended as expected, and all 17 controls were refused.
  That run took about 33 minutes, on a machine that was busy with other work; its record gives 9.1
  GiB of memory at most, and does not say whether mapped files are counted.
- It writes below its work directory, and Lake below `lean/.lake`. It deletes nothing: delete the
  work directory when you are done.

## What it prints

The first line names the work directory. Each call's `.out` and `.err` are there, and `RECORD.txt`,
one fact per line for a program to read. Each call of Lean gets a line under its name: Lean's exit
code, the seconds, the answer on the unchanged files, and how many controls were seen, ended as
expected and were refused. If the run prints a line `description:`, that comes first. The last line
begins `END:` and gives the exit code in words: 0, all ended as expected; 1, a wrong answer; 2,
something was not tested; 3, a run did not end. A call of Lean can outlive the script. It then runs until it ends or reaches its time limit, 4 hours by default, out of Ctrl-C's reach, and can still change its files in the work directory. Every running call does so if a signal that the script does not trap, such as QUIT or KILL, ends the script (then without `END:`). One call can do so if Ctrl-C, TERM or HUP falls in the first milliseconds after its start, in a rare case with no time limit at all.

The comment at the head of the script has the options, the table's fields and each exit code's
cases.
