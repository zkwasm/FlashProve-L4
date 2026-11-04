# For coding agents

A provider commits to a model's weights with `cp_commit`, then proves with `cp_prove` that each
reply passes an acceptance policy. Built from `prover/`, both run on one GPU. Clients check proofs
on a CPU with `cp_verify`, the C++ verifier in `verifier/`. [README.md](README.md), "For agents",
has every route with what it needs and how long it takes.

## Not as root

Run everything here from an ordinary account, not under `sudo` or as root. The programs, scripts and
tests need no root rights, and [as root a fault or a mistake reaches
further](docs/limits.md#running-the-programs-safely). Only setting up the machine needs root.

As root with no other account, make one as below. [Unpack or clone the repository as that
account](docs/quickstart.md#install) (put an archive in `/tmp`, not `/root`): in a copy that root
owns it cannot build. Without a terminal, `su - tester` leaves you root: use the second form. A line
with single quotes of its own, like the `nohup` line below, does not fit between its quotes: send it
as [a here-document](docs/quickstart.md#install).

```
useradd -m -s /bin/bash tester && su - tester
```

```text
su - tester -c 'cd DIRECTORY && COMMAND'
```

`DIRECTORY` is the repository's root, `COMMAND` the line to run.

## First commands

These need Linux, bash and g++ 13 or later. [Installing them](docs/quickstart.md#install) needs
root, so first ask whoever runs you. Work from the repository's root. The verifier's build, the
check of the stored proof and route 3's small example are rows of the [test suite](SUITE.md)
(`build.verifier`, `verifier.call_accounting.examples`, `gpu.quick_start`,
`gpu.run_target_example`), which pass on programs that the suite builds from these sources; no row
runs the one-call script `examples/quickstart-gpu.sh`, and none reads the lines that the build
prints; how long this page says a command takes or stays quiet was seen not on the released programs
(row E35 of [docs/limits-all-rows.tsv](docs/limits-all-rows.tsv)).

Build `cp_verify`. It stays quiet while it compiles. Success: a line beginning
`THE VERIFIER IS BUILT:`, exit code 0.

```
verifier/build.sh
```

With a time limit per call, build in the background. Send these two lines as separate calls:

```
echo "build started" >> verifier-build.log; nohup bash -c 'verifier/build.sh; echo "exit code $?"' >> verifier-build.log 2>&1 &
tail -n 2 verifier-build.log
```

Repeat the `tail` line until the last line begins `exit code`. Start no second build meanwhile. If
no such line appears within ten minutes, check that no build is running, then start again. A killed
build can leave a directory whose name begins `bin/cp_verify.building.`: run nothing from it. A line
that begins `build.sh: note:` is a warning: quote it in your report. It says what another account
could do there and, if the directory is yours, how to fix its permissions (`chmod go-w` on the
directory that it names): do that only if whoever runs you says so, and change no other permission.

The GPU script (README.md, route 3) goes into the background in the same way; it can go quiet for
more than a quarter of an hour. Nothing that it prints tells a dead run from a quiet one. The ten
minutes above are for the verifier's build. If no `exit code` line has come long after the time that
README.md gives for route 3, look whether anything of it still runs: if so, wait; if not, the run
was probably killed: start the script again. If each of your calls is a new shell, send the lines
that bring CUDA 13.2 Update 2 onto `PATH` in the same call, before this line:

```
echo "run started" >> quickstart-gpu.out; nohup bash -c 'examples/quickstart-gpu.sh; echo "exit code $?"' >> quickstart-gpu.out 2>&1 &
tail -n 2 quickstart-gpu.out
```

Check the stored proof. Success: last line `YES`, exit code 0. This shows only that `cp_verify` runs
and these files, for a small model with synthetic weights, fit together.

```
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)"
```

For other proofs, first read [when to accept](guide/verify-a-proof.md#when-to-accept) and [what each
argument means](verifier/README.md#the-call). General form:

```text
bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID
```

[`integrations/check_proof.py`](integrations/README.md) wraps that call: it compares the proof's
files with the client's own values and runs `cp_verify`. Every printed line and exit code:
[verifier/docs/output.md](verifier/docs/output.md). The file formats:
[docs/format/FORMAT.md](docs/format/FORMAT.md).

## Rules

- Never get past an error with `sudo`, `chmod`, `chown`, `rm` (the one exception: `chmod go-w` on
  the directory that the build's note names, if whoever runs you says so), or by adding an option,
  unless [the table of exit codes](docs/verifying.md#exit-codes) tells you to add it.
- Never edit a source file to change a result.
- Do not build `prover/` unless the task needs it. [The tests](SUITE.md#run-the-tests) overwrite
  `bin/cp_verify` and `bin/weight_sums`: run them in another checkout.
- Never take a commitment ID from the files that came with a proof, except in the stored examples.
  [Get it from a party you trust](guide/verify-a-proof.md). For a commitment that you made yourself,
  use the ID that `cp_commit` printed.
- Run `cp_commit`, `cp_prove` and the tools in `scales/` on synthetic or public weights only. On
  weights that must stay secret their logs, counts and exit codes stay private, from a coding agent
  too ([Keep these private](docs/operators-duties.md#every-day)): stop, and tell whoever runs you.
- If you lack a value or right, stop and ask whoever runs you.

## Reporting

- Give the exit code and the entire last line together, and the commitment ID that you passed.
- If the call accepts, also give `r=` and `e=` from the line `covered:`, and the lines that begin
  `commitment:`, `proof of work:` and `result-v1`. [Their bits are no security
  level](guide/known-limits.md).
- Say what you ran, and what was printed where it differs from the page.
- `YES` or `WITHIN LIMITS` on a stored example says nothing about a provider's model. A test pass
  shows only expected answers on synthetic inputs.
- A `YES` is `cp_verify`'s. Soundness is proved, in Lean, for the verifier written in Lean, for
  exact-match claims; the C++ verifier and the CUDA prover are written to follow the same protocol
  and are tested against it, not themselves proved ([what is proved](guide/what-is-proved.md)).
- A run does not show that the weights are hidden: for what is proved and tested of that, point to
  [weight hiding](guide/weight-hiding.md).

## Read on

[Getting started](guide/getting-started.md), [what is proved](guide/what-is-proved.md), [weight
hiding](guide/weight-hiding.md), [known limits](guide/known-limits.md), [how it
works](guide/how-it-works.md), [terms](guide/terms.md), [the repository's
layout](docs/architecture.md#repository-layout), [the limits at length](docs/limits.md).
