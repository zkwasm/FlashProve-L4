# Getting started

You can try FlashProve by four routes, from a CPU alone up to the full-size model. Only routes 3 and
4 make a proof about a model's tokens. Start with route 1.

No route runs a malicious prover, so a `YES` from these routes [shows only that honest files were
accepted](../docs/quickstart.md#what-these-routes-show).

## The four routes

| Route | You need | Time† | Host memory |
|---|---|---|---|
| 1. Check a stored proof, CPU | Linux, bash, g++ 13 or later | build about one to two minutes, check under two seconds | 1.6 GB (the compiler took about 1.3 GB, on [other sources](../docs/glossary.md#other-sources)) |
| 2. Make toy proofs, CPU | The same, and make, python3 | about a minute (one run of the script on this repository's own files, so † does not apply to it: 72 seconds, on a busy machine) | under 1.5 GB (in that run about 0.97 GB in its largest process, the compiler) |
| 3. A small model, one H100 | See below | build about 29 minutes at 24 CPUs, run about 23 to 30 seconds | 10 GB available |
| 3 at full size, synthetic weights | The same | about 42 minutes | 280 GB available |
| 4. Llama 4 Maverick, public weights | The same, and curl, Python 3.13, 245 GB of disk, and a calibration set that you make first | Hours | 330 GB available |

The test suite runs the verifier's build and routes 2 and 3 (of route 3 the small model, by the two
blocks of commands in section 3 of docs/quickstart.md that build and run the example, and not by the
one-call script, which no row runs, and not the full size) on programs that it builds from these
sources (rows `build.verifier`, `new.first_look`, `gpu.quick_start` and `gpu.run_target_example` of
[SUITE.md](../SUITE.md)). The times for routes 1 and 3 and the memory of route 1 come from three
other runs†, each on one H100 with the shell bound to 24 CPUs. "At 24 CPUs" describes the machine,
not a need.

## Before you start

Run every command in bash, from the repository's root, one line at a time. A minimal Ubuntu 24.04,
as in a container, lacks g++, make, python3 and unzip. This command adds them, as root or with
`sudo` before each `apt-get`:

```
apt-get update && apt-get install -y --no-install-recommends g++ make python3 unzip
```

It may ask two questions on the time zone, and if root is your only account, make another: [the
detailed docs](../docs/quickstart.md#install) cover both.

Run everything here from an ordinary account, not under `sudo` or as root. The programs, scripts and
tests need no root rights, and [as root a fault or a mistake reaches
further](../docs/limits.md#running-the-programs-safely). Only setting up the machine needs root.

## Route 1: check a stored proof

This builds `cp_verify`, the C++ verifier that clients run:

```
verifier/build.sh
```

It ends with `THE VERIFIER IS BUILT:` and the program's path, exit code 0. A long first line that
begins `build.sh: note:` is a warning, not an error.
A second build's line that begins `build.sh: overwrote` is not an error either.

This checks a stored proof for a small model with synthetic weights, which the programs generate
themselves:

```
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)"
```

Its last line is `YES`, with exit code 0.
That `YES` [shows only](../examples/accepted-proof/README.md#what-the-yes-says) that `cp_verify`
runs and that these files fit together.
It says nothing about a served model.

The example reads the commitment ID, 64 hex digits that identify one commitment, from its own files.
That is fine only for a demonstration: get a real one from [a party you
trust](verify-a-proof.md#what-you-need-first). A stored full-size proof, which also needs no GPU, is in
[examples/full-size/](../examples/full-size/README.md).

## Route 2: make toy proofs

Route 2 self-tests the proof system's library on a toy statement.

```
W=$(mktemp -d) && LOOK="$W/look" && echo "LOOK=$LOOK"
blocks/first_look.sh "${LOOK:?first run the line with mktemp}"
```

A good run has exit code 0, and its third line from the end begins
`AS EXPECTED: the honest copy was accepted`. If it stops with `… name a new folder` and exit code 2,
run both lines again. This is a self-test and nothing more: one program makes and checks the proofs,
with random values fixed in the source.

## Route 3: a small model on one H100

You need:

- CUDA 13.2 Update 2 (nvcc 13.2.86) on PATH
- gcc and g++ 13 first on PATH
- binutils, flock and python3
- an x86-64 CPU with the SHA extensions
- one NVIDIA H100 with 80 GB that nothing else is using
- a working NVIDIA driver: `nvidia-smi` must list the H100 (tested† with driver 580.178.04; the
  lowest driver that works is not known)
- on a machine with several H100s, perhaps NVIDIA's Fabric Manager service ([details](../docs/quickstart.md#3-gpu-a-small-model))
- the repository and the work directory on a local disk

[The prover's build page](../prover/docs/build.md#install-the-compiler) prints pip lines that
install that compiler. No other way has been tried.

This script builds `cp_verify` and two GPU programs: `cp_commit`, which commits to a model's
weights, and `cp_prove`, which proves requests. It then commits to an 8-layer model with synthetic
weights, proves one 64-token request and checks the proof.

```
examples/quickstart-gpu.sh
```

It can be quiet for over a quarter of an hour. It also logs to `quickstart-gpu.log`, in a directory
that it names first. A good run ends with exit code 0 and
`quickstart-gpu: all steps ended as expected; your files are in` and a directory. That shows only
that one honest request was proved and accepted.

**A clean exit is not enough here.** An older nvcc stops the build, but [a newer one does
not](../docs/limits.md#the-compiler). Each line that begins `Compiled by` must end in
`release 13.2, V13.2.86`.

## Route 3 at full size

After route 3's build, this runs `target_full_f2`, the full-size example, which has the shape of
Llama 4 Maverick, with synthetic weights. It uses the default setting, the proof parameters of every
example in this repository. Its proof covers 1,001 tokens.

```
W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"
prover/examples/run_target_example.sh target_full_f2 "${WORK:?first run the line with mktemp}"
```

A good run prints `status 0`, a line that begins `proof of work:`, then `YES`, exit code 0. Progress
goes to `$WORK/cp_commit.log`, not to the screen.

Memory can run out. The script checks it once, at the start, and exits with code 2 if there is too
little. Memory use peaks late in the run, and the check can miss a scheduler's limit. If memory runs
out, the kernel may kill other processes too.

## Route 4: Llama 4 Maverick

`examples/maverick/run.sh` downloads a third party's public 4-bit conversion of Llama 4 Maverick,
232.13 GB, and ends with a checked proof of one token. The repository holds none of Meta's files.
The commitment uses `cp_commit`'s option `-public-weights` and a published seed, and [hides nothing
of the weights](../examples/maverick/README.md#the-seed-is-published). Before the run you declare
the weight files public. Do that only for weights that are public: with this option the weights are
not hidden. You also make the calibration set that the script reads, which is not in this repository, from three published files; two of them open only once Meta has granted your request for access, so ask first.

This route has been run on one kind of machine: one H100 with 80 GB. In no run has `download`
fetched from Hugging Face, and no run has had only the 330 GB that the script asks for. [The detailed
docs](../docs/quickstart.md#4-llama-4-maverick-the-full-size-model) have the commands and how far the
route has been tried.

## Where to go next

Routes 3 and 4 are examples only. Before you commit to weights that must stay secret, read [serving
proofs](serving.md) and [weight hiding](weight-hiding.md). See also [verify a
proof](verify-a-proof.md), [what is proved](what-is-proved.md), [known limits](known-limits.md),
[performance](performance.md), [more stored examples](../examples/README.md) and [fixes for error
messages](../docs/quickstart.md#troubleshooting). To build and check the Lean library, follow
[lean/CHECKING.md](../lean/CHECKING.md).

† Not on the released programs: [which kinds of result these
are](../docs/limits.md#on-other-programs).
