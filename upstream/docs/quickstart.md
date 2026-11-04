# Quick start

This page shows how to build the programs and get to your first proof. Routes 1 and 2 need only a
CPU; routes 3 and 4 need an NVIDIA H100, and only those two make a proof about a model's tokens
(which tokens, the table says). If unsure, start with route 1. Coding agents: read
[AGENTS.md](../AGENTS.md) first. The Lean library has no route on this page:
[lean/CHECKING.md](../lean/CHECKING.md) has the commands that build and check it, and
[lean/README.md](../lean/README.md) and [lean/THEOREM.md](../lean/THEOREM.md) say what it proves.

| route | what you do | what you need | time† | host memory |
|---|---|---|---|---|
| [1. CPU: check a proof](#1-cpu-check-a-proof) | build `cp_verify`, check a stored proof, and see an altered copy rejected | Linux, g++ 13 or later, python3 | build: about one to two minutes; check: under two seconds | 1.6 GB (the compiler took about 1.3 GB, on [other sources](glossary.md#other-sources)) |
| [2. CPU: make toy proofs](#2-cpu-make-toy-proofs) | make [toy proofs](glossary.md#toy-proof) with the proof system's library and check them | Linux, g++ 13 or later, make, python3 | about a minute (one run of the script on this repository's own files, so † does not apply to it: 72 seconds, on a busy machine) | under 1.5 GB (in that run about 0.97 GB in its largest process, the compiler) |
| [3. GPU: a small model](#3-gpu-a-small-model) | commit to a small model, prove one reply and check the proof | Linux, python3, gcc and g++ 13, CUDA 13.2 Update 2, one NVIDIA H100 with a working NVIDIA driver, a CPU with the SHA extensions | build: about 29 minutes at 24 CPUs, without the install of CUDA; run: about 23 to 30 seconds | under 10 GB of memory |
| [3, at full size](#the-same-at-full-size) | the same, with the full-size example configuration | everything for route 3 | run: about 42 minutes, at the default setting | 280 GB available |
| [4. Llama 4 Maverick: the full-size model](#4-llama-4-maverick-the-full-size-model) | download the model's public weights, commit to them, prove the token that follows a request of 1,024 token ids from a formula, and check the proof | everything for route 3, plus `curl`, Python 3.13, a calibration set that you make first, and disk space: 245 GB | hours | 330 GB of available host memory; the highest peak seen is 310 GB†; never tried at 330 GB |

The test suite runs the verifier's build and routes 2 and 3 (of route 3 the small model, by the two
blocks of commands in section 3 below that build and run the example, and not by the one-call
script, which no row runs, and not the full size) on programs that it builds from these sources
(rows `build.verifier`, `new.first_look`, `gpu.quick_start` and `gpu.run_target_example` of
[SUITE.md](../SUITE.md)). The times for routes 1 and 3, and their memory but for the 280 GB, come
from three other runs†. Each ran on one H100, with the shell bound to 24 of the host's CPUs (which
tells the machine, not a need) and with `OMP_NUM_THREADS=1` set in the shell, which a reader's
machine would not have.

† Not on the released programs: [which kinds of result these are](limits.md#on-other-programs). The
same mark on a figure further down this page means the same.

With no GPU, you can also check a stored full-size proof in
[examples/full-size/](../examples/full-size/README.md) (synthetic weights), which shows what a client
of a full-size model downloads and runs.

No route on this page measures a rate: [Performance](../guide/performance.md) has the rates.

**How a check ends.** When `cp_verify` accepts a proof, the exit code is 0 and the last line it
prints is `YES`, or a line that begins `WITHIN LIMITS:` if a token of the request was checked under
a [clipped-gap policy](glossary.md#clipped-gap). When it rejects one, the last line starts with
`NO:` and the exit code is 1. [verifying.md](verifying.md#what-a-yes-means) explains what an
accepted proof means and lists [every other result](verifying.md#exit-codes).

**Before you start**

- Run every command in bash, from the repository's root directory (the one with README.md).
- Run one line at a time. If you paste a whole block, the shell keeps going even after a line fails.
- Check each result against this page before you move on. `echo $?` shows the last command's exit
  code.
- Some lines set a variable and print it, for example `WORK=…`. In a new shell, paste the printed
  text first. If your tool starts a new shell for every call, send the whole block as one call, its
  lines joined with `&&`.
- If you are a coding agent whose calls have a time limit, run each build, and
  `examples/quickstart-gpu.sh`, in the background, send the output to a file and read its end in
  later calls ([the verifier's build](../AGENTS.md)). Do not start a second run while you wait.

## Install

For routes 1 and 2; routes 3 and 4 list what else they need. The test suite needs more than the
command below installs, among other things numpy, pytest, clang++ and Lean
([SUITE.md](../SUITE.md#run-the-tests)). The verifier's build and route 2 have been run† on Ubuntu
24.04 (x86-64) with g++ 13.3, GNU Make 4.3 and Python 3.12 or 3.13.

A bare Ubuntu 24.04 system has no `g++`, `make`, `python3` or `unzip`; the command below installs
them. Run it as root, or put `sudo` before each `apt-get`. No other command on this page needs root,
but for the line on a new account further down. On a system without `/etc/localtime` it stops at two
questions of the package `tzdata`, in spite of `-y`, and waits for the answers. A tool without a
terminal needs `DEBIAN_FRONTEND=noninteractive` set for this command (not tried here). If you unpack an
archive, use a tool that keeps the files' execute permission: `unzip` does, and `python3 -m zipfile`
does not.

```
apt-get update && apt-get install -y --no-install-recommends g++ make python3 unzip
```

If you are root and have no other account, as in a fresh container, put the archive where a new
account can read it (`/tmp`, not `/root`), make the account with the line below, and unpack in its
home and go on as that account. Without a terminal, as in a tool that starts a new shell for every call, `su - tester` ends at once, silently,
with exit code 0, and the next call is root again: give each call as
`su - tester -c 'cd DIRECTORY && COMMAND'` (it starts in the account's home). `su - tester` shares
root's terminal: where the kernel lets a process push input into its terminal, hostile code run as
`tester` could type into root's shell. `su -P` in place of `su` closes that (reasoned, not tried).

```
useradd -m -s /bin/bash tester && su - tester
```

A block with quotes of its own, such as route 1's "See a rejection", does not fit between the quotes
of `-c '…'`. Send its lines as printed in a here-document instead. The quotes around `END` matter.

```
su - tester <<'END'
cd DIRECTORY || exit
THE BLOCK'S LINES, AS PRINTED
END
```

## 1. CPU: check a proof

**Build it.** The build produces `bin/cp_verify`.

```
verifier/build.sh
```

The script goes quiet while it compiles. A long line that begins with `build.sh: note:` may come
first: it is a warning, not an error ([Troubleshooting](#troubleshooting)). A successful build ends
with `THE VERIFIER IS BUILT:` and the program's path, and exit code 0.

**Check the stored proof.** The proof is for a small model with [synthetic
weights](glossary.md#synthetic-weights).

```
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)"
```

The arguments are the configuration directory, the [public key
directory](glossary.md#public-key-directory), the request, the file with the next token, and the
proof. The `-id` option gives the [commitment ID](glossary.md#commitment-id).

The check prints more than a hundred lines; the last is `YES`, and the exit code is 0. This shows
only that these files fit together: it says nothing about any model that anyone serves ([what the
YES says](../examples/accepted-proof/README.md#what-the-yes-says)).

The example includes its own request, configuration and commitment ID, which is fine for a
demonstration. When you check a real provider, pass the ID that you obtained before you sent the
prompt, from whoever vouches for the model, not through the server you are checking. Never read it
from files that came with the proof. See [when to accept](verifying.md#when-to-accept).

**See a rejection.** Flip one bit in a copy of the proof, then check the copy.

```
W=$(mktemp -d) && ALTERED="$W/altered.proof" && echo "ALTERED=$ALTERED"
python3 -I -c 'import sys; b = bytearray(open(sys.argv[1], "rb").read()); b[len(b) // 2] ^= 1; open(sys.argv[2], "xb").write(b)' examples/accepted-proof/example.proof "${ALTERED:?first run the line with mktemp}"
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y "${ALTERED:?first run the line with mktemp}" -id "$(cat examples/accepted-proof/ID.txt)"
```

The second line prints nothing. The third ends with a line that begins with `NO:`, and exit code 1.

[examples/README.md](../examples/README.md) lists the other stored examples.

## 2. CPU: make toy proofs

This route is a self-test of the proof system's library, on a toy statement about a small synthetic
matrix. It uses the same library, file formats and file parsers as `cp_commit`, `cp_prove` and
`cp_verify`.

```
W=$(mktemp -d) && LOOK="$W/look" && echo "LOOK=$LOOK"
blocks/first_look.sh "${LOOK:?first run the line with mktemp}"
```

The script echoes each command on a line that begins with `+`. First it compiles one small program,
and prints nothing meanwhile. Then it makes and checks six proofs, rejects 54 bad inputs
(altered copies, and honest files passed off as something else), and checks one proof before and
after flipping one bit. A successful run has exit code 0 and ends like this:

```text
AS EXPECTED: the honest copy was accepted, and the copy with one bit flipped was rejected. The last line of each check:
…/look/honest.proof: … bytes: ACCEPTED
…/look/bent.proof: … bytes: an opening does not lead to its root
```

The script writes only into the new directory, apart from the compiler's temporary files and, if you
have set `DEPENDENCIES_OUTPUT` or `SUNPRO_DEPENDENCIES`, the file that it names, and does not remove
anything. If it cannot use the directory, it says why and stops with exit code 2.

Treat this as a self-test only. One program both makes and checks the proofs, and the randomness
is hard-coded in the source, so it shows only that the library runs on your machine. The toy's
parameters count [proof of work](glossary.md#proof-of-work), so several of its lines say `YES; proof of work counted`. The word `ACCEPTED` comes from this program, not from
`cp_verify`.

## 3. GPU: a small model

**You also need:**

- CUDA 13.2 Update 2 (nvcc 13.2.86), with `nvcc` on `PATH` or in `/usr/local/cuda/bin`. NVIDIA's
  Python packages are the way that has been tried: pip puts them into a directory of your own, so
  this way also needs Python 3 with pip (a minimal Ubuntu 24.04 system may not have python3-pip).
  [build.md](../prover/docs/build.md#install-the-compiler) prints the ten lines, and says what they
  need and what has been tried with them. In every new shell the first of the ten lines and the line
  with `export` are needed again: without them a shell finds no nvcc, or another one. NVIDIA's apt
  package for Ubuntu 24.04 is `cuda-toolkit-13-2`; CUDA 13.2 Update 2 as NVIDIA's installer or apt
  installs it has not been tried
- gcc and g++ 13, first on `PATH`, plus binutils and `flock`
- the repository and the work directory on a local disk, because the build and the programs lock
  directories
- to run the programs, an x86-64 CPU with the SHA extensions and one NVIDIA H100 with 80 GB that
  nothing else is using (the build needs no GPU; if it finds one, it first runs a small test program
  on it; without one the sample is built and not run, and the build goes on with the check of the
  version alone)
- to run the programs, an NVIDIA driver that is installed and works: `nvidia-smi` must list the
  H100. It has been tested† with driver 580.178.04; the lowest driver that works is not known ([NVIDIA names a lowest driver for each CUDA
  release](../prover/docs/build.md#which-cuda-libraries-the-programs-have-run-with): 580 for
  CUDA 13)
- on a machine with several H100s, NVIDIA's Fabric Manager service may have to be running too: no
  run without it is known, so what the programs print then is not known

If nvcc is older, the build stops with a one-line message. CUDA 13.4 is not supported ([which
toolkit](limits.md#the-compiler)).

`cp_commit` and `cp_prove` stop with a message that names `gpu-in-use` if more than 2 GiB of the
GPU's memory is in use when they start. So `nvidia-smi` should show well under that beforehand. By
design, `cp_commit` fills nearly all of the card's free memory with a cache, even for the small
example.

One script runs this whole route:

    examples/quickstart-gpu.sh

Exit code 0 means success, with one care: a newer nvcc on `PATH` does not stop the script, so look
in its output for the lines that begin `Compiled by` (see "Build", below). The script first checks
most of what the machine needs, so a missing piece shows up in the first minute rather than after
the long build. It can go quiet for more than a quarter of an hour. Its first lines name a new
directory under `TMPDIR` or `/tmp`, and everything it prints also goes to `quickstart-gpu.log`
there. A run that finishes on its own ends with a line that begins with `quickstart-gpu:`; if that
line is missing, the run was cut short.

The rest of this section goes through the same steps by hand.

**Check the machine.** The dynamic loader must find the NVIDIA driver's library, `libcuda.so.1`,
through its own configuration. Setting `LD_LIBRARY_PATH` does not help, because the programs
re-execute themselves with it unset.

```
/sbin/ldconfig -p | grep -E 'libcuda|libcudart|libcublas'
grep -m1 -ow sha_ni /proc/cpuinfo
```

You should see a line for `libcuda.so.1`, then `sha_ni`. If `libcuda.so.1` is missing, the NVIDIA
driver is not installed, the container was started without the GPU, or a container finds it only
through `LD_LIBRARY_PATH`; see [prover/docs/build.md](../prover/docs/build.md#the-drivers-library).
Without `sha_ni`, the machine most likely cannot run this route ([what happens
then](limits.md#the-sha-instructions)). Then run `nvidia-smi`: it must list the H100. Finding the
library does not show that the driver works. `examples/quickstart-gpu.sh` runs `nvidia-smi` itself
before anything else, and `examples/maverick/run.sh` before each stage that starts a GPU program;
`prover/examples/run_target_example.sh`, which the steps below call, does not.

**Build.** Always build the verifier first. The prover's build copies `bin/cp_verify`, and stops at
once if that program is missing or was built from other sources.

```
unset OUT BUILD GPU_RUN
verifier/build.sh
prover/build.sh
```

The first line clears three environment variables that would otherwise redirect the prover's build
or the example. The prover's build is long, and prints little except compiler warnings, which a
successful build prints too. It ends with `THE PROGRAMS ARE BUILT:` and the directory
`prover/build/hc`, and exit code 0.

A newer nvcc does not stop the build: at each call of the compiler's check it prints a line that
begins `check_nvcc_version.sh: note:` and goes on, and the programs then come from another compiler
than the one that this version requires. So look in the build's output for the lines that begin
`Compiled by`: each must end in `release 13.2, V13.2.86`. If one does not, put the `bin` directory
of CUDA 13.2 Update 2 first on `PATH` and build again.

Do not run two prover builds at once. `flock` makes the second one stop with exit code 2. Where
`flock` cannot lock a directory, as on some network file systems, a lone build stops the same way
([build.md](../prover/docs/build.md#build)).

**Commit, prove and check.** The weights are [synthetic](glossary.md#synthetic-weights).
`target_small_f2_roomy` is an [example configuration](glossary.md#example-configuration): a model
with 8 layers and 64-token requests.

```
W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"
prover/examples/run_target_example.sh target_small_f2_roomy "${WORK:?first run the line with mktemp}"
```

The script prints five numbered steps and echoes each command before running it:

1. It copies the configuration.
2. `cp_commit` samples a seed, commits to the weights and prints the [commitment
   ID](glossary.md#commitment-id).
3. It copies the configuration's stored request: a synthetic 16-token prompt, then 48 tokens that
   are the model's own top choices.
4. `cp_prove` answers it.
5. `cp_verify` checks the proof as a client would, using public files only.

A successful run ends with the lines `status 0`, `proof of work: up to 32 bits counted in …` and
`YES`, and exit code 0. `status 0` is the exit code of `cp_verify`. [When to
accept](verifying.md#when-to-accept) explains the lines above them. The figures in the
`proof of work:` and `accounting:` lines are the programs' own, and none of them is a security level
([what these figures are worth](limits.md#the-soundness-figure)). If `cp_commit` or `cp_prove` fails, the script prints the
last lines of its log and exits with code 1.

**What this shows.** Only that one honest request was proved and accepted. The proof covers each of the 48 tokens, and the token after the request. Here one script
acts as both provider and client on one machine, using the prover's own copy of `cp_verify` and
taking the ID from `cp_commit`'s log. A real client must do neither ([when to
accept](verifying.md#when-to-accept)).

| in `$WORK` | what it is |
|---|---|
| `config/` | the three public configuration files |
| `ivk/` | the [public key directory](glossary.md#public-key-directory), for clients |
| `isk/` | the [secret key directory](glossary.md#secret-key-directory), which a provider keeps to itself |
| `in/example.req`, `out/example.y`, `out/example.proof` | the request, the token after it (the model's next choice) and the proof |
| `cp_commit.log`, `cp_prove.log`, `cp_verify.log` | the three programs' output |

Only your account can read `$WORK`. Nothing removes that directory, but many systems clear `/tmp` on
reboot or after a few days. Use it for examples only. For secret weights,
[operators-duties.md](operators-duties.md) explains where to put these directories. Before you
commit to weights that must stay hidden, read [docs/operators-duties.md, "A
record"](operators-duties.md#a-record), and keep its duties.

## The same at full size

This needs 280 GB of available host memory, and a few gigabytes of disk under `TMPDIR` or `/tmp`.
Three runs at the default setting† used up to 268 GB in the run's whole container, file cache not
counted (in the timed runs on synthetic weights the container reached 278.1 GB while committing:
[docs/limits-all-rows.tsv](limits-all-rows.tsv), row RES-24); each of the three runs† had a memory
limit of 463.9 GB. A run takes nearly all the memory of an H100 with 80 GB, and no GPU with less has
been tried.

Before it creates anything, the script checks the available host memory and, in a container, what is
left under the container's limit. If there is too little, it prints how much it needs and how much
it found, and stops with exit code 2. The check has two gaps:

- It runs once, at the start, but memory use peaks late in the run, so it misses whatever other
  processes take in the meantime.
- It can miss a limit set by a job scheduler such as Slurm, so if you run under a scheduler, request
  the memory yourself.

If memory still runs out, the kernel may kill other processes too.

Run the example again in a new work directory, with `target_full_f2`:

```
W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"
prover/examples/run_target_example.sh target_full_f2 "${WORK:?first run the line with mktemp}"
```

`target_full_f2` is the full-size example configuration under exact match, again with synthetic
weights and the default setting. The stored request has a prompt of 24 tokens and 1,000 tokens of
the model's own; the proof covers the 1,000 and the token after them.

The run takes far longer than the small one, and `cp_commit` prints nothing to the screen for most
of it; follow its progress with `tail -f "$WORK/cp_commit.log"` in another shell. A successful run
ends the same way: `status 0`, the proof-of-work line, and `YES`.

## 4. Llama 4 Maverick: the full-size model

`examples/maverick/run.sh` takes the public weights of Llama 4 Maverick from download to a checked
proof of one token, the one that follows a request of 1,024 token ids from a formula, on one H100,
at the [default setting](glossary.md#default-setting), which counts proof of work. It works in
stages: `needs`, `build`, `download`, `checksums`, `fit`, `commit`, `compare` and `prove`. The stage `fit` reads a calibration set, which you make first (**You need**, below).

This route has been run on one kind of machine: one H100 with 80 GB. With the released programs it
has run once, on the real 4-bit files, with the programs built beforehand and the weight files
already in place, so that run shows neither the stage `build`, nor a `download` that fetches, nor a
build that you make from these sources. In no
run has `download` fetched from Hugging Face. The highest memory peak seen† is 310 GB, and no run has had only the 330 GB that the script asks for: plan by the 330 GB. [The script's own
page](../examples/maverick/README.md#where-the-values-in-expectedtxt-come-from) has the runs.

The seed file, `public-seed.bin`, is published, so no commitment that this script generates hides
anything of the weights ([The seed is
published](../examples/maverick/README.md#the-seed-is-published)).

The repository hosts none of Meta's files ([which of this model's files are in the repository, and
why](../examples/maverick/README.md)). It gives you where to download the weights (the `download`
stage fetches a third-party conversion of them from Hugging Face), the code that makes the
commitment (the `fit` and `commit` stages), and the expected commitment ID, in
`examples/maverick/expected.txt`.

**You need** everything route 3 needs (its compiler has ten install lines of its own:
[build.md](../prover/docs/build.md#install-the-compiler)), plus `curl` and Python 3.13 (Ubuntu 24.04
comes with 3.12) with the packages pinned in `requirements.txt`. The `needs` stage names the CUDA
release but does not check it, and the `build` stage goes on with a newer nvcc: look at the lines
`Compiled by`, as in [route 3](#3-gpu-a-small-model)'s build. The table at the top has the host
memory and the disk space. Run the two install commands first, and the second again in every new
shell: [You need](../examples/maverick/README.md#you-need) has them, and says what the script
checks. You also need the calibration set, which the first two lines below make room for: see under them. Two of the three files that it is made from open only to users whose request for access Meta has granted, each at its own repository on huggingface.co, and each request includes accepting that model's licence: ask before you rent a machine.

**Run it.** Type the first two lines, then the line under this block, and only then the third.

```
W=$(mktemp -d /var/tmp/maverick.XXXXXX) && M="$W/work" F="$W/weights" && echo "M=$M F=$F"
(umask 022; mkdir "${F:?first run the line with mktemp}" && while read -r sum name; do python3 -I -c 'import os, sys; sys.tracebacklimit = 0; os.write(os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), (sys.argv[2] + "\n").encode())' "${F:?}/$name.weights-are-public" "$sum  $name" || exit 1; done < examples/maverick/weights.sha256)
examples/maverick/run.sh all "${M:?first run the line with mktemp}" --weights "${F:?first run the line with mktemp}"
```

Between the second line and the third, make the calibration set, as the account that runs the third line (no GPU; the three words in capitals are the paths of your copies of the three files that [scales/README.md](../scales/README.md#the-calibration-set) lists):

```
python3 scales/make_calibration_set.py TRAIN_PARQUET LLAMA2_TOKENIZER_JSON LLAMA4_TOKENIZER_JSON "${F:?first run the line with mktemp}/calibration_set.npy"
```

It must end with `the set is as expected` and exit code 0; with another result the third line stops, and a file written at exit code 1 stays where it is: [You need](../examples/maverick/README.md#you-need).

1. The first line picks two new directories under `/var/tmp`: `M`, the work directory, and `F`, the
   weights directory. Together they need 245 GB. Many systems delete old files under `/var/tmp`, so
   to keep yours, use another directory instead ([directory rules](#directory-rules)).
2. With the second line you declare that the weights are public; [Public
   weights](operators-duties.md#public-weights) explains what that option gives up. Do that only for
   weights that are public: with this option the weights are not hidden.
3. The third line checks the machine and then runs the stages in order. It prints each step's
   command, the expected result and the actual one, and stops at the first step where they differ.
   It downloads 232.13 GB and takes hours. With `needs` in place of `all`, the script prints what
   every stage needs and the measured times, checks the machine (not for what `build`
   needs), stops at the first lack (seen without a GPU: exit code 1), and writes nothing ([other
   ways to call the script](../examples/maverick/README.md#other-ways-to-call-the-script)). Any
   stage's name can replace `all` ([the stages](../examples/maverick/README.md#the-stages)).

To run the third line in the background, as a coding agent should, see [In the
background](../examples/maverick/README.md#in-the-background).

**If all goes well,** the `compare` stage prints `Your commitment is the published one.`, the last
stage, `prove`, reports `the last line YES`, and the exit code is 0.

[The script's own page](../examples/maverick/README.md) has the rest: other ways to call the script,
and why the fit may not repeat on your machine. Read there what to do [if you already have the weight
files](../examples/maverick/README.md#if-you-already-have-the-weight-files) or [if it
stops](../examples/maverick/README.md#if-it-stops), and which [memory limits the script does not
see](../examples/maverick/README.md#memory-limits-that-the-script-does-not-see).

<a id="files-of-8-or-16-bits"></a>**Weight files of 8 or 16 bits.** Route 4's script also takes the
same publisher's 8-bit or 16-bit files with `--width 8` or `--width 16` ([the script's
page](../examples/maverick/README.md#the-8-bit-and-the-16-bit-files)). No run on these files has used
the released programs. Runs on them without that script took far more host memory than route 4's
files do, and by hand nothing checks the memory first. `cp_commit` and `cp_prove` accept such files without warning, so it is for the provider to know how each tensor
of its files is stored: read [the limits on such files](limits.md#files-of-8-or-16-bits) before you
commit to one.

## What these routes show

No route on this page runs a prover that tries to cheat. So a `YES` here shows only that honest
files were accepted
([limits.md](limits.md#few-runs)).
[What is proved](../guide/what-is-proved.md) and [Weight hiding](../guide/weight-hiding.md) say what
is established, and of what.

## Troubleshooting

| what you see | why | what to do |
|---|---|---|
| `first_look.sh needs …`, or `the build failed` | make, python3, a C++17 compiler or a GNU coreutils tool is missing | see [Install](#install); `make.log` in route 2's new directory has the compiler's messages |
| `NOT AS EXPECTED` (exit code 1), or `the toy's files were not made or not accepted` (exit code 2) | a step failed, or the program accepted an altered copy or rejected an honest file | if an altered copy was accepted, report it as [SECURITY.md](../SECURITY.md) says; exit code 2 is also the script's code for wrong usage, an unusable path or a missing tool, so go by the printed line, not by the code |
| `… name a new folder`, exit code 2 | the scripts of routes 2 and 3 need a directory that does not exist yet | run the block again from its `mktemp` line |
| `… which another account may be able to change …`, exit code 2 | the scripts of routes 2, 3 and 4 check every directory above the new one before they create anything ([Directory rules](#directory-rules)) | the message names the directory and the fix |
| a line with `note:` and the words `another account may be able to change`, and the run continues | another account can write to a directory that the run depends on: the repository, route 4's weights directory, or one above them | this is not an error; the note says what another account could do there and, if the directory is yours, how to fix its permissions |
| `build.sh: warning: the programs get no rpath to the CUDA libraries:`, or `build.sh: note:` … `not to root` | the build does not fully trust the toolkit's library directory | [prover/docs/build.md](../prover/docs/build.md#where-the-programs-find-the-cuda-libraries) explains both messages |
| `this step failed with status 127` | the loader cannot find a CUDA library; the next line names it | for `libcuda.so.1`, see "Check the machine" in [route 3](#3-gpu-a-small-model); for any other, see [prover/docs/build.md](../prover/docs/build.md#where-the-programs-find-the-cuda-libraries) |
| `this step failed with status 132`, or a bare `Illegal instruction` | the CPU, or the virtual machine, probably lacks the SHA extensions | see "Check the machine" in [route 3](#3-gpu-a-small-model) |
| `this step failed with status 5`, and `cp_commit.log` has a line that begins `GPU fault:` | a CUDA call failed; the words after `GPU fault:` are the CUDA library's own | run `nvidia-smi`. If it fails, or does not list the H100, the driver is missing or does not work, or the container was started without the GPU. If it lists the H100 and the machine has several, see the bullet on Fabric Manager in [route 3](#3-gpu-a-small-model). Neither case has been seen in a run |
| `build.sh: a serving build is refused:` and a reason | `bin/cp_verify` is missing or was built from other sources, or a source directory (`prover/`, `blocks/` or `verifier/src/`) holds a stray file, such as an editor's dot-file | build the verifier, then the prover again; or build in a clean copy of the repository |
| `Permission denied`, a line with `belongs to root, in`, or a build that fails, after you ran a step as root | root owns whatever that step created | start again in a new copy of the repository ([more](limits.md#the-tests-and-your-files)) |
| step 2 or step 4 fails, and the log names `gpu-in-use` | something else is using the GPU | check `nvidia-smi`; wait, or choose a free GPU by its number in `nvidia-smi` (`export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`); then run the block again from its `mktemp` line |
| at full size, `cp_commit` or `cp_prove` starts slowly | likely too little free host memory, or, if you gave `-huge-pages`, the request for huge pages | wait: a slow start need not be a hang ([a program that hangs](faults.md#a-program-that-hangs)); see [A slow start at full size](#a-slow-start-at-full-size) |
| step 4 fails, and `cp_prove.log` ends with `p258 end`, an exit code and, in most cases, a name | once `cp_prove` has read its secrets, it prints at most a short name for the reason why it stopped | [prover/docs/ends-in-short.md](../prover/docs/ends-in-short.md) explains each name |
| not enough space in `/tmp`, or `/tmp` does not allow programs to run | routes 1 to 3 make their directories under `TMPDIR`, or `/tmp` by default | export `TMPDIR` as the full path of an existing directory on a local disk, outside the repository, that meets the [directory rules](#directory-rules) |
| in route 4, `the calibration set, …` or `… calibration_set.npy has the SHA-256 …`, exit code 1 | the script reads the set in the weights directory, and only at the SHA-256 that it holds | make the set by the line under [route 4's block](#4-llama-4-maverick-the-full-size-model); a wrong file stays where it is, so move it away first ([more](../examples/maverick/README.md#you-need)) |

<a id="directory-rules"></a>**Directory rules.** Every directory above the new directory that you
name must exist, be owned by you or by root, and not be writable by its group or by others unless it
has the sticky bit. If you are root, that means owned by root, or, under `sudo`, by the account that
ran it. The `mktemp -d` lines on this page meet this wherever `/tmp` (for route 4, `/var/tmp`) has
the sticky bit. In a rootless container where `/` shows an unmapped owner, the scripts reject every
directory: use a container in which `/` belongs to the container's own root, or run outside one.

<a id="a-slow-start-at-full-size"></a>**A slow start at full size.** Two causes are likely: too
little free host memory, since from reading the code the programs can take up to twice the weight
files' size at a start (the second copy is the system's file cache, which the peaks behind route 4's
330 GB leave out: [enough host memory](operators-duties.md#keys-seeds-and-machines)), or
`-huge-pages`, which no script named on this page passes.

A full-size `cp_commit` or `cp_prove` is also slow to exit after its last result. Wait for the
process to end before you start the next program. During such an end a call of `nvidia-smi` may
hang: give a monitoring script's call a time limit ([how long an exit
takes](../prover/docs/ends-in-short.md#how-long-an-exit-takes-at-full-size)).
