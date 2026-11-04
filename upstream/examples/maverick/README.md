# Llama 4 Maverick: from the public weights to a checked proof

`run.sh` downloads public 4-bit weight files of Llama 4 Maverick, generates the commitment to them,
compares it with the published one, proves one request and checks the proof. It asks for one NVIDIA
H100 with 80 GB, 330 GB of host memory and 245 GB of disk space, and takes hours.
[docs/quickstart.md, route 4](../../docs/quickstart.md#4-llama-4-maverick-the-full-size-model) has
the short form of this page. How these public weights fared in the Lean library's two worst-case
checks (neither is `cp_commit`'s), by a program of this project, and what that doesn't show:
[docs/limits.md](../../docs/limits.md#narrower-than-what-runs).

## What is here

None of Meta's files is in this repository. For this model it has three things: the address of the
weight files, the code that generates the commitment, and the commitment, as its commitment ID. The
other files here are checksums and digests to compare a run with, the seed file and the figures that
the script prints and checks for.

| file | what it is |
|---|---|
| `run.sh` | the script; without arguments it prints its stages and options. It has the weight files' address and revision, and calls the code that generates the commitment: `bin/weight_sums`, `scales/` and `cp_commit` |
| `weights.sha256` | the SHA-256 checksums of the five weight files |
| `public-seed.bin` | the published seed file, 32 bytes |
| `figures.txt` | measured times and peaks of host memory, which `run.sh` prints, and the host memory and disk space that it checks for |
| `expected.txt` | the commitment ID, and the SHA-256 digests of four files that a run generates: `wsums.txt`, `KL.bin`, `key.bin` and `leaf/0.bin` ([where they come from](#where-the-values-in-expectedtxt-come-from)) |
| `8-bit/`, `16-bit/` | `weights.sha256`, `figures.txt` and `expected.txt` for [the same publisher's 8-bit and 16-bit files](#the-8-bit-and-the-16-bit-files) |

## What is not here, and why

- **The weight files** (232 GB). They are a third party's 4-bit conversion to GGUF
  (`unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF` on Hugging Face) of Meta's
  `meta-llama/Llama-4-Maverick-17B-128E-Instruct`. The conversion's card has the fields
  `license: other` and `license_name: llama4`, and it carries the text of the Llama 4 Community
  License Agreement as the text of an access form. That `llama4` means this agreement is this
  repository's reading: [sources](../../THIRD_PARTY.md#a-public-models-weight-files). You fetch the
  weight files from their publisher, with the stage `download` or by yourself.
- **What a run computes from the weights:** the fitted scales and parameter file, the commitment's
  public key directory, a proof. Of these the repository has only digests and the commitment ID, so
  that as little as possible of what it publishes derives from Meta's model ([what else does, or
  may](../../THIRD_PARTY.md#what-depends-on-metas-models)). So to check a proof for Maverick you
  first generate the public key directory yourself: run the stages up to `compare`.
- **The calibration set.** You make it with `scales/make_calibration_set.py`, from three published
  files that you fetch yourself: the raw training split of WikiText-2 (the corpus that S. Merity, C.
  Xiong, J. Bradbury and R. Socher describe, arXiv 1609.07843; from `Salesforce/wikitext` on Hugging
  Face, whose listing gives its licence as CC BY-SA 4.0 and also lists CC BY-SA 3.0 and the GFDL)
  and the tokenizers of Llama 2 and Llama 4. The two tokenizers open only to users whose request for
  access their publisher, Meta, has granted, each at its own repository on huggingface.co, and each
  request includes accepting that model's licence
  ([scales/README.md](../../scales/README.md#the-calibration-set) lists the three files).

## The seed is published

The commitment is made with `-public-weights` and a published file of 32 bytes, so anyone who has
the weight files and makes the calibration set can remake it. Such a commitment does not hide the
weights, and no privacy between a batch's clients is claimed: it is no example of hiding at work.

For this reason the script asks you to declare each weight file public ([what that option gives
up](../../docs/operators-duties.md#public-weights)). Do that only for weights that are public: with
this option the weights are not hidden. Without `-public-weights`, `cp_commit`
draws the seed itself ([weight hiding](../../guide/weight-hiding.md)).

## You need

Type every line on this page in the repository's top directory, under one ordinary account, never as
root or under `sudo` ([why](../../docs/limits.md#running-the-programs-safely)). If root is the only
account, make another first ([Install](../../docs/quickstart.md#install)). The script's messages
name `sudo`; that is no advice to use it.

- **GPU:** one NVIDIA H100 with at least 80,000 MiB of memory, on which nothing else is running, and
  its driver: GPU 0, unless `CUDA_VISIBLE_DEVICES` names another.
- **Host memory:** 330 GB available for `commit`, 290 GB for `prove` ([limits that the script does
  not see](#memory-limits-that-the-script-does-not-see)).
- **Disk space:** 245 GB in round figures: 232.1 GB for the weight files, and 10 GB for `commit` and
  1 GB for `prove` in the work directory, which must be on a local disk.
- **For the build:** what [route 3](../../docs/quickstart.md#3-gpu-a-small-model) lists, with [the
  pinned compiler](../../prover/docs/build.md#install-the-compiler).
- **Also:** a CPU with the SHA extensions; `curl`, `flock`, `sha256sum`, `ldd`, and GNU `find`,
  `stat` and `readlink`.
- **Python 3.13** with the packages that `requirements.txt` pins.
- **Meta's grant of access** to the tokenizers of Llama 2 and Llama 4, for the calibration set: ask
  before you rent a machine.
- **Hours.** After the download, the longest steps are the build, measured at 29 minutes† on 11 CPU
  cores, and `commit`, at 43 minutes†. In one run† on a shared machine four times lay above those in
  `figures.txt` ([which](#where-the-values-in-expectedtxt-come-from)).

The stage `needs` prints all amounts and times, and checks the machine for most of this: not for the
build's needs, Meta's grant or the hard links that `fit` needs in the weights directory, and of the
Python packages only for the versions of `torch`, `numpy` and `gguf`. The times also depend on your
disk: `cp_commit` and `cp_prove` have the system drop the weight files from its file cache as they
load them, so each may read the 232 GB from the disk again, and `cp_commit` twice. No option turns
this off.

These two lines install the packages into a new directory `VENV` and put its `python3` first on
`PATH`:

```
python3.13 -m venv VENV && VENV/bin/pip install --require-hashes -r requirements.txt
source VENV/bin/activate
```

In a new shell, run the `source` line again. Ubuntu 24.04 comes with Python 3.12: if `python3.13` is
missing, install it alongside the system Python, for example from [the sources at
python.org](https://www.python.org/downloads/).

## Run it

```
W=$(mktemp -d /var/tmp/maverick.XXXXXX) && M="$W/work" F="$W/weights" && echo "M=$M F=$F"
(umask 022; mkdir "${F:?first run the line with mktemp}" && while read -r sum name; do python3 -I -c 'import os, sys; sys.tracebacklimit = 0; os.write(os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), (sys.argv[2] + "\n").encode())' "${F:?}/$name.weights-are-public" "$sum  $name" || exit 1; done < examples/maverick/weights.sha256)
python3 scales/make_calibration_set.py TRAIN_PARQUET LLAMA2_TOKENIZER_JSON LLAMA4_TOKENIZER_JSON "${F:?first run the line with mktemp}/calibration_set.npy"
examples/maverick/run.sh all "${M:?first run the line with mktemp}" --weights "${F:?first run the line with mktemp}"
```

1. The first line picks two new directories under `/var/tmp`: `M`, the work directory, and `F`, the
   weights directory. Many systems delete old files under `/var/tmp`, so to keep yours, use another
   directory. In a new shell, paste the printed `M=… F=…` line instead of running this line again.
2. With the second line you declare that the weights are public ([The seed is
   published](#the-seed-is-published)): for each weight file it creates a file with the same name
   plus `.weights-are-public`. Without these declarations the script stops.
3. The third line makes the calibration set. It needs no GPU, so you can make the set on another
   machine before you rent one, and copy it into place. Its three words in capitals are the paths of
   your copies of the three files. It must end by printing `the set is as expected`, with exit code
   0: the script stops (exit code 1) without the set, and at a set with another SHA-256 than the one
   that it holds. The program overwrites no file, so move yours away before you make the set again.
4. The fourth line checks the machine and then runs [the stages](#the-stages) in order. It prints
   each step's command, the expected result and the actual one, and stops at the first step where
   they differ.

Whichever directories you use, `M` must be on a local disk, the directory above it must exist, and
neither `M` nor `F` may be inside the other. The script rejects `M` if another account may be able
to change it or a directory above it, and `F` too while weight files are missing. The full path of
`M` may contain only letters, digits, `/`, `.`, `_` and `-`. Only the account that made a work
directory can use it again. From `fit` on, both directories must stay in place.

## In the background

To run it in the background, as a coding agent should, use this line in place of the last line under
[Run it](#run-it):

```
{ examples/maverick/run.sh all "${M:?}" --weights "${F:?}"; echo "status $?"; } >>"${M:?}.out" 2>&1 &
```

The run is over when the last line of `$M.out` is `status N`. It also stops if the terminal closes,
so over `ssh` use `tmux` or `screen`.

## What the last lines look like

If all goes well, the `compare` stage prints `Your commitment is the published one.`, and the exit
code is 0. Near the end the script repeats `cp_verify`'s output, with the line on proof of work and
the error bound ([when to accept](../../docs/verifying.md#when-to-accept)). The last lines look like
this:

```text
expected: exit status 0 and the last line YES
got:      exit status 0 and the last line YES

The proof is /var/tmp/maverick.XXXXXX/work/proof.1/out/example.proof.
The next token's id is the 4-byte number in example.y, next to it.

examples/maverick/run.sh: stage prove finished as expected, in N s
examples/maverick/run.sh: your files are in /var/tmp/maverick.XXXXXX/work
```

The request is 1,024 token ids from a formula, and the proof covers one token, the next one ([what
an accepted proof means](../../docs/verifying.md#what-a-yes-means)).

## The stages

`all` runs these stages in this order. Each stage from `fit` on needs what the stages before it
made.

| stage | what it does |
|---|---|
| `needs` | prints what the stages need, checks the machine for most of it ([You need](#you-need)), and looks for the declarations and the calibration set; it writes nothing |
| `build` | builds the verifier, the GPU programs and `bin/weight_sums` |
| `download` | fetches the five weight files at a pinned revision; a second call continues incomplete files |
| `checksums` | compares the files' SHA-256 checksums with `weights.sha256` |
| `fit` | computes the quantization scales from the files and the calibration set, on the GPU, compares the digests of `wsums.txt` and `KL.bin` with `expected.txt`, and asks `cp_verify`, in a planning call, whether it accepts the fitted files |
| `commit` | generates the commitment |
| `compare` | compares the commitment ID and the digests of four files with `expected.txt` |
| `prove` | proves the example request and checks the proof with `cp_verify` against the commitment ID in `expected.txt` |

## Other ways to call the script

```text
examples/maverick/run.sh <stage> <work directory> --weights <directory> [--width 8|16] [--limits <file>]
```

Any stage name works in place of `all`. Go on to the next stage only after exit code 0: a stage then
prints `stage NAME finished as expected`, and `needs` prints
`this machine has what the stages after build need; nothing was written`. The two options have
sections of their own: [`--width`](#the-8-bit-and-the-16-bit-files) and
[`--limits`](#commit-under-your-own-limits).

The script re-executes itself with a nearly empty environment, so `LD_LIBRARY_PATH` does not reach
anything it runs. It keeps only `PATH`, `CUDA_VISIBLE_DEVICES`, `SUDO_UID`, three proxy variables
(`https_proxy`, `all_proxy`, `no_proxy`, in lower or upper case) and `curl`'s two certificate
variables.

## What a run leaves in the work directory

The first call of a stage other than `needs` creates the work directory, closed to other accounts.
`fit`, `commit` and `prove` each write to a new numbered subdirectory, and a later stage uses the
newest finished fit and the newest complete commitment. Everything stays until you remove it, a
stopped call's files too.

| path | what it is |
|---|---|
| `fit.N/` | `config/`, the configuration: the fitted `policy.txt` and `KL.bin`, and `layout.txt` |
| `commitment.N/` | `ivk`, the public key directory; `isk`, the secret key directory; `cp_commit.log`, which has the commitment ID |
| `proof.N/` | `in/example.req`, the request; `out/example.y`, the next token's id; `out/example.proof`, the proof |
| `logs/` | `run.N.log`, a call's output (`needs` alone writes none), and the tools' logs |

Two stages also write into the weights directory. `download` creates the weight files and, next to
each one that it starts, an empty file with the same name plus `.download-may-append`. `fit` creates
`wsums.txt` and `fold.txt`, two small text files that the programs read there, where nothing has
these names; for that the file system must offer hard links.

The script does not remove, replace or rename any file that existed before the call, except the
builds' output, in the repository, and the programs' caches, in the work directory. It appends to
such a file in two cases: `download` to an incomplete weight file that has its
`.download-may-append` file, and `cp_prove` to the commitment's counter files (`count.bin`).

## If it stops

The exit code is then 1 or 2; 2 stands for a wrong command line, directory or limits file. The last
message begins with `examples/maverick/run.sh:`. It says what stopped the run and, in most cases,
what to do next. A call that the script cannot read, such as one with an unknown stage, gets the
usage text instead. Mostly you fix the cause and run the same line again.

A repeated `all` checks the machine again. It then skips `build` if the programs are built, weight
files that are complete, `checksums` and `fit` after a finished fit, and `commit` after a complete
commitment.

Ctrl-C at the terminal goes to the script and to what it started. If the script alone is ended, what
it started runs on, and until that ends a second call that would write to the work directory stops
(`another call of this script is still running`).

<a id="the-fit-on-another-machine"></a>**The fit may not repeat on your machine.** Your commitment
ID matches the published one only if your [quantization
scales](../../docs/glossary.md#quantization-scales) match too, and `fit` computes them in float32 on
the GPU, where rounding can differ between GPU models, drivers and `torch`'s CUDA libraries. If
`fit` stops because your scales file, `KL.bin`, differs, reinstall the pinned packages, or use
another H100 machine; no later stage accepts different scales. A difference in the fitted lines of
the parameter file `policy.txt` shows only in `compare`, as another commitment ID.

## If you already have the weight files

Run the first line under [Run it](#run-it), set `F` to your files' directory (in a new shell, again
after you paste the printed line), and run the fourth line with `needs` in place of `all`. Once the
machine passes its other checks, that call stops (exit code 1) where the declarations are missing,
with a command that creates them, and then where the calibration set is missing, with the command
that makes it. After both, run the fourth line as it stands: `download` fetches nothing, and
`checksums` checks your files.

The script prints these commands only if no other account can write in the weights directory or
replace a directory above it, and none owns anything in it. Yet with complete weight files it goes
on in such a directory, and has no defence against another account that writes there while it runs.

The directory must hold the five files themselves, under the names in `weights.sha256`, and no other
`*.gguf` file: `cp_commit` and `cp_prove` read every `*.gguf` file in it, follow no symbolic link
there and take no file that has a second hard link. So copy files that sit behind links, as in a
Hugging Face cache, into one directory first.

## Memory limits that the script does not see

In `needs`, and before `cp_commit` and `cp_prove` start, the script stops if less host memory is
available than it asks for. As available it counts the smaller of `MemAvailable` in `/proc/meminfo`
and, in a container, what is left under the container's memory limit. Under cgroup v1, or if it
cannot read the memory in use, it counts the whole limit as available.

It sees a limit only in the container's own files, `/sys/fs/cgroup/memory.max` or, under cgroup v1,
`/sys/fs/cgroup/memory/memory.limit_in_bytes`. So it misses a limit on a cgroup above the
container's, and one on a job without a cgroup namespace, as a job scheduler such as Slurm can set
it. There, ask the scheduler yourself for the memory that `needs` prints. Neither amount covers what
other processes take after the script has looked.

Each amount is the largest measured peak plus 20 GB, rounded up to a multiple of 5: 310 GB† while
`cp_commit` ran, 268 GB† while `cp_prove` ran (file cache not counted). While `cp_commit` checks the
weights against `wsums.txt`, it holds up to six buffers of 10.74 GB each in host memory; how many
varies, also between runs on one machine. Its peak varied between runs†, from 299 GB to 310 GB.
`commit` and `prove` have not been run with only these amounts: in the measured runs† the script
counted 452 GB as available.

## Commit under your own limits

The published commitment's acceptance policy is exact match: a checked token is accepted only if it
is the integer model's top choice. `commit --limits FILE` generates a commitment under a clipped-gap
policy instead, which is meant for replies that a floating-point engine served ([acceptance
policies](../../docs/verifying.md#acceptance-policies)). The file contains exactly the three lines
`C N`, `theta N` and `cap N`, as `integrations/vllm/integer_model/examples/fit_limits.py` prints
them ([how to fit
them](../../integrations/vllm/integer_model/OPERATING.md#fit-the-limits-of-a-clipped-gap-policy)).

A work directory takes one commitment, and `all` takes no limits, so call the stages one by one, in
a work directory that has no commitment. If the lines under [Run it](#run-it) have built the
programs and fetched the weight files, the first line below chooses a new work directory next to the
old one. If not, run the first three of those lines, then the fourth three times, with `build`,
`download` and `checksums` in place of `all`, and skip the first line below. Set `L` to the path of
your limits file. In a new shell, set `M` to what the first line below printed: with the first `M`,
`compare` and `prove` would run on the published commitment.

```
M="${M:?first run the line with mktemp}-limits" && echo "M=$M"
examples/maverick/run.sh fit "${M:?}" --weights "${F:?}"
examples/maverick/run.sh commit "${M:?}" --weights "${F:?}" --limits "${L:?}"
examples/maverick/run.sh compare "${M:?}" --weights "${F:?}"
examples/maverick/run.sh prove "${M:?}" --weights "${F:?}"
```

- Before `cp_commit` starts, `commit` prints your limits, as `expected: C N, theta N, cap N`. If
  that line is missing, the call was run without `--limits`: interrupt it with Ctrl-C and run the
  line again. Once such a call has finished, the work directory has the published commitment and
  takes no other.
- The commitment ID covers the limits, so it is your own: nothing is published to compare it with,
  and nothing checks the fitted lines of `policy.txt`, which it also covers. The example request has
  no reply, so a `YES` here shows nothing about your limits. To prove replies under them, go on with
  [Prove replies that were served
  earlier](../../integrations/vllm/REFERENCE.md#prove-replies-that-were-served-earlier), with the
  five shell variables that `prove` prints.

## The 8-bit and the 16-bit files

The publisher of the 4-bit files offers the same model in wider formats, in the same repository, and
`run.sh` takes two of them. Add `--width 8` or `--width 16` after the weights directory and before
`--limits`, at every stage, and in the second line under [Run it](#run-it) write
`examples/maverick/8-bit/weights.sha256` or `examples/maverick/16-bit/weights.sha256`. Call `needs`
first, and go on to fetch hundreds of gigabytes only after it has ended well.

| | no option | `--width 8` | `--width 16` |
|---|---|---|---|
| the publisher's directory | `UD-Q4_K_XL` (4 bits) | `Q8_0` (8 bits) | `BF16` (16 bits) |
| weight files | 5, 232 GB | 9, 426 GB | 18, 801 GB |
| host memory that `commit` asks for | 330 GB | 545 GB | 955 GB |
| host memory that `prove` asks for | 290 GB | 505 GB | 915 GB |
| disk space in all | 245 GB | 440 GB | 815 GB |
| `weights.sha256`, `figures.txt`, `expected.txt` | in this directory | in `8-bit/` | in `16-bit/` |

Use a weights directory and a work directory of their own for each width, each weights directory
with its own copy of the calibration set. Every stage stops (exit code 2) in a work directory that a
call with another width created. But in a new work directory `download` without the option, into a
weights directory that holds no weight file, starts to fetch the 4-bit files: stop it with Ctrl-C,
and go on in two new directories. The script removes nothing of what it has fetched.

`prove` gives `cp_prove` other values for `-memory` ([the
option](../../docs/service.md#options-of-the-service)): `lean,tensor-rooms=6` at 8 bits, and
`lean,tensor-rooms=0,experts-by-layer,experts-rooms=1` at 16 bits.

**How far this has been run.** At each width one run† has made the commitment from the real files,
on one machine, by one call each of `needs`, `fit`, `commit` and `compare` (not with the script as
it is here). It gave the five values in that width's `expected.txt`. For three of them, the
commitment ID and the digests of `key.bin` and `leaf/0.bin`, it is the only run: nothing shows that
a second run, or another machine, reproduces them. Its `compare` had no stored commitment ID to
compare with and stopped there, so on the real files no call of `compare` with `--width` has
finished with exit code 0. `build`, `download`, `checksums`, `prove` and `all` have not run with
`--width` on the real files, and the files' address has not been tried against Hugging Face.

The peaks in the two `figures.txt`, and the times of `cp_commit` and `cp_prove`, come from one run†
at each width without this script, on a machine with 2,018 GB of host memory; each proved one
request, and `cp_verify` accepted the proof. `sums_time` and `calibrate_time` come from fits on
other machines, as their lines say. The two amounts of host memory lie about a tenth above the
peaks; neither run had as little.

## Where the values in `expected.txt` come from

Eight runs of this route on the real 4-bit weight files gave these five values, and in each
`cp_verify` accepted the proof (by the runs' records, which are not in this repository).

- **One run was on the released programs.** It went through all the stages of `run.sh all`, in their
  order, on one H100: the run reported that it ended with exit code 0, that the five values came out
  as in `expected.txt`, byte for byte, that the commitment ID was the stored one, and that
  `cp_verify` ended with `YES`. The build stage was skipped, because the programs were built
  beforehand from these sources; the download stage fetched nothing, because the five weight files
  lay there beforehand at their checksums; and the tools that the stage `needs` asks for were
  installed beforehand, so building, downloading and installing the tools were not exercised. The
  script that ran differs from this repository's in comments and messages.
- **Seven were not on the released programs.**† One went through the whole route and ended with exit
  code 0, with a calibration set made for it by `scales/make_calibration_set.py` and programs built
  beforehand; it did not follow this page in every point, and this page does not tell in which. The
  other six used forms of the script that differ from the one here. One of them went through all the
  stages of `run.sh all`, in their order, on one H100, and ended with exit code 0, with the build
  stage skipped and nothing fetched. Three called the stages one by one, without `build` and
  `download`, on three machines of one kind: one H100 with 80 GB, in a container with 24 CPU cores
  and a memory limit of 463 GB (row SO-6.5 of
  [docs/limits-all-rows.tsv](../../docs/limits-all-rows.tsv)). Two were one call of `all` each, in
  which all eight stages ran.
- **Fetching is not shown.** `download` has never fetched a file from Hugging Face: in every run the
  weight files were already in place. Two of the calibration set's three source files were copies
  from other addresses than the pinned ones, with the pinned checksums. In the two runs by `all` in
  which all eight stages ran† the Python packages did not come from the public index, and another
  tool had installed the CUDA toolkit.
- **Times above the stored ones.** In one of those two runs,† on a machine that it shared with other
  work, four of the seven times lay above their lines in `figures.txt`: those of the build, of
  `scales/calibrate.py`, of `cp_commit` and of its worst-case check. No peak of memory and no amount
  of disk space did. For the four times above their lines no cause is shown (row RES-36 of
  [docs/limits-all-rows.tsv](../../docs/limits-all-rows.tsv)). In the run on the released programs
  `scales/calibrate.py` took longer than its line in `figures.txt` says.

Equal values show that generating the commitment repeats on another machine of that kind. They do
not show that the commitment is right, and they say nothing about another GPU model, driver, tool
chain or set of CUDA libraries. For two of the runs† the platform is recorded: programs built with
nvcc 13.2.86 and g++ 13.3.0, which loaded cuBLAS and cuBLASLt 13.4.1.3, the CUDA runtime 13.2.86 and
the driver's libcuda 580.178.04 (`torch` has CUDA libraries of its own).

Each time and peak in `figures.txt`† is the largest of the three runs that called the stages one by
one, rounded up, except `sums_time`, the range over all three, and `build_time`, which comes from a
separate build on another machine. The file's line `taken_with`, which `needs` prints, names those
runs' programs in its own words. In these runs a separate process kept dropping the weight files'
pages from the operating system's file cache while the script read them.

† Not on the released programs: [which kinds of result these
are](../../docs/limits.md#on-other-programs).
