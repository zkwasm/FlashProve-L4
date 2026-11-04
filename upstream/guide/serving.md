# Serving proofs

FlashProve's programs run, with masking on, for [one full-size model
shape](../docs/limits.md#what-the-programs-can-serve): that of Llama 4 Maverick, with a window of
1,024 tokens.

Before you commit to weights that must stay secret, read [Weight hiding](weight-hiding.md). Masking
is built in and on by default. By design, clients and the public still see some things: the
[declared list](../docs/what-is-not-hidden.md). The main risks are a mask lost to a bug or a hardware
fault, and a use counter set back by restoring or copying the secret key directory.

`cp_commit` and `cp_prove` accept weight files with any 8- or 16-bit tensor without warning, and they
are outside what we judge of hiding, so it is for the provider to know how each tensor of its files is stored. Read [the limits on such
files](../docs/limits.md#files-of-8-or-16-bits) before you commit to one.

## What you need

The [only supported platform](../prover/docs/build.md#what-you-need) is x86-64 Linux 4.14 or later,
a CPU with SHA extensions and one NVIDIA H100 80 GB. The GPU needs a working NVIDIA driver
(tested† with driver 580.178.04) and, on a machine with several H100s, perhaps NVIDIA's Fabric Manager
service: [details](../docs/quickstart.md#3-gpu-a-small-model).
[Plan for 330 GB of free host memory](../docs/quickstart.md#4-llama-4-maverick-the-full-size-model).
[Performance](performance.md) has the times to expect.

One commitment serves on one GPU, a second GPU needs a further commitment, and each such GPU is one
that you never give to anyone else, which rules out a rented one. [Performance](performance.md) has
what proving costs per GPU.

[Each program needs a GPU to itself](../docs/operators-duties.md#keys-seeds-and-machines). Never
share or pass on the machine, GPU or disks. After every run, [free GPU
memory holds hidden values](../docs/limits.md#gpu-working-values), some unmasked, that no known reset
clears.

## Commit

[The duties](#the-providers-duties) bind from here on: read them first, and put `ISK` on a local
disk that nothing snapshots or backs up. Your own model needs a configuration fitted to its weights,
in the first five of [six steps](../scales/README.md#a-real-model). So it is outside what we judge
of hiding, as any real model is ([what our judgement
covers](weight-hiding.md#if-your-weights-must-stay-secret)).

[After the build](../docs/quickstart.md#3-gpu-a-small-model), as the service account, set these in
each new shell.

```sh
umask 077
B=/THE/ABSOLUTE/PATH/THAT/THE/BUILD/PRINTED/IN/ITS/LAST/LINE
WEIGHTS=/ABSOLUTE/PATH/OF/THE/WEIGHTS
CONFIG=/ABSOLUTE/PATH/OF/THE/CONFIGURATION
ISK=/ABSOLUTE/PATH/OF/A/NEW/SECRET/KEY/DIRECTORY
IVK=/ABSOLUTE/PATH/OF/A/NEW/PUBLIC/KEY/DIRECTORY
S=/ABSOLUTE/PATH/OF/A/NEW/SPOOL/DIRECTORY
```

`cp_commit` makes the secret and the public key directory and prints the commitment ID.

```sh
"${B:?}/cp_commit" -weights "${WEIGHTS:?}" -config "${CONFIG:?}" -M 4 -ivk "${IVK:?}" -isk "${ISK:?}"
```

`-M` is the number of leaf tables, masked encodings of the weights. Each batch, failed or not, uses
part of one. At full size and the default setting one leaf table serves 688,127 batches, by the
programs' own limit. Choose `-M` for the service's whole life: leaf tables cannot be added later, and
[no proof links a new commitment ID to the old
one](verify-a-proof.md#when-the-commitment-id-changes). The service ends when the first leaf table
is full, so count on fewer batches than the limit times `-M`.

A provider that serves for long enough, or on a second GPU, needs a further commitment to the same
weights ([how](../docs/operators-duties.md#a-new-commitment)). Above all, each commitment must sample
its own secret seed, on a machine that was not cloned or restored. Each must also take nothing from
another commitment and keep the same configuration files ([all the
conditions](../docs/limits.md#a-further-commitment)), so two commitments to the same weights under
different acceptance policies do not meet them.

[On success](../docs/operators-duties.md#creating-a-commitment) it exits with code 0 and its last
line says `can be handed out`. Copy the lines `commitment ID: ` and `cp_commit: the limit:` into a
log of your own on another machine. The rest of the output is as secret as the weights. [After any
other ending](../docs/faults.md#a-killed-or-failed-cp_commit), remove nothing and give the next call
two new names. What is left can hold the seed: keep it as secret as the weights.

After your first proof has verified, [publish only](../docs/operators-duties.md#what-you-publish)
the public key directory's files, the configuration and the commitment ID, where every client sees
the same ID, not only through your server. The public files hold [masked sums of the weights](../docs/limits.md#derived-pads) (the out-of-domain (OOD) answers) for good. They stay masked only while the
provider's [seed](../docs/glossary.md#seed) stays secret and ChaCha20 is unbroken.
Keep the seed secret even after retirement.

## Prove

[`cp_prove` watches the spool directory](../docs/service.md#the-files), shared with your front end,
your own program facing clients. Requests go into `in/` and responses into `out/`: `<id>.y` and
`<id>.proof`, or an empty `<id>.failed`. Use one per commitment, holding nothing else: `cp_prove`
knows no commitment ID, and takes any directory with a well-formed [mark
file](../docs/service.md#the-mark-file). [Every start
removes files there by name](../docs/service.md#what-a-start-does-to-the-spool-directory).

```sh
mkdir -m 0700 "${S:?}"
```

Keep `cp_prove`'s output, exit code and counts [from clients and hosted
tools](../docs/operators-duties.md#every-day): its lines [can depend on hidden
values](../docs/what-is-not-hidden.md#front-end-rules). This starts the service.

```sh
HOME=/ABSOLUTE/PATH/OF/AN/EXISTING/DIRECTORY TMPDIR=/ABSOLUTE/PATH/OF/AN/EXISTING/DIRECTORY \
  "${B:?}/cp_prove" -isk "${ISK:?}" -spool "${S:?}"
```

Set `HOME` and `TMPDIR` to two directories that only the service account can read or write, not
`/tmp` itself. The production build of `cp_prove` refuses to start unless each holds the absolute
path of an existing directory.

At full size, add
[`-batch 24 -in-step 6 -memory lean,tensor-rooms=8`](../integrations/vllm/REFERENCE.md#start-the-proving-service).
No test of hiding on the released programs has tried these values. [Performance](performance.md) has waits and rates. On an
80 GB card, stay at 24 requests per batch: 64 a batch is not to be relied on. On synthetic weights
all eight recorded attempts, six of them†, failed at a CUDA call with no proof, cause unknown. On
the public weights all three services at 64 ended normally.

`cp_prove`'s own self-checks on masks and keys, and the proof library's guards on the prover's own
steps (exit code 1, name `guard`), can stop a run before a response gets its final name. Beyond them
it runs no verifier on what it makes: once a batch's responses are staged they are named, and a
proof that `cp_verify`, the C++ verifier, would reject is in `out/` like any other. The serving
scripts that ship (the vLLM integration, `driver/measure_rate.py`, the example scripts) hand on or
show no proof that `cp_verify` has not accepted. The vLLM integration hands on no proof, and nothing
of its response, before `cp_verify` has passed every proof of its batch (narrow exceptions, none
open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)), against
the commitment ID in its config file; where one is rejected it hands on nothing of that batch,
submits no new request to that spool directory, so new replies get no proof, until a person lifts
that stop, and says in its log that a response did not pass. A front end of your own must do the
same: it checks every proof of a batch, with a `cp_verify` built from the same sources as your
`cp_prove`, which you keep where the service's account cannot write, before it gives out any
response of that batch, a failed one too. No program compares the two builds. A machine that has
just made a false proof may also have made proofs that lost a mask and still verify, and the proofs
of a batch share its keys and randomness, so a rejection counts against the whole batch. That check
[catches only what a verifier rejects](../docs/limits.md#a-lost-mask): [a proof that lost a mask can
still pass, as tests† showed](weight-hiding.md).

[Nothing warns you before the use limit](../docs/operators-duties.md#the-count). Daily, compare the
size in bytes of each leaf table's `count.bin` with the first number of `cp_commit: the limit:`.
Enter each reading in your log. If a count is missing, replaced or shorter: stop, restore nothing,
and serve again only under a new commitment.

```sh
stat -c '%s %n' "${ISK:?}"/leaf/*/count.bin
```

The file `anchor/count.bin` never changes while you serve: a change there also puts the commitment
in doubt.

[To stop](../docs/operators-duties.md#stopping-and-starting), create an empty `stop` file. Taken
batches finish, and it exits with code 0, last line `p258 end 0`.

```sh
: > "${S:?}/stop"
```

Any other exit is bad, with [two narrow exceptions](../docs/faults.md#always-first). After a bad
exit, or a response that `cp_verify` rejects, [send clients nothing more](../docs/faults.md#always-first),
and treat earlier responses as suspect too. [Only a person restarts](../docs/faults.md#before-you-serve-again), and not before the
cause is known, gone and harmless. Otherwise, [make a new
commitment](../docs/operators-duties.md#a-new-commitment), and not on that machine while the cause
is unknown. If `cp_commit` prints the old ID, the seed has repeated: do not serve it.

## The example front end and vLLM

[This repository has no complete front end](../docs/service.md#what-your-program-must-do), only the
example front end, `integrations/proof_middleware.py`. It runs [inside
`vllm serve`](../integrations/vllm/README.md#what-gets-a-proof) and gets proofs of greedy replies.
vLLM runs the [integer model](terms.md) with proofs under [exact match](terms.md), or the
[floating-point model](../integrations/vllm/README.md#two-ways-to-serve) with proofs under [clipped
gap](terms.md). Under clipped gap, a client that chooses the checked tokens [can measure over many
requests](../docs/limits.md#which-policy) how far their logits lie below the top choice's.

[The example front end](../integrations/LIMITS.md#what-the-integration-reveals) does not meet [duty 11](#the-providers-duties), so a server with it alone in front is outside [our judgement on weight hiding](weight-hiding.md#the-programs). With the plain command above, it [keeps serving proofs after a bad
exit](../integrations/LIMITS.md#what-a-provider-must-do-because-no-program-does-it). A break-in into
vLLM, which shares `cp_prove`'s account, reaches the weights. The example front end has run on a tiny
synthetic model† and, at full size, on a public 4-bit conversion of Llama 4 Maverick, with the same
16 synthetic prompts†. A [known
fault](../integrations/LIMITS.md#above-a-public-directory): the example
front end's server, started with `sudo`, can be led to send out files that only root may read.

## The provider's duties

[Breaking a duty can leak weights](../docs/limits.md#the-conditions). No program can check most of
them, and, a breach of duty 6 apart, every proof verifies either way. What is published cannot be
recalled. What a breach gives away has not been tried on the released programs.

1. Never copy, snapshot, back up, restore or roll back the secret key directory (`-isk`), even for a test or as a backup
   that you never serve from.
2. Keep the directory on a local disk of the one serving machine.
3. Never let two runs continue from one saved state of a machine or process, because their masks
   can repeat.
4. Let only one service account read and write
   the directory, and let no one else write to a directory above it.
5. Do not let a plain, unsalted hash or checksum of the weights leave the provider.
6. Let no response leave you before `cp_verify` has accepted every proof of its batch: never give
   anyone a response that it rejected or has not yet checked, because [it can hold unmasked
   values](../docs/faults.md#always-first). That holds of every file of a response, its token file
   too: nothing leaves that `cp_verify` was not given. Use a `cp_verify` built from the same sources
   as your `cp_prove`, and keep it where the service's account cannot write. `cp_prove` runs no
   verifier: it releases a batch's responses once all of them are staged. Of what ships, the vLLM
   integration hands on no proof, and nothing of its response, before `cp_verify` has passed every
   proof of its batch (narrow exceptions, none open to a client:
   [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`,
   which hands nothing to anybody, has the proofs checked once `cp_prove` has ended, and reports no
   rate unless `cp_verify` has accepted them all; each example script checks its one request's proof
   before it shows the answer; a front end of your own must do the like.
7. Serve only with the [production build](../docs/glossary.md#production-build). Never use
   [`-public-weights`](../docs/what-is-not-hidden.md#public-weights) for weights that must stay
   hidden: with it nothing is hidden, and no verifier can tell.
8. Stop serving proofs on a machine after exit code 9 (a failed self-check), a response that
   `cp_verify` rejects, any other exit that you did not cause, or requests that keep failing.
   Until you know the cause, do not restart there, and after any of the first three hand on
   nothing more that the machine made, responses that already have their final names among them.
9. Publish a new commitment, with a new commitment ID, before a leaf table is used up, on [the
   conditions for a further commitment](../docs/limits.md#a-further-commitment).
10. Never create or replace a [counter
    file](../docs/operators-duties.md#if-the-count-is-lost-or-in-doubt) (`count.bin`), or put a
    writable mount over just that file.
11. Put a [front end](../docs/what-is-not-hidden.md#front-end-rules) in front of `cp_prove` that
    gives each client only its own files and limits how many requests each may send or have waiting.
    Under clipped gap it must ask for proofs only of replies that you served, over ranges that it
    sets.
12. Give the check of [duty 6](#the-providers-duties) the commitment ID from [your own
    log](../docs/operators-duties.md#serve), never from the secret key directory. Do not let
    anything restart the service automatically.
13. If a disk or machine has left your hands, assume that the directory has been
    read.

Keeping them all does not by itself show that the weights are hidden ([Weight
hiding](weight-hiding.md)). [docs/operators-duties.md](../docs/operators-duties.md) has the steps and
the further requirements under these duties, which bind as much.

This page leaves [other limits](../docs/limits.md) out: see [Known limits](known-limits.md). Where
these pages or the programs' messages are known to be wrong: [docs/known-errors.md](../docs/known-errors.md).

† Not on the released programs: [which kinds of result these are](../docs/limits.md#on-other-programs).
