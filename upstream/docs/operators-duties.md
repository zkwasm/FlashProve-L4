# Operator's duties

This runbook is for a provider who runs `cp_commit` and `cp_prove` as a service on weights that must
stay hidden. `cp_commit` commits to a model's weights once. `cp_prove` then answers requests with
proofs, in batches, and `cp_verify` checks a proof.
[known-errors.md](known-errors.md) lists known errors in the programs' messages and in these pages:
read it before you serve.

## What you can serve

- **The model.** The programs support one full-size model shape: that of Llama 4 Maverick, with a
  window of 1,024 tokens. The weights are GGUF files; `cp_prove` takes fewer storage types than
  `cp_commit`. The routed experts' matrices must be Q4_K, Q8_0, F16 or BF16, or Q5_K for the routed
  down projection only. So a model whose routed matrices have another type, or rows not in fours, is committed whole, and `cp_prove` then rejects it at every start (by a reading of the code: row M1.11). On 8- or 16-bit tensors, see [step 7](#from-nothing-to-the-first-proof).
  The example integration with vLLM is for greedy replies (temperature 0): sampled replies get no
  proof.
- **The machines.** One `cp_prove` runs on one secret key directory, on one machine ([Stopping and
  starting](#stopping-and-starting), 4), and it uses one GPU. So one GPU serves a commitment, and a second GPU means a further commitment to the same
  weights ([A new commitment](#a-new-commitment)). Each such GPU is one that you never give to anyone else, which rules
  out a rented one ([Keys, seeds and machines](#keys-seeds-and-machines)).
- **Hiding.** The masking is built in and on by default. What is proved of it, and how far that
  reaches: [guide/weight-hiding.md](../guide/weight-hiding.md). [Our judgement on weight
  hiding](limits.md#the-verdict) does not reach a real model's weights, or the parameter file and
  scales fitted to them: only one of five tests of hiding has run on any.

Hiding also depends on you. No program can check most of the duties below, and a proof verifies
whether or not you keep to them. The example front end does not meet [duty
11](limits.md#the-conditions) ([The front end](#the-front-end)). A row ID such as R24 names a row of [the table of all
rows](limits-all-rows.tsv): what the programs check, what they miss, how far that was tested.

## Read this first

Breaking any of these rules can leak weights. In most cases no program will notice, and a leak
cannot be undone. What a breach gives away has not been tried on the released programs. The limits
page lists the [thirteen duties](limits.md#the-conditions) on which weight hiding depends. Rules 1
to 7 below are that list's duties 1 to 7, in its order, and rule 2 takes in duty 10. Duty 11 is [The
front end](#the-front-end), and duty 13 stands under [Backups and restored
copies](#backups-and-restored-copies). Where this page says "duty", it means that list's number.

1. **Never copy the secret key directory,** the one you pass with `-isk`. Do not snapshot it, back
   it up, move it, restore it or roll it back, and never serve from two copies. The directory holds
   a count of how much of each [leaf table](glossary.md#leaf-table) has been used. Every copy keeps
   its own count, and a restored copy has an older one, so a leaf table can serve past its
   [limit](glossary.md#limit) unnoticed. Exclude the directory from backup agents, volume and
   hypervisor snapshots, and image builds.
2. **Keep the directory on one machine's local disk, and never touch a `count.bin`.** Network,
   shared and in-memory file systems can lose a count. Each `count.bin` is a
   [counter](glossary.md#counter), and if its count is too low, a leaf table can serve past its
   limit unnoticed. So do not create, shorten, replace or edit one, or mount anything writable over
   it. A new or replaced counter starts too low. Do not edit, move or rename any other file in the
   secret key directory either. Do not run a program that opens a table's columns without counting
   them: columns opened without a count spend a table's random entries unseen.
3. **Never let the same saved state of a machine or a process run twice,** because both runs can
   draw the same random bytes. Do not start two machines from the same snapshot of a running
   machine, or from an image that contains the secret key directory. Do not resume a snapshot after
   the machine has kept running, and never restore a process checkpoint. An ordinary reboot does not
   repeat anything, and machines that boot from the same clean image are a separate case ([Booting
   from an image](#booting-from-an-image)). The rule applies to the machine that commits as well as
   the one that serves. It also protects clients' requests, because if a batch's random bytes
   repeat, a later client can test guesses at earlier clients' requests. We infer this from the
   design; it has not been tried on the released programs.
4. **Never let another account read or write the secret key directory.** The programs catch only
   some of this ([Shared machines](#shared-machines)). No program notices a count that has gone
   down, and clients can then get more than the leaf table's masks cover, in proofs that verify.
5. **Never let an unsalted hash or checksum of the weights leave your machines.** Keep them out of
   CI logs, registries, object-store metadata and integrity monitors. Treat everything that
   `cp_commit` prints, but for the commitment ID and the line `the limit:`, and what the tools in
   `scales/` print, in the same way: both can show something of the weights.
6. **Let no response leave you before `cp_verify` has accepted every proof of its batch:** never
   give anyone a response that it rejected or has not yet checked, because such a response can hold
   unmasked values. That holds of every file of a response, its token file too: nothing leaves that
   `cp_verify` was not given. Use a `cp_verify` built from the same sources as your `cp_prove`, and
   keep it where the service's account cannot write. `cp_prove` runs no verifier: it releases a
   batch's responses once all of them are staged. Of what ships, the vLLM integration hands on no
   proof, and nothing of its response, before `cp_verify` has passed every proof of its batch
   (narrow exceptions, none open to a client:
   [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`,
   which hands nothing to anybody, has the proofs checked once `cp_prove` has ended, and reports no
   rate unless `cp_verify` has accepted them all; each example script checks its one request's proof
   before it shows the answer; a front end of your own must do the like; the helper functions in
   `driver/spool.py` and `integrations/spool.py` read responses from `out/` and check no proof.
7. **Never serve with anything but the production build.** Never use a test build, a test option or
   a test seed on a directory you serve from. The programs reject only some of these mistakes.
   Never give the real seed as a test seed, because nothing rejects it. Never
   use `-public-weights` on weights that must stay hidden ([Public weights](#public-weights)).

Six more rules keep the service safe to operate, and they protect the weights too (three of them are
[duties 8, 9 and 12](limits.md#the-conditions)):

- **Give the service its own physical machine, GPU and account,** and never pass on the GPU or an
  unwiped disk.
- **Stop `cp_prove` with its `stop` file,** even before a reboot, and wait for it to exit ([Stopping
  and starting](#stopping-and-starting)). After `KILL` nothing is wiped.
- **Make sure that nothing restarts it automatically.** After a bad exit, a person decides, once the
  cause is known (duty 12). An account that can write to the secret key directory and restart the
  service can learn about the weights from how each restart ends (row LIM-37.2); no trial of that on
  the released programs is known (row H21).
- **Never change permissions, or move or remove anything, to get past an error.** The few exceptions
  are described where they apply, each with its conditions: in [step
  4](#from-nothing-to-the-first-proof), under [Declaring a model file's weights
  public](#declaring-a-model-files-weights-public) for a `.weights-are-public` file, and in [When
  something goes wrong](faults.md). If in doubt, leave the service down.
- **After a bad exit, or a response that your own `cp_verify` rejects, put the service on hold**
  (duty 8). Do not send clients anything more, and keep what it printed. Then open [When something
  goes wrong](faults.md#always-first), and do not change anything in the spool directory unless that
  document says to.
- **A commitment wears out** (duty 9). Make a new one before `cp_prove` reaches its limit ([Every
  day](#every-day), [A new commitment](#a-new-commitment)). How long a leaf
  table lasts is told in [step 9](#from-nothing-to-the-first-proof).

On synthetic weights you may let a coding agent run the set-up. On a real commitment the logs,
counts and exit codes stay private, so a person handles faults.

The weights are meant to be hidden from clients, even several acting together. They are not hidden
from anyone who can read or write to your machine's directories, memory or GPU ([the whole
list](limits.md#against-whom-the-claim-is-made)).

## Find your task

| When | Go to |
|---|---|
| before you commit | [Keys, seeds and machines](#keys-seeds-and-machines), [A record](#a-record), [Shared machines](#shared-machines), [Under `sudo`](#under-sudo) |
| setting up | [From nothing to the first proof](#from-nothing-to-the-first-proof), [The front end](#the-front-end) |
| while you serve | [Every day](#every-day), [Stopping and starting](#stopping-and-starting), [The mark file](#the-mark-file) |
| after an error, a rejection, a crash or a hang | [After a refusal or a fault](#after-a-refusal-or-a-fault), [If the count is lost or in doubt](#if-the-count-is-lost-or-in-doubt) |
| renewing, moving, a lost disk, retiring | [A new commitment](#a-new-commitment), [Backups and restored copies](#backups-and-restored-copies) |
| public weights; an audit on synthetic weights | [Beside the main road](#beside-the-main-road) |

## How the service works

`cp_commit` writes a [secret key directory](glossary.md#secret-key-directory) (`-isk`) and a [public
key directory](glossary.md#public-key-directory) (`-ivk`), and prints the [commitment
ID](glossary.md#commitment-id). Your front end writes each request into the [spool
directory](glossary.md#spool-directory), and `cp_prove` answers them in batches
([service](service.md)).

- `cp_prove` runs no verifier: it releases a batch's responses once all of them are staged. Whoever
  hands a response on checks it with `cp_verify` first. The vLLM integration that ships hands on no
  proof, and nothing of its response, before `cp_verify` has passed every proof of its batch (narrow
  exceptions, none open to a client:
  [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`
  reports no rate, and an example script shows no answer, before `cp_verify` has accepted the
  proofs; and a front end of your own must do the like ([step
  17](#from-nothing-to-the-first-proof)). That check catches only what a verifier rejects, and a
  proof that has lost a mask can still pass (see [limits](limits.md#a-lost-mask) and [the main
  risk](limits.md#the-main-risk)).
- Every batch uses up part of one leaf table. `cp_prove` records that use before the batch begins,
  so a batch that fails or is killed still counts. It stops at the limit because a leaf table's
  masks cover no more than that, and anything shown beyond the limit would leak weights. You cannot
  add to a commitment later.

The programs' messages use other terms: `folder` for directory, `refused` for rejected, `leaf` or
`stored copy` for a leaf table, and `isk` and `ivk` for the two key directories. Terms of this
runbook and of [When something goes wrong](faults.md) (the rest: [glossary](glossary.md)):

| Term | Meaning |
|---|---|
| your record | Your log for one commitment: every call with its output (of what `cp_commit` prints, only what [step 10](#from-nothing-to-the-first-proof) names), the commitment ID, the limit that `cp_commit` printed, your daily readings, and every stop with its cause. Keep it as private as the logs, on another machine, where no one who can write to the secret key directory can write. |
| in doubt | A count, a key or a directory may not be what your record says ([the list](#if-the-count-is-lost-or-in-doubt)). |
| hold | Stop serving, and do not send clients anything more, until the named person lifts the hold ([how](faults.md#always-first)). The example front end's hold stops proofs only: vLLM may go on serving replies without them ([its limits](../integrations/LIMITS.md#after-a-fault-nothing-stops-by-itself)). |
| named person | The one human, or their deputy, who decides whether to start again after a stop. If that is not you, stop and ask them. |
| service account | The account that runs `cp_commit`, `cp_prove` and the part of your front end that reads and writes the spool directory. |
| good exit, bad exit | A good exit of `cp_prove` is exit code 0 with the last line `p258 end 0`. Anything else is a bad exit. The two exceptions are a `TERM` that you sent yourself, with no line on the wipe before the last line ([Stopping and starting](#stopping-and-starting), 2) and a start that was rejected with a sentence that begins `refused (cp-isk): the folder named with -isk is in use` ([what then](faults.md#cp-isk)). |

## Keys, seeds and machines

Have all of these in place before the first commitment. Changing swap, core dumps or
`fs.protected_hardlinks` needs root, but the programs themselves do not. If a message from
`cp_commit` sent you here, see [A killed or failed
cp_commit](faults.md#a-killed-or-failed-cp_commit). Particulars: rows R26 and R27.

| You need | Why, and how |
|---|---|
| a physical machine, with its GPU and disks, on which no untrusted code runs | This is a condition of weight hiding. While `cp_commit` or `cp_prove` is running, nothing else may run on that machine, under any account or in another virtual machine, except the operating system, the hypervisor and anything else you trust with everything these programs hold. Anything that runs under the service account, including the front end, can read the seed and the weights. With the example front end that is the whole vLLM server ([one account](../integrations/LIMITS.md#what-the-integration-reveals)). |
| one H100 for each program that runs | Each program uses the first GPU it sees. With more than one GPU, set `CUDA_VISIBLE_DEVICES` to one GPU in front of each command, a different one for each program that runs. With one GPU, stop the service before you commit. Neither program starts if more than 2 GiB of the GPU's memory is in use (`gpu-in-use`), but each checks only once, when it starts. |
| enough host memory | At full size, `cp_commit` and `cp_prove` each need several hundred gigabytes of host memory ([the peak measured while serving](../guide/performance.md); row R27). How host memory behaves in a longer service is not known. Neither program checks first how much is available, so a run that exhausts memory can make the kernel kill other processes too. So do not run `cp_commit` at full size beside other people's work on one machine. `cp_commit` can need more than a service, so size the machine for the larger of the two. A full-size program is slow to exit after its last result, so wait for the process to end before you start the next program or read the free memory. |
| a local disk that keeps appended data, honours `fsync` and file locks, and outlives every restart, for both key directories | Rule 2. The count is the only thing that stops a leaf table at its limit, and a disk that drops synced writes can lose its last entries. No program checks the file system type. Do not use a network or shared file system, tmpfs, a write cache that loses its contents when the power fails, a container's own layer, virtiofs, 9p, or overlayfs mounted `volatile`. |
| encrypted disks, or disks that hold nothing else and that you will wipe or destroy | Removing a file does not overwrite its data. This matters for the secret key directory, the weights and the spool directory. |
| no swap, or encrypted swap; no core dumps that others can read | `cp_prove` locks only the memory that holds the seed, and `cp_commit` only its own copies of the seed and the keys (row R26). From reading the code, the copy of a key that its random generator works from is not locked. If the lock fails, `cp_commit` warns and carries on, and `cp_prove` carries on silently. Other secrets in memory can be swapped out. Programs other than `cp_prove` and `cp_commit` that hold secrets, such as the model server, need more than a core-size limit of 0 on a system that pipes core dumps to a program. These settings are machine-wide, so change them only on a machine you share with no one. |
| a random source that is sound and does not repeat | Rule 3. Two runs from one saved state get the same masks, and subtracting two values that share a mask cancels the mask. Every proof still verifies. Hibernating writes memory to disk, and live migration leaves copies, so avoid both. A commitment publishes out-of-domain (OOD) answers (masked sums of the weights): see also [A record](#a-record). Use a C library in which CVE-2025-0577 is fixed (row H4.10). |
| a service account, with `umask 077` | No other program then reads the secret key directory with that account's permissions. This account also owns the spool directory, which with its subdirectories must be closed to group and others (mode 0700). |
| no automatic restart | If a service manager runs the service, turn restarts off (systemd: `Restart=no`). If your supervisor has to restart on something, have it restart only after a good exit. Have it treat anything else as a bad exit, a `KILL` included, so a list of exit codes is not enough. |
| enough time to stop | A service manager's stop command and every shutdown send a signal, so use the `stop` file first. Let the service manager wait more than 600 seconds before it sends KILL (systemd: `TimeoutStopSec=`), because that is how long the wipe may take, and allow extra time ([how long an exit takes](../prover/docs/ends-in-short.md#how-long-an-exit-takes-at-full-size)). |
| a GPU that you never give to anyone else | After every run, even a clean one, free GPU memory still holds hidden values, some of them unmasked. No exit of `cp_prove` or `cp_commit` wipes all of it. We do not know which reset, if any, clears that memory, and that includes a reboot ([Free GPU memory](#free-gpu-memory)). |
| a front end of your own | An example is included, but it does not meet the duty on which weight hiding depends ([duty 11](limits.md#the-conditions), [what it leaves to you](service.md#the-example-front-end)). See [The front end](#the-front-end). |
| a named person and a deputy | Only they lift a hold. |

<a id="files-of-the-secret-key-directory"></a>**What the secret key directory holds.** All of it is
private ([seed](glossary.md#seed)).

| File | What it is | What happens if it leaks, is lost or is changed |
|---|---|---|
| `seed.bin` | the seed, 32 bytes | Anyone who has it can unmask the [anchor](glossary.md#anchor)'s revealed columns and test any guess at the weights (and more: [A record](#a-record)). Under rule 1 there is no copy of it, so losing it ends the commitment. |
| `leaf/K/fresh.bin` | 32 secret bytes of leaf table `K` | Anyone who has both this file and the seed can unmask that table's revealed columns and test any guess at the weights. |
| `leaf/K/count.bin`, `anchor/count.bin` | a count: the file's length | With a shorter or older file a table can serve past its limit unnoticed. Clients then get exact equations that the weights satisfy. |
| `isk.txt` | settings, and the weights' path | `isk.txt` says which leaf tables `cp_prove` chooses from, and no key protects it. Never edit it. |
| `hcnodes.bin`, `entry.bin`, `key.bin`, `config/` | stored tree nodes, copies of public files | If `hcnodes.bin` or a file in `config/` is damaged, the start or the batch fails. `cp_prove` does not read `key.bin`, and of `entry.bin` it looks only at the form of the first part: damage there is not noticed. No program publishes the file of stored nodes, but every opening of a table sends some of these hashes, or hashes computed from them. |
| `audit/` | present only after `-keep-for-the-audit` | Together with the seed, it reveals every mask of that batch or commitment. |

<a id="public-weights"></a>**Public weights.** `-public-weights FILE`, where `FILE` is a 32-byte
file, is for weights that are public, and for nothing else. With this option the weights are not
hidden, and no client can tell that it was used ([why](what-is-not-hidden.md#public-weights); how
far it is tested: row HW-649). `cp_commit` and `cp_prove` both print that the option is on, and
`isk.txt` ends in `seed public`.

**Your duties,** if your weights must stay hidden:

- Any directory holding weights that must stay hidden must be writable by your account only.
- Do not leave a file that begins with the SHA-256 of one of their model files, such as saved
  `sha256sum` output, anywhere another account can write. Once renamed, it can pass `cp_commit`'s
  checks.
- Do not let any script add the option or write `.weights-are-public` files. The one exception is
  the Maverick example's script, which adds the option for weights that are public.
- The 32-byte file is meant to be published, so never pass a secret commitment's `seed.bin`, or any
  other secret, as `FILE`.
- Never add or remove the line `seed public` in an `isk.txt`.

If you, the model file's owner or root create a `.weights-are-public` file for weights that must
stay hidden, `cp_commit` with the option accepts it, and so accepts those weights. `cp_commit` does
not check the permissions of the directory itself, so public weights can live in a shared directory.
It also does not notice that a `.weights-are-public` file is some other file that was renamed to
that name. So anyone who can write to the directory can rename a file into place, if it belongs to
one of those three accounts and begins with the right 64 hex digits. Nor does it notice a model file
that is modified in place while it runs. A mount or a container that maps several accounts to one,
or a file system that lets an account give its files away, can get another account's file accepted.
We have not tested that. What it does check: row R30.

**Declaring a model file's weights public.** The two commands for it, and what holds of proofs under
the option, stand at the end of this page: [Beside the main
road](#declaring-a-model-files-weights-public).

**A record.**<a id="a-record"></a><a id="a-record-on-a-models-files"></a> The programs' messages
point here under this title; "record" is their word for published [out-of-domain (OOD)
answers](glossary.md#record) (masked sums of the weights). The word "record" in the parameter file's
line `record 1` names these answers, not the record that you keep. `cp_commit` publishes one
weighted sum of each row's weights, masked by a [pad](glossary.md#pad) of its own, in files that
every client downloads, for good. The programs do not draw the pads fresh: each is derived from the
run's seed (for a leaf table: and from that table's kept bytes). Each sum is hidden only as long as
the seed stays secret and ChaCha20, the cipher that derives the pads from it, is not broken. What a
lost pad costs, and what else a pad depends on: rows HW-584 and H48.

**What that asks of you:**

- Whoever gets the seed (for a leaf table: with that table's kept bytes) can open every published
  answer, at any time, with no proof served. So the seed of a commitment has to stay secret for as
  long as the weights do, also after the commitment is no longer used, and in any backup that exists
  despite the duty never to copy the secret key directory. (A leaf table's kept bytes are its file
  `leaf/K/fresh.bin`.)
- Treat the prover's machine and its memory as you treat the weights. While a proof is made, the
  pads of the table in use lie in ordinary host memory that the wipe at exit does not reach
  ([details](../prover/docs/ends-in-short.md)), and with its published answer a pad gives one exact
  equation on a row's weights.
- Make the commitment on a machine that was not cloned or restored from a saved state. Never run
  `cp_commit` twice from one saved state of a machine, such as a cloned or restored image ([what
  counts](#booting-from-an-image)). If one seed serves two commitments over different weights, their
  public files alone give, for every row, one exact linear equation in the two sets of weights: in
  the weights themselves, if one of the two sets is known (row HW-584). Over the same weights, which
  is the case at a renewal, a repeated seed gives the same key file's table and the same ID again;
  where the leaf tables' own random bytes repeat too, the second commitment is the first again with
  its counts at zero, and its masks would then be used beyond the limit for which they were sized.
- Never carry a commitment's seed or kept bytes over to another commitment or parameter file: call
  `cp_commit` again.
- Before you hand out public files, know that OOD answers cannot be called back. Hand them out only
  after `cp_commit` has ended with exit code 0. Public files that you have not yet given to anyone
  are to be guarded too.
- Give no one a public key directory that you doubt and have not yet handed out, and [make a
  new commitment](#a-new-commitment). Grounds for doubt: a saved state used twice, a fault, a seed
  that got out, programs that are not the production build. If the example front end's
  proof server is running on it, stop the server for good. Moving or renaming the directory, or a
  hold, does not stop it, and a restart alone, the directory still under its name, serves it again
  ([details](../integrations/LIMITS.md#above-a-public-directory); row R25 says what of this was seen
  or tested).
- If someone already has the doubted public key directory, that cannot be undone, so tell the named
  person. Tell them too if a seed gets out, or one saved state has served two commitments.
- From reading the code, `cp_commit` and `cp_prove` take a parameter file that says `answers 1`, as
  every parameter file under `prover/examples/` does. By the same reading, they refuse a parameter
  file that says 2, 3 or 4 answers, which no written proof and no review covers, under
  `-public-weights` too (`parameter-range`). If a public key directory with 2, 3 or 4 OOD answers a
  row on real model files exists anyway, do not give it to anyone.

With `-public-weights` the OOD answers hide nothing.

<a id="booting-from-an-image"></a>**Booting from an image.** You can boot several machines from one
clean image without breaking rule 3, on two conditions. The image must not contain a secret key
directory, and the system must give its random number generator a new seed at every boot. No program
can check that, nothing notices a source that repeats, and we have not examined any operating system
for this. Booting means a full boot, so resuming a snapshot, restoring a checkpoint and cloning a
machine after it has booted do not count. A container shares its host's generator, so what counts is
how the host was started. An image can carry a saved seed file for the generator, and two machines
booted from it then start out alike unless each adds its own randomness at boot (row HW-191). That
is general knowledge, and we have not tested it.

## Shared machines

For secret weights, use a machine, a GPU and disks that you share with no one. The programs protect
their secrets from other users of the same machine only in part.

**The rule.** On a machine or shared volume that other accounts use, work only where no other
account can write. An account that owns, or can write to, any location the rule covers can redirect
the paths you type and replace the programs you run. The rule covers:

- the checkout and the programs;
- the file you pass with `-worst-case-table`;
- every directory you pass to a program or script (`-weights`, `-config`, `-isk`, `-ivk`, `-in`,
  `-out`, `-spool`, a work or output directory);
- the directory in which you stand when you start a script or a test;
- `TMPDIR`, or `/tmp` if `TMPDIR` is unset. A sticky directory that root owns, as `/tmp` normally
  is, is enough here. If you set `TMPDIR`, name a directory that exists.

For each of these, it applies to the directory itself, whatever is already in it, and every
directory above it. Weights that must stay hidden need more: keep their directory inaccessible to
every other account (mode 0700). `cp_commit` does not check this.

**What to do.** Pick a location that already meets the rule ([step
4](#from-nothing-to-the-first-proof) says how to check, and what to do if yours does not). Do not
change the permissions on a directory that you share with others.

**For a trial,** create a new private directory of your own (mode 0700) directly under a root-owned
sticky directory such as `/tmp`. `D=$(mktemp -d /tmp/work.XXXXXX)` creates one. That is enough if no
other account can write to any directory above the sticky one. A private directory does not help if
it sits under a non-sticky directory that others can write to.

**What the programs check, and what they miss** (rows R28, R31 and R24,
[limits](limits.md#path-names)).

- `cp_prove` checks the owner and permissions of every directory on the `-isk` and `-spool` paths at
  every start, and rejects symbolic links ([step 13](#from-nothing-to-the-first-proof)). From
  reading the code, `cp_commit` does the same at its start for what it creates (`-isk`, `-ivk`,
  `-worst-case-table`). Unlike `cp_prove`, it accepts a relative path for these three, and completes
  it from the current directory first. These checks have gaps, and attacks on the guards ran only
  through test drivers.
- Nothing checks the checkout, the programs, the `cp_verify` that `cp_commit` starts next to it,
  the paths given with `-config` and `-weights`, or a single batch's `-in`.
- No path check detects a rollback. If you or root put an older copy of the secret key directory in
  its place between two starts, `cp_prove` accepts it, lower counts and all. An account that can
  write to a directory above it can swap it for another of yours with a lower [count](#the-count).
- The example front end keeps each configured directory open, so restart the server after any change
  to the directories that it serves. At a start behind your own `sudo` it can be led to send out
  closed files of root's. This is [a known fault of that front end, which these sources
  have](../integrations/LIMITS.md#above-a-public-directory). Who can do it, how to keep that out,
  and what a server that is not root still risks: row R25.

<a id="under-sudo"></a>**Under `sudo`.** To let an account run these programs with `sudo` is to make
it root: grant it to nobody whom you would not give root itself. The programs need no root rights.
If you run them under `sudo` all the same, run the service and its front end both with `sudo`, or
neither, because everything that these programs make under `sudo` belongs to root. The secret key
directory must then belong to root, so the commitment has to be made with `sudo` too. If you serve
as root through `sudo`, the account that ran `sudo` controls where the path to the secret key
directory leads. So between two starts, that account can put a different private, root-owned
directory at that path, such as an older copy with lower counts. We have not run `cp_prove` end to
end under a real `sudo`. Tests of its routines for `-out` and `-clear` under one, through host
programs, gave two bad results: an empty directory owned by root was made inside a closed directory
of root's, and another directory than the one named was cleared (row R28; four other tools under
`sudo`: row R31).

<a id="free-gpu-memory"></a>**Free GPU memory.** What stays in GPU memory can depend on the weights
and on clients' requests, so it matters even with public weights. Two work buffers of the [GPU
engine](glossary.md#compute-engine) are not cleared. Another process or tenant on that GPU or
machine could read such memory later. A report of reads of uninitialised GPU memory in a served
batch, not on the released programs, is not fully explained ([T3](limits-further.md#t3)).

<a id="commands"></a>
## From nothing to the first proof

Do the steps in order, and from step 6 on work as the service account. Enter in your record what
each step says to.

1. **Open your record** on another machine.
2. **Try it first** on [synthetic weights](glossary.md#synthetic-weights), with a separate
   commitment. The [quick start](quickstart.md) shows how, and steps 6, 16 and 17 say how a trial
   differs. Changing the GPU, driver, CUDA version, compiler or sources makes it a new program, so
   after any such change stop the service and rerun the trial.
3. <a id="policy"></a>**Choose the acceptance policy:** exact match or clipped gap ([what each gives
   away](limits.md#which-policy), [served or failed](what-is-not-hidden.md#served-or-failed)). Under
   either, a client learns something just from whether its request is served: whether the integer
   model has an output on the window at all, that is, whether every intermediate value stays within
   its range (the programs test narrower intervals, and a request can fail for rarer causes too).
   `cp_commit`'s worst-case check (exit code 7) is meant to rule such a failure out, which is not
   proved. Under exact match, the client also learns whether every checked token is the
   integer model's top choice. Under clipped gap, the client also learns something about how far the
   checked tokens' scores fall below the best. So a client that chooses the checked tokens can
   measure the model's hidden scores over many requests, and no program limits requests. Under
   clipped gap, have your front end request proofs only for replies that you served yourself, and
   let it set the checked range. How to fit a
   clipped-gap policy's `C`, `theta` and `cap` to a model: [Fit the limits of a clipped-gap
   policy](../integrations/vllm/integer_model/OPERATING.md#fit-the-limits-of-a-clipped-gap-policy).

   **The setting.** A configuration's parameter file, `policy.txt`, fixes its setting. Each
   configuration under `prover/examples/` has the [default
   setting](glossary.md#default-setting), which counts proof of work. Every commitment publishes [OOD answers](#a-record). A commitment is bound to its parameter file, so another parameter file gives another
   commitment, with another ID. For weights that must stay hidden, keep to one set of configuration
   files for all commitments to the same weights ([why](limits.md#a-further-commitment)). Which
   parameter files and files of scales the statements on weight hiding reach:
   [limits.md](limits.md#weight-hiding).
4. **Choose the place** for the checkout, the programs, both key directories and the spool
   directory. It must meet the requirements in [Shared machines](#shared-machines) and [Keys, seeds
   and machines](#keys-seeds-and-machines). Check every directory above it with `namei -l`. If one
   shows `w` for group or others and has no sticky bit, as in a clone made under `umask 002`, choose
   another place, under a private directory you own. Or run `chmod go-w` on the offending directory
   now, before you commit, but only if you are not root, the directory is yours and no one else
   needs to write to it. That does not undo what another account put there while it could write, so
   under it use only names you create yourself afterwards.
   - Give both key directories a parent directory that only you can write to and that no server
     exposes to clients.
   - Keep the secret key directory separate from everything you publish.
   - Do not keep a commitment below `/tmp`, which may be in memory and may be emptied at a reboot.
5. **Build** the [production build](glossary.md#production-build) with [the build
   commands](quickstart.md#3-gpu-a-small-model) in the quick start. Only the production build
   accepts real model files. Build on a machine with a GPU, and look for the line that begins
   `build.sh: the compiler passed the check by value` (row H8.4.1).
   - Build only while no program from that directory is running. Do not use anything from the
     directory until both commands have exited with code 0. After a failed or interrupted build,
     build again.
   - The last line begins `THE PROGRAMS ARE BUILT` and names the directory that holds the three
     programs. If it has a directory `test_build` below it, never serve with anything from there
     (rule 7).
   - If the build prints a line that begins `build.sh: note:` and names the CUDA libraries, use a
     toolkit that belongs to root, or build as the account that runs the programs: anyone who can
     write to the toolkit's directory controls code that runs with access to the seed and the
     weights. If the note names `OUT`, see [Shared machines](#shared-machines) (row R32).
   - The programs do not compare the `sha256` line of their `<program>.build` file with the output
     of `sha256sum` on the program, so do that yourself. Enter its `sources` line in your record.
     Serve only with programs you built from the unmodified sources in this repository.
6. **Set the variables** in each new shell. Replace the words in capitals with your own values.

   ```sh
   umask 077
   B=/THE/ABSOLUTE/PATH/THAT/THE/BUILD/PRINTED/IN/ITS/LAST/LINE
   WEIGHTS=/ABSOLUTE/PATH/OF/THE/WEIGHTS
   CONFIG=/ABSOLUTE/PATH/OF/THE/CONFIGURATION
   ISK=/ABSOLUTE/PATH/OF/A/NEW/SECRET/KEY/DIRECTORY
   IVK=/ABSOLUTE/PATH/OF/A/NEW/PUBLIC/KEY/DIRECTORY
   S=/ABSOLUTE/PATH/OF/A/NEW/SPOOL/DIRECTORY
   ```

   `cp_commit` creates `$ISK` and `$IVK` itself, so do not create them. Use absolute paths with no
   symbolic links in them, under directories that only you and root can write to. `cp_commit` does
   not check `$B`, `$WEIGHTS` or `$CONFIG`. For a real commitment, put `$ISK` on a local disk, never
   on an in-memory file system, which `/tmp` often is. For your own model, `$CONFIG` holds a
   configuration fitted to its weights: [scales, "A real model"](../scales/README.md#a-real-model)
   makes one, in six steps. Its step 6 runs `cp_commit` itself, as step 10 here does, and this
   runbook's rules hold for that call too.

   For a trial on synthetic weights, set `WEIGHTS=made-up-blocks`, which is a keyword, not a
   directory, and skip steps 7 and 8. Set `CONFIG` to a directory that does not exist yet. After the
   block above, at the top of the checkout, copy the example configuration there:
   `cp -R prover/examples/target_small_f2_roomy "${CONFIG:?}"`. If that directory already exists,
   `cp -R` makes the copy inside it instead, replacing the files of an earlier such copy without
   asking, and `CONFIG` is then wrong.
7. **Store the weights** as real files, with no symbolic link and no second hard link. A downloaded
   snapshot or a deduplicated store holds links and is rejected, so copy the files themselves. Set
   `fs.protected_hardlinks = 1` before you store them, and keep it set. If a weight file that must
   stay hidden has a second hard link that you did not make, do not copy the file and delete the
   original, because the other name keeps the content on the disk. The file is safe again only when
   that other name is gone and yours is the only one left. Taking it away needs write permission on
   the directory that holds it, so you may need that directory's owner or root. Keep your own name
   for the file until then. Whoever could open the file through the other name may have read it. See
   [weight_sums](../weight_sums/README.md) for `wsums.txt`.

   `cp_commit` and `cp_prove` take weight files in which a tensor that the programs read is stored
   in 8 or 16 bits (Q8_0, F16, BF16), and a mostly 4-bit file with a single such tensor, and no
   printed line says that one was read. So it is for you to know how each tensor of your files is
   stored: no statement on hiding covers such files ([why](limits.md#files-of-8-or-16-bits)). The public 4-bit
   conversion that the examples use holds no such tensor (by two listings of the files' headers).
8. **Keep `wsums.txt` as secret as the weights.** Each hash in `wsums.txt` is a plain SHA-256 of one
   tensor's integers. Anyone who gets the file can test guesses at a tensor offline, as often as
   they like, and can tell whether two providers serve the same weights. Keep every other unsalted hash or checksum of the weights just as secret, such as a
   download's list of sums or a backup's content hashes.
9. **Choose the number of leaf tables,** `-M`, so that they last for the traffic you expect. Each
   batch is served from one leaf table, drawn at random, and the service ends as soon as one leaf
   table is full. A batch uses the same number of columns whether it holds one request or many, so
   count batches, not requests. At full size, at the default setting, one leaf table can serve
   688,127 batches, by the programs' own limit (calculated, by a planning call: the programs' own
   accounting, not a security level), and no run has come near
   it. No lower bound in time is known for reaching the limit: a batch that fails still counts, the
   plain command proves one request at a time, and `cp_prove` does not wait for a batch to fill. By
   our arithmetic, a service with one leaf table that never paused would use it up in about a year
   and a quarter with 1 request per batch (58.796 seconds a batch). That time is the mean of three short sessions on
   the public weights, with one token checked per proof: six batches each, with 1 request per batch
   ([performance](../guide/performance.md)).
   `"${B:?}/cp_verify" -plan accounting "${CONFIG:?}" 128` (128 is the target in bits) prints how many batches one leaf table serves, in the line that says `a stored copy bears`. This call exits with code 4 when it
   succeeds, and its last line is then `PLAN ONLY: no proof was checked`. `cp_commit` immediately
   rejects an `-M` that is too large (`cp-commit-anchor-budget`).
10. <a id="creating-a-commitment"></a>**Commit.** At full size the call takes a long time. Ctrl-C, a
    closed terminal or a dropped connection ends it and the run is lost, so start it in a session
    that outlives your connection, such as `tmux`. The `4` is an example.

    ```sh
    "${B:?}/cp_commit" -weights "${WEIGHTS:?}" -config "${CONFIG:?}" -M 4 -ivk "${IVK:?}" -isk "${ISK:?}"
    ```

    Wait for the exit code rather than polling for the directory. Keep the whole output, standard
    output and standard error, as secret as the weights, and out of your record. After a file that `cp_verify` did not pass, `cp_commit`'s message points to `cp_verify`'s last line and
    prints a length. Both come under [duty 5](limits.md#the-conditions): keep them inside the provider, because what `cp_commit`
    prints can show something of the weights. To report the fault
    to anyone else, follow [SECURITY.md](../SECURITY.md): say in your own words what you saw and where
    the two programs come from, and send no line, length, exit code or count of that run. The call
    has succeeded if the exit code is 0 and the last line begins `cp_commit: done.` and says
    `is complete and can be handed out`. With `-public-weights` it says
    `is complete, and hides nothing` instead. Partway through, the output repeats what step 9
    printed, down to `PLAN ONLY: no proof was checked`, which is not a failure. Enter only the call,
    its exit code and two things from the output:

    - **The commitment ID,** 64 hex digits, from the one line that begins `commitment ID: `. This
      line comes before the last line, so a call that fails later prints an ID too.
    - **The line `cp_commit: the limit:`.** It says how many columns a leaf table may show in all,
      how many its own link has used, how many each batch uses, and so how many batches each leaf
      table serves.

    An error that says a directory exists does not count as a failed call. Leave that directory
    alone and use a new name. If the message says a `cp_commit` is running there, wait for that call
    instead. For a name that ends in `.making`, which holds that call's seed, and after any other
    error, see [A killed or failed cp_commit](faults.md#a-killed-or-failed-cp_commit).
11. **Check the directories.** Run this ownership check before you publish or serve, and keep its
    output.

    ```sh
    find "${ISK:?}" -type l
    find "${ISK:?}" ! -user "$(id -un)"
    find "${ISK:?}" -perm /077
    find "${ISK:?}" -type f -links +1
    realpath "${ISK:?}"
    namei -l "${ISK:?}"
    ls -lnAR "${ISK:?}"
    ```

    Each `find` should print nothing, and `realpath` should print the path you set. `namei` should
    show no link and no owner other than you or root. It should show no `w` for group or others,
    unless the directory has the sticky bit or you know that no other account has been in that group
    since you committed. `getent group` does not show accounts that have the group as their primary
    group. Any other deviation puts the commitment [in doubt](#if-the-count-is-lost-or-in-doubt). Do
    not repair anything. Assume that any secret file another account could read has been read, so
    `chmod` does not repair it.
12. <a id="what-you-publish"></a>**Publish three things, and nothing else.** You may do steps 13 to
    17 first and publish only after your own check has ended in YES, so that you never publish a
    commitment that `cp_prove` rejects.
    - the files of the public key directory
    - the three configuration files the commitment was made with: `policy.txt`, `KL.bin` and
      `layout.txt` (see [limits](limits.md#what-every-client-learns) for what they show)
    - the commitment ID. Publish it where every client sees the same ID for this commitment, and not
      only through your front end. A client has to get it before it sends its prompt, over a channel
      that the server it is checking does not control ([what a client
      needs](verifying.md#what-you-need-and-where-it-must-come-from)).

    Publish copies, without file-system metadata. Do not add any files to the public key directory.
    Keep a copy of what you publish, because `cp_commit` cannot regenerate a file of an existing
    commitment. Where clients fetch the files, add each one with a single rename of a complete file,
    never onto an existing name. A half-written file makes a client reject an honest proof, and a
    rename onto an existing name replaces a published file.
13. **Make the spool directory.** It must not exist yet. Every directory above it, from `/` down,
    must belong to root or to you, must not be writable by group or others unless it has the sticky
    bit, and must not be a symbolic link. `cp_prove` checks that path at every start and rejects it
    otherwise, naming the directory. It also rejects a `-spool` that does not begin with `/`. Both
    rules hold for `-isk` and for a single batch's `-out` too. For `-out` we know that from reading
    the code.

    ```sh
    mkdir -m 0700 "${S:?}"
    ```

    At every start `cp_prove` removes files there by name
    ([service](service.md#what-a-start-does-to-the-spool-directory)), so do not use the directory
    for anything else. Use a separate spool directory for each commitment and each policy. Never
    restore it, copy it or share it between services: no program notices a restored or copied one.
    Do not use the top directory of a file system, because its `lost+found` keeps it from counting
    as empty.
14. **Put a front end in front of the prover.** See [The front end](#the-front-end).
15. <a id="serve"></a>**Serve.** `cp_prove` runs in the foreground until you stop it.
    - Ctrl-C, a closed terminal or a dropped connection causes a bad exit, so start it in a session
      that outlives your connection, such as `tmux`. Stop it with the `stop` file ([Stopping and
      starting](#stopping-and-starting)).
    - Set `ID`, which step 17's `cp_verify` takes (`cp_prove` takes none), from your own record of
      what you published or, if you have not published yet, of what `cp_commit` printed in step 10.
      Publish that same ID later. Never take it from the secret key directory, from a script that
      reads it there, or from a later message. If you have no record of it, stop and ask the named
      person.
    - Set `HOME` and `TMPDIR` to two directories that only the service account can read or write,
      not `/tmp` itself. Create both first, with mode 0700, for example with
      `mkdir -m 0700 FIRST_DIRECTORY SECOND_DIRECTORY`. Each path must be absolute, at most 400
      characters long, and made up only of letters, digits and `/ . _ -`. The production build of
      `cp_prove` keeps four environment variables: these two, `CUDA_VISIBLE_DEVICES` and
      `CUDA_DEVICE_ORDER`. It drops every other variable without a message. As root it also reads
      `SUDO_UID` once, for the paths named with `-spool`, `-isk`, `-out` and `-clear`, and then
      drops it.
    - The weights must be at the same path as when you committed.
    - Keep the command line that starts the service, for example a script or a unit file, where no
      one who can write to the secret key directory can change it.
    - The plain command below proves one request at a time. `-batch 24` puts at most 24 requests in
      a batch, and `-in-step 6` makes `cp_prove` prove six of them in one pass over the weights.
      Both options are 1 unless they are given ([options](service.md#options-of-the-service)). At
      full size, [the example front end's
      reference](../integrations/vllm/REFERENCE.md#start-the-proving-service) tells you to give
      `-memory lean,tensor-rooms=8`, another layout of GPU memory, with `-batch 24 -in-step 6`; none
      of that integration's runs had `cp_prove` at full size without `-memory`
      ([integrations/LIMITS.md](../integrations/LIMITS.md#what-has-never-been-run)). Most tests of
      hiding ran at the default values of these options; how other values bear on hiding has hardly
      been tested, and not at all on the released programs
      ([integrations/LIMITS.md](../integrations/LIMITS.md), "Not examined, or seen and not cured").
      Rates measured with these options and with 1 request per batch, and their conditions:
      [guide/performance.md](../guide/performance.md).
    - With the example front end, use [its own start
      command](../integrations/vllm/REFERENCE.md#start-the-proving-service), which puts the policy
      on hold after any exit code other than 0. The plain command below does not. The example front
      end does not meet [duty 11](limits.md#the-conditions).

    ```sh
    ID=THE-64-HEX-DIGITS-FROM-YOUR-RECORD
    HOME=/ABSOLUTE/PATH/OF/AN/EXISTING/DIRECTORY TMPDIR=/ABSOLUTE/PATH/OF/AN/EXISTING/DIRECTORY \
      "${B:?}/cp_prove" -isk "${ISK:?}" -spool "${S:?}"
    ```

    Keep its output ([what it prints](service.md)). After its start-up lines it prints nothing until
    the first batch is released. `cp_prove` exits with code 1 if it cannot write one of its lines,
    so send them to a file on a disk with room, or to a reader that outlives `cp_prove`.
16. **Submit a request** from a second shell with the same variables set, using [the
    command](service.md#a-request-file) in the service document, typed at the top of the checkout.
    For a trial on synthetic weights, run
    `python3 -I -B prover/tools/make_request.py "${CONFIG:?}/policy.txt" REQUEST_FILE` from the top
    of the checkout, with `REQUEST_FILE` in your own directory. The response appears in `out/` as
    `<id>.y` and `<id>.proof`, or as the empty file `<id>.failed`.
17. **Check every response yourself,** batch by batch ([verifying](verifying.md); [The front
    end](#the-front-end)).
    - Read it once from `out/` into your own directory. Check those bytes with a `cp_verify` that
      the service account cannot write to, that is, a copy owned by another account, and pass on
      those same bytes.
    - Use the response's `.y` as the token file, here as for a real client. In a trial on synthetic
      weights you have no served reply to compare it with, so the call shows only that the files are
      consistent: for a real client, also compare that file with the reply that you served. A client
      does it the other way round: it writes the token file itself, from the reply that it received
      ([verifying](verifying.md#the-request-and-the-token-file)). The trial's call, typed at the top
      of the checkout, is
      `bin/cp_verify "${CONFIG:?}" "${IVK:?}" REQUEST_FILE TOKEN_FILE PROOF_FILE -id "${ID:?}"`.
      `REQUEST_FILE` is the file that step 16 wrote, and the other two are your copies of `<id>.y`
      and `<id>.proof`. A trial under a single account has no copy of `cp_verify` that another
      account owns: use your own there, but never for a real client.
    - If the last line is `NOT-A-VERIFICATION`, nothing has been accepted. Fix the call and check
      again ([what causes the line](verifying.md#exit-codes)). If the call was right and you get the
      same line, put the service on hold. If you passed `-no-proof-of-work` on purpose, do not drop
      it to get a `YES`: the line is then your answer. It needs no hold as long as
      `no-proof-of-work` is the only cause that the lines after `---- the call ----` name ([Always
      first](faults.md#always-first)).
    - If your own check does not accept a response that `cp_prove` released, whether it rejects it
      or ends `NOT CHECKED`, put the service on hold ([Always first](faults.md#always-first)), and
      keep that copy as secret as the secret key directory. It can contain unmasked hidden values,
      so never attach it to a bug report. Do not pass on what `cp_verify` printed either (row H49).
      No tool erases the copy or that output for you.

## The front end

Only the front end hands files to clients, and no verifier can tell whether one is there. Its basic steps, with the limits on waiting requests and
unfetched responses, are in the service document, under [What your program must
do](service.md#what-your-program-must-do). Those limits matter here too, because a client that fills
the queue uses up a leaf table. The front end must also do the following. The five rules that spell out [duty 11](limits.md#the-conditions) stand in [what a front end must do](what-is-not-hidden.md#front-end-rules). The service
document's nine steps and the points here take them in, with one exception: rule 2 also limits how
many requests each client may send, which neither list caps.

- **Keep clients apart.** Record which client owns each request ID, and give that ID's response to
  that client only. The `.tag` file is for you and never goes to a client. Do not show anyone a
  listing of `out/`, a file's timestamps, or anything about an ID that is not theirs.
- **Check every proof yourself.** Before you give out any response of a batch, a `failed` too, check
  every proof of that batch with a `cp_verify` that you built and keep where the service's account
  cannot write. No other check stands between a proof and a client: `cp_prove` runs no verifier.
  Give `cp_verify` the response's own token file (`<id>.y`), never a value of your own: where the
  request's range reaches the end of the window, the file holds the model's next token, and the
  proof covers it; where the range ends earlier, the file must be four zero bytes, and `cp_verify`,
  given the file, rejects anything else (read, not tried) ([the files of a spool
  directory](service.md#the-files)). If one proof of a batch is not accepted, give out nothing of
  that batch, take no more requests, and put the service on hold ([Always
  first](faults.md#always-first)). Treat no verdict as a no: some of the checks of `cp_verify` that
  end this way protect hiding. Until such a check ends, the memory of the process that makes it
  holds the request, the reply and a proof that has not yet passed, and so does the place from which
  it reads them. So would a core dump of that process ([Keys, seeds and
  machines](#keys-seeds-and-machines); of the example front end's check:
  [integrations/LIMITS.md](../integrations/LIMITS.md#core-dumps)). What `cp_verify` prints on such a
  proof falls under rule 1 of [what a front end must do](what-is-not-hidden.md#front-end-rules). The
  rule goes by the batch because a machine that has just made a false proof may also be making
  proofs that have lost a mask and still verify, and because the proofs of one batch share its keys
  and random values (reasoned, not shown).
- **Stay out of `.tmp/`.** Do not read or list anything there. What you send to a client, and when
  you answer, must not depend on what is staged there.
- **Submit the token IDs that the model really worked on.** Take a request's prompt and reply from
  the inference engine itself: the IDs it was given, with the chat template's tokens and whatever
  your front end added, and the IDs it produced. A proof covers token IDs, not text, and tokenizing
  the text again can give other IDs. Neither the commitment ID nor a proof covers the tokenizer, the
  chat template or the end-token IDs (row R33).
- **Do not add your own nonce, count or time to a request's tokens.** The rule does not apply to the
  32 random bytes in the spool's file format, which the submitting command adds.
- **Do not submit a waiting request twice.** Requests still waiting in `in/` when the service exits
  stay there for the next process. A request proved twice is counted twice, so resubmit waiting
  requests only if you give up that spool directory.
- **Record each request's final answer when you deliver it, and never change it.** A newly started
  `cp_prove` withdraws the response of every request it finds in `work/` and answers "failed", even
  if that response was already delivered.
- **Tell a client "failed" only where `cp_prove` wrote `<id>.failed`**, or where `cp_verify`, in
  your check of a batch, rejected one of that batch's proofs or gave no verdict on it: then, once
  the hold is lifted, tell it to every request of the batch, in the same form as `cp_prove`'s own,
  so that the answer by itself does not show which of the two it was. A "failed" that you decide on
  yourself, for instance to shed load, can depend on load and time, and so on hidden values.
- **Send real clients' requests only to a production build,** never to a test build. In a test build
  a batch's random bytes can come from a file, it can list which random values it samples, and
  nothing checks whose requests it receives.
- **If your policy is clipped gap,** follow the rule in [Choose the acceptance policy](#policy).
- **During a hold,** tell clients only that the service is down. Do not give them a cause, an exit
  code or a count. Do not send any response or `.failed` file that was made before the hold. The
  example front end tells them more than that: its answer says that the provider has put this
  policy's proofs on hold ([its
  limits](../integrations/LIMITS.md#after-a-fault-nothing-stops-by-itself)).

## Every day

1. <a id="the-count"></a>**Read each leaf table's count** and enter it in your record.

   ```sh
   stat -c '%s %n' "${ISK:?}"/leaf/*/count.bin
   ```

   The size of `isk/leaf/<k>/count.bin` in bytes is the number of columns shown. Compare it with the first number in `cp_commit`'s line `the limit:`. The
   third number in that line is the number of columns a batch uses. Batches left = columns left ÷
   that number. The count does not start at 0, because `cp_commit` itself uses some columns when it
   creates the leaf table. Without your record, nobody can tell that a counter has become shorter.
   `cp_prove` does not print anything about the count. When one leaf table is full, it exits with
   the last line `p258 end 1 cp-copy-spent`.
2. **Look for `<id>.failed` files in `out/`.** The log does not show them, because the line for a
   batch is the same whether its answers are good or all "failed".
   - You may resubmit a "failed" request once, as a new file with a new request ID (no statement on
     hiding covers a request that is sent again: row PS-32 in [the table of all
     rows](limits-all-rows.tsv)). Do not retry in a loop, because every batch uses up part of a leaf
     table.
   - A "failed" answer looks the same whatever caused it. One cause is that the program failed a
     check of its own work, which can point to a fault in the GPU or in the program. If "failed" answers keep coming on one machine,
     stop the service with the `stop` file and treat it as a bad exit.

These rules apply at all times.

- **A count that is missing, replaced, or shorter than at your last reading puts the commitment [in
  doubt](#if-the-count-is-lost-or-in-doubt).** The file `anchor/count.bin`, which the command above does not read, never changes while you serve, so a change there also puts the commitment in doubt.
- **Begin [a new commitment](#a-new-commitment) in time.** Start while the fullest leaf table has
  room for the batches you will serve until the new commitment is made, published and in your
  clients' hands.
- **Keep these private:** the programs' logs, exit codes and counts, the line for each batch, the
  file named with `-worst-case-table`, and what the tools in `scales/` print. Nothing a client can
  see may depend on them, except that the service stops where this document says it must. Keep them
  away from any tool that sends what it reads elsewhere, such as a hosted monitoring service or a
  coding agent.
- **Watch the time since responses were last released,** because neither program limits how long a
  batch or a run takes ([A program that hangs](faults.md#a-program-that-hangs)).
- **Do not run any other program on the secret key directory.** Anything that reads or hashes the
  files reads the seed too. The commands printed in this document are allowed, because they read
  names, sizes, owners and modes, not the files' contents. To test, rehearse or measure, use a
  separate commitment on synthetic weights or a public model's files, not this directory or a copy
  of it.

## Stopping and starting

1. **To stop,** create the `stop` file in the spool directory.

   ```sh
   : > "${S:?}/stop"
   ```

   `cp_prove` then stops taking batches, finishes the ones it has already taken, releases their
   responses, removes the file, and exits with code 0 and the last line `p258 end 0`. Also do this
   before a planned reboot, a driver update or an automatic upgrade. After a change of GPU, driver,
   CUDA version, compiler or sources, rerun the trial before you serve ([step
   2](#from-nothing-to-the-first-proof)). If you build the programs again, install the new
   `cp_verify` too, for your own check of the responses: no program checks that the two are built
   from the same sources, and `cp_prove`'s build record names no `cp_verify`. With a stale
   `cp_verify` every proof can fail that check after its batch has been counted on its leaf table,
   so each such batch uses up some of the table's limit. Programs built from other sources can
   reject the secret key directory (`ranges-not-guarded`): then make [a new
   commitment](#a-new-commitment), and do not edit `isk.txt`. If you change `cp_prove`'s source
   yourself, raise the number `the prover's revision` in the same change
   ([format/COIN-KEYS.md](format/COIN-KEYS.md), "The prover's revision"): nothing checks that you
   did, and row H4.5.1 tells what follows where it is not raised. Step 5 of [From nothing to the
   first proof](#from-nothing-to-the-first-proof) tells you to serve only with programs built from
   the unmodified sources. Leaving a `stop` file in place does not keep the service down, because
   the next start removes it and serves.
2. **`TERM` also shuts it down cleanly,** including the wipe it does before it exits. No test kills
   or signals `cp_prove` on purpose at the default setting
   ([service](service.md#how-the-service-ends)). But with a signal you lose the batch in progress
   (with `-overlap` or `-overlap-two`: the batches), and it is still counted.
   - After a `TERM` that you sent yourself, exit code 1 with the last line `p258 end 1 asked-to-end`
     is a clean exit, not a bad one, so start again. This does not apply if you sent it because the
     program hung ([A program that hangs](faults.md#a-program-that-hangs)), or if a line that begins
     `p258 not every step of the wipe ran` came before the last line ([what to do
     then](faults.md#what-you-see-and-what-to-do)).
   - Follow the steps for a bad exit if the exit code is 9, the last line is different or
     missing, or you did not send the signal.
   - `cp_prove` acts on a signal at its main thread's next check, and nothing limits the wait for
     that check. If it does not exit after `TERM`, follow [A program that
     hangs](faults.md#a-program-that-hangs).
   - A service manager's stop, a container's stop, a reboot, Ctrl-C, a closed terminal and a dropped
     connection all send a signal. So does a limit on CPU time or file size, so do not set one for
     the service.
3. **To start** after a good exit, follow [Serve](#serve). After a bad exit, start it only by hand,
   and only once the hold is lifted ([Before you serve again](faults.md#before-you-serve-again)).
4. **Run one `cp_prove` on one secret key directory, on one machine.** From reading the code,
   `cp_prove` locks the directory when it starts, and a second one under the same running system
   does not start ([`refused (cp-isk)`](faults.md#cp-isk)). It cannot see a `cp_prove` on a copy of
   the directory, or one on another machine that shares it, unless the file system applies the lock
   between machines, which a start does not show ([in use](faults.md#cp-isk)). Nor does it see every
   other route to the directory (a container's layer, a virtual machine, a second mount). Where locks
   are not honoured, or the directory is shared, two processes or two machines can serve from one
   table and each count for itself. In all these cases nothing orders what the two release, and we
   make no statement on hiding. So rule all of those out. Never copy the directory to get past the lock: a copy counts for itself. The
   lock does not hold off the audits or `cp_prove -clear`.

## The mark file

`mark.txt`, in a spool directory, says that the directory is a spool directory ([its
format](service.md#the-mark-file)). It is there so that a command that is given a directory of
another kind stops before it changes anything. `cp_prove` knows no commitment ID: it writes 64 zeros
in the file's third line and of that line in a file that it finds it asks only for 64 hex digits. So
the file does not keep two commitments' spool directories apart for `cp_prove`; the scripts in
`integrations/` do compare the line with the ID that they are given, and take 64 zeros as any
commitment's. The file guards against a mistake, not against someone who can write to the directory,
and it does not catch every mistake (rows R13.1.1 and R18.1). Never take a commitment ID from it,
and never write, copy, edit or remove it by hand.

- **If `cp_prove` rejects the directory for its `mark.txt`,** the line that begins `refused
  (cp-spool)` names that file, and nothing that was in the directory was touched. Other rejections
  begin with the same words, so go by what the sentence says. Usually `-spool` names a directory
  that is not a spool directory, so give `-spool` this commitment's directory. Whatever the file
  says, do not change it.
- **A directory that is not empty and has no `mark.txt`.** `cp_prove` does not accept it, so name
  this commitment's directory, or a new, empty one. Do not remove anything to make a directory fit
  ([what then](faults.md#clear), under "A directory that it rejects").
- **A file whose name begins with `mark.txt.making.`** is left by a program that was killed while it
  wrote the mark. In a spool directory every program ignores it, and none removes it. Leave it.

## Renewing before the number

Some program messages point here. The heading's "number" is the number of batches that a leaf table
can serve, its limit. You cannot extend a commitment, so make [a new one](#a-new-commitment) before
any leaf table fills up (see [Every day](#every-day)).

## A new commitment

A new commitment is the only way to renew. It means running `cp_commit` again, publishing a new
commitment ID, and having every client download new key and leaf files. It is also what to do if
`cp_prove` cannot lock the secret key directory where it is
([`refused (cp-isk)`](faults.md#cp-isk)): do not copy or move that directory. No call adds or retires a leaf table: `cp_commit` rejects one (`cp-commit-first-form-alone`), even if a printed message advises it.

From reading the code, `cp_commit` draws a new seed in every call without `-public-weights`,
so all the new commitment's masks are new; no run shows where a seed's 32 bytes come from, or that
they are drawn fresh at every call. One seed that serves two
commitments can leak, but an ordinary renewal cannot cause that. Only a copied directory, a repeated
machine state, a random source that repeats or a seed carried by hand can (rules 1 and 3 under "Read
this first"). A further commitment to the same weights, under the same configuration files, lies
inside the judgement on weight hiding, on thin ground and on [conditions of its
own](limits.md#a-further-commitment); if the old one is [in
doubt](#if-the-count-is-lost-or-in-doubt), for these weights as a whole the judgement does not speak
([R21](limits-further.md#r21)). A new commitment does not undo anything that an overused leaf table
has already revealed.

1. `cp_commit` needs a GPU to itself. If you have only one GPU, stop the service first ([Stopping
   and starting](#stopping-and-starting)) and plan for downtime while `cp_commit` runs.
2. Open a new shell and start a new record. Follow [From nothing to the first
   proof](#from-nothing-to-the-first-proof) from step 6, with two new key directories and a new
   spool directory. Do not reuse anything from the old secret key directory, and keep the
   configuration files unchanged.
3. Publish the new public key directory, its configuration files and the new commitment ID (a new
   commitment has a new ID: if `cp_commit` prints the old one, the seed has repeated, as after a
   restored machine state: do not serve it), as you did the first.
4. Retire the old commitment ([Backups and restored copies](#backups-and-restored-copies), step 5), unless it goes on serving beside the new one, as on a second GPU; one that is in doubt is always retired.
   Before you give up the old spool directory, run `cp_prove -clear` on it, and only on it
   ([how](faults.md#clear)). No program marks a commitment as given up, so after a lost or doubtful
   count make sure the old directory never serves again.
5. If you serve under several commitments, never let anything that must stay hidden decide which commitment a batch uses. One exception: the commitment may be
   picked by load or queue length. That can depend on hidden values, and the pick gives away something of time and load, [declared](what-is-not-hidden.md) items 3 and 8.

<a id="after-a-rejection-or-a-fault"></a>
## After a refusal or a fault

The programs' messages refer to this section by its heading, but the instructions themselves are in
a separate document, [When something goes wrong](faults.md). If a program's message disagrees with
these two documents, follow whichever is stricter. That includes removal, so do not remove anything
unless the message and both documents allow it. If a message says that `seed.bin` has permissions
for group or others, treat the seed as read. The commitment is then [in
doubt](#if-the-count-is-lost-or-in-doubt).

- <a id="always-first"></a>[Always first](faults.md#always-first): the first steps after any bad
  exit
- <a id="what-you-see-and-what-to-do"></a>[What you see, and what to
  do](faults.md#what-you-see-and-what-to-do): every exit code and last line
- <a id="a-killed-or-failed-cp_commit"></a>[A killed or failed
  cp_commit](faults.md#a-killed-or-failed-cp_commit): what it leaves behind
- <a id="a-program-that-hangs"></a>[A program that hangs](faults.md#a-program-that-hangs): how to
  end it
- <a id="before-you-serve-again"></a>[Before you serve again](faults.md#before-you-serve-again):
  when a restart is allowed

## If the count is lost or in doubt

1. Do not queue anything, and put the service on hold.
2. Stop every `cp_prove` on the directory with the `stop` file of its spool directory, and wait
   until each has exited.
3. Do not run `cp_commit`, `cp_prove` or any other program on the directory. Only the ownership
   check and the daily reading are allowed.
4. Do not restore anything, and do not remove anything.
5. Write what you saw in your record.
6. Find the kind of doubt in the list below, because the remedy differs by kind. An item can have
   two kinds, and then both apply.
   - **Directory:** serve again only under [a new commitment](#a-new-commitment). That stops further
     harm but does not undo anything.
   - **Machine or build:** keep the hold in place ([Before you serve
     again](faults.md#before-you-serve-again)).
   - **Left your control:** there is no remedy, and it cannot be undone. Tell the named person and
     the owner of the weights exactly what got out.

**What puts a commitment in doubt,** grouped by kind:

- **Directory**
  - A `count.bin` is missing, shorter than it was, replaced, or changed by someone unknown. That
    holds after a power cut too: do not judge how much may be missing.
  - A copied, restored, rolled-back or duplicated secret key directory may have served, or the
    directory was served from two machines.
  - Another account could read a secret file, or could write to a directory or a directory above it.
  - A test build has run on the directory.
  - `cp_commit` or `cp_prove` ran on two machines started from the same snapshot, or from the same
    image without fresh random bytes at each start.
  - A last line names a fault in the secret key directory, and no one can say who changed what and
    when.
  - Your record is lost, or is missing the commitment ID or the last reading. A missing limit alone
    is no ground for doubt. From reading the code, `cp_prove` takes the limit from the configuration
    and stops at it whatever your record says. Its numbers can be had again from the configuration
    files that the commitment was made with.
  - A power cut, crash or file-system repair happened while the directory was in use, every
    `count.bin` is as long as your last reading or longer, and you do not know that the disk keeps
    synced writes.
  - The directory has been on a network file system, tmpfs, a container's layer or a disk with a
    volatile write cache.
  - A snapshot or checkpoint of a running machine or process was resumed after the original had run
    on, or was resumed more than once.
  - The system's random source is known or suspected to be faulty.
  - Something other than you copied the directory, such as a hypervisor snapshot, a backup agent, a
    deduplicating store or a container image.
  - Any other program has read or hashed the directory.
  - `seed.bin`, a `fresh.bin` or a `count.bin` has a second hard link or a different owner.
  - A line of `isk.txt` has changed.
  - A killed `cp_commit` left files behind, and another account or a backup could read them.
    `<isk>.making` holds the seed.
  - Audit files were still in `isk/audit/` while `cp_prove` ran, or when any part of the commitment
    left the machine.
  - An audit ended NO or NOT WHOLE, found nothing kept before a first YES, or stopped after it had
    touched a kept file. See [The audit with the key](#the-audit-with-the-key) for the other ways an
    audit can end.
  - Someone copied tables or secrets by hand into a commitment with a different configuration,
    layout or set of weights.
- **Machine or build**
  - A program's build record (`<program>.build`) is missing or does not match the program.
  - The `cp_verify` that checks your responses was not built from the same sources as your
    `cp_prove`.
  - `cp_prove` ended in a way that you cannot explain. Examples are a bare `p258 end 1` that you did
    not cause and repeated `.failed` files.
  - A proof that `cp_prove` released was not accepted by your own check, and the call was right
    ([Always first](faults.md#always-first)).
- **Left your control**
  - Something secret has left your control. Examples are an unsalted hash of the weights
    (`wsums.txt`), the `-worst-case-table` file, and a rejected or unchecked response.
- **Left your control, and Directory**
  - A core file, an unwiped disk, or a GPU or machine handed to someone else has left your control.
    Any of these can hold the seed, so treat the directory as read as well.

If you are not sure whether something is on this list, assume that it is.

## Backups and restored copies

1. Do not back up the directory, even if you never serve from the backup. A backup is a second place
   where the seed can be read. A restored copy also resumes from its old count. If the directory is
   destroyed and nobody else has a copy, nothing leaks, and only [a new
   commitment](#a-new-commitment) can serve again. If a disk or machine left your hands, treat the
   directory as read.
2. **Keep a failed machine's disk switched off.** If it holds nothing else, wipe or destroy it
   before it boots again or leaves your hands. Otherwise its encryption has to protect what is on
   it.
3. **Do not start anything on a restored, copied or rolled-back secret key directory.** The
   commitment is [in doubt](#if-the-count-is-lost-or-in-doubt). The same applies to a directory
   baked into a container image, fetched by a startup script, or taken from a cloned snapshot.
   `cp_prove`'s lock on the directory sees none of these copies.
4. **A planned move** to another machine is not supported, so make [a new
   commitment](#a-new-commitment) there.
5. **No tool retires a commitment.** To retire one, stop its provers with the
   `stop` file and never serve from it again. For as long as its secret key directory and its spool
   directory exist, keep them inaccessible to every other account and as secret as the weights. Do
   the same for any copy that exists despite this document's rules. If the disk holds nothing else,
   wipe or destroy it before it leaves your hands. Otherwise its encryption has to protect what
   stays on it. Retiring does not undo anything, and the secrets stay dangerous for as long as they
   exist. Without `seed.bin` no one can recompute the commitment ID from the weights.
6. **A `cp_commit` that ended early.** Its clean-up does not touch what a finished step or a finished
   run has taken off its list. In a run that ends midway the seed and the keys stay in CPU memory
   until the large files are overwritten and synced ([what it leaves
   behind](faults.md#a-killed-or-failed-cp_commit)).

## Beside the main road

A provider whose weights must stay hidden needs neither of these two.

<a id="declaring-a-model-files-weights-public"></a>**Declaring a model file's weights public.** With
`-public-weights` ([Public weights](#public-weights)), on real model files, `cp_commit` also needs a
regular file `NAME.gguf.weights-are-public` next to each `NAME.gguf`, and it must begin with that
model file's SHA-256. By creating it you declare that file's weights public. To create it, run:

  ```sh
  (H=$(sha256sum NAME.gguf) && python3 -I -c 'import os, sys; sys.tracebacklimit = 0; os.write(os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), (sys.argv[2] + "\n").encode())' NAME.gguf.weights-are-public "$H")
  ```

If anything with that name already exists, the command fails with `FileExistsError` and writes
nothing. Do not write to the file with a shell's redirection, which follows a symbolic link.

If `cp_commit` rejects a `.weights-are-public` file of yours because group or others can write to
it, because it is empty, or because it holds another SHA-256, put a new file in its place with the
command below. If the SHA-256 differs, first find out why the model file has changed, because the
command declares public whatever `NAME.gguf` holds now. Run it only in a directory that no other
account can write to, as the account that owns the `.weights-are-public` file:

```sh
(N=NAME.gguf; H=$(sha256sum "$N") && python3 -I -c 'import os, stat, sys, tempfile
sys.tracebacklimit = 0
old = sys.argv[1]; line = os.fsencode(sys.argv[2] + "\n"); s = os.lstat(old)
old.endswith(".weights-are-public") and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.geteuid() or sys.exit(old + " is not a regular .weights-are-public file of yours with one name")
f, new = tempfile.mkstemp(prefix=os.path.basename(old) + ".", dir=os.path.dirname(old) or ".")
try: os.write(f, line); os.fchmod(f, 0o644); os.fsync(f); os.rename(new, old)
except BaseException: os.unlink(new); raise' "$N.weights-are-public" "$H")
```

The command renames a new file over the old one, so you do not remove anything by hand. It writes
through no link, and an account that still has the old file open cannot reach the new one. Narrowing
the old file's permissions instead is enough only if no other account opened the file while it could
still write to it. Such an account keeps the file open for writing and can still change what the
file declares, and `cp_commit` does not see that: it checks the file's permissions, not who has it
open. The command stops and writes nothing unless the name is a regular `.weights-are-public` file
with one name that belongs to the account that runs the command (under `sudo`: root). If the file
belongs to someone else, ask its owner to run the command. If another account can write to the
directory, do not run the command there: that account can exchange the entry between the check and
the renaming. The command can then end without an error although the file under that name is not the
one it wrote. `sha256sum` also follows a link under `NAME.gguf`, so the new file would then declare
another file public. Copy the model files into a directory that only you can write to, and declare
them there with the first command (row R29).

**Proofs under `-public-weights`.** We have no argument that the requests of one batch stay hidden
from one another. So serve other people's requests under it one request per batch, or not at all. No
program enforces that.

### The audit with the key

The audit accepts only commitments to synthetic weights. Both programs reject
`-keep-for-the-audit` on real model files, and `cp_prove` rejects it together with `-spool` (read in
the code: row R34). See [tools/audit](../tools/audit/README.md) for how to run the audit, what the
option keeps in `audit/` in the secret key directory, what each verdict erases, and what a YES does
not cover. If you make a commitment or prove a batch with `-keep-for-the-audit`, you take on these
duties (row R34):

- **Nothing may leave the machine before the audit's YES.** This applies to a commitment, even to
  its commitment ID, and to a batch's responses. No program enforces this. Together with the seed,
  the kept files reveal every mask and salt of what they cover. Until the audit has erased a batch's
  kept file, hiding does not hold for that batch. Do not start `cp_prove` on the directory before
  the YES.
- **If the audit ends NO or NOT WHOLE, no part of that commitment may ever leave the machine.** The
  commitment is then [in doubt](#if-the-count-is-lost-or-in-doubt), as it is after the other results
  named in that list. Keep both key directories as secret as the weights for as long as they exist,
  and make [a new commitment](#a-new-commitment). After any other ending,
  fix the cause and run the audit again. If the audit finds nothing kept after a YES, there is
  nothing to do. After the exit `cp-prove-kept-not-written`, do not decide by the audit's result:
  follow [its own row](faults.md#what-you-see-and-what-to-do). The ban after NOT WHOLE has one
  exception, under `sudo`, from a known fault of the scripts, which these sources have. Which words mark it, and what to do: [Under `sudo`](../tools/audit/README.md), row R34. Until that
  is done, nothing of that commitment leaves the machine. A NOT WHOLE with other words is no
  exception.
- **Never use the option on a directory that serves.** `cp_prove` does not start while anything is
  in `audit/` (`cp-kept-for-the-audit`). But it checks only whether that directory is empty, not
  whether the audit has said YES, and only at startup.

## Covered elsewhere

- See [integrations/vllm](../integrations/vllm/README.md) for the model server's files (vLLM).
- See the [glossary](glossary.md#test-build) for test builds. Never use a test option, a test seed
  or a reused file of test randomness with a real commitment. Under a [test
  seed](glossary.md#test-seed) nothing is hidden, no verifier can tell, and the programs reject only
  some kinds of misuse. Destroying such a run's secret key directory does not undo anything that was
  already revealed. No tool destroys one, and backups and images hold it too,
  so keep such a directory as secret as the weights for as long as it exists.
