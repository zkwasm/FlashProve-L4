# FlashProve

Version 0.16.0 · A research artifact: not maintained, not accepting contributions. · Apache 2.0

## What it is

FlashProve lets an AI provider prove which model answered a request, without revealing the model's
weights.

- **Commit.** The provider publishes a commitment ID, which [binds](guide/what-is-proved.md) it to
  one set of weights. The weights themselves stay secret: every published value that depends on them
  is masked, except a [declared list](docs/what-is-not-hidden.md).
- **Prove.** For each reply, the provider's GPU produces a proof that the reply is the committed
  model's output.
- **Verify.** Anyone can check the proof on an ordinary CPU in seconds with `cp_verify`, without the
  weights.

## What model it runs

FlashProve runs Llama 4 Maverick at full size, with a 1,024-token context window, on a single NVIDIA
H100. The prover reaches about 110 tokens/second of prefill at 24 requests per batch
([Performance](guide/performance.md)).

## What is proved

The protocol's soundness is proved in Lean 4. A provider that tries to pass off output that is not
the committed model's is caught, except with probability about q/2^124 + 2^-101 after q hash
queries. [lean/THEOREM.md](lean/THEOREM.md) has the theorem and how to check it.

<a id="who-wrote-it"></a>

## Built to be run by an agent

Building FlashProve, running it and checking its output take long builds and many exact comparisons.
That work is meant for a coding agent. Point yours at this repository with:

```text
Read README.md in the FlashProve repository, starting at "For agents", and AGENTS.md. Working from
the repository root as an ordinary (non-root) user, build the verifier, check the stored example
proof, and report the exit code and last line. Then tell me which other routes (toy proofs on CPU,
small model on an H100, full size) this machine can run, and ask me before starting any of them.
```

---

## For agents

Run everything from the repository root as an ordinary user, never as root or under `sudo`. Only
installing system packages needs root, so ask your user to do that. If root is the only account,
create one first ([how](docs/quickstart.md#install)). Also read [AGENTS.md](AGENTS.md) for the
rules on reporting results.

### Hardware and software

| Route | What it does | Needs | Time |
|---|---|---|---|
| 1. Verify a stored proof | Builds `cp_verify`, checks an example proof | Linux, bash, g++ 13+; under 2 GB RAM | build 1–2 min, check < 2 s |
| 2. Toy proofs | Self-test of the proof library on CPU | Route 1, plus make and python3; under 1.5 GB RAM | ~1 min |
| 3. Small model | Builds the prover, commits, proves and verifies an 8-layer synthetic model | GPU stack (below); 10 GB RAM | build ~29 min, run 23–30 s |
| 3, full size | Same at Llama 4 Maverick's shape, synthetic weights, 1,001 tokens | GPU stack; 280 GB RAM | ~42 min |
| 4. Llama 4 Maverick | Downloads a public 4-bit conversion (232 GB) and proves one token | GPU stack, curl, Python 3.13; 330 GB RAM, 245 GB disk; a calibration set that you make first | hours |

**GPU stack** (routes 3 and 4) is the only supported platform for the prover:
- x86-64 Linux 4.14 or later, on a CPU with the SHA extensions.
- One NVIDIA H100 80 GB, used by nothing else.
- CUDA 13.2 Update 2 (nvcc 13.2.86) on PATH. [prover/docs/build.md](prover/docs/build.md#install-the-compiler)
  prints the pip lines that install it.
- gcc and g++ 13 first on PATH; binutils, flock, python3.
- The repository and work directories on a local disk.

On minimal Ubuntu 24.04, route 1 and 2 dependencies are (needs root):

```
apt-get update && apt-get install -y --no-install-recommends g++ make python3 unzip
```

### 1. Build the verifier and check a stored proof

```
verifier/build.sh
```

It is silent for 1–2 minutes, then prints a line starting `THE VERIFIER IS BUILT:` and exits 0. A
line starting `build.sh: note:` is a warning: quote it in your report. If your tool calls time out,
run the build in the background:

```
echo "build started" >> verifier-build.log; nohup bash -c 'verifier/build.sh; echo "exit code $?"' >> verifier-build.log 2>&1 &
```

Then poll `tail -n 2 verifier-build.log` until the last line starts `exit code`. Don't start a
second build meanwhile.

```
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)"
```

Expect last line `YES`, exit code 0. This example is a small model with synthetic weights, so it
shows only that the verifier works, not anything about a served model.

### 2. Toy proofs (CPU)

```
W=$(mktemp -d) && LOOK="$W/look" && echo "LOOK=$LOOK"
blocks/first_look.sh "${LOOK:?first run the line with mktemp}"
```

Expect exit code 0, and the third line from the end starts `AS EXPECTED: the honest copy was
accepted`.

### 3. Prove on a GPU

```
examples/quickstart-gpu.sh
```

This builds `cp_verify`, `cp_commit` and `cp_prove`, then commits, proves one 64-token request and
verifies it.
- It can be silent for over 15 minutes. It logs to `quickstart-gpu.log` in a directory it names
  first.
- Success is exit code 0 and a line starting `quickstart-gpu: all steps ended as expected`.
- Also check that every `Compiled by` line ends in `release 13.2, V13.2.86`. A newer nvcc does not
  stop the build, but it is not supported.

Full size, after that build:

```
W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"
prover/examples/run_target_example.sh target_full_f2 "${WORK:?first run the line with mktemp}"
```

Expect `status 0`, a `proof of work:` line, then `YES`. Progress goes to `$WORK/cp_commit.log`.
With too little memory it exits with code 2 at the start.

For the real Maverick weights, see [route 4](guide/getting-started.md#route-4-llama-4-maverick) and
[`examples/maverick/run.sh`](examples/maverick/README.md). To commit to and serve a model of your
own, see [Serving](guide/serving.md) and the
[provider's duties](docs/operators-duties.md). Those pages are required reading before using weights
that must stay secret.

### Verifying a real proof

```text
bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID
```

- **The commitment ID:** get it from a party your user trusts, never from the server being checked
  or from files sent with the proof. The stored examples are the only exception.
- **The request file:** build it yourself ([Verify a proof](guide/verify-a-proof.md)).
- **What to read:** the exit code together with the last line.

| Exit | Last line | Meaning |
|---|---|---|
| 0 | `YES` / `WITHIN LIMITS: …` | Accepted under exact match / clipped gap |
| 1 | `NO: <name>` | Rejected (or a local fault) |
| 5 | `NOT CHECKED: <name>` | No verdict; retry once only for `internal-stage` or `internal-exception-parFor`, otherwise report |
| 14 | `NOT-A-VERIFICATION` | The call was invalid; see the lines after `---- the call ----` |
| 2, 3, 4, 15 | | Bad command line or file, or a partial call; no verdict |

[verifier/docs/output.md](verifier/docs/output.md#exit-codes) has the full table.
[`integrations/check_proof.py`](integrations/README.md) wraps the call and the comparisons.

### Tests

```sh
D=$(mktemp -d) && ./run_suite.py cpu "$D/work"
```

These are the CPU tests. They overwrite `bin/cp_verify` and
`bin/weight_sums`.
- Exit 0 means a pass; 77 means tools are missing.
- Exit 3 means a test saw something false accepted: report it per [SECURITY.md](SECURITY.md).

See [SUITE.md](SUITE.md).

### Rules

- Never get past an error with `sudo`, `chmod`, `chown`, `rm`, or by editing a source file. Stop and
  ask your user.
- Don't build `prover/` unless the task needs it.
- Report the exact command, the exit code and the full last line. For an accepted proof, also report
  the commitment ID you passed and the `covered:` and `result-v1` lines.

---

<a id="docs"></a>
<a id="how-it-works"></a>

## More

- **Guides:** [Getting started](guide/getting-started.md), [How it works](guide/how-it-works.md),
  [Verify a proof](guide/verify-a-proof.md), [Serving](guide/serving.md),
  [What is proved](guide/what-is-proved.md), [Weight hiding](guide/weight-hiding.md),
  [Known limits](guide/known-limits.md), [Performance](guide/performance.md), [Terms](guide/terms.md).
- **Reference:** [docs](docs/README.md), [glossary](docs/glossary.md), [protocol](docs/protocol.md),
  [format spec](docs/format/FORMAT.md), [Lean library](lean/README.md),
  [Lean verifier](tools/lean-verify/README.md), [tests](SUITE.md), [paper](paper/paper.pdf),
  [SECURITY.md](SECURITY.md).
- **Examples and checks:** [the stored example](examples/accepted-proof/README.md),
  [a full-size proof](examples/full-size/README.md),
  [rejected proofs](examples/rejected-proofs/README.md), [faults](docs/faults.md),
  [the integration's limits](integrations/LIMITS.md), [checking the Lean library](lean/CHECKING.md),
  [its results](lean/CHECK-RESULTS.md).
- **Limits:** [docs/limits.md](docs/limits.md), [docs/limits-further.md](docs/limits-further.md),
  [the table of all rows](docs/limits-all-rows.tsv), [lean/LIMITS.md](lean/LIMITS.md),
  [known errors](docs/known-errors.md).

## Licence and acknowledgments

Apache 2.0: see [LICENSE](LICENSE). Material from other projects keeps its own licence.
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) list it,
[PUBLISHED-WORKS.md](PUBLISHED-WORKS.md) lists the published works that the code applies or names,
and [licenses/](licenses/) has the texts.

We thank Alessandro Chiesa, Giacomo Fenzi, Guy Weissenberg, Miranda Christ, Rahul Dalal, and Ben Mayer for their input and collaboration on FlashProve. We also thank James Petrie for VerInf.
