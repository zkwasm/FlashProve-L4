# prover/: the two GPU programs

This directory holds the sources of `cp_commit`, which commits to a model's weights, and `cp_prove`,
which proves requests. This page is a map of it: which directory holds what, and which file to open
first. [../docs/architecture.md](../docs/architecture.md) explains what the programs do,
[docs/build.md](docs/build.md) how to build them, and [../docs/glossary.md](../docs/glossary.md)
defines the terms. Paths on this page are relative to this directory.

**Do not run any of this under `sudo` or as root.** The programs, scripts and tests need no root
rights (a few test cases that check what happens as root are skipped otherwise), and as root a fault
or a mistake reaches further ([more](../docs/limits.md#running-the-programs-safely)).

## The two programs here, and the verifier

| Program | What it does | Where it starts |
|---|---|---|
| `cp_commit` | Reads the weights and a [configuration](../docs/glossary.md#configuration), and writes a commitment's [public](../docs/glossary.md#public-key-directory) and [secret key directory](../docs/glossary.md#secret-key-directory) | `main()` in `hash_commitment/hf_commit.cu` |
| `cp_prove` | Proves one [batch](../docs/glossary.md#batch) of requests, or serves batch after batch from a [spool directory](../docs/glossary.md#spool-directory) | `main()` in `prover/cp_prove.cu`. A second part, `hash_commitment/hf_keeper.cu`, is compiled separately and linked in |
| `cp_verify` | Checks one proof per call, on a CPU | `../verifier/src/cp_verify.cpp`, outside this directory ([../verifier/README.md](../verifier/README.md)) |

All three compile the proof library, [../blocks/](../blocks/README.md), and between them run its
routines for the files' frame: configuration and plans, hash chains, trees and openings, the order
of a request's segments and messages, the proximity tests, the test on the committed table, the link
between two tables and the key file's proof of ranges. Inside a request's sub-proofs `cp_prove` and
`cp_verify` run implementations of their own of the masked sum-check, the look-ups and the range
check, so the library's tests say nothing about those ([../blocks/README.md](../blocks/README.md)).
What lies here is the model's arithmetic on the GPU, the [sub-proofs](../docs/glossary.md#sub-proof)
about it, the handling of weights, keys and files, and a GPU implementation of the library's heavy
steps.

`cp_commit` uses the verifier in two ways. It runs `cp_verify`, as a separate program, on what it
has made before it releases it. And `build.sh` compiles the verifier's source into it, so that it
rejects a configuration that `cp_verify` would reject. `cp_prove` does neither: it runs no verifier
and holds none of its code, so a configuration that only the verifier's rules reject gets past
its start.

Three names mislead:

- This directory holds both GPU programs; the directory `prover/` inside it is `cp_prove` alone.
- `cp_commit`'s main program is in `hash_commitment/`, but its command line and the reader of a
  model's files are in `weights_commitment/`. A part of `cp_prove` is in `hash_commitment/` too.
- `head/stack.cuh` holds steps of a layer's sub-proof, not of the output head's.

## The directories

The first seven rows stand in the order in which a proof is made; the last five hold what these rest
on. The third column follows the `#include` lines from the three `.cu` files named above. It also
counts a file that is included only under a test's switch.

| Directory | What it holds | Compiled into | Open first |
|---|---|---|---|
| `common/` | What the two programs share: compile-time constants, with the switch between the [production build](../docs/glossary.md#production-build) and [test builds](../docs/glossary.md#test-build); the reader of the [parameter file](../docs/glossary.md#parameter-file); hashes; the generator of [private randomness](../docs/glossary.md#private-randomness) and the table of its keys; the files of the two key directories; how a program that handles a secret key starts; the name of the reason why it ends; the zeroing of freed memory | both (a few files: `cp_prove` only; `common/judge.h`: `cp_commit` only) | `common/setting.h`, then `common/cp_files.h` and `common/coin_keys.h` |
| `weights_commitment/` | Taking in the weights: `cp_commit`'s command line; the reader of GGUF files; the conversion of stored weights to the committed integers; the scan of their ranges; the worst-case check, which asks whether any prompt can drive a value of the forward pass out of its range ([../scales/worst-case-bounds.md](../scales/worst-case-bounds.md)) | 7 files both, 4 `cp_commit`, 2 `cp_prove` | `weights_commitment/arguments.h`, then `weights_commitment/worst_case.cuh` |
| `hash_commitment/` | The commitment to the weights. `cp_commit`'s main program and its steps: building a [weight table](../docs/glossary.md#weight-table) (the weights, encoded with [masks](../docs/glossary.md#mask), under a Merkle tree); the [key file](../docs/glossary.md#key-file) and the [leaf files](../docs/glossary.md#leaf-file); how files are created, checked by `cp_verify` and cleaned up. Also the [table holder](../docs/glossary.md#table-holder), the part of `cp_prove` that keeps a weight table and opens it for each batch | about half of the files both, nearly as many `cp_commit` alone, a few `cp_prove` alone; three headers: tests only | `hash_commitment/hf_commit.cu`; for `cp_prove`'s side `hash_commitment/hf_keeper.cu` and `hash_commitment/hf_keeper_unit.h` |
| `prover/` | `cp_prove` itself: command line; checks on the secret key directory at the start; the service's loop and the spool directory; the plan of GPU memory and the streaming of weights; the forward pass through the layers, and their sub-proofs; a batch's keys and its [opening](../docs/glossary.md#opening) of the weight table; how a run ends | `cp_prove` | `prover/cp_prove.cu`, then `prover/prover.h` (the process's state) and `prover/service.cuh` (`run()`: the order of all steps) |
| `weights_blocks/` | How `cp_prove` holds a model that is stored as quantized blocks (GGUF): reading it from a model's files or as [synthetic weights](../docs/glossary.md#synthetic-weights); its forms in GPU memory; decoding; the experts' matrix products; the passes over the blocks that an opening needs | `cp_prove` (2 of 13 files: both) | `weights_blocks/block_weights.h` |
| `attn/` | Attention: the forward pass through scores, causal softmax and weighted sum, and the sub-proof for it | `cp_prove` | `attn/fork.cuh`, then `attn/front.cuh` (the forward pass) and `attn/plan.cuh` (the plan of lookup calls) |
| `head/` | The output head: the final normalization and the logits, and the sub-proof on them | `cp_prove` | `head/hidden_fork.cuh` |
| `hiding/` | What the sub-proofs share: private random values and their [self-checks](../docs/glossary.md#self-check); the shapes, buffers and random values of commitments to rows (the encoding and the trees are the engine's); the functions through which every sumcheck masks and sends its values; the lookup argument; the approximate range check; the record of a sub-proof's linear claims; the hash chain as GPU kernels use it | 31 files both, 26 `cp_prove` | `hiding/trees.cuh`, then `hiding/sumcheck_door.cuh` (masking and sending), `hiding/forest.cuh` (lookups), `hiding/walk.cuh` (linear claims) |
| `engine/` | The GPU implementation of the proof library's [compute engine](../docs/glossary.md#compute-engine) (`../blocks/engine/engine.h`): it stores the long rows of committed messages, encodes them, builds and opens Merkle trees over them, evaluates and folds them. It is meant to return, bit for bit, what the CPU engine in `../blocks/engine/` returns | 17 files both, 2 each one program | `engine/engine_gpu.cuh` |
| `field/` | The field p = 2^64 - 2^32 + 1 under the names that these sources use; arithmetic with delayed reduction for kernels' inner loops; the encoders' fast transform. Constants and plain arithmetic come from `../blocks/field/` | both | `field/field_names.h` |
| `second_writing/` | A second implementation, for the CPU, of the rule by which private random values are derived ([../docs/format/COIN-KEYS.md](../docs/format/COIN-KEYS.md)). The programs compile it in and compare its values with the first implementation's | 4 files both, 2 `cp_prove`, 1 `cp_commit` | [second_writing/README.md](second_writing/README.md), which also says how far it is independent of the first |
| `longrow/` | A slow CPU reference of how a private random value is taken from the generator's output | no program: only tests include it | `longrow/reference.h` |

## Building, running and testing

| Path | What it is |
|---|---|
| `build.sh` | Builds `cp_commit`, `cp_prove` and their test programs. Build the verifier first (`../verifier/build.sh`). Its first comment describes the main targets; each other target is described at its branch of the script's first `case` statement |
| `check_nvcc_version.sh`, `compiler_sample.cu` | `build.sh` uses both before it compiles anything else: the first checks that the CUDA compiler is recent enough; the second is a small GPU program on which one known compiler bug shows: the build stops if its values are wrong |
| `docs/` | Five pages of detail: [build.md](docs/build.md), how to build; [closed_front.md](docs/closed_front.md), what the production build of `cp_prove` prints once it reads secrets, and the files that it names; [ends-in-short.md](docs/ends-in-short.md), how a run ends and what is zeroed; [cuda-status.md](docs/cuda-status.md), the rule on CUDA's error status; [layout.md](docs/layout.md), the layout file |
| `examples/` | Six example configurations, four small and two at full size ([examples/TARGET-KITS.md](examples/TARGET-KITS.md)), and `examples/run_target_example.sh`, which commits, proves and verifies on synthetic weights at the configuration that you name ([../docs/quickstart.md](../docs/quickstart.md)). Start with `examples/target_small_f2_roomy/`: there the script runs to its `YES` (the suite's row `gpu.run_target_example`). It also takes `examples/target_small_f2_gap/` (that folder's README.md prints the call) and `examples/target_full_f2/`, which it refuses to start where less than 280 GB of host memory are available (run, not on the released programs). At two of the others it does not get through, and that is meant. It rejects `examples/target_small_file_f2/`, which has no layout file and is for tests on a model's files: exit status 2, "layout.txt is missing: target_small_file_f2 is no kit", and nothing is created. At `examples/target_small_f2/` `cp_commit`'s worst-case check fails: `cp_commit` prints "NOT every bounded site fits", makes no commitment and ends with status 7, and the script ends with status 1 and leaves its work directory; `target_small_f2_roomy` is that configuration with roomier scales, under which the check passes |
| `tools/` | `tools/make_request.py` writes a [request](../docs/glossary.md#request) file; `tools/pack_layout.py` makes a configuration's layout file |
| `tests/`, `engine/tests/`, `hash_commitment/tests/` | Unit tests for CPU and GPU, scripts that run the whole programs, the small configurations that tests use ([tests/configurations/README.md](tests/configurations/README.md)), and stored known answers. [../SUITE.md](../SUITE.md) lists the suite's tests, each with its command and what its output must show; `tests/unit.sh` builds and runs one unit test |

## Reading the sources

Nearly every source file has, at or near its top, a comment that says what the file is for. Many
identifiers use other words than the pages do: a sub-proof is `fork`, the table holder is `keeper`,
a private random value is `coin`. A file's first comment often gives the words that it uses, and the
glossary lists many of them ([Words in the source
code](../docs/glossary.md#words-in-the-source-code)).
