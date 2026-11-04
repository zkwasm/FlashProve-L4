# FlashProve-L4

A reproducible real-model inference proof on one NVIDIA L4, with independent CPU verification.

For a short, nontechnical explanation, read [what we achieved and why it matters](docs/OVERVIEW.md).

This repository adapts [Anthropic FlashProve](https://github.com/anthropics/FlashProve) at commit
`9e2bdd8ff5bf9c62f18263d7dac189b0400b260c` (version 0.16.0) to a small pretrained
[TinyStories260K model](models/stories260K/MODEL-DETAILS.md) and a 24 GB-class L4.
It includes the patched source, public model weights, configuration, commitment ID, public key
and leaf, request, output, proof, build records, and verification logs. No GPU is needed to check
the published proof. This is an independent adaptation, not an Anthropic release.

## What was demonstrated

The production programs completed this sequence on one L4:

1. `cp_commit` loaded the converted real pretrained weights, checked all 56 committed tensor
   checksums and worst-case ranges, and created a fresh key, commitment and one-time leaf.
2. `cp_prove` executed integer inference and generated the layer, attention, head and opening proofs.
   These are proofs of that execution, not a replay of upstream's stored synthetic example.
3. A separate CPU `cp_verify` process verified the key, leaf and request, returning exit **0**, **`YES`**.
4. An independently built verifier on an arm64 Mac also returned **`YES`**. The output token
   **310** matched a separate TypeScript CPU integer implementation.
5. Changing the input, output, proof, commitment ID or public key was rejected in all five tests.

The request is `r=255, e=256, T=256`: **one next-token claim**, with no claims about the intermediate
prompt transitions. Its input repeats a checked 64-token story prefix four times. The existing
GPU product-layer kernel imposes a minimum size that this 256-token request satisfies.
The proof is approximately 5.4 MiB. These observations are not performance benchmarks.

## Verify it yourself — CPU only

Requirements: Node.js **22.18 or later**; macOS with Apple Clang/Xcode command-line tools, or
Linux with **g++ 13 or later**, bash and GNU coreutils. No npm packages, Python, CUDA, model server,
cloud account or pretrained-weight download is needed for this step. Run as an ordinary user.

```sh
git clone https://github.com/zkwasm/FlashProve-L4.git
cd FlashProve-L4
node scripts/verify-published.mts
```

This checks every file in `MANIFEST.json`, builds the CPU verifier from source, verifies the
published proof, compares token 310 with the recorded CPU reference, and creates five changed
copies which must all fail. It leaves the published artifacts intact and writes fresh logs to
`runs/recheck-*/verification/`. Compilation can remain quiet for several minutes.

Expected final verification result:

```text
PASS: fresh real-model inference proof and five independent tampering rejections
```

Expected individual outcomes:

| Case | Exit | Entire final verifier line |
|---|---:|---|
| Original proof | 0 | `YES` |
| Changed input token | 1 | `NO: proof-refused` |
| Changed output token | 1 | `NO: reply-differs` |
| Changed proof byte | 1 | `NO: proof-refused` |
| Changed commitment ID | 1 | `NO: model-statement` |
| Changed public key | 1 | `NO: key-refused` |

The original results are in [results.json](runs/real-l4-v8/verification/results.json).
The repository's clean-checkout publication checks are recorded
[here](runs/publication-check/results.json), with the fresh [verification log](runs/publication-check/verify.log).
The manifest detects changed files; it does not authenticate the publisher or establish that
the included verifier is correct. Review the source and compare it with the pinned upstream revision.

### Direct verifier call

The trusted commitment ID is taken from **our committer's output**, not from the proof package:

```text
2cde2d3ad5214393c2adec56f3ee4ba4a97932f15ae92c2077101bc3fd937470
```

After the build, use `build/macos/cp_verify` on macOS or `upstream/bin/cp_verify` on Linux:

```sh
build/macos/cp_verify \
  runs/real-config-tu9cL1 runs/real-l4-v8/ivk \
  runs/real-l4-v8/in/example.req runs/real-l4-v8/out/example.y \
  runs/real-l4-v8/out/example.proof \
  -id 2cde2d3ad5214393c2adec56f3ee4ba4a97932f15ae92c2077101bc3fd937470
```

An existing compatible verifier can also be supplied to the full test:
`node scripts/verify-published.mts /absolute/path/to/cp_verify`.

## How this differs from Anthropic FlashProve

| Area | Pinned upstream | This adaptation |
|---|---|---|
| GPU build | Default `sm_90a`; large shared-memory kernels | Optional `FLASHPROVE_CUDA_ARCH=sm_89` for L4; original default preserved |
| Memory use | Includes 128 KiB shared-memory paths and a 24 GiB engine reservation | Smaller buffers/histograms, global-memory lookup paths, 8 GiB engine reservation; existing capacity checks retained |
| Integer matrix products | cuBLAS fast paths | Exact GPU integer fallback when L4/cuBLAS reports `NOT_SUPPORTED`, including graph capture |
| Host SHA instructions | Uses SHA instruction flags in affected build paths | Existing portable SHA implementation on G2 CPUs lacking those instructions |
| Weight reader | Fold constants and reader checks assume width/head width 5120/128 | Adds 256/32 with corresponding constants; original shape retained |
| Zero tensors | Nonzero-oriented scale/norm paths | Canonical zero exponents and exact zero norm handling for the new small shape; nonzero guards retained |
| Pretrained model | Original model architecture and examples | Actual TinyStories260K weights embedded into an accepted layout with zero padding and zero experts |
| Verification | Independent CPU verifier | **No edits under `upstream/verifier/src/`; equations, format and acceptance conditions retained** |
| macOS | Linux-oriented build/setup | Apple Clang verifier wrapper and `getentropy` fallback for public CPU tests |
| Evidence | Upstream's examples and tests | Fresh real-weight commitment, inference proof, independent Mac verification and five rejection tests |

All upstream changes are in [upstream-l4-small-model.patch](upstream-l4-small-model.patch).
They are **already applied** to the vendored `upstream/` directory; do not apply the patch twice.
[UPSTREAM.json](docs/UPSTREAM.json) records provenance. The vendored tree contains the tracked
files present in the tested sparse checkout; Lean dependencies and three unused large examples
are omitted. This repository does not claim to rebuild or recheck the Lean formalization.

### Model conversion and its limits

The source model is `karpathy/tinyllamas`, revision `0bd21da7698eaf29a0d7de3992de8a46ef624add`,
`stories260K`: width 64, FFN width 172, five layers, eight query heads, four KV heads and vocabulary
512. The raw checkpoint is 1,056,540 bytes. Its model card and license are included.

The accepted layout has width **256**, head width **32**, six layers, dense/shared FFN widths
**512/256**, and **two identically zero routed experts**. This is an embedding of the trained
dense model, not a separately trained MoE:

- Repeat the residual stream four times so GPU weight rows have 256 input columns and RMS epsilon
  stays unchanged. Average those copies when feeding the original FFN; repeat its output.
- Place each original RoPE pair at every fourth pair of the wider head, preserving its frequency.
  Double query weights to compensate for the changed square-root attention denominator.
- Pad FFNs with zero units. Put odd-layer dense FFNs in the shared expert and keep routed experts
  zero. Append a zero-output residual layer, which acts as an identity.
- Store the zero routed tensors as F16, exactly, because the GPU reader requires a block-decodable
  type. Other tensors are F32; all 56 integer tensor checksums are unchanged by that storage choice.

On the checked 64 positions, the original-width adaptation's 32,768 logits match the original
floating-point graph exactly. The wider embedding retains all 64 greedy choices; the largest
observed logit difference is about `2.67e-5`. The original author's pinned C program produces the
same original greedy text. See [float results](runs/real-model-ECt4fl/results.json).

Integer inference is an approximation of floating-point inference: 63/64 teacher-forced next-token
choices agree in that check, and free generation eventually diverges. The proof establishes the
specified **integer** computation, not universal equivalence to the original floating-point model
or general language quality. See [integer reference](runs/real-config-tu9cL1/integer-reference.json).
There is no claim that these numerical checks hold for every input.

## Reproduce inference or generate a new proof

See [REPRODUCE.md](docs/REPRODUCE.md) for:

- re-running the independent CPU integer reference against all tensor checksums;
- rebuilding the converted model from the original public checkpoint;
- rebuilding the production L4 prover and creating a fresh commitment and inference proof;
- prerequisites, paths, expected exits, and handling fresh one-time state.

Full proof generation currently needs **L4**. A full CPU prover has not been completed. The
earlier `cpu/commit.cpp` prototype and its scripts are experimental CPU commitment work, not the
real-model proof generator used for this result.

## Evidence and files

| Path | Contents |
|---|---|
| `upstream/` | Patched, pinned upstream source and retained upstream documentation/licenses |
| `scripts/` | TypeScript model conversion, CPU integer reference, configuration and verification; L4 runner |
| `cpu/l4_*.cu` | L4 regression tests and diagnostic probe |
| `models/stories260K/` | Original public checkpoint/tokenizer/reference C; converted model and tensor sums |
| `runs/real-config-tu9cL1/` | Public parameters, scales, tables/layout, request and integer reference |
| `runs/real-l4-v8/ivk/` | Public key and leaf; **not** secret prover state |
| `runs/real-l4-v8/in/`, `out/` | Actual request, four-byte output and generated proof |
| `runs/real-l4-v8/commit.log` | Fresh committer's ID, tensor checks, bounds and key/leaf checks |
| `runs/real-l4-v8/verification/` | Original independent Mac verification and rejection-test logs |
| `runs/l4-backend/` | Backend regression logs and production build records |
| `MANIFEST.json` | SHA-256 of the published source, model, configuration and evidence files |

Backend checks included 584 GPU/CPU engine tree comparisons with zero differences, 288 integer
matrix-product cases, histogram comparisons, and 10 independently verified lookup proof groups.
These checks supplement the real-model proof; they are not substitutes for it.

The GPU test used one NVIDIA L4 on GCP `g2-standard-8`, CUDA 13.2 Update 2 (`nvcc 13.2.86`),
driver 580.178.04, and g++ 13. The temporary VM and its boot disk were deleted after artifacts
were preserved. No secret prover seed, `isk/` directory, cloud credentials or prebuilt executables
are distributed. Build records retain the original paths and compiler flags for provenance.

## What `YES` means here

The [independent verifier log](runs/real-l4-v8/verification/honest.log) includes:

```text
commitment: 47 closeness rounds of the one-time files rest on another source than the bound within a third of the distance; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given
proof of work: up to 8 bits counted in 40 rounds; with it the least round has 128.00 bits, without it 120.31 bits
result-v1 status=0 claim=next-token id=2cde2d3ad5214393c2adec56f3ee4ba4a97932f15ae92c2077101bc3fd937470 bits=128 work=8 work-rounds=40 least-round=128.00 least-round-without-work=120.31 commitment-assumed-rounds=47 record=1 r=255 e=256 T=256 tokens=none y=yes policy=exact-match C=256 theta=256 cap=0 s=20 sh=8 sT=4 F=24 above-clip=0 leaf=0 stages=key,leaf,request dw=8 build=c8138323b58e5b35c1e1b379a14028acb19166a601b900ac763593c7a7a7d31e
YES
```

This is the program's accounting, including its stated commitment assumptions, not a new
cryptographic security proof or a weight-hiding audit. These are public pretrained weights.
The proof binds an integer model to the commitment ID; connecting those integers to the named
pretrained checkpoint also relies on the published conversion, checksums and reproducibility checks.

## License and attribution

FlashProve and the adaptation code are Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE),
[upstream/NOTICE](upstream/NOTICE) and [upstream/THIRD_PARTY.md](upstream/THIRD_PARTY.md).
The public TinyStories model is marked MIT in its [model card](models/stories260K/MODEL-CARD.md).
The pinned llama2.c reference is MIT; its license is preserved in
[REFERENCE-LICENSE](models/stories260K/REFERENCE-LICENSE).
