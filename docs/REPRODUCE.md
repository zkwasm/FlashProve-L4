# Reproducing the result

Run all commands from this repository's root as an ordinary user. The proof-checking command in
the root README needs only a CPU and C++ compiler. The commands below go further and re-evaluate
the model or generate a new proof. Published artifacts are never reused as secret prover state.

## 1. Re-run the independent CPU integer model

Requires Node.js 22.18 or later. The original checkpoint, tokenizer and committed tensor checksums
are already included. This checks integer conversion against the C++ `wsums.txt` digests, performs
the 64-position numerical check, evaluates the full 256-token request, and writes results to a
fresh configuration copy:

```sh
mkdir -p tmp
cp -R runs/real-config-tu9cL1 tmp/cpu-config
node scripts/check-integer-model.mts tmp/cpu-config runs/real-model-ECt4fl
```

Expected: `agree` is 63, `requestPositions` is 256, and `nextToken` is 310. The generated
`example.req` must match the published request. The CPU reference supports this conversion's
identically zero routed branches; it is not a general MoE implementation or a proof generator.

## 2. Rebuild the converted weights

Requires Node.js, Clang, and a C++17 compiler. Existing upstream Python 3 standard-library helpers
are needed only if recalibrating in step 3; no new Python implementation was added here.

```sh
mkdir -p build/macos tmp
node scripts/fetch-real-model.mts
node scripts/check-real-model.mts
node scripts/export-tinyllama.mts tmp/rebuilt-model
clang++ -O2 -std=c++17 -pthread upstream/weight_sums/weight_sums.cpp \
  -o build/macos/weight_sums
build/macos/weight_sums --eres 11 --threads 4 --out tmp/rebuilt-model/wsums.txt \
  tmp/rebuilt-model/model.gguf
cmp tmp/rebuilt-model/wsums.txt models/stories260K/flashprove32/wsums.txt
cmp tmp/rebuilt-model/model.gguf models/stories260K/flashprove32/model.gguf
```

The fetch script checks pinned checkpoint, tokenizer, reference source and license hashes; it
downloads only missing files and compiles the original C reference. The conversion checker
creates a new `runs/real-model-*` directory. Do not overwrite the published evidence.
Both `cmp` calls must exit 0. The GGUF SHA-256 is:

```text
36d8c24c3d4627dec3fe3bf2d0f79f5a9b0da1a4cebff513cd18b20fe435ca0a
```

The original checkpoint SHA-256 is:

```text
b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
```

## 3. Optionally regenerate public scales and layout

After building a CPU verifier, use its path and the calibration directory printed in step 2.
On Linux replace `build/macos/cp_verify` with `upstream/bin/cp_verify`.

```sh
node scripts/prepare-real-config.mts build/macos/cp_verify CALIBRATION_DIRECTORY \
  tmp/rebuilt-model
node scripts/check-integer-model.mts NEW_CONFIG_DIRECTORY CALIBRATION_DIRECTORY
```

The first script prints `NEW_CONFIG_DIRECTORY`. It calls the upstream scale fitter and table/layout
helpers, and requires the independent verifier's array and security-accounting planners to accept
the configuration. Use the new directory for subsequent proof generation. Public table partition
sizes and GPU minimum sizes are part of the supplied configuration, not inferred from a toy proof.

## 4. Build on an L4 and generate a fresh proof

The tested environment was Ubuntu 24.04, g++ 13, bash, GNU tools, one NVIDIA L4, and CUDA
**13.2 Update 2**, toolkit **13.2.2**, `nvcc` **13.2.86**. The tested driver was **580.178.04**.
The upstream build checks compiler behavior on the actual GPU. Provisioning another VM or running
this on another toolkit is not required to verify the published proof, and is not claimed as tested.

Build both the independent verifier and production GPU programs:

```sh
export PATH=/usr/local/cuda-13.2/bin:$PATH
bash upstream/verifier/build.sh
(cd upstream/prover && FLASHPROVE_CUDA_ARCH=sm_89 ./build.sh serving)
```

Use the production `serving` target. Test targets intentionally refuse actual model files.
Do not edit any source under `upstream/prover/`, `upstream/blocks/` or `upstream/verifier/src/`
while compiling: the build checks that its source snapshot has not changed. Compilation can
remain quiet for minutes. Success is exit 0; binaries and records appear in `upstream/prover/build/hc/`.

The runner uses the included converted weights and public configuration, creates a fresh work
directory and secret state, runs commitment, inference/proving, then independent verification:

```sh
mkdir -p tmp
bash scripts/run-real-l4.sh "$PWD/runs/real-config-tu9cL1" "$PWD/tmp/new-l4-run"
```

The work directory must not exist. The expected `commit.exit`, `prove.exit` and `verify.exit` are
all `0`; `verify.log` ends in `YES`. The runner's final line is:

```text
FULL REAL MODEL PROOF VERIFIED; NEGATIVE TESTS STILL REQUIRED
```

Then run the independent output comparison and five negative tests:

```sh
node scripts/check-real-proof.mts upstream/bin/cp_verify tmp/new-l4-run \
  runs/real-config-tu9cL1
```

Expected final line:

```text
PASS: fresh real-model inference proof and five independent tampering rejections
```

A fresh run will produce a **different commitment ID and different proof bytes**, because it uses
fresh randomness. Trust the new ID printed in that run's `commit.log`, saved as `trusted-ID.txt`.
Do not pass the published ID when checking a newly generated commitment. An `M=1` commitment has
one one-time leaf; do not reset or copy spent secret state to reuse it. Start a new commitment
for another run. Keep `isk/`, seeds and other private work files out of published artifacts.

If using temporary cloud hardware, preserve the public artifacts you need, then delete the VM
and its disks. The original test VM and boot disk have already been deleted.

## Scope of reproducibility

The repository includes source rather than prebuilt binaries. The CPU quick verification is
tested from a clean copy of these published files. The recorded original full GPU run used the
same patched source content, configuration and converted model, laid out under a different
working path; build records preserve those original paths. A fresh GPU proof has not been
regenerated merely to change repository packaging. Its public format and verifier inputs remain
unchanged. Model checks, exact integer checksums and accepted cryptographic proofs are separate
pieces of evidence; none alone proves general model quality or hides these public weights.
