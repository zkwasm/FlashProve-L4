# Tests

Run them from the parent directory, `integrations/vllm/integer_model`, as an ordinary account, not
as root ([OPERATING.md](../OPERATING.md), "As root or under sudo"). A `PYTHONPATH` handed through
`sudo` makes root import whatever code lies on that path.

Install with `pip install --require-hashes -r tests/requirements.txt`. The tests also need g++ 13 or
later and the rest of the repository (example configurations from `prover/examples/`, C++ sources
from `prover/`). A test that needs something the machine lacks is skipped, and `pytest -rs` prints
the reason; three tests fail, not skip, without the package gguf.

The tests keep their files in pytest's temporary directory and delete nothing. A run of the CPU
tests leaves about 3.0 GB there, and pytest keeps the last three runs. To keep `__pycache__` and
`.pytest_cache` out of the checkout, start them with
`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider`.

All tests run at small size on synthetic weights. The CPU tests pass on these sources, as the
suite's row `integer_model.cpu` (`SUITE.md`), with no failure and at most four skips, and the
comparison with Lean's definition ran on these sources; its times below span its runs on these and
on other sources ([its README](lean_vectors/README.md)). Every other result in the table below, and
every other time in the table of files, is from other sources of the package
([OPERATING.md](../OPERATING.md), "What has been run"); the comparison with the GPU prover ran on
other programs too. For the counts of the CPU tests and of `pytest`, those sources lack some checks
of `integer_model/model.py` and `integer_model/block_weights.py` and their tests, so your counts may
be higher. The results with a GPU are from one H100 with vLLM 0.29.0, PyTorch 2.13.0 and
transformers 5.17.0. The result without one is from Python 3.13, PyTorch 2.9.1, numpy 2.3.5,
safetensors 0.5.1 and transformers 4.57.6, below the package's pins. Other results:
[OPERATING.md](../OPERATING.md), "What has been run".

| To run | Command | Needs | Result |
|---|---|---|---|
| the CPU tests | `pytest -m "not vllm and not prover"` | no GPU | 935 passed, 4 skipped (four need `INTEGER_MODEL_TEST_DEVICE=cuda`) |
| the same on a GPU | `INTEGER_MODEL_TEST_DEVICE=cuda pytest -m "not vllm and not prover"` | a GPU with 8 GB free | 939 passed, in 11 to 13 minutes |
| the tests with vLLM | `pytest -m vllm` | vLLM 0.29.0 | 50 passed, in 5 to 6 minutes |
| the comparison with the GPU prover | `INTEGER_MODEL_TEST_DEVICE=cuda pytest -m prover` | one H100 and the prover's programs (below) | 6 passed, in 20 to 22 minutes, with the programs built on another machine and copied there |
| the comparison with Lean's definition | `python3 tests/lean_vectors/run.py` | Lean, no GPU ([its README](lean_vectors/README.md)) | `All steps ended as they must.`, in 10 to 34 minutes |
| everything the machine can run | `pytest` | | 985 passed, 10 skipped (six need the prover, four `INTEGER_MODEL_TEST_DEVICE=cuda`), about 14 minutes |
| one file | `pytest tests/test_model_folder.py` | | |

| Variable | Meaning |
|---|---|
| `INTEGER_MODEL_TEST_DEVICE` | the PyTorch device on which the tests run the model; `cpu` if unset |
| `INTEGER_MODEL_PROVER_BUILD` | the directory with `cp_commit`, `cp_prove` and `cp_verify` |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` or `0` selects vLLM's model runner on a GPU; without a GPU, the tests with vLLM run under its CPU worker, which does not use the V2 model runner |
| `OMP_NUM_THREADS` | PyTorch's CPU threads; at most four if unset |

Times: tests without a marker, on a busy machine; tests with `vllm` or `gpu`, on one H100;
`lean_vectors/`, on the machines that its README tells of.

| File | What it covers | Markers | Time |
|---|---|---|---|
| `test_against_recorded_values.py` | `IntegerModel`'s values at every layer and position against the recorded values (below) | | 34 s |
| `test_arithmetic_edges.py` | rounding ties, ends of clamps and tables, failed range conditions, equal logits, float64's exact sums | `gpu` for four tests | 26 s |
| `test_parameter_file.py` | the reader of the parameter file, against that of `cp_commit` and `cp_prove` | | 20 s |
| `test_model_folder.py` | `load_from_folder`: `integer_model.json`, its options, the files' SHA-256 digests | | 9 s |
| `test_block_decoding.py` | decoding of stored blocks against the GPU prover's own code | | 63 s |
| `test_gguf_files.py` | GGUF files load to the integers of their sums file; malformed files are rejected | | 106 s |
| `test_host_memory.py` | a model too large for free host memory is rejected before anything is allocated | | 10 s |
| `test_examples.py` | the example directories and the command lines of the scripts in `examples/` | | 34 s |
| `test_serve_gguf_look.py` | `examples/serve_gguf.sh`'s check of the folder for temporary files | | 6 s |
| `test_find_or_give_root_up.py` | what the scripts do where `vllm` or a Python package is not found, and as root | | 7 s |
| `test_float_pair.py` | the exported floating-point checkpoint, against Hugging Face transformers' Llama 4 code and vLLM | `vllm`, `gpu` for one test | 86 s, without its `gpu` test |
| `test_fit_limits.py` | `examples/fit_limits.py` and `score_reply.py` on clipped-gap limits | | up to 12 min on a busy machine |
| `test_seeded_score.py` | the public noise of a seeded score (`integer_model/noise.py`) and `score_reply.py --seed` | | 42 s |
| `test_readme_request.py` | runs the shell blocks of README.md's "A request from the model's own reply" | | 13 s |
| `test_against_the_prover.py` | which requests get a proof from `cp_prove` (below) | `prover`, `gpu` | 20 min |
| `test_vllm_model_class.py` | the plug-in's model class, loader and registration | `vllm` | 1 min |
| `test_vllm_engine.py` | vLLM's engine serves exactly the tokens that a stand-in model chooses | `vllm` | 2 min |
| `test_vllm_integer_model.py` | vLLM serves the integer model at two small sizes, with the replies of the model driven directly | `vllm` | 1 min |
| `lean_vectors/` | not a pytest file: Lean's definition of the model against this package ([its README](lean_vectors/README.md)) | | 10 to 34 min |

`pytest --markers` lists what each marker needs. `test_float_pair.py` and some tests of
`test_examples.py` and `test_fit_limits.py` need safetensors and transformers, which the quick start
installs with vLLM's pinned packages
([`integrations/vllm/requirements.txt`](../../requirements.txt)); without them they are skipped.

**The recorded values.** `tests/recorded_values.tsv` holds the expected values of
`test_against_recorded_values.py` and of 26 tests of `test_arithmetic_edges.py`. A separate C++
implementation of `docs/model.md`, which is not part of this repository, computed them, so you
cannot compute them again or add a case from this tree. The file holds each array's shape and the
first 128 bits of its SHA-256, so a mismatch names the array, not the entry. A case is found by a
digest of its inputs, and a test whose inputs change fails with "no recorded values". The recorded
logits, greedy choices and logit gaps were computed not by that implementation, which evaluates the
output head only while it makes a proof, but by a loop written to follow it.

**What the tests reach.** The C++ implementation, the prover and Lean's definition were written in
this project from the same document, so they are no independent check. Of 20 faults put into the
package one at a time, the tests against the recorded values caught 19; the one left, the exact
square root's upward correction left out, is needed by no tried value. No test here proves a reply
that vLLM served, or runs the proof middleware or the example client of `integrations/`.

## The comparison with the GPU prover

The programs print no intermediate value of the forward pass, so `tests/test_against_the_prover.py`
compares decisions. `IntegerModel`, on the CPU, says which requests meet an acceptance policy and
which miss it by the least amount, and `cp_prove` must prove the first and no other. Two tests use
the example `target_small_f2_roomy`. A third uses `target_small_file_f2` in four forms of GGUF file:
a mix of Q4_K, Q5_K, Q6_K and F32; Q8_0; F16; BF16. The tests leave about 5.6 GB in pytest's
temporary directory (on other programs). They compare no value at a single position, nothing at full
size and nothing on a real model's files.

They need one H100 with nothing else running on it, CUDA 13.2 Update 2 (nvcc 13.2.86), with its
libraries where the loader finds them without `LD_LIBRARY_PATH` (a later toolkit is not supported:
[prover/docs/build.md](../../../../prover/docs/build.md)), g++ 13, and this repository's programs.
From the top of the repository:

```sh
verifier/build.sh
prover/build.sh
export INTEGER_MODEL_PROVER_BUILD=$PWD/prover/build/hc
```

## Adding a test

- Take the device from the fixture `device` and vLLM from the fixture `vllm`, which sets vLLM's
  environment before the import. Write only below `tmp_path`.
- Mark what the test needs (`gpu`, `vllm`, `prover`): `conftest.py` makes it a skip.
- Where something must be rejected, match a part of the message that names the fault.
- Take expected integers from the provers' code or from Python's integers, never from this package.

## Licences

`tests/requirements.txt` pins pytest, gguf and the twelve packages that the two install. None is
copied into this repository; each is installed from its own distribution under its own licence, as
the metadata of the pinned wheel states it: pytest 9.0.3, MIT; iniconfig 2.3.0, MIT; pluggy 1.6.0,
MIT; packaging 26.3, Apache-2.0 OR BSD-2-Clause; pygments 2.21.0, BSD-2-Clause; gguf 0.19.0, MIT (by
its classifier and its licence file; it has no licence field). `../../LICENCES.tsv` has the other
eight.
