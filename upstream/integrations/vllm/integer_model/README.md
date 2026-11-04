# The integer model in PyTorch and vLLM

The proofs of FlashProve are about the **integer model**: an integer form of Llama 4 Maverick's
forward pass, on weights read from GGUF, llama.cpp's file format, with every step defined as an
exact function on integers ([docs/model.md](../../../docs/model.md)). It approximates the
floating-point model and is not equal to it
([limits](../../../docs/limits.md#the-integer-model-is-not-the-floating-point-model)).
`integer_model` evaluates it in PyTorch, and the plug-in `integer_model_vllm` lets an unmodified
[vLLM](https://github.com/vllm-project/vllm) 0.29.0 serve it. The package takes no part in making or
checking a proof ([../README.md](../README.md)).

Two ways to use it:

1. **Score a served reply.** vLLM serves the ordinary floating-point model; one prefill pass of the
   integer model then scores how far each token is from the greedy choice, as in
   [Token-DiFR](https://arxiv.org/abs/2511.20621). `examples/score_reply.py` computes the score. A
   proof covers whether the reply meets a clipped-gap policy's limits on that score.
2. **Serve the integer model**: greedy decoding, one sequence at a time, one GPU, windows of up to
   1,024 tokens.

An *acceptance policy* says which replies get a proof. *Exact match* accepts only the integer
model's own greedy reply; a *clipped-gap policy* also accepts replies near it, within limits that
the provider chooses (no verifier checks how they were chosen). A provider is expected to serve as
usual and prove under a clipped-gap policy; exact match is for replies that the integer model itself
served ([OPERATING.md, "Score a reply"](OPERATING.md#score-a-reply)).

## Install and quick start

Needs Linux x86-64, Python 3.13, g++ 13 or later and this repository; serving needs an NVIDIA GPU.
Do not run these scripts under `sudo` or as root ([why](OPERATING.md#as-root-or-under-sudo)).

```sh
cd integrations/vllm/integer_model &&
python3.13 -m venv .venv && . .venv/bin/activate &&
pip install --require-hashes -r ../requirements.txt &&
pip install --require-hashes -r tests/requirements.txt &&
pip install --no-deps --no-build-isolation . &&
pytest
```

Every test should pass; what the machine cannot run is skipped ([tests/README.md](tests/README.md)).
The install leaves `.venv/`, `build/` and `integer_model.egg-info/` in this directory. Published
security advisories cover some of the pinned packages, vLLM 0.29.0 among them: before you expose a
server, read [the checklist](../REFERENCE.md#the-operators-checklist).

```sh
examples/serve.sh small_synthetic  # a small synthetic model on 127.0.0.1:8000, until Ctrl-C
```

After "Application startup complete.", in a second shell in this directory:

```sh
examples/request.sh small_synthetic
```

With the example's synthetic weights, the reply's `choices[0].token_ids` is `[365, 287, 374, 290]`;
the CPU tests check these ids.

## Python API

```python
import integer_model
model = integer_model.load_from_folder(directory, "cuda")  # <the path that serve.sh prints>/model
sequence = model.new_sequence()
token = int(model.extend(sequence, [17, 400, 3])[-1])  # prefill: greedy choice after the prompt
token = int(model.extend(sequence, [token])[0])  # decoding
model.free(sequence)
```

## Status

This is a research artifact.

**Tested** (more in [OPERATING.md, "What has been run"](OPERATING.md#what-has-been-run)). The
package's tests that need no GPU, no vLLM and none of the three programs pass on synthetic weights
(row `integer_model.cpu` of [SUITE.md](../../../SUITE.md)). Lean's definition of the model, compiled
and run at small sizes on made-up weights, agreed with this package in every compared value
([tests/lean_vectors/README.md](tests/lean_vectors/README.md)).

The three results below are from runs on other programs or on other sources of the package ([what
these are](OPERATING.md#what-has-been-run)): the first two on both, the third on other sources of
the package.

- On Maverick's real weights at full size (a public 4-bit GGUF conversion), vLLM served 16 greedy
  replies from the package in each of four runs (three of them told in
  [integrations/LIMITS.md](../../LIMITS.md)), and each got a proof under exact match that passed its
  check.
- In one run on those weights under a clipped-gap policy, `cp_prove` decided all 178 requests as
  the package's scores forecast: 152 got a proof and 26 got none.
- With vLLM on one H100, at small size on synthetic weights, `pytest` passed 985 tests and skipped
  10 ([tests/README.md](tests/README.md)).

**Limits:**

- These are tests of agreement, not proofs: they say nothing about soundness or weight hiding. The
  prover is compared with the package by outcomes (proof or no proof, greedy choices), not value by
  value, because the programs have no option that prints an intermediate value.
- A pass under a clipped-gap policy says only that the tokens meet the policy on the committed
  model. A reply with one token replaced can pass, and so can another model's reply, or every reply
  if the provider committed to a model whose scores are close together
  ([limits](../../../docs/limits.md#within-limits)).
- Lean's soundness theorem is about the Lean verifier at exact match, not about `cp_verify`, the C++
  verifier that users run, and not about a clipped-gap policy: [What is
  proved](../../../guide/what-is-proved.md).
- Not tested: a real tokenizer (send prompts as token ids); this package on a real model's files in
  8 or 16 bits (Q8_0, F16, BF16); at full size, ordinary serving under a clipped gap through the
  proof middleware.
- What the package prints or writes from a model's files is not masked. With weights that must stay
  secret, read [Weight hiding](../../../guide/weight-hiding.md) and [OPERATING.md, "Threat
  model"](OPERATING.md#threat-model) first.

## A request from the model's own reply

A request file holds a window of token ids for the provers and says which of them a proof covers.
These lines make such a file on a CPU, from this directory, for the model to which
`prover/examples/run_target_example.sh target_small_f2_roomy` commits (a window of 64 tokens, on the
weights that `cp_commit` makes with `-weights made-up-blocks`). Run them in one shell:

```sh
W=$(mktemp -d) && echo "W=$W" && E=../../../prover/examples/target_small_f2_roomy &&
cp "$E/policy.txt" "$E/KL.bin" "$W" &&
echo '{"weights": "blocks", "weight_settings": {"blocks": "synthetic"}}' \
  > "$W/integer_model.json" &&
PYTHONPATH=$PWD python3 -B -c 'import json, sys
import integer_model
model = integer_model.load_from_folder(sys.argv[1], "cpu")
sequence = model.new_sequence()
prompt = [int(t) for t in sys.argv[2].split(",")]
reply = [int(model.extend(sequence, prompt)[-1])]
while len(reply) < int(sys.argv[3]):
    reply.append(int(model.extend(sequence, [reply[-1]])[0]))
model.free(sequence)
json.dump({"prompt_token_ids": prompt, "reply_token_ids": reply}, open(sys.argv[4], "x"))
print(",".join(map(str, reply)))' \
  "$W" 243,486,218,461,192,436,167,411,142,385,117,360,91,335,66,310 49 "$W/reply.json"
```

`$W` is then the model folder. The arguments after `$W` are the prompt's ids, the number of reply
tokens and a new file; put your own prompt in place of these 16 ids. The lines print the 49 ids of
the reply, the last of which is 284. Prompt and reply together may have at most one token more than
the window, or `extend` raises `WindowFull`.

Then write the request file. `T` is the window, 2 to the power of the line `lt` of `policy.txt`:

```sh
request() { python3 -I -c 'import json, struct, sys
one, T = json.load(open(sys.argv[1])), int(sys.argv[2])
p, a = one["prompt_token_ids"], one["reply_token_ids"]
assert p and a and len(p) + len(a) <= T + 1, "prompt and reply need a token each, and together at most T + 1"
words = struct.pack("<%dI" % (T + 2), *(p + a + [0] * T)[:T], len(p) - 1, len(p) + len(a) - 1)
open(sys.argv[3], "xb").write(words)' "$1" "$2" "$3"; }
test -d "${W:?first run the block above, or set W to the path that it printed}" &&
T=$((1 << $(sed -n 's/^lt //p' "$W/policy.txt"))) &&
request "$W/reply.json" "$T" "$W/reply.req"
```

These lines make a request file whose sha256 is
55655047a089095c99fac8cd13c30481c011c13c1816f27874705098220fbbe5.

The file, `NAME.req`, holds T + 2 little-endian 32-bit words: the window (the prompt's n ids, the
reply's m ids, then token 0 as padding up to T tokens), then r = n - 1 and e = n + m - 1. A proof
covers the positions r to e - 1, each about the token after it: exactly the reply. n + m may be at
most T + 1; then the reply's last token follows the window and is the token that `cp_prove`
announces in `NAME.y`: here 16 + 49 = 65, so y is 284.

Under exact match `cp_prove` is built to make a proof only if every covered token is the committed
integer model's first choice at its position, under the weights, `KL.bin` and `policy.txt` of the
commitment, and the range conditions hold at all T positions, the padding's too. Otherwise the
request gets `NAME.failed` and no proof.

## What is where

- `integer_model/model.py` (start here), `integer_model_vllm/` (plug-in),
  [`examples/`](examples/README.md), `tests/`.
- [OPERATING.md](OPERATING.md): full size, GGUF files, verification, settings, leaks, exactness,
  what has been run.
- [docs/quickstart.md](../../../docs/quickstart.md): Maverick, download to proof.
  [THIRD_PARTY.md](../../../THIRD_PARTY.md) and [PUBLISHED-WORKS.md](../../../PUBLISHED-WORKS.md):
  earlier public work. [LICENSE](LICENSE): Apache 2.0. [NOTICE](NOTICE): third-party notices.

## Third-party work in this package

The licences of these works, with their copyright lines, are in [NOTICE](NOTICE). THIRD_PARTY.md, at
the top of the source repository (in a checkout, three folders above this file), has their entries;
licenses/BLAKE2-54f4faa4.COPYING lies there too. What each is used for, and where its text comes
from:

- **ggml** (the tensor library of the llama.cpp project). Used in integer_model/block_weights.py
  (decoded_values: the block formats and their decoding, with the four expressions of
  get_scale_min_k4) and tests/test_block_decoding.py (blocks packed in that layout; exact_values,
  with the same four expressions written with remainders and quotients). The licence text in NOTICE
  is the file LICENSE of llama.cpp, unmodified, as carried in llama-cpp-python 0.3.7, whose
  changelog names the commit 794fe23f29fb40104975c91fe19f23798f7c726e.
- **gguf-py** (the llama.cpp project's Python package gguf, release 0.19.0). Used in
  integer_model/block_weights.py (decoded_values is adapted from gguf/quants.py; the reader of GGUF
  files takes the numbers of gguf/constants.py and the field order of gguf/gguf_reader.py).
- **VerInf** (J. Petrie, commit 89903c96). Used in integer_model/block_weights.py
  (committed_integers: the conversion rule grew out of VerInf's).
- **Hugging Face transformers** (version 5.17.0). Used in integer_model/model.py (the rotary step's
  pairs; the loop over experts, for which modeling_mixtral.py is prior art), integer_model/tables.py
  (the rotary angles) and integer_model/float_export.py (names, shapes and configuration fields).
  NOTICE also names src/transformers/integrations/gguf/dequant.py: a comparison found a few
  statements of the decoders in integer_model/block_weights.py equal to that file's but for names or
  language, and no taking is known.
- **vLLM** (release 0.29.0). Used in integer_model_vllm/, a plug-in for vLLM whose __init__.py and
  loader.py follow examples that vLLM publishes and whose model.py has the names and arguments that
  vLLM requires of a model class, in tests/cpu_only.py, in tests/test_vllm_model_class.py (two
  statements adapted from vllm/v1/worker/gpu/sample/penalties.py), in tests/vllm_tools.py
  (shut_down() ends an engine with two internal calls of vLLM, as, by its docstring, vLLM's own
  tests do), and in integer_model/noise.py (vllm/v1/worker/gpu/sample/gumbel.py is a source of the
  noise's rule: see Triton).
- **Triton** (release 3.7.1). Used in integer_model/noise.py: steps 1 and 3 of the noise's rule, and
  a part of step 2, aim to match the noise that the V2 sampler of vLLM 0.29.0 generates for a
  request's seed; their sources are triton/language/random.py and vLLM's gumbel.py (docs/model.md,
  "12 The noise of a seeded score").
- **Random123** (release 1.14.0). NOTICE has its entry. It is not confirmed that nothing was taken
  from include/Random123/philox.h, the code of the authors of the paper on Philox, and no taking is
  known: beside that file philox4x32_10 in integer_model/noise.py, which does not name Random123,
  has the same steps in other statements, and its two key steps stand in that file and are printed
  nowhere in the paper that its docstring cites. That paper is Salmon and others, SC '11 (section
  4.3). The three known answers in tests/test_seeded_score.py are meant to be those of Random123's
  file kat_vectors; they were compared with that file of Random123 1.14.0, by program, once, and all
  three are equal.
- **libcu++ (libcudacxx) in CCCL** (release 3.3.2). NOTICE has its entry. It is not confirmed that
  nothing was taken from libcudacxx/include/cuda/std/__random/philox_engine.h, and no taking is
  known: beside that file philox4x32_10 in integer_model/noise.py, which does not name libcudacxx,
  has the same steps in other statements, with 2 of the 5 statements of the round and its loop equal
  but for language and the 5 in that file's order, and by a second reading the round lies at the
  border of equal but for names or language.
- **torchao** (release 0.10.0). Used in integer_model/model.py (_rotary has the view-rotate-stack
  form of apply_rotary_emb).
- **gpt-fast**. _rotary in integer_model/model.py is like gpt-fast's model.py in form: both view the
  last axis as neighbouring pairs, rotate each by the rotary angle and stack them into one axis.
  _rotary's cosines and sines are integers from two tables.
- **splitmix64.c** (by Sebastiano Vigna). Used in integer_model/synthetic_weights.py (mix64), which
  block_weights.py calls for the synthetic model.
- **BLAKE2 reference source code package** (blake2b-ref.c, blake2.h, blake2-impl.h). Used in
  integer_model/tables.py (_rotr64 to _compress are adapted from blake2b-ref.c). 54f4faa4 is the
  first eight digits of the commit 54f4faa4c16ea34bcd59d16e8da46a64b259fc07 of the BLAKE2 reference
  package that Qt 6.4.2 says it bundles. licenses/BLAKE2-54f4faa4.COPYING is the file COPYING that
  Qt 6.4.2 bundles with its copies of the three files (src/3rdparty/blake2/), unmodified.
