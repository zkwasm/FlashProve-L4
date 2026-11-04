# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""The example small_float_pair: one small synthetic model, in integers and in floating point.

It is for a provider who serves a floating-point model and afterwards proves one forward pass of the
committed integer model over prompt and reply, under a clipped-gap acceptance policy, which lets the
reply differ, within limits, from the integer model's top choices. The runs that have served a
floating-point model and proved its replies are in docs/results.md. make_float_pair() creates
model/, the integer model, on which integrations/vllm/integer_model/examples/score_reply.py scores a
reply, and float_model/, a float32 Llama 4 checkpoint that vLLM serves without this package's
plug-in.

The limits were fitted to this example's synthetic weights only: they show nothing about the
thresholds of a real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import integer_model
from integer_model.float_export import write_float_model

from .example import copy_new, new_file, sizes, write_vllm_files

# CONFIGURATION names the directory under prover/examples that gives policy.txt and KL.bin, the
# quantization scales. WEIGHTS selects the synthetic blocks that `cp_commit` generates under
# `-weights made-up-blocks`. Under these scales, on 8 windows, no key of the SiLU table, taken
# before its clamp (`gw` in docs/model.md, "7 Feed-forward"), is past the table's end, and the
# floating-point model's top choice, in float32, is the integer model's at 501 of the 512
# positions. Under target_small_f2's scales the same weights saturate: 43 % of these keys are past
# the end, and the top choices agree at 23 of 512 positions. The lines `C`, `theta` and `cap` of
# policy.txt are the acceptance policy's limits; the directory's README.md defines them, and says
# how integrations/vllm/integer_model/examples/try_limits.py fitted them and what they accept.
CONFIGURATION = "target_small_f2_gap"
WEIGHTS = {"weights": "blocks", "weight_settings": {"blocks": "synthetic"}}

# An empty VLLM_PLUGINS loads no plug-in, this package's included. VLLM_USE_V2_MODEL_RUNNER is left
# to the caller.
ENVIRONMENT = {
    "VLLM_PLUGINS": "",
    "VLLM_NO_USAGE_STATS": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "PYTHONSAFEPATH": "1",
}


def make_float_pair(top: Path, repository: Path) -> list[str]:
    """Creates model/ and float_model/ in top. Returns the command that serves float_model."""
    model, float_model = top / "model", top / "float_model"
    model.mkdir()
    for file in ("policy.txt", "KL.bin"):
        copy_new(repository / "prover" / "examples" / CONFIGURATION / file, model)
    new_file(model / "integer_model.json", json.dumps(WEIGHTS) + "\n")
    vocabulary_size, window_length = sizes(model / "policy.txt")
    end_token = vocabulary_size - 1
    write_vllm_files(model, vocabulary_size, window_length, end_token)

    float_model.mkdir()
    loaded = integer_model.load_from_folder(model, "cpu")
    write_float_model(loaded.parameters, loaded.weights, float_model, end_token)
    for file in ("tokenizer.json", "tokenizer_config.json"):
        copy_new(model / file, float_model)
    # `--moe-backend triton`: with vLLM 0.29.0's Triton kernels for experts, the router's score
    # multiplies the expert's input once, as in the integer model.
    return [
        *(f"{variable}={value}" for variable, value in ENVIRONMENT.items()),
        *("vllm", "serve", str(float_model), "--host", "127.0.0.1"),
        *("--served-model-name", "small_float_pair", "--dtype", "bfloat16"),
        *("--max-model-len", str(window_length), "--moe-backend", "triton", "--enforce-eager"),
        *("--generation-config", "vllm"),
    ]
