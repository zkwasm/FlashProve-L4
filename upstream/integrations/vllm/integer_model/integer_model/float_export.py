# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Writes an integer model's floating-point counterpart as a stock Llama 4 checkpoint.

Undoing the integer model's power-of-two scales (docs/model.md) gives an ordinary Llama 4 text
model: each float matrix is an integer matrix times a constant. write_float_model writes it for
the class Llama4ForCausalLM of Hugging Face transformers and of vLLM.

The float model has none of the integer model's roundings, clamps, range conditions or table
cut-offs (SiLU, softmax), so the two models are close only where the integer model does not saturate
(an example: integrations/vllm/integer_model/integer_model_vllm/float_pair.py).

Tensor names, shapes and configuration fields are those of Hugging Face transformers,
modeling_llama4.py and configuration_llama4.py (Apache License 2.0; see NOTICE).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import torch

from .parameters import Parameters
from .weights import WeightSource


def float_tensors(p: Parameters, weights: WeightSource) -> Iterator[tuple[str, torch.Tensor]]:
    """The checkpoint's tensors, as pairs (name, float64 tensor).

    The float model's residual stream, norm outputs, keys, values, attention outputs, up
    projections and logits are in the integer model's units. The symbols are those of
    docs/model.md, "2 Shape parameters" and "3 Weights and quantization scales".
    """

    def scaled(exponent: int, name: str, layer: int | None = None, expert: int | None = None):
        return weights.matrix(name, layer, expert).to("cpu", torch.float64) * 2.0**exponent

    def gains(M: int, K: int) -> torch.Tensor:
        # The integer norm divides by the root of the sum of squares, RMSNorm by that of the mean.
        return torch.full((p.d,), 2.0 ** (M - K) / math.sqrt(p.d), dtype=torch.float64)

    all_tokens = torch.arange(p.V, device=weights.device)
    yield "model.embed_tokens.weight", weights.embedding_rows(all_tokens).to("cpu", torch.float64)
    for layer, k in enumerate(p.layers):
        here = f"model.layers.{layer}."
        yield here + "input_layernorm.weight", gains(k.M1, k.K1)
        # The integer softmax is that of q.k / 2^(sh + rx). Stock code divides q.k by sqrt(dh).
        to_softmax = math.sqrt(p.dh) * 2.0 ** -(k.sh + p.rx)
        yield here + "self_attn.q_proj.weight", scaled(p.cs - k.kq, "Wq", layer) * to_softmax
        yield here + "self_attn.k_proj.weight", scaled(p.cs - k.kk, "Wk", layer)
        yield here + "self_attn.v_proj.weight", scaled(-k.kv, "Wv", layer)
        yield here + "self_attn.o_proj.weight", scaled(-k.ko, "Wo", layer)
        yield here + "post_attention_layernorm.weight", gains(k.M2, k.K2)
        # The SiLU table holds 64 SiLU(key / 2^eg), and the sigmoid table 2^16 sigmoid(key / 2^8).
        shared = "feed_forward.shared_expert." if p.has_experts(layer) else "feed_forward."
        yield here + shared + "gate_proj.weight", scaled(-k.kg - p.eg, "Wg", layer)
        yield here + shared + "up_proj.weight", scaled(-k.ku, "Wu", layer)
        yield here + shared + "down_proj.weight", scaled(6 - k.kd, "Wd", layer)
        if p.has_experts(layer):
            yield here + "feed_forward.router.weight", scaled(-k.kr - 8, "Wr", layer)
            # The routed experts are stacked, input index before output index, gate joined with up.
            gate_up = [
                torch.cat([scaled(-k.kgR - p.eg, "Wgr", layer, e), scaled(-k.kuR, "Wur", layer, e)])
                for e in range(p.NE)
            ]
            yield here + "feed_forward.experts.gate_up_proj", torch.stack([m.T for m in gate_up])
            down = [scaled(6 - k.kd, "Wdr", layer, e).T for e in range(p.NE)]
            yield here + "feed_forward.experts.down_proj", torch.stack(down)
    yield "model.norm.weight", gains(p.MF, p.KF)
    yield "lm_head.weight", scaled(0, "Wh")


def float_config(p: Parameters, end_token: int, dtype: torch.dtype) -> dict:
    """The content of config.json, for Llama 4's text model (model_type llama4_text).

    rms_norm_eps is eps1 / d: RMSNorm adds it to the mean of the squares, the integer norm adds eps
    to their sum. The format has one field, so eps1 / d also stands in for eps2 / d and epsF / d.
    The difference matters only for a row whose sum of squares is comparable to these eps values or
    smaller.
    """
    if p.NE == 0:
        raise ValueError("Only a model with experts (NE > 0) can be exported.")
    return {
        "architectures": ["Llama4ForCausalLM"],
        "model_type": "llama4_text",
        "torch_dtype": str(dtype).removeprefix("torch."),
        "vocab_size": p.V,
        "hidden_size": p.d,
        "num_hidden_layers": p.L,
        "num_attention_heads": p.n_h,
        "num_key_value_heads": p.n_g,
        "head_dim": p.dh,
        "intermediate_size": p.fe,
        "intermediate_size_mlp": p.f,
        "num_local_experts": p.NE,
        "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 2,
        "hidden_act": "silu",
        "rms_norm_eps": p.eps1 / p.d,
        "no_rope_layers": [k.rp for k in p.layers],  # 1 where the layer applies the rotary encoding
        "rope_theta": float(p.theta0),
        "rope_scaling": None,
        "use_qk_norm": False,
        "attn_temperature_tuning": False,
        "attention_chunk_size": 8192,  # no effect: a window has at most 1,024 positions
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": p.T,
        "bos_token_id": None,
        "pad_token_id": None,
        "eos_token_id": end_token,
    }


def write_float_model(
    parameters: Parameters,
    weights: WeightSource,
    directory: str | os.PathLike,
    end_token: int,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Writes config.json and model.safetensors into an existing directory that has neither.

    float32 holds a weight times a power of two exactly, and an embedding entry of magnitude below
    2^24. q_proj and the gains can be rounded: their constants have a square root. bfloat16 keeps 8
    significant bits of a 13-bit weight. float16 is rejected: the residual stream would overflow it.

    Needs the package safetensors. All tensors are in memory at once, so this is for small models.
    """
    from safetensors.torch import save

    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"dtype must be torch.float32 or torch.bfloat16, not {dtype}.")
    if not 0 <= end_token < parameters.V:
        raise ValueError(f"end_token must be in 0 .. {parameters.V - 1}, not {end_token}.")
    config_file, tensor_file = Path(directory, "config.json"), Path(directory, "model.safetensors")
    for file in (config_file, tensor_file):
        if file.exists():
            raise FileExistsError(f"{file} exists already.")
    config = float_config(parameters, end_token, dtype)
    tensors = {name: t.to(dtype).contiguous() for name, t in float_tensors(parameters, weights)}
    with open(config_file, "x") as f:
        json.dump(config, f, indent=1)
    with open(tensor_file, "xb") as f:
        f.write(save(tensors, metadata={"format": "pt"}))
