# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The float32 forward pass of a Llama 4 text model, from token ids to the final norm.

scales/calibrate.py runs this pass over a calibration set to learn how large the model's activations
get, so that scales/fit_scales.py can choose the quantization scales of the integer model (the
network computed in integers, which the proofs are about). A calibration set is a list of windows,
and a window is one sequence of tokens. forward() passes the activation at each site (one of twelve
named points of the forward pass, listed in its docstring) to a hook, a function that the caller
supplies. The arithmetic is in float32, on the weights as scales/gguf_weights.py decodes them from
the model's GGUF files.

The pass follows the text model of Hugging Face's library transformers, version 5.17.0
(github.com/huggingface/transformers, src/transformers/models/llama4/modeling_llama4.py), step by step and is adapted
from it. Single statements are the same: the body of rms_norm, the rotary frequencies and the query factor. The comments
below cite that file as "HF" with line numbers. transformers has the Apache License, version 2.0, and the file has the
header "Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved." THIRD_PARTY.md, "Hugging Face
transformers", lists what this repository has of it.

This pass evaluates the model under these settings of configuration_llama4.py (in the same
directory as HF). Nothing in this file checks that a model has them.

- No biases (attention_bias = False) and SiLU gates (hidden_act = "silu").
- One routed expert per token (num_experts_per_tok = 1).
- No normalization of queries and keys (use_qk_norm = False). HF 362 says "the 128E model does not use qk_norm", and
  the 128E model is the public model with 128 experts, Llama 4 Maverick.
- The query factor on the layers without rotary encoding: each query is multiplied by a number
  that depends on its position (attn_temperature_tuning = True, floor_scale = 8192,
  attn_scale = 0.1, the defaults in configuration_llama4.py, lines 171-173).
- Rotary frequencies without rescaling (rope_type "default"). The pass does not read the tensor rope_freqs.weight, in
  which the converter of llama.cpp stores the factors of the "llama3" kind of rescaling (convert_hf_to_gguf.py, lines
  1671-1698, of llama.cpp, github.com/ggml-org/llama.cpp, as carried in llama-cpp-python 0.3.7, whose changelog names
  the commit 794fe23f29fb40104975c91fe19f23798f7c726e). The integer model's rotary tables have no rescaling either
  (rotary_tables() in scales/policy_tables.py).

The public model also limits attention to chunks of 8,192 tokens on the layers with rotary encoding
(attention_chunk_size = 8192). This pass does not. The two are the same function on a window of at
most 8,192 tokens, and as_windows() rejects a longer one.

No program in this repository compares the results of this pass with those of transformers.
scales/tests/test_calibrate.py compares them, site by site, with a second evaluation of the same
steps in numpy, on tiny models with pseudo-random weights.
"""

# Public description of the rotary layers, the chunk length and the normalization of queries and keys (named there for
# Llama 4 Scout only): Hugging Face, "Welcome Llama 4 Maverick & Scout on Hugging Face",
# huggingface.co/blog/llama4-release, April 5, 2025.
# Public description of the query factor: vLLM's API documentation, version 0.9.2, module
# vllm.model_executor.models.llama4 (docs.vllm.ai/en/v0.9.2/api/vllm/model_executor/models/llama4.html), class
# Llama4Attention, method _get_attn_scale.
import torch
from torch.nn.functional import linear, silu

LONGEST_WINDOW = 8192  # attention_chunk_size: up to this many tokens the public model's chunks do not matter
QUERY_FACTOR_PERIOD = 8192  # floor_scale
QUERY_FACTOR_SLOPE = 0.1  # attn_scale


class ForwardError(ValueError):
    """The pass cannot evaluate these tokens. The message is for the user."""


# The keys of `shape`, as gguf_weights.shape_words() returns them: d is the width of the residual
# stream ("the stream" in this file), L the number of layers, H and G the numbers of query heads and
# of key and value heads, dh the head dimension, f and fe the hidden widths of a dense layer and of
# one expert, NE the number of routed experts and V the vocabulary size. eps is the norms' epsilon,
# theta the base of the rotary frequencies, and moe and rope list the layers with routed experts and
# with rotary encoding. The first nine are named like the lines of the integer model's
# parameter file (docs/model.md, "2 Shape parameters").
def tensor_shapes(shape):
    """Return {name: dimensions} for the tensors that forward() reads in a model of this shape.

    The dimensions are in torch's order, the reverse of a GGUF file's. A layer's tensor has the name
    blk.<layer>.<kind>.weight, and the comments in this file call it by its kind, such as attn_q.
    """
    d, f, fe, experts = shape["d"], shape["f"], shape["fe"], shape["NE"]
    queries, keys = shape["H"] * shape["dh"], shape["G"] * shape["dh"]
    shapes = {"token_embd.weight": (shape["V"], d), "output_norm.weight": (d,)}
    for layer in range(shape["L"]):
        kinds = {"attn_norm": (d,), "attn_q": (queries, d), "attn_k": (keys, d), "attn_v": (keys, d)}
        kinds.update({"attn_output": (d, queries), "ffn_norm": (d,)})
        if layer in shape["moe"]:
            kinds.update({"ffn_gate_shexp": (fe, d), "ffn_up_shexp": (fe, d), "ffn_down_shexp": (d, fe)})
            kinds.update({"ffn_gate_exps": (experts, fe, d), "ffn_up_exps": (experts, fe, d)})
            kinds.update({"ffn_down_exps": (experts, d, fe), "ffn_gate_inp": (experts, d)})
        else:
            kinds.update({"ffn_gate": (f, d), "ffn_up": (f, d), "ffn_down": (d, f)})
        shapes.update({f"blk.{layer}.{kind}.weight": dims for kind, dims in kinds.items()})
    return shapes


def as_windows(tokens, shape):
    """Return token ids as an int64 tensor of the shape (windows, tokens per window).

    tokens is an integer array or tensor of that shape, or one window. The function raises
    ForwardError for tokens that forward() cannot evaluate on a model of this shape.
    """
    tokens = torch.as_tensor(tokens)
    if tokens.dim() == 1:
        tokens = tokens[None]
    if tokens.is_floating_point() or tokens.is_complex() or tokens.dtype == torch.bool:
        raise ForwardError("the token ids are not integers")
    if tokens.dim() != 2 or tokens.numel() == 0:
        raise ForwardError("the token ids do not have the shape (windows, tokens per window) with at least one token")
    if tokens.shape[1] > LONGEST_WINDOW:
        raise ForwardError(f"a window has {tokens.shape[1]} tokens, and the pass evaluates at most {LONGEST_WINDOW}")
    # The stream inherits the memory layout of the ids, and the order in which a product is summed
    # can depend on it. A contiguous copy makes the result independent of how the caller's array is
    # stored.
    tokens = tokens.to(torch.int64).contiguous()
    if int(tokens.min()) < 0 or int(tokens.max()) >= shape["V"]:
        raise ForwardError(f"a token id lies outside the vocabulary, which is 0 to {shape['V'] - 1}")
    return tokens


def rms_norm(x, eps):
    """Divide each row of x by the square root of (the mean of its squares + eps).

    The line follows Llama4TextRMSNorm._norm (HF 130-131). The caller applies the gain (the norm's
    weight vector), because the sites n1, n2 and nF are reported before it.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rotate(x, cos, sin):
    """Apply the rotary position encoding to queries or keys.

    x has the shape (windows, tokens, heads, dh), and cos and sin have the shape (tokens, dh / 2).
    In each head the neighbouring entries 2i and 2i + 1 are the real and the imaginary part of a
    complex number, which is multiplied by cos[t, i] + 1j * sin[t, i]
    (apply_rotary_emb, HF 233-242).

    Neighbouring entries come from neighbouring rows of attn_q or attn_k in the model's file. If a
    converter has permuted the rows, the pairs are wrong, and nothing in this file detects it (more
    in scales/README.md, "What the tools do not check").
    """
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:, None, :], sin[:, None, :]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def attend(q, k, v):
    """Return the output of causal softmax attention in one window, with the shape (tokens, H * dh).

    q has the shape (tokens, H, dh), and k and v have the shape (tokens, G, dh). Query head h reads
    key and value head h // (H / G) (repeat_kv, HF 245-254). The scores are q . k / sqrt(dh)
    (eager_attention_forward, HF 271-277; the scaling, HF 320).
    """
    tokens, heads, dh = q.shape
    groups = k.shape[1]
    q = q.reshape(tokens, groups, heads // groups, dh).permute(1, 2, 0, 3)  # (G, H / G, tokens, dh)
    k = k.permute(1, 2, 0)[:, None]  # (G, 1, dh, tokens)
    v = v.permute(1, 0, 2)[:, None]  # (G, 1, tokens, dh)
    later = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).triu(1)  # [i, j]: key j comes after query i
    scores = (torch.matmul(q, k) * dh**-0.5).masked_fill(later, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v).permute(2, 0, 1, 3).reshape(tokens, heads * dh)


def gated(x, gate, up, down):
    """Evaluate the feed-forward block of a dense layer, of the shared expert or of a routed expert.

    The result is three tensors: the block's output down (silu(g) * u), the gate projection g = gate x, and the up
    projection u = up x (Llama4TextMLP.forward, HF 101-103; Llama4TextExperts.forward, HF 79-82).
    """
    g, u = linear(x, gate), linear(x, up)
    return linear(silu(g) * u, down), g, u


# What forward() computes and on what values: scales/README.md, "The floating-point pass".
@torch.no_grad()
def forward(model, shape, tokens, hook, device="cpu", chunk=4):
    """Run all windows through the model, report the activation at each site, and return the final norm's output.

    model   the result of gguf_weights.open_model(), or any object with its method
            tensor(name, expert=None, device="cpu"), which returns a float32 tensor on the device;
            its exceptions pass through
    shape   the result of gguf_weights.shape_words(), or any dict with its keys
    tokens  token ids of the shape (windows, tokens per window), or one window; ForwardError is
            raised for ids that as_windows() rejects
    hook    hook(site, layer, tensor) gets a float32 tensor on the device, whose last dimension (at
            q, k and v: the last two) runs over the channels. For each site and layer at which it is
            called, it gets every token's values exactly once, in one or more calls. A call at the
            site `stream` has all windows. A call at gate_routed or uR has the tokens that chose one
            expert. A call at any other site has at most `chunk` windows. The hook must not change
            the tensor, which forward() may still use, and must copy what it needs to keep:
            forward() updates the stream in place, and the result overwrites it.
    device  the torch device that computes
    chunk   how many windows go through a layer at a time, a positive integer. A smaller number
            saves memory for most activations, but not for the stream of all windows and not for a
            routed expert, which takes all its tokens at once. It changes the result only by
            floating-point rounding.

    These are the sites, in the order of the calls in a layer. scales/calibrate.py records the eight
    marked with a star, which scales/fit_scales.py needs. Only scales/tests/test_calibrate.py reads
    the other four.

    n1 *         the stream after the first norm, before the gain attn_norm
    q *          the queries after the rotary encoding or the query factor, not divided by sqrt(dh)
    k *          the keys, after the rotary encoding if the layer has it
    v *          the values
    n2 *         the stream after the second norm, before the gain ffn_norm
    gate         the gate projection of the dense layer or of the shared expert, before SiLU
    u *          the up projection of the dense layer or of the shared expert
    router       the router's logits (mixture-of-experts layers only)
    gate_routed  the gate projection of each token's routed expert (mixture-of-experts layers only)
    uR *         the up projection of each token's routed expert (mixture-of-experts layers only)
    stream       the stream after the layer
    nF *         the stream after the final norm, before the gain output_norm, with layer = L

    The result has the shape (windows, tokens per window, d). The function does not compare the
    model's tensors with tensor_shapes(shape), and does not check that H is a multiple of G and dh
    is even, which attend() and rotate() need. check_tensors() in scales/calibrate.py does all this.
    """
    dh, layers, eps = shape["dh"], shape["L"], shape["eps"]
    tokens = as_windows(tokens, shape).to(device)

    # A token's stream starts as its row of the embedding, without scaling (Llama4TextModel.forward, HF 527).
    x = model.tensor("token_embd.weight", device=device)[tokens]  # (windows, tokens, d)
    parts = x.split(chunk)  # views of x, at most `chunk` windows each: adding to a part adds to x

    # Positions count from 0 in each window. The angle of pair i at position t is t * theta^(-2i / dh), in float32
    # (Llama4TextRotaryEmbedding, HF 214 and 225-227).
    position = torch.arange(tokens.shape[1], device=device, dtype=torch.float32)
    frequency = 1.0 / shape["theta"] ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh)
    angle = torch.outer(position, frequency)
    cos, sin = torch.cos(angle), torch.sin(angle)

    # The query factor 1 + 0.1 ln(1 + floor((t + 1) / 8192)), which Llama4TextAttention.forward (HF 366-374) calls
    # temperature tuning. HF 366 names its source: K. M. Nakanishi, "Scalable-Softmax Is Superior for Attention",
    # arXiv:2501.19399. The factor is exactly 1 at every position but the last of a window of 8,192 tokens.
    query_factor = torch.log1p(torch.floor((position + 1.0) / QUERY_FACTOR_PERIOD)) * QUERY_FACTOR_SLOPE + 1.0

    # To fit a large model into memory, forward() reads a layer's weights, sends all windows through
    # the layer, and drops its matrices before it reads the next layer's. Only one routed expert's
    # weights are in memory at a time.
    for layer in range(layers):

        def weight(kind, expert=None):
            return model.tensor(f"blk.{layer}.{kind}.weight", expert=expert, device=device)

        # Attention: the stream x becomes x + attn_output o (Llama4TextDecoderLayer.forward, HF 426-439).
        gain, to_q, to_k, to_v = weight("attn_norm"), weight("attn_q"), weight("attn_k"), weight("attn_v")
        to_stream = weight("attn_output")
        for part in parts:
            n1 = rms_norm(part, eps)
            hook("n1", layer, n1)
            a = n1 * gain
            q = linear(a, to_q).unflatten(-1, (-1, dh))  # (windows, tokens, H, dh)
            k = linear(a, to_k).unflatten(-1, (-1, dh))  # (windows, tokens, G, dh)
            v = linear(a, to_v).unflatten(-1, (-1, dh))
            if layer in shape["rope"]:
                q, k = rotate(q, cos, sin), rotate(k, cos, sin)
            else:
                q = q * query_factor[:, None, None]
            hook("q", layer, q)
            hook("k", layer, k)
            hook("v", layer, v)
            o = torch.stack([attend(q[window], k[window], v[window]) for window in range(len(part))])
            part += linear(o, to_stream)
        del to_q, to_k, to_v, to_stream, n1, a, q, k, v, o

        # Feed-forward: with h = rms_norm(x) * ffn_norm, the stream x becomes x + feed_forward(h) (HF 441-447). In a
        # mixture-of-experts layer feed_forward(h) is the sum of the shared expert, which runs on h itself, and of one
        # routed expert (Llama4TextMoe.forward, HF 166-174).
        # Source for one routed expert for each token next to a shared expert: Meta, "The Llama 4 herd: The beginning
        # of a new era of natively multimodal AI innovation", ai.meta.com/blog/llama-4-multimodal-intelligence, April
        # 5, 2025.
        # A public listing of the layer: vLLM's API documentation (cited under the module's docstring), class
        # Llama4MoE.
        # A description in prose, without a formula: Li et al., "MetaShuffling: Accelerating Llama 4 MoE Inference",
        # PyTorch blog, pytorch.org/blog/metashuffling-accelerating-llama-4-moE-inference, May 12, 2025.
        with_experts = layer in shape["moe"]
        ending = "_shexp" if with_experts else ""
        gain = weight("ffn_norm")
        gate, up, down = weight("ffn_gate" + ending), weight("ffn_up" + ending), weight("ffn_down" + ending)
        if with_experts:
            router = weight("ffn_gate_inp")
            chosen = torch.empty_like(tokens)  # each token's routed expert
            scaled = torch.empty_like(x)  # each token's input to that expert
        for number, part in enumerate(parts):
            n2 = rms_norm(part, eps)
            hook("n2", layer, n2)
            h = n2 * gain
            out, g, u = gated(h, gate, up, down)
            hook("gate", layer, g)
            hook("u", layer, u)
            if with_experts:
                logits = linear(h, router)
                hook("router", layer, logits)
                # Each token goes to the expert with the largest logit (Llama4Router.forward,
                # HF 149). Among equal largest logits the integer model takes the lowest index
                # (docs/model.md, "7 Feed-forward", step 2), and so does torch.max: "the indices of
                # the first maximal value are returned" (documentation of torch.max, torch 2.9).
                largest, index = logits.max(dim=-1)
                # The sigmoid of the chosen logit multiplies the expert's input, and nothing scales its output
                # (HF 151 and 170-173).
                windows = slice(number * chunk, number * chunk + len(part))
                chosen[windows], scaled[windows] = index, h * torch.sigmoid(largest)[..., None]
            part += out
        del gate, up, down, n2, h, out, g, u
        if with_experts:
            for expert in torch.unique(chosen).tolist():
                mine = chosen == expert
                routed, g, u = gated(scaled[mine], weight("ffn_gate_exps", expert), weight("ffn_up_exps", expert),
                                     weight("ffn_down_exps", expert))  # fmt: skip
                hook("gate_routed", layer, g)
                hook("uR", layer, u)
                x[mine] += routed
            del router, chosen, scaled, routed, g, u
        hook("stream", layer, x)

    # The final norm (Llama4TextModel.forward, HF 568). The pass ends here and does not read the output matrix.
    gain = model.tensor("output_norm.weight", device=device)
    for part in parts:
        nF = rms_norm(part, eps)
        hook("nF", layers, nF)
        part.copy_(nF * gain)
    return x
