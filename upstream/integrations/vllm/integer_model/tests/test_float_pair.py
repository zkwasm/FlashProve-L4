# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of integrations/vllm/integer_model/integer_model/float_export.py, which writes an integer
model's floating-point counterpart as a Llama 4 checkpoint, and of the example small_float_pair.
Hugging Face transformers' Llama 4 code runs the checkpoint on the CPU. One test, marked vllm and
gpu, serves it with vLLM.
"""

import json
import math
import os
import shlex
import statistics
import subprocess
import sys

import pytest
import torch
import test_examples
from test_examples import PACKAGE, REPOSITORY, make_example, printed, run

import integer_model
from integer_model.float_export import float_tensors, write_float_model
from integer_model_vllm import float_pair

pytest.importorskip("safetensors")
programs_that_print = test_examples.programs_that_print  # a fixture

SATURATING = REPOSITORY / "prover" / "examples" / "target_small_f2"
REPLY_LENGTH = 16


# 512: the vocabulary size of both configurations used here.
def prompts(number):
    return [[((37 + 2 * j) * i + 11 + 5 * j) % 512 for i in range(16)] for j in range(number)]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """The example's directory, created with the command line, and the command's standard error."""
    top = tmp_path_factory.mktemp("pair") / "new"
    done = make_example("small_float_pair", "--directory", top)
    assert (done.returncode, done.stdout) == (0, f"{top}\n"), done.stderr
    return top, done.stderr


@pytest.fixture(scope="module")
def integers(pair):
    return integer_model.load_from_folder(pair[0] / "model", "cpu")


def load_with_transformers(directory):
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        directory, attn_implementation="eager"
    )
    assert type(model).__name__ == "Llama4ForCausalLM"
    assert model.dtype == torch.float32
    return model.eval()


@pytest.fixture(scope="module")
def floats(pair):
    return load_with_transformers(pair[0] / "float_model")


def test_the_files_of_the_example(pair):
    top, _ = pair
    assert sorted(path.name for path in top.iterdir()) == ["float_model", "model"]
    assert sorted(path.name for path in (top / "float_model").iterdir()) == [
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    ]  # fmt: skip
    stock, ours = (
        json.loads((top / d / "config.json").read_text()) for d in ("float_model", "model")
    )
    assert stock["architectures"] == ["Llama4ForCausalLM"]
    for field in ("vocab_size", "eos_token_id", "max_position_embeddings"):
        assert stock[field] == ours[field]
    for file in ("tokenizer.json", "tokenizer_config.json"):
        assert (top / "float_model" / file).read_bytes() == (top / "model" / file).read_bytes()


def test_the_configuration_is_the_shipped_one(pair):
    source, model = REPOSITORY / "prover" / "examples" / float_pair.CONFIGURATION, pair[0] / "model"
    for file in ("policy.txt", "KL.bin"):
        assert (model / file).read_bytes() == (source / file).read_bytes()


@pytest.mark.parametrize("runner", [None, "1"])
def test_serve_sh_runs_the_command_that_the_example_prints(
    runner, pair, programs_that_print, tmp_path
):
    """For this example serve.sh sets no model runner: vLLM gets the caller's
    VLLM_USE_V2_MODEL_RUNNER, or none.
    """
    script = PACKAGE / "examples" / "serve.sh"
    callers = {} if runner is None else {"VLLM_USE_V2_MODEL_RUNNER": runner}
    arguments, variables = printed(
        run(script, "small_float_pair", env=programs_that_print | callers, cwd=tmp_path)
    )
    (directory,) = (tmp_path / "tmp").iterdir()
    command = shlex.split(pair[1].split("To serve it:\n")[1].replace(str(pair[0]), str(directory)))
    assert command[4:] == arguments
    assert "--load-format" not in arguments
    assert command[:4] == [f"{name}={variables[name]}" for name in float_pair.ENVIRONMENT]
    assert variables["VLLM_PLUGINS"] == ""
    assert variables.get("VLLM_USE_V2_MODEL_RUNNER") == runner


def stored_tensors(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as file:
        return {name: file.get_tensor(name) for name in file.keys()}


def test_each_tensor_is_an_integer_matrix_times_its_constant(pair, integers):
    p, weights = integers.parameters, integers.weights
    stored = stored_tensors(pair[0] / "float_model" / "model.safetensors")
    assert all(tensor.dtype == torch.float32 for tensor in stored.values())
    seen = set()

    def take(name):
        seen.add(name)
        return stored[name].double()

    def same(found, exponent, *matrix, factor=1.0):
        """found * 2^exponent / factor is the integer matrix, exactly if factor is 1."""
        found, expected = found * 2.0**exponent / factor, weights.matrix(*matrix).double()
        torch.testing.assert_close(found, expected, rtol=0 if factor == 1.0 else 1e-7, atol=0)

    def gains(name, M, K):
        assert take(name).tolist() == pytest.approx([2.0 ** (M - K) / math.sqrt(p.d)] * p.d)

    assert torch.equal(
        take("model.embed_tokens.weight").long(), weights.embedding_rows(torch.arange(p.V))
    )
    same(take("lm_head.weight"), 0, "Wh")
    gains("model.norm.weight", p.MF, p.KF)
    for layer, k in enumerate(p.layers):
        here = f"model.layers.{layer}."
        gains(here + "input_layernorm.weight", k.M1, k.K1)
        gains(here + "post_attention_layernorm.weight", k.M2, k.K2)
        to_integers = k.kq + k.sh + p.rx - p.cs
        same(
            take(here + "self_attn.q_proj.weight"), to_integers, "Wq", layer, factor=math.sqrt(p.dh)
        )
        same(take(here + "self_attn.k_proj.weight"), k.kk - p.cs, "Wk", layer)
        same(take(here + "self_attn.v_proj.weight"), k.kv, "Wv", layer)
        same(take(here + "self_attn.o_proj.weight"), k.ko, "Wo", layer)
        block = here + ("feed_forward.shared_expert." if p.has_experts(layer) else "feed_forward.")
        same(take(block + "gate_proj.weight"), k.kg + p.eg, "Wg", layer)
        same(take(block + "up_proj.weight"), k.ku, "Wu", layer)
        # 6: the SiLU table holds 64 times SiLU.
        same(take(block + "down_proj.weight"), k.kd - 6, "Wd", layer)
        if p.has_experts(layer):
            # 8: the sigmoid table's key is in units of 2^-8.
            same(take(here + "feed_forward.router.weight"), k.kr + 8, "Wr", layer)
            gate_up = take(here + "feed_forward.experts.gate_up_proj")
            down = take(here + "feed_forward.experts.down_proj")
            assert gate_up.shape == (p.NE, p.d, 2 * p.fe)
            assert down.shape == (p.NE, p.fe, p.d)
            for e in range(p.NE):
                same(gate_up[e, :, : p.fe].T, k.kgR + p.eg, "Wgr", layer, e)
                same(gate_up[e, :, p.fe :].T, k.kuR, "Wur", layer, e)
                same(down[e].T, k.kd - 6, "Wdr", layer, e)
    assert seen == set(stored)


# extend() records the logits under the name "lam".
def integer_logits(integers, tokens):
    found = {}
    sequence = integers.new_sequence()
    integers.extend(
        sequence, tokens, record=lambda name, layer, t: found.__setitem__(name, t.double())
    )
    integers.free(sequence)
    return found["lam"]


def test_transformers_computes_what_the_integer_model_computes_up_to_rounding(integers, floats):
    errors, same_choice = [], 0
    for prompt in prompts(4):
        tokens = (prompt * 4)[: integers.parameters.T]
        lam = integer_logits(integers, tokens)
        with torch.no_grad():
            logits = floats(torch.tensor([tokens])).logits[0].double()
        # At each position: the root-mean-square difference, in standard deviations of lam.
        errors += ((logits - lam).pow(2).mean(dim=1).sqrt() / lam.std(dim=1)).tolist()
        same_choice += int((logits.argmax(dim=1) == lam.argmax(dim=1)).sum())
    # The median is about 0.01. A wrong constant in one layer gives 0.06.
    assert statistics.median(errors) < 0.025
    assert same_choice >= 0.9 * len(errors)


def test_where_the_integer_model_saturates_the_two_models_differ(tmp_path):
    """Under target_small_f2's scales the two models' top choices agree at no more than 4 of 16
    positions. The float model has no clamps and no table cut-offs (SiLU, softmax), so the two are
    close only where the integer model does not saturate.
    """
    p = integer_model.Parameters.read(SATURATING / "policy.txt")
    weights = integer_model.SyntheticWeights(p)
    write_float_model(p, weights, tmp_path, end_token=p.V - 1)
    (prompt,) = prompts(1)
    lam = integer_logits(integer_model.IntegerModel(p, weights, "cpu"), prompt)
    with torch.no_grad():
        logits = load_with_transformers(tmp_path)(torch.tensor([prompt])).logits[0]
    assert int((logits.argmax(dim=1) == lam.argmax(dim=1)).sum()) <= 4


def gaps_of(integers, prompt, reply):
    """The reply tokens' logit gaps. Also evaluates the rest of the window, padded with token 0,
    because a proof needs the range conditions at every position of its window.
    """
    tokens = prompt + reply
    sequence = integers.new_sequence()
    _, gaps = integers.extend(sequence, tokens[:-1], gap_for=tokens[1:])
    integers.extend(sequence, tokens[-1:] + [0] * (integers.parameters.T - len(tokens)))
    integers.free(sequence)
    return [int(gap) for gap in gaps[len(prompt) - 1 :]]


def meets_the_policy(pair, gaps):
    """The clipped-gap acceptance policy, as integrations/vllm/integer_model/examples/score_reply.py
    computes it, from the lines of the example's parameter file (policy.txt).
    """
    lines = (pair[0] / "model" / "policy.txt").read_text().splitlines()
    policy = dict(line.split(maxsplit=1) for line in lines)
    C, theta, cap, sh = (int(policy[name]) for name in ("C", "theta", "cap", "sh"))
    costs = [gap + (1 << sh) for gap in gaps]
    return sum(min(c, C) for c in costs) <= theta * len(costs) and sum(c > C for c in costs) <= cap


def check_replies(pair, integers, replies):
    """Asserts that each prompt's reply meets the policy, that the next prompt's reply in its place
    does not, and that the median of the replies' mean gaps is at most 2^15.

    A wrong model, with the router's score applied twice, was found to meet the policy with every
    reply, but not this bound on the median.
    """
    given = prompts(len(replies))
    gaps = [gaps_of(integers, prompt, reply) for prompt, reply in zip(given, replies)]
    assert all(meets_the_policy(pair, g) for g in gaps)
    for i, prompt in enumerate(given):
        assert not meets_the_policy(
            pair, gaps_of(integers, prompt, replies[(i + 1) % len(replies)])
        )
    assert statistics.median(sum(g) / len(g) for g in gaps) <= 1 << 15


def test_replies_of_transformers_meet_the_policy_and_swapped_replies_fail(pair, integers, floats):
    sequences = prompts(8)
    for _ in range(REPLY_LENGTH):
        with torch.no_grad():
            choices = floats(torch.tensor(sequences)).logits[:, -1].argmax(dim=1).tolist()
        for sequence, choice in zip(sequences, choices):
            sequence.append(choice)
    check_replies(pair, integers, [sequence[-REPLY_LENGTH:] for sequence in sequences])


# Runs in a new process, because vLLM reads VLLM_PLUGINS once. vllm.LLM gets the engine options of
# the example's command line, and a smaller share of the GPU's memory.
SERVE_WITH_VLLM = """
import json, sys, vllm
llm = vllm.LLM(model=sys.argv[1], dtype="bfloat16", max_model_len=64, moe_backend="triton",
               enforce_eager=True, generation_config="vllm", gpu_memory_utilization=0.2)
sampling = vllm.SamplingParams(temperature=0, max_tokens=int(sys.argv[2]), ignore_eos=True)
served = llm.generate([{"prompt_token_ids": ids} for ids in json.loads(sys.argv[3])], sampling)
print("replies:", json.dumps([list(one.outputs[0].token_ids) for one in served]))
"""


@pytest.mark.vllm
@pytest.mark.gpu
def test_replies_of_vllm_meet_the_policy_and_swapped_replies_fail(pair, integers, tmp_path):
    command = [sys.executable, "-c", SERVE_WITH_VLLM, pair[0] / "float_model", str(REPLY_LENGTH)]
    done = subprocess.run(
        [*command, json.dumps(prompts(16))],
        env={**os.environ, **float_pair.ENVIRONMENT},
        capture_output=True,
        text=True,
        cwd=tmp_path,  # an empty folder of the test's own: nothing of the caller's can be imported
    )
    assert done.returncode == 0, done.stderr[-4000:]
    (line,) = (line for line in done.stdout.splitlines() if line.startswith("replies: "))
    check_replies(pair, integers, json.loads(line.removeprefix("replies: ")))


def test_what_write_float_model_rejects(integers, tmp_path):
    p, weights = integers.parameters, integers.weights
    with pytest.raises(ValueError, match="float32 or torch.bfloat16, not torch.float16"):
        write_float_model(p, weights, tmp_path, end_token=0, dtype=torch.float16)
    for end_token in (-1, p.V):
        with pytest.raises(ValueError, match=f"end_token must be in 0 .. 511, not {end_token}"):
            write_float_model(p, weights, tmp_path, end_token=end_token)
    assert list(tmp_path.iterdir()) == []
    for name in ("config.json", "model.safetensors"):
        directory = tmp_path / f"with_{name}"
        directory.mkdir()
        (directory / name).write_text("theirs")
        with pytest.raises(FileExistsError, match=f"{name} exists already"):
            write_float_model(p, weights, directory, end_token=0)
        assert [(path.name, path.read_text()) for path in directory.iterdir()] == [(name, "theirs")]


def test_bfloat16_rounds_each_entry_once(integers, tmp_path):
    p, weights = integers.parameters, integers.weights
    write_float_model(p, weights, tmp_path, end_token=0, dtype=torch.bfloat16)
    stored = stored_tensors(tmp_path / "model.safetensors")
    assert json.loads((tmp_path / "config.json").read_text())["torch_dtype"] == "bfloat16"
    for name, tensor in float_tensors(p, weights):
        assert torch.equal(stored[name], tensor.to(torch.bfloat16))
