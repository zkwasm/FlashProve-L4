# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The integer model itself, not the tests' stand-in, inside a running vLLM engine, at two small
sizes.

vLLM loads a small configuration with synthetic weights through the package integer_model, and
serves greedy replies. The test also loads the model without vLLM, on the CPU, and checks every
reply twice:

- The reply equals that of direct_reply(), which decodes token by token.
- In one pass over the prompt and the reply together, the model's greedy choice at the position
  before each token of the reply is that token. A proof under the exact-match policy states this.

With a GPU, vLLM's copy runs on the GPU, so the test also compares the two devices.
"""

import hashlib
import json
import os
import random
import shutil
import sys
import unittest

import pytest

import integer_model
import vllm_tools

pytestmark = pytest.mark.vllm

TOP = os.path.normpath(os.path.join(vllm_tools.HERE, "..", "..", "..", ".."))


def direct_reply(model, prompt, end_token):
    """The model's greedy reply, up to end_token or a full window (T tokens). A failed range
    condition ends it with end_token, as in vLLM: see IntegerModelForCausalLM.forward().
    """
    sequence, reply = model.new_sequence(), []
    try:
        token = int(model.extend(sequence, prompt)[-1])
        while True:
            reply.append(token)
            if token == end_token or len(prompt) + len(reply) == model.parameters.T:
                return reply
            token = int(model.extend(sequence, [token])[-1])
    except integer_model.RangeConditionFailed:
        return reply + [end_token]
    finally:
        model.free(sequence)


def library_files(source, folder):
    """Puts the files that integer_model.load_from_folder() reads into `folder`."""
    os.makedirs(folder, exist_ok=True)
    for name in ("policy.txt", "KL.bin"):
        shutil.copy(os.path.join(source, name), folder)
    with open(os.path.join(folder, "integer_model.json"), "w") as f:
        json.dump({"weights": "synthetic"}, f)
    return folder


@pytest.fixture(scope="class")
def both_copies(request, vllm, tmp_path_factory):
    """For all tests of a class: the model that is called directly, vLLM's engine with the same
    model, the prompts, the end token.
    """
    cls, work = request.cls, str(tmp_path_factory.mktemp(request.cls.__name__))
    cls.vllm = vllm
    source = os.path.join(TOP, *cls.FOLDER)
    cls.model = integer_model.load_from_folder(
        library_files(source, os.path.join(work, "direct")), "cpu"
    )
    V, T = cls.model.parameters.V, cls.model.parameters.T
    # Prompts: random tokens and, with SPREAD, the beginnings of one sequence spread over the
    # vocabulary.
    draw = random.Random(1)
    cls.prompts = [
        [draw.randrange(V) for _ in range(length)] for length in (1, 2, T // 4, T // 2, T - 1)
    ]
    if cls.SPREAD:
        tokens = [(k * 31153 // 128) % V for k in range(1, T + 1)]
        cls.prompts += [tokens[:length] for length in (1, 3, T // 2, T - 2)]
    # A synthetic model has no end-of-sequence token. The test takes the third token of the model's
    # first reply, so that some replies end with it.
    cls.end_token = direct_reply(cls.model, cls.prompts[0], None)[2]
    folder = library_files(
        source, vllm_tools.model_folder(os.path.join(work, "served"), V, T, cls.end_token)
    )
    options = {}
    if cls.DIGESTS:
        options["sha256"] = {}
        for name, where in (
            ("integer_model.json", folder),
            ("policy.txt", source),
            ("KL.bin", source),
        ):
            with open(os.path.join(where, name), "rb") as f:
                options["sha256"][name] = hashlib.sha256(f.read()).hexdigest()
    cls.llm = vllm.LLM(
        **vllm_tools.engine_arguments(
            folder, max_model_len=T, model_loader_extra_config=options, gpu_memory_utilization=0.1
        )
    )
    configuration = cls.llm.llm_engine.vllm_config
    runner = "newer" if configuration.use_v2_model_runner else "older"
    device = configuration.device_config.device_type
    print(
        f"\n{'/'.join(cls.FOLDER)}: V = {V}, T = {T}; vLLM runs on {device}, under the {runner} "
        "model runner",
        flush=True,
    )
    yield
    vllm_tools.shut_down(cls.llm)


@pytest.mark.usefixtures("both_copies")
class Configuration:
    """The test for one configuration. FOLDER is its directory, from the repository's top. SPREAD
    adds prompts (both_copies()). With DIGESTS, vLLM is given the SHA-256 digests of policy.txt,
    KL.bin and integer_model.json.
    """

    FOLDER = None
    SPREAD = DIGESTS = False

    def test_served_replies_are_the_integer_models_greedy_replies(self):
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt} for prompt in self.prompts],
            self.vllm.SamplingParams(temperature=0, max_tokens=None),
            use_tqdm=False,
        )
        endings = set()
        for prompt, output in zip(self.prompts, outputs):
            reply = list(output.outputs[0].token_ids)
            with self.subTest(prompt=prompt):
                self.assertEqual(reply, direct_reply(self.model, prompt, self.end_token))
                sequence = self.model.new_sequence()
                choices = self.model.extend(sequence, prompt + reply[:-1]).tolist()
                self.model.free(sequence)
                self.assertEqual(choices[len(prompt) - 1 :], reply)
            endings.add(output.outputs[0].finish_reason)
        self.assertEqual(endings, {"stop", "length"})
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(f"{len(outputs)} replies, {tokens} tokens in all", flush=True)


class TargetSmall(Configuration, unittest.TestCase):
    FOLDER = ("prover", "examples", "target_small_f2")
    SPREAD = DIGESTS = True


class TinyWithoutClamps(Configuration, unittest.TestCase):
    FOLDER = (
        "integrations",
        "vllm",
        "integer_model",
        "tests",
        "configurations",
        "tiny_without_clamps",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
