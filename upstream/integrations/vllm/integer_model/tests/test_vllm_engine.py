# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The vLLM model class inside a running vLLM engine, on a stand-in for the integer model.

vLLM itself runs, from the plug-in's entry point to the sampler. Only the model that
integer_model.load_from_folder() returns is replaced, by stand_in_integer_model.py. Replies must
equal stand_in.continuation(), the stand-in's choices computed without vLLM: if vLLM's calls lose,
repeat, reorder or invent a token, later choices very likely change.

Needs vLLM 0.29.0 (conftest.py). Without a GPU, vLLM's CPU worker runs in the test's process,
and cpu_only.py stubs a compiled function missing from a GPU build of vLLM.
"""

import json
import os
import sys
import types
import unittest

import pytest

import stand_in_integer_model as stand_in
import vllm_tools
from vllm_tools import (
    END_TOKEN,
    OUT_OF_RANGE_TOKEN,
    PROMPTS,
    VOCABULARY_SIZE,
    WINDOW_LENGTH,
    expected,
)

pytestmark = pytest.mark.vllm


@pytest.fixture(scope="class")
def engine(request, vllm, tmp_path_factory):
    """Sets the class attributes llm, one engine for all its tests, and vllm, the module."""
    request.cls.vllm = vllm
    folder = vllm_tools.model_folder(str(tmp_path_factory.mktemp("engine") / "model"))
    # The fraction only matters without a GPU, where it refers to the machine's memory.
    llm = request.cls.llm = vllm.LLM(
        **vllm_tools.engine_arguments(folder, gpu_memory_utilization=0.1)
    )
    configuration = llm.llm_engine.vllm_config
    runner = "newer" if configuration.use_v2_model_runner else "older"
    scheduling = "with" if configuration.scheduler_config.async_scheduling else "without"
    device = configuration.device_config.device_type
    print(
        f"\nvLLM runs on {device}, under the {runner} model runner, {scheduling} asynchronous "
        "scheduling",
        flush=True,
    )
    yield
    vllm_tools.shut_down(llm)


@pytest.mark.usefixtures("engine")
class Engine(unittest.TestCase):
    vllm: types.ModuleType  # set by the fixture engine

    def replies(self, prompts, **settings):
        """vLLM's replies to prompts given as token ids. By default decoding is greedy and runs to
        the end token or the window's end.
        """
        settings = {"temperature": 0, "max_tokens": None, **settings}
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            self.vllm.SamplingParams(**settings),
            use_tqdm=False,
        )
        self.assertEqual([output.prompt_token_ids for output in outputs], prompts)
        return [output.outputs[0] for output in outputs]

    def tearDown(self):
        """After each test the engine must still serve correctly."""
        self.assertEqual(list(self.replies([[1, 2, 3]])[0].token_ids), expected([1, 2, 3]))

    def test_greedy_replies(self):
        reasons = set()
        for prompts in (PROMPTS, PROMPTS[::-1], [PROMPTS[0]] * 3):
            for prompt, reply in zip(prompts, self.replies(prompts)):
                self.assertEqual(list(reply.token_ids), expected(prompt))
                self.assertEqual(
                    reply.finish_reason, "stop" if reply.token_ids[-1] == END_TOKEN else "length"
                )
                reasons.add(reply.finish_reason)
        self.assertEqual(reasons, {"stop", "length"})

    def test_replies_cut_by_max_tokens(self):
        for longest in (1, 2, 5):
            for prompt, reply in zip(PROMPTS, self.replies(PROMPTS, max_tokens=longest)):
                self.assertEqual(list(reply.token_ids), expected(prompt, longest))

    def test_sampling_settings_do_not_change_the_tokens(self):
        for settings in (
            {"temperature": 1.0},
            {"temperature": 2.0, "seed": 1},
            {"temperature": 1.0, "top_p": 0.5, "top_k": 3, "min_p": 0.1},
            {"temperature": 1.0, "top_k": 3},
            {"temperature": 1.0, "top_p": 0.5},
            {"repetition_penalty": 2.0, "frequency_penalty": 2.0, "presence_penalty": 2.0},
            {
                "temperature": 1.0,
                "repetition_penalty": 0.5,
                "frequency_penalty": -2.0,
                "presence_penalty": -2.0,
            },
            {"logit_bias": {token: 100.0 for token in range(0, VOCABULARY_SIZE, 2)}},
            {"logit_bias": {token: -100.0 for token in range(VOCABULARY_SIZE)}},
            # vLLM accepts any positive repetition penalty, even one that float32 rounds to 0 or
            # infinity. The two smallest run at temperature 0 only: they can make the chosen
            # logit infinite and the probabilities undefined.
            {"repetition_penalty": 1e-46},
            {"repetition_penalty": 1e-30, "frequency_penalty": 2.0, "presence_penalty": 2.0},
            {"repetition_penalty": 1e30, "frequency_penalty": 2.0, "presence_penalty": 2.0},
            {"repetition_penalty": 1e39},
            {"temperature": 2.0, "repetition_penalty": 1e-25},
            {"temperature": 2.0, "repetition_penalty": 1e39},
        ):
            with self.subTest(settings):
                for prompt, reply in zip(PROMPTS, self.replies(PROMPTS, **settings)):
                    self.assertEqual(list(reply.token_ids), expected(prompt))

    def test_a_reply_that_goes_on_after_the_end_token(self):
        prompt = PROMPTS[0]
        reply = list(self.replies([prompt], ignore_eos=True)[0].token_ids)
        self.assertEqual(
            reply, stand_in.continuation(prompt, VOCABULARY_SIZE, [], WINDOW_LENGTH - len(prompt))
        )
        self.assertIn(END_TOKEN, reply[:-1])

    def test_log_probabilities_say_nothing_but_the_choice(self):
        prompt = [1, 2, 3, 4]
        output = self.llm.generate(
            [{"prompt_token_ids": prompt}],
            self.vllm.SamplingParams(temperature=0, max_tokens=3, logprobs=2, prompt_logprobs=1),
            use_tqdm=False,
        )[0]
        self.assertEqual(list(output.outputs[0].token_ids), expected(prompt, 3))
        for token, row in zip(output.outputs[0].token_ids, output.outputs[0].logprobs):
            self.assertEqual(row[token].logprob, 0.0)
            self.assertTrue(
                all(entry.logprob < -1e9 for other, entry in row.items() if other != token)
            )
        # From the prompt's second position on, the most likely token is the stand-in's choice
        # after the tokens before it.
        for choice, row in zip(
            stand_in.choices(prompt, VOCABULARY_SIZE), output.prompt_logprobs[1:]
        ):
            self.assertEqual(row[choice].logprob, 0.0)

    def test_settings_that_exclude_tokens_are_obeyed(self):
        """Settings that exclude tokens can force a token other than the model's choice. The model
        then continues from the served token.
        """
        prompt = [1, 2, 3]
        allowed = [10, 11]
        reply = list(self.replies([prompt], allowed_token_ids=allowed, max_tokens=4)[0].token_ids)
        self.assertTrue(set(reply) <= set(allowed))
        first = expected(prompt)[0]
        reply = list(
            self.replies([prompt], logit_bias={first: float("-inf")}, max_tokens=3)[0].token_ids
        )
        self.assertNotEqual(reply[0], first)
        self.assertEqual(reply[1:], expected(prompt + reply[:1], 2))

    def test_a_failed_range_condition_ends_the_reply(self):
        for prompt in ([OUT_OF_RANGE_TOKEN], [1, 2, OUT_OF_RANGE_TOKEN, 4]):
            reply = self.replies([prompt])[0]
            self.assertEqual((list(reply.token_ids), reply.finish_reason), ([END_TOKEN], "stop"))
        reply = self.replies([[1, OUT_OF_RANGE_TOKEN]], ignore_eos=True, max_tokens=5)[0]
        self.assertEqual(list(reply.token_ids), [END_TOKEN] * 5)
        # After the failure the class offers only the end token, so a setting that bars it makes
        # the reply go on with tokens that neither the stand-in nor the class chose.
        for settings in (
            {"allowed_token_ids": [10, 11]},
            {"bad_words": ["<end>"]},
            {"logit_bias": {END_TOKEN: float("-inf")}},
            {"min_tokens": 4},
        ):
            with self.subTest(settings):
                reply = self.replies([[1, OUT_OF_RANGE_TOKEN]], max_tokens=4, **settings)[0]
                self.assertEqual((len(reply.token_ids), reply.finish_reason), (4, "length"))
                self.assertNotIn(END_TOKEN, reply.token_ids)
        # min_tokens bars the end token only for that many tokens. Then the reply ends as it does
        # without the setting.
        reply = self.replies([[1, OUT_OF_RANGE_TOKEN]], min_tokens=3)[0]
        self.assertNotIn(END_TOKEN, reply.token_ids[:3])
        self.assertEqual((list(reply.token_ids[3:]), reply.finish_reason), ([END_TOKEN], "stop"))

    def test_vllm_rejects_a_token_id_outside_the_vocabulary(self):
        """vLLM rejects the request before its engine sees it."""
        from vllm.exceptions import VLLMValidationError

        for token in (VOCABULARY_SIZE, VOCABULARY_SIZE + 1, 2**31 - 1, -1):
            with self.subTest(token=token):
                with self.assertRaisesRegex(VLLMValidationError, "out of vocabulary"):
                    self.replies([[1, token, 3]])
                self.assertEqual(list(self.replies([[4, 5]])[0].token_ids), expected([4, 5]))

    def test_vllm_accepts_no_temperature_above_2(self):
        """So test_sampling_settings_do_not_change_the_tokens covers the highest temperature that
        vLLM accepts.
        """
        from vllm.exceptions import VLLMValidationError

        self.vllm.SamplingParams(temperature=2.0)
        for temperature in (2.0001, 1e8, float("inf")):
            with self.assertRaisesRegex(VLLMValidationError, "temperature"):
                self.vllm.SamplingParams(temperature=temperature)

    def test_beam_search_returns_the_greedy_reply_first(self):
        from vllm.sampling_params import BeamSearchParams

        prompt = [1, 2, 3]
        beams = self.llm.beam_search(
            [{"prompt_token_ids": prompt}], BeamSearchParams(beam_width=2, max_tokens=4)
        )[0].sequences
        self.assertEqual(beams[0].tokens, prompt + expected(prompt, 4))
        self.assertNotEqual(beams[1].tokens, beams[0].tokens)

    def test_a_prompt_that_fills_the_window_is_rejected_by_vllm(self):
        from vllm.exceptions import VLLMValidationError

        with self.assertRaisesRegex(VLLMValidationError, "longer than the maximum model length"):
            self.replies([[1] * WINDOW_LENGTH])


class LargeVocabulary(unittest.TestCase):
    """The vocabulary size of prover/examples/target_full_f2. The model class passes its choices
    through vLLM as 32-bit floats, exact for ids below 2^24; large odd ids test that.
    """

    V = 202048

    @pytest.fixture(autouse=True)
    def work_folder(self, tmp_path, vllm):
        self.work, self.vllm = tmp_path, vllm

    def test_greedy_replies(self):
        folder = vllm_tools.model_folder(str(self.work / "model"), vocabulary_size=self.V)
        options = {"stand_in": True, "window_length": WINDOW_LENGTH}
        llm = self.vllm.LLM(
            **vllm_tools.engine_arguments(
                folder, model_loader_extra_config=options, gpu_memory_utilization=0.1
            )
        )
        self.addCleanup(vllm_tools.shut_down, llm)
        prompts = [[self.V - 3], [1, 65537, 131073], [200000] * 5]
        outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            self.vllm.SamplingParams(temperature=0, max_tokens=None),
            use_tqdm=False,
        )
        replies = [list(output.outputs[0].token_ids) for output in outputs]
        for prompt, reply in zip(prompts, replies):
            self.assertEqual(
                reply,
                stand_in.continuation(prompt, self.V, [END_TOKEN], WINDOW_LENGTH - len(prompt)),
            )
        self.assertGreater(
            sum(token > 2**16 and token % 2 == 1 for reply in replies for token in reply), 10
        )


class NoStart(unittest.TestCase):
    """The engine must not start with settings that let an id outside the vocabulary reach it."""

    @pytest.fixture(autouse=True)
    def model_directory(self, tmp_path, vllm):
        self.folder = vllm_tools.model_folder(str(tmp_path / "model"))
        self.vllm = vllm

    def assert_no_start(self, **changes):
        arguments = vllm_tools.engine_arguments(self.folder, gpu_memory_utilization=0.1, **changes)
        # The model class raises ValueError. Where the engine is a separate process, vLLM reports
        # that the process failed. The reason is not checked.
        with self.assertRaises((ValueError, RuntimeError)):
            self.addCleanup(vllm_tools.shut_down, self.vllm.LLM(**arguments))

    def test_without_a_tokenizer(self):
        """vLLM then checks no prompt's token ids."""
        self.assert_no_start(skip_tokenizer_init=True)

    def test_with_a_tokenizer_whose_base_vocabulary_has_all_the_tokens(self):
        """vLLM treats the base vocabulary's size as a valid id: it would accept VOCABULARY_SIZE."""
        path = os.path.join(self.folder, "tokenizer.json")
        with open(path) as f:
            tokenizer = json.load(f)
        self.assertTrue(tokenizer["added_tokens"])
        for token in tokenizer["added_tokens"]:
            tokenizer["model"]["vocab"][token["content"]] = token["id"]
        tokenizer["added_tokens"] = []
        self.assertEqual(len(tokenizer["model"]["vocab"]), VOCABULARY_SIZE)
        with open(path, "w") as f:
            json.dump(tokenizer, f)
        self.assert_no_start()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
