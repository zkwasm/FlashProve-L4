# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests of the vLLM model class, its loader and its registration. They need vLLM 0.29.0 and no GPU.

vLLM builds its configuration and loads the model; no engine runs. `Loading` tests registration,
the checks on the configuration and the tokenizer, and the loader. `Steps` calls forward() and
compute_logits() with hand-written inputs that imitate vLLM's model runners; test_vllm_engine.py
runs the class in a real engine. A stand-in (stand_in_integer_model.py), which records the calls
that it gets, replaces the model that integer_model.load_from_folder() returns ("the library"),
except in two tests of the loader.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import unittest
import unittest.mock

import pytest
import stand_in_integer_model as stand_in
import torch
import vllm_tools
from vllm_tools import (
    END_TOKEN,
    OUT_OF_RANGE_TOKEN,
    PROMPTS,
    VOCABULARY_SIZE,
    WINDOW_LENGTH,
    expected,
)

from integer_model_vllm import example

PACKAGE = pathlib.Path(vllm_tools.PACKAGES)
REPOSITORY = PACKAGE.parents[2]


def served(model, token_ids, first_position):
    """Runs one step as vLLM does. Returns the token that a greedy sampler picks after the step."""
    rows = model(
        input_ids=torch.tensor(token_ids, dtype=torch.int32),
        positions=torch.arange(first_position, first_position + len(token_ids), dtype=torch.int64),
        intermediate_tensors=None,
        inputs_embeds=None,
    )
    logits = model.compute_logits(rows[torch.tensor([len(token_ids) - 1])])
    assert logits.shape == (1, VOCABULARY_SIZE)
    assert logits.dtype == torch.float32
    return unique_maximum(logits[0])


def unique_maximum(row):
    best = int(row.argmax())
    assert int((row == row[best]).sum()) == 1, "the row has no unique maximum"
    return best


def dummy_run(model, tokens):
    """A dummy run of vLLM: ids and positions are all 0."""
    return model(
        input_ids=torch.zeros(tokens, dtype=torch.int32),
        positions=torch.zeros(tokens, dtype=torch.int64),
    )


def generate(model, prompt, chunks=None):
    """Serves one request greedily: the prompt in the given chunks, then one token per step."""
    position, reply = 0, []
    for size in chunks or [len(prompt)]:
        token = served(model, prompt[position : position + size], position)
        position += size
    assert position == len(prompt)
    # vLLM stops after an end token, and when the prompt and the reply together have reached
    # --max-model-len.
    while True:
        reply.append(token)
        if token == END_TOKEN or position + len(reply) == WINDOW_LENGTH:
            return reply
        token = served(model, [token], position + len(reply) - 1)


@pytest.fixture(scope="class")
def registered(request, vllm, tmp_path_factory):
    import integer_model_vllm

    request.cls.vllm = vllm
    integer_model_vllm.register()
    integer_model_vllm.register()  # vLLM may call it more than once
    # The tests load the model in this process, where vLLM need not have loaded its plug-ins.
    stand_in.register()
    request.cls.work = str(tmp_path_factory.mktemp("model_class"))
    request.cls.directory = vllm_tools.model_folder(os.path.join(request.cls.work, "model"))


@pytest.mark.vllm
@pytest.mark.usefixtures("registered")
class WithVllm(unittest.TestCase):
    def configuration(self, **changes):
        return self.vllm.EngineArgs(
            **vllm_tools.engine_arguments(self.directory, **changes)
        ).create_engine_config()

    def load(self, **changes):
        from vllm.model_executor.model_loader import get_model

        return get_model(vllm_config=self.configuration(**changes))


class Loading(WithVllm):
    def test_vllm_resolves_loads_and_accepts_the_class(self):
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        configuration = self.configuration()
        self.assertEqual(configuration.model_config.architecture, "IntegerModelForCausalLM")
        self.assertEqual(configuration.model_config.runner_type, "generate")
        model = self.load()
        self.assertEqual(type(model).__name__, "IntegerModelForCausalLM")
        self.assertTrue(is_text_generation_model(model))
        self.assertIsInstance(model.integer_model, stand_in.IntegerModel)
        self.assertEqual(list(model.parameters()), [])
        # While vLLM loads this model, PyTorch's default dtype is bfloat16. The loader gives
        # integer_model float32.
        self.assertEqual(
            (configuration.model_config.dtype, model.integer_model.default_dtype),
            (torch.bfloat16, torch.float32),
        )

    def test_vllm_finds_the_entry_point_of_the_built_distribution(self):
        site = vllm_tools.install(self.work)
        program = textwrap.dedent(f"""\
            import vllm
            import vllm.plugins

            vllm.plugins.load_general_plugins()
            from vllm.model_executor.model_loader import get_model_loader
            from vllm.config.load import LoadConfig

            loader = get_model_loader(LoadConfig(load_format="integer_model"))
            configuration = vllm.EngineArgs(model={self.directory!r}).create_model_config()
            resolved = vllm.ModelRegistry.resolve_model_cls(
                ["IntegerModelForCausalLM"], configuration
            )
            print(type(loader).__name__, resolved[0].__module__)
            """)
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join([site, vllm_tools.HERE])}
        found = subprocess.run(
            [sys.executable, "-c", program],
            cwd=self.work,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(
            found.stdout.split()[-2:], ["IntegerModelLoader", "integer_model_vllm.model"]
        )

    def test_settings_that_the_class_cannot_work_with_are_rejected(self):
        # The pattern begins with the model class's own words, so that a rejection by vLLM itself
        # does not pass for one by the class.
        ours = "The integer model cannot run with these settings.*"
        for changes, word in (
            ({"max_num_seqs": 2}, "--max-num-seqs 1"),
            ({"enable_prefix_caching": True}, "--no-enable-prefix-caching"),
            ({"enforce_eager": False}, "--enforce-eager"),
            ({"max_num_batched_tokens": WINDOW_LENGTH - 1}, "--max-num-batched-tokens"),
            ({"long_prefill_token_threshold": 8}, "--long-prefill-token-threshold"),
            ({"enable_prompt_embeds": True}, "--enable-prompt-embeds"),
            ({"skip_tokenizer_init": True}, "--skip-tokenizer-init"),
        ):
            with self.subTest(changes), self.assertRaisesRegex(ValueError, ours + word):
                self.load(**changes)

    def test_settings_that_need_more_than_this_machine_has_are_rejected(self):
        """vLLM does not build these configurations on every machine, so the test changes a good one
        by hand.
        """
        from integer_model_vllm.model import check_configuration

        for part, name, value, word in (
            ("parallel_config", "tensor_parallel_size", 2, "one GPU"),
            ("parallel_config", "pipeline_parallel_size", 2, "one GPU"),
            ("parallel_config", "data_parallel_size", 2, "one GPU"),
            ("speculative_config", None, object(), "speculative decoding"),
            ("lora_config", None, object(), "LoRA"),
            ("model_config", "enable_sleep_mode", True, "--enable-sleep-mode"),
            ("model_config", "enable_trace_replay", True, "--enable-trace-replay"),
            ("kv_transfer_config", None, object(), "key-value cache"),
        ):
            configuration = self.configuration()
            check_configuration(configuration)
            if name is None:
                setattr(configuration, part, value)
            else:
                setattr(getattr(configuration, part), name, value)
            with self.subTest(part=part, name=name), self.assertRaisesRegex(ValueError, word):
                check_configuration(configuration)

    def directory_with(self, name, end_in_config, end_in_generation_config=None):
        """A new model directory with the given eos_token_id in config.json and, if one is given, in
        generation_config.json.
        """
        directory = vllm_tools.model_folder(os.path.join(self.work, name))
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({**config, "eos_token_id": end_in_config}, f)
        if end_in_generation_config is not None:
            with open(os.path.join(directory, "generation_config.json"), "w") as f:
                json.dump({"eos_token_id": end_in_generation_config}, f)
        return directory

    def test_the_end_token_is_one_at_which_vllm_stops(self):
        for name, files, end_token in (
            ("one-id", (9,), 9),
            ("a-list", ([9, 7],), 9),
            ("the-other-file-wins", ([9, 7], 7), 7),
            ("the-other-files-list", (9, [8, 7]), 8),
        ):
            with self.subTest(name):
                self.assertEqual(
                    self.load(model=self.directory_with(name, *files)).end_token, end_token
                )
        for name, files in (("too-large", (VOCABULARY_SIZE,)), ("empty", (9, []))):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "eos_token_id"):
                self.load(model=self.directory_with(name, *files))

    def test_a_vocabulary_that_32_bit_floats_cannot_count_is_rejected(self):
        configuration = self.configuration()
        configuration.model_config.get_vocab_size = lambda: 2**24
        from integer_model_vllm.model import IntegerModelForCausalLM

        with self.assertRaisesRegex(ValueError, "2\\^24 or more tokens"):
            IntegerModelForCausalLM(vllm_config=configuration)

    def test_the_tokenizer_must_leave_vllm_no_id_outside_the_vocabulary(self):
        """vLLM accepts a prompt's ids up to the tokenizer's max_token_id. The tokenizer written by
        example.write_vllm_files() has 49 for 50 tokens; with the same words all in the base
        vocabulary it has 50, which the class rejects.
        """
        with self.assertLogs("vllm.integer_model_vllm.model", "INFO") as log:
            self.load()
        self.assertIn(
            "The tokenizer's max_token_id is 49, and the vocabulary has 50 tokens.",
            "\n".join(log.output),
        )
        directory = vllm_tools.model_folder(os.path.join(self.work, "ordinary_tokenizer"))
        with open(os.path.join(directory, "tokenizer.json")) as f:
            tokenizer = json.load(f)
        (added,) = tokenizer["added_tokens"]
        self.assertEqual(added["id"], VOCABULARY_SIZE - 1)
        tokenizer["model"]["vocab"][added["content"]] = added["id"]
        tokenizer["added_tokens"] = []
        with open(os.path.join(directory, "tokenizer.json"), "w") as f:
            json.dump(tokenizer, f)
        with self.assertRaisesRegex(
            ValueError,
            "The tokenizer's max_token_id is 50, but the integer model's largest token id is 49, "
            "so vLLM would accept token ids outside the vocabulary",
        ):
            self.load(model=directory)

    def test_the_loaders_that_come_with_vllm_are_rejected(self):
        from vllm.config import set_current_vllm_config

        configuration = self.configuration(load_format="auto", model_loader_extra_config={})
        with (
            set_current_vllm_config(configuration),
            self.assertRaisesRegex(ValueError, "--load-format integer_model"),
        ):
            self.load(load_format="auto", model_loader_extra_config={})

    def test_a_model_that_does_not_fit_the_configuration_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "at most 31 tokens, but vLLM would allow 32. Pass --max-model-len 31 or less",
        ):
            self.load(
                model_loader_extra_config={
                    **vllm_tools.STAND_IN,
                    "window_length": WINDOW_LENGTH - 1,
                }
            )
        self.load(max_model_len=WINDOW_LENGTH - 1)  # a shorter limit in vLLM is fine
        model = self.load()
        with self.assertRaisesRegex(
            ValueError, "vocabulary has 51 tokens, but the field vocab_size .* is 50"
        ):
            model.attach(stand_in.IntegerModel(VOCABULARY_SIZE + 1, WINDOW_LENGTH))

    def test_a_configuration_file_that_names_another_model_directory_is_rejected(self):
        """vLLM takes the directory that "speculators_config" in config.json names as the model
        directory. The entry also turns on speculative decoding, which the class rejects.
        """
        directory = vllm_tools.model_folder(os.path.join(self.work, "names_another_directory"))
        with open(os.path.join(directory, "config.json")) as f:
            configuration = json.load(f)
        configuration.update(
            # With IntegerModelForCausalLM here, vLLM itself rejects the file.
            architectures=["LlamaForCausalLM"],
            speculators_model_type="eagle3",
            transformer_layer_config={
                name: value for name, value in configuration.items() if name != "architectures"
            },
            speculators_config={
                "proposal_methods": [{"speculative_tokens": 1}],
                "verifier": {"name_or_path": self.directory},
            },
        )
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(configuration, f)
        self.assertEqual(self.configuration(model=directory).model_config.model, self.directory)
        with self.assertRaisesRegex(
            ValueError,
            'remove the options for speculative decoding, and the entry "speculators_config"',
        ):
            self.load(model=directory)

    def test_the_digests_of_the_files_reach_the_package(self):
        """No stand-in here: the loader passes the options of vLLM's --model-loader-extra-config on
        to integer_model, which loads the small model of configurations/tiny_without_clamps.
        """
        tiny = os.path.join(vllm_tools.HERE, "configurations", "tiny_without_clamps")
        directory = vllm_tools.model_folder(
            os.path.join(self.work, "tiny"), vocabulary_size=200, window_length=16
        )
        for name in ("policy.txt", "KL.bin"):
            shutil.copy(os.path.join(tiny, name), directory)
        with open(os.path.join(directory, "integer_model.json"), "w") as f:
            json.dump({"weights": "synthetic"}, f)
        given = {}
        for name in ("integer_model.json", "policy.txt", "KL.bin"):
            with open(os.path.join(directory, name), "rb") as f:
                given[name] = hashlib.sha256(f.read()).hexdigest()

        def load(**options):
            return self.load(model=directory, max_model_len=16, model_loader_extra_config=options)

        with self.assertLogs("vllm.integer_model_vllm.loader", "INFO") as log:
            self.assertEqual(load(sha256=given).integer_model.parameters.V, 200)
        self.assertEqual(len(log.output), 1)
        self.assertIn(
            "Verified: these files have the given SHA-256 digests: integer_model.json, "
            "policy.txt, KL.bin.",
            log.output[0],
        )
        # The same digests from a file.
        digest_file = os.path.join(self.work, "sha256.json")
        with open(digest_file, "w") as f:
            json.dump(given, f)
        os.chmod(digest_file, 0o600)  # others may neither read nor write it
        with self.assertLogs("vllm.integer_model_vllm.loader", "INFO") as log:
            self.assertEqual(load(sha256_file=digest_file).integer_model.parameters.V, 200)
        self.assertIn("Verified: these files have the given SHA-256 digests:", log.output[0])
        # The digests taken from reference copies, as examples/serve_gguf.sh asks for them.
        copies = os.path.join(self.work, "copies")
        os.mkdir(copies)
        for name in given:
            shutil.copy(os.path.join(directory, name), copies)
        with self.assertLogs("vllm.integer_model_vllm.loader", "INFO") as log:
            self.assertEqual(load(reference_directory=copies).integer_model.parameters.V, 200)
        self.assertIn(
            "Verified: these files have the given SHA-256 digests: integer_model.json, "
            "policy.txt, KL.bin.",
            log.output[0],
        )
        shutil.copy(os.path.join(directory, "policy.txt"), os.path.join(copies, "KL.bin"))
        with self.assertNoLogs("vllm.integer_model_vllm.loader", "INFO"):
            with self.assertRaisesRegex(
                ValueError,
                "KL.bin is not the file whose SHA-256 was given.* compare a file that is refused "
                "for its digest with its copy there",
            ):
                load(reference_directory=copies)
            with self.assertRaisesRegex(ValueError, 'give only one of "sha256", "sha256_file"'):
                load(reference_directory=copies, sha256=given)
            with self.assertRaisesRegex(ValueError, "wsums.txt': No such file"):
                load(reference_directory=copies, reference_files=5)
        with self.assertNoLogs("vllm.integer_model_vllm.loader", "INFO"):
            with self.assertRaisesRegex(
                ValueError, "KL.bin is not the file whose SHA-256 was given"
            ):
                load(sha256={**given, "KL.bin": given["policy.txt"]})
            with self.assertRaisesRegex(
                ValueError, 'with "sha256", no other option is allowed \\(found "weights"\\)'
            ):
                load(sha256=given, weights="synthetic")
        # The small model's weights come from no files, so the test marks them as weights from GGUF
        # files that were not compared with wsums.txt.
        from integer_model.synthetic_weights import SyntheticWeights

        not_compared = unittest.mock.patch.object(
            SyntheticWeights, "files_not_compared", True, create=True
        )
        with not_compared, self.assertLogs("vllm.integer_model_vllm.loader", "WARNING") as log:
            load(sha256=given)
        self.assertEqual(len(log.output), 1)
        self.assertIn(
            'The weight files were not verified, because "verify_sums" is not true.', log.output[0]
        )
        with not_compared, self.assertLogs("vllm.integer_model_vllm.loader", "WARNING") as log:
            load()
        self.assertEqual(len(log.output), 2)
        for options in ({}, {"sha256": None}):
            with (
                self.subTest(options),
                self.assertLogs("vllm.integer_model_vllm.loader", "INFO") as log,
            ):
                load(**options)
            self.assertEqual(len(log.output), 1)  # so no line reports "Verified"
            self.assertIn("WARNING", log.output[0])
            self.assertIn(
                "The model's files were not verified: no SHA-256 digests were given", log.output[0]
            )
            self.assertIn(
                "keep reference copies outside the model directory and pass, in",
                log.output[0],
            )

    def test_weights_in_gguf_files_outside_the_model_directory(self):
        """The example small_gguf; no stand-in."""
        if not shutil.which("g++"):
            self.skipTest("needs g++, which builds the program that writes the GGUF files")
        top = pathlib.Path(self.work, "small_gguf")
        example.main(
            ["small_gguf", "--directory", str(top), "--repository", str(REPOSITORY), "--quiet"]
        )
        description = json.loads((top / "model" / "integer_model.json").read_text())
        for verify_sums in (True, False):
            description["weight_settings"]["verify_sums"] = verify_sums
            (top / "model" / "integer_model.json").write_text(json.dumps(description))
            given = {}
            for where, names in (
                ("model", ("integer_model.json", "policy.txt", "KL.bin")),
                ("weights", ("wsums.txt", "fold.txt")),
            ):
                for name in names:
                    given[name] = hashlib.sha256((top / where / name).read_bytes()).hexdigest()
            with (
                self.subTest(verify_sums=verify_sums),
                self.assertLogs("vllm.integer_model_vllm.loader", "INFO") as log,
            ):
                self.load(
                    model=str(top / "model"),
                    max_model_len=64,
                    model_loader_extra_config={"sha256": given},
                )
            self.assertEqual(len(log.output), 2)
            self.assertIn(
                "digests: integer_model.json, policy.txt, KL.bin, wsums.txt, fold.txt.",
                log.output[0],
            )
            if verify_sums:
                self.assertIn("INFO", log.output[1])
                self.assertIn(
                    "Verified: every weight tensor has the SHA-256 that wsums.txt lists for it.",
                    log.output[1],
                )
            else:
                self.assertIn("WARNING", log.output[1])
                self.assertIn("The weight files were not verified", log.output[1])


class Steps(WithVllm):
    def setUp(self):
        self.model = self.load()
        self.calls = self.model.integer_model.calls

    def test_replies_equal_the_stand_ins_continuation(self):
        for prompt in PROMPTS + PROMPTS[::-1]:
            self.assertEqual(generate(self.model, prompt), expected(prompt))

    def test_the_library_sees_each_token_once(self):
        reply = generate(self.model, [1, 2, 3])
        generate(self.model, [4])
        first = (
            [("new",), ("extend", [1, 2, 3])]
            + [("extend", [token]) for token in reply[:-1]]
            + [("free",), ("new",)]
        )
        self.assertEqual(self.calls[: len(first)], first)

    def test_a_prompt_in_chunks_gets_the_same_reply(self):
        prompt = list(range(10, 30))
        for chunks in ([7, 7, 6], [1] * 20, [19, 1], [1, 19]):
            self.assertEqual(generate(self.model, prompt, chunks), expected(prompt))

    def test_every_prompt_position_has_a_choice(self):
        """A request with prompt_logprobs makes vLLM ask for the logits at all positions of the
        prompt.
        """
        prompt = [1, 2, 3, 4]
        rows = self.model(input_ids=torch.tensor(prompt), positions=torch.arange(4))
        logits = self.model.compute_logits(rows)
        self.assertEqual(
            [unique_maximum(row) for row in logits], stand_in.choices(prompt, VOCABULARY_SIZE)
        )

    def test_a_sequence_can_be_abandoned_or_computed_again(self):
        """A new sequence can start at any time: after an aborted request, or when vLLM computes a
        request again.
        """
        prompt = [1, 2, 3]
        token = served(self.model, prompt, 0)
        served(self.model, [token], 3)
        self.assertEqual(generate(self.model, prompt + [token]), expected(prompt + [token]))

    def test_a_step_after_the_end_token_does_no_harm(self):
        """With asynchronous scheduling vLLM plans one step ahead, so one more step can follow the
        step that produced the end token.
        """
        prompt = PROMPTS[0]
        reply = generate(self.model, prompt)
        self.assertEqual(reply[-1], END_TOKEN)
        served(self.model, [END_TOKEN], len(prompt) + len(reply) - 1)
        self.assertEqual(generate(self.model, [4]), expected([4]))

    def test_a_served_token_that_differs_from_the_choice(self):
        """The model continues from the served token, and the log has one warning per sequence."""
        with self.assertNoLogs("vllm.integer_model_vllm.model", "WARNING"):
            generate(self.model, [1, 2, 3])
            token = served(self.model, [1, 2, 3], 0)
        other = (token + 1) % 10
        with self.assertLogs("vllm.integer_model_vllm.model", "WARNING") as log:
            self.assertEqual(
                served(self.model, [other], 3),
                stand_in.choices([1, 2, 3, other], VOCABULARY_SIZE)[-1],
            )
            self.assertEqual(
                served(self.model, [other], 4),
                stand_in.choices([1, 2, 3, other, other], VOCABULARY_SIZE)[-1],
            )
        self.assertEqual(len(log.output), 1)
        self.assertIn(
            "vLLM served the token at position 3, but the integer model chose another one.",
            log.output[0],
        )
        with self.assertNoLogs("vllm.integer_model_vllm.model", "WARNING"):
            generate(self.model, [1, 2, 3])

    def test_dummy_runs_do_not_reach_the_library(self):
        for tokens in (2, WINDOW_LENGTH, 4 * WINDOW_LENGTH):
            rows = dummy_run(self.model, tokens)
            self.assertEqual(rows.shape[0], tokens)
            self.assertTrue(rows.dtype.is_floating_point)
            logits = self.model.compute_logits(rows[-1:])
            self.assertEqual(logits.shape, (1, VOCABULARY_SIZE))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(self.calls, [])

    def test_dummy_runs_between_steps_do_not_disturb_the_sequence(self):
        prompt = [1, 2, 3]
        token = served(self.model, prompt, 0)
        dummy_run(self.model, 8)
        self.assertEqual(served(self.model, [token], 3), expected(prompt)[1])

    def test_a_dummy_run_of_one_token_is_a_short_sequence(self):
        dummy_run(self.model, 1)
        self.assertEqual(self.calls, [("new",), ("extend", [0])])
        self.assertEqual(generate(self.model, [1, 2, 3]), expected([1, 2, 3]))

    def test_the_older_runners_sampler_warm_up(self):
        """The warm-up replaces a dummy run's hidden states with random numbers between 0 and 1 and
        computes logits from them.
        """
        rows = torch.rand_like(dummy_run(self.model, 1))
        self.assertEqual(self.model.compute_logits(rows).tolist(), [[0.0] * VOCABULARY_SIZE])

    def test_positions_that_fit_no_single_sequence_stop_the_engine(self):
        def step(ids, positions):
            return self.model(input_ids=torch.tensor(ids), positions=torch.tensor(positions))

        for name, before, ids, positions in (
            ("a start above position 0", None, [1, 2], [5, 6]),
            ("a gap", [1, 2, 3], [4], [4]),
            ("an overlap", [1, 2, 3], [4], [2]),
            ("two sequences in one step", [1, 2, 3], [4, 5, 6], [3, 0, 1]),
            ("two new sequences in one step", None, [1, 2, 3, 4], [0, 1, 0, 1]),
            ("padding after the tokens", None, [1, 2, 0, 0], [0, 1, 0, 0]),
            ("positions in reverse", None, [1, 2], [1, 0]),
        ):
            with self.subTest(name):
                model = self.load()
                if before:
                    served(model, before, 0)
                calls = list(model.integer_model.calls)
                with self.assertRaisesRegex(RuntimeError, "one sequence at a time"):
                    step(ids, positions)
                self.assertEqual(model.integer_model.calls, calls)

    def test_a_failed_range_condition_ends_the_reply(self):
        with self.assertLogs("vllm.integer_model_vllm.model", "ERROR") as log:
            self.assertEqual(served(self.model, [1, OUT_OF_RANGE_TOKEN, 3], 0), END_TOKEN)
        self.assertIn("At position 1 of the sequence", log.output[0])
        self.assertEqual(len(log.output), 1)
        calls = list(self.calls)
        self.assertEqual(calls[-1], ("free",))
        # A request that ignores the end token goes on. It gets end tokens, and the library is not
        # called.
        with self.assertNoLogs("vllm.integer_model_vllm.model", "WARNING"):
            self.assertEqual(served(self.model, [END_TOKEN], 3), END_TOKEN)
            self.assertEqual(served(self.model, [0], 4), END_TOKEN)
        # An id outside the vocabulary stops the engine then too.
        with self.assertRaisesRegex(RuntimeError, "has an id outside the vocabulary"):
            served(self.model, [VOCABULARY_SIZE], 5)
        self.assertEqual(self.calls, calls)
        self.assertEqual(generate(self.model, [1, 2, 3]), expected([1, 2, 3]))

    def test_a_failed_range_condition_in_a_decode_step(self):
        served(self.model, [1, 2, 3], 0)
        with self.assertLogs("vllm.integer_model_vllm.model", "ERROR"):
            self.assertEqual(served(self.model, [OUT_OF_RANGE_TOKEN], 3), END_TOKEN)
        self.assertEqual(generate(self.model, [4]), expected([4]))

    def test_other_exceptions_from_the_library_stop_the_engine(self):
        served(self.model, [1] * WINDOW_LENGTH, 0)
        with self.assertRaises(stand_in.WindowFull):
            served(self.model, [1], WINDOW_LENGTH)
        with self.assertRaisesRegex(RuntimeError, "faulty token"):
            served(self.load(), [1, vllm_tools.FAULTY_TOKEN], 0)

    def test_a_token_id_outside_the_vocabulary_stops_the_engine(self):
        """The library is not called, and the running sequence is left as it is."""
        for token_ids, first_position, position in (
            ([VOCABULARY_SIZE], 0, 0),
            ([1, 2, -1, 4], 0, 2),
            ([1, 10**9, -7], 0, 1),
            ([-1], 2, 2),
            ([VOCABULARY_SIZE], 2, 2),
        ):
            with self.subTest(token_ids=token_ids, first_position=first_position):
                served(self.model, [1, 2], 0)
                calls = list(self.calls)
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"The token at position {position} of the sequence has an id outside the "
                    "vocabulary \\(0 to 49\\), so the integer model stops the server",
                ):
                    served(self.model, token_ids, first_position)
                self.assertEqual(self.calls, calls)

    def test_large_token_ids_come_out_exactly(self):
        """The model class passes its choices from forward() to compute_logits() as 32-bit floats,
        which are exact for ids below 2^24. V is the full-size model's vocabulary size.
        """
        V = 202048
        directory = vllm_tools.model_folder(os.path.join(self.work, "large"), vocabulary_size=V)
        model = self.load(
            model=directory,
            model_loader_extra_config={"stand_in": True, "window_length": 32},
        )
        prompt = [V - 3, 65537, 131073, 200001, 16385]
        logits = model.compute_logits(
            model(input_ids=torch.tensor(prompt), positions=torch.arange(len(prompt)))
        )
        choices = stand_in.choices(prompt, V)
        self.assertEqual([unique_maximum(row) for row in logits], choices)
        self.assertTrue(any(choice > 2**16 and choice % 2 for choice in choices))

    def test_a_model_that_is_loaded_again_replaces_the_first(self):
        served(self.model, [1, 2, 3], 0)
        self.model.attach(stand_in.IntegerModel(VOCABULARY_SIZE, WINDOW_LENGTH))
        self.assertEqual(self.calls[-1], ("free",))
        self.assertEqual(generate(self.model, [1, 2, 3]), expected([1, 2, 3]))

    def test_a_library_that_returns_something_else_stops_the_engine(self):
        for wrong in (torch.tensor([1, 2]), torch.tensor([VOCABULARY_SIZE]), torch.tensor([-1])):
            self.model.integer_model.extend = lambda sequence, token_ids, wrong=wrong: wrong
            with self.assertRaisesRegex(RuntimeError, "fault in the package"):
                served(self.model, [1], 0)

    def test_the_chosen_token_survives_sampling_settings(self):
        """Repeats in plain PyTorch what vLLM's samplers do to a row before picking a token, at the
        limits of what vLLM accepts.

        vLLM accepts a bias of at most 100 either way on the completion and chat routes, and
        frequency and presence penalties of at most 2. The repetition penalty may be any positive
        number, rounded to 0 or infinity where float32 cannot hold it, and applies to the tokens
        that the sequence contains. vLLM's own samplers are used in test_vllm_engine.py
        (test_sampling_settings_do_not_change_the_tokens).

        The two statements for the repetition penalty are adapted from vLLM 0.29.0:
        vllm/v1/worker/gpu/sample/penalties.py (Apache License 2.0), with PyTorch's where in place
        of Triton's. The order of the steps (the bias, the penalties, the temperature) is that of
        sampler.py in the same directory."""
        rows = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        row = self.model.compute_logits(rows[-1:])[0]
        choice = stand_in.choices([1, 2, 3], VOCABULARY_SIZE)[-1]
        bias = torch.full_like(row, 100.0)
        bias[choice] = -100.0
        only_the_choice = torch.arange(VOCABULARY_SIZE) == choice
        # At most 2 for each of the full-size window's 1,024 tokens, plus 2 for presence.
        largest_penalties = 2.0 * 1024 + 2.0
        for penalty in (1e-46, 1e-30, 1e-25, 0.5, 2.0, 1e30, 1e39):
            for contained in (
                only_the_choice,
                ~only_the_choice,
                torch.ones_like(only_the_choice),
                torch.zeros_like(only_the_choice),
            ):
                scale = torch.where(contained, torch.tensor(penalty, dtype=torch.float32), 1.0)
                changed = row + bias
                changed = changed * torch.where(changed > 0, 1.0 / scale, scale)
                changed = changed - contained * largest_penalties
                self.assertEqual(unique_maximum(changed), choice)
                # The two smallest penalties can make the chosen token's logit infinite, and then
                # the probabilities are undefined.
                if penalty >= 1e-25:
                    for temperature in (0.01, 1.0, 2.0):
                        self.assertEqual(
                            float(torch.softmax(changed / temperature, dim=-1)[choice]), 1.0
                        )

    def test_inputs_other_than_token_ids_are_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "token ids only"):
            self.model.embed_input_ids(torch.tensor([1]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
