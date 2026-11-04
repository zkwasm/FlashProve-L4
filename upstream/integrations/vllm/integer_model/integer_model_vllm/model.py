# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The integer (fixed-point) model as a vLLM model class, for one sequence at a time.

vLLM schedules requests, tokenizes, samples and streams as for any model; the class replaces only
the forward pass. It passes the new token ids to the package integer_model, which evaluates the
model exactly and returns the greedy choice: the argmax of the integer logits, with the lowest id on
ties. The class gives vLLM's sampler a row of logits that marks only that choice. No logit of the
integer model, and no other number about a token, leaves the class, except positions and the number
of a failed range condition (an intermediate value outside its interval) in the server's log. Some
of what a client can still learn is in
integrations/vllm/integer_model/OPERATING.md, "What a client can do".

Written for vLLM 0.29.0 and both of its model runners (VLLM_USE_V2_MODEL_RUNNER 0 and 1); the
interface can change with any version.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import integer_model

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# A child of vLLM's logger, so that its lines get vLLM's format, level and destination.
logger = logging.getLogger("vllm." + __name__)

GUIDE = "integrations/vllm/integer_model/OPERATING.md"

# The logits of the chosen token and of every other token. Their difference, 2e9, dwarfs what vLLM's
# settings add: a bias of at most 100 on the completion and chat routes, and frequency and presence
# penalties of at most 2 each (the first for each occurrence of a token). A temperature, at most 2,
# divides both. A repetition penalty, any positive number, divides positive logits and multiplies
# negative ones, so neither moves past 0 and the chosen logit stays the largest. vLLM's samplers
# compute in float32: the chosen logit stays finite under any repetition penalty of 1e-27 or more,
# even at vLLM's lowest temperature above 0, which is 0.01. By arithmetic, not tried with vLLM: a
# penalty below about 3e-28 with a low enough temperature above 0 makes it infinite and the
# probabilities undefined.
CHOSEN_LOGIT, OTHER_LOGIT = 1.0e9, -1.0e9


def check_configuration(vllm_config: VllmConfig) -> None:
    """Raises ValueError, with the remedy, unless vLLM has the settings that the class needs."""
    parallel = vllm_config.parallel_config
    for good, remedy in (
        (vllm_config.scheduler_config.max_num_seqs == 1,
         "pass --max-num-seqs 1: this model handles one sequence at a time"),
        (not vllm_config.cache_config.enable_prefix_caching,
         "pass --no-enable-prefix-caching: a request that reuses a cached prefix would not start "
         "at position 0"),
        (vllm_config.scheduler_config.max_num_batched_tokens
         >= vllm_config.model_config.max_model_len,
         "make --max-num-batched-tokens at least --max-model-len: vLLM feeds a prompt to this "
         "model in one step, and a longer prompt would wait forever"),
        (vllm_config.scheduler_config.long_prefill_token_threshold == 0,
         "remove --long-prefill-token-threshold: this model expects a prompt in one step"),
        (vllm_config.model_config.enforce_eager,
         "pass --enforce-eager: this model runs Python code at every step, which cannot be "
         "compiled or captured in a CUDA graph"),
        (parallel.tensor_parallel_size == parallel.pipeline_parallel_size
         == parallel.data_parallel_size == 1,
         "use one GPU: remove the options for tensor, pipeline and data parallelism"),
        # This check also stops config.json from redirecting vLLM to another model directory that
        # names this class: with an entry "speculators_config", vLLM 0.29.0 treats the given
        # directory as a draft model, takes the entry's directory as the model directory and turns
        # on speculative decoding. It cannot stop a redirect to a directory that names another
        # architecture; what vLLM then serves was not tried.
        (vllm_config.speculative_config is None,
         'remove the options for speculative decoding, and the entry "speculators_config" if the '
         "model directory's config.json has it"),
        (vllm_config.lora_config is None, "remove the options for LoRA adapters"),
        (not vllm_config.model_config.enable_sleep_mode,
         "remove --enable-sleep-mode: vLLM would discard the GPU memory that holds the integer "
         "model's weights"),
        (not vllm_config.model_config.enable_trace_replay,
         "remove --enable-trace-replay: a request could then replace the model's tokens"),
        (not vllm_config.model_config.enable_prompt_embeds,
         "remove --enable-prompt-embeds: this model takes token ids only"),
        (not vllm_config.model_config.skip_tokenizer_init,
         "remove --skip-tokenizer-init and --tokens-only: only with a tokenizer does vLLM reject a "
         "prompt's token ids outside the vocabulary. If the model directory has no tokenizer, "
         "python3 -m integer_model_vllm.example vllm_files writes one"),
        (vllm_config.kv_transfer_config is None,
         "remove the options for transferring or offloading the key-value cache: a request would "
         "not start at position 0"),
    ):  # fmt: skip
        if not good:
            raise ValueError(
                f"The integer model cannot run with these settings. To fix this, {remedy}."
            )


def check_tokenizer(model_config: Any, vocabulary_size: int) -> None:
    """Raises ValueError unless vLLM rejects every prompt token id outside 0 .. vocabulary_size - 1.

    vLLM 0.29.0 rejects an id above max(tokenizer.max_token_id, vocabulary_size - 1)
    (vllm/v1/engine/input_processor.py). For a Hugging Face tokenizer, max_token_id is the larger of
    the largest id in get_vocab() and vocab_size (vllm/tokenizers/hf.py). In the transformers
    library, get_vocab() includes the added tokens and vocab_size counts the base vocabulary. So a
    tokenizer with vocabulary_size words and no added token has max_token_id = vocabulary_size, and
    vLLM would accept that id. Negative ids are rejected as well
    (integrations/vllm/integer_model/tests/test_vllm_engine.py).
    """
    from vllm.tokenizers import cached_tokenizer_from_config

    try:
        largest = cached_tokenizer_from_config(model_config).max_token_id
    except (OSError, ValueError, AttributeError, NotImplementedError) as error:
        raise ValueError(
            f"vLLM cannot load a tokenizer for the model directory ({type(error).__name__}: "
            f"{error}). python3 -m integer_model_vllm.example vllm_files writes one "
            f'({GUIDE}, "Serve from GGUF files").'
        ) from error
    logger.info(
        "The tokenizer's max_token_id is %d, and the vocabulary has %d tokens.",
        largest,
        vocabulary_size,
    )
    if largest >= vocabulary_size:
        raise ValueError(
            f"The tokenizer's max_token_id is {largest}, but the integer model's largest token id "
            f"is {vocabulary_size - 1}, so vLLM would accept token ids outside the vocabulary. "
            f"Use a tokenizer whose max_token_id is below {vocabulary_size}: for example the "
            "word-list tokenizer that python3 -m integer_model_vllm.example vllm_files writes into "
            f'a new model directory ({GUIDE}, "Serve from GGUF files"). Text is then only the '
            "words w<n>, so send prompts as token ids."
        )


class IntegerModelForCausalLM(nn.Module):
    """vLLM's view of the integer model: a module without parameters, for one greedy sequence.

    vLLM creates the object, IntegerModelLoader calls attach(), and then vLLM calls forward() and
    compute_logits() at each step.

    The class has no vLLM attention layer, so vLLM allocates no key-value cache for it (the package
    integer_model keeps its own) and passes it no information about requests: forward() sees only
    token ids and positions. Position 0 starts a sequence, and every later call must continue
    exactly where the last one ended. This works only while vLLM runs one sequence at a time, which
    check_configuration() enforces. That function also rejects settings that split a prompt into
    several steps. The tokens would still be right, but a later step of a single token counts as a
    decode step and can set off the warning below.

    A request can force a token other than the model's choice
    (integrations/vllm/integer_model/OPERATING.md, "What a client can do", lists ways). The class
    then logs one warning for the sequence and continues from the served token. It sees a served
    token only when vLLM feeds it back, so a forced last token of a reply goes without a warning.

    When a range condition fails (docs/model.md, "9 Range conditions"), the class ends the reply
    with an end-of-sequence token that the model did not choose and logs an ERROR line; the next
    request is served as usual. With ignore_eos the reply then runs on to its length limit, all
    end-of-sequence tokens. A setting that bars that token (min_tokens, allowed_token_ids,
    bad_words, a logit_bias) makes vLLM serve other tokens for as long as it bars it. Those tokens
    mean nothing.

    A token id outside the vocabulary, positions that fit no single sequence, a malformed result
    from integer_model and any exception from it but RangeConditionFailed make forward() raise,
    which stops vLLM's engine, rather than return a token that may be wrong. vLLM rejects prompt ids
    outside the vocabulary only if it has a tokenizer whose max_token_id is below the vocabulary
    size: check_configuration() and check_tokenizer() require one. Other ids in a request, such as
    the keys of logit_bias, are outside these checks, and what vLLM does with them was not examined.

    Source: vLLM's documentation describes the interface of a model class in "Basic Model"
    (https://docs.vllm.ai/en/latest/contributing/model/basic/), and
    vllm/model_executor/models/interfaces_base.py states it (vLLM 0.29.0, Apache License 2.0). The
    names and arguments of __init__, embed_input_ids, forward, compute_logits and load_weights are
    vLLM's.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        check_configuration(vllm_config)
        model_config = vllm_config.model_config
        self.device = vllm_config.device_config.device
        self.vocabulary_size = model_config.get_vocab_size()
        self.max_model_len = model_config.max_model_len
        if self.vocabulary_size >= 2**24:
            raise ValueError(
                "The vocabulary has 2^24 or more tokens, which this package does not support."
            )
        # The end-of-sequence token ids (eos_token_id), at which vLLM ends a reply. The class
        # returns the first after a failed range condition.
        end = model_config.try_get_generation_config().get("eos_token_id")
        ends = end if isinstance(end, list) else [end]
        end_token = ends[0] if ends else None
        if not isinstance(end_token, int) or not 0 <= end_token < self.vocabulary_size:
            raise ValueError(
                'The model directory needs the field "eos_token_id", one token id of the '
                "vocabulary or a list of them, in generation_config.json or, without that file, "
                "in config.json."
            )
        self.end_token = end_token
        check_tokenizer(model_config, self.vocabulary_size)
        self.integer_model: Any = None  # an integer_model.IntegerModel, set by `attach`
        self.sequence: Any = None  # integer_model's handle for the running sequence
        self.length = 0  # the number of tokens of the running sequence that `forward` has seen
        # The model's choice after the last of these tokens, until a served token differs from it.
        self.last_choice: int | None = None
        self.ended = False  # whether the class has ended the running sequence itself (see forward)

    def attach(self, loaded: integer_model.IntegerModel) -> None:
        """Takes the loaded model from the loader."""
        parameters = loaded.parameters
        if self.vocabulary_size != parameters.V:
            raise ValueError(
                f"The integer model's vocabulary has {parameters.V} tokens, but the field "
                f"vocab_size in the model directory's config.json is {self.vocabulary_size}. "
                "Make them equal."
            )
        if self.max_model_len > parameters.T:
            raise ValueError(
                f"A sequence of the integer model has at most {parameters.T} tokens, but vLLM "
                f"would allow {self.max_model_len}. Pass --max-model-len {parameters.T} or less."
            )
        self.release()
        self.integer_model = loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Raises ValueError: only the loaders that come with vLLM call this."""
        raise ValueError(
            "The integer model needs this package's loader. Start vLLM with --load-format "
            "integer_model."
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Raises NotImplementedError: vLLM requires the method, but calls it only for inputs other
        than token ids.
        """
        raise NotImplementedError("The integer model takes token ids only.")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Feeds one step's tokens to the integer model.

        vLLM expects one row of hidden states per token and later passes some of the rows to
        compute_logits(). Here a row is a single number: the greedy choice for the next token,
        plus 1, or 0 if there is no choice. The rows are float32, because one of vLLM's start-up
        runs accepts only floats; ids below 2^24 are exact in float32.
        """
        ids, where = input_ids.tolist(), positions.tolist()
        if len(where) > 1 and not any(where):
            # Several tokens, all at position 0: one of vLLM's dummy runs, which measure or warm up
            # the GPU without a request. A dummy run of one token looks like the start of a sequence
            # and is treated as one. This assumes, unchecked, that vLLM makes one-token dummy runs
            # only before it serves requests: one in the middle of a request would make its next
            # step raise or, at position 1, continue from the dummy token.
            return torch.zeros(len(where), 1, dtype=torch.float32, device=input_ids.device)
        if where != list(range(where[0], where[0] + len(where))) or where[0] not in (
            0,
            self.length,
        ):
            raise RuntimeError(
                "The integer model handles one sequence at a time and expected position 0 or "
                f"{self.length} next, but vLLM sent the positions "
                f"{where[:8]}{' and so on' if len(where) > 8 else ''}. Check that vLLM runs with "
                f'the options in {GUIDE}, "Settings".'
            )
        outside = [
            position for position, token in zip(where, ids) if not 0 <= token < self.vocabulary_size
        ]
        if outside:
            # The rest of vLLM's step expects ids of the vocabulary, so the step must not go on.
            raise RuntimeError(
                f"The token at position {outside[0]} of the sequence has an id outside the "
                f"vocabulary (0 to {self.vocabulary_size - 1}), so the integer model stops the "
                "server. With the settings that this model requires, vLLM rejects such a request "
                f'before it gets here ({GUIDE}, "What a client can do"). Restart the server.'
            )
        if where[0] == 0:
            self.release()
            self.sequence = self.integer_model.new_sequence()
        elif len(ids) == 1 and self.last_choice not in (None, ids[0]):
            logger.warning(
                "vLLM served the token at position %d, but the integer model chose another one. "
                "A setting of the request or of the server has overruled the model, so this "
                "reply is not the model's greedy reply.",
                where[0],
            )
            self.last_choice = None  # one warning per sequence
        # The choices for a sequence that the class has ended.
        choices = torch.full((len(ids),), self.end_token, dtype=torch.int64)
        if not self.ended:
            try:
                choices = self.integer_model.extend(self.sequence, ids)
            except integer_model.RangeConditionFailed as failure:
                # The prover produces no proof for this sequence. A model class cannot fail a single
                # request, so this one ends the reply instead.
                logger.error(
                    "%s The sequence ends here, with an end-of-sequence token that the integer "
                    "model did not choose. Only this reply is affected.",
                    failure,
                )
                self.release()
                self.ended = True
        if (
            choices.shape != (len(ids),)
            or choices.min() < 0
            or choices.max() >= self.vocabulary_size
        ):
            raise RuntimeError(
                "The package integer_model returned something other than one token id for each "
                "token. This is a fault in the package, not in the request."
            )
        if self.ended:
            self.last_choice = None  # the model has chosen nothing to compare the next token with
        elif where[0] == 0 or self.last_choice is not None:
            self.last_choice = int(choices[-1])
        self.length = where[-1] + 1
        return (choices + 1).to(dtype=torch.float32, device=input_ids.device).unsqueeze(1)

    def release(self) -> None:
        """Forgets the running sequence."""
        if self.sequence is not None:
            self.integer_model.free(self.sequence)
        self.sequence, self.length, self.last_choice, self.ended = None, 0, None, False

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns a row of logits for each given row of forward()'s result.

        A row without a choice gets equal logits. Such rows come only from dummy runs, one of which
        passes random numbers between 0 and 1 instead of forward()'s rows; the conversion to
        integers turns them into 0.
        """
        choices = hidden_states[:, 0].to(torch.int64) - 1
        chosen = choices >= 0
        logits = torch.zeros(
            len(choices), self.vocabulary_size, dtype=torch.float32, device=hidden_states.device
        )
        logits[chosen] = OTHER_LOGIT
        logits[chosen, choices[chosen]] = CHOSEN_LOGIT
        return logits
