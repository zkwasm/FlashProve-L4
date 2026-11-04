# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The model loader for "--load-format integer_model".

The loaders that come with vLLM fill a model's torch parameters from weight files. The integer
model, a forward pass in exact integer arithmetic, has no torch parameters: the package
integer_model reads its files and manages its memory itself. So this loader passes the model
directory to that package and logs which files were verified.

Source: the class has the form of the example in the docstring of register_model_loader
(vllm/model_executor/model_loader/__init__.py in vLLM 0.29.0, Apache License 2.0), and its methods
have the arguments of BaseModelLoader's abstract methods (base_loader.py in the same directory).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.utils.torch_utils import set_default_torch_dtype

import integer_model
from integer_model.parameters import digests_from_file

from . import LOAD_FORMAT
from .reference import replace_reference_directory

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from .model import IntegerModelForCausalLM

logger = logging.getLogger("vllm." + __name__)  # a child of vLLM's logger, as in model.py

GUIDE = "integrations/vllm/integer_model/OPERATING.md"


@register_model_loader(LOAD_FORMAT)
class IntegerModelLoader(BaseModelLoader):
    """Loads a model directory with integer_model.load_from_folder()."""

    def download_model(self, model_config: ModelConfig) -> None:
        """Does nothing: the model is a local directory."""

    def load_weights(self, model: IntegerModelForCausalLM, model_config: ModelConfig) -> None:
        """Loads the model directory and attaches the result to `model`.

        The dictionary given with --model-loader-extra-config goes to load_from_folder(), whose
        docstring describes the options. After a successful load, the log states what was verified:
        one line for the small files that the option "sha256" can cover and, if the weights are in
        GGUF files, one for those. Each is an INFO line that begins "Verified", or a WARNING.
        """
        options = dict(self.load_config.model_loader_extra_config)
        # The digests of the reference copies are taken here, in memory, and written nowhere.
        copies = options.get("reference_directory")
        replace_reference_directory(options, model_config.model)
        if "sha256_file" in options and "sha256" not in options:
            # The digests come from a file, so that they stand neither on vLLM's command line nor
            # in the arguments that its log prints.
            options["sha256"] = digests_from_file(options.pop("sha256_file"), model_config.model)
        # The package reports its progress on loggers under "integer_model", which vLLM does not
        # configure.
        progress, vllm_logger = logging.getLogger("integer_model"), logging.getLogger("vllm")
        if vllm_logger.handlers and not progress.handlers:
            for handler in vllm_logger.handlers:
                progress.addHandler(handler)
            progress.setLevel(vllm_logger.getEffectiveLevel())
        # vLLM may call this method with another default type for new floating-point tensors, often
        # bfloat16. The package gets PyTorch's usual default, float32.
        with set_default_torch_dtype(torch.float32):
            try:
                loaded = integer_model.load_from_folder(model_config.model, model.device, options)
            except integer_model.ParameterError as error:
                if copies is None:
                    raise
                # The package's sentences name two options that this caller did not give.
                raise integer_model.ParameterError(
                    f'{error} With the option "reference_directory", the SHA-256 digests are those '
                    f"of the copies in {copies!r}: compare a file that is refused for its digest "
                    "with its copy there. Those of wsums.txt and fold.txt are among them if the "
                    'two lie there or "reference_files" is 5, and go only with weights in GGUF '
                    "files."
                ) from None
            model.attach(loaded)

        if options.get("sha256") is None:
            logger.warning(
                "The model's files were not verified: no SHA-256 digests were given "
                "(integer_model.json, policy.txt, KL.bin; for GGUF weights also wsums.txt, "
                "fold.txt). To verify them, keep reference copies outside the model directory and "
                "pass, in --model-loader-extra-config, the directory that holds them as "
                '"reference_directory", or their digests as "sha256_file" (a path). "sha256" takes '
                "the digests themselves, which shows them in the process list and the log. See "
                '%s, "Verify what was loaded".',
                GUIDE,
            )
        else:
            logger.info(
                "Verified: these files have the given SHA-256 digests: %s.",
                ", ".join(options["sha256"]),
            )
        # Only `BlockWeights` (weights from quantized blocks) has these two attributes.
        weights = loaded.weights
        if getattr(weights, "sums_verified", False):
            logger.info(
                "Verified: every weight tensor has the SHA-256 that wsums.txt lists for it."
            )
        if getattr(weights, "files_not_compared", False):
            logger.warning(
                'The weight files were not verified, because "verify_sums" is not true. The '
                "digests of the other files do not cover them: altered weight files of the same "
                "shape load without an error. To check every weight against wsums.txt, set "
                '"verify_sums": true under "weight_settings" in integer_model.json. If you pass '
                "digests, edit your reference copy of integer_model.json, copy it into the model "
                'directory and pass its new digest (with "reference_directory" there is none to '
                'pass: the loader takes it from the copy). See %s, "Verify what was loaded".',
                GUIDE,
            )
