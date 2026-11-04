# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A plug-in that lets an unmodified vLLM 0.29.0 serve the integer model (package integer_model).

vLLM calls register() in each of its processes, through the entry point group "vllm.general_plugins"
(integrations/vllm/integer_model/pyproject.toml). vLLM documents this way to add a model from
outside its repository: https://docs.vllm.ai/en/latest/design/plugin_system/ and
https://docs.vllm.ai/en/latest/contributing/model/registration/.

Modules:
    model       the model class
    loader      the loader
    reference   the loader's option "reference_directory" (needs no vLLM)
    example     python3 -m integer_model_vllm.example: creates the examples' files (needs no vLLM)
    float_pair  the example small_float_pair
"""

# The entry of "architectures" in the model directory's config.json.
ARCHITECTURE = "IntegerModelForCausalLM"
LOAD_FORMAT = "integer_model"  # the value of --load-format


def register() -> None:
    """Registers the model class and the loader with vLLM, which may call this more than once.

    Source: the example in "Plugin System" in vLLM's documentation (vLLM: Apache License 2.0).
    """
    from vllm import ModelRegistry

    # This import registers the loader. The idea of registering a loader from this entry
    # point is that of vllm-gguf-plugin 0.0.5 (https://github.com/vllm-project/vllm-gguf-plugin,
    # Apache License 2.0).
    from . import loader  # noqa: F401

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        # Named by a string, as vLLM's documentation recommends: importing the class here could
        # initialize CUDA before a fork.
        ModelRegistry.register_model(ARCHITECTURE, f"{__name__}.model:{ARCHITECTURE}")
