# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The integer model (docs/model.md) in PyTorch: a Llama 4 forward pass in exact integer arithmetic.

A small model with synthetic weights, run from the top of the repository with the package installed:

    from integer_model import IntegerModel, Parameters, SyntheticWeights

    parameters = Parameters.read("prover/examples/target_small_f2/policy.txt")
    model = IntegerModel(parameters, SyntheticWeights(parameters), "cpu")
    sequence = model.new_sequence()  # room for parameters.T tokens
    token = int(model.extend(sequence, [1, 2, 3])[-1])  # the greedy choice after the prompt
    token = int(model.extend(sequence, [token])[0])  # decoding
    model.free(sequence)

load_from_folder() loads a model directory, described in
integrations/vllm/integer_model/OPERATING.md, "The files of a model".
extend(..., seed=...) samples at temperature 1 with the public noise of a request's seed (noise.py).
"""

from .model import IntegerModel, RangeConditionFailed, Sequence, WindowFull, load_from_folder
from .parameters import ParameterError, Parameters
from .synthetic_weights import SyntheticWeights
from .weights import NotEnoughMemory, WeightsInMemory, WeightSource, WeightsRejected

__all__ = [
    "IntegerModel",
    "NotEnoughMemory",
    "ParameterError",
    "Parameters",
    "RangeConditionFailed",
    "Sequence",
    "SyntheticWeights",
    "WeightSource",
    "WeightsInMemory",
    "WeightsRejected",
    "WindowFull",
    "load_from_folder",
]
