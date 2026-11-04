# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The tests' two small example models, their token windows and splits into calls, and edits to a
parameter file's text.
"""

import functools
import pathlib
import re

import numpy

import recorded_values
from block_tools import PACKAGE, SMALL_EXAMPLE
from integer_model import IntegerModel
from integer_model.parameters import Parameters
from integer_model.synthetic_weights import SyntheticWeights

# tiny_without_clamps: 4 layers, width 64, 8 experts, windows of 16 tokens, without the lines `rcl`
# (the router's clamp), `gcl` (the clamp on the SiLU table's key) and `xl` (a residual-stream
# value's fifth limb). Their effect: docs/model.md, "10 Parameters that change the function".
# target_small_f2: 8 layers, width 512, 4 experts, windows of 64 tokens, with all three.
# In both, layers l with l mod 4 = 3 apply no rotary position encoding.
PARAMETER_FILES = {
    "tiny_without_clamps": PACKAGE / "tests/configurations/tiny_without_clamps/policy.txt",
    "target_small_f2": SMALL_EXAMPLE / "policy.txt",
}
# recorded_values.tsv has no case for a window changed or added here.
FIXED_WINDOWS = {
    "tiny_without_clamps": (5, 106, 81, 77, 148, 25, 124, 69, 171, 55, 34, 136, 196, 61, 150, 9),
    # fmt: off
    "target_small_f2": (
        243,
        486,
        218,
        461,
        192,
        436,
        167,
        411,
        142,
        385,
        117,
        360,
        91,
        335,
        66,
        310,
        41,
        284,
        16,
        259,
        503,
        234,
        477,
        209,
        452,
        183,
        427,
        158,
        402,
        133,
        376,
        108,
        351,
        83,
        326,
        57,
        301,
        32,
        275,
        7,
        250,
        494,
        225,
        468,
        200,
        443,
        174,
        418,
        149,
        393,
        124,
        367,
        99,
        342,
        74,
        317,
        48,
        292,
        23,
        266,
        510,
        241,
        485,
        216,
    ),
    # fmt: on
}
RANDOM_WINDOWS = {"tiny_without_clamps": 8, "target_small_f2": 2}
RANDOM_SPLITS = {"tiny_without_clamps": 3, "target_small_f2": 1}


def windows(model_name: str) -> dict[str, tuple[int, ...]]:
    """A model's test windows of T tokens, by name."""
    p = Parameters.read(PARAMETER_FILES[model_name])
    found = {"fixed": FIXED_WINDOWS[model_name]}
    for number in range(RANDOM_WINDOWS[model_name]):
        random = numpy.random.default_rng(1000 + number)
        found[f"random_{number}"] = tuple(int(token) for token in random.integers(0, p.V, p.T))
    # With one token repeated, a layer without rotary position encoding has equal keys at all
    # positions and so equal scores.
    found["one_token_repeated"] = (p.V - 1,) * p.T
    return found


def splits(model_name: str) -> dict[str, list[int]]:
    """Ways to divide a window of T tokens into calls of IntegerModel.extend, by name: each
    call's number of tokens.
    """
    T = Parameters.read(PARAMETER_FILES[model_name]).T
    found = {"one_call": [T], "one_token_per_call": [1] * T}
    for number in range(RANDOM_SPLITS[model_name]):
        random = numpy.random.default_rng(2000 + number)
        cuts = numpy.sort(
            random.choice(numpy.arange(1, T), size=int(random.integers(1, T // 2)), replace=False)
        )
        found[f"random_chunks_{number}"] = [int(length) for length in numpy.diff([0, *cuts, T])]
    return found


@functools.cache
def load_model(model_name: str, device: str) -> IntegerModel:
    parameter_file = PARAMETER_FILES[model_name]
    return IntegerModel.load(
        parameter_file, SyntheticWeights(Parameters.read(parameter_file), device), device
    )


def recorded_forward(model_name: str, window: tuple[int, ...]) -> recorded_values.Case:
    """The recorded values for one window of an example model."""
    parameter_file = PARAMETER_FILES[model_name]
    return recorded_values.forward(parameter_file, parameter_file.parent / "KL.bin", window)


def with_line(text: str, name: str, replacement: str) -> str:
    """The parameter file's text with the one line "<name> <digits>", line feed included,
    replaced.
    """
    text, count = re.subn(rf"^{name} \d+\n", replacement, text, flags=re.MULTILINE)
    assert count == 1, f"The parameter file has {count} lines named {name}; one was expected."
    return text


def with_lines(values: dict[str, int]):
    """A function that replaces the values of the given lines in a parameter file's text."""

    def change(text: str) -> str:
        for name, value in values.items():
            text = with_line(text, name, f"{name} {value}\n")
        return text

    return change


def read_changed(text: str, folder: pathlib.Path) -> Parameters:
    """Parameters.read's result for the given text, written into `folder` beside a copy of
    target_small_f2's quantization scales.
    """
    (folder / "KL.bin").write_bytes((SMALL_EXAMPLE / "KL.bin").read_bytes())
    (folder / "policy.txt").write_text(text)
    return Parameters.read(folder / "policy.txt")
