# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Reads the integer model's expected values (recorded_values.tsv) and compares IntegerModel with
them (assert_model_equals_recorded).

A separate C++ implementation of docs/model.md computed them for test_against_recorded_values.py and
test_arithmetic_edges.py: for a forward pass, every named intermediate value at every layer and
position, the logits, the greedy choices (`first_choice`) and the logit gaps.

Limit: that implementation is not part of this repository, so the file cannot be regenerated, and a
new case cannot be added.
Limit: most arrays are too large to keep (42 MB for one window of 64 tokens), so the file has only
the shape and a digest (digest()) of all but KEPT_WHOLE: a mismatch names the array, not the entry.
Limit: that implementation's function for the output head runs only while it makes a proof. The
logits, greedy choices and logit gaps come from a loop written to follow it.

forward(), weights() and functions() find a case by a digest of its inputs (case_key): a test whose
inputs have changed fails with "no recorded values" and never passes against another case's values.

The file's lines have fields separated by tabs. "case", a key and a kind open a case. Each further
line is one array: its name, its layer or "-", its shape ("64x512"; "-" for a single number) and its
digest, or "=" and the entries, separated by commas. A forward pass in which a range condition fails
has only "defined" = 0 and "message", that implementation's sentence on the failure.
"""

import dataclasses
import functools
import hashlib
import pathlib

import numpy
import torch

FILE = pathlib.Path(__file__).resolve().parent / "recorded_values.tsv"

# An array's key: its name and its layer, None outside the layers.
Key = tuple[str, int | None]

# The parameter file's lines that state the model (docs/model.md, "2 Shape parameters" to
# "4 Public tables" and "10 Parameters that change the function"), and the synthetic weights' seed.
# No other line changes IntegerModel's values.
MODEL_LINES = (
    "lt L d H G dh f fe NE V eps1_lo eps1_hi eps2_lo eps2_hi cs theta0_lo theta0_hi seed_lo "
    "seed_hi S rx MF KF epsF_lo epsF_hi eg smax rcl gcl xl"
).split()

# The arrays whose entries the file has: tests read single entries, some before they run the model.
KEPT_WHOLE = ("defined", "first_choice", "gap", "norm_defined")


def _bytes_of(array) -> bytes:
    """The shape ("64x512"), a newline, and the entries as 64-bit little-endian integers. A plain
    integer counts as an array of one entry.
    """
    array = numpy.ascontiguousarray(numpy.asarray(array), dtype="<i8")
    if array.ndim == 0:
        array = array.reshape(1)
    return "x".join(map(str, array.shape)).encode("ascii") + b"\n" + array.tobytes()


def digest(array) -> str:
    """The first 128 bits of the SHA-256 of an array's shape and entries."""
    return hashlib.sha256(_bytes_of(array)).hexdigest()[:32]


def case_key(
    kind: str,
    parameter_file: pathlib.Path | None = None,
    scales_file: pathlib.Path | None = None,
    arrays: dict | None = None,
) -> str:
    """A digest of everything that a case's values depend on: its kind, the parameter file's
    MODEL_LINES and the quantization scales' bytes (both files or neither), and the arrays by name.
    """
    hashed = hashlib.sha256(kind.encode("ascii") + b"\n")
    if parameter_file is not None and scales_file is not None:
        words = (line.split() for line in pathlib.Path(parameter_file).read_text().splitlines())
        lines = {w[0]: w[1] for w in words if len(w) == 2 and w[0] in MODEL_LINES}
        for name in MODEL_LINES:
            hashed.update(f"{name} {lines.get(name, 'absent')}\n".encode("ascii"))
        hashed.update(pathlib.Path(scales_file).read_bytes())
    for name, array in sorted((arrays or {}).items()):
        hashed.update(name.encode("ascii") + b"\n" + _bytes_of(array))
    return hashed.hexdigest()[:32]


@dataclasses.dataclass
class Case:
    """One case of the file. arrays maps a key to (shape, digest), in the order of evaluation; whole
    maps the keys of KEPT_WHOLE to their entries.
    """

    kind: str
    arrays: dict[Key, tuple[str, str]] = dataclasses.field(default_factory=dict)
    whole: dict[Key, numpy.ndarray] = dataclasses.field(default_factory=dict)
    message: str = ""

    def difference(self, name: str, layer: int | None, found: numpy.ndarray) -> str | None:
        """None if the found array is the recorded one. Otherwise a sentence that names it."""
        where = name if layer is None else f"{name} of layer {layer}"
        if (name, layer) in self.whole:
            return difference(name, layer, self.whole[(name, layer)], found)
        shape, recorded = self.arrays[(name, layer)]
        found_shape = "x".join(map(str, found.shape))
        if found_shape != shape:
            return f"{where} has shape {found_shape} in the model; {shape} is recorded."
        if digest(found) != recorded:
            return f"{where} differs from the recorded array (shape {shape}), by its digest."
        return None


@functools.cache
def _cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    case = Case("")  # replaced at the file's first line
    for line in FILE.read_text().splitlines():
        name, second, third, *rest = line.split("\t")
        assert name == "case" or cases, f"{FILE.name} must begin with a line that opens a case."
        if name == "case":
            case = cases[second] = Case(third)
        elif name == "message":
            case.message = rest[0]
        else:
            key = (name, None if second == "-" else int(second))
            if rest[0].startswith("="):
                entries = numpy.array([int(entry) for entry in rest[0][1:].split(",")], dtype="<i8")
                shape = () if third == "-" else tuple(map(int, third.split("x")))
                case.whole[key] = entries.reshape(shape)
            else:
                case.arrays[key] = (third, rest[0])
    return cases


def _find(key: str, what: str) -> Case:
    assert key in _cases(), (
        f"{FILE.name} has no recorded values for {what}: the test's inputs are not those for which "
        "the values were recorded."
    )
    return _cases()[key]


def forward(parameter_file, scales_file, tokens, replacements=None) -> Case:
    """The recorded forward pass over a window of exactly T tokens.

    replacements: the chosen weights and embedding rows of chosen_model.py, if the case has them.
    """
    arrays = {"tokens": numpy.asarray(tokens)} | (replacements or {})
    return _find(case_key("forward", parameter_file, scales_file, arrays), "this model and window")


def weights(parameter_file, scales_file) -> Case:
    """Every synthetic weight matrix, indexed [input][output]."""
    return _find(case_key("weights", parameter_file, scales_file), "this model's weights")


def functions(inputs: dict) -> Case:
    """Single steps of the arithmetic on the given inputs: "rq" from "rq_a" and "rq_k", "clamp12"
    from "clamp12_in", "isqrt" from "isqrt_hi" and "isqrt_lo", and the norm's sigma, Rc, w and y
    ("norm_sigma" and so on) from "norm_x", "norm_M", "norm_K", "norm_eps" and "norm_limbs".

    "norm_defined" states for each row of norm_x whether every range condition holds. The norm's
    other arrays have only the rows for which it does.
    """
    return _find(case_key("functions", arrays=inputs), "these inputs")


class Recorder:
    """Collects what IntegerModel.extend reports through its argument record, over several calls."""

    def __init__(self):
        self.parts: dict[Key, list[torch.Tensor]] = {}

    def __call__(self, name: str, layer: int | None, tensor: torch.Tensor) -> None:
        self.parts.setdefault((name, layer), []).append(tensor.detach().cpu().clone())

    def arrays(self) -> dict[Key, numpy.ndarray]:
        """Each value at all positions fed so far: the first axis is the position."""
        return {key: torch.cat(parts).numpy() for key, parts in self.parts.items()}


def difference(
    name: str, layer: int | None, expected: numpy.ndarray, found: numpy.ndarray
) -> str | None:
    """None if the two arrays are equal. Otherwise a sentence that names their two shapes, or the
    first entry in which they differ.
    """
    where = name if layer is None else f"{name} of layer {layer}"
    if expected.shape != found.shape:
        return f"{where} has shape {found.shape} in the model; {expected.shape} is recorded."
    differing = numpy.argwhere(expected != found)
    if len(differing) == 0:
        return None
    position, *index = (int(i) for i in differing[0])
    return (
        f"{where} differs at position {position}, index {index}: the model has "
        f"{found[tuple(differing[0])]}; {expected[tuple(differing[0])]} is recorded. "
        f"{len(differing)} of {expected.size} entries differ."
    )


def feed(model, window, split, next_tokens, record=None):
    """Feeds a window to a new sequence of an IntegerModel, split[i] tokens in the i-th call.

    next_tokens gives, for each position, the token whose logit gap is wanted.
    """
    sequence = model.new_sequence()
    choices, gaps, start = [], [], 0
    for length in split:
        end = start + length
        choice, gap = model.extend(
            sequence, list(window[start:end]), gap_for=list(next_tokens[start:end]), record=record
        )
        choices.append(choice)
        gaps.append(gap)
        start = end
    return sequence, torch.cat(choices).numpy(), torch.cat(gaps).numpy()


def assert_model_equals_recorded(model, window, split, case: Case) -> dict[Key, numpy.ndarray]:
    """Feeds a window to the model and compares with a case of forward() every value that the model
    records, the greedy choices and logit gaps that it returns, and the cached keys and values after
    the last call. The assertion's message names the first array that differs.

    Returns the model's arrays, which then equal the recorded ones.
    """
    assert case.whole[("defined", None)], "By the recorded values, a range condition fails here."
    first_choice = case.whole[("first_choice", None)]
    # The gap at a position is that of the window's next token. After the last token it is that of
    # the greedy choice.
    next_tokens = [*window[1:], int(first_choice[-1])]
    recorder = Recorder()
    sequence, choices, gaps = feed(model, window, split, next_tokens, recorder)
    found = recorder.arrays()

    # The recorded arrays are in the order of evaluation, so the first difference is the earliest.
    expected = [*case.arrays, ("first_choice", None)]
    for name, layer in expected:
        assert (name, layer) in found, f"The model did not record {name} of layer {layer}."
        message = case.difference(name, layer, found[(name, layer)])
        assert message is None, message
    unknown = set(found) - set(expected)
    assert not unknown, f"The model recorded values that the file does not have: {sorted(unknown)}"

    message = difference("the returned greedy choice", None, first_choice, choices)
    assert message is None, message
    message = difference("the returned logit gap", None, case.whole[("gap", None)], gaps)
    assert message is None, message
    for layer in range(model.parameters.L):
        for name, cached in zip(("k", "v"), sequence.cached(layer)):
            message = case.difference(name, layer, cached.cpu().numpy().astype("int64"))
            assert message is None, f"The cached array: {message}"
    model.free(sequence)
    return found
