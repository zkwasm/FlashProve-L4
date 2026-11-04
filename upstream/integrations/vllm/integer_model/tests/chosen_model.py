# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A small model whose weights and embeddings a test chooses, for test_arithmetic_edges.py."""

import dataclasses
import math
import struct

import numpy
import pytest
import torch

import recorded_values
from example_models import PARAMETER_FILES, with_lines
from integer_model import IntegerModel, WeightsInMemory
from integer_model.parameters import LayerScales, Parameters
from integer_model.synthetic_weights import matrix_layout
from recorded_values import assert_model_equals_recorded

# The parameter files from which the chosen models are derived. The public tables depend on T, dh
# and a few other lines, and a parameter file states the tables' digests, so a chosen model keeps
# those lines and changes the rest of the shape.
BASES = {
    # T = 64, dh = 128.
    "with_clamps_and_five_limbs": PARAMETER_FILES["target_small_f2"],
    # T = 16, dh = 16.
    "without_clamps_with_four_limbs": PARAMETER_FILES["tiny_without_clamps"],
}
ALL_BASES = pytest.mark.parametrize("base", BASES)


class ChosenModel:
    """A chosen model as an IntegerModel, and its recorded values (recorded_values.py).

    The vocabulary has T tokens, and a window is always the tokens 0 .. T - 1. A test chooses T rows
    of d integers: row t is the embedding of token t, the row of the residual stream that enters the
    first layer at position t. All weights are 0 until a test sets them.

    The generator of synthetic weights takes all matrices of one kind, such as every layer's Wq,
    from one stream of numbers, and matrix_layout() gives each matrix's place in its stream. The
    chosen weights are stored as such streams (`streams`): the implementation that computed the
    recorded values read them in that form.

    That layout restricts the tests. In the stream of Wg, layer 1's matrix (d rows of fe entries)
    overlaps rows d / 2 to d - 1 of layer 0's matrix (d rows of f = 2 fe entries): row d / 2 + i of
    layer 0 is rows 2 i and 2 i + 1 of layer 1. The same holds for Wu. So that a weight set for
    layer 1 has no effect in layer 0, new_inputs() returns only inputs below d / 2, and a row that
    row() or on() makes from them has 0 in entries d / 2 to d - 2. Entry d - 1 is not 0 (row() fills
    it), so a test must also leave rows d - 2 and d - 1 of layer 1's matrices at 0: they are row
    d - 1 of layer 0's.
    """

    def __init__(
        self, folder, base: str, lines: dict[str, int], scales: list[LayerScales], device: str
    ):
        T = Parameters.read(BASES[base]).T
        text = with_lines({**lines, "L": len(scales), "V": T})(BASES[base].read_text())
        self.folder = folder
        self.parameter_file, self.scales_file = folder / "policy.txt", folder / "KL.bin"
        self.parameter_file.write_text(text)
        # KL.bin: 13 words a layer, then kr, kgR and kuR of each expert layer.
        words = [word for k in scales for word in dataclasses.astuple(k)[:13]]
        words += [
            word
            for layer, k in enumerate(scales)
            if layer % 2 == 1
            for word in (k.kr, k.kgR, k.kuR)
        ]
        self.scales_file.write_bytes(struct.pack(f"<{len(words)}I", *words))
        self.parameters = p = Parameters.read(self.parameter_file)
        self.device = device
        self.window = list(range(p.T))
        self.matrices = [("Wh", None, None)]
        for layer in range(p.L):
            self.matrices += [
                (name, layer, None) for name in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")
            ]
            if p.has_experts(layer):
                self.matrices.append(("Wr", layer, None))
                self.matrices += [
                    (name, layer, e) for name in ("Wgr", "Wur", "Wdr") for e in range(p.NE)
                ]
        # Streams 2 to 14. The embedding's, 13, stays empty: recorded() passes the rows as x0.
        lengths = dict.fromkeys(range(2, 15), 0)
        for key in self.matrices:
            stream, inputs, outputs, start = matrix_layout(p, *key)
            lengths[stream] = max(lengths[stream], start + inputs * outputs)
        self.streams = {
            stream: numpy.zeros(length, dtype=numpy.int64) for stream, length in lengths.items()
        }
        self._next_input = 0
        self._model = None

    def new_inputs(self, count: int) -> int:
        """Reserves the next count inputs of a row for one purpose and returns the first of them."""
        first = self._next_input
        self._next_input += count
        assert self._next_input <= self.parameters.d // 2, (
            "The chosen model needs a larger width d."
        )
        return first

    def matrix(
        self, name: str, layer: int | None = None, expert: int | None = None
    ) -> numpy.ndarray:
        """One weight matrix, indexed [input][output]. Assigning to its entries sets the weights."""
        stream, inputs, outputs, start = matrix_layout(self.parameters, name, layer, expert)
        return self.streams[stream][start : start + inputs * outputs].reshape(inputs, outputs)

    def set_sums(self, name, first_input: int, sums, layer=None, expert=None) -> None:
        """Sets weights so that a row with (1, 2047, 2047) at inputs first_input to first_input + 2
        adds sums[j] to output j of the product. 2,047 passes the norm's clamp, and three weights
        within -4,095 .. 4,095 reach any sum up to about 16.8 million in absolute value.
        """
        matrix = self.matrix(name, layer, expert)
        for j, total in enumerate(sums):
            large = (total + 1023) // 2047
            first_large = min(max(large, -4095), 4095)
            digits = (total - 2047 * large, first_large, large - first_large)
            assert all(abs(digit) <= 4095 for digit in digits), (
                f"The sum {total} is too large for three weights."
            )
            matrix[first_input : first_input + 3, j] = digits

    def row(self, entries: dict[int, int]) -> numpy.ndarray:
        """A row x of the residual stream that a norm leaves unchanged before its clamp (symbols of
        docs/model.md, "5 The norm").

        The row has the given entries and one more, at index d - 1, chosen so that
        sigma = isqrt(Vx + eps) is exactly 2^14. The norm must have M - K = 14 and eps = 1, as with
        scales_of() and the eps lines of both bases. Then Rc = 2^(M - 14) = 2^K, and
        w_k = rq(2^K x_k, K) = x_k. The norm's result is w clamped to -2,047 .. 2,047, so entries in
        that range pass unchanged. Weights that multiply entry d - 1 must be 0; this is not checked.
        """
        x = numpy.zeros(self.parameters.d, dtype=numpy.int64)
        for index, value in entries.items():
            x[index] = value
        missing = (1 << 28) - 1 - int((x * x).sum())
        # The smallest integer whose square is at least the missing amount.
        x[-1] = math.isqrt(missing - 1) + 1
        assert math.isqrt(int((x * x).sum()) + 1) == 1 << 14
        return x

    def on(self, *first_inputs: int, flags=()) -> numpy.ndarray:
        """The row that has (1, 2047, 2047) at each of the given triples of inputs and 1 at each of
        the given flags.
        """
        entries = {
            first + i: value for first in first_inputs for i, value in enumerate((1, 2047, 2047))
        }
        return self.row(entries | {flag: 1 for flag in flags})

    def recorded(self, rows: numpy.ndarray) -> recorded_values.Case:
        """The recorded values for the window whose tokens have the given rows as embeddings."""
        replacements = {"x0": rows} | {
            f"stream_{stream}": entries for stream, entries in self.streams.items()
        }
        return recorded_values.forward(
            self.parameter_file, self.scales_file, self.window, replacements
        )

    def model(self, rows: numpy.ndarray) -> IntegerModel:
        """One PyTorch model for all calls, with the chosen weights and `rows` as embeddings."""
        matrices = {
            key: torch.from_numpy(self.matrix(*key).T.astype(numpy.int16)) for key in self.matrices
        }
        weights = WeightsInMemory(
            matrices, torch.from_numpy(numpy.ascontiguousarray(rows)), self.device
        )
        if self._model is None:
            self._model = IntegerModel(self.parameters, weights, self.device)
        self._model.weights = weights
        return self._model

    def splits(self) -> list[list[int]]:
        T = self.parameters.T
        return [[T], [1] * T, [3, 1, T // 2, T - T // 2 - 4]]

    def assert_equals_recorded(self, rows: numpy.ndarray) -> dict:
        """Compares the model with the recorded values, feeding the window in each way that splits()
        lists (tokens per call). Returns the model's arrays, which then equal the recorded ones.
        """
        case = self.recorded(rows)
        found = [
            assert_model_equals_recorded(self.model(rows), self.window, split, case)
            for split in self.splits()
        ]
        return found[0]


def scales_of(base: str, **changes: int) -> LayerScales:
    """One layer's quantization scales for a chosen model.

    Both norms have the exponents of the base's final norm, for which M - K = 14. The products with
    Wv, Wo, Wg, Wu, Wr, Wgr and Wur have the shift 1, so a sum S gives the value rq(S, 1): S = 2 N
    gives N exactly, and S = 2 N - 1 gives N through an exact tie. Rotary encoding is off, so the
    rotary step only multiplies by 2^cs = 2^12, and the shift 13 of q and k also acts as rq(S, 1).
    """
    p = Parameters.read(BASES[base])
    values = dict(
        kq=13,
        kk=13,
        kv=1,
        ko=1,
        kg=1,
        ku=1,
        kd=10,
        M1=p.MF,
        K1=p.KF,
        M2=p.MF,
        K2=p.KF,
        sh=8,
        rp=0,
        kr=1,
        kgR=1,
        kuR=1,
    )
    return LayerScales(**(values | changes))


def shape(base: str, heads: int = 1, G: int = 1, f: int = 4, NE: int = 2) -> dict[str, int]:
    """The shape lines of a chosen model with at least the given number of query heads and a width d
    of at least 128.

    The width is d = H dh. The model's definition allows other widths, but the proof programs
    `cp_commit`, `cp_prove` and `cp_verify` reject them (rejection name `shape-heads`).
    """
    dh = Parameters.read(BASES[base]).dh
    H = max(heads, 128 // dh)
    return {"d": H * dh, "H": H, "G": G, "f": f, "fe": f // 2, "NE": NE}
