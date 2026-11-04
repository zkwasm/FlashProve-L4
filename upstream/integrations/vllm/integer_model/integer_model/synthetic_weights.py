# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic weights: every weight is a function of the seed in the parameter file.

The parameter file's lines seed_lo and seed_hi give the seed. The GPU programs' generator for
`-weights generated` (GeneratedWeights in prover/prover/weights_generated.cuh) draws its entries by
the same rule, synth_weight() in prover/prover/forward.cuh, from the same numbered streams. One
difference: stream_entries() does not have synth_weight()'s separate rule for entries of 16 bits.
"""

from typing import cast

import torch

from .parameters import Parameters

# The stream (see stream_entries) from which each kind of matrix takes its entries.
_STREAM = {
    "Wq": 2, "Wk": 3, "Wv": 4, "Wo": 5, "Wg": 6, "Wu": 7, "Wd": 8, "Wr": 9,
    "Wgr": 10, "Wur": 11, "Wdr": 12, "Wh": 14,
}  # fmt: skip
_EMBEDDING_STREAM = 13


def matrix_layout(
    p: Parameters, name: str, layer: int | None = None, expert: int | None = None
) -> tuple[int, int, int, int]:
    """The stream, shape and first index of one matrix: (stream, inputs, outputs, start).

    Entry [input][output] of the matrix is entry start + input * outputs + output of the stream. The
    "Wg" and "Wu" matrices of different layers can overlap in their streams, because start depends
    on each layer's hidden width, f or fe. GeneratedWeights has the same overlap.
    """
    hidden = p.fe if layer is not None and p.has_experts(layer) else p.f
    # For the type checker: only names that have a layer or an expert use these two.
    layer, expert = cast(int, layer), cast(int, expert)
    inputs, outputs, start = {
        "Wq": lambda: (p.d, p.dq, layer * p.d * p.dq),
        "Wk": lambda: (p.d, p.dkv, layer * p.d * p.dkv),
        "Wv": lambda: (p.d, p.dkv, layer * p.d * p.dkv),
        "Wo": lambda: (p.dq, p.d, layer * p.dq * p.d),
        "Wg": lambda: (p.d, hidden, layer * p.d * hidden),
        "Wu": lambda: (p.d, hidden, layer * p.d * hidden),
        "Wd": lambda: (hidden, p.d, layer * p.f * p.d),
        "Wr": lambda: (p.d, p.NE, layer * p.d * p.NE),
        "Wgr": lambda: (p.d, p.fe, (layer * p.NE + expert) * p.d * p.fe),
        "Wur": lambda: (p.d, p.fe, (layer * p.NE + expert) * p.d * p.fe),
        "Wdr": lambda: (p.fe, p.d, (layer * p.NE + expert) * p.fe * p.d),
        "Wh": lambda: (p.d, p.V, 0),
    }[name]()
    return _STREAM[name], inputs, outputs, start


def signed64(word: int) -> int:
    """An unsigned 64-bit number as the int64 number with the same bits."""
    word %= 2**64
    return word - 2**64 if word >= 2**63 else word


def shift_right64(words: torch.Tensor, count: int) -> torch.Tensor:
    """Logical right shift of unsigned 64-bit numbers that are stored in an int64 tensor."""
    return (words >> count) & (2 ** (64 - count) - 1)


def mix64(words: torch.Tensor) -> torch.Tensor:
    """One step of SplitMix64 on an int64 tensor whose elements represent unsigned 64-bit numbers.

    The step is the function next() of S. Vigna's splitmix64.c
    (https://prng.di.unimi.it/splitmix64.c, public domain). The generator is that of Steele, Lea and
    Flood, "Fast Splittable Pseudorandom Number Generators" (OOPSLA 2014), with David Stafford's
    Mix13 multipliers. See NOTICE.

    The arithmetic is modulo 2^64. It relies on PyTorch's int64 addition and multiplication keeping
    the low 64 bits of a result that overflows.
    """
    words = words + signed64(0x9E3779B97F4A7C15)
    words = (words ^ shift_right64(words, 30)) * signed64(0xBF58476D1CE4E5B9)
    words = (words ^ shift_right64(words, 27)) * signed64(0x94D049BB133111EB)
    return words ^ shift_right64(words, 31)


def stream_entries(seed: int, stream: int, index: torch.Tensor, bits: int) -> torch.Tensor:
    """The entries of one stream at the given indices, both int64 tensors.

    Entry i of stream st is u - 2^(bits - 1), where u is the low `bits` bits of
    mix64(seed + st * 2^40 + i), read as an unsigned number. The lowest value, -2^(bits - 1), is
    replaced by 0.
    """
    drawn = mix64(index + signed64(seed + (stream << 40))) & (2**bits - 1)
    entries = drawn - 2 ** (bits - 1)
    return torch.where(entries == -(2 ** (bits - 1)), 0, entries)


class SyntheticWeights:
    """A weight source (WeightSource in weights.py) for synthetic models.

    The entries are those of stream_entries, with bits = 13 for a matrix weight, which is then in
    -4,095 .. 4,095, and bits = 20 for an entry of the embedding.

    The source keeps every matrix that it has generated, two bytes per weight, so it is meant for
    small models only.
    """

    def __init__(self, parameters: Parameters, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self._p = parameters
        self._generated: dict[tuple[str, int | None, int | None], torch.Tensor] = {}

    def matrix(
        self, name: str, layer: int | None = None, expert: int | None = None
    ) -> torch.Tensor:
        key = (name, layer, expert)
        if key not in self._generated:
            stream, inputs, outputs, start = matrix_layout(self._p, name, layer, expert)
            index = torch.arange(start, start + inputs * outputs, device=self.device)
            by_input = stream_entries(self._p.seed, stream, index, 13).reshape(inputs, outputs)
            self._generated[key] = by_input.T.to(torch.int16).contiguous()
        return self._generated[key]

    def embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        d = self._p.d
        index = token_ids.to(self.device)[:, None] * d + torch.arange(d, device=self.device)
        return stream_entries(self._p.seed, _EMBEDDING_STREAM, index, 20)
