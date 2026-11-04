# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Where the integer model gets its weights.

The integer model (IntegerModel in model.py; specified in docs/model.md) stores no weights: each
time it needs a matrix, it asks a weight source (WeightSource) for it and uses it for one product.
"""

from typing import Protocol

import torch

from .parameters import WEIGHT_MAX, WEIGHT_MIN


class WeightsRejected(ValueError):
    """The weights cannot be used.

    For example, a file is malformed or does not match the parameter file (`policy.txt`), or a
    weight is out of range.
    """


class NotEnoughMemory(MemoryError):
    """Loading the weights would need more host memory than is free. Raised before any memory for
    the weights is allocated."""


class WeightSource(Protocol):
    """The integer weight matrices and the embedding of one model
    (docs/model.md, "3 Weights and quantization scales").

    d, dq, dkv, dh, f, fe, NE and V are the shape parameters of docs/model.md, "2 Shape parameters".
    `device` must be the model's device (not checked): token ids arrive on it, and every returned
    tensor is on it.

    matrix(name, layer, expert) takes these names, the symbols of docs/model.md, section 3:

        name   layer   expert   shape (outputs, inputs)   which matrix
        "Wq"   yes     no       (dq, d)                   query projection
        "Wk"   yes     no       (dkv, d)                  key projection
        "Wv"   yes     no       (dkv, d)                  value projection
        "Wo"   yes     no       (d, dq)                   attention's output projection
        "Wg"   yes     no       (f, d) or (fe, d)         gate of a dense layer or the shared expert
        "Wu"   yes     no       (f, d) or (fe, d)         up projection of the same
        "Wd"   yes     no       (d, f) or (d, fe)         down projection of the same
        "Wr"   yes     no       (NE, d)                   router (mixture-of-experts layers only)
        "Wgr"  yes     yes      (fe, d)                   gate of one routed expert
        "Wur"  yes     yes      (fe, d)                   up projection of one routed expert
        "Wdr"  yes     yes      (d, fe)                   down projection of one routed expert
        "Wh"   no      no       (V, d)                    output matrix (the head)

    docs/model.md writes the input index first ("1 Notation"), so matrix("Wq", l)[j][k] is its
    Wq[k][j]. The outputs of "Wq", "Wk" and "Wv" and the inputs of "Wo" are ordered by head: entry
    h * dh + c is entry c of head h.

    A matrix must have an integer dtype (torch.int16 is enough). Every entry must be in -4,096 ..
    4,095, the range of the 13 bits that the parameter file's format fixes (its lines wb_*, except
    wb_emb; WEIGHT_BITS in parameters.py). The products run in float64 (arithmetic.exact_product()),
    and the bounds that make them exact assume weights in this range; beyond it a product can round,
    and nothing reports it. The model does not test the range: the source guarantees it, as
    BlockWeights (block_weights.py) does by testing every weight once at loading. That every
    committed matrix weight is in this range, and that every committed embedding entry is a number
    of four signed bytes in base 256 (docs/model.md, section 3), is for the commitment's proof to
    show; that it does is argued, not proved (docs/limits.md).

    A returned matrix is valid until the next call to matrix(). A source may therefore decode every
    matrix into the same buffer.

    embedding_rows(token_ids) takes a one-dimensional int64 tensor of n token ids on `device`, each
    in 0 .. V - 1, and returns the rows Emb[token_ids[0]], .., Emb[token_ids[n - 1]] of the
    embedding as an int64 tensor of shape (n, d). The model bounds an embedding entry only through
    conditions 8 and 1 of layer 0's first norm (docs/model.md, "9 Range conditions").
    BlockWeights accepts entries below 2^30 in absolute value."""

    device: torch.device

    def matrix(
        self, name: str, layer: int | None = None, expert: int | None = None
    ) -> torch.Tensor: ...

    def embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor: ...


class WeightsInMemory:
    """A weight source that serves tensors from a dictionary, meant for small models and unit tests.

    The keys of `matrices` are the triples (name, layer, expert) with which WeightSource.matrix is
    called, with None where a matrix has no layer or no expert: ("Wq", 0, None), ("Wgr", 1, 5),
    ("Wh", None, None). `embedding` is the whole embedding, of shape (V, d); it is not checked.
    Of the matrices, the constructor checks only the entries' range (not dtype, shape or keys).
    """

    def __init__(
        self,
        matrices: dict[tuple[str, int | None, int | None], torch.Tensor],
        embedding: torch.Tensor,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        for key, tensor in matrices.items():
            if int(tensor.min()) < WEIGHT_MIN or int(tensor.max()) > WEIGHT_MAX:
                raise WeightsRejected(
                    f"The weight matrix {key} has an entry outside {WEIGHT_MIN} .. {WEIGHT_MAX}."
                )
        self._matrices = {key: tensor.to(self.device) for key, tensor in matrices.items()}
        self._embedding = embedding.to(self.device, torch.int64)

    def matrix(
        self, name: str, layer: int | None = None, expert: int | None = None
    ) -> torch.Tensor:
        return self._matrices[(name, layer, expert)]

    def embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding[token_ids]
