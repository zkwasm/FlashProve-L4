# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The integer model of docs/model.md, evaluated with a cache of keys and values.

The model is a decoder-only transformer in which every operation is an exact function on integers.
This module follows that document from "5 The norm" to "9 Range conditions" and uses its symbols as
variable names in most places (but `k` is a layer's LayerScales, and the document's k is `key`).
The steps that are specific to the Llama 4 family follow modeling_llama4.py and
configuration_llama4.py of Hugging Face transformers (NOTICE).

A proof is about a fixed number T of token positions, a window. A sequence is the tokens that a
server has put into one window so far.

IntegerModel evaluates the model on a Sequence. load_from_folder() loads a model directory
(`folder` in identifiers). _assert_arithmetic_is_exact() lists "bound (a)" to "bound (g)", which the
comments cite. NOTICE and OPERATING.md are in integrations/vllm/integer_model/.
"""

import json
import logging
import time
from collections.abc import Callable, Sequence as SequenceOf
from pathlib import Path
from typing import overload

import torch

from .arithmetic import (
    FLOAT64_EXACT,
    balanced_low_byte,
    clamp,
    exact_product,
    first_argmax,
    floor_quotient,
    rounded_quotient,
    wide_add,
    wide_isqrt,
    wide_sum_of_squares,
)
from .noise import TEMPERATURE_ONE, check_seed, noise
from .parameters import (
    CLAMPED_MAX,
    WEIGHT_MIN,
    ParameterError,
    Parameters,
    digests_from_file,
    given_digests,
    read_bytes,
)
from .tables import KEY_OFFSET_15, KEY_OFFSET_16, make_tables
from .weights import WeightSource

logger = logging.getLogger(__name__)

TokenIds = SequenceOf[int] | torch.Tensor
# record(name, layer, tensor) receives intermediate values: see IntegerModel.extend.
Record = Callable[[str, int | None, torch.Tensor], None]

_RANGE_CONDITIONS = {
    1: "a norm's sum of squares",
    2: "a norm's quotient",
    3: "a query or key entry before its clamp",
    4: "a value entry before its clamp",
    5: "a router logit",
    6: "a gate's key",
    7: "an up projection",
    8: "a value of the residual stream",
}
# The values that fit 16 bits and that the prover can write as two balanced limbs: as 256 b1 + b0
# with b0 and b1 in -128 .. 127.
_TWO_LIMBS = (-32768, 32639)
# The most noise values that _add_noise computes in one step. noise()'s temporary tensors then take
# roughly 100 bytes a value, about 0.4 GB (measured on a CPU).
_NOISE_ELEMENTS = 1 << 22
_MASKED_SCORE = -(1 << 62)  # the score of a later position; far below any real score


class RangeConditionFailed(Exception):
    """A range condition (docs/model.md, "9 Range conditions") fails at a position, so the prover
    produces no proof for a window that contains it.

    `position` is the first failing position, counted from 0 at the sequence's first token.
    `condition` is the number, in that section's table, of the condition that fails there. If
    several fail there, it is the one that the model evaluates first.
    """

    def __init__(self, position: int, condition: int):
        self.position = position
        self.condition = condition
        super().__init__(
            f"At position {position} of the sequence, {_RANGE_CONDITIONS[condition]} leaves the "
            f"range that the prover accepts (range condition {condition} of the integer model), "
            "so the prover makes no proof for this sequence."
        )


class WindowFull(Exception):
    """The call would take the sequence past the window of T tokens."""


class _RangeChecks:
    """Records, for each of the n positions of one call to extend(), the first range condition that
    fails there.

    A failure does not stop the call: the failing value becomes 0, so that later steps stay within
    their bounds, and extend() raises at the end, for the first failing position. That position's
    values depend on no later position, so the exception does not depend on how the tokens were
    divided into calls. The CPU also need not wait for the GPU at each condition.
    """

    def __init__(self, n: int, device: torch.device):
        # 0: no condition has failed at this position.
        self._first = torch.zeros(n, dtype=torch.int64, device=device)

    def require(self, condition: int, holds: torch.Tensor) -> None:
        """`holds` is a boolean tensor whose first axis is the n positions."""
        fails = ~holds.flatten(1).all(dim=1) if holds.dim() > 1 else ~holds
        self._first = torch.where((self._first == 0) & fails, condition, self._first)

    def admit(self, condition: int, value: torch.Tensor, low: int, high: int) -> torch.Tensor:
        """Requires low <= value <= high and returns value with every offending entry replaced by
        0."""
        holds = (value >= low) & (value <= high)
        self.require(condition, holds)
        return torch.where(holds, value, 0)

    def first_failure(self) -> tuple[int, int] | None:
        """The first failing position of the call, counted from the call's first token, and its
        condition."""
        for row, condition in enumerate(self._first.tolist()):
            if condition:
                return row, condition
        return None


class Sequence:
    """One sequence of tokens: how many were fed, and the cache of their keys and values.

    For each layer, position and key-value head the cache holds the key after the rotary step, the
    rounding shift and the clamp, and the value after the rounding shift and the clamp. Both are in
    -2,047 .. 2,047. Attention reads only these two of an earlier position.
    """

    def __init__(self, p: Parameters, device: torch.device):
        self.length = 0  # tokens fed so far; the next token's position
        self.failure: RangeConditionFailed | None = None
        self.keys = torch.zeros((p.L, p.T, p.dkv), dtype=torch.int16, device=device)
        self.values = torch.zeros((p.L, p.T, p.dkv), dtype=torch.int16, device=device)

    def cached(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The keys and the values of one layer, each of shape (tokens fed so far, dkv)."""
        return self.keys[layer, : self.length], self.values[layer, : self.length]


class IntegerModel:
    """Evaluates the integer model for one set of public parameters and one weight source.

    By design, this is the function that the repository's prover proves (how far the two were
    compared: OPERATING.md, "What has been run"). The same tokens give the same integers on every
    device whose numerical library meets the requirements in the docstrings of
    arithmetic.exact_product() and arithmetic.wide_isqrt().

    The model is a partial function: at eight places an intermediate value must be in a fixed
    interval (the range conditions). extend() checks them with the prover's intervals, which are
    narrower than the function's domain (docs/model.md, "9 Range conditions"), so it can raise
    RangeConditionFailed on an input on which the function is defined. The comparison of the
    prover's own range tests with that section's table has not been made on the released `cp_prove`,
    and nothing has been run for it (docs/limits-all-rows.tsv, row E26).

    A reply made without RangeConditionFailed meets the range conditions only at the positions that
    were evaluated, and a proof needs them at all T positions of its window, padding included
    (OPERATING.md, "What a served reply means"). Whether any window can break a condition is what
    the worst-case check of the committer, `cp_commit`, examines. docs/limits.md explains what a
    passed check gives.
    """

    def __init__(self, parameters: Parameters, weights: WeightSource, device: torch.device | str):
        self.parameters = parameters
        self.weights = weights
        self.device = torch.device(device)
        _assert_arithmetic_is_exact(parameters)
        self.tables = make_tables(parameters, self.device)

    @classmethod
    def load(
        cls, parameter_file: str | Path, weight_source: WeightSource, device: torch.device | str
    ) -> "IntegerModel":
        """Reads the parameter file, and the quantization scales from KL.bin in the same
        directory."""
        return cls(Parameters.read(parameter_file), weight_source, device)

    def new_sequence(self) -> Sequence:
        """An empty sequence."""
        return Sequence(self.parameters, self.device)

    def free(self, seq: Sequence) -> None:
        """Releases the sequence's cache. The sequence cannot be used afterwards."""
        # extend() tests for None before it reads them.
        seq.keys = seq.values = None  # type: ignore[assignment]

    @overload
    def extend(
        self,
        seq: Sequence,
        token_ids: TokenIds,
        gap_for: None = None,
        record: Record | None = None,
        seed: int | None = None,
    ) -> torch.Tensor: ...

    @overload
    def extend(
        self,
        seq: Sequence,
        token_ids: TokenIds,
        gap_for: TokenIds,
        record: Record | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def extend(
        self,
        seq: Sequence,
        token_ids: TokenIds,
        gap_for: TokenIds | None = None,
        record: Record | None = None,
        seed: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Feeds the next n >= 1 tokens of seq and returns the model's greedy choice after each.

        Prefill, chunked prefill and decoding all use this call. The integers at a position depend
        only on the tokens up to it, not on how the tokens were divided into calls.

        The result is an int64 tensor of n token ids on the CPU: entry i is the argmax of the logits
        at the position of token_ids[i], with the lowest id on ties (`first_choice` in the code).

        gap_for (n token ids) scores given tokens, for example a reply served by another engine. The
        call then returns a pair: the greedy choices, and at each position the logit gap of
        gap_for[i] (gap_t of docs/model.md, "8 Embedding, head and output"): the largest z minus z
        at gap_for[i], plus 1 if the greedy choice has a lower id, where z is the logits, or with a
        seed the logits plus noise. It is 0 exactly for the greedy choice.

        seed is a request's 64-bit seed (not Parameters.seed, which generates synthetic weights).
        The choice and the gap are then taken from the logits plus the seed's public noise
        (noise.py): Gumbel-max sampling at temperature 1. Pass the same seed in every call for a
        sequence. Every position gets noise, the prompt's too, except the window's last, T - 1;
        which positions count is up to the caller. The noise is in the logits' unit, 2^-s, and s
        must be at most 40. A seed is for scoring alone: the prover proves no seeded request, and no
        line of the parameter file speaks of a request's seed or a temperature.

        record(name, layer, tensor) receives named intermediate values. The tensor's first axis is
        the n positions, and `layer` is None outside the layers. The names, in order: in each layer
        x, a, qw, q, kw, k, vw, v, o, x_mid, m, then lr, e_star, sc, xr in a mixture-of-experts
        layer, then gw, key, s, u; after the layers x_last, aF, lam, first_choice. Before each
        norm's result (a, m, aF) come its parts, for example a_sigma, a_Rc, a_w.

        Raises ValueError if token_ids or gap_for is not a one-dimensional, non-empty sequence of
        integers in 0 .. V - 1, if their lengths differ, if seq was freed, if a weight matrix has
        the wrong shape, or if a seed is given that is no 64-bit integer or s is above 40. Raises
        WindowFull, before anything is computed, if the sequence would exceed T tokens. Raises
        RangeConditionFailed if a range condition fails at one of these n positions. The sequence is
        then unusable: every later call with valid arguments raises the same exception.
        """
        p = self.parameters
        tokens = self._token_tensor(token_ids)
        n = tokens.shape[0]
        checked = self._token_tensor(gap_for) if gap_for is not None else None
        if checked is not None and checked.shape[0] != n:
            raise ValueError("gap_for must name one token for each fed token.")
        if seed is not None:
            check_seed(seed)
            if p.s > 40:
                raise ValueError(
                    f"With a seed, s must be at most 40. The parameter file has {p.s}."
                )
        if seq.keys is None:
            raise ValueError("This sequence was freed. Start a new one with new_sequence().")
        if seq.failure is not None:
            raise seq.failure
        if seq.length + n > p.T:
            raise WindowFull(
                f"A window holds {p.T} tokens. The sequence has {seq.length}, and this call would "
                f"add {n}."
            )
        note = record if record is not None else _record_nothing
        checks = _RangeChecks(n, self.device)

        x = self.weights.embedding_rows(tokens).to(torch.int64)  # x^0_t = Emb[tok_t]
        for layer in range(p.L):
            note("x", layer, x)
            x = self._attention(layer, x, seq, checks, note)
            note("x_mid", layer, x)
            x = self._feed_forward(layer, x, checks, note)
        note("x_last", None, x)
        aF = self._norm(x, p.MF, p.KF, p.epsF, checks, note, None, "aF")
        lam = self._product(aF, "Wh", p.V)  # the logits; exact by bound (a)
        z = lam if seed is None else self._add_noise(lam, seed, seq.length)
        first_choice = first_argmax(z)
        note("lam", None, lam)
        note("first_choice", None, first_choice)

        seq.length += n
        failure = checks.first_failure()
        if failure is not None:
            seq.failure = RangeConditionFailed(seq.length - n + failure[0], failure[1])
            raise seq.failure
        if checked is None:
            return first_choice.cpu()
        gap = z.amax(dim=1) - z.gather(1, checked[:, None])[:, 0] + (first_choice < checked)
        return first_choice.cpu(), gap.cpu()

    def _add_noise(self, lam: torch.Tensor, seed: int, start: int) -> torch.Tensor:
        """lam[i][v] + N[start + i][v] for the logits lam of positions start, start + 1, .., but lam
        itself at position T - 1.

        |lam| is at most bound (a), and N is below 2^46 at temperature 1 and s <= 40, so the sum
        fits 64 bits.
        """
        p = self.parameters
        z = lam.clone()
        with_noise = min(lam.shape[0], p.T - 1 - start)
        step = max(1, _NOISE_ELEMENTS // p.V)
        for first in range(0, with_noise, step):
            last = min(first + step, with_noise)
            positions = range(start + first, start + last)
            z[first:last] += noise(seed, positions, p.V, TEMPERATURE_ONE, p.s, self.device)
        return z

    def _token_tensor(self, token_ids: TokenIds) -> torch.Tensor:
        try:
            if isinstance(token_ids, (list, tuple)) and any(
                isinstance(token, bool) for token in token_ids
            ):
                raise TypeError  # among integers, PyTorch would read a boolean as 0 or 1
            tokens = torch.as_tensor(token_ids)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("Token ids must be integers.") from None
        if tokens.dim() != 1 or tokens.shape[0] == 0:
            raise ValueError("Token ids must be a one-dimensional sequence of at least one id.")
        if tokens.dtype.is_floating_point or tokens.dtype.is_complex or tokens.dtype == torch.bool:
            raise ValueError("Token ids must be integers.")
        # An unsigned id of 2^63 or more becomes negative, and the next test rejects it.
        tokens = tokens.to(torch.int64)
        if int(tokens.min()) < 0 or int(tokens.max()) >= self.parameters.V:
            raise ValueError(f"A token id is outside 0 .. {self.parameters.V - 1}.")
        return tokens.to(self.device)

    def _product(
        self,
        x: torch.Tensor,
        name: str,
        outputs: int,
        layer: int | None = None,
        expert: int | None = None,
    ) -> torch.Tensor:
        """The rows of x times one weight matrix. The weights' range is not tested here: the weight
        source must guarantee it (WeightSource)."""
        w = self.weights.matrix(name, layer, expert)
        if tuple(w.shape) != (outputs, x.shape[1]):
            where = f" of layer {layer}" if layer is not None else ""
            raise ValueError(
                f"The weight matrix {name}{where} has shape {tuple(w.shape)}, not "
                f"{(outputs, x.shape[1])}."
            )
        return exact_product(x, w)

    def _norm(
        self,
        x: torch.Tensor,
        M: int,
        K: int,
        eps: int,
        checks: _RangeChecks,
        note: Record,
        layer: int | None,
        name: str,
    ) -> torch.Tensor:
        """N(x; M, K, eps) of docs/model.md, "5 The norm", on each row of x: a root-mean-square norm
        without gain.

        Condition 2 is a guard here: no M and K that Parameters accepts let it fail while conditions
        8 and 1 hold. With five limbs (NX = 5) it could fail at K = M - 14 <= 23, which Parameters
        rejects (docs/model.md, "9 Range conditions"). The argument for all accepted pairs is in
        integrations/vllm/integer_model/tests/test_arithmetic_edges.py, which tests only a few of
        them, including the one that comes closest to failing (M = 38, K = 24).
        """
        p = self.parameters
        # Condition 8. With five limbs the interval is symmetric. With four it is all that four
        # balanced limbs express.
        x_max = 127 * (256**p.NX - 1) // 255
        x = checks.admit(8, x, -x_max if p.NX == 5 else -x_max - x_max // 127, x_max)
        # Condition 1: Vx + eps < 2^78 with five limbs, Vx < 2^59 with four
        # (docs/model.md, "10 Parameters that change the function"). The sum can exceed 64 bits, so
        # it is a wide number (high, low) = high * 2^40 + low; bound (g) keeps its parts within 64
        # bits. A failing row continues as a row of zeros.
        Vx = wide_sum_of_squares(x)
        Vx_eps = wide_add(Vx, eps)
        holds = Vx_eps[0] < 1 << (78 - 40) if p.NX == 5 else Vx[0] < 1 << (59 - 40)
        checks.require(1, holds)
        x = torch.where(holds[:, None], x, 0)
        sigma = wide_isqrt((torch.where(holds, Vx_eps[0], 0), torch.where(holds, Vx_eps[1], eps)))
        Rc = floor_quotient((1 << M) + (sigma >> 1), sigma)
        # |x_k| <= sigma, so |x_k Rc| <= 2^M + 2^38 < 2^48: the product fits 64 bits.
        w = checks.admit(2, rounded_quotient(x * Rc[:, None], K), *_TWO_LIMBS)
        nr = clamp(w, CLAMPED_MAX)
        note(name + "_sigma", layer, sigma)
        note(name + "_Rc", layer, Rc)
        note(name + "_w", layer, w)
        note(name, layer, nr)
        return nr

    # Sources: NOTICE, "torchao" (apply_rotary_emb has the view-rotate-stack form) and
    # "Hugging Face transformers" (neighbouring entries form a pair).
    def _rotary(self, A: torch.Tensor, layer: int, start: int) -> torch.Tensor:
        """The rotary step of docs/model.md, "6 Attention", on the unshifted products A, whose rows
        are positions start, start + 1, ..

        Entries 2 i and 2 i + 1 of each head are rotated by the tables' entry (t, i). In a layer
        with rp = 0 (no rotary encoding) the step only multiplies by 2^cs, which the shifts kq and
        kk remove. |A| is at most bound (a), so the two results of each pair stay below 2^63 by
        bound (f).
        """
        p = self.parameters
        n = A.shape[0]
        if p.layers[layer].rp:
            C = self.tables.cos[start : start + n, None, :]
            S = self.tables.sin[start : start + n, None, :]
        else:
            C, S = 1 << p.cs, 0
        pairs = A.view(n, -1, p.dh // 2, 2)
        A0, A1 = pairs[..., 0], pairs[..., 1]
        return torch.stack((C * A0 - S * A1, S * A0 + C * A1), dim=-1).view(n, -1)

    def _attention(
        self, layer: int, x: torch.Tensor, seq: Sequence, checks: _RangeChecks, note: Record
    ) -> torch.Tensor:
        """Maps the stream x that enters the layer, at the n new positions, to x' after
        attention."""
        p, k = self.parameters, self.parameters.layers[layer]
        n, start = x.shape[0], seq.length
        end = start + n
        a = self._norm(x, k.M1, k.K1, p.eps1, checks, note, layer, "a")

        # The three products are exact by bound (a). The range conditions are on the wide values qw,
        # kw and vw, before the clamp.
        def rotated(name: str, outputs: int, shift: int) -> torch.Tensor:
            A = self._product(a, name, outputs, layer)
            return checks.admit(
                3, rounded_quotient(self._rotary(A, layer, start), shift), *_TWO_LIMBS
            )

        qw, kw = rotated("Wq", p.dq, k.kq), rotated("Wk", p.dkv, k.kk)
        vw = checks.admit(
            4, rounded_quotient(self._product(a, "Wv", p.dkv, layer), k.kv), *_TWO_LIMBS
        )
        q, key, v = clamp(qw, CLAMPED_MAX), clamp(kw, CLAMPED_MAX), clamp(vw, CLAMPED_MAX)
        for name, tensor in (("qw", qw), ("q", q), ("kw", kw), ("k", key), ("vw", vw), ("v", v)):
            note(name, layer, tensor)
        seq.keys[layer, start:end] = key.to(torch.int16)
        seq.values[layer, start:end] = v.to(torch.int16)

        # Causal attention with a table softmax, one key-value head g at a time. Row i of the call
        # is position start + i and reads the cached positions j <= start + i.
        heads_per_group = p.n_h // p.n_g
        position = torch.arange(end, dtype=torch.int64, device=self.device)
        causal = position[None, :] <= position[start:, None]
        u0 = self.tables.softmax.shape[0] - 1
        q = q.view(n, p.n_g, heads_per_group * p.dh)
        o = torch.empty_like(q)
        for g in range(p.n_g):
            keys_g = seq.keys[layer, :end, g * p.dh : (g + 1) * p.dh]
            values_g = seq.values[layer, :end, g * p.dh : (g + 1) * p.dh]
            # Exact by bound (b).
            s = exact_product(q[:, g].reshape(n * heads_per_group, p.dh), keys_g)
            s = torch.where(causal[:, None, :], s.view(n, heads_per_group, end), _MASKED_SCORE)
            m = s.amax(dim=-1, keepdim=True)
            u = (m - s) >> k.sh
            # Entry u0 is 0; it is also used for the later positions.
            e = self.tables.softmax[torch.clamp(u, max=u0)]
            # Z >= EX[0] > 0 (EX: the softmax table), because u = 0 at the maximal score.
            Z = e.sum(dim=-1, keepdim=True)
            acc = exact_product(e.view(n * heads_per_group, end), values_g.T)  # exact by bound (c)
            acc = acc.view(n, heads_per_group, p.dh)
            o[:, g] = floor_quotient(2 * acc + Z, 2 * Z).view(n, heads_per_group * p.dh)
        o = o.view(n, p.dq)
        note("o", layer, o)

        # Output projection and residual. o is a rounded convex combination of values, so |o| <=
        # 2,047: bound (d).
        return x + rounded_quotient(self._product(o, "Wo", p.d, layer), k.ko)

    def _route(
        self, layer: int, m: torch.Tensor, checks: _RangeChecks, note: Record
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The router of docs/model.md, "7 Feed-forward": each row's chosen expert e_star and the
        routed expert's input xr."""
        p, k = self.parameters, self.parameters.layers[layer]
        # The wide logit; exact by bound (a).
        lw = rounded_quotient(self._product(m, "Wr", p.NE, layer), k.kr)
        if p.router_clamp:
            # lw = 8 ly + l3 with l3 in 0 .. 7. The clamp acts on ly only, so lr as a function of lw
            # is not monotone at the two ends.
            ly = checks.admit(5, lw >> 3, *_TWO_LIMBS)
            lr = 8 * clamp(ly, CLAMPED_MAX) + (lw & 7)
        else:
            lr = checks.admit(5, lw, -KEY_OFFSET_15, KEY_OFFSET_15 - 1)
        e_star = first_argmax(lr)  # taken after the clamp; ties go to the lowest index
        sc = self.tables.sigmoid[lr.gather(1, e_star[:, None])[:, 0] + KEY_OFFSET_15]
        # The scale multiplies the routed expert's input, not its output.
        xr = rounded_quotient(sc[:, None] * m, 16)
        for name, tensor in (("lr", lr), ("e_star", e_star), ("sc", sc), ("xr", xr)):
            note(name, layer, tensor)
        return e_star, xr

    def _feed_forward(
        self, layer: int, x: torch.Tensor, checks: _RangeChecks, note: Record
    ) -> torch.Tensor:
        """Maps the stream x' after attention to the stream x'' that enters the next layer
        (docs/model.md, "7 Feed-forward").

        A dense layer has f hidden units, which read m. A mixture-of-experts layer has the shared
        expert's fe hidden units, which read m, and then the chosen routed expert's fe, which read
        xr. Bound (e)'s message calls a hidden unit a slot.
        """
        p, k = self.parameters, self.parameters.layers[layer]
        n = x.shape[0]
        m = self._norm(x, k.M2, k.K2, p.eps2, checks, note, layer, "m")
        experts = p.has_experts(layer)
        shared = p.fe if experts else p.f  # the hidden units that every token uses

        # The wide gate key gw and the up projection u of every hidden unit. All products are exact
        # by bound (a), because |xr| <= |m| <= 2,047.
        gw = torch.empty((n, p.f), dtype=torch.int64, device=self.device)
        u = torch.empty_like(gw)
        gw[:, :shared] = rounded_quotient(self._product(m, "Wg", shared, layer), k.kg)
        u[:, :shared] = rounded_quotient(self._product(m, "Wu", shared, layer), k.ku)
        # Visit only the chosen experts; each one's weights are fetched when it is visited. Prior
        # art: NOTICE, "Hugging Face transformers" (the loop over experts in modeling_mixtral.py).
        rows_of_expert: list[tuple[int, torch.Tensor]] = []
        if experts:
            e_star, xr = self._route(layer, m, checks, note)
            rows_of_expert = [
                (e, torch.nonzero(e_star == e)[:, 0]) for e in torch.unique(e_star).tolist()
            ]
            for e, rows in rows_of_expert:
                gw[rows, shared:] = rounded_quotient(
                    self._product(xr[rows], "Wgr", p.fe, layer, e), k.kgR
                )
                u[rows, shared:] = rounded_quotient(
                    self._product(xr[rows], "Wur", p.fe, layer, e), k.kuR
                )

        if p.gate_clamp:
            # gw = 256 gh + g0 with the balanced low byte g0. The clamp of gh only saturates: beyond
            # it the SiLU table Ts is constant.
            g0 = balanced_low_byte(gw)
            gh = checks.admit(6, (gw - g0) >> 8, -KEY_OFFSET_15, KEY_OFFSET_15 - 1)
            table_key = 256 * clamp(gh, 127) + g0
        else:
            table_key = checks.admit(6, gw, *_TWO_LIMBS)
        s = self.tables.silu[table_key + KEY_OFFSET_16]
        u = checks.admit(7, u, *_TWO_LIMBS)
        for name, tensor in (("gw", gw), ("key", table_key), ("s", s), ("u", u)):
            note(name, layer, tensor)

        # Down projection and residual. The shared and the routed part are added before the one
        # shift kd. The sum over all f hidden units is exact by bound (e), and so is each part.
        su = s * u
        ad = self._product(su[:, :shared], "Wd", p.d, layer)
        for e, rows in rows_of_expert:
            ad[rows] += self._product(su[rows, shared:], "Wdr", p.d, layer, e)
        return x + rounded_quotient(ad, k.kd)


def _record_nothing(name: str, layer: int | None, tensor: torch.Tensor) -> None:
    pass


def _assert_arithmetic_is_exact(p: Parameters) -> None:
    """Raises ParameterError unless the seven bounds (a) to (g) below hold for p.

    The bounds follow from the clamps, the range conditions and the range of a weight, -4,096 ..
    4,095, which the weight source must guarantee. Bounds (a) to (e) bound sums of absolute products
    in exact_product(), which is exact when they are below 2^53 and the library meets the
    requirement that its docstring states. (f) and (g) keep the rotary step and the parts of a
    norm's sum of squares within 64 bits. At full size (f = 16,384 and smax = 4,095), bound (e), the
    down projection, is 2^52.9996: almost no room to spare.
    """
    weight = -WEIGHT_MIN  # the largest magnitude of a weight
    normed_row_product = CLAMPED_MAX * weight * p.d
    softmax_sum = (1 << p.expS) * CLAMPED_MAX * p.T
    down_projection = p.smax * (1 << 15) * weight * p.f
    rotary_sum = (2 << p.cs) * normed_row_product
    bounds = [
        ("a", "a normed row or the router-scaled row xr times a weight matrix", normed_row_product),
        ("b", "a query times a key", CLAMPED_MAX * CLAMPED_MAX * p.dh),
        ("c", "T values, each times a softmax table entry of at most 2^S", softmax_sum),
        ("d", "attention's output times Wo", CLAMPED_MAX * weight * p.dq),
        (
            "e",
            "the down projection, with |s| <= smax and |u| <= 2^15 in each of f slots",
            down_projection,
        ),
        (
            "f",
            "two terms, each a rotary table entry of at most 2^cs times a sum of (a)",
            rotary_sum,
        ),
        ("g", "the width d, which is the number of terms in a norm's sum of squares", p.d),
    ]
    limits = dict.fromkeys("abcde", FLOAT64_EXACT) | {"f": 1 << 63, "g": 1 << 20}
    for letter, description, bound in bounds:
        if bound >= limits[letter]:
            raise ParameterError(
                f"This package cannot evaluate a model of this shape exactly: bound ({letter}), "
                f"{description}, is {bound} and must be below {limits[letter]}. The bound follows "
                "from the parameter file's shape and table parameters."
            )


# The settings of BlockWeights that integer_model.json can set, with their types.
_BLOCK_SETTINGS = {
    "blocks": str,
    "setup_folder": str,
    "shards": list,
    "hash_integers": bool,
    "verify_sums": bool,
    "pin_memory": bool,
    "check_free_memory": bool,
}
_JSON_NAMES = {str: "string", list: "list", bool: "boolean"}


def load_from_folder(
    folder: str | Path, device: torch.device | str, options: dict | None = None
) -> IntegerModel:
    """Loads the model that a directory describes (OPERATING.md, "The files of a model").

    Its `integer_model.json` is, for example, {"weights": "blocks", "weight_settings": {"blocks":
    "synthetic"}}. "weights" selects the weight source: "synthetic" (every weight is a function of
    the parameter file's seed) or "blocks" (quantized blocks, from GGUF files or synthetic).
    With "weights": "blocks", "weight_settings" holds arguments of BlockWeights of the same names
    (those in _BLOCK_SETTINGS): always "blocks", and "setup_folder" and "shards" if and only if
    "blocks" is "gguf". With "weights": "synthetic" it must be empty or left out. Each top-level
    entry of options replaces the file's entry whole. No object in the file may repeat a name.

    To verify the files, pass options = {"sha256": {...}} with the SHA-256 of `integer_model.json`,
    the parameter file `policy.txt` and `KL.bin`, each under the file's name (see given_digests()),
    or options = {"sha256_file": path}, the absolute path of a file outside the model directory that
    holds those digests as JSON (see digests_from_file()), so that they appear on no command line.
    With weights in GGUF files, and only then, add those of `wsums.txt` and `fold.txt` in the
    set-up directory. A file with another digest raises ParameterError. The digests must come from
    outside the model directory, so `integer_model.json` must not have the entry. With "sha256" no
    other option is allowed, so that only the verified file decides the weights and their settings.

    The digests do not cover the weight files: the weights' integers are compared with wsums.txt
    only if "verify_sums" is true. A path in "setup_folder" or "shards" that leads out of the model
    directory is followed, with or without digests; without digests a warning names it.

    Raises ParameterError, WeightsRejected or NotEnoughMemory, and OSError for a file that it cannot
    read. A parameter file that this package cannot evaluate exactly, or whose table digests do not
    match the tables computed here, is rejected before any weight file is opened.
    """
    folder = Path(folder)
    file = folder / "integer_model.json"
    from_options = "The options (with vLLM, --model-loader-extra-config)"

    def require(holds: bool, where: object, fault: str) -> None:
        if not holds:
            raise ParameterError(f"{where}: {fault}")

    require(isinstance(options or {}, dict), from_options, "expected a JSON object.")
    options = dict(options or {})
    if "sha256_file" in options:
        require("sha256" not in options, from_options, 'give "sha256" or "sha256_file", not both.')
        options["sha256"] = digests_from_file(options.pop("sha256_file"), folder)
    sha256: dict[str, str] | None = options.pop("sha256", None)
    # Raises ParameterError if sha256 is malformed.
    given_digests(sha256, ("integer_model.json", "policy.txt", "KL.bin"), ("wsums.txt", "fold.txt"))
    require(
        sha256 is None or not options,
        from_options,
        'with "sha256", no other option is allowed (found '
        + ", ".join(f'"{name}"' for name in options)
        + "), so that only the "
        "verified file decides the weights. Remove the others, or set them in your reference "
        "copy of integer_model.json, copy that into the model directory, and pass its new digest.",
    )
    data = read_bytes(file, sha256["integer_model.json"] if sha256 else None)

    def each_entry_once(pairs: list[tuple[str, object]]) -> dict:
        # On its own, json.loads() silently keeps the last of an object's entries of the same name.
        entries = {}
        for name, value in pairs:
            require(
                name not in entries,
                file,
                f"the entry {json.dumps(name)} is written more than once in the same JSON object. "
                "Keep the one that you mean and remove the others.",
            )
            entries[name] = value
        return entries

    try:
        settings = json.loads(data.decode(), object_pairs_hook=each_entry_once)
    except ParameterError:  # from each_entry_once(); it is a ValueError too
        raise
    except ValueError as error:  # not JSON, or not text
        raise ParameterError(f"{file} is not a JSON file: {error}.") from None
    require(isinstance(settings, dict), file, "expected a JSON object.")
    require(
        "sha256" not in settings,
        file,
        '"sha256" is not allowed here. Pass it as an option, from outside the model directory.',
    )
    settings |= options
    merged = f"{file}, with the options that replace its entries" if options else file
    # json.dumps(), here and below, so that an unknown name cannot start a new line in the message.
    for name in settings:
        require(
            name in ("weights", "weight_settings"),
            merged,
            f'unknown entry {json.dumps(name)}. The entries are "weights" and "weight_settings", '
            'and "sha256" among the options.',
        )
    require(
        settings.get("weights") in ("synthetic", "blocks"),
        merged,
        '"weights" must be "synthetic" or "blocks".',
    )
    weight_settings = settings.get("weight_settings", {})
    require(isinstance(weight_settings, dict), merged, '"weight_settings" must be an object.')
    kinds = _BLOCK_SETTINGS if settings["weights"] == "blocks" else {}
    for name, value in weight_settings.items():
        require(
            name in kinds,
            merged,
            f'"weight_settings" has no entry {json.dumps(name)} for these weights.',
        )
        require(
            isinstance(value, kinds[name]),
            merged,
            f'the setting "{name}" must be a {_JSON_NAMES[kinds[name]]}.',
        )
    shards = weight_settings.get("shards", [])
    require(
        all(isinstance(shard, str) for shard in shards),
        merged,
        'every entry of "shards" must be a string.',
    )
    blocks = weight_settings.get("blocks")
    gguf = blocks == "gguf"
    whole_entry = (
        ' The option "weight_settings" (with vLLM, an entry of --model-loader-extra-config) '
        "replaces the whole entry in the file: give every setting in the option, or leave the "
        "option out and edit the file."
        if "weight_settings" in options
        else ""
    )
    require(
        gguf or blocks == "synthetic" or settings["weights"] == "synthetic",
        merged,
        (
            '"weight_settings" needs an entry "blocks": '
            if blocks is None
            else f'in "weight_settings", "blocks" is {json.dumps(blocks)} but must be '
        )
        + '"gguf" (weights from GGUF files) or "synthetic" (weights computed from the seed in '
        "policy.txt)." + (whole_entry if blocks is None else ""),
    )
    require(
        gguf or not weight_settings.get("verify_sums", False),
        merged,
        '"verify_sums": true needs "blocks": "gguf".',
    )
    require(
        not gguf or ("setup_folder" in weight_settings and bool(shards)),
        merged,
        '"blocks": "gguf" needs a "setup_folder" and at least one entry in "shards".' + whole_entry,
    )
    require(
        gguf or not ("setup_folder" in weight_settings or "shards" in weight_settings),
        merged,
        '"blocks": "synthetic" does not go with "setup_folder" or "shards": no GGUF file would be '
        'read. To load GGUF files, set "blocks": "gguf". For synthetic weights, remove '
        '"setup_folder" and "shards".',
    )
    require(
        sha256 is None or ("wsums.txt" in sha256) == gguf,
        from_options,
        'the weights are in GGUF files, so "sha256" must also have "wsums.txt" and "fold.txt" '
        "(set-up directory)."
        if gguf
        else 'the weights are not in GGUF files, so "sha256" must not have "wsums.txt" and '
        '"fold.txt".',
    )

    parameters = Parameters.read(
        folder / "policy.txt",
        sha256={name: sha256[name] for name in ("policy.txt", "KL.bin")} if sha256 else None,
    )
    # IntegerModel repeats these two checks. They run here first because they take a fraction of a
    # second, and the weights, read next, are hundreds of gigabytes at full size.
    _assert_arithmetic_is_exact(parameters)
    make_tables(parameters, torch.device("cpu"))
    started = time.monotonic()
    logger.info("Loading the weights (%s) of the model in %s.", settings["weights"], folder)
    if settings["weights"] == "synthetic":
        from .synthetic_weights import SyntheticWeights

        weights: WeightSource = SyntheticWeights(parameters, device)
    else:
        from .block_weights import BlockWeights

        if sha256 and gguf:
            weight_settings = weight_settings | {
                "sha256": {name: sha256[name] for name in ("wsums.txt", "fold.txt")}
            }
        elif gguf:
            top = folder.resolve()
            outside = [
                path
                for path in (weight_settings["setup_folder"], *shards)
                if not (folder / path).resolve().is_relative_to(top)
            ]
            if outside:
                logger.warning(
                    "No SHA-256 digests were given, and these weight paths lead out of the model "
                    "directory %s: %s. That is fine if you chose them. To pin them, pass the "
                    'digests as "sha256" or "sha256_file". With vLLM, in '
                    '--model-loader-extra-config, use "reference_directory" or "sha256_file": '
                    '"sha256" there shows the digests in the process list and the log. To check '
                    'the weights in the files too, set "verify_sums": true.',
                    folder,
                    # repr, so that a path cannot start a new log line.
                    ", ".join(map(repr, outside)),
                )
        weights = BlockWeights(parameters, device, folder=folder, **weight_settings)
    model = IntegerModel(parameters, weights, device)
    logger.info("The model is loaded (%.0f s).", time.monotonic() - started)
    return model
