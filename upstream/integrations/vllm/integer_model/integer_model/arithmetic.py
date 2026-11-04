# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Exact integer arithmetic on PyTorch tensors.

Values and results are int64 tensors, except that exact_product's w may have any integer dtype.
Every function runs on its arguments' device and computes exactly the integer function that its
docstring defines (notation: docs/model.md, "1 Notation"), within the bounds that it states. No
function checks its bounds: the caller must meet them and also keep intermediate sums within int64,
such as a + 2^(k-1) in rounded_quotient. exact_product and wide_isqrt go through float64, and
their docstrings state what they need of the numerical library.
"""

import torch

FLOAT64_EXACT = 1 << 53  # float64 represents every integer of absolute value up to 2^53


def rounded_quotient(a: torch.Tensor, k: int) -> torch.Tensor:
    """rq(a, k) = floor((a + 2^(k-1)) / 2^k) for k >= 1: division by 2^k, rounded to nearest, ties
    up.

    On int64, >> is an arithmetic shift, so it floors negative numbers too
    (https://docs.pytorch.org/docs/2.13/generated/torch.bitwise_right_shift.html).
    """
    return (a + (1 << (k - 1))) >> k


def floor_quotient(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """floor(a / b) for b > 0, also for a negative a. (rounding_mode="trunc" would round towards
    zero: https://docs.pytorch.org/docs/2.13/generated/torch.div.html.)
    """
    return torch.div(a, b, rounding_mode="floor")


def clamp(a: torch.Tensor, bound: int) -> torch.Tensor:
    """clamp(a; bound) = min(max(a, -bound), bound)."""
    return torch.clamp(a, -bound, bound)


def balanced_low_byte(a: torch.Tensor) -> torch.Tensor:
    """low8(a) = ((a + 128) mod 256) - 128, which is in -128 .. 127."""
    return ((a + 128) & 255) - 128


def first_argmax(a: torch.Tensor) -> torch.Tensor:
    """The least index at which the last axis of a attains its maximum.

    The rule for ties is part of the model, so it is written out here and not left to a library's
    argmax.
    """
    size = a.shape[-1]
    index = torch.arange(size, dtype=torch.int64, device=a.device)
    return torch.where(a == a.amax(dim=-1, keepdim=True), index, size).amin(dim=-1)


# A slice of w in exact_product: at most this many elements (268 MB in float64), or one row.
_SLICE_ELEMENTS = 1 << 25


def exact_product(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T, exactly: out[i][j] = sum over k of x[i][k] * w[j][k], for x of shape (n, inputs) and
    an integer w of shape (outputs, inputs).

    The product runs in float64, because PyTorch has no int64 matrix product on CUDA
    (https://github.com/pytorch/pytorch/issues/44428). It is exact if, for every i and j, the sum
    over k of |x[i][k] * w[j][k]| is below 2^53. Then every product and every partial sum, in any
    order and with or without fused multiply-add, is an integer that float64 represents.

    Exactness also requires the library to form each entry as an ordinary inner product in true
    float64: not by a Strassen-like algorithm, which multiplies sums of entries (arXiv:cs/0601133,
    section 3.2.3), nor by cuBLAS's emulated FP64 (the Ozaki scheme). Nothing here can check this.
    To check the serving GPU: in integrations/vllm/integer_model/,
    OPERATING.md, "Check that float64 products are exact on your GPU".

    w is converted one slice of output rows at a time, so no float64 copy of a large w exists.

    Prior work: Dumas, Giorgi and Pernet, FFLAS and FFPACK (arXiv:cs/0601133); Slalom
    (arXiv:1806.03287); CryptGPU (arXiv:2104.10949); CrypTen (arXiv:2109.00984).
    PUBLISHED-WORKS.md, "Published methods of the integer model in vLLM", lists more.
    """
    x64 = x.to(torch.float64)
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.int64, device=x.device)
    rows = max(1, _SLICE_ELEMENTS // w.shape[1])
    for start in range(0, w.shape[0], rows):
        out[:, start : start + rows] = (x64 @ w[start : start + rows].to(torch.float64).T).to(
            torch.int64
        )
    return out


# Wide numbers. A norm's sum of squares can exceed 64 bits. A wide number is a pair (high, low) of
# int64 tensors that represents the non-negative integer high * 2^40 + low, with 0 <= low < 2^40 and
# 0 <= high < 2^62.
Wide = tuple[torch.Tensor, torch.Tensor]
_LOW_MASK = (1 << 40) - 1


def _wide(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Wide:
    """The wide number a * 2^40 + b * 2^21 + c, for |a|, |b| < 2^61 and 0 <= c < 2^62, if it is not
    negative."""
    # b = (b >> 19) * 2^19 + (b & (2^19 - 1)), also for a negative b.
    low = c + ((b & ((1 << 19) - 1)) << 21)
    return a + (b >> 19) + (low >> 40), low & _LOW_MASK


def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x = high * 2^20 + low with 0 <= low < 2^20. Then x^2 = high^2 * 2^40 + high * low * 2^21 +
    low^2."""
    return x >> 20, x & ((1 << 20) - 1)


def wide_sum_of_squares(x: torch.Tensor) -> Wide:
    """The sum of x^2 over the last axis, for |x| < 2^39 and at most 2^20 terms.

    With the two parts of _split, |high| <= 2^19 and low < 2^20. So the three sums are at most 2^58,
    2^59 and 2^60.
    """
    high, low = _split(x)
    return _wide((high * high).sum(dim=-1), (high * low).sum(dim=-1), (low * low).sum(dim=-1))


def wide_add(v: Wide, e: int) -> Wide:
    """v + e for 0 <= e < 2^62."""
    low = v[1] + e
    return v[0] + (low >> 40), low & _LOW_MASK


def _wide_less(v: Wide, w: Wide) -> torch.Tensor:
    return (v[0] < w[0]) | ((v[0] == w[0]) & (v[1] < w[1]))


def _wide_square(r: torch.Tensor) -> Wide:
    """r^2 for 0 <= r <= 2^39."""
    high, low = _split(r)
    return _wide(high * high, high * low, low * low)


def wide_isqrt(v: Wide) -> torch.Tensor:
    """isqrt(v), the largest integer r with r^2 <= v, for v < 2^78.

    The integer part of a float64 square root is an estimate, and two exact comparisons correct it.
    This requires torch.sqrt to be accurate to a few units in the last place. Then the estimate is
    off by at most 1: converting v to float64 changes it by a factor within 1 +- 2^-53, which moves
    its root, a number below 2^39, by less than 2^-14, and torch.sqrt adds an error of the same
    order, so the computed root is within 1/2 of the real one.
    """
    r = torch.sqrt(v[0].to(torch.float64) * float(1 << 40) + v[1].to(torch.float64)).to(torch.int64)
    r = r - _wide_less(v, _wide_square(r)).to(torch.int64)  # one too large: r^2 > v
    # One too small: (r + 1)^2 <= v.
    return r + (~_wide_less(v, _wide_square(r + 1))).to(torch.int64)
