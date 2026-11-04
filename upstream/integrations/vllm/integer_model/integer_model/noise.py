# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The public noise of a seeded request: integers computed from seed, position and token id.

A server can sample at a temperature by Gumbel-max sampling: it adds noise to every logit and takes
the largest sum. If the noise is a public function of a seed that the request specifies, the served
token can be scored: an honest one is (nearly) the argmax of the integer model's logits plus the
same noise (DiFR, Karvonen et al., arXiv:2511.20621, sections 3.2 and 4.2). As there, the
temperature multiplies the noise and not the logits. The values come from a table of integers near
vLLM's float32 noise (see noise()), so a served token is scored against this noise, not vLLM's own
draw. The noise also scales with the temperature temp and with s, the exponent of the logits' unit.

    from integer_model.noise import TEMPERATURE_ONE, noise

    nz = noise(seed=42, positions=range(7, 9), vocabulary_size=202_048, temp=TEMPERATURE_ONE, s=15)

noise() states the rule in four steps, as does docs/model.md, "12 The noise of a seeded score";
slots, table and _scaled_table compute steps 2 to 4. IntegerModel.extend(seed=...) adds the noise at
temperature 1. It is for scoring alone: the prover and the verifier take no request's seed.
"""

import functools
import hashlib
import struct
from collections.abc import Sequence

import torch

TEMPERATURE_ONE = 256  # temp is in units of 1/256
_TABLE_SHA256 = "090e3da3a772ee5a5427e0fc4c78d2179e98d3e8bfc40d255245fb9ab2340e85"
_MASK_32 = (1 << 32) - 1
_FRACTION_BITS = 48  # the noise table G (see table) and the logarithms are in units of 2^-48
_LN2 = 195_103_586_505_167  # round(ln 2 * 2^48)
# scale in _g, in units of 2^-55: Triton multiplies x (see slots) by the float32 4.6566127342e-10,
# which is this times 2^-55. vLLM's gumbel.py has that number as the lower bound of u only.
_FLOAT32_SCALE = (1 << 24) - 1


def philox4x32_10(
    k0: torch.Tensor | int,
    k1: torch.Tensor | int,
    c0: torch.Tensor,
    c1: torch.Tensor | int = 0,
    c2: torch.Tensor | int = 0,
    c3: torch.Tensor | int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four words of Philox4x32 with ten rounds for the key (k0, k1) and counter (c0 .. c3).

    Every argument is a 32-bit value, or an int64 tensor of such values; c0 must be a tensor, and
    tensors k0 and k1 must be on c0's device. The tensors broadcast. Each result is a new int64
    tensor of the broadcast shape with values in 0 .. 2^32 - 1.

    Defined in J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel Random Numbers: As
    Easy as 1, 2, 3", SC '11 (2011), section 4.3.
    """
    arguments = (k0, k1, c0, c1, c2, c3)
    if any(isinstance(a, torch.Tensor) and a.dtype != torch.int64 for a in arguments):
        raise TypeError("philox4x32_10 takes int64 tensors.")
    shape = torch.broadcast_shapes(*(a.shape for a in arguments if isinstance(a, torch.Tensor)))
    # The four words are two pairs (R, L).
    r0, l0, r1, l1 = (
        torch.as_tensor(c, dtype=torch.int64, device=c0.device).expand(shape)
        for c in (c0, c1, c2, c3)
    )
    for _ in range(10):
        # A round exchanges the R of the two pairs and then maps each pair, with its multiplier M
        # and its key word k, to R' = mulhi(R, M) ^ k ^ L and L' = mullo(R, M). An int64 product
        # keeps the low 64 bits, so it can be negative, and >> then fills in ones: hence the mask.
        first, second = r1 * 0xCD9E8D57, r0 * 0xD2511F53
        r0, l0 = ((first >> 32) & _MASK_32) ^ k0 ^ l0, first & _MASK_32
        r1, l1 = ((second >> 32) & _MASK_32) ^ k1 ^ l1, second & _MASK_32
        # The next round's key. The steps are the upper 32 bits of the paper's 64-bit constants:
        # floor(2^32 f) for f the fractional part of the golden ratio and of the square root of 3.
        k0, k1 = (k0 + 0x9E3779B9) & _MASK_32, (k1 + 0xBB67AE85) & _MASK_32
    return r0, l0, r1, l1


def slots(words: torch.Tensor) -> torch.Tensor:
    """Step 2 of the rule: the 16-bit table slot of each 32-bit word W.

    - x = W, or W with all bits inverted if W >= 2^31 (as in Triton's random.py; see table);
    - side = 1 if x >= 2^30, else 0;
    - k = 2^31 - 1 - x if side is 1, else x;
    - e = the bit length of k;
    - m = the ten bits below k's leading bit (missing bits are 0).
    The slot is 2^15 side + 2^10 e + m.
    """
    if words.dtype != torch.int64:
        raise TypeError("slots takes an int64 tensor.")
    x = torch.where(words >= 1 << 31, words ^ _MASK_32, words)
    side = x >> 30
    k = torch.where(side == 1, (1 << 31) - 1 - x, x)
    powers_of_two = 1 << torch.arange(31, dtype=torch.int64, device=words.device)
    # How many powers of two are at most k: k's bit length.
    e = torch.bucketize(k, powers_of_two, right=True)
    m = ((k >> (e - 11).clamp(min=0)) << (11 - e).clamp(min=0)) & 1023
    return (side << 15) | (e << 10) | m


def _log2(x: int) -> int:
    """The binary logarithm of an integer x >= 1 in units of 2^-48, rounded down at every step."""
    n = x.bit_length() - 1
    y, fraction = x << (63 - n), 0  # y / 2^63 is in [1, 2)
    for _ in range(_FRACTION_BITS):
        square = y * y  # square / 2^126 is in [1, 4)
        two = square >> 127
        fraction = 2 * fraction + two
        y = square >> (64 if two else 63)
    return (n << _FRACTION_BITS) | fraction


def _times_ln2(a: int) -> int:
    return (a * _LN2) >> _FRACTION_BITS


def _round_to_24_bits(v: int) -> int:
    """v rounded to 24 significant bits, ties to even: the integer that a float32 holds for v."""
    extra = v.bit_length() - 24
    if extra <= 0:
        return v
    quotient, rest, half = v >> extra, v & ((1 << extra) - 1), 1 << (extra - 1)
    if rest > half or (rest == half and quotient & 1):
        quotient += 1
    return quotient << extra


def _g(x: int) -> int:
    """-ln(-ln(1 - u)) + 5 ln 2 in units of 2^-48, for u = max(float32(x) * scale, scale).

    The constant 5 ln 2 keeps the result positive. It is the same for every token, so it cancels in
    a difference of two entries of G, and in a difference of two noise values up to one unit of
    2^-s (step 4 rounds down).
    """
    u = max(_round_to_24_bits(_round_to_24_bits(x) * _FLOAT32_SCALE), _FLOAT32_SCALE)  # u * 2^55
    if u < 1 << 39:  # u < 2^-16: -ln(1 - u) is taken as u
        return _times_ln2((60 << _FRACTION_BITS) - _log2(u))
    # -ln(1 - u) in units of 2^-48.
    minus_ln = _times_ln2((55 << _FRACTION_BITS) - _log2((1 << 55) - u))
    return _times_ln2((53 << _FRACTION_BITS) - _log2(minus_ln))


@functools.cache
def table() -> tuple[int, ...]:
    """Step 3 of the rule: G, 65,536 integers.

    Steps 1 and 3, and the x of step 2, aim to match the noise that the V2 sampler of vLLM 0.29.0
    generates for a request's seed, so that an unmodified server can serve what is scored here.
    Their sources are vllm/v1/worker/gpu/sample/gumbel.py (Apache-2.0) and triton/language/random.py
    of Triton 3.7.1 (MIT). There u is a float32 and the noise is -log(-log1p(-u)) in float32. Here u
    is the same number, computed in integers, and the two logarithms are one lookup.

    G[slot] is floor((_g(a) + _g(b)) / 2), where a and b are the smallest and largest x in the slot.
    A slot that no x reaches is 0. Raises RuntimeError if the SHA-256 of the entries, as
    little-endian 64-bit words, is not _TABLE_SHA256.
    """
    entries = [0] * 65536
    for e in range(31):
        for m in range(1024):
            if e <= 11 and (m if e == 0 else m & ((1 << (11 - e)) - 1)):
                continue  # no k has this slot
            width = 1 << max(e - 11, 0)
            first = 0 if e == 0 else (1 << (e - 1)) + (m * width if e > 11 else m >> (11 - e))
            last = first + width - 1
            top = (1 << 31) - 1
            entries[e << 10 | m] = (_g(first) + _g(last)) >> 1
            entries[1 << 15 | e << 10 | m] = (_g(top - last) + _g(top - first)) >> 1
    if hashlib.sha256(struct.pack("<65536Q", *entries)).hexdigest() != _TABLE_SHA256:
        raise RuntimeError(
            "The noise table computed here does not have the expected SHA-256. This is a bug."
        )
    return tuple(entries)


@functools.cache
def _scaled_table(temp: int, s: int, device: torch.device) -> torch.Tensor:
    """Step 4 of the rule for every slot, as an int64 tensor on the device.

    The tensor is cached and shared: do not change it in place.
    """
    if not 0 <= s <= 40:
        raise ValueError(f"s is {s}, expected 0 .. 40.")
    if not 1 <= temp <= 1024:
        raise ValueError(
            f"temp is {temp}, expected 1 .. 1,024 (in units of 1/256: temperature 1 is "
            f"{TEMPERATURE_ONE})."
        )
    return torch.tensor(
        [(temp * entry) >> (56 - s) for entry in table()], dtype=torch.int64, device=device
    )


def largest_noise(temp: int, s: int) -> int:
    """The largest value that noise() can return at this temp and s: 817,669 at temperature 1 and s
    = 15."""
    return int(_scaled_table(temp, s, torch.device("cpu")).max())


def check_seed(seed: int) -> None:
    """Raises ValueError unless seed is an int (not a bool) in -2^63 .. 2^64 - 1."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not -(1 << 63) <= seed < 1 << 64:
        raise ValueError("The seed must be an integer that fits 64 bits.")


def noise(
    seed: int,
    positions: Sequence[int] | torch.Tensor,
    vocabulary_size: int,
    temp: int,
    s: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """nz[t][v] for the given positions t and every token id v < vocabulary_size, as int64.

    1. K is the first word of Philox4x32-10 with the key (low half, high half of the seed) and the
       counter (t, 0, 0, 0). W is the first word with the key (K, 0) and the counter (v, 0, 0, 0). t
       is the position whose logits choose token t + 1.
    2. W selects one of 65,536 slots (slots).
    3. G[slot] is -ln(-ln(1 - u)) + 5 ln 2 in units of 2^-48, for u about x / 2^31 (exactly: _g),
       averaged over the slot's two ends (table).
    4. nz[t][v] = floor(temp G[slot] / 2^(56 - s)). It is never negative.

    A negative seed is taken modulo 2^64, so -1 is 2^64 - 1. temp is the temperature in units of
    1/256, from 1 to 1,024 (IntegerModel.extend passes 256). 2^-s is the unit of the integer logits
    (`s` in the parameter file). The result has the shape (len(positions), vocabulary_size) at 8
    bytes an entry, and the computation needs several times that memory: at a vocabulary of
    202,048, pass a few positions at a time.

    At temperature 1 the table is within 0.002 of vLLM's float32 noise. That difference, and the
    server's floating-point logits, can leave a served token short of the largest sum of logit and
    noise: its gap (docs/model.md, "8 Embedding, head and output") is then positive. In one run on
    one H100 (vLLM 0.29.0, PyTorch 2.13.0, Triton 3.7.1; a 2-layer model with random weights;
    temperature 1), this table's noise added to vLLM's logits picked the served token at all 1,984
    positions. What else was checked, and what was not: OPERATING.md, "What has been run".
    """
    check_seed(seed)
    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be at least 1.")
    device = torch.device(device)
    seed %= 1 << 64
    try:
        t = torch.as_tensor(positions)
        if t.dim() != 1 or (
            t.numel() and (t.dtype.is_floating_point or t.dtype.is_complex or t.dtype == torch.bool)
        ):
            raise ValueError
        t = t.to(device, torch.int64)
        if t.numel() and not 0 <= int(t.min()) <= int(t.max()) <= _MASK_32:
            raise ValueError
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(
            "positions must be a one-dimensional sequence of integers in 0 .. 2^32 - 1."
        ) from None
    keys = philox4x32_10(seed & _MASK_32, seed >> 32, t)[0]
    token_ids = torch.arange(vocabulary_size, dtype=torch.int64, device=device)
    words = philox4x32_10(keys[:, None], 0, token_ids[None, :])[0]
    return _scaled_table(temp, s, device)[slots(words)]
