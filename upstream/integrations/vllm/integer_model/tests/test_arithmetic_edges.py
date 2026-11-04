# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Edge cases of the integer arithmetic that random windows of tokens rarely reach, and what loading
a model must reject.

Part 1 calls the functions of `integer_model.arithmetic` at exact ties of a rounding shift, at the
ends of a clamp and at the largest sums. Part 2 feeds chosen rows to a norm. Part 3 runs whole
models with chosen weights (module `chosen_model`). Part 4 checks that loading rejects a shape for
which the arithmetic would not be exact, a public table whose digest differs from the parameter
file's, and weights outside their range.

Many expected values are those that a second implementation computed (module `recorded_values`,
which states their limits). The others are computed here in exact integer arithmetic.

Symbols are those of docs/model.md. A window is the model's input of exactly T tokens. Limbs
are the signed bytes in which the prover writes a value (docs/model.md, "9 Range conditions").
"Full size" is the shape of the example configuration prover/examples/target_full_f2:
d = 5,120, f = 16,384, dh = 128, V = 202,048 and T = 1,024.
"""

import math
import re

import numpy
import pytest
import torch

import recorded_values
from chosen_model import ALL_BASES, BASES, ChosenModel, scales_of, shape
from integer_model import (
    ParameterError,
    RangeConditionFailed,
    WeightsInMemory,
    WeightsRejected,
    arithmetic,
)
from integer_model.model import _RangeChecks
from integer_model.parameters import Parameters


def tensor(values, device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


# Part 1: the functions of arithmetic.py


def test_rounding_shift_on_ties_and_negative_numbers(device):
    """rq(a, k) at exact ties, next to them, and at the largest sums, for both signs and every
    allowed shift k.
    """
    for_each_shift = {}
    for k in range(1, 33):
        values = set()
        for multiple in (-3, -2, -1, 0, 1, 2, 3, -(1 << 20), 1 << 20):
            tie = multiple * (1 << k) + (1 << (k - 1))
            values |= {tie - 1, tie, tie + 1, multiple * (1 << k)}
        # A full-size down-projection sum (9.003e15), 2^53 - 1, and a 62-bit number.
        for large in (4095 * 32768 * 4095 * 16384, (1 << 53) - 1, (1 << 62) - 1):
            values |= {
                large,
                -large,
                large - (large % (1 << k)) + (1 << (k - 1)),
                -large + (large % (1 << k)) - (1 << (k - 1)),
            }
        for_each_shift[k] = sorted(values)
    a = [value for k in for_each_shift for value in for_each_shift[k]]
    shifts = [k for k in for_each_shift for _ in for_each_shift[k]]
    recorded = recorded_values.functions({"rq_a": numpy.array(a), "rq_k": numpy.array(shifts)})
    found = [
        value
        for k in for_each_shift
        for value in arithmetic.rounded_quotient(tensor(for_each_shift[k], device), k).tolist()
    ]
    assert recorded.difference("rq", None, numpy.array(found)) is None
    assert found == [(value + (1 << (k - 1))) >> k for value, k in zip(a, shifts)]
    # Ties go up.
    assert arithmetic.rounded_quotient(tensor([-3, -1, 1, 3], device), 1).tolist() == [
        -1,
        0,
        1,
        2,
    ]


def test_clamp_at_its_ends(device):
    values = [-32768, -2049, -2048, -2047, -2046, -1, 0, 1, 2046, 2047, 2048, 2049, 32639, 32767]
    recorded = recorded_values.functions({"clamp12_in": numpy.array(values)})
    found = arithmetic.clamp(tensor(values, device), 2047).cpu().numpy()
    assert recorded.difference("clamp12", None, found) is None
    assert arithmetic.clamp(tensor([-129, -128, -127, 127, 128], device), 127).tolist() == [
        -127,
        -127,
        -127,
        127,
        127,
    ]


def test_floor_quotient_of_negative_numbers(device):
    """A C division truncates towards zero. The model's quotients are floors."""
    numerators = [-9, -8, -7, -1, 0, 1, 7, 8, 9, -(1 << 35) - 1, -(1 << 35), (1 << 35) + 1]
    for divisor in (1, 2, 4, 7, 8194, (1 << 23) + 1):
        found = arithmetic.floor_quotient(
            tensor(numerators, device), tensor([divisor] * len(numerators), device)
        )
        assert found.tolist() == [numerator // divisor for numerator in numerators]


def test_balanced_low_byte(device):
    """The low byte g0 of a gate key is in -128 .. 127, not in 0 .. 255. The rest of the key is a
    multiple of 256.
    """
    values = [
        *range(-700, 700),
        4194175,
        -4194432,
        (1 << 40) + 127,
        (1 << 40) + 128,
        -(1 << 40) - 128,
        -(1 << 40) - 129,
    ]
    for value, low in zip(values, arithmetic.balanced_low_byte(tensor(values, device)).tolist()):
        assert -128 <= low <= 127
        assert (value - low) % 256 == 0


def test_first_argmax_takes_the_lowest_index(device):
    rows = [
        [5, 5, 5, 5],
        [1, 7, 7, 3],
        [0, 1, 2, 2],
        [-4, -9, -4, -4],
        [-1, -1, -1, 0],
        [9, 0, 0, 9],
    ]
    assert arithmetic.first_argmax(tensor(rows, device)).tolist() == [0, 1, 2, 0, 3, 0]
    # A row as long as the full-size vocabulary, which a library may search in parallel parts. The
    # maximum occurs many times.
    long_row = torch.zeros((1, 202048), dtype=torch.int64, device=device)
    long_row[0, 777::1000] = 12345
    assert arithmetic.first_argmax(long_row).tolist() == [777]


def test_exact_square_root_around_perfect_squares(device):
    """isqrt(v) at n^2 - 1, n^2, n^2 + 1 and (n + 1)^2 - 1, up to 2^78 - 1, the largest allowed."""
    random = numpy.random.default_rng(1)
    roots = {1, 2, 3, (1 << 39) - 1, (1 << 39) - 2}
    for exponent in range(1, 39):
        roots |= {(1 << exponent) - 1, 1 << exponent, (1 << exponent) + 1}
        roots |= {int(root) for root in random.integers(1 << exponent, 2 << exponent, 20)}
    # Roots near 2^26.5, whose squares are near 2^53, where float64 stops holding every integer.
    roots |= set(range(94906260, 94906272))
    values = sorted(
        {v for n in roots for v in (n * n - 1, n * n, n * n + 1, n * n + 2 * n) if 1 <= v < 1 << 78}
    )
    high = tensor([v >> 40 for v in values], device)
    low = tensor([v & ((1 << 40) - 1) for v in values], device)
    found = arithmetic.wide_isqrt((high, low)).tolist()
    recorded = recorded_values.functions(
        {
            "isqrt_hi": numpy.array([v >> 32 for v in values]),
            "isqrt_lo": numpy.array([v & 0xFFFFFFFF for v in values]),
        }
    )
    assert recorded.difference("isqrt", None, numpy.array(found)) is None
    assert found == [math.isqrt(v) for v in values]


def test_sum_of_squares_beyond_64_bits(device):
    """wide_sum_of_squares and wide_add are exact even far above 2^78, range condition 1's bound."""
    random = numpy.random.default_rng(2)
    largest = (1 << 39) - 1
    rows = [
        # 2^90.3: the full-size width, every value as large as wide_sum_of_squares allows.
        [largest] * 5120,
        [-largest] * 5120,
        [largest, -largest] * 2560,
        [0] * 5120,
        [1] * 5120,
        [(1 << 20) - 1, 1 << 20, (1 << 20) + 1, -(1 << 20), -(1 << 20) - 1] * 1024,
        *([int(x) for x in random.integers(-largest, largest + 1, 5120)] for _ in range(4)),
    ]
    high, low = arithmetic.wide_sum_of_squares(tensor(rows, device))
    for row, h, lo in zip(rows, high.tolist(), low.tolist()):
        assert 0 <= lo < 1 << 40
        assert (h << 40) + lo == sum(x * x for x in row)
    for eps in (1, (1 << 40) - 1, (1 << 48) - 1):
        h2, l2 = arithmetic.wide_add((high, low), eps)
        for row, h, lo in zip(rows, h2.tolist(), l2.tolist()):
            assert 0 <= lo < 1 << 40
            assert (h << 40) + lo == sum(x * x for x in row) + eps


@pytest.mark.parametrize(
    "rows_per_slice", [None, 1, 4], ids=["one_slice", "slices_of_one_row", "slices_of_four_rows"]
)
def test_products_at_their_largest_sizes(rows_per_slice, device, monkeypatch):
    """Products whose sums of absolute values come as close to 2^53 as the full-size shape allows.

    exact_product converts a large weight matrix in slices of rows, so the test also runs with
    slices of one and of four rows, which cut the down projection's weights.
    """
    if rows_per_slice is not None:
        monkeypatch.setattr(arithmetic, "_SLICE_ELEMENTS", rows_per_slice * 16384)
    random = numpy.random.default_rng(3)
    f, d = 16384, 5120
    # The largest |s u| of one hidden unit: |s| <= smax = 4,095 and u >= -32,768.
    su = 4095 * 32768
    down_inputs = numpy.stack(
        [
            numpy.full(f, -su),
            numpy.full(f, su),
            numpy.where(numpy.arange(f) % 2 == 0, su, -su),
            numpy.full(f, 4095 * 32639),
            random.integers(-su, su + 1, f),
            numpy.where(random.integers(0, 2, f) == 0, su, su - 1),
        ]
    )
    down_weights = numpy.stack(
        [
            numpy.full(f, 4095),
            numpy.full(f, -4096),
            numpy.where(numpy.arange(f) % 2 == 0, 4095, -4096),
            numpy.where(numpy.arange(f) == 12345, 4094, 4095),
            random.integers(-4096, 4096, f),
            numpy.where(random.integers(0, 2, f) == 0, -4096, -4095),
        ]
    )
    normed_inputs = numpy.stack(
        [numpy.full(d, 2047), numpy.full(d, -2047), random.integers(-2047, 2048, d)]
    )
    normed_weights = numpy.stack(
        [numpy.full(d, -4096), numpy.full(d, 4095), random.integers(-4096, 4096, d)]
    )
    # Attention's two products at dh = 128 and T = 1,024: queries times keys, and table entries up
    # to 2^12 times values.
    queries, keys = (
        numpy.stack([numpy.full(128, 2047), numpy.full(128, -2047)]),
        numpy.full((1, 128), -2047),
    )
    entries, values = (
        numpy.full((1, 1024), 4096),
        numpy.stack([numpy.full(1024, 2047), numpy.full(1024, -2047)]),
    )
    pairs = (
        (down_inputs, down_weights),
        (normed_inputs, normed_weights),
        (queries, keys),
        (entries, values),
    )
    for x, w in pairs:
        assert (
            int((numpy.abs(x)[:, None, :] * numpy.abs(w)[None, :, :]).sum(axis=2).max()) < 1 << 53
        )
        expected = (x[:, None, :] * w[None, :, :]).sum(axis=2)
        found = arithmetic.exact_product(
            tensor(x, device), torch.tensor(w, dtype=torch.int16, device=device)
        )
        assert found.cpu().numpy().tolist() == expected.tolist()
    # This sum is 9.005e15, just below 2^53 = 9.007e15.
    assert int(numpy.abs(down_inputs[0] * down_weights[1]).sum()) == 4095 * 32768 * 4096 * 16384


# Rows, inputs and outputs of each product. The four large shapes run only on a GPU.
SERVED_SHAPES = {
    "small_enough_for_a_cpu": (8, 96, 40),
    "one_row_small_enough_for_a_cpu": (1, 96, 40),
    "down_projection_of_a_whole_window": (1024, 16384, 5120),
    "down_projection_of_a_decoding_step": (1, 16384, 5120),
    "head_at_64_positions": (64, 5120, 202048),
    "head_of_a_decoding_step": (1, 5120, 202048),
}


def is_large(name: str) -> bool:
    return math.prod(SERVED_SHAPES[name]) > 1 << 24


@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=pytest.mark.gpu if is_large(name) else ()) for name in SERVED_SHAPES],
)
def test_products_of_the_shapes_that_a_server_takes(name, device):
    """Products in the shapes that a server computes, with sums of the largest sizes.

    test_products_at_their_largest_sizes uses a few rows only. A library may use one method for
    large matrices and another for small ones, or one for a single row and another for many. So the
    full-size down projection and head each run whole, with many rows, as for a prompt, and with one
    row, as for a decoding step. Two small shapes also run without a GPU.
    """
    rows, inputs, outputs = SERVED_SHAPES[name]
    if is_large(name):
        if torch.device(device).type != "cuda":
            pytest.skip("runs only with INTEGER_MODEL_TEST_DEVICE=cuda")
        # At the largest shape, w takes 2.1 GB, the products of one step of the loop below up to
        # 2.1 GB, and the rest under 1 GB.
        if torch.cuda.mem_get_info(device)[0] < 8 << 30:
            pytest.skip("needs 8 GB of free GPU memory")
    generator = torch.Generator(device=device).manual_seed(5)

    def random(
        low: int, high: int, shape: tuple[int, int], dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=generator, device=device, dtype=dtype)

    # Every entry of x has the largest size or one less, with a random sign, so that a sum needs all
    # its bits.
    largest = 4095 * 32768 if inputs == 16384 else 2047
    x = (largest - random(0, 2, (rows, inputs))) * (2 * random(0, 2, (rows, inputs)) - 1)
    w = random(-4096, 4096, (outputs, inputs), torch.int16)
    # One row of x and one of w give the largest sum that the model's bounds allow. Where x has only
    # one row, the product is taken twice: with the random row, and with that row.
    w[0] = -4096
    if rows == 1:
        operands = [x, torch.full_like(x, -largest)]
    else:
        x[0] = -largest
        operands = [x]
    assert inputs * largest * 4096 < 1 << 53
    outputs_at_a_time = max(1, (1 << 28) // inputs)
    for x in operands:
        found = arithmetic.exact_product(x, w)
        for row in range(rows):
            for start in range(0, outputs, outputs_at_a_time):
                expected = (w[start : start + outputs_at_a_time] * x[row]).sum(dim=1)
                assert expected.dtype == torch.int64
                assert torch.equal(found[row, start : start + outputs_at_a_time], expected), (
                    f"Row {row} differs."
                )
    assert int(found[0, 0]) == inputs * largest * 4096


# Part 2: the norm, row by row


def rows_with_sum_of_squares(total: int, largest: int, d: int) -> list[int]:
    """A row of d integers, none above `largest`, whose squares add up to exactly `total`."""
    row = []
    while total:
        row.append(min(largest, math.isqrt(total)))
        total -= row[-1] ** 2
    assert len(row) <= d
    return row + [0] * (d - len(row))


def failing_condition(row: list[int], M: int, K: int, eps: int, NX: int) -> int | None:
    """The first range condition that fails when a norm reads the row, or None.

    Conditions are numbered as in docs/model.md, "9 Range conditions", and checked in the order 8,
    1, 2. NX is the number of limbs, 4 or 5.
    """
    x_max = 127 * (256**NX - 1) // 255
    Vx = sum(x * x for x in row)
    if any(x > x_max or x < (-x_max if NX == 5 else -x_max - x_max // 127) for x in row):
        return 8
    if Vx + eps >= 1 << 78 if NX == 5 else Vx >= 1 << 59:
        return 1
    sigma = math.isqrt(Vx + eps)
    Rc = ((1 << M) + sigma // 2) // sigma
    if any(not -32768 <= (x * Rc + (1 << (K - 1))) >> K <= 32639 for x in row):
        return 2
    return None


NORM_PARAMETERS = [
    pytest.param("with_clamps_and_five_limbs", 44, 30, 1, id="five_limbs-M44-K30-eps1"),
    pytest.param(
        "with_clamps_and_five_limbs",
        44,
        30,
        53687,
        id="five_limbs-as_in_the_full_size_parameter_file",
    ),
    pytest.param("with_clamps_and_five_limbs", 34, 21, 1, id="five_limbs-M34-K21-eps1"),
    pytest.param("with_clamps_and_five_limbs", 38, 24, 1, id="five_limbs-M38-K24-eps1"),
    pytest.param(
        "with_clamps_and_five_limbs", 46, 32, (1 << 48) - 1, id="five_limbs-M46-K32-largest_eps"
    ),
    pytest.param("with_clamps_and_five_limbs", 24, 17, 1, id="five_limbs-M24-K17-eps1"),
    pytest.param("without_clamps_with_four_limbs", 46, 32, 1, id="four_limbs-M46-K32-eps1"),
    pytest.param("without_clamps_with_four_limbs", 31, 17, 5, id="four_limbs-M31-K17-eps5"),
]


@pytest.mark.parametrize("base, M, K, eps", NORM_PARAMETERS)
def test_norm_at_the_ends_of_its_ranges(base, M, K, eps, tmp_path, device):
    """Compares sigma, Rc, the quotient w and its clamp with the recorded values, and the range
    condition that fails (1 or 8) with failing_condition().

    Each row is the first token of a separate sequence, in a model whose weights are all 0 and whose
    three norms have the same parameters. So all three read the same row, and only the row decides
    whether a condition fails.
    """
    lines = shape(base) | {"MF": M, "KF": K}
    d = lines["d"]
    for name in ("eps1", "eps2", "epsF"):
        lines |= {name + "_lo": eps & 0xFFFFFFFF, name + "_hi": eps >> 32}
    chosen = ChosenModel(tmp_path, base, lines, [scales_of(base, M1=M, K1=K, M2=M, K2=K)], device)
    NX, T = chosen.parameters.NX, chosen.parameters.T
    x_max = 127 * (256**NX - 1) // 255
    x_min = -x_max if NX == 5 else -x_max - x_max // 127
    # Range condition 1 holds exactly if Vx is below this limit.
    limit = (1 << 78) - eps if NX == 5 else 1 << 59

    def single(x: int) -> list[int]:
        return [0, 0, x] + [0] * (d - 3)

    rows = [[0] * d, [x_max] * d, [x_min] * d]
    rows += [
        single(sign * x)
        for sign in (1, -1)
        for x in (1, 2, 3, 1000, (1 << 24) - 1, 1 << 24, 1 << 29, x_max, x_max + 1)
    ]
    rows += [single(x_min), single(x_min - 1)]
    # Condition 1 at its end: the largest allowed sum of squares, and the smallest that is not.
    rows += [
        rows_with_sum_of_squares(limit - 1, x_max, d),
        rows_with_sum_of_squares(limit, x_max, d),
    ]
    # Vx + eps next to perfect squares n^2, for the exact square root.
    for n in (5, 1 << 14, (1 << 14) + 1, 94906267, (1 << 29) - 3):
        rows += [
            rows_with_sum_of_squares(n * n + delta - eps, x_max, d)
            for delta in (-1, 0, 1)
            if n * n + delta - eps > 0
        ]
    # With a single entry x between 2^(M + 1) / 3 and 2^(M + 1), sigma = |x| (if eps <= 2 |x|) and
    # Rc = 1, so w = rq(x, K): the largest quotients, with |w| at most 2^(M + 1 - K). Condition 2
    # would fail from x = `tie` on, but no parameter file that Parameters.read accepts lets a row
    # get there: at such x, Rc is 0 or the row fails condition 8. This is argued here; the asserts
    # after the loop check it for these parameters only.
    tie = 32640 * (1 << K) - (1 << (K - 1))
    rows += [single(x) for x in (tie - 1, tie, 2 << M, -(2 << M), (2 << M) + 1, -(2 << M) - 1)]
    # sigma = 2^14 and 2^15. The quotient is then x * 2^(M - K - 14) or half of that: the clamp's
    # ends, and ties.
    for exponent in (14, 15):
        part = [2046, 2047, 2048, 2049, -2046, -2047, -2048, -2049, 1, -1, 3, -3, 4095, -4095]
        missing = (1 << (2 * exponent)) - eps - sum(x * x for x in part)
        if missing > 0:
            rows.append(part + [math.isqrt(missing - 1) + 1] + [0] * (d - len(part) - 1))

    recorded = recorded_values.functions(
        {"norm_x": numpy.array(rows), "norm_M": M, "norm_K": K, "norm_eps": eps, "norm_limbs": NX}
    )
    defined = recorded.whole[("norm_defined", None)]
    met = set()
    # What each of the three norms gives for the rows at which every range condition holds.
    admitted = {prefix: {"_sigma": [], "_Rc": [], "_w": [], "": []} for prefix in ("a", "m", "aF")}
    for start in range(0, len(rows), T):
        some = rows[start : start + T]
        model = chosen.model(numpy.array(some + [[0] * d] * (T - len(some))))
        for t, row in enumerate(some):
            expected_condition = failing_condition(row, M, K, eps, NX)
            met.add(expected_condition)
            assert bool(defined[start + t]) == (expected_condition is None), row
            recorder = recorded_values.Recorder()
            if expected_condition is not None:
                with pytest.raises(RangeConditionFailed) as raised:
                    model.extend(model.new_sequence(), [t])
                assert (raised.value.position, raised.value.condition) == (0, expected_condition), (
                    row
                )
                continue
            model.extend(model.new_sequence(), [t], record=recorder)
            found = recorder.arrays()
            for prefix, layer in (("a", 0), ("m", 0), ("aF", None)):
                for suffix, values in admitted[prefix].items():
                    values.append(found[(prefix + suffix, layer)][0])
    for prefix, by_suffix in admitted.items():
        for suffix, name in (("_sigma", "sigma"), ("_Rc", "Rc"), ("_w", "w"), ("", "y")):
            message = recorded.difference("norm_" + name, None, numpy.stack(by_suffix[suffix]))
            assert message is None, f"The norm {prefix}: {message}"
    # Every outcome occurs except a failure of condition 2.
    assert met == {None, 1, 8}
    quotients = numpy.stack(admitted["a"]["_w"])
    assert abs(quotients).max() <= 1 << (M + 1 - K)
    if (M, K, NX) == (38, 24, 5):  # the closest that an accepted file comes to condition 2's ends
        assert (quotients.min(), quotients.max()) == (-32639, 32639)
    if M - K == 14 and eps == 1:
        assert {2046, 2047, 2048, -2047, -2048} <= set(quotients.flatten().tolist())


def test_condition_2_on_the_norm_function_alone(tmp_path, device):
    """The ends of range condition 2: w = -32,768 and 32,639 are allowed; 32,640 and 32,768 are not.

    No parameter file that Parameters.read accepts lets this condition fail (see the test above), so
    in a loaded model the check is only a safeguard. The test calls the model's norm itself with
    M = 34 and K = 20, a pair that Parameters.read rejects with five limbs, and compares the result
    with the recorded values.
    """
    base, M, K, eps = "with_clamps_and_five_limbs", 34, 20, 1
    chosen = ChosenModel(tmp_path, base, shape(base), [scales_of(base)], device)
    p = chosen.parameters
    model = chosen.model(numpy.zeros((p.T, p.d), dtype=numpy.int64))
    tie = 32640 * (1 << K) - (1 << (K - 1))
    cases = {tie - 1: 32639, tie: 32640, 2 << M: 32768, -(2 << M): -32768}
    rows = [[0, 0, x] + [0] * (p.d - 3) for x in cases]
    recorded = recorded_values.functions(
        {
            "norm_x": numpy.array(rows),
            "norm_M": M,
            "norm_K": K,
            "norm_eps": eps,
            "norm_limbs": p.NX,
        }
    )
    admitted = {"n_w": [], "n": []}
    for number, (x, quotient) in enumerate(cases.items()):
        holds = -32768 <= quotient <= 32639
        assert bool(recorded.whole[("norm_defined", None)][number]) == holds, x
        found, checks = {}, _RangeChecks(1, torch.device(device))
        model._norm(
            tensor([rows[number]], device),
            M,
            K,
            eps,
            checks,
            lambda name, layer, value: found.update({name: value[0].tolist()}),
            None,
            "n",
        )
        assert checks.first_failure() == (None if holds else (0, 2)), x
        assert (x * found["n_Rc"] + (1 << (K - 1))) >> K == quotient
        if holds:
            for name, values in admitted.items():
                values.append(found[name])
    assert recorded.difference("norm_w", None, numpy.array(admitted["n_w"])) is None
    assert recorded.difference("norm_y", None, numpy.array(admitted["n"])) is None


# Part 3: whole models with chosen weights, against the recorded values

# Where a range condition fails, the recorded message has "NO PROOF", a sentence that names the
# value, and the layer. Each pattern maps a phrase of that sentence to the conditions of
# docs/model.md, "9 Range conditions", that it can mean. Exactly one pattern must match.
MESSAGE_PATTERNS = {
    # "2^30": a check that the second implementation makes with four limbs only, on the stream that
    # the final norm reads. There condition 1 is narrower (|x| >= 2^30 gives Vx >= 2^60 > 2^59), so
    # the model reports condition 1, or 8 for a value beyond 2,139,062,143.
    r"(four|five) limbs|2\^30": {1, 8},
    r"norm's quotient": {2},
    r"\b(qw|kw)\b": {3},
    r"\bvw\b": {4},
    r"\blr\b": {5},
    r"\bgate\b|\bgi\b": {6},
    r"\bu\b": {7},
}


def conditions_meant(sentence: str) -> set[int]:
    matching = [
        conditions
        for pattern, conditions in MESSAGE_PATTERNS.items()
        if re.search(pattern, sentence)
    ]
    assert len(matching) == 1, (
        f"It is not clear which range condition the recorded sentence means: {sentence}"
    )
    return matching[0]


def test_the_recorded_sentences_are_understood():
    """The sentences that a recorded message can have, each with the conditions that it means."""
    for sentence, conditions in {
        "a norm's sum of squares plus eps is 2^78 or more, or a residual-stream value does not fit "
        "five limbs": {
            1,
            8,
        },
        "a norm's sum of squares is 2^59 or more, or a residual-stream value does not fit four "
        "limbs": {
            1,
            8,
        },
        "the claimed last stream leaves 2^30": {1, 8},
        "a norm's quotient is outside [-32768, 32639]": {2},
        "qw or kw outside [-32768, 32639]": {3},
        "vw outside [-32768, 32639]": {4},
        "the wide lr outside -262144 .. 261119": {5},
        "a router logit (lr) is outside [-16384, 16383]": {5},
        "the gate's wide key outside -4194432 .. 4194175": {6},
        "gi outside [-32768, 32639]": {6},
        "u outside its two limbs (and, with UT, 16 bits)": {7},
    }.items():
        assert conditions_meant(sentence) == conditions


def test_changed_inputs_find_no_recorded_values(tmp_path, device):
    """With one weight changed, no recorded case matches: ChosenModel.recorded() raises."""
    base = "without_clamps_with_four_limbs"
    chosen = ChosenModel(tmp_path, base, shape(base), [scales_of(base), scales_of(base)], device)
    rows = numpy.stack([chosen.row({}) for _ in range(chosen.parameters.T)])
    chosen.matrix("Wq", 0)[0, 0] = 1234
    with pytest.raises(AssertionError, match="no recorded values for this model and window"):
        chosen.recorded(rows)


def assert_fails_as_recorded(
    chosen: ChosenModel, rows, position: int, condition: int, layer: int
) -> None:
    """A window in which one position fails one range condition: the recorded values and the model
    must report it, however the window is fed.
    """
    case = chosen.recorded(rows)
    assert not case.whole[("defined", None)], (
        "By the recorded values, no range condition fails in this window."
    )
    said = re.search(r"NO PROOF[^:]*: (.*), layer (\d+)", case.message)
    assert said, f"The recorded message has no NO PROOF, sentence and layer: {case.message}"
    assert condition in conditions_meant(said[1]), case.message
    assert int(said[2]) == layer, case.message
    for split in chosen.splits():
        model = chosen.model(rows)
        sequence = model.new_sequence()
        start = 0
        with pytest.raises(RangeConditionFailed) as raised:
            for length in split:
                model.extend(sequence, chosen.window[start : start + length])
                start += length
        assert (raised.value.position, raised.value.condition) == (position, condition)
        assert start <= position < start + length, (
            "The exception must come from the call that feeds the failing position."
        )
        # The sequence stays unusable, and the exception names the same failure.
        with pytest.raises(RangeConditionFailed, match=re.escape(str(raised.value))):
            model.extend(sequence, [0])


# The ends of the two-limb range -32,768 .. 32,639 and of the clamp to -2,047 .. 2,047, with their
# neighbours.
SIXTEEN_BITS = [
    -32768,
    -32767,
    -2049,
    -2048,
    -2047,
    -2046,
    -1,
    0,
    1,
    2046,
    2047,
    2048,
    2049,
    32638,
    32639,
]
# Wide gate keys gw = 256 gh + g0: the ends of range condition 6, of clamp(gh; 127) and of the low
# byte g0.
GATE_KEYS = [
    -4194432,
    -4194431,
    -32897,
    -32896,
    -32641,
    -32640,
    -32639,
    -129,
    -128,
    -127,
    0,
    127,
    128,
    129,
    32639,
    32640,
    32641,
    32768,
    32895,
    32896,
    4194174,
    4194175,
]


def sums_for(values: list[int]) -> list[int]:
    """Sums S with rq(S, 1) = the given values N: S = 2 N at even indices and S = 2 N - 1, an exact
    tie, at odd ones.
    """
    return [2 * value - (place % 2) for place, value in enumerate(values)]


def clamped_router_logit(lw: int) -> int:
    """The router logit lr with the router's clamp: 8 clamp(ly; 2,047) + l3, where lw = 8 ly + l3
    and l3 is in 0 .. 7.
    """
    return 8 * min(max(lw >> 3, -2047), 2047) + (lw & 7)


@ALL_BASES
def test_every_clamp_and_range_at_its_ends(base, tmp_path, device):
    """Queries, keys, values, router logits, gate keys, up projections and logits at chosen values.

    Each position of the window, except one whose row is all zeros, switches on one group of inputs
    (ChosenModel.on() in chosen_model.py), whose weights give the first outputs of the products a
    chosen sum (ChosenModel.set_sums()). Half of the sums made by sums_for() are exact ties of the
    rounding shift, many of them negative. Wo, Wd and Wdr stay 0, so the residual stream keeps its
    input.
    """
    NE, fe = 8, 32
    lines = shape(base, f=2 * fe, NE=NE)
    chosen = ChosenModel(tmp_path, base, lines, [scales_of(base), scales_of(base)], device)
    p = chosen.parameters
    gate_keys = GATE_KEYS if p.gate_clamp else SIXTEEN_BITS
    winner = 2  # the routed expert of the positions at which the feed-forward values are chosen

    # The group `ends` carries the edge values of Wq, Wk, Wv, Wg, Wu, Wgr and Wur; each router case
    # and each head case gets a group of its own. The winner's logit is so large that sc = 2^16.
    # The routed expert's input xr = rq(sc m, 16) is then m itself.
    ends = chosen.new_inputs(3)
    for layer in (0, 1):
        for name in ("Wq", "Wk", "Wv", "Wu"):
            chosen.set_sums(name, ends, sums_for(SIXTEEN_BITS), layer)
        chosen.set_sums("Wg", ends, sums_for(gate_keys), layer)
    chosen.set_sums("Wgr", ends, sums_for(gate_keys), 1, winner)
    chosen.set_sums("Wur", ends, sums_for(SIXTEEN_BITS), 1, winner)
    chosen.set_sums("Wr", ends, sums_for([3000 if e == winner else 0 for e in range(NE)]), 1)
    chosen.set_sums("Wh", ends, [3 * v - 7 for v in range(p.V)])

    # Positions for the router: wide logits lw of the NE experts.
    if p.router_clamp:
        lowest, highest = -262144, 261119
        router_cases = [
            [77] * NE,  # all equal
            [5, 9, 1, 200, 7, 200, 200, -3],  # three equal at the top
            # The clamp is not monotone: 16,384 ranks below 16,383.
            [16384, 16383, 16391, 16376, 16375, 0, 0, 0],
            [16382, highest, 16383, highest - 7, 0, 0, 0, 0],
            # The same at the lower end.
            [lowest, -16377, -16376, -16384, -16385, -20000, -16392, lowest + 7],
            [highest, lowest, 0, 0, 0, 0, 0, 0],
        ]
    else:
        lowest, highest = -16384, 16383
        router_cases = [
            [77] * NE,
            [5, 9, 1, 200, 7, 200, 200, -3],
            [16382, highest, highest, 0, 0, 0, 0, 0],
            [lowest] * NE,
            [lowest, lowest + 1, lowest, lowest + 1, lowest, lowest, lowest, lowest],
            [highest, lowest, 0, 0, 0, 0, 0, 0],
        ]
    router_inputs = [chosen.new_inputs(3) for _ in router_cases]
    for first, case in zip(router_inputs, router_cases):
        chosen.set_sums("Wr", first, sums_for(case), 1)

    # Positions for the head: logits lam of the V tokens. The head has no shift, so a sum is a
    # logit.
    head_cases = [
        [1000 if v in (3, 7) else -5 for v in range(p.V)],  # two equal at the top
        [42] * p.V,
        # All negative, the last two equal at the top.
        [-9 if v >= p.V - 2 else -10 - v for v in range(p.V)],
    ]
    head_inputs = [chosen.new_inputs(3) for _ in head_cases]
    for first, case in zip(head_inputs, head_cases):
        chosen.set_sums("Wh", first, case)

    # One flag (an input that ChosenModel.on() sets to 1) for each end of each range. A flag moves
    # one sum by 2, so it moves the value at the range's end out by 1. `failures` maps (the matrix's
    # key, "low" or "high") to (the flag, its group of inputs, the condition, the layer).
    failures = {}
    sites = [
        (("Wq", 0), 3, SIXTEEN_BITS),
        (("Wk", 0), 3, SIXTEEN_BITS),
        (("Wv", 0), 4, SIXTEEN_BITS),
        (("Wq", 1), 3, SIXTEEN_BITS),
        (("Wv", 1), 4, SIXTEEN_BITS),
        (("Wg", 0), 6, gate_keys),
        (("Wu", 0), 7, SIXTEEN_BITS),
        (("Wg", 1), 6, gate_keys),
        (("Wu", 1), 7, SIXTEEN_BITS),
        (("Wgr", 1, winner), 6, gate_keys),
        (("Wur", 1, winner), 7, SIXTEEN_BITS),
    ]
    for key, condition, values in sites:
        # The lists of values are sorted, so the ends of the range are in the first and the last
        # column.
        for end, column, step in (("low", 0, -2), ("high", len(values) - 1, 2)):
            flag = chosen.new_inputs(1)
            chosen.matrix(*key)[flag, column] = step
            failures[key, end] = (flag, ends, condition, key[1])
    # The last router case holds the highest and the lowest allowed logit.
    for end, column, step in (("high", 0, 2), ("low", 1, -2)):
        flag = chosen.new_inputs(1)
        chosen.matrix("Wr", 1)[flag, column] = step
        failures[("Wr", 1), end] = (flag, router_inputs[-1], 5, 1)
    assert len(failures) == 2 * (len(sites) + 1), 'A key of sites repeats, or is ("Wr", 1).'

    groups = [ends, *router_inputs, *head_inputs]
    rows = numpy.stack([chosen.on(groups[t % len(groups)]) for t in range(p.T)])
    # A row of zeros: sigma = 1, every product is 0, and every argmax is a tie of all its entries.
    rows[len(groups)] = 0
    arrays = chosen.assert_equals_recorded(rows)

    for layer in (0, 1):
        for name in ("qw", "kw", "vw", "u"):
            assert arrays[(name, layer)][0, : len(SIXTEEN_BITS)].tolist() == SIXTEEN_BITS
        assert arrays[("gw", layer)][0, : len(gate_keys)].tolist() == gate_keys
    assert arrays[("gw", 1)][0, fe : fe + len(gate_keys)].tolist() == gate_keys
    assert arrays[("u", 1)][0, fe : fe + len(SIXTEEN_BITS)].tolist() == SIXTEEN_BITS
    assert arrays[("e_star", 1)][0] == winner
    assert arrays[("sc", 1)][0] == 1 << 16
    for t, case in enumerate(router_cases, start=1):
        lr = [clamped_router_logit(lw) for lw in case] if p.router_clamp else case
        assert arrays[("lr", 1)][t].tolist() == lr
        assert arrays[("e_star", 1)][t] == lr.index(max(lr))
    for t, case in enumerate(head_cases, start=1 + len(router_cases)):
        assert arrays[("lam", None)][t].tolist() == case
        assert arrays[("first_choice", None)][t] == case.index(max(case))
    assert arrays[("first_choice", None)][len(groups)] == 0
    assert arrays[("e_star", 1)][len(groups)] == 0

    for number, (flag, group, condition, layer) in enumerate(failures.values()):
        position = (5 * number + 1) % p.T
        failing_rows = rows.copy()
        failing_rows[position] = chosen.on(group, flags=[flag])
        assert_fails_as_recorded(chosen, failing_rows, position, condition, layer)

    # Two conditions fail at one position: both implementations name the one that comes first in the
    # order of evaluation.
    failing_rows = rows.copy()
    failing_rows[3] = chosen.on(
        ends, flags=[failures[("Wu", 0), "high"][0], failures[("Wq", 0), "high"][0]]
    )
    assert_fails_as_recorded(chosen, failing_rows, 3, 3, 0)

    # Two failing positions: the exception names the earlier position, although the later one fails
    # in an earlier layer.
    failing_rows = rows.copy()
    failing_rows[4] = chosen.on(ends, flags=[failures[("Wu", 1), "high"][0]])  # condition 7
    failing_rows[9] = chosen.on(ends, flags=[failures[("Wq", 0), "low"][0]])  # condition 3
    for split in chosen.splits():
        model = chosen.model(failing_rows)
        sequence, start = model.new_sequence(), 0
        with pytest.raises(RangeConditionFailed) as raised:
            for length in split:
                model.extend(sequence, chosen.window[start : start + length])
                start += length
        assert (raised.value.position, raised.value.condition) == (4, 7)


@ALL_BASES
def test_softmax_table_at_its_last_entry_and_negative_sums(base, tmp_path, device):
    """Attention at the table keys u0 - 1 and u0, with negative sums, ties of the last rounding and
    a late maximum.

    Query head 0 has the query (2047, 1, 0, ..) at every position, so the score of position j is
    2047 b_j + c_j for the key (b_j, c_j, 0, ..), whatever the querying position is. Position 0 has
    the largest score until a later position exceeds it. Query head 1 has the query 0, so all its
    scores are equal and its output is the rounded mean of the values.

    The last non-zero entry of the table is EX[u0 - 1] = 1, against EX[0] = 4096. The five cases
    are positions 1 to 5, whose scores lie below position 0's by the first five distances(): table
    keys u0 - 1 twice, u0 twice, and far beyond. To make so small a weight visible, case c has two
    value columns, 2 (c - 1) and 2 c - 1. In the first, position 0 holds -2 and the case's position
    holds 2047. The weighted mean at the case's position is then just above -1.5 if the case's
    weight is 1, which rounds to -1, and about -2 if the weight is 0. The second column holds the
    opposite signs.
    """
    # sh, the softmax shift, is set per layer, so the two layers get different shifts.
    shifts = (4, 8)
    chosen = ChosenModel(
        tmp_path,
        base,
        shape(base, heads=2),
        [scales_of(base, kq=12, kk=12, sh=sh) for sh in shifts],
        device,
    )
    p = chosen.parameters
    u0 = len(chosen.model(numpy.zeros((p.T, p.d), dtype=numpy.int64)).tables.softmax) - 1
    query = chosen.new_inputs(2)
    keys = [chosen.new_inputs(2) for _ in shifts]
    columns = min(p.dh, 24)
    values = chosen.new_inputs(columns)
    for layer in range(len(shifts)):
        for i in (0, 1):
            # With the shift 12 = cs (the rotary tables' scale exponent), q and k are the products.
            chosen.matrix("Wq", layer)[query + i, i] = 1
            chosen.matrix("Wk", layer)[keys[layer] + i, i] = 1
        for column in range(columns):
            # With the shift 1, v is the input.
            chosen.matrix("Wv", layer)[values + column, column] = 2

    top = 2047 * 1000  # the score of position 0

    def distances(sh: int) -> list[int]:
        """For the positions 1, 2, ..: how far the position's score is below the score of position
        0.
        """
        step = 1 << sh
        return [
            # Positions 1 to 5: the five cases.
            (u0 - 1) * step,
            (u0 - 1) * step + step - 1,
            u0 * step,
            u0 * step + 1,
            top + 2047 * 2047,
            # The keys 1 and 0, a tie with the maximum, and a new, larger maximum.
            step,
            step - 1,
            1,
            0,
            -2047 * 500,
        ]

    random = numpy.random.default_rng(4)
    rows = []
    for t in range(p.T):
        entries = {query: 2047, query + 1: 1}
        for layer, sh in enumerate(shifts):
            below = [0, *distances(sh)]
            score = (
                top - below[t] if t < len(below) else top - int(random.integers(0, (u0 + 2) << sh))
            )
            b = (score + 1023) // 2047
            entries |= {keys[layer]: b, keys[layer] + 1: score - 2047 * b}
        if t == 0:
            entries |= {values + column: 2 if column % 2 else -2 for column in range(columns)}
        elif t <= 5:
            entries |= {values + 2 * (t - 1): 2047, values + 2 * (t - 1) + 1: -2047}
        else:
            entries |= {
                values + column: int(v)
                for column, v in enumerate(random.integers(-2047, 2048, columns))
            }
        rows.append(chosen.row(entries))
    arrays = chosen.assert_equals_recorded(numpy.stack(rows))

    for layer, sh in enumerate(shifts):
        q, k, o = (arrays[(name, layer)] for name in ("q", "k", "o"))
        scores = k[:, :2] @ q[0, :2]
        assert (top - scores[1:11]).tolist() == distances(sh)
        for case in range(1, 6):
            # The table's key is u0 - 1 in the first two cases and at least u0 in the others.
            weight_is_one = case <= 2
            assert o[case, 2 * (case - 1) : 2 * case].tolist() == (
                [-1, 1] if weight_is_one else [-2, 2]
            )
        # Head 1 at position 1: the means (-2 + 2047) / 2 and (2 - 2047) / 2 are exact ties, and
        # both go up.
        assert o[1, p.dh : p.dh + 2].tolist() == [1023, -1022]


@ALL_BASES
def test_rotary_step_on_chosen_products(base, tmp_path, device):
    """The rotary step at every position and pair, on large products in layer 0 and on very small
    ones in layer 1.
    """
    layers = [scales_of(base, rp=1, kq=22, kk=22), scales_of(base, rp=1)]
    chosen = ChosenModel(tmp_path, base, shape(base), layers, device)
    p = chosen.parameters
    group = chosen.new_inputs(3)
    for layer, size in enumerate((16000000, 3)):
        # The two products of each pair, A0 and A1, take the 25 combinations of these five values
        # in turn, as far as the matrix has pairs.
        steps = (-size, -size // 2, 0, size // 2 + 1, size)
        for name, outputs, offset in (("Wq", p.dq, 0), ("Wk", p.dkv, 7)):
            pairs = [
                (steps[(i + offset) % 5], steps[((i + offset) // 5) % 5])
                for i in range(outputs // 2)
            ]
            chosen.set_sums(name, group, [product for pair in pairs for product in pair], layer)
    arrays = chosen.assert_equals_recorded(numpy.stack([chosen.on(group)] * p.T))
    assert numpy.abs(arrays[("qw", 0)]).max() > 20000  # close to the end of range condition 3
    # The same row enters at every position, so only the rotary tables make the positions differ.
    assert len({tuple(row) for row in arrays[("qw", 1)].tolist()}) > p.T // 2


@ALL_BASES
def test_attention_at_its_largest_sums(base, tmp_path, device):
    """Every entry of every query, key and value at the clamp's ends: the largest scores and
    weighted sums.
    """
    for value_sign in (1, -1):
        folder = tmp_path / f"values_{value_sign}"
        folder.mkdir()
        chosen = ChosenModel(
            folder, base, shape(base, heads=4, G=2), [scales_of(base, sh=4)], device
        )
        p = chosen.parameters
        plus, minus = chosen.new_inputs(3), chosen.new_inputs(3)
        for first, sign in ((plus, 1), (minus, -1)):
            chosen.set_sums("Wq", first, [2 * 30000] * p.dq, 0)
            chosen.set_sums("Wk", first, [sign * 2 * 30000] * p.dkv, 0)
            chosen.set_sums("Wv", first, [value_sign * sign * 2 * 30000] * p.dkv, 0)
        pattern = [plus] * (p.T // 2) + [minus, plus] * (p.T // 4)
        arrays = chosen.assert_equals_recorded(numpy.stack([chosen.on(first) for first in pattern]))
        assert set(arrays[("q", 0)].flatten().tolist()) == {2047}
        assert (
            set(arrays[("k", 0)].flatten().tolist())
            == {-2047, 2047}
            == set(arrays[("v", 0)].flatten().tolist())
        )
        # A position with the key -2047 scores so far below the maximum that its table entry is 0,
        # so its value does not count.
        assert set(arrays[("o", 0)].flatten().tolist()) == {value_sign * 2047}


def total_weight_for(product: int, residue: int, shift: int, sign: int, largest: int) -> int | None:
    """An integer W of the given sign with |W| <= largest and product * W = residue modulo 2^shift,
    or None.

    Of all such integers, the function returns the one with the largest absolute value.
    """
    modulus = 1 << shift
    common = math.gcd(product, modulus)
    if residue % common:
        return None
    modulus //= common
    W = (residue // common) * pow(product // common, -1, modulus) % modulus if modulus > 1 else 0
    W = largest - (largest - W) % modulus if sign > 0 else -largest + (W + largest) % modulus
    return W if W * sign > 0 else None


def column_with_total(total: int, slots: int) -> numpy.ndarray:
    """`slots` weights in -4,095 .. 4,095 that add up to the given total, as many as possible at one
    end.
    """
    sign = 1 if total >= 0 else -1
    column = numpy.full(slots, sign * 4095, dtype=numpy.int64)
    excess = abs(int(column.sum()) - total)
    for slot in range(slots):
        step = min(excess, 8190)
        column[slot] -= sign * step
        excess -= step
    assert int(column.sum()) == total
    return column


def down_projection_model(folder, base: str, experts: bool, kd: int, device: str):
    """A model whose last layer has f = 16,384 hidden units, as at full size, all with the largest
    |s u|.

    Every unit has s = smax = 4,095. A row switches on one of two groups of inputs: the first gives
    u = -32,768 in every unit, the second u = 32,639. Returns the model, the two groups, the two
    products s u and the down projection's matrix, indexed [unit][output]. With experts, the layer
    is a mixture-of-experts layer: the first half of the units is the shared expert, the second
    half routed expert 1.
    """
    f = 16384
    layers = [scales_of(base), scales_of(base, kd=kd)] if experts else [scales_of(base, kd=kd)]
    chosen = ChosenModel(folder, base, shape(base, f=f), layers, device)
    layer = len(layers) - 1
    groups, products = [chosen.new_inputs(3), chosen.new_inputs(3)], [4095 * -32768, 4095 * 32639]
    halves = (
        [("Wg", "Wu", "Wd", None), ("Wgr", "Wur", "Wdr", 1)]
        if experts
        else [("Wg", "Wu", "Wd", None)]
    )
    for group, u in zip(groups, (-32768, 32639)):
        for gate, up, _, expert in halves:
            chosen.set_sums(gate, group, [2 * 20000] * (f // len(halves)), layer, expert)
            chosen.set_sums(up, group, [2 * u] * (f // len(halves)), layer, expert)
        if experts:
            # Expert 1 wins with sc = 2^16, so that xr = m.
            chosen.set_sums("Wr", group, [0, 2 * 3000], layer)

    # A sum that is -`tie` modulo 2^kd is an exact tie of rq(., kd); one less is rounded down.
    tie = 1 << (kd - 1)
    totals = [4095 * f, -4095 * f, 0]
    for product in products:
        for sign in (1, -1):
            for residue in (-tie, -tie - 1):
                W = total_weight_for(product, residue % (1 << kd), kd, sign, 4095 * f)
                if W is not None and abs(W) > 4000 * f:
                    totals.append(W)
    assert len(totals) <= chosen.parameters.d - 1
    down = numpy.zeros((f, chosen.parameters.d), dtype=numpy.int64)
    for output, total in enumerate(totals):
        down[:, output] = column_with_total(total, f)
    # Output 2 has the total 0, with partial sums as large as the order of summation makes them.
    down[::2, 2], down[1::2, 2] = 4095, -4095
    for number, (_, _, name, expert) in enumerate(halves):
        chosen.matrix(name, layer, expert)[:] = down[number * (f // len(halves)) :][
            : f // len(halves)
        ]
    return chosen, groups, products, down


@ALL_BASES
@pytest.mark.parametrize("experts", [False, True], ids=["dense_layer", "layer_with_experts"])
def test_down_projection_at_its_largest_sums(base, experts, tmp_path, device):
    """Sums of 16,384 terms up to 9.003e15, just below 2^53 = 9.007e15. Some are exact ties of the
    rounding shift.
    """
    five_limbs = Parameters.read(BASES[base]).NX == 5
    # Roughly the smallest shifts with which the sums still fit the residual stream.
    kd = 17 if five_limbs else 26
    chosen, groups, products, down = down_projection_model(tmp_path, base, experts, kd, device)
    p = chosen.parameters
    rows = numpy.stack([chosen.on(groups[t % 2]) for t in range(p.T)])
    arrays = chosen.assert_equals_recorded(rows)

    layer = p.L - 1
    sizes, ties = [], 0
    for t in (0, 1):
        assert set(arrays[("s", layer)][t].tolist()) == {4095}
        assert set(arrays[("u", layer)][t].tolist()) == {products[t] // 4095}
        sums = [products[t] * int(total) for total in down.sum(axis=0)]
        added = (arrays[("x_last", None)][t] - arrays[("x_mid", layer)][t]).tolist()
        assert added == [(total + (1 << (kd - 1))) >> kd for total in sums]
        sizes += [abs(total) for total in sums]
        ties += sum(
            (total + (1 << (kd - 1))) % (1 << kd) == 0 and abs(total) > 8.9e15 for total in sums
        )
    assert max(sizes) == 4095 * 32768 * 4095 * 16384
    assert ties >= (3 if five_limbs else 1)


@ALL_BASES
def test_stream_value_too_large_after_the_down_projection(base, tmp_path, device):
    """With a smaller shift the same sums leave the residual stream's range: condition 8, at the
    norm that follows.
    """
    kd = 14 if Parameters.read(BASES[base]).NX == 5 else 21
    chosen, groups, _, _ = down_projection_model(tmp_path, base, False, kd, device)
    rows = numpy.stack([chosen.row({}) for _ in range(chosen.parameters.T)])
    rows[6] = chosen.on(groups[0])
    assert_fails_as_recorded(chosen, rows, 6, 8, chosen.parameters.L)


# Part 4: what is rejected when a model is loaded


def test_shapes_whose_sums_could_round_are_rejected(tmp_path, device):
    """A product in float64 is exact only below 2^53. The down projection stays below at f = 16,384,
    not at 32,768.
    """
    base = "with_clamps_and_five_limbs"
    for f, fits in ((16384, True), (32768, False)):
        folder = tmp_path / str(f)
        folder.mkdir()
        chosen = ChosenModel(folder, base, shape(base, f=f), [scales_of(base)], device)
        rows = numpy.zeros((chosen.parameters.T, chosen.parameters.d), dtype=numpy.int64)
        if fits:
            chosen.model(rows)
        else:
            with pytest.raises(ParameterError, match=r"cannot evaluate a model .* bound \(e\)"):
                chosen.model(rows)


@ALL_BASES
@pytest.mark.parametrize("table", ["tsilu", "tsig", "trope", "texp"])
def test_a_table_with_another_digest_is_rejected(base, table, tmp_path, device):
    """The parameter file states the digests of the public tables. One changed word of one digest
    must stop the load: the tables computed at loading would not be the proved model's.
    """
    stated = Parameters.read(BASES[base]).table_digests[table][7]
    lines = shape(base) | {f"{table}7": stated ^ 1}
    chosen = ChosenModel(tmp_path, base, lines, [scales_of(base)], device)
    with pytest.raises(ParameterError, match=rf"\(lines {table}0 to {table}7\)"):
        chosen.model(numpy.zeros((chosen.parameters.T, chosen.parameters.d), dtype=numpy.int64))


def test_weights_at_and_beyond_the_ends_of_their_range(tmp_path, device):
    """A committed weight is in -4,096 .. 4,095: both ends are accepted and used as they are, and
    nothing beyond is. The second implementation takes no weight -4,096, so the expected values are
    computed here.
    """
    base = "with_clamps_and_five_limbs"
    chosen = ChosenModel(tmp_path, base, shape(base), [scales_of(base)], device)
    p = chosen.parameters
    rows = numpy.stack([chosen.row({0: 3, 1: -5})] * p.T)
    matrices = {
        key: torch.from_numpy(chosen.matrix(*key).T.astype(numpy.int16)) for key in chosen.matrices
    }
    matrices[("Wh", None, None)][:4, :2] = torch.tensor(
        [[-4096, 0], [4095, 0], [0, -4096], [-4096, 4095]]
    )
    model = chosen.model(rows)
    model.weights = WeightsInMemory(matrices, torch.from_numpy(rows), device)
    recorder = recorded_values.Recorder()
    model.extend(model.new_sequence(), [0], record=recorder)
    assert recorder.arrays()[("lam", None)][0, :4].tolist() == [
        3 * -4096,
        3 * 4095,
        -5 * -4096,
        3 * -4096 - 5 * 4095,
    ]
    for outside in (-4097, 4096):
        matrices[("Wh", None, None)][0, 0] = outside
        with pytest.raises(WeightsRejected, match="has an entry outside -4096 .. 4095"):
            WeightsInMemory(matrices, torch.from_numpy(rows), device)
