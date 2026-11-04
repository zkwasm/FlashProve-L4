# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The public noise of a seeded request (integer_model/noise.py), and the choices and logit gaps
that IntegerModel.extend computes with it. A token's gap is the choice's logit minus the token's,
noise added to both, plus 1 if the choice's id is the smaller."""

import hashlib
import os
import pathlib
import random
import shutil
import struct
import subprocess
import sys

import pytest
import torch

import integer_model.model
from integer_model import IntegerModel, Parameters, SyntheticWeights
from integer_model.noise import TEMPERATURE_ONE, largest_noise, noise, philox4x32_10, slots, table

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
SMALL_EXAMPLE = PACKAGE.parents[2] / "prover" / "examples" / "target_small_f2"
MASK_32 = (1 << 32) - 1


# The noise rule again, in plain Python integers, but for its table, which comes from table()


def plain_philox(
    key: tuple[int, int], counter: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Philox4x32 with ten rounds in plain Python integers: the four result words.

    J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel Random Numbers: As Easy as 1,
    2, 3", SC '11 (2011): the S-box and its multipliers in section 4.3, the round keys in equation 6
    with the upper halves of the two constants printed below it.
    """

    def s_box(left: int, right: int, multiplier: int, round_key: int) -> tuple[int, int]:
        high, low = divmod(right * multiplier, 1 << 32)
        return low, high ^ round_key ^ left

    steps = (0x9E3779B97F4A7C15 >> 32, 0xBB67AE8584CAA73B >> 32)
    words = list(counter)  # right, left of the first S-box, then right, left of the second
    for i in range(10):
        round_keys = [(k + i * step) % (1 << 32) for k, step in zip(key, steps)]
        words[0], words[2] = words[2], words[0]
        words[1], words[0] = s_box(words[1], words[0], 0xCD9E8D57, round_keys[0])
        words[3], words[2] = s_box(words[3], words[2], 0xD2511F53, round_keys[1])
    return words[0], words[1], words[2], words[3]


def plain_slot(word: int) -> int:
    """The 16-bit index into table() of a 32-bit random word, as noise.slots computes it."""
    x = word ^ MASK_32 if word >= 1 << 31 else word
    side = 1 if x >= 1 << 30 else 0
    k = (1 << 31) - 1 - x if side else x
    e = k.bit_length()
    below_the_leading_bit = format(k, "b")[1:11].ljust(10, "0") if e else "0"
    return (side << 15) + (e << 10) + int(below_the_leading_bit, 2)


def plain_noise(seed: int, t: int, v: int, temp: int, s: int) -> int:
    seed %= 1 << 64
    key = plain_philox((seed & MASK_32, seed >> 32), (t, 0, 0, 0))[0]
    word = plain_philox((key, 0), (v, 0, 0, 0))[0]
    return (temp * table()[plain_slot(word)]) >> (56 - s)


# The noise

# Three known-answer values of Philox4x32-10: counter, key, result. They are meant to be those of
# Random123's file kat_vectors. They were compared with the file tests/kat_vectors in Random123's
# release 1.14.0, by program, once, and all three are equal.
EXPECTED_VALUES = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((MASK_32,) * 4, (MASK_32,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize("counter, key, expected", EXPECTED_VALUES)
def test_philox_gives_the_expected_values(counter, key, expected, device):
    words = philox4x32_10(
        *key, *(torch.tensor([c], dtype=torch.int64, device=device) for c in counter)
    )
    assert tuple(int(word) for word in words) == expected
    assert plain_philox(key, counter) == expected


def test_the_table_has_the_shared_digest():
    digest = hashlib.sha256(struct.pack("<65536Q", *table())).hexdigest()
    assert digest == "090e3da3a772ee5a5427e0fc4c78d2179e98d3e8bfc40d255245fb9ab2340e85"
    assert sum(entry != 0 for entry in table()) == 43_008
    assert largest_noise(TEMPERATURE_ONE, 15) == 817_669


def test_a_wrong_table_is_not_used(monkeypatch):
    table.cache_clear()
    monkeypatch.setattr("integer_model.noise._LN2", 195_103_586_505_168)
    try:
        with pytest.raises(RuntimeError, match="does not have the expected SHA-256"):
            table()
    finally:
        # Even if table() returns the wrong table, it must not stay cached for later tests.
        monkeypatch.undo()
        table.cache_clear()


# Output of a C++ header with the same rule, which is not part of this repository, at a vocabulary
# of 202,048: seed, temp, s, position, the noise of the token ids in TOKEN_IDS, and the sum over all
# token ids.
TOKEN_IDS = (0, 1, 2, 511, 512, 65_535, 202_047)
VALUES_OF_THE_CPP_HEADER = [
    (0, 256, 15, 0, (270936, 117021, 125125, 113123, 116497, 93563, 48075), 26783318920),
    (1, 256, 15, 0, (138346, 64649, 91954, 149483, 173842, 140566, 147100), 26781350115),
    (42, 256, 15, 7, (169226, 123891, 131246, 101962, 139947, 125771, 93274), 26757244802),
    (
        (1 << 64) - 1,
        256,
        15,
        1023,
        (199980, 101728, 93781, 138701, 163733, 76036, 194152),
        26757382690,
    ),
    (
        0x0123456789ABCDEF,
        128,
        15,
        5,
        (57801, 73123, 77744, 49048, 63385, 57192, 32351),
        13378161818,
    ),
    (42, 256, 20, 7, (5415241, 3964523, 4199886, 3262806, 4478333, 4024688, 2984793), 856234990356),
    (42, 512, 12, 63, (48563, 64161, 37641, 23193, 33602, 34090, 30375), 6687654610),
]


@pytest.mark.parametrize("seed, temp, s, position, expected, total", VALUES_OF_THE_CPP_HEADER)
def test_the_noise_is_that_of_the_cpp_header(seed, temp, s, position, expected, total, device):
    row = noise(seed, [position], 202_048, temp, s, device)[0]
    assert tuple(int(row[v]) for v in TOKEN_IDS) == expected
    assert int(row.sum()) == total
    assert int(row.min()) >= 0
    assert int(row.max()) <= largest_noise(temp, s)


def test_the_noise_is_that_of_the_rule_in_plain_python(device):
    rng = random.Random(1)
    for _ in range(20):
        seed, temp, s = (
            rng.getrandbits(64),
            rng.choice((1, 128, 256, 1024)),
            rng.randrange(41),
        )
        positions = sorted(rng.sample(range(1024), 3)) + [rng.getrandbits(32), MASK_32]
        found = noise(seed, positions, 1000, temp, s, device)
        assert found.shape == (5, 1000)
        assert found.dtype == torch.int64
        for row, t in enumerate(positions):
            for v in rng.sample(range(1000), 5):
                assert int(found[row, v]) == plain_noise(seed, t, v, temp, s), (seed, t, v, temp, s)


def test_the_slots_at_the_ends_of_their_ranges(device):
    words = {0, 1, MASK_32, MASK_32 - 1}
    for bit in range(32):
        words |= {(1 << bit) + offset & MASK_32 for offset in (-2, -1, 0, 1, 2)}
        words |= {MASK_32 - (1 << bit) + offset & MASK_32 for offset in (-1, 0, 1)}
        words |= {
            (3 << bit) - 1 & MASK_32,
            3 << bit & MASK_32,
            (2047 << bit) & MASK_32,
            (2049 << bit) - 1 & MASK_32,
        }
    ordered = sorted(words)
    found = slots(torch.tensor(ordered, dtype=torch.int64, device=device))
    assert found.tolist() == [plain_slot(word) for word in ordered]
    # The table fills every slot that these words have.
    assert all(table()[slot] > 0 for slot in found.tolist())


def test_a_negative_seed_is_its_twos_complement(device):
    assert torch.equal(
        noise(-1, [3], 64, 256, 15, device), noise((1 << 64) - 1, [3], 64, 256, 15, device)
    )
    assert torch.equal(
        noise(-(1 << 63), [3], 64, 256, 15, device), noise(1 << 63, [3], 64, 256, 15, device)
    )
    assert not torch.equal(
        noise(1, [3], 64, 256, 15, device), noise(1 << 32, [3], 64, 256, 15, device)
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        ((1 << 64, [0], 8, 256, 15), "seed must be an integer that fits 64 bits"),
        ((-(1 << 63) - 1, [0], 8, 256, 15), "seed must be an integer that fits 64 bits"),
        ((1.0, [0], 8, 256, 15), "seed must be an integer that fits 64 bits"),
        ((True, [0], 8, 256, 15), "seed must be an integer that fits 64 bits"),
        ((0, [-1], 8, 256, 15), "positions must be"),
        ((0, [1 << 32], 8, 256, 15), "positions must be"),
        ((0, [[0]], 8, 256, 15), "positions must be"),
        ((0, [1.5], 8, 256, 15), "positions must be"),
        ((0, [1 << 63], 8, 256, 15), "positions must be"),
        ((0, [0], 0, 256, 15), "vocabulary_size must be at least 1"),
        ((0, [0], 8, 0, 15), "temp is 0, expected"),
        ((0, [0], 8, 1025, 15), "temp is 1025, expected"),
        ((0, [0], 8, 256, 41), "s is 41, expected 0 .. 40"),
        ((0, [0], 8, 256, -1), "s is -1, expected 0 .. 40"),
    ],
)
def test_arguments_out_of_range_are_rejected(arguments, message):
    with pytest.raises(ValueError, match=message):
        noise(*arguments)


# A changed copy of the example's parameter file


def parameter_file(directory: pathlib.Path, edit=lambda text: text) -> pathlib.Path:
    """target_small_f2's two files in a new directory, the parameter file changed by edit."""
    directory.mkdir()
    (directory / "policy.txt").write_text(edit((SMALL_EXAMPLE / "policy.txt").read_text()))
    shutil.copy(SMALL_EXAMPLE / "KL.bin", directory)
    return directory / "policy.txt"


# The choices and the gaps


class SmallHead(SyntheticWeights):
    """Synthetic weights with the output matrix divided by 256. With full 13-bit weights a random
    model's logits are millions of units apart, and noise of at most 817,669 changes the choice
    only rarely, too rarely for these tests.
    """

    def matrix(self, name, layer=None, expert=None):
        found = super().matrix(name, layer, expert)
        return found >> 8 if name == "Wh" else found


@pytest.fixture(scope="module")
def model(device) -> IntegerModel:
    """target_small_f2 with synthetic weights and a small head."""
    p = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    return IntegerModel(p, SmallHead(p, device), device)


def window(p: Parameters, number: int) -> list[int]:
    rng = random.Random(number)
    return [rng.randrange(p.V) for _ in range(p.T)]


def test_seed_none_is_no_seed(model):
    tokens, given = window(model.parameters, 1), window(model.parameters, 2)
    expected = model.extend(model.new_sequence(), tokens, gap_for=given)
    explicit = model.extend(model.new_sequence(), tokens, gap_for=given, seed=None)
    for a, b in zip(expected, explicit):
        assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 42, (1 << 64) - 1, -7])
def test_choices_and_gaps_with_a_seed_are_those_of_plain_python(model, seed):
    p = model.parameters
    tokens, given = window(p, 3), window(p, 4)
    recorded = {}
    choices, gaps = model.extend(
        model.new_sequence(),
        tokens,
        gap_for=given,
        seed=seed,
        record=lambda name, layer, tensor: recorded.update({name: tensor.clone()})
        if name == "lam"
        else None,
    )
    lam = recorded["lam"].tolist()
    moved = 0
    for t in range(p.T):
        z = [
            lam[t][v] + (plain_noise(seed, t, v, 256, p.s) if t < p.T - 1 else 0)
            for v in range(p.V)
        ]
        first = z.index(max(z))
        assert int(choices[t]) == first, t
        assert int(gaps[t]) == z[first] - z[given[t]] + (1 if first < given[t] else 0), t
        moved += first != lam[t].index(max(lam[t]))
    # The noise decides some positions and the logits others, or the test would show little.
    assert 0 < moved < p.T - 1
    assert torch.equal(model.extend(model.new_sequence(), tokens, seed=seed), choices)
    zero = model.extend(model.new_sequence(), tokens, gap_for=choices, seed=seed)[1]
    assert not zero.any()  # the seeded choice has the gap 0


def test_the_noise_is_in_the_unit_of_the_parameter_file(tmp_path, device):
    p = Parameters.read(
        parameter_file(tmp_path / "s_20", lambda text: text.replace("\ns 16\n", "\ns 20\n"))
    )
    model = IntegerModel(p, SyntheticWeights(p, device), device)
    tokens, given = window(p, 3)[:4], window(p, 4)[:4]
    recorded = {}
    choices, gaps = model.extend(
        model.new_sequence(),
        tokens,
        gap_for=given,
        seed=42,
        record=lambda name, layer, tensor: recorded.update({name: tensor.clone()})
        if name == "lam"
        else None,
    )
    for t, lam in enumerate(recorded["lam"].tolist()):
        z = [lam[v] + plain_noise(42, t, v, 256, 20) for v in range(p.V)]
        first = z.index(max(z))
        assert (int(choices[t]), int(gaps[t])) == (
            first,
            z[first] - z[given[t]] + (1 if first < given[t] else 0),
        ), t


def test_tensors_of_another_type_are_rejected():
    with pytest.raises(TypeError, match="int64"):
        philox4x32_10(0, 0, torch.zeros(1, dtype=torch.int32))
    with pytest.raises(TypeError, match="int64"):
        slots(torch.zeros(1, dtype=torch.int32))


def test_the_result_does_not_depend_on_how_the_tokens_are_divided(model, monkeypatch):
    p = model.parameters
    tokens, given = window(p, 5), window(p, 6)
    whole = model.extend(model.new_sequence(), tokens, gap_for=given, seed=9)
    for lengths in ([1] * p.T, [p.T - 1, 1], [5, p.T - 5], [p.T - 2, 2]):
        seq, parts, start = model.new_sequence(), [], 0
        for length in lengths:
            parts.append(
                model.extend(seq, tokens[start:][:length], gap_for=given[start:][:length], seed=9)
            )
            start += length
        for i in (0, 1):
            assert torch.equal(torch.cat([part[i] for part in parts]), whole[i]), lengths
    for elements in (1, 3 * p.V, 1 << 30):  # one position at a time, three, all
        monkeypatch.setattr(integer_model.model, "_NOISE_ELEMENTS", elements)
        for a, b in zip(model.extend(model.new_sequence(), tokens, gap_for=given, seed=9), whole):
            assert torch.equal(a, b), elements


def test_the_last_position_of_the_window_gets_no_noise(model):
    tokens = window(model.parameters, 7)
    expected = model.extend(model.new_sequence(), tokens)
    for seed in range(20):
        assert int(model.extend(model.new_sequence(), tokens, seed=seed)[-1]) == int(expected[-1])


def test_another_seed_gives_other_choices(model):
    tokens = window(model.parameters, 8)
    assert not torch.equal(
        model.extend(model.new_sequence(), tokens, seed=1),
        model.extend(model.new_sequence(), tokens, seed=2),
    )


def test_a_seed_that_cannot_be_used_is_rejected_before_anything_is_computed(
    model, tmp_path, device
):
    seq = model.new_sequence()
    with pytest.raises(ValueError, match="seed must be an integer that fits 64 bits"):
        model.extend(seq, [1, 2, 3], seed=1 << 64)
    assert seq.length == 0
    assert not seq.keys.any()
    p = Parameters.read(
        parameter_file(tmp_path / "s_41", lambda text: text.replace("\ns 16\n", "\ns 41\n"))
    )
    fine_unit = IntegerModel(p, SyntheticWeights(p, device), device)
    seq = fine_unit.new_sequence()
    with pytest.raises(ValueError, match="With a seed, s must be at most 40"):
        fine_unit.extend(seq, [1, 2, 3], seed=0)
    assert seq.length == 0
    assert fine_unit.extend(seq, [1, 2, 3]).shape == (3,)  # without a seed, s plays no part


def test_score_reply_takes_the_seed(tmp_path):
    """examples/score_reply.py on target_small_f2: the reply's last token is not the choice."""

    def score(directory, *options):
        (directory / "integer_model.json").write_text('{"weights": "synthetic"}')
        done = subprocess.run(
            [sys.executable, PACKAGE / "examples" / "score_reply.py", directory, "17,400,3"]
            + ["365,287,374,291", *options],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(PACKAGE)},
        )
        assert done.returncode == 1, done.stderr
        assert "Traceback" not in done.stderr, done.stderr
        return (done.stdout or done.stderr).splitlines()

    directory = parameter_file(tmp_path / "model").parent
    greedy, with_seed = score(directory), score(directory, "--seed", "42")
    assert greedy[:3] == with_seed[:3] == [f"token {token}: gap 0" for token in (365, 287, 374)]
    assert greedy[3].endswith("(greedy choice: 290)")
    assert with_seed[3].endswith("(seeded choice: 290)")
    difference = int(with_seed[3].split()[3]) - int(greedy[3].split()[3])
    assert difference == plain_noise(42, 5, 290, 256, 16) - plain_noise(42, 5, 291, 256, 16) != 0
