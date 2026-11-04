# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License.
# That project's notice:
#
# MIT License
#
# Copyright (c) 2023-2024 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests integer_model/block_weights.py against the GPU prover's own code for quantized blocks.

The tests cover the conversion rule, which turns stored blocks into the committed integers, and the
prover's synthetic model that is generated as blocks. Most expected values come from
block_reference (block_reference.cpp), compiled for a CPU from the prover's headers. Every
comparison is exact.
"""

import dataclasses
import hashlib
import math
import re
import struct
import subprocess
import sys

import numpy
import pytest
import torch

from block_tools import FULL_EXAMPLE, SMALL_EXAMPLE, TREE, load_files
from integer_model import block_weights
from integer_model.block_weights import (
    BF16,
    BLOCK_BYTES,
    EMBEDDING_LIMIT,
    F16,
    F32,
    MATRIX_LIMIT,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    BlockWeights,
    SyntheticBlocks,
    WeightsRejected,
    all_tensors,
    committed_integers,
    decoded_values,
    tensor_name,
    tensor_shape,
)
from integer_model.parameters import Parameters
from integer_model.synthetic_weights import SyntheticWeights, mix64, signed64

STORAGE_TYPES = {
    "Q4_K": Q4_K,
    "Q5_K": Q5_K,
    "Q6_K": Q6_K,
    "F32": F32,
    "Q8_0": Q8_0,
    "F16": F16,
    "BF16": BF16,
}
# The prover's numbers for the storage types.
PROVER_TYPE = {Q4_K: 0, Q5_K: 1, Q6_K: 2, F32: 3, Q8_0: 4, F16: 5, BF16: 6}
# The prover's rule (fr::fold, prover/weights_blocks/fold_rule.h) returns INT_MIN when the product
# is not finite or is 2,147,483,000 or more in magnitude.
NO_INTEGER = -(2**31)
NO_LIMIT = 2**31 - 1  # a limit above that bound, for tests that do not check the range


def prover_integers(
    programs, storage_type, blocks, exponent, multipliers=None, in_a_row=1
) -> torch.Tensor:
    """The prover's rule on blocks, shape (blocks, bytes of a block): int64, shape (blocks, 256).

    in_a_row: blocks per matrix row. multipliers: one float32 number per column.
    """
    data = b"" if multipliers is None else multipliers.cpu().numpy().tobytes()
    arguments = [
        "decode",
        PROVER_TYPE[storage_type],
        exponent,
        in_a_row,
        int(multipliers is not None),
    ]
    answer = subprocess.run(
        [programs / "block_reference", *map(str, arguments)],
        input=data + blocks.cpu().numpy().tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    return torch.from_numpy(numpy.frombuffer(answer, dtype="<i4").astype(numpy.int64)).view(-1, 256)


def module_integers(
    device, storage_type, blocks, exponent, multipliers=None, in_a_row=1, limit=NO_LIMIT
) -> torch.Tensor:
    """The rule of block_weights.py on the same blocks, in the same shape."""
    values = decoded_values(blocks.to(device), storage_type).view(-1, 256 * in_a_row)
    gamma = None if multipliers is None else multipliers.to(device)
    return committed_integers(values, gamma, exponent, limit).to(torch.int64).cpu().view(-1, 256)


# Blocks written byte by byte, from the layouts in
# docs/model.md, "11.2 Stored blocks and their values", which are ggml's


def pack_scales(scales, minima) -> bytes:
    """Eight 6-bit scales and eight 6-bit minima in the twelve bytes of a Q4_K or Q5_K block."""
    packed = bytearray(12)
    for j in range(4):
        packed[j] = scales[j] | (scales[j + 4] >> 4) << 6
        packed[j + 4] = minima[j] | (minima[j + 4] >> 4) << 6
        packed[j + 8] = (scales[j + 4] & 15) | (minima[j + 4] & 15) << 4
    return bytes(packed)


def pack_q4_k_or_q5_k(storage_type, h, h_min, scales, minima, codes) -> bytes:
    """h and h_min are the bits of the block's two binary16 numbers. codes has 256 entries."""
    low, fifth = bytearray(128), bytearray(32)
    for element, code in enumerate(codes):
        j, i = divmod(element, 32)
        low[32 * (j // 2) + i] |= (code & 15) << (4 * (j % 2))
        fifth[i] |= (code >> 4) << j
    middle = bytes(fifth) if storage_type == Q5_K else b""
    return struct.pack("<HH", h, h_min) + pack_scales(scales, minima) + middle + bytes(low)


def pack_q6_k(h, scales, codes) -> bytes:
    """scales has 16 entries from -128 to 127, codes has 256 entries from -32 to 31."""
    low, high = bytearray(128), bytearray(64)
    for element, code in enumerate(codes):
        a, rest = divmod(element, 128)
        t, i = divmod(rest, 32)
        low[64 * a + 32 * (t % 2) + i] |= ((code + 32) & 15) << (4 * (t // 2))
        high[32 * a + i] |= ((code + 32) >> 4) << (2 * t)
    return bytes(low) + bytes(high) + struct.pack("<16b", *scales) + struct.pack("<H", h)


def pack_q8_0(scales, codes) -> bytes:
    """A block of 256 elements: eight of the format's blocks, each the bits of a binary16 number and
    32 signed codes.

    The layout is that of the package gguf, version 0.19.0 (gguf/quants.py, the class Q8_0).
    """
    return b"".join(struct.pack("<H32b", scales[j], *codes[32 * j : 32 * j + 32]) for j in range(8))


def block_of_16_bits(patterns) -> bytes:
    """A block of type F16 or BF16 that starts with the given bit patterns and is padded with
    zeros."""
    return struct.pack("<256H", *patterns, *[0] * (256 - len(patterns)))


def as_blocks(*blocks: bytes) -> torch.Tensor:
    return torch.from_numpy(numpy.frombuffer(b"".join(blocks), dtype=numpy.uint8).copy()).view(
        len(blocks), -1
    )


def f32_block(values) -> bytes:
    """A block of type F32: the given values, then zeros."""
    return numpy.array(list(values) + [0.0] * (256 - len(values)), dtype="<f4").tobytes()


# Bits of binary16 numbers: zero, minus zero, the smallest and the largest subnormal number, the
# smallest normal number, 1, a number with a full significand, its negative, and the largest finite
# number with both signs.
EDGE_BINARY16 = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x2FFF, 0xAFFF, 0x7BFF, 0xFBFF]


def random_blocks(storage_type, count, seed) -> torch.Tensor:
    """Blocks of random bytes, but with every binary16 number below 2 and every binary32 or BF16
    number below 2^11 in magnitude.

    The integers then stay below 2^31 for exponents up to 12. The edge-value tests cover larger
    binary16 numbers.
    """
    random = numpy.random.default_rng(seed)
    blocks = random.integers(0, 256, (count, BLOCK_BYTES[storage_type]), dtype=numpy.uint8)
    if storage_type == F32:
        words = blocks.view("<u4")
        words[(words >> 23) & 255 >= 138] &= 0x807FFFFF  # these become subnormal numbers or zero
    elif storage_type == BF16:
        words = blocks.view("<u2")
        words[(words >> 7) & 255 >= 138] &= 0x807F  # these become subnormal numbers or zero
    elif storage_type in (Q8_0, F16):
        blocks[:, 1 :: 34 if storage_type == Q8_0 else 2] &= 0xBF
    else:
        for high_byte in (209,) if storage_type == Q6_K else (1, 3):
            blocks[:, high_byte] &= 0xBF
    return torch.from_numpy(blocks)


# The conversion rule


@pytest.mark.parametrize("type_name", ["Q4_K", "Q5_K", "Q6_K", "F32", "Q8_0", "F16", "BF16"])
@pytest.mark.parametrize("with_multipliers", [False, True])
def test_random_blocks_decode_as_in_the_prover(programs, device, type_name, with_multipliers):
    storage_type = STORAGE_TYPES[type_name]
    blocks = random_blocks(storage_type, 4096, seed=storage_type)
    multipliers = None
    if with_multipliers:
        # The bits of float32 numbers in [0.5, 2).
        bits = numpy.random.default_rng(5).integers(0x3F000000, 0x40000000, 512, dtype=numpy.int32)
        multipliers = torch.from_numpy(bits).view(torch.float32)
    for exponent in (0, 12):
        expected = prover_integers(
            programs, storage_type, blocks, exponent, multipliers, in_a_row=2
        )
        assert NO_INTEGER not in expected
        assert torch.equal(
            module_integers(device, storage_type, blocks, exponent, multipliers, in_a_row=2),
            expected,
        )


def test_the_random_blocks_reach_a_subtraction_that_rounds(device):
    """The subtraction is the only step of the decoding that rounds. The random blocks of the test
    above must contain many cases in which it does.
    """
    for storage_type in (Q4_K, Q5_K):
        blocks = random_blocks(storage_type, 256, seed=storage_type)
        values = decoded_values(blocks.to(device), storage_type).cpu().to(torch.float64)
        exact = exact_values(storage_type, blocks)
        assert int((values != exact).sum()) > 1000
        # One rounding of the exact difference.
        assert torch.equal(values, exact.to(torch.float32).to(torch.float64))


def exact_values(storage_type, blocks) -> torch.Tensor:
    """h * scale * code - h_min * minimum without any rounding: float64 represents every term and
    the difference exactly.
    """
    values = []
    for block in blocks.numpy():
        h, h_min = (float(x) for x in block[0:4].view("<f2"))
        b = block[4:16].astype(int)
        start = 16 if storage_type == Q4_K else 48
        for element in range(256):
            j, i = divmod(element, 32)
            if j < 4:
                scale, minimum = b[j] % 64, b[j + 4] % 64
            else:
                scale, minimum = (
                    b[j + 4] % 16 + 16 * (b[j - 4] // 64),
                    b[j + 4] // 16 + 16 * (b[j] // 64),
                )
            code = int(block[start + 32 * (j // 2) + i]) >> (4 * (j % 2)) & 15
            if storage_type == Q5_K:
                code += 16 * (int(block[16 + i]) >> j & 1)
            values.append(h * scale * code - h_min * minimum)
    return torch.tensor(values, dtype=torch.float64).view(-1, 256)


@pytest.mark.parametrize("type_name", ["Q4_K", "Q5_K"])
def test_edge_values_of_scales_and_minima(programs, device, type_name):
    storage_type = STORAGE_TYPES[type_name]
    top = 15 if storage_type == Q4_K else 31
    codes = [(0, top, 1, top - 1)[element % 4] for element in range(256)]
    blocks = []
    for h in EDGE_BINARY16:
        for h_min in EDGE_BINARY16:
            for scales, minima in (
                ([0] * 8, [0] * 8),
                ([63] * 8, [63] * 8),
                ([0, 63] * 4, [63, 0] * 4),
                ([1, 32] * 4, [31, 2] * 4),
            ):
                blocks.append(pack_q4_k_or_q5_k(storage_type, h, h_min, scales, minima, codes))
    blocks = as_blocks(*blocks)
    for exponent in (0, 3):
        expected = prover_integers(programs, storage_type, blocks, exponent)
        assert NO_INTEGER not in expected
        assert torch.equal(module_integers(device, storage_type, blocks, exponent), expected)


def test_edge_values_of_q6_k(programs, device):
    codes = [(-32, 31, 0, -1)[element % 4] for element in range(256)]
    blocks = as_blocks(
        *(
            pack_q6_k(h, scales, codes)
            for h in EDGE_BINARY16
            for scales in ([-128] * 16, [127] * 16, [0, 1, -1, 64] * 4)
        )
    )
    for exponent in (0, 2):
        expected = prover_integers(programs, Q6_K, blocks, exponent)
        assert NO_INTEGER not in expected
        assert torch.equal(module_integers(device, Q6_K, blocks, exponent), expected)


def every_finite_block(storage_type) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocks of a wider type that hold everything finite, and each block's largest magnitude as
    float64.

    For F16 and BF16 these are all finite bit patterns, sorted by magnitude, so that the numbers of
    a block have about the same size. For Q8_0 there is one block for each finite scale, with the
    codes -128 to 127.
    """
    patterns = numpy.arange(2**16, dtype="<u2")
    patterns = patterns[(patterns & 0x7FFF) < (0x7F80 if storage_type == BF16 else 0x7C00)]
    if storage_type == BF16:
        sizes = numpy.abs((patterns.astype("<u4") << 16).view("<f4").astype(numpy.float64))
    else:
        sizes = numpy.abs(patterns.view("<f2").astype(numpy.float64))
    if storage_type == Q8_0:
        blocks = numpy.zeros((len(patterns), 8, 34), dtype=numpy.uint8)
        blocks[:, :, 0:2] = patterns.view(numpy.uint8).reshape(-1, 1, 2)
        codes = numpy.arange(-128, 128).astype(numpy.int8).view(numpy.uint8)
        blocks[:, :, 2:34] = codes.reshape(1, 8, 32)
        return torch.from_numpy(blocks.reshape(len(patterns), -1)), torch.from_numpy(128 * sizes)
    order = numpy.argsort(sizes, kind="stable")
    blocks = patterns[order].view(numpy.uint8).reshape(-1, 512)
    largest = sizes[order].reshape(-1, 256).max(axis=1)
    return torch.from_numpy(blocks.copy()), torch.from_numpy(largest)


@pytest.mark.parametrize("type_name", ["Q8_0", "F16", "BF16"])
def test_everything_finite_of_a_wider_type_decodes_as_in_the_prover(programs, device, type_name):
    """The prover's rule gives integers only, so one scaling shows only the values of some sizes.
    The test goes through the scalings 2^-64 to 2^104 that BlockWeights admits (multipliers from
    2^-64 to 2^64, exponents from 0 to 40) in steps of 2^12. At each it takes the blocks whose
    integers stay below 2^30. A value has at most 18 significant bits, so an integer of 2^18 or more
    shows all of them. The last line checks that each block whose largest magnitude is from 2^-86 to
    just below 2^94 reached such an integer at some scaling. The smaller values of BF16 give small
    integers or 0, among them those that are subnormal in binary32. No admitted scaling brings the
    larger ones into range.
    """
    storage_type = STORAGE_TYPES[type_name]
    blocks, largest = every_finite_block(storage_type)
    shown = torch.zeros(len(blocks), dtype=torch.bool)
    for scaling in range(-64, 105, 12):
        exponent = min(max(scaling, 0), 40)
        multipliers = torch.full((256,), 2.0 ** (scaling - exponent))
        some = largest * 2.0**scaling < 2.0**30
        expected = prover_integers(programs, storage_type, blocks[some], exponent, multipliers)
        assert NO_INTEGER not in expected
        found = module_integers(device, storage_type, blocks[some], exponent, multipliers)
        assert torch.equal(found, expected)
        shown[some] |= expected.abs().amax(dim=1) >= 2**18
    assert torch.equal(shown, (largest >= 2.0**-86) & (largest < 2.0**94))


@pytest.fixture
def gguf_package(monkeypatch):
    """The package gguf, imported without the two tokenizer packages that it imports if they are
    installed (gguf/vocab.py) and does without if not; the decoder needs neither.

    One of them, sentencepiece, makes Python 3.12 and later print a deprecation warning when it
    exits, after pytest's summary, which run_suite.py expects on the last line. What was imported so
    is forgotten after the test, so that whatever imports gguf later in the run gets it whole.
    """
    first = "gguf" not in sys.modules
    for name in ("sentencepiece", "mistral_common"):
        monkeypatch.setitem(sys.modules, name, None)
    try:
        import gguf
    except ImportError:  # a skip would go unnoticed
        pytest.fail("This test needs the package gguf 0.19.0, which tests/requirements.txt pins.")
    yield gguf
    if first:
        for name in [name for name in sys.modules if name.partition(".")[0] == "gguf"]:
            del sys.modules[name]


@pytest.mark.parametrize("type_name", ["Q8_0", "F16", "BF16"])
def test_everything_finite_of_a_wider_type_decodes_as_in_the_package_gguf(
    device, type_name, gguf_package
):
    """Bit for bit, against the decoder of the format's own Python package (gguf/quants.py,
    dequantize)."""
    gguf = gguf_package
    storage_type = STORAGE_TYPES[type_name]
    blocks, _ = every_finite_block(storage_type)
    expected = gguf.quants.dequantize(blocks.numpy(), gguf.GGMLQuantizationType(storage_type))
    found = decoded_values(blocks.to(device), storage_type)
    assert found.dtype == torch.float32 and found.shape == expected.shape == (len(blocks), 256)
    assert numpy.array_equal(found.cpu().numpy().view(numpy.uint32), expected.view(numpy.uint32))


def test_a_packed_block_decodes_to_what_was_packed(device):
    """All scales and minima differ, and the codes change from element to element and from sub-block
    to sub-block, so that a field read from the wrong position changes the result.
    """
    one = 0x3C00  # the binary16 number 1
    scales, minima = [1, 2, 3, 5, 17, 33, 62, 63], [63, 40, 21, 16, 15, 3, 1, 0]
    for storage_type, top in ((Q4_K, 16), (Q5_K, 32)):
        codes = [(7 * element + element // 32) % top for element in range(256)]
        block = as_blocks(pack_q4_k_or_q5_k(storage_type, one, one, scales, minima, codes))
        expected = [scales[e // 32] * codes[e] - minima[e // 32] for e in range(256)]
        assert decoded_values(block.to(device), storage_type).cpu().tolist() == [expected]
    codes = [(11 * element + element // 16) % 64 - 32 for element in range(256)]
    scales = [-128, -77, -3, -1, 0, 1, 2, 5, 9, 17, 33, 65, 99, 126, 127, 64]
    block = as_blocks(pack_q6_k(one, scales, codes))
    assert decoded_values(block.to(device), Q6_K).cpu().tolist() == [
        [scales[e // 16] * codes[e] for e in range(256)]
    ]
    codes = [(37 * element + element // 32) % 256 - 128 for element in range(256)]
    scales = [0.5, 1.0, 1.5, 2.0, -1.0, 3.0, 0.25, 5.0]
    bits = [int(numpy.float16(scale).view(numpy.uint16)) for scale in scales]
    block = as_blocks(pack_q8_0(bits, codes))
    assert decoded_values(block.to(device), Q8_0).cpu().tolist() == [
        [scales[e // 32] * codes[e] for e in range(256)]
    ]
    # 1, 1 + 2^-10, -2, 2^-24 as binary16 numbers, and 1, 1 + 2^-7, -2, 2^-133 as BF16.
    block = as_blocks(block_of_16_bits([0x3C00, 0x3C01, 0xC000, 0x0001]))
    expected = [1.0, 1.0 + 2.0**-10, -2.0, 2.0**-24, 0.0]
    assert decoded_values(block.to(device), F16)[0, :5].tolist() == expected
    block = as_blocks(block_of_16_bits([0x3F80, 0x3F81, 0xC000, 0x0001]))
    expected = [1.0, 1.0 + 2.0**-7, -2.0, 2.0**-133, 0.0]
    assert decoded_values(block.to(device), BF16)[0, :5].cpu().tolist() == expected


def test_ties_go_to_the_even_integer(programs, device):
    halves = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4093.5, 4094.5, -4094.5]
    halves += [0.49999997, 0.50000006, 1.4999999, 2.5000002]
    block = as_blocks(f32_block(halves))
    integers = module_integers(device, F32, block, 0, limit=MATRIX_LIMIT)
    assert integers[0, : len(halves)].tolist() == [
        0,
        2,
        2,
        4,
        0,
        -2,
        -2,
        4094,
        4094,
        -4094,
        0,
        1,
        1,
        3,
    ]
    assert torch.equal(integers, prover_integers(programs, F32, block, 0))
    # Ties after a scaling by 2^12, then after a product with 0.5.
    block = as_blocks(f32_block([x / 4096 for x in halves]))
    assert torch.equal(module_integers(device, F32, block, 12), integers)
    codes = list(range(16)) * 16
    block = as_blocks(pack_q4_k_or_q5_k(Q4_K, 0x3C00, 0, [1] * 8, [0] * 8, codes))
    half = torch.full((256,), 0.5)
    integers = module_integers(device, Q4_K, block, 0, half)
    assert integers[0, :16].tolist() == [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8]
    assert torch.equal(integers, prover_integers(programs, Q4_K, block, 0, half))


def test_tiny_products_give_zero_whether_the_scaling_comes_first_or_last(programs, device):
    """fr::fold scales the multiplier by 2^exponent before the product; block_weights.py scales the
    product.
    """
    tiny = [
        2.0**-149,
        2.0**-140,
        -(2.0**-127),
        2.0**-126,
        2.0**-100,
        2.0**-41,
        3 * 2.0**-41,
        2.0**-40,
        0.0,
        -0.0,
    ]
    block = as_blocks(f32_block(tiny))
    for magnitude in (2.0**-64, 2.0**-20, 1.0):
        multipliers = torch.full((256,), magnitude)
        expected = prover_integers(programs, F32, block, 40, multipliers)
        assert torch.equal(module_integers(device, F32, block, 40, multipliers), expected)
    assert expected[0, : len(tiny)].tolist() == [0, 0, 0, 0, 0, 0, 2, 1, 0, 0]


def test_zero_multipliers_give_zero(programs, device):
    blocks = random_blocks(Q4_K, 16, seed=1)
    zero = torch.zeros(256)
    assert not module_integers(device, Q4_K, blocks, 12, zero).any()
    assert not prover_integers(programs, Q4_K, blocks, 12, zero).any()


def test_a_value_that_is_not_finite_is_rejected(programs, device):
    for storage_type, block in (
        # h is infinite.
        (Q4_K, pack_q4_k_or_q5_k(Q4_K, 0x7C00, 0, [1] * 8, [0] * 8, [1] * 256)),
        # h_min is NaN.
        (Q4_K, pack_q4_k_or_q5_k(Q4_K, 0x3C00, 0x7E00, [1] * 8, [1] * 8, [1] * 256)),
        # h is minus infinity.
        (Q5_K, pack_q4_k_or_q5_k(Q5_K, 0xFC00, 0, [1] * 8, [0] * 8, [1] * 256)),
        (Q6_K, pack_q6_k(0x7C00, [1] * 16, [1] * 256)),
        (Q6_K, pack_q6_k(0x7C00, [0] * 16, [1] * 256)),  # infinity times zero
        (F32, f32_block([float("nan")])),
        (F32, f32_block([float("-inf")])),
        (F32, f32_block([3.0e38])),  # the scaling overflows
        (Q8_0, pack_q8_0([0x3C00] * 7 + [0x7C00], [1] * 256)),  # the last h is infinite
        (Q8_0, pack_q8_0([0xFC00] + [0x3C00] * 7, [0] * 256)),  # infinity times zero
        (Q8_0, pack_q8_0([0x7E00] + [0x3C00] * 7, [1] * 256)),  # h is not a number
        (F16, block_of_16_bits([0x3C00, 0x7C00])),  # infinity
        (F16, block_of_16_bits([0x3C00, 0xFC00])),  # minus infinity
        (F16, block_of_16_bits([0x3C00, 0x7E00])),  # not a number
        (BF16, block_of_16_bits([0x3F80, 0x7F80])),  # infinity
        (BF16, block_of_16_bits([0x3F80, 0xFF80])),  # minus infinity
        (BF16, block_of_16_bits([0x3F80, 0x7FC0])),  # not a number
        # the largest finite number: the scaling overflows
        (BF16, block_of_16_bits([0x3F80, 0x7F7F])),
    ):
        blocks = as_blocks(block)
        assert NO_INTEGER in prover_integers(programs, storage_type, blocks, 12)
        with pytest.raises(WeightsRejected, match="A weight is not finite, or its absolute value"):
            module_integers(device, storage_type, blocks, 12)


def test_out_of_range_weights_are_rejected(device):
    """Tries each limit's smallest rejected value and accepted values near it. The embedding's
    limit, 2^30 - 1, is not a float32 number.
    """
    for limit, accepted, rejected in (
        (MATRIX_LIMIT, [4095.0, -4095.0, 4095.4], [4096.0, -4096.0, 4095.5]),
        (EMBEDDING_LIMIT, [2.0**30 - 64, 64 - 2.0**30], [2.0**30, -(2.0**30)]),
    ):
        assert module_integers(
            device, F32, as_blocks(f32_block(accepted)), 0, limit=limit
        ).max() == int(accepted[0])
        for value in rejected:
            with pytest.raises(WeightsRejected, match=f"absolute value exceeds {limit:,}\\."):
                module_integers(device, F32, as_blocks(f32_block([value])), 0, limit=limit)


def test_the_table_of_tensors_is_that_of_the_model_definition():
    """block_weights.py's table of tensors equals the table in
    docs/model.md, "11.3 Which tensor is which matrix": each tensor's matrix, the norm whose gains
    are multiplied into it, and which constant c_0 multiplies the gains first.

    The other tests' digests are listed by tensor name, so they cannot show that a tensor becomes
    the right matrix.
    """
    page = TREE / "docs" / "model.md"
    if not page.exists():
        pytest.skip("docs/model.md is not in this checkout")
    heading = "### 11.3 Which tensor is which matrix\n"
    text = page.read_text()
    assert heading in text, (
        f"docs/model.md has no heading {heading.strip()!r}; this test reads the table under it."
    )
    stated = {}
    for line in text.split(heading)[1].split("\n#")[0].splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4 or not cells[0].startswith("`"):
            continue
        tensors = [
            t.removeprefix("blk.l.").removesuffix(".weight")
            for t in re.findall("`([^`]+)`", cells[0])
        ]
        norms = [n.removeprefix("blk.l.") for n in re.findall("`([^`]+)`", cells[2])]
        for tensor, matrix in zip(tensors, re.findall("W[a-z]+|Emb", cells[1]), strict=True):
            stated[tensor] = (matrix, norms[0] if norms else None, "sqrt(dh)" in cells[3])
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    found = {}
    for name, layer in all_tensors(parameters):
        kind = block_weights._KINDS[name]
        tensor = re.sub(r"^blk\.[0-9]+\.", "", tensor_name(parameters, name, layer)).removesuffix(
            ".weight"
        )
        found[tensor] = (name, kind.norm, kind.constant == block_weights._C0_QUERY)
    assert found == stated
    assert len(stated) == 16
    for constant, value in (
        (block_weights._C0, math.sqrt(5120) / 64),
        (block_weights._C0_QUERY, math.sqrt(40) / 64),
    ):
        assert constant == int(numpy.float32(value).view(numpy.uint32))


# The synthetic model in blocks


def test_mix64_has_the_known_first_outputs_of_splitmix64(device):
    """SplitMix64 (splitmix64.c by Sebastiano Vigna; NOTICE) gives 0xE220A8397B1DCDAF first from the
    state 0. The second expected value is its second output: the state advances by
    0x9E3779B97F4A7C15.
    """
    states = torch.tensor([0, signed64(0x9E3779B97F4A7C15)], dtype=torch.int64, device=device)
    outputs = [word % 2**64 for word in mix64(states).tolist()]
    assert outputs == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def prover_digests(programs, *arguments) -> list[str]:
    """What "block_reference synthetic" prints: type, SHA-256 of bytes, SHA-256 of integers."""
    command = [programs / "block_reference", "synthetic", *map(str, arguments)]
    return subprocess.run(command, capture_output=True, check=True, text=True).stdout.split()


def test_synthetic_blocks_equal_those_of_the_prover(programs, device, synthetic_weights):
    """Compares the type, the bytes and the integers of every tensor of type Q4_K or Q5_K, at the
    shape of prover/examples/target_small_f2.
    """
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    source = SyntheticBlocks(parameters, device)
    types = set()
    for name, layer in all_tensors(parameters):
        if source.storage_type(name, layer) == F32:
            continue
        experts, outputs, inputs = tensor_shape(parameters, name, layer)
        stream = block_weights._KINDS[name].stream
        prover_type, of_bytes, of_integers = prover_digests(
            programs, parameters.seed, stream, layer, 0, experts, outputs, inputs // 256
        )
        assert PROVER_TYPE[source.storage_type(name, layer)] == int(prover_type)
        assert (
            hashlib.sha256(
                source.blocks(name, layer, range(experts * outputs)).cpu().numpy()
            ).hexdigest()
            == of_bytes
        )
        assert synthetic_weights.digests[tensor_name(parameters, name, layer)] == (
            experts * outputs * inputs,
            of_integers,
        )
        types.add((name in ("Wgr", "Wur", "Wdr"), source.storage_type(name, layer)))
    assert types == {(False, Q4_K), (False, Q5_K), (True, Q4_K), (True, Q5_K)}


@pytest.mark.parametrize(
    "name, layer, expert",
    [("Wgr", 47, 127), ("Wdr", 1, 64), ("Wdr", 3, 0), ("Wd", 46, 0), ("Wq", 3, 0)],
)
def test_synthetic_blocks_at_the_shape_of_the_full_size_model(
    programs, device, name, layer, expert
):
    """One expert or one matrix at full size (400 billion parameters), including the highest layer
    and expert numbers.
    """
    parameters = Parameters.read(FULL_EXAMPLE / "policy.txt")
    source = SyntheticBlocks(parameters, device)
    _, outputs, inputs = tensor_shape(parameters, name, layer)
    stream = block_weights._KINDS[name].stream
    _, of_bytes, of_integers = prover_digests(
        programs, parameters.seed, stream, layer, expert, 1, outputs, inputs // 256
    )
    blocks = source.blocks(name, layer, range(expert * outputs, (expert + 1) * outputs))
    assert hashlib.sha256(blocks.cpu().numpy()).hexdigest() == of_bytes
    digest = hashlib.sha256()
    for first in range(0, outputs, 1024):
        some = blocks[first * (inputs // 256) : (first + 1024) * (inputs // 256)]
        values = decoded_values(some, source.storage_type(name, layer)).view(-1, inputs)
        integers = committed_integers(values, source.multipliers(name, layer), 12, MATRIX_LIMIT)
        digest.update(integers.to(torch.int16).cpu().numpy())
    assert digest.hexdigest() == of_integers


def test_router_embedding_and_head_equal_those_of_the_generator(device, synthetic_weights):
    """The prover gives these three the integers of its other synthetic model, which
    SyntheticWeights generates. This test compares two classes of this package;
    test_against_the_prover.py compares BlockWeights with the GPU prover only through the
    prover's answers to a few requests.
    """
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    generator = SyntheticWeights(parameters, device)
    assert torch.equal(synthetic_weights.matrix("Wh"), generator.matrix("Wh"))
    for layer in range(1, parameters.L, 2):
        assert torch.equal(synthetic_weights.matrix("Wr", layer), generator.matrix("Wr", layer))
    token_ids = torch.tensor([0, 511, 3, 3, 200], device=device)
    assert torch.equal(
        synthetic_weights.embedding_rows(token_ids), generator.embedding_rows(token_ids)
    )


def test_the_default_dtype_of_pytorch_changes_nothing(file_model, device, synthetic_weights):
    """vLLM loads a model while PyTorch's default floating-point type is the served model's, for
    example bfloat16.
    """
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        again = BlockWeights(
            Parameters.read(SMALL_EXAMPLE / "policy.txt"),
            device,
            blocks="synthetic",
            hash_integers=True,
        )
        load_files(file_model, device, verify_sums=True)
    finally:
        torch.set_default_dtype(before)
    assert again.digests == synthetic_weights.digests


def test_rows_that_are_not_whole_blocks_are_rejected(device):
    parameters = dataclasses.replace(Parameters.read(SMALL_EXAMPLE / "policy.txt"), f=256, fe=128)
    with pytest.raises(
        WeightsRejected,
        match="blk.1.ffn_down_shexp.weight: the number of inputs, 128, is not a multiple",
    ):
        BlockWeights(parameters, device, blocks="synthetic")


def test_decoding_in_small_steps_changes_nothing(device, synthetic_weights, monkeypatch):
    monkeypatch.setattr(BlockWeights, "WEIGHTS_AT_ONCE", 37 * 512)
    again = BlockWeights(
        Parameters.read(SMALL_EXAMPLE / "policy.txt"),
        device,
        blocks="synthetic",
        hash_integers=True,
    )
    assert again.digests == synthetic_weights.digests
