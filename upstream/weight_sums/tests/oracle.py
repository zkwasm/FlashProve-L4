# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Parts of this file are adapted from ggml (llama.cpp), which is under the MIT License. That project's notice, which
# applies to those parts:
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

# Started by weight_sums/tests/run_tests.sh, which the runner starts; by hand: below a directory of your own.
"""Write two synthetic GGUF shards and the sums file that the conversion rule prescribes for them.

usage: python3 oracle.py <seed> <eres> <new folder>

A shard is one of the GGUF files of a model. The conversion rule turns every stored weight of the
tensors that the prover commits to into one integer. The sums file (`wsums.txt` in
weight_sums/README.md) has one line for each of these tensors, with the SHA-256 of its integers.
docs/model.md, "11 From the model's files to the integers", defines the rule, and
weight_sums/README.md, "Which models, and the rule", summarizes it. docs/model.md, 11.4, also states
three of the rule's rejections: a value that is not finite, a multiplier (a normalization weight
times its constant) that is not 0 but is below 2^-100 in absolute value, and a matrix's exponent
outside -100 .. 100. The table of exit statuses in weight_sums/README.md, "Make the file", has
them too. Two programs compute a sums file from shards: bin/weight_sums and weight_sums/weight_sums.py.
weight_sums/tests/run_tests.sh runs each on this script's shards and compares its file with
expected.txt. The script makes the folder and writes three files into it:

  oracle-00001-of-00002.gguf, oracle-00002-of-00002.gguf   two shards of a synthetic model
  expected.txt                                             the sums file of both shards together

<seed> is a decimal integer from 0 to 2^64 - 1, and the shards depend on it only. <eres> is e_res,
the exponent of the embedding: the value from -100 to 100 that the two programs take as --eres.
Every byte of the three files is a function of <seed> and <eres>. The exit status is 0 if the three
files are written, and 2 for a wrong command line or a folder that exists or cannot be made. Every
<eres> up to 20 is valid. At a larger one the rule can reject the embedding. The script then writes
only the shards, and its exit status is 25.

The expected sums come from a further implementation of the rule, which differs from the two
programs in its arithmetic. Every number is an exact rational (fractions.Fraction), the two
roundings of the rule are functions on exact numbers, and no floating-point operation takes part.
The script imports neither program, nor numpy or gguf-py, so it needs only Python's standard
library. But all three implement the same written definition of the rule, and like the programs'
decoders its block decoder goes back to ggml's public code (see the last paragraph, and
../../THIRD_PARTY.md for the programs). So an error in that definition or in ggml's code, or a
reading of either that all three share, does not show when their sums are compared.

weight_sums/tests/refusals.py imports write_shard(), sums_file() and Rejected.

The decoder of the block formats Q4_K, Q5_K and Q6_K follows ggml's public code and is adapted from
it, and the writer follows ggml's description of the GGUF container. The files are
ggml/src/ggml-common.h, ggml/src/ggml-quants.c and ggml/include/gguf.h of llama.cpp
(github.com/ggml-org/llama.cpp; MIT licence) as carried in llama-cpp-python 0.3.7, whose changelog
names the commit 794fe23f29fb40104975c91fe19f23798f7c726e. Two habits of the written files, the
padding after every tensor and the three keys of a split model, are those of the writer in gguf-py
0.19.0 (gguf/gguf_writer.py; MIT licence). ../../THIRD_PARTY.md has the full citations and the text
of the licences.
"""

import collections
import contextlib
import hashlib
import math
import os
import re
import struct
import sys
from fractions import Fraction

# ---------------------------------------------------------------------------------------------------------------------
# The GGUF container
#
# GGUF is the file format of the ggml and llama.cpp projects. The layout is the one that the comment
# at the top of ggml/include/gguf.h describes, in the copy of llama.cpp that the docstring above
# names. Its parts are the magic, the version, the number of tensors, the number of metadata pairs,
# the pairs, the tensor table, padding and the tensors' data. All numbers are little-endian. A
# string is a 64-bit length followed by that many bytes.

# The value types of metadata (enum gguf_type in ggml/include/gguf.h).
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
SCALAR_FORMAT = {UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i", FLOAT32: "<f",
                 BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d"}  # for struct.pack

# The storage types of tensors that the conversion rule reads (enum ggml_type in
# ggml/include/ggml.h), and for each the number of elements and of bytes in one block (the
# structures block_q4_K, block_q5_K, block_q6_K in ggml/src/ggml-common.h). The numbers and sizes of
# F16, Q8_0 and BF16 are those of the package gguf, version 0.19.0 (gguf/constants.py:
# GGMLQuantizationType and GGML_QUANT_SIZES).
F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, BF16 = 0, 1, 8, 12, 13, 14, 30
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210), BF16: (1, 2)}


def as_bytes(text):
    """Return a key, a name or a string value as bytes. A str is encoded as UTF-8."""
    return text.encode() if isinstance(text, str) else bytes(text)


def gguf_string(text):
    return struct.pack("<Q", len(as_bytes(text))) + as_bytes(text)


def gguf_value(value_type, value):
    """Return the bytes of one metadata value.

    The value of an ARRAY is a pair (value type of the elements, list of elements). For a value type
    number that GGUF does not define, the value must be bytes, and they are written as they are.
    """
    if value_type == STRING:
        return gguf_string(value)
    if value_type == ARRAY:
        element_type, elements = value
        return struct.pack("<IQ", element_type, len(elements)) + b"".join(gguf_value(element_type, x) for x in elements)
    if value_type in SCALAR_FORMAT:
        return struct.pack(SCALAR_FORMAT[value_type], value)
    return bytes(value)


# What write_shard() returns. Every number is a position in bytes from the start of the file, and
# every list is in the order of the arguments:
#   pairs       for each metadata pair, the positions of the key's length, of the value type and of
#               the value
#   table       for each tensor, the positions of the name's length, of the number of dimensions, of
#               the first dimension, of the storage type and of the offset
#   data_start  the position of the data section
#   data        for each tensor, the position of its first data byte
#   size        the length of the file
Layout = collections.namedtuple("Layout", "pairs table data_start data size")


def write_shard(path, metadata, tensors, alignment=32, version=3):
    """Write one GGUF file, which must not exist yet, and return its Layout.

    A test uses the Layout to overwrite fields of the file.

    metadata   a list of (key, value type number, value), written in this order. See gguf_value()
               for arrays.
    tensors    a list of (name, dims, storage type number, data bytes), written in this order.
    alignment  the data section and every tensor in it start at a multiple of this number, and zero
               bytes fill the gaps, also after the last tensor, as in the files that gguf-py's
               writer makes. 0 counts as 1.
    version    the version number in the header.

    The function adds nothing to its arguments and does not check them. In particular it writes the
    key general.alignment only if the metadata contain it, and it does not compare the length of the
    data with the dimensions. So a test can write a file whose parts contradict each other. A key, a
    name or a string value is a str or bytes. A BOOL is written as the byte 0 or 1.
    """
    def padding(length):
        return bytes(-length % max(alignment, 1))

    head = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)))
    pairs, table, offsets = [], [], []
    for key, value_type, value in metadata:
        positions = [len(head)]
        for field in (gguf_string(key), struct.pack("<I", value_type), gguf_value(value_type, value)):
            head += field
            positions.append(len(head))
        pairs.append(tuple(positions[:3]))
    offset = 0  # counts from the start of the data section
    for name, dims, storage_type, data in tensors:
        positions = [len(head)]
        for field in (gguf_string(name), struct.pack("<I", len(dims)), struct.pack("<%dQ" % len(dims), *dims),
                      struct.pack("<I", storage_type), struct.pack("<Q", offset)):
            head += field
            positions.append(len(head))
        table.append(tuple(positions[:5]))
        offsets.append(offset)
        offset += len(data) + len(padding(len(data)))
    head += padding(len(head))
    with open(path, "xb") as file:
        file.write(head)
        for _, _, _, data in tensors:
            file.write(data + padding(len(data)))
    return Layout(pairs, table, len(head), [len(head) + at for at in offsets], len(head) + offset)


# ---------------------------------------------------------------------------------------------------------------------
# Exact numbers
#
# Every finite binary16 or binary32 value is a rational number, and this part works on such numbers
# as Fractions. The two formats are binary interchange formats of IEEE 754, and the rounding is its
# roundTiesToEven, with its rule for overflow. Zero has no sign here. The rule never needs the sign,
# because a zero of either sign becomes the integer 0.
#
# An infinity or a NaN has no Fraction. No step of the rule turns one back into a finite number (an
# infinity times zero is a NaN), and the rule rejects every decoded value, multiplier and product
# (see "The conversion rule" below) that is not finite. So from_bits() raises Rejected for a stored
# infinity or NaN, and rn32() for a result that IEEE 754 rounds to an infinity.

CONTAINER, TYPE, NOT_FINITE, BOUND = 22, 23, 24, 25  # the exit statuses with which the programs reject an input


class Rejected(Exception):
    """The conversion rule rejects the input.

    `status` is the exit status that bin/weight_sums must give. weight_sums/weight_sums.py must give
    the same status for every fault that the rule defines. sums_file() lists inputs on which the two
    programs differ.
    """

    def __init__(self, status, why):
        super().__init__(why)
        self.status = status


TWO = Fraction(2)
HALF = Fraction(1, 2)


def rne(x):
    """RNE: return the integer nearest to the exact number x, and at a tie the even one."""
    below = x.numerator // x.denominator  # the floor of x
    rest = x - below
    if rest > HALF or (rest == HALF and below % 2 == 1):
        return below + 1
    return below


def exponent_of(x):
    """Return the integer k with 2^k <= abs(x) < 2^(k + 1), for an exact number x other than 0."""
    k = abs(x.numerator).bit_length() - x.denominator.bit_length()
    return k if abs(x) >= TWO ** k else k - 1


def rn32(x):
    """RN32: return the binary32 value nearest to the exact x, ties to the even significand."""
    if x == 0:
        return x
    # A significand has 24 bits, so the values from 2^k to 2^(k + 1) are 2^(k - 23) apart. Below
    # 2^-126, the smallest normal number, the subnormal numbers continue with the spacing 2^-149.
    step = TWO ** (max(exponent_of(x), -126) - 23)
    nearest = rne(x / step) * step
    if abs(nearest) >= TWO ** 128:
        raise Rejected(NOT_FINITE, "a binary32 result is an infinity")
    return nearest


def from_bits(word, exponent_bits, fraction_bits):
    """Return the value of a floating-point word: sign bit, exponent field, fraction field."""
    fraction = word & ((1 << fraction_bits) - 1)
    field = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    if field == (1 << exponent_bits) - 1:
        raise Rejected(NOT_FINITE, "a stored number is an infinity or a NaN")
    if field == 0:  # zero or a subnormal number
        size = fraction * TWO ** (1 - bias - fraction_bits)
    else:
        size = ((1 << fraction_bits) + fraction) * TWO ** (field - bias - fraction_bits)
    return -size if word >> (exponent_bits + fraction_bits) else size


def binary16(two_bytes):
    return from_bits(int.from_bytes(two_bytes, "little"), 5, 10)


def binary32(four_bytes):
    return from_bits(int.from_bytes(four_bytes, "little"), 8, 23)


def bfloat16(two_bytes):
    """The upper 16 bits of a binary32 word: 8 exponent bits and 7 fraction bits below the sign."""
    return from_bits(int.from_bytes(two_bytes, "little"), 8, 7)


# ---------------------------------------------------------------------------------------------------------------------
# The decoded values of a tensor
#
# The byte layout of the block formats Q4_K, Q5_K and Q6_K, and the rule that decodes them, are
# those of ggml, the tensor library of the llama.cpp project (github.com/ggml-org/llama.cpp). They
# are defined in ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K) and in
# ggml/src/ggml-quants.c (get_scale_min_k4; dequantize_row_q4_K, _q5_K, _q6_K) of the copy of
# llama.cpp that the docstring at the top names. The functions below follow these definitions, as
# they are in that copy, and docs/model.md, "11.2 Stored blocks and their values".
# scale_and_minimum has the same two cases and the same four expressions as get_scale_min_k4.
# ggml has the MIT licence, "Copyright (c) 2023-2024 The ggml authors".
# ../../THIRD_PARTY.md, section "ggml", has the full citation and the text of the licence.
#
# values_q4_k_or_q5_k() and values_q6_k() yield the 256 decoded values of one super-block (a block
# of these formats, which consists of sub-blocks) in the order of the elements. Every product and
# every difference is rounded by RN32 in the order in which both ggml and gguf-py 0.19.0
# (gguf/quants.py) evaluate it.


def scale_and_minimum(packed, j):
    """Return the 6-bit scale and the 6-bit minimum of sub-block j of a Q4_K or Q5_K block.

    `packed` is the block's twelve bytes that hold eight such pairs. The function has the same two
    cases and the same four expressions as ggml's get_scale_min_k4.
    """
    if j < 4:
        return packed[j] & 63, packed[j + 4] & 63
    return (packed[j + 4] & 15) | ((packed[j - 4] >> 6) << 4), (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4)


def values_q4_k_or_q5_k(block):
    """Q4_K (144 bytes): d, dmin, 12 packed bytes, 128 code bytes.

    Q5_K (176 bytes) has 32 bytes qh before the codes. Sub-block j has 32 elements. Two sub-blocks
    share 32 code bytes: the even one has the low halves, the odd one the high halves. In Q5_K, bit
    j of qh[i] is the fifth bit of element i of sub-block j. d multiplies the scales, and dmin the
    minimums, which are subtracted.
    """
    d, dmin, packed = binary16(block[0:2]), binary16(block[2:4]), block[4:16]
    qh, codes = (block[16:48], block[48:176]) if len(block) == 176 else (bytes(32), block[16:144])
    for j in range(8):
        sc, m = scale_and_minimum(packed, j)
        scale, minimum = rn32(d * sc), rn32(dmin * m)
        for i in range(32):
            byte = codes[32 * (j // 2) + i]
            q = (byte >> 4 if j % 2 else byte & 15) + 16 * ((qh[i] >> j) & 1)
            yield rn32(rn32(scale * q) - minimum)


def values_q6_k(block):
    """Q6_K (210 bytes): ql (128 bytes), qh (64 bytes), sixteen signed scales, d.

    Write the number of an element as n = 128 a + 32 t + i, with t < 4 and i < 32. The low four bits
    of its code are in ql[64 a + 32 (t mod 2) + i], in the low half of the byte for t < 2 and in the
    high half otherwise. The two high bits are bits 2 t and 2 t + 1 of qh[32 a + i]. The code is
    that 6-bit number minus 32. Each run of 16 elements has its own scale.
    """
    ql, qh, scales, d = block[0:128], block[128:192], block[192:208], binary16(block[208:210])
    for n in range(256):
        a, t, i = n // 128, n // 32 % 4, n % 32
        byte = ql[64 * a + 32 * (t % 2) + i]
        q = ((byte & 15 if t < 2 else byte >> 4) | (((qh[32 * a + i] >> (2 * t)) & 3) << 4)) - 32
        s = scales[n // 16] - 256 if scales[n // 16] >= 128 else scales[n // 16]  # two's complement
        yield rn32(rn32(d * s) * q)


def values_q8_0(block):
    """Q8_0 (34 bytes): d (binary16), then 32 signed codes q. v = RN32(d * q).

    Public sources: the package gguf, version 0.19.0, gguf/quants.py, Q8_0.dequantize_blocks; llama.cpp's wiki page
    "Tensor Encoding Schemes" and Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf): "Each block has 32
    weights", "w = q * block_scale".
    """
    d = binary16(block[0:2])
    for byte in block[2:34]:
        yield rn32(d * (byte - 256 if byte >= 128 else byte))  # two's complement


def decoded_values(storage_type, data):
    """Yield the decoded value v of every element of a tensor, in the order of the file.

    The caller must pass a storage type that is a key of BLOCK, and data that are a whole number of
    blocks, as stored_bytes() returns them. Neither is checked.

    F16 stores binary16 numbers, and BF16 the upper 16 bits of binary32 numbers (Hugging Face, "GGUF",
    https://huggingface.co/docs/hub/gguf; the package gguf, version 0.19.0, gguf/quants.py:
    dequantize() and the class BF16). v is the stored number.
    """
    size = BLOCK[storage_type][1]
    for start in range(0, len(data), size):
        block = data[start:start + size]
        if storage_type == F32:
            yield binary32(block)
        elif storage_type == F16:
            yield binary16(block)
        elif storage_type == BF16:
            yield bfloat16(block)
        elif storage_type == Q8_0:
            yield from values_q8_0(block)
        elif storage_type == Q6_K:
            yield from values_q6_k(block)
        else:
            yield from values_q4_k_or_q5_k(block)


# ---------------------------------------------------------------------------------------------------------------------
# The conversion rule
#
# A norm is the weight vector (the gains) of an RMS normalization. A matrix (any committed tensor
# but the embedding) reads a norm if the model applies it to that normalization's output. Each
# decoded value v of such a matrix gives the product p = RN32(gamma_j * v), with the multiplier
# gamma_j of its input index j (see multipliers_of()). In a matrix that reads no norm, p = v.

WIDTH, HEAD_DIMENSION = 5120, 128  # the only model width and head dimension for which the two constants below are right
NORM_CONSTANT = from_bits(0x3F8F1BBD, 8, 23)  # RN32(sqrt(5120) / 64)
QUERY_CONSTANT = from_bits(0x3DCA62C2, 8, 23)  # RN32(sqrt(5120) / 64 / sqrt(128))
LARGEST_INTEGER = 4095  # of a matrix
EXPONENTS = range(-100, 101)  # the exponents that a matrix may have, and the values that e_res may have
SMALLEST_MULTIPLIER = TWO ** -100  # a multiplier is 0 or at least this in absolute value
EMBEDDING_BOUND = 2 ** 30  # every integer of the embedding is smaller than this in absolute value

# For each kind of matrix in a layer: the norm whose gains it reads and the constant c by which
# multipliers_of() multiplies them, or None if it reads no norm.
LAYER_MATRICES = {
    b"attn_q": (b"attn_norm", QUERY_CONSTANT),
    b"attn_k": (b"attn_norm", NORM_CONSTANT),
    b"attn_v": (b"attn_norm", NORM_CONSTANT),
    b"attn_output": None,
    b"ffn_gate": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_up": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_down": None,
    b"ffn_gate_inp": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_gate_shexp": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_up_shexp": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_down_shexp": None,
    b"ffn_gate_exps": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_up_exps": (b"ffn_norm", NORM_CONSTANT),
    b"ffn_down_exps": None,
}
LAYER_NORMS = (b"attn_norm", b"ffn_norm")
LAYER_NAME = re.compile(rb"(blk\.(0|[1-9][0-9]{0,8})\.)([a-z_]+)\.weight")
# A layer number has no leading zero and at most nine digits (see sums_file()).


def shown(name):
    """Return a key or a tensor's name as text for a message."""
    return name.decode("utf-8", "backslashreplace")


@contextlib.contextmanager
def naming(name):
    """Prefix the tensor's name to the message of a Rejected that is raised in the block."""
    try:
        yield
    except Rejected as fault:
        raise Rejected(fault.status, "%s: %s" % (shown(name), fault)) from None


def model_shape(metadata):
    """Check the model's width and head dimension, and return its number of layers.

    `metadata` is the first shard's. A head dimension that is absent passes.
    """
    pairs = {as_bytes(key): (value_type, value) for key, value_type, value in metadata}

    def number(key):
        value_type, value = pairs.get(key, (None, None))
        return value if value_type in (UINT8, UINT16, UINT32, UINT64) else None

    value_type, architecture = pairs.get(b"general.architecture", (None, None))
    if value_type != STRING:
        raise Rejected(CONTAINER, "the first shard's metadata lack the string general.architecture")
    prefix = as_bytes(architecture) + b"."
    width, layers, head_dimension = (number(prefix + key) for key in (b"embedding_length", b"block_count",
                                                                 b"attention.key_length"))
    if width is None or layers is None:
        raise Rejected(CONTAINER, "the first shard's metadata lack %sembedding_length or %sblock_count as an unsigned "
                       "integer" % (shown(prefix), shown(prefix)))
    if width != WIDTH or head_dimension not in (None, HEAD_DIMENSION):
        raise Rejected(BOUND, "the rule is defined for the width %d and the head dimension %d only, and the metadata "
                       "give %d and %s" % (WIDTH, HEAD_DIMENSION, width, head_dimension))
    return layers


def role_of(name, layers):
    """Return what the rule does with the tensor of this name.

    The result is "ignored", "norm", "embedding", or for a matrix a pair: the name of the norm that
    it reads and the constant, or (None, None) if it reads no norm.
    """
    if name == b"token_embd.weight":
        return "embedding"
    if name == b"output_norm.weight":
        return "norm"
    if name == b"output.weight":
        return b"output_norm.weight", NORM_CONSTANT
    match = LAYER_NAME.fullmatch(name)
    if not match or (match[3] not in LAYER_MATRICES and match[3] not in LAYER_NORMS):
        return "ignored"
    if int(match[2]) >= layers:
        raise Rejected(CONTAINER, "the model has only %d layers" % layers)
    if match[3] in LAYER_NORMS:
        return "norm"
    norm, constant = LAYER_MATRICES[match[3]] or (None, None)
    return (match[1] + norm + b".weight" if norm else None), constant


def elements_of(dims):
    """Return the number of elements of a tensor with these dimensions."""
    elements = math.prod(dims)
    if not 1 <= len(dims) <= 4 or not 0 < elements < 2 ** 64:
        raise Rejected(CONTAINER, "the dimensions %s are none, more than four, or have a product of 0 or of 2^64 or "
                       "more" % (list(dims),))
    return elements


def stored_bytes(dims, storage_type, data):
    """Return the bytes that hold the elements of a committed tensor or a norm.

    Further bytes of `data` are a gap in the file.
    """
    if storage_type not in BLOCK:
        raise Rejected(TYPE, "the rule does not read the storage type %d" % storage_type)
    in_block, block_bytes = BLOCK[storage_type]
    if dims[0] % in_block:
        raise Rejected(CONTAINER, "a row of %d elements is not a whole number of blocks" % dims[0])
    needed = elements_of(dims) // in_block * block_bytes
    if len(data) < needed:
        raise Rejected(CONTAINER, "the data have %d bytes, and the dimensions ask for %d" % (len(data), needed))
    return data[:needed]


def multipliers_of(gains, constant):
    """Return the multiplier gamma_j = RN32(g_j * c) of every input index j."""
    multipliers = [rn32(g * constant) for g in gains]
    if any(0 < abs(gamma) < SMALLEST_MULTIPLIER for gamma in multipliers):
        raise Rejected(BOUND, "some multiplier is nonzero but smaller than 2^-100 in absolute value")
    return multipliers


def integers_of_matrix(products):
    """Return the exponent e of a matrix with these products p, and its integers RNE(2^e * p).

    e is the largest exponent at which no integer exceeds LARGEST_INTEGER in absolute value.
    """
    largest = max(abs(p) for p in products)
    if largest == 0:
        raise Rejected(BOUND, "the matrix has only zeros")
    # RNE(2^e * largest) does not fall as e grows. So the search goes down from 101, one above the
    # allowed range, and the first e that meets the bound is the rule's exponent. If that is 101
    # itself, the exponent is 101 or more.
    e = EXPONENTS[-1] + 1
    while rne(TWO ** e * largest) > LARGEST_INTEGER:
        e -= 1
    if e not in EXPONENTS:
        raise Rejected(BOUND, "the exponent of the matrix is outside -100 .. 100")
    return e, [rne(TWO ** e * p) for p in products]


def integers_of_embedding(values, eres):
    integers = [rne(TWO ** eres * v) for v in values]
    if any(abs(w) >= EMBEDDING_BOUND for w in integers):
        raise Rejected(BOUND, "an integer of the embedding is 2^30 or more in absolute value")
    return integers


def bits_of(integers):
    """Return the least b >= 1 with -2^(b-1) <= w < 2^(b-1) for every integer w."""
    b = 1
    while min(integers) < -(2 ** (b - 1)) or max(integers) >= 2 ** (b - 1):
        b += 1
    return b


def line_of(name, role, tensor, gains, eres):
    """Return the line of the sums file for one committed tensor.

    weight_sums/README.md, "The file's form", defines its five fields. `role` is role_of()'s result,
    `tensor` is (dims, storage type, data), and `gains` maps the name of each norm to its bytes.
    """
    dims, storage_type, data = tensor
    values = list(decoded_values(storage_type, stored_bytes(dims, storage_type, data)))
    if role == "embedding":
        e, integers, integer_bytes = eres, integers_of_embedding(values, eres), 4
    else:
        norm, constant = role
        if norm:
            if norm not in gains:
                raise Rejected(CONTAINER, "the norm %s is absent" % shown(norm))
            if dims[0] != WIDTH:
                raise Rejected(CONTAINER, "a matrix that reads a norm must have rows of %d elements" % WIDTH)
            multipliers = multipliers_of(decoded_values(F32, gains[norm]), constant)
            # The input index j of an element is its first index, which runs fastest in the file.
            values = [rn32(multipliers[position % WIDTH] * v) for position, v in enumerate(values)]
        (e, integers), integer_bytes = integers_of_matrix(values), 2
    digest = hashlib.sha256(b"".join(w.to_bytes(integer_bytes, "little", signed=True) for w in integers))
    return b"%s %d %d %d %s\n" % (name, len(integers), e, bits_of(integers), digest.hexdigest().encode())


def sums_file(shards, eres):
    """Return the bytes of the sums file that the rule prescribes for these shards and this e_res.

    shards is a list of (metadata, tensors), one pair for each shard, in the form that write_shard()
    takes. The function raises Rejected for an input that the rule rejects, and ValueError for an
    eres outside EXPONENTS.

    The function applies the rule to the lists. It does not check the file that write_shard() makes:
      - It reads the metadata that model_shape() looks up, in the first shard only. Of
        general.alignment it looks only at the value type, in every shard. It accepts nested arrays
        and value types that GGUF does not define.
      - It does not look at the storage type or the data of an ignored tensor, or at the values of a
        norm that no matrix reads.
      - It accepts an ignored tensor of up to 2^64 - 1 elements. bin/weight_sums allows at most 2^40
        in any tensor.
      - It rejects data that are shorter than the dimensions ask for. In a file, padding or the next
        tensor follows the data. So a program rejects the file only if the bytes that the dimensions
        ask for run past the end of the file.

    The two programs differ on some inputs that no model file has. Where the function looks at such
    an input, it does as bin/weight_sums does:
      - of a metadata key that appears twice, the last value counts;
      - a number in the metadata counts as absent unless its value type is UINT8, UINT16, UINT32 or
        UINT64;
      - a tensor with a dimension of 0 is rejected, even if the rule ignores its name;
      - a name with a layer number of ten or more digits is ignored;
      - a committed tensor of a storage type number that ggml does not define has the status 23.
    There are two exceptions: the function accepts an empty architecture name and, as said above, an
    ignored tensor of more than 2^40 elements, and bin/weight_sums rejects both.
    weight_sums/weight_sums.py also rejects what gguf-py's reader rejects in a part that the
    function does not look at, such as an ignored tensor of an undefined storage type, or a key or a
    name that is not valid UTF-8.
    """
    if eres not in EXPONENTS:
        raise ValueError("eres must be an integer from -100 to 100")
    layers = model_shape(shards[0][0])
    table, roles = {}, {}
    for metadata, tensors in shards:
        types = {as_bytes(key): value_type for key, value_type, _ in metadata}
        if types.get(b"general.alignment", UINT32) != UINT32:
            raise Rejected(CONTAINER, "general.alignment is not a UINT32")
        for name, dims, storage_type, data in tensors:
            name = as_bytes(name)
            with naming(name):
                if name in table:
                    raise Rejected(CONTAINER, "the name appears twice")
                table[name] = (list(dims), storage_type, data)
                roles[name] = role_of(name, layers)
                elements_of(dims)

    gains = {}  # for each norm, its stored bytes
    for name, role in roles.items():
        if role == "norm":
            with naming(name):
                if table[name][1] != F32:
                    raise Rejected(TYPE, "a norm must be an F32 tensor")
                gains[name] = stored_bytes(*table[name])
                if len(gains[name]) != 4 * WIDTH:
                    raise Rejected(CONTAINER, "a norm must have %d elements" % WIDTH)

    lines = []
    for name in sorted(table):  # bytes compare as the sums file orders its lines
        if roles[name] not in ("ignored", "norm"):
            with naming(name):
                lines.append(line_of(name, roles[name], table[name], gains, eres))
    return b"".join(lines) + b"END %d\n" % len(lines)


# ---------------------------------------------------------------------------------------------------------------------
# The synthetic model
#
# The model has every kind of tensor that the rule names and every storage type that it reads. Its
# rows are as long as the rule demands (5,120 elements where a matrix reads a norm), and it has few
# of them, so that exact arithmetic takes seconds. Most values are drawn at random around a chosen
# size. The first blocks of every tensor in the list COMMITTED, and some gains of every norm, have
# the values that random bytes seldom give.

LAYERS = 11  # most tensors are in layers 2 and 10: "blk.10." sorts before "blk.2.", and 10 is the last layer
# Committed tensors: name and dims, the length of a row first. synthetic_model() adds one more.
COMMITTED = (
    ("blk.2.attn_q.weight", (WIDTH, 2)),
    ("blk.2.attn_k.weight", (WIDTH, 1)),
    ("blk.2.attn_v.weight", (WIDTH, 2)),
    ("blk.2.attn_output.weight", (512, 4)),
    ("blk.2.ffn_gate.weight", (WIDTH, 2)),
    ("blk.2.ffn_up.weight", (WIDTH, 1)),
    ("blk.2.ffn_down.weight", (768, 3)),
    ("blk.10.ffn_gate_inp.weight", (WIDTH, 3)),
    ("blk.10.ffn_gate_shexp.weight", (WIDTH, 1)),
    ("blk.10.ffn_up_shexp.weight", (WIDTH, 2)),
    ("blk.10.ffn_down_shexp.weight", (256, 7)),
    ("blk.10.ffn_gate_exps.weight", (WIDTH, 1, 3)),
    ("blk.10.ffn_up_exps.weight", (WIDTH, 2, 2)),
    ("blk.10.ffn_down_exps.weight", (256, 3, 3)),
    ("output.weight", (WIDTH, 3)),
    ("token_embd.weight", (WIDTH, 3)),
)
NORMS = ("blk.2.attn_norm.weight", "blk.2.ffn_norm.weight", "blk.10.ffn_norm.weight", "output_norm.weight")
NORM_LEVELS = (0, 50, -60, -3)  # the gains of one norm are near 2^0, those of the next near 2^50, and so on
# Tensors that the rule ignores: name, dims, storage type number (F32, F16, Q8_0, Q3_K, BF16) and
# number of data bytes.
IGNORED = (
    ("rope_freqs.weight", (64,), 0, 256),
    ("oracle.f16", (32, 2), 1, 128),
    ("oracle.q8_0", (64, 2), 8, 136),
    ("oracle.q3_k", (256,), 11, 110),
    ("oracle.bf16", (48,), 30, 96),
)


class Draws:
    """A sequence of numbers that depends on the seed only, and not on the version of Python.

    word() is one step of SplitMix64 (../../THIRD_PARTY.md, "SplitMix64", has the citation).
    """

    def __init__(self, seed):
        self.state = seed  # from 0 to 2^64 - 1

    def word(self):
        """Return the next 64 bits."""
        self.state = (self.state + 0x9E3779B97F4A7C15) % 2 ** 64
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 % 2 ** 64
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB % 2 ** 64
        return z ^ (z >> 31)

    def below(self, n):
        return self.word() % n

    def between(self, low, high):
        """Return an integer from low to high, both included."""
        return low + self.below(high - low + 1)

    def random_bytes(self, n):
        return b"".join(self.word().to_bytes(8, "little") for _ in range((n + 7) // 8))[:n]

    def shuffled(self, items):
        """Return the items in a random order, by the Fisher-Yates shuffle."""
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def float_bytes(negative, exponent, fraction, exponent_bits, fraction_bits):
    """Return the stored bytes of the number +-(1 + fraction / 2^fraction_bits) * 2^exponent.

    `negative` is the sign bit. An exponent below the format's normal range gives the exponent field
    0, that is a zero or a subnormal number.
    """
    field = max(exponent + 2 ** (exponent_bits - 1) - 1, 0)
    assert field < 2 ** exponent_bits - 1, "the number would be an infinity or a NaN"
    word = (((negative << exponent_bits) | field) << fraction_bits) | fraction
    return word.to_bytes((1 + exponent_bits + fraction_bits) // 8, "little")


def half(draws, exponent, fraction=None):
    """Return the bytes of a binary16 number of random sign with this exponent.

    The fraction field is random unless one is given.
    """
    return float_bytes(draws.below(2), exponent, draws.below(2 ** 10) if fraction is None else fraction, 5, 10)


def single(draws, exponent, fraction=None):
    """The same for binary32."""
    return float_bytes(draws.below(2), exponent, draws.below(2 ** 23) if fraction is None else fraction, 8, 23)


# Each of the next five functions returns the bytes of 256 consecutive elements of one or two storage
# types. `variant` chooses one of the special blocks, and any other value gives an ordinary block. In
# variant 0 all values are multiples of one power of two, the step, and `lift` makes the step 2^lift
# times larger (synthetic_tensor() explains why). All values are below 2^(level + 1) in absolute
# value, and with a lift they can reach it. The comments use the decoders' names: d, dmin, a scale
# sc or s, and a code q (values_q4_k_or_q5_k(), values_q6_k()).


def synthetic_f32(draws, level, variant, lift=0):
    if variant == 0:  # the step is 2^(level - bits)
        bits = 12 - lift
        shifts = [draws.between(0, bits) for _ in range(256)]
        return b"".join(single(draws, level - s, draws.below(2 ** (bits - s)) << (23 - bits + s)) for s in shifts)
    if variant == 1:
        # Zeros of both signs, subnormal numbers, the smallest normal number and the smallest
        # subnormal number.
        kinds = ((-127, 0), (-127, None), (-126, 0), (-127, 1))  # exponent and fraction
        return b"".join(single(draws, *kinds[draws.below(4)]) for _ in range(256))
    return b"".join(single(draws, level - draws.between(0, 11)) for _ in range(256))


def synthetic_q4_k_or_q5_k(draws, size, level, variant, lift=0):
    """size is the number of bytes of the super-block, 144 for Q4_K and 176 for Q5_K."""
    block = bytearray(draws.random_bytes(size))  # every pattern of bits is a valid set of scales, minimums and codes
    exponent = level - 12  # sc * q is below 2^11
    # The exponent of dmin is up to 12 below or up to 3 above that of d, so that many subtractions
    # have to round.
    block[0:2], block[2:4] = half(draws, exponent), half(draws, exponent + draws.between(-12, 3))
    if variant == 0:
        # d and dmin are the same power of two, apart from the sign, and that is the step. The
        # largest values of a Q4_K tensor are about half as large as those of a Q5_K tensor (sc * q
        # stays below 2^10 and not 2^11), so its step is half as large too.
        step = exponent + lift - (2 if size == 176 else 3)
        block[0:2], block[2:4] = half(draws, step, 0), half(draws, step, 0)
    elif variant == 1:  # every scale and minimum is 63, and every code has its largest value
        block[4:] = b"\xff" * (size - 4)
    elif variant == 2:  # every scale and minimum is 0
        block[4:16] = bytes(12)
    elif variant == 3:  # d and dmin are subnormal
        block[0:2], block[2:4] = half(draws, -15), half(draws, -15)
    elif variant == 4:
        # dmin = d, every scale is 5 and every minimum is 15, so an element with the code 3 has the
        # difference 0. The first eight elements of sub-blocks 0 and 1 get that code.
        block[2:4] = block[0:2]
        block[4:16] = b"\x05" * 4 + b"\x0f" * 4 + b"\xf5" * 4
        block[16:24] = bytes(8)  # in Q5_K, the fifth bits of these elements
        codes = size - 128  # where the code bytes start, which is 16 in Q4_K
        block[codes:codes + 8] = b"\x33" * 8
    elif variant == 5:  # d and dmin are zeros of different signs
        block[0:4] = b"\x00\x00\x00\x80"
    return bytes(block)


def synthetic_q6_k(draws, level, variant, lift=0):
    block = bytearray(draws.random_bytes(210))  # every pattern of bits is a valid set of codes and scales
    exponent = level - 12  # s * q is at most 2^12
    block[208:210] = half(draws, exponent)
    if variant == 0:  # d is a power of two, and that is the step
        block[208:210] = half(draws, exponent + lift - 1, 0)
    elif variant == 1:  # every code is -32 and every scale is -128
        block[0:192], block[192:208] = bytes(192), b"\x80" * 16
    elif variant == 2:  # every code is 31 and every scale is 127
        block[0:192], block[192:208] = b"\xff" * 192, b"\x7f" * 16
    elif variant == 3:  # every scale is 0
        block[192:208] = bytes(16)
    elif variant == 4:  # d is subnormal
        block[208:210] = half(draws, -15)
    return bytes(block)


def synthetic_q8_0(draws, level, variant, lift=0):
    """Eight blocks of Q8_0."""
    blocks = [bytearray(draws.random_bytes(34)) for _ in range(8)]  # every pattern of bits is a valid set of codes
    for block in blocks:
        block[0:2] = half(draws, level - 7 - draws.between(0, 3))  # |q| is at most 2^7
        if variant == 0:  # d is a power of two, and that is the step
            block[0:2] = half(draws, level - 13 + lift, 0)
        elif variant == 1:  # every code is -128
            block[2:] = b"\x80" * 32
        elif variant == 2:  # every code is 127
            block[2:] = b"\x7f" * 32
        elif variant == 3:  # d is a zero of either sign
            block[0:2] = half(draws, -15, 0)
        elif variant == 4:  # d is subnormal
            block[0:2] = half(draws, -15)
    return b"".join(bytes(block) for block in blocks)


def synthetic_16_bits(draws, fraction_bits, level, variant, lift=0):
    """256 values of F16 (10 fraction bits) or of BF16 (7 fraction bits)."""
    exponent_bits = 15 - fraction_bits

    def value(exponent, fraction=None):
        fraction = draws.below(2 ** fraction_bits) if fraction is None else fraction
        return float_bytes(draws.below(2), exponent, fraction, exponent_bits, fraction_bits)

    lowest = 1 - 2 ** (exponent_bits - 1)  # one below the exponent of the smallest normal number
    if variant == 0:
        # The step is 2^(level - 13 + lift). A value with the exponent level - s has fraction_bits bits below its
        # leading bit, of which those below the step must be 0.
        made = []
        for _ in range(256):
            s = draws.between(0, 13 - lift)
            free = min(fraction_bits, 13 - lift - s)
            made.append(value(level - s, draws.below(2 ** free) << (fraction_bits - free)))
        return b"".join(made)
    if variant == 1:  # zeros of both signs, subnormal numbers, the smallest normal and the smallest subnormal number
        kinds = ((lowest, 0), (lowest, None), (lowest + 1, 0), (lowest, 1))
        return b"".join(value(*kinds[draws.below(4)]) for _ in range(256))
    return b"".join(value(level - draws.between(0, 11)) for _ in range(256))


def synthetic_tensor(draws, storage_type, dims, level, tilts, steps_at=None):
    """Return the data of one tensor of the synthetic model.

    Its values in columns 256 b to 256 b + 255 are near 2^(level - tilts[b]) or below. A column is
    an input index j. The first blocks of the tensor are the special ones (variants 0, 1, ..) of its
    storage type.

    Variant 0 produces ties in the rule's last rounding, RNE(2^e * p). Its step is about 2^13 times
    smaller than the largest values of the tensor. The rule's scale 2^e turns the largest values
    into integers of 12 bits. So it turns the step into 1/2, 1/4 or 1/8, and many of the multiples
    are halfway between two integers.

    In a matrix that reads a norm this needs multipliers that are powers of two. synthetic_norm()
    provides them in the columns 256 steps_at to 256 steps_at + 255, so such a matrix has variant 0
    there, in every row, and nowhere else. The other multipliers are one to four times as large as
    these, and they set the scale. So the step is four times larger in such a matrix.
    """
    blocks = []
    special = 0 if steps_at is None else 1  # the variant of the next block outside the columns of steps_at
    for _ in range(math.prod(dims[1:])):  # the rows of all experts
        for b in range(dims[0] // 256):
            if b == steps_at:
                variant, lift = 0, 2
            else:
                variant, lift, special = special, 0, special + 1
            if storage_type == F32:
                blocks.append(synthetic_f32(draws, level - tilts[b], variant, lift))
            elif storage_type == Q6_K:
                blocks.append(synthetic_q6_k(draws, level - tilts[b], variant, lift))
            elif storage_type == Q8_0:
                blocks.append(synthetic_q8_0(draws, level - tilts[b], variant, lift))
            elif storage_type in (F16, BF16):
                fraction_bits = 10 if storage_type == F16 else 7
                blocks.append(synthetic_16_bits(draws, fraction_bits, level - tilts[b], variant, lift))
            else:
                blocks.append(synthetic_q4_k_or_q5_k(draws, BLOCK[storage_type][1], level - tilts[b], variant, lift))
    return b"".join(blocks)


def synthetic_norm(draws, level):
    """Return the data of a norm and two descriptions of it, tilts and steps_at.

    tilts is a list: the gains 256 b to 256 b + 255 are near 2^(level + tilts[b]). steps_at is a
    dict: for each of the rule's two constants c, the gains 256 b to 256 b + 255 with b =
    steps_at[c] have multipliers RN32(g * c) that are the same power of two, apart from the sign.
    (Only an attn_norm is read with the second constant.) Up to sixteen special gains then replace
    gains at random places, also in these runs.
    """
    tilts = [draws.between(-4, 4) for _ in range(WIDTH // 256)]
    gains = [single(draws, level + tilts[j // 256]) for j in range(WIDTH)]  # the signs are random
    steps_at = dict(zip((NORM_CONSTANT, QUERY_CONSTANT), draws.shuffled(range(WIDTH // 256))))
    for constant, b in steps_at.items():
        power = TWO ** (level + tilts[b] + exponent_of(constant))
        g = rn32(power / constant)
        assert rn32(g * constant) == power  # this holds for both constants at every power that the model uses
        fraction = int((g / TWO ** exponent_of(g) - 1) * 2 ** 23)
        gains[256 * b:256 * b + 256] = [single(draws, exponent_of(g), fraction) for _ in range(256)]
    for _ in range(8):
        gains[draws.below(WIDTH)] = single(draws, -127, 0)  # a zero of either sign
        gains[draws.below(WIDTH)] = single(draws, -95)  # its multiplier is not far above the bound 2^-100
    return b"".join(gains), tilts, steps_at


def synthetic_value(draws, value_type):
    """Return a metadata value of this type, which is not ARRAY."""
    if value_type == STRING:
        return "".join("ab \u00e9\u4e2d"[draws.below(5)] for _ in range(draws.below(12)))
    if value_type == BOOL:
        return bool(draws.below(2))
    if value_type in (FLOAT32, FLOAT64):
        return draws.between(-4096, 4096) / 64
    bits = 8 * struct.calcsize(SCALAR_FORMAT[value_type])
    return draws.below(2 ** bits) - (2 ** (bits - 1) if value_type in (INT8, INT16, INT32, INT64) else 0)


def synthetic_metadata(draws, number, tensors):
    """Return metadata that the rule does not read.

    They are the three keys of a split model, one value and one array of each value type but ARRAY,
    and a long array of strings. `number` is the shard's number, 0 or 1, and `tensors` is the number
    of tensors in both shards together.
    """
    # gguf-py's writer gives every part of a split model these three keys (gguf/gguf_writer.py of
    # gguf-py 0.19.0).
    metadata = [("split.no", UINT16, number), ("split.count", UINT16, 2), ("split.tensors.count", INT32, tensors)]
    for value_type in range(13):
        if value_type != ARRAY:
            several = [synthetic_value(draws, value_type) for _ in range(draws.between(1, 5))]
            metadata += [("oracle.one.%d" % value_type, value_type, several[0]),
                         ("oracle.several.%d" % value_type, ARRAY, (value_type, several))]
    # A real model file has long arrays of strings for its tokenizer.
    metadata.append(("tokenizer.ggml.tokens", ARRAY, (STRING, [synthetic_value(draws, STRING) for _ in range(300)])))
    return metadata


def synthetic_model(seed):
    """Return the two shards as a list of (metadata, tensors), and the list of their alignments."""
    draws = Draws(seed)
    norms, tilts, steps_at = {}, {}, {}
    for i, name in enumerate(NORMS):
        data, tilts[name.encode()], steps_at[name.encode()] = synthetic_norm(draws, NORM_LEVELS[(i + seed) % 4])
        norms[name] = (name, (WIDTH, 1) if i == 2 else (WIDTH,), F32, data)  # a further dimension of 1 changes nothing

    matrices = []
    for i, (name, dims) in enumerate(COMMITTED):
        # Four seeds in a row below 100 give a tensor each of the types F32, Q4_K, Q5_K and Q6_K once. Three seeds in a
        # row from 100 on give it each of the types Q8_0, F16 and BF16 once.
        storage_type = (F32, Q4_K, Q5_K, Q6_K)[(i + seed) % 4] if seed < 100 else (Q8_0, F16, BF16)[(i + seed) % 3]
        role = role_of(name.encode(), LAYERS)
        if role == "embedding":
            level = 9  # the integers are wider than 16 bits at e_res = 11 and within the bound 2^30 up to e_res = 20
        elif role[0]:
            # With a tilt of up to 4, a binary16 number, be it a value of F16 or the scale of Q8_0, stays finite.
            level = draws.between(7, 20 if seed < 100 else 11)
        elif storage_type in (F16, Q8_0):
            level = draws.between(3, 15)
        elif storage_type in (F32, BF16):  # both have binary32's range of exponents
            level = draws.between(-80, 80)
        else:
            level = draws.between(3, 24)  # d is then a normal binary16 number, and dmin is finite
        if role != "embedding" and role[0]:
            # The matrix has small values where the gains of its norm are large, so that the
            # products are of one size.
            data = synthetic_tensor(draws, storage_type, dims, level, tilts[role[0]], steps_at[role[0]][role[1]])
        else:
            data = synthetic_tensor(draws, storage_type, dims, level, [0] * (dims[0] // 256))
        matrices.append((name, dims, storage_type, data))

    # One more matrix, with one dimension. The value with the largest absolute value is minus a
    # power of two, so the integers reach -2,048 and not 2,048, and they fit 12 bits.
    level = draws.between(-80, 80)
    data = bytearray(synthetic_f32(draws, level - 2, None))
    at = 4 * draws.below(256)
    data[at:at + 4] = float_bytes(1, level, 0, 8, 23)
    matrices.append(("blk.0.attn_output.weight", (256,), F32, bytes(data)))

    ignored = [(name, dims, storage_type, draws.random_bytes(size)) for name, dims, storage_type, size in IGNORED]

    # The matrices and the ignored tensors alternate between the shards. All norms are in one shard,
    # so some matrices read a norm of the other shard. Inside a shard the order is random, and so is
    # the order of the metadata.
    others = draws.shuffled(matrices) + draws.shuffled(ignored)
    tensors = [others[0::2], others[1::2]]
    tensors[seed % 2] += norms.values()
    total = len(others) + len(norms)
    # Both shards have the keys that the rule reads, so a program can take either of them as the
    # first one.
    needed = [("general.architecture", STRING, "llama4"), ("llama4.embedding_length", UINT32, WIDTH),
              ("llama4.block_count", UINT32, LAYERS)]
    if seed % 2 == 0:  # where the key is absent, the head dimension is 128
        needed.append(("llama4.attention.key_length", UINT32, HEAD_DIMENSION))
    alignments = [32, 32 << draws.between(1, 3)]  # the first shard has no key for its alignment, which is then 32
    metadata = [needed + synthetic_metadata(draws, 0, total),
                needed + [("general.alignment", UINT32, alignments[1])] + synthetic_metadata(draws, 1, total)]
    return [(draws.shuffled(metadata[i]), draws.shuffled(tensors[i])) for i in range(2)], alignments


def main(arguments):
    if (len(arguments) != 3 or not re.fullmatch("[0-9]+", arguments[0]) or not re.fullmatch("-?[0-9]+", arguments[1])
            or int(arguments[0]) >= 2 ** 64 or int(arguments[1]) not in EXPONENTS):
        print("usage: python3 oracle.py <seed> <eres> <new folder>   (seed: 0 to 2^64 - 1; eres: -100 to 100)\n"
              "It writes two synthetic GGUF shards and the sums file expected.txt into the new folder.",
              file=sys.stderr)
        return 2
    seed, eres, folder = int(arguments[0]), int(arguments[1]), os.path.abspath(arguments[2])
    try:
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        os.mkdir(folder, 0o700)  # fails if the name is taken; closed to other users
    except OSError as fault:
        print("oracle.py: cannot make the folder, which must not exist yet: %s" % fault, file=sys.stderr)
        return 2
    shards, alignments = synthetic_model(seed)
    for i, (metadata, tensors) in enumerate(shards):
        write_shard(os.path.join(folder, "oracle-%05d-of-00002.gguf" % (i + 1)), metadata, tensors, alignments[i])
    try:
        expected = sums_file(shards, eres)
    except Rejected as fault:
        print("oracle.py: at eres = %d the conversion rule rejects the synthetic model (%s). The two shards are "
              "written, and expected.txt is not. Every eres up to 20 is valid." % (eres, fault), file=sys.stderr)
        return fault.status
    with open(os.path.join(folder, "expected.txt"), "wb") as file:
        file.write(expected)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
