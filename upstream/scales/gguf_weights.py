# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Parts of this file are adapted from gguf-py, which is under the MIT License. That project's notice, which applies to
# those parts:
# MIT License
#
# Copyright (c) 2023 Georgi Gerganov
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
#
# Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License. That project's notice:
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

"""Read a GGUF model: its tensors as float32 torch tensors, and its shape parameters.

GGUF is the public container format of the ggml and llama.cpp projects. A model is a folder with
one or more GGUF files, its shards, and each tensor is wholly in one shard. The shape parameters
are the network's sizes and constants (width, number of layers, ...). scales/calibrate.py, which
computes a model's quantization scales, reads the model through this module. Typical use:

    model = open_model(folder)
    shape = shape_words(model)                              # a dict of the shape parameters
    w = model.tensor("blk.0.attn_q.weight")                 # shape (out, in)
    w = model.tensor("blk.1.ffn_up_exps.weight", expert=3)  # one expert's matrix, shape (out, in)

Model.tensor() reads only the tensor types F32 (plain float32), Q4_K, Q5_K and Q6_K (ggml's
quantized block formats with codes of 4, 5 and 6 bits) and the wider types Q8_0, F16 and BF16, and
shape_words() accepts only the architecture llama4. ModelError reports unusable input, but a malformed file may instead hang
gguf's reader or exhaust memory: see Model.__init__(). The module prints nothing.

The public package gguf (the folder gguf-py of github.com/ggml-org/llama.cpp) parses the container,
that is the header, the metadata and the tensor table. The package's decoder of the stored blocks,
gguf.quants.dequantize, runs in numpy on the CPU. This module decodes the blocks in torch instead,
so that a model of hundreds of gigabytes can be decoded on the GPU that runs the forward pass.
scales/tests/test_gguf_weights.py compares the two decoders bit for bit, this module's on the CPU
and, if torch finds one, on a CUDA device.

The decoder of the block formats Q4_K, Q5_K and Q6_K is adapted from the classes Q4_K, Q5_K and Q6_K in gguf/quants.py
of the package gguf, version 0.19.0. Like them it decodes whole arrays of blocks at once by reshaping, shifting and
masking, and it does the floating-point operations in the same order. It uses torch in place of numpy, and the three
formats share helper functions. The function _scales_and_minimums is adapted from Q4_K.get_scale_min of that file, but
its four expressions are those of ggml's get_scale_min_k4.

The byte layout of these three formats and the rule that decodes them are ggml's. They are defined in
ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K) and ggml/src/ggml-quants.c (get_scale_min_k4;
dequantize_row_q4_K, _q5_K, _q6_K) of llama.cpp (github.com/ggml-org/llama.cpp) as carried in llama-cpp-python 0.3.7,
whose changelog names the commit 794fe23f29fb40104975c91fe19f23798f7c726e. The package gguf and llama.cpp have the MIT
licence, and THIRD_PARTY.md has both notices in full.

The decoders of the wider types Q8_0, F16 and BF16 follow the same version of the package gguf. gguf/constants.py gives
the type numbers (GGMLQuantizationType: F16 = 1, Q8_0 = 8, BF16 = 30) and the sizes (GGML_QUANT_SIZES). _decode_q8_0 and
_decode_bf16 are adapted from the method dequantize_blocks of the classes Q8_0 and BF16 in gguf/quants.py, and
_decode_f16 does what the function dequantize of that file does for F16.
"""

import importlib.metadata
import operator
import os

import gguf
import numpy
import torch

GGUF_PY_VERSION = "0.19.0"  # open_model requires this version of the package gguf; the decoders are compared with it
ARCHITECTURE = "llama4"  # the value of the key general.architecture, and the prefix of the other metadata keys
REQUIRED = object()  # marks a key of SHAPE_KEYS without which the model is rejected
# The shape parameters that the metadata give. A row has the parameter's name in the result of shape_words, its type,
# its key after the prefix, and its value if the first shard does not have the key. The last two defaults are those of
# the class Llama4TextConfig of the library transformers, version 5.17.0 (github.com/huggingface/transformers), in
# src/transformers/models/llama4/configuration_llama4.py, lines 151 and 111. A float32 number in the file is widened
# exactly, so the float32 number nearest to 1e-5 gives 9.999999747378752e-06.
SHAPE_KEYS = (
    ("d", int, "embedding_length", REQUIRED),  # width of the residual stream
    ("L", int, "block_count", REQUIRED),  # number of layers
    ("dh", int, "attention.key_length", 128),  # head dimension
    ("eps", float, "attention.layer_norm_rms_epsilon", 1e-5),  # constant added to the mean square in an RMS norm
    ("theta", float, "rope.freq_base", 500000.0),  # base of the rotary position embedding (RoPE)
)
# The shape parameters that the tensors give. Each comes from the output dimension of the named
# matrix (for the embedding table, the number of rows). The module does not read the metadata keys
# for these sizes, because a computation uses the tensors as they are. A model without one of these
# tensors, for example a model without experts, is rejected. In the names, `shexp` is the shared
# expert and `ffn_gate_inp` the router. shape_words() looks at layers 0 and 1 only; check_tensors()
# in scales/calibrate.py checks the dimensions of every tensor that the forward pass reads.
SHAPE_TENSORS = (
    ("H", "blk.0.attn_q.weight"),  # query heads: the dimension divided by dh, rounded down
    ("G", "blk.0.attn_k.weight"),  # key and value heads: the dimension divided by dh, rounded down
    ("f", "blk.0.ffn_gate.weight"),  # hidden width of a dense feed-forward layer
    ("V", "token_embd.weight"),  # vocabulary size
    ("fe", "blk.1.ffn_gate_shexp.weight"),  # hidden width of one expert
    ("NE", "blk.1.ffn_gate_inp.weight"),  # routed experts of a mixture-of-experts layer
)
# Every second layer (1, 3, 5, ..) is a mixture-of-experts layer, as in
# docs/model.md, "2 Shape parameters". This is Llama4TextConfig's rule
# range(step - 1, layers, step) for a step of 2 (configuration_llama4.py, lines 186 to 196).
# shape_words() assumes MOE_INTERVAL and NO_ROPE_INTERVAL: it reads no metadata key for either.
MOE_INTERVAL = 2
# Every fourth layer (3, 7, 11, ..) has no rotary position embedding. This is Llama4TextConfig's default list,
# int((layer_idx + 1) % no_rope_layer_interval != 0) with an interval of 4 (the same file, lines 168 and 180 to 183).
NO_ROPE_INTERVAL = 4
# (key after the prefix, required value). float_forward.py sends each token to one routed expert.
# A model whose metadata specify a different number is rejected; one without the key is accepted.
EXPERTS_PER_TOKEN = ("expert_used_count", 1)


MESSAGE_LENGTH = 1000  # a ModelError's message is cut after this many characters


class ModelError(ValueError):
    """Unusable model files or gguf version, an unknown tensor name, or an invalid expert index.

    No message quotes a name, a key or a metadata value from inside a model file: whoever made the
    file chose them, and in a damaged file any bytes can be read as one. (A tensor's type and shape
    are given.) A message can quote a file's path or a caller's argument. So it has Python's escape
    for a backslash and for each character that is not printable ASCII, and it is cut after
    MESSAGE_LENGTH characters: printed, it stays one line of bounded length and sends no control
    sequence to a terminal.
    """

    def __init__(self, message):
        plain = (c if " " <= c <= "~" and c != "\\" else ascii(c)[1:-1] for c in message[: MESSAGE_LENGTH + 1])
        shown = "".join(plain)
        super().__init__(shown if len(shown) <= MESSAGE_LENGTH else shown[:MESSAGE_LENGTH] + " ... (cut)")


# The decoders. Each takes the blocks of a tensor as a uint8 tensor of shape (blocks, bytes per
# block) and returns their elements as a float32 tensor of shape (blocks, elements per block).
#
# Q4_K, Q5_K and Q6_K. A block contains 256 consecutive elements of one row, in 8 sub-blocks of 32
# elements (Q4_K, Q5_K) or 16 of 16 (Q6_K). An element's value is h * scale * code - h_min * minimum.
# The code is the element's own stored integer. The scale and the minimum are integers stored once
# per sub-block. h and h_min are binary16 numbers (IEEE 754 half precision) stored once per block.
# Q6_K has no h_min and no minimums.
#
# The byte layout of the block formats Q4_K, Q5_K and Q6_K, and the rule that decodes them, are those of ggml, the
# tensor library of the llama.cpp project (github.com/ggml-org/llama.cpp). They are defined in ggml/src/ggml-common.h
# (block_q4_K, block_q5_K, block_q6_K; lines 275 to 320) and in ggml/src/ggml-quants.c (get_scale_min_k4, lines 631 to
# 638; dequantize_row_q4_K, _q5_K, _q6_K, lines 1280 to 1302, 1482 to 1507 and 1690 to 1719). The line numbers are those
# of the copy of llama.cpp that the head of this file names. ggml and llama.cpp have the MIT licence. The file LICENSE
# of that copy says "Copyright (c) 2023-2024 The ggml authors"; THIRD_PARTY.md, section "ggml", has its text. The
# package gguf, from which the decoders below are adapted, has the MIT licence too. Its file LICENSE says
# "Copyright (c) 2023 Georgi Gerganov"; THIRD_PARTY.md, section "gguf-py", has its text.
# docs/model.md, "11.2 Stored blocks and their values", describes the layouts in words, and the
# code follows it.
#
# The decoded values are bit-identical on a CPU and on an NVIDIA GPU. The products are exact in
# float32, which has 24 significant bits: a binary16 number has at most 11, the magnitude of a scale
# at most 7 and that of a code at most 5 (the magnitudes 128 and 32 are powers of two and have 1).
# So only the subtraction in _values() rounds, and fusing it with the product would change nothing.
# No result is subnormal (a device may flush such a number to zero), because each is a multiple of
# 2^-24. A CPU and an NVIDIA GPU round a single float32 product or difference the same way, as
# IEEE 754 prescribes ("Floating Point and IEEE 754 Compliance for NVIDIA GPUs", section 5.4,
# https://docs.nvidia.com/cuda/floating-point/).


def _binary16(blocks, at):
    """Return the binary16 number at byte offset `at` of each block, widened exactly to float32.

    Shape: (blocks, 1).
    """
    return blocks[:, at : at + 2].contiguous().view(torch.float16).to(torch.float32)


def _half_bytes(packed, run):
    """Split bytes into 4-bit numbers: shape (blocks, n) gives shape (blocks, 2 n).

    Each group of `run` consecutive bytes gives the low halves of its bytes in order, then the high
    halves in order.
    """
    runs = packed.reshape(len(packed), -1, 1, run)
    return torch.cat((runs & 15, runs >> 4), dim=2).reshape(len(packed), -1)


def _bit_fields(packed, width):
    """Split bytes into `width`-bit numbers: shape (blocks, n) gives shape (blocks, 8 n / width).

    Each group of 32 consecutive bytes gives the lowest field of its bytes in order, then the next
    higher field, and so on.
    """
    runs = packed.reshape(len(packed), -1, 1, 32)
    shifts = torch.arange(0, 8, width, dtype=torch.uint8, device=packed.device).reshape(1, 1, -1, 1)
    return ((runs >> shifts) & (2**width - 1)).reshape(len(packed), -1)


def _scales_and_minimums(packed):
    """Unpack the 6-bit scale and the 6-bit minimum of each of the 8 sub-blocks of Q4_K and Q5_K.

    `packed` is the 12 bytes of each block that store them, shape (blocks, 12). Sub-blocks 0 to 3
    have their scale in the low 6 bits of bytes 0 to 3 and their minimum in the low 6 bits of bytes
    4 to 7. Sub-blocks 4 to 7 have the low 4 bits of both in bytes 8 to 11, the scale in the low
    half, and the high 2 bits in the top 2 bits of bytes 0 to 3 (scale) and 4 to 7 (minimum). The
    result is the pair (scales, minimums), each of shape (blocks, 8).

    The function is adapted from Q4_K.get_scale_min in gguf/quants.py of the package gguf. Its four expressions are
    those of ggml's get_scale_min_k4, applied to four sub-blocks at a time.
    """
    a, b, c = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    scales = torch.cat((a & 63, (c & 15) | ((a >> 6) << 4)), dim=1)
    minimums = torch.cat((b & 63, (c >> 4) | ((b >> 6) << 4)), dim=1)
    return scales, minimums


def _values(codes, steps, offsets=None):
    """Return steps * codes - offsets in float32.

    `codes` has shape (blocks, 256). `steps` and `offsets` have shape (blocks, sub-blocks).
    """
    values = codes.to(torch.float32).reshape(steps.shape[0], steps.shape[1], -1)
    values *= steps.unsqueeze(2)
    if offsets is not None:
        values -= offsets.unsqueeze(2)
    return values.reshape(len(codes), -1)


def _decode_q4_k(blocks):
    """Decode Q4_K blocks of 144 bytes.

    In order: h, h_min (binary16), 12 bytes of scales and minimums, 128 bytes with the 4-bit codes.
    """
    scales, minimums = _scales_and_minimums(blocks[:, 4:16])
    codes = _half_bytes(blocks[:, 16:144], 32)
    return _values(codes, _binary16(blocks, 0) * scales, _binary16(blocks, 2) * minimums)


def _decode_q5_k(blocks):
    """Decode Q5_K blocks of 176 bytes.

    As Q4_K, with 32 bytes for the fifth bit of each code before the 128 bytes of low 4 bits.
    """
    scales, minimums = _scales_and_minimums(blocks[:, 4:16])
    codes = _half_bytes(blocks[:, 48:176], 32) | (_bit_fields(blocks[:, 16:48], 1) << 4)
    return _values(codes, _binary16(blocks, 0) * scales, _binary16(blocks, 2) * minimums)


def _decode_q6_k(blocks):
    """Decode Q6_K blocks of 210 bytes.

    In order: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 signed 8-bit scales, and h
    (binary16). A code is the 6-bit number minus 32.
    """
    codes = (_half_bytes(blocks[:, 0:128], 64) | (_bit_fields(blocks[:, 128:192], 2) << 4)).to(torch.int8) - 32
    return _values(codes, _binary16(blocks, 208) * blocks[:, 192:208].view(torch.int8))


# Q8_0, F16 and BF16. A Q8_0 block holds 32 consecutive elements of one row in 34 bytes: a binary16 number h, then one
# signed 8-bit code per element. An element's value is h * code. An element of F16 is a binary16 number, and an element
# of BF16 is the upper 16 bits of a float32 number whose lower 16 bits are 0. Both are stored in 2 bytes, the low byte
# first, and the module treats each element as a block of its own. Its value is the stored number, widened exactly.
# These are the values of value8() and value16() in prover/weights_blocks/fold_rule.h.
#
# Public sources. The package gguf, version 0.19.0: gguf/constants.py, GGML_QUANT_SIZES (Q8_0: (32, 2 + 32); F16 and
# BF16: (1, 2)), and gguf/quants.py, Q8_0.dequantize_blocks, BF16.dequantize_blocks and the function dequantize (for
# F16). llama.cpp (github.com/ggml-org/llama.cpp), ggml-quants.h at tag b2261, read in a third party's public mirror,
# not in the project's own repository:
# "#define QK8_0 32", and the structure block_q8_0, whose two members are "ggml_fp16_t d;" and "int8_t  qs[QK8_0];"; the
# project's wiki page "Tensor Encoding Schemes" and Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf): "Each
# block has 32 weights", "w = q * block_scale". Hugging Face, the same page: F16, "16-bit standard IEEE 754
# half-precision floating-point number"; BF16, "16-bit shortened version of the 32-bit IEEE 754 single-precision
# floating-point number".
#
# Nothing rounds here, so these values too have the same bits on a CPU and on a GPU. The product of Q8_0 is exact in
# float32, because h has at most 11 significant bits and a code at most 8, and it is a multiple of 2^-24, so it is not
# subnormal. A BF16 element can be a subnormal float32 number. _decode_bf16 moves its bits as an integer, so that no
# floating-point setting of a device can turn it into 0.


def _decode_q8_0(blocks):
    """Q8_0, 34 bytes: h (binary16), 32 signed 8-bit codes. Adapted from Q8_0.dequantize_blocks in gguf/quants.py."""
    return _binary16(blocks, 0) * blocks[:, 2:34].view(torch.int8)


def _decode_f16(blocks):
    """F16, 2 bytes: one binary16 number (gguf/constants.py; IEEE 754). Does what dequantize in gguf/quants.py does for F16."""
    return _binary16(blocks, 0)


def _decode_bf16(blocks):
    """BF16, 2 bytes: the upper half of one float32 number. Adapted from BF16.dequantize_blocks in gguf/quants.py."""
    bits = blocks.view(torch.int16).to(torch.int32)
    bits <<= 16
    return bits.view(torch.float32)


# ggml's type number of plain float32
F32 = int(gguf.GGMLQuantizationType.F32)
# The block types that Model.tensor() decodes. ggml's type number: (bytes per block, decoder)
BLOCK_TYPES = {
    int(gguf.GGMLQuantizationType.Q4_K): (144, _decode_q4_k),
    int(gguf.GGMLQuantizationType.Q5_K): (176, _decode_q5_k),
    int(gguf.GGMLQuantizationType.Q6_K): (210, _decode_q6_k),
    int(gguf.GGMLQuantizationType.Q8_0): (34, _decode_q8_0),
    int(gguf.GGMLQuantizationType.F16): (2, _decode_f16),
    int(gguf.GGMLQuantizationType.BF16): (2, _decode_bf16),
}


class Model:
    """The opened shards of one model, as open_model() returns them. The files are never changed."""

    def __init__(self, paths):
        """Parse the container of each file in `paths`. metadata() reads the first file only."""
        self._first = None  # the first shard's reader
        self._tensors = {}  # tensor name: (path of its shard, gguf's record of the tensor)
        for path in paths:
            # gguf's reader maps the file read-only and reads GGUF versions 2 and 3 alike. The
            # format's specification is docs/gguf.md of github.com/ggml-org/ggml.
            # A model's files are defined in docs/model.md, "11.1 The container".
            # A malformed file may raise none of the exceptions below and instead keep the reader
            # running forever or make it allocate a very large amount of memory. What to check
            # before opening a downloaded file: scales/README.md, "A real model".
            # The message gives the kind of gguf's exception and not its text, which can quote a name
            # or a key from the file. "from None": otherwise a traceback would also print that text.
            try:
                reader = gguf.GGUFReader(path)
            except OSError as fault:
                raise ModelError(f"{path} cannot be read: {fault.strerror or type(fault).__name__}") from None
            except (ValueError, KeyError, IndexError, RecursionError) as fault:
                raise ModelError(f"{path} cannot be read as a GGUF file ({type(fault).__name__})") from None
            if reader.endianess != gguf.GGUFEndian.LITTLE or reader.byte_order != "I":
                raise ModelError(f"{path} or this machine is big-endian; both must be little-endian")
            for entry, record in enumerate(reader.tensors, 1):
                if record.name in self._tensors:
                    raise ModelError(
                        f"entry {entry} of the tensor table of {path} repeats the name of a tensor "
                        f"in {self._tensors[record.name][0]}"
                    )
                self._tensors[record.name] = (path, record)
            if self._first is None:
                self._first = reader

    def _find(self, name):
        if name not in self._tensors:
            raise ModelError(f"the model has no tensor {name}")
        return self._tensors[name]

    def names(self):
        """Return the names of the tensors of all shards, sorted."""
        return sorted(self._tensors)

    def info(self, name):
        """Return (ggml's type number, dimensions, shard's path, offset, size in bytes) of a tensor.

        The dimensions are in the file's order, the contiguous one first: for a weight matrix they
        are (in, out) or (in, out, experts). The offset, in bytes, counts from the start of the
        shard.
        """
        path, record = self._find(name)
        dims = tuple(int(n) for n in record.shape)
        return int(record.tensor_type), dims, path, record.data_offset, record.n_bytes

    def tensor(self, name, expert=None, device="cpu"):
        """Return a tensor's decoded values as a new float32 torch tensor on `device`.

        The shape is the file's dimensions reversed: (n,) for a vector, (out, in) for a matrix and
        (experts, out, in) for a stacked tensor. If `expert` is an integer, the result is that
        expert's matrix only, and only its bytes are read, so that the experts of a large model
        never need to be in memory all at once. Raise ModelError for an unknown name, an unsupported
        type, a tensor without elements, an `expert` that is not an integer or does not fit the
        tensor, or a decoded value that is not finite.
        """
        path, record = self._find(name)
        number = int(record.tensor_type)
        if number != F32 and number not in BLOCK_TYPES:
            raise ModelError(
                f"the tensor {name} in {path} has the type {record.tensor_type.name}; "
                "only the types F32, Q4_K, Q5_K, Q6_K, Q8_0, F16 and BF16 are supported"
            )
        # gguf gives the stored bytes as an array in numpy's order of dimensions, the reverse of the
        # file's. For a block type the array's last axis counts bytes, so the shape in elements
        # comes from the file's dimensions.
        # (F16 is the exception: gguf gives an array of binary16 numbers, which the view below turns into bytes.)
        stored, shape = record.data, tuple(int(n) for n in record.shape)[::-1]
        if 0 in shape:
            raise ModelError(f"the tensor {name} in {path} has no elements")
        if expert is not None:
            if len(shape) != 3:
                raise ModelError(f"the tensor {name} is not a stack of experts: it has the shape {shape}")
            try:
                expert = operator.index(expert)
            except TypeError as fault:
                raise ModelError(f"the expert index must be an integer, not {expert!r}") from fault
            if not 0 <= expert < shape[0]:
                raise ModelError(f"the tensor {name} has the experts 0 to {shape[0] - 1}, not an expert {expert}")
            stored, shape = stored[expert], shape[1:]
        # numpy.array copies the bytes out of the read-only mapping, so that the result shares no
        # memory with the file.
        values = torch.from_numpy(numpy.array(stored)).to(device)
        if number != F32:
            block_bytes, decode = BLOCK_TYPES[number]
            values = decode(values.view(torch.uint8).reshape(-1, block_bytes)).reshape(shape)
        if not torch.isfinite(values).all():
            raise ModelError(f"the tensor {name} in {path} has a value that is infinite or not a number")
        return values

    def metadata(self, key, default=None):
        """Return the value of a metadata key of the first shard as a Python object.

        Return `default` if the key is absent.
        """
        field = self._first.get_field(key)
        try:
            return default if field is None else field.contents()
        except (ValueError, IndexError) as fault:  # contents() is where gguf decodes the value's bytes
            kind = type(fault).__name__  # not the exception's text; "from None": see Model.__init__()
            raise ModelError(f"the value of the first shard's metadata key {key} cannot be read ({kind})") from None


def open_model(folder):
    """Open a model. Its shards are the files in `folder` whose names end in ".gguf".

    The first shard, from which the metadata are read, is the first in the order of the file names,
    compared as bytes. The installed package gguf must have the version GGUF_PY_VERSION.
    """
    try:
        names = sorted((name for name in os.listdir(folder) if name.endswith(".gguf")), key=os.fsencode)
    except OSError as fault:
        raise ModelError(f"the folder {folder} cannot be read: {fault.strerror}") from fault
    if not names:
        raise ModelError(f'the folder {folder} has no file with a name that ends in ".gguf"')
    try:
        installed = importlib.metadata.version("gguf")
    except importlib.metadata.PackageNotFoundError:
        installed = "unknown"
    if installed != GGUF_PY_VERSION:
        raise ModelError(f"the installed package gguf has the version {installed}; install version {GGUF_PY_VERSION}")
    return Model([os.path.join(folder, name) for name in names])


def _outputs(model, name):
    """Return the output dimension of a weight matrix."""
    dims = model.info(name)[1]
    if len(dims) != 2:
        raise ModelError(f"the tensor {name} is not a matrix: it has the dimensions {dims}")
    return dims[1]


def shape_words(model):
    """Return the shape parameters as a dict.

    Its keys are d, dh, L, H, G, f, fe, NE, V, eps, theta, moe and rope. `moe` is the list of the
    mixture-of-experts layers and `rope` the list of the layers with a rotary position embedding,
    both in ascending order. SHAPE_KEYS and SHAPE_TENSORS explain the other keys.
    """
    architecture = model.metadata("general.architecture")
    if architecture != ARCHITECTURE:
        raise ModelError(f"the model's general.architecture is not {ARCHITECTURE!r}, the only one that is supported")
    shape = {}
    for parameter, kind, key, if_absent in SHAPE_KEYS:
        value = model.metadata(f"{ARCHITECTURE}.{key}", if_absent)
        if value is REQUIRED:
            raise ModelError(f"the model's first shard has no metadata key {ARCHITECTURE}.{key}")
        if type(value) is not kind or (kind is int and value < 1):
            wanted = "a positive integer" if kind is int else "a floating-point number"
            raise ModelError(f"the metadata key {ARCHITECTURE}.{key} of the model's first shard must be {wanted}")
        shape[parameter] = value
    key, wanted = EXPERTS_PER_TOKEN
    if model.metadata(f"{ARCHITECTURE}.{key}", wanted) != wanted:
        raise ModelError(f"the key {ARCHITECTURE}.{key} is not {wanted}; only {wanted} expert per token is supported")
    layers = range(shape["L"])
    shape["moe"] = [layer for layer in layers if (layer + 1) % MOE_INTERVAL == 0]
    shape["rope"] = [layer for layer in layers if (layer + 1) % NO_ROPE_INTERVAL != 0]
    for parameter, name in SHAPE_TENSORS:
        shape[parameter] = _outputs(model, name)
    shape["H"] //= shape["dh"]
    shape["G"] //= shape["dh"]
    return shape
