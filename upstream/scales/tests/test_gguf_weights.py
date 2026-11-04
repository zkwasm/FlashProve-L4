#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Test scales/gguf_weights.py: block decoders, models of several shards, shape_words(), rejections.

usage: python3 test_gguf_weights.py      (no arguments)

The test writes small GGUF files (ggml's model file format; a model's files are its shards) into
a temporary folder with the public package gguf and reads them through the module. It draws the
integers of each block of 256 elements (factors, scales, minimums, codes), packs them into bytes
itself (pack()) and computes the values one element at a time (values()) by the formulas of
docs/model.md, "11.2 Stored blocks and their values". It compares the
module's tensors, bit for bit, with these values and with the output of
gguf.quants.dequantize; where there is a CUDA device, it repeats this with the tensors on it.
The block formats are ggml's. The head of gguf_weights.py names the files that define them, and
THIRD_PARTY.md, section "ggml", has the licence.

For the wider types Q8_0, F16 and BF16 the test does the same on three models, one per type, in
which every matrix has that type and only the norm and the router are F32. It also reads one file
that holds every finite 16-bit pattern of F16 and of BF16, and every finite scale of Q8_0 with every
code. The layouts are those of the package gguf, version 0.19.0 (gguf/constants.py,
GGML_QUANT_SIZES; gguf/quants.py, the classes Q8_0 and BF16 and the function dequantize).

Two more cases read the small model, 0.5 GB of synthetic weights, which the prover's test tool
make_model writes from the arguments in prover/examples/target_small_file_f2/model.txt. They are
skipped unless the environment variable SMALL_MODEL names that model's folder or, if it is not
set, MAKE_MODEL names the make_model executable: the test then makes the model in the temporary
folder.

Exit status: 0 if every case ran and passed; the last line then begins with "OK". 1 if a case
failed or an error stopped the test. 77 if none failed but some did not run: the last line begins
with "in part" if the small model's cases were skipped, "not run" if an import failed and none ran.
Without a CUDA device the comparisons on it do not run, and that is not counted as skipped.
"""

import contextlib
import importlib.metadata
import io
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest.mock

PASSED, FAILED, NOT_ALL_RUN = 0, 1, 77  # the exit statuses
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True  # the import of gguf_weights leaves no __pycache__ in the tree
try:
    import gguf
    import gguf_weights
    import numpy
    import torch
    from gguf.quants import dequantize
except ImportError as fault:
    print(f"not run: {fault}")
    sys.exit(NOT_ALL_RUN)

Q4_0, Q4_K, Q5_K, Q6_K = (gguf.GGMLQuantizationType[name] for name in ("Q4_0", "Q4_K", "Q5_K", "Q6_K"))
Q8_0, F16, BF16 = (gguf.GGMLQuantizationType[name] for name in ("Q8_0", "F16", "BF16"))
# For each block type: the number of sub-blocks in a block, the range of a sub-block's scale and
# the range of a code.
FIELDS = {Q4_K: (8, (0, 63), (0, 15)), Q5_K: (8, (0, 63), (0, 31)), Q6_K: (16, (-128, 127), (-32, 31))}
f32 = numpy.float32
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]  # the values are compared on each of these


def draw_blocks(rng, kind, count):
    """Return `count` blocks of type `kind`, each as integers (h, h_min, scales, minimums, codes).

    h and h_min are the bits of the block's two binary16 factors, which are finite. The first block
    has -0.0 and 0.0 there and every other field at its lowest value. The second has the largest
    and the most negative number and every other field at its highest value. The third has plus
    and minus the smallest subnormal. All else is random. Q6_K has no h_min or minimums: pack() and
    values() ignore them for that type.
    """

    def finite_binary16():
        bits = rng.getrandbits(16)
        return bits if (bits >> 10) & 31 != 31 else bits ^ (1 << 10)  # an exponent field of 31 means infinity or NaN

    sub_blocks, of_scale, of_code = FIELDS[kind]
    blocks = []
    chosen = [(min, 0x8000, 0x0000), (max, 0x7BFF, 0xFBFF), (rng.randint, 0x0001, 0x8001)]
    for n in range(count):
        pick, h, h_min = chosen[n] if n < 3 else (rng.randint, finite_binary16(), finite_binary16())
        scales = [pick(*of_scale) for _ in range(sub_blocks)]
        minimums = [pick(0, 63) for _ in range(sub_blocks)]
        codes = [pick(*of_code) for _ in range(256)]
        blocks.append((h, h_min, scales, minimums, codes))
    return blocks


def pack(kind, block):
    """Return the block's bytes in ggml's layout for the type `kind`.

    Q4_K (144 bytes): h, h_min, 6-bit scales and minimums, 4-bit codes.
    Q5_K (176): the same, with the codes' fifth bits before their low four bits.
    Q6_K (210): the low four bits of each code + 32, then the high two bits, signed scales, h.
    In full: docs/model.md, "11.2 Stored blocks and their values".
    """
    h, h_min, scales, minimums, codes = block
    if kind == Q6_K:
        out = bytearray(210)
        for e, code in enumerate(codes):
            a, t, c = e // 128, e % 128 // 32, e % 32
            out[64 * a + 32 * (t % 2) + c] |= ((code + 32) & 15) << (4 * (t // 2))
            out[128 + 32 * a + c] |= ((code + 32) >> 4) << (2 * t)
        out[192:208] = struct.pack("16b", *scales)
        out[208:210] = struct.pack("<H", h)
        return bytes(out)
    low_bits_at = 16 if kind == Q4_K else 48
    out = bytearray(low_bits_at + 128)
    out[0:4] = struct.pack("<2H", h, h_min)
    for j in range(4):
        out[4 + j] = scales[j] | ((scales[j + 4] >> 4) << 6)
        out[8 + j] = minimums[j] | ((minimums[j + 4] >> 4) << 6)
        out[12 + j] = (scales[j + 4] & 15) | ((minimums[j + 4] & 15) << 4)
    for e, code in enumerate(codes):
        j, i = e // 32, e % 32
        out[low_bits_at + 32 * (j // 2) + i] |= (code & 15) << (4 * (j % 2))
        if kind == Q5_K:
            out[16 + i] |= (code >> 4) << j
    return bytes(out)


def values(kind, block):
    """Return the block's 256 values, h * scale * code - h_min * minimum.

    Each numpy.float32 operation rounds once.
    """

    def widened(bits):
        return f32(numpy.uint16(bits).view(numpy.float16))

    h, h_min, scales, minimums, codes = block
    out = []
    for e, code in enumerate(codes):
        s = e * len(scales) // 256
        value = widened(h) * f32(scales[s]) * f32(code)
        out.append(value if kind == Q6_K else value - widened(h_min) * f32(minimums[s]))
    return numpy.array(out, dtype=f32)


def block_tensor(rng, kind, shape):
    """Draw a tensor of type `kind`. `shape` is in numpy's order, its last entry a multiple of 256.

    The result is (the type, the stored bytes as gguf's writer takes them, the expected values).
    """
    blocks = draw_blocks(rng, kind, math.prod(shape) // 256)
    stored = numpy.frombuffer(b"".join(pack(kind, block) for block in blocks), dtype=numpy.uint8)
    expected = numpy.concatenate([values(kind, block) for block in blocks])
    return kind, stored.reshape(*shape[:-1], -1), expected.reshape(shape)


def value_of_16_bits(kind, bits):
    """Return the value of a finite bit pattern of F16 (binary16) or BF16, computed from its three fields.

    binary16 has 5 bits of exponent and 10 of fraction. BF16 has float32's 8 bits of exponent and the upper 7 of its
    23 bits of fraction. The result, a Python float, is exact, and it is a float32 number.
    """
    fraction_bits, bias = (10, 15) if kind == F16 else (7, 127)
    exponent, fraction = (bits & 0x7FFF) >> fraction_bits, bits & (2**fraction_bits - 1)
    if exponent == 0:
        size = math.ldexp(fraction, 1 - bias - fraction_bits)
    else:
        size = math.ldexp(2**fraction_bits + fraction, exponent - bias - fraction_bits)
    return -size if bits >> 15 else size


def finite_patterns(kind):
    """Return all 16-bit patterns of F16 or BF16 that are finite numbers, in ascending order."""
    infinity = 0x7C00 if kind == F16 else 0x7F80  # a pattern without its sign bit is finite if it is below this one
    return [bits for bits in range(2**16) if bits & 0x7FFF < infinity]


def wide_tensor(rng, kind, shape):
    """Draw a tensor of the type Q8_0, F16 or BF16. "shape" is in numpy's order; its last entry is a multiple of 32.

    The first 16-bit numbers (for Q8_0 the scales of the first blocks) are -0.0, the least subnormal number, the largest
    and the most negative number. The first two blocks of Q8_0 have all codes at -128, the next two at 127. Everything
    else is drawn uniformly from the finite patterns and from the codes -128 to 127. Each expected value is computed by
    itself in Python's binary64 arithmetic, where the product of Q8_0 is exact. The result is as that of block_tensor.
    """
    finite = finite_patterns(F16 if kind == Q8_0 else kind)
    numbers = math.prod(shape) // (32 if kind == Q8_0 else 1)
    patterns = [0x8000, 0x0001, finite[len(finite) // 2 - 1], finite[-1]] + rng.choices(finite, k=numbers - 4)
    if kind == Q8_0:
        codes = [[-128] * 32, [-128] * 32, [127] * 32, [127] * 32]
        codes += [rng.choices(range(-128, 128), k=32) for _ in range(numbers - 4)]
        stored = b"".join(struct.pack("<H32b", h, *of_block) for h, of_block in zip(patterns, codes))
        expected = [value_of_16_bits(F16, h) * code for h, of_block in zip(patterns, codes) for code in of_block]
    else:
        stored = struct.pack(f"<{numbers}H", *patterns)
        expected = [value_of_16_bits(kind, bits) for bits in patterns]
    stored = numpy.frombuffer(stored, dtype=numpy.uint8).reshape(*shape[:-1], -1)
    return kind, stored, numpy.array(expected, dtype=numpy.float64).astype(f32).reshape(shape)


def every_pattern():
    """Return three tensors: all finite patterns of F16, all of BF16, and each finite scale of Q8_0 with each code.

    A row of the third tensor has one scale, in eight blocks that hold the codes -128 to 127 in order.
    """
    tensors = {}
    for kind in (F16, BF16):
        finite = finite_patterns(kind)
        stored = numpy.array(finite, dtype="<u2").view(numpy.uint8).reshape(-1, 512)
        expected = numpy.array([value_of_16_bits(kind, bits) for bits in finite], dtype=numpy.float64)
        tensors[f"every.{kind.name}"] = (kind, stored, expected.astype(f32).reshape(-1, 256))
    scales, codes = finite_patterns(F16), numpy.arange(-128, 128)
    stored = numpy.zeros((len(scales), 8, 34), dtype=numpy.uint8)
    stored[:, :, 0:2] = numpy.array(scales, dtype="<u2").view(numpy.uint8).reshape(-1, 1, 2)
    stored[:, :, 2:34] = codes.astype(numpy.int8).view(numpy.uint8).reshape(1, 8, 32)
    expected = numpy.array([value_of_16_bits(F16, bits) for bits in scales], dtype=numpy.float64).reshape(-1, 1) * codes
    tensors["every.Q8_0"] = (Q8_0, stored.reshape(len(scales), -1), expected.astype(f32))
    return tensors


def float_tensor(rng, shape):
    """Draw a finite F32 tensor (type None) with -0.0, the smallest subnormal and the maximum.

    The rest are random bit patterns. The tensor needs at least 3 elements.
    """
    bits = [0x80000000, 0x00000001, 0x7F7FFFFF] + [rng.getrandbits(32) for _ in range(math.prod(shape) - 3)]
    bits = numpy.array([b if (b >> 23) & 255 != 255 else b ^ (1 << 23) for b in bits], dtype=numpy.uint32)
    return None, bits.view(f32).reshape(shape), bits.view(f32).reshape(shape)


def write_shard(path, metadata, tensors, architecture="llama4", endianess=gguf.GGUFEndian.LITTLE):
    """Write one GGUF file.

    `metadata` is {key: a str, int, float, bool or list}; `tensors` is {name: (type, stored, _)}.
    Type None: gguf's writer goes by the array's dtype.
    """
    writer = gguf.GGUFWriter(path, architecture, endianess=endianess)
    add = {str: writer.add_string, int: writer.add_uint32, float: writer.add_float32, list: writer.add_array}
    add[bool] = writer.add_bool
    for key, value in metadata.items():
        add[type(value)](key, value)
    for name, (kind, stored, _) in tensors.items():
        writer.add_tensor(name, stored, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def bytes_of(path):
    """Return a file's bytes."""
    with open(path, "rb") as f:
        return f.read()


def same_bits(tensor, expected):
    """Say whether `tensor` is float32 with the shape and bits of the float32 array `expected`."""
    found = tensor.cpu().numpy()
    if found.dtype != f32 or found.shape != expected.shape:
        return False
    return numpy.array_equal(found.view(numpy.uint32), numpy.ascontiguousarray(expected).view(numpy.uint32))


def rejected(call, *arguments):
    """Say whether the call raises a ModelError that is a ValueError and has a message."""
    try:
        call(*arguments)
    except gguf_weights.ModelError as fault:
        return isinstance(fault, ValueError) and str(fault) != ""
    return False


# Text that whoever makes a model file can put into it, or give the file as its name: an escape
# sequence that clears a terminal, a carriage return, a line feed and a line of its own.
HOSTILE = "\x1b[2J\r\nforged line"
LONGEST = 1010  # characters in a message: gguf_weights.MESSAGE_LENGTH and the mark of a cut


def harmless(call, *arguments, in_a_path=False):
    """Say whether the call prints nothing and raises a ModelError that is safe to print.

    Safe: the message is not empty, has printable ASCII characters only and at most LONGEST of
    them, does not hold the word "forged" of HOSTILE (unless in_a_path: a path is quoted), and no other
    exception is attached whose text a traceback would print. What the call writes to sys.stdout or
    sys.stderr is caught and must be empty; by default the module logging sends a warning to
    sys.stderr.
    """
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            call(*arguments)
    except gguf_weights.ModelError as fault:
        text = str(fault)
        plain = 0 < len(text) <= LONGEST and all(" " <= c <= "~" for c in text) and (in_a_path or "forged" not in text)
        alone = fault.__cause__ is None and (fault.__context__ is None or fault.__suppress_context__)
        return plain and alone and printed.getvalue() == ""
    return False


METADATA = {
    "general.name": "synthetic",
    "llama4.embedding_length": 1024,
    "llama4.block_count": 5,
    "llama4.attention.key_length": 2,
    "llama4.attention.layer_norm_rms_epsilon": 1e-6,
    "llama4.rope.freq_base": 10000.0,
    "llama4.expert_used_count": 1,
}
# Keys that shape_words() ignores (see gguf_weights.SHAPE_TENSORS, MOE_INTERVAL). Their value, 77,
# is wrong on purpose: using one would change the shape parameters, which must equal SHAPE.
NOT_READ = ("attention.head_count", "attention.head_count_kv", "feed_forward_length", "expert_feed_forward_length")
NOT_READ += ("expert_count", "vocab_size", "interleave_moe_layer_step")
METADATA.update({"llama4." + key: 77 for key in NOT_READ})
SHAPE = {"d": 1024, "dh": 2, "L": 5, "H": 4, "G": 2, "f": 6, "fe": 2, "NE": 3, "V": 8, "theta": 10000.0}
SHAPE.update({"eps": float(f32(1e-6)), "moe": [1, 3], "rope": [0, 1, 2, 4]})  # the file holds eps as a float32 number


def synthetic_cases(work, check):
    """Run the cases on files written under the folder `work`; check(name, passed) records each."""
    rng = random.Random(1)
    first = {
        "token_embd.weight": block_tensor(rng, Q6_K, (8, 1024)),
        "blk.0.attn_norm.weight": float_tensor(rng, (1024,)),
        "blk.0.attn_q.weight": block_tensor(rng, Q4_K, (8, 1024)),
        "blk.0.attn_k.weight": block_tensor(rng, Q5_K, (4, 1024)),
        "blk.0.ffn_gate.weight": block_tensor(rng, Q5_K, (6, 1024)),
        "blk.1.ffn_gate_inp.weight": float_tensor(rng, (3, 1024)),
    }
    second = {
        "blk.1.ffn_down_exps.weight": block_tensor(rng, Q4_K, (3, 4, 256)),
        "blk.1.ffn_gate_exps.weight": block_tensor(rng, Q6_K, (3, 2, 1024)),
        "blk.1.ffn_gate_shexp.weight": block_tensor(rng, Q4_K, (2, 1024)),
        "blk.1.ffn_up_exps.weight": block_tensor(rng, Q5_K, (3, 2, 1024)),
        "blk.1.other_exps.weight": float_tensor(rng, (3, 2, 8)),
    }
    both = {**first, **second}
    other = {"other.weight": (Q4_0, numpy.zeros((2, 18), dtype=numpy.uint8), None)}  # a block of Q4_0 has 18 bytes
    good = os.path.join(work, "good")
    shards = [os.path.join(good, "synthetic-0000%d-of-00002.gguf" % n) for n in (1, 2)]
    os.mkdir(good)
    write_shard(shards[0], METADATA, first)
    write_shard(shards[1], {"general.name": "second shard"}, {**second, **other})
    with open(os.path.join(good, "notes.txt"), "w") as f:
        f.write("The module reads only the files with names that end in .gguf.\n")
    model = gguf_weights.open_model(good)

    def of_type(kind):
        return [(name, stored, expected) for name, (its_kind, stored, expected) in both.items() if its_kind == kind]

    stacked = [name for name in both if name.endswith("_exps.weight")]
    for device in DEVICES:

        def read(name, expert=None):
            found = model.tensor(name, expert=expert, device=device)
            return found if found.device.type == device else None

        on = f"-on-{device}"
        check("F32-values-as-stored" + on, all(same_bits(read(name), e) for name, _, e in of_type(None)))
        for kind in (Q4_K, Q5_K, Q6_K):
            found = [(read(name), stored, expected) for name, stored, expected in of_type(kind)]
            check(kind.name + "-values-as-computed-from-the-integers" + on, all(same_bits(t, e) for t, _, e in found))
            as_gguf = all(same_bits(t, dequantize(s, kind)) for t, s, _ in found)
            check(kind.name + "-values-as-gguf-decodes-them" + on, as_gguf)
        check(
            "one-expert-is-that-slice-of-the-stacked-tensor" + on,
            all(same_bits(read(name, e), both[name][2][e]) for name in stacked for e in (0, 1, numpy.int64(2))),
        )
    check("names-of-all-shards-sorted", model.names() == sorted([*both, *other]))

    as_in_file = True
    for shard, tensors in zip(shards, (first, second)):
        for name, (kind, stored, expected) in tensors.items():
            number, dims, path, offset, size = model.info(name)
            with open(path, "rb") as f:
                f.seek(offset)
                as_in_file &= f.read(size) == stored.tobytes() and path == shard
            as_in_file &= number == (0 if kind is None else int(kind)) and dims == expected.shape[::-1]
            as_in_file &= all(type(n) is int for n in (number, *dims, offset, size))
    check("info-finds-each-tensor's-bytes", as_in_file)

    before = [bytes_of(shard) for shard in shards]
    for name in both:
        model.tensor(name).zero_()
    unchanged = all(same_bits(model.tensor(name), both[name][2]) for name in both)
    check("a-result-shares-no-memory-with-the-file", unchanged and before == [bytes_of(shard) for shard in shards])

    check("metadata-is-the-first-shard's", model.metadata("general.name") == "synthetic")
    check("metadata-default", model.metadata("general.none") is None and model.metadata("general.none", 7) == 7)

    check("info-describes-a-tensor-of-a-type-that-is-not-decoded", model.info("other.weight")[0] == int(Q4_0))
    check("tensor-rejects-a-type-that-is-not-decoded", rejected(model.tensor, "other.weight"))
    check("rejects-a-name-that-the-model-lacks", rejected(model.tensor, "none") and rejected(model.info, "none"))
    check("rejects-an-expert-of-a-matrix", rejected(model.tensor, "token_embd.weight", 0))
    check("rejects-an-expert-out-of-range", all(rejected(model.tensor, stacked[0], e) for e in (-1, 3)))
    check("rejects-an-expert-that-is-not-an-integer", all(rejected(model.tensor, stacked[0], e) for e in (1.0, "1")))

    def folder_with(name, *files):
        """Make a folder under `work` and return its path. `files` are pairs (file name, bytes)."""
        os.mkdir(os.path.join(work, name))
        for file_name, content in files:
            with open(os.path.join(work, name, file_name), "wb") as f:
                f.write(content)
        return os.path.join(work, name)

    # A block whose h is infinite, and an F32 tensor with a NaN. Each has finite neighbours.
    infinite = [(0x7C00, *block[1:]) if n == 1 else block for n, block in enumerate(draw_blocks(rng, Q4_K, 3))]
    infinite = numpy.frombuffer(b"".join(pack(Q4_K, block) for block in infinite), dtype=numpy.uint8).reshape(3, -1)
    not_finite = {"blocks.weight": (Q4_K, infinite, None), "floats.weight": (None, f32([1, 2, math.nan, 4]), None)}
    empty = {"empty.weight": (Q4_K, numpy.zeros((0, 144), dtype=numpy.uint8), None)}
    write_shard(os.path.join(folder_with("not-finite"), "a.gguf"), METADATA, {**not_finite, **empty})
    opened = gguf_weights.open_model(os.path.join(work, "not-finite"))
    check("tensor-rejects-a-value-that-is-not-finite", all(rejected(opened.tensor, name) for name in not_finite))
    check("tensor-rejects-a-tensor-without-elements", rejected(opened.tensor, "empty.weight"))

    shard = bytes_of(shards[0])
    version_2 = gguf_weights.open_model(folder_with("version-2", ("a.gguf", shard[:4] + b"\2" + shard[5:])))
    check("reads-version-2-as-version-3", all(same_bits(version_2.tensor(name), first[name][2]) for name in first))
    check("rejects-a-folder-that-does-not-exist", rejected(gguf_weights.open_model, os.path.join(work, "none")))
    check("rejects-a-folder-without-shards", rejected(gguf_weights.open_model, folder_with("bare", ("a.txt", shard))))
    faulty = {
        "that-is-empty": b"",
        "without-the-magic": b"\0" * 4 + shard[4:],
        "of-version-1": shard[:4] + b"\1" + shard[5:],
        "of-version-4": shard[:4] + b"\4" + shard[5:],
        "cut-after-the-counts": shard[:24],
        "cut-in-the-header": shard[:100],
        "cut-in-the-data": shard[:-100],
        "with-a-key-twice": shard.replace(b"llama4.attention.head_count", b"llama4.attention.key_length"),
        "with-too-large-a-tensor-count": shard[:8] + struct.pack("<Q", len(first) + 1) + shard[16:],
    }
    for name, content in faulty.items():
        check("rejects-a-file-" + name, rejected(gguf_weights.open_model, folder_with(name, ("a.gguf", content))))
    write_shard(os.path.join(folder_with("big-endian"), "a.gguf"), METADATA, first, endianess=gguf.GGUFEndian.BIG)
    check("rejects-a-big-endian-file", rejected(gguf_weights.open_model, os.path.join(work, "big-endian")))
    os.mkdir(os.path.join(folder_with("folder"), "a.gguf"))
    check("rejects-a-folder-in-the-place-of-a-file", rejected(gguf_weights.open_model, os.path.join(work, "folder")))
    twice = folder_with("twice", ("a.gguf", shard), ("b.gguf", shard))
    check("rejects-a-tensor-name-in-two-shards", rejected(gguf_weights.open_model, twice))
    not_text = folder_with("not-text", ("a.gguf", shard.replace(b"synthetic", b"synthet\xff\xff")))
    check("rejects-a-string-that-is-not-UTF-8", rejected(gguf_weights.open_model(not_text).metadata, "general.name"))

    def no_record(package):
        raise importlib.metadata.PackageNotFoundError(package)

    with unittest.mock.patch.object(gguf_weights, "GGUF_PY_VERSION", "0.0.0"):
        check("rejects-another-version-of-the-package-gguf", rejected(gguf_weights.open_model, good))
    with unittest.mock.patch.object(importlib.metadata, "version", no_record):
        check("rejects-a-package-gguf-of-unknown-version", rejected(gguf_weights.open_model, good))

    def shape_of(name, metadata, tensors, architecture="llama4"):
        write_shard(os.path.join(folder_with(name), "a.gguf"), metadata, tensors, architecture)
        return gguf_weights.shape_words(gguf_weights.open_model(os.path.join(work, name)))

    def without(table, *keys):
        return {key: value for key, value in table.items() if key not in keys}

    check("shape", gguf_weights.shape_words(model) == SHAPE)
    required = ("llama4.embedding_length", "llama4.block_count")
    least = shape_of("least", {key: METADATA[key] for key in required}, both)
    defaults = {"dh": 128, "H": 8 // 128, "G": 4 // 128, "eps": 1e-5, "theta": 500000.0}
    check("shape-from-the-required-keys-only", least == {**SHAPE, **defaults})
    one_layer = shape_of("one-layer", {**METADATA, "llama4.block_count": 1}, both)
    check("shape-with-a-block-count-of-1", one_layer == {**SHAPE, "L": 1, "moe": [], "rope": [0]})
    check(
        "rejects-a-shape-without-a-required-key",
        all(rejected(shape_of, "no-key-%d" % n, without(METADATA, key), both) for n, key in enumerate(required)),
    )
    sized = [name for name in both if name != "blk.0.attn_norm.weight" and not name.endswith("_exps.weight")]
    check(
        "rejects-a-shape-without-a-tensor-that-gives-a-size",
        all(rejected(shape_of, "no-size-%d" % n, METADATA, without(both, name)) for n, name in enumerate(sized)),
    )
    wrong = {key: ([4, 4], 0, 4.0, True) for key in ("embedding_length", "block_count", "attention.key_length")}
    wrong.update({key: ([1.0, 1.0], 1, True) for key in ("attention.layer_norm_rms_epsilon", "rope.freq_base")})
    for key, values_of_key in wrong.items():
        changed = [{**METADATA, "llama4." + key: value} for value in values_of_key]
        check(
            "rejects-a-value-of-the-wrong-kind-for-" + key,
            all(rejected(shape_of, f"{key}-{n}", metadata, both) for n, metadata in enumerate(changed)),
        )
    vector = {**both, "blk.0.attn_q.weight": float_tensor(rng, (8,))}
    check("rejects-a-vector-where-a-matrix-gives-a-size", rejected(shape_of, "vector", METADATA, vector))
    check("rejects-another-architecture", rejected(shape_of, "llama", METADATA, both, "llama"))
    two_experts = {**METADATA, "llama4.expert_used_count": 2}
    check("rejects-two-experts-per-token", rejected(shape_of, "two-experts", two_experts, both))

    # What the maker of a file chooses must not reach a message as it is. In a shard's bytes, a name
    # or a key is replaced by text of the same length, so that the rest of the header stays valid.
    def hostile(old):
        return (HOSTILE.encode() + b"w" * len(old))[: len(old)]

    q, k = b"blk.0.attn_q.weight", b"blk.0.attn_k.weight"
    count, length = b"llama4.attention.head_count", b"llama4.attention.key_length"
    named = shard.replace(b"token_embd.weight", hostile(b"token_embd.weight"))  # the first entry of the tensor table
    files = {
        "a-name-in-two-shards": [("a.gguf", named), ("b.gguf", named)],
        "a-name-twice-in-one-shard": [("a.gguf", shard.replace(q, hostile(q)).replace(k, hostile(k)))],
        "a-key-twice": [("a.gguf", shard.replace(count, hostile(count)).replace(length, hostile(length)))],
        "a-file-name": [(HOSTILE + ".gguf", b"not a GGUF file")],
    }
    in_place = all(HOSTILE.encode() in content or HOSTILE in name for case in files.values() for name, content in case)
    check("control-each-of-these-files-has-the-hostile-text", in_place)
    for name, case in files.items():
        folder = folder_with("safe-" + name, *case)
        safe = harmless(gguf_weights.open_model, folder, in_a_path=name == "a-file-name")
        check("safe-to-print-the-message-on-" + name, safe)
    check("safe-to-print-the-message-on-a-value", harmless(gguf_weights.open_model(not_text).metadata, "general.name"))
    check("safe-to-print-the-message-on-an-architecture", harmless(shape_of, "hostile", METADATA, both, HOSTILE))

    def fails_at_length():
        raise gguf_weights.ModelError(HOSTILE + "\\" + "w" * 5000)

    check("safe-to-print-a-long-message", harmless(fails_at_length, in_a_path=True))
    try:
        gguf_weights.open_model(os.path.join(work, "safe-a-name-in-two-shards"))
        said = ""
    except gguf_weights.ModelError as fault:
        said = str(fault)
    a, b = (os.path.join(work, "safe-a-name-in-two-shards", name) for name in ("a.gguf", "b.gguf"))
    wanted = f"entry 1 of the tensor table of {b} repeats the name of a tensor in {a}"
    check("a-repeated-name-is-told-by-its-entry", said == str(gguf_weights.ModelError(wanted)))  # the paths escaped


def wide_cases(work, check):
    """Run the cases on the types Q8_0, F16 and BF16, on files that the test writes under the folder work."""
    rng = random.Random(2)

    def opened(name, tensors):
        os.mkdir(os.path.join(work, name))
        write_shard(os.path.join(work, name, "a.gguf"), METADATA, tensors)
        return gguf_weights.open_model(os.path.join(work, name))

    def as_expected(model, tensors, device):
        """Say whether each tensor has the expected values, and the values of gguf's decoder, on the device."""
        good = True
        for name, (kind, stored, expected) in tensors.items():
            found = model.tensor(name, device=device)
            good &= found.device.type == device and same_bits(found, expected)
            good &= kind is None or same_bits(found, dequantize(stored, kind))
        return good

    for kind in (Q8_0, F16, BF16):
        # The shapes of synthetic_cases. The embedding table has the type too, as every matrix but the router has.
        tensors = {
            "token_embd.weight": wide_tensor(rng, kind, (8, 1024)),
            "blk.0.attn_norm.weight": float_tensor(rng, (1024,)),
            "blk.0.attn_q.weight": wide_tensor(rng, kind, (8, 1024)),
            "blk.0.attn_k.weight": wide_tensor(rng, kind, (4, 1024)),
            "blk.0.ffn_gate.weight": wide_tensor(rng, kind, (6, 1024)),
            "blk.1.ffn_gate_inp.weight": float_tensor(rng, (3, 1024)),
            "blk.1.ffn_down_exps.weight": wide_tensor(rng, kind, (3, 4, 32)),
            "blk.1.ffn_gate_exps.weight": wide_tensor(rng, kind, (3, 2, 1024)),
            "blk.1.ffn_gate_shexp.weight": wide_tensor(rng, kind, (2, 1024)),
        }
        model = opened("all-" + kind.name, tensors)
        stacked = [name for name in tensors if name.endswith("_exps.weight")]
        for device in DEVICES:
            on = f"-on-{device}"
            check(kind.name + "-values-as-computed-and-as-gguf-decodes-them" + on, as_expected(model, tensors, device))
            check(
                kind.name + "-one-expert-is-that-slice-of-the-stacked-tensor" + on,
                all(same_bits(model.tensor(n, e, device), tensors[n][2][e]) for n in stacked for e in range(3)),
            )
        check(kind.name + "-shape", gguf_weights.shape_words(model) == SHAPE)
        check(
            kind.name + "-info",
            all(model.info(n)[0::4] == (int(kind), s.nbytes) for n, (k, s, _) in tensors.items() if k is not None),
        )

    tensors = every_pattern()
    model = opened("every-pattern", tensors)
    for device in DEVICES:
        for name, tensor in tensors.items():
            check(f"{name}-finite-pattern-on-{device}", as_expected(model, {name: tensor}, device))

    # One value that is not finite among finite neighbours. For Q8_0 an infinite scale gives an infinity, or with the
    # code 0 a NaN, and a scale that is a NaN gives a NaN.
    not_finite = {}
    for kind, patterns in ((F16, (0x7C00, 0xFC00, 0x7C01, 0xFFFF)), (BF16, (0x7F80, 0xFF80, 0x7F81, 0xFFFF))):
        for bits in patterns:
            stored = numpy.array([0x3C00, bits, 0x3C00, 0x3C00], dtype="<u2").view(numpy.uint8)
            not_finite[f"{kind.name}.{bits:04x}"] = (kind, stored, None)
    for h, code in ((0x7C00, 1), (0xFC00, -128), (0x7C00, 0), (0x7E00, 1)):
        blocks = struct.pack("<H32b", 0x3C00, *[1] * 32) + struct.pack("<H32b", h, *[code] * 32)
        not_finite[f"Q8_0.{h:04x}.{code}"] = (Q8_0, numpy.frombuffer(blocks, dtype=numpy.uint8), None)
    model = opened("wide-not-finite", not_finite)
    for kind in (Q8_0, F16, BF16):
        names = [name for name in not_finite if name.startswith(kind.name + ".")]
        check(kind.name + "-rejects-a-value-that-is-not-finite", all(rejected(model.tensor, name) for name in names))

    # A row of Q8_0 must hold whole blocks. gguf's writer makes no other file, so the test changes the dimensions
    # (32, 2) in a file's tensor table to (16, 4), which is the same number of elements and of bytes.
    opened("whole-blocks", {"rows.weight": (Q8_0, numpy.zeros((2, 34), dtype=numpy.uint8), None)})
    content = bytes_of(os.path.join(work, "whole-blocks", "a.gguf"))
    before, after = struct.pack("<2Q", 32, 2), struct.pack("<2Q", 16, 4)
    os.mkdir(os.path.join(work, "part-blocks"))
    with open(os.path.join(work, "part-blocks", "a.gguf"), "wb") as f:
        f.write(content.replace(before, after))
    check(
        "Q8_0-rejects-a-row-that-is-not-whole-blocks",
        content.count(before) == 1 and rejected(gguf_weights.open_model, os.path.join(work, "part-blocks")),
    )


def small_model_cases(folder, check):
    """Run the cases on the small model in `folder`.

    Its values are compared only with gguf's decoder, and only on the CPU.
    """
    with open(os.path.join(TREE, "prover", "examples", "target_small_file_f2", "model.txt")) as f:
        layers, heads, kv_heads, width, expert_width, experts, vocabulary, _ = (int(word) for word in f.read().split())
    model = gguf_weights.open_model(folder)
    expected = {"d": 5120, "dh": 128, "L": layers, "H": heads, "G": kv_heads, "f": width, "fe": expert_width}
    expected.update({"NE": experts, "V": vocabulary, "eps": float(f32(1e-5)), "theta": 500000.0})
    expected.update({"moe": list(range(1, layers, 2)), "rope": [n for n in range(layers) if n % 4 != 3]})
    check("small-model-shape", gguf_weights.shape_words(model) == expected)

    differ, names = [], []
    for shard in sorted(name for name in os.listdir(folder) if name.endswith(".gguf")):
        for record in gguf.GGUFReader(os.path.join(folder, shard)).tensors:
            names.append(record.name)
            # Of a stacked tensor, the first and the last expert are compared.
            for expert in (None,) if len(record.shape) < 3 else (0, int(record.shape[2]) - 1):
                stored = record.data if expert is None else record.data[expert]
                if not same_bits(model.tensor(record.name, expert=expert), dequantize(stored, record.tensor_type)):
                    differ.append(record.name)
    print(f"small model: {len(names)} tensors compared with gguf (first and last expert); differ: {differ or 'none'}")
    check("small-model-values-as-gguf-decodes-them", not differ and model.names() == sorted(names))


def main():
    tally = {True: 0, False: 0}

    def check(name, passed):
        tally[bool(passed)] += 1
        print(f"case {name}: {'passed' if passed else 'FAILED'}")

    work = tempfile.mkdtemp()
    try:
        synthetic_cases(work, check)
        wide_cases(work, check)
        small_model = os.environ.get("SMALL_MODEL")
        if not small_model and os.environ.get("MAKE_MODEL"):
            small_model = os.path.join(work, "small_model")
            with open(os.path.join(TREE, "prover", "examples", "target_small_file_f2", "model.txt")) as f:
                subprocess.run([os.environ["MAKE_MODEL"], small_model, *f.read().split()], check=True)
        if small_model:
            small_model_cases(small_model, check)
    finally:
        shutil.rmtree(work)
    if tally[False]:
        print(f"FAILED: {tally[False]} of {tally[True] + tally[False]} cases; see the lines above that end in FAILED")
        return FAILED
    if not small_model:
        how = "set SMALL_MODEL=<model's folder> or MAKE_MODEL=<make_model>"
        print(f"in part: {tally[True]} cases passed; 2 skipped ({how})")
        return NOT_ALL_RUN
    print(f"OK: {tally[True]} cases passed")
    return PASSED


if __name__ == "__main__":
    sys.exit(main())
