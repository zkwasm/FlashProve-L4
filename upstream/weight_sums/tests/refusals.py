#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Started by weight_sums/tests/run_tests.sh, which the runner starts; by hand: below a directory of your own.
"""Test which commands and model files the two weight_sums programs reject, and what they write.

bin/weight_sums (weight_sums/weight_sums.cpp) and weight_sums/weight_sums.py read a model's weight
files and write the sums file, against whose hashes the committer program cp_commit checks the
weights before it commits to them. A wrong command must change no file that is there.

usage: refusals.py <new folder> --kind cpp|py -- <program and its first arguments>

The kind says which program follows: cpp stands for bin/weight_sums and py for weight_sums.py.
Every case runs in a folder of its own, so the paths after "--" must be absolute. For example, with
the path of the tree for <tree>:

  python3 refusals.py "$(mktemp -d)/cases" --kind cpp -- <tree>/bin/weight_sums
  python3 refusals.py "$(mktemp -d)/cases" --kind py -- python3 -B <tree>/weight_sums/weight_sums.py

The script makes the folder, closed to other users, and removes nothing. It writes small synthetic
shards (a model's files, in the llama.cpp project's format GGUF) into <folder>/in/ and runs its
cases, up to four at a time. On standard output it reports each case that did not end as expected,
then what else it found, and in the last line the counts. It needs Python 3.9 or later, with the
standard library only, and Linux 5.3 or later. Its exit status is

  0  if every case that ran ended as expected,
  1  if a case did not, if after the last case in/ is not as the script wrote it or run/ contains
     anything but the cases' folders, or if one of the script's own cases is not what it is meant
     to be,
  2  for a wrong command line of this script, a folder that exists or cannot be made, a program
     that cannot be run, or a system without os.pidfd_open.

On SIGTERM or SIGINT the script kills the programs that are running and ends with status 128 plus
the signal's number.

A case is one command. It runs in a new folder, <folder>/run/<case>, with an empty standard input.
If it has not ended after 120 seconds it is killed, and the script prints which processes it had and
what it had written. A faulty case is meant to have one fault, or several faults with the same exit
status, unless it is there to show which of two faults counts. The valid cases show that a program
does not reject more than it should. After the command the script checks these points:

  - The exit status is the expected one.
  - Nothing was written to standard output. A command that asks weight_sums.py for its help text
    must write something there.
  - A valid command other than a request for help made one new file, at its output path, with
    exactly the expected bytes. Its permission bits are 0600, although every command runs under the
    umask 000. Any other command made nothing, so no temporary file is left either.
  - Everything else in the folder of the case is as it was before the command, and so are the
    shards that the command names. The script compares the bytes of the files that it can read and
    the targets of links, not permission bits or times.
  - A command that was rejected wrote a message to standard error.
  - No message tells the user to remove something: after a mistake in the command, the file that is
    in the way can be one of the model's files. The check searches standard error for "remove",
    without regard to case, and messages quote paths, so the path of <new folder> must not contain
    that word.
  - Standard error shows no SHA-256 from the expected sums file (which only a valid command has), no
    Python traceback and no warning, because the programs must not print a number that is derived
    from a weight.
  - Standard error shows no escape character, nor the text that some cases put after an escape
    sequence into a shard's metadata (see ESCAPE).
  - Standard error has at most MOST_ON_STANDARD_ERROR bytes, so that no message prints a long name
    from a file.

weight_sums/README.md explains the programs' exit statuses, the sums file that they write (a line
for each committed tensor, with the field `bits` and a SHA-256 of its integers, then a closing
line), which tensors are committed, and, in short, the conversion rule, which turns the stored
weights into these integers. In the comments below, a norm is a tensor of normalization weights,
which are its gains, and a matrix is a committed tensor other than the embedding
(token_embd.weight). A matrix reads a norm if the rule multiplies the gains of that norm
into it. The product of a decoded value v is then RN32(multiplier * v), where the multiplier is
RN32(gain * c) and the constant c depends on the matrix. In any other matrix the product is v. mx is
the largest absolute value of a matrix's products. The integers are RNE(2^e * product) in a matrix,
whose exponent is e, and RNE(2^e_res * v) in the embedding, and --eres gives e_res. RN32 rounds to
the nearest binary32 number and RNE to the nearest integer, both with ties to even.
The rule is defined in docs/model.md, "11 From the model's files to the integers", which writes
c_0 for c. Its paragraph "Refusals", in 11.4, says that values must be finite and gives the bounds
on a multiplier and on e. The comments in bound_cases() state these bounds too.
"""

import collections
import concurrent.futures
import hashlib
import math
import os
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading

sys.dont_write_bytecode = True  # so that the import of oracle.py leaves no __pycache__ folder in the tree

# oracle.py writes GGUF files and computes their sums file in exact rational arithmetic.
from oracle import Rejected, sums_file, write_shard

# The exit statuses of both programs for a rejected command. Status 26 (out of memory, or a worker
# process that was killed or could not start) has no case here. fork_failure.py tests it for a
# worker of weight_sums.py that cannot start.
USAGE = 21  # an invalid command line, or a file that cannot be read or written
CONTAINER = 22  # the GGUF container
TYPE = 23  # the type of a tensor
NOT_FINITE = 24  # a value that is not finite
BOUND = 25  # a bound of the conversion rule

SECONDS = 120  # a command that takes longer is killed
# The number of cases that run at the same time. A command spends most of its time starting the
# program, so this saves most of the waiting.
AT_A_TIME = min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)

# Numbers of the GGUF format, from the public sources of llama.cpp (github.com/ggml-org/llama.cpp):
# llama.cpp as carried in llama-cpp-python 0.3.7, whose changelog names the commit
# 794fe23f29fb40104975c91fe19f23798f7c726e. The value types of metadata are enum gguf_type in
# ggml/include/gguf.h of that copy.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
# The storage types of tensors are enum ggml_type in ggml/include/ggml.h. BYTES_OF_256 says how many
# bytes 256 elements take, by the block structures in ggml/src/ggml-common.h. Q4_0, Q3_K and F64 stand
# for the types that the programs do not read. Their numbers and sizes are those of the package gguf,
# version 0.19.0 (gguf/constants.py: GGMLQuantizationType and GGML_QUANT_SIZES).
F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, F64, BF16 = 0, 1, 2, 8, 11, 12, 13, 14, 28, 30
BYTES_OF_256 = {F16: 512, Q4_0: 144, Q8_0: 272, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, F64: 2048, BF16: 512}
UNDEFINED_TYPE = 99  # no storage type has this number
# The positions of the binary16 numbers d and dmin in a block of 256 elements. A block of Q6_K has no dmin. The byte
# layout of the block formats Q4_K, Q5_K and Q6_K is that of ggml, the tensor library of the llama.cpp project. It is
# defined in ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K). ggml has the MIT licence, "Copyright (c)
# 2023-2024 The ggml authors" in the file LICENSE of that copy. ../../THIRD_PARTY.md, section "ggml", has the full
# citation and the text of the licence.
D_AT = {Q4_K: 0, Q5_K: 0, Q6_K: 208}
DMIN_AT = {Q4_K: 2, Q5_K: 2}

WIDTH = 5120  # the only model width that the programs accept


def u32(number):
    return struct.pack("<I", number)


def u64(number):
    return struct.pack("<Q", number)


def floats(values):
    """Return the values as binary32 numbers. Every value in this file is exact in that format."""
    return struct.pack("<%df" % len(values), *values)


def ramp(count):
    """Return count F32 elements, which cycle through the multiples of 1/64 from -0.75 to 0.75."""
    return floats([(i % 97 - 48) / 64 for i in range(count)])


def put(data, position, new):
    """Return a copy of data in which the bytes new replace those that start at the position."""
    return bytes(data[:position]) + new + bytes(data[position + len(new):])


def with_word(data, index, word):
    """Return a copy of F32 data in which the element at index has the binary32 bit pattern word."""
    return put(data, 4 * index, u32(word))


def blocks(storage_type, count, d=0x2C00, dmin=0x2800):
    """Return blocks of a K-quant type (Q4_K, Q5_K or Q6_K).

    d and dmin are binary16 bit patterns, by default 2^-4 and 2^-5. The other bytes, which hold the
    scales and the codes, follow a pattern. Every byte value is valid there.
    """
    made = b""
    for n in range(count):
        block = bytes((n + 37 * i) % 251 for i in range(BYTES_OF_256[storage_type]))
        block = put(block, D_AT[storage_type], struct.pack("<H", d))
        if storage_type in DMIN_AT:
            block = put(block, DMIN_AT[storage_type], struct.pack("<H", dmin))
        made += block
    return made


def q8_0(count, d=0x2C00):
    """Return blocks of Q8_0, of 32 elements each. d is a binary16 bit pattern. The codes follow a pattern."""
    return b"".join(struct.pack("<H", d) + bytes((n + 37 * i) % 251 for i in range(32)) for n in range(count))


def varied(rows, change=None, drop=(), add=()):
    """Return a copy of a list of metadata pairs or of tensors.

    change maps a key or a name to the rest of its new row, drop lists keys or names to leave out,
    and the rows in add go to the end.
    """
    change = change or {}
    return [(row[0],) + tuple(change.get(row[0], row[1:])) for row in rows if row[0] not in drop] + list(add)


# The parts from which the cases build their shards. A row of metadata is (key, value type, value),
# and a tensor is (name, dimensions, storage type, data), as write_shard() in oracle.py takes them.
# The keys, and the name rope_freqs of the tensor that the programs ignore, are those of real model
# files (gguf/constants.py of gguf-py 0.19.0, the llama.cpp project's Python package `gguf`).
METADATA = [
    ("general.architecture", STRING, "llama4"),
    ("llama4.embedding_length", UINT32, WIDTH),
    ("llama4.block_count", UINT32, 1),
    ("llama4.attention.key_length", UINT32, 128),
    ("general.name", STRING, "synthetic"),
    ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "bc", "def"])),
    ("tokenizer.ggml.token_type", ARRAY, (INT32, [1, 1, 1])),
]
ARCHITECTURE, EMBEDDING_LENGTH, BLOCK_COUNT, KEY_LENGTH, NAME, TOKENS, TOKEN_TYPES = range(7)  # the rows of METADATA
SMALL, LONG = ramp(256), ramp(WIDTH)
GAINS = floats([(1.0, 0.5, -1.5, 2.0)[j % 4] for j in range(WIDTH)])
PLAIN = ("blk.0.attn_output.weight", (256, 1), F32, SMALL)  # a matrix that reads no norm
QUANTIZED = ("blk.0.ffn_down.weight", (256, 1), Q4_K, blocks(Q4_K, 1))  # another one
NORM = ("blk.0.ffn_norm.weight", (WIDTH,), F32, GAINS)
READER = ("blk.0.ffn_up.weight", (WIDTH, 1), F32, LONG)  # a matrix that reads NORM
EMBEDDING = ("token_embd.weight", (256, 1), F32, SMALL)
IGNORED = ("rope_freqs.weight", (64,), F32, floats([1.0] * 64))  # the programs ignore a tensor of this name
ESCAPE = "\x1b[2Jtext-of-the-file"  # an architecture's name or the start of a key; no message may show a part of it
# In bytes. The longest message with its usage text has fewer; a name from a crafted file can have more.
MOST_ON_STANDARD_ERROR = 4000


def stop(why):
    """End the script because one of its own cases is not what it is meant to be."""
    print("%s: %s" % (os.path.basename(sys.argv[0]), why), file=sys.stderr)
    sys.exit(1)


# One case, as Cases.add() makes it. out is relative to the folder of the case, and expected is the
# bytes that the file at out must have, or None.
Case = collections.namedtuple("Case", "name before arguments status out expected")


class Cases:
    """The cases for one of the two programs, and the shards that they use."""

    def __init__(self, folder, kind):
        self.kind = kind
        self.inputs = os.path.join(folder, "in")
        self.runs = os.path.join(folder, "run")
        self.rows = []
        self.lists = {}  # for each shard in in/, by its path: the lists that say what it contains
        self.computed = {}  # the sums files that have been computed, by their shards and their e_res
        # How most commands start: with one thread or worker, and with the output path.
        self.start = ["--threads" if kind == "cpp" else "--workers", "1", "--out", "out.txt"]
        os.mkdir(self.inputs)
        os.mkdir(self.runs)

    def shard(self, name, tensors, metadata=METADATA, edit=None, like=None, **how):
        """Write in/<name>.gguf and return its path.

        how may contain alignment and version, which go to write_shard(). edit, if given, is a
        function that takes the bytes of the file and the positions of its fields (the Layout that
        write_shard() returns), and returns the bytes that the file gets in their place. like is a
        pair (metadata, tensors) from which sums_file() computes the expected sums. It is needed
        where they do not follow from the arguments metadata and tensors: after an edit that changes
        which bytes a tensor has, and where sums_file() rejects lists that one of the programs
        accepts.
        """
        path = os.path.join(self.inputs, name + ".gguf")
        positions = write_shard(path, metadata, tensors, **how)
        if edit:
            with open(path, "rb") as file:
                data = file.read()
            with open(path, "wb") as file:
                file.write(edit(data, positions))
        self.lists[path] = like or (metadata, tensors)
        return path

    def add(self, name, cpp, py, arguments, eres=11, out="out.txt", before=(), shards=None):
        """Add one case. bin/weight_sums must give the status cpp, and weight_sums.py the status py.

        None says that the case is not meant for that program. If the status is 0, the expected sums
        file is computed for the shards in in/ that the arguments name, in their order, at
        e_res = eres. shards names them where the arguments do not. out is the path of the output,
        and None for an accepted command that must write no file.

        before is a list of actions, carried out in the list's order in the folder of the case
        before the command runs:
          ("file", path)          make a regular file
          ("dir", path)           make a folder
          ("link", path, target)  make a symbolic link
          ("copy", shard, path)   make a copy of a shard in in/
          ("mode", path, bits)    set the permission bits of the path. The superuser may read and
                                  write every file, so a case with this action is left out when the
                                  superuser runs the script
        """
        status = cpp if self.kind == "cpp" else py
        if status is None:
            return
        expected = None
        if status == 0 and out is not None:
            shards = tuple(shards or [argument for argument in arguments if argument in self.lists])
            if (shards, eres) not in self.computed:
                try:
                    self.computed[shards, eres] = sums_file([self.lists[shard] for shard in shards], eres)
                except Rejected as rejected:
                    stop("the case %s is meant to be valid, but oracle.py finds status %d" % (name, rejected.status))
            expected = self.computed[shards, eres]
        if name in [case.name for case in self.rows]:
            stop("two cases have the name %s" % name)
        self.rows.append(Case(name, before, arguments, status, out, expected))

    def one(self, name, cpp, py, tensors, metadata=METADATA, **how):
        """Add a case whose command names one shard, written for it. how goes to shard()."""
        self.add(name, cpp, py, self.start + [self.shard(name, tensors, metadata, **how)])

    def valid(self, name, tensors, metadata=METADATA, **how):
        """Add a case with one shard that both programs accept."""
        self.one(name, 0, 0, tensors, metadata, **how)

    def fault(self, name, status, tensors, metadata=METADATA):
        """Add a case: one shard that both programs reject with this status, by the conversion rule.

        The exact arithmetic of oracle.py must reject the shard with the same status. This guards
        against a shard that is valid after all, or in which oracle.py first meets a fault of
        another status.
        """
        try:
            sums_file([(metadata, tensors)], 11)
            found = 0
        except Rejected as rejected:
            found = rejected.status
        if found != status:
            stop("the case %s is meant to have status %d, but oracle.py finds %d" % (name, status, found))
        self.one(name, status, status, tensors, metadata)


def command_cases(cases):
    """Add the cases of the command line. The programs' grammars differ, and so do many statuses."""
    shard = cases.shard("plain", [PLAIN, EMBEDDING])
    second = cases.shard("second", [QUANTIZED], [])
    # An embedding so small that its integers stay below their bound, 2^30, at every e_res that the
    # programs accept.
    tiny = cases.shard("tiny", [("token_embd.weight", (7,), F32, floats([i * 2.0 ** -80 for i in range(-3, 4)]))])
    start = cases.start
    out = ["--out", "out.txt"]  # for the commands that give --threads or --workers themselves, or neither

    cases.add("command.defaults", 0, 0, out + [shard])
    cases.add("command.no-argument", USAGE, USAGE, [])
    cases.add("command.no-shard", USAGE, USAGE, out)
    cases.add("command.no-out", USAGE, USAGE, [shard])
    cases.add("command.out-without-value", USAGE, USAGE, [shard, "--out"])
    cases.add("command.eres-without-value", USAGE, USAGE, start + [shard, "--eres"])
    cases.add("command.unknown-option", USAGE, USAGE, start + ["--no-such-option", shard])
    cases.add("command.same-shard-twice", CONTAINER, CONTAINER, start + [shard, shard])
    for case, option in (("short", "-h"), ("long", "--help")):  # only weight_sums.py has a help text
        cases.add("command.help-" + case, USAGE, 0, [option], out=None)

    # The values of the options: --eres from -100 to 100, --threads from 1 to 256, --workers from 1.
    # 4294967307 is 2^32 + 11, which a program that keeps only 32 bits would take for 11.
    for case, text in (("empty", ""), ("text", "x"), ("text-after-digits", "5x"), ("fraction", "5.0"),
                       ("minus-101", "-101"), ("101", "101"), ("4294967307", "4294967307")):
        cases.add("command.eres-" + case, USAGE, USAGE, start + ["--eres", text, tiny])
    cases.add("command.eres-5", 0, 0, start + ["--eres", "5", shard], eres=5)
    cases.add("command.eres-plus-5", 0, 0, start + ["--eres", "+5", shard], eres=5)
    cases.add("command.eres-minus-100", 0, 0, start + ["--eres", "-100", tiny], eres=-100)
    cases.add("command.eres-100", 0, 0, start + ["--eres", "100", tiny], eres=100)
    for text in ("x", "0", "-1"):
        cases.add("command.threads-" + text.replace("-", "minus-"), USAGE, None, out + ["--threads", text, shard])
        cases.add("command.workers-" + text.replace("-", "minus-"), None, USAGE, out + ["--workers", text, shard])
    cases.add("command.threads-257", USAGE, None, out + ["--threads", "257", shard])
    cases.add("command.threads-256", 0, None, out + ["--threads", "256", shard])
    cases.add("command.threads-2", 0, USAGE, out + ["--threads", "2", shard])  # each program rejects the other's option
    cases.add("command.workers-2", USAGE, 0, out + ["--workers", "2", shard])

    # bin/weight_sums rejects an option that is given twice. weight_sums.py takes the last value.
    cases.add("command.eres-twice", USAGE, 0, start + ["--eres", "10", "--eres", "5", shard], eres=5)
    cases.add("command.out-twice", USAGE, 0, ["--out", "other.txt"] + start + [shard])

    cases.add("command.options-before-the-shards", 0, 0, start + [shard, second])
    cases.add("command.options-between-the-shards", 0, USAGE, [shard] + start + [second])
    cases.add("command.options-after-the-shards", 0, 0, [shard, second] + start)

    # Forms that only Python's module argparse knows, and the options that only weight_sums.py has.
    cases.add("command.eres-with-equals-sign", USAGE, 0, start + ["--eres=5", shard], eres=5)
    cases.add("command.eres-abbreviated", USAGE, 0, start + ["--ere", "5", shard], eres=5)
    cases.add("command.decode-twice", USAGE, 0, start + ["--decode-twice", shard])
    cases.add("command.any-gguf-version", USAGE, 0, start + ["--any-gguf-version", shard])

    # bin/weight_sums takes an argument that begins with a single dash for the path of a shard, and
    # argparse takes it for an option (see Python's documentation of argparse). Both understand "./"
    # before the name.
    cases.add("command.shard-named-dash-letter", 0, USAGE, start + ["-x"], before=[("copy", shard, "-x")],
              shards=[shard])
    cases.add("command.shard-named-dot-slash-dash-letter", 0, 0, start + ["./-x"], before=[("copy", shard, "-x")],
              shards=[shard])


def path_cases(cases):
    """Add the cases of unreadable shards and unusable output paths. Such a fault has status 21."""
    shard = os.path.join(cases.inputs, "plain.gguf")  # command_cases wrote it
    start = cases.start
    jobs = start[:2]  # one thread or worker, for the commands that name another output path
    cases.add("file.absent", USAGE, USAGE, start + ["absent.gguf"])
    cases.add("file.second-absent", USAGE, USAGE, start + [shard, "absent.gguf"])
    cases.add("file.folder", USAGE, USAGE, start + ["folder.gguf"], before=[("dir", "folder.gguf")])
    cases.add("file.unreadable", USAGE, USAGE, start + ["shard.gguf"],
              before=[("copy", shard, "shard.gguf"), ("mode", "shard.gguf", 0o000)])
    cases.add("file.output-folder-absent", USAGE, USAGE, jobs + ["--out", "absent/out.txt", shard])
    cases.add("file.output-folder-locked", USAGE, USAGE, jobs + ["--out", "locked/out.txt", shard],
              before=[("dir", "locked"), ("mode", "locked", 0o555)])

    # A program rejects the command if anything exists at the output path or at its own temporary
    # path: <out>.part for bin/weight_sums, <out>.tmp for weight_sums.py. It ignores the other's.
    for suffix, cpp, py in (("", USAGE, USAGE), (".part", USAGE, 0), (".tmp", 0, USAGE)):
        path = "out.txt" + suffix
        for thing, before in (("file", [("file", path)]), ("folder", [("dir", path)]),
                              ("link-to-a-file", [("file", "target"), ("link", path, "target")]),
                              ("link-to-nothing", [("link", path, "absent-target")])):
            cases.add("output.%s-at-out%s" % (thing, suffix.replace(".", "-")), cpp, py, start + [shard], before=before)

    # The programs look at the output path before they read a shard. To show this order, the first
    # case has faults of two statuses, 21 and 22.
    faulty = cases.shard("output.wrong-magic", [PLAIN], edit=lambda data, at: put(data, 0, b"GGUX"))
    cases.add("output.file-at-out-and-a-faulty-shard", USAGE, USAGE, start + [faulty], before=[("file", "out.txt")])
    cases.add("file.second-faulty", CONTAINER, CONTAINER, start + [shard, faulty])

    # A shard at the path of the output or of a temporary file must survive.
    cases.add("output.names-the-shard", USAGE, USAGE, jobs + ["--out", "shard.gguf", "shard.gguf"],
              before=[("copy", shard, "shard.gguf")])
    cases.add("output.names-the-shard-before-an-unknown-option", USAGE, USAGE,
              ["--out", "shard.gguf", "--no-such-option", "shard.gguf"], before=[("copy", shard, "shard.gguf")])
    cases.add("output.twice-with-the-shard-first", USAGE, 0, ["--out", "shard.gguf"] + start + ["shard.gguf"],
              before=[("copy", shard, "shard.gguf")], shards=[shard])
    cases.add("output.twice-with-the-shard-second", USAGE, USAGE, start + ["--out", "shard.gguf", "shard.gguf"],
              before=[("copy", shard, "shard.gguf")])
    cases.add("output.shard-at-out-part", USAGE, 0, start + ["out.txt.part"], before=[("copy", shard, "out.txt.part")],
              shards=[shard])
    cases.add("output.shard-at-out-tmp", 0, USAGE, start + ["out.txt.tmp"], before=[("copy", shard, "out.txt.tmp")],
              shards=[shard])

    cases.add("output.empty", USAGE, USAGE, jobs + ["--out", "", shard])
    cases.add("output.ends-in-a-slash", USAGE, USAGE, jobs + ["--out", "out.txt/", shard])
    cases.add("output.ends-in-a-slash-and-a-dot", USAGE, USAGE, jobs + ["--out", "out.txt/.", shard])
    cases.add("output.space-in-the-path", 0, 0, jobs + ["--out", "a b.txt", shard], out="a b.txt")
    cases.add("output.absolute-path", 0, 0,
              jobs + ["--out", os.path.join(cases.runs, "output.absolute-path", "out.txt"), shard])


def container_cases(cases):
    """Add the cases of the GGUF container: cut files, contradictory fields and unusual layouts.

    bin/weight_sums has its own reader, and weight_sums.py reads the container with gguf-py. The two
    readers differ in what they check, above all on tensors that the programs ignore. So many cases
    have two different statuses.

    In the edits, at is the file's Layout (oracle.py): at.pairs[i] has the positions of metadata
    pair i's key length, value type and value, and at.table[i] those of tensor i's name length,
    number of dimensions, first dimension, storage type and offset. The header has 24 bytes, with
    the number of tensors at byte 8 and the number of pairs at byte 16.
    """
    base = [PLAIN, QUANTIZED, IGNORED]
    committed, ignored = 0, 2  # the rows of base that the edits below change
    far = 2 ** 40  # a count or a length that runs far past the end of every file here

    cases.valid("container.base", base)

    def padding_cut(data, at):
        if at.table[-1][4] + 8 == at.data_start:
            stop("the shard of container.cut-in-the-padding has no padding")
        return data[:at.data_start - 1]

    for case, edit in (
        ("magic", lambda data, at: put(data, 0, b"GGUX")),
        ("empty", lambda data, at: b""),
        ("cut-in-the-header", lambda data, at: data[:23]),
        ("cut-after-the-header", lambda data, at: data[:24]),
        ("cut-in-a-key", lambda data, at: data[:at.pairs[EMBEDDING_LENGTH][0] + 8 + 5]),
        ("cut-in-a-value", lambda data, at: data[:at.pairs[EMBEDDING_LENGTH][2] + 2]),
        ("cut-in-an-array", lambda data, at: data[:at.pairs[TOKENS][2] + 4 + 8 + 8 + 1 + 8 + 1]),  # in the string "bc"
        ("cut-in-an-array-of-numbers", lambda data, at: data[:at.pairs[TOKEN_TYPES][2] + 4 + 8 + 4 + 2]),
        ("cut-in-the-tensor-table", lambda data, at: data[:at.table[1][2] + 3]),
        ("cut-in-the-padding", padding_cut),
        ("key-length", lambda data, at: put(data, at.pairs[NAME][0], u64(far))),
        ("key-length-largest", lambda data, at: put(data, at.pairs[NAME][0], u64(2 ** 64 - 1))),
        ("string-length", lambda data, at: put(data, at.pairs[NAME][2], u64(far))),
        ("tensor-name-length", lambda data, at: put(data, at.table[committed][0], u64(far))),
        ("array-length", lambda data, at: put(data, at.pairs[TOKENS][2] + 4, u64(far))),
        ("pair-count", lambda data, at: put(data, 16, u64(far))),
        ("pair-count-one-more", lambda data, at: put(data, 16, u64(len(METADATA) + 1))),
        ("pair-count-one-fewer", lambda data, at: put(data, 16, u64(len(METADATA) - 1))),
        ("tensor-count", lambda data, at: put(data, 8, u64(far))),
        ("tensor-count-one-more", lambda data, at: put(data, 8, u64(len(base) + 1))),
        ("dimension-count", lambda data, at: put(data, at.table[committed][1], u32(2 ** 32 - 1))),
    ):
        cases.one("container." + case, CONTAINER, CONTAINER, base, edit=edit)
    for version in (2, 4):
        cases.one("container.version-%d" % version, CONTAINER, CONTAINER, base, version=version)
    # A little-endian reader finds the version 0x03000000 in a big-endian file of version 3. In the
    # first file only this number is big-endian. The second file is big-endian throughout. gguf-py
    # can read it, but the programs reject it.
    cases.one("container.version-big-endian", CONTAINER, CONTAINER, base, version=0x03000000)
    path = os.path.join(cases.inputs, "container.big-endian-file.gguf")
    with open(path, "xb") as file:
        file.write(big_endian_shard())
    cases.add("container.big-endian-file", CONTAINER, CONTAINER, cases.start + [path])

    # A file that ends inside the data of its last tensor.
    def cut(count):
        return lambda data, at: data[:at.data[-1] + count]

    cases.one("container.cut-in-committed-data", CONTAINER, CONTAINER, [IGNORED, PLAIN], edit=cut(1024 - 64))
    cases.one("container.cut-in-a-norm", CONTAINER, CONTAINER, [READER, NORM], edit=cut(4 * WIDTH - 64))
    cases.one("container.cut-in-ignored-data", 0, CONTAINER, [PLAIN, IGNORED], edit=cut(256 - 64))

    def more(*rows):
        return METADATA + list(rows)

    # Metadata that the programs have no use for.
    cases.one("container.value-type-13", CONTAINER, CONTAINER, base, more(("test.value", 13, bytes(4))))
    cases.one("container.array-of-arrays", CONTAINER, 0, base, more(("test.value", ARRAY, (ARRAY, [(UINT32, [7])]))))
    cases.one("container.key-twice", 0, CONTAINER, base, more(METADATA[NAME]))
    cases.one("container.key-with-escape-twice", 0, CONTAINER, base, more(*[(ESCAPE + ".x", UINT32, 7)] * 2))
    cases.one("container.key-not-utf-8", 0, CONTAINER, base, more((b"test.\xff\xfe", UINT32, 7)))

    # Of a key that appears twice, bin/weight_sums takes the later value. Here the key is the number
    # of layers.
    layer_1 = [("blk.1.attn_output.weight",) + PLAIN[1:]]
    count = "llama4.block_count"
    cases.one("container.layers-1-then-2", 0, CONTAINER, layer_1, more((count, UINT32, 2)))
    cases.one("container.layers-2-then-1", CONTAINER, CONTAINER, layer_1,
              varied(METADATA, {count: (UINT32, 2)}, add=[(count, UINT32, 1)]))

    # An ignored tensor's name may be empty, or 300 bytes long.
    cases.valid("container.name-lengths", base + [("x" * length,) + IGNORED[1:] for length in (0, 63, 64, 65, 300)])
    # gguf-py reads every name as UTF-8.
    cases.one("container.name-not-utf-8", 0, CONTAINER, base + [(b"\xff\xfe.weight",) + IGNORED[1:]])

    # The number of dimensions and their sizes. bin/weight_sums checks them on every tensor.
    cases.one("container.dimensions-0-committed", CONTAINER, CONTAINER, [QUANTIZED, (PLAIN[0], (), F32, SMALL[:4])])
    cases.one("container.dimensions-0-ignored", CONTAINER, 0, [QUANTIZED, (IGNORED[0], (), F32, SMALL[:4])],
              like=(METADATA, [QUANTIZED]))
    for where, name in (("committed", PLAIN[0]), ("ignored", IGNORED[0])):
        cases.one("container.dimensions-5-" + where, CONTAINER, CONTAINER,
                  [QUANTIZED, (name, (32, 2, 2, 2, 1), F32, SMALL)])
        cases.one("container.product-overflows-" + where, CONTAINER, CONTAINER,
                  [QUANTIZED, (name, (2 ** 32, 2 ** 32), F32, SMALL)])
    for case, name in (("matrix", PLAIN[0]), ("embedding", EMBEDDING[0])):
        cases.one("container.%s-of-shape-256-by-0" % case, CONTAINER, BOUND, [QUANTIZED, (name, (256, 0), F32, b"")])
    cases.one("container.ignored-of-shape-64-by-0", CONTAINER, 0, [QUANTIZED, (IGNORED[0], (64, 0), F32, b"")],
              like=(METADATA, [QUANTIZED]))

    # bin/weight_sums allows at most 2^40 elements in a tensor. It does not check that the data of
    # an ignored tensor are inside the file. gguf-py does, so weight_sums.py rejects both files.
    cases.one("container.ignored-2-to-the-40-elements", 0, CONTAINER,
              [QUANTIZED, (IGNORED[0], (2 ** 20, 2 ** 20)) + IGNORED[2:]])
    cases.one("container.ignored-too-many-elements", CONTAINER, CONTAINER,
              [QUANTIZED, (IGNORED[0], (2 ** 20, 2 ** 20 + 1)) + IGNORED[2:]])

    # Alignments. Each file of the loop is written with the alignment that it states.
    # bin/weight_sums accepts 1 to 2^20, and gguf-py all powers of two. The first tensor takes 144
    # bytes, so at the alignments 1, 16 and 24 the offset of the second is no multiple of 32, the
    # default alignment. The files with large alignments have one tensor, to keep them small.
    for alignment, cpp, py in ((1, 0, 0), (16, 0, 0), (64, 0, 0), (2 ** 20, 0, 0), (24, 0, CONTAINER),
                               (2 ** 21, CONTAINER, 0)):
        tensors = [PLAIN] if alignment >= 2 ** 20 else [QUANTIZED, PLAIN]
        cases.one("container.alignment-%d" % alignment, cpp, py, tensors,
                  more(("general.alignment", UINT32, alignment)), alignment=alignment)
    cases.one("container.alignment-0", CONTAINER, CONTAINER, base, more(("general.alignment", UINT32, 0)))
    cases.fault("container.alignment-as-uint64", CONTAINER, base, more(("general.alignment", UINT64, 32)))
    cases.fault("container.alignment-as-string", CONTAINER, base, more(("general.alignment", STRING, "32")))

    # Offsets. In the first case, the first tensor's data after the edit are its last 255 elements
    # and the first element of the second tensor.
    second = floats([i / 256 for i in range(1, 257)])
    two = [PLAIN, ("blk.0.ffn_down.weight", (256, 1), F32, second)]
    cases.one("container.offset-not-aligned-committed", CONTAINER, 0, two, edit=lambda data, at: moved(data, at, 0, 4),
              like=(METADATA, [PLAIN[:3] + (SMALL[4:] + second[:4],), two[1]]))
    cases.valid("container.offset-not-aligned-ignored", [IGNORED, PLAIN], edit=lambda data, at: moved(data, at, 0, 4))
    cases.one("container.every-offset-moved", CONTAINER, CONTAINER, [QUANTIZED, PLAIN],
              edit=lambda data, at: moved(moved(data, at, 0, 4), at, 1, 4))
    cases.one("container.offset-past-the-end-committed", CONTAINER, CONTAINER, base,
              edit=lambda data, at: moved(data, at, committed, at.size))
    cases.one("container.offset-past-the-end-ignored", 0, CONTAINER, base,
              edit=lambda data, at: moved(data, at, ignored, at.size))
    # An offset so large that the sum of the start of the data section and the offset does not fit
    # into 64 bits.
    cases.one("container.offset-wraps-around-committed", CONTAINER, CONTAINER, base,
              edit=lambda data, at: put(data, at.table[committed][4], u64(2 ** 64 - 32)))
    cases.one("container.offset-wraps-around-ignored", 0, CONTAINER, base,
              edit=lambda data, at: put(data, at.table[ignored][4], u64(2 ** 64 - 32)))
    # After this edit both tensors have the data of the first one.
    cases.valid("container.same-offset", two, edit=lambda data, at: moved(data, at, 1, -1024),
                like=(METADATA, [PLAIN, two[1][:3] + (SMALL,)]))
    cases.valid("container.gap", [PLAIN[:3] + (SMALL + bytes(64),), QUANTIZED])
    cases.valid("container.bytes-after-the-last-tensor", base, edit=lambda data, at: data + b"more" * 25)
    cases.valid("container.padding-not-zero", base,
                edit=lambda data, at: put(data, at.table[-1][4] + 8, b"\xff" * (at.data_start - at.table[-1][4] - 8)))


def big_endian_shard():
    """Return the bytes of a big-endian GGUF file of version 3 with one F32 matrix.

    The file is what the writer of gguf-py 0.19.0 makes for a big-endian machine
    (gguf/gguf_writer.py, the functions write_header_to_file and _pack): it packs every number as
    big-endian, except the magic, whose four bytes are the same as in a little-endian file. See also
    the specification of GGUF (docs/gguf.md in github.com/ggml-org/ggml).
    """
    def string(text):
        return struct.pack(">Q", len(text)) + text.encode()

    head = b"GGUF" + struct.pack(">IQQ", 3, 1, 3)  # the version, one tensor, three metadata pairs
    head += string("general.architecture") + struct.pack(">I", STRING) + string("llama4")
    head += string("llama4.embedding_length") + struct.pack(">II", UINT32, WIDTH)
    head += string("llama4.block_count") + struct.pack(">II", UINT32, 1)
    head += string(PLAIN[0]) + struct.pack(">IQQIQ", 2, 256, 1, F32, 0)  # two dimensions, the type and the offset
    head += bytes(-len(head) % 32)
    return head + struct.pack(">256f", *struct.unpack("<256f", SMALL))


def moved(data, at, index, by):
    """Return a copy of a shard's bytes with `by` added to the offset of the tensor at index."""
    position = at.table[index][4]
    return put(data, position, u64(struct.unpack_from("<Q", data, position)[0] + by))


def metadata_cases(cases):
    """Add the cases of metadata that describe the model.

    These are the architecture's name and, under keys that begin with that name, the width
    (embedding_length), the number of layers (block_count) and the head dimension
    (attention.key_length, 128 if absent). The programs read them from the first shard only. They
    accept no width but 5120 and, in the value types that each reads (see below), no head
    dimension but 128.
    """
    free = [PLAIN, EMBEDDING]  # tensors that read no norm, so that the width matters only through the metadata
    keys = [row[0] for row in METADATA]
    width, layers, head = keys[EMBEDDING_LENGTH], keys[BLOCK_COUNT], keys[KEY_LENGTH]

    for case, drop in (("architecture", [keys[ARCHITECTURE]]), ("width", [width]), ("layers", [layers]),
                       ("width-and-layers", [width, layers])):
        cases.fault("metadata.no-" + case, CONTAINER, free, varied(METADATA, drop=drop))
    cases.valid("metadata.no-head-dimension", free, varied(METADATA, drop=[head]))
    cases.fault("metadata.architecture-as-uint32", CONTAINER, free, varied(METADATA, {keys[ARCHITECTURE]: (UINT32, 4)}))
    cases.fault("metadata.width-4096", BOUND, free, varied(METADATA, {width: (UINT32, 4096)}))
    cases.fault("metadata.width-5120-plus-2-to-the-32", BOUND, free,
                varied(METADATA, {width: (UINT64, 2 ** 32 + WIDTH)}))
    cases.fault("metadata.head-dimension-64", BOUND, free, varied(METADATA, {head: (UINT32, 64)}))
    cases.valid("metadata.layers-0", [EMBEDDING], varied(METADATA, {layers: (UINT32, 0)}))

    # The architecture may have any name, but bin/weight_sums rejects the empty string.
    def renamed(name):
        return [(key.replace("llama4", name), value_type, name if value == "llama4" else value)
                for key, value_type, value in METADATA]

    cases.valid("metadata.another-architecture", free, renamed("foo"))
    cases.fault("metadata.another-architecture-head-dimension-64", BOUND, free,
                varied(renamed("foo"), {"foo.attention.key_length": (UINT32, 64)}))
    cases.one("metadata.architecture-empty", CONTAINER, 0, free, renamed(""), like=(METADATA, free))
    # Messages about the keys below must not print the architecture's name, which here starts with a
    # terminal escape sequence.
    cases.valid("metadata.architecture-with-escape", free, renamed(ESCAPE))
    for case, key in (("width", ".embedding_length"), ("layers", ".block_count")):
        cases.fault("metadata.architecture-with-escape-no-" + case, CONTAINER, free,
                    varied(renamed(ESCAPE), drop=[ESCAPE + key]))
    cases.fault("metadata.architecture-with-escape-layers-as-string", CONTAINER, free,
                varied(renamed(ESCAPE), {ESCAPE + ".block_count": (STRING, "1")}))

    # The value types of the three numbers. Both programs read an unsigned type. bin/weight_sums
    # rejects the width or the number of layers in another type, and treats a head dimension in
    # another type as absent. weight_sums.py reads every numeric type. The head dimension is 64
    # here, which shows whether a program has read it.
    for name, value_type, unsigned in (("uint64", UINT64, True), ("int32", INT32, False), ("float32", FLOAT32, False)):
        cases.one("metadata.width-as-" + name, 0 if unsigned else CONTAINER, 0, free,
                  varied(METADATA, {width: (value_type, WIDTH)}), like=(METADATA, free))
        cases.one("metadata.layers-as-" + name, 0 if unsigned else CONTAINER, 0, free,
                  varied(METADATA, {layers: (value_type, 1)}), like=(METADATA, free))
        cases.one("metadata.head-dimension-64-as-" + name, BOUND if unsigned else 0, BOUND, free,
                  varied(METADATA, {head: (value_type, 64)}))
    # Neither a string of digits nor an array of one number stands for the number.
    cases.one("metadata.width-as-string", CONTAINER, BOUND, free, varied(METADATA, {width: (STRING, "5120")}))
    cases.one("metadata.width-as-array", CONTAINER, BOUND, free, varied(METADATA, {width: (ARRAY, (UINT32, [WIDTH]))}))
    cases.fault("metadata.layers-as-string", CONTAINER, free, varied(METADATA, {layers: (STRING, "1")}))
    # weight_sums.py reads a floating-point number, and finds that this one is not the width.
    cases.one("metadata.width-5120-and-a-half", CONTAINER, BOUND, free, varied(METADATA, {width: (FLOAT32, 5120.5)}))

    # A second shard may have no metadata, or metadata that contradict the first shard's. When the
    # same two shards are named in the other order, the other shard's metadata count, and swapped is
    # the expected status. The shard `first` gives the model two layers and has a tensor in layer 1.
    first = cases.shard("metadata.first", [("blk.1.attn_output.weight",) + PLAIN[1:]],
                        varied(METADATA, {layers: (UINT32, 2)}))
    for case, metadata, swapped in (
        ("no-metadata", [], CONTAINER),
        ("another-width", varied(METADATA, {width: (UINT32, 4096), layers: (UINT32, 2)}), BOUND),
        ("fewer-layers", METADATA, CONTAINER),
        ("another-architecture", varied(renamed("foo"), {"foo.block_count": (UINT32, 2)}), 0),
        ("another-head-dimension", varied(METADATA, {head: (UINT32, 64), layers: (UINT32, 2)}), BOUND),
    ):
        second = cases.shard("metadata.second-with-" + case, [EMBEDDING], metadata)
        cases.add("metadata.second-with-" + case, 0, 0, cases.start + [first, second])
        cases.add("metadata.first-with-" + case, swapped, swapped, cases.start + [second, first])
    empty = cases.shard("metadata.no-tensors", [], [])
    cases.add("metadata.second-without-tensors", 0, 0, cases.start + [first, empty])


def name_cases(cases):
    """Add the cases of tensor names: ignored forms, layer numbers out of range, repeated names."""
    # A name is that of a committed tensor or of a norm only if it has exactly the right form. These
    # names do not, so the programs ignore the tensors.
    cases.valid("name.ignored-forms", [PLAIN] + [(name,) + PLAIN[1:] for name in (
        "blk.00.attn_q.weight", "blk.01.attn_q.weight", "blk.+0.attn_q.weight", "blk.-1.attn_q.weight",
        "blk..attn_q.weight", "Blk.0.attn_q.weight", "blk.0.attn_q.weight ", "blk.0.attn_q.bias", "blk.0.attn_q",
        "blk.0.attn_x.weight", "output.weight.x")])

    # METADATA gives the model one layer, so 0 is the only valid layer number. bin/weight_sums
    # ignores a name whose layer number has ten digits or more, and weight_sums.py rejects it.
    def matrix(digits):
        return [PLAIN, ("blk.%s.ffn_down.weight" % digits,) + PLAIN[1:]]

    cases.fault("name.matrix-in-layer-1", CONTAINER, matrix("1"))
    cases.fault("name.norm-in-layer-1", CONTAINER, [PLAIN, ("blk.1.ffn_norm.weight",) + NORM[1:]])
    cases.fault("name.matrix-in-layer-999999999", CONTAINER, matrix("999999999"))
    cases.one("name.matrix-in-layer-1000000000", 0, CONTAINER, matrix("1000000000"), like=(METADATA, [PLAIN]))
    cases.one("name.norm-in-layer-1000000000", 0, CONTAINER, [PLAIN, ("blk.1000000000.ffn_norm.weight",) + NORM[1:]],
              like=(METADATA, [PLAIN]))
    # A layer number of 100,000 digits: no message may print such a name (see MOST_ON_STANDARD_ERROR).
    cases.one("name.matrix-in-a-layer-of-100000-digits", 0, CONTAINER, matrix("1" * 100000), like=(METADATA, [PLAIN]))

    for case, tensor in (("committed", QUANTIZED), ("norm", NORM), ("ignored", IGNORED)):
        cases.fault("name.%s-twice-in-one-shard" % case, CONTAINER, [PLAIN, tensor, tensor])
        first, second = cases.shard("name.%s-1" % case, [PLAIN, tensor]), cases.shard("name.%s-2" % case, [tensor], [])
        cases.add("name.%s-twice-in-two-shards" % case, CONTAINER, CONTAINER, cases.start + [first, second])


def type_cases(cases):
    """Add the cases of storage types. Where the type is defined, the data have the right length."""
    def typed(tensor, storage_type):
        return tensor[:2] + (storage_type, bytes(BYTES_OF_256[storage_type] * (len(tensor[3]) // 1024)))

    cases.fault("type.matrix-f64", TYPE, [QUANTIZED, typed(PLAIN, F64)])
    cases.fault("type.matrix-q4-0", TYPE, [QUANTIZED, typed(PLAIN, Q4_0)])
    cases.fault("type.embedding-q3-k", TYPE, [QUANTIZED, typed(EMBEDDING, Q3_K)])
    cases.fault("type.norm-bf16-read", TYPE, [READER, typed(NORM, BF16)])
    cases.valid("type.ignored-f16", [QUANTIZED, ("other.weight",) + typed(PLAIN, F16)[1:]])
    cases.fault("type.norm-q4-k-read", TYPE, [READER, typed(NORM, Q4_K)])
    cases.fault("type.norm-f16-not-read", TYPE, [PLAIN, typed(NORM, F16)])
    # The number of bytes of a tensor of an undefined type is not defined either, so such a tensor
    # is the last one in its file.
    cases.one("type.matrix-undefined", TYPE, CONTAINER, [QUANTIZED, PLAIN[:2] + (UNDEFINED_TYPE, bytes(32))])
    cases.one("type.ignored-undefined", 0, CONTAINER,
              [QUANTIZED, ("other.weight", (256, 1), UNDEFINED_TYPE, bytes(32))])


def shape_cases(cases):
    """Add the cases of the dimensions of norms and of committed tensors."""
    def norm(*dims):
        return (NORM[0], dims, F32, (GAINS + GAINS)[:4 * math.prod(dims)])

    cases.fault("shape.norm-of-5119", CONTAINER, [READER, norm(5119)])
    cases.fault("shape.norm-of-5121", CONTAINER, [READER, norm(5121)])
    cases.fault("shape.norm-of-256-not-read", CONTAINER, [PLAIN, norm(256)])
    cases.valid("shape.norm-1-by-5120", [READER, norm(1, WIDTH)])  # only the number of elements counts
    cases.valid("shape.matrix-with-its-norm", [NORM, READER])
    cases.fault("shape.rows-of-2560", CONTAINER, [NORM, (READER[0], (2560, 2), F32, LONG)])
    cases.fault("shape.rows-of-256", CONTAINER, [NORM, (READER[0], (256, 1), F32, SMALL)])
    cases.fault("shape.norm-absent", CONTAINER, [PLAIN, READER])
    cases.add("shape.norm-in-the-other-shard", 0, 0,
              cases.start + [cases.shard("shape.matrix-alone", [READER]), cases.shard("shape.norm-alone", [NORM], [])])
    # In an F32 tensor that reads no norm, the length of a row need not be a multiple of 256.
    for dims in ((3, 5), (7,)):
        data = ramp(math.prod(dims))
        cases.valid("shape.f32-" + "-by-".join(map(str, dims)),
                    [(PLAIN[0], dims, F32, data), (EMBEDDING[0], dims, F32, data)])
    cases.valid("shape.four-axes", [(QUANTIZED[0], (256, 2, 3, 2), Q4_K, blocks(Q4_K, 12))])
    # A row of a K-quant tensor is a whole number of blocks of 256 elements.
    cases.fault("shape.k-quant-rows-of-128", CONTAINER, [PLAIN, (QUANTIZED[0], (128, 2)) + QUANTIZED[2:]])
    cases.one("shape.ignored-k-quant-rows-of-128", 0, CONTAINER, [PLAIN, ("other.weight", (128, 2)) + QUANTIZED[2:]])
    # A row of a Q8_0 tensor is a whole number of blocks of 32 elements. F16 and BF16 have no blocks.
    cases.valid("shape.q8-0-rows-of-32", [(QUANTIZED[0], (32, 3), Q8_0, q8_0(3))])
    cases.fault("shape.q8-0-rows-of-48", CONTAINER, [PLAIN, (QUANTIZED[0], (48, 2), Q8_0, q8_0(3))])
    for storage_type, name in ((F16, "f16"), (BF16, "bf16")):
        data = struct.pack("<15H", *range(0x3C00, 0x3C0F))
        cases.valid("shape.%s-3-by-5" % name, [(PLAIN[0], (3, 5), storage_type, data)])

    # A command without committed tensors is valid and gives a file with the closing line only.
    cases.valid("shape.only-an-ignored-tensor", [IGNORED])
    cases.valid("shape.only-a-norm", [NORM])
    cases.valid("shape.no-tensor", [])


# Bit patterns of numbers in the formats of IEEE 754: binary16 for the block scales d and dmin,
# binary32 for everything else.
HALF_INFINITY, HALF_MINUS_INFINITY, HALF_NAN = 0x7C00, 0xFC00, 0x7E00
INFINITY, MINUS_INFINITY, NAN = 0x7F800000, 0xFF800000, 0x7FC00000
LARGEST = 0x7F7FFFFF  # the largest finite binary32 number
SMALLEST = 0x00000001  # the smallest positive binary32 number, 2^-149
MINUS_ZERO = 0x80000000


def value_cases(cases):
    """Add the cases of values that are not finite: stored ones, and results that overflow."""
    def quantized(storage_type, **scales):
        return [QUANTIZED[:2] + (storage_type, blocks(storage_type, 1, **scales))]

    cases.fault("value.q4-k-d-infinity", NOT_FINITE, quantized(Q4_K, d=HALF_INFINITY))
    cases.fault("value.q4-k-dmin-nan", NOT_FINITE, quantized(Q4_K, dmin=HALF_NAN))
    cases.fault("value.q5-k-d-nan", NOT_FINITE, quantized(Q5_K, d=HALF_NAN))
    cases.fault("value.q5-k-dmin-infinity", NOT_FINITE, quantized(Q5_K, dmin=HALF_INFINITY))
    cases.fault("value.q6-k-d-minus-infinity", NOT_FINITE, quantized(Q6_K, d=HALF_MINUS_INFINITY))
    cases.fault("value.q8-0-d-infinity", NOT_FINITE, [QUANTIZED[:2] + (Q8_0, q8_0(8, d=HALF_INFINITY))])
    cases.fault("value.q8-0-d-nan", NOT_FINITE, [QUANTIZED[:2] + (Q8_0, q8_0(8, d=HALF_NAN))])
    cases.valid("value.q8-0-subnormal-scale", [QUANTIZED[:2] + (Q8_0, q8_0(8, d=0x8003))])
    ones = [0x3C00] * 255  # as binary16: 1; as bfloat16: 2^-7
    for case, word in (("infinity", HALF_INFINITY), ("nan", HALF_NAN)):
        cases.fault("value.f16-" + case, NOT_FINITE, [QUANTIZED[:2] + (F16, struct.pack("<256H", *ones, word))])
    for case, word in (("minus-infinity", MINUS_INFINITY >> 16), ("nan", NAN >> 16)):
        cases.fault("value.bf16-" + case, NOT_FINITE, [QUANTIZED[:2] + (BF16, struct.pack("<256H", *ones, word))])
    # The pattern of binary16's infinity is the finite number 2^121 in BF16. The tensor's exponent is then below -100.
    with_2_to_the_121 = struct.pack("<256H", *ones, HALF_INFINITY)
    cases.fault("value.bf16-of-2-to-the-121", BOUND, [QUANTIZED[:2] + (BF16, with_2_to_the_121)])
    # Block scales that are subnormal binary16 numbers are valid.
    cases.valid("value.q4-k-subnormal-scales", quantized(Q4_K, d=0x0003, dmin=0x8001))

    # A comparison that catches an infinity can let a NaN pass, so each place below has both.
    for case, word in (("nan", NAN), ("infinity", INFINITY), ("minus-infinity", MINUS_INFINITY)):
        if case != "minus-infinity":
            cases.fault("value.matrix-" + case, NOT_FINITE, [PLAIN[:3] + (with_word(SMALL, 100, word),)])
            cases.fault("value.gain-" + case, NOT_FINITE, [READER, NORM[:3] + (with_word(GAINS, 100, word),)])
        if case != "infinity":
            cases.fault("value.embedding-" + case, NOT_FINITE, [EMBEDDING[:3] + (with_word(SMALL, 100, word),)])
    # Neither program looks at the values of an ignored tensor, or of a norm that no matrix of the
    # command reads.
    cases.valid("value.nan-in-an-ignored-tensor", [PLAIN, IGNORED[:3] + (with_word(IGNORED[3], 5, NAN),)])
    cases.valid("value.nan-in-a-norm-not-read", [PLAIN, NORM[:3] + (with_word(GAINS, 100, NAN),)])

    # The constant c is about 1.118 for attn_k, so the multiplier of the largest finite gain
    # overflows binary32.
    norm, matrix = "blk.0.attn_norm.weight", "blk.0.attn_k.weight"
    cases.fault("value.multiplier-overflows", NOT_FINITE,
                [(matrix,) + READER[1:], (norm,) + NORM[1:3] + (with_word(GAINS, 100, LARGEST),)])
    huge = floats([2.0 ** 100] * WIDTH)
    cases.fault("value.product-overflows", NOT_FINITE, [(matrix, (WIDTH, 1), F32, huge), (norm, (WIDTH,), F32, huge)])

    # Two tensors with faults of different statuses, and two threads or workers. The status must
    # not depend on which fault is found first. For bin/weight_sums the tensor that comes first by
    # its name counts, and for weight_sums.py the larger one. Here that is the same tensor, and its
    # fault shows only at its end, after the fault of the small one.
    two_jobs = [cases.start[0], "2", "--out", "out.txt"]
    large, small = ramp(64 * 256), ("blk.0.ffn_down.weight", (256, 1), F32)
    both = cases.shard("value.large-zeros-small-nan", [(PLAIN[0], (256, 64), F32, bytes(len(large))),
                                                       small + (with_word(SMALL, 0, NAN),)])
    cases.add("value.large-zeros-small-nan", BOUND, BOUND, two_jobs + [both])
    both = cases.shard("value.large-nan-small-zeros", [(PLAIN[0], (256, 64), F32, with_word(large, 64 * 256 - 1, NAN)),
                                                       small + (bytes(1024),)])
    cases.add("value.large-nan-small-zeros", NOT_FINITE, NOT_FINITE, two_jobs + [both])

    # Three shards in which a different tensor counts for each program, so the statuses differ. In
    # the first, the tensor that comes first by its name is the small one. In the other two the
    # tensors have equal sizes, so that for weight_sums.py the first in the tensor table counts.
    # Names are compared as bytes, so "blk.10." comes before "blk.2.".
    zeros, nan = bytes(1024), with_word(SMALL, 0, NAN)
    for case, cpp, py, tensors in (
        ("small-zeros-large-nan", BOUND, NOT_FINITE,
         [PLAIN[:3] + (zeros,), ("blk.0.ffn_down.weight", (256, 64), F32, with_word(large, 0, NAN))]),
        ("layer-2-nan-layer-10-zeros", BOUND, NOT_FINITE,
         [("blk.2.attn_output.weight", (256, 1), F32, nan), ("blk.10.attn_output.weight", (256, 1), F32, zeros)]),
        ("table-against-the-names", NOT_FINITE, BOUND, [small + (zeros,), PLAIN[:3] + (nan,)]),
    ):
        both = cases.shard("value." + case, tensors, varied(METADATA, {"llama4.block_count": (UINT32, 11)}))
        cases.add("value." + case, cpp, py, two_jobs + [both])


def bound_cases(cases):
    """Add the cases at the rule's bounds on a multiplier, mx, e and the embedding's integers."""
    def gain(word):
        return [READER, NORM[:3] + (with_word(GAINS, 100, word),)]

    # A multiplier must be 0 or at least 2^-100 in absolute value. For READER the constant c has the
    # binary32 bit pattern 0x3F8F1BBD. The gain with the pattern 0x0D64F92E gives exactly 2^-100,
    # and the next smaller gain gives less.
    cases.valid("bound.multiplier-2-to-the-minus-100", gain(0x0D64F92E))
    cases.fault("bound.multiplier-next-smaller", BOUND, gain(0x0D64F92D))
    cases.fault("bound.gain-2-to-the-minus-105", BOUND, gain(0x0B000000))
    cases.fault("bound.gain-smallest", BOUND, gain(SMALLEST))
    cases.valid("bound.gain-0", gain(0))
    # The constant c of attn_q is about 0.099, so the multiplier of the smallest gain rounds to 0,
    # which is valid.
    cases.valid("bound.multiplier-rounds-to-0",
                [("blk.0.attn_q.weight",) + READER[1:], ("blk.0.attn_norm.weight",) + gain(SMALLEST)[1][1:]])

    # The largest absolute value mx of a matrix's products must not be 0.
    cases.fault("bound.all-gains-0", BOUND, [READER, NORM[:3] + (bytes(4 * WIDTH),)])
    cases.fault("bound.matrix-of-zeros", BOUND, [PLAIN[:3] + (bytes(1024),)])
    cases.fault("bound.matrix-of-minus-zeros", BOUND, [PLAIN[:3] + (u32(MINUS_ZERO) * 256,)])

    # The exponent e of a matrix is the largest one with RNE(2^e * mx) <= 4,095, and must lie in
    # -100 .. 100. Each matrix here has mx = 2^power, so e = 11 - power.
    for case, power, status in (("100", -89, 0), ("101", -90, BOUND), ("minus-100", 111, 0), ("minus-101", 112, BOUND)):
        tensors = [PLAIN[:3] + (floats([2.0 ** power] * 256),)]
        if status:
            cases.fault("bound.exponent-" + case, status, tensors)
        else:
            cases.valid("bound.exponent-" + case, tensors)

    # Every integer RNE(2^e_res * v) of the embedding must be below 2^30 in absolute value. At the
    # default e_res = 11 the bound is at v = 2^19 = 524,288. The largest binary32 number below it
    # gives 2^30 - 64.
    def embedding(value):
        return [EMBEDDING[:3] + (put(SMALL, 400, floats([value])),)]

    cases.valid("bound.embedding-524287-and-a-half", embedding(524287.5))
    cases.valid("bound.embedding-largest", embedding(524288.0 - 2.0 ** -5))
    cases.fault("bound.embedding-524288", BOUND, embedding(524288.0))
    cases.fault("bound.embedding-minus-524288", BOUND, embedding(-524288.0))
    ones = cases.shard("bound.embedding-of-ones", [EMBEDDING[:3] + (floats([1.0] * 256),)])
    cases.add("bound.embedding-of-ones-at-29", 0, 0, cases.start + ["--eres", "29", ones], eres=29)
    cases.add("bound.embedding-of-ones-at-30", BOUND, BOUND, cases.start + ["--eres", "30", ones])
    # The field `bits` of the sums file is the least b >= 1 with -2^(b-1) <= w < 2^(b-1) for every
    # integer w of the tensor. It is 1, 1 and 2 for these three embeddings.
    for case, value in (("zeros", 0.0), ("integers-minus-1", -(2.0 ** -11)), ("integers-1", 2.0 ** -11)):
        cases.valid("bound.embedding-of-" + case, [EMBEDDING[:3] + (floats([value] * 256),)])


def found_at(path):
    """Describe what is at a path: None for nothing, a regular file's SHA-256, else a few words."""
    if os.path.islink(path):
        return "link to " + os.readlink(path)
    if os.path.isdir(path):
        return "folder"
    try:
        with open(path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except FileNotFoundError:
        return None
    except PermissionError:
        return "unreadable file"


def contents(top):
    """Return what a folder contains: found_at() for each path below it, relative to the folder."""
    found = {}
    for folder, folders, files in os.walk(top):
        for name in folders + files:
            found[os.path.relpath(os.path.join(folder, name), top)] = found_at(os.path.join(folder, name))
    return found


RUNNING = set()  # one file descriptor from os.pidfd_open for each program that is running
STARTING = threading.Lock()  # held while a program is started and its descriptor is noted


def end_everything(number, frame):
    """Handle SIGTERM or SIGINT: kill the programs that are running, and end the script.

    A signal to the process group of this script does not reach the programs, which have groups of
    their own. Each program is signalled through its descriptor, not by its process id: once a
    case's thread has waited for its program, the system may give the id to another process. A
    descriptor stands for the process for which it was opened, so the signal reaches that process or
    no one. A case's thread may also close a descriptor that is still in this function's copy of
    RUNNING. The call then fails, because the number is free or belongs to a file that another
    thread has opened. It cannot belong to another program: while cases run, os.pidfd_open is called
    only under the lock STARTING, which this function holds by then. The worker processes of
    weight_sums.py get no signal: each ends once its main process has ended (start_worker() in
    weight_sums/weight_sums.py).
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second signal must not come here again and wait for the lock
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    STARTING.acquire()  # from here on no case starts a program
    for descriptor in list(RUNNING):
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        except OSError:  # the program has just ended, or its thread has closed the descriptor
            pass
    os._exit(128 + number)


def group_members(group):
    """Return one line for each process of a process group: id, state, where it waits, and name.

    The lines come from the files /proc/<id>/stat and /proc/<id>/wchan of Linux (see the manual page
    proc(5)). In the first file the name is in parentheses, and after it come the state, the
    parent's id and the process group's id. On a system without /proc the list is empty.
    """
    found = []
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else []:
        try:
            with open("/proc/%s/stat" % entry) as file:
                text = file.read()
            state, _, its_group = text[text.rindex(")") + 1:].split()[:3]
        except (OSError, ValueError):  # not a process, or one that has just ended
            continue
        if its_group == str(group):
            try:
                with open("/proc/%s/wchan" % entry) as file:
                    waits = file.read().strip()
            except OSError:  # not every system lets this file be read
                waits = "?"
            found.append("%s %s %s %s" % (entry, state, waits, text[text.index("(") + 1:text.rindex(")")]))
    return found


def as_far_as_it_came(stream, data):
    """Return what a killed command had written to a stream, as lines for a report.

    The first line gives the number of lines. Of more than 40 lines, the first and last 20 follow.
    An escape character in the output is not replaced here.
    """
    lines = ["| " + line for line in data.decode(errors="replace").splitlines()]
    count = "%s, number of lines: %d" % (stream, len(lines))
    if len(lines) > 40:
        lines = lines[:20] + ["(%d lines are left out here)" % (len(lines) - 40)] + lines[-20:]
    return [count] + lines


def run_case(case, cases, program):
    """Run one case, and return a list of the points in which it did not end as expected."""
    here = os.path.join(cases.runs, case.name) + os.sep
    os.mkdir(here)
    for action in case.before:
        if action[0] == "file":
            with open(here + action[1], "xb") as file:
                file.write(b"old\n")
        elif action[0] == "dir":
            os.mkdir(here + action[1])
        elif action[0] == "link":
            os.symlink(action[2], here + action[1])
        elif action[0] == "copy":
            shutil.copyfile(action[1], here + action[2])
        else:
            os.chmod(here + action[1], action[2])

    def look():
        """Return the state of the case's folder and of the paths in in/ that the command names."""
        seen = contents(here)
        seen.update((word, found_at(word)) for word in case.arguments if word.startswith(cases.inputs))
        return seen

    arguments = "the arguments were: " + (shlex.join(case.arguments) or "none")
    wanted = look()
    if case.expected is not None:
        wanted[case.out] = hashlib.sha256(case.expected).hexdigest()
    try:
        # The program gets a process group of its own, so that after a timeout its worker processes
        # can be killed with it. It runs under the umask 000, so that a program that leaves its
        # file's permission bits to the umask is caught. PYTHONDONTWRITEBYTECODE keeps a Python that
        # is called without -B from writing __pycache__ folders, which under this umask everybody
        # could write.
        with STARTING:
            process = subprocess.Popen(program + case.arguments, cwd=here, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, umask=0,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            descriptor = os.pidfd_open(process.pid)  # not yet waited for, so the id is its own
            RUNNING.add(descriptor)
        printed, messages = process.communicate(timeout=SECONDS)
    except subprocess.TimeoutExpired:
        record = ["its processes when it was killed (id, state, where it waits in the kernel, name):"]
        record += group_members(process.pid)
        os.killpg(process.pid, signal.SIGKILL)
        printed, messages = process.communicate()
        record += as_far_as_it_came("standard output", printed) + as_far_as_it_came("standard error", messages)
        return ["it did not end within %d seconds" % SECONDS, arguments,
                "what is known of it:\n    " + "\n    ".join(record)]
    except OSError as error:
        return ["the program could not be started: %s" % error.strerror, arguments]
    else:
        found = look()
    finally:
        if "descriptor" in locals():
            RUNNING.discard(descriptor)
            os.close(descriptor)
        for action in case.before:
            if action[0] == "mode" and os.path.lexists(here + action[1]):  # so that the folder can be removed
                os.chmod(here + action[1], 0o700 if os.path.isdir(here + action[1]) else 0o600)

    wrong = []
    if process.returncode < 0:
        wrong.append("signal %d ended it" % -process.returncode)
    elif process.returncode != case.status:
        wrong.append("the exit status is %d and not %d" % (process.returncode, case.status))
    if case.status == 0 and case.out is None:
        if not printed:
            wrong.append("no help text on standard output")
    elif printed:
        wrong.append("it wrote to standard output")
    for path in sorted(set(wanted) | set(found)):
        if found.get(path) == wanted.get(path):
            continue
        if case.expected is not None and path == case.out:
            wrong.append(("%s is not the expected file" if found.get(path) else "it wrote no file at %s") % path)
        elif path.startswith(cases.inputs):  # other cases use the same file
            wrong.append("%s changed while the command ran (a case that ran at the same time may have done it)" % path)
        elif wanted.get(path) is None:
            wrong.append("it made %s, which should not exist" % path)
        else:
            wrong.append("it %s %s" % ("changed" if found.get(path) else "removed", path))
    if case.expected is not None:
        if found.get(case.out) == wanted[case.out]:
            bits = stat.S_IMODE(os.stat(here + case.out).st_mode)
            if bits != 0o600:
                wrong.append("the permission bits of %s are %o and not 600" % (case.out, bits))
        if any(line.split()[-1] in messages for line in case.expected.splitlines()[:-1]):
            wrong.append("a SHA-256 of the sums file is on standard error")
    if b"Traceback" in messages or b"Warning" in messages:
        wrong.append("a traceback or a warning is on standard error")
    if b"remove" in messages.lower():
        wrong.append("standard error tells the user to remove something")
    if b"\x1b" in messages or ESCAPE[4:].encode() in messages:
        wrong.append("text that the case has put into the model file is on standard error")
    if len(messages) > MOST_ON_STANDARD_ERROR:
        wrong.append("standard error has %d bytes, more than %d" % (len(messages), MOST_ON_STANDARD_ERROR))
    # The quotation below must not send an escape character to the terminal either.
    shown = messages.decode(errors="replace").replace("\x1b", "\\x1b")
    lines = [line.strip() for line in shown.split("\n") if line.strip()]
    if case.status != 0 and not lines:
        wrong.append("no message on standard error")
    if wrong and lines:
        # The reason for a rejection is the first line if a usage text follows it, and otherwise the
        # last line. The inputs are synthetic, so nothing secret can show here.
        wrong.append("standard error says: " + " ... ".join(dict.fromkeys([lines[0][:200], lines[-1][:200]])))
    return wrong + [arguments] if wrong else []


def main(fill):
    """Do what the command line asks for, and return the exit status.

    fill is the function that adds the script's cases to a Cases object: edge.py passes its own.
    """
    script, arguments = os.path.basename(sys.argv[0]), sys.argv[1:]
    if not (len(arguments) > 4 and arguments[1] == "--kind" and arguments[2] in ("cpp", "py") and arguments[3] == "--"):
        print("usage: %s <new folder> --kind cpp|py -- <program and its first arguments>\nThe kind cpp says that the "
              "program is bin/weight_sums, and py that it is weight_sums.py. Give absolute paths after \"--\"."
              % script, file=sys.stderr)
        return 2
    if shutil.which(arguments[4]) is None:
        print("%s: %s is not a program that can be run. Give its absolute path" % (script, arguments[4]),
              file=sys.stderr)
        return 2
    try:
        os.close(os.pidfd_open(os.getpid()))  # tried before any program is started: end_everything() depends on it
    except (AttributeError, OSError):
        print("%s: this system has no os.pidfd_open, which needs Python 3.9 and Linux 5.3 or later" % script,
              file=sys.stderr)
        return 2
    program = [os.path.abspath(shutil.which(arguments[4]))] + arguments[5:]
    folder = os.path.abspath(arguments[0])
    try:
        os.makedirs(folder, 0o700)  # closed to other users: a faulty program may leave files that anybody may write
    except OSError as error:
        print("%s: cannot make the new folder %s: %s" % (script, folder, error.strerror), file=sys.stderr)
        return 2
    cases = Cases(folder, arguments[2])
    fill(cases)

    ran = [case for case in cases.rows if os.geteuid() != 0 or not any(action[0] == "mode" for action in case.before)]
    inputs = contents(cases.inputs)
    failed = 0
    signal.signal(signal.SIGTERM, end_everything)
    signal.signal(signal.SIGINT, end_everything)
    # A caller that ignores SIGCHLD passes that setting on to this script. The kernel would then
    # reap each program as soon as it ends: its exit status would be lost and read as 0, and its
    # process id could go to another process while run_case() still uses it.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    # The cases share the files in in/, which no command may change.
    with concurrent.futures.ThreadPoolExecutor(AT_A_TIME) as pool:
        for case, result in [(case, pool.submit(run_case, case, cases, program)) for case in ran]:
            # Python handles a signal in the main thread only, which is this one (see Python's
            # documentation of the module signal). If another thread has received the signal,
            # nothing wakes this one, so it waits for a result for at most a second at a time.
            while not concurrent.futures.wait([result], timeout=1).done:
                pass
            if result.result():
                failed += 1
                print("%s: %s: %s" % (script, case.name, "; ".join(result.result())))
    now = contents(cases.inputs)
    touched = sorted(path for path in set(inputs) | set(now) if inputs.get(path) != now.get(path))
    if touched:
        print("%s: after all cases, these paths in %s are not as they were: %s"
              % (script, cases.inputs, ", ".join(touched)))
    strays = sorted(set(os.listdir(cases.runs)) - set(case.name for case in ran))
    if strays:
        print("%s: these paths in %s should not exist: %s" % (script, cases.runs, ", ".join(strays)))
    if len(ran) < len(cases.rows):
        print("%s: %d cases that depend on permission bits were left out, because the superuser may read and write "
              "every file" % (script, len(cases.rows) - len(ran)))
    if failed:
        print("%s: the folder of each case is %s" % (script, os.path.join(cases.runs, "<case>")))
    print("%s: %d cases ran, %d did not end as expected" % (script, len(ran), failed))
    return 1 if failed or touched or strays else 0


def all_cases(cases):
    command_cases(cases)
    path_cases(cases)
    container_cases(cases)
    metadata_cases(cases)
    name_cases(cases)
    type_cases(cases)
    shape_cases(cases)
    value_cases(cases)
    bound_cases(cases)


if __name__ == "__main__":
    sys.exit(main(all_cases))
