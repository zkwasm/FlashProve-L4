#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Write a model's fitted quantization scales into two public files of its configuration.

usage: words_to_policy.py BASE_FILE FITTED_VALUES SUMS_FILE NEW_PARAMETER_FILE NEW_SCALES_FILE

BASE_FILE is a parameter file (policy.txt) of the model's shape. FITTED_VALUES is the words.json
that fit_scales.py, calibrate.py or worst_case.py writes. SUMS_FILE is the sums file (wsums.txt)
that a program of weight_sums/ writes for the same weights, or "-" for none. NEW_PARAMETER_FILE
becomes BASE_FILE with new values in the 15 written lines (WRITTEN_LINES) and, if it has digest
lines (digests of the public SiLU, sigmoid, rotary and softmax tables), new ones (see new_lines()).
NEW_SCALES_FILE becomes the scales file (KL.bin). The programs cp_commit (the committer), cp_prove
(the prover) and cp_verify (the verifier) read both files. This program overwrites nothing: it
rejects a call if anything exists at an output path. It either writes both files or rejects the
input and writes nothing.

It leaves out some of the verifier's checks, so a successful run does not show that the verifier
accepts the files. To check that, run cp_verify -plan request, which checks the public inputs and
reads no proof. The call is shown in scales/README.md, "What the tools do not check".

words.json is an object with these entries:
  "KL"        a list with one object per layer: LAYER_NAMES, and in a mixture-of-experts layer also
              EXPERT_NAMES
  "W"         the model-wide values: MODEL_NAMES
  "ew"        each weight tensor's power-of-two exponent, by the tensor's name
  "emb_bits"  the width of the embedding's integers, in bits
Every value is a JSON integer, except that theta, the rotary base, may be digits followed by ".0".
With "-" as SUMS_FILE, "emb_bits" is required and "ew" must be empty. fit_scales.py writes an empty
"ew", for such a call, and calibrate.py a full one, for a call with the sums file. S, rx and cs are
constants of the softmax and rotary tables. e_res and E_logit are the power-of-two exponents of the
residual stream and of the logits.
Every name is defined in scales/README.md, "The recorded sites and the scales".

All output goes to standard output. Every line begins with "w2p ", and the last line is
"w2p end <exit status>". The fitted values depend on the weights, so the program prints names,
counts and SHA-256 digests, and never a value. A successful run exits with status 0; convert()
lists its lines.

The checks run in the steps of the table below. A row with a status begins a step, and the rows
without one belong to the step above them. The run stops at the first step in which something
fails. It then prints one line "w2p refuse <class> <count>" for each failing class of that step,
sorted by name, and exits with the step's status. From the step of class eres on, the line
"w2p ew" comes before these lines. The count is 1 unless the table says what it counts. For the
classes ew, ew-cover and wide the numbers of tensors are in the line "w2p ew".

At a rounding site, a named point of the forward pass, an accumulator is divided by 2^k and
rounded; the result is the site's quotient. k is the site's shift, one of the ten names in FORM_OF.
The shift's digit form is a line of the base file: the number of bytes (0, 1 or 2) for the
remainder's bits above the low 16, so that the largest shift is 16, 24 or 32. The four sites of the
table are the outputs of the up projections (up), of the down projections (down), of attention's
output projection (o) and of the query and key projections (qk).

  status  class                what failed
  2       call                 The call does not have five arguments that are not empty, or an
                               output is "-", or the two outputs are the same. A usage line comes
                               first.
  21      out-exists           Something exists at an output path, or an output path is also an
                               input.
  20      input-base           The base file cannot be read or is not ASCII.
  20      input-words          words.json cannot be read, is not ASCII, has a name twice in one
                               object, or is not JSON as Python's json module reads it.
  20      input-wsums          The sums file cannot be read or is not ASCII.
  20      base                 The base file is missing a line that the program reads
                               (READ_LINES), or its value is out of range, or it has a line xl
                               with a value other than 5.
  20      input-words          words.json is not an object with a list "KL" of one object per
                               layer and an object "W".
  20      word-absent          A value is absent or is not a whole number in its range. The count
                               has 1 for each such layer, 1 for each such name of "W", and 1 if
                               "ew" or "emb_bits" does not have its type.
          word-unknown         A layer's object has a name that does not belong to it. Counts
                               layers.
  4       form                 A shift is below 1 or above the largest that its digit form sets.
                               Counts shifts.
          norm                 A layer's pair (M1, K1) or (M2, K2) is out of range. M must be in
                               24 .. 47 and K in 17 .. 32, with M - K at most 14. Counts pairs.
          final-norm           The pair (MF, KF) is out of range, by the same rule.
          rp                   rp is neither 0 nor 1. Counts layers.
          logit-exponent       E_logit is above 63.
          arith-eps            eps1, eps2 or epsF is not d * 1e-5 * 4^e_res, rounded to the
                               nearest integer. Counts the three names.
                               If the rounded number is 0, all three count.
          arith-const          S, rx, cs, theta are not 12, 8, 12, 500000. Counts the four names.
  3       law-U-<site>,        A no-overflow condition (see check_no_overflow()) does not hold at
          law-W-<site>         the site up, down, o or qk. Counts shifts.
          law-kq-kk            kq and kk differ in a layer in which one of them is below 16.
                               Counts layers.
          law-sh               sh is outside 4 .. 12. Counts layers.
  5       emb-bits-absent      There is no sums file, and "emb_bits" is absent.
          ew-without-wsums     There is no sums file, and "ew" is not empty.
  20      wsums                The sums file does not have the form that parse_sums() describes.
  5       eres                 The embedding's exponent in the sums file is not e_res.
          emb-bits             The embedding's width is outside 2 .. 21 bits, or "emb_bits"
                               differs from the sums file's. The count is 2 if both hold.
          ew                   An entry of "ew", other than the embedding's, is not in the sums file
                               with the same exponent.
          ew-cover             Except for the embedding, "ew" and the sums file do not name the same
                               tensors.
          wide                 A tensor other than the embedding has more than 13 bits.
  20      base-line-twice      The base file has one of the 15 written lines more than once.
  20      base-line-absent     The base file is missing one of the 15 written lines. Counts names.
  20      input-policy-tables  The base file has digest lines, and policy_tables.py is not next to
                               this program or cannot compute them.
  21, 22  out-exists, write    An output file could not be created or written. What the run
                               created is removed.
  23      internal             Anything else. What the run created is removed.
"""

import hashlib
import importlib.util
import json
import os
import re
import struct
import sys

USAGE = "usage: words_to_policy.py BASE_FILE FITTED_VALUES SUMS_FILE NEW_PARAMETER_FILE NEW_SCALES_FILE"

# The exit status of each rejection class.
STATUS = {
    "call": 2,
    "law-U-up": 3, "law-U-down": 3, "law-U-o": 3, "law-U-qk": 3,
    "law-W-up": 3, "law-W-down": 3, "law-W-o": 3, "law-W-qk": 3, "law-kq-kk": 3, "law-sh": 3,
    "form": 4, "norm": 4, "rp": 4, "final-norm": 4, "logit-exponent": 4, "arith-eps": 4, "arith-const": 4,
    "ew": 5, "ew-cover": 5, "eres": 5, "wide": 5, "emb-bits": 5, "emb-bits-absent": 5, "ew-without-wsums": 5,
    "input-base": 20, "input-words": 20, "input-wsums": 20, "base": 20, "base-line-absent": 20, "base-line-twice": 20,
    "word-absent": 20, "word-unknown": 20, "wsums": 20, "input-policy-tables": 20,
    "out-exists": 21, "write": 22, "internal": 23,
}  # fmt: skip

# The permission bits with which the two output files are created, before the process's umask.
OUT_MODE = 0o600

# The lines of the base file that read_shape() requires, each with its smallest and largest value:
# layers (L), routed experts (NE), width of the residual stream (d), hidden width of a dense
# layer's feed-forward network (f), query heads (H), head dimension (dh), 1 if an up projection's
# output is limited to 16 bits (ut), width of the embedding's integers in bits (wb_emb), and the
# seven digit forms (see the module's docstring). The program checks neither the file's grammar
# nor the order of its lines. The verifier does, and it rejects a file in which a line is repeated.
READ_LINES = {"L": (1, 256), "NE": (0, 4096), "d": (1, 2**20), "f": (1, 2**20), "H": (1, 4096), "dh": (1, 4096),
              "ut": (0, 1), "wb_emb": (0, 64),
              "hq": (0, 2), "hv": (0, 2), "ho": (0, 2), "hgD": (0, 2), "huD": (0, 2), "hdD": (0, 2), "hr": (0, 2)}  # fmt: skip
# A value line: a name, one space, and one to ten digits after an optional minus sign.
VALUE_LINE = re.compile(r"([^ ]+) (-?[0-9]{1,10})")
# The lines of the base file that new_lines() rewrites. A line whose name ends in `_lo` has a
# value's low 32 bits, and one in `_hi` the rest. theta0_lo and theta0_hi get theta, and the line s
# gets E_logit.
WRITTEN_LINES = ("eps1_lo", "eps1_hi", "eps2_lo", "eps2_hi", "epsF_lo", "epsF_hi", "theta0_lo", "theta0_hi",
                 "MF", "KF", "S", "rx", "cs", "s", "wb_emb")  # fmt: skip
# The name of a digest line, which has one 32-bit word of a public table's digest.
DIGEST_NAME = re.compile(r"(tsilu|tsig|trope|texp)[0-9]*")

# LAYER_NAMES and EXPERT_NAMES are in the order of KL.bin.
LAYER_NAMES = ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "M1", "K1", "M2", "K2", "sh", "rp")
EXPERT_NAMES = ("kr", "kgR", "kuR")
MODEL_NAMES = ("eps1", "eps2", "epsF", "MF", "KF", "S", "rx", "cs", "theta", "e_res", "E_logit")
# The line of the base file that holds each shift's digit form.
FORM_OF = {"kq": "hq", "kk": "hq", "kv": "hv", "ko": "ho", "kg": "hgD", "kgR": "hgD", "ku": "huD", "kuR": "huD",
           "kd": "hdD", "kr": "hr"}  # fmt: skip

# The embedding tensor's name in "ew" and in the sums file.
EMBEDDING = "token_embd.weight"

# Bounds used in the no-overflow conditions: on a clamped activation, on a weight and on an entry
# of the SiLU table.
ACTIVATION, WEIGHT, GATE_OUTPUT = 2**11, 2**12, 2**12
CELL = 2**18  # the bound on a cell, one committed integer
TWO_CELLS = 257 * CELL  # the bound on a value stored in two cells
# The bound on the quotient at the sites o and down of check_no_overflow(), by the value of the
# optional line xl. A value of the residual stream has four limbs (signed bytes) without the line
# and five with "xl 5".
STREAM_QUOTIENT = {None: 2**31, 5: 2**40}


class Rejected(Exception):
    """Stops the run. `counts` maps each failing class of one step to its count."""

    def __init__(self, counts):
        super().__init__()
        self.counts = counts


def reject(name):
    raise Rejected({name: 1})


def reject_if_any(counts):
    """Rejects with the classes whose count is not 0."""
    counts = {name: int(count) for name, count in counts.items() if count}
    if counts:
        raise Rejected(counts)


def check_call(arguments):
    if len(arguments) != 5 or "" in arguments or "-" in arguments[3:] or arguments[3] == arguments[4]:
        print("w2p " + USAGE)
        reject("call")


def check_outputs(inputs, outputs):
    """A dangling symbolic link at an output path counts as something that exists. Paths are
    compared with symbolic links resolved, whether or not the input exists."""
    used = {os.path.realpath(path) for path in inputs}
    if any(os.path.lexists(path) or os.path.realpath(path) in used for path in outputs):
        reject("out-exists")


def read_ascii(path, name):
    """Returns the file's text. Rejects with the class `name` if the file cannot be read or is not
    ASCII."""
    try:
        with open(path, "rb") as f:
            return f.read().decode("ascii")
    except (OSError, UnicodeDecodeError):
        reject(name)


class NotAnInteger(str):
    """The text of a JSON number that has a fraction or an exponent."""


def object_without_repeats(pairs):
    """Raises ValueError for an object that has a name twice. The JSON standard, RFC 8259, says in
    section 4: "The names within an object SHOULD be unique"."""
    if len({name for name, _ in pairs}) < len(pairs):
        raise ValueError("an object has the same name twice")
    return dict(pairs)


def parse_json(text):
    """Python's parser also accepts NaN, Infinity and -Infinity, which the JSON standard, RFC 8259,
    section 6, does not permit. They pass here, as values that are not integers."""
    try:
        return json.loads(text, object_pairs_hook=object_without_repeats, parse_float=NotAnInteger)
    except (ValueError, RecursionError):
        reject("input-words")


def read_shape(lines):
    """Returns {name: value} for the lines of READ_LINES and for the line xl (None if absent). Where
    a name occurs more than once, its last value line is used."""
    values = {m[1]: int(m[2]) for m in map(VALUE_LINE.fullmatch, lines) if m}
    if not all(name in values and low <= values[name] <= high for name, (low, high) in READ_LINES.items()):
        reject("base")
    if values.get("xl") not in STREAM_QUOTIENT:
        reject("base")
    return {name: values.get(name) for name in (*READ_LINES, "xl")}


def has_experts(shape, l):
    """If NE > 0, the odd layers, counted from 0, are the mixture-of-experts layers."""
    return shape["NE"] > 0 and l % 2 == 1


def check_structure(words, shape):
    if not (isinstance(words, dict) and isinstance(words.get("KL"), list) and isinstance(words.get("W"), dict)):
        reject("input-words")
    if len(words["KL"]) != shape["L"] or not all(isinstance(layer, dict) for layer in words["KL"]):
        reject("input-words")


def is_integer(value):
    return type(value) is int  # not isinstance(): True and False are instances of int


def is_whole(value, limit):
    return is_integer(value) and 0 <= value < limit


def stated_width(words):
    """Returns "emb_bits", or None if it is absent. null counts as absent, and so do NaN, Infinity
    and -Infinity, which are the only values that the parser returns as floats."""
    value = words.get("emb_bits")
    return None if isinstance(value, float) else value


def check_words(words, shape):
    """Checks presence, type and range: 0 .. 2^32 - 1 in "KL", 0 .. 2^53 - 1 in "W", 0 .. 24 for
    e_res. An unknown name is rejected in a layer's object, and ignored in "W" and at the top
    level. An absent "emb_bits" passes here. A theta that ends in ".0" is replaced by the
    integer."""
    model, exponents = words["W"], words.get("ew")
    theta = model.get("theta")
    if isinstance(theta, NotAnInteger) and re.fullmatch(r"[0-9]{1,16}\.0", theta):  # 2^53 has 16 digits
        model["theta"] = int(theta[:-2])
    absent = sum(not is_whole(model.get(name), 25 if name == "e_res" else 2**53) for name in MODEL_NAMES)
    absent += not (isinstance(exponents, dict) and all(is_integer(e) for e in exponents.values())
                   and (stated_width(words) is None or is_integer(stated_width(words))))  # fmt: skip
    unknown = 0
    for l, layer in enumerate(words["KL"]):
        names = LAYER_NAMES + (EXPERT_NAMES if has_experts(shape, l) else ())
        absent += not all(is_whole(layer.get(name), 2**32) for name in names)
        unknown += not all(name in names for name in layer)
    reject_if_any({"word-absent": absent, "word-unknown": unknown})


# The verifier's limits on a norm's pair (M, K): the norm multiplies by a reciprocal that is scaled
# by 2^M and then shifts right by K bits. The verifier answers `YES` only if also K <= M (its
# condition norm-shifts). This program does not check that.
def norm_is_admitted(m, k):
    return 24 <= m <= 47 and m - 14 <= k and 17 <= k <= 32


# The verifier also applies the limits of `form`, `norm`, `final-norm`, `rp` and `logit-exponent`,
# and it requires cs = 12. Otherwise `arith-eps` and `arith-const` are stricter than the verifier:
# this program accepts only a model with the norm epsilon 1e-5 and the rotary base 500000, and only
# S = 12 and rx = 8 (defined in scales/README.md, "The fixed exponents"). The verifier rejects fewer
# values of S and rx, but it answers `YES` only with these two values (its condition
# softmax-table) and with a rotary base of 10000 or 500000 (rotary-table).
def check_ranges(layers, model, shape):
    shifts = [(layer[name], shape[h]) for layer in layers for name, h in FORM_OF.items() if name in layer]
    norms = [(layer[m], layer[k]) for layer in layers for m, k in (("M1", "K1"), ("M2", "K2"))]
    # The integer model's norm epsilon (docs/model.md, "3 Weights and quantization scales", which
    # calls it the regularizer) is d * 4^e_res / 100,000, rounded to the nearest integer (the
    # rule: scales/README.md, "The epsilon of a norm"). The verifier rejects an epsilon of 0.
    # This program's formula is exact, and calibrate.py and fit_scales.py are not: calibrate.py
    # takes the epsilon from the model file, which stores it as a float32 number, and both round a
    # product in binary64. At d = 5,120 a model file's epsilon gives the exact integer up to
    # e_res = 14. From 15 on it gives a different one, and this program rejects the file.
    eps = (2 * shape["d"] * 4 ** model["e_res"] + 100000) // 200000
    constants = (("S", 12), ("rx", 8), ("cs", 12), ("theta", 500000))
    reject_if_any({
        "form": sum(not 1 <= k <= 16 + 8 * form for k, form in shifts),
        "norm": sum(not norm_is_admitted(m, k) for m, k in norms),
        "rp": sum(layer["rp"] > 1 for layer in layers),
        "final-norm": not norm_is_admitted(model["MF"], model["KF"]),
        "logit-exponent": model["E_logit"] > 63,
        "arith-eps": sum(model[name] != eps or eps == 0 for name in ("eps1", "eps2", "epsF")),
        "arith-const": sum(model[name] != value for name, value in constants),
    })  # fmt: skip


def quotient_is_unique(k, span):
    """The condition of the classes law-U-<site>, for a rounding site with the shift k. `span` is
    the number of integers in the quotient's range."""
    return span * 2 ** max(k, 16) <= 2**60


def key_does_not_overflow(k, form, accumulator, quotient):
    """The condition of the classes law-W-<site>, for a rounding site with the shift k and the digit
    form `form`. `accumulator` and `quotient` are bounds on the site's accumulator and quotient."""
    high_digit = (0, CELL + 128, TWO_CELLS + 32896)[form]
    return 2 ** max(0, 16 - k) * (accumulator + 2 ** (k - 1) + quotient * 2**k) + 2**16 * high_digit < 2**60


def check_no_overflow(layers, model, shape):
    """The verifier also applies every condition that is checked here.

    quotient_is_unique() and key_does_not_overflow() are no-overflow conditions, a term of
    verifier/docs/parameter-file.md, "The optional lines on the model's arithmetic".
    Each compares a bound with 2^60, which is below the field's prime, p = 2^64 - 2^32 + 1. They are
    meant to ensure that an equation that a proof checks modulo p also holds over the integers.
    That they do so is not proved. The verifier applies them at more sites, under the names
    no-wrap-U-<site> and no-wrap-W-<site> (verifier/src/cp_verify.cpp)."""
    product = shape["d"] * ACTIVATION * WEIGHT
    stream = STREAM_QUOTIENT[shape["xl"]]
    counts = dict.fromkeys(STATUS, 0)
    for layer in layers:
        if shape["ut"] == 1:
            up_span, up_quotient, up_output = 2**16, 2**15, 2**15
        else:
            # The verifier rejects this case, ut 0. At the site up, the projection's output is
            # bounded only as a value of two cells. At the site down, its bound follows from the
            # accumulator's bound at the site up and the smaller of the layer's up shifts.
            k = min(layer[name] for name in ("ku", "kuR") if name in layer)
            up_span, up_quotient, up_output = 2 * TWO_CELLS + 1, TWO_CELLS, (product + 2 ** (k - 1)) // 2**k + 1
        sites = (  # name, shifts, accumulator, span, quotient
            ("up", ("ku", "kuR"), product, up_span, up_quotient),
            ("down", ("kd",), shape["f"] * GATE_OUTPUT * up_output * WEIGHT, 2 * stream, stream),
            ("o", ("ko",), shape["H"] * shape["dh"] * ACTIVATION * WEIGHT, 2 * stream, stream),
            ("qk", ("kq", "kk"), 2 * 2 ** model["cs"] * product, 2**16, 2**15),
        )
        for site, shifts, accumulator, span, quotient in sites:
            form = shape[FORM_OF[shifts[0]]]
            for k in (layer[name] for name in shifts if name in layer):
                counts["law-U-" + site] += not quotient_is_unique(k, span)
                counts["law-W-" + site] += not key_does_not_overflow(k, form, accumulator, quotient)
        counts["law-kq-kk"] += layer["kq"] != layer["kk"] and min(layer["kq"], layer["kk"]) < 16
        counts["law-sh"] += not 4 <= layer["sh"] <= 12
    reject_if_any(counts)


def parse_sums(text):
    """Returns {tensor name: (exponent, bits)} from the text of a sums file.

    A tensor's line has five fields with single spaces between them: name, elements,
    exponent, bits, SHA-256 (defined in weight_sums/README.md, "The file's form"). The exponent
    is one to ten digits after an optional minus sign, and the bits are one to ten digits. No name
    occurs twice, and the embedding must be present. The last line is "END <number of tensors>".
    Empty lines are skipped. The order of the lines, the elements and the SHA-256 are not
    checked."""
    lines = [line for line in text.split("\n") if line]
    if not lines or lines[-1] != "END %d" % (len(lines) - 1):
        reject("wsums")
    sums = {}
    for name, *fields in (line.split(" ") for line in lines[:-1]):
        if len(fields) != 4 or name in sums or not re.fullmatch("-?[0-9]{1,10} [0-9]{1,10}", " ".join(fields[1:3])):
            reject("wsums")
        sums[name] = (int(fields[1]), int(fields[2]))
    if EMBEDDING not in sums:
        reject("wsums")
    return sums


def check_exponents(words, sums_text):
    """Compares "ew", "emb_bits" and e_res with the sums file.

    Without a sums file it prints "w2p ew none", and the embedding's width is "emb_bits".

    With a sums file it prints "w2p ew <a> of <b> absent <c> wide <e> tensors <n>": "ew" has b
    entries, of which a have the sums file's exponent and c are not in that file; the file has n
    tensors, of which e have more than 13 bits. Only n counts the embedding."""
    emb_bits = stated_width(words)
    if sums_text is None:
        reject_if_any({"emb-bits-absent": emb_bits is None, "ew-without-wsums": len(words["ew"]) > 0})
        print("w2p ew none")
        reject_if_any({"emb-bits": not 2 <= emb_bits <= 21})
        return
    sums = parse_sums(sums_text)
    emb_exponent, bits = sums.pop(EMBEDDING)
    exponents = {name: e for name, e in words["ew"].items() if name != EMBEDDING}
    same = sum(name in sums and sums[name][0] == e for name, e in exponents.items())
    absent = sum(name not in sums for name in exponents)
    wide = sum(width > 13 for _, width in sums.values())
    print("w2p ew %d of %d absent %d wide %d tensors %d" % (same, len(exponents), absent, wide, len(sums) + 1))
    reject_if_any({
        "eres": emb_exponent != words["W"]["e_res"],
        "emb-bits": (emb_bits is not None and emb_bits != bits) + (not 2 <= bits <= 21),
        "ew": same < len(exponents),
        "ew-cover": set(exponents) != set(sums),
        "wide": wide > 0,
    })  # fmt: skip


def digest_lines(lines):
    """Returns the digest lines, which policy_tables.py computes from the other lines of a parameter
    file. The module is loaded from this file's directory, with symbolic links resolved."""
    sys.dont_write_bytecode = True  # loading the module leaves no __pycache__ directory next to it
    try:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "policy_tables.py")
        spec = importlib.util.spec_from_file_location("policy_tables", path)
        policy_tables = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_tables)
        return policy_tables.table_lines(policy_tables.digests(lines))
    except (OSError, ValueError):  # the file is missing, or `digests` cannot compute the tables
        reject("input-policy-tables")


def new_lines(lines, model):
    """Returns the lines of the new parameter file, and how many of the 15 written lines get a
    different value.

    The new file is the base file, byte for byte, with two exceptions: the lines of WRITTEN_LINES
    get new values, and the digest lines, if the base file has any, are computed again."""
    # A written line is its name, one space, and text without a space, which may be empty and need
    # not be a number.
    fields = [line.split(" ") for line in lines]
    places = {name: [i for i, f in enumerate(fields) if len(f) == 2 and f[0] == name] for name in WRITTEN_LINES}
    reject_if_any({"base-line-twice": any(len(at) > 1 for at in places.values())})
    reject_if_any({"base-line-absent": sum(not at for at in places.values())})
    # 21 bits is the only width that the verifier accepts for the embedding.
    values = {"MF": model["MF"], "KF": model["KF"], "S": model["S"], "rx": model["rx"], "cs": model["cs"],
              "s": model["E_logit"], "wb_emb": 21}  # fmt: skip
    for line, name in (("eps1", "eps1"), ("eps2", "eps2"), ("epsF", "epsF"), ("theta0", "theta")):
        values[line + "_lo"], values[line + "_hi"] = model[name] % 2**32, model[name] // 2**32
    lines, moved = list(lines), 0
    for name, (at,) in places.items():
        old = VALUE_LINE.fullmatch(lines[at])
        moved += old is None or int(old[2]) != values[name]
        lines[at] = "%s %d" % (name, values[name])
    # All digest lines that the base file has, however many and wherever, are replaced by the full
    # set, computed from the new values, at the position of the first. In a file whose lines are in
    # the format's order, that is directly before the first call line (`call ...`, a lookup call).
    digests = [i for i, line in enumerate(lines) if DIGEST_NAME.fullmatch(line.split(" ")[0])]
    if digests:
        before = lines[: digests[0]]
        after = [line for i, line in enumerate(lines) if i > digests[0] and i not in digests]
        lines = before + digest_lines(before + after) + after
    return lines, moved


def create(path, data, created):
    """Writes a new file and adds it to `created` as soon as the file exists, so that main() can
    remove it. Class out-exists: something has appeared at the path since check_outputs()."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, OUT_MODE)
        created.append((path, os.fstat(descriptor)))
        with os.fdopen(descriptor, "wb") as f:
            f.write(data)
    except FileExistsError:
        reject("out-exists")
    except OSError:
        reject("write")


def remove_created(path, made):
    """Removes the file that this run created at path. `made` is what os.fstat() returned for it
    while it was open. Another account that can change a directory above the file can make the path
    lead to another file, so this opens the file's directory and removes the name only if it is
    still that file, following no link. Errors are ignored: the caller reports the first failure."""
    try:
        directory = os.open(os.path.dirname(path) or ".", getattr(os, "O_PATH", 0) | os.O_DIRECTORY)
        try:
            name = os.path.basename(path)
            if os.path.samestat(os.stat(name, dir_fd=directory, follow_symlinks=False), made):
                os.unlink(name, dir_fd=directory)
        finally:
            os.close(directory)
    except OSError:
        pass


def convert(arguments, created):
    """Runs the steps in the order of the module docstring's table. A successful run prints:

      w2p ew ...                          see check_exponents()
      w2p layers <L> moe <m> klwords <n>  layers, mixture-of-experts layers, 32-bit words of KL.bin
      w2p moved <n>                       see new_lines()
      w2p policy <SHA-256 of the new parameter file>
      w2p kl <SHA-256 of the new scales file>
      w2p end 0                           printed by main()
    """
    check_call(arguments)
    base_path, words_path, sums_path, out_policy, out_scales = arguments
    check_outputs([base_path, words_path] + [sums_path] * (sums_path != "-"), [out_policy, out_scales])
    base_text = read_ascii(base_path, "input-base")
    words = parse_json(read_ascii(words_path, "input-words"))
    sums_text = None if sums_path == "-" else read_ascii(sums_path, "input-wsums")

    # The final newline, if any, is set aside, so that the output ends as the base file does.
    end = "\n" if base_text.endswith("\n") else ""
    lines = base_text[: len(base_text) - len(end)].split("\n")
    shape = read_shape(lines)
    check_structure(words, shape)
    check_words(words, shape)
    layers, model = words["KL"], words["W"]
    check_ranges(layers, model, shape)
    check_no_overflow(layers, model, shape)
    check_exponents(words, sums_text)
    lines, moved = new_lines(lines, model)

    # KL.bin is a sequence of unsigned 32-bit little-endian words: 13 for each layer, then 3 for
    # each mixture-of-experts layer.
    experts = [layer for l, layer in enumerate(layers) if has_experts(shape, l)]
    scales = [layer[name] for layer in layers for name in LAYER_NAMES]
    scales += [layer[name] for layer in experts for name in EXPERT_NAMES]
    policy_bytes, scales_bytes = ("\n".join(lines) + end).encode("ascii"), struct.pack("<%dI" % len(scales), *scales)
    create(out_policy, policy_bytes, created)
    create(out_scales, scales_bytes, created)
    print("w2p layers %d moe %d klwords %d" % (len(layers), len(experts), len(scales)))
    print("w2p moved %d" % moved)
    print("w2p policy " + hashlib.sha256(policy_bytes).hexdigest())
    print("w2p kl " + hashlib.sha256(scales_bytes).hexdigest())


def main(arguments):
    created = []
    try:
        convert(arguments, created)
        counts = {}
    except Rejected as rejected:
        counts = rejected.counts
    except Exception:
        counts = {"internal": 1}
    if counts:
        for path, made in created:
            remove_created(path, made)
    for name in sorted(counts):
        print("w2p refuse %s %d" % (name, counts[name]))
    status = STATUS[min(counts)] if counts else 0  # the classes of one step have one status
    print("w2p end %d" % status)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
