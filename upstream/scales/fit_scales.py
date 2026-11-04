#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Choose the integer (fixed-point) model's quantization scales from calibration results.

usage: fit_scales.py <shape.json> <maxima.json> <exponents.json> <bits> <out words.json> [--eres N] [--eg N]

The proofs are about the integer model: the network computed in integers. An integer n with the
exponent E represents the real value n / 2^E, and a lower exponent is coarser. The model divides a
matrix product by a power of two (a right shift, with rounding) to bring it to the exponent of its
destination: a cell (a signed 12-bit activation), the residual stream or a table's lookup key. This
program chooses these shifts, and the norms' constants, for one model: they are the scales (`words`
in identifiers and file names). scales/README.md, "Terms", defines these terms, and
scales/README.md, "The fit of the scales", states the rule that this program applies.

calibrate.py derives the inputs from a model's GGUF files, its sums file (wsums.txt: each weight
tensor's exponent and checksum) and a calibration set, and calls fit(). Run this program for inputs
from another source, or for --eg, which calibrate.py does not have. fit() describes the inputs,
parse() the options, and the comment at main() the output file. The exit codes are the constants
DONE, OUT_OF_RANGE and FAULT.

The scales of a real model depend on its weights, so the output file has mode 600, and the program
prints names, counts and differences in bits, but never the value of a scale. From the output file
words_to_policy.py writes the published files, policy.txt (the parameter file) and KL.bin (the
quantization scales). A difference in bits ("made B bits coarser") is not in them, so protect what
the program prints on a real model as you protect the weights.

The method is static calibration by the largest absolute value, with scales that are powers of two:
  - H. Wu et al.,
    "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation",
    arXiv:2004.09602, section 3.4.
  - With scale factors 2^-k "scaling with s corresponds to simple bit-shifting". M. Nagel et al.,
    "A White Paper on Neural Network Quantization", arXiv:2106.08295, section 2.2 (2.2.2).
  - A product of integers with the scale factors S1 and S2 is brought to an output's scale factor
    S3 by the multiplier S1 S2 / S3. B. Jacob et al., "Quantization and Training of Neural Networks
    for Efficient Integer-Arithmetic-Only Inference", arXiv:1712.05877, section 2.2, equation 5.
  - With powers of two "the scaling operation reduces to a rather simple bit-shift (with
    round-to-nearest)". S. R. Jain et al., "Trained Quantization Thresholds for Accurate and
    Efficient Fixed-Point Inference of Deep Neural Networks", arXiv:1903.08066, appendix A.2,
    equation 16.
So a shift is the sum of the exponents of a product's two factors minus the exponent of the
destination.
"""

import json
import math
import os
import sys
from typing import NamedTuple

USAGE = "usage: fit_scales.py <shape.json> <maxima.json> <exponents.json> <bits> <out words.json> [--eres N] [--eg N]"

# Exit codes. FAULT covers a usage error, an output path that exists or cannot be created, and
# an input that cannot be used. "Far out of range" means outside RANGES. DONE does not promise that
# words_to_policy.py accepts the file: it checks more.
DONE = 0  # the file is written
OUT_OF_RANGE = 1  # the file is written, but a scale in it is far out of range
FAULT = 2  # nothing is written

# Constants of the integer model (listed in scales/README.md, "The fixed exponents"). cs, S, rx and
# eg are lines of the parameter file (docs/model.md, "2 Shape parameters"). The tables are defined
# in docs/model.md, "4 Public tables".
CELL_BITS = 12  # a cell is a signed integer of this many bits
CELL_MAX = 2 ** (CELL_BITS - 1) - 1  # the largest absolute value in a cell
ROTARY_EXPONENT = 12  # cs: an entry of the rotary tables is a cosine or a sine times 2^cs
SOFTMAX_OUT_EXPONENT = 12  # S: the largest softmax weight, which stands for 1, is 2^S
SOFTMAX_KEY_EXPONENT = 8  # rx: the exponent of the softmax table's lookup key
SILU_KEY_EXPONENT = 8  # eg: the default exponent of the SiLU table's lookup key
SILU_OUT_EXPONENT = 6  # the exponent of the SiLU table's output
ROUTER_EXPONENT = 8  # the exponent of a router's logit, which is the sigmoid table's lookup key
# The integer norm multiplies each value of a token's stream vector by about 2^M divided by the
# vector's Euclidean length, and then shifts the product right by K. A layer has M1 and K1 for its
# first norm and M2 and K2 for its second. The final norm has MF and KF. The exact steps are in
# docs/model.md, "5 The norm".
NORM_K = 32  # the value of K1, K2 and KF
NORM_SPAN = 14  # M - K is at most this
# sh is the shift from an attention score to the softmax table's lookup key. The fitted value is
# clamped to this range (scales/README.md, "The shifts"). The verifier program cp_verify accepts
# only 4 .. 12, which words_to_policy.py checks.
SH_CLAMP = (0, 12)
# Inputs that would make a cell coarser by more than this many bits are rejected. A cell has only
# 12 bits, so the limit is generous.
COARSER_MOST = 31
MAXIMUM_LEAST = 1e-300  # a maximum below this is rejected

# fit() reports a scale outside these ranges as "far out of range". The range of a norm's M is the
# verifier's (docs/model.md, "9 Range conditions"). The range of a shift is wider than the
# verifier's, whose upper limit is 16, 24 or 32 and depends on a line of the parameter file. fit()
# does not read that file; words_to_policy.py applies the narrower range.
RANGES = {
    **{name: (24, 47) for name in ("M1", "M2", "MF")},
    **{name: (1, 63) for name in ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "kr", "kgR", "kuR")},
}


class InputFault(ValueError):
    """An input cannot be used. The message states why and where, and it never shows a value."""


class Note(NamedTuple):
    """One line for the user about the result of a fit."""

    text: str
    admissible: bool  # False if the line names a scale that is far out of range


def pick(table, keys, where):
    """Return table[keys[0]][keys[1]]... `where`, here and below, names the entry in a message."""
    for key in keys:
        if not isinstance(table, dict) or key not in table:
            raise InputFault(f"{where} is missing")
        table = table[key]
    return table


def whole(value, where):
    """Return `value`, which must be a JSON integer."""
    if type(value) is not int:  # isinstance would let true and false pass
        raise InputFault(f"{where} is not an integer")
    return value


def finite(value, where):
    """Return `value`, which must be a JSON number in the range of binary64."""
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        raise InputFault(f"{where} is not a finite number")
    return value


def plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def cell_exponent(mx):
    """Return the largest integer E with mx * 2^E <= CELL_MAX, for a positive binary64 number mx.

    The result is exact: it comes from math.frexp(), not from a rounded logarithm.
    """
    mantissa, exponent = math.frexp(mx)  # mx = mantissa * 2^exponent with 0.5 <= mantissa < 1
    if mantissa * 2 ** (CELL_BITS - 1) <= CELL_MAX:  # this product is exact
        return CELL_BITS - 1 - exponent
    return CELL_BITS - 2 - exponent


def fit(shape, maxima, exponents, e_res=11, eg=SILU_KEY_EXPONENT):
    """Compute the quantization scales by the rule of scales/README.md, "The fit of the scales".

    Return the pair (words, notes). `words` has the keys "KL" and "W" of the output file (see
    main()), and `notes` is a list of Note in printing order. Raise InputFault if an input cannot
    be used. e_res is the exponent of the residual stream, and eg that of the SiLU table's lookup
    key: the values of --eres and --eg (see parse()).

    `shape`, `maxima` and `exponents` are the parsed JSON objects of the three input files. A
    layer's key is its number as a string: "0", "1", ... Entries that are not listed below are
    ignored, except that all of `maxima` must have the form {site: {layer: value}}.

      shape      {"d": width of the residual stream, "dh": head dimension, "L": number of layers,
                  "eps": epsilon of the floating-point model's norms,
                  "theta": base of the rotary encoding, "moe": [the mixture-of-experts layers],
                  "rope": [the layers that apply the rotary encoding]}
                 All seven entries are required, although no scale depends on dh. Of d, fit() checks
                 only that it is a positive integer; the comment at `t` says at which d the shifts
                 are right.
      maxima     {site: {layer: largest absolute value that the calibration pass saw at this
                 site}}, each at least 1e-300. A site is a point of the forward pass; in the
                 comments, Es(site) is the exponent of its cell (defined in
                 scales/README.md, "The exponent of each cell"). The fit reads these sites, in
                 every layer unless noted:
                   n1, n2  the outputs of a layer's first and second norm, before the gains
                   q       the queries as they enter the scores, divided by sqrt(dh)
                   k       the keys as they enter the scores
                   v       the values
                   u       the up projection of a dense layer or of the shared expert
                   uR      the up projection of the routed expert; mixture-of-experts layers only
                   nF      the output of the final norm, before the gains; once, under the key "<L>"
                 A norm's gains are the vector by which the floating-point model multiplies the
                 normalized activation.
      exponents  {kind: {layer: e}, "output": e}, where e is the exponent that the conversion of
                 the weights to integers gave to the tensor blk.<layer>.<kind>.weight or to
                 output.weight. It is the third field of the tensor's line in the sums file
                 (wsums.txt). The conversion first multiplies a matrix that reads a norm's output
                 by that norm's gains and by sqrt(d) / 64, and the query matrix also by 1 / sqrt(dh)
                 (defined in docs/model.md, "11.3 Which tensor is which matrix"). That is why
                 the norms' sites are before the gains and why q is divided by sqrt(dh). A kind is
                 one of attn_q, attn_k, attn_v and attn_output in every layer; ffn_gate, ffn_up and
                 ffn_down in a dense layer; ffn_gate_inp, ffn_gate_shexp, ffn_up_shexp,
                 ffn_down_shexp, ffn_gate_exps, ffn_up_exps and ffn_down_exps in a
                 mixture-of-experts layer.

    scales/README.md, "The recorded sites and the scales", defines the names of the scales. One that
    begins with k is the right shift after a matrix product.
    """

    def of_shape(key):
        return pick(shape, [key], f"the entry {key} of the shape"), f"the entry {key} of the shape"

    def layer_set(key):
        listed, where = of_shape(key)
        if not isinstance(listed, list) or any(type(layer) is not int for layer in listed):
            raise InputFault(f"{where} is not a list of layer numbers")
        return set(listed)

    d, layers = whole(*of_shape("d")), whole(*of_shape("L"))
    if min(d, layers) < 1:
        raise InputFault("the entries d and L of the shape must be positive")
    whole(*of_shape("dh"))
    moe, rope = layer_set("moe"), layer_set("rope")
    # The layer keys of the maxima are read as integers, so "01" also means layer 1, and of two keys
    # for the same layer the later one is used. In the exponents, layer 1 is found only as "1".
    try:
        maxima = {site: {int(layer): value for layer, value in values.items()} for site, values in maxima.items()}
    except (AttributeError, ValueError):
        raise InputFault("the maxima do not have the form {site: {layer number: value}}") from None

    def fitted(site, layer, divisor=1.0):
        """Return the largest exponent at which the site's maximum / divisor fits in a cell."""
        where = f"the maximum of site {site}, layer {layer},"
        value = finite(pick(maxima, [site, layer], where), where)
        if value <= 0:
            raise InputFault(f"{where} is not positive")
        if value < MAXIMUM_LEAST:
            raise InputFault(f"{where} is below {MAXIMUM_LEAST}")
        return cell_exponent(value / divisor)

    # The exponent of the weight tensor blk.<layer>.<kind>.weight.
    def e(kind, layer):
        where = f"the exponent of {kind}, layer {layer},"
        return whole(pick(exponents, [kind, str(layer)], where), where)

    # Write sqrt(d) = c 2^t with an integer t and 1 <= c < 2. A norm's output is 2^(M - K) / sqrt(d) times the
    # normalized activation. With M - K = Es + t that factor is 2^Es / c, so the maximum divided by
    # c is the value that must fit in the cell. The square root and the quotient are binary64
    # operations, each rounded once. The conversion's constant sqrt(d) / 64 is c 2^(t - 6), so in
    # the product of a norm's output with a weight matrix c cancels and 2^(t - 6) is left. That
    # factor is 1 for d from 4,096 to 16,383 (the programs in weight_sums/ accept only 5,120). At
    # another d the shifts after these products, and E_logit, are off by t - 6 bits. fit() does not
    # check for this.
    t = (d.bit_length() - 1) // 2
    c = math.sqrt(d / 4**t)

    # M - K = Es + t must not exceed NORM_SPAN.
    def fitted_norm(site, layer):
        return min(fitted(site, layer, c), NORM_SPAN - t)

    cells = []
    for layer in range(layers):
        es = {"n1": fitted_norm("n1", layer), "n2": fitted_norm("n2", layer)}
        for site in ("q", "k", "v", "u", "uR") if layer in moe else ("q", "k", "v", "u"):
            es[site] = fitted(site, layer)
        cells.append(es)
    final = fitted_norm("nF", layers)

    # A score q . k has the exponent Es(q) + Es(k), and sh lowers it to SOFTMAX_KEY_EXPONENT. All
    # layers share one sh, the smallest that any layer needs; each Es(q) is then set to match it.
    # If that smallest value is negative, SH_CLAMP raises it to 0, and in each layer that needs a
    # negative sh q's maximum no longer fits in its cell. fit() reports neither this nor any other
    # sh below 4; words_to_policy.py rejects them.
    sh = min(es["q"] + es["k"] - SOFTMAX_KEY_EXPONENT for es in cells)
    sh = min(max(sh, SH_CLAMP[0]), SH_CLAMP[1])

    rows = []
    coarser = []
    for layer, es in enumerate(cells):
        es["q"] = SOFTMAX_KEY_EXPONENT + sh - es["k"]
        if layer in moe:
            gate, up, down = "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"
            # The shared and the routed down projection are added before one shift kd, and values
            # that are added need one scale (Jacob et al., cited at the top of this file, appendices
            # A.2 and A.3). `finer` is the number of bits by which the shared expert's product is
            # finer than the routed expert's, negative if it is coarser. The finer product is made
            # as coarse as the other by lowering Es of its up projection's cell.
            finer = es["u"] + e(down, layer) - es["uR"] - e("ffn_down_exps", layer)
            if abs(finer) > COARSER_MOST:
                raise InputFault(
                    f"the scales of the two down projections of layer {layer} are more than {COARSER_MOST} bits apart. "
                    "Check this layer's maxima of u and uR and its exponents of ffn_down_shexp and ffn_down_exps."
                )
            if finer > 0:
                es["u"] -= finer
            else:
                es["uR"] += finer
            if finer != 0:
                side, other = ("shared", "routed") if finer > 0 else ("routed", "shared")
                text = (
                    f"layer {layer}: the {side} expert's down projection is made {plural(abs(finer), 'bit')} coarser, "
                    f"to share one shift with the {other} expert's"
                )
                coarser.append(Note(text, True))
        else:
            gate, up, down = "ffn_gate", "ffn_up", "ffn_down"
        row = {
            # Before their shift, the rotary step multiplies q and k by table entries with the
            # factor 2^ROTARY_EXPONENT, or, with rp = 0, only by that factor.
            "kq": es["n1"] + e("attn_q", layer) - es["q"] + ROTARY_EXPONENT,
            "kk": es["n1"] + e("attn_k", layer) - es["k"] + ROTARY_EXPONENT,
            "kv": es["n1"] + e("attn_v", layer) - es["v"],
            "ko": es["v"] + e("attn_output", layer) - e_res,
            "kg": es["n2"] + e(gate, layer) - eg,
            "ku": es["n2"] + e(up, layer) - es["u"],
            "kd": SILU_OUT_EXPONENT + es["u"] + e(down, layer) - e_res,
            "M1": NORM_K + es["n1"] + t,
            "K1": NORM_K,
            "M2": NORM_K + es["n2"] + t,
            "K2": NORM_K,
            "sh": sh,
            "rp": 1 if layer in rope else 0,
        }
        if layer in moe:
            row["kr"] = es["n2"] + e("ffn_gate_inp", layer) - ROUTER_EXPONENT
            row["kgR"] = es["n2"] + e("ffn_gate_exps", layer) - eg
            row["kuR"] = es["n2"] + e("ffn_up_exps", layer) - es["uR"]
        rows.append(row)

    # The integer norm adds its epsilon to the sum, not the mean, of the d squares of stream values
    # with the exponent e_res, so the floating-point model's epsilon is multiplied by d 4^e_res. The
    # product is a binary64 number, and it is rounded to the nearest integer, ties to even.
    # words_to_policy.py requires the exactly rounded integer. With "eps": 1e-05 the two agree for
    # e_res from 3 to 19 at every d up to 2^20. At another e_res they can differ by 1, and
    # words_to_policy.py then rejects the file.
    try:
        eps = round(math.ldexp(d * finite(*of_shape("eps")), 2 * e_res))
    except OverflowError:
        raise InputFault("the entry eps of the shape is too large") from None
    theta = finite(*of_shape("theta"))
    shared = {
        "eps1": eps,
        "eps2": eps,
        "epsF": eps,
        "MF": NORM_K + final + t,
        "KF": NORM_K,
        "S": SOFTMAX_OUT_EXPONENT,
        "rx": SOFTMAX_KEY_EXPONENT,
        "cs": ROTARY_EXPONENT,
        "theta": int(theta) if theta == int(theta) else theta,  # a whole number is written without a point
        "e_res": e_res,
        # A logit of the model is an integer logit divided by 2^E_logit.
        "E_logit": final + whole(pick(exponents, ["output"], "the exponent of output"), "the exponent of output"),
    }

    notes = []
    for place, values in [(f"layer {layer}", row) for layer, row in enumerate(rows)] + [("the final norm", shared)]:
        for name, (least, most) in RANGES.items():
            if name in values and not least <= values[name] <= most:
                notes.append(Note(f"{place}: {name} is far out of range, outside {least} .. {most}", False))
    notes += coarser
    if eg != SILU_KEY_EXPONENT:
        notes.append(Note(f"use these scales only with a parameter file that has the line eg {eg}", True))
    return {"KL": rows, "W": shared}, notes


def report(notes):
    """Print the notes of a fit and return the exit code that they call for."""
    for note in notes:
        print(note.text)
    return DONE if all(note.admissible for note in notes) else OUT_OF_RANGE


def write_private_json(path, obj):
    """Write obj as JSON to a new file that only its owner can read.

    Raise FileExistsError if the path exists.
    """
    text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    made = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(text)
    except OSError:
        remove_created(path, made)
        raise


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


def read_json(path):
    """Return the JSON object in the file at path."""
    try:
        with open(path, encoding="utf-8") as file:
            obj = json.load(file)
    except OSError as fault:
        raise InputFault(f"{path} cannot be read: {fault.strerror}") from fault
    except (ValueError, RecursionError) as fault:  # a ValueError is also raised for bytes that are not UTF-8
        raise InputFault(f"{path} is not a JSON file") from fault
    if not isinstance(obj, dict):
        raise InputFault(f"{path} is not a JSON object")
    return obj


def parse(arguments):
    """Return the five positional arguments, with <bits> as an integer, and then e_res and eg.

      <bits>    the width of the embedding's integers in bits. It is only copied to the output.
      --eres N  gives the residual stream the exponent N, from 1 to 24 (default 11). The embedding
                must be converted with the same N, and outside 3 .. 19 words_to_policy.py can reject
                the file (see the comment on eps in fit()).
      --eg N    gives the SiLU table's lookup key the exponent N, from 4 to 8 (default 8). Below 8
                the parameter file needs the line eg N, and with that line no run of cp_verify ends
                in `YES` (its condition `no-range-words`).

    The options may appear anywhere among the other arguments. A usage error raises ValueError with
    one line for the user.
    """
    options = {}
    positional = []
    rest = iter(arguments)
    for argument in rest:
        if argument in ("--eres", "--eg"):
            if argument in options:
                raise ValueError(f"{argument} is given twice")
            options[argument] = next(rest, "")
        elif argument.startswith("--") or argument == "-h":
            raise ValueError(f"unknown option {argument}. {USAGE}")
        else:
            positional.append(argument)
    if len(positional) != 5:
        raise ValueError(f"expected 5 arguments besides the options, got {len(positional)}. {USAGE}")
    e_res = options.get("--eres", "11")
    if not (e_res.isascii() and e_res.isdigit() and len(e_res) < 100 and 1 <= int(e_res) <= 24):
        raise ValueError("--eres needs a number from 1 to 24, in digits only")
    eg = options.get("--eg", str(SILU_KEY_EXPONENT))
    if eg not in ("4", "5", "6", "7", "8"):
        raise ValueError("--eg needs one of 4, 5, 6, 7 and 8")
    try:
        positional[3] = int(positional[3])
    except ValueError:
        raise ValueError("<bits> is not an integer") from None
    return *positional, int(e_res), int(eg)


# The output file is a JSON object with four keys: "KL", a list with one object of scales per layer;
# "W", the values that all layers share; "ew", the weight tensors' exponents by tensor name, which
# this program leaves empty and calibrate.py fills in; and "emb_bits", which is <bits>. Its path
# must not exist before the run.
#
# calibrate.py writes a maxima.json; no program writes shape.json or exponents.json.
# words_to_policy.py checks the output file before it writes the published files. It accepts this
# program's file only with "-" in place of a sums file.
def main(arguments):
    def stop(why):
        print(f"fit_scales.py: {why}", file=sys.stderr)
        return FAULT

    try:
        shape_path, maxima_path, exponents_path, bits, out, e_res, eg = parse(arguments)
    except ValueError as reason:
        return stop(reason)
    if os.path.lexists(out):
        return stop(f"{out} exists. Give a path that does not exist yet.")
    try:
        shape, maxima, exponents = (read_json(path) for path in (shape_path, maxima_path, exponents_path))
        words, notes = fit(shape, maxima, exponents, e_res, eg)
        write_private_json(out, {**words, "ew": {}, "emb_bits": bits})
    except InputFault as fault:
        return stop(fault)
    except OSError as fault:
        return stop(f"{out} cannot be written: {fault.strerror}")
    status = report(notes)
    experts = sum("kr" in row for row in words["KL"])
    print(f"wrote {out}: {plural(len(words['KL']), 'layer')}, {experts} with experts")
    if status == OUT_OF_RANGE:
        print("words_to_policy.py will reject this file. Check the named layers' maxima and exponents, and --eres.")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
