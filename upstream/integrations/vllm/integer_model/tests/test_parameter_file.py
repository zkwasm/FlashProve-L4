# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of Parameters.read: the parameter file (policy.txt) and quantization scales (KL.bin).

Most tests change one line of prover/examples/target_small_f2/policy.txt. A file must mean the same
model here and in cp_commit and cp_prove, so the changed copies in PARAMETER_FILE_VARIANTS and the
files of shipped_parameter_files() also go to those programs' reader, "the provers' reader" below
(prover/common/parameter_file.h). prover/tests/parameter_file_judge.cpp makes a CPU program of it,
which prints "admitted" or the rejection and no values, so only the verdicts are compared. Each
entry of PARAMETER_FILE_VARIANTS states which of the two readers accept the copy, and so shows where
Parameters.read is stricter and where it is laxer. Line names and their order are those of
_PROOF_SYSTEM_LINES and _LINE_ORDER in integer_model.parameters.

In order: changed copies of one file, the lines on the proof system, the norms' exponents, every
shipped file, lines that only concern the proof, call lines, error messages, the tables' digests
(integer_model.tables), the scales' length, parameters built by a program, and the scales'
values.
"""

import dataclasses
import importlib
import pathlib
import re
import struct
import subprocess

import numpy
import pytest
import torch

from block_tools import PACKAGE, TREE
from example_models import PARAMETER_FILES, read_changed, with_line, with_lines
from integer_model import ParameterError
from integer_model.parameters import _ONLY_VALUE, Parameters
from integer_model.tables import make_tables, table_digest

EXAMPLE = PARAMETER_FILES["target_small_f2"]

# Which readers accept a changed copy: Parameters.read, then the provers' reader.
BOTH, NEITHER = (True, True), (False, False)
ONLY_HERE, ONLY_THE_PROVERS = (True, False), (False, True)
# The message for a line from `prof` on that does not have the form of a value line.
FORM = r'expected "name value", with one space and a decimal value below 2\^32'


def with_proof_system_line(name: str, replacement: str):
    """A function that replaces the whole line of the given name among the lines on the proof system
    (`version` to `table`). with_line() finds only a line with one decimal value; some of these have
    several values, or a word.
    """

    def change(text: str) -> str:
        text, count = re.subn(rf"^{name} .*\n", replacement, text, flags=re.MULTILINE)
        assert count == 1, f"The parameter file has {count} lines named {name}; one was expected."
        return text

    return change


def line_and_next(text: str, name: str) -> tuple[str, str]:
    """The line of the given name and the line after it, each with its line feed."""
    lines = text.splitlines(keepends=True)
    (at,) = [at for at, line in enumerate(lines) if line.startswith(name + " ")]
    return lines[at], lines[at + 1]


# Changed copies of target_small_f2's parameter file, each with the readers that accept it and, if
# Parameters.read rejects it, a pattern for the message. The file has "rcl 1" and "gcl 1", which
# clamp the router's logits and the gate's keys, and "xl 5": the provers write each value of the
# residual stream that a norm reads as five limbs (signed bytes), where the default is four.
PARAMETER_FILE_VARIANTS = {
    "unchanged": (lambda text: text, BOTH),
    "without_the_three_range_parameters": (
        lambda text: with_line(with_line(with_line(text, "rcl", ""), "gcl", ""), "xl", ""),
        BOTH,
    ),
    "other_silu_table": (lambda text: with_line(text, "rcl", "eg 7\nsmax 2047\nrcl 1\n"), BOTH),
    # A reader that skipped the lines it cannot parse would read these files without a clamp or the
    # fifth limb: another model.
    "blank_after_the_value": (lambda text: with_line(text, "rcl", "rcl 1 \n"), NEITHER, FORM),
    "blank_before_the_name": (lambda text: with_line(text, "rcl", " rcl 1\n"), NEITHER, FORM),
    "two_blanks": (lambda text: with_line(text, "xl", "xl  5\n"), NEITHER, FORM),
    "tab": (lambda text: with_line(text, "xl", "xl\t5\n"), NEITHER, FORM),
    "carriage_return": (lambda text: with_line(text, "gcl", "gcl 1\r\n"), NEITHER, FORM),
    "two_lines_in_one": (lambda text: with_line(text, "rcl", "rcl 1 "), NEITHER, FORM),
    "empty_line": (lambda text: with_line(text, "rcl", "\nrcl 1\n"), NEITHER, FORM),
    "no_line_feed_at_the_end": (
        lambda text: text.removesuffix("\n"),
        NEITHER,
        "the last line does not end with a newline",
    ),
    "plus_sign": (lambda text: with_line(text, "d", "d +512\n"), NEITHER, FORM),
    "leading_zero": (lambda text: with_line(text, "d", "d 0512\n"), NEITHER, FORM),
    "value_of_33_bits": (
        lambda text: with_line(text, "d", f"d {(1 << 32) + 512}\n"),
        NEITHER,
        FORM,
    ),
    "value_2_to_the_32": (
        lambda text: with_line(text, "seed_lo", f"seed_lo {1 << 32}\n"),
        NEITHER,
        FORM,
    ),
    "value_2_to_the_32_less_1": (
        lambda text: with_line(text, "seed_lo", f"seed_lo {(1 << 32) - 1}\n"),
        BOTH,
    ),
    "negative_value": (lambda text: with_line(text, "d", "d -512\n"), NEITHER, FORM),
    "hexadecimal_value": (lambda text: with_line(text, "d", "d 0x200\n"), NEITHER, FORM),
    "value_of_eleven_digits": (with_lines({"d": 10000000000}), NEITHER, FORM),
    "value_of_five_thousand_digits": (
        lambda text: with_line(text, "d", "d 1" + "0" * 4999 + "\n"),
        NEITHER,
        FORM,
    ),
    "rcl_0": (with_lines({"rcl": 0}), NEITHER, "rcl is 0, expected 1"),
    "rcl_2": (with_lines({"rcl": 2}), NEITHER, "rcl is 2, expected 1"),
    "gcl_0": (with_lines({"gcl": 0}), NEITHER, "gcl is 0, expected 1"),
    "xl_4": (with_lines({"xl": 4}), NEITHER, "xl is 4, expected 5"),
    "xl_6": (with_lines({"xl": 6}), NEITHER, "xl is 6, expected 5"),
    "eg_9": (
        lambda text: with_line(text, "rcl", "eg 9\nsmax 2047\nrcl 1\n"),
        NEITHER,
        "eg must be in 4 .. 7 or be omitted",
    ),
    "eg_3": (lambda text: with_line(text, "rcl", "eg 3\nrcl 1\n"), NEITHER, "eg must be in 4 .. 7"),
    "smax_4000": (
        lambda text: with_line(text, "rcl", "eg 7\nsmax 4000\nrcl 1\n"),
        NEITHER,
        "smax must be 2,047, 1,023, 511 or 255 or be omitted",
    ),
    # An optional line must not state its default.
    "eg_8": (
        lambda text: with_line(text, "rcl", "eg 8\nrcl 1\n"),
        NEITHER,
        "eg 8 and smax 4095 are the defaults and must be omitted",
    ),
    "smax_4095": (
        lambda text: with_line(text, "rcl", "smax 4095\nrcl 1\n"),
        NEITHER,
        "eg 8 and smax 4095 are the defaults and must be omitted",
    ),
    "unknown_line": (
        lambda text: with_line(text, "rcl", "abc 1\nrcl 1\n"),
        NEITHER,
        r"expected the line tsilu0 \(or .* lines nminH, eg, smax, rcl, gcl, xl\), found abc",
    ),
    "two_lines_swapped": (
        lambda text: text.replace("rcl 1\ngcl 1\n", "gcl 1\nrcl 1\n"),
        NEITHER,
        r"expected the line tsilu0 \(or .* lines xl\), found rcl",
    ),
    "line_missing": (
        lambda text: with_line(text, "cs", ""),
        NEITHER,
        "expected the line cs, found theta0_lo",
    ),
    "line_twice": (
        lambda text: with_line(text, "d", "d 512\nd 512\n"),
        NEITHER,
        "expected the line H, found d",
    ),
    "other_profile": (with_lines({"prof": 19}), NEITHER, "prof is 19, expected 20"),
    "other_reading": (
        with_proof_system_line("reading", "reading 2\n"),
        NEITHER,
        "reading is 2, expected 1",
    ),
    "other_rowcost": (
        with_proof_system_line("rowcost", "rowcost 2\n"),
        NEITHER,
        "rowcost is 2, expected 1",
    ),
    "other_answers": (
        with_proof_system_line("answers", "answers 2\n"),
        NEITHER,
        "answers is 2, expected 1",
    ),
    "other_listed": (
        with_proof_system_line("listed", "listed 1\n"),
        NEITHER,
        "listed is 1, expected 0",
    ),
    "no_routed_experts": (with_lines({"NE": 0}), NEITHER, "NE is 0, a model without routed"),
    # The format has none of these four lines. cp_verify rejects hb, oxc and magic as
    # `policy-call-line` (verifier/src/cp_verify.cpp).
    "temp_after_xl": (
        lambda text: with_line(text, "xl", "xl 5\ntemp 256\n"),
        NEITHER,
        "expected the line tsilu0, found temp",
    ),
    "hb_after_F": (
        lambda text: with_line(text, "F", "F 24\nhb 128\n"),
        NEITHER,
        r"expected the line tsilu0 \(or .*\), found hb",
    ),
    "oxc_after_the_digests": (
        lambda text: text.replace("call ", "oxc 11\ncall ", 1),
        NEITHER,
        "expected a call line, found oxc",
    ),
    "magic_before_version": (
        lambda text: "magic 1295397120\n" + text,
        NEITHER,
        'line 1: expected "version H821"',
    ),
    # nminH is optional, is allowed only directly after lidb, and must be above nmin and at most 28.
    # The file has nmin 20 and lidb 24.
    "nminH_after_lidb": (lambda text: with_line(text, "lidb", "lidb 24\nnminH 27\n"), BOTH),
    "nminH_before_lidb": (
        lambda text: with_line(text, "lidb", "nminH 27\nlidb 24\n"),
        NEITHER,
        "found nminH. The line nminH is allowed once, directly after a line lidb",
    ),
    "nminH_in_place_of_lidb": (
        lambda text: with_line(text, "lidb", "nminH 27\n"),
        NEITHER,
        "found nminH. The line nminH is allowed once, directly after a line lidb",
    ),
    "nminH_after_rcl": (
        lambda text: with_line(text, "rcl", "rcl 1\nnminH 27\n"),
        NEITHER,
        "found nminH. The line nminH is allowed once, directly after a line lidb",
    ),
    "nminH_after_lidb_and_eg": (
        lambda text: with_line(text, "lidb", "lidb 24\neg 7\nnminH 27\nsmax 2047\n"),
        NEITHER,
        "found nminH. The line nminH is allowed once, directly after a line lidb",
    ),
    "nminH_twice": (
        lambda text: with_line(text, "lidb", "lidb 24\nnminH 27\nnminH 27\n"),
        NEITHER,
        "found nminH. The line nminH is allowed once, directly after a line lidb",
    ),
    "nminH_with_the_value_of_nmin": (
        lambda text: with_line(text, "lidb", "lidb 24\nnminH 20\n"),
        NEITHER,
        "nminH is 20, expected 21 .. 28. For the value 20, omit the line",
    ),
    "nminH_below_nmin": (
        lambda text: with_line(text, "lidb", "lidb 24\nnminH 19\n"),
        NEITHER,
        "nminH is 19, expected 21 .. 28",
    ),
    "nminH_just_above_nmin": (lambda text: with_line(text, "lidb", "lidb 24\nnminH 21\n"), BOTH),
    "nminH_28": (lambda text: with_line(text, "lidb", "lidb 24\nnminH 28\n"), BOTH),
    "nminH_29": (
        lambda text: with_line(text, "lidb", "lidb 24\nnminH 29\n"),
        NEITHER,
        "nminH is 29, expected 21 .. 28",
    ),
    # Parameters.read does not check nmin, so nmin 0 is accepted, but nminH must still be 2 or more:
    # its smallest value is the larger of nmin + 1 and 2.
    "nmin_0": (with_lines({"nmin": 0}), ONLY_HERE),
    "nminH_1_at_nmin_0": (
        lambda text: with_line(with_line(text, "nmin", "nmin 0\n"), "lidb", "lidb 24\nnminH 1\n"),
        NEITHER,
        "nminH is 1, expected 2 .. 28",
    ),
    # lidb is optional and must not state its default, 22.
    "without_lidb": (lambda text: with_line(text, "lidb", ""), BOTH),
    "lidb_22": (
        with_lines({"lidb": 22}),
        NEITHER,
        "lidb is 22, expected 23 .. 30. For the value 22, omit the line",
    ),
    "lidb_23": (with_lines({"lidb": 23}), BOTH),
    "lidb_30": (with_lines({"lidb": 30}), BOTH),
    "lidb_31": (with_lines({"lidb": 31}), NEITHER, "lidb is 31, expected 23 .. 30"),
    "a_digest_of_nine_words": (
        lambda text: text.replace("tsig0 ", "tsilu8 1\ntsig0 "),
        NEITHER,
        "expected the line tsig0, found tsilu8. Each table's digest has 8 lines",
    ),
    "a_digest_of_seven_words": (
        lambda text: with_line(text, "texp7", ""),
        NEITHER,
        "expected the line texp7",
    ),
    # Parameters.read requires the digests. The provers' reader accepts all 32 lines or none.
    "without_the_digests": (
        lambda text: re.sub(r"^t(silu|sig|rope|exp)\d \d+\n", "", text, flags=re.MULTILINE),
        ONLY_THE_PROVERS,
        "expected the line tsilu0. The 32 lines tsilu0 to texp7 carry the digests",
    ),
}


# Shapes and table parameters. The file has lt 6, d 512, H 4, G 2, dh 128, f 1024, fe 512, NE 4,
# V 512, cs 12, theta0 10000, S 12 and rx 8. MF and KF are the final norm's exponents M and K. Most
# rejected copies break a single rule of Parameters.__post_init__, so that a test fails if that rule
# is missing. The provers' reader requires la > lt, so a longer window (a larger lt) comes with
# another la, which the model does not read.
PARAMETER_FILE_VARIANTS |= {
    "other_shape": (
        with_lines({"lt": 10, "la": 11, "H": 8, "G": 8, "dh": 64, "rx": 12}),
        BOTH,
    ),
    # The provers' reader compares NE and V with the last number of the line `table`.
    "other_shape_with_the_fewest_experts_and_the_largest_vocabulary": (
        with_lines({"lt": 10, "la": 11, "H": 8, "G": 8, "dh": 64, "NE": 2, "V": 1 << 24, "rx": 12}),
        ONLY_HERE,
    ),
    "window_too_long": (
        with_lines({"lt": 11, "la": 12}),
        ONLY_THE_PROVERS,
        "lt must be in 4 .. 10",
    ),
    "window_too_short": (with_lines({"lt": 3}), NEITHER, "lt must be in 4 .. 10"),
    # Breaks two rules. It is here because a reader that computed T = 2^lt before checking lt would
    # not return.
    "window_of_2_to_the_4294967295_tokens": (
        with_lines({"lt": (1 << 32) - 1}),
        NEITHER,
        "lt must be in 4 .. 10",
    ),
    "head_dimension_too_large": (
        with_lines({"dh": 256, "H": 2}),
        NEITHER,
        "dh must be a power of two from 2 to 128",
    ),
    "head_dimension_no_power_of_two": (
        with_lines({"dh": 96, "H": 4, "d": 384}),
        NEITHER,
        "dh must be a power of two from 2 to 128",
    ),
    "head_dimension_1": (with_lines({"dh": 1}), NEITHER, "dh must be a power of two from 2 to 128"),
    "key_value_heads_no_power_of_two": (
        with_lines({"H": 3, "G": 3, "d": 384}),
        NEITHER,
        "G must be a power of two of at most 64",
    ),
    "key_value_heads_do_not_divide_query_heads": (
        with_lines({"G": 8}),
        NEITHER,
        "G must divide H",
    ),
    "too_many_heads": (
        with_lines({"H": 64, "G": 1, "dh": 8}),
        NEITHER,
        r"H \+ G must be at most 64",
    ),
    "width_other_than_heads_times_head_dimension": (with_lines({"d": 500}), ONLY_HERE),
    "width_too_large": (
        with_lines({"d": (1 << 20) + 1}),
        NEITHER,
        r"d and f must be in 1 .. 2\^20",
    ),
    "expert_width_no_power_of_two": (
        with_lines({"f": 768, "fe": 384}),
        NEITHER,
        "fe must be a power of two, and f = 2 fe",
    ),
    "hidden_width_is_not_twice_the_expert_width": (
        with_lines({"fe": 256}),
        NEITHER,
        "fe must be a power of two, and f = 2 fe",
    ),
    # cp_commit and cp_prove need f below 2^16: their forward pass adds 2 f products of signed bytes
    # into one 32-bit sum (the rule prover-only-f in prover/common/parameter_file.h).
    # Parameters.read takes f up to 2^20. IntegerModel rejects this file all the same: it needs
    # smax * 2^27 * f below 2^53, so with smax = 4,095, as here, and f a power of two (f = 2 fe), f
    # can be at most 16,384 (test_model_folder.py has f = 32,768).
    "hidden_width_of_2_to_the_16": (with_lines({"f": 1 << 16, "fe": 1 << 15}), ONLY_HERE),
    "experts_no_power_of_two": (
        with_lines({"NE": 3}),
        NEITHER,
        "NE must be a power of two from 2 to 4,096",
    ),
    "one_expert": (with_lines({"NE": 1}), NEITHER, "NE must be a power of two from 2 to 4,096"),
    "too_many_experts": (
        with_lines({"NE": 8192}),
        NEITHER,
        "NE must be a power of two from 2 to 4,096",
    ),
    "vocabulary_of_one_token": (with_lines({"V": 1}), NEITHER, r"V must be in 2 .. 2\^24"),
    "vocabulary_too_large": (
        with_lines({"V": (1 << 24) + 1}),
        NEITHER,
        r"V must be in 2 .. 2\^24",
    ),
    "smaller_rotary_scale": (with_lines({"cs": 11}), NEITHER, "cs must be 12"),
    "larger_rotary_scale": (with_lines({"cs": 13}), NEITHER, "cs must be 12"),
    "other_rotary_base": (
        with_lines({"theta0_lo": 12345}),
        ONLY_THE_PROVERS,
        "theta0 must be 10,000 or 500,000",
    ),
    "softmax_scale_0": (with_lines({"S": 0}), NEITHER, "S must be in 1 .. 14"),
    "softmax_scale_too_large": (with_lines({"S": 15}), NEITHER, "S must be in 1 .. 14"),
    "softmax_sum_too_large": (
        with_lines({"lt": 10, "la": 11, "S": 14}),
        NEITHER,
        r"lt \+ S \+ 1 must be at most 24",
    ),
    "softmax_step_0": (with_lines({"rx": 0}), NEITHER, "rx must be in 1 .. 12"),
    "softmax_step_too_large": (with_lines({"rx": 13}), NEITHER, "rx must be in 1 .. 12"),
    "no_layer": (with_lines({"L": 0}), NEITHER, "KL.bin must hold 13 words for each of the 0"),
    "regularizer_0": (
        with_lines({"eps1_lo": 0}),
        NEITHER,
        r"the first norm of layer 0 needs 1 <= eps < 2\^48",
    ),
    "regularizer_of_49_bits": (
        with_lines({"eps2_hi": 1 << 16}),
        NEITHER,
        r"the second norm of layer 0 needs 1 <= eps < 2\^48",
    ),
    "final_norm_M_too_large": (
        with_lines({"MF": 48, "KF": 32}),
        NEITHER,
        "the final norm needs 24 <= M <= 47",
    ),
    "final_norm_M_too_small": (
        with_lines({"MF": 23, "KF": 17}),
        NEITHER,
        "the final norm needs 24 <= M <= 47",
    ),
    "final_norm_K_too_large": (
        with_lines({"MF": 47, "KF": 33}),
        NEITHER,
        "the final norm needs .* 17 <= K <= 32",
    ),
    "final_norm_K_too_small": (
        with_lines({"MF": 30, "KF": 16}),
        NEITHER,
        "the final norm needs .* 17 <= K <= 32",
    ),
    "final_norm_M_exceeds_K_by_15": (
        with_lines({"MF": 45, "KF": 30}),
        NEITHER,
        "the final norm needs .* K >= M - 14",
    ),
    # With five limbs and K <= 23, both readers require K >= M - 13
    # (test_the_norms_bound_on_K_depends_on_the_limbs).
    "final_norm_M_exceeds_K_by_14_at_K_23": (
        with_lines({"MF": 37, "KF": 23}),
        NEITHER,
        "M - 13 with xl 5",
    ),
    "final_norm_M_exceeds_K_by_13_at_K_23": (with_lines({"MF": 36, "KF": 23}), BOTH),
    "final_norm_M_exceeds_K_by_14_at_K_24": (with_lines({"MF": 38, "KF": 24}), BOTH),
}


# The lines on the proof system, from `version` to `table`. Parameters.read checks their names and
# order, the number of their values and the values' spelling. The provers' reader also checks
# values.
PARAMETER_FILE_VARIANTS |= {
    "proof_system_two_blanks": (
        with_proof_system_line("sec", "sec  128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_blank_after_the_value": (
        with_proof_system_line("sec", "sec 128 \n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_blank_before_the_name": (
        with_proof_system_line("sec", " sec 128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_tab": (
        with_proof_system_line("sec", "sec\t128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_carriage_return": (
        with_proof_system_line("sec", "sec 128\r\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_empty_line": (
        with_proof_system_line("sec", "\nsec 128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_leading_zero": (
        with_proof_system_line("sec", "sec 0128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_plus_sign": (
        with_proof_system_line("sec", "sec +128\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_hexadecimal_value": (
        with_proof_system_line("sec", "sec 0x80\n"),
        NEITHER,
        'line 2: expected "sec" and 1 decimal value',
    ),
    "proof_system_two_blanks_in_the_name": (
        with_proof_system_line("work field", "work  field 0 0 0\n"),
        NEITHER,
        'line 8: expected "work field" and 3 decimal values',
    ),
    "proof_system_two_blanks_between_values": (
        with_proof_system_line("work field", "work field 0  0 0\n"),
        NEITHER,
        'line 8: expected "work field" and 3 decimal values',
    ),
    "proof_system_value_of_15_digits": (
        with_proof_system_line("slots", "slots 100000000000000\n"),
        ONLY_HERE,
    ),
    "proof_system_value_of_16_digits": (
        with_proof_system_line("slots", "slots 1000000000000000\n"),
        NEITHER,
        'line 5: expected "slots" and 1 decimal value',
    ),
    "empty_line_before_version": (
        lambda text: "\n" + text,
        NEITHER,
        'line 1: expected "version H821"',
    ),
    "without_proof_of_work": (
        with_proof_system_line("work positions", "work positions 0 0 0\n"),
        BOTH,
    ),
    "target_of_0_bits": (with_proof_system_line("sec", "sec 0\n"), ONLY_HERE),
    "slots_no_power_of_two": (with_proof_system_line("slots", "slots 63\n"), ONLY_HERE),
}

# Every version but H821 is rejected: one each with another number for the field (H811) and the hash
# (H822), three with another format number or letter, and other strings.
OTHER_VERSIONS = ["H811", "H822", "H721", "HG06", "H921", "H82", "H8211", "h821", "825252419", ""]
PARAMETER_FILE_VARIANTS |= {
    f"version_{version}": (
        with_proof_system_line("version", f"version {version}\n"),
        NEITHER,
        'line 1: expected "version H821". This package reads no other format',
    )
    for version in OTHER_VERSIONS
}


@pytest.fixture(scope="module")
def provers_reader(compiler, tmp_path_factory):
    """A function that gives a parameter file to the provers' reader and returns the line that is
    printed: "admitted", or the rejection.

    It runs prover/tests/parameter_file_judge.cpp, compiled once with the options that the source
    file names, for the field and the hash of version H821.
    """
    program = tmp_path_factory.mktemp("parameter_file_judge") / "parameter_file_judge"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-fno-ipa-modref", "-DBLK_FIELD=2", "-DBLK_HASH=1"]
        + ["-I", TREE / "prover", "-I", TREE / "blocks"]
        + ["-o", program, TREE / "prover/tests/parameter_file_judge.cpp"],
        check=True,
    )

    def verdict(parameter_file) -> str:
        done = subprocess.run(
            [program], input=f"{parameter_file}\n", capture_output=True, text=True, check=True
        )
        return done.stdout.removesuffix("\n")

    return verdict


@pytest.mark.parametrize("variant", PARAMETER_FILE_VARIANTS)
def test_changed_parameter_files_are_accepted_or_rejected(variant, tmp_path):
    """Parameters.read accepts the copy, or rejects it with the message of its entry."""
    change, (accepted, _), *message = PARAMETER_FILE_VARIANTS[variant]
    changed = change(EXAMPLE.read_text())
    if accepted:
        read_changed(changed, tmp_path)
        return
    with pytest.raises(ParameterError, match=message[0]):
        read_changed(changed, tmp_path)


@pytest.mark.parametrize("variant", PARAMETER_FILE_VARIANTS)
def test_the_provers_reader_on_the_changed_parameter_files(variant, provers_reader, tmp_path):
    """The provers' reader accepts or rejects the copy as its entry says."""
    change, (_, admitted), *_ = PARAMETER_FILE_VARIANTS[variant]
    (tmp_path / "policy.txt").write_text(change(EXAMPLE.read_text()))
    verdict = provers_reader(tmp_path / "policy.txt")
    assert (verdict == "admitted") == admitted, verdict


# The names of the lines on the proof system, in the file's order.
PROOF_SYSTEM_LINES = [
    *"version sec reading rowcost slots answers record".split(),
    *(f"work {kind}" for kind in "field sample closeness positions smallness".split()),
    *"salt listed".split(),
    *(
        f"{line} {group}"
        for group in "link folded omega own once-folded once-omega once-own once-ranges".split()
        for line in ("group", "stages")
    ),
    *"ranges-part table".split(),
]


@pytest.mark.parametrize("name", PROOF_SYSTEM_LINES)
def test_each_line_on_the_proof_system_is_required_in_its_place(name, tmp_path):
    """The line cannot be removed, repeated, exchanged with the next line or renamed. The message
    gives the number of the first line that is out of place and the name expected there.
    """
    lines = EXAMPLE.read_text().splitlines(keepends=True)
    at = PROOF_SYSTEM_LINES.index(name)
    before, line, following, after = lines[:at], lines[at], lines[at + 1], lines[at + 2 :]
    assert line.startswith(name + " ")
    # What a message expects at each line. After `table` comes `prof`, which is a value line.
    due = ["version H821", *PROOF_SYSTEM_LINES[1:], "name value"]
    for changed, number in (
        ([following], at + 1),
        ([line, line, following], at + 2),
        ([following, line], at + 1),
        ([line.replace(name, name + "x", 1), following], at + 1),
    ):
        with pytest.raises(ParameterError, match=f'line {number}: expected "{due[number - 1]}"'):
            read_changed("".join(before + changed + after), tmp_path)


@pytest.mark.parametrize("name", PROOF_SYSTEM_LINES[1:])
def test_the_values_of_a_line_on_the_proof_system(name, tmp_path):
    """Other values change no parameter, but for the lines that take one value (_ONLY_VALUE). One
    value more or fewer is rejected. A line `stages` has 4, 8, .. or 32 values.
    """
    text = EXAMPLE.read_text()
    count = len(line_and_next(text, name)[0].split(" ")) - len(name.split(" "))
    taken, rejected = ([] if name in _ONLY_VALUE else [count]), [count - 1, count + 1]
    if name.startswith("stages "):
        taken, rejected = [4, 32], [0, 3, 5, 31, 33, 36]

    def with_values(number: int) -> str:
        return with_proof_system_line(name, " ".join([name] + ["7"] * number) + "\n")(text)

    for number in taken:
        assert read_changed(with_values(number), tmp_path) == Parameters.read(EXAMPLE), number
    for number in rejected:
        with pytest.raises(
            ParameterError, match=f'expected "{name}" and [0-9, .or]+ decimal value'
        ):
            read_changed(with_values(number), tmp_path)


def test_other_values_in_all_lines_on_the_proof_system_change_no_parameter(tmp_path):
    """As the test above, with the lines after `version` that take more than one value changed in
    one copy: a file for the same model under another setting of the proof system differs in many
    of them.
    """
    text = EXAMPLE.read_text()
    for name in PROOF_SYSTEM_LINES[1:]:
        if name in _ONLY_VALUE:
            continue
        count = len(line_and_next(text, name)[0].split(" ")) - len(name.split(" "))
        text = with_proof_system_line(name, " ".join([name] + ["7"] * count) + "\n")(text)
    assert read_changed(text, tmp_path) == Parameters.read(EXAMPLE)


@pytest.mark.parametrize(
    "model_name, MF, KF, accepted",
    [
        ("target_small_f2", 31, 17, False),
        ("target_small_f2", 34, 20, False),
        ("target_small_f2", 37, 23, False),
        ("target_small_f2", 34, 21, True),
        ("target_small_f2", 38, 24, True),
        ("tiny_without_clamps", 31, 17, True),
    ],
)
def test_the_norms_bound_on_K_depends_on_the_limbs(model_name, MF, KF, accepted, tmp_path):
    """With five limbs (target_small_f2) and K <= 23, a norm with K = M - 14 is rejected; with four
    limbs (tiny_without_clamps) it is accepted.

    At such a pair the norm's quotient can leave the range that the provers accept for range
    condition 2 while every other condition holds (docs/model.md, "9 Range conditions").
    """
    example = PARAMETER_FILES[model_name]
    parameter_file, scales_file = tmp_path / "policy.txt", tmp_path / "KL.bin"
    parameter_file.write_text(with_lines({"MF": MF, "KF": KF})(example.read_text()))
    scales_file.write_bytes((example.parent / "KL.bin").read_bytes())
    if not accepted:
        with pytest.raises(ParameterError, match="M - 13 with xl 5"):
            Parameters.read(parameter_file)
        return
    p = Parameters.read(parameter_file)
    assert (p.MF, p.KF) == (MF, KF)


def shipped_parameter_files() -> list[pathlib.Path]:
    """The parameter files of the example configurations for version H821
    (prover/examples/target_*_f2*) and of integrations/vllm/integer_model/tests/configurations/,
    found by pattern, so that every such file in the tree is tested.
    """
    found = sorted((TREE / "prover/examples").glob("target_*_f2*/policy*.txt"))
    assert len(found) >= 5, "The repository's example configurations were not found."
    return found + sorted((PACKAGE / "tests/configurations").glob("*/policy.txt"))


def test_every_shipped_parameter_file_is_accepted():
    """Each also states the digests of the tables that make_tables() computes."""
    faults = []
    for parameter_file in shipped_parameter_files():
        try:
            # Raises an exception if a table's digest is not the stated one.
            make_tables(Parameters.read(parameter_file), torch.device("cpu"))
        except Exception as error:  # whatever it is, the message must name the file
            faults.append(f"{parameter_file}: {type(error).__name__}: {error}")
    assert not faults, "\n".join(faults)


def test_the_provers_reader_on_every_shipped_parameter_file(provers_reader):
    """It accepts the example configurations and rejects those of these tests, which are too small
    for it (integrations/vllm/integer_model/tests/configurations/README.md).
    """
    for parameter_file in shipped_parameter_files():
        verdict = provers_reader(parameter_file)
        assert (verdict == "admitted") == (PACKAGE not in parameter_file.parents), (
            f"{parameter_file}: {verdict}"
        )


def test_the_optional_line_nminH(tmp_path):
    changed = with_line(EXAMPLE.read_text(), "lidb", "lidb 24\nnminH 27\n")
    assert read_changed(changed, tmp_path) == Parameters.read(EXAMPLE)


# Lines from `prof` on that only concern the proof and may have any value here.
PROOF_LINES = (
    "ll hq hv ho hgD huD hdD hr la nmin D lmaf NAf lmad NAd lmar lmb lmq lmx iota_0 iota_7 lB kFM "
    "kFD lmS NAs lmh lmo lmbA iotaA_0 lmhf NHf lmhd lmhr lmbH iotaH_0 iotaH_5 C theta cap sh sT"
).split()


@pytest.mark.parametrize("name", PROOF_LINES)
def test_each_proof_line_is_required_in_its_place(name, tmp_path):
    """Another value changes no parameter. The line cannot be removed, repeated, exchanged with the
    next line or renamed.
    """
    text = EXAMPLE.read_text()
    line, following = line_and_next(text, name)
    assert line != f"{name} 9\n"
    assert read_changed(with_lines({name: 9})(text), tmp_path) == Parameters.read(EXAMPLE)
    next_name = following.split(" ")[0]
    for changed, due, found in (
        (with_line(text, name, ""), name, next_name),
        (with_line(text, name, line + line), next_name, name),
        (text.replace(line + following, following + line), name, next_name),
        (with_line(text, name, line.replace(name, name + "x")), name, name + "x"),
    ):
        assert changed != text
        with pytest.raises(ParameterError, match=rf"expected the line {due}, found {found}\."):
            read_changed(changed, tmp_path)


# Lines that the model does not read and that allow one value only: that value, then values that are
# rejected. lidb and nminH, which allow a range, are in PARAMETER_FILE_VARIANTS.
ONLY_VALUES = {
    "ut": (1, (0, 2)),
    "tau": (128, (127, 129, 160)),
    "wb_qk": (13, (12, 14)),
    "wb_r": (13, (0, 12, 14)),
    "toksrc": (1, (0, 2)),
    "wb_emb": (21, (13, 20, 22)),
    "wb_head": (13, (12, 14, 21)),
    "F": (24, (23, 25)),
}


@pytest.mark.parametrize("name", ONLY_VALUES)
def test_lines_that_allow_one_value_only(name, tmp_path):
    text = EXAMPLE.read_text()
    only, rejected = ONLY_VALUES[name]
    assert f"\n{name} {only}\n" in text
    for value in rejected:
        with pytest.raises(ParameterError, match=rf": {name} is {value}, expected {only}\."):
            read_changed(with_lines({name: value})(text), tmp_path)


def test_the_numbers_of_call_lines(tmp_path):
    """The file has the most call and callH lines allowed, 8 and 6. callA lines have no limit."""
    text = EXAMPLE.read_text()
    kinds = [line.split(" ")[0] for line in text.splitlines() if line.startswith("call")]
    assert (kinds.count("call"), kinds.count("callH")) == (8, 6)
    read_changed(text.replace("callA ", "callA X R16 Y\ncallA ", 1), tmp_path)
    with pytest.raises(ParameterError, match="has 9 call lines. It may have at most 8"):
        read_changed(text.replace("call ", "call X R16 Y\ncall ", 1), tmp_path)
    with pytest.raises(ParameterError, match="has 7 callH lines. It may have at most 6"):
        read_changed(text + "callH X R16 Y\n", tmp_path)


def test_the_length_of_a_call_line(tmp_path):
    """A call line has at most 1,022 characters, the line feed not counted."""
    text = EXAMPLE.read_text()
    last = text.splitlines()[-1]
    for length, taken in ((1022, True), (1023, False)):
        changed = text[:-1] + " " + "x" * (length - len(last) - 1) + "\n"
        assert len(changed.splitlines()[-1]) == length
        if taken:
            read_changed(changed, tmp_path)
        else:
            with pytest.raises(ParameterError, match="in at most 1,022 characters"):
                read_changed(changed, tmp_path)


def test_the_order_and_the_form_of_the_call_lines(tmp_path):
    text = EXAMPLE.read_text()
    order = "must end with at least one call line, then at least one callA line, then any callH"
    for changed in (
        text + "call X R16 Y\n",
        text + "callA X R16 Y\n",
        re.sub(r"^callA .*\n", "", text, flags=re.MULTILINE),
        re.sub(r"^call .*\n", "", text, flags=re.MULTILINE),
    ):
        with pytest.raises(ParameterError, match=order):
            read_changed(changed, tmp_path)
    for line in ("call X R16", "callB X R16 Y", "call X  R16 Y", "call X R16 Y ", "call X R16 Y+"):
        with pytest.raises(ParameterError, match="expected a call line: call, callA or callH"):
            read_changed(text + line + "\n", tmp_path)


def test_messages_name_the_file_and_the_line(tmp_path):
    text = EXAMPLE.read_text()
    place = re.escape(str(tmp_path / "policy.txt"))
    # A file whose version is not H821 may differ in other lines too. The message must name the
    # version, not the first such line.
    other = with_proof_system_line("version", "version H811\n")(
        with_proof_system_line("sec", "")(text)
    )
    with pytest.raises(ParameterError, match=f'^{place}, line 1: expected "version H821"'):
        read_changed(other, tmp_path)
    with pytest.raises(
        ParameterError,
        match=f"^{place}, line 33: prof is 19, expected 20, the profile with attention",
    ):
        read_changed(with_lines({"prof": 19})(text), tmp_path)
    with pytest.raises(ParameterError, match=f"^{place}, line 37: {FORM}"):
        read_changed(with_line(text, "d", "d 0512\n"), tmp_path)
    with pytest.raises(ParameterError, match=f"^{place}, line 38: expected the line H, found d"):
        read_changed(with_line(text, "d", "d 512\nd 512\n"), tmp_path)
    with pytest.raises(ParameterError, match=f"^{place}, line 184: expected a call line: call,"):
        read_changed(text + "\n", tmp_path)
    with pytest.raises(ParameterError, match=rf"cs must be 12\. \(Read from {place} and "):
        read_changed(with_lines({"cs": 11})(text), tmp_path)
    (tmp_path / "policy.txt").write_bytes(text.encode("ascii") + b"\xc3\xa9\n")
    with pytest.raises(ParameterError, match=f"^{place} is not an ASCII text file"):
        Parameters.read(tmp_path / "policy.txt")


@pytest.mark.parametrize("table", ["tsilu", "tsig", "trope", "texp"])
def test_a_table_digest_has_eight_lines(table, tmp_path):
    text = EXAMPLE.read_text()
    assert [len(words) for words in Parameters.read(EXAMPLE).table_digests.values()] == [8, 8, 8, 8]
    assert len(table_digest([1, 2, 3])) == 8
    last, _ = line_and_next(text, f"{table}7")
    for changed in (
        text.replace(last, ""),
        text.replace(last, last + f"{table}8 1\n"),
        text.replace(last, last + f"{table}8 1\n{table}9 1\n"),
        re.sub(rf"^{table}\d \d+\n", "", text, flags=re.MULTILINE),
    ):
        assert changed != text
        with pytest.raises(ParameterError, match="expected (the line |a call line, found texp8)"):
            read_changed(changed, tmp_path)


def test_a_table_digest_equals_that_of_policy_tables(monkeypatch):
    """scales/policy_tables.py has the same Python source of the hash, so this comparison can show
    only a copying error. The lists end before, at and after the end of the hash's blocks of 32
    words. test_every_shipped_parameter_file_is_accepted compares with other code: whole tables with
    the digests in the example configurations, which cp_verify compares with tables that it computes
    and hashes itself.
    """
    monkeypatch.syspath_prepend(TREE / "scales")
    policy_tables = importlib.import_module("policy_tables")
    seen = set()
    for length in (0, 1, 2, 31, 32, 33, 63, 64, 65, 1000):
        for words in ([0] * length, [0xFFFFFFFF] * length, [0x01020304 + i for i in range(length)]):
            digest = table_digest(words)
            assert digest == policy_tables.hash_words(72, words)  # 72 marks a public table
            seen.add(tuple(digest))
    assert len(seen) == 28  # all differ; the three lists of length 0 are one list
    for word in (-1, 1 << 32):
        with pytest.raises(ValueError):
            table_digest([1, word, 3])


@pytest.mark.parametrize("table", ["tsilu", "tsig", "trope", "texp"])
def test_every_word_of_a_table_digest_is_compared(table):
    p = Parameters.read(PARAMETER_FILES["tiny_without_clamps"])
    make_tables(p, torch.device("cpu"))
    for i in range(8):
        words = list(p.table_digests[table])
        words[i] ^= 1
        changed = dataclasses.replace(p, table_digests=p.table_digests | {table: words})
        with pytest.raises(ParameterError, match=rf"\(lines {table}0 to {table}7\)"):
            make_tables(changed, torch.device("cpu"))


@pytest.mark.parametrize("change_in_words", [-1, 1])
def test_quantization_scales_of_another_length_are_rejected(change_in_words, tmp_path):
    parameter_file, scales_file = tmp_path / "policy.txt", tmp_path / "KL.bin"
    parameter_file.write_bytes(EXAMPLE.read_bytes())
    words = (EXAMPLE.parent / "KL.bin").read_bytes()
    scales_file.write_bytes(words[:-4] if change_in_words < 0 else words + words[-4:])
    with pytest.raises(ParameterError, match="KL.bin must hold 13 words for each of the"):
        Parameters.read(parameter_file)


def test_rules_also_hold_for_parameters_that_a_program_constructs():
    """Some rules cannot be broken in isolation through the two files: the reader or another rule
    rejects such files first.
    """
    p = Parameters.read(EXAMPLE)
    # A model without experts, which only a program can make.
    dense = dict(NE=0, router_clamp=False, gate_clamp=False)
    dataclasses.replace(p, **dense)
    dataclasses.replace(p, **dense, f=1 << 20)
    for changes, reason in (
        (dict(L=0, layers=()), "L must be at least 1"),
        (dict(layers=p.layers[:-1]), "one set of quantization scales for each layer"),
        (dict(NX=3), "xl must be 5 or be omitted"),
        (dict(NX=6), "xl must be 5 or be omitted"),
        (dict(NE=0), "rcl and gcl need a model with experts"),
        (dict(NE=0, router_clamp=False), "rcl and gcl need a model with experts"),
        (dict(NE=0, gate_clamp=False), "rcl and gcl need a model with experts"),
        (dict(**dense, f=(1 << 20) + 1), r"d and f must be in 1 .. 2\^20"),
        (dict(**dense, f=0), r"d and f must be in 1 .. 2\^20"),
        (dict(d=0), r"d and f must be in 1 .. 2\^20"),
        (dict(n_h=0), "and H at least 1"),
    ):
        with pytest.raises(ParameterError, match=reason):
            dataclasses.replace(p, **changes)


# The names of a layer's 13 words in KL.bin, in the file's order, and of the 3 further words of a
# layer with experts.
SCALE_WORDS: tuple[str, ...] = tuple("kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp".split())
EXPERT_SCALE_WORDS: tuple[str, ...] = tuple("kr kgR kuR".split())
SCALES_TO_REJECT = [
    (word, value) for word in SCALE_WORDS[:7] + EXPERT_SCALE_WORDS for value in (0, 33)
]
SCALES_TO_REJECT += [
    ("sh", 3),
    ("sh", 13),
    ("rp", 2),
    ("M1", 48),
    ("M2", 45),
    ("K1", 33),
    ("K2", 16),
    ("M1", 23),
]


def read_with_scales(folder, layer: int = -1, **changes: int) -> Parameters:
    """Reads target_small_f2's files with other words in one layer. The last layer has experts, and
    kr, kgR and kuR are changed there whatever `layer` is.
    """
    (folder / "policy.txt").write_bytes(EXAMPLE.read_bytes())
    words = list(numpy.frombuffer((EXAMPLE.parent / "KL.bin").read_bytes(), dtype="<u4"))
    L = Parameters.read(EXAMPLE).L
    for word, value in changes.items():
        index = (
            13 * (layer % L) + SCALE_WORDS.index(word)
            if word in SCALE_WORDS
            else len(words) - 3 + EXPERT_SCALE_WORDS.index(word)
        )
        words[index] = value
    (folder / "KL.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    return Parameters.read(folder / "policy.txt")


@pytest.mark.parametrize("word, value", SCALES_TO_REJECT)
def test_quantization_scales_outside_their_intervals_are_rejected(word, value, tmp_path):
    """One word of KL.bin is changed so that one rule of Parameters.__post_init__ fails. The words
    are those of docs/model.md, "3 Weights and quantization scales".
    """
    reason = {
        "sh": "sh in layer",
        "rp": "rp in layer",
        "M1": "the first norm of layer",
        "K1": "the first norm of layer",
        "M2": "the second norm of layer",
        "K2": "the second norm of layer",
    }.get(word, "every product's shift in layer")
    with pytest.raises(ParameterError, match=reason):
        read_with_scales(tmp_path, **{word: value})


@pytest.mark.parametrize("layer", [0, -1])
@pytest.mark.parametrize("M, K", [("M1", "K1"), ("M2", "K2")])
def test_a_layers_norms_have_the_bound_of_the_final_norm(M, K, layer, tmp_path):
    """See test_the_norms_bound_on_K_depends_on_the_limbs. target_small_f2 has five limbs."""
    with pytest.raises(ParameterError, match="M - 13 with xl 5"):
        read_with_scales(tmp_path, layer, **{M: 34, K: 20})
    scales = read_with_scales(tmp_path, layer, **{M: 34, K: 21}).layers[layer]
    assert (getattr(scales, M), getattr(scales, K)) == (34, 21)
