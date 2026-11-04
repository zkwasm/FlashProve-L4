#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Started by weight_sums/tests/run_tests.sh, which the runner starts; by hand: below a directory of your own.
"""Test the exponent that bin/weight_sums and weight_sums.py choose at a matrix's rounding edge.

usage: edge.py <new folder> --kind cpp|py -- <program and its first arguments>

The command line, the way in which a case runs, the output and the exit status are those of
weight_sums/tests/refusals.py, whose docstring describes them. In short: --kind is cpp for
bin/weight_sums and py for weight_sums.py, <new folder> must not exist yet, and the paths after
"--" must be absolute.

The conversion rule turns a model's stored weights into integers. As defined in
docs/model.md, "11.4 From a decoded value to an integer", it gives each matrix an exponent e, the
largest integer with RNE(2^e * mx) <= 4,095. Here mx is the largest absolute value in the matrix,
and RNE rounds to the nearest integer and takes a tie to the even neighbour. The integer of an
element p is then RNE(2^e * p). In the programs' output, the sums file, a matrix has one line with
the fields name, elements, e, bits and sha256 (of the integers).

The edge is at 2^e * mx = 4,095.5. That is a tie, and RNE takes it to 4,096, which is too large. So
a matrix with mx = 4,095.5 * 2^k has the exponent -k - 1, and so has a matrix whose mx is a little
larger. A matrix whose mx is a little smaller has the exponent -k. The number
4,095.5 * 2^k = 8,191 * 2^(k-1) has 13 significant bits, so at the k used here it is a binary32
number (IEEE 754 single precision), and the values nearest to the edge are its two neighbours in
that format.

Each case is a shard (one GGUF file) with one F32 (binary32) matrix of 256 elements. The matrix
reads no norm: the rule multiplies no normalization weights into it, and rounds the stored values
themselves. POWERS, STEPS and TIES below give its values. This script does not try the edge on a
quantized matrix or on one that reads a norm.

weight_sums/tests/oracle.py computes each case's expected result in exact arithmetic. This script
also gives two fields of the sums file in closed form, and stops if oracle.py disagrees:

  e     -k or -k - 1, as above. Outside -100 .. 100, the programs must reject the shard (status 25).
  bits  the least b with -2^(b-1) <= w < 2^(b-1) for every integer w of the matrix. It is 13, with
        one exception. On the edge and above it, the largest element becomes 2,048 or -2,048, and
        all other integers are smaller in size. In two's complement, twelve bits hold -2,048 but
        not 2,048, so bits is 12 if the sign is minus.

The programs' lower limit for a multiplier (a norm's weight times a constant) has a rounding edge
too. The cases bound.multiplier-* of weight_sums/tests/refusals.py test it. Both that limit and the
exponent's are listed under exit status 25 in weight_sums/README.md, "Make the file".
"""

import struct
import sys

sys.dont_write_bytecode = True  # so that importing refusals.py leaves no __pycache__ directory

from refusals import BOUND, F32, floats, main, stop, u32

# The values of k. The rule is the same at every power of two, so k matters only where something
# else changes with it.
#   -1 and 0       The exponent goes from 1 through 0 to -1, so the field e changes its sign.
#   -101 and -100  The upper limit of the exponent. At k = -101 the edge decides whether the matrix
#                  is valid: the exponent is 101 below it and 100 on it. At k = -100 it is 100 and
#                  99, both valid.
#   99 and 100     The lower limit. At k = 100 the exponent is -100 below the edge and -101 on it.
#                  At k = 99 it is -99 and -100, both valid.
POWERS = (-101, -100, -1, 0, 99, 100)

# The largest element: the edge itself, and its neighbours among the binary32 numbers. mx can lie
# below the tie, on it or above it, and the values nearest to the tie are the hardest ones to get
# right. An entry: the word in the case's name, the step in the bit pattern.
STEPS = (("below", -1), ("on", 0), ("above", 1))

# The other elements that are not 0, as multiples of 2^k. They are ties of RNE(2^e * p), which show
# through the field sha256 that a program rounds with the exponent that it states, and to the even
# neighbour. The first row has ties at the exponent -k, where 2^e * p is the number itself. The
# second row has ties at -k - 1, where it is half the number. Each row has the two smallest ties of
# each sign and two large ones. In each pair one tie goes down in size to the even neighbour, and
# the other goes up. 4,095 is left out of the second row: it would become 2,048, and bits would be
# 13 in every case.
TIES = (0.5, 1.5, -0.5, -1.5, 4093.5, 4094.5,
        1.0, 3.0, -1.0, -3.0, 4091.0, 4093.0)


# One case for each k, place and sign, named k-<k>.<place>.<sign>, as in k-minus-101.below.plus.
def edge_cases(cases):
    for k in POWERS:
        # The bit patterns of positive binary32 numbers are in the order of the numbers, so a
        # neighbour differs by 1. The highest bit is the sign (IEEE 754).
        edge = struct.unpack("<I", floats([4095.5 * 2.0 ** k]))[0]
        rest = floats([tie * 2.0 ** k for tie in TIES]) + bytes(4 * (255 - len(TIES)))
        for place, step in STEPS:
            for sign, sign_bit in (("plus", 0), ("minus", 0x80000000)):
                case = ("k-%d.%s.%s" % (k, place, sign)).replace("--", "-minus-")
                tensors = [("blk.0.attn_output.weight", (256, 1), F32, u32((edge + step) | sign_bit) + rest)]
                e = -k if step < 0 else -k - 1
                bits = 12 if step >= 0 and sign == "minus" else 13
                if not -100 <= e <= 100:
                    cases.fault(case, BOUND, tensors)  # it stops the script if oracle.py does not find this status
                    continue
                cases.valid(case, tensors)
                fields = cases.rows[-1].expected.decode().split()
                if fields[2:4] != [str(e), str(bits)]:
                    stop("in the case %s the exponent must be %d and the field bits %d, but oracle.py finds %s and %s"
                         % (case, e, bits, fields[2], fields[3]))


if __name__ == "__main__":
    sys.exit(main(edge_cases))
