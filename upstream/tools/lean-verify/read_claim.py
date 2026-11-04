#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Reads the claim off a caller's files, by the format page alone, and prints it.

usage: python3 -I -B tools/lean-verify/read_claim.py CONFIG_DIR REQUEST_FILE TOKEN_FILE ID_FILE

tools/lean-verify/lean_verify.sh calls it before Lean. It is a second reading of the claim, beside
Lean's own (PV.Program.claimOf, with PV.Front.idOf and PV.Front.userOf), about which there is no
theorem: the script gives a verdict only where the six lines below are, byte for byte, the six lines
in which Lean prints the claim that it judges. It shares no code with Lean's readers and none with
the C++ verifier: it follows docs/format/FORMAT.md, 12.2, and takes T and V from the parameter file
(CONFIG_DIR/policy.txt, the lines "lt" and "V"). It judges no proof and writes no file.

It prints six lines: the commitment ID, the two positions r and e, the prompt's tokens, the output's
tokens, the number of the request's tokens behind position e, and the parameter file's seven lines on
the acceptance policy (FORMAT.md, 12.5).
Exit code 0: the six lines were printed. 1: a file fails one of the format page's five tests, the ID
file is not 64 hex digits in lower case and then at most a newline, or the parameter file holds a
control character or does not have each line that is needed exactly once and in the plain form
"<name> <number>", the number of at most 20 digits. 2: wrong arguments, or a file that cannot be
read.
"""
import os
import re
import struct
import sys

LIMIT = 2 ** 28  # no file above this many bytes is read
POLICY = ("C", "theta", "cap", "s", "sh", "sT", "F")  # docs/format/FORMAT.md, 12.5


def stop(code, why):
    print("read_claim.py: " + why, file=sys.stderr)
    sys.exit(code)


def read(path, role):
    try:
        if not os.path.isfile(path):
            stop(2, role + " is no regular file")
        if os.path.getsize(path) > LIMIT:
            stop(2, role + " has more than 2^28 bytes")
        with open(path, "rb") as f:
            return f.read(LIMIT + 1)
    except OSError as e:
        stop(2, role + " cannot be read: " + str(e.strerror))


def words(data, role, count):
    # Test 1: the length is a multiple of 4 bytes, and the file is not empty.
    if len(data) == 0 or len(data) % 4 != 0:
        stop(1, role + ": its length is 0 or no multiple of 4 bytes (test 1)")
    # Test 2: the number of words. Held before the bytes are turned into numbers.
    if len(data) != 4 * count:
        stop(1, role + " must have exactly %d words of four bytes (test 2)" % count)
    return list(struct.unpack("<%dI" % count, data))


def line_of(lines, name):
    found = [l.split() for l in lines if l.split()[:1] == [name]]
    if len(found) != 1 or len(found[0]) != 2 or not found[0][1].isdigit() or len(found[0][1]) > 20:
        stop(1, "the parameter file does not have exactly one line '%s <number>', of at most 20 digits" % name)
    return int(found[0][1])


def main(argv):
    if len(argv) != 5:
        stop(2, "usage: read_claim.py CONFIG_DIR REQUEST_FILE TOKEN_FILE ID_FILE")  # wrong arguments
    try:
        policy = read(os.path.join(argv[1], "policy.txt"), "the parameter file").decode("ascii")
    except UnicodeDecodeError:
        stop(1, "the parameter file is no ASCII text")
    # A line such as "C<NUL> 255" would be no line of the name C to this reader, and still lie in the file.
    if any((c < " " and c != "\n") or c == "\x7f" for c in policy):
        stop(1, "the parameter file holds a control character other than the newline")
    lines = policy.split("\n")
    lt = line_of(lines, "lt")
    if lt > 26:  # a request file of T + 2 words would pass the limit on a file's bytes
        stop(1, "the parameter file's line 'lt' is above 26")
    T = 2 ** lt
    V = line_of(lines, "V")
    request = words(read(argv[2], "the request file"), "the request file", T + 2)
    token = words(read(argv[3], "the token file"), "the token file", 1)
    ident = read(argv[4], "the commitment ID's file")

    tokens, r, e, y = request[:T], request[T], request[T + 1], token[0]
    if not r < e <= T:
        stop(1, "r < e <= T does not hold (test 3)")
    if e < T and y != 0:
        stop(1, "e < T, and the token file does not hold 0 (test 4)")
    if any(t >= V for t in tokens) or y >= V:
        stop(1, "a token is not below the vocabulary's size (test 5)")
    if ident[64:] not in (b"", b"\n") or len(ident) < 64 or any(c not in b"0123456789abcdef" for c in ident[:64]):
        stop(1, "the ID file is not 64 hex digits in lower case and then at most a newline")

    # Position t is about token t + 1; position T - 1 is about the next token, y.
    after = tokens + [y]
    prompt, output, further = tokens[: r + 1], after[r + 1 : e + 1], tokens[e + 1 :]
    show = lambda l: "[" + ", ".join(str(t) for t in l) + "]"
    # The seven lines of the acceptance policy, in the file's order. Each name once, and each line
    # plain: else what is shown could be read otherwise than the file (a second line of one name, a
    # tab for the blank, a carriage return or an escape sequence that repaints a terminal's line).
    named = [l for l in lines if l.split()[:1] and l.split()[0] in POLICY]
    if sorted(l.split()[0] for l in named) != sorted(POLICY) or not all(
            re.fullmatch(r"[A-Za-z]+ (0|[1-9][0-9]{0,19})", l) for l in named):
        stop(1, "the parameter file does not have, for each of C, theta, cap, s, sh, sT and F, exactly one "
                "line, of the plain form '<name> <number>', the number of at most 20 digits")
    print("claim: commitment ID: " + ident[:64].decode("ascii"))
    print("claim: checked positions: from r = %d up to, not including, e = %d" % (r, e))
    print("claim: prompt, %d tokens: %s" % (len(prompt), show(prompt)))
    print("claim: output, %d tokens: %s" % (len(output), show(output)))
    print("not part of the claim: tokens of the request behind position e, which go to the verifier "
          "with the proof: %d" % len(further))
    print("public files: the acceptance policy, by the parameter file's own lines: [" + ", ".join(named) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
