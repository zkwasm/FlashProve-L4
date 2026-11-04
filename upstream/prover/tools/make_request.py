#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Writes a request file for the programs cp_prove and cp_verify: "which token follows this prompt?"
or "are these tokens the model's own?"

usage: make_request.py <policy.txt> <request file to write> [<file of tokens> [<r> <e>]]

Lines `lt` and `V` of the parameter file, policy.txt, give T = 2^lt and the vocabulary size V; the
file is not validated. A request file is T + 2 little-endian 32-bit words: the window of T
token ids, then r and e. All T tokens are the model's input. A proof is about tokens r + 1 to
min(e, T - 1) of the window, counted from 0, and, if e = T, about y, the token that follows the
window (defined in verifier/README.md). Without <r> and <e>, this tool writes r = T - 1 and e = T,
as `cp_verify -next-token` requires, so the window is the prompt and the proof is about y only.

With <r> and <e> (decimal, r < e <= T, only with a file of tokens), it writes these values. Under
the exact-match policy cp_prove makes a proof only if every token r + 1 to min(e, T - 1) of the
window is the model's top choice after the tokens before it. Otherwise cp_prove writes an empty
`.failed` file; that is the outcome for synthetic tokens, which are not the model's top choices.
Example requests whose tokens are top choices are listed in
prover/examples/TARGET-KITS.md, "The examples' requests".

Without a file of tokens, the window contains synthetic tokens that depend only on T and V. With
one, its first T little-endian 32-bit words are the window.
The request file is created with mode 0600, so that only its owner can read it. If anything has
its name already, nothing is written.
Exit status 0: written. 1: an uncaught exception, such as a missing policy.txt.
2: wrong number of arguments, fewer than T tokens, a token not below V, r and e that are not decimal
numbers with r < e <= T, or an existing request file.
"""

import os
import struct
import sys


def main():
    a = sys.argv[1:]
    if len(a) not in (2, 3, 5):
        print(__doc__[__doc__.index("usage: ") :].split("\n")[0], file=sys.stderr)
        return 2
    with open(a[0], encoding="ascii") as f:
        word = dict(line.split() for line in f if len(line.split()) == 2)
    tokens_in_all, vocabulary = 1 << int(word["lt"]), int(word["V"])
    # r and e
    first, end = tokens_in_all - 1, tokens_in_all
    if len(a) == 5:
        if not (a[3].isascii() and a[3].isdigit() and a[4].isascii() and a[4].isdigit() and int(a[3]) < int(a[4]) <= tokens_in_all):
            print("make_request.py: r and e are decimal numbers with r < e <= %d" % tokens_in_all, file=sys.stderr)
            return 2
        first, end = int(a[3]), int(a[4])
    if len(a) >= 3:
        with open(a[2], "rb") as f:
            given = f.read(4 * tokens_in_all)
        if len(given) != 4 * tokens_in_all:
            print("make_request.py: %s holds fewer than %d tokens" % (a[2], tokens_in_all), file=sys.stderr)
            return 2
        tokens = struct.unpack("<%dI" % tokens_in_all, given)
        if max(tokens) >= vocabulary:
            print("make_request.py: %s holds a token that is not below %d" % (a[2], vocabulary), file=sys.stderr)
            return 2
    else:
        # The constants only scatter ids below V.
        tokens = [(2654435761 * (t + 1) >> 7) % vocabulary for t in range(tokens_in_all)]
    try:
        # O_EXCL: the creation fails if anything has the name, a symbolic link too.
        # 0o600: the file may hold a person's prompt.
        with open(os.open(a[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600), "wb") as f:
            f.write(struct.pack("<%dI" % (tokens_in_all + 2), *tokens, first, end))
    except FileExistsError:
        print("make_request.py: %s exists already, and this tool overwrites nothing: name a new file" % a[1], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
