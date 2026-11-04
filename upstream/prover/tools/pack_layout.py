#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Makes a configuration's layout file, `layout.txt`: where a request's proof stores each array.

usage: cp_verify -plan arrays <configuration directory> <bits> |
         pack_layout.py --arrays - <log2 of a row's message length> <free> > <new layout.txt>

A request's sub-proofs commit to arrays (matrices), which are stored in the rows of oracle
messages (encoded rows, sent as Merkle roots). Every row costs bytes in each opened column and time
in the encoding, so lay() packs the arrays tightly. The prover, cp_prove, and the verifier,
cp_verify, both read and check `layout.txt`, and neither has a default layout.
blocks/frame/req_frame.h defines the file's lines (a group's, a block's) and their rules.

An input line
  array: <link> <moment> <group> <class> <life> <index> <rows> <columns> <low> <name>
names one array and the oracle message (<group>) that is to hold it. <link> numbers a segment, one
sub-proof's part of the proof; <moment> numbers a commit point within it (`link` and `moment` in
identifiers and printed messages). <class> is N (rows over the base field F), W (rows over E, its
cubic extension) or B (rows over F, kept for <life> later segments). <class> and <life> go unchanged
into the group's line, <index> into the lines of the array's blocks. <low> is r if neighbouring rows
of a column are stored side by side, else c. Lines that don't begin with "array:" are skipped.

The message length is 2^l for the l of the line `group link J n l nz lz` of the parameter file,
`policy.txt`. The last <free> entries of every row stay empty (<free> is a power of two), because an
oracle message's last row can end in the pad that masks its out-of-domain sample's answers: three
entries for each answer in a row over F (blocks/iop/send.h, Send::pad()).

Refused, with status 2, one line on standard error and nothing on standard output: <free> at or above
the row's length; a length above 2^31; a line of 2^20 characters or more; an "array:" line without
the 11 fields, or whose numbers are not whole numbers; an array with no rows or columns, or with 2^48
or more of either; a group that would take 2^32 rows or more. Beyond these, and the plain checks on
the call and the file, the tool doesn't look at what a line says: neither the length nor <free> is
compared with the parameter file. The programs check the file that comes out
(blocks/frame/req_frame.h).
"""

import sys

USAGE = (
    "usage: pack_layout.py --arrays <file holding the output of cp_verify -plan arrays, or - for standard input>"
    " <log2 of a row's message length> <entries left empty at each row's end, a power of two>"
)
# The limits of the format. A group's l is below its n, which is at most 32 (read_group(), blocks/frame/parameters.h).
MOST_LOG_LENGTH = 31
# A group has fewer than 2^32 rows, and a block's sides are below 2^48 (layout_read(), blocks/frame/req_frame.h).
MOST_ROWS = (1 << 32) - 1
SIDES_BELOW = 1 << 48
LINES_BELOW = 1 << 20  # characters: so that a source that sends no line feed can't fill the memory
# The fields of an "array:" line that hold a number, by name and place.
NUMBERS = (("link", 1), ("moment", 2), ("group", 3), ("life", 5), ("index", 6), ("rows", 7), ("columns", 8))


def stop(text):
    print("pack_layout.py: " + text, file=sys.stderr)
    sys.exit(2)


def is_number(word):
    """Whether a word is a whole number of ASCII digits, 19 at most."""
    return word.isascii() and word.isdigit() and len(word) < 20


def pieces(n):
    """n as a sum of powers of two, the largest first: (offset, log2 size)."""
    out, at = [], 0
    for bit in range(62, -1, -1):
        if (n >> bit) & 1:
            out.append((at, bit))
            at += 1 << bit
    return out


# Source for free regions of power-of-two size that are halved until they fit: the buddy system of
# memory allocation: Wikipedia, "Buddy memory allocation", which names, after Knuth, Harry Markowitz
# (1963) and Knowlton, "A fast storage allocator", Comm. ACM 8 (1965). There one dimension, in the
# order of the requests; here two, sorted by size.
def lay(arrays, rows, places):
    """Places arrays in the rows 0 .. rows - 1, within the first `places` entries of each.

    arrays: (index, R, K, low_is_row) for each array of R x K elements. Returns the blocks, each as
    the fields of a block's line from the array's index on, or None if the arrays do not fit, which
    by_arrays() never meets: it gives lay() rows enough.

    An array is cut into rectangles whose sides are powers of two, each aligned to its own size. The
    rows are cut into free regions of 2^br rows by 2^bj entries, aligned in the same way. The
    largest rectangles go first, each into the smallest free region that can hold it, which is
    halved down to the rectangle's size. If there is none, the rectangle is halved. Since every
    rectangle can be halved down to one element, the arrays fit if they have at most
    rows * places elements.
    """
    items = [[] for _ in range(64)]  # by log2 size: (array, R_u, a_u, K_w, e_w, low_is_row)
    for index, r, k, low in arrays:
        for r_u, a_u in pieces(r):
            for k_w, e_w in pieces(k):
                items[a_u + e_w].append((index, r_u, a_u, k_w, e_w, low))
    free = [[] for _ in range(64)]  # by log2 size: (r0, br, j0, bj)
    for j0, bj in pieces(places):
        for r0, br in pieces(rows):
            free[br + bj].append((r0, br, j0, bj))
    for f in free:
        f.sort(reverse=True)  # pop() gives the lowest row, then the lowest place
    blocks = []
    for size in range(63, -1, -1):  # the largest rectangles first
        items[size].sort(reverse=True)
        while items[size]:
            item = items[size].pop()
            room = size  # the smallest free region that holds it
            while room < 64 and not free[room]:
                room += 1
            if room == 64:  # none: halve the rectangle, its `low` side only if the other is 1 long
                if size == 0:
                    return None
                index, r_u, a_u, k_w, e_w, low = item
                columns = e_w > 0 if low else a_u == 0
                for half in (1, 0):
                    if columns:
                        items[size - 1].append((index, r_u, a_u, k_w + (half << (e_w - 1)), e_w - 1, low))
                    else:
                        items[size - 1].append((index, r_u + (half << (a_u - 1)), a_u - 1, k_w, e_w, low))
                continue
            r0, br, j0, bj = free[room].pop()
            while room > size:  # halve the region, its rows first, then its places; the upper half stays free
                room -= 1
                if br:
                    br -= 1
                    free[room].append((r0 + (1 << br), br, j0, bj))
                else:
                    bj -= 1
                    free[room].append((r0, 0, j0 + (1 << bj), bj))
            index, r_u, a_u, k_w, e_w, low = item
            blocks.append((index, r_u, a_u, k_w, e_w, "r" if low else "c", r0, j0, br, bj, 0))
    return blocks


def by_arrays(path, log_ell, free):
    """Writes a `layout.txt` from the lines "array:" of the file `path` ("-": standard input)."""
    if log_ell > MOST_LOG_LENGTH:
        stop("a row's message length of 2^%d is too long: take l from the line `group link J n l nz lz` of policy.txt; it is at most %d" % (log_ell, MOST_LOG_LENGTH))
    if free >= 1 << log_ell:
        stop(
            "%d entries left empty leave no room in a row of 2^%d entries: check the row's length, the l of the line"
            " `group link J n l nz lz` of policy.txt, or give fewer empty entries" % (free, log_ell)
        )
    groups, order = {}, []  # (link, moment, group) -> [class, life, [(index, rows, columns, low_is_row)]]
    try:
        with sys.stdin if path == "-" else open(path) as f:
            while True:
                line = f.readline(LINES_BELOW)
                if not line:
                    break
                if len(line) == LINES_BELOW:
                    stop("%s has a line of %d characters or more: give the output of cp_verify -plan arrays" % (path, LINES_BELOW))
                w = line.removeprefix("\ufeff").split()  # a byte-order mark is no part of the first word
                if not w or w[0] != "array:":
                    continue
                shown = ascii(line.strip()[:200])
                if len(w) != 11:
                    stop("a line that begins array: has %d fields, not 11: is the list cut off? The line: %s" % (len(w), shown))
                for name, at in NUMBERS:
                    if not is_number(w[at]):
                        stop("%s %s is not a whole number of at most 19 digits, in the line: %s" % (name, ascii(w[at][:40]), shown))
                if not (1 <= int(w[7]) < SIDES_BELOW and 1 <= int(w[8]) < SIDES_BELOW):
                    stop("an array needs at least 1 row and 1 column, and fewer than 2^48 of each, in the line: " + shown)
                key = (int(w[1]), int(w[2]), int(w[3]))
                if key not in groups:
                    groups[key] = [w[4], int(w[5]), []]
                    order.append(key)
                if groups[key][:2] != [w[4], int(w[5])]:
                    stop("the arrays of link %d, moment %d, group %d differ in class or life" % key)
                groups[key][2].append((int(w[6]), int(w[7]), int(w[8]), w[9] == "r"))
    except OSError as error:
        stop("%s can't be read: %s" % (path, error.strerror))
    except UnicodeDecodeError:
        stop("%s is not text: give the output of cp_verify -plan arrays" % path)
    if not order:
        stop("no line 'array:' was read")
    places = (1 << log_ell) - free
    for key in order:
        need = (sum(x[1] * x[2] for x in groups[key][2]) + places - 1) // places
        if need > MOST_ROWS:
            stop(
                "link %d, moment %d, group %d would take %d rows of %d usable entries, and a group has fewer than 2^32 rows:"
                " check the row's length, the l of the line `group link J n l nz lz` of policy.txt" % (*key, need, places)
            )
    print("# a group is an oracle message; its rows are counted from 0")
    rows_F = rows_E = count = 0
    for key in order:
        cls, life, listed = groups[key]
        rows = (sum(x[1] * x[2] for x in listed) + places - 1) // places
        blocks = lay(listed, rows, places)  # rows is enough: rectangles halve down to one element
        print(*key, cls, rows, life)
        for b in blocks:
            print(*key, *b)
        count += len(blocks)
        if cls == "W":
            rows_E += rows
        else:
            rows_F += rows
    print("pack_layout.py: %d rows over F, %d rows over E, %d blocks" % (rows_F, rows_E, count), file=sys.stderr)


def main():
    args = sys.argv[1:]
    if "--arrays" in args:
        args.remove("--arrays")
        if len(args) == 3 and is_number(args[1]) and is_number(args[2]) and int(args[2]) and not int(args[2]) & (int(args[2]) - 1):
            return by_arrays(args[0], int(args[1]), int(args[2]))
    stop(USAGE)


if __name__ == "__main__":
    main()
