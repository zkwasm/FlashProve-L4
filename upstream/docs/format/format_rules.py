#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""The rules R1.x to R8.x and R12.x to R15.x of FORMAT.md as a program, with their known answers.

It takes no argument and prints the known answers: format-vectors.txt is this output. From the
repository's top:
    python3 docs/format/format_rules.py | diff - docs/format/format-vectors.txt
The readers beside it (read_*.py) import its functions.

It needs python3 with its standard library and nothing else, and it imports nothing of the
programs (blocks/). Every hash is one call of the library's BLAKE2b (hashlib), except the second
hash function, K of R15.3, which needs BLAKE2b's compression function written out (blake2b_F). That
text is adapted from the BLAKE2 reference implementation, as is the text in blocks/: a comparison
of the two can show an error made in adapting, not an error of that implementation. Chains and
sampling serve both fields; proof_head, point and symbol serve field 2 alone.

What it does NOT have: sizes and plans (section 8 of FORMAT.md, in 08-parameters.md), a segment's
sub-proof (section 11), a verifier's tests on content (section 13). The functions for R15.x were
written from the programs' source and not from the rules.
"""
import hashlib
import struct

MASK = (1 << 64) - 1
PRIME = {1: (1 << 61) - 1, 2: (1 << 64) - (1 << 32) + 1}

# ---- R2.x and R3.x: the hash ----
START = {162: "the chain's value", 164: "run", 166: "configuration", 171: "seed", 173: "configuration", 174: "configuration", 178: "configuration"}
NUMBERS = {160, 162, 164, 166, 168, 169, 171, 172, 173, 174, 176, 178, 179}


LEAVES = {164, 173, 174}


def H(number, tree, index, start, data, size=32, width=0, salt_words=0):
    assert (0 < width < 1 << 24 and 5 <= salt_words <= 8) if number in LEAVES else width == salt_words == 0
    assert number in NUMBERS and 0 <= tree < 1 << (48 if number == 162 else 16) and 0 <= index < 1 << 63 and len(data) <= (1 << 32) - 144
    assert (start is not None) == (number in START) and size == (64 if number == 162 else 32)
    start = bytes(32) if start is None else start
    assert len(start) == 32
    return hashlib.blake2b(label(number, tree, index, width, salt_words) + data, digest_size=size, salt=start[:16], person=start[16:]).digest()


def label(number, tree, index, width=0, salt_words=0):
    return struct.pack("<QQ", number + (tree << 16) + (width << 32) + (salt_words << 56), index)


def oracle_string(half, number, tree, index, start, data, size=32, width=0, salt_words=0):
    start = bytes(32) if start is None else start
    return bytes([half, size, 0, 1, 1]) + bytes(28) + start + label(number, tree, index, width, salt_words) + data


# ---- R1.x: bytes ----
def u64(x):
    return struct.pack("<Q", x)


def string(b):
    return u64(len(b)) + b


def text(s):
    return string(s.encode("ascii"))


def ext(e):
    return b"".join(u64(c) for c in e)


# ---- R4.x: the first query ----
def item(name, value):
    return text(name) + u64(len(value)) + value


def plan_bytes(field, salt_bytes, nonce_bytes, position_bytes, samples, saved, rounds):
    """R4.3. rounds: (name, count, work_bits, message_bytes, roots, answers, draw_E, draw_nonzero,
    draw_positions, log_n, draw_trits, rows)."""
    out = b"".join(u64(x) for x in (field, 1, salt_bytes, nonce_bytes, position_bytes, samples, saved, len(rounds)))
    for r in rounds:
        assert len(r) == 12
        out += text(r[0]) + b"".join(u64(x) for x in r[1:])
    return out


def parameters_head(field, plan):
    return item("the format's version", ("H8%d1" % field).encode("ascii") + bytes(4)) + plan


def configuration_digest(files):
    return H(169, 0, 2, None, u64(len(files)) + b"".join(u64(len(f)) + f for f in files))


def table_name(root, configuration, a, message, tail, log_rows, answers, salt_bytes):
    return root + configuration + b"".join(u64(x) for x in (a, message, tail, log_rows, answers, salt_bytes))


def published_id(configuration, first_table_name, seeds, record_answers):
    """R4.11: the commitment ID. seeds: for each answer of a row a pair: the list of the point's
    encoded elements, and its encoded factor (R6.9). record_answers: the encoded elements of E,
    by row and within a row by answer. An encoded element has 24 bytes."""
    listed = u64(sum(len(z) + 1 for z, beta in seeds)) + b"".join(b"".join(z) + beta for z, beta in seeds)
    return H(179, 0, 0, None, configuration + string(first_table_name) + listed + u64(len(record_answers)) + b"".join(record_answers) + u64(0))


def parameters_digest(parameters):
    return H(169, 0, 1, None, parameters)


def run_value(label, parameters, random_bytes):
    return H(172, 0, label, None, parameters + random_bytes)


def first_value(label, parameters, statement, run, record=b""):
    data = parameters + u64(len(statement)) + statement + u64(len(record)) + record
    data += u64(0) if run is None else u64(1) + run
    return H(160, 0, label, None, data)


# ---- R5.x and R6.x: a round's query, and sampling ----
ALGEBRAIC, EPSILON, SEED, POSITIONS, BITS, SEAL, EMPTY = 1, 2, 3, 4, 5, 6, 7
TRIES, TRIES_NONZERO = 6, 2


class Chain:
    def __init__(self, rho0, field, label):
        self.rho, self.round, self.field, self.label = rho0, 0, field, label
        self.before, self.tail, self.first, self.taken = None, None, 0, 0

    def call(self, rho, i, message, kind, w, nonce, j):
        assert len(message) % 8 == 0 and (w > 0 or nonce == 0) and (kind != EMPTY or w == 0)
        return H(162, self.label, i, rho, message + u64(kind + (w << 8) + (j << 16)) + u64(nonce), 64)

    def block(self, b):
        """Block b of the present round's answer."""
        c = self.call(self.before, self.round, *self.tail, b // 2)
        return c[32 * (b % 2):32 * (b % 2) + 32]

    def passes(self, message, kind, w, nonce):
        c = self.call(self.rho, self.round + 1, message, kind, w, nonce, 0)
        return w == 0 or struct.unpack("<Q", c[:8])[0] & ((1 << w) - 1) == 0

    def query(self, message, kind, w=0, nonce=0):
        """False, and the chain stays as it was, if the proof-of-work test (R5.5) fails."""
        if not self.passes(message, kind, w, nonce):
            return False
        self.before, self.round, self.tail = self.rho, self.round + 1, (message, kind, w, nonce)
        self.first = 2 if w else 1          # the first block of the verifier's message
        self.rho, self.taken = self.block(self.first - 1), 0
        return True

    def least_nonce(self, message, kind, w):
        n = 0
        while not self.passes(message, kind, w, n):
            n += 1
        return n

    def word(self):
        b = self.block(self.first + self.taken // 4)
        self.taken += 1
        return struct.unpack("<4Q", b)[(self.taken - 1) % 4]

    def base(self):
        for _ in range(TRIES):
            x = element_of_word(self.word(), self.field)
            if x is not None:
                return x
        return 0

    def element(self, nonzero=False):
        for _ in range(TRIES_NONZERO if nonzero else 1):
            e = (self.base(), self.base(), self.base())
            if not nonzero or e != (0, 0, 0):
                return e
        return (1, 0, 0)

    def position(self, k):
        return self.word() & ((1 << k) - 1)

    def trits(self, n):
        out, w = [], 0
        for i in range(n):
            if i % 32 == 0:
                w = self.word()
            out.append((-1 if w & 2 else 1) if w & 1 else 0)
            w >>= 2
        return out

    def seal_digest(self):
        return b"".join(u64(self.word()) for _ in range(4))


def element_of_word(w, field):
    if field == 1:
        w &= PRIME[1]
        return None if w == PRIME[1] else w
    return None if w >= PRIME[2] else w


def base_from_words(words, field):
    """R6.4 on a given sequence of words: (the element, how many words were used)."""
    for n, w in enumerate(words[:TRIES]):
        x = element_of_word(w, field)
        if x is not None:
            return x, n + 1
    return 0, TRIES


def words_at_most(draw_E, draw_nonzero, positions, portions, seal):
    return 18 * (draw_E - draw_nonzero) + 36 * draw_nonzero + positions + sum((n + 31) // 32 for n in portions) + (4 if seal else 0)


def record_seeds(name, variables, answers, field):
    c = Chain(first_value(1 << 32, bytes(32), name, None), field, 1 << 32)
    c.query(b"", SEED)
    return [([c.element() for _ in range(variables)], c.element(True)) for _ in range(answers)]


# ---- R7.x: a tree and its opening ----
def leaf_bytes(column, salt):
    assert len(salt) % 8 == 0 and 40 <= len(salt) <= 64
    data = b"".join(u64(s) for s in column)
    if (16 + len(data)) % 128 + len(salt) > 128:
        data += bytes(128 - (16 + len(data)) % 128)
    return data + salt


def leaf(number, tree, a, x, start, column, salt):
    assert number in (164, 174)
    return H(number, tree, (1 << a) + x, start, leaf_bytes(column, salt), 32, len(column), len(salt) // 8)


def binding_leaf(depth, place_, configuration, seal, salt):
    """R7.18."""
    return H(173, 0, (1 << depth) + place_, configuration, leaf_bytes(struct.unpack("<4Q", seal), salt), 32, 4, len(salt) // 8)


def vertex(configuration, left, right):
    return H(166, 0, 0, configuration, left + right)


def binding_vertex(configuration, left, right):
    """R7.18: as a vertex of any tree, under the number 178."""
    return H(178, 0, 0, configuration, left + right)


def binding_root(depth, place_, configuration, seal, salt, siblings):
    """R14.7. siblings: from the leaf upward."""
    assert place_ >> depth == 0 and len(siblings) == depth
    node = binding_leaf(depth, place_, configuration, seal, salt)
    for s in siblings:
        node = binding_vertex(configuration, s, node) if place_ & 1 else binding_vertex(configuration, node, s)
        place_ >>= 1
    return node


def short_position(x, b):
    return x & ((1 << b) - 1)


def build(a, configuration, leaves):
    value = {a: list(leaves)}
    for d in range(a - 1, -1, -1):
        value[d] = [vertex(configuration, value[d + 1][2 * i], value[d + 1][2 * i + 1]) for i in range(1 << d)]
    return value


def walk(a, configuration, xs, opened, given, enter=None):
    """xs: the list, increasing. opened[j]: the digest of the leaf at xs[j]. given(d, i): the next
    given digest. Returns the root. enter(d, i, value), if there is one, is told every value of the
    walk, computed or given (R7.14)."""
    enter = enter or (lambda d, i, value: None)
    now = list(zip(xs, opened))
    for d in range(a, 0, -1):
        for i, x in now:
            enter(d, i, x)
        nxt, v = [], 0
        while v < len(now):
            i, x = now[v]
            if i % 2 == 0 and v + 1 < len(now) and now[v + 1][0] == i + 1:
                nxt.append((i >> 1, vertex(configuration, x, now[v + 1][1])))
                v += 2
                continue
            other = given(d, i ^ 1)
            enter(d, i ^ 1, other)
            nxt.append((i >> 1, vertex(configuration, other, x) if i & 1 else vertex(configuration, x, other)))
            v += 1
        now = nxt
    return now[0][1]


def lay_trees(notices):
    """R7.12. notices: (leaf class, a or None, b or None) for the oracle messages of one prover's
    message, in the order of sending: a, b the logarithms of the lengths of its main rows and of
    its mask rows. Returns the trees in the order in which they are opened, each as the list of its
    parts, by their numbers in the order of sending."""
    trees = []  # [class, a, parts]
    for mask_only in (False, True):
        for i, (cls, a, b) in enumerate(notices):
            assert a is not None or b is not None
            if (a is None) != mask_only:
                continue
            for t in trees:
                if t[0] == cls and (mask_only or t[1] == a):
                    t[2].append(i)
                    break
            else:
                trees.append([cls, a, [i]])
    return [sorted(t[2]) for t in trees]


def number_trees(events, first=0):
    """R7.13. events, in the order of the protocol: a list of notices (a prover's message that sends
    oracle messages), or the word "adopt" (the chain adopts one tree of another chain). first: the
    number of places that the list has at the chain's beginning (0, or that of the chain that it
    continues). Returns for each event the places taken; for a message they are its trees' numbers."""
    out = []
    for e in events:
        n = 1 if e == "adopt" else len(lay_trees(e))
        out.append(list(range(first, first + n)))
        first += n
    assert first <= 1 << 16
    return out


class OneValue:
    """R7.14: the record of values under one root's 32 bytes, for one session of a verifier."""

    def __init__(self):
        self.seen = {}

    def enter(self, depth, place_, value):
        """False: this vertex under this root has another value already."""
        return self.seen.setdefault((1 << depth) + place_, value) == value


class Session:
    """R7.14: one record for every root that is read in a session, whichever tree, chain or file it
    came with."""

    def __init__(self):
        self.records = {}

    def record(self, root):
        assert len(root) == 32
        return self.records.setdefault(bytes(root), OneValue())


# ---- R12.x to R14.x: types, items and rounds ----
NO_CODE = (1 << 32) - 1


def code_bytes(log_n, message, tail):
    return u64(log_n) + u64(message) + u64(tail)


def codes_bytes(base, mask):
    return u64(len(base)) + b"".join(code_bytes(*c) for c in base) + u64(len(mask)) + b"".join(code_bytes(*c) for c in mask)


def notice_bytes(code, mask_code, main_rows, degree, mask_rows, answers, leaf_class):
    return b"".join(u64(x) for x in (code, mask_code, main_rows, degree, mask_rows, answers, leaf_class))


def choices_bytes(masks_ride):
    return u64(masks_ride)


def stages_bytes(stages):
    return u64(len(stages)) + b"".join(b"".join(u64(x) for x in s) for s in stages)


def chain_of_links_bytes(code, mask_code, links, at_start, answers, later_stages, masks_ride):
    """R12.12 to R12.14: a chain of segments. links: its segments, a list of (plain, [messages]); a
    message: (main rows, 1 or 3, mask rows, leaf class, moment, life)."""
    out = u64(code) + u64(mask_code) + u64(len(links))
    for plain, messages in links:
        out += u64(plain) + u64(len(messages)) + b"".join(b"".join(u64(x) for x in m) for m in messages)
    out += u64(len(at_start)) + b"".join(notice_bytes(*n) for n in at_start)
    return out + u64(answers) + stages_bytes(later_stages) + choices_bytes(masks_ride)


def shape_bytes(code, mask_code, main_rows, degree, mask_rows, answers):
    """An oracle message's shape: a notice (R12.3) without its last word, the leaf class."""
    return b"".join(u64(x) for x in (code, mask_code, main_rows, degree, mask_rows, answers))


def table_shape_bytes(a, message, tail, log_rows, answers, salt_bytes):
    return b"".join(u64(x) for x in (a, message, tail, log_rows, answers, salt_bytes))


def link_bytes(table, nu_0, nu_j, code, mask_code, rows, rows_high, leaf_class, masks_ahead, later_stages, masks_ride):
    """R13.9."""
    return table_shape_bytes(*table) + b"".join(u64(x) for x in (nu_0, nu_j, code, mask_code, rows, rows_high, leaf_class, masks_ahead)) + stages_bytes(later_stages) + choices_bytes(masks_ride)


def test_on_the_table_bytes(table, folded, folded_high, folded_class, columns, k_lo, matrix_rows, omega, w, omega_high, w_high, shared_split, own_split, classes,
                            shared_mask_code, own_mask_code, masks_ahead, tests, masks_ride):
    """R13.13. folded, omega, w, shared_split, own_split: shapes of six words. classes: four leaf
    classes. tests: four lists of stages."""
    out = table_shape_bytes(*table) + shape_bytes(*folded) + u64(folded_high) + u64(folded_class) + u64(columns) + u64(k_lo) + u64(matrix_rows)
    out += shape_bytes(*omega) + shape_bytes(*w) + u64(omega_high) + u64(w_high) + shape_bytes(*shared_split) + shape_bytes(*own_split)
    out += b"".join(u64(x) for x in classes) + u64(shared_mask_code) + u64(own_mask_code) + u64(masks_ahead)
    return out + b"".join(stages_bytes(t) for t in tests) + choices_bytes(masks_ride)


def _words(xs):
    return u64(len(xs)) + b"".join(u64(x) for x in xs)


def ranges_bytes(lookups, notice, number, stages, masks_ride):
    """R13.14. A lookup: (d, mu, live, digit weights, offset, padding key, parts, first bounds); a
    part: (the logarithm of the number of its cells, runs); a run: four words."""
    out = u64(len(lookups))
    for d, mu, live, weights, offset, padding, parts, bounds in lookups:
        out += u64(d) + u64(mu) + u64(live) + _words(weights) + u64(offset) + u64(padding) + u64(len(parts))
        for log_places, runs in parts:
            out += u64(log_places) + u64(len(runs)) + b"".join(b"".join(u64(x) for x in r) for r in runs)
        out += _words(bounds)
    return out + notice_bytes(*notice) + u64(number) + stages_bytes(stages) + choices_bytes(masks_ride)


def psc_rounds(numbers, degree_y=2, values=2, with_y=True):
    """R13.16. numbers: g for each variable. Rounds as (name, number of bytes of the prover's
    message, elements sampled, of them not 0)."""
    out = [("PSC epsilon", 24, 1, 0)] + [("PSC round", 24 * (g - 1), 1, 0) for g in numbers]
    return out + ([("PSC Y", 24 * degree_y, 1, 1), ("PSC values", 24 * values, 0, 0)] if with_y else [])


def product_tree_rounds(levels, more):
    return [r for l in range(levels) for r in psc_rounds([3] * (l + more)) + [("LINE", 0, 1, 0)]]


def lookup_rounds(d, mu, live, part_logs, k_col):
    """R13.17. part_logs: each part's word lp."""
    out = [("LKP", 0, 1, 0)]
    for j in range((live or 1 << mu) - 1):
        out += psc_rounds([], 3, 2 if j else 3)
    for _ in part_logs[1:]:
        out += psc_rounds([])
    out.append(("JOIN", 0, mu, 0))
    for lp in part_logs:
        out += product_tree_rounds(lp + k_col, 0)
    out += product_tree_rounds(d, mu) + psc_rounds([3] * d + [2] * mu, with_y=False) + [("ZERO", 0, mu + d, 0)]
    return out + psc_rounds([3] * (mu + d), 2, 1)


def test_rounds(stages, first_mask_has_a_tree, masks_ride, answers=1):
    """R13.19. None: the number depends on sizes that this function is not given."""
    assert len(stages) >= 2
    out = [("BAT", 0, 1, 0)]
    for i, s in enumerate(stages[:-1]):
        root = first_mask_has_a_tree if i == 0 else not masks_ride
        out += [("FOLD epsilon", 56 if root else 24, 1, 0)] + [("FOLD round", 48, 1, 0)] * s[2] + [("SEND seed", 32, None, answers), ("CS", 24 * answers, 1, 0)]
    return out + [("BASE lambda", 24 if masks_ride else 56, 1, 0), ("BASE", None, 0, 0)]


def key_rounds(lookups, ranges_stages, ranges_ride, v, k_lo, matrix_rows, split, masks_ahead, tests, masks_ride, answers=1):
    """R13.18, R13.19. lookups: (d, mu, live, part_logs). ranges_ride: the choices' word of the item
    "ranges". split: the folded row's and Omega's base codes differ. masks_ahead, tests (four
    lists), masks_ride: of the item "the test on the table". answers: the plan's sixth word.
    Returns the three chains' rounds."""
    send = lambda messages: [("SEND seed", 32 * messages, None, answers), ("SEND answer", 24 * answers * messages, 0, 0)]  # noqa: E731
    one = send(1) + send(1)
    for d, mu, live, part_logs in lookups:
        one += lookup_rounds(d, mu, live, part_logs, k_lo + v)
    one += test_rounds(ranges_stages, not ranges_ride, ranges_ride, answers)
    if sum(len(part_logs) for _, _, _, part_logs in lookups) > 1:
        one += [("BAT-sigma", 0, 1, 0)] + psc_rounds([3] * (k_lo + v), with_y=False)
    one += send(1) + [("SEAL", 0, 0, 0)]
    shared = [("LIN point", 0, v + k_lo, 0)] + send(1) + [("LOC", 0, 0, 0)] + send(2 if split else 1) + [("LIN zeta", 0, _log_up(matrix_rows), 0)]
    if split:
        shared += [("SPLIT", 24, 0, 0)] + test_rounds(tests[0], not masks_ahead, masks_ride, answers) + test_rounds(tests[1], True, masks_ride, answers)
    else:
        shared += test_rounds(tests[0], not masks_ahead, masks_ride, answers)
    return one, shared + [("SEAL", 0, 0, 0)], test_rounds(tests[2], not masks_ahead, masks_ride, answers)


def layout_bytes(links):
    """R14.5. links: for each segment a list of arrays (name, message, rows, columns, [blocks]); a
    block: nine words."""
    out = u64(len(links))
    for arrays in links:
        out += u64(len(arrays))
        for name, message, rows, columns, blocks in arrays:
            assert all(len(b) == 9 for b in blocks)
            out += text(name) + u64(message) + u64(rows) + u64(columns) + u64(len(blocks)) + b"".join(b"".join(u64(x) for x in b) for b in blocks)
    return out


def proof_head(copy):
    """R14.2."""
    return b"CPPF" + b"H821" + struct.pack("<I", copy)


def proof_statement(root, answers, copy, request, file_words):
    """R14.3: the statement of a proof's first chain. answers: the bytes of the record's answers
    (R4.6). file_words: the caller's request itself, byte for byte."""
    return (item("table", string(root) + u64(len(answers) // 24) + answers) + item("stored copy", u64(copy)) + item("request", string(request))
            + item("the request file's words", string(file_words)))


def first_chain_tail(pending_rows, k_col, answers=1):
    """R14.4: the rounds of a proof's first chain after its last proximity test."""
    out = [("BAT-sigma", 0, 1, 0)] + psc_rounds([3] * k_col, with_y=False) if pending_rows > 1 else []
    return out + [("SEND seed", 32, None, answers), ("SEND answer", 24 * answers, 0, 0), ("SEAL", 0, 0, 0)]


# ---- R15.x: the second hash function, and the item "request" ----
# BLAKE2b's compression function, for the hash K of this section alone, which puts words of its own into the function's counter:
# hashlib has no argument for the counter. RFC 7693, section 3.2, specifies the function. The text from rotr64 to the end of
# blake2b_F is adapted from the BLAKE2 team's reference implementation in C. The head of its file blake2b-ref.c, which its two
# headers have too:
# /*
#    BLAKE2 reference source code package - reference C implementations
#
#    Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
#    terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
#    your option.  The terms of these licenses can be found at:
#
#    - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
#    - OpenSSL license   : https://www.openssl.org/source/license.html
#    - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
#
#    More information about the BLAKE2 hash function can be found at
#    https://blake2.net.
# */
# Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
# source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
# It is a translation into Python, under the reference's names: rotr64, blake2b_IV, blake2b_sigma, blake2b_init_param with what
# blake2b_init does before it calls it, the macros G and ROUND, and blake2b_compress, which is blake2b_F here.
# What differs from the reference:
# - A Python integer has no fixed width, so every sum and every rotation is cut to 64 bits with MASK.
# - G and ROUND are functions, not macros: G also takes m and returns the four new words, and ROUND also takes v and m and
#   writes into v. The twelve lines ROUND( 0 ) to ROUND( 11 ) are a loop.
# - blake2b_F has no state structure S, no buffer and no load64. What the reference keeps in S, and the block, are arguments:
#   h is S->h; t is the whole counter, S->t[0] + 2^64 S->t[1]; last is whether the last-block flag S->f[0] is set; m is the
#   block as sixteen words. The arguments are not written into, and the new chaining value is returned.
# - In the reference S->t[1] is the upper half of a count of bytes; here older_K puts num + 2^32 L there. S->f[1], which marks
#   a last node in tree hashing, is always 0 here.
# - blake2b_init_param takes the two lengths, not a filled parameter block, and returns S->h.


def rotr64(w, c):
    return (w >> c | w << (64 - c)) & MASK


blake2b_IV = (
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
)  # fmt: skip

blake2b_sigma = (
    ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15),
    (14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3),
    (11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4),
    ( 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8),
    ( 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13),
    ( 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9),
    (12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11),
    (13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10),
    ( 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5),
    (10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0),
    ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15),
    (14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3),
)  # fmt: skip


def blake2b_init_param(outlen, keylen):
    """Returns the initial chaining value, the reference's S->h after blake2b_init (keylen = 0) or
    blake2b_init_key: blake2b_IV XOR the parameter block P, read as little-endian words (load64)."""
    # digest_length, key_length, fanout, depth; all later fields are 0
    P = bytes([outlen, keylen, 1, 1]) + bytes(60)
    return [blake2b_IV[i] ^ int.from_bytes(P[8 * i : 8 * i + 8], "little") for i in range(8)]


def G(r, i, a, b, c, d, m):
    a = (a + b + m[blake2b_sigma[r][2 * i + 0]]) & MASK
    d = rotr64(d ^ a, 32)
    c = (c + d) & MASK
    b = rotr64(b ^ c, 24)
    a = (a + b + m[blake2b_sigma[r][2 * i + 1]]) & MASK
    d = rotr64(d ^ a, 16)
    c = (c + d) & MASK
    b = rotr64(b ^ c, 63)
    return a, b, c, d


def ROUND(r, v, m):
    v[0], v[4], v[8], v[12] = G(r, 0, v[0], v[4], v[8], v[12], m)
    v[1], v[5], v[9], v[13] = G(r, 1, v[1], v[5], v[9], v[13], m)
    v[2], v[6], v[10], v[14] = G(r, 2, v[2], v[6], v[10], v[14], m)
    v[3], v[7], v[11], v[15] = G(r, 3, v[3], v[7], v[11], v[15], m)
    v[0], v[5], v[10], v[15] = G(r, 4, v[0], v[5], v[10], v[15], m)
    v[1], v[6], v[11], v[12] = G(r, 5, v[1], v[6], v[11], v[12], m)
    v[2], v[7], v[8], v[13] = G(r, 6, v[2], v[7], v[8], v[13], m)
    v[3], v[4], v[9], v[14] = G(r, 7, v[3], v[4], v[9], v[14], m)


def blake2b_F(h, m, t, last):
    """RFC 7693, 3.2: BLAKE2b's compression function; t has 128 bits."""
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]
    v[8] = blake2b_IV[0]
    v[9] = blake2b_IV[1]
    v[10] = blake2b_IV[2]
    v[11] = blake2b_IV[3]
    v[12] = blake2b_IV[4] ^ (t & MASK)  # S->t[0]
    v[13] = blake2b_IV[5] ^ (t >> 64)  # S->t[1]

    # S->f[0]: blake2b_set_lastblock sets it to (uint64_t)-1
    v[14] = blake2b_IV[6] ^ (MASK if last else 0)
    v[15] = blake2b_IV[7]  # S->f[1] is 0
    for r in range(12):
        ROUND(r, v, m)
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def older_K(num, w):
    """R15.3: the second hash function, of a list of 32-bit words under a number: eight words."""
    L = len(w)
    blocks = max(1, -(-L // 32))
    h = blake2b_init_param(64, 0)
    for b in range(blocks):
        blk = list(w[32 * b:32 * b + 32])
        blk += [0] * (32 - len(blk))
        m = [blk[2 * i] | blk[2 * i + 1] << 32 for i in range(16)]
        fed = 128 * (b + 1) if b < blocks - 1 else 4 * L
        h = blake2b_F(h, m, fed | (num | L << 32) << 64, b == blocks - 1)
    return [x for y in h[:4] for x in (y & 0xFFFFFFFF, y >> 32)]


def older_TK(w):
    """R15.3."""
    assert len(w) >= 8
    num = 10
    while True:
        w = [x for i in range(0, len(w), 128) for x in older_K(num, w[i:i + 128])]
        if len(w) == 8:
            return w
        num = 11


def words32(b):
    assert len(b) % 4 == 0
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def text_words(s):
    """R15.4."""
    b = s.encode("ascii")
    return words32(b + bytes(-len(b) % 4))


CALLH_IF_NONE = "callH R16a R16 NF NFH HD0\ncallH R16b R16 HD1\ncallH R8 R8 ROW HD2\ncallH FEa FE HE0\ncallH FEb FE HE1\ncallH FC FC CNF\n"  # R15.6


def statement_words(parameter_file, scale_file, request_file):
    """R15.5, R15.6, R15.7: three byte strings in, the item's 708 bytes out."""
    lines = [x.split(" ") for x in parameter_file.decode("ascii").split("\n") if x]
    first = {}
    for x in lines:
        first.setdefault(x[0], x[1:])
    val = lambda name: int(first[name][0])  # noqa: E731
    vals = lambda names: [val(n) for n in names.split()]  # noqa: E731
    text_of = lambda word: "".join(" ".join(x) + "\n" for x in lines if x[0] == word)  # noqa: E731
    calls = lambda word: older_K(73, text_words(text_of(word) or {"callH": CALLH_IF_NONE}[word]))  # noqa: E731
    eight = lambda name: [val("%s%d" % (name, i)) for i in range(8)]  # noqa: E731
    assert val("prof") == 20 and val("NE") != 0
    w = [0x4D363100] + text_words(first["version"][0])
    w += vals("prof lt L ll d H G dh f fe NE V eps1_lo eps1_hi eps2_lo eps2_hi cs theta0_lo theta0_hi hq hv ho hgD huD hdD hr ut")
    w += calls("call")
    w += vals("la nmin D tau lmaf NAf lmad NAd lmar lmb lmq lmx wb_qk wb_v wb_o wb_g wb_u wb_gs wb_us wb_gr wb_ur wb_dn wb_dnr wb_r")
    w += vals("iota_0 iota_1 iota_2 iota_3 iota_4 iota_5 iota_6 iota_7 seed_lo seed_hi")
    w += eight("tsilu") + eight("tsig") + eight("trope") + older_TK(words32(scale_file))
    w += vals("lB S rx kFM kFD lmS NAs lmh lmo lmbA iotaA_0") + eight("texp") + calls("callA")
    w += vals("MF KF epsF_lo epsF_hi toksrc lmhf NHf lmhd lmhr lmbH wb_emb wb_head iotaH_0 iotaH_1 iotaH_2 iotaH_3 iotaH_4 iotaH_5 C theta cap s sh sT F")
    request = words32(request_file)
    assert len(request) == (1 << val("lt")) + 2
    w += calls("callH") + older_TK(request) + [0]
    b, n, i, m, e, shared = (int(x) for x in first["table"])
    w += [val("sec"), shared, b, n, m]
    assert len(w) == 177
    return struct.pack("<177I", *w)


def rounds_text(rounds):
    return "".join("%s|%s|%s|%d\n" % (n, "-" if b is None else b, "-" if e is None else e, z) for n, b, e, z in rounds).encode()


def _log_up(x):
    return 0 if x <= 1 else (x - 1).bit_length()


def _block(rows):
    return (1 << _log_up(rows)) if rows else 0


def link_groups(links):
    """R12.16. For the groups 1 .. L + 1: (k, h, number of oracle messages in the group's tree that
    get an out-of-domain sample, width of the tree's columns in elements of F)."""
    out, alive = [], []
    for l in range(len(links) + 1):
        main = mask = sampled = wide = mask_rows = 0
        if l > 0:  # the carrying message: two rows over E
            main, sampled, wide = 2, 1, 6
        still = []
        for rows, degree, left in alive:  # a kept message is sent again in this group's tree
            sampled, wide = sampled + 1, wide + rows * degree
            if left > 1:
                still.append((rows, degree, left - 1))
            else:
                main += _block(rows)
        alive = still
        for m in links[l][1] if l < len(links) else []:
            sampled, wide, mask_rows = sampled + 1, wide + m[0] * m[1], mask_rows + m[2]
            if m[5] > 0:
                alive.append((m[0], m[1], m[5]))
            else:
                main += _block(m[0])
            mask += _block(m[2])
        h = _log_up(mask)
        out.append((max(_log_up(main), h, 1), h, sampled, wide + 3 * mask_rows + 3 * (1 << h)))
    return out


def links_rounds(variables, links, answers, later_stages):
    """R12.17. The rounds of the example protocol "links": (name, kind, number of bytes of the
    prover's message, number of roots); None for the last message's number of bytes."""
    groups, out = link_groups(links), []
    fold = lambda k: [("FOLD epsilon", 2, 24, 0)] + [("FOLD round", 1, 48, 0)] * k  # noqa: E731
    for l in range(len(links)):
        out.append(("SEND seed", 3, 32, 1))
        out.append(("SEND answer", 7, 24 * answers * groups[0][2], 0) if l == 0 else ("LOC", 4, 24 * answers * groups[l][2], 0))
        out += [("ZERO", 1, 0, 0), ("PSC epsilon", 2, 24, 0)] + [("PSC round", 1, 48, 0)] * variables
        out += [("PSC Y", 1, 48, 0), ("PSC values", 7, 48, 0), ("BAT", 1, 0, 0)] + fold(groups[l][0])
    out += [("SEND seed", 3, 32, 1), ("LOC", 4, 24 * answers * groups[-1][2], 0), ("BAT", 1, 0, 0)]
    for k in ([groups[-1][0]] + [s[2] for s in later_stages])[:-1]:
        out += fold(k) + [("SEND seed", 3, 32, 1), ("CS", 4, 24 * answers, 0)]
    return out + [("BASE lambda", 1, 24, 0), ("BASE", 4, None, 0)]


ROOT_32 = pow(7, 5 * ((1 << 32) - 1), PRIME[2])  # of order 2^32


def point(a, x):
    """R12.11, field 2: the point of position x of a code of length 2^a, a <= 32."""
    assert a <= 32 and 0 <= x < 1 << a
    z = 7
    for i in range(a):
        if x >> i & 1:
            z = z * pow(ROOT_32, 1 << (31 - i), PRIME[2]) % PRIME[2]
    return z


def symbol(a, message, tail, x):
    z, r = point(a, x), len(tail)
    return (sum(t * pow(z, j, PRIME[2]) for j, t in enumerate(tail)) + sum(m * pow(z, r + j, PRIME[2]) for j, m in enumerate(message))) % PRIME[2]


# ---- the known answers ----
def pattern(n, a=7, b=3):
    return bytes((a * i + b) % 251 for i in range(n))


def toy_run():
    """The values of the toy run that makes a key file and a leaf file (blocks/frame/handover_files.cpp
    with blocks/frame/tests/toy_file.h), typed by hand: three items as bytes, the key file's rounds by
    key_rounds(), and three of the values that key_rounds() takes.
    tests/new/second/vectors_against_the_toy_files.py compares the items and the rounds with the
    files that the tool writes."""
    table = (14, 803, 1001, 3, 1, 40)
    lookups = [(3, 4, 14, [5, 5, 5, 5]), (5, 4, 11, [4])]
    tests = [[(0, 0, 2, 0), (1, 1, 2, 1), (2, 2, 0, 0)], [(3, 0, 1, 0), (4, 1, 2, 1), (5, 2, 0, 0)], [(3, 3, 2, 2), (6, 4, 2, 1), (7, 5, 0, 0)], []]
    ranges_test = [(8, 6, 3, 3), (9, 7, 2, 1), (10, 8, 0, 0)]
    four = [(5, [(0, 16 * i, 16 if i < 3 else 2, 0), (16, 50 + 16 * i, 16 if i < 3 else 2, 0)]) for i in range(4)]
    return {
        "the link": link_bytes(table, 76, 76, 0, 0, 4, 1, 0, 1, [(1, 1, 2, 1), (2, 2, 0, 0)], 1),
        "the test on the table": test_on_the_table_bytes(table, (0, NO_CODE, 4, 3, 0, 1), 1, 0, 76, 3, 100, (3, NO_CODE, 1, 3, 0, 1), (3, NO_CODE, 1, 3, 0, 1), 1, 1,
                                                         (NO_CODE, 0, 0, 1, 1, 1), (NO_CODE, NO_CODE, 0, 1, 0, 1), (1, 2, 0, 3), 0, 3, 1, tests, 1),
        "ranges": ranges_bytes([(3, 4, 14, [1], 4, 3, four, [1, 0]), (5, 4, 11, [1, 8], 16, 17, [(4, [(0, 0, 10, 50)])], [])], (8, 6, 1, 1, 5, 1, 0), 1, ranges_test, 1),
        "the table": table,
        "lookups": lookups,
        "the stages of ranges": ranges_test,
        "the four lists": tests,
        "the key file's rounds": key_rounds(lookups, ranges_test, 1, 3, 3, 100, True, 1, tests, 1),
    }


def main():
    for field, c in ((1, 5), (2, 2)):  # R1.4: X^3 - c is irreducible
        assert (PRIME[field] - 1) % 3 == 0 and pow(c, (PRIME[field] - 1) // 3, PRIME[field]) != 1
    say = lambda name, value: print("%s = %s" % (name, value.hex() if isinstance(value, bytes) else value))  # noqa: E731
    print("# Known answers of FORMAT.md, rules R1.x to R8.x and R12.x to R15.x, written by format_rules.py. pattern(n): byte i is (7 i + 3) mod 251; pattern(n) with a i + b: byte i is (a i + b) mod 251.")
    print("# The names below, and one heading, use other words than FORMAT.md, because the programs' tests ask for them letter by letter. drawn: sampled; behind: after; bits of work: proof-of-work bits; three links, the first link: segments; look-up: lookup; the published ID: the commitment ID; the stored copy, the copy: the leaf table; section 7's columns and salts: those of the tree under the heading R7.x above it; sections 5 and 6 of the rules: sections 4 and 5 of FORMAT.md.")
    print("# R2.x and R3.x")
    say("the oracle's string of the leaf H(164, 9, 2^33 + 12, start = pattern(32); one word 1, a salt of 40 zero bytes), W = 1, s8 = 5", oracle_string(0, 164, 9, (1 << 33) + 12, pattern(32), u64(1) + bytes(40), 32, 1, 5))
    say("the oracle's string of the second half of H(162, 0, 3, start = pattern(32); pattern(4))", oracle_string(1, 162, 0, 3, pattern(32), pattern(4), 64))
    say("H(168, 0, 0, none; no byte)", H(168, 0, 0, None, b""))
    say("H(168, 0, 5, none; pattern(128))", H(168, 0, 5, None, pattern(128)))
    say("H(168, 0, 2^33 + 5, none; pattern(129))", H(168, 0, (1 << 33) + 5, None, pattern(129)))
    say("H(164, 9, 12, start = pattern(32); pattern(64)), W = 3, s8 = 5", H(164, 9, 12, pattern(32), pattern(64), 32, 3, 5))
    say("a vertex: H(166, 0, 0, start = pattern(32); pattern(64))", H(166, 0, 0, pattern(32), pattern(64)))
    say("a vertex of the binding tree: H(178, 0, 0, start = pattern(32); pattern(64))", H(178, 0, 0, pattern(32), pattern(64)))
    say("H(162, 2^32, 3, start = pattern(32); the message pattern(8), kind 4, w = 8, j = 1, nonce 5), 64 bytes", H(162, 1 << 32, 3, pattern(32), pattern(8) + u64(4 + (8 << 8) + (1 << 16)) + u64(5), 64))
    say("H(162, 0, 3, start = pattern(32); no message, kind 1, w = 0, j = 0, nonce 0), 64 bytes", H(162, 0, 3, pattern(32), u64(1) + u64(0), 64))
    rounds = [("toy: challenge", 1, 0, 48, 0, 0, 1, 0, 0, 0, 0, 0), ("toy: nothing", 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0),
              ("toy: positions", 1, 8, 32, 1, 0, 1, 0, 5, 10, 0, 3), ("toy: seed", 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0),
              ("toy: matrix", 1, 0, 24, 0, 1, 0, 0, 0, 0, 40, 0), ("toy: seal", 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0)]
    for field in (2, 1):
        print("# R4.x: a toy plan of six rounds, field %d" % field)
        parameters = parameters_head(field, plan_bytes(field, 40, 8, 0, 1, 1, rounds)) + item("configuration", configuration_digest([b"toy\n"])) + item("toy: a table's length", u64(1024))
        if field == 2:
            say("the parameters' bytes", parameters)
        pd = parameters_digest(parameters)
        say("the parameters' digest", pd)
        statement = item("toy: claim", ext((1, 2, 3))) + item("toy: name", text("abc"))
        if field == 2:
            say("the statement's bytes", statement)
        run = run_value(7, pd, pattern(32, 5, 1))
        say("the run's value (label 7, random bytes pattern(32) with 5 i + 1)", run)
        c = Chain(first_value(7, pd, statement, run), field, 7)
        say("rho_0", c.rho)
        say("rho_0 of the same chain without random bytes", first_value(7, pd, statement, None))
        print("# sections 5 and 6 of the rules: R5.x and R6.x")
        assert c.query(ext((4, 5, 6)) + ext((7, 8, 9)), ALGEBRAIC)
        say("round 1, kind 1, message (4, 5, 6), (7, 8, 9): rho_1", c.rho)
        say("round 1: the element drawn", " ".join(map(str, c.element())))
        assert c.query(ext((1, 1, 1)), EMPTY)
        say("round 2, kind 7, message (1, 1, 1): rho_2", c.rho)
        root = pattern(32, 11, 2)
        nonce = c.least_nonce(root, POSITIONS, 8)
        say("round 3, kind 4, message one root, pattern(32) with 11 i + 2, 8 bits of work: the least nonce", nonce)
        assert c.query(root, POSITIONS, 8, nonce)
        say("round 3: block 0, the test block", c.block(0))
        say("round 3: rho_3", c.rho)
        say("round 3: five positions in [0, 2^10), as drawn", " ".join(str(c.position(10)) for _ in range(5)))
        say("round 3: the element behind them", " ".join(map(str, c.element())))
        assert c.query(b"", SEED)
        say("round 4, kind 3, empty message: rho_4", c.rho)
        say("round 4: a seed with a point of two coordinates", "; ".join(" ".join(map(str, e)) for e in (c.element(), c.element(), c.element(True))))
        assert c.query(ext((9, 9, 9)), BITS)
        say("round 5, kind 5, message the answer (9, 9, 9): rho_5", c.rho)
        say("round 5: forty entries", " ".join(map(str, c.trits(40))))
        assert c.query(u64(77), SEAL)
        say("round 6, kind 6, message the word 77: rho_6", c.rho)
        say("round 6: the seal", c.seal_digest())
        z, beta = record_seeds(table_name(pattern(32, 13, 4), configuration_digest([b"toy\n"]), 10, 3, 40, 9, 1, 40), 2, 1, field)[0]
        say("record seed of the name with root pattern(32) with 13 i + 4, the configuration of one file 'toy' and a line feed, a = 10, 3 entries, tail 40, 2^9 rows, 1 answer, salt 40; two coordinates: the point", "; ".join(" ".join(map(str, e)) for e in z))
        say("the same: the factor", " ".join(map(str, beta)))
    print("# R1.x: eight bytes as the encoding of an element of F (R1.3; - : rejected)")
    for field, ws in ((2, (0, PRIME[2] - 1, PRIME[2], MASK)), (1, (0, PRIME[1] - 1, PRIME[1], 1 << 61, MASK))):
        for w in ws:
            say("field %d, the bytes %s" % (field, u64(w).hex()), w if w < PRIME[field] else "-")
    print("# R6.x: a word and its attempt (- : the attempt fails)")
    for field, ws in ((2, (0, PRIME[2] - 1, PRIME[2], MASK)), (1, (0, PRIME[1] - 1, PRIME[1], 1 << 61, MASK, MASK - 1))):
        for w in ws:
            x = element_of_word(w, field)
            say("field %d, word %d" % (field, w), "-" if x is None else x)
    print("# R6.x: an element of F from a given sequence of words: the element; words used")
    bad = {2: MASK, 1: PRIME[1]}
    for field in (2, 1):
        for fails in (0, 1, 5, 6, 7):
            say("field %d, %d words that fail, then the word 5" % (field, fails), "%d; %d" % base_from_words([bad[field]] * fails + [5], field))
    print("# R6.x: the most words of a verifier's message")
    say("one element of E; one that is not 0; a seed with two coordinates; five positions and an element; forty entries; a seal",
        "; ".join(str(x) for x in (words_at_most(1, 0, 0, [], 0), words_at_most(1, 1, 0, [], 0), words_at_most(3, 1, 0, [], 0),
                                   words_at_most(1, 0, 5, [], 0), words_at_most(0, 0, 0, [40], 0), words_at_most(0, 0, 0, [], 1))))
    print("# R7.x: a leaf's bytes (salt: pattern(40) with 3 i + 9; start: the synthetic digest pattern(32) with 5 i + 1)")
    salt, run = pattern(40, 3, 9), pattern(32, 5, 1)
    for count in (3, 9, 10, 11, 14, 16, 25, 26):
        data = leaf_bytes(list(range(1, count + 1)), salt)
        say("column 1 .. %d: bytes; the byte at which the salt begins" % count, "%d; %d" % (len(data), len(data) - 40))
        say("  its leaf under 164: tree 5, a = 3, position 6", leaf(164, 5, 3, 6, run, list(range(1, count + 1)), salt))
    say("two shapes that give one string of bytes and differ in the hash's label alone: columns 1 .. 10 and 1 .. 10, 0: equal digests", "yes" if leaf(164, 5, 3, 6, run, list(range(1, 11)), salt) == leaf(164, 5, 3, 6, run, list(range(1, 11)) + [0], salt) else "no")
    say("a binding leaf: depth 2, place 3, that start as the configuration's digest, seal pattern(32), that salt", binding_leaf(2, 3, run, pattern(32), salt))
    say("the leaf of the column 1 .. 3 under 174, tree 0 (R7.4), the same position, with that start in the place of the configuration's digest", leaf(174, 0, 3, 6, run, [1, 2, 3], salt))
    print("# R7.x: a tree with two main rows of length 2^3 and one mask row of length 2^1, number 5")
    a, b = 3, 1
    column = lambda x: [100 + x, 200 + x, 300 + short_position(x, b)]  # noqa: E731
    salt_of = lambda x: pattern(40, 3, 9 + x)  # noqa: E731
    leaves = [leaf(164, 5, a, x, run, column(x), salt_of(x)) for x in range(1 << a)]
    conf = configuration_digest([b"toy\n"])
    say("the configuration's digest of one file, the four bytes 'toy' and a line feed", conf)
    value = build(a, conf, leaves)
    say("column of x: 100 + x, 200 + x, 300 + (x mod 2); salt pattern(40) with 3 i + 9 + x: the root", value[0][0])
    for drawn in ([6, 1, 6, 3], [5], [0, 4, 2, 6], [2, 3]):
        xs, order = sorted(set(drawn)), []

        def given(d, i):
            order.append((d, i))
            return value[d][i]

        assert walk(a, conf, xs, [leaves[x] for x in xs], given) == value[0][0]
        say("drawn %s: the given digests, as (depth, place)" % drawn, " ".join("(%d, %d)" % o for o in order))
        body = b"".join(b"".join(u64(s) for s in column(x)) + salt_of(x) for x in xs) + b"".join(value[d][i] for d, i in order)
        say("  the opening: bytes; H(168, 0, 0, none; the opening)", "%d; %s" % (len(body), H(168, 0, 0, None, body).hex()))

    print("# R7.x: which oracle messages share a tree, and the trees' numbers (R7.12, R7.13); a notice: (leaf class, a, b), - for no such rows")
    five = [(1, 4, None), (1, None, 2), (1, 3, None), (2, None, 2), (1, 4, 2)]
    say("(1, 4, -) (1, -, 2) (1, 3, -) (2, -, 2) (1, 4, 2): the trees, each as its parts", " ".join(str(t) for t in lay_trees(five)))
    say("(3, -, 2) (3, 5, 2) (3, -, 2): the trees", " ".join(str(t) for t in lay_trees([(3, None, 2), (3, 5, 2), (3, None, 2)])))
    say("(1, -, 2) (2, -, 2) (1, -, 2): the trees", " ".join(str(t) for t in lay_trees([(1, None, 2), (2, None, 2), (1, None, 2)])))
    say("a chain with the first case, then (1, 4, -), then the third case: the numbers of each message's trees", " | ".join(" ".join(map(str, n)) for n in number_trees([five, [(1, 4, None)], [(1, None, 2), (2, None, 2), (1, None, 2)]])))
    say("a chain that continues it, with (1, 4, -): its tree's number", number_trees([[(1, 4, None)]], 6)[0][0])
    say("a chain that continues it, adopts a tree of another chain and then sends (1, 4, -): the adopted tree's place; the new tree's number", "%d; %d" % tuple(x[0] for x in number_trees(["adopt", [(1, 4, None)]], 6)))
    print("# R7.x: one value at a vertex (R7.14), on the tree above")
    session, verdicts = Session(), []
    record = session.record(value[0][0])
    for xs in ([1, 3, 6], [3, 5]):
        walk(a, conf, xs, [leaves[x] for x in xs], lambda d, i: value[d][i], lambda d, i, x: verdicts.append(record.enter(d, i, x)))
    say("the honest openings at 1 3 6 and then at 3 5: values entered, computed or given; all admitted", "%d; %s" % (len(verdicts), "yes" if all(verdicts) else "no"))
    say("vertices in the record", len(record.seen))
    say("then the vertex (2, 1), which the first opening computed, given as 32 zero bytes: admitted", "yes" if record.enter(2, 1, bytes(32)) else "no")
    say("the same value at the same vertex of another tree, whose root is the same 32 bytes: admitted", "yes" if session.record(bytes(value[0][0])).enter(2, 1, bytes(32)) else "no")
    say("the same value at the same vertex under another root: admitted", "yes" if session.record(pattern(32, 5, 9)).enter(2, 1, bytes(32)) else "no")
    print("# R4.x: a stored table's name")
    name = table_name(pattern(32, 13, 4), conf, 10, 512, 40, 9, 1, 40)
    say("root pattern(32) with 13 i + 4, the configuration above, a = 10, 512 entries, tail 40, 2^9 rows, 1 answer, salt 40", name)
    three = lambda *x: b"".join(u64(w) for w in x)
    say("the published ID of that configuration and that name with one seed, the point (7, 8, 9), (10, 11, 12) and the factor (13, 14, 15), and the two answers (1, 2, 3) and (4, 5, 6): a test of the coding alone: that name's own seed has 9 coordinates, its table 512 answers", published_id(conf, name, [([three(7, 8, 9), three(10, 11, 12)], three(13, 14, 15))], [three(1, 2, 3), three(4, 5, 6)]))
    say("the same with that name's own seed (R4.8; field 2: 9 coordinates and the factor), sampled and not given, and the two answers (1, 2, 3) and (4, 5, 6)", published_id(conf, name, [([ext(x) for x in z], ext(beta)) for z, beta in record_seeds(name, 9, 1, 2)], [three(1, 2, 3), three(4, 5, 6)]))
    say("the item named 'table' of a table with that root and the one answer (1, 2, 3)", item("table", string(pattern(32, 13, 4)) + u64(1) + three(1, 2, 3)))
    for field in (2, 1):
        print("# R8.x: a whole chain in a file, field %d" % field)
        plan = [("toy file: seed", 1, 0, 32, 1, 0, 3, 1, 0, 0, 0, 0), ("toy file: answer", 1, 0, 24, 0, 1, 0, 0, 0, 0, 0, 0),
                ("toy file: positions", 1, 8, 24, 0, 0, 0, 0, 4, 3, 0, 0)]
        pd = parameters_digest(parameters_head(field, plan_bytes(field, 40, 8, 0, 1, 1, plan)) + item("configuration", conf))
        random_bytes = pattern(32, 5, 1)
        run = run_value(9, pd, random_bytes)
        leaves = [leaf(164, 0, a, x, run, column(x), salt_of(x)) for x in range(1 << a)]
        value = build(a, conf, leaves)
        c = Chain(first_value(9, pd, b"", run), field, 9)
        assert c.query(value[0][0], SEED)
        c.element(), c.element(), c.element(True)
        assert c.query(ext((9, 9, 9)), EMPTY)
        nonce = c.least_nonce(ext((4, 5, 6)), POSITIONS, 8)
        assert c.query(ext((4, 5, 6)), POSITIONS, 8, nonce)
        draws = [c.position(3) for _ in range(4)]
        xs, order = sorted(set(draws)), []

        def given_in_file(d, i):
            order.append((d, i))
            return value[d][i]

        assert walk(a, conf, xs, [leaves[x] for x in xs], given_in_file) == value[0][0]
        opening = b"".join(b"".join(u64(s) for s in column(x)) + salt_of(x) for x in xs) + b"".join(value[d][i] for d, i in order)
        whole = random_bytes + value[0][0] + ext((9, 9, 9)) + ext((4, 5, 6)) + u64(nonce) + opening
        say("label 9, empty statement, parameters: version, a plan of three rounds (kinds 3, 7, 4 with 8 bits of work and four positions in [0, 2^3)), configuration: the parameters' digest", pd)
        say("the tree of section 7's columns and salts, number 0, under this chain's run's value: the root", value[0][0])
        say("the nonce; the draws; the given digests", "%d; %s; %s" % (nonce, " ".join(map(str, draws)), " ".join("(%d, %d)" % o for o in order)))
        say("the file: random bytes | root | answer (9, 9, 9) | element (4, 5, 6) | nonce | opening: bytes", len(whole))
        say("the file", whole)
    print("# R12.x: synthetic values of the first types")
    say("codes: base (10, 512, 40), (6, 16, 40); mask (10, 64, 40)", codes_bytes([(10, 512, 40), (6, 16, 40)], [(10, 64, 40)]))
    say("an oracle message's notice: base code 0, mask code 0, 3 main rows over F, 1 mask row, 1 answer, leaf class 0", notice_bytes(0, 0, 3, 1, 1, 1, 0))
    say("the same without a mask code", notice_bytes(0, NO_CODE, 3, 1, 0, 1, 0))
    say("two stages: (0, 0, 2, 0), (1, 0, 0, 0)", stages_bytes([(0, 0, 2, 0), (1, 0, 0, 0)]))
    say("the choices: the masks ride", choices_bytes(1))
    print("# R12.x: a chain of segments (R12.12 to R12.17): three segments as in the example protocol \"links\"")
    three = [(0, [(2, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 1)]), (0, [(1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 1)]), (0, [(1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0)])]
    later = [(1, 1, 1, 1), (2, 2, 0, 0)]
    say("the chain of links: codes 0, 0; three links; nothing at the start; 1 answer; later stages (1, 1, 1, 1), (2, 2, 0, 0); the masks ride", chain_of_links_bytes(0, 0, three, [], 1, later, 1))
    say("its groups: k, h, sampled messages, width of a column", "; ".join("%d, %d, %d, %d" % g for g in link_groups(three)))
    rounds = links_rounds(5, three, 1, later)
    say("its rounds with 5 variables: how many; the names' first letters and the messages' bytes", "%d; %s" % (len(rounds), " ".join("%s%s" % ("".join(w[0] for w in r[0].split()), "" if r[2] is None else r[2]) for r in rounds)))
    print("# R13.x: three items, with the values of the public files of a toy run that makes a key file and a leaf file")
    toy = toy_run()
    say("the link: that table; 76 and 76 columns; codes 0, 0; 4 rows, high; class 0; masks ahead; later stages (1, 1, 2, 1), (2, 2, 0, 0); the masks ride", toy["the link"])
    say("the test on the table, of that run", toy["the test on the table"])
    say("ranges, of that run", toy["ranges"])
    print("# R13.x: the rounds of that run's key file (R13.16 to R13.19). A round is the line name|bytes of the message|elements sampled|of them not 0, with - for a number that depends on sizes, and a line feed")
    looks, ranges_test, tests = toy["lookups"], toy["the stages of ranges"], toy["the four lists"]
    for title, rounds in zip(("one-time", "shared", "second"), toy["the key file's rounds"]):
        say("the %s chain: how many rounds; the SHA-256 of their lines" % title, "%d; %s" % (len(rounds), hashlib.sha256(rounds_text(rounds)).hexdigest()))
    print("# the same with single words changed, for branches that this key file does not take. No key file has been made under these words, and none of the seven lines has been compared with a plan that the programs' planning routines wrote")
    for what, chains in (("the choices' word of ranges 0", key_rounds(looks, ranges_test, 0, 3, 3, 100, True, 1, tests, 1)),
                         ("no mask sent ahead", key_rounds(looks, ranges_test, 1, 3, 3, 100, True, 0, tests, 1)),
                         ("the choices' word of the test on the table 0", key_rounds(looks, ranges_test, 1, 3, 3, 100, True, 1, tests, 0)),
                         ("all three words 0", key_rounds(looks, ranges_test, 0, 3, 3, 100, True, 0, tests, 0)),
                         ("two answers to a sample", key_rounds(looks, ranges_test, 1, 3, 3, 100, True, 1, tests, 1, 2)),
                         ("the second look-up alone, so one part in all", key_rounds(looks[1:], ranges_test, 1, 3, 3, 100, True, 1, tests, 1)),
                         ("Omega under the folded row's code, so no split: one test, the first list", key_rounds(looks, ranges_test, 1, 3, 3, 100, False, 1, tests, 1))):
        say("%s: rounds and SHA-256 of the one-time, the shared, the second chain" % what, "; ".join("%d %s" % (len(r), hashlib.sha256(rounds_text(r)).hexdigest()) for r in chains))
    print("# R14.x: a proof file")
    say("the head of a proof that was made on the stored copy 1", proof_head(1))
    say("the statement of its first chain: the table's root pattern(32), the one answer (1, 2, 3); the copy 1; the request 01 02 03 04; the request file's words 05 06 07 08", proof_statement(pattern(32), ext((1, 2, 3)), 1, bytes([1, 2, 3, 4]), bytes([5, 6, 7, 8])))
    say("the item layout of three links without an array", layout_bytes([[], [], []]))
    say("the item layout: no array in the first link; in the second the array x, message 1, 8 rows, 4 columns, one block 0, 0, 2, 16, 3, 2, 1, 4, 1", layout_bytes([[], [("x", 1, 8, 4, [(0, 0, 2, 16, 3, 2, 1, 4, 1)])]]))
    say("the item binding tree: depth 2, salts of 40 bytes", u64(2) + u64(40))
    for place_ in range(4):
        say("the binding tree's root: depth 2, place %d, configuration pattern(32), seal pattern(32) with 5 i + 1, salt pattern(40) with 3 i + 9, siblings pattern(32) with 11 i + 2 and with 13 i + 4" % place_,
            binding_root(2, place_, pattern(32), pattern(32, 5, 1), pattern(40, 3, 9), [pattern(32, 11, 2), pattern(32, 13, 4)]))
    for pending in (1, 3):
        tail = first_chain_tail(pending, 6)
        say("the first chain's rounds behind its last test, %d pending row%s, 2^6 columns: how many; the SHA-256 of their lines" % (pending, "" if pending == 1 else "s"), "%d; %s" % (len(tail), hashlib.sha256(rounds_text(tail)).hexdigest()))
    print("# R15.x: the second hash function. words(n): word i is (2654435761 (i + 1)) mod 2^32. A digest is printed as its eight words' bytes, the lowest first")
    wn = lambda n: [(2654435761 * (i + 1)) & 0xFFFFFFFF for i in range(n)]  # noqa: E731
    d8 = lambda d: struct.pack("<8I", *d)  # noqa: E731
    for num, n in ((73, 0), (73, 1), (10, 32), (10, 33), (11, 128)):
        say("K(%d; words(%d))" % (num, n), d8(older_K(num, wn(n))))
    say("K(0; no word): num = 0 and L = 0 leave the counter as BLAKE2b's, so this is the first 32 bytes of BLAKE2b-512 of the empty string", d8(older_K(0, [])))
    for n in (8, 128, 129, 1024, 1025, 16385):
        say("TK(words(%d))" % n, d8(older_TK(wn(n))))
    say("the words of the text 'call a b c' with its line feed", struct.pack("<3I", *text_words("call a b c\n")))
    say("K(73; the words of the text of the two lines 'call a b c', 'call d e')", d8(older_K(73, text_words("call a b c\ncall d e\n"))))
    say("K(73; the words of the text of the six lines that stand for the lines callH where a file has none)", d8(older_K(73, text_words(CALLH_IF_NONE))))
    print("# R12.x: a code's points and symbols, field 2 (R12.11)")
    say("w = 7^(5 (2^32 - 1))", ROOT_32)
    say("the points of (a, x) = (15, 12345), (15, 2), (14, 9999)", "%d %d %d" % (point(15, 12345), point(15, 2), point(14, 9999)))
    message, tail = [1000003 * j + 17 for j in range(8)], [PRIME[2] - 1 - 5 * j for j in range(3)]
    for x in range(16):
        say("code (4, 8, 3), message 1000003 j + 17, tail p - 1 - 5 j, position %d: the point; the symbol" % x, "%d; %d" % (point(4, x), symbol(4, message, tail, x)))


if __name__ == "__main__":
    main()
