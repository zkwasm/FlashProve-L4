#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Computes the round errors of the proximity tests a second time, for three configurations.

usage: kit_rounds_second.py <cp_verify for the second field, or a folder with its outputs plan.<kit>.txt> <new folder>
status 0: as expected; 1: not; 2: a wrong call
Given a program, it makes the planning call of kit_rounds_test.py for each configuration (`kit` in
names and messages) and writes the outputs into the new folder; given a folder of outputs, it writes
nothing. Only planning calls are made: no proof is made or checked.
WHY. kit_rounds_test.py shows that a stored figure has changed, not that it is right: prover,
verifier and planning call take a round's error from one set of functions (blocks/numbers/), and
these printed the stored lines (verifier/tests/known_answers/kit_rounds/). This script computes the
same figures from a configuration's parameter file (policy.txt).
FORMULAS. From this protocol's technical report, not the programs' code. Three things were changed
after a first comparison with the stored lines. The integer part of the list bound is taken
(a list's size is an integer; the report prints the real number). The random coordinates of Omega (a
message of the test on the weight table) count the shared test and the requests' tests separately;
this shows only in the configurations with proof of work. The cost of a capped draw of a challenge
was added, from docs/format/FORMAT.md. What comes from the programs' headers is marked
FROM THE HEADERS, in Stage.close() and
compute(): choices and shapes of the programs, not error formulas, with one exception: the weight
table (next paragraph).
HOW INDEPENDENT IT IS. Theorem 4.6 of ePrint 2025/2055 is used as the technical report restates it,
which agrees with the restatement in blocks/numbers/codes.h. Whether these two restatements are
independent of each other is not known, and neither was compared with that posting for this script:
an error in the restatement would show in neither computation.
The weight table's two distances, its numbers of columns and the rounds
RECORD, LIN point, LINK point, LINK and LOC (table, columns) are computed by the formulas of
blocks/numbers/blocks.h, not the technical report's, with which they were not compared for this
script. For these lines the script is a second arithmetic, not a second derivation: it would show
a slip in the programs' search for the distances or in their numbers, not a wrong formula.
WHAT IT CHECKS. (1) That the output has the ten tests of TESTS. For every stage of each: the two
codes with their random coordinates, list size, positions; for a first stage also the integer m and
the distance, for a later one the numbers of rows. Also the numbers of opened columns of the weight
table. (2) That each list of positions meets the target less its work, and that with the same codes
one position fewer does not. (3) The stored lines named LOC, CS, BASE, BASE lambda, FOLD epsilon,
FOLD round, LINK, LINK point, LIN point, RECORD and RNG, and the SEND seed and BAT of the tests:
name, bits, work, number of rounds (names: blocks/numbers/round_names.h). The figure is compared
to two decimals, so terms that change no digit are not checked: the batching's term in a CS, the
list's in BASE lambda, the theorem's two small terms.
NOT CHECKED. PSC, LKP, LINK phi, ZERO, LINE, LIN zeta, JOIN, BAT-sigma; the SEND seed and BAT of a
request's segments (`link` in policy.txt, the output and identifiers) and of the committer's range
proof (`ranges`); the SEND seed of the committer's masks. Their shapes (variables, degrees, keys,
rows, which messages share a seed) belong to the sub-proofs, which the report does not size. The
last line says how many bits the lowest of these lines has. No sum of errors is checked.
TAKEN FROM THE OUTPUT: the number of main rows of each group's first stage; which segments a request
has; the number of mask rows at the first stage of the committer's range proof: the
first number at which the integer m computed here equals the printed one. FROM THE STORED LINES: how many FOLD rounds a segment has; in how many of the FOLD rounds of
two first stages in a commitment's key file, the range proof's and the own test's, the mask rows'
term occurs (both errors are computed; the numbers are taken at which the lines fit best, and the
last line prints them, so that a change shows). So a program that used too few mask rows at these
two stages would not be noticed here. Everything else comes from the parameter file or the headers.
ASSUMED. A request's own rounds are not multiplied by the number of requests in a batch; only the
shared point's round is. That is the technical report's rule: a query on a request's own chain
covers one request, so many requests cost many queries. The programs read it the same way
(blocks/numbers/setting.h, Setting::instances, which no header in blocks/frame/ sets), so this
comparison cannot test it. The message with a request's masks has one row (blocks/frame/files.h,
sized(), computes that number from the sub-proofs' rows; not evaluated here)."""
import os, pathlib, re, struct, subprocess, sys
from collections import defaultdict
from decimal import Decimal as mpf, getcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

getcontext().prec = 120  # mpf: a real number of 120 digits; rates are exact fractions
TREE = pathlib.Path(__file__).resolve().parent.parent.parent
KITS = ("target_full_f2", "target_small_f2", "pinned_no_proof_of_work")
# Where a configuration lies. The last is the tests' own, with no proof of work.
HOME = {kit: TREE / "prover" / "examples" / kit for kit in KITS[:2]} | {KITS[2]: TREE / "prover" / "tests" / "configurations" / KITS[2]}
RANGES = ("once", "ranges' multiplicities")
TESTS = [("request", "link"), ("request", "last test"), RANGES, ("once", "folded tables of a link")] + [(x, y) for x in ("batch", "once") for y in ("folded table", "Omega", "own test")]
P = 2**64 - 2**32 + 1  # the field's modulus (Goldilocks)
E = P**3               # the size of its cubic extension, |E| in the comments

# The differences that remain: (configuration, round's name, stored bits, computed bits): cause. In
# each the stored line claims less (fewer bits), so the programs are on the safe side. A new
# difference fails, and so does one that has gone or has changed.
WIDER_LIST = ("the mask message that parts a row is priced with the largest list of the groups in which it is tested, here Omega's (blocks/frame/lin_frame.h, lin_size_of(); blocks/numbers/blocks.h, "
              "list_of()). By the technical report it has its own code's list, about 6; a group's list is at least that")
KNOWN = {
    ("target_full_f2", "SEND seed (table, Omega)", "172.40", "172.75"): WIDER_LIST,
    ("target_full_f2", "SEND seed (table, Omega)", "172.41", "172.76"): WIDER_LIST,
    ("target_small_f2", "SEND seed (table, Omega)", "171.14", "171.60"): WIDER_LIST,
    ("target_small_f2", "SEND seed (table, Omega)", "171.37", "171.82"): WIDER_LIST,
    ("pinned_no_proof_of_work", "SEND seed (table, Omega)", "171.15", "171.61"): WIDER_LIST,
    ("pinned_no_proof_of_work", "SEND seed (table, Omega)", "171.41", "171.87"): WIDER_LIST,
}

sqrt = lambda x: x.sqrt()
log = lambda x, b=None: x.ln() / mpf(b).ln() if b else x.ln()
floor = lambda x: x.to_integral_value(ROUND_FLOOR)
ceil = lambda x: x.to_integral_value(ROUND_CEILING)
bits = lambda x: -log(x, 2)
cents = lambda b: int(floor(b * 100))                # the stored bits are cut, not rounded, to two decimals
shown = lambda c: "%d.%02d" % divmod(c, 100)         # hundredths of a bit as the stored lines write them
real = lambda q: mpf(q.numerator) / q.denominator
lg = lambda x: (x - 1).bit_length()                  # log2, rounded up


def draws(n, not_zero=0):
    """Returns what the capped draws of a round cost, as a count over |E|: n elements of E, and
       not_zero more that must not be 0 (a seed's factor). A try at a coordinate fails with chance
       q; after six failed tries the coordinate is 0. So an element of E is within 3 q^6 of uniform;
       one that must not be 0 is within twice that and 1/|E|^2. Positions are bits of a word,
       exactly uniform [docs/format/FORMAT.md R6.3 to R6.5, R6.11]."""
    q = mpf(2**64 - P) / 2**64
    return ((3 * n + 6 * not_zero) * q**6 + mpf(not_zero) / E**2) * E


class Code:
    """A plain Reed-Solomon code: length n, message length ell, r random coordinates."""
    def __init__(s, n, ell, r):
        s.n, s.ell, s.r, s.dim = n, ell, r, ell + r
        s.rho = Fraction(s.dim - 1, n)   # the cited theorem counts a code of dimension k + 1 with rho = k/n; 1 - (minimum distance)/n is the same number
        s.shape = (ell, r, n.bit_length() - 1)


def count_46(n, rho, delta, m=None):
    """Returns how many challenges can make far words look close, up to the Johnson radius. With
       m = max(ceil(sqrt(rho) / (1 - sqrt(rho) - delta)), 3) it is
       (2 (m + 1/2)^5 + 3 (m + 1/2) delta rho) / (3 rho^(3/2)) n + (m + 1/2) / sqrt(rho);
       no factor for interleaved rows
       [ePrint 2025/2055, Theorem 4.6, M = 1, as the technical report restates it]."""
    rho = real(rho)
    root = sqrt(rho)
    if m is None:
        if 1 - root - delta <= 0:
            return mpf("inf")            # at or beyond this code's own Johnson radius no lemma covers it
        m = max(int(ceil(root / (1 - root - delta))), 3)
    x = mpf(m) + mpf(1) / 2
    return (2 * x**5 + 3 * x * delta * rho) / (3 * rho * root) * n + x / root


def list_size(lam, delta):
    """Returns the integer part of a bound on the number of codewords within delta:
       (a - lam) / (a^2 - lam), with a = 1 - delta and lam = 1 - the relative minimum distance. The
       bound uses the minimum distance only, so it holds for interleaved words and for main and mask
       rows together."""
    a, lam = 1 - delta, real(lam)
    assert a * a > lam, "a distance at or beyond the Johnson radius has no list bound"
    return floor((a - lam) / (a * a - lam))


def seed_error(messages):
    """Returns the error of an out-of-domain sample's seed: each oracle message that shares the seed
       adds L^2 / 2 times max(a / |E|, 1 / (|E| - 1)). L: its list size; a: its cube's variables."""
    return sum(L * L / 2 * max(mpf(a) / E, mpf(1) / (E - 1)) for L, a in messages) + draws(max(a for L, a in messages), 1) / E


class Stage:
    """One stage of a proximity test: 2^k main rows and 2^h mask rows (a shorter mask code is
       repeated to the main length n) at one distance. The code with the larger rate is the ruling
       code: the joint list bound uses the smaller minimum distance, and the distance must lie below
       both Johnson radii."""
    def __init__(s, k, h, main, mask):
        s.k, s.h, s.main, s.mask, s.n, s.m, s.t = k, h, main, mask, main.n, None, None
        s.lam = max(main.rho, mask.rho)  # the ruling code's rate

    def far(s, code):
        return (code.n - code.dim + 1) * (s.n // code.n)     # minimum distance at the main length

    def set_m(s, m):
        """Sets the distance: the ruling code's Johnson radius less the slack sqrt(rho) / m."""
        root = sqrt(real(s.lam))
        s.m, s.delta = m, 1 - root - root / m
        s.e, s.L = int(floor(s.delta * s.n)), list_size(s.lam, s.delta)

    def close(s, code):
        """Returns the count in one code's proximity error (`closeness` in policy.txt), at the main
           length: within a third of the code's distance floor(delta n) + 1; beyond it count_46(),
           for the mask code with its own list size added."""
        if 3 * s.e < s.far(code):
            return mpf(s.e + 1)
        # For the ruling rate sqrt(rho) / (1 - sqrt(rho) - delta) IS m. FROM THE HEADERS, a choice: where
        # both codes have that rate, the programs compute the mask code's integer from the distance,
        # rounding upward, and get m + 1 (blocks/numbers/codes.h, at_distance()): the safe side.
        a = count_46(s.n, code.rho, s.delta, (s.m + int(code is s.mask and s.mask.rho == s.main.rho)) if code.rho == s.lam else None)
        return a + list_size(code.rho, s.delta) if code is s.mask and a < mpf("inf") else a

    def fold_epsilon(s):                 # the list over |E|
        return (s.L + draws(1)) / E

    def fold_round(s, j):                # main rows' proximity term, in the first h rounds twice the mask rows', twice the list
        return (s.close(s.main) + (2 * s.close(s.mask) if j <= s.h else 0) + 2 * s.L + draws(1)) / E

    def base_lambda(s):                  # both rows' proximity terms and the list
        return (s.close(s.main) + s.close(s.mask) + s.L + draws(1)) / E

    def seed(s):                         # one cube holds the main table (k + log ell bits) and the mask table (h + log ell_zk); one more variable tells them apart
        return seed_error([(s.L, max(s.k + lg(s.main.ell), s.h + lg(s.mask.ell)) + 1)])

    def holds(s, target, last, sampled):
        """Tells whether the rounds that depend on the distance, except the draws of positions, meet
           the target. last: the stage is a test's last (BASE, no FOLD); sampled: its oracle message
           has a sample."""
        errs = [s.base_lambda()] if last else [s.fold_epsilon(), s.fold_round(s.k)] + ([s.fold_round(1)] if s.h else [])
        return all(x <= target for x in errs + ([s.seed()] if sampled else []))

    def size(s, target, last, sampled):
        """Sets the largest m, so the smallest slack, at which holds() is true."""
        lo, hi = 3, 6
        s.set_m(hi)
        while s.holds(target, last, sampled):
            lo, hi = hi, 2 * hi
            s.set_m(hi)
        while hi - lo > 1:
            s.set_m((lo + hi) // 2)
            lo, hi = (s.m, hi) if s.holds(target, last, sampled) else (lo, s.m)
        s.set_m(lo)
        assert s.holds(target, last, sampled), "no m from 3 on holds the stage's rounds at the target"

    def positions(s, target):            # the least t with (1 - delta)^t at most the target
        return int(ceil(log(target) / log(1 - s.delta)))



def read_policy(path):
    """Returns the lines of policy.txt that are used here. Formats: group NAME rule log_n log_ell
       mask_log_n mask_log_ell; stages NAME, then k h log_n mask_log_n for each later stage; work
       KIND request shared once; table: see compute() [the top of blocks/frame/parameters.h defines
       them]."""
    pol = defaultdict(dict)
    for w in (l.split() for l in path.read_text().splitlines() if l.split()):
        if w[0] in ("group", "stages", "work"):
            pol[w[0]][w[1]] = [int(x) for x in w[2:]]
        elif w[0] in ("table", "sec", "slots", "answers", "record", "reading", "rowcost"):
            pol[w[0]] = [int(x) for x in w[1:]]
    return pol


def read_print(text):
    """Parses the lines "group:", "table:" and "link" of a planning call's output."""
    groups, part, links, table = {}, "request", defaultdict(int), []
    for line in text.splitlines():
        g = re.match(r"link \d+: sub-proof ([A-Z]+)", line)
        if g:
            links[g.group(1)] += 1
        if line.startswith("table:"):
            table += [int(x) for x in re.findall(r"\b(\d+) (?:columns|of the further)", line)]
        part = "batch" if line == "group: in a batch:" else "once" if line == "group: in the committer's files:" else part
        g = re.match(r"group: ([^:]+): code 2\^(\d+), message (\d+) \+ (\d+) random, 2\^(\d+) rows; mask code 2\^(\d+), (\d+) \+ (\d+); distance ([0-9.]+),(?: m = (\d+),)? list (\d+), (\d+) positions", line)
        if g:
            v, cur = [x and (x if "." in x else int(x)) for x in g.groups()[1:]], (part, g.group(1))
            groups[cur] = [dict(k=v[3], main=(v[1], v[2], v[0]), mask=(v[5], v[6], v[4]), delta=v[7], m=v[8], L=v[9], t=v[10])]
        g = re.match(r"group:   stage \d+: 2\^(\d+) rows of (\d+) \+ (\d+) in 2\^(\d+); 2\^(\d+) mask rows of (\d+) \+ (\d+) in 2\^(\d+); list (\d+), (\d+) positions", line)
        if g:
            v = [int(x) for x in g.groups()]
            groups[cur].append(dict(rows=(v[0], v[4]), main=tuple(v[1:4]), mask=tuple(v[5:8]), L=v[8], t=v[9]))
    return groups, links, table


def settle(make, target, targets_pos, last=False, sampled=True):
    """Returns the stages that make() builds, with their numbers of positions. Positions and random
       coordinates depend on one another: a row has as many random coordinates as there are draws at
       which it is opened, and these lower the distance. So each number is raised from 1 until no
       stage needs more. That this is the least such point is not proved: a step
       of m can lower the need as the rate grows (the same holds for solve_group() in
       blocks/frame/derive.h)."""
    t = [1] * len(targets_pos)
    while True:
        stages = make(*t)
        for st in stages:
            st.size(target, last, sampled)
        need = [st.positions(x) for st, x in zip(stages, targets_pos)]
        if all(a <= b for a, b in zip(need, t)):
            for st, n in zip(stages, t):
                st.t = n
            return stages
        t = [max(a, b) for a, b in zip(need, t)]


def compute(pol, printed, links):
    """Returns, for one configuration: the stages of every test; the numbers of columns; the lists
       of positions as (name, error, error with one position fewer, target); the rounds as (error,
       number of rounds, name, bits of work)."""
    assert (pol["reading"], pol["rowcost"], pol["answers"]) == ([1], [1], [1]) and pol["record"] == [1] and all(pol["work"][x] == [0, 0, 0] for x in ("field", "sample", "closeness", "smallness")), \
        "this script knows kits with reading 1, rowcost 1, answers 1, record 1 and work before draws of positions alone"
    target, N_batch = mpf(2)**-pol["sec"][0], pol["slots"][0]
    w_request, w_shared, w_once = pol["work"]["positions"]     # w bits of work before a draw: its error times 2^-w [after ePrint 2021/582, Theorem 6]
    out, lists, chains = [], [], {}

    # FROM THE HEADERS: h = 0, one mask row, at the first stage of a segment, the last test, the
    # folded row, Omega and the two folded rows (blocks/frame/derive.h, shape_of();
    # blocks/frame/req_frame.h, req_frame(); blocks/frame/lin_frame.h, lin_sized();
    # blocks/iop/link.h, test()).
    def first(name, key, r_main, r_mask, h=0):
        _, ln, le, lnz, lez = pol["group"][name]
        return Stage(printed[key][0]["k"], h, Code(1 << ln, 1 << le, r_main), Code(1 << lnz, 1 << lez, r_mask))

    def test(key, name, where, st, w, first_folds=True):
        """Adds the rounds of a proximity test: at each stage but the last FOLD (k + 1 rounds) and
           CS, the code switch (2 rounds); at the last BASE (2 rounds). first_folds False: main()
           adds the first stage's FOLD rounds. A CS's other term, some 2^-174, is left out of
           Stage.positions() and tested through `lists`."""
        chains[key], quads = [st], pol["stages"][name]
        for i in range(0, len(quads), 4):
            k, h, ln, lnz = quads[i:i + 4]
            # The 2^h mask rows take in both old mask rows, all old random coordinates and the
            # sample's pad: ellz is the smallest power of two that fits these, at least 2k + 1 - h.
            ell, ellz = st.main.ell >> k, max(1 << lg(-(-(2 * st.mask.ell + st.main.r + 2 * st.mask.r + 1) >> h)), 1 << lg(2 * k + 1 - h))
            nxt, = settle(lambda t: [Stage(k, h, Code(1 << ln, ell, t), Code(1 << lnz, ellz, t))], target, [target * 2**w], i + 4 == len(quads))   # a stage's rows are opened at its one list
            out.append((st.fold_epsilon(), 1, "FOLD epsilon (%s, stage #)" % where, 0))
            if first_folds or len(chains[key]) > 1:
                out.extend([(st.fold_round(1), st.h, "FOLD round (%s, stage #)" % where, 0), (st.fold_round(st.k), st.k - st.h, "FOLD round (%s, stage #)" % where, 0)])
            out.append((nxt.seed(), 1, "SEND seed (%s, stage #)" % where, 0))
            cs = lambda t: (1 - st.delta)**t + (nxt.L * (1 + 3 * t) + draws(1)) / E   # the other term: the next stage's list times the batched claims less one: 1 old, 1 sample, 3 a position
            out.append((cs(st.t), 1, "CS (%s, stage #)" % where, w))
            lists.append(("CS (%s), stage %d" % (where, len(chains[key])), cs(st.t), cs(st.t - 1), target * 2**w))
            chains[key].append(nxt)
            st = nxt
        out.extend([(st.base_lambda(), 1, "BASE lambda (%s, last stage)" % where, 0), ((1 - st.delta)**st.t, 1, "BASE (%s, last stage)" % where, w)])
        lists.append(("BASE (%s)" % where, (1 - st.delta)**st.t, (1 - st.delta)**(st.t - 1), target * 2**w))

    # A request's segments. Every row is opened at its own closing's list, so r = r_zk = t. A
    # closing's draw of positions (LOC) has the error (1 - delta)^t.
    link, = settle(lambda t: [first("link", ("request", "link"), t, t)], target, [target * 2**w_request], sampled=False)
    chains[("request", "link")] = [link]
    lists.append(("LOC of a link", (1 - link.delta)**link.t, (1 - link.delta)**(link.t - 1), target * 2**w_request))
    for x, long_name in (("G", "G#"), ("A", "attention #"), ("H", "head")):
        y = x if x == "H" else x + "#"
        out.extend([((1 - link.delta)**link.t, links[x], "LOC (%s)" % y, w_request), (link.fold_epsilon(), links[x], "FOLD epsilon (%s)" % y, 0), (link.fold_round(1), None, "FOLD round (%s)" % y, 0)])
        out.append((target, links[x], "RNG (%s, smallness test)" % long_name, 0))    # r combinations give 2^-r; r is the least that meets the target [without log N: see ASSUMED]
    # The last test, on the message into which the last segment is closed: the fold's row and the
    # carried row, no mask rows of its own; the segments' codes and distance.
    last = first("link", ("request", "last test"), link.t, link.t)
    last.set_m(link.m)
    last.t = link.t
    out.append((seed_error([(last.L, last.k + lg(last.main.ell))]), 1, "SEND seed (last group)", 0))
    out.append(((last.L * (1 + 3 * link.t) + draws(1)) / E, 1, "BAT (last group)", 0))    # the list times the claims less one: 1 old, 1 sample, 3 a position
    test(("request", "last test"), "link", "last group", last, w_request)

    # The line `table` (the weight table): 2^v rows of a code of length n_M; data and random
    # coordinates; the number 0. FROM THE HEADERS, the
    # sixth number: the coordinates of the point that a batch shares, the v that fold the rows among
    # them: as many as the matrix has column variables (top of blocks/frame/parameters.h;
    # blocks/iop/lin.h, k_col()).
    v, lnM, ellM, rM, radius, kcol = pol["table"]
    nM = 1 << lnM
    # FROM THE HEADERS, formulas too (blocks/numbers/blocks.h: size_recorded_table(),
    # table_fold_error(), plan_record(), plan_lin_shared(), plan_link()): see HOW INDEPENDENT IT
    # IS. The table has an out-of-domain sample of its own, and is read beyond unique
    # decoding. Its two distances are the Johnson radius less the slack sqrt(rho) / m. A fold of
    # all rows at one point costs v proximity terms and v times the list. The one-time m is the
    # largest at which the fold point's round with two tables' folds, and the record's round,
    # meet the target; a later test's m likewise with the batch's requests at the point and
    # without the record's round, and it is at most the one-time m.
    assert radius == 0, "the line table names no radius"
    code = Code(nM, ellM, rM)
    root = sqrt(real(code.rho))
    far = lambda m: 1 - root - root / m
    fold = lambda m: v * count_46(nM, code.rho, far(m), m) + v * list_size(code.rho, far(m))
    record = lambda m: seed_error([(list_size(code.rho, far(m)), lg(ellM))])

    def largest(ok):
        lo, hi = 3, 6
        assert ok(lo), "no m from 3 on holds the table's rounds at the target"
        while ok(hi):
            lo, hi = hi, 2 * hi
        while hi - lo > 1:
            lo, hi = ((lo + hi) // 2, hi) if ok((lo + hi) // 2) else (lo, (lo + hi) // 2)
        return lo
    m_once = largest(lambda m: (kcol + 2 * fold(m)) / E <= target and record(m) <= target)
    m_later = min(m_once, largest(lambda m: (N_batch * kcol + 2 * fold(m)) / E <= target))
    # A column misses a wrong fold with chance 1 - distance.
    later, once = 1 - far(m_later), 1 - far(m_once)
    points = [(v + fold(m_once) + fold(m_later) + draws(v)) / E, (kcol + fold(m_once) + draws(kcol)) / E, (N_batch * kcol + fold(m_later) + draws(kcol)) / E]
    out.extend([(record(m_once), 1, "RECORD (table)", 0), (record(m_once), 1, "RECORD (further table)", 0)])
    # `parent`: the anchor (the first weight table) in a LINK proof.
    nu = {x: int(ceil(log(target * 2**w) / log(y))) for x, y, w in (("batch", later, w_shared), ("once", once, w_once), ("parent", later, w_once))}
    lists.extend(("columns of the table (%s)" % x, y**nu[x], y**(nu[x] - 1), target * 2**w) for x, y, w in (("batch", later, w_shared), ("once", once, w_once), ("parent", later, w_once)))
    out.extend([(later**nu["batch"], 1, "LOC (table, columns)", w_shared), (once**nu["once"], 1, "LOC (table, columns)", w_once), (max(later**nu["parent"], once**nu["once"]), 1, "LINK (link)", w_once)])
    out.extend([(points[0], 1, "LINK point (link)", 0), (points[1], 1, "LIN point (table)", 0), (points[2], 1, "LIN point (table)", 0)])

    # The test on the table, in a batch (N requests) and in the committer's files (N = 1). Three
    # groups: the folded table re-encoded; Omega, the table times the shared point's equality
    # vector; own, Omega with a request's own vector W.
    for part, pre, N, w_sh, w_own in (("batch", "", N_batch, w_shared, w_request), ("once", "once-", 1, w_once, w_once)):
        # Omega is opened by the shared test and by N requests' tests; the mask message that splits
        # a claim ("parts a row" in the messages) between Omega's group and the folded row's, by
        # both shared tests. The own group has Omega's main code. FROM THE HEADERS, a shape: where
        # W's columns fill its rows, its pad is in a mask row, the own group's second: h = 1
        # (blocks/frame/lin_frame.h, lin_sized()).
        ell_w = 1 << pol["group"][pre + "own"][2]
        full, a_w = (1 << kcol) % ell_w == 0, lg(-(-(1 << kcol) // ell_w)) + lg(ell_w)
        fo, om, ow = settle(lambda tf, to, tw: [first(pre + "folded", (part, "folded table"), tf, tf + to), first(pre + "omega", (part, "Omega"), to + N * tw, tf + to), first(pre + "own", (part, "own test"), to + N * tw, tw, int(full))],
                            target, [target * 2**w_sh, target * 2**w_sh, target * 2**w_own])
        a_om = om.k + lg(om.main.ell)
        own_list = list_size(om.mask.rho, max(om.delta, fo.delta))     # the mask message that parts a row has its own code's list, at the larger of the two distances
        # W: an entry for each column; with a mask row one more variable tells the two tables apart [without a factor N: see ASSUMED]
        out.append((seed_error([(ow.L, max(a_w, lg(ow.mask.ell)) + 1 if full else a_w)]), 1, "SEND seed (table, W)", 0))
        out.extend([(seed_error([(fo.L, fo.k + lg(fo.main.ell))]), 1, "SEND seed (table, folded row)", 0), (seed_error([(om.L, a_om), (own_list, lg(om.mask.ell))]), 1, "SEND seed (table, Omega)", 0)])
        # BAT: the list times the claims less one. Folded row: the columns, two samples, one part of
        # the split claim, one more for its answer (FROM THE HEADERS, a shape: the top
        # of blocks/iop/foldeq.h); Omega: two samples, the other part; own: two claims, three samples.
        out.extend([((fo.L * (nu[part] + 3) + draws(1)) / E, 1, "BAT (folded row)", 0), ((om.L * 2 + draws(1)) / E, 1, "BAT (Omega)", 0), ((ow.L * 4 + draws(1)) / E, 1, "BAT (a request's own test)", 0)])
        test((part, "folded table"), pre + "folded", "folded row", fo, w_sh)
        test((part, "Omega"), pre + "omega", "Omega", om, w_sh)
        test((part, "own test"), pre + "own", "a request's own test", ow, w_own, part != "once")   # in a key file the number of mask rows depends on the range proof (blocks/frame/files.h, `sized`): `main` adds these rounds

    # A leaf file's LINK proof (also `link` in the output): the two folded rows in one group under
    # the codes of `once-folded`. BAT: both tables' columns, two samples, the claim that equates
    # them, one more for each table's answer (blocks/frame/link_frame.h, link_sized()).
    two, = settle(lambda t: [first("once-folded", ("once", "folded tables of a link"), t, t)], target, [target * 2**w_once])
    out.extend([((two.L * (nu["parent"] + nu["once"] + 4) + draws(1)) / E, 1, "BAT (the two folded rows)", 0), (seed_error([(two.L, two.k - 1 + lg(two.main.ell))] * 2), 1, "SEND seed (link)", 0)])
    test(("once", "folded tables of a link"), "once-folded", "the two folded rows", two, w_once)
    # How many mask rows the first stage of the committer's range proof has depends on the sub-proof
    # (blocks/frame/wf_frame.h, wf_sized()).
    # The distance depends on that number: the first number at which the integer m equals the
    # printed one.
    ranges = [settle(lambda t: [first("once-ranges", RANGES, t, t, h)], target, [target * 2**w_once])[0] for h in range(printed[RANGES][0]["k"] + 1)]
    test(RANGES, "once-ranges", "ranges", next((st for st in ranges if st.m == printed[RANGES][0]["m"]), ranges[0]), w_once, False)
    return chains, [nu["once"], nu["parent"], nu["once"], nu["batch"]], lists, out


def compare(stored, out):
    """Pairs the rounds computed here with the stored ones. A stored line names no stage ("#"), and
       a batch's and the committer's tests share names, so the rounds of one name and number of work
       bits are paired together: equal figures first, what is left in ascending order. Gives each
       stored line a verdict: same, less or more (its claim against the figure computed here), open
       (not computed). Also returns, in words, where the pairing fails."""
    rows, here, faults = [], defaultdict(list), []
    for line in stored:
        g = re.match(r"round:  2\^-(\d+)\.(\d\d)(?: = (\d+)\.(\d\d) by the lists \+ (\d+) by work)?, (\d+) round\(s\): (.*)$", line)
        rows.append(dict(c=int(g.group(1) + g.group(2)), w=int(g.group(5) or 0), n=int(g.group(6)), name=g.group(7), text=line[8:], side="same", by=mpf(0)))
        if g.group(5) and int(g.group(3) + g.group(4)) + 100 * int(g.group(5)) != rows[-1]["c"]:
            faults.append("the bits by the lists and by work do not add up: " + line[8:])
    for err, n, name, w in out:
        here[name, w] += [bits(err) + w] * (sum(r["n"] for r in rows if r["name"] == name) if n is None else n)
    names = set(name for name, w in here)
    for key in sorted(set((r["name"], r["w"]) for r in rows)):
        left, rest = sorted(here.pop(key, [])), []
        for r in (r for r in rows if (r["name"], r["w"]) == key for _ in range(r["n"])):
            hit = next((b for b in left if cents(b) == r["c"]), None)
            left.remove(hit) if hit is not None else rest.append(r)
        for r in sorted(rest, key=lambda r: r["c"]):
            if not left:
                r["side"] = "open" if r["side"] == "same" else r["side"]
                if key[0] in names:
                    faults.append("a stored round of a name that is computed here has no partner of that name and work: " + r["text"])
                continue
            b = left.pop(0)
            side, by = ("less", b - mpf(r["c"]) / 100) if cents(b) > r["c"] else ("more", mpf(r["c"] + 1) / 100 - b)    # the most that the two can lie apart, the stored bits being cut
            r["side"], r["by"], r["computed"] = "more" if "more" in (side, r["side"]) else side, max(r["by"], by), b
        here[key] = left
    return rows, faults + ["a round is computed here that no stored line has: 2^-%.4f, of them %d by work: %s" % (b, w, name) for (name, w), x in sorted(here.items()) for b in x]


def main():
    if len(sys.argv) != 3 or os.path.lexists(os.path.abspath(sys.argv[2])):
        print(__doc__.split("\n")[2].strip(), file=sys.stderr)
        return 2
    source, work = pathlib.Path(sys.argv[1]), pathlib.Path(os.path.abspath(sys.argv[2]))
    if not source.is_dir():
        work.parent.mkdir(parents=True, exist_ok=True)
        work.mkdir(mode=0o700)  # fails if the name is taken; closed to other users
        (work / "y").write_bytes(struct.pack("<I", 0))
    total, seen, lowest, fitted = defaultdict(int), set(), 10**6, []
    words = dict(main="main code (message, random coordinates, log2 of the length)", mask="mask code (message, random coordinates, log2 of the length)", L="list size", t="positions")
    for kit in KITS:
        at = HOME[kit]
        if source.is_dir():
            text = (source / ("plan.%s.txt" % kit)).read_text()
        else:    # kit_rounds_test.py's planning call
            tokens = 1 << int(next(l.split()[1] for l in (at / "policy.txt").read_text().splitlines() if l.startswith("lt ")))
            (work / "request").write_bytes(struct.pack("<%dI" % (tokens + 2), *([0] * tokens + [tokens - 1, tokens])))
            call = [sys.argv[1], "-plan", "request", str(at / "policy.txt"), str(at / "KL.bin"), "128", "-layout", str(at / "layout.txt"), "-prompt", str(work / "request"), "-reply", str(work / "y"), "-shapes"]
            r = subprocess.run(call, capture_output=True, text=True, errors="replace", check=False)
            text = r.stdout
            (work / ("plan.%s.txt" % kit)).write_text(text)
            if r.returncode != 4:
                print("NOT AS EXPECTED: %s: the planning call ended with status %d, not 4; what it printed is in %s" % (kit, r.returncode, work / ("plan.%s.txt" % kit)))
                return 1
        printed, links, table = read_print(text)
        if sorted(printed) != sorted(TESTS):
            total["off the print"], total["derived"] = total["off the print"] + 1, total["derived"] + 1
            seen |= set(key for key in KNOWN if key[0] == kit)
            print("NOT AS PRINTED: %s: this script knows the tests %s; the planning call prints %s: nothing of this kit is computed; teach the script the new print" % (kit, sorted(TESTS), sorted(printed)))
            continue
        chains, columns, lists, out = compute(read_policy(at / "policy.txt"), printed, links)
        here = defaultdict(int)
        pairs = [("columns of the table (one-time test; a link's first and further table; a batch)", columns, table)]
        for key, chain in chains.items():
            for i, (st, p) in enumerate(zip(chain, printed[key])):
                pairs += [("%s, %s, stage %d: %s" % (key + (i + 1, words[x])), y, p[x]) for x, y in (("main", st.main.shape), ("mask", st.mask.shape), ("L", int(st.L)), ("t", st.t))]
                pairs += [("%s, %s: the integer m" % key, st.m, p["m"]), ("%s, %s: distance" % key, "%.6f" % st.delta, p["delta"])] if i == 0 else [
                    ("%s, %s, stage %d: log2 of the main and of the mask rows" % (key + (i + 1,)), (st.k, st.h), p["rows"])]
            pairs.append(("%s, %s: number of stages" % key, len(chain), len(printed[key])))
        for what, a, b in pairs:
            here["derived"] += 1
            if a != b:
                here["off the print"] += 1
                print("NOT AS PRINTED: %s: %s: computed here %s, the planning call prints %s; see by %s which of the two is right" % (kit, what, a, b, at / "policy.txt"))
        here["lists"] = len(lists)
        for name, a, b, tgt in lists:
            if not a <= tgt < b:
                here["lists off"] += 1
                print("NOT THE LEAST LIST: %s: %s: %s; look at its distance and its number of positions" % (
                    kit, name, "it misses its target, 2^-%.4f against 2^-%.4f" % (bits(a), bits(tgt)) if a > tgt else "one position fewer would do"))
        stored, first = (TREE / "verifier" / "tests" / "known_answers" / "kit_rounds" / (kit + ".txt")).read_text().splitlines(), chains[RANGES][0]

        own = chains[("once", "own test")][0]

        def with_mask_rows(h, g):    # the FOLD rounds of two first stages in a key file, the ranges' and the own test's: the first h (g) have the mask rows' term
            first.h, own.h = h, g
            return (h, g) + compare(stored, out + [(first.fold_round(1), h, "FOLD round (ranges, stage #)", 0), (first.fold_round(first.k), first.k - h, "FOLD round (ranges, stage #)", 0),
                                                   (own.fold_round(1), g, "FOLD round (a request's own test, stage #)", 0), (own.fold_round(own.k), own.k - g, "FOLD round (a request's own test, stage #)", 0)])
        h, g, rows, faults = min((with_mask_rows(h, g) for h in range(first.k + 1) for g in range(own.k + 1)), key=lambda x: len(x[3]) + sum(r["side"] in ("less", "more") for r in x[2]))
        here["faults"] = len(faults)
        print("".join("NOT PAIRED: %s: %s; see whether the programs' rounds or this script's have changed\n" % (kit, x) for x in faults), end="")
        for r in rows:
            here[r["side"]] += 1
            lowest = min(lowest, r["c"]) if r["side"] == "open" else lowest
            if r["side"] in ("less", "more"):
                key = (kit, r["name"], shown(r["c"]), shown(cents(r["computed"])))
                said = "%s: the stored line claims %s than is computed here (2^-%.4f), by at most %.4f bit: %s" % (kit, r["side"], r["computed"], r["by"], r["text"])
                seen.add(key)
                here["known" if key in KNOWN else "off the table"] += 1
                print("known: %s. Cause: %s" % (said, KNOWN[key]) if key in KNOWN else "NOT ON THE TABLE KNOWN: %s; look at the price, then mend it or the table at the head of this script" % said)
        print("%s: %d stored lines: %d the same, %d known to claim less, %d off the table KNOWN, %d not computed; %d of %d derived numbers as printed; %d lists of positions; in the key 2^%d mask rows at the ranges' first stage, 2^%d at the own test's" % (
            kit, len(rows), here["same"], here["known"], here["off the table"], here["open"], here["derived"] - here["off the print"], here["derived"], here["lists"], h, g))
        fitted.append("%d %d" % (h, g))
        for x in here:
            total[x] += here[x]
    for key in sorted(set(KNOWN) - seen):
        total["off the table"] += 1
        print("NOT ON THE TABLE KNOWN: %s: the table has %s, stored 2^-%s, computed 2^-%s: that difference is gone or has moved; mend the table at the head of this script" % key)
    bad = total["off the print"] + total["lists off"] + total["faults"] + total["off the table"]
    print("kit_rounds_second.py: %d kits: %d stored lines, %d computed: %d the same, %d known to claim less, %d off the table KNOWN; of the %d not computed the lowest has %s bits; "
          "%d of %d derived numbers as printed; %d lists of positions, %d too short or longer than need be; %d faults in the pairing; log2 of the mask rows fitted in each key (the ranges', the own test's): %s: %s" % (
              len(KITS), sum(total[x] for x in ("same", "less", "more", "open")), sum(total[x] for x in ("same", "less", "more")), total["same"], total["known"], total["off the table"], total["open"],
              shown(lowest), total["derived"] - total["off the print"], total["derived"], total["lists"], total["lists off"], total["faults"], ", ".join(fitted), "NOT AS EXPECTED" if bad else "as expected"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
