#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Whether an error of the platform's mathematical library can change an entry of the public
tables.

usage: table_ties.py

The tables (docs/model.md, section 4) are defined by a procedure of binary64 operations. IEEE 754
fixes +, * and /. The definition takes exp, pow, cos and sin correctly rounded; the programs take
them from the platform's library, which need not round correctly. An entry is the procedure's last
number rounded to an integer, so a small error of the library changes it only if that number lies
very near a tie, k + 1/2.

This script looks at EVERY entry of every table that the standard parameter form admits
(verifier/docs/parameter-file.md, the conditions softmax-table and rotary-table): cs 12; theta0
10,000 and 500,000; dh a power of two up to 128; T up to 1,024; S 12, rx 8; eg 8 with smax 4,095.
The rotary tables for dh 128 and T 1,024 hold those for every smaller dh and T: pair i of dh has
the angle of pair i * 128 / dh. For each entry the script

  1. carries out the procedure by its own lines below, with correctly rounded functions: Python's
     decimal at 60 digits, rounded once to binary64. These lines are the judge;
  2. compares the entry with that of scales/policy_tables.py on this platform, which is the thing
     under test, for every dh;
  3. takes the real number that is rounded last, at 60 digits, from the binary64 argument that the
     procedure has there, and its distance from the nearest tie;
  4. bounds from above what one unit in the last place of each library result can move that number,
     and divides: a library must be off by at least so many units to change the entry;
  5. carries it out again with every library result moved by K units, in every pattern of signs,
     for K = 1, 16 and 1,024, and counts the entries that change;
  6. rounds the last number also as C's round() does, which the verifier uses where this procedure
     says floor(v + 0.5), and counts the entries that differ.

It asks that the least distances and margins are at least the floors below, and that 2, 5 and 6 find
nothing.

The SiLU table is another case. At the keys c = 2 mod 4 above 0 the number 64 z is itself a tie,
and the real value 64 z / (1 + exp(-z)) lies just below it. There the library decides one thing
only: whether 1 + exp(-z) rounds to 1, that is, on which side of 2^-53 its exp(-z) lies. From key
9,406 on the sum is 1, and the binary64 number that is rounded IS a tie, at 1,745 keys: what fixes
these entries is the rule "away from zero", not a distance. 1,744 of them are one above the
correctly rounded real function; the last is cut to smax either way. Below key 9,406 the real value
comes as near to the tie as 2.64e-13, and at 88 keys within 1e-12, and binary64 division still puts
the number below the tie. The script asks for these counts, that exp(-z) is at least 1e13 units
away from 2^-53 at every such key, and that the least power of two of units of exp at which any
entry of the table changes is 2^38.

A control. Outside the standard form an entry can depend on the library: at theta0 3,675,548,936 and
dh 128 the entry C(689, 1) lies 1.5e-10 from a tie. The script asks that its steps 3 and 4 see that.

What this is not. It is a computation, not a proof. It trusts decimal's exp and ln, and its own
series for sine and cosine. The bound of step 4 is of first order. Step 5 moves all results by the
same number of units; that no smaller error changes an entry follows because each step is monotonic
there, not from these runs. It takes as given: binary64 +, * and / in the stated order, rounded to
nearest, and no reciprocal in place of a division ("fast math"). Extended precision is a real
condition: with 80-bit intermediate results 9 entries of the SiLU table differ, or 488 if the number
is not stored as binary64 before it is rounded (by a computation that is not in this script). A
fused multiply-add changes nothing: each product that an addition follows has a power of two as a
factor and is exact. A run says nothing of another machine's library: step 2 is what a run there
adds.

It needs nothing but Python. About a minute. Exit status 0 and a last line "OK: <n> checks": every
check passed. 1: one did not; the last line says which.
"""

import decimal
import itertools
import math
import os
import sys
from decimal import Decimal

sys.dont_write_bytecode = True  # the import below leaves no file in the tree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import policy_tables  # noqa: E402

CONTEXT = decimal.getcontext()
CONTEXT.prec = 60
HALF = Decimal("0.5")
UNITS = (1, 16, 1024)
NOWHERE = (Decimal("Infinity"), "")
checks = 0


def ask(holds, what):
    global checks
    checks += 1
    if not holds:
        print("FAILED: " + what)
        sys.exit(1)


def arctangent_of_one_over(n):
    term = total = Decimal(1) / n
    for k in itertools.count(3, 2):
        term = -term / (n * n)
        if abs(term) < Decimal(10) ** -(CONTEXT.prec + 2):
            return total
        total += term / k


CONTEXT.prec = (
    100  # an angle of up to 1,023 radians is reduced with more digits than it is used with
)
TWO_PI = 2 * (16 * arctangent_of_one_over(5) - 4 * arctangent_of_one_over(239))  # Machin's formula
CONTEXT.prec = 60
ask(
    str(TWO_PI).startswith("6.28318530717958647692528676655900576839433879875021"),
    "this script's own 2 pi is %s: its series or decimal's arithmetic is at fault" % TWO_PI,
)


def sine_and_cosine(x):
    """Of the Decimal x >= 0: the angle is reduced, divided by 16, put into the two series, and
    doubled four times."""
    turns = (x / TWO_PI).to_integral_value(rounding=decimal.ROUND_FLOOR)
    CONTEXT.prec = 100
    rest = x - turns * TWO_PI
    CONTEXT.prec = 60
    small = rest / 16
    square = small * small
    sine = sine_term = small
    cosine = cosine_term = Decimal(1)
    for n in itertools.count(2, 2):
        cosine_term = -cosine_term * square / ((n - 1) * n)
        sine_term = -sine_term * square / (n * (n + 1))
        cosine += cosine_term
        sine += sine_term
        if abs(cosine_term) < Decimal(10) ** -64:
            break
    for _ in range(4):
        sine, cosine = 2 * sine * cosine, cosine * cosine - sine * sine
    return sine, cosine


def moved(v, k):
    """The binary64 number v moved by k units in its last place. (At a binade's edge the unit
    changes, which does not matter here.)"""
    return v + k * math.ulp(v)


def rha(v):
    """To the nearest integer, ties away from zero, with a binary64 addition."""
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def as_c_rounds(v):
    """C's round(): exact, ties away from zero."""
    whole = math.floor(abs(v))
    whole += abs(v) - whole >= 0.5
    return whole if v >= 0 else -whole


def nearest(x):
    """The Decimal x to the nearest integer, ties away from zero."""
    return int(x.to_integral_value(rounding=decimal.ROUND_HALF_UP))


def from_a_tie(x):
    return abs(x - x.to_integral_value(rounding=decimal.ROUND_FLOOR) - HALF)


def unlike(judge, theirs, place):
    """How many entries of policy_tables.py's list differ from the judge's, a missing or surplus
    entry counted as one that differs, and where the first is."""
    found = [n for n, pair in enumerate(zip(judge, theirs)) if pair[0] != pair[1]]
    if len(judge) != len(theirs):
        found.append(min(len(judge), len(theirs)))
    return len(found) + max(abs(len(judge) - len(theirs)) - 1, 0), place(found[0]) if found else ""


def told(name, entries, distance, margin, changed, differing, otherwise):
    print(
        "%s: %d entries; least distance from a tie %.4g (%s); a library must be off by at least "
        "%.4g units (%s); entries that change if every result is moved by K units, every pattern "
        "of signs: %s; entries that differ from policy_tables.py's on this platform: %d; under C's "
        "round(): %d"
        % (
            name,
            entries,
            distance[0],
            distance[1],
            margin[0],
            margin[1],
            ", ".join("K = %d: %d" % pair for pair in changed.items()),
            differing[0],
            otherwise,
        ),
        flush=True,
    )
    ask(not any(changed.values()), name + ": an entry changes if the library's results are moved")
    ask(
        differing[0] == 0,
        "%s: policy_tables.py's table is not the definition's, first at %s: this platform's "
        "library or arithmetic gives another table, so its digests will not be the parameter files'"
        % (name, differing[1]),
    )
    ask(otherwise == 0, name + ": C's round() gives another entry than floor(v + 0.5)")


# The rotary tables: b = (-2 i) / dh; w = pow(theta0, b); x = t * w; C(t, i) = rha(2^12 * cos(x));
# S(t, i) likewise with sin.
def power(theta0, i):
    return float((Decimal(theta0).ln() * Decimal((-2 * i) / 128)).exp())


def looked_at(w, t):
    """For S(t, .) and C(t, .) at the binary64 number w: the number that is rounded, the distance of
    its real value from a tie, and what one unit of each library result moves at most."""
    x = t * w
    sine, cosine = sine_and_cosine(Decimal(x))
    found = []
    for value, other in ((sine, cosine), (cosine, sine)):
        # One unit of pow moves x by t units of w and by at most one more rounding of the product;
        # the derivative is the other function. Then one unit of the function itself, and the
        # rounding of rha's addition.
        angle = Decimal(t * math.ulp(w) + math.ulp(x))
        one_unit = 4096 * (abs(other) * angle + Decimal(math.ulp(float(value))))
        found.append((float(value), from_a_tie(4096 * value), one_unit + Decimal(math.ulp(4096.0))))
    return found


for theta0, least, units in ((10000, 2.0e-6, 100000), (500000, 9.3e-8, 2500)):
    distance, margin, changed, otherwise, judge = NOWHERE, NOWHERE, dict.fromkeys(UNITS, 0), 0, {}
    for i in range(64):
        w = power(theta0, i)
        for t in range(1024):
            here = looked_at(w, t)
            others = {(k, a): looked_at(moved(w, a), t) for k in UNITS for a in (-k, k)}
            for which in (0, 1):
                value, d, one_unit = here[which]
                entry, at = rha(4096 * value), "%s(%d, %d) at dh 128" % ("SC"[which], t, i)
                judge[which, t, i] = entry
                otherwise += as_c_rounds(4096 * value) != entry
                distance, margin = min(distance, (d, at)), min(margin, (d / one_unit, at))
                for (k, _), there in others.items():
                    changed[k] += any(
                        rha(4096 * moved(there[which][0], c)) != entry for c in (-k, k)
                    )
    differing = (0, "")
    for dh in (128, 64, 32, 16, 8, 4, 2):
        cosines, sines = policy_tables.rotary_tables(1024, dh, 12, theta0)
        for which, theirs in ((1, cosines), (0, sines)):
            pairs = dh // 2
            ours = [judge[which, t, i * 64 // pairs] for t in range(1024) for i in range(pairs)]
            where = "%s(%%d, %%d) at dh %d" % ("SC"[which], dh)
            more = unlike(ours, theirs, lambda n: where % divmod(n, pairs))
            differing = (differing[0] + more[0], more[1] or differing[1])
    name = "rotary tables, theta0 %d" % theta0
    told(name, len(judge), distance, margin, changed, differing, otherwise)
    ask(distance[0] >= least and margin[0] >= units, name + ": below the floors")
value, d, one_unit = looked_at(power(3675548936, 1), 689)[1]
print(
    "the control outside the standard form: C(689, 1) = %d; %.4g from a tie; %.4g units"
    % (rha(4096 * value), d, d / one_unit)
)
ask(
    rha(4096 * value) == -548 and d < 2e-10 and d / one_unit < 10,
    "the control: expected the entry -548, less than 2e-10 from a tie and less than 10 units: this "
    "script does not see a near tie, so what it says of the other tables does not count",
)

# The softmax table: EX[u] = rne(2^12 * exp(-u / 2^8)), up to the last entry that is not 0. The
# first 0 is looked at too. Python's round() is rne; no number here is a tie.
distance, margin, changed, otherwise, judge = NOWHERE, NOWHERE, dict.fromkeys(UNITS, 0), 0, []
while not judge or judge[-1]:
    u = len(judge)
    real = (Decimal(-u) / 256).exp()
    e = float(real)
    entry, d, at = round(4096 * e), from_a_tie(4096 * real), "u = %d" % u
    judge.append(entry)
    otherwise += as_c_rounds(4096 * e) != entry
    distance = min(distance, (d, at))
    margin = min(margin, (d / (4096 * Decimal(math.ulp(e))), at))
    for k in UNITS:
        changed[k] += any(round(4096 * moved(e, c)) != entry for c in (-k, k))
differing = unlike(judge[:-1], policy_tables.softmax_table(12, 8), lambda n: "u = %d" % n)
told("softmax table, S 12, rx 8", len(judge), distance, margin, changed, differing, otherwise)
ask(len(judge) == 2308 and distance[0] >= 2.4e-4 and margin[0] >= 1e9, "softmax table: the floors")

# What the procedure's own three roundings can move in the two tables below, added to each bound.
ROUNDINGS = 3 * Decimal(math.ulp(4096.0))
UNIT = Decimal(2) ** -52  # one unit in the last place, relative to the number, at most

# The sigmoid table: e = exp(-c / 256); g = 1 + e; q = 4096 / g; Sig[c] = 16 rha(q).
distance, margin, changed, otherwise, judge = NOWHERE, NOWHERE, dict.fromkeys(UNITS, 0), 0, []
for c in range(-32768, 32768):
    real = (Decimal(-c) / 256).exp()
    e = float(real)
    entry, d, at = rha(4096 / (1 + e)), from_a_tie(4096 / (1 + real)), "key %d" % c
    judge.append(16 * entry)
    otherwise += as_c_rounds(4096 / (1 + e)) != entry
    one_unit = (
        4096 * real / (1 + real) ** 2 * UNIT + ROUNDINGS
    )  # by the derivative of 4096 / (1 + e)
    distance, margin = min(distance, (d, at)), min(margin, (d / one_unit, at))
    for k in UNITS:
        changed[k] += any(rha(4096 / (1 + moved(e, s))) != entry for s in (-k, k))
differing = unlike(judge, policy_tables.sigmoid_table(), lambda n: "key %d" % (n - 32768))
told("sigmoid table", len(judge), distance, margin, changed, differing, otherwise)
ask(distance[0] >= 1.7e-4 and margin[0] >= 1e7, "sigmoid table: the floors")


# The SiLU table, eg 8: c' = c cut to -16,384 .. 16,383; z = c' / 256; g = 1 + exp(-z); s = z / g;
# Ts[c] = rha(64 * s) cut to -4,095 .. 4,095. The keys outside c' repeat the first and last entry.
def silu(z, e):
    return min(max(rha(64 * (z / (1 + e))), -4095), 4095)


distance, margin, changed, otherwise, judge = NOWHERE, NOWHERE, dict.fromkeys(UNITS, 0), 0, {}
ties, above, near, below, astray, threshold, exponentials = 0, 0, 0, NOWHERE, 0, NOWHERE, {}
for c in range(-16384, 16384):
    z = c / 256
    real_e = Decimal(-z).exp()
    e = exponentials[c] = float(real_e)
    real = 64 * Decimal(z) / (1 + real_e)
    entry, d, at = silu(z, e), from_a_tie(real), "key %d" % c
    judge[c] = entry
    rounded = min(max(nearest(real), -4095), 4095)
    otherwise += as_c_rounds(64 * (z / (1 + e))) != rha(64 * (z / (1 + e)))
    for k in UNITS:
        changed[k] += any(silu(z, moved(e, s)) != entry for s in (-k, k))
    if c > 0 and c % 4 == 2:  # 64 z is itself a tie
        threshold = min(threshold, (abs(real_e - Decimal(2) ** -53) / Decimal(math.ulp(e)), at))
        if 1 + e == 1:
            ties += (64 * (z / (1 + e))) % 1 == 0.5
            above += entry == rounded + 1
        else:
            astray += entry != rounded
            near += d < Decimal("1e-12")
            below = min(below, (d, at))
    else:
        astray += entry != rounded
        one_unit = 64 * abs(Decimal(z)) * real_e / (1 + real_e) ** 2 * UNIT + ROUNDINGS
        distance, margin = min(distance, (d, at)), min(margin, (d / one_unit, at))
name = "SiLU table, eg 8 (distance and margin: but for the keys at which 64 z is a tie)"
whole = [judge[min(max(c, -16384), 16383)] for c in range(-32768, 32768)]
ask(list(policy_tables.KEYS) == list(range(-32768, 32768)), "policy_tables.py's KEYS")
differing = unlike(whole, policy_tables.silu_table(), lambda n: "key %d" % (n - 32768))
told(name, len(judge), distance, margin, changed, differing, otherwise)
for power_of_two in range(11, 53):
    if any(
        silu(c / 256, moved(e, s << power_of_two)) != silu(c / 256, e)
        for c, e in exponentials.items()
        for s in (-1, 1)
    ):
        break
print(
    "SiLU table, the keys at which 64 z is a tie: at %d keys the number that is rounded is a tie, "
    "and %d entries are one above the correctly rounded real function; below them the real value's "
    "least distance from the tie is %.4g (%s), and %d keys lie within 1e-12; exp(-z) is at least "
    "%.4g units from 2^-53 (%s). The least power of two of units of exp at which an entry of the "
    "table changes: 2^%d"
    % (ties, above, below[0], below[1], near, threshold[0], threshold[1], power_of_two)
)
ask(distance[0] >= 1.6e-4 and margin[0] >= 1e7, "SiLU table: the floors")
ask(
    (ties, above, near, astray) == (1745, 1744, 88, 0),
    "SiLU table: expected 1,745 ties, 1,744 entries above the real function, 88 keys within 1e-12 "
    "and 0 other entries that are not the rounded real function; found %d, %d, %d, %d: this "
    "Python's arithmetic is not binary64 in the stated order" % (ties, above, near, astray),
)
ask(
    Decimal("2.63e-13") < below[0] < Decimal("2.64e-13") and below[1] == "key 9402",
    "SiLU table: expected the nearest real value below the ties at key 9,402, 2.636e-13 from its "
    "tie",
)
ask(threshold[0] >= 1e13 and power_of_two == 38, "SiLU table: exp(-z) against 2^-53, or 2^38")

print("OK: %d checks" % checks)
