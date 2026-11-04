#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Compares the toy run's known answers with the files that the toy tool writes.

usage: vectors_against_the_toy_files.py PAGE DIRECTORY
PAGE is docs/format. DIRECTORY holds key.one-time.parameters, key.shared.parameters,
key.second.parameters and entry.parameters, as blocks/frame/handover_files writes them
(run_honest.sh: its folder key-and-entry).

Why. docs/format/format-vectors.txt has three items and the rounds of "a toy run that makes a key
file and a leaf file". docs/format/format_rules.py prints them from values that are typed by hand
(toy_run()), and the test that compares the file with that print cannot tell whether the typed
values are those of the files that the tool writes. This test can.

What it asks.
1. The items "ranges" and "the test on the table" in the one-time chain's parameters, and "the
   link" in the leaf file's, are byte for byte what toy_run() gives.
2. The plans of the key file's three chains have, beside RECORD, as many entries as key_rounds() gives rounds for
   toy_run()'s values, each entry for one round, with the same names in the same order; and where
   the rule gives a number (the message's bytes, the elements sampled, how many of them are not 0)
   the plan has that number. The one-time chain's plan begins with the entry RECORD, which is no
   round: it must have the eleven words of R13.15, and key_rounds() gives what follows it.
3. format-vectors.txt has these very bytes and numbers of rounds in its lines on that run, and one
   line on each.

What it does not show. That the rules are right where they give no number (a "-" in a round's
line). Anything on the other words of a plan's entry (work bits, roots, answers, positions, k,
matrix entries, rows), on the plan's first seven words, on the format's version, on the other
items, on the copy of "the test on the table" in the shared chain's parameters, or on the leaf
file's plan: it compares none of them. Anything on the seven lines with single words changed.
Anything on a proof's content: it reads parameters alone, no key file and no leaf file.

The last line ends with ALL AGREES, and the exit code is 0. Otherwise lines on the differences come
first, the last line ends with their number, and the exit code is 1: a round gets one line, on its
first difference, and once there are more than 20 lines the rest of a chain's rounds are not
compared.
Exit code 2: wrong arguments; or a file is missing or cannot be read; or format_rules.py cannot be
used; or an item's name appears twice in a parameters file.
"""
import hashlib
import struct
import sys

if len(sys.argv) != 3:
    print(__doc__.split("\n\n")[1]); sys.exit(2)
PAGE, D = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
sys.path.insert(0, PAGE)
try:
    import format_rules as R  # noqa: E402
    toy = R.toy_run()
except Exception as e:  # noqa: BLE001: whatever a hand-edited script raises is told in one line
    print("cannot use %s/format_rules.py: %s: %s" % (PAGE, type(e).__name__, e)); sys.exit(2)


def parameters(name):
    """The plan's entries and the items behind it (R4.2, R4.3)."""
    par = open(D + "/" + name, "rb").read()
    at = len(R.item("the format's version", b"H821" + bytes(4)))
    eight = struct.unpack("<8Q", par[at:at + 64]); at += 64
    entries = []
    for _ in range(eight[7]):
        n = struct.unpack("<Q", par[at:at + 8])[0]
        entries.append((par[at + 8:at + 8 + n].decode("ascii"),) + struct.unpack("<11Q", par[at + 8 + n:at + 96 + n])); at += 96 + n
    items = {}
    while at < len(par):
        n = struct.unpack("<Q", par[at:at + 8])[0]; name_ = par[at + 8:at + 8 + n].decode("ascii"); at += 8 + n
        m = struct.unpack("<Q", par[at:at + 8])[0]
        if name_ in items:
            raise ValueError("the item \"%s\" appears twice" % name_)
        items[name_] = par[at + 8:at + 8 + m]; at += 8 + m
    if at != len(par):
        raise ValueError("the last item runs past the end of the file")
    return entries, items


try:
    reading = ""
    files = {}
    for t in ("one-time", "shared", "second"):
        reading = "key.%s.parameters: " % t
        files[t] = parameters("key.%s.parameters" % t)
    reading = "entry.parameters: "
    leaf = parameters("entry.parameters")
    reading = "format-vectors.txt: "
    pairs = [line.rstrip("\n").split(" = ", 1) for line in open(PAGE + "/format-vectors.txt", encoding="ascii") if " = " in line and not line.startswith("#")]
except (OSError, ValueError, struct.error) as e:
    print("cannot read: %s%s" % (reading, e)); sys.exit(2)

bad = []
for name, got in (("ranges", files["one-time"][1].get("ranges")), ("the test on the table", files["one-time"][1].get("the test on the table")), ("the link", leaf[1].get("the link"))):
    if got is None:
        bad.append("the item \"%s\" is not in the tool's parameters" % name)
    elif got != toy[name]:
        words = [i // 8 for i in range(0, max(len(got), len(toy[name])), 8) if got[i:i + 8] != toy[name][i:i + 8]]
        bad.append("the item \"%s\": the tool writes %d bytes, toy_run() gives %d; the first words of eight bytes that differ, counted from 0: %s" % (name, len(got), len(toy[name]), ", ".join(map(str, words[:8]))))
    line = [v for k, v in pairs if k == name + ", of that run" or k.startswith(name + ":")]
    if len(line) != 1:
        bad.append("format-vectors.txt: it has %d lines on the item \"%s\", not one" % (len(line), name))
    elif line[0] != toy[name].hex():
        bad.append("format-vectors.txt: its line on the item \"%s\" is not what toy_run() gives" % name)
for title, rule in zip(("one-time", "shared", "second"), toy["the key file's rounds"]):
    plan = files[title][0]
    if title == "one-time":
        _, message, _, _, answers, _ = toy["the table"]
        entry = ("RECORD", 1, 0, 0, 0, 0, answers * (R._log_up(message) + 1), answers, 0, 0, 0, 0)
        if plan[:1] != [entry]:
            bad.append("the one-time chain: the tool's plan does not begin with the entry RECORD of R13.15: the rules give %s, the tool's plan has %s" % (entry, plan[:1]))
        plan = plan[1:]
    if len(plan) != len(rule):
        bad.append("the %s chain: the tool's plan has %d entries, the rules give %d rounds" % (title, len(plan), len(rule)))
    for number, (e, r) in enumerate(zip(plan, rule), 1):
        want = [("name", r[0], e[0]), ("rounds in a row like this one", 1, e[1]), ("bytes of the message", r[1], e[3]), ("elements sampled", r[2], e[6]), ("of them not 0", r[3], e[7])]
        for what, a, b in want:
            if a is not None and a != b:
                bad.append("the %s chain, round %d: %s: the rules give %s, the tool's plan has %s" % (title, number, what, a, b)); break
        if len(bad) > 20:
            break
    said = [v for k, v in pairs if k == "the %s chain: how many rounds; the SHA-256 of their lines" % title]
    if len(said) != 1:
        bad.append("format-vectors.txt: it has %d lines on the %s chain's rounds, not one" % (len(said), title))
    elif said != ["%d; %s" % (len(rule), hashlib.sha256(R.rounds_text(rule)).hexdigest())]:
        bad.append("format-vectors.txt: its line on the %s chain's rounds is not what the rules give for toy_run()" % title)

for b in bad:
    print(b)
if bad:
    print("If the tool's values were changed on purpose, type them into toy_run() in docs/format/format_rules.py. Then, or if the lines above name format-vectors.txt alone, see what the rules give now, from the repository's top, without writing a file: python3 docs/format/format_rules.py | diff docs/format/format-vectors.txt - (lines marked > are what the rules give now; format-vectors.txt is a file of the repository, so edit it to match only if those differences are the ones you meant)")
print("three items and %s rounds of the key file's three chains: %s" % (", ".join(str(sum(e[0] != "RECORD" for e in files[t][0])) for t in ("one-time", "shared", "second")), "ALL AGREES" if not bad else "LINES ON DIFFERENCES: %d" % len(bad)))
sys.exit(1 if bad else 0)
