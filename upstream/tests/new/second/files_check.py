#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by tests/new/second/run_honest.sh, which the runner starts; by hand: below a directory of your own.
"""Checks a toy `key.file` and `entry.file` with the Python verifiers; makes faulty copies of them.

The key file (`key.file`) has the root of a commitment's first table (the anchor) and proofs about
it. The leaf file (`entry.file`) ties a copy of that table (a leaf table; `the copy` in messages)
to the anchor. blocks/frame/handover_files.cpp (the C++ tool) writes both files, with lists of the
intermediate values and rejection messages of the C++ verifier (printed: "the C++ reader").
tests/new/second/key_check.py and tests/new/second/entry_check.py (printed: "the second verifier")
check the files: they, not this script, decide whether a file is accepted. One author wrote this
script, both Python verifiers and the C++ tool; the verifiers' docstrings say where they are not
independent of the C++ programs.

usage: files_check.py <directory of format_rules.py> <directory of the files> [mode]
  (no mode)  checks both honest files and compares values with the C++ verifier's lists
  --write    also writes the four hash chains' values: write_chain()
  --cheats <directory of false files> <list of expected results>
  --altered [<directory for the copies>] | --more-altered | --spoil-vertex-record
  --bend <directory of false files>
--write and --more-altered write into the directory of the files; --altered and --bend, into the
directory given with them (--altered without one writes no file). An output file replaces a file
or symbolic link of its name. The script exits rather than write into a directory that is a
symbolic link, belongs to another account, or is writable by its group or others: new_file() in
tests/new/second/toy_check.py.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import toy_check as tc  # noqa: E402
import key_check as kc  # noqa: E402
import entry_check as ec  # noqa: E402

# docs/format/format_rules.py; tests/new/second/toy_check.py imports it from directory argv[1]
R = tc.R


# Short names as in column "expected" of the list given with --cheats. `THEIRS`: the C++ messages
WHERE = {"the link's chain, its proximity test": "LINK", "the first table's opening": "the first table's opening", "the copy's opening": "the copy's opening", "the one-time chain, WF's own test": "WF", "the shared chain, the test on the folded row's group": "U", "the shared chain, the test on Omega's group": "OM",
         "the second chain, the test on the own rows' group": "OWN"}
CHECKS = {"a vertex of a tree gets two values under one root": "D3b", "BASE: the linear check fails": "D4", "BASE: a check at a position fails": "D5", "a nonce has not the work of its round": "D2", "an opening does not lead to its root": "D3"}
THEIRS = {"BASE: the masked message in the proof does not satisfy the linear claim": "D4", "BASE: the encoding of the masked message differs from the opened symbols at a listed position": "D5",
          "a nonce does not give the proof of work that its round asks for": "D2", "an opening does not lead to its root": "D3",
          "the opened columns of the table do not lead to its root": "D3"}


def cheats(get, folder, false_folder, list_path):
    """Compares how each false file is rejected with column "expected" of `list_path`.

    It compares the failed check (D2 to D5) for both verifiers, and the proximity test or opening
    for the Python verifier. With D2 expected, the proximity test is not compared.
    """
    import re
    names = {"first.parameters": "key.one-time.parameters", "first.statement": "key.one-time.statement", "shared.parameters": "key.shared.parameters", "second.parameters": "key.second.parameters"}
    bad, n = 0, 0
    for l in open(list_path, encoding="utf-8").read().splitlines():
        if l.startswith("#") or l.startswith("file	") or not l.strip():
            continue
        which, name, _, _, _, expected = l.split("	")[:6]
        stem = "%s/%s.cheat-%s" % (false_folder, which, name)
        try:
            data = open(stem + ".file", "rb").read()
        except FileNotFoundError:   # ANOTHER_KEY: the honest leaf file under another anchor's root
            assert which == "entry" and name == "ANOTHER_KEY"
            data = get("entry.file")
        try:
            words = open(stem + ".txt").read().strip()
        except FileNotFoundError:
            words = "(not judged by the C++ reader)"
        try:
            if which == "key":
                kc.verify(lambda tail: data if tail == "file" else get(names[tail]))
            else:
                raw, first = get("entry.parameters"), open(stem + ".first-table", "rb").read()
                ec.verify(ec.parse(raw), raw, R.item("the first table", first), data, [])
            mine, place, where = "ACCEPTED", "", ""
        except tc.Refused as why:
            said, _, where = str(why).partition(" [in ")
            mine, place = CHECKS.get(said, said), WHERE.get(where.rstrip("]"), "the round «LINK»" if which == "entry" else "outside a test")
        check = "D" + expected.split("D", 1)[1][0]
        behind = expected.split(check, 1)[1]
        test = behind[4:].split(";")[0].split(",")[0].split(" ")[0] if behind.startswith(" of ") else None
        opening = "the copy's opening" if behind.startswith(", at the copy's opening") else "the first table's opening" if behind.startswith(", at the FIRST table's opening") else None   # BENT_COLUMN, BENT_SWAP: the opening that must reject
        lucky = check == "D2" and mine == "D3"    # the list allows D3: the work test passed by luck
        cpp = THEIRS.get(words) == mine   # the C++ verifier's message names the same check
        ok = (mine == check or lucky) and (test is None or place == test or check == "D2") and (opening is None or where.rstrip("]") == opening) and cpp
        place = opening if opening and ok else place
        n, bad = n + 1, bad + (not ok)
        print("%s	%s	expected beforehand: %s	the second verifier: %s%s	the C++ reader: %s	%s" % (which, name, expected, mine, " in " + place if place else "", words,
                                                                                                 "as expected" if ok else "NOT AS EXPECTED"))
    print("%d false files, %s" % (n, "ALL REFUSED AS EXPECTED" if not bad else "%d NOT AS EXPECTED" % bad))
    return 1 if bad else 0


def write_chain(folder, name, ch, more):
    """Writes `<name>.digests.tsv` and `<name>.chain.tsv` (a line per round).

    The values are the Python verifier's or computed here from them. Each round's hash call 0 is
    recomputed directly with hashlib and compared.
    """
    import hashlib
    with tc.new_file("%s/%s.digests.tsv" % (folder, name)) as o:
        o.write("what\thex\n")
        for what, value in [("the chain's label", "%x" % ch.label), ("the configuration's digest", ch.conf.hex()), ("the parameters' digest", ch.pd.hex()), ("the chain's random bytes", ch.random.hex()),
                            ("the run's value", ch.run.hex()), ("the first chain value", ch.rho0.hex())] + more:
            o.write("%s\t%s\n" % (what, value))
    with tc.new_file("%s/%s.chain.tsv" % (folder, name)) as o:
        o.write("round\tname\tkind\twork bits\tnonce\tthe chain's value before\tbytes of the prover's message\tsha256 of the prover's message\tcall 0's 64 bytes\tthe chain's value after\telements drawn\tpositions drawn\n")
        for m in ch.rounds:
            data = m["message"] + R.u64(m["kind"] + (m["work"] << 8)) + R.u64(m["nonce"])
            call0 = R.H(162, ch.label, m["round"], m["before"], data, 64)
            assert hashlib.blake2b(R.label(162, ch.label, m["round"]) + data, digest_size=64, salt=m["before"][:16], person=m["before"][16:]).digest() == call0
            o.write("\t".join([str(m["round"]), m["name"], str(m["kind"]), str(m["work"]), str(m["nonce"]), m["before"].hex(), str(len(m["message"])), hashlib.sha256(m["message"]).hexdigest(), call0.hex(),
                               m["after"].hex(), "; ".join(tc.dec(e) for e in m["E"]), " ".join(map(str, m["X"]))]) + "\n")


def altered(get, folder, out):
    """Makes the copies that the C++ tool's altered.tsv lists; the Python verifier must reject each.

    In its column "at byte", n means that bit 0 of byte n is flipped; "-", that the last byte is
    cut off or a zero byte is added. The copies are compared with the C++ tool's copies, where
    these are in `folder`.
    """
    import hashlib
    import os
    import re
    names = {"first.parameters": "key.one-time.parameters", "first.statement": "key.one-time.statement", "shared.parameters": "key.shared.parameters", "second.parameters": "key.second.parameters"}
    first = dict(kc.lc.items_of(get("entry.statement")))["the first table"]
    bad, n, held = 0, 0, 0
    print("file\tcopy\twhat was changed\tat byte\tsha256 of the altered copy\tthe second verifier\tthe C++ reader's words")
    for l in get("altered.tsv").decode().splitlines()[1:]:
        which, number, what, at, words = l.split("\t")
        if which not in ("key", "entry") or not re.fullmatch(r"[0-9]+", number):   # both become parts of file names
            raise SystemExit("altered.tsv: a line names the file %r and the copy %r; expected: key or entry, and a number" % (which, number))
        data = bytearray(get(which + ".file"))
        if at != "-":
            data[int(at)] ^= 1
        elif what == "the last byte is cut off":
            data.pop()
        else:
            assert what == "a byte is added"
            data.append(0)
        data = bytes(data)
        theirs = "%s/%s.altered-%s.file" % (folder, which, number)
        if os.path.exists(theirs) and int(number) <= 12:   # 13 to 15: --more-altered's; no C++ copy
            held += 1
            if open(theirs, "rb").read() != data:
                bad += 1
                print("%s %s: THE RECIPE DOES NOT GIVE THE TOOL'S OWN COPY" % (which, number))
        if out:
            with tc.new_file("%s/%s.altered-%s.file" % (out, which, number), "wb") as o:
                o.write(data)
        try:
            if which == "key":
                kc.verify(lambda tail: data if tail == "file" else get(names[tail]))
            else:
                raw = get("entry.parameters")
                ec.verify(ec.parse(raw), raw, R.item("the first table", first), data, [])
            mine = "ACCEPTED"
            bad += 1
        except tc.Refused as why:
            mine = "refused: %s" % why
        n += 1
        print("\t".join((which, number, what, at, hashlib.sha256(data).hexdigest(), mine, words)))
    print("%d altered copies made from the recipe, %d of them held against the C++ tool's own copies (three of the key's are this program's own; the C++ reader has judged those as files); %s" % (
        n, held, "ALL REFUSED" if not bad else "%d THINGS ARE WRONG" % bad))
    return 1 if bad else 0


# Writes three more altered copies of the key file (13 to 15), each with one bit flipped at an
# offset known only after parsing. Prints their lines for altered.tsv without the C++ verifier's
# message, which tests/new/second/run_honest.sh adds.
def more_altered(get, folder):
    names = {"file": "key.file", "first.parameters": "key.one-time.parameters", "first.statement": "key.one-time.statement", "shared.parameters": "key.shared.parameters", "second.parameters": "key.second.parameters"}
    A, B, _ = kc.verify(lambda tail: get(names[tail]))
    m, data = dict(A.marks), get("key.file")
    for number, what, at in ((13, "the place's word", m["the place"]), (14, "the place's salt", m["the place"] + 8), (15, "the first symbol of the table's first opened column", B.table_opening_at)):
        bent = bytearray(data)
        bent[at] ^= 1
        with tc.new_file("%s/key.altered-%02d.file" % (folder, number), "wb") as o:
            o.write(bytes(bent))
        print("key\t%02d\t%s\t%d" % (number, what, at))
    return 0


# Tests the Python verifier itself. No file can trigger its check "one value at a vertex" without a
# hash collision. So the verifier's record of vertex values (`zkpy.SESSION`; not a table's record)
# is overwritten between a tree's two openings; that check must then reject the honest key file.
def spoil_record(get):
    import zkpy
    names = {"file": "key.file", "first.parameters": "key.one-time.parameters", "first.statement": "key.one-time.statement", "shared.parameters": "key.shared.parameters", "second.parameters": "key.second.parameters"}
    honest, bad = zkpy.opening, 0
    for number, what in ((2, "the split's mask message's tree, which is opened at two lists of the shared chain"), (1, "Omega's tree, which is opened in the shared chain and, taken over, in the second")):
        done = []

        def spoiling(ch, t, draws):
            columns = honest(ch, t, draws)
            if not t.stored and (ch.label, t.number) == (0x4c494e01, number) and not done:
                done.append(True)
                for vertex in zkpy.SESSION[t.root]:
                    zkpy.SESSION[t.root][vertex] = bytes(32)
            return columns

        zkpy.opening = spoiling
        try:
            kc.verify(lambda tail: get(names[tail]))
            said = "ACCEPTED"
        except tc.Refused as why:
            said = str(why)
        zkpy.opening = honest
        ok = said.startswith("a vertex of a tree gets two values under one root")
        bad += not ok
        print("the record of values at vertices, spoiled behind the first opening of %s: %s" % (what, said))
    print("THE RECORD OF VALUES AT VERTICES IS KEPT AND READ" if not bad else "THE RECORD OF VALUES AT VERTICES IS NOT READ")
    return 1 if bad else 0


def bend(get, out):
    """Writes two false leaf files, made from the honest file's bytes without a prover.

    BENT_COLUMN has the anchor's first opened column in place of the leaf table's; BENT_SWAP has
    the two openings exchanged. The anchor's item of the statement (its root and its record's
    answers) is written beside each (`.first-table`).
    """
    raw, data = get("entry.parameters"), get("entry.file")
    par = ec.parse(raw)
    ch = ec.verify(par, raw, get("entry.statement"), data, [])
    a, b, c = ch.openings_at
    column = 8 << par["link"]["first"]["v"]     # 2^v symbols of 8 bytes
    first = dict(kc.lc.items_of(get("entry.statement")))["the first table"]
    for name, bent in (("BENT_COLUMN", data[:b] + data[a:a + column] + data[b + column:]), ("BENT_SWAP", data[:a] + data[b:c] + data[a:b] + data[c:])):
        assert len(bent) == len(data) and bent != data
        for tail, value in (("file", bent), ("first-table", first)):
            with tc.new_file("%s/entry.cheat-%s.%s" % (out, name, tail), "wb") as o:
                o.write(value)
        print("%s: made from the honest entry: %d bytes; the openings stand at %d, %d, %d" % (name, len(bent), a, b, c))
    return 0


# A missing key.targets.tsv or entry.targets.tsv is skipped; values in the first that the Python
# verifier does not note do not count as differences.
def main():
    folder = sys.argv[2]
    get = lambda name: open("%s/%s" % (folder, name), "rb").read()  # noqa: E731
    if "--altered" in sys.argv:
        i = sys.argv.index("--altered")
        return altered(get, folder, sys.argv[i + 1] if len(sys.argv) > i + 1 else None)
    if "--spoil-vertex-record" in sys.argv:
        return spoil_record(get)
    if "--more-altered" in sys.argv:
        return more_altered(get, folder)
    if "--bend" in sys.argv:
        return bend(get, sys.argv[sys.argv.index("--bend") + 1])
    if "--cheats" in sys.argv:
        i = sys.argv.index("--cheats")
        return cheats(get, folder, sys.argv[i + 1], sys.argv[i + 2])
    conf = R.configuration_digest([get("public-file-1.txt"), get("public-file-2.txt")])
    names = {"file": "key.file", "first.parameters": "key.one-time.parameters", "first.statement": "key.one-time.statement", "shared.parameters": "key.shared.parameters",
             "second.parameters": "key.second.parameters"}
    bad = 0
    for p in ("key.one-time.parameters", "key.shared.parameters", "key.second.parameters", "entry.parameters"):
        same = dict(kc.lc.parse(get(p), False)["items"])["configuration"] == conf if not p.startswith("entry") else dict(ec.parse(get(p))["items"])["configuration"] == conf
        bad += not same
        print("%s: the item «configuration» is the digest of the two public files: %s" % (p, same))
    try:
        A, B, same = kc.verify(lambda tail: get(names[tail]))
        print("key.file: ACCEPTED by the second verifier (%d bytes); the one-time chain's statement, formed by the reader, is the C++ Binder's: %s" % (len(get("key.file")), same))
        bad += not same
        try:
            theirs = [l.split("\t") for l in get("key.targets.tsv").decode().splitlines()[1:]]
        except FileNotFoundError:
            theirs = []
        if theirs:   # the C++ verifier's i-th value under a name against our i-th
            ours = {}
            for what, value in A.notes[:A.first_notes] + B.notes + A.notes[A.first_notes:]:
                ours.setdefault(what, []).append(tc.dec(value))
            seen, equal, differ, unknown = {}, 0, 0, {}
            for what, value in theirs:
                i = seen.get(what, 0)
                seen[what] = i + 1
                if what not in ours or i >= len(ours[what]):
                    unknown[what] = unknown.get(what, 0) + 1
                elif ours[what][i] == value:
                    equal += 1
                else:
                    differ += 1
                    print("   DIFFERS: %s, number %d" % (what, i + 1))
            print("   key.targets.tsv: %d values: %d equal to the second verifier's, %d differ, %d that it does not note (%s)" % (len(theirs), equal, differ, sum(unknown.values()), "; ".join(sorted(unknown))))
            bad += differ
        if "--write" in sys.argv:
            for (name, ch), more in zip(A.runs, ([("its seal", A.seals[0].hex())], [("the binding tree's root (its random bytes, and an item of its statement)", A.binding_root.hex()), ("its seal", A.seals[1].hex())], [])):
                write_chain(folder, name, ch, more)
        print("   its parts begin at: " + "; ".join("%s %d" % m for m in A.marks))
        print("   trees of the one-time and second chain, in the order of the book (number, depth, width in F): " + ", ".join("(%d, %d, %d)" % (t.number, t.depth, t.width) for t in A.trees))
        print("   trees of the shared chain: " + ", ".join("(%d, %d, %d)" % (t.number, t.depth, t.width) for t in B.trees))
    except tc.Refused as why:
        print("key.file: REFUSED by the second verifier: %s" % why)
        bad += 1
    try:
        notes = []
        raw = get("entry.parameters")
        che = ec.verify(ec.parse(raw), raw, get("entry.statement"), get("entry.file"), notes)
        if "--write" in sys.argv:
            write_chain(folder, "entry", che, [])
        print("entry.file: ACCEPTED by the second verifier (%d bytes); %s: %s" % (len(get("entry.file")), notes[0][0], notes[0][1]))
        try:
            theirs = [tuple(l.split("\t")) for l in get("entry.targets.tsv").decode().splitlines()[1:]]
        except FileNotFoundError:   # blocks/frame/handover_proof.cpp's output has no such list
            theirs = None
            print("   entry.targets.tsv: none lies there: nothing is compared")
        if theirs is not None:
            ours = [(what, tc.dec(value)) for what, value in notes[1:]]
            differ = sum(a != b for a, b in zip(theirs, ours)) + abs(len(theirs) - len(ours))
            print("   entry.targets.tsv: %d values: %s" % (len(theirs), "all equal to the second verifier's, name by name, in the same order" if not differ else "%d DIFFER" % differ))
            bad += differ
        print("   trees of the link's chain (number, depth, width in F): " + ", ".join("(%d, %d, %d)" % (n, t[0], t[1]) for n, t in enumerate(che.trees)))
    except tc.Refused as why:
        print("entry.file: REFUSED by the second verifier: %s" % why)
        bad += 1
    print("ALL AGREES" if not bad else "%d THINGS DIFFER" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
