#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Checks that the two names of cp_verify's NOT CHECKED that may be the machine's fault agree in the places that give them.

    verifier/tests/not_checked_names.py

No GPU, no build, about a second. Prints ALL AS EXPECTED and exits 0, or prints what differs and
exits 1.

The verifier program cp_verify prints NOT CHECKED: <name> as its last line when it has itself failed
and has verified nothing. Under internal-stage and internal-exception-parFor the machine may be the
cause, and verifier/README.md says to run the call again. The names are separate literals at each
place (listed at what_differs()). A third such name, or one renamed, would have one place tell a
reader to run the call again where another place says nothing of it.

Not shown: that cp_verify uses these names only when the machine is the cause (it does not:
verifier/README.md); that the usage text and verifier/README.md give no third such name elsewhere;
what the program does with the line «const bool machine = ..»: it is read, not its use.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(os.path.dirname(HERE))
NAMES = ("internal-stage", "internal-exception-parFor")
PRODUCED = (("verifier/src/cp_verify.cpp", 'const string &name = "%s")' % NAMES[0], 1), ("verifier/src/cp_verify.cpp", 'failedName = "%s"' % NAMES[0], 1),
                               ("verifier/src/vf_common.h", ' = "%s";' % NAMES[1], 2))


def text(path):
    return open(os.path.join(TOP, path), encoding="utf-8").read()


# Cuts each line at its first "//", even inside a string; "/* */" comments remain.
def code_of(source):
    return "\n".join(line.split("//")[0] for line in source.split("\n"))


def what_differs(read):
    """Returns one message per mismatch. read(path) returns a file's text.

    The places: (1) cpNotChecked() in verifier/src/cp_verify.cpp, which chooses the sentence printed
    before the last line; it must compare exactly the two NAMES with the part of the printed name
    after its last colon, so that the prefix of a sub-proof, as in G0:internal-f-join-place, makes no
    difference. (2) That program's usage text and (3) the table in verifier/README.md, "Last line and
    exit status", must each contain, exactly once, a fixed phrase with both names. (4) Where the names
    are produced: changed only there, the other places would still agree.
    """
    differs = []

    def once(path, sentence):
        if read(path).count(sentence) != 1:
            differs.append("%s: the words «%s» stand %d times, not once" % (path, sentence, read(path).count(sentence)))

    code = code_of(read("verifier/src/cp_verify.cpp"))
    picks = re.findall(r"(?m)^  const bool machine = (.*);$", code)
    if len(picks) != 1 or sorted(re.findall(r'"([^"]+)"', picks[0])) != sorted(NAMES) or picks[0] != " || ".join('bare == "%s"' % name for name in NAMES):
        differs.append("verifier/src/cp_verify.cpp: the line «const bool machine = ..» compares %s, not the bare name with each of the two once" % (picks or "nothing"))
    if code.count("const string bare = name.substr(name.rfind(':') + 1);") != 1:
        differs.append("verifier/src/cp_verify.cpp does not take the part of the name behind the last colon")
    once("verifier/src/cp_verify.cpp", "%s and %s the machine may be the cause" % NAMES)
    once("verifier/README.md", "Run it again where the name is `%s` or `%s`" % NAMES)
    for path, words, times in PRODUCED:
        if code_of(read(path)).count(words) != times:
            differs.append("%s: the words «%s» stand %d times in code, not %d" % (path, words, code_of(read(path)).count(words), times))
    if "NOT CHECKED: " not in read("verifier/src/cp_verify.cpp"):
        differs.append("verifier/src/cp_verify.cpp prints no line «NOT CHECKED: ..»")
    return differs


# Negative controls (`TEETH`): (file, text that occurs in it once, replacement). main() repeats the
# check on text altered in memory (no file is written); each must be reported.
TEETH = (
    ("verifier/src/cp_verify.cpp", 'bare == "internal-stage"', 'bare == "internal-sums-line"'),
    ("verifier/src/cp_verify.cpp", 'bare == "internal-stage" || bare == "internal-exception-parFor"', 'bare == "internal-stage" || bare == "internal-exception-parFor" || bare == "internal-sums-line"'),
    ("verifier/src/cp_verify.cpp", ' || bare == "internal-exception-parFor"', ""),
    ("verifier/src/cp_verify.cpp", "const string bare = name.substr(name.rfind(':') + 1);", "const string bare = name;"),
    ("verifier/src/cp_verify.cpp", 'const string &name = "internal-stage")', 'const string &name = "internal-stages")'),
    ("verifier/src/cp_verify.cpp", 'failedName = "internal-stage"', 'failedName = "internal-stages"'),
    ("verifier/src/vf_common.h", 'badName1 = "internal-exception-parFor";', 'badName1 = "internal-exception-parfor";'),
    ("verifier/src/vf_common.h", 'badName = "internal-exception-parFor";', 'badName = "internal-exception-parfor";'),
    ("verifier/src/cp_verify.cpp", "internal-stage and internal-exception-parFor the machine", "internal-stage and internal-sums-line the machine"),
    ("verifier/README.md", "the name is `internal-stage` or `internal-exception-parFor`", "the name is `internal-stage` or `internal-sums-line`"),
)


def main():
    differs = what_differs(text)
    for path, words, turned in TEETH:
        if text(path).count(words) != 1:
            differs.append("the teeth: %s: the words «%s» stand %d times, not once" % (path, words, text(path).count(words)))
        elif not what_differs(lambda p: text(p).replace(words, turned) if p == path else text(p)):
            differs.append("the teeth: %s read with «%s» in the place of «%s»: not noticed" % (path, turned, words))
    for d in differs:
        print("DIFFERS   " + d)
    print("the teeth: %d altered readings" % len(TEETH))
    print("ALL AS EXPECTED" if not differs else "%d THINGS DIFFER" % len(differs))
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
