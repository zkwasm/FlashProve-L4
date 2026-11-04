#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Compares examples/quickstart-gpu.sh with section 3 of docs/quickstart.md (the commands, the
expected result, the compiler version, the heading), prover/check_nvcc_version.sh (the compiler
version) and prover/build.sh (the first words of the warning that the script quotes).
It only reads files, so it needs no GPU.

The section runs from the heading that begins "## 3." to the next heading of that level. Its
commands are the lines of all its fenced code blocks, without lines that begin "unset": the script
cleans its environment itself. The script's list (`commands`) must match them, in the same order
except for the line with "mktemp -d", which the script runs first.

usage: quickstart_gpu_test.py
Exit status: 0 if all is as expected, 1 if not, 2 if a file cannot be read.
"""

import os
import re
import sys

ME = "quickstart_gpu_test.py"
REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def check(name, good, seen):
    results.append(bool(good))
    print(f"{name}: {seen}: {'as expected' if good else 'not as expected'}")


def text_of(path):
    try:
        with open(os.path.join(REPOSITORY, path), encoding="utf-8") as f:
            return f.read()
    except OSError as fault:
        print(f"{ME}: {path} cannot be read: {fault.strerror}", file=sys.stderr)
        sys.exit(2)


def but_mktemp(lines):
    return [line for line in lines if "mktemp -d" not in line]


def assigned(script, name):
    """The value that the script assigns to this variable, up to the first blank."""
    return "".join(re.findall(rf"^{name}=(\S+)", script, re.M))


def main():
    script = text_of("examples/quickstart-gpu.sh")
    page = text_of("docs/quickstart.md")
    section = "".join(re.findall(r"^## 3\..*?(?=^## |\Z)", page, re.M | re.S))
    fence = "```"
    blocks = re.findall(rf"^{fence}[a-z]*\n(.*?)^{fence}$", section, re.M | re.S)
    wanted = [line for line in "".join(blocks).splitlines() if not line.startswith("unset ")]
    blocks = re.findall(r"^mapfile -t commands <<'END'\n(.*?)^END$", script, re.M | re.S)
    listed = "".join(blocks).splitlines()
    check("the-document-prints-the-same-commands",
          wanted and sorted(wanted) == sorted(listed) and but_mktemp(wanted) == but_mktemp(listed),
          f"{len(wanted)} lines in the section's blocks (without 'unset'), {len(listed)} in the "
          f"script's list; in only one of the two: "
          f"{' | '.join(sorted(set(wanted) ^ set(listed))) or 'none'} (if none and yet not as "
          f"expected: the order differs, or a line is repeated)")
    verdict = assigned(script, "good_verdict")
    flat = " ".join(section.split())  # the document may break a line inside the words looked for
    absent = [text for text in ("exit code 0", f"`{verdict}`") if text not in flat]
    check("the-document-states-the-expected-result", verdict and not absent,
          f"the script expects exit status 0 and {verdict!r}; words not found in the section: "
          f"{' | '.join(absent) or 'none'}")
    texts = {"the script": script, "the section": section,
             "check_nvcc_version.sh": text_of("prover/check_nvcc_version.sh")}
    named = {name: set(re.findall(r"nvcc\s+(\d+\.\d+\.\d+)", text)) for name, text in texts.items()}
    check("the-three-files-name-the-same-compiler",
          len(named["the script"]) == 1 and len(set(map(frozenset, named.values()))) == 1,
          "; ".join(f"nvcc in {name}: {' '.join(sorted(found)) or 'none'}"
                    for name, found in named.items()))
    cited = "".join(re.findall(r"""^section='docs/quickstart\.md, "(.*)"'$""", script, re.M))
    check("the-script-cites-the-section's-heading", cited and section.startswith(f"## {cited}\n"),
          f"the script cites {cited!r}; the section begins {section[:60].splitlines()[:1]}")
    words = "".join(re.findall(r"^no_run_path='(.*)'$", script, re.M))
    check("the-build-prints-the-warning-that-the-script-quotes",
          words and f'echo "{words} ' in text_of("prover/build.sh"),
          f"the script looks in the build's output for a line that begins {words!r}; "
          f"prover/build.sh should have an echo of a line that begins so")
    print(f"{ME}: {sum(results)} as expected, {len(results) - sum(results)} otherwise")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
