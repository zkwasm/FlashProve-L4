#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Runs planning calls of cp_verify on altered input files; counts crashes and internal errors.

cp_verify is the verifier program; its planning calls (`-plan`) verify nothing. Each altered file is
meant to drive cp_verify out of an array or into undefined arithmetic. One of four files is altered
at a time: KL.bin (quantization scales), layout.txt (where a proof places each array), the request,
or the token file (`y`), which holds the next token. The parameter file, policy.txt, is never
altered: parameter_walk.py does that. The unaltered request and token file are generated here. No
file written by a prover is read, and no proof is checked.

The test is meant for a cp_verify built with the compiler's sanitizers; sanitized_walk_test.sh
runs it on such a build. On a build without them it still detects signals, failed assertions of the
C++ library and rejection names ("NO: <name>") with "internal-".

usage: extreme_files_test.py <cp_verify> <new work folder> <config folder> [--jobs <n>] [--every <k>]
<config folder> (`kit` in the code): policy.txt, KL.bin and layout.txt.
--jobs: altered files tried at a time (default 8). --every: selects every k-th word of KL.bin and
every 30 k-th line of layout.txt, and always the first and the last (default 1).

Every altered file gets a call "-plan request". The unaltered files, a KL.bin or layout.txt of
another length and a KL.bin with all words set also get "-plan accounting" on their folder. A word
is 32 bits, little-endian. A request is T tokens, then r and e, which bound the positions that a
proof is about. T is 2^`lt`; `lt` and V, the size of the vocabulary, come from policy.txt.
  KL.bin      cut short, or extended by bytes 00 or ff (lengths()); all words set to each of WORDS;
              each selected word set to 0 and to 2^32 - 1
  layout.txt  cut and extended likewise; each selected line removed, and doubled; each number of
              each selected line set to each of NUMBERS; a NUL byte; a line of 2^20 letters;
              line ends replaced by spaces
  request     cut and extended likewise; the first and the last token, r and e set to values at or
              next to 0, V, T, 2^31 and 2^32; some pairs (r, e) at the edges
  y           lengths from 0 to 4096 bytes; the word set to values at or next to 0, V, 2^31 and 2^32

What is counted (one printed line for each such call; totals in the last line):
  fault        killed by a signal; an exit status that is not cp_verify's own (the build of
               sanitized_walk_test.sh exits with 97 on a report), or 15 (see OWN); or output with
               a sanitizer's report or a failed assertion of the C++ library
  not-checked  status 5 (`NOT CHECKED`), or a last line "NO: internal-.." or
               "NO: <word>:internal-.."; cp_verify uses these names for faults in itself
  slow         not finished within 300 s
  honest       a call on the unaltered files, none of the above, with a status other than 4
               (`PLAN ONLY`)
Any other status of cp_verify's own passes. The last line counts 4 as "planned", 1 and 14 as
"refused", 2 as "usage".
The files of a counted call stay under <new work folder>; the others are removed.
Exit status 0: none of the four. 1: some. 2: a wrong call, or <new work folder> exists.
A run without a fault shows that these inputs caused no fault that this build can detect. It shows
nothing about other inputs, and nothing about soundness.
"""

import argparse
import concurrent.futures
import os
import re
import struct
import subprocess
import sys

# cp_verify's exit statuses (verifier/docs/output.md, "Exit codes"), except 15: the answer could not
# be written to standard output. This test reads all output, so 15 counts as a fault.
OWN = (0, 1, 2, 3, 4, 5, 14)
WORDS = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
NUMBERS = ("0", "4294967295", "18446744073709551615", "9" * 40, "-1")
LIMIT = 300


def cuts(n):
    return sorted(x for x in {0, 1, 3, 4, 5, 8, n - 4, n - 1} | {n * i // 16 for i in range(1, 16)} if 0 <= x < n)


def lengths(b):
    """Yields (name, bytes) for each cut or extended copy of b."""
    for x in cuts(len(b)):
        yield "cut-%d" % x, b[:x]
    for more in (1, 4, 4096, 1 << 20):
        yield "more-%d" % more, b + bytes(more)
        yield "more-ff-%d" % more, b + b"\xff" * more


# The indices below n that --every selects.
def taken(n, every):
    return sorted({0, n - 1} | set(range(0, n, every)))


def with_word(b, i, w):
    return b[: 4 * i] + struct.pack("<I", w) + b[4 * i + 4 :]


def main():
    ap = argparse.ArgumentParser(usage="extreme_files_test.py <cp_verify> <new work folder> <config folder> [--jobs <n>] [--every <k>]")
    ap.add_argument("program"), ap.add_argument("work"), ap.add_argument("kit")
    ap.add_argument("--jobs", type=int, default=8), ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    if os.path.lexists(os.path.abspath(a.work)) or a.jobs < 1 or a.every < 1 or not os.access(a.program, os.X_OK):
        ap.print_usage(sys.stderr)
        return 2
    program, work = os.path.abspath(a.program), os.path.abspath(a.work)
    own = {}
    for name in ("policy.txt", "KL.bin", "layout.txt"):
        with open(os.path.join(a.kit, name), "rb") as f:
            own[name] = f.read()
    word = dict(l.split() for l in own["policy.txt"].decode("ascii").split("\n") if len(l.split()) == 2)
    T, V = 1 << int(word["lt"]), int(word["V"])
    own["req"] = struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1) >> 7) % V for t in range(T)], T - 1, T)
    own["y"] = struct.pack("<I", 5 % V)
    calls = []  # (name, altered file, its bytes, expected status or None, "-plan accounting" too?)

    def add(name, which=None, b=None, want=None, folder_too=False):
        calls.append((name, which, b, want, folder_too))

    add("honest", want=4, folder_too=True)
    kl, lay, req = own["KL.bin"], own["layout.txt"], own["req"]
    for name, b in lengths(kl):
        add("KL." + name, "KL.bin", b, folder_too=True)
    for w in WORDS:
        add("KL.all=%d" % w, "KL.bin", struct.pack("<I", w) * (len(kl) // 4), folder_too=True)
    for i in taken(len(kl) // 4, a.every):
        for w in (0, 0xFFFFFFFF):
            add("KL.word-%d=%d" % (i, w), "KL.bin", with_word(kl, i, w))
    for name, b in lengths(lay):
        add("layout." + name, "layout.txt", b, folder_too=True)
    rows = lay.decode("ascii").split("\n")[:-1]
    text = lambda r: ("\n".join(r) + "\n").encode("ascii")
    for i in taken(len(rows), 30 * a.every):
        add("layout.line-%d.left-out" % i, "layout.txt", text(rows[:i] + rows[i + 1 :]))
        add("layout.line-%d.twice" % i, "layout.txt", text(rows[: i + 1] + rows[i:]))
        for j, m in enumerate(re.finditer(r"\d+", rows[i])):
            for v in NUMBERS:
                add("layout.line-%d.number-%d=%s" % (i, j, v[:20]), "layout.txt", text(rows[:i] + [rows[i][: m.start()] + v + rows[i][m.end() :]] + rows[i + 1 :]))
    add("layout.nul", "layout.txt", lay.replace(b"\n", b"\0\n", 1))
    add("layout.long-line", "layout.txt", b"x" * (1 << 20) + b"\n" + lay)
    add("layout.no-line-ends", "layout.txt", lay.replace(b"\n", b" "))
    for name, b in lengths(req):
        add("request." + name, "req", b)
    for i in (0, T - 1, T, T + 1):
        for w in sorted({0, 1, V - 1, V, T - 1, T, T + 1, 0x80000000, 0xFFFFFFFF}):
            add("request.word-%d=%d" % (i, w), "req", with_word(req, i, w))
    for r, e in ((0, 0), (0, 1), (0, T), (T, T), (T, T + 1), (T - 1, T - 1), (5, 3), (0xFFFFFFFF, 0), (0xFFFFFFFF, 0xFFFFFFFF), (0, 0xFFFFFFFF)):
        add("request.r=%d.e=%d" % (r, e), "req", with_word(with_word(req, T, r), T + 1, e))
    for n in (0, 1, 3, 5, 8, 4096):
        add("y.bytes-%d" % n, "y", bytes(n))
    for w in sorted({0, V - 1, V, 0x80000000, 0xFFFFFFFF}):
        add("y=%d" % w, "y", struct.pack("<I", w))

    # Runs one call. Returns "fault", "not-checked", "slow" or "status-<n>", and a line to print.
    def ended(words):
        try:
            r = subprocess.run([program] + words, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False, timeout=LIMIT)
        except subprocess.TimeoutExpired:
            return "slow", "no end within %d s" % LIMIT
        lines = r.stdout.strip().split("\n")
        sign = next((l for l in lines if "runtime error:" in l or "ERROR: AddressSanitizer" in l or ("Assertion '" in l and "' failed" in l)), None)
        if r.returncode not in OWN or sign:
            return "fault", "status %d | %s" % (r.returncode, (sign or lines[-1])[:200])
        if r.returncode == 5 or re.match(r"NO: (\w+:)?internal-", lines[-1]):
            return "not-checked", "status %d | %s" % (r.returncode, lines[-1][:200])
        return "status-%d" % r.returncode, lines[-1][:120]

    # Runs the calls on one set of files.
    def one(numbered):
        n, (name, which, b, want, folder_too) = numbered
        folder = os.path.join(work, "%05d" % n)
        os.makedirs(folder)
        for f, own_bytes in own.items():
            with open(os.path.join(folder, f), "wb") as out:
                out.write(b if f == which else own_bytes)
        at = lambda f: os.path.join(folder, f)
        plays = [("request", ["-plan", "request", at("policy.txt"), at("KL.bin"), "128", "-layout", at("layout.txt"), "-prompt", at("req"), "-reply", at("y"), "-q"])]
        if folder_too:
            plays.append(("accounting", ["-plan", "accounting", folder, "128"]))
        seen = []
        for call, words in plays:
            kind, said = ended(words)
            if want is not None and kind != "status-%d" % want:
                kind = "honest" if kind.startswith("status-") else kind
            seen.append((name + "." + call, kind, said))
        if all(k.startswith("status-") for _, k, _ in seen):
            for f in own:
                os.remove(at(f))
            os.rmdir(folder)  # the files of a call that is counted stay
        return seen

    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    counts = {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        for seen in pool.map(one, enumerate(calls)):
            for name, kind, said in seen:
                counts[kind] = counts.get(kind, 0) + 1
                if not kind.startswith("status-"):
                    print("%s | %s | %s" % (kind, name, said), flush=True)
    bad = {k: counts.get(k, 0) for k in ("fault", "not-checked", "slow", "honest")}
    print("extreme_files_test.py: %d calls on %d files: planned %d, refused %d, usage %d; fault %d, not-checked %d, slow %d, honest calls not as expected %d"
          % (sum(counts.values()), len(calls), counts.get("status-4", 0), counts.get("status-1", 0) + counts.get("status-14", 0), counts.get("status-2", 0),
             bad["fault"], bad["not-checked"], bad["slow"], bad["honest"]))  # fmt: skip
    return 1 if any(bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
