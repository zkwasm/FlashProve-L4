#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Stress test of the verifier program cp_verify against a parameter file (policy.txt) that the
model's provider wrote to harm anyone who uses it: can such a file crash cp_verify, or cost it much
time or memory? The script makes planning calls only (cp_verify -plan request), which print what
follows from the public inputs, read no file from a prover and check no proof. It writes and removes
files only in the work folder, which must not exist yet (it creates missing parent folders). With
--sanitized, cp_verify itself also leaves sanitizers' report files in reports/ beside the program.

usage: parameter_walk.py <cp_verify> <new work folder> <config folder> .. [--steps n] [--wide n] [--jobs n] [--seed s]
                         [--seconds n] [--mb n] [--sanitized] [--walk-only] [--whole] [--reach]

The script has three parts, run for each config folder ("the configuration"; `kit` in identifiers
and in the output). A call "plans" a file if it ends with status 4 (PLAN ONLY); a file counts as
planned if the call with -arrays (below) plans it.
forms   Copies of the configuration's policy.txt in a changed form: lines swapped or moved; odd
        bytes, such as a tab, 0xFF or a byte order mark; CR LF line ends; no final newline; an empty
        file; newlines only; numbers of up to 4,000 digits; lines of 1,022 to 1,025 bytes; the file
        twice; 8 MB of blank lines or of copies of the last line. Every neighbouring pair of lines
        is swapped; the other changes to a line are made to a sample of the lines.
        Some of these files are still valid, and cp_verify is right to plan them: two swapped lines
        of one kind (call, callA or callH), for example. Each planned file is listed ("form changed,
        and planned") for a person to assess, and is not counted. Skipped with --walk-only.
walk    A random search that stays among the planned files, since an early rejection exercises
        little code, and moves towards costly ones. Each of the --steps steps makes --wide copies of
        the current file and changes one to three numbers in each: to a value near it, to a multiple
        or half of it, or to a power of two up to 2^33 or its neighbour. Numbers in the call lines,
        seed_*, wb_* and the public tables' digests are never chosen. Then line d is set to H * dh,
        as cp_verify requires. A number is chosen more often the more of its changes all three calls
        (below) have planned. The next step starts from a planned copy, preferably (with --whole:
        only) one that all three calls planned. Among these it takes, in turn, the copy that cost
        the most memory, the most time, and the one with the least sum (below). Every 50 steps the
        walk restarts from the configuration's file. KL.bin (quantization scales) and layout.txt
        (where a proof places each array) are always the configuration's. So no call plans a file
        with another number of layers, only the call with -arrays plans one whose arrays have other
        sizes, and with --whole the walk changes neither.
reach   (--reach, in place of walk) Finds, for each line of a name and one number except the tables'
        digests, with the other lines as in the configuration, which of about 125 values below 2^32
        all three calls plan: 0 to 40, powers of two and their neighbours, values derived from the
        configuration's. It prints the smallest and largest such value, their number, greatest cost
        and least sum. Then, line by line and keeping each change, it sets the line to the costliest
        value that all three calls plan, and runs the resulting file ("all at their dearest").
The three calls on each file: -plan request with -arrays, which builds the sub-proofs' plans (one
sub-proof for each part of the forward pass) and needs no layout file; and -plan request with the
configuration's layout.txt, at 0 and at 128 requested bits ("asked" in the output), which also plans
key file, leaf file and proof. --whole and --reach need a configuration whose own file all three
calls plan; otherwise the script exits with status 2.
The least sum of a file that all three calls planned is the smallest x that the call at 128 bits
prints in a line "all rounds' errors added: 2^-x" (x counts proof of work if the file sets any).
cp_verify decides nothing by it; here it steers the walk towards weak files. If all three calls plan
the configuration's own file and no such line is found, the script exits with status 2: the pattern
SUMS no longer fits cp_verify's output.

What is counted; each such call is printed on one line, with its arguments:
fault      a signal ended the call, or its status is 15 (cp_verify could not write its output) or
           not one of cp_verify's own; with --sanitized also: a sanitizer's report file not about
           its bound on memory (the first 20 are listed at the end).
unchecked  the last line is "NOT CHECKED: ..", "NO: internal-.." or "NO: <sub-proof>:internal-..":
           cp_verify failed one of its internal checks, which no file should cause, or ran out of
           memory. With "memory" in that line or "bad_alloc" in the output: large.
slow       not finished within --seconds (default 60) of wall-clock time, with up to --jobs calls
           running at once.
large      more than --mb MB resident (default 2,048); with --sanitized (a build made by
           sanitized_walk_test.sh) also: stopped by the bound on memory linked into that build
           (status 97 with a sanitizer's message on memory). A plain build has a limit on address
           space of four times --mb; a call that it stops counts by how it ends (fault, unchecked,
           or large by the rule under unchecked).
           The MB figure is the system's peak for the child process, which is never below this
           script's own size when it starts the child (some tens of MB): read it as "at most".
At the end come the walk's costliest planned files ("dear") and those with the least sums; they stay
in the work folder, which the script neither empties nor removes. A planned file of the walk whose
call was printed as counted (above), and that ranked among them at an earlier step, is removed once
it drops out of the ranking: the path in its printed line then leads to no file.
Exit status 0: no fault. 1: a fault. 2: bad arguments, or SUMS fits no line (above). Unchecked,
slow and large calls are listed and do not change the status: cp_verify sets itself no limit on time
or memory, and this script measures what a file can cost. It shows nothing about files that it did
not make, and nothing about soundness.
"""
import argparse
import collections
import concurrent.futures
import glob
import os
import random
import re
import select
import struct
import subprocess
import sys
import time

# cp_verify's own exit statuses, except 15, which is a fault here: this script reads all the output.
OWN = (0, 1, 2, 3, 4, 5, 14)  # defined in docs/format/FORMAT.md, "14. Interface and results"
ODD = [b"\r", b"\t", b"\x0b", b"\x7f", b"\x80", b"\xff", b"\xef\xbb\xbf"]
MEMORY = "hard rss limit|allocation-size-too-big|out-of-memory"  # a sanitizer's words at its bound on memory
SUMS = r"(?m)^accounting:   all rounds' errors added: 2\^-([0-9.]+)(?: with the work,|$)"  # the second form: a file without proof of work
KINDS = ("fault", "unchecked", "slow", "large")
# Result of the three calls on one file. key: the call with -arrays planned it. whole: all three
# did. mb, seconds: the largest of the three. low: the least sum, or None. said: (kind, line to
# print) pairs.
Played = collections.namedtuple("Played", "path key whole mb seconds low said")


class Play:
    def __init__(self, a):
        self.a, self.v, self.work = a, os.path.abspath(a.program), os.path.abspath(a.work)
        self.reports = os.path.join(os.path.dirname(self.v), "reports")
        os.makedirs(os.path.dirname(self.work), exist_ok=True)
        os.mkdir(self.work, 0o700)  # fails if the name is taken; closed to other users
        self.counts, self.n = dict.fromkeys(KINDS, 0), 0

    def one(self, words):
        """Makes one call. Returns (kind or "", status, seconds, MB, last line, output). MB: peak
        resident memory, from wait4."""
        t0 = time.time()
        # A shell sets the limits, then execs the program. A sanitized program gets no limit on
        # address space: it reserves terabytes of it.
        limits = "ulimit -c 0; " + ("" if self.a.sanitized else "ulimit -v %d; " % (4 * self.a.mb << 10))
        p = subprocess.Popen(["/bin/sh", "-c", limits + 'exec "$0" "$@"', self.v] + words, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        end, got, slow = t0 + self.a.seconds, [], False
        while True:
            left = end - time.time()
            if left <= 0:
                slow = True
                p.kill()
                break
            if select.select([p.stdout], [], [], left)[0]:
                b = os.read(p.stdout.fileno(), 1 << 16)
                if not b:
                    break
                got = (got + [b])[-64:]
        p.stdout.close()
        _, st, ru = os.wait4(p.pid, 0)
        p.returncode = 0  # the child is reaped here; Popen must not wait for it again
        s, mb = time.time() - t0, ru.ru_maxrss >> 10
        if slow:
            return "slow", "none", s, mb, "", ""
        status = os.WEXITSTATUS(st) if os.WIFEXITED(st) else -os.WTERMSIG(st)
        text = b"".join(got).decode(errors="replace")
        last = (text.strip().splitlines() or [""])[-1][:100]
        if status not in OWN:
            own = "".join(open(f, errors="replace").read() for f in glob.glob(os.path.join(self.reports, "*.%d" % p.pid)))
            return "large" if status == 97 and re.search(MEMORY, own + text) else "fault", status, s, mb, last, ""
        if last.startswith("NOT CHECKED") or re.match(r"NO: ([A-Za-z0-9]+:)?internal-", last):
            return "large" if "memory" in last or "bad_alloc" in text else "unchecked", status, s, mb, last, ""
        return "large" if mb > self.a.mb else "", status, s, mb, last, text

    # more: KL.bin, layout.txt, then the options -prompt and -reply, each with its file.
    def three(self, path, more):
        calls = [["-plan", "request", path, more[0], "0"] + more[2:] + ["-arrays"]]
        calls += [["-plan", "request", path, more[0], bits, "-layout", more[1]] + more[2:] + ["-q"] for bits in ("0", "128")]
        key, whole, mb, sec, sums, said = False, True, 0, 0.0, [], []
        for words in calls:
            kind, status, s, m, last, text = self.one(words)
            key, whole = key or (words[-1] == "-arrays" and status == 4), whole and status == 4
            mb, sec = max(mb, m), max(sec, s)
            if status == 4 and words[4] == "128":
                sums += [float(x) for x in re.findall(SUMS, text)]
            if kind:
                said.append((kind, "%s | status %s | %.1f s | %d MB | %s | cp_verify %s" % (kind, status, s, m, last, " ".join(words))))
        return Played(path, key, whole, mb, sec, min(sums) if whole and sums else None, said)

    def play(self, pool, kit, more, batch):
        """Writes the files (name, bytes), makes the three calls on each, prints and counts, and
        removes the files that were neither planned nor listed."""
        paths = []
        for name, b in batch:
            self.n += 1
            paths.append(os.path.join(self.work, kit, "%06d.%s.txt" % (self.n, name)))
            with open(paths[-1], "wb") as f:
                f.write(b)
        got = list(pool.map(lambda path: self.three(path, more), paths))
        for g in got:
            for kind, line in g.said:
                self.counts[kind] += 1
                print(line, flush=True)
            if not g.said and not g.key:
                os.remove(g.path)
        return got


def forms(pol):
    """Yields (name, bytes) for each file of the part forms."""
    lines = pol.split(b"\n")[:-1]
    join = lambda ls: b"\n".join(ls) + b"\n"
    for i in range(len(lines) - 1):
        yield "swap-%d" % i, join(lines[:i] + [lines[i + 1], lines[i]] + lines[i + 2 :])
    for i in range(1, len(lines), 7):
        yield "first-%d" % i, join([lines[i]] + lines[:i] + lines[i + 1 :])
        yield "last-%d" % i, join(lines[:i] + lines[i + 1 :] + [lines[i]])
    for i in sorted({0, 1, 2, 3, len(lines) // 2, len(lines) - 1} | set(range(5, len(lines), 11))):
        w = lines[i].split(b" ", 1)
        for j, o in enumerate(ODD):
            for k, new in enumerate((o + lines[i], w[0] + o + b" " + w[1], w[0] + b" " + o + w[1], lines[i] + o)):
                yield "odd-%d-%d-%d" % (i, j, k), join(lines[:i] + [new] + lines[i + 1 :])
        for digits in (10, 11, 20, 4000):
            yield "digits-%d-%d" % (i, digits), join(lines[:i] + [w[0] + b" " + b"1" * digits] + lines[i + 1 :])
        for n in (1022, 1023, 1024, 1025):
            yield "long-%d-%d" % (i, n), join(lines[:i] + [w[0] + b" " + b"1" * (n - len(w[0]) - 1)] + lines[i + 1 :])
            yield "name-%d-%d" % (i, n), join(lines[:i] + [b"n" * (n - 2) + b" 1"] + lines[i + 1 :])
    yield "crlf", pol.replace(b"\n", b"\r\n")
    yield "no-last-newline", pol[:-1]
    yield "empty", b""
    yield "newlines", b"\n" * 4096
    yield "twice", pol + pol
    yield "blank-first", b"\n" + pol
    yield "blank-8mb", b"\n" * (8 << 20) + pol
    yield "after-8mb", pol + b"\n" * (8 << 20)
    yield "calls-8mb", pol + (lines[-1] + b"\n") * ((8 << 20) // (len(lines[-1]) + 1))


def moved(lines, rnd, weight):
    """Returns the lines with one to three numbers changed and d set to H * dh; the changes as text;
    and the changed numbers' names. weight: name -> relative chance of being chosen (default 1). A
    number's name is the words before its line's first number, plus "#" and the number's index among
    the line's words if the line has several."""
    new = [l.split(" ") for l in lines]
    at = {}
    for i, w in enumerate(new):
        places = [k for k, x in enumerate(w) if x.isdigit()]
        if w[0] != "call" and not w[0].startswith("call") and places and all(x.isdigit() for x in w[places[0] :]):
            for k in places:
                at[" ".join(w[: places[0]]) + ("#%d" % k if len(places) > 1 else "")] = (i, k)
    val = lambda n: int(new[at[n][0]][at[n][1]])

    def put(n, v):
        if n in at:
            new[at[n][0]][at[n][1]] = "%d" % max(0, v)

    names = [n for n in at if not re.fullmatch(r"seed_..|wb_.*|t(silu|sig|rope|exp)\d", n)]
    said, drawn = [], set(rnd.choices(names, [weight.get(n, 1.0) for n in names], k=rnd.choice((1, 1, 2, 3))))
    for n in sorted(drawn):
        v = val(n)
        p2 = 1 << rnd.randrange(0, 34)
        v2 = rnd.choice((v + 1, v + 1, v - 1, v + 2, v - 2, 2 * v, v // 2, 2 * v + 1, 4 * v, p2, p2 - 1, p2 + 1, v + 16, v + 256))
        put(n, v2)
        said.append("%s=%d" % (n, max(0, v2)))
    if "H" in at and "dh" in at:  # H dh = d
        put("d", val("H") * val("dh"))
    return [" ".join(w) for w in new], ",".join(said), drawn


def text_of(lines):
    return ("\n".join(lines) + "\n").encode()


def bits(low):
    return "none" if low is None else "%.2f" % low


# The part reach. joined: the changes to earlier lines are kept.
def reach(play, start):
    here = list(start)
    for joined in (False, True):
        for i, l in enumerate(start):
            if not re.fullmatch(r"\S+ \d+", l) or re.fullmatch(r"(t(silu|sig|rope|exp)\d) \d+", l):
                continue
            n, v = l.split(" ")[0], int(l.split(" ")[1])
            tries = set(range(41)) | {(1 << k) + d for k in range(33) for d in (-1, 0, 1)} | {v - 2, v - 1, v, v + 1, v + 2, 2 * v, 4 * v, v // 2}
            tries = sorted(x for x in tries if 0 <= x < 1 << 32)
            base = here if joined else start
            got = play([("reach-%s-%d" % (n, x), text_of(base[:i] + ["%s %d" % (n, x)] + base[i + 1 :])) for x in tries])
            for g in got:
                if os.path.exists(g.path) and not g.said:
                    os.remove(g.path)
            ok = [(x, g) for x, g in zip(tries, got) if g.whole]
            if joined and ok:
                here[i] = "%s %d" % (n, max(ok, key=lambda o: (round(o[1].mb / 16), round(o[1].seconds, 1), -(o[1].low or 999)))[0])
            elif ok:
                print("reach | %s | kit %d | planned %d .. %d (%d of %d values) | at most %d MB, %.1f s | least sum %s"
                      % (n, v, ok[0][0], ok[-1][0], len(ok), len(tries), max(g.mb for _, g in ok), max(g.seconds for _, g in ok),
                         bits(min((g.low for _, g in ok if g.low is not None), default=None))), flush=True)  # fmt: skip
            else:
                print("reach | %s | kit %d | none of %d values planned with the other lines as they now are" % (n, v, len(tries)), flush=True)
    (g,) = play([("all-at-their-dearest", text_of(here))])
    print("all at their dearest | planned by all three calls: %s | %d MB | %.1f s | least sum %s | changed lines: %s"
          % (g.whole, g.mb, g.seconds, bits(g.low), "; ".join(h for h, s0 in zip(here, start) if h != s0)), flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser(usage=__doc__.split("\n\n")[1][len("usage: ") :])
    ap.add_argument("program")
    ap.add_argument("work")
    ap.add_argument("kits", nargs="+")
    for name, default in (("steps", 200), ("wide", 16), ("jobs", 8), ("seconds", 60), ("mb", 2048)):
        ap.add_argument("--" + name, type=int, default=default)
    ap.add_argument("--seed", default="1")
    for name in ("sanitized", "walk-only", "whole", "reach"):
        ap.add_argument("--" + name, action="store_true")
    a = ap.parse_args()
    if os.path.lexists(os.path.abspath(a.work)):
        print("parameter_walk.py: the work folder exists already: %s" % a.work)
        sys.exit(2)
    if not os.access(a.program, os.X_OK):
        print("parameter_walk.py: no program that can be run: %s" % a.program)
        sys.exit(2)
    p = Play(a)
    if a.sanitized and (not os.path.isdir(p.reports) or os.listdir(p.reports)):
        print("parameter_walk.py: --sanitized needs an empty folder reports/ beside the program (sanitized_walk_test.sh's build)")
        sys.exit(2)
    dear, weak = set(), set()
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        for folder in a.kits:
            kit = os.path.basename(os.path.abspath(folder))
            pol = open(os.path.join(folder, "policy.txt"), "rb").read()
            T = 1 << int(re.search(rb"(?m)^lt (\d+)$", pol).group(1))
            os.makedirs(os.path.join(p.work, kit))
            req, y = os.path.join(p.work, kit, "req"), os.path.join(p.work, kit, "y")
            # Request file: T token ids (all 0), then its range r, e. Token file: the next token, y.
            open(req, "wb").write(struct.pack("<%dI" % (T + 2), *([0] * T + [T - 1, T])))
            open(y, "wb").write(struct.pack("<I", 0))
            more = [os.path.join(os.path.abspath(folder), "KL.bin"), os.path.join(os.path.abspath(folder), "layout.txt"), "-prompt", req, "-reply", y]
            play = lambda batch, kit=kit, more=more: p.play(pool, kit, more, batch)
            (own,) = play([("the-kit", pol)])
            print("%s: the kit's own file: planned by the call with -arrays: %s, by all three calls: %s; %d MB, %.1f s, least sum %s"
                  % (kit, own.key, own.whole, own.mb, own.seconds, bits(own.low)), flush=True)  # fmt: skip
            if (a.whole or a.reach) and not own.whole:
                print("parameter_walk.py: --whole and --reach need a kit whose own file all three calls plan, at 128 asked bits too: not %s" % kit)
                sys.exit(2)
            if own.whole and own.low is None:
                print("parameter_walk.py: all three calls planned %s's own file, and no line 'all rounds' errors added' was found in the accounting: the walk would steer by nothing: SUMS in this script no longer fits the line that blocks/numbers/accounting.h prints" % kit)
                sys.exit(2)
            if not a.walk_only:
                got = play(list(forms(pol)))
                print("%s forms: %d files, %d of them planned by the call with -arrays" % (kit, len(got), sum(g.key for g in got)), flush=True)
                for g in got:
                    if g.key:
                        print("form changed, and planned (by all three calls: %s); is it a good file all the same? | %s" % (g.whole, g.path))
            start = pol.decode().splitlines()
            if a.reach:
                reach(play, start)
                continue
            rnd = random.Random("%s %s" % (kit, a.seed))
            # all6: the number of copies that all three calls planned.
            here, inside, all6, weight = start, 0, 0, {}
            for step in range(a.steps):
                copies = [moved(here, rnd, weight) for _ in range(a.wide)]
                got = play([("walk-%d" % step, text_of(c[0])) for c in copies])
                for g, c in zip(got, copies):
                    for n in c[2]:
                        weight[n] = min(8.0, weight.get(n, 1.0) + 1.0) if g.whole else max(0.15, 0.85 * weight.get(n, 1.0))
                ok = [(g, c) for g, c in zip(got, copies) if g.key]
                inside += len(ok)
                all6 += sum(g.whole for g, _ in ok)
                before = {d[2] for d in dear} | {w[1] for w in weak}
                dear |= {(g.mb, round(g.seconds, 1), g.path, c[1]) for g, c in ok}
                dear = set(sorted(dear, reverse=True)[:5]) | set(sorted(dear, key=lambda d: (-d[1], -d[0]))[:5])
                weak = set(sorted(weak | {(g.low, g.path, c[1]) for g, c in ok if g.low is not None})[:5])
                kept = {d[2] for d in dear} | {w[1] for w in weak}
                order = (lambda o: (o[0].whole, o[0].mb, o[0].seconds), lambda o: (o[0].whole, o[0].seconds, o[0].mb), lambda o: (o[0].whole, -(o[0].low or 999)))
                best = max([o for o in ok if o[0].whole or not a.whole], key=order[step % 3], default=None)
                listed = {g.path for g, _ in ok if g.said}  # this step's alone: a file that an earlier step listed goes once it ranks no more
                for path in (before | {g.path for g, _ in ok}) - kept - listed:
                    os.remove(path)
                if best:
                    here = best[1][0]
                if step % 50 == 49:  # restart, so that one dead end does not trap the walk
                    here = start
            print("%s walk: %d steps of %d files, %d planned by the call with -arrays, %d by all three calls" % (kit, a.steps, a.wide, inside, all6), flush=True)
    for mb, s, path, said in sorted(dear, reverse=True):
        print("dear | %d MB | %.1f s | %s | last change %s" % (mb, s, path, said))
    for low, path, said in sorted(weak):
        print("least sum | %.2f bits at 128 asked | %s | last change %s" % (low, path, said))
    other = []
    if a.sanitized:
        other = [f for f in glob.glob(os.path.join(p.reports, "*")) if not re.search(MEMORY, open(f, errors="replace").read())]
        for f in other[:20]:
            print("report file that is not on the bound on memory | %s" % f)
    print("parameter_walk.py: %d files; %s%s" % (p.n, ", ".join("%s %d" % (k, p.counts[k]) for k in KINDS),
                                               "; %d report files that are not on the bound on memory" % len(other) if a.sanitized else ""))  # fmt: skip
    return 1 if p.counts["fault"] or other else 0


if __name__ == "__main__":
    sys.exit(main())
