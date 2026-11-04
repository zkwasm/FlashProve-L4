#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Measures the steady rate of a proving service (cp_prove -spool) from outside the program.

usage: python3 driver/measure_rate.py --programs DIR --isk DIR --ivk DIR --config DIR --id HEX
                                      --work NEW_DIRECTORY [options] (--help lists the options)

It makes N times B requests of T tokens, each asking only for the token after its window, and
submits all of them to a new spool directory, so that every batch is full. It starts the production
build's cp_prove once, for N batches of B requests, and notes from outside when each request leaves
in/ and when its proof appears in out/. After cp_prove has ended, it has every proof checked by a
separate, whole call of cp_verify, with the token that the response itself names. It calls both
programs by their documented options (docs/service.md, "Options of the service"; docs/verifying.md,
"Run the check") and takes no figure from what they print. What a run costs and leaves behind, what
the script never does and what a run does not show: driver/README.md, "The rate".

The steady rate. A batch's release is the latest change time (st_ctime) of its proof files in out/.
Only the kernel sets that time, at the moment when the service gives a file its final name. Take
the intervals between the releases of successive batches. The steady rate is B times T tokens
divided by their mean, cut after two decimal places. The script prints every release and every
interval, so the rate can be checked by hand. No rate is given if a batch was not full, a request
failed or got no response, not every proof was accepted, cp_prove's exit code was not 0, or a
signal to stop came before the figures.

The windows come from one of three places, and no window is sent twice.
  - By default the script draws them: token t of window w is word t mod 8 (little-endian, 32 bits)
    of SHA-256("measure_rate window", then the seed in 8 bytes, w in 4 and t div 8 in 4,
    little-endian), reduced modulo the vocabulary size.
  - With --text FILE they are text. FILE is a NumPy file (format 1.0, little-endian int32, C order)
    of n rows of T token ids. Window w is row w mod n. From window n on, its first id stays and its
    other T - 1 ids are rotated to the left by 100 times (w div n) places. Only the rows that the
    windows need are read.
  - With --requests DIR they are the request files of that directory.
Whether a batch's time depends on its tokens is not measured.

Linux only. It needs no package outside the standard library and parses under Python 3.8.

Exit code 0: the rates were printed. 1: cp_prove was started, or was about to be, and no rate is
given; the last lines say why. 2: the call was refused, and nothing was written."""

import argparse
import ast
import collections
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True  # else importing spool.py would write a cache outside the work directory
import spool  # noqa: E402

ME = "measure_rate.py"
TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = ("policy.txt", "KL.bin", "layout.txt")
LOOK_S = 0.1  # the pause between two listings of the spool directory
GPU_S = 1.0  # the pause between two samples of the GPU's memory
GPU_WAIT_S = 10  # a call of nvidia-smi that has not answered after so many seconds is given up
APART_NS = 2 * 10**9  # two readings of the clock at one release that differ by more are reported
IN_USE = "refused (cp-isk): the folder named with -isk is in use"
PLACES_A_TURN = 100  # each further pass through the rows of --text rotates them by this many more places
MOST_REQUESTS = 100000  # with more than these in in/, cp_prove need not take the oldest first (docs/service.md, "How a request becomes a response")

# One response as it was first seen. seen_ns: this script's clock right after the listing that showed it.
Answer = collections.namedtuple("Answer", "failed changed_ns modified_ns seen_ns size")

asked = []  # the numbers of the signals that asked this script to stop


def refuse(why):
    """Prints why on standard error and exits with code 2. Only for use before cp_prove is started."""
    print("%s: %s" % (ME, why), file=sys.stderr)
    sys.exit(2)


def cut(numerator, denominator, places):
    """Returns numerator / denominator (the latter positive) in decimal, cut toward 0 after that many places."""
    scaled = abs(numerator) * 10**places // denominator
    return "%s%d.%0*d" % ("-" if numerator < 0 else "", scaled // 10**places, places, scaled % 10**places)


def seconds(ns, places=3):
    return cut(ns, 10**9, places)


def steady_rate(releases_ns, requests, tokens):
    """Applies the rule of the steady rate (the module's head) to the batches' release times.

    Returns the intervals in ns, how many there are, their sum, the rate in tokens a second and the
    seconds a request, both as cut decimal text. Raises ValueError for fewer than three releases or
    releases that do not increase."""
    intervals = [later - earlier for earlier, later in zip(releases_ns, releases_ns[1:])]
    if len(intervals) < 2 or min(intervals) <= 0:
        raise ValueError("the rule needs at least three releases, each later than the one before")
    counted = len(intervals)
    total = sum(intervals)
    return {
        "intervals_ns": intervals,
        "counted": counted,
        "sum_ns": total,
        "tokens_a_second": cut(requests * tokens * counted * 10**9, total, 2),
        "seconds_a_request": cut(total, counted * requests * 10**9, 3),
    }


def clocks_apart(clocks_ns):
    """Returns the pairs of names in {name: ns} whose times differ by more than APART_NS."""
    names = sorted(clocks_ns)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if abs(clocks_ns[a] - clocks_ns[b]) > APART_NS]


def window(seed, number, tokens, vocabulary):
    """Returns the token ids of one window, by the rule in the module's head."""
    words = []
    for block in range((tokens + 7) // 8):
        words += struct.unpack("<8I", hashlib.sha256(b"measure_rate window" + struct.pack("<QII", seed, number, block)).digest())
    return [word % vocabulary for word in words[:tokens]]


def text_window(rows, number):
    """Returns the token ids of one window from the rows of --text, by the rule in the module's head."""
    row, turn = list(rows[number % len(rows)]), number // len(rows)
    places = PLACES_A_TURN * turn % max(1, len(row) - 1)
    return row[:1] + row[1 + places :] + row[1 : 1 + places]


def batches_of(answers, left_ns):
    """Groups responses into batches and returns them in the order of their releases.

    The service gives all files of a batch one modification time, so that time says which responses
    belong together; it enters no figure. answers: {id: Answer}; left_ns: {id: when the request was
    seen to have left in/}."""
    groups = collections.defaultdict(list)
    for name, answer in answers.items():
        groups[answer.modified_ns].append(name)
    batches = [
        {
            "ids": sorted(names),
            "started_ns": min(left_ns[name] for name in names),
            "last_started_ns": max(left_ns[name] for name in names),
            "first_release_ns": min(answers[name].changed_ns for name in names),
            "release_ns": max(answers[name].changed_ns for name in names),
            "modified_ns": modified,
            "seen_ns": max(answers[name].seen_ns for name in names),
        }
        for modified, names in groups.items()
    ]
    return sorted(batches, key=lambda batch: batch["release_ns"])


def batch_faults(batches, requests, asked_batches):
    """Returns one sentence for each fault: a wrong number of batches, a batch without exactly `requests`
    requests, two batches whose starts or releases overlap."""
    faults = []
    if len(batches) != asked_batches:
        faults.append("the number of batches seen is %d, not %d" % (len(batches), asked_batches))
    for number, batch in enumerate(batches, 1):
        if len(batch["ids"]) != requests:
            faults.append("batch %d did not have exactly %d requests: it had %d" % (number, requests, len(batch["ids"])))
    for number, (one, next_one) in enumerate(zip(batches, batches[1:]), 1):
        if one["release_ns"] >= next_one["first_release_ns"] or one["last_started_ns"] > next_one["started_ns"]:
            faults.append("batches %d and %d cannot be told apart: their starts or their releases overlap" % (number, number + 1))
    return faults


def ranges(numbers):
    """Returns sorted whole numbers as text such as 0-11,24-35."""
    runs = []
    for n in sorted(numbers):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in runs)


def numbers_in(text):
    """Returns the numbers of a list such as 0-11,24-35, as the kernel prints a node's CPUs."""
    found = set()
    for part in filter(None, text.split(",")):
        first, _, last = part.partition("-")
        found.update(range(int(first), int(last or first) + 1))
    return found


def first_line(path):
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return f.readline().strip()
    except OSError:
        return None


def numbers_of(path):
    """Returns {first word, without its colon: the whole number after it} for the lines of a file such as /proc/meminfo."""
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            lines = [line.split() for line in f]
    except OSError:
        return {}
    return {words[0].rstrip(":"): int(words[1]) for words in lines if len(words) >= 2 and words[1].isdigit()}


def memory_total_limit_available():
    """Returns the machine's memory, the container's limit and the memory available now, in bytes; None for what is not found.
    Available: the smaller of Linux's MemAvailable and the limit less what is in use under it, file cache not counted as in
    use; the limit itself if the use cannot be read. Only a limit in /sys/fs/cgroup is seen."""
    machine, group = numbers_of("/proc/meminfo"), "/sys/fs/cgroup/"
    limit = next((int(t) for t in map(first_line, (group + "memory.max", group + "memory/memory.limit_in_bytes")) if t and t.isdigit() and int(t) < 2**60), None)
    available = machine["MemAvailable"] * 1024 if "MemAvailable" in machine else None
    if limit is not None and available is not None:
        used, cache = first_line(group + "memory.current"), numbers_of(group + "memory.stat")
        known = used and used.isdigit() and "inactive_file" in cache and "active_file" in cache
        available = min(available, max(0, limit - max(0, int(used) - cache["inactive_file"] - cache["active_file"])) if known else limit)
    return machine["MemTotal"] * 1024 if "MemTotal" in machine else None, limit, available


class Report:
    """Prints each line and writes it to an open file at once. Neither a reader that has gone away nor a
    full disk stops a run in which cp_prove may be proving. lost: why the file is not whole, or None."""

    def __init__(self, file):
        self.file, self.lost = file, None

    def say(self, *lines):
        for line in lines or ("",):
            try:
                print(line, flush=True)
            except OSError:  # What is still buffered goes to /dev/null too: else Python fails on it at its exit, with code 120.
                nowhere = os.open(os.devnull, os.O_WRONLY)
                os.dup2(nowhere, sys.stdout.fileno())
                os.close(nowhere)
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except OSError as why:
                self.lost = self.lost or why.strerror

    def close(self):
        try:
            self.file.close()
        except OSError:  # `lost` has it already; unclosed, Python would print the fault at its exit
            pass


class GpuWatch(threading.Thread):
    """Asks nvidia-smi about one GPU; as a thread, samples the memory in use until `over` is set. samples: [(ns, MiB)].

    gave_up: None, or (ns, process ID) of a call that had not answered after GPU_WAIT_S seconds. Such a call gets no signal:
    it is left to end by itself, and nvidia-smi is not called again."""

    def __init__(self, tool, which):
        super().__init__(daemon=True)
        self.tool, self.which, self.samples, self.missed, self.over, self.gave_up = tool, which, [], 0, False, None

    def ask(self, fields):
        """Returns nvidia-smi's values of these fields, or None if it gives no clear answer."""
        command = [self.tool, "-i", self.which, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        try:  # not subprocess.run(), which kills its child at a time limit or if Ctrl-C interrupts it
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
            lines = child.communicate(timeout=GPU_WAIT_S)[0].decode("ascii", "replace").strip().splitlines()
        except OSError:
            return None
        except subprocess.TimeoutExpired:
            self.gave_up = (time.time_ns(), child.pid)
            return None
        cells = [cell.strip() for cell in lines[0].split(",")] if len(lines) == 1 else []
        return cells if child.returncode == 0 and len(cells) == len(fields) else None

    def run(self):
        while self.tool and not self.over and not self.gave_up:
            cells = self.ask(["memory.used"])
            if cells and cells[0].isdigit():
                self.samples.append((time.time_ns(), int(cells[0])))
            else:
                self.missed += 1
            time.sleep(GPU_S)


def arguments():
    p = argparse.ArgumentParser(
        prog=ME,
        description="Measures the steady rate of a proving service from outside the program. Each batch uses up one of the commitment's batches for good. Measure on synthetic weights or a public model's files: with secret weights a "
        "commitment made for the measurement is a further commitment to the same weights, which our judgement on weight hiding covers only on thin evidence and on conditions of its own. driver/README.md, \"The rate\", says what else a run costs, what it leaves behind and what it does not show.",
    )
    p.add_argument("--programs", required=True, metavar="DIR", help="the directory with the production build's cp_prove and cp_verify")
    p.add_argument("--isk", required=True, metavar="DIR", help="the secret key directory, by its absolute path; only passed on to cp_prove")
    p.add_argument("--ivk", required=True, metavar="DIR", help="the public key directory")
    p.add_argument("--config", required=True, metavar="DIR", help="the configuration directory: policy.txt, KL.bin and layout.txt")
    p.add_argument("--id", required=True, metavar="HEX", help="the commitment ID for cp_verify, 64 hex digits, from your own record of what cp_commit printed")
    p.add_argument("--work", required=True, metavar="NEW_DIRECTORY", help="the work directory: one that does not exist yet, or an empty one of yours with mode 0700")
    p.add_argument("--batch", type=int, default=24, metavar="B", help="requests in a batch (cp_prove -batch); default 24")
    p.add_argument("--in-step", type=int, metavar="K", help="requests in one pass over the weights (cp_prove -in-step); default 6, or B if B is smaller")
    p.add_argument("--batches", type=int, default=5, metavar="N", help="batches (cp_prove -most-batches), at least 3, and N times B at most %d; default 5" % MOST_REQUESTS)
    p.add_argument("--memory", default="lean,tensor-rooms=8", metavar="WORDS", help="cp_prove -memory; default lean,tensor-rooms=8; an empty text leaves the option out")
    p.add_argument("--least-memory", type=int, metavar="GB", help="refuse to start unless so many GB (10^9 bytes) of host memory are available, on the machine and under the "
                   "container's limit: a cp_prove that the system kills for want of memory ends in the worst way; default: no such check")  # fmt: skip
    p.add_argument("--public-weights", metavar="FILE", help="cp_prove -public-weights, for a commitment that was made with that option: nothing is hidden")
    p.add_argument("--huge-pages", action="store_true", help="cp_prove -huge-pages")
    p.add_argument("--text", metavar="FILE", help="make the windows from this file's rows of token ids instead of drawing them: a NumPy file (format 1.0, "
                   "little-endian int32, C order) of rows of T ids; the script's head has the rule")  # fmt: skip
    p.add_argument("--requests", metavar="DIR", help="take the requests from this directory's files, in the order of their names, instead of making them: "
                   "at least N times B different files, each T tokens, then r = T - 1 and e = T, as little-endian 32-bit words")  # fmt: skip
    p.add_argument("--seed", type=int, default=1, metavar="n", help="the seed of the windows that the script draws, 0 to 2^64 - 1; default 1")
    p.add_argument("--checks-at-once", type=int, metavar="n", help="calls of cp_verify that run side by side, each with -j 4; "
                   "at most the CPUs that this process may use; default: a quarter of them, at least 1 and at most 6")  # fmt: skip
    return p.parse_args()


def settings_of(config):
    """Returns T, the vocabulary size and the largest batch (or None) from the parameter file."""
    try:
        with open(os.path.join(config, "policy.txt"), encoding="ascii", errors="replace") as f:
            word = dict(line.split() for line in f if len(line.split()) == 2)
        lt, vocabulary, slots = int(word["lt"]), int(word["V"]), int(word["slots"]) if "slots" in word else None
    except (OSError, KeyError, ValueError):
        lt, vocabulary, slots = -1, 0, None
    if not (0 <= lt <= 16 and vocabulary >= 2):
        refuse("%s/policy.txt is not a parameter file that this script can use: it needs a line lt with a whole number up to 16 and a line V "
               "with one from 2 on (verifier/docs/parameter-file.md). Nothing was written" % spool.printable(config))  # fmt: skip
    return 1 << lt, vocabulary, slots


def id_of_the_files(o):
    """Returns the commitment ID that cp_verify computes from --config and --ivk's key.bin, or None.

    This says nothing of --isk: a secret key directory of another commitment is found only by the
    checks after the run."""
    words = [os.path.join(o.programs, "cp_verify"), "-plan", "id", o.config, os.path.join(o.ivk, "key.bin")]
    try:  # not subprocess.run(), which kills its child at a time limit or if Ctrl-C interrupts it
        child = subprocess.Popen(words, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        printed = child.communicate(timeout=300)[0].decode("ascii", "replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.findall(r"^commitment ID: ([0-9a-f]{64})$", printed, re.M)
    return found[0] if child.returncode == 4 and len(found) == 1 else None


def check_the_call(o):
    """Refuses a call that cp_prove or cp_verify would refuse later. Reads files, writes nothing. Returns T and V."""
    for name in ("programs", "ivk", "config", "public_weights", "text", "requests"):
        setattr(o, name, getattr(o, name) and os.path.abspath(getattr(o, name)))
    for program in ("cp_prove", "cp_verify"):
        path = os.path.join(o.programs, program)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            refuse("%s is not a program that you can run: --programs names the directory that the build printed in its last line. Nothing was written" % spool.printable(path))
    for name in SETTINGS:
        if not os.path.isfile(os.path.join(o.config, name)):
            refuse("%s has no file %s: --config names the configuration directory that the commitment was made with. Nothing was written" % (spool.printable(o.config), name))
    if not os.path.isfile(os.path.join(o.ivk, "key.bin")):
        refuse("%s has no file key.bin: --ivk names the commitment's public key directory. Nothing was written" % spool.printable(o.ivk))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", o.id):
        refuse("--id takes the commitment ID: 64 hex digits, without blanks. Nothing was written")
    if not o.isk.startswith("/"):
        refuse("--isk takes an absolute path: cp_prove refuses any other. Nothing was written")
    tokens, vocabulary, slots = settings_of(o.config)
    if not 1 <= o.batch <= min(64, slots or 64):
        refuse("--batch takes 1 to %d: cp_prove takes at most 64, and at most the number on the parameter file's line slots. Nothing was written" % min(64, slots or 64))
    if o.in_step is None:
        o.in_step = min(6, o.batch)
    if not 1 <= o.in_step <= min(16, o.batch):
        refuse("--in-step takes 1 to 16, and at most --batch. Nothing was written")
    if o.batches * o.batch > MOST_REQUESTS:
        refuse("--batches times --batch may be at most %d requests: each batch uses up one of the commitment's batches for good, and with more requests "
               "waiting cp_prove need not take the oldest first. Nothing was written" % MOST_REQUESTS)  # fmt: skip
    if o.batches < 3:
        refuse("--batches takes at least 3: the first batch has no interval before it, so fewer would leave fewer than two intervals to count. Nothing was written")
    if o.text and o.requests:
        refuse("--text and --requests cannot be given together: each says where the windows come from. Nothing was written")
    if not 0 <= o.seed < 2**64:
        refuse("--seed takes 0 to 2^64 - 1. Nothing was written")
    available = memory_total_limit_available()[2]
    if o.least_memory is not None and (available is None or available < o.least_memory * 10**9):
        refuse("%s, and --least-memory asks for %d GB. Stop other programs of your own, raise the container's memory limit or use a larger machine. Nothing was written"
               % ("only %s GB of host memory are available, on the machine or under the container's limit" % cut(available, 10**9, 1) if available is not None
                  else "how much host memory is available could not be read from /proc/meminfo", o.least_memory))  # fmt: skip
    cpus = len(os.sched_getaffinity(0))
    if o.checks_at_once is None:
        o.checks_at_once = max(1, min(6, cpus // 4))
    if not 1 <= o.checks_at_once <= cpus:
        refuse("--checks-at-once takes 1 to %d, the number of CPUs that this process may use. Nothing was written" % cpus)
    if id_of_the_files(o) != o.id.lower():
        refuse("cp_verify gives these files another commitment ID than --id, or none: see --id, --ivk and --config, or run %s/cp_verify -plan id %s %s/key.bin. Nothing was written"
               % (spool.printable(o.programs), spool.printable(o.config), spool.printable(o.ivk)))  # fmt: skip
    return tokens, vocabulary


def open_regular(path):
    """Opens a regular file for reading and returns it with its size. Anything else raises OSError."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)  # O_NONBLOCK: to open a named pipe would otherwise wait for a writer
    about = os.fstat(descriptor)
    if not stat.S_ISREG(about.st_mode):
        os.close(descriptor)
        raise OSError(errno.EINVAL, "it is not a regular file")
    return os.fdopen(descriptor, "rb"), about.st_size


def rows_of(path, tokens, vocabulary, most):
    """Returns the first `most` rows of the file given with --text, each a tuple of `tokens` ids below
    `vocabulary`, and the number of rows in the file. Refuses any other file, before it reads the rows."""
    kind = "a NumPy file of format 1.0 that holds little-endian int32 ('<i4') in C order, in rows of %d" % tokens
    try:
        f, size = open_regular(path)
        with f:
            start = f.read(10)
            text = f.read(struct.unpack("<H", start[8:])[0]) if start[:8] == b"\x93NUMPY\x01\x00" and len(start) == 10 else b""
            try:
                head = ast.literal_eval(text.decode("latin-1"))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                head = None
            shape = head.get("shape") if isinstance(head, dict) else None
            if not (shape and head.get("descr") == "<i4" and head.get("fortran_order") is False and isinstance(shape, tuple)
                    and [type(n) for n in shape] == [int, int] and shape[0] >= 1 and shape[1] == tokens and size == 10 + len(text) + 4 * shape[0] * tokens):  # fmt: skip
                refuse("%s, given with --text, is not %s. Nothing was written" % (spool.printable(path), kind))
            count = min(shape[0], most)
            body = f.read(4 * count * tokens)
    except OSError as why:
        refuse("%s, given with --text, cannot be read (%s). Nothing was written" % (spool.printable(path), why.strerror))
    if len(body) != 4 * count * tokens:
        refuse("%s, given with --text, became shorter while it was read. Nothing was written" % spool.printable(path))
    ids = struct.unpack("<%di" % (count * tokens), body)
    if min(ids) < 0 or max(ids) >= vocabulary:
        refuse("%s, given with --text, holds a token id that is not from 0 to %d, as the parameter file's line V asks. Nothing was written" % (spool.printable(path), vocabulary - 1))
    return [ids[row * tokens : (row + 1) * tokens] for row in range(count)], shape[0]


def requests_of(o, tokens, vocabulary):
    """Returns N times B different requests, as the bytes that cp_verify takes, and a line on where they come from."""
    count = o.batches * o.batch
    if o.text:
        rows, n = rows_of(o.text, tokens, vocabulary, count)
        made = [spool.request(text_window(rows, number), tokens - 1, tokens) for number in range(count)]
        if len(set(made)) != count:
            refuse("the %d rows of %s do not give %d different windows by this script's rule. No window is sent twice. Nothing was written" % (n, spool.printable(o.text), count))
        return made, ("%d windows from the %d rows of %s (sha256 %s): window w is row w mod %d; from window %d on, its first id stays and its other %d ids are "
                      "rotated to the left by %d x (w div %d) places" % (count, n, spool.printable(o.text), spool.sha256(o.text), n, n, tokens - 1, PLACES_A_TURN, n))  # fmt: skip
    if not o.requests:
        made = [spool.request(window(o.seed, number, tokens, vocabulary), tokens - 1, tokens) for number in range(count)]
        if len(set(made)) != count:
            refuse("the seed %d does not give %d different windows. No window is sent twice: name another with --seed. Nothing was written" % (o.seed, count))
        return made, "%d windows of %d token ids below %d, drawn by this script's rule from the seed %d" % (count, tokens, vocabulary, o.seed)
    try:
        names = sorted(os.listdir(o.requests))
    except OSError as why:
        refuse("the directory %s, given with --requests, cannot be listed (%s). Nothing was written" % (spool.printable(o.requests), why.strerror))
    if len(names) < count:
        refuse("%s holds %d files, and %d batches of %d need %d. No window is sent twice. Nothing was written" % (spool.printable(o.requests), len(names), o.batches, o.batch, count))
    taken = []
    for name in names[:count]:
        try:
            with open_regular(os.path.join(o.requests, name))[0] as f:
                data = f.read(4 * (tokens + 2) + 1)
        except OSError as why:
            refuse("%s in %s cannot be read (%s). Nothing was written" % (spool.printable(name), spool.printable(o.requests), why.strerror))
        words = struct.unpack("<%dI" % (len(data) // 4), data[: len(data) // 4 * 4])
        if len(data) != 4 * (tokens + 2) or max(words[:tokens]) >= vocabulary or words[tokens:] != (tokens - 1, tokens):
            refuse("%s in %s is not a request that this script takes: %d token ids below %d, then r = %d and e = %d, as little-endian 32-bit words. Nothing was written"
                   % (spool.printable(name), spool.printable(o.requests), tokens, vocabulary, tokens - 1, tokens))  # fmt: skip
        taken.append(data)
    if len(set(taken)) != count:
        refuse("two of the first %d files in %s hold the same request. No window is sent twice. Nothing was written" % (count, spool.printable(o.requests)))
    return taken, "the first %d of the %d files in %s, in the order of their names" % (count, len(names), spool.printable(o.requests))


def tell_the_machine(report, o, work, gpu_watch):
    """Prints the properties of the machine that can change a run's speed, and returns them for figures.json."""
    fields = ["name", "driver_version", "memory.total", "memory.used", "pci.bus_id", "uuid"]
    tool, which = gpu_watch.tool, gpu_watch.which
    cells = gpu_watch.ask(fields) if tool else None
    gpu = dict(zip(fields, map(spool.printable, cells))) if cells else None
    # sysfs names a PCI device with a domain of four hex digits in lower case, nvidia-smi with eight in upper case
    node = first_line("/sys/bus/pci/devices/%s/numa_node" % gpu["pci.bus_id"].lower()[-12:]) if gpu else None
    node_cpus = first_line("/sys/devices/system/node/node%s/cpulist" % node) if node and node.isdigit() else None
    mine = os.sched_getaffinity(0)
    total, limit, available = memory_total_limit_available()
    room = os.statvfs(work)
    sums = {name: spool.sha256(os.path.join(o.programs, name)) for name in ("cp_prove", "cp_verify")}
    sums.update((name, spool.sha256(os.path.join(o.config, name))) for name in SETTINGS)
    found = {
        "gpu": gpu,
        "gpu_asked_as": which,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "cpus": ranges(mine),
        "gpu_numa_node": node,
        "gpu_numa_node_cpus": node_cpus,
        "container_memory_limit_bytes": limit,
        "machine_memory_bytes": total,
        "available_memory_bytes": available,
        "free_bytes_for_the_work_directory": room.f_bavail * room.f_frsize,
        "transparent_hugepage": first_line("/sys/kernel/mm/transparent_hugepage/enabled"),
        "version": first_line(os.path.join(TOP, "VERSION")),
        "sha256": sums,
    }
    report.say(
        "THE MACHINE",
        "GPU: %s; driver %s; %s MiB of memory, %s MiB in use now" % (gpu["name"], gpu["driver_version"], gpu["memory.total"], gpu["memory.used"]) if gpu
        else "GPU: not known: %s gave no clear answer about GPU %s%s" % (spool.printable(tool), spool.printable(which), given_up(gpu_watch, None)) if tool
        else "GPU: not known: nvidia-smi was not found on PATH, so the GPU is not described and its memory is not sampled",
        "  CUDA_VISIBLE_DEVICES is %s, CUDA_DEVICE_ORDER is %s" % tuple(spool.printable(os.environ[n]) if n in os.environ else "not set" for n in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")),
    )  # fmt: skip
    if tool and not gpu_watch.gave_up:
        report.say("  %s was asked about GPU %s, and is asked for its memory in use once a second while cp_prove runs" % (spool.printable(tool), spool.printable(which)))
    if tool and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        report.say("  nvidia-smi numbers GPUs by PCI bus; CUDA does so only with CUDA_DEVICE_ORDER=PCI_BUS_ID. On a machine with GPUs of",
                   "  different models the programs may take another GPU than the one described and sampled here. To be sure, set",
                   "  CUDA_DEVICE_ORDER=PCI_BUS_ID")  # fmt: skip
    report.say(
        "CPUs that this process may use: %d (%s)" % (len(mine), ranges(mine)),
        "the GPU's NUMA node: " + ("%s (CPUs %s); %d of this process's CPUs are on it" % (node, node_cpus, len(mine & numbers_in(node_cpus))) if node_cpus else "not known"),
        "host memory: %s; this container's limit: %s; available now, with file cache counted as available: %s"
        % tuple(cut(n, 10**9, 1) + " GB" if n is not None else none for n, none in ((total, "not known"), (limit, "none found"), (available, "not known"))),
        "transparent huge pages (/sys/kernel/mm/transparent_hugepage/enabled): %s; -huge-pages: %s"
        % (found["transparent_hugepage"] or "not known", "given" if o.huge_pages else "not given"),
        "free room on the work directory's file system, where the proofs will lie: %s GB" % cut(room.f_bavail * room.f_frsize, 10**9, 1),
        "version (the file VERSION beside this script's directory): %s" % (found["version"] or "not known"),
    )
    report.say(*("sha256 %s  %s" % (digest, name) for name, digest in sums.items()))
    if os.geteuid() == 0:
        report.say("This script runs as root, and so will cp_prove and cp_verify.")
    return found


def given_up(gpu_watch, started_ns):
    """Returns the words on a call of nvidia-smi that was given up, or no words. started_ns: cp_prove's start, or None before it."""
    if not gpu_watch.gave_up:
        return ""
    before = started_ns is None or gpu_watch.gave_up[0] < started_ns
    return "; %s a call of nvidia-smi (process %d) had not answered after %g s: the script left it alone and calls nvidia-smi no more, so %s" % (
        "before cp_prove's start" if before else "%s s after cp_prove's start" % seconds(gpu_watch.gave_up[0] - started_ns, 1),
        gpu_watch.gave_up[1], GPU_WAIT_S, "the GPU's memory is not sampled" if before else "a later peak is not seen")  # fmt: skip


def tell_before_the_start(report, o):
    report.say(
        "",
        "BEFORE THE START",
        "- This run uses up %d of the commitment's batches for good. cp_commit's line \"the limit:\" says how many a leaf table serves." % o.batches,
        "- Measure on a commitment that was made for the measurement and serves nobody, on synthetic weights or a public model's\n  files: with secret weights it is a further commitment to the same weights, which our judgement on weight hiding\n  covers only on thin evidence and on conditions of its own (driver/README.md, \"The rate\").",
        "- This script reads nothing in the secret key directory, so it cannot see whether a service is using it. It relies on",
        "  cp_prove's own lock: a second cp_prove on a directory in use does not start. That lock cannot see a cp_prove on a copy",
        "  of the directory, or one on another machine that shares it, unless the file system applies the lock between machines.",
        "  It may also miss one that reaches the directory by another route on the same machine (through a container's layer,",
        "  from a virtual machine, or by a second mount). See docs/service.md, \"What cp_prove does at startup\".",
        "- The work directory will hold the requests, the responses and both programs' output. Keep it as secret as the secret",
        "  key directory: this script removes none of it.",
        "- All times are this machine's own clock.",
        "- To stop early, press Ctrl-C once. The script then creates the service's stop file and waits for cp_prove to end by",
        "  itself. Do not kill cp_prove: any end but exit code 0 puts the commitment under docs/faults.md, \"Always first\".",
    )


def command_of(o, spool_dir):
    words = [os.path.join(o.programs, "cp_prove"), "-isk", o.isk, "-spool", spool_dir,
             "-batch", str(o.batch), "-in-step", str(o.in_step), "-most-batches", str(o.batches)]  # fmt: skip
    words += ["-memory", o.memory] if o.memory else []
    words += ["-public-weights", o.public_weights] if o.public_weights else []
    return words + (["-huge-pages"] if o.huge_pages else [])


def ended(child):
    """Returns (exit code, or minus the signal's number; largest resident set in KiB) once the child has ended, else None."""
    pid, status, usage = os.wait4(child.pid, os.WNOHANG)
    if not pid:
        return None
    child.returncode = -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
    return child.returncode, usage.ru_maxrss


def look_at(spool_dir, name, now):
    for ending in ("proof", spool.FAILED):
        try:
            about = os.lstat(spool.path_of(spool_dir, name, ending))
        except FileNotFoundError:
            continue
        return Answer(ending == spool.FAILED, about.st_ctime_ns, about.st_mtime_ns, now, about.st_size)
    return None


def watch(report, child, spool_dir, ids, started_ns):
    """Looks at in/ and out/ until the child has ended, and once more after that.

    Returns {id: when it was seen to have left in/}, {id: Answer}, the result of ended() and the
    time at which the end was seen. If a signal has asked this script to stop, it keeps a stop file
    in place: a service that is still starting removes one without obeying it."""
    waiting, left_ns, answers, told = set(ids), {}, {}, False
    while True:
        end = ended(child)
        end_ns = time.time_ns()
        still_in = {name[:-4] for name in os.listdir(os.path.join(spool_dir, "in")) if name.endswith(".req")}
        now = time.time_ns()
        if len(waiting) == len(ids) and waiting - still_in:
            report.say("%s s after cp_prove's start: the first batch has started" % seconds(now - started_ns, 1))
        left_ns.update((name, now) for name in waiting - still_in)
        waiting &= still_in
        new = (spool.answered(spool_dir) & set(ids)) - set(answers)
        now = time.time_ns()
        seen = {name: look_at(spool_dir, name, now) for name in new}
        answers.update((name, answer) for name, answer in seen.items() if answer)
        if new:
            report.say("%s s after cp_prove's start: %d of %d requests have a response" % (seconds(now - started_ns, 1), len(answers), len(ids)))
        if end:
            return left_ns, answers, end, end_ns
        if asked and not os.path.lexists(os.path.join(spool_dir, "stop")):
            spool.stop(spool_dir)
            if not told:
                report.say(
                    "",
                    "%s: this script was asked to stop. It has created the stop file, %s." % (name_of(asked[0]), os.path.join(spool_dir, "stop")),
                    "cp_prove takes no further batch, finishes the batches that it has taken and ends by itself, which can take as long",
                    "as two batches. The script waits for that, sends no signal and will give no rate.",
                )
            told = True
        time.sleep(LOOK_S)


def stand_by(child, why, log, spool_dir):
    """For a fault of this script's own (why) at or after the start of cp_prove (child, or None if there is none).
    Says on standard error whether cp_prove runs, waits for its end without a limit and says how it ended."""

    def tell(*lines):
        try:
            print(*lines, sep="\n", file=sys.stderr, flush=True)
        except OSError:  # nobody reads any more; the wait below goes on
            pass

    fault = spool.printable("%s: %s" % (type(why).__name__, why))
    if child is None:
        return tell("", "%s: this script has failed (%s). cp_prove was not started, so none of the commitment's batches is used up." % (ME, fault), "The fault in full:")
    stop = os.path.join(spool_dir, "stop")
    if child.returncode is None:
        tell(
            "",
            "%s: this script has failed (%s), and cp_prove is still running, as process %d." % (ME, fault, child.pid),
            "cp_prove ends by itself after its batches. Do not kill it, and start no other cp_prove on this commitment: any end but",
            "exit code 0 puts the commitment under docs/faults.md, \"Always first\". Its output goes to %s." % log,
            "To stop it early, press Ctrl-C once or create the empty file %s." % stop,
            "The script gives no rate any more. It only waits for cp_prove's end, sends no signal, and then says how cp_prove ended.",
        )
    try:
        while child.returncode is None and not ended(child):
            if asked and not os.path.lexists(stop):
                try:
                    spool.stop(spool_dir)
                except OSError:  # tried again at the next turn
                    pass
            time.sleep(LOOK_S)
    except ChildProcessError:  # the system has collected the child itself, as it does where SIGCHLD is ignored
        return tell("", "%s: cp_prove (process %d) has ended, and the system did not keep its exit code. It was 0 if the last line of" % (ME, child.pid),
                    "%s is \"p258 end 0\". If not, read docs/faults.md, \"Always first\", before you start cp_prove on this commitment again." % log,
                    "The script's own fault in full:")  # fmt: skip
    tell(
        "",
        "%s: cp_prove (process %d) has ended with %s." % (ME, child.pid, end_in_words(child.returncode)),
        *(["Do not start cp_prove on this commitment again, and remove nothing in the work directory, before you have read",
           "docs/faults.md, \"Always first\" and \"What you see, and what to do\"."] if child.returncode else []),
        "The script's own fault in full:",
    )  # fmt: skip


def verdict_in(path):
    """Returns the verdict in the last line of what cp_verify printed (verifier/README.md, "Last line and exit status"), or "none".
    Only the fixed words and a fault's name are passed on: the rest of a line can hold values of the parameter file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    found = re.fullmatch(r"(YES|NOT-A-VERIFICATION|(?:NO|NOT CHECKED): [A-Za-z0-9_:-]{1,60}|WITHIN LIMITS(?=: ).*)", lines[-1] if lines else "")
    return "none" if not found else "WITHIN LIMITS" if found.group().startswith("WITHIN") else found.group()


def own_folders(work):
    """Returns the environment for cp_prove and cp_verify: this process's, with HOME and TMPDIR below the work directory."""
    return dict(os.environ, HOME=os.path.join(work, "home"), TMPDIR=os.path.join(work, "tmp"))


def check_the_proofs(report, o, work, spool_dir, proved, heard):
    """Runs one whole call of cp_verify on each (number, id) of proved, o.checks_at_once side by side.

    heard: how many of the signals in `asked` do not stop the checks. Returns {number: {"exit_code", "verdict", "ns"}}. A proof is
    missing from it if a further signal came before the proof's call was started; the calls that run then are waited for."""
    waiting, running, results = list(proved), {}, {}
    while running or (waiting and len(asked) == heard):
        while waiting and len(running) < o.checks_at_once and len(asked) == heard:
            number, name = waiting.pop(0)
            command = [os.path.join(o.programs, "cp_verify"), o.config, o.ivk, os.path.join(work, "requests", "%04d.req" % number),
                       spool.path_of(spool_dir, name, "y"), spool.path_of(spool_dir, name, "proof"), "-id", o.id.lower(), "-next-token", "-j", "4"]  # fmt: skip
            # cwd: where core files are written, one of a cp_verify that crashes would hold a request and a proof
            with open(os.path.join(work, "checks", "%04d.txt" % number), "xb") as out, open(os.path.join(work, "checks", "%04d.err.txt" % number), "xb") as err:
                running[number] = (subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, cwd=work, env=own_folders(work)), time.time_ns())
        for number, (child, began_ns) in list(running.items()):
            if child.poll() is not None:
                results[number] = {"exit_code": child.returncode, "verdict": verdict_in(os.path.join(work, "checks", "%04d.txt" % number)), "ns": time.time_ns() - began_ns}
                del running[number]
                if len(results) % 24 == 0 or not (running or waiting):
                    report.say("cp_verify has ended on %d of %d proofs" % (len(results), len(proved)))
        time.sleep(0.02)
    return results


def least_mean_most(values, unit, places):
    """Returns "least a, mean b, most c" for whole numbers, each divided by unit and cut."""
    return "least %s, mean %s, most %s" % (cut(min(values), unit, places), cut(sum(values), unit * len(values), places), cut(max(values), unit, places))


def end_in_words(code):
    return "exit code %d" % code if code >= 0 else "signal %d" % -code


def name_of(number):
    return "Ctrl-C (SIGINT)" if number == signal.SIGINT else signal.Signals(number).name


def tell_a_bad_end(report, code, log):
    """Prints how cp_prove ended, its last lines and what to read next."""
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    report.say("", "cp_prove ended with %s. The last lines of its output (%s):" % (end_in_words(code), log))
    report.say(*("  | " + spool.printable(line) for line in lines[-3:]))
    if any(line.startswith(IN_USE) for line in lines):
        report.say(
            "The secret key directory is in use: cp_prove did not start, so nothing was measured. The refusal says nothing against",
            "the process that is running there. Measure on a commitment of its own, never on a copy of this directory",
            "(docs/faults.md, \"refused (cp-isk)\"). Measure on synthetic weights or a public model's files: with secret weights such\na commitment is a further commitment to the same weights, which our judgement on weight hiding covers only\non thin evidence and on conditions of its own (driver/README.md, \"The rate\").",
        )
    else:
        report.say(
            "Do not start cp_prove on this commitment again, and remove nothing in the work directory, before you have read",
            "docs/faults.md, \"Always first\" and \"What you see, and what to do\".",
        )


def tell_the_times(report, batches, started_ns, ended_ns):
    report.say(
        "",
        "THE TIMES, in seconds since 1970 by this machine's clock",
        "cp_prove was started at %s and seen ended at %s." % (seconds(started_ns, 9), seconds(ended_ns, 9)),
        "start: when this script saw the batch's first request gone from in/. RELEASE: the latest change time of the batch's files.",
        "Two more readings of that clock at the release enter no figure: stamp, the modification time that cp_prove itself gives",
        "the files, and seen, this script's own reading when it first saw the batch's last response.",
        "%5s %8s  %-20s  %-20s  %-20s  %-20s" % ("batch", "requests", "start", "RELEASE", "stamp", "seen"),
    )
    notes = []
    for number, batch in enumerate(batches, 1):
        report.say("%5d %8d  %s  %s  %s  %s" % (number, len(batch["ids"]), *(seconds(batch[key], 9) for key in ("started_ns", "release_ns", "modified_ns", "seen_ns"))))
        clocks = {"RELEASE": batch["release_ns"], "stamp": batch["modified_ns"], "seen": batch["seen_ns"]}
        notes += ["NOTE: batch %d: %s and %s lie %s s apart, more than 2 s." % (number, one, other, seconds(abs(clocks[one] - clocks[other]))) for one, other in clocks_apart(clocks)]
    if notes:
        report.say(*notes, "These should be readings of one clock: find out why they differ before you rely on a figure of this run.")


def tell_the_rates(report, o, tokens, batches, started_ns):
    """Prints the three rates, with what the steady rate rests on, and returns them for figures.json."""
    steady = steady_rate([batch["release_ns"] for batch in batches], o.batch, tokens)
    all_tokens = o.batches * o.batch * tokens
    rates = dict(
        steady,
        from_the_first_batch=cut(all_tokens * 10**9, batches[-1]["release_ns"] - batches[0]["started_ns"], 2),
        with_start_up=cut(all_tokens * 10**9, batches[-1]["release_ns"] - started_ns, 2),
    )
    report.say(
        "STEADY RATE: %s tokens a second, that is %s seconds a request" % (steady["tokens_a_second"], steady["seconds_a_request"]),
        "  %d requests a batch, %d in one pass over the weights; %d tokens a request, all of them counted; each proof checks one" % (o.batch, o.in_step, tokens),
        "  token, the one after its window; -public-weights %s" % ("given: nothing is hidden" if o.public_weights else "not given"),
        "  the intervals between releases, s: " + " ".join(seconds(interval, 9) for interval in steady["intervals_ns"]),
        "  their mean: %s s; %d x %d tokens / that mean = the rate, cut after two places and not rounded" % (cut(steady["sum_ns"], steady["counted"] * 10**9, 9), o.batch, tokens),
        "rate from the first batch's start to the last release: %s tokens a second" % rates["from_the_first_batch"],
        "rate with start-up, from cp_prove's start to the last release: %s tokens a second" % rates["with_start_up"],
    )
    return rates


def keep_out_of_the_given_directories(o, work):
    """Refuses a work directory that is, or lies in, a directory that the call names. The secret key directory's path is
    compared as text alone, so that nothing is looked at there; cp_prove refuses a link in that path anyway."""
    given = [("--isk", os.path.normpath(o.isk))]
    for option, folder in (("--ivk", o.ivk), ("--config", o.config), ("--programs", o.programs), ("--requests", o.requests)):
        given += [(option, form) for form in (folder, os.path.realpath(folder))] if folder else []
    for option, folder in given:
        if (work + "/").startswith(folder.rstrip("/") + "/"):
            refuse("the work directory %s is, or lies in, the directory given with %s. This script makes many files in its work directory and removes none: "
                   "name a work directory outside the directories that the call names, for example beside them. Nothing was written" % (spool.printable(work), option))  # fmt: skip


def main():
    o = arguments()
    tokens, vocabulary = check_the_call(o)
    work = os.path.abspath(o.work)
    keep_out_of_the_given_directories(o, work)
    requests, whence = requests_of(o, tokens, vocabulary)
    found = spool.others_can_change(os.path.dirname(work))
    if found:
        refuse("the directory above the work directory, %s, does not exist. Nothing was written" % spool.printable(found[0]) if found[1] == spool.GONE
               else spool.refusal("work directory", o.work, *found))  # fmt: skip
    work = os.path.join(os.path.realpath(os.path.dirname(work)), os.path.basename(work))
    keep_out_of_the_given_directories(o, work)
    if not re.fullmatch(r"[A-Za-z0-9/._-]{1,390}", work):
        refuse("the work directory's full path, %s, must have at most 390 characters, all of them letters, digits or / . _ -: cp_prove "
               "accepts its HOME and TMPDIR, which lie below it, only in this form. Nothing was written" % spool.printable(work))  # fmt: skip
    os.umask(0o077)
    try:
        os.mkdir(work, 0o700)
    except FileExistsError:
        about = os.lstat(work)
        if not stat.S_ISDIR(about.st_mode) or about.st_uid != os.geteuid() or about.st_mode & 0o777 != 0o700 or os.listdir(work):
            refuse("the work directory %s exists and is not an empty directory of yours with mode 0700. This script writes only into a new or "
                   "empty directory and removes nothing that it finds there: name one that does not exist yet. Nothing was written" % spool.printable(work))  # fmt: skip
    except OSError as why:
        refuse("the work directory %s cannot be made (%s). Nothing was written" % (spool.printable(work), why.strerror))
    spool_dir = os.path.join(work, "spool")
    for name in ("spool", "home", "tmp", "requests", "checks"):
        os.mkdir(os.path.join(work, name), 0o700)
    report = Report(open(os.path.join(work, "report.txt"), "x", encoding="utf-8"))
    tool = shutil.which("nvidia-smi")
    which = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
    gpu = GpuWatch(tool, which)
    record = {"machine": tell_the_machine(report, o, work, gpu), "requests_from": whence}

    report.say("", "THE REQUESTS", whence + ".", "Each asks only for the token after its window (r = %d, e = %d)." % (tokens - 1, tokens))
    keys = {}
    numbers = {}
    for number, data in enumerate(requests):
        with open(os.path.join(work, "requests", "%04d.req" % number), "xb") as f:
            f.write(data)
        name, keys[name] = spool.lay(spool_dir, data, with_key=True)
        numbers[name] = number
    spool.private_folders(spool_dir, "out")
    report.say("All %d are submitted to %s; this script's copies are in %s." % (len(requests), spool_dir, os.path.join(work, "requests")))

    tell_before_the_start(report, o)
    command = command_of(o, spool_dir)
    log = os.path.join(work, "cp_prove.txt")
    report.say("", "THE SERVICE", "HOME=%s TMPDIR=%s \\" % (os.path.join(work, "home"), os.path.join(work, "tmp")), "  " + " ".join(map(shlex.quote, command)), "Its output goes to %s." % log)
    gpu.start()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda received, frame: asked.append(received))
    if asked:
        raise KeyboardInterrupt
    started_ns = time.time_ns()
    child = None
    try:
        with open(log, "xb") as out:
            # A session of its own: the terminal's Ctrl-C and hang-up go to this script alone.
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=own_folders(work))  # fmt: skip
        report.say("cp_prove is process %d." % child.pid)
        left_ns, answers, (code, resident_kib), ended_ns = watch(report, child, spool_dir, numbers, started_ns)
    except BaseException as why:  # any fault at all: cp_prove may be running, and whoever started the script must be told
        stand_by(child, why, log, spool_dir)
        raise
    gpu.over = True
    if code:
        tell_a_bad_end(report, code, log)

    why_not = []
    heard = len(asked)  # the signals up to here do not stop the checks; a later one lets no further check start
    if code:
        why_not.append("cp_prove ended with %s, not with exit code 0" % end_in_words(code))
    failed = sorted(numbers[name] for name, answer in answers.items() if answer.failed)
    if failed:
        why_not.append("%d of the requests failed (an empty file <id>.failed in out/): numbered %s" % (len(failed), ranges(failed)))
    if len(answers) < len(numbers):
        why_not.append("%d of the requests got no response: numbered %s" % (len(numbers) - len(answers), ranges(n for name, n in numbers.items() if name not in answers)))
    faulty = {}
    for name, answer in answers.items():
        faults = spool.read_member(spool_dir, name, sums=False, key=keys[name]).faults
        if faults:
            faulty[name] = faults
            why_not.append("the response to request %d is not as the service promises: %s" % (numbers[name], "; ".join(faults)))
    batches = batches_of(answers, left_ns)
    why_not += batch_faults(batches, o.batch, o.batches) if answers else []

    proved = sorted((numbers[name], name) for name, answer in answers.items() if not answer.failed and name not in faulty)
    if proved:
        report.say("", "THE CHECKS", "%d proofs, one whole call of cp_verify each, %d side by side, each with -j 4; their output goes to %s." % (len(proved), o.checks_at_once, os.path.join(work, "checks")))
    checks_began_ns = time.time_ns()
    checks = check_the_proofs(report, o, work, spool_dir, proved, heard)
    checks_ns = time.time_ns() - checks_began_ns
    accepted = sorted(n for n, c in checks.items() if c["exit_code"] == 0 and c["verdict"] == "YES")
    for number in sorted(set(checks) - set(accepted)):
        why_not.append("cp_verify did not accept the proof of request %d: %s, verdict %s; its output is in checks/%04d.txt. Keep that response as secret as the secret key directory"
                       % (number, end_in_words(checks[number]["exit_code"]), checks[number]["verdict"], number))  # fmt: skip
    if len(checks) < len(proved):
        why_not.append("%d proofs were not checked, because a signal asked this script to stop" % (len(proved) - len(checks)))

    if batches:
        tell_the_times(report, batches, started_ns, ended_ns)
    figures = {"served": len(answers) - len(failed), "failed": len(failed), "accepted": len(accepted), "not_accepted": len(checks) - len(accepted)}
    if asked:  # the last look: a signal that comes later changes nothing
        why_not.insert(0, "%s asked this script to stop" % name_of(asked[0]))
    report.say("", "THE FIGURES")
    if not why_not:
        figures.update(tell_the_rates(report, o, tokens, batches, started_ns))
    if batches:
        report.say(
            "start-up, from cp_prove's start to the first batch's start: %s s" % seconds(batches[0]["started_ns"] - started_ns, 1),
            "a request's wait, from its batch's start to its release, s: " + least_mean_most([batch["release_ns"] - batch["started_ns"] for batch in batches], 10**9, 1),
            "from the last release to cp_prove's end: %s s" % seconds(ended_ns - batches[-1]["release_ns"], 1),
            "the GPU's memory in use, sampled once a second: %s"
            % ("the largest of %d samples is %d MiB" % (len(gpu.samples), max(m for _, m in gpu.samples)) if gpu.samples else "no sample") + ("; %d got no answer" % gpu.missed if gpu.missed else "") + given_up(gpu, started_ns),
            "host memory, the largest resident set of cp_prove: %s GB" % cut(resident_kib * 1024, 10**9, 1),
        )
    sizes = [answer.size for answer in answers.values() if not answer.failed]
    if sizes:
        report.say("a proof's size, MB: " + least_mean_most(sizes, 10**6, 2))
    report.say("requests: %d served with a proof, %d failed, %d without a response" % (figures["served"], len(failed), len(numbers) - len(answers)))
    if checks:
        report.say(
            "cp_verify: %d accepted, %d not accepted, %d not checked" % (len(accepted), len(checks) - len(accepted), len(proved) - len(checks)),
            "  a call's seconds, with %d side by side: %s; all calls together: %s s" % (o.checks_at_once, least_mean_most([c["ns"] for c in checks.values()], 10**9, 1), seconds(checks_ns, 1)),
        )
    if why_not:
        report.say("", "NO RATE IS GIVEN, because:", *("- " + why for why in why_not))

    record.update(
        options={name: value for name, value in vars(o).items()},
        tokens_in_a_request=tokens,
        command=command,
        process_id=child.pid,
        started_ns=started_ns,
        ended_ns=ended_ns,
        exit_code=code,
        largest_resident_set_kib=resident_kib,
        requests=[
            dict(number=number, id=name, left_in_ns=left_ns.get(name), faults=faulty.get(name, []), check=checks.get(number), **(answers[name]._asdict() if name in answers else {}))
            for name, number in sorted(numbers.items(), key=lambda item: item[1])
        ],
        batches=batches,
        gpu_samples=gpu.samples[:],
        gpu_sampling_given_up=gpu.gave_up,
        figures=figures,
        no_rate_because=why_not,
    )
    unwritten = None
    try:
        with open(os.path.join(work, "figures.json"), "x", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
    except OSError as why:
        unwritten = why.strerror
    report.say("", "These lines are in %s, and every time that they rest on in %s." % (os.path.join(work, "report.txt"), os.path.join(work, "figures.json")))
    for name, why in (("report.txt", report.lost), ("figures.json", unwritten)):
        if why:
            report.say("NOTE: %s is not whole: a write to it failed (%s)." % (name, why))
    report.close()
    return 1 if why_not else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # comes only before cp_prove is started: from then on main()'s own handlers take the signals
        sys.exit("%s: interrupted: cp_prove was not started. The work directory is not removed: if it exists by now, name another for the next run" % ME)
