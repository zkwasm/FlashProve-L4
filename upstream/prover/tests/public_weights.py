#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Tests the option `-public-weights <file of 32 bytes>` of cp_commit and cp_prove.

    prover/tests/public_weights.py <serving build> <build with the test options> <new work directory>
                            <configuration for model files> <make_model>
                            <configuration that passes> <configuration that fails>

The option is for weights that are public anyway. The commitment's seed is derived from the file,
and no fresh randomness goes into any of its files, so anyone who has the weights, the configuration
and the file can create the commitment again and compare it byte for byte. In exchange, the
commitment hides nothing. So a single mistake must neither create nor serve such a commitment: on
model files cp_commit also asks for a marker file next to each, the last line of the commitment's
isk.txt is `seed public`, and cp_prove needs both that line and the option with the same file.

The two builds are directories (`folder` in names) with cp_commit, cp_prove and cp_verify: the
production build (prover/build.sh; `serving` in names) and the test build, which has the test
options (prover/build.sh tests). Only the production build reads model files and has the option. The
configuration for model files is prover/examples/target_small_file_f2; make_model (prover/build.sh
make_model) writes the model files for its model.txt. The other two are
prover/examples/target_small_f2_roomy, whose scales pass cp_commit's worst-case check (can a prompt
drive a value out of its range?) on synthetic weights, and prover/examples/target_small_f2, whose
scales fail it. All weights are synthetic (`made-up` in names). Needs one H100 that is otherwise
idle; GPU_RUN in the environment is a command prefix for each call of cp_commit and cp_prove. The
work directory must be new (its missing parents are created), takes three sets of model files
(530 MB each) and is not removed; this script writes and removes files only there. Prints
ALL AS EXPECTED and exits with status 0, or reports what differs and exits with status 1. If
cp_commit, cp_prove, cp_verify or make_model has not ended after an hour, it is sent no signal: the
test prints its name, the ID of the process that the test started and the log's path, leaves it
running, and exits with status 2.

  checks kept    with the option: scales that fail the worst-case check: status 7. A file of 31
                 bytes, a file that does not exist, -keep-for-the-audit, or a test variable in the
                 environment: status 2. Nothing is created
  marker         the call on a directory of model files (here a stand-in: one empty file a.gguf).
                 The test writes what sha256sum prints for a.gguf into the marker,
                 a.gguf.weights-are-public. Without that file, or with only `weights-are-public`
                 in the directory: status 2, cp-public-weights-marker, no warning line, and nothing
                 is created. With a directory of the marker's name, a link that leads nowhere, a
                 link to a good marker, or an empty file of that name: the same, and the message
                 says why: not a regular file, a symbolic link (both links), empty or malformed.
                 With the marker the call gets past this check and prints its warning (it then fails
                 on the stand-in), also where the directory has mode 775: a shared directory is
                 accepted. With a marker of mode 664: rejected (`refused` in messages) as writable
                 by its group or by everybody. Without the option the file is not asked for. A
                 second model file without its marker: rejected again, with that marker's name
  model files    on the two model files that make_model writes, each with its marker: status 0, and
                 cp_commit prints that it computes the SHA-256 of the files, then that each has the
                 declared one, and only then how much memory it page-locked. The same call into
                 other directories: the same commitment ID, and every file of both key directories,
                 the public one (`ivk`) and the secret one (`isk`), is identical. The same call on a
                 copy of the model directory: the same ID and the same public files. Then, in that
                 copy, a model file is replaced while its marker stays: the second model file and
                 wsums.txt become those of another model, which differs in one row of one matrix.
                 Status 2, cp-public-weights-marker; the message names the file and its marker; the
                 output contains no model file's SHA-256 and not the line on page-locked memory;
                 nothing is created. After sha256sum on the new file: status 0 and another
                 commitment ID. With the second file's SHA-256 in the first file's marker: rejected
                 like the replaced file, and the message names the first file. Then one bit of one
                 weight is changed and the file declared again: the markers pass, and the check
                 against wsums.txt rejects the weights: status 6. With a model file's marker moved
                 aside and a link to it in its place, or with a second name (a hard link) for it:
                 rejected, and the message says which
  created        on synthetic weights, with scales that pass the check: status 0. cp_commit prints
                 its warning once. isk.txt ends with the lines `ranges` and `seed public`. seed.bin
                 is the seed derived from the file. isk/leaf/0/fresh.bin, normally 32 random bytes
                 for the masks of leaf table 0 (`stored copy` in messages), is zero. The closing
                 line ends with the statement that the commitment hides nothing and does not call
                 the secret key directory secret
  created again  the same call into other directories: the same commitment ID, and every file of
                 both key directories is identical
  two builds     the test build, with `-test-seed` on a file that holds the derived seed, and
                 `-worst-case`: its public key directory is identical to the production build's, and
                 so is the secret one except isk.txt's last line (`seed test`)
  another file   other 32 bytes: another key.bin and another ID
  without        without the option, twice: isk.txt has no line on the seed, fresh.bin is not zero,
                 the closing line says that the commitment can be handed out and that the secret key
                 directory is secret, there is no warning line, and the two commitments differ
  not served     cp_prove without the option; the option on the commitment with a drawn seed; with a
                 file of 31 bytes; with the other file: each ends with status 1 and the last line
                 `p258 end 1 cp-isk-seed-public`, the first three after a sentence that says why. No
                 proof is written
  line removed   a copy of the secret key directory without the line `seed public`, served without
                 the option: the zero fresh.bin stops the first batch (cp-leaf-record); no proof
  line added     a copy of the secret key directory of the commitment with a drawn seed, with the
                 line added, served with the option: the seed is not the derived one
                 (cp-isk-seed-public), and there is no proof
  control        the commitment with a drawn seed is served without the option, and an unaltered
                 copy of the other secret key directory with it: so the three rejections above on
                 these directories come from what was changed
  served         cp_prove with the option prints its warning once and serves the request,
                 and cp_verify accepts the response under the commitment ID: YES, status 0. A second
                 run on the same request writes the same output token and another proof: a batch's
                 randomness is still drawn. On the commitment to model files: one run, also accepted
  test options   the test build's cp_commit and cp_prove do not know the option, and its cp_prove
                 rejects a commitment that has the line

Not shown: the full size; a marker that cannot be read; a marker that belongs to another account
(prover/tests/public_weights_test.cpp tests that rule); a model file that is replaced or modified
while cp_commit runs; that two machines give the same bytes (this test runs on one); what the
production build writes under a drawn seed, beyond the points above; that cp_commit empties its
environment; cp_prove's service on a spool (-spool) under the option: every run here is one batch
(-in, -out).
"""

import glob
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

OPTION = "-public-weights"
LINE = "seed public"
NAME = "refused (cp-isk-seed-public): "
LAST = "p258 end 1 "
MARKER = ".weights-are-public"  # appended to a model file's name
NO_MARKER = "cp_commit: refused (cp-public-weights-marker): "
# The part of each cp-public-weights-marker message that only it has. %s: the marker's path; in
# OTHER_BYTES, the model file's path and then the marker's.
NO_FILE = "there is no regular file %s. "
NOT_REGULAR = "%s is not a regular file. "
A_LINK = "%s is a symbolic link, which is not accepted. "
SECOND_NAME = "%s has a second name (a hard link), which is not accepted. "
MALFORMED = "%s is empty or does not start with 64 lower-case hexadecimal digits"
OPEN_TO_OTHERS = "%s can be written by its group or by everybody, "
OTHER_BYTES = "the SHA-256 of %s differs from the one in %s, "
COMPUTES = "cp_commit: next: computing each model file's SHA-256 (files: %d; "
COMPARED = "cp_commit: each model file has the SHA-256 that was declared public: "
MEMORY_LOCKED = " GB of host memory page-locked for "
NOT_HIDDEN = "with this option the weights are NOT hidden: whoever has the 32-byte file can work out every mask. "
NO_MARK = "The public files carry no mark of it: key, entries, ID and proofs look like any other commitment's"
COMMIT_WARNS = "cp_commit: FOR PUBLIC WEIGHTS: " + NOT_HIDDEN + NO_MARK + "\n"
PROVE_WARNS = "p258 FOR PUBLIC WEIGHTS: the commitment's seed is public, so nothing is hidden\n"
CLOSING = (
    ', which is complete, and hides nothing: it was made with -public-weights (isk.txt says "seed public"), and '
    + NOT_HIDDEN + NO_MARK + ". Never use this option on weights that are to stay hidden"
)
HANDED_OUT = ", which is complete and can be handed out. "
VARIABLE = {"FORMF_PLANTED_INTEGER": "1:5000"}
# make_model's last argument for the second model: in this matrix, which is in the second model
# file, row 0 is copied over row 1. Every row of the second model is a row of the first, so the
# scales that pass the worst-case check on the first model should pass it on the second too.
OTHER_MODEL = "blk.3.attn_v:0=1"
SEED_WORD = 0x57505043  # "CPPW" (common/public_weights.h)


def derived_seed(file_bytes):
    """The first 32 bytes of the ChaCha20 block (RFC 8439, section 2.3, specifies it) whose key is
    the file's 32 bytes and whose last four input words are "CPPW", 0, 0, 0."""
    start = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", file_bytes), SEED_WORD, 0, 0, 0]
    s = list(start)
    M = 0xFFFFFFFF

    def rotl(x, n):
        return (x << n | x >> 32 - n) & M

    # The quarter round of prover/common/coin_chacha.h, quarter_round(), in Python.
    # Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version 20080118, public domain).
    # RFC 8439, section 2.1, specifies the same function.
    def quarter_round(a, b, c, d):
        s[a] = (s[a] + s[b]) & M
        s[d] = rotl(s[d] ^ s[a], 16)
        s[c] = (s[c] + s[d]) & M
        s[b] = rotl(s[b] ^ s[c], 12)
        s[a] = (s[a] + s[b]) & M
        s[d] = rotl(s[d] ^ s[a], 8)
        s[c] = (s[c] + s[d]) & M
        s[b] = rotl(s[b] ^ s[c], 7)

    # The order of the eight quarter rounds is that of the round loop in D. J. Bernstein's chacha-merged.c.
    for _ in range(10):
        for i in range(4):
            quarter_round(i, 4 + i, 8 + i, 12 + i)
        for i in range(4):
            quarter_round(i, 4 + (i + 1) % 4, 8 + (i + 2) % 4, 12 + (i + 3) % 4)
    return struct.pack("<8I", *((x + y) & 0xFFFFFFFF for x, y in zip(s[:8], start[:8])))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def files_under(directory):
    """{path below the directory: SHA-256} of every file there."""
    return {
        os.path.relpath(os.path.join(at, name), directory): sha256_file(os.path.join(at, name))
        for at, _, names in os.walk(directory)
        for name in names
    }


def contents(path):
    with open(path, "rb") as f:
        return f.read()


def declare_public(path, marker=None):
    """Writes what sha256sum prints for a model file, run in the file's directory, into its marker,
    NAME.weights-are-public. marker: another path for the output. The umask is 022 (main()), so
    only the owner can write the marker, as cp_commit requires."""
    directory, name = os.path.split(path)
    with open(marker or path + MARKER, "wb") as f:
        subprocess.run(["sha256sum", name], cwd=directory, stdout=f, check=True)


def change_one_weight(directory):
    """Flips the lowest bit of one 4-bit weight in one model file, in place, and returns that file's
    name. GGUF, version 3: magic, version, number of tensors, number of key-value pairs; the pairs;
    for each tensor its name, dimensions, type and offset; then the data, aligned. Q4_K (type 12):
    blocks of 144 bytes for 256 weights, of which the last 128 hold two weights each."""
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for path in sorted(glob.glob(os.path.join(directory, "*.gguf")), reverse=True):
        with open(path, "r+b") as f:
            u32 = lambda: struct.unpack("<I", f.read(4))[0]  # noqa: E731
            u64 = lambda: struct.unpack("<Q", f.read(8))[0]  # noqa: E731
            text = lambda: f.read(u64()).decode()  # noqa: E731

            def value(kind):
                if kind == 8:
                    return text()
                if kind == 9:
                    k, n = u32(), u64()
                    return [value(k) for _ in range(n)]
                return f.read(size[kind])

            if f.read(4) != b"GGUF":
                sys.exit(path + " is no GGUF file")
            _, tensors, pairs = u32(), u64(), u64()
            alignment = 32
            for _ in range(pairs):
                key, kind = text(), u32()
                v = value(kind)
                if key == "general.alignment":
                    alignment = struct.unpack("<I", v)[0]
            table = []
            for _ in range(tensors):
                name, dimensions = text(), u32()
                cells = 1
                for _ in range(dimensions):
                    cells *= u64()
                table.append((cells, u32(), u64()))
            data = (f.tell() + alignment - 1) // alignment * alignment
            for cells, kind, offset in table:
                if kind == 12 and cells >= 512:
                    at = data + offset + cells // 256 // 2 * 144 + 16
                    f.seek(at)
                    old = f.read(1)[0]
                    f.seek(at)
                    f.write(bytes([old ^ 1]))
                    return os.path.basename(path)
    sys.exit("no weight was changed: the model files have no Q4_K tensor")


class Call:
    """One call of cp_commit into <work>/<name>/ivk and <work>/<name>/isk."""


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        return 2
    serving, tests, work, for_files, make_model, passing, failing = (os.path.abspath(x) for x in sys.argv[1:])
    for program in [os.path.join(b, p) for b in (serving, tests) for p in ("cp_commit", "cp_prove", "cp_verify")] + [make_model]:
        if not os.access(program, os.X_OK):
            sys.exit(program + " is missing; build it first (the text at the top of this file names the builds)")
    if re.search(r"[^A-Za-z0-9_./-]", work):
        sys.exit("the work directory's path may contain only letters, digits and / . _ -")
    if os.path.lexists(work):
        sys.exit(work + " exists already; name a new directory")
    os.umask(0o022)  # cp_commit rejects a marker that the group can write; the markers that this test creates do not depend on the caller's umask
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name has been taken since the check; closed to other users
    for name in ("home", "tmp", "in"):
        os.mkdir(os.path.join(work, name))
    gpu_run = os.environ.get("GPU_RUN", "").split()
    clean = dict(os.environ, HOME=os.path.join(work, "home"), TMPDIR=os.path.join(work, "tmp"))  # cp_prove needs both to name directories
    differ = [0]

    def check(good, what, detail=""):
        differ[0] += not good
        print("%-9s %s" % ("as wanted" if good else "DIFFERS", what), flush=True)
        if not good and detail:
            print("          " + detail, flush=True)
        return bool(good)

    def write_file(name, data, mode=0o644):
        path = os.path.join(work, name)
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as f:
            f.write(data)
        return path

    def run(name, words, env=None, gpu=True, limit=3600):
        log = os.path.join(work, name + ".log")
        program, words = words[0], (gpu_run if gpu else []) + words
        with open(log, "wb") as f:
            f.write((" ".join(words) + "\n").encode())
            f.flush()
            child = subprocess.Popen(words, env=dict(clean, **(env or {})), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                status = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                # No signal is sent: it could end a GPU program in the middle of its work, and with
                # GPU_RUN it would reach only the prefix command, not the program.
                print("NOT ENDED after %d s: %s. The process that this test started has the ID %d. It was sent no signal. The GPU is not free until it ends. "
                      "Its log: %s" % (limit, program, child.pid, log), flush=True)
                sys.exit(2)
        with open(log, errors="replace") as f:
            return status, f.read().split("\n", 1)[1]

    def commit(name, options, configuration=passing, weights="made-up-blocks", build=serving, env=None):
        c = Call()
        c.name, c.folder = name, os.path.join(work, name)
        os.makedirs(c.folder)
        c.ivk, c.isk, c.config = os.path.join(c.folder, "ivk"), os.path.join(c.folder, "isk"), configuration
        words = [os.path.join(build, "cp_commit"), "-weights", weights, "-config", configuration, "-M", "1", "-ivk", c.ivk, "-isk", c.isk]
        c.status, c.said = run(name + ".cp_commit", words + list(options), env=env)
        found = re.findall(r"(?m)^commitment ID: ([0-9a-f]{64})$", c.said)
        c.id = found[0] if len(found) == 1 else None
        return c

    def left_by(c):
        return sorted(os.listdir(c.folder))

    def not_created(what, c, status, wanted, shown=None, also=True):
        if c.status == 0 or left_by(c):
            print("WAS NOT REFUSED: cp_commit, %s; status %d; created: %s" % (what, c.status, left_by(c)), flush=True)
        check(
            c.status == status and wanted in c.said and also and not left_by(c),
            "%s: status %d, %s, and nothing is created" % (what, status, shown or wanted.strip()),
            "status %d; %s" % (c.status, c.said[-300:]),
        )

    def lines_of(c):
        with open(os.path.join(c.isk, "isk.txt")) as f:
            return f.read().splitlines()

    def closing_of(c):
        return [x for x in c.said.splitlines() if x.startswith("cp_commit: done. ")]

    def prove(c, name, options, build=serving, isk=None):
        """One batch of one request: (status, output, names in -out except .tmp, -out's path)."""
        out = os.path.join(work, name + ".out")
        words = [os.path.join(build, "cp_prove"), "-isk", isk or c.isk, "-in", requests[c.config], "-out", out]
        status, said = run(name + ".cp_prove", words + list(options))
        return status, said, sorted(x for x in (os.listdir(out) if os.path.isdir(out) else []) if x != ".tmp"), out

    def not_served(what, ran, status, last, sentence=None):
        ended, said, written, _ = ran
        if ended == 0 or [x for x in written if not x.endswith(".failed")]:
            print("WAS NOT REFUSED: cp_prove, %s; status %s; out: %s" % (what, ended, written), flush=True)
        check(
            ended == status
            and said.splitlines()[-1:] == [last]
            and (sentence is None or sentence in said)
            and not [x for x in written if not x.endswith(".failed")],
            "not served: %s: status %d, the last line %s, and no response" % (what, status, last),
            "status %s; %r; out: %s" % (ended, said[-300:], written),
        )

    def request_for(configuration, name):
        """A directory with one request: synthetic tokens; its range is the last position only."""
        with open(os.path.join(configuration, "policy.txt")) as f:
            word = dict(x.split() for x in f if len(x.split()) == 2)
        count, vocabulary = 1 << int(word["lt"]), int(word["V"])
        tokens = [(2654435761 * (t + 1) >> 7) % vocabulary for t in range(count)]
        os.makedirs(os.path.join(work, "in", name))
        write_file(os.path.join("in", name, "r.req"), struct.pack("<%dI" % (count + 2), *tokens, count - 1, count))
        return os.path.join(work, "in", name)

    file = write_file("public.seed", hashlib.sha256(b"public weights").digest())
    other_file = write_file("other.seed", bytes(range(32)))

    # `for_files` has no layout.txt. Its copy gets one: rows of 2^17 entries, of which 4 stay free.
    files_config = os.path.join(work, "config_for_files")
    os.makedirs(files_config)
    for name in ("policy.txt", "KL.bin"):
        shutil.copy(os.path.join(for_files, name), files_config)
    _, arrays = run("arrays", [os.path.join(serving, "cp_verify"), "-plan", "arrays", files_config, "128"], gpu=False)  # a planning call: its status is 4
    write_file("arrays.txt", arrays.encode())
    pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "pack_layout.py")
    made = subprocess.run([sys.executable, "-B", pack, "--arrays", os.path.join(work, "arrays.txt"), "17", "4"], capture_output=True)
    if made.returncode:
        sys.exit("the layout file cannot be made: " + made.stderr.decode(errors="replace")[:300])
    with open(os.path.join(files_config, "layout.txt"), "wb") as f:
        f.write(made.stdout)
    requests = {passing: request_for(passing, "synthetic"), files_config: request_for(files_config, "files")}

    # ---- checks kept
    not_created("checks kept: scales that fail the worst-case check", commit("fails_the_check", [OPTION, file], failing), 7, "NOT every bounded site fits")
    not_created("checks kept: a file of 31 bytes", commit("short_file", [OPTION, write_file("short.seed", bytes(31))]), 2, "cp_commit: refused (usage): " + OPTION)
    not_created("checks kept: a file that does not exist", commit("no_file", [OPTION, os.path.join(work, "no-such-file")]), 2, "cp_commit: refused (usage): " + OPTION)
    not_created("checks kept: with -keep-for-the-audit", commit("kept", [OPTION, file, "-keep-for-the-audit"]), 2, "usage: cp_commit")
    not_created("checks kept: a test variable in the environment", commit("test_variable", [OPTION, file], env=VARIABLE), 2, "the environment holds FORMF_PLANTED_INTEGER")

    # ---- marker. cp_commit treats a directory with a file *.gguf as model files; it asks for each
    # model file's marker before it opens a model file.
    stand_ins = os.path.join(work, "stand_ins")
    os.makedirs(stand_ins)
    printed = os.path.realpath(stand_ins)  # cp_commit prints the directory as realpath() gives it
    stand_in = write_file(os.path.join("stand_ins", "a.gguf"), b"")
    marker = stand_in + MARKER

    def on_stand_ins(what, name, options, asked, of="a.gguf", message=NO_FILE):
        c = commit(name, options, files_config, stand_ins)
        if c.status == 0 or left_by(c):
            print("WAS NOT REFUSED: cp_commit, %s; status %d; created: %s" % (what, c.status, left_by(c)), flush=True)
        if asked:
            good = c.status == 2 and NO_MARKER in c.said and COMMIT_WARNS not in c.said
            good = good and message % os.path.join(printed, of + MARKER) in c.said
            wanted = "status 2, cp-public-weights-marker, the marker's name, no warning line"
        else:
            good = c.status != 0 and NO_MARKER not in c.said and "gguf: " in c.said and c.said.count(COMMIT_WARNS) == (OPTION in options)
            wanted = "the call gets past the marker's check, and fails on the stand-in"
        check(good and not left_by(c), "marker: %s: %s, and nothing is created" % (what, wanted), "status %d; %s" % (c.status, c.said[-300:]))

    on_stand_ins("no such file", "marker_missing", [OPTION, file], True)
    declare_public(stand_in, os.path.join(stand_ins, "weights-are-public"))
    on_stand_ins("only the directory has one", "marker_of_directory", [OPTION, file], True)
    os.mkdir(marker)
    on_stand_ins("a directory of that name", "marker_directory", [OPTION, file], True, message=NOT_REGULAR)
    os.rmdir(marker)
    os.symlink("elsewhere", marker)
    on_stand_ins("a link that leads nowhere", "marker_dangling", [OPTION, file], True, message=A_LINK)
    declare_public(stand_in, os.path.join(stand_ins, "elsewhere"))
    on_stand_ins("a link to a good marker", "marker_link", [OPTION, file], True, message=A_LINK)
    os.unlink(marker)
    on_stand_ins("without the option and without the file", "marker_not_asked", [], False)
    write_file(os.path.join("stand_ins", "a.gguf" + MARKER), b"")
    on_stand_ins("an empty file, reported as empty or malformed", "marker_empty", [OPTION, file], True, message=MALFORMED)
    os.unlink(marker)
    declare_public(stand_in)
    on_stand_ins("the output of sha256sum", "marker_file", [OPTION, file], False)
    os.chmod(stand_ins, 0o775)
    on_stand_ins("the same in a directory that its group can write", "marker_shared_directory", [OPTION, file], False)
    os.chmod(stand_ins, 0o755)
    os.chmod(marker, 0o664)
    on_stand_ins("its group can write it", "marker_writable", [OPTION, file], True, message=OPEN_TO_OTHERS)
    os.chmod(marker, 0o644)
    write_file(os.path.join("stand_ins", "b.gguf"), b"")
    on_stand_ins("a second model file", "marker_second", [OPTION, file], True, "b.gguf")

    # ---- model files. Copied whole: the production build follows no link in a model directory.
    first_model, second_model, changed_model = (os.path.join(work, x) for x in ("model", "other_model", "changed_model"))
    with open(os.path.join(for_files, "model.txt")) as f:
        shape = f.read().split()
    for directory, last in ((first_model, []), (second_model, [OTHER_MODEL])):
        status, said = run(os.path.basename(directory) + ".make_model", [make_model, directory] + shape + last, gpu=False)
        if status:
            sys.exit("make_model ended with status %d: %s" % (status, said[-300:]))
    in_first, in_second = files_under(first_model), files_under(second_model)
    shards = sorted(x for x in in_first if x.endswith(".gguf"))
    replaced = shards[-1]
    check(
        len(shards) == 2 and in_first.keys() == in_second.keys() and {x for x in in_first if in_first[x] != in_second[x]} == {replaced, "wsums.txt"},
        "model files: make_model has written two model files, and the second model differs from the first in the second model file and in wsums.txt only",
        "files that differ: %s" % sorted(x for x in in_first if in_first[x] != in_second.get(x)),
    )
    for shard in shards:
        declare_public(os.path.join(first_model, shard))
    shutil.copytree(first_model, changed_model)
    digests = [x[s] for x in (in_first, in_second) for s in shards]
    public = commit("model_public", [OPTION, file], files_config, first_model)
    check(
        public.status == 0 and public.said.count(COMMIT_WARNS) == 1 and lines_of(public)[-1] == LINE and bool(public.id),
        "model files: each of the %d model files with its marker: status 0, the warning once, and isk.txt ends with the line %s" % (len(shards), LINE),
        "status %d; %s" % (public.status, public.said[-300:]),
    )
    steps = [COMPUTES % len(shards), COMPARED, MEMORY_LOCKED]
    check(
        [public.said.count(x) for x in steps] == [1, 1, 1] and sorted(steps, key=public.said.index) == steps,
        "model files: cp_commit prints once that it computes the SHA-256 of %d files, then once that each has the declared one, and only then"
        " how much memory it page-locked" % len(shards),
        " / ".join(x for x in public.said.splitlines() if "SHA-256" in x)[-300:],
    )
    public_again = commit("model_public_again", [OPTION, file], files_config, first_model)
    check(
        public.status == 0 and public_again.status == 0 and public_again.id == public.id
        and bool(files_under(public.ivk)) and files_under(public.ivk) == files_under(public_again.ivk) and files_under(public.isk) == files_under(public_again.isk),
        "model files: the same call into other directories: the same ID, and every file of both key directories is identical",
        "status %d, %d" % (public.status, public_again.status),
    )

    def other_bytes(what, name, shard):
        """A call that must be rejected: this model file does not have the declared SHA-256."""
        c, path = commit(name, [OPTION, file], files_config, changed_model), os.path.join(os.path.realpath(changed_model), shard)
        known = digests + [sha256_file(os.path.join(changed_model, s)) for s in shards]
        not_created(
            "model files: " + what, c, 2, NO_MARKER + OTHER_BYTES % (path, path + MARKER),
            "cp-public-weights-marker, the paths of %s and of its marker, no model file's SHA-256, no report that memory was page-locked" % shard,
            not any(x in c.said for x in (COMPARED, MEMORY_LOCKED, *known)),
        )

    elsewhere = commit("model_public_elsewhere", [OPTION, file], files_config, changed_model)
    check(
        elsewhere.status == 0 and bool(public.id) and elsewhere.id == public.id and files_under(elsewhere.ivk) == files_under(public.ivk),
        "model files: the same call on a copy of the model directory: the same ID, and every public file is identical",
        "status %d" % elsewhere.status,
    )
    for name in (replaced, "wsums.txt"):
        os.replace(os.path.join(second_model, name), os.path.join(changed_model, name))
    other_bytes("a model file replaced while its marker stays", "model_replaced", replaced)
    declare_public(os.path.join(changed_model, replaced))
    declared_again = commit("model_declared_again", [OPTION, file], files_config, changed_model)
    check(
        declared_again.status == 0 and bool(declared_again.id) and declared_again.id != public.id,
        "model files: after sha256sum on the new file: status 0 and another commitment ID",
        "status %d; %s" % (declared_again.status, declared_again.said[-300:]),
    )
    declare_public(os.path.join(changed_model, replaced), os.path.join(changed_model, shards[0] + MARKER))
    other_bytes("the first model file's marker has the SHA-256 of the second", "model_exchanged", shards[0])
    declare_public(os.path.join(changed_model, shards[0]))
    changed = change_one_weight(changed_model)
    declare_public(os.path.join(changed_model, changed))
    not_created(
        "model files: one bit of one weight changed and the file declared again: the markers pass, and the check against wsums.txt rejects the weights",
        commit("model_one_bit", [OPTION, file], files_config, changed_model), 6, COMPARED, "the line that each file has the declared SHA-256",
    )
    # Both of cp_commit's marker checks reject a link and a second name; here the first one does.
    whole = os.path.join(os.path.realpath(first_model), shards[0])
    os.rename(whole + MARKER, whole + ".moved")
    os.symlink(os.path.basename(whole) + ".moved", whole + MARKER)
    not_created(
        "model files: a link to the marker in its place", commit("model_marker_link", [OPTION, file], files_config, first_model), 2,
        A_LINK % (whole + MARKER), "cp-public-weights-marker, and the message says that the marker is a symbolic link",
    )
    os.unlink(whole + MARKER)
    os.link(whole + ".moved", whole + MARKER)
    not_created(
        "model files: the marker has a second name", commit("model_marker_second_name", [OPTION, file], files_config, first_model), 2,
        SECOND_NAME % (whole + MARKER), "cp-public-weights-marker, and the message says that the marker has a second name",
    )
    os.unlink(whole + ".moved")

    # ---- created
    made = commit("made", [OPTION, file])
    if made.status:
        check(False, "created: cp_commit with the option exits with status 0", "status %d; %s" % (made.status, made.said[-400:]))
        print("%d CHECKS DIFFER" % differ[0])
        return 1
    last = lines_of(made)[-2:]
    check(made.said.count(COMMIT_WARNS) == 1 and "every bounded site fits" in made.said, "created: cp_commit prints its warning once, and the worst-case check passes", made.said[:400])
    check(last[0].startswith("ranges ") and last[1] == LINE, "created: isk.txt ends with the lines ranges and %s" % LINE, str(last))
    check(contents(os.path.join(made.isk, "seed.bin")) == derived_seed(contents(file)), "created: seed.bin is the seed derived from the file's bytes")
    check(contents(os.path.join(made.isk, "leaf", "0", "fresh.bin")) == bytes(32), "created: the stored copy's fresh.bin is 32 zero bytes")
    check(
        len(closing_of(made)) == 1 and closing_of(made)[0].endswith(CLOSING) and " is secret" not in closing_of(made)[0],
        "created: the closing line says complete, ends with the statement that the commitment hides nothing, and calls nothing secret",
        str(closing_of(made))[-500:],
    )

    # ---- created again
    again = commit("made_again", [OPTION, file])
    same = (files_under(made.ivk) == files_under(again.ivk), files_under(made.isk) == files_under(again.isk)) if again.status == 0 else ()
    check(
        again.status == 0 and bool(made.id) and again.id == made.id and same == (True, True),
        "created again: the same ID, and every file of both key directories is identical",
        "status %d; public, secret: %s" % (again.status, same),
    )

    # ---- two builds. -test-seed takes the file's bytes as the seed itself, so it gets the derived seed.
    tested = commit("by_the_other_build", ["-test-seed", write_file("derived.seed", derived_seed(contents(file)), 0o600), "-worst-case"], build=tests)
    ours, theirs = files_under(made.isk), files_under(tested.isk) if tested.status == 0 else {}
    texts = ours.pop("isk.txt", None), theirs.pop("isk.txt", None)
    check(
        tested.status == 0
        and bool(files_under(made.ivk)) and files_under(made.ivk) == files_under(tested.ivk)
        and None not in texts and bool(ours) and ours == theirs
        and lines_of(made)[:-1] == lines_of(tested)[:-1] and (lines_of(made)[-1], lines_of(tested)[-1]) == (LINE, "seed test"),
        "two builds: the build with the test options, with -test-seed on the derived seed and -worst-case: the public files, the secret files"
        " except isk.txt, and isk.txt up to its last line are identical",
        "status %d; secret files that differ: %s; %s" % (tested.status, sorted(x for x in ours.keys() | theirs.keys() if ours.get(x) != theirs.get(x)), tested.said[-300:] if tested.status else ""),
    )

    # ---- another file
    other = commit("other", [OPTION, other_file])
    check(
        other.status == 0 and bool(other.id) and other.id != made.id
        and sha256_file(os.path.join(other.ivk, "key.bin")) != sha256_file(os.path.join(made.ivk, "key.bin")),
        "another file: another key.bin and another ID", "status %d" % other.status,
    )

    # ---- without
    drawn, drawn_again = commit("drawn", []), commit("drawn_again", [])
    if drawn.status or drawn_again.status or other.status:
        check(False, "the commitments for the remaining checks are created", "status %d, %d, %d" % (drawn.status, drawn_again.status, other.status))
        print("%d CHECKS DIFFER" % differ[0])
        return 1
    for c in (drawn, drawn_again):
        check(
            not [x for x in lines_of(c) if x.startswith("seed")] and contents(os.path.join(c.isk, "leaf", "0", "fresh.bin")) != bytes(32)
            and COMMIT_WARNS not in c.said and len(closing_of(c)) == 1 and HANDED_OUT in closing_of(c)[0] and closing_of(c)[0].endswith(c.isk + " is secret"),
            "without: %s: isk.txt has no line on the seed, fresh.bin is not zero, there is no warning line, and the closing line is the usual one" % c.name,
            str(closing_of(c))[-300:],
        )
    check(
        drawn.id != drawn_again.id and contents(os.path.join(drawn.isk, "seed.bin")) != contents(os.path.join(drawn_again.isk, "seed.bin"))
        and not set(files_under(drawn.ivk).values()) & set(files_under(drawn_again.ivk).values()),
        "without: two calls give different seeds, IDs and public files",
    )

    # ---- not served
    not_served("without the option", prove(other, "no_option", []), 1, LAST + "cp-isk-seed-public", NAME + 'isk.txt says "seed public"')
    not_served("the option on a commitment with a drawn seed", prove(drawn, "option_on_drawn", [OPTION, file]), 1, LAST + "cp-isk-seed-public", NAME + OPTION + " is given, but")
    not_served("with a file of 31 bytes", prove(other, "short_file_served", [OPTION, os.path.join(work, "short.seed")]), 1, LAST + "cp-isk-seed-public", " is not a readable file of 32 bytes")
    not_served("with the other file", prove(other, "wrong_file", [OPTION, file]), 1, LAST + "cp-isk-seed-public")

    # ---- line removed, line added
    def copy_of(c, name, lines):
        isk = os.path.join(work, name + ".isk")
        shutil.copytree(c.isk, isk)
        os.chmod(os.path.join(isk, "isk.txt"), 0o600)
        with open(os.path.join(isk, "isk.txt"), "w") as f:
            f.write("".join(x + "\n" for x in lines))
        return isk

    not_served(
        "line removed: without the line and the option the first batch stops at the zero fresh.bin",
        prove(made, "line_removed", [], isk=copy_of(made, "line_removed", lines_of(made)[:-1])), 1, LAST + "cp-leaf-record",
    )
    not_served(
        "line added: on a commitment with a drawn seed, with the line added and the option, the seed is not the derived one",
        prove(drawn, "line_added", [OPTION, file], isk=copy_of(drawn, "line_added", lines_of(drawn) + [LINE])), 1, LAST + "cp-isk-seed-public",
    )

    # ---- controls
    for what, ran, warns in (
        ("the commitment with a drawn seed is served without the option", prove(drawn, "drawn_served", []), 0),
        ("an unaltered copy of the secret key directory is served with the option",
         prove(made, "copy_served", [OPTION, file], isk=copy_of(made, "copy_served", lines_of(made))), 1),
    ):
        check(
            ran[0] == 0 and ran[1].count(PROVE_WARNS) == warns and ran[2] == ["r.proof", "r.y"],
            "control: " + what, "status %s; out: %s; %s" % (ran[0], ran[2], ran[1][-300:]),
        )

    # ---- served
    def served(c, names, seed_file):
        runs = [prove(c, name, [OPTION, seed_file]) for name in names]
        for name, (ended, said, written, out) in zip(names, runs):
            check(
                ended == 0 and said.count(PROVE_WARNS) == 1 and written == ["r.proof", "r.y"],
                "%s: cp_prove prints its warning once and serves the request" % name, "status %s; out: %s; %s" % (ended, written, said[-300:]),
            )
            if written != ["r.proof", "r.y"]:
                continue
            words = [os.path.join(serving, "cp_verify"), c.config, c.ivk, os.path.join(requests[c.config], "r.req"), os.path.join(out, "r.y"), os.path.join(out, "r.proof"), "-id", c.id]
            status, verdict = run(name + ".cp_verify", words, gpu=False)
            check(status == 0 and verdict.splitlines()[-1:] == ["YES"], "%s: cp_verify accepts the response under the commitment ID: YES, status 0" % name, "status %d; %s" % (status, verdict[-200:]))
        return runs

    first, second = served(other, ("served", "served_again"), other_file)
    check(
        first[2] == second[2] == ["r.proof", "r.y"]
        and contents(os.path.join(first[3], "r.y")) == contents(os.path.join(second[3], "r.y"))
        and contents(os.path.join(first[3], "r.proof")) != contents(os.path.join(second[3], "r.proof")),
        "served: two runs on the same request write the same output token and different proofs",
    )
    if public.status == 0:
        served(public, ("served_model_files",), file)

    # ---- test options
    not_created("test options: cp_commit does not know the option", commit("tests_commit", [OPTION, file], build=tests), 2, "usage: cp_commit")
    ended, said, written, _ = prove(other, "tests_option", [OPTION, other_file], build=tests)
    check(ended != 0 and "unknown option " + OPTION in said and not written, "test options: cp_prove does not know the option", "status %s; %s" % (ended, said[-300:]))
    ended, said, written, _ = prove(other, "tests_line", [], build=tests)
    check(
        ended != 0 and NAME in said and "This build does not serve such a commitment" in said and not written,
        "test options: cp_prove rejects a commitment that has the line", "status %s; %s" % (ended, said[-300:]),
    )

    print("ALL AS EXPECTED" if not differ[0] else "%d CHECKS DIFFER" % differ[0])
    return 1 if differ[0] else 0


if __name__ == "__main__":
    sys.exit(main())
