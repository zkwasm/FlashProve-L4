#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Test: under one process key, no context of the private randomness is created twice, and the
contexts under each key are exactly the rule's (below).

    Started by the runner; by hand: below a directory of your own (name a new work folder there),
    since the script does not check the directories above it. Runner: run_suite.py.
    prover/tests/coin_contexts_listed.py self-test
    prover/tests/coin_contexts_listed.py faults <coin_contexts_faults_test> <new work folder>
    prover/tests/coin_contexts_listed.py programs <folder with cp_commit, cp_prove, cp_verify> <the planting cp_prove, or -> <kit> <new work folder>
    prover/tests/coin_contexts_listed.py services <folder with cp_commit, cp_prove, cp_verify> <kit> <new work folder> [shape <requests> <a batch's> <in step>] [-- <further words of cp_prove> ..] ..

Why. Every private random value (`coin` in names: a random tail, mask, pad or salt) that a proof, a
key file or a leaf file (`ENTRY` in the code) draws through blocks/ comes from a generator created
from a process key and a context (counter, version, object, member), defined in
docs/format/COIN-KEYS.md, "2a. From a key to a random value". A process key belongs to one
batch, or to the creation of one file. A second generator with the same key and context begins again
at the first stream and gives the same values to other messages. Every proof would still verify, and
no self-check of the programs compares two generators: only hiding is lost. Nothing checks this in
every run, by a deliberate choice: this test covers only its own runs, at their shapes.

How. With COINS_CONTEXTS=<file> in the environment, a test build on synthetic weights under a test
seed appends to the file, as each generator is destroyed (prover/hiding/engine_coins.cuh), a head
line and the generator's list of draws (`ledger` in the code and the output). The head line has the
process key's number (in order of first use in the process; a key that returns keeps its number) and
the context's four words; a draw's line has its stream, length and kind. These are public numbers:
no bit of a key or of a random value.

The checks on a listing. One that fails is reported by name (printed: "cried"):
  a-context-twice      no (key, counter, version, object, member) occurs twice
  a-stream-twice       a generator's draws have the streams 16384, 16385, ..: none is opened twice
  a-foreign-object     the object is 2 (a request), 4 (a batch), 5 (creating the key file) or 10
                       (creating a leaf file); other numbers belong to other generators
                       under the same keys
  not-as-foretold      the contexts under each key are exactly the rule's (below)
                       and all contexts have the same version
  other-ledgers        all requests of a listing have equal lists of draws: what is drawn must
                       follow from public shapes, not from a request's values, even if one fails
The rule, in (counter, object, member). The committer program cp_commit, with M leaf tables
(`copies`: further tables of the same weights): 1 + M keys, with (0, 5, 0) under the first and
(0, 10, 0) under each other. The prover program cp_prove: one key for each batch, with (0, 4, 0)
and, for each occupied slot j (place in the batch), (j, 2, j). The rule follows from the programs'
four calls that create a generator, and from public shapes.

`kit`: a configuration's folder, such as prover/examples/target_small_f2. cp_prove needs the
verifier program cp_verify in its own folder.
self-test   Runs no program. On listings written here, each check must fail on the listing spoiled
            for it, and none on the correct listing.
faults      The program given, a test program that is not part of this repository, creates
            generators with engine_coins(), as the programs do: correctly; the batch's context twice
            in a row; that context under keys A, B and A again; two requests under one context. The
            first must fail no check, the others must give a-context-twice, and the last must also
            break the rule. No GPU.
programs    Runs the full programs (test builds) on one GPU that cp_prove accepts. GPU_RUN in the
            environment is a command prefix for cp_commit and cp_prove. Synthetic weights, a test
            seed and -test-coins: nothing is hidden.
              commit     cp_commit -M 2
              batch      one batch of four requests, two in step (-in-step 2: two share each pass
                         over the weights)
              service    one process, two batches of two (a spool)
              again      that batch in a further process on the same secret folder follows the
                         rule. This does not show that two runs have different keys. That rests on
                         each batch's 32 bytes from the system (known by reading
                         begin_the_batch_on_a_leaf() in prover/prover/isk.cuh, and from runs under a
                         drawn seed; not shown by this test): under -test-coins both runs take them
                         from one file.
              failed.*   with the planting cp_prove, which makes slot 1's request in
                         the batch of four fail in three ways that an honest run can meet
                         (each chosen by PLANT in the environment): a combination
                         of an approximate range check (`smallness` in PLANT) leaves its window; a
                         value of the forward pass leaves its range; the same, and a self-check
                         fails too. Each time: r1.failed, and no check on the listing fails, so the
                         failed request draws as the others do. This repository cannot build the
                         planting cp_prove: with "-" in its place the three are skipped.
              above      the ordinary cp_prove: slot 1's request has a range of two positions
                         (r = T - 2, T the request's length) of synthetic tokens, whose summed costs
                         exceed the parameter file's bound: a failure (flags 00000040) that a client
                         can cause at will. Required as for failed.*, with one failed slot logged.
            Each run must end with status 0, and each cp_prove must leave the expected files.
services    One spool of six requests in three batches of two, two in step, or with the numbers
            after "shape"; the last batch need not be full. The spool is served once plainly and
            once for each list of further cp_prove arguments after a "--" (for example: -- -overlap
            -memory lean -- -overlap-two -memory lean), each time from a copy of the spool as first
            written, with the same -test-coins file. Each run: status 0, a proof for every request,
            the rule's contexts. Every listing must also have the plain run's generators with the
            same lists of draws, in any order. These checks catch a slot's generator created under
            the next batch's key, and an earlier batch's opening (its part on the weight table,
            delayed by -overlap) that draws from another batch's generator.

Not run in the mode "programs": overlapping batches; a restart in the middle of a batch; any other
configuration or size; the contexts of the generator in prover/hiding/trees.cuh (object 1),
where set_attempt() exits unless the counter rises.
The last line is ALL AS EXPECTED (status 0) or "<n> THINGS OTHERWISE" (status 1). Status 2: a wrong
call, or a work folder that already exists."""

import hashlib
import os
import shlex
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(HERE))
# COIN_FIRST_STREAM in blocks/coins/coins.h.
FIRST_STREAM = 16384
# ENGINE, ENGINE_BATCH, ENGINE_KEY_FILE, ENGINE_ENTRY in prover/hiding/long_rows_words.h, where
# REQUEST is 1: another generator's object.
REQUEST, BATCH, KEY_FILE, ENTRY = 2, 4, 5, 10
otherwise = 0


def check(good, what, more=""):
    global otherwise
    print("%s %s%s" % ("as expected" if good else "OTHERWISE  ", what, ("\n            " + more) if more and not good else ""), flush=True)
    otherwise += not good
    return good


def read(text):
    """A listing's text -> a list of (key, counter, version, object, member, draws), where draws is
    [(stream, length, kind), ..]. Raises ValueError at a malformed line, or if a head line's count
    of draws is not met."""
    out, lines = [], text.splitlines()
    at = 0
    while at < len(lines):
        w = lines[at].split()
        if len(w) != 13 or w[0] != "context" or w[1::2] != ["key", "counter", "version", "object", "member", "draws"]:
            raise ValueError("line %d is no head line: %s" % (at + 1, lines[at]))
        key, counter, version, obj, member, draws = (int(x) for x in w[2::2])
        ledger = []
        for line in lines[at + 1 : at + 1 + draws]:
            d = line.split()
            if len(d) != 3 or d[2] not in ("elements", "bytes"):
                raise ValueError("no line of a ledger: %s" % line)
            ledger.append((int(d[0]), int(d[1]), d[2]))
        if len(ledger) != draws:
            raise ValueError("the last ledger is cut short")
        out.append((key, counter, version, obj, member, ledger))
        at += 1 + draws
    return out


def cries(listing, foretold):
    """The names of the checks that fail on a listing, sorted. The second argument lists, by key
    number, the expected sets of (counter, object, member)."""
    cried = set()
    heads = [g[:5] for g in listing]
    if len(set(heads)) != len(heads):
        cried.add("a-context-twice")
    for g in listing:
        if [d[0] for d in g[5]] != list(range(FIRST_STREAM, FIRST_STREAM + len(g[5]))):
            cried.add("a-stream-twice")
        if g[3] not in (REQUEST, BATCH, KEY_FILE, ENTRY):
            cried.add("a-foreign-object")
    found = {}
    for g in listing:
        found.setdefault(g[0], set()).add((g[1], g[3], g[4]))
    if found != dict(enumerate(foretold)) or len({g[2] for g in listing}) > 1:
        cried.add("not-as-foretold")
    if len({tuple(g[5]) for g in listing if g[3] == REQUEST}) > 1:
        cried.add("other-ledgers")
    return sorted(cried)


# The rule's contexts under one batch's key.
def of_a_batch(slots):
    return {(0, BATCH, 0)} | {(j, REQUEST, j) for j in range(slots)}


# The rule's contexts under each of cp_commit's keys.
def of_a_commitment(copies):
    return [{(0, KEY_FILE, 0)}] + [{(0, ENTRY, 0)}] * copies


# Passes if exactly the checks in `wanted` fail on the listing at `path`.
def held(what, path, foretold, wanted=()):
    try:
        with open(path) as f:
            listing = read(f.read())
    except (OSError, ValueError) as e:
        return check(False, "%s: the listing can be read" % what, str(e))
    cried = cries(listing, foretold)
    return check(cried == sorted(wanted), "%s: %d generators under %d keys, %d draws: %s" % (what, len(listing), len({g[0] for g in listing}), sum(len(g[5]) for g in listing),
                                                                                           "nothing is cried" if not wanted else "cried: " + ", ".join(wanted)), "cried: %s" % (", ".join(cried) or "nothing"))  # fmt: skip


def self_test():
    def text(generators):
        return "".join("context key %d counter %d version %d object %d member %d draws %d\n" % (*g[:5], len(g[5])) + "".join("%d %d %s\n" % d for d in g[5]) for g in generators)

    ledger = [(FIRST_STREAM, 32, "bytes"), (FIRST_STREAM + 1, 5, "elements")]
    right = [(k, c, 7, o, m, list(ledger)) for k in (0, 1) for c, o, m in ((0, REQUEST, 0), (1, REQUEST, 1), (0, BATCH, 0))]
    foretold = [of_a_batch(2)] * 2
    check(read(text(right)) == right, "a listing that is written and read comes back")
    check(cries(right, foretold) == [], "the right listing: nothing is cried")
    spoiled = {
        "a-context-twice": right + [right[2]],
        "a-stream-twice": right[:1] + [right[1][:5] + ([ledger[0], (FIRST_STREAM, 5, "elements")],)] + right[2:],
        "a-foreign-object": right[:2] + [(0, 0, 7, 1, 0, list(ledger))] + right[3:],
        "not-as-foretold": right[:5],
        "other-ledgers": right[:1] + [right[1][:5] + (ledger + [(FIRST_STREAM + 2, 1, "elements")],)] + right[2:],
    }
    for name, listing in spoiled.items():
        cried = cries(listing, foretold)
        check(name in cried, "spoiled for %s: it is cried" % name, "cried: %s" % cried)
    check(cries(spoiled["a-context-twice"], foretold) == ["a-context-twice"], "a context that stands twice is cried by that name alone")
    check(cries([(0, 0, 7, BATCH, 0, ledger), (1, 0, 7, BATCH, 0, ledger), (0, 0, 7, BATCH, 0, ledger)], [{(0, BATCH, 0)}] * 2) == ["a-context-twice"], "keys A, B, A: cried")
    for bad in ("context key 0\n", text(right)[:-9], "16384 5 coins\n"):
        try:
            read(bad)
            check(False, "a text that is no listing is refused", bad)
        except ValueError:
            check(True, "a text that is no listing is refused")


def faults(program, work):
    work = os.path.abspath(work)
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    for scene, foretold, wanted in (("honest", [of_a_batch(2)] * 2, ()), ("twice", [{(0, BATCH, 0)}], ("a-context-twice",)), ("aba", [{(0, BATCH, 0)}] * 2, ("a-context-twice",)),
                                    ("shared", [of_a_batch(2)], ("a-context-twice", "not-as-foretold"))):  # fmt: skip
        path = os.path.join(work, scene + ".txt")
        done = subprocess.run([program, scene], env={**os.environ, "COINS_CONTEXTS": path}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if check(done.returncode == 0, "%s: the program makes its generators (status 0): no own test of the generators ends it" % scene, done.stdout.decode(errors="replace")[-300:]):
            held(scene, path, foretold, wanted)


def programs(build, planting, kit, work):
    work = os.path.abspath(work)  # made absolute once, so that cp_prove gets absolute names for -isk and -spool
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    gpu = shlex.split(os.environ.get("GPU_RUN", ""))
    with open(os.path.join(kit, "policy.txt")) as f:
        word = dict(line.split() for line in f if len(line.split()) == 2)
    T, V = 1 << int(word["lt"]), int(word["V"])
    seed, coins = os.path.join(work, "seed"), os.path.join(work, "coins")
    with open(seed, "wb") as f:
        f.write(bytes(range(1, 33)))
    with open(coins, "wb") as f:
        f.write(hashlib.sha512(b"coin_contexts_listed").digest() * 2)  # 32 bytes a batch: four batches
    requests = os.path.join(work, "in")
    os.makedirs(requests)
    # A request file: T token ids, r, e, little-endian 32-bit words (request() in driver/spool.py).
    for j in range(4):
        with open(os.path.join(requests, "r%d.req" % j), "wb") as f:
            f.write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1 + 1000 * j) >> 7) % V for t in range(T)], T - 1, T))

    def run(name, words, more=None):
        listing = os.path.join(work, name + ".contexts.txt")
        with open(os.path.join(work, name + ".log"), "wb") as log:
            status = subprocess.run(gpu + words, env={**os.environ, "COINS_CONTEXTS": listing, **(more or {})}, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        return status, listing

    isk = os.path.join(work, "isk")
    status, listing = run("commit", [os.path.join(build, "cp_commit"), "-weights", "made-up-blocks", "-config", kit, "-M", "2", "-ivk", os.path.join(work, "ivk"), "-isk", isk, "-test-seed", seed])
    if not check(status == 0, "commit: cp_commit -M 2: status 0", "status %d: %s" % (status, os.path.join(work, "commit.log"))):
        return
    held("commit", listing, of_a_commitment(2))

    def batch(name, program, more=None, failed=(), folder=requests):
        out = os.path.join(work, "out." + name)
        status, listing = run(name, [program, "-isk", isk, "-in", folder, "-out", out, "-in-step", "2", "-test-coins", coins], more)
        owed = sorted(x for j in range(4) for x in (["r%d.failed" % j] if j in failed else ["r%d.proof" % j, "r%d.y" % j]))
        if check(status == 0 and sorted(os.listdir(out)) == owed, "%s: cp_prove's status 0 and the files that it owes" % name, "status %d; %s" % (status, sorted(os.listdir(out)) if os.path.isdir(out) else "no folder")):
            held(name, listing, [of_a_batch(4)])

    prove = os.path.join(build, "cp_prove")
    batch("batch", prove)
    spool = os.path.join(work, "spool")
    os.mkdir(spool, 0o700)
    for j in range(4):
        subprocess.run([sys.executable, "-B", os.path.join(TREE, "driver", "spool.py"), "lay", spool, os.path.join(requests, "r%d.req" % j)], check=True, stdout=subprocess.DEVNULL)
    status, listing = run("service", [prove, "-isk", isk, "-spool", spool, "-batch", "2", "-in-step", "2", "-most-batches", "2", "-test-coins", coins])
    proofs = len([x for x in os.listdir(os.path.join(spool, "out")) if x.endswith(".proof")])
    if check(status == 0 and proofs == 4, "service: one process, two batches of two: status 0, four proofs", "status %d, %d proofs" % (status, proofs)):
        held("service", listing, [of_a_batch(2)] * 2)
    batch("again", prove)
    above = os.path.join(work, "in.above")
    os.makedirs(above)
    for j in range(4):
        with open(os.path.join(above, "r%d.req" % j), "wb") as f:
            f.write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1 + 1000 * j) >> 7) % V for t in range(T)], T - 2 if j == 1 else T - 1, T))
    batch("above", prove, failed=(1,), folder=above)
    with open(os.path.join(work, "above.log"), errors="replace") as f:
        marks = [line.strip() for line in f if line.startswith("# seat ")]
    check(len(marks) == 1 and "flags 00000040" in marks[0], "above: the one line on a failed seat names the mark above alone (flags 00000040)", "; ".join(marks) or "no line")
    for name, words in (("window", "smallness.combination 0 0 0 2 =200000"), ("range", "ffn.accumulator 0 7 5 1 +1099511627776"), ("range-and-tie", "qkv.accumulator 0 3 5 1 +1099511627776")):
        if planting != "-":
            batch("failed." + name, planting, {"PLANT": words, "LIE_SEAT": "1"}, failed=(1,))
        else:
            print("not played  failed.%s: no planting cp_prove was named" % name)


def services(build, kit, work, variants, requests=6, size=2, in_step=2):
    work = os.path.abspath(work)  # made absolute once, so that cp_prove gets absolute names for -isk and -spool
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    batches = -(-requests // size)  # the last batch has requests % size requests, if that is not 0
    foretold = [of_a_batch(size)] * (requests // size) + ([of_a_batch(requests % size)] if requests % size else [])
    gpu = shlex.split(os.environ.get("GPU_RUN", ""))
    with open(os.path.join(kit, "policy.txt")) as f:
        word = dict(line.split() for line in f if len(line.split()) == 2)
    T, V = 1 << int(word["lt"]), int(word["V"])
    seed, coins, isk = (os.path.join(work, x) for x in ("seed", "coins", "isk"))
    with open(seed, "wb") as f:
        f.write(bytes(range(1, 33)))
    with open(coins, "wb") as f:
        f.write(b"".join(hashlib.sha256(b"coin_contexts_listed %d" % b).digest() for b in range(batches)))  # 32 bytes a batch
    with open(os.path.join(work, "commit.log"), "wb") as log:
        status = subprocess.run(gpu + [os.path.join(build, "cp_commit"), "-weights", "made-up-blocks", "-config", kit, "-M", "1", "-ivk", os.path.join(work, "ivk"), "-isk", isk, "-test-seed", seed],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode  # fmt: skip
    if not check(status == 0, "cp_commit: status 0", "status %d" % status):
        return
    laid = os.path.join(work, "spool.laid")
    os.mkdir(laid, 0o700)
    for j in range(requests):
        path = os.path.join(work, "r%d.req" % j)
        with open(path, "wb") as f:
            f.write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1 + 1000 * j) >> 7) % V for t in range(T)], T - 1, T))
        subprocess.run([sys.executable, "-B", os.path.join(TREE, "driver", "spool.py"), "lay", laid, path], check=True, stdout=subprocess.DEVNULL)
    plain = None
    for n, words in enumerate([[]] + variants):
        name = "plain" if not n else " ".join(words)
        spool, listing = os.path.join(work, "spool.%d" % n), os.path.join(work, "contexts.%d.txt" % n)
        subprocess.run(["cp", "-a", laid, spool], check=True)
        with open(os.path.join(work, "service.%d.log" % n), "wb") as log:
            status = subprocess.run(gpu + [os.path.join(build, "cp_prove"), "-isk", isk, "-spool", spool, "-batch", str(size), "-in-step", str(in_step), "-most-batches", str(batches), "-test-coins", coins] + words,
                                    env={**os.environ, "COINS_CONTEXTS": listing}, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode  # fmt: skip
        proofs = len([x for x in os.listdir(os.path.join(spool, "out")) if x.endswith(".proof")]) if os.path.isdir(os.path.join(spool, "out")) else 0
        if not check(status == 0 and proofs == requests, "%s: status 0, %d proofs" % (name, requests), "status %d, %d proofs: %s" % (status, proofs, os.path.join(work, "service.%d.log" % n))):
            continue
        if held(name, listing, foretold):
            with open(listing) as f:
                generators = sorted((g[:5], tuple(g[5])) for g in read(f.read()))
            if plain is None:
                plain = generators
            else:
                check(generators == plain, "%s: the same generators with the same ledgers as in the plain service" % name)


def main():
    words = sys.argv[1:]
    if words == ["self-test"]:
        self_test()
    elif len(words) >= 4 and words[0] == "services" and not os.path.lexists(os.path.abspath(words[3])) and (len(words) == 4 or words[4] == "--" or (
            len(words) >= 8 and words[4] == "shape" and all(w.isdigit() and int(w) for w in words[5:8]) and (len(words) == 8 or words[8] == "--"))):  # fmt: skip
        shape = [int(w) for w in words[5:8]] if words[4:5] == ["shape"] else [6, 2, 2]
        variants, rest = [], words[8 if words[4:5] == ["shape"] else 4 :]
        for w in rest:
            if w == "--":
                variants.append([])
            else:
                variants[-1].append(w)
        services(words[1], words[2], words[3], variants, *shape)
    elif len(words) == 3 and words[0] == "faults" and not os.path.lexists(os.path.abspath(words[2])):
        faults(*words[1:])
    elif len(words) == 5 and words[0] == "programs" and not os.path.lexists(os.path.abspath(words[4])):
        programs(*words[1:])
    else:
        print(__doc__.split("\n\nWhy.")[0])
        return 2
    print("ALL AS EXPECTED" if not otherwise else "%d THINGS OTHERWISE" % otherwise)
    return 1 if otherwise else 0


if __name__ == "__main__":
    sys.exit(main())
