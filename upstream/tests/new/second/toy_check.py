#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by tests/new/second/run_honest.sh, which the runner starts; by hand: below a directory of your own.
"""A second verifier, beside the C++ one (toy_verify_file(), blocks/toy/product.h), of the toy run
"product" (<run>: `product` or, smaller, `tiny`): a proof that committed vectors satisfy
a_i * b_i = c_i. Other verifiers in this directory import its field arithmetic and channel (`Run`);
links_check.py also calls report() and cheats(); files_check.py and proof_check.py call new_file().
Its hashes, queries, draws and digests of leaves and vertices are format_rules.py's, which is the
format document (docs/format/FORMAT.md, cited here by rule: R7.9) as a program; its formulas
are taken from the text on that toy (docs/format/toys/1-product.md), not word for word as that page stands,
with its writers' written answers to questions. It calls no code of blocks/.
Where it is not independent of the C++ programs, both rest on the same understanding of a text or
rule, and comparing the two would not show an error in that understanding. It is not independent on
these points: the first query (hashed before the prover's first message), the hash chain (the
Fiat-Shamir transcript), the order of the blocks (the protocol's steps), the fields of a plan entry,
a stage and a code, the place of random tail and message in a codeword and, for --cheats and the
comparison of both verifiers' checks, the malicious provers' switches (not among these sources) and the shapes
of the masks sent in the stages' trees. Nor is it independent of these files, in what they fix
(paths under blocks/): toy/product.h, the C++ verifier itself (also the toy's oracle message, the
sumcheck's summand, a row's place in the main table and the three claims that the sumcheck leaves);
toy/handover.cpp, the C++ tool that writes its input files; chain/public.h, chain/chain.h,
iop/message.h, numbers/setting.h; parts of chain/rounds.h, hash/hash.h, iop/context.h,
numbers/plan.h (a round's fields) and numbers/blocks.h (a stage's choice); the top comment of
iop/iopp.h; the first lines of `Code` in code/code.h (its three fields; where the message's
coefficients begin); `Codes` in iop/relation.h; and, for --cheats and the comparison of checks,
lines of iop/fold.h, iop/base_case.h, iop/psc.h, iop/send.h, iop/cs.h and engine/engine_cpu.h.
Nor is it independent of the C++ programs in what `Run` has for the other verifiers (a chain's
label, a statement, given random bytes, stored trees, a seal, the plan entry `RECORD`, several
chains in one file) or in its record of one value at each tree vertex (R7.14).
Those parts are not of the program's first writing. Nor are report(), made
from main() to serve other toy runs, the comparison of plan.tsv and codes.tsv with the parameters'
bytes, --cheats and the comparison of both verifiers' checks. None of these is a formula of
verify(). The formulas of verify() and of what it calls stand as they stood when --cheats and the
comparison of checks were written, but for two assertions in verify(): it has none on a stage's fifth
word (a stage has four words), and the one on par["choices"] is narrower: == (1,).
This program and blocks/toy/handover.cpp have one author, who wrote none of the blocks.
The algebra of the blocks, the codes' points and the Merkle path check come from the text on the
toy and the format document. Whether this program is independent of the rest has not been examined:
of every other file of the C++ programs, in blocks/ or elsewhere, and of the other parts of the
files named above only in part, nothing is known either way.

It takes <run>.parameters (plan, codes, shapes) as given, compares <run>.configuration only with
its digest there, and computes no soundness error. With --write it writes four lists of what it
hashed and read (<run>.*.tsv) beside its input files, replacing files of these names; that directory
must belong to the caller and give group and others no write permission (see new_file()). Exit code
0 if all agrees, otherwise 1.
usage: toy_check.py <directory of format_rules.py> <directory of the files> <run> [--write]
       toy_check.py <directory of format_rules.py> <directory of the files> <run>
                    --cheats <directory of malicious provers' proofs> <list of expected checks>
"""
import contextlib
import os
import stat
import struct
import sys

sys.path.append(sys.argv[1])   # last: modules beside this script take precedence
import format_rules as R  # noqa: E402

# F is the Goldilocks field, p = 2^64 - 2^32 + 1; the format document numbers it 2 (R1.2, R4.3).
P = R.PRIME[2]


# A proof or file is rejected. Asserts guard the parameters and this program's bookkeeping.
class Refused(Exception):
    pass


@contextlib.contextmanager
def new_file(path, mode="w"):
    """Opens a new file for writing, which gets the name `path` only if the with-block ends without
    an exception; otherwise the new file is removed and whatever was at `path` stays as it was.
    The new file is created exclusively in the same directory, under a name of its own, and then
    renamed. So nothing is written into, or through, whatever is at `path` already: a file or a
    link of that name is replaced. A process that a signal kills within the with-block leaves
    <path>.<pid>.new behind.
    `folder`, the path of that directory, must name a directory and not a link to one. The directory
    must belong to the account that runs the program (the effective user) and give neither its group
    nor others write permission: an account that owns it or can write in it could replace what is
    written there. Otherwise the program exits (SystemExit) with a message that names the first
    condition that fails, and nothing is written; if nothing is at `folder`, os.lstat() raises
    FileNotFoundError. The directories above `folder` are not checked."""
    folder = os.path.dirname(path) or "."
    state = os.lstat(folder)
    why = ("is a link or is not a folder" if not stat.S_ISDIR(state.st_mode)
           else "belongs to another account, which could replace what is written there" if state.st_uid != os.geteuid()
           else "can be written by its group or by others, who could replace what is written there" if state.st_mode & 0o022 else None)
    if why:
        raise SystemExit("nothing is written into %s: it %s. Name a folder of your own that is no link and in which only you can write "
                         "(a new one: mkdir -m 700)." % (folder, why))
    beside = "%s.%d.new" % (path, os.getpid())
    handle = os.open(beside, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666)  # fails if that name is taken; the mode follows the umask, as for any new file
    try:
        with os.fdopen(handle, mode) as o:
            yield o
        os.replace(beside, path)
    except BaseException:
        os.unlink(beside)
        raise


# ---- E = F[X] / (X^3 - 2) ----
def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P, (a[2] + b[2]) % P)


def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P, (a[2] - b[2]) % P)


def mul(a, b):
    a0, a1, a2 = a
    b0, b1, b2 = b
    return ((a0 * b0 + 2 * (a1 * b2 + a2 * b1)) % P, (a0 * b1 + a1 * b0 + 2 * a2 * b2) % P, (a0 * b2 + a1 * b1 + a2 * b0) % P)


def scale(f, a):  # f in F
    return (f * a[0] % P, f * a[1] % P, f * a[2] % P)


ZERO, ONE = (0, 0, 0), (1, 0, 0)


def emb(f):
    return (f % P, 0, 0)


def inner(u, v):
    s0 = s1 = s2 = 0
    for a, b in zip(u, v):
        if a == ZERO or b == ZERO:
            continue
        a0, a1, a2 = a
        b0, b1, b2 = b
        s0 += a0 * b0 + 2 * (a1 * b2 + a2 * b1)
        s1 += a0 * b1 + a1 * b0 + 2 * a2 * b2
        s2 += a0 * b2 + a1 * b1 + a2 * b0
    return (s0 % P, s1 % P, s2 % P)


def eq_table(z):
    """eq(j; z) for all j; bit 0 of j goes with z[0]."""
    t = [ONE]
    for x in z:
        lo = sub(ONE, x)
        t = [mul(v, lo) for v in t] + [mul(v, x) for v in t]
    return t


def fold(table, g):
    """Folds a table of n = 2^len(g) rows: out[c] = sum over b of eq(b; g) table[b + n c]."""
    e, n = eq_table(g), 1 << len(g)
    return [inner(e, table[c:c + n]) for c in range(0, len(table), n)]


def log2(n):
    assert n > 0 and n & (n - 1) == 0
    return n.bit_length() - 1


# ---- codes ----
# A code is (a, l, r): length 2^a, l message entries, r entries of random tail.
# W32 = 7^(5 (2^32 - 1)); ROOT[j] has order 2^j.
W32 = 0xE2AF70635744FD26
ROOT = {32: W32}
for j in range(31, 0, -1):
    ROOT[j] = ROOT[j + 1] * ROOT[j + 1] % P


def point(x):
    z, i = 7, 0
    while x >> i:
        if x >> i & 1:
            z = z * ROOT[i + 1] % P
        i += 1
    return z


def powers(z, n):
    out, v = [], 1
    for _ in range(n):
        out.append(v)
        v = v * z % P
    return out


# The symbol at position x; the tail has the low coefficients.
def symbol(code, x, message, tail):
    pw = powers(point(x), code[2] + code[1])
    s = ZERO
    for j, t in enumerate(tail):
        s = add(s, scale(pw[j], t))
    for j, m in enumerate(message):
        s = add(s, scale(pw[code[2] + j], m))
    return s


# ---- reading ----
class Bytes:
    def __init__(self, b):
        self.b, self.at = b, 0

    def take(self, n):
        if self.at + n > len(self.b):
            raise Refused("the file ends before its last item")
        self.at += n
        return self.b[self.at - n:self.at]

    def word(self):
        return struct.unpack("<Q", self.take(8))[0]

    def text(self):
        return self.take(self.word()).decode("ascii")


def base_of(b):
    w = struct.unpack("<Q", b)[0]
    if w >= P:
        raise Refused("eight bytes that are no element's spelling")
    return w


def ext_of(b):
    return tuple(base_of(b[i:i + 8]) for i in (0, 8, 16))


# Round name -> kind number (R5.2); other verifiers add theirs.
KIND = {"SEND seed": 3, "SEND answer": 7, "ZERO": 1, "PSC epsilon": 2, "PSC round": 1, "PSC Y": 1, "PSC values": 7, "BAT": 1, "FOLD epsilon": 2,
        "FOLD round": 1, "CS": 4, "BASE lambda": 1, "BASE": 4, "SEAL": 6}
ENTRY = ("count", "work", "bytes", "roots", "answers", "draw_E", "nonzero", "positions", "log_n", "trits", "rows")


# "head" (R4.3): field, hash, salt, nonce and position bytes, answers to a sample, 1, plan entries.
# "notice": base code, mask code, main rows, degree, mask rows, answers, leaf class.
# A stage: the indices of its base and mask codes, then k and h: 2^k main rows, 2^h mask rows.
# "choices": 1 if the folds' fresh masks and BASE's masking word are sent in their stages' trees.
def parse_parameters(raw):
    b = Bytes(raw)
    out = {}
    assert b.text() == "the format's version" and b.word() == 8 and b.take(8) == b"H821" + bytes(4)
    head = [b.word() for _ in range(8)]
    out["head"] = head
    assert head[0] == 2 and head[1] == 1 and head[6] == 1
    out["rounds"] = []
    for _ in range(head[7]):
        name = b.text()
        out["rounds"].append(dict(zip(ENTRY, [b.word() for _ in range(11)]), name=name))
    items = []
    while b.at < len(raw):
        name = b.text()
        items.append((name, b.take(b.word())))
    out["items"] = items
    assert [n for n, _ in items] == ["configuration", "codes", "the oracle message a, b, c", "the stages of the proximity test", "the choices of the proximity test"]
    v = Bytes(items[1][1])
    out["base"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    out["mask"] = [tuple(v.word() for _ in range(3)) for _ in range(v.word())]
    out["notice"] = struct.unpack("<7Q", items[2][1])
    v = Bytes(items[3][1])
    n = v.word()
    out["stages"] = [tuple(v.word() for _ in range(4)) for _ in range(n)]
    out["choices"] = struct.unpack("<1Q", items[4][1])
    again = R.parameters_head(2, R.plan_bytes(2, head[2], head[3], head[4], head[5], head[6], [(r["name"],) + tuple(r[k] for k in ENTRY) for r in out["rounds"]]))
    again += R.item("configuration", items[0][1]) + R.item("codes", R.codes_bytes(out["base"], out["mask"]))
    again += R.item("the oracle message a, b, c", R.notice_bytes(*out["notice"])) + R.item("the stages of the proximity test", R.stages_bytes(out["stages"]))
    again += R.item("the choices of the proximity test", b"".join(R.u64(x) for x in out["choices"]))
    assert again == raw, "the parameters' bytes are not what the page's rules spell"
    return out


class Run:
    """The verifier's channel for one hash chain: reads the prover's messages, compares each round
    with its plan entry, makes the queries, draws, checks openings. Lists the rounds, leaves and
    vertices hashed."""

    def __init__(self, par, raw_parameters, configuration_text, proof, label=1, statement=b"", random=None):
        """proof: bytes, or a `Bytes` that the file's earlier chains have read from. random: the
        chain's 32 random bytes if its first query has them as an item (R4.5 (b))."""
        self.par, self.f, self.next, self.label = par, proof if isinstance(proof, Bytes) else Bytes(proof), 0, label
        # configuration_text None: a test fixture with no public file; the digest is taken as given
        self.conf = par["items"][0][1] if configuration_text is None else R.configuration_digest([configuration_text])
        if self.conf != par["items"][0][1]:
            raise Refused("the item configuration is not the digest of the public file")
        self.pd = R.parameters_digest(raw_parameters)
        self.random = self.f.take(32) if random is None else random
        self.run = R.run_value(label, self.pd, self.random)
        if self.run == bytes(32):
            raise Refused("a run's value of zeros")
        self.first_query = self.pd + R.u64(len(statement)) + statement + R.u64(0) + R.u64(1) + self.run
        self.chain = R.Chain(R.first_value(label, self.pd, statement, self.run), 2, label)
        self.rho0 = self.chain.rho
        self.rounds, self.leaves, self.vertices, self.fields = [], [], [], [(0, 32, "the chain's random bytes")]
        self.values = {}   # root -> {2^depth + place: value} (R7.14)
        self.message = b""

    def recv(self, what, n=1):
        at = self.f.at
        out = [ext_of(self.f.take(24)) for _ in range(n)]
        self.message += self.f.b[at:self.f.at]
        self.fields.append((at, 24 * n, what))
        return out

    def recv_root(self, what):
        self.fields.append((self.f.at, 32, what))
        r = self.f.take(32)
        self.message += r
        return r

    def close(self, name):
        while self.par["rounds"][self.next]["name"] == "RECORD":   # a stored table's own sample: no round of this chain (R4.3)
            self.next += 1
            self.passed = getattr(self, "passed", 0) + 1
        e = self.par["rounds"][self.next]
        self.next += 1
        assert e["name"] == name and e["bytes"] == len(self.message), (name, e, len(self.message))
        nonce = 0
        if e["work"]:
            self.fields.append((self.f.at, self.par["head"][3], "nonce of round %d" % (self.chain.round + 1)))
            nonce = int.from_bytes(self.f.take(self.par["head"][3]), "little")
        before = self.chain.rho
        if not self.chain.query(self.message, KIND[name], e["work"], nonce):
            raise Refused("a nonce has not the work of its round")
        rec = dict(round=self.chain.round, name=name, kind=KIND[name], work=e["work"], nonce=nonce, before=before, message=self.message, after=self.chain.rho, E=[], X=[])
        self.rounds.append(rec)
        self.message, self.entry, self.rec = b"", e, rec
        return e

    def element(self, nonzero=False):
        x = self.chain.element(nonzero)
        self.rec["E"].append(x)
        self.rec["words"] = self.chain.taken
        return x

    def positions(self):
        xs = [self.chain.position(self.entry["log_n"]) for _ in range(self.entry["positions"])]
        self.rec["X"] = xs
        self.rec["words"] = self.chain.taken
        return xs

    def opening(self, tree, depth, width, root, draws, stored=False):
        """stored: a tree that was made once and is read by many runs: its leaves have the hash
        number 174 and start from the configuration digest."""
        xs, salt, columns, digests = sorted(set(draws)), self.par["head"][2], {}, []
        number, begin = (174, self.conf) if stored else (164, self.run)
        start = self.f.at
        for x in xs:
            col = [base_of(self.f.take(8)) for _ in range(width)]
            s = self.f.take(salt)
            columns[x] = col
            data = R.leaf_bytes(col, s)
            d = R.leaf(number, tree, depth, x, begin, col, s)
            lab = R.label(number, tree, (1 << depth) + x, len(col), len(s) // 8)   # a leaf's label also holds the column's number of elements and the salt's words
            self.leaves.append((tree, x, lab + data, d, begin))
            digests.append(d)
        given_at = self.f.at

        seen = []

        def given(d, i):
            v = self.f.take(32)
            seen.append((d, i))
            return v

        # R7.14, one value at a vertex: one record for each root that this `Run` (one chain) reads,
        # not, as R7.14 asks, for all chains of one verdict: zkpy.py's opening() has that. Leaves
        # and given digests are vertices. Only what an accepted opening shows is entered.
        record, fresh = self.values.setdefault(root, {}), {}

        def enter(d, i, value):
            if record.get((1 << d) + i, value) != value or fresh.get((1 << d) + i, value) != value:
                raise Refused("a vertex of a tree gets two values under one root")
            fresh[(1 << d) + i] = value

        # the Merkle path check of R7.9
        now = list(zip(xs, digests))
        for d in range(depth, 0, -1):
            for i, val in now:
                enter(d, i, val)
            nxt, v = [], 0
            while v < len(now):
                i, val = now[v]
                if i % 2 == 0 and v + 1 < len(now) and now[v + 1][0] == i + 1:
                    left, right = val, now[v + 1][1]
                    v += 2
                else:
                    other = given(d, i ^ 1)
                    enter(d, i ^ 1, other)
                    left, right = (other, val) if i & 1 else (val, other)
                    v += 1
                parent = R.vertex(self.conf, left, right)
                self.vertices.append((tree, d - 1, i >> 1, R.label(166, 0, 0) + left + right, parent))
                nxt.append((i >> 1, parent))
            now = nxt
        if now[0][1] != root:
            raise Refused("an opening does not lead to its root")
        record.update(fresh)
        self.fields.append((start, given_at - start, "opening of tree %d: %d columns of %d elements of F, each with its salt" % (tree, len(xs), width)))
        self.fields.append((given_at, self.f.at - given_at, "opening of tree %d: %d given digests: %s" % (tree, len(seen), " ".join("(%d,%d)" % s for s in seen))))
        return columns

    def seal(self):
        """Ends a chain that other chains build on (R6.8)."""
        self.close("SEAL")
        return self.chain.seal_digest()

    def finish(self, last=True):
        assert self.next == len(self.par["rounds"]) and self.message == b""
        if last and self.f.at != len(self.f.b):
            raise Refused("bytes are left over")


# The linear claim (`row` in identifiers) of an out-of-domain sample, a random linear form of an
# oracle message with seed (z, beta): coefficients u on the main table and w on the mask table,
# whose last entry, the pad, has beta.
def sample_row(z, beta, bits_main, bits_mask):
    a = max(bits_main, bits_mask) + 1
    assert len(z) == a
    front_m, front_z = sub(ONE, z[a - 1]), z[a - 1]
    for t in range(bits_main, a - 1):
        front_m = mul(front_m, sub(ONE, z[t]))
    for t in range(bits_mask, a - 1):
        front_z = mul(front_z, sub(ONE, z[t]))
    u = [mul(front_m, e) for e in eq_table(z[:bits_main])]
    w = [mul(front_z, e) for e in eq_table(z[:bits_mask])]
    w[-1] = beta
    return u, w


def seed(ch, coordinates):
    e = ch.entry
    assert e["draw_E"] == coordinates + 1 and e["nonzero"] == 1, (e, coordinates)
    return [ch.element() for _ in range(coordinates)], ch.element(True)


# u(g): its inner product with FOLD's folded fresh mask row is the hiding polynomial's value at g.
def mask_weights(g, k, h, length):
    A = lambda l: prod(g[l:])  # noqa: E731   A_l = the product of g_j, j > l (g_1 is g[0])
    u = [A(h)] + [mul(g[l - 1], A(h)) for l in range(1, h + 1)]
    if h < k:
        u.append(mul(mul(g[h], g[h]), A(h + 1)))
        for l in range(h + 2, k + 1):
            u += [mul(g[l - 1], A(l)), mul(mul(g[l - 1], g[l - 1]), A(l))]
        u.append(ONE)
    return u + [ZERO] * (length - len(u))


def prod(v):
    out = ONE
    for x in v:
        out = mul(out, x)
    return out


# FOLD: reduces the claim <c, main table> + <d, mask table> = tau to one on three single rows.
# Follows the verifier's side of the zero-knowledge sumcheck IOR of
# the paper that this protocol builds on (see THIRD_PARTY.md).
def fold_block(ch, k, h, c, d, tau, mask_length):
    sigma, = ch.recv("FOLD: the hiding polynomial's sum")
    ch.close("FOLD epsilon")
    eps = ch.element()
    claim, g = add(sigma, mul(eps, tau)), []
    for j in range(k):
        q0, q2 = ch.recv("FOLD: round %d: the constant and the quadratic coefficient" % (j + 1), 2)
        ch.close("FOLD round")
        x = ch.element()
        q1 = sub(sub(claim, scale(2, q0)), q2)
        claim = add(q0, mul(x, add(q1, mul(x, q2))))
        g.append(x)
    c1 = [mul(eps, v) for v in fold(c, g)]
    f = mul(eps, prod(g[h:]))
    d1 = [mul(f, v) for v in fold(d, g[:h])]
    return g, c1, d1, mask_weights(g, k, h, mask_length), claim


# The protocol's blocks, in order: SEND (the oracle message a, b, c and one out-of-domain sample),
# ZERO and PSC (the sumcheck with a hiding polynomial, for a * b = c), then the proximity test:
# BAT (batches four linear claims into one), FOLD, CS (the code switch), FOLD, CS, BASE (the last
# rows are sent masked and compared with opened symbols). No claim is tested before BASE; from BAT
# on there is one claim, (c, d, tau).
def verify(par, raw_parameters, configuration_text, proof, notes):
    ch = Run(par, raw_parameters, configuration_text, proof)
    base, mask, st = par["base"], par["mask"], par["stages"]
    assert par["notice"] == (0, 0, 3, 1, 1, 1, 0) and par["choices"] == (1,) and par["head"][5] == 1 and len(st) == 3
    C, Z = [base[s[0]] for s in st], [mask[s[1]] for s in st]
    for i in range(3):
        assert C[i][0] == Z[i][0], "this program is for mask words as long as main words"
    L, lzk = log2(C[0][1]), Z[0][1]
    roots = [None] * 3
    # SEND
    roots[0] = ch.recv_root("SEND: the root of tree 0")
    ch.close("SEND seed")
    z, beta = seed(ch, max(2 + L, log2(lzk)) + 1)
    y, = ch.recv("SEND: the answer to the sample")
    ch.close("SEND answer")
    u, w = sample_row(z, beta, 2 + L, log2(lzk))
    # ZERO
    ch.close("ZERO")
    zeta = [ch.element() for _ in range(L)]
    # PSC
    mu, = ch.recv("PSC: the hiding polynomial's sum")
    ch.close("PSC epsilon")
    eps = ch.element()
    claim, gamma = mu, []
    for j in range(L):
        q1, q2 = ch.recv("PSC: round %d: the linear and the quadratic coefficient" % (j + 1), 2)
        ch.close("PSC round")
        x = ch.element()
        q0 = sub(claim, mul(zeta[j], add(q1, q2)))
        claim = add(q0, mul(x, add(q1, mul(x, q2))))
        gamma.append(x)
    p1, p2 = ch.recv("PSC: the round in Y: the linear and the quadratic coefficient", 2)
    ch.close("PSC Y")
    by = ch.element(True)
    claim = add(claim, mul(by, add(p1, mul(by, p2))))
    v1, v2 = ch.recv("PSC: the two end values", 2)
    ch.close("PSC values")
    # PSC's three claims, each (c, d, value; printed: target); main row r's entry i is at r + 4 i.
    e = eq_table(gamma)
    n_main = 4 << L
    rows = []
    cU, dU = [ZERO] * n_main, [ZERO] * lzk
    for i in range(1 << L):
        cU[2 + 4 * i] = sub(ZERO, mul(eps, e[i]))
    for l in range(L):
        dU[2 * l], dU[2 * l + 1] = gamma[l], mul(gamma[l], gamma[l])
    dU[2 * L], dU[2 * L + 1], dU[2 * L + 2] = ONE, by, mul(by, by)
    rows.append((cU, dU, sub(claim, mul(eps, mul(v1, v2)))))
    for r, v in ((0, v1), (1, v2)):
        c_, d_ = [ZERO] * n_main, [ZERO] * lzk
        for i in range(1 << L):
            c_[r + 4 * i] = e[i]
        d_[2 * L + 3 + r] = by
        rows.append((c_, d_, v))
    rows.append((u, w, y))
    # BAT
    ch.close("BAT")
    assert ch.entry["rows"] == len(rows)
    x = ch.element()
    c, d, tau, f = [ZERO] * n_main, [ZERO] * lzk, ZERO, ONE
    for c_, d_, t_ in rows:
        c = [add(a, mul(f, b)) if b != ZERO else a for a, b in zip(c, c_)]
        d = [add(a, mul(f, b)) if b != ZERO else a for a, b in zip(d, d_)]
        tau, f = add(tau, mul(f, t_)), mul(f, x)
    notes.append(("BAT: target", tau))
    for s in range(2):  # FOLD and CS from stage s into stage s + 1
        k, h = st[s][2], st[s][3]
        k2, h2 = st[s + 1][2], st[s + 1][3]
        g, c0, d0, d1, tau = fold_block(ch, k, h, c, d, tau, Z[s][1])
        notes.append(("FOLD: target", tau))
        l, r, lz, rz = C[s][1], C[s][2], Z[s][1], Z[s][2]
        assert C[s + 1][1] << k2 == l and (Z[s + 1][1] << h2) >= 2 * lz + r + 2 * rz + 1
        roots[s + 1] = ch.recv_root("CS: the root of tree %d" % (s + 1))
        ch.close("SEND seed")
        bits_main, bits_mask = log2(l), h2 + log2(Z[s + 1][1])
        z, beta = seed(ch, max(bits_main, bits_mask) + 1)
        alpha, = ch.recv("CS: the answer to the sample")
        ch.close("CS")
        draws = ch.positions()
        theta = ch.element()
        main_symbols = 3 if s == 0 else 3 << k
        width = main_symbols + 2 * (3 << h)
        col = ch.opening(s, C[s][0], width, roots[s], draws)
        assert ch.entry["rows"] == 2 + 3 * len(draws)
        room = Z[s + 1][1] << h2
        # The new mask table: the folded old mask row at 0, the folded fresh one at A1, then the
        # random tails of the folded main row (T) and of these two (T0, T1).
        A1, T, T0, T1 = lz, 2 * lz, 2 * lz + r, 2 * lz + r + rz
        nc, nd = list(c0), [ZERO] * room
        for j in range(lz):
            nd[j], nd[A1 + j] = d0[j], d1[j]
        ntau = tau
        u, w = sample_row(z, beta, bits_main, bits_mask)
        f = theta
        nc = [add(a, mul(f, b)) for a, b in zip(nc, u)]
        nd = [add(a, mul(f, b)) for a, b in zip(nd, w)]
        ntau = add(ntau, mul(f, alpha))
        eg, egh = eq_table(g), eq_table(g[:h])
        for n, xx in enumerate(draws):
            cc = col[xx]
            if s == 0:
                main = [emb(cc[0]), emb(cc[1]), emb(cc[2]), ZERO]
            else:
                main = [tuple(cc[3 * b:3 * b + 3]) for b in range(1 << k)]
            old = [tuple(cc[main_symbols + 3 * b:main_symbols + 3 * b + 3]) for b in range(1 << h)]
            fresh = [tuple(cc[main_symbols + (3 << h) + 3 * b:main_symbols + (3 << h) + 3 * b + 3]) for b in range(1 << h)]
            ys = (inner(eg, main), inner(egh, old), inner(egh, fresh))
            if n == 0:
                notes += [("CS: y_main at the first draw", ys[0]), ("CS: y_0 at the first draw", ys[1]), ("CS: y_1 at the first draw", ys[2])]
            pw = powers(point(xx), max(r + l, rz + lz))
            f = mul(f, theta)
            for j in range(l):
                nc[j] = add(nc[j], scale(pw[r + j], f))
            for j in range(r):
                nd[T + j] = add(nd[T + j], scale(pw[j], f))
            ntau = add(ntau, mul(f, ys[0]))
            for which, (A, TT) in enumerate(((0, T0), (A1, T1))):
                f = mul(f, theta)
                for j in range(lz):
                    nd[A + j] = add(nd[A + j], scale(pw[rz + j], f))
                for j in range(rz):
                    nd[TT + j] = add(nd[TT + j], scale(pw[j], f))
                ntau = add(ntau, mul(f, ys[1 + which]))
        c, d, tau = nc, nd, ntau
        notes += [("CS: target", tau), ("CS: c'[0]", c[0]), ("CS: d'[0]", d[0]), ("CS: d's last entry", d[-1])]
    # BASE
    # M0, R0: message and tail of the masked main word (code C[2]); M1, R1: of the masked mask word
    # (Z[2]). Masked: its symbols must equal the masking word's + lam * the last oracle message's.
    l, r, lz, rz = C[2][1], C[2][2], Z[2][1], Z[2][2]
    tstar, = ch.recv("BASE: the masking word's target")
    ch.close("BASE lambda")
    lam = ch.element()
    M0, M1 = ch.recv("BASE: the masked main message", l), ch.recv("BASE: the masked mask message", lz)
    R0, R1 = ch.recv("BASE: the masked main tail", r), ch.recv("BASE: the masked mask tail", rz)
    left, right = add(inner(c, M0), inner(d, M1)), add(tstar, mul(lam, tau))
    notes += [("BASE: left side of the linear check", left), ("BASE: right side of the linear check", right)]
    if left != right:
        raise Refused("BASE: the linear check fails")
    ch.close("BASE")
    draws = ch.positions()
    col = ch.opening(2, C[2][0], 12, roots[2], draws)
    for xx in sorted(set(draws)):
        cc = col[xx]
        # the last oracle message's and the masking word's main symbols, then their mask symbols
        g3m, gsm, g3z, gsz = (tuple(cc[3 * i:3 * i + 3]) for i in range(4))
        if symbol(C[2], xx, M0, R0) != add(gsm, mul(lam, g3m)) or symbol(Z[2], xx, M1, R1) != add(gsz, mul(lam, g3z)):
            raise Refused("BASE: a check at a position fails")
    ch.finish()
    return ch


def dec(e):
    return " ".join(map(str, e))


# This verifier's rejection messages, by check, as defined in
# docs/format/toys/1-product.md, "5. The verifier's decision".
# Run.__init__()'s two messages have no entry: an exception ends report() or cheats().
CHECK = {"a nonce has not the work of its round": "D2", "an opening does not lead to its root": "D3", "a vertex of a tree gets two values under one root": "D3b", "BASE: the linear check fails": "D4",
         "BASE: a check at a position fails": "D5", "eight bytes that are no element's spelling": "D1", "the file ends before its last item": "D1", "bytes are left over": "D1"}


# The C++ verifier's rejection messages, by check.
THEIRS = {"a nonce does not give the proof of work that its round asks for": "D2", "an opening does not lead to its root": "D3",
          "BASE: the masked message in the proof does not satisfy the linear claim": "D4",
          "BASE: the encoding of the masked message differs from the opened symbols at a listed position": "D5",
          "an opening, or another answer to the verifier's queries, is cut short or not in canonical form": "D1", "bytes are left over at the end of the proof": "D1",
          "an opening is cut short or holds a symbol that is not in canonical form": "D1"}


def cheats(folder, run, where, expected_file, parse_parameters=None, verify=None):
    """Malicious ("cheating") provers' proofs: the given verifier and the C++ verifier (its message:
    the .txt file beside the proof) must reject each at the check that expected_file names.
    folder, where: the directories of the run's files and of these proofs. A name in expected_file
    without a proof, or no proof at all, is no failure."""
    import glob
    import os
    parse_parameters, verify = parse_parameters or globals()["parse_parameters"], verify or globals()["verify"]
    expected = {}
    for l in open(expected_file).read().splitlines():
        if l.startswith("#") or l.startswith("name\t"):
            continue
        name, places, _, _, exp, _ = l.split("\t")
        for p in places.split(", "):
            expected[name if p == "-" else name + "-" + p] = exp.split()[-1]
    raw, conf = open("%s/%s.parameters" % (folder, run), "rb").read(), open("%s/%s.configuration" % (folder, run), "rb").read()
    par, bad, out = parse_parameters(raw), 0, []
    for f in sorted(glob.glob("%s/%s.cheat-*.proof" % (glob.escape(where), glob.escape(run)))):
        tag = os.path.basename(f)[len(run) + 7:-6]
        theirs = open(f[:-6] + ".txt").read().strip()
        try:
            verify(par, raw, conf, open(f, "rb").read(), [])
            mine = "ACCEPTED"
        except Refused as why:
            mine = "%s (%s)" % (CHECK[str(why)], why)
        ok = mine.startswith(expected[tag]) and THEIRS.get(theirs) == expected[tag]
        bad += not ok
        out.append("%s\texpected beforehand: refused at %s\tthe second verifier: %s\tthe C++ verifier: %s\t%s" % (tag, expected[tag], mine, theirs, "as expected" if ok else "NOT AS EXPECTED"))
    print("\n".join(out))
    print("%s: %d cheating provers' proofs, %s" % (run, len(out), "ALL REFUSED AS EXPECTED" if not bad else "%d NOT AS EXPECTED" % bad))
    return 1 if bad else 0


def main():
    folder, run = sys.argv[2], sys.argv[3]
    if len(sys.argv) > 5 and sys.argv[4] == "--cheats":
        return cheats(folder, run, sys.argv[5], sys.argv[6])
    return report(folder, run, parse_parameters, verify)


def report(folder, run, parse_parameters, verify):
    """Checks a run's honest proof and altered copies with the given verifier, and compares its
    values with the C++ tool's lists."""
    read = lambda tail: open("%s/%s.%s" % (folder, run, tail), "rb").read()  # noqa: E731
    assert point(12345) == 4649013750641269600 and point(2) == 1970324836974592 and point(9999) == 9335015899764920215
    raw, conf = read("parameters"), read("configuration")
    assert read("parameters.hex").decode().strip() == raw.hex() and read("statement") == b""
    par = parse_parameters(raw)
    notes = []
    ch = verify(par, raw, conf, read("proof"), notes)
    label = ch.label
    print("%s: the honest proof: ACCEPTED by the second verifier (%d bytes, %d rounds)" % (run, len(ch.f.b), len(ch.rounds)))
    bad = 0
    theirs = [l.split("\t") for l in read("targets.tsv").decode().splitlines()[1:]]
    mine = [[a, dec(b)] for a, b in notes]
    print("targets.tsv: %d lines, %s" % (len(theirs), "all equal" if theirs == mine else "DIFFER"))
    bad += theirs != mine
    # plan.tsv and codes.tsv render the parameters' bytes for reading; the bytes are authoritative.
    rows_ = [l.split("\t") for l in read("plan.tsv").decode().splitlines()]
    col = {n: i for i, n in enumerate(rows_[0])}
    names = (("name", "name"), ("work bits", "work"), ("bytes of the prover's message", "bytes"), ("roots", "roots"), ("answers", "answers"), ("elements of E drawn", "draw_E"),
             ("of these not 0", "nonzero"), ("positions drawn", "positions"), ("log2 of their range", "log_n"), ("entries drawn", "trits"), ("rows batched", "rows"))
    same = len(rows_) - 1 == len(par["rounds"]) and all(str(e[k]) == r_[col[n]] for r_, e in zip(rows_[1:], par["rounds"]) for n, k in names)
    same = same and all(r_[col["its kind's number in the query"]] == str(KIND[e["name"]]) for r_, e in zip(rows_[1:], par["rounds"]) if e["name"] != "RECORD")
    lists = [l.split("\t") for l in read("codes.tsv").decode().splitlines()]
    same = same and [tuple(map(int, l[2:5])) for l in lists if l[0] == "base"] == par["base"] and [tuple(map(int, l[2:5])) for l in lists if l[0] == "mask"] == par["mask"]
    print("plan.tsv and codes.tsv against the parameters' bytes: %s" % ("equal" if same else "DIFFER"))
    bad += not same
    dg = dict(l.split("\t") for l in read("digests.tsv").decode().splitlines()[1:])
    same = dg == {"the configuration's digest": ch.conf.hex(), "the parameters' digest": ch.pd.hex(), "the chain's random bytes": ch.random.hex(),
                  "the run's value": ch.run.hex(), "the chain's first value": ch.rho0.hex()}
    print("digests.tsv: %s" % ("all five equal" if same else "DIFFER"))
    bad += not same
    lines = [l.split("\t") for l in read("chain.tsv").decode().split("\n")[1:] if l]
    differ = 0
    for t, m in zip(lines, ch.rounds):
        call0 = R.H(162, label, m["round"], m["before"], m["message"] + R.u64(m["kind"] + (m["work"] << 8)) + R.u64(m["nonce"]), 64)
        me = [str(m["round"]), m["name"], str(m["kind"]), str(m["work"]), str(m["nonce"]), m["before"].hex(), m["message"].hex(), call0.hex(), m["after"].hex(),
              "; ".join(dec(e) for e in m["E"]), " ".join(map(str, m["X"]))]
        differ += t != me
    print("chain.tsv: %d rounds, %s" % (len(lines), "all equal" if not differ and len(lines) == len(ch.rounds) else "%d DIFFER" % differ))
    bad += differ or len(lines) != len(ch.rounds)
    for l in read("altered.tsv").decode().splitlines()[1:]:
        n, what, at, words = l.split("\t")
        try:
            verify(par, raw, conf, read("altered-%02d.proof" % int(n)), [])
            print("altered %s: ACCEPTED BY THE SECOND VERIFIER" % n)
            bad += 1
        except Refused as why:
            same = CHECK[str(why)] == THEIRS.get(words)
            bad += not same
            print("altered %s: refused at %s: %s  | the C++ verifier: %s: %s" % (n, CHECK[str(why)], why, words, "the same check" if same else "ANOTHER CHECK"))
    if "--write" in sys.argv:
        import hashlib

        def again(start, string, size):
            """Hashes a written string again, by hashlib alone (R2.1). report() asserts that each
            result is the digest written beside that string; otherwise the list is not written."""
            return hashlib.blake2b(string, digest_size=size, salt=start[:16], person=start[16:]).digest()

        assert again(bytes(32), R.label(160, 0, label) + ch.first_query, 32) == ch.rho0, "the printed first query does not hash to the printed first value"
        with new_file("%s/%s.queries.tsv" % (folder, run)) as o:
            o.write("round\tcall\tstart (the chain's value before)\tthe hashed string: label, the prover's message, the word of kind, work and call, the nonce's word\tthe 64 bytes\n")
            o.write("first query (number 160)\t\t\t%s\t%s\n" % ((R.label(160, 0, label) + ch.first_query).hex(), ch.rho0.hex()))
            for m in ch.rounds:
                first = 2 if m["work"] else 1
                blocks = first + (m.get("words", 0) + 3) // 4
                for j in range((blocks + 1) // 2):
                    data = m["message"] + R.u64(m["kind"] + (m["work"] << 8) + (j << 16)) + R.u64(m["nonce"])
                    string, answer = R.label(162, label, m["round"]) + data, R.H(162, label, m["round"], m["before"], data, 64)
                    assert again(m["before"], string, 64) == answer, "a printed call's string does not hash to its printed answer"
                    o.write("%d\t%d\t%s\t%s\t%s\n" % (m["round"], j, m["before"].hex(), string.hex(), answer.hex()))
        with new_file("%s/%s.leaves.tsv" % (folder, run)) as o:
            o.write("tree\tposition\tthe hashed string: label, column, fill, salt (start: the run's value)\tdigest\n")
            for t, x, s, d, begin in ch.leaves:
                assert again(begin, s, 32) == d, "a printed leaf's string does not hash to its printed digest"
                o.write("%d\t%d\t%s\t%s\n" % (t, x, s.hex(), d.hex()))
        with new_file("%s/%s.vertices.tsv" % (folder, run)) as o:
            o.write("tree\tdepth\tplace\tthe hashed string: label, left, right (start: the configuration's digest)\tdigest\n")
            for t, d_, i, s, d in ch.vertices:
                assert again(ch.conf, s, 32) == d, "a printed vertex's string does not hash to its printed digest"
                o.write("%d\t%d\t%d\t%s\t%s\n" % (t, d_, i, s.hex(), d.hex()))
        with new_file("%s/%s.fields-by-the-second-verifier.tsv" % (folder, run)) as o:
            o.write("place\tbytes\twhat\n")
            for f in ch.fields:
                o.write("%d\t%d\t%s\n" % f)
    print("%s: %s" % (run, "ALL AGREES" if not bad else "%d THINGS DIFFER" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
