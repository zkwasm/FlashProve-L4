# The toy run `product`: its protocol and the verifier's decision

A **toy run** is a small run of the library's blocks on synthetic data. Sections 0 to 6 describe the
toy run `product`. Parts 1 to 4, after them, give the verifier's side of each block that it uses, as
mathematics and bytes. Parts 5 to 10, to which some sentences point, are on the two other toy pages:
part 5 in `2-links.md`, parts 6 to 10 in `3-key-and-entry.md`. Rule numbers such as R7.1 are those
of the format page, `FORMAT.md` in the directory above this one, which defines the bytes and the
terms; it calls `product` an example protocol (R12.6). File names are given from the repository's
root (`blocks/iop/send.h`) or from `blocks/` (`chain/rounds.h`). A bare name such as `cs.h` is a
file of `blocks/iop/`, a test's bare name such as `psc_test` is a program of `blocks/iop/tests/`,
and a bare name that ends in `.py` is a file of `tests/new/second/`. A sentence marked "(from the programs' source)" was read in the source and not run. Beyond the runs told in section 6, this page's text has been compared with the programs' code by reading alone, and not all of it as it stands here. No program, the second verifiers among them, has been written from this page's text word for word. It has not been compared with the paper that this protocol builds on: there may be differences that are not listed here, and what it gives as a choice of the programs has not been held against that paper's text and is not claimed as new.

Where this page speaks of hiding, it
gives what the design is meant to do and the reasons of the programs' source. All that this page
tells of a prover's side, and every test of hiding that it names, is the library's CPU reference at
toy size; it says nothing of the GPU engine, the kernels and the sampling of `cp_commit` and
`cp_prove` (from the programs' source, `blocks/tests/hiding/README.md`). Take an outcome of a test of hiding that this page reports as
seen on test builds, not on the released programs.

## 0. What it is

`product` is the smallest whole run of the protocol's machinery: one hash chain, one committed
oracle message, one zero-knowledge sumcheck, one proximity test. It has no model, no key file and no
stored table. The prover knows three vectors a, b, c over F, of length l, with a_i b_i = c_i for all
i. (F is the base field and E its cubic extension: R1.2, R1.4.) The protocol is designed to show
this without showing the vectors. The statement's bytes in the first query are empty: what a
malicious prover gets wrong is the relation between the vectors that it has committed to.

It comes in two sizes. The full size (l = 2^12) is planned for the target of 128 bits, with 20
proof-of-work bits before each sampling of positions. What that plan rests on, the theorem on codes
(Theorem 4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon, Haböck,
Kopparty and Saraf published at STOC 2026) included, is in part 3, section 6. The programs' accounting takes that
theorem as given; a statement on codes held to follow from it is proved in the Lean library (`FORMAT.md`, 13.7, condition 4). The tiny size (l = 2^4) is a test size that
reaches no security level (a target of 10 bits, 6 proof-of-work bits); no check is weakened in it,
and a person can follow it by hand.

The blocks, in the order of the run: SEND (the oracle message a, b, c with its mask row, and one
out-of-domain sample), ZERO and PSC (the sumcheck for a b = c), BAT (four rows into one), then the
proximity test: FOLD, CS, FOLD, CS, BASE. Parts 1 to 4 give each block's verifier as mathematics.

## 1. The first query

The chain's label is 1. The parameters are, in this order: the format's version; the plan, round by
round; the items `configuration` (the digest of the run's one public file, a text of thirteen
lines), `codes`, `the oracle message a, b, c`, `the stages of the proximity test` (four words for
each stage), `the choices of the proximity test` (one word: whether the masks ride). The format page
defines how each of these is encoded: the version and the plan in R4.3; an item in R4.1;
`configuration` in R4.9; `codes` in R12.1 and R12.2; `the oracle message a, b, c`, a notice, in
R12.3; a stage in R12.4; the choices in R12.5. The chain has 32 random bytes at the start of the
proof (case (a) of R4.5).

## 2. The plan

- Full size: 30 rounds, each one query of the chain, counted from 1; the proof has 160,520 bytes.
  There is one root in each of the rounds 1, 23 and 27. Positions are sampled in round 24, `CS`: 73
  in [0, 2^15); in round 28, `CS`: 74 in [0, 2^14); in round 30, `BASE`: 74 in [0, 2^14); with 20
  proof-of-work bits before each. Rows batched: 4 (round 19), 221 (round 24), 224 (round 28).
- Tiny size: 22 rounds, each one query of the chain, counted from 1; the proof has 7,648 bytes.
  There is one root in each of the rounds 1, 15 and 19. Positions are sampled in round 16, `CS`: 3
  in [0, 2^7); in round 20, `CS`: 3 in [0, 2^8); in round 22, `BASE`: 5 in [0, 2^9); with 6
  proof-of-work bits before each. Rows batched: 4 (round 11), 11 (round 16), 11 (round 20).

The proofs' lengths and the numbers of positions come out of the planning routines and, for the
lengths, of the positions that a run samples.

The tool `blocks/toy/handover.cpp` writes the whole plan as a table, `<run>.plan.tsv`, and where
each item of the proof is, `<run>.fields.tsv`. The proof holds, in this order: 32 random bytes; the
root; the answer to the out-of-domain sample; the hiding sum; the sumcheck's round polynomials, two
coefficients each; two coefficients in Y; two end values; then for each of the two folds a hiding
sum, its round polynomials, a root, an answer, a nonce and an opening; the masking word's target;
the last step's shown words in the clear; a nonce; an opening. Openings enter no query; a nonce is
the last word of its own round's query and of no other (R5.2). An opening comes right after the
nonce of the round that sampled its positions.

What may surprise a reader:

- (a) a leaf holds one column of the symbols of several oracle messages, rows over F and rows over E
  side by side;
- (b) some rounds have an empty verifier's message, some an empty prover's message;
- (c) in a round `CS` one verifier's message holds positions and then one element of E;
- (d) positions can repeat: rows follow the draws, in the order sampled, repeats counted, while an
  opening follows the list, the sorted set (R6.6);
- (e) every sumcheck message leaves out the coefficient that the check would fix, so no sumcheck
  equation is ever tested: it is used to get that coefficient back;
- (f) some sampled elements must not be 0 (the beta of each out-of-domain sample; PSC's beta);
- (g) the fresh masks of both folds and the last step's masking word are oracle messages without an
  out-of-domain sample, which ride in the tree of their own stage's rows.

## 3. Trees

There are three, all perfect binary trees with one kind of leaf (column, fill, one salt of 40
bytes). Inside a column all main symbols come first, part by part in the order of sending, then all
mask symbols (R7.1).

| tree | a leaf's column | width in F | opened at |
|---|---|---|---|
| 0 | a(x), b(x), c(x) over F; the message's mask row (E); the first fold's fresh mask row (E) | 3 + 3 + 3 = 9 | the first round `CS` |
| 1 | the switch's two main rows (E, E); its two mask rows; the second fold's two fresh mask rows | 6 + 6 + 6 = 18 | the second round `CS` |
| 2 | the switch's main row, the masking word's main row; the switch's mask row, the masking word's mask row | 6 + 6 = 12 | the round `BASE` |

## 4. Codes (field 2, Goldilocks)

A code is (n = 2^a, l, r). Position x in [0, n) with bits x_0 (lowest), x_1, .. has the point z(x) =
7 times the product, over the set bits i of x, of w_(i+1), where w_j is the root of unity of order
2^j that is got from w_32 = 7^(5 (2^32 - 1)) by squaring 32 - j times; so positions 2m and 2m + 1
are z and -z. The word of a message m (l entries) with random tail rho (r entries) has at x the
symbol sum_(j<r) rho_j z^j + sum_(j<l) m_j z^(r+j). A row over E is encoded coordinate by
coordinate. The codes of a run are in the item `codes`; the tool prints them as `<run>.codes.tsv`.

## 5. The verifier's decision

The verifier accepts if and only if the public file passes the tests at the end of this section and
all of these hold.

- (D1) Form. Every element read is canonical; the file has exactly the planned items; no byte is
  left over; the run's value is not 32 zero bytes (R4.5).
- (D2) Proof of work. In each of the three rounds that sample positions the nonce passes that
  round's proof-of-work test.
- (D3) Openings. For each tree the opened leaves (hashed by the verifier itself from column, fill
  and salt, with its own run's value, tree number and position) and the given digests lead to the
  root that was received.
- (D3b) One value at a vertex (R7.14). No vertex, leaf or given digest may get two values under one
  root. The verifier keeps a record of values at vertices, one for every root's 32 bytes that it
  reads in a session (R7.14: one run of a verifier, which ends in one verdict); only what an
  accepted opening shows is entered.
- (D4) The last step's linear check (part 3).
- (D5) The last step's checks at the positions (part 3): for each sampled position and both rows,
  the encoding of the shown word at that position equals the masking word's opened symbol plus
  lambda times the last oracle message's opened symbol.

Nothing else is tested on the proof's bytes. Everything before the last step only moves a claim. The
verifier carries a list of linear rows with public targets, then one batched row, which is a
function of the plain messages, the sampled elements and, from the first switch on, the opened
symbols. SEND gives the sample row (part 1, section 1); PSC the rows (U), (E_1), (E_2) (part 2,
section 2); BAT one row out of the four; each FOLD a claim on virtual rows; each CS the carried
claim, the new message's sample row and three rows for each sampled position, batched again; the
last step tests what is left.

On D3b. In this toy run each tree is opened at one list, so the clause could act only if two trees
of one proof had the same 32 bytes as their roots. With a real hash no file can tell a verifier that
keeps the record from one that does not: it would take a collision. So no run on a file shows the
clause at work; `tests/new/second/vertex_values_test.py` puts a value into the second verifier's own
record instead, and sees the honest proof rejected; by its head no C++ verifier is tested in this
way. The format page gives the reason (R7.14): two different values at one vertex under one root
mean a collision of the hash. By R7.14 the record is a cheap guard that makes that event visible,
and no bound depends on it.

Before it reads a byte of the proof, the programs' verifier sizes the run from the numbers in the
public file. It rejects a public file that is not, byte for byte, the file that the run sized from
these numbers would write. It also rejects a file that lacks one of its numbers or has one out of
bounds: a format's version other than the programs' own, a `target` above 256, `work-on-positions` above 40, `position-bytes` other than 0, `log-length` outside 1 .. 20, `log-n` or `mask-log-n` above 24 (`toy_from_text` and
`toy_setting` in `blocks/toy/product.h`; from the programs' source). And it rejects parameters whose
round errors miss their target (the errors are the planner's figures: part 1, section 6). It takes
the public file's text and the proof, and nothing else. (The second verifier takes the parameters'
bytes as given: section 6.)

## 6. The files, what has been compared with what, and limits

`bash tests/new/second/run_honest.sh <a new work folder>` builds the three tools, makes the honest
files, their altered copies and two leaf files made false out of an honest one's own bytes, and has
each checked twice. It builds no malicious prover, so it makes no false proof in the sense below.
The false proofs of which this page speaks were made by another script of the programs' tests,
which also builds malicious provers; it is not part of this repository. The names in capitals
that this page gives to false proofs and to injected faults (`ENGINE_FAR_WORD`, say) come from
this project's own tests: the test switches of these names, and the code under them, are not
in this repository and cannot be built from it. Twelve false files of the toy runs
are stored in the repository, in `examples/protocol-toy-false-files/`; among them are one false
proof of `product` and one of `links`, both at the tiny size, in `false/product/` and
`false/links/`; the two runs' `.parameters` and `.configuration` files are in `honest/product/` and
`honest/links/`. The directory's `LIST.tsv` gives each file's SHA-256, what is false in it, the
checker that it was given to and the words of the rejection (from the programs' source). `run_honest.sh` rejects a work directory that exists or that is inside the repository, and
also one whose parent directory does not exist or that is below a directory that another account may
be able to change (from the programs' source; `tests/new/second/README.md`). For each run the tool
`blocks/toy/handover.cpp` writes the public file (`.configuration`), the proof, the bytes of the
first query's parameters (`.parameters`: these bytes decide, and every table only prints them;
`.parameters.hex` has the same bytes as hexadecimal digits), the statement's bytes (`.statement`,
empty for these runs), plan, codes, fields, five digests, every round of the chain (`.chain.tsv`),
nineteen values on the verifier's way (`.targets.tsv`), and thirteen altered copies with the words
of their rejection. A small public number (1, or the one given on the command line) fixes witness
and randomness, so nothing is hidden in these files, and a run can be made again byte for byte.

The second verifier, `tests/new/second/toy_check.py`, calls no code of the programs. On the files
that the script makes it accepts the honest proofs (full, tiny, and tiny under a public number at
which a position is sampled twice in a switch's list; no check in the script looks for the repeat). It finds every digest, every round's query,
answer and sampled values and all nineteen values equal to the C++ tool's. It rejects each altered
copy at the same check as the C++ verifier. A **false proof** is a proof that the run's own prover
has made with one block malicious in one named way and all other blocks honest (from the programs'
source: the head of the list of this run's false proofs, which is not in this repository). The
second verifier rejects each false proof at the check that this list names.

Limits, of the files of `product` and of `toy_check.py`:

- The second verifier,
  `tests/new/second/toy_check.py`, is not independent of the C++ programs on the first query, the
  chain, the order of the blocks, the fields of a plan's entry, of a stage and of a code, and where
  tail and message are in a code word: on these it can share a misreading with them. For the false
  proofs and for the comparison of the two verifiers' checks it is also not independent on the
  malicious provers' switches and on the shapes of the masks that are sent in the stages' trees. Nor
  is it independent in what its channel has for the other second verifiers, or in its record of one
  value at each vertex of a tree (R7.14). Its head lists the files. Of every other file of the C++ programs, and of the other parts of the files
  that it names only in part, the head says that nobody has examined whether the second verifier is
  independent of them: nothing is known either way.
- No file exercises D3b (section 5).
- No file exercises a non-canonical element in a message. (An opened symbol that is not an element's
  canonical form is altered copy 11.)
- Field 1 has no second verifier: all second verifiers compute in field 2 alone.
- The false proof `ENGINE_FAR_WORD` is made at the full size only. It is caught by the sampled
  positions, half of which it spoils; at the tiny size (3, 3, 5 positions) it would be accepted with
  a chance of 2^-3 to 2^-5. (from the programs' source: the head of the list of this run's false
  proofs)
- The parameters' bytes have not been derived from the public file a second time: the second
  verifier takes them as given, and encodes them again from their own content by the format page's
  rules.

---

# Part 1: SEND (sending oracle messages and their out-of-domain samples) and BAT (batching rows)

This part gives what the verifier does in the blocks SEND and BAT, as mathematics and bytes. It
follows the programs' source: `blocks/iop/send.h`, `bat.h`, `relation.h` and `mask.h` in the same
directory, and `blocks/toy/product.h`.

## 0. Notation

F is the base field: p = 2^64 - 2^32 + 1 (field 2) or p = 2^61 - 1 (field 1). The programs of this
repository can be built for field 2 alone (`blocks/base.h`; from the programs' source). E = F[X] /
(X^3 - c) is its cubic extension with the basis 1, X, X^2, where c = 2 for p = 2^64 - 2^32 + 1 and c
= 5 for p = 2^61 - 1. An element x of F is read in E as (x, 0, 0). As bytes an element of E is 24
bytes: its coordinates at 1, X, X^2 in this order, each a word of 8 bytes, low byte first, below p.
That is **canonical form**. Everything else about bytes (a round's query, leaves, openings, nonces)
is on the format page.

Words used below. The **plan** is the public list of a chain's rounds, each with its numbers of
items, of sampled values and of proof-of-work bits and with its error; both sides compute it from
the public files. The **channel** is the verifier's (or the prover's) end of the hash chain: it
reads a prover's message from the proof and derives the verifier's sampled values. **Ends the
process** marks a fault of the program or of its configuration, never a property of a proof's bytes:
on hostile bytes a verifier rejects. A **place** is an index within a row's message; a **position**
is an index within a code word, at which a test opens it. A **code switch** is the step of a
proximity test that goes over to a shorter code (part 4; CS in the sources).

For a point z = (z_0, z_1, ..) and an index j with bits j_0 (the lowest), j_1, ..:

    eq(j; z_0 .. z_(a-1)) := the product over t < a of ( j_t z_t + (1 - j_t)(1 - z_t) ).

An **oracle message** has **main rows** (each a message of length l under the base code; over F or
over E) and **mask rows** (each a message of length l_zk over E under the mask code); either number
can be 0. What is sent is each row's code word; the **underlying message** of an oracle message is
the tuple of the rows' messages. l and l_zk are powers of two.

A **table** collects the rows of one kind. With R real rows it has 2^b rows, b = ceil(log2 R); rows
beyond the real ones are 0 and nobody sends them. It is read as one vector: the entry at (row r,
place c) has the index r + 2^b c, so the row's bits are low. (`09-key-and-leaf-files.md` numbers the
entries of a message's mask rows in the same way, in other words: what is a place here is an entry
there, and its q is what b is here: "Entry c of row r has the number r + 2^q c".) A_m = b_m + log2 l
is the number of index bits of the main table, A_z = b_z + log2 l_zk that of the mask table.

A **row** (of the statement; not to be confused with a table's row) is a linear equation with a
public target on the underlying messages of oracle messages: the sum over the entries of
(coefficient x entry) = target, all in E.

## 1. What SEND is for

The programs' source gives this reason. Beyond half the minimum distance a committed word can be
close to several code words, and a prover could speak of one of them in one step and of another
elsewhere. SEND makes the prover answer, right after it has sent an oracle message, with the value
of a random linear form on the whole underlying message. For all but a small fraction of the seeds,
two different messages near one word give different values. The equation "form(message) = answer" is
kept as a row, the **sample row**: the row of an out-of-domain sample. Every proximity test that
reads the oracle message takes it in (section 4). So all later claims speak of one message.

## 2. SEND: the rules

**2.1** *The prover sends,* in one prover's message, one or more oracle messages. In a proof these
are the roots of their trees (which oracle messages share a tree, and a leaf's bytes, are on the
format page). Each oracle message's public shape says how many answers s it owes. An oracle message
with s > 0 gets an out-of-domain sample. An oracle message with s = 0 is sent along and gets none (a
fold's fresh mask, the last step's masking word): it gets no answer, no sample row and no ledger
(2.5), and does not count in 2.2. (The sources' word "rides" is wider: it is also used of messages
with s > 0 that go out in another block's prover's message.) Every oracle message with s > 0 has the
same s, the setting's number `samples`: the routines below would serve differing numbers, but the
plan has one for all, and the channel ends the process on any other count.

**2.2** *The verifier samples the seed,* in a verifier's message that holds nothing else (round
`SEND seed`). For an oracle message with both tables let a_o = max(A_m, A_z) + 1; with one table
only, a_o is that table's number of index bits. Let a be the largest a_o among the oracle messages
with s > 0 of this prover's message, and s the setting's number of answers. Sampled are, s times
over: z_0, .., z_(a-1), any elements of E, and then beta, an element of E that is not 0. All oracle
messages with s > 0 share these sampled values; each uses the first a_o coordinates.

**2.3** *The prover answers,* as the first thing of its next message: by oracle message in the order
of sending, and for each by the number i < s of the answer, one element y of E.

*Who closes the round of the answers.* Either SEND itself closes it, by a round in which the
verifier says nothing (round `SEND answer`). Or, where the block that calls SEND has a verifier's
message of its own to follow, that message closes it. (A code switch closes it with its sampling of
positions: round `CS`, or `LOC` in the form that only samples positions.) Which of the two it is, is
fixed by the calling block and is in the plan. Nothing is sampled between seed and answers in either
case.

**2.4** *What the verifier keeps: one sample row for each answer.* With (z, beta) the i-th seed, m
the main table and w the mask table as vectors:

    the sum over j of u[j] m[j]  +  the sum over j of v[j] w[j]  =  y,      where, at every index that is not a pad's place,
    u[j] = (1 - z_(a_o - 1)) * the product over t = A_m .. a_o - 2 of (1 - z_t) * eq(j; z_0 .. z_(A_m - 1)),
    v[j] =      z_(a_o - 1)  * the product over t = A_z .. a_o - 2 of (1 - z_t) * eq(j; z_0 .. z_(A_z - 1)).

That is: all entries are on one cube of a_o bits; the top bit is 0 for the main table and 1 for the
mask table; each table is in the low bits, and the bits between are 0. (The product is empty for the
table with more bits. With one table only there is no factor in front: the weight is eq(j; z_0 ..
z_(a_o - 1)).) An index on a row that nobody sends has its weight formally and meets the entry 0.

*The pad.* One element of E in the message is set apart for each answer. It has no weight under z;
in the row of its own answer its coefficient is beta, in the rows of the other answers 0.

*Where the pad is.* No word of a proof says it; it follows from the shape.

- If the oracle message has mask rows: in the last real mask row (number R - 1), the pad of answer i
  at the place l_zk - s + i: index (R - 1) + 2^(b_z) (l_zk - s + i); coefficient beta.
- Without mask rows, main rows over E: the same in the last real main row, with l for l_zk.
- Without mask rows, main rows over F: three entries for each answer, at the places l - 3 s + 3 i +
  t, t = 0, 1, 2, of the last real main row, with the coefficients beta * X^t: together they are one
  element of E read through the basis.

The routine that makes the row is `Send::form`; the places, `Send::pad`; the cube, `Send::cube`.

**2.5** *What else the verifier records.*

- (i) The oracle message's out-of-domain sample is answered (the sources say that the message is
  "sampled"): only from here on can a row, a group or a mask name it. (Two things look like
  exceptions and belong to other parts. A later chain can take over such an oracle message with its
  sample rows and pad from an earlier chain's statement, without a ledger. And a stored table's
  record is an out-of-domain sample of the same form, with z and beta from a chain of its own, which
  does not go through SEND: part 9.)
- (ii) The pad's places: no later row may have a coefficient other than 0 there.
  (`Statement::add_row` ends the process: that would be a fault of the program, because both sides
  make rows from public values.)
- (iii) The **ledger** of free mask entries: the indices of the mask table from 0 upward, below the
  first pad's index, leaving out those on rows that nobody sends: R (l_zk - s) + (R - 1) of them
  (`Send::free_mask_entries`). Blocks that need a mask entry (the zero-knowledge sumcheck, SPLIT,
  ..) take the next ones in this order, each once. The ledger closes when the first proximity test
  on a group that holds the oracle message begins. The ledger of a code switch's oracle message is
  closed at once: its mask rows hold data, so it never gives an entry.

*What the verifier checks on the spot:* nothing of SEND's own; all is deferred into the sample rows.
The channel rejects a proof in which an answer is missing or is not in canonical form. Any round in
which the verifier samples can carry proof of work, if the public files give proof-of-work bits to
its kind of error: the proof then holds a nonce before the sampling, and the channel tests it (R5.5,
R8.2). The public files of the toy runs give proof-of-work bits only to rounds that sample
positions, so `SEND seed` and `BAT` have none there.

**2.6** *The prover's side, which no verifier sees.* Four kinds of randomness are in or around an
oracle message, and they are not to be confused. The programs' source gives each its purpose
(comments in `send.h`, `blocks/code/code.h` and `blocks/frame/hiding_conditions.h`):

- The pad of an answer (above). By the source, if the pad is uniform and enters nothing else that is
  shown, then the answer is uniform, whatever the message. By the same comment the pad does enter
  more, as an entry of its row's message: the row's symbols at opened positions, the prover's
  messages in each proximity test on the row, and the new oracle message of a code switch that sends
  the row again; the random tails, that test's masks and the new message's pad must cover these.
- The free mask entries. Each masks one scalar that some block shows.
- The random tail of each row: r further entries under the code's randomized encoding. In field 2,
  the field of every file of these pages, they have the coefficients 0 .. r - 1 and come before the
  message (R12.11); in field 1 they come after it (part 4, section 0). By the source, with a uniform
  tail a row's symbols at up to r distinct positions are uniform and independent of the message (r -
  1 positions for an even r in field 1: `hidden_positions` in `blocks/numbers/codes.h`). At one
  position more that ends. The positions themselves are public.
- The salt of each leaf of a tree. By the source, the leaves' salts are to guard the columns that
  are not opened.

SEND itself samples every pad. The routine through which every oracle message with s > 0 goes out
(`send_prove`) ends the process unless it finds the pad's places empty; anyone who makes a table
leaves them empty. Where the pad is in a mask row, the routine writes fresh randomness into the
pad's places. Where the pad is in a main row, it gives the fresh values to the engine, the part of a
prover that keeps and encodes the main tables (`blocks/engine/engine.h`). A pad that nobody samples
is 0; the answer then shows a linear form of the message in the clear, and no verifier can notice.
The same holds of a pad that is sampled and does not reach its table. A test, `send_pad_test`, reads
the pads in the prover's memory.

A pad left unsampled is such a fault: in the range check's prover (`iop/rng.h`), for a message without
mask rows, its answer would show a linear form of the message in the clear, and every verifier would
accept. `send_pad_test` and `blocks/tests/hiding/rng_exact_test.cpp`, which runs the range check in
that shape, are there to show that `send_prove` samples every pad. No run of this repository shows
that either test fails on a sender that samples no pad. By its head, `send_pad_test` does not test
how the pads are distributed.

Against a pad that is sampled and does not reach its table, `send_prove` reads every pad back, in a
mask row or in a main row. It does so after the messages are encoded and committed and before
anything has left the process. It reads from the tables as the prover keeps them, at a place that it
computes again from the shape, and compares what it finds with a copy of the value that it kept when
it sampled it and, where the generator numbers its sampled values, with that value computed a second
time from its number. If one differs, it ends the process (`own-test-pad-kept`). This is in every
build. By the source it does not see six things: a pad that is overwritten after this comparison and
before the answer is taken (the answer is then unmasked, but by the source the proof does not
verify); under a generator that does not number its sampled values, a value that is lost right after
it is sampled, which is then the same in the kept copy and in the table; a fault in the formula for
a pad's place, which serves both to put the pad there and to find it; a value that is sampled once
and used again; the copies that the engine has encoded by then; and entries that
are written later, or that the GPU engine writes again when it moves a table. (from the programs'
source: `send_prove` in `send.h`)

**2.7** *The error that the plans count* for the round of the seed: the sum over the oracle messages
with s > 0 of (L^2 / 2) e^s, with L a bound on the number of code words within the later test's
distance of any word, and e = max(a_o / |E|, 1 / (|E| - 1)). The round of the answers: 0.
(`plan_send`.) In a request's own rounds every error is further multiplied by the setting's number
of instances, if that is above 1. L is the list bound that the caller gives; the toy runs give that
of the group in which the message will be tested.

*Choices of the programs.* The form of the out-of-domain sample (2.4: multilinear weights on one
cube, and a pad under a second scalar beta that is not 0), the bound e with the short argument given
for it (section 6), the pad's place, and that the pad can be in a main row are set out here as the
programs have them. The programs' source names as the origin of the factor L^2 / 2 Lemma 9.1 of
Chiesa, Fenzi, Weissenberg (ePrint 2026/391, version of February 25, 2026), and calls the form a
variant of Lemma 9.3 there. Where the calling block closes the round of the answers, the plan has
one round fewer than a count with an empty round. SEND has no variant for packed cells that adds a
range predicate to the claim: the range check is a block of its own.

## 3. SEND in the toy run `product`

The toy run is designed to prove a_i b_i = c_i for three vectors of length l = 2^n over F, which it
is meant to hide. Its public file fixes n (`log-length`), the codes and the number s of answers
(`samples`; 1 in the files of this repository). The files that go with this page have two sizes: n =
12 and n = 4 (the repository's tests also run n = 10).

The first prover's message holds O1 = "a, b, c" and with it, without an out-of-domain sample, the
fresh mask of the proximity test's first fold. O1 has three main rows over F, so b_m = 2 and the
main table has 2^(n+2) index values, m[r + 4 p] = (a, b, c)_r[p] for r < 3 and m[3 + 4 p] = 0; and
one mask row over E of length l_zk, b_z = 0. Both oracle messages are in one tree: one root.

- l_zk = the least power of two that is at least max((1 + 2 n + 2 + 2) + s, 5): the bracket is what
  the sumcheck takes of the ledger; the 5 is 2 k + 1 - h with k = 2, h = 0 (section 5), the room
  that the first fold's mask needs. n = 12: 32. n = 4: 16.
- A_m = n + 2, A_z = log2 l_zk, a = A_m + 1. n = 12: 15 coordinates and beta, 16 sampled elements; n
  = 4: 7 and beta, 8 sampled elements.

The sample row (s = 1):

    u[j] = (1 - z_(a-1)) eq(j; z_0 .. z_(A_m - 1));
    v[i] = z_(a-1) * the product over t = A_z .. a - 2 of (1 - z_t) * eq(i; z_0 .. z_(A_z - 1)) for i < l_zk - 1;
    v[l_zk - 1] = beta.

The ledger (s = 1): the indices 0, 1, .., l_zk - 2 (in general up to l_zk - s - 1). The sumcheck
takes the first 1 + 2 n + 4 of them in order.

SEND closes the round of the answers itself here. The later oracle messages of the proximity test
(part 4) have 2^k' main rows over E and 2^h' mask rows, both powers of two, so their pad is the mask
table's last index; their answers' round is closed by the code switch's sampling of positions.

## 4. BAT: the rules

**4.1** A **group** is a set of whole oracle messages under one base code and one mask code that one
proximity test reads together. Its two tables hold its members' rows (`Statement::layout`). A member
with K real rows of a kind takes an aligned block of 2^ceil(log2 K) rows. The blocks are placed from
row 0 by decreasing size, and blocks of one size in the order in which their members were sent. The
table has 2^k main rows (2^h mask rows), and the rows after the last block are 0. h is the least
number that holds the mask blocks, k the least that holds the main blocks with k >= max(h, 1). A
coefficient at (row r, place p) of a member goes to (first row of the member's block + r, place p)
of the group's table (`add_to_group_row`). An index of a group's table is row + 2^k * place (2^h for
the mask table). A member can be **kept**: its main rows are read at the group's positions and
tested with the group, but are not folded with the others, because a later part of the protocol
reads them again. They are not among the 2^k rows (parts 4 and 5).

**4.2** BAT is the first step of a proximity test on a group. The caller does not choose the rows
that are batched. `Statement::begin_run` closes the group (no row, member or mask entry can be added
any more) and passes on three kinds of rows, in this fixed order:

- first, all open rows on the group but those of the third kind, in the order in which they were
  added to the statement (`Statement::add_row`);
- second, the sample rows of all members, the members in the order of sending, each member's by the
  number of the answer;
- third, the rows that a code switch has made from sampled positions (**generator rows**), in the
  order in which they were added. There is one set for each of the draws, so a position that is
  sampled twice gives its rows twice (part 4).

In the toy run `product` the first stage has none of the third kind. In a chain of segments it has:
there a code switch leaves such rows open on the group that the next round `BAT` batches (part 5).

**4.3** The prover sends nothing. The verifier samples one element x of E, any value (round `BAT`).
The number J of rows is compared with the number from which the round's planned error was computed
(a difference is a fault of the program, not of a proof).

**4.4** *What the verifier keeps: one row on the group.* With row number i = (c_i on the main table,
d_i on the mask table, target t_i):

    c = the sum of x^i c_i,  d = the sum of x^i d_i,  target = the sum of x^i t_i,  i from 0, so the first row has the factor 1.

(`Bat::combine`.) The row says <c, main table> + <d, mask table> = target, and is what the first
fold starts from (part 3). Where the group has kept members the row has a third piece: for each kept
member that a row reads, in the order of sending, a form on that member's own main table, batched by
the same powers. The equation is then <c, main table> + the sum over the kept members of <their
form, their main table> + <d, mask table> = target, and anyone who uses the row takes in all three.
BAT checks nothing on the spot.

**4.5** *The error that the plans count:* L (J - 1) / |E|, L as in 2.7 for the group's word, with
the same factor for instances. (`plan_bat`.)

**4.6** *A second form, BAT-sigma:* J >= 1 rows on any oracle messages whose out-of-domain samples
are answered, none of them a sample row, into one such row, by the same powers of one sampled
element (round `BAT-sigma`), well before any test, for example before a row on two groups is split
(SPLIT: part 9). No list factor: (J - 1) / |E|. (`Bat::sigma`; `plan_bat_sigma`.)

**4.7** *Inside a proximity test,* at the later stages, the batching element comes in one verifier's
message with the positions, after them. In the programs all rows of the new group are added to the
statement after it is sampled. Of these only the generator rows depend on that same verifier's
message, and they are fixed by the roots and the positions. (Part 4.)

*Choices of the programs.* The rows are combined by the powers of one element. The order of 4.2 is
the programs' choice. 4.7 is an exception to the rule "every row is fixed before the batching
element is sampled".

## 5. BAT in the toy run `product`

The group has one member, O1; its tables are O1's tables, index for index (k = 2 row bits, h = 0).

There are J = 3 + s rows, in this order:

- R_0, `hiding polynomial and c`, which part 2 calls (U): the row that the zero-knowledge sumcheck
  leaves. It holds mask entries, and row c of m at gamma, the point in E^n at which the sumcheck
  ends. The sumcheck does not show the value of c at gamma: it stays as a term of this row (part 2);
- R_1, `a at the point`, and R_2, `b at the point`, which part 2 calls (E_1) and (E_2): the
  sumcheck's two end claims, each with its mask;
- R_3 ..: the sample row or rows.

On the main table: c_1[r + 4 p] = [r = 0] eq(p; gamma), c_2[r + 4 p] = [r = 1] eq(p; gamma), c_0[r +
4 p] = [r = 2] * (the factor on c in row (U): part 2) * eq(p; gamma), c_3 = u. The mask parts and
targets of R_0 to R_2 are in part 2; d_3 = v, t_3 = y.

BAT runs once in this toy run, at the first stage: target = t_0 + x t_1 + x^2 t_2 + x^3 y.

## 6. Limits of this part

What this part does not cover:

- More than one answer (s > 1): no second verifier has read a file with s > 1: `zkpy.py`,
  `links_check.py` and `proof_check.py` stop on any number of answers but 1.
- With several answers and several mask rows a few entries beside the pads are in no ledger; they
  are uniform and carry their weight under z like all others. No test looks at it.
- 2.6 is on the prover's side, which no verifier can see, so no second verifier checks it.
- The errors of 2.7 and 4.5 are what the planning routines compute, not something that a run shows.
  The same holds wherever these pages say that a verifier rejects parameters whose round errors miss
  their target: the errors are the planner's figures, and no chain's first query holds a round's
  error. What is checked at a run is a round's name, kind, counts and proof-of-work bits, and what
  the plan says beside a round of the numbers from which its error was counted. For some rounds such
  a number is taken on the calling routine's word, and nothing at the run checks it: the groups
  whose list sizes count at an out-of-domain sample or at a batching; the numbers a_o of the smaller
  oracle messages under one seed; a lookup's table and keys; the next stage's list size in the
  batching term of a code switch; a stored table's distances at every round that is counted from
  them. (from the programs' source: the type `Priced` in `blocks/numbers/plan.h`; the comment before
  `run_verifier` in `blocks/chain/closed.h`)
- That two different messages near one word differ under the form for all but a fraction e of the
  seeds is argued in a few lines at the head of `send.h`.
- Which oracle messages share a tree, a leaf's bytes and label, the bytes of a round's query, and
  how an element that is not 0 is sampled are on the format page, not in this part.
- The list bound L and the numbers of positions belong to the part of the programs that plans
  (`numbers/`), not to this part.

---

# Part 2: ZERO and PSC in the toy runs: the verifier, written out

This part gives the verifier's side of two blocks of the library. ZERO turns the claim that a_i b_i
= c_i for all i into a claim on one sum. PSC is the zero-knowledge sumcheck for that sum, with its
hiding polynomial. The part speaks of the toy runs `product` and `links`. The words plan, channel,
oracle message, main row, mask row, table, place, row, sample row, group, pad and ledger are used as
in part 1.

## 0. Notation

- F is the base field of the build and E its cubic extension (R1.2, R1.4). Nothing below depends on
  which of the two fields it is. All scalars below are elements of E (24 bytes each in a proof).
- K is the number of variables, which is log2 of the vectors' length. In `product` K is the public
  file's number `log-length` (12 at the full size, 4 at the tiny size). Part 1 writes n for it.
- a, b, c in F^(2^K) are main rows 0, 1, 2 of the run's one oracle message with main rows, which is
  called O below (part 1: O1). The index i is in [0, 2^K); bit_j(i) is bit j of i, and bit 0 is the
  least significant. Variable j belongs to bit j.
- For a point x = (x_0 .. x_(K-1)) in E^K: eq(x, i) = the product over j of ( x_j if bit_j(i) = 1,
  else 1 - x_j ). (Part 1 writes eq(i; x).) For a vector u of length 2^K: u~(x) = sum_i eq(x, i)
  u_i.
- O's main table has 2^(K+2) entries: the entry at (row r, place i) is at index r + 4 i. The two row
  bits are low, because three rows round up to four. No row of this part has a coefficient on row 3.
- O's mask row is one row over E; m_t is entry number t of its message, t = 0, 1, 2, .. . The
  entries are given out in increasing order from t = 0, each once (routines `Send::admit`,
  `MaskLedger::take`). The pad of O's out-of-domain sample is the last s entries of the mask row's
  message, where s is the number of answers to the out-of-domain sample (part 1, 2.1). The ledger
  gives out only the entries below the pad (routines `Send::pad`, `Send::free_mask_entries`), so no
  entry of this part belongs to a pad. PSC is the only block that takes entries in `product`, so it
  gets m_0 .. m_(2K+4).
- The claim to be shown: a_i b_i = c_i for all i.

## 1. ZERO

1. The prover sends nothing: the round's prover's message is empty.
2. The verifier samples one message of K elements of E, zeta_0, .., zeta_(K-1), in this order. The
   round's name is `ZERO`; it is an algebraic round (kind 1 of R5.2), and 0 is allowed. Routines
   `toy_verify_preamble`, `plan_zero`.
3. The verifier keeps zeta. The claim becomes sum_i eq(zeta, i) (a_i b_i - c_i) = 0: the target is T
   = 0, and there is no mask form (routine `toy_claim`).
4. It checks nothing.

The plan counts the error of this step as K / |E|.

## 2. PSC

The sumcheck has K + 1 variables. X_0 .. X_(K-1) are each summed as g -> (1 - zeta_j) g(0) + zeta_j
g(1): the equality factor is part of the way of summing, not of the polynomial. One further
variable, Y, is summed as g -> g(0). Every round polynomial has 3 coefficients (degree at most 2),
lowest first. In every round the constant coefficient is the one that is not sent. (Routines
`psc_vars`, `Weight::equality`, `toy_summand`.)

What the honest prover runs the sumcheck on (for the reader; the verifier never sees it):

    H(X, Y) = U(X, Y) + eps * [ (a~(X) + c_1 Y)(b~(X) + c_2 Y) - c~(X) ],
    U(X, Y) = sum over l = 0 .. K-1 of ( m_(2l) X_l + m_(2l+1) X_l^2 )  +  m_(2K) + m_(2K+1) Y + m_(2K+2) Y^2,
    c_1 = m_(2K+3),  c_2 = m_(2K+4).

### Messages, in order (2K + 5 scalars of E sent, 2K + 5 mask entries used; no root)

| # | the prover sends | the verifier's message | the round's name |
|---|---|---|---|
| 1 | mu' (1 scalar) | eps, one element of E (0 allowed) | `PSC epsilon` |
| 2 + j, j = 0 .. K-1 | q_(j,1), q_(j,2) (2 scalars) | gamma_j, one element of E (0 allowed) | `PSC round` |
| K + 2 | p_1, p_2 (2 scalars) | beta, one element of E, never 0 | `PSC Y` |
| K + 3 | v_1, v_2 (2 scalars) | nothing | `PSC values` |

Routine `psc_verify`. The last round makes a query of the chain like every other round; nothing is
sampled from it.

Why the order is as it is. mu' is the sum of U under weights that hold zeta, so it can only be sent
after ZERO's message. And eps must come after mu'. The programs' source (`psc.h`) gives as the
reason: a prover that knows eps first fits mu' to any false target. So (mu'; eps) is a round of its
own, and eps cannot be part of ZERO's message.

Why beta is never 0. v_1 = a~(gamma) + beta c_1, so at beta = 0 the two end values would be shown
without a mask. The block's test of hiding (`psc_hiding_test`) has that case as a control, and in it
the test tells two witnesses apart.

### What the verifier computes and keeps

    R_0 := mu' + eps * T = mu'                                   (T = 0)
    For j = 0 .. K-1:
       q_(j,0) := R_j - zeta_j * ( q_(j,1) + q_(j,2) )
       R_(j+1) := q_(j,0) + q_(j,1) gamma_j + q_(j,2) gamma_j^2
    Round in Y:
       p_0 := R_K
       R_(K+1) := p_0 + p_1 beta + p_2 beta^2

The R_j are the sumcheck's running claims, not the rows R_0, R_1, .. of part 1, section 5. The
routine that computes q_(j,0) and p_0 is `Weight::complete`. For X_j the weights of the coefficients
of 1, X, X^2 are 1, zeta_j, zeta_j; for Y they are 1, 0, 0.

The verifier keeps gamma = (gamma_0 .. gamma_(K-1)), beta, eps, R_(K+1), v_1 and v_2.

### What it checks on the spot

Nothing algebraic. A sumcheck has two kinds of check: the first polynomial sums to the claim, and
each next one sums to the value of the one before it. Both are used up to compute the coefficients
that are not sent, so they hold by construction. The routine rejects only if the channel rejects (an
item cannot be read, or the close of a round has rejected). Everything else is deferred into the
three rows below.

That the coefficient which a check fixes is not sent is a choice of the programs.

### The three rows that it passes on

All three go into the one group that holds O (routines `toy_rows`, `psc_rows`, `hp_form`). A row is
a coefficient vector on O's main table, coefficients on O's mask entries, and a target in E. It
asserts that the sum over the main entries plus the sum over the mask entries equals the target.

Row (U), named `hiding polynomial and c`:

- main table: the coefficient - eps * eq(gamma, i) at index 2 + 4 i (row c), for i in [0, 2^K); 0
  elsewhere;
- mask entries: gamma_l on m_(2l) and gamma_l^2 on m_(2l+1), for l = 0 .. K-1; 1 on m_(2K); beta on
  m_(2K+1); beta^2 on m_(2K+2);
- target: R_(K+1) - eps * v_1 * v_2.

In one line:

    sum_l ( gamma_l m_(2l) + gamma_l^2 m_(2l+1) ) + m_(2K) + beta m_(2K+1) + beta^2 m_(2K+2) - eps c~(gamma)  =  R_(K+1) - eps v_1 v_2.

Row (E_1), named `a at the point`: main table: eq(gamma, i) at index 0 + 4 i (row a); mask entries:
beta on m_(2K+3); target v_1.

    a~(gamma) + beta m_(2K+3) = v_1.

Row (E_2), named `b at the point`: main table: eq(gamma, i) at index 1 + 4 i (row b); mask entries:
beta on m_(2K+4); target v_2.

    b~(gamma) + beta m_(2K+4) = v_2.

They are added to the group in this order: (U), (E_1), (E_2).

In the source's letters the summand is S = Y_1 Y_2 - U_1. (These are not the variable Y and the
polynomial U above: Y_1 and Y_2 are the values of a and b, which are shown under masks, and U_1 is
the value of c, which is not shown.) At the end values S is the affine form A_0 + A_1 * (c's value)
with A_0 = v_1 v_2 and A_1 = -1. A_0 goes into the target, and A_1 into the coefficient on c.

Why an honest run satisfies (U): R_(K+1) = H(gamma, beta) = U(gamma, beta) + eps ( v_1 v_2 -
c~(gamma) ).

### Which mask entry is used where (each once)

m_(2l), m_(2l+1), l = 0 .. K-1: the coefficients of X_l, X_l^2 of U. m_(2K), m_(2K+1), m_(2K+2):
those of 1, Y, Y^2 (only the last variable has a constant). m_(2K+3), m_(2K+4): the inner masks of
a's and b's end values. (Routines `hp_place`, `psc_masks`: first the hiding polynomial's
coefficients, by variable and then by increasing degree, then the inner masks.)

### Errors as the plan counts them (routine `plan_psc`)

eps: 1/|E|; each X_j: 2/|E|; Y: 2/(|E| - 1); the round of the values: none.

### Two more choices of the programs, and what the block leaves to its caller

- The honest prover stops on a false claim. The prover's routine (`psc_prove`) ends the process if a
  round polynomial does not sum to the running claim. So an honest prover of this block cannot go on
  after a failure in an earlier step. In the toy runs no honest run reaches that state.
- What the block itself does not enforce. The block enforces this much: an entry is given out once,
  only from an oracle message whose out-of-domain sample has been answered, and not after a
  proximity test on that message has begun; and mu' is the last thing sent before eps, in a round of
  its own. It does not enforce in which group's mask rows each entry is, or that eps comes after the
  last message that enters the weights, the summand's public parameters or the target. These two are
  the calling routine's duty. In the toy runs the calling routines are `toy_verify_preamble` and
  `segments_decide`: there SEND comes before ZERO and ZERO before PSC, and all entries come from the
  one mask row.

## 3. The same block in `links`

Each segment l = 1 .. L runs ZERO and PSC exactly as above, on x_l, b_l, c_l instead of a, b, c
(routine `segments_decide`), with K = `log-length` - 1. A vector fills the lower half of a row, so a
row's coefficient vector is eq(gamma, i) on the places i < 2^K and 0 on the upper half (routine
`segments_vector_at`).

Where the three vectors are (routine `segments_places`): in the first segment x and b are main rows
0 and 1 of the segment's own first oracle message. In a later segment x is row 0 of the kept oracle
message as the frame (part 5) has sent it again, and b is row 0 of the segment's own first message.
In every segment c is row 0 of the segment's own second message.

The 2K + 5 mask entries are m_0 .. m_(2K+4) of the mask row of the segment's own first message. The
three rows, named `hiding polynomial and c`, `x at the point`, `b at the point`, go into the
segment's group in that order, before the frame closes the segment. All else about a segment is in
part 5.

## 4. What the decision at the end owes to this part

This part decides nothing by itself. Its rows (U), (E_1), (E_2) go, together with SEND's sample rows
(part 1, 2.4), to the later blocks (parts 1, 3, 4, 5), which are there to test that they hold for
the words that the roots commit to. In `product` 3 + s rows are batched, with s the number of
answers to the out-of-domain sample.

## 5. Tests of this part, and what they show

- The second verifier `tests/new/second/toy_check.py` calls no code of the programs. It agrees with the C++ verifier on every query, every sampled value and every recorded
  value of `product`, at the full and at the tiny size. `tests/new/second/run_honest.sh` makes the
  honest files and runs this comparison on them.
- The block's own tests in the repository (`blocks/iop/tests`: `hp_test`, `psc_test`,
  `psc_chain_test`, `psc_hiding_test`, `psc_stops_test`) run the same routines on
  other shapes (small cases over small fields, a product tree), not on the shapes of the toy runs.
- `psc_hiding_test` counts, over small fields and all values of the mask entries, that the sent
  scalars are uniform for each of the shapes and witnesses that it takes, and that each of its
  spoiled maskings is noticed. It says nothing about a whole proof. By its own head it shows nothing
  about the large field as such, about shapes that it does not list or about witnesses that it does
  not take, and, where it runs two sumchecks one after the other, nothing about lists of the
  verifier's sampled values that it did not run; nothing about what the rest of the protocol shows
  of the mask entries; and nothing about running time or memory. It checks the program against the
  hiding property and does not prove the property. (from the programs' source: the head of
  `psc_hiding_test.cpp`)

## 6. Limits of this part

- This part does not define how an element of E is read from a round's answer (canonical form,
  rejections, "never 0" for beta), or when the channel rejects an item. The format page does (R6.4,
  R6.5; R1.3, R1.4, R8.1), with the part of the programs that runs the chain. This part uses only
  that `psc_verify` asks for a sampled value that is not 0 for beta, for a plain sampled value for
  each gamma_j, and for the epsilon round's sampled value for eps.
- This part does not define the byte order of a scalar in the file: R1.1, R1.3, R1.4.
- That the mask row's message has room for 2K + 5 entries and the pad is the matter of the sizing
  routines of the toy runs (`toy_size`, `segments_size`).

---

# Part 3. FOLD and BASE: the verifier as mathematics

This part gives the verifier's side of two blocks of a proximity test. FOLD turns one claim on the
rows of a group into one claim on single rows. BASE is the last step of the test: it checks such a
claim against the last oracle message. The words plan, channel, "ends the process", oracle message,
main row, mask row, table, place, position, group, member, kept and row are used as in part 1.

## 0. Conventions

- E is the field of the challenges; every scalar below is an element of E unless said otherwise. An
  element of E is sent as three canonical words of the base field F (R1.3, R1.4). Variables are
  counted from 1 here; the code counts from 0.
- A table of 2^a rows is one vector: the entry at (row b, place c) has the index b + 2^a c; bit 0 of
  the row's number belongs to variable 1.
- Folding a table of 2^a rows at (g_1, .., g_a): fold(T; g)[c] = sum over the rows b of eq(g, b)
  T[b + 2^a c], with eq(g, b) = prod_(t = 1..a) (g_t if bit t - 1 of b is 1, else 1 - g_t). It is
  done in steps: fixing the lowest variable at g puts (1 - g) (row 2b) + g (row 2b + 1) instead of
  the rows 2b and 2b + 1 (routines `fold_low_once`, `fold_low`). The same rule folds messages,
  encoding randomness, opened columns of symbols and coefficient vectors (on a structured
  coefficient vector: `Form::fix_low`).
- <x, y> = sum_i x_i y_i.
- Enc_C(M, R)(x) is the symbol at position x of the encoding, under the code C, of the message M (l
  entries) with the encoding randomness R (r entries) (routine `symbol_at`; the codes: section 4 at
  the head of this page, and R12.11). The encoding randomness is the random tail of R12.11.
- cp(x) = x modulo the mask code's length: the position of the mask code that goes with position x
  of the main code (routine `mask_position`). It is the short position of R7.2. The main code is the
  code of the main rows, which part 1 and the format page call the base code.
- The letter l has two senses. In 2.1 and in the formula for u in 2.5, l with 1 <= l <= k counts the
  variables of U. Everywhere else l is the number of entries of a main row's message, and l_zk that
  of a mask row's message.

## 1. What FOLD starts from: a row on a group (`GroupRow`)

A group has 2^k main rows m and 2^h mask rows mu, with h <= k and k >= 1. m is a table of 2^k rows
of l places: the messages of its members' main rows, each member's rows in an aligned block; rows
that nobody sent are 0. mu is a table of 2^h rows of l_zk places. A group can also have kept
members: their main rows are read and tested with the group but are not among the 2^k rows, because
a later segment reads them again. The claim, which BAT (part 1) makes from all open rows of the
group, is

    <c, m> + sum over the kept members K of <c_K, M_K> + <d, mu> = tau.

Here c is a coefficient vector on the 2^k l entries of m, given in structured form (a sum of product
forms and single entries); c_K is the like on K's own main table M_K; d is written out on the 2^h
l_zk entries of mu; tau is public.

In `product` the first fold has k = 2, h = 0 (three sent rows and a zero row; one mask row), the
second fold has k = h = 1, and there is no kept member. For `links`, part 5 says which members are
kept.

## 2. FOLD, message by message (`fold_start_verify`, `fold_epsilon`, `fold_rounds_verify`; together `fold_verify`)

### 2.1 The prover: the fresh mask, and one element

- The **fresh mask** is an oracle message with no main row and 2^h mask rows under the mask code. It
  gets no out-of-domain sample. It can have been sent with an earlier prover's message: then nothing
  is sent for it here. In the toy runs this is so: it is sent in the prover's message and the tree
  of its stage's or segment's rows. In both cases the verifier records which oracle message it is
  and marks it as used up: an oracle message serves as a mask once, or else the process ends.
- One element, sigma: the sum over {0,1}^k of the hiding polynomial U. It is sent here in every
  case, also where the mask was sent earlier.
- U has 2k + 1 coefficients: a constant w, and for each variable l a linear one, a_l, and a
  quadratic one, b_l: U(X_1..X_k) = w + sum_(l = 1..k) (a_l X_l + b_l X_l^2) prod_(j > l) X_j.
- How the 2^h rows hold the coefficients (`fold_mask_table`; the verifier never sees this, it needs
  only u of 2.5). Places 0 .. 2k - h of a row are used, and all further ones are 0. Say that row b
  reaches l (1 <= l <= h) if the bits l - 1, .., h - 1 of b are all 1. Then in row b: place 0 holds
  s + the sum of a_l over the l that b reaches, with s = a_(h+1) if h < k and s = w if h = k; place
  l (1 <= l <= h) holds b_l if b reaches l, else 0; the later places are the same in every row:
  b_(h+1), then for l = h + 2 .. k the two entries a_l, b_l, then w (none of these if h = k).
- Example: k = 3, h = 1 with (a_1, b_1, a_2, b_2, w, a_3, b_3) = (1, 2, 3, 4, 5, 6, 7) gives row 0 =
  (3, 0, 4, 6, 7, 5, 0, ..) and row 1 = (4, 2, 4, 6, 7, 5, 0, ..). This is a known answer of
  `fold_rounds_test`.

### 2.2 The verifier: eps

One element, in a verifier's message of its own, after the fresh mask's root and after sigma. It may
be 0.

### 2.3 k rounds, j = 1 .. k

- The verifier keeps a running claim: claim_0 = sigma + eps tau.
- The prover sends two elements (q_0, q_2): the constant and the quadratic coefficient of the
  round's polynomial q_j(X) = q_0 + q_1 X + q_2 X^2.
- The coefficient that is not sent follows from q_j(0) + q_j(1) = claim_(j-1): q_1 = claim_(j-1) - 2
  q_0 - q_2.
- The verifier samples g_j, one element (it may be 0). Then claim_j = q_0 + q_1 g_j + q_2 g_j^2.
- An honest q_j is the sum over the later variables in {0,1} of H(g_1..g_(j-1), X, ..), with H as
  follows, where c(X), m(X), .. are the tables folded at X and V_kept = sum_K <c_K, M_K>:

      H(X) = U(X) + eps ( <c(X), m(X)> + <d(X_1..X_h), mu(X_1..X_h)> X_(h+1) .. X_k + V_kept (1 - X_1) .. (1 - X_k) )

### 2.4 What the verifier checks in FOLD

Nothing on the spot. Everything goes into the claim that it passes on.

### 2.5 What it passes on: single rows (struct `SingleRows`; made by `fold_output`, one routine for both sides)

The single rows are virtual rows: none of them is sent as it is. They come in this order:

    M_0 = fold(m; g_1..g_k)                     (l entries)      the folded main row
    the kept members' main rows, as they were sent, by member and row
    M_1 = fold(mu; g_1..g_h)                    (l_zk entries)   the folded old mask row
    M_2 = fold(the fresh mask's rows; g_1..g_h) (l_zk entries)   the folded fresh mask row

With them goes one claim:

    <c', M_0> + sum_K <c'_K, M_K> + <d'_1, M_1> + <d'_2, M_2> = tau'   with
    c'   = eps fold(c; g_1..g_k):   c'[i] = eps sum_(b < 2^k) eq(g, b) c[b + 2^k i]
    c'_K = eps (1 - g_1) .. (1 - g_k) c_K                                        (fold_kept_factor)
    d'_1 = eps g_(h+1) .. g_k fold(d; g_1..g_h)                                  (fold_mask_form)
    d'_2 = u(g), the public vector with <u(g), M_2> = U(g)                       (fold_mask_weights)
    tau' = claim_k.

The vector u, with A_l = prod_(j > l) g_j: u[0] = A_h; u[l] = g_l A_h for 1 <= l <= h; then, in the
order of 2.1's later places: g_(h+1)^2 A_(h+1); for l = h + 2 .. k the two entries g_l A_l, g_l^2
A_l; then 1; 0 elsewhere. For h = k: u = (1, g_1, .., g_k, 0, ..).

Examples. k = 2, h = 0: u = (g_1 g_2, g_1^2 g_2, g_2, g_2^2, 1, 0, ..). k = h = 1: u = (1, g_1, 0,
..). k = 3, h = 1 at g = (2, 3, 5): u = (15, 30, 45, 5, 25, 1, 0, ..).

How a position of these rows is read (`single_rows_symbols`) by the block that tests them later (the
code switch of part 4, or BASE). Open, at position x, the columns of all members of the group and of
the fresh mask. Put the members' main symbols at x into a vector of 2^k entries, at the rows where
the layout puts them (0 for rows that nobody sent; a kept member's symbols are set aside), and their
mask symbols into a vector of 2^h entries. Then

    y_0(x) = fold(main column; g_1..g_k);   the kept rows' symbols as they are;
    y_1(x) = fold(mask column; g_1..g_h);   y_2(x) = fold(the fresh mask's column; g_1..g_h).

Encoding is linear and acts row by row. So for an honest prover y_0(x) = Enc_C(M_0, R_0)(x), with
R_0 the randomness folded in the same way, and y_j(x) = Enc_Z(M_j, R_j)(cp(x)) for the two mask
rows.

In the toy runs the single rows of every FOLD go into a code switch (part 4), not into BASE.

## 3. BASE (`base_verify`)

### 3.1 What it starts from

Single rows again, here plain rows of the last oracle message O_last (one main row over E, one mask
row; k = h = 0, no challenge): M_0 (l entries, code C) and M_1 (l_zk entries, mask code Z), with
y_0(x) = O_last's main symbol at x and y_1(x) = its mask symbol for cp(x). The claim <c, M_0> +
<d_1, M_1> = tau comes from the last stage's row on its group. A group has at least one row
variable, also a group of one main row, so that row's c has a second row, which is zero. The
variable is fixed at 0: c[i] := c_group[0 + 2 i] (`iopp_last_rows`, part 4). The routine takes one
main row and up to two mask rows, which is what `single_rows.h` can describe; single rows with kept
rows end the process.

### 3.2 The prover: the masking oracle, and one element

- g* is an oracle message of the same shape: a main row M*_0 under C and a mask row M*_1 under Z,
  with all entries and all encoding randomness uniform. It gets no out-of-domain sample. It is the
  masking word of the last step of R12.10. In the toy runs it is sent in the last stage's prover's
  message and tree, so nothing is sent here. The verifier records which oracle message it is and
  marks it as used up. On the prover's side: if M*_0 were lost before g* is committed, then oracle
  message, tau* and shown word would still agree, no verifier could notice, and lambda M_0 would be
  sent unmasked. Against that, `base_prove` compares M*_0 as the engine gives it back with the row
  that was sampled, entry by entry, and ends the process if they differ (`own-test-base-mask`). That
  comparison alone does not see a row that was lost right where it was sampled. So `base_prove` also
  compares M*_0 with the sampled values themselves, computed a second time by their number, and, in
  the same way, the mask rows M*_1 and their random entries R*_1, and ends the process under the
  same name if one differs. Under a generator that does not number its values this second comparison
  is not made. By the source it does not see R*_0, which the engine samples for itself; a row that
  was empty at its source (a generator whose values are not random gives the same values in both
  computations); a fault that strikes both computations; or a sampled value whose number was not
  noted or did not arrive, of which nothing is then asked, and nothing says so. Nor can it see a
  value that is lost after the comparison, or a loss in the copies of these rows that the engine has
  encoded by then. And it does not compare the prover's other values that are sampled on the host:
  the coefficients of FOLD's fresh mask, which by the same source FOLD compares itself
  (`own-test-fold-mask`, in `fold.h`: only if the generator numbers its sampled values), and the
  sampled tables and random entries of other oracle messages. (from the programs' source:
  `base_case.h`, its head and `base_prove`) In `cp_commit` and `cp_prove` something
  does look at R*_0: their GPU engine compares every main row's random entries with the generator's
  once more, right after the rows' hash tree is made, and ends the program if they differ
  (`own-test-coins-bound`), and R*_0 is among these entries. Under the CPU reference engine nothing
  looks at R*_0. By the same comment that comparison does not see a kernel that reads fewer entries
  than were sampled, two rows that point at one row's entries, or a fault in code that the comparing
  and the sampling kernels share. (from the programs' source: the comment at `own-test-coins-bound`
  in `prover/hiding/engine_coins.cuh`)
- One element, tau*. An honest prover sends

      tau* = <c, M*_0> + <d_1, M*_1>.

### 3.3 The verifier: lambda

One element. It may be 0.

### 3.4 The prover, in the clear: the masked messages and their randomness

(l + l_zk) + (r + r_zk) elements of E: first the messages M'_0, M'_1, each whole, then the
randomness R'_0, R'_1. An honest prover sends

    M'_j = M*_j + lambda M_j,   R'_j = R*_j + lambda R_j.

### 3.5 Check B1, on the spot

    <c, M'_0> + <d_1, M'_1> = tau* + lambda tau.

Only here is c written out (`single_rows_main_form`).

### 3.6 The verifier: the list of positions

The list of positions in [n] (R6.6), with the round's proof of work if the plan has one (R5.5,
R8.2).

### 3.7 The prover: the opening

The opening of O_last and g* at the list: one column for each distinct position. The verifier
recomputes the roots (`read_opened`). That check belongs to the trees (R7.8, R7.9), not to BASE.

### 3.8 Check B2

It is made once for each distinct listed position x, in ascending order, and for every row:

    Enc_C(M'_0, R'_0)(x)     = g*_0(x)     + lambda y_0(x)
    Enc_Z(M'_1, R'_1)(cp(x)) = g*_1(x)     + lambda y_1(x)

Here g*_0(x) is the opened main symbol of g* at x, and g*_1(x) is its opened mask symbol in the leaf
at x, which stands for position cp(x) of the mask code. How a leaf holds mask symbols is R7.1 and
R7.2.

### 3.9 When BASE accepts

BASE accepts if and only if B1, all B2 and the openings hold, every received element is canonical
(R1.3, R1.4), and the round's nonce passes its proof-of-work test (3.6). That the proof's bytes are
used up (R8.1) is asked at the chain's end, not by BASE (part 4, 8.5).

## 4. Orders that the verifier's program enforces by construction

The fresh mask's root and sigma come before eps; (q_0, q_2) of round j before g_j; the root of g*
and tau* before lambda; M' and R' of 3.4 before the positions. Each is one call after the other on
the one channel, and the channel ends the process if a round's name, its size or its number of
sampled values differs from the plan.

## 5. Choices of the programs

- A round sends two scalars: the linear coefficient is never sent (2.3). So a fold sends 2k + 1
  scalars.
- eps has a verifier's message of its own. It is not merged with the batching seed.
- Kept rows enter by the factor of 2.5 and the term of 2.3.
- BASE masks as section 3 says.
- There is no packed variant, in which masks would be sent in a switch's mask rows and be left out
  of its out-of-domain sample.

## 6. Limits of this part

- `toy_check.py` and `links_check.py` compare this text with the programs at the shapes of `product`
  and `links` only: k = 2, h = 0 and k = h = 1, and kept rows in `links`.
- The other second verifiers run FOLD and BASE too. `zkpy.py` (routine `iopp_`; `key_check.py`,
  `lin_check.py` and `proof_check.py` call it) and `entry_check.py` (routine `iopp`) call the FOLD
  routine of `toy_check.py` (`fold_block`) with the k and h of a file's stages, and each has BASE
  written out once more.
- That the rows of 2.1 and the vector u of 2.5 give back U(g) is tested for all h <= k <= 5 by a
  test of the programs themselves (`fold_rounds_test`), by its head over prime fields of its own and
  never over the library's field or its extension.
- This part does not say whether the mask code has the main code's length at every stage of a toy
  run (then cp is the identity). The public files of the toy runs say it. BASE and the code switch
  use cp whatever the lengths.
- This part does not define how BAT makes (c, c_K, d, tau) from the open rows, or how a code switch
  turns FOLD's single rows into rows on the next oracle message: parts 1 and 4. Which members are
  kept, and for how long, is in part 5.
- That the rows of 2.1 make U's coefficients uniform and independent of everything else that is
  shown is an argument on hiding. This part does not make that argument.
- The errors of this part's rounds, and what they rest on (from the programs' source: `plan_fold`,
  `plan_fold_rounds`, `plan_base`, `fold_round_error` and `base_error` in `blocks/numbers/blocks.h`;
  the head of `blocks/numbers/codes.h`). These errors are what the planning routines compute. No run
  shows them, and this page proves none of them.
  - The errors. For eps (round `FOLD epsilon`): list / |E|. For g_j (round `FOLD round`, j = 1 ..
    k): mca(main, 2^(k - j) sides) + 2 list / |E|, plus 2 mca(mask, 2^(h - j)) if the group has a
    mask code and j <= h. For lambda (round `BASE lambda`): mca(main, sides) + list / |E|, plus
    mca(mask, 1) if the group has a mask code. Each has the factor for instances of part 1, 2.7.
    Here list is the group's list size: a bound on how many code words can be within the group's
    distance of one word. sides is the number of rows for which one main row stands where rows are
    folded side by side (the side count of 8.4, item 5): 2 in the segments' group of `links` (a main
    row and one kept row), and 1 in every other group of `product` and `links`. mca(code, rows) =
    count / |E|, where the **closeness count** bounds how many challenges z can make u + z v close
    to the code where (u, v) are not jointly close to it. The format page gives the formulas for count and list (8.4, items 6 to 9); the number of rows has no factor in them.
  - What a closeness count rests on follows from the distance, and from whether the distance was made
    from the code's m (8.4, item 6). Where it was not, within a third of the code's minimum distance it is that of
    a lemma on mutual correlated agreement within a third of the minimum distance; the lemma's
    source is named in `THIRD_PARTY.md` at the repository's top, and `blocks/numbers/codes.h` names
    other papers as sources of the same count; of a part of these citations it says that they are
    second-hand and were not checked against the papers. Beyond a third of the minimum distance,
    within unique decoding as well as up to the Johnson radius, and at every distance where it was, the count is that
    of Theorem 4.6, taken with M = 1, of IACR ePrint report 2025/2055, the theorem on codes. The
    programs' accounting takes it as given. A statement on codes held to follow from that theorem, in its case of two words, is proved
    in this repository's Lean library, by way of another formalization's theorem and a comparison of bounds; the paper's
    own form of the bound is not proved there, and that the library's statement follows from the paper's,
    and that these programs' parameters meet the theorem's conditions, rest on reading, not on a proof (lean/LIMITS.md, L15). The report states it and prints no whole proof of it (8.4, item 6, says what it prints and names the page that tells what has
    been checked of it). Where a count is not
    the lemma's, the programs add one list size to it for a mask code, which is read as repeated to
    the main code's length, and for a circle code (field 1). That this term covers the step from a
    code to a part of it is an argument of this project (comments in the two files named above), not
    a statement of the cited report.
  - The toy runs are sized under the theorem on codes. `toy_size` (`blocks/toy/product.h`) and
    `segments_size` (`blocks/toy/segments.h`) size their first group by `size_group` and the later
    stages by `solve_stages`, both up to the Johnson radius less a slack
    (8.4, item 6), at every size. So the distances, the numbers of positions and the round errors of
    `product` and `links`, at the full size too, are counted under that theorem.

---

# Part 4. CS (the code switch, in its two forms) and the proximity test on a group: what the verifier does, as mathematics and bytes

This part gives the verifier's side of two blocks of the library: the code switch, in its two forms
CS^sigma and CS, and the proximity test on a group, which joins BAT, FOLD, CS and BASE into one
test. It needs parts 1 and 3 and the format page.

## 0. Conventions

Fields, tables, rows of the statement, eq, SEND and BAT are defined in part 1. Folding, <x, y>,
FOLD, BASE and the single rows are defined in part 3. This part speaks of three toy runs: `product`,
`links` and `key-and-entry` (parts 6 to 10). What is used here:

**Index rule.** A table of 2^b rows is one vector: the entry at (row u, place c) has the index u +
2^b c.

The **chain** is the hash chain from which all of the verifier's messages are derived (the format
page, sections 4 and 5); a **query** is one call of it. A **chain of segments** is something else:
part 5's sequence of sub-proofs (R12.14).

A **code** C = (n, l, r) has the length n, a power of two, l message entries and r random entries
for each row (the row's **randomness**; the programs say "tail"). The code of main rows is called
the main code here; the format page says base code. The symbol of a row at position x is a public
linear form of message and randomness:

    Enc_C(M, R)(x) = <Gm_C(x), M> + <Gt_C(x), R>.

The pair (Gm_C(x), Gt_C(x)), with entries in F, is the **generator row** of C at x (routines
`generator_at`, `generator_row`). Part 1, 4.2, and 5.3 (c) below give the same name to a row of the
statement that has this pair as its coefficients. In both fields a position has factors f_0(x),
f_1(x), .. . With b_j(x) := the product of the f_i(x) over the bits i of j that are 1,

    Gm_C(x)[j] = ms(x) b_j(x) for j < l,   Gt_C(x)[j] = ts(x) b_j(x) for j < r.

- Field 2^64 - 2^32 + 1: f_i = z^(2^i), where z is the point of position x (R12.11); ms = z^r, ts
  = 1. So Gm_C(x)[j] = z^(r + j) and Gt_C(x)[j] = z^j.
- Field 2^61 - 1 (a code on the circle): ms = 1 and ts = b_v(x), where v is l rounded up to a
  multiple of the least power of two that is at least r. The programs of this repository
  can be built for field 2 alone and have no routine that encodes with this code; only their
  planning part has sizes for it (`blocks/base.h`, `blocks/code/`, `blocks/numbers/codes.h`; from
  the programs' source).

The **mask code** is Z = (n_zk, l_zk, r_zk), with n_zk <= n, both powers of two. Where a main row is
read at x, a mask row is read at cp(x) = x mod n_zk (R7.2; routine `mask_position`).

Letters without a prime belong to the **old** rows, which the switch reads; primed letters belong to
the **new** oracle message. s is the number of answers to an out-of-domain sample and t the number
of positions of a round; both are in the plan.

The verifier **rejects**: it says no to the proof. **The process ends**: the program stops, on a
fault of the public configuration or of itself. The programs' rule is that nothing in a proof's
bytes brings that about (head of `chain/rounds.h`). These stops are the programs' own; each is named where it occurs.

## 1. What a switch does, and its two forms

The prover sends a new oracle message G whose message holds the messages of the old rows and also
their encoding randomness. The symbol of an old row at a position x is then a public linear form of
G's message: the generator row of the old code at x. The verifier reads the old rows at t random
positions and keeps, for every symbol that it has read, the claim "this form of G's message has this
value". The programs' source gives as the reason: if the old rows are far from the encoding of what
G holds, some such claim is false but for a chance of (1 - distance)^t. After the switch no row of
the statement speaks of the old rows.

One routine (`switch_verify`) serves two forms.

- CS^sigma (`cs_sigma_verify`; its round of positions is named `LOC`). The claims wait as open rows
  on a group that the caller names; the caller may add members and rows to that group before its
  test begins. This form is used between the segments of a chain of segments (part 5).
- CS (`cs_verify`; round `CS`). After the positions, in the same verifier's message, comes one
  element theta of E, and all rows of the new group are batched at once with its powers. That step
  is BAT, the opening move of a proximity test on the new group (part 1, 4.7), with its element
  sampled early. This form is used between the stages of a proximity test (section 8); no other
  block calls it.

## 2. What a switch starts from: single rows (part 3, 2.5; struct `SingleRows`)

Public: the old group's layout (its members, k, h, which of the 2^k main and 2^h mask rows each
member holds, which members are **kept**), FOLD's challenges g_1, .., g_k, and which oracle message
is FOLD's fresh mask. The rows follow. None of them was sent as it is, but for the kept ones. Each
has a message and randomness:

- (M_0, R_0): the main rows folded at g_1..g_k: l and r entries of E;
- (M_Kb, R_Kb): row b of the kept member K, as it was sent, by member in the order of sending and by
  row; kappa such rows in all, under the code C;
- (M_1, R_1): the group's mask rows folded at g_1..g_h: l_zk and r_zk entries;
- (M_2, R_2): the fresh mask's rows folded at g_1..g_h: the same lengths.

**The claim** (part 3 writes its letters with a prime):

    <c, M_0> + the sum over the kept members K of <c_K, table of K> + <d_1, M_1> + <d_2, M_2> = tau.

c stays in structured form, c_K is a form on K's own main table, d_1 and d_2 are written out. A
group whose members have no mask row at all has h = 0 all the same, and one mask row that nobody
sent: M_1 = 0, R_1 = 0, and its symbol is 0 everywhere. The switch treats it like any other.

## 3. Where things are in G: public data (struct `SwitchTarget`), fixed before any proof is read

This public data is the **target** of the switch. (A row's target is something else: the row's right
side. The target in bits of section 7 is the security level that a plan aims at.)

### 3.1 The shape of G

Main rows over E under a code C' = (n', l', r'); mask rows, if it has any, over E under a mask code
Z' = (n'_zk, l'_zk, r'_zk); s >= 1 answers; and two aligned blocks of rows which together are all
its rows.

**The main block** is the rows 0 .. 2^p - 1 of the main table. It takes M_0, row index first: M_0[j]
is in row j mod 2^p at place j div 2^p. Let b_m be the main table's number of row bits: p if the
rest is in mask rows, p + 1 if it is in main rows (for q < p the table then has rows that nobody
sends). Then M_0[j] has the index (routine `cut_index`)

    mu(j) = (j mod 2^p) + 2^(b_m) (j div 2^p).

**The rest's block** has 2^q rows. It is either all mask rows of G (then the main block is all its
main rows), or, if G has no mask rows, the main rows 2^p .. 2^p + 2^q - 1 (then q <= p, because a
block begins at a multiple of its size).

### 3.2 The rest

**The rest** is one vector (routine `switch_parts`):

    M_1 | M_2 | R_0 | the R_Kb, by member and row | R_1 | R_2 | the old pads of the kept members, by member, each pad's places by increasing index.

Where its parts begin:

    A_1 = 0;  A_2 = l_zk;  T_0 = 2 l_zk;  T_Kb = 2 l_zk + r (1 + the row's number among the kappa kept rows, from 0);  T_1 = 2 l_zk + r (1 + kappa);  T_2 = T_1 + r_zk;  O_K = T_2 + r_zk + the number of pad's places of the kept members before K.

A pad has one place for each answer if the rows are over E, three if they are over F (part 1, 2.4).

A choice of the programs: where the two kept items, the R_Kb and the old pads, are among the others.

### 3.3 The places of the rest (class `SwitchPlaces`)

Number the entries of the rest's block row index first, as in 3.1: number j is (first row of the
block + j mod 2^q, place j div 2^q). Strike out the places of G's own pad, which SEND puts at the
end of G's last real mask row or, without mask rows, of its last real main row (part 1, 2.4; routine
`Send::pad`): in both cases a row of this block. P(i) is the index, in its table, of the i-th entry
that is left, i from 0. **The number of places** is 2^q times the length of a row of that block
(l'_zk or l'), less s. Entry i of the rest is at P(i). Below, m(j) is the entry of G's main table at
mu(j), and w(i) the entry of G at P(i).

### 3.4 Free room

The places P(i) from the rest's length on (and, where M_0 is shorter than its block, the block's
entries after it) belong to G's message like all others: the sample row gives them their weight, and
no other row reads them. The source's honest prover fills them with uniform elements of E in CS and
with zeros in CS^sigma, and that is what the programs of this repository do. A verifier cannot tell
the two apart. One comment in the source (`blocks/iop/cs.h`) gives as the reason for the first that
the argument for hiding that CS rests on counts these entries as uniform. The comment names no text.
The reason is the source comment's alone: that
hiding needs uniform values in this room is not shown here (section 11).

### 3.5 Kept members

Each kept member K of the old group is committed to a second time, as an oracle message K^new of its
own in G's prover's message: the new group's code, fresh randomness, an out-of-domain sample of its
own; the same rows, field, message length and number of answers as K; no mask rows. Its table is K's
but at the pad's places: there K^new has a fresh pad of its own, which SEND samples (part 1, 2.6),
and K's old pad is in the rest (3.2). The target says for each whether K^new is a kept member of the
new group once more or an ordinary one.

A choice of the programs: K^new is an oracle message of its own in both cases, also where its rows
are not kept once more. That costs one answer and one pad more for each kept member at its last
switch than if its rows became rows of G.

### 3.6 What is asked of a target (`switch_target_check`; else the process ends)

The codes exist; l is a power of two (a generator row on M_0 is kept as a product over the bits of
j); blocks as in 3.1; 2^p <= l <= 2^p l'; the rest's length is at most the number of places (3.3);
one K^new for each kept member, shaped as in 3.5. The routine also asks (from the programs' source,
`cs.h`): the single rows have exactly two mask rows, the old one and the fresh one; G has at least
one main row; M_0 goes into main rows; a kept member has no mask rows, neither K nor K^new.

A choice of the programs: a main message need not fill its block (l < 2^p l' is allowed). The
proximity test asks for the exact filling (8.1). Only a test of the programs is known to use the
wider case.

### 3.7 The two targets in use

**A stage** of a proximity test (`switch_target_stage`): 2^k' main rows, 2^h' mask rows over E under
Z' = (n'_zk, l'_zk, r'_zk), p = b_m = k', q = h', no kept member. So mu(j) = j: G's main table, read
as a vector, is M_0. P(i) is the i-th index of the mask table that is not a pad's; the pads have the
indices (2^h' - 1) + 2^h' (l'_zk - s + i), i < s. With s = 1 the pad is the table's last index, and
P(i) = i.

**The carrying message** of a chain of segments (part 5; R12.14): two main rows over E, no mask
rows, p = q = 0, b_m = 1. So mu(j) = 2 j (row 0) and P(i) = 1 + 2 i (row 1), whose last s places are
G's pad.

## 4. The messages, in order (`switch_verify`)

(1) Prover, one prover's message: G; after it the K^new in the order of the old members; after them
whatever else the caller sends along. That is either **riders**, which get no out-of-domain sample
(the fresh mask of the next fold, BASE's masking word: 8.3), or oracle messages of the caller's own
that get one (a segment's: part 5). In the proof: the roots of their trees, by R7.12 and R8.6.

(2) Verifier, round `SEND seed`, alone in its message: s times a point z and a factor beta that is
not 0 (part 1, 2.2). All oracle messages of (1) that get an out-of-domain sample share it: z has as
many coordinates as the largest of their cubes needs, and there are as many seeds as the largest of
their numbers of answers (in every plan these numbers are one and the same s: part 1, 2.1). For a
stage, where G is the one oracle message that gets an out-of-domain sample, z has max(k' + log2 l',
h' + log2 l'_zk) + 1 coordinates.

(3) Prover: the answers, one oracle message after the other as they were sent, each message's
answers by their number (part 1, 2.3). No round of SEND's own closes them: the next verifier's
message does.

(4) Verifier, round `LOC` or `CS`, one message: t positions x_1, .., x_t, each in [0, n), where n is
the old main code's length; in CS then one element theta of E, which may be 0 (R6.6, R6.10). A
position can come twice. Both the draws, in their order with repeats, and the list, the set of the
sampled positions in increasing order, are used below. If the plan gives this round proof of work,
its nonce is in the file after the answers and is the last word of this round's own query (R5.2,
R8.2).

(5) Prover, in the file alone, in no query of the chain, in the order of R8.7: the openings, at the list, of the trees that hold the old group's
members and the fresh mask.

There is no empty round inside a switch. In a proximity test FOLD ends with the verifier's g_k, so
(1) opens the next prover's message, and after (5) follows the next block's first item.

## 5. What the verifier computes and keeps

### 5.1 It reads the columns (`single_rows_read`, `read_opened`)

The oracle messages asked for are the old group's members, kept ones included, and the fresh mask.
They must be all that their trees hold, else the process ends: a leaf is opened whole. Oracle
messages that get an out-of-domain sample and share a tree must also belong to the same groups, else
the process ends (R7.17 (a)). A tree as long as the main code is opened at the list; a shorter tree,
of mask rows alone, at the set of the short positions (R7.15). Every opening is checked against its
root (section 6).

### 5.2 It forms the rows' symbols (`single_rows_symbols`)

At each listed x, as part 3, 2.5 says: y_0(x) = the fold at g_1..g_k of the column of main symbols,
placed by the layout, 0 for rows that nobody sent; v_Kb(x) = the kept rows' symbols as they are (a
symbol of F is read in E); y_1(x), y_2(x) = the folds at g_1..g_h of the group's and of the fresh
mask's columns of mask symbols.

In a tree as long as the main code the mask symbols used for x are the copies in the leaf of x
(R7.1). Nothing in such a tree makes two leaves over one short position agree (R7.11). If the
symbols y_1 or y_2 formed from them differ, two generator rows below have one left side and two
targets, so one of them is false. (This last sentence is an inference of this text, not a statement
of the programs' source.) In a shorter tree of mask rows alone (a fresh mask that is sent by itself
under a short mask code: 8.3, the choice 0) there is one leaf for all x over a short position, the
leaf of cp(x), and the question does not arise.

### 5.3 It makes rows (`switch_rows`, one routine for both sides)

For a form v on the main table of a kept member K, whose pad has the places e_0 < e_1 < .., write
(routine `switch_add_kept`)

    carry_K(v) = the sum over the indices e that are not a pad's place of v[e] K^new[e]  +  the sum over i of v[e_i] w(O_K + i).

The rows are made in this order.

(a) **The carried claim:**

    sum_(j<l) c[j] m(j) + sum_K carry_K(c_K) + sum_(j<l_zk) d_1[j] w(A_1 + j) + sum_(j<l_zk) d_2[j] w(A_2 + j) = tau.

(b) **The old sample rows** of the kept members, each as a row of its own, member after member as
they were sent, and by the answer's number:

    carry_K(the left side of K's old sample row) = K's old answer.

At the pad of its own answer that left side is the old factor beta, times the basis element if K is
over F; in this row it meets the old pad in G.

(c) **The generator rows:** for each of the draws i = 1..t in the order sampled, repeats counted,
with x = x_i and x' = cp(x):

    main row:       sum_(j<l) Gm_C(x)[j] m(j) + sum_(j<r) Gt_C(x)[j] w(T_0 + j) = y_0(x);
    each kept row:  carry_K(Gm_C(x) placed on row b of K's table) + sum_(j<r) Gt_C(x)[j] w(T_Kb + j) = v_Kb(x);
    old mask row:   sum_(j<l_zk) Gm_Z(x')[j] w(A_1 + j) + sum_(j<r_zk) Gt_Z(x')[j] w(T_1 + j) = y_1(x);
    fresh mask row: the same with A_2, T_2 and y_2(x).

In words: what G holds, encoded again under the old codes, has at x the symbol that the old trees
show.

That makes 1 + (the kept members' answers) + t (3 + kappa) rows. The new sample rows, of G and of
each K^new, are SEND's (part 1, 2.4) and stay with their oracle messages until a batching takes
them.

Why (b), as the comment before `switch_rows` in `cs.h` argues it. A kept member's old sample row was
batched into the claim (a) with all other rows of the old group. By the source, that alone would tie
K^new to K's message only through the count of the challenges of a fold's rounds that can make far
words look close. Beyond a third of the minimum distance that count rests on the theorem on codes,
which the programs' accounting takes as given (`numbers/codes.h`). So says this comment of the
source, with the file that it names.
Of the code words near K's old word, the one that the old group's rows speak of is the one that
agrees with the old answer. What ties K^new to that message is this row, read on K^new and, at the
pad's places, on the old pad in G. By the source, as a row of its own it lets the tie rest on the
list bound alone.

### 5.4 It admits them (`switch_admit`, which only the routine that has just run SEND and sampled the positions can call)

If G has mask rows, its ledger of free mask entries (part 1, 2.5) is closed at once: they hold what
the switch carries, and none is given out as a mask. G joins the group that the caller named; each
K^new joins it, kept or not as the target says; the rows of 5.3 become open rows of that group. (A
row with a coefficient other than 0 on a member's pad would end the process. None has one: P leaves
G's pad out, the main block does not reach it, and carry_K has 0 on the pad of K^new.) With that
CS^sigma is over.

### 5.5 In CS alone: the batching (`cs_verify`; `Statement::begin_run`, `Bat::combine`; part 1, 4.2 and 4.4)

The group is closed and passes on its rows in the fixed order: rows that are neither sample rows nor
generator rows, as created; the members' sample rows, the members in the order of sending; the
generator rows, as created. Row number N gets the factor theta^N, N from 0. Their number J is
compared with the plan's number for the round (a difference ends the process).

Between the stages of a proximity test the new group is made for the switch and has the one member
G, so J = 1 + s + 3 t:

- number 0: the carried claim;
- numbers 1 .. s: G's sample rows, with the answers alpha_1, .., alpha_s;
- numbers 1 + s + 3 (i - 1), and the two that follow it: for number i of the draws, the main row,
  the old mask row, the fresh mask row.

**The new row on the group** (c_new on the main table, d_new written out on the mask table, tau_new)
is the sum of theta^N times row number N:

    tau_new = tau + the sum over u from 1 to s of theta^u alpha_u + the sum over i from 1 to t of theta^(1 + s + 3 (i - 1)) ( y_0(x_i) + theta y_1(x_i) + theta^2 y_2(x_i) ).

For k' >= 1 the group's two tables are G's tables, index for index; for the last stage see 8.4. At a
pad's index only that pad's sample row has a coefficient: with s = 1, d_new's last entry is theta
beta.

Kept for the next step: (c_new, d_new, tau_new), the new group's layout, and which oracle messages
the riders are.

The values that the C++ verifier records, for anyone who compares another verifier with it (no
decision reads them). In both forms, CS^sigma too, and under the same names:
`CS: y_main at the first draw`, `CS: y_0 at the first draw`, `CS: y_1 at the first draw` are y_0,
y_1, y_2 of this text at x_1. After a CS alone, written by the proximity test's routine:
`CS: target` is tau_new; `CS: c'[0]`, `CS: d'[0]`, `CS: d's last entry` are entries of c_new and
d_new. So in `links` a reader meets five triples of recorded y values and two values `CS: target`.

A choice of the programs: where a caller's own rows are in this order. In a chain of segments they
come after (a) and (b) and before the new sample rows.

## 6. What a switch checks on the spot

No equation on field elements: all is deferred into the rows of 5.3 and, through the later steps,
into BASE's checks. What makes the verifier reject a proof inside a switch:

- (i) an opening that does not lead to its root (R7.10), or that gives a vertex a second value under
  one root: the record of values is kept by the root's 32 bytes for all chains of one verdict
  (R7.14; `chain/session.h`);
- (ii) an answer or an opened symbol that is missing or not in canonical form (R1.3);
- (iii) a nonce that is missing or fails the proof-of-work test, in a round to which the plan gives
  proof of work (R5.5). In the plans of `product` and `links` the rounds that sample positions
  (`CS`, `LOC`, `BASE`) have proof of work and the seeds' rounds have none;
- (iv) a file that ends too early: a root, a salt or a digest that is not there.

The verifier takes from the file exactly as many roots, answers, columns, salts and digests as the
plan, the shapes and the list fix; no count is read from a proof. There is no surplus inside an
opening: further bytes are read as the next item, and at the end of the file the verifier rejects
what is left (R8.1). Positions, theta, z and beta are derived from the chain, never read.

## 7. Orders, errors and hiding: the reasons that the programs' source gives

This section is argument, not proof (section 11). It follows the comments in `cs.h` (its head and
those before `SwitchTarget`, `switch_rows` and `switch_prove`) and in `iopp.h`, and the planning
routines.

Order. G comes before its seed and the seed before the answers: SEND keeps that. The positions come
after the answers. The source gives as the reason: near the word that was sent for G there may be
several code words, and a prover that knew the positions before it answered could pick the one that
fits them. So the positions are sampled by the routine that has just run SEND, and the channel's
handle for a seed's round can sample nothing else. By the source, theta may share the positions'
message because the rows that it batches are fixed by what was sent before and by the positions.

The errors that the plans count (`plan_send`, `plan_loc`, `plan_cs`). The seed's round: part 1, 2.7.
`LOC`: (1 - delta)^t, where delta is the distance at which the old group is tested. `CS`: (1 -
delta)^t + L' (J - 1) / |E|, where L' is a bound on the number of code words near the new group's
word. t is the least number for which that error, counted once for each instance that a setting lets
share the round, is at most 2^-(the target in bits - the round's proof-of-work bits) (routine
`positions`). Where the mask code is shorter than the main code the plans count it as its repetition
to the main code's length, because nothing makes the copies in the leaves agree.

Hiding. By the source it rests on this:

- the randomness that encodes G is fresh, uniform, and covers every position at which G is opened
  later;
- G's pad is uniform;
- the old rows were encoded with randomness enough for the positions read here (the prover's side
  counts the positions of every tree and ends the process at one too many: `open_oracles`);
- in CS the free room is uniform.

## 8. The proximity test on a group (`iopp_verify`)

### 8.1 Stages

Public: a list of S >= 2 stages, each four words (main code, mask code, k, h: R12.4), and one word
of choice (R12.5; 8.3). They are in the chain's first query. A **frame** is the part of the programs
that runs a protocol's blocks in their order (parts 5, 6, 9 and 10). The two entry points for
callers take the stages and the choice as handles to the first query. The routine under them takes
plain values; a frame that has read them out of its own item of the first query calls that routine
(`links` does). s is the plan's number, for every switch. Stage 1 is the group that is tested: its
codes and its layout's k and h must be stage 1's words, else the process ends.

The stages' sizes must fit each other (`iopp_fault`). `links` and the frames of parts 6, 9 and 10
ask that routine when they read the public files, and reject files that fail it. `product` does not:
there it is asked only inside `iopp_verify`, where a fault ends the process. (`toy_size` asks
another routine, `iopp_sized_fault`: whether stages and codes are those from which the plan's rounds
were counted.) The conditions:

- S >= 2 and s >= 1;
- at every stage: both codes exist and are codes of the field; l and l_zk are powers of two; n_zk <=
  n; h <= k; k is at most the largest logarithm of a code's length (`CODE_MOST_LOG_N`, 60);
- at every stage but the last: k >= 1, and l_zk >= 2 k + 1 - h (room for the coefficients of FOLD's
  hiding polynomial); at the last: k = h = 0;
- from a stage to the next (primed): 2^k' l' = l exactly; 2^h' l'_zk >= 2 l_zk + r + 2 r_zk + s; s
  <= l'_zk.

### 8.2 The sequence of blocks

1. BAT on the group (part 1, section 4): all its open rows and all its members' sample rows into one
   row (c, d, tau).
2. For i = 1, .., S - 1: FOLD at stage i (part 3, section 2), with that stage's fresh mask: out come
   g_1..g_k and single rows. Then CS (sections 3 to 6) into a new group whose one member is G_(i+1),
   shaped by stage i + 1's words: out comes the row on that group.
3. The last row becomes single rows without a challenge (8.4), and BASE decides (part 3, section 3).

A group with a kept member cannot enter this test: the stage's target has no K^new, and the process
ends (3.6). In a chain of segments such a group goes through BAT, FOLD and CS^sigma instead (part 5,
sections 6 and 7).

The rounds, their names and how each message is cut into items: R12.8, R12.9.

### 8.3 Which prover's message holds what

FOLD's fresh mask (no main rows, 2^h mask rows under its stage's mask code) and BASE's masking word
(a main row over E under the last main code and a mask row under the last mask code) are made from
randomness alone and never get an out-of-domain sample. In which prover's message they are sent is a
public choice.

The choice is 1, **masks ride**, as in `product` and `links`: the prover's message of the switch
into stage i + 1 holds G_(i+1) and, after it, the fresh mask of the fold at stage i + 1 or, if that
stage is the last, BASE's masking word. All have the leaf class 0, so by R7.12 they share one tree,
with one root (R12.10 has the leaves' widths). FOLD's first message is then the sum of its hiding
polynomial alone, and BASE's first message its one element.

The choice is 0: FOLD sends its fresh mask's root at the head of its own first message, and BASE its
masking word's likewise. A stage then has two trees; the fresh mask's has the mask code's length and
is opened at the short positions.

The first fold's mask is the caller's matter, whatever the choice: the caller tells the test which
oracle message it is, if it was sent earlier (in `product`: with a, b, c); else FOLD sends it.

Why a mask may ride, as the programs' source argues it: it is read at the same one list as its
stage's oracle message and at no other. A tree that holds such a rider is opened once in all; a
second opening ends the process (R7.17).

So each list opens one stage: the list of the switch out of stage i opens the tree or trees of stage
i (its rows and its fold's fresh mask), and BASE's own list opens the last stage's.

A choice of the programs: there is no packed variant, in which the next stage's hiding coefficients
would be inside the switch's own mask rows.

### 8.4 The last stage (`iopp_last_rows`)

G_S has a single main row and a single mask row. On G_S's own tables the step is the identity: the
claim is <c_last, main row> + <d_last, mask row> = tau_new, with c_last[j] the batched coefficient
on G_S's main entry j and d_last = d_new; no challenge, no fresh mask. Only the program's storage
differs: a group has at least one row variable, so this group's main table has a second row that
nobody sent, c_new has the index u + 2 j, and the variable is fixed at 0: c_last[j] = c_new[2 j].
Nothing of that formal row enters G_S's out-of-domain sample: its point has max(log2 l', log2
l'_zk) + 1 coordinates.

### 8.5 The verdict

`iopp_verify` says yes if no step has rejected and BASE's checks hold; every other equation of the
run has by then been folded into BASE's claim. That yes says nothing of bytes that are left over. In
the toy runs the whole verifier's side runs inside one routine (`run_verifier`), which also asks
that every planned round was run, that no byte of the proof is left, and that no chain of the
verdict has rejected (part 5, section 10).

## 9. In the toy run `product`

Three stages, (k, h) = (2, 0), (1, 1), (0, 0), with codes (C_i, Z_i) at stage i; masks ride; s is
the public file's `samples` (1 in the files of this repository). O1 is the toy run's first oracle
message, with the rows a, b, c and one mask row (part 1, section 3); tree 0, tree 1 and tree 2 are
the trees of the three prover's messages that hold oracle messages, in their order. The lengths n of
the main codes are 2^15, 2^14, 2^14 at the full size (the files named `product`) and 2^7, 2^8, 2^9
at the tiny size (the files named `tiny`). The program that writes the files for second verifiers
(`blocks/toy/handover.cpp`) makes the mask codes as long as the main codes, so that cp is the
identity, unless it is given the word `short-masks` (then 2^11, 2^12, 2^14 at the full size). l =
2^12, 2^11, 2^11 (tiny size: 2^4, 2^3, 2^3). The other numbers (r, r_zk, l_zk of the later stages,
t) come from the planning routines and are in the run's tables of codes and rounds.

1. BAT on {O1}: part 1, section 5.
2. FOLD at (C_1, Z_1, 2, 0) with the fresh mask that rode with O1: g_1, g_2.
3. CS into stage 2: G_2 has 2 main rows (G_2's main table as a vector is M_0) and 2 mask rows;
   rider: the second fold's fresh mask, 2 mask rows under Z_2. The list, over [0, n_1), opens tree
   0: a, b, c, O1's mask row, the first fold's fresh mask. In y_0 the fourth row is 0.
4. FOLD at (C_2, Z_2, 1, 1): g_1, at which the mask columns are folded too.
5. CS into stage 3: G_3's single main row takes M_0 whole (l_3 = l_2); it has a single mask row;
   rider: BASE's masking word. The list, over [0, n_2), opens tree 1.
6. 8.4, then BASE; its list, over [0, n_3), opens tree 2.

Each switch batches 1 + s + 3 t rows, where t is its own round's number.

## 10. What the tests cover, and what they do not

**The second verifiers in Python** (`tests/new/second/`) call none of the programs' code. All of
them have one writer, so they are not independent of each other.

**How independent they are of the C++ sources.** The verifier for `product` is not independent of
the C++ programs on these: the first lines of the struct that describes a code (among them, where
the message's coefficients begin), parts of `chain/rounds.h` and `iop/context.h`, and the head of
`iopp.h`. What it does with false proofs is not independent of lines of `cs.h`. Its head lists more
points and files (section 6 at the head of this page). On where randomness and message are in a code word it can share a misreading with the C++
programs. The factors ms = z^r and ts = 1 of section 0 are that very point. The other second
verifiers use its channel and its field, so these limits are theirs too.

- `product`: two switches and the last step, masks that ride, mask codes as long as the main codes.
  A script of the programs' tests, which also builds malicious provers and is not part of this
  repository, makes the files (the page's section 6). On these files the second verifier accepts the
  honest proofs at both sizes and one in which a position is sampled twice in a switch's list. It
  finds every value that the C++ verifier records (5.5) equal to its own. It rejects altered copies
  and the false proofs of malicious provers.
- Which false proofs bear on this part. For `product`: provers whose G holds, at either switch,
  another main entry than the folded row has, another carried mask entry, or another carried entry
  of the main row's randomness (in this text's letters: one entry of M_0, of M_1, of R_0). For
  `links`: another main entry at three switches, and another kept row, old pad or kept row's
  randomness. No false proof in the lists changes M_2, R_1 or R_2.
- `links`: the same formulas after a chain of three segments (five switches).
- The key file of `key-and-entry`: the general form of the test, that is masks that ride or are sent
  by themselves, a mask code shorter than its main code, several oracle messages and trees in a
  group. The second verifier accepts the honest key file. The leaf file's verifier has the same formulas written out once more by
  the same writer: it is not an independent implementation.
- A proof file of a toy batch: `proof_check.py` has the formulas of the switch between segments once
  more (routine `links`: the carried claim, the old sample row of a kept message, the generator
  rows), carried over from `links_check.py` by the same writer. Its last proximity test is that of
  `zkpy.py`. By its head it is less independent than the other verifiers in its directory: it is not
  independent of the C++ tool that writes its files, nor of the C++ programs on the points that the
  head lists, where it can share a misreading with them. The list names files of `frame/`
  (`check.h`, two of its tests and `handover_files.cpp` among them), files of `iop/` or parts of
  them (`req.h`, `batt.h`, `batt_shown.h`, `psc.h`, `hp.h`, `member.h`, `tensor.h` and `lin.h` among
  them), and every point and file that the heads of the other verifiers name; it is longer than this
  sentence. By the same head, of the unnamed parts of these files and of any other file of the C++
  programs nothing is known either way.

**The C++ programs' own tests** are not evidence of the kind above: they share this text's reading
of the protocol.

## 11. Limits of this part

- This part says more, in general form, than the toy runs use
  (3.1 to 3.6 for any p and q; 3.3 with several pads). The general form is read from the routines;
  outside the two targets of 3.7 nothing has tested it.
- More than one answer (s > 1): with several mask rows the pads are not side by side in the index,
  and P steps over each of them. This is read in `SwitchPlaces`, not run: no file has s > 1.
- The routine would also take CS with kept members, and a rest in mask rows beside
  kept members. No caller has these two cases. FOLD always passes on two mask rows. The source ends the
  process unless there are two (`switch_target_check` in `cs.h`). The test is in `cs_test`: with no
  mask row and with one, the prover's switch and the verifier's must each end the process with that
  stop's words. (from the programs' source)
- The field 2^61 - 1: `code/code.h` does not have the formulas of section 0, because the programs can be built
  for field 2 alone. The format page does not define that field's points, and no second verifier has
  run it.
- Not in this part. Which point belongs to which position, a leaf's bytes, the walk of an opening, a
  round's query, how elements and positions are sampled: the format page. The sample row: part 1.
  FOLD, BASE, and how kept rows are placed in a group and in FOLD's claim: part 3. Which members are
  kept, for how long, and where the last carrying message goes: part 5.
- Section 7 is argument, not proof, and so are section 1's "(1 - distance)^t", 3.4's reason, 5.3's
  "Why (b)" and 8.3's "Why a mask may ride". They follow the comments in `cs.h` (its head and those
  before `SwitchTarget`, `switch_rows` and `switch_prove`) and in `iopp.h`, and the planning
  routines. In particular, these three are argued and
  not proved: that the old sample row of 5.3 (b) is what soundness needs and is enough; that uniform
  free room is needed in CS and zeros do no harm in CS^sigma; that a short mask code may be counted
  as its repetition.
- delta, L' and t: the planning part of the programs (`numbers/`), not this part.
