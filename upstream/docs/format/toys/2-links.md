# The toy run `links`: a chain of three segments: which protocol it runs, and the verifier's decision

A toy run is a small run of the library's blocks on synthetic data. This page builds on the page on
the toy run `product` (`1-product.md`: its sections and parts 1 to 4) and says only what is new. The format page is `FORMAT.md` in the
directory above; its rules are cited by number (R12.12). In prose, one piece of a request's proof is
a segment. Hashed and printed names call it a link: the run's name `links`, the item
`the chain of links`. A sentence marked "(from the programs' source)" was read in the source and not run. No program, the second verifiers among them, has been written from this page's text word for word. It has not been compared with the paper that this protocol builds on: there may be differences that are not listed here, and what it gives as a choice of the programs has not been held against that paper's text and is not claimed as new.

Where this page speaks of hiding, it
gives what the design is meant to do and the reasons of the programs' source. All that this page
tells of a prover's side is the library's CPU reference at toy size; it says nothing of the GPU
engine, the kernels and the sampling of `cp_commit` and `cp_prove` (from the programs' source,
`blocks/tests/hiding/README.md`).

## 0. What it is

The run is the frame in which a request's sub-proofs follow each other (`blocks/iop/req.h`), at its
smallest: three segments, each a product of two vectors, where the product of one segment is a
factor of the next. The prover knows x_1, b_1, b_2, b_3 over F. With c_m := x_m * b_m entry by entry
and x_(m+1) := c_m, its claim is that the three committed products are right. The protocol is
designed to show this without showing the vectors. There is no model, no key file, no stored table
and no next token. There is one chain, and the chain's label is 2. The statement's bytes are empty.

There are two sizes, as for `product`. The full size has rows of 2^12 entries and vectors of 2^11;
it is planned for the target of 128 bits, with 20 proof-of-work bits. What that plan rests on, the
theorem on codes (Theorem 4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson,
Carmon, Haböck, Kopparty and Saraf published at STOC 2026) included, is in part 5, section 14, and in part 3, section 6. The
tiny size has rows of 64 entries; it is a test size that reaches no security level, and no check is
weakened in it.

The same as in `product`: the format page; the blocks SEND, ZERO, PSC, BAT, FOLD, CS, BASE, each as
in parts 1 to 4; codes, leaves, openings; the decision, D1 to D5 with D3b, word for word; the last
proximity test, with every mask riding.

What is new:

- (a) One prover's message holds several oracle messages, up to five, in one tree. Those that get an
  out-of-domain sample share one seed: its point has as many coordinates as the largest of them
  needs, and each reads the first coordinates (R6.9). The next prover's message begins with one
  answer for each, in the order of sending.
- (b) Oracle messages without mask rows. The pad of their out-of-domain sample is in a main row: one
  entry if the rows are over E, three entries of F if they are over F (part 5, sections 1 and 3).
- (c) A kept row. The product c_m of segment m is read again by segment m + 1. It is sent again in
  segment m + 1's tree, with a fresh pad, and two things are there to tie the second sending to what
  went before. A fourth row at each sampled position is there to tie it to the old tree. And the
  kept message's old sample row, read on the second sending and the old pad, is a row of its own
  among the next group's rows. That the two tie the second sending to the first is an argument of
  the programs' source (the comment before `switch_rows` in `blocks/iop/cs.h`) and not a proof (part 5, sections 7 and 14). (A sample row is the row that states
  one answer to an out-of-domain sample: part 1.)
- (d) A round `LOC`: positions without an element after them. The rows that a switch makes there are
  not batched on the spot: they wait as open rows of the next segment's group and are batched at its
  closing (its BAT and FOLD).
- (e) Groups of several members, some of whose rows nobody sent; a group without a mask row (the
  last); leaves that mix rows over E and over F among their main symbols; a vector that fills the
  lower half of a row.

## 1. The first query

The parameters are, in this order: the item `the format's version`; the plan; the items
`configuration`, `codes`, `the chain of links`. The last is given word by word in part 5, section
11, and is defined by R12.12 to R12.14: a message has six words, a stage four; the last test's first
stage is not written, because it follows.

## 2. The plan, the trees

- Full size: 76 rounds, each one query of the chain, counted from 1; the proof has 249,456 bytes.
  One root each in rounds 1, 21, 42, 64, 69, 73. Positions are sampled in round 22 `LOC`: 73 in [0,
  2^15); round 43 `LOC`: 73 in [0, 2^15); round 65 `LOC`: 73 in [0, 2^15); round 70 `CS`: 73 in [0,
  2^15); round 74 `CS`: 74 in [0, 2^14); round 76 `BASE`: 74 in [0, 2^14); 20 proof-of-work bits
  before each. Rows batched: 5 (round 18), 301 (round 38), 301 (round 59), 221 (round 66), 221
  (round 70), 224 (round 74).
- Tiny size: 58 rounds, each one query of the chain, counted from 1; the proof has 12,800 bytes. One
  root each in rounds 1, 15, 30, 46, 51, 55. Positions are sampled in round 16 `LOC`: 3 in [0, 2^9);
  round 31 `LOC`: 3 in [0, 2^9); round 47 `LOC`: 3 in [0, 2^9); round 52 `CS`: 3 in [0, 2^9); round
  56 `CS`: 3 in [0, 2^9); round 58 `BASE`: 3 in [0, 2^10); 6 proof-of-work bits before each. Rows
  batched: 5 (round 12), 21 (round 26), 21 (round 41), 11 (round 48), 11 (round 52), 11 (round 56).

Each segment runs SEND, ZERO, PSC, BAT, FOLD. Then come the next segment's SEND and, in the round
`LOC`, the positions at which the segment's own tree is opened. After segment 3 come the last
group's SEND and `LOC`, then BAT and the last proximity test (FOLD, CS, FOLD, CS, BASE). Tree number
i, counted from 0, is opened at the (i + 1)-th round that samples positions. At the closings of the
three segments, 5, 9 + 4 t and 9 + 4 t rows are batched, and at the last group's 2 + 3 t, where t is
the number of positions of a round `LOC`. R12.16 to R12.18 define groups, rounds and trees; the tool
`blocks/toy/handover.cpp` writes `<run>.plan.tsv` and `<run>.fields.tsv`. Part 5, section 13, prints some of them
again (rounds, positions, rows batched).

(There are no sections 3 and 4 here: the trees are in section 2, and leaves, openings and codes are
as in sections 3 and 4 of the page on `product`.)

## 5. The verifier's decision

The decision is D1 to D5 of `product`, D3b among them, with six trees and six rounds that sample
positions. Nothing else is tested on the proof's bytes. No equation of a segment is tested on the
spot: every claim is carried from segment to segment, and the only test of it is at the last step.
Each tree is opened at one list, so what the page on `product` says on D3b holds here too.

## 6. The files, and the limits of what they cover

The script and the kinds of files are the same as for `product` (section 6 there, which also says
why nothing is hidden in these files); the C++ verifier notes values on its way (part 5, section
10).

The second verifier is `tests/new/second/links_check.py`. On the files that the script makes,
`run_honest.sh` checks three things of the second verifier. The second verifier accepts the honest
proofs. It finds all digests, rounds and values equal to the C++ tool's. It rejects each altered
copy at the same check as the C++ verifier. `run_honest.sh` makes no false proof
(`tests/new/second/README.md`). A fourth thing was checked by another script of the programs' tests,
which builds malicious provers and is not part of this repository: the second verifier rejects each
false proof at the check that the list of this run's false proofs names; that list is not in this
repository either. The honest proofs are three: full, tiny, and tiny under a public number at which
a position is sampled twice in a list of a round `LOC` after a segment with a kept row. Among the
false proofs are some that only the frame's own rows can reject. Where a false proof changes an
entry of a second sending or of the carrying message (part 5, section 2), the list names the kinds
of row that are false.

Limits:

- No file exercises D3b, or a non-canonical element in a message. (An opened symbol that is not an
  element's canonical form is altered copy 11.)
- As for `product`, the parameters' bytes have not been derived from the public file a second time:
  the second verifier takes them as given.
- Of the 18 lines of the list of this run's false proofs, 10 name the tiny size alone: those false
  proofs are not made at the full size.
- Not covered: field 1; more than three segments; more than one kept row at a time.

---

# Part 5. The frame of a chain of segments: what the verifier does, as mathematics and bytes

Section 14 lists the limits of this part: what only the toy's shapes
cover, and what no test covers.

## 0. What this part covers, and its conventions

The proof of one request is a chain of L segments. A **segment** is a piece of the computation with
a sub-proof of its own. Its prover sends oracle messages, each of which gets an out-of-domain sample
at once (SEND, part 1), and its sub-proof ends in rows: linear equations, with public coefficients
and public targets, on the underlying messages of these oracle messages. The frame knows nothing
else of a sub-proof. It gathers a segment's oracle messages in one group. It batches all rows of the
group into one claim and folds the group (BAT, part 1; FOLD, part 3): these two steps are the
segment's **closing**. It carries what the fold leaves into the next segment's group by the code
switch without a batching element (part 4). After the last segment it runs one full proximity test
(part 4). The frame checks no equation. Before that test's last step (BASE, part 3) its verifier
checks form, proof of work and openings alone (section 7 (d)). But for the errors of the rounds on
the way, a prover that departs from what it has committed to merely turns one false claim into the
next. Those errors are the planning routines' figures; the ones of FOLD and BASE, and what they rest
on, are in part 3, section 6.

This part says what the frame adds to those blocks, and describes the verifier. The code is
`blocks/iop/req.h` (the frame), `blocks/iop/cs.h` (the switch), `blocks/iop/relation.h` (groups and
rows, which the frame uses and does not define), `blocks/iop/req_rows.h` and
`blocks/frame/req_frame.h` (section 12). The running example is the toy run `links`,
`blocks/toy/segments.h`.

*What a reader needs.* For a verifier of the toy run `links`: sections 1 (its last paragraph, on the
toy run, with the list after it), 2, 4 to 9, 11 and 13. Section 3, "Seeds and answers" in section 4,
6 (d) and 7 (a) to (d) say once more, in this part's letters and at the frame's shapes, what parts
1, 3 and 4 say in general. Plain scalars, what rides at the start, later moments and lives above 1
(section 1 defines the four) do not occur in the toy, nor does anything of section 12.

*The letters* of this part are those of the C++ verifier's notes and of the second verifier. Parts
1, 3 and 4 use other letters for the same things, some with shifted numbers. Side by side:

| this part | part 4 (and parts 1, 3) | what it is |
|---|---|---|
| y_main, y_0, y_1 | y_0, y_1, y_2 | the folded symbols at a position: main rows, old mask rows, fresh mask |
| M*, W*, Y* | M_0, M_1, M_2 | the folded messages, in the same order |
| rho*, sigma*, pi* | R_0, R_1, R_2 | their randomness ("tail" here) |
| kappa | the R_Kb | the kept rows' tails |
| w | kappa | the number of kept rows |
| c*, K*_i, d_0, d_1, tau* | c, c_K, d_1, d_2, tau | the claim that FOLD leaves |
| A0, A1, T, TK_q, T0, T1, OP_i | A_1, A_2, T_0, T_Kb, T_1, T_2, O_K | where the parts of "the rest" begin |
| G[2 j], G[1 + 2 i] | m(j), w(i) | entries of the carrying message: of its main block, of the rest |
| gamma_0, .., gamma_(k-1) | g_1, .., g_k | FOLD's challenges |
| x' | cp(x) | the short position under x |

Some letters have two uses inside this part. In the table of section 4, "P:" and "V:" mark the
prover's and the verifier's messages, while P_m is an oracle message of the toy. K alone is a number
of rows, while K_i and K*_i are forms. c is the constant of the field's extension and also the
coefficient vector of a claim, and c_m is a vector of the toy.

Conventions:

- F is the base field and E = F[X] / (X^3 - c) its cubic extension (R1.2 and R1.4 of the format
  page). An element of F counts as an element of E wherever one is wanted.
- *Positions and places* (part 1). A position is always a position of a code word, a number in [n].
  The place of an entry in a row of a table is its number in that row, from 0. (R14.5 calls a place
  a position in the message's table.)
- *The index rule.* A table of K rows of one length is read as one vector with 2^rb rows, where rb =
  ceil(log2 K) is its number of row bits: the entry in row i at place j has the index i + 2^rb j.
  Rows beyond the K real ones are zero. Bit 0 of an index belongs to the first variable.
- eq(g, i), for a point g of d coordinates in E and 0 <= i < 2^d, is the product over u < d of g_u
  if bit u of i is 1, and of 1 - g_u if it is 0. The fold at g of a vector v of 2^d entries is the
  sum over i of eq(g, i) v[i].
- "Field 1" is the field of 2^61 - 1 elements and "field 2" that of 2^64 - 2^32 + 1. [n] is the set
  of the positions 0, .., n - 1.
- A code C = (n, l, r) has length n = 2^(log n), l message entries and r random entries, the tail.
  The symbol at position x of the row with message m and tail rho is sum_{j<l} Gm_C(x)[j] m[j] +
  sum_{j<r} Gt_C(x)[j] rho[j], with public coefficients in F: the **generator row** of C at x. In
  field 2, Gm_C(x)[j] = z^(r+j) and Gt_C(x)[j] = z^j, where z is the point of position x (R12.11). A
  mask code Z = (n_zk, l_zk, r_zk) is a code whose length divides n; where a main row is read at x,
  a mask row is read at x' = x modulo n_zk.
- A **row** is an equation "sum of coefficient times entry = target" on entries of the underlying
  messages of oracle messages, never on tails. The **sample row** of an answer is the row that
  states this answer to an out-of-domain sample (part 1).
- The entries of the mask rows of a message that gets an out-of-domain sample serve as masks of
  scalars that are shown. Its **ledger** gives them out, in the order of their indices and each only
  once, and only those whose index is below the first pad's index (`blocks/iop/mask.h`; part 1,
  2.5).
- Segments, groups and trees are counted from 1 here. The programs count all three from 0, the
  format page counts trees from 0.
- "The draws" and "the list" of a round that samples positions are those of R6.6: the positions in
  the order sampled, repeats kept; and the set of them in increasing order.
- "The process ends" means that the program stops, on a mistake of its own or of the public
  parameters (as in parts 1, 3 and 4). The programs' rule is that nothing which depends on a proof's
  bytes does that: there the verifier rejects. (How far that rule has been followed through the
  code: section 14.)
- A sentence that begins "A choice of the programs:" states an arrangement that is these programs'
  own and could have been made otherwise.

## 1. What the frame is given: the shape

All that follows is a function of three public things. The first is the shape (`ReqShape`), which is
in the chain's first query (R4.3) as the item `the chain of links` (its bytes: section 11). The
second is the list of codes. The third is the plan, which has the number t of positions of a round
`LOC` (section 4; the same at every such round, because all segments' groups are sized as one), the
proof-of-work bits of every round and the number s of answers to an out-of-domain sample. The shape
names:

- one main code C = (n, l, r) and one mask code Z = (n_zk, l_zk, r_zk), the same for the groups of
  all segments;
- s >= 1, the number of answers to every out-of-domain sample of the chain. (It is among the plan's
  words too; the frame's constructor ends the process if the two differ.)
- L >= 1 segments. A segment has a number of **plain scalars** (section 4) and a list of oracle
  messages. Each message has: its main rows, all over F or all over E; its mask rows, always over E;
  a leaf class (R7.12); a **moment**; and a **life**.
  - A segment's oracle messages are sent in one or several prover's messages, its moments 0, 1, 2,
    ..; the sub-proof speaks between them.
  - Life 0: only the message's own segment reads it. Life f > 0: later segments read its main rows
    again, the last of them f segments on. Such a message is **kept**.
- **what rides at the start**: oracle messages that are sent in the very first prover's message and
  belong to no group of the frame;
- the stages of the last proximity test and the one word of its choices (part 4).

*What a shape must meet* (`public_fault` of `ReqMessage`, `ReqLink` and `ReqShape`; with the codes:
`req_fault`). Every message has at least one row, and at most 2^20 of either kind. A segment's
messages are listed by moment, moment 0 is not empty and no moment is left out. A kept message has
main rows, has no mask rows, is sent at moment 0, and its last reader is a segment of the chain. The
first segment has a message of life 0 with main rows (from the programs' source). What rides at the
start has mask rows alone, gets s answers, and has a leaf class greater than that of every message
of every segment. The last test has at least two stages, and its first stage is (C, Z, k, h) of the
last group, where a group has 2^k main rows and 2^h mask rows (section 2). No group has more row
variables than the field's longest code has position bits. With the codes: every code that is named
exists; s < l and s < l_zk, and s is below the message length of the mask code of whatever rides at
the start; for each segment's group l_zk >= 2 k + 1 - h (room for the coefficients of FOLD's hiding
polynomial); what a closing carries on fits one row (section 7 (a)); and the last test's sizes close
(`iopp_fault`, part 4). These routines return a reason and end nothing, so that a verifier which
reads a shape from a file can ask them first and reject. The routine that writes a first query and
the frame's constructor ask them once more and end the process on a fault: a shape is fixed long
before a proof is read.

The frame itself allows mask rows in any message that is not kept. The routine that builds a shape
from a layout file (section 12) gives each segment one mask row, in the first message of its first
moment, and every other message that gets an out-of-domain sample then has its pad in its last main
row (section 3). So does the toy.

*The toy run `links`.* The prover knows vectors x_1 and b_1, .., b_L over F, of l / 2 entries each.
Its claim is that it knows c_m = x_m * b_m, entry by entry, with x_(m+1) = c_m: each segment
multiplies what the segment before has left. The protocol is designed to show this without showing
the vectors. The public file names L; all files that the script of section 14 makes have L = 3, and
s = 1. A vector fills the lower half of a row, places 0 to l/2 - 1; the upper half is zero but for a
pad. There are no plain scalars, nothing rides at the start, every segment has the one moment 0, and
every leaf class is 0. Segment m sends two oracle messages of its own, in this order:

- B_m: main rows over F, two for m = 1 (row 0 = x_1, row 1 = b_1) and one for m > 1 (b_m); one mask
  row, the segment's; life 0.
- P_m: one main row over F, c_m; no mask row; life 1 for m < L, because segment m + 1 reads it as
  its x; life 0 for m = L.

## 2. What follows from the shape: three kinds of oracle message, and the groups (`req_groups`)

There are L + 1 groups, all under (C, Z): group m for segment m, and group L + 1, the **last
group**, on which the last proximity test runs. Three kinds of oracle message are in no word of the
shape, because their shapes follow from it. G_m and U_m have the leaf class 0; a kept message that
is sent again keeps the leaf class of its first sending (`req_groups`; R14.4 (d)).

- G_m, the **carrying message** (R12.14), for m = 2, .., L + 1: two main rows over E under C, no
  mask rows, s answers. It carries group m - 1 on into group m (section 7).
- A **kept message sent again**. A message P of segment j with life f is sent again in the first
  prover's message of each of the segments j + 1, .., j + f. Each sending is a separate oracle
  message that has exactly P's shape (the same rows over the same field under C, no mask rows, s
  answers), with a fresh tail, a fresh pad and a fresh out-of-domain sample.
- U_m, the **fresh mask** of the fold of group m, for m = 1, .., L + 1: 2^h mask rows over E under
  Z, with h of group m, and no answers. It gets no out-of-domain sample and is a member of no group
  (part 3 says what it holds).

*Members.* Group m has as members, in the order of sending: G_m (if m > 1); the kept messages of
earlier segments whose life reaches segment m, as they were sent again for segment m, the one that
was first sent earliest coming first; and segment m's own messages, moment by moment. A kept message
of segment j with life f is a **kept member** of the groups j, .., j + f - 1, each time in its
latest sending, and an ordinary member of group j + f.

*Layout* (`Statement::layout` and `lay`, `blocks/iop/relation.h`). An ordinary member with K main
rows takes an aligned block of up(K) rows among the group's 2^k main rows, where up(K) denotes K
rounded up to a power of two. Blocks are placed from row 0 by decreasing size, ties in the order of
sending; rows that nobody sent are zero. Mask rows are placed in the same way among 2^h rows. A kept
member's main rows have no place among the 2^k rows. h is the logarithm, rounded up, of the number
of mask rows so counted (0 if there are none), and k is the greatest of: the logarithm, rounded up,
of the number of main rows so counted; h; and 1. (R12.16 states the same rule.) If a member's block
begins at row number first, the entry of the member's own table in row i at place j has the index
(first + i) + 2^k j in the group's main table, and likewise with 2^h in the mask table. The frame
computes k and h from the shape alone, the class `Statement` computes them from the members that
have joined, and the process ends at a closing if the two differ.

The last group has the one member G_(L+1), because nothing is kept beyond the last segment: k = 1, h
= 0. Its one mask row is a row that nobody sent.

At the toy, writing P'_m for P_m sent again (it is sent with segment m + 1):

- group 1: members B_1, P_1 (kept). Main rows: 0 = x_1, 1 = b_1. k = 1. Kept row: c_1.
- group 2: members G_2, P'_1 (ordinary: its life ends here), B_2, P_2 (kept). Main rows: 0, 1 = G_2;
  2 = c_1 in P'_1; 3 = b_2. k = 2. Kept row: c_2.
- group 3: members G_3, P'_2, B_3, P_3 (life 0, ordinary). Main rows: 0, 1 = G_3; 2 = c_2 in P'_2; 3
  = b_3; 4 = c_3; 5, 6, 7 zero. k = 3. Nothing kept.
- group 4, the last: member G_4. Main rows 0, 1. k = 1.

In groups 1 to 3 mask row 0 is B_m's, and h = 0 throughout.

*How a kept row is treated, and why.* It is read at the group's positions and tested with the group,
but FOLD leaves it as it was sent. A claim's coefficients on it get a public factor, eps *
prod_{j<k} (1 - gamma_j), where eps and gamma_0, .., gamma_(k-1) are FOLD's challenges (section 6
(d)). One way to see the factor: widen each of the 2^k main rows by one further row of C for each
kept row; put a copy of kept row q into the q-th widening of every one of the 2^k rows, and the
claim's coefficients on the copy in row 0 alone. The fold's weights eq(gamma, i) add up to 1, so
equal copies fold to the kept row itself; coefficients that are in row 0 alone fold to eq(gamma, 0)
times themselves. Nothing is sent, opened or folded twice, and k counts the ordinary rows alone. The
programs' error analysis counts a main row and the kept rows beside it as `slots`: 1 + the number of
kept rows, rounded up to a power of two (2 at the toy). No formula of the verifier has that number,
and it is not a request's slot in a batch. This arrangement is in the code of FOLD and of the class
`Statement`, not in the frame's (section 14).

A choice of the programs: the frame has one group for each segment. A segment with two groups, the
splitting of a row that touches both, and the message of masks alone that such a splitting needs,
are a block of their own (SPLIT, part 9), which the frame does not use.

## 3. The pad of an out-of-domain sample: the three kinds that occur here (`Send::pad`; part 1)

The pad of an answer is one element of E in the message. In that answer's sample row its coefficient
is the seed's factor beta, and it gets no weight from the seed's point z; no other row may read it
(`Statement::add_row` ends the process on a row that does). It is at the very end of the last real
mask row or, if the message has no mask rows, of the last real main row. For answer number i < s of
a message whose table in question has K real rows:

- with mask rows (B_m): the entry of the mask table in row K - 1 at place l_zk - s + i;
- without mask rows, main rows over E (G_m): the entry of the main table in row K - 1 at place l -
  s + i;
- without mask rows, main rows over F (P_m, P'_m): three entries p_0, p_1, p_2 of F, in row K - 1 at
  the places l - 3 s + 3 i + u for u = 0, 1, 2. The pad is p_0 + p_1 X + p_2 X^2, so the entry p_u
  has the coefficient beta X^u in the sample row.

The sample row of answer i has the coefficient 0 on the pads of the other answers. At s = 1 the
indices are: l_zk - 1 in B_m's mask table; 1 + 2 (l - 1) = 2 l - 1 in G_m's table; l - 3, l - 2, l -
1 in P_m's.

## 4. The messages of the frame, in order (`ReqVerifier`: `first`, `later`, `close`, `finish`; the plan: `req_plan`)

The caller makes these calls, for m = 1, .., L: `first(m)`; the sub-proof's part after moment 0; for
each further moment `later(m)` and the sub-proof's part after it; `close(m)`; and after the last
segment `finish()`. The process ends on any other order. In the table, P is a prover's message and V
the verifier's; the last column has the round's name in the plan.

| call | P: | V: | round |
|---|---|---|---|
| `first(m)`, if segment m has plain scalars | these scalars, elements of E | nothing | `REPLY` |
| `first(m)`, m = 1 | the roots for [ segment 1's messages of moment 0, U_1, what rides at the start ] | the seeds | `SEND seed` |
| | the answers | nothing | `SEND answer` |
| `first(m)`, m > 1 | the roots for [ G_m, the kept messages sent again, segment m's messages of moment 0, U_m ] | the seeds | `SEND seed` |
| | the answers | t positions in [n], and no element after them | `LOC` |
| | in the file only: the opening of group m - 1's trees | | |
| `later(m)` | the roots for [ segment m's messages of the next moment ] | the seeds | `SEND seed` |
| | the answers | nothing | `SEND answer` |
| `close(m)` | | theta | `BAT` |
| | FOLD's hiding sum | eps | `FOLD epsilon` |
| | k times: two elements | gamma_j | `FOLD round` |
| `finish()` | the roots for [ G_(L+1), U_(L+1) ] | the seeds | `SEND seed` |
| | the answers | t positions | `LOC` |
| | in the file only: the opening of group L's trees | | |
| | then the whole proximity test on the last group, beginning with its `BAT` (section 9) | | |

Where the plan gives a round proof of work (at the toy, of the frame's own rounds: `LOC`), the
round's nonce is in the file after the prover's message that the round closes.

So the positions of a group are sampled only after the oracle message that encodes its fold again
has been sent, has got its out-of-domain sample and has been answered. The next segment's first
oracle messages share that prover's message, its trees and its seeds. Whatever a sub-proof has sent
and not closed by a round of its own is in the query of the next round of the frame.

*The order inside a prover's message* is the order of the brackets above, and within a segment the
order of the shape's list. This is "the order of sending", by which oracle messages are numbered.

*Seeds and answers* (part 1; R6.9). A round `SEND seed` samples s seeds, each a point z in E^a and
then a factor beta that is not 0. All oracle messages of the prover's message that get an
out-of-domain sample share them: that is, all in the brackets but the fresh mask. a is the largest
**cube** among them. The cube of a message with one table is that table's number of index bits: log
l for P_m and P'_m, 1 + log l for G_m. With both tables it is the greater of the two numbers plus 1:
max(1 + log l, log l_zk) + 1 for B_1, max(log l, log l_zk) + 1 for the other B_m. A message reads
the first coordinates of z, as many as its cube has, and the coordinates beyond enter its sample row
in no way. (The plan's numbers cannot tell this rule from others. The last step's linear check can:
the second verifier, changed so as to put a factor 1 - z_u for each coordinate z_u beyond the cube,
or to read the last coordinates, rejected an honest proof.) The answers are ordered first by oracle
message, as sent, and then by the number of the answer.

*Trees* (`lay_trees` in `blocks/iop/context.h`; R7.12; the frame uses this rule and does not define
it). Within one prover's message, the oracle messages that have the same leaf class, and main rows
of the same length, share a tree; a message of mask rows alone joins the first tree of its class. At
the toy every class is 0, so each of the frame's four prover's messages has one root: trees 1 to 4,
tree m holding group m's members and U_m. What rides at the start has a class of its own, and so a
tree of its own in the first prover's message. A segment with later moments has one or more further
trees for each. In the proof the roots of a prover's message come by increasing tree number, which
need not be the order of the brackets. A group's trees are opened by increasing tree number, and a
shorter tree of mask rows alone is opened at the short positions (part 4, 5.1).

*Plain scalars.* Something public that the prover can name only late is not in the chain's first
query. It is sent as a plain scalar, unmasked, in a round of its own in which the verifier says
nothing, right before the first prover's message of the segment whose rows depend on it. So it is
bound before the seeds of that message's out-of-domain samples and before the positions of the
segment before. The frame reads it (an element of E in canonical form, or the proof is rejected),
keeps it for that segment's sub-verifier alone (`ReqFrame::plain`) and tests nothing else. In a
whole proof file the one use is the next token, in the round `REPLY`: one scalar of the last
segment. The verifier's side passes the next token on as received, and the caller compares it with
the next token that the client was given (`check_proof` in `blocks/frame/check.h`, and nowhere
else); testing the form of a next token is the sub-verifier's task.

*What rides at the start* gets its out-of-domain sample with the first prover's message (it counts
for a, and its answers come last). After that the frame leaves it alone: it joins no group of the
frame, no list of the frame opens its tree, and an acceptance by the frame says nothing about it.
The caller gets its number (`ReqFrame::at_start`) and must see it tested. In a whole proof file it
is the request's index-mask message, whose entries mask the request's rows on the committed table
(part 8). A choice of the programs: its entries are given out by a ledger, one by one and each only
once. What the ledger does not catch, a prover's code that adds one entry to two scalars among it,
is listed under "Not caught" in `blocks/iop/mask.h`.

## 5. Where the toy's sub-proof reads, and the three rows that it leaves (`segments_places`, `segments_rows`)

After `first(m)` the toy's verifier samples nu = log l - 1 elements in a round `ZERO`, takes the
masks that PSC needs from the ledger of B_m's mask row (entries 0, 1, .. in order) and runs PSC
(part 2). The three vectors are here. x is in B_1's row 0 for segment 1; for m > 1 it is in row 0 of
P'_(m-1), the second sending, which is in segment m's own tree and group, never in the old oracle
message. b is in B_1's row 1, or in B_m's row 0 for m > 1. c is in P_m's row 0. The value at a point
pt in E^nu of the vector in row i of a table with rb row bits is the form with the coefficient
eq(pt, j) at the index i + 2^rb j for j < l / 2, and 0 at the places from l / 2 on. PSC's output
becomes three open rows of group m, in this order: `hiding polynomial and c` (PSC's mask terms, and
PSC's coefficient times c at PSC's point); `x at the point`; `b at the point` (each with its own
mask terms). Points, coefficients, mask terms and targets are PSC's.

## 6. `close(m)`: all rows of the group into one claim, then FOLD

(a) *The order of the rows* (`Statement::begin_run`) has three parts: the open rows that are neither
sample rows nor generator rows, as they were created; then the sample rows, member by member in the
order of sending, kept members too, s for each; then the generator rows, as they were created. From
this call on the group takes no row and no member, and its members' ledgers give out no mask. For a
segment's group that gives:

    the carried claim | the old sample rows of the kept messages | the sub-proof's rows | the new sample rows | the generator rows, by the draws in order

where the first, second and last part come from the switch into the group (section 7 (c)) and are
missing in group 1. The number of rows is

    (the sub-proof's rows) + s * (members) + [ for m > 1:  1 + s * (kept members of group m - 1) + t * (3 + kept rows of group m - 1) ],

and the process ends if it is not the number from which the round's planned error was computed. At
the toy, numbered from 0:

- group 1, 5 rows: 0, 1, 2 the sub-proof's; 3 the sample row of B_1; 4 that of P_1.
- groups 2 and 3, 9 + 4 t rows each: 0 the carried claim; 1 the old sample row of P_(m-1); 2, 3, 4
  the sub-proof's; 5, 6, 7, 8 the new sample rows of G_m, P'_(m-1), B_m, P_m; then for each of the
  draws of the round in `first(m)`, in the order sampled and with repeats counted, four rows: main,
  kept row, old mask, fresh mask.
- group 4, 2 + 3 t rows, batched by the last test's own BAT: 0 the carried claim; 1 the sample row
  of G_4; then three rows for each of the draws, because group 3 has no kept row.

A choice of the programs: the sub-proof's rows come after the old sample rows and before the new
sample rows.

(b) *The elements.* theta in E, then FOLD's hiding sum (U_m itself was sent in `first(m)`), then eps
in E, then FOLD's k rounds with gamma = (gamma_0, .., gamma_(k-1)). theta and eps may be 0.

(c) *The group row* (`Bat::combine`, `add_to_group_row`) is the sum, over the rows in the order of
(a), of theta^e times row number e, coefficients and targets alike, with each row's coefficients
carried to the group's indices by section 2. Coefficients on the main table of a kept member do not
enter the vector c on the group's main table: they are gathered in a form of their own, K_i, on that
member's own table. So the claim is

    <c, main table> + sum_i <K_i, table of kept member i> + <d, mask table> = tau.

At the toy's groups 1 and 2 there is one such form, on P_m: theta^e times the part of
`hiding polynomial and c` on c_m, plus theta^e' times P_m's sample row, with (e, e') = (0, 4) in
group 1 and (2, 8) in group 2.

(d) *What FOLD leaves* (`fold_output`; the rounds are part 3's). Let M_i, rho_i be message and tail
of main row i of the group, W_i, sigma_i those of its mask row i, Y_i, pi_i those of row i of U_m,
and gamma' = (gamma_0, .., gamma_(h-1)). The folded rows, which are never sent as such, are M* =
sum_{i<2^k} eq(gamma, i) M_i in E^l with the tail rho* (the same sum of the rho_i), W* = sum_{i<2^h}
eq(gamma', i) W_i in E^(l_zk) with the tail sigma*, and Y* likewise with pi*. The verifier keeps
gamma, the layout and one claim:

    <c*, M*> + sum_i <K*_i, table of kept member i> + <d_0, W*> + <d_1, Y*> = tau*,   where
    c*[j] = eps * sum_{i<2^k} eq(gamma, i) c[i + 2^k j],   j < l;
    K*_i = eps * prod_{j<k} (1 - gamma_j) * K_i,   on the kept member's table, unfolded;
    d_0[j] = eps * prod_{h<=u<k} gamma_u * sum_{i<2^h} eq(gamma', i) d[i + 2^h j],   j < l_zk;
    d_1 = FOLD's public weights on the folded fresh mask (fold_mask_weights, part 3);
    tau* = the value of FOLD's last round polynomial at gamma_(k-1).

Nothing is checked on the spot but the form of what is received.

## 7. `first(m + 1)`, or `finish()`: the switch without a batching element, from group m into group m + 1 (`cs_sigma_verify`, `switch_rows`)

Let w be the number of kept rows of group m, over all its kept members.

(a) *What G = G_(m+1) holds, if the prover is honest.* The verifier never sees it; it fixes what the
rows below speak of. G's table has two rows, so an index is row + 2 * place.

- Row 0 is the folded main message: G[2 j] = M*[j], j < l.
- Row 1 holds "the rest" R, entry i at place i: G[1 + 2 i] = R[i]. After R comes free room, which is
  zero, and the last s places of row 1 are the pads of G's own out-of-domain sample.
- R = W* | Y* | rho* | kappa | sigma* | pi* | the old pads, of the lengths l_zk, l_zk, r, w r, r_zk,
  r_zk, and, for each kept member, s if it is over E and 3 s if it is over F. kappa is, for each
  kept row, by member and row, the tail with which that row was encoded in group m's tree. The old
  pads are, for each kept member, the entries that were at its pad's places in group m's tree, in
  the order of section 3.
- The offsets in R: A0 = 0 (W*), A1 = l_zk (Y*), T = 2 l_zk (rho*), TK_q = T + r (1 + q) for kept
  row number q < w, T0 = T + r (1 + w) (sigma*), T1 = T0 + r_zk (pi*), OP_i = T1 + r_zk + the
  lengths of the old pads of the kept members before member i.
- The shape must have |R| + s <= l (`req_fault`).

No row but G's sample row reads the free room, so nothing forces a malicious prover to leave it
zero; what it puts there moves its own answer and nothing else. A choice of the programs: where
kappa and the old pads are among the rest; and zeros in the free room, whereas in the stages of a
proximity test the surplus entries are uniform (part 4). That order is written down in this section
and in part 4 (3.2, 3.7) alone: R12.17 does not define what the carrying message holds.

(b) *What a kept message sent again holds:* the old table at every place but the pad's places, where
a fresh pad is. A choice of the programs: a kept message is always sent again as a message of its
own, with an out-of-domain sample of its own, at every switch that it outlives, the last one too. At
its last switch its rows could become rows of the carrying message instead; compared with that, the
programs' form costs one answer and one pad more for each kept message at its last switch.

(c) *The rows that the verifier makes* on group m + 1. It reads group m's columns at the list and
folds them (`single_rows_symbols`). At each listed position x the opened main symbols of the
ordinary members are placed on the 2^k rows of section 2, rows that nobody sent being 0, and folded
at gamma: that is y_main(x). The mask symbols at x' are placed on the 2^h rows and folded at gamma':
that is y_0(x). The 2^h symbols of U_m at x', folded at gamma', are y_1(x). The symbol v_q(x) of
kept row q is taken as it is in the column.

A form Q on the table of the old kept member i is **carried over** thus (`switch_add_kept`): its
coefficients at all places but the pad's go to the same places of that member's new sending; its
coefficient at the u-th place of the old pad goes to G[1 + 2 (OP_i + u)]. In words: Q is read on the
new message with the old pad put back.

- *The carried claim:*  sum_j c*[j] G[2 j] + sum_i carried-over(K*_i) + sum_j d_0[j] G[1 + 2 (A0 +
  j)] + sum_j d_1[j] G[1 + 2 (A1 + j)] = tau*.
- *The old sample rows,* right after it: for each kept member i of group m in the order of sending,
  and each answer:  carried-over(the left side of that sample row) = the answer given then. With
  (z_old, beta_old) the seed of the out-of-domain sample that the member of group m got, and
  alpha_old its answer, at the toy this reads

      sum over the places j < l - 3 of eq(z_old's first log l coordinates, j) P'_m[j] + sum_{u<3} beta_old X^u G[1 + 2 (OP + u)] = alpha_old.

- *The generator rows,* for each x of the draws, in the order sampled; a position that is sampled
  twice is opened once and gives its rows twice:

      main:        sum_{j<l} Gm_C(x)[j] G[2 j] + sum_{j<r} Gt_C(x)[j] G[1 + 2 (T + j)] = y_main(x);
      kept row q:  carried-over(the form with Gm_C(x)[j] at place j of that row) + sum_{j<r} Gt_C(x)[j] G[1 + 2 (TK_q + j)] = v_q(x);
      old mask:    sum_{j<l_zk} Gm_Z(x')[j] G[1 + 2 (A0 + j)] + sum_{j<r_zk} Gt_Z(x')[j] G[1 + 2 (T0 + j)] = y_0(x);
      fresh mask:  the same with A1, T1 and y_1(x).

  A kept row's generator row is there to tie the second sending to the old tree: "the new message
  with the old pad put back, encoded under C with the old tail, has at x the symbol that the old
  tree shows".
- The sample rows of G and of the messages sent again are SEND's, each on its own table with its own
  pad.

G and the messages sent again join group m + 1, and the rows wait there as open rows. Nothing is
batched here: that happens at `close(m + 1)`, or in the last test's BAT.

*Why the old sample row is a row of its own.* It is there twice: it was batched with group m's other
rows, went into K_i and travels inside the carried claim; and it is the open row above. A shorter form would have the first alone, so that a switch left one row fewer for each answer of a
kept message. No attack on that shorter form is known,
and no proof of its soundness is written (below). `switch_rows` in `blocks/iop/cs.h` makes the row. `cs_test` asks that a switch leaves
one claim, the kept messages' old sample rows and the generator rows. It also changes, in the
prover's tables, the message that was sent again by a vector that the three other kinds of row do
not see, and asks that this row alone is then false. `links_check.py` makes the row by itself among
the rows that it batches. No test of the repository shows a false proof that this row alone stops:
by a comment in `blocks/iop/tests/cs_test.cpp`, the prover for which the row is there is a lucky
one, which no run can produce. (from the programs' source)

What follows is an argument and not a proof; the comment
before `switch_rows` in `blocks/iop/cs.h` gives the programs' own wording of it. Take the shorter
form, without the separate row. The argument for it looks inside a soundness proof of FOLD, in three
steps. First, the candidates for the group's word, kept rows among its rows, are fixed before theta
is sampled. Second, a candidate that misses the old sample row makes the batched claim false but for
a chance of (rows - 1) / |E|, summed over the list. Third, every code word near the folded word is
the fold of such a candidate. The third step is not part of what a statement on FOLD, used as a
black box, gives: such a statement yields some member of the list that meets the batched row, and
not necessarily the one that folds to what comes out. So nothing that uses FOLD as a black box ties
the new message to the one message that the old word stands for: that of the code word near the old
word which agrees with the old answer. The separate row is there to do that. It costs no byte of a
proof and no round.

(d) *Checked on the spot* in a switch. All of it is in routines of other parts, and each failure
rejects:

- every answer is an element of E in canonical form;
- where the plan gives a round proof-of-work bits (at the toy, of the frame's own rounds: `LOC`),
  the nonce is read and the proof-of-work test must hold before anything is sampled;
- positions are derived by the verifier, never read;
- the opening has exactly the columns, salts and digests that the list and the trees' shapes fix,
  every opened symbol is in canonical form, the opening leads to the root received earlier, and no
  vertex gets two values under one root (R7.14).

No equation is checked.

## 8. Which list opens what

The list of the round in `first(m + 1)`, or in `finish()` for m = L, has the range [n] and opens all
trees of group m, each whole, once: those of all its moments. The oracle messages asked for are the
group's members in the order of sending and then U_m (`single_rows_oracles`); the process ends if a
tree holds a part that is not among them. No tree of the frame is opened at a second list. At the
toy, list m opens tree m. Tree 1 holds x_1, b_1 | c_1 | B_1's mask row | U_1. Tree m = 2, 3 holds
G_m's two rows over E | c_(m-1) sent again | b_m | c_m | B_m's mask row | U_m. Tree 4 holds G_4 |
U_4 and is opened at the first list of the last proximity test. Where columns, the copies of mask
symbols and the salt are in a leaf, and the order of an opening's digests, are defined in section 6
of the format page (R7.1 to R7.9); the widths at the toy are in R12.18.

## 9. `finish()`: the last proximity test

After the switch out of group L the last group holds G_(L+1) alone, with 1 + s + 3 t open and sample
rows: the carried claim, G's sample rows, three generator rows for each of the draws. On it runs the
whole proximity test of part 4 (`iopp_verify`): BAT, then FOLD and the code switch with a batching
element stage by stage, then BASE. Its first stage is (C, Z, 1, 0), which is why the shape's bytes
leave that stage out; its first fold uses U_(L+1), which was sent with G_(L+1). The word of the
choices says whether the masks of the later stages ride with the switch that makes their stage; at
the toy they do. The group's mask table is the one zero row that nobody sent. So every row has the
coefficient 0 on it, d_0 = 0, and the folded symbol y_0(x) is 0 at every position. The switch out of
this group still makes its "old mask" generator rows, and each of them has the target 0.

A choice of the programs: a segment's closing passes its carrying message on to the segment that
follows, as a member of that segment's group. For the last segment there is no next one: its
carrying message forms a group by itself, on which the request's one full proximity test runs.

## 10. The verifier's acceptance, and the values on the way

An acceptance by the frame (`req_verify`, or the calls of section 4 made one by one, as in the toy)
says that nothing was rejected up to and including BASE's checks. It says nothing of bytes that are
left over, of what rides at the start, or of a group that a sub-proof has made itself: these are the
caller's.

The toy's verifier (`segments_verify_file`) takes the public file's text and the proof and nothing
else. It sizes and plans the run again from the text and rejects a text that is not in its one
encoding. Its whole verifier's side then runs inside one routine that gives it nothing else
(`run_verifier` in `blocks/chain/closed.h`). That routine rejects if the planned round errors miss
the target (the errors are the planner's figures: part 1, section 6). It writes the first query
(label 2: R12.15; the items `configuration`, `codes`, `the chain of links` and an empty statement:
sections 0 and 1 at the head of this page). And it gives the deciding routine the channel and the
handles of these items alone, so that, but for the exceptions in the next paragraph, whatever the
decision reads is an item of the first query, a received message, a sampled value or a constant of
the program.

Why, as the head of `blocks/chain/closed.h` gives it: a verifier whose caller gives it, beside the
bound objects, a plain value of the caller's own can accept a false claim. On test builds, several times a toy run's verifier accepted a false claim because its test
gave it, beside the bound objects, a plain value of the test's own: a list of positions, a count, a
shape (toy size, in the programs' own tests). For a verifier's side that runs inside
`run_verifier`, the routine's form closes it: the deciding routine is a plain function that gets the
channel and the handles, not the inputs. No test in the repository is known to show this by itself. (from the programs' source: the same head and the comment before `run_verifier`) That
comment names two things that the routine does not close, and that only reading can find: a plain
copy of an input among the handles, and a field that a type's `public_bytes` leaves out. It also
excepts global state, what the caller's list of notes already holds and the plan: the deciding
routine can read all three. The channel gives it the caller's whole plan, and of a plan the first
query holds each round's name, counts and proof-of-work bits, but neither the target in bits nor any
error bound.

Every channel belongs to a session (`blocks/chain/session.h`), and the verdict is read from the
session: no chain has rejected, and every chain that was started was brought to its end, where a
byte left over rejects. At the end the process ends if on some group no proximity test was begun, or
if an oracle message that got an out-of-domain sample is in no group
(`Statement::require_all_tested`): that would be a mistake of the program, not of a proof. By the
comment before that routine in `blocks/iop/relation.h` it does not see a row that was never added to
a group, an oracle message that never got an out-of-domain sample, or whether a test that was begun
was carried through.

Because no equation is checked before BASE, an accepted file shows only that all blocks agree. So
the C++ verifier can note values on the way, which no decision reads, for anyone who compares
another verifier with it. They are: at each close the group row's target tau and FOLD's target tau*;
at each switch y_main, y_0 and y_1 at the first of the draws; and the last test's own (part 4). At
the toy that makes 34 values.

## 11. The bytes of the item `the chain of links` (`public_bytes` of `ReqShape`; R12.12 to R12.14)

Every word has 8 bytes, the lowest first (R1.1); a list is its length as a word and then its entries
(R1.7). Within the shape what follows from other words is not written, and every public fact is
written once, with one exception. The notice of a message that rides at the start repeats s as its
sixth word (the shape's test compares the two); the toy has no such message.

    the shape  =  C's number | Z's number | LIST of segments | LIST of what rides at the start, each a notice of seven words (R12.3) | s | LIST of the last test's stages from the second on | the word of the choices (R12.5)
    a segment  =  the number of its plain scalars | LIST of its messages       (a segment's name is for prints and is not written)
    a message  =  six words: main rows | 1 if they are over F or if there are none, 3 if over E | mask rows | leaf class | moment | life
    a stage    =  four words: base code | mask code | k | h (R12.4)

Not written, because they are the shape's: a message's codes and its number of answers. Not written,
because it follows: the last test's first stage, (C, Z, 1, 0). The C++ structs hold these facts
again for their readers' ease, and `public_fault` compares each copy with the word that it repeats.
One fact is twice in a first query: s is also the plan's sixth word (R12.14); the frame's
constructor compares the two.

At the toy's full size the item has 57 words, 456 bytes (read off a file's parameters):

    0 | 0 | 3 | [0 | 2 | (2 1 1 0 0 0) (1 1 0 0 0 1)] [0 | 2 | (1 1 1 0 0 0) (1 1 0 0 0 1)] [0 | 2 | (1 1 1 0 0 0) (1 1 0 0 0 0)] | 0 | 1 | 2 | (1 1 1 1) (1 2 0 0) | 1

At the tiny size the last stage is (2 2 0 0), and all else is the same.

## 12. From a sub-proof's arrays to a row (class `ReqRows`), and the layout. The toy does not use this section

A real sub-proof works with **arrays**: named matrices of committed entries. A public **layout**
(`ReqLayout`) names, for each segment, its arrays; for each array the oracle message of its segment
that holds it, its numbers of rows and columns, and the aligned **blocks** into which it is cut. A
block holds the elements (i0 + i, c0 + c) with i < 2^bi and c < 2^bc. Element number x = c 2^bi + i
(if the block's last word is 1) or i 2^bc + c (if it is 0) is in row row0 + (x div 2^bp) of the
oracle message's own table at place place0 + (x mod 2^bp). The block takes 2^br rows, br + bp = bi +
bc, and i0, c0, row0, place0 are multiples of their sides.

A segment's sub-proof ends in one equation (`ReqEquation`): terms, each of which names an array, by
the segment that sent it and its name, with weights of rank one, scale * on_rows[i] * on_columns[c]
on element (i, c); coefficients on the first entries of the segment's mask row; and a public target.
Each of the two weights is either of product form, with one factor for each bit of its index, or a
written-out vector; at most one of the two is written out. `ReqRows::row` makes the row, by one
routine for both sides. The row has each term's weights at the table entries where the layout puts
the elements, and 0 elsewhere. These entries are of the oracle message where that array is in the
present segment (`ReqFrame::where`): the message itself, or, for an array of an earlier segment, its
latest sending; the process ends if the present segment cannot read it. The coefficients on masks go
to the entries 0, 1, .. of the mask row of the segment's first message that has mask rows, which
must have exactly one. These entries are taken from the ledger at this call, all at once, and the
process ends if any was taken before. The caller adds the row to the segment's group. That weights,
coefficients and target are functions of the first query, of what was received and of sampled values
is the caller's duty.

*What a layout must meet* (`public_fault` of `ReqLayout`; with shape and codes: `req_layout_fault`).
The constructor of `ReqRows` asks `req_layout_fault` and ends the process on a fault, so that no
caller can forget it. The conditions are: as many segments as the shape; every array has a name, at
least one element, fewer than 2^48 rows, fewer than 2^48 columns and at least one block; names that
differ within a segment; every array is in an oracle message that its segment has and that has main
rows; in every block br + bp = bi + bc, each of bi, bc, br, bp is below 48 and each of i0, c0, row0,
place0 is below 2^48; every block aligned, inside its array and inside real rows of its table; every
element of an array in exactly one block; no entry of a table under two elements; and no block over
a pad's place.

*Its bytes:* a list of segments; a segment is a list of arrays; an array is its name (a text), the
number of its oracle message in its segment, its rows, its columns, and the list of its blocks; a
block is nine words: i0, c0, row0, place0, bi, bc, br, bp, and 1 or 0 as above. R14.5 defines this
item, `layout`.

*The layout file* (`layout_read` and `req_frame` in `blocks/frame/req_frame.h`) is a text from which
both sides fill the shape's segments and the layout; the comment before `LayoutGroup` in that file
says how it is encoded. A segment's one mask row is not in the layout file: `req_frame` gives it to
the segment, in the first oracle message of its first moment, which therefore cannot be kept. A
shape and a layout that come out of it must also pass the tests of section 1 and of this section.
How a prover reads its arrays out of its own memory (`blocks/iop/req_arrays.h`) is not part of the
protocol.

A choice of the programs: the frame places arrays into rows and knows nothing of what a cell means.
Packing several cells into one entry, and the message of partial evaluations that goes with it, are
the sub-proofs' business: the verifier's side of this block has nothing of them. (On the prover's
side the routine that reads arrays out of the prover's memory can read three signed bytes b0, b1, b2
as the one entry b0 + 2^e b1 + 2^(2 e) b2, e a public number of bits.)

## 13. The numbers of the toy's two sizes

Full size: C = (2^15, 4096, 73), Z = (2^15, 32, 73), t = 73 in the three rounds `LOC`, which carry
20 proof-of-work bits; 76 rounds. Rows batched at the four `BAT`: 5, 301, 301, 221, that is 5, 9 + 4
t, 9 + 4 t, 2 + 3 t. Elements sampled in the frame's four seeds: 15, 14, 14, 14, that is a + 1 with
a = 14, 13, 13, 13. Answers: 2, 4, 4, 1. One root in each of the four prover's messages. The offsets
of 7 (a) in groups 1 and 2, which have one kept row over F: T, TK_0, T0, T1, OP, |R| = 64, 137, 210,
283, 356, 359; in group 3, which keeps nothing: T, T0, T1, |R| = 64, 137, 210, 283.

Tiny size, which reaches no security level: C = (2^9, 64, 3), Z = (2^9, 16, 3), t = 3, 6
proof-of-work bits, 58 rounds; rows 5, 21, 21, 11; seeds of 9, 8, 8, 8 elements; the same offsets:
32, 35, 38, 41, 44, 47, and 32, 35, 38, 41.

## 14. Limits of this part

- *What the second verifier covers.* `links_check.py`, a second verifier in Python, calls none of
  the programs' code. A script of the programs' tests, which also builds malicious provers, makes
  files and compares it with the C++ verifier on them. The files are: honest proofs at the full
  size, at the tiny size, and at the tiny size under a public number for which a position is sampled
  twice in one of the first two lists; altered copies of each, rejected at the same check; and the
  false proofs of malicious provers, rejected where the list of false proofs says. That script is not part of this repository:
  `run_honest.sh` does the same on the honest proofs and the altered copies alone, and builds no
  malicious prover (from the programs' source; `tests/new/second/README.md`). The files' shapes are:
  field 2; three segments; one moment in each segment; in each of segments 1 and 2 one kept message
  of one row over F with life 1, so one kept message at a time; k = 1, 2, 3, 1; one mask row in a
  segment's group (h = 0) and none in the last group; mask codes as long as the main codes; one
  answer to an out-of-domain sample.
- *How independent the second verifier is.* It is not independent of the C++ programs on the toy's file
  `blocks/toy/segments.h`, on some 25 lines of the frame's file `blocks/iop/req.h`, and on all that
  `toy_check.py`, whose channel it uses, is not independent of: on these it can share a misreading
  with them. By its head, whether it is independent of any other file of the C++ programs, or of the
  rest of a file that is named only in part (such as `blocks/iop/req.h`), has not been examined:
  nothing is known either way.
- *What `links_check.py` does not test:* later moments, a life above 1, several kept messages, kept
  rows over E or in tables of several rows, plain scalars, what rides at the start, all of section
  12, field 1. The programs' own C++ tests, in which prover and verifier share their code, build
  shapes with these, field 1 excepted (`blocks/Makefile` has the one setting, field 2).
- *What another second verifier adds.*
  `tests/new/second/proof_check.py` checks the proof files of small synthetic batches, which the
  same script makes; their first chain runs this frame. Its files have one plain scalar in the last
  segment and one message that rides at the start. In one of its three runs the second segment has a
  message in a leaf class of its own and a message at moment 1, each of one row over F, of which the
  sub-proof claims nothing. It reads the item `layout` to its end and tests of it only that it has
  one list of arrays for each segment. It has one answer, mask codes as long as the main codes, h =
  0, lives of at most 1 and one kept message at a time. Its formulas of the frame are those of
  `links_check.py`. What it has beyond them is less independent: it is not independent of the C++
  sources that its head lists, parts of `blocks/iop/req.h` among them, and on these it can share a
  misreading with the C++ programs. It takes the
  parameters' bytes as given.
- *What no second verifier has tested:* a life above 1, several kept messages, kept rows over E or
  in tables of several rows, the rows of section 12, field 1 (the programs of this repository can be
  built for field 2 alone).
- *For what no test at all is known,* not even among the programs' own: h above 0 in a segment's
  group, a kept message in a leaf class other than 0. A segment's message with a leaf class other
  than 0 is known in one test alone, the run of `proof_check.py` named above, and that message is
  not kept. A mask code shorter than the main code in a segment's group is known in the programs'
  own test alone, and in no second verifier: `blocks/toy/tests/segments_test.cpp` sizes `links` with
  a main code of length 2^13 and a mask code of length 2^9, at a target of 32 bits a round and not
  128, and asks that the honest proof is accepted and that proofs with a flipped bit (one bit in
  every 61st byte, unless the call says otherwise), proofs cut at evenly spaced lengths and a proof
  with a byte added are rejected (from the programs' source). Several answers are
  known in the same way, in a test of the programs alone: `blocks/tests/other_parameters_test.cpp`
  sizes `links` with the same two code lengths, at the same target of 32 bits, and with two answers
  to an out-of-domain sample. It asks that the honest proof is accepted, and that this proof and the
  proof with one answer are each rejected by the other's verifier (from the programs' source).
- *Everything that `links_check.py` does not test is written from the code alone.* Of these, the
  formulas most likely to hide a slip of this page are the offsets TK_q and OP_i with several kept
  members, the pads at s > 1, and "the seed of the out-of-domain sample that the member of group m
  got" at a life above 1.
- *"The process ends on nothing that depends on a proof's bytes"* is the programs' rule. It was
  followed, by reading, for the frame's four calls, SEND, the switch, the channel and the openings,
  where it holds; not for PSC, BASE and the check of a tree's walk.
- *How kept rows are arranged in a group* (section 2) is in the code of FOLD and of the class
  `Statement`: rows widened by copies, with k counting the ordinary rows alone. A choice of the
  programs: kept rows have no places among the 2^k rows. In the other arrangement, in which they
  have such places, each closing with a kept row would have one fold round more at the toy (k = 2, 3
  instead of 1, 2); in general that depends on how the numbers of rows round up. Not known: whether
  the analysis on which the planned errors rest treats kept rows in the programs' arrangement or in
  the other one. If it treats them in the other one, then what is analysed differs from what runs,
  in the rounds of FOLD on a group with kept rows.
- No proof is written that the shorter form without a separate old sample row would be sound
  (section 7). That the separate row ties the two sendings is an argument and not a proof. The last
  group (section 9) and the order of a group's rows (section 6 (a)) are choices of the programs.
- *The errors of FOLD's rounds, of eps and of BASE's lambda, and what they rest on,* are in part 3,
  section 6. `links` is sized, at every size, under the rule "up to the Johnson radius less a
  slack". Under that rule the closeness counts are those of the theorem on codes (part 3, section 6; from the programs'
  source: `segments_size` in `blocks/toy/segments.h`; `blocks/numbers/codes.h`).
- *Stated here only as far as the frame uses it, and defined elsewhere:* the layout of a group and
  the order of its rows (`blocks/iop/relation.h`); trees, leaves and openings
  (`blocks/iop/context.h`, `blocks/tree/`; the format page defines them); SEND's form and pads (part
  1); which of the toy's rows has which coefficient, sign and mask term (PSC, part 2); FOLD's
  rounds, d_1 and the content of U_m (part 3); the channel, proof of work and the session
  (`blocks/chain/`).
- *Places where the process ends,* all on a mistake of the program or of public parameters and none
  on a proof's bytes: a pad's place that holds something before sending; a row that reads the pad of
  an out-of-domain sample; a switch whose two blocks of rows are not all the rows of the new
  message; calls of the frame out of order; a closing whose k, h or number of rows is not the
  planned one.
- A segment's group with mask rows in several messages, or mask rows in a later moment, is admitted
  by the shape's tests. Whether every such shape meets the rule on which tree holds a mask row's
  copies (R7.5) has not been examined.
