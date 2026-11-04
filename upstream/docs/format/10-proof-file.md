*Part of [the file format](FORMAT.md). Rule numbers and terms are those of that page.*

## 10. A proof, round by round

This section gives the first queries of a proof's three chains, the request's slot between them, and
every round of the three chains in the order of the file. The order of the file's parts and its head
are R14.1 and R14.2 (section 7). The types of R12.1 to R12.16 and the proximity test of R13.19 are
in section 9, the rounds of a segment's sub-proof in section 11, the item `request` and the meaning
of the next token in section 12 (R15.1, R15.7), and the tests that a verifier makes on the items in
section 13. Nothing in this section depends on the field, except the version word and the plan's
first word in each first query (R4.3) and the prime p.

A proof file goes to one client and is about one request of a batch. It is meant to show that no
value of the integer model's run on the request, by the weights in a stored table, leaves a table or
a range at any position, that the checked tokens meet the acceptance policy and, if e = T, that the
next token which the client was given is the model's top choice at the last position (12.2, 12.5),
as far as the sub-proofs of its segments check a model's steps. If e < T it says nothing of a next
token. The **frame** of a proof is everything in its three chains that is not a sub-proof. The frame
knows a sub-proof only by the oracle messages of its segment (R12.12, R12.13), by the rounds that
the plan has between the frame's own rounds, and by the rows that the sub-proof leaves.

A **row**, which section 9 calls a linear row (9.1, item 3), is a linear equation with a public
right side, its **target**, on the vectors that oracle messages commit to. In sections 9 to 11 and
in 13.9 "target" has this sense, a row's target; elsewhere the target is the number of bits of the
line `sec` (8.2). A main row, a mask row, the folded row, and a row of a table, a matrix or an array
are not rows in this sense: they are always named in full.

*Terms from other sections.* A segment, a request's slot, a leaf table, a key file, a leaf file and
the next token: section 1. An oracle message, its notice, main rows, mask rows and leaf class:
R12.3, R7.1, R7.12. A seed and the answers to an out-of-domain sample: R6.9, R8.3. A tree's number
and place: R7.13. A stage of a proximity test, the fresh mask of a fold and the masking word of the
last step: R12.4, R12.10. A message's moment and life: R12.12. A kept message: R12.16. The carrying
message's shape, and what is in a group's tree: R12.14, R12.16, R12.18. A sumcheck with a hiding
sum: R13.16. The stored table, its matrix, the numbers v, k_lo and k_col = k_lo + v, the folded row,
Omega, the shared split's mask message and the word on a mask sent ahead: R4.6, R13.13, R13.19. W, a
request's own vector: R14.4 (e). A seal: R6.8. The binding tree: R7.18.

### 10.1 Four things that are easy to get wrong

1. *Which round takes the answers to an out-of-domain sample.* These answers always begin the
   prover's next message (R8.3), but which round holds that message differs. After the first
   segment's first message, after every later moment of a segment, and after W, the folded row and
   Omega, it is a round `SEND answer`, in which nothing is sampled. After every later segment's
   first message, and after the last carrying message, there is no `SEND answer`: the answers are
   the message of `LOC`. Inside a proximity test there is none either: the answers are the message
   of `CS`. So the `LOC` of the shared chain, which comes after a `SEND answer`, has an empty
   message. In `CS` the positions are sampled first and the one element of E after them, from the
   same round's answer (R6.10).

2. *Openings follow the list, rows follow the draws* (R6.6). A position that is sampled twice is
   opened once and gives its rows twice. The number of rows that a round batches is a word of its
   entry (R4.3) and does not depend on the sampled positions. The order in which rows are batched is
   fixed: 10.6.

3. *A tree has a number and a place, and the two can differ.* The number and the run's value of the
   chain that sent the tree enter the digests of its leaves (R7.4, R7.13). Openings after a round
   come by increasing place (R8.7). In the first chain the request's index-mask message takes a tree
   number of its own (number 1, where the first segment's messages of moment 0 have one leaf class),
   so every later tree has a number that is greater by one than in a plain chain of segments (R14.4
   (a)). In the second chain Omega's tree takes a new place and keeps the number and the run's value
   that it had in the shared chain (R14.10). The shared chain has no random bytes in the file: the
   binding tree's root is used instead of them (R14.9).

4. *The request's slot is compared with nothing beyond its range* (R14.7). A wrong salt or sibling
   gives another root and so another first value of the shared chain, and then that chain's sampled
   values are others.

### 10.2 The first chain's first query

- **R14.3** *The first chain's first query.* The chain's label is 0x46524d01. The **caller** of the
  frame is the protocol that supplies the segments' sub-proofs (for a model: sections 11 and 12).
  After the version's item and the plan (R4.3) come these parameter items, in this order:
  1. `configuration` (R4.9);
  2. `codes` (R12.2);
  3. `the test on the table` (R13.13);
  4. `request's shape`, a chain of segments by R12.14;
  5. `layout` (R14.5);
  6. `links' identifiers`, a list of words, one for each segment (in this name, as in the parameter
     lines `group link` and `stages link`, a link is a segment);
  7. `pending rows`, one word, the number of rows on the table that the segments leave, at least 1
     (for a model: 13.4, 11.5);
  8. `binding tree` (R14.6);
  9. last of the parameters, any items that the caller's sub-proofs take as data, which this page
     does not define. The programs place no item there (13.2; from the programs' source).

  *In which order a proof's tests enter their stages into `codes`* (R12.2): the last proximity test
  of the first chain, whose first stage has the two codes of the segments' groups, so that these
  have the number 0 in their lists; the test on the folded row; the test on Omega, if it is made;
  the test on a request's own rows. The order is hashed with the item, and another order gives other
  challenges (from the programs' source).

  The statement has four items, in this order:
  1. `table`, by R4.6, for the leaf table on which the proof was made. `table` holds the stored answers of the leaf
     table's record, as they were read from its leaf file (R13.3). The library's routine for a proof
     file (`check_proof`) rejects, before it reads a byte of the file, a table that has passed no
     check: one whose key file or leaf file was rejected or was never checked. `cp_verify`, which
     checks the three files in three processes, calls `check_proof_if_tested`: there the table is
     read from its file without the proofs that follow it and is taken as if it had passed, so this
     test shows nothing, and the result is worth a yes only if the two other checks have passed on
     the same bytes, which that routine does not test (from the programs' source:
     `blocks/frame/check.h`; `blocks/chain/closed.h`, "Not checked"). How `cp_verify` ties the three
     together is in 13.6 and 13.7.
  2. `stored copy`, one word of 8 bytes, the leaf table's number, which is the head's number
     (R14.2).
  3. `request`, a string of bytes (R1.6): the request, whole, as the caller of the frame passes it
     in. What these bytes are is the caller's matter: in the programs they are 177 words, which hold a digest of the request
     file and not the file (R15.1, R15.5).
  4. `the request file's words`, a string of bytes (R1.6): the caller's request itself, byte for
     byte. In the programs: the request file, whole: its T + 2 words of four bytes, the low byte
     first (12.2). The item's length word is 4 (T + 2) + 8, the string's own 4 (T + 2). So every
     challenge of the chain hangs on the tokens and the range through H alone (R4.6, R5.2), whatever
     the digest in `request` is worth (on what that needs of a second, fixed hash function, built on BLAKE2b's compression function: R15.3). The caller
     makes items 3 and 4 from the one request file (R0.5). The library's routine takes both from its
     caller and does not compare them: a caller that checks a relation on tokens must make items 3
     and 4 from those same tokens; soundness for the tokens leans on item 4.

  The next token is not an item (R14.8).

  The version word (`H821`, R4.3) does not tell every way of making a proof apart. A proof file that
  was made without item 4, or with a key file's labels on its shared and second chains (R14.9,
  R14.10), has it too, and is rejected (13.8): the first like a proof of another request, in its
  first chain; the second in its shared chain. A commitment ID whose input lacks the seeds' list or
  the closing word 0 (R4.11) differs from the one that a verifier forms, and a call that names it is
  rejected (`model-statement`, 13.5).

  *What a proof's items must be.* The list of notices in `request's shape` (R12.14) has exactly one
  notice (R14.4 (a)); the first words of its segments are 0, .., 0, 1 (R14.8); `links' identifiers`
  and `layout` have as many entries as it has segments. A verifier rejects items that do not meet
  these conditions. No plan of a proof holds an entry `RECORD` (R13.15).

- **R14.5** *The item `layout`* is a list with one entry for each segment, each a list of arrays. An
  **array** is: its name, a text; the number of the segment's oracle message into whose main rows it
  is placed, from 0, in the order of R12.13's list; the number of its rows and the number of its
  columns; and the list of its blocks. The **message's table** is that message's main rows, one
  under the other; a position in it is the number of an entry in a row's message (from the programs'
  source). A **block** is nine words:
  1. the first row of the block in the array;
  2. the first column of the block in the array;
  3. the first row of its rectangle in the message's table;
  4. the first position of its rectangle in the message's table;
  5. b for the block's rows, where it has 2^b of them;
  6. b for the block's columns, in the same way;
  7. b for the rectangle's rows;
  8. b for the rectangle's positions;
  9. a word that is 1 if the array's row index is in the low bits.

  Write a, e and f for words 5, 6 and 8. The array's element in row (word 1 + i) and column (word
  2 + c), i < 2^a, c < 2^e, has the number x = c 2^a + i if word 9 is 1, and x = i 2^e + c if it
  is 0. It is in the message's table in row (word 3 + (x >> f)), at position (word 4 + (x mod 2^f)).
  Words 7 and 8 add up to a + e (from the programs' source).

  The order in both lists is that of the blocks' lines in the layout file (13.4, step 7). The arrays
  of a segment come in the order in which a block's line of the file first names each of them, and
  the blocks of an array come in the order of that array's blocks' lines in the file. Nothing sorts
  them, by number or by position. The order is hashed with the item, so a verifier that forms the
  item in another order has other challenges (from the programs' source: `blocks/frame/req_frame.h`;
  `blocks/iop/req_rows.h`).

  The conditions on the item and on an array: the item has as many entries as the request has
  segments; an array's name is not empty; no two arrays of a segment have the same name; an array's
  number of rows and its number of columns are each at least 1 and below 2^48; an array has at least
  one block; the number of an array's oracle message is below the number of the segment's oracle
  messages, and that message has main rows. A verifier that takes the item `layout` as an input
  tests these, and the conditions on a block, before it reads a proof (from the programs' source:
  `blocks/iop/req_rows.h`).

  The other conditions on a block are in 13.8, in the rows `plan-layout`: the four first numbers are
  multiples of the block's sides; a block is inside its array; the blocks of an array do not overlap
  and hold each element once; no two blocks of one oracle message share an entry of the message's
  table; a block is in rows that are sent; no block covers a pad.

- **R14.6** *The item `binding tree`* is two words: the depth of the batch's binding tree, at most
  48, and the number of bytes of a leaf's salt. The tree is R7.18's. The number of bytes of a salt
  is also the third of the plan's eight words (R4.3); the salt in the slot (R14.7) has the length
  that the item's word gives.

### 10.3 The first chain's rounds

The chain begins with its 32 random bytes (R14.1, case (a) of R4.5). A request has L **segments**,
each a part of the computation with a sub-proof of its own. Each segment has a **group**: the oracle
messages that are tested together at one list of positions (a group of layers is another thing:
section 11). A segment's oracle messages go out at one or more **moments** (R12.12). A segment is
**closed** in two steps: all rows on its group are batched into one row (`BAT`), and the group's
main rows and mask rows are folded (`FOLD epsilon`, `FOLD round`). What the fold leaves is committed
to again in the **carrying message** (two main rows over E and no mask row: R12.14, R12.16, R12.18),
which goes out with the next segment's first prover's message. Only then are the positions of the
closed segment's group sampled. In a model's proof every segment has three moments. The segments are, in this order: the layers of the first group of
layers, the attention of that group, the layers of the second group, its attention, and so on, and
last the output head, whose segment has the next token (from the programs' source; section 11).

- **R14.4** *The first chain's rounds.* They are those of a chain of segments (R12.16, R12.17), with
  the changes of (a) to (e). In which order a round batches the rows of a group is in 10.6. *The
  frame's own rounds of a segment* are: at its beginning `SEND seed` with `SEND answer` (the first
  segment) or with `LOC` (a later segment); at its end `BAT`, `FOLD epsilon` and k rounds
  `FOLD round`, k being the group's number of R12.16. R12.17's rounds from `ZERO` to `PSC values`
  are the sub-proof of the run that R12.17 describes, and are not part of a proof's frame.
  - (a) *What goes out with the very first prover's message.* R12.14's list of notices holds one
    notice: the request's **index-mask message**. It has mask rows alone, under the mask code of a
    request's own group (R13.13), s answers to its out-of-domain sample, and as its leaf class the
    least number above the class of every message of every segment (the class is the last word of
    the notice, R12.3, and the notice is hashed, so the number itself matters; prover and verifier
    each form that number from the segments' classes, and no proof file holds it:
    `blocks/frame/files.h`; the library's test on a `request's shape` asks less, only that the class
    is above these classes: `blocks/iop/req.h`; these two clauses: from the programs' source). The
    message goes out in the first segment's first prover's message, after that segment's own
    messages and the fresh mask of its fold, and gets its out-of-domain sample with them: its s
    answers are the last in `SEND answer`. Its class gives it a tree of its own, the last of that
    prover's message: where the segment's messages have one class, that `SEND seed` holds two roots,
    the segment's first, and the index-mask message's tree has the number 1. It joins no segment's
    group, and no round of the first chain opens its tree. So every later tree of R12.16 and R12.17
    has a number that is greater by one. Segment m, for m above 1, has tree m. The last group has
    tree L + 1. The last test's later trees go on from L + 2. W's tree takes the number after them.
    Each `LOC` opens the trees of the group before it; R12.17's sentence on the (i + 1)-th round
    that samples positions does not hold in a proof.
  - (b) The next token's round, `REPLY` (R14.8).
  - (c) Between the frame's rounds of a segment are the rounds of its sub-proof, as the plan has
    them.
  - (d) *Later moments and further classes.* For each moment j = 1, 2, .. of a segment (R12.12) the
    frame has two rounds amid the sub-proof's, where the plan has them: `SEND seed`, whose message
    is the roots of the trees that R7.12 makes from the segment's messages of that moment alone, and
    `SEND answer`, s answers to the out-of-domain sample of each of these messages. The trees take
    the next numbers. A group's trees are all that hold its carrying message, a kept message that is
    sent again, a message of the segment, or the fresh mask of its fold; the `LOC` after the
    segment's closing opens them all, by increasing place (R13.12). The carrying message and every
    fresh mask have the leaf class 0; a kept message that is sent again keeps its class. With such
    segments the numbers of (a) move on accordingly. In the programs every message of a segment has
    the leaf class 0 too, so each moment of a segment has one tree, and the index-mask message has
    the class 1. With the three moments that every segment has there (10.3), the numbers are:
    segment 1 has the trees 0, 2 and 3, and the index-mask message's tree is 1; segment m, for m
    above 1, has the trees 3 m - 2, 3 m - 1 and 3 m; the last group has tree 3 L + 1; the last
    test's later trees go on from 3 L + 2 (by R7.12 and R7.13). There the first message of a
    segment's moment 0 has one mask row, the segment's mask row (11.1), and the segment's other
    messages have none (from the programs' source).
  - (e) *The last proximity test* is R13.19's. Its first stage is the last group (R12.16); its later
    stages are the list in `request's shape` (R12.14). Its first fold's mask goes out with the last
    carrying message, in that message's tree. The masks of its later stages go out with the switch
    that makes their stage if the last word of `request's shape` is 1 (R12.5; section 9 then says
    that masks ride). The programs always write 1 for that word (from the programs' source). *After
    the last proximity test* come these rounds, directly or after what the caller's sub-proofs send
    there, as the plan has it. A model's sub-proofs send nothing there (from the programs' source).
    If `pending rows` is above 1: `BAT-sigma`, one element sampled, and a sumcheck with k_col
    variables of g = 3, without a round in Y (R13.16), as in R13.18. In each `PSC round` of this
    sumcheck the linear coefficient of the round's polynomial is the one that is not sent (from the
    programs' source). Then, whatever `pending rows` is, `SEND seed` and `SEND answer` for W. Then
    `SEAL`. **W** is the request's own vector, one entry of E for each of the matrix's 2^k_col
    columns, as an oracle message (R13.13 has its shape; from the programs' source). W's `SEND seed`
    has one root, whatever the word of `the test on the table` on a mask sent ahead is. Where it is
    1, the fresh mask of the first fold of the test on the request's own rows goes out with W, in
    W's leaf class, and so is in W's tree (R7.12). Where it is 0, W's tree holds W alone, and that
    mask goes out in the second chain, in a tree of its own (R13.19). *This page does not define* a
    sub-proof that makes a group and trees of its own, which would take places in the chain's list
    and move every later number.

- **R14.8** *The next token* is one element of E, unmasked. It is a prover's message by itself, in a
  round `REPLY`, of kind 7, in which the verifier says nothing. The round comes directly before the
  `SEND seed` with which the last segment begins; where the last segment is also the first, it is
  the chain's first round. The first word of a segment (R12.13) is 1 for the last segment and 0 for
  every other. A verifier compares the element with the next token that the client was given and
  rejects if they differ. How a token becomes an element: R15.7.

### 10.4 The slot

- **R14.7** *The slot* in the file is three things: one word, which is the request's slot itself,
  below 2^depth; the salt of that slot's leaf; and as many digests as the depth, 32 bytes each with
  no length before them (R1.5): the sibling of the leaf first, the child of the root last. A
  verifier rejects if that word is not below 2^depth. The leaf is R7.18's, with the first chain's
  seal as its content. At each level the lowest bit of what is left of the slot says where the
  sibling is: at 0 it is the right child, at 1 the left; then the slot is halved. The value that is
  reached is **the binding tree's root**. It is not a byte of the file. At the depth 0 the slot is
  the word 0 and the salt, as in a key file (R13.6). Why: the next chain's first query holds the
  root to which seal, salt and siblings lead, so the slot comes before that chain in the file.

### 10.5 The shared chain and the second chain

- **R14.9** *The shared chain.*
  - *The first query.* The chain's label is 0x46524d03; the names and the order of its items are
    those of a key file's shared chain (R13.5): `configuration`, `codes`, `the test on the table`;
    `table`, `the binding tree's root`. `codes`, `the test on the table` and `table` are byte for
    byte the first chain's items: a verifier that takes the chains' first queries as inputs compares
    all three and rejects if one differs. Why: each chain is bound to the three by itself.
    The programs compare all three when the second chain begins; a difference there ends the process
    and is not a rejection, and no file's bytes can bring it about, since the programs make both
    first queries from the same values (from the programs' source). The value of
    `the binding tree's root`, like that of a seal's item, is the digest's 32 bytes, with no length
    of their own (R1.5). As in a key file (R13.6) that root is used instead of the chain's 32 random
    bytes in the run's value (case (b) of R4.5). So the chain has no random bytes in the file. Its
    bytes are the same in the files of all requests of a batch (R14.1).
  - *The rounds* are R13.19's. Of the v + k_lo elements of `LIN point` the first k_lo belong to the
    columns within a row of the table, and the last v are the point at which the table's 2^v rows
    are folded (from the programs' source). The folded row's group holds the folded row and the
    shared split's mask message; Omega's group holds Omega and the same mask message; where no split
    is made, one group holds the folded row and Omega (from the programs' source).
  - *What its rounds that sample positions open,* here and in a key file, by increasing place
    (R13.12). `LOC`: the stored table, and no tree of the chain. The first round that samples
    positions in the test on the folded row: the folded row's tree; the tree of the shared split's
    mask message, or, where no split is made, Omega's tree instead; and the tree of that fold's
    fresh mask, if it has one of its own. The first round that samples positions in the test on
    Omega: Omega's tree; the tree of the split's mask message, which is so opened a second time; and
    the tree of that fold's fresh mask. Every later round that samples positions opens the tree of
    its stage and, where the masks do not go out with the switches (R12.5), the tree of that step's
    mask.
  - *The record:* `table` holds the record's stored answers here too. The number of positions that `LOC` samples
    is the plan's. The test on the folded row batches
    one more row for each of the stored answers that a row of the table has (R13.8, R13.15). In the
    programs that number is s. These rows come first in the order of 10.6 (from the programs'
    source).

- **R14.10** *The second chain.* The chain's label is 0x46524d04; its one parameter item is
  `configuration`; its statement is two seals (R4.7), each an item `the seal of an earlier chain`
  whose value is the seal's 32 bytes with no length of their own (R1.5): the first chain's and then
  the shared chain's. It begins with 32 random bytes of its own (R14.1). It continues the first
  chain and adopts one tree of the shared chain, Omega's, as in a key file (R13.7). Its rounds are
  the test on a request's own rows (R13.19): the first is `BAT`, with an empty message, and the last
  is `BASE`; it has no `SEAL`. Its own first query holds neither stages nor codes: it takes the
  third list of stages of `the test on the table`, and `codes`, from the first query of the chain
  that it continues, the first chain (from the programs' source). The group of that test holds, in
  the order of sending, the index-mask message, W and Omega (from the programs' source). Its first
  round that samples positions opens, by increasing place (R13.12): the index-mask message's tree
  (R14.4 (a)); W's; the adopted one; and, where the word on a mask sent ahead is 0, a fourth: the
  tree of the first fold's fresh mask, which the second chain has sent itself and which has the
  place after the adopted one. The second chain ends the file: a verifier rejects a file with a byte
  after it (R8.1). This page does not define parameters in which W's base code is not Omega's, as in
  R13.19.

### 10.6 All rounds of the three chains

The table has every round of the frame, in the order of the file, for a request of L segments. The
rows marked "each segment" are gone through once for each segment, segment after segment: a
segment's `SEND seed`, its `SEND answer` or `LOC`, its sub-proof and later moments, its `BAT`,
`FOLD epsilon` and `FOLD round`; only then the next segment. In the same way the rows marked "each
stage but the last" are gone through once for each stage. `REPLY` is listed first and comes where
its row says. "Stage" and "switch" are section 9's (R12.4, R13.19). A root in square brackets is
present only under the condition named.

In the table, s is the number of answers to each out-of-domain sample (R4.3), and "s answers" always
means answers of that kind. Every item of a message except a root is an element of E, 24 bytes; a
root is 32 bytes. The numbers of seeds' coordinates, positions and rounds are the plan's, and a
chain has exactly the rounds of its plan, in the plan's order. The four items of a `BASE` have l,
l', r and r' elements: l entries of a message and r of the random tail for the last stage's base
code, l' and r' for its mask code (R12.1; from the programs' source). A verifier rejects an element
with a coordinate that is not below p (R1.3) and a file that ends before an item that the plan has
(R8.1).

Which rounds have proof of work is a word of each entry (R4.3); the first and the second chain have
the proof-of-work bits of a request's rounds, the shared chain those of a batch's (section 8). After
the message of a round with proof of work comes its nonce. After a round that samples positions
come, in this order, the nonce, the listed positions if the plan lists them, and the openings (R8.7). (No plan of a key file, a leaf file or a proof file lists them: the line `listed` is 0, 8.1.)

| chain | round | kind | the prover's message | what is sampled | what is opened after it |
|---|---|---|---|---|---|
| first; once, directly before the `SEND seed` with which the last segment begins | `REPLY` | 7 | the next token, one element (R14.8) | nothing |  |
| first; segment 1 | `SEND seed` | 3 | the roots of the trees (R7.12, R8.6) that hold, in the order of sending: the segment's messages of moment 0 (the segment's mask row, 11.1, is in the first of them: from the programs' source); the fresh mask of the segment's fold; the index-mask message | s seeds |  |
| first; segment 1 | `SEND answer` | 7 | s answers for each message that gets an out-of-domain sample, in the order of sending: the segment's messages get one, and so does the index-mask message, whose answers come last; the fresh mask gets none | nothing |  |
| first; each segment m > 1 | `SEND seed` | 3 | the roots of the trees (R7.12, R8.6) that hold, in the order of sending: the carrying message of segment m - 1; the kept messages that are sent again, the one that was first sent earliest coming first (of two that were first sent in one prover's message, the one that comes first in their segment's list, R12.13: from the programs' source); the segment's messages of moment 0 (the segment's mask row, 11.1, is in the first of them: from the programs' source); the fresh mask of the segment's fold | s seeds |  |
| first; each segment m > 1 | `LOC` | 4 | s answers for each message that gets an out-of-domain sample, in the order of sending: the carrying message, the kept messages and the segment's messages get one, the fresh mask none | the positions, and no element | all trees of the group of segment m - 1 |
| first; each segment | the sub-proof's rounds after moment 0 (section 11) |  |  |  |  |
| first; each segment, each later moment | `SEND seed` | 3 | the roots of the trees that hold the segment's messages of that moment | s seeds |  |
| first; each segment, each later moment | `SEND answer` | 7 | s answers for each of these messages | nothing |  |
| first; each segment, each later moment | the sub-proof's rounds after that moment (section 11) |  |  |  |  |
| first; each segment | `BAT` | 1 | nothing | one element |  |
| first; each segment | `FOLD epsilon` | 2 | the sum of the hiding polynomial | one element |  |
| first; each segment | `FOLD round`, k times | 1 | two coefficients of the round's polynomial, the constant and the quadratic one | one element |  |
| first; after segment L | `SEND seed` | 3 | one root: the tree of the last carrying message and of the fresh mask of the last test's first fold | s seeds |  |
| first; after segment L | `LOC` | 4 | s answers for the carrying message | the positions, and no element | all trees of the group of segment L |
| first; the last proximity test | `BAT` | 1 | nothing | one element |  |
| first; the last proximity test; each stage but the last | `FOLD epsilon` | 2 | [from the second stage on, if the masks do not go out with the switches, a root: the tree of that fold's fresh mask;] the sum of the hiding polynomial, last in the message (R8.4) | one element |  |
| first; the last proximity test; each stage but the last | `FOLD round`, k times (the stage's k) | 1 | two coefficients, the constant and the quadratic one | one element |  |
| first; the last proximity test; each stage but the last | `SEND seed` | 3 | one root: the tree of the next stage | s seeds |  |
| first; the last proximity test; each stage but the last | `CS` | 4 | s answers for the next stage's oracle message | the positions, then one element | first stage: the tree of the last carrying message; a later stage: its tree, and its fresh mask's tree if that has one of its own |
| first; the last proximity test | `BASE lambda` | 1 | [a root, if the masks do not go out with the switches: the tree of the masking word of the last step;] the target of the masking word | one element |  |
| first; the last proximity test | `BASE` | 4 | the masked main message; the masked mask message; the masked main tail; the masked mask tail | the positions | the tree of the last stage; and the masking word's tree, if it has one of its own |
| first; after the last proximity test | what the caller's sub-proofs send there, if the plan has it (R14.4 (e)); in a model's proof nothing |  |  |  |  |
| first; only if `pending rows` is above 1 | `BAT-sigma` | 1 | nothing | one element |  |
| first; only if `pending rows` is above 1 | `PSC epsilon` | 2 | the sum of the hiding polynomial | one element |  |
| first; only if `pending rows` is above 1 | `PSC round`, k_col times | 1 | two coefficients, the constant and the quadratic one | one element |  |
| first | `SEND seed` | 3 | one root: W's tree | s seeds |  |
| first | `SEND answer` | 7 | s answers for W | nothing |  |
| first | `SEAL` | 6 | nothing | the seal (R6.8); it is in no file |  |
| between the chains | the slot (R14.7): not a round, and in no query |  |  |  |  |
| shared | `LIN point` | 1 | nothing | v + k_lo elements |  |
| shared | `SEND seed` | 3 | one root: the folded row's tree, which also holds the fresh mask of the first fold of its test where the word on a mask sent ahead is 1 (from the programs' source) | s seeds |  |
| shared | `SEND answer` | 7 | s answers for the folded row | nothing |  |
| shared | `LOC` | 4 | nothing | the positions, in the range of the stored table's code, and no element | the stored table, and no tree of the chain |
| shared | `SEND seed` | 3 | two roots: Omega's tree, then the tree of the shared split's mask message; one root, Omega's, if the folded row's and Omega's base codes are one code | s seeds |  |
| shared | `SEND answer` | 7 | s answers for Omega, then s for the mask message, if there is one | nothing |  |
| shared | `LIN zeta` | 1 | nothing | as many elements as the logarithm of the number of the matrix's rows, rounded up |  |
| shared; only if the two base codes differ | `SPLIT` | 7 | one element | nothing |  |
| shared; the test on the folded row (on both, if the two base codes are one code) | `BAT` | 1 | nothing | one element |  |
| shared; the test on the folded row (on both, if the two base codes are one code); each stage but the last | `FOLD epsilon` | 2 | [a root: at the first stage if the word on a mask sent ahead is 0, at a later stage if the masks do not go out with the switches: the tree of that fold's fresh mask;] the sum of the hiding polynomial, last in the message (R8.4) | one element |  |
| shared; the test on the folded row (on both, if the two base codes are one code); each stage but the last | `FOLD round`, k times (the stage's k) | 1 | two coefficients, the constant and the quadratic one | one element |  |
| shared; the test on the folded row (on both, if the two base codes are one code); each stage but the last | `SEND seed` | 3 | one root: the tree of the next stage | s seeds |  |
| shared; the test on the folded row (on both, if the two base codes are one code); each stage but the last | `CS` | 4 | s answers for the next stage's oracle message | the positions, then one element | first stage: R14.9; a later stage: its tree, and its fresh mask's tree if that has one of its own |
| shared; the test on the folded row (on both, if the two base codes are one code) | `BASE lambda` | 1 | [a root, if the masks do not go out with the switches: the tree of the masking word of the last step;] the target of the masking word | one element |  |
| shared; the test on the folded row (on both, if the two base codes are one code) | `BASE` | 4 | the masked main message; the masked mask message; the masked main tail; the masked mask tail | the positions | the tree of the last stage; and the masking word's tree, if it has one of its own |
| shared; the test on Omega, only if the two base codes differ | `BAT` | 1 | nothing | one element |  |
| shared; the test on Omega, only if the two base codes differ; each stage but the last | `FOLD epsilon` | 2 | [a root: at the first stage always, at a later stage if the masks do not go out with the switches: the tree of that fold's fresh mask;] the sum of the hiding polynomial, last in the message (R8.4) | one element |  |
| shared; the test on Omega, only if the two base codes differ; each stage but the last | `FOLD round`, k times (the stage's k) | 1 | two coefficients, the constant and the quadratic one | one element |  |
| shared; the test on Omega, only if the two base codes differ; each stage but the last | `SEND seed` | 3 | one root: the tree of the next stage | s seeds |  |
| shared; the test on Omega, only if the two base codes differ; each stage but the last | `CS` | 4 | s answers for the next stage's oracle message | the positions, then one element | first stage: R14.9; a later stage: its tree, and its fresh mask's tree if that has one of its own |
| shared; the test on Omega, only if the two base codes differ | `BASE lambda` | 1 | [a root, if the masks do not go out with the switches: the tree of the masking word of the last step;] the target of the masking word | one element |  |
| shared; the test on Omega, only if the two base codes differ | `BASE` | 4 | the masked main message; the masked mask message; the masked main tail; the masked mask tail | the positions | the tree of the last stage; and the masking word's tree, if it has one of its own |
| shared | `SEAL` | 6 | nothing | the seal |  |
| second; the test on the request's own rows | `BAT` | 1 | nothing | one element |  |
| second; the test on the request's own rows; each stage but the last | `FOLD epsilon` | 2 | [a root: at the first stage if the word on a mask sent ahead is 0, at a later stage if the masks do not go out with the switches: the tree of that fold's fresh mask;] the sum of the hiding polynomial, last in the message (R8.4) | one element |  |
| second; the test on the request's own rows; each stage but the last | `FOLD round`, k times (the stage's k) | 1 | two coefficients, the constant and the quadratic one | one element |  |
| second; the test on the request's own rows; each stage but the last | `SEND seed` | 3 | one root: the tree of the next stage | s seeds |  |
| second; the test on the request's own rows; each stage but the last | `CS` | 4 | s answers for the next stage's oracle message | the positions, then one element | first stage: R14.10; a later stage: its tree, and its fresh mask's tree if that has one of its own |
| second; the test on the request's own rows | `BASE lambda` | 1 | [a root, if the masks do not go out with the switches: the tree of the masking word of the last step;] the target of the masking word | one element |  |
| second; the test on the request's own rows | `BASE` | 4 | the masked main message; the masked mask message; the masked main tail; the masked mask tail | the positions | the tree of the last stage; and the masking word's tree, if it has one of its own |

*The order of the rows in a batching.* A round `BAT`, and the element of a round `CS`, batch all
rows of one group: row number i, from 0, is multiplied by the i-th power of the sampled element, and
the rows are added, targets included. The rows come in three classes, in this order.

1. The rows that are in neither of the next two classes, in the order in which they are made. In a
   segment from the second on these are: the closed segment's one claim, carried over; then, for
   each kept message that is sent again, in the order of sending, the s rows of its earlier
   out-of-domain sample, which count here and not in class 2. That earlier out-of-domain sample is
   the one that the message got when it was sent into the group that has just been closed: for a
   message with a life above 1 that is its latest sending before this one, and not its first. After
   these rows come the rows that the sub-proof leaves. In the test on the folded row:
   the record's rows (R14.9), then the split's row. In the test on Omega: the split's other row, the
   one on Omega's group. Where the two base codes are one code and one test is made on both, the row
   is not split, and it comes whole where the split's row would come. In the second chain: the row
   on W, then the row on Omega and W.
2. The rows of the out-of-domain samples that the group's messages got when they were sent into this
   group: message by message in the order of sending, and within a message by answer.
3. The rows that stand for reads at sampled positions, by the draws (R6.6). After a closing and in a
   `CS` each draw gives, in this order, a row for the folded main row, one for each kept main row,
   by message and then by row, one for the folded mask row of the group, and one for the folded
   fresh mask. After the shared chain's `LOC` each draw gives one row.

What each row is: 13.9. (All of this passage on the order of the rows is from the programs' source.)

*How many rows a round batches* is the last of the eleven words of its entry (R4.3). In the table t
is the number of positions that the round, or the `LOC` before it, samples; J is the number of kept
messages that are sent again in the segment's first prover's message; K is the number of their main
rows. (The whole table is from the programs' source.)

| round | rows |
|---|---|
| a segment's `BAT` | the rows that its sub-proof leaves; s for each message of its group that gets an out-of-domain sample; and, from the second segment on, 1 + s J + t (3 + K) |
| the `BAT` of the last proximity test of the first chain | 1 + 3 t + s |
| a `CS` | 1 + s + 3 t |
| the frame's `BAT-sigma` | the word `pending rows` |
| the `BAT` of the test on the folded row | t + 1 + 2 s, t being the number of positions of the shared chain's `LOC`; and one more for each of the stored answers that a row of the table has (R13.8, R13.15; in the programs that number is s) |
| the `BAT` of the test on Omega | 1 + 2 s |
| the `BAT` of the second chain | 2 + 3 s |

*How long a proof file is.* No file holds a length. Every part of a proof file has the length that
the plan and the shapes give it, except the openings after a round that
samples positions. Their length depends on the sampled positions: a position that is sampled twice is opened once (R6.6), and the number g of given digests follows from the list (R7.9). An
opening at a list of m positions has m (8 W + s) + 32 g bytes. These are the letters of R7.8: in
this formula alone, W is the width of a column and s the number of bytes of a salt. The planning
call (14.6) prints two lengths for each file, and forms them in this way, for a round that samples t
positions and a tree of depth a (from the programs' source). *At most:* m = t in every opening, and g is R7.9's bound, with 2^(a-1) instead of t where t is above it; this
is taken in all openings of the file at once. *On average,* over t independent and uniform
positions: m = 2^a (1 - (1 - 2^-a)^t), and g is the sum over d = 1, .., a of 2^d ((1 - 2^-d)^t -
(1 - 2^(1-d))^t). A tree that is shorter than the list's range (R7.15) is counted with its own
depth. Each length is the sum
over all parts of the file.

### 10.7 The names of the rounds

- **R14.11** *The names of the format's rounds, all of them.* A first query holds the rounds' names
  and not their kinds (R4.3), so the kind follows from the name, and the list of names is part of
  the format. Kind 1: `BAT`, `BAT-sigma`, `PSC round`, `PSC Y`, `ZERO`, `LINE`, `LKP shift`, `JOIN`,
  `LKP`, `FOLD round`, `BASE lambda`, `LIN point`, `LIN zeta`, `LINK point`, `LINK phi`. Kind 2:
  `PSC epsilon`, `FOLD epsilon`. Kind 3: `SEND seed`. Kind 4: `LOC`, `CS`, `BASE`, `LINK`. Kind 5:
  `RNG`, whose sampled values are matrix entries (R6.7), as many as the tenth of its entry's eleven
  words says. Kind 6: `SEAL`. Kind 7: `SEND answer`, `SPLIT`, `PSC values`, `REPLY`, `LAST`, which
  comes after a chain's last prover's message if nothing else closes it. `RECORD` names an entry
  that is not a round (R13.15). That is thirty names: 29 rounds and `RECORD`. A plan of a key file,
  a leaf file or a proof file that holds another name is not a plan of the format, and a verifier
  rejects it. Where `LKP shift` and `RNG` come, and what their messages hold, is in section 11: they
  belong to the sub-proofs of a model's segments.
