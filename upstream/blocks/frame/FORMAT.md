# The format

This page says which bytes stand in which file, in which order, and where each length comes from. The code that it describes is
`frame/files.h` (the chains of a file), `iop/req.h` (the order of a request's links), `frame/lin_frame.h` and `frame/link_frame.h` (the test on the committed
table, the link between two tables), `numbers/blocks.h` (the order of the messages in a block), `chain/rounds.h` (how a message enters a file and a hash
chain), `frame/heads.h` (heads) and `frame/check.h`, `check_key.h`, `check_entry.h` (first queries, the walk through a file).

**Which page rules.** What is HASHED, byte by byte, and how an item is spelled is ruled by `docs/format/FORMAT.md`: there the page is right and a program that differs is wrong. This page only DESCRIBES how the frame lays the files of the real instance; where it and the code differ, this page is wrong.
Where this page and `docs/format/FORMAT.md` differ, `docs/format/FORMAT.md` decides.

`cp_commit`, `cp_prove` and `cp_verify` write and read these files, at full size too.

## 1. Ground rules

1. **Bytes.** Little-endian throughout. An element of F is 8 bytes and is refused unless it is below p. An element of E is three
   elements of F, 24 bytes. A digest is 32 bytes. A nonce is 8 bytes here (the ruling page admits 1 to 8). A salt in a Merkle-tree leaf has the length of the parameter line `salt`; the chain's salt (section 2) has 32 bytes.
2. **No file holds a length, a count or an offset.** Every length follows from the parameter file (and, for the sub-proofs, from the
   statement's shapes; and, for an opening, from the sampled positions (docs/format/FORMAT.md, R13.2)). A reader plans the file first and then takes the planned items off it, one after the other. It refuses a file
   that ends before the last planned item, and a file with a byte left after it.
3. **One version word** (`frame/version.h`): four characters, `H`, the format's number `8`, the field's number, the hash's number.
   `H821` is p = 2^64 - 2^32 + 1 with plain Reed-Solomon codes and BLAKE2b. There is one hash, number 1. A program serves this one word and
   refuses every other as `version-unknown`. What is hashed, byte by byte, is ruled by `docs/format/FORMAT.md`. The word
   stands in the first line of the parameter file, in the head of every file and, as the first of the parameters, in the first query of every hash chain.
4. **One parameter file, one parser** (`frame/parameters.h`): one order, one spelling. Every byte of the file, and of every other public file that configures
   a program (a layout of arrays, public constants of the model), enters the digest of the configuration (`frame/heads.h`), and that digest enters the first
   query of every hash chain. So a proof made under another target, work vector, rate, code
   length or cut into stages has other challenges from its first round on, and is refused.
5. **What has no line is derived** (`frame/derive.h`): slack and distance of every group, numbers of positions and of columns, random
   coordinates, mask messages of the later stages, combinations of the smallness test.

## 2. A hash chain in a file

A file is a head followed by one or more chains. A chain's bytes are, in the order of the protocol:

| item | bytes | enters the chain |
|---|---|---|
| the chain's salt, if the chain hides something | 32 | in the first query |
| what the prover sends before a verifier's message: sample answers first, then roots, then scalars (elements of E), a hiding sum last | roots 32 each, elements 24 each | yes |
| a nonce, if the round that follows has work bits | 8 | in that round's query |
| after a draw of positions: the listed positions, if the line `listed` is not 0 (the parser takes `listed` at 0 alone, so they are in no file of the programs) | `listed` each, sorted, each once | no: compared with the derived ones |
| after a draw of positions, for each tree that is opened at the list: see section 3 | | no: checked against the tree's root |

The first query holds the statement: version word, digest of the parameters, what the chain is about (section 4), and the salt. Every round makes
its query, also one in which the verifier says nothing: the query takes the prover's message in, and nothing is drawn from its answer. No root stands in such a round:
a prover's message with a root is followed by a verifier's message that is not empty (a plan that has it otherwise is refused, and a channel ends the process). A chain that other chains build on ends in a seal, one more query; nothing of it is written.

## 3. An opened tree

What the prover has sent after the latest verifier's message that was not empty is gathered into trees, one for each set of lists at which rows are read. Every tree is a perfect
binary tree. A leaf holds the symbols of all its rows at one position, then, for rows of a shorter code that ride in the tree (mask rows, read at the same
lists), a COPY of their symbols at that position modulo the shorter length, then the leaf's own salt. Mask rows that cannot ride so have a tree of their own at their
own length. For a list of drawn positions, equal ones taken once, in increasing order:

| item | bytes |
|---|---|
| for each position: the leaf's symbols (8 for a row over F, 24 over E), copies among them, and its salt | |
| the sibling digests that the paths of all opened leaves need and cannot compute, level by level from the leaves, left to right | 32 each |

The most digests for t draws in a tree of depth d, with 2^a <= t < 2^(a+1): t (d - a - 1) + 2^(a+1) - t. Which oracles share a tree and a leaf is
data of the program (`iop/context.h`), fixed by the rule above.

## 4. The files

### key
`head`: tag, version (2 words of 4 bytes). Then the root of the table and its record: for each row as many answers as the line `answers` says (the parser takes 1 alone) (the seeds stand in no file: they are a function of the
table's name, and whoever needs them makes them). Root and record are items of the file, no messages of a chain. The root and the record of every stored table of which a chain speaks, whether or not it opens a column of that table (the key's one-time chain and a proof's first chain open none), stand among the
items of its statement, by value, before its other items, one item a table: the root as a string of bytes, and then the list of the record's answers. The item is named `table`; a link's chain reads two tables, and its
items are named `the first table` and `the further table`. The table's name is computed from the root, from the configuration's digest and from the table's shape (six numbers of 8 bytes each), so that the name has 32 + 32 + 48 = 112 bytes: it is these bytes themselves, in this order, and no digest of them; the record's answers do not enter it. It
stands in two places: as the statement of the record's own chain, and in the input of the commitment ID. No item holds it. Then three chains, all with the committer's work bits:
1. **one-time chain**: the masks of the committer's rows
   on the table as an oracle message with its sample; the proof that the table's entries are in their ranges (`frame/wf_frame.h`, `iop/wf.h`): the look-ups' multiplicities as an
   oracle message with its sample, the look-ups part by part, that message's own proximity test, and the rows that the parts leave on the table joined into one; the committer's own
   vector as an oracle message with its sample; seal. Behind it the **place**, as in a proof, in a tree of one leaf: the word 0 and the leaf's salt.
2. **shared chain** as in a proof, for a batch of one, at the distance of the committer's one-time test, under a label of its own (`docs/format/09-key-and-leaf-files.md`, R13.4).
3. **second chain** as in a proof, under a label of its own.

### entry (a further stored copy of the table)
`head`: tag, version. Then the root of the further table and its record, as in a key. One chain, whose statement begins with both tables' items, `the first table` and `the further table`: the link: fold point, both folded rows re-encoded in one prover
message with their samples, the columns of both tables in one draw, the point of comparison; the proximity test on the two folded rows.

### proof
`head`: tag, version, number of the entry (3 words). Then:
1. **first chain** (the request's own; bound to the stored copy of the table by its root and record, and to the request; the reply is one unmasked scalar; it has its
   own round, in which the verifier says nothing, right before the last link's first prover message; the reader compares it with the reply that it was given). With the very first prover message go the
   masks of the request's rows on the table, in a tree of their own: they are read at the list of the second chain's test and at no link's. For each link of the
   request, in this order:
   first prover message (the message into which the link before was closed, kept rows, cells, multiplicities; as copies in its leaves the link's mask row
   and the fresh mask of its fold), seed, answers | *then the positions of the link before, and its opened trees* | the sub-proof's look-ups and relations |
   second prover message (partial evaluations of the packed cells), seed, answers | the smallness test's matrix | third prover message (digits and their
   multiplicities), seed, answers | the digits' look-up, the rows on the table | batching | fold over the row variables.
   After the last link: the message into which it is closed, alone, with its sample; its positions; the proximity test on that message down to the
   last stage; the pending rows on the table into one; the request's own vector as an oracle message with its sample; seal.
2. **place**: the place of the request's leaf among the leaves of the tree over the seals (8 bytes), the salt of that leaf and its path in the tree. At which place a request stands is the prover's choice: to hide the order of arrival it draws a random arrangement, so this is not the request's number in the order of arrival. A leaf of that tree is the hash of a
   first chain's seal and a salt; a leaf that no request takes is a uniform digest.
   It stands here because the next chain's first query holds the root to which seal, salt and path lead.
3. **shared chain** (one for the batch, repeated in each request's proof; statement: `table`, then `the binding tree's root`, the root of the tree over the seals of the batch's first chains;
   no salt): the shared point | the folded table re-encoded, seed, answer | the table's columns | the table times the equality vector of the shared
   point and, since the two lie under different codes, a message of masks alone that parts the one row which lies on both; seed, answers | the point of
   comparison | the target of one part of that row | the proximity tests on the two groups | seal.
4. **second chain** (the request's own; statement: both seals): the proximity test on the request's own group.

## 5. Planning calls

`cp_frame -plan id|key|leaf|request FILE [-rounds] [-fields]` (`cp_verify` has planning calls of its own: `-plan id | request | accounting | arrays`). They run the routines by which a checking run
plans itself and print: the digest of the parameters; every group with its derived numbers; with `-rounds` every round (kind, error, work), the largest
error first; the accounting (`numbers/accounting.h`): the largest round error, ONE verdict, the prover's hash calls t as a free variable, the verifier's own,
the digests' term, the compiled bound, both levels, and beneath them the stricter readings (largest error without work; sum of all errors with and
without work; where the file has no work: the sum alone); the files' lengths; with `-fields` every field of every file (`frame/fields.h`). Last line: `PLAN ONLY: no proof was checked`. Status 4; 6 if some round does not meet the file's target; 7 if all do but a condition of hiding is missed.

## 6. The rule for a distance

**Every group goes up to the Johnson radius less a derived slack**, at all its stages and in the committer's files too: that is the rule, the first number of a line `group`, which is 1. A group's closeness term is that of the theorem on codes (Theorem 4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf published at STOC 2026), which the programs' accounting takes as given: the theorem is not in the proceedings; the full version states it, prints one paragraph behind it, and refers for the proof
to a note that it cites as a personal communication (`numbers/codes.h` has the places and the authors' own words); a statement on codes held to follow from it is proved in the Lean library (`docs/format/FORMAT.md`, 13.7, condition 4). So to which table a key and its entries are bound
rests on the theorem on codes, and so does that an accepted request is true for that table. A distance within a third of a code's distance, where a program computes one, takes the count of a lemma that is proved in the paper that this protocol builds on, which THIRD_PARTY.md names (`numbers/codes.h`). A planning call prints each group's rule and footing, and for each file how
many of its rounds with a closeness term stand on which footing.

## 7. A first chain's statement: two of its four items, and one parameter line

- The request in a first chain's statement: the item `request` is the caller's bytes, whole. In `cp_verify` these are 177 words (708 bytes), with the next token's word set
  to 0. They hold nine digests made with a second, fixed hash function, built on BLAKE2b's compression function: whoever forms the item makes five (of three lists of lines, the scale file and the request file),
  and four more that were made so stand in the parameter file: `cp_verify` computes each of these four tables and its digest, places its own digest in the item, and rejects if the file's line differs. The file's lines for these four digests are optional, all 32 or none: a file with some of them is rejected (`policy-tables`). If it has none, nothing is compared, and a call that checks a proof ends `NOT-A-VERIFICATION` (status 14) at best (`docs/format/FORMAT.md`, R14.3, R15.1 to R15.7).
- The request itself in a first chain's statement: the item `the request file's words`, right behind `request`, is the caller's second byte string, whole. In `cp_verify` and
  `cp_prove` it is the request file: T + 2 words (the tokens, then r and e), four bytes each, the low byte first. So the first query holds the request by value under this
  library's hash; the digest of the same words in `request` stays beside it, and `request` keeps its bytes. The library compares the two with nothing: each program makes both
  from the one request file (`docs/format/FORMAT.md`, R0.5, R14.3).
- The line `tau` (combinations of the smallness test) is among the statement's lines, and it is tested: a proof's run is rejected (`plan-tau`) unless the number that is
  derived here equals it.
