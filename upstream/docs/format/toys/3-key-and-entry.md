# The toy run `key-and-entry`: a key file and a leaf file on a committed table: which protocols they run, and the verifiers' decisions

A **toy run** is a small run of the library's blocks on synthetic data. Sections 0 to 6 describe the
toy run `key-and-entry`. Parts 6 to 10, after them, give the verifier's side of each block that it
uses, as mathematics and bytes. This page builds on the pages on the toy runs `product` and `links`
(parts 1 to 5): SEND, ZERO, PSC, BAT, FOLD, CS, BASE, the proximity test, groups and their layout,
the three kinds of pad. The format page is `FORMAT.md` in the directory above this one; it defines
the bytes and the terms, and its rules are cited by number (R4.9). File names are given from the
repository's root (`blocks/iop/wf.h`) or from `blocks/` (`iop/wf.h`). A bare name that ends in `.h`,
such as `lkp.h`, is a file of `blocks/iop/` unless its directory is named beside it, and a bare name
that ends in `.py` is a file of `tests/new/second/`. In prose a commitment's first
stored table is the anchor, a further one a leaf table, and their files are the key file and the
leaf file. Hashed, printed and typed names use other words and do not change: the run's name
`key-and-entry`, the files `key.file` and `entry.file`, the items `the first table` and
`the further table`. A sentence marked "(from the programs' source)" was read in the source and not run. No program, the second verifiers among them, has been written from this page's text word for word. It has not been compared with the paper that this protocol builds on: there may be differences that are not listed here, and what it gives as a choice of the programs has not been held against that paper's text and is not claimed as new.

Where this page speaks of hiding, it
gives what the design is meant to do and the reasons of the programs' source. All that this page
tells of a prover's side, and every test of hiding that it names, is the library's CPU reference at
toy size; it says nothing of the GPU engine, the kernels and the sampling of `cp_commit` and
`cp_prove` (from the programs' source, `blocks/tests/hiding/README.md`). Take an outcome of a test of hiding that this page reports as
seen on test builds, not on the released programs.

## 0. What it is

A **committer** has a matrix M over F (here 100 rows, 64 columns, synthetic numbers). It places M
into a **table** of 2^3 rows, encodes each row with a random tail and hashes the columns into one
**stored tree**. This table is the anchor. The key file is designed to show, without showing M: (a)
ranges: certain public linear maps of M's entries are in small ranges (lookups); (b) the committed
word is close to one code word, and (a) is said of that word's matrix. The leaf file comes with a
leaf table, which has the same rows under fresh randomness, and is designed to show that the leaf
table's rows are the anchor's. Each verifier takes the public files and its file; the leaf file's
verifier also takes what the key file's verifier has kept of the anchor. They take nothing else.
There is no model here: that a key file belongs to a model of some name would be its publisher's
word, outside the protocol.

How far this is from the first two toy runs:

| | the key file | the leaf file |
|---|---|---|
| hash chains | Three in one file, one after the other: the one-time chain (the chain's label 0x46524d02), the path, 48 bytes (the slot and the salt of the seal's leaf in a binding tree of one leaf, and no sibling: part 9, section 5; R13.2 lists the two together under the name slot), the shared chain (0x4c494e01), the second chain (0x4c494e02). The first two end in a seal. The shared chain has no random bytes in the file: a digest computed from the first chain's seal is used instead of them and is in its statement. The second chain's statement is the two seals, and it continues the first chain's statement, rows and trees | one chain, the chain's label 0x4c4e4b01 |
| roots in a prover's message | once two: entry 5 of the shared chain's plan (two oracle messages of different leaf classes in one prover's message) | at most one |
| mask rows shorter than main rows | yes: in the group of the key file's own rows the mask code has length 2^12 under a main code of 2^13: a mask row's symbol at position x is read at x mod 2^12; and there are trees of mask rows alone, opened at the short positions | no: every mask code is as long as its main code |
| stored trees | one, the anchor's: leaves under number 174, started from the configuration's digest; its root is in the file before the chains and in the statement; the columns at 76 sampled positions are opened under it | two: the anchor's (76 draws) and the leaf table's (76 draws) |
| a tree opened at several lists | Two trees. Omega's tree (Omega is alone in it) is opened at a list of the shared chain and, adopted with its number, start and root, at a list of the second chain. The tree of the split's mask message (mask rows alone) is opened at two lists of the shared chain itself: that message is a member of both of that chain's groups (rounds 14 and 26) | none; each table's tree is opened at its part of one list |

Also new: a head of 8 bytes; bytes of the file that are not a message of a chain (the root, the
record's stored answers, the path); statements that are not empty; the record's seeds, sampled from
a chain of their own on the table's name; the
blocks LKP (with PSC in its generality, LINE, JOIN), WF, BATT, LIN with FOLDEQ and SPLIT, LINK; a
proof of work of 8 bits in every round that samples positions; three verdicts: no, yes, and yes with
proof of work (here the last: for the target of 128 bits at least one round needs its proof of
work).

## 1. The public files, the configuration's digest, the commitment ID

There are two public files. The tool writes them as `public-file-1.txt` (32 lines: the frame's
parameters) and `public-file-2.txt` (4 lines: `links`, `pending`, `matrix`, `identifiers`). The
configuration's digest is theirs by R4.9, in this order. Everything else that is public (settings,
codes, shapes, all four chains' plans) follows from them by one routine, `configure()`, and by a
routine of the calling program that `configure()` is handed. That second routine holds the lookups'
sizes, offsets, weights and padding keys (part 6, section 0). It also gives a request's range
checks, the numbers on which the condition of hiding `range-checks` is tested
(`Shapes::range_checks` in `frame/configure.h`; from the programs' source). All these are in no
public file, and the range checks are in no chain's first query either. A second verifier takes the
parameters' bytes as given and does not derive them: that they follow from the files has not been
shown a second time.

The **commitment ID** (R4.11) is one hash, under a number of its own (179), of the configuration's
digest, the anchor's name, its seeds and its stored answers (the whole string:
R4.11). It names what is committed under which public files, and does not depend on any
leaf table. No verifier's decision here reads it, and no file here holds it.

## 2 and 3. The key file (part 6) and the leaf file (part 10): what is where

The key file, in order:

- the head: `CPIK`, then the version's four bytes, `H821`;
- the anchor's root, 32 bytes;
- the record's stored answers;
- the one-time chain: 32 random bytes; SEND of the index-mask message (mask rows alone); WF: SEND of
  the multiplicities, two lookups (LKP), WF's own proximity test; BATT: five rows on the matrix into
  one; LIN's first part: SEND of W; the seal;
- the path: the slot, which is the word 0, and a salt of 40 bytes: the seal's leaf in a binding tree
  of one leaf (number 173). It is the path of a batch of one (R13.2): a slot is the place of a leaf
  in a binding tree (R7.18), not a number in an order of arrival;
- the shared chain, without random bytes of its own: the fold point; FOLDEQ: SEND of the folded row,
  and the anchor's columns at 76 sampled positions opened under the file's root; SEND of Omega and
  of a mask message for a split; SPLIT; two proximity tests; the seal;
- the second chain: 32 random bytes; Omega is adopted; the row that ties W to Omega; one proximity
  test. The file ends with it.

The leaf file, in order:

- the head: `CPEN`, then the version;
- the leaf table's root;
- its record's stored answers;
- the link's chain: 32 random bytes (R13.3); the fold point; SEND of both tables' folded rows in one
  tree, with a fold's mask; 76 + 76 positions in one verifier's message (round `LINK`): the
  anchor's columns opened under the key file's root, then the leaf table's under the leaf file's;
  the point of comparison; one proximity test.

Statements that a verifier forms on its way. The one-time chain's: the item `table` (the root as a
string of bytes, then the list of the record's stored answers). The shared chain's: `table`,
`the binding tree's root`. The second chain's: `the seal of an earlier chain`, twice.

Which tree is which (the numbers follow below). One-time and second chain: the index-mask message;
the multiplicities with their mask rows and a fold's mask; two of WF's test; W with a fold's mask;
then, adopted, Omega's tree, with the shared chain's number and start (it uses up one number of the
reading chain's count); two of the second chain's own test. Shared chain: the folded row with a
fold's mask; Omega, alone in its tree; the split's mask message; two of the first test; a fold's
mask in a tree of its own; two of the second test.

The numbers below are a program's print from the files that the tool makes.

- The key file has 503,744 bytes. Its parts begin at byte: head 0; the anchor's root 8; the record's
  stored answers 40; the one-time chain 232; the path 161,768; the shared chain 161,816;
  the second chain 408,912.
- Rounds: one-time 711, shared 34, second 13; the one-time chain's plan begins with the entry
  `RECORD`, so it has 712 entries. Numbers of bytes of the parameters: 76,647, 4,982, 1,521.
- Items of `key.one-time`, in order, each with the number of bytes of its value: `configuration` 32;
  `index-mask message` 56; `ranges` 720; `binding tree` 16; `codes` 496;
  `the test on the table` 728.
- Items of `key.shared`, in order, each with the number of bytes of its value: `configuration` 32;
  `codes` 496; `the test on the table` 728.
- Items of `key.second`, in order, each with the number of bytes of its value: `configuration` 32.
- Statement of the one-time chain: `table` 240.
- Trees, as (number, depth, width in F), in the order in which a chain holds them, which is the
  order of openings at a list. One-time and second chain: (0, 12, 9), (1, 14, 40), (2, 14, 24), (3,
  15, 12), (4, 13, 15), (1, 13, 3), (6, 13, 24), (7, 15, 12). Shared chain: (0, 13, 15), (1, 13, 3),
  (2, 13, 3), (3, 13, 24), (4, 15, 12), (5, 13, 3), (6, 13, 24), (7, 15, 12).
- Rounds that sample positions (round, name, how many, log2 of the range; 8
  proof-of-work bits each). One-time: 693 `CS` 50 14; 698 `CS` 50 14; 700 `BASE` 61 15. Shared: 4
  `LOC` 76 14; 14 `CS` 63 13; 19 `CS` 51 13; 21 `BASE` 49 15; 26 `CS` 46 13; 31 `CS` 51 13; 33
  `BASE` 49 15. Second: 6 `CS` 46 13; 11 `CS` 43 13; 13 `BASE` 41 15.
- Sampled is not opened: a list holds each position once (R6.6). Rounds of these files in which a
  position is sampled more than once: `key.one-time`: none; `key.shared`: none; `key.second`: round
  6: 46 sampled, 45 different; the leaf file: round 4: 152 sampled, 151 different.
- The leaf file's round `LINK`: of the first 76 draws 75 are different (so many columns of the
  anchor are opened), of the last 76 draws 76 (the leaf table's); no position is sampled in both
  parts.
- Trees of the link's chain, as (number, depth, width in F): (0, 13, 27), (1, 13, 24), (2, 15, 12).
  They are opened at its second, third and fourth sampling of positions.
- The lookups: d = 3, 2^4 plane slots, 14 live, 1 digit, offset 4, padding key 3, parts of 2^(5, 5,
  5, 5) places; d = 5, 2^4 plane slots, 11 live, 2 digits, offset 16, padding key 17, parts of 2^(4)
  places.
- The leaf file has 131,432 bytes and 19 rounds; its plan begins with the entry `RECORD`, so it has
  20 entries, and its parameters have 2,648 bytes. Items: `configuration` 32; `codes` 160; `the link`
  192. Statement: `the first table` 240; `the further table` 240. Rounds that sample positions: 4
  `LINK` 152 14; 12 `CS` 63 13; 17 `CS` 43 13; 19 `BASE` 41 15.

## 4. Names of rounds that are new, with their kinds in the query

Kind 1 (elements of E): `LKP`, `JOIN`, `LINE`, `BAT-sigma`, `LIN point`, `LIN zeta`, `LINK point`,
`LINK phi`. Kind 4 (positions): `LOC`, `LINK` (no element after them). Kind 7 (nothing): `SPLIT`.
Kind 6: `SEAL`. The plans of the one-time chain and of the link's chain begin with an entry `RECORD`,
which is not a round of its chain (R13.15).

## 5. The verifiers' decisions

The key file: yes, or yes with proof of work, if and only if all of these hold.

- The head.
- (D1) Form. Every element read is canonical; the file has exactly the planned items; no byte is
  left over. Under D1 also, each on the spot: no chain has a run's value of 32 zero bytes (R4.5);
  the slot is below 2^d: with the key file's binding tree of one leaf, d = 0, the slot's word is 0
  or the verifier rejects.
- (D2) Proof of work, in each of the thirteen rounds that sample positions.
- (D3) Every opening up to its root: the chains' own trees, and the anchor's columns up to the root
  that is in the file.
- (D3b) One value at a vertex (R7.14). No vertex, leaf or given digest may get two values under one
  root. The verifier keeps a record of values at vertices (not to be confused with a table's
  record). There is one record for every root's 32 bytes that is read in a session, that is, in all
  chains of one verdict, whichever tree, chain or file the root came with. Only what an accepted
  opening shows is entered. The clause is live here for the two trees that are opened at two lists;
  a tree that a later chain adopts finds its values there.
- (D4), (D5) in each of the four proximity tests: the last step's linear check and its checks at the
  positions.

Nothing else is tested on the proof's bytes. No equation of LKP, WF, BATT, LIN, FOLDEQ or SPLIT is
tested on the spot: each moves a claim into a row of a group, and each group is closed by one of the
four tests. The groups are: WF's group (the lookups' rows); U (the folded row: the columns' rows,
and one share of the row that ties Omega to it); OM (Omega: the other share); OWN (the claim on W;
the row that ties W to Omega). The digest that the path gives from the first chain's seal is
compared with nothing: a wrong salt gives another shared chain, under which its bytes fail.

On D3b. With a real hash no file can tell a verifier that keeps that record from one that does not:
it would take a collision. So no run on a file shows the clause at work. The format page gives the
reason (R7.14): two different values at one vertex under one root mean a collision of the hash. By
R7.14 the record is a cheap guard that makes that event visible, and no bound depends on it.

The leaf file: the same, with one chain, two stored trees and one test; each of its trees is opened
at one list. D3b is not an empty clause for a leaf file's verifier as a function: if the leaf
table's root had the anchor's 32 bytes, the two stored trees would share one record of values, and a
position can be sampled in both parts. The leaf file's second verifier keeps the record too.

Before a byte is read, both verifiers reject, by name, public files that are not admitted (part 6,
sections 1 and 6).

## 6. The files, what has been compared with what, and limits

`bash tests/new/second/run_honest.sh <a new work folder>` builds the tool, makes the files and has
each checked twice. It builds no malicious prover. Most false files of the table below were made
by another script of the programs' tests, which also builds malicious provers; it is not part of
this repository. The names in capitals that this page gives to malicious committers and provers,
to injected faults and to false files (`KEY_ENTRY_OUTSIDE`, say) come from this project's own
tests: the test switches of these names, and the code under them, are not in this repository and
cannot be built from it.
Of the false files of the table `run_honest.sh` makes only the two leaf files that
`files_check.py` bends out of the honest leaf file's bytes (`expected/bent.tsv`), beside the honest
files and their altered copies, and it has each checked twice. Some false files are stored in the
repository, in `examples/protocol-toy-false-files/`: five false key files, two false leaf files,
three false proof files of a toy batch and one false proof of each of the toy runs `product` and
`links`, with an honest key file, an honest leaf file and two honest proof files as controls. The
directory's `LIST.tsv` gives each file's SHA-256, what is false in it, the checker that it was given
to and the words of the rejection (from the programs' source). `run_honest.sh`
rejects a work directory that exists or that is inside the repository, and also one whose parent
directory does not exist or that is below a directory that another account may be able to change
(from the programs' source; `tests/new/second/README.md`). The tool,
`blocks/frame/handover_files.cpp`, writes the two public files, `key.file`, `entry.file`, the bytes
of the four chains' parameters (these bytes decide), two statements, the values that the two C++
verifiers note on their way (106 and 30), and altered copies with a recipe (`altered.tsv`). Matrix
and randomness come from small numbers in the tool's source: nothing is hidden in these files.

The second verifiers are `key_check.py` (on `lin_check.py` and `zkpy.py`) and `entry_check.py`,
called by `files_check.py`. The heads of `key_check.py`, `lin_check.py` and `entry_check.py` each
name the programs' sources on which that check is not independent of the C++ programs. For the
formulas these are what says in which order the blocks are called, and how the tests' small set-ups
are made. The heads name more: for the false key files, lines of `iop/wf.h`, `iop/lkp.h`,
`iop/tests/wf_test.cpp` and one switch in `iop/base_case.h`; for the leaf file, the first query's
items and their bytes; the parsers, the record of values at vertices and the noted values. And by
the head of `lin_check.py` that check follows the same rules as `iop/lin.h`, `iop/foldeq.h`,
`iop/split.h` and `iop/index_cpu.h`, which part 9 describes: a rule understood wrongly is wrong in
both, and the check does not show it. Of every other file of the programs the heads of
`key_check.py`, `lin_check.py` and `entry_check.py` say only that nobody has examined whether the
check is independent of it.

| what | checked by | outcome |
|---|---|---|
| the honest key file and leaf file | `files_check.py` | Both are accepted. The configuration's digest is computed again from the two public files. The statements are equal. 105 of the key file's 106 values and all 30 of the leaf file's are equal; the 106th, `BATT: the first weight's scalar`, the second verifier does not note. Limit: 14 + 23 of these values have in the second verifiers the names that the C++ lists give them, and the 23 are in their list's order, so on which values are noted, on their names and on that order the two are not independent |
| 27 altered copies (15 of the key file, 12 of the leaf file) | both the C++ verifiers and the second verifiers | All are rejected. Three of the key file's are at spots that only a program that reads the file knows: the path's slot (rejected on the spot), the path's salt (rejected at the next proof-of-work test), a symbol of the anchor's first opened column (rejected at its opening). The first stored answer of the record, altered in each file, is rejected at the first proof-of-work test, because the answers stand in the chain's first query: so no altered copy reaches the rows on the record |
| 14 cases of false key files in 21 files (one case is a committer that mends its false claim at one code switch, with a file for each of the 8 code switches), nine false leaf files, one false pair (made and run at toy size, on synthetic data, with [other programs](../../glossary.md#other-programs); the list of these files is not in this repository) | both C++ verifiers; the second verifiers, which name the check, the proximity test and so the chain | Each is rejected, and in every case the two verifiers name the same check. At the linear check: four cases of key files (eleven files) in the one-time chain's own test, three in the shared chain's first test, five in the second chain's with all three earlier tests passing, and six leaf files. At the checks at the positions: one key file, one leaf file. At a proof-of-work test: one key file, and the pair. At an opening: two leaf files. Limits: no call of the tool takes another anchor, so the C++ verdict on the pair comes from a test build's own call of `check_entry`; of the two leaf files rejected at an opening the C++ verifier's place says only that it was before the test |
| every digest, first query, round, seal and opening of both honest files | the two programs `../read_key.py` and `../read_entry.py` | no difference |

Limits:

- No file exercises: the shared chain's second test (OM) by a false file; the mask words' half of
  D5; a non-canonical element; D3b. Instead of a file for D3b,
  `files_check.py --spoil-vertex-record` injects a fault into the record of values of the key file's
  second verifier, and `vertex_values_test.py` puts a value into that of the leaf file's: each then
  rejects the honest file.
- No second verifier in `tests/new/second/` covers: LKP's cubes as cubes, its weights as lists of
  product forms, the canonical bytes of its public object (in a key file it is derived and is in no
  item), a shift, key blocks of padding; field 1.
- BATT with its end value shown (part 8, section 5: a request's segments use its messages, the key
  file does not) is not covered by the second verifiers of this toy run, `key_check.py` and
  `entry_check.py`. `proof_check.py`, the second verifier of the proof files of a toy batch, has the
  block as its routine `claims_leave`, `run_honest.sh` calls `proof_check.py` on honest and altered
  proof files, and the script that makes the false files (above) also on false ones (by a list that
  is not in this repository). That routine was written from the comments and lines of the C++
  headers, the whole of `batt_shown.h` among them, and not from part 8; so it is not independent of
  these headers and can share a misreading with the C++ programs.
- The parameters' bytes have not been derived from the public files a second time (section 1).

---

# Part 6: The key file and the block WF (the committer's proof that the entries of the committed matrix are in their ranges): what the verifier does, as mathematics and bytes

This part follows the programs' source: `blocks/frame/check_key.h`, `key_fixed.h` and `wf_frame.h`
in the same directory, `blocks/iop/wf.h`, and their tests.

This part covers the file's order, the three chains' first queries, and WF. It does not repeat the
blocks that they call: SEND and BAT (part 1), the zero-knowledge sumcheck (part 2), FOLD and BASE
(part 3), the code switch and the proximity test (part 4), LKP (part 7), BATT and the member's first
part (part 8; a member is a request of a batch), LIN with FOLDEQ, the record and the binding tree
(part 9). Where they enter, it says what goes in and what comes out.

## 0. What a key file says, and what it does not

A committer commits once to a table: 2^v rows, each a code word of the table's code, under one root.
The table holds a matrix M of matrix_rows x 2^k_col entries of F (a model's weights in digits). The
key file is one file which is designed to show (a) ranges: certain public linear maps of M's
entries, the "keys" of lookups, are in range tables; (b) that the committed word is close to one
code word, within the distance of the one-time test, and that (a) is said of that word's matrix. (b)
is LIN, run as for a batch of one request.

What a yes says: under the file's root is such a table, under these public files. It says so up to
the error that the key file's plans count at the public files' target (with "yes with proof of
work": counting the proofs of work), and on the protocol's analysis, which no program proves. More
exactly: every entry of the committed matrix is in the range that the public parameters name for
single entries, and every further lookup that they name holds on the rows that they name. Which
ranges and lookups these are follows from the public files by a rule of the programs (5.2, 5.6).

No verifier judges whether they are the right ones for a model: not the width d and the offset of
the covering lookup, not whether a second lookup (on weights of two digits) is there at all or
reaches all the rows that need it. No line of a public file names a range: width, offset, weights
and rows of each lookup are written by a routine of the calling program from the files' counts
(`frame/tests/toy_file.h` for the toy runs, `frame/wf_units.h` at full shapes). They are bound in
the first query of the key file's one-time chain, as its parameter `ranges`; the configuration's
digest, and so the commitment ID, is a digest of the files' bytes and does not hold that routine. So
anyone who relies on a commitment ID must judge the public files and the program's rule (or the
bytes of the key file's first query) for themselves.

A yes also says nothing about whose weights they are: no item of a chain names a model, since no
step could check it; that a commitment belongs to a model of some name is its publisher's word,
outside the protocol.

What a verifier keeps of an accepted key file: the table's shape, the configuration's digest, the
root, the record's stored answers, and under what the key file was checked (`TestedTable`). Every
later proof on the table is judged under that. A `TestedTable` holds the distances at which the key
file's one-time test was sized, that test's setting (target, proof-of-work bits) and the public values under which the key file was checked, and it passes them on
inside one process only. What this does not cover is a limit, first of all for a verifier that checks
key file, leaf file and proof in separate processes, as `cp_verify` does (part 9, section 12).

## 1. Where everything public comes from

One routine, `configure`(public files, a routine that reads the sub-proofs' shapes from them), makes
one value, `Configured`: the configuration's digest (a hash of the public files), all codes, shapes
and plans. The committer's `make_key`(`Configured`, the holder of the table, two objects that do the
committer's heavy computations, engine, randomness, nonce search, file) and the verifier's
`check_key`(`Configured`, file, `key`, `why`, `notes`, `place`) take that and the file and nothing
else. `configure()` itself rejects, by name, public files under which no key file can be made or
that miss a condition of hiding, so a committer learns it before it commits to a table. The words of
such a rejection begin with the part that rejected, for example: `ranges:` .. for what section 5.6
rejects (it is asked where the ranges' numbers are made), `LIN:` .. for the test on the table,
`key:` .. for what is left to the key file's own question (item 2 of section 6: the binding tree,
the message's number, the index-mask message), and `hiding:` .. . Others are `once-ranges:` ..
(`frame/wf_frame.h`) and `once-own:` .. (`frame/files.h`), after the groups of these names in the
parameter file.

Why: the programs keep to the rule that every public fact has one home. The matrix's size is in
LIN's shape alone, and WF takes it from there; what LKP is told is derived from bound objects by a
routine without captured state; and the verifier asks whether every row is covered (5.6). Suppose
that a caller could hand in parts of the public description by itself, and that one fact, such as
the matrix's number of rows, were in two of them. Then the two could be set apart by one, and a
false key file would be accepted: the last row would never be looked at.

`wf_matrix_of` takes the rows from LIN's shape alone, and `wf_fault` asks whether every row
is covered (`iop/wf.h`). (from the programs' source) At toy size, in the lookup's test, a second description of one public fact, with nothing that
ties the two, let a verifier's side accept keys that are in no row of the table: part 7, 4.8.

## 2. The file, byte by byte in order

One verifier goes from the first byte to the last; no part has its own length as a word.

1. The head, 8 bytes: `CPIK`, then the format's version, four bytes: `H`, `8`, a digit for the
   field, a digit for the hash (R13.1). All 64-bit words in the file and in the items below are
   little-endian.
2. The table's root, 32 bytes.
3. The record's stored answers: 2^v * s_M elements of E (24 bytes each, canonical), where s_M is the
   shape's number of stored answers. By row, within a row by answer. Their number
   is not in the file as a word (the verifier knows it from the shape). The record's seeds are in no
   file.
4. The one-time chain: its 32 random bytes, then its messages, nonces and openings as its plan says
   (section 5).
5. The path: the slot, a 64-bit word 0, then the salt of a leaf of the binding tree. No sibling:
   that tree has one leaf.
6. The shared chain: no random bytes (the binding tree's root is used instead of them).
7. The second chain: its 32 random bytes, then its messages. The file ends with it: a byte left over
   is a rejection.

## 3. Before any chain

The verifier takes root and stored answers from the file. From `Configured` it has the table's shape
(six words: the code's three; v; the number of stored answers; the number of bytes of a leaf's salt)
and the configuration's digest. It tests form: the head's kind and version, canonical elements, the
number of stored answers, the shape (`index_key_fault`).

    name := root | the configuration's digest | the shape's six words

The name has 112 bytes; it is computed, and is in no file. It is read in two places. One is the
statement of the record's own chain, from which the record's seeds come (part 9). The other is the
commitment ID (`published_id`): one hash, under a number of its own (R4.11), of the configuration's
digest | the name as a string of bytes (the word 112, the bytes) | the list of the elements of the
record's seeds | the list of the record's stored answers | an empty list (each list: its length,
also when it is 0; the elements).

These steps alone are also a routine of their own, `read_key_table_unproved`: head | root | stored
answers -> the table, with every test of form and without the proofs. `check_key` begins with it.
Anyone else who takes a table from it relies on another run of `check_key` on the same bytes.

## 4. The three chains and their first queries

Every first query holds the version's item and the plan, then the items below in this order. P =
parameter, S = statement. These are on the format page, not in this part: how an item is encoded
(its name as a string, the length of its bytes, the bytes), that the plan's bytes are not an item,
that all parameters enter as one digest and the statement as bytes, and that every word is
little-endian.

- One-time chain, the chain's label 0x46524d02 (`bind_one_time_chain`): P `configuration` (digest) |
  P `index-mask message` (a notice) | P `ranges` (5.1) | P `binding tree` (two words: depth 0; the
  number of bytes of a leaf's salt) | P `codes` | P `the test on the table` (LIN's shape) | S
  `table`: the root as a string of bytes (the word 32, the 32 bytes), then the list of the record's
  stored answers (its length, the elements). No other item of statement.
- Shared chain, the chain's label 0x4c494e01: P `configuration` | P `codes` | P
  `the test on the table` | S `table` | S `the binding tree's root` (digest), which is also used
  instead of the chain's random bytes. The binding tree's root is its one leaf's digest; that leaf
  holds the one-time chain's seal and the salt of item 5 of section 2 (part 9; the format page).
- Second chain, the chain's label 0x4c494e02: P `configuration` | S `the seal of an earlier chain`
  (digest): the one-time chain's | S `the seal of an earlier chain` (digest): the shared chain's.
  (Two items of one name, in this order: `Binder::follows`.)
- A notice of an oracle message: six 64-bit words (code, mask code, main rows, degree of the main
  rows over F, mask rows, answers), then the leaf class as a seventh. One shape has one encoding:
  where there are no main rows the code's word is "no code", 2^32 - 1, and the degree's word is 1;
  where there are no mask rows the mask code's word is 2^32 - 1. (So for the index-mask message,
  which has mask rows alone: 2^32 - 1 | its mask code | 0 | 1 | its mask rows | its answers | its
  leaf class.)
- One session: the three channels of a verifier share one record of "one value at a vertex" for each
  root, and the verdict is read from the outcomes of all three chains (`VerifierSession`).

*Choices of the programs.* The committer's one-time test reuses the three-part form of a batch:
first part, shared part, second part; a binding tree of one leaf.

## 5. The one-time chain, block by block

- (i) SEND of the index-mask message: one oracle message of mask rows alone (oracle number 0), which
  gets an out-of-domain sample as in part 1 with one table.
- (ii) WF (5.2 to 5.5).
- (iii) If WF leaves more than one pending row: BATT joins them (it takes 1 + 2 k_col entries of the
  index-mask message's ledger). Then LIN's first part: SEND of the vector W (`member_first_verify`;
  part 8). In: the pending rows of 5.4 in their order. Out: LIN's kept claim.
- (iv) The seal, and the chain's end.

### 5.1 The item `ranges` (`WfPublic`) as bytes

All integers are 64-bit words; a list is its length, then its items; an element of F is 8 bytes.

- The item: list of lookups | the oracle message's notice | its number on the channel (1) | list of
  stages of WF's proximity test (each four words: code, mask code, k, h) | one word of choices
  (whether a fold's fresh mask rides with its stage's rows).
- A lookup: d | mu | live | list digit_weight (F) | offset (F) | padding_key (F) | list of parts |
  list first_bound (words). (Its name is not hashed.)
- A part: log_places | list of runs.
- A run: first_place | first_row | places | stride.
- k_col and matrix_rows, used below, are read from `the test on the table` (`wf_matrix_of`):
  matrix_rows is a word of LIN's shape; k_col is not a word of its own: k_col = k_lo + v, LIN's
  number of low column variables and the table's v (part 9).

### 5.2 What a lookup says

A part has 2^log_places **places**. Place first_place + p (p < places) of a run is **covered**; its
key in column c < 2^k_col is

    key = offset + the sum over q < (number of digit weights) of digit_weight[q] * M[first_row + p + q * stride][c].

A place in no run is **padding**: its key is the public padding_key, which is a row of the range
table; it is not counted in the multiplicities and reads nothing of M. Key number within the part:
c + 2^k_col * place. Part i has v_i = log_places_i + k_col index bits.

The range table is the range 0, 1, .., 2^d - 1. A multiplicity has "live" bit planes among 2^mu
plane slots (live = 0 means all 2^mu; live = 2^mu is rejected: one shape has one encoding).

The planes follow from the parts where the frame makes the parameters (`wf_lines`, `wf_planes_of`):
planes = the least number >= 2 that exceeds every v_i and with which 2^planes exceeds the number of
all keys of the lookup, padding among them; mu = max(1, ceil(log2 planes)); live = planes, or 0 if
planes = 2^mu. They are in `ranges` as words; a verifier takes them from there, and 5.6 asks that
they suffice.

first_bound fixes the order of the rounds of each part's last key level (for a fast prover): the
sumcheck there binds the variables that first_bound names first, in its order, then all others from
the highest down. Empty: all from the highest down.

What LKP is told (`wf_lookups`): the parts' v_i; d, mu, live; that it reads oracle 1; first_bound;
and its padding as a list of stretches, by part and rising: for every stretch [p0, p1) of places
that no run covers: first key p0 * 2^k_col, length (p1 - p0) * 2^k_col, the table row padding_key,
step 1. No count is told: LKP computes it. Everything else that LKP's public object has is at its
plainest: one function for the keys, one table part, which is the range table itself (no rows are
listed), no trees' numbers, no shift (so LKP samples no further element for one): part 7.

### 5.3 WF's messages

SEND of one prover's message. Oracle number 1 has main rows over F with the multiplicities' live
planes as bits, and mask rows over E. If masks ride, the fresh mask of the first fold of WF's
proximity test goes with it as oracle number 2, with s = 0 (so it gets no out-of-domain sample: part
1). Seed, answer and empty round as in part 1 (both tables).

Where a bit is (`wf_layout`). Let planes = live, or 2^mu if live = 0. For each lookup in turn cut
its planes 0 .. planes - 1 into aligned blocks, the largest first: from first = 0 take the largest b
with first + 2^b <= planes and 2^b dividing first; block (first, b); first += 2^b. A block has
2^(b + d) entries: entry (plane - first) + 2^b * t is bit number "plane" of the multiplicity of
table row t. Sort all lookups' blocks by size, largest first, ties in the order (lookup, block).
Place them one after the other from (row 0, place 0); here a place is an index within a row's
message (part 1, section 0), not a place of a part (5.2). A block that does not fit into what is
left of a row begins the next row at place 0. Index in the main table: row + 2^(row bits) * place,
row bits = ceil(log2(the message's main rows)). All other entries are 0 and no row reads them.

The group `ranges` has the one member oracle 1.

For each lookup in turn: LKP (part 7). It samples alpha, runs the tie of the two roots, the parts'
roots, each part's key levels, the table side, the leaves and the bit layer, and adds its rows to
the group itself. Its mask entries come from oracle 1's ledger in order, but the two inner masks of
the last key level of each part come from oracle 0's ledger (the index-mask message): two for each
part, in the order of the parts.

- How a vector on (mu plane variables low, d row variables above) is read on the live planes
  (`wf_planes_at`): for each block (first, b) of this lookup keep the b low plane variables and the
  d row variables, and put the plane variables b .. mu - 1 at the bits of "first" (their factors go
  into the scalar); the result, on b + d variables, is at the block's place.
- Out of LKP for each part: a point z in E^(v_i), weights w, two mask terms on oracle 0 and a
  target, saying: the sum over key numbers i of w[i] * key[i] + <masks> = target, with w[i] = eq(i;
  z) at a covered place and 0 at a place of padding: LKP has moved the padding's share into the
  target itself.

Then WF's own proximity test on the group (BAT, then folds and code switches by the stages, BASE),
with the first fold's mask as sent. Rows batched by BAT: the sum over the lookups of [(planes - 1) +
(parts - 1) + the sum of the v_i + d + 4], and the sample rows (`wf_rows_batched`).

### 5.4 The pending row of a part (`wf_pending_row`): what WF passes on

z = (z_col | z_place): the first k_col coordinates, then log_places. e[p] := eq(p; z_place). covered
:= the sum of e[p] over the covered places.

The row says

    < g (x) eq(. ; z_col), M >  +  <the claim's two mask terms>  =  target,   with
    g[first_row + p + q * stride] += e[first_place + p] * digit_weight[q]   for every run, p < places, q;   g = 0 elsewhere;
    target = (LKP's target) - offset * covered.

(The weights eq(. ; z_col) add up to 1, so a constant in a key contributes constant * e[place].)

A stop on both sides, not a check of a proof: the sum of all of LKP's weights w must equal
"covered", else the process ends: LKP's padding and WF's runs would speak of different places.

Order of the pending rows: lookup by lookup, part by part. They are passed on all at once; a caller
that drops them ends the process.

### 5.5 What WF checks on the spot: nothing

Everything is deferred into rows of its own group, which its proximity test closes, and into the
pending rows, which LIN closes. So WF alone says nothing about the committed matrix: all that ties
the lookups' keys to M are the pending rows. WF ends the process (a fault of the configuration or of
the caller, not of a file) if 5.6 finds the parameters unfit, if oracle 0 is not an oracle message
of mask rows alone whose out-of-domain sample is answered, if its own message has another number of
answers than the setting's `samples`, or if its message does not get the number 1. By the comment
before `wf_start_check` in `iop/wf.h` the stop on oracle 0 is there so that no challenge under which
a value of the matrix is shown is sampled before the index-mask message is there with its
out-of-domain sample answered: a mask that is chosen after its challenge can be fitted to any false
value.

### 5.6 Which parameters are rejected (`wf_fault`)

The question is asked by `configure()`, by both sides of WF and by the key file's routines.

- In themselves: no lookup; the message has not main rows over F and mask rows, or has s = 0; mu
  outside 1 .. 6, live >= 2^mu, fewer than 2 or more than 62 planes, d < 1, d + planes at least the
  field's bits; no digit, no part; padding_key >= 2^d; first_bound names a variable twice; a part
  with more than 2^40 places; runs that overlap, fall, begin beyond or reach beyond their part; a
  run or a part that covers no place; a stride other than 0 where a key has one digit; a digit of
  weight 0 in any lookup (its rows would enter no key); a stride 0 where a key has several digits
  (they would be one row).
- Against the matrix: no rows, more than 2^40 rows or columns; a part with v_i < 1 or v_i >= planes;
  first_bound names a variable >= v_i - 1 of some part; a run whose last digit's last row is beyond
  the matrix; more keys in a lookup than 2^planes - 1 can count; no lookup whose key is offset + one
  entry (one digit, of weight exactly 1) and whose runs reach every row of the matrix
  (`wf_covers_all_rows`): this line carries the reading "every entry is in range".

  Why exactly 1: under a weight w the key is offset + w * entry, and an entry w^-1 * (a small
  number), a huge element of the field, would pass as "in range". Under the weight 0 every key is
  the constant offset, and a key file would be accepted for any matrix.

  `wf_fault` rejects a digit of weight 0, and the covering lookup needs the weight 1 (`iop/wf.h`).
  (from the programs' source)
- Against the codes: a code that is not there; a block of planes longer than a main row; another
  number of main rows than the layout fills; mask rows without room for the lookups' own masks and
  the pad; no stage at all, or a first stage with other codes than the message's; what the proximity
  test's own question rejects.
- At the entry points it is also asked that the message's number of answers is the setting's.
- In a key file LIN's own question comes first and is stricter on the matrix's size (at most 30 row
  variables, 30 low column variables, 60 column variables), so the bounds of 2^40 above never decide
  there. Where the frame makes the numbers (`frame/wf_frame.h`) it rejects more, by words of its
  own. Among them: d above 40; a mask row no longer than the pad; a part that would read more rows
  than the matrix has; a lookup of more than 256 parts; planes that the caller's shape names itself;
  a block of planes that is longer than a main row; no number of mask rows with room for the
  lookups' masks. The format page's 8.4 has the conditions on the parameter file. (from the
  programs' source)

*Choices of the programs.* WF sends one oracle message with all multiplicities as bit planes and
runs a proximity test of its own on it, inside WF, before the joining. Lookups are cut into parts
with padding, and the number of planes is not a free parameter of the public files (it follows from
the parts, 5.2). There is one multiplicity vector for each lookup over all its parts, and no other
cut than by rows. The question whether every row is covered is a check of the programs' own.
first_bound is the programs' own too. An honest committer whose matrix has an entry out of its range
writes no file at all (`make_key` says no).

## 6. `check_key()`: the order of its tests, its verdicts and its words

1. The public files were admitted by `configure()` (else no: `the public files are not admitted`).
2. The key file's configuration (`key_fault`): LIN's and WF's questions (5.6, with the matrix's size
   from LIN's shape); a binding tree of one leaf; WF's message has the number 1; the index-mask
   message is mask rows alone, has s > 0, and has room for 2 * (all parts) + (1 + 2 k_col if there
   is more than one part) entries.
3. All three plans meet the target. (Then one stop that depends on no byte of a file: the three
   plans are under one setting, else the process ends.)
4. Head, root, stored answers (section 3).
5. One-time chain: SEND | WF | BATT and LIN's first part | seal | the chain's end.
6. The path -> the binding tree's root.
7. Shared chain: LIN's shared part (the fold point, the folded row, the columns opened under the
   root of the file, ..), its proximity tests (one or two, by the codes), seal, its end; every group
   of the shared chain's own statement has been tested (else the process ends).
8. Second chain: LIN's second part, its proximity tests (one or two), its end (no byte is left).
9. Every group of the one-time and second chains' statement has been tested (else the process ends);
   the session's verdict.

Three verdicts: no; yes; yes with proof of work (at least one round of the key file's chains needs
its proof of work, in the sense of `docs/format/08-parameters.md`, 8.4, item 22). In the source:
`ProofVerdict::NO`, `ProofVerdict::YES`, `ProofVerdict::YES_WITH_WORK`. These are the library's
verdicts. `cp_verify` has no ending of its own for the third: in a call without the option
`-no-proof-of-work`, proof of work that is counted changes neither the last line nor the exit code;
with it, a call in which some round needs its proof of work ends at best with `NOT-A-VERIFICATION`
(the format page's 14.3, 14.4).

`why` (the words of a rejection: a text, not words of 8 bytes): inside a chain the channel's own
words where it has any (they read alike in every chain: for example
`BASE: the masked message in the proof does not satisfy the linear claim`), else the value of
`place`. Outside the chains it has words of its own: `the public files are not admitted`; those of
`key_fault`; `the parameters are not admitted (the first cause is on stderr)`; the head's
(`file-too-short`, `file-of-another-kind`, four that begin `version-`); `the table's root`;
`the record's answers`; what the test of the table's form says; `the seal's place`;
`a chain of the key has refused`.

`place`: where in the file's order the rejection fell: `before the chains`,
`one-time chain: the index-mask message`, `one-time chain: the ranges`,
`one-time chain: the rows on the table`, `one-time chain: its end`,
`between the one-time chain and the shared chain`, `shared chain`, `shared chain: a proximity test`,
`shared chain: its end`, `second chain`, `second chain: a proximity test`, `second chain: its end`,
`behind the chains`.

`notes`: values that the blocks of all three chains note on the way, for a second verifier. No
decision reads `place` or `notes`.

## 7. Examples and malicious committers (all at toy size; synthetic data; test randomness, so nothing is hidden in these files)

*From two public files,* the toy run `key-and-entry` (`frame/handover_files.cpp`, and a test of the
key file's check that is not in this repository; the files: `frame/tests/toy_file.h`,
`toy_shapes.h`): a matrix of 100 rows x 64 columns at p = 2^64 - 2^32 + 1 (field 2; the programs of
this repository can be built for field 2 alone, `blocks/base.h`); toy bytes -4 .. 3 in every
row, toy weights low + 8 high in -16 .. 15; the real target of 128 bits under the frame's own plans,
with proofs of work. The parameter file's line `work positions 4 6 8` gives the proof-of-work bits
before a sampling of positions in a request's own rounds, a batch's shared rounds and the
committer's rounds. All three chains of a key file are the committer's, so each of its rounds that
samples positions has 8. The key file has 503,744 bytes (sections 2 and 3); verdict: yes with proof
of work. One stored answer for each row (s_M = 1).

*The smallest whole key file* (`frame/tests/small_key_test.cpp`): 4 rows x 4 columns, one lookup,
hand-written plans. The test prints the files' lengths; by its head a round's repeated positions are
opened once, which shortens a file, and no round here draws two. WF alone without its proximity
test, for tests of hiding: `iop/tests/wf_small_world.h`.

*Malicious committers,* each of which must be rejected. Those named here were made and run at toy
size, on synthetic data, with other programs; section 6 gives what they came to.

- Through `make_key()` under the public files (builds of `frame/tests/false_keys_test.cpp`, which in
  this repository is the honest control alone; the first three below are made, and are rejected at
  BASE's linear claim). `KEY_ENTRY_OUTSIDE`: a byte just outside its range in the table and in all
  that the committer computes, counted at row 0 of its table so that the numbers add up: it is
  rejected in the one-time chain, at WF's own test. `KEY_OTHER_MATRIX`: the committed matrix is out
  of range while WF, the joining and the products run on another matrix, in range: it is rejected in
  the second chain's test. `KEY_PART_CLAIM_LIE`: other end values with the same product in one
  part's last key level, so that only that part's pending row is false: it is rejected in the second
  chain's test; it is the one that only the seam between WF and the table can catch.
  `KEY_ENTRY_OUTSIDE_THEN_FIT`: the committer of `KEY_ENTRY_OUTSIDE` also follows the reader's
  running claims, so it knows by how much a carried claim is false, and mends it at one code switch
  by moving the carried mask row by a whole code word (`CS_FIT_MASK_WORD`), at every code switch of
  the commit run in turn, 8 in all: it is rejected at each, in the one-time chain, at WF's own test.
  `check_key` takes plans and codes from one `Configured`, and the stops `require_group` and
  `require_codes` (`chain/rounds.h`) stand beside that.
- Also whole key files under the public files, each rejected at BASE's linear check. In the one-time
  chain, at WF's own test: `WF_KEY_OUTSIDE` (one cell of the committed matrix just outside its range),
  `LKP_WRONG_ROOT` (the range proof uses a wrong root for each look-up table). In the shared chain, in
  the test on the folded row's group: `FOLDEQ_OTHER_ROW` (another folded tail), `FOLDEQ_OTHER_PAD` (a
  folded pad raised by one), `LIN_OTHER_OMEGA` (a false Omega). In the second chain's test:
  `MEMBER_OTHER_SUM` (one claim, on the row weights, has a false target), `MEMBER_W_OFF_THE_POINT` (W
  changed by a vector that vanishes at the claim's point, so that only the other claim on W is
  false), `LIN_OTHER_W` (a false W).
- `BASE_WRONG_WORD`: no row is false: the last step shows a masked message that meets the linear
  claim and is not the underlying message of the opened symbols. It is rejected at BASE's check at
  the positions, in the one-time chain, at WF's own test.
- `KEY_ROOT_OF_ANOTHER_TABLE`: an honest key file with the root and the stored answers of a second
  table written into the file. It is rejected at a proof-of-work test of the one-time chain.

None of these malicious committers has run above toy size. On malicious committers at full size:
the format page's 17.5, item 32.

## 8. Limits of this part

What this part does not cover:

- Section 5.6 (which parameters are rejected) has no known second implementation.
- Nothing in this part has been run by a second verifier above toy size.
- That the checks suffice is a matter of the protocol's analysis and is proved by no program.
- The orders that must hold in WF, by comments in `iop/wf.h` (before `wf_prove`, `WfPending` and
  `wf_start_check`). They are the source's arguments, not proofs. Everything of WF's public object
  is in the first query, as the item `ranges`. (The public object of a lookup inside WF is another
  object: it is derived from that item and from the item `the test on the table`, and is in no item
  itself.) The multiplicities are sent, and their out-of-domain sample is answered, before the first
  alpha is sampled: the lookup's first error term rests on this order. One call does all three, so
  that a caller cannot write it otherwise, and no test runs a prover that knows alpha first: by the
  source such a prover cannot be run. No challenge under which a value of the matrix is shown is
  sampled before the index-mask message is there with its out-of-domain sample answered (5.5). Every
  pending row, with its target, is fixed before the caller samples the challenge that joins the rows
  (the malicious provers on the routines that the caller then calls: `BATT_TARGET_AFTER_X`,
  `MEMBER_W_AFTER_POINT`, `LIN_TARGET_AFTER_BATCHING`). (from the programs' source)
- The bytes of `codes` and of `the test on the table` are in part 9 and on the format page, not in
  this part.
- The numbers of trees and their order across the SENDs of one chain, and the labels of leaves (of a
  stored tree, of the binding tree), are on the format page. This part does not trace them.
- Hiding. The multiplicities are a function of the weights; nothing but mask rows, the pad of the
  out-of-domain sample, the random tails and, for what is not opened, the salts of the leaves is
  there to hide them. One condition of hiding no verifier can see: a run of the committer that has
  broken off is never taken up again with the same mask messages. By the comment before
  `wf_index_masks` in `iop/wf.h`, two values of the matrix that are shown under one mask entry give,
  subtracted with the right factors, a linear function of the matrix without any mask. `make_key`
  samples all its randomness itself, and a second call samples it again (`frame/check_key.h`). The
  committer also has a self-check: right after the index-mask message is sent, the process ends
  unless the message, but for its pads' positions, and its tail have the values that were sampled
  for them (`own-test-index-mask`). The message of the multiplicities has a self-check of the same
  kind: right after it is sent, the process ends unless its mask rows and their random tail have the
  values that were sampled for them (`own-test-ranges-mask`, in `iop/wf.h`); of a fold's mask that
  is sent with it, this check compares the random tail alone, and FOLD's own self-check compares the
  hiding polynomial's coefficients too, each only if the generator numbers its sampled values
  (`own-test-fold-mask`, in `iop/fold.h`; its other limits: [COIN-KEYS.md](../COIN-KEYS.md), section
  3). By the source the first two comparisons, on the index-mask message and on the message of the
  multiplicities, read the tables as the prover keeps them, and they have these limits. They do not
  see the copies that are handed to the engine for the encoding, which are made before the
  comparison: a value that is lost in such a copy, or in the encoding, passes. They do not look at
  the pad's places. A table whose sampling nobody noted is not compared, and nothing says so; the
  same holds of every table under a generator that does not number its sampled values, so a program
  that relies on these comparisons has to test by itself that its generator numbers them
  (`not_as_drawn` in `iop/send.h`). (from the programs' source) Exact tests of hiding on WF alone
  and on the smallest whole key file are in `blocks/tests/hiding/` (`wf_exact_test.cpp`,
  `key_exact_test.cpp`).
- A key file at full size (about 3 million rows x 2^18 columns): this part has
  not been compared with such a file, and no second verifier has read one (`docs/format/FORMAT.md`,
  17.3).

---

# Part 7: LKP, the lookup by product trees

This part gives what the verifier does in the block LKP, as mathematics. It follows the programs'
source: `lkp.h` in `blocks/iop/`, and the tools that it uses. Two words have a second sense here. A "key" in this part is an
entry of the vector that is looked up, never the key file. "Part i", "part p", with a letter, is a
part of the keys, and a "table part" is a part of the table; "part 6", with a number, is a part of
these pages.

## 0. Conventions

- All scalars are elements of E. The index rule: bit 0 is variable 0 and is the least significant
  bit. eq(z, i) = prod_t (z_t if bit t of i is 1, else 1 - z_t). For a vector u of 2^a entries:
  u~(z) = sum_i eq(z, i) u_i.
- A **mask term** is (an entry m of some oracle message's mask rows, a coefficient). A **mask form**
  L is a list of mask terms; <L> = the sum of coefficient * entry.
- A **map** V is a vector of 2^a entries as above, read through V~; with a = 0 it is one scalar, and
  its point is empty. A **point claim** on a map V is (point z, mask form L, value y). It says
  V~(z) + <L> = y (`struct PointClaim`).
- A **row** is a linear equation with public coefficients and a public target on main tables and
  mask entries (part 1).
- A **ledger** gives out the free entries of one oracle message's mask rows by increasing index,
  each once (`MaskLedger::take`: with R mask rows placed in a table of 2^ceil(log R) rows, the
  indices on a row of the table that is not a real row are skipped). "Take n" means: the next n
  entries.
- In a list of messages, "P:" is what the prover sends and "V:" is the verifier's message.

## 1. The two tools

### 1.1 PSC(vars, L_in, T; S)

The routine is `psc_verify`; its rows are made by `psc_rows`; part 2 has PSC whole.

*In.* k >= 0 variables in the order of the rounds. The variable of round j has a weight point z_j
(it is summed as g -> (1 - z_j) g(0) + z_j g(1)) and a number d_j in {2, 3} of coefficients. Then a
mask form L_in and a target T. Then a summand S with s shown maps of total degree D (s = 0 exactly
if D = 0) and at most one kept map. (d_j in {2, 3} and at most one kept map are LKP's uses: the
routine takes any d_j >= 2 and any number of kept maps, and BATT has J of them, part 8. With k = 0
and s = 0 there is nothing to sum over and nothing to show, and the process ends. From the programs'
source: `iop/hp.h`, `iop/summand.h`, `iop/psc.h`.) A **shown map** is one whose value at the end
point the prover sends under a mask (v_i below). A **kept map** is one whose value is not sent: it
stays in row (U). The claim is:

    the sum over x in {0,1}^k of prod_j eq-factor(z_j, x_j) * S(x, maps at x)  +  <L_in>  =  T.

*Fresh entries.* There are n = 1 + sum_j (d_j - 1) + D + s of them, in this order (`psc_masks`,
`hp_place`):

- for round j = 0 .. k-1: the coefficients of X_j^1, .., X_j^(d_j - 1); then
- if s > 0: the coefficients of Y^0, Y^1, .., Y^D, then the inner masks c_1, .., c_s;
- if s = 0: there is no Y, and the last round's variable has its constant too: its entries are those
  of X_(k-1)^0, X_(k-1)^1, .., in this order (only the last variable of the list has a constant).

They are named w_(j,t), w_(Y,t), c_i.

*Messages.*

| the prover sends | the verifier's message | round |
|---|---|---|
| mu' (1 scalar) | eps, one element, may be 0 | `PSC epsilon` |
| for j = 0 .. k-1: q_(j,1), .., q_(j,d_j-1) | gamma_j, one element, may be 0 | `PSC round` |
| if s > 0: p_1, .., p_D | beta, one element, never 0 | `PSC Y` |
| if s > 0: v_1, .., v_s | nothing | `PSC values` (a round of its own, in which the verifier says nothing) |

*The verifier computes:*

    R_0 = mu' + eps T.
    q_(j,0) = R_j - z_j (q_(j,1) + .. + q_(j,d_j-1));   R_(j+1) = sum_t q_(j,t) gamma_j^t.
    If s > 0: p_0 = R_k, R_end = sum_(t=0..D) p_t beta^t; else R_end = R_k.

(This is `Weight::complete` with the moments (1, z, z, ..) of an equality weight, and (1, 0, 0, ..)
for Y.)

*It checks* nothing algebraic: the round checks are used up to get the left-out constants back. It
fails only if the channel rejects.

*Out.* gamma = (gamma_0 .. gamma_(k-1)), by round. (A_0, A_1) = S at (gamma, v), where S = A_0 +
A_1 * (the kept map's value).

    Row (U):   sum_(j<k) sum_(t>=1) gamma_j^t w_(j,t)  +  [s > 0: sum_(t=0..D) beta^t w_(Y,t)]  [s = 0: + w_(k-1,0)]  +  eps <L_in>  +  eps A_1 (kept map)~(gamma)   =   R_end - eps A_0.
    (E_i), i = 1..s: the point claim (gamma, {(c_i, beta)}, v_i) on shown map i.

(In the source every way of summing phi comes with a polynomial psi for which phi(psi) = 1. All
weights here have psi = 1, so the products of psi in `hp_form` and the factor of <L_in> are 1.)

*The errors that the plan counts:* 1/|E| for eps; (d_j - 1)/|E| for each round; D/(|E| - 1) for Y.

### 1.2 LINE(claim_0, claim_1), both at one point z

The routine is `line`. V: rho, one element, may be 0; round `LINE`. Out comes a point claim on the
map whose two halves (top index bit 0, 1) the two claims speak of:

    point (z, rho): rho is appended as the highest coordinate;   mask form (1 - rho) L_0 + rho L_1, the terms of L_0 first;   value (1 - rho) y_0 + rho y_1.

The error that the plan counts: 1/|E|.

## 2. What is public (`struct LkpPublic`), and its canonical bytes

*The bytes.* Every integer, bool and enum is one 64-bit word (a negative int is its two's
complement: function = -1 is 2^64 - 1). An element of E is three canonical words of F. A list is its
length as a word, then its items (the routines `public_bytes`). How a word becomes bytes is on the
format page (low byte first).

    LkpPublic = list parts (words; at least one) | d | mu | live | h | g | shifted | list read_from (oracle numbers, words) | list keys | list tables | list padding | list last_key_level_order (words)
    a key block = first | log_trees | function | padding_row
    a table part = first | log_rows | function | list rows (E)
    a padding (a run of places) = part | first | log_step | length | row   (five words)

`lkp_verify` takes the object as a `Public<LkpPublic>`: a value that the chain's `Binder` has
written into the first query as a parameter, or one that a routine without captured state has
derived from such values (`derive` and `element` in `chain/public.h`). The second is the key file's
case: there the object is derived from the items `ranges` and `the test on the table` and is in no
item itself (part 6, section 1).

*Derived.* planes = live if live != 0, else 2^mu. w = 2^mu is the number of **plane slots**. (v_0,
.., v_(k-1)) = parts. k = the number of parts. N = sum_i 2^(v_i) is the number of all keys. (From
here on k, v_i and w have these senses; inside a PSC the letters of 1.1 hold, and w_(j,t), w_(Y,t)
are always mask entries.)

*The table* has 2^d rows tau(t) in E, after the shift theta (theta = 0 if not shifted) (`lkp_row`,
`lkp_rows`):

    h = 0: the table part (first, log_rows, f, rows) holds the rows first .. first + 2^log_rows - 1:  tau(first + i) = own(i) + f theta;
    h > 0: table part number c = first holds the rows c + 2^h t, t < 2^(d-h):  tau(c + 2^h t) = own(t mod 2^log_rows) + f theta;
    own(i) = i (as an element of F) if the list rows is empty (a range table), else rows[i].

*Its extension* at x in E^d (`lkp_table_at`) is tau~(x) = the sum over the table parts of W *
(own~ + f theta), with

    h = 0:  W = prod_(j = log_rows .. d-1) (x_j if bit j of first is 1, else 1 - x_j),   own~ at (x_0 .. x_(log_rows-1));
    h > 0:  W = prod_(j < h) (x_j if bit j of first is 1, else 1 - x_j),                 own~ at (x_h .. x_(h+log_rows-1));
    own~(y) = sum_j 2^j y_j for a range table, else the multilinear extension of rows.

*The multiplicities.* b[j + 2^mu t] in {0, 1} is bit j (j < planes) of the number of keys, padding
not counted, that equal row t (a key that equals several rows is counted at one of them). b = 0 at
the plane slots j >= planes, which are not sent. Where b is in what was sent is the caller's matter.
The caller passes in a routine, written planes(row, V) or planes() in this part (planes without
brackets is the number above), which adds <V, b> to a row for a product-form vector V (a product
form: part 8, section 0) on mu + d variables (plane variables low) and reads the live planes alone
(`LkpPlanesAt`; for the key file: part 6).

*The keys.* Part i is a vector K_i of 2^(v_i) entries that is never sent. With one part: a key's
number is y + 2^g x, where y is its tree's number; the key blocks tile the 2^g tree numbers. A key
block with function f >= 0: its keys are function f's, and LKP runs on K + f theta. A key block with
function -1: trees of padding alone, all of whose places hold tau(padding_row).

*Padding.* A run (part, first, log_step, length, row) says: the key numbers first + 2^log_step i, i
< length, of that part are **places of padding**. Each holds tau(row), the table's row as the shift
has moved it, whatever was sent there, and is not counted in the multiplicities. There is no other
description of the padding: how many places there are is a function of the runs.

*How LKP counts the padding* (`lkp_shape`, in this order): first the list padding, each run as
(length, tau(row)); then, for each key block with function -1, in the list's order: (2^(v_0 - g +
log_trees), tau(padding_row)). pad = the sum of the first entries (`lkp_padding_count`).

*A run as cubes* (`lkp_cubes`). A cube (mask, value) is the set of key numbers i with (i AND mask) =
value. With x = first >> log_step and left = length, repeat: b = the largest number with 2^b | x and
2^b <= left; the cube has the free bits log_step .. log_step + b - 1, and all other bits of the
v_part bits are fixed to those of (first mod 2^log_step) + 2^log_step x; then x += 2^b, left -= 2^b.
(This b and x belong to this paragraph alone.) The trees of a key block are the cube with the bits
log_trees .. g - 1 fixed to those of `first`. For a point z: W(cube) = the product over the fixed
bits t of (z_t if the bit is 1, else 1 - z_t) = the sum of eq(z, i) over the cube.

## 3. Tests of form, before any message

The routines are `lkp_fault`, `lkp_public_check` and `lkp_mask_messages_check`. A failure ends the
process: it is a fault of the configuration or of the caller, never of a proof.
`lkp_fault(LkpPublic)`, which is the object's `public_fault`, tests what the object alone shows. The
`Binder` calls it when the object is bound. A derived object (section 2) does not pass the `Binder`;
`lkp_public_check` calls `lkp_fault` on every object, bound or derived, when LKP begins. A verifier
that reads such an object from a file calls it first and rejects the file on a fault. Whether the
out-of-domain samples are answered is asked of the statement when LKP begins. The tests:

- read_from is not empty, and the out-of-domain sample of every message in it is answered; so is
  that of both messages that give mask entries;
- parts is not empty;
- 1 <= mu <= 6;
- live < 2^mu ("all plane slots are live" has one encoding, live = 0; live = 2^mu is rejected);
- 2 <= planes < 64;
- every v_i >= 1, v_i < planes, and N < 2^planes;
- d + planes < the field's bits (64 for p = 2^64 - 2^32 + 1: `F_BITS`);
- several parts only with g = 0 and no shift;
- g <= v_0;
- the key blocks are aligned, do not overlap and cover the 2^g numbers;
- without the shift every function number is 0 (or -1 for a key block of padding; every negative
  function number of a key block is taken as "padding": from the programs' source);
- h <= d;
- the table parts are aligned blocks that do not overlap and cover the 2^d rows (h = 0), or are
  numbered 0, 1, .., 2^h - 1 in the list's order with log_rows <= d - h (h > 0);
- a written-out table has 2^log_rows rows;
- a key block's padding_row < 2^d (tested for key blocks of padding alone: from the programs'
  source);
- there are at most 4,096 runs of padding, and for each: its part exists, length >= 1, log_step <
  v_part, its last place is in the part, row < 2^d, and with g > 0 log_step >= g (a run stays in one
  tree) and that tree belongs to a function;
- the runs are strictly increasing by (part, first), and no two meet;
- something is left to speak of: some key block has a function, and in every part, and (with one
  part) in every function's block of trees, at least one place is not padding (`lkp_only_padding`);
- last_key_level_order names no variable twice, and only variables below (the least v_i) - 1.

Why the last but one: a place of padding is not read. So under parameters that declare every place
of the lookup, of a part or of a function's trees as padding, what the block states about those keys
is void. With that test taken out of a copy of the source, the block's test `LKP_ONLY_PADDING` gets
keys that are in no row accepted under four such sets of parameters (by a comment in its code the
proofs are then expected to be accepted; neither that code nor the table that describes that copy is
in this repository). `lkp_test` asks that
`lkp_fault` rejects such sets. No program of this repository makes such parameters, and a prover
cannot choose them. Whether the right places are padding, no test of form can know: a caller derives
them from its own bound sizes.

While LKP runs, every row that it makes may read main tables of the messages in read_from alone (the
routine `leave()` at the head of `lkp_run`); a row on another message ends the process.

## 4. The run, message by message (`lkp_run`)

"own" is the ledger of the message `masks`. "key" is the ledger of `key_masks` if the caller names
one, else own.

### 4.0 theta and alpha

If shifted, V: theta, one element, may be 0; round `LKP shift`. Then V: alpha, one element, may be
0; round `LKP`.

    a = the product over the list (number of places, value) of section 2, "How LKP counts the padding", of (alpha - value)^(number of places).

Both rounds have an empty prover's message unless the caller has left a prover's message open
(section 9).

### 4.1 The tie: planes - 1 PSCs without variable (k = 0)

The hidden values are Z (the product of all alpha - key), G_j (the product of alpha - tau(t) over
the rows whose bit j is set) and P_j = G_j G_(j+1)^2 G_(j+2)^4 .. (so P_(planes-1) = G_(planes-1)).

- Step 0: PSC(no variable, L_in empty, T = 0; s = 3, D = 3, shown (Z, P_1, G_0), A_0 = v_1 - a v_2^2
  v_3). Take 7 from own: w_(Y,0..3), c_1, c_2, c_3. Row `LKP tie, step 0`: sum_(t=0..3) beta^t
  w_(Y,t) = R_end - eps (v_1 - a v_2^2 v_3). Kept: the claims on Z, on P_1 and on G_0.
- Step j = 1 .. planes - 2: PSC(no variable, (L_in, T) = the claim on P_j; s = 2, D = 3, shown
  (P_(j+1), G_j), A_0 = v_1^2 v_2). Take 6 from own. Row `LKP tie, step j`: sum_(t=0..3) beta^t
  w_(Y,t) + eps <L_in> = R_end - eps v_1^2 v_2, with L_in = {(the c under which P_j was shown, that
  step's beta)} and R_0 = mu' + eps (the shown value of P_j).

The claim on P_(planes-1) of the last step is the claim on G_(planes-1). (With planes = 2 there is
step 0 alone; its P_1 is G_1.)

### 4.1b The parts' roots, only if k > 1: k - 1 PSCs without variable

Q_0 = Z, Q_i = Z_i Q_(i+1), Q_(k-1) = Z_(k-1), where Z_i is the product of alpha - key over part i.

- Step i = 0 .. k - 2: PSC(no variable, the claim on Q_i; s = 2, D = 2, shown (Z_i, Q_(i+1)), A_0 =
  v_1 v_2). Take 5 from own. Row `LKP parts, step i`: sum_(t=0..2) beta^t w_(Y,t) + eps <L_in> =
  R_end - eps v_1 v_2.

Kept: a claim on each Z_i (a point claim with the empty point). With one part: the claim on Z.

### 4.2 The point r

V: r = (r_0 .. r_(mu-1)), one message of mu elements, each may be 0; round `JOIN`. The claim on TL_0
(the vector of the w plane roots) at r has:

    the point r;
    the value sum_(j < planes) eq(r, j) y_j + sum_(planes <= j < w) eq(r, j),  y_j the shown value of G_j;
    the mask form, for j = 0, 1, .., planes - 1 in turn: (G_j's c, eq(r, j) * its beta).

### 4.3 The key side, part by part (p = 0 .. k-1), level l = 0 .. v_p - 1 (`lkp_level_chain`)

    KL_(v_p)[i] = alpha - (K_p[i], shifted);   KL_l[i] = KL_(l+1)[i] * KL_(l+1)[i + 2^l], i < 2^l;   KL_0 = Z_p.

- In: a claim (pi, L, y) on KL_l, with pi in E^l by variable.
- The order of the rounds: order[j] is the variable that round j binds. At the levels l < v_p - 1:
  order = (l-1, l-2, .., 0). At the last level l = v_p - 1: first the variables of
  last_key_level_order as listed, then the others from the highest down (`lkp_order_beginning`).
- PSC(l variables; round j: z_j = pi[order[j]], d_j = 3; L_in = L, T = y; s = 2, D = 2, shown (the
  half with top bit 0, the half with top bit 1) of KL_(l+1), A_0 = v_1 v_2).
- Entries: 2 l + 5. At the levels l < v_p - 1: take all from own. At the last level: take 2 l + 3
  from own (the w), then 2 from key (c_1, c_2) (`lkp_fresh`).
- Row `LKP keys, level l` (with several parts: `LKP keys, part p, level l`): sum_(j<l) (gamma_j
  w_(j,1) + gamma_j^2 w_(j,2)) + sum_(t=0..2) beta^t w_(Y,t) + eps <L> = R_end - eps v_1 v_2.
- The point is put back into the order of the variables: gamma'[order[j]] = gamma_j
  (`lkp_by_variable`). LINE on the two end claims at gamma' gives the claim on KL_(l+1) at (gamma',
  rho).

Level 0 has no variable: it is a PSC without rounds in X, and its LINE gives a point of one
coordinate.

### 4.4 The table side, level l = 0 .. d - 1

The same with mu + l variables, from the claim of 4.2.

    TL_d[j + 2^mu t] = 1 + b[j + 2^mu t] (alpha - 1 - tau(t));   TL_l[i] = TL_(l+1)[i] * TL_(l+1)[i + 2^(mu+l)].

The order is always from the highest variable down. Entries: 2 (mu + l) + 5, all from own. Rows
`LKP table, level l`.

### 4.5 The leaves

- In: the claim (pi, L, y) on TL_d, with pi in E^(mu+d).
- PSC(mu + d variables, order from the highest down: round j binds variable mu + d - 1 - j; z_j =
  pi[that variable]; d_j = 3 for a table variable (number >= mu), 2 for a plane variable; L_in = L,
  T = y; s = 0; one kept map b; A_0 = 1, A_1 = alpha - 1 - tau~(gamma'_mu .. gamma'_(mu+d-1))).
- Entries: 1 + 2 d + mu from own: for each of the d table rounds (X, X^2); for each of the plane
  rounds but the last X; for the last round (variable 0) its constant, then X. There is no round in
  Y, and nothing is shown.
- Row `LKP leaves`: the sum over the rounds of gamma_j^t w_(j,t) + w_(last,0) + eps <L> + eps A_1
  b~(gamma') = R_end - eps, where b~(gamma') = <eq(gamma', .), b> is read through planes().

### 4.6 The bits are bits

V: zeta = (zeta_0 .. zeta_(mu+d-1)), one message, by variable, each may be 0; round `ZERO`.

- PSC(mu + d variables, the same order; z_j = zeta[that variable], d_j = 3; L_in empty, T = 0; s =
  1, D = 2, shown b, A_0 = v^2 - v). Entries: 1 + 2 (mu + d) + 2 + 1 from own.
- Row `LKP bits`: sum_j (gamma_j w_(j,1) + gamma_j^2 w_(j,2)) + sum_(t=0..2) beta^t w_(Y,t) =
  R_end - eps (v^2 - v).
- Row `LKP bits at the point`: b~(gamma') + beta c = v.

### 4.7 The number of keys

No message. Row `LKP number of keys`:

    the sum over j < w, t < 2^d of 2^j b[j + 2^mu t] = N - pad

as an element of F. The vector is the product form with the pair (1, 2^(2^i)) on plane variable i
and (1, 1) on every table variable.

### 4.8 What LKP passes to its caller (`struct LkpKeys`; the end of `lkp_run`)

For each part p, from the claim (z, L, y) on KL_(v_p):

    the sum over the key numbers i of the part of  weights[i] * key[i]  +  <-L>  =  target,

where key[i] is key number i before the shift, as the caller has sent it (or can read it off what it
sent).

- weights[i] = eq(z, i) at a real place, and 0 at every place of padding. As a list of product
  forms: eq(z, .); then, each with the factor -1: eq(z, .) restricted to the cube of every key block
  with function -1 (in the list's order), then to every cube of every run of this part (runs in the
  list's order, cubes in section 2's order).
- target = (alpha - y) - the sum over those cubes of W(cube) * tau(the row that its places hold) -
  theta * the sum over the key blocks b with function f_b > 0 of f_b * R_b, where R_b = W(the
  block's cube) - the sum of W over the cubes of the runs that are in the block.
- -L has two terms: ((c_1 of the last level), -(1 - rho) beta) and ((c_2), -rho beta), with beta and
  rho those of the part's last level.

The caller places the weights on what it has sent and adds nothing public of its own. It must make a
row of every such claim (the process ends if one is dropped). For a function's own key vector (index
(y - first) + 2^log_trees x), `lkp_weights_of_block` sets the tree bits log_trees .. g - 1 to the
block's number and moves their factors into the scalars.

Why: the programs' source gives this reason. In the block's test, a verifier's side whose caller had
a second description of the padding (a list of places beside a count; a number of key variables
beside the list of parts; a function beside the place's block) accepted keys that are in no row of
the table. So each public fact is in one place, and LKP itself makes the claim that it passes on.
The acceptances are kept in a test that is not part of this repository, so a reader cannot repeat
them, under two switches: `LKP_SURPLUS_PADDING` (the first) and `LKP_TWO_PADDINGS` (the other two,
as one second description). The public inputs cannot hold a second description, so the test
injects it into the block. Under each switch a prover that alone uses the second description is
rejected; in the control both sides use it, and the keys that are in no row are accepted. (from that test's source) And a padding value is a row
of the table, so that even a false number of places lets no foreign key in. That is argued in the
source, not proved: the right side of section 8's identity is then a product of factors alpha - (a
row), whatever the number of places.

## 5. The order of the rows that LKP adds to the group, and the counts

*The rows,* in order:

    tie 0 .. planes-2 | parts 0 .. k-2 | for each part: its levels 0 .. v_p-1 | table levels 0 .. d-1 | leaves | bits | bits at the point | number of keys

That is (planes - 1) + (k - 1) + sum v_p + d + 4 rows.

*Mask entries,* as many as scalars are sent (`lkp_masks`):

    7 + 6 (planes - 2) + 5 (k - 1) + sum_p sum_(l < v_p) (2 l + 5) + sum_(l < d) (2 (mu + l) + 5) + (1 + mu + 2 d) + (1 + 2 (mu + d) + 3)

Of them 2 k come from the second ledger (`lkp_key_masks`). The ledgers are asked in the order of the
run: each PSC takes its entries when it begins.

*The verifier's messages,* in order:

    [shift] | alpha | for each tie step: eps, beta, (empty) | for each parts step: the same | r | for each key level: eps, l rounds, beta, (empty), rho | for each table level: the same with mu + l rounds | leaves: eps, mu + d rounds | zeta | bits: eps, mu + d rounds, beta, (empty)

The routine that plans the same rounds is `plan_lookup`. This part has not been compared with `plan_lookup` line by line.

*Elements that must not be 0:* every PSC's beta, and nothing else.

## 6. What is checked on the spot

Nothing algebraic. All is deferred into section 5's rows, which the proximity test on the caller's
group closes, and into the key claims, which the caller closes. `lkp_verify` is false only if the
channel has rejected.

## 7. Errors

As the head comment of `lkp.h` states them (for alpha it writes 2^v - 1, for one part; `plan_lookup`
in `numbers/blocks.h` counts N - 1): (N - 1)/|E| for alpha; mu/|E| for r; (mu + d)/|E| for zeta;
(rows of the table)/|E| for the shift; PSC's and LINE's as in section 1.

## 8. Why an honest run satisfies the rows (for orientation)

The product over all keys of (alpha - key) = a * prod_t (alpha - tau(t))^(m_t), with m_t = sum_j 2^j
b[j, t]; and Z = a P_0 with P_0 = prod_j G_j^(2^j). Row 4.7 gives both sides one degree in alpha, so
that for keys outside the table the two polynomials in alpha differ.

## 8b. Choices of the programs

- The tie (4.1): nothing is sent; it is a chain of sumchecks without variable on hidden scalars.
- The row of 4.7, the number of keys: section 8 says what it is for. The conditions N < 2^planes and
  d + planes < the field's bits make it speak of integers.
- What is passed on (4.8) is not a claim under plain equality weights: the weights are 0 at every
  place of padding, and the padding's and the shift's shares are taken out of the target.
- These are the programs' own, and this page knows of no other text that describes them: padding
  (section 2; the factor a of 4.0); several functions in one table by a shift, with a verifier's
  message before alpha; keys in several parts (4.1b); fewer live planes than plane slots; a public
  order of the rounds in a part's last key level; the last level's two inner masks from a second
  oracle message.
- The routine that plans a lookup's rounds admits one border case (d = planes = 32 at p = 2^64 -
  2^32 + 1) that the tests of form of section 3 reject. No toy run is at that case. It also admits
  live = 2^mu, which section 3 rejects (from the programs' source).

## 9. Limits of this part

Among this project's malicious provers of this block, every kind of row but two is once the only false
one. The two are the row on the number of keys and the rows `LKP parts, step k`: no prover makes
either alone false, and none makes `LKP parts, step 0` false. Beyond the runs of these provers, that the formulas of this part are what the code computes rests on readings of the source, not on a run.

What only the block's test covers, and no second verifier:

- the cubes of section 2 as cubes;
- the weights of 4.8 as lists of product forms;
- the canonical bytes of section 2 (in a key file the object is derived and is in no item);
- the shift;
- key blocks of padding.

What this part does not cover, or takes on a caller's word:

- Where the multiplicities are is passed in as a routine of the caller's. The condition that it
  "must read the live planes only" is a comment in the source (before `LkpPlanesAt` in `iop/lkp.h`,
  which adds that LKP does not check this): it is not data in `LkpPublic`, and no test of form
  checks it. The two blocks that call LKP, WF (`wf_planes_at` in `iop/wf.h`) and the smallness test
  (`rng_planes_at` in `iop/rng.h`), make the routine from bound objects alone; test programs also
  pass routines of their own (from the programs' source).
- Whether the right places are declared as padding is the caller's word (section 3).
- Two duties of the caller that LKP cannot test (from the programs' source, the comment at
  `LkpPublic::read_from`): that the caller's row on the keys reads no other oracle message than
  those in read_from; and that a message whose out-of-domain sample was answered in another chain
  has its root in this chain's first query.
- That a false number of places of padding lets no foreign key in is argued, not proved (4.8).
- This page knows of no checked analysis of the tie in the form of 4.1.
- The key blocks of section 2 (several trees that share their rounds) have not been compared with
  any analysis of such a forest.
- Whether `LKP shift` and `LKP` come with an empty prover's message in a given caller depends on
  what the caller sent last. (For the key file, SEND's round of the answers is closed before: parts
  1 and 6.)
- How the channel reads an element, rejects, or makes the query of an empty round, and which bytes a
  64-bit word has, are on the format page, not in this part.

## 10. Values that the verifier notes, for checking the block by itself

The routine is `VerifierContext::note`; no decision reads these values. In the order in which they
are noted in one lookup: `LKP: alpha`; `LKP: the public factor a` (4.0);
`LKP: the table side's first claim` (4.2's value); `LKP: the leaves' factor` (A_1 of 4.5); then, for
each part in turn, `LKP: a part's key claim` (the target of 4.8, with padding and shift taken out).

---

# Part 8: BATT and the member's first part

This part gives what the verifier does in the block BATT and in the member's first part, as
mathematics. A **member** is a request of a batch, as the sources call it; the member's first part
is the first part of a request's own share of the test on the table. The routines are
`member_first_verify` and its neighbours in `member.h`; a key file runs the same routines. The part
follows the programs' source: `blocks/iop/batt.h`, and `member.h`, `batt_shown.h`, `psc.h`, `hp.h`,
`mask.h`, `tensor.h` and `lin.h` in the same directory, and `numbers/round_names.h`. No proof of soundness or of hiding is given
or claimed here.

PSC and HP belong to part 2. Part 2 writes them out for weights with an equality factor in the way
of summing, where the constant coefficient of a round is left out. BATT uses the other way of
summing, the plain sum, in which the linear coefficient is left out and the hiding polynomial has
further factors. So this part writes PSC out again for that case.

## 0. Conventions

- All scalars are elements of E. K = k_col is the number of column variables. The matrix M has
  matrix_rows rows and 2^K columns, with M[y][c] in F. k_row is the number of row variables, and
  matrix_rows <= 2^k_row.
- Variable t belongs to bit t of an index, bit 0 the least significant. eq(z, c) for z in E^a and an
  index c is the product over t < a of (z_t if bit t of c is 1, else 1 - z_t). eq(z, x) for two
  points is the product over t of (z_t x_t + (1 - z_t)(1 - x_t)).
- **Row weights** g are a vector in E^(2^k_row), of which only the entries y < matrix_rows count. In
  the programs g is a list of product forms (`Tensor`) and means their sum. A **product form** is a
  scalar s, pairs (lo_t, hi_t) for t < k_row, and possibly one written-out factor: entry y = s *
  prod_t (hi_t if bit t of y is 1, else lo_t) * (a listed value, chosen by some of y's bits, on
  which (lo, hi) = (1, 1)). `member_weights_on_rows()` makes such forms for weights that are written
  out on a stretch of rows and are 0 elsewhere. A reader who writes g out as a vector loses nothing.
  *Choice of the programs:* row weights as sums of product forms, and the written-out factor. No
  bound depends on the shape; the verifier's cost does: it is linear in a written-out stretch.
- W = g^T M in E^(2^K): W[c] = sum_(y < matrix_rows) g[y] M[y][c]. W~(x) = sum_c eq(x, c) W[c].
- A **mask entry** is named by (the number of its oracle message, the number of the entry in that
  message's mask table read as one vector). A **ledger** gives out the free entries of one oracle
  message in increasing order, each once; take(n) gives the next n.
- A **row on the matrix** (`TensorRow`) is (g, psi in E^K, a list of mask terms (entry,
  coefficient), target tau). Everything in it but the entries' values is public. It says

      sum_(y, c) g[y] eq(psi, c) M[y][c]  +  the sum over its mask terms of coefficient * entry  =  tau,      that is  W~(psi) + <masks> = tau.

## 1. What BATT is given

BATT is given J >= 1 rows on the matrix, (g_j, psi_j, masks_j, tau_j), j = 0 .. J - 1, in an order
that both sides know (for the key file: part 6), and one ledger: the one that belongs to this
chain's **index-mask message**, an oracle message of mask rows alone, sent at the chain's start,
whose out-of-domain sample is answered.

The process ends (a fault of the caller, not of a file) if there is no row, if K = 0, if the points
differ in length, if a row has no weights, or if the weights differ in their number of variables.

The caller's duty, which no line of BATT can test: g_j, psi_j, tau_j, and which entries with which
coefficients are in masks_j, are functions of the chain's first query, of what the prover has sent
on the chain, and of its sampled values, all before x is sampled.

## 2. BATT, message by message (`batt_verify`; `psc_verify`)

1 + 2 K scalars are sent, and no root. There are K + 2 verifier's messages and no empty round.

| # | the prover sends | the verifier's message | the round's name | kind |
|---|---|---|---|---|
| 0 | nothing of BATT's (whatever the block before has left open is in this query) | x, one element of E, 0 allowed | `BAT-sigma` | algebraic |
| 1 | mu' (1 scalar, sent as "the hiding sum") | eps, one element of E, 0 allowed | `PSC epsilon` | a kind of its own |
| 2 + t, t = 0 .. K - 1 | q_(t,0), q_(t,2) (2 scalars, in this order) | gamma_t, one element of E, 0 allowed | `PSC round` | algebraic |

There is no further variable Y, no end value and no round `PSC values`: nothing is shown. The round
of x carries the number of rows: the channel ends the process if the plan's entry for that round
counts another J.

### 2.1 After x

    chi_j := x^j  (chi_0 = 1).        T := sum_j chi_j tau_j.

The incoming mask terms are, in the order j = 0, 1, .., and inside a row in the row's own order:
(entry, chi_j * coefficient). Write sigma for the value of this form: sigma = sum_j chi_j <masks_j>.
The verifier never knows sigma.

### 2.2 The ledger

Right after x and before mu' is read, BATT takes the next 1 + 2 K entries of the index-mask
message's ledger: w_0, w_1, .., w_(2K), in increasing order of their numbers.

### 2.3 The rounds

    R_0 := mu' + eps * T.
    For t = 0 .. K - 1:
       q_(t,1) := R_t - 2 q_(t,0) - q_(t,2)          (the check q(0) + q(1) = R_t, solved for the linear coefficient; nothing is divided)
       R_(t+1) := q_(t,0) + q_(t,1) gamma_t + q_(t,2) gamma_t^2.

### 2.4 What it computes at the end and passes on: one row on the matrix

Write G_l := prod_(l < j < K) gamma_j (G_(K-1) = 1) and P := prod_(t < K) gamma_t, and

    a_j := eps * chi_j * eq(psi_j, gamma),   j < J.

The row (g', psi', masks', tau'):

    g'   = sum_j a_j g_j        (in the programs: the product forms of row 0, then of row 1, .., each with its scalar multiplied by a_j);
    psi' = gamma = (gamma_0, .., gamma_(K-1));
    masks', in this order:
       for l = 0 .. K - 2:   (w_(2l),  gamma_l G_l),   (w_(2l+1),  gamma_l^2 G_l);
       for the last variable l = K - 1:   (w_(2K-2), 1),   (w_(2K-1), gamma_(K-1)),   (w_(2K), gamma_(K-1)^2);
       then every incoming mask term (entry, chi_j * coefficient) as (entry, eps * P * chi_j * coefficient), in the order of 2.1;
    tau' = R_K.

In one line, with W_j = g_j^T M and with U as in 2.5: eps sum_j chi_j eq(psi_j, gamma) W_j~(gamma) +
U(gamma) + eps P sigma = R_K. (With K = 1 the first group is empty: w_0, w_1, w_2 have the
coefficients 1, gamma_0, gamma_0^2.)

*Choice of the programs:* the weights of the row that comes out carry the factor eps, because the
row is taken as the sumcheck leaves it. The row whose weights lack eps is the same row up to that
public factor, which is 0 only within the round's error.

### 2.5 What the honest prover runs it on (the verifier never sees it)

    H(X) = U(X) + eps * [ sum_j chi_j eq(psi_j, X) W_j~(X) + sigma * X_0 X_1 .. X_(K-1) ],
    U(X) = sum_(l = 0 .. K-2) ( w_(2l) X_l + w_(2l+1) X_l^2 ) * X_(l+1) .. X_(K-1)  +  w_(2K-2) + w_(2K-1) X_(K-1) + w_(2K) X_(K-1)^2.

mu' is the sum of U over {0,1}^K. In round t the polynomial q_t(X) is H(gamma_0, .., gamma_(t-1), X,
x_(t+1), .., x_(K-1)) added up over all x_(t+1), .., x_(K-1) in {0,1}; it has degree 2.

Why the factor X_0 .. X_(K-1) beside sigma: its plain sum over the cube is 1, so the sum of H is
mu' + eps (sum_j chi_j (W_j~(psi_j) + <masks_j>)) = mu' + eps T for true rows.

### 2.6 What BATT checks on the spot

Nothing algebraic: each round's check is used up to get the left-out coefficient back. BATT rejects
only if the channel rejects (an item missing or not canonical, the file at its end, a round without
its proof of work). By the comment before `batt_verify` in `batt.h`, a false row that comes in, or a
sumcheck on other vectors than the W_j, makes the row that comes out false, except within the errors
of 2.7. That row is tested much later (section 4).

### 2.7 Errors as the plan counts them

By `numbers/blocks.h`: `plan_batt` is `plan_bat_sigma`, then `plan_psc` with K variables of 3
coefficients and no Y.

    x: (J - 1) / |E|;   eps: 1 / |E|;   each gamma_t: 2 / |E|.

## 3. The member's first part (`member_first_verify`)

*In:* the public inputs of the test on the table as the chain's first query holds them (the codes;
the shape of the test on the table, `LinShape`; the item `table`: the table's root as a string of
bytes and the list of its record's stored answers); the number of the index-mask message; the
**pending rows** (rows on the matrix that blocks on the same chain have left for this part: part 6,
5.4, and 5.3 below), J >= 1, in their order.

*First of all, on both sides, four tests that end the process* (`member_rows_fit`):

- (a) the message whose number is passed in as the index-mask message's has no main rows, has mask
  rows, has s > 0 (it gets an out-of-domain sample: part 1), and its mask code is the shape's
  `own_mask_code`;
- (b) every mask entry of every pending row is in that message (a row that named another message's
  entry would bring that message into the request's last group);
- (c) there is a pending row, and K = the shape's k_col;
- (d) every product form of every pending row's weights has the shape's k_row variables.

And where a caller writes row weights out on a stretch of rows (`member_weights_on_rows` with the
shape): the stretch is inside the matrix, or the process ends. Why: a caller forms its stretches
from a layout of the matrix's rows of its own, and the test on the table silently drops the weights
of the rows from matrix_rows on (which is right for product weights, since these run over the whole
cube). A layout with more rows than the shape counts would in this way lose its claims on the
surplus rows, unseen. Here the two descriptions of the matrix's rows meet and are compared.

A known limit: `member.h` has a second form of that routine (`member_inner::stretch_in_the_cube`),
which takes the number of row variables instead of the shape and asks only that the stretch is in
the cube of 2^k_row rows. A caller that used it with a layout of more rows than the shape counts
would lose those claims as described. Only its namespace and a comment keep it from a caller.
Nothing in `blocks/` uses it so: there both forms are called by tests alone. This block's test and
the set-up that it builds call the form with the shape; one test, of the cutting of a stretch into
aligned blocks, calls the second form on purpose.

*Choice of the programs:* where one of these conditions fails, the process ends, on both sides; the
verifier does not reject. By a reading of the source, all of them depend on the public structure
alone (the configuration, the caller's code) and on no byte of a proof. A second verifier may leave
them out, or better: assert them.

*Then, in this order:*

- (i) J > 1: BATT (section 2) with the index-mask message's ledger. J = 1: no message, nothing
  sampled, no entry taken: the one row goes on as it is, and the plan has no round for it
  (`plan_member_rows`). *Choice of the programs:* the case J = 1 has no batching. J is public. At
  the full model's shapes there are always several rows; tests at toy size run J = 1.
- (ii) The row (g', psi', masks', tau') becomes the claim of the test on the table (`member_claim`):
  the same four things, with each mask term as a unit (oracle message, its mask table, the entry's
  number, coefficient), in the same order.
- (iii) The first part of the test on the table (`lin_first_verify`). It is SEND (part 1) of one
  prover's message with these oracle messages, in this order (`lin_message_w`). First W, the shape's
  `w` (main rows over E; s > 0), with its leaf class. Then, if the shape needs a split of the
  request's own rows, a message of mask rows alone with s > 0, with a leaf class of its own. Then,
  if masks go ahead (the shape's word masks_ahead is 1: part 9, section 1), the fresh mask of the
  first fold of the request's own proximity test, which goes out here, under W's leaf class, with s
  = 0. So: root or roots, `SEND seed`, the answers, the empty round `SEND answer`. What an honest W
  holds: entry c of the vector g'^T M = sum_j a_j W_j is at the index lay(c) of W's main table, and
  there are zeros where the vector ends. Here lay is the layout of a vector in a table
  (`vector_lay`; part 9, section 0, gives it under `VectorLay`). The pad of W's out-of-domain sample
  is sampled by SEND itself.
- (iv) A new group, `LIN: a request's own rows`, with W's main code and the shape's `own_mask_code`.
  Its members (here in part 1's sense: the oracle messages of a group) join in this order: those
  that the claim's mask units name, each once, by increasing number (for the key file and for a
  request: the index-mask message alone); then W; then the split's message, if any
  (`lin_first_rows`).
- (v) One row is added to that group, `LIN: the claim on W`:

      sum_c eq(psi', c) * (W's main table at lay(c))  +  the sum over the claim's units of coefficient * entry  =  tau'.

  In words: W~(psi') + <masks'> = tau'. It names W and the index-mask message; it does not name the
  table.
- Kept for the second part: W's number, the split's, the group, the fold's mask if it went ahead,
  and the claim (g' is needed there, for the row that ties W to the vector at the shared point).
- (vi) The caller then seals the chain: round `SEAL` (R6.8; part 6, section 5; part 9, section 4).
  At the seal the open prover's message is empty. After the seal the caller runs the chain's end; if
  it has not, a chain that follows the seal ends the process.

*Checked on the spot* in the first part: nothing but the channel's own tests and SEND's tests of
form.

## 4. What the decision at the end owes to this part

Nothing is decided here. BATT's row lives on only as the row of 3 (v). That row, W's and the
index-mask message's sample rows, and the row that ties W to the vector at the shared point are
batched and tested by the request's own proximity test on the second chain (part 9). So it is there,
and nowhere earlier, that a false pending row can be rejected.

These cases are in the programs' tests: a pending row with a false target; a request
that speaks of another matrix throughout; and two malicious provers. The first sends BATT's hiding sum raised by 1 (of the two rows
of the request's own group only `LIN: the claim on W` is then false). The second sends W with a
vector added whose extension vanishes at psi' (only the row that ties W to the vector at the shared
point is false). In whole key files both are rejected by the linear check at the end of the second
chain's own proximity test, with the range proof's test and both shared tests passing. With the one
false row taken out of a scratch copy of both sides, each is accepted (the programs' table of
accepting controls, which is not in this repository). Of these four, this repository builds none and
checks a stored file of one, the first malicious prover's key file
(`examples/protocol-toy-false-files/LIST.tsv`: `MEMBER_OTHER_SUM`). Akin, in whole proof files of a
toy batch: it stores `SEAM_OTHER_MATRIX`, a whole batch on another matrix's table, which is rejected
earlier, at the opened columns against the root, `SEAM_FALSE_ROW` (rejected in the first chain) and
`SEAM_V_FITTED` (only the row handed on to the second chain is false). It builds none of these, and
it stores no file of the second malicious prover.

## 5. BATT with its end value shown (`batt_shown_verify`): how claims on the matrix leave a segment of a request

The key file does not use it. A request's segments use its messages; it is given here because it is
the same block with one more variable.

A limit on trust (from the programs' source, the head of `batt_shown.h` and the comment before
`batt_shown_prove`): neither the verifier program `cp_verify` nor the prover program `cp_prove`
includes that header. `cp_verify` runs the block's messages by code of its own
(`verifier/src/vf_weights_merge.h`, `wsForkMerge`), and `cp_prove` by kernels of its own. So the
tests of this block, and its malicious provers, which are not in this repository, exercise
`batt_shown.h`, a twin of what the two programs run, and by themselves say nothing of the programs'
own code; that code is exercised by whole runs (`verifier/tests/`).

### 5.1 What it is given

J rows on the matrix whose mask entries are in the segment's mask row, and two ledgers: that of the
segment's mask row ("own") and that of the index-mask message. A row may carry any number of mask
terms. Not exercised: a row that brings along all the hiding masks of a product sumcheck that ran
before it. No block of this repository makes such a row, and the block's test runs rows with one
term each (BATT's test runs several).

### 5.2 Messages

3 + 2 K scalars are sent. There are K + 3 verifier's messages and one empty round.

| # | the prover sends | the verifier's message | the round's name |
|---|---|---|---|
| 0 | nothing of its own | x, 0 allowed | `BAT-sigma` |
| 1 | mu' | eps, 0 allowed | `PSC epsilon` |
| 2 + t, t < K | q_(t,0), q_(t,2) | gamma_t, 0 allowed | `PSC round` |
| K + 2 | p_1 (1 scalar) | beta, never 0 | `PSC Y` |
| K + 3 | v (1 scalar) | nothing | `PSC values` |

The ledgers, right after x: first the next 2 + 2 K entries of the segment's ledger, w_0 .. w_(2K+1);
then the next one entry of the index-mask message's ledger, c.

### 5.3 Computed

chi_j, T, the incoming terms, R_0 and the K rounds are as in 2.1 and 2.3. Then comes the round in Y
(summed as g -> g(0): the constant is left out):

    p_0 := R_K;   R_(K+1) := p_0 + p_1 beta.

G_l and P are as in 2.4 (Y adds no factor). a_j := chi_j eq(psi_j, gamma), with no factor eps here.

Row (U) is on the segment's mask row alone; the caller adds it to the segment's group before the
segment is closed:

    for l = 0 .. K - 1:  (w_(2l), gamma_l G_l), (w_(2l+1), gamma_l^2 G_l);   (w_(2K), 1);   (w_(2K+1), beta);   then every incoming term as (entry, eps P chi_j coefficient);     target  R_(K+1) - eps v.

Row (E) is a row on the matrix, which waits as a pending row for section 3:

    g = sum_j a_j g_j;   psi = gamma;   masks = ((c, beta));   target v.            sum_j chi_j eq(psi_j, gamma) W_j~(gamma) + beta c = v.

The honest prover:

    H(X, Y) = U(X, Y) + eps [ F(X) + c Y + sigma X_0 .. X_(K-1) ],  F(X) = sum_j chi_j eq(psi_j, X) W_j~(X),
    U(X, Y) = sum_(l < K) ( w_(2l) X_l + w_(2l+1) X_l^2 ) X_(l+1) .. X_(K-1) + w_(2K) + w_(2K+1) Y;    p_1 = w_(2K+1) + eps c;    v = F(gamma) + beta c.

(U) + eps (E) is the last check R_(K+1) = H(gamma, beta).

### 5.4 Checks on the spot, and errors

Checks on the spot: none, as in 2.6. Errors: as in 2.7, and for beta: 1 / (|E| - 1).

### 5.5 What soundness rests on

The programs' source gives this: the index-mask message (its root and the answer to its
out-of-domain sample) is in the chain before beta. When a program's ledger gives c out does not
matter: its number follows from the order of the takes. By the comment before `batt_shown_slots` in
`iop/batt_shown.h`, c is in one oracle message only: a program that keeps a segment's masks in an
array by slots keeps c's value in a cell outside that array; the segment's own entry under v's
number stays as committed, unused, and no row may name it.

### 5.6 Two conditions of the protocol that the blocks do not enforce by themselves

- (a) What may leave a segment. That every claim which speaks of the table and is masked out of a
  segment's own mask row has gone through this block is seen only at the request's end, by test (b)
  of section 3. That row (U) gets into the segment's closing test is a line of the caller's code:
  nothing in the blocks enforces it. No guard for it (one that would ask, at a group's test, that
  every entry that a ledger has given out is named by some row) is in these sources.
- (b) No second run of a segment on the old index masks. Two values shown under one c give,
  subtracted with the right factors, a linear function of the weights without any mask. Inside one
  run a ledger gives out the next entry, so the fault cannot arise. A program that began the
  index-mask message's ledger again (a retry that keeps the message) would be stopped by nothing in
  the blocks, and no verifier can see it. The blocks have no retry: where a range check fails the
  prover gives up and there is no proof. The fault is injected in a test
  (`iop/tests/batt_hiding_test.cpp`, the third of its three negative controls, which `make test`
  runs; by `iop/batt_shown.h` for the sent scalars only, at toy size, with the verifier's messages
  fixed): the verifier accepts, and two matrices are told apart.

## 6. The canonical bytes of its public inputs

BATT and the member's first part bind nothing of their own into a first query. What they read comes
from three sources.

- (a) The public inputs of the test on the table (the codes, the shape, the item `table`): part 9's
  bytes and part 6's list of items. `table` is the first item of the statement. It holds the root as
  a string of bytes and the list of the record's stored answers, and not the table's name. The name
  (root, configuration's digest, the shape's words) is in no first query; it is derived where the
  record's seeds are computed (part 9).
- (b) The plan, whose entries the first query binds: the entry of round `BAT-sigma` counts J, and K
  rounds `PSC round` follow.
- (c) The pending rows, which are the output of blocks on the same chain.

No length, count or coefficient of this part is read from the proof.

In the file this part is: 1 + 2 K elements of E (J > 1) in the order of the table in section 2, then
SEND's items for W's prover's message. An element's bytes and its canonical form are on the format
page.

## 7. Numbers to compare a second verifier with

`member_first_verify` notes these, if a list is passed in, in this order; the first five only if J >
1; no decision reads them.

- `BATT: target` = tau' = R_K;
- `BATT: gamma_0`;
- `BATT: the first weight's scalar` = a_0 * (the scalar of row 0's first product form);
- `BATT: coefficient of its first mask entry` = gamma_0 G_0 (K > 1) or 1 (K = 1);
- `BATT: coefficient of the last mask term` = eps P chi_(J-1) * (the last coefficient of the last
  row);
- `member: the claim's target` = tau' (J = 1: the one pending row's target).

## 8. Limits of this part

What no second verifier covers:

- Section 5 (the form with a shown value): no second verifier has been written from it.
  `proof_check.py` in `tests/new/second/` has the block as its routine `claims_leave`. By its own
  head that routine was written from the comments and lines of the C++ headers, not from this part,
  so it can share a misreading with them.
- `cp_verify` and `cp_prove` do not include `batt_shown.h` (section 5): what is said here of that
  header is not said of their own code.
- A row with all the hiding masks of an earlier product sumcheck is not exercised (5.1).

Hiding and soundness:

- Here hiding is tested, not proved. With the verifier's messages fixed, what sections 2 and 5 send is an
  affine map of the mask entries taken, of full rank (entries = scalars = rank), in five shapes, at
  toy size only (`iop/tests/batt_hiding_test.cpp`; by a comment in it, that the map is affine is
  tried at one random point). By a comment in the
  test, with full rank the scalars are uniform over all tuples whatever the matrix is. The test does
  not cover (from the programs' source, its head): roots, the answers to out-of-domain samples,
  openings, the oracle message W and all that follows it; the hash chain, that is, sampled values
  that depend on the messages; that a program samples its randomness uniformly; and the prover
  program `cp_prove`, which does not call `batt_shown_prove`. Three injected faults of the prover's
  randomness (one c for two segments; w_(2K+1) = 0; a segment run again under the same c) leave the
  verifier accepting, the rank short and two matrices told apart.
- Here soundness is tested by runs, not proved. The errors of 2.7 and 5.4 are the plan's formulas, by
  counting zeros.
- Two conditions of the protocol are not enforced by the blocks (5.6): that row (U) gets into the
  segment's closing test, and that no segment is run a second time on the old index masks. No
  verifier can see a breach of the second.
- BATT takes on the caller's word that its rows are fixed before x is sampled (section 1).
- A second form of the routine for written-out row weights does not compare the stretch with the
  matrix's rows (section 3).
- That the four tests of section 3 depend on no byte of a proof is read in the source, not tested.

What is in other parts or on the format page:

- lay(c), the place of W's entry c in its main table, and the pad's place are in parts 9 and 1; this
  part uses `vector_lay()` and SEND as they are.
- How x, eps, gamma_t and beta are read from a round's answer (canonical form, the cap on attempts,
  "never 0") is in the format page's rules. This part only knows which call asks for which: an
  algebraic sampling for x and gamma_t, the closing of the hiding sum's round for eps, a sampling
  that is never 0 for beta.
- That the hiding sum mu' is written and hashed like any other scalar has not been read in the
  source. It has a call of its own so that the channel can insist that eps follows it. The second
  verifier in Python reads it as a plain scalar and accepts.
- Whether an empty round is between the block before BATT and the round of x depends on that block
  (for the key file: the range proof's last message; part 6).
- Entries of g' from matrix_rows on do not count; the test on the table cuts them off when it forms
  its later row (`tensor_below`; part 9).

---

# Part 9. LIN with FOLDEQ, SPLIT, the record and the binding tree (the test on a committed table): the verifier as mathematics and bytes

This part gives the verifier's side of LIN, the test on a committed table, with the two blocks that
it calls, FOLDEQ and SPLIT, with a stored table's record, and with a batch's binding tree. It
follows the programs' source: `blocks/iop/lin.h`, `foldeq.h` and `index_cpu.h` in the same
directory, and `iop/tests/lin_world.h`. Routines are named, not line numbers. Paths that begin with
`iop/`, `chain/`, `code/`, `frame/` or `numbers/` are in `blocks/`. SEND and BAT are in part 1; FOLD
and BASE in part 3; CS, and how a proximity test (IOPP in the sources) is put together, in part 4;
the key file in part 6. Hashes, chains, first queries and trees are on the format page, `FORMAT.md`
in the directory above this one; "R4.8" means its rule of that number. The limits of this part are in section 12.

## 0. Conventions

The toy run of which this part speaks is `key-and-entry` (sections 0 to 6 of this page). The **toy
world** is the small caller of this part's routines that the block's own tests use
(`iop/tests/lin_world.h`).

Index rule, eq and tables as vectors are as in parts 1 and 4. Bit 0 is the first variable. eq(j; z_0
.. z_(a-1)) is the product over t < a of (z_t if bit t of j is 1, else 1 - z_t). A table of 2^b rows
of length L is one vector: the entry at (row r, place c) has the **table index** r + 2^b c.

[j < N] is 1 if j < N, else 0. **Cut at N** means: multiplied by [j < N]. Every form below is cut
where its data ends; the code does that by the routine `tensor_below`, which changes no value of the
formulas here.

E = F[X]/(X^3 - c); theta_t = X^t for t = 0, 1, 2 (`E::theta`). D = 3 (`E_DEGREE`).

A code is C = (n = 2^log_n, l, r). In every code of the two lists of `Codes` (section 1), which are
the codes of oracle messages, l is a power of two; otherwise the process ends (`iop/relation.h`;
from the programs' source). The table's own code C_T is written in `IndexShape` itself, not in those
lists, and its l need not be a power of two (part 10, 9). Gm_C(x)[j] (j < l) and Gt_C(x)[j] (j < r)
are the generator row at position x, as in part 4, section 0. (In field 2, Goldilocks, p = 2^64 -
2^32 + 1, they are z^(r+j) and z^j, with z the point of x.) lam(C) = ceil log2 l (`log_ell`), which
is log2 l where l is a power of two; rho(C) is the least a with 2^a >= r (`log_r`).

A **row** (of the statement, as in part 1; not to be confused with a table's row) is a linear
equation with public coefficients and a public target on the underlying messages of oracle messages.
Its kinds are plain, sample and generator. Rows are put into **groups**; a proximity test takes a
group's rows in the order of section 9.

How a vector is placed in an oracle message (`VectorLay`, `foldeq.h`). An oracle message O with R_O
real main rows over E, under a code with message length L = 2^p, has b = ceil log2 R_O row bits. A
vector a[0], a[1], .. is placed in its main table in one of two ways, by a public flag for each
oracle message:

    rows_high = false:  entry j is at table index j                             (row j mod 2^b, place j div 2^b); allowed only if R_O = 2^b;
    rows_high = true:   entry j is at table index (j div L) + 2^b (j mod L)     (row j div L, place j mod L: a row holds L consecutive entries).

`O<j>` stands for "the entry of O's main table at which entry j of the vector is". All rows of the
statement below are stated on `O<j>`. A table entry that is not an `O<j>` of the vector, is not a
pad, and is in a real row is 0 for an honest prover, and has the coefficient 0 in every row but O's
own sample row.

The pad of O's out-of-domain sample (part 1), at s = 1 answer, is one entry. Where O has mask rows,
it is the last entry of the last real one of them. An O of main rows alone has it at the same spot
among those: row R_O - 1, place L - 1, so table index (R_O - 1) + 2^b (L - 1). A layout must end
before the pad (`vector_room`, `foldeq.h`; `lin_fault` rejects the shape otherwise).

Three objects of the programs for a stored table (`foldeq.h`). A **holder** is the committer's
object that has the table (`IndexHolder`). A **reader** is the verifier's object for it
(`IndexReader`): it gives the table's shape, commitment and record, and reads opened columns against
the commitment. A table's **key** is the reader of a table whose commitment is a root (`IndexKey`;
part 10, 2 (a)); it is not the key file, which is always written in full.

## 1. What is public, and its canonical bytes

`public_bytes` (`chain/public.h`) encodes every integer, flag and enum as 8 bytes, little-endian; an
element of F or E and a digest as the format page encodes them; a list as its length (8 bytes) and
then its items. Below, "|" separates fields, in their order.

- `Code` = log_n | l | r. `Codes` = the list of base codes | the list of mask codes. A `CodeId` is a
  place in one of the two lists.
- `OracleShape` = code | mask_code | main_rows | main_deg | mask_rows | answers (the shape's name is
  for printed lines only). One shape has one encoding (R12.3): without main rows the word code is
  2^32 - 1 ("no code") and the word main_deg is 1; without mask rows the word mask_code is 2^32 - 1,
  whatever the field in memory has. In `LinShape` that applies to Omega, which has no mask rows, and
  to a split's message, which has no main rows. `OracleNotice` = shape | leaf_class (the leaf class:
  R7.12).
- `IndexShape` = `Code` | v | answers | salt bytes: six words in all. The table has 2^v rows over F
  under the code C_T = (n_T, l_T, r_T), and a = answers >= 1 stored answers to an out-of-domain
  sample for each row. The last D a entries of each row's message are **pads**: pad (i,
  t) is at the place P(i, t) = l_T - D a + D i + t (i < a, t < D). plain = l_T - D a.
- `FoldeqShape` = `IndexShape` | the `OracleShape` of G_u | rows_high | leaf_class | columns
  (`foldeq.h`). columns = nu: how many positions of the table a proof samples.
- `LinShape` (`public_bytes`, `lin.h`) = `FoldeqShape` | k_lo | matrix_rows | the shape of Omega |
  the shape of W | omega_rows_high | w_rows_high | the shape of the shared split's mask message |
  the shape of a request's split's mask message | four leaf classes (Omega, shared split, W, own
  split) | shared_mask_code | own_mask_code | masks_ahead | the four tests `test_u`, `test_omega`,
  `test_own`, `test_own_omega`, each a list of stages (the number of stages, then for each stage
  four words: code | mask_code | k | h) | the choice: one word, masks_ride. (Stages and choice are
  both encoded by `iopp.h`'s own `public_bytes`. The choice holds no number of answers for the
  oracle messages that a test's switches send: that number is the setting's s, which is in the
  plan's words. A shape's own word for its number of answers is 0 or that s: part 1, 2.1.)
- `IndexRecord` = the list y of its answers, and nothing else (no seeds). `IndexName`, the item
  `table`, = the commitment as a string of bytes | `IndexRecord`. For a tree of the library's kind
  the commitment is the root: the word 32, the 32 bytes, then the list of answers. A string of bytes
  is encoded as its length as a word, then the bytes as they are (R1.6).
- `LinBindingShape` = log_leaves | salt_bytes: two words. (The value from which the tree's hashes
  start is the chain's item `configuration`.)

Derived numbers: m = matrix_rows; k_col = k_lo + v; k_row = ceil log2 m. Two codes or one: the
**shared split** is on if and only if G_u's and Omega's base codes differ; the **own split** if and
only if W's and Omega's differ (`LinShape::shared_split_needed`, `own_split_needed`).

Tests of form on these numbers (`public_fault(LinShape)` for what the shape shows by itself,
`lin_fault` with the codes; `foldeq_fault`, `index_fault`) are not part of a proof: a verifier that
reads the numbers from a file asks these routines first, and the entry points end the process on a
fault. Among the tests: m 2^k_lo <= plain; G_u, Omega and W have main rows over E and >= 1 answer;
Omega has no mask rows; each layout ends before its pad. Three tests are made only where the split
concerned is on: Omega and the shared split's message have different leaf classes; W and the own
split's message have different leaf classes; the split's message has mask rows only, gets an
out-of-domain sample, and has its groups' mask code.

## 2. The matrix, the table, its name, its record

The matrix M has m rows and 2^k_col columns over F. Column c = j + 2^k_lo r (j < 2^k_lo, r < 2^v).
The entry M[y][c] is in row r of the table, at the message place j + 2^k_lo y (the comment before
`LinShape` in `lin.h`). T_r[p] is place p of row r's message (p < l_T), and t_r[q] is entry q of its
random tail (q < r_T). The places m 2^k_lo <= p < plain mean nothing: they are **spare**. (Fold),
the one row through which LIN reads the matrix, has no weight there; (Rec), (Col) and G_u's sample
row read them like every other place of a row's message.

A choice of the programs: (Fold) is cut at the matrix's last row. So the spare places are bound (one
definite table is committed, and a link compares them), but nothing proves them 0. No claim on an
entry of the matrix reaches them; an honest committer places zeros there.

The tree is a perfect binary tree with n_T leaves. Leaf x holds the 2^v symbols of the rows at
position x, row 0 first, and then one salt of `IndexShape`'s number of salt bytes. It is a stored
tree: its leaves are hashed under the number 174 with the tree number 0 (a constant of the format
for every stored table's tree: the anchor and its leaf tables are told apart by their roots), its
vertices under 166, and both start from the configuration's digest (R7.4). The tree's shape is a
function of `IndexShape` and the configuration's digest (`index_tree`).

Commitment and name are two things. The **commitment** is the root: what a key file gives, and what
the item `table` of a first query holds. The **name** (`index_name`; R4.10) has 112 bytes: the root
(32) | the configuration's digest (32) | six words of 8 bytes: log_n_T, l_T, r_T, v, a, salt bytes.
Each side computes it from the root, its own `IndexShape` and its own configuration's digest. It is
in no first query and is read in two places: the record's seeds are a function of it (R4.8), and the
commitment ID hashes the anchor's name beside the configuration's digest and the record's seeds and
answers (R4.11; `frame/heads.h`, which does not belong to this part).

A choice of the programs: the record's seeds are sampled on a hash chain of their own, whose
statement is the name. A stored tree's hashes start from the digest of all public files, so the root
itself depends on that digest: the same rows under two sets of public files have two roots, two
names and two seeds. What hiding needs is therefore a duty of the committer, which no verifier sees:
the same pads, random entries and leaves' salts are never under two roots (R4.12). An anchor and
each of its leaf tables have roots of their own, so they share none of these either.

The record. Its seeds are (z^(i), beta^(i)), i < a, with z^(i) in E^lam(C_T) and beta^(i) in
E \ {0}. They are `record_seeds`(name, lam(C_T), a): a function of the name's bytes alone, outside
every other chain (`chain/rounds.h`; R4.8). Its answers are y[r a + i] in E, by row, then by answer.
An honest committer has (`record_value`, `foldeq.h`)

    y[r a + i] = sum over p < plain of eq(p; z^(i)) T_r[p]  +  beta^(i) * sum over t < D of theta_t T_r[P(i, t)].

(No pad place has a weight under z; beta is on the pad of answer i only.)

The verifier, before any chain (`record_fault`; `IndexReader::fault`; `foldeq_require`): the record
has a 2^v answers. (Another number in a file is a rejection; inside the blocks it ends the process.)
The verifier computes the seeds from the name whenever it needs them; no file and no item holds a
seed. Why: a public fact that is written in two places invites a caller that reads the wrong one, so
the seeds are in one place, the routine that derives them. (One hash's input holds the anchor's
seeds all the same, beside the name from which they follow: the commitment ID's; R4.11 says why.)
Nothing else about y can be checked here: y is what defines "the table" among near ones
(section 14, `INDEX_TWO_TABLES`).

## 3. Three chains for each request's view, and their first queries

A **view** of request c is (the first part of c, the path, the shared part, the second part of c).
The verifier of a view is given nothing else of the other requests. A view does hold two things that
depend on them: the path's d sibling digests (section 5), each a salted hash over a neighbour's
seal, a uniform filler or a hash of such; and the shared part, which is byte for byte the same in
all proofs of a batch. That these tell nothing of the neighbours rests on the hash and the salts: it
is the reason that the programs' source gives (`iop/lin.h`), not a result. Anyone who holds two
proofs of one batch can see that they shared it
([what-is-not-hidden.md](../../what-is-not-hidden.md)).

A choice of the programs: each request's first part runs on a hash chain of its own, which is
sealed. The seals are salted leaves of a small binary tree (section 5). That tree's root is an item
of the statement of the batch's shared chain, and also that chain's random bytes. Each request's
last part runs on a further chain that holds both seals.

The items of a first query are, in order (`Binder`; R4.1 to R4.6): version, plan | `configuration` |
further parameters | statement.

- The first chain of request c (its label is the caller's; in the toy world: 1). Parameters: `codes`
  = `Codes`, `the test on the table` = `LinShape` (`lin_bind_parameters`), and the caller's own (in
  the toy world, `lin_bind_first` in `iop/tests/lin_world.h`: `a request's mask message` =
  `OracleNotice`, `the binding tree` = `LinBindingShape`). Statement: first `table` = `IndexName`
  (`lin_bind_table`), then the caller's items (in the toy world: `the claims' seed`, a word of 8
  bytes, and `request` = c). Random bytes: the chain's own, at the start of the proof.
- The shared chain, label 0x4c494e01 (in a proof file: 0x46524d03, R14.9). Parameters: `codes` and `the test on the table` = `LinShape`,
  as in the first chain. Statement: `table`, then `the binding tree's root` = a digest
  (`lin_bind_shared`). Its 32 random bytes are that root; none are written into the proof.
- The second chain of request c, label 0x4c494e02 (in a proof file: 0x46524d04, R14.10). It has no parameters of LIN's. Statement:
  `the seal of an earlier chain` twice: the seal of the request's own first chain, then the shared
  chain's (`lin_bind_second`). Random bytes: the chain's own, at the start of that chain's bytes in
  the proof (R4.5). The statement, the list of trees and all rows of the first chain's context go on
  into it (`lin_go_on`).

Anyone who makes the `Binder` names the configuration first; `lin.h` does not know it.

Checks of the program, not of the proof (made by the entry points on `LinPublic`). The table's key
that is passed in (section 0) has the commitment and the record that the first query holds, has no
fault of its own, and its tree starts from the chain's configuration. The second chain's first query
holds the seal of the very shared chain whose context is passed in. And the items `codes`,
`the test on the table` and `table` of the request's first chain and of the shared chain have the
same bytes. (They are written in both chains, because each must be bound to them by itself; the
second chain follows both seals and reads both.)

Why the last comparison: a caller could bind two different shapes in the two chains, injected on both
sides, the prover's and the verifier's.
`lin_second_prove` and `lin_second_verify`
end the process on a difference.

## 4. First part, on the request's own chain

The claim of a request (`LinClaim`, `lin.h`) consists of public row weights g in E^(2^k_row) (a sum
of structured vectors, `Tensor` in `iop/tensor.h`: v[j] = a scalar times the product over t of one
of two factors, by bit t of j; entries y >= m do not count), a point psi in E^k_col, a list of mask
terms (oracle message o_q, mask table index i_q, coefficient a_q), and a target tau. It says

    sum over y < m, c < 2^k_col of g[y] eq(c; psi) M[y][c]  +  sum over q of a_q * (mask entry i_q of o_q)  =  tau.

The caller's duty: g, psi, the mask terms and tau are functions of what this chain has absorbed
before its seal. Nothing in LIN enforces it. (The toy world sends one mask-only message,
`a request's masks`, by SEND, with its out-of-domain sample. The prover then sends tau as one
scalar, which the verifier compares with the tau that it is given. g and psi come from the two
statement items. There is one mask term: the first free entry of that message's ledger, with the
coefficient 1.)

(1) Prover, one prover's message (`lin_message_w`), in this order: W (main rows over E; leaf class
`w_leaf_class`); if own split, the own split's mask message (mask rows only; a leaf class of its
own, so a short tree of its own); if masks_ahead, the fresh mask of the first fold of the request's
own test (it gets no out-of-domain sample; W's leaf class: its symbols are copied into W's leaves).
An honest prover has

    W<c> = sum over y < m of g[y] M[y][c],   c < 2^k_col.

(2) Verifier, round `SEND seed` (kind 3, seeds): one seed for the whole prover's message: z in E^A,
and then beta != 0. The cube of an oracle message is the number of coordinates that its
out-of-domain sample needs (a_o in part 1, 2.2); A is the largest cube among the oracle messages
with s > 0 of this prover's message. Each of them uses the first coordinates of z, as many as its
own cube has.

(3) Prover: the answers, one for each oracle message with s > 0, in the order of sending (W; the
split's message). Round `SEND answer`: the verifier says nothing.

(4) The verifier keeps (`lin_first_rows`): a new group OWN (base code: W's; mask code:
own_mask_code) with these members: the oracle messages that the claim's mask terms name, W, and the
own split's message if there is one; the sample rows of (3); and the plain row

    (Eval)   sum over c < 2^k_col of eq(c; psi) W<c>  +  sum over q of a_q (mask entry i_q of o_q)  =  tau.

(5) The caller can add rows of its own to OWN. Then comes the seal: round `SEAL`; at that moment the
prover's message is empty. The chain ends (bytes left over: rejected).

On the spot nothing is checked but SEND's tests of form.

## 5. The binding tree and the path

The binding tree is a perfect binary tree with 2^d leaves, d = log_leaves. (In `lin.h`:
`LinBindingShape`, `lin_binding_leaf`, `lin_binding_vertex`, `LinBinding`, `lin_binding_root`.) A
request's **slot** is the place of its leaf among the 2^d leaves (R7.18). The prover chooses it. The
library leaves that to its caller: by `iop/lin.h`, to hide the order of arrival the caller must
assign places at random. By a comment in `prover/prover/batch_opening.h` the prover program takes a
random arrangement, so that a place says nothing of the order of arrival. So the slot is not the
request's number in the batch's list.

    leaf(slot, seal, salt) = H(173, tree 0, W = 4, s8 = salt_bytes / 8, index 2^d + slot, start = configuration's digest;  seal (32 bytes) | salt (salt_bytes))     (R7.18; the first word of the hash's label holds W and s8, R2.1);
    vertex(l, r) = H(178, tree 0, index 0, start = configuration's digest;  l | r).

The configuration's digest is the channel's (`first_ch.configuration()`). At a slot that no request
takes, the leaf's digest itself is 32 uniform bytes of the prover's (no hash call; no step of a
verifier reads this).

The path in a view is: the slot (8 bytes, little-endian) | the salt | d digests, the siblings from
the leaf upward. The verifier (`lin_binding_root`) rejects a slot >= 2^d, or missing bytes. It sets
h := leaf(slot, its own first chain's seal, salt); then for e = 0 .. d - 1, with s the next digest,
h := vertex(s, h) if bit e of the slot is 1, else vertex(h, s). The result is
`the binding tree's root` of section 3. Bytes left over in the path are rejected (by the caller).
Nothing compares the root with anything: it only starts the shared chain, so a wrong path gives
another shared chain, under which the shared part's bytes fail.

## 6. Shared part, once for each batch. No claim enters it.

(1) Verifier, round `LIN point` (kind 1, elements of E; "algebraic" in the sources; no prover's
message comes before it): psi* in E^k_col, any values. It is cut in two:

    psi_lo = (psi*_0 .. psi*_(k_lo-1)),    gamma = (psi*_k_lo .. psi*_(k_col-1)) in E^v.

Groups: U (base code: G_u's; mask code: shared_mask_code); OM is a second group (Omega's code, the
same mask code) if shared split, else OM = U.

Steps (2) to (8) are FOLDEQ at gamma (`foldeq_verify`). Let e_r := eq(r; gamma), r < 2^v. The
**fold** is, over E,

    u[p] = sum_r e_r T_r[p] (p < l_T),    ut[q] = sum_r e_r t_r[q] (q < r_T).

(2) Prover, one prover's message (`lin_message_u`): G_u; if masks_ahead, the fresh mask of the first
fold of the test on U (it gets no out-of-domain sample; G_u's leaf class: it is copied into G_u's
leaves). An honest G_u holds one vector (`foldeq_table`):

    G_u<p> = u[p] for p < l_T (pads' folds among them);   G_u<B + q> = ut[q] for q < r_T, with B = tail_first = l_T rounded up to a multiple of 2^rho(C_T);   0 between and after;   its own out-of-domain sample's pad where section 0 says.

(3) Verifier, round `SEND seed`. (4) Prover: one answer (G_u's); round `SEND answer`, in which the
verifier says nothing.

(5) The verifier keeps: G_u joins U; G_u's sample row; and for each i < a the plain row
(`foldeq_record_row`), with (z^(i), beta^(i)) = `record_seeds`(`IndexShape`, the table's name)[i]
(the form of `foldeq.h`, which calls that of section 2), computed here:

    (Rec_i)  sum over p < plain of eq(p; z^(i)) G_u<p>  +  beta^(i) sum over t < D of theta_t G_u<P(i, t)>  =  sum over r of e_r y[r a + i].

(The coefficient is 0 on the pads of the other answers. The row reads spare places too.)

(6) Verifier, round `LOC` (kind 4, positions): nu positions x_1 .. x_nu, each uniform in [n_T];
repeats are possible and are kept.

(7) Prover, in the file only: the opening of the table's tree at the sorted set of distinct
positions (R7.8, R8.7).

(8) The verifier checks the opening against the root in the table's key, and rejects if it does not
lead there (`the opened columns of the table do not lead to its root`). col_x[r] in F is row r's
symbol at x. For each draw i = 1 .. nu, in the order sampled, repeats counted, it makes a generator
row in U (`foldeq_column_row`), with x = x_i:

    (Col_i)  sum over p < l_T of Gm_T(x)[p] G_u<p>  +  sum over q < r_T of Gt_T(x)[q] G_u<B + q>  =  sum over r of e_r col_x[r].

Here FOLDEQ ends.

(9) Prover, one prover's message (`lin_message_omega`): Omega (a leaf class of its own: it is alone
in its tree); if shared split, the shared split's mask message (mask rows only, another leaf class:
a short tree of its own). An honest prover has

    Omega<y> = sum over j < 2^k_lo of eq(j; psi_lo) u[j + 2^k_lo y] = sum over c of eq(c; psi*) M[y][c],   y < m.

(10) Verifier, round `SEND seed` (one seed for both). (11) Prover: the answers, Omega's and then the
split's message's; round `SEND answer`, in which the verifier says nothing. Both sides end the
process if anything shares Omega's tree (`lin_shared_join`): Omega's leaves are opened at 1 + (the
number of requests) lists (the reason: section 14, `LIN_MASK_IN_OMEGAS_LEAVES`). Omega joins OM; the
split's message joins OM and U.

(12) Verifier, round `LIN zeta` (kind 1; "algebraic"): zeta in E^k_row.

(13) The plain row (`lin_fold_row`)

    (Fold)   sum over y < m of eq(y; zeta) Omega<y>  -  sum over y < m, j < 2^k_lo of eq(j; psi_lo) eq(y; zeta) G_u<j + 2^k_lo y>  =  0.

With one code it goes into U. With two codes: SPLIT (section 8) with g1 = OM, g2 = U, and the shared
split's message as the mask message.

(14) Proximity tests (`lin_shared_tests_verify`): the proximity test on U with the stages `test_u`;
its first fold's mask is the one of (2) if masks_ahead, else the fold sends it. Then, if shared
split, the proximity test on OM with `test_omega`, whose first fold always sends its fresh mask
itself, alone, in a short tree of its own (never ahead, never in Omega's leaves).

(15) Round `SEAL`; the chain ends.

## 7. Second part, on the request's second chain

(1) No message. Omega enters the request's statement (`lin_adopt`) as a new oracle message with
Omega's shape: the second chain adopts Omega's tree (R7.13). Its tree is the shared chain's: the
same root, the same shape, with the tree's number and the run's value under which its leaves were
hashed there. Its sample row is the one made in 6 (10) and (11), coefficient for coefficient, with
the same target; its out-of-domain sample counts as answered. Its number in the request's statement
is the next free one. The process ends unless Omega is alone in its tree. Group AR: if own split, a
new group (Omega's code, own_mask_code) with the members Omega and the own split's message; else AR
= OWN, which Omega joins.

(2) The plain row (`lin_dot_row`)

    (Dot)    sum over y < m of g[y] Omega<y>  -  sum over c < 2^k_col of eq(c; psi*) W<c>  =  0.

With one code it goes into OWN. With two codes: SPLIT with g1 = AR, g2 = OWN, and the own split's
message as the mask message.

(3) Proximity tests (`lin_second_tests_verify`): the proximity test on OWN with `test_own` (its
first fold's mask is the one of 4 (1) if masks_ahead); then, if own split, the proximity test on AR
with `test_own_omega` (its fold sends its mask itself).

(4) The chain ends with the last test's positions; it has no seal. Bytes left over are rejected. The
view is accepted if and only if all three chains and the path are.

How the rows fit together, in one line (an argument, not a proof): (Eval) says W(psi) + masks = tau;
(Dot), at a psi* that is sampled after W, says W = g^T M', where Omega = M' eq(psi*); (Fold) at zeta
ties Omega to the folded row; (Col) and (Rec) tie the folded row to the committed table.

## 8. SPLIT, as LIN uses it (`iop/split.h`)

Given are a plain row R: sum of terms = t, each of whose oracle messages belongs to exactly one of
two groups g1, g2, and a mask-only message K that is a member of both and whose out-of-domain sample
is answered.

(1) kappa := the next free entry of K's ledger (both sides take it before the share is read).

(2) The prover sends one scalar, the share sigma in E. Round `SPLIT`: the verifier says nothing.

(3) Two plain rows:

    into g1:  (R's terms on members of g1) + kappa = sigma;   into g2:  (R's terms on members of g2) - kappa = t - sigma.

An honest sigma is the left side of the first. For (Fold): g1 = OM has Omega's terms, g2 = U has
G_u's. For (Dot): g1 = AR has Omega's, g2 = OWN has W's.

## 9. What each proximity test is given: members, layout, order of rows

The members of a group are kept by increasing number (that is, in the order of sending in that
statement). Layout (`iop/relation.h`): among the members with main rows, blocks of 2^(row bits)
rows, larger blocks first, equal ones in the members' order, from row 0; likewise for mask rows; h =
ceil log2(mask rows used), k = max(ceil log2(main rows used), h, 1). These must equal the first
stage's (k, h) in `LinShape`, and the group's two codes must equal the stage's (the process ends
otherwise).

The order of rows for BAT: plain rows as created | sample rows, by member in order | generator rows
as created.

- U, one code: (Rec_0) .. (Rec_(a-1)), (Fold) | the sample row of G_u, the sample row of Omega |
  (Col_1) .. (Col_nu). These are a + 1 + 2 + nu rows.
- U, two codes: (Rec_i), the second share of (Fold) | the sample row of G_u, the sample row of the
  split's message | (Col_i). These are a + 1 + 2 + nu rows.
- OM, two codes: the first share of (Fold) | the sample row of Omega, the sample row of the split's
  message. These are 3 rows.
- OWN, one code: (Eval), [the caller's rows], (Dot) | the sample rows of: the claim's mask messages,
  W, Omega (numbers rising: Omega is the newest).
- OWN, two codes: (Eval), [the caller's rows], the second share of (Dot) | the sample rows of: the
  mask messages, W, the own split's message.
- AR, two codes: the first share of (Dot) | the sample row of the own split's message, the sample
  row of Omega (the split's message was sent earlier, so it has the lower number).

With s > 1 answers, each oracle message that gets an out-of-domain sample has s sample rows, one
after the other.

## 10. Mask entries: which ledger, in which order

LIN itself takes exactly: one entry of the shared split's message (its first free one: index 0 of
its mask table), at 6 (13), if shared split; and one entry of the own split's message (its first
free one), at 7 (2), if own split. It takes nothing else: LIN shows no scalar but answers to
out-of-domain samples (each under its message's pad) and the two shares. The claim's mask terms are
the caller's, taken from the caller's messages before the first part is called. A ledger's entries
are taken in the order 0, 1, 2, .. (part 1); it closes when a test on a group of its message begins.
The fresh masks of folds and the masking words of last steps are whole oracle messages, not ledger
entries (part 3).

## 11. What is checked on the spot; what ends the process

One session. The verifier of a view makes one session object (`chain/session.h`) before its first
chain and passes it to all of them. For every root's 32 bytes that are read in the view, whichever
chain sent or reads them, the session keeps one record of the values seen at the vertices under that
root. Only what an accepted opening shows is entered. A second value at a vertex under one root
rejects the view (R7.14). The stored table's opening of 6 (8) goes into that record under the
table's root like every other. The verdict is the session's: no chain has rejected. Where none has
rejected, a chain that was begun and not brought to its end is not a rejection: it ends the process
(`chain/session.h`; from the programs' source).

Rejections (a proof's fault): any missing or non-canonical item (the channel); the table's opening
against the root, 6 (8); the path's form, section 5; tau against the given one (the caller's check);
bytes left over at a chain's end; whatever SEND, SPLIT (only: the share is missing), BAT, FOLD, CS
and BASE reject. Beta, like every sampled value that must not be 0, is the verifier's own and is
never read. All equations (Eval), (Rec), (Col), (Fold), (Dot) are deferred into the tests.

Ends of the process (a program's fault, never reached by a proof's bytes):

- a shape with a fault;
- a table's key (`IndexKey`) that is not the named table's, or whose tree starts from another
  configuration;
- two chains of a view that hold different codes, shapes or tables;
- a second chain that does not hold the seal of the shared chain that is passed in;
- a number of sampled columns other than nu in the plan;
- a plan that was counted for another table, or for another of the table's tests (the one-time test,
  a later test, a link), than the run says, or that names none: asked where the point is sampled and
  again where the columns are sampled (`require_table` in `chain/rounds.h`; section 14; from the
  programs' source);
- a plan that counts the error of `LIN point` for another number of requests N (section 12) than the
  binding tree has slots, 2^d, or that names none (`require_sharing` in `chain/rounds.h`); that
  number of the plan is in no first query (`numbers/plan.h`; from the programs' source);
- LIN called as a link's test; a caller that says that a one-time test reads a tested table, or a
  later test a table under test (`iop/lin.h`, `iop/foldeq.h`; from the programs' source);
- something in Omega's tree;
- a group that is never tested, or an oracle message with an answered out-of-domain sample that is
  in no group (`require_all_tested`);
- a chain that follows a chain of another session;
- a table's reader that names other distances than the run (section 12; from the programs' source);
- in a later test, a reader that has passed no one-time test, or a table that was tested under
  another setting (section 12; from the programs' source).

On the prover's side also: a batch that was not counted against the stored table; main rows opened
at more positions than their random tail hides.

## 12. Limits of this part

What nothing has compared with the programs:

- The two cases of `lin_test.cpp` with tight random tails.
- Everything at more than one answer to an out-of-domain sample.

What this part does not decide, what these sources do not have, and what no test covers:

- More than one answer (s > 1). The order of two seeds' coordinates, and SEND's number `users`, are
  matters of part 1 and the format page. LIN takes s from the shapes, but every test of this block runs it with s = 1, and the parser of the parameter file refuses any `answers` but 1.
- The circle code (field 1, p = 2^61 - 1). The programs of this repository can be built for field 2
  alone and have no routine that encodes with this code; only their planning part has sizes for it
  (`blocks/base.h`, `blocks/code/`, `blocks/numbers/codes.h`; from the programs' source).
- The GPU programs for a real model's weight table are not described here. Their table's tree is
  hashed by these rules and can be opened in this format; this page reports no run of that at full
  size.
- These sources have no form with four verifier's rounds, in which Omega is sent together with G_u.
  This page reports runs at toy sizes only, and none under the parameter files of real size
  (`blocks/frame/examples/record.f2.txt` and `prover/examples/target_full_f2/policy.txt`; from the
  programs' source). No extractor and no simulator exists as code.
- The reader's own shape is the bound one. The verifier's entry into FOLDEQ (`foldeq_require`)
  compares the table reader's own shape with the shape that the first query binds, in all four
  parts: the code, v, the number of answers, the number of bytes of a leaf's salt. A difference is a
  caller's mistake and ends the process. (The hash's label of a leaf holds the salt's length, so a
  root opens under one length only.) The test is in
  `iop/tests/lin_test.cpp`: a key with another length of the salt ends the verifier's process. (from
  the programs' source, `iop/foldeq.h` and that test)
- The table's distances are not bound (from the programs' source, `iop/foldeq.h`). `IndexShape` has
  a further member, `distances`: the distances at which the tests on the table are sized.
  `public_bytes()` and `index_name()` leave it out, so it is in no first query and not in the
  table's name. By the source, the errors that a plan counts for a later test and for a link are
  true only of a table whose one-time test ran at the same distances, in another run. `check_key`
  and `check_entry` return their own run's shape inside a `TestedTable`, which ordinary code cannot
  alter, and a later run ends the process if its reader names other distances than its own
  (`foldeq_columns_verify`). A
  reader that is made by hand, or that a holder returns, has passed no test: it counts as untested,
  and a later test or the anchor's side of a link that is given one ends the process. This is
  enforced inside one process only (the next item).
- A later test relies on an earlier one, and the library enforces that inside one process only (from
  the programs' source: the comment before `run_verifier` in `chain/closed.h`; `iop/foldeq.h`;
  `blocks/README.md`).
  - By the source, the errors that a plan counts for a later test and for a link are true only of a
    table whose key file was tested at the same distances, target and proof of work. A one-time test
    that was sized under a setting with a lower target passes on fewer columns, and a batch at a
    higher target then relies on it: a chain of tests is worth its weakest target.
  - Enforced inside one process: that a one-time test at these distances and under this setting (target, proof-of-work bits) has passed on the table. A later
    test and the anchor's side of a link read a `TestedTable` (`iop/foldeq.h`). Only
    `check_key_under` and `check_entry_under` return one (`frame/check_key.h`,
    `frame/check_entry.h`), as their last step, after the whole file has been accepted, and it has
    no public data member and no routine that changes it. It holds the table's reader, the setting
    under which the test passed, and the public values under which the key file was checked (the
    ranges of the matrix's entries, its rows: `key_public_bytes` in `frame/key_fixed.h`); a leaf
    table's holds those of the key file to which it is linked. Where the columns are read,
    `foldeq_columns_verify` ends the process on a reader that is made by hand, on a reader that a
    holder returns, on a table that was tested at other distances, and on a table that was tested
    under another target or other proof-of-work bits. `check_entry_under` and `check_proof` end the
    process if the key file's public values are not those of their own configuration. Why the public
    values: some of them follow from the public files by a routine of the calling program
    (`ShapesOf` in `frame/configure.h`), and the configuration's digest covers the files alone, so
    two such routines give two configurations with one digest, one setting and one commitment ID.
  - Without the types that enforce this, test programs in ordinary code that call the library directly got false claims accepted
    in six ways, all at toy size (by `blocks/README.md`). Four ways were run at targets of 8 to 32
    bits, with claims on another matrix than the committed one. One public member of a checked
    reader was assigned. A reader was made by hand. A second holder, built on the later run's shape
    from the same tables and randomness, had the same root, and its reader named the later run's
    distances though no test was run at them: two false claims were accepted, on synthetic data, at
    a target of 24 bits, on the CPU (the two claims and the target are in a comment in
    `iop/foldeq.h`, not in `blocks/README.md`). A key file was tested at a lower target than the batch that relied on it: in a
    test harness the error counted for the batch's test was untrue by some 4.6 bits. Two ways had a
    matrix out of its ranges. A key file was checked under laxer ranges than the proof's check
    stated, by two configurations made from the same public files (a target of 128 bits with proof
    of work counted). The same two configurations were swapped by the caller's own routine for the
    segments while `check_proof` ran. By the source all six are closed inside one process, by the type and the stops of the item before
    this one; a reading holds that some are closed only in part (the format page's 17.5, item 11). The tests are the sources on a tested table that
    `blocks/tests/must_not_build/README.md` lists (by that page no target of the Makefile builds
    them, and no row of `SUITE.md` runs the stage that compiles them), the cases in
    `iop/tests/lin_test.cpp` and `iop/tests/link_test.cpp` that must end the process, and
    `frame/tests/tested_table_test.cpp` (a weight of 27 where the ranges allow -16 to 15), with
    eight lines of the programs' table of accepting controls (section 14; not in this repository),
    from `a-later-test-wants-a-tested-table` to `a-proof-behind-the-callers-routine`.
  - Not enforced: anything across processes or files. A program that checks key file, leaf file and
    proof in separate processes, as `cp_verify` does, reads the tables from the first bytes of the
    key file and of the leaf file, without the proofs that follow them (`UnprovedTable`). Its checks
    return a result of the type `IfTested`, which has no plain acceptance: it is a rejection, or an
    acceptance that counts only if the table is tested. That all of the program's processes passed,
    and on the same bytes, is that program's own logic: no type and no stop of the library enforces
    it. An `UnprovedTable` holds the setting and the key file's public values of the caller's own
    configuration, so in such a process the stops of the item "Enforced inside one process" compare
    a configuration with itself. Of `cp_verify` the source says this: the stage key checks the
    anchor's proofs, the stage leaf ties the leaf table to the anchor, and the stages leaf and
    request work on unproved tables; the three stages are forked from one process that has read
    every file once, so all three configure from the same bytes; and the parent process accepts only
    if all three stages have passed (`chain/closed.h`; `verifier/src/cp_verify.cpp`; the format
    page's 13.6).
  - Not enforced: a test that vouches for itself. A source that defines
    `BLK_TESTED_TABLE_FOR_A_TEST` vouches for a table itself (by the source two toy tools define it,
    beside tests: `frame/handover_files.cpp` and `frame/handover_proof.cpp`), and one that defines
    `BLK_KEY_UNDER_FOR_A_TEST` or `BLK_ENTRY_UNDER_FOR_A_TEST` chooses the plans under which a file
    is checked, and an `EntryUnder` that is made that way skips the comparison of the key file's
    public values (`chain/closed.h`). None of the three compiles in a production build (the switch
    `SERVING_BUILD` in `blocks/fence.h`).
  - Not enforced: a caller that composes the blocks below the routines that check whole files. A
    caller that writes a block's steps out itself over the public pieces (`foldeq_columns_verify`,
    the channel's own calls), instead of calling `link_verify` or `lin_shared_verify`, is the author
    of a verifier: it says itself which side of a link is under test, and it checks what it chooses
    to check. Below the two routines that check whole files nothing compares the key file's public
    values: LIN and LINK speak of closeness alone.
  - Not enforced: that a result of the type `IfTested` cannot be read in another way. What the
    routine fills beside it (`ProofFacts`, the reader's position in the file) also tells whether it
    rejected.
  - Not enforced: what the calling program supplies. It is believed. That is its `ShapesOf` routine
    (the ranges and lookups that it derives from the files are whatever it says; the stop on the key
    file's public values enforces only that one process does not mix two such routines), its
    routines `links` and `more`, its `bind` and `decide`, plans made from sizes of its own choice,
    and a reader class of its own. `IndexReader` can be derived from, and on a side that is under
    test (the one-time test; the leaf table's side of a link) such a class answers `read()` as it
    likes. The routines that check whole files make their own readers, so this does not reach them.
  - Not enforced: a caller that sets out to defeat the types. What is claimed is claimed of code
    whose behaviour the C++ standard defines, which names no private member, declares nothing in
    namespace `blk` or `std`, and nothing at global scope before the library's headers, keeps the
    global `operator new`, and defines none of the switches of `blocks/fence.h`. Outside that
    rule `numbers/plan.h` lists the ways that were found; `chain/closed.h` names these of them:
    macros; casts and what does a cast's work; a reference kept beyond its object's life; an
    explicit instantiation or specialisation over a private name; declarations in namespace `blk`.
    The list in `numbers/plan.h` adds that `prover/` and `verifier/` declare names in namespace
    `blk` themselves, which only code review guards, and that the switch
    `BLK_PLANS_BY_HAND_FOR_A_TEST` does not compile under `SERVING_BUILD`, with which the plain
    targets `cp_prove` and `cp_commit` of `prover/build.sh` are not built: there the script does not
    install a finished program that holds the marker string of `blocks/fence.h`. All the listed ways
    apply to `TestedTable` and `UnprovedTable`. Among the declarations in namespace `blk` are a
    source file that defines one of their friend routines itself instead of including its header,
    and the struct `LinkAccess`, which `iop/link.h` leaves to the tests. One such way was run to a
    false acceptance (by `blocks/README.md`): a test program that is not in this repository reached
    a plan's private list of rounds through an explicit specialisation, lowered the number of
    combinations in the smallness test from 128 to 1, left that round's stated error at 2^-128, and
    got a false claim (a cell of 1000 under a bound of 3) accepted by the unchanged `run_verifier`,
    at the target of 128 bits, at toy size. This is not closed. By the same page a second way (an
    operator template declared before the headers) gives a wrong plan; a program of that kind, run in the tester's own process at the target of 128 bits, at toy size on made-up data, had a false claim accepted in 8 of 48 tries, and the same program without the declaration in 0 of 48 (17.5, item 11); a third (a
    planner's flag changed between two reads, by a replaced `operator new` or a second thread) was
    run to a false acceptance in a test program (by the head of
    `numbers/tests/planners_read_once_test.cpp`); it is closed
    in part only: for the fourteen listed planners, which copy their sizes first, and not for
    `plan_iopp`, the planners of `frame/`, the routines that run a proof or the
    plan's scope.
  - By the source, a program keeps key file, leaf file and batch under one sizing by giving its
    checkers one `Configured` and nothing else that is public. By the source `cp_verify` does so
    apart from its `links` callback and its `ShapesOf` routine: both work on one set of sub-proofs'
    plans and shapes, which the program builds before it calls `configure()` and which the library
    trusts. Also by the source, a key file or a leaf file that was made under another parameter file
    opens nowhere, because every leaf and vertex of a table's tree starts from the configuration's
    digest. By `blocks/README.md`, no file, argument or setting brought `cp_verify` to any of the
    ways of this item in the attacks that were made on it; that they cannot reach it rests on
    reading its source.
- The errors of this part's rounds, as the plan counts them (from the programs' source:
  `plan_lin_first`, `plan_lin_shared`, `table_fold_error` and `plan_record` in `numbers/blocks.h`).
  They are the planning routines' figures. No run shows them, and this page proves none of them. The
  one-time distance and the later distance of a table are those of the format page's 8.4, item 16.
  - `LIN point`: N k_col / |E|, with N the number of requests that share the point (1 in a key
    file). The table's closeness term is added: the sum over j = 1 .. v of mca(C_T, 2^(v - j)), plus list v / |E|. It
    is taken at the one-time distance in the one-time test and at the later distance in a batch's
    test. mca and list are as in part 3, section 6.
  - `LOC`, the table's columns: agree^nu. In the one-time test agree is 1 minus the one-time
    distance. In a batch's test it is 1 minus the later distance.
  - `LIN zeta`: k_row / |E|. `SEND seed` and `SEND answer`: part 1, 2.7. `SPLIT` and `SEAL`: 0. The
    entry `RECORD`: the error of part 1, 2.7, for one message, with the list size at the one-time
    distance, a_o = ceil log2 l_T, and the record's number of answers as s.
  - The rounds of the proximity tests are in parts 1, 3 and 4. What the closeness counts of these
    tests and of the table rest on is in part 3, section 6: within a third of the minimum distance a
    lemma; beyond it, the table included, the theorem on codes (Theorem 4.6 of the full
    version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf
    published at STOC 2026), which the programs' accounting takes as given; a statement on codes held to follow from it is proved
    in the Lean library (`FORMAT.md`, 13.7, condition 4).
  - The errors of `LIN point`, of the columns' `LOC` and of `RECORD` are counted from the table's
    distances, which the plan takes on the calling routine's word (`numbers/plan.h`; the two items
    above that begin "The table's distances are not bound" and "A later test relies on an earlier
    one"; part 1, section 6).
- A toy world's acceptance is an acceptance at toy soundness. The routines of this part ask whether
  the public numbers are well formed (section 1). They do not ask whether the stages' codes, the
  number of sampled columns and the lists of positions give any number of bits. That is asked by the
  check on the numbers inside `configure()` (`frame/`), not in the toy worlds of `iop/tests`.
- Zeros that no row proves: G_u's entries between the message and the tail and after the tail; the
  spare places of section 2. An honest prover places 0 there; nothing that is claimed reads them.
- Not in this part. LINK (two tables): part 10. How a request's rows on the table become one claim
  (g, psi, masks, tau): `iop/member.h`, part 8.
- The prover's duties that no verifier sees. Before anything of a batch, a link or the key file's
  one-time test is made, the prover asks one lasting counter of a stored table
  (`IndexHolder::begin`: a test on the table asks it through `lin_begin`, a link by a call of its
  own in `iop/link.h`). The counter counts the columns that a run will sample, in full, when the run
  is begun, whether or not a column is then opened. It is compared with the number of positions that
  the table's tail covers (`hiding_positions`: r_T in field 2, the one field for which the programs
  can be built; in field 1, by the format page's 16.1, r_T - 1 for an even r_T). (from the programs'
  source: `iop/index_cpu.h`, `frame/hiding_conditions.h`) Omega's opened positions over 1 + N tests
  are counted against its tail, at Omega's tree, in one number for all chains that read it. That
  every such run asks `IndexHolder::begin` first, and that the counter must outlast the process, are
  rules that the programs set themselves; no verifier can see whether they are kept. The holder of a
  table that the toy runs use (`index_cpu.h`) counts in memory, so its counter does not outlast the
  process.
- Two more choices of the programs. At a free slot of the binding tree is a uniform digest; the
  programs' source gives as the reason that a path then does not show how many requests a batch had.
  The fresh mask of a test's first fold can go out with an earlier message (4 (1), 6 (2)), but never
  in Omega's leaves.

## 13. Notes (`V.note`) for a second verifier

The verifier can be asked to note some of its own values by name (`V.note`); they change no byte.
(Parts 5 and 10 call them values on the way.) They are listed in the order in which a view's
verifier makes them.

- First chain: `LIN: (Eval)'s target`, `LIN: (Eval) on W<1>` (the coefficient eq(1; psi)).
- Shared chain: `FOLDEQ: e_0`; `FOLDEQ: (Rec_0)'s target`; `FOLDEQ: (Col_1)'s target`,
  `FOLDEQ: (Col_1) on G_u<0>` (= Gm_T(x_1)[0]), `FOLDEQ: (Col_1) on G_u<B>` (= Gt_T(x_1)[0]; made
  only if r_T > 0); `LIN: (Fold) on Omega<0>` (= eq(0; zeta)), `LIN: (Fold) on G_u<1>` (= - eq(1;
  psi_lo) eq(0; zeta) if k_lo > 0); then the tests' own notes (parts 3, 4).
- Second chain: `LIN: (Dot) on Omega<0>` (= g[0]), `LIN: (Dot) on W<1>` (= - eq(1; psi*)); then the
  tests' own.

A split's share is a received scalar and has no note.

## 14. Orders that must hold, and the malicious provers that try them

Each malicious prover is a build of `iop/tests/lin_test.cpp` with one switch (none of the switches
is in this repository: section 6 of this page; the table of malicious provers (two files, neither of
which is in this repository) says at which check each must fail). For four of the malicious provers
the programs' table of accepting controls, which is not in this repository, holds an
**accepting control**: a scratch copy of the verifier from which exactly the named binding is taken
out must accept the false proof. That shows that the attack is whole and that this binding is what
rejects it. All were run at toy sizes, on test builds, not on the released programs. The
reasons below are arguments, not proofs.

A planning routine that reads its sizes twice, once for a round's error and once for what the plan
says beside the round, can give a finished plan that names the one-time test and is counted as a
later test. The fourteen listed
planning routines copy each of their sizes once (`numbers/plan.h`, READ ONCE). By that header this
does not guard `plan_iopp` and the planning routines of `frame/`, the routines that run a proof, or
the plan's scope; by the same header these are examples, not a complete list. The test is
`numbers/tests/planners_read_once_test.cpp`. By its own head it shows nothing of the routines that
run a proof, of the planning routines that are not listed, of fields that no case changes, of groups
with mask rows or of another scope than the one-time one, and it turns each field one way. By the
same head, of the planning routines' 22 copies of an argument there are four that it cannot detect
when a reference replaces the copy: `tree` of `plan_lin_shared`, and `trees` of `plan_loc`,
`plan_cs` and `plan_base`. By the comment before its `operator new` it works only while the compiler
lets a replaced `operator new` write globals: a compiler that assumes otherwise may compile a header
that reads twice as if it read once; the test then succeeds and detects nothing, and nothing in this
repository would notice or report it. `numbers/plan.h` lists this way among those of a caller that
sets out to defeat the library's types (section 12, the item on such a caller). Two slips of
careless code have given a wrong plan: a size object that two threads use gave a plan that meets
its target and names numbers from which it was not counted, and so did an object that is used after
a move (in one case a round was counted at 2^-181.00 where 2^-174.93 was due). By `numbers/plan.h`
(READ ONCE) the copying covers a second thread's write for the listed planning routines: a round's
bound and what is stated with it come from the copy alone. By the same comment a copy that allocates
memory is not made in one step: the caller's code can change the original in the middle of the copy,
which then has the earlier parts unchanged and the later ones changed; that is an object that the
caller could have passed in. By `numbers/tests/plan_types_test.cpp` a `TestedIn` that was moved from
is not empty: a move is a copy.

- The table for which the plan was counted. Where the point and the columns are sampled, the verifier
  compares the bound table with the one for which the plan was counted, and ends the process on a
  difference (`require_table` in `chain/rounds.h`). Both sides say whether they run the one-time test
  or a later one (`TableTest`; there is no default), the plan says for which of the two it was
  counted, and a difference ends the process at the same place. A caller that says of the first test
  on a fresh table that it is a later test must give that test a `TestedTable`, which is enforced
  inside one process only (section 12).
- The table's distances. A key that is tested at a far radius must not serve a batch or a link that is
  sized at a near one: `foldeq_columns_verify` stops it. What is not enforced is in section 12.
- The claim's target before the first chain's seal. A prover that can name tau after it has seen the
  element that batches the rows of its own test makes two false rows cancel.
  `LIN_TARGET_AFTER_BATCHING`; control: tau is not sent on the first chain: accepted. (LIN itself
  cannot enforce this order: section 4, the caller's duty.)
- W before the shared point. A W that is changed by a vector that vanishes at a psi* known
  beforehand passes (Dot). What forbids it: W's root is under the first chain's seal, the seal under
  the binding tree's root, the root in the shared chain's first query. `LIN_W_AFTER_POINT`; control:
  the shared chain is bound to a fixed digest: accepted.
- The shared part from which a second chain goes on is the one in the view. Two shared parts that
  differ only in the randomness of their proximity tests are each valid. Mixed into one view, they
  are told apart only by the shared chain's seal in the second chain's first query.
  `LIN_OTHER_SHARED`; control: the seal is not followed: accepted.
- The record's answers before every gamma, and one record on a set of pads. Anyone who knows gamma
  first fits the answers to any near table: `FOLDEQ_RECORD_AFTER_POINT` (its own control: a verifier
  that leaves the record out of the first queries accepts). A second record on the same pads at
  other seeds gives exact linear functionals of the rows: `INDEX_SECOND_RECORD`; control: both
  sides' rows take the committer's seeds: accepted.
- G_u, its seed, its answer, then the positions. `CHAIN_ANSWER_OUTSIDE_AT`: each answer to an
  out-of-domain sample of a batch in turn is kept out of the prover's chain; the verifier's chain
  then samples other values.
- Beyond half the distance a word can speak for two tables. `INDEX_TWO_TABLES`: a word that is half
  one table and half another, with all sampled columns lucky. It is rejected by the folded record
  alone.
- False oracle messages: `FOLDEQ_OTHER_ROW` (another folded tail: only the (Col) rows read it),
  `FOLDEQ_OTHER_PAD`, `LIN_OTHER_W`, `LIN_OTHER_OMEGA`. Each is rejected at the last step's linear
  check of a proximity test. With two codes `LIN_OTHER_OMEGA` is rejected in the test on U, not on
  OM: the prover names the share sigma on the Omega that it sent, so OM's share holds and U's is
  false by eq(0; zeta).
- An injected layout, which ends the process on both sides:
  `LIN_MASK_IN_OMEGAS_LEAVES` (6 (11); its code is not part of this repository): a mask that is
  copied into Omega's leaves would be shown further at every request's list, and a few requests
  together would hold the whole mask.
- Stops for hiding on the prover's side. Two are covered by a test in which the process must end
  with the stop's own words: a batch with one request more than Omega's random tail covers; a batch
  that was not counted against the table. No such test covers the like stops for the splits' mask
  messages, for W, and for the table's columns.

---

# Part 10. The leaf file and LINK: the verifier of a leaf table, as mathematics and bytes

This part gives what the reader of a leaf file does: the file's bytes, and the verifier's side of
LINK, the block that ties a leaf table to the anchor.

The programs: `blocks/iop/link.h` (LINK), `iop/foldeq.h` (a table's shape, key and name; a folded
row), `frame/link_frame.h` (from the parameter file's lines to shapes and plan),
`frame/check_entry.h` (the file), `frame/configure.h`, `numbers/blocks.h` (`plan_link`),
`chain/rounds.h`; paths that begin with `iop/`, `frame/`, `numbers/` or `chain/` are in `blocks/`.
The conventions (index rule, eq, fold, codes, generator rows, rows, a group's rows) are those of
part 4, section 0; a folded row and a record are part 9's. Some letters differ from part 9's. This
part writes l, r for the table's code (part 9: l_T, r_T); s for the number of stored answers of a
row (part 9: a), and `samples` for the setting's number of answers to an out-of-domain sample (part
9: s); a for ceil(log2 l) (part 9: lam(C_T)); b for a row's number (part 9: r); w_b for part 9's
e_r; t0 for part 9's B; eq(z, e) for part 9's eq(e; z). The format page is `FORMAT.md` in the
directory above this one; its rules R13.1 and R13.3 define the file's bytes. In the programs' names,
and in the words that they print, "entry" is a leaf file, "copy" and "the further table" a leaf
table, and "the first table" the anchor.

The programs' interface. `make_entry`(c, the anchor's holder, the leaf table's holder, engine,
randomness, search, file) and `check_entry`(c, the anchor's key, file, out: the leaf table's key,
why, notes, place; each key inside a `TestedTable`, part 9, section 12) take c, which is what
`configure`(public files) has made, and nothing else that is public: plan, codes, shape and the
configuration's digest all come out of it. A **holder** is the committer's object that has a stored
table (`IndexHolder`); a table's **key** is what a verifier keeps of it (2 (a)). `check_entry`
answers `NO`, `YES` or `YES_WITH_WORK` (a round needs its proof of work; the reader then sees how
many bits the lists give alone). Public files that are not admitted: `NO`, before a byte is read.

## 0. What a leaf file says

A key file (part 6) fixes the anchor: 2^v rows over F, each a message of l entries that is encoded
with r random entries (its **tail**) under one code C_T = (n_T = 2^log_n, l, r); a column of the 2^v
code words is a leaf of one stored tree. The programs' source gives the reason for leaf tables: a
table's tails hide a bounded number of opened columns, so a committer makes further tables of the
same rows with fresh randomness. A **leaf file** is the file that comes with a leaf table. What an
accepted leaf file is meant to say: under the leaf file's root is a table that is close to one tuple
of code words, and the messages of these agree with the anchor's in every entry e < cut, row by row.
(At which distances: section 8, which tells what the plan counts. What a verifier can conclude is
argued in comments in `link.h` (its head and the comment before `class Link`). No proof is written.)

Public numbers, all from the configuration, each in one place (`LinkShape`; `IndexShape`; the plan's
setting):

- The tables' one shape: C_T = (log_n, l, r); v; s = the number of answers of a record for each row;
  the number of bytes of a salt in a leaf of a table's tree.
  The anchor and all its leaf tables have this one shape, and the hash's labels of their trees all
  have the tree number 0, which the format fixes: leaf tables are told apart by their roots alone.
  The last 3 s entries of a row's message are pads (3 is the degree of E over F):
  plain = l - 3 s entries come before them.
- The cut = plain: a function of the shape, not a number of its own. A choice of the programs: every
  entry before the pads is compared, also at places that no matrix uses and that an honest table
  fills with zeros. The reason in the comment before `LinkShape` in `link.h`: that covers whatever a
  later test on the table can read, and it leaves no place of a table that is neither compared nor
  fresh randomness.
- nu_0, nu_j: how many columns of the anchor and of the leaf table are sampled.
- The one shape of the oracle message that holds a folded row: R main rows over E under the group's
  code C_u = (n_u, l_u, r_u), no mask rows; as many answers to its out-of-domain sample as the setting's `samples` says (1: the parser refuses any other `answers`); the word rows_high (1 in every file
  that has been made); one leaf class for both messages and the mask that rides with them.
- The group's mask code; the word masks_ahead (1 in every file that has been made); the later stages
  of the proximity test (the first stage is not written: it is the group: code C_u, the mask code, k
  = 1 + ceil(log2 R), h = 0); the one choice masks_ride (part 4; 1 in every file that is made from a
  parameter file: `frame/link_frame.h`, from the programs' source).

## 1. The file, byte by byte

| item | bytes | what the reader does |
|---|---|---|
| tag | 4 | the bytes 'C' 'P' 'E' 'N'; another tag: rejected, `file-of-another-kind`. A file that ends before the 8 bytes of tag and version: rejected, `file-too-short` |
| version | 4 | the format's version; another: rejected by name |
| root_j | 32 | the leaf table's root; any 32 bytes are read |
| y_j[b s + i], b < 2^v, i < s | 24 each (3 canonical elements of F) | the answers of the leaf table's record, row by row. A coordinate that is not a canonical element: rejected, `the record's answers`. The record's seeds are not in the file |
| the link's chain | the rest | sections 3 to 7. It begins with the chain's 32 random bytes. It ends with the file: a byte left over is rejected |

No part has a length of its own: the file is read once, from the first byte to the last. Nothing of
the anchor is in a leaf file: its root, its answers and its table reader (part 9, section 0) come
from a key file that the reader has checked before. The leaf table's number is neither in the file
nor in the chain's first query.

## 2. What the reader forms before the chain starts

(a) The leaf table's key = (the tables' shape, the configuration's digest: both the reader's own;
root_j, the answers: both the file's). It has no other field: how a column is hashed (a stored tree
of depth log_n, leaves of 2^v symbols of F and one salt, the tree's number 0, all hashes started
from the configuration's digest) is a function of shape and digest (`index_tree`).

(b) The leaf table's name (`index_name`), derived, 112 bytes = root_j (32 bytes) | the
configuration's digest (32 bytes) | six words of 8 bytes, little-endian: log_n, l, r, v, s, salt
bytes. It is in no first query and in no file. In a leaf file's check it is read in one place: (c).
(Outside a leaf file's check the anchor's name has a second reader: the input of the commitment ID,
R4.11.)

(c) The record's seeds (`record_seeds`): a chain of its own with the label 2^32, no random
bytes, no parameters, and the name's 112 bytes as its statement; one round, with an empty prover's
message, of kind 3 (seeds); from its answer, for i = 0 .. s - 1 in turn: z_i in E^a, a = ceil(log2
l), then beta_i in E without 0. The seeds are computed where a row needs them (5 (a)), stored
nowhere, and compared with nothing: a record is its answers alone. How a query is made and how
elements are sampled from an answer: R4.8, with R6.4, R6.5 and R6.9.

(d) `link_key_fault` for both keys, a rejection and never the process's end. It asks: the key's own
fault (the shape is a code with room beside its pads; the digest is not zeros; 2^v s answers); that
the shape equals the link's in all six words; that the digest from which the key's tree is hashed is
the chain's configuration's digest, and nothing else passes (`link_same_configuration`). A table
reader that does not say from which digest its tree starts answers zeros, and is rejected like any
other digest; on the prover's side a holder that does not say ends the process.

The test is plain equality
(`iop/link.h`). The test of it is in `iop/tests/link_test.cpp`: a table reader that does not say its
digest is rejected both as the anchor's and as the leaf table's, and a holder that does not say ends
the prover's process. (from the programs' source)

## 3. The chain's first query (`link_bind`)

The chain's label: 0x4c4e4b01.

Parameters, in this order: what every chain's first query begins with (the format's version; field,
hash, widths, among them `samples`; the plan, entry by entry) | the item `configuration` = the
configuration's digest | the item `codes` = all codes of the chain (the tables' own code is not
among them) | the item `the link`, word by word:

    the tables' shape: log_n, l, r | v | s | salt bytes;   nu_0;   nu_j;   the group's code (its number in the item "codes");   the group's mask code;   R;   rows_high;   leaf class;   masks_ahead;   the later stages: a list (its length, then for each stage four words: code, mask code, k, h);   masks_ride.

Statement, in this order: the item `the first table` (the anchor) | the item `the further table`
(the leaf table). Each is the table's root as a string of bytes (the word 32, the 32 bytes), then
the list of its record's answers (its length, 2^v s, then the elements), by row, then by answer
(R4.6).

Random bytes: 32, the prover's, at the start of the chain's bytes in the file.

The plan's first entry is `RECORD`, once, for the leaf table (the anchor's is in the plan
of the key file's one-time chain). Its eleven words are 1 | 0 | 0 | 0 | 0 | s (a + 1) | s | 0 | 0 |
0 | 0. It is not a round of the chain: the channel steps over it. The query whose error it counts is
that of 2 (c).

Every bound object is asked for a fault of its own before it is written. What only several objects
together show (shape against codes and `samples`) is asked by `link_fault`, before the first query
is made. `link_prove` and `link_verify` make this query themselves from the holders or keys that
they are given, and own the channel. By the comment before `class Link` in `iop/link.h`, no caller
can therefore give them a chain that was begun on anything else; the way round that the tests have,
the struct `LinkAccess`, is in part 9, section 12. The verifier's rows take shape and answers out of
these items; of the two table readers it calls `name()` (for 2 (c)) and `read()` (the columns
against the root).

## 4. The rounds, message by message (`Link::steps_verify`; the plan: `plan_link`)

T0 and Tj are the two tables, the anchor and the leaf table; M_b^(i) in F^l and rho_b^(i) in F^r are
the message and the tail of row b of table i in {0, j}.

(1) Verifier, round `LINK point`: gamma in E^v. (The prover's message of this round is empty.) w_b
:= eq(gamma, b), b < 2^v.

(2) Prover, one prover's message, one tree, one root: the oracle messages G_0, G_j and, where
masks_ahead is 1, riding in the same leaves, the fresh mask P of the first fold of the proximity
test (one mask row). Where masks_ahead is 0 the tree holds G_0 and G_j alone, and the mask goes out
in the fold's own first message (part 4). What an honest G_i holds, as a vector g^(i) of entries
over E (part 9):

- entries 0 .. l - 1: the folded message U^(i)[e] = sum_b w_b M_b^(i)[e], pads included;
- entries t0 .. t0 + r - 1: the folded tail sum_b w_b rho_b^(i)[e - t0], where t0 is the least
  multiple of 2^ceil(log2 r) that is >= l;
- the pad of G_i's own out-of-domain sample where SEND puts it (part 1): with rows_high, at entry R
  l_u - 1 for one answer; zero everywhere else.

The room: t0 + r must not pass the first place of that pad (with rows_high and one answer: t0 + r <=
R l_u - 1). `foldeq_fault` asks it, and through `link_fault` and `configure()` public files that
miss it are rejected by name. The entries between l and t0 and after t0 + r are read by no row but
the sample row: an honest prover leaves them zero; nothing forces a malicious one to, and whatever
it places there moves the answer to its own out-of-domain sample and nothing else.

The layout: G_i's main table has 2^a_r rows (a_r = ceil(log2 R); rows beyond R are zero rows that
nobody sent) of 2^a_p places (2^a_p = l_u). With rows_high, entry e is in row e div 2^a_p at place e
mod 2^a_p, that is, at the index (e div 2^a_p) + 2^a_r (e mod 2^a_p) of the table read as one vector
by part 4's index rule. Without it, entry e is index e, and R must be a power of two.

(3) Verifier, round `SEND seed`: one seed (z, beta) for both out-of-domain samples (part 1). The
plan counts the out-of-domain sample's error twice.

(4) Prover: the answers alpha_0, then alpha_j (as many each as `samples` says). Round `SEND answer`:
the verifier says nothing; the round has a query of its own, whose prover's message is these
answers.

(5) Verifier, round `LINK`, one message: nu_0 + nu_j positions, each uniform in [n_T], with
repetition, in the order sampled: x_1 .. x_{nu_0} are the anchor's, x_{nu_0 + 1} .. x_{nu_0 + nu_j}
the leaf table's. If the round has proof-of-work bits, its one nonce is in the file before anything
is sampled.

(6) Prover, in the file only (in no query of the chain): the opening of T0's tree at the sorted set of distinct positions among the first nu_0
draws | the opening of Tj's tree at that of the last nu_j. A position that is sampled for both
tables is opened in each. An opening has, for each listed position, the column's 2^v symbols of F
and one salt, and then the sibling digests (R7.8; R7.19 and R8.7 for the two parts).

(7) Verifier, round `LINK phi`: phi in E^a, a = ceil(log2 l). (Empty prover's message.)

(8) The proximity test on the one group {G_0, G_j} (parts 3 and 4: BAT, then FOLD and CS stage by
stage, then BASE), with the mask P of (2).

(9) The chain's end: no byte of the file is left.

## 5. The rows that the verifier makes (all on the group {G_0, G_j}; all targets public)

Below, g^(i)[e] is "entry e of G_i's vector" as placed in 4 (2); a coefficient that is not named
is 0.

(a) The record's rows, made right after (4), the anchor's first: for i in {0, j} and each
answer q < s, with (z, beta) = seed q of table i, computed from its name (2 (c)):

    sum_{e < l, e no pad place} eq(z, e) g^(i)[e] + sum_{t < 3} beta theta^t g^(i)[plain + 3 q + t] = sum_b w_b y^(i)[b s + q],

with theta^t the basis of E over F. (The code writes eq(z, .) on all e < l and then adds, at every
pad place p, the unit coefficient (beta theta^t if the place is answer q's, else 0) - eq(z, p): so
the pad places of other answers end at the coefficient 0.)

(b) The columns' rows, made after (6), one for each draw, repeats counted, the anchor's draws first,
each in the order sampled: for a draw x of table i with the opened column (c_b)_{b < 2^v}, c_b in F:

    sum_{e < l} Gm_{C_T}(x)[e] g^(i)[e] + sum_{e < r} Gt_{C_T}(x)[e] g^(i)[t0 + e] = sum_b w_b c_b.

In words: what G_i holds, encoded again under the table's code, has at x the fold of the column that
the table's tree shows.

(c) The equality row, made after (7):

    sum_{e < cut} eq(phi, e) (g^(0)[e] - g^(j)[e]) = 0.

It reads nothing at or after the cut: no pad, no tail, no free room.

(d) The sample rows of G_0 and G_j: SEND's (part 1), with the targets alpha_0, alpha_j.

No target is named by the prover: the targets are 0, folds of the answers in the keys, folds of
symbols read under the roots, or the answers to an out-of-domain sample, which are in the chain
before the positions.

The order under BAT's powers (part 4): plain rows as made: (a) for table 0, (a) for table j, (c) |
the members' sample rows | generator rows as made: (b). Their number, 2 s + 1 + 2 samples + nu_0 +
nu_j, is compared with the plan's.

The group: code C_u, the group's mask code. Each member takes an aligned block of 2^ceil(log2 R)
main rows, so k = 1 + ceil(log2 R) and h = 0 (no member has mask rows). The test's first stage is
this group, made from the shape's fields and written nowhere else.

## 6. What is checked on the spot, and what is deferred

Before a byte is read: public files that `configure()` does not admit are a rejection
(`the public files are not admitted`). It asks, by name, `link_fault` on shape, codes and `samples`;
that every round of the plan reaches its target (by the planner's figures: part 1, section 6); and
the ten named conditions on the parameter files of `frame/hiding_conditions.h` (the format page's
16.1 defines them; its 8.4, item 23, has their names). These are not all that hiding asks (from the
programs' source, that header and `frame/configure.h`). As `configure()` calls them, the condition
`table` counts no link's columns on the anchor: how often a link was begun on a table is not a fact
of the files, and the table's holder counts it. By the comment before `hiding_conditions_missed` in
that header, passing them proves nothing by itself, and that comment lists sixteen further
conditions that are not arithmetic on parameters and must be covered by tests of the code.

Also before a byte of the leaf file is read: the anchor's key file was accepted. `check_entry` takes
the anchor as a `TestedTable` (part 9, section 12). An empty one has passed no check: it is what a
rejected key file leaves, and what a caller has whose key file was never checked. It is a rejection,
in the words `the first table has passed no check: its key was refused, or was never checked`, with
the place `the parameters`. By the source it is a rejection and not an end of the process because it
can depend on a key file's bytes. A second verifier needs the same step: it accepts no leaf file
under an anchor whose key file it has not accepted. After that step the process ends if the anchor's
key file was checked under other public values (ranges, rows, codes) than this configuration's; that
depends on the caller's own objects alone. The test of both is `frame/tests/tested_table_test.cpp`,
with the lines `a-refused-key-gives-a-refusal` and `an-entry-under-the-keys-public-values` of the
programs' table of accepting controls (not in this repository). These two lines are not accepting
controls: no false file is made, and the table asks only that this test fails once the named check
is taken out of a scratch copy. (from the programs' source, `frame/check_entry.h`, `iop/foldeq.h`)

On the spot: the head; the canonical form of every element read; the two keys (2 (d)); proof of
work, where a round has proof-of-work bits;
both openings against root_0 and root_j (`the opened columns of the table do not lead to its root`);
inside the test its own openings and BASE's two checks; the end of the file.

Deferred: every equation of section 5. No equation on field elements is tested in LINK itself; all
go into the group's rows and so into BASE's claim.

The verdict is yes if and only if nothing above rejected. The reader then keeps the leaf table's key
(shape, configuration's digest, root_j, answers), together with the setting under which the link
passed and the public values of the anchor's key file (a `TestedTable`), for every proof that is
made on that table.

One verdict, one session: `link_verify` is given a session by its caller (`chain/session.h`). What
an accepted opening shows under a root is entered in the session's one record for that root, and a
second value at a vertex rejects. In one link each stored tree is opened at one list, so in a leaf
file that these programs make the record has nothing to clash with (the other case: section 5 at the
head of this page). It matters where several links on one anchor are judged for one verdict, which
no program does. `check_entry` makes a session of its own.

The reader without the link (`read_entry_table_unproved`): head | root_j | answers -> the leaf
table's key, with every test of form of sections 1 and 2 and nothing of sections 3 to 7.
`check_entry` begins with it. Anyone who takes a table from it relies on another run of
`check_entry` on the same bytes.

Why and where. `check_entry`'s "why" is the first reason in words:
`the public files are not admitted`; a text of a fault of the link's public object (`link_fault`),
or the words of a plan that misses its target, neither of which is reached after an admitted
configuration; `the first table has passed no check: its key was refused, or was never checked`; a
head's name; `the copy's root`; `the record's answers`; a text of a key's fault; else the channel's
own words for what it met first (a nonce that does not give the proof of work; an opening that does
not lead to its root; one of BASE's two checks; a byte left over; ..). Its "place" says where in the
file's order that fell, because one reason can come from several places: `the parameters` |
`head, root and answers` | `the keys` | `before the test` (steps (1) to (7)) |
`the proximity test` | `the end`. No decision reads either.

## 7. Trees and lists

- T0's and Tj's trees are stored trees: perfect binary, depth log_n, a leaf = 2^v symbols of F and
  one salt; leaves and vertices start from the configuration's digest; the tree's number in their
  hash's labels is 0. They are in no chain's list of trees and have no place there: in the round
  `LINK` they are opened in the order of the statement's two items (R8.7).
- The chain's own trees: tree 0 for the prover's message of 4 (2): a leaf holds G_0's R symbols over
  E, then G_j's R, then (with masks_ahead) the mask's one: 3 (2 R + 1) elements of F, and one salt;
  then, with masks_ride = 1, one tree for each later stage of the test (part 4). Each is opened at
  exactly one list: tree 0 at the first code switch's list. Without masks_ride a later fold's fresh
  mask and the last step's masking word have trees of their own (`iop/iopp.h`, `plan_iopp` in
  `numbers/blocks.h`; from the programs' source).
- The two tables' columns are opened at the two lists of the round `LINK` and at no other list of
  this chain.

## 8. What the rounds' errors count (`plan_link`): the plan's reading, not a proof

- `LINK point`: v / |E| (two tuples of rows that differ in an entry before the cut fold to vectors
  that differ there) + the leaf table's closeness term at the one-time distance + the anchor's at
  its later distance. By `iop/link.h` that term is proved only up to a third of the
  minimum distance; on what it rests beyond it: part 3, section 6. That a one-time test passed on the anchor, at these distances and under this
  plan's setting, is enforced inside one process only. LINK's verifier takes the anchor as a
  `TestedTable`. Given a reader that is not one, the process ends where the anchor's columns are
  read, unless the file is rejected earlier. Given an empty one, which has passed no check, the leaf
  file is rejected: by `check_entry` before a byte is read (section 6), and by `link_verify` at the
  two keys (2 (d)), by the key's own fault (by the source, an empty table's reader names no
  configuration). Across processes, as in `cp_verify`, no routine of the library enforces it (part
  9, section 12; from the programs' source, `chain/closed.h`, `iop/link.h`, `iop/foldeq.h`,
  `frame/check_entry.h`).
- `SEND seed`: 2 x the out-of-domain sample's error at the group's list size.
- `LINK`: the larger of agree_0^nu_0 and agree_j^nu_j, not their sum. The plan's reason: which of
  G_0, G_j holds another row than its table's fold is fixed before the sampling, and that table's
  own list must miss it. agree_0 is taken at the anchor's later distance and agree_j at the one-time
  distance (`plan_link`; part 9, section 12).
- `LINK phi`: ceil(log2 l) / |E|.
- Then the proximity test's rounds (parts 3, 4).

At `record.f2.txt`, a parameter file of the programs, a link reads 62 columns of each table
(`docs/format/08-parameters.md`, 8.5; this page reports no leaf file of that size).

## 9. Limits of this part

What nothing has compared with the programs:

- section 8 (what the errors count);
-
- more than one answer to an out-of-domain sample (no parameter file gives it);
- the places of rejections (`check_entry`'s "place", section 6);
- the words of rejections (section 6). `files_check.py` has one
  comparison of them (routine `cheats`): for each false file of section 10, the words of the
  programs' reader and the second verifier's own rejection must name the same check.

What this part does not decide, what these sources do not have, and what no test covers:

- The exact bytes of a first query, of an item, of a round's query and of sampled values: this part
  gives the routines' order; the format page defines the bytes.
- 4 (6): no file lists a position: the parser refuses any `listed` but 0.
- 5 (b): Gm and Gt are part 4's for field 2 (Goldilocks); for the circle code of field 1, for which
  the programs of this repository cannot be built, the generator row has another form. For the
  table's code, l need not be a power of two.
- 5, the order of rows under BAT: what supports it is a second verifier's agreement on BAT's target
  at s = 1; the rule itself is part 4's.
- 4 (2): where SEND puts a pad, and the sample row's cube for a message whose R is not a power of
  two, are part 1's. In every leaf file that a second verifier has run, R was a power of two.
- Section 8 is what the planning code counts. What a verifier can conclude is argued in comments in
  `link.h` (its head and the comment before `class Link`).
  No proof is written.
- Plan and shape are two writings of one thing. Beside the forms that take c, `check_entry` and
  `make_entry` have a plain form that takes a plan and a shape as two values (the tests use it; so
  does `link_verify`, like every block's entry point). How the draws of the round `LINK` are divided
  is in the shape. The round's error was computed by the plan's maker from its own nu_0 and nu_j,
  each the least number that reaches the target for its own table, and the eleven words of the
  plan's entry keep only their sum. With a plan and a shape that do not come from one source (the
  two numbers swapped, say), an acceptance would rest on fewer columns of one table than the
  plan's bits count. No file can do that: neither number is a line of a public file; `configure()`
  derives both sides by one rule, and `link_plan` ends the process if they differ. So anyone who
  calls the plain forms must take plan and shape from one source; the forms that take c do.
- What the programs do against such a caller (from the programs' source). A plan in memory has nu_0
  beside the round `LINK` (`first_list` in `numbers/plan.h`; it is in no first query), and both
  sides of LINK end the process if the shape has another nu_0 (`require_first_list`). (from the
  programs' source, `iop/link.h`) That is true only of a plan that the routines of
  `numbers/blocks.h` made: by the comment "Two types" in `numbers/plan.h` the compiler lets only
  these routines add a round with a verifier's message and compute its bound, in a source that uses
  the headers as they are meant. The head of `numbers/plan.h` says that the types of a plan guard
  against mistakes, not against a caller who sets out to defeat them. The plain forms of the key
  file and of the leaf file are fenced off for tests alone. The leaf file's are `check_entry_under`
  and `make_entry_under`, which take plan and shape through the type `EntryUnder`; only
  `frame/check_entry.h` itself, from c, and a file that defines `BLK_ENTRY_UNDER_FOR_A_TEST` can
  make one. `link_verify` takes both as they are given.
- The orders that must hold in LINK, by the comment before `class Link` in `iop/link.h`. They are
  the source's arguments, not proofs. Both roots and both records come before gamma: a committer
  that knows gamma first makes a leaf table whose difference from the anchor folds to zero at gamma
  (`LINK_TABLE_AFTER_POINT`; by the table of malicious provers (two files, neither of which is in
  this repository) its control, a scratch pair of prover and verifier whose first query leaves the
  table out, accepts). G_0, G_j, the seed and both answers come before the positions: a prover that
  knows the positions before it sends G_j fits G_j's random tail to the sampled columns and gives
  G_j any message that it likes (`LINK_ROW_AFTER_POSITIONS`). phi comes after G_0 and G_j: otherwise
  two different rows are chosen to agree at phi; the comment names no malicious prover for this
  order. (from the programs' source) The two malicious provers are builds of
  `iop/tests/link_test.cpp` that are not in this repository.
- The tables' distances are not bound: part 9, section 12. `link_key_fault` (2 (d)) does not look at
  them; they are compared where a table reader's columns are read, and a difference ends the process
  (from the programs' source, `iop/link.h`).
- The library's tests run at toy size alone, on the CPU holder alone, in worlds that a test
  makes by hand and in files that the library's tests make from a parameter file
  (`blocks/frame/tests/toy_file.h`).
- Hiding. There is one exact test by linear algebra at toy size
  (`blocks/tests/hiding/link_exact_test.cpp`: two random tables, one list of challenges). Its
  result: with the verifier's messages kept fixed, the records and the field elements that are
  sent have the same distribution for both tables. The test leaves out roots, sibling digests
  and salts. It takes uniform randomness for granted. It covers the pair of tables and the list of
  challenges that were run, no more. By the head of `exact.h` a leak can show for special witnesses
  only: three faulty provers, tried with random pairs, all passed (one leaves zero entries unmasked,
  one multiplies a mask by the witness, one sends one bare bit of the witness). That head also names
  as not seen running time, and a prover that stops, or samples again, for a reason that depends on
  the witness. By the test's head the challenges are fixed on the prover's side only: no verifier
  runs in it, and none accepts such a proof. Its prover is the CPU reference. (from the programs'
  source: the test's head and the head of `blocks/tests/hiding/exact.h`) Three negative controls of
  the test are told apart: the leaf table made with the anchor's tails; the leaf table's tails
  constant; both tables' pads zeros. Before any randomness
  is sampled, the prover asks both tables' holders whether their rows' tails cover the link's
  columns as well; on a "no" from either, no leaf file is made.
- Sizes. Toy leaf files (`frame/tests/check_entry_test.cpp`, field 2, v = 3, n_T = 2^14, l = 803, r
  = 1,001, 128 bits with 6 proof-of-work bits): about 143,000 bytes (of which the answers are 192,
  s = 1). From the toy run's two public files: nu_0 = nu_j = 76, about 131,000 bytes. The exact lengths move with the sampled positions (pruned paths): they are not test
  vectors.
- Values on the way for a second verifier (an argument of `link_verify` and `check_entry`; no
  decision reads them). LINK's own, in the order made: first
  `LINK: the target of the first table's record's first row`,
  `LINK: the target of the further table's record's first row`; then
  `LINK: the fold of the first table's column at its first draw`,
  `LINK: the fold of the further table's column at its first draw` (the targets of the first row (b)
  of each table), `LINK: the equality row's coefficient at entry 0 of the first table's folded row`
  (= eq(phi, 0)). BAT's, FOLD's and CS's own come beside them.

## 10. False leaf files, and where a reader must reject them (nine files, and one false pair)

They are made on the toy run's two public files (nu_0 = nu_j = 76), at toy size, with other programs; the switches of which the table speaks are not in this repository (section 6 of this page). In the first six and in
`BASE_WRONG_WORD` the key file is untouched and valid. In the first six every opening leads to its
root, every nonce is right, and all answers and round messages are those of a prover that follows
the rules on the rows as it has them. So nothing before the end of the proximity test can reject,
and the rejection is BASE's linear check. (That it is the linear one of BASE's two checks is seen in
every run, not argued.)

| name | how it is made | what is false |
|---|---|---|
| `LINK_ONE_CELL` | no switch; the leaf table's rows differ from the anchor's in one cell before the cut; the honest prover | the equality row (5 (c)) and nothing else |
| `LINK_SHORT_CUT` | a switch on the prover's side, and the leaf table differs in the last entry before the cut | none of the prover's own rows (its equality row stops one entry early); the row that the reader forms with the right cut |
| `LINK_OTHER_COPY_ROW` | a switch alone; equal tables; entry 0 of G_j's row raised by 1 | the equality row, all nu_j column rows of the leaf table and the rows of its record; none of the anchor's |
| `LINK_OTHER_COPY_TAIL` | a switch alone; equal tables; the first entry of G_j's folded tail raised by 1 | all nu_j column rows of the leaf table and nothing else (the equality row reads no tail) |
| `FOLDEQ_OTHER_ROW` | a switch alone; both G_0 and G_j hold another folded tail than their table's fold, the same change in both | the column rows of both tables; the equality row holds |
| `LINK_FAR_TABLE` | the leaf table's committed word has the true table's symbols at even positions, another table's at odd ones; all that is sent speaks of the true table | the committed word at the odd positions |
| `BASE_WRONG_WORD` | a switch of the proximity test's last step alone (not LINK's own); equal tables | no row: all rows hold. In the link's one last step the prover shows a masked message that meets the linear claim and is not the underlying message of the opened symbols: BASE's linear check holds, and its other check, the shown word against the opened symbols, fails. It is the one file of this list that is rejected at BASE's second check. (This is seen in the run, not argued here: BASE is part 3's) |
| a column altered in the leaf table's opening | bytes of an honest file | nothing reaches a row: the leaf table's opening does not lead to its root |
| the two openings swapped | bytes of an honest file | the same, at the anchor's opening |
| another key | no file of its own: the honest leaf file, byte for byte, judged under the root and the record of a second commit run for the anchor; under the key file's own root it must be accepted | false is only the pair (file, anchor's root): all challenges move, and it is rejected at the first thing that depends on one (in the run: a proof-of-work test) |
