# The parsed form

What `parse_files.py` writes, byte by byte. [README.md](README.md) says what the parser is for, what
may be trusted, and how to run it. Numbers such as R7.9 are rules of
[docs/format/FORMAT.md](../../docs/format/FORMAT.md), which defines the three original files. Terms:
[docs/glossary.md](../../docs/glossary.md); "run's value" (R4.5), "short position" (R7.2) and "given
digest" (R7.9) are FORMAT.md's.

| file | what it is | read by |
|---|---|---|
| `key.parsed.bin`, `proof.parsed.bin` | a key file and a proof file, as [seventeen fields](#2-a-key-file-and-a-proof-file-seventeen-fields) | a reader written in Lean: `lean/OnBytes/Maker/Carrier.lean` (`read`), run by `lean/Ask/KeyCompiledSmall.lean` and `lean/Ask/ProofCompiledSmall.lean`: an untrusted reader, for tests; it tests the form alone: see that file's comment for what it does not test |
| `leaf-<k>.parsed.bin` | the leaf file that the proof's head names, as [ten fields](#3-a-leaf-file-ten-fields) | a reader written in Lean: the test program of `lean/OnBytes/ShellOf/EntryParsed.lean` (`rdLeaf`), run by `lean/Ask/EntryParsed.lean`: an untrusted reader; see that file's comment for what it does not test |
| `<parsed file>.seeds`, `leaf-<k>.parsed.bin.seeds0` | [the seeds of a table's record](#4-the-seeds-files) | a control ([section 4](#4-the-seeds-files)) |
| `<parsed file>.order` | [what else is needed to make the original file again](#5-the-order-file) | tests; whoever makes a run's value again ([section 6](#6-what-is-computed-and-not-copied)) |
| `key.lines.jsonl`, `leaf-<k>.lines.jsonl`, `proof.lines.jsonl` (with `--lines`) | [every item of a file by name](#9-the-lines), one line of JSON each | people |

A reader of a parsed file needs no hash, no plan and no public file. It reads the fields one after
the other and rejects the file if a byte is left over.

## 1. Encoding

It is the format's own (R1.1 to R1.7).

| type | bytes |
|---|---|
| `u64` | 8 bytes, the lowest first (R1.1) |
| `flag` | a `u64` that is 0 or 1 |
| `F` | an element of the base field: one `u64` (R1.3) |
| `E` | an element of the extension field: three `u64` (R1.4) |
| `digest` | 32 bytes, with no length before them (R1.5) |
| `bytes` | a `u64` length, then the bytes (R1.6) |
| list of T | a `u64` count, then the items (R1.7) |
| a struct | its fields in the order given: no names, no padding |

Elements, digests and salts are copied from the original file as they are. `parse_files.py` refuses
a file in which a word of an element or of a column is not below the prime p (R1.3). A verifier
still has to test that itself: a parsed file comes from an untrusted parser.

## 2. A key file and a proof file: seventeen fields

Both files hold three [hash chains](../../docs/glossary.md#chain) and one slot of a
[binding tree](../../docs/glossary.md#binding-tree), so one layout serves both. "Computed" marks a
digest that the parser hashes and that is in no file. It is not the whole list of what is in no file:
[section 6](#6-what-is-computed-and-not-copied) has every field's kind.

| # | field | type | in `key.parsed.bin` | in `proof.parsed.bin` |
|---|---|---|---|---|
| 0 | magic | 8 bytes | `FPKEYC01` | `FPPRFC01` |
| 1 | root | digest | the anchor's root: bytes 8 to 39 of the key file (R13.2) | the leaf table's root: bytes 8 to 39 of the leaf file (R13.3). The proof file does not hold it |
| 2 | answers | list of E | the stored answers of the anchor's record, by row and within a row by answer (R13.8) | those of the leaf table's record, from the leaf file |
| 3 | run1 | digest | the one-time chain's run's value (R4.5). Computed | the first chain's. Computed |
| 4 | copy | u64 | 0 | the number in the proof file's head (R14.2) |
| 5 | rounds1 | list of Round | all rounds of the one-time chain, its last round `SEAL` too | all rounds of the first chain, `SEAL` too |
| 6 | run2 | digest | the shared chain's run's value. Computed | the same |
| 7 | rounds2 | list of Round | all rounds of the shared chain, `SEAL` too | the same |
| 8 | run3 | digest | the second chain's run's value. Computed | the same |
| 9 | rounds3 | list of Round | all rounds of the second chain | the same |
| 10 | trees | list of TreeRows | one entry for each tree that the file sends and opens, by increasing number | the same |
| 11 | stored | list of Row | the anchor's opened leaves, by increasing leaf, each once | the leaf table's opened leaves |
| 12 | bindRoot | digest | the binding tree's root (R7.18). Computed | the same |
| 13 | place | u64 | the slot: 0 (R13.6) | the slot (R14.7) |
| 14 | places | u64 | the number of the binding tree's leaves: 1 | 2^depth of the batch's binding tree |
| 15 | slotSalt | bytes | the salt of the slot's leaf | the same |
| 16 | slotPath | list of digest | empty | the slot's siblings as in the file, the leaf's sibling first (R14.7) |

`places` follows from the public files. It is here so that a reader can compare it with the number
that it derives itself.

The root and answers in `proof.parsed.bin` are the prover's word (copied from the leaf file that the
proof's own head names). They are worth nothing for the user's
[commitment ID](../../docs/glossary.md#commitment-id) until a verifier has tied them to it: the leaf's
link checked against the anchor, the same leaf root and answers in the proof's and the leaf's parsed
files, the same anchor root and answers in the link, the key's check and the ID's string.

The ID is formed from the public files and the anchor's root and answers (R4.11), and a leaf table is
tied to the anchor by the leaf file's link chain alone (R13.9 to R13.11). The parser checks only the
cut of that chain. A leaf file on another table, with a false link that is well cut, is parsed, and
`proof.parsed.bin` then repeats that table's root.

**Round** := items : list of Item · nonce : u64

Every round of the chain's plan has one Round, in order. (A plan's entry `RECORD` is not a round:
R4.3.) A round with an empty message has no items. `nonce` is 0 in a round without proof of work.

**Item** := kind : u64 · value

| kind | value | |
|---|---|---|
| 0 | digest | a root |
| 1 | F | never written |
| 2 | u64 | never written |
| 3 | E | any element that is not an answer |
| 4 | E | an answer to an out-of-domain sample (R8.3) |

The items are the round's whole message in its own order; its roots come first (their order: R8.6).
In the three files every byte of a message outside a root belongs to an element of E (no plain word:
R8.8), so kinds 1 and 2 do not occur.

**TreeRows** := tree : u64 · chain : u64 · rows : list of Row

- `tree` is the tree's index among all roots that the file sends, counted from 0 through the three
  chains in order. It is not one of the two numbers of R7.13, though it can be equal to one. A tree
  that the second chain adopts from the shared chain keeps its index. The weight table's tree is not
  counted.
- `chain` is 0, 1 or 2: the chain whose round sent the root. It repeats what the rounds say: the
  tree's index falls among the roots in that chain's rounds, and the plan fixes how many roots each
  round sends. Like `places`, it is here so that a reader can compare it with what it derives
  itself. A verifier must not take it from here alone.
- `rows` holds the leaves at which the tree is opened anywhere in the file, by whichever chain, and
  no others, by increasing leaf. A leaf that two openings share is written once; the parser refuses
  the file if its two columns or its two salts differ (R7.14).

**Row** := leaf : u64 · column : list of F · salt : bytes · path : list of digest

- `leaf` is the position in the tree itself: the short position, below 2^depth (R7.2, R7.15).
- `column` is the opened column (R7.1), without the fill of R7.3.
- `salt` is the leaf's salt as in the file, and nothing else. It has 40, 48, 56 or 64 bytes, the same
  number for all leaves of a tree: the plan fixes it (R7.3). So does `slotSalt`.
- `path` is the leaf's whole Merkle path: as many digests as the tree is deep, the leaf's sibling
  first. Entry k is the value of the vertex at place (leaf >> k) xor 1, k levels above the leaves.
  Computed: the file holds only the digests that cannot be computed from the other opened leaves
  (R7.8, R7.9).

## 3. A leaf file: ten fields

A leaf file has one chain, the link's, which opens two weight tables: the anchor and the file's own
leaf table. It has no slot. The layout follows that of section 2, with the same Round, Item,
TreeRows and Row.

| # | field | type | what it is |
|---|---|---|---|
| 0 | magic | 8 bytes | `FPENTC01` |
| 1 | root0 | digest | the anchor's root, from the key file. The leaf file does not hold it |
| 2 | answers0 | list of E | the stored answers of the anchor's record, from the key file |
| 3 | root | digest | the leaf table's root: bytes 8 to 39 of the leaf file (R13.3) |
| 4 | answers | list of E | the stored answers of the leaf table's record, as they follow the root |
| 5 | run | digest | the link's chain's run's value. Computed |
| 6 | rounds | list of Round | all rounds of the link's chain (R13.10) |
| 7 | trees | list of TreeRows | the chain's own trees; `chain` is 0 |
| 8 | stored0 | list of Row | the anchor's leaves that are opened after the round `LINK` (R7.19) |
| 9 | stored1 | list of Row | the leaf table's leaves that are opened there |

The leaf table's number k is in no field. It is in the file's name, and it is `copy` in
`proof.parsed.bin`.

Two pairs stand in two parsed files each: `root0` and `answers0` are fields 1 and 2 of
`key.parsed.bin`, and `root` and `answers` are fields 1 and 2 of `proof.parsed.bin`. A verifier
has to take each pair from one place, or compare the two copies.

## 4. The seeds files

**Seeds** := list of Seed · **Seed** := z : list of E · beta : E

The file has no magic and nothing after the list. It holds the seeds of one weight table's
[record](../../docs/glossary.md#record) (R4.8), as many as a row has stored answers. `z` is the point of the out-of-domain sample and `beta` the factor of the pad
(R6.9). All of it is computed: the seeds are drawn from the hash at the table's name (R4.10). They
are public values, which anyone can compute from the public files and the table's root. They have
nothing to do with the provider's secret [seed](../../docs/glossary.md#seed).

A seeds file is a control: the seeds that a verifier draws can be compared with it. A verifier has to
draw the seeds itself, from the table's root and the public files; it need not read the file. Whatever
does read a seeds file has to hold its values against seeds that it draws itself: they come from the
untrusted parser.

| file | whose record |
|---|---|
| `key.parsed.bin.seeds` | the anchor's |
| `leaf-<k>.parsed.bin.seeds0` | the anchor's (the same bytes) |
| `leaf-<k>.parsed.bin.seeds` | the leaf table's |
| `proof.parsed.bin.seeds` | the leaf table's (the same bytes) |

## 5. The order file

A parsed file gathers the openings by tree. The order file says in which round each stood, and
holds the few bytes of the original file that the parsed file leaves out. With both, the original
file can be made again byte for byte, which is how the tests check the parser
(`read_parsed.py`). It also holds the 32 random bytes of each chain that has some, with which a run's
value can be made again ([section 6](#6-what-is-computed-and-not-copied)).

**Order** := magic : the 8 bytes `FPORDER1` · kind : u64 · tableHere : flag · chains : list of Chain · openings : list of Opening

- `kind`: 1 key file, 2 leaf file, 3 proof file.
- `tableHere`: 1 if the parsed file's `root` and `answers` stand in the original file, after its head
  (key file, leaf file); 0 if not (proof file).

**Chain** := label : u64 · nonceBytes : u64 · hasRandom : flag · random : digest · names : list of bytes · hasNonce : list of flag

- `label`: the chain's label (R13.4, R14.3, R14.9, R14.10).
- `nonceBytes`: how many bytes a nonce has in the original file; 0 if no round of the chain has one.
- `hasRandom`, `random`: whether the chain's 32 random bytes stand in the file, before its first
  round (R4.5), and these bytes (zeros if not).
- `names`, `hasNonce`: for each round its name in the plan, in ASCII, and whether it has a nonce.

**Opening** := chain : u64 · round : u64 · stored : flag · tree : u64 · idx : list of u64

The openings are in the file's order. `chain` is the chain that reads the tree, `round` the round's
number in that chain, from 1. With `stored` = 0, `tree` is the index of section 2. With `stored` = 1
the tree is a weight table's, and `tree` is 0, or in a leaf file 0 for the anchor and 1 for the leaf
table. `idx` lists the leaves at which the tree is opened there, strictly increasing.

The original file, from both:

1. the head: `CPIKH821`, `CPENH821` or `CPPFH821` (R13.1, R14.2), and in a proof file `copy` in 4 bytes;
2. if `tableHere`: `root`, then `answers`;
3. for each chain: `random` if `hasRandom`; then for each round the values of its items, the nonce
   in `nonceBytes` bytes if it has one, and the round's openings;
4. an opening: for each leaf of `idx` its column and its salt; then the given digests, which are the
   whole paths pruned by R7.9: level by level from the leaves upward, and within a level by
   increasing place, the sibling of each vertex that is an opened leaf or above one, if that sibling
   is not itself an opened leaf or above one;
5. after the first chain of a key file or a proof file: `place` in 8 bytes, `slotSalt`, `slotPath`.

## 6. What is computed and not copied

Every field is of one of four kinds.

| kind | meaning |
|---|---|
| cut | bytes copied from one of the three files as they are (`copy` and a nonce are widened to 8 bytes) |
| plan | follows from the public files, through the tables that `pubtool` writes from them, or is a constant of the format; the parser hashes nothing for it |
| count | counted from the order of what is cut |
| hash | comes out of the parser's own hashing |

Where a cut falls is not the plan's alone. An opening's length follows from the positions that were
sampled. So after a chain's first opening every offset, and how many trees and rows there are, come out
of the parser's hashing. The bytes themselves are copied.

No byte of the public files or of the request is in a parsed file. But every field of the kinds plan
and hash hangs on the public files (if the tables are `pubtool`'s of these public files, which the
parser does not check in full). In `proof.parsed.bin` these hang on the request too: `bindRoot`,
`run2`, which trees have an entry, which leaves have rows, the computed digests in their paths, where
each cut falls, and how many rows there are; and so do the openings in the proof's order file. `run1`
and `run3` do not, but for where `run3`'s random bytes are cut. A key's and a leaf's parsed file hang
on no request.

The last column says whether a verifier that has the parsed file, the public files and the request
can make the field again by itself; where it also needs the order file, the cell says so. A "yes, by
replaying" takes the chain's run's value as given.

**`key.parsed.bin` and `proof.parsed.bin`**

| # | field | kind | from what | can it be made again? |
|---|---|---|---|---|
| 0 | magic | | a constant of the form | |
| 1, 2 | root, answers | cut | key: the key file. Proof: the leaf file that the proof file's head names. How many answers: plan | no: the prover's word ([section 2](#2-a-key-file-and-a-proof-file-seventeen-fields)) |
| 3, 8 | run1, run3 | hash | one hash call (R4.5) on the chain's label, on the digest of the chain's parameters (R4.4), which come from the public files, and on the chain's 32 random bytes, which are cut from the file | **not from the parsed file alone**: the 32 random bytes are in the order file (`random`, a plain cut). With them: yes, by that one hash call. A verifier that does not read them takes these two digests as part of the prover's message. That neither is 32 zero bytes (R4.5) is the verifier's to ask |
| 4 | copy | cut | the 4 bytes in the proof file's head; in a key's parsed file the constant 0 | no: the prover's word |
| 5, 7, 9 | rounds1 to rounds3 | cut, plan | each item's value and each nonce: cut (0 in a round without proof of work: a constant). How many rounds, how many items, each item's kind and length: plan | the plan's share: yes |
| 6 | run2 | hash | as run1, with `bindRoot` in the place of the random bytes (R13.6, R14.9) | yes, from `bindRoot` |
| 10 | `tree`, `chain` | count | the tree's index among the roots sent; its sender | yes, from the rounds and the plan |
| 10, 11 | which trees have an entry | hash | found by trial: after a round that samples positions the parser hashes the next bytes of the file and sees to which roots of the chain's list they lead. So it hangs on the prover's bytes | which trees have to be opened follows from the plan and the rules (R7.19, R13.12, R14.9, R14.10), not from the replay |
| 10, 11 | which leaves have rows; `leaf` | hash | the short positions (R7.2) of the positions that are sampled from the answer to the round's query (R5.4, R6.1, R6.6). What a chain hashes is listed below the tables | yes, by replaying the chain |
| 10, 11 | `column`, `salt` | cut | their lengths: plan | no: the prover's word |
| 10, 11 | `path` | cut, hash | the given digests are cut (R7.9). The others are hashed from the other leaves of the same opening and the given digests below them. A leaf's digest (R7.4) takes the tree's number (R7.13; it is not the field `tree`), 2^depth + the position, the column's and the salt's lengths, and the column, fill and salt; it is hashed under the sending chain's run's value, or under the configuration's digest for a weight table. A vertex's value is hashed under the configuration's digest (R7.7). Nothing in a path says which entry is cut and which is hashed | it need not be: a verifier has to hash the leaf and follow the path to the root. One that walks a tree once, over all its opened leaves together, reads only some entries of the paths: the others can then be anything |
| 12 | bindRoot | hash | the slot's leaf is hashed on the first (one-time) chain's seal and `slotSalt`, with 2^depth + `place`, under the configuration's digest, and is then taken up `slotPath`, left or right by `place` (R7.18, R14.7). The seal is the first block of the verifier's message in that chain's last round (R6.8), so it hangs on all that the chain has hashed | yes, by replaying that chain |
| 13, 15, 16 | place, slotSalt, slotPath | cut | their lengths: plan | no: the prover's word |
| 14 | places | plan | 2^depth of the binding tree | yes |

**`leaf-<k>.parsed.bin`**

| # | field | kind | from what | can it be made again? |
|---|---|---|---|---|
| 0 | magic | | a constant of the form | |
| 1, 2 | root0, answers0 | cut | the key file | no: the prover's word |
| 3, 4 | root, answers | cut | the leaf file. Which leaf file is parsed: the number in the proof file's head | no: the prover's word |
| 5 | run | hash | as run1 | **not from the parsed file alone**, as run1 |
| 6 | rounds | cut, plan | as rounds1 | the plan's share: yes |
| 7, 8, 9 | trees, stored0, stored1 | | as fields 10 and 11 above. Of the positions sampled after `LINK` the first ones are the anchor's and the rest the leaf table's; how many: plan (R7.19) | as above |

**The other files**

| file | kind | from what | can it be made again? |
|---|---|---|---|
| a seeds file | hash | sampled from a chain of its own (R4.8), which starts at the table's name: the root, the configuration's digest and the six numbers of the table's shape (R4.10). So the seeds hang on the prover's root | yes, from the root and the public files |
| the order file | | `kind`, `tableHere`, `label`, `hasRandom`: constants of the form and the format. `nonceBytes`, `names`, `hasNonce`: plan. `random`: cut. An opening's `chain` and `round`: count. That there is such an opening, its `stored` and `tree`: hash, by the trial above. `idx`: hash, as `leaf` | |

**What a chain hashes.** A chain starts from one hash call (R4.6) on its label, its parameters' digest,
its statement and its run's value. Every round's query (R5.2) then takes the label, the value before
it, the round's number, kind and bits of work, the prover's message and the nonce. The statements:

| chain | its statement holds |
|---|---|
| a key file's one-time chain | the table's root and answers |
| a proof file's first chain (R14.3) | the table's root and answers; `copy`; the item `request`, 177 words made from the parameter file (`policy.txt`), the scale file (`KL.bin`) and the request file; and the request file's words, whole |
| a shared chain | the table's root and answers; `bindRoot` |
| a second chain (R14.10) | the seals of the two chains before it, and nothing else |
| a leaf file's link chain | both tables' roots and answers |

The item `request` holds five digests of a second hash mode of our own, built on BLAKE2b's
compression function (R15.3): three over lines of the parameter file, one over the scale file, one
over the request file. The parser computes these too.

The parser also computes the configuration's digest, each chain's parameters' digest and first value,
each round's answer, all that is sampled from it, each opened leaf's digest, each root again, and the
seals. Of these a parsed file holds a leaf's digest, where it is an entry of a neighbouring leaf's
path, and, as the tables say, the sampled positions as `leaf`; no other. The inner vertices in a path
are in the table above. With `--lines` many of them are in [the lines](#9-the-lines), and so are the
chains' random bytes.

All of this is the prover's word: it comes from the untrusted parser. A verifier that takes a computed
value from a parsed file has to make it again or check it itself, or has to be sound whatever it is.
`run1`, `run3` and a leaf file's `run` can be made again only with the order file's `random`. A
verifier that does not read it takes them as part of the prover's message. (A verifier that reads the
original file hashes the 32 random bytes itself: R4.5.) Which chains have random bytes of their own is
fixed by the protocol (R4.5): a verifier must not take that from the flag `hasRandom`, which comes from
the untrusted parser too. Where the flag is 0, `random` is 32 zero bytes.

The same holds for what a parsed file lacks: a verifier samples the positions itself, and has to
refuse if one of them has no row. `cp_verify` also refuses an opening that no round asks for; a
verifier that does not refuse a row at a leaf that no round samples parts from it there. That alone
does not make the two agree on every file ([section 7](#7-what-the-parser-checks)).

**The readers written in Lean.** The readers of the three parsed files, and the checks that take
their output, are in `lean/`: for a leaf's parsed file at the small and the full size
(`lean/Ask/EntryParsed.lean`); for a key's and a proof's at the small example's size alone
(`lean/Ask/KeyCompiledSmall.lean`, `lean/Ask/ProofCompiledSmall.lean`, and all three together
`lean/Ask/AllCompiledSmall.lean`). They are tests: nothing here is a statement of soundness.
`cp_verify` does not use them: it reads the original files. `lean/Ask/README.md` says how to run the
key's, the proof's and the joined check, and what a yes from them does not mean.

## 7. What the parser checks

Only what it needs to cut the files. It refuses a file

- that is no regular file, or has more than 2^28 bytes;
- whose first eight bytes are not its kind's (R13.1, R14.2), that ends early, or that has a byte left
  over;
- in which a round's proof-of-work test fails (R5.5);
- that is a proof file, if the request file has not the length that the public files ask for;
- in which a weight table is not opened where the plan asks for it: after the shared chain's round
  `LOC` the file's table, after the round `LINK` both tables;
- in which, after a round that samples positions, no tree can be opened at them. Bytes that do not
  lead to a tree's root, or in which a column's word is not below p, are not taken as an opening:
  they are then read as what comes next, and the file fails there;
- in which a word of a message's element or of a stored answer is not below p;
- whose slot is not below 2^depth, or is not 0 in a key file;
- in which one leaf is opened with two columns or two salts.

It makes no other check of the protocol. Above all:

- One opening after a round that samples positions is enough for it; where a weight table has to be
  opened, that opening alone is enough. A file that leaves out further openings of such a round, or
  opens another tree than the plan asks for, is parsed all the same. The parsed file then lacks rows,
  or has rows at leaves that no round asks for. If the leaves of an opening that was left out are all
  opened elsewhere in the file, it is the same as the whole file's. No verifier behind the parser can
  then tell the two files apart, while, by a reading of the code, `cp_verify` refuses the shorter one.
- It does not refuse a run's value of 32 zero bytes (R4.5).
- It compares nothing with the commitment ID or with the user's next token.

So a file that has been damaged, or made up by someone who replays the hash chains, can be parsed.
That it is parsed says nothing about whether it proves anything.

## 8. What a parsed file never holds

The user's copy of the [commitment ID](../../docs/glossary.md#commitment-id), of the request, of the
[next token](../../docs/glossary.md#next-token) or of the three public files; the plan; the
configuration's digest.

- The proof file itself states the next token: the prover sends it in the first chain's round
  `REPLY`, as the element (y, 0, 0) (R15.7). It is in `rounds1` of `proof.parsed.bin` like every other
  message. A verifier has to compare it with the next token that the user gives (R14.8), and must
  never take the next token from there.
- The parser never sees the commitment ID. To form the ID from the public files and the anchor's root,
  seeds and answers (R4.11), and to compare it with the user's, is the verifier's work.
- Of the request the parser asks its length (2^lt + 2 words of 4 bytes) and that the proof's first
  chain fits it. All of the request's own rules
  ([FORMAT.md, 12.2](../../docs/format/FORMAT.md#122-the-callers-two-files)) remain the verifier's to
  ask, on the user's files. The parser never sees the token file.
- The parser is given the public files and the request, because it cannot cut the files without them.
  It passes on none of their bytes. What it derives from them is listed in
  [section 6](#6-what-is-computed-and-not-copied); the order file also holds the plan's names of the
  rounds. The lines hold far more ([section 9](#9-the-lines)): among it the SHA-256 of every file that
  was read, the configuration's digest, each chain's parameters' digest, first value and chain values,
  what each round samples, the seals and the chains' random bytes.
- The first of a proof file's lines also holds the SHA-256 of the request file. It has no salt: it
  shows whether two proofs are of one request, and a guess at a request can be tested against it.
  Whoever passes the lines on passes that on.

## 9. The lines

With `--lines` the parser also writes each file as JSON Lines, in the file's order.

- The first line is the head: `form` (`lines-0`), `file` (`key`, `leaf` or `proof`), `name`, `bytes`,
  `sha256`, and `beside`: the SHA-256 of every other file that was read to cut this one.
- A line with `len` above 0 is one item of the file. The items tile the file: the first has `off` 0,
  each next one begins where the one before ends, and the last ends at the file's end. So the joined
  `hex` of these lines is the file.
- A line with `len` 0 and `kind` `derived` is a computed value, at the place where it arises.

| field of an item | meaning |
|---|---|
| `off`, `len`, `hex` | where the item is, and its bytes |
| `chain` | `head`, `table` (a root and stored answers after the head), `slot`, or the chain: `one-time`, `first`, `shared`, `second`, `link` |
| `round`, `rname` | the round's number in its chain, from 1, and its name; 0 and empty outside a round |
| `part` | `message`, `nonce`, `opening`, or empty outside a round |
| `kind` | `magic`, `version`, `u32`, `word`, `random`, `root`, `answer`, `E`, `nonce`, `column`, `salt`, `digest` |
| `i` | the item's index among those of its kind in that part; in an opening, within that opening |
| `v` | the value in decimal: the words of an element or a column, or the integer |
| `cls`, `sub`, `where`, `msg` | the names under which `tools/audit/cut.py` and the plan know the piece |
| `row`, `answer` | of a stored answer: its row, and its index within the row |
| `open`, `tree`, `pos` | in an opening: its index in the round; its tree, as the `uid` of the `list` line before it (not the index of section 2); the leaf |
| `level`, `node` | of a given digest: the depth and the place of its vertex |
| `pieces` | of a column: which oracle message has how many of its words |

A computed value is under `value` (hex) or `v`. Some lines have more: `label` (a chain's), `rkind` and
`work` (a round's kind and its bits of proof of work), `shape` (a weight table's), `depth` and `place`
(the slot's).

| `name` of a computed value | meaning |
|---|---|
| `configuration's digest` | R4.9 |
| `table's name`, `record seeds` | R4.10, R4.8; `which` says of which table |
| `parameters digest`, `run's value`, `first value` | of a chain (R4.4, R4.5, R4.6) |
| `chain's value` | after each round's query (R5.1); every round has this line, also one without a byte in the file |
| `elements`, `draws`, `trits`, `seal` | what the round samples (the rules R6.1 and following) |
| `list` | before each opening: the leaves at which the tree is opened, and under `tree` the tree's `depth`, `width`, `salt_bytes`, `sender`, whether it is a weight table's (`stored`), its two numbers of R7.13 (`number`, `place`), its index of section 2 (`order`), a running number over all files of the call (`uid`) and its oracle messages (`parts`) |
| `binding root` | R7.18 |
