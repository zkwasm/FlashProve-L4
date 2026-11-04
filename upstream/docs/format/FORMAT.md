# The file format, byte for byte

This page defines the files that the programs in this repository write and read, and what a verifier
does with them: which bytes are in which file, what is hashed and in which order, how each challenge
is read out of a hash, and which tests a verifier makes. It is written for someone who wants to
write a second verifier, or a formal definition of one. What it does not define is listed in 17.1.
"This page" means this file together with the files that hold its sections 8 to 11;
[README.md](README.md) lists all files of this directory. It gives no reasons why the protocol
convinces: these are on the [protocol page](../protocol.md), which describes a run step by step,
groups the verifier's checks, and points here for every byte.

*Version.* The
format's version word is `H821` in field 2 and `H811` in field 1 (R4.3). Where this page is known to differ
from the programs is listed in 17.2.

*This page decides.* On bytes, hashed strings, order and tests this page takes precedence over every
other page of the repository. Where it and a program differ, that is a fault: please report it.

*Rule numbers.* Every rule has a number, R0.1 to R15.7 (there are no rules R9.x, R10.x, R11.x), so
that it can be cited; some comments in the sources and the scripts cite them. The numbers do not
follow the sections of this page:

| rules | section | file |
|---|---|---|
| R0.x | 13.1 | this file |
| R1.x | 2 | this file |
| R2.x, R3.x | 3 | this file |
| R4.x, R5.x | 4 | this file |
| R6.x | 5 | this file |
| R7.x | 6 | this file |
| R8.x; R13.1 to R13.3; R14.1, R14.2 | 7 | this file |
| R12.x; R13.4 to R13.19 | 9 | [09-key-and-leaf-files.md](09-key-and-leaf-files.md) |
| R14.3 to R14.11 | 10 | [10-proof-file.md](10-proof-file.md) |
| R15.x | 12 | this file |

Sections 8, 11, 13 (after 13.1) and 14 and the subsections 12.2 and 12.5 have no numbered rules:
they are written from the programs' source. Sections 1 and 15 to 18 and the subsections 12.1 and
12.6 state no numbered rules either. From the programs' source is also every sentence elsewhere that
ends "(from the programs' source)". Text without a number binds a second verifier like a numbered
rule wherever it says what a verifier hashes, derives or rejects: the numbered rules alone are not
the whole format. Examples are the five tests of 12.2, the eight conditions of 12.5, the passages
from the programs' source in section 9, and the two hashed facts of 17.1, item 2.

*How to read a rule.* "A verifier rejects .." is a duty of every verifier. MUST and MUST NOT mark a
duty of a committer, a prover or an operator whose breach no verifier can see. Names in code marks
are literal: item names, round names and the version word are hashed, so they are bytes of the
format. One word needs care. In prose this page says **segment** for one piece of a request's proof,
and **link** only for the tie between the anchor and a leaf table (section 1). In the parameter
file's lines `group link` and `stages link`, in the item name `links' identifiers`, in the field
`link` of the layout file's lines (13.4, step 7) and in the names `plan-link`, `link-claim-count`
and `internal-g-link` (13.8), "link" names a request's segment. So it does in the example protocol
"links" and its item `the chain of links` (R12.15).

## 1. The objects

This section lists the objects of the format: what each is called, in which file it is, and which
rule defines it. What the main objects are for and who makes them is on the [protocol
page](../protocol.md). Some hashed or printed names use other words than the prose; the last column
gives them.

| object | where it is | defined in | names in the bytes |
|---|---|---|---|
| **public files**: the parameter file, the layout file, the scale file. Together they are a **configuration** | `policy.txt`, `layout.txt`, `KL.bin` in the configuration directory | section 8; 13.3; 13.4 | |
| **configuration's digest** | computed from the public files; in the first query of every chain: as a parameter item, and in the chain of R4.8 inside the table's name (R4.10) | R4.9 | the item `configuration` |
| **stored table**: the weights, encoded row by row; its columns are the leaves of a Merkle tree. The first one of a commitment is the **anchor**; a further one, with the same rows under fresh randomness, is a **leaf table**, number k from 0 | with the committer; a verifier sees its root and opened columns | R7.4; R4.10; 11.5; R4.6; R13.5; R14.3 | the items `table`, `the first table` (the anchor), `the further table` (a leaf table) and `stored copy` (a leaf table's number); the parameter line `table` |
| **key file**: the anchor's root, and the proof that the table is well formed | `key.bin` in the public key directory; begins `CPIK` | R13.1, R13.2; 7.6; 9.7; 13.3 | |
| **leaf file**: a leaf table's root, and its link to the anchor | `leaf/<k>.bin` in the public key directory; begins `CPEN` | R13.1, R13.3; R14.2; 7.7; 9.8; 13.8 | "entry" in rejection names |
| **link**: the tie between the anchor and a leaf table | the chain of a leaf file | R13.9, R13.10, R13.11; R7.19; 9.8 | the rounds `LINK`, `LINK point`, `LINK phi`; the item `the link` |
| **commitment ID** | computed from the configuration's digest, the anchor's name and the seeds and the stored answers of its record; given to the verifier by its caller | R4.11; 13.5 | `published_id` in `format_rules.py` |
| **request** and **next token** | the request file and the token file of the caller | 12.2; 12.3; 12.4; R14.3 | the items `request` and `the request file's words`; the round `REPLY` |
| **acceptance policy** | seven lines of the parameter file | 12.5 | |
| **proof file**: three chains and the request's slot in its batch | one file; begins `CPPF` | R14.1, R14.2; 7.8; section 10 | |
| **batch**: the requests that share one opening of a leaf table. A request's **slot** is the place of its leaf in the batch's binding tree, which the prover samples (section 16) | not a file: its shared chain is in each of its proofs | R14.6, R14.7, R14.9; 6.5; 8.2 | the items `binding tree`, `the binding tree's root`; the line `slots` |
| **chain**, **first query**, **item**, **plan**, **round**, **seal** | a chain's messages are in a file; its hash values are in none | section 4; R6.8 | |
| **segment**: one step of a request's walk through the model (a group of neighbouring layers without their attention function; that group's attention function; the output head), with its **sub-proof** | rounds of a proof's first chain | R14.4; section 11; 8.2; R14.3 | `group link`, `stages link`, `links' identifiers` |

*Who is who.* The **caller** is the one who runs a verifier: as a rule the **client**, who sent the
request and was given the next token. The **provider**
is the one from whom the caller has the public key directory and the proof file (14.1): the
committer's and the prover's side. The caller of the frame (R14.3) is another thing: a protocol.

## 2. Bytes and elements

This section says how integers, field elements, digests, texts and lists are encoded as bytes. The
same rules hold for hashed strings and for files. Every object has exactly one encoding, so hashing
the encoding of a parsed object and accepting only canonical encodings come to the same thing
(R0.1). Some integers are not encoded as one word of 8 bytes, each by a rule of its own: in a file's
head, after the four bytes that say the file's kind, the version word and a proof file's number have
four bytes each (R13.1, R14.2); in a file, a nonce and a listed position, whose numbers of bytes are
the plan's (R8.2); and the words of 32 bits of the item `request` (R15.1), of the request file, of
the token file and of `KL.bin` (13.3). In the hash's label four numbers share one word (R2.1). The
public files are texts or tables of their own form and are not encoded by this section. Their bytes
are hashed as they are in the file (R4.9), and the layout file has more than one admitted encoding
(13.4, step 7; 17.1, item 9).

- **R1.1** A **byte** has 8 bits. A **word** is an integer in [0, 2^64). It is encoded as 8 bytes,
  the lowest byte first. Nothing in the format is encoded highest byte first.
- **R1.2** The base field F is Z / p. Field 1: p = 2^61 - 1. Field 2: p = 2^64 - 2^32 + 1. A program
  is built for one field; the programs of this repository can be built for field 2 alone (from the
  programs' source). Field 2 is the reference.
- **R1.3** An element of F is encoded as the word that is its least non-negative residue. A reader
  rejects 8 bytes whose word is not below p, so each element has one encoding. This is not the map
  of R6.3, which makes an element out of a sampled word: in field 1 the word 2^61 is rejected as an
  encoding, and gives 0 as an attempt.
- **R1.4** The extension field is E = F[X] / (X^3 - c), with c = 5 in field 1 and c = 2 in field 2.
  In both fields 3 divides p - 1 and c^((p-1)/3) is not 1, so c is not a cube and the polynomial is
  irreducible. The element e_0 + e_1 X + e_2 X^2 is encoded as e_0, e_1, e_2 in this order: 24
  bytes. A reader rejects it if it rejects one of the three.
- **R1.5** A **digest** is any 32 bytes. Where a digest is read as four words, word i is its bytes 8
  i to 8 i + 7, by R1.1. A digest is encoded as its 32 bytes, with no length before them. Two rules
  put a length before a digest, and each says so: as the whole value of an item a digest has the
  item's length, the word 32, before it, like every value (R4.1); and a stored table's root is
  encoded inside its item as a string of bytes, with the word 32 before it (R4.6).
- **R1.6** A **text**, and any other **string of bytes**, is encoded as its length as a word and
  then the bytes as they are. The names in this format are ASCII.
- **R1.7** A **list** of values of one kind is encoded as its length as a word and then the values,
  each by its own rule. A string of bytes is not a list of integers: it follows R1.6. A **pair** is
  encoded as its first part and then its second.
- **R1.8** An integer, a truth value (0 or 1) or a member of an enumeration is encoded as one word.
- **R1.9** R1.1 to R1.8 hold for hashed strings and for files alike. Anyone who reads an encoding
  back rejects: an element by R1.3 or R1.4; a truth value that is not 0 or 1; a member of an
  enumeration that is not one of its numbers; a length that is greater than the number of bytes
  left; a name with a byte outside 0x20 to 0x7E. For a list the length counts values, each of which
  has at least one byte, so the same test on the length holds; a list also fails where the bytes run
  out. The encoding of each kind in this section shows where it ends, so the encoding of a pair is
  one to one. The types of R4.2 have the same duty.

## 3. The hash and its uses

The format has one hash function, H, which is BLAKE2b with a 16-byte label in front of the hashed
bytes, and, inside one item, a second, fixed hash function, built on BLAKE2b's compression function
(R3.5). Every use of H has a number of its
own, and the three uses under 169 differ in the index, so no two uses ever hash the same string.
This section defines H (3.1) and lists its uses (3.2). It needs section 2. That BLAKE2b behaves like
a random function is an assumption. This page makes no claim about it.

### 3.1 The hash

- **R2.1** Every hash of the format is a value H(number, tree, index, start; bytes). It is BLAKE2b
  exactly as it is specified, in its sequential mode and without a key, of the string

      the word number + 2^16 tree + 2^32 W + 2^56 s8,   the word index,   bytes

  with *start* as its salt and personalization (R2.2). In this and all later formulas the four
  arguments come before the first semicolon of H(..; ..); the commas and semicolons after it mean
  that the encodings follow one another. The first 16 bytes of that string are the **hash's label**.
  W and s8 are 0 except under the three numbers of leaves, 164, 173 and 174: there W is the number
  of words of the leaf's content and s8 the number of words of its salt (R7.3). BLAKE2b: Aumasson,
  Neves, Wilcox-O'Hearn, Winnerlein, "BLAKE2: simpler, smaller, fast as MD5", with the parameter
  block in its section 2.8; RFC 7693 has the compression function, and prints only the block's first
  four bytes.
- **R2.2** The parameter block P has 64 bytes. Byte 0, the digest length, is 32, and under the
  number 162 it is 64. Byte 1, the key length, is 0. Bytes 2 and 3, fanout and depth, are 1 and 1.
  Bytes 4 to 31 are 0. Bytes 32 to 47, BLAKE2b's field "salt", are bytes 0 to 15 of start. Bytes 48
  to 63, its field "personalization", are bytes 16 to 31 of start. The last-node flag is never set.
  In Python:

      hashlib.blake2b(struct.pack("<QQ", number + (tree << 16) + (W << 32) + (s8 << 56), index) + bytes, digest_size=32 or 64, salt=start[:16], person=start[16:]).digest()

  BLAKE2b's field "salt" has nothing to do with a leaf's salt, which is part of *bytes* (R7.3).
- **R2.3** *number* is one of the thirteen to which R3.1 gives a use; 177 is not one of them (R2.5).
  *tree* is an integer below 2^16, under the number 162 below 2^48, and is 0 where R3.1 leaves its
  cell empty. *index* is an integer below 2^63. Under a leaf's number W is at least 1 and below
  2^24, and s8 is 5 to 8. *bytes* is a string of at most 2^32 - 144 bytes. *start* is a digest under
  the numbers to which R3.1 gives one, and 32 zero bytes under the others, written "none" in the
  formulas. The number says which; no number is used both ways.
- **R2.4** *The oracle's strings.* A proof about the format treats the hash as one random function O
  from byte strings to strings of 32 bytes. The string of a hash is one byte for the half, then P,
  then the label, then bytes. A hash with a digest length of 32 is O on its string with the half
  0x00. A hash with a digest length of 64 counts as two queries: its first 32 bytes are O on its
  string with the half 0x00, and its last 32 bytes are O on its string with the half 0x01. So the
  map from (half, number, tree, W, s8, index, start, bytes) to strings is one to one. Its image is
  exactly the strings of at least 81 and at most 2^32 - 63 bytes such that: the number in the label
  is one of the thirteen to which R3.1 gives a use, 177 not among them; the digest length is 64 if
  that number is 162, and 32 otherwise; the half is 0x00, or 0x01 under the number 162; bytes 1 to
  31 of the block are as in R2.2; start is 32 zero bytes under a number to which R3.1 gives none;
  tree, W, s8 and index are within R2.3's bounds; W and s8 are 0 under a number that is not a
  leaf's; under a leaf's number the bytes after the label are 8 W bytes, the fill of R7.3, all
  zeros, and 8 s8 bytes; tree is 0 where R3.1 leaves its cell empty or names 0 (173, 174); and index
  is one of the values that R3.1 names where it names single values (166, 176, 178 and 179: 0; 169:
  0, 1, 2). What the bytes after the label must be, use by use, is the rule named in R3.1's last
  column.
- **R2.5** *Reserved.* No hash of the format is O on a string whose first byte is not 0 or 1, and no
  hash is made under the number 177. Both are left to a formal definition that needs strings outside
  the format's image.
- **R2.6** BLAKE2b makes ceil(n / 128) calls of its compression function on a string of n > 0 bytes.
  So a vertex (80 bytes: R7.7) costs one compression, and a call of a round's query (R5.2) costs one
  compression if the prover's message has at most 96 bytes. A program that keeps the state after the
  message's whole blocks pays one compression for each further call j of a round, and two if the
  word with j ends a block, that is, if the prover's message has 104 bytes modulo 128.
- **R2.7** A verifier rejects parameters, a statement, a round's message or a leaf's bytes that
  would make a hashed string longer than R2.3 admits. It also rejects a chain of 2^63 rounds and a
  tree in which 2^a + position reaches 2^63. Three more inputs of a verifier or a prover come under
  R2.3's bound, and their lengths stand in no plan. The input of the configuration's digest (R4.9):
  the programs' only test on the length of a parameter file or a layout file as such asks for fewer
  than 2^32 bytes (13.3), which is more than R2.3 admits; a verifier rejects public files that
  together are too long. The input of the commitment ID (R4.11): a verifier rejects a record whose
  seeds and answers would make it too long. The bytes under the number 176 (`COIN-KEYS.md`, section
  2), which only a committer or a prover hashes: whoever forms them stops (in the programs a guard
  of the library ends the process: `blocks/coins/run_coins.h`). None of these is a plan, statement,
  file or shape of the format. (How the programs end in these cases: 17.2, item 1.)

### 3.2 The uses

- **R3.1** The numbers, and what each argument of H is under each. An empty cell means 0 for *tree*
  and 32 zero bytes for *start*. Four uses lead to no rule of this page: 168 (where each program
  that writes such a digest has an index of its own), 169 with the index 0, 171, and 176 (the
  committer's and the prover's alone; `COIN-KEYS.md`, section 2, has its bytes). No test of a
  verifier depends on a hash under them: the programs' verifier makes none under 168, 171 and 176,
  and the digest that it makes of the parameter file's text under 169 with the index 0 enters no
  test and no other hash (from the programs' source).

| number | use | tree | index | start | bytes |
|---|---|---|---|---|---|
| 160 | a chain's first value | | the chain's label | | R4.6 |
| 162 | a call of a round's query; 64 bytes | the chain's label | the round, 1, 2, .. | the chain's value before the round | R5.2 |
| 164 | a leaf of a tree that is made in a run | the tree's number | 2^a + position | the run's value | R7.3 |
| 166 | a vertex from its two children | | 0 | the configuration's digest | left digest, right digest |
| 168 | the digest of a statement | | which writer | | its bytes |
| 169 | the digest of parameters | | 0: a parameter file's text; 1: the bytes of R4.3; 2: the configuration (R4.9) | | their bytes |
| 171 | synthetic randomness of tests and toy runs; not a prover's generator | | a counter | the test's seed | none |
| 172 | a run's value | | the chain's label | | R4.5 |
| 173 | a leaf of a batch's **binding tree**, whose leaves stand for the requests of the batch | 0 | 2^depth + place | the configuration's digest | R7.18 |
| 174 | a leaf of a tree that is made once, stored and read by many runs | 0 (R7.4) | 2^a + position | the configuration's digest | R7.3 |
| 176 | the digest of named inputs; the routine hashes whatever its caller adds. For a proof they must cover all that the prover's messages depend on; for a commitment's table they are public facts and nothing of the weights (by a comment of `blocks/hash/domains.h`). It enters a key of the committer's or the prover's private generator, and no verifier computes it | | 0 | | not defined on this page: no verifier computes it |
| 177 | reserved (R2.5) | | | | |
| 178 | a vertex of a batch's binding tree | | 0 | the configuration's digest | left digest, right digest |
| 179 | the commitment ID | | 0 | | R4.11 |

- **R3.2** A number gets no second meaning. 161, 163, 165, 167, 170 and 175 are numbers not in use, and none of them is ever given a use. Numbers below 160, and 240 and 241, are not used as numbers of H. (The second hash function of R15.3 has numbers of its own, 10, 11, 72
  and 73: 12.6.)
- **R3.3** *Where the public parameters are bound.* The chain's parameters' digest (R4.4) is in a
  first value as its first bytes, in a round's calls through the chain's value, and in the leaves of
  a run's tree through the run's value. A stored tree and the binding tree are made before the
  chains that read them, and are read by chains with different plans (a leaf of the binding tree
  holds the seal of a request's first chain, R7.18, so that tree is made after the batch's first
  chains and before every chain whose statement has its root); their leaves start from the
  configuration's digest (R4.9), which the parameters of every chain that has parameters hold as an
  item. Every vertex of every tree starts from the configuration's digest too: it is one constant
  for all that is made under one configuration. Of the hashes on which a test of a verifier depends,
  one kind holds no digest of the parameters by itself, the chain of R4.8; it holds one through the
  table's name (R4.10). The hashes under 168 and 171 hold none either (R3.1).
- **R3.4** A leaf's string is never a vertex's: the numbers differ.
- **R3.5** A verifier of this format makes no hash but H. In particular the tree of a stored table,
  which holds the weights, is a stored tree (R7.4), under the numbers 174 and 166. The one exception
  is R15.3: the item that it defines holds nine digests of the second hash function. Everyone who
  forms the item computes five of them. Four are lines of the parameter file; the programs compute
  these four too, and reject if a line differs (R15.6; 17.2, item 7).
- **R3.6** (This number is not in use on this page.)

## 4. A chain: first query, rounds, answers

A **chain** turns one interactive protocol into a non-interactive proof, by the Fiat-Shamir
transform. It has a **label**, the chain's label, an integer that says which protocol it is and,
where two kinds of file run one protocol, in which kind of file: each of the seven chains of a key
file, a leaf file and a proof file has a label of its own (R13.4, R14.3, R14.9, R14.10). These
labels are below 2^32; the one label from 2^32 on is that of R4.8. Everything that is fixed before
the prover speaks enters the chain's first value (4.1). The hash that gives this value is the
chain's **first query**; whatever is among the chain's parameters or in its statement (R4.1) is said
to be in the first query. Then each message of the prover is hashed together with the chain's
current value, and the hash gives the next value and the verifier's message (4.2). How a verifier's
message is read out of the hash is section 5. This section needs sections 2 and 3.

The rules below use some terms of later sections. A **stored table** is a table that a committer
commits to once and that many runs read (R7.4). In an **out-of-domain sample**, seeds are sampled,
each a point and a factor (R6.9), and the committer or the prover gives its answers to them,
elements of E, on what it has committed to. R13.8 says what an answer is for a stored table, and
R8.3 and 13.9 say it for an oracle message (section 6). A stored table's **record** is the list of
the stored answers to the table's own out-of-domain sample (R4.8), the same number for each row.
The first stored table
of a commitment is the **anchor**; a further one, with the same rows under fresh randomness, is a
**leaf table**. Key files, leaf files and the link between the anchor and a leaf table are defined
in sections 7 and 9; batches and requests in sections 7 and 10; a session in R7.14.

### 4.1 The first query

- **R4.1** The public inputs of a chain come in two lists of items, the **parameters** and the
  **statement**. The parameters are what is fixed before a request is known. An **item** is a name
  and a value. It is encoded as the name (a text), the length of the value's bytes as a word, and
  the value's bytes. (The items of a chain in a file, R8.1, are another thing: they have no names
  and no lengths.)
- **R4.2** The value's bytes are the **canonical bytes** of its type: R1.3 to R1.8 for the types
  named there. For a type that is made of several fields they are the fields that change what a
  verifier does, in a fixed order, each by its own rule. Sections 9, 10 and 12 list these types. Two
  hashed facts of these types are fixed by no numbered rule and are given from the programs' source
  alone: the values of some leaf classes, and the order in which codes enter `codes` (17.1, item 2).
- **R4.3** The parameters' bytes are, in this order:
  1. the item named `the format's version`. Its value is one word, whose four low bytes are, from
     the lowest, the ASCII characters `H`, `8`, the field's digit and the hash's digit `1`, and
     whose other bytes are 0. So the version word is `H821` for field 2 (as a number, 0x31323848)
     and `H811` for field 1 (the programs of this repository serve `H821` alone: 17.2, item 11);
  2. the chain's **plan**: eight words, then its entries. The plan is not an item: no name and no
     length come before it. The eight words are the field (1 or 2); the hash (1); the number of
     bytes of a leaf's salt; the number of bytes of a nonce in a file; the number of bytes of a
     listed position (0: none is listed); the number of answers to each out-of-domain sample;
     whether a sumcheck's message leaves out the coefficient that the check fixes, which is 1 in
     every plan of this format; and the number of entries;
  3. the item `configuration` (R4.9), and then the other parameter items, in the order in which the
     protocol names them.

  An **entry** of the plan is a text, the round's name, and eleven words, each a number: how many
  rounds in a row are like this one; the proof-of-work bits; how many bytes of the prover's message
  enter the query; how many roots are among them; how many answers to out-of-domain samples are
  among them; how many elements of E are sampled; how many of these must not be 0; how many
  positions are sampled; k, where positions are sampled from [0, 2^k); how many matrix entries are
  sampled; and, for a round that batches rows (a row, in this sense, is a linear equation on what
  oracle messages commit to: section 10), how many rows. The first of the eleven words is 1 in every
  plan of a chain. The kind of a round (R5.2) is not among the words: it follows from the round's
  name, so the list of the names with their kinds is part of the format (R14.11). A plan can hold
  one entry that is not a round of its chain, under the name `RECORD`: a stored table's own
  out-of-domain sample (R4.8), in a committer's chains. It is encoded like
  any entry, and the chain's rounds pass over it. Which plans hold it, where, and with which words
  is R13.15.
- **R4.4** The **parameters' digest** is H(169, 0, 1, none; the parameters' bytes).
- **R4.5** A chain can have 32 **random bytes**. There are three cases, and which one holds is fixed
  by the protocol. (a) The prover samples them in private, and they are the first 32 bytes of the
  chain in its file. (b) Both sides take 32 bytes that are in this chain's first query as an item.
  (c) There are none; such a chain sends no root. In cases (a) and (b) the **run's value** is H(172,
  0, label, none; the parameters' digest, the random bytes). In spite of its name it is a value of
  one chain. A run, as section 16 defines it, is one commit run for a key file or for a leaf file,
  or one batch with the proofs of its requests (a **commit run** is the part of a call of the
  committer that builds one stored table, one key file or one leaf file; one call can hold several);
  it has several chains, and each chain of case (a) or (b) has a run's value of its own. A verifier
  rejects a proof in which a chain's run's value is 32 zero bytes. No two chains that send roots
  have the same label and the same random bytes: in case (a) they are sampled afresh for each chain
  (a prover's duty, which no verifier sees: section 16), and a protocol gives case (b) only to a
  chain whose 32 bytes are made of fresh randomness of that run. In a batch, the two chains of a
  request have case (a). The shared chain has case (b): its 32 bytes are the item
  `the binding tree's root` of its statement.
- **R4.6** The chain's **first value** is

      rho_0 = H(160, 0, label, none;  the parameters' digest,
                                      the statement's length as a word, the statement's bytes,
                                      the record field's length as a word, the record field's bytes,
                                      the word 1 and the run's value, or in case (c) the word 0)

  No chain of this format fills the record field: its length is always the word 0, and no byte of
  the field follows.

  The root and the record of every stored table of which a chain speaks, whether or not it opens a
  column of that table, are among the items of its statement, by value, before its other items, one
  item for each table (a second chain has them only through the two seals of its statement: R13.5,
  R14.10). The item's name belongs to the protocol; it is `table` in the chains of a key file and of
  a proof file (sections 9 and 10). Its value is the root as a string of bytes (the word 32 and the
  32 bytes), and then the record, which is the list of its answers, elements of E, by row and within
  a row by answer. The record's seeds are in no item
  and in no file (the commitment ID's input holds the anchor's: R4.11): anyone who needs them
  computes them (R4.8). This holds for the chain of a committer's one-time test on the table too.

  Why statement and not parameters: root and record are read from a key file, and a file never sets
  a public parameter (R0.2). So a chain's parameters' digest, and its run's value, depend on the
  configuration, the plan and the shapes alone. In a key file the root and the record's answers are
  items of the file that come before the chain (R8.9); they are not messages of the chain. The root
  under which a verifier checks the openings of a stored table is the root in this item, and so in
  the chain's first value. A verifier never opens under a root that it has from elsewhere. The
  commitment ID (R4.11) is in no first query: a link has both tables' items, and a session on a leaf
  table has that table's own root and answers.
- **R4.7** A chain that builds on earlier chains of the same session has in its statement one item
  named `the seal of an earlier chain` for each of them, in an order that the protocol fixes. Its
  value is that chain's seal (R6.8). So the chain of a request after a batch's shared rounds has
  two: the seal of the request's own first chain, and then the seal of the shared chain.
- **R4.8** *The seeds of a stored table's record.* They are sampled from a chain of their own. It
  has the label 2^32, 32 zero bytes instead of the parameters' digest, the table's name (R4.10) as
  its statement, an empty record field and no random bytes. It has one round, with an empty message,
  of kind 3 and without proof of work. From that round's answer (R5.4) are sampled as many seeds as
  a row has stored answers, which is the fifth of the six words of the name. Each seed is as in
  R6.9, whose v is here the number c of R4.10: the c coordinates of the point, each an element of E
  by R6.4, and then one element of E by R6.5. Why no parameters: the seeds are a function of the
  table's name alone, so that a commitment has one record however often an honest committer starts
  again. Two records of one table would reveal part of its content. So a table has the same seeds
  and one record whatever its role. What an honest committer must do so that this holds is R4.12.
- **R4.9** The **configuration's digest** is H(169, 0, 2, none; the number of public files as a
  word, then every byte of each file after its length as a word). The parameter file comes first.
  The programs of this repository take three files: the parameter file, the layout file and the
  scale file, in this order (17.1, item 9). The digest holds the format's version and all enumerated
  public parameters. It is a parameter item of every chain but that of R4.8, which has no
  parameters, under the name `configuration`, and the first of the items after the plan (R4.3). A
  toy run, that is, a small run of the library `blocks/` on synthetic data, as of the example
  protocols of R12.6 and R12.15, has a configuration too: its one public file is the text from which
  its codes, shapes and setting are made. The known answers in `format-vectors.txt` use such a
  configuration. A verifier that keeps what it has read of a stored table keeps this digest beside
  the root and the shape, and compares it with its own before it uses them (R13.10); no file of the format holds
  the digest. The item `configuration` is a value of the hash. What follows from that for a
  definition that models the hash as a random function is part of this rule, and is in section 18.
- **R4.10** A **stored table's name** is the root of its tree; the configuration's digest; and six
  words, each a number: a, where its rows have length 2^a; how many entries a row's message has,
  pads (R13.8) among them; how many entries a row's random tail has; v, where the table has 2^v
  rows, so that a column has 2^v symbols; the number of stored answers to an out-of-domain sample
  for each row (by the paragraph under R13.8, it is the plan's
  sixth word, which is the S of the parameter line `answers`, 8.4); the number of bytes of a leaf's
  salt. Together they are 112 bytes. The name is computed, by one routine, from the root, from the
  configuration's digest, and from the table's shape, which is in a parameter item and holds the six
  words. It is used in two places: as the statement of the chain of R4.8, after the statement's
  length, and in the input of the commitment ID (R4.11). No item holds it. The point of an
  out-of-domain sample has c coordinates, where c is the least integer with 2^c not below the number
  of entries of a message. The name does not say whether the table is the anchor or a leaf table.
  The seeds of R4.8 are sampled from the name, which holds the root. So what is committed is fixed
  before the seeds are known, and the numbers that decide how the sequence of words (R6.1) is cut
  are in the name.
- **R4.11** The **commitment ID** is H(179, 0, 0, none; the configuration's digest; the anchor's
  name as a string of bytes, that is, the word 112 and the 112 bytes; the list of the elements of
  that table's record's seeds (R4.8), that is, their number s (c + 1) as a word, where s is the
  fifth of the name's six words and c the number of R4.10, and then, seed by seed, the c coordinates
  of its point and then its factor (R6.9), 24 bytes each; the list of the record's answers, that is,
  their number as a word and the answers, 24 bytes each, by row and within a row by answer; and one
  word 0). That last word closes the string. It is an empty list: this page defines no element of
  it. A second implementation must not leave it out. (The seeds' list has no word c of its own: the
  name, which stands before it, holds s and fixes c.) Why the answers are in the input: a root alone
  does not name the weights. The committed word can be close to several tables, and the table is the
  close one that agrees with the stored answers (by the comments of the programs' source, which this
  page repeats and does not prove: R13.8). Why the seeds are in the input: each is sampled from a
  round's answer (R4.8, R5.4), from the name alone; whoever forms a commitment ID by this rule has
  first asked H the queries of R4.8. The configuration's digest is in the input twice, by itself and
  as bytes 32 to 63 of the name; that is on purpose, and a second implementation must not leave one
  out. (`format_rules.py` calls the commitment ID `published_id`.) What the commitment ID says: what
  is committed, under which public files. It holds nothing of the key file's proofs, which anybody
  can check, and nothing of a leaf table, which its link ties to the anchor; so it stays the same
  when a new leaf table is made. What it does not say: whose weights are under the root. That a
  commitment ID belongs to a model of some name is its publisher's word, outside the protocol. No
  item named `model` is in a statement, and no step of a verifier reads one.
- **R4.12** *The committer's rule: the same randomness is never used under two roots.* A stored
  table is made under one configuration. Its pads, its rows' random entries and its leaves' salts
  come from randomness that belongs to that table's commit run alone. Anyone who changes a public
  file, the table's shape or layout, the weights, or the table's role (anchor or leaf table) or its
  number among the leaf tables MUST make the table again, with fresh randomness, and MUST NOT place
  old rows under a new tree. Why: the leaves and vertices of a stored tree start from the
  configuration's digest (R7.4), so the same rows under another configuration have another root, and
  with it other seeds and another record (R4.8). Two records over the same pads reveal exact
  equations on the rows. The same random entries under two roots are one
  code word on which two allowances of openings are spent; under other weights they show the
  encoding of the difference at every position that is read in both. Remaking a table from the
  secret that was kept, with none of these inputs changed, gives the same table again: its root, its
  record and its counter of the columns that have been shown are unchanged. No verifier can see a
  breach of this rule: the committer's own program has to keep it (section 16; on what the rule and
  its reasons rest: 17.1, item 10). What the rule protects is given up under the programs' option
  `-public-weights`, under which the weights are not hidden (section 16, "One option of the programs
  gives these duties up on purpose").

### 4.2 A round's query and its answer

- **R5.1** The rounds of a chain are numbered 1, 2, .. . Every message of the prover is followed by
  a message of the verifier, which can be empty, and the two make a **round**. Every round makes its
  query. After a chain's last round the prover sends nothing into the chain; openings can follow
  (R8.7). Where a protocol's last prover's message is followed by no message of the verifier, a
  round of kind 7 closes it. So a chain can end with a round of any kind, no closing round is added
  where the last round samples something, and every byte that the prover sends into the chain (R8.2)
  enters some query.
- **R5.2** **Call j of round i**, j = 0, 1, .., is

      C_i(j) = H(162, label, i, rho_(i-1);  the prover's message,  the word kind + 2^8 w + 2^16 j,  the nonce as a word)      64 bytes

  The **prover's message** is all that the prover has sent in this round and that enters the chain,
  in the order sent, each item by section 2. Its length is a multiple of 8. The **kind** says what
  the verifier's message consists of: 1 elements of E; 2 the element that joins a hiding polynomial
  to a claim; 3 the seeds of out-of-domain samples; 4 positions; 5 entries of a matrix over {-1, 0,
  1}; 6 a seal; 7 nothing. w is the round's proof-of-work bits, 0 to 64, from the plan (the programs
  admit fewer: 17.2, item 5). j is below 2^48. *label* is the chain's label, where other uses have
  the tree's number: so the queries of two protocols are never the same string.
- **R5.3** In a round with w = 0 the nonce is 0 and no file holds it. A round of kind 7 has w = 0
  (R5.7). So has a round of kind 6, a seal: a plan in which a seal has w > 0 is not a plan of the
  format. R5.7's cases do not test this. The programs' planner makes no such plan: a guard would end
  the process where such a plan is made (17.2, item 1), and no public file gives a seal
  proof-of-work bits (8.4, item 3). So a verifier that derives its plans by section 8 meets none. A
  verifier that takes plans or first queries as inputs (8.4, the second way; section 18) tests it
  and rejects, or every statement on that verifier has it as a hypothesis.
- **R5.4** The **answer** of round i is a sequence of **blocks** of 32 bytes, B_0, B_1, .. : B_(2j)
  is the first half of C_i(j) and B_(2j+1) its second half. (By R2.4 each block is a query of O of
  its own.) Their meaning:

| | w = 0 | w > 0 |
|---|---|---|
| the test block | none | B_0 |
| rho_i, the chain's next value | B_0 | B_1 |
| the blocks of the verifier's message, in order | B_1, B_2, .. | B_2, B_3, .. |

- **R5.5** *The proof-of-work test.* The low w bits of the first word of the test block are 0; at w
  = 64 the whole word is 0. Nothing else of the test block is read. If the test fails, the verifier
  rejects the proof, and nothing is sampled from that answer. (An attempt of a prover's search
  computes C_i(0) alone.)
- **R5.6** Of a chain's hash values nothing is in a file: no value rho_i and no block. Of what
  enters a round's hashes and a run's value, a file has the prover's messages, the chain's random
  bytes in case (a) of R4.5, and the nonces (R8.1, R8.2). Other bytes of a file enter a first value
  (R4.6): a stored table's root and stored answers (7.6; 7.7); the number in a proof file's head, as
  the item `stored copy` (R14.2, R14.3); and a slot's word, salt and path, from which the binding
  tree's root is computed (R14.7; R13.6 for a key file). That root is also used instead of the
  shared chain's random bytes (R4.5, case (b)), so these bytes enter that chain's run's value too.
- **R5.7** A function that implements this section rejects: a prover's message that is not whole
  words; a nonce other than 0 at w = 0; proof of work in a round of kind 7; w above 64; a kind
  outside 1 to 7; a chain's value that does not have 32 bytes; a label from 2^48 on. (How the
  programs end in these cases: 17.2, item 1.)

## 5. Sampling a verifier's message

This section says how the verifier's message of a round is read out of the round's answer (R5.4).
The rules are for the verifier's messages alone. The prover's private randomness (masks, pads,
random entries, salts) is not part of the format: no verifier depends on how it is made. This
section needs sections 2 and 4.

- **R6.1** The verifier's message of a round is sampled from a **sequence of words**: the four words
  of its first block, then those of its second, and so on (R5.4, R1.5). Each sampled value takes the
  next words of the sequence. No word is used twice, and none is skipped. (R6.6 and R6.7 leave bits
  of a word unread.) What is left of the sequence after the round's last sampled value is dropped.
- **R6.2** *The order of a round's sampled values* follows from its kind and its entry in the plan.
  Kind 1: the elements of E, one after the other; either each of them must be non-zero (R6.5), or
  each may be 0 (R6.4). (The only sampled values of this format that must not be 0 are the challenge
  of a sumcheck's further variable, alone in its round (the round `PSC Y`: 9.4, 11.1), and a seed's
  factor (R6.9).) Kind 2: exactly one element, which may be 0. ("May be 0" is a rule on sampling
  alone. At some sampled values that are 0, or that make a weight 0, no verifier gives an
  acceptance: 13.9 lists them, with `coin-zero` first.) Kind 3: the seeds, one after the other
  (R6.9). Kind 4: R6.10; at least one position. Kind 5: the matrix entries. Kind 6: the seal. An
  entry that has other sampled values than those of its kind is not an entry of the format.
- **R6.3** *An attempt at an element of F.* Field 1: take the low 61 bits of the word; the attempt
  fails if they are p. Field 2: take the word; the attempt fails if it is not below p. A word is
  never reduced modulo p.
- **R6.4** *An element of F:* at most six attempts, on the next words of the sequence. It is the
  result of the first attempt that does not fail, and the words after that attempt stay in the
  sequence. If all six fail, the element is 0, and the six words are used up. *An element of E* is
  three elements of F, sampled one after the other, as e_0, e_1, e_2.
- **R6.5** *An element of E that is not 0:* at most two elements of E by R6.4. It is the first of
  them that is not 0. If both are 0, the element is 1.
- **R6.6** *A position in [0, 2^k),* k < 64, is the low k bits of the next word. The positions of a
  round are sampled one after the other, and a position can come twice. Their number is the plan's.
  Two things are passed on. The **draws** are the positions in the order sampled, repeats kept. The
  **list** is the set of the sampled positions in increasing order. An opening follows the list: a
  position that is sampled twice is opened once. Every row that stands for a read follows the draws
  (a row in the sense of R4.3: section 10): a position that is sampled twice gives its rows twice,
  each with its own position in the batching that follows. So the number of rows does not depend on
  which positions are sampled. One round's draws can belong to several trees, each with a list of
  its own: R7.19 (the round `LINK`).
- **R6.7** *n entries of a matrix over {-1, 0, 1}.* Each run of 32 entries takes one fresh word, and
  entry j of the run reads bits 2 j and 2 j + 1 of it, counted from the lowest. The entry is 0 if
  bit 2 j is 0; otherwise it is 1 if bit 2 j + 1 is 0, and -1 if it is 1. What is left of the last
  word is dropped. A round samples all its n entries together: the runs of 32 are counted from the
  round's first entry.
- **R6.8** *A seal.* A chain that other chains build on ends with a round of kind 6. Its **seal** is
  a digest: the next four words of that round's sequence, which are the first block of its message,
  because a round of kind 6 samples nothing before it. Nothing is sent or sampled on the chain
  afterwards. A seal is in no file.
- **R6.9** *The seed of an out-of-domain sample* with a point of v coordinates is v elements of E by
  R6.4 and then one element by R6.5, the factor. A round of kind 3 samples s seeds, one after the
  other, each whole; s is the plan's number of answers to an out-of-domain sample, and all have the
  same v. So its entry has s (v + 1) elements, of which s must not be 0, and that s is the sixth of
  the plan's eight words. One round of kind 3 is in a chain that has no plan: the round that samples
  the seeds of a stored table's record (R4.8; the entry `RECORD` that counts it is in another
  chain's plan: R13.15). Its number of seeds is the fifth of the six words of the table's name, and
  its v is the number c of R4.10. The oracle messages (section 6) of one prover's message that get
  an out-of-domain sample (section 9) share these seeds: v is the largest number of coordinates that
  one of them needs, and each reads the first coordinates of a point. Answer number i of each
  message belongs to seed number i.
- **R6.10** In a round of kind 4 the positions come first. If the plan's entry has an element of E
  as well, it is sampled after them, by R6.4.
- **R6.11** *A total function of a fixed number of blocks.* The sampled values of a round use at
  most K words: 18 for each element of E by R6.4, 36 for each by R6.5, 1 for each position, ceil(n /
  32) for n matrix entries, 4 for a seal. K follows from the plan's entry. So the verifier's message
  is a total function of its first ceil(K / 4) blocks. The plan fixes this longest sequence of
  blocks, and a program may compute a block only when a sampled value needs it. What the cap costs,
  with the hash taken as a random function (R2.4): in statistical distance an element of E by R6.4
  is within 3 q^6 of uniform, where q is the chance that an attempt fails, which is below 2^-32 in
  field 2 and is 2^-61 in field 1; one by R6.5 within twice that plus 1 / |E|^2. A plan's error for
  a round carries that term. The sampled values of a round are independent, although a failed
  attempt shifts the later ones along the sequence: each takes fresh words.

## 6. Trees and openings

This section says how a prover commits to oracle messages in Merkle trees: what a leaf holds and how
it is hashed, how a vertex is hashed, which bytes an opening has and how a verifier checks them,
which oracle messages share a tree, and how trees are numbered. It uses the encoding of words and
elements (R1.1 to R1.4), the hash H(number, tree, index, start; bytes) and the hash's label (R2.1 to
R2.3), the run's value (R4.5), the configuration's digest (R4.9), and the draws and the list of a
round that samples positions (R6.6). A chain, its first query, its statement, its plan and the
plan's entries are defined in R4.1 to R4.6, a round and its nonce in R5.1 and R5.2, a seal in R6.8.
A stored table, the anchor, a leaf table, a key file, a leaf file, a batch, and a request with its
slot are introduced in section 1. R8.7 says where an opening is in a file.

An **oracle message** is a message that the prover does not send in full. It consists of rows, each
of which is a code word. A file has the root of a tree over these rows and, later, their symbols at
the sampled positions. A **tree** holds the oracle messages, or parts of them, that are sent in one
prover's message and are read at the same lists of positions. What a tree holds of one oracle
message is a **part** of the tree. A **group** is a set of oracle messages on which one proximity
test (R13.19) runs; the test reads all of them at one list.

**Main rows** have length 2^a and are read at a position x. **Mask rows** have length 2^b, b <= a,
and are read at the short position of x (R7.2). All mask rows of one tree have one length. A tree of
mask rows alone is a tree whose rows have length 2^b, and for this section they are its main rows.

The **shape** of a tree is what a verifier knows of it before it reads a file: a, b, the number of
elements of F that each part has in a column, of its main rows and of its mask rows, whether it is a
stored tree (R7.4), and the length of a salt. The tree's number (R7.13) and the two values from
which its hashes start (R7.4, R7.7) are not part of the shape: they follow from the verifier's plan
and run (R7.10).

### 6.1 The leaves

- **R7.1** The **column** at position x, 0 <= x < 2^a, is a list of symbols. First come the symbols
  at x of all main rows, part by part in the order of the parts and row by row within a part. After
  them come the symbols at the short position of x of all mask rows, in the same order. A symbol
  over E counts as its three coordinates in order. So a column is a list of W elements of F, with W
  fixed by the tree's shape, and a mask row's symbol is in 2^(a-b) columns. The order within a
  column is not: the first part's main symbols, the first part's mask symbols, the second part's
  main symbols.
- **R7.2** The **short position** of x among rows of length 2^b is x mod 2^b. It is a plain rule of
  the format; all that is used of it is that each short position has equally many positions over it.
  There is one such map, and it is used alike for the copies in the leaves and wherever a verifier
  reads a mask row, or forms the row of a mask code's generator matrix, at a position. In no step of
  a protocol is a main row blinded by a mask row that is added to it, with the sum then compared
  with a single code. (In field 2's order of the evaluation points, R12.11, this map is not the map
  of the points under X -> X^(2^(a-b)).)
- **R7.3** A leaf's bytes are the column, 8 bytes for each element; then the **fill**; then the
  **salt**. A prover MUST give every leaf a salt of its own, 2^a salts for each tree: no two leaves
  share a salt, and leaves whose positions have the same short position are no exception. A salt has
  s bytes, the same number for all leaves of a tree: a multiple of 8, at least 40 and at most 64.
  The number s is the third of the plan's eight words (R4.3; in the programs one parameter line,
  `salt`, sets it for every tree, stored trees and the binding tree too: 8.2), and a plan that
  breaks these bounds is not a plan of the format, even where none of its rounds has an opening.

  The fill counts the 16 bytes of the hash's label: it is empty if ((16 + 8 W) mod 128) + s <= 128.
  Otherwise it is the zero bytes that bring 16 + 8 W + (the fill's length) to the next multiple
  of 128. So the salt is whole in the last block of 128 bytes that BLAKE2b compresses. The fill is
  hashed and is in no file. The hash's label of a leaf holds W and s / 8 (R2.1): so label, column
  and salt can be read back from the hashed string without a shape, and no two shapes give one
  string.
- **R7.4** The leaf's digest is H(number, tree, 2^a + x, start; the leaf's bytes), with W and s / 8
  in the hash's label. In a tree that is made in a run, number is 164 and start is the run's value
  of the chain that sends the tree's root (R4.5), even where a later chain reads the tree.

  In a **stored tree**, the tree of a stored table, number is 174 and start is the configuration's
  digest (R4.9). The tree of a stored table has the tree's number 0, whichever table it is: the
  number is fixed by this format, and no shape and no file has it. A second kind of stored tree
  would get a number of its own. So all stored tables under one configuration, the anchor and the
  leaf tables alike, have the same hash's labels and the same start in their leaves, position by
  position. The tables differ by their symbols, their salts and so their roots alone. A verifier
  tells them apart by the roots, which are in its first query.

  The tree of a stored table has one part, without mask rows. Its depth is the a of the table's
  shape (R4.10). Its column has one symbol of F for each row of the table, row 0 first, so W = 2^v.
  Its salts have the number of bytes that is the shape's sixth word. (These four sentences are from
  the programs' source.)
- **R7.5** Which tree holds a mask row's copies. They are in the leaves of a tree that is opened at
  exactly the lists at which the mask row is read, no more and no fewer. They are in one tree of the
  group only. That tree's root is sent before the first message of the verifier after which an entry
  of that mask row is used. A message of mask rows that cannot be in the leaves of one such tree
  keeps a tree of its own, of depth b. Two examples: a message of mask rows alone that is in two
  groups; a message of mask rows that is sent in a prover's message without main rows.

### 6.2 The vertices

- **R7.6** All trees are binary and perfect. A tree's depth is a. The vertex of depth d and place i,
  0 <= i < 2^d, has the children (d + 1, 2 i) and (d + 1, 2 i + 1). The root is (0, 0). The leaf of
  position x is the vertex (a, x), and its value is the leaf's digest.
- **R7.7** The value of an inner vertex is H(166, 0, 0, the configuration's digest; the left child's
  value, the right child's value). It holds nothing of the tree, of the vertex's place or of the
  run. The hashed string has 80 bytes: the hash's label with the number 166, and the two digests, 32
  bytes each (R1.5). The start is one constant of the configuration. A batch's binding tree (R7.18)
  has the number 178 instead of 166, and all else is the same. All that binds a tree to its run, its
  number and its positions is in the leaves. This rule is for perfect binary trees alone: a leaf
  beside an inner vertex, a top layer that is cut off, or a vertex with four children would get a
  number of its own.

### 6.3 An opening

- **R7.8** A tree is opened at a list x_1 < x_2 < .. < x_m (R6.6) of at least one position. The
  **opening**'s bytes are, without any length: for each x_j in order, the column and its salt (no
  fill); then the **given digests**, 32 bytes each with no length before them (R1.5), in the order
  of R7.9. So an opening has m (8 W + s) + 32 g bytes, where g is the number of given digests.
- **R7.9** The **walk**. It goes level by level from depth a up to depth 1, and within a level from
  left to right. At each depth it has the vertices that are above an opened leaf, by increasing
  place, each with its value: the level's vertices. At depth a these are the opened leaves. The walk
  goes through them by increasing place. If a vertex has an even place and the next of the level's
  vertices is its sibling, the two give their parent's value, nothing is given, and the walk goes on
  after the sibling. Otherwise the sibling's value is the next given digest, and the vertex and that
  digest give their parent's value. In both cases the child with the even place is the left one
  (R7.7). The parents, in the order in which they are made, are the level's vertices at the next
  depth. After depth 1 one vertex is left, the root.

  So the given digests are ordered by level, the deepest level first, and not path by path. Their
  number g follows from a and the list of positions alone: a verifier runs the walk once without
  values to count them before it reads. For a list of t positions with 2^e <= t < 2^(e+1), g is at
  most t (a - e - 1) + 2^(e+1) - t. Example: in a tree of depth 3 that is opened at the list 0, 5, 7
  the given digests are the values of the vertices (3, 1), (3, 4), (3, 6), (2, 1), in this order.
- **R7.10** The verifier reads the columns by its own shape and rejects a symbol by R1.3. It places
  the fill itself and hashes every opened leaf itself. The start, the tree's number and the position
  in that hash follow from its own plan and run, never from the file. It makes the walk, in which
  left and right come from the bits of the places, and rejects unless the walk uses every given
  digest, lacks none and ends in the root that was sent or, for a stored tree, in the root that is
  in the chain's first query (R4.6). It takes no index, count or direction from the file. (The one
  exception is a request's slot in the binding tree: R7.18.) It rejects a file that ends before the
  opening's last byte. The leaves of every tree are salted and are bound in this way.
- **R7.11** Two opened positions with one short position each show the mask rows' symbols there. The
  tree does not force the two to be equal. A verifier does not compare them: for each position it
  reads the mask rows' symbols from that position's own column (from the programs' source). What
  follows if they differ is the protocol's matter, not the format's.
- **R7.14** One value at a vertex. A verifier rejects if a vertex gets two different values under
  one root, whether they are computed or given. Its **record of values** is keyed by the root's 32
  bytes, within a session: there is one record for every root that is read in a session, whichever
  tree, chain or file the root came with. So two trees that are sent with one root, a tree that a
  later chain takes over, and a stored table that several chains read have one record for each root.
  Within a record a value is keyed by the number 2^depth + place, never by a digest's value. A leaf
  is a vertex, and its value is its digest. The root's own value is not entered: every walk's end is
  compared with the root. Only what an accepted opening shows is entered: an opening that is
  rejected leaves the record as it found it. (This record of values is not the record of a stored
  table, R4.8.)

  A **session** is one run of a verifier that ends in one verdict: all chains of a file, and all
  files that one verdict reads (the programs: 17.2, item 4). Between two sessions, as for a stored
  tree that many proofs open, nothing is remembered. Where every root is read at one list there is
  nothing to compare, and a verifier need keep no record.

  Why: if a vertex has two different values, and each is on a path that ends in the same root, then
  somewhere from that vertex up to the root two different inputs of H have one digest. That is a
  collision of the hash. The record is a cheap guard that makes that event visible, and no bound
  depends on it. An honest prover loses nothing by it: two honest trees with one root have the same
  values at every vertex that both have.
- **R7.15** A shorter tree under a longer list. Where a group is read at a list over [0, 2^k) and
  one of its trees has depth b < k, that tree is opened at the set of the short positions (R7.2) of
  the list, in increasing order, each once.
- **R7.16** A tree of depth 0 has one leaf, and its root is that leaf's digest. A tree's depth is at
  most 48.

### 6.4 Which oracle messages share a tree, and how trees are numbered

- **R7.12** Which oracle messages share a tree. Each oracle message has a **leaf class**, a number
  in its notice (R12.3). The oracle messages of one prover's message are placed into trees in two
  passes, each in the order of sending. First those with main rows: each joins the first tree of
  this prover's message that has its leaf class and main rows of its length, and starts a new tree
  if there is none. Then those without main rows: each joins the first tree of this prover's message
  that has its leaf class, and starts a new tree, of mask rows alone, if there is none. Within a
  tree the parts are in the order of sending, whichever pass they joined in: so a message of mask
  rows alone can be between two parts that joined before it. An oracle message has at least one row.
  The rule looks at nothing but class, length and order.

  So a protocol chooses its leaf classes in such a way that every tree that this rule makes has mask
  rows of one length (the introduction of this section) and meets R7.5. A shape that does not is not
  a shape of the format. Shapes are public parameters, fixed before a proof is seen: a verifier
  compares them with no file. That mask rows of one tree have one length can be seen from a shape,
  and a program stops, without a verdict, where it fails. R7.5's first clause is kept by R7.17; its
  other two are duties of anyone who writes a protocol.
- **R7.13** A chain's list of trees, and a tree's number. Every chain keeps a list of the trees that
  it can read. A tree's **place** is its number in that list, from 0. (This is not a vertex's place
  in its level, R7.6.) The list grows in three ways.
  - (a) A chain **continues** at most one earlier chain of its session, which the protocol names,
    and then begins with that chain's whole list. A request's second chain (R14.1) continues the
    request's own first chain. It does not continue the shared chain, although its first query has
    that chain's seal as well (R4.7).
  - (b) A tree that the chain sends takes the next place; the trees of one prover's message do so in
    the order in which R7.12 starts them.
  - (c) A tree that the chain **adopts** from another chain of its session takes the next place at
    the moment at which the protocol adopts it.

  A **tree's number**, which is in the hash's labels of its leaves, is the place that it got in the
  list of the chain that sent it. So an adopted tree uses up a place, and with it a number, of the
  chain that adopts it. It keeps its own number in its leaves. Among the trees that one chain reads,
  two can therefore have one number. Their leaves then differ in their start, the run's value of the
  chain that sent them. A list has at most 2^16 places, 0 to 2^16 - 1; a plan that needs more is not
  a plan of the format.
- **R7.17** What keeps R7.5's first clause. A leaf is opened whole, so all parts of a tree are
  opened together, at every list. From that follow four rules on a protocol's layout, that is, on
  which oracle messages are in which tree, to which groups they belong, and at which lists each tree
  is opened. (It is not the layout file of R4.9.)
  - (a) The parts of a tree that get an out-of-domain sample (their notices have a number of answers
    above 0, R12.3) belong to the same groups, each of whose tests reads them at its one list.
  - (b) A tree that holds an oracle message without an out-of-domain sample, that is, the mask of
    one step (the fresh mask of a fold, the masking word of a last step: R12.10), is opened once in
    all, and its other parts belong to one group at most.
  - (c) Such a tree is read in no other chain than the one that sent it. Here a chain that continues
    the sender (R7.13 (a)) counts as the sender: what (c) bars is adoption (R7.13 (c)) and nothing
    else.
  - (d) A chain that adopts a tree adopts it with all its parts.

  A layout is public, so a breach is an error of a program and never of a proof. The routine that
  plans an opening (`plan_open` in `blocks/numbers/blocks.h`) rejects, with a reason that names the
  tree, a second opening of a tree of (b) within one plan, and every opening of such a tree that is
  marked as another chain's. By its own comment it sees less than the rule: the other chain's
  planner must set that mark, nothing in the planning notices if it does not, and a tree whose sizes
  are filled in anew counts as another tree. Prover and verifier alike stop, without a verdict, at
  the first opening that breaks (a) or (b), and at the adoption that breaks (c) or (d) (from the
  programs' source).

  A tree that passes to a continuing chain takes the count of its lists with it, so over both chains
  it is opened once in all. An example is the tree of the oracle message W of R14.4 (e), which is
  not this section's number W. Where the word of `the test on the table` on a mask sent ahead is 1
  (R13.13, item 16), as in every file of the programs, that tree holds the fresh mask of the first
  fold of the second chain's test. It is sent on a proof's first chain, or on a key file's one-time
  chain, and it is opened on the second chain, which continues that chain (9.7, R14.4 (e), R14.10).
  (What this rule says of a continuing chain is from the programs' source.)

### 6.5 A batch's binding tree

- **R7.18** A batch's **binding tree** is a tree of this section with the tree's number 0, the
  number 173 for its leaves and 178 for its vertices, and the configuration's digest as the start of
  both. A request's slot in its batch is the place (R7.6) of its leaf in this tree, which the prover
  samples (section 16). The content of that leaf is the seal of the request's first chain, 32 bytes,
  which count as four words: W = 4. After it comes the salt, by R7.3; the fill is empty.

  At a slot that no request has taken the leaf's digest itself is 32 uniform bytes of the prover's
  private randomness: no hash is made and there is no content. That is a rule for the prover alone;
  no step of a verifier reads it. (The intent is that the number of a batch's requests does not show
  from the tree; this page makes no claim about that.)

  A file shows one leaf of this tree, by its slot, its salt and its path (R14.7, R13.6). A verifier
  computes the root from them and compares it with nothing: the root becomes an item of the shared
  chain's first query and is used instead of that chain's random bytes (R4.5, case (b)).

### 6.6 A round whose draws are divided among several trees

- **R7.19** A step of a protocol can give the draws of one round to several trees: the first n_1
  draws to one tree, the next n_2 to the next, and so on. Each n_i is at least 1 (from the programs'
  source). The numbers n_i are in a parameter item of that step; the plan's entry for the round has
  their sum as its number of positions, and one logarithm of the range for all. In the round `LINK`
  of a leaf file the item is `the link` and its words are nu_0 and nu_j (R13.9): the first nu_0
  draws are the anchor's and the last nu_j the leaf table's. Each tree's list is made by R6.6 from
  that tree's draws alone: sorted, each position once. A position that is among the draws of two
  trees is in both lists, and both trees are opened there. What follows the draws, repeats counted,
  follows the draws of its tree, in their order.

  Where the plan lists positions (R4.3, R8.2), the round has one list: the set of all its draws,
  sorted, each position once even if it is among the draws of two trees. It is right after the nonce
  and before all openings (R8.7).

  In a link each of the two stored trees is opened at one list. So, unless the two roots are the
  same 32 bytes, R7.14's records have nothing to compare within a leaf file, and a verifier whose
  session is that one file and that keeps none for a stored tree gives the same verdict. In a
  session of all three files (R7.14) there is something to compare: the anchor's tree is opened in
  the key file's shared chain and again in the link, and the leaf table's tree in the link and again
  in the proof file's shared chain (17.2, item 4).

### 6.7 Easy to get wrong

1. The given digests come after all columns and salts of the opening, level by level from the leaves
   up and from left to right within a level, not path by path. Their number is in no file: it is
   computed from the depth and the list before anything is read (R7.8, R7.9).
2. A column has the main symbols of all parts first and the mask symbols of all parts after them.
   Parts are in the order of sending, although trees are made in two passes (R7.1, R7.12). The fill
   is hashed and never written, and the 16 bytes of the hash's label count in it (R7.3).
3. The openings of a chain's own trees are ordered by the trees' places in the list of the chain
   that reads them. Stored trees have no place: those that a round reads are opened before the
   chain's own trees, in the order of the statement's items that hold their roots (R8.7). The hash's
   label of a leaf has the tree's number, which is its place in the list of the chain that sent it,
   and the leaf's hash starts from that chain's run's value. For a tree that a chain adopts, the
   place and the number can differ. For every tree that a chain takes over from the chain that it
   continues, and for an adopted tree, the start is not the reading chain's run's value (R7.4,
   R7.13).
4. An opening follows the list, in which each position occurs once. Whatever a protocol does once
   for each sampled position follows the draws, in which a position can come twice (R6.6).
5. The binding tree's root is computed and never compared (R7.18).

## 7. A chain in a file; the three files

This section says which bytes of a chain are in a file and in which order, and how the three kinds
of file, a key file, a leaf file and a proof file, are put together from a head, chains and a few
items between them. It uses the encodings of R1.1 to R1.5; a chain's first query, statement, plan
and random bytes (R4.1 to R4.7); rounds, kinds, nonces and a round's answer (R5.1 to R5.6); the
draws and the list (R6.6) and the seal (R6.8); and oracle messages, trees, openings and the binding
tree (R7.1 to R7.19). What each round of each chain sends is in sections 9 to 11.

### 7.1 Ground rules

All integers are encoded lowest byte first (R1.1). An element of F is 8 bytes and an element of E is
24 bytes; a verifier rejects an element with a coordinate that is not below p (R1.3, R1.4). A
digest, and so a root, is 32 bytes, with no length before them (R1.5), and any 32 bytes are a
digest. No file has a length, a count or an offset. A verifier first derives the plan of every chain
from the public files (section 8), and then takes the planned items off the file one after the
other. The length of an opening depends on the positions that were sampled, so the length of a file
does not follow from the public files alone, and a verifier cannot test it before it runs the
chains. All chains of a file are read from one stream of bytes: after the first rejection nothing
more is read, and the verdict is a rejection (from the programs' source).

### 7.2 The items of a chain

- **R8.1** A chain's bytes in a file are its items (those of R8.2's table, not the named items of
  R4.1) in the order of the protocol, without lengths, names or separators. Every length follows
  from the plan, from the shapes and from the sampled positions. A verifier rejects a file that ends
  before the last item, and a file with a byte after it.
- **R8.2** The items, and whether they enter a query:

| item | bytes | enters a query |
|---|---|---|
| the random bytes, in case (a) of R4.5, as the chain's first item | 32 | through the run's value |
| an answer to an out-of-domain sample | 24 | yes, as part of the prover's message |
| an element of F; of E; a word | 8; 24; 8 | yes, the same |
| a root | 32 | yes, the same |
| the nonce of a round with w > 0, after the round's message | the plan's number of bytes of a nonce, which is 1 to 8 even where no round has proof of work. They are the low bytes of the nonce, lowest first; its other bytes are 0 | as the last word of its own round's calls, and of no other |
| after a round that samples positions, if the plan's number of bytes of a listed position is not 0: the list, in increasing order, each position once | that many bytes for each position, lowest first | no. The verifier compares it with the list that it has derived, and rejects if it differs |
| an opening (R7.8) | | no. It is checked against its root |

Three widths, the number of bytes of a leaf's salt, of a nonce and of a listed position, are the
third, fourth and fifth of the plan's eight words (R4.3). The parameter lines `salt` and `listed`
set the first and the third; a nonce has 8 bytes in the three files of this section. The parameter
file `record.f2.txt` (section 8) has a salt of 40 bytes and no listed positions, and asks for proof
of work only in rounds that sample positions. The chain's 32 random bytes are not a leaf's salt
(R7.3), which has its own length. A round with w = 0 has no nonce in the file and makes its query
with the nonce 0 (R5.3). Nothing else of a chain is in a file: no value of the chain, no block of a
round's answer, no seal (R5.6, R6.8).

### 7.3 The order within a prover's message

The kinds of a round are those of R5.2: 1 elements of E; 2 the element that joins a hiding
polynomial to a claim; 3 the seeds of out-of-domain samples; 4 positions; 5 entries of a matrix over
{-1, 0, 1}; 6 a seal; 7 nothing.

- **R8.3** After a round of kind 3 the prover's next message begins with all answers to
  out-of-domain samples that the seeds ask for. They come oracle message by oracle message in the
  order sent, and for each oracle message as answer 1, 2, .. . An oracle message whose notice
  (R12.3) has q answers sends one answer for each of the first q seeds, whatever the number of its
  rows, and the round has as many seeds as the largest such q (from the programs' source). So where
  all these oracle messages have one number of answers, there is one answer for each oracle message
  and seed. The plans that the programs make count it so: s seeds, and s answers for each of these
  oracle messages, with the s of R6.9 (from the programs' source: `blocks/numbers/blocks.h`). With
  plans and shapes that are derived as 8.4 says, every such q is 0 or that s, so oracle messages
  with different numbers of answers above 0 do not arise (from the programs' source:
  `blocks/frame/lin_frame.h`, `link_frame.h`, `req_frame.h`, `wf_frame.h`); for a verifier that
  takes plan and shapes as inputs this is a condition on them (section 18). The number of all these
  answers is a word of the plan's entry for that next round, the fifth of its eleven (R4.3); that it
  is what the seeds ask for is a condition on plans. (A round of kind 3 samples, for each answer
  number, the coordinates of a point by R6.4 and then one element that is not 0 by R6.5. All oracle
  messages in that round's own prover's message share these seeds. An oracle message whose notice
  (R12.3) has 0 answers gets no out-of-domain sample and sends no answer.)
- **R8.4** In a round of kind 2 the sum of the hiding polynomial is the last item of the prover's
  message.
- **R8.5** A root is in the message of a round of kind 1 to 5. This is a condition on protocols: no
  oracle message is sent before an empty message of the verifier, or before a seal.
- **R8.6** The roots of one prover's message are in the order of their trees' numbers. There is one
  root for each tree that R7.12 makes of the message's oracle messages.
- **R8.8** How a message is cut into items is part of the format: a verifier has to know which 8
  bytes are an element, and so can be rejected, and which are a plain word. (In the three files of
  sections 9 to 11 no prover's message has a plain word: all its bytes outside a root belong to
  elements, and each element is tested for its canonical form. From the programs' source: the
  routine of `blocks/chain/rounds.h` that receives a plain word is called by a test alone.) The
  plan's entry gives lengths alone. The list of items, round by round, belongs to each protocol
  (sections 9 to 11).

### 7.4 After a round that samples positions

- **R8.7** Where an opening is. After a round that samples positions come, in this order: the
  round's nonce, if it has proof of work; the listed positions, if the plan lists them; the openings
  of the trees that are read at this list; and then the items of the next prover's message. The
  chain's own trees are opened by increasing place in the reading chain's list of trees (R7.13), not
  by the numbers in the hash's labels of their leaves, which can differ from the places for an
  adopted tree. Stored trees (R7.4) are in no chain's list and have no place: those that a round
  reads are opened first, in the order of the statement's items that hold their roots (R4.6), and
  then the chain's own trees. Where the draws of a round are divided among several trees (R7.19),
  "this list" is for each tree the list of its own draws, and the listed positions are one list for
  the whole round.
- **R8.10** A plan whose number of bytes of a listed position is n > 0 and that samples positions
  from [0, 2^k) with k > 8 n is not a plan of the format.

The number of listed positions is the length of the list, not the number of draws (R6.6), so a
verifier derives the list before it reads the listed positions. A tree that is less deep than the
list's range is opened at the short positions of the list (R7.15). Which trees a round opens is in
R13.12.

### 7.5 Several chains in a file

- **R8.9** Chains in a file. A file can hold several chains, one after the other, with other items
  between them, in an order that belongs to the kind of file (R13.2, R13.3, R14.1). R8.1's rejection
  of a byte after the last item is made once, at the end of the file.

A chain on which other chains build ends with one more round, of kind 6. Its **seal** (R6.8) is 32
bytes of that round's answer. No byte of the seal is in a file. In the three files of this section
that round's prover's message is empty (from the programs' source). A later chain has the seal in
its first query, as an item `the seal of an earlier chain` (R4.7). The item's value is the seal's 32
bytes (R1.5). The item's own word for the length of its value, 32, comes before them as in every
item (R4.1), and there is no second length.

A verifier that has rejected anything in a chain, before its seal or after it, rejects every chain
that has that chain's seal. The verdict on a file is an acceptance only if no chain has rejected,
every chain has run all the rounds of its plan, and no byte is left. (These two sentences are from
the programs' source.)

### 7.6 A key file

A **key file** comes with a commitment. It has the root of the anchor, the commitment's first stored
table (section 4), and the committer's proof about that table (section 9).

- **R13.1** The head. A key file and a leaf file each begin with 8 bytes: four bytes that say what
  kind of file it is, `CPIK` for a key file and `CPEN` for a leaf file, and the four bytes of the
  format's version, `H821`. A verifier rejects a file of another kind or version before it reads on.
  (`H821` is the version word of field 2; field 1 has `H811`.)
- **R13.2** A key file, in order. No part has its own length as a word: a verifier knows every
  length from the public files and, for an opening, from the sampled positions (R8.1). A verifier
  rejects a file in which a byte is left over.

| part | bytes |
|---|---|
| the head | 8 |
| the anchor's root | 32 |
| the stored answers of its record, by row and within a row by answer (R13.8) | 24 each. Their number is in no file |
| the **one-time chain**: its 32 random bytes, then its rounds by R8.1 to R8.10 | by its plan |
| the **slot**: the word 0, and the salt of a leaf of the binding tree | 8 and the number of bytes of that salt |
| the **shared chain**: its rounds by R8.1 to R8.10, without random bytes | by its plan |
| the **second chain**: its 32 random bytes, then its rounds by R8.1 to R8.10 | by its plan |

The root and the stored answers are items of the file and not messages of a chain. A verifier reads
them first and puts them into the statement of every chain that speaks of the table, whether or not
that chain opens a column of it (R4.6). In a key file these are the one-time chain, which opens no
column, and the shared chain, and the item's name is `table`; the second chain has them only through
the two seals of its statement. A leaf file's link has the anchor's root and stored answers under
the name `the first table` (R13.5). In the file the root is its 32 bytes alone; in either item it is
encoded as a string of bytes, with its length 32 as a word before it.

A table has 2^v times as many stored
answers as a row has (the fifth word of R4.10). The seeds of a record are in no file (R4.8).

The one-time chain and the shared chain end in a seal; the second chain does not. The slot is that
of a batch of one: a binding tree of depth 0 (R13.6), so a verifier rejects a word other than 0. The
second chain continues the one-time chain and adopts one tree of the shared chain (R13.7).

### 7.7 A leaf file

A **leaf file** comes with a leaf table, a further stored table with the same rows under fresh
randomness, and links it to the anchor (section 9).

- **R13.3** A leaf file, in order: the head; the leaf table's root, 32 bytes; the stored answers of
  the leaf table's record, as in R13.2; the **link's chain**: its 32 random bytes, then its rounds
  by R8.1 to R8.10. The file ends with the chain.

The link's chain has no seal. Its statement has both tables' items; the anchor's root and stored
answers are those that the verifier has kept from the key file, not bytes of the leaf file (R13.5).
Its round `LINK` samples the positions of both tables at once. After the nonce, and after the listed
positions if the plan lists them, come the anchor's opening at the list of the first nu_0 draws and
then the leaf table's opening at the list of the last nu_j draws (R7.19, R8.7).

### 7.8 A proof file

A **proof file** goes to one client and is about one request of a batch (section 10).

- **R14.1** The file, in order:
  1. the head;
  2. the request's **first chain**: its 32 random bytes, then its rounds by R8.1 to R8.10;
  3. the request's slot in its batch;
  4. the batch's shared chain: its rounds by R8.1 to R8.10, without random bytes;
  5. the request's second chain: its 32 random bytes, then its rounds by R8.1 to R8.10.

  No part has its own length as a word, and a verifier rejects a file in which a byte is left over,
  as in R13.2. The shared chain's bytes are the same in the files of all requests of a batch.
- **R14.2** The head is 12 bytes: the four bytes `CPPF`; the four bytes of the format's version,
  `H821` (R13.1); and a number in four bytes, the lowest first. It is the number under which the
  leaf table is kept on which the proof was made. Beside the version word it is the only integer to
  which the form of the three files itself gives four bytes; a listed position has four only where
  the plan says so (R8.2). The same number is the item `stored copy` of the first chain's statement
  (R14.3). A verifier rejects a number other than the one under which it holds the table that it
  reads.

  How the programs keep this rule (from the programs' source): leaf table k has its leaf file at
  `leaf/<k>.bin`, with k in decimal, in the public key directory, the directory that holds the key
  file (section 1). The programs' verifier chooses the table by the head. It takes k from the
  proof's head, makes no test on k itself, and opens that leaf file, for k = 0 as for every other k.
  It rejects if there is no such file. It checks the proof against the root in that leaf file and
  never against the root in the key file. A root that is in a file under a number counts for nothing
  by itself: in the same call the verifier checks that leaf file's link to the anchor and the key
  file, on the same bytes, and it accepts only if all three checks pass (13.2; 13.6). So the number
  under which it holds the table is always the head's, and the rejection above has no case of its
  own there. The prover picks the number: by a comment in the source of `cp_verify`, where it opens
  the leaf file, k is the proof head's.

In the file the slot is three things: one word, which is the request's slot itself and which a
verifier rejects unless it is below 2^depth; the salt of that slot's leaf, whose length is the
second word of the item `binding tree` (R14.6); and as many digests as the binding tree's depth, the
sibling of the leaf first (R14.7). The depth is the least d with 2^d not below the parameter line
`slots`: 6 for the 64 of `record.f2.txt`, so that a slot has 8 + 40 + 6 x 32 = 240 bytes there (from
the programs' source). The slot's word is the only position that a verifier takes from a file.

The slot is between the first chain and the shared chain because the first chain's seal, the salt
and the path give `the binding tree's root`, which the shared chain's first query needs (R7.18,
R14.9). The item's value is the root's 32 bytes (R1.5), after the item's own word for their length
(R4.1), with no second length. That root is also used instead of the shared chain's random bytes
(R4.5, case (b)), which is why that chain has none in the file.

The first chain and the shared chain end in a seal; the second chain does not. The second chain's
statement has both seals, the first chain's and then the shared chain's (R4.7); it continues the
first chain and adopts one tree of the shared chain, as in a key file (R7.13, R13.7, R14.10).

| file | chain | random bytes in the file | ends in a seal | its list of trees begins |
|---|---|---|---|---|
| key file | one-time chain | yes | yes | empty |
| key file | shared chain | no | yes | empty |
| key file | second chain | yes | no | with the one-time chain's list |
| leaf file | link's chain | yes | no | empty |
| proof file | first chain | yes | yes | empty |
| proof file | shared chain | no | yes | empty |
| proof file | second chain | yes | no | with the first chain's list |

### 7.9 What a verifier does with each kind of bytes

| bytes | what a verifier does |
|---|---|
| the head's bytes | compares them with its own values |
| elements of F or E: scalars, answers to out-of-domain samples, stored answers, columns | rejects unless every coordinate is below p |
| digests: roots, given digests, a slot's path | reads any value |
| a nonce | tests the round's proof of work with it (R5.5), before anything is sampled from the round's answer |
| listed positions | compares them with the list that it derives |
| salts and a chain's random bytes | reads any value. It rejects a chain whose run's value is 32 zero bytes (R4.5) |
| the word of a slot | rejects unless it is below 2^depth |

At every item a verifier rejects a file that ends too early.

### 7.10 Easy to get wrong

1. The shared chain has no random bytes in the file; the other chains begin with 32 (R13.2, R14.1).
2. In a key file and in a leaf file the root, and the record's stored answers, come
   before the chain's random bytes (R13.2, R13.3).
3. After a round that samples positions the order is: nonce, listed positions, stored trees'
   openings, the chain's own trees by place, next message (R8.7).
4. The test that no byte is left is made once, after the last chain of the file (R8.9).
5. A seal is in no file, and neither is the binding tree's root (R6.8, R14.7).

## 8. The parameter file, and what is derived from it

This section is a file of its own: [08-parameters.md](08-parameters.md).

## 9. A key file and a leaf file, round by round

This section is a file of its own: [09-key-and-leaf-files.md](09-key-and-leaf-files.md).

## 10. A proof, round by round

This section is a file of its own: [10-proof-file.md](10-proof-file.md).

## 11. The sub-proofs of a segment

This section is a file of its own: [11-sub-proofs.md](11-sub-proofs.md).

## 12. The request, the next token, the policy

This section says what the programs `cp_prove` and `cp_verify` bind as the items `request` and `the
request file's words` in the first query of a proof's first chain (R14.3, items 3 and 4), how the
next token is sent and compared (R14.8), and where the acceptance policy is. Its last subsection,
12.6, writes out the second hash function of R15.3, with known answers. It needs sections 2 to 4 and section
10: item and first query are defined in R4.1 to R4.6, round in R5.1, segment in section 1, the
parameter file and the scale file in sections 1 and 8. In this section a **word** has 32 bits and is
encoded in four bytes, the lowest first, unless its width is named. The word of R1.1, which has 8
bytes, occurs once, in R15.1. The letters K and TK are this section's own.

### 12.1 Easy to get wrong

- The item `request` is not the bytes of the request file. It is the **statement**: a sequence of
  177 words that each program makes for itself, by its own code, from the parameter file, the scale
  file and the request file (R15.5). No word of it comes from a key file, a leaf file or a proof
  file. The bytes of the request file are an item of their own, `the request file's words`, directly
  after the item `request` (R14.3, item 4): both go into the first query.
- Nine digests in the statement are made with the second hash function (R15.3, 12.6) and not with H of
  section 3: five of texts and files, and four of public tables, which are also lines of the
  parameter file (R15.6).
- The next token is not in the item. Its position, word 171, holds 0 (R15.7). The prover sends the
  next token in a round of its own, and a verifier compares it with the caller's next token once,
  after all segments (12.4).
- A request whose checked positions do not include the last one has no next token: the caller's
  token file must then hold 0 (12.2).

### 12.2 The caller's two files

The tests in this subsection are not among the numbered rules (from the programs' source); section
13 has their names in one table.

The **request file** holds T + 2 words, where T = 2^lt and lt is a line of the parameter file: T
**tokens**, and then two positions, r and e. The proof **checks** the positions from r up to and not
including e: these are the **checked positions** (the programs print "covered": the line `covered:`,
14.7). Position t is about token t + 1 of the request; position T - 1 is about the **next token**,
the token that follows the request. The token that a checked position is about is a **checked
token** if it is one of the request's own, so the checked tokens are the tokens r + 1 to min(e,
T - 1) of the request. The next token is not counted among them. If e = T, position T - 1 is checked
and is about the next token, which has no cost and must be the top choice of that position (the
index of its largest logit, the lowest index on ties), whatever the policy is (12.5). The **token
file** (the programs call it the y file) holds one word, y: the next token that the caller was
served.

A verifier rejects the call unless all of the following hold.

1. The length of each file is a multiple of 4 bytes, and neither file is empty
   (`statement-required`).
2. The request file has exactly T + 2 words and the token file exactly one (`statement-range`). T
   comes from the parameter file alone.
3. r < e <= T (`statement-range`).
4. If e < T then y = 0 (`statement-range`). Why: in that case no checked position reads y, so a
   value other than 0 would suggest a next token of which the proof says nothing.
5. Every one of the T tokens, checked or not, and y are below V, the vocabulary's size, a line of
   the parameter file (`token-range`).

Every token of the file is an input of the model's run, so a proof fits one request file only. But
an acceptance is about the checked tokens and, if e = T, the next token, and about no other token.
The request file, whole, is the value of the item `the request file's words` (R14.3, item 4), and a
digest of it is among the words of the item `request` (R15.5, R15.6; 12.6).

### 12.3 The statement

The rules of this subsection hold for one form of the parameter file: the line `prof` is 20:
attention, the embedding and the output head are proved (the programs' notes call this profile 4);
`NE` is not 0; and the 32 lines of the tables' digests are there. `cp_verify` accepts under no other
form. It runs some other forms to their end without an acceptance. This page does not define them.
Of their statements it says this much. The 32 lines can be absent, and must be absent
where `NE` is 0; the eight-word positions then hold the digests that the program computes itself,
and zeros for `tsig` where `NE` is 0.

The rules say which value is in which word. They do not say what a word means for the model, and
they do not list the tests that the programs make on these numbers; section 13 lists those.

- **R15.1** The value of the item `request` is 177 words of four bytes each, the lowest byte first:
  708 bytes. With its length before it, as a word of 8 bytes (R1.6), it is encoded in 716 bytes.
- **R15.2** *The lines that are read.* The parameter file is a text of lines in ASCII. Each line
  ends with a line feed and is a name and then numbers or names, separated by one blank. A number is
  a decimal numeral without a sign and without a leading zero, below 2^32. In a file that the
  programs admit, none of the names below occurs twice. First come the frame lines, from `version`
  to `table` (section 8). Of these the statement takes three: `version`, whose value is the four
  characters of the version word (R4.3); `sec B`, the target: the number of bits required of every
  round; and `table b n i m e w`: the committed table has 2^b rows (b is the v of R4.10) in a code
  of length 2^n, i entries in a row's message and m entries in its random tail, the number e, a constant of the format (8.2), and
  w coordinates of the point that a batch shares. Then come the lines of the names in R15.5, from
  `prof` to `F`, each a name and one number. Then come lines that give no word. Then come 32 lines
  for four digests, `tsilu0` to `tsilu7`, `tsig0` to `tsig7`, `trope0` to `trope7`, `texp0` to
  `texp7`, each one word. Then come the **call lines**, whose name is `call`, `callA` or `callH`.
  The programs reject a file in any other order or encoding; the tests are in section 13.
- **R15.3** *The second hash function.* Here R3.5 does not hold: anyone who forms this item makes five
  digests with the **second hash function**, and not with H, and four more
  that were made so are in the parameter file (the programs compute these four too, and reject if a
  line differs: R15.6). The request file's words also enter the first query under H itself (R14.3,
  item 4). The digest of the request file in the item `request` is then redundancy that the verifier
  recomputes. Of the two hash functions, binding the request then needs H alone, treated as a random
  oracle, so long as the second hash function, which is built on H's own compression function, does not
  disturb that treatment: the two differ, under the four numbers in use, in the counter's high half
  (12.6), and that this keeps them apart is a weak assumption about the pair, which no formal proof
  states. The second hash function is used here as the fixed, public function that it is, not as a second
  random oracle. One thing hangs on the second hash function alone: which four tables, bit for bit, a
  parameter file names, and so a commitment, since only their digests under K (defined below) stand
  in the file (12.6; 17.1, item 8). A verifier's verdict is about the tables that it computes
  itself, or takes as data (11.1), and holds against the file's 32 digest lines. Two verifiers, or a
  definition that takes the tables as data, speak of one function if their tables are the same; that
  each one's tables fit the lines shows this only as far as nobody can find two tables with one
  digest under K (collision resistance of K: an assumption of such a statement, and of no other on
  this page). The second hash function's digests in the item, other than the request's, are of public data:
  digests of lines of the parameter file and of the scale file, whose every byte the configuration's
  digest under H (R4.9; section 3) also covers, so that the commitment ID that the client holds
  binds them under H; and digests of the four tables (SiLU, sigmoid, rotary, softmax) that
  `cp_verify` and `cp_prove` compute themselves from the parameters (12.6): they put their own
  digests of them into the item and only compare the parameter file's lines with these (R15.6);
  `format_rules.py`, and a verifier that takes the tables as data, take the item's words for them
  from the lines (17.2, item 7). Taking the item's words from the lines does not excuse a verifier
  from holding its tables against them: one that takes the four tables as data still computes each
  table's digest under K and rejects if it differs from the file's line, or its verdict is about
  tables that nothing ties to the commitment. That the prover's and the verifier's tables agree to
  the bit is a matter of completeness, not of soundness. (This paragraph's account of what rests on
  which hash is a reading, not a proved statement.) Take the compression
  function of BLAKE2b (RFC 7693, 3.2), with a state of eight words of 64 bits, a block of sixteen, a
  counter t of 128 bits and a flag. Two words of 32 bits make one of 64, the low one first; on the
  way back a word of 64 bits gives its low half first. **K(num; w)**, for a number num below 2^32
  and a list w of n words, n below 2^32: the state begins as BLAKE2b's IV with 0x01010040 xored into
  its first word. The list is cut into blocks of 32 words, at least one; the last is filled up with
  zero words. Block j, from 0, is fed by one call of the compression function. The low 64 bits of t
  are the number of bytes fed up to and including this block: 128 (j + 1), and 4 n in the last call.
  The high 64 bits of t are num + 2^32 n, in every call. The flag is set in the last call alone. The
  digest is the first eight words of the last state: the first four words of 64 bits, each low half
  first. **TK(w)**, for a list of at least eight words: cut w into pieces of 128 words, the last
  perhaps shorter, and put K(10; piece) for each piece; while more than eight words are left, do the
  same to the list of digests under the number 11.
- **R15.4** **The words of a text** are its ASCII bytes, four to a word, the first byte lowest; the
  last word is filled up with zero bytes.
- **R15.5** *The 177 words, in order.* A name means the number of the line of that name.

  | words | what |
  |---|---|
  | 0, 1 | 0x4D363100; the version word: the four characters of the line `version` as one word, the first lowest |
  | 2 to 28 | `prof` `lt` `L` `ll` `d` `H` `G` `dh` `f` `fe` `NE` `V` `eps1_lo` `eps1_hi` `eps2_lo` `eps2_hi` `cs` `theta0_lo` `theta0_hi` `hq` `hv` `ho` `hgD` `huD` `hdD` `hr` `ut` |
  | 29 to 36 | the digest of the lines `call` (R15.6) |
  | 37 to 70 | `la` `nmin` `D` `tau` `lmaf` `NAf` `lmad` `NAd` `lmar` `lmb` `lmq` `lmx`; `wb_qk` `wb_v` `wb_o` `wb_g` `wb_u` `wb_gs` `wb_us` `wb_gr` `wb_ur` `wb_dn` `wb_dnr` `wb_r`; `iota_0` to `iota_7`; `seed_lo` `seed_hi` |
  | 71 to 102 | four digests: `tsilu0` to `tsilu7`; `tsig0` to `tsig7`; `trope0` to `trope7`; the digest of the scale file (R15.6) |
  | 103 to 113 | `lB` `S` `rx` `kFM` `kFD` `lmS` `NAs` `lmh` `lmo` `lmbA` `iotaA_0` |
  | 114 to 129 | two digests: `texp0` to `texp7`; the digest of the lines `callA` |
  | 130 to 154 | `MF` `KF` `epsF_lo` `epsF_hi` `toksrc` `lmhf` `NHf` `lmhd` `lmhr` `lmbH` `wb_emb` `wb_head` `iotaH_0` to `iotaH_5` `C` `theta` `cap` `s` `sh` `sT` `F` |
  | 155 to 170 | two digests: of the lines `callH`; of the request (R15.6) |
  | 171 | 0: the position of the next token (R15.7) |
  | 172 | B of the line `sec` |
  | 173 to 176 | of the line `table`: w; b; n; m |

- **R15.6** *The digests that are computed.* Of the lines whose name is `call`: K(73; the words of
  the text that is these lines, each whole, with its name and its line feed, in the file's order).
  The lines `callA` and the lines `callH` likewise, each kind by itself, under 73 too. A file need
  not have a line `callH`. If it has none, the text is that of these six lines, each with its line
  feed: `callH R16a R16 NF NFH HD0`, `callH R16b R16 HD1`, `callH R8 R8 ROW HD2`,
  `callH FEa FE HE0`, `callH FEb FE HE1`, `callH FC FC CNF`. (That is what `cp_verify` does.
  `cp_commit` and `cp_prove` reject such a file, so no proof of the programs is made under one.) A
  file without a line `call` or `callA` is rejected by all three
  programs. Of the scale
  file: TK(its bytes as words). Of the request: TK(the words of the request file), all T + 2 of them
  (12.2). The four digests that are lines of the parameter file are K(72; table) of four public
  tables of the model's functions. What is in the statement is the digest that the program has
  computed itself; since it ends the run if the line differs, the two are the same wherever a run
  goes on. R15.6 does not define how the tables are made; 12.6 gives their lists from the programs'
  source.

### 12.4 The next token

- **R15.7** *The next token's position.* In the programs' own statement word 171, the word before B,
  holds the next token. In the item it is 0: the next token is not part of what the first query
  binds, the prover sends it in the round `REPLY` (the round's name means the next token), and the
  verifier compares (R14.8). There the token y is the element of E with e_0 = y and e_1 = e_2 = 0
  (R1.4).

The comparison, as `cp_verify` makes it: the verifier turns the caller's y into the element (y, 0,
0). It reads the prover's element in the round `REPLY`, which comes directly before the last
segment's first message. It goes through all segments of the first chain, to the end of their last
proximity test. Then, before it reads anything more of the first chain, it compares the two
elements, all three coordinates. If they differ it rejects (`reply-differs`). An element that is not
of the form (y, 0, 0) with y below V is rejected earlier, inside the last segment (`header-y`,
13.8). The next token is compared at this one point and nowhere else. The next token enters the
chain's hash as a prover's message, so every value that the verifier samples later depends on it.

### 12.5 The acceptance policy

The **acceptance policy** is not a file. It is seven lines of the parameter file, `C` `theta` `cap`
`s` `sh` `sT` `F`, the last seven of the lines that give a word: words 148 to 154 of the statement.
So the policy is fixed three times over for a whole call. The configuration's digest (R4.9) takes
every byte of the parameter file, and it is a parameter item of each of the proof's three chains
(R14.3, R14.9, R14.10). The item `request`, in the first chain's statement (R14.3), holds the seven
words. And the commitment ID (R4.11) takes the same digest: each stage forms the ID from the public
files and the beginning of `key.bin`, and compares it with the one that the caller names after `-id`
(13.5; without that option nothing is compared, and the call cannot end with an acceptance). The
first two tie the proof to the policy. The third ties the caller's ID to the public files and to the
anchor's root and stored answers. By itself it ties no proof to an ID: the ID is in no first query
(R4.6), and that comparison is the only test of a stage in which the caller's ID meets the files
(13.5). A proof meets the ID only by way of the tables. Its item `table` holds the leaf table's root
and stored answers (R14.3), and the stage leaf tests the leaf file's link against the anchor's root
and stored answers (R13.5; 13.6), which the ID takes. So a whole call accepts a proof under an ID
only beside a leaf file whose link passes against that ID's anchor (13.7). This subsection has no
numbered rules (from the programs' source).

A checked position t < T - 1 has a **cost**. Write win for the position's token and lam[v] for the
position's integer logit of the vocabulary entry v. The position's **gap** is the largest, over all
v < V, of lam[v] - lam[win] + [v < win], where [v < win] is 1 if v is below win and 0 otherwise. So
the gap is 0 exactly if the token's logit is the largest, with the lowest index on ties, and a token
that ties with a lower-numbered one has the gap 1, not 0. The cost is the gap plus 2^sh; or, where
sT >= sh + 4, so that the head's table holds exponentials and is not the step table (below; 11.4;
13.4, step 4), the token's negative log-probability at the exponent sT, if that is greater. That
number is in units of the integer logits, like the gap: it is 2^sT times the natural logarithm of
the sum over v < V of exp((lam[v] - lam[win]) / 2^sT). C and theta are in units of a cost, not of a
gap. (11.4 also has a margin, one for each v: the number that the prover sets against the position's
cost, less lam[v] - lam[win] + [v < win].)

Let R be the number of checked positions below T - 1 (RC in 11.4). A proof is about the following:
the costs, each counted as at most C, add up to at most theta R; at most cap of them are above C;
and, if e = T, y is the top choice of position T - 1, with the lowest index on ties.

A proof is also about the run itself, at all T positions, checked or not: the segments before the
head, the final norm, its wide output and the clamp are tied at every position (11.4). So it is also
about this: no value of the integer model leaves a table or a range at any position. And a token
outside the checked positions, also one that only fills the request up to T tokens, can bring the
model out of a table or a range, and then an honest prover has no proof of the request (this page's
reading of the programs' source; that a request whose values leave a range fails is said by a
comment of `prover/hash_commitment/hf_commit.cu` and by row H7 of `docs/limits-all-rows.tsv`).

A verifier rejects a parameter file (`policy-gap-words`) unless all of these hold:

- theta <= C;
- cap <= T and cap < 2^16;
- F = 24;
- s <= 63;
- sh is 7 or 8;
- sT = sh - 4 or sh + 4 <= sT <= sh + 10;
- 2 V <= 2^F;
- 2^(la - lt) divides JVa. Here JV is V rounded up to a multiple of 2^(la - lt), and JVa is half the
  least power of two that is not below JV.

With sT = sh - 4 the head's table is the **step table**, under which every cost is the gap plus
2^sh. The line `s` is what the provider gives as the exponent of the model's logits. It is printed
and compared (below), and it is in no equation.

The lines set one of four **kinds** of policy:

| kind | condition |
|---|---|
| `exact-match` | C = theta = 2^sh, cap = 0, sT = sh - 4: every checked token is the top choice of its position |
| `clipped-gap` | sT = sh - 4, with any other C, theta, cap |
| `log-probability` | not one of the two above, and sT = s |
| `another-table` | every other case |

What this subsection says of a table of exponentials (sT >= sh + 4: the kinds `log-probability` and
`another-table`, and the second form of a cost) is about the stages' code alone. No whole call of
`cp_verify` judges a token under such a table: under an sT other than sh - 4 a whole call ends with
the rejection `NO: policy` before any stage starts (14.3). `cp_commit` and `cp_prove` reject such a
parameter file at their start (`prover-only-policy-kind`), so no proof of the programs is made under
one.

Under other kinds than `exact-match` an acceptance can say less, down to nothing: theta = C with cap
= T admits any tokens, and so does theta = C with any cap that is at least R. That is so of the
proof; `cp_verify` rejects a call that checks a token of the request under such numbers, with the
option `-policy` or without it (below). Where C <= 2^sh and theta < C, no request with a checked
position below T - 1 is accepted, since every cost then counts as C. So the kind limits the result
of a call (13.7, condition 11; 14.3). A whole call of `cp_verify` takes the kinds `exact-match` and
`clipped-gap` alone. Without the option `-policy`, a call that checks a token of the request can end
with an acceptance only under `exact-match`. With the option it can do so only if the kind and the
five numbers C, theta, cap, s, sh that the caller names equal those of the file, and only if the
named numbers exclude something: C and theta are at least 2^sh, theta is at most C, and theta = C
does not come with 0 < R <= cap. A call that checks a token of the request and misses one of these
ends with the rejection `NO: policy`, before the stages; 14.3 has the four cases. A call that checks
no token of the request (R = 0: the next token alone) is not rejected for theta = C, whatever cap
is. `exact-match`, which has theta = C and cap = 0, is never rejected by this test. The other tests
on what `-policy` names are made in a call that checks no token of the request too (13.7, condition
11). Beyond that the program does not judge the numbers: whether a theta is strict enough for a
purpose is the caller's question. The option can name `exact-match` and `clipped-gap` and no other
kind. An acceptance in which a token of the request was checked under a `clipped-gap` policy has the
same exit code as one under `exact-match`, 0, and another last line:
`WITHIN LIMITS: clipped-gap C=.. theta=.. cap=.. s=.. sh=..`, and not `YES` (14.3).

### 12.6 The second hash function

The item `request` holds nine digests that are not made by the hash H of R2.1 but by the second hash function (R15.3; `verifier/HASHES.md` names its source files; it is the one exception to R3.5). A second verifier
needs it to form that item and to check the 32 digest lines of the parameter file. This part defines
the second hash function completely, lists every digest that is made with it, and says for each whether it
enters a hashed string of the format or is compared with a line of the parameter file. It gives the functions K and TK of R15.3 again, in steps; R15.3 to R15.6 take precedence,
and what goes beyond them is marked "(from the programs' source)". The programs' sources call K and TK `H` and `TH`. Names in the sources that begin with `earlier_hash_` belong to this hash function (`prover/common/hash.h`, `verifier/src/vf_earlier_hash.h`), and a comment in the first of these files cites this part as "12.6 The earlier hash". In `format_rules.py` K and TK are `older_K` and `older_TK`, and `blocks/hash/older.h` has this hash function in a struct named `Older`.

*In this part a word has 32 bits,* as in R15.1 to R15.7, and is encoded in four bytes, the lowest
first. A **long word** has 64 bits. Two words make a long word, the low word first; a long word
gives back its low word first. Lists are indexed from 0.

#### The functions

**F** stands for BLAKE2b's compression function. RFC 7693 (M-J. Saarinen (ed.), J-P. Aumasson, "The
BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", November 2015) defines it in
section 3.2, with the parameters of BLAKE2b in section 2.1. F(h, m, t, last) takes a state h of
eight long words, a block m of sixteen long words, a counter t of 128 bits and a flag, and returns a
state. This page does not repeat its steps or its constants. F is not the field F of section 2.

A **state** is written here as 16 words and a **block** as 32 words, each pair of words one long
word. A **digest** is 8 words: the first eight words of a state, which are its first four long
words, each low word first.

The **first state** is the state from which BLAKE2b starts for a digest of 64 bytes and no key (RFC
7693, section 3.3, with nn = 64 and kk = 0).

**K(num; w)**, for a number num below 2^32 and a list w of n words, n below 2^32:

1. The state is the first state.
2. Cut w into blocks of 32 words. There is at least one block, also if n is 0. Fill the last block
   with zero words.
3. Feed the blocks in order, each by one call of F. In the call for block j, the low 64 bits of t
   are the number of bytes fed up to and including this block: 128 (j + 1), but 4 n in the last
   call. The high 64 bits of t are num + 2^32 n, in every call. The flag is set in the last call
   alone.
4. K(num; w) is the digest of the last state.

**TK(w)**, for a list w of at least eight words:

1. Cut w into pieces of 128 words; the last can be shorter. Replace each piece by K(10; the piece),
   where n is that piece's own length. Join the digests in order.
2. While more than eight words are left, do the same to them under the number 11. A whole piece is
   then sixteen digests.
3. The eight words that are left are TK(w).

So every list is hashed at least once: TK of eight words is K(10; w), not w. The number 10 is used
for pieces of the list itself and for nothing else, and 11 for pieces of digests and for nothing
else.

*For a reader who knows BLAKE2b.* The first state, the zero padding, the one block of an empty
input, the low half of the counter and the flag are those of BLAKE2b with a digest of 64 bytes. The
one difference is the high half of the counter. BLAKE2b leaves it at 0; here it holds num (low word)
and n (high word) in every call. So K under any number in use is not BLAKE2b, and no library's
BLAKE2b computes it. With that half set to 0 the same steps give BLAKE2b-512 of the list's bytes: a
good first test of an implementation. A digest is a truncation of the 64-byte result. It is not
BLAKE2b with a shorter output, which starts from another state. n is the length of the whole list
and is the same in every call; only n and the counter tell a list from the same list followed by
zero words.

*The words of a text* (R15.4): its bytes, four to a word, the first byte lowest; the last word is
filled with zero bytes.

*How a digest is printed* (from the programs' source): its eight words in order, each as eight
lower-case hexadecimal digits, the most significant first: 64 digits. That is not the order in which
a BLAKE2b library prints its bytes. A digest line of the parameter file holds one word as a decimal
number (R15.2).

#### The numbers

Four numbers are in use: 10 and 11 in TK; 72 for a public table; 73 for a list of lookup calls. The
numbers reserved for this function are 3 to 16, 64 to 67, 72, 73, 75, 83, 84, 90 to 110, 112, 113,
120 to 127, 130 to 139, 240 and 241; none of these is ever given a new use (R3.2). The numbers of H
are those of R3.1, from 160 to below 240.

#### The digests

| digest | of what | function | what is done with it |
|---|---|---|---|
| of the lines `call` | the words of the text that is these lines, each whole, with its name and its line feed, in the file's order (R15.6) | K under 73 | words 29 to 36 of the item `request` (R15.5) |
| of the lines `callA` | the same for these lines | K under 73 | words 122 to 129 of the item |
| of the lines `callH` | the same for these lines, or for the six lines of R15.6 if the file has none | K under 73 | words 155 to 162 of the item |
| of the scale file | all its words | TK | words 95 to 102 of the item |
| of the request | the 2^lt + 2 words of the request file | TK | words 163 to 170 of the item |
| of the table SiLU | 65,536 words | K under 72 | compared with the lines `tsilu0` to `tsilu7`; words 71 to 78 of the item |
| of the table sigmoid | 65,536 words | K under 72 | compared with `tsig0` to `tsig7`; words 79 to 86. Without experts this table is not formed and the eight words are zeros (from the programs' source); such a model misses the standard parameter form (13.7, `routed-experts`) |
| of the rotary table | 2^lt dh words | K under 72 | compared with `trope0` to `trope7`; words 87 to 94 |
| of the softmax table | u0 words | K under 72 | compared with `texp0` to `texp7`; words 114 to 121 |

The item `request` is part of the first query of a proof's first chain (R14.3). So the nine
digests are bound by H, like the item's other words. A verifier computes each of the four tables itself, or takes them as
data (R15.3), and rejects if a digest differs from the file's lines. Why: the tables are made with
floating-point functions, and which function is proved must not depend on the last bit of a
library's result.

*The lists of the four tables* (from the programs' source; R15.6 does not define them). Each entry
is one word. round(x) rounds to the nearest integer, halves away from 0. All real arithmetic is in
IEEE 754 double precision. The formulas do not fix the tables bit for bit: the programs evaluate
them with the platform's exp, pow, cos and sin, whose last bit no standard fixes. What fixes the
tables is the 32 digest lines of the parameter file, so for a formal definition the four tables are
data (17.1, item 8; R15.3).

- SiLU: for t = 0 to 65,535: 4096 + clamp(round(64 (z / (1 + exp(-z)))), -smax, smax), with z = c /
  2^eg and c = t - 32,768 clamped to -64 2^eg .. 64 2^eg - 1. eg and smax are the optional lines of
  these names; without the lines they are 8 and 4095.
- Sigmoid: let sig(t) = 16 round(4096 / (1 + exp(-(t - 32,768) / 256))). Without the line `gcl 1`,
  the entry for t = 0 to 65,535 is sig(t). With it, the entry is sig(t + 16,384) for t below 32,768,
  and 128 + clamp(t - 49,152, -127, 127) from there on.
- The rotary table (the lines `trope0` to `trope7`): first the cosines and then the sines. Within
  each, for t = 0 to 2^lt - 1 and within t for i = 0 to dh / 2 - 1: 4096 + round(2^cs f(t theta0^(-2
  i / dh))), with f the cosine or the sine. theta0 = `theta0_lo` + 2^32 `theta0_hi`.
- The softmax table (the lines `texp0` to `texp7`): for u = 0, 1, ..: the nearest integer to 2^S
  exp(-u / 2^rx), halves to the even integer. The list ends before the first u for which this is 0;
  u0 is that u.

*What a verifier rejects* (from the programs' source): a scale file of fewer than eight words; a
parameter file or layout file of 2^32 bytes or more; a request file or token file that fails test 1
or 2 of 12.2.

#### Known answers

Digests are printed as said above. "a..b" is the list of the words a, a + 1, .., b.

| what | value |
|---|---|
| F(the first state, m, 3, last), where m's first long word is 0x636261 and the others are 0: BLAKE2b-512 of "abc" (RFC 7693, Appendix A), printed byte by byte, as a library prints it | `ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923` |
| K(10; the empty list) | `642033eebd8188a6b2b6dba3ad4e81de70b03c7d9bc86eec189b6da891098fc0` |
| K(10; the one word 7) | `9199d8e76712b9d5a45ca6ffa7333389d9332b0a830f4274cc71888ff4dd0629` |
| K(10; the two words 7, 0) | `5270ece847c0c6e6215c895e003b513dc694c460d5091150f1a79692658fcb04` |
| K(3; 1..32): one full block | `3d11700a6c1b3f588c605790c666049f2a3b0711fb1cdb03fc8cba0a088f932b` |
| K(3; 1..33): a second block of one word | `63879cbc0f65a48f466bbb5507fb9c4244a169c026ffae47fb3f0fa72bd88212` |
| TK(1..8) = K(10; 1..8) | `2c9ad89066617790478f356ad1401491956327cfb43ec51194e7ad7eb36bdbba` |
| TK(1..136): pieces of 128 and 8 words, then 16 words under 11 | `b790a9d3eb4b5173034538aa7d2ddd3ab7359b67c77b274469f8de7e5f0044b2` |
| TK(1..2049): 17 pieces, then 136 words and then 16 words, both under 11 | `3323f7b96604ef89cbe4e3b4ae6641bb73ccc22b581b898b679d62882b68a13e` |
| K(73; the words of the four lines `callA KR KR RM RD`, `callA KFD KFD FM FD`, `callA R16 R16 WL IX`, `callA WH WH WH`, each with its line feed) | `5ff06a4e40d0021f4ba7515aa4d74160b82fe017a8ce80e892326b39170855ef` |
| the softmax table for S = 12, rx = 8: u0 = 2,307 words | `d97d13478f344a82f98e4e7c1f34e3e56fb160cab55e6da166e4b89809da2b38` |
| the rotary table for lt = 6, dh = 128, cs = 12, theta0 = 10,000: 8,192 words | `8517dfcc808aec6cd68d86e31db8ca99db10e1aab2c430305baa799aace5bea5` |
| the table SiLU without the lines `eg` and `smax` | `7a1813d90b78365daaa463baaa1ac68da6c3700254ae8558dbb989119a9b815d` |
| the table sigmoid with the line `gcl 1` | `752e6aec8ca4a0e885eadb45909c7352e6a98320127af997143af24e9b0e60d0` |

The four tables' rows have the numbers of the example configuration
`prover/examples/target_small_f2/policy.txt`, and their values are that file's 32 digest lines. The steps of K do not give the row of "abc", since a list of words
always has a multiple of 4 bytes: that row tests F and the first state alone. The file
`verifier/tests/hash_vectors.txt` has 57 more: 44 lines with the whole last state of K under 3, 10,
107 and 2147483651 for lists of 0 to 70,001 words, and 13 lines with TK for lists of 8 to 1,000,001
words. The parts of a line are separated by one space. A line is `H`, the number and n in decimal,
and the state's 16 words; or `TH`, n in decimal, and the digest. State and digest are printed as
said above, one word after the other with nothing between them: 128 digits or 64. The first 64
digits of a state are K's digest. In the list of n words, word k is ((k + 1) 0x89ABCDEF) mod 2^32,
for k = 0 to n - 1 (from the programs' source).

#### What this part does not specify

This part does not define what the four tables mean for the model, the order and the meaning of the
item's other words (R15.5), or the tests that a verifier makes on the public files. It makes no
statement on the security of K and TK. What a statement on soundness asks of them is told in R15.3.

## 13. What a verifier does, in order

This section says what a verifier does with the files of one call. It says which files are read, and
in which order; what is formed from the public files before a key file, a leaf file or a proof file
is looked at; where the commitment ID is compared; how the work is cut into three stages; and under
which name each test rejects. It points to sections 8 to 12 for the bytes.

Key file, leaf file, anchor, leaf table, batch, slot, proof file, segment, next token and commitment
ID are defined in section 1; chain, first query, item, plan and seal in section 4; stored table,
record and stored answer in R4.6 to R4.10; oracle message in section 6; sub-proof in sections 10
and 11. A **stage** is one of the three parts of the verifier's work. The stages have the printed
names `key`, `leaf` and `request` (13.6): the stage key checks the key file, the stage leaf checks
the leaf file, and the stage request checks the proof file. A stage of a proximity test (R12.4) is
another thing, and this section always names it together with its test. A **word** has two widths in
this section. A word of `KL.bin`, of the request file, of the token file or of the statement (the
value of the item `request`), and the one word of a digest line of the parameter file, has 32 bits.
In `KL.bin`, in the request file, in the token file and in the item it is four bytes, the lowest
first (13.3; the start of section 12; R15.1); in a digest line it is a decimal number (R15.2). So
the 13 L words of the layers' records in `KL.bin` are 52 L bytes, and a request file's T + 2 words
are 4 (T + 2) bytes. In a file's head the version word has four bytes, and so has the number k in a
proof file's head (R13.1; R14.2); in the first line of the parameter file the version word is four
characters (8.1; R4.3). Every other word is that of R1.1, of 8 bytes.

Except for the five duties in 13.1 this section has no numbered rules: it is written from the source
of `cp_verify`, the verifier among the three programs `cp_commit`, `cp_prove` and `cp_verify`, and
"the verifier" in 13.3 to 13.8 means that program. A second verifier must make the same tests on key
file, leaf file, proof file, request file and token file. It must also make the tests of 13.8 on the
three public files. The provider writes these files, and some of the tests, above all the
no-overflow conditions, are meant to make an integer relation of a sub-proof mean what it says. A
second verifier need not use the same names or the same order of tests: the order decides only under
which name a bad call is rejected.

The tests on the public files are of three kinds, which the tables of 13.8 do not separate (from the
programs' source).

- (a) Tests that give a relation its meaning or a round its error: the names with `no-wrap`, among
  them `no-wrap-pin-forest`, and `forest-keys`, which keeps a multiplicity of 32 bits wide enough
  for all keys.
- (b) Bounds on sizes for which the source gives no such reason: at most 16 calls (`forest-calls`)
  and 256 trees (`forest-trees`) in a lookup, `slots` at most 256 (`policy-leafd`; 8.3; 17.2, item
  5). A verifier that admits more than these bounds differs from `cp_verify` only on public files
  that `cp_verify` rejects. But this page says nothing on such sizes, so the duty holds for these
  bounds too (the paragraph after this list).
- (c) The ten conditions on hiding (8.4, item 23; defined in 16.1). `cp_verify` rejects a
  configuration that misses one of them (`parameters-not-admitted`; for the room in a segment's mask
  row, part (a) of condition 3, `plan-mask-row`; 16.1, "Whose duty", says under which other names a
  call can end first), so a verifier that leaves them out accepts under some configurations under
  which `cp_verify` rejects. They are a duty of the committer and the prover (section 16), not of a
  verifier. Three parts of them are asked of every verifier all the same. Part (a) of condition 3
  (11.1, the room in the mask row): a scalar whose number is beyond the mask row has no mask entry.
  Part (d) of condition 3: where it is missed, the row that is split between two groups names a free
  entry of a message that has none (13.9, the round `SPLIT`), and nothing else asks it. The fourth
  line of condition 6: the weights leave the positions of a row's pads free; it is
  also the first of the further conditions of 8.4, item 23, so a verifier that derives its numbers
  by 8.4 has asked it there. Others of the ten such a verifier has from its derivation or from the
  parser; 16.1 says of each of the ten who else asks it. A verifier that takes plan, codes and
  shapes as public inputs (8.4, the second way) has none of them from a derivation: for it parts of
  five of the ten are conditions on its inputs, two of which carry soundness. 8.4 names these parts,
  and section 18, in its item "What is then assumed of the inputs", lists them with what else is
  then asked of the inputs.

This page does not sort every name of 13.8 into these kinds, so the duty is on all of them but (c),
and on the three parts of (c) that are named there (on more of (c) for a verifier on the second way:
section 18, "What is then assumed of the inputs"). Where 13.8 names such a test without its formula,
this page does not define it, and the programs' source is the only text.

### 13.1 Five duties of every verifier, and of anyone who writes a protocol

- **R0.1** A verifier hashes the encoding of objects that it has parsed, or accepts canonical
  encodings only. The two come to the same: by section 2 every object has one encoding. The public
  files are the exception: they are hashed byte for byte as they are in the file (R4.9), and the
  layout file has more than one admitted encoding (13.4, step 7).
- **R0.2** A key file, a proof file or any other file never sets a public parameter. It can repeat
  one, and the verifier then compares it with its own.
- **R0.3** Whatever depends on the bytes of a file makes a verifier answer "rejected". It never ends
  the process, and it never leaves the answer open.
- **R0.4** Whatever a verifier's decision reads that is neither a message of the prover nor a
  message of the verifier is read out of the items of its chain's first query, or is computed from
  them. That holds as well of the rows that a caller places on a block's claims.
- **R0.5** *Every public fact has one home.* No item holds a field that follows from its other
  fields, and no two items of a first query speak of one thing: what follows is computed. Where two
  forms of one fact must both exist (a plan beside the shapes that it was planned from; what is read
  out of public files beside the digest of those files; a request file beside the digest of it in
  the item `request`: R14.3, items 3 and 4), one routine makes both from the one source,
  and a verifier takes that routine's result alone. An item that contradicts itself is not an item
  of the format, and a verifier that reads such an item from a file rejects.

### 13.2 Easy to get wrong

- The commitment ID (R4.11) is a hash of five things: the configuration's digest; the anchor's name
  as a string of bytes, that is, the word 112 and then the name's 112 bytes, which hold the
  configuration's digest a second time; the list of the elements of the record's seeds, one flat
  list, with their number as a word before them and no word of its own before any seed; the list of
  the record's stored answers, their number as a word before them; and an empty list, that is, the
  word 0. The
  configuration's digest takes the three public files in the order `policy.txt`, `layout.txt`,
  `KL.bin` (R4.9). The verifier also prints this digest as `h_par=`, written
  another way, and compares that line with nothing; nothing reads the line. The ID's 64 hex digits are the digest's 32 bytes in order, two
  digits each; `h_par=` is printed word by word, each 32-bit word with its most significant digit
  first.
- No stage passes anything to another (13.6). The stages leaf and request read the anchor's root
  from the beginning of `key.bin`, and the stage request reads the leaf table's root from the
  beginning of the leaf file. Each of them tests only the form of these bytes. The design relies on
  two things: all three stages work on the same bytes, and the call accepts only if all three pass.
  A verifier that checks a proof alone has checked nothing about the table.
- `cp_verify` always checks a proof on a leaf table, that of the leaf file `leaf/<k>.bin`, with k
  from the proof's head, also for k = 0. It never checks one on the anchor, whose root is in
  `key.bin`. This is the program's way to keep R14.2, not a rule of its own: the rule asks that the
  head's number is the one under which the verifier holds the table that it reads, and the routine
  of the library `blocks/` takes the table and the number that its caller gives it (from the
  programs' source).
- In the first query of a proof's first chain, no parameter item follows `binding tree` (R14.3). The
  sub-proofs' public data are functions of the three public files, which the item `configuration`
  binds (the four tables of 12.6 only through their digest lines; on what that rests: R15.3).
- The item `request` is not the request file (12.1). The request file is the value of the item after
  `request`, `the request file's words` (R14.3, item 4).

### 13.3 The files, in the order in which they are read

The verifier reads every file once, whole, into memory, before any stage starts. Every parse and
every hash of a file works on these bytes, so that one path cannot mean two things in one call. Only
a regular file is read; symbolic links are followed. If one of the files 1 to 7 of the table cannot
be read, the call ends there, before any stage, with exit code 2 and without a result: that is not a
rejection (13.8, after the first table). One test comes before that for files 5 and 6: their lengths
are tested before either is read (the table's rows 5 and 6), so a whole call in which one of the two
has a wrong length is rejected there, also where that file or the other cannot be read. Two cases
are outside that sentence. Files 1 to 4 are read before these lengths are tested: if one of them
cannot be read or is not a regular file, the call ends before that test, whatever the two lengths
are. And a path of file 5 or 6 that names no regular file has no test on its length: it is left to
the read, which ends the call with exit code 2 unless the other file's length was rejected before.
In the table's paths `<config folder>` is the configuration directory and `<ivk folder>` is the
public key directory; the two words are those of the program's usage text. No file holds a length, a
count or an offset: every length follows from the three public files (R0.2) and, for an opening,
from the sampled positions (R8.1).

| # | file | path | read as | tested before the stages |
|---|---|---|---|---|
| 1 | the parameter file | `<config folder>/policy.txt` | bytes | fewer than 2^32 bytes; its first line is `version`, a blank, the program's version word, a line feed |
| 2 | the scale file | `<config folder>/KL.bin` | 32-bit words | its length is a multiple of 4 bytes; at least 8 words |
| 3 | the layout file | `<config folder>/layout.txt` | bytes | if the name exists it must be readable; fewer than 2^32 bytes |
| 4 | the key file | `<ivk folder>/key.bin` | bytes | at least 8 bytes; `CPIK`; the version word |
| 5 | the request file | as named | 32-bit words | before a byte of it is read, its length as the file system gives it: 4 (T + 2) bytes (13.8, `statement-required` and `statement-range` in the table "Before the stages", which also says where the test is left out); once file 6 is read, the test `policy` (13.8) reads its length and its words r and e, with lines of the parameter file |
| 6 | the token file | as named | 32-bit words | its length as the file system gives it: 4 bytes. This is tested after the length of file 5 and before a byte of file 5 or 6 is read (13.8, the same row) |
| 7 | the proof file | as named | bytes | at least 12 bytes; `CPPF`; the version word; bytes 8 to 11 give k |
| 8 | the leaf file | `<ivk folder>/leaf/<k>.bin`, k in decimal | bytes | nothing, not even its head |

The program is built for one field and accepts one version word (R4.3). The programs of this
repository can be built for field 2 alone, so `cp_verify` accepts `H821`, and it rejects `H811`, the word of field 1. There is no test on k: a k without a leaf file ends
where the leaf file is read. The program reads no environment variable, writes no file and keeps
nothing between calls.

### 13.4 What every stage does first: the public files

Each of the three stages begins with the same steps on the three public files. Until they are done,
a stage cannot read the key file, the leaf file, the request file, the token file or the proof file.
So the number, kind and length of everything in a file is fixed before a request is known, but for
the length of an opening.

1. *The parameter file.* The frame lines, from `version` to `table`, are read by their parser
   (section 8). The other lines are admitted in one order and one encoding only: the lines from
   `prof` to `F`, in the order of R15.5; then the optional lines, each at most once, in the order
   `lidb`, `nminH` (only directly after `lidb`), `eg`, `smax`, `rcl`, `gcl`, `xl`; then the 32 lines
   of the tables' digests, all or none; then the call lines: `call`, then `callA`, then `callH`. The verifier writes each value into its own statement (R15.5) as it
   goes.
2. *The scale file.* For each of the L layers it holds a record of 13 words of 32 bits: kq, kk, kv,
   ko, kg, ku, kd, M1, K1, M2, K2, sh, rp. After all layers, if the model has routed experts, it
   holds three words kr, kgR, kuR for each layer with an odd number, counted from 0.
3. *The shape.* The numbers of the statement are tested one by one and together, among them the
   no-overflow conditions, which are meant to keep every integer relation of the sub-proofs below
   the field's size (that they do is not proved: 11.2, second half). The tables of 13.8 list these
   tests.
4. *The public tables.* The verifier computes four public tables of the model's functions (SiLU,
   sigmoid, rotary, softmax), takes K(72; table) of each (R15.3), writes the digest into the
   statement, and rejects if the parameter file's line differs (R15.6). Their lists are in 12.6. It
   computes them with the platform's floating-point functions, so the 32 lines are what fixes them
   (on what that rests: R15.3). A fifth table, the head's table of exponentials, is made by an
   integer recurrence and has no digest.
5. *The segments.* Let NG = L / 2^ll be the number of groups of layers. For g = 0, .., NG - 1 come
   the feed-forward segment G_g (11.2) and directly after it the attention segment A_g (11.3). The
   head's segment H, that of the output head (11.4), is the last. Their identifiers are 1 + g for
   G_g, 2 + NG + g for A_g and 1 + NG for H. These numbers are the item `links' identifiers`
   (R14.3): in this item's name, in the public files and in the reasons that the verifier prints
   (13.8), the word "link" names a request's segment. The item `the link` and the three round names
   with the word, `LINK`, `LINK point` and `LINK phi`, belong to the tie between the anchor and a
   leaf table (section 1; R14.11). G_g and H leave a row on the stored table, A_g leaves none, so
   the item `pending rows` is NG + 1. The last segment has one plain scalar, the next token.
6. *The arrays.* Each segment commits to arrays in **moments**. First moment: the packed arrays
   `M<i>` of the segment's own matrices (11.1 writes `M`mu; the programs call them symbol matrices),
   i being the matrix's index in the sub-proof's list, first those that no other sub-proof reads,
   then those that one does. Then one moment for each vector over E that the sub-proof has: in all
   three kinds of segment that is `U3` alone (11.1). Last moment: `ZZ` and `BZ`, the digits of the
   smallness test and their multiplicities. The layout file numbers the arrays of a moment in this
   order, from 0.
7. *The layout file.* It says into which oracle messages the arrays go and where each element is. It
   has two kinds of line, of numbers and single letters.

   A **group's line** is `link moment group class rows life`: one oracle message of that moment of
   that segment; the field `link` is the segment's number. (In this file a "group" is one oracle
   message. It is not the group of sections 8 to 10, the oracle messages that one proximity test
   checks together.) Class `N`: rows over F. Class `B`: rows over F that are kept for `life` later
   segments. Class `W`: rows over E. The first oracle message of a segment's first moment also has
   the segment's one mask row, which the file does not place. So four of the six words of a
   segment's message (R12.12) come from its group's line: the number of main rows is `rows`; the
   second word is 3 for class `W` and 1 for `N` and `B`; the moment is `moment` less 1; the life is
   `life`. The other two are not in the file: the number of mask rows is 1 for that first oracle
   message and 0 for every other, and the leaf class is 0 for all three classes (from the programs'
   source).

   A **block's line** is `link moment group array R a K e low row place br bj 0`: the elements (R +
   i, K + c), i < 2^a, c < 2^e, of that array. Element number x = c 2^a + i (`low` = `r`) or x = i
   2^e + c (`low` = `c`) is in row `row` + (x >> bj) of the group, at position `place` + (x mod
   2^bj). So the block fills a rectangle of 2^br rows and 2^bj positions of the group, and br + bj =
   a + e. Segments, moments and groups are counted from 1; arrays, rows and positions from 0. An
   array's number is its place among the arrays of its own moment, from 0 anew in each moment, in
   step 6's order, and the array's name, which is hashed (R14.5), is the one that step 6 gives it.
   The file chooses the letter `low` of each block. The letter is tested for being `r` or `c` (13.8,
   the rows `plan-layout`); by this page's reading of the source no test compares it with a letter
   that anything else prescribes (`blocks/frame/req_frame.h`, `blocks/iop/laid.h`). The choice is
   hashed: it is the ninth word of the block in the item `layout`, 1 for `r` and 0 for `c` (R14.5).
   The condition `one-placement` of the standard parameter form (13.7) compares it, like the block's
   eight other words, between the groups of layers, for the matrices that another sub-proof reads
   (`verifier/src/cp_verify.cpp`).

   A line of blanks, tabs and carriage returns only, and a line whose first byte is `#`, are
   skipped; a line that holds nothing but a form feed or a vertical tab is not skipped, and is
   rejected. A last line without a line feed is read. A block's line need not come after its group's
   line: all the groups' lines are taken first, in the file's order, and then all the blocks' lines.
   The order of the blocks' lines is the order of a segment's arrays and of an array's blocks in the
   hashed item `layout` (R14.5). All tests on the file are in 13.8, in the rows `plan-layout`.

   Unlike the parameter file, this file has more than one admitted encoding, because the programs
   read each line with the C library's `sscanf` (from the programs' source): fields are separated by
   any white space; leading zeros pass; a sign passes ("+3" is 3, "-0" is 0, and by the C library's
   rule a negative number is taken modulo 2^64); the number of digits is not bounded, and what a
   number that does not fit 64 bits gives is left to the C library; no white space is needed around
   the letter ("1 1 1N5 0" is a group's line); and what comes after a zero byte in a line is not
   read. A group's line to which no block's line belongs is admitted: its oracle message then holds
   no array. A second verifier that admits the plain encoding alone rejects some files that
   `cp_verify` admits (17.1, item 9).

   The layout file's bytes enter the configuration's digest as they are, so two encodings of one
   layout give two commitment IDs; the parsed layout is encoded again, canonically, as the item
   `layout` (R14.5). In that item the arrays of a segment come in the order in which their first
   block's lines come in the file.
8. *Sizes and plans.* One routine takes the bytes of the three files, in the order of R4.9, and the
   sub-proofs' shapes, and derives every size and the plan of every round of key file, leaf file and
   proof file (section 8). The stages take its result alone (R0.5).

### 13.5 The commitment ID

After these steps each stage reads the beginning of `key.bin` for its form. That is the head, the
root (32 bytes) and the record's stored answers (R13.2). Each of the table's 2^v rows (v: R4.10) has
the same number of stored answers, and each answer is a canonical element of E. From the root, its own configuration's digest and its own shape of the table
it forms the anchor's name (R4.10). From the name it samples the record's seeds (R4.8), which are in
no file. From its configuration's digest, the name, the seeds and
the stored answers it forms the commitment ID (R4.11). It compares the ID, as 64 lower-case hex
digits, with the one that the caller has named, in which capital letters count as small ones (14.2),
and rejects if they differ (`model-statement`). This comes before the stage looks at anything else
of key file, leaf file, request or proof file, and it is the only test of a stage in which the
caller's ID meets the files. (The parent process of `cp_verify` compares the ID that the stages have
formed with the caller's once more, as a guard on itself: `internal-id-two-ways`, 14.3.) Nothing of
the key file's proofs, of a leaf file, of the request or of the proof file enters the ID. Without
the option `-id` nothing is compared, every other test of the three stages is made, and the call
cannot end with an acceptance (13.7, condition 1). A whole call without it ends at best with
`NOT-A-VERIFICATION` and, like every call that ends so, before the parent process's four last guards
on the program's own lines on proof of work and on the policy: two under `internal-sums-line`,
`internal-policy-not-given` and `internal-policy-kind` (13.8; 14.3, the row `NOT CHECKED`; 14.4 says
in which call a part of one is made earlier; from the programs' source:
`verifier/src/cp_verify.cpp`).

### 13.6 The three stages

The verifier runs three stages side by side, each in a process of its own, on the same bytes. A
second verifier can run them one after the other, in any order.

| stage | reads | steps after 13.4 and 13.5 |
|---|---|---|
| key | `key.bin` | the error accounting of the key file's plan; conditions 2, 4 and 13 of 13.7; every test on the whole file (section 9) |
| leaf | `key.bin`, its beginning only; the leaf file | the error accounting of the leaf file's plan; conditions 2, 4 and 13 of 13.7; every test on the whole leaf file, against the anchor's root and stored answers (section 9) |
| request | `key.bin`, its beginning only; the leaf file, its beginning only; the request file; the token file; the proof file | the tests on the caller's two files (12.2); the digest of the request, which completes the statement of 12.1, the item `request` (R15.6; the request file itself is the item after it: R14.3, item 4); the beginning of the leaf file, for its form: head, root, stored answers (R13.3); the error accounting of the proof's plan; conditions 2 and 7 to 13 of 13.7; every test on the whole proof |

The **error accounting** of a file's plan: every round's error, proof of work counted, must be at
most 2^-sec, and sec must be at least the requested bits, the number after `-bits`
(`security-bits`). It reads no byte of key file, leaf file or proof file. Its figures are for that
one file's proof, and for one request's, not for a batch's (8.4, item 22). It counts, source by
source, on what the closeness terms of the plan's rounds rest (8.4, item 6; 14.7). It also notes
whether a round needs its proof of work (8.4, item 22). In a call without the option
`-no-proof-of-work` that changes neither the last line nor the exit code of `cp_verify`: it changes
a line of its own near the end of the output, `proof of work: ..` (14.3). With the option a call in
which some round does so ends at best with `NOT-A-VERIFICATION` (14.4, cause 4).

The accounting is over the rounds of the plans. A test that has no round of its own has no term of
its own in it. Four tests are of this kind. Each is a statement on every position, or on every head
and position, and is tested at one sampled point: with routed experts, that the selector's entries
sum to 1 at every token (11.2); that attention's denominator is the sum of the weights (11.3); and
the head's equations (B4) and (Z), on every position with a cost (11.4). Of all four, comments in
the source say that they are carried by the rounds that sample the point's coordinates, those of the
lookup's last key level, and that a round's error is the largest of the errors of the false
relations that it would have to rescue at once, not their sum (`prover/prover/plan_rounds.h`,
`prover/attn/plan_rounds.h`, `prover/head/plan_rounds.h`). Each of the three comments sets a limit
beside that rule: the largest is taken only across false relations that must all be rescued. Inside
one relation's own chain the errors add wherever one message of the verifier gives that relation
several ways out: a point of n coordinates that is sampled in one message costs n / |E|, batching by
powers costs the number of rows less one, divided by |E|, and where either all positions miss or a
batching challenge cancels, both terms are counted. By the same comments the rule needs three
conditions, each of which the comments argue and do not prove; the first of the three comments adds
that no code checks them; the second says that it argues them from a reading of the code, and calls
informal its argument that the rounds of the lookup's last key level bear the error of an event that
has no round of its own. This page does not say whether some round's error covers the chance that
such a statement holds at the point and fails at a position (17.1, item 3).

The proof is read by one reader from its first byte to its last, in this order (R14.1):

1. the head, 12 bytes; the number in it must be the k of the leaf file whose table is used;
2. the first chain. Its first query has the chain's label 0x46524d01 and the items of R14.3; the
   item `table` holds the root and the stored answers of the leaf table, the item `stored copy`
   holds k, the leaf table's number, the item `request` is that of section 12, and the item
   `the request file's words` holds the request file itself, whole (R14.3, item 4). Then come the
   segments, each with its sub-proof (sections 10 and 11), to the end of their last proximity test;
3. the comparison of the next token (12.4);
4. the rows that the segments leave on the stored table, joined into one; the request's own vector;
   the first chain's seal;
5. the request's slot in its batch, which with the seal gives the binding tree's root (R14.7);
6. the shared chain (R14.9): the test on the leaf table, its proximity tests, its seal;
7. the second chain (R14.10), whose statement is the two seals; its proximity tests;
8. the end of the file: no byte is missing and none is left over.

### 13.7 From the stages to the result

The call rejects if any stage rejects. Its rejection name is that of the first stage that rejected,
in the order key, leaf, request. It accepts only if all three stages have passed, have formed the
same commitment ID, and that ID is the caller's, and only if the conditions below hold. Section 14
has the results of `cp_verify` and the conditions that are the command's own.

*What an acceptance requires* (from the programs' source). That every test passes is not enough. The
person who writes the parameter file is not the one who relies on a proof. So wherever a line lets
the writer choose something laxer than what an acceptance is meant to say, a condition is required.
The source names the lines `work ..` as an exception, and says that nobody has shown that these are
all such lines (`verifier/src/vf_common.h`; 17.5, item 10). These conditions bind every verifier, like
the tests of this section: they are part of what an acceptance means (section 15). A missed
condition says nothing against the proof. `cp_verify` makes every test on the files all the same,
but not the parent process's last guards on itself (14.4 has the exceptions to both; 14.3, the row
`NOT CHECKED`; 14.8), and, if none fails, ends with `NOT-A-VERIFICATION` and the name in brackets
(14.4), but for conditions 3, 5, 6, 11 and 14, where it rejects; a verifier with two answers rejects. A
verifier gives an acceptance only if:

1. *The commitment is named.* The caller has named a commitment ID, and it is the ID that the stages
   form (13.5) (`id-named`). Why: an acceptance about a commitment that nobody has named would say
   nothing.
2. *The target of 128 bits.* The error accounting of each of the three files' plans (13.6) is made
   with at least 128 requested bits. So the line `sec` is at least 128, and every round's error, by
   the programs' count and with proof of work counted, is at most 2^-128. The unit of that count is
   one key file's, one leaf file's and one request's proof, not a batch's: counted jointly for a
   full batch of N requests, each round of a request's own chains has log2(N) bits less (8.4, item
   22, has the line that the programs print). Because proof of work is counted, an acceptance does
   not mean that every round meets the target without its proof of work: up to 32 of a round's 128
   bits can be proof of work (condition 5). Neither the last line nor the exit code of `cp_verify`
   shows how many are; its line `proof of work: ..` does (14.3). A caller of `cp_verify` who wants
   the tests alone to carry the target passes the option `-no-proof-of-work`, with which no call in
   which a round needs its proof of work ends with an acceptance (14.4, cause 4). (`cp_verify` has
   no name for this condition: it prints that fewer than 128 bits were requested.) More than 128
   bits cannot be requested either: above 128, half a digest's bits, `cp_verify` rejects the call
   before it reads a file (`cp-bits-above-digest`; 13.8; 14.2). So 128 is the only number of
   requested bits under which a call can end with an acceptance.
3. *How a round's error is counted: the reading.* The parser takes the line `reading` at 1 alone:
   the theorem's integer is read as the theorem is printed (section 8).
4. *How a round's error is counted: the key file and the leaf file.* Every closeness term in the plan
   of the key file, and of the leaf file, rests on the bound that holds within a third of a code's
   distance or on the theorem on
   codes (Theorem 4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon,
   Haböck, Kopparty and Saraf published at STOC 2026), which the programs' accounting takes as given
   (8.4, item 6). That holds by the derivation: no verifier tests it. The stored table's distances are made from an m, so the closeness terms of its own
   rounds rest on the theorem (8.4, items 7 and 16). A statement on codes held to follow from that theorem, in its case of two words, is proved
   in this repository's Lean library, by way of another formalization's theorem and a comparison of bounds; the paper's
   own form of the bound is not proved there, and that the library's statement follows from the paper's,
   and that these programs' parameters meet the theorem's conditions, rest on reading, not on a proof (lean/LIMITS.md, L15). So an
   acceptance rests closeness terms of the key file and the leaf
   file, and with them the binding of the weights, on the theorem on codes. The lines
   `accounting: one-time files: ..` and `commitment: ..`
   (14.7) say how many. What else an acceptance rests on:
   17.5, item 19.
5. *How a round's error is counted: proof of work.* Each of the fifteen numbers of the lines
   `work ..` is at most 32 (`policy-work-most`). Here `cp_verify` does not wait for the end: it
   rejects a parameter file with a larger number (13.8). Why: the parser of the frame lines admits
   up to 58 bits (8.1). With more than 32, every round's error, proof of work counted, would be at
   most 2^-sec all the same, so the cap closes no way to a false proof; but without it up to 58 of
   the 128 bits of an acceptance could be proof of work.
6. *The whole model.* `prof` is 20: the programs refuse any other value (13.8).
7. *The model's input.* `toksrc` is 1: the model's input is the embedding of the request's tokens
   (`prompt-source`; a file with another value is rejected earlier).
8. *The clamps.* The lines `rcl 1` and `gcl 1` are both there, so that the router's clamp and the
   gate's clamp are part of what is proved (`both-clamps`).
9. *The norms.* KF <= MF, and K1 <= M1 and K2 <= M2 in every layer's record of `KL.bin`
   (`norm-shifts`). Why: a norm multiplies by a reciprocal that is scaled by 2^M and then shifts
   right by K bits. The tests of 13.8 bound M - K from above and not from below, and from K = M + 3
   on the norm's output is 0 for every input. The condition excludes that; a small M - K leaves
   little of the output all the same.
10. *The public tables.* The 32 lines of the tables' digests are there (`table-digests`). Why: four
    public tables are computed in floating point, and only these lines fix them (13.4, step 4; on
    what that rests: R15.3).
11. *The acceptance policy.* Four things are required. (a) sT = sh - 4, so that the policy's kind is
    `exact-match` or `clipped-gap` (12.5), whether or not a token of the request is checked. (b) If
    a token of the request is checked (r < T - 1), the kind is `exact-match`, or else the caller has
    said which kind and which five numbers C, theta, cap, s and sh it expects. (c) Where the caller
    has named a kind and numbers, they are the parameter file's, also if no token is checked. (d)
    The numbers that the caller has named exclude something: C and theta are at least 2^sh, theta is
    at most C, and theta = C does not come with 0 < n <= cap, where n is the number of checked
    tokens. Here too `cp_verify` does not wait for the end: a whole call that misses one of the four
    ends with the rejection `NO: policy` before the stages (14.3). The stage request asks (b) and
    (c) once more (`exact-match`, `policy-named`; 14.4). Why: under other values of these lines an
    acceptance can say less, down to nothing (12.5). Where a token of the request was checked under
    `clipped-gap`, `cp_verify` gives the acceptance a last line of its own, `WITHIN LIMITS: ..`, and
    not `YES` (14.3).
12. *The one form of the parameters.* The other lines that give a choice have the values of the
    table after this list.
13. *The form of the test on the table: two codes.* The folded row and Omega are encoded with two
    different codes, in a batch's test on the leaf table and in the key file's test on the anchor:
    the lines `group folded` and `group omega` differ in code length or in message length, and so do
    the lines `group once-folded` and `group once-omega` (`two-codes`; asked in all three stages;
    section 8). Why: where such a pair has the same code length and message length, the programs
    test both in one group and leave out the step that splits a row between two groups. By the
    programs' source the bounds that it calls argued on paper for this protocol are for the test
    with two codes and the split, and do not cover the other form. The arguments are the written
    proofs in this protocol's technical report. The proofs are about the protocol as the
    report describes it, not about these programs ([limits](../limits.md), the start of "Soundness",
    "What the bounds rest on depends on the parameter file" and "No theorem is about `cp_commit`,
    `cp_prove` or `cp_verify`"). By the programs' source no attack on the other form is known, and
    nobody has searched for one.
14. *The table's record: one answer.* The parser takes the line `answers` at 1 alone
    (section 8). Why: the stored table has an
    out-of-domain sample of its own, and the line `answers` sets the number
    of answers to every out-of-domain sample, this one too. By the programs' source the same bounds
    (condition 13 says what they are) cover such a table only where its own out-of-domain sample has
    one answer. By the same source no attack on a table with more is known, and nobody has searched
    for one.

| name in `cp_verify` | what must hold |
|---|---|
| `routed-experts` | NE is above 0 |
| `five-limbs` | the line `xl 5` is there |
| `placement-table` | the configuration directory has a layout file (without one the call is rejected earlier) |
| `no-range-words` | neither the line `eg` nor the line `smax` is there |
| `three-bytes` | S is below 32 (a file with a larger value is rejected earlier) |
| `ring-degree` | D is 2048 |
| `one-shift` | all layers of a group of layers have the same word sh in `KL.bin` |
| `one-placement` | the layout file places the matrices that another sub-proof reads in the same way in all groups of layers, number by number, the letter of each block's line among them (13.4, step 7) |
| `softmax-table` | S is 12 and rx is 8 |
| `rotary-table` | theta0 is 10000 or 500000, dh is at most 128, lt is at most 10 |

The ten rows of this table are not the ten conditions on hiding (`main-rows` to `batch`; 16.1),
which are a duty of committers and provers and, but for three parts of them (the start of this
section, (c)), not of a verifier.

Why condition 12: it keeps an acceptance to one form of the parameters, the form for which sections
11 and 12 are written (17.1, item 6). The source says of `softmax-table` and `rotary-table` that
they are meant to allow only values at which every entry of the table, which is computed with the
platform's functions, is far from a rounding tie, so that every faithfully rounding library gives
the same table (the programs' documentation of the parameter file). For the values that the two
conditions allow, no entry changes if all results of the platform's library are moved by 1, 16 or
1,024 units in the last place, given binary64 arithmetic in the stated order. That has been computed
for every entry, by a test of this repository. That no smaller error changes an entry is argued in
that test and not run, and nothing of this is proved; by that test's own text some entries differ
with 80-bit intermediate results, by a computation that is not in the test
(`scales/policy_tables.py`; `scales/tests/table_ties.py`). For the other rows of the table this page
has no reason beyond the first.

Conditions 7 to 13 are the **standard parameter form**; of condition 11, its parts (b) and
(c). All of them but 11 read the three public files alone. The stage request asks 7 to 13; the
stages key and leaf ask 13. The lines of conditions 3, 6 and 14 the parser holds at one value. Every stage also asks condition 2, which is not part of
the form.

By this page's reading of the source, none of the ten conditions on hiding (16.1) is part of what an
acceptance means: none is a hypothesis of an error that a plan counts. By their nature this is so of
`main-rows`, `mask-rows`, `table`, `range-checks`, `salt` and parts (a) and (b) of `leaves`. Of the
others it holds only because each is asked or met at another place on these pages, which 16.1 names
for each ("Who else asks each of the ten"); part (d) of `mask-room` alone is asked nowhere else, and
every verifier asks it as 16.1 has it. The weightiest is the fourth line of `padding-entries`, which
is also the first of the further conditions of 8.4, item 23: a link compares the
plain entries alone (R13.8; under R13.10), so what an acceptance says of a leaf table's weights
needs that none of them is at a pad's position. `batch` is the only one of the ten that fixes a number which is a factor of a round's error: N,
the number of requests for which the test on the table is sized. Others read numbers that are in a
round's error too: a round's number of positions, the number of answers to an out-of-domain sample,
N again, the number of combinations of a smallness test, and the value of `sec`. They bound them, if
at all, from above; of the number of combinations, `range-checks` also asks that it is at least 1.
The tests on the form of the plans, which the programs report under the same names (16.1), are
equalities; a plan that is derived as sections 8 to 10 say has that form. A verifier does not need
the condition `batch` to have N: it takes N from the line `slots` when it derives the plan (8.4,
items 16 and 18), and the plan's number must be the binding tree's number of leaves (from the
programs' source).

### 13.8 Every rejection by name

A rejection ends the stage, or the call, at once. The last line of the output is `NO: <name>`, and
the exit code is 1. Where a stage has rejected, the call's line before the last begins with
`refused by the stage <stage>`, after which come `: ` and the stage's reason where the call repeats
it (14.7), and the stages' own output is printed only under the option `-show-stages` (section 14).
Under `model-statement` two lines that begin with `id:` come instead, and the stages' output is not
printed at all. So "the stage's line before its last", in the tables below, is a line of that
output. The names are printed words and are not hashed. The tables below list the tests in the
verifier's source, roughly in the order of a run. "What must hold" is the condition whose failure
gives the name. In that column a bare name such as lt or NAf means the number of the parameter
file's line of that name; a name without its ending _lo or _hi, such as eps1 or theta0, means the
low line plus 2^32 times the high line; T = 2^lt; "with experts" means that NE is above 0. One name can belong to several tests. The names are listed for a reader of the
program's output.

Some sentences keep a word of the programs for an object that section 11 defines under another word.
A lookup forest is a lookup with its trees, and the depth of a tree is dT (11.1). A request matrix
is a matrix of a segment's list (11.1, the smallness test), a symbol matrix is its packed array
`M`mu, and a symbol tree is that array's tree. An expanded call is a call of an attention segment's
expanded list (11.3). A feed-forward sub-proof, an attention sub-proof and the head's sub-proof are
the sub-proofs of the three kinds of segment (13.4, step 5). A product layer is a product level
(11.1); the formulas of `shape-product-layer` are in the second halves of 11.2 and 11.4. A class of
rounding sites is the rounding sites that share one of the lines `hq` to `hr` (11.2, second half). A
weight kind is what one of the twelve lines `wb_qk` to `wb_r` belongs to; this page does not define
the kinds (17.1, item 5).

Before the stages:

| name | what must hold | on | 
|---|---|---|
| `cp-bits-above-digest` | The requested bits, the number after `-bits`, are at most 128, half a digest's bits. | `-bits` | 
| `policy` | If the call has the option `-policy`, the numbers that it names do not by themselves empty a result of its meaning: C is at least 2^sh, theta is at least 2^sh, and theta is at most C. (A token's cost is at least 2^sh.) The test reads the command line alone and is made before any file is opened. Only a whole call has the option. Under this name the line before the last is `policy:` and the reason. | `-policy` | 
| `close-list-long` | `policy.txt` has fewer than 2^32 bytes, and so has `layout.txt`. | `policy.txt`, `layout.txt` | 
| `KL-length` | The length of `KL.bin` is a multiple of 4 bytes, and it has at least 8 words. | `KL.bin` | 
| `plan-layout` | The configuration directory has a name `layout.txt`. Nothing is rejected here if it has none: every stage rejects later, under this name. | `layout.txt` | 
| `version-unknown (policy)` | `policy.txt` begins with the eight bytes `"version "`, four characters and a line feed, and the four characters are the program's version word (R4.3). | `policy.txt` | 
| `cp-key-length` | `key.bin` has at least 8 bytes, its head. | `key.bin` | 
| `cp-key-head` | The first four bytes of `key.bin` are `CPIK`, and the next four are the version word of `policy.txt` (R13.1). | `key.bin` | 
| `statement-required`, `statement-range` | In a whole call, before a byte of the request file or of the token file is read: the request file has 4 (T + 2) bytes and the token file has 4 bytes, by the length that the file system gives for the path (the system call `stat`, which follows symbolic links). The request file's length is tested first, then the token file's, and only then are the two files read. A length of 0, or one that is not a multiple of 4 bytes, gives `statement-required`; any other wrong length gives `statement-range`. The test on the request file is left out if the line lt cannot be read from the parameter file or is above 20; by the source every stage rejects such a call. The test on either file is left out if its path names no regular file: the read that follows ends the call (after this table). Why: a file of many gigabytes that is named instead of either is rejected and not loaded into memory. These are tests 1 and 2 of 12.2, made on the lengths alone; the stage request makes both again, under the same names, on what was read. Where both lengths are wrong, the name here can differ from the one that the order of 12.2 gives: here both tests are made on the request file before the token file is looked at, and 12.2 makes test 1 on both files first. So a whole call whose request file or token file has a wrong length ends here, under one of these two names, whatever a later test or a stage would have found in the other files. A planning call does not make this test: it reads both files whole, and tests 1 and 2 of 12.2 follow. | request file, token file; `policy.txt` | 
| `policy` | In a whole call, once the request file and the token file are read and before the proof file is opened, four tests, in this order. (a) In the parameter file sT = sh - 4: the policy's kind is `exact-match` or `clipped-gap` (12.5). This is asked with and without `-policy`, and also where no token of the request is checked. (b) If the call has `-policy`, the kind and the five numbers C, theta, cap, s and sh that it names are the parameter file's. (c) If the call has no `-policy` and a token of the request is checked (r < T - 1), the parameter file's policy is `exact-match`. (d) If the call has `-policy`, it is not the case that theta = C and 0 < n <= cap, where n = min(e, T - 1) - r is the number of checked tokens: under such numbers no token is ruled out. No stage has tested the files at this point. So the tests are left out if the length of the request file or of the token file is not a multiple of 4 bytes, which the row before lets through only where its own test is left out; all four are left out if one of the lines C, theta, cap, s, sh, sT cannot be read from the parameter file; and (c) and (d) are left out if the line lt cannot be read or is above 20, if the request file has not T + 2 words, or if r < e <= T fails. By the source a stage rejects such a call. A part-call and a planning call do not make these tests. The stage request asks (b) and (c) once more, as the conditions `policy-named` and `exact-match`; they are parts (c) and (b) of condition 11 in 13.7. | `-policy`; `policy.txt`; request file | 
| `cp-proof-length` | The proof has at least 12 bytes, its head. | proof file | 
| `cp-proof-head` | The first four bytes of the proof are `CPPF`, and the next four are the version word of `policy.txt` (R14.2). | proof file | 
| `cp-entry-missing` | The leaf file `leaf/<k>.bin`, with k from the proof's head in decimal, is a regular file that can be read. There is no other test on k. The name belongs to a whole call alone: there k is the proof's word and not the caller's, so a missing file is a rejection. In the part-call `-leaf` the caller names the leaf file, and a file that cannot be read ends the call like any other unreadable input (after this table). | public key directory; proof file | 

An input file that cannot be read gives no rejection (from the programs' source). If `policy.txt`,
`KL.bin`, `layout.txt` (where the configuration directory has that name), `key.bin`, the request
file, the token file or the proof file is missing, is not a regular file, or cannot be read to its
end, the call ends at that point, before any stage. It prints the line
`input-unreadable-<role>: <the file> cannot be read: <reason>` on standard output; there is no line
`NO:` and no result; the exit code is 2, that of a bad command line (14.3). `<role>` is `policy`,
`KL`, `layout`, `key`, `request`, `y` or `proof`, and `entry` for the leaf file that the caller of
the part-call `-leaf` names. The source's reason: no stage has started, so this is not a rejection
of a proof; a file that can be read and has a wrong length or content is rejected where it is
tested. The leaf file of a whole call is the one exception (`cp-entry-missing`).

On the public files, in every stage. All three stages make these tests on the same bytes, so a call
with such a fault is rejected by the stage key, the first in the order:

| name | what must hold | on | 
|---|---|---|
| `parameter-<name> (policy)` | The frame lines, from `version` to `table`, are admitted by their parser (section 8). The name is the parser's. | `policy.txt` | 
| `statement-required` | The line `prof` is there with a value from 16 to 31, and the line `toksrc` is there with the value 1. Here and in the next row a line is there if some line begins with the name and one blank, and the first such line has one to ten digits and nothing else up to its end. So a line `prof` or `toksrc` with another byte in its number gives this name. | `policy.txt` | 
| `policy-name-lt` | The line `lt` is there. | `policy.txt` | 
| `shape-words` | lt is from 4 to 16. | `policy.txt` | 
| `header-version` | The version word in the statement is the program's. It cannot fail after the test on the first line. | `policy.txt` | 
| `policy-line-long` | No line after the frame lines has more than 1023 bytes, its line feed counted. | `policy.txt` | 
| `policy-zero-byte` | No line holds a zero byte. | `policy.txt` | 
| `policy-line-<i>` | The i-th name of the list in R15.5 (counted from 2 for prof): its line is a name and a number below 2^32, and nothing follows. i counts names, not the words of R15.5's left column: la is 29 and MF is 74. | `policy.txt` | 
| `policy-name-<name>` | Each line from prof to F has the name that is due in the order of R15.5. The rejection ends with the name that was due. | `policy.txt` | 
| `policy-line-form` | Every line with one number is encoded in one way: a name of printable bytes without a blank, one blank, a decimal numeral without sign or leading zero, at most ten digits, below 2^32, one line feed. The same name is given, before the names of the two rows above and of the optional lines, to a line that has more than 19 digits in a row anywhere. | `policy.txt` | 
| `parameter-table-shape (policy)` | i and m of the line `table` are below 2^32. It cannot fail on a file that the parser admits: there both are below 2^n, and n is at most 32 (8.1, tests 36 and 38). (The parser's own test of the line `table` prints the same name: section 8.) | `policy.txt` | 
| `policy-work-most` | None of the proof-of-work bits in the lines `work` is above 32. | `policy.txt` | 
| `policy-leafd` | slots is at most 256: the binding tree has depth at most 8. | `policy.txt` | 
| `policy-cert-words` | There is no line `clt`, `clns` or `cms`. | `policy.txt` | 
| `policy-lidb` | The optional line `lidb` is the first of the optional lines, occurs once, and is above 22, the number that holds where the line is absent. | `policy.txt` | 
| `plan-identity-cap` | lidb is at most 30. By the programs' documentation `cp_prove` rejects, under the same name, a file whose lidb is above its build's EIDB, which is 28 in the default build (`prover/build.sh`), and `cp_commit` makes no such test, so it can commit under a file that `cp_prove` then rejects (`verifier/docs/parameter-file.md`, the line `lidb`). | `policy.txt` | 
| `policy-nminH` | The optional line `nminH` comes directly after `lidb`, occurs once, is from 2 to 28, and is above nmin. | `policy.txt` | 
| `policy-gate-table` | The optional lines `eg` (4 to 7) and then `smax` (2047, 1023, 511 or 255) occur once each, in their position (13.4). | `policy.txt` | 
| `policy-rcl` | The optional line `rcl` has the value 1, occurs once, and is in its position (13.4). | `policy.txt` | 
| `policy-gcl` | The optional line `gcl` has the value 1, occurs once, and is in its position (13.4). | `policy.txt` | 
| `policy-xl` | The optional line `xl` has the value 5, occurs once, and is in its position (13.4). | `policy.txt` | 
| `policy-tables` | The 32 lines `tsilu0` to `texp7` are all there or none is; they are in their position (13.4), before the first call line; they follow one another directly, in the order of R15.2; each is one word; and they are there only with NE above 0. | `policy.txt` | 
| `policy-call-line` | Every other line is a call line: it ends with a line feed; its tokens are separated by single blanks, none is empty, and each consists of the bytes A-Z a-z 0-9 . _ - only; its second token, the call's name, has at most 8 characters; it has at least four tokens; the first is call, callA or callH; and the three kinds come in this order. | `policy.txt` | 
| `policy-length` | All lines from prof to F are there. | `policy.txt` | 
| `hid-tau` | tau is a multiple of 32 from 32 to 256, and tau less the largest power of two that is not above it is 0 or a power of two. | `policy.txt` | 
| `hid-hb` | sec is from 1 to 4096. It cannot fail: the parser of the frame lines admits sec from 1 to 256 (section 8). The name has a second test, where an attention sub-proof is planned: sec less the first number of the line `work positions` is not 0. It cannot fail either: the parser rejects proof-of-work bits that are not below sec. | `policy.txt` | 
| `policy-tau-hb` | tau is at least sec. | `policy.txt` | 
| `hid-planes-<call>` | A constant of every lookup call is 5. It cannot fail. | `policy.txt` | 
| `statement-length`, `header-magic`, `policy-profile` | prof is from 16 to 31 (`policy-profile`), which cannot fail after `statement-required`. The other two names are tests of the verifier on the statement that it has built itself. | `policy.txt` | 
| `policy-profile` | prof is 20. | `policy.txt` | 
| `shape-words` | lt is from 4 to 16; ll from 1 to 6; d and f from 1 to 2^20; with experts fe is a power of two, f = 2 fe, and not both ll = 1 and fe = 1. | `policy.txt` | 
| `shape-L` | L is not 0 and is a multiple of 2^ll. | `policy.txt` | 
| `shape-groups` | 2 + 2 L / 2^ll is below 2^16. | `policy.txt` | 
| `plan-groups-base` | There are at most 28 groups of layers: 2 + 2 L / 2^ll is at most 59. | `policy.txt` | 
| `shape-la` | lt < la <= 16; 2^(la - lt) divides d, f, fe (with experts), G dh and (H + G) dh; lmq >= la; lmaf >= la. | `policy.txt` | 
| `policy-norm` | eps1 and eps2 are from 1 to below 2^59. | `policy.txt` | 
| `shape-heads` | H, G, dh not 0; dh a power of two, at least 2; H dh = d; H + G <= 64. | `policy.txt` | 
| `policy-cs` | cs is 12. | `policy.txt` | 
| `header-absent-class` | Without experts hr is 0. (The line of a class of rounding sites that this form of the model does not have must be 0; the classes of the other six lines, hq, hv, ho, hgD, huD, hdD, are always present.) | `policy.txt` | 
| `policy-forms` | hq, hv, ho, hgD, huD, hdD, and with experts hr, are at most 2. | `policy.txt` | 
| `policy-ut` | ut is 1. | `policy.txt` | 
| `shape-D` | D is 2048 or 32. (The name has two tests: D is a power of two from 2 to 4096, where the sub-proofs' layout is derived from the statement; and D is 2048 or 32, where the layout of the committed matrix is made, 11.5. Every stage reaches both.) | `policy.txt` | 
| `shape-NE` | With experts NE is a power of two from 2 to 4096. | `policy.txt` | 
| `policy-gcl` | gcl is there only with experts. | `policy.txt` | 
| `policy-rcl` | rcl is there only with experts. | `policy.txt` | 
| `shape-request-matrix` | lmaf, lmad, lmar, lmq are from lt to 24; NAf, NAd from 1 to 64; lmx from la to 24. | `policy.txt` | 
| `plan-tensor-layout-fill-<arr>` | The tensors of a request matrix cover each of its features exactly once, and 2^(la - lt) divides the features of each tensor. `<arr>` is the matrix: stream or qa. It cannot fail after `shape-la`, `shape-heads` and `shape-words`: the tensors' first features are the program's own running sums, and the features of each tensor are d, H dh or G dh. `shape-words` and `shape-heads` keep these from 0; 2^(la - lt) divides d and G dh by `shape-la`, and H dh = d by `shape-heads`. | `policy.txt` | 
| `policy-call-list` | The lines `call`: at most 8; each has a name, a kind and at least one call's segment; no two have one name; none is named V, a, b or E; each call's segment that is named is present in this form of the model, has the call's kind, and is in exactly one call; every call's segment that is present is named. | `policy.txt` | 
| `shape-lmb` | lmb is at most 24. | `policy.txt` | 
| `shape-words` | nmin is at most 28. | `policy.txt` | 
| `shape-call-<call>` | For each call q of the lines `call`, with 2^n_q keys: iota_q is below 2^31; n_q is at least 10, at least la, and at most 44; its columns are at least 2^(la - lt); and iota_q equals n_q - dT. dT, the depth of a tree of the lookup (11.1), is the least of lidb (22 if the line is absent) and all n_q. That n_q is at least nmin is tested too and cannot fail: nmin is a floor where n_q is made (11.2). The name `hid-call-floor` belongs to repeats of the test n_q >= 10, here and in every lookup; none of them can fail. | `policy.txt` | 
| `forest-depth-G` | The depth of a tree of a feed-forward segment's lookup forest is at least la. | `policy.txt` | 
| `header-absent-call` | iota_q is 0 for every q from the number of calls up to 7. | `policy.txt` | 
| `KL-length` | `KL.bin` has exactly 13 L words and, with experts, 3 more for every second layer (13.4). | `KL.bin` | 
| `KL-shift-range` | Each shift kg, ku, kd, kq, kk, kv, ko of a layer, and kr, kgR, kuR of a layer with experts, is from 1 to 16, 24 or 32, as the line of its class (hq to hr) is 0, 1 or 2. Which line belongs to which shift is in 11.2, second half (a rounding site). | `KL.bin` | 
| `KL-kq-kk` | kq and kk are equal, or both at least 16. | `KL.bin` | 
| `KL-norm` | For both norms of a layer: 24 <= M <= 47, 17 <= K <= 32, K + 14 >= M. | `KL.bin` | 
| `KL-rp` | rp is 0 or 1. | `KL.bin` | 
| `table-digest-silu`, `table-digest-sigmoid` | The SiLU table and the sigmoid table that the verifier has computed have the digests K(72; table) of the lines `tsilu0` to `tsilu7` and `tsig0` to `tsig7` (R15.6), where the file has these lines. | `policy.txt` | 
| `header-absent-hash` | Without experts the position of the sigmoid table's digest holds zeros. It cannot fail: the tables' lines are kept apart from the statement, whose words at this position are the program's own zeros. | `policy.txt` | 
| `no-wrap-<name>` | The feed-forward segments' no-overflow conditions, computed in integers of 128 bits from the lines and from `KL.bin`. Most of them say that a sum of bounds stays below 2^60 or a smaller power of two; some say at most 2^60, and one compares two sums. `<name>` is one of U-`<site>`, W-`<site>` (sites gate, up, norm, down, qk, o, v, lr, xr), guard-radix, guard-radix-FS, guard-radix-clamp, guard-radix-gate-clamp, guard-flag-gate-clamp, clamp-keys, MR, one-hot, linear, ledger, ledger-eps. The single conditions are in 11.2, second half (the no-overflow conditions). | `policy.txt`, `KL.bin` | 
| `header-absent-kind` | The line `wb_..` of a weight kind that this form of the model does not have is 0. | `policy.txt` | 
| `policy-weight-kind` | Each of the twelve lines `wb_qk` to `wb_r` whose weight kind is present, and wb_head, is 13; wb_emb is from 2 to 31. | `policy.txt` | 
| `shape-V` | V is from 2 to 2^24. | `policy.txt` | 
| `policy-weight-width-emb` | wb_emb is 21. | `policy.txt` | 
| `hid-vocab-cols` | The number of bits of the vocabulary's index, the least c with 2^c >= V, is at most w, the last number of the line `table`. | `policy.txt` | 
| `w-list-token-factor` | The vocabulary's index fits into the column variables of the committed matrix, for the shape and for each unit of the embedding and the head. It cannot fail after `shape-V` and `hid-vocab-cols`. | `policy.txt` | 
| `shape-words` | Where the layout of the committed matrix is made (11.5): L is from 1 to 4096; d, dh, G dh, H dh and f are from 1 to 2^24; with experts fe and NE likewise; V is from 1 to 2^24. By this page's reading of the source none of these can fail at this point: all follow from the tests above, the bound on G dh not from the rows on shapes alone but through `no-wrap-ledger` (the row on the feed-forward segments' no-overflow conditions, above; 11.2). `forest-keys`, in a later table, bars it too. The names `w-shape-lk` and `w-shape-mW`, in a later table, are raised at this point too. | `policy.txt` | 
| `shape-theta0` | theta0 is not 0. | `policy.txt` | 
| `table-digest-rope` | The rotary table that the verifier has computed has the digest of the lines `trope0` to `trope7` (R15.6), where the file has these lines. | `policy.txt` | 
| `policy-norm-F` | The final norm: 24 <= MF <= 47; 17 <= KF <= 32; KF + 14 >= MF; 1 <= epsF < 2^59. | `policy.txt` | 
| `policy-toksrc` | toksrc is 1. It cannot fail after the test under `statement-required`. | `policy.txt` | 
| `policy-gap-words` | The tests on the acceptance policy's lines (12.5). | `policy.txt` | 
| `shape-request-matrix-H` | la <= lmhf <= 24; 1 <= NHf <= 64; lt <= lmhd <= 24; lt <= lmhr <= 24; lmbH <= 24. | `policy.txt` | 
| `policy-call-list-H` | The lines `callH` or, if the file has none, the six lines of R15.6: at most 6; each has a name, a kind and at least one call's segment, which cannot fail after the four tokens of `policy-call-line`; no two have one name; each of the nine call's segments NF NFH HD0 HD1 ROW HD2 CNF HE0 HE1 is in exactly one call of its kind; no other call's segment is named. | `policy.txt` | 
| `shape-call-H-<call>` | For each call q of the lines `callH`, with 2^n_q keys: n_q is at least 10, at least la, and at most 44; its columns are at least 2^(la - lt); and iotaH_q equals n_q less the depth of a tree of the head's lookup (11.1). That n_q is at least nminH (nmin if that line is absent) is tested too and cannot fail: it is a floor where n_q is made. | `policy.txt` | 
| `forest-depth-H` | The depth of a tree of the head's lookup forest is at least la. | `policy.txt` | 
| `header-absent-call-H` | iotaH_q is 0 for every q from the number of calls up to 5. | `policy.txt` | 
| `no-wrap-H-<name>` | The head's no-overflow conditions, of the same kinds. `<name>` is one of U-NF, W-NF, W-HD0, three-digits, HE, Z, E2, E3, E4, J-E, guard-radix. The single conditions are in 11.4, second half (the no-overflow conditions). | `policy.txt` | 
| `policy-call-list-A` | The lines `callA`: at least one and at most 12; the kind is KR, KFD, R16 or WH; no two have one name; each call's segment is one of RM RD FM FD WL WH IX; and with one call for each different shift sh of a group of layers, for the kinds that begin with K, there are at most 12 calls. | `policy.txt`, `KL.bin` | 
| `af-shape-heads` | G is a power of two and divides H. | `policy.txt` | 
| `shape-request-matrix-A` | lt <= lmh <= 24; lt <= lmo <= 24; lmbA <= 24; lmS <= 24; 1 <= NAs <= 64. | `policy.txt` | 
| `af-policy-wd` | S is below 32. | `policy.txt` | 
| `policy-iotaA` | iotaA_0 is 32 plus the depth of a tree of the attention sub-proof's lookup forest, in every group of layers. | `policy.txt` | 
| `plan-tensor-layout` | The feed-forward sub-proof, which owns the request matrix qa, and the attention sub-proof, which reads it, list the same tensors q, kx, v, o with the same numbers; Lam, H dh, G dh and la - lt are the same on both sides; 2^(la - lt) divides dh; and the four tensors have H dh or G dh features (11.3). The same name guards comparisons of the program's own numbers inside the sub-proofs. | `policy.txt` | 
| `table-digest-exp` | The softmax table that the verifier has computed has the digest of the lines `texp0` to `texp7` (R15.6), where the file has these lines. | `policy.txt` | 
| `shape-product-layer`, `shape-product-layer-H` | Each product layer has at least 9 variables: in a feed-forward segment lNI + lPd + lt, ll + lPf + lt and, with routed experts, ll - 1 + lfe + lt (11.2, steps 6, 12 and 14) are each at least 9 (`shape-product-layer`), and in the head lPd + lt is (`shape-product-layer-H`). | `policy.txt` | 
| `policy-sym-readers` | Each symbol tree that a feed-forward sub-proof owns is planned for one reader more than the number of sub-proofs that read it from outside; the trees of the head and of attention are planned for one reader, and the head and attention own no tree that another sub-proof reads. | `policy.txt` | 
| `plan-layout`, `plan-mask-row` | Every array that a segment reads is placed by the layout file and, if an earlier segment has sent it, is still kept (`plan-layout`). The mask rows have room (`plan-mask-row`): a segment's number of scalars plus answers is at most the number of entries of a message of the request's mask code (11.1, the room in the mask row), and the index-mask message has room for the rows on the table with their batching (8.4, item 18). | `layout.txt`, `policy.txt` | 
| `plan-link` | Every committed array of a segment has exactly one name, and the sub-proof's lists of its arrays have equal lengths. | `policy.txt`, `KL.bin`, `layout.txt` | 
| `plan-link` | The list of matrices of a segment's smallness test has fewer than 256 matrices, which cannot fail with NAf, NAd, NHf and NAs at most 64; and at least one of them is the segment's own and is read by no other segment (11.1, third piece). The same name guards the program's own order of steps inside a sub-proof, after the sub-proof's prefix; there it also asks that the arrays which a sub-proof commits to at a moment are exactly those that the layout file has at that moment. | `policy.txt`, `layout.txt` | 
| `plan-fold-source` | A matrix that a segment reads from an earlier segment belongs to the sub-proof of an earlier feed-forward segment, is one that its owner lists as read by others, and has the same rows and columns on both sides. | `policy.txt`, `KL.bin`, `layout.txt` | 
| `plan-layout` | `layout.txt` is there, is read without a fault, fits the arrays that the sub-proofs declare, and fits the codes (13.4). The stage's line before its last gives the reason, but not where the file is missing. | `layout.txt` | 
| `plan-link` | The number of segments with claims on the weights is the number of groups of layers plus 1. | `policy.txt`, `KL.bin`, `layout.txt` | 
| `parameters-not-admitted` | The frame lines are admitted together with the sub-proofs' shapes. | `policy.txt` | 
| `plan-layout` | Each line that is not skipped is a group's line or a block's line, with nothing after its last field (13.4). | `layout.txt` | 
| `plan-layout` | In a block's line: `link`, `moment`, `group` are not 0; `link`, `moment`, `group`, `array` are below 2^20; the letter is r or c; the last number is 0; a, e, br, bj are below 48. | `layout.txt` | 
| `plan-layout` | In a group's line: `link`, `moment`, `group` are not 0 and below 2^20; the class is N, B or W; rows is from 1 to below 2^32; life is below 2^20, and is not 0 exactly in class B. | `layout.txt` | 
| `plan-layout` | A group is in a segment and a moment that the request has; the groups of a moment are numbered in turn; the moments of a segment, and the segments, come in turn; the first oracle message of a segment is not kept. | `layout.txt` | 
| `plan-layout` | A block is in a group that the file has and belongs to an array that its moment has; an array over E is in rows over E, and one over F in rows over F; it is kept at least as long as its last reader needs it; all blocks of an array are in one group; every array has a block. | `layout.txt` | 
| `plan-layout` | In a block's line br + bj = a + e; `R`, `K`, `row` and `place` are below 2^48 and are multiples of 2^a, 2^e, 2^br and 2^bj. A block is inside its array, inside the rows that its oracle message sends, and inside a row's message. The blocks of an array do not overlap and hold each of its elements once. No two blocks of one oracle message share an entry. No block covers a pad of the message's out-of-domain sample. | `layout.txt` | 
| `plan-head`, `plan-link`, `plan-shape` | The list of segments has one head's segment, and it is the last (`plan-head`); no two segments have one identifier, and each has a plan of its rounds (`plan-link`); there are 2 to 64 segments (`plan-shape`). None of these can fail: the list is the program's own, and with at most 28 groups of layers it has 2 to 57 segments. | `policy.txt` | 
| `parameters-not-admitted` | The three files and the shapes of the sub-proofs are admitted by the routine that derives every size and plans every round (section 8). The stage's line before its last gives the reason. | `policy.txt`, `layout.txt`, `KL.bin` | 

On the key file, the leaf file, the caller's files and the proof file:

| name | what must hold | on | stage |
|---|---|---|---|
| `key-refused` | The beginning of `key.bin` has the form of R13.2: head, root, and the record's stored answers, each a canonical element of E, as many as the table's shape says. | `key.bin` | every stage |
| `model-statement` | The commitment ID that the stage has formed from the files equals the one after `-id` (13.5). Without `-id` nothing is compared. | `-id`; `key.bin`; `policy.txt`, `layout.txt`, `KL.bin` | every stage |
| `security-bits` | In the plan of the stage's own file every round's error, proof of work counted, is at most 2^-sec, and sec is at least the requested bits. The first half cannot fail at this point: the routine that plans every round has already rejected, in every stage and for all three files, a plan with a round above the target (`parameters-not-admitted`). The program's condition has a third part: some round of the plan has an error above 0 (`formed` in `blocks/numbers/accounting.h`). This page does not say whether a file that has passed the earlier tests can miss that part. A planning call also ends under this name if, in the plans of the three files together, some round's error, proof of work counted, is above 2^-sec (`print_plan` in `blocks/frame/prints.h`). | `policy.txt`; `-bits` | every stage |
| `key-refused` | Every test on `key.bin` (section 9), from its first byte to its last. The stage's line before its last gives the reason and where in the file it arose. | `key.bin` | key |
| `entry-refused` | Every test on the leaf file (section 9), from its first byte to its last, against the anchor as read from the beginning of `key.bin`. | leaf file; `key.bin` | leaf |
| `statement-required` | 12.2, test 1. | request file, token file | request |
| `statement-range` | 12.2, test 2. | request file, token file | request |
| `statement-range` | 12.2, tests 3 and 4. | request file, token file | request |
| `token-range` | 12.2, test 5. The name `token-length` belongs to a repeat of a test that has passed by then; it cannot fail. | request file, token file | request |
| `entry-refused` | The beginning of the leaf file has the form of R13.3: head (`CPEN`, version word), root, and the record's stored answers. | leaf file | request |
| `proof-refused` | Every test on the proof (sections 10 and 11) other than the comparison of the next token. The steps are those of 13.6, 1 to 8; at the end, no chain has rejected. The stage's line before its last gives the reason. | proof file | request |
| `reply-differs` | The next token in the proof equals the caller's next token (12.4). | proof file; token file | request |
| `header-y` | The element that the proof holds in the round `REPLY` has the form of a token: e_1 = e_2 = 0 and e_0 is below V. The head's sub-proof tests this before the comparison of 12.4. The same name belongs to two repeats, which cannot fail. The first, where the statement is formed, repeats test 5 of 12.2 on y. The second, in the head's sub-proof, is on the token of every checked position: for a token of the request it repeats test 5 of 12.2, and at position T - 1 it repeats this row's own test. | proof file | request |
| `plan-link`, `plan-mask-row` | With the first segment exactly one oracle message is sent at the start, the one with the masks of the request's rows on the table (`plan-link`), and it has a free entry for each segment with claims on the weights and for the batching of their rows (`plan-mask-row`). | `policy.txt`, `layout.txt` | request |
| `plan-head` | A segment's first message holds a plain scalar only in the head's segment, and there exactly one, the next token. | `policy.txt`, `layout.txt` | request |
| `link-claim-count` | A segment with claims on the weights has at least one claim, as many as its plan says, and a point with as many coordinates as the committed matrix has column variables. No byte of a proof file changes these numbers: they follow from the public files and from the program's own walk through the segment (from the programs' source). | `policy.txt`, `layout.txt` | request |

The names `key-refused`, `entry-refused` and `proof-refused` each stand for all tests that the
library `blocks/` makes on that file, and `chain-refused`, in a later table, for its tests on a
chain inside a sub-proof. In names and printed lines "entry" means a leaf file, "copy" a leaf table
and "reply" the next token. The library gives a reason as a sentence, which the stage prints in its
line before its last, after `key:`, `entry:`, `proof:` or `chain:`; no verdict depends on it. The
program reads it once, for a name: where the reason is `the proof is about another reply`, the stage
ends under `reply-differs` and not under `proof-refused`, and both are rejections. The call repeats
a reason that came after `key:`, `entry:` or `proof:` in its own line before the last, after the
stage's name and a colon; it leaves the reason out if the sentence has a digit or any character
other than letters, blanks and , . : ; ' ( ) - (from the programs' source). Where the chain itself
has no reason, the words say where in the file's order the test failed, for example
`first chain: the links`, `shared chain: a proximity test` or `second chain: its end`. The reasons:

| reason, as printed | what must hold | in |
|---|---|---|
| `file-too-short` | The file has the bytes of its head (R13.1, R14.2). | key file, leaf file, proof file |
| `file-of-another-kind` | The first four bytes are the tag of this kind of file. | key file, leaf file, proof file |
| `version-unknown` | The next four bytes are the program's version word. | key file, leaf file, proof file |
| `the table's root` | The 32 bytes of the anchor's root are there. | key file |
| `the copy's root` | The 32 bytes of the leaf table's root are there. | leaf file |
| `the record's answers` | Each stored answer is there and is a canonical element of E. | key file, leaf file |
| `the table's numbers are not a code`, `the table has too many rows`, `a row has no room beside its pads`, `the leaves of the table's tree have no admitted salt` | The table's shape, which follows from the public files, is admitted: its numbers are those of a code; it has at most 2^40 rows; the pads leave room in a row's message; the number of bytes of a leaf's salt is admitted. | public files |
| `the key names no configuration` | The verifier's configuration's digest is not 32 zero bytes. | public files |
| `the record has not one answer for each row and seed` | The record has exactly one stored answer for each row and seed. | key file, leaf file |
| `a table without a commitment` | The item `table` of a first query holds a root. | key file, leaf file |
| `the chain's salt is missing` | The chain's 32 random bytes are in the file (R4.5, case (a)). | key file, leaf file, proof file |
| `a chain's run's value is the digest of zeros` | The run's value is not 32 zero bytes (R4.5). | key file, leaf file, proof file |
| `an earlier chain of this proof was refused` | No earlier chain of the same file has rejected (R4.7). | key file, leaf file, proof file |
| `a value that the prover sends is missing or not in canonical form`, `a value that the prover sends is missing` | Every element and every root of a round's message is in the file, and every coordinate of an element is below p (R1.3). The second reason is that of a word. | key file, leaf file, proof file |
| `a nonce is missing` | In a round with proof-of-work bits the file holds a nonce (R5.3). | key file, leaf file, proof file |
| `a nonce does not give the proof of work that its round asks for` | The nonce passes the round's proof-of-work test (R5.5). Nothing is sampled from the answer of a round that fails it. | key file, leaf file, proof file |
| `an opening is cut short or holds a symbol that is not in canonical form` | Every symbol, salt and digest of an oracle message's opening is in the file, and every symbol is canonical (section 6). | key file, leaf file, proof file |
| `an opening does not lead to its root` | The opened leaves, with the digests that the file gives, lead to the root that was sent (section 6). | key file, leaf file, proof file |
| `a vertex of a tree gets two values under one root` | R7.14. | key file, leaf file, proof file |
| `an opening, or another answer to the verifier's queries, is cut short or not in canonical form` | The same test as three rows above (`an opening is cut short ..`), where a block has read the file itself: a stored table's columns, salts and digests. | key file, leaf file, proof file |
| `the opened columns of the table do not lead to its root` | The opened columns of a stored table lead to the root in the chain's first query (section 6; R7.14). | key file, leaf file, proof file |
| `BASE: the masked message in the proof does not satisfy the linear claim` | In the last stage of a proximity test, the message that the prover sends satisfies the stage's linear claim. This test comes before the stage's positions are sampled. | key file, leaf file, proof file |
| `BASE: the encoding of the masked message differs from the opened symbols at a listed position` | In the last stage of a proximity test, the encoding of that message agrees with the opened symbols at every position of the list. | key file, leaf file, proof file |
| `the seal's place` | After a key file's one-time chain: the word is 0, and the salt of the binding tree's one leaf is there (R13.6). | key file |
| `a chain of the key has refused` | All three chains of the key file have accepted. | key file |
| `the public files are not admitted` | The public files have been admitted before a byte of the proof is read. | public files |
| `the proof was made on another stored copy of the table` | The number in the proof's head is the number of the leaf file whose table is used. | proof file |
| `the proof is about another reply` | 12.4. The call then ends under the name `reply-differs`. | proof file; token file |
| `the request's place in its batch` | The slot, which the programs print as "place", is below 2^depth, and the salt and the digests of its path are there (R14.7). | proof file |
| `bytes are left over at the end of the proof` | The file ends with its last chain. | key file, leaf file, proof file |
| `a chain of the proof was refused` | Every chain of the proof was brought to its end, and none has rejected. | proof file |

Further reasons speak of the public files alone; after the steps of 13.4 no call reaches them. The
reasons of the library that follow are neither in the table nor about the public files (from the
programs' source). `the first table has passed no check: its key was refused, or was never checked`
and `the table has passed no check: its key or entry was refused, or was never checked` are the
library's answers to a caller that passes a table which no check has given out. No call of
`cp_verify` reaches them. A stage that has read a table for its form alone (13.6) gives it to
routines that are made for such a table. These routines have no plain pass: their best answer is a
pass on the condition that the table is tested, and that test is another stage's work.
`a block's verifier has refused without giving its reason` belongs to a routine that the example
protocols of `blocks/toy/` call and `cp_verify` does not.
`a key has another shape than the link's table` and
`a key's tree starts from another configuration's digest than the chain names` are a link's answers
to a caller that passes a table of another shape or configuration (`blocks/iop/link.h`). No file
reaches them: the routines that read a table from a key file or a leaf file give it the verifier's
own shape and configuration's digest, and take only the root and the stored answers from the file
(`blocks/frame/check_key.h`, `blocks/frame/check_entry.h`).

The next table has the names that are raised where the plans of the sub-proofs are made, in every
stage, or inside a sub-proof of a segment, in the stage request. Section 11 has some of the objects
that they speak of; for the others see the start of 13.8. A name that ends with `-` is completed by
the name of a call, a matrix or a tree. Inside a sub-proof many of the names come after the
sub-proof's prefix and a colon, and others, among them `coin-zero`, `header-y` and `chain-refused`,
are printed bare (from the programs' source). The prefixes are as in `G0:`, `A11:`, `H:`.

| name | what must hold | on |
|---|---|---|
| `af-shape-tiles` | lt is from 2 to 16, and lB is from 0 to lt - 2. | `policy.txt` |
| `af-shape-sh` | A group of layers has exactly 2^ll values sh. | `KL.bin` |
| `af-shape-la` | 0 <= la - lt <= log2 dh, and la <= 16; la <= lmo and la <= lmq; 2^(la - lt) divides 2^lmo / T and 2^lmq / T. | `policy.txt` |
| `af-policy-softmax` | S is from 1 to 14, rx from 1 to 12, and lt + S + 1 <= 24; the softmax table reaches the value 0 at a row u0 <= 2^20 + 1. | `policy.txt` |
| `af-shape-lmS` | lmS is at least 1 and at most nD (11.3). | `policy.txt` |
| `af-policy-kF` | kFD is from 8 to 16; the table with 2^kFD rows has room for its first u0 + nHD rows, and at most nHD rows more, for each shift (u0: 12.6; nHD: 11.3, second half). | `policy.txt` |
| `af-policy-sh` | Every value sh that occurs is from 4 to 12. | `KL.bin` |
| `af-policy-C` | For each shift sh of a group of layers, the constant C_sh = u0 2^sh of the attention sub-proof (11.3, second half) has C_sh + 65536 <= 2^24. It is not the line `C` of the acceptance policy. | `policy.txt`, `KL.bin` |
| `af-no-wrap-N3`, `af-no-wrap-N4`, `af-no-wrap-N1`, `af-no-wrap-N2`, `af-no-wrap-N5`, `af-no-wrap-N6`, `af-no-wrap-N9` | Attention's no-overflow conditions N1 to N6 and N9: sums of bounds stay below 2^60, below 2^61 - 1, below 2^32 or below 2^20. The single conditions are in 11.3, second half. | `policy.txt`, `KL.bin` |
| `af-policy-call-list` | Every line `callA` has at least one call's segment; each of the seven call's segments occurs exactly once in the whole list, in a line of its kind (RM, RD in KR; FM, FD in KFD; WL, IX in R16; WH in WH); there are at most twelve expanded calls. | `policy.txt` |
| `af-policy-nmin` | nmin is at least 10. | `policy.txt` |
| `af-shape-call-` | Each expanded call of attention has 2^n keys with n from 10 to 44. | `policy.txt` |
| `forest-depth-A` | The depth of a tree of attention's lookup forest is at least lmS and at least lt + log2 dh. | `policy.txt` |
| `hid-a-owner-uses` | A number that the plan passes to the attention sub-proof, how many sub-proofs read the matrix qa, is not 0. | `policy.txt` |
| `hid-a-fold-list` | The smallness test of an attention sub-proof has NAs + 4 matrices, at most 256. | `policy.txt` |
| `hid-a-weight-matrix` | An attention sub-proof has no weight matrix in its list. By this page's reading of the source it cannot fail (11.3, second half, the programs' self-checks). | `policy.txt` |
| `hid-a-slots` | An attention sub-proof sends fewer than 2^30 scalars. | `policy.txt` |
| `hid-g-slots` | A feed-forward sub-proof sends fewer than 2^30 scalars. | `policy.txt` |
| `hid-h-slots` | The head's sub-proof sends fewer than 2^30 scalars. | `policy.txt` |
| `af-unclosed-call` | In an attention sub-proof, before the join, every lookup call has been given its keys exactly once. By this page's reading of the source it cannot fail (11.3, second half, the programs' self-checks). | `policy.txt` |
| `hid-profile-segment-` | Every call's segment of a feed-forward sub-proof is one that the sub-proof has a relation for. By this page's reading of the source it cannot fail (11.2, "What cannot fail"). | `policy.txt` |
| `shape-pt-words-` | For each lookup call: its table has 2^k rows with k from 5 to 16; its other numbers are in range; a table that is given as a list of values has 2^k entries. | `policy.txt` |
| `forest-row-value-` | Every public value of a lookup table's row is below the field's prime. | `policy.txt` |
| `forest-calls` | A lookup forest has 1 to 16 calls. | `policy.txt` |
| `forest-depth`, `forest-trees`, `forest-keys` | In a lookup forest: a tree has depth at least 10 (`forest-depth`); there are at most 256 trees (`forest-trees`); there are at most 2^31 keys (`forest-keys`). | `policy.txt` |
| `no-wrap-pin-forest` | 2^(kT + cK + 32) is below p, where 2^kT is the largest table's number of rows and cK is the least number with 2^cK not below the number of calls. | `policy.txt` |
| `pt-pub-forest`, `forest-only-padding-` | No tree of a lookup forest has more padding keys than positions (`pt-pub-forest`), and no call consists of padding only (`forest-only-padding-`). | `policy.txt` |
| `forest-planes` | The matrix that holds a forest's multiplicities in bits has 2^lm rows with 5 <= lm <= kT + 5, where 2^kT is the largest table's number of rows; lm is lmb, lmbA or lmbH. | `policy.txt` |
| `shape-matrix-`, `hid-k-range`, `t2f-weight-class`, `shape-foreign-`, `t2f-columns-range` | In the list of matrices of a smallness test: each has at least one row (`shape-matrix-`) and 1 to 8191 columns (`hid-k-range`); none is a weight matrix (`t2f-weight-class`); a matrix that an earlier segment has committed is that segment's whole matrix, with the same numbers (`shape-foreign-`); twice the sum of all rows is below 2^32 (`t2f-columns-range`). | `policy.txt` |
| `plan-tau` | The line `tau` equals the number of combinations of the smallness test that follows from the frame lines (section 8). | `policy.txt` |
| `plan-w-zero` | A number that the plan passes with each symbol matrix is 0. | `policy.txt` |
| `w-shape-lk` | w, the last number of the line `table`, is from 1 to 20. | `policy.txt` |
| `w-shape-mW` | The committed matrix has at most 2^22 rows. | `policy.txt` |
| `chain-refused` | Inside a sub-proof, the chain has not rejected: every scalar that is read is there and canonical, and in a round with proof-of-work bits the nonce is there and passes the round's test. The name is also given where the library's routine for a segment answers that a call was not due. A round that is not the one that the plan has next gives no rejection: by the source that is a fault of the program and not of a proof, and a guard of the library ends the process (17.2, item 1). Where the chain has rejected, the stage's line before its last gives the chain's reason (the table of reasons above). | proof file |
| `coin-zero` | In a feed-forward sub-proof, the element eps that is sampled in `PSC epsilon` of the second token sumcheck (11.2, step 9), whose claim is divided by eps afterwards, is not 0. | proof file |
| `shape-functional-` | A linear functional on a matrix of the smallness test has a length that is not 0 and divides the matrix's number of rows, and as many coefficients as its length. | `policy.txt` |

Names that begin with `internal-` are guards of the program on itself. A call in which a stage ends
under such a name, also after a sub-proof's prefix, and in which no other stage rejects, does not
end with `NO` but with `NOT CHECKED: <name>` (14.3): nothing has been verified, and this is not an
acceptance. Eight of them can also be raised by a rare value of what the verifier samples, which
depends on the proof's bytes: `internal-f-table-read`, `internal-t2f-untyped-`,
`internal-t2f-untiled`, `internal-sym-rowvec`, `internal-f-tie-weights`, `internal-g-free-slot`,
`internal-a-free-slot`, `internal-h-free-slot`. The tests on sampled values under these eight names
are not the program's own: they are tests on content of every verifier, and 13.9 says what a
verifier does at such a value and how often it comes. Nor is every other test under such a name one
that a second verifier can leave out: some are tests on the public files, which every verifier makes
(`internal-qa-layout`, 11.3; `internal-weight-limbs`, 17.2, item 1); 11.1, "What cannot fail", says
how far these pages tell the two kinds apart. A verifier with two answers, as R0.3 asks, rejects in
all the cases of this paragraph. The list below has the names under which a stage can end. One of
them, `internal-claim-values`, is raised only in code that serves tests: by a comment in
`verifier/src/vf_subproof_engine.h` `cp_verify` never sets the values that this code reads, so it
never raises this name; checkers under `verifier/tests/` do. Seven more belong to the process that
starts the stages and reads their results, and are given to the call as a whole: `internal-stage`,
`internal-sums-line`, `internal-covered-line`, `internal-id-two-ways`, `internal-result-line`,
`internal-policy-not-given` and `internal-policy-kind` (14.3). The last two guard the tests under
`policy`: a call without `-policy` has come as far as `WITHIN LIMITS`, or a whole call has come to
its last line under a kind of policy other than `exact-match` and `clipped-gap`. The names of the
stages:

`internal-W-length`, `internal-W-length-H`, `internal-af-W-length`, `internal-af-calls`,
`internal-af-padding-key`, `internal-call-`, `internal-fork-not-finished`, `internal-header-length`,
`internal-layout-words`, `internal-line-`, `internal-qa-layout`, `internal-sc3-`,
`internal-segment-`, `internal-table-`, `internal-tensor-`, `internal-L-index`, `internal-L-row`,
`internal-L-slot`, `internal-a-TX-place`, `internal-a-Ty`, `internal-a-claim-count`,
`internal-a-free-slot`, `internal-a-tree-list`, `internal-batching-bare`, `internal-claim-count`,
`internal-claim-factors`, `internal-claim-unit`, `internal-coef-`, `internal-exception-parFor`,
`internal-eq-index`, `internal-g-claim-count`, `internal-g-free-slot`, `internal-gamma-rows`,
`internal-h-claim-count`, `internal-h-free-slot`, `internal-h-owner-TX`, `internal-h-owner-shape`,
`internal-h-tree-list`, `internal-nmlab`, `internal-shape-moe`, `internal-slot-count`,
`internal-stack`, `internal-t2-order`, `internal-t2f-TX-order`, `internal-t2f-ZF-order`,
`internal-t2f-functional-`, `internal-t2f-lists`, `internal-t2f-map`, `internal-t2f-unit-`,
`internal-t2f-untiled`, `internal-t2f-untyped-`, `internal-t2f-blocks`, `internal-weight-matrix`,
`internal-a-calls-booked`, `internal-a-owner-shape`, `internal-a-table-index`,
`internal-a-table-length`, `internal-fold-columns`, `internal-g-calls-booked`,
`internal-g-owner-shape`, `internal-h-calls-booked`, `internal-t2f-table-outside`,
`internal-t2f-tables-overlap`, `internal-a-owner-place-`, `internal-af-rows-booked`,
`internal-border-count`, `internal-f-bz-weights`, `internal-f-exist`, `internal-f-fun-closed`,
`internal-f-fun-open`, `internal-f-join-place`, `internal-f-matrix-order`, `internal-f-order`,
`internal-f-padding-key`, `internal-f-table-index`, `internal-f-table-read`,
`internal-f-tables-gap`, `internal-f-tail-slot`, `internal-f-tie-weights`,
`internal-f-read-weights`,
`internal-f-zzcol`, `internal-g-owner-place-`, `internal-g-rows-booked`, `internal-h-owner-place-`,
`internal-h-rows-booked`, `internal-sym-point`, `internal-f-fun-moved`, `internal-sym-rowvec`,
`internal-sym-weight`, `internal-root-behind-sd`, `internal-f-fold-array`, `internal-sym-shape`,
`internal-f-draws`, `internal-build-word`, `internal-naming-first`, `internal-statement-cover`,
`internal-statement-prescan`, `internal-statement-twice`, `internal-th`, `internal-version-word`,
`internal-f-door`, `internal-last-line`, `internal-refusal-name`, `internal-par-digests`,
`internal-a-merged-point`, `internal-forest-join`, `internal-forest-place`, `internal-forest-point`,
`internal-forest-pub`, `internal-psc-degree`, `internal-psc-order`, `internal-claim-not-added-`,
`internal-claim-values`, `internal-forest-no-proof-read`, `internal-forest-padding-places`,
`internal-g-join-place`, `internal-h-join-place`, `internal-slot-order`, `internal-slot-unused`,
`internal-g-forest`, `internal-g-link`, `internal-g-merged-point`,
`internal-g-routed-without-terms`, `internal-g-table-index`, `internal-g-table-point`,
`internal-h-forest`, `internal-h-merged-point`, `internal-h-table-index`, `internal-slot-marks`,
`internal-slot-twice`, `internal-cp-cut`, `internal-round-name`,
`internal-parameter-name`, `internal-row-form`, `internal-shapes-order`, `internal-sealed-request`,
`internal-weight-limbs`, `internal-configured-twice`, `internal-work-two-ways`,
`internal-asked-bits`, `internal-plan-moment`

### 13.9 The rows that a verifier forms, and the tests on content

Like the rest of this section, this subsection is taken from the programs' source. It needs the
terms of sections 8 to 10: oracle message, main row, mask row, pad, free entry and where a vector is
in an oracle message's main rows (both at the start of section 9), group, row (section 10; section 9
says linear row), segment, carrying message, switch, index-mask message, the oracle messages W and
Omega, and the folded row, whose oracle message is written G_u here. End value and merge are defined
in 11.1, live plane in R13.14 and R13.17, own group in section 9.

While the rounds of a chain run, a verifier rejects only for form: a value that is missing or not
canonical, a failed proof-of-work test, a listed position that differs. No equation is tested in a
round. Instead the verifier forms **rows**: linear equations < coefficients, entries of oracle
messages > = target, each on the members of one group. (This is a row's target, as in section 10,
not the target of the line `sec`.) A sumcheck's round fills in the coefficient that is not sent and
carries a running claim on. At a group's end all its rows are batched into one, the group is folded,
and what is left goes into the next group. So a verifier must form every coefficient and every
target exactly as below, from what the first query holds, what it has received and what it has
sampled (R0.4). In the frame, that is, outside the sub-proofs, the tests on content are these five,
and no others, beside the tests of form of 13.8. Some tests of 13.8 are neither among the five nor
named in the first sentence above, and every verifier makes them too. Three of them are easy to
miss: no chain's run's value is 32 zero bytes (R4.5); the number in the proof's head is that of the
leaf table that is read (R14.2); and the slot is below 2^depth (R14.7). A verifier that takes first
queries, plans, codes or shapes as inputs, and does not form them itself, has comparisons of its own
to make: R13.5 and R14.9 (three items that two chains both hold) and section 18, "What ties the
inputs of one chain or file to another's". Inside a sub-proof there are `header-y` (13.8) and the
tests on sampled values that come after this list, `coin-zero` among them.

1. Every opened leaf, with its path, leads to the root that was sent, and every opened column of a
   stored table leads to the root in the chain's first query (section 6), and no vertex of a tree
   gets two values under one root, in whichever chain of the file it is computed or given (R7.14).
2. At the end of each proximity test, in `BASE`, before the stage's positions are sampled: the
   masked message M' that the prover sends in the clear satisfies < c, M'_main > + < d, M'_mask > =
   target* + lambda target. Here (c, d, target) is the last stage's one row; target* is the element
   that the prover has sent first, its value of the same row on the masking oracle message g*; and
   lambda is the element that the verifier has sampled after it.
3. In `BASE`, at every position of the list and for both rows: the encoding of M' with its
   randomness R', which the prover sends with it, equals the opened symbol of g* plus lambda times
   the opened symbol of the row; a mask row is read at the position modulo its length.
4. The next token is the caller's (12.4).
5. Every chain was brought to its end, and none has rejected. (That every group has had its
   proximity test follows from the plan: no file can change it.)

The request's slot in its batch is tested for its form alone (13.8): seal, salt and path give the
binding tree's root, which enters the shared chain's first query (R14.7, R14.9).

*Sampled values at which a verifier gives no acceptance* (from the programs' source). After a
segment's last claim is added, every weight that the segment's one equation puts on a committed
entry or on a scalar is known (11.1). These weights are public: they are made of public numbers and
sampled values, and in a feed-forward segment some also have a scalar that the proof sends as a
factor. The source gives two reasons for testing them: a weight that is 0 means a cell or a scalar
that no claim weighs, and two equal row weights of a matrix would permit a forgery. For event 9 it
gives a third: the test guards hiding, because that every scalar is in a claim is what lets whoever verifies a proof before it is handed on
see an end value that was sent without its mask. The tests of the table below are tests on content
of every verifier, like the five above: if one of these events happens in some segment of a proof,
the verifier gives no acceptance. A verifier with two answers (R0.3) rejects. `cp_verify` rejects at
event 1; at the others it ends with `NOT CHECKED` and the name in the table, after the sub-proof's
prefix (14.3), and that is not an acceptance either. Section 11 defines each event, in the part that
the table names. Only this subsection says what a verifier does at them.

| no. | event, in one segment | name in `cp_verify` | defined in | how often |
|---|---|---|---|---|
| 1 | the element eps of the second token sumcheck is 0 | `coin-zero` | 11.2, step 9, and "Two tests beside the claims" | 1 value of eps; feed-forward segments only |
| 2 | all tau entries of a column of the sampled matrix Theta are 0, so that colw_mu(c) = 0 for every chi | `internal-f-table-read` | 11.1, third piece, event 1 | a chance of exactly 2^-tau for each column of each matrix of the list: by R6.7 an entry is 0 with the chance 1/2 |
| 3 | colw_mu(c) = 0 for a column whose entries are not all 0 | `internal-f-table-read` | 11.1, third piece, event 1 | at most tau - 1 values of chi for each column: colw_mu(c) is a polynomial in chi of a degree below tau |
| 4 | chi = 0 | one of the names of this table, or `internal-f-tie-weights` | 11.1, third piece, events 2 to 4 | 1 value of chi |
| 5 | two row weights rw_mu(i') of one matrix are equal | `internal-sym-rowvec` | 11.1, third piece, event 4 | an exact number of values of chi, given below |
| 6 | a weight of the join on `ZZ` is 0: e_0 or e_1 is 0, or eq(zk, .) is 0 at a key that is not padding | `internal-t2f-untiled`, `internal-t2f-untyped-` | 11.1, third piece, events 2 and 3 | only if one of the v_Z coordinates of zy and zx is 0 or 1: a chance of at most 2 v_Z divided by the size of E |
| 7 | a weight on a column of `BZ` is 0: eps_L = 0, A_L = 0, or one of the last 11 coordinates of zb_0 or of zb_1 is 0 or 1 | `internal-t2f-untyped-` | 11.1, third piece, event 2 | a chance of at most 47 divided by the size of E: 1 for eps_L, 2 for A_L, whose degree in the coordinates of zb_0 is 2, and 44 for the coordinates |
| 8 | the weight of an entry of `U3` is 0, or the weight that the claims READ_q put on a column of `ZZ` is 0 | `internal-t2f-untyped-` | 11.1, third piece, event 2 | below |
| 9 | the coefficient of a scalar in the sum of all claims is 0; the merge's end value fv' is left out | `internal-g-free-slot`, `internal-a-free-slot`, `internal-h-free-slot` | the second halves of 11.2, 11.3 and 11.4 | below |

*What these tests cost.* For events 2 and 5 the plan's error for the round holds whether a verifier
rejects at the event or not, so these two tests add nothing to soundness. This follows from how the
plan counts. Event 2: the plan gives the round `RNG` the error 2^-tau, and the note that its
accounting prints says that this counts one column of the committed matrices, the first that no
small integer column explains, which is fixed before the sampling (`blocks/numbers/blocks.h`,
`blocks/numbers/accounting.h`; that note calls the count argued, and names a union bound over all
columns as a stricter reading). A column of Theta whose tau entries are all 0 is the case in which
that column enters no combination, and this case is within the 2^-tau (the last step is this page's
reasoning, proved nowhere). Event 5 (this page's reasoning, proved nowhere; a comment at the test in
the source says the same in one sentence and gives no proof: whether two of these weights are equal
or 0 depends only on the sampled element, and a committed false matrix meets one fixed polynomial in
that element, which the batching round's error already counts): the committed values are fixed
before chi is sampled, so a false set of them is one fixed polynomial in chi, and the error of the
segment's last `BAT-sigma` bounds the chance that it vanishes at chi, whether or not a verifier
rejects at two equal weights. At full size the event itself is likelier than that bound in field 2
and less likely in field 1 (the numbers are below). It costs completeness only. All nine give the
format a completeness error: a proof that an honest prover makes of a true statement meets one of
these events with a small chance, and then no verifier accepts it. The chances here take every
sampled value as uniform and independent of the others (R6.11 has the distance from that). |E|, the
size of E, is p^3, about 2^192 in field 2 and about 2^183 in field 1. The sizes are those of the
full-size example configuration, `prover/examples/target_full_f2/`: tau = tauP = 128 and 25
segments. The numbers of columns, rows, weights and values of chi below are this page's own count
from that configuration's files; no other file of the repository that states them is known. The
sizes follow from the model's lines alone, by this page's reading. The numbers for field 1 are
arithmetic at the same shapes: the repository has no configuration for field 1.

- *Event 2 is by far the most likely.* A verifier admits tau = `sec` alone (11.1), so the chance is
  2^-sec for each column, with sec the number of that line. At full size a feed-forward segment has
  24,086 columns in the matrices of its list, an attention segment 10,248 and the head 20,703. That
  is 432,711 columns in a proof, so a proof meets event 2 with a chance of 432,711 x 2^-128, about
  2^-109.3, in both fields.
- *Event 5, counted exactly.* rw_mu(i') = chi^(tauP i') for i' < mp_mu. For chi other than 0, two of
  these are equal exactly if the multiplicative order of chi^tauP is below mp_mu. Let N = |E| - 1, g
  = gcd(N, tauP), and mp the largest mp_mu of the segment. The number of such chi is g times the sum
  of Euler's function of d over the divisors d of N / g that are below mp. At full size every
  segment has a matrix of 2^16 cell rows, so mp = 21,846: that gives 40,300,544 values of chi in
  field 2, a chance of about 2^-166.7 for each segment, and 2,163,730 values in field 1, about
  2^-162.0.
- *Events 3, 8 and 9, as far as chi decides.* chi is the last value that a sub-proof samples. Each
  of these weights is a polynomial in chi whose degree is below n, the n of the segment's last
  `BAT-sigma` (11.1), and whose coefficients are fixed before chi is sampled. If the coefficients
  are not all 0, at most n - 1 values of chi make the weight 0. At full size n is about 32 million
  in a feed-forward segment, which has the largest, so below 2^25; and a segment has fewer than 2^22
  weights under test, about 3.4 million in a feed-forward segment. So these events together have a
  chance below 2^47 / |E| for each segment. This is an upper bound, not a count.
- *What this page does not derive.* The coefficients of the weight of an entry of `U3` are all 0
  only at event 2. For the other weights of event 8 and for event 9 this page has no chance that all
  coefficients are 0. They are sums of products of values that are sampled before chi: entries of
  the weight vectors phi_f (11.2 to 11.4), and the coefficients with which the claims name a scalar.
  The source says of a scalar's coefficient that for an honest proof it is 0 "with chance about
  1/|E|", and gives no derivation.

Notation: eq(a, b) is the product over t of (a_t b_t + (1 - a_t)(1 - b_t)); eq(., z) as a
coefficient means eq(index bits of the entry, z); < , > is the inner product; (x) is the tensor
product; the **generator row** of a code at a position x is the vector whose inner product with
(message, encoding randomness) is the code word's symbol at x. A vector v of 2^k entries folded at
g_1 .. g_k is the sum over r of eq(bits of r, g) v[r], where g_1 goes with the lowest bit of r.
Folded at fewer elements, it keeps its higher bits.

Terms (from the programs' source). The **summand** of a sumcheck is the public polynomial that is
summed: a function of a point and of the values of some maps at that point. A map that is **shown**
is one whose value at the sampled point the prover sends at the end, under a mask of its own, as an
end value. A **kept map** is one whose value is not sent: the summand is A_0 + sum_i A_i (kept map
i), where A_0, A_1, .. depend on the point and on the end values, and the kept maps stay as terms of
the row that the sumcheck leaves. In a lookup, G_j is the root of live plane j's product tree, Z is
the product of the roots of the parts' product trees, and P_j is a value that the prover shows for
P_(j+1)^2 G_j. A **cube of padding** is an aligned block of 2^b keys of padding, and its **product
form** at a point is eq(point, .) on these keys and 0 on all others. The **single rows** are what a
fold leaves of a group: one main row, the fold of the group's main rows; the kept rows, which are
the main rows of the group's kept members as they were sent; the **old mask row**, the fold of the
group's mask rows; and the **fresh mask row**, the fold of the rows of the fold's fresh mask. The
**kept forms** are the batched row's coefficients on the kept members' main rows.

Out-of-domain samples and sumchecks, in every chain:

| where | what the verifier forms |
|---|---|
| after a root, round `SEND seed` | The entries of an oracle message are indexed by a cube. The main table has ceil(log2(main rows)) + log2(entries of a row's message) bits, the mask table likewise with its own numbers, row bits low. With one table the cube has that table's bits. With both it has the larger number plus 1, and the top bit is 0 for the main table and 1 for the mask table. The number of bits is the number of coordinates of the point of an out-of-domain sample. Where one round samples for several oracle messages, answer number i has one point z_i and one beta_i for all of them. The point has as many coordinates as the largest of their cubes, and each message reads the first ones, as many as its own cube has. |
| the same | The pad of answer i is in the mask rows if there are any, else in the main rows; in the highest real row; at the last s entries, s being the number of answers (3 s entries for main rows over F, read through the basis 1, X, X^2). Answer i, from 0, has entry l - s + i of that row, l being the number of entries of a row's message; for main rows over F the three entries from l - 3 s + 3 i on. |
| round `SEND answer` | For each answer i of an out-of-domain sample, one row on the message, with the answer as its target. Its coefficient is eq(., z), z = z_i the point of answer i, times a front factor, on every entry that is not a pad; beta_i times the basis on the pad of answer i; 0 on the pads of the other answers. With both tables the front factor is (1 - z_top) for the main table and z_top for the mask table, times the product of (1 - z_t) over the cube's bits above the table's own. The rows wait until the message joins a group. |
| a sumcheck with a hiding sum (R13.16) | It leaves one row: the public weights of the hiding polynomial at the sampled values, on its fresh mask entries; plus eps times the claim's old mask entries, each with the product of the weights psi_t(gamma_t) (in a sub-proof the start form can also name entries of arrays, and these come into the row under the same factor: 11.1, second piece, the claim that a sumcheck leaves); plus eps times sum_i A_i times (kept map i)(gamma); equal to q_last(the last sampled value) - eps A_0. Here (A_0, A_1, ..) is the summand at gamma with the end values put in. The weight on the entry of variable l and degree t is gamma_l^t times the product of psi_j(gamma_j) over j > l; entries come by variable, then by degree upwards; degrees 1 to d - 1 for every variable but the last, 0 to d - 1 for the last. With equality weights psi is 1; with plain weights psi_j(gamma_j) = gamma_j. |
| the same | For each map that is shown, one claim at the point gamma: (map j)(gamma) + beta c_j = end value j, where beta is the element sampled in `PSC Y` and c_j is the next fresh mask entry after those of the hiding polynomial. |

A key file's one-time chain: the proof that the table's entries are in their ranges:

| where | what the verifier forms |
|---|---|
| one-time chain, each lookup (R13.17) | The padding's public factor: the product, over the runs of padding, of (alpha - the padding row's value) to the power of the run's length. |
| each lookup, the tie | a is the padding's public factor. Step 0 proves Z - a P_1^2 G_0 = 0: a sumcheck without variables, target 0, end values (Z, P_1, G_0), A_0 = y_0 - a y_1^2 y_2. Step j, for j = 1 to the number of live planes less 2, starts from the claim on P_j, has the end values (P_(j+1), G_j) and A_0 = y_0^2 y_1. The last P is the last live plane's G. |
| each lookup, after the tie | After the products: one claim on the root Z and one on each live plane's product G_j. Each step's row of the kind above joins the group of the multiplicities. |
| each lookup with several parts | For each part but the last: a sumcheck without variables on the claim on the product of the roots that are left, end values (this part's root, the product of the rest), A_0 = y_0 y_1. |
| each lookup | One claim on the table side's top level at r: its value is sum_j eq(r, j) times the value of G_j, where a j that is not below the number of live planes, a plane of public zeros, counts with the value 1; the masks are weighted alike. |
| each level of a product tree | Equality weights at the claim's point, end values (the two halves of the level below in its last variable), A_0 = y_0 y_1. Then `LINE`, with the sampled rho: the point gets rho as its last coordinate, the value is (1 - rho) y_0 + rho y_1, and the masks of the two end claims are taken times (1 - rho) and rho. |
| each lookup, the leaves of the table's side | The row of the leaves: the bits b at the point, under eps (alpha - 1 - tau(the point's table coordinates)), with A_0 = 1; tau of the range table at z is sum_j 2^j z_j. The map is on the live planes alone. |
| the test that each b is a bit | Equality weights at the point of the round `ZERO`, target 0, one end value, A_0 = y_0^2 - y_0. |
| the same | The row of the test that each b is a bit, and its end claim as a row: b(gamma) + beta c = end value, with beta the element sampled in `PSC Y`. |
| each sumcheck of a lookup | The order in which it binds its variables is in R13.17: from the highest down, but for the last level of each part's product tree. Each sampled value is put back at its variable's coordinate before the point is used. |
| each lookup | A row without a message: the sum over planes j and table rows t of 2^j b[j, t] equals the number of all keys less the number of padding positions. |
| each lookup, for each part | The claim on the leaves alpha - key becomes a claim on the keys: the target is alpha - value, the masks change sign, the weights are eq(point, .) less one product form for each cube of padding, and the padding's share is taken out of the target. |
| each lookup, for each part | The claim on a part's keys becomes a row on the committed matrix: its point psi on the columns is the first k_col coordinates of the claim's point; its weight on the matrix row of (position, digit q) is eq(z, position) times the weight of digit q, on the covered positions; its target is the claim's target less the offset times the sum of eq(z, .) over the covered positions; its masks are two entries of the index-mask message. |

The rows on the committed matrix: a key file's one-time chain, a proof file's first chain:

| where | what the verifier forms |
|---|---|
| first chain, after each segment with claims on the weights | The segment leaves one row on the committed matrix M: < w (x) eq(psi, .), M > + beta c = v, where c is the next free entry of the index-mask message. w are the row weights that the sub-proof's claims add up to, on the rows that the claims' units span. psi is the point of the sub-proof's merge with its coordinates in reverse order, so that the coordinate of column bit 0 comes first. beta and v are the merge's (section 11). |
| one-time chain and first chain, joining the rows on the table | A sumcheck with a hiding sum over the k_col column variables joins J rows into one. With J = 1 nothing is done and the row is taken as it is. Let x be the element of the round `BAT-sigma`, chi_j = x^j for row j, counted from 0 in the order in which the rows were left, tau_j the target of row j, psi_j its point, w_j its weights, and mu the hiding sum. The running claim starts at mu + eps sum_j chi_j tau_j. Each round fills in q_1 = claim - 2 q_0 - q_2, and the new claim is q at the sampled value. The row that comes out has the point psi = gamma; the weights w = sum_j a_j w_j with a_j = eps chi_j eq(psi_j, gamma); as masks the 1 + 2 k_col fresh entries of the index-mask message under the hiding polynomial's weights, and then the incoming masks times eps chi_j times the product of the gamma_t; and the last claim as its target. |
| after it, with the oracle message W | A new group, the own group, under W's code: its members are the index-mask message and W. Row (Eval): W at psi, that is eq(psi, .) on W's entries, plus the claim's mask terms, equals the claim's target. The group is tested in the second chain. |

The shared chain and the second chain, of a key file and of a proof file:

| where | what the verifier forms |
|---|---|
| shared chain | For each answer number i, one row on the folded row G_u: the record's form, with the seeds of R4.8, on the folded message equals sum_r eq(gamma, r) y[r, i], y the stored answers. These rows come before the columns' rows. |
| shared chain, round `LOC` | For each of the draws x, in their order, and twice for a position that is sampled twice: the generator row of the table's code at x, on G_u's message and randomness, equals sum_r eq(gamma, r) column_x[r]. gamma is the rest of the shared point psi* of `LIN point`, after its first k_lo coordinates. |
| shared chain, round `LIN zeta` | Row (Fold): eq(zeta, .), cut below the matrix's number of rows, on Omega, minus eq(psi_lo, .) (x) eq(zeta, .), cut likewise, on G_u; target 0. psi_lo is the first k_lo coordinates of the shared point psi* of `LIN point`. |
| shared chain, where the two are under different codes | That row is on two groups, so it is split with a masked share: on Omega's group, (the part on Omega) + kappa = share; on G_u's group, (the part on G_u) - kappa = - share. kappa is the next free entry of the shared split's mask message; the share is the one element of the round `SPLIT`. |
| second chain | Row (Dot), without a message: the claim's row weights w, cut below the matrix's number of rows, on Omega, minus eq(psi*, .) on W, psi* the shared point; target 0. Omega joins the own group, and the row is added there. W and Omega are always under one code: a parameter file with two is rejected (`parameter-own-code`). |

A leaf file: the link:

| where | what the verifier forms |
|---|---|
| a leaf file's chain | As in the shared chain: for each answer number, one row on the anchor's folded row, then the same for the leaf table's, each with its own table's seeds and stored answers. gamma is the point of `LINK point`, one for both tables. |
| a leaf file's chain, round `LINK` | As in the shared chain's `LOC`, for each of the draws in their order: the first nu_0 draws give rows on the anchor's folded row, from the anchor's columns; the other nu_j give rows on the leaf table's (R13.10). |
| a leaf file's chain, round `LINK phi` | The equality row: < eq(phi, .) on the entries before the pads, the anchor's folded message - the leaf table's folded message > = 0. |

A segment of a proof file's first chain:

| where | what the verifier forms |
|---|---|
| first chain, at a segment's end | The sub-proof ends in one equation with a public target: terms scale times (weight on rows) times (weight on columns) on named arrays, of this segment or of an earlier segment that is still kept, and coefficients on the first entries of the segment's mask row, in order. It becomes one row of the segment's group. |
| the same | Where each member's rows are among the group's 2^k main rows and 2^h mask rows: a member with K real rows takes an aligned block of 2^ceil(log2 K) rows; blocks are placed from row 0 by decreasing size, ties in the order of sending; kept members are not among the main rows. |
| round `BAT` | All rows of the group become one: row i is taken times theta^i (the order is in 10.6), and the target is sum_i theta^i target_i. |
| rounds `FOLD epsilon`, `FOLD round` | The running claim starts at the hiding sum + eps times the batched row's target. In each round the verifier fills in q_1 = claim - 2 q_0 - q_2, so that q(0) + q(1) = claim, and the new claim is q at the sampled value g_j. No test is made here. |
| after the fold | What a closed group leaves: three single rows and the kept rows, with one claim. The main coefficients are the batched row's, fixed at g_1 .. g_k in their low k variables and times eps; the kept forms times eps prod_j (1 - g_j); the old mask row's coefficients folded at g_1 .. g_h, times eps g_(h+1) .. g_k; the fresh mask row's coefficient at position 0 is g_(h+1) .. g_k (1 if h = k); at position l, for l = 1 to h, it is g_l times that product; then come, in this order, the pairs (variable l, degree t) = (h+1, 2), (h+2, 1), (h+2, 2), .., (k, 1), (k, 2), each with g_l^t times the product of g_j over j > l; then, if h < k, one position with 1; all later positions have 0; the target is the last claim. |
| round `LOC` of the next segment, or `CS` of a proximity test | For each position of the list, the symbols of the single rows: the column of the 2^k main rows folded at g_1 .. g_k; the kept rows' symbols as they are; the column of the 2^h mask rows, and the fresh mask's 2^h symbols, each folded at g_1 .. g_h. Rows that nobody has sent are 0. |
| the same | What the fold leaves is placed in the new oracle message as two vectors. The first is the folded main row's message. The second holds, in this order: the message of the old mask row, of the fresh mask row, the encoding randomness of the main row, of each kept row, of the two mask rows, and the old pads of the kept members. Between two segments the first vector is main row 0 of the carrying message and the second is main row 1. In a stage of a proximity test the first fills the stage's 2^k main rows and the second its 2^h mask rows. A vector that fills 2^b rows has its entry j in row j mod 2^b, at position j div 2^b. The second vector skips the entries that are pads of the new message's own out-of-domain sample. |
| the same | The switch leaves these rows for the new group, in this order. First, the fold's claim, read on the carrying message and on the messages that are sent again, with the fold's target. Then, for each kept member, in the order of sending, each of its old rows of an out-of-domain sample, with the old answer as target: the out-of-domain sample that the member got when it was sent into the group that is being closed, so its latest sending and not its first. Last, for each of the draws in their order, for the main row, the kept rows, the old mask row and the fresh mask row: the generator row of the old code at x (for mask rows at x modulo the mask code's length), on message and encoding randomness as they are in the carrying message, with the folded symbol as target. |

A proximity test:

| where | what the verifier forms |
|---|---|
| proximity test, stage i | The switch's rows (the carried claim; three generator rows for each of the draws) and the s rows of the new message's out-of-domain sample are batched with the powers of theta, in the order: the carried claim, the rows of the out-of-domain sample, the generator rows. |
| proximity test, `BASE` | The last stage is taken as single rows: one main row and one mask row of the last oracle message. The group's one row variable is fixed at 0. |

## 14. Interface and results

This section describes the verifier program `cp_verify` as a command: its arguments and options,
every result with its exit code and last line, what keeps a call without a failed test from ending
with an acceptance, the part-calls and planning calls, the lines of the output, and the one line
that is made for programs. It has no numbered rules: it is written from the program's source. Every
printed word, option and name is in code marks, letter for letter. It needs sections 12 and 13.

### 14.1 The call

    cp_verify <config folder> <ivk folder> <request file> <y file> <proof file> -id <64 hex digits> [-bits <asked>]
              [-no-proof-of-work] [-j <threads>] [-q] [-show-stages] [-next-token]
              [-policy <kind>,<C>,<theta>,<cap>,<s>,<sh>]

One call checks the key file `key.bin`, the leaf file that the proof file names, and the proof file.
The five arguments come first, in this order; the options follow them, in any order.

| argument | what it is | whose it is |
|---|---|---|
| `<config folder>` | the configuration directory, with `policy.txt`, `KL.bin`, `layout.txt` (13.3) | the caller's own copy; the commitment ID names it together with the anchor |
| `<ivk folder>` | the public key directory, with `key.bin` and the leaf files `leaf/<k>.bin` | the provider's |
| `<request file>` | T tokens, then r and e (12.2) | the caller's |
| `<y file>` | the token file: one token, the next token (12.2) | the caller's |
| `<proof file>` | the proof file | the provider's |

### 14.2 The options

A number on the command line is 1 to 7 decimal digits and nothing else: no sign, blank, point or
exponent, and no leading zero but in `0` itself. The numbers of `-policy` can have up to ten digits.
`-bits`, `-id`, `-j` and `-policy` can be given once each. An option without a value can be given
more than once, and that does the same as once. An unknown option, a second one of these four, or an
option without its value gets the usage text, whose last line says what was not accepted (exit
code 2).

| option | value | what it does |
|---|---|---|
| `-id` | 64 hex digits, small or capital letters | the commitment ID that the caller expects. Each stage compares it with the ID of the files (13.5); files with another ID end the call with `NO: model-statement`, and their own ID is not printed (14.7). The caller gets the ID before sending the prompt (not the request file, which is built from prompt and reply: `docs/glossary.md`), from the party that vouches for the model, not through the server that is being checked, best from where the commitment is published, and never from the files that are checked. A value of another form: the line `-id requires 64 hex digits` on standard error, exit code 2. Without the option: 14.4 |
| `-bits` | 0 to 1000000; without the option 128 | the requested bits: the number of bits required of every round's error, proof of work counted (14.3). Above 128, half a digest's bits, the call ends `NO: cp-bits-above-digest` before a file is read. Below 128 every test on the files is made, and 14.4 applies. So 128 is the only value under which a call can end with an acceptance |
| `-no-proof-of-work` | none | the caller says that the tests alone must carry the target. If a round needs its proof of work (8.4, item 22) for the parameter file's target, the line `sec`, every test on the files is made all the same, and the call ends at best with `NOT-A-VERIFICATION` (14.4, cause 4). Where no round needs its proof of work the option changes nothing; that is so too where the parameter file declares proof of work that no round needs. The option does not take proof of work out of a proof: how much a proof has is a matter of the parameter file's lines `work ..`. It applies under whatever parameter file is given, and it changes no number of that file and no test. A whole call and both part-calls take it (14.5); a planning call does not (14.6) |
| `-j` | 1 to 256 | the number of threads of the stage request; the stages key and leaf take an eighth of it each, at least one. It changes no result |
| `-q` | none | leaves out most progress lines |
| `-show-stages` | none | prints the stages' output also where the call does not pass; without the option it is printed only above an ending that can be a pass (14.7). After a rejection that output holds values that were read from the rejected files: it is for finding out which test failed, and nothing in it is to be trusted or copied. Under the name `model-statement` nothing of it is printed even so |
| `-next-token` | none | the caller says that the proof must be about the next token alone: r = T - 1 and e = T. With other r, e: 14.4 |
| `-policy` | `<kind>,<C>,<theta>,<cap>,<s>,<sh>`, with commas and no blanks; the kind is `exact-match` or `clipped-gap` | the acceptance policy that the caller expects (12.5). If the parameter file sets another kind or number, or if the named numbers say nothing, the call ends `NO: policy` (14.3). A value of another form, another word for the kind among them: one line on standard error that begins `-policy requires`, exit code 2 |

### 14.3 The results

The last line of standard output and the exit code go together. Exit code 0 has two last lines,
`YES` and `WITHIN LIMITS: ..`, so a caller reads both the exit code and the last line. Of the
results below, the first that applies is the call's.

| last line | exit code | when |
|---|---|---|
| the usage text, which ends with a line that begins `cp_verify: this command line was not accepted: `, or the one line that 14.2 gives for a malformed `-id` or `-policy`, on standard error | 2 | a bad command line |
| `input-unreadable-<role>: <the file> cannot be read: <reason>`, on standard output | 2 | a file that the caller names is missing, is not a regular file, or cannot be read to its end. This is not a verdict: no stage has started, and there is no line `NO:`. `<role>` is `policy`, `KL`, `layout`, `key`, `request`, `y` or `proof`, and `entry` for the leaf file of the part-call `-leaf` (14.5). `<the file>` names the file by its role, never by its path. The files are read in the order of 13.3, among the tests that are made before the stages, so of this row and the next the one that is met first ends the call. A readable file of a wrong length or content is rejected where it is tested. A configuration directory without the name `layout.txt` is not under this row: the call ends `NO: plan-layout`. Nor is the leaf file of a whole call: its number k is the proof file's word and not the caller's, so a leaf file that cannot be read is a rejection (`cp-entry-missing`) |
| `NO: <name>` | 1 | a test failed (or, under a name that begins `internal-`, the program met a fault of its own before any stage, which is no verdict on the proof: 17.2, item 1), either before the stages (13.8, the table "Before the stages") or in a stage. Before the stages no line names a stage; under the name `policy` the line before begins `policy:` and says why (below). In a stage, `<name>` is the rejection name of the first stage that rejected, in the order key, leaf, request, and the line before is `refused by the stage <stage>`, followed by `: ` and the stage's reason where the call repeats it (14.7). Under the name `model-statement` two lines that begin `id:` are there instead (14.7). A rejection goes before a failure: if one stage rejects and another is killed, the call ends with `NO` |
| `NOT CHECKED: <name>` | 5 | nothing has been verified: the program or the machine failed, or one of the program's guards on itself (13.8) ended a stage or the call. The name is one of three kinds. (1) A name under which a stage ended and which begins with `internal-`, also after a sub-proof's prefix (13.8); eight of these names come at the rare sampled values at which no verifier gives an acceptance (13.9). (2) `internal-stage`: a stage was ended by a signal, could not be started, ended in a way that is none of its own, or printed more than there was memory for. The process that starts the stages ends so itself, in a planning call too, for want of memory, if a thread cannot be started, or if core dumps of the process cannot be switched off. (3) Names of the parent process's guards on itself. `internal-covered-line`, `internal-sums-line`, `internal-id-two-ways`: the stages' printed lines do not fit together, or the ID under which the stages passed is not the caller's. `internal-policy-not-given`: a call without `-policy` came as far as `WITHIN LIMITS`. `internal-policy-kind`: a whole call came as far as its last line under another kind of policy than `exact-match` or `clipped-gap`. `internal-result-line`: a stage's output holds the word `result-v1` (14.8). Under `internal-stage` and `internal-exception-parFor` the machine can be the cause, and a second call can end otherwise; the program can be the cause too, and the parent process cannot tell which (from the programs' source, `verifier/src/cp_verify.cpp`, before `cpNotChecked`). One part of this row is out of the table's order: the last tests under `internal-sums-line`, and those under `internal-policy-not-given` and `internal-policy-kind`, are made only where the call would otherwise end with `YES`, `WITHIN LIMITS` or `PART PASSED`, so they come after the row `NOT-A-VERIFICATION`. In a call with `-no-proof-of-work` the first half of one of these last tests under `internal-sums-line` is also made earlier, before the call adds its own causes of 14.4. If no stage has ended with `NOT-A-VERIFICATION`, and the call's record says that some round needs its proof of work while no stage has ended with `STAGE PASSED WITH WORK`, or the other way round, the call ends `NOT CHECKED: internal-sums-line`, and the line before ends with `the stages' lines on work and the call's record disagree`, the same words as at the later test. Why (from the programs' source): with the option such a call would otherwise end with `NOT-A-VERIFICATION` before it comes to the later test. Such a disagreement is a fault of the program, like the others of this row's third kind |
| `NOT-A-VERIFICATION` | 14 | no test that ran failed, but the call does not count (14.4). The lines before say why |
| `WITHIN LIMITS: clipped-gap C=<C> theta=<theta> cap=<cap> s=<s> sh=<sh>` | 0 | every stage passed every test, no cause of 14.4 applies, and a token of the request was checked (r < T - 1) under a policy of the kind `clipped-gap`, which the caller has named with `-policy`. The five numbers are the parameter file's. The program's source says of this result that it is not a `YES`: it bounds the checked tokens' costs and does not say that each of them is the top choice of its position (12.5) |
| `YES` | 0 | every stage passed every test, no cause of 14.4 applies, and no token of the request was checked under another policy than `exact-match`: the policy is `exact-match`, or r = T - 1 and the call is about the next token alone. Proof of work may have been counted (below), but not in a call with `-no-proof-of-work`. The binding of the weights rests on the theorem on codes, which is taken as given (below) |
| no last line | 15 | the answer could not be written to standard output: it is closed, the disk is full, or nobody reads the pipe while the signal `SIGPIPE` is ignored. One sentence on standard error says so. The caller takes the call as not made and drops whatever part of an answer has arrived. This result takes the place of an acceptance and of every other result that the call returns at its end. A rejection that ends the program where it is found, as those before the stages do, keeps exit code 1, an unreadable file keeps exit code 2, and the process's own three endings under `internal-stage` (above) keep exit code 5: none of these can be taken for an acceptance |

*Proof of work.* Proof of work counts toward the target, and in a call without the option
`-no-proof-of-work` (14.2) it changes neither the last line nor the exit code. So in such a call
exit code 0 does not mean that every round meets the target without its proof of work, and the word
`YES` does not mean it either. Each of the fifteen numbers of the parameter file's lines `work ..`
can be up to 32 (13.7, condition 5), so up to 32 of a round's 128 bits can be proof of work. That g
proof-of-work bits before a verifier's message multiply its round's error by 2^-g is an assumption
of the accounting, and the program prints that it is one wherever one of the fifteen numbers is
above 0. A caller sees how much was counted in the line that begins `proof of work: `. It comes near
the end of every call that ends with `YES`, `WITHIN LIMITS`, `PART PASSED` or `NOT-A-VERIFICATION`:
in a part-call directly before the last line, in a whole call directly before the line `result-v1`,
which is directly before the last line (14.7). It has two forms:

- `proof of work: none counted`: every round's error is at most 2^-sec with its proof of work left
  out;
- `proof of work: up to <n> bits counted in <m> rounds; with it the least round has <a> bits,
  without it <b> bits` (`round` where m is 1): m rounds need their proof of work (8.4, item 22); n
  is the largest of the fifteen numbers of the lines `work ..`; and the largest error of any round
  is 2^-a with proof of work counted and 2^-b with it left out, a and b each cut to two decimals.
  The two need not belong to the same round. a counts each round's declared proof of work, also that
  of a round that does not need it (14.8). Each figure is one round's error in bits, by
  `cp_verify`'s own count (13.7, condition 2): neither 2^-a nor 2^-b is the chance that a false
  proof is accepted.

What decides between the two forms is an exact comparison of every round's error with 2^-sec, in
which neither a nor b takes part. The program prints both in this second form of the line and, in a
whole call, in the line `result-v1`; the call's lines `accounting: ..` have them too. Where a stage
has ended with `STAGE PASSED WITH WORK`, and the call would otherwise end with `YES`,
`WITHIN LIMITS` or `PART PASSED`, one of the parent process's last guards asks that b agree, to
within 0.01, with the least figure of bits in the stages' lines `work: ..` (14.7); if it does not,
the call ends with `NOT CHECKED: internal-sums-line` (the row `NOT CHECKED`). In other calls that
guard does not look at b. No guard looks at a (14.8). In a whole call the rounds are those of the
three files' plans taken together, in a part-call those of that file's plan. The fields `work=`,
`work-rounds=`, `least-round=` and `least-round-without-work=` of the line `result-v1` repeat the
four numbers for a program, and `work=0` there means that none was counted (14.8). A caller who
wants the tests alone to carry the target passes `-no-proof-of-work` (14.2): the call then ends with
an acceptance only beside `proof of work: none counted` and `work=0`. A caller who does not pass the
option gets the same by accepting only beside `proof of work: none counted`, or `work=0`. Whether
proof of work counts is the caller's decision (from the programs' source:
`blocks/frame/check_key.h`, `blocks/frame/check_entry.h`, `blocks/numbers/accounting.h`).

*The binding of the weights.* With exit code 0 and the word `YES` the binding of the weights rests on the theorem on codes,
taken as given (13.7, condition 4), not on the bound within a third of the distance. A
person sees it in the line `commitment: ..`, which ends with `; the table has a record: ..`
(14.7), and a program in the field `commitment-assumed-rounds=` of the line `result-v1` (14.8). `cp_verify` has no
option that excludes it: a caller who has not decided to accept it accepts no answer. A part-call has
neither line; its stage's output has the line `accounting: one-time files: ..` (14.7), and a
part-call that cannot pass shows that output only with `-show-stages`. What else an acceptance
rests on: 17.5, item 19.

*Under the example configurations.* Under the files `policy.txt` of `prover/examples/target_full_f2/` and
`prover/examples/target_small_f2/` some rounds need their proof of
work (17.5, item 3). So a whole call that passes under them prints the second form of the line
`proof of work: ..` and `work=32`; like every whole call that passes, its line `commitment: ..` ends with the clause on a record, and
its line `result-v1` has `record=1` and a number above 0 in `commitment-assumed-rounds`. A call with
`-no-proof-of-work` ends at best with `NOT-A-VERIFICATION` under these files, and a caller who
accepts only beside `work=0` accepts no answer under them.

*The rejection `policy`.* A whole call compares the acceptance policy that the caller expects with
the parameter file and the request before it opens the proof file or the leaf file and before any
stage starts. It ends `NO: policy`, with a line before it that begins `policy:` and says why, in
four cases. In them n = min(e, T - 1) - r is the number of checked tokens (12.2).

1. The parameter file has an sT other than sh - 4, so that its policy's kind is `log-probability` or
   `another-table` (12.5). This is so with or without `-policy`, and also where no token is checked:
   a whole call takes `exact-match` and `clipped-gap` alone.
2. The option `-policy` names another kind, or another C, theta, cap, s or sh, than the parameter
   file sets.
3. The call has no `-policy`, n is above 0, and the file's policy is not `exact-match`.
4. The numbers that `-policy` names say nothing: C is below 2^sh, the least cost of a token, so that
   no cost is read; or theta is below 2^sh, so that no request can meet it; or theta is above C,
   which no parameter file may set; or theta = C and 0 < n <= cap, so that no checked token is
   excluded. The first three of these are tested on the command line alone, before any file is
   opened.

Where a line of the policy or a word of the request that this comparison needs is missing or
malformed, or where the length of the request file or of the token file is not a multiple of 4 bytes
(the test below lets such a length through only where it is left out), the comparison stops without
a result, and by the source a stage rejects the call (13.8, the table "Before the stages", has the
whole list in its second row `policy`). Part-calls and planning calls do not make it. The stage
request asks cases 2 and 3 once more, of the values under which it judges (14.4).

*The lengths of the caller's two files.* Before that comparison, and before it reads a byte of
either file, a whole call tests the length that the file system gives for the request file, 4 (T +
2) bytes, and then for the token file, 4 bytes. A length of 0 or one that is not a multiple of 4
ends the call with `NO: statement-required`, any other wrong length with `NO: statement-range`, and
no line before the last says more. Why: a very large file that is named instead of either is
rejected and not loaded. The stage request makes the same tests, under the same names, on what was
read (12.2, tests 1 and 2). A call that ends here has no line `refused by the stage ..`. Where
another file has a fault that only a later test or a stage finds, the call ends under this name and
not under that fault's. 13.8, the table "Before the stages", says where the test is left out.

*What a script reads.* Exit code 0 alone is not a reason to accept an answer. A script takes exit
code 0 together with the last line `YES` for an acceptance under `exact-match` or about the next
token alone; exit code 0 together with a last line that begins `WITHIN LIMITS: ` for an acceptance
within the limits of the `clipped-gap` policy that it has named itself; and every other exit code
for no acceptance. If it does not count proof of work it passes `-no-proof-of-work`, or it also asks
for `proof of work: none counted`. If it has not decided to accept a commitment that rests on the theorem on codes it
accepts no answer (above). A
call on a request with r = T - 1 ends with a true `YES` about one token and says nothing about the
request's own tokens, so the script also compares what the answer is about with its own values: the
fields of the line `result-v1` are made for that (14.8).

### 14.4 What keeps a call from an acceptance

A call in which no test fails ends with `NOT-A-VERIFICATION` if one of these causes applies. Every
test on the files is made all the same, with these exceptions: without `-id` the files' ID is
compared with nothing (13.5); without the 32 lines of the tables' digests the four public tables are
compared with nothing (13.7, condition 10); and a test build, which never ends in an acceptance
(cause 2), can leave a test out (by `verifier/src/vf_common.h` a test build is one with a
switch of the library for a test or for a malicious prover; by the head of `blocks/fence.h`,
which lists these switches (its list has switches for tests alone), some of them let a
test disable checks, and this page does not trace which tests of `cp_verify` they reach; under
`SERVING_BUILD`, with which `verifier/build.sh` compiles, a build with a listed switch on its
command line does not compile). The parent process's last guards on itself are not made, but for a
part of one in a call with `-no-proof-of-work` in which no stage has ended with `NOT-A-VERIFICATION`
(14.3, the row `NOT CHECKED`; 14.8). Each cause is named in a stage's line `result:` or, where it
belongs to the call as a whole, in a line that begins `the call:`. The stage's line is part of the
stages' output, which such a call prints only under `-show-stages`; after `---- the call ----` the
call repeats the causes from it, in a line `stage <name>: not a verification: <causes>` (14.7).
Cause 1 binds every verifier; causes 2 to 4 are this command's own, and a second verifier need not
have them.

1. A condition of 13.7 is missed, other than 3, 5, 6, 11 and 14, where `cp_verify` rejects. 13.7 has the
   conditions, why each is required, and the name under which `cp_verify` reports it. The
   table below says how the command's options meet them.
2. The program is a test build. Its first printed line begins `TEST BUILD:`.
3. The call has the option `-next-token`, and r = T - 1 and e = T do not both hold (`next-token`;
   asked in the stage request). Here the caller has said that the proof must be about the next token
   alone.
4. The call has the option `-no-proof-of-work`, and a round needs its proof of work (8.4, item 22)
   for the parameter file's target (`no-proof-of-work`; asked by the call as a whole, in a whole
   call and in a part-call). The call asks two things, and either is enough: whether a stage has
   ended with `STAGE PASSED WITH WORK`, and whether the call's record of the accounting says that
   some round needs its proof of work. Why both (from the programs' source): a stage that needs its
   proof of work and ends with `NOT-A-VERIFICATION` does not end with `STAGE PASSED WITH WORK`. The
   cause is one line, which comes after the line on `id-named` where both are printed: `the call:
   not the standard parameter form: no-proof-of-work (the call had -no-proof-of-work, and <m> rounds
   meet the parameter file's target only with proof of work counted)`, with `1 round meets` instead
   of `<m> rounds meet` where m is 1. The number m is that of the line `proof of work: ..` (14.3).
   The cause has no last line, exit code or rejection name of its own, and the line `result-v1` has
   no field for it: a whole call prints `status=14` there, and every other field is as in the same
   call without the option (14.8). Proof of work that the parameter file declares and no round needs
   is not a cause. The rounds are compared with the parameter file's target, the line `sec`, and not
   with the requested bits: under a `sec` above 128 the cause can apply where every round has 128
   bits without its proof of work. Here the caller has said that the tests alone must carry the
   target.

| condition of 13.7 | in a call of `cp_verify` |
|---|---|
| 1, `id-named` | missed by a call without `-id`; named in the line `the call: not the standard parameter form: id-named (the call named no ID: add -id with the commitment ID)` |
| 2, the target of 128 bits | missed by a call with `-bits` below 128; the note says that fewer than 128 bits were requested. Each stage asks it, and the call asks it again |
| 5, `policy-work-most` | not a cause of this subsection: the call ends with `NO` and this name (13.8) |
| 11, the acceptance policy | not a cause of this subsection in a whole call: the call ends with `NO: policy` before the stages (14.3) |
| 11, `exact-match` | the stage request's second check of case 3 of that rejection. Asked only without `-policy`, and only if a token of the request is checked (r < T - 1): the policy's kind is `exact-match` (12.5) |
| 11, `policy-named` | the stage request's second check of case 2. Asked only under `-policy`, and then whether or not a token is checked: the kind and the five numbers that the option names equal those of the parameter file (12.5) |
| 7 to 13 | the names of all missed conditions come after the words `not the standard parameter form:`. `cp_verify` prints `id-named`, `next-token` and `no-proof-of-work` after these words too |

No cause of this subsection reads on which source a closeness round rests (13.7, condition 4): a
call can end with `YES`,
`WITHIN LIMITS` or `PART PASSED` with its key file and leaf file on the theorem on codes. A whole call's line `commitment: ` says how many closeness rounds of the key file
and the leaf file rest on another source than the bound within a third of the distance, and that the table has a record
(14.7). A part-call has no such line: of the record it shows only its stage's line
`accounting: one-time files: ..` (14.7), and a part-call that cannot pass shows that line only with
`-show-stages`.

### 14.5 Part-calls

    cp_verify -key  <config folder> <key.bin> [-id <64 hex digits>] [-bits <asked>] [-no-proof-of-work] [-j <threads>]
              [-q] [-show-stages]
    cp_verify -leaf <config folder> <key.bin> <entry file> [-id <64 hex digits>] [-bits <asked>] [-no-proof-of-work]
              [-j <threads>] [-q] [-show-stages]

A part-call runs one stage alone: `-key` the stage key, `-leaf` the stage leaf, on the leaf file
that is named as `<entry file>`. It is for the committer's self-check and never says `YES`. It needs
no `-id`; one that is given is compared as in a whole call. `-next-token` and `-policy` are not
admitted, and the rejection `policy` (14.3) is not made. `-no-proof-of-work` is admitted, and it is
asked of the one stage and of that file's plan (14.4, cause 4). `-leaf` takes the anchor from the
beginning of `key.bin` and tests only the form of these bytes. It prints a line that begins
`key: NOT CHECKED BY THIS CALL`. Its result counts only beside a `-key` that has passed on the same
`key.bin`, configuration directory, bits and build.

| last line | exit code |
|---|---|
| `PART PASSED: key`, `PART PASSED: leaf` | 3 |
| `NOT-A-VERIFICATION`, `NO: <name>`, `NOT CHECKED: <name>`, the usage text, a line `input-unreadable-<role>: ..`, no last line | as in 14.3 |

Without `-no-proof-of-work`, whether the stage needed its proof of work does not show in the last
line or the exit code: the line `proof of work: ..` directly before `PART PASSED` or
`NOT-A-VERIFICATION` says it (14.3). A part-call prints no line `commitment ID: `, `covered:`,
`commitment: ` or `result-v1` (14.7, 14.8).

### 14.6 Planning calls

    cp_verify -plan id <config folder> <key.bin>
    cp_verify -plan request <policy.txt> <KL.bin> <asked> [-layout <layout.txt>] -prompt <request file> -reply <y file>
              [-shapes] [-fields] [-arrays] [-j <threads>] [-q]
    cp_verify -plan accounting <config folder> <asked>
    cp_verify -plan arrays <config folder> <asked>

A planning call checks no proof. It prints what follows from the public inputs and ends with the
line `PLAN ONLY: no proof was checked`, exit code 4. Files that a stage would reject in 13.4 are
rejected here under the same names, exit code 1, and a file that cannot be read ends the call as in
14.3, exit code 2. A planning call has no stages, so a name that begins with `internal-` ends it as
`NO: internal-..`, exit code 1, and not as `NOT CHECKED`. For want of memory or of a thread, or if
core dumps cannot be switched off, a planning call too ends with `NOT CHECKED: internal-stage`, exit
code 5 (14.3). A bad command line ends it with the usage text, exit code 2, and an answer that
cannot be written to standard output ends it with exit code 15, both as in 14.3. A planning call
does not take `-no-proof-of-work`: a command line with it is a bad one. Beside these endings, a
guard of the library (17.2, item 1) ends a planning call by a signal, after the guard's line on
standard error: the call then has no last line and none of the exit codes of 14.3, since no parent
process is there to report it (from the programs' source, `blocks/base.h`). A planning call does not
make the rejection `policy` (14.3); the calls `-plan request` and `-plan accounting` print a line on
its case 1 (below).

- `-plan id` prints `h_par=` and the line `commitment ID: ` with 64 hex digits, the commitment ID of
  these files as they are, for the one who publishes it. It reads the beginning of `key.bin` for its
  form and checks nothing. Both lines are printed together, after the steps of 13.4 and after the
  beginning of `key.bin` is read (13.5): a call on files that are rejected there prints no digest of
  them. An ID that this call prints for files that were sent to the caller is not one to pass as
  `-id`.
- `-plan request` makes the steps of 13.4, the tests on the caller's two files (12.2) and the error
  accounting of key file, leaf file and proof file together, and prints the plan: the groups of
  codes with their derived numbers, the accounting, each file's greatest length, the conditions of
  14.4 that are missed (never `next-token`, `policy-named` or `no-proof-of-work`, which hang on
  options that a planning call does not have, never `exact-match`, which a planning call does not
  ask, and never `id-named` or the 128 requested bits, which only a call with stages asks), a line
  `statement: <n> words, y at word <i>`, `h_par=`, `configuration=` (the configuration's digest,
  R4.9) and `dw=`. `-shapes` prints every round; `-fields` every field of every file;
  `-arrays` lists the arrays and ends.
- `-plan accounting` prints the same accounting and lengths for a synthetic request of T tokens 0
  with r = 0, e = T and y = 0: they follow from the configuration directory alone.
- `-plan arrays` lists the arrays that the segments commit to (13.4, step 6), one line `array:`
  each, from `policy.txt` and `KL.bin` alone, for the tool that makes a layout file.
- A plan of request or accounting that comes to its end under a parameter file with an sT other than
  sh - 4 prints the line `plan: the policy's kind is <kind>: every whole call under this parameter
  file ends NO: policy`, with `log-probability` or `another-table` for `<kind>` (12.5). That is case
  1 of the rejection `policy` (14.3), which the planning call itself does not make: it goes on to
  its end. A whole call that an earlier test ends, or that cannot read a file, does not come as far
  as that rejection.
- A plan of request or accounting in which a round needs its proof of work (8.4, item 22) prints a
  line that begins `work:`: a checking call under these files will count proof of work (14.3), and
  with `-no-proof-of-work` a whole call under them ends at best with `NOT-A-VERIFICATION` (14.4,
  cause 4). A part-call with that option ends so only if a round of its own file's plan needs its
  proof of work (14.5).
- The lengths are printed in lines that begin `length:`, one for each of key file, leaf file and
  proof file (printed as `key`, `entry`, `proof`), each with two numbers (from the programs'
  source). The first, after the words `at most`, is the file's greatest length. It takes every
  opening by itself: the t sampled positions all differ, so the opening has t columns with their
  salts, and it gives as many digests as the bound of R7.9 allows, with t cut to half the number of
  the tree's leaves. The second is an average. It takes the t positions as independent and uniform
  on the tree's 2^a leaves: an opening then has 2^a (1 - (1 - 2^-a)^t) columns, and the sum over d =
  1 to a of 2^d ((1 - 2^-d)^t - (1 - 2^(1-d))^t) digests. Everything else in a file has one length.
- A plan of request or accounting that comes to its end prints the two lines that begin
  `accounting: one-time files: `, the first on
  the key file's plan and the second on the leaf file's, before its line on the missed conditions
  (14.7 has their words).

### 14.7 The output of a whole call

    ---- stage key ---- <n> thread(s), <seconds> s
    (the stage's lines)
    ---- stage leaf ---- ..
    ---- stage request ---- ..
    leaf number <k>
    (the stage's other lines)
    ---- the call ----
    accounting: ..
    stage <name>: not a verification: <causes>
    the call: not the standard parameter form: <cause>
    commitment ID: <the commitment ID>
    id: named by the caller and equal
    h_par=<64 hex digits>
    weight hiding: NOT CHECKED BY THIS CALL (it rests on how the commitment was made)
    covered: ..
    stage <name>: work: <bits> + <work> (..)
    commitment: ..
    proof of work: ..
    result-v1 ..
    YES

The stages' output is printed, in the order of the stages, when all stages have ended. Without the
option `-show-stages` it is printed only if, at that point, the call can end with a pass: no stage
has rejected, failed (14.3, the row `NOT CHECKED`) or ended with `NOT-A-VERIFICATION`, no guard of
the parent process on a stage's output has failed, a whole call has `-id`, the requested bits are
not below 128, the program is not a test build, and cause 4 of 14.4 does not apply: in a call with
`-no-proof-of-work` no stage has ended with `STAGE PASSED WITH WORK`, and the call's record names no
round that needs its proof of work. So without `-show-stages` a call that ends with `NO` or
`NOT-A-VERIFICATION` prints none of it, and of the calls that end with `NOT CHECKED` only one that a
guard of the parent process ends after that point prints it. Why (from the programs' source): a
caller could copy or trust a value from files that were just rejected, or take a stage's
`STAGE PASSED` for the call's answer. With `-show-stages` it is printed at these endings too, but
not under the name `model-statement` (below). A stage's own lines end with the record of its
accounting, a line `work:` if it needed proof of work, a line `result:`, and one of
`STAGE PASSED: <stage>`, `STAGE PASSED WITH WORK: <stage>`, `NOT-A-VERIFICATION`, `NO: <name>`. The
record of the accounting is the lines that begin `plan-sums:`, `plan-rests:`, `plan-near:`,
`plan-worst:`, `plan-lists:`, `plan-round:`, `plan-union:`. The last of these is printed only where
the plan has a smallness test, so the stages key and leaf can lack it. It prints, for information, a
count for the round `RNG` taken as a union over all its columns; the accounting does not use it, and
no result depends on it. Each stage also prints the ID that it has computed, in a line that begins
`computed commitment ID: `. The stage request's first line is `leaf number <k>`, the number in the
proof file's head. It also prints a line `seal`, a blank and 64 hex digits, the seal of the proof's
first chain, and a line of its own on weight hiding, which begins
`weight hiding: NOT CHECKED BY THIS RUN`.

The lines after `---- the call ----` depend on the result. Where a stage has rejected, the call
prints only that heading, the line `refused by the stage <stage>`, with `: ` and the stage's reason
after it where the call repeats it, and `NO: <name>`. The call repeats a reason only if the stage's
line before its last begins `key: `, `entry: ` or `proof: `, so under the names `key-refused`,
`entry-refused`, `proof-refused` and `reply-differs`. A reason under another name, such as
`parameters-not-admitted` or `plan-layout` (13.8), is in the stage's output alone. The reason is a
constant text of the test that failed; a text that holds a digit, or any character other than
letters, blanks and punctuation marks, is not repeated, so that the reason shows nothing that was
read from the files. A rejection name can end with the name of a call of the parameter file, which
has at most 8 characters (13.8, `policy-call-line`); nothing else in these lines comes from a file.
With `-show-stages` the stages' output comes before the heading: first that of the stages that did
not reject, then that of those that did. Under the name `model-statement` the call prints three
lines and nothing else, with the option too:
`id: these are not the files of the commitment that you named; nothing in them was checked`, `id:
take the commitment ID from your own record of where it was published, never from these files` and
`NO: model-statement`. Why: a caller could copy or trust a value from files that were just rejected,
their ID above all. A rejection before the stages prints no heading: its last line, and under
`policy` the line before it, are all (14.3). Where the call ends with `NOT CHECKED`, the line before
the last says how the verifier failed. Without `-id` the line `id:` reads
`id: not compared: the call named none`, and the word `not-shown-without-id` is printed instead of
the ID, in the line `commitment ID: ` and wherever the stages' output has the ID; such a call prints
that output only under `-show-stages`. Why: a caller who copied the files' ID into `-id` would
compare the files only with themselves. The word is put instead of the ID in a whole call alone. A
part-call needs no `-id` (14.5); without one its line `id:` reads the same, and nothing is replaced:
the stage's output there has the ID that the stage has formed from the files. That ID is not one to
pass as `-id`. In a call in which some round needs its proof of work, this part has one more line
that begins `work:`, printed with the accounting, beside the lines `stage <name>: work:`. In each of
these lines the two numbers belong to one round: of all rounds of the plan that the line is about,
the one with the largest error when proof of work is left out. That error is 2^-`<bits>`, cut to two
decimals, and the round has `<work>` proof-of-work bits.

A whole call that ends with `YES`, `WITHIN LIMITS` or `NOT-A-VERIFICATION` ends with four lines,
always in this order: the line `commitment: ..`, the line `proof of work: ..` (14.3), the line
`result-v1 ..` (14.8) and the last line. A whole call that ends in another way has none of the first
three. The line `commitment: ` is about the commitment's own files, the key file and the leaf file.
It reads `commitment: <n> closeness rounds of the one-time files rest on another source than the bound within a third of the distance; the table has a record: the
binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as
given` (`commitment: 1 closeness round of the one-time files rests ..` where n is 1). A closeness
round is a round whose error has a closeness term; n counts those rounds of the two files' plans
whose term rests on another bound than the one that holds within a third of a code's distance (13.7,
condition 4; section 8): the printed words `another source` mean the theorem on codes (8.4, item 6).
The proof file's own rounds are not counted in it. Closeness rounds of a proof rest on the theorem on codes too, `LIN point` of
the proof's shared chain among them, and at the two files of 8.5 all of them
(8.4, item 16; `plan_lin_shared` in `blocks/numbers/blocks.h`); no condition of
13.7 asks on what they rest. The distance in those words is the code's distance (8.4, item 6). By the programs' source the paper that this protocol builds on proves it
(`THIRD_PARTY.md` names the paper). The printed words do not mean a proof in this repository, or a
proof that a machine has checked. By this page's reading of the
source n is above 0 in every call: both distances of the table are made from an m,
and the source of such a count is "Johnson" at every distance (8.4, items 7 and 16), so the key file's `LIN point` is
counted, and with a table of more than one row the leaf file's `LINK point` too. The line `result-v1` has the field `record=1` directly behind
`commitment-assumed-rounds=` (14.8).

The stages key and leaf each print one
line on their own file's plan, in the stage's output (the first paragraph of this subsection says
when that is printed), and the planning calls `-plan request` and `-plan accounting` print both
lines: `accounting: one-time files: the plan of a key: <n> closeness rounds rest on the theorem on
codes (ePrint 2025/2055, Thm 4.6), taken as given, <m> on the bound within a third of the distance; the table has a record`
(`rounds rest` where n is 1 too), and likewise with `the plan of an entry`. (from the programs' source:
`verifier/src/vf_common.h`, `verifier/src/cp_verify.cpp`)

After the words `accounting: THE ROUNDS:` the accounting's lines have `HOLD`, `HOLD WITH WORK` or
`NO`. Where none of the fifteen numbers of the lines `work ..` is above 0, no line of the accounting
speaks of proof of work as counted: the line `accounting: target: ..` ends with
`no round has proof of work; every list of positions is derived from the target`, and among the
stricter readings at the end there is one line on sums, `all rounds' errors added: ..`, with one
figure, and no line on the largest round error without the proof of work. After the line
`accounting: what stands behind it:` each source of a closeness count (8.4, item 6) has a line with
the number of rounds that rest on it. The line on the first source is always printed. Where no round rests on
the theorem on codes, a line says that none does. The accounting's last line names the unit of all
its figures: one key file's, one leaf file's and one request's proof, not a batch's (8.4, item 22,
quotes it). It is the last line of the accounting's own routine. In a whole call or a part-call one
more line with the same first word follows it: `accounting: asked of this call: <n> bits`, with n
the requested bits. A script finds a line by the words with which it begins, never by counting
lines, and takes the result from the exit code and the last line. It reads the line
`commitment ID: ` only at exit code 0.

No run of a verifier can know how masks and salts were made, so every call that ends with `YES`,
`WITHIN LIMITS`, `PART PASSED` or `NOT-A-VERIFICATION` prints the line on weight hiding.

The line `covered:` says what the call's answer is about: the checked positions, the checked tokens,
and whether position T - 1, which is about the next token, is checked (12.2). It is printed in a
whole call that ends with `YES`, `WITHIN LIMITS` or `NOT-A-VERIFICATION`, also under `-q`. Of the
other calls only one that a guard of the parent process ends after this line has it (14.3, the end
of the row `NOT CHECKED`). Its fields have a fixed order: `r=`, `e=`, `T=`, `C=`, `theta=`, `cap=`,
`s=`, `sh=`, `positions=` (r to e - 1), `tokens=` (r + 1 to min(e, T - 1), or `none` if r = T - 1),
`y=` (`yes` if e = T, else `no`), `next-token=` (`yes` if r = T - 1), `policy=` (the kind that the
parameter file sets, 12.5, whatever `-policy` names: `exact-match` or `clipped-gap`, since a whole
call under another kind has ended with `NO: policy`) and `above-clip=`. The line shows r and e,
which are the caller's own input.

The field `above-clip=` is a number that a caller can compute alone, from the policy's lines and
from r, e and T. Let n = min(e, T - 1) - r, the number of tokens under `tokens=`, and m = min(cap,
n). If C <= 2^sh the field is m. Else it is the least of m and n (theta - 2^sh) / (C - 2^sh),
rounded down, with 0 for the quotient where theta <= 2^sh. It is an upper bound on how many of the n
checked tokens can have a cost above C (12.5), and a token whose cost is above C can be any token.
Why: at most cap costs are above C; each of these counts as C in the sum, every other cost is at
least 2^sh, and the sum is at most theta n. It is 0 under `exact-match`. It is not a limit on the
tokens that can be any token: how much a cost of at most C excludes depends on the model's logits,
which no verifier sees. Under the step table the number is reached whatever the model, if the lines
admit a proof at all: so many tokens of any kind, and the model's top choices, at 2^sh each, for the
rest (from the programs' source).

The request stage's line `result:` has the fields `word=` (the version word as a number), `ms=` (the
commitment ID), `sec=`, `t=` (the number of positions of a segment's list), `asked=`,
`by=largest-round-error`, `T=`, `lim=-`, `C=`, `theta=`, `cap=`, `s=`, `sh=`, `sT=`, `F=`, `leaves=`
(the line `slots`), `place=` (the request's slot in its batch), `card=0`, `judged=1 of 1`, `dj=`
(the seal of the request's first chain), `build=` (a digest of the program's sources, set by the
build script), `dw=` (8, the number of 32-bit words of a digest). The lines `result:` of the stages
key and leaf have `word=`, `sec=` and `asked=`. Every stage's line ends with `pow=`, `pows=`,
`powc=`, the three numbers of the line `work positions`, and a note in brackets, which names the
causes of 14.4, in the stage request also the lines `lidb` and `nminH` with their numbers where the
parameter file has them, and ends with the words `no security level is claimed`. The line holds no
plain function of the caller's request or next token, with two exceptions: it names `next-token` or
`exact-match` where that condition is missed, which tells something of r and e.

### 14.8 The line for programs

A whole call that ends with `YES`, `WITHIN LIMITS` or `NOT-A-VERIFICATION` prints, directly before
its last line, one line that begins with the word `result-v1` and a blank. It says what the answer
is about, as named fields, so that a calling program can compare each field with its own value
before it accepts the answer, and need not take a number out of a sentence. The fields come in the
order of the table below, with one blank between them, and no value holds a blank. Every value is a
number, a string of hex digits, or one of the words that the table lists. The first word names the
line's format. Any
change to a field or to the order requires another word. The stages' lines `result:` (14.7)
are something else. This subsection is written from the programs' source. The line that `examples/accepted-proof` prints, with its ID and build left out
(14.3 says what the example configurations give):

    result-v1 status=0 claim=exact-match id=<64 hex digits> bits=128 work=32 work-rounds=38 least-round=128.00 least-round-without-work=96.23 commitment-assumed-rounds=49 record=1 r=60 e=64 T=64 tokens=61..63 y=yes policy=exact-match C=256 theta=256 cap=0 s=16 sh=8 sT=4 F=24 above-clip=0 leaf=0 stages=key,leaf,request dw=8 build=<64 hex digits>

| field | value | what it is | what a caller compares it with |
|---|---|---|---|
| `status` | `0` or `14` | the exit code with which the call ends, but where the answer cannot be written to its end: then the exit code is 15 (14.3), whatever the field says. A signal can end the process after the line too (`verifier/docs/output.md`, the table of the fields) | only 0 is an acceptance, and only beside exit code 0 |
| `claim` | `next-token`, `exact-match` or `policy` | what the answer is about. `next-token`: no token of the request is checked (r = T - 1), and the answer is about the next token alone, as the top choice of position T - 1. `exact-match`: checked tokens of the request, each the top choice of its position. `policy`: checked tokens of the request whose costs are within the limits of another policy (12.5) | what the caller meant to have checked |
| `id` | 64 lower-case hex digits, or `not-shown-without-id` | the commitment ID that the caller has passed with `-id` and that every stage has formed from the files | the caller's own ID |
| `bits` | a whole number | the requested bits: the number after `-bits`, or 128 without the option. It is the number requested and not a security level | the caller's own number |
| `work` | 0 to 32 | 0 if every round meets the target, the line `sec`, without its proof of work. Otherwise the largest of the fifteen numbers of the lines `work ..` | 0, if the caller has decided not to count proof of work. In a call with `-no-proof-of-work` the line has `status=14` wherever this field is above 0 (14.4, cause 4) |
| `work-rounds` | a whole number | how many rounds need their proof of work (8.4, item 22); it is 0 exactly if `work=0` | |
| `least-round` | a number with two decimals | a, where 2^-a is the largest error of any round, proof of work counted | |
| `least-round-without-work` | a number with two decimals | b, where 2^-b is the largest error of any round, proof of work left out | |
| `commitment-assumed-rounds` | a whole number | the number n of the line `commitment: ` (14.7). The proof file's own rounds are not counted | a caller accepts a line only if it accepts a commitment that rests on the theorem on codes |
| `record` | `1`, a constant of the format | the weight tables have a record, and the binding of the weights rests on the theorem on codes (14.7) | a caller that has not decided to accept a commitment that rests on the theorem on codes accepts no line |
| `r`, `e`, `T` | whole numbers | the last two words of the request file, and its number of tokens | the caller's own r and e |
| `tokens` | `<first>..<last>`, or `none` | the checked tokens, counted from 0: r + 1 to min(e, T - 1); `none` if r = T - 1 | the tokens that the caller meant to have checked |
| `y` | `yes` or `no` | whether the answer is about the next token too (e = T). The token itself is not shown | `yes`, if the caller means the next token that it was served |
| `policy` | `exact-match` or `clipped-gap` | the kind of policy that the parameter file sets | the kind that the caller has named with `-policy` |
| `C`, `theta`, `cap`, `s`, `sh` | whole numbers | the policy's numbers, as the stage request has used them | the five numbers that the caller has named with `-policy` |
| `sT`, `F` | whole numbers | two more lines of the policy, as the stage request has used them | the caller's own sh less 4, and 24 (12.5) |
| `above-clip` | a whole number | as in the line `covered:` (14.7) | |
| `leaf` | a whole number | k, the number of the leaf file that the proof file names and the call has checked | |
| `stages` | `key,leaf,request` | a constant: the line belongs to a whole call | |
| `dw` | `8` | the number of 32-bit words of a digest | |
| `build` | 64 hex digits | a digest of the program's sources, set by the build script. It covers neither the compiler nor its flags, and a tampered binary can print any digest here, so the field does not show which build ran (`verifier/docs/output.md`, the table of the fields; `verifier/build.sh`) | the build that the caller trusts |

*Exit code, claim and last line go together.* With `status=0` and `claim=next-token` or
`claim=exact-match` the last line is `YES`. With `status=0` and `claim=policy` the last line is
`WITHIN LIMITS: `, the word of `policy=`, and `C=` to `sh=` with the same values. With `status=14`
the last line is `NOT-A-VERIFICATION`, and the other fields are filled as at `status=0`.
`claim=next-token` can come beside `policy=clipped-gap`: the next token is then checked as the top
choice, and no token of the request is checked under the policy.

*What a program does with the line.* It accepts only if the exit code is 0, the last line is the one
that it expects, and every field for which the table's last column names something agrees with its
own value. The values with which it compares are its own: it never takes them from the parameter
file, from a response or from the output of the call.

*The fields on proof of work* repeat the line before (14.3): `work`, `work-rounds`, `least-round`
and `least-round-without-work` are that line's four numbers, and `work=0 work-rounds=0` is printed
where that line reads `proof of work: none counted`. The two figures in bits are printed at `work=0`
too. Where the parameter file declares proof of work that no round needs, `least-round` counts it
all the same, beside `work=0`, beside `proof of work: none counted` and, in a call with
`-no-proof-of-work`, beside an acceptance: a caller who wants a figure without proof of work reads
`least-round-without-work`. The two figures are for a reader and decide nothing: each stage compares
every round's error with the target exactly, and a printed figure is a floating-point logarithm of
an error, cut after two decimals and never rounded up, so it can read a hundredth below the number
that was met. To find out whether proof of work was counted, a program tests `work=0`, and does not
compare a figure with `bits`.

*When the line is printed.* Only in a whole call, and only at exit codes 0 and 14, also under `-q`.
A call that ends with `NO` or `NOT CHECKED`, a call with a bad command line or an unreadable file, a
part-call and a planning call print no such line. No other line of standard output begins with
`result-v1`, at any result. As a guard on itself, the parent process does not print a stage's output
that holds the word anywhere, and such a call ends with `NO` if a stage has rejected and otherwise
with `NOT CHECKED` (under the name `internal-result-line` if nothing else has failed first). At exit
code 15 the caller drops whatever part of the line has arrived. At exit code 14 the line is printed
before the parent process's last guards on itself, which are made only on the way to exit code 0, or
to a part-call's exit code 3 (14.3, the row `NOT CHECKED`): that the stages' exit codes, their lines
on proof of work and the call's own record agree, that no call without `-policy` has come as far as
`WITHIN LIMITS`, and that the policy's kind is `exact-match` or `clipped-gap`. So at `status=14` the
fields `claim`, `work`, `work-rounds`, `least-round-without-work` and `policy` have not passed these
guards. No guard compares `least-round`, at either status. One part of the first guard is made
before the line in a call with `-no-proof-of-work` in which no stage has ended with
`NOT-A-VERIFICATION`: that the stages' exit codes and the call's record agree on whether proof of
work is needed (14.3, the row `NOT CHECKED`).

*What the line shows of the request.* It shows r and e, as the line `covered:` does, and nothing
else of the request: no token and no digest of tokens. A caller who passes a call's output on and
wants to keep r and e private leaves out both lines.

## 15. What an acceptance states

An acceptance means that every test of section 13 passed, on the files and under the commitment ID
that the caller named, and that every condition of 13.7 holds. This is so for every verifier. For
`cp_verify` an acceptance is one of the two results with exit code 0 (14.3). `YES` is the first.
`WITHIN LIMITS: ..` is the second: it is the result of a call in which a token of the request was
checked under a `clipped-gap` policy that the caller has named, and the program's source says of it
that it is not a `YES`, since it bounds the checked tokens' costs and does not say that each is the
top choice of its position (12.5). In a call without the option `-no-proof-of-work` neither result,
and so not exit code 0 either, means that every round meets the target of 128 bits without its proof
of work: up to 32 of a round's bits can be proof of work (13.7, conditions 2 and 5), and the line
`proof of work: ..` of the output says how much was counted (14.3). With that option a call ends
with an acceptance only if, by the programs' count, no round needs its proof of work to meet the
parameter file's target (14.4, cause 4). That proof of work before a message counts toward the
target is itself an assumption for these hash chains, by the line that the programs print (8.4, item
22). Nor does either result mean that every round's error rests on a proved bound: a closeness count
has one of two sources (8.4, item 6), and no condition of 13.7 asks the one that the programs'
source calls proved, the bound within a third of a code's distance, so in the groups of a proof file a
count can rest on the theorem on codes, which is taken as given (13.7, condition 4; 8.4, item 6; 17.5, item 3). An acceptance rests closeness counts of the key file and the leaf file, and with
them the binding of the weights, on the theorem on codes as well. The closing line `commitment: ..` and the stages' lines `accounting: one-time files: ..` tell how many (14.7). The
theorem on codes is not all on which an acceptance rests: what else is told in `prover/examples/TARGET-KITS.md`, "The table's record" (17.5, item 19).
Two of the steps of this project's own that that page names bear on a proof
file's rounds and on the binding of the weights too: that a closeness
error beyond a third has no factor for the rows of an interleaved word (17.5, item 4), and
the step from the
theorem's code to a mask code (17.5, item 3). The errors of the other rounds, and the compiled
bound, are by the accounting's own printed line cited from published work, some of it not peer-reviewed, or argued on paper in a technical report that is not public and not refereed (17.5,
items 1 and 2). The unit of the 128 bits, and of every printed figure, is one key file's, one leaf
file's and one request's proof, and not a batch's: counted jointly for a full batch of N requests,
each round of a request's own chains has log2(N) bits less (8.4, item 22; 13.7, condition 2). The
128 bits are the target with which each round's error is compared by itself; the rounds' errors are
not added. The figure that the accounting derives from the compiled bound and prints as one number
(17.5, item 1) is another number, and it is below 128 in a call that ends with an acceptance and has
`bits=128` in its result line (14.8). At `prover/examples/target_full_f2/policy.txt` that figure is
126.21 bits with proof of work counted and 96.23 without; the errors of all the rounds of one key
file's, one leaf file's and one request's proof, added (the comparison with the target does not add
them), give 2^-119.38 with proof of work counted and 2^-92.42 without (all four as a planning call
of `cp_verify` prints them: `-plan accounting`, 14.6; which rounds are added, from rows SO-1 and
SO-3.1 of [docs/limits-all-rows.tsv](../limits-all-rows.tsv)). No printed figure is a security
level. Three causes of the command's own can keep a call from an acceptance (14.4). What follows
from that, and under which assumptions, is told once, on the [protocol page](../protocol.md).

## 16. Duties that no verifier can see

No verifier can tell how a committer or a prover made its private randomness, and nothing in this
section follows from an acceptance. Every proof verifies whether or not these duties are kept, and
keeping them does not show that the weights are hidden. That holds of every item of the list below;
where a verifier can test a part of a duty, the item says so. A value that is used twice, under
other data or other challenges, can reveal hidden values. How things stand on weight hiding:
[limits](../limits.md#weight-hiding). What is declared as not hidden:
[what-is-not-hidden.md](../what-is-not-hidden.md). A provider's conditions, and what breaks each:
[operators-duties.md](../operators-duties.md).

The private random values of the format are of two kinds.

*Values of a stored table:* the random entries of its rows (a row's random tail, R4.10), its leaves'
salts (R7.3) and its pads (R13.8). They are made once, with the table, and
stay with it.

*Values of a run.* A **run** is one commit run for a key file or for a leaf file, or one batch with
the proofs of its requests. Its values are all that the protocol has a committer or a prover sample
in private, above all: the random entries of the rows that it encodes; mask rows (sections 8 and 9);
the pad of each oracle message's out-of-domain sample, which is in a mask row or, where the message
has none, in a main row (section 9); hiding polynomials (9.4, 11.1); the salts of the leaves of its
trees and of the binding tree (R7.3, R7.18); the 32 random bytes of a chain in case (a) of R4.5; the
digest of a leaf of the binding tree at a slot that no request has taken (R7.18); the arrangement of
a batch's requests over the slots of the binding tree (R7.18); and the message of the main row of
the masking word of a proximity test's last step (R12.10), whose mask row and random entries are
among the mask rows and random entries named above (this last one: from the programs' source). The
duties below reach all of these. They also reach a value that this page does not give as the
protocol's: the elements with which the programs of this repository fill the room that a switch
(section 9) leaves free in the rows of the next stage's oracle message (from the programs' source;
the item on free room below says on what that filling rests). One more value is a choice of the
programs of this repository and not the protocol's: they also sample the leaf table that serves a
batch (from the programs' source).

- A committer MUST keep R4.12: a stored table's values are never used under two roots.
- A committer and a prover MUST NOT open more columns of a stored table, in the key file's one-time
  test, all links and all batches together, than its rows' random entries cover (8.4, item 17; 8.5
  has the limit of a leaf table). For that they MUST keep one lasting counter of the opened columns
  of every stored table, the anchor too, and MUST raise it, and make the raise lasting, before an
  opening begins. The programs of this repository keep it in a file for each stored table
  (`count.bin`) and never decrease it (from the programs' source). A batch, a link and a one-time
  test count in full when they are begun on a stored table, whether or not a column was opened or
  sent. That holds of a run that stopped, failed, was withheld or was begun again. It also holds of
  a test or a rehearsal under the same secrets, and on every process and machine that holds them
  (16.1, condition 5; from the programs' source). An operator MUST NOT let a table be opened under a
  second or an older counter. So a secret key directory is never copied, backed up, restored or
  rolled back ([operators-duties.md](../operators-duties.md)).
- A committer and a prover MUST use only a configuration that meets the ten conditions on hiding
  that 16.1 defines (`blocks/frame/hiding_conditions.h`). `cp_verify` rejects a configuration that
  misses one (`parameters-not-admitted`; for part (a) of condition 3, `plan-mask-row`), and no other
  verifier need do so, but for that part (11.1), part (d) of condition 3 and the fourth line of
  condition 6 (why: 11.1 for part (a), 16.1 for the other two). Others of the ten every verifier has
  from another place on these pages, among them conditions 4, 8 and 10 and part (c) of condition 9
  (16.1, "Who else asks each of the ten"). 16.1 also says when a committer and a prover MUST test
  the ten, and that the ten are not all that hiding asks.
- A prover MUST NOT begin a request's second chain (R14.10) a second time on the same batch's Omega
  (section 9). Each further beginning is one more list of positions on Omega, whose tail is sized
  for N such lists and the batch's own (16.1, condition 1; from the programs' source).
- A committer or a prover MUST sample every value of a run afresh for that run.
- A field element among these values, of either kind, MUST be exactly uniform on the field. A word
  that is only reduced modulo p is not. The 32 random bytes of a chain and the digest of R7.18 MUST
  be uniform on their 32 bytes. A prover MUST choose the slots of a batch's requests by an
  arrangement that is uniform among all arrangements and independent of the requests and of their
  order of arrival. A salt need not be uniform, but it MUST be hard to guess. At 40 bytes the
  programs' salts have 320 bits in a run's trees, uniform if the generator's output is, and at least
  304 bits of min-entropy in a stored table (COIN-KEYS.md, section 2a).
- Where later messages take their masks from the entries of an oracle message's mask rows (9.1, item
  3; a segment's mask row, 11.1), a committer or a prover MUST sample every entry of these rows,
  also an entry that no later message takes. The programs of this repository fill such rows whole
  (from the programs' source).
- In a proximity test's last step a committer or a prover MUST sample the whole masking word
  (R12.10): the message of its main row, the message of its mask row, and the random entries of
  both. Why: each of the four parts of the message of the round `BASE` (R12.9) is the masking word's
  part plus lambda times the same part of the last stage's rows, in the clear, where lambda is the
  element that the round `BASE lambda` samples. By the source these sums are uniform, whatever the
  last stage's rows are, if the masking word's parts are uniform and serve this one last step only.
  If the message of the masking word's main row is left at 0, the masked main message is lambda
  times the last stage's message, unmasked, and every proof verifies (from the programs' source:
  `blocks/iop/base_case.h`).
- The free room of a row that is encoded again MUST hold zeros or values that do not depend on a
  secret. Where a switch (section 9) leaves such room in the rows of the next stage's oracle
  message, a committer or a prover MUST fill it with sampled elements, uniform like every other
  field element of a run. That is what the programs of this repository do: they fill this room with
  uniform values, to which the message's out-of-domain sample gives a weight and which no other
  claim reads. Why: one comment in the source (`blocks/iop/cs.h`) gives as the reason that the
  argument for hiding that the switch rests on counts these entries as uniform. The comment names no
  text, and these pages hold no such argument and no proof. These pages do not say whether this
  protocol's technical report covers that filling. The reason is the source comment's alone: that
  hiding needs uniform values in this room is not shown here (`toys/1-product.md`, part 4, section
  11). In a carrying message (section 10; the source calls the switch that sends it CS^sigma:
  `toys/1-product.md`, part 4) the source's honest prover fills such room with zeros. A verifier
  cannot tell the two fillings apart (from the programs' source: `blocks/iop/cs.h`).
- The entries of the main rows of a segment's oracle message that no block of the layout file fills
  (13.4, step 7) MUST be 0, but for the entries of the message's pad (section 9). The programs of
  this repository clear these rows before they lay the arrays (from the programs' source:
  `prover/engine/lay.cuh`; `blocks/iop/req_arrays.h`). Why: an entry left as allocated holds what
  earlier work left in that memory: cells and sums of an earlier request and random tails, salts and
  mask rows of an earlier run (`prover/engine/device.cuh` lists what may lie there); no other claim
  reads such an entry, but the message's out-of-domain sample gives it a weight like every other
  (`blocks/iop/send.h`). No verifier can test these entries, and `cp_prove` tests none of them but
  the pad's, which must be 0 before the pad is drawn. What the zeros mean for hiding nobody has
  worked out ([limits](../limits.md)). At `target_full_f2/layout.txt` all 99 oracle messages have
  such entries: 2.66 % of the entries of their main rows (this page's count from the file, the pads'
  entries among them).
- In a round with proof of work a committer or a prover MUST take the least nonce, counted from 0,
  that passes the round's test (R5.5). Why: that nonce is a function of values that the verifier
  computes too, and of nothing else. A verifier tests only that the nonce passes. The programs of
  this repository take the least (from the programs' source).
- Whether and at which round a committer or a prover stops a run MUST depend only on what the
  verifier has seen and on randomness that is independent of a stored table's pads and random
  entries. What it does when a search for a nonce reaches a cap MUST depend on that fact alone.
  After a failed range check (16.1, condition 7) what a prover sends for that request MUST be fixed
  values. The first two duties are from the programs' source: a comment in
  `blocks/frame/hiding_conditions.h`, in its list of what tests of the code must hold. Of a failed
  range check that comment asks less than the third duty here: that none of that request's proof is
  handed out, and that of its computed values its batch takes in only a salted digest. `cp_prove`
  does not send fixed values: by the programs' documentation (row C3.1.1 of
  [docs/limits-all-rows.tsv](../limits-all-rows.tsv), which gives it as read in the code, with
  nothing run for it) it sets a flag, goes on with the digits of the real combination, cut to their
  ranges, and runs everything to its end; at the batch's end that request alone gets no proof. So
  what it computes for that request after the failure depends on hidden values. By a reading of its
  sources the seal of that request's first chain goes into the batch's binding tree all the same, as
  a salted leaf (`prover/hiding/tail_f.cuh`, `prover/prover/in_step.cuh`; R14.7). This page does not
  say whether that costs anything. By row C3.1.1, that a failed check costs one request alone
  supposes that every rejection by `cp_verify` that depends on data is mirrored by a flag in
  `cp_prove`, and that is shown nowhere: where it fails, the response leaves `cp_prove` as served, and a verifier outside it rejects it (read in the code), and whether a client can bring such a rejection about at will is not known.
- A committer or a prover MUST NOT let anything of its private randomness out except through the
  uses that the protocol gives to a value: no seed or key from which values are derived, no output
  of a generator that is not used, and no attempt that was rejected. In the programs of this
  repository every key below the seed counts as secret as the seed (from the programs' source). One
  option of these programs that is not a test's writes down bytes from which keys are derived: under
  `-keep-for-the-audit`, `cp_commit` keeps the 32 random bytes of the key file's commit run and of
  each leaf file's, and `cp_prove` those of one batch, in a file of the secret key directory, for
  one audit. Hiding is void for a batch whose random bytes are kept, until the audit has erased
  them, and the erasure reaches the file's present blocks and its name alone: a journal, a
  copy-on-write file system, a snapshot, a back-up or a flash medium may hold the bytes afterwards
  (`tools/audit/kept.py`); together with the seed the bytes that `cp_commit` keeps open every mask
  in the key file and the leaf files; and each kept file is a secret like the seed (COIN-KEYS.md,
  section 3; from the programs' source). The audit, called with `--erase-kept-files`, erases a
  batch's kept file at each of its three verdicts, not at a yes alone, so an erased file does not
  free the batch. With `--rehearsal` it erases nothing, whatever word it prints, and it erases
  nothing at an ending that comes before a verdict: a rejected call, an interrupt, output that
  cannot be written (`tools/audit/audit.py`). Once it has come to a verdict and begun to erase, an
  interrupt, or a last line that cannot be written, ends it with no verdict printed, and the file
  can be overwritten or gone. Where the file is not what `cp_prove` writes, or was moved, replaced
  or changed while the audit ran, the audit does not erase it, says so and does not say yes
  (`tools/audit/kept.py`): an operator MUST NOT hand out a response of a batch whose random bytes
  were kept unless the audit has said yes. The files that `cp_commit` keeps are destroyed only when
  every check of the commitment's audit has passed, right before it says yes. At every other ending
  they stay, but where that destruction has begun and stops, or the yes cannot be written: then some
  or all of them are gone, no yes was said, the commitment can no longer be audited and counts as a
  no: nothing of it may be handed out (`tools/audit/audit.py`; COIN-KEYS.md, section 3).
- Each value of a run is sampled for one role in one run: as one random entry of one row, one entry
  of one mask row, one coefficient, one salt. In that role it has the uses that the protocol gives
  it, and they can be several. The random entries of an encoded row enter its code word, which
  openings show in part, and a switch (section 9) places them again, folded or as they are, in the
  message of the next stage's oracle message. An entry of a mask row is in the committed row, and it
  also masks a value that is sent or is a coefficient of a sumcheck's hiding polynomial (9.1, item
  3; 11.1; from the programs' source). A committer or a prover MUST NOT use a value of a run in a
  second role, or in its own role after other earlier messages or challenges.

How the programs of this repository derive these values from one secret seed and fresh bytes of the
operating system is not part of the format. No verifier derives them, and no byte of a file says how
they were made. It is described in [COIN-KEYS.md](COIN-KEYS.md).

*One option of the programs gives these duties up on purpose (from the programs' source).*
`-public-weights <file of 32 bytes>`, an option of a production build's `cp_commit` and `cp_prove`,
is for weights that are public anyway. The commitment's seed is then derived from the file, and no
random byte of the operating system goes into any file of the commitment. Anyone who has the file
can compute every mask of the key file, of the leaf files and of the stored tables' columns that
proofs open, so the weights are not hidden. By the programs' usage texts nothing has been argued on
whether the requests of one batch stay hidden from one another under it; they say to serve other
people's requests under it with one request in a batch, or not at all. By the programs'
documentation no program enforces that ([operators-duties.md](../operators-duties.md)). The key
file, the leaf files and the commitment ID bear no mark of the option, and by `cp_commit`'s usage
text the proofs look like any other commitment's. So from these files a verifier cannot tell such a
commitment from another, and an acceptance does not say which of the two it is. A client cannot tell
from the public files that a commitment was made to hide nothing. An operator MUST NOT use the
option for weights that must stay hidden (`prover/common/public_weights.h`;
`prover/prover/arguments.h`; `prover/weights_commitment/arguments.h`; COIN-KEYS.md, section 2).

### 16.1 The ten conditions on hiding

This part writes out the ten conditions on hiding that 8.4, item 23, names. All of it is from the
programs' source: it follows `blocks/frame/hiding_conditions.h`. Where an item says "the source:",
it gives what the comments of that file say that the condition counts; this page makes no statement
of its own on what is hidden. The part needs 8.4 for the derived numbers, R12.16 for a segment's
group, 11.1 for a segment's scalars and for the smallness test, and sections 9 and 10 for the names
of the oracle messages. The ten are not the ten rows of the table in 13.7 (condition 12,
`routed-experts` to `rotary-table`). Those rows are part of the standard parameter form: every
verifier asks them, and a miss there withholds an acceptance. The ten of this part are `main-rows`
to `batch`, and none of them is part of what an acceptance means (13.7); of some of them that is so
only because another place on these pages asks the same, and part (d) of condition 3, which no other
place asks, every verifier asks as it is written here ("Who else asks each of the ten", below).

*What they are for.* Conditions 1 to 6 say that every row, stored table and mask has at least as
many random entries as values are ever shown of it. Under a configuration that misses one of the
ten, honest programs without any fault can give proofs that verify and that show more than the masks
cover; in the source's words such a proof "passes every check on the proof itself and may reveal
something about the weights". Conditions 7 to 10 are of other kinds, and each says below what it
asks. As `cp_verify` asks them, all ten compare numbers that follow from the public files alone;
condition 5 has three further counts, which only the holder of a stored table knows. No proof shows
whether the ten were met. The source says two things of a configuration that meets all ten: the
written arguments for hiding, which by the source are not in this repository, are made for
configurations that meet them, and meeting them proves nothing by itself. By the source these
arguments speak of those who see no position of a stored table beyond what proofs open: a counter
that stops exactly at the bound leaves no column for any other reading of the table, and the columns
that are not opened are to be guarded by the leaves' salts.

*What the ten leave out.* The ten are not all that hiding asks. After them the source lists sixteen
further conditions that are not arithmetic on parameters and must be covered by tests of the code.
The routine that asks the ten does not ask them. The code of a committer and of a prover has to keep
them, as the holder of a stored table has to keep the three counts of condition 5. This part does
not write them out. Those that this page has as duties of a committer and a prover are in section 16
and in R4.12.

*Whose duty.* The party that must reject is the one that holds the secret. A committer and a prover
MUST use only a configuration that meets all ten conditions, and MUST test them when the source
says: before anything that depends on a secret is shown or stored. The part on the frame lines alone
(below) comes among a program's first lines. The whole ten come before each of three moments: before
a column of a stored table is spent, that is, before the counter of section 16 is raised for it;
before a request is taken; and before a byte that depends on a seed, a weight or private randomness
is written to a file, a cache or a socket. The source does not ask that the whole ten come before a
seed or a weight is read. `cp_prove` asks the part on the frame lines before it reads a secret. It
asks the whole ten after it has read the seed and the weights, with no request taken. The comment at
that place says why the first two moments come later (batches begin in the service, after it) and
gives no reason for the third (`prover/prover/chain_frame.cuh`). That nothing which depends on a
secret is sampled, sent or written before that is this page's reading of the sources alone. At a
miss it ends, whatever the seed: under a test seed only the name of the ending differs. A test seed
makes an exception at the part on the frame lines alone: there a run under a test seed goes on to
where the whole ten are asked, and a miss there ends it (by a reading of the sources:
`prover/prover/inputs.cuh`, `prover/prover/chain_frame.cuh`). A verifier can test the ten.
`cp_verify` does: it rejects a configuration that misses one. By the documentation that rejection
has been seen under two of the ten only (under the others another check of the file rejects first),
and the ten have not been seen to stop `cp_commit` or `cp_prove` (`docs/limits-all-rows.tsv`, rows
T23.1 and HW-228). The name is `parameters-not-admitted` (8.4, item 23, step 6), but for the misses
that the parser finds first (the end of this paragraph) and for part (a) of condition 3, which
`cp_verify` asks earlier under the name `plan-mask-row` (8.4, item 23), and which every verifier
asks (11.1, the room in the mask row). Every verifier also asks part (d) of condition 3 and the
fourth line of condition 6; each says why. Others of the ten a verifier that derives its numbers by
8.4 has from another place on these pages ("Who else asks each of the ten", below, which also says
what holds for a verifier that takes plan, codes and shapes as public inputs). Under the name
`plan-mask-row`, and before part (a), `cp_verify` also asks the count at run time on a proof's index-mask message (the last paragraph of this part). So a call does not show every miss under
`parameters-not-admitted: hiding: `. The parser comes first: a miss of condition 8 ends under
`parameter-range (policy)`, and a miss of part (c) of condition 9 under
`parameter-own-code (policy)` (8.1, tests 18 and 33). A miss of condition 4 or of the fourth line of
condition 6 ends in step 3 of 8.4, item 23, under `parameters-not-admitted` without the word
`hiding:`. By this page's reading of the source, with numbers that are derived as 8.4 says, only
conditions 5 and 7 can be the first miss that a call prints after
`hiding: `.

*Notation.*

- *From 8.4.* B, N, S, w_T, m_T, ceil_log2 and "div" are those of 8.4. So S is the value of
  `answers`. It is not the model's line `S` (8.6), which the table of 13.7 means by S.
- *From the frame lines.* b and i are the first and the third number of the line `table`, and Y is
  the value of `salt`. m0 is the number of rows of the matrix (8.4, item 18; 11.5).
- *A tail.* cov(r) is the number of positions that a tail of r entries covers (8.4, item 4): r in
  field 2; in field 1, r - 1 for an even r > 0 and r for every other r.
- *The nine proximity tests.* The conditions speak of nine proximity tests, each written by a short
  name: the last test of a request's first chain, written `link` (in this name the word `link` means
  a request's segment: 8.2); the tests on the groups `folded`, `omega`, `own`, `once-folded`,
  `once-omega`, `once-own` and `once-ranges`; and the test of a leaf file's link.
- *A stage.* Take stage j of a proximity test X, where stage 1 is the group that is tested. r(X, j)
  is the tail of its base code. ell_mask(X, j) and r_mask(X, j) are the number of message entries
  and the tail of its mask code. k(X, j) and h(X, j) are its two numbers of R12.4. t(X, j) is the
  number of positions that its round samples, `CS` or, at the last stage, `BASE`: the eighth word of
  that round's entry (8.4, item 21).
- *A segment.* For segment g of a request, t_g is the number of positions that the round `LOC` after
  its closing samples, and k_g and h_g are the numbers of its own group (R12.16). n_g is the number
  of scalars of its sub-proof (11.1); the next token of `REPLY` is not among them.
- *Columns.* c_batch, c_once, c_anchor and c_leaf are the four numbers of columns of 8.4, item 17:
  of a batch, of the key file's one-time test, and of a leaf file's link on the anchor and on the
  leaf table. lim is the limit of a leaf table (8.4, item 17).
- *The two families.* The conditions on the test on the table are asked twice: for a proof, with the
  groups `folded`, `omega`, `own` and R = N, and for a key file, with `once-folded`, `once-omega`,
  `once-own` and R = 1. Below, the three short names stand for either family.
- *The split case.* The **split case** is the case in which `folded` and `omega` differ in n or in l
  (8.4, item 18).

1. `main-rows`. For every stage of every proximity test: cov(r(X, j)) >= t(X, j) + a(X, j). For
   every segment g: cov(r(`link`, 1)) >= t_g. Here a(X, j) = 0 for j > 1, and a(X, 1) is in the
   table after condition 2. The source: the random entries of every main row cover all sampled
   positions of all lists at which that encoding of the row is opened. What is counted is the draws
   (R6.6), repeats kept, and the counts of all lists on one encoding of a row add up. A kept message
   is sent again, with fresh random entries, in every segment that reads it, so each encoding is
   read at one list. Omega is read by the batch's test and by every request's own test.

2. `mask-rows`. For every stage of every proximity test: cov(r_mask(X, j)) >= t(X, j) + z(X, j). For
   every segment g: cov(r_mask(`link`, 1)) >= t_g. Here z(X, j) = 0 for j > 1, and z(X, 1) is in the
   table. The source: the same as condition 1 for every mask row, under its mask code; the shared
   split's mask message is read at both groups' lists.

   | proximity test X | a(X, 1) | z(X, 1) |
   |---|---|---|
   | `folded` | 0 | t(`omega`, 1) |
   | `omega` | R t(`own`, 1) | t(`folded`, 1) |
   | `own` | t(`omega`, 1) + (R - 1) t(`own`, 1) | 0 |
   | `link`; `once-ranges`; a leaf file's link | 0 | 0 |

   Outside the split case there is no test `omega`. Then a(`folded`, 1) = R t(`own`, 1),
   z(`folded`, 1) = 0, a(`own`, 1) = t(`folded`, 1) + (R - 1) t(`own`, 1) and z(`own`, 1) = 0.

3. `mask-room`. It has five parts. Write J(x) = 1 + 2 w_T if x > 1 and J(x) = 0 else. P is the
   number of parts of all lookups of the key file's proof of ranges together (8.4, item 20). The
   source: every message from which later steps take mask entries has room for them, each taken
   once, and for its own pad.
   - (a) *A segment's mask row.* For every segment g: ell_mask(`link`, 1) >= n_g + S. The source:
     every scalar that a sumcheck with a hiding sum sends is masked by one entry of the segment's
     mask row; S entries are the mask row's own pad. n_g counts every scalar of the segment's
     sub-proof, also the end value of the merge, whose mask is an entry of the index-mask message
     (11.1).
   - (b) *The index-mask message of a proof*, with its Q mask rows and the u pending rows of a
     request (8.4, item 18): Q ell_mask(`own`, 1) >= u + J(u) + S. The source: one entry for each
     pending row; the hiding polynomial of their batching, 1 + 2 for each column variable, if there
     are several rows; the pad.
   - (c) *The index-mask message of a key file*, with its own Q: Q ell_mask(`once-own`, 1) >=
     max(P, 1) + P + J(P) + S. That is 2 P + J(P) + S if P >= 1, and 1 + S if P = 0. The source:
     there the pending rows come from the proof of ranges, one for each part of each lookup, and
     each takes a second entry; where there is no part, one pending row is counted.
   - (d) *The shared split's mask message*, in the split case, for a proof and for a key file. It
     has one mask row: ell_mask(`folded`, 1) >= 1 + S. By this page's reading of the source: one
     entry, and the pad. By this page's reading of the source nothing else asks this part, there or
     on these pages. So every verifier asks it, like part (a), and rejects public files that miss
     it. Why: the row that is split between two groups names the first free entry of this message
     (13.9, the round `SPLIT`). One row of 2^lz entries has 2^lz - S free entries (section 9, the
     free entries), so a message that misses the part has none, and these pages define no row there.
     `cp_verify` rejects such files with the ten, under `parameters-not-admitted`. Beyond that test
     the library has only a guard, which ends the run where an entry is taken and none is left
     (`blocks/iop/mask.h`); a run that a guard ends gives `NOT CHECKED` (17.2, item 1). With S = 1, the one value that the parser takes (8.1, test 14), the part cannot be missed: the
     mask message of stage 1 has at least 2 k + 1 - h entries, and 2 k + 1 - h >= 3 (8.4, item 23, the third of the
     further conditions; here k >= 1 and h = 0, and the two lines have one lz: 8.1, test 34).
   - (e) *The multiplicities' message* of a key file, with its G mask rows (8.4, item 20): G
     (ell_mask(`once-ranges`, 1) - S) + G - 1 >= x_all - 2 P, where the left side counts as 0 if G =
     0 or ell_mask(`once-ranges`, 1) <= S. x_all is the sum of x over the lookups, and x is the
     number of elements of E that all rounds of one lookup send (R13.17). Take a lookup with w
     planes (R13.17), the words d and mu (R13.14) and parts of v_1, v_2, .. variables (8.4, item
     20). Then:

     x = 7 + 6 (w - 2) + 5 (the number of its parts - 1) + the sum over its parts of (v^2 + 4 v) + (d^2 + 4 d + 2 mu d) + (1 + mu + 2 d) + (4 + 2 (mu + d)).

     The first two terms are what step 2 of R13.17 sends; the other five are what steps 3, 5, 6, 7 and 9 send, in this order. The source: every scalar that a lookup's sumchecks send is masked by one entry. Two for each part are masked by entries of the index-mask message, all others by the mask rows of the multiplicities' message. Each such row gives up the entries of the pad's positions, and all rows but one have one more free entry.

4. `fold-room`. For every stage of every proximity test: ell_mask(X, j) >= 2 k(X, j) + 1 - h(X, j).
   For every segment g: ell_mask(`link`, 1) >= 2 k_g + 1 - h_g. The source: the hiding polynomial of
   a fold over k row variables, h of them shared with the mask rows, fits a mask row: 2 k + 1 - h
   entries. k and h of stage 1 follow from the rows of the group's members: 8.4, items 15, 18, 19
   and 20; the main rows of the multiplicities' message are under R13.14.

5. `table`. cov(m_T) >= max(q_once c_once + q_anchor c_anchor, q_leaf c_leaf) + lim c_batch. q_once
   says how often the one-time test was begun on the anchor. q_anchor says how often a link was
   begun on the anchor. q_leaf says how often a leaf table's own link was begun on it. The source:
   the columns of the one-time test, of the links on this stored table and of all batches that the
   counter admits are covered by the table's random entries.

   The three counts are not facts of the public files. So a test on the public files alone says
   nothing of an anchor on which links were begun: only the counter of section 16 can guard that.
   With the true counts, taken for each stored table by itself, and with the number of batches that
   its counter admits on it in lim's place, the condition is what that counter must keep at run
   time. A batch, a link and a one-time test count in full when they are begun on a stored table,
   whether or not a column was opened or sent (what else counts as begun: section 16, the item on
   the counter).

   On a configuration alone the programs take q_once = 1, q_anchor = 0 and q_leaf = 1. Since c_leaf
   = c_once (8.4, item 17), the condition is then cov(m_T) >= c_once + lim c_batch, and by the
   definition of lim it holds exactly if cov(m_T) >= c_once. lim is formed without a link on the
   anchor. So a stored table that serves lim batches may lack the room for a link as well. At
   `target_small_f2/policy.txt` and at `target_full_f2/policy.txt` the counts 1, 1, 1 miss the condition. The miss is condition 5's single
   formula's: it lays the anchor's links and a leaf table's batches on one stored table. The
   programs of this repository begin no batch on the anchor (`prover/common/cp_files.h`; 13.2), and
   their counters count each stored table by itself (section 16). Counted so, the counts 1, 1, 1 fit
   at both files: at `target_full_f2/policy.txt` the anchor opens 62 + 62 = 124 columns (one
   link; each further one 62 more) and a leaf table 62 + 688,127 x 64 = 44,040,190 (the columns'
   numbers: 8.5), of cov(m_T) = 44,040,191 (this section's table). No rule of these pages keeps a
   batch off the anchor (13.2): for a prover that begins one there, condition 5's single formula
   stands.

6. `padding-entries`. For a proof and for a key file. rows_F, rows_O and rows_W are the numbers of
   main rows of the folded row, of Omega and of W (8.4, item 18), and l_F and l_O are the l of
   `folded` and of `omega`.
   - rows_F 2^l_F >= i + m_T + S.
   - rows_O 2^l_O >= m0 + S.
   - rows_W 2^l_O >= 2^w_T + S if W has no mask row, and rows_W 2^l_O >= 2^w_T if it has one.
   - i >= m0 2^(w_T - b) + 3 S.

   The source: an oracle message of the test on the table has room for one entry for each answer to
   its out-of-domain sample, beside its data. A row of the stored table has three
   entries off the weights' layout for each answer. The source adds: where W has a mask row, W's pad
   is in that row, and this condition does not test that row's room for the pad; a run ends at a
   guard where the row is too short (`blocks/iop/send.h`).

   By this page's reading of the source the fourth line carries more than hiding, and every verifier
   asks it: it is also the first of the further conditions of 8.4, item 23, which are not among the
   ten (`blocks/iop/lin.h`: the matrix fits the table's rows beside their pads). Why:
   a leaf file's link compares the plain entries of the two folded rows alone (R13.8; under R13.10).
   Weights at the positions of a leaf table's pads would be tied to the anchor, and so to the
   commitment ID and to the key file's proof of ranges, by nothing. A
   verifier that takes its shapes as inputs has it as a condition on them (section 18).

7. `range-checks`. A **range check** is seven numbers: rows, columns, combinations, bound, range and
   two numbers of digit bits d_0, d_1. A request has a list of them. In the programs the list has
   one range check for each matrix of each segment's smallness test (number mu in 11.1's list; not
   the mu of condition 3), the matrices of earlier segments among them (third piece of 11.1): rows =
   m_mu, columns = k_mu, combinations = tau, bound = 129, range = 2^17, d_0 = 10, d_1 = 8. The
   condition is missed if the list is empty. A range check is **in form** if all of these hold:
   - rows, columns, combinations and range are at least 1;
   - bound is at least 2;
   - range and rows are below 2^40, bound and columns below 2^32, combinations below 2^16;
   - d_0 + d_1 <= 41;
   - 2^(d_0 + d_1) = 2 range.

   Its bits are: with y = (14426 range^2) div (10000 columns (bound - 1)^2), they are 2^30 if y >=
   2^30, and else y - 1 - ceil_log2(rows) - ceil_log2(combinations). The condition is missed if a
   range check is not in form or has bits below 0. Let least be the least bits of the list, and
   tables = 1 + (cov(m_T) - min(cov(m_T), c_once)) div max(1, c_anchor). The condition is: least -
   ceil_log2(the number of range checks) - ceil_log2(N) - ceil_log2(max(1, lim)) -
   ceil_log2(tables) >= B.

   The source: an honest prover fails a range check when a combination of cells leaves the range of
   the digits, [-range, range). The source counts the chance of this in the error of hiding (its
   words: the zero-knowledge error). The condition asks that the chance of this, over all requests
   that one commitment can serve, is at most 2^-B by the bound below. Every cell is taken to be
   below bound in absolute value, whatever request a client sends; by the source that is not checked
   there (in the programs a cell is a signed byte: `prover/hiding/tail_f.cuh`). A coefficient is 0
   with probability 1/2, and 1 or -1 with probability 1/4 each. So a combination of C = columns
   cells is at least D = range in absolute value with probability at most 2 exp(-D^2 / (C (bound -
   1)^2)). 1.4426 is below log2(e). The subtracted terms are a union. The first two are over a range
   check's rows and combinations. The next is over a request's range checks. The last three are over
   the requests that one commitment can serve: the requests of a batch, times the batches that a
   stored table can serve, times the number of stored tables, which is 1 plus the number of links
   for which the anchor's tail has room after its one-time test. Each factor is rounded up to a
   power of two by itself.

8. `salt`. Y >= 40. The source: a leaf's salt has at least 40 bytes. The parser's test 18 (8.1) asks
   the same.

9. `leaves`. The source: opening a leaf shows all of it, so oracle messages share a leaf only if
   they are read at exactly the same lists. The leaf classes are words of the shapes (R12.12, R14.4
   (a), R13.13, items 4 and 13). The condition has three parts; (b) and (c) are asked for a proof
   and for a key file.
   - (a) The leaf class of a request's index-mask message differs from the leaf class of every
     message of every segment.
   - (b) In each of the three prover's messages of the table below, two oracle messages with one
     leaf class are read at the same set of lists.
   - (c) W and Omega are under one base code.

   | the prover's message with | oracle message | read at the list of stage 1 of |
   |---|---|---|
   | the folded row | the folded row; the fresh mask of the first fold of the test `folded`, where it is sent ahead (R13.13, item 16) | `folded` |
   | Omega | Omega | `omega` and `own` |
   | Omega | the shared split's mask message | `folded` and `omega` |
   | Omega | every other oracle message | `omega` |
   | W | W; the fresh mask of the first fold of the test `own`, where it is sent ahead | `own` |

   Outside the split case read `folded` for `omega` in the last column, and there is no shared
   split's mask message. So in the split case Omega's leaf class differs from that of the shared
   split's mask message, and no other oracle message of that prover's message has Omega's leaf
   class. The source: no mask of the batch's test on Omega may be in Omega's leaves, because every
   request's own test reads them. The leaf classes that the programs write (R13.19; R14.4 (d)) meet
   the condition.

10. `batch`. The number of requests for which 8.4, item 18, sizes the three groups is N for a proof
    and 1 for a key file. The source: the test on the table is sized for as many requests as a batch
    may hold.

*The form of the plans.* The programs take every t from the plans, and they report a miss under the
name in parentheses if a plan has other rounds that sample positions than these. The shared chain of
a proof or of a key file: the `LOC` of the stored table's columns, then one round for each stage of
`folded`, then in the split case one for each stage of `omega` (`main-rows`). The second chain: one
for each stage of `own`, and at least one (`main-rows`). A key file's one-time chain: one for each
stage of `once-ranges` (`main-rows`). A leaf file's chain: the round `LINK`, then one for each stage
of its test (`main-rows`); and `LINK` samples c_anchor + c_leaf positions (`table`). A request's
first chain: one `LOC` for each segment, then one for each stage of the last test (`main-rows`). The
lookups of the proof of ranges send at least 2 P elements (`mask-room`). The plans of sections 9 and
10 have this form. These tests are not conditions on hiding, although they have the names of the
ten: they say that plan and shapes fit each other. A plan that is derived as sections 8 to 10 say
passes them; for a verifier that takes the plan as an input they are conditions on that input
(section 18).

Where each of these is asked, within the order that follows below: that on a request's first chain
before the first segment; that on the one-time chain before the stages of `once-ranges`; that on the
lookups' elements just before 3 (e); that on the shared chain and the second chain after condition
10 and before condition 6; that on a leaf file's chain before the stages of its test; and that on
what `LINK` samples just before condition 5.

*The order, and what is printed.* The programs stop at the first miss. They ask in four steps, and
in each step in the order of its list. The numbers in the lists are those of the conditions.

1. On a request's first chain:
   - 8;
   - 9 (a);
   - 9 (c) and (b), for a proof and then for a key file;
   - for each segment in order: 1, 2, 4, 3 (a);
   - the stages of `link` in order, each with 1, 2, 4.
2. On the proof of ranges:
   - the stages of `once-ranges`, each with 1, 2, 4;
   - 3 (e).
3. On the stored table:
   - for a proof and then for a key file: 10; then 6 in the order fourth line, folded row, Omega,
     W; then 3 (b) or (c); then 3 (d); then the stages of `folded`, `omega` and `own`, each with 1,
     2, 4;
   - the stages of a leaf file's link;
   - 5.
4. Last, condition 7.

The sentence that `cp_verify` prints begins `parameters-not-admitted: hiding: `, then has the
condition's name and a colon; where two numbers are compared it ends
`<have> where <need> are needed`. In condition 7 a left side below 0 is printed as 0.

*The part on the frame lines alone.* The source has a first part for a program's first lines, before
any shape of a model is known. It is the third step of the order above, with 8 and 9 (c) and (b)
before it, on these numbers: m0 = (i - min(i, 3 S)) div 2^(w_T -
b); u = 1; Q = 1 for a proof and for a key file; and P = 0, which gives 1 + S as the right side of 3
(c). It first derives its numbers by 8.4; if 8.4 rejects them, it reports the sentence of that
rejection after the words `parameters-not-admitted: `. By the source's comment at this part it does
not replace the whole (`blocks/frame/hiding_conditions.h`). When the whole ten are due is told under
"Whose duty".

*Beside 8.4.* The source derives nothing a second time. It takes each t from the plans, the rows
that a list reads from the shapes, and each tail from the codes, and compares. Numbers that are
derived as 8.4 says meet conditions 1, 2 and 10 by their derivation: items 13 to 19 size every tail
for the sums of conditions 1 and 2, and item 21 rejects a plan whose round samples more (for `LOC`:
the number without the margin is never the larger). Condition 4 is also among the further conditions
of 8.4, item 23: it is the third and the sixth of them.

*Who else asks each of the ten* (by this page's reading of the source). Some of the ten concern the
committer and the prover alone, and a verifier that leaves them out loses nothing of what an
acceptance means. The others a verifier that derives its numbers by 8.4 has from another place on
these pages, all but one.

- Conditions 1 and 2: the committer's and the prover's alone. They ask only that a tail is long
  enough for the positions that are opened; a round's error is counted on the code as it is.
- Condition 3, part (a): every verifier asks it (11.1, the room in the mask row).
- Condition 3, parts (b), (c) and (e): every verifier has them by deriving the numbers of mask rows
  as 8.4, items 18 and 20, say, by the count at run time (the last paragraph of this part). Why: a row
  names the next free entry, and there must be one.
- Condition 3, part (d): nothing else asks it, so every verifier asks it as it is written above.
- Condition 4: every verifier asks it as the third and the sixth of the further conditions of 8.4,
  item 23. At the last stage k = h = 0, and the condition asks one entry. Why: the verifier puts
  weights on 2 k + 1 - h entries of a mask row.
- Condition 5: the committer's, the prover's and the table's holder's alone. Outside the ten no test
  of a verifier reads lim (`cp_verify` prints it, in a planning call of request or accounting:
  `a stored copy bears <n> batches`, `blocks/frame/prints.h`; no decision depends on the print), and
  no verifier forms anything from m_T but the table's code and the length of the folded row's vector
  (8.4, item 18).
- Condition 6, first three lines: every verifier has them by deriving the numbers of main rows as
  8.4, item 18, says. The rule itself is stricter: a vector ends before the first entry of its
  message's pad (section 9), and for the folded row the vector has the length that item 18 calls
  entries, which is not below i + m_T.
- Condition 6, fourth line: every verifier asks it as the first of the further conditions of 8.4,
  item 23 (above).
- Condition 7: the committer's and the prover's alone. It bounds the chance that an honest prover
  fails a smallness test. What a verifier asks of tau and of a matrix's number of columns it asks
  under other names (11.1, the tests on the public numbers: `hid-tau`, `policy-tau-hb`, `plan-tau`,
  `hid-k-range`), and bound, range and digit bits have fixed values in the programs' list (condition
  7, above).
- Condition 8: by its nature the committer's and the prover's; every verifier asks it all the same,
  as the parser's test 18 (8.1).
- Condition 9, parts (a) and (b): the committer's and the prover's alone. What shares a leaf is
  committed together in either case, and only what an opening shows changes. The leaf classes are
  derived or constant (R13.19; R14.4 (a), (d)), so no public file sets them, and the tests on a
  request's shape ask part (a) once more (`blocks/iop/req.h`). The tests on the shape of the test on
  the table ask a share of part (b) once more: in the split case Omega's leaf class is not that of
  the shared split's mask message (`blocks/iop/lin.h`; section 18, "What these pages have here
  alone", under `lin_fault`).
- Condition 9, part (c): every verifier asks it as the parser's test 33 (8.1), with the one tail
  that 8.4, item 18, gives to `omega` and `own`. With two base codes the test on the table would
  have a further split, which these pages do not define (under R13.19).
- Condition 10: a rule of derivation more than a test, and every verifier's: N, by which the error
  of `LIN point` is multiplied (8.4, item 16), is the number of leaves of the binding tree, the
  value of `slots`, and 1 in a key file. No public file can miss it.

A verifier that takes plan, codes and shapes as public inputs has nothing from a derivation or from
the parser: section 18 lists what it must then ask or assume.

*At the example files.* All ten conditions hold at these parameter
files of field 2, but for what is said below of conditions 9 and 10. Under `blocks/frame/examples/`:
`record.f2.txt`. It has no
model's lines; of it the library's test asks that it misses none
(`blocks/frame/tests/hiding_conditions_test.cpp`). The numbers of a model are here those of the
stand-in shapes of `blocks/frame/files.h` and of `blocks/frame/tests/hiding_conditions_test.cpp`, of
which, by `blocks/frame/files.h`, nothing is claimed but that the number of the matrix's rows is
from a public model's shape: 25 segments; m0 = 3,076,096; and 25 range checks, each of 2^14 rows,
2^16 columns, 128 combinations, bound 41, range 2^17 and digit bits 10 and 8. Under
`prover/examples/`: `policy.txt` in `target_small_f2/` and in `target_full_f2/`.
For these two the result is this page's own count by the formulas of this part, from the
numbers that section 8 derives. Parts (a) and (b) of condition 9 were not evaluated: no public
file sets a leaf's class. Of condition 10 the count shows only that the tails are sized for 64
requests in a batch and for one in a key file. This page's count was held against no other file of
this repository. The equalities
below and the known answers below are this page's own count by the
same formulas. At the stand-in shapes
`blocks/frame/tests/hiding_conditions_test.cpp` has 214, the terms 5, 6, 20 and 20, and 44,040,191.
This page gives no result for the other parameter files under `prover/examples/`, those of
`target_small_f2_roomy/` and `target_small_f2_gap/` among them. At these three files every
inequality of conditions 1 and 2 holds with equality. At `target_full_f2/policy.txt` condition 5
holds with one column to spare, and at `target_small_f2/policy.txt` the first line of condition 6
holds with equality. The fourth line of condition 6 reads
524,288 >= 475,139 at `target_small_f2/policy.txt` and 402,653,184 >= 393,740,291 at
`target_full_f2/policy.txt`. Some values, as known answers:

| value | `target_small_f2/policy.txt` | `target_full_f2/policy.txt` |
|---|---|---|
| segments; the largest k_g, with h_g | 5; 9, 0 | 25; 8, 0 |
| the largest n_g; ell_mask(`link`, 1) | 3,204; 8,192 | 4,190; 8,192 |
| u; Q of a proof; u + J(u) + S | 3; 1; 27 | 13; 1; 51 |
| P; Q of a key file; 2 P + J(P) + S | 8; 1; 40 | 24; 1; 86 |
| the lookups: parts, v, w, d, mu | 4, 25, 28, 8, 5 and 4, 24, 27, 13, 5 | 12, 36, 40, 8, 6 and 12, 35, 39, 13, 6 |
| x_all - 2 P; G; G (ell_mask - S) + G - 1 | 6,573; 7; 7,167 | 34,885; 9; 36,863 |
| cov(m_T); c_once + lim c_batch | 65,535; 65,506 | 44,040,191; 44,040,190 |
| range checks; the largest number of columns; least | 33; 2,048; 717 | 176; 7,200; 186 |
| the four subtracted terms of condition 7; the result | 6, 6, 11, 11; 683 | 8, 6, 20, 20; 132 |

*The room of masks at run time.* Parts (b) and (c) of condition 3 are written otherwise than the count of free entries at run time, which is Q (ell_mask - S) + Q - 1 (section 9, the free entries). The two differ
by (Q - 1) (S - 1), so with S = 1, the one value that the parser takes, the two are the same. The library keeps a guard
all the same: the routine that gives out the free entries stops when none is left
(`blocks/iop/mask.h`). `cp_verify` asks this count
of a proof's index-mask message under the name `plan-mask-row` (8.4, item 23). Part (e) is that count itself.

## 17. Limits of this page

This section says what this page does not define (17.1), where it is known to differ from the
programs of this repository (17.2), and what to know about the scripts beside it (17.3). Then it
lists limits of the programs that these pages
rest on (17.5). The items of 17.1 to 17.3 were found by reading the programs' source; only the one
on a file's length (17.2, item 2) was also tried in runs. The release's list of what is not proved
is in [docs/limits.md](../limits.md), and what is not hidden is in
[docs/what-is-not-hidden.md](../what-is-not-hidden.md). This section lists what is known. That it
does not name a rule or a section does not mean that the rule or section was compared with the
programs.

### 17.1 What this page does not define

1. *The content of a segment's sub-proof.* Section 11 gives every round of a segment by name, what
   each committed cell and each sent scalar holds, how a lookup's keys are formed, and every claim
   as a formula, in the order of the claims. All of it is written from the programs' source: no
   numbered rule covers it. It was compared with the source formula by formula. It was compared with no proof file: no second verifier written from this page has
   decided on a proof of the programs. Section 11 does not give the arguments why the claims imply
   the model's integer function, or an honest prover's choices where a verifier accepts several
   values.
2. *Three facts that no numbered rule fixes.* This page gives them from the programs' source alone:
   the values of the leaf classes in `the link`, `the test on the table` and `ranges` (under
   R13.19); the order in which the codes of a file's proximity tests enter `codes` (R12.2; under
   R13.5 for a key file, under R14.3 for a proof); and the lengths of the four parts of a `BASE`
   message (under R12.9; 10.6). The first two are hashed, so another choice gives other challenges.
3. *Sizes and plans.* No numbered rule covers what is derived from the parameter file: section 8
   describes what the programs compute. Two things are told only in part: how the shapes of a
   segment's oracle messages follow from the model's lines (8.6, 13.4), and the errors of the
   rounds, of which 8.4 gives those that sizing uses, those of `LIN point` and `LINK point`, and
   those of the rounds that sample positions (item 22). Nor does this page say which round's error,
   if any, covers a test that has no round of its own (13.6). So this page does not decide whether a
   plan meets its target or needs its proof of work; a second verifier takes the plan, the codes and
   the shapes as inputs (section 18, which also says from whom). No first query contains a round's
   error: the eleven words of a plan's entry (R4.3) have none. So the hashed bytes do not show
   whether a plan was derived for the shapes that it runs on. By the programs' source (the comment
   before `run_verifier` in `blocks/chain/closed.h`), the library enforces that a stored table has
   passed its one-time test at the distances and under the setting that a later test assumes, and it
   enforces this inside one process only. It enforces nothing across processes or files: `cp_verify`
   checks key file, leaf file and proof in three processes, of which the stages leaf and request
   work on tables that are read without their proofs, and that all three passed, on the same bytes,
   is `cp_verify`'s own logic (13.6). The same comment names what else is not enforced: switches for
   tests, which do not compile in a production build, and one of which, the one that declares a
   table tested, two toy tools set as well (`blocks/frame/handover_files.cpp`,
   `blocks/frame/handover_proof.cpp`); a caller that composes the library's blocks itself below the
   routines that check whole files; that a caller reads the outcome of a check on a table that was
   read without its proof from that check's result alone, which has no plain yes (what the routine
   fills in beside its result, and the reader's position in the file, also show whether it
   rejected); whatever the calling program supplies, which is believed (the routine that derives
   ranges and lookups from the public files, its other routines, plans from sizes of its own choice,
   a reader class of its own); and source that is written to defeat the C++ types, of which
   `blocks/README.md` names three ways, none closed whole. By the first, a test program that is not
   part of this repository got a false claim accepted by the unchanged library at the target of 128
   bits, at toy size. The second gives a wrong plan; a program of that kind, run
   at the same target and size, had a false claim accepted in 8 of 48 tries, and in 0
   of 48 without the declaration (reported: 17.5, item 11). The third led to a false
   acceptance at 128 bits too and is closed in part only. Each item is told in
   `toys/3-key-and-entry.md`, part 9, section 12.
4. *Tests on the public files.* 13.8 lists the tests of `cp_verify` on them by name; the no-overflow
   conditions (the names with `no-wrap`) are given one by one in the second halves of 11.2, 11.3 and
   11.4. The programs write them for a bound of 2^60, which is the bound that field 1 needs,
   although they can be built for field 2 alone (17.2, item 11; 17.5, under completeness). This page
   does not give most of the single conditions that end in `parameters-not-admitted` (those on a
   segment's messages are under R12.12, and the ten conditions on hiding are in 16.1), or part of
   what is rejected in `ranges` (R13.14).
5. *The model.* This page does not define what a line of the parameter file or a word of the scale
   file means for the model (8.6, R15.5), how tensors become the committed integers, or which entry
   of a tensor a cell of the stored table holds (11.5).
6. *Other forms of the parameters.* Sections 11 and 12 describe one form: `prof` is 20, `NE` is
   above 0, and the 32 digest lines are present. This page does not define the other forms that
   `cp_verify` runs without an acceptance (12.3, 14.4), or parameters in which W's base code is not
   Omega's (R13.19), which the programs reject.
7. *Field 1.* R12.11 gives a code's evaluation points for field 2 alone, so a verifier for the
   version word `H811` cannot be written from this page, and the programs of this repository can be
   built for field 2 alone (17.2, item 11). For code lengths above 2^32 no numbered rule gives the
   points. No line of the parameter file admits such a length (R12.11). For a verifier that takes
   the codes as inputs (8.4, the second way), which does not run the parser, a length of at most
   2^32 is a condition on these inputs (section 18).
8. *Four public tables, bit for bit.* The gate function, the sigmoid, the rotary embedding and
   attention's exponential function are given by formulas in double-precision floating point (12.6),
   which the programs evaluate with the platform's exp, pow, cos and sin. Two libraries can differ
   in a last bit, so only the 32 digest lines of the parameter file fix the tables (on what that
   rests: R15.3). Without these lines `cp_verify` gives no acceptance (14.4).
9. *The layout file and the scale file.* R4.9 fixes only that the parameter file comes first; the
   order layout file, scale file is the programs'. The layout file has more than one admitted
   encoding, and what a sign or a number beyond 64 bits gives there is left to the C library (13.4):
   a verifier that admits the plain encoding alone can only reject more.
10. *Private randomness.* The input of number 176 (R3.1), and how the values of section 16 are made,
    are outside the format. Nor does this page define the secret files of a committer or a prover,
    or any digest of them. The committer's rule (R4.12: the same randomness is never used under two
    roots) and its reasons are reasoning only: nothing of it is proved, the reasons are argued on
    paper only, and no verifier can see a breach (`docs/limits-all-rows.tsv`, row H4.4: the two
    tests named there look at how `cp_commit` keeps the rule, not at its reasons, and neither is in
    this repository).
11. *A parser for the items of a first query.* R1.9 says what a reader of an encoding rejects; the
    programs build these items and never read them back.

### 17.2 Where this page and the programs are known to differ

1. *R0.3: a third result.* `cp_verify` can end with `NOT CHECKED` (14.3), which is neither an
   acceptance nor a rejection, when a guard of the library ends a stage or a stage ends under a name
   that begins with `internal-` (13.8). Such guards are the cases of R2.7 and R5.7, where the rule
   says "rejects", that of R5.3 (no proof of work at a seal; it is asked where a plan is made),
   those of R7.12, R7.13 (a tree's number is below 2^16), R7.16 (a tree's depth is
   at most 48), R7.17, R8.5, R12.14, R13.16, and R13.5 and R14.9 (the three items that a file's
   first or one-time chain and its shared chain both hold are the same bytes), the tests on the
   matrix's units under R13.14 (`internal-weight-limbs`), and every misfit between a run and its
   plan. A verifier with two answers rejects in all of them. Beside `NOT-A-VERIFICATION` (exit code
   14; 13.7 and 14.4), where a verifier with two answers rejects too, two more exit codes of a whole
   call of `cp_verify` are neither an acceptance nor a rejection (14.3): exit code 2, where the
   command line is bad or a file that the caller names is missing, is not a regular file or cannot
   be read to its end, and exit code 15, where the answer cannot be written. No byte that was read
   from a file brings either of the two about. A last line that begins `NO: internal-`, with exit
   code 1, is no rejection of a proof either: it can come from the first steps of a checking call,
   before any stage has begun (`docs/limits-all-rows.tsv`, row C2.4). The rare sampled values at
   which `cp_verify` ends with `NOT CHECKED` are not such a guard: there the format itself gives no
   acceptance (13.9).
2. *What reaches a guard.* By a rule that the source sets for its guards, nothing that depends on a
   proof's bytes (`blocks/base.h`: no guard may depend on them; the source states this as a rule and
   does not say that it is kept); this was followed to the end for R5.7, whose cases depend on the
   plan and on the program's own calls, and for R5.3's guard, whose condition no public file can
   set. Nobody has shown that no file can reach a guard or a name that begins with `internal-`
   (`verifier/docs/output.md` says so of these names). Known to end a call with `NOT CHECKED` are
   the rare sampled values of 13.9, under eight of the `internal-` names, with the chances given
   there; public files with more than 4096 stretches of padding in `ranges` (R13.14); and a key
   file, leaf file, proof file or public file of some gigabytes, which is read whole before its
   length can be tested: one such file can be rejected on one machine and end so on another.
3. *R0.4 and R14.3: what a sub-proof reads.* The sub-proofs read shape and layout out of items, but
   they take their call lists, tables and padding from the public files: the programs place no item
   after `binding tree`. A first query holds these files only through the digest in `configuration`,
   and many of their numbers again in `request` (R15.5). The four tables of 12.6 are not in these
   files: of them the parameter file holds the numbers that their formulas take and can hold their
   digest lines (11.1; on what that rests: R15.3).
4. *R7.14: the session.* By the rule a session is all files that one verdict reads. The programs
   check key file, leaf file and proof file in three processes, each with records of values of its
   own. So the programs compare no vertex's value between the files, where the rule's one session
   does: the anchor's root is read in the key file's shared chain and again in the leaf file's link,
   and the leaf table's root in the link and again in the proof file's shared chain (R7.19). The two
   differ only on files that show a collision of the hash.
5. *Narrower bounds.* The binding tree's depth is at most 48 by R14.6 and at most 8 in `cp_verify`
   (8.3). A round has 0 to 64 proof-of-work bits by R5.2; the parser admits at most 58, and fewer
   than `sec` (8.1), and `cp_verify` at most 32 (8.3). A nonce has 1 to 8 bytes by R8.2 and always 8
   in the three files.
6. *Tests made elsewhere.* R7.3's bounds on a salt's length are tested by the parser (8.1) and
   wherever a leaf is hashed, not on the plan. R15.2 has numbers below 2^32, and the parser of the
   frame lines admits 15 digits (8.1). R2.3's bound on the length of *bytes* is tested before a
   round's nonce is added, so under the number 162 the programs would hash a string that is 8 bytes
   longer than R2.3 admits. No file that `cp_verify` admits shows any of these differences. By a comment
   in `blocks/hash/hash.h` nothing there checks that a leaf has at least one symbol (R2.3).
7. *R3.5 and R15.3: the second hash function.* By these two rules whoever forms the item `request` makes
   five digests with it, and four more are lines of the parameter file. The programs make all nine,
   the four tables' among them (R15.6; 12.6), and no others.
8. *Branches that the programs do not take.* They always set to 1 the word on a fold's mask that is
   sent ahead and the word on masks that go out with the switches, and no parameter line changes
   that: what R13.19, R14.4, R14.9 and R14.10 say for the value 0 is in no file of the programs. In
   a proof file `pending rows` is NG + 1 (13.4, step 5), so at least 2, and `cp_verify` rejects
   public files that give another number (`plan-link`): what 13.9 and section 10 say for one row,
   where nothing is joined, is in no proof file of the programs. What 11.4 and 12.5 say of a table
   of exponentials (sT other than sh - 4) is in the code of the stages alone: a whole call of
   `cp_verify` under such a parameter file ends `NO: policy` before the stages (14.3), and
   `cp_commit` and `cp_prove` reject the file (`prover-only-policy-kind`). The parser takes `listed` at 0 alone, so what R7.19, R8.2, R8.7 and R8.10 say of listed positions is in no file of the programs; every parameter file of this format in the repository has `salt 40`. `cp_commit` makes the leaf tables of a commitment in one call, all of them in a
   production build (in a test build the option `-leaves` makes some or none), and rejects a second
   call on it (`cp-commit-first-form-alone`; the comments of `prover/weights_commitment/arguments.h`
   on a second form of the call, which would add leaf tables later, describe a call that this
   rejection excludes), so what R4.11 says of a new leaf table has no case in the programs.
9. *R0.5: one home for each fact.* The format holds the number of answers more than twice (the
   plan's sixth word; R12.14, item 4; the sixth word of every notice, R12.3, and of the shapes of
   R13.13, but for a message that does not exist, whose word is 1; also the fifth
   word of a stored table's shape and name, R4.10; section 18 has it as a condition that they agree;
   the seventh word of every entry `SEND seed` holds the number too, R6.9, and that of
   the entry `RECORD`, R13.15), a salt's length three times (R14.6; the plan's third word; the sixth
   word of a stored table's shape and name, R4.10), and one shape that no verifier reads (R13.13).
   Each of the three lengths is read for its own use: the item's word for the salt in the slot
   (R14.6), the shape's word for the leaves of a stored table, and the plan's word for the trees
   that a run makes. The programs derive all three from the parameter line `salt`, so under them no
   file makes the three differ (from the programs' source: `derive.h`, `configure.h`, `key_fixed.h`,
   `lin_frame.h` and `link_frame.h` in `blocks/frame/`; `blocks/chain/public.h`). No rule of this
   page says what a verifier does where they differ; for a verifier that takes plan and shapes as
   inputs, that they are equal is a condition on the inputs (section 18). Outside the first queries,
   the commitment ID's input holds the anchor's seeds beside the name that fixes them, on purpose
   (R4.11). `cp_verify` parses the frame lines more than once.
10. *R7.17 (c) and R13.13: two rules that this page words more exactly than their short form.* The
    short form of R7.17 (c), which the programs' own messages use, is that a tree with the mask of
    one step is read in no other chain than the one that sent it. Read to the letter it would bar
    what R13.7, R14.4 (e) and R14.10 describe: in the programs' files W's tree holds such a mask, is
    sent on one chain and is opened on the chain that continues that one. The programs bar adoption
    alone, and R7.17 says so here. The short form of R13.13 gives the shared split's mask message
    the six words 2^32 - 1, the mask code, 0, 1, 1, 1. The programs write the number of answers s as
    the sixth word, and R13.13 says so here. The two forms agree at s = 1, the one value that the parser takes.
11. *Field 1.* The rules of this page define the format in two fields, with the version words `H811`
    and `H821` (R4.3). The programs of this repository can be built for field 2 alone: a build for
    another field does not compile (`blocks/base.h`), a program serves exactly one version word,
    `H821`, and it compares the first line of the parameter file and the head of every file with
    that word (`blocks/frame/version.h`). The repository has no parameter file with the word `H811`;
    `cp_verify` rejects such a file under the name `version-unknown (policy)`. In the
    library the numbers of field 1 serve tests of numbers alone, as data: by
    `blocks/numbers/setting.h` only the tests in `blocks/numbers/tests/` call for them. So whatever
    this page says of field 1 can be compared with no program of this repository.
12. *R15.6: the text of the call lines.* By the rule the three digests under the number 73 are of
    the file's own lines, each whole. `cp_verify` does not hash these bytes: it forms each text
    again from what it has parsed, as the line's first word, the call's name, its kind and the names
    of the call's segments, with one blank between two of them and a line feed at the end, in the
    order of its own list of calls (`verifier/src/cp_verify.cpp`). For the lines `callA` a segment's
    name is looked up and written back from a table of seven names. The tests on a call line (13.8,
    `policy-call-line`; those that bear on this: a line feed at the end, single blanks, no empty
    token, every byte of a token one of A-Z a-z 0-9 . _ -, and the order `call`, `callA`, `callH`)
    and on the lists of calls are meant to leave no file at which the two texts differ. That they
    leave none is this page's reading of the source and is not proved. The rule's rejection of a
    file without a line `call` has no test of its own in `cp_verify`, as that of a file without a
    line `callA` has (`policy-call-list-A`). The program puts one of two lists of its
    own instead of the missing lines, and the test that every one of the calls' segments is named in
    exactly one call rejects either list (`policy-call-list`), since neither names the call's
    segment `UT`; the source says so there.

*Compared with no file of the programs and with no second implementation:* 8.4's errors, as its text gives them, at
`prover/examples/target_full_f2/policy.txt` and `prover/examples/target_small_f2/policy.txt`; the claims of section 11; the placement of 11.5, beyond its number
of rows; the rows of 13.9; the record beyond the honest toy files, on which the second verifiers run its rows and read its answers (`tests/new/second/`: the entry `RECORD` itself they pass over, and an altered answer falls at the first proof of work, so those rows have decided on no false file); whatever holds only where the two base codes are
one code, that is, outside the split case of 16.1 (R13.19); whatever holds only with listed
positions, which the parser refuses, among it a round `LINK` under such a plan (R7.19); and, in a
chain of segments, a message that is kept for more than two segments and a group with h above 0
(R12.14, R12.16 to R12.18), since the layout files of the repository give a kept message the lives 1
and 2 alone and a segment has one mask row (13.4, step 7). What the words of a lookup (R13.14), k_lo
and the words on the high bits (R13.9, R13.13) mean was compared with nothing either: the three
items `ranges`, `the link` and `the test on the table`, encoded from values that were given field by
field, equal the bytes of one toy key file and leaf file, which would hold under many wrong readings
of these words, and only the tests on content depend on their meaning (13.9; 17.3). Of `cp_commit`
and `cp_prove` only headers and their comments were read. Of R13.16 to R13.19, no branch that the
toy key file (711, 34 and 13 rounds) does not take was compared with a file. That `cp_prove` and `cp_verify`
form the same 177 words (R15.1) in every case that both admit is known from reading their sources.

### 17.3 The scripts

`format_rules.py` has the rules of sections 2 to 6, one known answer for section 7, and the
encodings and rounds of sections 9, 10 and 12. Its functions for section 12 were written from the
programs' source and not from the rules. Its text of BLAKE2b's compression function (`blake2b_F`,
for R15.3) is adapted from the BLAKE2 reference implementation, as is the text in `blocks/` and in
`verifier/src/`: a comparison of the two can show an error made in adapting, not an error of that
implementation. It has nothing of sections 8, 11 and 13: no sizes, no sub-proof of a segment, no
test on content. Of the seven cases of R5.7 its function for a round's call (`Chain.call`, with `H`)
stops at five. It does not test that w is at most 64 or that the kind is 1 to 7: its caller must,
since from w = 256 on the bits of w would run into those of j in the hashed word (R5.2).

`format-vectors.txt` is what `format_rules.py` prints: it shows what the script computes, not that
the programs compute the same. Two tests of the programs,
`blocks/chain/tests/format_vectors_test.cpp` and `blocks/tests/format_types_test.cpp`, form part of
its lines with the programs' own routines and print how many lines they do not ask for. Neither asks
for the two strings of R2.4 (the first says so in a comment; the second has no line on them), and by
a comment in the second neither compares, among others, the numbers of trees (R7.13), the item
`table`, R12.6 to R12.10, R12.12 to R12.18 or R13.x to R15.x. A script beside the second verifiers,
`tests/new/second/vectors_against_the_toy_files.py`, compares further lines with the parameters that
the library's tool `blocks/frame/handover_files` writes for the toy run `key-and-entry`;
`tests/new/second/README.md` says which. For the commitment ID `format-vectors.txt` has no case
whose numbers fit together: its two IDs are formed from one name, which has 2^9 rows and 1 answer
for each, so 2^9 answers (R13.8) and one seed, and 512 entries in a row's message, so 9 coordinates
in the seed's point (R4.8, R4.10). One ID is formed with a given
seed whose point has two coordinates and with two answers, and one with that name's own seed,
sampled by R4.8, and two answers: this one fits the name in its seed and not in its answers.

`read_key.py`, `read_entry.py` and `read_proof.py` walk a file: they form every query, test the
proof of work, sample, follow every opening to its root, and ask that no byte is left. They make
none of the tests on content (13.9): that a file is read whole says nothing on whether it proves
anything. `read_proof.py` and `read_key.py` find by trial which trees are opened after a round that
samples positions, and do not know which trees a round must open: they read a file whole even if the
second opening of a tree has been taken out. Of `read_proof.py` its head says so. By its head `read_proof.py`
stops where positions are sampled and no tree opens after them, and asks at the file's end that
every tree of the three chains has been opened at least once; `read_key.py` asks neither; its one
stop of that kind is in the shared chain's round LOC, where the stored table must open.
`read_entry.py` finds nothing by trial: after the round LINK it opens both stored tables, after
every other round that samples positions the chain's next tree, and it stops where an opening does
not lead to its root. The three keep no record of values (R7.14), although a key file and a proof
file open some trees at two lists. The programs' test that runs `read_proof.py` takes two configurations of the tests,
`prover/tests/configurations/pinned_no_proof_of_work` and `prover/tests/configurations/pinned`.
Of the full-size
configuration this page knows of one file that has been read by a script: the proof of the stored
example `examples/full-size`, which `read_proof.py` read whole (reported; no record of the run is in this repository). It knows of no key file or leaf file of `cp_commit`
that has been read by a script (of such a leaf file `read_proof.py` takes the root and the
answers, and no more): the files that this page knows `read_key.py` and `read_entry.py`
to have read are toy-size files of the library `blocks/`.

The three readers derive no plan: each takes the bytes of every chain's parameters from files that
tools of the programs write (`handover_files` and `handover_proof` in `blocks/frame/`; for a proof
of `cp_prove`, `verifier/tests/first_queries_tool.cpp`, which writes them from the configuration
that the programs' own code computes). None asks R5.3's rule on a seal, nor do the tests' second
verifiers (`tests/new/second/toy_check.py`), and neither does `format_rules.py`, whose function for
a round's call asks w = 0 of kind 7 alone: at an entry `SEAL` with proof-of-work bits `read_key.py`
and `read_proof.py` would read a nonce, test it (R5.5) and go on. None of the three readers reads
listed positions, which the parser refuses. `read_key.py` takes the salt in the slot as 40 bytes;
it and `read_entry.py` serve field 2 and a configuration of two public files only. For a proof of
the three programs `read_proof.py` needs its options `--public`, `--table` and `--statement`:
without `--table` it reads a proof with the number 0 against the key file's table, which is not what
R14.2 says.

### 17.4 This number is not in use

### 17.5 Limits of the programs that these pages rest on

This section is not the account of the programs' limits; that account is
[docs/limits.md](../limits.md), with
[docs/how-hiding-is-checked.md](../how-hiding-is-checked.md). Where this section reports a result
from a row or a page, that place may hold worse results on the same matter, which are not repeated
here: read it first.

These are limits of the programs, not of this page. They are told here because these pages rest on
the programs: most of what the pages say was read from the programs' source.
[docs/limits.md](../limits.md) has the release's list, and names the pages that carry it on. This
section has the limits that bear on what these pages describe and that their other sections do not
tell, or tell only in part. Each item but item 14, which speaks of these pages themselves, says at
its end where it was read, and says so if it was not compared with the source; nothing was run for
it but planning calls of `cp_verify` (14.6), where an item says so. The items are grouped by what
they bear on: soundness, hiding, completeness, trust. Items 32 and 33 are at the end, outside their
groups: 32 bears on soundness, 33 on hiding.

*Soundness*

1. *What the accounting's numbers are.* The numbers that the error accounting prints (13.6, 14.7)
   are the program's own estimate. They are not a security level and must not be quoted as one. The
   compiled bound comes from this protocol's technical report, which states it for the protocol as
   the report defines it; by the accounting's printed line, which speaks of the blocks' round errors
   and the compiled bound together, it is cited or argued on paper (8.4,
   item 22, has the line), and by `docs/limits.md`, "The error bounds that the programs print are
   their own calculation, not a security level", it is not a theorem about these programs, and no
   configuration of the programs is known to meet all its hypotheses. It has the number t of a
   malicious prover's hash queries as a free variable. Where all rounds meet the target, the program
   prints the bound's two terms at three example values, t = 2^64, 2^96 and 2^120, and one figure
   that is derived from the bound (`as one number, by the compiled bound: <L> bits`): the largest L
   such that a prover with at most 2^L hash calls, the verifier's counted in, succeeds with
   probability at most its number of calls divided by 2^L. Under a parameter file that counts proof
   of work, the bound's first term, the three example values and this figure count it; the program
   says so in each of these lines and prints the figure without it beside the other
   (`, proof of work counted (without it: <b> bits)`). At
   `prover/examples/target_full_f2/policy.txt` the figure is 126.21 bits with proof of work counted
   and 96.23 without (as a planning call prints it, 14.6). None of these is a security level either,
   and the accounting prints that itself: `no figure here is a security level`. The unit of every
   figure is one key file's, one leaf file's and one request's proof, not a batch's. A round that
   the requests of a batch share is counted for all of them at once; counted jointly for a full
   batch of N requests, each round of a request's own chains has log2(N) bits less, 6 bits at N
   = 64. The program prints that in the last line of the accounting's routine, and computes the
   joint count nowhere (8.4, item 22). (from the programs' source, `blocks/numbers/accounting.h`,
   and documentation)
2. *The errors of the rounds.* The errors of rounds other than closeness terms are argued in
   comments of the library's sources, or taken there from a lemma of the paper that this protocol
   builds on (`blocks/iop/`, `blocks/numbers/blocks.h`). For the step from the single parts to a
   whole chain `docs/limits.md`, "What the bounds rest on depends on the parameter file", names the
   technical report's proofs of composition and compilation: if one of them fails, nothing ties the rounds' errors into
   a bound. The accounting itself prints that the round errors and the compiled bound are cited or
   argued on paper. (from
   the programs' source, `blocks/numbers/accounting.h`, and documentation)
3. *On what an acceptance rests.* Every
   acceptance of a proof file rests on the theorem on codes (8.4, item 6), which the programs'
   accounting takes as given; a statement on codes held to follow from it is proved in the Lean library (13.7, condition 4).
   It is the theorem of the next sentence, and the bound there is its
   count for a closeness term. The bound rests on a theorem from the full version (ePrint 2025/2055,
   Theorem 4.6: not in the proceedings; that version gives an outline of its proof) of a paper
   published at STOC 2026, taken as given — we have checked no proof of the paper's own form of the
   bound — (what has been checked of it, and what has not:
   [docs/limits.md](../limits.md#what-the-theorem-assumes)), and on further steps argued on paper in
   our report; that our parameters meet the theorem's conditions is our own reckoning, with
   no bits to spare. At `target_full_f2/policy.txt` 43 rounds with a closeness term hold with less
   than 0.01 bit to spare, proof of work counted, by the planning call. To which stored table a key
   file and its leaf files are bound rests on the theorem on codes too (item 19): at
   `target_full_f2/policy.txt` and at `target_small_f2/policy.txt` a planning call of `cp_verify`
   counts no round with a closeness term on the bound within a third of the distance (8.5). In the files `policy.txt` of
   `target_full_f2`, `target_small_f2`, `target_small_f2_roomy` and `target_small_f2_gap` under
   `prover/examples/` the line `work positions` is
   `20 28 32`, so an acceptance there, of a key file and a leaf file too, also rests on the
   accounting's assumption on proof of work (14.3): at `target_full_f2/policy.txt` and at `target_small_f2/policy.txt` the least round has 96.23 bits
   without it (the line `work:` of a planning call, 14.6). (from the programs' source: the lines
   `work positions` of these four files, read by program) For a mask code, which is read as repeated to the base code's length, and in field 1
   for every code, a circle code, the step from the theorem's code to a part of it is one more
   argument of this project. (from the programs' source, `blocks/numbers/codes.h`, `blocks/numbers/blocks.h`, and documentation)
4. *No factor for the rows.* A closeness
   error has no factor for the number of rows of an interleaved word (8.4, item 9). The cited
   theorem does not say this. The reason is a short argument of the source's own, which the source
   calls unrefereed: an interleaved Reed-Solomon code is a Reed-Solomon code over a larger field on
   the same domain, and the cited bound counts exceptional challenges without regard to the field.
   That comment speaks of the bound of the theorem on codes. The accounting's printed line calls it
   `a bound argued on paper for this protocol`. (from the programs' source:
   `blocks/numbers/setting.h`, `blocks/numbers/accounting.h`)
5. *The round `LINK` is counted at the larger of two errors.* In the round `LINK` a leaf file opens
   two lists of columns, and the round's error is the larger of the two lists' errors, not their sum
   (8.4, item 17). The reason is an argument of the source: which of the two folded rows is false is
   fixed before the positions are sampled, and that table's own list must miss it. Counted as a sum
   the round would have up to one bit less, and the numbers of columns have no margin (8.4, item
   17), so a round that meets its target as counted can miss it as a sum; whether it does depends on
   the parameter file. By row SO-10 of `docs/limits-all-rows.tsv` the round as a sum would read between half
   a bit and one bit lower than as counted: at `target_full_f2/policy.txt`, where `LINK` as counted has 129.57 bits with its proof of work, still above 128 bits; at
   `target_small_f2/policy.txt`, where it has 128.77 bits, it may read below 128 bits (about 127.8 if its two errors are equal there); the programs do not compute
   the sums. (from the
   programs' source, `blocks/iop/link.h`, and documentation)
6. *Padding of a lookup.* A padding position of a lookup contains a row of the lookup's table
   (R13.14, 11.1). That such padding lets no key outside the table pass, whatever the number of
   padding positions, is argued in a comment of the source and is marked there as not proved. By that comment one
   test shows one case: surplus padding with a free value lets such a key pass, and surplus padding
   with a row of the table does not (that test, `LKP_SURPLUS_PADDING`, is not part of this
   repository). This bears on the key file's range proof, which runs this routine, and on the
   segments' lookups, which pad in the same way (`toys/3-key-and-entry.md`, part 7, section 4.8,
   tells the argument). (from the programs' source: `blocks/iop/lkp.h`)
7. *Numbers that a plan takes on a routine's word.* For some rounds a number from which the round's
   error was counted is taken on the calling routine's word, and nothing at the run checks it: the
   groups whose list sizes count at an out-of-domain sample or at a batching, the smaller oracle
   messages under one seed, a lookup's table and keys, the next stage's list size in a code switch,
   and a stored table's distances. Of the distances this holds in part: prover and verifier compare
   the run's distances with those that the plan states beside the round (`Rounds::require_table`),
   but what is hashed of the table leaves them out, and the bounds of a later test and of a link are
   true only of a table whose one-time test ran at the same distances, and that test is in another
   file's plan. The source calls each such round an open item, and the accounting prints how many
   rounds of a plan are of this kind (`toys/1-product.md`, part 1, section 6). (from the programs'
   source, the type `Priced` in `blocks/numbers/plan.h`, and documentation) Two more things of this
   kind may go unchecked, neither brought to a false acceptance: how the coordinates of the shared
   point of the test on the table are split, and two small planning routines of the lookup that
   nothing checks when they are called alone. This page does not say whether these two go unchecked
   in the programs of this repository: that header's list names neither, and this page names no
   source for them.
8. *That `cp_verify` counts from the numbers on which it runs.* That `cp_verify` computes each
   round's error from the same numbers on which it then runs is known from reading its source and
   from a test on weakened copies of the program. By row SO-4.1 of `docs/limits-all-rows.tsv`, where
   `cp_verify` compares the number from which a round's error was computed with the number on which
   the round runs, crafted files have tested that comparison for one such number; for the others no
   input has tested it. By row SO-13, that every planned round meets the target is tested in one
   place, and no file or test has made that test reject on a round of a sub-proof. (from the
   programs' source and documentation)
9. *No list of comparisons for a soundness statement.* That the derived numbers of a plan meet the
   conditions of a soundness statement is checked by no whole list of exact comparisons: such a list
   does not exist. Section 18, last point, gathers the conditions that these pages know, those on
   one chain's plan, codes and shapes and those that tie the inputs of one chain or file to
   another's, and its lists need not be whole. (the first
   sentence from the programs' documentation: `docs/limits-all-rows.tsv`, row SO-4.2)
10. *Lines by which the writer of a parameter file could choose a laxer count.* The writer of a
    parameter file is not the one who relies on a proof. For the lines that were found by which the
    writer could choose a laxer count, the parser takes one value alone (8.1, tests 10, 11 and 14) or the standard parameter form has a
    condition (13.7), with this exception: for the five lines `work` there is a cap alone (condition 5 of 13.7: more than
    32 proof-of-work bits in a round are rejected, `policy-work-most`). Closeness terms of the key file and the leaf
    file rest on the theorem on codes (13.7, condition 4; item 19), and closeness rounds of a proof too, `LIN point` among them (8.4, item 16);
    no condition looks at that either. Nobody has shown that these are all such lines. (from the
    programs' documentation: `verifier/docs/parameter-file.md`, "Laxer choices";
    `docs/limits-all-rows.tsv`, row SO-6.3; the sentence on `LIN point` is in neither: it is from
    the programs' source, `plan_lin_shared` and `analysis` in `blocks/numbers/blocks.h`, and
    `verifier/src/cp_verify.cpp`, which asks of no plan on what a term rests)
11. *The library is not an interface.* In test programs that call the library's routines directly,
    false claims have been accepted. By the source the six ways that were found for
    ordinary code are closed inside one process; by `docs/limits.md` that is one reading,
    another, which counts them otherwise, holds that some are closed only in part, and no one has
    reconciled the two. Ways that are taken on purpose and break the library's stated contract are
    not closed, and the source's list of them is one of "Known ways"
    (`blocks/numbers/plan.h`); of what its guard against the third way below leaves uncovered it
    says: "These are examples, not a complete list". Of the cases that were found `docs/limits.md`,
    "The library under the programs can be misused", says that there may be other cases. Three are named. By the first, a false claim was accepted at the target of 128 bits, at
    toy size (17.1, item 3). The second, an operator template that is declared before the headers,
    gives a wrong plan; a program of that kind, run in the tester's own process at
    a target of 128 bits, at toy size on made-up data, had a false claim accepted in 8 of 48 tries;
    the same program without the declaration in 0 of 48 (reported:
    `docs/limits-further.md`, SO-5.1). Nothing of `cp_verify` or of any file a provider sends is
    touched by it. The third, a size that is changed between two reads of a planning routine, led to
    a false acceptance at 128 bits (the key file's test passed for two tables under one commitment,
    in a test program, with a malicious prover's switch on the prover's side; told by no file outside
    `docs/format/`) and is closed in part only. A calling program that
    gives the routines configurations or tables which do not belong together may find a way to a false acceptance. No file, argument or setting has brought `cp_verify` to any of these ways in
    the attacks that were made on it; that they cannot reach it rests on reading its source and is
    not proved. (from the programs' source: `blocks/README.md`, `blocks/chain/closed.h`,
    `blocks/numbers/plan.h`)
12. *Parameter files under which nothing was tried.* `cp_verify` admits parameter files under which
    `cp_commit` and `cp_prove` make no files. By row SO-16.1 it bounds single numbers of the
    parameter file (no code is longer than 2^32) but not the sizes that follow from them. By row
    SO-6.3.1 `cp_commit` and `cp_prove` keep the table's numbers to narrower ranges, among them 4 to
    18 column variables and an odd number of random entries; the line `table` names no distance, the committer's tests can run up to the
    Johnson radius, and these two programs put no cap on the numbers of opened columns beyond the count against what a row's random entries hide: by that row no other cap on these numbers was found in the
    provers' sources or the proof library below 2^32. By the programs' source the library computes the one-time test's distance and every number of opened columns. `cp_verify` takes 1 to 20 column
    variables (`w-shape-lk`) and whatever the library can size. In that wider region there are no
    honest files, so no altered file and no malicious prover has been run there, and by row SO-6.3
    nobody has traced whether a `YES` can be reached there. By the same row attention's list of
    lookup calls (the lines `callA` of the parameter file) is a further part of that wider region:
    `cp_commit` and `cp_prove` take one list alone, no parameter file in this repository has
    another, no proof has been made or checked under another, and a run under another list that the
    verifier accepts can end `YES` all the same, since no condition of the standard parameter form
    fixes the list. Row SO-16.1 tells planning calls (`cp_verify -plan accounting`), which only count and read no
    key file: with the line `table` of the full-size example alone changed to `table 20 32 3000000000 44040191 0 20`,
    a key file of up to 3.9 x 10^9 and a leaf file of up to 7.7 x 10^9 bytes (the largest of sixteen tries; no search for a maximum has been made; run once, not on the released programs). (from the
    programs' documentation: `docs/limits-all-rows.tsv`, rows SO-6.3, SO-6.3.1 and SO-16.1; the
    sentences on the routine from the programs' source, `prover/hash_commitment/hf_table_params.h`)
13. *A leaf file has no authentication of its own.* It is checked against the anchor in every call.
    That a false leaf file can only make a verifier reject an honest proof is the design's intent
    and is not proved. (from the programs' documentation: `docs/limits-all-rows.tsv`, row SO-8.3)

*Hiding*

14. These pages hold no proof that a key file, a leaf file or a proof file hides the weights; what
    is known of hiding is on [docs/limits.md](../limits.md). A copy of the repository that does not
    have that page claims nothing about hiding: where these pages speak of hiding, they give what
    the design is meant to do and the reasons of the programs' source.
15. *An acceptance says nothing about hiding.* No run of a verifier can know how masks and salts
    were made, how a leaf table was chosen (section 16), or whether a leaf table is retired or past
    its limit. So a verifier cannot tell a commitment that `cp_commit` made under its option
    `-public-weights` from any other. Such a commitment does not hide the weights from anyone who
    has the option's file of 32 bytes (`cp_prove`'s usage text:
    `nothing is hidden from anyone who has it`); its key file, leaf files and commitment ID bear no
    mark of that, and by `cp_commit`'s usage text its proofs look like any other commitment's
    (section 16, last paragraph before 16.1). (from the programs' documentation:
    `docs/limits-all-rows.tsv`, rows HW-566 and M1.10; the option: from the programs' source,
    `prover/common/public_weights.h`, `prover/prover/arguments.h`,
    `prover/weights_commitment/arguments.h`)
16. *The hiding polynomial of a sumcheck.* That the messages of a sumcheck with a hiding polynomial
    show nothing beyond the claim rests, by the source's comment, on coefficients that are uniform,
    fresh and enter nothing else that is shown, and on one fact: for every choice of challenges, the
    map from the polynomial's random coefficients to the messages is one to one and onto the tuples
    that pass the verifier's tests. The source states this in a comment, and a test
    (`blocks/iop/tests/hp_test.cpp`) checks it for some small shapes only, by enumeration over the
    fields of 3, 5 and 7 elements. (from the programs' source: `blocks/iop/hp.h`)
17. *The types that guard the masks do not see everything.* The library's types stop the plain ways
    of using a mask entry twice, too early or too late. By the source's own comment a prover's code
    can still use an entry after the proximity test on its message has begun, add one entry to two
    scalars, or name an entry by its number; a proximity test that another `Statement` runs on the
    same message leaves the message's ledger of entries open; a program that does not include
    `blocks/iop/send.h` can define the access type itself; and the routine that sends an oracle
    message cannot tell mask rows with fresh randomness from mask rows with data. No honest run
    shows such a fault, and no verifier does where both sides make the same calls, which is the
    comment's case: every test passes, and what is lost is hiding. (from the programs' source:
    `blocks/iop/mask.h`)
18. *Self-checks that no test of this repository has seen fail in the whole programs.* No test of
    this repository, and no run of a production build, shows that a whole `cp_commit` or `cp_prove`
    stops when one of these self-checks at its start or in a table's commit run fails: the known
    answers of the masks' generator and of the other routines for randomness, the key schedule, the
    second implementation of masks, salts and keys, and the key file's key. (from the programs' documentation: `docs/limits-all-rows.tsv`, row
    E18.1)
19. *The table's record.* `blocks/frame/examples/record.f2.txt` has no model's lines; its 32 lines are the first 32
    of `target_full_f2/policy.txt` (8.5). The list of a leaf file's commit-run key
    holds the leaf table's stored answers, and the list of a batch's process key holds the stored
    answers of the batch's leaf table, and no test compares the bytes that `cp_commit` and
    `cp_prove` pass for the two (`COIN-KEYS.md`, section 2, "The copy's answers"). (the two keys
    and their lists: from the programs' source, `prover/common/coin_keys.h`; that no test compares
    the two is taken over from that page and was not compared with the source) By the programs'
    documentation: what has run, without and with counted proof of work, is told in
    `prover/examples/TARGET-KITS.md` and in the pages to which it sends, and this page does not
    repeat it. Of the suite's rows (`SUITE.md`), `verifier.record_altered_sets.world` asks for `YES`
    on an honest set with counted proof of work and for a rejection by name of
    23 copies with a few bytes changed, and shows no more than that `cp_verify`
    notices these changes: no cheating committer's or prover's work; the row
    `gpu.run_target_example` names a configuration with counted proof of work, `target_small_f2_roomy`; and the second computation of the example configurations' figures,
    `verifier/tests/kit_rounds_second.py`, by its own head computes the weight table's two
    distances, its numbers of columns and the rounds `RECORD`, `LIN point`, `LINK point`, `LINK` and
    `LOC` of the table by the formulas of `blocks/numbers/blocks.h`, not the
    technical report's, with which they were not compared: for these lines it is "a second
    arithmetic, not a second derivation"; `cp_commit`, `cp_prove` and `cp_verify` reject a parameter file with any other value of `answers` than 1
    (`parameter-range`), the first two under `-public-weights` too: the written proof covers one OOD answer a row, so two to four are outside it, and this project's proofs cover a table only where it has one answer; no attack on a table with more is known, and nobody has searched for one. (from the programs' source)
    `integrations/check_proof.py` accepts a `YES` by itself, with exit code 0: its
    last line then repeats the verifier's line `commitment:` whole, with the clause on the record
    (from the script's source and the text that it prints for `--help`). On what an acceptance rests is told in `prover/examples/TARGET-KITS.md` ("The table's record"). By
    that page, at what it calls the default setting: the truth of an accepted request and the
    binding of the weights both rest on the theorem on codes (taken as given: 13.7, condition 4); that the parameters meet that theorem's conditions is
    the project's own reckoning, with no bits to spare; both lean on counted
    proof of work in the rounds that draw positions, and that counting is itself an assumption for
    the programs' hash chains; the bound rests besides on three further steps that the technical
    report argues on paper; binding is meant there in a modified sense, in which the hash itself supplies the key's
    seed; that sense does not imply the binding definition that the report cites, and the report
    proves it for a commitment ID whose hashed string does not take what was sampled for the record:
    the released programs' string takes it, and the written proof does not cover that string. This page
    claims neither sense of binding for the released programs. This page knows of no corollary of
    the report that covers counted proof of work, of no comparison of the shipped parameter files with the report's setting, and of no check that
    a file that the verifier accepts meets the conditions of the report's corollary; what Lean proves at this setting, of its own verifier and not of `cp_verify`, is told on `prover/examples/TARGET-KITS.md`. Where that page says more or
    other than is reported here, that page holds.
20. *What the files and the output show.* The next token is in the proof file unmasked (R14.8), and the printed lines `covered:` and `result-v1` show r and e (14.7, 14.8). The output and both files of a response are to be kept as private as the
    request and the reply. Anyone who serves the leaf files learns which leaf table a verifier asks
    for, and when, and the number of the chosen leaf table is in every proof file of the batch
    (R14.2). (from the programs' documentation: `docs/limits-all-rows.tsv`, rows H38 and H11.5)

*Completeness*

21. *One test on a plan is too strict.* All rounds of a plan that name the same two codes must name
    one integer m. That is stricter than it need be: by a reading of the
    source the test's key is the two codes alone, and a stage's m also hangs on its k, h, load and
    scope. No lawful file is known that it rejects at the values that the parser takes; six planning
    calls, none refused, show no more than that. Such a rejection would be a false rejection, not a false acceptance. (from the programs' source:
    `blocks/numbers/plan.h`)
22. *The no-overflow conditions are stricter than field 2 needs.* They are written for a bound of
    2^60, which field 1 needs, in programs that can be built for field 2 alone (17.1, item 4). So
    some parameter files are rejected without need. (from the programs' source:
    `verifier/src/cp_verify.cpp`, `verifier/src/vf_attention_shape.h`)
23. *Sizing may give more than is needed.* That the sizing finds the least number of random entries
    is not proved, and the search for the integer m is said to return the largest good
    m where the good values are one stretch of integers that begins below 67, and not otherwise; it
    finds none if no m below 67 is good. By the comments what either returns is always valid, so
    sizes may be larger than needed. (from the programs' source: `blocks/frame/derive.h`,
    `blocks/numbers/codes.h`)

*Trust*

24. *At which sizes the library is tested.* The tests of the library `blocks/`, and those malicious
    provers and injected faults of this project's own tests of it that are not in this repository,
    run its parts at toy size; some run at toy message lengths but at the real target of 128
    bits (the head comments of `blocks/frame/tests/lin_frame_test.cpp` and
    `blocks/frame/tests/link_frame_test.cpp`), and by the second of these comments "at full size
    nothing runs": that test only sizes and plans a full-size parameter file. The files that the
    library's own programs write are toy-size files. Most runs of the programs that their
    documentation tells used synthetic weights. The runs on a real model's weights are told in
    [integrations/LIMITS.md](../../integrations/LIMITS.md#what-has-never-been-run) and, for the four
    runs under clipped-gap limits fitted to a real model, in
    [docs/limits.md](../limits.md#within-limits). [docs/what-is-not-hidden.md](../what-is-not-hidden.md#public-weights) and
    [examples/maverick/README.md](../../examples/maverick/README.md) are the pages that tell of the
    makings of a commitment to a third party's public 4-bit conversion of a real model's weights and
    of the proofs under it; it is
    made under a published seed, so it hides nothing. (from the programs' documentation, as far as
    it tells them; for the rest this page names no source)
25. *The sub-proofs run a second implementation.* The smallness test and the lookups of the
    sub-proofs in `cp_verify` and `cp_prove` are a second implementation of what `blocks/iop/rng.h`
    and `blocks/iop/lkp.h` contain, and by row SO-12 of `docs/limits-all-rows.tsv` so is the
    sub-proofs' masked sumcheck of what the library holds. So the library's tests of these three say
    nothing of what the programs run inside a request's sub-proofs; by the same row, for the key
    file's proof of ranges `cp_verify` runs the library's own lookup. By the same row, that the
    three are a second implementation is known from reading the source, and for `cp_prove` only from
    its files' head comments and includes. (from the programs' source and documentation)
26. *What an acceptance covers.* An acceptance covers the positions r to e - 1 of the request file
    and no others. The request file, the next token and the numbers after `-policy` must be the
    caller's own, never taken from files that came with a response, from the parameter file or from
    the line `covered:`. The commitment ID comes from the party that vouches for the model, which
    may be the provider itself, but not through the server that is being checked and never from the
    files that are checked (14.2, `-id`). (from the programs' documentation: `docs/glossary.md`, the
    entries on the commitment ID and on the party that vouches for the model; `docs/limits.md`, "A
    `YES` covers only the positions that you name, in an integer model" and "Exit code 0 alone does
    not mean accepted"; `docs/verifying.md`, "Compare, never copy"; `docs/limits-all-rows.tsv`, rows
    SO-18 and SO-49.1)
27. *Where a reply ends.* The proof system has no notion of a token that ends a reply: a reply that
    was cut short verifies as a shorter reply. `cp_verify` does not compare the last checked token
    with a model's end tokens. `integrations/check_proof.py` does: it rejects unless the last
    checked token is one of the token IDs that its caller passes after `--end-token`, and it rejects
    if an earlier token of the reply is one of them. It knows no end tokens but that list, and after
    `--end-token none` it makes neither comparison and prints that the end of the reply was not
    checked. The comparison is worth as much as the policy. Under `exact-match` an acceptance is
    meant to say that every checked token is the top choice of its position, so an end token at a
    position where the model's top choice is another token is a false claim like any other. Under
    `clipped-gap` a false end token counts as one wrong token and is limited only as much as any
    other wrong token is, so a provider can end a reply early and still pass. (from the programs'
    source, `integrations/check_proof.py`, and documentation: `docs/limits-all-rows.tsv`, row SO-29)
28. *How faithful the integer model is.* Nothing is said on it. The scale file and eight lines of
    the parameter file (`MF`, `KF` and the six lines that begin with `eps`) are set by the provider
    and are only tested against ranges. Badly chosen values inside these ranges round everything
    away. The worst case that the documentation tells is this. With scales in the file `KL.bin` that
    the standard parameter form admits, the contribution of the layers can be rounded away
    completely, for weights of ordinary size: for example with the two constants of every norm set
    equal (M = K = 24); the function that is proved is then only embedding, norm and output head,
    and the output depends only on the last token. C and
    theta are counted in the integer model's units, so scales that shrink the logits also make a
    bound on gaps easier to meet. The commitment ID is a hash over all of these values, and no
    program tests that they are sensible. Anyone who publishes a commitment ID, or relies on one,
    should read these values or evaluate the integer model that the files describe; once is enough,
    not for every response. (from the programs' documentation: `verifier/docs/parameter-file.md`;
    `docs/limits-all-rows.tsv`, rows SO-6, SO-6.1 and SO-6.2)
29. *Nobody vouches for the configuration files.* The tests on the parameter file judge the lines
    that bear on the proof system, not whether this is the model that the caller has in mind. (from
    the programs' documentation: `verifier/docs/parameter-file.md`, "What it leaves to you")
30. *The policy's bound is on gaps of logits.* `cp_verify` accepts a whole call only where sT = sh -
    4, that is, under the step table (12.5), and every example configuration has sT = 4 and sh = 8.
    The policy's bound is then on gaps of the integer model's logits, not on log-probabilities.
    (from the programs' source: `verifier/src/cp_verify.cpp`; the files `policy.txt` under
    `prover/examples/`)
31. *Files are opened by name.* `cp_verify` follows symbolic links (13.3) and reads key file, leaf
    file and proof file whole before it tests their lengths (17.2, item 2). A caller keeps the
    directories of a call where only its own account can write, and uses a verifier that it built
    itself. (from the programs' documentation: `docs/limits.md`, "A hostile file can exhaust memory
    or time" and "The provider chooses the committed model and its settings";
    `docs/limits-all-rows.tsv`, rows SO-16.2 and SO-31)
32. *This page tells of no malicious prover or committer tried against the whole programs with
    proof of work, as the four files of item 3 have it (item 19); what was tried is in rows SO-15.5, SO-15.6 and SO-15.7 of
    `docs/limits-all-rows.tsv`.* `SUITE.md` has no row
    with a malicious prover or committer against the whole programs: no test there
    runs a weakened or cheating copy of `cp_commit`, `cp_prove` or `cp_verify`, or injects a fault
    into their running state. None of its rows runs a program built with a cheat switch; some rows run a prover that argues for something false, made so by the test's own code, all
    on synthetic data and mostly at toy size, and none of them changes `cp_commit`, `cp_prove` or
    `cp_verify`. By `prover/tests/configurations/README.md` and row SO-15.5 of
    `docs/limits-all-rows.tsv`, the project's tests with cheating or fault-injected programs, which
    are not in this repository, and the verifier's tests on real proofs run at small parameter files
    that are kept fixed for them, with some exceptions, which the row names. In the files of these
    that are in this repository (`policy.txt` in the directories `pinned..` under
    `prover/tests/configurations/`)
    those that count proof of work have the first 32 lines of
    `prover/examples/target_small_f2/policy.txt`. Some rows of `SUITE.md` run the programs
    themselves at example configurations whose `policy.txt` asks for proof of work, among them
    `gpu.one_changed_weight` and `gpu.serving_links` (at `target_small_file_f2`) and
    `verifier.extreme_files.full` (at `target_full_f2`); by the head comments of their scripts none
    of these puts a malicious prover's or committer's files before `cp_verify`: in the first two
    `cp_commit` and `cp_prove` stop by themselves (unchanged programs), and the third makes planning
    calls alone. A fourth row, `verifier.made_up_files.full`, puts synthetic key, leaf and proof
    files before `cp_verify` at the full-size example's `policy.txt`: it checks that they are rejected cleanly where they are read and that all three checking stages
    run, and real proofs and crafted forgeries are untried. No test ends a program by a signal under
    a parameter file that asks for proof of work, either. (from the programs' documentation:
    `docs/limits.md`; `docs/limits-all-rows.tsv`, rows SO-15, SO-15.5, SO-15.6 and SO-15.7; `SUITE.md`;
    `prover/tests/configurations/README.md`)
33. *What the sweep of lost random values found.* The sweep (on test builds of [other
    sources](../glossary.md#other-sources): rows HW-654, HW-660) overwrites sampled values in
    `cp_commit` and `cp_prove`: in each run all the values that one call of a generator makes, or
    the last of them alone, were overwritten, or their computation was left out. These runs lost
    random values and altered no claim on purpose, so this page does not count a proof that verifies
    in such a run as a false acceptance; what such a fault can cost is hiding. A file with a lost
    mask can still verify, because the verifier does not check for masks and cannot see most of
    them. What the sweeps found: [docs/limits.md](../limits.md), "One main risk: a lost mask"; rows
    HW-654, HW-619 and HW-660 of [docs/limits-all-rows.tsv](../limits-all-rows.tsv); in short,
    [guide/weight-hiding.md](../../guide/weight-hiding.md).

## 18. For a formal definition

This section is for anyone who defines the format in a proof assistant or argues about it with the
hash modelled as a random function O (R2.4). It adds no duty of a verifier that derives its numbers
from the public files (8.4, the first way). For a verifier that takes plan, codes and shapes as
inputs (the second way), its last item lists what that verifier asks of its inputs, or what a
statement on it assumes of them, first chain by chain and then between the chains and the files. The
section gathers what a formal definition must take care of.

- *Strings outside the format.* No hash of the format is O on a string whose first byte is not 0 or
  1, and none is made under the number 177 (R2.5). A definition that needs strings which no hash of
  the format uses can take these.
- *Each block is a query of its own.* A hash with 64 bytes of output counts as two queries of O
  (R2.4), so each block of a round's answer (R5.4) is the value of O on a string of its own.
- *A verifier's message is a total function of a fixed number of blocks* (R6.11). The plan's entry
  gives K, the most words that the round's sampled values use, and so the blocks to take. The caps
  on attempts (six for an element of F, two for an element of E that is not 0) make the function
  total; R6.11 gives their cost in statistical distance.
- *Some public inputs are answers of the oracle.* (Part of R4.9.) The item `configuration` (R4.9) is
  a hash. These are hashes too, or depend on hashes: in a statement, a table's root and the answers
  of its record (R4.6, R4.8), a binding tree's root (R13.6, R14.7) and a seal (R4.7); outside the
  first queries, the commitment ID (R4.11), whose input holds the seeds of the anchor's record,
  values sampled from a round's answer, that is, from answers of O (R4.8, R5.4). A model must not
  fix these bytes before O is chosen. A claim of the form "for all parameters' bytes, with
  probability over O" says nothing about the honest case: bytes that are fixed beforehand contain
  the digest of the public files with probability 2^-256, and a verifier that compares rejects all
  others. What is fixed before O is the text of the public files, and whatever of the parameters
  follows from that text without a hash. The item `configuration` is computed from the text after O
  is chosen (R0.5: what follows is computed), and roots, seals and answers come from a prover or a
  committer who may query O. The nine digests in the item `request` (R15.5) are values of the
  second hash function, which is not O: it is a fixed function built on the same compression function. What
  a statement then needs of it, and of the four tables where a definition takes them as data, is
  told in R15.3.
- *The plan can be taken as an input.* A verifier that is defined from this page can take the plan
  of R4.3 as a public input in its canonical bytes. Two questions then lie outside the verifier's
  definition: whether the plan's numbers are the ones that section 8 derives from the parameter
  file, and whether a statement on soundness can be made for these numbers. From whom the plan is
  taken matters: its numbers carry every round's error. By R0.2 it never comes from a key file, a
  leaf file or a proof file. A plan that the provider hands over would let the provider choose the
  errors, so the caller derives it, or has it derived by a program that it trusts like its verifier,
  from its own copy of the public files. The same holds of the codes and of the shapes: the party
  that derives the plan derives them too, from the same copy of the public files, and never from a
  key file, a leaf file or a proof file (R0.2).
- *What is then assumed of the inputs.* A verifier that takes plan, codes and shapes as public
  inputs (8.4, the second way) runs neither the parser nor steps 3 to 6 and 8 of 8.4, item 23. What
  these ask of the three inputs is then a hypothesis of every statement on such a verifier, or the
  verifier tests it itself. Such a statement has two honest forms. The first has one hypothesis:
  plan, codes and shapes, those of all chains of the three files together, are what 8.4 gives on one
  copy of public files that 8.4 does not reject. By this page's reading of the source everything
  below that is a condition on plan, codes and shapes follows from it. This form needs a definition of 8.4. The second has explicit
  conditions on the inputs; the lists below are for it. In the programs' source the conditions are
  in routines that return nothing or a sentence: `lin_fault` (`blocks/iop/lin.h`), `index_fault` and
  `foldeq_fault` (`blocks/iop/foldeq.h`), `link_fault` (`blocks/iop/link.h`), `iopp_fault`
  (`blocks/iop/iopp.h`), `req_fault` (`blocks/iop/req.h`), `req_layout_fault`
  (`blocks/iop/req_rows.h`), `wf_fault` (`blocks/iop/wf.h`) and, for what a key file's values show
  only together, `key_fault` (`blocks/frame/key_fixed.h`). The routine that derives the numbers from
  the public files asks them and rejects by name. Every block of the library asks its own routine
  again at its entry, on the prover's side and on the verifier's, and a sentence there ends the run
  (a guard: 17.2, item 1). The two routines that read a whole key file or leaf file ask before that,
  and a sentence there is a rejection (`blocks/frame/check_key.h`, `blocks/frame/check_entry.h`).

  *Parts of the ten conditions on hiding* (16.1), named by their numbers there. For a verifier that
  leaves the ten out and derives nothing, no other place asks them. The first two belong among the
  conditions of a soundness statement. The other four are conditions of room: where one is missed, a
  row of 13.9 names an entry that does not exist or that holds something else. These pages do not
  define what such a row says, so a soundness statement needs these four as well.
  1. *Condition 6, fourth line.* (matrix rows) * 2^k_lo is at most
     the number of plain entries of a row's message, in the words of R13.13 and R13.8: l - 3 s. It is the first of the further conditions of 8.4, item 23; in the
     source it is asked of the shape by itself, which `lin_fault` does first. A link compares the
     two folded rows on the plain entries alone (under R13.10). What a folded row holds after them,
     the fold of the pads, zeros and the fold of the random tails (section 9), is tied to the anchor
     by nothing. So without this condition a statement on a leaf table's weights does not follow
     from one on the anchor's.
  2. *Condition 10.* This is not a comparison of two numbers of the inputs. A plan's entry holds no
     error and no number of requests (R4.3; 17.1, item 3), and no shape holds one; the one such
     number among the inputs is the depth of the item `binding tree` (R14.6). The condition is on
     the statement's own error term: for the round `LIN point` of a proof it has the factor 2^depth,
     that is, the term 2^depth * fe(w_T) (8.4, item 16, with N = 2^depth), and in a key file the
     depth is 0 (8.4, item 23, step 8). In the source a plan notes the number beside the round, and
     the block compares the note with the number of leaves of the hashed shape where the point is
     sampled (`require_sharing` in `blocks/chain/rounds.h`; `blocks/iop/lin.h`).
  3. *Condition 4.* The mask message has at least 2 k + 1 - h entries at every stage but the last of
     every proximity test, and for every group but the last of the chain of segments (8.4, item 23,
     the third and the sixth of the further conditions; `iopp_fault`, `req_fault`).
  4. *Condition 3, parts (a) to (e)*, each by the count of free entries (section 9; 16.1, the room of masks at run time): for such a verifier s is an input, and with two or more answers that count is the stricter one. (a) A segment's mask row: 11.1, every
     verifier's. (b) A proof's index-mask message: 8.4, item 18, derives its number of mask rows so;
     `cp_verify` asks it under `plan-mask-row`. (c) A key file's index-mask message: `key_fault`.
     (d) The shared split's mask message: nothing but the ten asks it (16.1). (e) The
     multiplicities' message: `wf_fault`. After these the library has one guard, where an entry is
     taken and none is left (`blocks/iop/mask.h`).
  5. *Condition 6, first three lines,* in the stricter form of section 9: the folded row's vector
     with its tail, Omega's and W's each end before the first entry of a pad among the main rows of
     their own oracle message (`foldeq_fault`; `lin_fault`). The numbers of main rows of 8.4, item
     18, are derived so. The same is asked of the two folded rows of a leaf file's link, of which
     none of the ten speaks (`link_fault`, which asks `foldeq_fault`; 8.4, item 19). In the letters
     of section 9, with R main rows of l entries: the vector's length is at most R l and, where the
     pad is in a main row, at most the number that the pad's first entry has as an entry of the
     vector. The length is the number that 8.4, item 18, calls entries for a folded row, the number
     of rows of the matrix for Omega, and 2^k_col for W. The routine that counts this room gives 0,
     so that no vector fits, where the word on the high bits is 0 and R is not a power of two, and
     where ceil_log2(R) plus the logarithm of l is above 60 (`vector_room` in
     `blocks/iop/foldeq.h`).
  6. *Condition 9, part (c).* W's base code is Omega's, in all three numbers. On the first way the
     parser's test 33 (8.1) and the one tail of 8.4, item 18, give it. This page does not define
     parameters in which the two differ (under R13.19; 17.1, item 6).

  *What else the routines ask.* The six are the lesser share. These conditions are none of the ten,
  and by this page's reading of the source they carry soundness:
  - every oracle message of a request, of a key file's one-time chain, of the test on the table and
    of a leaf file's link gets an out-of-domain sample, that is, its number of answers is at least
    1; the fresh mask of a fold and the masking word of the last step alone get none (R12.10,
    R12.18). That number is the fourth item of R12.14 for a request, and the sixth word of their
    shapes for the folded row, Omega and W (R13.13, items 2, 8 and 9); that these three words are at
    least 1 is said on these pages here alone. For the two oracle messages that a key file's
    one-time chain sends before the test on the table, it is the sixth word of the notice: that of
    `index-mask message` (R13.5) and that of the multiplicities' message in `ranges` (R13.14). The
    two folded rows of a link and the oracle message of every switch have no such word: their number
    is the plan's sixth word. All messages of a request, those of R12.14, item 3, among them, have
    the request's number of answers (under R12.12; R12.14; `req_fault`, which asks it through
    `req_shape_fault`, where a comment says that the soundness argument needs it; `lin_fault`;
    `foldeq_fault`; `link_fault`; `iopp_fault`; `wf_fault`; `key_fault`);
  - some lookup with one digit of weight 1 covers every row of the matrix, no digit has the weight
    0, and no lookup has 2^w keys or more; a lookup's padding key is below 2^d, that is, a row of
    its table; d + w is below 64, the length in bits of field 2's prime; where a key has several
    digits the stride is not 0, so that its digits are in different rows; and every run is inside
    the matrix's rows, the last digit's last row included (under R13.14, in these letters;
    `wf_fault`);
  - 3 s < l in a stored table's shape (R13.8);
  - the first stage of the last test is the group of the last carrying message (R12.14, item 5;
    `req_fault`), and the first stage of the test of `ranges` has the codes of the multiplicities'
    message (under R13.14; `wf_fault`). The first stage of each test of R13.13, item 17, is the
    group that it tests as well: its two codes are those of the group's members, and its k and h
    follow from the members' numbers of rows as 8.4, item 18, says. None of the routines above asks
    this of these tests; the library has two guards, where an oracle message joins a group and where
    a test begins (`blocks/iop/relation.h`; `blocks/iop/iopp.h`). A link's first stage is not among
    the inputs: it is formed from the words of `the link` (under R13.10);
  - the six words of the tables' shape are the same in `the link` and in `the test on the table` of
    a key file and of a proof, and every stored table that a chain reads has that shape and was made
    under the chain's configuration (R13.9; under R13.10; `link_key_fault` in `blocks/iop/link.h`,
    which rejects; for a key file and a proof guards in `blocks/iop/foldeq.h` and
    `blocks/iop/lin.h`);
  - Omega has no mask rows, and a split's mask message has mask rows alone, at least one answer and
    the mask code of its groups (`lin_fault`);
  - the stages of every proximity test close: there are at least two; 2^k l is the l of the stage
    before; 2^h l' is at least 2 l'_0 + r_0 + 2 r'_0 + s; s is at most l'; h is at most k; the last
    stage has k = 0; a mask code is not longer than its base code (under R13.14, in these letters;
    `iopp_fault`);
  - a key file's binding tree has depth 0, and the multiplicities' message is the second oracle
    message of the one-time chain (8.4, item 23, step 8; `key_fault`).

  Further conditions of room: what a segment's closing carries on fits one main row, and S is below
  2^l and 2^lz (8.4, item 23, the seventh and the fifth of the further conditions; `req_fault`); the
  conditions on a block of the layout (R14.5; `req_layout_fault`). The lists of 8.4, item 23, and
  under R12.12 and R13.14 have more, and need not be whole (17.1, item 4). The parser's bound on a
  code's length, a <= 32 (8.1, tests 22, 29 and 36), is a condition on the codes too: the test under
  R12.1 admits a up to 60, and above 32 these pages give a code no points (R12.11). For tests 22 and
  29 the source's own reason is one of soundness, not of room: the proximity bound for a mask code
  that is repeated along the longer main domain needs that domain to be one coset of a subgroup (a
  comment in `blocks/frame/parameters.h`; section 8's head has it, and what the comment at test 36
  says).

  *What these pages have here alone.* The same routines ask the conditions of the next list, which
  these pages state as conditions at no other place, or there of one test alone
  (`toys/1-product.md`, part 4, 8.1, lists those of `iopp_fault`). Numbers that are derived by 8.4
  meet them, so on the first way nothing depends on them. By this page's reading of the source the three on leaf
  classes concern hiding alone, and the others are conditions of form and of room: where one of
  these is missed, these pages define no encoding, no group or no row. What the same routines ask
  that carries soundness is in the lists above. The words are those of R13.13, R13.9 and R12.14, and
  s is the plan's sixth word.
  - `foldeq_fault`, of the folded row's oracle message (R13.13, item 2). It has at least one main
    row, and its main rows are over E. Its base code is in `codes` (R12.2). The word on the stored
    table's columns (item 5) is at least 1. The routine first asks `index_fault` of the tables'
    shape (under R13.8).
  - `lin_fault`, which also asks `foldeq_fault`. Omega and W have at least one main row each, and
    their main rows are over E. Their base codes and the two mask codes of items 14 and 15 are in
    `codes`. Mask rows of the folded row's message are under the mask code of item 14, and mask rows
    of W under that of item 15. Where the folded row's base code is not Omega's (the split case of
    16.1), Omega's leaf class is not that of the shared split's mask message; where W's base code is
    not Omega's, W's leaf class is not that of a request's split's mask message. The first is a
    share of part (b) of condition 9 (16.1). The second is in none of the ten: part (c) shuts its
    case out (item 6 above). The bounds on the matrix's rows and columns are in the first of the
    further conditions of 8.4, item 23. The routine does not read the four tests of item 17: the
    routine that derives the numbers asks `iopp_fault` of each test that is made, and each test asks
    it again where it begins.
  - `link_fault`, of `the link` (R13.9). nu_0 and nu_j are at least 1. R is at least 1. The list of
    the later stages is not empty. The mask code is in `codes`. The routine also asks `index_fault`
    of the tables' shape; `foldeq_fault` of an oracle message with R main rows over E under the
    group's code, no mask rows and s answers; and `iopp_fault` of the test that begins with the
    first stage of R13.10.
  - `iopp_fault`, of every proximity test, beside what the list above has. Both codes of every stage
    are in `codes` and are codes (R12.1). The messages of both have a number of entries that is a
    power of two. k is at most 60 (field 2's bound: under R13.14). k is at least 1 at every stage
    but the last. Under R13.14 these pages have these conditions for the test of `ranges` alone;
    R13.19 has the last of them for every test, and the parser's tests 27 to 31 (8.1) ask part of
    them of the parameter file's lines.
  - `req_fault`, of each notice of R12.14, item 3; in a proof there is one, the index-mask message
    (8.4, item 23, step 8). It has no main rows and 1 to 2^20 mask rows. It has the request's number
    of answers. Its leaf class is above that of every message of every segment, which is more than
    part (a) of condition 9 asks. Its mask code is in `codes`, and s is below the number of entries
    of that code's message (8.4, item 23, the fifth of the further conditions). Of the chain itself:
    its two codes are in `codes`, and k is at most 60 in every group but the last.
  - `key_fault`, of a key file's index-mask message. It has no main rows, at least one mask row and
    at least one answer, and its mask code is in `codes`.
  - *The row of a pad.* The row that holds a pad (section 9) is not shorter than the pad: s is at
    most the number of entries of a mask row's message or of the message of a main row over E, and 3
    s is at most that of a main row over F. For a request's messages it follows from what
    `req_fault` asks: s is below both numbers of entries, and what a closing carries on, which is
    more than 3 s, fits one main row. For the stages of a test after the first `iopp_fault` asks it.
    For the multiplicities' message and a key file's index-mask message it follows from the count of
    their free entries, which is 0 where a mask row is shorter than the pad (`wf_fault`;
    `key_fault`), and for the shared split's mask message from part (d) of condition 3. For the
    folded row, Omega, W and a link's two folded rows none of the routines asks it, and no rejection
    has a name for it. There the condition is s <= 2^l, with l of `group folded`, of `group omega`
    and of the lines `group once-folded` and `group once-omega`; for a W that has a mask row it is s
    <= 2^lz, with lz of `group own` or `group once-own`. By this page's reading of the source the
    library has a guard alone (`blocks/iop/send.h`), and the count of a vector's room runs into it
    from inside `foldeq_fault` and `lin_fault`. With `answers` held at 1 no public file that the parser takes misses it; for a verifier that takes plan,
    codes and shapes as inputs it stays a condition. Such a verifier, if it has the two results of R0.3
    alone, accepted and rejected, rejects a file that misses it (section 9).

  *That plan and shapes fit each other.* 16.1 lists part of these conditions as the form of the
  plans; the programs ask that part with the ten, and it is not a condition on hiding. There is one
  round that samples positions for each stage of each test; the `LOC` of the stored table samples
  the shape's number of columns (R13.13); `LINK` samples nu_0 + nu_j positions (R13.9). Both sample
  from [0, 2^a), with a of the tables' shape: that is the ninth word of the round's entry (R4.3).
  The rules define these words by each other (R13.13, item 5; the table of R13.10), and none is
  written as a comparison. In the source the comparisons are guards (17.2, item 1: a verifier with
  two answers rejects there): where the plans are made (`blocks/frame/lin_frame.h`,
  `blocks/frame/link_frame.h`), and again where the columns are read (`blocks/iop/foldeq.h`,
  `blocks/iop/link.h`). Of `LINK` the plan's bytes have the sum alone. Where the list of positions
  is cut in two is the shape's word nu_0, and the round's error was computed for each part by itself
  (the next paragraph, `require_first_list`). The plan's sixth word is the number of answers in the
  shape of every message that exists, and its third word the salt's length in `binding tree` and in
  every stored table's shape (17.2, item 9). The plan's sixth word is also the number
  of stored answers for each row, the fifth word of a stored table's shape and name (R4.10; under
  R13.8); this too is a condition on the inputs. The programs derive both from the one parameter
  line `answers` (from the programs' source). Item 22 of 8.4 holds for every chain. Three conditions
  on the plan's own form belong here too, since by this page's reading of the source the routine
  that walks a plan tests none of them: the seventh of the eight words is 1 (R4.3); a plan has at
  most one entry `RECORD`, as its first entry, only where R13.15 says and with R13.15's words; and
  the second of the eleven words of an entry `SEAL`, its proof-of-work bits, is 0 (R5.3). That
  routine passes over every entry `RECORD` wherever it is (`blocks/numbers/plan.h`). The third is
  asked where a plan is made, as a guard, and not again at a run, which asks the same of a round of
  kind 7 alone (`blocks/numbers/plan.h`, `blocks/chain/rounds.h`). A fourth condition that routine
  does test, as a guard: the first of an entry's eleven words is 1 (R4.3).

  *That the errors belong to the inputs.* Every round's error was computed for the codes, the
  groups, the stored table with its distances, the numbers of rows and of combinations and the
  number of requests on which the verifier runs. The hashed bytes do not show it (17.1, item 3). In
  the source a plan notes beside a round from what its error was computed, and seven questions of
  the channel compare the notes with the run and end it if they differ (`require_rows`,
  `require_first_list`, `require_group`, `require_codes`, `require_table`, `require_combinations`,
  `require_sharing` in `blocks/chain/rounds.h`). Of these notes only the number of rows is among the
  plan's bytes, as the last of an entry's eleven words (R4.3). So a statement on a verifier that
  takes the plan as bytes has its own error terms, and this condition is on them.

  For a proximity test the condition reads: the stages for which the errors of its rounds were
  computed are the shape's own stages under the codes that are hashed. That is the same number of
  stages and, stage by stage, the same k and h and the same three numbers of the base code and of
  the mask code. No other place of this page states this as a condition. The source asks it where a
  plan is made (`iopp_sized_fault` in `blocks/iop/iopp.h`): of the tests on the table and of a
  link's test as a guard (`blocks/frame/lin_frame.h`, `blocks/frame/link_frame.h`), of the test of
  `ranges` with a rejection (`blocks/frame/wf_frame.h`), and of a request's last test as a guard,
  beside which the two codes of the segments' group are compared (`blocks/iop/req.h`). By that
  routine's own comment it does not reach a caller that hashes another list of codes in the run than
  it has planned with; the run stops there, where a stage begins (`require_group`). By the source's
  own word one thing in a request's plan has no such question at a run and is believed: the group
  with whose list the out-of-domain sample of the message of R12.14, item 3, was counted
  (`blocks/iop/req.h`).

  *What ties the inputs of one chain or file to another's.* But for one, on the tables' shape (in
  "What else the routines ask"), every condition above is on the plan, the codes and the shapes of
  one chain, or of the chains of one file. A key file has three chains, a leaf file one and a proof
  file three, and an acceptance rests on all seven (13.7). With every condition above met chain by
  chain, the inputs of two chains can describe two different matrices, tables or settings. On the
  first way that does not happen: one derivation, from one copy of the public files, gives the
  inputs of all seven chains, and every first query is made from its values (8.4; R0.5). On the
  second way each tie of the list below is a hypothesis of the statement, or the verifier makes the
  comparison itself and rejects where it fails. The first form of a statement (above) has the ties
  on plan, codes and shapes only because its one hypothesis is on all seven chains and on one copy
  of the public files. The ties on a statement's items (seals, roots, tables) are duties of every
  verifier, by the rules that are named at each; they are listed for a verifier, or for a statement
  on one, that takes whole first queries as inputs. In the source most of the comparisons are guards
  (17.2, item 1: a verifier with two answers rejects there), and the programs take all the values
  from one object, which one routine makes from the public files (`Configured` in
  `blocks/frame/configure.h`). By the source's own word the guards hold inside one process only, and
  nothing is enforced across processes or files, which is how `cp_verify` runs (the comment before
  `run_verifier` in `blocks/chain/closed.h`; 17.1, item 3; 13.6). The same comment has the ties
  between files in short: key file, leaf file and batch must be sized from one configuration (by
  this page's reading of the comment, the object that one derivation makes, not the three public
  files alone), since the error bounds of a later test and of a link are true only of a table whose
  one-time test ran at the same distances, target and proof of work. Like the lists above, this list
  need not be whole (17.5, item 9).

  - *Three items that two chains of one file both hold.* `codes`, `the test on the table` and
    `table` are byte for byte the same in a proof's first chain and in its shared chain, and in a
    key file's one-time chain and in its shared chain (R14.9; R13.5). In the source:
    `lin_same_public`, which `lin_second_verify` asks where the second chain begins, in a proof and
    in a key file (`blocks/iop/lin.h`; a guard).
  - *The seals and the binding tree's root.* The two seals in a second chain's statement are the
    seal of this file's first chain (in a key file, of its one-time chain) and the seal of this
    file's shared chain, in that order, as the verifier has computed them (R4.7; R14.10; R13.5). The
    value of `the binding tree's root` in the shared chain is the value that the first chain's seal
    gives with the slot, the salt and the siblings of the file, at the depth and with the salt's
    length of the first chain's item `binding tree` (R14.7; R13.6). A verifier computes all three
    and takes none of them as an input. In the source: `stands_on` (`blocks/chain/rounds.h`), which
    `lin_second_verify` asks of the shared chain, and which is asked of the first chain where the
    second chain continues it (`lin_go_on` in `blocks/iop/lin.h`; `blocks/iop/context.h`); both are
    guards. `lin_binding_root` (`blocks/iop/lin.h`) computes the root, and `check_proof` and
    `check_key_under` put its result into the shared chain's first query (`blocks/frame/check.h`;
    `blocks/frame/check_key.h`). By comments in `blocks/iop/lin.h`: the second chain takes the
    shared point, Omega's root and the row of Omega's out-of-domain sample from the shared chain,
    and what makes that sound is the shared chain's seal in the second chain's first query; the root
    is in the shared chain's first query so that the shared point depends on every request's claim,
    W and answers. A file is accepted only if each of its chains is. By a comment in
    `blocks/chain/rounds.h` a malicious prover knows beforehand what the programs give as the seal
    of a rejected chain, the digest of zeros, and can make a proof for the next chain on it, so a
    chain whose statement holds such a seal is rejected from its start.
  - *The depth is in one chain, its factor in another.* The depth of the binding tree is a word of
    the first chain's item `binding tree` (R14.6), and the shared chain has no such item. The factor
    2^depth belongs to the error of `LIN point`, which is a round of the shared chain (item 2
    above). In the source: `require_sharing`, where the point is sampled, with a number that
    `lin_bind_shared` takes from the root's own value (`LinBindingRoot` in `blocks/iop/lin.h`). By
    the comment there, the number of requests that can share a batch's point so reaches the shared
    chain from a hashed shape and not as a number that a caller states; the caller of
    `lin_binding_root` must pass the shape that the request's first chain holds, which by the same
    comment is not checked there; and the root fixes the depth, since a leaf's hash takes in the
    number 2^depth + slot and leaves and vertices are hashed under two numbers (R7.18), so that a
    path of another depth does not give the same root. By another comment in that file no type
    protects the number further: a caller could overwrite the public field `LinPublic::places` after
    the binding; and only a scan of the sources, whose list of names is not part of this repository,
    and not the compiler, keeps to tests the routine `made_up_without_a_tree`, which makes a root
    from any digest with a stated number of places.
  - *One configuration.* The item `configuration` has one value in all seven chains (R4.9). Every
    stored table that a chain reads was made under that value, and the leaves and vertices of the
    binding tree start from it (under R13.10; R7.18). In the source: a chain whose statement holds
    the seal of a chain with another value ends the process (`configuration_digest` of the class
    `Binder` in `blocks/chain/public.h`). That ties the three chains of a key file or of a proof,
    since the second chain has the seals of both others. `lin_shared_verify` compares the table's
    value with its chain's (`blocks/iop/lin.h`; a guard), and `link_key_fault` does so for both
    tables of a link (`blocks/iop/link.h`; a rejection). `published_id` does so for the anchor
    before it forms the commitment ID (`blocks/frame/configure.h`; a guard). Between files the value
    is the first of the key file's public values (below). By the comment in `blocks/chain/closed.h`,
    a key file or a leaf file that was made under another parameter file opens nowhere, since every
    leaf and vertex of a table's tree starts from the digest. What the digest does not cover is told
    at the key file's public values.
  - *One setting.* The plans of all seven chains are made under one setting. What the hashed bytes
    hold of a setting is among the first seven of a plan's eight words (R4.3), and each entry holds
    its round's proof-of-work bits. A plan's bytes do not hold the target (the line `sec`); the parser holds the lines `reading` and
    `rowcost` at 1. One item holds the target: word 172 of a proof's item `request` (R15.5). No item
    of a key file or of a leaf file holds it as a number: of that line these chains hash only the
    digest `configuration` (R4.9). So, by this page's reading of the source, the tie has two parts: the
    first seven words are the same in all seven plans, which is a comparison of inputs; and the
    errors of all seven plans were computed under one target and one table of proof-of-work bits, which is a condition on the statement's own error terms.
    Apart from sizes that `setting_of` leaves at the values of the field and the hash (the degree of
    E; the numbers of bytes of an element, a digest, a label and a hash block, and two sizes of a
    chain's query, `chain_extra_bytes` and `chain_free_words`), the source's setting has one more
    field that is in no hashed byte and that `same_setting` compares, `instances`. No line of a parameter file sets it: the programs leave `instances`, the
    number of requests by which the error of a request's round is multiplied, at 1, so that their
    bound on such a round is for one request alone (17.5, item 1) (by the comment there, outside
    tests only the planning tool `blocks/numbers/cp_plan.cpp` sets it, from the line `instances` of
    its own input file) (`blocks/numbers/setting.h`;
    `setting_of` in `blocks/frame/derive.h`). In the source: `same_setting` of the class `Plan`
    (`blocks/numbers/plan.h`), which compares every field of two settings. `check_key_under` asks it
    of a key file's three plans (`blocks/frame/check_key.h`; a guard). The routine that derives the
    plans asks it of every chain of the three files, against the one setting that it has derived,
    and rejects (`merged` in `blocks/frame/files.h`; `blocks/frame/configure.h`).
    `foldeq_columns_verify` asks it between the setting under which a stored table's own file was
    accepted and the plan of a chain that reads the table as a tested one: a proof's shared chain,
    and a link's chain for the anchor (`blocks/iop/foldeq.h`; a guard). Where a chain continues
    another, the third words of the two plans are compared once more (`blocks/iop/context.h`; a
    guard). Without it, in a test harness at toy size
    (`blocks/README.md`, "If you write a verifier on this library: a known limit"): a key file was
    tested at a lower target than the batch that relied on it, and the error counted for the batch's
    test was untrue by some 4.6 bits.
  - *The key file's public values.* A key file is checked under values of six items of its one-time
    chain: `configuration`, `codes`, `the test on the table`, `index-mask message`, `ranges` and
    `binding tree`. These six values come from the same derivation as the inputs under which the
    leaf file and the proof are checked. In the source: `key_public_bytes`
    (`blocks/frame/key_fixed.h`) encodes the six. A table that has passed a key file's check carries
    these bytes, and a leaf table carries those of the key file to which it is linked (`TestedTable`
    in `blocks/iop/foldeq.h`). `check_entry_under` compares them with those of its own configuration
    (`blocks/frame/check_entry.h`). `check_proof` does so twice, before the chains and after the
    caller's routine for the segments has run (`blocks/frame/check.h`). All three are guards. By the
    comment before `key_public_bytes` they guard one process only, and in the routines that
    `cp_verify` calls the comparison shows nothing about the key file's check (17.1, item 3). Why,
    by the same comment: some of these values, the ranges and the matrix's rows, follow from the
    public files by a routine of the calling program, which the configuration's digest does not
    cover; they are compared because otherwise a table whose key file was checked under laxer ranges
    could be used in a check that states stricter ones. By the comment in `blocks/chain/closed.h`,
    two such routines give two configurations with one digest, one setting and one commitment ID.
    Without it: two false acceptances with a matrix out of its ranges, in test
    programs at toy size (`blocks/README.md`, "If you write a verifier on this library: a known
    limit"). In the first a key file was checked under laxer ranges than the proof's check stated,
    by two configurations made from the same public files (a target of 128 bits with proof of work
    counted). In the second the caller's own routine swapped the same two configurations while the
    proof was checked. These pages define one such routine (under R13.14, the lookups of a model's
    matrix; 11.5). On the second way no object says which key file's inputs go with a proof's
    inputs; the next three ties say what of them a proof relies on.
  - *One matrix and one shape of the tables in the three files.* This tie is by this page's reading
    of the source. The tables' shape, k_lo and the number of rows of the matrix (R13.13, items 1, 6
    and 7) are the same in the key file's item `the test on the table` and in the proof's, and the
    tables' shape is item 1 of `the link` too (R13.9). The item's other words follow from other
    lines of the parameter file in the two files (the lines `group once-..` for a key file: 8.4,
    item 18) and need not be equal. For the tables' shape the source compares with what was kept of
    the earlier file ("What else the routines ask", above). For k_lo and the number of rows it has
    no comparison between files beside that of the key file's public values: it makes both shapes
    from the same two numbers (`lin_lines` in `blocks/frame/lin_frame.h`), and it makes the table's
    own numbers once, in an object that the sizing of all three files reads (`sized` in
    `blocks/frame/files.h`; `link_sized` in `blocks/frame/link_frame.h` rejects where that object is
    not the table of its own lines). What depends on it: `ranges` is in the key file alone, and the
    condition that some lookup covers every row is on the number of rows of the key file's own item
    (`wf_matrix_of` and `wf_covers_all_rows` in `blocks/iop/wf.h`; under R13.14). A proof whose item
    names more rows can hold claims on entries of the table that no lookup has read, with every
    condition above met chain by chain: the fourth line of condition 6 bounds the rows by the plain
    entries of a row's message alone (item 1 above), and a link compares all plain entries, whatever
    they hold. A proof whose item has another k_lo reads the same table as another matrix (11.5). Of
    the same fault inside a key file a comment in `blocks/iop/wf.h` says: ranges that count one row
    fewer than the test would leave that row unchecked, and every check would pass.
  - *The matrix of the segments' claims.* In a proof the number of rows of the matrix and k_col
    (R13.13) are m0 and lk of 11.5, from which a verifier forms the rows that the segments leave on
    the stored table. In the source: `wsPending` (`verifier/src/vf_weights_pending_claim.h`) ends under the name
    `internal-gamma-rows` where m0 is not the item's number of rows (17.2, item 1).
    `member_weights_on_rows` asks that a stretch of row weights is inside the matrix, and
    `member_rows_fit` that every such row's point has k_col coordinates and its weights one variable
    for each bit of a row's number (`blocks/iop/member.h`; guards). By the comment before
    `member_weights_on_rows` its guard does not hold against overflow: the first row's number plus
    the stretch's length can pass 64 bits, the wrapped sum can pass the guard, and the result is
    then empty, with no weights and no message. `check_proof` compares the number of these rows with
    the item `pending rows` (a guard). Why, by the comments in both files: the test on the table
    silently ignores the row weights of every row after the matrix's last one, so a layout with more
    rows than the item counts would lose its claims on the surplus rows.
  - *The width of a weight.* The key file's item `ranges` states of every weight of two digits, that
    is, of every number of every unit of 11.5 but the embedding's, that it is from -2^12 to 2^12 -
    1, and it reads such a weight as its low digit plus 256 times its high digit (under R13.14, the
    second lookup: d = 13, the digits' weights 1 and 256, the offset 4096, and runs that cover every
    unit of two digits; 11.5). None of the routines above asks that they do: `wf_fault` asks only
    that some lookup with one digit, of weight 1, covers every row of the matrix. The item also
    states of every cell of the matrix, so of each of the four digits of a weight of the embedding
    too, that it is from -2^7 to 2^7 - 1 (under R13.14, the first lookup: d = 8, the offset 128). Of
    that covering lookup `wf_fault` asks no particular d and no particular offset. A proof relies on
    a bound here as well: the head's no-overflow condition `J-E` takes each digit of a weight of the
    embedding as below 2^21 in size (the second half of 11.4), and no request's proof shows that. It
    is the key file's range proof, not the request's, that states a bound on these digits. So d = 8
    and the offset 128 of that covering lookup are a condition on `ranges` too; at the least, d and
    the offset together must give the bound that `J-E` takes: a lookup states that a digit is from
    -offset to 2^d - offset - 1, and every number of that range has to be above -2^21 and below
    2^21. A proof assumes both the bound and the reading by two digits: the no-overflow conditions
    of a request's sub-proofs take BW = 2^12 as the bound on such a weight, and the rows that the
    segments leave on the stored table read a weight by the same two digits (the second half of
    11.2; 11.5). No request's proof shows the bound. In the source: a comparison of constants that
    is made when `cp_verify` is built, which also has the 8 bits of a cell that the first lookup
    states (`verifier/src/vf_segments.inc`; `verifier/src/vf_widths_and_bounds.h`), and the rejection
    `policy-weight-kind`, which asks the number 13 of every line on the width of a weight of two
    digits (the second halves of 11.2 and 11.4). By the comment at that comparison, the key file's
    range proof must show what a request assumes: the width of a weight, and that a weight is its
    low digit plus a constant times its high digit. On the second way `ranges` is an input, so this
    is a condition on it.
  - *The same tables, and files that were accepted.* The item `the first table` of a link's chain
    holds the root and the stored answers of the key file's item `table`. The item
    `the further table` holds those at the beginning of the leaf file. The item `table` of a proof
    holds those of that leaf table, and `stored copy`, like the number in the proof's head, is the
    number k under which the verifier has that leaf file (R13.5; R14.2; R14.3; 13.6). The key file
    and the leaf file have themselves been accepted, on the same bytes, and the anchor is that of
    the commitment ID that the caller has named (13.2; 13.5; 13.7). In the source no caller gives
    the three items that hold a table: `link_verify` and `check_proof` form them from the tables
    that they are given, which are those that the checks of the earlier files have returned and, for
    the leaf table of a link, what `check_entry_under` has read at the beginning of the leaf file
    (`blocks/iop/link.h`; `blocks/frame/check.h`; `blocks/frame/check_entry.h`). `lin_shared_verify`
    compares the root and the stored answers of the table that it reads with its chain's item (a
    guard). The number k is the caller's: `check_proof` writes the number that it is given into
    `stored copy` and rejects a head with another number, and nothing in the library ties that
    number to the table (13.2). `check_entry_under` and `check_proof` reject a table that has passed
    no check, and `foldeq_columns_verify` has a guard for the same. Why, by comments in
    `blocks/iop/lin.h` and `blocks/iop/foldeq.h`: the error of a later test, and of a link on the
    anchor's side, is a true bound only of a table that has passed its one-time test, or a link to
    such a table.
  - *The distances.* The distances at which a stored table is read, in its one-time test and in a
    later test or a link (8.4, item 16), and for which the errors of the rounds on the table were
    computed, are the same for the key file's plans, for the link's and for the proof's. They are in
    no hashed byte: the tables' shape has six words, and the source keeps the distances beside them,
    outside the encoding (`IndexShape` in `blocks/iop/foldeq.h`). So this tie is on the statement's
    own error terms, like the conditions of "That the errors belong to the inputs". In the source:
    `foldeq_columns_verify` compares the distances with which the table's own file was checked with
    those of the run (a guard), and `require_table` compares the run's with the plan's note. The same note says for which test on the
    table a plan's errors were computed: the one-time test in a key file, a later test in a proof, a
    link in a leaf file.
  - *What a second chain takes from the two others.* A second chain's first query holds neither
    codes nor stages (R14.10; R13.5). Its test runs on the third list of stages of
    `the test on the table` and on `codes` of the chain that it continues, on that chain's
    index-mask message and W, and on Omega, whose shape and tree come from the shared chain (R13.7).
    Its plan is an input of its own. So "That plan and shapes fit each other" and "That the errors
    belong to the inputs" tie the second chain's plan to items of another chain. The errors are tied
    the other way too: the out-of-domain samples of the index-mask message and of W, in the chain
    that the second chain continues, and of Omega, in the shared chain, were counted with the list
    size of the group that the second chain tests, at the distance of that test, and Omega's with
    the larger of that list size and the list size of the group in which the shared chain tests
    Omega (`lin_size_of` in `blocks/frame/lin_frame.h`; `file_proof` and `file_key` in
    `blocks/frame/files.h`). Nothing at a run asks it (17.5, item 7). In the source one routine
    makes that plan from the object that holds these items (`lin_plan_second` in
    `blocks/frame/lin_frame.h`). The index-mask message is in another item than the test that reads
    it: in `request's shape` (R12.14, item 3) or in `index-mask message` (R13.5). It has mask rows
    alone and at least one answer, and its mask code is item 15 of `the test on the table`
    (`member_rows_fit` in `blocks/iop/member.h`; a guard; for a key file also under R13.14).

  Four things beside the list. More ties between the items of one first query are in 8.4, item 23,
  step 8. A verifier on the second way reads the parameter file all the same, for the item `request`
  (the model's lines and, of the frame lines, `version`, `sec` and `table`: R15.2) and for the
  segments' sub-proofs, so its inputs are tied to that file's lines as well. Words 172 to 176 of
  `request` are B of `sec` and w, b, n and m of `table` (R15.5): B is the target of "One setting", w
  is k_col, and b, n and m are v, a and the number of entries of a row's tail in the tables' shape
  (R4.10). The three tests of 8.3 that tie a frame line to a line of the model (`hid-vocab-cols`,
  `policy-tau-hb`, `plan-tau`) then speak of k_col, of that target, and of the number of
  combinations for which the error of `RNG` was computed (8.4, item 21; `require_combinations`). In
  the source nothing compares the five words with the shapes: `cp_verify` writes them from the frame
  lines as it has parsed them (`verifier/src/cp_verify.cpp`). And the target and the
  proof-of-work bits under which the errors were computed ("One setting", above) are those of the
  lines `sec` and `work ..`, of which conditions 2 and 5 of 13.7 are asked;
  else these conditions are asked of lines that the errors do not follow. One comparison between
  chains is on the bytes of files and not on inputs, and every verifier makes it: a vertex has one
  value under one root, whichever chain or file the root came with (R7.14). And one tie between the
  shapes of the three files is, by this page's reading of the source, on hiding alone: condition 5
  of 16.1 adds up the columns that the key file's test, the links and the batches open on one stored
  table.
