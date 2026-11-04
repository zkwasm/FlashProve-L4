# docs/format: the file format

This directory defines the files that the programs in this repository write and read: a key file, a
leaf file and a proof file, with the public files that fix their shape. It also tells what a
committer and a prover do that no verifier can see.

*Which page decides.* On bytes, hashed strings, order and tests [FORMAT.md](FORMAT.md), with the
files that hold its sections 8 to 11, takes precedence over every other page of the repository. That
includes the pages on the prover's side.

*Sources.* Many of the methods that these pages describe come from published work, and these pages
name few of the sources. `PUBLISHED-WORKS.md`, at the top of the repository, lists, under "Published
methods of the proof system", the publications that comments of the programs' source cite, each with
files that name it. Other sections of that file cover further sources, first among them "The paper
that this protocol builds on".

## What a verifier reads

| file | what it is |
|---|---|
| [FORMAT.md](FORMAT.md) | the format's main page: the objects; bytes; the hash; chains; sampling; trees and openings; the three files (sections 1 to 7); the request, the next token and the acceptance policy; what a verifier does, with every rejection by name; interface and results; what an acceptance states; duties that no verifier can see; the page's limits; notes for a formal definition (sections 12 to 18) |
| [08-parameters.md](08-parameters.md) | section 8: the parameter file, and what is derived from it |
| [09-key-and-leaf-files.md](09-key-and-leaf-files.md) | section 9: a key file and a leaf file, round by round |
| [10-proof-file.md](10-proof-file.md) | section 10: a proof file, round by round |
| [11-sub-proofs.md](11-sub-proofs.md) | section 11: the sub-proofs of a segment: its head, with links to its five parts |
| [11-1-machinery.md](11-1-machinery.md), [11-2-feed-forward-segment.md](11-2-feed-forward-segment.md), [11-3-attention-segment.md](11-3-attention-segment.md), [11-4-output-head.md](11-4-output-head.md) | 11.1 to 11.4: for the shared machinery and for each kind of segment: the rounds; the cells, the lookups' keys and every claim |
| [11-5-weights-in-the-table.md](11-5-weights-in-the-table.md) | 11.5: how the weights are placed in the stored table |

## Scripts and known answers

| file | what it is |
|---|---|
| `format_rules.py` | sections 2 to 6, one known answer for section 7, and the encodings and rounds of sections 9, 10 and 12, as a Python program; it uses only Python's standard library and imports nothing of the programs; the one part of a hash that it writes out itself, BLAKE2b's compression function for R15.3, is adapted from the BLAKE2 reference implementation, as the programs' own is |
| `read_key.py`, `read_entry.py`, `read_proof.py` | walk a key file, a leaf file or a proof file with the functions of `format_rules.py`; what they do not do is in FORMAT.md, 17.3. `read_key.py` and `read_entry.py` read the toy files of the library `blocks/` alone (a configuration of two public files, with the parameter files that the library's tool writes beside them): they do not read a `key.bin` or a leaf file of `cp_commit`, such as those in `examples/`. `read_proof.py` reads a proof of `cp_prove` with its options `--public`, `--table` and `--statement`, and with the chains' parameter files, which a tool of the tests writes (`verifier/tests/first_queries_tool.cpp`); which proofs it has read is in FORMAT.md, 17.3 |
| `format-vectors.txt` | known answers: what `format_rules.py` prints |

The parameter files that section 8 names are in `blocks/frame/examples/` and in `prover/examples/`;
whole configurations (parameter file, layout file, scale file) are in `prover/examples/`, in five of
its six directories. The sixth, `target_small_file_f2`, has no `layout.txt`, so it is not a
complete configuration ([docs/glossary.md](../glossary.md#example-configuration)). In a
configuration directory the programs read the file named `policy.txt` alone (13.3), so a call under
another parameter file needs a copy of the directory in which that file has the name `policy.txt`.
The planning call `cp_verify -plan request` alone is given its parameter file by path (14.6).

## What a committer and a prover do

| file | what it is |
|---|---|
| [COIN-KEYS.md](COIN-KEYS.md) | how the committer and the prover of this repository derive their private randomness; not part of the format (FORMAT.md, section 16) |

## Names on the protocol page and in the format

The format uses other names than the [protocol page](../protocol.md) for some things. Names in code
marks are literal strings that are hashed, so they never change; the two in the fourth row,
`published_id` and `format_rules.py`, are a function and a file of the scripts, and the two in the
sixth row, `smallness` and `work smallness`, are a word of the programs' output and a line of the
parameter file.

| the protocol page | the format |
|---|---|
| weight matrix | the committed matrix of the weights' digits (11.5) |
| weight table: anchor; leaf table | stored table: anchor; leaf table. In the bytes: `table`, `the first table`, `the further table`, `stored copy` |
| query bound | the number of random entries of a row, its random tail (R4.10) |
| commitment ID | commitment ID (R4.11); `published_id` in `format_rules.py` |
| range proof; one-time test | the one-time chain, with the item `ranges`; a key file's shared and second chains |
| range check (not the range proof of the row above) | the smallness test (11.1); `smallness` in the parameter file's line `work smallness` (section 8) and in the programs' output |
| leaf file; link proof | leaf file ("entry" in rejection names); link: the item `the link`, the round `LINK` |
| request | request, the items `request` and `the request file's words` |
| next token | next token; the round `REPLY` |
| segment; trace | segment, with its sub-proof (`group link`, `stages link` in the bytes); its oracle messages |
| request vector w; shared vector | W; Omega |
| test on the table | `the test on the table` |
| batch tree; seal | `binding tree`; `the seal of an earlier chain` |
| a chain's input | a chain's first query: parameters and statement |

## The library's blocks, with small worked runs

| file | what it is |
|---|---|
| [toys/1-product.md](toys/1-product.md) | the toy run `product`; parts 1 to 4: the verifier's side of sending oracle messages with their out-of-domain samples and of batching rows, of the sumcheck with a hiding polynomial, of the fold and the last stage, of the code switch and the proximity test on a group, as mathematics and bytes |
| [toys/2-links.md](toys/2-links.md) | the toy run `links`; part 5: the frame of a chain of segments |
| [toys/3-key-and-entry.md](toys/3-key-and-entry.md) | the toy run `key-and-entry`; parts 6 to 10: the key file and the proof that the table's entries are in their ranges, the lookup by product trees, the batching of claims on the matrix, the test on a committed table, the leaf file and its link |

A toy run is a small run of the library `blocks/` on synthetic data.

One of the library's tools, `blocks/frame/handover_proof`, makes toy proofs on a leaf table, under
the number 1, and also on the key file's own table, with the number 0 in the head. The second kind
are a test tool's files and no files of this format: by R14.2 a proof is made on a leaf table, for
the number 0 as for every other.
