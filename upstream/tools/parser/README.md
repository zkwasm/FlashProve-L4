# tools/parser: from the three files to typed values for Lean

A verifier in Lean that takes typed values cannot read a file's bytes. The scripts in this directory
turn a [key file](../../docs/glossary.md#key-file), a [leaf file](../../docs/glossary.md#leaf-file)
and a proof file into **parsed files**: flat files with lengths, which a short reader written in Lean
can load without a hash and without a plan. [PARSED-FORM.md](PARSED-FORM.md) defines them.

## What is trusted, and what is not

- **The parser is untrusted for the three files that it parses:** the key file, the leaf file and
  the proof file. All three come from the prover's side, so whatever the parser makes of their bytes
  is just the prover's last step.
- **The claim does not pass through it.** What the user asserts is four things: the
  [commitment ID](../../docs/glossary.md#commitment-id); the
  [request](../../docs/glossary.md#request), which holds the prompt and the reply; the
  [next token](../../docs/glossary.md#next-token) (the
  [token file](../../docs/glossary.md#token-file)); and the three public files of the
  [configuration](../../docs/glossary.md#configuration), which fix the protocol's description. They
  must reach the verifier straight from the user: typed by the user, or read from files that the user
  holds, by code that is part of what is trusted. They must not be taken from the parser's output.
  Otherwise the verifier can soundly check a claim other than the one that the user meant.
- So the parser turns the key file, the leaf file and the proof file into parsed files, and the ID,
  the request, the next token and the public files go in beside them. No parsed file holds the user's
  copy of any of these.
- One message of the proof itself states the next token: the prover sends it in the round `REPLY`
  ([FORMAT.md, 12.4](../../docs/format/FORMAT.md#124-the-next-token)). The parser copies it like
  every other message. The verifier has to compare it with the next token that the user gives.
- The root and answers in `proof.parsed.bin` are the prover's word (copied from the leaf file that
  the proof's own head names). They are worth nothing for the user's commitment ID until a verifier
  has tied them to it
  ([PARSED-FORM.md, section 2](PARSED-FORM.md#2-a-key-file-and-a-proof-file-seventeen-fields)).
- The parser is *given* the public files and the request, because it cannot cut the files without
  them (next section). It hands on none of their bytes. But it hands on values that it computes from
  them: the cut itself and hash values. They are the prover's word like the rest.
  [PARSED-FORM.md, section 6](PARSED-FORM.md#6-what-is-computed-and-not-copied) lists every field:
  whether it is cut or computed, from what, and whether a verifier can make it again.
- With `--lines` the parser hands on more, among it digests of the public files and the SHA-256 of
  the request file ([PARSED-FORM.md, section 8](PARSED-FORM.md#8-what-a-parsed-file-never-holds)).
- Where a bound on soundness counts the prover's queries to a hash, the parser's queries count as the
  prover's, to the format's hash and to a second hash mode of our own, built on BLAKE2b's compression
  function, alike.

These sentences say how the parser is meant to be used. Nothing in this directory proves them.

## Why a parser has to hash

No file of the format holds a length, a count or an offset
([FORMAT.md, 7.1](../../docs/format/FORMAT.md#71-ground-rules)). The plan fixes the length of every
message, nonce and slot. It does not fix the length of an opening: that depends on the positions
that were sampled, and these come out of the hash chain. So the parser replays every hash chain of
a file, with the same hash as the verifier, to find where each opening ends. The first chain of a
proof has the request in its first query, which is why the parser needs the request.

Some values in a parsed file are in no original file and come out of the same replay: each chain's
run's value, the binding tree's root, and the whole path of each opened leaf (a file holds only the
digests that cannot be computed from the other opened leaves). The seeds of a table's
[record](../../docs/glossary.md#record) are hashed too, in a chain of their own that starts at the
table's root, and go into a file of their own. The programs, their files and the format say "the
record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say
"the record's seeds" for its OOD seeds ([glossary](../../docs/glossary.md#record)).

## Run it

From the repository's top directory:

```
W=$(mktemp -d) && echo "W=$W"
tools/parser/parse_files.sh examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.proof "${W:?first run the line with mktemp}/work"
```

The first four arguments are `cp_verify`'s CONFIG_DIR, PUBLIC_KEY_DIR, REQUEST_FILE and PROOF_FILE;
its token file is not among them. The fifth names a directory that must not exist:
the script creates it, writes only there (the compiler also uses the system's temporary
directory) and removes nothing. It refuses a directory below a
place that another account could change; the one that `mktemp -d` makes is fine. Run it as an
ordinary user, not with sudo.

It needs bash, GNU coreutils, g++ 13 or later and python3. It first compiles the verifier's source and
`tools/audit/pubtool.cpp`: allow about 1.5 GB of memory and a few minutes. It exits with code 0, and its last line begins `PARSED AND READ BACK:`
and names the directory with the parsed files:

| file | what it is |
|---|---|
| `key.parsed.bin`, `leaf-<k>.parsed.bin`, `proof.parsed.bin` | the parsed files; k is the number in the proof's head |
| `<parsed file>.seeds` (for a leaf file also `.seeds0`) | the seeds of the record of the table whose root the parsed file holds (`.seeds0`: of the anchor's). It is a control ([PARSED-FORM.md, section 4](PARSED-FORM.md#4-the-seeds-files)) |
| `<parsed file>.order` | which round opens which tree at which positions, and what else is needed to make the original file again. It is for tests. It also holds the 32 random bytes of each chain that has some, with which a verifier can make these chains' runs' values again |

A file that the parser cannot cut is refused: exit code 1, a last line that begins `REFUSED: `, and
no parsed file. [PARSED-FORM.md, section 7](PARSED-FORM.md#7-what-the-parser-checks) lists what is
refused. Not every damaged file is: see [Limits](#limits). Use parsed files only from a call that
ended with exit code 0: one that a signal or a full disk ends while it writes can leave some of them.

| script | what it does |
|---|---|
| `parse_files.sh` | the call above |
| `parse_files.py` | the parser. With `--lines` it also writes every item of each file, by name, as JSON lines, for people: `key.lines.jsonl`, `leaf-<k>.lines.jsonl`, `proof.lines.jsonl` |
| `read_parsed.py` | a second program that reads the parsed files, which shares with the parser only the hash of `docs/format/format_rules.py`: it follows every path to its root, makes the original file again and reads the seeds files |
| `parser_test.py` | the test of the two, on a stored example and on damaged copies of it ([SUITE.md](../../SUITE.md), `tools.parser.small`) |

## Limits

- The parser makes only the checks that it needs to cut the files
  ([PARSED-FORM.md, section 7](PARSED-FORM.md#7-what-the-parser-checks)). A file that parses has been
  checked for nothing but its cut: that says nothing on whether it proves anything. A damaged file,
  or one made up by someone who replays the hash chains, can be parsed.
- It does not derive a plan. The plans come from `tools/audit/pubtool.cpp`, which runs the C++
  verifier's own configuration code on the three public files.
- It cuts the files with `tools/audit/cut.py` and `docs/format/format_rules.py`, so it shares their
  limits: it finds by trial which trees are opened after a round that samples positions.
  One opening after such a round is enough for it. So a file that leaves out a further tree's
  opening, or opens another tree than the plan asks for, is parsed if the rest still fits. The parsed
  file then lacks rows, or has rows that no round asks for: the verifier has to notice that itself.
- A verifier behind the parser judges the parsed file, not the original file. One original file has
  many parsed files: this parser writes one, and another hand can fill entries of a path otherwise
  where the verifier does not read them. And several original files can have one parsed file: a file
  that leaves out an opening whose leaves are all opened elsewhere in the file parses to the same
  parsed file as the whole file. Such a verifier is handed the very same parsed file, so it answers on
  the same claim and accepts, while `cp_verify`, which reads every given digest and refuses a byte
  left over, refuses the file (by a reading of the code: no such file has been made). So the two need
  not agree on every file. On files as `cp_prove` writes them they can say the same. (An original
  file in which a given digest is altered does not get through this parser: it follows every opening
  by itself to its root.)
- By [FORMAT.md, section 13](../../docs/format/FORMAT.md#13-what-a-verifier-does-in-order) a second
  verifier must make the same tests as `cp_verify` on the key file, the leaf file, the proof file, the
  request file and the token file, and the tests of 13.8 on the three public files. The parser with a
  verifier behind it is not a second verifier in that sense: that verifier reads parsed files, and on
  the bytes of the original files only the parser's few checks are made.
- It reads each file whole into memory, and refuses one of more than 2^28 bytes (a full-size proof
  file has about 19 MB).
- `read_parsed.py` does not check what enters neither a path nor the original file. Its first lines
  list what that is.
- If `read_parsed.py` refuses what the parser wrote, which would be a fault of these scripts,
  `parse_files.sh` ends with that refusal and the parsed files stay in place.
- The parsed files are input for readers written in Lean, which lay them out as the typed values
  that checks written in Lean take. They are tests: none of them is a statement of soundness.
  `lean/Ask/README.md` and the comments of the files beside it say how to run them and what a yes
  does not mean. `cp_verify` reads the original files, not parsed ones.
