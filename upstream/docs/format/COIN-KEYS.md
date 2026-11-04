# How the committer and the prover derive their private randomness

This page is a rule for the programs of this repository that commit and prove. It is not part of the
format: no verifier derives any of these keys, and no byte of a key file, a leaf file or a proof
file says how they were made. It is written for someone who audits or re-implements the committer or
the prover. Rule numbers such as R4.9 point to the [format page](FORMAT.md).

Sentences of this page that are not marked "(from the programs' source)" have not been held against
the programs' source.

Hiding depends on these keys all the same, and no verifier, honest run or ordinary test can see a
fault in them. A random value that is used twice, under other data or other challenges, can reveal
hidden values while every proof still verifies. The duties that follow for every committer and
prover, whatever their way to make randomness, are in section 16 of the format page and in R4.12.

The **private random values** are the random entries of encoded rows, mask rows, pads, hiding
polynomials, and the salts of leaves and the random bytes of chains. All of them are derived from
one secret seed and from bytes of the operating system's random source, by ChaCha20's block
function. So are three more values of a batch: the choice of its leaf table, the arrangement of its
requests over the slots, and the digests of the binding tree's free leaves (section 2a; from the
programs' source). One option of a production build, `-public-weights`, gives up both the secret
seed and, in `cp_commit`, the bytes of the system: the seed is then derived from a file that may be
published, no random byte of the operating system goes into any file of the commitment, and nothing
is hidden (section 2, "A seed from a public file"). A **commit run** is the part of a call of the
committer that builds one stored table, one key file or one leaf file; one call can hold several.
Sections 1 and 2 give what a key may read and the schedule of the keys. Section 2a says how a random
value comes out of a key. Section 3 says what hiding rests on, and section 4 where all this is in
the programs. The comments of `blocks/coins/coins.h` tell the first part of section 2a, below a
process key, at more length.

## 1. The facts that a key may read

Every input but one is a fact that is defined elsewhere; the one is a number that this page defines
(the last row). None is a digest of an object that exists for the keys alone.

| sign | what | where it is defined |
|---|---|---|
| c | the configuration's digest: of the parameter file (which holds the format's version), the layout file and the scale file | R4.9 |
| S | a stored table's shape: the words that its name holds after its two digests | R4.10 |
| name | a stored table's name: root, c, S | R4.10 |
| answers | the table's stored answers | R4.8 |
| ID | the commitment ID | R4.11 (the hash's number 179) |
| which table | the anchor, or a leaf table with its number k, from 0 | |
| the requests | of a batch, as the service has read them: clients' data | |
| the prover's revision | a number that stands for how a version of `cp_prove` turns its inputs and random values into a proof's bytes | section 2, "The prover's revision" |

## 2. The schedule

    K = take( take( child(kind, number), the 8 words of 32 bytes ), the 8 words of I )          I = H(176, 0, 0; item, item, ..)

- key(K; a, b, c, d) is words 0 to 7 of ChaCha20's block function (RFC 8439, 2.3) under the key K,
  with a instead of the block counter and b, c, d instead of the nonce.
- child(kind, number) = key(seed; "CPSD", kind, number, 0). **take** feeds eight words, four to a
  call, each call under the key that the call before has given: take(K, w) = key( key(K; w_0, w_1,
  w_2, w_3); w_4, w_5, w_6, w_7 ). 32 bytes are eight words of 32 bits, lowest byte first.
- H(176, 0, 0; ..) is the format's hash (section 3 of the format page) under the number 176, tree 0,
  index 0, without a start.
- An **item** is: the length of its name in 8 bytes (lowest first), the name, the length of its
  bytes in 8 bytes (lowest first), the bytes. A list of items can be read back from the hashed
  string in one way only. The first item of every list is the key's **title**: a name without bytes.
  A number is 8 bytes, lowest first; a digest is its 32 bytes; S is its words of 8 bytes each, in
  the name's order; a request is its words of 4 bytes each.

The titles and the items' names are hashed, so they are bytes: they are given letter for letter. The
item `the published ID` holds the commitment ID. In these names "the first table" is the anchor,
"copy" and "stored copy" a leaf table, "entry" a leaf file, and "making" a commit run.

| key | child | the 32 bytes | title, then the items |
|---|---|---|---|
| the anchor's commit run | (1, 0) | zeros, a constant of the rule: this key has no secret input but the seed, which is sampled in the call of `cp_commit` that makes the anchor and in no other (under `-public-weights` it is derived in that call from the option's file, and anyone who has that file can compute it: "A seed from a public file", below) | `a stored table's making`; `the configuration`: c; `the shape`: S; `the first table` |
| leaf table k's commit run | (7, k) | from the system's random source, taken for this commit run, kept with the leaf table's secret files (under `-public-weights`: zeros, and the kept file `fresh.bin` holds 32 zero bytes) | `a stored table's making`; `the configuration`: c; `the shape`: S; `copy`: k |
| the key file's commit run | (8, 0) | from the system's random source (under `-public-weights`: zeros), kept nowhere (except under `cp_commit`'s option `-keep-for-the-audit`, until the commitment's audit says yes; at every other ending of that audit they stay: section 3) | `the key file's making`; `the published ID`: ID |
| leaf file k's commit run | (5, k) | from the system's random source (under `-public-weights`: zeros), kept nowhere (except under `cp_commit`'s option `-keep-for-the-audit`, until the commitment's audit says yes; at every other ending of that audit they stay: section 3) | `an entry's making`; `the published ID`: ID; `the copy's name`: leaf table k's name; `the copy's answers`: its answers |
| a batch's first key | (6, 0) | from the system's random source, kept nowhere (except under `cp_prove`'s option `-keep-for-the-audit`, for one batch, until the audit erases the kept bytes, which it does only when called with `--erase-kept-files`, and not always then: section 3) | `a batch`; `the configuration`: c; `the prover's revision`: 1 (below the table); `copies`: the number of all leaf tables of the commitment, live or not; `the first live copy`: the number of the first live leaf table (live: section 2a, "Below a batch's first key"); `slots`: the length of the batch's list of requests; then for each number of that list `request`: the request's words, or `no request` |
| a batch's process key | none: take( key(the first key; "CPBK", 0, 0, 0), the 8 words of I ) | | `the batch's stored copy`; `the copy's name`: the name of the leaf table that was chosen; `the copy's answers`: its answers |

The items `the first table` and `no request` have no bytes.

*The prover's revision.* The item `the prover's revision` is a number that this page defines. It
is 1; counting begins at 1. Whoever makes a change after which `cp_prove`, on equal
inputs (files, seed, system bytes, batch, options), sends another byte or draws its random values in
another order or number (a change of what a proof's chains hash is one), while the format's version
word stays, raises this number by one in the same change, wherever it is written. One number stands
for one behaviour of `cp_prove`: several such changes may share one raise only if no build made
between them has served a secret key directory that matters. Nothing in the programs or the tests
checks that. The number is written on this page (here and in the row `a batch's first key`), in
`prover/common/coin_keys.h` (`PROVER_REVISION`), in `prover/second_writing/schedule_keys.h`
(`kProverRevision`), in `prover/tests/coin_keys_values.py` and in `tools/audit/coins.py`. Three
known answers change with it: a batch's first key, its process key, and the leaf table drawn under
the first key. They stand in `prover/common/cp_keys_own.h` and in
`prover/tests/coin_keys_values.txt` (section 4).

The item matters only if the system's 32 bytes repeat (section 3). Two versions of `cp_prove` that
serve one secret key directory and get the same batch would then, without the item, derive the same
first key, choose the same leaf table, draw the same random values, and send them under different
challenges. With different numbers the two lists differ, and so, short of a collision of H or of
take(), the first key differs. The batch's random values are then unrelated as far as ChaCha20's
block function is a pseudo-random function: the assumption on which every key of the schedule rests
(section 3). The same leaf table can still be drawn, and always is where only one is live. Its
lasting random entries are made by no batch's key and serve many batches by design: only the
directory's count of shown columns guards them, and two copies of one saved state each go on with a
count of their own (docs/limits-all-rows.tsv, row H4.1). So the item ends one harm of such a repeat
and leaves that one.

`cp_commit` makes no batch, so none of its keys reads the item. The key file's and a leaf file's
commit runs have chains too, and their keys have no such item: a version that changed what those
chains take in, under the same version word, would not be kept apart from the version before it.

The first two keys are **a table's commit-run keys**. The key file's commit-run key, a leaf file's
commit-run key and a batch's process key are **process keys**. Section 2a treats the two kinds
apart.

*Four letters as a word (from the programs' source).* The eight words that are written as four
letters follow two conventions. In "CPSD", "CPLF", "CPBK", "CPKW" and "CPPW" the first letter is the
lowest byte. In "MASK", "SALT" and "PADS" the first letter is the highest byte.

| letters | word |
|---|---|
| "CPSD" | 0x44535043 |
| "CPLF" | 0x464C5043 |
| "CPBK" | 0x4B425043 |
| "CPKW" | 0x574B5043 |
| "CPPW" | 0x57505043 |
| "MASK" | 0x4D41534B |
| "SALT" | 0x53414C54 |
| "PADS" | 0x50414453 |

From a table's commit-run key come its random entries, its leaves' salts and its
pads: all from the one secret. The leaf table that serves a batch is chosen from the batch's first
key (section 2a).

*The record.* Every stored table has a record. A table's pads are
a third family of inputs below its commit-run key, "PADS" (section 2a, "Below a table's
commit-run key"). What hiding asks of
the pads is in R4.12 and in a comment of `blocks/iop/foldeq.h` (in `IndexHolder`): a set of pads
serves one record only. On a
model's files `cp_commit` and `cp_prove` take a parameter file that says `answers 1`.
`cp_commit` and `cp_prove` reject a parameter file with any other value of `answers` than 1 (`parameter-range`), under `-public-weights` too: the written proof covers one OOD answer a row, so two to four are outside it, and this project's proofs cover a table only where it has one answer; no attack on a table with more is known, and nobody has searched for one.
On a model's files, without that option, `cp_commit` prints a note before it opens a file of
the model; among its words: "never run cp_commit twice from one saved state of a machine (a cloned
or restored image): two commitments with the same masks over different weights give the weights away
in part" (from the texts that the two programs print; `docs/operators-duties.md`, "A record").

*The copy's answers.* `the copy's answers` holds the answers of the table's record as the key
file or the leaf file has them after the root: by row and within a row by answer, 24 bytes each,
without their number as a word before them (R4.11 has the order). The key file's commit-run key reads the anchor's answers through
the ID (R4.11). Leaf file k's commit-run key reads leaf table k's answers
(`prover/hash_commitment/hf_commit_keys.h`, `take_an_entrys()`), so a leaf table's answers are made
before the random values of its leaf file are keyed. A batch's process key reads the answers of the
leaf table that was chosen. A table's commit-run key reads no answer: the seeds of a record are
sampled from the table's name, which holds the root (R4.8, R4.10), so the answers are made after the
table. It reads their number through the shape S. `cp_commit` reads the finished leaf file's first
part back and compares its record's answers with the bytes that it passed; a difference is an
internal error, and the leaf file is written nowhere (`prover/hash_commitment/hf_commit.cu`). No
test compares the bytes that `cp_commit`'s and `cp_prove`'s calls pass. If `cp_prove`'s bytes
differed all the same, nobody would notice and nothing would break: they feed two different keys,
which are never compared with each other. A batch's first key has 32 bytes of its own from the
system's random source; the answers matter in the process key only as a guard against a random
source that repeats (section 3).

*A test seed.* Under a test seed nothing is hidden: `cp_commit` uses zeros instead of the 32 bytes
from the system, and a test build's `cp_prove` can read a batch's 32 bytes from a file
(`-test-coins`). One seed file that is given to two runs on different weights under the same three
public files makes two tables with equal random entries, salts and pads: no table's
commit-run key reads anything of the weights beyond these files (section 3 says what equal pads give
away). The programs keep such a seed out of service in four ways. A production build has no option
that takes a test seed or a test's random values. Only a test build accepts one, and a test build
rejects a real model's files. A commitment that was made under a test seed says so in its secret key
directory, in the line `seed test` of `isk.txt`, and a production build's `cp_prove` rejects such a
commitment (`cp-isk-seed-test`). No key protects that line: whoever can write into the directory can
remove it. Both `cp_commit` and `cp_prove` reject a seed that is written in the source, outside a
test's call, under the rejection name `hc-key-test` (`prover/hash_commitment/hc_seed.h`,
`seed_draw()`; `prover/prover/isk.cuh`). `seed_draw()` rejects 32 zero bytes under the same name
(from the programs' source).

*A seed from a public file (from the programs' source).* A production build's `cp_commit` and
`cp_prove` have the option `-public-weights <file of 32 bytes>`, for weights that are public anyway.
It is not a test's option, and no other build has it. With it `cp_commit` does not sample the seed:
the seed is key(the file's 32 bytes; "CPPW", 0, 0, 0). No random byte of the operating system then
goes into any file of the commitment: the 32 bytes of a leaf table's, the key file's and a leaf
file's commit run are zeros. So anyone who has the directory of the weights, the configuration and
the file gets the same key file, the same leaf files and the same commitment ID, and can compare
them with published ones. Equal values would not show that a commitment
is right, and they would say nothing about a GPU model, driver, tool chain or set of CUDA libraries
other than those of the makings. The price: anyone who has the file can compute every mask of the
key file, of the leaf files and of the stored tables' columns that proofs open, so the weights are
not hidden (the source's own words, and this page's short form "nothing is hidden", are wider: "can
compute every mask, so such a commitment hides nothing"; a batch's 32 bytes come from the system's
random source under the option too: the schedule's row of a batch's first key). So a batch's own
random values do not follow from the file alone: they depend on the batch's 32 fresh bytes, under a
key that anyone who has the file can compute. The key file, the
leaf files and the commitment ID bear no mark of that, and by `cp_commit`'s usage text the proofs
look like any other commitment's. An operator
MUST NOT use the option for weights that must stay hidden. Why the seed is derived and is not the
file's bytes themselves: `cp_commit` accepts any file of 32 bytes, and with the bytes as they are a
call under the same configuration that named another commitment's `seed.bin` would repeat the masks
of that commitment's anchor; on the same weights it would make the anchor again and would open
further columns of it without raising its counter of opened columns. Every key of a commitment is
derived from its seed under the first word "CPSD", so, by the source, the derived seed shares no key
with the commitment whose `seed.bin` was named (`prover/common/public_weights.h`;
`prover/prover/arguments.h`; `prover/weights_commitment/arguments.h`).

*What guards that option (from the programs' source).* By the source no single mistake may lead to
such a commitment. To make one from a real model's files, `cp_commit` needs the option and, next to
each model file, a marker file of the same name with `.weights-are-public` appended, which holds the
model file's SHA-256. It rejects a marker that its group or others can write. By the header's own
two lists, at its head and before `marker_owner_counts()`
(`prover/hash_commitment/public_weights_markers.h`), it does not detect six things: a marker that
was created for weights that must stay hidden; a file that anyone who can write to the directory
renames to the marker's name, if it begins with the right 64 digits and belongs to one of the three
accounts whose markers count (the account that runs `cp_commit`, the model file's owner, root);
owners that are not what they seem (a mount or a container that shows several accounts as one, a
file system that lets an account give its files away); what root or the model file's owner does; a
symbolic link that another account puts under the marker's name between the operator's look at it
and the shell's redirection, which follows the link; and a model file that is modified in place
while the call reads it. A seventh thing is on neither of the header's lists, and no test of the programs tries it: a descriptor for writing
that another account opened while it could still write to the marker goes on working after the
marker's permissions are narrowed. Narrowing them afterwards does not suffice: `chmod` decides who
can open the file from then on, not what a descriptor that is open already can do. So an account
that opened the marker for writing before can go on changing what the marker declares, and
`cp_commit`, which looks at the owner, the permissions and the content, finds no fault in such a
marker (by this page's reading of the source). A marker alone
switches nothing on: for such a descriptor to cost anything, an operator must also call `cp_commit`
with `-public-weights` on weights that must stay hidden. So a marker is to be created closed to
other accounts, as the first command under "Declaring a model file's weights public" in
[operators-duties.md](../operators-duties.md#public-weights) does, and not closed afterwards. What to do
with a marker that its group or others could ever write, where it cannot be ruled out that another
account has it open, is told in the same passage. By the comment before `marker_owner_counts()` a
SHA-256 is no secret, and several accounts may be able to write to the directory; that is the
comment's reason why a marker counts only if it belongs to one of the three accounts and neither its
group nor others can write it. By [what-is-not-hidden.md](../what-is-not-hidden.md#public-weights)
the option and the markers guard against some mistakes, not against intent: a marker that was made
for weights that should stay hidden passes like any other. The duties that follow for an operator
are in [operators-duties.md](../operators-duties.md). The last line of the commitment's `isk.txt` is
then `seed public`; no key protects that file. `cp_prove` serves such a commitment only with that
line, the option, and the file from which the seed was derived, and it accepts the option for no
other commitment (`cp-isk-seed-public`). It does not read the marker files. The option switches off
one rejection: `cp_prove` then accepts the 32 zero bytes in a leaf table's `fresh.bin`. Every other build's `cp_prove` rejects a commitment
that has the line. `cp_commit` rejects the option together with `-keep-for-the-audit`: no bytes are
sampled for a commit run, so there is nothing to keep. A batch's 32 bytes are taken from the
system's random source under the option too, by the head of the test
`prover/tests/public_weights.py`, which expects two runs on one request to give the same next token
and two different proofs; what the paragraph above
says of the price holds all the same (`prover/common/public_weights.h`;
`prover/hash_commitment/public_weights_markers.h`; `prover/common/cp_files.h`;
`prover/weights_commitment/arguments.h`).

## 2a. From a key to a random value

f(K; a, b, c, d) is the whole block function: sixteen words of 32 bits. It is never run as a stream
with a counter that is stepped. Every call's four words are written out by the caller, so no counter
can wrap. Under each key every call has one use, which follows from the path to that key and from
the call's own four words: it gives a further key (words 0 to 7; the rest is dropped), or it gives
bits. Where a call gives values of 64 bits, value number i is word 2 i plus 2^32 times word 2 i + 1
(from the programs' source).

### Below a process key

The label of a random value is sixteen words, taken in four at a time.

    K1 = key(K_proc; counter_lo, counter_hi, 0, 0)     K2 = key(K1; 0, 0, version, object)     K_ctx = key(K2; member, 0, 0, 0)     block = f(K_ctx; call_lo, stream, retry, call_hi)

- The first twelve words name the **context**: one request's proof, or what a batch's requests
  share, or one commit run for a key file or a leaf file. The counter word is the request's number
  in the batch's list of requests, from 0: the order in which the items `request` of the list
  `a batch` are hashed (section 2). It is not the word that the format page calls the request's slot
  (R14.7): that one the prover samples from the batch's private randomness (below). By a search of
  the sources the number in the list goes into no byte of a file. The version is the format's
  version word. A **generator** is the part of a program that gives out the random values of one
  context. The object says whose random values these are: 2 a request's, 4 a batch's, 5 those of a
  key file's commit run, 10 those of a leaf file's commit run, all from the generator of
  `blocks/coins`; and 1 those of the **kernels' library** (`prover/common/coin_chacha.h`, with its
  state in `prover/hiding/trees.cuh`), which serves the sub-proofs' kernels (from the programs'
  source). Under one batch's key both generators work, with the objects 2 and 4 and the object 1.
  Neither sees the other's list of opened streams: the object word alone keeps them apart
  (`prover/hiding/long_rows_words.h` has the static_asserts). The batch's own context has the
  counter word 0 and the member word 0, and so have a key file's and a leaf file's commit run. So
  the first request's two contexts and the batch's differ in the object word alone. The kernels'
  library gives the sub-proofs' masks and nothing else (`prover/common/coin_chacha.h`, rule 1).
- The last four words name the **call**: its number on 64 bits, its stream, and its retry word. (In
  the generator of `blocks/coins` a context has at most 2^30 streams, so that the last has the
  number 16,384 + 2^30 - 1, and a stream is opened once. The kernels' library does not number its
  streams in the order of its calls but by a formula: the pads of the sub-proof with the identifier
  gid have the stream 4 (4096 + 64 gid) + 1, by the second implementation,
  `prover/second_writing/coin_streams.h`. This page gives no further rule for that library's
  streams.) A call gives four values of 64 bits, its **lanes**, from words 0 to 7 of its block;
  nobody reads words 8 to 15. Word i of a stream is lane i mod 4 of call i div 4 at retry 0.
- *Streams.* Every call of the generator of `blocks/coins` that samples, for elements or for bytes,
  opens a stream of its own (`again()`, which derives sampled values once more, opens none: section
  3). A context's streams are numbered from 16,384 in the order of these calls, which depends on
  public shapes alone, with one exception in the batch's context (object 4): an attempt of the
  slots' arrangement whose bytes are taken again is one more call (below, "What the batch's
  generator gives first"). This page does not give that order, and no page does: from this page
  alone nobody can derive a random value below a process key again. A replay of the order, written
  from a reading of the provers, is in `tools/audit/level1.py` and `level1_proof.py`. By the head of
  `level1.py` a fault that these scripts and the programs share is therefore not seen by the audit's
  comparison.
- Bytes (the salts of the leaves of a run's trees and of the binding tree, the digests of its free
  leaves, the words of the slots' arrangement, a chain's 32 random bytes) are these words as they
  come, lowest byte first: uniform on their set if the generator's output is.
- A field element is word i if it is below p. Otherwise the word is rejected, never reduced, and
  read again by a call in which only the retry word changes, at the retry word 8 k + lane, for k =
  1, 2, .., as lane 0 of that block. So an element is exactly uniform if the generator's output is.
  The programs of this repository can be built for field 2 alone (`blocks/base.h`; from the
  programs' source). In field 1 the rule is R6.3's: only the low 61 bits of a word are looked at,
  and the word is rejected if they equal p.
  What this page says of field 1 below a table's commit-run key is in `toy/mask_rules.py`; the plain
  twin and the second implementation named at the end of that part have field 2 alone (from the
  programs' source). An element of the extension field is its components, each made so.
- *Caps (from the programs' source).* The generator of `blocks/coins` ends the process if k reaches
  2^29. The attempts of the choice of the leaf table and of the slots' arrangement (both below) have
  no cap. By a reading of `prover/hiding/trees.cuh` the kernels' library has none either, and its
  retry word would wrap.

*What the batch's generator gives first.* Before a request is proved the batch's generator (object
4) gives: the salts of all leaves of the binding tree, in one call for bytes; then, leaf by leaf, 32
bytes, which stay the digest of a leaf that no request takes; then the arrangement of the slots. The
arrangement is a shuffle of 0 .. 2^depth - 1: for i from 2^depth down to 2, entry i - 1 changes
places with entry b, where b is below i. For b, 8 bytes (lowest first) are taken in a call of their
own and taken again unless their value is below the greatest multiple of i that is at most 2^64 - 1;
b is that value modulo i. So the arrangement is exactly uniform if the generator's output is.

A request gets the last entry that is still free when its first chain is sealed
(`prover/prover/batch_opening.h`; `blocks/iop/lin.h`, `LinBinding`). `cp_prove` seals the first
chains in increasing order of the requests' numbers in the batch's list of requests. A prover that
seals them in another order, in parallel for example, gives the requests other slots out of the same
arrangement. Every attempt for b is a call of its own and opens a stream. So after an attempt whose
bytes are taken again the later streams of the batch's context have numbers one higher, and these
numbers do not follow from public shapes alone (from the programs' source).

The generator of a key file's commit run (object 5) gives the first two of these kinds too, for the
key file's binding tree, which has the depth 0 and so one leaf (R13.6): after the one-time chain's
`SEAL` it gives that leaf's salt in one call for bytes, and then 32 bytes in a second call. The
one-time chain's seal takes the leaf, so these 32 bytes are not used. It makes no arrangement
(`blocks/frame/check_key.h`; from the programs' source).

*Under one process key a context MUST be made once.* A second generator under the same twelve words
would begin again at the first stream and give the same random values to other messages. For the
generator of `blocks/coins` no code enforces this during a run (`prover/hiding/engine_coins.cuh`). A
test asks it of runs of the whole programs, on synthetic weights under a test seed, from lists of
contexts that the programs then write (`prover/tests/coin_contexts_listed.py`). By its own head it
guards what it runs, at the shapes that it runs, and nothing enforces the rule in every run (from the programs' source). The kernels' library enforces its
own half: section 3.

- In the prover the source text is arranged to give it. A generator gets its process key as an
  argument. All generators of a batch are made in one block at the head of the batch: the batch's
  own, and one for every request of the batch, whose counter word and member word are the request's
  number in the batch's list of requests. There the key is derived, and the variable that held it is
  wiped (`prover/prover/isk.cuh`). After the head no routine of `cp_prove` is passed the key, and
  none may call a routine that makes a generator. So none is made under a neighbouring batch's key
  while batches overlap. The key's bytes live on in the kernels' library's state and in the second
  implementation of the sub-proofs' masks (`prover/prover/slot_pads_second.cuh`), which the batch
  needs until its own steps are over. While a batch is set aside, its key is in neither. Under
  `cp_prove`'s option `-keep-for-the-audit` the 32 bytes from which the batch's first key is derived
  outlive the process, in a file of the secret key directory, until the audit erases it (section 3;
  from the programs' source).
- This is a rule on the source text, and the compiler does not enforce it. By a comment of
  `prover/prover/isk.cuh` (in `begin_the_batch_on_a_leaf()`), that no other code makes a generator
  or reads the key is checked only on the sources' text,
  and not for the second implementation's copy of the key.
- The committer makes one run at a time and takes each run's key out of the kernels' library's
  state, through a header of its own (`prover/hash_commitment/hf_commit_coins.h`). There, making a
  context once is each caller's duty.

### The comparing word

f(K_proc; 0, 0, "CPKW", 0) gives bits, not a key: word 0 is the word by which two holders of a
process key compare it. Every other call under a process key is the call that gives K1, whose third
word is 0, so the two uses never meet. These bits serve nothing else. They are compared inside the
process and go into no file, message or printed line (`prover/common/coin_chacha.h`, rule 2).

### Below a batch's first key

There are two calls and no more.

- f(first key; "CPLF", t, 0, 0), for t = 0, 1, .., gives bits: the choice of the leaf table. Let M
  be the number of live leaf tables, that is, the number in the item `copies` (section 2) less the
  first live leaf table's number, and v the first 64 bits of the block, read as one number with the
  lowest byte first: v = word 0 + 2^32 x word 1 of the block's words of 32 bits. The live leaf
  tables are those from the first live one's number up (from the programs' source). The t that
  counts is the least one whose v is below the greatest multiple of M that is at most 2^64. So the
  choice is exact. The leaf table's number is the first live leaf table's number plus v mod M (from
  the programs' source).
- key(first key; "CPBK", 0, 0, 0) leads to the process key.

The chosen number has three later uses. It says whose name and answers are taken in after "CPBK",
and it is the number in the leaf table's child of the seed when the leaf table's commit-run key is
derived again. It is also in every proof of the batch.

### Below a table's commit-run key

Every call gives bits; there is no further key.

    blk(r, g, try) = f(K; r, g, try, "MASK")          eight values of 64 bits: the random entries 8 g .. 8 g + 7 of row r
    f(K; I mod 2^32, I div 2^32, 0, "SALT")           the first five values of 64 bits: the salt of leaf I
    pad(r, g, try) = f(K; r, g, try, "PADS")          eight values of 64 bits: the pads 8 g .. 8 g + 7 of row r

- The row is in one word and the block within the row in another (in the full-size example
  configuration, `prover/examples/target_full_f2`, 2^11 rows of 2^22.4 blocks). A block is **good**
  if all its eight values are below the greatest multiple of p that fits into 64 bits. That multiple
  is p in field 2 and 8 p in field 1. Where a row's number of random entries is no multiple of 8,
  the last block's values beyond it are dropped; they count all the same for whether the block is
  good. The good block of the least attempt is taken (try is the attempt's number), and its values
  are reduced modulo p. Every residue has equally many pre-images, so the entries are exactly
  uniform if the generator's output is. An attempt's number does not wrap: the run ends first.
- A stored table's salt is five values reduced modulo p, without a second attempt. It is not
  uniform, on purpose: a salt has to be hard to guess and no more. It loses 0.17 bit of min-entropy
  for each element at p = 2^61 - 1 and 1 bit for each element at p = 2^64 - 2^32 + 1: a salt keeps
  304 and 315 bits.
- The three families differ in the fourth word. Under K the block function must run on these three
  families of inputs and on no other (`prover/hash_commitment/hf_seed.h`).
- *The pads* (a stored table's; the pads of section 3's "Further guards" are an oracle message's and
  come from a process key). A table whose shape has s stored answers for each row (R4.10: the fifth
  word of S; the parameter line `answers`) has 3 s pads in each
  of its 2^v rows. The parser (`blocks/frame/parameters.h`) refuses every other number
  of answers than 1 (`parameter-range`): this project's proofs cover a table only where it has one answer; no attack on a table with more is known, and nobody has searched for one. Every line `answers` in a parameter file here
  says 1. The row's number r is the one by which the random entries and the record's answers
  are counted. The pad of answer i and coordinate t (i < s, t < 3; R13.8) is pad number j = 3 i + t
  of its row: value j mod 8 of the good block pad(r, j div 8, try). It is at entry M - 3 s + j of
  the row's message, where M is the number of entries of a message, pads counted in: the third
  number of the parameter line `table`. (R13.8 writes l for M. The programs' sources write i0 for
  it, and their letter l is a smaller number, the weights' cells in a row: the pads are after the
  cells.)
- For the pads a good block, the least attempt, the reduction modulo p and the end of the run where
  no attempt is good are those of the random entries, so every pad is exactly uniform if the
  generator's output is. The values of a good block whose number 8 g + sigma is 3 s or more are
  dropped; a block is judged by all its eight values, the dropped ones too.
- The weights' cells of a row end before its pads. A parameter file under which they would not is
  rejected: by the shared library (its message: "the matrix does not fit the table's rows beside
  their pads", `blocks/iop/lin.h`) and by `cp_commit`'s and `cp_prove`'s own guard
  (`policy-pad-place`, `prover/hash_commitment/hf_table_params.h`). In
  `prover/examples/target_small_f2/policy.txt` M is 524,288, and in
  `prover/examples/target_full_f2/policy.txt` it is 402,653,184.
- K is the commit-run key of that table alone: the anchor's, or leaf table k's. It must key that
  table's random entries, salts and pads and nothing else: that is a duty, in the
  words of `prover/second_writing/mask_generator.h`.
- A table has no form without a key. With a null pointer instead of the key such a table
  is not made at all, an opening of such a table ends the process, and the plain twin's routine for
  a pad ends the process too (`prover/hash_commitment/hf_commit_handed.h`, `hf_open_api.h`,
  `hf_mask_ref.h`). A second copy of the key that is missing or all zeros fails the self-check
  `own-test-pads` (section 4; `prover/hash_commitment/hf_keeper_unit.h`).
- The pads have four users, and each derives them for itself by this call, into a buffer of its own:
  the table as `cp_commit` builds it, the published answers, the folded row of an opening, and the
  columns that an opening recomputes (the head of `prover/hash_commitment/hf_pads.cuh`; section 4,
  `own-test-pads`). By that head one buffer for all four, if it were lost, would let table, folded
  row, columns and answers agree on zeros: every check would pass, and the answers would be
  published without their pads.
- The record's answers are computed once, by one routine, `hf_side_record()`
  (`prover/hash_commitment/hf_open_side.cuh`), below `make_the_record()` (`hf_commit.cu`). An answer
  has no second computation by value in the programs. By a comment of
  `prover/hash_commitment/hf_commit_keys.h` the compiler enforces only the type of the key that
  `make_the_record()` takes, and `hf_side_record()` takes whatever pads it is given. Two records
  over one set of pads would reveal exact equations on the weights (R4.12).
- *The test switch.* In a test build a switch, `ROWS_TABLE_FAULT`
  (`prover/hash_commitment/hf_commit_table.cuh`), zeroes pads on purpose. By the comments of
  `prover/common/made_up_alone.h` it is one of the switches that ask whether the weights are
  synthetic and the seed a test's, and a production build never reads it. That header only answers:
  it cannot make a switch ask, and of this switch it names random entries and salts, not the pads.
  No test reaches that question (section 3, "What no test of this repository has shown").
- The routines of `hf_mask.cuh` that make random entries or salts have no form without a key: they
  take the key itself, by reference or by value, or call `refuse_the_zero_key()` first, which ends
  the process on a null pointer as it does on 32 zero bytes. `mask_limit()` takes a null pointer and makes neither. In
  `hf_mask_columns()` (`hf_passes.cuh`) a null pointer means that the masks come through a fill.
- Source: `prover/hash_commitment/hf_mask.cuh` (the pads: `hf_pads.cuh`); a plain twin,
  `hf_mask_ref.h`; a second implementation, `prover/second_writing/mask_generator.h`; a fourth in
  Python with known answers, `toy/mask_rules.py`, which does not have the pads. The pads' known
  answers are in `prover/hash_commitment/hf_pads_known.h`, for field 2: by its head they come from a
  separate implementation in Python of this page's rule, and no
  program here prints them. By a comment of `mask_generator.h` its routine for the pads shares no
  code with `hf_pads.cuh`, but the two have the same authors; by the head of `hf_mask_ref.h` the
  plain twin is not an independent implementation either. By comments of `hf_mask_ref.h`,
  `mask_generator.h` and `toy/mask_rules.py`, what depends on the field in them was written
  together, from the rule's words: for field 2 the three are separate from `hf_mask.cuh`, not from
  one another. By the comment in `mask_generator.h` an error in that one reading of the rule's words
  would be in all three, and a comparison among them would not show it.

## 3. What hiding rests on, and what is only a guard

*The rule.* The same pads, random entries and salts MUST NOT be under two roots (R4.12). Why: a
row's random entries cover a fixed number of opened columns. If one set of them serves twice over
the same weights, the openings under both roots add up against that one number. If it serves over
different weights, anyone who sees a column of each at one position subtracts them. The random part
cancels. What is left is a symbol of the code word of the weights' difference, unmasked. Two records over the same pads reveal exact equations on the rows (R4.12; "Known limits",
below).

*What meets the rule.* In a production build, unless the call has the option `-public-weights`, the
seed is sampled in the call of `cp_commit` that makes the anchor, and in no other, and no second
anchor is made under it. The anchor's commit-run key has no secret input but the seed. A leaf
table's commit-run key has the seed and 32 bytes of its own, which are sampled from the system's
random source for that commit run (under `-public-weights` they are zeros). The seed is not the
anchor's alone: all five kinds of key are children of the one seed (section 2), so every other key
of the schedule descends from it too. With `-public-weights` no secret is sampled for a commit run,
so such a call does not meet the rule, and nothing is hidden: the seed is derived from the option's
file, a leaf table's 32 bytes are zeros, and the same call on the same files gives the same
commitment again (section 2, "A seed from a public file"). Two things belong to what meets the rule
and are not extras.

- Two entry points to a stored table's key.
  - "Make" takes what is fresh itself, derives the key itself, and is the only caller of the routine
    that makes a table's tree. It has no argument through which bytes or a key could be passed in,
    but two options reach it, each in its own build. In a production build `-public-weights` names
    the file from which the seed is derived, and sets a flag that leaves a leaf table's 32 bytes at
    zero; nothing is hidden then. In a test build `-test-seed` names the seed's file and sets the
    same flag. The flag (`repeatable`) is an argument in every build, of "make" for a leaf table and
    of the two routines that take the key file's and a leaf file's commit-run key
    (`prover/hash_commitment/hf_commit_keys.h`): neither a signature nor the compiler keeps a caller
    from setting it. That the two options alone set it rests on the three places in `cp_commit`'s
    main source that say it.
    By the source, with neither `-public-weights` nor the line `seed public` in `isk.txt` nothing in
    either program changes (`prover/common/public_weights.h`): in a production build neither the
    seed nor these bytes can then be passed in.
  - "Make again" derives a table's key from what was kept. For a leaf table `cp_prove` does it, in
    every batch, from the seed and the leaf table's kept bytes (`fresh.bin`;
    `prover/prover/isk.cuh`). For the anchor the committer has a routine that takes the seed from
    its file; the one call that leads to it, adding a leaf table to a commitment that exists, is
    rejected in every build (`cp-commit-first-form-alone`). In the call that makes a commitment, the
    key file's commit run and the links open the anchor under the key that "make" hands out. All of
    these reach only routines that compare what they make with the kept vertices of the table's tree
    and write nothing of it. A table that is made again is the same table, under the same root.
  - How far this is enforced: the compiler enforces one type, so that only "make" can construct what
    its own call of the tree's routine takes, and what `make_the_record()` takes, the
    routine that derives the pads for the answers and compares them. The routines below that call
    take a plain key, and `hf_side_record()`, which computes the answers, takes
    whatever pads it is given. Three more things are checked only by a test that reads the sources (not in this repository): that these routines
    have one caller each outside tests, that nothing else samples a leaf table's 32 bytes or the
    seed, and that the two options alone set the flag `repeatable` (above). It stays a reading of sources: it guards against a careless
    edit, not against a writer who means to go round it. By the comment before `Make` in
    `prover/hash_commitment/hf_commit_keys.h` a macro or a pointer to a routine still evades it.
    That comment calls the check weak: by patterns alone, without its digests of the code, it missed 21 of 21 deliberate evasions (from the programs' source).
- One lasting counter of opened columns for every stored table, the anchor too (the file
  `count.bin`; `prover/common/cp_files.h`). It is raised and made lasting before an opening begins.
  A second commit run for a key file or a leaf file has random values of its own and counts against
  the table's limit a second time (under `-public-weights` it has the same random values: the same
  call on the same files gives the same files again). No call of `cp_commit` makes a second one (by
  this page's reading of the head of `prover/hash_commitment/hf_commit.cu`).

*No random value twice inside a batch.* Under a batch's process key every request has one counter
word, and with it two contexts, of the objects 2 and 1. The counter word must rise from one request
to the next. The kernels' library ends the program at a counter word that is not above the last
(`prover/hiding/trees.cuh`, `set_attempt()`), and at a value that is sampled a second time under one
counter word (`enter_draw()`). That stop is on a second sampling, not on a second derivation or a
second use. By the comment at `enter_draw()` it sees only what callers enter there: values that are
derived from a context's key without such a call are not recorded, how far it covers the arrays that the programs sample is not known, and it is not proved that under one
process key no two random values come from the same input of the generator. Both generators have a
way, on purpose, by which values that a context has sampled are derived once more. In the kernels'
library it is `enter_draw_again()` (`prover/hiding/trees.cuh`); it ends the program unless the
context has sampled every value of the range.
In the generator of `blocks/coins` it is `again()`, by its comment for a caller that computes a
row's random entries or a salt again and does not keep them, and `not_the_coins()`, which compares
kept values with what `again()` gives; both open no stream, and end the process at values that were
never sampled (`blocks/coins/coins.h`). Neither looks at what the values that are derived again are
used for: that they serve the object for which they were sampled, and no other message, is each
caller's duty (by this page's reading of the source). The generator of `blocks/coins` has no stop
like `enter_draw()`: there every call that samples (`fill()`, `bytes()`) opens a new stream, and a
request's generator is made once, at the batch's head. By reading, `cp_prove` has no retry of a
request inside a batch. So no message is made a second time by a retry under the random values of a
message that has been sent; that no caller of a second derivation makes one is each caller's duty,
as said above. Every batch begins by taking its 32 bytes from the system's random source
(`prover/prover/isk.cuh`, `begin_the_batch_on_a_leaf()`), whatever went before. As far as names
show, no source under `prover/` outside test directories reads back the bytes that are kept for an
audit (section 2; the end of this section): by the source each file of `-keep-for-the-audit` is
written once, and the audit reads it (`prover/hash_commitment/hf_commit_keys.h`, at
`[keep-for-the-audit]`; `prover/common/cp_files.h`, at `AUDIT_BATCH_FILE`). Requests that are served
again after a failed batch are served in a new batch, all of whose random values are new; no statement of the written proofs on hiding covers a request that is sent again to the service, at once or later (`docs/limits-all-rows.tsv`, row PS-32).

*Further guards in the source (from the programs' source; of the GPU side nothing was read for this paragraph).* No verifier sees whether they are there. The generator of `blocks/coins` is held by one
thread at a time. By its comment two threads that sampled side by side could get one stream number,
that is, the same values for two objects, and no verifier would see it; so a call by a thread that
does not hold the generator, to sample, to derive values again or to ask for the last stream, ends
the process (`ledger()` checks nothing), and the generator is handed over, not taken
(`blocks/coins/coins.h`). Before a pad is placed, the library asks that the entries where it goes
are 0 (`own-test-pad-place`). After the oracle messages are committed, and before anything leaves
the process, it reads every pad back at a position that it computes again from the shape, and
compares it with the value that it kept when the pad was sampled and, where the generator numbers
its sampled values, with the value derived a second time (`own-test-pad-kept`). By the comment there
a pad that is lost or is one position off leaves an answer of the message unmasked in a proof that
verifies. The same comment lists what this guard does not see, among them a pad that is overwritten
after this comparison and a fault in the formula for the position, through which a pad is both
placed and looked up (`blocks/iop/send.h`). For a key file the committer's library compares the
index-mask message, but for its pads' positions, and its random entries with the values that were
sampled for them (`own-test-index-mask`; `blocks/frame/check_key.h`). Right after messages of the
test on the table are sent, in a key file's commit run and in a proof, the library compares the mask
rows of each such message, but for the pads' positions, and their random entries, as the prover
keeps them, with the sampled values, derived a second time by their numbers; a difference ends the
process (`own-test-lin-mask`; `blocks/iop/lin.h`). By the comment there a value that is lost between
its sampling and the sending gives a proof, or a key file, that verifies and hides less: under the
mask of the step that splits a row, an element is sent unmasked. The same comment lists what this
guard does not see: the mask rows and random entries of the folded row's message, which the test's
other sending, for the shared part, sends; random values that are not random; a fault that strikes
both derivations; a value that is overwritten after the comparison; and the copies that the engine
has encoded by then. Of a first fold's mask that is sent with the compared messages this guard
compares the random entries alone, and one that is sent with the folded row's message it leaves out.
By the same comment the fold compares them itself (`own-test-fold-mask`, below). Under a generator
that does not number its sampled values nothing is compared. Before a message with mask rows is
opened, the library makes the same comparison on it: on the random entries of its mask rows, and on
the rows themselves where the number of their sampling was kept (`own-test-mask-rows`;
`blocks/iop/context.h`). Its limits: it sees the tables that the prover keeps, not the copies that
the engine has encoded, from which the opened symbols come; under a generator that does not number
its sampled values it compares nothing; and it is not made on a message that another part of the
batch has sent under its own generator (`lin_second_prove_on_shape()`, `blocks/iop/lin.h`). A fold's
fresh mask has a guard of its own, made before the fold's sum is sent, when the root of the mask's
tree has gone out already, and again before its first round: the coefficients of the hiding
polynomial, each by the number of its sampling, and the random entries of the mask rows, as the
prover keeps them, are compared with the sampled values, derived a second time; a difference ends
the process (`own-test-fold-mask`; `blocks/iop/fold.h`). By the comment there a coefficient that is
lost before the mask's table is made gives a proof that every verifier accepts and in which an
element of the rounds' messages, or a public combination of them, goes out without its mask; a
random entry that is lost before the rows are encoded also gives a proof that verifies and hides
less. The same comment lists what this guard does not see: a fault in both computations of a sampled
value; random values that are not random; a sampling's number that is overwritten with the value for
"no number", whose coefficient is then not compared; and a loss in a copy, that is, in what the
engine has encoded, in the copy from which the sum is computed, or in what the rounds hold. By the
comment such a copy disagrees with the rest, so the steps after the fold are to reject the proof,
but an element in it may lack its mask: a program must verify a proof before it releases it. `cp_prove` does not: whoever hands a response on runs `cp_verify` on it first ([duty 6](../limits.md#the-conditions); [operators-duties.md](../operators-duties.md)). The
mask's table as the prover keeps it is compared with nothing, and the engine's part of each round's
polynomial is not checked. Under a generator that does not number its sampled values nothing is
compared, and nothing reports that: by the comment at `not_as_drawn()` (`blocks/iop/send.h`) a
program that relies on such a check must itself require a generator that numbers its sampled values.
Two more guards of this kind are not told here: one on the masking vectors of a proximity test's
last step, but for the main row's random entries (`own-test-base-mask`; `blocks/iop/base_case.h`)
and one on the mask rows of the key file's message on the ranges (`own-test-ranges-mask`;
`blocks/iop/wf.h`). They are told in `toys/1-product.md` and `toys/3-key-and-entry.md`. Under a
generator that does not number its sampled values the second compares nothing, and the first
compares the main row with a copy that was kept when it was sampled, and no more. A leaf table's
`fresh.bin` is read only if it is a regular file and not a symbolic link, has exactly 32 bytes,
belongs to the calling account, has one name and has no permission for group and others
(`prover/common/cp_files.h`). A test build's `cp_prove` has, beside `-test-coins` (section 2), two
options that it accepts only together with it: `-test-leaf`, which holds every batch on one leaf
table, and `-test-no-requests`, under which a batch's random values do not depend on its requests'
words; a production build has neither (`prover/prover/arguments.h`).

*A guard against a broken or repeating random source of the system, and nothing more.* A table's
commit-run key also reads c, S and which table it is. With equal bytes from the system, another
configuration, shape or table gives other random values all the same. Anyone who changes a public
file and carries a secret key directory over gets other random values, so that the opening stops at
its self-check. For the same reason every run's key reads, beside its fresh bytes, the data that the
run's messages are meant to depend on. It reads the weights through the commitment's public values:
the commitment ID and the names stand for the tables. It does not take in the weight files as this
run reads them: if the 32 bytes repeat for the same requests and the files differ between the two
runs, the same mask entries can cover different data (reasoned; whether one of `cp_prove`'s self-checks would catch that has not been examined: row H4.5.1). `cp_prove`'s options that change its
schedule or its use of memory are not among what a key reads: they are meant to leave every byte of
a proof unchanged (docs/limits-all-rows.tsv, row T15). How the program turns these data into
messages no key reads. Apart from the format's version word, which stands in the context below every
process key (section 2a), that is covered for a batch alone, by `the prover's revision` (section 2),
and only as far as that number was raised.

*What no test of this repository has shown.* While they run, `cp_commit` and `cp_prove` derive keys
of the schedule a second time, by the second implementation (section 4), before a value is sampled
under them, and `cp_commit` compares a table's random entries, a sample of its salts and its pads with that implementation too; a difference ends the run. By the comment before
`Make` in `prover/hash_commitment/hf_commit_keys.h` a table is made in one or more steps, each of
which makes all its random entries again, and only the first step's are compared. By
`docs/limits-all-rows.tsv`, row H3.5, no salt is compared while a table is made: the salts that are
compared are those of the columns that an opening writes and, by the row's reading of the code, of
all the columns under an opened column's lowest stored node of the hash tree. In `cp_prove` the keys
that are derived a second time are a batch's two keys, the first key and the process key
(`prover/prover/batch_keys_held.h`). That header has no comparison for the leaf table's commit-run
key, which `cp_prove` derives again in every batch ("make again", above): a wrong one shows only
where what is made again is compared with the kept vertices of the table's tree. By the header's own
head its comparison does not see the seed, the system's bytes or the facts, which the program passes
to both derivations; a fault that both texts share; or the key after the comparison. A `cp_prove`
that passed an empty list of requests to both derivations would pass every self-check, and every
proof would verify. No test of this
repository, and no run of a production build, has shown a whole `cp_commit` or `cp_prove` end at one
of these self-checks: not at a wrong key of the schedule, not at the comparison of random entries,
salts, pads and keys with the second implementation, not at `own-test-pads`, not at the key file's
key, and not at the known answers that both programs compute at their start. By
`docs/limits-all-rows.tsv`, row H3.3, the salts of the tree that binds a batch are not compared.

*Known limits: the operators' duty.* No key can help where a secret is doubled. An operator MUST NOT
let either of these happen:

- a secret key directory is copied, or restored from an older state. That gives the same table with
  a second or an older counter of opened columns. No program can see either. By the source
  `cp_prove` holds the directory by one handle from its start on, and neither that handle nor the
  program's walk along the path sees an older copy of the same directory that its own account or
  root has put under the name between two starts: the copy is taken, its counter is the older one,
  and random entries that were used since are used again without being counted (from the programs'
  source: `prover/common/cp_files.h`);
- `cp_commit` or `cp_prove` runs in two starts of one machine image or snapshot, or a running one is
  resumed. The system's random bytes can then repeat, and no program notices. If they repeat in
  `cp_commit`: with the same weights it makes one table on two machines, each with a counter of
  opened columns of its own. With other weights under the same three public files it makes two
  tables, under two roots, whose random entries and salts are equal, and their pads
  too. No table's commit-run key reads anything of the weights beyond the three public files. The
  harm needs no opening, and no counter stops it. The two tables have different roots, so
  different seeds (R4.8), over the same pads. The two commitments' public files alone then give, for
  each row and each answer, one exact equation over E on the two sets of weights, for good: what is
  published cannot be taken back. With `y = <eq(z), m> + beta pad` and
  `y' = <eq(z'), m'> + beta' pad`, where (z, beta) and (z', beta') are the two tables' seeds for
  that answer (R4.8; beta is never 0), pad is the element of E that the answer's three pads give,
  and m and m' are the plain entries of the two rows' messages (R13.8), the value
  `beta' y - beta y' = beta' <eq(z), m> - beta <eq(z'), m'>` holds no pad. Whoever knows one of the
  two sets of weights has an exact linear equation on the other alone.

With derived random values hiding is computational at best: it needs ChaCha20's block function to be
a pseudo-random function. It also needs a seed that stays secret, and that need does not end with a
run. A seed that was derived under `-public-weights` is known to anyone who has the option's file,
so nothing is hidden from them (section 2). Anyone who reads `seed.bin`, at any later time too, can
remove the random entries and salts from every column of the anchor that a key file or a leaf file
has shown, and can test a guess of the weights against the anchor's root. They can
also remove the pads from the anchor's published answers, each of which is then an exact linear
equation on the weights. With a leaf table's `fresh.bin` as well, the same holds for every column of
that leaf table that a leaf file or a proof has shown, and for that leaf table's
published answers. A past batch's 32 bytes are kept nowhere, with one exception (the next
paragraph). That its random values do not follow from `seed.bin` and `fresh.bin` all the same rests
on a further property of the programs' random generator, how it behaves with a secret input under a
known key, which no one has examined (`docs/limits-all-rows.tsv`, row H35.1).

*The exception: bytes that are kept for an audit.* Under `cp_prove`'s option `-keep-for-the-audit` a
call with `-in` and `-out` writes the 32 bytes of its one batch to `audit/batch.fresh` in the secret
key directory, for one audit (`tools/audit/audit_batch.sh`). Hiding is void for a batch whose random
bytes are kept, until the audit has erased them: together with `seed.bin` the file opens every mask,
pad and salt of that batch's proofs. `cp_prove` never erases the file. The audit, called with
`--erase-kept-files`, erases it at each of its three verdicts. It erases nothing at an ending that
comes before a verdict (a rejected call, an interrupt, output that cannot be written), and with
`--rehearsal` it erases nothing, whatever word it prints. Once it has come to a verdict and begun to
erase, an interrupt, or a last line that cannot be written, ends it with no verdict printed, and the
file can be overwritten or gone (`tools/audit/audit.py`; section 16 of the format page). By the head
of `tools/audit/audit.py` to erase or to destroy a kept file is to overwrite it with zeros, to sync
it and to unlink it: journals, copy-on-write, snapshots, flash remapping and backups may still hold
its bytes, and not every copy in the audit's own memory is wiped. The erasure itself can stop, or
can leave the file alone: the audit erases only a regular file of at most 32 bytes that belongs to
the caller and has one name, and only the file that it looked at when it began, not one that was
moved, replaced or changed while it ran. Its last line then says so, the audit does not say yes, and
hiding stays void for the batch. The erasure reaches the file's present blocks and its name, not
what a medium keeps elsewhere (`tools/audit/kept.py`; `tools/audit/README.md`, "What a YES does not
cover", point 8, which by that page holds for a batch's audit as well). `cp_prove` rejects the
option with `-spool` and on a real model's files, and rejects every start on a secret key directory
while anything is in `audit/`. That test sees only that the directory is empty, not that an audit
has said yes, and it is made once, at a start: a service that already runs on the directory goes on
beside the kept file (from the programs' source and documentation). So a batch's first key, and
with it every key and value below it, can be derived again, by anyone who reads the file and
`seed.bin`, as long as the file is there, and the file is one more secret that an operator MUST
guard like `seed.bin`. An operator MUST NOT hand out a response of such a batch unless the audit has
said yes. `cp_prove` prints that duty, and no program enforces it: the responses are in the
directory named with `-out` when `cp_prove` ends. `cp_commit`'s option of the same name does the
like for the 32 bytes of the key file's commit run and of each leaf file's (`audit/key.fresh`,
`audit/entry.<k>.fresh`): together with the seed they open every mask in the key file and the leaf
files. They stay after a good ending, until the audit destroys them before it says yes. Here the two
audits differ: a batch's kept file is erased at each of its audit's three verdicts, a commitment's
kept files only when every check has passed, directly before the yes. At every other ending of the
commitment's audit the kept files stay beside the seed, until a later audit says yes or someone
removes them by hand; where the destruction itself stops, those that it has not reached stay, one of
them perhaps overwritten (`tools/audit/audit.py`), and the commitment can no longer be audited and
counts as a no: nothing of it may be handed out ([operators-duties.md](../operators-duties.md)). A
kill, a signal that the program does not catch, a crash or a loss of power during the call leaves
them too, beside the seed's file, and no line is printed (`prover/hash_commitment/hf_commit_keys.h`,
at `[keep-for-the-audit]`). Both audits take synthetic weights alone: the audit rejects a secret key
directory whose `isk.txt` has not the line `weights generated` or `weights made-up-blocks`
(`tools/audit/audit.py`), and by the programs' documentation `cp_commit`, like `cp_prove`, rejects
`-keep-for-the-audit` on a real model's files ([operators-duties.md](../operators-duties.md)). So no
commitment to a real model's files, and no proof on them, can be audited in this way.

## 4. Where it is in the programs

All files of this table except those in `blocks/coins/` and `toy/mask_rules.py` are under `prover/`:
the GPU side and the programs that commit and serve. `blocks/coins/` holds the routines on which the
schedule is built: in `coins.h` a child of the seed and `process_key()`, which
`prover/common/coin_keys.h` calls for all five kinds of child, and in `run_coins.h` the list of
items. `coins.h` also has `making_key()`, with the same body; by a search of the sources other than
the GPU sources only a test calls it.

| what | where |
|---|---|
| the one text that the programs call: a routine for each list, a routine for each key | `prover/common/coin_keys.h`, on `blocks/coins/coins.h` and `run_coins.h` |
| a second implementation of the schedule, on a block function and a BLAKE2b in files of its own, which include no file of the programs. By `prover/second_writing/README.md` its `schedule_keys.h` follows the table in the head of `prover/common/coin_keys.h` and is not independent of the programs' own key schedule. Two outside sources are common to both implementations (`THIRD_PARTY.md`; `prover/second_writing/README.md`; `blocks/coins/chacha.h` names the first at the quarter round). ChaCha20's two block functions are different texts around the quarter round, but both go back to D. J. Bernstein's ChaCha code: the second implementation's as a whole, the programs' in the quarter round and the order of its eight calls. BLAKE2b's compression function is adapted in both from the BLAKE2 team's reference implementation, and so are its feeding and finishing: the programs' (`HasherT`, `blocks/hash/hash.h`) from that implementation's `blake2b_update` and `blake2b_final` as the second implementation's `blake2b_plain.h` has them. So the whole hash, not the compression function alone, is one text in two adaptations. For ChaCha20 the comparison of the two can show an error made in adapting that source, or in what the programs have of their own around it. The comparison does not see an error of either source, nor a slip in adapting a source, where both implementations have the slip; by `prover/second_writing/README.md` both sides took each source through the same intermediate texts of this project, and for the key schedule it says the same of a misreading of the rule | `prover/second_writing/schedule_keys.h`, `blake2b_plain.h`, `chacha_block.h` |
| a third, in Python, which prints the known answers | `prover/tests/coin_keys_values.py` |
| the known answers with which `cp_commit` and `cp_prove` compare, at their start, before a seed is read or sampled, what the routines give that lead from a seed to the keys and that choose a batch's leaf table (`own-test-coins`). By the header's comments they guard against a change, a compiler or a machine, on synthetic inputs; a misreading of the derivation that the Python script and the programs' routines share is not seen | `prover/common/cp_keys_own.h` |
| a test of these routines: the known answers, compared with the programs' routines and with the second implementation; and, for every item of every list but `the prover's revision`, an input in which that item alone is changed, at which the key must change and the two must agree (`the prover's revision` is no input: each implementation holds the number itself, and the test checks that the two C++ implementations hold the same number. A raise in one of the two alone makes that check and the comparison of the keys fail, and a raise in `prover/common/coin_keys.h` alone the checks against the known answers too; a raise in `prover/tests/coin_keys_values.py` alone makes its output differ from the stored `coin_keys_values.txt`; a raise in `tools/audit/coins.py` alone makes `tools/audit/coins_known.py` fail against that file) | `prover/tests/coin_keys_test.cpp` |
| the two entry points to a stored table's key | `prover/hash_commitment/hf_commit_keys.h`; "make again" of a leaf table: `prover/prover/isk.cuh` |
| the generator of a stored table's random entries and salts | `prover/hash_commitment/hf_mask.cuh`, `hf_mask_ref.h`, `prover/second_writing/mask_generator.h`, `toy/mask_rules.py` |
| the pads of a stored table, and their self-check `own-test-pads`. By the head of `hf_pads.cuh` each of the four users of the pads (section 2a) derives a copy of its own and compares it by value twice, once its buffers are filled and again after their last read, before it passes on anything that was computed from them; every copy in GPU memory is read back and compared with the host's buffer; and the rule's known answers are checked once in each process, before the first pad is derived. By the same head what the two kernels add to the table and to the columns is not compared, and no test skips a launch of either. What bounds that, by the same head: no kernel adds a pad to an answer, so a launch that never runs cannot publish an answer without its pad; if a launch never runs in all parts of a table, or in an opening, the recomputed columns do not hash to the stored nodes (`hc-own-node`); if it never runs in one part, that shows only when an opened column is in that part. The comparison is with the plain twin under the key's second copy. By the head of `prover/hash_commitment/hf_mask_own.cuh`, `cp_commit` derives that copy from the seed's file, read again: there the comparison also catches an erased seed in memory and a fault in one derivation, but no error in the function or in a leaf table's 32 fresh bytes, which both derivations share. By the same head `cp_prove` derives the key once and copies it twice (`Unit::leaf_begin()` in `prover/hash_commitment/hf_keeper.cu`): there the comparison catches an erased or overwritten copy, but no wrong derivation, which fails the comparison with stored tree nodes (`hc-own-node`). `cp_commit` also compares the pads with the second implementation; in every other program that comparison is empty and compares nothing. `own-test-pads` only derives each pad a second time: it does not see an error that all these implementations share, and it does not see the answers, which are computed once (section 2a). By a reading of the sources it is the only check that sees a loss of the pads which all their users share: no run on this code has shown it. No verifier can see whether the pads are there. Nor does the audit: by `tools/audit/README.md` ("What a YES does not cover", point 9) it recomputes no answer, so a committer that leaves the pads out gets its YES, and each published answer is then one exact linear equation on a row of the table. A CPU test at toy tables with 1, 2 and 4 stored answers compares the three implementations with one another and with the known answers, and has `own-test-pads` fail on: all pads 0; one pad 0; one pad cut to 8 or to 24 bits; one row's pads in all rows; one answer's pads for two answers; the first random entries instead of the pads; another key's pads; a missing second copy of the key. No row of `SUITE.md` runs that test | `prover/hash_commitment/hf_pads.cuh`, `hf_pads_known.h`, `hf_mask_ref.h`, `hf_table_format.h` (the name), `hf_commit_derivation_check.cuh`, `prover/common/derivation_check_seam.h`, `prover/second_writing/mask_generator.h` |
| the block function's uses, the four-letter words, the comparing word | `prover/common/coin_chacha.h` |
| ChaCha20's block function itself, in further files | `blocks/coins/chacha.h`, `prover/common/coin_chacha.h`, `prover/hash_commitment/hc_chacha.cuh`, `prover/hash_commitment/hf_mask_ref.h` (`block()`) |
| the choice of a batch's leaf table | `prover/common/cp_files.h`, `draw_leaf()` |
| the option `-public-weights`: the derivation of the seed and the word "CPPW"; the marker files; the line `seed public` | `prover/common/public_weights.h`; `prover/hash_commitment/public_weights_markers.h`; `prover/common/cp_files.h` |

*Where the programs ask the system for random bytes* (`getrandom`, no flags): the seed
(`prover/hash_commitment/hc_seed.h`); the 32 bytes of a leaf table's, the key file's and a leaf
file's commit run and of a batch (`cp_fresh32()`, `prover/common/coin_chacha.h`): a call that fails,
gives fewer bytes or gives only zeros ends the program; 8 bytes once in a process, which only move
the positions that a self-check samples, and from which no value of a proof is derived (if that call
fails the program goes on with fixed positions, by a comment there "a weaker check, no other harm";
`prover/engine/stream_coins.cuh`); the nonce of a request's file, a front end's
(`prover/prover/spool.h`): none of the three routines that draw it (that one, `driver/spool.py`, `integrations/spool.py`) checks the bytes; in `spool.h` a call that fails ends the routine, and bytes that are only zeros do not. `blocks/coins/coins.h` has a fifth call, which no `cp_` program reaches.
Under `-public-weights` `cp_commit` derives the seed from the option's file and samples no bytes for
a commit run (`prover/common/public_weights.h`; `prover/weights_commitment/arguments.h`).
