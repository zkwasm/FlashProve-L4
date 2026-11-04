*Part of [the file format](FORMAT.md). Rule numbers and terms are those of that page.*

## 8. The parameter file, and what is derived from it

The **parameter file** is the first of a configuration's three public files (section 1; R4.9). It is
a text of named numbers. This section has seven parts: the grammar (8.1); the meaning of the first
32 lines (8.2); what `cp_verify` asks beyond the grammar (8.3); how every number without a line
follows from the lines (8.4); the values at the example files (8.5); the model's lines (8.6); what
is easy to get wrong (8.7). It needs sections 2 and 4, section 6 for oracle messages, and R12.1 to R12.4 for codes
and stages. `cp_commit` can admit a parameter file of one's own making under which `cp_prove` can
serve no request and ends at its start (`docs/limits-all-rows.tsv`, row C5). So use the parameter
files of this repository, or prove and verify one request under a new file before a commitment made
under it is given out.

No numbered rule covers this ground: the numbered rules take the plan of R4.3, the codes and the
shapes as given. This section is written from the programs' source. It describes what the programs
compute, and it has no rule numbers of its own; cite it by subsection and item. The programs of this
repository can be built for field 2 alone (R1.2). What this section says of field 1 is arithmetic:
the programs' source has the formulas of 8.4 for both fields (`blocks/numbers/`), and those of field
1 are used by tests alone (`blocks/numbers/tests/`), never on a parameter file. For the n of tests
22 and 29 of 8.1 the parser has one bound, 32. The source gives its reason from field 2's p: the
proximity bound for a mask code that is repeated along the longer main domain needs that domain to
be one coset of a subgroup, so its length must divide p - 1, and beyond 2^32 a domain is a union of
cosets (a comment in `blocks/frame/parameters.h`). Test 36 has the same bound for the n of the
stored tables' code. The comment at that test says that the points of this code are one coset of a
subgroup, that a longer code would lie on several cosets, and that key files, leaf files and proofs
for a longer code are untested. This section gives no bound on these three numbers for field 1.

Why it matters: every byte of the file enters the configuration's digest (R4.9), and the numbers
that are derived from it enter every first query, as codes (R12.2), as fields of the shapes, and as
words of the plan's entries (R4.3). A verifier that derives one of them differently has other
challenges from the first round on and rejects every honest proof.

### 8.1 The grammar of the frame lines

The file has two parts. The **frame lines** are its first 32 lines, from `version` to `table`. One
strict parser reads them, the same for committer, prover and verifier. The **model's lines** follow
them (8.6); the parser of the frame lines passes them on unread.

- *A line* is a name of one or two words and its values, separated by single blanks (byte 0x20), and
  ended by one line feed (byte 0x0A). The last line ends with a line feed too.
- *A value* is a decimal numeral of 1 to 15 digits, without a sign and without a leading zero; the
  numeral `0` is admitted. The one exception is the value of `version`, which is four characters.
- *Nothing else is admitted:* no blank line, no comment, no carriage return, no tab, nothing before
  the first line.
- *The order is fixed:* the 32 lines come in the order of the table in 8.2, each exactly once.
- *One encoding:* two files that differ in one byte are two configurations, with two digests and two
  commitment IDs, even if they mean the same.

The parser takes the lines one after the other and stops at the first test that fails. The letters
in the tests are those of the lines in 8.2: B of `sec`, N of `slots`, and so on. Tests 1 to 4 are
made on every line, and tests 5 and 6 on every line but `version`, in this order, before the line's
own tests. The reason is printed as `<name>[: <which line>] (line <number>)`, where the number
counts lines from 1. After the colon comes the line's name in tests 1 to 4 and under
`parameter-range`, and the group's name alone, such as `link`, in tests 21 to 31; tests 5 to 8 and
33 to 42 have no colon and no name. A stage of `cp_verify` (13.6) prints `parameter file: ` and the
reason, and the call ends with `NO: <name> (policy)`; the blank and the parentheses are part of that
name. The line with the reason is part of the stage's output, which a call that ends with `NO`
prints only under the option `-show-stages` (14.2; 14.7). A whole call and the part-calls `-key` and
`-leaf` test the first line earlier, before the stages (13.8). Unless the file begins with
`version `, four bytes and a line feed, such a call ends with `NO: version-unknown (policy)`; a word
that is not the build's gets its name by test 8; and no line `parameter file: ` is printed. So in
these calls a fault of the first line never has the name of one of tests 1 to 4.

| # | test | rejection |
|---|---|---|
| 1 | The line ends with a line feed: a byte 0x0A follows. | `parameter-line-missing` |
| 2 | The line begins with the expected name, of one or two words such as `work field` or `group once-folded`, and one blank. This fixes the order and the encoding of the names. | `parameter-line-order` |
| 3 | No value is empty: no two blanks in a row, no blank at the end of the line. | `parameter-line-blanks` |
| 4 | The line has the expected number of values; a line `stages` has one or more. | `parameter-line-values` |
| 5 | A value has 1 to 15 characters, and no leading zero unless it is the single digit `0`. | `parameter-number-spelling` |
| 6 | Every character of a value is a decimal digit. | `parameter-number-spelling` |
| 7 | The value of `version` has exactly four bytes. | `version-unknown` |
| 8 | The word is the one that the build serves; a program is built for one field and one hash. If not, the name is `version-unknown`. | `version-unknown` |
| 9 | `sec`: 1 <= B <= 256. | `parameter-range` |
| 10 | `reading`: 1: the theorem's integer is read as the theorem is printed. | `parameter-range` |
| 11 | `rowcost`: 1. | `parameter-range` |
| 12 | `slots`: 1 <= N <= 2^20. | `parameter-range` |
| 13 | `slots`: N is a power of two. | `parameter-range` |
| 14 | `answers`: 1: this project's proofs cover a table only where it has one answer; no attack on a table with more is known, and nobody has searched for one. | `parameter-range` |
| 15 | `record`: M is 1. | `parameter-range` |
| 16 | `work ..`: each of the three values is at most 58. | `parameter-range` |
| 17 | `work ..`: each of the three values is below B of `sec`. | `parameter-range` |
| 18 | `salt`: 40 <= Y <= 64. | `parameter-range` |
| 19 | `salt`: Y is a multiple of 8. | `parameter-range` |
| 20 | `listed`: 0. | `parameter-range` |
| 21 | `group`: J is 1. | `parameter-group-shape` |
| 22 | `group`: n is at most 32. | `parameter-group-shape` |
| 23 | `group`: l < n. | `parameter-group-shape` |
| 24 | `group`: nz <= n. | `parameter-group-shape` |
| 25 | `group`: lz < nz. | `parameter-group-shape` |
| 26 | `stages`: the number of values is a multiple of four and at most 32, so a test has one to eight later stages. | `parameter-stages-count` |
| 27 | `stages`: k <= 8. | `parameter-stages-shape` |
| 28 | `stages`: h <= k. | `parameter-stages-shape` |
| 29 | `stages`: n is at most 32. | `parameter-stages-shape` |
| 30 | `stages`: nz <= n. | `parameter-stages-shape` |
| 31 | `stages`: the last stage has k = 0, and so h = 0; every other stage has k >= 1. | `parameter-stages-shape` |
| 32 | `ranges-part`: 1 <= L <= 40. | `parameter-range` |
| 33 | `own` has the n and l of `omega`, and `once-own` those of `once-omega`. | `parameter-own-code` |
| 34 | `folded` has the nz and lz of `omega`, and `once-folded` those of `once-omega`. | `parameter-one-mask-code` |
| 35 | `table`: b <= 20. | `parameter-table-shape` |
| 36 | `table`: n <= 32. | `parameter-table-shape` |
| 37 | `table`: i is not 0. | `parameter-table-shape` |
| 38 | `table`: i + m < 2^n. | `parameter-table-shape` |
| 39 | `table`: e is 0. | `parameter-table-shape` |
| 41 | `table`: w <= 40. | `parameter-table-shape` |
| 42 | `table`: w >= b. | `parameter-table-shape` |

Tests 21 to 31 are made on each of the eight pairs `group` and `stages`, pair by pair: 21 to 25 on
the line `group`, then 26 on the line `stages`, then 27 to 31 stage by stage. Tests 33 and 34 are
made after the line `ranges-part` has been read, so their reason names line 31. These tests look at
shapes only. Whether a file's numbers reach its own target is decided in 8.4.

### 8.2 The frame lines

The last column has the values of the file `record.f2.txt` in `blocks/frame/examples/`, in field 2.
A group is a set of rows that is tested at one list of positions; a stage is one step
of a proximity test; both are defined in 8.4, item 5.

| # | line | meaning | admitted by the parser | `record.f2.txt` |
|---|---|---|---|---|
| 1 | `version W` | The version word: H, the format number 8, the field digit, the hash digit (R4.3). | the one word that the build serves | `H821` |
| 2 | `sec B` | The **target** B: every round's error, times 2^-w for w proof-of-work bits, is at most 2^-B. | 1 to 256 | `128` |
| 3 | `reading R` | The formula that ties the integer m of a distance to the slack (8.4, item 6): eta = sqrt(rho) / m. | 1 | `1` |
| 4 | `rowcost C` | What combining the rows of an interleaved word costs in a closeness error: nothing. | 1 | `1` |
| 5 | `slots N` | The largest number of requests in a batch. The binding tree (R7.18) has this many leaves, one for each slot. | 1 to 2^20, a power of two | `64` |
| 6 | `answers S` | The number of answers to each out-of-domain sample. | 1 | `1` |
| 7 | `record M` | A constant of the format: every stored table has a record (R4.8). | 1 | `1` |
| 8 | `work field A B C` | The number of proof-of-work bits before a verifier's message in a round whose error is of the kind "a non-zero polynomial vanishes at the challenge". | each at most 58 and below B of `sec` | `0 0 0` |
| 9 | `work sample A B C` | The number of proof-of-work bits before a verifier's message in a round whose error is of the kind "two members of a list have the same answers to an out-of-domain sample". | each at most 58 and below B of `sec` | `0 0 0` |
| 10 | `work closeness A B C` | The number of proof-of-work bits before a verifier's message in a round whose error is of the kind "a combination of words is close to the code although the words are not jointly close". | each at most 58 and below B of `sec` | `0 0 0` |
| 11 | `work positions A B C` | The number of proof-of-work bits before a verifier's message in a round whose error is of the kind "a word that is far from what is claimed agrees with it at all sampled positions". | each at most 58 and below B of `sec` | `20 28 32` |
| 12 | `work smallness A B C` | The number of proof-of-work bits before a verifier's message in a round whose error is of the kind "a vector that is not small has small images under all sampled combinations". | each at most 58 and below B of `sec` | `0 0 0` |
| 13 | `salt Y` | The number of bytes of a leaf's salt (R7.3), in every tree alike. This one line gives the third word of every plan (R4.3), the sixth word of the tables' shape (R4.10) and the second word of the item `binding tree` (R14.6). | 40 to 64, a multiple of 8 (so 40, 48, 56 or 64) | `40` |
| 14 | `listed P` | A file lists no sampled position (R8.2). | 0 | `0` |
| 15 | `group link J n l nz lz` | The group of a segment of a request's first chain, and the message into which the last segment is closed. In the names `group link` and `stages link` the word `link` means a request's segment. | tests 21 to 25 | `1 23 22 17 13` |
| 16 | `stages link k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 22 17 3 3 19 16 3 3 16 15 4 4 13 13 0 0 15 15` |
| 17 | `group folded J n l nz lz` | The group of the folded table, encoded again, in a batch's shared chain. | tests 21 to 25 | `1 24 22 13 6` |
| 18 | `stages folded k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `5 2 21 12 4 2 17 12 4 2 13 12 0 0 14 14` |
| 19 | `group omega J n l nz lz` | The group of Omega, that is, the table times the equality weights of the shared point, in a batch's shared chain. | tests 21 to 25 | `1 22 18 13 6` |
| 20 | `stages omega k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 18 14 4 3 14 13 0 0 15 15` |
| 21 | `group own J n l nz lz` | The group of Omega again, with a request's own vector W, in the request's second chain. | tests 21 to 25 | `1 22 18 13 8` |
| 22 | `stages own k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 18 15 3 3 15 14 3 3 13 13 0 0 15 15` |
| 23 | `group once-folded J n l nz lz` | As `folded`, in a key file; also the two folded rows of a leaf file's link. | tests 21 to 25 | `1 24 22 13 6` |
| 24 | `stages once-folded k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `5 2 21 12 4 2 17 12 4 2 13 12 0 0 14 14` |
| 25 | `group once-omega J n l nz lz` | As `omega`, in a key file. | tests 21 to 25 | `1 22 18 13 6` |
| 26 | `stages once-omega k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 18 14 4 3 14 13 0 0 15 15` |
| 27 | `group once-own J n l nz lz` | As `own`, in a key file. | tests 21 to 25 | `1 22 18 13 8` |
| 28 | `stages once-own k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 18 15 3 3 15 14 3 3 13 13 0 0 15 15` |
| 29 | `group once-ranges J n l nz lz` | The group of the multiplicities of the lookups in the key file's proof of ranges. | tests 21 to 25 | `1 22 18 16 12` |
| 30 | `stages once-ranges k h n nz ..` | The later stages of the proximity test on that group, four numbers for each stage. | tests 26 to 31 | `4 3 19 17 3 3 17 17 0 0 20 20` |
| 31 | `ranges-part L` | A part of a lookup in the key file's proof of ranges reads 2^L rows of the committed matrix. | 1 to 40 | `18` |
| 32 | `table b n i m e w` | The stored table; see the second table. | tests 35 to 42 | `11 32 402653184 44040191 0 18` |

In each line `work ..`, A is for a request's own rounds, B for a batch's shared rounds and C for the
committer's rounds (8.4, item 3). In each line `group`, J is a constant of the format; the
group's base code has length 2^n and messages of 2^l entries; its mask code has length 2^nz and
messages of 2^lz entries.

The six numbers of the line `table`:

| number | meaning | admitted by the parser | `record.f2.txt` |
|---|---|---|---|
| b | The table has 2^b rows. | at most 20 | `11` |
| n | Its code has length 2^n. | at most 32 | `32` |
| i | The number of entries of a row's message. The last 3 * S of them are the record's pads (R13.8), S the value of `answers`. | at least 1; i + m below 2^n | `402653184` |
| m | The number of random entries of a row. This is the only number of random entries that has a line. | i + m below 2^n | `44040191` |
| e | A constant of the format. Both distances of the stored table are derived (8.4, item 16). | 0 | `0` |
| w | The number of coordinates of the point that a batch shares; the b that fold the rows are among them. | b to 40 | `18` |

The four numbers of a stage in a line `stages` are:

- k: the folded row of the stage before is cut into 2^k main rows.
- h: the stage has 2^h mask rows.
- n: its base code has length 2^n.
- nz: its mask code has length 2^nz.

The message lengths of a later stage and all numbers of random entries have no line (8.4, item 14).

In every group, and in all later stages of the test on it, a distance goes up to the Johnson radius
less a derived slack (8.4, item 6). The count of the code from whose m the distance is made, and
every closeness count beyond a third of the minimum distance, rest on the theorem on codes (Theorem
4.6 of the full version, ePrint 2025/2055, of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and
Saraf published at STOC 2026), which the programs' accounting takes as given; a statement on codes held to follow from it is
proved in the Lean library (13.7, condition 4; 8.4, items 6 and 7).

A parameter file with other lines
`group once-..` and `stages once-..` than those of `record.f2.txt` can be accepted too, if it meets
the conditions of 13.7.

### 8.3 What `cp_verify` asks beyond the parser

The parser's ranges (8.1) are the same for all programs. The program `cp_verify` is narrower in four
respects (rows 2 to 5 of the table), ties three lines to the model's lines, and compares the target
with the caller's own demand. What row 1 asks follows from tests 36 and 38 of 8.1. A rejection
ends the call with `NO: ` and the name. The last two rows reject nothing. The first of them lowers the call's result (13.7; 14.3). In a call without the option `-no-proof-of-work` the second changes neither the call's last line nor its exit code: proof of work counts toward the target. With
that option it lowers the call's result too (14.4, cause 4).

| what is asked | if not |
|---|---|
| i and m of `table` are below 2^32 | `parameter-table-shape (policy)` |
| each of the fifteen values of the lines `work ..` is at most 32 | `policy-work-most` |
| `slots` is at most 256 | `policy-leafd` |
| w of `table` is 1 to 20 | `w-shape-lk` |
| the committed matrix, as 11.5 places it from the model's lines, has at most 2^22 rows (8.4, item 23, asks only ceil_log2 of that number to be at most 30) | `w-shape-mW` |
| 2^w is at least the vocabulary's size: ceil_log2(V) <= w, V the model's line `V` | `hid-vocab-cols` |
| `sec` is at most the model's line `tau` | `policy-tau-hb` |
| the model's line `tau` equals T of 8.4, item 21, in the request scope: B less A of `work smallness`. With the row above, that leaves `tau` = `sec` and A of `work smallness` = 0. The line `tau` is 32, 64, 96, 128, 160, 192 or 256 (`hid-tau`), so `sec` is one of these too | `plan-tau` |
| all that 8.4 derives exists, and every condition of 8.4, items 22 and 23, is met | `parameters-not-admitted`; a line `parameters-not-admitted: ..` in the stage's output (8.1) has a sentence that names the group or the round |
| for the plan of each file that a stage of the verifier (13.6) reads: every round meets `sec`, its proof of work counted (8.4, item 22), and `sec` is at least the requested bits (option `-bits`, 128 without it) | `security-bits` |
| `group folded` and `group omega` differ in n or in l, and so do `group once-folded` and `group once-omega` (8.4, item 18, last paragraph) | no rejection: the condition `two-codes` (13.7, condition 13) is missed, and the call ends at best with `NOT-A-VERIFICATION`. The reason in the programs' source: the test on the table then runs without its split step; the bounds that the programs' source calls argued on paper for this protocol do not cover that form; no attack on it is known, and nobody has searched for one (`verifier/src/vf_common.h`) |
| no round of a file's plan needs its proof of work to meet `sec` (8.4, item 22) | no rejection, and in a call without the option `-no-proof-of-work` no other result: the call can end with `YES` and exit code 0. With that option a call in which a round needs its proof of work ends at best with `NOT-A-VERIFICATION` (14.4, cause 4). A line near the call's end (14.3) tells it: `proof of work: none counted`, or `proof of work: up to <n> bits counted in <m> rounds; with it the least round has <a> bits, without it <b> bits` (`round` where m is 1), with n the most bits that a line `work ..` declares, m the number of rounds that need their proof of work, and a and b the bits of the round with the largest error when proof of work is counted and of the one when it is left out; in a whole call m, a and b are those of the three files' plans taken together, in a part-call those of that file's plan. a and b are each one round's error in bits, by `cp_verify`'s own count, and neither 2^-a nor 2^-b is the chance that a false proof is accepted (14.3). So a `YES` of a call without the option does not say that every round meets `sec` without its proof of work: under `work positions 20 28 32` up to 32 of a round's 128 bits are proof of work |

The requested bits are in no file. None of these tests reads a byte of a key file, a leaf file or a
proof file. Every rejection of the table but `security-bits` comes before a stage opens one of these
files; `security-bits` comes after the stage has compared the commitment ID (13.4 to 13.6). The call
itself compares the heads of the key file and of the proof file with the version word before any
stage parses lines 2 to 32 (13.8); so where a head and one of these lines are both faulty, the head
gives the name. `cp_verify` also copies five numbers of the frame lines into the item `request`: B
of `sec`, and w, b, n, m of `table`, in this order (R15.5, words 172 to 176).

Where a whole call ends with `YES`, closeness
errors of the key file and the leaf file, and with them the binding of the weights, rest on the
theorem on codes (8.4, item 6; 13.7, condition 4, says on what such a `YES` rests). By a comment in `verifier/src/vf_common.h` these
two files also lean on a closeness error with no factor for the rows (8.4, item 9) and on one more list size for the repeated
mask code (8.4, item 6); by items 9 and 6 both rest on arguments of this project's own.

### 8.4 What is derived

No line gives a number of positions, a number of random entries (except m of `table`), the message
length of a later stage, or a distance. The programs derive them from the
frame lines and from a few numbers of the shapes of the oracle messages. One routine does this for
committer, prover and verifier. Why these numbers have no line (a comment in
`blocks/frame/parameters.h`): so that no file can ask for a list of positions that is too short for
its own target, proof of work counted; by the same comment the line `work positions` shortens the
lists.

The programs compute in integer interval arithmetic with directed rounding (item 1). They iterate to
a fixed point (items 13 and 14). Why: a row needs enough random entries to hide the positions at
which it is opened; the random entries raise the code's rate; and the rate fixes the number of
positions. Three groups are solved jointly, because they share oracle messages (item 18).

A second verifier has two ways. The first: it rebuilds this subsection bit for bit. The numbers, the
codes and the words of the plans can be rebuilt from it; whether a plan holds (item 22) cannot,
because not every round's error is given. The ten conditions on hiding are named in item 23 and
defined in 16.1. The second: it takes the plan, the codes and the shapes as public inputs in their
canonical bytes (R4.3; section 18). Then it checks files against those inputs, and that they follow
from the parameter file is a separate matter. On the second way the parser and steps 3 to 6 and 8
of item 23 are not run, so on that way nothing asks what they ask of the plan, the codes and the
shapes. The same holds of the three rows of 8.3 that tie a frame line to a model's line
(`hid-vocab-cols`, `policy-tau-hb`, `plan-tau`; section 18, the paragraph after its last list). Each
such condition is then a condition on the inputs: the verifier tests it itself, or the party that
derives the inputs must have tested it and every statement on such a verifier has it as a
hypothesis. These pages do not repeat this at each condition: they say it once for all of them
(here; section 18). Step 6 asks all ten conditions on hiding, but only parts of five of them concern
a verifier; the others are the committer's and the prover's alone and are a hypothesis of no such
statement (16.1). On the first way they hold by the derivation or are asked a second time, by the
parser or in item 23 (16.1 says where for each); part (d) of condition 3 alone is asked nowhere but
among the ten, so every verifier asks it as 16.1 has it, and the fourth line of condition 6 too
every verifier asks, on either way (the start of section 13, (c)). The parts are the fourth line of
condition 6 and condition 10, which carry soundness; condition 4; parts (b) to (e) of condition 3
(part (a) every verifier asks in any case: 11.1); the first three lines of condition 6; and part (c)
of condition 9. They are the lesser share: the routines of the programs' source that ask them ask
more, and some of that carries soundness too. Section 18 lists both, each with its place on these
pages and in the programs' source. It also lists, routine by routine, the conditions that these
pages have there alone; that plan, codes and shapes fit each other; that the errors were computed
for these inputs; and what ties the inputs of one chain or file to another's.

Notation: ceil_log2(v) is the least a with 2^a >= v, so it is 0 for v = 0 and v = 1. up(v) =
2^ceil_log2(v), so up(0) = 1 (R12.16 writes up(0) = 0). "div" is integer division, rounded down. B,
N and S are the values of `sec`, `slots` and `answers`; S is the s of R13.8. In items 16 to 23, w_T and m_T are w and m of the line `table`. Where an item says
"reject" or "rejected", the public files are rejected: `cp_verify` ends with
`NO: parameters-not-admitted`, and a line `parameters-not-admitted: ..` in the stage's output (8.1)
has a sentence that names the group, the stage or the round.

1. *The arithmetic.* Integers are exact and of any size. A **Real** is u * 2^e with an integer u >=
   0 and an integer e. Every operation that cannot be exact takes a direction, down or up, and its
   result lies on that side of the true value. floor and ceil give integers. Comparisons of Reals
   are exact. The operations:
   - cut(x, dir): if u has more than 192 bits, let s be the excess; drop the lowest s bits of u and
     add s to e; if dir is up and a dropped bit was 1, add 1 to u; the result is not cut again, so u
     can have 193 bits.
   - add(a, b, dir): if a is 0 the result is b, if b is 0 it is a; else bring both to the smaller
     exponent, add exactly, cut.
   - sub(a, b, dir), for a >= b: the same with a subtraction; if b is 0 the result is a.
   - mul(a, b, dir): the exact product, cut.
   - div(a, b, dir): 0 if a is 0; else with s = 192 + 2 + (the number of bits of b's u), q = (a's
     u * 2^s) div (b's u), plus 1 if dir is up and the remainder is not 0; the exponent is e_a - s -
     e_b; cut.
   - sqrt(a, dir): 0 if a is 0; else s = 2 * 192 + 4, plus 1 if e_a - s is odd; r is the integer
     square root, rounded down, of a's u * 2^s, plus 1 if dir is up and r^2 is not that number; the
     exponent is (e_a - s) / 2; cut. The programs' routine takes other steps and gives the same
     value (`Real::sqrt` in `blocks/numbers/real.h`): with 2^w the worth of the root's 192nd bit, it
     takes the integer square root, rounded down, of floor(a / 4^w), plus 1 if dir is up and a / 4^w
     is not the square of an integer; the exponent is w. Both are the root of a, rounded in the
     direction dir to 192 bits: to round twice in one direction is to round once.
   - pow(a, n, dir): start with r = 1 and go through the bits of n from the lowest: if the bit is 1,
     r = mul(r, a, dir); if higher bits remain, a = mul(a, a, dir).

   A **Bound** is a pair lo <= x <= hi of Reals with a flag "bad". From an integer: lo = cut(down),
   hi = cut(up). The operations:
   - a + b = (add(a.lo, b.lo, down), add(a.hi, b.hi, up)).
   - a - b is bad unless b.hi < a.lo; else (sub(a.lo, b.hi, down), sub(a.hi, b.lo, up)).
   - a * b = (mul(a.lo, b.lo, down), mul(a.hi, b.hi, up)).
   - a / b is bad if b.lo is 0; else (div(a.lo, b.hi, down), div(a.hi, b.lo, up)).
   - sqrt and pow act on each end in its direction.
   - ratio(x, y) of two integers is Bound(x) / Bound(y).

   A result inherits "bad" from its operands. Only non-negative values occur. Every decision is
   taken at the safe end: an error is compared at its hi, and a bad error counts as 1. Where a plan
   is made (item 21), a round whose error is bad rejects the plan, with the words "a formula was
   asked outside its range". The rejection is recorded and the routine goes on; of several reasons
   the first is kept, and a rejected plan does not hold, whatever its rounds are like
   (`blocks/numbers/plan.h`). "Surely a < b" means: neither is bad and a.hi < b.lo.

   Every formula below is evaluated in the order in which it is written, parentheses first, and each
   operation rounds. A power is taken before a product or a quotient, and these before a sum or a
   difference; operations of one rank go from left to right. So a + b * c is a + (b * c), 3 * x^6 is
   3 * (x^6), and 1 / x^2 is 1 / (x^2). A formula that is regrouped can give another last bit, and
   so, near a threshold, another integer.
2. *Constants that no line sets.* The field and the hash are those of the version word: field 2 is p
   = 2^64 - 2^32 + 1 with plain Reed-Solomon codes, field 1 is p = 2^61 - 1 with circle codes (R1.2;
   for them the programs' source names Haböck, Levit and Papini, "Circle STARKs", ePrint 2024/278);
   E has p^3 elements (R1.4), written |E|. An element of F is 8 bytes, an element of E 24, a digest
   32, a nonce in a file 8. A sumcheck's message leaves out the coefficient that the check fixes
   (the seventh word of a plan, R4.3). An element of F is sampled in at most 6 attempts and an
   element of E that is not 0 in at most 2 (R6.4, R6.5).
3. *Error kinds, scopes and proof-of-work bits.* Each round has an **error kind**, which follows
   from its name, and each chain has a **scope**. The round's proof-of-work bits (the second word of
   its entry, R4.3) are the value of the line `work <kind>` in the column of the chain's scope. The
   **request scope**, column A: a proof's first and second chains. The **batch scope**, column B: a
   proof's shared chain. The **committer scope**, column C: all chains of a key file and of a leaf
   file.

   | line | rounds |
   |---|---|
   | `work field` | `BAT`, `BAT-sigma`, `PSC epsilon`, `PSC round`, `PSC Y`, `ZERO`, `LINE`, `LKP shift`, `JOIN`, `LKP`, `FOLD epsilon`, `LIN zeta`, `LINK phi` |
   | `work sample` | `SEND seed` |
   | `work closeness` | `FOLD round`, `BASE lambda`, `LINK point`, `LIN point` |
   | `work positions` | `LOC`, `CS`, `BASE`, `LINK` |
   | `work smallness` | `RNG` |
   | none: 0 bits | `SEAL`, `SEND answer`, `SPLIT`, `PSC values`, `LAST`, `REPLY`, and the entry `RECORD` |

   A plan is
   rejected if a round's bits are not below the target B of `sec`, or are above 58. A round with
   proof-of-work bits has a nonce of 8 bytes in its file (R8.2).
4. *A code as the formulas read it.* A code is (a, ell, r): length len = 2^a, ell message entries, r
   random entries, the **tail** (R12.1). Field 2: its dimension is ell + r, and the formulas use
   kappa = ell + r - 1, the rate rho = ratio(kappa, len), and the minimum distance d = len - kappa.
   Field 1: the dimension is ell + r if r = 0, else ell rounded up to a multiple of up(r), plus r;
   and kappa is the dimension itself. A tail of r entries covers r positions in field 2; in field 1
   an even r > 0 covers r - 1, and every other r covers r. tail(t), the least tail that covers t
   positions, is t in field 2; in field 1 it is t + 1 for an even t > 0, and t else. Three numbers
   from outside are a code only if 1 <= a <= 60, ell >= 1, ell <= len, r <= len, the dimension is at
   most len, and in field 2 the dimension is at most 2^min(a, 32). The points and symbols of a code
   are R12.11. A code of length 2^a' **repeated** to the length 2^a, a >= a', is read as the pair
   len = 2^a, kappa = kappa_own * 2^(a - a'); for a = a' it is the code itself.
5. *Groups and stages.* A **group** is 2^k main rows under one **base code** (R12.2) and 2^h mask
   rows under one **mask code**. They are the rows of the oracle messages that section 9 puts into
   one group. All are read at one list of positions, a mask row at the position modulo its length
   (R7.2), and they have one joint distance. Rows that nobody sends count as zero rows. Every group
   of a line `group` and every later stage has a mask code, also where h = 0: then it has one mask
   row. A group can have a **side count** c > 1: then every main row stands for c rows that are
   folded side by side. A **stage** is a group inside a proximity test; stage 1 is the group that is
   tested, and each later stage contains the folded row of the stage before, cut into new rows
   (R12.4, R13.19). The mask code **governs** the distance if kappa_mask * len_base > kappa_base *
   len_mask, that is, if it has the larger rate; else the base code does. A group is faulty if its
   mask code is longer than its base code, if h > k, or if k > 0, h = 0 and the mask code governs.
6. *The rule for a distance.* A distance delta is kept as the **agreement** alpha = 1 -
   delta. len, kappa and rho are those of the governing code of the group; if it is the mask code,
   repeated to the base code's length. For an integer m >= 3: root = sqrt(rho); eta = root / m; alpha = root + eta; delta = 1 - alpha. eta is the
   **slack** below the Johnson radius 1 - sqrt(rho). m is sought (item 11).

   A closeness count has one of two sources, which item 7 names "third" and "Johnson". Which of
   them a code gets follows from the distance, and from whether the distance was made from that
   code's m. Item 7 has the test as the programs make it; the next paragraphs say on what each
   source rests.

   The first source, "third". A code that item 7's first paragraph reads within a third of its
   minimum distance has the count of a lemma on mutual
   correlated agreement within a third of the minimum distance, which the programs' source states
   for every additive code and so for every interleaving: if 3 floor(delta * len) is below the
   minimum distance, the count is floor(delta * len) + 1. By the programs' source this count follows a bound of the paper that this
   protocol builds on, and is proved there; for that paper the programs' source points to
   `THIRD_PARTY.md` at the repository's top. By `docs/limits.md`, "What the bounds rest on depends
   on the parameter file", this source too rests in part on proofs in the technical report. The
   programs' source names published statements of the same count. Of the first four works it says
   that its citations are second-hand and were not checked against the papers. They are: for a fold
   of two rows, for every linear code and fewer errors than a third of its minimum distance, a
   result of Roth and Zemor in the Ligero paper (ePrint 2022/1608, Appendix A), written out by
   Diamond and Posen (ePrint 2023/630, Theorem 2.1); for tables, and for all coordinates together,
   Angeris, Evans and Roh (ePrint 2024/1399), with its proof in Diamond and Gruen (ePrint 2024/1351,
   Theorem 3.6). Of the fifth work, the survey ePrint 2026/680, the source gives the page (p. 18)
   and does not say whether it was checked: its Theorem 4.8, which the survey credits to the Ligero
   paper's Appendix A, gives for any linear code and delta up to a third of its distance an error of
   at most (delta times the code's length, plus 1) divided by the size of the field, for mutual
   agreement too (by the survey's Lemma 4.6). A second comment gives a sixth work, for every linear
   code at a radius below a third of its distance, and does not say whether it was checked: Bordage,
   Chiesa, Guan and Manzur, "All Polynomial Generators Preserve Distance with Mutual Correlated
   Agreement" (ePrint 2025/2051, version of 6 February 2026), Theorem 6.1, taken for two words, and
   Corollary 8.6. By that comment Theorem 6.1 prints max(e, 1), for e errors, where the other works
   have e + 1 (`blocks/numbers/codes.h`).

   The second source, "Johnson", is that of every other count: of the code from whose m
   the distance is made, at every distance; of the group's other code beyond a third of its
   minimum distance, also between a third and a half of it; of the stored table at every distance
   (item 16); and, beyond a third of its minimum distance, of a code at a distance that a caller of
   the library names itself, which no parameter file can do. Every such closeness count is that of Theorem 4.6, taken with the
   theorem's own parameter M set to 1, of IACR ePrint report 2025/2055. The report is a paper on
   proximity gaps of Reed-Solomon codes by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf. It is
   read as posted on 6 November 2025; the same paper is ECCC TR25-169 of 7 November 2025. These
   pages call this theorem the theorem on codes; it is the theorem of the next sentence, and the
   bound there is the closeness count that it gives. The bound rests on a theorem from the full
   version (ePrint 2025/2055, Theorem 4.6: not in the proceedings; that version gives an outline of
   its proof) of a paper published at STOC 2026, which the programs' accounting takes as given — we have
   checked no proof of the paper's own form of the bound; a statement on codes held to follow from it is proved in the Lean
   library (13.7, condition 4) — (what has been checked of it, and what has not:
   [docs/limits.md](../limits.md#what-the-theorem-assumes)), and on further steps argued on paper in
   our report; that our parameters meet the theorem's conditions is our own reckoning, with
   no bits to spare. The comments of the programs' source say that no proof of it was verified
   (`blocks/numbers/codes.h`); the accounting prints
   `which we take as given and of which we have verified no proof ourselves`.

   The theorem speaks of a whole Reed-Solomon code. Two kinds of code of this format are only a part
   of one: a circle code (field 1), and a mask code that is repeated to the base code's length (both
   fields). For these the programs add one list size to the count (items 8 and 9). They add it
   wherever the count's source is not "third". That this term covers the
   step from a code to a part of it is an argument of this project (comments in
   `blocks/numbers/codes.h` and `blocks/numbers/blocks.h`), not a statement of the cited paper.
   By the comment in `blocks/numbers/blocks.h` the repeated words are a part of a Reed-Solomon code
   under a numbering of the positions that is chosen for the proof (the code of degree at most t
   (d - 1) on the subgroup whose order is the base code's length, with t the number of copies and d,
   in this bracket alone, the mask code's dimension; its rho by item 4's rule for field 2, (d - 1)
   over the mask code's length, is the mask code's own), and the count, as `blocks/numbers/codes.h`
   defines it, and how long a list is, depend on the code as a set of words alone, not on how its
   positions are numbered. By the same comment it is not claimed, and not true, that under the
   format's copy map (R7.2) and the base code's own numbering (R12.11) a repeated word is in a
   Reed-Solomon code on the base code's domain. By the same comment the repeated mask code's count
   requires seven things of the format: one public copy map under which every position of the mask
   code has the same number of copies; a coset as the mask code's domain; a length of the base code
   that divides p - 1; a distance below the mask code's own Johnson radius; mask symbols that
   are combined and compared with mask symbols only; symbols combined position by position only; and a code that is linear over E. The comment does not say, and this section does
   not show, that the format meets the seven. R7.2 has the map, and of a mask row's uses it denies
   only one (a mask row that is added to a main row, with the sum then compared with a single code),
   which is less than the fifth thing asks; R12.11 has the domains, this section's head the bound on
   the length, and items 1, 7, 8 and 10 what becomes of a group whose distance is not below the
   repeated mask code's Johnson radius. The comment names no field. A code's length is a power of
   two (item 4). In field 2 every such length up to 2^32 divides p - 1. In field 1, which the
   programs of this repository do not serve (R1.2), p - 1 has the factor 2 only once, so no such
   length above 2 divides it. The report 2025/2055 states its Theorem 4.6 and prints only an outline
   of a proof: one paragraph of four sentences after the theorem (p. 29), which keeps to the
   separable case and computes no bound, and a pointer to a note that its list of references marks
   as a personal communication (pp. 28, 48). On what the counts for several rows rest is told at the
   end of item 9.

   The programs take the formula as the theorem prints it, m = max(ceil(sqrt(rho) / eta), 3); the same
   paper's Theorem 1.5 and page 24 have sqrt(rho) / (2 eta), and by the programs' source so has
   Theorem 4.12 of the survey ePrint 2026/680; the programs do not use that form. The programs' source
   calls the printed form the weaker reading (`blocks/numbers/setting.h`). Weaker means here that it
   takes less from the theorem, not that it is less safe: by the source, at a given slack its m, and
   so its bound, is never smaller (`blocks/numbers/setting.h`). What the printed bounds rest on under this theorem is told in
   `docs/limits.md`, "What the bounds rest on depends on the parameter file". The protocol page,
   `docs/protocol.md`, calls it the theorem on codes, names the theorem and its paper, says that it
   is taken as given, and says what rests on it; it does not speak of the line `reading`.

   What an acceptance asks of the sources: nothing. No condition of 13.7 reads on which source a
   closeness count rests, in the groups `once-..` or in a request's own groups (`link`, `folded`,
   `omega`, `own`): a `YES` does not say on which source a proof's closeness rounds rest; the
   accounting does (`verifier/src/vf_common.h`).
7. *A code at an agreement.* Given a code (len, kappa) and an agreement alpha: the result is bad if kappa >= len;
   delta = 1 - alpha; far = floor of the hi of delta * len, the most positions at which a word at
   this distance differs. If 3 * far < len - kappa: the **count** is far + 1, the **list size** is
   1, and the count's source is "third". Else the source is "Johnson". Then: root = sqrt(rho); eta = alpha - root; q = root /
   eta. m = max(ceil of q's hi, 3); the result is
   bad if that ceiling does not fit 64 bits. The count is count(m, delta). The list size is 1 if
   surely 2 * delta * len < len - kappa, and list(alpha) else; the result is bad if the quotient in
   list(alpha) is bad. Here, with Mh = ratio(2 m + 1, 2),

       count(m, delta) = (2 * Mh^5 + 3 * Mh * delta * rho) / (3 * rho * root) * len + Mh / root
       list(alpha)     = floor of the hi of (alpha - rho) / (alpha * alpha - rho)

   For a distance that was made from m by item 6: the count is count(m, delta), the list size is
   list(alpha), and the source is "Johnson", at every distance; the list size is not set to 1 within
   unique decoding. The result is bad if m < 3 or kappa >= len, or if the quotient in list(alpha) is
   bad.
8. *A group at its distance.* Take the agreement alpha from the governing code by item 6.
   - The base code's count, list size and source are those of item 7 at alpha; if the
     base code governs, they are those of item 7's last paragraph for m.
   - If the group has a mask code, it is read as repeated to the base code's length (item 4). Its
     count, list size and source are those of item 7 at the same alpha. If the mask
     code governs, they are those of item 7's last paragraph for m. Where the source is not "third",
     this list size is added to this count.
   - The group's list size is the larger of the base code's list size and the list size that item
     7's first paragraph gives for the repeated mask code at alpha. This holds also where the mask
     code governs. Without a mask code it is the base code's list size.
   - The result is bad if the group is faulty, or if the base code's or the mask code's result is
     bad.
9. *The errors that sizing uses.*
   - fe(g) = ratio(g, |E|).
   - pe(rows) = ratio(rows - 1, |E|), and 0 for rows = 0.
   - se(lambda, cube) = (lambda * lambda) / 2 * x^S, with x = fe(cube) if cube >= 2 and x = ratio(1,
     |E| - 1) else; lambda * lambda is an exact integer. The factor lambda * lambda / 2, a union
     over pairs of code words, is Lemma 9.1 of Chiesa, Fenzi and Weissenberg, "Zero-knowledge IOPPs
     for constrained interleaved codes" (ePrint 2026/391, version of February 25, 2026); the factor
     x is a count of the programs' source, by a short argument that the source's comment gives
     itself, for a form that the comment calls a variant of Lemma 9.3 of the same paper
     (`blocks/iop/send.h`).
   - mca(code, rows) = count / |E|; in field 1, if the source is not "third", that code's own
     list size (item 8, first two bullets; not the group's) is added before the division. A mask code's count comes here with item 8's list
     size already in it. So in field
     1 a mask code's count beyond a third has its list size twice.

   The closeness errors: fold(j) is the error of the j-th `FOLD round` of a fold, j = 1 .. k, and
   last that of `BASE lambda`, the first round of a last stage. Here base is the group's base code,
   mask its mask code, c its side count and list the group's list size (item 8):

       fold(j) = mca(base, 2^(k - j) * c) + list * fe(2)   [ + 2 * mca(mask, 2^(h - j))   if the group has a mask code and j <= h ]
       last    = mca(base, c) + list * fe(1)               [ + mca(mask, 1)               if the group has a mask code ]

   A closeness error has a source of its own: the weaker source among the counts that enter it, of
   which "Johnson" is the weaker. So fold(j) with j > h
   has the base code's source, whatever the mask code's source is; fold(j) with j <= h, and last,
   have the weaker of the two. By this source the programs' accounting counts rounds (from
   the programs' source).

   On what the counts for several rows rest, beyond item 6. That a closeness error has no factor for the
   rows of an interleaved word is, in the words of the source, "our own short argument, unrefereed":
   an interleaved Reed-Solomon code is a Reed-Solomon code over a larger field on the same domain,
   and the cited bound counts exceptional challenges without regard to the field. Lemma 4.7
   of ePrint 2026/680 is the alternative, which the programs do not use (`blocks/numbers/setting.h`). In that comment "the cited bound" is
   the bound of Theorem 4.6. The programs' printed accounting
   calls three steps "a bound argued on paper for this protocol". The first: beyond a third of the
   distance a fold's round has no factor for the number of interleaved rows ("for linear codes over
   E and two tables combined by one challenge"); this line is printed only where a plan has a
   closeness term of the source "Johnson". The second: a fold of 2^v
   rows at one point counts v closeness terms (tfold in item 16); this line is printed only where a
   plan has a closeness term at all. The third: in field 1 each term beyond a third has one list
   size more; this line is printed only in field 1, and only where a plan has a closeness term of
   the source "Johnson". Of these three it prints, where a plan has a closeness term at
   all, a line that says on what the bounds for this protocol's own steps rest, and points to
   `docs/limits.md#what-the-theorem-assumes` (`blocks/numbers/accounting.h`). No condition of 13.7
   reads the line `rowcost`, and 1 is the only value that the parser admits (from
   `verifier/docs/parameter-file.md`).

10. *Whether a group holds.* goal(kind, scope) = 2^(w - B), w the value of `work <kind>` for the
    scope. A group is sized under a **load** of three numbers: cube, messages, rows. It holds if its
    result is not bad and: fold(1), or last if k = 0, is at most goal(`closeness`); messages *
    se(list, cube) is at most goal(`sample`); and list * (pe(rows) + fe(1)) is at most
    goal(`field`). cube(g), the cube of a group g, is k + ceil_log2(ell) without a mask code, and
    max(k + ceil_log2(ell), h + ceil_log2(ell_mask)) + 1 with one.
11. *The integer m* is the result of this search, with good(m) meaning that the group
    with m holds. Let low be the first good integer in 3 .. 66; if there is none, m = 0 and the
    group is bad. Let high = low + 1. While high < 2^40 and good(high): low = high, high = 2 * high.
    If high >= 2^40: if good(2^40), m = 2^40 and the search ends; else high = 2^40. While high -
    low > 1: mid = low + (high - low) div 2; if good(mid), low = mid, else high = mid. Then m = low.
12. *A number of positions.* pos(alpha, bits, extra) is the least t with alpha^t + extra <= 2^-bits,
    found so: it is 0 if alpha or extra is bad, if alpha.hi is not below 1, or if extra.hi is not
    below 2^-bits. ok(t) means add(pow(alpha.hi, t, up), extra.hi, up) <= 2^-bits. Let high = 1;
    while not ok(high), double high, and give 0 if it exceeds 2^32. Let low = high div 2. While
    high - low > 1: mid = low + (high - low) div 2; if ok(mid), high = mid, else low = mid. The
    result is high. For a scope, positions(alpha, extra) = pos(alpha, B - w, extra), w the value of
    `work positions` for the scope. While sizing, extra is the **margin** 2^-(B + 30), which stands
    for what else the round's error contains; item 21 derives the number again with the whole error.
13. *One group by itself.* Start with t = 0. In each turn: r = tail(t) for the base code and for the
    mask code; reject if a dimension exceeds its code's length; size the group (items
    6 to 11). If it holds, t' = positions(alpha, margin). If it does not hold because it is faulty,
    drop the mask code and h, size the base code alone under the same load, and let t' be its
    positions(alpha, margin) if that holds, else 0. In every other case t' = 0. Reject if the group
    does not hold and t' <= t, and if t' = 0. If t' <= t, stop: the group of this turn and t are the
    result. Else t = t'. Reject after 32 turns.
14. *The later stages of a test.* Let o be the stage before, and (k', h', n', nz') the stage's four
    numbers. Its rows are over E and its side count is 1. The main message: ell' = ell_o * c_o /
    2^k', rejected if the division leaves a remainder. The mask message: carried = 2 * ell_mask_o +
    r_o * c_o + 2 * r_mask_o; least = 2 k' + 1 - h' if k' > 0, else 1; ell_mask' = max(up((carried +
    S + 2^h' - 1) div 2^h'), up(least)). The number of entries that fill the mask rows up, the pad
    among them, is 2^h' * ell_mask' - carried. Then, from t = 0, the iteration of item 13 with r' =
    r_mask' = tail(t) and the load cube = cube(stage), messages = 1, rows = 4 + 3 * max(r_o, 1024),
    in the chain's scope. Here item 13's rejection at t' = 0
    is not made; a stage that ends with t = 0 is rejected when the plan is made (item 21). A test
    needs k > 0 at stage 1 and at least one later stage.
15. *The group `link`.* All segments of a request have one pair of codes, one distance and one
    number of positions. Request scope. The group has k = the largest k of R12.16 over all groups of
    the chain of segments; h = 0; side count = the largest up(1 + q) over the groups, q the number
    of main rows of the messages that are kept into the next group. This k, h = 0 and this side
    count are for the sizing alone: the fold of a segment has as many rounds `FOLD round` as the k
    of the segment's own group (R14.4), and the plan computes the errors of that fold with the k,
    the h and the side count of the segment's own group. The load is:
    - cube = max(cube(group), the largest cube of one oracle message of the chain, the carrying message (R12.14) among them and the index-mask message (R14.4 (a)) not among them), where a message with r_1 main and r_2 mask rows has the cube ceil_log2(r_1) + l, or ceil_log2(r_2) + lz, or the larger of the two plus 1 if it has both;
    - messages = max(8, 2 * (the most messages of a segment) + 2);
    - rows = x + 4096 * y.

    x is the largest, over the groups, of the sum of these four numbers:
    - the number of open rows that the segment's sub-proof leaves: 0 for the last group, and 1 for every segment of a model, whose sub-proof ends in one equation (13.9);
    - 1 + S if the group is not the first;
    - for each kept message that enters the group, twice the number of answers to its out-of-domain sample;
    - for each message of the segment, the number of answers to its out-of-domain sample.

    y is 3 plus the largest number of main rows of kept messages that enter one group. The group is solved by item 13. The message into which the last segment is closed has the same two codes, distance and t; its group has the k of the last group of R12.16, h = 0, side count 1 and rows over E. It is stage 1 of the last test, whose later stages follow by item 14 from the line `stages link`.
16. *The stored table.* Its code is (n, i, m_T) of the line `table`, and it has 2^b rows. The
    **one-time test** is the test that a committer makes on the anchor once, in the key file, before
    a batch or a link reads it; a leaf table is read at the same agreement once, in its leaf file's
    link (item 17). Rejected if the code's dimension (item 4) exceeds 2^n. The **later agreement** is the one at which
    a batch and a link read a table that has passed its one-time test.

    Both distances of the stored table, the **one-time distance** and that of the later agreement,
    come from the formula of item 6 on the table's code.
    - For an integer m, x is the table's code at the distance that item 6 makes from m: its count, list size and source are those of item 7's last paragraph. In this item, list is the list size of x.
    - tfold(x) = the sum over j = 1 .. b of mca(x, 2^(b - j)), added one by one from 0, then + list * fe(b).
    - good(m) in a scope with a number s of sharing requests: s * fe(w_T) + 2 * tfold(x) <= goal(`closeness`, scope).
    - m_once is the result of item 11's search in the committer scope with s = 1 and the further condition se(list, ceil_log2(i)) <= 2^-B: the target itself, since the entry `RECORD` has no proof of work.
    - m_later is the smaller of m_once and the search's result in the batch scope with s = N and without that condition. Rejected if m_later is 0.
    - The one-time agreement is that of m_once, the later agreement that of m_later.

    The closeness errors of the table's own rounds (from the programs' source). Let once be the table's code at the one-time agreement, and later the table's code at the agreement of m_later. `LIN point` in a key file's shared chain: fe(w_T) + tfold(once). `LIN point` in a proof's shared chain: N * fe(w_T) + tfold(later). `LINK point`: fe(b) + tfold(once) + tfold(later). Such an error has the source of its count; `LINK point` has the weaker of the two. `LIN point` samples w_T elements of E and `LINK point` b of them, and all may be 0 (x and y of item 22).
17. *Columns of the stored table*, each without a margin. A batch opens positions(later agreement)
    columns in the batch scope. The key file's one-time test opens positions(one-time agreement) in
    the committer scope. A leaf file's link opens, in one round `LINK`, positions(later agreement)
    columns of the anchor and positions(one-time agreement) of the leaf table, both in the committer
    scope. The round's error is the larger of the two powers alpha^t, not their sum, and the eighth
    word of its entry is the sum of the two numbers. The ground for the larger is an argument in a
    comment of the programs' source, not a proof: which of the two folded rows is another row than
    its table's fold is fixed before the positions are sampled, and that table's own list must miss
    it (`blocks/iop/link.h`; item 22 has what the programs print of all such errors). The numbers of
    columns are fields of the shapes (R13.9, R13.13). Let cov be the number of columns that the
    table's tail covers (item 4: m_T in field 2). The limit of a leaf table, that is, the number of
    batches that it can serve after its link, is (cov - one-time columns) div (a batch's columns),
    and 0 if cov is not above the one-time columns. No verifier tests that count on a proof.
18. *The three groups of the test on the table:* `folded`, `omega`, `own` in a proof (batch, batch
    and request scope; N requests), and `once-folded`, `once-omega`, `once-own` in a key file
    (committer scope; 1 request). First the shapes (R13.13). entries = i rounded up to a multiple of
    up(m_T), plus m_T; this rounding is made in both fields, in field 2 too, where item 4 has none.
    l_folded is l of the line `group folded`, l_omega is l of `group omega`, and lz_own is lz of
    `group own`; in a key file they are those of `group once-folded`, `group once-omega` and
    `group once-own`. The messages:
    - The folded row has ceil((entries + S) / 2^l_folded) main rows.
    - Omega has ceil((matrix rows + S) / 2^l_omega) main rows, where matrix rows is the number of rows of the committed matrix (R13.13), which for a model is m0 of 11.5 and not i / 2^(w_T - b).
    - W has ceil(2^w_T / 2^l_omega) main rows, and one mask row if its rows leave fewer than S free cells.
    - The shared split's mask message has one mask row.
    - The index-mask message has the least number Q of mask rows, at most 4,096, with Q * (2^lz_own - S) + Q - 1 >= the number of entries taken from it; the files are rejected if no Q up to 4,096 serves. The entries taken from it are, in a proof, one for each pending row of a request (13.4, step 5), plus 1 + 2 w_T if there is more than one pending row; in a key file, two for each part of the lookups of item 20, plus 1 + 2 w_T if there is more than one part.

    All rows are over E, and every group has the side count 1. k and h of a group follow from its members as in R12.16: each member counts up(r) for its r rows of that kind, and 0 if it has none; h = ceil_log2(mask rows that count); k = max(ceil_log2(main rows that count), h, 1).

    Members: `folded`: the folded row and the shared split's mask message; `omega`: Omega and the shared split's mask message; `own`: the index-mask message, W and Omega.

    Then the numbers, all three together, from t_f = t_o = t_w = 0. In each turn let D = t_o + requests * t_w, and size all three groups with the values that t_f, t_o, t_w have at the start of the turn, each with cube = cube(group) and messages = 2:
    - `folded` with the tails r = tail(t_f), r_mask = tail(t_f + t_o) and the load rows = S + columns + 1 + 2 S, where columns is the number of columns that this test opens (item 17);
    - `omega` with r = tail(D), r_mask = tail(t_f + t_o), rows = 1 + 2 S;
    - `own` with r = tail(D), r_mask = tail(t_w), rows = 2 + 3 S.

    Reject if a dimension exceeds its code's length. For each group the new number is positions(alpha, margin) if the group holds; if it does not hold, for whatever reason, size its base code alone (the mask code and h dropped, the same load): the new number is positions(alpha, margin) of that sizing if it holds, and 0 if it does not. The files are rejected if the new number is 0. t becomes the larger of the old and the new number. Stop when no t has changed; reject after 33 turns, and reject if a group does not hold at the end. Why jointly: Omega is opened by the batch's test and by every request's own test, and the shared split's mask message by two tests. Then the later stages of each of the three tests follow by item 14.

    All of the above is the case in which `folded` and `omega` differ in n or in l, as they do at all files of 8.5. If they are equal, the condition `two-codes` is missed (8.3); no split is made, and there is no shared split's mask message and no group `omega`: the members of the group `folded` are then the folded row and Omega, and its tails are r = tail(t_f + requests * t_w) and r_mask = tail(t_f), and D = t_f + requests * t_w.
19. *A leaf file's link.* Its one group, under the line `group once-folded`, contains the two folded
    rows. Each has fr = ceil((entries + S) / 2^l) main rows over E, with l of that line. k = 1 +
    ceil_log2(fr), h = 0, side count 1. Committer scope. Load: cube = cube(group); messages = 2;
    rows = 2 S + both numbers of columns of item 17 + 1 + 2 S. It is solved
    by item 13, and its later stages by item 14 from `stages once-folded`. Its numbers differ from
    those of the key file's group `once-folded`, whose mask rows are opened at two lists.
20. *The key file's proof of ranges.* A lookup (R13.14) whose list of the digits' weights has q
    entries, g = ceil_log2(q), is cut into parts of 2^(L - g) cells (lp of R13.14 is L - g), L the
    value of `ranges-part`. Rejected if L exceeds ceil_log2(matrix rows), or if a lookup would have
    more than 256 parts. A part has v = L - g + w_T variables. planes, the number of the lookup's
    planes, is the least number that is at least 2, at least v + 1, and large enough that 2^planes
    exceeds the sum of 2^v over its parts; mu = max(1, ceil_log2(planes)). The message of the
    multiplicities has main rows over F by the layout of the planes' blocks, and the least number of
    mask rows with room, as in item 18, for the lookups' own mask entries. Its group, under
    `group once-ranges`, has k and h from these two numbers as in item 18, and the side count 1.
    Committer scope. Load: cube(group); messages = 1; rows = the number of rows that the test
    batches. It is solved by items 13 and 14. The layout and the count of mask entries follow from
    the fields of R13.14; this section does not list them. The count of batched rows is under
    R13.18.
21. *What a round samples.* When the plan is made, each number of positions is derived once more,
    with the round's whole error, in the chain's scope. `LOC` is the round of a segment's group and
    of the table's columns, `CS` that of every stage of a test but the last, `BASE` that of the last
    stage.
    - `LOC` and `BASE`: positions(alpha, 0).
    - `CS`, the switch from a stage to the next: start with t = 0 and repeat t = positions(alpha, list_next * pe(1 + S + sym * t)), with sym = c + 2 if the stage has a mask code and sym = c else, until t does not change or is 0; alpha, c and sym are those of the stage itself, list_next is the list size of the next stage. After the positions a `CS` round samples one element of E, which may be 0 (section 9), so x = 1 and y = 0 in item 22; the eleventh word of its entry is 1 + S + sym * t.
    - A plan is rejected if such a number is 0, or, for `CS` and `BASE`, exceeds the t for which the stage's tails were sized. The number is the eighth word of the round's entry, and the ninth is a, where the base code has length 2^a (R4.3).
    - `RNG` samples K * T matrix entries (the tenth word of its entry). K is the number of columns of all matrices of the segment's smallness test (11.1). T = B - w is the number of combinations, w the value of `work smallness` for the scope. (In a request's scope the programs add a further term, ceil_log2 of the number of requests that share a session; it is 0, because no line sets that number and it is 1.) The round's error is 2^-T.
    - The record has S * 2^b elements of E; the entry `RECORD` has S * (ceil_log2(i) + 1) elements of E, S of them not 0 (R13.15), and the error se(list at the one-time distance, ceil_log2(i)).

    How long a proof file is with these numbers, at most and on average, is in 10.6.
22. *The verdict on a plan.* Each round of each chain has an error, an upper bound as a Real. Its
    whole error adds the distance of its sampled values from uniform: unit = 3 * ratio(2^z - p,
    2^z)^6, z the number of bits of p; whole = error + unit * (x + y) + (1 / |E|^2) * y, each step
    rounded up, where the round samples x elements of E, of which y must not be 0; whole = error if
    x = 0. A plan holds if all of these are so:
    - no step above has rejected it;
    - every round has whole <= 2^(w - B), w its proof-of-work bits;
    - no root is in a round in which the verifier says nothing (R8.5);
    - every round has one of the names of R14.11;
    - all rounds on the same two codes have their errors from the same m. The source itself calls this condition too strict (`blocks/numbers/plan.h`): two stages of one plan that share both codes must share the choice, so a public file that is otherwise in order can be rejected (no lawful file is known at the values that the parser takes; six planning calls, none refused, show no more than that); the source says that the right comparison would be stage by stage, not for each pair of codes. It also names what the condition does not reach: a number that a round reads of another stage, and the lists that a stage's own out-of-domain sample and batching read;
    - the plan does not end inside a prover's message or inside a repeat of rounds;
    - no round's note on what its error was computed from contradicts the round (`word_fault` in `blocks/numbers/plan.h` has the four cases): for one, no round with an error above 0 has the note that its error was computed from nothing.

    There is no sum over rounds: each round is compared by itself. A round **needs its proof of work** if whole > 2^-B. Proof of work counts toward the target: a plan in which rounds need it holds like any other (8.3). On what that rests the programs print, where a line `work ..` has a number above 0: `that g bits of work before a message multiply its round's error by 2^-g is assumed for this program's hash chains; it is proved for another setting in ePrint 2021/582, Theorem 6`. This section gives the errors that sizing uses (item 9), those of `LIN point` and `LINK point` (item 16), and those of the rounds that sample positions, alpha^t plus the term of item 21. It does not list the error of every other kind of round. Of all of them the programs' accounting prints that the blocks' round errors and the compiled bound are cited from published work, some of it not peer-reviewed, or argued on paper in a technical report that is not public and not refereed. The unit of every comparison and of every printed figure is one key file's, one leaf file's and one request's proof, not a batch's. The last line that the accounting's routine prints says so (14.7): `the unit of all figures above: one key's, one entry's and ONE request's proof; a round that the requests of a batch share is counted for all of them at once; counted jointly for a full batch of N requests, each round of a request's own chains reads log2(N) bits less (6 bits at 64)`. In it `entry` means a leaf file, and a request's own chains are those of the request scope (item 3). By a comment beside that line the joint count is computed nowhere there: the library's means for it is the number of requests that share a session (item 21), which no line of a parameter file sets; the planning tool `blocks/numbers/cp_plan.cpp` sets it from a line of its own input file, and the printed line does not follow it (all three lines: `blocks/numbers/accounting.h`).
23. *The order of the whole, and further conditions.* From the public files one routine of the
    programs takes the eight steps below, in this order. The first failure ends the routine, and
    `cp_verify` ends with `NO: parameters-not-admitted`. Before it starts the routine, `cp_verify`
    asks three things itself, in this order. First, a layout file that is missing, malformed or at
    odds with the arrays that the program declares ends the call with `plan-layout`; one that is
    there and cannot be read ends the call earlier, as an unreadable input (14.3). Second,
    `cp_verify` takes step 3 once by itself, and ends with `NO: parameters-not-admitted` if it
    fails. Third, it asks that the layout fits the codes (`plan-layout`), and that the mask rows
    have room for the entries that are taken from them (`plan-mask-row`). This last is two
    conditions: that on Q of item 18, for a proof; and, for every segment, that the number of
    scalars that its sub-proof sends, plus S, is at most 2^lz of `group link`. So public files that
    fail in the sizing and whose layout does not fit the codes get the name
    `parameters-not-admitted`. These three are not all that `cp_verify` asks there, nor is what
    follows (from the programs' source, `verifier/src/vf_segments.inc`; this bears on names and their
    order alone). Before the first come tests on the program's own list of segments (`plan-head`,
    `plan-link`, `plan-shape`, `plan-fold-source`). Between the first and the second: the number of
    segments with claims on the weights is the number of groups of layers plus 1 (`plan-link`; NG +
    1 in the letters of 13.4, step 5), only the embedding's units have four digits, and the list of
    the matrix's units has none of the faults of `wf_layout_fault` in `blocks/frame/wf_units.h`
    (both `internal-weight-limbs`). Within the third, in the same loop over the segments as the second
    condition of `plan-mask-row`: every array that a segment reads is placed by the layout file and,
    if an earlier segment has sent it, is kept up to this segment (`plan-layout`).
    1. Parse the frame lines (8.1).
    2. Read the model's lines and the other files (8.6).
    3. Derive items 15 to 18, then item 20, then item 19: the key file's proof of ranges comes before a leaf file's link. The order decides only which rejection is printed where several apply.
    4. Make the plans of a proof file, a key file and a leaf file. Here a guard (17.2, item 1), not a rejection, asks that no seal has proof-of-work bits (R5.3), and nothing asks it again at a run; by item 3 no public file gives a seal such bits.
    5. Ask item 22 of every chain.
    6. Ask ten conditions on hiding, by the names `main-rows`, `mask-rows`, `mask-room`, `fold-room`, `table`, `padding-entries`, `range-checks`, `salt`, `leaves`, `batch`.
    7. Form the configuration's digest (R4.9).
    8. Ask that the shapes fit each other; among the conditions are these seven:
        - a key file's binding tree has depth 0;
        - the multiplicities' message is the second oracle message of the one-time chain;
        - there is one identifier for each segment;
        - there is one message of masks at the start of a request;
        - there is at least one pending row;
        - there is one plain scalar, the next token, in the last segment and in no other;
        - the layout fits the codes.

    The depth of a batch's binding tree is ceil_log2(N) (R14.6). As `cp_verify` asks them, the conditions on hiding compare numbers that follow from the public files alone; condition 5 has three further counts, which only the holder of a stored table knows (16.1). No proof can show whether the conditions were met. This section names the ten conditions; 16.1 defines them. They are not the ten rows of 13.7's table (`routed-experts` to `rotary-table`), which every verifier asks. They can only reject: a verifier that leaves them out accepts some public files that `cp_verify` rejects, and judges no file under admitted public files differently. Three parts of them every verifier asks all the same (why: 11.1 for part (a), 16.1 for the other two): parts (a) and (d) of condition 3, and the fourth line of condition 6, which is the first of the further conditions below. Others of the ten are asked again by the parser or among the further conditions, or hold by the derivation (16.1, "Who else asks each of the ten").

    These further conditions read a frame line. They are asked within step 3, and each ends in `NO: parameters-not-admitted`:
    - The matrix has at least one row; w_T - b (k_lo of R13.13) <= 30; ceil_log2(matrix rows) <= 30; w_T (k_col of R13.13) <= 60; and (matrix rows) * 2^(w_T - b) is at most i - 3 S. The item `ranges` has bounds of its own on the matrix's rows and on k_col (under R13.14: 40 for each, which is narrower for k_col alone), and `cp_verify` asks w_T <= 20 (8.3).
    - The table's three numbers are a code by item 4, so in field 2 i + m_T <= 2^32. Also 3 S < i.
    - At every stage but the last the mask message has at least 2 k + 1 - h entries. This holds for stage 1 too, where the length 2^lz is a line's number.
    - Every code of every stage is a code by item 4.
    - S < 2^l and S < 2^lz of `group link`; and S < 2^lz of `group own`, for the index-mask message, which is sent at the start of a request.
    - For every group of the chain of segments but the last (R12.16), with its own k and h: 2^lz of `group link` is at least 2 k + 1 - h. A segment's group is not a stage of a test, so the condition on the stages does not reach it.
    - For every such group: 2 * 2^lz + r + 2 * r_mask + S, plus, for each kept message that enters the next group, (its main rows) * r and its answers to an out-of-domain sample, these counted three times if its rows are over F and once if they are over E, is at most 2^l. Here l and lz are those of `group link`, and r and r_mask are the two tails of item 15. That is: what a segment's closing carries on fits one main row of the carrying message (13.9).
    - The same step tests the shape of the chain of segments. These tests read the layout file and no frame line: for example, no message is kept beyond the last segment, and the first segment has a main row that is not kept. They are listed under R12.12.
    - In item 20: a part is not smaller than a pair of rows (R13.14), that is, L - g is at least 1 in a lookup that makes each row of a pair a key of its own, and at least 0 in every other; mu <= 6; d of R13.14 is 1 to 40; no block of planes is longer than a main row; and 2^lz > S.

    This list need not be whole (17.1, item 4).

### 8.5 The values at the example files

The tables below give what 8.4 yields at two parameter files: `record.f2.txt`, in
`blocks/frame/examples/`, and `target_small_f2/policy.txt`, in `prover/examples/`. The first 32
lines of `prover/examples/target_full_f2/policy.txt` are those of `record.f2.txt`, so that column is
its column too. Both are in field 2 with `sec 128`, `reading 1`, `rowcost 1`, `slots 64`,
`answers 1` and `work positions 20 28 32`.

As the parameter file the programs read the file named `policy.txt` in the directory that they are
given, and no other. The frame lines of `target_small_f2/policy.txt` are also those of `policy.txt`
in `target_small_f2_roomy/` and `target_small_f2_gap/`.

These values follow from 8.4; a second implementation can use them as known answers.

In each cell: the base code as (length, ell, r); the mask code likewise; the integer m; delta rounded to six decimals (its two ends agree in all of them); the list size; and t, the
number of positions. An empty cell: the test has fewer stages there. At both files the number that a
round samples (8.4, item 21) equals the number for which the tails were sized.

The values do not depend on k, on the side count or on the loads of stage 1. The reason
is, at `record.f2.txt`, that the first condition of 8.4, item 10, is the only one that binds; that condition does not read
the numbers of rows. For `target_small_f2/policy.txt` this
page has no such reason, only a trial of its own: the cells came out the same at a few other values
of k and h; the side count and the loads were not varied. h of stage 1 is 0 for `link`, `folded`,
`omega`, `once-folded`, `once-omega` and a leaf file's link. It is above 0 for `once-ranges`, and
for `own` and `once-own` where W has a mask row. No cell depends on whether h of `own` and
`once-own` is 0; `once-ranges` needs h > 0 wherever its mask code governs (8.4, item 5), as it does
in `record.f2.txt`.

| group | stage | `record.f2.txt` | `target_small_f2/policy.txt` |
|---|---|---|---|
| `link` | 1 | (2^23, 4194304, 219); (2^17, 8192, 219); m 258; delta 0.290134; list 54; t 219 | (2^18, 131072, 218); (2^17, 8192, 218); m 517; delta 0.290939; list 107; t 218 |
| `link` | 2 | (2^22, 262144, 55); (2^17, 4096, 55); m 158; delta 0.748392; list 238; t 55 | (2^17, 8192, 55); (2^17, 4096, 55); m 319; delta 0.748391; list 477; t 55 |
| `link` | 3 | (2^19, 32768, 55); (2^16, 2048, 55); m 241; delta 0.748756; list 362; t 55 | (2^16, 1024, 44); (2^16, 2048, 44); m 260; delta 0.820690; list 599; t 44 |
| `link` | 4 | (2^16, 4096, 55); (2^15, 1024, 55); m 367; delta 0.747672; list 546; t 55 | (2^15, 128, 44); (2^15, 1024, 44); m 300; delta 0.818948; list 682; t 44 |
| `link` | 5 | (2^13, 256, 46); (2^13, 256, 46); m 378; delta 0.807808; list 798; t 46 | (2^15, 128, 73); (2^15, 4096, 73); m 520; delta 0.642667; list 469; t 73 |
| `link` | 6 | (2^15, 256, 44); (2^15, 1024, 44); m 345; delta 0.819027; list 785; t 44 |  |
| `folded` | 1 | (2^24, 4194304, 101); (2^13, 64, 152); m 182; delta 0.497247; list 91; t 101 | (2^17, 32768, 101); (2^13, 64, 156); m 483; delta 0.498201; list 241; t 101 |
| `folded` | 2 | (2^21, 131072, 54); (2^12, 256, 54); m 168; delta 0.723703; list 222; t 54 | (2^14, 2048, 68); (2^12, 256, 68); m 600; delta 0.640111; list 535; t 68 |
| `folded` | 3 | (2^17, 8192, 54); (2^12, 256, 54); m 293; delta 0.724400; list 388; t 54 | (2^14, 2048, 68); (2^14, 1024, 68); m 600; delta 0.640111; list 535; t 68 |
| `folded` | 4 | (2^13, 512, 54); (2^12, 256, 54); m 510; delta 0.724799; list 674; t 54 |  |
| `folded` | 5 | (2^14, 512, 51); (2^14, 1024, 51); m 489; delta 0.743446; list 711; t 51 |  |
| `omega` | 1 | (2^22, 262144, 3571); (2^13, 64, 152); m 159; delta 0.746720; list 237; t 51 | (2^18, 16384, 3831); (2^13, 64, 156); m 295; delta 0.721371; list 384; t 55 |
| `omega` | 2 | (2^18, 16384, 51); (2^14, 512, 51); m 277; delta 0.748715; list 415; t 51 | (2^14, 1024, 69); (2^13, 1024, 69); m 527; delta 0.634203; list 459; t 69 |
| `omega` | 3 | (2^14, 1024, 51); (2^13, 256, 51); m 489; delta 0.743446; list 711; t 51 | (2^15, 1024, 68); (2^15, 4096, 68); m 520; delta 0.642881; list 470; t 68 |
| `omega` | 4 | (2^15, 1024, 41); (2^15, 1024, 41); m 299; delta 0.819201; list 681; t 41 |  |
| `own` | 1 | (2^22, 262144, 3571); (2^13, 256, 55); m 159; delta 0.746720; list 237; t 55 | (2^18, 16384, 3831); (2^13, 256, 59); m 295; delta 0.721371; list 384; t 59 |
| `own` | 2 | (2^18, 16384, 55); (2^15, 1024, 55); m 277; delta 0.748684; list 415; t 55 | (2^14, 1024, 75); (2^13, 1024, 75); m 528; delta 0.633201; list 458; t 75 |
| `own` | 3 | (2^15, 2048, 55); (2^14, 512, 55); m 423; delta 0.746127; list 624; t 55 | (2^15, 1024, 73); (2^15, 4096, 73); m 520; delta 0.642667; list 469; t 73 |
| `own` | 4 | (2^13, 256, 46); (2^13, 256, 46); m 378; delta 0.807808; list 798; t 46 |  |
| `own` | 5 | (2^15, 256, 44); (2^15, 1024, 44); m 345; delta 0.819027; list 785; t 44 |  |
| `once-folded` | 1 | (2^24, 4194304, 97); (2^13, 64, 146); m 182; delta 0.497247; list 91; t 97 | (2^17, 32768, 97); (2^13, 64, 146); m 483; delta 0.498231; list 241; t 97 |
| `once-folded` | 2 | (2^21, 131072, 52); (2^12, 256, 52); m 167; delta 0.724589; list 222; t 52 | (2^14, 2048, 66); (2^12, 256, 66); m 600; delta 0.640281; list 536; t 66 |
| `once-folded` | 3 | (2^17, 8192, 52); (2^12, 256, 52); m 292; delta 0.725290; list 388; t 52 | (2^14, 2048, 66); (2^14, 1024, 66); m 600; delta 0.640281; list 536; t 66 |
| `once-folded` | 4 | (2^13, 512, 52); (2^12, 256, 52); m 509; delta 0.725690; list 676; t 52 |  |
| `once-folded` | 5 | (2^14, 512, 49); (2^14, 1024, 49); m 489; delta 0.743685; list 712; t 49 |  |
| `once-omega` | 1 | (2^22, 262144, 98); (2^13, 64, 146); m 158; delta 0.748371; list 238; t 49 | (2^18, 16384, 98); (2^13, 64, 146); m 277; delta 0.748356; list 415; t 49 |
| `once-omega` | 2 | (2^18, 16384, 49); (2^14, 128, 49); m 277; delta 0.748730; list 415; t 49 | (2^14, 1024, 49); (2^13, 128, 49); m 489; delta 0.743685; list 712; t 49 |
| `once-omega` | 3 | (2^14, 1024, 49); (2^13, 64, 49); m 489; delta 0.743685; list 712; t 49 | (2^15, 1024, 39); (2^15, 512, 39); m 345; delta 0.819451; list 787; t 39 |
| `once-omega` | 4 | (2^15, 1024, 39); (2^15, 512, 39); m 345; delta 0.819451; list 787; t 39 |  |
| `once-own` | 1 | (2^22, 262144, 98); (2^13, 256, 49); m 158; delta 0.748371; list 238; t 49 | (2^18, 16384, 98); (2^13, 256, 49); m 277; delta 0.748356; list 415; t 49 |
| `once-own` | 2 | (2^18, 16384, 49); (2^15, 128, 49); m 277; delta 0.748730; list 415; t 49 | (2^14, 1024, 49); (2^13, 128, 49); m 489; delta 0.743685; list 712; t 49 |
| `once-own` | 3 | (2^15, 2048, 49); (2^14, 64, 49); m 423; delta 0.746489; list 625; t 49 | (2^15, 1024, 39); (2^15, 512, 39); m 345; delta 0.819451; list 787; t 39 |
| `once-own` | 4 | (2^13, 256, 41); (2^13, 64, 41); m 470; delta 0.809509; list 1002; t 41 |  |
| `once-own` | 5 | (2^15, 256, 29); (2^15, 256, 29); m 201; delta 0.906440; list 981; t 29 |  |
| `once-ranges` | 1 | (2^22, 262144, 49); (2^16, 4096, 49); m 138; delta 0.746717; list 206; t 49 | (2^22, 262144, 49); (2^16, 1024, 49); m 158; delta 0.748395; list 238; t 49 |
| `once-ranges` | 2 | (2^19, 16384, 39); (2^17, 2048, 39); m 195; delta 0.822111; list 455; t 39 | (2^19, 16384, 39); (2^17, 512, 39); m 195; delta 0.822111; list 455; t 39 |
| `once-ranges` | 3 | (2^17, 2048, 33); (2^17, 1024, 33); m 210; delta 0.873427; list 730; t 33 | (2^17, 2048, 33); (2^17, 256, 33); m 210; delta 0.873427; list 730; t 33 |
| `once-ranges` | 4 | (2^20, 2048, 25); (2^20, 4096, 25); m 91; delta 0.936628; list 684; t 25 | (2^20, 2048, 22); (2^20, 1024, 22); m 74; delta 0.954980; list 801; t 22 |
| a leaf file's link | 1 | (2^24, 4194304, 97); (2^13, 64, 97); m 182; delta 0.497247; list 91; t 97 | (2^17, 32768, 97); (2^13, 64, 97); m 483; delta 0.498231; list 241; t 97 |
| a leaf file's link | 2 | (2^21, 131072, 49); (2^12, 128, 49); m 182; delta 0.748580; list 274; t 49 | (2^14, 2048, 66); (2^12, 128, 66); m 600; delta 0.640281; list 536; t 66 |
| a leaf file's link | 3 | (2^17, 8192, 49); (2^12, 128, 49); m 318; delta 0.748480; list 476; t 49 | (2^14, 2048, 66); (2^14, 512, 66); m 600; delta 0.640281; list 536; t 66 |
| a leaf file's link | 4 | (2^13, 512, 50); (2^12, 128, 50); m 570; delta 0.737851; list 805; t 50 |  |
| a leaf file's link | 5 | (2^14, 512, 40); (2^14, 512, 40); m 348; delta 0.816087; list 776; t 40 |  |

The stored table and what else has one value for each file:

| value | `record.f2.txt` | `target_small_f2/policy.txt` |
|---|---|---|
| B less the value of `work positions`, for the request, batch and committer scopes | 108, 100, 96 | 108, 100, 96 |
| the table's rate rho, and kappa | 0.104004 (kappa = 446693374) | 0.070312 (kappa = 589822) |
| one-time distance; source of the count; list size | 0.664066; Johnson; 26 | 0.731602; Johnson; 114 |
| m_once, m_later | 24, 24 | 82, 82 |
| later agreement | 0.335934 | 0.268398 |
| columns that a batch opens | 64 | 53 |
| columns that the key file's one-time test opens | 62 | 51 |
| columns that a leaf file's link opens: anchor, leaf table | 62, 62 | 51, 51 |
| the limit of a leaf table, in batches | 688127 | 1235 |
| main rows of the folded row | 107 | 18 |
| combinations of the smallness test, for the request, batch and committer scopes | 128, 128, 128 | 128, 128, 128 |
| depth of a batch's binding tree | 6 | 6 |
| pads in a row's message | 3 | 3 |
| the record | 49152 bytes; a point of 29 coordinates; 30 elements of E sampled; error 2^-178.74 (without item 22's addition) | 6144 bytes; a point of 19 coordinates; 20 elements of E sampled; error 2^-175.09 (without item 22's addition) |

`record.f2.txt` ends with the line `table` and has no model. For its column of the two tables above
take for k and h of stage 1: `link` 8, 0, and 1, 0 in the last test; `folded` 7, 0; `omega` 4, 0;
`own` 5, 1; `once-folded` 7, 0; `once-omega` 4, 0; `once-own` 5, 1; `once-ranges` 4, 4; a leaf
file's link 8, 0; the side count 1, 3,076,096 matrix rows, 13 pending rows, lookups of 24 parts in
all (8.4, item 20), and for `link` and `once-ranges` any load: no cell depends on the choice. At
`target_small_f2/policy.txt`, whose model has 59,392 matrix rows, 3 pending rows and lookups of 8
parts in all, k of stage 1 is, as a planning call of `cp_verify` prints it (14.6): `link` 9, and 1
in the last test; `folded` 5; `omega` 2; `own` 3; `once-folded` 5; `once-omega` 2; `once-own` 3;
`once-ranges` 3; a leaf file's link 6. In this page's own trial no cell of its column changed with
h.

The sources of the counts. By the lines `footing:` of a planning call of `cp_verify` (14.6),
under `target_full_f2/policy.txt` no round with a closeness term rests on the bound within a third of the distance: all 63
such rounds of the key file's plan, all 23 of a leaf file's and all 255 of a proof's rest on the
theorem on codes (8.4, item 6). Under `target_small_f2/policy.txt` these are all 37, all 12 and
all 78.

### 8.6 The model's lines

After the line `table` a parameter file for `cp_verify` goes on with the lines of the model. The
parser of the frame lines does not read them (8.1), but every byte of them enters the
configuration's digest. `cp_verify` reads them with a second strict scan. A **value line** is a
name, one blank, and a decimal numeral below 2^32 without a sign or a leading zero; a value of 64
bits takes two lines, `<name>_lo` and `<name>_hi`. Every line ends with one line feed and has at
most 1,023 bytes, its line feed counted. An optional line is left out when its value would be the
one that holds without the line, so that given values have one encoding. The lines come in this
order:

| part | lines |
|---|---|
| profile and shape of the model | `prof` `lt` `L` `ll` `d` `H` `G` `dh` `f` `fe` `NE` `V` |
| constants of the integer arithmetic | `eps1_lo` `eps1_hi` `eps2_lo` `eps2_hi` `cs` `theta0_lo` `theta0_hi` `hq` `hv` `ho` `hgD` `huD` `hdD` `hr` `ut` |
| committed matrices of a feed-forward segment | `la` `nmin` `D` `tau` `lmaf` `NAf` `lmad` `NAd` `lmar` `lmb` `lmq` `lmx` |
| widths of the weights | `wb_qk` `wb_v` `wb_o` `wb_g` `wb_u` `wb_gs` `wb_us` `wb_gr` `wb_ur` `wb_dn` `wb_dnr` `wb_r` |
| lookup calls of a feed-forward segment, and the seed | `iota_0` to `iota_7`, `seed_lo` `seed_hi` |
| attention | `lB` `S` `rx` `kFM` `kFD` `lmS` `NAs` `lmh` `lmo` `lmbA` `iotaA_0` |
| output head | `MF` `KF` `epsF_lo` `epsF_hi` `toksrc` `lmhf` `NHf` `lmhd` `lmhr` `lmbH` `wb_emb` `wb_head` `iotaH_0` to `iotaH_5` |
| acceptance policy | `C` `theta` `cap` `s` `sh` `sT` `F` |
| optional, each may be absent; `nminH` is admitted only directly after `lidb` | `lidb` `nminH` `eg` `smax` `rcl` `gcl` `xl` |
| digests of the four public tables, all 32 or none | `tsilu0` to `tsilu7`, `tsig0` to `tsig7`, `trope0` to `trope7`, `texp0` to `texp7` |
| call lines | all lines `call`, then all `callA`, then all `callH`: the word, the call's name, the kind of its table, and the names of one or more of the call's segments, separated by single blanks; each of these consists of letters, digits, `.`, `_` and `-` alone, and the call's name has at most 8 characters (else `policy-call-line`) |

Three of these lines are read by name before the scan: `prof` must be 16 to 31 and `toksrc` must be
1, else the call ends with `NO: statement-required`; `lt` is the third. A whole call also reads `C`,
`theta`, `cap`, `s`, `sh`, `sT` and `lt` by name before the stages, and can end there with
`NO: policy` (14.3): for one, where `sT` is not `sh` - 4 (13.8 has the cases in which these tests
are left out). With `lt` it also tests the request file's length there, before it reads the file
(13.8, the table "Before the stages", `statement-range`). Which of these lines give a word of the
item `request`, and in which order, is R15.2 and R15.5. Two of them meet the frame lines: `tau` and
`V` (8.3).

This section does not specify the model's lines further: not the range of each value, not the tests
of the scan with their rejection names, and not how the shapes of a segment's oracle messages, the
layout and the lookups follow from them. Section 11 describes the sub-proofs as far as this page
goes, and section 13 lists the rejections. The other two public files, from which the shapes of a
segment's oracle messages also follow, are described in 13.4: the scale file in step 2, the layout
file in step 7. The segments of a model, their order and identifiers, and its number of pending rows
are in 13.4, step 5; the committed matrix and the number of its rows, m0, are in 11.5.

### 8.7 Easy to get wrong

- *The derived numbers are hashed, and they are not round numbers.* No file contains a number of
  positions, a tail or the mask length of a later stage. They come out of fixed-point iterations in
  the arithmetic of 8.4, item 1, and they are coupled across groups: in `record.f2.txt`, Omega's
  tail is 51 + 64 * 55 = 3,571. They enter the first queries as codes and as plan entries. An
  implementation in floating point, or one that regroups a formula, can be off by one position and
  then rejects every honest proof. Either rebuild the arithmetic bit for bit, and compare with 8.5,
  or take the plan, the codes and the shapes as inputs.
- *A ceiling at the upper end.* In 8.4, item 7, m is the ceiling of the hi of q, not of q. Where
  base code and mask code are the same code and the base code governs, the exact
  quotient root / eta is the group's own m; its hi is a little more, so the mask code is counted at
  m + 1. This decides four cells of 8.5, at `record.f2.txt`: `link`, stage 5, and `own`, stage 4 (m
  378), `omega`, stage 4 (m 299), and `once-own`, stage 5 (m 201). With the mask code at m the search of item 11 would end at
  379, 300 and 202, with another delta and list size.
- *Two numberings that are not the expected ones.* The kind of a round in which the verifier says nothing is 7 in the query (R5.2), and
  `RECORD` is an entry of a plan and not a round (R13.15). The line `work positions` is the fourth
  of the five lines `work ..`, and the columns A, B, C are request, batch, committer, not the order
  of the files.
- *Where things are inside a code and a row.* In field 2 the tail comes first: the symbol at z is
  the sum of t_j z^j over j < r plus z^r times the sum of m_j z^j over j < ell, on the coset 7 H in
  bit-reversed order (R12.11). In field 2 the rate in every formula is (ell + r - 1) / 2^a, not
  (ell + r) / 2^a; in field 1 kappa is the dimension itself (8.4, item 4). The pads
  are the last 3 * S of the i entries of a row's message, not entries beside them.
