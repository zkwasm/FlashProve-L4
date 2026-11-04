# What the planning calls print

A planning call checks nothing. It runs the routines by which a checking call plans itself and prints what follows from the public files. Its last line is `PLAN ONLY: no proof was checked`, its status 4. Files that a checking call would refuse for their parameters are refused here under the same names. An exception: a planning call does not refuse a parameter file because its `sT` is not `sh - 4`, though no whole call under such a file can pass (`NO: policy`); `-plan request` and `-plan accounting` say so in the line `plan: the policy's kind is <kind>: every whole call under this parameter file ends NO: policy`. The calls themselves, with all their options, are in the program's usage text; [../README.md](../README.md), "Planning calls", names all four and prints one of them.

The lines are made for programs as well as for people: one line, one fact, the kind of line as its first word. Most are printed by `blocks/`: `blocks/frame/prints.h` (groups, tables, lengths, footing), `blocks/numbers/accounting.h` (the accounting), `blocks/frame/fields.h` (fields).

## Which call prints what

| call | lines |
|---|---|
| `-plan accounting <config folder> <asked>` | `group:`, `table:`, `smallness test:`, `accounting:`, `work:`, `footing:`, `length:`, `plan:`, `statement:`, `h_par=`, `configuration=`, `dw=` |
| `-plan request ..` | the same; without `-q` also `ok ..`, `layout:` and one line `link <n>:` for each sub-proof |
| .. `-shapes` | also `round:` |
| .. `-fields` | also `field:` |
| `-plan arrays <config folder> <asked>`, or `-plan request .. -arrays` | `array:` lines (`-plan request .. -arrays` without `-q` prints the line `ok ..` first); it needs no layout file |
| `-plan id <config folder> <key.bin>` | `h_par=`, `commitment ID: ` |

## The lines

**`group:`** One line for each group of oracle messages that a proximity test covers, then one for each later stage of that test: code length, message length and random coordinates, rows, the mask rows' code, the distance, the list size, the number of positions, the rule by which the distance was chosen and what the group's closeness terms rest on. First a request's own groups, then, behind `group: in a batch:` and `group: in the committer's files:`, those of the test on the committed table, and last the group of the folded tables of a link. The parameter file gives the code lengths, message lengths and rows of the test's stages; `blocks/frame/derive.h` derives the other sizes from the target less the work.

**`table:`** How many columns of the committed table are opened: in the key's one-time test, in an entry's link, in a batch; and how many batches a stored copy bears.

**`round:`** `round:  2^-<bits>[ = <bits> by the lists|bound + <work> by work], <n> round(s): <name>[  ABOVE THE TARGET]`. Every round of key, entry and proof, the largest error first. Rounds with one error whose names differ only in numbers share a line, with `#` for the number.

**`accounting:`, `work:`, `footing:`** The error accounting: [../README.md](../README.md), "The error accounting". `work: <bits> + <work> (..)` is printed if the rounds hold and at least one of them would fall short of the target without its proof of work: a checking call under these files will count proof of work, and will say so in the line `proof of work: ..` near its end (a part-call: the line before the last; a whole call: the third from the end).

**`length:`** For key, entry and proof: the greatest length in bytes, and the mean over the drawn positions. No file holds a length; how long an answer to a draw of positions is depends on the positions.

**`field:`** `field: <file> at|at-most <place> <bytes> <class>: <name>`. Every field of every file; the list covers every byte. The class says what a reader does with the bytes: `words` (compared with the reader's own value), `elements` (refused unless every coordinate is below p), `digest`, `salt` (any value can be read), `nonce` (tested against the round's work). A place is exact up to the first answer to a draw of positions; from there on it is the greatest possible one, and the line says `at-most`.

**`link <n>:`** The sub-proof that is link n of a request's chain (G0, A0, G1, A1, .., H), its identifier, its commitments, its rows over the two fields, how many of them later links read, its arrays and their blocks, its scalars.

**`array:`** `array: <link> <moment> <group> <kind> <life> <index> <rows> <columns> <low> <name>`, for the tool that makes `layout.txt` (`prover/tools/pack_layout.py --arrays`). Links, moments and groups count from 1, a moment's arrays from 0. The arrays of a moment that no later link reads share one oracle message; an array that is kept has one of its own. kind: `N` over the base field, `W` over the extension field, `B` kept for later links; life: for how many. low: `r` if neighbouring rows of a column are to lie side by side, else `c`.

**`plan:`** Whether a run under these files can be a verification: the profile, and the conditions of the standard parameter form that the three files decide (those of the request stage). Two lines **`accounting: one-time files:`** come between the line on the profile and the line on the missed conditions. They say, for the key's and the entry's plan, how many closeness rounds rest on the theorem on codes and how many on the bound within a third of the distance; each ends `; the table has a record` ([parameter-file.md](parameter-file.md), the row on `record`). The programs, their files and the format say "the record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say "the record's seeds" for its OOD seeds ([glossary](../../docs/glossary.md#record)). The conditions on a call's request and options (`next-token`, `exact-match`, `policy-named`, `id-named`, `no-proof-of-work`) are not asked here. Then what the optional lines of the parameter file set: `lidb`, `nminH`, the clamps, the limbs of the residual stream, and whether this program's four public tables match the file's digests.

**`statement:`** The statement's length in words of 32 bits, and the place of y in it.

**`h_par=`, `configuration=`, `dw=`** One digest of the three configuration files, in two forms, and the words of a digest. `configuration=` prints, byte by byte, the digest that every hash chain of a proof binds (`blocks/frame/heads.h`). `h_par=` prints the same digest word by word, so its digits differ; it is printed, and no verifier compares it with anything; cp_commit writes it into isk.txt's line `ranges` when its worst-case check has passed, and cp_prove compares that line with its own digest of the three files that it has read, and refuses to start if they differ or the line is absent (`ranges-not-guarded`; under `seed test`, in test builds, it does not ask for the line).
