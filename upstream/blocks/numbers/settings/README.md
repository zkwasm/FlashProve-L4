# Files for cp_plan

`cp_plan FILE [--rounds] [--messages]` reads a text file of lines `word key=value ...`. `#` starts a comment. A number is decimal, `2^a`, or a sum or difference of these (`2^22+512`). The program runs no part of a protocol: it calls the functions of `numbers/blocks.h`, with which prover and verifier make their own plans, and prints what they derive. It reads strictly. It rejects an unknown word, key or name, a key that a line does not take, and a line of the setting that comes after a line of another kind.

`full-goldilocks.txt` is a test input of this library and no setting that is offered to anybody. It describes a whole key, entry and proof at full size. No program makes it: it is written out block by block, **in an arrangement other than that of the files of `frame/`**. In that arrangement the stored table's code has length 2^33 and the one-time test has a distance of 1/8; the committer's groups, like the others, go up to the Johnson radius (less a slack); closeness rounds have proof of work as well (20 bits), not only the draws of positions; and the committer's range proof is a stand-in of scalars alone. It is a text from which `cp_plan` plans a whole protocol with the routines of `blocks.h`; only `cp_plan` reads it. **Where this file and the files of `frame/` differ, the file format's numbers are those of `frame/`** (`cp_frame` on `frame/examples/record.f2.txt`), with which the programs that make and check files make their plans. `tests/cp_plan_files.py` runs `cp_plan` on it, on copies with a line or two replaced, and on a text of codes and groups alone.

## The setting

| line | meaning |
|---|---|
| `field goldilocks` | p = 2^64 - 2^32 + 1 with plain Reed-Solomon codes; the challenge field is the cubic extension. It is the only field, and the line may be left out |
| `target 128` | every round's error times 2^-work must be at most 2^-target |
| `saved_coefficient 0` | plan round polynomials with all their coefficients (the programs never send the one that the check fixes) |
| `instances 24` | every round of a request's own part counts this many times |
| `samples 1` | the number of answers to each out-of-domain sample |
| `bytes salt=40 nonce=8 digest=32 block=128` | widths |
| `work positions request=20 shared=28 once=32` | bits of proof of work by the kind of a round's error (`field`, `sample`, `closeness`, `positions`, `smallness`) and by who pays (`request`, `shared` or `once`) |

## Names

| line | meaning |
|---|---|
| `code NAME log_n=23 ell=2^22 r=513 [E=1]` | a code: length 2^log_n, message length, random coordinates; E=1: symbols in the challenge field |
| `group NAME main=CODE [mask=CODE] k=8 h=0 [slots=1] [scope=request] [regime=..] [cube=40] [messages=4] [rows=2^20]` | 2^k main rows, 2^h mask rows. regime: `johnson` (the least slack for which all rounds that depend on the distance hold; the default), `m:73`, `delta:1/8`. cube, messages, rows: the largest sample cube, the largest number of oracle messages that share a seed, and the largest batching that the group must support. |
| `tree NAME log_n=23 rows_F=120 rows_E=2 [copied_rows_E=2 copied_rows_F=0]` | one Merkle tree: the rows whose symbols a leaf holds, and the shorter rows of which it holds copies |
| `table NAME code=CODE v=11 delta_once=1/8 [delta_later=a/b answers=1]` | the committed table: 2^v rows; the later tests' distance and the number of answers for each row, from 1 to 8 |
| `sum NAME of=part+part` | a total over parts |

## The protocol

`part NAME` and `scope request|shared|once` say to which file, and to whose work, the lines that follow belong. `repeat N` .. `end` plans the lines between them once and counts their rounds and messages N times. Every block line takes `where=TEXT` (no blanks), which appears in what is printed.

`chain [salted=1]`, `send samples=GROUP:CUBE,.. roots=1`, `bat group= rows=`, `bat_sigma rows=`, `split`, `seal`,
`psc variables= [degree=3 degree_Y=2 values=2 Y=1]`, `zero variables=`, `line`, `shift entries=`, `join variables=`, `lkp keys_log=`, `lookup v= d= mu= [shifted=0]` (a whole look-up),
`product_tree levels= [more=0]`, `rng columns=`, `scalars n=`, `fold group= [roots=0]`,
`loc group= trees=T1,T2 [opens=1]`, `iopp group= trees= rows= stages=k:h:log_n:mask_log_n,.. [roots=1 masks_ride=0 opens=1]`
(opens: at how many lists of drawn positions the group's main rows are opened in all, for the hiding budget),
`record table=`, `lin table= tree= instances= k_lo= rows_log= w=GROUP:CUBE u=GROUP:CUBE omega=GROUP:CUBE [one_time=0]`,
`link table= tree= u=GROUP:CUBE`.

## What is printed

- `setting:` the setting.
- `code:` each group's main code, then its mask code. On such a line:
  - `1 x 2^8 rows` is (rows folded side by side in one row) x (rows of the group);
  - `closeness count` names the bound from which the count comes; `count a` is that count (the number of challenges that can make far words look close; the closeness error is a / |E|); `closeness` is the largest error of a round that depends on it;
  - `eta` is the slack below the Johnson radius; `m` is the cited theorem's integer for it;
  - `list` bounds the number of codewords within the distance;
  - `positions` is the least number of drawn positions for the bits that the line names.
- `stage:` the derived stages of each proximity test.
- `hiding budget:` random coordinates against opened positions.
- `the plan's own verdict:` holds, NO with the first cause, or none (printed: `none: the file lays no round with an error`). The exit status follows this verdict and the hiding budgets.
- `round:` (with --rounds) every group of rounds, the largest error first.
- `accounting:` the accounting (`accounting.h`).
- `size:` each part in bytes: the expected size over the drawn positions, and the most that any positions can need; then by kind of message.
- `message:` (with --messages) every kind of message.
