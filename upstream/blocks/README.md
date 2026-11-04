# blocks/: prover and verifier, one file for each block of the protocol

This folder holds plain C++17 headers, which build without CUDA. Three headers for GPU code (`hash/work_search.cuh`, `hash/lanes.cuh`, `chain/chain_lanes.cuh`) are beside them; no target of the Makefile builds these, and this repository has no GPU test of them alone. The folder holds the foundations (field, hash, hash chain, code, tree, numbers) and, for each block of the protocol, one routine that plans it, one prover routine and one verifier routine. With the CPU engine the prover routines form the library's reference prover (routines, with their tests: this repository has no program that proves a model's reply on a CPU); with a GPU engine the same routines form the GPU prover. The order of messages is written once for both provers. Between them the programs cp_commit, cp_prove and cp_verify run these routines for the files' frame: configuration and plans, hash chains, trees and openings, the order of a request's segments and messages, the proximity tests, the test on the committed table, the link between two tables and the key file's proof of ranges. Inside a request's sub-proofs cp_prove and cp_verify run implementations of their own of the masked sum-check, the look-ups and the range check, so this folder's tests say nothing about those; from this folder the sub-proofs take the `Plan` into which their rounds are entered (numbers/plan.h), the chain's channel and, for the masked sum-check, iop/hp.h; the lists of the sub-proofs' own rounds stand outside it, in ../prover/prover/plan_rounds.h, ../prover/head/plan_rounds.h and ../prover/attn/plan_rounds.h.

## Layout

| folder | what | 
|---|---|
| `base.h` | integer names, `BLK_HD`, `BLK_REQUIRE`, the settings |
| `field/` | F, E = F[X]/(X^3 - c), canonical parsing, one attempt at a draw; equality vectors, small polynomials |
| `code/` | a code = (length, message length, length of the random tail); encoding; a generator row as a product form |
| `hash/` | BLAKE2b as specified, on a label (number, tree, index) and the bytes, with a start value as salt and personalization: leaf, vertex, a round's calls, digests of parameters |
| `chain/` | the Fiat-Shamir hash chain with proof of work (`chain.h`); public inputs and the first query, which holds them (`public.h`); the channel through which blocks send and receive, with typed rounds (`rounds.h`) |
| `tree/` | the Merkle tree of oracle messages: perfect binary, one kind of leaf (a column and its salt); openings at a list of positions |
| `numbers/` | parameters, round errors, sizes; the Plan of a run and `plan_<block>` for every block |
| `iop/` | `message.h` (bytes of a proof), `tensor.h` (structured coefficient vectors), `relation.h` (oracle messages, groups, rows with public targets, masks), `context.h` (what a block's prover or verifier holds), and the blocks: `send.h bat.h split.h hp.h psc.h line.h fold.h cs.h base_case.h iopp.h req.h lkp.h rng.h batt.h foldeq.h lin.h link.h member.h` |
| `engine/` | the heavy steps of a prover behind one interface (`engine.h`), on the CPU (`engine_cpu.h`) |
| `coins/` | the prover's private randomness from a seed, stream by stream |
| `frame/` | the files around a proof: format version, parameter files, what is derived from them |
| `toy/` | whole runs at toy size, prover against verifier: one product (`product.h`), a chain of links (`segments.h`) |
| `<folder>/tests/`, `tests/` | tests of a folder; of whole runs |

## Rules of the folder

1. One namespace, `blk`. Includes are written from this folder: `#include "field/field.h"`; build with `-I blocks`.
2. Settings are numbers given to the compiler: `-DBLK_FIELD=2` (Goldilocks, X^3 - 2, plain Reed-Solomon; the only field; it must be
   given); `-DBLK_HASH=1` (BLAKE2b; the default and the only one). No width of a field
   element, a hash block or a digest is written as a number outside `field/` and `hash/`. Rates, lengths, numbers of
   positions and work bits are data.
3. Indices. Bit 0 of an index is the least significant and belongs to the first variable. A table of 2^k rows of length l
   is the vector with entry (row b, place c) at b + 2^k c: row variables first. A fold at g replaces the rows 2b, 2b + 1
   by (1 - g) row 2b + g row 2b + 1.
4. A block is three routines in one header: `plan_x` (in `numbers/blocks.h`) appends the block's messages and rounds to
   the Plan, with the kind of each verifier message, its error and its proof of work; `x_prove` and `x_verify` go through those
   entries. Each call that closes a round names the round; the channel ends the process if that is not the next planned round or if
   the message does not have the planned size.
5. Everything that a prover sends before a draw goes through one call that writes it into the proof and absorbs it; the
   verifier's matching call reads it, rejects a non-canonical element, and absorbs it. Only answers to the verifier's queries
   (columns, salts, siblings) are in the proof without being absorbed.
6. Where steps must come in a certain order, another order is made hard to write (a caller who sets out to get round the types can: "If you write a verifier on this library", below):
   - a verifier message holds draws of one kind: algebraic challenges; or the seeds of samples; or the one element that
     joins a hiding polynomial to a claim, which is drawn only after that polynomial's sum was the last thing sent; or
     positions (inside a proximity test followed by one batching element);
   - after a round of seeds the next prover message begins with the answers;
   - no verifier message of any kind lies between a seed and its answers, so positions are never drawn while a sample is
     open; an oracle message can join a group, and so be tested, only once its sample is answered;
   - a mask entry is a value that can only be moved: it is handed out once, after the sample of its oracle message has
     been answered, and is used up by the row that it masks.
7. Two ways to fail. `BLK_REQUIRE` ends the process: for a mistake of the program (a block out of order, a mask used
   twice). Whatever depends on the bytes of a proof makes a verifier routine return false: no exceptions, no exit. Once
   a read has failed, every later read fails too.
8. Which oracle messages share a Merkle tree is a rule of `iop/context.h`, not of a block: blocks deal in oracle
   messages and positions. Oracle messages that share a tree are opened together, always.
9. The prover's private randomness comes through `Coins` (`engine/table.h`); a program supplies its generator.
10. Public inputs (`chain/public.h`). A public input that is in no query of the hash chain changes no challenge,
    so a prover who can influence it picks it after the challenges. So a channel is started from a `Binder` only, which
    writes parameters and statement into the first query; what it hands back, a `Public<T>`, has no getter: only the channel
    of that chain reads it. A routine that runs on a channel takes, beside its context, `Public<..>` values, what other
    routines have made on the same channel, and the prover's private data: no plain number, struct, table or callback.

## If you write a verifier on this library: a known limit

**In test programs that call these routines directly, false claims have been accepted.** The six ways that were found for ordinary code are closed, by one reading of the sources; another holds some closed only in part ([docs/limits.md](../docs/limits.md#false-claims-in-test-programs), "The library under the programs can be misused"). Ways that are taken on purpose and break the library's stated contract are not: three are named below. The library is not offered as an interface: a calling program that hands its routines configurations or tables which do not belong together may still find a way to a false accept. `chain/closed.h` states the contract. In short:

- *Held:* plan, run and a table's reader name one set of four numbers (two radii, two m), or the process ends. A later test on a committed table, and the first side of a link, read a `TestedTable` (`iop/foldeq.h`). Only the routines that check a whole key or entry file hand one out, nobody can alter it, and it carries the setting (target, work bits) under which its test passed and the public values under which its key was checked (the ranges of the entries, the matrix's rows). An untested reader, other distances or another setting end the process where the table's columns are read; `check_proof()` and `check_entry()` take nothing else by type, and end the process if the key's public values are not their own configuration's.
- *Closed:* on weaker forms of these checks, ordinary test code got false claims accepted† (on another matrix; in the last two ways on a matrix out of its ranges), at toy size, in six ways, the first four at targets of 8 to 32 bits: one public member of a checked reader assigned; a reader made by hand; a second keeper of the same table, whose reader named the later run's distances; a key tested at a lower target than the batch that relied on it (in a test harness at toy size the batch's error bound was then untrue by some 4.6 bits); a key checked under laxer ranges than the proof's check stated, by two configurations made from the same public files (a target of 128 bits with proof of work counted); the same two configurations, swapped by the caller's own routine for the links while `check_proof()` ran. To see that each is closed: `tests/must_not_build/` (eight sources on a tested table), the cases in `iop/tests/lin_test.cpp` and `iop/tests/link_test.cpp` that must end the process, and `frame/tests/tested_table_test.cpp`.
- *Not held:* anything across processes or files: a program that checks key, entry and proof in separate processes works on tables read without their proofs and gets results of the type `IfTested`, which has no plain yes; that all its processes passed, on the same bytes, is its own logic. What the calling program supplies is believed: its routine that reads the shapes off the files, its callbacks, a reader class of its own, plans from sizes of its own choice. So are the two forms in which `check_proof()` takes a request, the item `request` and the request file's words: the library compares the two with nothing, so a caller that checks a relation on tokens must make both from those same tokens, and soundness for the tokens leans on the second (`docs/format/10-proof-file.md`, R14.3). And the four-argument `published_id()` takes the anchor's name and its record's seeds from its caller and checks neither (`frame/heads.h`). And a caller who sets out to defeat the types: what is claimed is claimed of code whose behaviour the standard defines, naming no private member, declaring nothing in `blk` or `std`, and nothing at global scope before these headers, operator new kept, no `fence.h` macro set; outside that rule C++ has casts, macros, unions, references that outlive their objects, and no access check on a name that is the argument of an explicit template specialisation (`numbers/plan.h` has the list). The programs, their files and the format say "the record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say "the record's seeds" for its OOD seeds ([glossary](../docs/glossary.md#record)).
- *Shown:* a test program took the last of these ways to a plan's private list of rounds. It lowered the number of combinations in the test that committed cells are small from 128 to 1, left that round's stated error at 2^-128, and got a false claim (a cell of 1000 under a bound of 3) accepted by the unmodified `run_verifier()` at the full target of 128 bits (toy size)†. This way takes a source file that is written to defeat the type; how far the programs are from it is told in the next point. A second way of that kind (an operator template declared before the headers) yields a wrong plan: a test program run in the tester's own process at the same target, at toy size on made-up data, had a false claim accepted in 8 of 48 tries, and in 0 of 48 without the declaration†. Nothing of `cp_verify` or of any file a provider sends is touched by it. A third (a planner's flag changed between two reads, by a replaced `operator new` or a second thread) led to a false acceptance at 128 bits too† (`docs/format/toys/3-key-and-entry.md`, part 9, section 14, tells the case) and is closed in part only: for the fourteen listed planners, which copy their sizes first (`numbers/plan.h`, READ ONCE), and not for `plan_iopp`, the frames' planners, the routines that run a proof or the plan's scope. These trials were made with test programs that do not ship.
- *How a program keeps the contract:* it hands its checkers one `Configured` and nothing else that is public, as `cp_verify` does: a stage calls `configure()` once, makes no keeper, and gets its tables on that `Configured` alone. No file, argument or setting has brought `cp_verify` to any of these ways in the rows of [SUITE.md](../SUITE.md) under "Refusals of altered, malformed and extreme files"; that they cannot reach it rests on reading its source. `cp_verify` itself does not keep one clause of the rule under *Not held*, read to the letter: before its first header of `blocks/`, its own headers declare names at global scope (the types `u32`, `u64`, `i64` and `u8`, constants, the namespace `f2b` and several static functions), and it defines two macros, `HD` and `DIGEST_WORDS`, which no file of `blocks/` uses, by a text search. That none of these is a declaration that the library's headers could take in place of their own rests on a reading of the code, not on a proof and not on a run.

## Build and test

    B=$(mktemp -d)
    make -C blocks OUT=$B -j 4 test
    make -C blocks OUT=$B toy
    make -C blocks OUT=$B SETTINGS=f2h1 test

`mktemp -d` makes a new directory that only you can write. With `OUT` alone, `make` does what `all` does: `test`, `toy` and `programs`. `test` builds and runs every `<folder>/tests/*.cpp`, every `tests/*.cpp` and every `tests/hiding/*.cpp` (89 tests; one of these tests, `tests/hiding/key_exact_test`, runs with its argument `long`: by the Makefile's comment it takes about nine minutes and 5.4 GB). The comment at the top of the Makefile says how to run chosen tests only (`TESTS=`).

The Makefile runs the programs that are in OUT, so no other user may be able to put a file there. It creates an OUT that does not exist, with access for you only, and it refuses an OUT that exists unless it is a directory (not a symbolic link) that you own and that neither its group nor other users can write. It goes by type, owner and mode alone, so it misses a default access control list on OUT (reasoned from the code, not tried), and it examines neither what already lies in OUT nor the directories above it. So use a new directory from `mktemp -d` for each checkout, or one for which every directory above it is owned by you or by root and is not writable by its group or by others unless it has the sticky bit, as `/tmp` has. Choose one outside the whole tree: the Makefile refuses an OUT inside `blocks/`, but not one elsewhere in the tree. Use a path of letters, digits and `/ . _ -` alone: make splits names at blanks, and the rules hand OUT to the shell unquoted, so a blank, a quotation mark or a sign such as `;`, `&`, `$` or `*` is refused, breaks the build, or makes make write, and run what it built, in folders that the check never looked at.

Run none of this under `sudo` or as root ([why](../docs/limits.md#running-the-programs-safely)). `sudo make` on an `OUT` that your own account made, as by the `mktemp -d` above, is refused (`OUT=… cannot be used`, exit status 2, nothing built): a known fault. If you go on under `sudo` all the same, all that is built belongs to root, and your own account cannot read or remove it; an `OUT` that does not exist yet below a directory of your own is taken too, but whoever can act as your account can then put something else in its place while root builds there and runs what it built. This was seen as root†; such a swap and an ordinary account's `sudo` were not tried.

The rules start the compiler with your environment: its temporary files go to your `TMPDIR` (to `/tmp` if it is not set), which the Makefile does not examine either, and with `DEPENDENCIES_OUTPUT` or `SUNPRO_DEPENDENCIES` set, g++ appends to the file that the variable names, outside OUT. Unset these two variables before a build, and give `TMPDIR` a directory that only you can write to, or leave it unset.

A test is a `main()` that prints one line and returns 0.

† Not on the released programs. What it ran on, and how far that speaks for them: [what was run on these programs, and what was not](../docs/limits.md#what-was-run-on-these-programs).
