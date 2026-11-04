# numbers/: parameters, round errors, sizes

The prover, the verifier and planning use one set of routines. From a setting (field, widths, target, bits of proof of work) and the shapes of the blocks, these routines derive every number that the protocol needs and every number that is printed about it.

| file | what it holds |
|---|---|
| `real.h` | Exact integers, numbers with directed rounding, and intervals. Every ceiling and every comparison with the target is an exact integer comparison at the safe end of an interval of 192 bits. `double` appears only in what is printed. |
| `setting.h` | Everything that is given: the field, widths in bytes, the target, and the bits of proof of work by kind of error and by who pays for them (each request, each batch, or the committer once). |
| `codes.h` | From a code and a distance it derives the list size, the count behind the closeness error with its source, and the number of positions. Below half the code's distance the list size is 1. |
| `plan.h` | All rounds with their error and work, and all messages with their bytes, in protocol order. There are two types: the planners write a `Plan`; `finish()` makes a `Planned` of it, which all other code reads. Also the `Cursor` with which the hash chain steps through a plan. |
| `round_names.h` | The names of the kinds of rounds. |
| `blocks.h` | One function for each block, which appends the block's rounds and messages to a `Plan`, and the list of the functions that may add a round (`MayLay`). Also groups and how their slack is sized, trees and openings, and the stages of a proximity test. |
| `accounting.h` | The one routine that prints the accounting: the largest round error, the verdict, the hash calls, the compiled bound, both levels (with proof of work counted and without it), and, last, the stricter readings. In its lines on what the figures rest on, each source is preceded by the number of rounds that rest on it. Where no round rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), it says so (the lines `footing:` of `cp_frame` say it file by file). |
| `cp_plan.cpp` | A program that plans a protocol from a text file (`settings/README.md`). |
| `tests/` | `numbers_test`: arithmetic; derived numbers against exact comparisons made in another way and against known values; counts of rounds and scalars; stage identities; rejections of what is outside a formula's range; the committed table's blocks; `Cursor`; the accounting. `holds_test`: a plan from derived parameters (for a segment, its closing, `plan_iopp`'s proximity test) must hold at the error target in both fields that the test tries (Goldilocks, the programs' field, and 2^61 - 1), with and without proof of work. The test has no negative control: no run derives one position fewer per draw, one step less slack, or, in the approximate range check, one combination fewer. Positions are checked elsewhere: `numbers_test` checks, for a fixed set of code sizes, rates and bit counts, at the distance with 1 - delta = (n + k + 1) / 2n and at distances of the Johnson rule, that `positions()` gives the fewest positions that reach the bits asked for when the draws are a round's only error. `planners_read_once_test`: each planner that the test calls reads each of its sizes once (some planners, such as plan_iopp, it does not call). The test detects a second read only while the compiler lets a replaced operator new write globals (g++ 13.3 and clang++ do). Under a compiler that assumes otherwise it would pass and detect nothing, and nothing in this repository would notice that. `cp_plan_files.py`: `cp_plan`'s exit status and the lines on its verdict, on eleven files. |

## How a number of positions comes about

1. The code enters the cited bounds with rho = k / n (`codes.h` says which k; it need not be the code's dimension), at the dimension that the encoder really uses.
2. Of a group's main code and mask code, the one with the larger rate determines the distance.
3. Each group goes up to the Johnson radius less a slack (`Radius` in `codes.h`). The slack is eta = sqrt(rho) / m, where m is the largest integer for which every round whose error depends on the distance, except the draws of positions, meets the target with its work. These rounds are FOLD's rounds (closeness of main and mask rows, list term), BASE's first round, the samples' seeds and the batching.
4. The number of positions t is the least integer with (whole error of the round) x 2^-work <= 2^-target. The error is (1 - delta)^t, plus, inside a proximity test, the batching term.
5. A row gets as many random coordinates as hide the positions at which it is opened. In the later stages of a proximity test this number is found by iteration, since the rate depends on it.

## What is not here

- Which oracles share a Merkle tree: the caller passes roots and trees.
- The shapes of a request's sum-checks: the files in `settings/` use stand-ins and say so.
- Running times.
