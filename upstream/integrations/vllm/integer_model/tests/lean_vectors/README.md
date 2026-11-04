# Lean's definition of the model against this package, value by value

The Lean library's theorems ([their terms](../../../../../lean/TERMS.md)) speak of a function
`model`, defined in [`lean/IntegerModel/Model.lean`](../../../../../lean/IntegerModel/Model.lean).
Lean proves nothing about this Python package or the GPU programs, so whether `model` is the integer
model that they run rests on reading and on tests, not on a proof. The files here are such a test:
they run this package on made-up weights, write down its inputs and reported values (the vectors),
have compiled Lean code evaluate `model` on the same inputs, and compare the two, value by value.

From the repository's top directory:

```sh
python3 integrations/vllm/integer_model/tests/lean_vectors/run.py
```

It needs what the package's CPU tests need ([`../README.md`](../README.md)), which [the package's
install block](../../README.md#install-and-quick-start) brings (about 8 GB), and `lean` and `leanc`
on the PATH in the version that `lean/lean-toolchain` names. If they are
[elan](https://github.com/leanprover/elan)'s, elan downloads that version at the first call (about
540 MB, 2.8 GB on disk, into `~/.elan`). It needs no GPU, no Mathlib and no `lake`: the three files
of `lean/` that it compiles import only one another and Lean's core library.

The script writes only below a new temporary directory, `OUT` on this page, whose path it prints. It
removes nothing; you may delete `OUT` afterwards. `OUT` takes about 1.1 GB, and the run about 1.1 GB
of memory. Set `TMPDIR` to choose where; `OMP_NUM_THREADS` limits PyTorch's threads. `TMPDIR` and
every directory above it must belong to you or to root, and none may be open to writing by its group
or by others without the sticky bit; otherwise the run fails after some minutes and prints the
reason.

- `--vectors-only` stops when the vectors are made, needs no Lean, and prints one SHA-256 over all
  arrays, which is meant to be the same on every machine:
  `a24bfa7913cdef90021e42d7b5469e6f7a90f917e6fddfc673d3e7eddcdc0094`, without `--large`.
- `--large` adds "The large case" (1.8 GB more on disk and up to 4.6 GB of memory, which the script
  does not limit). A `ulimit -v` must lie well above that, because Lean's allocator reserves more
  than it uses: under 5.3 GB, Lean's program stopped with `out of memory`.

The script first prints the SHA-256 of the three files of `lean/` and says whether each is the file
of the runs on these sources, reported below; it goes on if not. Then it numbers its steps:

1. makes the vectors;
2. checks their sums;
3. builds the Lean program;
4. compares, and prints the closing count, which must be
   `103 cases, 614 runs, 30761 arrays, 66598727 values compared; 0 runs differ` (with `--large`:
   `105 cases, 616 runs, 30815 arrays, 67367156 values compared; 0 runs differ`);
5. tries the damaged copies.

The programs' output is kept in `OUT/*.txt`, the comparison's in `OUT/compare.txt`. Only the last
line `All steps ended as they must.`, with exit status 0, says that they did. A program that has not
ended after four hours is stopped; a program that it started, such as the C compiler, is not.

| File | What it is |
|---|---|
| `run.py` | the one command |
| `make_vectors.py` | chooses settings, weights and prompts, runs the package, writes the vectors and their damaged copies |
| `edge_cases_plugin.py` | a pytest plugin that writes the models and prompts of `../test_arithmetic_edges.py` as vectors |
| `CompareVectors.lean` | reads the vectors, evaluates Model.lean's definitions and compares |

## What the comparison shows, and what it does not

On 99 cases with 610 prompts, Lean's `model` gave the expected outcome: the package's token on 511,
undefined on 61, and defined on 38 where the package refuses. All 66,367,041 compared values were
equal, 41,134,476 of them in arrays that are neither all 0 nor repeats. The closing count adds the
library's five tables, whose 231,683 entries equal those that this package computes for the
full-size parameter file, and three settings files. At range conditions 3, 4, 5 and 7, `model` was
defined at 32,639, 32,640 and 32,767 and undefined at 32,768. All 146 damaged copies were caught,
each at the value named beforehand. With `--large`, all 768,423 values of "The large case" were
equal too, and so were six choices over the full vocabulary.

| the model's weights | cases | prompts | values compared |
|---|---|---|---|
| all 0, the embedding aside; one layer | 6 | 274 | 456,890 |
| the edge tests' (`../test_arithmetic_edges.py`) chosen weights; one or two layers | 33 | 33 | 18,393,045 |
| seeded; two to four layers, and once 48 | 43 | 286 | 47,465,681 |
| seeded, one output chosen, for the range conditions' upper ends; two layers | 17 | 17 | 51,425 |

The runs reported here were on machines without a GPU, with Lean 4.29.1, Python 3.13.16, PyTorch
2.9.1 and numpy 2.3.5, the versions of the top `requirements.txt`. A run took 10 to 34 minutes, and
18 to 50 with `--large` (3.7 to 4.6 GB in its largest process). These spans take in the runs on
these sources and runs on other sources of the package ([what these
are](../../OPERATING.md#what-has-been-run)) or of the three files of `lean/`. Under this package's
pins (PyTorch 2.13.0, numpy 2.4.6) only `--vectors-only` was run, and as an ordinary account, too,
only `--vectors-only`; both printed the SHA-256 above. Both ran on other sources of the package.

This is a test, not a proof. Its limits:

- **Reach.** A few hundred prompts on made-up weights at small sizes. Except in the large case, all
  cases use the tables of `prover/examples/target_small_f2`. No case has the settings and weights of
  the library's theorems: 48 layers at Llama 4 Maverick's sizes and the full vocabulary. The Lean
  side is compiled code, not an evaluation by Lean's kernel.
- **One source.** Both sides were written from [`docs/model.md`](../../../../../docs/model.md): an
  error there passes, and agreement does not show that the function is the intended Llama 4
  computation.
- **What is compared.** Only the 32 values that the package reports, and nothing that the GPU prover
  computes. In 64 of the 99 runs that the package refuses (61 + 38) only the outcome is compared.
  The all-0 models try only the norm and range conditions 1 and 8. In the edge tests' cases the next
  token is 0.
- **Not evaluated.** `weightsAreValid`, `modelNextToken` and, in the large case, `model` itself.
  `settingsAreValid` is evaluated on settings alone. `settingsInRange` passes on one case,
  `whole.48-layers`. Asked of each model's own layers by the comparing program, its clauses hold for
  68 of the 99 cases; the two cases `unlike.*` and the edge tests' 29 models of two layers lie
  outside, and so does the large case.
- **The ends of the ranges.** Between the package's limits and the model's (32,640 .. 32,767),
  values are compared in 8 runs of one token, against this package with its interval widened; in the
  other 30 such runs only "defined" is compared, and in 10 of them a token. At condition 5 the value
  stands in no reported array, so that it is reached rests on the maker's own arithmetic. The wide
  key and the up projections of a layer with experts stop at 32,640. Range condition 2 never fails,
  and its closest cases stop at -32,639 and 32,639.
- **What it would miss.** Another layer's `eps1` or `eps2` being read, or the wrong range condition
  being named. That a failure told at the wrong position is noticed rests on one run. No damaged
  copy has its error in a table's lowest key, the sizes that `wellFormed` tests, a prompt that
  `isPrompt` turns down, a run's list with a name left out, the large case, or the argmax over the
  full vocabulary.
- **Not in the suite.** No row of `SUITE.md` runs this comparison.

## Directory layout

`run.py` writes the vectors to `OUT/vectors`. Below that directory:

```text
cases/NAME/case.json            one model and one or more prompts
cases/NAME/*.bin                the model's arrays: weight matrices and the embedding
cases/NAME/runNNN/*.bin         what the package reported for prompt NNN
tables/ID/tables.json, *.bin    five lookup tables, shared by every case whose "tables" is ID
functions/NAME/function.json    one function on given inputs (with --large)
settings/NAME.json              settings, and what the library's settingsAreValid must say of them
damaged/NAME--KIND/             a copy of one case with one deliberate error
parameters/NAME/                the parameter files that the package read for the seeded cases
```

## Arrays

An array is a raw file without a header: little-endian two's complement integers in row-major order
(the last index varies fastest). Each array has an entry in a list named "arrays", with the keys
name, kind, symbol, shape, type (int16, int32 or int64), file (relative to the directory of the JSON
file) and sha256.

| kind | type | shape and order |
|---|---|---|
| weight | int16 | [inputs, outputs]: the input index comes first, as in docs/model.md, section 1, and in Model.lean. (The package's `matrix` returns the transpose.) |
| embedding | int64 | [V, d]. "rows_are_tokens" is "all" or, in the large case, the sorted list of the token ids whose rows are present. |
| intermediate | int64 | the first index is the position |
| table | int32 | see "Tables" |

Weight names: `L3.Wq`, `L3.Wk`, `L3.Wv`, `L3.Wo`, `L3.Wg`, `L3.Wu`, `L3.Wd`, `L3.Wr`, `L3.E5.Wgr`,
`L3.E5.Wur`, `L3.E5.Wdr` (layer 3, expert 5), `Wh`, `Emb`.

## case.json

- `format` (1), `name`, `what` (one sentence).
- `parameter_lines`: always `{"rcl": 1, "gcl": 1, "xl": 5}`, the only variant of the model that
  Model.lean defines (docs/model.md, section 10).
- `settings`: T, L, d, n_h, n_g, dh, f, fe, NE, V, cs, eps1, eps2, epsF, MF, KF, and `layers`, which
  gives each layer's kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, sh, rp, kr, kgR, kuR. The last
  three are 0 in a dense layer.
- `tables`: the ID of the tables' directory.
- `every_matrix_is_written`: true if the case holds the whole model.
- `arrays`: the model's arrays.
- `runs`: one entry for each prompt, with `prompt`, `expected`, `positions_compared` and `arrays`
  (the reported values).

## Names

The package reports 32 values (`IntegerModel.extend`, argument `record`). In each layer, in this
order:

| name | Model.lean |
|---|---|
| x | Layer.input |
| a_sigma, a_Rc, a_w, a | attention.norm[t].root, .reciprocal, .quotient, .out |
| qw, q, kw, k, vw, v | attention.projected[t].wideQuery, .query, .wideKey, .key, .wideValue, .value |
| o | attention.mixed (entry h dh + c belongs to entry c of head h) |
| x_mid | attention.out |
| m_sigma, m_Rc, m_w, m | feedForward[t].norm.root, .reciprocal, .quotient, .out |
| lr, e_star, sc, xr (layers with experts) | feedForward[t].routed.logits, .chosen, .scale, .input |
| gw, key, s, u | feedForward[t].shared.wideGate, .key, .activation, .up; in a layer with experts these fe columns are followed by the fe columns of routed.units |

After the layers: x_last (Trace.last); aF_sigma, aF_Rc, aF_w, aF (finalNorm[t]); lam (logits);
first_choice (choices).

Two names are easy to confuse. `k` is attention's key, and `key` is the key at which the SiLU table
is read. `s` is the SiLU table's output, not an attention score.

A listed name that the program does not know counts as a difference, and so does a
run that compares any position and does not list all 32.

## expected

| kind | meaning | Model.lean's `model` |
|---|---|---|
| token | the package's greedy choice at the prompt's last position is `token` | some token |
| none | the package refuses the prompt, and the model is undefined | none |
| package refuses | the package refuses the prompt, and `cp_prove` makes no proof of it, but every value is inside the column "defined" of docs/model.md, section 9 | some |

For the last two kinds, `position` is the first failing position, and `condition` is the range
condition that the package evaluates first there. The package tests condition 8 before condition 1,
and Model.lean has no separate condition 8, so where Model.lean's condition 1 fails the package
often says 8.

A "none" run also has `undefined_from`, the first position at which the model is undefined. It can
lie behind `position`, because the package's intervals are narrower. The Lean program asks that
`model` is defined on the tokens before that position and undefined with it, and that `wellFormed`
and `isPrompt` hold, so that the "none" comes from a range condition at that place.

Some "package refuses" runs have `token_with_the_interval_widened`: the package's result after its
interval -32,768 .. 32,639 is widened to -32,768 .. 32,767. No program proves such a token. Where
`values_with_the_interval_widened` is true, the run's arrays are what the package reported under
that widened interval, and all positions are compared.

`positions_compared`: but for these, the arrays of a failing run hold only the positions before the
first failing one. From that position on the package replaces each offending value by 0 and goes on,
so its values mean nothing.

How the kind of a failing run is found, in the edge tests' cases and at the upper ends: the package
runs again with the widened interval. If it gives a token, the kind is "package refuses". If it
still refuses at a condition other than 8, the kind is "none". Where condition 8 fails, the kind is
"none" if the row's squares plus eps reach 2^78.

## Tables

`cosine` and `sine`: [T, dh / 2]. `exponential`: [u0], lowest key 0. (The package appends a 0 for
all keys from u0 on; that entry is not written.) `silu`: [65536], lowest key -32,768. `sigmoid`:
[32768], lowest key -16,384. `OUT/vectors/tables/ID/tables.json` also gives the parameters from
which the tables were computed and the four digests in the parameter file.

`"compare_with_the_librarys_tables": true` marks the tables of the full-size parameter file. The
Lean program compares them with the library's `lookupTables`.

## The large case

With `--large`, `maverick-sizes.2-layers.3-tokens` has a dense layer and a layer with 128 experts at
Llama 4 Maverick's sizes (d = 5,120; 40 and 8 heads of 128; f = 16,384; T = 1,024), with the
vocabulary cut to 4,096. Only the matrices that the package asked for are written: of the routed
experts, those that a position chose. `model` would answer `none`, because it asks for all 128
experts, so the Lean program evaluates `attention`, `feedForward`, `rmsNorm`, `product` and
`firstArgmax` one after the other and puts them together as Model.lean's `trace` does
(`compareByFunction`).

`functions/first_argmax.V202048` has six rows of 202,048 logits (`lam`) and the package's choice for
each (`first_choice`), with ties at both ends. Model.lean's `firstArgmax` recurses as deep as the
row is long and overflows a stack of 8 MB, so `run.py` allows the Lean program 1 GB of stack.

## Damaged copies

Each directory `damaged/NAME--KIND/` is laid out like the vectors' directory and holds one case with
one deliberate error. The comparison must fail on every one. `DAMAGE.json` says what was changed
and, under `a_comparison_must_fail_at`, the first value that must differ; `run.py` checks that too.

- One number is off by one: a weight, an embedding or table entry, a setting, an expected value or
  token.
- Two settings are exchanged, mostly in the cases `unlike.*`, in which no two such settings are
  equal.
- The expected outcome is another: a run on which the model is defined expects "none", and a run on
  which it is undefined expects "defined".
- The model is said to be undefined from another position: 4 or 6, where it is 5.
- Every listed array has lost its last position, and `positions_compared` is one less.
- One entry of each of the five full-size tables is off by one, so that the comparison with the
  library's tables must fail.
- The full-size settings have kd = 20 in one layer and are still listed as valid, so that
  `settingsAreValid` must disagree.
- `stale-sum`: a file is changed and its SHA-256 in case.json is not. In all other copies the sums
  follow the change, so that the values must reveal it.

A weight that is off by one does not always change a reported value, because a rounding shift
absorbs most single steps. So for such a weight, in each of three matrices, the maker tries the
places from the middle on until a reported value shows the change. For the sine table it takes the
first sine of position 1 whose change shows. `the_package_on_the_damaged_input_differs_in` lists the
reported values that a changed input moves.

## What the package cannot produce

- A model without routed experts (NE = 0): it refuses such a parameter file. Dense layers are tested
  as the even layers.
- A failure of range condition 2: it accepts no parameter file under which a norm's quotient leaves
  its range (`../test_arithmetic_edges.py`, `test_norm_at_the_ends_of_its_ranges`: argued there, and
  asserted for the parameters tried).
- A separate eps1 and eps2 for each layer.
