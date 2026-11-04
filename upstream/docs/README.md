# The detailed docs

New readers of FlashProve should start at the project's [README](../README.md) and the short pages
under [guide/](../guide/). This folder holds the long pages. On the pages of limits
([limits.md](limits.md), [limits-further.md](limits-further.md),
[limits-all-rows.tsv](limits-all-rows.tsv), [known-errors.md](known-errors.md)), unless an entry
says otherwise, its results were not seen on the released programs, and a fault of the code, told in
the present tense, holds of the released programs ([what was run on these programs, and what was
not](limits.md#what-was-run-on-these-programs)). Each page's head tells its rule in full; the head
of limits-all-rows.tsv has two exceptions. The other pages mark such a result, with † or in words.
[figures.tsv](figures.tsv) has a rule of its own, told below.

Pages cite rows of the table of limits by id. To read one, search for its id at the start of a line:

```
grep -P '^SO-15\.7\t' docs/limits-all-rows.tsv
```

- **Using the programs:** [quickstart.md](quickstart.md), [verifying.md](verifying.md),
  [operators-duties.md](operators-duties.md), [service.md](service.md), [faults.md](faults.md).
- **Limits and results:** [limits.md](limits.md), [limits-further.md](limits-further.md),
  [limits-all-rows.tsv](limits-all-rows.tsv), [known-errors.md](known-errors.md),
  [what-is-not-hidden.md](what-is-not-hidden.md),
  [how-hiding-is-checked.md](how-hiding-is-checked.md), [results.md](results.md),
  [figures.tsv](figures.tsv).
- **Reference:** [architecture.md](architecture.md), [protocol.md](protocol.md),
  [model.md](model.md), [format/](format/), [glossary.md](glossary.md),
  [sources-notes.tsv](sources-notes.tsv).

[figures.tsv](figures.tsv) has most of the figures that these pages print, and others, one row each.
An empty `programs` cell means the timed runs, on the released programs; any other cell says what
ran instead, and no time, rate or memory peak carries over from such a row. The rows of the runs
that timed the serving scripts' check stand apart: their runs were on the released programs, one of
them with one constant of the scripts set to a value that is not this repository's, and are none of
the timed runs. Rates are cut, not rounded, their means too; in the rows of the timed runs a memory
peak, disk in use and a largest size are rounded up, the lower end of a range of them cut, and
times, and the means of times and sizes, are rounded to the nearest. In these rows a standard
deviation (sd), a percentage and a multiple against stock vLLM are rounded to the nearest too. 1 GB
is 10^9 bytes, 1 MB 10^6, 1 MiB 2^20 and 1 GiB 2^30; the "80 GB" in the GPU's name is its maker's
figure. An id tells nothing of the weights: read the column `weights`. The timed runs measure speed
and are no test of weight hiding. "Other programs" and "other sources" in a cell:
[docs/glossary.md](glossary.md#other-programs).