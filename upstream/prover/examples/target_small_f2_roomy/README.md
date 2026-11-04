# target_small_f2_roomy: the small example of the default setting

This configuration is a small model with [synthetic
weights](../../../docs/glossary.md#synthetic-weights) under the [default
setting](../../../docs/glossary.md#default-setting). Every stored table has an out-of-domain sample
of its own (the line `record 1`). Its answers are published in the [key
file](../../../docs/glossary.md#key-file) and in each [leaf
file](../../../docs/glossary.md#leaf-file). The committer's code groups are tested up to the
[Johnson radius](../../../docs/glossary.md#johnson-radius), as a request's and a batch's are. The
verifier counts [proof of work](../../../docs/glossary.md#proof-of-work) in the rounds that draw
positions. [TARGET-KITS.md](../TARGET-KITS.md) has the figures at this size and at full size.

**Weight hiding, before you run it.** Every stored table has out-of-domain (OOD) answers (masked
sums of the weights). What they ask of an operator, above all of whoever keeps the seed, and what
the tests of hiding covered: [../TARGET-KITS.md](../TARGET-KITS.md#weight-hiding).

The published answers are sums of the weights, each hidden by a random pad alone, and what is
published cannot be taken back. If a committer's random source repeats over other weights, the two
commitments' public files alone give exact equations in the two sets of weights
([COIN-KEYS.md](../../../docs/format/COIN-KEYS.md), section 3).

`cp_prove` runs no verifier on the proofs that it writes: let no response leave you before `cp_verify` has accepted every proof of its batch ([duty 6](../../../docs/limits.md#the-conditions)).

To run it you need one H100 and the programs, built as
[docs/quickstart.md](../../../docs/quickstart.md), "3. GPU: a small model", says. From the
repository's top directory:

```sh
T=$(mktemp -d) && W="$T/small_example" && echo "W=$W" # a new folder, outside the tree, below one that only you can write to
prover/examples/run_target_example.sh target_small_f2_roomy "$W"
```

It commits to synthetic weights, proves one request and checks the proof. The script's screen ends
with `YES`. Above its status line it shows the verifier's line `commitment:`, the one closing line
that says on what the binding of the weights rests, and below it the line `proof of work:`. The
verifier's whole output is in `"$W"/cp_verify.log`.

*What the `YES` does not say.* It says nothing about any model that anyone serves, and it shows
nothing about weight hiding.

## What a YES under this configuration rests on

**On what it rests.** The truth of an accepted request and the binding of the weights both rest on
the [theorem on codes](../../../docs/glossary.md#theorem-on-codes), and at the default setting both
lean on counted proof of work in the rounds that draw positions. The written proof of the binding of
the weights, in a [technical report](../../../docs/glossary.md#technical-report) that is not public
and not refereed, does not cover the programs' commitment ID, whose hashed string
takes in more ([binding](../../../docs/glossary.md#binding)).

(That this counting of work is itself assumed for the hash chains; what that theorem is and how far
it has been checked; the three further steps that are argued on paper; what has and what has not
been compared; what the verifier's lines show; and what Lean's theorems say of the table:
../TARGET-KITS.md, "The table's OOD answers", "What it rests on"; in plain words below.)

## The same in plain words

- Everything rests on one theorem, and part of it on counted work besides. The truth of a request,
  and which weights a key file and its leaf files are bound to, both rest on the theorem on codes
  (Theorem 4.6 of the full version, ePrint 2025/2055), which we take as given: we have checked no
  proof of the theorem in its paper's own form ([what has been checked of it, and what has
  not](../../../docs/limits.md#what-the-theorem-assumes)). And 38 of the 11,857 rounds of a key
  file, a leaf file and a proof reach 128 bits only because the verifier counts the proof of work in
  them (20, 28 or 32 bits by the round's kind); that counting is assumed, not proved, for the
  programs' hash chains, and without it the weakest of those rounds has 96.23 bits (the verifier's
  own accounting at this parameter file).
- There is no margin. Were the count in the theorem on codes twice as large, every closeness round
  with less than a bit to spare would miss 128 bits: at full size, the verifier's own accounting of
  the programs counts 255 closeness rounds of a proof and 86 of a key file and a leaf file, all on
  the theorem on codes, taken as given, and 325 of these 341, so most of them, hold with less than a
  bit to spare (324 with less than a tenth of a bit; counted from the list of rounds in
  verifier/tests/known_answers/kit_rounds/target_full_f2.txt); at this small file the verifier's own
  accounting puts 63 of its 127 closeness rounds within a hundredth of a bit of the target with the
  work counted. The 38 rounds that hold only by their proof of work gain nothing from that theorem's
  count: without the work the weakest has 96.23 bits. This page gives no figure for what would be
  left if that theorem failed. If that theorem failed, the programs' accounting would rest on one
  other cited bound, itself unverified and not wholly independent of it; what would remain is, by
  the report's calculation on paper, fewer bits than the target (the size: row LIM-17.2 of
  docs/limits-all-rows.tsv), for the truth of an accepted request and for binding alike, and the
  report does not call it firm; the Lean verifier's own bound does not take the theorem on codes as
  given.
- No statement of the report is about a program. This parameter file has not been compared with the
  report's choice of parameters.
- Lean has a soundness theorem for its own verifier, `verify`, not for the C++ program that users
  run. The small example's files are out of scope: the Lean function `verify()` rejects them. (More
  limits of that theorem: ../TARGET-KITS.md, "What it rests on".)
- `integrations/check_proof.py` passes a proof under this configuration (row integrations.cp_verify,
  on the stored example).
