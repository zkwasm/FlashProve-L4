# scales: fitting a model's quantization scales

The proofs in this repository are about an integer (fixed-point) model. The tools in this directory
take a floating-point model in GGUF files and choose how coarsely the integer model rounds each
activation. The result is a *configuration directory*: three public files that `cp_commit` (the
program that commits to the weights), the prover and the verifier read.

| file | what it holds | written by |
|---|---|---|
| `policy.txt` | the parameter file: the model's shape, the proof's parameters, the acceptance policy, and the scales that all layers share | `words_to_policy.py`, from a base file that you choose |
| `KL.bin` | the quantization scales of each layer | `words_to_policy.py` |
| `layout.txt` | the layout file: where each array of a proof is stored | `prover/tools/pack_layout.py`, from the other two |

Only the provider, who holds the weights, runs these tools. Clients get the three files from the
provider. The verifier checks the values in them, not how they were chosen. Every byte of the three
files enters the commitment ID, so other files mean another commitment.

"Try it" walks the whole route on a small synthetic model. "A real model" is the same route for your
own weight files. The sections after it are for looking things up. The last one is reference: the
rule of the fit, in full. How the worst-case check computes its bounds is told on a page of its own,
[worst-case-bounds.md](worst-case-bounds.md).

## The programs

| step | program | reads | writes |
|---|---|---|---|
| 1 | `make_calibration_set.py` | a text corpus, two tokenizer files | a calibration set |
| 2 | `bin/weight_sums` ([../weight_sums/](../weight_sums/README.md)) | the shards | `wsums.txt`; a typed line adds `fold.txt`, which tells `cp_commit` how the sums were made |
| 3 | `calibrate.py` | the shards, `wsums.txt`, the calibration set | `maxima.json`, `words.json` (the fitted scales) |
| 4 | `words_to_policy.py`, "the converter" | `words.json`, `wsums.txt`, a base parameter file | `policy.txt`, `KL.bin` |
| 5 | `bin/cp_verify -plan`, `prover/tools/pack_layout.py` | `policy.txt`, `KL.bin` | `layout.txt`, and the verifier's verdict on the three files |
| 6 | `cp_commit` | the configuration directory, the shards, `wsums.txt`, `fold.txt` | the commitment, and the table of its worst-case check |
| after status 7 in step 6 | `worst_case.py` | that table, `words.json` | a coarser `words.json`, for step 4 |

`fit_scales.py` is step 3's fit without a model file: it reads three JSON files.
`policy_tables.py --check` checks the 32 lines of `policy.txt` with the digests of the four public
lookup tables; the converter computes them with the same file. `gguf_weights.py` and
`float_forward.py` are modules of `calibrate.py`.

Each Python program's source file is the reference for its arguments, files and exit statuses: start
at the docstring at its top. No program here overwrites a file: if an output path is taken, it stops
and changes nothing.

## Terms

- *Exponent* E: the integer n represents n * 2^-E.
- *Cell*: a rounded activation, clamped to -2,047 .. 2,047 (12 bits with the sign).
- *Shift* k: a matrix product is divided by 2^k and rounded. Most scales are shifts.
- *Site*: a named point of the forward pass, such as `v`, the output of a value projection.
- *Limit*: the size (mostly 16 bits with the sign) that a site's value may have before the clamp.
  Beyond it an honest prover produces no proof.
- *Worst-case check*, *bits short*: before it commits, `cp_commit` computes for each site a bound on
  that value, from the weights and the scales alone, meant to hold for every prompt. The site is b
  bits short for the least b >= 0 with bound <= 2^b * limit, and it fits if b is 0.
- *Standard parameter form*: the verifier's conditions on the three files. If one is missed, no
  verification under these files can end with YES.
- *Conversion*: how stored weights become the committed integers
  ([../weight_sums/README.md](../weight_sums/README.md)).
- *Sums file* (`wsums.txt`): one line for each tensor, with its exponent and a checksum of its
  integers.
- *Shard*: one GGUF file of a model.
- *Lookup key*: the integer that selects an entry of a public table, such as the SiLU table.
- *Calibration set*: token sequences of equal length, the *windows*, each run as a separate prompt.

## Try it

This example goes from weight files to a commitment, on a small synthetic model (4 layers, 16
experts, 0.5 GB). Run everything from the top of the repository, on x86-64 Linux with the SHA
instructions, g++ 13 or later and Python 3.13. Only step 6 needs a GPU: one H100 and the built GPU
programs ([../prover/docs/build.md](../prover/docs/build.md)). The block works in a new temporary
directory, where the Python packages take about 7 GB (set `TMPDIR` first if `/tmp` is small or in
memory).

    W=$(mktemp -d) && echo "W=$W"
    python3 -m venv "${W:?first type the line with mktemp}/venv" && . "$W/venv/bin/activate" &&
    pip install --require-hashes -r requirements.txt &&
    EXAMPLE=prover/examples/target_small_file_f2 &&
    OUT="${W:?first type the line with mktemp}/build" prover/build.sh make_model &&
    "$W/build/make_model" "$W/model" $(cat "$EXAMPLE/model.txt") &&
    python3 -P -c "import sys, numpy as np; np.save(sys.argv[1], np.random.default_rng(2026).integers(0, 2048, (3, 40)).astype('int32'))" "$W/tokens.npy" &&
    GGUF=$W/model BASE=$EXAMPLE/policy.txt TOKENS=$W/tokens.npy ERES=11 BUILD=prover/build/hc WORDS=$W/fit/words.json N=1

All variables are set, and `make_model` has written two shards and, beside them, the `wsums.txt`
and `fold.txt` of step 2. Of "A real model" skip the first block, the list of variables and steps 1
and 2, and type steps 3 to 6.

What to expect, from runs of tools and programs whose sources differ from this repository's: step 3 takes about
ten seconds and under 1.5 GB of memory on a CPU. Step 4 prints `w2p moved 1`: only the line `s`
differs from the base file's. Step 5 prints the lines of the verifier's build and one line of
`pack_layout.py`, and its last call the one line `PLAN ONLY: no proof was checked`. The `KL.bin` that you get is not the one in the example's directory, which was fitted on
other tokens, and that directory's line `s` was calculated for Llama 4 Maverick
([../THIRD_PARTY.md](../THIRD_PARTY.md), "What depends on Meta's models").

## A real model

Supported: GGUF version 3, architecture `llama4`, width 5,120, head dimension 128, experts in every
second layer with one routed expert per token, norm epsilon 1e-5, rotary base 500,000, as in Llama 4
Maverick; tensors stored as F32, F16, BF16, Q8_0, Q4_K, Q5_K or Q6_K, the norms' gains (`attn_norm`,
`ffn_norm`, `output_norm`) as F32 alone (`bin/weight_sums` and `cp_commit` reject a norm of another
type). The programs take a file in which a tensor that they read is stored in 8 or 16 bits (Q8_0,
F16, BF16) without a warning, so know how each tensor of your files is stored: [what such files mean
for weight hiding](../docs/limits.md#files-of-8-or-16-bits).

For Maverick's public weights `examples/maverick/run.sh` runs steps 2 to 6 on the set that "The
calibration set" describes, which you make first, and under the public seed file
`examples/maverick/public-seed.bin`, so that two people's runs can give the same commitment ID
([../examples/maverick/README.md](../examples/maverick/README.md) says when); such a commitment
hides nothing of the weights. By the steps below `cp_commit` draws a secret seed, and the ID is
yours alone.

Compare each downloaded file's SHA-256 with its publisher's before any program reads it: a malformed
file can keep a program running for ever or make it request a very large amount of memory. If
`calibrate.py` says that a file "cannot be read as a GGUF file", the file is damaged or is not a
GGUF file that the package `gguf` can read: compare its SHA-256 again, and fetch the file again if
the two differ. Work from the top of the repository, on the platform of "Try it". First let this
shell create only what you alone can read (`umask 077`: it holds for what you make in this shell,
a copy of the weight files too, unless the command keeps the source's modes, as `cp -a` and
`rsync -a` do), make the working directory `W` (mode 700) and set up Python in it:

    umask 077
    W=$(mktemp -d ~/scales.XXXXXX) && echo "W=$W"
    python3 -m venv "${W:?first type the line with mktemp}/venv" && . "$W/venv/bin/activate" &&
    pip install --require-hashes -r requirements.txt

`W` must lie on a local disk that survives a restart, not on a file system in memory: step 6 makes
the commitment there, with its secret directory
([../docs/operators-duties.md](../docs/operators-duties.md)). If your home directory is no such
place, write another directory of your own for `~`. Then set:

- `GGUF`: the weights directory, one that only you can write to and, unless the weights are public,
  only you can read. If yours is not such a directory, set `GGUF` to it and type the line under
  "Where a step stops", "Step 2 is repeated": it makes one, with a copy of the shards. Yours stays
  as it was, open to whoever could read it. The weights directory holds the shards themselves, not
  links (`cp_commit` refuses a symbolic or hard link that it sees), and no other `.gguf` file. Step
  2 writes two small files into it.
- `BASE`: a parameter file of the model's shape: `prover/examples/target_full_f2/policy.txt` for
  Maverick's. No tool here writes one for another shape
  ([../prover/examples/TARGET-KITS.md](../prover/examples/TARGET-KITS.md) says how the examples'
  were made; "The recorded sites and the scales" says where their scales come from).
- `TOKENS=$W/calibration_set.npy`: the calibration set, which step 1 makes.
- `ERES=11`: e_res, at the tools' default ("The residual stream's exponent, e_res"). For Llama 4
  Maverick's public weights set 10, as `examples/maverick/run.sh` does.
- `BUILD=prover/build/hc`: the default build's programs, the only ones that take a model's files.
- `WORDS=$W/fit/words.json` and `N=1`: the scales, and the number of the round of steps 4 to 6.
  Every file and directory of a round has N in its name.

1. Make the calibration set ("The calibration set").
2. Make the sums file. The last line writes `fold.txt`, which states the same e_res. The second and
   the last line each create their file and never write to one that exists: if a file or a link is
   already there under that name, they stop ("Where a step stops", "Step 2 is repeated"). The last
   one then prints `FileExistsError` and ends with status 1.

       weight_sums/build.sh && (umask 077 &&
       bin/weight_sums --eres "${ERES:?}" --out "${GGUF:?}/wsums.txt" "$GGUF"/*.gguf &&
       python3 -I -c 'import os, sys; sys.tracebacklimit = 0; os.write(os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), b"eres %d\nqkrows asis\n" % int(sys.argv[2]))' "$GGUF/fold.txt" "$ERES")

3. Run the model over the calibration set and fit the scales. `--device cuda` computes on a GPU.
   Status 1: both files are written, but a scale is far out of range, and step 4 will reject them.
   In the line below, `--out` names a directory that does not exist yet. Under `sudo`,
   `calibrate.py` refuses at its start (status 2) an `--out` directory that your own account made
   beforehand. One that it makes itself belongs to root with the two files in it, and steps 4 to 6
   need root too. This is read in the code, not tried.

       python3 scales/calibrate.py --gguf "${GGUF:?}" --wsums "$GGUF/wsums.txt" --tokens "${TOKENS:?}" --out "${W:?}/fit" --eres "${ERES:?}"

4. Write `policy.txt` and `KL.bin` into a new configuration directory. A good run ends `w2p end 0`.

       mkdir -m 700 "${W:?}/config${N:?}" &&
       python3 scales/words_to_policy.py "${BASE:?}" "${WORDS:?}" "${GGUF:?}/wsums.txt" "$W/config$N/policy.txt" "$W/config$N/KL.bin"

5. Make `layout.txt`, and let the verifier judge the three files, on a CPU. The lines build the
   verifier if it is not there (two minutes), list the arrays that follow from the two files, place
   them in rows of 2^l entries (l is the third number of the line `group link` of `policy.txt`; 4
   entries stay empty at each row's end), and plan a verification at 128 bits, without a proof.

       { [ -x bin/cp_verify ] || verifier/build.sh; } &&
       { bin/cp_verify -plan arrays "${W:?}/config${N:?}" 128 > "$W/arrays$N.txt" || [ $? -eq 4 ] || { tail -n 1 "$W/arrays$N.txt"; false; }; } &&
       python3 -B prover/tools/pack_layout.py --arrays "$W/arrays$N.txt" "$(awk '$1 == "group" && $2 == "link" { print $5 }' "$W/config$N/policy.txt")" 4 > "$W/config$N/layout.txt" &&
       bin/cp_verify -plan accounting "$W/config$N" 128 | grep -E '^(plan: .*(can be a verification|ends NO: policy)|PLAN ONLY|NO:)'

   A planning call that admits the files ends with status 4, not 0. Good: the last call prints the
   one line `PLAN ONLY: no proof was checked`. `NO: <name>`: the verifier rejects the files. A line
   that begins `plan:` names what the files miss of the standard parameter form, or says that every
   verification under this `policy.txt` ends with `NO: policy` (its `sT` is not `sh` - 4).

6. Commit. `-M` is the number of stored copies of the weight table
   ([../docs/operators-duties.md](../docs/operators-duties.md)). `cp_commit` uses the first GPU that
   it sees, and nearly all of its memory.

       "${BUILD:?}/cp_commit" -weights "${GGUF:?}" -config "${W:?}/config${N:?}" -M 1 -ivk "$W/ivk$N" -isk "$W/isk$N" -worst-case-table "$W/table$N.txt"

   Before it draws its seed or makes the two directories, `cp_commit` checks:

   | what | exit status if not |
   |---|---|
   | the three files are there, the verifier admits them, and they have the standard parameter form | 1 or 2 |
   | the weights directory holds the shards themselves, `wsums.txt` and `fold.txt`, and `wsums.txt` lists the tensors that `policy.txt` asks for | 2 |
   | every tensor converts to integers | 3 |
   | the integers hash to the sums in `wsums.txt` | 6 |
   | every integer lies in the range that `policy.txt` declares for its kind (the embedding: at most 2^30 - 1) | 16 |
   | the worst-case check: every bounded site fits | 7 |

   Status 0: the commitment is made. Publish only `$W/ivk$N` (the public key directory), the
   commitment ID that `cp_commit` prints and the three files of `$W/config$N`. Publish the files
   under directory names of your own, without N: N - 1 is the number of earlier rounds ("What the
   files reveal"). `$W/isk$N`, the
   prover's directory, is secret. The weights directory must stay where it is: the prover reads it
   again. `$W/isk$N` must stay too: once it is removed, this commitment cannot be made again (a new
   seed gives a new ID). So look into a working directory of an earlier run (`ls -a`) before you
   remove it. That the worst-case check has passed is recorded as the line `ranges` of `isk.txt` in
   the secret directory, without which the prover does not start. No key protects that line: whoever
   can write there can write it.

   After any other status nothing is committed. Status 7: go on below. Otherwise fix the cause, type
   `N=$((N + 1))`, so that all names are new, and repeat steps 4 to 6. What the earlier rounds left
   stays secret ("What the files reveal").

**After status 7** `$W/table$N.txt` says which sites are short. Make them coarser, and repeat steps
4 to 6 with the new scales:

    python3 scales/worst_case.py "${W:?}/table${N:?}.txt" "${WORDS:?}" "$W/words$((N + 1)).json" &&
    WORDS=$W/words$((N + 1)).json N=$((N + 1))

The change does not guarantee that every site fits; only the next run of `cp_commit` shows it.

On a real model's weights, a public 4-bit conversion of Llama 4 Maverick's (its repository and
revision: [../THIRD_PARTY.md](../THIRD_PARTY.md), "A public model's weight files"),
`examples/maverick/run.sh` has run `calibrate.py`, `words_to_policy.py` and `cp_commit` once on the
released programs (by the run's record, which is not in this repository), with `--device cuda` and
e_res 10 ([../examples/maverick/README.md](../examples/maverick/README.md)); `worst_case.py` is no
step of that run. With the released programs, on such weights, the steps have not been run as this
page prints them, step 3 has not run on a CPU, and no step has run with an e_res other than 10. A
commitment and one proved request have run from each of a public Q8_0 and a public BF16 conversion,
on [other programs](../docs/glossary.md#other-programs), and F16 files have been read in tests
alone, with synthetic values. Step 6's line has not been run as printed, and neither `worst_case.py`
after status 7 nor a round on the scales that it writes has run on the released programs.

## Where a step stops

- **The converter prints `w2p refuse <class> <count>`** and writes nothing. Its docstring lists all
  classes. `law-W-down`: `kd`, the shift of a down projection, is out of range (with the lines
  `f 16384` and `xl 5` of the full-size base file it must be 10 to 19). The class alone: too small;
  lower e_res, which raises every `kd` by one per step. With `law-U-down` beside it: too large;
  raise e_res. That remedy follows from the fit's formula for `kd` ("The shifts") and has not been
  tried. To fit again at another e_res, set `ERES`, type the line with `mktemp` again for a new
  `W`, set `WORDS=$W/fit/words.json N=1`, and repeat from step 2.
- **Step 2 is repeated.** It needs a weights directory without a `wsums.txt` and a `fold.txt`.
  Either make a new one with a copy of the shards. This line makes it, closed to other accounts from
  the start, copies the shards so that only you can read the copies, and sets `GGUF` to it:

      umask 077 && NEW=$(mktemp -d ~/weights.XXXXXX) && cp --reflink=auto "${GGUF:?}"/*.gguf "${NEW:?}"/ && GGUF=$NEW && echo "GGUF=$GGUF"

  The copy takes the shards' size again, unless both directories lie on one file system that shares
  blocks.

  Or take the same directory, once you have put the two old files elsewhere, where only you can read
  them (not if a commitment has been made from this directory: its prover reads them again).
- **Step 5 says `NO:` to files that the converter wrote.** The converter applies only some of the
  verifier's conditions. Seen with synthetic values, not on the tools in this repository, but on
  tools that differ from them in places: `kr`, a router's shift, below 8 passes it, and the verifier
  says `NO: no-wrap-guard-radix-clamp`. For that one the tools have no remedy.
- **`worst_case.py` prints that e_res has changed:** the residual stream was short. Lower `ERES` by
  the bits that it prints, repeat step 2, and go on with step 4; step 3 need not be repeated. If the
  converter then prints `w2p refuse emb-bits 1`, set `"emb_bits"` in the new `words.json` to the
  fourth field of the line `token_embd.weight` of the new `wsums.txt`.
- **`worst_case.py` ends with status 3:** the file is written with all other changes, but a site is
  short that the program cannot change; `WORDS` and `N` are as before. A gate or a router: their
  outputs are lookup keys of fixed public tables. Both base files named here already have the lines
  `gcl 1` and `rcl 1` that the program names, and `eg` leaves the standard parameter form, so no
  remedy keeps the form. A norm's quotient, the head margin, or a site that the program does not
  know: no remedy. An up site of the wrong kind for its layer: the table and `words.json` probably
  do not belong together.
- **A partial test of the worst-case check:** `worst_case.py against TABLE WORDS_JSON MAXIMA_JSON`,
  with the first round's table and the two files of `calibrate.py`. It compares the sites `q`, `k`,
  `v` and the three up sites, on the calibration set alone. Status 1: a recorded maximum lies more
  than 0.01 bits above its bound, so the files are not from one fit, or the check is wrong.

## What the files reveal

Weight hiding as a whole: [../guide/weight-hiding.md](../guide/weight-hiding.md). The public files
that these tools write are not masked. `policy.txt` shows the model's whole shape.
`KL.bin` and the fitted lines of `policy.txt` are computed from the weights and the calibration set,
and they say something about every weight matrix. By the formulas in `fit_scales.py` they give at
least what follows. The counts are for Llama 4 Maverick's shape: 433 matrices without the embedding,
where a layer's routed experts of one kind count as one matrix, and a number for them is the largest
over all experts.

- **97 matrices, one by one** (every gate, every router, the output matrix): the largest entry to
  within a factor of two, after the norm's gains are multiplied in.
- **The other 336, in 168 pairs** (query and key, value and output, up and down): the product of the
  pair's two largest entries to within a factor of four. The files do not give the largest output of
  the key, value or up projection on the calibration set. Whoever knows or guesses that number for a
  pair gets the pair's two largest entries one by one; that works for 120 of the pairs.
- **What a passed check tells.** A published commitment has passed the worst-case check under public
  shifts. That puts an upper limit on one more number of each of the 216 query, key, value and up
  matrices, roughly the longest row's length relative to the largest entry, and one joint upper
  limit on the output and down matrices and the embedding. Where `worst_case.py` has raised a shift,
  the site may just fit: for whoever knows or guesses that it does, the limit is the number itself
  to within a factor of two. A lowered e_res tells as much about the residual stream's bound.
- **Every norm:** how large the largest normalized activation on the calibration set was, in steps
  of a factor of two.
- **A test of a guessed model.** Whoever holds candidate weights can run these tools on them and
  compare the result with the published files, for example to ask whether the committed model is a
  fine-tune of a public one. For the 265 numbers of the first two points that takes the candidate's
  weights alone (`bin/weight_sums`): a secret calibration set does not prevent it.

The list is not shown to be complete. What was tried, not on the tools in this repository, but on
tools that differ from them in places: the numbers of the first two points were confirmed on
synthetic models and numbers, and getting a pair's entries one by one was tried on synthetic numbers
only, never with a related model's activation sizes. The rest is calculated or read, not tried, and
no files fitted to weights at full size were examined.

The other files reveal more. Guard them like the weight files. Publish none of them.

| what | what it reveals |
|---|---|
| `wsums.txt` | each matrix's exponent, and an unsalted SHA-256 of its integers, against which anyone can test a guess |
| `words.json`, and an `exponents.json` that you write for `fit_scales.py` | the same exponents |
| `maxima.json` | the largest activation at each site and layer |
| each round's worst-case table | bounds that are computed from the weights |
| the configuration directories of earlier rounds | compared with the published files: every layer's bits short |
| what the tools and `cp_commit` print | among it the fit's lines "made B bits coarser", all output of `worst_case.py`, and whether `cp_commit`'s checks passed |
| the number N in the name of a round's file or directory | one small number: N - 1 is the number of earlier rounds. A round that ended with status 7 did so because the worst-case check found sites too narrow, which depends on the weights |
| whatever an unfinished call of `cp_commit` left under `$W` | it can hold the seed |

Of these, `wsums.txt`, the JSON files that the tools write, `policy.txt`, `KL.bin` and the tables
are created readable by their owner only.

## What the tools do not check

- **Whether the verifier takes the files.** The converter leaves out some of the verifier's checks. A
  planning call of `cp_verify` checks the public inputs and reads no proof: step 5 of "A real model"
  runs two (`-plan arrays`, `-plan accounting`), and
  `bin/cp_verify -plan request <policy.txt> <KL.bin> 128 -layout <layout.txt> -prompt <request file> -reply <y file>`
  does the same for one request (the program's usage text has its options).
- **The acceptance policy.** Of its seven lines the converter copies `C`, `theta`, `cap`, `sh`, `sT`
  and `F` from the base file and rewrites only `s`, the logits' exponent. Both base files named here
  have the exact match: every token that a proof covers is the model's first choice. Under other
  values an acceptance can say less, and whether a request is served depends on how far a token's
  score lies below the best. Thresholds that you fit on your model's scores are public too
  ([../docs/what-is-not-hidden.md](../docs/what-is-not-hidden.md)).
- **That `policy.txt` and `KL.bin` come from one call of the converter.** A mixed pair defines a
  function that belongs to neither fit, and no program notices.
- **How close the integer model is to the floating-point model.** No tool here compares their
  outputs ([../docs/limits.md](../docs/limits.md#the-integer-model)). Values that the verifier
  accepts can still be badly chosen.
- **Other prompts.** At the eight recorded sites the fit makes the calibration set's maxima just fit
  ("The sites"). On another prompt a value beyond its cell is clamped, which is part of the model
  that is proved. A value beyond its site's limit makes the request fail, and a failed request tells
  its sender something about the weights
  ([../docs/what-is-not-hidden.md](../docs/what-is-not-hidden.md)). The worst-case check is there to
  rule that out for all prompts, if it computes its bounds correctly and the programs make no other
  range test than those it bounds; not all of that is established
  ([worst-case-bounds.md](worst-case-bounds.md)).
- **That `float_forward.py` is the publisher's model.** It is adapted from the Llama 4 text model of
  Hugging Face transformers 5.17.0 (`modeling_llama4.py`), under the settings in its docstring;
  nothing compares the two. Like that file it rotates pairs of neighbouring coordinates, in the
  order of the query and key matrices' stored rows. It cannot tell whether a converter has permuted
  these rows, as llama.cpp's `convert_hf_to_gguf.py` does for the architecture `llama`. To
  find out, compare `blk.0.attn_q.weight` of an unquantized GGUF file with the publisher's query
  matrix. It skips tensors that it does not know, such as `rope_freqs.weight`, and prints how many.
- **Weights that the forward pass does not read:** a routed expert that no token of the calibration
  set chooses, and the output matrix. A value there that is not finite goes unnoticed.
- **The result of `worst_case.py`.** It validates neither the new scales (the converter does) nor
  that the table was made under the scales that you pass.
- **The table digests on another platform.** The tables are computed with the platform's
  mathematical library. That every library gives the same tables has not been shown.

## The residual stream's exponent, e_res

The residual stream and the embedding share one exponent, 11 unless you choose another. One less
means one fractional bit less in the stream, and `ko` and `kd` one higher in every layer. For Llama
4 Maverick's public weights `examples/maverick/run.sh` takes 10, a value settled by work on the fit
to those weights ([../THIRD_PARTY.md](../THIRD_PARTY.md), "What depends on Meta's models"). When to
change it: "Where a step stops".

It is stated three times: `--eres` of `bin/weight_sums`, the line `eres` of `fold.txt`, and `--eres`
of `calibrate.py`. A mismatch stops `calibrate.py` (status 2), the converter (`w2p refuse eres 1`)
or `cp_commit` (status 6). The base file does not bind it: the converter rewrites the lines that
depend on it. `calibrate.py` takes 1 to 24, but at width 5,120 at most 2 to 14 lead to files: at 1
and from 15 on the converter prints `w2p refuse arith-eps 3`.

## The calibration set

`calibrate.py` reads a `.npy` integer array of shape (windows, tokens per window), or raw 32-bit
little-endian ids with `--window`. A window should begin with the model's start token and can have
up to 8,192 tokens. The program checks only that length and the ids' range; whether the set suits
the model is for you to judge.

`make_calibration_set.py` makes one fixed set for Llama 4 Maverick: 32 windows of 1,024 ids from the
training split of WikiText-2. It samples the corpus as the GPTQ authors' code does
(github.com/IST-DASLab/gptq, `datautils.py`; E. Frantar et al., arXiv:2210.17323), here with
the Llama 2 tokenizer, and encodes the sample again with the Llama 4 tokenizer. Its docstrings have
the rule and all sources. The rule is fixed so that anyone can make the same file again; it is no
recommendation for other models. The call below writes the set into `W`, the working directory that
the first block of "A real model" makes, which must exist.

    python3 scales/make_calibration_set.py TRAIN_PARQUET LLAMA2_TOKENIZER_JSON LLAMA4_TOKENIZER_JSON "${W:?}/calibration_set.npy"

`examples/maverick/run.sh` does not look there. It reads the set as `calibration_set.npy` in its own
weights directory, the one that you give it with `--weights`: for that script, write that path in
place of the call's last argument, and make the set as the account that runs the script. Without
that file, or with one of another SHA-256, the script stops with status 1 before it fits. It
replaces no file, so you move a wrong one yourself.

`make_calibration_set.py` fetches nothing: you fetch the three files yourself. It accepts only these
three, each at `https://huggingface.co/<repository>/resolve/<revision>/<path>`, unless you pass
`--any-files`. The two tokenizers' repositories are open only to users whose request for access the
publisher has granted; the request includes accepting the Llama 2 or the Llama 4 Community License
Agreement. So fetch these two files with an account that has that access.

| input | repository | revision | path | bytes | SHA-256 |
|---|---|---|---|---|---|
| WikiText-2, raw, training split | datasets/Salesforce/wikitext | b08601e04326c79dfdd32d625aee71d232d685c3 | wikitext-2-raw-v1/train-00000-of-00001.parquet | 6,357,543 | e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7 |
| the Llama 2 tokenizer | meta-llama/Llama-2-7b-hf | 01c7f73d771dfac7d292323805ebc428287df4f9 | tokenizer.json | 1,842,767 | bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e |
| the Llama 4 tokenizer | meta-llama/Llama-4-Maverick-17B-128E-Instruct | 73d14711bcc77c16df3470856949c3764056b617 | tokenizer.json | 27,948,578 | 172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4 |

The set has 131,200 bytes and the SHA-256
a2ce04bd55b1b6fa2b9148744e087f5a2a484c1b3d525ea9fb2fc1ab8c6c2c02. The program compares it; status 1
means that the file is written and a sum differs. The sum was checked with Python 3.13.15, numpy
2.3.5, pyarrow 23.0.1 and tokenizers 0.22.2; another version of Python or of tokenizers may give
another set.

## The recorded sites and the scales

`maxima.json` holds the largest absolute value at eight types of site: `n1` and `n2` (a layer's two
norms, before the gains), `q`, `k`, `v` and `u` (the up projection) in every layer, `uR` (the routed
expert's up projection) in layers with experts, and `nF` (the final norm) once. The fit gives each
cell the highest exponent at which its recorded maximum still fits, and then makes some coarser, so
that all layers share one `sh` and a layer's experts one `kd` ("The fit of the scales" has the
rule).

`KL.bin` is a row of 32-bit little-endian integers: for every layer
`kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp`, then for every layer with experts `kr kgR kuR`. The k's
are the shifts of the query, key, value, attention output, gate, up and down projections, of the
router and of the routed experts' gate and up; M and K are a norm's two constants; `sh` (another
number than the line `sh` of `policy.txt`) shifts an attention score to the softmax table's lookup
key; `rp` is 1 where the layer applies the rotary encoding. In `policy.txt` the converter rewrites
15 lines, which its constant `WRITTEN_LINES` lists.

The last digits of `maxima.json` depend on the device, the libraries' versions and `--chunk`. A
scale changes only if a maximum lies that close to a power of two. The result is another valid set
of scales, and another commitment ID.

Where the examples' scales come from. What this paragraph says of how the numbers were chosen rests
on the authors' own records. They are not in this repository, so a reader cannot check them.
According to them the numbers in the `KL.bin` next to the full-size base file (`BASE` in "A real
model") are constants, typed in to suit the ranges of the synthetic-weight generator; `M1` and `M2`
were picked by hand. When they were typed in, scales had been fitted to Llama 2 7B, a different
model with public weights, and the ranges that they took were known. A second set of ranges, for the
norms' constants and the attention shift, was known as well. Where it came from is not established,
and whether Llama 4 Maverick's weights had a part in it is not known
([../THIRD_PARTY.md](../THIRD_PARTY.md), "What depends on Meta's models"). One of a layer's fields,
the attention shift `sh`, very likely goes back to a calibration run on Llama 2 7B; that is
inferred, not recorded. Whether the second set had a part in `sh`, and whether the known ranges
shaped any other field of the file, is unknown. The same records say that the `KL.bin` of
`prover/examples/target_small_f2` holds typed-in constants as well. Several of its numbers are also
the full-size file's, `M1` and `M2` included; its `sh` is a different number. The Llama 2 7B ranges
were known at that time too; whether they, or the second set, shaped this file's numbers is equally
unknown. Three more examples have that file with some scales changed
([../prover/examples/TARGET-KITS.md](../prover/examples/TARGET-KITS.md) names them), so the same
holds of the values that they keep. "Try it" says where its own example's scales come from.
[../THIRD_PARTY.md](../THIRD_PARTY.md), "What depends on Meta's models", names in its rows the files
under `prover/examples/` that hold such values. Step 4 reads none of these files: the `KL.bin` that
it writes is new.

The integer model itself: [../docs/model.md](../docs/model.md). Licences of the code that these
tools adapt or import (llama.cpp's gguf, Hugging Face transformers):
[../THIRD_PARTY.md](../THIRD_PARTY.md).

<a id="fit"></a>
## The fit of the scales

The rule by which step 3, `calibrate.py`, chooses the scales. The whole computation is the function
`fit` of `fit_scales.py`, whose inputs `calibrate.py` collects. How the integer model uses the
scales: [../docs/model.md](../docs/model.md). What the published scales tell about the weights:
"What the files reveal". What a fit does not settle: "What the tools do not check", "Other prompts".

The method is static calibration by the largest absolute value, with scales that are powers of two.
Its sources, which `fit_scales.py` cites with their titles: H. Wu et al., arXiv:2004.09602, section
3.4 (the scale follows from the largest absolute value that the calibration run records); M. Nagel
et al., arXiv:2106.08295, section 2.2, and S. R. Jain et al., arXiv:1903.08066, appendix A.2 (with
powers of two, scaling is a bit shift with rounding); Jain et al. also constrain scales to powers of
two (section 3.1) and choose one from a trained threshold (section 3.2, a related technique); the
file cites neither section; B. Jacob et al., arXiv:1712.05877, section 2.2 (what takes a product of
integers to its output's scale) and appendices A.2 and A.3 (values that are added need one scale).
None of them gives the rules below for the norms' sites and for `sh`.

| what | how it is chosen |
|---|---|
| `kq kk kv ko kg ku kd kr kgR kuR sh M1 M2` in `KL.bin`; `MF` and `s` in `policy.txt` | fitted: from the weight tensors' exponents and the recorded maxima, so from the weights and the calibration set |
| `eps1 eps2 epsF` | computed from the width, the model's norm epsilon and e_res |
| `K1 K2 KF S rx cs` | constants in `fit_scales.py` |
| `rp` | 1 where the layer applies the rotary encoding: a pattern fixed in `gguf_weights.py` ("What the fit reads") |
| `theta0` | the rotary base in the model's files |
| `wb_emb` | always 21, written by the converter |
| e_res | you choose it ("The residual stream's exponent, e_res") |
| the 32 lines with the tables' digests | if the base file has them, computed again by the converter from the new values |
| every other line of `policy.txt` | copied from the base file |

<a id="fit-inputs"></a>
### What the fit reads

- The shape (`shape_words` in `gguf_weights.py`). From the first shard's metadata: the width d, the
  number of layers L, the head dimension dh (128 if the key is absent), eps_model (the epsilon of
  the floating-point model's norms; 1e-5 if absent) and theta (the rotary base, not the line `theta`
  of `policy.txt`; 500,000 if absent). A model whose metadata name another architecture than
  `llama4`, or another number of routed experts per token than 1, is rejected. Fixed in that file:
  layers 1, 3, 5, ... have experts, and all layers but 3, 7, 11, ... apply the rotary encoding.
- e(T), the exponent of the weight tensor T: the third field of its line in `wsums.txt`
  (`exponents_for_the_fit` in `calibrate.py`). The committed integers are about 2^e(T) times the
  stored values, with the factors of "The sites" multiplied in first
  ([../docs/model.md](../docs/model.md), section 11.4). Below, e(attn_q) is short for the exponent
  of `blk.<layer>.attn_q.weight`. A layer's routed experts of one kind are one tensor.
- mx(site, layer), the recorded maxima ("The sites").
- e_res, from `--eres`, and eg, which "The fixed exponents" explains.

<a id="fit-sites"></a>
### The sites

| site | the activation | layers |
|---|---|---|
| `n1` | the residual stream after the layer's first norm, before the norm's gains | all |
| `q` | the queries after the rotary encoding (in a layer without it: after the query factor of "The floating-point pass"), divided by sqrt(dh) | all |
| `k` | the keys after the rotary encoding (in a layer without it: as projected) | all |
| `v` | the values | all |
| `n2` | the residual stream after the layer's second norm, before the gains | all |
| `u` | the up projection of a dense layer or of the shared expert | all |
| `uR` | the up projection of each token's routed expert | those with experts |
| `nF` | the last residual stream after the final norm, before the gains | once, under the key L |

mx is the largest absolute value over all windows, tokens, heads and channels, and for `uR` over all
experts that some token chose (`record` in `calibrate.py`). A value that is not finite, or a maximum
of 0, ends the run with status 2.

The gates, the routers' logits and the residual stream are not recorded. Their exponents are fixed
(eg for the gates, 8 for the routers' logits) or yours (e_res, for the stream), so the fit does
nothing to keep them within their limits, not even on the calibration set. At the eight sites it is
the floating-point pass's maxima that fit, not the integer model's own values.

The conversion ([../docs/model.md](../docs/model.md), section 11.3) multiplies a norm's gains and
the constant sqrt(d) / 64 into every matrix that reads the norm's output, and 1 / sqrt(dh) into the
query matrix. That is why the norms' sites lie before the gains and why `q` is divided by sqrt(dh):
the forward pass reports the undivided query, and `calibrate.py` multiplies its maximum by dh^-0.5.

<a id="fit-pass"></a>
### The floating-point pass

`forward` in `float_forward.py` runs each window as a prompt of its own, from the embedding to the
final norm, in float32, on the values that `gguf_weights.py` decodes from the files (for 4-bit and
8-bit files: the quantized values, not the publisher's). It computes no logit and does not read the
output matrix. In a layer without the rotary encoding the queries at position p are multiplied by
the query factor, 1 + 0.1 ln(1 + floor((p + 1) / 8,192)), which is 1 everywhere but at the last
position of a window of 8,192 tokens. The integer model has no such factor. The file's docstring
lists the settings of the publisher's model that the forward pass assumes.

Nothing compares the forward pass with the publisher's implementation ("What the tools do not
check"). So two things that the integer model assumes as the forward pass does are unconfirmed too:
that neighbouring rows of the stored query and key matrices are the rotary pairs, and that each
tensor is the matrix that [../docs/model.md](../docs/model.md), section 11.3, assigns to its name.

<a id="fit-fixed"></a>
### The fixed exponents

Constants at the top of `fit_scales.py`. They belong to the integer model's public tables and its
norm.

| name | value | what it is |
|---|---|---|
| (cell) | 12 bits | a cell holds -2,047 .. 2,047 |
| `cs` | 12 | an entry of the rotary tables is a cosine or a sine times 2^cs |
| `rx` | 8 | the exponent of the softmax table's lookup key |
| `S` | 12 | the largest softmax weight, which stands for 1, is 2^S (expS in [../docs/model.md](../docs/model.md)); it enters no formula below |
| eg | 8 | the exponent of the SiLU table's lookup key; `fit_scales.py --eg` takes 4 to 8, and below 8 `policy.txt` needs the line `eg`, which leaves the standard parameter form |
| (SiLU output) | 6 | the exponent of the SiLU table's output |
| (router) | 8 | the exponent of a router's logit, the sigmoid table's lookup key |
| `K1 K2 KF` | 32 | "The constants of a norm" |

Of these the converter holds `S`, `rx` and `cs` to these values, and with them theta and the
epsilons: it prints `w2p refuse arith-const` unless `S`, `rx`, `cs` and theta are 12, 8, 12 and
500,000, and `w2p refuse arith-eps` unless the three epsilons are those for eps_model = 1e-5 ("The
epsilon of a norm").

<a id="fit-cells"></a>
### The exponent of each cell

Es(site, layer), as the code's comments call it, is the exponent of the cell that stores the site's
value: a real value x is stored as about x * 2^Es (at a norm's site: x * 2^Es / c, step 2). `fit`
computes it in four steps.

1. **q, k, v, u and uR.** Es is the largest integer E with mx * 2^E <= 2,047 (`cell_exponent`). `u`
   and `uR` are fitted to 12 bits too, although the integer model does not clamp an up projection
   and lets it have 16 bits.
2. **n1, n2 and nF.** Let t be the largest integer with 4^t <= d, and c = sqrt(d / 4^t), so that
   sqrt(d) = c * 2^t and 1 <= c < 2; at d = 5,120, t = 6 and c = sqrt(1.25). Es is the largest
   integer E with (mx / c) * 2^E <= 2,047, but at most 14 - t (`fitted_norm`; the root and the
   quotient are float64 operations). Why c: the integer norm divides by the root of the sum of the d
   squares, not of their mean, so its output is the normalized activation times 2^(M - K) / sqrt(d),
   which is 2^Es / c for M - K = Es + t ("The constants of a norm"). In a product with a weight
   matrix the conversion's constant sqrt(d) / 64 cancels c only where t = 6. So the shifts below and
   E_logit are right for widths from 4,096 to 16,383 only; the conversion accepts only 5,120.
3. **One sh for all layers.** A score has the exponent Es(q) + Es(k), and `sh` shifts it to the
   softmax table's lookup key, whose exponent is rx. `sh` is the least of Es(q) + Es(k) - rx over
   the layers, brought into 0 .. 12. Then Es(q) is recomputed as rx + sh - Es(k) in every layer.
   That is coarser than step 1 where the layer asked for more than `sh`. If `sh` was raised to 0, it
   is finer where the layer asked for less: there the recorded maximum does not fit, and the
   integer model clamps it.
4. **One kd for a layer with experts.** The shared and the routed expert's down projections are
   added before the one shift `kd`. Let x = (Es(u) + e(ffn_down_shexp)) - (Es(uR) +
   e(ffn_down_exps)). If x > 0, Es(u) is lowered by x; if x < 0, Es(uR) is lowered by -x. The
   program prints which side was made how many bits coarser. Beyond 31 either way the run ends with
   status 2.

<a id="fit-shifts"></a>
### The shifts

A shift is the sum of the exponents of a product's two factors, minus the exponent of what receives
the result. Es is the final value of the four steps of "The exponent of each cell", and the steps
named in the table are those. gate, up and down stand for `ffn_gate`, `ffn_up` and `ffn_down` in a
dense layer, and for `ffn_gate_shexp`, `ffn_up_shexp` and `ffn_down_shexp` in a layer with experts.

| scale | value | note |
|---|---|---|
| `kq` | Es(n1) + e(attn_q) - Es(q) + cs | before this shift the rotary step multiplies by table entries, which have the factor 2^cs (in a layer without the rotary encoding: by 2^cs itself) |
| `kk` | Es(n1) + e(attn_k) - Es(k) + cs | likewise |
| `kv` | Es(n1) + e(attn_v) - Es(v) | |
| `ko` | Es(v) + e(attn_output) - e_res | attention's output is a weighted mean of values and has their exponent |
| `kg` | Es(n2) + e(gate) - eg | |
| `ku` | Es(n2) + e(up) - Es(u) | |
| `kd` | 6 + Es(u) + e(down) - e_res | the first factor is the SiLU table's output times u; by step 4, 6 + Es(uR) + e(ffn_down_exps) - e_res is the same number |
| `kr` | Es(n2) + e(ffn_gate_inp) - 8 | |
| `kgR` | Es(n2) + e(ffn_gate_exps) - eg | the routed expert's input, the norm's output times the sigmoid of the router's logit, keeps the exponent Es(n2) |
| `kuR` | Es(n2) + e(ffn_up_exps) - Es(uR) | likewise |
| `sh` | the least over the layers of Es(q) + Es(k) - rx, with the Es of step 1, brought into 0 .. 12 | step 3; the same number in every layer |

`fit` does not make sure that the verifier admits a scale. The run ends with status 1 if one of the
ten shifts `kq` to `kuR` lies outside 1 .. 63 or a norm's M outside 24 .. 47; it does not test `sh`.
The converter applies narrower ranges, which its docstring lists. `sh` must be 4 to 12 there
(`w2p refuse law-sh`), so a fit whose `sh` is 0 to 3 gives no files.

<a id="fit-norm-constants"></a>
### The constants of a norm

K1 = K2 = KF = 32. M1 = 32 + Es(n1) + t, M2 = 32 + Es(n2) + t and MF = 32 + Es(nF) + t, with the t
of "The exponent of each cell", step 2. The integer norm multiplies a stream value by about 2^M
divided by the root of the sum of its row's squares, and shifts the product by K
([../docs/model.md](../docs/model.md), section 5). Es is at most 14 - t, so M is at most K + 14, as
the verifier requires.

<a id="fit-epsilon"></a>
### The epsilon of a norm

eps1 = eps2 = epsF = d * eps_model * 4^e_res, rounded to an integer
([../docs/model.md](../docs/model.md) calls the three the regularizers). The integer norm adds it to
the sum, not the mean, of d squares with the exponent 2 e_res: hence the factors. At d = 5,120 and
e_res = 11 the number is 214,748.

It is computed in two ways. `fit` takes eps_model as it is given (from `calibrate.py`: as the shard
stores it, a float32 number), multiplies in float64 and rounds to the nearest integer, ties to even.
The converter (`check_ranges`) and `worst_case.py` (`refit`) compute floor((2 d 4^e_res + 100,000) /
200,000) in integers, for eps_model = 1e-5 itself. The converter rejects three numbers that are not
its own, or are 0; "The residual stream's exponent, e_res" says at which e_res that happens.

`policy.txt` has each of the three, and theta as `theta0`, in two lines, such as `eps1_lo` and
`eps1_hi`: the first is the number modulo 2^32, the second the number divided by 2^32, rounded down.

<a id="fit-logits"></a>
### The exponent of the logits

E_logit = Es(nF) + e(output), where e(output) is the exponent of `output.weight`: an integer logit
divided by 2^E_logit is about the model's logit. No shift follows the output matrix. The converter
writes E_logit as the line `s` of `policy.txt`, and rejects a value above 63
(`w2p refuse logit-exponent`).

<a id="fit-coarser"></a>
### Making a cell coarser

Es of the cells `q`, `k`, `v`, `u` and `uR` enters the shift that writes the cell with a minus sign,
and the scale of the step that reads it with a plus sign. So raising the first by b and lowering the
second by b makes the cell b bits coarser and changes no other exponent. e_res enters `ko` and `kd`
with a minus sign; the stream's readers are the norms, which divide its scale out, so only their
epsilons and the embedding's conversion follow it. Hence four sums do not depend on these exponents:
`kv` + `ko` + e_res, `ku` + `kd` + e_res, `kuR` + `kd` + e_res, and `kq` + `kk` + `sh`.

`worst_case.py` (`refit`) changes the scales so that the four sums stay the same; its docstring has
the rule for each site. The result is another function, which rounds more coarsely at the changed
cells.

Not free in this way: `kg`, `kgR` and `kr`. Their results are lookup keys of public tables whose
keys have a fixed exponent: under another shift the same activation would select another entry. For
the norms' constants `worst_case.py` has no rule. "Where a step stops" says what to do in these
cases.
