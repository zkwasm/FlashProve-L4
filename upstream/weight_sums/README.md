# weight_sums

This directory makes `wsums.txt`, the *sums file* that `cp_commit` needs to commit to a real model's
weights. For each weight tensor that the prover commits to, the file holds the tensor's exponent and
the SHA-256 of its integers. `cp_commit` converts every tensor itself and commits only if every hash
matches, so the file fixes which weights are committed. You need it only for a real model: the
example configurations generate synthetic weights, and the tests write their own sums files.

## Keep the file as you keep the weights

Each hash is a plain SHA-256 of a tensor's integers, with no secret or random input. Anyone who
holds the file can test a guess of a tensor's values against it, offline and without limit. The
exponents alone give each matrix's largest value to within a factor of two. Both programs
create the file with mode 0600, for its owner alone, whatever the umask. Guard it like the weight
files; the `umask 077` below closes whatever else you write in that shell, such as a saved copy of
`cp_commit`'s output.
`cp_commit`'s line `<n> of <N> tensors' integers have the sums of wsums.txt` is computed from the
weights too: keep that output the same way.

## Make the file

You need g++ 13 or later (tested with g++ 13.3.0 only) and the model's files in GGUF, the file
format of the ggml and llama.cpp projects: its *shards*. Give all shards in one command, the shard
that holds the model's metadata first: in a model split into numbered files, usually the one
numbered 00001. From the top of the tree:

```sh
weight_sums/build.sh
umask 077
W=$(mktemp -d) && echo "W=$W" &&
bin/weight_sums --eres 11 --threads 8 --out "$W/wsums.txt" \
  model-00001-of-00002.gguf model-00002-of-00002.gguf
```

The build takes a few seconds and writes `bin/weight_sums`, or, with `- <output directory>`, into
that directory. It overwrites a `weight_sums` that is there, with a notice on standard error; a
build that fails or is interrupted leaves it unchanged. You will run the program that is built
there, so the script refuses (status 2) an output directory that another account could change, and
its message names the place. For the default directory, inside the checkout, it prints a note and
builds if the place lies at or above the top of the checkout, since whoever can change such a place
can change the script and its sources too.

Do not build under `sudo`. A build under `sudo` that makes `bin/` in a checkout of yours leaves a
`bin/` and a program that belong to root, and your own build into that `bin/` then stops with
status 2. Name another output directory, with `- <output directory>`; the tests and other scripts
of the repository look for `weight_sums` in `bin/`, so for them start again in a new copy of the
repository. If a build is killed, its work directory, `weight_sums.building.XXXXXX`, can stay in
the output directory, and builds do not remove it. The script does not empty your environment:
`PATH` chooses the compiler, and variables such as `COMPILER_PATH`, `GCC_EXEC_PREFIX` and `CPATH`
choose its parts and add headers. Build from a shell whose environment you know.

The commands below use `$W`: run them in the same shell, or set W again to the printed path. Many
systems empty /tmp at a restart: at
the end of this section, move `wsums.txt` and `fold.txt` to the directory that holds the model's
files. Neither program writes over an existing file (below). `--eres` is e_res, the exponent of the
embedding and of the residual stream, a free parameter of the integer model (11 if omitted). Choose
it first and give the same value here, in `fold.txt` below, and to the tools in `scales/` that fit
the model's scales, which read this file ([scales/README.md](../scales/README.md), "A real model"
and "The residual stream's exponent, e_res"). This directory takes -100 to 100, the tools in
`scales/` 1 to 24 (with GGUF files, only 2 to 14 lead to usable files), and `cp_commit` 0 to 40, and
at most 24 for a layer's matrix other than the router if its type is Q4_K, Q5_K, Q8_0, F16 or BF16.
A value that does not suit the model shows only when a tool in `scales/` rejects the fit: then
choose another and make the file again.

Then write, next to it, a file `fold.txt` with these two lines, the first with your value of e_res:

```text
eres 11
qkrows asis
```

The second line is required, and `asis` is its only value: the rows of the query and key matrices
stay in the order in which the file stores them. If `fold.txt` has another e_res than `--eres`, the
embedding's hash differs, and `cp_commit` stops.

Optionally, a second program computes the same file in Python, so that an error in either shows as
a difference. Run it in a virtual environment of its own, since its `pip` line replaces any other
numpy. After the `pip` line, the block checks that the first program's file is there:

```sh
pip install numpy==2.3.5 gguf==0.19.0
ls -l "${W:?first run the block that sets W, or set W to the path that it printed}/wsums.txt" &&
python3 weight_sums/weight_sums.py --eres 11 --workers 8 --decode-twice --out "$W/wsums-py.txt" \
  model-00001-of-00002.gguf model-00002-of-00002.gguf &&
cmp "$W/wsums.txt" "$W/wsums-py.txt"
```

| option | program | meaning | default |
|---|---|---|---|
| `--out PATH` | both | where to write the file; required | |
| `--eres N` | both | e_res, from -100 to 100 | 11 |
| `--threads N` | `bin/weight_sums` | threads, 1 to 256 | 8 |
| `--workers N` | `weight_sums.py` | worker processes | 8 |
| `--decode-twice` | `weight_sums.py` | decode each matrix twice, to use far less memory | off |
| `--any-gguf-version` | `weight_sums.py` | run with a gguf-py (the pip package `gguf`) other than 0.19.0 | off |
| `--help` | `weight_sums.py` | print a help text | |

Every other argument is a shard; put `./` before a shard whose name begins with `-`. Give each
option once, before the shards. The file depends only on the shards and on e_res: threads, workers
and `--decode-twice` give the same bytes, and so, by design, do other machines and compilers (tested
on one machine and one compiler). `weight_sums.py` refuses a gguf-py other than 0.19.0 unless
`--any-gguf-version` is given: the rule fixes the order of the decoding's floating-point operations,
0.19.0 carries them out in that order, and the tests check the script at that version only. The
version of numpy is not checked; the tested Python is 3.13.

| exit status | meaning |
|---|---|
| 0 | the file is complete |
| 21 | a bad command line; a file that cannot be read or written; something already at the output path; for `weight_sums.py` also a missing numpy or gguf-py, or another gguf-py |
| 22 | a malformed GGUF file, missing metadata, a repeated tensor, a layer number too large, a wrong dimension, or a missing normalization weight that a committed tensor needs; for `weight_sums.py` also an unexpected internal error |
| 23 | a tensor of a storage type that the rule does not cover |
| 24 | an infinity or a NaN among the decoded weights or the normalization weights, or in their product |
| 25 | a matrix with no value other than 0; an exponent outside -100 .. 100; a product of a normalization weight and its constant that is not 0 but below 2^-100; an integer of the embedding of 2^30 or more; an integer above 4,095 after a shard changed during the run; or a model of another width or head dimension |
| 26 | not enough memory, or a worker of `weight_sums.py` that was killed or could not start |

**Neither program overwrites a file that is there when it starts.** Each exits with status 21 if
something exists at the output path or at its temporary file (`PATH.part` for `bin/weight_sums`,
`PATH.tmp` for `weight_sums.py`), or if the output's directory is missing or cannot be written. The
finished file gets its name in one step at the end, so a stopped run never leaves an incomplete
file. That step fails, again with status 21, if something has appeared at the output path meanwhile;
only on a file system that has neither of the two calls for it does `bin/weight_sums` look and then
rename. A run that is killed can leave its temporary file. The next run with the same `--out` then
stops: give `--out` another name, as the message says, and remove the leftover only when you are
sure that no other run is writing it. `weight_sums.py` needs a file system with hard links at the
output path.

Both programs report progress on standard error; only `weight_sums.py --help` prints on standard
output. No message contains a tensor's value or its position, but a message, like the exit status,
can tell the kind of fault, such as a matrix that holds only zeros.

## The file's form

Plain ASCII, one line per committed tensor, with five fields separated by single spaces:

| field | meaning |
|---|---|
| name | the tensor's name in the GGUF file, such as `blk.0.attn_q.weight` |
| elements | its number of elements |
| e | its exponent |
| bits | the least b >= 1 with -2^(b-1) <= w < 2^(b-1) for all its integers w; the prover reads it but does not use it |
| sha256 | the SHA-256 of its integers, 64 lower-case hexadecimal digits |

The hash is taken over the integers in the file's order of elements, each as a little-endian
two's-complement number of 4 bytes for `token_embd.weight` and 2 bytes for every other tensor, with
nothing else. The lines are sorted by name as byte strings, so `blk.10.` comes before `blk.2.`. A
last line `END <number of lines before it>` closes the file, and every line ends with one line
feed. The first and the last three lines for the tests' synthetic model:

```text
blk.0.attn_k.weight 5242880 12 13 b541aa63a42f456c4ecc08097c18d6be83cf68c0a08380047de15edd1d140482
output.weight 10485760 10 13 0ba5e53ad9aadeeb3f714c5a016754f7ba11468546694550a095c0d82dd671db
token_embd.weight 10485760 11 12 f7c6d43bbf56246d0cb5ba4c0b1bfdff9016b224f778ab1c810523ef992d19c4
END 38
```

## Who reads the file, and what is checked

Put `wsums.txt`, `fold.txt` and the GGUF files into one directory (you may also write the two files
straight into the GGUF files' directory), with no other file whose name ends in `.gguf`, and pass
the directory to `cp_commit` as `-weights DIRECTORY`. Lay the files themselves there: the default
build, the only one that takes a model's files, refuses a file there that is a symbolic link or that
it sees under a second name (a hard link). Where an account that you do not trust can make hard
links on that file system, that check is no reliable hurdle: your own setup must keep such links
out, for example with `fs.protected_hardlinks = 1` on every host that mounts that file system, set
before the files are put in place ([docs/operators-duties.md](../docs/operators-duties.md), "From
nothing to the first proof", step 7). `cp_commit` and `cp_prove` take the shards in the byte order
of their names and read the metadata from the first, so the shard with the metadata must sort first.

- `cp_commit` converts every committed tensor on the GPU, with its own implementation of the rule,
  taking the embedding's exponent from `fold.txt` and every other exponent from `wsums.txt`, and
  prints `cp_commit: <n> of <N> tensors' integers have the sums of wsums.txt`. If a tensor that the
  parameter file describes has no line in `wsums.txt` ("wsums.txt has no line for <name>"), or the
  file lists more tensors than that, it exits with status 2; if a hash differs, with status 6.
  Either way it has drawn no seed and written nothing; where a tensor has no line, the empty file that
  `-worst-case-table` names, if given, stays behind (read in the code). Only when every hash matches
  does it draw its secret seed and write its two directories. It records the directory's absolute
  path in its secret directory.
- `cp_prove` reads that path at every start and loads the weights from there, so the directory must
  stay where it was, with the same bytes. It takes only the exponents from `wsums.txt` and the two
  lines of `fold.txt`, and does not check the hashes.
- The package in `integrations/vllm/integer_model/` reads both files when it serves from GGUF files.
  With `"verify_sums": true` it compares every tensor's hash, and it can check the SHA-256 of the
  two files against digests that the operator gives (`integrations/vllm/integer_model/OPERATING.md`,
  "Verify what was loaded").

The commitment ID covers the configuration files and, of `key.bin`, the anchor's root over the
committed integers and its shape (also the published [OOD answers](../docs/glossary.md#record) and their OOD seeds), not the proofs inside that file; it covers neither `wsums.txt` nor `fold.txt` nor the GGUF files themselves.
A verifier never sees them: they bind only the provider's own run of `cp_commit`.

## What the file does not show

- **Completeness.** The programs of this directory convert the committed tensors that they find and
  do not know which ones a model should have: with a shard left out they still exit with status 0.
  `cp_commit` then refuses the file with status 2, since a tensor that the parameter file describes
  has no line.
- **Which model.** Matching hashes say nothing about whether the files hold the model that you
  expect.
- **Shared errors.** The two programs, the tests' oracle and `cp_commit` all implement the same
  statement of the rule, and all follow public sources: ggml's decoders for Q4_K, Q5_K and Q6_K, and
  for Q8_0, F16 and BF16 ggml's code, gguf-py's decoders and public descriptions of the formats. So
  an error in that statement, a reading that all of them share, or an error taken over from those
  sources does not show. Only this directory's two programs compute the exponents; `cp_commit`
  copies them.
- **Changes during a run.** Do not change a shard while a program runs: neither notices every
  change.
- **The refusals of `cp_commit`.** What this page says of `cp_commit` and `cp_prove` is read from
  their sources. One GPU test (`gpu.public_weights` in SUITE.md) changes one bit of one weight, leaves
  `wsums.txt` as it is and asks for status 6, under `-public-weights`.

## Which models, and the rule

The shards must be GGUF version 3, little-endian, with the tensor names that gguf-py 0.19.0 (the
llama.cpp project's Python package `gguf`) lists for the architecture `llama4`
(`gguf/constants.py`). The width (`embedding_length`) must be 5,120 and the head dimension 128, as
in Llama 4 Maverick, and the committed tensors F32, F16, BF16, Q8_0, Q4_K, Q5_K or Q6_K. The
committed tensors are `token_embd.weight`, `output.weight`, and in each layer l the tensors
`blk.l.KIND.weight` of the kinds `attn_q`, `attn_k`, `attn_v`, `attn_output`, and either `ffn_gate`,
`ffn_up`, `ffn_down` or, in a layer with experts, `ffn_gate_inp`, the three `_shexp` and the three
`_exps` kinds. The normalization weights are read but have no line of their own; any other tensor is
ignored. `tests/refusals.py` lists every refused input that is tested, with the exit status that
each program gives; on unusual files the two programs can differ.

The rule that turns a stored weight into an integer is defined in [docs/model.md](../docs/model.md),
"11 From the model's files to the integers". In short: each weight is decoded to binary32 as ggml
decodes it (Q8_0, F16 and BF16: as gguf-py does); in a matrix that reads a normalization, multiplied
by that normalization's weight and a constant; each matrix gets the largest exponent e for which
every value times 2^e rounds to at most 4,095 in absolute value; and each integer is the value times
2^e, rounded to nearest, ties to even. The embedding's exponent is e_res, and its integers must stay
below 2^30 in absolute value. The scaling steps are common in the quantization of neural networks
(in the table, the gains of a norm are the normalization's weights, and d is the model's width):

| method | publication |
|---|---|
| The gains of a norm are multiplied into the matrices that read the norm | S. Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv:2404.00456, section 3.4 and section 4 (Stage 1a, Figure 3); S. Ashkboos et al., "SliceGPT: Compress Large Language Models by Deleting Rows and Columns", arXiv:2401.15024, section 3.2 (Figure 3) |
| The factor sqrt(d) goes with the gains | the same paper on SliceGPT, section 2.1, equation 1 |
| One power-of-two scale per tensor, no zero point | M. Nagel et al., "A White Paper on Neural Network Quantization", arXiv:2106.08295, section 2.2 |
| The range taken from the largest absolute value | H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation", arXiv:2004.09602, section 3.4; S. R. Jain et al., "Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks", arXiv:1903.08066, section 4.2 |
| Rounding of ties to even in such a quantizer | the same paper of Jain et al., section 3.2 |

The byte layout of the block formats Q4_K, Q5_K and Q6_K, and the rule that decodes them, are those
of ggml, the tensor library of the llama.cpp project (github.com/ggml-org/llama.cpp):
ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K) and ggml/src/ggml-quants.c
(get_scale_min_k4; dequantize_row_q4_K, _q5_K, _q6_K). The container's layout is the one that the
comment at the top of ggml/include/gguf.h describes. These paths are those of llama.cpp at the
commit 794fe23f29fb40104975c91fe19f23798f7c726e. ggml and llama.cpp have the MIT licence ("Copyright
(c) 2023-2024 The ggml authors"; [THIRD_PARTY.md](../THIRD_PARTY.md), "ggml"). `bin/weight_sums` has
its own reader and a decoder adapted from ggml's; `weight_sums.py` uses gguf-py's. Q8_0, F16 and
BF16 are storage types of ggml too. For Q8_0 and BF16 the decoder of `bin/weight_sums` is adapted
from those of gguf-py 0.19.0 (`gguf/quants.py`), for F16 it follows that file's `dequantize`, and
the three types' numbers and sizes are those of its `gguf/constants.py`; THIRD_PARTY.md, "ggml" and
"gguf-py", says where the layout of a Q8_0 block was read.

## Size and memory

`bin/weight_sums` reads each matrix twice from disk, in pieces of 262,144 elements, and its memory
does not grow with a tensor's size; it is fastest when the files fit in the file cache or lie on a
fast disk. For a worker of `weight_sums.py` allow about 0.2 GB plus 4 bytes for each element of its
matrix (20 GB for a matrix of 5 billion elements), or about 0.2 GB with `--decode-twice`. Before its
workers start, `weight_sums.py` reads the shards' metadata, which took about half a minute and 1.2
GB for the synthetic metadata of a 200,000-token tokenizer, in one measurement (not on the released
programs).

On the tests' synthetic model (0.53 GB, 38 tensors), one run each on one machine (not on the
released programs): `bin/weight_sums` with one thread, 13.1 s and 6.8 MB; `weight_sums.py` with one
worker, 12.1 s and 0.44 GB, and with `--decode-twice`, 15.8 s and 0.20 GB. There are no figures at
full size or on a real model's weights.

Exit status 26 means that memory ran out, or that the operating system killed a worker: use fewer
threads or workers, or add `--decode-twice`. The operating system may also kill the whole program,
which then gives no exit status of its own.

## Tests

```sh
weight_sums/tests/run_tests.sh
```

It builds `bin/weight_sums`, replacing any that is there, and runs every test in a few minutes, on
synthetic files in a temporary directory that it removes at the end, also after a failure, SIGINT or
SIGTERM, but not after SIGKILL. It needs g++ 13 or later, python3, about 0.6 GB of free space and no
GPU; the steps for `weight_sums.py` need numpy and gguf-py 0.19.0 and are skipped without them. The
steps `model.cpp` and `model.py` need an x86-64 CPU with the SHA instructions, and fail without
them. It ends with exit status 0 and the last line
`run_tests: every step that ran passed; 0 skipped`, or status 77 if a step was skipped, 1 if a step
failed or the script cannot make its temporary directory, and 2 after an unknown option or for a
`TMPDIR` that it refuses. `--keep` keeps the temporary directory, `--without-unit-test` leaves out
the unit tests and counts them as skipped, `TMPDIR` sets where the directory is made, and
`MAKE_MODEL` names a built `make_model`. This is the row `weight_sums` of the tree's test table,
`SUITE.md`.

Each program must write exactly the expected file for `tests/oracle.py`'s shards, whose expected
file that script computes with exact rational arithmetic and no floating point, at every storage
type (its decoder also follows public sources: see "Shared errors" above), and for `tests/edge.py`'s
tensors at the exponent's rounding boundary. Each must refuse `tests/refusals.py`'s inputs with the
listed exit statuses. On a synthetic model of 0.53 GB each must write the sums file that
`make_model` (`prover/tests/weights_blocks/make_model.cpp`) writes with a decoder from the prover's
sources.
