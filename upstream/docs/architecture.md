# Architecture

FlashProve is a proof system for language-model inference. A provider commits to a model's weights
once. For each reply it serves, it produces a proof that the reply's tokens satisfy an agreed rule,
the [acceptance policy](glossary.md#acceptance-policy), under the committed weights. A client checks
the proof on a CPU, without the weights.

To build and run it, start with [quickstart.md](quickstart.md). To check a proof, read
[verifying.md](verifying.md). [glossary.md](glossary.md) defines the terms.

## Who runs what

| Party | Program | Runs on | Directory |
|---|---|---|---|
| Provider | `cp_commit` commits to the weights, once | one H100 | `prover/` |
| Provider | `cp_prove` proves served replies, in batches | one H100 | `prover/` |
| Provider | a [front end](glossary.md#front-end) between clients and `cp_prove`. The example, `proof_middleware.py`, is middleware inside an unmodified vLLM | the model server | `integrations/` |
| Client | `cp_verify` checks one proof per call | CPU | `verifier/` |
| Client | `check_proof.py` runs `cp_verify` and compares the files and the result against the client's own values | CPU | `integrations/` |
| Client | `lean-verify` runs the executable form of `verify`, the Lean verifier that the soundness theorem `sound_of_verify_at` is about, on a real full-size proof | CPU | `tools/lean-verify/` |

`cp_commit`, `cp_prove` and `cp_verify` share one library, `blocks/`. `cp_commit` also runs
`cp_verify` on its own output. `cp_prove` does not.

Without an H100, you can check the example proofs in `examples/` with `cp_verify`, and make and
check [toy proofs](glossary.md#toy-proof) with `blocks/first_look.sh`. Proving a model's reply needs
the GPU.

## What a proof states

A proof is about the [integer model](glossary.md#integer-model): the network with quantized weights,
computed in integers only.

A [request](glossary.md#request) file fixes the statement. The file holds a window of T token ids
(the prompt followed by the reply) and two numbers, r and e. Positions are numbered from 0. At
position t the model predicts token t + 1, so a proof over positions r to e − 1 checks tokens r + 1
to e, the [checked tokens](glossary.md#checked-token).

For example, with a prompt of 3 tokens and a reply of 2, r = 2 and e = 4, and the checked tokens are
tokens 3 and 4, which are the reply. In general, a prompt of n tokens and a reply of m need r = n −
1 and e = n + m − 1.

If the prompt and reply are shorter than T, filler tokens pad the window, and the client's file must
use the same filler as the front end's ([how to build the
file](verifying.md#the-request-and-the-token-file)).

The programs support two acceptance policies, and the [parameter file](glossary.md#parameter-file)
(`policy.txt`) selects one.

- **[Exact match](glossary.md#exact-match).** Every checked token must be the committed integer
  model's top choice at its position.
- **[Clipped gap](glossary.md#clipped-gap).** Each checked token gets a cost: its [score
  gap](glossary.md#score-gap) plus a constant. At most `cap` of the costs may exceed `C`. Each cost
  is then clipped to `C`, and the clipped costs must average at most `theta`. So some of the checked
  tokens, up to a number that `cp_verify` prints, can be any token at all. Every other token costs
  at most `C`, which rules out nothing when `C` is large or the model's scores lie close together,
  and a verifier cannot tell when that is so ([limits.md](limits.md#within-limits)). This scoring
  follows Token-DiFR's at temperature 0 ([arXiv 2511.20621](https://arxiv.org/abs/2511.20621)), with
  four differences, which [PUBLISHED-WORKS.md](../PUBLISHED-WORKS.md#a-published-method) lists.

The [commitment ID](glossary.md#commitment-id), a hash that identifies the commitment, covers the
parameter file but does not tell you which policy the file sets, so tell `cp_verify` which policy
you expect. Unless you pass `-policy`, it assumes exact match ([when it rejects the
call](../verifier/docs/output.md#rejection-names)).

A proof says nothing about tokens outside the range, with one exception. If the range runs to the
end of the window (e = T), the last position predicts the token after the window, called the [next
token](glossary.md#next-token). The policy does not apply to it, so in this case the checked tokens
end at token T − 1. Instead, under either policy, the proof states that the next token is the
integer model's top choice.

[verifying.md](verifying.md#what-a-yes-means) explains in full what an accepted proof means.

## Before you rely on it

This is research code. These five cautions are the main ones; [limits.md](limits.md) lists the known
limits, and may be incomplete.

- **Soundness.** Nothing of soundness has been proved about the three programs themselves
  (`cp_commit`, `cp_prove`, `cp_verify`). The error bounds in bits that they print are the programs'
  own calculation, not a security level, and at the default setting some of the bits count proof of
  work ([what the figures are worth](limits.md#the-soundness-figure)).
  [guide/what-is-proved.md](../guide/what-is-proved.md) says what is proved, and about what.
- **Weight hiding.** By design, the programs mask proofs and a commitment's public files, apart from
  a [declared list](what-is-not-hidden.md), and they do so by default. What is proved of it, and how
  far that reaches: [guide/weight-hiding.md](../guide/weight-hiding.md). What a provider must do:
  [operators-duties.md](operators-duties.md).
- **The committed model.** The provider chooses the weights and writes the configuration, so no
  acceptance policy can show that the committed weights are the model the provider advertises, or
  even a sensible model.
- **The integer model.** A proof does not say how close the integer model is to the floating-point
  model.
- **What has run.** [limits.md](limits.md#what-was-run-on-these-programs) says which results come from
  these programs and which do not. The full-size
  runs used synthetic weights or a third party's 4-bit, 8-bit or 16-bit (BF16) conversion of Llama 4
  Maverick's weights. On synthetic weights, and in the runs through vLLM, the requests were token IDs
  and not text. [guide/performance.md](../guide/performance.md) gives what was measured.

## From weights to a checked proof

```text
Provider, once:
  model files -> scales/, weight_sums/ -> quantization scales, checksum file
  weights + configuration -> cp_commit -> public key directory, commitment ID  (published)
                                       -> secret key directory                 (kept secret)

For each reply:
  client: prompt -> front end -> engine -> reply -> client
  front end: request file (prompt and reply tokens, r, e) -> cp_prove
  cp_prove, for each batch -> proof file -> front end
  front end: cp_verify on every proof of the batch -> proof file -> client

Client, for each proof:
  cp_verify on the inputs listed under "The files" -> YES or WITHIN LIMITS; any other result: not accepted
```

The steps below describe what the programs are designed to do.

1. **Prepare.** The provider writes a base parameter file. The program in `weight_sums/` writes a
   checksum file, and the provider adds `fold.txt`, a two-line file. Next, the tools in `scales/`,
   which read the checksum file, choose the model's quantization scales and write them into `KL.bin`
   and the final parameter file (`policy.txt`). Last, `prover/tools/pack_layout.py` writes the
   layout file (`layout.txt`). `policy.txt`, `KL.bin` and `layout.txt` make up the
   [configuration](glossary.md#configuration). `cp_commit` and `cp_prove` need the checksum file and
   `fold.txt` in the same directory as the model's weight files
   ([weight_sums/README.md](../weight_sums/README.md)). Keep the checksum file as secret as the
   weights.
2. **Commit.** `cp_commit` reads the weights and the configuration, encodes the weights as a [weight
   table](glossary.md#weight-table), and builds a Merkle tree over it. This first table is the
   [anchor](glossary.md#anchor), and the commitment ID names it. `cp_commit` produces a one-time
   proof that the anchor's entries are within their ranges, although for the embedding that range is
   wider than the model's own ([limits-further.md](limits-further.md#so-11)). `cp_commit` then makes
   M [leaf tables](glossary.md#leaf-table) (`-M`) and ties each to the anchor with a proof called
   the link. Before it writes each public file, it has `cp_verify` check the file's bytes.
3. <a id="publish"></a>**Publish.** The provider publishes the configuration, the public key
   directory (`ivk`) and the commitment ID. If a provider could choose the commitment after seeing a
   prompt, it could commit to a small model that gives the answer it wants, and prove that honestly.
   So a client must have the ID before it sends its prompt, and must not get it through the server
   that it is checking, alone or with the files. Otherwise an accepted proof says nothing about
   which model the tokens were checked against.
4. **Serve.** The client sends a prompt, and an engine serves the reply. In [ordinary
   serving](glossary.md#ordinary-serving), the expected mode, vLLM or any other engine serves the
   floating-point model. The proof is made afterwards, as one prefill pass over prompt and reply,
   under the clipped-gap policy ([what a client learns under each policy](limits.md#which-policy)).
   In [integer serving](glossary.md#integer-serving), the engine runs a package designed to compute
   the integer model, so that a served reply can be proved under exact match.
5. **Prove.** The front end writes a request file from the token ids that the engine reports that it
   read and produced. `cp_prove` takes a batch of 1 to 64 requests and picks one leaf table for it
   at random. If a batch runs to completion, each of its requests gets either a proof file and a
   token file, or an empty `.failed` file. The token file holds the next token, or 0 if the range
   ends before the window does.
6. **The provider's check.** A response is a request's proof file and token file. `cp_prove` runs
   no verifier on what it makes: once a batch's responses are staged, they get their final names.
   So the provider's front end runs `cp_verify` on every response of a batch before it gives any of
   them to a client. The serving scripts that ship (the vLLM integration, `driver/measure_rate.py`, the example scripts)
   hand on no proof that `cp_verify` has not accepted, and the vLLM integration goes by the batch.
   This check is the provider testing itself, so clients must still verify. Nor is it a check on
   hiding: it catches only proofs that `cp_verify` rejects. `cp_verify` does not check that
   [masks](glossary.md#mask) are present, and for most masks it cannot, so a proof whose mask was
   lost before the prover committed to the masked value still passes
   ([limits.md](limits.md#a-lost-mask)).
7. **Verify.** The client builds its own request file from the token ids it sent and received, and
   its own token file ([the files](#the-files)). It then calls `cp_verify` with `-id` and, under
   clipped gap, `-policy`. The proof is accepted only if the exit code is 0 and the last line is
   `YES` or `WITHIN LIMITS: …`. `cp_verify` prints `WITHIN LIMITS` rather than `YES` if any token of
   the request was checked under clipped gap. The last line does not show whether the result counts
   proof of work, or what the binding of the weights rests on: read the lines `proof of work:` and
   `commitment:` too ([when to accept](verifying.md#when-to-accept)).
   [verifying.md](verifying.md#check-a-proof) has the commands, and [its
   table](verifying.md#exit-codes) lists the last lines and exit codes.

**Why an anchor and leaf tables.** The commitment ID is computed from the anchor, not from the leaf
tables, so it is the same for every leaf table of a commitment. Proofs are made against leaf tables
because each batch opens some columns of its table, and a table's masks cover only a fixed number of
opened columns, its [limit](glossary.md#limit) ([limits.md](limits.md#the-limit-of-use)). As soon as
any leaf table runs out of room for another batch, `cp_prove` exits, and the service under that
commitment ends. So [watch the counts](operators-duties.md#the-count), and make and publish a [new
commitment](glossary.md#new-commitment) before then. If anything has leaked by then, a new
commitment does not undo it.

**One batch inside `cp_prove`.** The top of `prover/prover/service.cuh` describes these steps.

1. It takes the oldest requests from the [spool directory](glossary.md#spool-directory) and fixes
   their order.
2. It samples 32 fresh random bytes. It uses these bytes, the [seed](glossary.md#seed) and the
   requests to pick the batch's leaf table. It adds the batch's columns to that table's
   [counter](glossary.md#counter) before it reads any of the table's files. Then it derives the key
   for the batch's randomness.
3. It proves the requests, `-in-step` of them per pass over the weights.
4. It opens the columns of the leaf table that the batch needs, once for the whole batch. This takes
   two more passes over the weights.
5. For each request, it stages either a proof file and a token file, or a `.failed` file.
6. All the batch's responses get their final names together.

In the secret key directory, `cp_prove` holds a lock while it runs and appends to the counter of the
batch's leaf table. It writes its results to the directory named with `-out` or `-spool`, and its
messages to stdout and stderr.

**Why step 4 needs two passes.** The first pass folds the table's rows at a point sampled only after
every request in the batch has finished its first [chain](glossary.md#chain). Next the prover
commits to the folded row, and only then samples the positions of the columns the second pass
computes. The order of the messages is at the top of `blocks/iop/lin.h` and `blocks/iop/foldeq.h`.

A front end must give each client only its own response. The example front end has no notion of
clients. It hands a proof to anyone who has the proof ID and has the shared API key where vLLM was
given one that the middleware can read (how it learns that key, and where it cannot: [the operator's
checklist](../integrations/vllm/REFERENCE.md#the-operators-checklist); what was run: row H11.2 of
[the table](limits-all-rows.tsv)). [operators-duties.md](operators-duties.md) covers the duties of a front
end and an operator.

## The files

A client needs these inputs to run `cp_verify`.

```text
cp_verify <configuration directory> <public key directory> <request file> <token file> <proof file> -id <commitment ID> [-policy <expected policy>]
```

| Input | Made by | The client gets it from | Trusted, or checked by the call |
|---|---|---|---|
| commitment ID (`-id`) | `cp_commit` | the party that vouches for the model, not through the server you are checking; best where the commitment is published; before sending the prompt | trusted |
| the expected policy (`-policy`): its kind and values | the provider, which sets them in `policy.txt` | the same source as the ID; before sending the prompt | trusted |
| tokenizer, chat template, end-token ids | the model's publisher | the place where the model is published; before sending the prompt | trusted; neither the ID nor a proof covers them |
| the `cp_verify` program itself | the client, who builds it | this repository | trusted; the ID does not cover any part of the program |
| `policy.txt`, `KL.bin`, `layout.txt`: the configuration | the provider | anyone | identified: the ID covers every byte. `cp_verify` tests it, but no test shows that the model it defines is the advertised one |
| `ivk/key.bin`: the [key file](glossary.md#key-file) | `cp_commit` | anyone | checked by `cp_verify`: the ID covers the anchor's root and shape, its [out-of-domain (OOD) answers](glossary.md#record) (masked sums of the weights) and its OOD seeds, and the call checks the proofs in the file |
| `ivk/leaf/<k>.bin`: the [leaf file](glossary.md#leaf-file) of the leaf table that the proof names | `cp_commit` | anyone | checked by `cp_verify`: the call checks its link to the anchor |
| `<name>.req`: the request | the client | its own record of the tokens sent and received | the client's own |
| `<name>.y`: the token file | the client (`cp_prove` writes its own with each proof) | its own record: the next token it was served, or 0 if the range ends before the end of the window | the client's own |
| `<name>.proof`: the proof file | `cp_prove` | the provider | checked |

The commitment ID is a single hash that covers every byte of the three configuration files, plus the
anchor's root and shape and its [OOD answers](glossary.md#record) and OOD seeds. The function
`published_id` in `blocks/frame/heads.h` computes the ID, and `cp_commit` and `cp_verify` both call
it. The ID does not cover the proofs inside `key.bin`, or the leaf files. `cp_verify` checks those
when it reads them ([what is known about such files](limits.md#what-the-id-does-not-cover)).

Never publish the [secret key directory](glossary.md#secret-key-directory) (`isk`), and never copy
or restore it. It holds the seed, the configuration as committed and, for each weight table, its
files and its counter. Anyone who reads the directory can remove the masks from opened columns and
test any guess at the weights, and that cannot be undone. A restored copy resumes from an old
counter, so more columns could be opened than the masks cover ([backups and restored
copies](operators-duties.md#backups-and-restored-copies)).

## The proof system in outline

The proof system is hash-based and non-interactive: an interactive oracle proof compiled with Merkle
trees and Fiat-Shamir. This outline says what each part is designed to do.
[protocol.md](protocol.md) describes the protocol, and
[guide/what-is-proved.md](../guide/what-is-proved.md) says what has been proved.

- **Field and hash.** Arithmetic is in the Goldilocks field, p = 2^64 − 2^32 + 1, and its cubic
  extension. A proof's Merkle trees and its chains, which are its Fiat-Shamir transcripts, use
  BLAKE2b. A second, fixed hash function, built on BLAKE2b's compression function, produces some
  digests that go into a proof's statement ([hash mode](glossary.md#hash-mode)).
- **Weight commitment.** Each row of a weight table holds weights together with random entries,
  called masks, and is encoded with a Reed-Solomon code. The table's columns are the leaves of a
  Merkle tree.
- **A request's proof.** A proof file holds three chains. The first belongs to the request. It works
  through the model [segment](glossary.md#segment) by segment, reducing every step to claims about
  what the proof has committed to and about the weight table. The second is shared by the batch, and
  opens columns of the table once for all the batch's requests. The third again belongs to the
  request, and completes the request's part of that opening.
- **Challenges.** The verifier does not send any. Instead, each challenge is computed from a hash
  over its chain's [first query](glossary.md#first-query) and the earlier prover messages that enter
  the chain. The first query holds the digest of the configuration files, the plan of the chain's
  [rounds](glossary.md#round), and the statement. Two things enter that hash late or never, and both
  are known limits. The next token does not enter the chain until just before the output head's part
  of the proof, so the statement in the first query omits the next token
  ([SO-50](limits-further.md#so-50)). Opened columns, salts and Merkle paths never enter it (row
  SO-53 in [the table of all rows](limits-all-rows.tsv)).

## Repository layout

| Directory | Contents |
|---|---|
| `blocks/` | the library: field, hash, codes, the steps of the proof system, file formats, error accounting; toy proofs |
| `verifier/` | `cp_verify`, its build script and its tests |
| `prover/` | `cp_commit` and `cp_prove`: GPU sources (CUDA kernels and host code), build script, tests, and example configurations in `prover/examples/` ([its README](../prover/README.md) maps its directories) |
| `scales/`, `weight_sums/` | tools that choose a model's quantization scales and write the checksum file that `cp_commit` and `cp_prove` need |
| `driver/` | the operator's side of the spool directory, `driver/spool.py` (not the same program as `integrations/spool.py`); `driver/measure_rate.py`, which measures the rate of proving ([driver/README.md](../driver/README.md#the-rate)) |
| `tools/` | `lean-verify`, which runs the Lean verifier's executable form on a full-size proof (a long build first: [its README](../tools/lean-verify/README.md)); the sandbox that the tests run in; an audit tool that re-derives a commitment's random values, or one batch's, from the secret key directory (it only works on synthetic weights) |
| `toy/` | small test scripts, including `quick_start_test.py` |
| `tests/`, `run_suite.py`, `SUITE.md` | tests and shared test helpers, the script that runs them, and the table of tests |
| `examples/` | four stored proofs on synthetic weights, two small and two at [full size](glossary.md#full-size), each with what a client needs to check it; a script that runs `cp_verify` on altered files; stored toy files for `blocks/`; and `quickstart-gpu.sh` ([examples/README.md](../examples/README.md) lists them). This repository stores no false files for `cp_verify` made by cheating programs, so the trials with such files (row E28) cannot be repeated from it |
| `examples/maverick/` | how to commit to public weights of Llama 4 Maverick, in a third party's 4-bit conversion: where to download them, their checksums, a script that goes from the download to a checked proof, and the commitment ID to compare the result with ([its README](../examples/maverick/README.md)); no weights |
| `docs/` | the docs ([where to read on](README.md)) |
| `integrations/` | the example front end for vLLM (`proof_middleware.py`) and its library `spool.py`, whose `make-request` command is for clients; the client's script, `check_proof.py`; an example client, `ask_and_check.py`; a demo, `vllm/end_to_end/end_to_end.py`; the package designed to let vLLM run the integer model, `vllm/integer_model/`; and the limits of all of these, `LIMITS.md` |
| `lean/` | the Lean library ([what it proves](../guide/what-is-proved.md); its soundness theorem on `verify`: [lean/THEOREM.md](../lean/THEOREM.md)), and in `lean/OnBytes/` Lean text that follows `cp_verify`'s source (see below the table) |
| `licenses/` | third-party licence texts |
| `paper/` | `paper.pdf` is the paper: the protocol, what is proved of it in Lean and what is not, and the measurements |

Model weights are not part of the repository. Nor is any key file, leaf file or proof that was made
from a real model's weights. [THIRD_PARTY.md](../THIRD_PARTY.md#what-depends-on-metas-models) lists
the values here that depend on a real model, such as checksums of its files.

<a id="lean-onbytes"></a>**`lean/OnBytes/`.** This folder holds Lean text that writes `cp_verify`'s
checks again, following `cp_verify`'s source ([what it holds](../lean/OnBytes/README.md); [its
limits, and where it is known to differ from `cp_verify`](../lean/OnBytes/LIMITS.md)). That second
writing is unproved, and this repository builds no stand-alone program from it; one test
(tools/accepts/) compiles it and has Lean load the compiled code while it evaluates files of
lean/Ask/. Other Lean files, those of `verify`'s executable form among them, import definitions from
this folder; what `verify` takes from it is read with `verify`'s statement
([lean/GAPS.md](../lean/GAPS.md), 58). No theorem connects the second writing to `cp_verify`, and
the soundness theorem for the Lean verifier is not about it. These docs call the second writing
`verify-on-bytes`; where they tell a run of a stand-alone program built from it, that program was a
yardstick for tests: no soundness theorem reaches a call of it that checks a real proof, and no
theorem bounds the chance that it accepts a false file. Because it follows `cp_verify`'s source
text, the two are not independent, one witness and its port, not two witnesses: where they agreed,
that is not independent evidence, and a mistake that both share does not show.

## Where to read more

- **File format:** [format/README.md](format/README.md) points to the specification of the bytes and
  the checks. Section 17 of `FORMAT.md` describes what it leaves undefined and where it is known to
  differ from the programs. For bytes, hashed strings, ordering and checks, the specification takes
  precedence over every other page. If a program disagrees with it, please report a bug.
- **The `blocks/` library:** [blocks/README.md](../blocks/README.md).
