# Glossary

**In short.** This page is a reference, not something to read through. It has two lists: the terms
used in these docs, A to Z, and [terms for readers of the source code](#words-in-the-source-code).
An entry says what a thing is or is meant to do. Where there is a fuller account, the link at its
end leads there. For what has actually been proved or is enforced, see [guide/what-is-proved.md](../guide/what-is-proved.md) and [limits.md](limits.md). A
row, such as E30 or SO-25.1, is a line of [the table of all rows](limits-all-rows.tsv).

(The terms for what Lean proves: [lean/TERMS.md](../lean/TERMS.md).)

<a id="standard-terms"></a>**This project's words and the standard terms.** Some words on these
pages, in the programs' files and in the Lean library are this project's own. The table gives the
standard cryptographic term for each. A committed table's *OOD sample* (out-of-domain sample) is a pair: its
*OOD seeds*, one for each answer a row, each a point and a factor for the mask, hashed from the
table's name, and its *OOD answers*, the committer's masked answers at these, which are published
with the table's root. (In the literature the sample is the verifier's point alone.) The programs,
their files and the format pages say "the record" for the published answers and "the record's seeds"
for the seeds.

| On these pages and in the files | The standard term |
|---|---|
| [record](#record) (the format, the programs) | a committed table's OOD answers; with the OOD seeds, which are derived by hash and stored nowhere, they make up its OOD sample. The OOD seeds are public, anyone can compute them, and they are not the provider's secret [seed](#seed). Lean's type `Rec` (`Iop.HolographicProtocol.Rec`, `lean/Hiop/RecInterface.lean`) is the pair of seed and answer |
| the record's answers; masked sums of the weights | OOD answers |
| what was sampled for the record; the record's seeds | OOD seeds |
| [anchor](#anchor) | first table |
| [leaf table](#leaf-table) | copy |
| [weight table](#weight-table), stored table | committed table |
| [leaf file](#leaf-file); in the Lean library: entry file | the copy's file: the copy's root, its OOD answers and its link proof |
| [link](#link) | link proof: a proof that two committed tables encode the same weights |
| description (of the protocol, in Lean) | the interactive oracle proof (IOP) as written in Lean |
| [plan](#plan), planner | the list of rounds with each round's error bound; the program that makes it |
| [statement on codes](#statement-on-codes); [theorem on codes](#theorem-on-codes) | mutual correlated agreement (MCA) for Reed–Solomon codes up to the Johnson bound; Theorem 4.6 of ePrint 2025/2055; a weaker fall-back: ePrint 2025/2051, Lemma 9.3 (row LIM-17.2) |
| [range check](#range-check), `smallness` | approximate range proof |
| refuse; draw; control | reject; sample; negative test |

## A

- <a id="acceptance-policy"></a>**Acceptance policy.** The rule that a request's [checked
  tokens](#checked-token) must satisfy. The programs support two policies: [exact
  match](#exact-match) tests each token separately, and [clipped gap](#clipped-gap) scores them
  together. Seven lines of the [parameter file](#parameter-file) set the policy (`C`, `theta`, `cap`, `s`, `sh`, `sT`, `F`: [what each means](../verifier/docs/parameter-file.md#8-output-head-and-acceptance-policy)). Of these, `C`, `theta` and `cap`, read with `sh`, decide which kind it is. The caller names the policy it expects with `-policy`.
  ([verifying.md](verifying.md#acceptance-policies))
- <a id="anchor"></a>**Anchor.** A [commitment](#commitment)'s first [weight table](#weight-table).
  `cp_prove` never proves a request against the anchor.
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))

## B

- <a id="batch"></a>**Batch.** A group of requests that `cp_prove` proves together against one [leaf
  table](#leaf-table). ([service.md](service.md#options-of-the-service))
- <a id="binding"></a>**Binding.** The property that a published key fixes at most one
  [index](#index), or none. Binding of the weights is meant in a modified sense, in which the hash
  itself supplies the [OOD seeds](#record). The written argument for that sense, in a technical
  report that is not public and not refereed, covers a commitment ID formed without those seeds, not
  the ID that the programs form, which hashes them. So of the programs' own commitment no binding
  theorem is proved. What Lean proves of the Lean verifier is that any commitment, with its
  committer's hash queries, settles at the given public files one set of weights or none before any
  proof (index binding: [lean/TERMS.md](../lean/TERMS.md)). Of an illustrative committer written in
  Lean, which models one part of `cp_commit` (no theorem ties the two) and was compared with it in
  values at small size only, Lean proves, at public files where [`verify`](#verify)'s tests on them
  pass, that the weights so settled are the ones put in, at an error of about 2^-89 at that
  committer's own query count, and that the commitment cannot be opened to other weights, at about
  (t + 2^23)/2^124 after t hash queries ([What is proved](../guide/what-is-proved.md);
  [lean/THEOREM.md](../lean/THEOREM.md#what-the-commitment-gives-as-proved)). Of the compiled system
  (the same verifier, `verify`, with the committer that the compiler yields from the Lean provers)
  Lean proves, at public files in use, soundness with binding, except with probability about
  q/2^124 + 2^-100 after q hash queries, and that within that committer's own budget of 2^48 hash
  queries the extractor gives back the committed weights, except with probability about 2^-155; no
  theorem ties that committer to `cp_commit` ([What is proved](../guide/what-is-proved.md)). In
  these programs, a key bounds the embedding's entries only loosely, and nothing shows what the
  padding cells hold ([SO-11](limits-further.md#so-11), [SO-22](limits-further.md#so-22)).
- <a id="blocks"></a>**`blocks/`.** The library that all three programs build on: field, hash,
  codes, the steps of the proof system, file formats and [error accounting](#error-accounting). It
  also runs by itself on a CPU, at toy size.
  ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="proved-lemma"></a>**Bound within a third of the distance.** A lemma proved in the paper
  that this protocol builds on; these pages claim no proof of that lemma itself in this repository.
  The Lean library under lean/ also proves a statement of its own of that kind (`Cb.ld_mca_third`:
  mutual correlated agreement within a third of the distance, for any code that is linear as the
  protocol reads it (Lean: LdCode.Laws, lean/Cb/Ld/LdRel.lean: encoder and reading map linear,
  encoder injective, reading map reading only declared positions)), and no theorem ties it to the
  programs' count. What the programs count on is an adapted form of that lemma, which rests in part
  on proofs in a technical report that is not public and not refereed ([what that bound rests
  on](limits.md#what-the-theorem-assumes)). At the default setting no closeness round rests on it:
  the planner counts none on the bound within a third of the distance. (Row LIM-11;
  [verifier/docs/output.md](../verifier/docs/output.md#the-commitment-line);
  [verifier/README.md](../verifier/README.md#the-error-accounting))

## C

- <a id="c-theta-cap"></a><a id="theta"></a><a id="cap"></a>**`C`, `theta`, `cap`.** Three lines of
  the [parameter file](#parameter-file) that set the limits for [clipped gap](#clipped-gap).
  ([verifying.md](verifying.md#acceptance-policies))
- <a id="chain"></a>**Chain.** One [Fiat-Shamir](#fiat-shamir) transcript: a sequence of hashes that
  absorbs a first query with the public inputs, and then the prover messages that enter the chain.
  The challenges are computed from it. It does not cover everything from the start, though. The
  [next token](#next-token) enters only before the output head's part of the proof
  ([SO-50](limits-further.md#so-50)), and opened columns, salts and Merkle paths never enter (row
  SO-53 in [the table of all rows](limits-all-rows.tsv)).
- <a id="check-proof"></a>**`check_proof.py`.** A script for clients
  (`integrations/check_proof.py`). It runs `cp_verify` and compares the files and the result with
  the client's own values. ([../integrations/README.md](../integrations/README.md))
- <a id="checked-token"></a>**Checked token.** A token of the request that a checked position
  predicts, and that the [acceptance policy](#acceptance-policy) applies to. A request names
  positions r to e − 1, counted from 0, and the model's output at position t predicts token t + 1.
  With a prompt of 3 tokens and a reply of 2, r = 2 and e = 4, so tokens 3 and 4 are checked. If e =
  T, the last position predicts the [next token](#next-token), which comes after the
  [window](#window) and is treated differently.
  ([architecture.md](architecture.md#what-a-proof-states))
- <a id="checksum-file"></a>**Checksum file.** `wsums.txt`. For each weight tensor that is
  committed, it holds the tensor's exponent and the SHA-256 of its integers. It is written by the
  program in `weight_sums/`, whose README calls it the sums file. Keep it as secret as the weights.
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof), step 1;
  [../weight_sums/README.md](../weight_sums/README.md))
- <a id="client"></a>**Client.** The party that sends a prompt, receives a reply and checks the
  proof.
- <a id="clipped-gap"></a>**Clipped gap.** An [acceptance policy](#acceptance-policy) that scores
  the [checked tokens](#checked-token) together. At most `cap` of their [costs](#cost) may exceed
  `C`. Each cost is then clipped to `C`, and the clipped costs must average at most `theta`. A proof
  accepted under this policy does not show that the reply is the model's own.
  ([limits.md](limits.md#within-limits); how much a pass allows: row SO-25.1) Where the pages tell of Lean's runs, they call the clipped-gap policy stored in `examples/full-size-clipped-gap/` "the tolerant policy".
- <a id="closeness-round"></a>**Closeness round.** A round in which a challenge combines several
  rows into one. The main term in its error is the chance that the combination is close to the code
  even though the rows are not jointly close. At the [default setting](#default-setting) every
  closeness round rests on [the theorem on codes](#theorem-on-codes), taken as given in the programs'
  accounting. `cp_verify` prints "closeness" for such rounds.
- <a id="column"></a>**Column.** One position across all rows of an encoded table, and one leaf of
  its Merkle tree. A proof opens (reveals) some columns of its [leaf table](#leaf-table).
  ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="commitment"></a>**Commitment.** The output of `cp_commit`, made once from weights and a
  [configuration](#configuration): the [public key directory](#public-key-directory), the [secret
  key directory](#secret-key-directory) and the [commitment ID](#commitment-id).
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="commitment-id"></a>**Commitment ID.** 64 hex digits (32 bytes) that identify one
  [commitment](#commitment): a hash over the three [configuration](#configuration) files' digest,
  the [anchor](#anchor)'s name (root and shape), the OOD seeds and the OOD answers (by design; the soundness theorem on `verify` is stated at given public files and does not itself say that one ID cannot verify under two sets of them — so hold the three configuration files as you hold the ID). `cp_verify` takes it with `-id`. Get it before you send your prompt,
  from the [party that vouches for the model](#vouching-party), not through the server you are
  checking; best from where the commitment is published. Never take it from the files that the
  server sent. ([verifying.md](verifying.md#what-you-need-and-where-it-must-come-from))
- <a id="committer-word"></a>**Committer.** See [`cp_commit`](#cp-commit).
- **Conditions (the provider's).** See [duties](#duties).
- <a id="configuration"></a>**Configuration.** Three public files for one model shape: `policy.txt`,
  `KL.bin` (most of the [quantization scales](#quantization-scales)) and `layout.txt` (where a proof
  places each array). A configuration directory holds them. `cp_commit` takes it with `-config`, and
  `cp_verify` as its first argument. ([architecture.md](architecture.md#the-files))
- <a id="constant-committer"></a>**Constant committer.** A committer that returns one constant
  whatever the weights, and so commits to nothing; the committer of the six lines (which six:
  [limits.md](limits.md#what-the-lean-proofs-cover); the system that has it: row PS-71) is another:
  the one that `compile` makes (`lean/Johnson/Face/Compiler.lean`), which encodes the weights and
  hashes the key.
- <a id="cost"></a>**Cost.** A [checked token](#checked-token)'s [score gap](#score-gap) plus a
  constant, 2^`sh`. The [top choice](#top-choice) therefore costs 2^`sh`, not 0. `sh` is a line of
  the [parameter file](#parameter-file). ([verifying.md](verifying.md#acceptance-policies))
- <a id="count-bin"></a>**`count.bin`.** See [counter](#counter).
- <a id="count"></a>**Count.** In [operators-duties.md](operators-duties.md) and
  [faults.md](faults.md), a [leaf table](#leaf-table)'s [counter](#counter), or the number given by
  its length: how many columns have been shown.
- <a id="counter"></a><a id="batch-counter"></a>**Counter (`count.bin`).** One file per [weight
  table](#weight-table), in the [secret key directory](#secret-key-directory). It grows by one byte
  for each column that a proof reveals, including proofs that were started but never finished. No
  [production build](#production-build) ever shortens it, and an operator must not either.
  ([operators-duties.md](operators-duties.md#the-count))
- <a id="covered-line"></a>**`covered:`.** A line that `cp_verify` prints near the end of a [whole
  call](#whole-call) in which no stage rejected or failed. It repeats which positions the call was
  about. It also prints `above-clip`, the number of checked tokens that may cost more than `C` under
  [clipped gap](#clipped-gap); under [exact match](#exact-match) that number is 0.
  ([verifying.md](verifying.md#when-to-accept))
- <a id="cp-commit"></a><a id="committer"></a>**`cp_commit` (the committer).** The GPU program that
  commits to the weights, once. ([architecture.md](architecture.md#who-runs-what))
- <a id="cp-prove"></a><a id="prover"></a>**`cp_prove` (the prover).** The GPU program that answers
  requests with proofs, in batches, on one H100. ([architecture.md](architecture.md#who-runs-what))
- <a id="cp-verify"></a><a id="verifier"></a>**`cp_verify` (the verifier).** The program that checks
  one proof in one call, on a CPU, without the weights. ([verifying.md](verifying.md))

## D

- <a id="default-setting"></a>**Default setting.** The setting that the [parameter
  file](#parameter-file) `policy.txt` has in every [example configuration](#example-configuration)
  (two of them, `target_small_f2_gap` and `target_full_f2_gap`, with a clipped-gap policy in place
  of exact match). In it [proof of work](#proof-of-work) is counted toward the target ([what the
  figures are worth](limits.md#the-soundness-figure)).
- <a id="duties"></a><a id="conditions"></a>**Duties (the provider's).** What a provider must always
  do, or never do, so that the [masks](#mask) cover everything that is shown. No program can check
  most of them, and every proof verifies whether or not they are kept.
  ([operators-duties.md](operators-duties.md))

## E

- <a id="encoder"></a>**Encoder (Lean's own).** In the Lean library, the function that turns the
  weights and given random values into an encoded [weight table](#weight-table), before any hashing:
  it puts a [pad](#pad) at the end of each row's message and encodes the row together with random
  entries (its [tail](#tail)). It is part of the interactive oracle proof (IOP) as written in Lean.
  The programs have code of their own for this step, and no theorem ties the two.
- <a id="engine"></a>**Engine.** The program that serves replies, such as vLLM. It is not part of
  the proof system.
- <a id="error-accounting"></a>**Error accounting.** The programs' own calculation, from the
  [configuration](#configuration) files, of an error bound for each round. A round's error is the
  chance that its challenge lets a false claim survive. `cp_verify` prints the largest bounds and
  accepts only if each round reaches the target, counting any [proof of work](#proof-of-work). This
  is a calculation, not a proved [security level](#security-level).
  ([limits.md](limits.md#how-the-programs-count))
- <a id="exact-match"></a>**Exact match.** The [acceptance policy](#acceptance-policy) under which
  every checked token must be the [integer model](#integer-model)'s [top choice](#top-choice) at its
  position. `cp_verify` expects it for a request's [checked tokens](#checked-token) when `-policy`
  is not given. ([verifying.md](verifying.md#acceptance-policies))
- <a id="example-configuration"></a>**Example configuration.** A directory under `prover/examples/`
  with a ready-made [configuration](#configuration). `target_full_f2` and `target_full_f2_gap` are
  at [full size](#full-size), and the directories whose names begin with `target_small_f2` are for
  small synthetic models. One more directory, `target_small_file_f2`, is for tests on a model's
  files. It has no `layout.txt`, so it is not a complete configuration.
  ([quickstart.md](quickstart.md))
- <a id="exit-code"></a>**Exit code.** The number `cp_verify` returns: 0, `YES` or the weaker
  [`WITHIN LIMITS`](#within-limits); the link at the end lists the other codes. Read it together
  with the last line of output. The exit codes of `cp_prove` are listed in [service.md](service.md).
  ([verifying.md](verifying.md#exit-codes))
- <a id="extractor"></a>**Extractor.** In Lean's soundness statement, a fixed rule that gives
  weights, or none, from the commitment and the hash queries of whoever made it. It is a
  mathematical function defined with choice and is not shown to be computable. No theorem says that
  the weights that it gives are the ones that `cp_commit` was given. Where it gives none, every
  accepted proof counts as a failure. It does not show that the prover knew the weights.
  [lean/LIMITS.md](../lean/LIMITS.md#whose-weights) says what the function is defined from.

## F

- <a id="failed-file"></a>**`.failed` file.** An empty file, `<name>.failed`, which means that the
  request gets no proof. ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="fiat-shamir"></a>**Fiat-Shamir.** The technique that makes the protocol non-interactive.
  Instead of being sent by a verifier, each challenge is computed from a hash over its chain's first
  query and the earlier prover messages that enter the chain. A malicious prover can retry messages
  until a challenge suits it, which is why the bounds count hash queries. "Before hashing" means the
  interactive protocol; "after hashing", the argument made from it by Merkle trees and the
  Fiat-Shamir transformation, in the random-oracle model.
  ([limits.md](limits.md#the-soundness-figure))
- <a id="field"></a>**Field.** The programs compute in the Goldilocks field, p = 2^64 − 2^32 + 1,
  and its cubic extension. ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="filler"></a>**Filler tokens.** The tokens that fill up a [window](#window) when prompt and
  reply are shorter than T. They are part of the model's input, so the client and the [front
  end](#front-end) must use the same ones.
  ([verifying.md](verifying.md#the-request-and-the-token-file))
- <a id="fitted-scales"></a>**Fitted scales.** [Quantization scales](#quantization-scales) chosen
  for a model with the tools in `scales/`. They are public and are computed from the weights, so the
  [configuration](#configuration) files reveal the largest weights of many weight matrices to within
  small factors, and anyone can compare a guessed model against them. The docs declare that this is
  not hidden. ([what-is-not-hidden.md](what-is-not-hidden.md#fitted-settings))
- <a id="fixed-message-provers"></a>**Fixed-message provers.** In Lean's results on hiding: provers
  that send the same messages whatever the weights, and so show nothing of them. At public files in
  use (row PS-71) the protocol's honest provers are other provers; they fall back on the
  fixed-message ones only where a function that makes an honest prover refuses. Fixed-message
  provers stand in the place of some or all of the protocol's own three provers (key file, link
  proof, request). Fixed messages hide the weights for free, so on the provers' own messages a
  result on hiding counts only beside completeness of the same provers.
  ([limits.md](limits.md#what-lean-proves))
- <a id="front-end"></a>**Front end.** The provider's program between clients and `cp_prove`. The
  example is `integrations/proof_middleware.py`, a middleware inside an unmodified vLLM.
  ([operators-duties.md](operators-duties.md#the-front-end))
- <a id="full-size"></a>**Full size.** The shape of Llama 4 Maverick 400B, in integers: 48 layers,
  128 routed experts, a [window](#window) of 1,024 tokens (`prover/examples/target_full_f2`,
  `prover/examples/target_full_f2_gap`).

## H

- <a id="hash-mode"></a>**Hash functions (hash mode).** A proof uses two hash functions. The first, standard BLAKE2b
  on labelled strings, hashes the [chains](#chain), the trees and the [commitment
  ID](#commitment-id). The other is a second, fixed hash function, built on BLAKE2b's compression function. It produces nine digests that go into a proof's statement, where the request file's own
  words stand too: those of the request file, `KL.bin`, the [parameter file](#parameter-file)'s
  three lists of lookup lines, and four public tables. It also produces digests that the programs
  print or store. It is outside BLAKE2b's published analysis. [The
  hash](limits.md#what-the-theorem-assumes) says what soundness asks of it.
  ([format/README.md](format/README.md))
- <a id="hiding"></a>**Hiding.** See [weight hiding](#weight-hiding).
- <a id="honest-run"></a>**Honest run.** A run in which no one has changed the provider's programs,
  no fault has been injected, and the compiler and the machine do what the published sources say. A
  bug in those sources is not excluded. ([limits.md](limits.md#honest-run))

## I

- <a id="id"></a>**ID.** See [commitment ID](#commitment-id).
- <a id="in-step"></a>**`-in-step N`.** Makes `cp_prove` prove N requests (1 to 16) of a
  [batch](#batch) in one pass over the weights.
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="index"></a>**Index.** The committed data: the matrix of the weights' digits. Some theorems
  in the Lean library use the word, and so do most names in the code that begin with it, such as
  `IndexShape` and `index_tree()`. Elsewhere, and in `IndexWeights`, the word has its ordinary
  meaning, a position in a list. ([lean/GLOSSARY.md](../lean/GLOSSARY.md))
- <a id="integer-model"></a>**Integer model.** The network with quantized weights and activations,
  computed in integers only. Every proof is about this model, not the floating-point model.
  [model.md](model.md) defines it. ([limits.md](limits.md#the-integer-model))
- <a id="integer-serving"></a>**Integer serving.** Serving in which the engine runs a package
  designed to compute the [integer model](#integer-model). The aim is for a served reply to be
  provable under [exact match](#exact-match).
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="isk"></a>**`isk`.** The index secret key. See [secret key
  directory](#secret-key-directory). `cp_prove` also writes its counters there.
- <a id="ivk"></a>**`ivk`.** The index verification key. See [public key
  directory](#public-key-directory).

## J

- <a id="johnson-radius"></a>**Johnson radius.** A bound from coding theory: there can be only a
  limited number of code words within this distance of a word. The protocol runs the [proximity
  tests](#proximity-test) up to this distance in both phases (for a request and a [batch](#batch),
  and for the committer's tests on a [weight table](#weight-table)).
  ([limits.md](limits.md#what-the-theorem-assumes))

## K

- <a id="key"></a>**Key.** In some theorems of the Lean library, the public key that the committer
  outputs. The programs' counterpart is the [key file](#key-file) with the [leaf files](#leaf-file),
  but no theorem is about `cp_commit`, `cp_prove` or `cp_verify`, which write and read those files.
  Elsewhere, "key" can also mean the key file, or a secret value from which the programs derive
  randomness. ([lean/GLOSSARY.md](../lean/GLOSSARY.md))
- <a id="key-file"></a>**Key file.** `ivk/key.bin` holds the [anchor](#anchor)'s root, its [OOD
  answers](#record) and the committer's one-time proofs about the anchor.
  ([architecture.md](architecture.md#the-files))
- <a id="key-bin"></a>**`key.bin`.** See [key file](#key-file).
- <a id="kl-bin"></a>**`KL.bin`.** See [quantization scales](#quantization-scales).

## L

- <a id="layout-txt"></a>**`layout.txt`.** See [configuration](#configuration).
- <a id="leaf-file"></a>**Leaf file.** `ivk/leaf/<k>.bin` holds [leaf table](#leaf-table) k's root,
  its [OOD answers](#record) and a proof that ties that table to the [anchor](#anchor). The Lean
  library calls it the entry file. ([architecture.md](architecture.md#the-files))
- <a id="leaf-table"></a>**Leaf table.** Another [weight table](#weight-table) for the same weights,
  with its own key and [masks](#mask). It isn't a leaf of a Merkle tree. Proofs are made against
  leaf tables, and each proof names the one it used.
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="lean"></a>**Lean.** A proof assistant: a program that checks proofs written in its
  language. ([limits.md](limits.md#what-the-lean-proofs-cover))
- <a id="verify-on-bytes"></a>**Lean text `verify-on-bytes`.** A Lean text (`lean/OnBytes/`) from
  which this repository builds no stand-alone program; one test (tools/accepts/) compiles it and has
  Lean load the compiled code while it evaluates files of lean/Ask/. A stand-alone program built
  from it was run beside `cp_verify` as a yardstick for tests. What it is, and what is not known of
  it: [architecture.md](architecture.md#lean-onbytes).
- <a id="limit"></a>**Limit (of a leaf table).** How much a [leaf table](#leaf-table) can show
  before its [masks](#mask) are used up. `cp_commit` prints it in a line that begins
  `cp_commit: the limit:`, and the numbers on that line depend on the setting. The first is how many
  columns a table may show in total, the second how many its own link has used, the third how many a
  batch uses, and the fourth how many batches each leaf table can serve. A table's masks cover
  exactly that many columns, with no margin. Columns shown beyond the limit give exact linear
  equations in the weights. The proofs that show them still verify, and the leak cannot be undone.
  It has not been proved for the programs that staying below the limit is enough (row HW-760).
  ([limits.md](limits.md#the-limit-of-use))
- <a id="link"></a>**Link.** The proof in a [leaf file](#leaf-file) that ties a [leaf
  table](#leaf-table) to the [anchor](#anchor). In printed output, the round names `LINK`,
  `LINK point` and `LINK phi` belong to it. In some files and comments "link" means something else:
  a [segment](#segment) of a request's proof. ([architecture.md](architecture.md#the-files))
- <a id="logit"></a>**Logit.** The [integer model](#integer-model)'s score for one token at one
  position, before any normalisation.

## M

- <a id="mark-file"></a>**Mark file (`mark.txt`).** A file of three lines at the top of a [spool
  directory](#spool-directory). A front end that knows the [commitment ID](#commitment-id) writes
  that ID in the third line, and `cp_prove` writes 64 zeros there. `cp_prove` starts only on a
  directory that has a well-formed mark file or [counts as
  empty](service.md#what-a-start-does-to-the-spool-directory), so that a call that is given a
  directory of another kind changes nothing there. It does not tell one commitment's directory from
  another's. The scripts in `integrations/` do, and take a mark of zeros as any commitment's. The
  file guards against a mistake, not against someone who can write in the directory.
  ([service.md](service.md#the-mark-file))
- <a id="mask"></a>**Mask.** A random value, also called blinding. A program combines it with a
  hidden value, or encodes it together with that value, before it sends anything that depends on the
  value. In a [weight table](#weight-table), the masks are the random entries encoded with each row.
  ([limits.md](limits.md#a-lost-mask))

## N

- <a id="new-commitment"></a><a id="renew"></a>**New commitment.** `cp_commit` can't add [leaf
  tables](#leaf-table) to an existing [commitment](#commitment), so a provider makes a new one
  before any of its leaf tables is used up. The new one has a new [seed](#seed) and a new
  [commitment ID](#commitment-id), and that ID must be published.
  ([operators-duties.md](operators-duties.md#a-new-commitment))
- <a id="next-token"></a>**Next token (`y`).** The token after the [window](#window). If a request's
  range ends at the end of the window (e = T), the last checked position predicts it; otherwise `y`
  is 0. It isn't a [checked token](#checked-token) and has no cost. Where a page says "one token
  checked per proof", it means this token. Whatever the policy, the proof states that `y` is the
  [integer model](#integer-model)'s [top choice](#top-choice). Sentences on Lean's theorems use the
  words more widely: there the integer model's next token is its top choice at any position.
  ([verifying.md](verifying.md#the-request-and-the-token-file))
- <a id="no"></a>**`NO: <name>`.** The last line `cp_verify` prints when it rejects a call, most
  often because a check failed. `<name>` is the [rejection name](#rejection-name). [Exit
  code](#exit-code) 1. ([verifying.md](verifying.md#exit-codes))
- <a id="no-overflow-conditions"></a><a id="size-conditions"></a>**No-overflow conditions.**
  Inequalities on the public parameters that `cp_verify` tests before it checks a proof. They are
  meant to ensure that an equation that a proof checks modulo the prime also holds over the
  integers. That they suffice for the programs' equations is not proved. These docs also call them
  size conditions, conditions on the sizes, and no-wrap conditions.
  ([limits.md](limits.md#which-verifier-has-a-theorem))
- <a id="no-proof-of-work-option"></a>**`-no-proof-of-work`.** A `cp_verify` option for a caller who
  does not accept counted [proof of work](#proof-of-work). If some round would miss the
  [target](#target) without its counted work, the call still checks the proof, but ends at best in
  [`NOT-A-VERIFICATION`](#not-a-verification). Where no round needs its work, the option changes
  nothing. [`check_proof.py`](#check-proof) has the same option, `--no-proof-of-work`.
  ([verifying.md](verifying.md#when-to-accept))
- <a id="not-checked"></a>**`NOT CHECKED: <name>`.** The last line `cp_verify` prints when it ends
  without a verdict: the verifier itself failed, or it met a rare challenge value that it does not
  handle. Nothing has been verified. [Exit code](#exit-code) 5.
  ([verifying.md](verifying.md#exit-codes))
- <a id="not-a-verification"></a>**`NOT-A-VERIFICATION`.** The last line `cp_verify` prints when no
  check failed but the call does not count as a verification. [Exit code](#exit-code) 14. ([the
  causes](../verifier/docs/output.md#exit-codes))

## O

- <a id="ood-answers"></a><a id="record"></a>**OOD answers (in the programs: "record").**
  Out-of-domain (OOD) answers (masked sums of the weights) are the answers to out-of-domain sampling
  on a committed table, without the OOD seeds: the committer's answers to samples of each row, kept
  with the table's root in the [key file](#key-file) and in each [leaf file](#leaf-file). Every
  commitment publishes them. Each answer is a sum of the row's weights with public factors, masked
  by a [pad](#pad) that the programs derive from a secret [seed](#seed), and it stays public for
  good. One pad is all that covers each answer ([protocol.md](protocol.md), section 7). They let the
  committer's [proximity tests](#proximity-test) run up to the [Johnson radius](#johnson-radius). In
  the programs' own accounting, an accepted proof rests the binding of the weights, too, on [the
  theorem on codes](#theorem-on-codes), taken as given and used with no bits to spare, beside the
  counted work where the file counts it ([limits.md](limits.md#what-the-theorem-assumes)).
  [verifying.md](verifying.md#when-to-accept) says which line of a call names them.

  "Record" is the programs' own word for these answers: in what they print, in their files and in
  the format pages. `record 1` is their line in the parameter file, and `record=1` their field in
  [`result-v1`](#result-v1). "The record's seeds" are the OOD seeds. "A record" is also the title of
  a passage of docs/operators-duties.md, to which the programs' messages send their reader.
- <a id="opening"></a>**Opening.** The part of a [batch](#batch)'s proof that reveals some columns
  of the [leaf table](#leaf-table), along with their Merkle paths.
  ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="operator"></a>**Operator.** The provider in its day-to-day role of running the service.
  ([operators-duties.md](operators-duties.md))
- <a id="ordinary-serving"></a>**Ordinary serving.** The expected way to serve. vLLM, or any other
  engine, serves the floating-point model, and the proof follows as one [prefill](#prefill) pass
  over the prompt and reply, under the clipped-gap policy. ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="other-programs"></a>**Other programs.** *Other programs* (test builds and test programs among them) are programs that are not the released programs: [which those are](limits.md#what-was-run-on-these-programs), and [what these pages tell of the others](limits.md#on-other-programs). What a run found on them does not carry over to the released programs; what a reading found carries over only as far as the differences up to the released code were read (row HW-679). A fault found on them may be in the released programs too. Not every test build in these pages was made from these sources. Where a run on a test build is told, no public test of SUITE.md is named and the sources are not said, count what it found in favour as found on other sources; a fault that it found may be in the released programs too. *These sources* are the sources in this repository.
- <a id="other-sources"></a>**Other sources.** Sources that these pages do not tell as this repository's: [which they tell so](limits.md#what-was-run-on-these-programs). See [other programs](#other-programs).

## P

- <a id="p258"></a>**`p258`.** The prefix at the head of most of the fixed lines that `cp_prove`
  prints ([the list](../prover/docs/closed_front.md)); the last one is `p258 end <exit code>`,
  usually followed by a name. It has no meaning: the source calls it "a prefix without meaning"
  (`prover/prover/closed_front.h`). Two labels that `cp_prove` hashes for a request's id and a
  response's tag begin with it too (`p258-spool-req-1`, `p258-spool-tag-1`;
  `prover/prover/spool.h`).
- <a id="padding"></a>**Padding.** For the tokens that fill up a [window](#window), see [filler
  tokens](#filler). The padding cells of a [weight table](#weight-table) are something else; see
  [limits-further.md](limits-further.md#so-22).
- <a id="parameter-file"></a>**Parameter file.** `policy.txt`. Despite its name, it holds far more
  than the [acceptance policy](#acceptance-policy): the model's shape, the proof parameters and the
  policy's lines. ([architecture.md](architecture.md#the-files))
- <a id="part-passed"></a>**`PART PASSED: …`.** The last line of a [part-call](#part-call) whose
  checks passed. No proof has been accepted. ([verifying.md](verifying.md#exit-codes))
- <a id="part-call"></a>**Part-call.** A call to `cp_verify` that checks only the [key
  file](#key-file) (`-key`) or only one [leaf file](#leaf-file) (`-leaf`). If the checks pass, it
  says so and returns [exit code](#exit-code) 3, unless it ends in `NOT-A-VERIFICATION`, for example
  with fewer than 128 [requested bits](#requested-bits) or in a [test build](#test-build). No proof
  has been accepted: only `cp_verify`'s checks on that file passed, and they don't show whose
  weights the file commits to. A `-leaf` result counts only if a `-key` call has also passed on the
  same files. ([limits-further.md](limits-further.md#so-49.2))
- <a id="vouching-party"></a>**Party that vouches for the model.** The person or organisation that a
  client trusts to say which model a [commitment ID](#commitment-id) stands for: the provider itself
  or a third party. No program checks that claim.
  ([verifying.md](verifying.md#what-you-need-and-where-it-must-come-from))
- <a id="plan-only"></a>**`PLAN ONLY: no proof was checked`.** The last line of a [planning
  call](#planning-call) that succeeds. ([verifying.md](verifying.md#exit-codes))
- <a id="planning-call"></a>**Planning call.** A call of the form `cp_verify -plan …`. It checks no
  proof. Most forms print what a [parameter file](#parameter-file) requires of one. If it succeeds,
  it exits with [code](#exit-code) 4. `-plan id` prints the [commitment ID](#commitment-id) of the
  files as they are, for the party that publishes it. If the files were sent to you, do not use the
  ID this call prints as your `-id`. ([../verifier/README.md](../verifier/README.md#planning-calls))
- <a id="policy-option"></a>**`-policy`.** A `cp_verify` option that names the policy the caller
  expects: first its kind, then `C`, `theta`, `cap`, `s` and `sh`.
  ([verifying.md](verifying.md#acceptance-policies); [when a call ends in
  `NO: policy`](../verifier/docs/output.md#rejection-names))
- <a id="policy"></a>**Policy.** See [acceptance policy](#acceptance-policy).
- <a id="policy-txt"></a>**`policy.txt`.** See [parameter file](#parameter-file).
- <a id="position"></a>**Position.** A place in the [window](#window), counted from 0. Position t
  holds token t, and the model's output at position t predicts token t + 1.
  ([architecture.md](architecture.md#what-a-proof-states))
- <a id="prefill"></a>**Prefill.** One forward pass over the prompt and reply together. A proof is
  about a prefill pass of the [integer model](#integer-model), not token-by-token generation.
- <a id="production-build"></a>**Production build.** What `prover/build.sh` builds when it is called
  without a target, or with a target that begins with `serving`. The docs under `prover/docs/`, and
  `tools/audit/README.md`, call it the default build. Its programs lie in the directory `hc` of the build's output (`prover/build/hc`). Every other build of `cp_commit` and `cp_prove` is a test build. [quickstart.md](quickstart.md) has the command.
- <a id="the-programs"></a>**Programs.** "The programs" (elsewhere "the code") are the programs in
  this repository as they are built and run, above all the committer and the prover (`cp_commit`,
  `cp_prove`), which handle the weights. What is proved (in Lean) or argued (on paper) is about
  something else: the protocol. Nothing proves that the programs do exactly what the protocol says,
  or that `cp_verify` makes exactly the checks of the protocol's verifier, or only those.
  ([guide/what-is-proved.md](../guide/what-is-proved.md))
- <a id="prompt"></a>**Prompt.** The tokens a client sends. In a request they come first in the
  [window](#window).
- <a id="proof-file"></a>**Proof file.** `<name>.proof` holds the [leaf table](#leaf-table)'s
  number, then the proof of one request. ([architecture.md](architecture.md#the-files))
- <a id="proof-of-work"></a>**Proof of work.** A search for a number that makes a set number of bits
  of a hash zero. The [default setting](#default-setting) makes the prover do this before some
  rounds, which makes each retry of such a round more expensive for a malicious prover. The [error
  accounting](#error-accounting) counts it toward the target.
  ([verifying.md](verifying.md#when-to-accept))
- <a id="proved"></a>**Proved (in Lean).** "Proved" on these pages means that Lean's build accepted
  the proof, and a build alone does not fail on an unfinished proof or on a new axiom: the final
  checks do ([lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md)).
- <a id="prover-word"></a>**Prover.** See [`cp_prove`](#cp-prove).
- <a id="provider"></a>**Provider.** The party that owns the weights and runs the committer, the
  prover and a [front end](#front-end).
- <a id="proximity-test"></a>**Proximity test.** A test on a group of oracle messages, or on a
  [weight table](#weight-table). It is designed to show that the rows are close to code words whose
  messages satisfy the group's linear claims. A [leaf file](#leaf-file) holds one; a [key
  file](#key-file) and a [proof file](#proof-file) each hold more than one.
  ([protocol.md](protocol.md))
- <a id="public-key-directory"></a>**Public key directory (`ivk`).** The [key file](#key-file) and
  one [leaf file](#leaf-file) for each [leaf table](#leaf-table). The provider publishes it.
  ([architecture.md](architecture.md#the-files))
- <a id="public-weights-mode"></a><a id="published-seed"></a>**Public-weights mode.** An option for
  weights that are already public: `cp_commit -public-weights <file>` derives the [seed](#seed) from
  a 32-byte file instead of sampling it. With this option the weights are not hidden: whoever has
  the file can compute every [mask](#mask) of the [key file](#key-file), of the [leaf
  files](#leaf-file) and of the [weight tables](#weight-table)' columns that proofs open. Nothing
  else is claimed to be hidden either: not one client's request from another client in the same
  batch. The public files do not record the mode, so a verifier cannot tell that it was used.
  ([what-is-not-hidden.md](what-is-not-hidden.md#public-weights))

## Q

- <a id="quantization-scales"></a>**Quantization scales.** The numbers that say how the [integer
  model](#integer-model) rounds its values to integers. Most are in `KL.bin`, and the rest are lines
  of the [parameter file](#parameter-file). The tools in `scales/` choose them.
- <a id="query-bound"></a>**Query bound.** [protocol.md](protocol.md)'s term for the number of
  positions of an encoded row that its random entries can hide. See [limit](#limit).

## R

- <a id="random-oracle-model"></a><a id="ideal-hash"></a>**Random-oracle model.** An idealisation
  used in the proofs: the hash function is treated as a random function that every party can only
  query. The programs use BLAKE2b and a hash built on its compression function, and nothing has been
  proved about either. ([limits.md](limits.md#what-the-theorem-assumes))
- **Record.** The programs' word for a committed table's [OOD answers](#ood-answers).
- <a id="rejection-name"></a>**Rejection name.** The name a program prints for the check that
  failed. `cp_verify` prints it after `NO: ` ([`NO: <name>`](#no)). `cp_commit` and `cp_prove` print
  it in brackets as part of a sentence, as in `refused (cp-isk)`. ([faults.md](faults.md))
- <a id="released-programs"></a>**Released programs.** `cp_commit`, `cp_prove` and `cp_verify` as
  this repository's sources build them (the production build). Which runs these pages tell as runs
  on them, and by what rule: [limits.md](limits.md#what-was-run-on-these-programs). The programs
  that made the runs that these pages tell as runs on the released programs, or on a test build of
  these sources, were built from sources that differ from these, in what is told there and in
  nothing else, and these pages say of them that they were "built from these sources"; what has run
  on programs built from exactly these sources is told there too. More:
  [guide/terms.md](../guide/terms.md).
- <a id="released-response"></a>**Released response.** A response that `cp_prove` has given its final name in `out/`. Until then it
  is staged, under `out/.tmp/`. `cp_prove` runs no verifier, so a released response is not a checked
  one. ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="reply"></a>**Reply.** The tokens the provider's engine served in answer to a prompt. In a
  request they follow the prompt in the [window](#window).
- <a id="request"></a>**Request.** A file (`<name>.req`) with a [window](#window) of T token ids and
  two numbers, r and e, that say which tokens are checked. It isn't the prompt. The [front
  end](#front-end) builds it from the prompt and reply tokens, and the client does the same for its
  own check. In a [spool directory](#spool-directory) the file has 32 extra random bytes, and
  `cp_verify` rejects a request file that still has them.
  ([verifying.md](verifying.md#the-request-and-the-token-file))
- <a id="requested-bits"></a>**Requested bits (`-bits`).** The number of bits a caller requires of
  `cp_verify`'s [error accounting](#error-accounting): 128 unless the option is given. The
  [parameter file](#parameter-file)'s target must be at least as strong. A lower value is for tests:
  the proof is still checked, but it is not accepted.
  ([../verifier/README.md](../verifier/README.md#the-call))
- <a id="response"></a>**Response.** What `cp_prove` returns for a request: either a [proof
  file](#proof-file) and a [token file](#token-file), or a `.failed` file.
  ([architecture.md](architecture.md#from-weights-to-a-checked-proof))
- <a id="result-v1"></a>**`result-v1`.** A line for programs. In a [whole call](#whole-call) that
  exits with code 0 or 14, `cp_verify` prints it immediately before its last line ([its
  fields](../verifier/docs/output.md#the-line-for-programs)). [Exit code](#exit-code) 0 alone is not
  reason enough to accept a proof, so compare the fields with your own values.
  ([verifying.md](verifying.md#when-to-accept))
- <a id="round"></a>**Round.** One message from the prover together with the verifier's message that
  follows it. In the interactive form of the protocol the verifier would send challenges; here they
  are computed from a hash, and some rounds have none. The programs compute an error bound for each
  round. ([architecture.md](architecture.md#the-proof-system-in-outline))

## S

- <a id="s-line"></a>**`s`.** A line in the [parameter file](#parameter-file) that says each of the
  model's logits is an integer logit divided by 2^`s`. That is the provider's statement, and no
  equation in the proof ties it to the model. The caller names it with `-policy`.
  ([verifying.md](verifying.md#acceptance-policies))
- <a id="salt"></a>**Salt.** Random bytes hashed together with a column, so that no one can confirm
  a guess at the column from its hash. ([protocol.md](protocol.md))
- <a id="score-gap"></a><a id="shortfall"></a>**Score gap.** How far a token's logit falls below the
  largest logit at its position, plus 1 if the [top choice](#top-choice) has a lower token id than
  the token. ([verifying.md](verifying.md#acceptance-policies))
- <a id="score-policy"></a>**Score policy.** See [clipped gap](#clipped-gap), the only score policy
  the programs support.
- <a id="secret-key-directory"></a>**Secret key directory (`isk`).** The provider's secret files:
  the [seed](#seed), the [configuration](#configuration) as committed and, for each [weight
  table](#weight-table), its files and its [counter](#counter). Never publish it, because anyone who
  reads it can remove the [masks](#mask) from columns that have been shown, and that cannot be
  undone. Never copy or restore it, because a restored copy resumes from an old counter.
  ([operators-duties.md](operators-duties.md#backups-and-restored-copies))
- <a id="secret-seed"></a>**Secret seed.** See [seed](#seed). Compare [test seed](#test-seed) and
  [public-weights mode](#public-weights-mode).
- <a id="security-level"></a>**Security level.** A number of bits k such that breaking a system is
  expected to take about 2^k operations. These docs do not state one for the programs. Any number of
  bits a program prints is that program's own calculated error bound.
  ([limits.md](limits.md#the-soundness-figure))
- <a id="seed"></a>**Seed.** 32 random bytes that `cp_commit` samples and keeps in the [secret key
  directory](#secret-key-directory). (In [public-weights mode](#public-weights-mode) it derives them
  from a published file instead, so they are not secret.) The programs derive the
  [anchor](#anchor)'s masks and salts, and the pads of its [OOD answers](#record), from the seed
  only, and every other mask and salt from the seed plus further random bytes. Outside that mode,
  only the provider is meant to hold the seed, and verifiers never need it. Anyone who gets it can
  remove [masks](#mask) from the anchor's columns that have been shown (and, with a [leaf
  table](#leaf-table)'s `fresh.bin`, from that table's). They can also remove the pads from the
  anchor's published answers (and, with a leaf table's `fresh.bin`, from that table's answers), each
  of which is then an exact linear equation in the weights. And they can test any guess at the
  weights. That cannot be undone. ([limits.md](limits.md#the-seed))
- <a id="seed-bin"></a>**`seed.bin`, `fresh.bin`.** `seed.bin` holds the [seed](#seed), and
  `fresh.bin` holds a [leaf table](#leaf-table)'s own random bytes. Both are in the [secret key
  directory](#secret-key-directory).
- <a id="segment"></a>**Segment.** The part of a request's proof that belongs to one sub-proof. It
  covers a layer group without its attention, that group's attention, or the output head. Some files
  and comments call it a link: the parameter lines `group link` and `stages link`, and the first
  field of each non-comment line of `layout.txt`. See also [link](#link).
  ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="self-check"></a>**Self-check.** A check that `cp_commit` or `cp_prove` makes on its own
  work on every run. Some checks are off under a test seed (row HW-277). Some self-checks recompute
  random values and compare, but they cover only some of the random values.
- <a id="the-server"></a>**Server (the server you are checking).** The machine that takes your
  prompt and sends you the reply, the proof and the files. The provider runs it. What reaches you
  through it is what a check is there to test, so nothing that the check takes on trust may come
  that way ([verifying.md](verifying.md#what-you-need-and-where-it-must-come-from)).
- <a id="serving-build"></a>**`serving-build`.** The programs' own name for a [production
  build](#production-build): `cp_prove -h` begins `This is a serving build.`, and some rejections
  print `refused (serving-build)`.
- <a id="setting"></a>**Setting.** A complete choice of the protocol's parameters. A
  [configuration](#configuration) fixes a setting along with a model's shape and scales. Sentences
  on Lean's theorems use the word differently: there, the settings of the model are its sizes and
  its rescaling numbers, not the protocol's parameters. All valid settings share one set of sizes,
  and only the rescaling numbers vary. The protocol's parameters Lean fixes in one table, except for
  a few that its theorems do not fix, among them the number of leaf tables and the salts' lengths.
- **Size conditions.** See [no-overflow conditions](#no-overflow-conditions).
- <a id="soundness"></a>**Soundness.** The property that a false claim is accepted only with small
  probability, whatever the provider does within a bounded number of hash queries. Lean has a
  soundness theorem for its own verifier, `verify`, not for the C++ program that users run. [No
  theorem is about `cp_commit`, `cp_prove` or `cp_verify`](limits.md#which-verifier-has-a-theorem).
  Both the Lean theorem and a written argument, in a technical report that is not public and not refereed, treat the
  hash as a [random oracle](#random-oracle-model). The written argument rests on [the theorem on
  codes](#theorem-on-codes), taken as given. [guide/what-is-proved.md](../guide/what-is-proved.md)
  says what is proved, and [limits.md](limits.md#soundness) what the arguments rest on and what they do not reach.
- <a id="spool-directory"></a>**Spool directory.** The directory through which a service
  (`cp_prove -spool`) takes requests, in `in/`, and returns responses, in `out/`.
  ([operators-duties.md](operators-duties.md#how-the-service-works))
- <a id="stage"></a>**Stage.** One of the three parts of a `cp_verify` call: `key` checks the [key
  file](#key-file), `leaf` ties the [leaf file](#leaf-file) to it, and `request` checks the proof.
  Each computes the [commitment ID](#commitment-id) and compares it with `-id`.
  ([../verifier/README.md](../verifier/README.md#the-call))
- <a id="standard-parameter-form"></a>**Standard parameter form.** A list of named requirements on
  the [configuration](#configuration) and the call; for example, the call must name the [commitment
  ID](#commitment-id) (`id-named`). If one is not met, the call ends at best in
  `NOT-A-VERIFICATION`, even if no check fails.
  ([../verifier/docs/parameter-file.md](../verifier/docs/parameter-file.md#the-standard-parameter-form))
- <a id="statement-on-codes"></a><a id="assumed-statement"></a>**Statement on codes.** A statement
  on Reed–Solomon codes that stands, in the Lean library, in the place of a theorem from the
  literature (Theorem 4.6 of ePrint 2025/2055; [what has been checked of it, and what has
  not](limits.md#what-was-checked-of-the-theorem); a weaker fall-back, not called firm: ePrint
  2025/2051, Lemma 9.3, row LIM-17.2). Its match to the published statement has not been
  independently compared. The library proves it, from files adapted from ArkLib and Mathlib
  (`lean/ThirdParty/`). The theorems that name it keep it as a hypothesis; others rest on its proof,
  and their statements show no trace of it. Row L15 and
  [limits.md](limits.md#what-the-lean-proofs-cover) say more. See also [theorem on
  codes](#theorem-on-codes).
- <a id="stored-table"></a>**Stored table.** Another name for a [weight table](#weight-table), used
  in the format's docs and in some sentences about the provider's files.
- <a id="synthetic-weights"></a>**Synthetic weights.** Weights the programs generate themselves,
  without reading any model's files (`cp_commit -weights generated` or `-weights made-up-blocks`).

## T

- <a id="t-r-e"></a>**`T`, `r`, `e`.** T is the [window](#window)'s length in tokens. r and e, at
  the end of a request, name the checked positions: r to e − 1.
  ([architecture.md](architecture.md#what-a-proof-states))
- <a id="tag"></a>**Tag.** `<id>.tag`: 32 bytes that `cp_prove -spool` writes next to a response.
  The [front end](#front-end) checks the files against the tag, and the proof with `cp_verify`, before passing them on. The tag isn't
  part of the response. ([service.md](service.md#the-tag))
- <a id="target"></a>**Target.** The error bound every round must meet in the programs' [error
  accounting](#error-accounting) before `cp_verify` accepts. The parameter line `sec` sets it: with
  `sec 128`, as in every example, each round must reach 128 bits. A round's bits are those of its
  error bound plus those of any [proof of work](#proof-of-work). The work does not lower the chance
  that a challenge lets a false claim survive: it makes each retry cost the prover.
  ([limits.md](limits.md#what-the-theorem-assumes))
- <a id="technical-report"></a>**Technical report.** A document of this project that is not public and not refereed, which defines a
  protocol and states theorems about it. The programs run a protocol that differs from it in places.
  It is not in this repository, so you cannot check what these docs say about it.
- <a id="test-build"></a>**Test build.** A build made for tests. The targets `tests`, `cp_prove` and
  `cp_commit` of `prover/build.sh` build the GPU programs so that they accept [test
  options](#test-option), and they then reject a real model's files. A test build of `cp_verify`
  takes no further option: it is compiled with one of the library's switches for a test
  (`blocks/fence.h`). It never accepts a proof; at best it prints `NOT-A-VERIFICATION`.
- <a id="test-option"></a>**Test option.** An option for tests, such as `-test-seed`. Only a [test
  build](#test-build) accepts one.
- <a id="test-seed"></a>**Test seed.** A [seed](#seed) read from a file passed with `-test-seed`, so
  that a run can be reproduced. A test seed hides nothing, and a verifier can't tell that one was
  used. Only a [test build](#test-build) accepts one, and a test build rejects a real model's files.
  A [production build](#production-build) has a separate mode for a published seed, [public-weights
  mode](#public-weights-mode). The programs reject only some misuses of the [test
  options](#test-option); for example, nothing stops a [commitment](#commitment)'s real seed from
  being passed as a test seed.
- <a id="tests-on-the-public-files"></a>**Tests on the public files.** Tests that `verify` makes on
  the public files alone, whatever the proof, beyond the test of the scope, `InScope`. Where one
  fails, `verify` rejects everything, and the soundness theorem on `verify` holds for that reason:
  it is then vacuous at these files. That they pass is proved, in a side library, at the library's
  written copies of the two stored sets of public files. One constant beneath passes the kernel only
  at a larger stack, and `leanchecker` fails on it. At all other files in scope it is not proved.
  Passing them does not make the theorem non-vacuous: no theorem says these are all; that `verify`
  accepts the honest prover's proofs is proved of the compiled system alone, in the random-oracle
  model. (In Lean: `protocolDataWellFormed`. [lean/TERMS.md](../lean/TERMS.md);
  [lean/THEOREM.md](../lean/THEOREM.md).)
- <a id="theorem-on-codes"></a>**Theorem on codes.** The theorem on which every [closeness
  round](#closeness-round) rests at the [default setting](#default-setting) (the [planning
  call](#planning-call) counts none on [the bound within a third of the distance](#proved-lemma)): Theorem 4.6 of the full
  version (ePrint 2025/2055) of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, STOC
  2026, which the programs' accounting takes as given; the proceedings state the plain form up to
  the [Johnson radius](#johnson-radius) (Theorem 1.5), not this one. A statement on codes held to follow from that theorem, in its case of two words, is proved in this repository's Lean library, by way of another formalization's theorem and a comparison of bounds; the paper's own form of the bound is not proved there, and that the library's statement follows from the paper's, and that these programs' parameters meet the theorem's conditions, rest on reading, not on a proof (lean/LIMITS.md, L15; [statement on codes](#statement-on-codes); [what has been
  checked of it, and what has not](limits.md#what-was-checked-of-the-theorem)). The rounds that rest
  on it hold at the target with no bits to spare. The closeness rounds' error bounds also rest on
  further steps argued in a technical report that is not public and not refereed. The field
  `commitment-assumed-rounds` of [`result-v1`](#result-v1) counts the closeness rounds of the key
  file and the leaf file that rest on a source other than the bound within a third of the distance; the request's own rounds
  are not counted. The fall-back if that theorem should fail (a further, weaker cited lemma, ePrint
  2025/2051, Lemma 9.3): row LIM-17.2.
- <a id="timed-runs"></a>**Timed runs.** The runs behind the figures on
  [guide/performance.md](../guide/performance.md) that carry no mark. Which they are:
  [guide/terms.md](../guide/terms.md). They were made on the released programs:
  [limits.md](limits.md#the-timed-runs). The runs that timed the serving scripts' check, by itself
  and beside a serving `cp_prove`, on the released programs, in one of which a constant of the
  scripts is not this repository's, are none of them: their figures on guide/performance.md say so
  in words.
- <a id="token-file"></a>**Token file.** `<name>.y` holds the [next token](#next-token), or 0 if the
  range ends before the [window](#window)'s end. `cp_prove` writes one with each proof. A client
  writes its own for its check. ([architecture.md](architecture.md#the-files))
- <a id="top-choice"></a>**Top choice.** The token with the largest logit at a position; among equal
  logits, the one with the lowest id.
- <a id="toy-proof"></a><a id="toy-size"></a>**Toy proof.** A proof about a small synthetic matrix,
  not about a model. `blocks/first_look.sh` makes and checks some on a CPU.
  ([quickstart.md](quickstart.md))

## U

- <a id="unique-decoding"></a>**Unique decoding.** Half of a code's minimum distance: within that
  distance of a word there is at most one code word.
  ([limits.md](limits.md#what-the-theorem-assumes))

## V

- <a id="verifier-word"></a>**Verifier.** See [`cp_verify`](#cp-verify).
- <a id="verify"></a>**`verify`**, the Lean verifier (in Lean `VerifierAndGoals.verify`, at
  `VerifierAndGoals.theDescription`): the verifier of the soundness theorem `sound_of_verify_at`
  ([lean/THEOREM.md](../lean/THEOREM.md)). It runs the compiled verifier (`innerCompiledVerifier`:
  [lean/TERMS.md](../lean/TERMS.md)) at the parameters that the description gives for these files,
  behind the scope test, a proof parser and two public readings. It does not read the prover's
  files. It takes five byte strings, which a small parser outside Lean makes of them, and a list of
  further tokens. `verify` as typed cannot be executed. Its executable form can: `lean-verify`
  ([tools/lean-verify/README.md](../tools/lean-verify/README.md)) runs it on a real full-size proof.
- <a id="vouching"></a>**Vouching party.** See [party that vouches for the model](#vouching-party).

## W

- <a id="weight-hiding"></a>**Weight hiding.** The goal that clients learn nothing about the weights
  from what a provider publishes and sends, beyond what
  [what-is-not-hidden.md](what-is-not-hidden.md) declares. To that end the programs are designed to
  [mask](#mask) proofs and a [commitment](#commitment)'s public files by default.
  [guide/weight-hiding.md](../guide/weight-hiding.md) says how far that has been shown. It is a goal
  for the programs and the service, wider and looser than [index privacy](#index-privacy), which is
  a property of a committer, a prover and a verifier on paper or in Lean.
- <a id="weight-table"></a>**Weight table.** All the weights, arranged as one matrix whose cells are
  the weights' base-256 digits. Each row is encoded together with random [masks](#mask), and a
  Merkle tree is built over the columns of the encoded table.
  ([architecture.md](architecture.md#the-proof-system-in-outline))
- <a id="whole-call"></a>**Whole call.** A call to `cp_verify` with all its inputs. It checks the
  [key file](#key-file), the [leaf file](#leaf-file) and the proof. Only a whole call can accept a
  proof. ([verifying.md](verifying.md#check-a-proof))
- <a id="window"></a>**Window.** The prompt and reply as one row of T token ids. The [parameter
  file](#parameter-file) fixes T, and if the two together are shorter than T, filler tokens pad the
  window. ([architecture.md](architecture.md#what-a-proof-states))
- <a id="within-limits"></a>**`WITHIN LIMITS: clipped-gap …`.** The last line `cp_verify` prints
  when it accepts a proof in which a token of the request was checked under the clipped-gap policy
  that the caller named with `-policy`. It is weaker than a `YES`: it does not show that the reply
  is the model's own. The [exit code](#exit-code) is 0 in both cases, so use the last line to
  distinguish them. See [clipped gap](#clipped-gap) for what an accepted proof means under that
  policy. ([verifying.md](verifying.md#exit-codes))

## Y

- <a id="y-token"></a>**`y`.** See [next token](#next-token).
- <a id="yes"></a>**`YES`.** The last line `cp_verify` prints when it accepts a proof under [exact
  match](#exact-match), or a proof that is only about the [next token](#next-token). Every check
  passed, and the call counts as a verification. [Exit code](#exit-code) 0. Match the whole line,
  not a prefix. A line shortly before it begins `proof of work:`. It says `none counted`, or how
  many bits of [proof of work](#proof-of-work) are counted and in how many rounds.
  ([verifying.md](verifying.md#when-to-accept))

## Z

- <a id="zero-knowledge"></a><a id="index-privacy"></a>**Zero knowledge, index privacy.** Index
  privacy is the term of the definition that this protocol follows (lean/CREDITS.md, "Further work
  that this protocol builds on") for hiding in an argument system. It asks for a simulator that is
  given the weights' public parameters, the number of proofs and the requests with their replies,
  but not the weights. What it makes (the public key, the hash function's answers and all proofs)
  must be statistically close to what a verifier really sees. It is that definition's form of zero
  knowledge. The Lean statement nearest to it is `HasStrongIndexPrivacyWith`; `HasIndexPrivacy` is
  weaker than the definition, in spite of its name. In Lean's `ZeroKnowledge` whoever picks the
  requests sees as much as in the definition, and the key is narrower: 256 bits (lean/TERMS.md, note
  9). For the programs, see [weight hiding](#weight-hiding).

## Words in the source code

This part is for people reading the source code. When the code calls something by a different name,
the entry gives that name. Entries describe what the code is meant to do, so "designed to" states a
goal, not a proved or tested result. [protocol.md](protocol.md) describes the protocol, and
[format/README.md](format/README.md) specifies every byte.

- <a id="id-61"></a>**`61`.** The names `setting_m61()` (`blocks/numbers/setting.h`) and `P61` (`tests/s8_model.py`) have the number 61 in them, and one test writes the parameter line `field m61` (`blocks/numbers/tests/cp_plan_files.py`). By the comments at the two names both mean 2^61 − 1, which is not the prime of the programs' field, and only the tests in `blocks/numbers/tests/` call the function. The number says nothing about the field that the programs use ([field](#field)).
- <a id="src-base"></a>**BASE.** The block that ends a [proximity test](#proximity-test). It sends
  the last rows, masked, and compares them at a list of positions.
- <a id="src-bat"></a>**BAT.** The block that batches many [linear claims](#linear-claim) into one,
  using powers of a challenge. It has two forms: one is the first step of a [proximity
  test](#proximity-test), and the other, `BAT-sigma`, combines claims well before any test.
- <a id="src-batt"></a>**BATT.** [BAT](#src-bat) for claims on the weight matrix. It reduces several
  [linear claims](#linear-claim), each at a separate column point, to one claim at a fresh point.
- <a id="binding-tree"></a>**Binding tree.** A Merkle tree whose leaves represent the requests of a
  batch. Its root goes into the batch's shared [chain](#chain). [protocol.md](protocol.md) calls it
  the batch tree.
- <a id="src-block"></a>**Block.** A named step of the proof system, such as [SEND](#src-send), with
  a prover side and a verifier side. The library's code is in `blocks/iop/`. `cp_prove` and
  `cp_verify` do not run all of it: for the range check, and inside a request's sub-proofs for the
  lookups and the masked sumcheck among others, they have a second implementation of their own.
- <a id="src-cell"></a>**Cell, limb.** A cell is one committed integer; in an honest commitment or
  proof it is a signed byte. A limb is one base-256 digit of a value, stored in one cell. Under
  `prover/engine/`, the type `Cell` is something else: how a message's entries are stored in GPU
  memory.
- <a id="challenge"></a>**Challenge.** Any value taken from a verifier message, that is, from a
  round's hash output.
- <a id="code-switch"></a>**Code switch.** See [CS](#src-cs).
- <a id="column-count"></a>**Column count.** The length of a table's `count.bin`; see
  [counter](#counter) above. Identifiers call it `count`.
- <a id="commit-point"></a>**Commit point.** A numbered prover message within a [segment](#segment),
  in which a [sub-proof](#sub-proof) commits to its arrays. Identifiers call it `moment`.
- <a id="compute-engine"></a>**Compute engine.** The interface to a prover's heavy steps: storing
  and encoding [main rows](#main-row), hashing columns into a tree, folding, fetching opened
  columns, and searching for a proof of work. The CPU version is in `blocks/engine/` and the GPU
  version in `prover/engine/`. The code calls it `Engine`; don't confuse it with the serving
  [engine](#engine).
- <a id="configuration-digest"></a>**Configuration's digest.** A single hash over the three
  configuration files, each length-prefixed. It is a parameter of every [chain](#chain) that has
  parameters, and the hash of each leaf of a [weight table](#weight-table)'s tree begins with it.
- <a id="src-counter"></a>**Counter (in comments).** Usually one of the four fields (`counter`,
  `version`, `object` and `member`) that name a context: one proof or message with random values of
  its own. The context is part of the label that each private random value is derived from. The
  design requires that under one key no label is used twice with different inputs, and that no
  context is used twice under one run's process key. By the comment in `blocks/coins/coins.h`, the
  second is the caller's duty, and nothing there checks it. This counter is not `count.bin`, whose
  length is the column count.
- <a id="src-cs"></a>**CS.** The code switch. It takes the single rows that [FOLD](#src-fold) leaves
  and re-encodes them under new codes as a new [oracle message](#oracle-message). Kept rows are sent
  again, each as a separate oracle message.
- <a id="digits"></a>**Digits.** In the [range check](#range-check), the low and high parts of each
  random combination of cells. A lookup bounds each part.
- <a id="src-f-e"></a>**`F`, `E`.** `F` is the base field, the integers modulo p = 2^64 − 2^32 + 1.
  `E` is its cubic extension, from which challenges are sampled.
- <a id="first-query"></a>**First query.** The hash input that starts a chain. It fixes the chain's
  public inputs before the prover's first message: its label, a digest of the parameters (including
  the [plan](#plan) and the configuration's digest), the [statement](#statement), and the [run
  value](#run). The one exception is the next token: the statement holds it as 0, and it enters the
  chain later; see [chain](#chain).
- <a id="src-fold"></a>**FOLD.** The block that reduces one [linear claim](#linear-claim) on a
  group's rows to one on a few single rows, using a sumcheck with a hiding polynomial. It leaves one
  main row and up to two [mask rows](#main-row). Kept rows stay as they are.
- <a id="src-foldeq"></a>**FOLDEQ.** The block that converts claims about the fold of a committed
  table into claims on a fresh [oracle message](#oracle-message).
- <a id="gram"></a>**Gram pass.** One pass over the two vectors of a product layer. Their inner
  products give the messages of three sumcheck rounds.
- <a id="src-group"></a>**Group.** A set of whole [oracle messages](#oracle-message) under one code,
  which one [proximity test](#proximity-test) reads at one list of positions.
- <a id="hash-chain"></a>**Hash chain.** The code's full name for a [chain](#chain). Each round's
  hash output gives the verifier's message and the next chain value. A proof file has three: the
  request's first chain, the batch's shared chain and the request's second chain.
- <a id="src-hp"></a>**HP.** The hiding polynomial: the random polynomial that a masked sumcheck
  adds to its summand.
- <a id="index-mask-message"></a>**Index-mask message.** An [oracle message](#oracle-message) that
  consists only of [mask rows](#main-row). A request sends one in its first prover message, and the
  committer sends one at the start of the [key file](#key-file)'s one-time chain. Its entries mask
  the claims on the [weight table](#weight-table) and the step that joins them.
- <a id="src-iopp"></a>**IOPP.** The usual abbreviation for an interactive oracle proof of
  proximity. Here it means the [proximity test](#proximity-test) on a group: [BAT](#src-bat), then
  [FOLD](#src-fold) and CS stage by stage, then [BASE](#src-base).
- <a id="item"></a>**Item.** A name-value pair in a [chain](#chain)'s [first query](#first-query).
  Also (the type `Item`) one typed piece that a side writes into a proof or reads from it.
- <a id="kept-rows"></a>**Kept rows.** The [main rows](#main-row) of an [oracle
  message](#oracle-message) that is marked as kept when it joins a group. They are read and tested
  with the group, but [FOLD](#src-fold) doesn't fold them, because a later part of the protocol
  reads them again. At a code switch, each kept message is committed again as a separate oracle
  message.
- <a id="layer-group"></a>**Layer group.** 2^`ll` consecutive layers, where `ll` is set in the
  [parameter file](#parameter-file). Each group has two [segments](#segment): one for its layers and
  one for its attention.
- <a id="src-lin"></a>**LIN.** The block that proves [linear claims](#linear-claim) on the [weight
  table](#weight-table), one for each request in a batch. It folds the table once for the whole
  batch.
- <a id="src-line"></a>**LINE.** The block that reduces two claims on the halves of a multilinear
  map to one claim on the map, using one challenge.
- <a id="linear-claim"></a>**Linear claim.** The assertion that a given linear form, applied to the
  messages encoded in some [oracle messages](#oracle-message), has a given public value. Identifiers
  and error messages call it `Row` or "row".
- <a id="src-link"></a>**LINK.** Two things: the block designed to prove that a [leaf
  table](#leaf-table) has the same entries as the [anchor](#anchor), and the resulting proof, stored
  in a [leaf file](#leaf-file). The docs call that proof a [link](#link).
- <a id="src-lkp"></a>**LKP.** The product-tree lookup argument, which proves that every entry of a
  vector (the key vector) occurs in a public table.
- <a id="main-row"></a>**Main row, mask row.** The two kinds of row in an [oracle
  message](#oracle-message). Main rows hold data; mask rows hold random elements and have a separate
  code. In a code switch's message, the mask rows instead hold the folded mask rows' messages and
  the old rows' encoding randomness. A [weight table](#weight-table) has no mask rows. The type
  `Row` in the code is something else: see [linear claim](#linear-claim).
- <a id="mask-entry"></a>**Mask entry.** One random element in a [mask row](#main-row) of an [oracle
  message](#oracle-message). A block adds the entry to a value that it shows. The design requires
  that each entry be used only once, and the programs check that in part. On the prover's side, the
  `blocks/` library issues each entry only once. After a [sub-proof](#sub-proof)'s entries were last
  used, `cp_prove` compares them with a second derivation, but only a test build counts how often
  each entry is used. `cp_verify` gives every masked value of a sub-proof its own entry number, and
  does not accept a proof that does not fit that numbering. Neither side detects a use that bypasses
  the library's bookkeeping, such as one random value placed in two entries.
- <a id="masked-sumcheck"></a>**Masked sumcheck.** Another name for [PSC](#src-psc). The prover
  first sends the hiding polynomial's sum, and a challenge then combines that polynomial with the
  summand.
- <a id="src-omega"></a>**Omega.** The [oracle message](#oracle-message) in [LIN](#src-lin) that a
  batch's requests share, with one entry for each row of the weight matrix.
- <a id="one-time-file"></a>**One-time files.** The files that `cp_commit` creates once per
  commitment: the [key file](#key-file) and the [leaf files](#leaf-file). `cp_verify` uses the term
  in this sense. `cp_commit`'s usage text and some comments under `prover/` use it more narrowly,
  for the committer's proofs about the [anchor](#anchor), which are part of the key file.
- <a id="one-time-test"></a>**One-time test.** The test that `cp_commit` runs once on the
  [anchor](#anchor); its proof is in the [key file](#key-file). It is [LIN](#src-lin) for a batch of
  one. Together with [WF](#src-wf), it makes up the key file's one-time proofs.
- <a id="oracle-message"></a>**Oracle message.** A set of encoded main rows and [mask
  rows](#main-row), committed under a Merkle tree and sent in one prover message. The verifier reads
  only a few of its columns. Several oracle messages in the same prover message can share a tree.
- <a id="pad"></a>**Pad.** In [SEND](#src-send), the one random element that masks the answer to a
  sample. In the sub-proofs' code (`pad`, `slot_pads`), a [mask entry](#mask-entry). In a [weight
  table](#weight-table), the element that masks one stored [OOD answer](#record); the programs
  derive it from a secret [seed](#seed). It is kept as three entries at the end of the row's
  message. Two stored tables of one commitment are, by design, shown equal by the link proof in
  every entry but their pads and random entries, which each table derives for itself and which the
  link does not compare. Every proof that reads a table holds them to that table's own root and
  published answers, and to nothing of another table. No verifier can check that they are there, or
  that they are random.
- <a id="pending-claim"></a>**Pending claim.** A [linear claim](#linear-claim) on the weight matrix
  that a [sub-proof](#sub-proof), or [WF](#src-wf), leaves for [LIN](#src-lin). The source code also
  calls it a "pending row".
- <a id="plan"></a>**Plan.** The list of a [chain](#chain)'s rounds and messages, in protocol order,
  computed from the configuration. It gives each round's kind, proof of work and error bound, and
  each message's size. Each round's name, work bits and message sizes go into the [first
  query](#first-query), inside the digest of the parameters, so both sides follow the same plan. The
  error bounds do not go into the first query.
- <a id="private-randomness"></a>**Private randomness.** The random values that `cp_commit` and
  `cp_prove` use to hide what they send: the random entries encoded with each row, [mask
  rows](#main-row), pads, hiding polynomials and salts. The programs derive them with ChaCha20 from
  the [seed](#seed) and fresh system bytes, but the [anchor](#anchor)'s own masks and salts, and the
  pads of its [OOD answers](#record), come only from the seed. `cp_verify` does not check them, and
  for most of them it cannot.
- <a id="src-psc"></a>**PSC.** The sumcheck with a hiding polynomial. It reduces a claim about a sum
  to [linear claims](#linear-claim) with public values.
- <a id="range-check"></a>**Range check.** The approximate check on committed cells, RNG. It is
  designed to show that the cells are small integers, within a bound looser than the one that honest
  cells meet. It is not the committer's range proof; that is [WF](#src-wf). The programs' output and
  the [parameter file](#parameter-file) call it `smallness`, and identifiers call it `tail`.
- <a id="src-req"></a>**REQ.** The outer structure of one request's own proof: a sequence of
  [segments](#segment), each with one group of [oracle messages](#oracle-message).
- <a id="src-rng"></a>**RNG.** See [range check](#range-check).
- <a id="run"></a>**Run, run value.** One run produces one [key file](#key-file), one [leaf
  file](#leaf-file), or one batch with the proofs of its requests. Each [chain](#chain) has its own
  run value, a hash of the chain's label, the digest of its parameters and its 32 random bytes. In a
  batch's shared chain, the [binding tree](#binding-tree)'s root replaces those bytes. In every tree
  that a chain sends, the hash of each leaf begins with the chain's run value. One chain has no run
  value and sends no tree: the one-round chain that gives a table's [OOD seeds](#record).
- <a id="src-sc3"></a>**SC3, SCD.** Two kinds of sumcheck in the attention [sub-proof](#sub-proof),
  which runs several SC3 and one SCD. SC3 is a batched degree-3 sumcheck on sums of products of two
  vectors with public weights. SCD runs over two tokens and a key-value head, and its rounds have
  different degrees.
- <a id="seal"></a>**Seal.** The final digest of a [chain](#chain) that other chains build on. A
  later chain puts this digest in its [first query](#first-query). It is not written to any file.
- <a id="src-send"></a>**SEND.** The block that sends [oracle messages](#oracle-message) and takes
  out-of-domain samples of each, except those that only mask a fold or a proximity test's last step.
  The samples are designed to pin down one message for each oracle message.
- <a id="slot"></a>**Slot.** Several meanings. In the format, the position of a request's leaf in
  its batch's [binding tree](#binding-tree). The prover samples that position so that it doesn't
  depend on the order in which requests arrived. Identifiers and printed lines call it `place`. In
  `cp_prove`'s code and messages, "slot j" is the j-th request in a batch's list. In a
  [sub-proof](#sub-proof)'s code (`slots`), a slot is the index of one masked value that the
  sub-proof sends. In the library, a `MaskSlot` names one [mask entry](#mask-entry) of an [oracle
  message](#oracle-message).
- <a id="smallness-test"></a>**`smallness`.** See [range check](#range-check).
- <a id="src-split"></a>**SPLIT.** The block that replaces a [linear claim](#linear-claim) on two
  groups with one claim on each. It adds one [mask entry](#mask-entry) to one claim and subtracts it
  from the other.
- <a id="src-stage"></a>**Stage (of a proximity test).** A set of rows and their codes. A test with
  n stages runs n − 1 FOLDs, each with its code switch, and the last stage goes to
  [BASE](#src-base). A [stage](#stage) of `cp_verify` is something else.
- <a id="statement"></a>**Statement.** The part of a [chain](#chain)'s [first query](#first-query)
  that says what is being proved. The parameters are another part, fixed before a request is known.
  The type `Statement` in `blocks/iop/relation.h` is something else: the public running state that
  prover and verifier each keep.
- <a id="sub-proof"></a>**Sub-proof.** The argument for one part of the forward pass: a layer group,
  its attention, or the output head. It ends in [linear claims](#linear-claim) on the [oracle
  messages](#oracle-message) of its [segment](#segment). The sub-proofs for a layer group and the
  output head also leave one pending claim each on the [weight table](#weight-table), but the
  attention sub-proof leaves none. Identifiers call it `fork`.
- <a id="table-holder"></a>**Table holder.** The only translation unit of `cp_prove` that works
  inside a [leaf table](#leaf-table) during a batch's opening. It handles the table's masks and
  salts, the stored nodes of its tree, and the two passes over the weights. Identifiers call it
  `keeper`, and error messages `the keeper of the weight table`.
- <a id="tail"></a>**Tail.** The random entries encoded together with a row's message, also called
  the random tail. They are meant to hide the message as long as only a bounded number of positions
  are opened. In the sub-proofs' code (`Tail`, `HiddenTail`, `tail_f.cuh`), `tail` is the [range
  check](#range-check) instead.
- <a id="turn"></a>**Turn, pass.** A pass is the subset of a batch's requests that `cp_prove` proves
  at the same time; `-in-step` sets how many. A turn is one request's work on one [layer
  group](#layer-group) or, last, on the output head. Every request in a pass takes its turn on a
  group before any of them moves on to the next.
- <a id="src-wf"></a>**WF.** The committer's range proof in the [key file](#key-file), designed to
  show by lookups that every entry of the weight matrix is in its range.
