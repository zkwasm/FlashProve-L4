# Terms

Each entry says what a term means, not that this release achieves it.
Other terms and identifiers are in [`docs/glossary.md`](../docs/glossary.md). The terms for what Lean
proves are in [lean/TERMS.md](../lean/TERMS.md).

- **Acceptance policy**: the test that a request's checked tokens must pass, exact match or clipped
  gap. The parameter file sets it, and the client names it with `-policy`.
- **Binding**: several properties go by this word, so these pages say which one they mean. *Index
  binding* is what the soundness theorem on `verify` proves, at exact match and within its bound and
  [stated conditions](what-is-proved.md). One set of weights, or none, is settled when the
  commitment is made, and an accepted output is the model's at those weights. *Correctness of extraction* would
  say that those weights are the ones that the provider put in. *Opening-form binding* and *no two
  openings* would say that a commitment cannot be opened to other weights. In a line that
  `cp_verify` prints, "binding" is the programs' own word. Which of these is proved, and of what:
  [What is proved](what-is-proved.md).
- **Checked token**: a token that the acceptance policy applies to. The request names the range, and
  `cp_verify` prints it after `covered:`. A result says nothing about other tokens.
- **Client**: the party that sends a prompt, receives a reply and checks the proof.
- **Clipped gap**: the policy that scores checked tokens' [costs](../docs/glossary.md#cost)
  together. At most `cap` costs may exceed `C`, each cost is clipped to `C`, and the average must be
  at most `theta`. Acceptance under it [does not show that the reply is the model's
  own](../docs/limits.md#within-limits).
- **Commitment**: what one run of `cp_commit` makes from a model's weights and a
  [configuration](../docs/glossary.md#configuration): public files, secret files that the provider
  keeps, and the commitment ID.
- **Commitment ID**: 64 hex digits that identify one commitment, given to `cp_verify` with `-id`.
  Get it from [a party you trust](verify-a-proof.md#what-you-need-first).
- **Completeness**: the property that an honest prover's proof of a true claim is accepted. In Lean
  it is proved, in the random-oracle model, for the protocol's honest prover and `verify`, on served
  requests; for the CUDA prover it is tested by runs, not proved: [What is
  proved](what-is-proved.md).
- **`cp_commit`**: the GPU program that makes a commitment from a model's weights.
- **`cp_prove`**: the GPU program that proves requests in batches.
- **`cp_verify`**: the C++ verifier that clients run. It checks one proof per call on a CPU, without
  the weights.
- **Default setting**: the proof parameters of every example in this repository. Its error bounds [rely on the
  theorem on codes](../docs/limits.md#what-the-bounds-rest-on-depends-on-the-parameter-file), and it counts proof of work toward the target.
- **Exact match**: the policy under which every checked token must be the integer model's top
  choice. It is `cp_verify`'s default.
- **Executable form** (of `verify`): a Lean term that a machine can run. It computes `verify`'s own
  decision, and is proved sound in the same sense, at the same bound; both hold where the functions
  on bytes that stand for the hash function answer with 32 bytes. `lean-verify` runs it inside Lean.
  A run trusts Lean's evaluator, compiler and runtime, the helper library and the C compiler that
  builds it, and the file that drives the run.
- **Extractor**: in Lean's soundness statement, a fixed rule that gives weights, or none, from the
  commitment and the hash queries of whoever made it. It is a mathematical function defined with
  choice and is not shown to be computable. No theorem says that the weights that it gives are
  the ones the release's committer put in (it is proved for an illustrative committer written in
  Lean and, within its own budget of 2^48 hash queries, for the committer that the compiler yields
  from the Lean provers; no theorem ties either to the release's committing program: [What is
  proved](what-is-proved.md);
  [lean/THEOREM.md](../lean/THEOREM.md#what-the-commitment-gives-as-proved)); where it gives none,
  every accepted proof counts as a failure. It does not show that the prover knew the weights.
- **Full size**: the shape of Llama 4 Maverick 400B: 48 layers, 128 routed experts, a window of
  1,024 tokens.
- **Honest committer**, **honest prover**: the committer or prover that follows the protocol.
- **Honest run**: a run with unchanged programs and no compiler or hardware fault.
- **Index**: the word, in the definition that the Lean statements follow
  ([lean/TERMS.md](../lean/TERMS.md)), for what is committed to: here, the model's weights.
- **Instance**: the same definition's word for what one proof is about: here, a prompt with its
  output tokens.
- **Integer model**: the network with quantized weights and activations, computed in integers only
  and defined in [`docs/model.md`](../docs/model.md). Every proof is about it, not the floating-point model.
- **Key file**: `ivk/key.bin` holds the [anchor](../docs/glossary.md#anchor)'s root, its [OOD answers](../docs/glossary.md#record) and the committer's one-time proofs about the anchor.
- **Leaf table**: another weight table for the same weights, with its own key and masks. In [the
  standard terms](../docs/glossary.md#standard-terms), which the Lean pages and the paper use, a
  leaf table is a *copy*, and the commitment's first weight table, the *anchor*, is the *first
  table*. It isn't a leaf of a Merkle tree. Proofs are made against leaf tables, and each proof
  names the one it used.
- **Lean**: a proof assistant, which checks proofs written in its language.
- **[Mask](../docs/limits.md#a-lost-mask)**: a random value combined with a hidden value before
  anything depending on the hidden value is published or sent.
- **[OOD answers](../docs/what-is-not-hidden.md#record-sums)** (out-of-domain answers): masked sums of
  the weights, which every commitment publishes for good. The programs call
  them the record (`record 1`, `record=1`).
- **Other programs** (test builds and test programs among them): programs that are not the released programs. What a run found on them does not carry over to the released programs. A fault found on them may be in the released programs too. The whole definition: [`docs/glossary.md`](../docs/glossary.md#other-programs).
- **Parameter file**: `policy.txt`, which holds the model's shape, the proof parameters and the
  acceptance policy.
- **Prefill**: one forward pass of the integer model over prompt and reply. Proofs are about this
  pass, not token-by-token generation.
- **[Production build](../docs/glossary.md#production-build)**: what `prover/build.sh` builds when
  it is called without a target, or with a target that begins with `serving`.
- **Proof of work**: a hash search that the default setting requires before some
  [rounds](../docs/glossary.md#round), so each retry costs a malicious prover more. It counts toward
  the target but [does not lower the chance](../docs/limits.md#how-the-programs-count) that a false
  claim survives a challenge.
- **Provider**: the party that owns the weights, serves the model and makes the proofs.
- **[Provider's duties](../docs/limits.md#the-conditions)**: what a provider must always or never do
  if the masks are to cover everything shown. [Serving](serving.md#the-providers-duties) lists them.
- **Public files**: for whoever checks a proof, the three configuration files (`policy.txt`, which is the parameter file, `KL.bin` and `layout.txt`) and the public key directory: `key.bin` and `leaf/<k>.bin` ([Verify a proof](verify-a-proof.md#what-you-need-first)). Sentences on Lean's theorems use the words more narrowly, for the three configuration files alone ([What is proved](what-is-proved.md)). "The public settings" are what these three files state.
- **Random oracle**: an idealisation in the proofs that treats the hash as a random function that
  parties can only query. The programs use BLAKE2b and a second, fixed hash function, built on BLAKE2b's
  compression function. [Nothing is proved about either](../docs/limits.md#what-the-theorem-assumes).
- **Released programs**: `cp_commit`, `cp_prove` and `cp_verify` as this repository's sources build
  them (the production build). Which runs these pages tell as runs on them, and by what rule: [what was run on these programs, and what was not](../docs/limits.md#what-was-run-on-these-programs). The programs that made the runs that these pages tell as runs on the released programs, or on a test build of these sources, were built from sources that differ from these, in what is told there and in nothing else, and these pages say of them that they were "built from these sources"; what has run on programs built from exactly these sources is told there too. A
  result marked † or "not on the released programs" was not seen on the released programs: which kinds of result these are is described there. A † time, rate or memory peak is an estimate at most. The timed runs were made on the released programs: see *Timed runs*.
- **Reply**: the tokens the provider's engine served in answer to a prompt. In a request they follow the prompt in the window. Sentences on Lean's theorems use the word differently: there a reply is what the Lean prover hands out at a request, well-formed or not: a proof, or the empty proof (a reply that holds no proof). A refused request gets an empty proof; so, with small probability, does a served one ([limits](../docs/limits.md#the-verdict)).
- **Request**: a file, `<name>.req`, with a window of tokens and two numbers that say which tokens
  are checked. The client builds its own from the prompt's and reply's tokens.
- **Request that can be served**: a request can be served when every token is within the vocabulary,
  the output is not empty, and the committed model's values on it stay within the protocol's range
  conditions; which requests those are depends on the weights and is visible.
- **Served request**: a request that can be served whose claimed output is the model's own result on
  it. In Lean's words: a well-formed request (every token within the vocabulary, neither the prompt
  nor the output empty, at most 1,025 tokens together) on which the model has a result, and that
  result is the claimed output; a theorem says that the two descriptions name the same requests
  (`wellFormed_and_result_iff`). "Served" is also one value of the bit that is revealed for each
  request; the other is "refused": a request is refused if it is ill-formed, if the model has no
  result on it, or if its result is other tokens. A service that chooses to say "failed" (that is
  not the shipped programs' own "failed": [limits](../docs/limits.md#narrower-than-what-runs))
  reveals one label of three in its place (served, failed or other tokens).
- **Seed**: 32 random bytes that `cp_commit` samples and keeps in the secret key directory. Outside
  public-weights mode, only the provider is meant to hold the seed, and verifiers never need it.
- **Serving scripts**: the scripts that ship for serving: the vLLM integration, `driver/measure_rate.py`, the example scripts. What they check before they hand on a proof: [Serving](serving.md#prove).
- **[Soundness](what-is-proved.md)**: the property that a false claim is accepted only with small
  probability, whatever the provider does within a bounded number of hash queries. Lean has a
  soundness theorem for its own verifier, `verify`, not for `cp_commit`, `cp_prove` or `cp_verify`:
  [what it says, and what it does not](what-is-proved.md).
- **Synthetic weights**: weights that the programs generate themselves, without any model's files.
- **Target**: the error bound that every round must meet, in the programs' own calculation with
  proof of work counted: 2^-128 in every example in this repository. It is [not a security
  level](../docs/limits.md#the-soundness-figure), and none is claimed.
- **[Test build](../docs/glossary.md#test-build)**: a build made for tests, whose GPU programs
  accept test options and reject a real model's files.
- **Theorem on codes**: Theorem 4.6 of ePrint 2025/2055, a coding-theory theorem that the default
  setting's error bounds rely on, which the programs' accounting takes as given. A statement on
  codes held to follow from that theorem, in its case of two words, is proved in this repository's
  Lean library, by way of another formalization's theorem and a comparison of bounds; the paper's
  own form of the bound is not proved there, and that the library's statement follows from the
  paper's, and that these programs' parameters meet the theorem's conditions, rest on reading, not on
  a proof ([lean/LIMITS.md](../lean/LIMITS.md), L15).
- **Timed runs**: the runs behind the figures on [guide/performance.md](performance.md) that carry
  no mark, made on the released programs: within three hours of one day, three runs each of six
  kinds of run at full size (at the default setting: on a third party's public 4-bit files a
  commitment with one request, and services at 24, at 64, and at 8 and 1 requests a batch; on
  synthetic weights a commitment, its audit and services at 24; at that setting with a clipped-gap
  policy, on the public files, sessions with every reply token checked and like sessions that check
  none); a clipped-gap example was made once, and one call and one service at 64 on synthetic
  weights were each run once; the call and the service failed. More of them: [the timed
  runs](../docs/limits.md#the-timed-runs). Every other run of `cp_commit`, `cp_prove`, `cp_verify`
  or the Lean verifier that was timed bears a † or says in words that it was not on the released
  programs, or that it was on them and not made to be timed, or belongs to the runs that timed the
  serving scripts' check, by itself and beside a serving `cp_prove`, on the released programs, in
  one of which a constant of the scripts is not this repository's.
- **Top choice**: the token with the largest logit at a position; on a tie, the lowest token id.
- **`verify`**, the Lean verifier (in Lean `VerifierAndGoals.verify`, at `VerifierAndGoals.theDescription`): the verifier of [the soundness theorem `sound_of_verify_at`](what-is-proved.md). It does not read the prover's files. It
  takes the public files, the commitment, the prompt, the output and the proof. `VerifierAndGoals.verify` is a definition in [lean/Ladder.lean](../lean/Ladder.lean) (definitions
  only: no proof is in that file). As written it cannot be executed: see *Executable form*.
- **[Weight hiding](weight-hiding.md)**: the goal that clients learn nothing about the weights from
  what a provider publishes and sends, beyond what
  [`docs/what-is-not-hidden.md`](../docs/what-is-not-hidden.md) lists.
- **Weight table**: all the weights, arranged as one matrix whose cells are the weights' base-256 digits. Each row is encoded together with random [masks](../docs/glossary.md#mask), and a Merkle tree is built over the columns of the encoded table.
- **Window**: prompt and reply as one row of token ids, padded with filler to a length set in the
  parameter file.
- **`WITHIN LIMITS`**: see `YES`.
- **[`YES`](verify-a-proof.md#when-to-accept)**: `cp_verify`'s last line when it accepts under exact
  match. Under clipped gap it prints the weaker `WITHIN LIMITS: …`. Both exit with code 0, so [tell
  them apart by the last line](../docs/verifying.md#exit-codes).
- **Zero knowledge**: the cryptographic form of [weight hiding](weight-hiding.md): all a verifier sees
  could be produced ("simulated") without the weights, from the public settings and each proof's prompt and output tokens. The settings include scales chosen from the weights. In Lean's theorems, in addition, one bit per request is revealed (served or refused), and it depends on the weights: over many requests, whoever tries each candidate token for a prompt learns the model's next one from this bit alone, unless all are refused ([limits](../docs/limits.md#the-verdict)). No statement bounds what many bits reveal together.

† Not on the released programs: [which kinds of result these
are](../docs/limits.md#on-other-programs).
