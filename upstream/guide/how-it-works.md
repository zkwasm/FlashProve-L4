# How FlashProve works

A provider commits to a model's weights, then proves per request that the checked tokens pass an
acceptance policy against that model. In the expected use, an accepting check ends in
`WITHIN LIMITS`, which says much less than `YES`.

## Who runs what

- `cp_commit` and `cp_prove`: the provider, on one NVIDIA H100, makes a commitment and proves
  requests in batches.
- `cp_verify`, the C++ verifier: the client checks one proof per call, on a CPU, without the
  weights.

## The files

`cp_commit` prints the commitment ID and writes the secret key directory `isk` and the public key
directory `ivk`, holding the key file and the leaf files. The provider publishes the ID, `ivk` and
the configuration: `policy.txt`, `KL.bin` and `layout.txt`. It never publishes, copies or restores
`isk`.

Each request has a request file `<name>.req`, a token file `<name>.y` with the next token or 0, and
a proof `<name>.proof`. The client builds `.req` and `.y` from the tokens it sent and received,
never from files that came with a proof. Then it runs:

```text
cp_verify <configuration directory> <public key directory> <request file> <token file> <proof file> -id <commitment ID> [-policy <expected policy>]
```

An accepting call ends with exit code 0 and a last line `YES` or `WITHIN LIMITS: …`, but that alone
is [not enough](verify-a-proof.md#when-to-accept).

The [commitment ID](../docs/architecture.md#the-files) is a hash, in 64 hex digits, of the
configuration, the Merkle root and shape of the first weight table, a masked encoding of the
weights, and the out-of-domain (OOD) seeds and answers. Hold the three configuration files as you
hold the ID: the ID covers them by design, but the soundness theorem on `verify` is stated at given
public files and does not itself say that one ID cannot verify under two sets of them. The ID does
not cover the leaf files or the key file's proofs, which `cp_verify` checks on reading.

The client must trust what no call checks: the commitment ID, the expected policy and its numbers,
the tokenizer and chat template, and `cp_verify` itself. Get the ID from [a party you
trust](verify-a-proof.md#what-you-need-first).

## What a proof says

A request is a window of T token ids, holding prompt, reply and filler, and two numbers, r and e.
Client and provider must use the same filler.

Positions count from 0, and position t predicts token t + 1, so the range r to e − 1 checks tokens
r + 1 to e. For a prompt of P tokens and a reply of R, r = P − 1 and e = P + R − 1. Using the reply
tokens' own positions leaves the first reply token unchecked. Nothing outside the range is covered.

An accepted proof [does not show](../docs/limits.md#who-chooses-the-model) that the provider
committed the advertised model or a sensible one, or that the reply is complete.

## The two acceptance policies

Under exact match, a `YES` [is meant to show](../docs/limits.md#what-a-yes-covers) that each checked
token is the committed integer model's top choice at its position, up to the [error
bound](../docs/limits.md#the-soundness-figure).

[Clipped gap](../docs/glossary.md#clipped-gap) gives each checked token a cost: how far its logit
lies below the top choice's, plus a constant. It accepts, with `WITHIN LIMITS: …`, if at most `cap`
costs exceed `C` and the costs, clipped to `C`, average at most `theta`. A token that costs more
than `C` is charged `C`, so it may be any token at all; there are at most `cap` of them, and fewer
where the average binds. The cost is based on the Token-DiFR score at temperature 0 (arXiv
2511.20621), with [four differences](../PUBLISHED-WORKS.md#a-published-method), and no sampling seed
enters a proof.

`policy.txt` sets the policy. The commitment ID cannot be read to show which, so the client names it
with `-policy`.

`WITHIN LIMITS` [says only](../docs/limits.md#within-limits) that the checked tokens' costs together
meet the limits that the client named, not that the reply is the model's.
With large limits, or close logits, almost any tokens pass, and a verifier cannot tell.

## The integer model

A proof needs exact arithmetic, so the committed model is an [integer
model](../docs/model.md#the-integer-model). It approximates the floating-point model, and [no proof
says how close the two are](../docs/limits.md#the-integer-model).

In the [expected use](../docs/glossary.md#ordinary-serving), vLLM or another engine serves the
floating-point model. The proof comes afterwards, from one prefill pass of the integer model, under
clipped gap. This has not run live as one service on a
trained model. Exact match fails wherever the engine's token differs from the integer model's top
choice.

## What is established

Soundness is proved of the verifier written in Lean, a proof assistant, at exact match.
Completeness, where the model has a result, and zero knowledge are proved in Lean for the protocol,
at [full size](terms.md) and at the exact-match policy, each up to a small stated error, with fresh
random masks, for one commitment and one request per proof and, after compiling, on one [leaf
table](terms.md), not for batches. One bit per request is revealed (served or refused), and it
depends on the weights. The honest prover in these theorems is a Lean definition, not the CUDA
prover. Nothing is proved of the C++ verifier that the examples and most users run, of the CUDA
prover, or of the committer, and no theorem says that they run the protocol the Lean verifier
checks: that rests on tests. [What is proved](what-is-proved.md) has the rest.

By design, the programs mask proofs and a commitment's public files by default, apart from a
[declared list](../docs/what-is-not-hidden.md#what-is-not-hidden). [Weight hiding](weight-hiding.md)
says what that gives.

See also [known limits](known-limits.md).

## The proof system

The proof is [hash-based](../docs/architecture.md#the-proof-system-in-outline): an interactive
oracle proof made non-interactive with Merkle trees and Fiat-Shamir. Sumcheck and lookup arguments
are designed to reduce "these committed numbers are a correct run of the model" to a few linear
claims. A [proximity test](../docs/protocol.md#5-tools) is designed to show that the committed data is close to a codeword, and to show these claims too. It is the test of Chiesa, Fenzi and
Weissenberg, ePrint 2026/391, with
[differences](../PUBLISHED-WORKS.md#the-paper-that-this-protocol-builds-on). [Two published attacks on
Fiat-Shamir proofs](known-limits.md) have not been examined against
this protocol or these programs.

Merkle trees, Fiat-Shamir transcripts and the commitment ID use standard BLAKE2b. A [second, fixed hash
function](../docs/glossary.md#hash-mode), built on BLAKE2b's compression function, feeds nine digests
into a proof's statement and is outside BLAKE2b's published analysis.
The printed error bounds [rest on assumptions](../docs/limits.md#what-the-theorem-assumes), none
proved for the programs.
[What is proved](what-is-proved.md) lists them.

## Where the code is

- `blocks/`: the shared library.
[Programs that call the `blocks/` library directly](../docs/limits.md#false-claims-in-test-programs)
can be made to accept false claims; at least one way works. That `cp_verify` is not exposed
rests on reading its source, not on a test. Check proofs with `cp_verify`.
- `prover/`: `cp_commit`, `cp_prove`, example configurations.
- `verifier/`: `cp_verify`.
- `integrations/`: the example front end for vLLM, and client scripts.
- `examples/`: stored proofs on synthetic weights, and `examples/maverick/`, from a public Llama 4
  Maverick download to a checked proof.
- `lean/`: the Lean library.
- `tools/lean-verify/`: `lean-verify`, which runs the Lean verifier on a full-size proof's files.
- `tests/`, `run_suite.py`, `SUITE.md`: the tests.
