# Limits

This page holds the larger limits that we know of in FlashProve, each with what can go wrong and
what you can do about it. For one page only, read [Known limits](../guide/known-limits.md). The list
may be incomplete: the last reviews behind it found new limits.

**How to read the entries.** Each entry says what can go wrong, what you can do about it, and what
evidence there is. Most end with a Basis line; [Kinds of evidence](#how-a-thing-is-known) defines
its labels. An entry's last line names its rows in [the table of all rows](limits-all-rows.tsv). A
row whose ID begins with `LIM-` holds, in full, what the entry here tells in short: the runs one by
one, the figures and the bounds on each. The table is large, so look a row up by its ID:
`grep -P '^SO-15\.7\t' docs/limits-all-rows.tsv`.

**Evidence.** Many of the tests and readings that this repository reports were not made on the
released programs, but on programs built from [other sources](glossary.md#other-sources); some, the
tests with deliberately cheating provers among them, on programs that also held test-only code. The
compiled programs differ too; the machine code that differs is not shown equivalent and has not been
read. Two of the five tests of weight hiding have run on the released programs: the first at small
size on synthetic weights and at full size, once on synthetic and once on a real model's public
weights, and the fifth once, at small size on synthetic weights. One of the fifth's two audits, the
audit of a commitment, also ran on them at full size, in the timed runs (row LIM-63.2, line 5). At
full size the first test has also run on other programs, and the fifth's two audits have run
together only on other programs. The second and the third ran on test builds of other sources, which
hold the code that injects a fault, and the fourth on programs of other sources that held test-only
code. What ran on which programs: [what was run on these programs, and what was
not](#what-was-run-on-these-programs). An entry says "not on the released programs" of a run on any
other programs; where it says neither that nor "on the released programs" of a run, and does not
tell it as made "in the timed runs" ([what these are](#the-timed-runs)), take the run as not made on
the released programs. A fault of the code, told in the present tense, holds of the released
programs.

**Elsewhere.** Measured figures are in [guide/performance.md](../guide/performance.md), and
[glossary.md](glossary.md) defines the terms. Smaller limits and bad results get a line each in
[limits-further.md](limits-further.md), and the vLLM integration's limits are in
[integrations/LIMITS.md](../integrations/LIMITS.md). Where this repository's own words are known to
be wrong (a page, a comment or a printed message that is untrue or claims too much),
[known-errors.md](known-errors.md) gives the place, what it says, what is true and what to do.

The project is not maintained and not accepting contributions.

(The terms for what Lean proves: [lean/TERMS.md](../lean/TERMS.md).)

## In short

**If you check replies**

- [The provider chooses the committed model and its settings.](#who-chooses-the-model)
- [A `YES` covers only the positions that you name, in an integer model.](#what-a-yes-covers)
- [Exit code 0 alone does not mean accepted.](#what-counts-as-a-pass)
- [A pass under a clipped-gap policy says less than a `YES` under exact match.](#within-limits)
- [A proof does not name a client or a time.](#no-freshness)
- [A hostile file can exhaust memory or time.](#hostile-files)
- [The error bounds that the programs print are their own calculation, not a security
  level.](#the-soundness-figure)
- [An honest proof can be rejected.](#an-honest-proof-can-be-rejected)

**If you review or build on the code**

- [The library under the programs can be misused.](#false-claims-in-test-programs)
- [No theorem is about `cp_commit`, `cp_prove` or `cp_verify`.](#which-verifier-has-a-theorem)
- [What the bounds rest on depends on the parameter file.](#what-the-theorem-assumes)
- [The programs' calculation of their error bounds rests on steps that are assumed or not
  checked.](#how-the-programs-count)

**If you serve a model with secret weights**

- [One main risk: a lost mask.](#the-main-risk)
- [Where it stands.](#the-verdict)
- [What a provider must do.](#the-conditions)
- [The biggest risks.](#what-went-wrong)

**For everybody**

- [What was run on these programs, and what was not.](#what-was-run-on-these-programs)
- [Few kinds of test have run at the default setting, and many ran only once.](#few-runs)
- [The reviews are not independent.](#review)
- [What the Lean proofs cover.](#what-the-lean-proofs-cover)
- [What was never tried.](#what-was-never-tried)
- [Kinds of evidence: what the labels in a Basis line mean.](#how-a-thing-is-known)

**If you run the programs**

- [What the programs can and cannot do.](#what-the-programs-can-and-cannot-do)
- [Running the programs safely.](#running-the-programs-safely)
- [The vLLM integration.](#the-vllm-integration)
- [The integer model is not the floating-point model.](#the-integer-model)

## Soundness

This section distinguishes three things: the programs in this repository, the protocol as [this
protocol's technical report](glossary.md#technical-report) describes it (a technical report that is
not public and not refereed), and the protocol as Lean describes it. Each entry below is about the
programs unless it says otherwise, but what a `YES` means is stated for the protocol. The programs
are known to differ from the protocol, and it has not been shown that the differences do not matter
([No theorem is about `cp_commit`, `cp_prove` or `cp_verify`](#which-verifier-has-a-theorem)).
A run is to be taken as made at the [default setting](glossary.md#default-setting) only where its
entry says so ([kinds of evidence](#how-a-thing-is-known)). Few kinds of test have run at the
default setting ([which](#few-runs)).

**Soundness, in short.**

- **Proved in Lean:** soundness at exact match for `verify`, the Lean verifier of the protocol, as
  Lean describes it, for configurations that its scope test passes, fixed before the hash is drawn
  ([lean/THEOREM.md](../lean/THEOREM.md)). That the programs run the protocol as Lean describes it
  is not proved; it is tested. One command, `lean-verify`, runs `verify`'s executable form inside
  Lean. A run trusts Lean's evaluator, compiler and runtime, the helper library and the C compiler
  that builds it, and the file that drives the run ([what the Lean proofs
  cover](#what-the-lean-proofs-cover)).
- **Not proved:** anything about `cp_verify`, the C++ verifier that users run. [No theorem is about
  `cp_commit`, `cp_prove` or `cp_verify`](#which-verifier-has-a-theorem), and none bounds the chance
  that `cp_verify` accepts a false file.
- **Argued on paper:** conditionally. It is an argument about the protocol on paper, at the report's
  own setting, with the hash taken as a random oracle, whose conditions the report shows for no
  configuration. It is not about the programs. It rests on the five things told under [what the
  bounds rest on](#what-the-theorem-assumes), none of which is proved for the programs.
- **Tested:** few kinds of test have run at the default setting ([which](#few-runs)): evidence, not
  a guarantee. Nobody searched for a lie that passes where the sampling predicts a rejection.
- **Accepted, though the committer lied:** `cp_verify` has said `YES` on the files of a lying
  committer whose stored table was altered at few positions.
  - In 24 full checks the table was altered at 1, 16 or 256 of 8,388,608 positions. Each `YES` was
    of a true statement. Up to a collision of BLAKE2b-256, each of the six altered tables is the
    honest table changed in exactly 1, 16 or 256 columns, so every one of its 256 words lies at that
    distance from the Reed–Solomon codeword of the committed weights: far below half the code's
    least distance (3,899,393), with every other codeword beyond the verifier's radius. That was
    checked by a separate encoding on CPUs.
  - In a second run 180 more checks passed, on 45 of 80 further lying commitments with one table
    altered at 257 to 524,287 of 8,388,608 positions (in 23 of the 45 the leaf table: a lying leaf
    file under the key file and commitment ID of the run with nothing altered). The largest count
    accepted was 131,072, the least refused 8,192, and all 16 commitments at 262,144 or more were
    refused. Each `YES` stood beside the reply that the same edited programs gave with nothing
    altered. By a separate program's count of the changed columns (BLAKE2b-256 taken as
    collision-free), every accepted table lies far inside the distance (3,899,392 positions) up to
    which a table is close to the honest codeword alone. That count measures from the run with
    nothing altered; that its tables hold the honest codewords was checked at all positions under
    one of the run's four test seeds only (row SO-15.7).
  - There is no threshold: in the second run, by the edited programs' own print, which no other
    program has counted, a check passed exactly when none of its sampled columns was altered. Each
    commitment was made once, so this is for one try.
  - Small size; synthetic weights; not on the released programs; under the default setting's
    parameter file as the small example `target_small_f2_roomy` has it (rows LIM-2 and SO-15.7).
    That `cp_prove`'s own kernels use those integers is not shown. That a table within that distance
    is close to one codeword only is the error-correcting code's property, on paper for the
    protocol; [of the programs' own commitment no binding theorem is
    proved](#what-the-theorem-assumes). (Rows SO-15.7 and LIM-2.)
- **The library:** [a program that calls the `blocks/` library
  directly](#false-claims-in-test-programs) can be made to accept a false claim, and test programs
  have done so; that `cp_verify` is not exposed to this rests on reading its source, not on a test.
- **Other settings:** under a parameter file other than the default setting's, a `YES` rests on that
  file's own numbers besides ([what a provider's own file can still do](#what-the-theorem-assumes)).

**What you can rely on.** The attacks ran mostly not on the released programs, and of the stronger
kinds of attack few have run at full size ([what has run at the default setting](#few-runs)). A
`YES` does not tell you which model was committed, and it covers only the positions that you name,
in an integer model. So take a `YES` under exact match as a check, not as a proof of security. Take
it so only if you got the commitment ID from the party that vouches for the model, over a channel
that the server you are checking does not control. The request and the policy must be your own, and
you must have built `cp_verify` yourself ([the rule, in one place](verifying.md#when-to-accept)).
`WITHIN LIMITS` says much less.

### If you check replies

The mathematics behind a `YES` is not about the three programs `cp_commit`, `cp_prove` and
`cp_verify`. [No theorem bounds the chance](#which-verifier-has-a-theorem) that a program of this
repository accepts a false file. In full: row LIM-1.

<a id="who-chooses-the-model"></a><a id="what-you-bring"></a><a id="what-the-id-does-not-cover"></a>
#### The provider chooses the committed model and its settings

Under exact match, a `YES` ties a reply to one fixed integer model, the one whose [commitment
ID](glossary.md#commitment-id) you had before you sent the prompt, up to [the error
bound](#the-soundness-figure). That is what the protocol is meant to give you. For the programs the
last step, from an accepted proof to the model's top choice, has not been shown ([No theorem is
about `cp_commit`, `cp_prove` or `cp_verify`](#which-verifier-has-a-theorem)). Nor has the binding
of the weights to the commitment ID ([why](#what-the-theorem-assumes); row LIM-2).

A `YES` does not tell you which model that is. The provider alone chooses the weights, the scales
and every setting, and the verifier checks only that the settings are within accepted ranges, not
that they make sense for a model. If the ID, the files and the programs all reach you in one
package, a `YES` shows only that they fit together.

**What to do.** Get the commitment ID before you send the prompt, from [the party that vouches for
the model](glossary.md#vouching-party), over a channel that the server you are checking does not
control. Where you can, take it from a place that shows every client the same ID, such as where the
commitment is published: a provider that gave different clients different IDs could serve them
different models, and only a shared, public ID makes that visible. A verifier call on a single file
prints the ID that it computes from the files it was given. So never take the ID from a verifier's
output, or from files that came with a response. Pass the whole ID to `cp_verify` with `-id`, which
compares it with the files. If you do not have an ID, stop. Build `cp_verify` yourself.

**Worst case:** the provider commits to a small model that gives the answer it wants, or to settings
that round every layer to zero.

**[Basis](#how-a-thing-is-known):** run, not on the released programs, at a small size; read. Rows:
SO-6, SO-6.4, SO-8, SO-8.1, SO-8.3, SO-30, SO-49.2. Two IDs for the same weights, and what the ID
does not cover: row LIM-2.

<a id="what-a-yes-covers"></a>
#### A `YES` covers only the positions that you name, in an integer model

Under exact match, a `YES` is meant to show that each checked token is the top choice of the
committed integer model at its position, up to [the error bound](#the-soundness-figure).

For the committed weights, a proof claims three things, each only "as far as the sub-proofs of its
segments check a model's steps" (the words of [the format's page on a proof
file](format/10-proof-file.md), from which the first is taken too):

1. No value of the integer model's run on the request, by the weights in a stored table, leaves a
   table or a range at any position.
2. The checked tokens meet the acceptance policy.
3. If the last position, T - 1, is in the range, the next token is the model's top choice there,
   under every policy.

A request contains a window of T tokens and a range from r to e. The checked tokens are r + 1 to
min(e, T - 1), and if e = T the proof also covers the next token after the window. Under exact
match, a `YES` is meant to show that each of these tokens is the top choice of one exact integer
program, given every token before it, up to [the error bound](#the-soundness-figure). For the
programs, the step from the conditions that a proof establishes to the model's reply has not been
shown ([No theorem is about `cp_commit`, `cp_prove` or `cp_verify`](#which-verifier-has-a-theorem)).
The committed key and the configuration files determine which program that is. The result does not
cover tokens 0 to r, and says nothing about sampling (a request does not contain a seed), text or
the floating-point model (see [the integer model](#the-integer-model)). A truncated reply verifies
as a shorter reply.

At full size, only a few runs have covered tokens inside the window. Most full-size runs that
measured speed cover only a request's last position.

**What to do.** Build the request file yourself from the tokens that you sent and received, and
compare the `covered:` line with your own range (see [when to accept](verifying.md#when-to-accept)).
Because a truncated reply also verifies, check that the reply ends with one of the model's end
tokens.

**[Basis](#how-a-thing-is-known):** read (`verifier/src/cp_verify.cpp`); reported (the run behind
`examples/full-size/` and the one under a clipped gap). Rows: SO-18, SO-28, SO-29, SO-36, E9. The
runs one by one: row LIM-3.

<a id="what-counts-as-a-pass"></a><a id="no-plain-yes"></a>
#### Exit code 0 alone does not mean accepted

`cp_verify` compares three things with the values that you pass in: the commitment ID, the next
token and the policy's values. It cannot tell whether those values are your own. Three more
comparisons are left to you: the range, the end of the reply and the tokenizer. Skip them, and a
`YES` can be about the wrong thing.

Three other cases need care:

- The middleware that ships passes `cp_verify` the parameter file's own policy, and a provider's own
  check may do so too. That is right for a provider's filter but wrong for a client, so copy no
  provider's call into a client's example.
- A call with `-key` or `-leaf` checks only one file. Its last line begins `PART PASSED:`, with exit
  code 3, and is never `YES`.
- At the default setting every accepted proof counts [proof of work](glossary.md#proof-of-work), so
  a plain `YES` does not mean that no work was counted. See [point 6 of the
  rule](verifying.md#when-to-accept). Whether to count the work is up to you ([the
  figures](#the-soundness-figure)). The provider's parameter file also picks what the bounds rest
  on, and the last line does not show that either ([details](#what-the-theorem-assumes)).

**What to do.** Accept a proof only if it satisfies the rule in
[verifying.md](verifying.md#when-to-accept). Pass in only your own values, and check the printed
fields against them.

**Worst case:** a wrapper accepts exit code 0 even though no token of the reply was checked.

**[Basis](#how-a-thing-is-known):** read. Rows: SO-7, SO-49, SO-49.1. How far `check_proof.py` has
been tested: row SO-49.

<a id="within-limits"></a>
#### A pass under a clipped-gap policy says less than a `YES` under exact match

Exact match accepts only the integer model's top choice. A [clipped-gap](glossary.md#clipped-gap)
policy also accepts tokens that score close to it. Under a clipped-gap policy a passing check that
covers a token of the request ends `WITHIN LIMITS`, not `YES`. It tells you only that the checked
tokens' costs meet the limits you named: their average, with each cost counted only up to the clip,
and the number of costs above the clip. It does not say that the reply is the model's. What it says
about a single token depends on your limits, and can be nothing.

[The soundness theorem for the Lean verifier](../lean/THEOREM.md) does not cover it. Its premise is
that the policy lines `C`, `theta`, `cap` read 256, 256, 0. That is a premise, not a test in the
text of `verify` (that nothing deeper inside `verify` rejects other policies rests on a reading),
and files with another policy are in scope (`VerifierAndGoals.zGap` is one). There this theorem is
silent. Soundness without the exact-match premise (`PackageSound`, `lean/Ladder/Package.lean`) is
not proved at all files in scope, and is not expected to hold. So a user, or the program that calls
`verify` for them, must check these three lines. The compilation theorem
(`innerCompiledVerifier_sound`) does not cover it either: it claims that every output token is the
model's first choice, and no other acceptance policy is covered
([lean/LIMITS.md](../lean/LIMITS.md), V5). The theorem under a policy
(`soundUnderPolicy_of_verify_at`), a companion of the soundness theorem for the Lean verifier, does
cover other policies, weakly, and of `verify`, not of `cp_verify`: for all public files in scope, an
output that `verify` accepts meets the policy that the files state, at the weights that the
protocol's own extractor returns, except on an event of probability at most the stated bound. By
itself it bars nothing against a committer who commits to weights under which every output meets the
policy ([lean/THEOREM.md](../lean/THEOREM.md), "The theorem under a policy: other acceptance
policies"). The written soundness argument reaches the clipped-gap policy only through an extension
under further size conditions: it covers the conditions of the proof's last segment, which runs the
clipped-gap equations, only in part. Under that policy the mean and the count run over one output
that the prover picks, and prompt tokens are not scored; the stored clipped-gap example's accepted
reply is itself not the model's first choice at one position, so at exact match it would be false.

A provider can commit to a model whose scores are close together, and then every token passes. The
provider also sets the length of the reply, and the allowance grows with it. `WITHIN LIMITS` and
`YES` both come with exit code 0, so only the last line and the `claim=` field of `result-v1`
distinguish them. An end token is scored like any other token, so a provider can end a reply early
and still pass. Of the separate `cp_verify` calls that the docs report, none checked a reply that
misses its limits. The output head's relations let a prover state, for each counted token, any
hidden cost from the true one up to the clip C (docs/format/11-4-output-head.md); a proof with such
a cost satisfies the same claims, by the source's comments (docs/format/11-4-output-head.md, item
9). The hidden costs are a witness, not part of the claim, and are not unique.

A reply that meets fitted limits says little about which of two similar models served it, but one
that misses them says a lot. In one trial on a small synthetic model, replies from a slightly
altered model met limits fitted to the honest model: 47 of 64 served in bfloat16, and 54 of 64 in
float32. The replies were scored, not proved, with one altered model and one seed. In another trial,
under looser limits, all 64 replies from a slightly altered model passed, though for 40 of them only
a written report was kept.

A trial on a real model also shows weaknesses. One run scored long stored replies from a stock
inference engine running Llama 4 Maverick's public weights, against limits fitted to the 16-bit
engine's honest replies. With one token replaced by the model's second choice, 28 of 32 replies
still met the limits. The replies were scored, not proved, in one run, not with the released model
package. More replies still met the limits: 27 of 32 with one random token; 49 of 96 sampled at
temperature 1; in a second run 86 and 85 of 96 at two other sampling settings. Neither run tested a
model with altered weights, and the first no cheaper model. When the counts apply to the released
integer model: row LIM-5.3. So these limits hardly tell another sampling setting of the same model
from greedy serving. And nothing on [the route for replies that were served
earlier](../integrations/vllm/REFERENCE.md#prove-replies-that-were-served-earlier) asks for a
temperature: there a sampled reply that meets the limits gets a proof as any other reply does, and
the proof shows only that it meets them (reasoned from the code; in a run told in
[integrations/LIMITS.md](../integrations/LIMITS.md), 14 cuts of replies that Maverick had sampled
met the limits and got a proof).

**What to do.** When you get the commitment ID, read the limits in the parameter file that it
covers. Decide before you send a prompt whether you accept them, and pass them with `-policy`. Do
not take the values for `-policy` from files that came with a response. Read also the final norm's
two exponents, the parameter file's lines `MF` and `KF`: `cp_verify` prints them in no line. How
much a clipped-gap policy excludes depends on their difference as well as on the policy's three
numbers. At a small difference, a policy whose threshold fits the file's line can exclude nothing,
whatever the weights. That is a theorem about Lean's model (`Policy.exists_policy_of_close_scales`,
lean/Policy/Front/MeetsPolicyIdleAtScales.lean). If you cannot assess the numbers, accept only exact
match.

**Worst case:** with large limits, or with a committed model whose scores are close together, almost
any tokens pass. `cp_verify` rejects a few policies by their numbers (`NO: policy`). A policy that
it does not reject can still let every token through, and such a call ends `WITHIN LIMITS`.

**[Basis](#how-a-thing-is-known):** read; argued on paper; run, not on the released programs, and
reported (the runs that made proofs under fitted limits); scored in runs (no proof made). That all
requests of a run that was timed got a proof says nothing about how often honest replies do: these
runs add no new kind of reply or prompt, and none is a rate of acceptance. Rows: SO-25, SO-26,
SO-26.4, SO-27, SO-29, SO-33, SO-35, T43. The runs that made proofs under fitted limits, every
count, and what Lean and the report say of such a pass: rows LIM-5.1 to LIM-5.3.

<a id="no-freshness"></a>
#### A proof does not name a client or a time

A valid proof verifies for anyone who holds the same request, commitment ID and configuration, at
any time. It does not show when it was made, who it was made for, or that it answers the request at
hand. Nor can the verifier tell a current leaf table from one that the provider has retired.

**What to do.** If freshness matters, put your own nonce in the prompt. Keep the request, the reply
and the proof private.

**Worst case:** a proof made for one client is replayed to another as the answer to their identical
request.

**[Basis](#how-a-thing-is-known):** read; run (the sweep over a real proof's fields: the public test
`verifier.proof_fields_sweep`, which passed, over files that a test build made). No test has walked
a real proof's later fields one by one, and none of these tests runs at full size. See [what has run
at the default setting](#few-runs). Rows: SO-19, SO-19.1, SO-19.2, SO-21.

<a id="hostile-files"></a>
#### A hostile file can exhaust memory or time

`cp_verify` does not limit its own time or memory, or the sizes that a file makes it plan for. It
processes the configuration before it compares the commitment ID, and it reads each file in full
before it tests the file's length, except the request and the [token file](glossary.md#token-file)
in a whole call, whose lengths it tests first. It also follows symlinks.

**What to do.** Run `cp_verify` under your own memory and time limits, set no lower than what it
takes to check an honest proof on your machine. Treat a call that hangs or is killed as not
accepted. Keep all the files for a call in directories that only your account can write to.

**Worst case:** a crafted file exhausts the machine or makes the call fail. No crafted file that was
tried, or followed through the source by reading (rows SO-15.7, PS-19), makes the call end in `YES`
on a claim that is false; that rests on tests and reading, not on a proof, and whether a file in
which some of the cases of rows PS-6 and PS-19 occur can do so is not known (row PS-50.1). Crafted
files whose claim stays true were accepted (row SO-15.7), a lying prover's proofs of true statements
and a lying committer's files among them; that each such `YES` is no false acceptance rests on an
argument and a reading, not on a test.

**[Basis](#how-a-thing-is-known):** read; run and reported, mostly not on the released programs
(rows SO-15.5, SO-15.7). Rows: SO-16, SO-16.1, SO-16.2, SO-31.

<a id="the-soundness-figure"></a>
#### The error bounds that the programs print are their own calculation, not a security level

**In short.** The programs print bounds on the chance that a false reply is accepted. The bounds are
calculations on paper, and they rest on assumptions, on published results whose proofs this project
has not checked, and on proofs in the technical report. They are not machine-checked, and have not
been checked independently. These figures are the programs' own arithmetic, on stated assumptions;
they say nothing of the whole system or of an attack's real cost, so they are not a security level.

Never quote "128 bits" by itself. None of the figures here is a security level. They are the
programs' own accounting of the protocol's rounds, made by a planning call, not a bound proved for
these programs. By the programs' own accounting, under exact match, at the default setting and at
full size, a false reply is accepted with probability below about N/2^126 if proof of work is
counted, and below about N/2^96 if it is not (reckoned by a planning call:
`cp_verify -plan accounting` at the default's parameter file). A false reply is one that differs, at
a checked position, from the committed integer model's top choice. N counts all the hashes that the
committer, the prover and the verifier compute. After 2^64 computations that is one in 2^62 with the
work counted, and one in 2^32 without. The bound counts computations because a malicious prover can
retry until a challenge suits it. The bound is for one key, with public parameters and a policy that
are fixed in advance and not taken from the prover.

Counting the [proof of work](glossary.md#proof-of-work) rests on [a separate
assumption](#how-the-programs-count), so read both columns of the table. By the programs' own
calculation, at the default setting and full size:

| | proof of work counted | proof of work not counted |
|---|---|---|
| the weakest single round | 128.00 bits | 96.23 bits |
| the whole bound "as one number", as `cp_verify -plan accounting` prints it (by the planning call) | 126.21 bits | 96.23 bits |
| all rounds of one key file's, one leaf file's and one request's proof, added | 119.38 bits | 92.42 bits |
| the bound for a cheater who computes no hashes, for one call, at the planner's count of the verifier's own calls (printed by no program) | 108.32 bits, calculated from the planning call's prints | 76.55 bits, calculated from the planning call's prints |

The third row is one of what the printout calls "stricter readings of the same rounds, which decide
nothing". For the fourth row: Even if a cheater computes no hashes, N is not zero, because the
verifier's own hashes count: 839,115 calls by the planner's count, which is no proved ceiling: it
plans every opening at its largest and leaves out some hashing outside the rounds; in a checking run
of a stored full-size proof on the released `cp_verify` the verifier counted 794,273, fewer than
planned; hashing of public files and tables, and words that a draw skips, are planned nowhere, so no
rule keeps a run's count at or under the plan; on the stored sets it stays under, because the plan
counts every opened path whole and paths share vertices. N/2^126 is a looser summary of that bound,
so dividing by hand gives a slightly weaker number. The program's printout names one key file's, one
leaf file's and one request's proof, that is one call, as "the unit of all figures above". The same
printed line goes on: "counted jointly for a full batch of N requests, each round of a request's own
chains reads log2(N) bits less (6 bits at 64)". By the planner's own tables the least round of a
request's own chains is the smallness test at 128.00 bits; counted jointly for a full batch of 64 it
reads 122.00; it is not known whether that joint count is the one that applies to a single request.
That a round which the requests of a batch share may be charged with one run's error, with no factor
for the number of requests in the batch, is argued in the report alone (row SO-1.4). The two values
for a cheater who computes no hashes apply to one call and are weaker for a batch: 2^-102.32 with
the proof of work counted, 2^-70.55 without crediting it, at the planner's count, for 64 whole
calls, calculated from the planning call's prints; printed by no program. The numbers are the
programs' own: `cp_verify -plan accounting` puts the round errors and hash counts that it prints
into a bound of the same form as the technical report's, with one larger constant. They are not
figures of the report. The bound assumes that [the results it rests on](#what-the-theorem-assumes)
hold and that the hash behaves as a random oracle, against classical adversaries. Both columns rest
on the same footing for the [closeness rounds](glossary.md#closeness-round), the theorem on codes,
taken as given, and in both the binding of the weights rests on that theorem too; the right-hand
column only leaves the position rounds' work uncounted. No number in the table is a floor, not even
in the right-hand column. Each still rests on [everything else that the bounds rest
on](#what-the-theorem-assumes), and the bound weakens with every hash that a cheater computes. A
[round](glossary.md#round)'s bits are not the chance of a forgery, which grows with every hash
computed, the verifier's own included.

By the same accounting there is no margin: with the work counted, 68 rounds meet the target by under
0.01 bit, and 71 rounds hold only because their work is counted.

**What to do.** Quote a row's two numbers together, and always with [what they rest
on](#what-the-theorem-assumes).

**[Basis](#how-a-thing-is-known):** reasoned; reported (the planning call that printed the figures
at the default setting: it checks no proof). Rows: SO-1, SO-1.1, SO-1.2, SO-1.4, SO-2.2, SO-3,
SO-3.1, SO-32.2. What the bound is for, a batch (weaker), a cheater who computes nothing, and that
no configuration of the programs is known to meet all the hypotheses of any of the report's
statements: rows LIM-7.1 to LIM-7.3.

<a id="an-honest-proof-can-be-rejected"></a>
#### An honest proof can be rejected

`cp_verify` can reject a valid parameter file, and a rare value of the challenges can make a call on
an honest proof end in `NO` (at one of the nine such events that [the format's
page](format/FORMAT.md) lists) or in `NOT CHECKED` (at the others). `NOT CHECKED` never means
accepted, but a malicious provider may be able to turn a `NO` into it. The service returns "failed"
for a request that it does not prove, and that answer proves nothing about the reply.

**What to do.** Treat a rejection or a `NOT CHECKED` as not accepted. Every batch uses up part of
the commitment, so ask for a new proof once, not in a loop. Do not read "failed" as a statement
about the reply.

**[Basis](#how-a-thing-is-known):** read. Rows: SO-9, C2, SO-54.

### If you review or build on the code

<a id="false-claims-in-test-programs"></a>
#### The library under the programs can be misused

**In short.** A program that calls the `blocks/` library directly can be made to accept a false
claim. Test programs have got false claims accepted that way. We believe that `cp_verify` is not
exposed, but that belief comes from reading its source, not from a test.

By one reading of the code, the library's guards close the known cases for ordinary code inside one
process, but not across processes. Another reading, which counts the cases otherwise, holds that
some are closed only in part: of 22 ways (by a reading of other sources; nothing was run for it) 15
are closed, 2 only in part, 4 only where one running program performs all the tests of a weight
table, and 1 is open; a stricter count, which takes two of the 15 that lean on the caller and the 4
as closed in part, gives 13, 8 and 1. Two further ways for ordinary code stand outside the 22; they
are closed inside one process only. "Closed" is said only of source code that breaks no rule of the
language, defines none of the tests' macros and calls the library's own entry points. The two
readings do not agree; which of them holds is not known; and there may be other cases. The guards do
not stop code that keeps the language's rules but breaks the library's stated contract
(blocks/chain/closed.h; among its clauses: naming no private member of the library; declaring
nothing at global scope before its headers; keeping operator new). At least one such program works:
it gets a false claim accepted at the full target, at toy size ([SO-5.1](limits-further.md#so-5.1)).

**What to do.** Check proofs with `cp_verify`. Do not build your own verifier on `blocks/` unless
you keep to the contract in `blocks/chain/closed.h`.

**Worst case:** If one of these cases did reach `cp_verify`, a provider could get a `YES` for a
reply that its committed model does not give.

**[Basis](#how-a-thing-is-known):** reported; read. Rows: SO-5, SO-5.1, T7.1.11; the cases by
family, and what closes each: row RES-51. Only the known cases are closed in these sources, one of
them in part only (one that, by a comment that stands in other sources and not in these, reads 8
columns where about 89 million are required), and not the family: where a round's error was computed
from a group, its codes or a table, prover and verifier compare the run's numbers with the plan's,
and the process ends on a difference (`blocks/chain/rounds.h`); not covered by those checks: a
number that a round takes from another stage, and the list sizes.

<a id="which-verifier-has-a-theorem"></a><a id="compiled-programs"></a>
#### No theorem is about `cp_commit`, `cp_prove` or `cp_verify`

**In short.** The programs are known to differ from the protocol that the theorems describe. It has
not been shown that the differences do not matter. No statement takes all of them together. So a
`YES` may mean less than the theorems say. The technical report's theorems are about the protocol as
that report describes it, and the Lean theorems are about [the protocol as Lean describes
it](#what-the-lean-proofs-cover). No written statement covers a whole run of the programs.

No configuration of the programs is known to meet all hypotheses of the technical report's theorems.
`cp_verify` has not been checked independently. A review is the one fresh reading of large parts of
it on record (what was tried on `cp_verify`: row E28). Some gaps are known. The report's comparison
lists four places where its statements on soundness do not cover what the programs do, or cover it
only in part. No row was compared under a parameter file that `cp_verify` admits. The cases of rows
PS-6 and PS-19 were not looked for at files that `cp_verify` admits; a provider can write files
where some occur, some of which `cp_verify` accepts (row PS-50.1).

Other points that by the report's own account are not shown, or are shown only in part or under an
assumption, also bear on whether an accepted reply is right. One is the step from the conditions
that a proof establishes to the model's reply. The other is the binding of the compiled argument,
that is, of the protocol once hashing draws its challenges (rows PS-17, on the public parameters,
covered in part, and PS-7, on a request's digest, covered under an assumption, and not for the
released programs; both marks are the report's own and speak of its protocol on paper: row PS-67).
The step to the model's reply has not been shown for the programs under any policy, exact match
included. Where it fails, the verifier can accept a reply that is not the model's, or an honest
prover can fail. One link of that step, which rewrites a weight claim over a tensor's indices onto
the committed matrix's rows and columns, has no written proof (rows SO-17.14, PS-69).

Every `YES` also depends on two parts of `cp_verify` that rest, wholly or in part, on a reading of
the source (rows SO-17.3.1 and SO-17.7). The prover and the verifier are also built on one library,
`blocks/`. So comparing the two cannot reveal a fault in the shared code.

The comparison of the programs with the report was made by a reading of other sources, and it did
not cover all of the code. In particular, it did not compare the arithmetic of each check inside the
proximity tests and the folds with the report's. It says nothing of three places at which the
released programs leave the protocol that the report's statements are about: what a proof's first
hash takes in, the labels of two of a proof's chains, and how the commitment ID is formed. Where the
report tells of code that differs from the compared sources, that code was read by difference alone,
for minutes each (source files that the compared sources lack, in full); two files of GPU code that
computes values could not be compared by difference. Except for a comparison of numbers by planning
calls, nothing was compared at the default parameter file (rows PS-37 to PS-39, PS-46, PS-47,
PS-62).

**How far the differences that are not, or not fully, covered were attacked.** Some readings tried
to refute an account of the programs; none tried to attack the protocol at these differences. That
none describes an attack is not a finding of safety, and at some differences nothing was tried at
all (row PS-51). `cp_verify` also admits forms of a parameter file that no row of the differences
(rows PS-1 to PS-22 and PS-33) tells: two are known, from a list that is not complete (row PS-52).
Row PS-50 says where each difference that is not, or not fully, covered arises: some arise in every
proof that `cp_verify` can accept. The differences that are not, or not fully, covered have not been
ranked by risk.

**What to do.** Treat the theorems as evidence about the design, not about the programs ([what is
proved](../guide/what-is-proved.md)).

**[Basis](#how-a-thing-is-known):** read; argued on paper. Rows: E1, E28, PS-2, PS-3, PS-4, PS-5,
PS-6, PS-7, PS-17, PS-17.1, PS-19, PS-37, PS-38, PS-39, PS-46, PS-47, PS-50, PS-50.1, PS-51, PS-52,
PS-62, PS-67, PS-69, SO-7, SO-9.1, SO-9.3, SO-11, SO-12, SO-17.3.1, SO-17.7, SO-17.14, SO-60,
SO-60.1. Each gap in full, four more rows (PS-2 to PS-5) that are covered only in part, the
no-overflow conditions, the sub-proofs' second implementation, of which the library's tests say
nothing, and four tests that have no round and no error term of their own (named in [the format's
page](format/FORMAT.md), 13.6, which does not say whether some round's error covers them): rows
LIM-8.1 to LIM-8.3 and RES-62. One of these gaps touches weight hiding too (rows PS-19 and LIM-8.2).
The layout files in this repository leave places that belong to no matrix, in every oracle message
that they place. No claim reads these places, and the report has no statement for that form. By a
reading of the prover's source it writes zeros there, the pad's own entries aside; nothing was run
to show it. What the zeros mean for hiding has not been worked out.

<a id="what-the-theorem-assumes"></a>
#### What the bounds rest on depends on the parameter file

**In short.** The printed bounds rest on all five of the things below, and none of them is proved
for the programs. If one fails, an accepted proof is weaker evidence than the printed bits say, and
in some cases no bound is left.

The truth of an accepted request and the binding of the weights both rest on the theorem on codes,
and, at the default setting, both lean on counted proof of work in the rounds that draw positions.
We take that theorem as given. What the Lean library proves in the theorem's place, what it does
not, and what has been checked of that: [what was checked, and what was
not](#what-was-checked-of-the-theorem); `lean/LIMITS.md`, L15. The bound rests besides on further
steps argued on paper in our report alone; that our parameters meet the theorem's conditions is our
own reckoning, with no bits to spare. The provider's parameter file also picks what a request's
bounds rest on, and neither the last line nor `result-v1` shows that.

1. **Cited soundness.** Each proximity test has three steps: fold, code switching and a last step.
   Their round-by-round soundness is cited from the paper that this protocol builds on
   ([PUBLISHED-WORKS.md](../PUBLISHED-WORKS.md#the-paper-that-this-protocol-builds-on) names it).
   The technical report restates it and does not prove it. Every round's error rests on it, in the
   form that the report cites, at every setting. This project has not checked its proofs. Of the
   differences from the paper that are known (PUBLISHED-WORKS.md lists some, and marks where one is
   not told; more: row SO-2.1.1; unknown ones may exist too), one: in the fold, rows are kept
   unfolded for a later part of the protocol ([kept rows](glossary.md#kept-rows)). Whether the cited
   analysis carries over to these differences has not been shown. A reader cannot look up all the
   parts of the cited analysis that these pages lean on, and no analysis that a reader can look up,
   that we know of, covers the steps that are cited from it, for this protocol. Which weights a key
   binds also rests on the theorem on codes, taken as given: the steps that test closeness work with
   lists of candidates.
2. **A source for each closeness round.** By the programs' own planned accounting the tightest
   rounds sit exactly at the target; any loss in the cited bound, however small, puts them below it,
   in a proof and in what the committer makes once. The theorem and its sources, in full: row LIM-9.
   The counts of rounds: row LIM-10. The bound within a third of the distance, on which no round
   rests at the default setting: row LIM-11. What a provider's own file can still pick, unseen in
   the last line: rows LIM-13.1 and LIM-13.2.
3. **Counted proof of work,** at the default setting. For the programs' hash chains this is an
   assumption ([details](#how-the-programs-count)). How much work is counted, and where: row LIM-12.
4. **The hash, taken as a random oracle.** Soundness rests on BLAKE2b, taken as a random oracle. A
   second, fixed hash function, built on BLAKE2b's compression function, also appears in the
   programs.
   - The second hash function digests the request file and public data (three lists of lines of the
     parameter file, the scale file's words, and four tables computed from lines of the parameter
     file).
   - The request file's own words also stand in the first BLAKE2b query of each proof's first chain
     beside that digest, so, for a prover under a commitment the client already holds, nothing it
     chooses reaches a challenge through the second hash function alone.
   - The public files are bound through the commitment ID the client holds (its string carries their
     BLAKE2b digest and the verifier recomputes the ID from the files it is given), so no prover can
     change them under a commitment the client already holds; `cp_verify` reads them from its
     caller, not from the prover. The theorems are stated for public files fixed before the hash
     function is drawn; they cover each set of public files one at a time, not a committer that
     searches for files with the hash in hand. Against a prover, the client holds the commitment ID,
     which by design binds those files, before it sees any proof.
   - For a prover, then, the second hash function is used as the fixed, public function that it is,
     not as a second random oracle; that it does not disturb BLAKE2b's standing as a random oracle
     is a weak assumption about the pair.
   - One thing hangs on the second hash function alone: which four tables, bit for bit, a parameter
     file names, and so a commitment, since only their digests under the second hash function stand
     in the file. A verifier's answer is about the tables that it computes itself and holds against
     those lines. Two verifiers, or a definition that takes the tables as data, speak of one
     function if their tables are the same; that both tables fit the lines shows this only as far as
     nobody can find two tables with one digest under the second hash function (collision resistance
     of the second hash function: an assumption of such a statement).
   - *How this is known.* By a reading; there is no formal proof. The technical report models both
     hash functions together as one random oracle, for which there is no argument: the report's
     bound is stated under the two-functions-as-one-oracle assumption.
     ([blocks/hash/MODE.md](../blocks/hash/MODE.md) says which inputs the second hash function
     hashes; [hash mode](glossary.md#hash-mode) says what is known of it.) From a reading of the
     code, the second hash function's tree hash (TH in [verifier/HASHES.md](../verifier/HASHES.md)),
     which makes two of the nine digests, the request file's and the scale file's, takes only the
     list that it hashes, with no number or word that names the list's use, so one list hashed for
     two purposes gives one digest. No exploit is known, and no search for one is known (row
     SO-14.1).
   - The layout of BLAKE2b's labelled strings has not been reviewed;
     [blocks/hash/MODE.md](../blocks/hash/MODE.md) lists six points that it does not decide.
   - *What the model leaves out.* The model counts each hash call as one query, so it says nothing
     about collisions inside a long input. Collision resistance alone would not give the bounds. The
     analysis treats attackers as classical, and nothing is proved about BLAKE2b. Whether two
     published attacks on hash-compiled proof systems apply to these programs has not been examined
     (row SO-57).
5. **The proofs in the technical report.** The printout names two bounds that the report proves,
   after public works (preprints among them). These proofs have been read in portions only, never as
   a whole, and never in the wording in which they stand. The correction of one fault, in a
   hypothesis of a theorem on the code switching, has had single readings only. Faults that we know
   of may stand in these proofs uncorrected, and there may be more that we do not know of. The
   report's proof of soundness does not cover three things that the programs do (what a proof's
   first hash takes in; the labels of two of a proof's chains; how the commitment ID is formed). The
   bounds are cited from published work, some of it not peer-reviewed, or argued on paper in the
   technical report, which is not refereed; they are not proved for these programs. The program's
   own printed accounting says so too. Of up to two steps it prints "a bound argued on paper for
   this protocol": argued in the technical report, not machine-checked for these programs. Which
   public works, and how they and the proofs were read: row LIM-15. How far the report's proofs were
   read (in portions, each once, in minutes), and the dozen or so statements found false that other
   readings passed: row PS-35.

<a id="what-was-checked-of-the-theorem"></a>**What has been checked of the theorem on codes, and
what has not.** In full, with the counts of files, declarations and runs: row LIM-9.

- *Which theorem.* That theorem is one of Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, whose
  paper is published at STOC 2026; the theorem our programs use, Theorem 4.6 of its full version
  (ePrint 2025/2055), is not in the proceedings, which state the plain form up to the Johnson radius
  (Theorem 1.5); for its proof the full version gives an outline of one paragraph, and credits a
  proof of the property, up to the Johnson bound, to a note that it cites as a personal
  communication, which is public as ePrint 2025/2110 (the same note, by its citation; texts not
  compared).
- *Taken as given.* The report and the programs' accounting take the theorem as given. A statement
  on codes held to follow from that theorem, in its case of two words, is proved in this
  repository's Lean library, by way of another formalization's theorem and a comparison of bounds;
  the paper's own form of the bound is not proved there, and that the library's statement follows
  from the paper's, and that these programs' parameters meet the theorem's conditions, rest on
  reading, not on a proof (lean/LIMITS.md, L15). No proof of the paper's own form of the bound has
  been checked. ([lean/THEOREM.md](../lean/THEOREM.md), "What it rests on".) We have not compared
  the paper's theorem with later work; a correction to it may exist that we do not know of.
- *In a public Lean library.* A public Lean library, ArkLib (Apache-2.0), has a theorem of this
  kind: mutual correlated agreement of Reed–Solomon codes on affine lines, for every radius below
  the Johnson radius (`ReedSolomon.mcaError_affineLine_johnson_le`: proved there, by our reading of
  its source, through the package VCV-io, which we did not build; not among what we checked); the
  paper's own form of the bound is not proved there.
- *The files adapted for this repository's Lean library.* The Lean library proves, from files
  adapted from ArkLib and Mathlib (`lean/ThirdParty/`; `lean/Cb/Bridge/`), the statement that it
  puts in the theorem's place, which by our reading restates the theorem for two words, but not the
  paper's own form of the bound. The adapted files were compared with their sources by programs,
  without following the mathematics line by line. The statements of the compilation theorem
  (`innerCompiledVerifier_sound`) at the two sizes take that statement as a hypothesis. The
  compilation theorem itself does not apply the proof; second forms at the two sizes and the bounds
  on the entry file's closed check do. The soundness theorem for the Lean verifier rests on theorems
  on Reed–Solomon codes, above all mutual correlated agreement below the Johnson radius. As typed in
  this repository they are proved in this library, not assumed. That says nothing on whether a
  published paper's theorem is formally verified. That the typed statement follows from the
  published theorem, and that the statement which Lean has accepted is the bound that the written
  proof uses, at our parameters, rest on readings alone. The entry file is what `docs/` calls the
  leaf file.
- *Our own use of it.* Our use prices many interleaved rows as a line through two words by a step
  that our report proves after a statement in the abstract of S. Jo (ePrint 2026/891; we have not
  looked for a refereed version), at the theorem's M = 1, and the steps from the theorem are our
  own. That our parameters meet the theorem's conditions is our own reckoning; and the rounds that
  rest on it hold at the target with no bits to spare. What is said here of the report's steps
  cannot be checked from this repository.

**Every commitment has published [out-of-domain (OOD) answers](glossary.md#record) (masked sums of
the weights)**, so the binding of the weights, and not only the request's truth, rests on the
theorem on codes, with no bits to spare, and, where the file counts it, on proof of work (binding in
a modified sense, in which the hash itself supplies the OOD seeds; that sense does not imply the
binding definition that the report cites). The written argument for that sense covers a commitment
ID formed without those seeds, not the ID that the programs form, which hashes them. So of the
programs' own commitment no binding theorem is proved. What Lean proves of the Lean verifier is that
any commitment, with its committer's hash queries, settles at the given public files one set of
weights or none before any proof (index binding). Of an illustrative committer written in Lean,
which models one part of `cp_commit` (no theorem ties the two) and was compared with it in values at
small size only, Lean proves, at public files where `verify`'s tests on them pass, that the weights
so settled are the ones put in, at an error of about 2^-89 at that committer's own query count, and
that the commitment cannot be opened to other weights, at about (t + 2^23)/2^124 after t hash
queries ([What is proved](../guide/what-is-proved.md); [lean/THEOREM.md](../lean/THEOREM.md), "What
the commitment gives, as proved"). What Lean states, and what its soundness theorem for the Lean
verifier does not say: [what the Lean proofs cover](#what-the-lean-proofs-cover). The `commitment:`
line and `record=1` in `result-v1` say on every accepted call that OOD answers are published; read
them as the commitment's footing ([verifying.md, point 7](verifying.md#when-to-accept)). Where
counted proof of work comes in, in which (modified) sense the weights are bound, what Lean and the
report state, what was compared and what has run: rows LIM-14.1 to LIM-14.4.

**Parameter files that no test covers.** `cp_verify` also accepts such files:

- Files in which a proximity test folds its message down to a single entry.
- Files with sizes outside the ranges that `cp_commit` and `cp_prove` can handle. No honest files
  exist at those sizes, so neither an altered file nor a malicious prover has been tried. No one has
  traced whether a `YES` can be reached there (rows SO-6.3 and SO-6.3.1).

The rest, and two forms of a file for which the comparison with the report has no row: row LIM-16.

**The cost exponent 7.** The C++ verifier's test of the parameter file admits the cost exponent 7 as
well as 8 (a token's cost is its logit gap + 2^8); the Lean verifier's scope, the theorem and every
written argument cover 8 only; all shipped files use 8, and the Lean verifier's command reports
exact match only at 8. So under exact match the C++ verifier also admits a scored-head offset of
2^7, a setting at which neither the Lean model (which fixes 2^8) nor any written argument applies;
the Lean verifier refuses it.

**How far to rely on a `YES`.** The provider's parameter file fixes which results a `YES` rests on.
So at every setting, the default included, rely on a `YES` only as far as you take the theorem on
codes as given, accept counted proof of work, and take the further steps and our own reckoning that
this section names as they stand; under a parameter file other than the default setting's a `YES`
rests on that file's own numbers besides, and is not to be relied on by itself ([which setting is
it?](verifying.md#which-setting)).

**What to do.** Do not quote a bound without these conditions. For the parameter file you were
given, run `cp_verify -plan accounting <config folder> 128` and read the `footing:` lines. Each
says, for one file, how many of its rounds rest "on the theorem on codes (ePrint 2025/2055, Theorem
4.6), taken as given". If a count is not 0, an acceptance under that file depends on that theorem.
[verifying.md](verifying.md#which-setting) shows how to tell whether the file has the default
setting.

**Worst case:** Every setting also rests on taking the hash as a random oracle and on the
round-by-round soundness of the proximity test's steps, which the report cites and does not prove
and whose proofs this project has not checked. If either of those failed, no argument would be left
at any setting, and the proofs and commitments already handed to clients could not be called back.
The report's proofs of composition and compilation have not been checked independently. If one of
the report's proofs of composition and compilation fails, the rounds' own errors may stand, but
nothing ties them into a bound on the chance that a false reply is accepted, at any setting. We know
of no forgery. (What did pass: row SO-15.7.)

**[Basis](#how-a-thing-is-known):** argued on paper ([what is proved](../guide/what-is-proved.md));
read; reported (what Lean accepted, and its re-checker's replays). Rows: SO-2, SO-2.1, SO-2.2,
SO-2.7, SO-6.3, SO-6.3.1, SO-6.3.2, PS-52, SO-38, SO-38.1, SO-56, SO-57, SO-57.1. What fails with
the theorem on codes, with one of our own steps or with the counting of work, and the fall-back that
the report names: rows LIM-17.1 to LIM-17.3.

<a id="how-the-programs-count"></a>
#### The programs' calculation of their error bounds rests on steps that are assumed or not checked

**In short.** A provider's parameter file can pass every check in `cp_verify` and yet give less
soundness than the printed figures say.

The provider chooses the public parameters. A file can pass every check and yet fail a hypothesis of
the technical report's soundness theorem, and a failed hypothesis can break soundness. The report
specifies where weights, cells and tables go, but it never lists the remaining public parameters for
a complete configuration. So no one has shown that any configuration meets the theorem's hypotheses,
and the report does not claim that one exists.

The calculation itself also assumes, or does not check, these things:

- Counting proof of work rests on one step: that g bits of work may be counted as a factor of 2^-g
  in a round's error. The work makes each retry more expensive for a malicious prover. It does not
  lower the chance that a single challenge lets a false claim pass. For the programs' own hash
  chains the step is assumed, as the program itself prints. No theorem ties the chains of the
  technical report or of Lean to them (what is known of how Lean's hashing differs from the
  programs': row PS-68; `lean/GAPS.md`, 22; `lean/LIMITS.md`, L21). At the default setting only the
  rounds that draw positions have proof of work: up to 32 of such a round's 128 bits rest on this
  step, and at every size in this repository these rounds miss the target without it (rows SO-32 and
  SO-59). The rest are meant to bound the chance that one cheating attempt passes the round, by the
  programs' own calculation and under results that are cited or assumed. One attempt at the work can
  also cost less than one hash call, so g counted bits mean fewer than 2^g calls. By how much has
  not been counted for this layout. (A round's 128 bits: a per-round target the program computes,
  not a security level.)
- At full size, hundreds of rounds get their error, in part or wholly, from a number that the
  parameter file or the program's own code supplies. Some of the numbers are checked later, but
  nothing checks at that round that the round runs with that number. Where a later comparison
  exists, it often takes the number from the same source, so it checks the form, not the value. The
  count of such rounds covers only plans that the library's own planners make. The printed numbers
  say nothing about how the rounds are put together: it has not been checked that the degrees and
  counts passed to the library are those that the inner steps of a segment run; the formula was
  compared, what is put into it was not (row SO-4).
- At the default setting's full-size file some rounds would miss the target of 2^-128 under a
  stricter count than the written arguments need. The range check's round (`smallness` in the
  printout) counts one column of the check's matrix, not all columns. In the round that the printout
  calls LINK, the calculation takes the larger of two errors, not their sum. The figures under the
  stricter counts (by a planning call 113.44 bits for the range check's round; LINK as a sum may
  read below 128 bits at the small example configuration), and that the proof that the smaller
  errors are enough exists on paper only, for the report's protocol, and has not been checked
  independently: row LIM-18.2. A second kind of round that a stricter count would put below the
  target (the 25 rounds in which a switch with kept members draws its positions, near 122.5 bits
  with their work counted and a factor for a list of 54): row SO-2.3.1. The range check's own 25
  rounds too are counted with no factor for a list of more than one codeword (by the report, each
  sent message's out-of-domain sample singles out one codeword before any later challenge); with a
  factor for a list of 54 they would stand near 122.2 bits ([the paper](../paper/paper.pdf), Table
  15). The count for a whole batch: [above](#the-soundness-figure). None of the figures here is a
  security level.

A unit test guards against each of four faults in this calculation that would make the program print
more bits than there are (row SO-4.3).

**What to do.** Read the line `proof of work:`, which says how much work is counted, in how many
rounds, how many bits the weakest round has with that work counted, and how many the weakest has
without it. To see whether a parameter file has the default setting, compare the SHA-256 of its
first 32 lines with [the list in verifying.md](verifying.md#which-setting). That list covers the
sizes in this repository only, and says nothing about the later lines of the file.

**[Basis](#how-a-thing-is-known):** reported (the planning call that printed the figures at the
default setting: it checks no proof); reasoned; argued on paper. Rows: SO-1.3, SO-2.3.1, SO-4,
SO-4.3, SO-6.4, SO-6.6, SO-6.7, SO-10, SO-32, SO-59. Each point in full: rows LIM-18.1 and LIM-18.2.

## Weight hiding

<a id="the-verdict"></a>
### Where it stands

Masking is built in and on by default. By design, every published value that depends on the weights
is masked or sits under a salted hash, apart from a [declared list](what-is-not-hidden.md). The aim:
clients learn nothing about the weights beyond that list. The short account is
[guide/weight-hiding.md](../guide/weight-hiding.md); this section is the long form.

*In short.* Zero knowledge, Lean's form of weight hiding, is proved in Lean for the protocol, at
full size: honest-verifier zero knowledge for the interactive protocol (IOP), and, [after
hashing](glossary.md#fiat-shamir), in the random-oracle model, zero knowledge of the compiled
(non-interactive) committer and prover, with the Lean verifier that is proved sound, against any
verifier within 2^64 hash queries and 688,127 replies, at a statistical distance below 2^-154
(bounds: below). The honest prover in these theorems is a Lean definition, not the CUDA prover. One
bit per request is revealed (served or refused: a request is refused if it is ill-formed, if the
model has no result on it, or if its result is other tokens), and it depends on the weights. A
service that chooses to say "failed" reveals one label of three in its place (served, failed or
other tokens). The theorem is at the exact-match policy, for one commitment with one [leaf
table](glossary.md#leaf-table) and one request to a proof: batches, which are what the programs
make, are not covered by this result. It is with all the committer's and the prover's coins, their
masks among them, fresh and uniform; the programs derive theirs from a secret seed, which no Lean
theorem covers. Nothing is proved for the programs. The protocol has a written proof that is not
settled, and it is of the protocol on paper, not as the programs run it. For the programs there are
tests, readings of the code and a judgement (below). Main risks: a mask lost to a bug or a hardware
fault; a use counter set back by restoring or copying the secret key directory.

The default setting is the setting of the release. Every commitment publishes each committed table's
OOD answers. The committed tables are the weights, encoded, under the commitment. "Before hashing"
means the interactive protocol; "after hashing", the argument made from it by Merkle trees and the
Fiat-Shamir transformation, in the random-oracle model.

*Before hashing: the interactive protocol (IOP).* The protocol's three honest provers ([key
file](glossary.md#key-file), [link proof](glossary.md#link), request) are Lean definitions, and Lean
proves honest-verifier zero knowledge for them, at full size. The simulators are chosen first, and
they never get the weights, a table's randomness or a prover's coins. What an honest verifier sees
of each run has exactly the distribution that the simulator makes from the public files, the
verifier's messages, the out-of-domain samples, the columns of the tables that the verifier reads
and, for a request, the request with its claimed output and the copy's number. The one exception is
where the prover stops for a private reason: the chance of that is 0 for the key file's proof, 0 for
the link proof and at most 2^-174 for a served request. Over the whole life of one commitment, up to
710,324 copies (a copy is a further committed table of the same weights, with masks of its own: a
[leaf table](glossary.md#leaf-table)) with 688,127 requests each, what an unbounded observer sees of
the honest provers on the real weights is within 2^-135 (710,324 × 688,127 × 2^-174, by the
statement's formula) of what it sees of the simulators working from tables of any other fixed
weights. That life is a budget that the statement fixes: one run for the key file, one link for each
copy, requests on the copies alone, asked in any order and adaptively; before hashing no committer
is part of it. Its conditions are three, all on the public files: they are in scope, they are read,
and [the tests on the public files](glossary.md#tests-on-the-public-files) pass. At this level one
bit per request is not hidden: whether the request is served, which depends on the weights; the
simulators are given that bit, and where a request is not served the prover shows nothing. After
hashing it is one bit too: served or refused; in a corollary, one label of three (below).

*After hashing: the compiled (non-interactive) committer and prover, with the same Lean verifier.*
The commitment is the 256-bit commitment ID, not the whole public key, and the compiled system's
verifier is `verify`. In the random-oracle model Lean proves zero knowledge against any verifier,
with no bound on its running time. ([Index privacy](glossary.md#index-privacy) is the cited
definition's form of zero knowledge.) The verifier gets one commitment and then, in any order and
adaptively, asks up to 2^64 hash queries and up to 688,127 replies, each at a request, well-formed
or not, that is picked from all that it has seen (well formed: every token in the vocabulary, prompt
and output not empty, at most 1,025 tokens together). There is one simulator, fixed before the
weights, which may depend on the public files. It gets each request, the largest number of replies
and, of the weights, one bit for each request: served or refused (in the corollary, one label for
each request: served, failed or other tokens). It makes the commitment and the replies, and it
answers the hash queries in the hash function's place, never one query in two ways. What the
verifier sees with the honest committer and prover (the commitment, the replies, the hash answers,
its own coins) is within 2^-157 + 688,127 · 2^-174, which is below 2^-154, in total variation, of
what the simulator makes. A nonce search that runs out is inside what is simulated: it shows as the
empty proof (a reply that holds no proof) at a served request, which looks like the reply to a
refused request; in the corollary, as the empty proof beside the label "served".

*The bounds.* After hashing: one commitment with one copy of the table, one request to a proof,
exact match, and public files that are in scope, are read and pass [the tests on the public
files](glossary.md#tests-on-the-public-files); the library's own copy of the stored full-size files
is such a set. The bit depends on the weights, and it is declared, not hidden: for every request of
its choice a verifier learns whether the claimed output is the model's result. (It reads the bit off
the reply: a refused request gets an empty proof; so, with small probability, does a served one,
where a search for a nonce runs out.) So over many requests, chosen adaptively, it learns what the
model outputs on inputs of its choice: whoever tries each candidate token for a prompt learns the
model's next one from this bit alone, unless all are refused. Against three labels it hides one
thing: whether a refusal means that the model has no result or that its result is other tokens. It
hides that only from a verifier that asks few requests. A request's window (the 1,024 tokens that
the model's run reads: the prompt, the claimed output, then zeros) can itself be sent as a prompt,
with one more token as the claimed output, which the run does not read: a verifier that tries every
candidate for that token and is refused each time learns that the model has no result on that
window, and one that is served once learns that it has one (argued from the definitions; no theorem
states it). Each such search costs up to 202,048 of the 688,127 replies. A true output is refused if
the model has no value further on in the window. No statement bounds what many bits reveal together.
A request is served if it is well formed, the model has a result on it and the claimed output is
that result. Every other request is refused: one that is ill-formed, one on which the model has no
result, one whose claimed output is not that result: one value for all three. In the corollary the
label is "failed" where the model has no result on a well-formed request, and "other tokens" where
its result is not the claimed output. In the corollary "failed" can come from the padding behind the
output, so it tells more than what the model answers. Nothing in the prover counts replies, and
beyond these numbers the theorem says nothing. It says nothing of what the public files themselves
tell of the weights, of timing, of batches or of a second commitment. The simulators' running time
is not modelled. The theorems assume that the verifier makes no hash query before the commitment,
and that the weights and the public files are fixed before the hash function is drawn. A second,
fixed hash function inside the format is not treated as random. The numbers are statistical
distances, not a security level. The definitions of zero knowledge at both levels are this library's
own; that they say what these words say is a reading, not a theorem. A statement of this form is met
by a prover that hands out nothing, so it counts only beside completeness: in the same Lean value
stand soundness, binding and completeness of the same committer, prover and verifier. All of it is
about a Lean definition of the prover, not the CUDA prover, default values included, with all the
committer's and the prover's coins, their masks among them, fresh and uniform; the programs derive
theirs from a secret seed. It says nothing of the C++ and CUDA programs: for these nothing is
proved.

*The tables, before hashing.* At the default setting Lean proves that what is shown of the committed
tables, within a budget of opened columns, has the same distribution whatever the weights (Lean's
encoder, fresh [pads](glossary.md#pad), full size, before any hashing).

*What no theorem covers.* No Lean theorem covers the programs, or the OOD answers as the programs
make them ([what Lean proves](#what-lean-proves)).

*The written proof.* The report's protocol, with OOD answers and fresh masks, has a written,
unsettled proof of zero knowledge against any verifier, batches included, under its own bound and
conditions (before hashing, its statement for the interactive protocol is for an honest verifier,
within a query bound, under the report's theorems' conditions on the parameters). Under conditions
it also reaches rejected requests, [clipped gap](glossary.md#clipped-gap) (a policy that also
accepts tokens that score close to the top) and several commitments. It stands in the technical
report, which is not public and not refereed. Its authors call the proof "complete but not settled":
it has been checked too little to rely on it ([on paper](#on-paper)). These points, among others,
are open between what Lean proves for the protocol, the written proof and the protocol as run: the
OOD answers; derived masks; clipped gap; a rejected request and a prover that stops; a further
commitment and the counter's limit; the number of random values in a kept row; differences between
the written protocol and the programs'. Only one, the OOD answers, was attacked on purpose, on
paper, and it was not broken (in two written arguments other than the report's, one of them read
only in part: row HW-766). With the protocol's conditions (each mask random, secret and used once;
each encoding opened within its limit; secret seeds independent) kept, no counterexample is known at
any of them; at the others that means only that none was reported. With masks derived from a secret
seed, as the programs' are, it covers these answers only under four conditions that have not been
shown of the programs. As the programs run it, the protocol misses at least four of that proof's
hypotheses (a count apart from those four conditions), one on derived masks, two on a rejected
request and a prover that stops, one on batches that overlap in time. `cp_prove` runs no verifier;
the provider's check (duty 6) withholds the whole batch at a rejection, so the two together are on
paper such a prover (inferred: row HW-740); `cp_prove` also stops at tests with no stated bound, so
for it the report states no bound ([more](#a-prover-that-stops)).

<a id="narrower-than-what-runs"></a>*Where Lean's results are narrower than what runs.* They are
about the protocol as Lean describes it, not about the programs: as is usual, no theorem ties them
to a program. Whether Lean's description is, to the letter, the technical report's protocol has not
been shown. The programs derive their random values from a secret seed, so their hiding is
computational at best, which is usual. Their prover keeps a counter, while the cited definition is
stated for a prover that keeps no state. The programs count batches, and stop an encoding at 688,127
at full size. Behind the programs' counter no Lean theorem stands: nothing in Lean keeps count of
the columns that a table has opened ([what Lean proves](#what-lean-proves)). A [kept
row](glossary.md#kept-rows) has 219 random values for 219 positions shown, with no slack. No result
in Lean says anything of that number: it is argued on paper only (that number against the positions
shown, and on what it rests: rows HW-762, RES-58). In Lean's results on hiding "the weights" are the
weight matrices (one matrix in which they lie). The three public files stand outside: the scale file
and some lines of the parameter file are fitted to the weights, and what they tell lies outside
every statement on hiding; the sizes are those of one fixed format and model size, and the look-up
tables are recomputed from the public files. All sizes are public. Lean's results on hiding are for
one request to a proof: batches, which are what the programs make, are not covered by these results,
and the cited definition has no batches. They model no size, time, memory or restart, and no proved
statement on hiding covers a prover that stops a batch; at a request that is not served the Lean
prover hands out the empty proof and nothing else; a label beside it is a service's choice. The
programs' service answers "failed" for every request that it does not prove, whatever the cause; a
client can partly tell the causes apart (row HW-532). That answer is neither the bit that the
simulator is told for each request (whether it is served) nor the corollary's label "failed", and no
theorem ties it to what the Lean prover hands out. The theorems do count a stop of the honest prover
at a served request: as a value of the proof; the chance of a stop for a private reason is a term of
the error. They also leave out faults, the order of work and what the proofs of one batch share, so
they say nothing about them either. Whether a request can be served depends on the weights and is
not hidden. A theorem states that requests that can be served exist (`canServe_exists`). The
statement names no weights, and the weights in its proof are all zero, made up for the proof. A
theorem shows valid weights, constructed, not a real model's, at which no request can be served at
the default setting's model parameters (`canServe_nowhere_atHaltingWeights`). These weights do not
pass the first of the Lean library's two worst-case checks
(`not_passesWorstCaseCheck_atHaltingWeights`). At weights that pass the first worst-case check, or
the least-bound check with numbers that hold, the model has a result on every well-formed request
(`modelNeverFails_of_passesSomeWorstCaseCheck`). These are the Lean library's own checks; neither is
`cp_commit`'s worst-case check: no theorem relates either to a check that any program makes. The
weights that are shown by name to pass the first check are made up; no theorem says whether a real
model's weights meet that premise, and nothing in a commitment, in a proof or in what a verifier
does shows that the committed weights meet it ([What is proved](../guide/what-is-proved.md),
"Completeness"; lean/LIMITS.md, L27, L93). By the word of a Python program of this project that
mirrors the Lean checks, Llama 4 Maverick's public weights, at settings fitted by this project, of
which by its word they are valid weights, fail the first worst-case check and pass the least-bound
check. It is not a Lean evaluation, no theorem joins the program to Lean, and the program is not in
this repository; each check was run once, and some margins are thin. That the honest proof is
accepted at these weights is not shown. That these settings are valid settings is not shown either.
(lean/LIMITS.md, L7, has the whole.) Nothing shows that a request that the programs serve is one
that can be served in Lean's sense, and no theorem in the Lean library shows that any request to a
real model is one on which the integer model passes all its range checks on all 1,024 tokens. A
search for such a theorem found none, which is weaker than showing that none exists: every theorem
found that concludes this of a request either uses weights made up for the proof or takes it as a
premise. The search was made over other Lean files, not over this repository's (row PS-63). The
theorems at weights that pass the first worst-case check, or the least-bound check with numbers that
hold, are of neither kind: their premise is a condition on the weights, which no theorem shows of a
real model's weights. Completeness is proved on served requests: the requests that can be served
(`CanServe`) whose claimed output is the model's own (`finalComplete_iff_completeOn`). Zero
knowledge after hashing is stated at every request, well-formed or not
(`zeroKnowledgeAtEveryRequest`); whether a request is served depends on the weights and is revealed,
not hidden: the simulator is told it. The corollary with a label shown is stated at every
well-formed request. Lean's theorem on zero knowledge is for exact match: under a clipped-gap policy
no Lean theorem covers what a whole proof reveals.

*The judgement.* We judge that weight hiding likely holds at the default setting, in honest runs, if
the provider keeps its duties. The judgement is given of the released programs: the first and the
fifth of its five tests have run on them, with the outcomes told under [five tests](#verdict-runs),
both at small size on synthetic weights and the first at full size too, once on synthetic weights
and once on a real model's public weights; the fifth ran once. The second and the third ran on test
builds of other sources, and the fourth on programs of other sources; at full size the fifth's two
audits ran together only on programs of other sources. The one whole reading of the sources was of a
text from which these differ in code in 68 of 384 files, and in 136 more in comments, names and the
wording of messages alone; two more of the 384 are not among these sources; in the 68 the
differences in code were read for hiding, by difference alone, apart from renamed names and reworded
messages, and the files were not read whole again (row HW-745; [on which programs it was
made](#on-which-programs)). That judgement rests on the design, what is proved in Lean for the
protocol, code review and tests; it is not a proof ([the reviews are not independent](#review)). It
has [conditions of its own](#the-verdicts-own-conditions) beside the duties. An honest run: nobody
has changed the provider's programs, and no injected, compiler or hardware fault has struck. A bug
in the sources is not excluded. "Likely holds" is the second of [four grades](#the-grades): with
these conditions kept no way to a leak beyond the declared list is known, and the design's argument
is believed. The programs themselves are not formally verified, as is usual. Of the programs it
rests on [five tests](#verdict-runs) and on readings of the code. The tests ran on synthetic
weights, under exact match, nearly all at small size. That judgement does not cover a real model's
weights: only one of the five tests has run on any (every published sum that it checked was masked
by the rules that the test carries, which does not show that the masks hide); no problem is known.
No other test of hiding that these pages tell of has used such weights (row HW-747 has all that
did). In full: [what was run on these programs, and what was not](#what-was-run-on-these-programs);
run by run: row LIM-63.2. How far the code was read: [the readings](#verdict-reading). A provider's
own default build from these sources comes under the judgement only as a correct build, the first of
its conditions: every build tested here is our own. The judgement reaches a
[clipped-gap](glossary.md#clipped-gap) policy, on less evidence: audits apart, no test of hiding ran
under such a policy. Both audits with the provider's secrets ran under one, at small size, on the
released programs: they show that the masks are there as derived, which is no search for a leak (row
H13.8). Lean's theorem on zero knowledge is for exact match: under a clipped-gap policy no Lean
theorem covers what a whole proof reveals. Within the judgement's reach no attack or leak is known
that goes beyond the list, though recovering weights from published files has barely been tried, and
no loss of a mask has been tried on the released programs, which have no option for injecting a
fault ([risk 1](#a-lost-mask)). Outside that reach hidden values have come out in trials, not on the
released programs (rows H1 and H46.4). That no other kind of leak has been found does not mean that
there is none. [Not covered](#not-covered), among other things: weight files with an 8- or 16-bit
tensor, which the programs take without a warning. The examples' public 4-bit conversion has no such
tensor. Not covered either: any other setting, or `-public-weights`, which hides nothing.

<a id="a-further-commitment"></a>*A further commitment.* A provider that serves for long enough, or
on a second GPU, needs a further commitment to the same weights
([how](operators-duties.md#a-new-commitment)). The judgement takes that in, on thin ground, whether
the commitments follow one another or serve at once on several GPUs. Each comes from its own call of
`cp_commit`, on a machine that was not cloned or restored, and samples its own secret seed. Nothing
is carried over from another commitment, and each commitment stays within its limit. Nothing hidden
may choose which commitment serves a batch. One exception: the commitment may be picked by load or
queue length. That can depend on hidden values, and the pick gives away something of time and load,
[declared](what-is-not-hidden.md) items 3 and 8. Each commitment meets the judgement's other
conditions. All commitments keep the same three configuration files (`policy.txt`, `KL.bin`,
`layout.txt`). So two commitments to the same weights under different configuration files, for
instance under two acceptance policies, may each be inside alone, but not together. Why the same
configuration files: under two sets more is told of the weights than under one, in kinds that the
[declared list](what-is-not-hidden.md) names. Beyond these kinds no way is known in which the
commitments or the proofs tell more; only one short argument on paper has looked. By design, one
commitment already holds several tables of the same weights, each with its own randomness and OOD
answers. Eight short readings, each of a part of the host code, found no way in which two
commitments tell more than one. Three read other sources that differ from these sources in places,
and five read other sources that differ in more. None read the GPU kernels, but for the two kernels
that add the [pads](glossary.md#pad) to a table and to its columns. A count over two small
commitments, not on the released programs, found no derived random value twice, a check for gross
faults only. No test has searched two commitments together for leaks, and Lean has no statement on
hiding for more than one commitment. So the thin ground is the design, these readings and that count
(what a written proof says of several commitments: row H5.2). A further commitment resets only the
count of opened columns: each proved request's share of any bound on hiding, and what the
served-or-failed answers give, run on across commitments.

*What is not shown.* Lean's theorems on zero knowledge do not cover: batches, which are what the
programs make; after hashing, more than one leaf table; more than one commitment; a clipped-gap
policy; the simulators' running time. Coins derived from a seed are outside these theorems, and
nothing ties the Lean prover to the programs. Beyond Lean: tests on real weights and at full size
are few, and no serious try at recovering weights has been made. Of the Lean a reader needs to check
the statements and definitions, not the proofs ([which](../lean/AUDIT.md)).

*The main risks.* A lost mask, silently dropped by a bug or a hardware fault: a file with an
unmasked value can still verify, and no verifier can see it. Of the two, a hardware fault lies
outside an honest run, and so outside the judgement; a bug in the sources lies inside both. One lost
mask can give away at least one exact equation in hidden values, in a public file to everyone and
for good. In one small test, on test builds of other sources, 30 of 165 injected losses of random
values in CPU memory went unnoticed, and none of 130 in GPU memory (picked cases, so no rate). What
an unnoticed loss exposes these tests do not show ([more](#the-main-risk)). And a set-back use
counter: at full size, at the default setting, each copy of the first committed table (a [leaf
table](glossary.md#leaf-table)) has masks for 688,127 batches (calculated). Served further, its
masks run out, and whoever holds those proofs gets exact equations in the weights, unnoticed. What
the batches past the limit give away (by reasoning: row H21.11) has not been tried on the released
programs. By a reading of the code, the prover books each batch in the counter before it opens
anything, and waits until that is on disk. It takes no batch once a copy has no room for one (that
the counter keeps a table within its limit has not been proved for the programs: row HW-760; how far
the stop at the limit was tested: row HW-701, (4)). By the same reading a program that dies can only
leave the count too high, and a lock lets only one prover at a time run on a directory. That lock
may miss a prover on another machine, or one that reaches the directory by another route on the same
machine. A loss of power can lose a count; on a disk that keeps synced writes (a
[duty](operators-duties.md#keys-seeds-and-machines)) it loses none of a batch that has opened
anything. Even with that duty kept a loss of power can lose the count of a batch that was begun and,
by a reading of the code, has opened nothing, and the written proof of hiding does not cover a
prover that loses one (rows PS-15, PS-32). No loss of power has been simulated. What no program
detects: a secret key directory that is restored, copied or rolled back, or a counter file that is
shortened or replaced. Never do either ([duty](#the-conditions)).

Whether this evidence is enough for weights that must stay secret is each provider's own decision.

What Lean proves on hiding, in the longer form: [row LIM-64](limits-all-rows.tsv).

For a provider: [what you must do](#the-conditions) comes next; then [what the judgement rests
on](#what-the-verdict-rests-on): its conditions, the readings and runs behind it, what it does not
cover and one main risk, a lost mask; then [the biggest risks](#what-went-wrong) and [the rest in
brief](#the-rest-in-brief).

<a id="the-conditions"></a>
### What a provider must do

Breaking one of these duties can leak weights. No program can check most of them, so most breaches
go unnoticed, and, a breach of duty 6 apart, every proof verifies either way. Following them all
does not show that the weights are hidden. What is published cannot be recalled, and a new
commitment repairs nothing. The [runbook](operators-duties.md) gives the steps, and further
requirements under these duties that are as binding as those below.

1. **Never copy, snapshot, back up, restore or roll back the [secret key
   directory](glossary.md#secret-key-directory) (`-isk`), even for a test or as a backup that you
   never serve from.** A [leaf table](glossary.md#leaf-table) has masks for a fixed number of
   batches. A restored, copied or rolled-back directory, or a counter shortened or set back, lets
   the table serve [past its limit](#the-limit-of-use). Clients then get exact equations in the
   weights, and no program catches that.
2. **Keep the directory on a local disk of the one serving machine.**
3. **Never let two runs continue from one saved state of a machine or process, because their masks
   can repeat.**
4. <a id="the-providers-machine"></a><a id="shared-machines"></a>**Let only one service account read
   and write the directory, and let no one else write to a directory above it.**
5. **Do not let a plain, unsalted hash or checksum of the weights leave the provider.** Anyone who
   holds it can test guesses at the weights offline, as often as they like.
6. **Let no [response](glossary.md#response) leave you before `cp_verify` has accepted every proof
   of its batch: never give anyone a response that it rejected or has not yet checked, because such
   a response can hold unmasked values.** That holds of every file of a response, its token file
   too: nothing leaves that `cp_verify` was not given. Use a `cp_verify` built from the same sources
   as your `cp_prove`, and keep it where the service's account cannot write. `cp_prove` runs no
   verifier: it releases a batch's responses once all of them are staged. Of what ships, the vLLM
   integration hands on no proof, and nothing of its response, before `cp_verify` has passed every
   proof of its batch (narrow exceptions, none open to a client:
   [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`,
   which hands nothing to anybody, has the proofs checked once `cp_prove` has ended, and reports no
   rate unless `cp_verify` has accepted them all; each example script checks its one request's proof
   before it shows the answer; a front end of your own must do the like
   ([why](#a-response-that-does-not-verify); [after a kill or a failure](faults.md)).
7. **Serve only with the production build.** Under a test build's [test seed](glossary.md#test-seed)
   nothing is hidden, and no verifier can tell. **Never use `-public-weights` for weights that must
   stay hidden:** with it, nothing is claimed to be hidden
   ([details](what-is-not-hidden.md#public-weights)).
8. **Stop serving proofs on a machine after exit code 9, a response that `cp_verify` rejects, any
   other exit that you did not cause, or requests that keep failing. Do not restart there until you know
   the cause.** A fault that showed once can strike again where no check looks. Later responses may
   hold unmasked values, and so may earlier ones from the same process or machine. After any of the
   first three, hand on nothing more that this machine made, responses that already have their final
   names among them, until you know the cause. What a client already holds cannot be taken back.
9. <a id="the-limit-of-use"></a>**Publish a new commitment, with a new [commitment
   ID](glossary.md#commitment-id), before a leaf table is used up.** A commitment runs out and
   cannot be extended. A further commitment to the same weights, under the same configuration files,
   lies inside the judgement, on thin ground and on [conditions of its own](#a-further-commitment):
   a count over two small commitments, not on the released programs, found no derived random value
   twice (a check for gross faults only), and no test has searched two together for leaks. A single
   full leaf table stops the service, and the programs give no warning. So keep the line
   `the limit:` that `cp_commit` prints, and monitor each `count.bin` against it.
10. **Never create or replace a counter file, or put a writable mount over just that file,** or else
    a leaf table can serve past its limit unnoticed (duty 1).
11. **Put a [front end](glossary.md#front-end) in front of `cp_prove` that gives each client only
    its own files and limits how many requests each may send or have waiting.** The example vLLM
    middleware in this repository does not meet this duty ([the
    rules](what-is-not-hidden.md#front-end-rules)). So a server behind it alone is outside the
    judgement. Under [the clipped-gap policy](#which-policy), the front end must ask for proofs only
    of [replies](glossary.md#reply) that you served, over ranges that it sets. That narrows, but
    does not eliminate, what a client can measure ([risk 3](#which-policy)).
12. **Give the check of duty 6 the commitment ID from your own record, never from the secret key
    directory. Do not let anything restart the service automatically.**
13. <a id="the-seed"></a>**If a disk or machine has left your hands, assume that the directory has
    been read.** Anyone who reads it can strip the masks from every weight-table column ever shown,
    in the public files, and in every proof ever handed out that they also hold. That is argued, not
    tested. They can also open every published sum, with no proof served.

Public files that you have not given to anyone are themselves to be guarded. Why each duty stands,
and what a breach gives away: rows LIM-37.1 and LIM-37.2. What a breach gives away has not been
tried on the released programs.

Keeping all thirteen does not by itself bring a use under the judgement on hiding. Conditions of the
judgement's own, which are no duty of these pages: [told with the
judgement](#the-verdicts-own-conditions). None of the thirteen asks for these.

**Share the machine with no one.** The hiding claim also assumes that no untrusted code shares the
physical machine, under any user or in any virtual machine. Whatever runs under the service account,
a front end included, must be fully trusted. Caches, branch predictors and counters show other code
something of the programs' branches and memory accesses, and any user sees the GPU's load. So for
secret weights use a machine, a GPU and disks that you share with no one.

<a id="what-the-verdict-rests-on"></a>
### What the judgement rests on

**Words used below.**<a id="words"></a> A *test build* lets testers inject faults. A fault *went
unnoticed* if the file verified and nothing stopped it. An *audit* is a tool that makes part of the
published bytes again from the provider's secrets and compares them; the rest it only parses and
replays. For proofs it needs a *kept batch*: one whose random bytes were kept for the audit. *The
default setting* counts [proof of work](glossary.md#proof-of-work) toward its soundness figure;
every commitment publishes [OOD answers](#a-record). What an audit can check on a served proof: row
LIM-24.

*What the judgement means.* Of the programs "weight hiding" is the goal that the
[glossary](glossary.md#weight-hiding) states: clients learn nothing about the weights beyond the
declared list. It is not the cited definition's statistical index privacy, which these programs
cannot have at any useful error: they derive their masks from a seed, so that hiding is
computational at best. Nor do they have that definition's form: their prover keeps a counter, and
the definition is stated for a prover that keeps no state. <a id="the-grades"></a>The judgement is
the second of four grades: proved; likely holds (after the readings and tests told here, in honest
runs, with the provider's duties and the judgement's own conditions kept, no way to a leak beyond
the declared list is known, and the argument for the design is believed, but nothing is proved for
the programs); cannot say; does not hold (such a way is known). In the runs and readings that the
judgement rests on no flaw in how the pads are made and used was found (at five places the code is
not to the letter of the written proof's conditions: [below](#derived-pads)), and within its reach
no leak is known beyond [the declared list](what-is-not-hidden.md). That judgement does not cover a
real model's weights, or what a fit to them writes (a parameter file, a file of scales): only one of
the five tests has run on any. It reaches a [clipped-gap policy](glossary.md#clipped-gap), on less
evidence ([below](#which-policy)).

<a id="the-verdicts-own-conditions"></a><a id="further-conditions"></a>*Its conditions.* A correct
build on sound hardware. The thirteen duties and their further requirements ([the
runbook](operators-duties.md)). Conditions of the judgement's own, which are no duty of these pages:
the default setting; neither `-overlap` nor `-overlap-two` (two options for synthetic weights alone,
which, by a reading of the code, `cp_prove` rejects on a model's files: row H11.4; they are another
thing than the service's own way of taking the next batch before the last is handed out, which the
judgement takes in: "The default's own overlap."); exact match, or a clipped-gap policy in a
parameter file that differs from the default setting's in that policy's three numbers alone; for the
check of [duty 6](#the-conditions), a `cp_verify` built from the same sources as `cp_prove`, which
no program checks; and a seed that stays secret for as long as the weights do, also after the
commitment is retired, and in any backup that exists despite the duty never to copy the secret key
directory. Beyond what the programs themselves do, and what the duties bid after a program's bad
exit, or after a no of `cp_verify`, or no verdict, in the provider's check of a batch, at which
every request of the batch is answered "failed" as `cp_prove` itself answers it ([the declared
list](what-is-not-hidden.md)), neither the provider nor its front end answers "failed", withdraws a
response, or stops or restarts the service for a reason that can depend on hidden values.
`cp_commit` never runs twice from one saved state of a machine, such as a cloned or restored image.
Keeping them all does not show that the weights are hidden. A further commitment to the same weights
has [conditions of its own](#a-further-commitment).

<a id="honest-run"></a>**An honest run.** No one has changed the provider's programs, no fault has
been injected, and the compiler and the machine do what the published sources say. A bug in those
sources is not excluded: [the reviews](#what-supports-it) looked for such bugs.

<a id="on-which-programs"></a>*On which programs it was made.* The judgement is given of the
released programs. On them the first and the fifth of the tests told below ran: both at small size,
on synthetic weights; the first at full size too, once on synthetic weights and once on a real
model's public weights; the fifth ran once. Each ended as written down beforehand. One of the
fifth's two tools, the audit of a commitment, also ran at full size in [the timed
runs](#the-timed-runs), behind each of four honest commitments, one of them under a clipped-gap
policy, which is no negative test; the timed runs were on the released programs too. The second and
the third ran on test builds of other sources, and the fourth on programs of other sources; at full
size the fifth's two audits ran together only on programs of other sources; what the readings read:
[the readings](#verdict-reading). So what a run on other programs found tells nothing certain about
these programs unless it is repeated on them. The judgement does not reach other code: these sources
if changed, or if built otherwise (row LIM-63).

<a id="verdict-reading"></a>*What it rests on: the readings.* 384 files were read whole for hiding,
once and fast, and no leak beyond the declared list was found in a correct build. In the text that
was read, the 384 are all the files that a production build of `cp_commit` and `cp_prove` reads, but
for `cp_verify`'s own 36. In one of them the routine that feeds and finishes BLAKE2b hashing is not
the text that was so read; it was read for hiding apart (row HW-748). Nothing was built or run for
these readings. Of the 384, 68 differ in code in these sources, and 136 more in comments, names and
the wording of messages alone; two more are not among these sources; in the 68 the differences in
code were read for hiding, by difference alone, apart from renamed names and reworded messages, and
the files were not read whole again. (Rows HW-745, LIM-25.)

<a id="verdict-runs"></a>*What it rests on: five tests,* at the
default setting's small form or at full size, under exact match. Of the five, the first and the
fifth have run on the released programs, the fifth at small size, and one of its two audits, the
audit of a commitment, at full size too, on none but honest commitments, in the timed runs. The
second and the third ran on test builds of other sources, which hold the code that injects a fault,
and the fourth on programs of other sources that held test-only code ([what was run on these
programs, and what was not](#what-was-run-on-these-programs)). Unless a test says otherwise, it ran
on synthetic weights. None of the five searches for a leak: they test that the masks are there and
that a loss is noticed. The outcomes are reported: told from the runs' own accounts and logs, not
from a second, separate run. That the programs were the ones named rests on the runs' records.

1. *The OOD answers by value.* A separate program, written mainly from these pages, which shares no
   routine with `cp_commit`, computes the published answers again from the provider's secrets. On
   the released programs it ran three times at small size, on synthetic weights, and twice at full
   size, once on synthetic weights and once on a real model's public weights: at small size all
   1,536 answers of two commitments in one run, all 512 of one commitment in a second, and all 512
   of one commitment to a small synthetic model in files of the real files' form in a third (row
   LIM-63.2, line 11), and at full size all 4,096 answers of the commitment's two tables, both
   times, each commitment under a seed that `cp_commit` drew, were the row's weighted sum plus the
   public factor times the row's own pad; no pad came twice, and none was 0; at full size none
   equals any of the some 112 billion random entries and salt values that the rules which the test
   carries give for the first table; all 8 faults that the program injected for itself were found
   each time. (The run on a real model's public weights: row HW-747.) On other programs the first
   test also ran, several times at small size, and at full size twice on synthetic weights and three
   times on a real model's public weights, and ended the same way as on the released programs (one
   further trial ended with a failing status of its own controls: row LIM-29). Bounds: a short
   account of the source was at hand when it was written; the program holds ten known answers out of
   a header in one form of it and 16 known answers in another (where the six further ones come from
   has not been asked, and which form ran in these runs is not told), and the weights' integers
   reach it through the project's integer model. The path on which a block of pads needs a second
   try has not been seen to run. The 8 faults show that the program's count on the pads can ring,
   not that each of its three counts (equal pads; equal single field values; pads that are 0) could
   have: no fault tells them apart, and that the program counts all three is read, not tested. Nor
   do they show that the search among the some 112 billion values would have counted a match with a
   pad: its control plants two values that are counted on a branch of their own, so that too is
   read, not tested. Whence a seed's 32 bytes come, and that they are drawn fresh at every call, the
   test does not show. The pads are derived from a secret seed, so that they are random rests on the
   cipher. The test reads the stored answers at the front of the public files, not the committer's
   own proofs behind them; that the pads under a table's root are those in the answers is not its
   matter. It says nothing on what an answer gives away, or on whether the masks hide.
2. *Lost pads.* 8 of 8 injected losses of pads ended `cp_commit`, under its self-check on pads and
   no other, and none left a public file. All eight strike the key file's table: none was of a
   single pad or row, at a leaf table or in `cp_prove`. A test build of other sources, a test seed,
   all weights 0, small size. Bounds: by a reading of the code one self-check in `cp_commit` is all
   that stands between a fault that loses every pad and a public file without pads; in a trial on a
   test build (without proof of work, which the default setting counts), with it switched off, a
   commitment without pads passed everything else that `cp_commit` does.
3. *The sweep of lost random values*, at small size, on test builds of other sources, whose tie to
   the released programs has not been shown (row HW-660): 295 runs with a picked loss. In CPU memory
   165: 123 were stopped, 30 went unnoticed, and after 12 more the files were byte for byte an
   honest run's, so these show nothing: 30 of the 153 that could show anything went unnoticed. In
   GPU memory 130: all were stopped, 122 of them by one self-check alone. The 295 runs struck 189 of
   73,177 draws, and the 30 are 27 different draws. It is one fixed sample (seed, coins, request,
   parameter file). Bounds: no rate; not every random value was struck; the sweep does not reach the
   pads on the OOD answers, the sub-proofs' masks or the weight tables' random entries, which come
   from other generators; what an unnoticed loss exposes the test does not show.
4. *The riskier losses outside that sweep*, replayed at small size, on programs that held test-only
   code, whose tie to the released programs has not been shown: of 263 losses of draws that the
   self-checks hold all 263 were stopped; in a control build without these self-checks the same 263
   all went through under the verifier's yes; 9 more, at neighbouring draws that no check holds,
   went unnoticed. No rate; the 9 are single examples: far more draws of their kinds are held by no
   check. Not run: six folds' coefficients (and all but the last of two more), 17 folds' tails, the
   OOD answers' own pads. Bounds: none strikes a draw of the sub-proofs' own generator; others of
   that sort were not among the losses replayed; nothing at full size.
5. *Both audits with the provider's secrets*, on a commitment and a kept batch. On the released
   programs, at small size, once: with this repository's own audit files both audits accepted the
   honest files, and `cp_verify` accepted the four proofs of two kept batches, under two
   commitments, each under a seed that `cp_commit` drew; when a false file stood in turn for each of
   three kept files, under the tool's `--rehearsal` option, the audit said no each time; for a
   fourth kept file none was tried; no altered public file was put to them. At full size, in [the
   timed runs](#the-timed-runs), only the audit of a commitment has run, on three honest
   commitments, and on a fourth under a clipped-gap policy (row H13.7): by the runs' own account
   alone it accepted all four, which is no negative test. Not on the released programs, both audits
   said yes at full size too, on one commitment and one kept batch of two proofs, and the
   commitment's audit said no when 32 other bytes stood for its kept secret (in that run one of
   three kept files was exchanged at that size; another run: row LIM-63.2, line 5). How much an
   audit compares by value, on the released programs: at small size the commitment's audit 32.94 %
   to 33.00 % of the bytes of key file and one entry (the tool's word for a leaf file; three
   commitments, two of them under a clipped-gap policy), and 29.27 % of those of key file and two
   entries (one commitment); the batch's audit, with two proofs beside these files, 18.76 %, 18.78 %
   under a clipped-gap policy, and 17.87 % with two entries. At full size one share is known, under
   a clipped-gap policy: 10.18 %, by the commitment's audit (row H13.7); under exact match none is.
   In a run not on the released programs, at full size, 10.22 % and 7.66 % (of the two proofs alone
   8.91 %). Run by run: row LIM-63.2, line 5. What an audit does not compare it parses, replays and
   holds to its roots. Bounds: the audits hold no weights, so at either size they compute again only
   what is made of random values, and no value that depends on the weights; whether a mask was laid
   on the value that it should hide they do not ask; by their own output they do not check pads
   inside tables of real values, the pads of the stored tables' OOD answers (their output says
   "records"), or the stored tables' columns; they have never been shown a faulty public file at
   full size; they reach no served proof.

Tests 2 and 3 in full, with the runs that tried the rejection of more than one OOD answer a row (at
two alone; at three or four no run), and the 30 by their kinds: rows LIM-26.1 and LIM-26.2. All
five, run by run: rows LIM-63.2 and LIM-63.

<a id="what-a-reader-cannot-repeat-on-hiding"></a>*What a reader cannot repeat, on hiding.*
No conditional in the sources reads a switch that builds a fault of hiding into a test program. What
the sources do have are test hooks and test variables by which every `cp_commit` and `cp_prove` but
the production build's can be told to inject a fault of hiding; those that were read for it act only
in a run on synthetic weights under a test seed, and by a search no test of this repository sets one
of them (row E29.1). What fails on purpose in a reader's copy are faults that a test injects into
its own random values, fixture or generator, among them: the five negative controls of
`blocks/tests/hiding/table_exact_test.cpp` and the three of `blocks/iop/tests/batt_hiding_test.cpp`
(one of the three can be made in only four of that test's five cases); those of `psc_exact_test`,
`lin_exact_test`, `link_exact_test`, `hull` and `exact_self_test`; and, for the prover's
self-checks, those of `send_pad_test` (own-test-pad-kept), `fold_base_test` (own-test-fold-mask,
own-test-base-mask) and `context_test` (own-test-mask-rows and the guards on opened positions). No
run in this repository shows that `pads_fault()` (`blocks/iop/send.h`) reports a pad that is missing
or used twice; that `salts_test` sees shared or zero salts; that `coins_own_test()`
(`blocks/coins/coins.h`) catches a generator that rejects no word; that `fold_rounds_test` finds one
value under two coefficients; that `toy_exact_test`, `rng_exact_test`, `lin_exact_test`,
`link_exact_test`, `key_exact_test` or `hull` fails on a faulty form of the library's own prover;
that `stream_coins_test` or `coin_keys_test` fails on a faulty build; that a verifier rejects a
batch or a key under a second OOD sample on a table's pads, or that two such OOD samples reveal an
exact linear equation on each row (`docs/format/FORMAT.md`, R4.12); that a request that is made to
fail by one of three injected faults (a value inside the prover is pushed out of its range or
window) samples its random values as the others do (`prover/tests/coin_contexts_listed.py`); or that
a verifier's routines reject a value that the GPU code sends without its mask (row T40). Nor does
any check show that the GPU library's products write every word that is read afterwards (row
T3.1.1).

*The default's own overlap.* The judgement takes in the service as it runs by default, with batches
that overlap in time, on thin ground: no proved statement covers it. No attack is known: three
readings looked for one, minutes each, one of them in other sources, with nothing of the service
run, and no test of hiding was aimed at it.

*Batches with a failed request.* The judgement takes in batches with a failed request, which anyone
who can submit tokens of their choosing can cause, on thin ground: the other requests' files of such
batches have not been examined ([what was never tried](#what-was-never-tried)). A request that a
client makes fail is no injected fault, so such a run counts as honest.

<a id="a-separate-audit-at-full-size"></a>*A separate audit at full size.* One full-size commitment at the
default setting, on synthetic weights, not made by the released programs, was audited with a
separately written program, typed from a text that differs from these docs (by a reading, the rule
is the same in both), with gaps filled from the sources, which derived the values again from a copy
of the secret key directory (why a copy was made here, and must never be made of weights that must
stay hidden: row LIM-28). The audit ran once. It is partial: it does not reach the commitment ID,
the random values of the proofs in the key file and the leaf file, or anything of a request's proof
beyond the stored tables' columns and salts; the tables were not made again whole; the pads were
typed once only, with no second typing to hold them against; on what the synthetic weights are that
program is no second voice; no control changed a file of the programs, and none aims at a pad's
second try. It shows, as far as one typing of the pads can, that the sums in these files were masked
as the pages say, at full size. It says nothing on whether the masks hide. None of its checks
applies to the released programs unless it is repeated there. What it compared, what agreed, and the
rest: row LIM-28.

<a id="not-run-or-barely"></a>*Not run, or barely.* The trial with the self-check on pads switched
off (test 2's bounds) is one trial, on a test build. No exact-law test of the Lean library is known
to have run on the published OOD answers. Whether weights can be recovered from the files has barely
been tested: one short search of the public side of two small honest sets found nothing, and on the
first of these sets its first look cannot see any fault of a pad at all (how its controls went, and
an uneven count in the pads that was seen afterwards and booked as chance: row LIM-29). Two unit
tests that failed in the runs that the judgement rests on, and why: row LIM-29.

<a id="limits-of-the-programs"></a><a id="not-covered"></a>*Not covered.* Whatever falls outside
[these conditions](#the-verdicts-own-conditions), such as any other setting (one with 2, 3 or 4 OOD
answers a row among them), any other policy, or a service run with `-overlap` or `-overlap-two`.
Beyond that: a real model's weights, and what a fit to them writes (a parameter file, a file of
scales): only one of the five tests has used any. Weight files in which any tensor that the programs
read is stored in 8 or 16 bits ([why](#files-of-8-or-16-bits)); nor does anything else that these
pages say about weight hiding cover these. Any commitment made with `-public-weights`: that option
of `cp_commit` hides nothing, and neither the public files nor the proofs show that it was used
([details](what-is-not-hidden.md#public-weights)). Commitments to the same weights, taken together,
where one of [the conditions](#a-further-commitment) is broken, above all where their configuration
files differ, as under two acceptance policies. The judgement is about clients, not about anyone on
the provider's machine. How `-overlap` and `-overlap-two` work inside has not been read. What these
pages say about weight hiding has been checked only as far as [the judgement](#the-verdict) tells.
The judgement is given at [target_full_f2](../prover/examples/TARGET-KITS.md)'s files and at the two
small forms at which its tests ran, target_small_f2 and target_small_f2_roomy; it reaches the two
_gap configurations on less evidence. It does not reach target_small_file_f2's parameter file, which
differs in more than a policy's three numbers: of the tests of hiding the first alone has run at it,
once on the released programs (row LIM-63.2, line 11). At target_small_f2 its tests ran on test
builds under a test seed, which leaves cp_commit's worst-case check out. The judgement covers no
other parameter file that the programs take, but for a file that differs from the default setting's
in a clipped-gap policy's three numbers alone, which it reaches on less evidence. The list above in
its longer form: row LIM-30.

<a id="the-main-risk"></a>
*One main risk: a lost mask.* If a bug or hardware fault silently drops a random
[mask](glossary.md#mask), a file with an unmasked value can still verify, and no verifier can see
it. [Tests 3 and 4 above](#verdict-runs) tried such losses: their counts and bounds stand there.

In test 3's sweep 30 of 165 picked losses in CPU memory went unnoticed (12 of the 165 changed no
byte of any file, so 30 of the 153 that could show anything), which gives no rate. Test 4: all 263
replayed losses of the riskier sort (of values whose loss by itself could, by readings of the code,
expose something of the weights) were stopped; 9 more, at neighbouring draws that no check holds,
went unnoticed (other values of the riskier sort were not among those replayed). No rate; the 9 are
single examples: far more draws of their kinds are held by no check. The written proof counts some
of these values as random, and whether it needs that of each is not known.

What share of all draws the kinds of these 30 make up has not been reckoned for this sweep. 16 of
the 30 are spare random entries of mask rows, and nine of these 16 are the heads of stretches of 58
to 5,116 neighbouring draws, 14,970 in all, of which the head alone was struck (row HW-660). By two
short readings of other sources, about ten minutes each, none of the 30 exposes anything of the
weights by itself, but for one case, where it is not known (a batch's lost salt beside a failed
request); no run has shown that. The written proof of hiding counts some of the 30, or values of
their kinds, as random or fresh: a run that has lost one lies outside what that proof is stated for,
and what that costs is not known. No pair of losses was tried, and no loss later than right after
the draw. In small computed examples some pairs of losses did leak; none of this is proved or
checked (rows HW-660 and H17.3.1.1). Tests 3 and 4 ran at small size, on synthetic weights, not on
the released programs; not every random value was struck, neither reaches the pads on the OOD
answers (test 2 does), and what an unnoticed loss exposes they do not show. The self-checks narrow
this risk but do not remove it ([the outcomes](how-hiding-is-checked.md)).

<a id="a-response-that-does-not-verify"></a>*A response that does not verify.* `cp_prove` runs no
verifier: it releases a batch's responses once all of them are staged. In an honest run on a sound
machine no case is known in which a verifier has anything to hold back, but for the rare hash
outputs at which `cp_verify` does not accept an honest proof (row C2.2); that there is no other has
not been shown (row HW-580). `cp_prove`'s self-checks on masks, pads, salts and keys, and the proof
library's guards on the prover's own steps (exit code 1, name `guard`), end the run after some bugs
and hardware faults, before the batch's responses have their final names (row C3); which they let
through has not been sorted. After one that they let through, a response can leave `cp_prove` that
`cp_verify` rejects. A response that `cp_verify` rejects can hold hidden values unmasked: by one
reading an exact linear form in the weights; no upper bound is known for any cause. The cases that
are known are read in the code, and none was run (row H26). No test has tried a file of that kind
(row H26.3). So `cp_verify` must accept every proof of a batch before anything of that batch leaves
you ([duty 6](#the-conditions)). Of what ships, the vLLM integration hands on no proof, and nothing
of its response, before `cp_verify` has passed every proof of its batch (narrow exceptions, none
open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions));
`driver/measure_rate.py`, which hands nothing to anybody, has the proofs checked once `cp_prove` has
ended, and reports no rate unless `cp_verify` has accepted them all; each example script checks its
one request's proof before it shows the answer; a front end of your own must do the like. The check
of duty 6 leaves the main risk as it is: a file that has lost a mask and still verifies passes it.

Without its pad each published [OOD answer](#a-record) is one exact linear equation on a row of the
weights. Masked sums of the weights are public for good, a single self-check is all that stands
between a fault that loses every pad and a public file without pads, and the seed must stay secret
for as long as the weights matter. The longer form, with what one seed under two commitments gives
away, and what a lost mask in a proof costs: row LIM-65.

**What a lost mask is.** A mask is lost when a bug, a lost write, or a fault in the hardware, driver
or compiler keeps it from reaching the value it should hide. A single lost mask can give away at
least one exact equation in hidden values: to one client, to a whole batch or, in a public file, to
everyone and for good.

Such faults are real: four compiler faults are known, and at least three of them can act silently.
By the build scripts' own words, the host compiler can silently drop writes to memory. The build
turns the faulty optimisation off, but whether that removes every case is not known. Whether the
required GPU compiler has another of the four is not known either ([the compiler](#the-compiler)).

<a id="what-supports-it"></a>**How thoroughly was hiding checked in the programs?** Not thoroughly.
The readings of the code took minutes, not days, and nothing of the programs was built or run for
them; they looked at what is drawn, placed, zeroed, stored, sent or printed, not at whether the
arithmetic is right. No reading has held the released programs' sources against what the written
proof asks of the code (what was held against other code: rows HW-584 and PS-62). The runs are few,
on synthetic weights, mostly at small size ([the five tests](#verdict-runs)). The reviews of the
code, and how far each went: row LIM-31. <a id="whose-verdict"></a>The [judgement](#the-verdict) is
one on the evidence, not a proof. The written proof's grade, "complete but not settled", is about a
proof on paper; it does not say that the programs hide the weights. Whether the evidence is enough
for weights that must stay secret is each provider's decision, for its own weights.

<a id="on-paper"></a>**On paper.** The report's proof of hiding, for a prover that the report
defines, is what its authors call "complete but not settled": the proof is written out in full, but
it has been checked too little to rely on it. It concerns only the protocol on paper, and it says
nothing of the programs. It has been read in parts, mostly for minutes, and never, as far as is
known, as a whole; nearly every time that a whole part was read, further sentences were found that
were false as worded. Reading has missed real errors: four in statements that the proof rests on, a
fifth in the definition of zero knowledge itself, and more that have not been counted. It is not
claimed that every known error is fixed. There may be more errors. The report also lists differences
between its protocol and the programs, found by reading other sources (row PS-67); twenty-two of
them leave hiding open. What is hidden depends on the prover's GPU version of a segment's inner
steps, but most of that version was not compared, so the comparison says less about hiding than
about binding (rows PS-38, PS-39 and PS-47). At three more places the programs leave the protocol
that the report's statements are about (what a proof's first hash takes in; the labels of two of a
proof's chains; how the commitment ID is formed). Read literally, the technical report's statements
on random values that the prover derives from a stored key do not cover the way in which the
released programs are described as deriving them. The proof rests on one lemma, on composing steps
whose masks are sent in advance. Readings name two points at which one hypothesis of that lemma is
wanting: it lacks a clause that the proof uses (row RES-66), and it does not say what it asks of a
run that fails. So, read strictly, one use of the lemma in the proof of hiding lies outside it: the
use for the range check, when that check fails. So does the use that repeats it for the protocol on
paper taken in the programs' order of messages. That the proof covers a failing range check all the
same is the word of two of the three readings that found the point, after minutes each. Without the
lemma the whole protocol would have no written proof, and a failure of it could point to a leak in
the protocol. Every figure drawn from the written bound on hiding would be without a written proof
too. Whether the lemma as it is worded holds is not known: it is not claimed that either point on
its hypothesis is answered (row HW-763). Whether its conclusion could be had by another route if it
failed cannot be said. No way of reading the lemma as it stands under which it is false is known
(what was tried, for minutes, and what was not searched for: row RES-57). The lemma as it is worded
has not been read for the judgement on weight hiding. Lean proves a lemma of this kind
(`Iop.goodT_seq_masked`, lean/Ior/KnowCollRulesMaskProofs.lean; stated in
lean/Ior/KnowCollRulesMask.lean); nothing on these pages ties it to the written proof's lemma. The
errors: row HW-579. That lemma: row HW-763. Further limits on that proof, among them for which
prover it is, what it assumes about the hash and about derived random values, what its results do
not reach, and that the report states its security levels for another kind of last segment than the
programs run: rows PS-23 to PS-34 and PS-44. What the written proof says of batches that overlap in
time, and what stays outside it (no limit is known on what time or a stop reveals): row LIM-19.

<a id="what-is-built"></a>**What is built.** The masking needs nothing switched on, but hiding is
not proved for the programs ([what is proved, and of what](#weight-hiding)). By design, every
weight-dependent value that the programs publish is masked or salted, apart from a [declared
list](what-is-not-hidden.md); with [`-public-weights`](what-is-not-hidden.md#public-weights) nothing
is hidden. That no other option turns hiding off rests on a reading of other sources.

<a id="against-whom-the-claim-is-made"></a>
**From whom.** The weights are meant to be hidden from clients, even if they pool what they hold,
though this has not been tested with pooled proofs. They are not hidden from the operator, or from
anyone who can read or write the provider's directories, memory, swap, backups or logs. Nor are they
hidden from whoever gets the GPU next while unmasked values are in its memory ([hidden values left
in GPU memory](#gpu-working-values)). The project makes no claim about timing, memory use, machine
faults, or several machines serving from one secret key directory.

<a id="what-has-run"></a> **What has run, and on what.** Of the tests
of hiding that these docs report, only those named in the table under [what was run on these
programs](#what-was-run-on-these-programs) ran on the released programs. The released programs
differ in code, much of it bearing on masks, from the programs behind most other results; those
results do not cover what differs. Unless stated otherwise, each trial ran once, at small size, on
synthetic weights. Few tests ran at full size. The self-checks have been seen to fire only in test
builds: no self-check on a lost random value has been seen to fire in a production build, which has
no option for injecting a fault. Many random values are compared with nothing (rows LIM-33.1 and
LIM-33.2). Test by test: row LIM-63.2; more on what has run: rows LIM-32.2 to LIM-32.4; the replays
of losses outside the sweep: row LIM-36. How far it is known which programs ran: rows HW-678, E30 to
E32 and LIM-26.1.

<a id="what-hiding-rests-on"></a>
**What else is missing.**

- No security level is quoted for the programs' hiding. The project's 128 bits are a per-round
  target for soundness that the program computes, not a security level. Where these pages compare
  the written bound on hiding with 128 bits, they speak of the protocol on paper
  ([there](#a-prover-that-stops)).
- Two commitments to the same weights have not been searched together. A provider that serves long
  enough will need a second ([duty 9](#the-conditions)), and Lean has no statement on hiding for
  more than one commitment.
- It has not been shown that the declared list is complete.
- More: what no proved statement covers, what ran at the default setting, the one self-check on the
  OOD answers, and what is not shown or not known: rows LIM-34.1 and LIM-34.2, row LIM-23.

<a id="what-went-wrong"></a>
### The biggest risks

One finding, in which no leak was shown: the number of events in a reply streamed by vLLM
([integrations/LIMITS.md](../integrations/LIMITS.md)). No one has looked for a way for a client to
cause the fault under 1.

1. <a id="a-lost-mask"></a>**A lost mask: the proof, or the commitment's public file, can still
   verify** ([more](#the-main-risk)). No loss of a mask has been tried on the released programs,
   which have no option for injecting a fault. What the self-checks do not cover: row LIM-67. A bug
   or a hardware fault can also spoil a proof so that it does not verify; `cp_prove` runs no
   verifier, so such a response is released like any other, and it can hold unmasked values ([a
   response that does not verify](#a-response-that-does-not-verify)).
2. <a id="a-broken-duty"></a>**A broken duty, or access to the provider's machine, can undo the
   masks with no fault in the programs.** What breaks in each case, under duties 1, 2, 4, 10, 12 and
   13: [what a provider must do](#the-conditions) and rows LIM-37.1 and LIM-37.2. What a breach
   gives away has not been tried on the released programs.
3. <a id="which-policy"></a><a id="what-every-client-learns"></a><a id="what-the-design-shows"></a>**What
   any client learns, by design.** At least eleven things are not hidden ([the declared
   list](what-is-not-hidden.md)), among them each reply, whether a request was served or failed, the
   time that the work takes, the model's architecture and all its sizes, settings fitted to the
   weights, and [masked sums of the weights](what-is-not-hidden.md#record-sums). Under every
   acceptance policy a client learns each reply, which itself depends on the weights, and whether a
   request is served.
   - Under [the clipped-gap policy](glossary.md#clipped-gap), each outcome (served or failed) tells
     the client more. A client that chooses the checked tokens can measure the model's hidden scores
     over many requests. No theorem bounds this or what many yes-or-no answers reveal together. No
     one has measured it, and none of these programs limits requests.
   - Exact match is the default's policy; ordinary serving uses the clipped-gap policy. So under
     ordinary serving the model's hidden scores are not hidden from a client that sends many
     requests and chooses the checked tokens, and no statement bounds how much that tells of the
     weights. What each outcome tells and what such a client can measure are both on [the declared
     list](what-is-not-hidden.md).
   - The judgement reaches clipped gap, on less evidence. (A clipped-gap policy in a parameter file
     that differs from the default setting's in that policy's three numbers alone: [its
     conditions](#the-verdicts-own-conditions).) No Lean theorem covers what the outcomes (served or
     failed) tell about the model's scores. Lean's theorem on zero knowledge is for exact match:
     under a clipped-gap policy no Lean theorem covers what a whole proof reveals, and, audits
     apart, no test of hiding ran under a clipped-gap policy. What the technical report says of such
     a policy: row PS-49; how its corollary on hiding under a policy was read, and what was found:
     row LIM-23. What was read and run instead, with its bounds: row LIM-27, row LIM-40.

### The rest in brief

- <a id="files-of-8-or-16-bits"></a>**Weight files of 8 or 16 bits.** What these pages say about
  weight hiding does not cover a commitment made from weight files in which any tensor that the
  programs read is stored in 8 or 16 bits (Q8_0, F16, BF16), or the proofs made under it. No test of
  hiding has run on such files on the released programs. One trial with a BF16 file, not on the
  released programs, ended badly (row H46.4 of [the table of all rows](limits-all-rows.tsv)):
  somebody who could write that file but not read it learned the magnitudes of two neighbouring
  weights of their choice in about 32 starts (small size; synthetic weights; reported). No fault was
  injected in that trial. Such files meet no rejection. `cp_commit` and `cp_prove` take them, and a
  mostly 4-bit file with a single such tensor, under a sampled seed like any other file, and no
  printed line says that one was read. So it is for the provider to know how each tensor of its
  files is stored. The public 4-bit conversion that the examples use holds no such tensor (by two
  listings of the files' headers). What changes with such files in GPU memory and in timing, and how
  little the three kernels are tested: row LIM-41.
- <a id="a-record"></a>**Published OOD answers.** A commitment also publishes, for good, for each
  stored table, one weighted sum of each row's weights, under a random pad of its own. The programs
  derive the pads from a secret seed ([what that assumes](#derived-pads)). No verifier can see
  whether the pads are there, or are random. At how many OOD answers a row the programs take a
  parameter file (no written proof and no review covers 2, 3 or 4 OOD answers a row) and at how many
  they reject it, and which runs tried that: row LIM-42.
- <a id="derived-pads"></a>**The pads on the OOD answers are derived: an assumption.** The written
  proof's theorem on published OOD answers draws each pad fresh. These programs derive it. So a
  published answer is hidden only as long as the seed stays secret and ChaCha20, used as a
  pseudorandom generator, is not broken, while the file is public for good. The technical report has
  a corollary for derived pads, under four conditions on the rule that derives them. It is graded
  "complete but not settled", like the whole [written proof](#on-paper). The report does not say
  that the programs' own rule meets them. In any reading that corollary is about the protocol on
  paper, not about the programs ([what is missing](#what-hiding-rests-on)). What the report says on
  published OOD answers has not had an unhurried reading. How a pad is derived, how two short
  readings held the four conditions against the code of other programs (at five places the code is
  not to the letter of them), and which self-checks the corollary does not cover: rows LIM-43 and
  H47.8.
- <a id="if-a-cipher-breaks"></a>**If a cipher breaks.** The programs derive their random values
  with ChaCha20, so hiding is computational at best. If the cipher or the hash is broken later,
  everything published by then can be unmasked or tested against guesses.
- <a id="a-prover-that-stops"></a>**On paper, below 128 bits.** For a prover that stops whole
  batches a term of the written bound on hiding comes out below 128 bits. `cp_prove` runs no
  verifier; the provider's check (duty 6) withholds the whole batch at a rejection, so the two
  together are on paper such a prover, and the term applies (inferred: row HW-740). For the released
  prover, which also stops at tests for which no bound is stated, the report has no bound at all.
  The longer form: row LIM-44. The technical report's computed figures on hiding are for fresh
  random values alone and leave out several terms: one has no number at all. The technical report
  that is not public builds the lookup in two forms. By its own comparison, not on the released
  programs, the programs run the second, with tags and padding, but for what a lookup passes on (row
  PS-5). The report's figures are for the first. A term below 128 bits is not a leak, and no attack
  through it is known to us (these docs' own words, not the report's; no search for one is known).
  Three lower figures that the report prints, and two things that it leaves unsaid: row LIM-44.1.
- <a id="gpu-working-values"></a>**Hidden values left in GPU memory.** A separate finding, from
  reading the code, with no fault injected: GPU functions leave hidden working values, decoded
  weights among them, in memory of the provider's own GPU that nothing clears or that the programs
  cannot wipe. The machine code of most of these functions has not been read; the two that were
  read, in other programs, both leave values there that are computed from a request's activations
  (rows HW-212 to HW-218 and HW-434). By that reading no client sees that memory in a run without a
  fault, as far as is known (a report that bears on it, reads of uninitialised GPU memory in a
  served batch, not on the released programs, is not fully explained: [T3](limits-further.md#t3));
  the service needs a GPU of its own, shared with no one and never passed on. After every run, even
  a clean one, free GPU memory holds hidden values, some of them unmasked, and we do not know which
  reset, if any, clears that memory, and that includes a reboot
  ([operators-duties.md](operators-duties.md#keys-seeds-and-machines)).

## Evidence and review

<a id="what-was-run-on-these-programs"></a>
### What was run on these programs, and what was not

*In short.* The runs that this section reports as runs on the released programs were made on
programs built from sources that differ from these in comments, names and the wording of messages
alone; of the programs that these sources themselves build it reports three small runs and one at
full size ("Which programs ran", below); the sources of `cp_verify` do not differ. In the test runs
that cover all 173 public tests, most of which start none of the three programs, 170 passed, one
passed in part, two were not run and none failed. On synthetic weights, the released `cp_verify` and
the Lean verifier were given the same sets of full-size files, as reported: both accepted all 47
honest proofs, and neither accepted one of the 57 lying proofs; at 3 of the 57 the Lean command's
parser refused the file, which is not a verdict. The Lean verifier also accepted 41 further proofs,
on synthetic weights, as reported of the timed runs. One proof made at full size, on synthetic
weights, by the programs that these sources build, as reported, was accepted by `cp_verify` and by
the Lean verifier. `examples/maverick/run.sh` ran at full size on a real model's public weight
files, without building, downloading or installing: its five stored values came out the same, byte
for byte, and `cp_verify` ended with `YES`. In the timed runs on those weight files, at the default
setting, all 1,851 requests were served, and all 1,851 proofs, each of the one token after the
prompt, were accepted by the runs' own calls of `cp_verify`. The first and the fifth of the five
tests of weight hiding ended as written down beforehand, the first also at full size, once on
synthetic weights and once on a real model's public weight files.

*What is left.* The 57 lying proofs are files made beforehand by test programs that this repository
cannot build, all at one commitment and for three requests: no cheating prover was run against the
released programs, and no refusal at the lookup checks themselves has been seen. At small size
`cp_verify` accepted 228 replayed calls on altered stored tables: that is right if each table still
decodes to the honest weights alone, which rests on counts of the changed columns: no decoder was
run. Neither test of weight hiding shows that the masks hide, and the other three of the five were
made on other programs; no run shows what the programs leave in host or GPU memory, and no audit
with the provider's secrets has run on a commitment to a real model's weight files. Two runs at 64
requests on synthetic weights failed, and no cause is shown. The tests and scripts of those test
runs differ from this repository's, a few in code; of the public tests that need a GPU or the CUDA
compiler only two have been run on this repository's own files. The programs that ran and the runs'
records are not in this repository, so a reader cannot check how they are tied to these sources.
Nothing in this section is a proof about the compiled programs.

*Which programs ran.* "The released programs" are `cp_commit`, `cp_prove` and `cp_verify` as the
sources in this repository build them (the production build). The programs that made the runs that
these pages tell as runs on the released programs, or on a test build of these sources, were built
from sources that differ from these in comments, in the names of identifiers and files, in seven
quoted texts that hold such a name and in the wording of eight messages (seven that `cp_commit` or
`cp_prove` prints, one that a build script prints), and in nothing else. These pages tell a program
built from those as built from these sources, a run on it as a run on the released programs or on a
test build of these sources, and a reading of those as a reading of these sources; how far that tie
is shown, and where it ends, is told under [What a reader can check](#what-a-reader-can-check). This
rule speaks of `cp_commit`, `cp_prove` and `cp_verify` and of their test builds. How the tests' own
files, the serving scripts and the other scripts of those runs differ from this repository's (they
differ in code too) is told under "What these runs do not show", at "The public tests". Of the
`cp_commit` and `cp_prove` that this repository's sources build, these pages tell four runs, three
of them here alone: the first and the fifth test of weight hiding, once each, at small size, on
synthetic weights, each ending as written down beforehand; and a comparison with the middle build
told under [What a reader can check](#what-a-reader-can-check), on synthetic weights: under one test
seed the test builds of both sides gave the same files, byte for byte, at small size, and so did
both production builds of `cp_commit`, called with `-public-weights`, at small size. The fourth made
one commitment and one proof at full size, on synthetic weights: the second table's last row tells
it. The counts of runs elsewhere on these pages leave these four out. These pages tell of them no
timed run, no other run at full size, no run of `examples/maverick/run.sh` and no public test that
starts one of the two. The times, speeds and peaks of memory that these pages give as the released
programs' were measured on the programs of the runs, and are told of the released programs on the
comparison alone. No program file is in this repository: a reader builds them, and no run told on
these pages was made on a reader's or a provider's own build. The runs told on these pages are of
these kinds, told below in turn: runs on the released programs (the timed runs are among these);
runs on [other programs](#on-other-programs), test builds among them; and, told apart, the calls of
the Lean verifier. Rows are those of [docs/limits-all-rows.tsv](limits-all-rows.tsv).

*What ran on the released programs.* The programs that ran were built from the sources told above,
and are tied to those by two digests: the verifier's build word, and the digest in each prover's
build record (`cp_commit.build`, `cp_prove.build`). The verifier's build word is this repository's
too; the provers' digest is not: comments, names and the wording of messages turn it. Those records
are not in this repository, so a reader cannot check that tie. Where a run's record holds no digest,
its programs' checksums are those of the build whose record holds both. Pages lie outside both
digests, and so do nearly all tests and most scripts. Of the 469 files under the provers' digest 33
are not sources, six scripts and a Makefile among them ([what a reader can
check](#what-a-reader-can-check)). A [test build](glossary.md#test-build) leaves no build record.
That the test builds of these runs came from these sources rests only on the logs of the test runs
that made them, which are not in this repository; each checked a checksum of the sources before it
built. Weights are synthetic, and each run was made once, unless a line says otherwise. The table
tells what ran on them. Of the public tests only some rows start one of the three: how many, and how
many run test builds instead, is told under "What these runs do not show". A run that neither the
tables nor the part [The public tests, in full](#the-public-tests-in-full) names is to be taken as
not made on the released programs, but for one test that is not in this repository, which by a
reading of its commands ran them and passed (row RES-39), and the runs of the check by the vLLM
integration's middleware told under [the timed runs](#the-timed-runs), and the four runs told under
"Which programs ran".

| what ran | size and data | how it ended |
|---|---|---|
| the public tests ([SUITE.md](../SUITE.md): 173 rows) | one H100 a test run, or a CPU machine; small sizes, synthetic weights, but for ten rows that give `cp_verify` a full-size parameter file or a stored full-size example, also on synthetic weights, and a few that run a single kernel, sub-proof or matrix at a full-size shape, also on synthetic data (among them `gpu.base_kernels_test`); no row runs `cp_commit` or `cp_prove` at full size | On programs that each test run built from these sources, 170 of the 173 rows passed, one passed in part, two were not run and none failed. Apart from those test runs, on this repository's files alone (without `lean/`; some pages, tables, `paper/paper.pdf`, one list and licence and notice files differ), 90 of the 105 rows of the kind `cpu` passed and two passed in part; the other 13 need a set-up row that needs a GPU, and the runner's `cpu` mode leaves such rows out. Two of the rows that need a GPU or the CUDA compiler have been run on this repository's own files, once, as far as their files were compared, and passed. In full: [below the tables](#the-public-tests-in-full). |
| `cp_verify` | the four stored examples, two at small size and two at full size; 20 altered sets of the two at small size, ten of each; 57 lying proofs at full size, made by test programs that this repository cannot build, and replayed on the released `cp_verify`, beside 47 honest full-size proofs and 5 controls; 15 bad parameter files; 13 small proofs on synthetic weights, from a test program that cheats, which are not in this repository: 2 of true claims and 11 false proofs; 2,720 kept calls (a kept call is a command line and its files, stored with the ending that other programs gave it, and replayed), among them 228 calls, at small size, on altered stored tables; [planning calls](glossary.md#planning-call) at 15 configurations, of which one was read against these pages' figures (12 of the 15 are in this repository), and three more planning calls told here; and 21 made-up settings, each a set of files in which one file is changed | All four stored examples accepted (two with `YES`, two with `WITHIN LIMITS`), and all 20 altered sets rejected. All 47 honest proofs accepted. All 57 lying proofs rejected, and all 5 controls (4 `NO: reply-differs`, 1 `NO: model-statement`). All 15 bad parameter files refused (10 `NO: parameter-range (policy)`, 4 `NO: parameter-group-shape (policy)`, 1 `NO: parameter-table-shape (policy)`). Of the 13 small proofs the 2 of true claims were accepted (`YES`) and the 11 false proofs refused (`NO: proof-refused`), as expected. Of the 2,720 kept calls 2,672 ended as recorded, in exit code and last line, the 228 among them, which `cp_verify` accepted. Each of the 228 is on an altered table that, by separate programs' counts of the changed columns, lies far inside the distance at which it decodes to the honest weights alone, so that the accepted request and reply are the honest run's own. That rests on those counts: no decoder was run, nothing of it is machine-checked, and BLAKE2b-256 is taken as collision-free. The counts measure from the table of the run with nothing altered; that this table holds the honest codewords was checked at all positions under one of the four test seeds only, under which 24 of the 57 sets of files behind the 228 stand (these 57 are other files than the 57 lying proofs). In 72 others of these calls `cp_verify` passed a check of one part on a key file or leaf file whose own table was altered; a passed check of one part is not an acceptance (row SO-15.7). The other 48 read parameter files that the released programs refuse, and end with `NO: parameter-range (policy)`: they show nothing beyond that refusal. No call was more accepting than recorded. All 21 made-up settings were refused, the one with `answers` 2 with `NO: parameter-range (policy)`. The planning calls at the 15 configurations all ended with exit code 4 (`PLAN ONLY`): a planning call judges no proof. The three others told here ended with exit codes 1, 1 and 2, as written down beforehand; on which files they were made these pages do not tell. The one that was read against these pages printed the figures that they give as its own. |
| tests of weight hiding | The first and the fifth of the five tests (row LIM-63.2, lines 1 and 5), under exact match, on all three of the released programs, each commitment under a seed that `cp_commit` drew. Both at small size, the first three times and the fifth once. The first also at full size, once on synthetic weights and once on a real model's public weight files. Both audits also under a clipped-gap policy, at small size, on two proofs. One of the two audits, the audit of a commitment, also at full size, in [the timed runs](#the-timed-runs). | Each ended as written down beforehand. The first: all published OOD answers were the row's weighted sum plus the public factor times the row's own pad; no pad came twice, and none was 0. The fifth: both audits (checks, with the provider's secrets, of a commitment and of a kept batch) accepted the honest files, and with a false file in turn for each of three kept files the answer was no each time. The figures: row LIM-63.2, lines 1 and 5. Under the clipped-gap policy both audits accepted the honest files, and `cp_verify` accepted both proofs as within the policy's limits. |
| `examples/maverick/run.sh` | full size, a real model's public weight files; all the stages of `run.sh all`, in their order, on one H100; the build stage was skipped: the programs were built beforehand from these sources; the download stage fetched nothing: the five weight files were laid there beforehand at their checksums; and the toolkit was installed beforehand by that page's typed lines, so building, downloading and installing the tools were not exercised | Exit code 0. The five values in `examples/maverick/expected.txt` came out the same, byte for byte, and the commitment ID was the stored one; `cp_verify` ended with `YES`. By the run's own record, which two readers read apart, it changed no file of the repository. |
| the timed runs | told [below](#the-timed-runs) | told [below](#the-timed-runs) |

The command `lean-verify`, which evaluates the Lean verifier, is not one of the released programs.
It ran over this repository's Lean files. On which files and over which sources it ran is told under
"What these runs do not show", below.

| what ran | size and data | how it ended |
|---|---|---|
| the Lean verifier, `lean-verify` ([lean/THEOREM.md](../lean/THEOREM.md), "What has been run") | full-size proofs, which are inputs made beforehand by other programs; 109 calls, on synthetic weights: 47 honest proofs (42, as reported, made by programs built from other sources; the 2 stored full-size examples; 3 by test programs), 57 lying proofs, and 5 controls, each an honest proof beside a claim that is not its own | All 47 honest proofs accepted, 45 at exact match and 2 under the stored clipped-gap policy, where an accept says less. None of the 57 lying proofs accepted: 54 rejected, and at 3 the command's parser refused the file before Lean was called, which is no verdict and no rejection. All 5 controls rejected. |
| `lean-verify`, 41 calls more | full-size proofs; 14 requests, 7 of them also among those of the 47, at 3 commitments; 38 proofs for one next token, 3 for 1,001 tokens | 41 proofs, as reported of the timed runs, all on synthetic weights and none of them among the 42 of the table, have been put to the Lean verifier, over this repository's Lean files; it accepted each (`lean/CHECK-RESULTS.md`). |
| `lean-verify`, nine calls more | a small stored example beside altered parameter files | Each ended at the parser with no verdict, as written down beforehand. |
| the command of the public test `tools.lean_verify.full_size`, run once apart from the test runner, beside the 109 calls | the proof of the stored example `examples/full-size` (on synthetic weights), which is one of the 2 stored examples among the 47; over this repository's Lean files | Exit code 0 and "VERDICT: ACCEPT (exact match)" (`lean/CHECK-RESULTS.md`). Through the runner that row has never reached Lean in a run that these pages tell; whether it can end in an accept there, under the runner's own HOME and confinement, is untested ([SUITE.md](../SUITE.md) names the obstacles). |
| `cp_commit`, `cp_prove`, `cp_verify` and `lean-verify`, one behind the other | one commitment, under a seed that `cp_commit` drew, and one proof at full size (the second call of `cp_prove`; the first was refused for a fault in the testers' command), for 1,001 tokens, on synthetic weights, made, as reported, by the `cp_commit` and `cp_prove` that this repository's sources build; judged over this repository's Lean files | `cp_verify` ended with `YES`, and the Lean verifier accepted the proof: exit code 0 and "VERDICT: ACCEPT (exact match)", as written down beforehand. It is one proof of one request: it says nothing of other requests, of lying proofs or of a real model's weights. |

<a id="the-public-tests-in-full"></a>*The public tests, in full.* This part gives in full what the
first table's row on the public tests gives in short.

*The four test runs.* For 172 of the 173 rows, on programs that each test run built from these
sources: 169 passed, one passed in part (`tools.confine`: of its check "a device node" the part on
the character device 0:0 was not run on that machine), two were not run (`verifier.clang_build`: no
`clang++` on the machine; `tools.lean_verify.full_size`, the one row that would give a proof to a
Lean verifier: the Lean library was not built), none failed; all on synthetic weights, in four test
runs, which ran the tests by name. The 173rd row, `verifier.rejected_proofs`, ran in a test run of
its own on a verifier built from these sources, on a CPU machine, and passed. Of the 172 rows nine
start `cp_commit` or `cp_prove` in the production build and 20 others start the production
`cp_verify` alone; the rest run test builds, copies, test programs or scripts, only build, or only
read files. The four test runs ran in a tree that holds files that this repository does not hold,
and that differs from it in some tests and scripts (described under "What these runs do not show",
at "The public tests").

*The rows that need a GPU or the CUDA compiler.* Nearly every public row that needs a GPU or the
CUDA compiler compiles the provers' sources, and so reads files that differ as "Which programs ran"
describes; beyond that: `gpu.forest_lookup_test` and `build.forest_lookup_test` (their test, in
code), `gpu.coins_own_gpu_test` and `build.coins_own_gpu_test` (one printed message of their test),
`gpu.quick_start` (the page that it reads), and `gpu.serving_first_run` and
`gpu.coin_contexts.services.not_full` (they run `driver/spool.py`, in which comments and docstrings
alone differ); each of them ran, in the four test runs, under a runner that differs as described
under "What these runs do not show", at "The public tests". The tests' own files that differ in
comments, names and printed texts alone are described there too. `build.forest_lookup_test` and
`gpu.forest_lookup_test` have been run once on this repository's own files, as far as their files
were compared, by their rows' own commands, apart from the four test runs, and passed; from which
build the checking program of `gpu.forest_lookup_test` came is not shown. No other row that needs a
GPU or the CUDA compiler has been run on this repository's own files. The builds for the comparison
described under [What a reader can check](#what-a-reader-can-check) were made apart from these rows.

*This repository's files alone.* Apart from the four test runs, most of the rows to which
[SUITE.md](../SUITE.md) gives the kind `cpu` or `long` (107 of the 173: the rows that themselves
need neither a GPU nor the CUDA compiler) were run on a machine with no GPU, each row confined by
the runner. On this repository's files alone, without `lean/`, the runner was asked in one round for
all 105 rows of the kind `cpu`: 92 were run to their end, and 90 of them passed. That round ran on
files that differ from this repository's in 47 paths: 38 pages, 4 tables, `paper/paper.pdf`, the
list `tools/lean-verify/native-modules.txt` and 3 licence and notice files. No source, test or
script is among them. The 90 are all the rows of the kind `cpu` except `weight_sums`,
`tools.confine` and the 13 described below; `verifier.clang_build`, which the four test runs could
not run, is one of them. `weight_sums` and `tools.confine` passed in part. Of `weight_sums` 13 steps
were skipped without two Python packages; run with both installed it passed, with no step skipped.
Of `tools.confine` one check was not run, "as root, another account's file in /dev/shm", since the
test already ran inside a row that the runner had confined as root. Sixteen rows were also run on
files that differ from this repository's in 30 paths, of the same kinds, and all passed;
`integer_model.cpu` (below) is one of them. Which rows read a path that differs has not been traced.
Neither the round nor these sixteen ran on this repository's files as they are.

*Rows left out or not run.* On this repository's files alone the runner left out 13 rows of the kind
`cpu` whose name begins `verifier.` (those that need `setup.target.other_parameters_worlds` or its
`.work`): each needs a set-up row that needs the CUDA compiler and a GPU. The rows that were run to
their end and these 13 make up all 105 rows of the kind `cpu`. Of the two rows of the kind `long`,
`integer_model.cpu` passed on this repository's files alone, in the environment that
[SUITE.md](../SUITE.md) names for it, and `tools.lean_verify.full_size` was not run: that tree held
no `lean/`. Its command, run once apart from the test runner, is described above, with the calls of
the Lean verifier.

*A row that can fail on a busy machine.* `driver.measure_rate` has failed twice, once on this
repository's files alone and once in another tree with the same `driver/`, each time on a machine
under load: one of its timing tests found no sample from its stand-in for `nvidia-smi`. Run again it
passed both times, and in the round described above it passed at once. That the load is the cause is
not shown.

*What no row runs.* No row runs `tools/accepts/accepts.sh`, the real
`tools/lean-verify/LeanVerifyWalk.lean` (`tools.lean_verify.script` runs a stand-in for it),
`integrations/vllm/integer_model/tests/lean_vectors/run.py`, or `make_vectors.py` and
`CompareVectors.lean` beside it, whose runs the integer-model package's pages describe.

*What these runs do not show.*

- *The public tests.* Most rows run test programs or scripts, or only build, and start none of the
  three released programs. By a reading of the commands of the 172 rows, not a trace, nine rows
  start `cp_commit` or `cp_prove` in the production build, and 32 others start `cp_verify`: ten of
  these a copy of it that a test build's folder holds, and two a sanitizer build of it. Of the
  public tests eight rows run `cp_commit` or `cp_prove` as test builds of these sources. Those are
  not the released programs, and such rows are told as "on test builds". Two of the eight are also
  among the nine. Thirteen further rows read or judge files that two of these eight made. In each
  GPU test run the two digests were found among the records of its production and verifier builds;
  test builds are tied to no digest. One production build is tied to none: `gpu.quick_start` builds
  its own programs in a copy of the tree. Each test run ran named rows (`--only`), so the commands
  that [SUITE.md](../SUITE.md) prints were not used. The tree that the rows ran in holds 473 files
  that this repository does not hold, nearly all of them tests, and no `lean/`. Of this repository's
  1,218 files outside `lean/`, `paper/paper.pdf` apart, 880 stand the same, byte for byte, in that
  tree. The others are 97 pages and tables, three of them tables that tests read; nine files that
  stand at another path; 129 files of the provers' sources, which differ as "Which programs ran"
  tells; 12 files of the provers' tests, in comments, names and printed texts as reported, and one
  of them, `prover/tests/forest_lookup_test.cu`, in code; 52 other tests; the runner,
  `run_suite.py`, in a comment; eight files of the serving scripts and their tests; one file of
  `tools/lean-verify`, in which a comment alone differs; one list,
  `tools/lean-verify/native-modules.txt`, which names the Lean modules that the command runs
  compiled: it was drawn anew; of the rows that were run none takes a module's name from it
  (`tools.lean_verify.script` lays a stand-in list of its own); one script each of `tools/accepts`
  and `examples/rejected-proofs`; eight tests and four sources of the integer-model package; four
  files under `docs/format/` and `tools/lean-verify/`, in comments, docstrings and two printed
  headings; three licence and notice files, one of which that tree lacks; and seven others: three
  `figures.txt`, `examples/maverick/run.sh`, `examples/quickstart-gpu.sh`,
  `scales/make_calibration_set.py` and `toy/mask_rules.py`. By a comparison of tokens, of the 52
  other tests 35 differ in comments and docstrings alone, one in a name too, 11 in string literals
  too, two are text or data, and three differ in code (`integrations/tests/test_ask_and_check.py`,
  `test_end_to_end.py` and `test_spool.py` beside it); of the eight files of the serving scripts and
  their tests two differ in comments and docstrings alone and six in code
  (`integrations/proof_middleware.py` and five tests or helpers; the runs of the middleware's check,
  told under [the timed runs](#the-timed-runs), ran another text of that script, and in what it
  differs is told there); the scripts of `tools/accepts` and `examples/rejected-proofs` differ in
  code; of the integer-model package's files and the seven others none differs in code. A string
  literal is not always only printed: some are values that a test compares, or lines that are
  written into a file. Where a test or a script differs in code, what passed in those test runs is
  the text that ran there, not this repository's. `prover/tests/forest_lookup_test.cu` is the test
  of the rows `build.forest_lookup_test` and `gpu.forest_lookup_test`. Apart from those test runs,
  `tools/accepts`, this repository's script and sheet by their checksums, ran at small size alone
  over this repository's Lean files: all 4 calls of Lean ended as expected, and all 17 controls were
  refused. It runs single parts' checks, not the Lean verifier. A row that reads pages read that
  tree's. A row that reads the stored examples read them as they lie in this repository. Three rows
  ran a second time in one test run, and passed both times. Two rows passed with some of their cases
  skipped, as their entries in [SUITE.md](../SUITE.md) allow (`integrations.unit`,
  `integer_model.cpu`). The two rows `setup.target.other_parameters_worlds` and
  `setup.target.other_parameters_worlds.work` made 17 sets of files under other parameters (the
  tests call them worlds); the set-up leaves 3 out. Of each row its verdict line was read, not all
  that it printed. The row `gpu.frame_readings`, one of the 169, compares the prover's and the
  verifier's readings of the public files. In the four test runs it ran over more folders than
  [SUITE.md](../SUITE.md) names; over the eight that it names it passed, at small size, in a test
  run of its own, on another machine, on the tree that the rows ran in. At full size, on four
  folders (two different configurations), on synthetic weights, it ran in a test run of its own, in
  a form that [SUITE.md](../SUITE.md) does not have, on one H100, and ended as expected. What failed
  in other runs of these tests: row T45.
- *`cp_verify`.* The kept calls say nothing of a byte that is hashed on a path that none of them
  takes: a change there would not show. At the kept refusals the replay is blind: those calls end
  just as a changed digest would end them. Of the runs on the 57 files, each run's exit status, the
  number of its lines on standard output and its last two lines, cut at 120 characters, were kept; a
  refused run prints no build word. Twenty-two of the 57 lying proofs were tried once each, and all
  of the twenty-two claim the true reply. Of the 57, 24 claim a false reply and 33 the true reply
  with a lie inside, as reported. All 57 lying proofs stand at one commitment and at exact match,
  for three requests. The stored files were checked, not made, and which programs made them is not
  traced. The altered sets are honest files with a few bytes changed, not the work of a cheating
  prover. How narrow the tests against lying proofs are, and that the lies of those twenty-two were
  not aimed at the lookup checks: [guide/known-limits.md](../guide/known-limits.md). Of the other
  35, at least three, by their makers' labels (the 24 made for false replies have records that name
  no kind), aimed at a lookup (a key in no row; a count at another row) or at a digit's range; these
  three were refused at the check that the proof's masked message fits the linear claim, before any
  check of a lookup or a range was reached, so no refusal at the lookup checks themselves has been
  seen (as reported; row T1.1).
- *Tests of weight hiding.* Neither test shows that the masks hide (each test's bounds: row
  LIM-63.2, lines 1 and 5). No run or comparison told in this section shows what the programs leave
  in host or GPU memory (rows H39.1 and HW-748). On the released programs, of the audits with the
  provider's secrets, the audit of a commitment alone ran at full size: in [the timed
  runs](#the-timed-runs), behind each of four honest commitments, one of them under a clipped-gap
  policy: by the runs' own account it accepted all four. At full size the fifth's two audits have
  run together only on other programs (below). An audit's yes to honest files is no negative test.
  At small size the false files stood for kept files: no altered public file was put to the audits.
  With the provider's secrets no audit has run on a commitment to a real model's weight files: by a
  reading of the code, these programs do not let it run there (row HW-747). Under the clipped-gap
  policy the audits at small size show that the masks are there as derived: that is no search for a
  leak; in that run the batch's two requests have one and the same reply, and no false proof was put
  to `cp_verify` (row H13.8). Of the five tests behind the judgement on weight hiding these two ran
  on the released programs: [on which programs it was made](#on-which-programs). The other three,
  and most of the readings behind that judgement, were made on other programs, so it reaches the
  released programs on less evidence, and it does not reach code that differs from the released
  programs'.
- *`examples/maverick/run.sh`.* One run, of the one example request of
  [examples/maverick/README.md](../examples/maverick/README.md), on one card, with no batch. It
  neither built the programs nor downloaded the weights. The commitment is the published one, made
  with `-public-weights` and this repository's public seed: nothing under a secret seed was tried. A
  program typed that page's lines, and a reader's machine may differ in CPUs, memory and disk. The
  script that ran differs from this repository's in comments, in its usage text and in some
  messages, and in nothing else.
- *The Lean verifier.* These calls were made over this repository's Lean files. This part speaks of
  the 109 calls, the 41, the nine and the run of the public test's command; the call on the one
  full-size proof of the second table's last row is described at its end. The command and the parser
  were held by checksums before and after the calls; every file of the command and of the parser
  outside `lean/` that the traced calls beside them opened has this repository's checksum (none of
  the run's own calls was traced: `lean/GAPS.md`, 14, "Which programs (k)"). The outcomes were known
  beforehand, from other calls, over other Lean files, and those written down beforehand were these.
  The program `cp_verify` was not called in these 109 calls: the script that made the calls calls
  the command alone, and traces of one small call and of one full-size call, neither of them among
  the 109, show no start of that program. Its source file is compiled into the parser's unit. The
  same 109 sets of files were also put to the released `cp_verify`, as reported: it accepted the 47
  honest proofs and rejected the 57 lying proofs and the 5 controls. The sets were paired
  afterwards, by the label, the policy and the first 16 hex digits of the proof file's checksum; the
  other files of a set were not paired. The 42 of the table are, as reported, 41 of the 363 at exact
  match and the one under a clipped-gap policy that [What is proved](../guide/what-is-proved.md),
  "In runs", tells. They were taken by rules fixed beforehand and have little variety: as reported,
  the 42 are of 38 requests and 2 shapes (38 proofs for one next token, 4 for 1,001 tokens), at 4
  commitments. All full-size proofs of these calls are at one model size: each set's three
  configuration files are those of one of the two stored full-size examples, which part in three
  lines of the parameter file alone. Who made a file, of the 109 or of the 41, on which weights, by
  which rules it was taken, and that it is honest or lying, is as reported: it rests only on the
  word of those who made and chose them. None of the 57 lying proofs and none of the 5 controls was
  put to the command under the clipped-gap policy. All calls ran on one machine, beside other work.
  By the command's text the natively compiled part and the parser's unit are each compiled by one
  compiler; what these two compilers print as their versions was read off that machine after the
  calls. That the command and the parser read the files that the traced calls opened, and no others,
  is not shown. At the calls of the first two rows of the second table that reached Lean the command
  loaded the compiled files of this repository's Lean library. By the word of those who made them
  and of those who keep them, a build of these Lean files made them from nothing, in several calls
  of Lake; that they were compiled from these files is Lake's own finding, not the run's. The
  checksums of the library's own compiled files, taken before and after the calls of the first three
  rows of the second table, are the same. The compiled files of the packages that the library
  imports, which that build did not make, were not held by checksum at the calls. These pages tell
  no comparison of two builds of these Lean files. The natively compiled part the command compiled
  itself, in a call of its own before the 109, but for three objects of a package that lay in the
  tree before that call. The natively compiled objects' checksums, taken before and after the calls,
  are the same; each call links the library anew from them, and that library's checksum was printed
  only at the call that compiled them. In full: [lean/GAPS.md](../lean/GAPS.md), 14. `lean-verify`
  evaluates the Lean verifier, natively compiled in part: that is a run, not a proof. Lean reads
  each file's parsed form, made by the command's parser, which no theorem covers. The parser's
  compiled code had the same checksums at each of the 109 calls that reached it and at a control
  compiled apart; at the run of the public test's command (the table's fourth row), at another path,
  its checksum is another, and the same once names are struck. The call on that proof was held less
  closely. As reported, the compiled Lean files of its folder had the checksum of the build
  described above before and after that folder's first call, which was another call; no checksum was
  taken after this one. That the Lean files, the command and the parser of that folder are this
  repository's rests on the testers' word alone. It ran on another machine.

<a id="the-timed-runs"></a>*The timed runs.* These are the runs behind the figures on
[guide/performance.md](../guide/performance.md) that carry no mark, made on the released programs.
They were made within three hours of one day, on H100 GPUs: six kinds of run at full size, made 3
times each, each kind's three on three machines (at the default setting: on a real model's public
weight files (1) a commitment with one request, and services (2) at 24, (3) at 64, and (4) at 8 and
1 requests a batch; on synthetic weights (5) a commitment, its audit and services at 24; (6) at that
setting with a [clipped-gap](glossary.md#clipped-gap) policy: on the public weight files, sessions
with every reply token checked and like sessions that check none). Each of the nine machines that
served on the public files at the default setting made its own commitment first; three of the nine
commitments were named beforehand as the runs behind the figures. The audit in these runs is a check
of a commitment with the provider's secrets. One run each was made of the making of an example under
a [clipped-gap](glossary.md#clipped-gap) policy, with an audit of its commitment, and of one service
and one call at 64 on synthetic weights, both on one commitment made for them on that machine, which
was not audited; those last two failed.

On synthetic weights every proof was accepted, and every altered copy and call refused. On the
public weight files, in the four kinds of run at the default setting, all 1,851 requests were
served, and all 1,851 proofs were accepted by the runs' own calls of `cp_verify` behind the timed
stretches (1,845 in the runs behind the figures; six more, one on each of six further commitments);
in the three runs of the services at 24 requests a batch, the only runs on these weights with that
test, all 249 altered copies and calls were refused. Each of these 1,851 proofs is of the one token
after the prompt, under exact match. So are the 720 proofs of the services on synthetic weights;
four more there, one on each of four commitments, cover 1,001 positions each: 1,000 tokens of the
window are checked and the token behind it is stated; so does the clipped-gap example's one proof,
under that policy. In the sixth kind of run all 720 requests were served and all 720 proofs accepted
by the runs' own calls of `cp_verify` behind the timed stretches: the 360 of the sessions with whole
replies checked check every reply token of their request, and passed within the clipped-gap policy's
limits, not by exact match; their control's 360 check the last position alone, and passed by exact
match. The requests of both were 48 stored windows of 1,024 tokens, some sent more than once. The
requests were, on the public weight files, in the four kinds of run at the default setting, by each
run's own line, a set of 96 windows of 1,024 tokens, and at 24 and 64 a batch some were sent more
than once; on synthetic weights 96 windows of made-up tokens (row RES-6). The altered copies are
honest files with a few bytes changed, not the work of a cheating prover. The audits that were part
of these runs are told under the tests of weight hiding. Whether a difference from a figure of
another run comes from the programs is not shown: such runs were on other machines. The scripts
`driver/spool.py` and `integrations/spool.py` that ran differ from this repository's in comments and
docstrings alone. The check by the vLLM integration's middleware also ran in runs of its own, in
none of the timed runs, with the released `cp_verify`, on stored full-size proofs on synthetic
weights, as reported, called directly, with no inference engine and no HTTP server around it. By
itself, on a batch of 24, all 24 were released, which is no negative test: no false proof was put to
it. Beside a serving `cp_prove`, in one run, it read stored proofs, not that prover's own (row
RES-32). By the checksums that these runs' records print, the scripts that ran differ from this
repository's in comments, docstrings and the texts of messages of `integrations/proof_middleware.py`
alone; for one run the constant `CHECK_THREADS` was also set to 4 (row RES-32).

*What came out badly, or otherwise than expected.*

- On the released programs: 48 of the 2,720 kept calls of `cp_verify` end otherwise than recorded,
  all with exit code 1 and `NO: parameter-range (policy)`: they read parameter files that the
  released programs refuse.
- In the timed runs:
  - Two runs at 64 requests failed, both on synthetic weights: a service of five batches of 64 in
    its first batch, and one call of the prover on 64 requests at once. None of their 320 and 64
    requests got a proof, the GPU's memory was nearly full in both (81,053 of 81,559 MiB at the
    largest sample, which is a reading of the samples, not a line of the program's), and no cause is
    shown; one run each, so no failure rate comes of it. No batch of 64 has been proved on synthetic
    weights: every attempt on record failed, eight in all, six of them on other programs. On the
    public weight files such services ended well (rows T50 and RES-11).
  - On the public weight files the services at 64 requests a batch came to their end with 930 MiB of
    the card's 81,559 MiB free at the fullest (80,629 MiB in use); in like runs on other programs:
    866 MiB (row RES-11).
  - With the start-up counted, one of the three sessions with whole replies checked, at 24 requests
    a batch under a clipped-gap policy, stays under 100 tokens/second (99.33; the others 103.94 and
    103.79), and so does one of the three sessions of their control (98.12; 102.47 and 102.54): both
    on one machine, the slower in both sets. No cause is shown. Both ran at the default setting but
    for the three numbers of the clipped-gap policy in the parameter file, on stored replies with no
    inference engine; the rates with whole replies checked, accepted under that weaker policy,
    cannot be set against the project's target of 100 tokens/second or the rates at exact match (row
    RES-12).
- On other programs no run reached 100 tokens/second with the start-up counted, at the default
  setting (row RES-33), two commitments took far longer than their like (row RES-9), and one proof
  was longer than any of the timed runs', with no cause shown (row RES-6).
- In the timed runs, and in one set of three runs on other programs, the audit with the provider's
  secrets took longer in each run than in any of another set of three on other programs, on other
  machines: a small shift upward is neither shown nor ruled out, and no cause is named (the times in
  the timed runs, and the range of the others: row RES-15).
- On other programs: some tests notice little of a fault in `HasherT`, the class that feeds and
  finishes BLAKE2b (its source: [docs/sources-notes.tsv](sources-notes.tsv); what that costs the
  self-checks: row E2.4). Faults were planted in a like class. Of three, 12, 10 and 12 of 50 CPU
  rows noticed each in turn. Of three more, planted for the GPU's code alone, the comparison of host
  and GPU noticed all three, three tests noticed one, and two tests noticed none.
- On other programs, in other runs of these tests: a GPU test that is not in this repository failed
  on a wrong count in the test itself, with all its cases agreeing, and a runner marked two such
  tests as failed, against wrong counts in its own table (row H1.4); one CPU row,
  `integrations.unit`, failed for a cause in the test machine, and passed with that cause removed.
- Among the public tests: `driver.measure_rate` has failed on a busy machine, and `weight_sums` and
  `tools.confine` passed in part ([above](#the-public-tests-in-full)).

<a id="on-other-programs"></a>*What these pages tell of other programs.* *Other programs* are
programs that are not the released ones: built from sources that differ from this repository's in
more than "Which programs ran" tells, and are not part of it, or from these sources with other
options (test builds). What a run found on them does not carry over to the released programs; what a
reading found carries over only as far as the differences were read (row HW-679). A fault found on
them may be in the released programs too. The kinds of result below were got on other programs. Most
bear the mark "not on the released programs" (in the guides †) or, where a fault, a test switch or a
test seed is built in, "on test builds"; a run told with no mark, and named neither in the tables,
nor in the part [The public tests, in full](#the-public-tests-in-full), nor among the timed runs, is
to be taken as one of them, but for the test of row RES-39, the runs of the middleware's check and
the four runs told under "Which programs ran". A leak of weights found there is to be taken as
holding of the released programs unless a page says otherwise. The calls of the Lean verifier, and
any run of `tools/accepts`, that this section tells outside its tables are told apart: over which
Lean files each ran stands at its sentence or at the head of the Lean verifier's bullet.

- Runs with cheating or faulty programs, and searches for attacks. No cheating prover was built or
  run against the released programs. The 57 lying proofs of the table are files that test programs
  made beforehand; here `cp_verify` alone ran on them. This repository cannot build such a prover:
  the code is not in it. It does hold the functions that such a prover would use, which, by a
  reading, do nothing here (that their calls leave no instruction in the host code is not shown),
  and test code that can inject a fault in every build but the production one (row E29.1). No public
  test builds or runs a prover under a cheating switch; some run a prover that argues for something
  false, with code of the test's own. `verifier.forest_twin_test` has no cheating prover. Eleven of
  the twelve stored false files of `examples.protocol_toy_false_files` were made by builds that this
  repository cannot make.
- Tests of weight hiding, but for the table's: the second and the third of the five, on test builds
  of other sources, which hold the code that injects a fault; the fourth, on programs of other
  sources that hold test-only code; the fifth's two audits together, at full size; and the audits by
  value that stand beside the five. What each found, and what went unnoticed: row LIM-63.2 and [What
  it rests on: five tests](#verdict-runs).
- Other replays of altered, crossed or spliced honest files (what they found: row SO-15.7).
- Nine planning calls at small size, eight on parameter files made up for the call and one on the
  unchanged files of a small example of this repository: four were admitted, three refused, and two
  aborted at an internal check; none read a key, a leaf or a proof (row SO-6.3.2). The released
  programs refuse a file with `answers` other than 1, with `NO: parameter-range (policy)` (read in
  the code; run at 2).
- Every figure marked †. A † time, rate or memory peak is an estimate at most.
- Every run of the Lean verifier that these pages tell and this section does not, and the runs of
  `cp_verify` side by side with a stand-alone program built from
  [`verify-on-bytes`](glossary.md#verify-on-bytes) (row L14), which this repository does not build
  as one.
- The vLLM integration's runs, but for the check told under [the timed runs](#the-timed-runs)
  ([integrations/LIMITS.md](../integrations/LIMITS.md)).
- Most readings of the code: a reading is of other sources, unless its line says otherwise ([Kinds
  of evidence](#how-a-thing-is-known)).

Which refusals no test in this repository shows:
[SUITE.md](../SUITE.md#refusals-of-altered-malformed-and-extreme-files). Which faults of hiding no
test in this repository shows: [above](#what-a-reader-cannot-repeat-on-hiding).

<a id="what-a-reader-can-check"></a>*What a reader can check.* `cp_verify` prints a build word
(`build=`), the SHA-256 of a list of checksums of at least every file of this repository that it
includes. A reader can compute this repository's word from the sources, and build a `cp_verify` that
prints it; the word stands in [examples/README.md](../examples/README.md). It covers sources alone,
tells neither the compiler nor the flags, and does not cover the rest of the provers' sources, and a
printed word shows only what a program says of itself.

*This repository's sources and the programs of the timed runs.* The sources under the provers'
digest were compared, token by token: but for the comments, the names, the seven quoted texts and
the wording of eight messages nothing differs. The built programs were compared in two steps, by way
of a middle build, production and test, of sources that differ from this repository's in the wording
of those messages and in one line of comment alone. What either step leaves out, the tie leaves out;
what is told below as not compared, not held or not covered holds of both. First step: the middle
build was compared, by tools made for it, over objdump, readelf and cuobjdump, with the programs of
the timed runs and with one test build of the sources of those runs, without being run. The programs
that the test runs built for themselves were not in the comparison. `cp_verify`, `weight_sums` and
`make_model` came out the same, byte for byte. The GPU code of `cp_commit` and `cp_prove` is the
same, byte for byte: every function and its constant banks; every line of PTX too, but for the names
that differ. In their host code every instruction of every function is the same, at the same place
and of the same length, and each address in it points at the same thing. About half of the functions
differ in raw bytes (by a count that is no part of the comparison; its program was read a second
time and run anew, and it noticed a planted byte), in address fields alone, since names of other
lengths move the data behind them. The data that the programs read (constants, tables, texts) is the
same, but for the names that differ, where a program holds them (a kernel's or a type's name), and
for texts that hold one: some printed texts begin with a routine's name, and follow it, and one
names a source file. The files of the two programs have other checksums. Not compared: the note that
holds the build's identifier, `.gnu.hash`, `.eh_frame_hdr`, and the relocation tables beyond their
targets. Not held: up to 31 zeros of padding behind a string or table, the program and section
headers, and the entry point. Of the files whose comments differ, four are compiled into nothing:
two build scripts, a settings file and a header that nothing includes; of the seven quoted texts
three stand in no program: two of `prover/common/hash_tree_ref.h`, and one that only the compiler
prints. Code behind a switch that is off in both builds is not covered. Bytes were planted in the
production `cp_commit` and `cp_prove`, and the comparison noticed each; their test builds were
compared by the same tools with no plant. A second reader derived every figure of the comparison's
result anew, with a program of its own. Second step: a production and a test build of this
repository's sources were compared in the same way with the middle build, without being run.
`cp_verify`, `weight_sums` and `make_model` came out the same, byte for byte. The GPU code of
`cp_commit` and `cp_prove` is the same, and the container of the GPU code is the same, byte for
byte. In their host code every instruction of every function is the same, and each address in it
points at the same thing. About three functions in ten differ in raw bytes, in address fields alone,
since texts of other lengths move the data behind them. Of the data that the programs read, the
quoted texts of those messages alone differ: eight texts, of the seven messages that the programs
print, one of which is two texts. The eighth message stands in a build script, in no program. As far
as a reading of these sources goes, each is only printed, and none is a key, a label that goes into
a hash or a part of anything hashed. One of them is the name under which a refusal is printed, and
the test that looks for that name differs in the same way. Two test files of this repository differ
too at this step: `prover/hash_commitment/tests/hf_params_test.cpp`, in texts that follow the
messages, and `prover/tests/forest_lookup_test.cu`, in code; they lie outside the provers' digest
and are in no compared program. At this step, by a look beside the comparison, the entry point and
the program headers are the same in all four pairs of programs (`cp_commit` and `cp_prove`,
production and test), but for two headers of each `cp_prove`, which differ by 32 bytes, the
difference in length of its read-only data. Not held at this step: where inside the container of the
GPU code a pointer of the host code points. Faults were planted at this step too, 69 in all, and
each was noticed: 60 on the host and GPU sides of the production `cp_commit` and `cp_prove`, 8 at
the check of the texts that differ (2 for each of the four programs; none of these turns a byte of
code), and one at the count of raw bytes. At 2 controls with nothing planted the tools stayed
silent. The host and GPU code of the test builds was compared by the same tools with no plant. The
result and the tools were each read a second time, by readers looking for a fault, with nothing run:
they found no way for the comparison to pass wrongly, beyond what is told here as not compared or
not held. Neither comparison ran anything. No reader of either comparison is an outside party. Apart
from the comparison, under one test seed the test builds of both sides of the first step gave the
same files, byte for byte, at small and at full size, on synthetic weights, and so did both
production builds of `cp_commit`, called with `-public-weights`, at small size. At full size there
was no control and no second run of one side; at small size each of six controls came out different,
as a control must. Proofs of the production builds are drawn, so nothing is claimed of them by
bytes. Equal bytes under one seed show that the two sides computed the same on those inputs; they
are no test of weight hiding. The like run on both sides of the second step is the comparison with
the middle build that "Which programs ran" tells: it was at small size alone; two runs of one side
gave equal bytes, and each of seven controls came out different.

*Programs outside the comparison.* Both steps cover `cp_commit` and `cp_prove`, production and test
builds, `cp_verify`, `weight_sums` and `make_model`, and no other program: not the two helper
programs that the audit tool (`tools/audit/`) builds for its own use, the programs of `blocks/`, the
tests' own programs or the parser of `lean-verify`. The audits with the provider's secrets are runs
of the audit tool. Its 14 scripts and the 2 sources of its helpers lie under neither digest. These
16 files, and the 118 sources that the helpers compile beside them (a list that a program drew from
the `#include` lines, not a compiler's), have the same checksums in this repository as in the tree
that the rows ran in; from which tree the audits' own copy of the tool came is not shown. One file
that the tool imports from outside its folder, `docs/format/format_rules.py`, differs from that
tree's: by a comparison of tokens, in docstrings, a comment and two string literals, both headings
that it prints into a file.

*What a reader cannot check.* The files of the programs that ran and the runs' records, the provers'
build records among them, are not in this repository, so a reader cannot check these results against
them.

*Two builds of the sources of the runs.* Two builds of those sources were compared file by file,
without running them: each, by the account of those who built them, from a fresh copy with nothing
kept from another build, on one machine, at one path, with the same compilers. `cp_verify`,
`weight_sums` and `make_model` came out the same, byte for byte. `cp_commit` and `cp_prove`,
production and test builds alike, differ only in the name of a temporary file of the compiler,
which, by a reading, holds a number that differs from build to build and stays in the program's list
of symbols: a few bytes (6 and 11 in this pair; 5 and 12 in the test builds), at the same sizes. All
their GPU code is the same, byte for byte. So the files of a second build of these two have another
checksum. `cp_verify`, whose sources do not differ from this repository's, was also built on a
second machine (Ubuntu 24.04, g++ 13.3.0), with the repository's own script: the same file, byte for
byte. A third build of those sources, in the production build, in a fresh container on a GPU node,
at the same path, with nvcc 13.2.86 installed at another place, was compared with the programs of
the timed runs: `cp_verify` came out the same, byte for byte; in `cp_commit` and `cp_prove` no
function differs, on the host or on the GPU, by a comparison function by function, but other parts
of the files do. Each of the two differs from its counterpart in the same eleven sections of the
host file, among them the container of the GPU code and the section that says which libraries to
load and where to look for them, and in one note inside that container; none of these differences
has been traced to its cause. That pair does not show what another build path does. Whether it was
made on another machine than the programs of the timed runs, and which compilers made those, is not
in the records that were compared. No test build was in that comparison. That comparison ran none of
the three programs. It noticed a byte planted in a host function and bytes planted in the GPU
images; the unpacking itself is not tested, and nothing was planted in the PTX.

*How other sources differ.* How far the sources of other programs differ from these is told with
each kind (row RES-39; the head of this page).

*Test code in the production build.* That a production build holds no test code is checked by a
search for strings, which does not read code (row E17.2); at least 7 of the 93 lines of
`prover/tests/serving_build_words.txt`, which it reads, stand nowhere in the sources of `prover/`,
`blocks/` and `verifier/` outside their tests, so their absence shows nothing; that the search ran
is shown by a build's records, not by a log. Some code that no run can reach is in these sources
(rows E29 and E29.1). The readings behind that were not independent ([the reviews](#review)).

*Libraries and counts.* Each run used the CUDA libraries, the driver and the C and C++ libraries of
its machine, and says nothing of others. Counts of lines, files, kernels and functions on these
pages may be off: no such count has been made on these sources, but for those that a row tells of
the released programs (rows RES-39, RES-54, H39.1 to H39.4.1, HW-213 and HW-218).

*Weight hiding and soundness.* What is proved, of what, and what is only tested is told on
[guide/what-is-proved.md](../guide/what-is-proved.md) and
[guide/weight-hiding.md](../guide/weight-hiding.md). Nothing in this section is a proof about the
compiled programs.

<a id="few-runs"></a><a id="attacked-at-small-sizes-only"></a>
### Few kinds of test have run at the default setting, and many ran only once

No test in the repository's test table uses a real model's weights. The project itself checked every
accepted proof described in these docs, on its own machines, using the prover's files, commitment ID
and `cp_verify`. That is checking our own work. A passing test shows only that the programs it names
gave the expected answers, or rejected what they should have. It shows nothing more about soundness
or hiding.

**At the default setting.** Few kinds of test have run there; what has run there on hiding is told
under [the five tests](#verdict-runs). In the timed runs, at full size, three runs on each kind of
weights gave honest proofs and altered copies of their own files. On synthetic weights: 723 of 723
proofs, each accepted by a separate run of the verifier (241 of 241 in each run); 240 of 240 altered
copies rejected (80 of 80 in each); 9 of 9 altered calls rejected (3 of 3 in each). On a real public
model's weights: 720 of 720 proofs, each accepted by a separate run of the verifier (240 of 240 in
each run); 240 of 240 altered copies rejected (80 of 80 in each); 9 of 9 altered calls rejected (3
of 3 in each). By a reckoning from the test's own rule and the files' lengths, made for like runs
not on the released programs, none of the 64 altered places in a run's key file and leaf file was
among the bytes of the [OOD answers](glossary.md#record).

All of this is altered copies of honest files. An altered copy of an honest file is not a
deliberately constructed false proof, and an altered proof that is rejected at the first hash check
says little about the tests that come after it. As far as we know, apart from one cheating
committer's false leaf file (a false copy's file) at full size, tried once through a form of the
executable `verify` that is not in this repository and through a C++ verifier built from other
sources, and rejected by both, in `verify` at that file's own check, by report (`lean/GAPS.md`, 14
(i) and 33), no malicious committer has run at the default setting at full size. The only malicious
provers that have are these:

- The test provers behind 57 lying proofs (24 made for a false claim, on one request; 33 carrying a
  planted lie, for the true replies to three requests, that one among them): none was accepted by
  either verifier (the Lean verifier's parser refused 3 outright, and a no-verdict is not a
  rejection). Calls on the 57 lying proofs are among those of the Lean verifier's reference run;
  what is known of the command and the parser there stands in row L9. Whether the command and the
  parser at the Lean verifier's calls told in this bullet outside that run were this repository's
  has not been compared. No proof of a false claim that also carries such a planted lie has been put
  to either verifier. One commitment, synthetic weights, exact match, lies of a few kinds, every
  rejection at one verifier step. The released `cp_verify` refused all 57 ([what was run on these
  programs, and what was not](#what-was-run-on-these-programs)). The test programs that made the
  lying proofs are not in this repository, so those trials cannot be repeated from it
  ([lean/THEOREM.md](../lean/THEOREM.md), "What has been run").
- A cheating prover that alters no value and only goes on past its own test on the policy, under the
  clipped-gap policy. Its one proof of a true claim was accepted and its two proofs of false claims
  were rejected. Neither rejection is shown to be owed to the policy's limits (each names a step of
  the proof file's check). One run, not on the released programs; both verifiers gave the same three
  answers (`lean/GAPS.md`, 14). The Lean files and the parser's sources in this repository differ
  from those of the Lean verifier's calls told in this bullet.

That these proofs lie, and where, is how the runs were set up; a proof file does not show it.

Apart from these, no prover that itself sends a false value, and, but for one cheating committer's
false leaf file (a false copy's file) at full size, tried once through a form of the executable
`verify` that is not in this repository and through a C++ verifier built from other sources, and
rejected by both, in `verify` at that file's own check, by report (`lean/GAPS.md`, 14 (i) and 33),
no committer that adapts its later messages to an earlier false one, has run at full size at the
default setting. Other runs at the default setting: row LIM-47, and rows T49 and T50; 703 calls on
altered files at full size: row SO-15.7. [What was never tried](#what-was-never-tried) lists further
attacks.

For binding, what has been run on the protocol is this, all at exact match, and most of it not on
the released programs. At the default setting, at full size: the published out-of-domain answers and
the commitment IDs of honest commitments, recomputed by value and found equal (also at small size,
and on a real model's public weights), and a real model's commitment ID remade; altered copies of
honest key files, leaf files and proofs, all refused, among them alterations inside the
out-of-domain answers, and calls under a false commitment ID; hostile copies of the stored full-size
example, all refused; and one comparison of the default's parameter file with the technical report's
setting by a planning call, which checks no proof. At the small example's setting: the 23 lies of a
lying committer's build, refused in every whole call; edited committers' tables altered within the
distance that the verifier allows, whose proofs were accepted as the bound allows, each held true by
a count of how far its table was altered (a reading; no decoder was run), and other altered tables
refused, some far inside that distance: there is no threshold; among these, three further commitment
IDs under which one leaf table's proofs were accepted, by design (that such tables decode to the
same weights is an argument on paper, not tried); eleven cases with a stored table that is another
matrix's, in whole or in one row, or close to it, all refused, and four false replies under one of
them, refused at the leaf file's check; and an illustrative committer written in Lean, which agreed
in values with `cp_commit` on one example, which a reader cannot repeat. Apart from one cheating
committer's false leaf file (a false copy's file) at full size, tried once through a form of the
executable `verify` that is not in this repository and through a C++ verifier built from other
sources, and rejected by both, in `verify` at that file's own check, by report (`lean/GAPS.md`, 14
(i) and 33), no lying or edited committer has been put to the verifier at the default setting at
full size. That file was tried over Lean files that differ from this repository's; no such trial has
been made on this repository's files. Every run, with its bounds: rows LIM-63 to LIM-63.2, HW-747,
M1.4, LIM-47, SO-7.2, SO-15.5, SO-15.7, LIM-2, LIM-14.3 and [lean/LIMITS.md](../lean/LIMITS.md) L61.

**Measured figures.** The prover has been timed in [the timed runs](#the-timed-runs): runs of the
released programs, made within three hours of one day. Six kinds of run at full size were made 3
times each, each kind's three on three machines: at the default setting, on a third party's public
4-bit files of Llama 4 Maverick a commitment with one request, and services at 24, at 64, and at 8
and 1 requests a batch; on synthetic weights of that shape a commitment, its audit and services at
24 (`cp_verify`'s timed calls are part of these runs); at that setting with a clipped-gap policy, on
the public files, sessions with every reply token checked and like sessions that check none. A
clipped-gap example was made once, and one call and one service at 64 on synthetic weights were each
run once; the call and the service failed. Of the public tests (173 rows), 172 were in four test
runs and one in a test run of its own (row RES-22); two of the 172 did not run. The tests ran on
programs that each test run built from these sources, and none was timed. Every other time, rate,
size or memory peak is marked as not on the released programs (row RES-6), or as on them and not
made to be timed (row RES-9), or is a figure of the runs that timed the serving scripts' check, by
itself and beside a serving `cp_prove`, on the released programs, in one of which a constant of the
scripts is not this repository's (row RES-32). All of it was got at the default setting, at that
setting with a clipped-gap policy or on the small examples, or involves no prover. All that ran on a
GPU, on one kind of GPU. No other time, rate, size or memory peak has been measured (row RES-6).

**What to do.**

- Read a passing test as "ran once, at a small size", and a count of attacks as "one value false".
- Check proofs on your own machine, with a `cp_verify` you built yourself and a commitment ID that
  you did not get from the prover.

**[Basis](#how-a-thing-is-known):** reported (the full-size runs at the default setting); run, not
on the released programs; read. Rows: T1, T1.1, E5, SO-13, SO-15, SO-15.1, SO-15.3, SO-15.5,
SO-15.6. The full entry: row LIM-45 and the rows named above, in [the table](limits-all-rows.tsv).

<a id="review"></a>
### The reviews are not independent

The project that wrote the code also ran every test and did every review described in these docs, so
a mistake that writers and reviewers share can go unnoticed. There has been no independent security
audit. Large parts of the code have had only a partial reading or none. The proof written on paper
has been read in portions only, never as a whole, and some portions only in a wording other than the
one in which they stand. Of the Lean library, Lean's kernel has checked the proofs, so a reviewer
need read only the statements and the definitions beneath them ([lean/AUDIT.md](../lean/AUDIT.md);
how much of these has been read: [lean/GAPS.md](../lean/GAPS.md), 2.5 and 18 (c)). Some comments in
the source code state a guarantee more strongly or more widely than the code gives it. For what
holds, read these limits and not the comments. (Row E25.)

**What to do.** Read the parts that matter to you before you rely on them.

**[Basis](#how-a-thing-is-known):** reasoned (from how the work was done).

## More limits, by subject

### What the programs can and cannot do

The provider sees every prompt and every reply. A commitment is good for only a fixed amount of use
and cannot be extended; see [the limit of use](#the-limit-of-use).

<a id="what-the-programs-can-serve"></a>
#### One full-size model shape, two acceptance policies, one kind of machine

The programs support one full-size model shape: that of Llama 4 Maverick, with a
[window](glossary.md#window) of 1,024 tokens. They accept two [acceptance
policies](glossary.md#acceptance-policy), exact match and clipped gap, and reject all others. The
only supported platform is Linux with a single 80 GB H100.

**64 requests per batch on such a card is not to be relied on.** In the timed runs, on synthetic
weights, under a drawn seed, a service of five batches of 64 ended at a failed CUDA call in its
first batch, and so did one call of the prover on a folder of 64 requests, made after it on the same
machine and commitment, so the two attempts are not independent; none of the 320 requests and none
of the 64 got a proof, the card's largest sample was 81,053 of 81,559 MiB in each, and no cause is
shown. Not on the released programs, on synthetic weights, under a drawn seed, one call of the
prover on a folder of 64 full-size requests and one service of five batches of 64 each ended at a
failed CUDA call, the service in its first batch; none of the 64 requests and none of the 320 got a
proof, and the cause isn't shown. Not on the released programs, the like call failed in each of four
more runs: in three of them, each on its own machine and commitment, 0 of 192 requests got a proof
(0 of 64 in each); each call ended with exit code 1, at a failed CUDA call; no request was marked as
failed; the card's largest sample was 81,047 of 81,559 MiB in each run. In the timed runs, on the
real, public weights, under a drawn seed, services at 64 requests a batch ended well in all three
runs: 960 of 960 proofs, each accepted by a separate run of the verifier (320 of 320 in each run);
at most 80,629 MiB of the card's 81,559 were in use
([guide/performance.md](../guide/performance.md); rows T50 and RES-11).

A deployment also needs components that are not in this repository; see [the vLLM
integration](#the-vllm-integration).

<a id="the-sha-instructions"></a>
**`cp_commit` needs a CPU with the x86 SHA instructions.** It is built to use them, and it does not
check for them when it starts.

**What to do.** If a program rejects your model, your files or your machine and offers no fix, the
programs cannot serve them, so do not work around the rejection. Run `cp_commit` on a CPU that has
the SHA instructions: `grep -m1 -ow sha_ni /proc/cpuinfo` prints `sha_ni` where the system shows
them (row C5.2 has a surer check that needs no GPU).

**[Basis](#how-a-thing-is-known):** read; reasoned (`cp_commit` without the SHA instructions);
reported (the runs at 64). Rows: C5.2, P4. What is recorded of the tested machines: row LIM-48. What
happens without the SHA instructions: row LIM-49.

<a id="after-a-fault"></a>
#### A fault can stop the service, and the output says little about the cause

Most faults that `cp_prove` detects in its own work stop the process, and the whole batch fails. If
the GPU hangs in the middle of a batch, the service hangs too: shutdown never begins, so the
clean-up timeout never fires.

Once it has begun to read secrets, the [production build](glossary.md#production-build) gives the
reason for a stop as one word from a fixed list, or not at all. Exit code 0 does not mean that
requests were served, and it never says that a proof verifies: check that each request has a proof.

**What to do.**

- After a fault, do not restart the service. Disable anything that restarts it automatically. Only a
  person should restart it, once the cause is known.
- Follow [When something goes wrong](faults.md#always-first) from its first step, which puts the
  service on hold.

**Worst case:** when a self-check fails, [a mask may be lost](#a-lost-mask), and then even a proof
that `cp_verify` rejects may contain unmasked hidden values.

**[Basis](#how-a-thing-is-known):** read. Rows: R6, C3, C3.3. Which self-checks end the service, how
late, and what runs have shown: row LIM-50.

### Running the programs safely

Do not run any of this repository's code under `sudo` or as root. The programs, scripts and tests
need no root rights, and as root a fault or a mistake reaches further. Only setting up the machine
needs root: [installing the build tools, or making an ordinary account](quickstart.md#install), the
GPU's driver, and a provider's system settings.

We believe that a mistake such as a wrong argument cannot make the three programs delete or
overwrite anything outside the directories named on the command line, and that they signal only
their own child processes. Both beliefs rest on reading the programs' source files, not on tests.
That reading was of other sources, which differ from these in places, among them in the code that
deletes files. The claim also holds only while no one else can change what your paths resolve to.
The reading did not cover the build scripts, the tests, the CUDA driver, runtime and cuBLAS, or
other kinds of harm, such as exhausting memory, disk or threads.

Two walks through [route 3](quickstart.md#3-gpu-a-small-model) bear on the second belief (the small
model; each printed command entered on a fresh GPU machine; 48 checks; not on the released
programs). In one, 47 of the 48 checks ended as expected: two `sshd` processes that stood in the
list of the machine's processes made at the start were gone at the end. Both belonged to the ssh
connection from which the walk was started. When they ended, and what ended them, is not known. The
other walk kept no ssh connection open, and all 48 checks ended as expected; this neither explains
nor clears the first. No run of route 3 after these two is known that compares the machine's
processes at its end with those at its start; that rests on a search that was not complete. That
route 3 sends no signal to a process that it did not start has not been shown by a run (row RES-21).

<a id="path-names"></a>
#### The guards against redirected paths are little tested

`cp_commit` and `cp_prove` guard against another account on the machine trying to redirect the paths
they use. The guards are little tested: attacks by a second account ran only through test drivers,
and the guards have several gaps, so follow the steps under "What to do" below.

**`cp_commit` checks only the paths it writes to.** It does not check the paths given with `-config`
and `-weights`, and it follows symbolic links there. The other gaps (among them: an older copy of
the secret key directory is accepted), and one more in the example script: rows LIM-51.1 to
LIM-51.3.

**What to do.**

- Give `cp_commit` output directories that sit under directories only you and root can write to, and
  put the `-worst-case-table` file in such a directory too.
- Put the secret key directory and the [spool directory](glossary.md#spool-directory) where no one
  else can write to them or to any directory above them.
- For a single batch, name `-in` and `-out` where no one else can write to any directory above them,
  or to `-in` itself.
- Under a sticky-bit directory such as `/tmp`, use a fresh directory from `mktemp -d`, never a name
  that others can predict.

**[Basis](#how-a-thing-is-known):** run, not on the released programs, through test drivers around
the programs' file-handling code, not through `cp_commit` or `cp_prove` themselves (but for one run
of a test build's `cp_commit`: row R2.6.1), and with no record in this repository; reported (the
check on the example script); read (the rest). Rows: R2, R2.1, R2.2, R2.3, R2.3.1, R2.4, R2.6.1,
R2.10, R24.

<a id="a-killed-cp-commit"></a>
#### A killed `cp_commit` can leave the seed behind

After a crash, a power failure, SIGKILL or any other signal that it does not catch, `cp_commit`
deletes nothing. It leaves two working directories whose names end in `.making`, one of which holds
the [seed](glossary.md#seed), and no tool deletes them. The program's final status line can wrongly
say that everything was removed. What can stay after a signal that it catches, the file named with
`-worst-case-table` among it: row LIM-52.

**What to do.**

- After any call that was killed or failed, check the output directories yourself rather than
  relying on the final status line. See [faults.md](faults.md#a-killed-or-failed-cp_commit) for what
  is left where, and what you must not touch.
- Keep everything that `cp_commit` writes on an encrypted disk.

**[Basis](#how-a-thing-is-known):** read (`prover/hash_commitment/hf_commit_making.h`); row R7.7
says what has been tested; reported (the wrong status line). Rows: R3, R3.1, R3.2, R11, R7.7.

<a id="the-spool-directory-is-cleaned"></a>
#### `cp_prove -spool` removes files from the directory that you give it

When `cp_prove -spool` starts, it cleans the spool directory, and while it serves it deletes some
files from `in/`. In a directory that it accepts, if one of your own files has a name that the
program cleans up, it is deleted without asking. Row R18 of [the table of all
rows](limits-all-rows.tsv) lists those names. What it rejects before it deletes, and what it catches
only after it has deleted a part: row LIM-53.

**What to do.** Give `-spool` a directory that you use for nothing else.

**[Basis](#how-a-thing-is-known):** read (`prover/prover/spool.h`); run, not on the released
programs (a test harness on that header, without a GPU and without the check for `mark.txt`). Rows:
R13.1.1, R18, R18.1, R18.4, R24.

<a id="one-commitment-one-directory"></a>
#### `cp_prove` does not tell one commitment from another

`cp_prove` is given no commitment ID and knows none: it does not tell one commitment's directories
from another's, and it reads no `key.bin`. Any well-formed `mark.txt` fits, and where `cp_prove`
started first the mark fits every commitment's front end, whose clean-up can then remove responses
there. A changed configuration file is caught by one guard alone. So a service can start on the
wrong directories: requests go to the wrong service, its proofs fail at the first `cp_verify` that
is given the right commitment ID, and the requests use up a leaf table's count of openings.

**What to do.** Keep one spool directory for each commitment, and give the check of [duty
6](#the-conditions) the commitment ID from your own record.

**[Basis](#how-a-thing-is-known):** read. Rows: R40, R13.1.1, H21.15.

<a id="the-tests-and-your-files"></a>
#### The tests build over your programs, and are only partly confined

One test builds the verifier into the repository's `bin/`, and the test `weight_sums` builds
`bin/weight_sums` there, so they replace a `cp_verify` and a `weight_sums` that you built there. For
that one problem, run the tests in a separate checkout. Each replacement is an honest build of the
code. The test runner also confines each test only partly ([details](limits-further.md#r1)), and the
worst case below says what a faulty test could do despite the confinement. The confinement guards
writing; it limits reading nowhere: a test run as root can read whatever root can (row R1.2). With
`--unconfined` the runner does not confine the tests at all: each then has all the rights of your
account. So does a test's command from `SUITE.md` that you type yourself, without the runner.

**After a run under `sudo`.** If the tests run under `sudo` in a clone that had no `bin/`, root owns
what they leave: `bin/`, the programs built there and the work directory. Under `sudo` the tests act
as root on names inside the clone, so whoever owns the clone can steer what root does there, as with
any program run by `sudo` from someone's copy. To go on with the tests, start again in a new copy of
the repository. What root made there, only root can remove or give to another account. This page
prints no command for that: whoever does it names each directory by its full path, and first checks
that it is the one meant and that no part of the path is a link.

**What to do.**

- Give the tests a work directory that does not exist yet, outside the repository, because the
  runner rejects anything else. If you set `BUILD`, point it at a directory that you use for nothing
  else.
- Put that work directory, and any directory that you name to `make_model`, under a directory that
  only you can write to.
- Serve only with the programs directly inside the production build's directory, `hc`, never with
  those in `hc/test_build` or below.

**Worst case:** if the tests run as root, a faulty test keeps root's other powers: the confinement
is a guard against a test's mistake, not against a test that means harm (row R1.1). On Linux 5.13 to
6.11, a faulty test running under any account can signal that account's other processes one by one,
and one run as root any process of the machine. Under any account, before Linux 6.2 a faulty test
can empty any file that the account may write. Under any account it can also create, write and
remove files in the repository's `bin/` and in the run's shared directory (rows R1, R1.1). It can do
the same in the machine's `/dev/shm`, unless it runs as root on a machine that allows it a separate
mount namespace. As root, on a machine that allows that namespace, each test gets `/proc/sys`,
`/proc/sysrq-trigger` and three more places of `/proc` read-only (seen for four of the five: row
R1.1); a test that uses a GPU could get past that on `/proc/sys`, by either of two raw system calls
(seen; nothing was written). What else a faulty test can do, as root and under any account
(`/dev/shm`, signals, `/proc`, `bin/`): rows LIM-54.1 and LIM-54.2.

**[Basis](#how-a-thing-is-known):** read (`tools/confine.py`, the scripts and the test table's
commands); reported (what was seen as a true root and under `sudo`, on one kernel); run, not on the
released programs (what a run under `sudo` leaves: one run, and one imitation of `sudo`). The
confinement level for Linux 5.13 to 6.11 has not run on such a kernel. No test with a GPU was run as
root: the rule of --gpu was tried on machines without one. Rows: R1, R1.1, R19, R19.1.

<a id="the-compiler"></a>
#### The CUDA compiler can miscompile a kernel silently

A miscompiled kernel can silently compute a wrong value, and a proof with a wrongly computed mask
still verifies (see [a lost mask](#a-lost-mask)).

The build rejects nvcc 13.0.88. Many of the tests and runs that these docs tell as not on the
released programs used GPU programs built with nvcc 13.0.88; far less has been run on programs built
with nvcc 13.2.86. nvcc 13.0.88 has two known bugs that can make a program silently compute a wrong
value. One is published. NVIDIA lists it as fixed in CUDA 13.2 Update 2, whose compiler is nvcc
13.2.86. We have not checked that fix by a test of our own. Whether the bug affected the programs
that nvcc 13.0.88 built, which are other programs, is not known. The other bug worries us more. We
have not found it in any release notes, and the build does not work around it. What was seen of it
(wrong values on one small kernel, and 0 of 17 cases passed in one build of one routine's test): row
H8.2. As far as we know, the machine code has not been searched for either bug's pattern.

CUDA 13.4 is not supported, though the build's version check lets it, and any later release, pass.

**What to do.** Build with the compiler version named in
[prover/docs/build.md](../prover/docs/build.md). It is not known whether that version is free of the
bug that worries us most: the build tests for it on one small sample kernel only, and with no usable
GPU it skips that test. Do not take a test result in these docs as a result for your own build.

**[Basis](#how-a-thing-is-known):** read (the vendor's release notes;
`prover/check_nvcc_version.sh`); reported (the other bug; the tries with CUDA 13.4). Rows: H8, H8.2,
H8.2.2, H8.3, H8.4, P4, P4.5. Two more known faults, one in the host compiler, and which toolkits
were tried: row LIM-55.

### The vLLM integration

The limits of the integration and of the client scripts are in a separate document,
[integrations/LIMITS.md](../integrations/LIMITS.md).

A known fault, which these sources have: at a start behind your own `sudo`, the example front end's
server, run as root, can be led to send out closed files of root's. If a directory that the server
is set up to serve lies inside a directory of your own, such as your home, a program that runs under
your own account can put another directory of root's in its place. It can do so before any start, a
supervisor's too, by renaming, or by putting a link in the place of a directory above it.

- `clean`, `hold` and `lift-hold` then act on whatever of root's stands under the names at that
  moment.
- A server that starts then sends a closed `key.bin`, leaf file or `layout.txt` of root's that is
  there to anyone who can reach it and has vLLM's API key, where vLLM was given one. How the example
  front end's middleware learns that key, and where it cannot: [the operator's
  checklist](../integrations/vllm/REFERENCE.md#the-operators-checklist). Where vLLM has a key that
  the middleware neither reads nor suspects, these files go out without a key unless `VLLM_API_KEY`
  is set. This is known by reading, and by tests with a stand-in for vLLM's application, which call
  these files' routes too; the middleware has run under vLLM, not on the released programs, at toy
  size, and four times at full size, there with the key in `VLLM_API_KEY` alone, and of these files'
  routes only those of the published `key.bin` and of a leaf file have been called under vLLM, at no
  server led astray (rows H11.2, HW-275).
- Nothing is logged, and the server stays so until it is restarted, though the names are right
  again.

Keep the directories, the config file and the checkout below directories that belong to root and
that only root can write ([the way out, also for a server that is not
root](../integrations/LIMITS.md#above-a-public-directory); row R25). What a swap under a running
server does, public files, and a server that is not root: row LIM-56.

### The model

<a id="the-integer-model"></a>
#### The integer model is not the floating-point model

Every proof is about an integer model of the network, and none says how close that model is to the
floating-point model.

Three rates were counted in one run, at the same positions of the same texts:

- 6.81 %: at 5,779 of 84,893 positions in 216 texts the integer model's top token differed from that
  of the 16-bit weights evaluated in float32.
- 6.65 %: at 5,643 of the same 84,893 positions in the same 216 texts the top token of the public
  4-bit files evaluated in float32 differed from that of the 16-bit weights evaluated in float32. No
  integer model and no fitted scale is in this rate.
- 3.11 %: at 2,638 of the same 84,893 positions the integer model's top token differed from that of
  the public 4-bit files, from which it is made, evaluated in float32.

The three rates do not add up, and need not: each counts the positions at which two models' top
tokens differ, and the 3.11 % and the 6.65 % are against different references. Since all three were
counted at the same positions, arithmetic gives this much: at 4,392 or more of the 5,779 positions
at which the integer model differs from the 16-bit weights, more than three quarters of them, the
4-bit files evaluated in float32 differ from the 16-bit weights too. So at the greater part of the
positions behind the 6.81 % the step from 16 bits to the public 4-bit files, which a third party
made, changes the top token already, before any integer arithmetic. The step to integers adds 136
positions net (0.16 points; with the texts as the units, these counts cannot tell whether that is
more than chance), but that is not its size: it changes the top token at 2,638 positions, either
away from the 16-bit weights' token (at 136 positions or more), or back to it, or from one differing
token to another; how many of each was not counted. A differing top token is not in itself an error
of either model, and none of the three rates says how good a reply is.

The 6.81 %: measured three times, on three machines, each time with new passes on both sides (5,779
in one, 5,778 in two, of which one against another public copy of the 16-bit weights), with a Python
implementation of the integer model that is not this repository's; the run whose counts stand here
was on integer weights and quantization scales with the same checksums as a fit by other programs
gives; the parameter file's other fitted lines were not compared; the weight files were not held to
their checksums in that run. All the prompts behind the 6.81 % were built on WikiText-2, and no task
benchmark was run. No such measurement has been made on the released programs. (Under what
conditions the other two were measured, and by which programs: row LIM-57 of [the
table](limits-all-rows.tsv).)

We also compared programs to see whether they compute exactly this integer model. We did so only at
small example configurations and on synthetic weights, and the programs were not written
independently ([OPERATING.md, "What has been
run"](../integrations/vllm/integer_model/OPERATING.md#what-has-been-run)). At full size, on real
weights, only greedy choices and which requests get a proof have been compared, not scores, and the
choices through the proofs' acceptance under exact match, not value by value (row M1.6). On
synthetic weights an agreement of greedy choices says little about attention, routing or the scales
([OPERATING.md, "What has been
run"](../integrations/vllm/integer_model/OPERATING.md#what-has-been-run)). Nothing has been compared
where a range condition fails.

**What to do.** If you vouch for a commitment ID, load the committed `policy.txt`, `KL.bin` and
weights into the integer model itself, then compare its replies with the floating-point model's on
your own text.

**Worst case:** a reply proved under [exact match](glossary.md#exact-match) is the integer model's
own, but it can differ from the floating-point model's. Under a
[clipped-gap](glossary.md#clipped-gap) policy it need not even be the integer model's own. No proof
or program bounds the difference.

**[Basis](#how-a-thing-is-known):** run, not on the released programs (how often the top choices
differ); reported (the runs on a real model's weights); read (the rest). Rows: M1, M1.2, M1.6, M1.7,
H29. The full entry, with a finding on GPU memory: row LIM-57.

## What the Lean proofs cover

- **Proved.** The Lean verifier, `verify` (not the C++ verifier that users run by default), is
  proved sound at the given public files, in the random-oracle model and at exact match (greedy
  decoding: the policy, set in the parameter file, under which every output token must be the
  model's own choice), against a committer and a prover who may both cheat. (The theorem:
  `VerifierAndGoals.AtTheName.sound_of_verify_at`, in
  `lean/Johnson/Goal/Skeleton/SoundOfVerifyAt.lean`.) It accepts a false claim with probability at
  most about q/2^124 + 2^-101 after q queries to the random oracle, the committer's and the prover's
  added up. That is 124 bits per hash query, and the bound is never smaller than about 2^-101 ([how
  that stands to the target of 128 bits a round](protocol.md#64-check-one-proof)). "False" means
  false at the weights that a fixed extractor reads off the commitment and the committer's hash
  queries. The theorem has three hypotheses: the public files are in scope, are parsed without
  error, and ask for exact match. The main conditions, beside exact match (which `verify` does not
  test: you check it in the parameter file): one proof format and model size (`verify` rejects
  others), and public files that are fixed before the hash function is sampled (a provider that
  tunes them to the real hash function is outside the statement). The oracle stands for the format's
  hash function; a second, fixed hash function is taken as separate from it. The same is proved of
  its executable form, a Lean term that a machine can run, where the functions on bytes that stand
  for the hash function answer with 32 bytes, as the real one does. Lean's kernel has checked the
  proofs, which rest on Lean's three standard axioms alone and on no conjecture on error-correcting
  codes; the random oracle and the second hash function belong to the statement, not to the proof's
  axioms. Completeness, where the model has a result, and zero knowledge are proved in Lean for the
  protocol, at full size and at the exact-match policy, each up to a small stated error, with fresh
  random masks; after compiling, both are on one leaf table. The honest prover in these theorems is
  a Lean definition, not the CUDA prover: nothing is proved for the programs.
- **Run.** The Lean files and the parser's sources in this repository differ from those of most of
  the calls told here; 109 calls of the reference run ([lean/THEOREM.md](../lean/THEOREM.md), "What
  has been run") were made over this repository's Lean files; the command and the parser were held
  by checksums before and after the calls; every file of the command and of the parser outside
  `lean/` that the traced calls beside them opened has this repository's checksum (none of the run's
  own calls was traced: [lean/GAPS.md](../lean/GAPS.md), 14, "Which programs (k)"); that the command
  and the parser read the files of the run's own tree, its compiled Lean files among them, and no
  others is not shown; the Lean verifier's other calls told here, but for the 41 on proofs of the
  timed runs told below, were not made over this repository's Lean files. At full size and on
  synthetic weights `verify`'s executable form has accepted every honest proof put to it in a call
  that ran to its end. At exact match: 363 of the 723 that a prover built from other sources made
  (not the 723 of the timed runs), handed over by a rule fixed beforehand (reported), and 18 by
  other provers: ten on three of the 96 requests of one shape that the 363 cover, seven on the one
  other request that they cover (two of these seven by report alone), and one on a request that no
  other proof here is of, so that 17 of the 18 add no request to those of the 363. Of the 363, 360
  are for one next token (96 requests of one shape from one list, each proved under each of three
  commitments, by their makers' word to one set of synthetic weights, 24 of them twice under each),
  and three are of one other request. Under the stored clipped-gap policy, where an accept says
  less: one by that prover, and four by others. `cp_verify` had accepted each of the 363 and the one
  before `lean-verify` saw it; they were judged by `lean-verify`
  ([tools/lean-verify/README.md](../tools/lean-verify/README.md)); that this repository's class that
  feeds and finishes BLAKE2b (row RES-39) was in its parser at those calls rests on files held by
  checksums before and after them, and at none of these calls was the code of its parser's compiled
  C++ program looked at. Of the timed runs' proofs only 41 (that they are of these runs: as
  reported), all on synthetic weights, have been put to the Lean verifier, over this repository's
  Lean files; it accepted each ([lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md)). The form has
  rejected all 26 proofs put to it that were made for a claim false by the run's own policy (24 of
  them share one request; the two under the clipped-gap policy have not been put to the command).
  The command has accepted none of 57 lying proofs: it rejected these 24, and 30 of 33 that claim
  the true reply and lie inside the proof, all 54 at the verifier's step that reads the proof file;
  at the other three, whose files its parser refuses before Lean is called, it gave no verdict. One
  commitment; exact match. The test programs that made the lying proofs are not in this repository.
  A proof file does not show which program made it: that the honest proofs in these runs came from
  the provers told here, and the lying ones from test programs, is how the runs were set up, not
  something a verifier can check. A run trusts Lean's evaluator, compiler and runtime, the helper
  library and the C compiler that builds it, and the file that drives the run (which runs were
  driven by files that are not in this repository: `lean/GAPS.md`, 45 (d)). Altered copies of an
  honest proof have been accepted too: of 41 altered inputs that were put to the executable form of
  `verify` directly, on parsed files, not through the released command, it rejected 32 and accepted
  nine, all nine altering Merkle-path entries of the parsed proof file that the verifier's walk does
  not take, the claim untouched and true; at three of those `cp_verify` and the parser rejected the
  altered raw file, so the two verifiers part there (lean/THEOREM.md, "What has been run";
  lean/GAPS.md, 66). That is malleability of the proof string, not a failure of soundness: in no run
  that lean/THEOREM.md records has an altered claim, or a claim false by the run's own policy, been
  accepted.
- **Not proved.** Nothing is proved about the C++ programs, the verifier that users run by default
  among them, nor that these programs run the protocol that `verify` checks. Completeness where the
  model has no result on the request, or another result than the claimed output; under a policy
  other than exact match; for batches; and of the CUDA prover, for which it is tested by runs
  ("Run.", above). (What Lean proves, with each theorem's name and file: [What is
  proved](../guide/what-is-proved.md), "Completeness".) Nor is correctness of extraction (that the
  extracted weights are the ones put in) proved for the release's own committing program. Nor are
  binding of the commitment in the opening form and no two openings proved of the release's
  committing program (of an illustrative committer written in Lean all three are: lean/THEOREM.md,
  "What the commitment gives, as proved"; of the committer that the compiler yields from the Lean
  provers, Lean proves, at public files in use, soundness with binding and, within that committer's
  own budget of 2^48 hash queries, that the extractor gives back the committed weights: [What is
  proved](../guide/what-is-proved.md), "The commitment"). The theorem's "model" is a program in
  integer arithmetic, not the floating-point model. No real model's weights are shown to fit it:
  that narrows what can be certified; it does not weaken soundness.
- **Weight hiding.** Lean proves zero knowledge, its form of weight hiding, for the protocol, at
  full size: honest-verifier zero knowledge before hashing, and after hashing, in the random-oracle
  model, against any verifier within 2^64 hash queries and 688,127 replies, at a statistical
  distance below 2^-154 ([where it stands](#the-verdict)). The honest prover in these theorems is a
  Lean definition, not the CUDA prover. It is at the exact-match policy, with fresh random masks,
  for one commitment with one leaf table and one request to a proof (batches, which are what the
  programs make, are not covered by these results), and one bit per request is revealed (in a
  corollary, one label of three). It says nothing of the programs. The soundness theorem on `verify`
  itself says nothing on hiding.
- **Sources, and what to read.** Part of the Lean library is adapted from third-party code
  ([lean/CREDITS.md](../lean/CREDITS.md)). The six top statements (`Sound`, `SoundAndBinding`,
  `SoundOpened`, `Complete`, `ZeroKnowledge`, `CompleteSoundZeroKnowledge`) are definitions, not
  theorems (row L18). The six lines (six theorems of `lean/AtFullSide/Face/TheSixLines.lean`:
  `sound`, `soundAndBinding`, `completeOnCanServe`, `zeroKnowledgeOnCanServe`,
  `refusedHandsOutNoProof`, `repliesFit`) are another set, the library's goals on the compiled
  prover and verifier: they are stated of the compiled system (the same verifier, `verify`, with the
  honest committer and prover that the compiler yields). The structure `FinalStatementOneBit` has
  six fields (`sound`, `complete`, `zeroKnowledge`, `binding`, `handsOutNoProof`, `repliesFit`), and
  one theorem, `theFinalStatementOneBitHolds`, proves all six of these (these and the six with the
  label shown: `theFinalStatementBothHold`). How the two sets stand to each other:
  [lean/LIMITS.md](../lean/LIMITS.md), L93. A pass of the kernel, the full check or the independent
  checker does not show that the definitions beneath the statements, `verify`'s own text among them,
  mean what these pages say of them: that is a reading; nothing machine-checks it.

Left out here, and told at [lean/THEOREM.md](../lean/THEOREM.md): what an accept means; three things
that you must check yourself (the parameter file's three policy lines; the scale file, which can
switch off any of the model's blocks, up to all of them; the output's length); and two that are
granted (the code that turns your files into the claim; what a run trusts). No further property of
the second hash function is assumed, and the theorem is not proved for every function in its place
([lean/THEOREM.md](../lean/THEOREM.md)).

The library holds the soundness theorem for the Lean verifier; the compilation theorem
(`innerCompiledVerifier_sound`), under 16 named hypotheses (which are not all that it needs: its
inputs also carry laws in their types, lean/LIMITS.md, V4) of which, at the library's descriptions
other than `theDescription`, the four on which soundness turns (`keySound`, `proofSound`,
`commitment`, `sound`) are not proved (a reduction, not a proof of soundness); and goals typed for
any IOP, as definitions, in lean/Ladder.lean ([lean/README.md](../lean/README.md);
[lean/GAPS.md](../lean/GAPS.md)). The soundness theorem on `verify` does not go through the
compilation theorem. It goes through the compilation theorem's form at `theDescription` (not the
theorem under a policy of [lean/THEOREM.md](../lean/THEOREM.md), which is about files under another
policy), and each condition that this form asks is supplied by a theorem, so the soundness theorem
on `verify` has none of them as a hypothesis. The sixteen conditions on the public files are tests
inside `verify`, not theorems. Soundness is argued on paper, conditionally, for the technical
report's protocol, and not settled; for the programs as they run, none is claimed.

A stand-alone program built from [`verify-on-bytes`](glossary.md#verify-on-bytes), which this
repository does not build as one, follows `cp_verify`'s source text and was run beside it. No
theorem bounds that program's chance of accepting a false file. No theorem ties it to `cp_verify`;
none ties it, without hypotheses that are unproved, to `innerCompiledVerifier`, the compiler's
product at open parameters; and the soundness theorem for the Lean verifier is not about it. Where
the two agree, that is not independent evidence (row PS-68).

The scope of the soundness theorem for the Lean verifier is one format and one size: the format's
default setting at full size. The sixteen tests that `verify` runs on the public files are proved to
pass at the library's own copy of the full-size public files; that this repository's files equal
that copy is checked by a script (`lean/scripts/check_stored_examples.py`), not by a theorem.
`lean-verify` runs `verify`'s executable form inside Lean; it is not a program of its own.

**How it is checked.** The independent checker has accepted the soundness theorem on `verify`
(`sound_of_verify_at`), with every declaration beneath it, and nothing that does not lie beneath it,
in a pass on that theorem's export alone: the exporter wrote that export out of a copy of the
compiled files of this repository's build (that the compiled files it read were that build's is
known from a report alone; Lean's own files were not compared with a list of checksums); the results
on completeness, zero knowledge and weight hiding stated here ([lean/LIMITS.md](../lean/LIMITS.md),
L62 to L65, L88, L90 to L92, and what L93 tells beside the six lines), and four theorems of L89
(`soundAsIOPAtTheProtocol`, `probabilityInScope`, `probabilityInScope79` and
`sound_of_verify_at_through_the_IOP`), are not among what the checker accepted in that pass, and for
them there is Lean's own kernel alone (the pinned version of Lean is older than published fixes of
soundness bugs: [lean/LIMITS.md](../lean/LIMITS.md), L45;
[lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md); what such a pass is worth, and why a reader
cannot repeat it: [lean/CHECKING.md](../lean/CHECKING.md), "The independent checker"). Of the six
lines ([lean/LIMITS.md](../lean/LIMITS.md), L93) the independent checker has accepted `sound`,
`repliesFit` and `refusedHandsOutNoProof` by name, over a build of this repository's files; over
such a build the three on binding, completeness and zero knowledge have been checked by Lean's
kernel alone. That the compiled files that these passes read were that build's is known from a
report alone; Lean's own files were not compared with a list of checksums. (Rows L43, LIM-59.)

**Further limits,** each told in full in [lean/LIMITS.md](../lean/LIMITS.md). In short: row LIM-60.

**What to do.** Take the commitment from a source that you trust. Build the request yourself. If you
are handed a request, compare all of it with what you sent and received. This is a duty toward the
programs, to which no theorem is tied. Read the statements yourself, starting from
[lean/README.md](../lean/README.md).

**[Basis](#how-a-thing-is-known):** proved (of `verify`; of the Lean committer and provers, as
above); read; reported (what the library's pages say of the checks). Rows: those named L with a
number, in [the table of all rows](limits-all-rows.tsv).

<a id="what-lean-proves"></a>

What Lean proves on hiding: **honest-verifier zero knowledge of the protocol's three honest provers
before hashing, and zero knowledge of the compiled committer and prover after hashing:** told, with
every bound, [above](#the-verdict). Two lemmas beside these:

1. **The committed tables.** At the default setting, for the encoder as written in Lean, before any
   hashing: what is shown of them has the same distribution whatever the weights. It covers no
   message of a prover (below).
2. **The link proof.** Before hashing, at full size, Lean proves perfect honest-verifier zero
   knowledge for it (`homeExactAt`). A written simulator is given the two tables' OOD seeds and
   answers, the verifier's challenges and the columns opened in the two tables. It produces exactly
   the distribution of what the link proof's honest prover, as written in Lean, discloses. That
   holds for all weights, all encoding randomness and every challenge, with no error and no premise.
   At full size, where the tests on the public files pass, the link's verifier accepts the same
   prover at every challenge that passes proof of work, at one fixed public-files digest, at which
   `verify` is not proved ever to be called. This is no statement that the weights are hidden:
   answers and columns are computed from them.

The programs themselves are not formally verified, as is usual.

In the library's own terms: [`lean/TERMS.md`](../lean/TERMS.md), "What is proved, in these terms".
Where these results are narrower than what runs: [there](#narrower-than-what-runs).

Lean proves, for the encoder as written in Lean and before any hashing, that what is shown of the
committed tables has exactly the same distribution whatever the weights. That holds at the default
setting, at the 400B model's shape, for each table's OOD answers together with up to 44,040,191
opened columns.

*Its bounds.* It covers no message of a prover: the provers in its statement are fixed-message
provers, which send the same messages whatever the weights and so show nothing of them. So it is not
index privacy. At public files in use (row PS-71) the protocol's honest provers are other provers;
they fall back on the fixed-message ones only where a function that makes an honest prover refuses.
It supposes fresh random values (the pads, which mask these answers, and the rows' encoding
randomness) and sample points that depend on nothing hidden. The programs derive the random values
from a secret seed and get the points by hashing the table's root, so neither holds of them. It
stops before any hashing: roots, salts and paths are outside it. What the provers' own messages add,
and hashing, are covered by the theorems on zero knowledge ([where it stands](#the-verdict)). The
44,040,191 columns are the budget of a table's whole life. Nothing in Lean keeps count of it, and at
one column more the single-table statement fails. The programs keep that count, in a file for each
table. As a whole it is stated at that shape and nowhere else. As is usual, no theorem ties it to a
program. In sum, no Lean theorem covers OOD answers as the programs make them (from a seed, at
hashed points). After hashing, the hashed scheme in Lean too gets the points by hashing the root;
its random values stay fresh.

## What was never tried

These docs do not report a test of anything on this list, on the released programs or on any others,
so we do not know whether any of it works or fails. Attacks on soundness and hiding come first,
platforms last.

- A malicious key file that passed its check only at a lower security [target](glossary.md#target)
  and is then checked under the 128-bit parameters.
- Parameter files that `cp_verify` accepts but that no honest program here writes.
- A reply under a clipped gap that misses both of the policy's limits at once.
- Faults in a batch of several requests, except three cases: one false response, one value pushed
  out of range (row T30), and batches with a failing request. In those batches, no one examined the
  other requests' files. A request whose masks are lost stands apart from this list: every value
  that they should hide can then go out unmasked in a proof that verifies (row HW-686), and such a
  loss has not been tried on the released programs.
- How much timing reveals in total ([more](limits-further.md#h11.1)).
- Prompts chosen against the prover, beyond one small search; clients that pool their proofs;
  solving for another request's values (row HW-16).
- Giving `cp_verify` a named pipe, or a file that changes while it is read; searching it for data
  races or uninitialised memory.
- A second, separately written prover to compare this one with ([more](#compiled-programs)).
- A rejection at a provider's check ([duty 6](#the-conditions)) in a batch of 24 at full size.
- Killing or signalling a program on purpose when its parameter file asks for proof of work, as the
  default setting does (row T29.1).
- A power loss during a commit or a batch ([operators-duties.md](operators-duties.md)).
- Most of the commands that the docs print, run from start to end on a fresh machine.
- A network file system or a shared volume.
- Any CUDA compiler other than nvcc 13.0.88 and nvcc 13.2.86, except the few tries with nvcc 13.4.59
  ([the compiler](#the-compiler)).
- A CUDA 13.2 toolkit as NVIDIA's installer or a system package manager such as apt installs it
  ([the compiler](#the-compiler)).
- A bash older than 5.2: the docs accept bash 4.4 or later, but we know of no run of the shell
  scripts under one.
- Any GPU other than an NVIDIA H100, or any system other than Linux on x86-64.

<a id="how-a-thing-is-known"></a>
## Kinds of evidence

Each limit ends with a line that begins **Basis:** and names the kind of evidence. The kind tells
you what was done, not whether the result is right. If an entry has rows in the [table of all
rows](limits-all-rows.tsv), their IDs are at the end of its Basis line. A figure that bears the
bracket "reckoned by a planning call …", or is told "by the planning call", has that call for its
basis, whatever the basis told beside it says of the rest. Such a figure is the released programs'
own accounting, and the rule on unmarked runs (under "Run", below) does not touch it: at the
default's full-size parameter file the released `cp_verify` has made the call, and it printed the
figures that these pages give as its own (by report).

- **Proved:** a theorem in the library under `lean/`, checked by Lean's kernel, which is trusted,
  not proved (the pinned version of Lean is older than published fixes of soundness bugs:
  [lean/LIMITS.md](../lean/LIMITS.md), L45). It holds under its own assumptions, in the mathematical
  model that it names. Lean does not check that anything meets those assumptions, or that this page
  describes the statement accurately. The Lean files of this repository have the checksum 620d8930.
  One Lean file, with one definition and four theorems, which nothing else imports, was added after
  the runs reported here and checked on its own. Three of the check script's pinned values were
  changed and three of its stored lists were extended for it; nothing else in the script or the
  lists changed. (The file: `lean/AtFullSide/Face/ExecutableVerifierIsLeanVerifier.lean`; the full
  account: [lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md).) Wherever these pages say that a run
  was made over this repository's Lean files, over a build of them or over the released Lean files,
  that file was not among them. Without that file and these changes, at the checksum c06b37e6, these
  have run: the build of the whole library with its two side libraries, with no error and no `sorry`
  (with Mathlib and the other packages compiled beforehand, the library's own files from nothing, in
  several calls of Lake, the last of them over all three libraries:
  [lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md) tells how); Lean's own print of the named
  theorems and of the axioms that each rests on; and, of the library's check, the steps on the
  files' text (tree, words, imports); and its audit of axioms and of the lists of names, inside the
  whole run of that check that is told below (10,772 names; 184 stored statements). The Lean
  verifier's reference run has been made over this repository's Lean files (row L9), and the
  independent checker has been run, on the soundness theorem's export and, in passes of their own,
  on those of three further theorems (`sound`, `repliesFit`, `refusedHandsOutNoProof`), as the
  exporter writes them over this repository's build; what the checker accepted (the soundness
  theorem on `verify`, three further theorems, and what lies beneath each: row L25):
  [lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md). One whole run of the library's full check
  (`lean/scripts/check.py`) over the Lean library in `lean/` has the last line "All steps passed:
  tree, words, imports, build (kernel replay, audit). …", at the checksum c06b37e6. That line also
  says that no theorem got a verdict on binders or on `Classical.choice`: every line of
  `lean/scripts/top-theorems.txt` says "general" (the whole line:
  [lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md)). It is one run, on one machine. In its kernel
  replay 1,438 modules go by another call than `leanchecker`; Mathlib and the other packages are
  taken as they were compiled beforehand, and are not replayed ([lean/GAPS.md](../lean/GAPS.md), 81
  (g)); the 13 files under `Ask/` are read by the text steps alone. The script runs no mutation
  tests and no second, independent proof checker.
- **Run:** programs were run and the outcome was checked. "The released programs" are `cp_commit`,
  `cp_prove` and `cp_verify` as the sources in this repository build them (the production build);
  which runs these pages tell as runs on them, and by what rule, is told under [what was run on
  these programs, and what was not](#what-was-run-on-these-programs). Where this page says neither
  "on the released programs" nor "not on the released programs" of a run, and does not tell it as
  made "in the timed runs" ([what these are](#the-timed-runs)), take it as the second; a Basis line
  or a cell of the table that says only "run" means: not on the released programs, unless the row
  tells the timed runs. A test of the public table ([SUITE.md](../SUITE.md)) is told apart: 171 of
  its 173 rows have run on programs that each test run built from these sources (most run test
  programs or scripts, not the three released programs themselves; eight run `cp_commit` or
  `cp_prove` as test builds: [what was run on these programs, and what was
  not](#what-was-run-on-these-programs)). A fault of the code, told in the present tense, holds of
  the released programs.
- **Run, not on the released programs:** the same, but on other programs, so it tells you nothing
  certain about the released programs unless it is repeated. Count a leak found not on the released
  programs as a leak of the released programs, unless a page says that these sources do not have it.
- **On test builds:** on programs with a fault built in, a test switch or a test seed.
- **Run in a trial; run in a trial, not over the files of this repository:** labels of the rows
  whose names begin with L only, said of the Lean library and not of the programs. "Run in a trial"
  means that a program was run, but not as a run of record; "not over the files of this repository",
  that it ran over Lean files that differ from this repository's. The page
  [lean/LIMITS.md](../lean/LIMITS.md) explains "Basis"; [lean/GLOSSARY.md](../lean/GLOSSARY.md)
  explains "run of record", "trial" and "walk".
- **Read:** the code was read, and nothing was run. The line adds "not tested" where no test covers
  the claim. Unless the line says otherwise, it was read in other sources, that is, sources that
  these pages do not tell as this repository's ([which they tell
  so](#what-was-run-on-these-programs)). In the table "read" means the same.
- **Reasoned:** worked out from the design, the code or a configuration, but not observed.
- **Argued on paper:** a written argument that Lean has not checked.
- **Reported:** told from an account of the run or the reading, whose files were not checked for
  this page; of which programs ran, the same holds as for "run", and of which code was read, the
  same as for "read".
- **As the note tells** (in the table only): the row's note gives the kind for each thing that it tells.
- **Nothing checks it** (in the table only, row L43): nothing in this repository shows or checks it.
- **Not examined:** no one has run, read or tried it.

**The setting is a second matter.** The labels that begin "Run" tell which code ran, not at which
setting. A run is to be taken as made at the [default setting](glossary.md#default-setting) only
where its entry says so; otherwise its result holds there only if the same test has run there. The
same holds of what an entry gives as read, reasoned or argued on paper, where it depends on the
setting.

"The technical report" is a technical report of this project that is not public and not refereed. It
states a corollary with counted proof of work and a proposition on what would remain without the
theorem on codes: fewer bits than the target, for the truth of an accepted request and for binding
alike, and the report does not call it firm. The corollary and the proposition have had one review,
of the parts of each that concern soundness and no others, in readings of minutes each. Sentences in
both were found false as worded; by the report's own account it corrects them (row PS-35). Unless
they say otherwise, what these docs tell as read, reviewed or compared of the technical report was
of a text from which the report differs.

"No theorem" or "not proved", said of Lean, means that we searched the library and found no theorem
that states the claim. It does not mean that the claim is false.

Where these docs say that something has not been run, tried, tested, repeated, measured, checked or
read, they mean that they report no such run, trial, test, measurement, check or reading.
