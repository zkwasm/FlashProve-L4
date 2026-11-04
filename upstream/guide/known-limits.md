# Known limits

<a id="soundness-and-weight-hiding"></a>

**What we have.** Soundness is proved, in Lean, for the verifier written in Lean: it accepts a proof
of a false claim with probability at most about 2^-124 per hash query (with a floor of about
2^-101), in the random-oracle model, for [exact-match](terms.md) claims; under the [clipped-gap
policy](terms.md) that serving uses, Lean proves only that the reply meets that policy (below). The
C++ verifier and the CUDA prover are written to follow the same protocol and are tested against it,
not themselves proved. Completeness, where the model has a result, and zero knowledge are proved in
Lean for the protocol, at full size and at the exact-match policy, each up to a small stated error,
with fresh random masks; after compiling, both are on one leaf table. The honest prover in these
theorems is a Lean definition, not the CUDA prover: nothing is proved for the programs. Where the
model has no result on a request, completeness says nothing. So, for a prototype, we have reasonably
high trust in what the system produces: soundness, which matters most for a verifier, is
machine-checked for the Lean verifier, and the C++ verifier is tested against it. We expect, rather
than know, that the programs hide the weights, within the bounds told under [Weight
hiding](weight-hiding.md); few attacks were sought, and no test on the released programs hunts for a
leak. More: [What is proved](what-is-proved.md).

**What follows is what is left:** what a user of FlashProve can trip over or be misled by. The known
limits at length: [docs/limits.md](../docs/limits.md),
[docs/limits-further.md](../docs/limits-further.md), [one table](../docs/limits-all-rows.tsv) and,
for Lean, [lean/LIMITS.md](../lean/LIMITS.md). The list may be incomplete: the last reviews behind
it found new limits.

## If you check replies

**The provider chooses the model.** A yes does not say whose weights they are, that the shipped
committer made the commitment, or which real model it is. A provider can commit to any model it
likes, a small or degenerate one included, and a scale file within the admitted range can switch off
model blocks, up to all of them (worked out by hand). Build `cp_verify` yourself. [Get the
commitment ID from a party you trust](verify-a-proof.md#what-you-need-first), never from the server
you check or from files sent with a proof, and hold the three configuration files as you hold the
ID.

**`WITHIN LIMITS` does not say the reply is the model's.** Under the looser clipped-gap policy an
accept says only that the output's token costs meet the policy's limits at the weights the extractor
reads off the commitment, which by itself bars little: with loose limits, close scores or flat
(all-zero) weights almost any output passes. Lean's theorem for that policy ([the theorem under a
policy](what-is-proved.md)) says only that the output meets it, and the argument on paper (below)
covers its equations only in part. The policy's mean and count of token costs run over one output
that the prover picks, and prompt tokens are not scored. No run shows the clipped-gap policy's own
limits rejecting anything at a verifier (replies that missed the limits got no proof from the honest
prover). Both kinds of accept exit with code 0, so read the last line. Yet clipped gap is the
expected use: of the 96 honest replies scored† from each of two stock engines none would pass exact
match, and 28 of 32 replies† with one token swapped for the model's second choice met limits fitted
to honest replies (both scored by the integer-model package, in Python, not proved). The project
recommends no limits: if you cannot assess them, accept only exact match.

**A pass covers only checked tokens.** Only the named output positions are covered, and a reply
stopped short verifies as a shorter one. [Build the request file yourself](verify-a-proof.md),
compare the `covered:` line with your range, and check that the reply ends with an end token: [exit
code 0 alone does not mean accepted](verify-a-proof.md#when-to-accept). A proof names no client or
time. A hostile file can exhaust memory or time: treat a call that hangs or is killed as not
accepted.

**Proofs are about the integer model,** quantized and fixed-point. How far its outputs track the
original floating-point model's is a measurement, not part of any proof (in one measurement†, with a
Python implementation of the integer model, the top token differed at 6.81 % of positions), and no
trained weights are shown to meet the protocol's range conditions (a count† outside Lean, by the
integer-model package's own range checks, found none failing on 544 of 544 windows, at a public
model's 4-bit weights).

**Printed bits are not a security level.** The C++ programs' printed "bits" are their own
accounting, not a theorem: the weakest round prints 128.00 bits with up to 32 bits of proof of work
counted by assumption (96.23 without), no margin, about 6 bits fewer per batch of 64. That
accounting rests on two published results that are cited, not re-derived for these programs (a
proximity-test soundness analysis and a theorem on codes), and on an argument on paper, in a
technical report that is not public and not refereed, whose conditions no complete set of parameters
of any size is known to meet and which a configuration file can fail while still passing the
verifier's checks. The proved figure is the Lean theorem's, about 124 bits per hash query (with a
floor of about 2^-101), not the programs' printed one. Lean also has theorems on single rounds,
which are not the chance of a forgery: [What is proved](what-is-proved.md). Two published attacks on
Fiat–Shamir transformations (ePrint 2025/118; 2026/1838) have not been examined against this
protocol or these programs.

## If you serve a model

**The provider's duties.** Hiding depends on what the provider does: [the thirteen
duties](serving.md#the-providers-duties). Keeping them all does not show that the weights are
hidden: little has looked for a leak or an attack, and any real model is outside what we judge of
hiding (only one of five tests of hiding has run on any). [The example front
end](../integrations/LIMITS.md#what-the-integration-reveals) does not meet [duty 11](../docs/limits.md#the-conditions), so a server with it alone in front is outside what we judge of hiding. By that duty a front
end asks only about served replies, over ranges that it sets. What breaking a duty can cost, what is
visible by design, and what tests of lost masks found: [Weight hiding](weight-hiding.md).

**The prover does not check its own proofs.** `cp_prove` releases a batch's responses once all of
them are staged. After a hardware fault or a bug, a response that `cp_verify` rejects can hold
unmasked values. The vLLM integration that ships hands on no proof, and nothing of its response, before `cp_verify` has passed every proof of its batch (narrow exceptions, none open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py` reports no rate, and an example script shows no answer, before `cp_verify` has accepted the proofs; and a front end of your own must do the like ([Weight hiding](weight-hiding.md);
[docs/limits.md](../docs/limits.md#a-response-that-does-not-verify)).

**A commitment runs out.** At full size and the default setting each leaf table (a masked encoding
of the weights) serves 688,127 batches, by calculation (the same number in Lean's theorems counts
replies, each to one request, not batches: [Weight hiding](weight-hiding.md), "The counter").
Nothing warns you before that limit, and a commitment cannot be extended. A new commitment has a new
ID, and [no proof links the two](verify-a-proof.md#when-the-commitment-id-changes). Compare the size
of each leaf table's `count.bin` with `cp_commit`'s line `the limit:`.

**64 per batch is not to be relied on with an 80 GB card:** all eight recorded attempts on synthetic
weights, six of them†, failed with no proof, cause unknown. On a public model's 4-bit weights all
three services at 64 ended normally (960 of 960 proofs accepted), but the GPU reached 80,629 of its
81,559 MiB. The two kinds of run differ in more than their weights ([Performance](performance.md)).

**The example server and `sudo`.** The example front end's server, started with `sudo`, [can be led
to send out files that only root may read](../integrations/LIMITS.md#above-a-public-directory). Run
nothing here as root.

**Also:** [after a fault, do not restart the service](../docs/limits.md#after-a-fault); [a killed
`cp_commit` can leave the seed behind](../docs/limits.md#a-killed-cp-commit); [`cp_prove -spool`
removes files from the directory that you give
it](../docs/limits.md#the-spool-directory-is-cleaned); [`cp_prove` does not tell one commitment from
another](../docs/limits.md#one-commitment-one-directory).

## How far the tests go

Many of the tests and readings that this repository reports were not made on the released programs,
but on programs built from [other sources](../docs/glossary.md#other-sources); some, the tests with
deliberately cheating provers among them, on programs that also held test-only code. The compiled
programs differ too; the machine code that differs is not shown equivalent and has not been read.
Two of the five tests of weight hiding, the first and the fifth, have run on the released programs,
the first at full size too, the fifth at small size, and one of its two audits, the audit of a
commitment, at full size too, on none but honest commitments, in the timed runs. The second, the
third and the fourth ran on other programs ([Weight hiding](weight-hiding.md)). On the released
programs `examples/maverick/run.sh` has gone once through all the stages of `run.sh all`, at full
size, on a real model's public weight files: the run reported that the five values in
`examples/maverick/expected.txt` came out the same, byte for byte, and that `cp_verify` ended with
`YES`; its build stage was skipped, its download stage fetched nothing, and the tools that its stage
`needs` asks for were installed beforehand, so building, downloading and installing the tools were
not exercised. What ran on which programs: [what was run on these programs, and what was
not](../docs/limits.md#what-was-run-on-these-programs). There has been no independent security
audit. AI systems, under human direction, wrote nearly all of the software, the Lean proofs and the
documentation, and ran the tests reported here.

**Tests against lying proofs are narrow.** At full size 57 lying proofs were tried (24 made for a
false claim, on one request; 33 carrying a planted lie, for the true replies to three requests):
none was accepted by either verifier (the Lean verifier's parser refused 3 outright, and a
no-verdict is not a rejection). Calls on the 57 lying proofs are among those of the Lean verifier's
reference run; what is known of the command and the parser there stands in row L9 of [the
table](../docs/limits-all-rows.tsv). Whether the command and the parser at the Lean verifier's calls
told in this paragraph outside that run were this repository's has not been compared. No proof of a
false claim that also carries such a planted lie has been put to either verifier at full size in the
runs this page reports. Those 57 trials were on one commitment, synthetic weights, exact match, lies
of a few kinds, in 22 of the 57 none aimed at the lookup checks (by report), every rejection at one
verifier step; all 24 were refused at one and the same place inside that step. None was put to
`lean-verify` under a clipped-gap policy. No prover was built to search for forgeries; apart from
one cheating committer's false leaf file (a false copy's file) at full size, tried once through a
form of the executable `verify` that is not in this repository and through a C++ verifier built from
other sources, and rejected by both, in `verify` at that file's own check, by report
([lean/GAPS.md](../lean/GAPS.md), 14 (i) and 33), no malicious committer ran at full size at the
default setting; and other attacks were mostly at small sizes. That file was tried over Lean files
that differ from this repository's; no such trial has been made on this repository's files. The test
programs that made the lying proofs are not in this repository, so those trials cannot be repeated
from it. Most of the Lean verifier's runs on honest proofs cannot be repeated from this repository
either: except for the two stored full-size examples' proofs, the honest proofs are not in it.

**The two verifiers are close, not identical.** The Lean verifier runs sixteen tests of its own on
the public files and refuses where one fails; those tests are proved, in a side library, to pass at
the library's own copy of the full-size public files; that this repository's files equal that copy
is checked by a script (`lean/scripts/check_stored_examples.py`), not by a theorem. Elsewhere the
theorem may hold only because the Lean verifier refuses. A parameter file can pass the C++ verifier
and yet give less soundness than the figures it prints. More: [lean/GAPS.md](../lean/GAPS.md),
[lean/LIMITS.md](../lean/LIMITS.md).

**The `blocks/` library can be misused.** A program that calls it directly, outside the contract in
`blocks/chain/closed.h`, can be made to accept a false claim; at least one such way is known to work
at the programs' full configured target (128 bits by their own accounting), at toy size. That the
C++ verifier is not exposed rests on reading its source, not on a test. Check proofs with the C++
verifier or the Lean verifier; do not build a verifier on `blocks/` without keeping the contract.

<a id="bad-results"></a>**Altered stored tables can verify.** A committer that alters part of the table it
stores beside a commitment can still have its proofs accepted by the C++ verifier (observed with up
to 131,072 of 8,388,608 positions altered, on a small configuration, and 24 of 24 tries at 1 to 256
positions; on test builds). In every such case the table, by separate programs' counts of the changed columns (BLAKE2b-256 taken as collision-free), not machine-checked and not by a proof, still lay far inside the distance at which it decodes to the honest weights alone, so that the accepted claim was the honest run's own: a yes vouches for the claim, not for every byte stored.

**Proofs are malleable.** Some bytes of an honest proof, or of a commitment's files, can be altered
and the proof still accepted, for the same true claim; non-malleability is not claimed, and an
accept does not show that this exact file came from the prover. The Lean files and the parser's
sources in this repository differ from those of the Lean verifier's calls told in this paragraph. Of
41 altered inputs put to the Lean verifier's executable form directly, on parsed files, not through
the released command, nine were accepted (all altering Merkle-path entries of the parsed proof file
that the verifier's walk does not take; the claim untouched). The two verifiers can part on such
altered files: at three of four altered copies of one honest proof, the claim unchanged and true,
the Lean verifier accepted the parsed file where the C++ verifier rejected the raw one (one run†).

† Not on the released programs: [which kinds of result these
are](../docs/limits.md#on-other-programs).
