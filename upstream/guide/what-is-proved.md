# What is proved, and of what

- **Soundness** is proved, in Lean, for the verifier written in Lean: it accepts a proof of a false
  claim with probability at most about 2^-124 per hash query (with a floor of about 2^-101), in the
  random-oracle model, for [exact-match](terms.md) claims; under the [clipped-gap policy](terms.md)
  that serving uses, Lean proves only that the reply meets that policy ([Known
  limits](known-limits.md)). The C++ verifier and the CUDA prover are written to follow the same
  protocol and are tested against it, not themselves proved.
- **[Completeness](terms.md) and [zero knowledge](weight-hiding.md)** are proved in Lean for the
  protocol, at [full size](terms.md), each up to a small stated error: first for the interactive
  protocol (IOP), then for the compiled (non-interactive) prover and the same Lean verifier, in the
  random-oracle model. The honest prover in these theorems is a Lean definition, not the CUDA
  prover. Completeness covers well-formed requests on which the model has a result and the claimed
  output is that result. Where the model has no result it says nothing. Zero knowledge holds against
  an honest verifier in the IOP and, after compiling, against any verifier within a bound on its
  hash queries and replies; the simulator's running time is not modelled. Both are at the
  exact-match policy, with fresh random masks, for one commitment and one request per proof and,
  after compiling, on one [leaf table](terms.md), not for batches. One bit per request is revealed
  (served or refused: a request is refused if it is ill-formed, if the model has no result on it, or
  if its result is other tokens), and it depends on the weights. A service that chooses to say
  "failed" (that is not the shipped programs' own "failed":
  [limits](../docs/limits.md#narrower-than-what-runs)) reveals one label of three in its place
  (served, failed or other tokens). Neither says anything of what the public files themselves tell
  of the weights, or of timing. In the code both are tested, not verified: every honest proof tried
  was accepted, and in honest runs of [the released
  programs](../docs/limits.md#what-was-run-on-these-programs) we have seen nothing beyond a declared
  list ([how far these tests reach](weight-hiding.md#the-programs)).
- So, for a prototype, we have **reasonably high trust** in what the system produces: soundness,
  which matters most for a verifier, is machine-checked for the Lean verifier, and the C++ verifier
  is tested against it. We expect, rather than know, that the programs hide the weights, within the
  bounds told under [Weight hiding](weight-hiding.md); few attacks were sought, and no test on the
  released programs hunts for a leak. Before you commit weights that must stay secret, read [Weight
  hiding](weight-hiding.md).

This page gives the conditions and the limits for soundness and completeness; [Weight
hiding](weight-hiding.md) does so for zero knowledge. The same in two pages, with one table: [the
paper](../paper/paper.pdf), section 1.3 and Table 1. The theorem in exact terms:
[lean/THEOREM.md](../lean/THEOREM.md).

FlashProve has two verifiers. The C++ verifier, `cp_verify`, is the program that users run. The Lean
verifier, `verify`, is written in Lean, a proof assistant, so that theorems about it can be checked
by machine ([the terms](../lean/TERMS.md)).

## Soundness

**The bound.** The Lean verifier accepts a proof of a false claim with probability at most about
q/2^124 + 2^-101 after q hash queries (about 124 bits per query; not "128 bits"). "False" means
false at the weights that a fixed extractor (not shown to be computable) reads off the commitment
and the committer's hash queries, which need not be the provider's. The bound is on the chance of a
false accept, not on the chance that a given accept is false, and it weakens as queries grow, saying
nothing from about 2^124 queries on. Lean's kernel has checked the proof. Its conditions:

- **Exact match.** The theorem is for the acceptance policy under which the claimed output must be
  the model's exactly: every output token is the model's top choice. For any other policy a second
  theorem, *the theorem under a policy*, says only that an accepted output meets that policy ([Known
  limits](known-limits.md)).
- **The random-oracle model** for the hash, with a second, fixed hash function, built on BLAKE2b's
  compression function, assumed independent of it, against classical adversaries.
- **Configuration files fixed before the hash function is drawn:** the three public files
  (parameter, layout and scale file). A provider that tunes them to the real hash function is
  outside the statement.
- **One proof format and model size:** full size, not the small examples'. "Admissible
  configuration" means the full-size layout and proof format, with eight parameter-file numbers free
  within stated ranges and any scale file that Lean's reader accepts; outside it nothing is claimed
  and the Lean verifier refuses.
- **One proof.** N proofs handed in at once under one commitment, or N members of a batch, cost N
  times that error, taken at the whole list's hash queries; for proofs checked one after another, or
  members checked in separate sessions, no bound is stated.

Three further theorems bound the error of every round by 1/2^128, at public files in scope where `verify`'s own tests on them pass; they bound single rounds, not the chance of a forgery. Proof of work is counted: without its 20 to 32 bits, the rounds that sample query positions have errors of about 2^-96 to 2^-110 (added up by a program, no theorem). Neither figure, 124 bits per query or 128 bits per round, is claimed for the C++ programs.

**The proved verifier runs.** The executable form of `verify` computes `verify`'s own decision, and
is proved sound in the same sense, at the same bound (for hash functions with 32-byte answers, as
the real one). One command, [`lean-verify`](../tools/lean-verify/README.md), runs it on the prover's
proof files. lean/ also holds Lean text that writes cp_verify's checks again, following its source
(`lean/OnBytes/`); it is not `verify`, the soundness theorem on `verify` is not about it, and
`lean-verify` is the command to use.

**What the proof rests on.** No conjecture on codes is assumed: the coding theory it needs is proved
inside the library, in files adapted from other formalizations, whose match to the published
statements has not been independently compared. (The C++ programs' printed bounds do take a theorem
on codes as given: [Known limits](known-limits.md).) `verify` and the definitions beneath the
statements are definitions: the kernel checks proofs, not what a definition means, so a reader of
the statements must read those definitions. An independent proof checker (a second kernel) reads
text that our own exporter writes, no run shows it refusing a wrong proof, and for four theorems
(three about the commitment), in a pass over Lean files that differ from this repository's, it gave
no verdict; wider passes, also over Lean files that differ from this repository's, ended without a
verdict for many more, most for memory or time, and refused nothing
([lean/GAPS.md](../lean/GAPS.md), 18 (b)). The independent checker has accepted the soundness
theorem on `verify` (`sound_of_verify_at`), with every declaration beneath it, and nothing that does
not lie beneath it, in a pass on that theorem's export alone: the exporter wrote that export out of
a copy of the compiled files of this repository's build (that the compiled files it read were that
build's is known from a report alone; Lean's own files were not compared with a list of checksums);
the results on completeness, zero knowledge and weight hiding stated here
([lean/LIMITS.md](../lean/LIMITS.md), L62 to L65, L88, L90 to L92, and what L93 tells beside the six
lines), and four other theorems, of L89 (`soundAsIOPAtTheProtocol`, `probabilityInScope`,
`probabilityInScope79` and `sound_of_verify_at_through_the_IOP`), are not among what the checker
accepted in that pass, and for them there is Lean's own kernel alone
(the pinned version of Lean is older than published fixes of soundness bugs:
[lean/LIMITS.md](../lean/LIMITS.md), L45; [lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md); what
such a pass is worth, and why a reader cannot repeat it: [lean/CHECKING.md](../lean/CHECKING.md),
"The independent checker"). Of the six lines ([lean/LIMITS.md](../lean/LIMITS.md), L93) the
independent checker has accepted `sound`, `repliesFit` and `refusedHandsOutNoProof` by name, over a
build of this repository's files; over such a build the three on binding, completeness and zero
knowledge have been checked by Lean's kernel alone. That the compiled files that these passes read
were that build's is known from a report alone; Lean's own files were not compared with a list of
checksums.

**The commitment.** The provider publishes a commitment ID, which, by the protocol, binds it to one
set of weights: a commitment, whoever made it, and the committer's hash queries settle, at the given
public files, one set of weights, or none, before any proof is made. That the committed weights are
the ones put in, and that a commitment cannot be opened to other weights, is proved of a committer
written in Lean, which models one part of `cp_commit`, at files where `verify`'s tests on the public
files pass, at errors of about 2^-89 and (t + 2^23)/2^124 after t hash queries. It is not proved of
`cp_commit`, the C++ committing program (the two were compared in values at small size only). Of the
compiled system (the same verifier, `verify`, with the committer that the compiler yields from the
Lean provers) Lean proves soundness with binding (`SoundAndBinding`, the final statement's field
`binding`; `finalSoundAndBinding_value`): nobody opens that committer's commitment to other weights
than the ones the extractor reads off it, except with probability about q/2^124 + 2^-100 after q
hash queries; and within that committer's own budget of 2^48 hash queries the extractor gives back
the committed weights, except with probability about 2^-155 (`finalExtracts_value`). To open is to
name the weights and one number. These read no prover, and no theorem ties that committer to
`cp_commit`. No theorem ties a commitment to one set of public files or one policy. Neither kind of
binding says that a commitment holds any weights: where the extractor gives none, any accept counts
as false.

**The library's six top-level statements** (`lean/VerifiableInference/Statements.lean`: `Sound`,
`SoundAndBinding`, `SoundOpened`, `Complete`, `ZeroKnowledge`, `CompleteSoundZeroKnowledge`). Of the
Lean verifier's own system Lean has only `Sound`, restricted to the files that ask for exact match.
Of the compiled system (the same verifier, `verify`, with the honest committer and prover that the
compiler yields) Lean proves the final statement's six fields (`FinalStatementOneBit`,
`lean/Johnson/Face/FinalStatementOneBit.lean`; `theFinalStatementOneBitHolds`; these and the six
with the label shown: `theFinalStatementBothHold`), at all public files in use (in scope, readable,
at the exact-match policy, and passing the verifier's first test). Two of these fields are top
statements as written, `Sound` and `SoundAndBinding`; completeness and zero knowledge are proved in
other forms, for the reasons that follow. As written, `Complete` asks an accepted proof at every
request whose output is the model's own, the empty output and requests that cannot be served among
them. Of a system that meets the goal of lean/Ladder.lean on soundness under a policy, which counts
an accepted empty output as a forgery, it holds at no small error (read; no theorem says it). So
what is proved is the form on served requests: the requests that can be served whose claimed output
is the model's own (`finalComplete_iff_completeOn`). That is a fault of the statement, not of the
protocol. As written, the unrestricted `ZeroKnowledge` is expected to fail (lean/GAPS.md, 39); what
is proved, of the compiled system, is a form of it at every request, with one bit per request
revealed, and as a corollary a form at well-formed requests with one label of three shown ([Weight
hiding](weight-hiding.md)).

## Completeness

In one line: proved in Lean for the protocol, on served requests, at the exact-match policy, first
for the interactive protocol and then for the compiled prover and verifier; where the model has no
result it says nothing; in the code, tested.

**In Lean.** The honest committer and all three honest provers are written in Lean, as definitions,
not programs. *After compiling* (`FinalComplete`, the final statement's field `complete`,
`lean/Johnson/Face/FinalStatement.lean`; proved by `finalAtT1_filled_withRoom`,
`lean/AtFullSide/E2E/TheCompiledStepFilled.lean`, inside `theFinalStatementOneBitHolds`,
`lean/AtFullSide/Face/TheFinalStatementOneBitHolds.lean`): under a random hash function, at all
public files in use (in scope, readable, at the exact-match policy, and passing the verifier's first
test), for all weights and every [served request](terms.md), the Lean committer commits, the Lean
prover proves, and `verify` rejects that proof under that commitment with probability at most 2^-40.
The probability is over the hash function and the committer's and the prover's coins, and the 2^-40
covers the compiled prover's own failures too, such as a search for proof of work that gives up.
*Before hashing* (`complete_in_scope`, `lean/AtFullSide/Honest/CompleteAtTheFiles.lean`): at all
public files in scope that pass the verifier's first test, on served requests, the honest provers'
run fails with probability at most 2^-41, over the provers' coins and the challenges that pass the
proof of work; that form of completeness for an interactive protocol is this library's own
definition. **What it does not say.** Where the model has no result on the request, or another
result than the claimed output, it says nothing. It is in the random-oracle model: at BLAKE2b no
theorem shows that `verify` accepts a proof, and acceptance there is seen in runs
([lean/GAPS.md](../lean/GAPS.md), 52). It is at the exact-match policy and for one request per proof
and, after compiling, on one [leaf table](terms.md), not for batches. That some request is served is
shown at made-up weights (`complete_in_scope_with_a_served_request`); no theorem says which requests
a real model serves. It says nothing of the CUDA prover.

**At weights that meet a condition.** Beside the six lines (the six theorems of
`lean/AtFullSide/Face/TheSixLines.lean`: [which six](../docs/limits.md#what-the-lean-proofs-cover))
there is a further theorem ([lean/LIMITS.md](../lean/LIMITS.md), L93), which is not a field of the
final statement (`theHonestPart_complete_where_some_worst_case_check_passes`,
`lean/AtFullSide/E2E/CompleteWhereSomeWorstCaseCheckPasses.lean`): at public files in use and at
weights that pass the first of the Lean library's two worst-case checks, or the least-bound check
with numbers that hold, for every prompt that is not empty and whose tokens are in the vocabulary,
and every output length of at least 1 that fits (at most 1,025 tokens together), there is exactly
one output of that length that is the model's own, and the honest proof for that prompt and that
output is accepted except with probability at most 2^-40. At weights that pass the first worst-case
check, or the least-bound check with numbers that hold, the model has a result on every well-formed
request (`modelNeverFails_of_passesSomeWorstCaseCheck`,
`lean/VerifiableInference/Bounds/SomeWorstCaseCheck.lean`). The theorem covers no request that the
final statement's field `complete` does not already cover: it does not make completeness hold where
the model has no result; it names weights at which that case does not arise. It is not `Complete`
either: it takes a hypothesis on the weights.

These are the Lean library's own checks; neither is `cp_commit`'s worst-case check: no theorem
relates either to a check that any program makes. Of that premise: its first arm is this library's
own decidable test on settings, tables and weights; its second is a statement that numbers exist
that hold together ("numbers that hold": each stretch bound that the least-bound check reads is at
least 0, and no vector of whole numbers gets longer by more than that factor when it is multiplied
with its matrix; which matrices, and what they are made from: `LeastBoundNumbers.Hold`,
`lean/IntegerModel/Bounds/LeastBoundCheck.lean`), and that with them a second test, the least-bound
check, passes: a statement, not a test. No theorem says that the second arm holds of any weights of
which the first does not: the premise contains the first check (whatever passes it meets the
premise), and no theorem says that it is wider. The weights that are shown by name to pass the first
check are made up. As it is written the first check can be evaluated at small sizes; at the full
sizes it cannot in practice (the head of `lean/IntegerModel/Bounds/Check.lean`); the least-bound
check, as its definitions stand, is costly to evaluate (the head of
`lean/IntegerModel/Bounds/LeastBoundCheck.lean`). No theorem says whether a real model's weights
meet that premise, and nothing in a commitment, in a proof or in what a verifier does shows that the
committed weights meet it ([lean/LIMITS.md](../lean/LIMITS.md), L93).

**In runs.** The Lean files and the parser's sources in this repository differ from those of most of
the calls told in this paragraph; 42 of the 364, those of the reference run
([lean/THEOREM.md](../lean/THEOREM.md), "What has been run"), have been repeated over this
repository's Lean files; the command and the parser were held by checksums before and after the
calls; every file of the command and of the parser outside `lean/` that the traced calls beside them
opened has this repository's checksum (none of the run's own calls was traced:
[lean/GAPS.md](../lean/GAPS.md), 14, "Which programs (k)"); that the command and the parser read the
files of the run's own tree, its compiled Lean files among them, and no others is not shown; the
Lean verifier's other calls told in this paragraph, but for the 41 on proofs of the timed runs told
below, were not made over this repository's Lean files. Every honest full-size proof from one prover
built from [other sources](../docs/glossary.md#other-sources) that was put to the Lean verifier in a
call that ran to its end was accepted (evaluated, not proved; which prover made a proof: as
reported): 363 at exact match, chosen by a rule fixed beforehand out of the 723 that it made on
synthetic weights (not the 723 of the timed runs), and one under a clipped-gap policy, where an
accept says less, each after its makers' own `cp_verify` had accepted it. All of it was at
essentially two configurations, on synthetic weights, most requests of one shape. Of the timed runs'
proofs only 41 (that they are of these runs: as reported), all on synthetic weights, have been put
to the Lean verifier, over this repository's Lean files; it accepted each
([lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md)). The proofs from the runs on real weights were
checked by `cp_verify` alone; none has been put to the Lean verifier, so no proof that the released
`cp_prove` made on such weights has been put to it either. At the public files of exact-match runs†
on other programs (taken from the same source folder, fixed before the runs; not compared byte for
byte here) the scope test passed by one evaluation and the Lean verifier's sixteen [tests on the
public files](../docs/glossary.md#tests-on-the-public-files) were not evaluated, and the files of
the runs† under the other policy were not put to the test, so whether the Lean verifier would accept
those proofs is not known ([lean/GAPS.md](../lean/GAPS.md), 57). The runs matter because a verifier
that rejects everything is sound too, with error 0 (`rejectAll_sound`,
lean/VerifiableInference/Checks.lean), and completeness is proved in the random-oracle model alone:
at BLAKE2b no theorem shows that `verify` accepts a proof. A guard, an extra check inside `verify`,
can cost completeness (a refusal), never soundness.

## The programs

The C++ verifier and the CUDA prover are written to follow the same protocol and are tested against
it. They are known to differ from the protocol as Lean describes it at listed points
([lean/GAPS.md](../lean/GAPS.md)); whether any of the differences matters has not been shown either
way. On the released programs `examples/maverick/run.sh` has gone once through all the stages of
`run.sh all`, at full size, on a real model's public weight files: the run reported that the five
values in `examples/maverick/expected.txt` came out the same, byte for byte, and that `cp_verify`
ended with `YES`; its build stage was skipped, its download stage fetched nothing, and the tools
that its stage `needs` asks for were installed beforehand, so building, downloading and installing
the tools were not exercised ([what was run on these programs, and what was
not](../docs/limits.md#what-was-run-on-these-programs)). In the timed runs on the public weights at
the default setting all 1,851 proofs were accepted by `cp_verify` (1,845 in the runs behind the
figures; six more, one on each of six further commitments), and all 249 altered copies and calls in
these runs were refused; in the timed sessions under a clipped-gap policy and their control all 720
proofs were accepted, 360 of them within that policy's limits, not by exact match; the released
verifier refused the 57 kept lying proof files that were put to it (full size, synthetic weights,
one commitment: row LIM-63.1); 228 kept calls, at small size, on synthetic weights, on altered
stored tables were accepted: each table, by separate programs' counts of the changed columns
(BLAKE2b-256 taken as collision-free), not machine-checked, still lies far inside the distance at
which it decodes to the honest weights alone, so that the accepted claim is the honest run's own
(row SO-15.7). The Lean files and the parser's sources in this repository differ from those of most
of the Lean verifier's calls told in this paragraph; of these, 109 calls of the reference run have
been made again over this repository's Lean files; the command and the parser were held by checksums
before and after the calls; every file of the command and of the parser outside `lean/` that the
traced calls beside them opened has this repository's checksum (none of the run's own calls was
traced: [lean/GAPS.md](../lean/GAPS.md), 14, "Which programs (k)"); that the command and the parser
read the files of the run's own tree, its compiled Lean files among them, and no others is not
shown; the Lean verifier's other calls told in this paragraph, but for the 41 on proofs of the timed
runs told under "In runs." above, were not made over this repository's Lean files. On every proof
file that both verifiers have judged they agreed; where no common file could be made (four altered
copies† of one honest proof), they disagreed on three (the claim true, the C++ verifier the
stricter). Nothing is proved of the C++ verifier that the examples and most users run, of the CUDA
prover, or of the committer, and no theorem says that they run the protocol the Lean verifier
checks: that rests on tests. BLAKE2b is treated as a random oracle; its implementation and the
ChaCha20 stream are ordinary code tested against published vectors and against each other, and the
GPU kernels are tested in part and mostly at small sizes ([SUITE.md](../SUITE.md)); none of it is
verified; compilers and hardware are trusted. The programs and the proofs have had no independent
security audit. What the tests tried, and how far they go: [Known limits](known-limits.md).

## Look first at four things

Before you rely on an accept by the Lean verifier:

1. **The policy.** The Lean verifier does not itself test that the policy is exact match. Check that
   the parameter file's (`policy.txt`) lines `C`, `theta`, `cap` read 256, 256, 0: of its [seven
   policy lines](../docs/glossary.md) these decide the kind.
2. **The public files.** Hold the right public files as you hold the commitment, and read the scale
   file, which can switch off model blocks ([Known limits](known-limits.md)). No file shows that
   they were fixed before the provider could tune them to the hash function: that you grant.
3. **The output's length.** Any prefix of the model's output counts as correct, so a reply cut short
   passes too (the empty one is barred only by the theorem under a policy). Check the output's
   length yourself.
4. **The claim.** `lean-verify` turns your three files into commitment ID, prompt and output with
   short functions that no theorem covers; a fault there would give an accept about a different
   claim than yours (the command prints the claim it judged). The proof's bytes, by contrast, need
   no trust: the theorem holds for every proof string. A run also trusts Lean's compiler, evaluator
   and runtime, the helper library and the C compiler that builds it, the file that drives the run,
   and some constants of Lean's own library and the packages, none of this library's, whose compiled
   code is not the kernel-checked definition.

## How to check it

A build alone does not fail on an unfinished proof or on a new axiom: the final checks do.
lean/CHECKING.md has the set-up, the commands and what machine they were run on;
`python3 -B scripts/check.py`, typed in `lean/`, rebuilds the library, replays it in Lean's kernel
and audits its axioms. The library's full check builds the third library, ShapeLines/, too, and
replays its modules and the side library's by another call of the same function of Lean's, at a
larger stack, not by `leanchecker` ([lean/CHECKING.md](../lean/CHECKING.md)); the theorems on
completeness and zero knowledge rest on that third library; a whole run takes some hours (in one run
4 hours 26 minutes by the clock, on one machine, held to 84 cores, with the scratch copy in memory;
`LEAN_NUM_THREADS=72`, `--jobs 48 --wide-jobs 24`), and that run, over the Lean library in `lean/`
without one file that was added afterwards and checked on its own (row L25), passed all its steps
(what its last line says, and what the check does not cover:
[lean/CHECK-RESULTS.md](../lean/CHECK-RESULTS.md)). To check the theorem by name:
[lean/THEOREM.md](../lean/THEOREM.md#how-to-check-it). The statements:
[lean/AUDIT.md](../lean/AUDIT.md). The gaps between Lean and the C++ programs:
[lean/GAPS.md](../lean/GAPS.md).

† Not on the released programs: [which kinds of result these
are](../docs/limits.md#on-other-programs).
