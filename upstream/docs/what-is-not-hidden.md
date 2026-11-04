# What is not hidden

This page lists what clients and the public can see even when everything works as designed.

Masking is built in and on by default. The design covers every published value that depends on the weights with a [mask](glossary.md#mask), a random value that hides a secret one, or publishes only a hash of it under a secret salt. The exceptions are on the declared list. **At least eleven things are visible.**

| | What is visible | Who sees it |
|---|---|---|
| 1 | [Each reply](#replies) | the client, and anyone who gets its proof |
| 2 | [Whether a request was served or failed](#served-or-failed) | the client; by timing, the batch's other clients too (item 3) |
| 3 | [Time](#time) | every client |
| 4 | [The architecture and every size](#architecture) | anyone |
| 5 | [Settings fitted to the weights](#fitted-settings) | anyone |
| 6 | [That `cp_commit`'s checks passed](#commit-checks) | anyone |
| 7 | [That a batch failed or the service stopped](#failures) | every client with a request in progress |
| 8 | [How much a provider has served, and when](#volume-and-schedule) | every client, and anyone who collects proofs |
| 9 | [Who shared a batch, and how many](#batch-mates) | any client, and anyone who holds two proofs |
| 10 | [Lengths, places in the batch and nonces](#lengths-and-places) | anyone who holds the files |
| 11 | [Masked sums of the weights](#record-sums) | anyone who holds the public files |

**The list says what is visible by design, not that everything else is hidden.** No one has shown that the list is complete or assessed what the items reveal together. We do not claim that revealing them is harmless, and no limit is known on what items 1, 2, 3 and 7 reveal.

<a id="what-was-read-and-run"></a>No one has checked the list item by item against the released programs, or against any run at the default setting. One fast reading of the sources of `cp_commit` and `cp_prove`, whole but for `cp_verify`'s own files, found no leak beyond this list with the four conditions met. It was of a text from which these sources differ in code in 68 of 384 files (in 136 more in comments, names and the wording of messages alone; the file of one class is among the 68), and two more of the 384 are not among these sources; in the 68 the differences in code were read for hiding, by difference alone, apart from renamed names and reworded messages, and the files were not read whole again (row HW-745).

"Weight hiding", or "hiding", is this project's plain name for a goal: clients and the public learn nothing about the weights beyond this list. The cryptographic property nearest to it is [index privacy](glossary.md#index-privacy) ([lean/TERMS.md](../lean/TERMS.md)). What the protocol aims at is close to that property, not the same: it has batches, and as the programs run it, it is weaker in several ways. In Lean zero knowledge is proved for the protocol, with a Lean definition of the prover, and not for the programs ([Weight hiding](../guide/weight-hiding.md)).

**Main risks.** A lost mask, silently dropped by a bug or a hardware fault: a file with an unmasked value can still verify, and no verifier can see it. A bug or a hardware fault can also spoil a proof so that it does not verify, and it can then lack a mask: `cp_prove` runs no verifier, so unless a self-check stops the run the file leaves the prover. So no response may leave the provider before `cp_verify` has accepted every proof of its batch ([duty 6](limits.md#the-conditions)): the vLLM integration that ships hands on no proof, and nothing of its response, before `cp_verify` has passed every proof of its batch (narrow exceptions, none open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py` reports no rate, and an example script shows no answer, before `cp_verify` has accepted the proofs; and a front end of your own must do the like. The list assumes that no mask is lost in a file that verifies, and that no file that does not verify reaches a client ([more](limits.md#the-main-risk)). On test builds of [other sources](glossary.md#other-sources), under injected faults at small size, some injected losses went unnoticed† (rows HW-647 and LIM-63.2 of [the table](limits-all-rows.tsv)). A second main risk: a use counter set back by restoring or copying the secret key directory, which no program detects ([duty](limits.md#the-conditions)). What the batches past the limit give away (by reasoning: row H21.11) has not been tried on the released programs.

**The list assumes four conditions. If any of them fails, a client can learn more than this page lists.**

- The run is [honest](limits.md#honest-run) and uses a production build that does what its published sources say.
- The provider follows the thirteen [duties](limits.md#the-conditions), no untrusted code shares its machine, and its GPU is never given to anyone else ([why](limits.md#gpu-working-values)).
- The machine has no faults. No duty can guarantee that. Among what a fault can send out: a file that verifies and lacks a mask ("Main risks" above), and item 7, "A response that does not verify".
- The [seed](glossary.md#seed) is secret, and `cp_commit` itself sampled it. So the commitment was not made under `-public-weights`. Under a [test seed](glossary.md#test-seed), one that a test build reads from a file, nothing is hidden, and no verifier can tell ([duty 7](limits.md#the-conditions)).

With all four met, the design aims to show nothing beyond this list. What these pages say about weight hiding does not cover a commitment made from weight files in which any tensor that the programs read is stored in 8 or 16 bits, or the proofs made under it, and the programs take such files without a warning ([why](limits.md#files-of-8-or-16-bits)). For other settings and two commitments taken together, see [Not covered](limits.md#not-covered). The judgement on weight hiding reaches a clipped-gap policy on less evidence ([docs/limits.md](limits.md#which-policy)).

**What index privacy counts, and what it does not.** Its definition compares the public files and the proofs that a verifier gets with what a program that never sees the weights can make from the public settings and from each request that is put to the prover, with its reply. So it is worth less the more the settings hold (items 4 and 5).

What requests and replies let someone work out about the weights, up to rebuilding a model from many replies, does not count against it (item 1). Only requests that pass the [acceptance policy](glossary.md#acceptance-policy) are put to the prover there: one that fails it has no place in the definition (item 2). One that passes it and is still refused, or whose batch fails, for a reason that lies in the weights, counts against it, and no one has shown that this cannot happen here (item 2, "Rarer failures"; item 7). Time has no place in the definition either (item 3). This list counts all of these.

## What a client learns

Some words first:

- A client asks for a proof about text it already has. Its *request* names a *window* of tokens (a prompt and the model's *reply*; 1,024 at full size) and a range of *checked tokens* within it.
- A *front end* is the provider's own program between clients and `cp_prove`. Unless it restricts them, the client chooses window and range.
- A request is *served* if its checked tokens pass the *acceptance policy*: exact match or a score policy. The *response* is a proof file and a token file, or "failed".
- Every proof is about the *integer model*, the network with quantized weights.

Statements rest on reading the code unless an item says that something ran. That reading, a quick one, covered nearly all of the source files of other programs ([which](limits.md#what-was-run-on-these-programs)), and none of the released programs was built or run for it (row HW-118). The order is our judgement of how much each item reveals, not a measurement.

1. <a id="replies"></a>**Each reply.** *The client, and anyone who gets its proof.* Replies depend on the weights, and there is no limit on what many replies together reveal.
   - **The next token.** If the checked range runs to the window's end, the proof file contains the *next token*, the integer model's top choice, unmasked.
   - **A proof file is as sensitive as its request.** Anyone who has it and the public files can test any guess at the whole request.
   - **Where a reply ends.** Where vLLM serves the integer model itself (integer serving), where a reply ends can depend on a hidden internal value ([integrations/vllm/integer_model/OPERATING.md](../integrations/vllm/integer_model/OPERATING.md#what-a-client-can-do), "What a client can do", leak 2).

2. <a id="served-or-failed"></a>**Whether a request was served or failed.** *The client; by timing, the batch's other clients too (item 3).* What this reveals depends on the policy. None of the programs counts or limits a client's requests: that is left to the provider's front end ([rule 2](#front-end-rules)), and the example front end does not do it.
   - **Under a score policy** such as [clipped gap](glossary.md#clipped-gap), the outcome depends on how far each checked token's score falls below the top score. So a client that sends many requests with tokens it chooses can measure hidden scores. Published work recovers a model's last layer, up to a linear map, from many such scores (rows HW-414 and HW-557 of [the table](limits-all-rows.tsv)).
   - **Under [exact match](glossary.md#exact-match)**, the outcome shows whether every checked token is the integer model's top choice, but, by design, not how close any choice was. That has been worked out for one configuration only (row HW-121). One full-size request tests up to 1,023 of a candidate model's choices. By reasoning alone, a client that halves the checked range finds the first difference in about ten more requests (row H7.6).
   - **Rarer failures** can depend on hidden values. One kind is an internal value that leaves its range. `cp_commit`'s worst-case check is meant to rule such a failure out; that it does rests on an argument and on reading the code, and is not proved. A client that chooses the checked range can leave it empty of the request's tokens. The policy then cannot fail the request, and, apart from rare events of chance and faults, a "failed" that `cp_prove` wrote then tells just whether an internal value left its range (from reading the code; not run).
   - **What a provider can do.** Apply [rule 5](#front-end-rules) under exact match too, though no duty requires it there. We argue, but have not tested, that this narrows the leak without ending it ([duty 11](limits.md#the-conditions)).

3. <a id="time"></a>**Time.** *Every client.* `cp_prove` proves requests together in a *batch*, and a client sees how long its batch took. Nothing is done to hide that. (The times of a response's files: [prover/docs/closed_front.md](../prover/docs/closed_front.md#what-this-list-does-not-cover), "File timestamps".)
   - **The written proof leaves time out.** The [technical report](glossary.md#technical-report) puts the prover's time and memory outside what an adversary sees (row HW-577 of [the table](limits-all-rows.tsv)).
   - **No one has bounded** what timing or any other side channel reveals, or measured it on the released programs.
   - **What it depends on.** A batch's time depends on how its tokens are routed to experts, and therefore on the weights and on other clients' prompts. It also depends on how many requests the batch holds and on whether one failed, and perhaps on where (from comments in the code; not measured). By a reading of the code, it may also depend on a request's other hidden values, on the masks and on the seed (rows HW-14, HW-122, HW-266 and HW-529).
   - **Neighbouring batches.** The time at which a response is released also follows the batches next to its own, which hold other clients' requests (row HW-553). Under load, how long one batch takes also decides which requests share the next one, and clients who compare their proofs can see which requests shared a batch (item 9).
   - **What was measured.** One trial† ([integrations/LIMITS.md](../integrations/LIMITS.md)): on the model server alone, with no prover behind it, time to first token told two of our synthetic models apart, at prompts we chose (row HW-265 of [the table](limits-all-rows.tsv)). No trial at the default setting has measured what a batch's time shows.
   - **A run of `cp_commit`.** How long it takes is not among this list's items. That time and the run's memory traffic on the GPU can depend on the weights' values, which has not been measured. Do not let others see when such a run begins and ends (rows H37.1, H37.2 and HW-530).

4. <a id="architecture"></a>**The architecture and every size.** *Anyone.* The whole configuration is public: layers, widths, experts, vocabulary and window length.

5. <a id="fitted-settings"></a>**Settings fitted to the weights.** *Anyone.* Two sets of public values are computed from the weights and a calibration set. They are the quantization scales (`KL.bin`) and the lines of the parameter file that the tools in `scales/` fit.
   - **A guessed model can be tested.** Anyone with a candidate model can run the same tools on it and compare the results, mostly without the calibration set.
   - **They show roughly how large the largest weights are.** For about a fifth of the weight matrices, the values give the largest entry to within a factor of two. For the others, they give the product of two matrices' largest entries to within a factor of four. Anyone who knows or guesses one activation size for each pair can split most of those products.
   - **How far this is known.** We calculated it for tools that differ from the released ones, and have confirmed it only on small synthetic models. Our only bound on what the files reveal is their size, about three kilobytes at full size. The full-size example's `KL.bin` has 2,784 bytes, and the fitted lines of the parameter file come on top ([the tools](../scales/README.md)).
   - **Other settings.** Any other setting a provider picks after looking at the weights is public too. The sharpest case is a score policy whose thresholds are fitted on a model's replies. What they reveal has not been worked out (row HW-230).

6. <a id="commit-checks"></a>**That `cp_commit`'s checks passed.** *Anyone.* A [key file](glossary.md#key-file) exists only if the checks passed. So it says that every weight outside the embedding lies in its declared range, and a client can rule out any model that would fail the checks. A service that takes requests shows that `cp_prove`'s startup tests on the model's files passed too. That is what an honest run gives away. It is no evidence for someone who doubts the provider: a commitment does not show that the checks ran on these weights.

   Keep what `cp_commit` prints, all but the commitment ID and the line `the limit:` (duty 9), as secret as the weights: which message it prints, and a few small numbers in it, can depend on them.

7. <a id="failures"></a>**That a batch failed or the service stopped.** *Every client with a request in progress.* A client sees that a stop happened and when (row HW-562). How much one stop can reveal about the weights has not been worked out, and many stops can reveal more.
   - **Stops that could depend on hidden data.** Three possible causes are named: an allocation of GPU memory, an overwrite of masks, and a response that `cp_verify` rejects, or gives no verdict on, in the provider's check of a batch ([duties 6 and 8](limits.md#the-conditions)), though `cp_prove` had not failed the request (at the vLLM integration that is a hold on the policy, and `cp_prove` goes on). None has been ruled out (of the stops at 64 requests a batch the cause is not shown: row T50), and no one has shown that there is no other.
   - **Stops that a client causes.** A client can cause a stop on its own in three known ways, one of them by using up a leaf table's batches (item 8; the other two: rows HW-402 and HW-412). Not known: whether a client can cause a stop that reveals something about the weights.
   - **Whom it reaches.** A stop reaches the clients of more than one batch: from reading the code, by default an end of `cp_prove` can withhold one batch that it has proved and not yet named, and the next start removes what was named and not yet synced (in the ordinary case of the batch named last; the code sets no bound).
   - **A response that does not verify.** `cp_prove` runs no verifier. A response that `cp_verify` rejects can hold hidden values unmasked: by one reading an exact linear form in the weights; no upper bound is known for any cause (row H26). A client to whom it is handed on holds all of that, whatever its own verifier then says. [Duty 6](limits.md#the-conditions) forbids the handing on, so by then the list's second condition has failed; after a fault of the machine the third has too. A bug in the published sources breaks none of the four by itself; nor would a request whose hidden data made `cp_verify` reject its response, which has not been ruled out (row HW-580). Handed on, such a response reaches its own client alone; the "failed" that keeps it back reaches the whole batch. A response can also fail to verify with no bug and no fault, at one of the rare hash outputs at which an honest proof is rejected. If the failing test is on rare challenge values (row C2) that `cp_verify` makes and the written proof's verifier does not, the prover that the written proof defines carries on. So what it puts out is masked like any other proof that it makes (inferred, and of the written proof's protocol only). What any other rejected proof shows, the written proof leaves unsaid. `cp_prove`'s code was not read for these events. At a first challenge 0 the programs divide by it; the written protocol does not.
   - **Running out of memory** stops the service.
   - **What a provider can do.** Tell a client "failed" only where `cp_prove` wrote it, or where `cp_verify`, in your check of a batch, rejected one of that batch's proofs or gave no verdict on it: then, once the hold is lifted, tell it to every request of the batch, in the same form as `cp_prove`'s own, so that the answer by itself does not show which of the two it was (at the vLLM integration the road to it does: 503 until a person lifts the hold). Stop the service with its `stop` file ([operators-duties.md](operators-duties.md)). That "failed" is a signal too: after a fault, whether a proof verifies can follow a hidden value (row HW-577), and whether the check gives a verdict can depend on load and time. It is the third of the causes named above. A "failed" that the provider itself decides on, for instance to shed load, or causes by terminating `cp_prove` with a signal, can depend on load and time, and therefore on hidden values (row HW-568).

8. <a id="volume-and-schedule"></a>**How much a provider has served, and when.** *Every client, and anyone who collects proofs.*
   - **Batches.** A commitment's leaf tables can be counted from its public files, and each proof names its batch's [leaf table](glossary.md#leaf-table). A full leaf table stops the service until the provider makes a new commitment ([limits](limits.md#the-limit-of-use)). So a stop or a new commitment shows roughly how many batches ran.
   - **Load.** Every client sees when batches run, and the time its own request waits for its batch shows how much other clients are asking for.
   - **Two commitments.** The same request sent under two commitments with the same configuration files gives two outcomes and next tokens, and where they differ, so do the weights. A provider that serves for long enough, or on a second GPU, needs a further commitment to the same weights ([how](operators-duties.md#a-new-commitment); [all the conditions](limits.md#a-further-commitment)). Above all, each commitment must sample its own secret seed, on a machine that was not cloned or restored. No test has searched two commitments together for leaks.

9. <a id="batch-mates"></a>**Who shared a batch, and how many.** *Any client, and anyone who holds two proofs.* One part of a proof file is identical in every proof of a batch, so two proofs show whether they shared a batch. A batch's time shows roughly how many requests it held.

10. <a id="lengths-and-places"></a>**Lengths, places in the batch and nonces.** *Anyone who holds the files.* The files show their lengths and a request's place in the batch (by design every length follows from public things alone; a proof's length hangs on which drawn positions fall together, and that has not been followed further: rows LIM-63 and HW-411; lengths that differed between runs and were not traced to that rule: row RES-61; the place: row HW-758). With [proof of work](glossary.md#proof-of-work), the files contain one nonce for each round in which work is counted. From reading the code, it is found by a search over bytes that the client sees, so the client could find it itself.
    - **Mask entries that anyone can work out.** In every proof some mask entries can be worked out by anyone, because they cover a public constant or nothing at all. If a bug made such an entry cover a second number by itself, everyone could read that number (row HW-581).

11. <a id="record-sums"></a>**Masked sums of the weights, public for good.** *Anyone who holds the public files.* `cp_commit` publishes, for each row of each weight table, a sum of the row's weights with public factors. Each sum is masked by random values of its own, its pads. The masked sums are meant to show nothing of the weights. Whether these programs achieve that has been checked only in part ([how](how-hiding-is-checked.md), test 1).
    - **What protects them, and what repeated random bytes cost.** Each is hidden only as long as the seed stays secret and ChaCha20, the cipher that derives the pads from it, is not broken, while the file is public for good. If the provider's random source repeats under other weights, as when an image is started twice, the two commitments' public files alone give, for every row of each table, one exact linear equation in the two sets of weights, for good: in the weights themselves, if one of the two sets is known. Keeping that from happening is the provider's duty.
    - **What checks it.** Neither the programs nor the repository's two audits compute a published sum a second time. Suppose that a fault loses a pad at every place where `cp_commit` uses it. Then only one [self-check](glossary.md#self-check) stands in the way, and that self-check looks at no published sum.
    - **On a model's files.** The programs take a model's files, at `answers 1`, the one value that they take; which runs have tried that: row HW-648 of [the table](limits-all-rows.tsv).

**Nothing beyond this list is meant to be visible. That is the design's goal: Lean's form of it, zero knowledge, is proved for the protocol, within bounds; the goal is not proved for the programs, and only partly tested** ([Weight hiding](../guide/weight-hiding.md); [the tests](how-hiding-is-checked.md)).

<a id="other-requests"></a>
## The other requests in a batch

This section is about other clients' requests. Hiding them is a different property from index privacy, whose definition takes every request as known to the verifier. Hiding the other requests in a batch is a goal of the design (row H11 of [the table](limits-all-rows.tsv)).

- **Digests.** A request can fail because of a range check, a wrong token or one of the few self-checks on data that fail a single request. It then leaves a salted digest of its state in the batch's other proofs. That state is covered by a hash and a secret salt, not by a mask, and what the digest reveals has not been worked out. The same holds for the digest that every request leaves there, served or failed.
- **Random bytes that repeat.** If a batch's random bytes repeat, a later client can test guesses at the requests that shared an earlier batch with it.

<a id="front-end-rules"></a>
## What a front end must do

These five rules spell out [duty 11](limits.md#the-conditions).

1. Pass on nothing that `cp_prove` prints, and not its exit code. Its last line, its alarms, the line that says how many requests a batch held and the line that says that not every step of the wipe ran can depend on hidden values or on other clients' requests, and so can anything an operator visibly does because of them. The same holds of what your own `cp_verify` prints at the check of duty 6: its last line, its rejection name, its exit status.
2. Limit how many requests each client may send, how many it may have waiting, and how many responses it may leave unfetched. A client that floods the service can steer who shares its batches (row H34).
3. Give each response only to its owner.
4. Show no one a listing, file timestamps or anything else about a request that is not theirs ([operators-duties.md](operators-duties.md)).
5. Under a score policy, ask for proofs only of the provider's own replies, over ranges that the front end sets.

These five are not all that a front end must do: first of all, no response may leave the provider before `cp_verify` has accepted every proof of its batch ([duty 6](limits.md#the-conditions)). Read also the nine steps of [the service document](service.md#what-your-program-must-do) and the eleven points of [the runbook](operators-duties.md#the-front-end).

**The example front end that ships, a vLLM middleware, breaks all five rules** ([its own page](../integrations/LIMITS.md) lists its limits; it does run `cp_verify`, once a job of a batch has been asked for, and hands on no proof of it before `cp_verify` has passed every proof of its batch (narrow exceptions, none open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)), but it does not see the service's exit code: [after a fault](../integrations/LIMITS.md#after-a-fault-nothing-stops-by-itself)):

- **Rule 1,** at least in part, from reading its code: every client gets the same error once the service is on hold, and that error tells every client that the policy's proofs are on hold; the order of a job's states before it can tell one cause of a stop from another (inferred from the code, not tried). (More: row HW-563.)
- **Rule 2.** Its limits are per policy, not per client.
- **Rule 3.** It has no notion of clients, so it hands a proof to anyone who has the proof ID and has the shared key where vLLM was given one that the middleware can read ([the operator's checklist](../integrations/vllm/REFERENCE.md#the-operators-checklist)). Where vLLM has a key that the middleware neither reads nor suspects, the proof ID alone is enough. Count on no warning: try one call without a key. Setting `VLLM_API_KEY` closes that. That it asks for that key, and that the variable closes the rest, is by reading, and by tests with a stand-in for vLLM's application (`integrations.unit` in SUITE.md; what has run: rows H11.2 and HW-275).
- **Rule 4,** at two thresholds on other clients' load: a full queue and a full store.
- **Rule 5.** It lets a client move both ends of the checked range.
- **Besides, it filters nothing that vLLM itself serves,** including the floating-point model's scores.
- vLLM, behind the example front end, shows more of its own: where the model's directory lies on the provider's disk, a fingerprint by which a client can test a guess of that path and of the settings, and, without a key, all clients' traffic ([integrations/vllm/integer_model/OPERATING.md](../integrations/vllm/integer_model/OPERATING.md#what-a-client-can-do), "What a client can do"; [integrations/LIMITS.md](../integrations/LIMITS.md#smaller-limits)).

**A failed request that comes again.** Do not rely on holding back a failed request that comes again, that is, on making it wait at the front end before it is passed on: the written proof leaves out a failed request that is sent again, at once or later, and time still decides who shares a batch (row HW-553).

<a id="public-weights"></a>
## The option `-public-weights`

This `cp_commit` option is for weights that are already public. **With this option, nothing is claimed to be hidden:** not the weights, and not one client's request from another client in the same batch.

- **Why.** The seed comes from a 32-byte file, and under the option `cp_commit` puts no other randomness into its files. Anyone who has that file can work out every mask used for the key file, the [leaf files](glossary.md#leaf-file) and the table columns that proofs open. So the weights are not hidden. Each batch still gets fresh secret random bytes, but no argument and no test covers what they protect once the seed is public.
- **Other people's requests.** Serve them one request per batch, or not at all. No program enforces that.
- **No one can tell.** The public files, the commitment ID, proofs and responses carry no mark of the option. So no client or verifier can tell such a commitment from an ordinary one, and a provider can pass one off as ordinary.
- **What guards against a mistake.** Such a commitment needs the option on the command line and, for a model's files, a marker file beside each, and no client can trigger it. This guards against some mistakes but not against intent (rows R30 and HW-479).
- **How far it is tested.** By few runs, none of them a test of hiding: rows M1.4 and HW-649 of [the table](limits-all-rows.tsv).

† Not on the released programs: [which kinds of result these are](limits.md#on-other-programs).
