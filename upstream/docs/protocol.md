# The protocol

This page explains how the proof system works: who sends what to whom, in the order it happens in a run, and why each step convinces a verifier or is meant to protect the weights. It assumes you know what a hash function and a polynomial are. Sections 1 to 4 are the short version.

It describes the protocol that the C++ and CUDA programs are written to run. The protocol as written in Lean is in the paper and in the proofs of section 8, and the arguments of section 7 apply to the form of it in a technical report (not public: [row HW-763](limits-all-rows.tsv)). That the programs run the protocol written in Lean is tested by runs of both verifiers, not proved. What is proved, of the protocol and of the programs, is described in [guide/what-is-proved.md][status]. When sections 4 to 6 say that a value is masked, salted or looks random, they describe the design: the verifier does not check it ([Limits][limits-risk]).

- **The format.** [The format][format] specifies the files byte by byte, the hash inputs and the verifier's tests, and lists what it does not specify (format §17). Read it if you want to write a second verifier. On this page, a number like **R4.11** refers to one of its rules, and **format §17** to one of its sections. A bare number like **6.3** refers to a section of this page.
- **The paper.** [The paper](../paper/paper.pdf) describes the protocol as written in Lean, prints proofs that follow the Lean proofs, and marks which statements are checked in Lean and which are not.
- **The Lean proofs.** Section 8 says where to read them.
- **Sources.** The proof system builds on a paper and on published methods. [PUBLISHED-WORKS.md][published-works] says which, part by part.

To check a proof, see [Verifying a proof][verifying]. To run a provider, see [the quick start][quickstart] and [the operator's duties][duties].

## 1. The problem

A *provider* runs a language model and wants to keep its weights secret. A *client* sends a prompt, receives generated tokens, and wants to know that they agree with one fixed model, not with a cheaper or altered one.

The provider *commits* to the weights once, using the *committer* (`cp_commit`). The commitment has a 32-byte fingerprint, the *commitment ID*. From then on the *prover* (`cp_prove`) can attach a *proof* to the provider's output. Both programs run on a GPU. The client's *verifier* (`cp_verify`) runs on a CPU and checks the proof against the commitment ID and the files published with the commitment.

The protocol has two goals. They say what the design aims at, not what is proved: Section 7 sketches the argument for each, and [What is proved][status] says how much of each is proved, of the protocol and of the programs.

- **Soundness with binding.** The commitment is to fix one set of weights, and a proof of a false claim about them is to be accepted only with small probability, whatever the provider does within a bounded number of hash queries.
- **Weight hiding.** The public files and the proofs are to reveal nothing about the weights beyond a declared list ([What is not hidden][not-hidden]). The cryptographic property nearest to it is zero knowledge, in the form called [index privacy](glossary.md#index-privacy).

## 2. What a proof claims

A proof is about an *integer model*: a model in which every operation is defined exactly on integers. A proof never refers to floating-point results, so the claim has no error margin.

A *request* asks for a proof, not for text, and in ordinary use it is built after the text exists. It consists of a sequence of exactly T tokens and a range of positions, with T fixed by the configuration.

The range determines which tokens are checked. Tokens and positions are numbered from 0, and position i checks token i + 1. If position i is in the range, token i + 1 is a *checked token*: it is checked as the token that follows tokens 0 to i. Take a prompt of 3 tokens (tokens 0, 1 and 2) and 2 generated tokens (tokens 3 and 4). Position 2 checks token 3 and position 3 checks token 4, so the range is positions 2 and 3. In general, for a prompt of n tokens and m generated tokens, with n + m at most T, positions n - 1 to n + m - 2 check all the generated tokens. A client that uses the generated tokens' own positions instead leaves the first generated token unchecked.

Positions that the text does not fill are padded with arbitrary tokens. A proof matches exactly one request, including its padding, so both sides must build the same one ([Verifying a proof][verifying-request]).

For the committed weights, a proof claims three things, each only "as far as the sub-proofs of its segments check a model's steps" (the words of the format's page on a proof file; 6.3 divides the model into segments, and a [sub-proof](glossary.md#sub-proof) is the argument for one segment):

1. When the integer model is run on the request with the committed weights, every value that it computes stays inside the lookup table or the range of numbers for which its step is defined. (The format's words: "No value of the integer model's run on the request, by the weights in a stored table, leaves a table or a range at any position.")
2. The checked tokens meet the acceptance policy.
3. If the last position, T - 1, is in the range, the next token is the model's top choice there, under every policy.

The *top choice* at a position is the highest-scoring token or, in a tie, the one with the lowest token number. The *next token* is the token that the model would put after the whole sequence of T tokens. It matters only when the last position, T - 1, is in the range, as for a text of T + 1 tokens: the client then supplies the last of them as the next token. Otherwise it is 0 and means nothing ([Verifying a proof][verifying-request]).

The *acceptance policy* is part of the configuration, so the commitment ID fixes it. The default is *exact match*: each checked token must be the top choice, as in greedy decoding. Sampling is not checked under any policy, because a request carries no random seed.

The protocol does not prescribe how the text is produced, and a provider may generate it with an ordinary floating-point engine. Such text may differ from the integer model's top choice at some tokens, in which case exact match fails. Of 96 stored honest replies each from a stock inference engine at 16 bits and at FP8 (up to 500 tokens each; scored, not proved)†, none meets exact match at every token. For such text there is one score policy, the *clipped gap* (format §12.5).

A token's *[score gap][glossary-gap]* is how far its score lies below the best score at its position, under the committed model, plus 1 if the top choice has the lower token number; its *[cost][glossary-cost]* is that gap plus a constant. The clipped gap bounds the average of the checked tokens' costs, with each cost counted only up to a clip `C`, and it bounds how many tokens may cost more than `C`. These are two bounds on all checked tokens together: one token's high cost can be offset by the others' low costs, as long as both bounds still hold. As many tokens as the second bound allows may cost more than `C`, and these can be any tokens, at any positions. With loose bounds, acceptance therefore says little or nothing ([Limits][limits-policy]).

The commitment ID fixes the policy's numbers, but the provider chose them, and a client that has not read the configuration does not know how loose they are. So under the clipped gap the client must state the numbers that it expects, and the verifier accepts only if they match the configuration's. A call that checks only the next token does not need them.

**In Lean.** Under exact match, Lean has a soundness theorem for the Lean verifier, `verify` ([at length][status]). It is about `verify`, not about `cp_verify`, the C++ verifier that users run.

**Check the policy yourself.** The soundness theorem on `verify` covers exact match only: the lines [`C`, `theta` and `cap`](glossary.md#c-theta-cap) of the configuration's parameter file read 256, 256, 0. That is a premise of the theorem, and, as far as reading the code shows, `verify` has no test for it. So a user, or the program that calls `verify` for them, must check these three lines. An accept under another policy is outside this theorem ([lean/THEOREM.md](../lean/THEOREM.md), "Its reach").

[Limits][limits-policy] says more about what acceptance under the clipped gap means. For such policies Lean has a further theorem; by itself it bars nothing against a committer who commits to weights under which every output meets the policy ([lean/THEOREM.md](../lean/THEOREM.md), "The theorem under a policy: other acceptance policies").

## 3. What an accepted proof means

An accepted proof means that every test in the format passed for the commitment ID, the request and the next token that the client supplied. This is meant to show two things: the commitment fixed one set of integer weights, and the claim of section 2 is true for those weights. [What is proved][status] says how much of that is proved.

Acceptance does not tell you:

- **Whose weights they are,** or whether the model is any good. A provider can commit to any model, and it writes the configuration that defines the model together with the weights. Get the commitment ID before you send the prompt, from the party that vouches for the model, not through the server you are checking; best from where the commitment is published. Never take it from files that arrive with a proof.
- **Anything about tokens that are not checked.** With the same tokens and next token, the claim can be true for one range and false for a wider one. Build the request yourself, from what you sent and received, and choose the range yourself.
- **How close the integer model is to the floating-point model** it was made from. That depends on the public quantization scales, the factors that map floating-point values to integers.
- **Whether the reply is complete.** The verifier knows nothing about end tokens or the tokenizer, and neither the commitment ID nor a proof covers them: a reply that was cut short verifies as a shorter reply. Get the model's end-token ids and its tokenizer from where the model is published, and check the end of the reply yourself ([Verifying a proof][verifying]).
- **Whether the provider hides the weights.** A commitment made with `cp_commit`'s `-public-weights` option hides nothing, and neither its public files nor its proofs give any sign of it (section 7). The same applies to a commitment that a [test build](glossary.md#test-build) made from a seed in a file. Nor does the verifier check for masks: a proof that has lost one through a bug or a hardware fault can still be accepted ([Limits][limits-risk]).
- **Under the score policy, how the text was produced.** You learn only that the checked tokens score within the policy's bounds.

An accepted proof is only as good as what the client supplies itself: the commitment ID, the request with its range, the next token, the policy's numbers if it names a policy, and a verifier it built itself. [Verifying a proof][verifying-endings] explains which of the verifier's outcomes count as acceptance.

## 4. Overview

1. **Commit** (the *offline phase*). The committer encodes the weights, with random elements added, as a large *weight table*. It hashes the table into a Merkle root and proves in a *key file* that the table is well formed. The key file is public: it contains no key, seed or counter. The provider's private *secret key directory* holds its secrets (a seed, and fresh bytes for each leaf table), one level of each table's hash tree and the counters (6.3). The tables themselves are not stored: in each batch, `cp_prove` recomputes the parts it needs from the model's weight files and compares them with the stored level.
2. **Copy.** Every batch of proofs reveals a few columns of a table, which uses up some of the table's randomness. A batch counts as soon as it starts, whether or not it is full and whether or not it succeeds. So the first table, the *anchor*, is copied. A copy, called a *leaf table*, holds the same weights, encoded with new randomness of its own. (The name has nothing to do with the leaves of a Merkle tree, which this page calls tree leaves.) Its public *leaf file* holds a *link proof* that it encodes the same weights as the anchor. Batches run on leaf tables, never on the anchor (R14.2).
3. **Prove** (the *online phase*). The prover handles requests in *batches*. Working through the model piece by piece, it reduces each request to one claim about the weights. One test for the whole batch opens a few columns of one leaf table, and a final step ties each request's claim to that test. Each request gets its own *proof file*.
4. **Check.** The verifier checks the key file, the leaf file and the proof file in one call.

**The configuration** consists of three public files: proof parameters, layout and quantization scales. In practice the provider writes it and publishes it with the commitment. The commitment ID is a hash of the configuration, of the anchor's root and shape, and of the anchor's out-of-domain (OOD) answers (masked sums of the weights) with their OOD seeds (5.5) (R4.9, R4.11), so a client that holds the right commitment ID cannot, by design, be given a different configuration. The soundness theorem on `verify` is stated at given public files and does not itself say that one ID cannot verify under two sets of them, so the client holds the three configuration files as it holds the ID.

**Names.** Some names on this page are the project's own. A *weight table* is a committed table: the *anchor* is the first table, a *leaf table* is a copy of it, and a *leaf file* is the copy's file. The *range check* (6.3) is an approximate range proof (not the range proof of 6.1). A committed table's *OOD sample* (5.5) has two parts: the *OOD seeds*, which are the verifier's message (each a point and a factor for the mask, derived by hash; R4.8, R6.9), and the *OOD answers*, which are the committer's reply. The OOD seeds are not the provider's secret seed (step 1 above): anyone can compute them. Every parameter file has the line `record 1`, which stands for that sample, and the format says "record" for the OOD answers and "stored table" for a weight table, which is committed once and read by many runs, though the programs recompute the parts that they need (section 9; [the glossary](glossary.md#record)).

**A commitment wears out.** In the full-size example configuration a batch can hold up to 64 requests (not to be relied on with an 80 GB GPU: 6.3, "Failures"; the figures are on [guide/performance.md](../guide/performance.md)), and in the default setting (5.6) one leaf table serves at most 688,127 batches (the programs' own accounting, not a security level). Before each batch, `cp_prove` looks at the counters of all the leaf tables, and it stops as soon as any one of them has no room for another batch. That way, whether it stops does not depend on which table was picked. So the provider needs a new commitment before `cp_prove` stops ([A new commitment](operators-duties.md#a-new-commitment)). Leaf tables cannot be added later (6.2), so decide how many the commitment will need over its whole lifetime.

A new commitment to the same weights gives those weights a second key file. The design then needs conditions that no program checks. Among them: the new commitment has a seed of its own, drawn independently, and counters of its own, it keeps the same configuration, scales included, the weights were fixed beforehand, and no counter is ever copied or restored ([the full list](limits.md#a-further-commitment)).

A new commitment does not repair anything: if a table has served more batches than its limit allows, nothing that section 7 says about hiding applies to those weights.

## 5. Tools

All values are numbers modulo a fixed prime of about 64 bits (a *field*). Random challenges come from a larger field built on it.

**5.1 An encoding that hides** (a zero-knowledge encoding). A row of the weight table starts as a vector of numbers that represent weights, together with random elements. The vector defines a polynomial, and the encoded row lists the polynomial's values at many more points than the vector has elements (a Reed-Solomon code). Different vectors give rows that differ in a large fraction of positions (positions of the encoded row, not token positions), yet any *t* values of an encoded row look random, whatever the weights are. The number *t* is the *query bound*: it is at most the number of random elements, and roughly equal to it. A *column* consists of the values of all rows at one position. This page uses the word only in that sense.

**5.2 Hash trees.** To *commit* to encoded rows, the prover sends the root of a Merkle tree with one tree leaf per column. Later it *opens* chosen columns by sending them with their paths (R7.8). A random salt in each tree leaf stops anyone from guessing an unopened column and checking the guess (R7.3).

**5.3 Reductions.** A *sumcheck* reduces a claim about a sum of millions of terms to a claim about the polynomials that define the terms, at one random point. A *lookup* shows that every element of a committed vector occurs in a public list. Together they reduce "these committed numbers are a correct run of the model" to a few *linear claims*, each saying that a known combination of a committed vector's elements equals a known value. Random masks hide the values sent along the way.

**5.4 Proximity tests.** The verifier sees only a few columns, so it needs to know that what was committed is *close* to a *codeword*: that it differs from the encoding of some vector in less than a fixed fraction of the positions. A *proximity test* shows this. The prover combines the rows, with random factors, into one row (a *fold*). It splits that row into shorter rows and commits to them under a shorter code (*code switching*), and the verifier checks the new commitment against the old one at random positions. This repeats until what remains is short enough to send in masked form. The test also proves linear claims about the vector: they are batched into one claim, the folds carry it along, and the verifier checks it at the end against what was sent.

**5.5 Out-of-domain samples.** When a test allows a large distance, committed rows can be close to several codewords (*list decoding*). To pin one message down, the verifier samples a random point, with several coordinates in the larger field, after every committed message except the mask of a step (R12.10). The prover answers with one masked element for the whole committed message: the message's entries are the values of a multilinear polynomial at the points whose coordinates are all 0 or 1, and the element is that polynomial's value at the sampled point. Except with small probability, all nearby codewords that fit the answer encode the same message, and every later step refers to that message. The tests allow distances up to the *Johnson radius*, at which several codewords can be close, so the samples are needed. The weight tables get a sample too: the point is derived by hashing the table's root, and the committer publishes one masked answer per row with the root, in the key file or in the table's leaf file. These answers are the table's *OOD answers* (R4.8, R13.8).

**5.6 Hash chains.** A proof is a file, so there is nobody to send challenges. Instead the prover keeps a running hash, a *chain* (the Fiat-Shamir transformation; R4.6, R5.2). Each challenge is the hash of the chain's previous value and the prover's latest message, so the prover cannot learn a challenge before it has fixed what the challenge tests. A *round* is one message from the prover together with the challenge that follows it. The prover's messages in one chain, written one after another, make up part of a file, and this page calls that part a chain too. Below, "the verifier samples" is shorthand for "the next hash of the chain gives".

A chain starts from a hash of its *input*, which has three parts: the parameters, with a digest of the configuration; 32 random bytes, which the prover derives in the same way as its other randomness (section 7); and the items that state what is being proved. Section 6 names only the third part. Where that part holds a weight table's root, it also holds the table's OOD answers (R4.6). Each chain also has a *label*: an integer that says which protocol the chain runs and, where two kinds of file run one protocol, in which kind of file. The label goes into the hash that gives the chain's first value and into the hash of every round (R4.6, R5.2).

A chain that later chains build on ends in a digest, its *seal* (R6.8). A parameter file can also require *proof of work* before chosen rounds, which makes it expensive to retry a round for a luckier challenge (R5.5). The parameter files under `prover/examples/` all state one *setting*, the default setting. It requires proof of work before every round that samples positions. A parameter file can differ from it, for instance in how much proof of work it counts, and `cp_verify` does not hold it to this setting. This page does not describe such files, nor what an acceptance rests on under them: read the file ([Which setting is it?][verifying-setting]). "The target" (6.4) says what an acceptance rests on, and section 7 what the OOD answers mean for hiding.

## 6. A run

### 6.1 Commit, once

1. **Encode.** The committer writes all weights as small signed digits and arranges them in one matrix M, the *weight matrix*. The arrangement follows a fixed public rule, which uses only numbers from the parameter file and constants in the programs. M[i, j] is the digit at index (i, j). The index j is divided into *blocks* of consecutive values, and each row of the weight table holds one block: the digits M[i, j] for every i and every j in that block. (So a row of the table is not a row of M.) The committer adds random elements to each row, encodes the rows and builds the tree. (It derives these elements from the secret seed; section 7 describes how the programs derive their randomness.)
2. **Range proof.** Lookups show that the digits are in fixed small ranges, and that the weights assembled from them are too. For the embedding, however, only the digits are checked. Without this step a committer could use huge field elements, for which the model's equations hold only modulo the prime.
3. **One-time test.** The committer proves that the table is close to a codeword, and that the range proof's claims are true for the encoded matrix. It uses the procedure that every batch uses later, the *test on the table* (6.3), and runs it once on the anchor.

The key file holds the root, the OOD answers and these proofs, in the form of a proof file (6.3): a first chain, which holds the range proof and states the claim about the table, then a slot, a shared chain and a second chain. Its three chains have labels of their own (5.6): none of them is the label of a chain of a proof file (R13.4). A valid key file is meant to fix one weight matrix: except with small probability, the one encoded in every close codeword that fits the OOD answers (5.5). ([What is proved][status] says how far this is proved.)

The binding of the weights itself rests on the theorem on codes (Theorem 4.6 of the full version, ePrint 2025/2055, not in the proceedings, of the paper by Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, STOC 2026), which the programs' accounting takes as given. A statement on codes held to follow from that theorem, in its case of two words, is proved in this repository's Lean library, by way of another formalization's theorem and a comparison of bounds; the paper's own form of the bound is not proved there, and that the library's statement follows from the paper's, and that these programs' parameters meet the theorem's conditions, rest on reading, not on a proof ([lean/LIMITS.md](../lean/LIMITS.md), L15). The theorem is used with no bits to spare ("The target", 6.4; [what the theorem assumes](limits.md#what-the-theorem-assumes)).

Binding is meant here in a modified sense, in which the hash itself supplies the OOD seeds (section 4; 5.5). The written argument for that sense (section 7) covers a commitment ID formed without those seeds, not the ID formed here, which hashes them (R4.11), so this page claims no binding of the weights for the programs. Two theorems in [the paper](../paper/paper.pdf) cover an ID that hashes the seeds: "Extraction and extractability for the explicit committer", for a committer that models part of `cp_commit`, and "Soundness with extractability, openings and extraction", for the committer compiled from the Lean provers. Nothing is proved of `cp_commit` itself.

What Lean proves — of `verify`, that for any commitment, whoever made it, one set of weights or none is settled before any proof is made; and, for each of the two Lean committers just named, that those are the weights put in and that the commitment cannot be opened to others — is in [What is proved][status] and [lean/THEOREM.md](../lean/THEOREM.md#what-the-commitment-gives-as-proved), "What the commitment gives, as proved".

### 6.2 Link each leaf table

In the same call, the committer encodes the matrix again with new randomness, as many times as the provider expects to need. Leaf tables cannot be added later.

For each leaf table, the committer folds the rows of the anchor and the rows of the leaf table with the same random factors, and commits to both folded vectors. Then it opens random columns of both tables. A proximity test shows two things: each folded vector fits the opened columns of its own table, and the two folded vectors agree at a random point in the part that holds the weights. That is enough, because different matrices give different folds for all but a small fraction of factors, and different folds disagree at almost every point. The test also shows that each folded vector fits its table's OOD answers, folded with the same factors.

Each link opens columns of the anchor too, so the anchor's query bound limits the number of leaf tables. The committer counts these columns as it runs.

### 6.3 Prove a batch

The prover runs as a service. It runs no verifier on what it makes: the provider's [front end](glossary.md#front-end) must do that before any [response](glossary.md#response) of a batch leaves the provider ("The check before anything leaves", near the end of this section). While that check runs, the prover can prove the next batch.

The prover prepares a batch in three steps.

1. It collects requests into a batch and picks a leaf table at random.
2. Before it opens anything, it adds the number of columns the batch will open to that table's counter on disk. It opens nothing until the disk reports the count as written. So every column is counted before it is opened, provided the disk keeps what it reports as written ([the operator's duties][duties]).
3. It runs the integer model on each request.

What happens when something fails is described at the end of this section ("Failures").

Each served request gets a proof file with five parts, in this order:

1. a header, which names the leaf table
2. the first chain
3. the request's slot
4. the shared chain
5. the second chain

The first and second chains are the request's own, whereas the shared chain is byte for byte the same in every proof file of the batch.

**The first chain** reduces the request's claim to one equation involving the weight matrix M. Its input is the leaf table's root and number, an item that holds a digest of the request, and, last, the request itself, byte for byte: its tokens and its range (R14.3).

The model is divided into *segments*, and for each segment the prover commits to the intermediate values, the *trace*. Sumchecks and lookups then reduce the segment's equations to two kinds of claim: linear claims about what was committed, and claims about the weight table. The request's tokens enter the equations as public values. The last segment is the output head, whose equations state that the checked tokens meet the policy and, if the last position is checked, that the next token is the top choice. The next token is sent in the clear.

These equations are proved modulo the prime, but the claim is about integers, so each segment includes a [range check][glossary-range] (the programs' output calls it `smallness`; it is not the range proof of 6.1). After the segment's trace is committed, the verifier samples factors, each -1, 0 or 1. A lookup of digits shows that the trace's values, combined with these factors, are small. Except with small probability over the factors, the values themselves are then small too, within a looser bound. Given that, and under conditions on the sizes of the model's numbers, equations that hold modulo the prime also hold over the integers. `cp_verify` tests the public parameters for conditions of this kind (its *no-overflow conditions*). A lemma in the technical report of section 7 says that such tests imply the report's conditions, with exceptions. It is known only from reading the source that `cp_verify`'s tests are the ones the lemma refers to, and it is not proved that the programs' conditions are enough for the programs' equations ([Limits][limits-compiled]).

The prover folds each segment's commitments into one short commitment and carries it into the next segment, so it need not keep the whole trace. A proximity test is run on the last short commitment.

At the end of the first chain, all the claims about the weight table are batched into one:

g · M · e + μ = v

Here g · M · e is the sum over all i and j of g[i] · M[i, j] · e[j]. The verifier knows the factors g and e and the value v. The term μ is a known combination of masks that the prover committed to at the start of the chain. The prover now commits to the *request vector* w = g · M, with one element per j. The chain ends in a seal.

**The slot.** The shared challenges must not be sampled until every request has fixed its claim. So the seals of the batch's requests become the tree leaves of a small salted Merkle tree, the *batch tree*, and the tree's root goes into the shared chain's input. A request's *slot* is its position in that tree. A proof file holds its slot, the salt of its tree leaf and the path. Requests are placed in random slots, and unused slots hold random digests. This is meant to keep the batch's size and the other requests out of a proof file's bytes, but the time a batch takes reveals its size. A request that fails keeps its own tree leaf. Like every request's tree leaf, it is a salted hash of a seal that depends on the weights, and it enters the batch's other proofs: it is covered by a hash and a secret salt rather than by a mask ([What is not hidden][not-hidden-others]).

**The shared chain** solves a problem that all requests share. Every request makes a claim about the same huge matrix M, through its own vector w, and w = g · M must be checked against the committed table. Opening the table once for each request would use up the table's randomness much faster. So the batch opens the table once, for all requests.

The verifier samples one new random point, the same for all requests, and a fixed formula turns the point into factors e\*, one per j. The factors e were derived from a point in the same way. If some w is not g · M, then w · e\* differs from g · (M · e\*) for all but a small fraction of points. So one vector, the *shared vector* M · e\*, serves every request. The shared chain proves that the shared vector is correct. After that, two equations involving each request's w are enough, and the second chain proves them.

The shared chain's input is the root of the leaf table and the root of the batch tree; the batch tree's root also takes the place of the random bytes (5.6).

The proof that the shared vector is correct follows the table's layout. The formula makes each e\*[j] a product of two factors: one for the block that j is in, and one for where j lies within that block. Recall that each row of the table holds one block. The chain has five steps.

1. The prover folds the table's rows with the block factors and commits to the result, the *folded vector*.
2. The verifier samples positions, and the prover opens the table's columns there.
3. Anyone can fold an opened column with the block factors, so each opened column gives a linear claim about the folded vector. These claims tie the folded vector to the table. The OOD answers, folded the same way, give one more.
4. The prover derives the shared vector from the folded vector, using the factors for the offsets within a block, and commits to it. This is a linear relation between the two vectors.
5. Proximity tests prove all these linear claims about the two vectors.

The chain ends in a seal. It is the only step in a batch that opens the leaf table. The shared chain and a request's second chain together make up *the test on the table*. The one-time test of section 6.1 is the same pair of chains, run once on the anchor, with parameters and labels (5.6) of its own. The two are not interchangeable, because a batch's test is sound only for a table that has already passed a first test: the anchor passes it in 6.1, and a leaf table in its link proof (6.2).

**The second chain** ties the request's claim to the leaf table. Its input is the two seals. A proximity test shows that two equations hold: w · e + μ = v, as the first chain claimed, and w · e\* = g · (M · e\*).

**Release** is the last step. `cp_prove` writes a batch's responses under temporary names and then *releases* them: it gives them their final names. It runs no verifier. So after a bug or a hardware fault it can release a response that `cp_verify` rejects, and from reading the code such a response can hold unmasked values. `cp_prove` does not notice, and goes on serving until one of its self-checks on secrets fails ("Failures", below) or something else stops it.

**The check before anything leaves.** The provider's front end must therefore run `cp_verify` on every proof of a batch before any response of that batch leaves the provider. If `cp_verify` rejects a proof or gives no verdict (rarely it does so on an honest proof: 6.4), no response of the batch may leave, and after a rejection the provider must stop serving proofs on that machine until it knows the cause ([duties 6 and 8](limits.md#the-conditions)). The vLLM integration in this repository makes this check (narrow exceptions, none open to a client: [integrations/LIMITS.md](../integrations/LIMITS.md)); a front end of your own must do the same. A shipped test has the integration reject one changed proof and hold it back, as a batch of one response that the test writes itself, at small size, with no vLLM and no `cp_prove` running; in the integration's own check, the real `cp_verify` has never been given a false proof at full size, in a batch of several responses, or in one batch among several. The check does not see a lost mask in a proof that verifies ([Limits][limits-risk]).

**Failures.** Three kinds are described here, and [how a run ends](../prover/docs/ends-in-short.md) describes other endings (what a failure reveals: [What is not hidden][not-hidden-failures]). "The service stops" means that the `cp_prove` process exits.

1. **One request fails.** It is answered "failed", and the rest of its batch is still served. This happens when the request's claim is false, a check that concerns only that request fails, or a challenge in the request's own chain takes a rare value. So "failed" does not tell a client that the claim is false.
2. **A whole batch fails.** Every request in it fails. This happens when the cause is not tied to any single request, or is in the shared chain, the part of the proof that all the requests in a batch share (above). Most failures of this kind also stop the service, for example a fault in the secret key directory.
3. **A self-check on secrets fails.** A *self-check* is a check that the prover runs on its own work. The self-checks in question test whether secret masks, salts or keys are what they should be. A failure of this kind always stops the service, whereas a single failed request does not. These self-checks cover only some of the random values ([Limits][limits-lost-mask]).

This account of what happens at a failure, above and below, comes from a brief reading of this repository's sources and their comments, in the service's own files only: which checks raise which failure was not traced into the GPU code or the proof library; how many failures of the second kind stop the service was not counted, and the self-checks of the third kind were not examined one by one. In a batch with a failed request, the other requests' files have not been examined. No run of a [production build](glossary.md#production-build) that shows a failure of the third kind is known.

At full size a batch of 64 is not to be relied on with an 80 GB GPU. On a real model's public weight files, three runs of the service at 64 requests a batch all made every proof (960 of 960), with up to 80,629 of the GPU's 81,559 MiB in use. On synthetic weights of the same shapes every known attempt at 64, six calls of the prover on 64 requests and two services, ended with a failed call to CUDA, with no proof for any request: eight attempts, six of them not on the released programs†. The cause is not known, and the two kinds of job differ in more than their weights ([Limits](limits.md#what-the-programs-can-serve)).

After a failed self-check on secrets:

- Nothing more is released, apart from a batch whose responses are being released at that moment, the delay in the last point below, and a short stretch at the start of the process's exit, in which a response can still get its final name ([how a run ends](../prover/docs/ends-in-short.md#the-three-exit-statuses)). That holds for the failing batch and for later batches, not for what was already released, which by default includes the batch before; the provider must not pass any of it on until it knows the cause ([duty 8](limits.md#the-conditions)).
- The affected requests are answered "failed" only after the service has been restarted.
- But a few of these checks do not stop the service immediately. It stops only the next time a batch ends. For those few, release stops only after the service has seen the result of the failed check.

### 6.4 Check one proof

The verifier takes the configuration, the commitment ID, the key file, the leaf file that the proof names, the request, the next token, the proof file and, unless the client expects exact match, the policy it expects ([Verifying a proof][verifying] gives the command). The format lists what the verifier does, in order, and names every reason for rejection (format §13). The checks fall into ten groups:

| check | what it stops | format |
|---|---|---|
| Recompute the commitment ID from the configuration and the key file. Compare it with the client's | another model, policy, scales or table | R4.11; format §13.5 |
| Take sizes, rounds and error bounds from the configuration and the verifier's code, never from the key, leaf or proof file. Hold every round to the target (below) | a file that sets its own, weaker parameters | R0.2, R4.9; format §8.3, §13.6, §14.3 |
| Allow one valid reading of a file: no byte is left over, and field elements are below the prime | two readings of one file | R0.1, R1.3, R8.1 |
| Recompute every challenge from its chain. Test every proof of work | challenges that the prover chose | R5.2, R5.5 |
| Hash every opened column up to its root | answers that change after the commitment | R7.10, R7.14 |
| Key file: check the range proof and the one-time test. Leaf file: check the link proof | "weights" that overflow; a table that fixes no matrix; a leaf table that holds another matrix | R13.10, R13.18, R13.19 |
| Segments: follow the sumchecks, lookups and range checks. They leave linear claims, which the next proximity test settles | an incorrect run of the model; tokens that fail the policy; equations that hold only modulo the prime | R14.4; format §13.9 |
| Check the out-of-domain samples, the proximity tests and the test on the table (shared and second chain) | commitments that are far from the code or fit two vectors; claims about weights other than the committed ones | R13.19, R14.9, R14.10 |
| Recompute the seals, the tree leaf of the request's slot and the root of the batch tree. Hash them into the next chain's input | parts of different runs spliced together | R7.18, R14.7 |
| Compare the proof's next token with the client's. Put the request itself, and a digest of it, into the first chain's input | a proof of another request or next token | R14.3, R14.8, R15.5, R15.6 |

**The target.** The parameter file sets a *target* for a round's error, and the verifier accepts only if that target is at least 128 bits. (A target of 128 bits means that each round's error must be at most 2^-128.) The verifier's own code computes each round's error, and it counts the round's proof of work when it does. Every round's error, computed that way, must meet the target.

By the programs' own calculation, at the default setting and full size, every round of a [planning call](glossary.md#planning-call), the key file's and a leaf file's as well as a proof's, reaches the target of 128 bits. Of these rounds, 71 reach it only when proof of work is counted, which is assumed for the programs' hash chains: up to 28 bits in a request's proof, 20 in most of its rounds that have any, and 32 in the rounds of the files that are made once. Without that work the weakest round has 96.23 bits. The call is `cp_verify -plan accounting` at the default's parameter file.

The same planning call prints a lower reading of the same rounds: the whole bound as one number, 126.21 bits with the work counted and 96.23 without (the chance of a false acceptance is below about N/2^126, or N/2^96, where N counts all the hashes computed, the verifier's own included). [The table](limits.md#the-soundness-figure) has two more readings of the same rounds, lower still: the lowest is 108.32 bits with the work counted and 76.55 without.

All 341 [closeness rounds][glossary-closeness], the key file's and a leaf file's as well as a proof's (6.1), rest on the theorem on codes, which the programs' accounting takes as given; a statement on codes held to follow from it is proved in the Lean library (6.1; [what the theorem assumes](limits.md#what-the-theorem-assumes)). They also rest on further steps argued in the technical report of section 7, and 324 of them reach the target by less than a tenth of a bit (counted in `verifier/tests/known_answers/kit_rounds/target_full_f2.txt`). That the parameters meet the theorem's conditions is the project's own assessment, with no bits to spare. The truth of an accepted request and the binding of the weights both rest on that theorem, and at the default setting both lean on counted proof of work in the rounds that draw positions. A round's bits are not the chance of a forgery: that chance grows with every hash computed, the verifier's own included, and even the figure without the work rests on results that are cited, one of them taken as given, and on the hash taken as a random oracle.

These figures are a calculation, not a security level, and no theorem is about the programs ([Limits][limits-figure] says what they are worth). For the programs, no security level is claimed or proved.

**The soundness theorem on `verify`: its own bound** (section 2). The error is at most about 16·(q + 2^23)/2^128 after q hash queries, the committer's and the prover's added up (the 2^23 is a fixed allowance for the verifier's own queries), with the format's hash function modelled as a random oracle. That is about q/2^124 + 2^-101: 124 bits per hash query, and the bound is never smaller than about 2^-101. The bound is for one commitment, one prompt, one output, one proof and one call of `verify`; for many proofs that one prover hands in at once under one commitment there is a corollary, and for separate sessions no theorem states it. Three further theorems bound the error of every round by 1/2^128, at public files in scope where `verify`'s own tests on them pass; they bound single rounds, not the chance of a forgery. Proof of work is counted: without its 20 to 32 bits, the rounds that sample query positions have errors of about 2^-96 to 2^-110 (added up by a program, no theorem). Neither figure, 124 bits per query or 128 bits per round, is claimed for the C++ programs. [lean/THEOREM.md](../lean/THEOREM.md) has the conditions.

Whether to count the work is up to you. If you do not count it, pass `-no-proof-of-work` to `cp_verify`: a call in which some round meets the target only with its work then does not accept; at the default setting and full size, that is every call. Without the option, the last line of the output and the exit code are the same whether or not work was counted, and they do not show what the error bounds rest on either: read the lines that [Verifying a proof][verifying] describes.

At some rare challenge values, the verifier exits without a verdict, or rejects, even on an honest proof. Treat both as not accepted ([Verifying a proof][verifying-endings]).

**Where the format specifies each step**

| step | the format |
|---|---|
| 6.1 Commit | R4.10, R7.4, R12.11, R13.2, R13.14, R13.18, R13.19; format §11.5 for the placement |
| 6.2 Link | R13.3, R13.9, R13.10 |
| 6.3 Parts of a proof file | R14.1, R14.2 |
| 6.3 First chain | R14.3, R14.4, R14.8, R15.1, R15.6; format §11 for what a segment sends, to the extent that the format specifies it |
| 6.3 Slot | R7.18, R14.7 |
| 6.3 Shared chain | R14.9, R13.19 |
| 6.3 Second chain | R14.10; format §13.9 |

## 7. The two arguments, in short

**What is proved.** Lean proves three properties of the protocol as written in Lean, in its non-interactive form (5.6): soundness of the Lean verifier, `verify` (section 2), within a stated bound, for one commitment and one proof (6.4); completeness with that verifier; and [zero knowledge](glossary.md#zero-knowledge) (the cryptographic property nearest to weight hiding) against any verifier within a bound on its hash queries and replies. All three are at [full size](glossary.md#full-size) and under exact match, with the hash function as a random oracle that is independent of the format's [second, fixed hash function](glossary.md#hash-mode). Nothing is proved of the C++ and CUDA programs, which are written to follow the same protocol and are tested against it; they are known to differ from it at [listed points](../lean/GAPS.md).

Completeness and zero knowledge are about a committer and a prover that are Lean definitions, and hold up to a small stated error, for fresh random masks, one commitment, one leaf table and one request to a proof. Completeness covers requests that [can be served](../guide/terms.md) and whose claimed output is the model's own. Zero knowledge holds up to one bit per request (served or refused), which depends on the weights: over many requests these bits tell what the model answers, and no statement bounds what they reveal together. The theorem on zero knowledge says nothing of what the public files themselves tell of the weights, or of timing, and the simulator's running time is not modelled. Lean checks the proofs, not that the statements say what is meant; they are the Lean library's own wording. [What is proved][status] and [guide/weight-hiding.md](../guide/weight-hiding.md) have the other conditions, the bounds and how far each proof has been checked.

**The rest of this section.** Two sketches for intuition, not proofs, one for each goal of section 1. The second covers batches, which Lean's theorems on completeness and zero knowledge do not cover. Each follows a full argument in a technical report of this project that is not public, not refereed and not in this repository ([row HW-763](limits-all-rows.tsv)), so a reader cannot check this page's account of it, which is the project's own. Neither argument has been checked enough to rely on it. Both take the hash as a random oracle and are about the report's form of the protocol, from which the programs differ and which is not shown to be, to the letter, the protocol as written in Lean; some of their steps are cited, one of them taken as given; and no complete set of parameters, of any size, is written out in the report or known to meet every condition of its theorems, for soundness or for hiding.

**Why a false claim should be rejected.** Everything a challenge tests is fixed before the challenge exists, because each challenge is a hash of what the prover has already sent (5.6). So a prover with a false claim needs luck: at some challenge the false claim has to turn into one that passes. Each round gives it only a small chance of that, and in a single try its overall chance is at most the sum of the rounds' chances (5.3, 5.4). Several pieces keep the claim tied to one set of weights:

- Hash trees stop the prover from changing its answers after a commitment (5.2).
- Proximity tests show that a commitment is close to a codeword. At the distances that these tests allow several codewords can be close, and an out-of-domain sample leaves only one message (5.4, 5.5). That the tests show this much, on a request and on a weight table, rests on the theorem on codes (6.1, 6.4).
- The range proof, the range check and conditions on the sizes of the model's numbers keep the equations from holding only modulo the prime (6.1, 6.3).
- The test on the table ties a request's claim to a leaf table, the link proof ties the leaf table to the anchor, and the commitment ID fixes the anchor's root and the configuration (6.3, 6.2, section 4).

A cheater can also try again with different messages. Each try costs a hash call, so the bound grows with the number of hash calls.

**Why the weights should stay hidden.** The goal is stated in terms of a *simulator*, a program that never sees the weights. It is given the configuration, a set of clients' requests and next tokens, whether each of these requests can be proved at all (for a failed range check or a wrong token, that depends only on the weights and the request; rarer failures also depend on random values), how many batches ran, and which of these requests shared a batch and in which order. It also answers the hash queries, those clients' and those of whoever chooses the requests, and so sees them. From that it should be able to produce, up to a small error, everything that those clients receive. That does not include how long an answer takes: the goal does not cover a client that measures time.

By design, what depends on the weights is protected as follows. Opened columns of a weight table look random as long as their number stays within the query bound (5.1), and the programs count the opened columns so as to stop before that limit (6.2, 6.3). Everything else that is committed carries enough randomness of its own for the positions that are read, and salts protect what is only hashed (5.2). Other values that are sent are masked, except the next token, which is sent in the clear (5.3, 6.3).

That is the design. How far it is proved: [guide/weight-hiding.md](../guide/weight-hiding.md), and section 8 for Lean's theorem on zero knowledge.

The design depends on:

- **The operator,** the provider in its day-to-day role of running the service. Among its duties: keep the secret key directory secret, never copy or restore it, never let two runs both resume from the same saved state of the machine, and let no response of a batch leave before `cp_verify` has accepted all its proofs ([the operator's duties][duties] has the list).
- **Derived randomness.** The programs derive their random values, with ChaCha20, from a secret seed and fresh bytes from the system. So the programs' hiding is computational at best. The technical report's bound on the chance that two derived values repeat is stated only for ways of deriving that meet printed conditions and have a named property, and neither is shown for a released program. Read literally, the programs' way fails some of the printed conditions, and by two readings it also lacks the first clause of that property.
- **A build that behaves as its published source code says, and a machine without faults.** A bug or a hardware fault can lose a mask, and the proof or public file can then still verify, because the verifier does not check for masks. [Limits][limits-risk] calls this the main risk.

**Batches overlap by default.** The service takes the next batch before it hands out the last one. The machine's timing decides the order and who shares a batch, and a stop in a later batch can withhold earlier ones. Time, stops and who shared a batch are on [the list of what is not hidden][not-hidden], and no limit is known on what time or a stop reveals.

**The `-public-weights` option.** A commitment made with `cp_commit`'s `-public-weights` option falls outside everything this section says about hiding. Its seed comes from a file, which may be published, and no fresh randomness goes into its key file or its leaf files. For such a commitment, nothing is claimed to be hidden: neither the weights nor one client's request from the other clients in the same batch. Its public files and its proofs do not show that the option was used, so a client has no way to tell it from an ordinary commitment.

**The OOD answers.** Every weight table has OOD answers (5.5, 5.6). They are published for good, and one pad is all that covers each of them. The programs derive the pads. Each time it makes a commitment, `cp_commit` draws a secret seed from the operating system; a leaf table's pads also depend on 32 fresh bytes, which are kept secret alongside the seed. An answer's hiding depends on several things, including these. The seed and those bytes stay secret: whoever learns them can take the pad off every answer that the commitment has published, however long ago. The system's random source does not repeat when the seed is drawn, as it can when a machine image is started twice: if it does, on other weights, the two commitments' public files give one exact equation for each row, for good. The number of opened columns of a table stays within the query bound (5.1), which the programs count toward (6.2, 6.3). ChaCha20, the cipher that derives the pads, is not broken. No verifier can see whether the pads are there: from reading the code, one self-check in `cp_commit` is all that stands between a fault that loses every pad and a public file without pads.

**Some things are not hidden, on purpose.** Among them are the architecture, the quantization scales, each reply, how long a batch takes, and whether a request was served or failed ([What is not hidden][not-hidden]). The public files also hold [masked sums of the weights](what-is-not-hidden.md#record-sums), published for good: they are meant to show nothing about the weights. One consequence is not on purpose: under load, how long one batch takes also decides which requests share the next one. Beyond the reply itself, each answer, whether it comes with a proof or says "failed", tells the client at least one yes-or-no fact about the model's hidden scores, and so about its weights, even if the masks hold; for the programs no statement bounds how much, and nothing bounds what many answers reveal together.

## 8. Machine-checked proofs

[lean/THEOREM.md](../lean/THEOREM.md) has the soundness theorem on `verify` (section 2) as Lean prints it, the same in plain words, its reach, what it rests on, what it does not say and how to check it. The same page has Lean's theorems on completeness and zero knowledge. Like the soundness theorem on `verify`, they are about the protocol as written in Lean. They hold under exact match, for a prover that is a Lean definition, for one commitment and one request to a proof and, in the non-interactive form (5.6), for one leaf table: not for batches, and not for the programs. [The library's own page][lean] says what the Lean library proves and at which parameters, and [lean/TERMS.md](../lean/TERMS.md) gives the terms. [tools/lean-verify/README.md](../tools/lean-verify/README.md) gives the one command that runs `verify`'s executable form from this repository.

## 9. Names

The format uses different names for some things. [Its first page][format] has a table of them.

## Note

† Not on the released programs. What it ran on, and how far that speaks for them: [what was run on these programs, and what was not](limits.md#what-was-run-on-these-programs).

[format]: format/README.md
[verifying]: verifying.md
[quickstart]: quickstart.md
[not-hidden-failures]: what-is-not-hidden.md#failures
[lean]: ../lean/README.md
[not-hidden]: what-is-not-hidden.md
[duties]: operators-duties.md
[status]: ../guide/what-is-proved.md
[limits-policy]: limits.md#within-limits
[limits-risk]: limits.md#the-main-risk
[limits-lost-mask]: limits.md#a-lost-mask
[limits-compiled]: limits.md#compiled-programs
[glossary-gap]: glossary.md#score-gap
[glossary-cost]: glossary.md#cost
[glossary-range]: glossary.md#range-check
[not-hidden-others]: what-is-not-hidden.md#other-requests
[published-works]: ../PUBLISHED-WORKS.md
[verifying-request]: verifying.md#the-request-and-the-token-file
[verifying-endings]: verifying.md#exit-codes
[verifying-setting]: verifying.md#which-setting
[limits-figure]: limits.md#the-soundness-figure
[glossary-closeness]: glossary.md#closeness-round
