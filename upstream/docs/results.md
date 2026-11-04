# Results

- What was measured, on which machines, and how far the figures carry: [Performance](../guide/performance.md).
- Measured figures, each with its conditions: [figures.tsv](figures.tsv).
- What the tests show, and what they do not: [Known limits](../guide/known-limits.md) and [limits.md](limits.md).
- Bad results and limits of the runs, among them those that this page does not tell, stand as rows of
  [limits-all-rows.tsv](limits-all-rows.tsv) whose ids begin RES-.

## Runs that served a floating-point model and proved its replies

In the expected use a provider serves a floating-point model with a stock engine. Afterwards it proves one
pass of the committed integer model over prompt and reply, under a
[clipped-gap policy](glossary.md#clipped-gap), which lets the reply differ from the integer model's top
choices within limits. None of the runs below was made on the released programs.

- **As one course: at small size only.** vLLM 0.29.0 served a small synthetic floating-point model,
  at one small test configuration, with counted proof of work, and the proof middleware had each
  demonstration's eight replies proved as one batch. A pass at a test configuration carries no
  security claim. All 24 replies of three demonstrations, to the same 8 prompts of synthetic token
  IDs, were proved and accepted as within limits. This was no live service: no proving service stood
  beside the engine, and in two of the three demonstrations vLLM was stopped while `cp_prove` ran.
  The limits were fitted to that pair of small models. The one reply that each demonstration
  changed, with every token replaced, got no proof, and the runs do not show why. That is no test of
  the policy (row T42 of [limits-all-rows.tsv](limits-all-rows.tsv)).
- **At full size: in two separate steps.** A stock engine's replies on Llama 4 Maverick's public weights
  were stored first and proved later, under limits fitted to such replies: 48 of 48 got a proof. The same
  replies had been scored before, so this is no rate of acceptance. Limits fitted on the first 24 of the
  fitted replies alone are missed by 3 of the other 24, all of them honest. Of 24 greedy replies of a
  smaller model of the same family none got a proof, and none of 48 replies set after another prompt.
  Edited replies were not tried in that run. In a scoring run, which made no proofs, 28 of 32 replies with one
  token replaced by the model's second choice met the limits (rows T43 and T46). In a further run, 128 fresh
  greedy replies of a stock engine, of up to 500 tokens, were put to the prover under limits fixed beforehand:
  127 got a proof that was accepted as within limits, and 1 got none. That holds for these limits, one model
  and one kind of text. The limits are the looser of two sets, and by scoring alone they let 7 of 96 greedy
  replies of the smaller model through. Of 48 replies cut in pairs one token apart, the 24 that meet the limits
  got a proof and the 24 that miss them got none: that is the honest prover's own decision, and no forged proof
  was tried, so nothing here shows the verifier rejecting one. One token apart is not close in value: at the
  limit on the number of costly tokens a comparison off by one would have shown, but at the limit on the mean
  the pairs need not have shown it ([integrations/LIMITS.md](../integrations/LIMITS.md)).
- **Never as one live service at full size.** At full size vLLM, the middleware and the example client have
  run together only with vLLM serving the integer model itself, under exact match, and stopped while
  `cp_prove` proved the 16 replies as one batch (row T41).

The rate with whole replies checked is on [Performance](../guide/performance.md).
