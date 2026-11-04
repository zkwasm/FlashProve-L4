# Further limits, a short entry each

Many of the tests and readings that this repository reports were not made on the released programs,
but on programs built from [other sources](glossary.md#other-sources); some, the tests with
deliberately cheating provers among them, on programs that also held test-only code. The compiled
programs differ too; the machine code that differs is not shown equivalent and has not been read.
Two of the five tests of weight hiding have run on the released programs: the first at small size on
synthetic weights and at full size, once on synthetic and once on a real model's public weights, and
the fifth once, at small size on synthetic weights. One of the fifth's two audits, the audit of a
commitment, also ran on them at full size, in the timed runs (row LIM-63.2, line 5). At full size
the first test has also run on other programs, and the fifth's two audits have run together only on
other programs. The second and the third ran on test builds of other sources, which hold the code
that injects a fault, and the fourth on programs of other sources that held test-only code. ([What
was run on these programs](limits.md#what-was-run-on-these-programs).)

**In short.** Read [limits.md](limits.md) first: it has the limits that every reader should know.
This page holds further known limits, a few sentences each. If a link or another page gave you an ID
(for example `SO-8` or `H31`), search this page for it. [limits.md](limits.md#how-a-thing-is-known)
defines the evidence labels, and [glossary.md](glossary.md) the terms.

Unless an entry says otherwise, its results were not seen on the released programs. A fault of the
code, told in the present tense, holds of the released programs.

A test or a file that an entry names and that is not in this repository is the project's own, and
readers cannot run or inspect it.

[limits-all-rows.tsv](limits-all-rows.tsv) has the full list under the same IDs, including entries
on neither page. The table has no column for this page's headings. To pick out the rows that belong
under one of them, go by the first word in the table's column "kind" (soundness, hiding, safe to
run, completeness, model, evidence, speed, use), or by how the ID begins (`SO-`; `H` and `HW-`; `R`;
`C`; `M`; `T` and `E`; `P`). Both fit most rows, not all. Most entries here are shortened: the row
under the same ID, with the rows under it, has the rest.

## Soundness

- <a id="so-1.1"></a>**SO-1.1** ([SO-1](limits.md#the-soundness-figure)). The per-round error
  figures do not count a malicious prover's hash queries. There is no margin. The bound against a
  prover that makes N hash queries, what N counts, and the rounds' margins: row SO-1.1. **What you
  can do:** Run `cp_verify -plan accounting` and read its caveats too.
  **[Basis](limits.md#how-a-thing-is-known):** reasoned; reported.
- <a id="so-3"></a>**SO-3**. Every printed figure is for one key file, one leaf file and one
  request's proof. The count of errors over all rounds is no bound over several leaf files. The
  bound for a cheater who computes nothing weakens with a batch.
  **[Basis](limits.md#how-a-thing-is-known):** reasoned. Also row LF-1.
- <a id="so-5.1"></a>**SO-5.1** ([SO-5](limits.md#false-claims-in-test-programs)). Code that keeps
  the language's rules but breaks the library's stated contract (blocks/chain/closed.h; among its
  clauses: naming no private member of the library; declaring nothing at global scope before its
  headers; keeping operator new) can still get `blocks/` to accept a false claim. **What you can
  do:** Call the three programs, not the library. **[Basis](limits.md#how-a-thing-is-known):** read;
  run, not on the released programs; reported (the first way: of 16 tries 9 were carried through and
  all 9 accepted, in its harness on this library; the second: 8 of 48, control 0 of 48, on this
  library). The ways one by one: row SO-5.1.
- <a id="so-6"></a>**SO-6**. The programs check only that a provider's settings are in allowed
  ranges, and not every file that they accept has been shown to be sound. With scales in `KL.bin`
  that the standard parameter form allows, a provider can round every layer's contribution to zero
  for ordinary-sized weights. The proved function is then only embedding, norm and output head, so
  the output depends only on the last token. Rows [SO-6](limits-all-rows.tsv) to
  [SO-6.7](limits-all-rows.tsv) have the rest. **What you can do:** Before you publish or rely on a
  commitment ID, read these values once, or evaluate the integer model that the files describe.
- <a id="so-6.2"></a>**SO-6.2** ([SO-6](limits.md#who-chooses-the-model)). Quantization scales that
  shrink the logits also make the clipped-gap limits easier to meet, so a client should check the
  scales before relying on those limits. **What you can do:** Inspect the scales and compare the
  whole commitment ID. **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="so-6.5"></a>**SO-6.5** ([SO-6](limits.md#who-chooses-the-model)). The full-size
  configuration in this repository is meant for synthetic weights. A real model needs scales and
  some parameter-file lines fitted to its weights. Every proof about a real model therefore runs
  under a configuration that is not in this repository, and the theorems' hypotheses have not been
  checked for such a configuration. **What you can do:** Read the fitted values, or evaluate the
  integer model. **[Basis](limits.md#how-a-thing-is-known):** read. Also row LF-13.
- <a id="so-8"></a>**SO-8**. A `YES` is about the commitment whose ID the caller names. An equal ID
  means the same anchor (the commitment's first weight table) under the same configuration files,
  not the same bytes of the key file. It does not say whose weights the table holds. Which named
  model an ID belongs to rests on its publisher's word, and is outside the protocol. An accepted key
  file shows that the table's entries meet the ranges and lookups named in the public parameters,
  not that these are right for the model. The commitment ID does not cover the code that generates
  them, so whoever vouches for the model must review that code too. **What you can do:** Take the
  commitment ID from whoever vouches for the model, never from the server.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-11"></a>**SO-11**. The key file bounds each embedding entry byte by byte, to 32 bits
  rather than the 21 bits that the model states. What a prover gains from that has not been
  examined. **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="so-15.1"></a>**SO-15.1** ([SO-15](limits.md#attacked-at-small-sizes-only)). The project's
  tests with a malicious `cp_prove` or an injected failure are not in this repository and have run
  little over the released programs' field (what has run: row SO-15.7).
  **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="so-17"></a>**SO-17**. `cp_verify` does not re-check parts of its work, such as the
  positions checked, most claims' coefficients and the order of attention heads, and no input or
  test reaches its two guards against an unfinished sub-proof check. A bug there could make it check
  positions other than those its `YES` names, accept a false statement or reveal something hidden.
  And a commitment made with other conversion tools can describe a function its maker did not
  intend. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-20"></a>**SO-20**. `cp_prove` runs no verifier on the proofs that it makes. The vLLM
  integration that ships hands on no proof, and nothing of its response, before `cp_verify` has
  passed every proof of its batch (narrow exceptions, none open to a client:
  [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); `driver/measure_rate.py`
  reports no rate, and an example script shows no answer, before `cp_verify` has accepted the
  proofs; and a front end of your own must do the like, but that check is the provider's own, not an
  independent one. **What you can do:** Verify proofs outside the provider too.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-22"></a>**SO-22**. The provider chooses the content of the weight table's padding cells,
  and nothing proves that it is 0, so a commitment ID says nothing about them. What filled cells
  would do to hiding has not been examined. **[Basis](limits.md#how-a-thing-is-known):** read;
  argued on paper. A second limit, on cells beyond the true sizes: row SO-22.1.
- <a id="so-26.1"></a>**SO-26.1** ([SO-26](limits.md#within-limits)). How often another model's
  reply would pass under a clipped gap has been measured for a real model only for one smaller model
  of the same family. Two results on small synthetic models are bad. What was seen with a real
  model's altered or resampled replies: row SO-26.1. **[Basis](limits.md#how-a-thing-is-known):**
  read.
- <a id="so-32"></a>**SO-32**. `cp_verify` accepts a response that passes only when proof of work is
  counted, just like any other response that passes, because whether to count it is the client's
  decision. At the default setting every response passes only that way, and that the work
  may be counted toward a round's bits is assumed for these programs' hash chains, not proved for
  them. **What you can do:** Run `cp_verify` yourself and read both printed numbers.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-37"></a>**SO-37**. `g++` 13.3.0 can silently drop the writes that fill in an object
  passed by reference, so the build scripts pass `-fno-ipa-modref`. The bug has never been seen in
  `cp_verify`, but nothing in this repository says that `cp_verify` was examined for it. **What you
  can do:** Build with the repository's scripts, or pass `-fno-ipa-modref` yourself.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-40"></a>**SO-40**. Four public tables depend on the platform's floating-point library,
  so `cp_verify` may reject every honest proof on a platform that rounds differently (by analysis
  only; not tested). It checks the tables only against digests that the provider writes, so a match
  does not show that they are the format's tables. **What you can do:** Use a parameter file that
  states the four digests. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-41"></a>**SO-41**. The engine sends a reply before any proof of it exists, and a
  rejected proof does not stop the engine from serving. **What you can do:**
  Treat a reply as unproved until your own check passes.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-43"></a>**SO-43**. A `YES` vouches only for the weights that the request reads, not for
  padding cells or the embedding rows of tokens not in the request.
- <a id="so-45"></a>**SO-45**. The tag that `cp_prove -spool` writes next to a response does not
  bind the commitment, the policy or the time, and anyone who can read the stored request file can
  forge it. **What you can do:** Restrict request files and tag keys to the service's account.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-48"></a>**SO-48**. A wrong table value in `cp_commit` can end up in a commitment.
  **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="so-50"></a>**SO-50**. Unlike in the standard Fiat–Shamir construction, the next token
  enters the hash chain only just before the output head's segment, so a malicious prover can reuse
  everything earlier for each false token it tries. It can also make many attempts under one binding
  tree and finish the luckiest, a saving that has not been worked out. The bound counts such tries
  as hash calls, not as whole proofs, and has no factor for vocabulary size.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="t17"></a>**T17**. Each parameter file, and so each acceptance policy, needs a separate
  commitment, and nothing public shows that two commitment IDs stand for the same weights. A client
  has only the provider's word for it. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-63"></a>**SO-63**. The commitment ID covers the configuration files but not the
  programs' built-in constants. Only a version word that is typed in stands in for those, so equal
  IDs do not show that two builds have the same constants. **What you can do:** Read a commitment ID
  together with the build of `cp_verify` that checks it, and build the programs from this
  repository's source. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-64"></a>**SO-64**. Position t checks token t + 1, so a prompt of n tokens and a reply of
  m tokens need the range r = n - 1, e = n + m - 1. If you name the reply tokens' own positions
  instead, the first reply token goes unchecked. **What you can do:** Compare the `covered:` line
  with what you meant to check. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-1.4"></a><a id="so-1.5"></a><a id="so-2.1.1"></a><a id="so-2.2"></a><a id="so-2.2.1"></a><a id="so-2.3"></a><a id="so-2.4"></a><a id="so-2.5"></a><a id="so-2.6"></a><a id="so-2.7"></a><a id="so-3.1"></a><a id="so-4"></a><a id="so-4.1"></a><a id="so-4.2"></a><a id="so-6.3"></a><a id="so-6.3.1"></a><a id="so-6.3.2"></a><a id="so-6.8"></a><a id="so-7"></a><a id="so-7.1"></a><a id="so-7.2"></a><a id="so-8.4"></a><a id="so-12"></a><a id="so-12.1"></a><a id="so-14"></a><a id="so-14.1"></a><a id="so-15.1.1"></a><a id="so-15.1.2"></a><a id="so-15.2"></a><a id="so-15.3"></a><a id="so-15.5"></a><a id="so-15.6"></a><a id="so-16.3"></a><a id="so-16.4"></a><a id="so-17.1"></a><a id="so-17.3.1"></a><a id="so-17.6"></a><a id="so-17.7"></a><a id="so-17.7.1"></a><a id="so-19.2"></a><a id="so-21"></a><a id="so-21.1"></a><a id="so-22.3"></a><a id="so-23"></a><a id="t9"></a><a id="t9.2"></a><a id="t9.3"></a><a id="so-26.2"></a><a id="so-26.3"></a><a id="so-31"></a><a id="so-31.1"></a><a id="so-32.2"></a><a id="so-32.3"></a><a id="so-33"></a><a id="so-33.1"></a><a id="so-34"></a><a id="so-39"></a><a id="so-39.1"></a><a id="so-42"></a><a id="so-44"></a><a id="so-46"></a><a id="so-47"></a><a id="so-51"></a><a id="so-52"></a><a id="so-52.1"></a><a id="so-53"></a><a id="r20"></a><a id="t17.1"></a><a id="so-58"></a><a id="so-59"></a><a id="so-61"></a><a id="so-62"></a><a id="so-65"></a>Further
  rows: [the table](limits-all-rows.tsv).

## Weight hiding

No fault has been injected into the released programs, which have no option for it, and no broken
duty has been tried on them, unless an entry says so: what an entry says of a fault or a broken duty
is read or reasoned, unless it tells a run. Which tests of hiding ran at the default setting, and
one count from them: rows HW-439, HW-461, H3.1 and HW-584.

- <a id="h2"></a>**H2**. From the published files, anyone can recover exponents that were fitted to
  the weights, with no fault, request or secret file. **What you can do:** Treat everything fitted
  to the weights as public, and read the list of what is shown on purpose once before you publish a
  commitment. **[Basis](limits.md#how-a-thing-is-known):** read. The list of what is shown on
  purpose: row H2.
- <a id="h3.7"></a>**H3.7** ([H3](limits.md#a-lost-mask)). A request's encoded rows with too few
  random entries, or with some replaced by zeros, have not been tried on the released programs, and
  no one has checked whether a client could cause either. The released programs do not have the
  start-up check on how many random entries each row gets. Some faults are no losses, and a
  self-check that computes a value a second time by the same rule agrees with them: a mask that is
  never generated, a value that is used twice, a rule shared by prover and verifier that masks too
  little. None of the three has been tried on the released programs.
  **[Basis](limits.md#how-a-thing-is-known):** read. See also rows E14.1 and HW-398.
- <a id="h3.9"></a>**H3.9** ([H3](limits.md#a-lost-mask)). A filler cell that is left unzeroed keeps
  the value that lay there, and whether that value would be masked like any other cell has only been
  argued (row HW-692). Filler cells left unzeroed have not been tested on the released programs, and
  a launch lost at these cells has not been tried on them. There is no known way for a client to
  cause this fault, but no one has looked for one. **[Basis](limits.md#how-a-thing-is-known):**
  read. What the programs do: rows H3.9 and HW-570.
- <a id="h3.10"></a>**H3.10** ([H3](limits.md#a-lost-mask)). A fault in the generator of random
  values, or on the masks of the claims about the weights: whether the self-checks of the released
  programs would stop either has not been tried. Such a fault can strike every proof alike, and each
  proof then gives exact equations in hidden values (what that is worth: row HW-445). A bug or
  failing hardware might cause such a fault, and no one has looked for a way a client could.
  **[Basis](limits.md#how-a-thing-is-known):** reasoned.
- <a id="h4.7"></a>**H4.7** ([H4](limits.md#the-conditions)). On a rented machine you may have no
  way of knowing whether the host saves and resumes its state (snapshots, live migration), and that
  case was not examined. So a provider on rented machines may not be able, on its own, to meet the
  condition against running a saved state twice, and has to decide whether to accept that.
  **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="h4.10"></a>**H4.10** ([H4](limits.md#the-conditions)). Booting a fresh machine from an
  image does not count as running a saved state twice, as long as the system seeds its random
  generator freshly at every start and the image holds no secret key directory. No program can check
  the first condition, and an image can carry a saved seed file for the system's generator. None of
  these counts as freshly booted: a machine resumed from a snapshot of a running machine, a process
  restored from a checkpoint, or a clone made after boot. Whether any given system, image or
  hypervisor seeds freshly at every start was not examined here. What a fork that comes from outside
  the programs, or a checkpoint of a process and a restore from it, does with the state that the C
  library may keep for `getrandom()` has not been examined for these programs, and for the sake of
  these draws no version of that library or of the kernel is asked. From version 2.41 on, and in
  some distributions' packages from 2.39, the GNU C Library may answer `getrandom()` without a
  system call, from a state in the process's own memory. For such packages in Fedora 40, Fedora 41
  and CentOS Stream 10, CVE-2025-0577 records a fault: after a fork that runs concurrently with a
  call, a later call may return predictable bytes. The seed, the 32 random bytes of a leaf table's,
  the key file's and a leaf file's commit run, and those of every batch come from `getrandom()`.
  (What this says of the library, the kernel and the CVE's record has not been checked a second
  time.) **What you can do:** Confirm that your system, image or hypervisor reseeds at every start.
  Use a C library in which CVE-2025-0577 is fixed. **[Basis](limits.md#how-a-thing-is-known):**
  read.
- <a id="h5.3"></a>**H5.3** ([H5](limits.md#the-limit-of-use)). A leaf table can be made to serve
  more batches than it has masks for. A careless provider can bring a crossing about, and so can an
  accident. So can anyone who can write the provider's parameter file before the commitment, shorten
  the counter on the provider's machine, or write, copy or roll back the secret key directory.
  `cp_prove` then stays silent. Past a leaf table's limit the opened columns give equations in the
  weights without masks (row H21.11); whether weights can be solved for from them has not been tried
  on the released programs. None of these ways to a crossing has been tried on the released
  programs. **What you can do:** Let only one account write to the secret key directory, and never
  serve from a copied, restored or rolled-back one. **[Basis](limits.md#how-a-thing-is-known):**
  reasoned. See also rows R22 and H21.11.
- <a id="h5.4"></a>**H5.4** ([H5](limits.md#the-limit-of-use)). Using several commitments does not
  hide how many there are, how many leaf tables each has, which one a batch ran under, when each was
  first shown, or that two have the same public files. **What you can do:** Do not pick a request's
  commitment based on anything hidden. **[Basis](limits.md#how-a-thing-is-known):** argued on paper.
- <a id="h5.5"></a>**H5.5** ([H5](limits.md#the-limit-of-use)). A new commitment does not undo
  overuse. Once a leaf table has served more batches than its limit, no statement of the written
  argument holds for those weights, even under a later commitment. **What you can do:** Never serve
  from an `isk` whose counter is in doubt. **[Basis](limits.md#how-a-thing-is-known):** argued on
  paper.
- <a id="h9"></a>**H9**. By design, a commitment made under a test seed hides nothing from anyone
  who knows the seed. Its public files carry no mark of the test seed, so someone who is handed them
  cannot tell. No test of hiding covers overlapping batches. Of `cp_prove`'s options `-overlap` and
  `-overlap-two` (row H9.6) there is no argument either, and how they work inside has not been read
  in the released programs' code. No statement with a proof covers the service as it runs by
  default, with batches that overlap in time, that is, it takes a new batch before the last one has
  been handed out, and can have counted and proved that batch before the last one reaches a client
  (inferred). **What you can do:** Serve with the production build only; never reuse a test's seed
  or randomness file. **[Basis](limits.md#how-a-thing-is-known):** run, not on the released
  programs; read.
  - <a id="h9.1"></a>**H9.1**. The programs reject only some misuses of the test options. In
    particular, nothing stops you from passing a commitment's real seed to a test build as its test
    seed. **What you can do:** Never use the real seed as a test seed.
    **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h11.1"></a>**H11.1** ([H11](what-is-not-hidden.md#other-requests)). A client learns which
  proofs shared a batch and, from timing, roughly how many requests shared it and something about
  their prompts. Only a little was measured, not on the released programs and on synthetic weights,
  and never how much timing reveals overall. **What you can do:** Do not rely on these programs to
  hide timing. **[Basis](limits.md#how-a-thing-is-known):** run, not on the released programs. What
  was measured (two models told apart by the time to the first token): row LF-17.
- <a id="h14"></a>**H14** ([hidden values left in GPU memory](limits.md#gpu-working-values)). Two
  tensor-core kernels leave hidden values in the GPU's shared memory and registers when they finish,
  and nothing clears them. One leaves sums and scale pairs; the other leaves decoded weights,
  activations and sums. These two are not the only ones: other kernels that multiply on the tensor
  cores also end without clearing their shared memory (by one count from the sources, not checked:
  twelve in all, in eight files). It is not known which reset clears GPU memory. **What you can
  do:** Give the service a dedicated GPU. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h17.1"></a>**H17.1** ([H17](limits-all-rows.tsv)). The fault here is a mask lost on its way
  into GPU memory. The value that the mask should hide can then go out unmasked in a file that
  verifies (row HW-441). The fault has not been tried on the released programs, and no way for a
  client to cause it has been looked for. Among other things, the read-back check, which the
  released programs have, cannot see a mask overwritten after it has looked, or a fault in the one
  formula that both places a mask and finds it again. **What you can do:** Read `blocks/iop/send.h`
  for what the read-back misses. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h17.3.1"></a>**H17.3.1** ([H17.3](#h17.1)). A random mask lost in CPU memory can pass
  unnoticed and leave the message under it readable without any secret. What the exposed message
  reveals about the weights is not known. **What you can do:** Where you can, audit a commitment
  before its public files leave the machine; that is possible only on synthetic weights. The audit
  has blind spots (pads inside tables of real values; unopened leaves). Nothing detects such a loss
  in a proof that a service made. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h21.1"></a>**H21.1** ([H21](limits.md#the-seed)). Someone who can write to the secret key
  directory can point its line on the weights' directory at a directory of their own. They then
  learn about the weights from how each start of `cp_prove` ends. So does someone who can pass a
  directory of their own as the weights' directory. Of
  what writers of the secret key directory read from how starts end, no trial on the released
  programs is known. **What you can do:** Restrict write access to the secret key directory.
  **[Basis](limits.md#how-a-thing-is-known):** read; reported. See also rows H21 and HW-433.
- <a id="h21.5"></a>**H21.5** ([H21](limits.md#the-seed)). Anyone who can set the environment for
  the first start of `cp_prove` or `cp_commit`, or the system's preload file, gets to run code
  inside the program. Starting a second time with a reduced environment does not protect against a
  library built to hide itself. So you have to trust anyone who can write to the secret key
  directory, the weight files' headers or the first start's environment. **What you can do:**
  Control who can set these programs' start-up environment.
  **[Basis](limits.md#how-a-thing-is-known):** read; reported.
- <a id="h21.15"></a>**H21.15** ([H21](limits.md#the-seed)). If an older copy of the secret key
  directory is put at its path before a start, `cp_prove` resumes from the copy's lower counters and
  no program notices, so a commitment can be served past its limit of use. What the batches past
  that limit give away (by reasoning: row H21.11) has not been tried on the released programs.
  **What you can do:** Never serve from copies. Let only one account write to the directory and the
  directories above it. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h26.1"></a>**H26.1** ([H26](limits-all-rows.tsv)). After a kill, a crash or a power loss,
  staged responses are left intact on disk, under `out/.tmp` in a spool directory or `.tmp` in the
  `-out` directory. **What you can do:** Never release staged files. If a call with `-out` ended
  with an exit code other than 0, or was cut off by a kill or a power loss, run `cp_prove -clear` on
  that directory only. Encrypt that disk. **[Basis](limits.md#how-a-thing-is-known):** run, not on
  the released programs; read; reported. See also rows HW-491 and HW-480.
- <a id="h26.2"></a>**H26.2** ([H26](limits-all-rows.tsv)). Once a request has been served, its
  file, which holds its tokens, is only unlinked, not overwritten. **What you can do:** Treat the
  spool directory's disk as permanently secret. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r12"></a>**R12** ([a lost mask](limits.md#a-lost-mask)). A bad exit of `cp_prove` is
  recorded only in the service's output and exit code, so a restarted service carries on as if
  nothing had happened. **What you can do:** Keep the log. Never restart automatically after a
  non-zero exit. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h27.2"></a>**H27.2** ([H27](limits-all-rows.tsv)). Like `wsums.txt` itself, the digest of
  `wsums.txt` lets anyone with a candidate model test whether it is the one being served. If you
  pass the digests themselves with the option `"sha256"`, that digest ends up on vLLM's command
  line, and so in the process list and in vLLM's log. `fold.txt` holds no hash of the weights, but
  one of its two lines is e_res, a setting that can be fitted to the weights. **What you can do:**
  Keep `fold.txt`, that log and the process list as secret as the weights.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h28.3"></a>**H28.3** ([H28](limits-all-rows.tsv)). Any other tool that reads or hashes the
  secret key directory reads the seed too. Unlike the three programs, it keeps the seed in ordinary
  memory that is neither locked nor cleared, and does not mark its process non-dumpable. **What you
  can do:** Do not let other tools read the secret key directory.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h31"></a>**H31** ([the conditions](limits.md#the-conditions)). With `-worst-case-table`,
  `cp_commit` writes a file computed from the weights. Every build accepts the option, even the
  production build on a model's files. **What you can do:** Never share that file outside the
  provider, and keep `cp_commit`'s log as private as the file.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h34"></a>**H34** ([which policy](limits.md#which-policy)). Clients who send quickly, or
  flood the service, can steer when they are batched and with whom. That lets them aim the declared
  leaks (timing, shared failure, which proofs share a batch) at one client. A front end to
  `cp_prove` can limit each client and give each response only to its owner, but whoever builds it
  sees every prompt and response. **What you can do:** Put a front end that limits each client in
  front of `cp_prove`. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r21"></a>**R21** ([the conditions](limits.md#the-conditions)). If a leaf table's counter
  (`count.bin`) is missing, replaced, truncated or not yours, you no longer know what the table has
  shown. Hiding for that table may be lost, and you need [a new
  commitment](operators-duties.md#a-new-commitment). [Our judgement on weight
  hiding](limits.md#the-verdict) does not cover a commitment that is in doubt. The one that replaces
  it may be inside that judgement alone, on [conditions of its own](limits.md#a-further-commitment),
  but for these weights as a whole the judgement does not speak. It does not cover what the old one
  may already have shown. No program can detect a counter truncated, rolled back or replaced by a
  file of the right kind. It is served anyway, and columns shown beyond the limit reveal
  combinations of weights. **What you can do:** Never recreate or restore the counter.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h38"></a>**H38** ([what every client learns](limits.md#what-every-client-learns)). The next
  token is in the proof file unmasked, and `cp_verify`'s line `covered:` shows the range of
  positions. **What you can do:** Keep proofs and `cp_verify` output as private as the request.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="h1"></a><a id="h1.3"></a><a id="h1.4"></a><a id="h2.1"></a><a id="h2.2"></a><a id="h3"></a><a id="h3.1"></a><a id="h3.5"></a><a id="h3.6"></a><a id="h3.7.1"></a><a id="h3.8"></a><a id="h3.11"></a><a id="h3.13"></a><a id="h3.14"></a><a id="h3.15"></a><a id="h3.16"></a><a id="h3.18"></a><a id="h3.19"></a><a id="h3.20"></a><a id="h3.21"></a><a id="h4.1"></a><a id="h4.2"></a><a id="h4.3"></a><a id="h4.4"></a><a id="h4.5"></a><a id="h4.8"></a><a id="h4.9"></a><a id="h4.11"></a><a id="h5.1"></a><a id="h5.2"></a><a id="h6.1"></a><a id="h6.2"></a><a id="h6.3"></a><a id="h7"></a><a id="h7.1"></a><a id="h7.1.1"></a><a id="h7.2"></a><a id="h7.3"></a><a id="h7.4"></a><a id="h8.1"></a><a id="h8.4"></a><a id="h9.2"></a><a id="h9.3"></a><a id="h9.4"></a><a id="h10"></a><a id="h11"></a><a id="h11.2"></a><a id="h11.3"></a><a id="h11.4"></a><a id="h11.5"></a><a id="h11.7"></a><a id="h12"></a><a id="h12.2"></a><a id="h13.1"></a><a id="h13.1.1"></a><a id="h13.1.2"></a><a id="h13.1.2.1"></a><a id="h13.2"></a><a id="h13.3"></a><a id="h13.3.1"></a><a id="h13.5"></a><a id="h13.6"></a><a id="h14.1"></a><a id="h15.1"></a><a id="h15.2"></a><a id="h15.3"></a><a id="p3"></a><a id="p3.1"></a><a id="p3.2"></a><a id="p3.3"></a><a id="p3.4"></a><a id="h16.1"></a><a id="h16.2"></a><a id="h17.3.1.1"></a><a id="h17.3.3"></a><a id="h17.4"></a><a id="h17.7"></a><a id="h19.1"></a><a id="h19.1.1"></a><a id="h19.2"></a><a id="h20"></a><a id="h20.1"></a><a id="h21.2"></a><a id="h21.3"></a><a id="h21.4"></a><a id="h21.6"></a><a id="h21.8"></a><a id="h21.11"></a><a id="h21.13"></a><a id="h21.14"></a><a id="h25"></a><a id="h26"></a><a id="h27"></a><a id="h28"></a><a id="h28.1"></a><a id="h29"></a><a id="h30"></a><a id="h32"></a><a id="h33"></a><a id="h33.1"></a><a id="h34.1"></a><a id="h35"></a><a id="h35.1"></a><a id="h37"></a><a id="r22"></a><a id="h38.1"></a><a id="h39"></a><a id="h39.1"></a><a id="h39.2"></a><a id="h39.3"></a><a id="h39.4"></a><a id="h39.5"></a><a id="h39.6"></a><a id="h40"></a><a id="h41"></a><a id="h42"></a><a id="c6.1"></a><a id="h44"></a><a id="h44.1"></a><a id="h45"></a><a id="h45.1"></a><a id="h45.2"></a><a id="hw-345"></a><a id="hw-538"></a><a id="hw-550"></a><a id="hw-551"></a><a id="hw-552"></a><a id="hw-553"></a><a id="hw-554"></a><a id="hw-555"></a><a id="hw-556"></a><a id="hw-557"></a><a id="hw-558"></a><a id="hw-559"></a><a id="hw-560"></a><a id="hw-561"></a><a id="hw-562"></a><a id="hw-563"></a><a id="hw-564"></a><a id="hw-565"></a><a id="hw-566"></a><a id="hw-567"></a><a id="hw-568"></a><a id="hw-569"></a><a id="hw-570"></a><a id="hw-571"></a><a id="hw-572"></a><a id="hw-573"></a><a id="hw-574"></a><a id="hw-575"></a><a id="hw-576"></a><a id="hw-577"></a><a id="hw-578"></a><a id="hw-579"></a><a id="hw-580"></a><a id="h23"></a>Further
  rows: [the table](limits-all-rows.tsv).

## Safe to run

- <a id="r1"></a>**R1**. The test runner confines its tests with Linux's Landlock, and the
  confinement has gaps (read in the code). A test can write to `/dev/shm` and to whatever directory
  the variable `BUILD` names, and truncate files on older Linux kernels. Only on Linux 6.12 or later
  does the confinement limit which processes a test signals. On Linux 5.13 to 6.11 a faulty test can
  signal your other processes, and one run as root any process on the machine (row R1.1), and that
  level of the confinement has never run on such a kernel. This guards against a test's mistake, not
  against a test that means harm: a test run as root keeps root's other powers. `chmod`, `chown` and
  `touch` on any file are not limited (read in the code).
  **[Basis](limits.md#how-a-thing-is-known):** read; reported. Also row LF-16.
  - <a id="r1.3"></a>**R1.3**. The test runner never signals a test that has begun to use the GPU,
    because killing a program while one of its kernels runs can take the whole machine out of
    service. It names a surviving process group in capitals and leaves it running (read in the
    code). **What you can do:** After such a line, wait for the named group to exit, or end it
    yourself when you know what it is doing. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r2.6"></a>**R2.6** ([R2](limits.md#path-names)). Not on the released programs, a second
  account able to write to the parent directory could swap `cp_commit`'s secret working directory
  mid-run. In a trial the anchor's count went to the wrong counter in 8 of 8 runs. `cp_commit`
  reaches the counter through a directory it holds open (read, in other sources; the whole program
  has not been run that way). A swapped-in link below a working directory would be followed. Files
  that a run only reads are looked up by their whole path. **What you can do:** Give `cp_commit`
  output directories whose parent only you can write to, and name the file for `-worst-case-table`
  in such a directory too. **[Basis](limits.md#how-a-thing-is-known):** run, not on the released
  programs.
- <a id="r4"></a>**R4**. The audit in `tools/audit/` has helper programs that run where they can
  read the seed. `audit_commitment.sh` and `audit_batch.sh` keep them in a directory closed to
  everyone else and ignore `PYTHONPATH`, but trust the caller's other variables: `PATH`, `LD_*`,
  `BASH_ENV` and others can name code. Every other `python3` start, in `play.sh` or by hand, honours
  `PYTHONPATH`, and `audit.py` by hand loads `libfield.so` from `AUDIT_LIB` or the tool's directory
  without checking the owner. **What you can do:** Use one of the two scripts, a new work directory,
  and a shell whose variables you set yourself. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r5"></a>**R5**. `prover/build.sh` does not build `cp_verify`: it copies whatever is at the
  verifier's build path, and `cp_commit`'s check of its public files
  relies on that copy. What a production build checks of
  that copy, and that this does not catch a hostile copy: row LF-15. **What you can do:** Rebuild
  the verifier before each build of the prover. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r6"></a>**R6**. Once a production build of `cp_prove` starts reading secrets, it explains
  its exit with at most one fixed word, and a crash or an unlisted cause gets none. One exception:
  if a word was set before the crash, that word is printed, with exit code 9 after a failed
  self-check. Exit code 0 can also mean that every request failed, and it never says that a
  proof verifies. **What you can do:** Check that
  each request has a proof; see `prover/docs/ends-in-short.md`.
  **[Basis](limits.md#how-a-thing-is-known):** run, not on the released programs; read.
- <a id="r7"></a>**R7**. No exit of `cp_prove` wipes all memory: its GPU compute engine's memory
  pool is never wiped. A crash (`SEGV`, `BUS`, `FPE`, `ILL`, `ABRT`, `SYS`, `TRAP`) or a `KILL`
  skips the wipe altogether. **What you can do:** Treat the machine's memory as holding secrets
  after every exit. **[Basis](limits.md#how-a-thing-is-known):** read; reported.
- <a id="r8"></a>**R8**. Test scripts do shell arithmetic on words that they read, and a crafted
  word would run a command unless first checked to be all digits. A reading of 102 shell scripts
  (the 33 in this repository by eye, the others, which are not in it, sorted with a script) ran
  nothing, and at least one script in this repository was not among them. By that reading, every
  such word from a file that the script did not write itself is checked first. Of the 42 checked
  words, 23 are checked only by the pattern of the `sed` or `grep` command that fetches them. Five
  variables in four scripts are not checked. Not read: shell text inside other files (`SUITE.md`'s
  rows, Makefiles, Python tests). **What you can do:** Run the tests only on the repository's own
  files. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r13"></a>**R13**. `cp_commit` gives each `cp_verify` run a time limit and a supervisor that
  kills it if `cp_commit` dies, but these guards have gaps. A `cp_verify` stuck in a system call
  cannot be killed. A supervisor killed outright leaves its `cp_verify` running, and the next
  `cp_commit` does not look for orphaned runs. **What you can do:** Before the next call, make sure
  that no check of the last one still runs. **[Basis](limits.md#how-a-thing-is-known):** read.
  - <a id="r13.1"></a>**R13.1**. If `cp_commit` is killed outright, a supervisor kills each
    `cp_verify` check it started; that has been shown only on a CPU, with small substitutes for both
    programs. On a GPU the same routine was shown to work once, in another program than `cp_commit`,
    not on the released programs, with a substitute for `cp_verify`, but not for the true
    `cp_verify`'s children, a production build or full size.
    After a power loss, a request can in one case land in a second
    batch. **What you can do:** Never restore, copy or share a spool directory between services.
    Keep one for each commitment and policy. **[Basis](limits.md#how-a-thing-is-known):** read;
    reported (the run on a GPU).
- <a id="r14"></a>**R14**. Anyone who can hard-link a weight file can make `cp_commit` or `cp_prove`
  reject the provider's weights, which is a denial of service. Directories with legitimate hard
  links, such as hard-link copies or deduplicating stores, are rejected too. **What you can do:**
  Keep `fs.protected_hardlinks = 1`, and store real copies where other accounts cannot link to them.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r16"></a>**R16**. Availability is not guaranteed: a client can fill the disk with unfetched
  responses, flood the spool directory or use up a leaf table's batches. **What you can do:** Put a
  front end in front of the service. Have it limit each client and the number of waiting requests,
  check each request's length, and set a timeout for each request. Have it also delete a response
  once it is delivered, or once it has gone unfetched for a time you set.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r18"></a>**R18**. At startup, `cp_prove -spool` cleans the directory it is given and
  removes, without asking, any file of yours under a name it cleans (read in the code). It starts
  only on a directory that has a well-formed `mark.txt`, any commitment's, or
  counts as empty. A directory counts
  as empty if it holds nothing, or only what a front end may put there before the first start (row
  [R18.1](limits-all-rows.tsv)). In a directory that counts as empty, files in `in/` with a
  request's name and length are served as requests: moved and deleted. **What you can do:** Dedicate
  a directory to `-spool`. **[Basis](limits.md#how-a-thing-is-known):** read. Also row LF-5.
- <a id="r18.4"></a>**R18.4** ([R18](limits.md#the-spool-directory-is-cleaned)).
  `cp_prove -clear <directory>` removes files. It deletes responses that a previous run staged, and
  recognises them only by name. `-clear` says what it left, not what it took. It touches only
  staging directories that a `mark.txt` marks as `cp_prove`'s. It is given no commitment ID, so it
  clears any commitment's marked directory. Not on the released programs, and without the mark file,
  two test drivers ran these routines, not `cp_prove` itself. In that trial any short name of a
  directory one level below `.tmp` counted as staged. From reading the code with the mark file, for
  `-clear` that holds only in a single batch's marked `.tmp`. **What you can do:** Pass it only your
  `-spool` or `-out` directory, as an absolute path. **[Basis](limits.md#how-a-thing-is-known):**
  run, not on the released programs; read; reported. See also row HW-480. Also row LF-6.
- <a id="e16"></a>**E16**. The service's promises after a power loss rest on the file system's
  directory sync, and no power loss has been simulated. After a restart, `cp_prove` fails every
  request it finds in `work/` and withdraws its response, even if a front end has already delivered
  it. **What you can do:** Decide each request's outcome once and store it yourself: the first
  complete response you read is the outcome. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r19.1.3"></a>**R19.1.3** ([R19.1](limits.md#the-tests-and-your-files)). The public test
  `spool_clear_test`, started by hand through `prover/tests/unit.sh`, checks the path first, and
  ends as not run if a directory above is open to another account (read in the code; tried once, as
  root, without a second account). Not on the released programs it wrote three small files below
  your `TMPDIR` before it looked at the directories above, and followed symbolic links. The script's
  other test programs have not been read for the same fault. **What you can do:** Run the tests
  through the test runner. **[Basis](limits.md#how-a-thing-is-known):** read; reported.
- <a id="r23"></a>**R23**. `cp_prove` releases a response by renaming its files, and nothing in the
  prover detects
  a change made to a staged file before the rename, and a change made before the keyed tag beside a
  response (row SO-45) is computed gets a valid tag. Making one takes someone running as the
  service's account. In one trial, not on the released programs, a byte of a staged file was changed
  after the program's last read of it, and nothing noticed. **What you can do:** Run `cp_verify` on
  the very bytes that you hand out. **[Basis](limits.md#how-a-thing-is-known):** read. Also row
  LF-7.
- <a id="r24"></a>**R24**. At startup, `cp_prove` checks ownership and permissions along the
  `-spool`, `-isk` and `-out` paths, but not along a single batch's `-in` (read in the code). Anyone
  who can write to a directory above it can make `cp_prove` read requests from elsewhere. The output
  directory itself is still made by the full path, on a path that
  has been checked. Under sudo, the account that ran sudo is trusted like
  root along the path, and may also own the spool directory and the directory named with `-clear`.
  Such an account can redirect the programs through a directory it owns. What else passes these
  checks all the same, a sticky directory for one: rows R24 and R28. **What you can do:** For a
  single batch, name directories no other account can write to, with none above them that another
  account can write to. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="r1.1"></a><a id="r1.2"></a><a id="r2.1"></a><a id="r2.2"></a><a id="r2.5"></a><a id="r3"></a><a id="r3.1"></a><a id="r3.3"></a><a id="r3.4"></a><a id="r5.2"></a><a id="r7.1.1"></a><a id="r7.3"></a><a id="r7.4"></a><a id="r7.5"></a><a id="r7.7"></a><a id="p3.4.1"></a><a id="r9"></a><a id="r13.1.1"></a><a id="r15"></a><a id="r16.3"></a><a id="r18.1"></a><a id="r18.3"></a><a id="r18.4.1"></a>Further
  rows: [the table](limits-all-rows.tsv).

## Completeness

- <a id="c2.4"></a>**C2.4** ([C2](limits.md#an-honest-proof-can-be-rejected)). A last line that
  begins `NO: internal-`, with exit code 1, does not reject a proof; it means `cp_verify` itself
  failed. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="c5"></a>**C5**. `cp_commit` admits parameter files under which `cp_prove` can serve no
  request and ends at its start. Completeness is stated for the parameter files that `cp_prove`
  admits; the example configurations are such files. **What you can do:** Start `cp_prove` once on a
  new commitment before you rely on it. **[Basis](limits.md#how-a-thing-is-known):** run, not on the
  released programs.
- <a id="so-9.1"></a><a id="so-9.2"></a><a id="c1"></a><a id="c1.1"></a><a id="c1.2"></a><a id="c2.1"></a><a id="c2.2"></a><a id="c2.3"></a><a id="c2.5"></a><a id="c3.1"></a><a id="e9.1.1"></a><a id="t14"></a><a id="c4"></a>Further
  rows: [the table](limits-all-rows.tsv).

## The model

- <a id="m1"></a>**M1**. Nothing checks the rule used to convert public weight files, and the
  converting programs accept any architecture name. A commitment cannot be recomputed from files
  only, except under `-public-weights`, where the seed is a published file and nothing is hidden.
  The scales' floating-point pass, tensor names and rotary pairing follow a public library's source
  but were never compared with its output. **[Basis](limits.md#how-a-thing-is-known):** read.
  - <a id="m1.2"></a>**M1.2**. A `YES` is about the integer model that the commitment and the
    configuration define. How often its top choice differs from the token that a floating-point
    implementation served was counted once, in coarse bins, on 96 greedy replies, not on the
    released programs. That count did not check the weight files against their checksums (row
    [M1.7](limits-all-rows.tsv)), and no such count has been made on the released programs. The
    result carries over only to a commitment with the same fitted scales. Another measurement of it,
    on other data, is told in [limits.md](limits.md#the-integer-model).
    **[Basis](limits.md#how-a-thing-is-known):** run, not on the released programs.
  - <a id="m1.4"></a>**M1.4**. A vouching party cannot confirm the commitment ID of hidden weights
    by committing again, because every `cp_commit` run draws fresh secret randomness and produces a
    new ID. For public weights, `-public-weights` derives the seed from a published file so that
    others can regenerate the commitment. Such a commitment hides nothing, and nothing in its public
    files says so. **What you can do:** In public-weights mode, serve other people's requests one
    per batch, or not at all. **[Basis](limits.md#how-a-thing-is-known):** reported.
  - <a id="m1.9"></a>**M1.9**. If `policy.txt` and `KL.bin` come from different runs of the fitting
    tools, no program notices, and the proofs are about a model that belongs to neither run. **What
    you can do:** Before vouching, load the committed files into the integer model.
    **[Basis](limits.md#how-a-thing-is-known):** reported.
  - <a id="m1.11"></a>**M1.11**. `cp_commit` can commit to a model that `cp_prove` then rejects at
    every start, so the long run is wasted and the commitment cannot be served, though nothing false
    is proved. This is known from source review, and nothing was run. **What you can do:** Start
    `cp_prove` on a new commitment before publishing its ID.
    **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="m1.5"></a><a id="m1.6"></a><a id="m1.7"></a><a id="m1.8"></a><a id="m1.10"></a><a id="m2"></a>Further
  rows: [the table](limits-all-rows.tsv).

## Tests and evidence

- <a id="t2"></a>**T2**. Some tests would miss the fault they exist to catch, and most tests of
  other programs were not ported to the released programs. Those not ported include most of those
  that appeared to guard weight hiding (92 of 111). **[Basis](limits.md#how-a-thing-is-known):**
  read.
- <a id="t3"></a>**T3**. Most GPU unit test programs run under no sanitizer, and three test programs
  lack a full race check. None of NVIDIA's four checkers has run on a GPU on a test build of the
  released programs' sources (rows T3 and T3.2 of [the table](limits-all-rows.tsv)). No sanitizer
  can watch the production build (from reading the code; no run that tried it is known), and no tool
  looks for races on global memory. In a unit test where two threads use the GPU at once, reports of
  uninitialised reads (the checker printed 1,800 of 11,332; one report is in the programs' own
  kernel) and one illegal memory access are not explained. A memory checker reported reads of
  uninitialised GPU memory in a served batch, in both runs of one test, not on the released
  programs, at a small size; a one-line fix to the warm-up removed the reports seen, but it was
  never shown that all those reads came from the warm-up, and no test replays the check.
  **[Basis](limits.md#how-a-thing-is-known):** run, not on the released programs; reported.
- <a id="t4"></a>**T4**. Two builds from the same sources can produce different program files
  (`cp_commit` and `cp_prove` did; `cp_verify` has come out the same), so a file's hash need not
  identify a build. `cp_verify` has no build record, the `build=` value it prints ignores the
  compiler and its options, and a build record does not stop a malicious builder. **What you can
  do:** Compare `<program>.build` line by line, not file hashes, the record's own among them: its
  `sha256` line is the program's own sum and differs with it. Look at the `sources` line, the
  `command` and `flag` lines, and the lines on the tools (among them `compiler`, `host-compiler`,
  `host-linker`, `device-compiler`, `device-assembler`, each with a `-sha256` line). Build
  `cp_verify` yourself. **[Basis](limits.md#how-a-thing-is-known):** run, not on the released
  programs; reported.
  - <a id="t4.2"></a>**T4.2**. A build record's lines can be joined into shell commands, so anyone
    who runs them executes whatever its author wrote. **What you can do:** Run a build record's
    commands only if it comes from a trusted source. **[Basis](limits.md#how-a-thing-is-known):**
    read.
- <a id="t5"></a>**T5**. No test in the suite examines what a `cp_prove` proof reveals; the runs
  that did so are outside the suite. The suite never searches a run's output for stray secrets, and
  a search for keys derived from the seed has not been tried.
  **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="e4"></a>**E4** ([our judgement on hiding](limits.md#the-verdict)). A `YES` from the audit
  tool does not show that no mask was lost. The tool rejects a real model's files, and so does the
  option that keeps the random bytes for it, so on a real model nothing can be audited. The tool has
  been read in other sources, not in these. **What you can do:** See `tools/audit/README.md` for
  what a `YES` covers and does not cover. **[Basis](limits.md#how-a-thing-is-known):** reported.
- <a id="e7"></a>**E7**. A review of the source code, in a form that is not in this repository, left
  75 medium-severity findings unresolved. The repository does not list the 75 findings, so you
  cannot tell which of them affect soundness or weight hiding.
  **[Basis](limits.md#how-a-thing-is-known):** reported. More on the 75: row E7.
- <a id="e8"></a>**E8**. The CPU test of the serving loop fails now and then for no known reason and
  passes on a rerun, so we cannot rule out a rare fault in how the service orders a batch's steps.
  **What you can do:** Treat a pass as "passed at least once".
  **[Basis](limits.md#how-a-thing-is-known):** run, not on the released programs. See also row T45.
- <a id="e14"></a>**E14** ([a lost mask](limits.md#a-lost-mask)). At least eleven run-time
  self-checks of other programs are missing from the released programs. The eleventh, which has no
  name, is a comparison at every start of each row's count of random entries with an independently
  written rule. That rule is not in the repository, but what it guards exists. Whether the released
  programs need the other ten self-checks has not been examined.
  **[Basis](limits.md#how-a-thing-is-known):** read. Also row LF-8.
- <a id="e17"></a>**E17**. The source-reading tests only find names they already know. A hidden
  value printed under a name not on their word list goes unnoticed. These tests did not notice
  faults put into copies of the sources on purpose, among them secrets printed under the names of a
  batch's keys. We know of no source-reading test that looks at what the proof library's protocol
  steps send on the GPU engine, and none reads the code that opens the weight table.
  **[Basis](limits.md#how-a-thing-is-known):** read. Also row LF-9.
- <a id="e17.6"></a>**E17.6** ([E17](#e17)). About 900 calls in the programs print to standard
  output or standard error, and there is no list of them. The only check on them sees just the calls
  that mention a word such as seed, key, mask or salt, so a hidden value under another name gets
  through. **What you can do:** Keep what `cp_commit` prints inside the provider.
  **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="e18"></a>**E18**. We have no complete list of the places that hiding or soundness depends
  on. We do list 47 guards in `cp_commit` and `cp_prove`, but only a source-reading test checks
  them. That test fails if a guard's text is removed or disabled, but not if the guard is moved,
  bypassed or left in a file that no build compiles. For all but two of the 47 guards, no test shows
  that the guard works when a run needs it. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="t1.1"></a><a id="t2.3"></a><a id="t2.3.1"></a><a id="t2.4"></a><a id="t2.5"></a><a id="t4.3"></a><a id="e1.1"></a><a id="e1.2"></a><a id="e2"></a><a id="e2.1"></a><a id="e2.2"></a><a id="e2.3"></a><a id="e2.4"></a><a id="e2.5"></a><a id="e2.6"></a><a id="e2.7"></a><a id="e2.8"></a><a id="e2.8.1"></a><a id="e2.9"></a><a id="t5.2"></a><a id="t5.3"></a><a id="t6"></a><a id="t7"></a><a id="t7.1"></a><a id="t7.2"></a><a id="t8"></a><a id="t8.1"></a><a id="t10"></a><a id="t10.1"></a><a id="t11"></a><a id="t11.1"></a><a id="t11.2"></a><a id="t11.3"></a><a id="t11.3.1"></a><a id="t11.4"></a><a id="t11.4.1"></a><a id="e3"></a><a id="t12"></a><a id="t12.1"></a><a id="e4.2"></a><a id="e4.3"></a><a id="e4.4"></a><a id="e4.5"></a><a id="e4.6"></a><a id="e9"></a><a id="e9.1"></a><a id="e10"></a><a id="e10.1"></a><a id="e11"></a><a id="e12"></a><a id="e13"></a><a id="e14.1"></a><a id="e17.2"></a><a id="e17.3"></a><a id="e17.4"></a><a id="e17.5"></a><a id="e17.7"></a><a id="e17.8"></a><a id="e17.9"></a><a id="e18.1"></a><a id="e18.1.1"></a><a id="t15"></a><a id="t16"></a><a id="t36"></a><a id="e19"></a><a id="e20"></a><a id="e21"></a><a id="e22"></a><a id="e23"></a><a id="e24"></a><a id="e25"></a><a id="e26"></a><a id="t40"></a><a id="t40.1"></a><a id="p4.1"></a><a id="e29"></a>Further
  rows: [the table](limits-all-rows.tsv).

## What no test covers

This section names behaviour that tests exercise partly or never, so what the docs say about it
rests on source review or argument.

"Test" means a test that is in this repository and listed in the public table of `SUITE.md`. Other
tests may cover some gaps, but no run is known of some of them, and readers cannot run or inspect
them. No test in this repository runs a weakened or cheating copy of `cp_commit`, `cp_prove` or
`cp_verify`, or injects a fault into their running state. Tests that cheat do so with code of their
own, in test programs of their own, at toy size. None runs `cp_commit` or `cp_prove` at full size,
none gives `cp_verify` a cheating prover's full-size files, and none uses a real model's weights.

Worst first:

- **False claims.** Malicious provers and malicious committers are tested only at toy size or small
  fixed parameter sets, with the exceptions that row [SO-15.5](limits-all-rows.tsv) and "What has
  run" ([limits.md](limits.md#what-has-run)) name. A malicious prover ran at the default setting's
  small form, in one round; a malicious committer ran once at that form's copy with more room. What
  full-size runs did with altered copies of their own files, outside the test table, is told in
  [limits.md](limits.md#attacked-at-small-sizes-only). No test tries a tie for the largest score
  before `cp_prove` and `cp_verify` (the integer model's own test meets such a tie, without a proof:
  row T30.1). See also rows T30.2, SO-15.6, T37 and T22.
- **Leaks.** One untested line of `cp_prove` closes the output directory before memory is wiped.
  Without it, proofs made from wiped memory at exit would be released without masks, the random
  values meant to hide weights. No test makes `cp_commit` or `cp_prove` reject a parameter file that
  misses a condition for hiding. Other guards on columns, random values and memory layout are tested
  partly or never.
- **Side channels.** No test records the GPU operations a program issues, or shows that `cp_prove`
  prints the same lines when a request fails.
- **Failures.** No test kills or signals, on purpose, a program whose parameter file asks for proof
  of work, as the default does. The search for that work is one GPU kernel, and what a kill inside
  it leaves behind has not been tried. Also untested: batches near the size limit, a failing random
  source in a whole program, most wrong `cp_verify` calls, files that change mid-read, and the
  sections that messages cite. Nothing shows that tests reach every named error exit.

Do not take `NO` as proof that nothing crashed, and watch for starts that hang on a fresh machine.

- <a id="t18"></a><a id="t19"></a><a id="t20"></a><a id="t21"></a><a id="t22"></a><a id="t23"></a><a id="t23.1"></a><a id="t24"></a><a id="t25"></a><a id="t26"></a><a id="t27"></a><a id="t28"></a><a id="t29"></a><a id="t29.1"></a><a id="t30"></a><a id="t30.1"></a><a id="t30.2"></a><a id="t31"></a><a id="t31.1"></a><a id="t32"></a><a id="t33"></a><a id="t33.1"></a><a id="t33.2"></a><a id="t34"></a><a id="t37"></a><a id="t38"></a><a id="t39"></a>Further
  rows: [the table](limits-all-rows.tsv).

## Speed and size

- <a id="p1"></a>**P1**. Speed was measured at the [default setting](glossary.md#default-setting)
  (the sessions with whole replies checked: under a clipped-gap policy), in three whole runs of each
  job, partly in [the timed runs](limits.md#the-timed-runs), which not every job had, and partly not
  on the released programs. The figures are on [guide/performance.md](../guide/performance.md). It
  was measured on one H100 at a time, on shared machines, with requests of 1,024 tokens, in few
  serving sessions, of five or six batches each. The plain baselines behind the overhead rows were
  not measured in the timed runs. Of the services at 24 requests a batch on the public files: in the
  timed runs the six services of this job, at the same setting, lay between 110.07 and 112.04
  tokens/second steady; not on the released programs six like services lay between 100.81 and 111.55
  tokens/second, and three of the timed runs' six lie above that range (the timed runs' six are
  counted by another rule, every interval between two batches: row RES-4). The widest difference
  between two services at 24 requests a batch that ran one after the other on one machine, card and
  commitment was 0.7 % in the timed runs (108.44 and 109.23 tokens/second, in the like job on
  synthetic weights; the slower under the faster), and 6.3 % not on the released programs (104.56
  and 111.55 tokens/second); a gap between two rows of the figures no larger than the larger of
  these shows nothing about the programs, an option, the weights or how busy a machine was. The
  steady rate is the best case: it leaves out the program's start-up, the commitment, and the check
  of [duty 6](limits.md#the-conditions), which comes after a release; the first two have rows of
  their own there. Each proof behind the headline rate checks one token. The row for proofs that
  check whole replies under the clipped gap rests on the timed runs' three sessions, each with a
  session of its control. With nearly every token of a request checked, no rate has been measured at
  the default setting. **[Basis](limits.md#how-a-thing-is-known):** reported.

## Use

- <a id="r11.1"></a>**R11.1** ([R11](limits.md#a-killed-cp-commit)). Before it loads a weight,
  `cp_commit` tries the lock, the rename and the sync that the run needs, on one empty directory in
  the parent of each key directory. It tries each step once, so a file system that fails later, or
  only sometimes, is not found. A CPU test in this repository tries the routine with injected
  failures, not on a real file system without locks. **What you can do:** Make the commitment on a
  local disk. **[Basis](limits.md#how-a-thing-is-known):** read.
- <a id="so-49.2"></a>**SO-49.2** ([SO-49](limits.md#what-counts-as-a-pass)). A
  [part-call](glossary.md#part-call) of `cp_verify` (only `-key` or `-leaf`) without `-id`, and
  `cp_verify -plan id`, print a commitment ID computed from the very files they were given. That
  value says nothing about who published it, so if you pass it to a whole call as `-id`, the `YES`
  you get is about those files and no others. **What you can do:** Get the commitment ID from
  whoever vouches for the model, never from a call's own output.
  **[Basis](limits.md#how-a-thing-is-known):** read.
  - <a id="so-49.2.1"></a>**SO-49.2.1**. A request's proof by itself opens too few columns of the
    weight table to tie its root to one set of weights. That tie comes from the one-time checks of
    the key file and the leaf file, under the same commitment ID and settings. **What you can do:**
    Rely only on a whole call's `YES`. **[Basis](limits.md#how-a-thing-is-known):** read.
- Further rows: [the table](limits-all-rows.tsv).

<a id="h21.19"></a>Faults that these sources do not have (known from reading the code; nothing was
run where the first fault shows) are told in [the table](limits-all-rows.tsv): row H21.19, among
those that answer "does it hold of the released programs?" with "no: these sources do not have it".
