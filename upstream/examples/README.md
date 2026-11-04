# Examples

Start with `accepted-proof/`, which you can check without a GPU; [route 1 of the quick
start](../docs/quickstart.md#1-cpu-check-a-proof) shows how.

All stored proofs are for models with [synthetic weights](../docs/glossary.md#synthetic-weights), at
the [default setting](../docs/glossary.md#default-setting), and none was made by [the released
programs](../docs/limits.md#what-was-run-on-these-programs); the released `cp_verify` accepts each. Some settings in the stored configuration files come from work on real models;
[THIRD_PARTY.md](../THIRD_PARTY.md#what-depends-on-metas-models) says which ones, and on which model
each depends or may depend. The README in each proof's directory explains how its files were made.

By design, the programs mask proofs and a commitment's public files, apart from a [declared
list](../docs/what-is-not-hidden.md), and they do so by default. [Weight
hiding](../guide/weight-hiding.md) says what that gives.

| path | contents | what success looks like |
|---|---|---|
| [`accepted-proof/`](accepted-proof/README.md) | an honest proof for a small model, under [exact match](../docs/glossary.md#exact-match) | the last line of `cp_verify` is `YES` |
| [`accepted-proof-clipped-gap/`](accepted-proof-clipped-gap/README.md) | an honest proof under a [clipped-gap](../docs/glossary.md#clipped-gap) policy, for a second small configuration | the last line of `cp_verify` begins with `WITHIN LIMITS:` |
| [`full-size/`](full-size/README.md) | an honest proof for a full-size model, under exact match: one proof covers 1,001 tokens, 1,000 of a request of 1,024 and the token after it; you can check it without a GPU | the last line of `cp_verify` is `YES` |
| [`full-size-clipped-gap/`](full-size-clipped-gap/README.md) | an honest proof for a full-size model, under a [clipped-gap](../docs/glossary.md#clipped-gap) policy whose three limits were fitted to no model or reply: one proof covers 1,001 tokens, 1,000 of a request of 1,024 and the token after it; by a run of our own of the integer model, not by the proof, one of the 1,000 is the model's second choice, which we put in, and the others are its first; you can check it without a GPU | the last line of `cp_verify` begins with `WITHIN LIMITS:` |
| [`rejected-proofs/`](rejected-proofs/README.md) | a script, `try_altered_copies.py`, that makes altered copies of `accepted-proof/`, or of `accepted-proof-clipped-gap/`, and checks each one with `cp_verify` | its last line contains `altered sets: all rejected`, and its exit code is 0 |
| `protocol-toy-false-files/` | twelve small false files and four honest ones from the proof library's [toy proofs](../docs/glossary.md#toy-proof), with a test, `stored_false_test.py`; these files are not in the format that `cp_verify` reads | its last line begins with `stored_false_test.py: all 12 false files refused`, and its exit code is 0 |
| `quickstart-gpu.sh` | a script that runs [route 3 of the quick start](../docs/quickstart.md#3-gpu-a-small-model) on one H100: it builds the programs, commits to a small synthetic model, proves one request and checks the proof | exit code 0 |
| `maverick/` | a script, `run.sh`, that runs [route 4 of the quick start](../docs/quickstart.md#4-llama-4-maverick-the-full-size-model) on one H100: it takes a public 4-bit conversion of Llama 4 Maverick from download to a checked proof, with a calibration set that you make first; see its [README](maverick/README.md) | exit code 0 ([what has run](maverick/README.md#where-the-values-in-expectedtxt-come-from)) |

[SUITE.md](../SUITE.md) has the command for `protocol-toy-false-files/`, in its row
`examples.protocol_toy_false_files`. This repository stores no false files for `cp_verify` made by
cheating programs, so the trials with such files (row E28 of
[limits-all-rows.tsv](../docs/limits-all-rows.tsv)) cannot be repeated from it. `rejected-proofs/` alters an honest proof, and the toy files are not in the format that
`cp_verify` reads.

Every directory that holds a stored proof has the same layout;
[accepted-proof/README.md](accepted-proof/README.md#what-is-here) lists the files.

`cp_verify` prints `build=93afdaf592393a54ebd17b681d1c03295eb883f0161ff0e6ab872a3b39939595` in its
line `result-v1` if it was built from this repository's sources: a digest of the source files that
it is built from, not of the compiler or its flags. Yours should match it. If it does not, your
`cp_verify` was built from other sources than this repository's: find out why before you rely on its
answers. The example pages show a blank there. No test in this repository compares the word printed
here with what a built `cp_verify` prints.

When you check `accepted-proof-clipped-gap/` or `full-size-clipped-gap/`, pass the policy with
`-policy`, or `cp_verify` will not accept the proof. Each directory's `README.md` gives the full
command, and [acceptance policies](../docs/verifying.md#acceptance-policies) explains the option.

To see altered proofs rejected, build the verifier ([route
1](../docs/quickstart.md#1-cpu-check-a-proof)) and run `rejected-proofs/try_altered_copies.py` as
[its page](rejected-proofs/README.md#try-it) shows. Exit code 3 means the verifier accepted an
altered set. Please report that as described in [SECURITY.md](../SECURITY.md). Exit code 2 means a
wrong call, a faulty table or a missing file, and exit code 1 anything else.
