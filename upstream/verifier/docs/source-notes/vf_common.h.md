# Notes on `verifier/src/vf_common.h`

A comment in that file names an entry here by the key in brackets.

### [L:66e592269867] inside `refuse()`, at ``if (refuseThrows) throw Refusal{bare.empty() ? string("internal-refusal-name") : bare};``

An empty name would read as "none", so a thrown rejection takes the name `internal-refusal-name`. A rejection ends a process only here and in `cpStagesRun()`, which passes on a stage's line.

### [L:ef3e3f5b882a] before `mix64()`, at ``static u64 mix64(u64 z) {``

The SplitMix64 mixer, which is not cryptographic. It serves synthetic values and `funDigest()` (`vf_smallness_test_members.inc`).

### [L:fefe5f20a906] at ``static_assert(!versionWordAdmitted(1213018161u) && !versionWordAdmitte``

None of the twenty-six numbers in the assertion is accepted. Twenty-four of them are words that this format does not use: twenty-two version words and two format words of certificate files (1212367665, 2135734530). The assertion fails if a change of `versionWordAdmitted()` lets one of them in. The other two, 0x48434335 (1212367669) and 0x48434338, are format words of range files, not version words. A hashed list under one of the twenty-four could have the length and number of a list of this format; the first word differs.

### [L:84d37b91c124] before `TAU`, at ``static constexpr int LE = FIELD_BITS * CE;``

`TAU` is the number of random combinations in the approximate range check of a request's sub-proofs (printed: smallness test), from the parameter file's line `tau`. No static array is sized by it.

### [L:047ce97e2642] before `NOT_A_VERIFICATION_STATUS`, at ``static constexpr int NOT_A_VERIFICATION_STATUS = 14;``

The causes: a test build, a profile below 4 (line `prof` minus 16; 4 is the whole model), `-bits` below the threshold, the standard parameter form not met, no `-id` (a part-call needs none).

### [L:a19bbe35a24b] before `the macro WORKLINE`, at ``#define WORKLINE "proof of work: "``

Begins the line on proof of work. A call that ends YES, WITHIN LIMITS, PART PASSED or NOT-A-VERIFICATION prints it just before its last line (part-call) or the line `RESULTWORD` (whole call). It says how much was counted: a call that needs work to meet its target passes too.

### [L:c33c302c9004] before `the macro RESULTWORD`, at ``#define RESULTWORD "result-v1"``

First word of the line for programs, a whole call's last line but one at status 0 or 14. Its named fields, in a fixed order, say what the answer is about, so that a caller can compare each with its own value. It shows no tokens. Any change needs a new word.

### [L:32a702f1e759] before `struct WorkNote`, at ``struct WorkNote {``

If a file's rounds meet the target only with work, the stage notes the round with the largest error without work: that error in bits (for the printed line; never compared with the target), its work, its name. A stage that passes with a note ends STAGE PASSED WITH WORK; the parent checks that against its `work:` line and the call's record and prints `WORKLINE`.

### [L:2a77dd5f777f] before `struct Ex`, at ``struct Ex {``

An element c[0] + c[1] x + c[2] x^2 of E. No constructor takes one or two arguments, and `Ex` is no aggregate, so a brace list of one or two values is an error.

### [L:2b73e1f3d01f] before `eInv()`, at ``static Ex eInv(Ex x) {``

Inverse of x = (a, b, c) in E.

### [L:2315e951af98] before `struct Call`, at ``struct Call {``

One lookup call (the lookup is described in `docs/format/11-1-machinery.md`): 2^n keys, padding included, in 2^iota identities (trees of the lookup forest); 2^k table rows. `forestPad()` (`vf_lookup_argument.h`) sets `alpha` and `zk`. No code uses `Z`, `Y0`, `zb`, `vk` or `vb`.
