# What `cp_verify` prints

This page is a reference for the verifier's output. It covers the closing lines, the exit codes, the
rejection names and the format of the two files you write.

- For how to check a proof and when to accept the result, see
  [docs/verifying.md](../../docs/verifying.md).
- For how to build and run the program, see [verifier/README.md](../README.md).

**In short.**

- **Check this.** Accept only if a whole call exits with code 0, the last line is the verdict you
  expect, and every field of the `result-v1` line for which [its table](#the-line-for-programs)
  gives a comparison passes it. If the call ends any other way, treat the reply as unproved.
- **Never do this.** Never search the output for a word: take the verdict from the last line only.
  Never copy an ID or a number from the output into a later call.
- **The output cannot show this.** The checks above cover points 4 to 7 of the eight-point
  [acceptance rule](../../docs/verifying.md#when-to-accept), and the output cannot show you the
  rest. The figures in bits are the program's own count, not a security level.

These terms appear throughout:

- **Kinds of call.** A whole call runs all three stages (key, leaf and request), and it is the only
  kind that gives a verdict on a proof. A part-call (`-key`, `-leaf`) lets a committer test the
  files it made, and a planning call (`-plan`) does not check a proof at all.
- **The request.** Your request file holds T tokens followed by two numbers, r and e. Counting from
  0, the checked tokens are r + 1 to min(e, T - 1). The token after the request is y, and the
  verdict covers it only if e = T.
- **Acceptance policies.** Exact match and clipped gap: see [Acceptance
  policies](../../docs/verifying.md#acceptance-policies).
- **Words in messages.** Some messages and names give ordinary words a special meaning. For example,
  `entry` means the leaf file. [The glossary](../../docs/glossary.md) explains many others.

## The closing lines

A whole call that exits with code 0 or 14 ends with four lines, always in this order:

| Line | What it says |
|---|---|
| 1. `commitment:` | what the error bounds of the commitment's own files rest on ([below](#the-commitment-line)) |
| 2. `proof of work:` | how much proof of work was counted ([The line on proof of work](#the-line-on-proof-of-work)) |
| 3. `result-v1` | what the verdict covers, as named fields ([The line for programs](#the-line-for-programs)) |
| 4. the verdict | `YES`, `WITHIN LIMITS: ...` or `NOT-A-VERIFICATION` ([Exit codes](#exit-codes)) |

Further up, a whole call prints these lines among others:

| Line begins | What it says |
|---|---|
| `commitment ID:` | the ID that `cp_verify` computed from the files. Without `-id`, the line reads `commitment ID: not-shown-without-id`. The line is also printed at exit code 14, so rely on it only at exit code 0 |
| `id:` | that this ID equals the one you passed. Without `-id`, the line reads `id: not compared: the call named none` |
| `h_par=` | the digest of the three configuration files, which a planning call also prints as `configuration=`, in another order of digits. `cp_verify` does not compare it with anything, so do not rely on it |
| `weight hiding:` | `NOT CHECKED BY THIS CALL (it rests on how the commitment was made)`, in every accepted call; the `request` stage's output has the same line with `RUN` in the place of `CALL` ([what an accepted proof does not tell you](../../docs/verifying.md#what-a-yes-does-not-say)) |
| `covered:` | a line for human readers, which shows the same range and policy as `result-v1`. One routine builds both from the same values, so for the range and the policy you can compare either line: this page's rule names `result-v1`, the acceptance rule in docs/verifying.md names `covered:` ([its fields `y=` and `next-token=`](../README.md#run); [point 5 of the acceptance rule](../../docs/verifying.md#when-to-accept)) |
| `stage <name>: work:` | `work: A + B`: of all rounds of this stage, the one whose error is largest when proof of work is not counted has, by the program's own count, an error of 2^-A, and B bits of proof of work |
| `accounting:` | the program's own error accounting ([verifier/README.md](../README.md#the-error-accounting)) |
| `STAGE PASSED` | one stage's result. It is not an acceptance |

<a id="the-commitment-line"></a>**The `commitment:` line** reads
`commitment: <n> closeness rounds of the one-time files rest on another source than the bound within a third of the distance`
(or, for a single round, `1 closeness round ... rests ...`) and ends in
`; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given`.
The table's record is its published out-of-domain (OOD) answers (masked sums of the weights). Its count
decides nothing: no count turns a `YES` and exit code 0 into another ending. Point 7 expects this clause ([what that theorem is
worth](../../docs/limits.md#what-the-theorem-assumes)).

In these lines, "the bound within a third of the distance" is a name: a lemma proved in the paper that this
protocol builds on, of which this repository holds no proof. What the programs count on is an adapted
form of that lemma, which rests in part on written arguments that are not machine-checked ([what
that bound rests on](../../docs/limits.md#what-the-theorem-assumes)). No round's bound at the default setting rests on the lemma proved within a third of the distance; that count is reached for two mask codes and read by no round. One `accounting:` line says of the theorem on
codes "which we take as given and of which we have verified no proof ourselves". That line speaks of
the programs' accounting, which takes the theorem as given. [verifier/README.md](../README.md#the-error-accounting)
says what is proved of that theorem, and what is not.

In the `commitment:` line, "the one-time files" are the commitment's own files: the key file and the
leaf file. The line counts the [closeness rounds](../../docs/glossary.md#closeness-round) of these
two files that do not rest on the bound within a third of the distance. It does not count the request's own rounds.
[verifier/README.md](../README.md#the-error-accounting) lists the two sources that such a round's
error bound can rest on. The stages for the key file and the leaf file each print, further up, a line that begins `accounting: one-time files:`
and ends in `; the table has a record`. It says how many of that file's closeness rounds rest on
each source. A part-call prints that line too.

**Never copy an ID from the output into `-id`.** Every printed ID is computed from the very files
that the call was given, so it says nothing about who published it. A whole call given that ID says
`YES` about those files and no others. The planning call `-plan id` prints `commitment ID:` too. A
part-call (`-key`, `-leaf`) prints `computed commitment ID:` instead, even without `-id`, if it
passes or is run with `-show-stages`.

**When the stages' output is left out.** The stages' output is printed only if the call can still
pass. It is left out if a stage rejected, failed or ended in `NOT-A-VERIFICATION`, if `-id` is
missing, if `-bits` is below 128, if `-no-proof-of-work` rules out an acceptance, or in a test
build. Such a call prints only its closing part, starting at the heading `---- the call ----`. A
test build prints one line before it, which begins `TEST BUILD:`. After a rejection, that part
consists of the heading, the stage's name along with the check that failed, and `NO: <name>`. For
example, here is the entire output for the stored proof in `examples/accepted-proof` with one bit
flipped in the middle of the file (the suite's row `verifier.rejected_proofs` holds the last line;
the two above it are from one reported run†):

```text
---- the call ----
refused by the stage request: an opening does not lead to its root
NO: proof-refused
```

The rule for the stages' output has two exceptions:

- With `-show-stages`, the stages' output is printed anyway. Do not trust or copy anything in it.
- If the files belong to a different commitment ID, none of it is printed, even with that option,
  and there is no heading. The call prints two lines that begin `id:`, followed by
  `NO: model-statement`.

After printing the stages' output, `cp_verify` runs further internal checks. If one fails, which
means a bug in `cp_verify` itself, you see `STAGE PASSED` lines followed by `NOT CHECKED`.

**The acceptance rule in terms of these lines.** The full [acceptance
rule](../../docs/verifying.md#when-to-accept) applies to every call. This table restates its points
4 to 8. "Passes" means the comparison in the last column of [the table of
fields](#the-line-for-programs).

| Point of the rule | What `cp_verify` prints |
|---|---|
| 4, the verdict | The exit code is 0. Under exact match, the last line of standard output is exactly `YES`. Under a clipped-gap policy you named, if any token of the request was checked, the last line begins `WITHIN LIMITS: clipped-gap` and is never `YES`. |
| 5, what was checked | `id=`, `r=`, `e=`, `tokens=`, `y=`, `claim=` and `policy=` pass, and under a clipped gap `C=`, `theta=`, `cap=`, `s=` and `sh=` too. |
| 6, proof of work | `work=0`, and the line before it reads `proof of work: none counted`, unless you decided beforehand to count proof of work. No parameter file under `prover/examples/` yields these two, so under those files this point is met only if you decided beforehand to count proof of work. |
| 7, the commitment's rounds | `record=` and `commitment-assumed-rounds=` pass. The first of the four lines ends in the clause that begins `; the table has a record`. |
| 8, end tokens | Nothing. Compare the reply's last token (the last checked token, or y if e = T) with your end-token ids, and check that no earlier checked token is one of them. |

Read only the last line for the verdict, and never search the output for a word, because words
mislead in both directions. None of these is an acceptance: `NOT-A-VERIFICATION`; `PART PASSED`;
`PLAN ONLY`; the `STAGE PASSED` lines inside a whole call's output; and exit code 0 from another
program, such as the build script or `cp_prove`. Conversely, the words `NOT CHECKED` appear in every
accepted call, in the line on weight hiding.

## The line for programs

The third of the four closing lines is meant for programs. It is a single line: the fields appear in
the order of the table below, separated by single spaces, and no value contains a space. If any field or the order ever changes, so does the
first word.

This line is from the stored proof in `examples/accepted-proof`, at the default setting; its request
covers 3 tokens and y. A run of the prover† made that stored example, and `cp_verify` accepts it.
The default setting counts proof of work, so the line meets point 6 above only if you decided
beforehand to count it. Under a
parameter file that counts no work, `work=` and `work-rounds=` are 0, and `least-round` equals
`least-round-without-work`.

```text
result-v1 status=0 claim=exact-match id=<64 hex digits> bits=128 work=32 work-rounds=38 least-round=128.00 least-round-without-work=96.23 commitment-assumed-rounds=49 record=1 r=60 e=64 T=64 tokens=61..63 y=yes policy=exact-match C=256 theta=256 cap=0 s=16 sh=8 sT=4 F=24 above-clip=0 leaf=0 stages=key,leaf,request dw=8 build=<64 hex digits>
```

| Field | Value | Meaning | Compare it with |
|---|---|---|---|
| `status` | `0` or `14` | the exit code the call intends to return. Exit code 15 or a signal can still follow the line, so check the real exit code too | 0; no other value is an acceptance |
| `claim` | `next-token`, `exact-match` or `policy` | what the verdict covers. `next-token`: only y, as the model's top choice after the request (r = T - 1). `exact-match`: tokens of the request, each the model's top choice. `policy`: tokens of the request whose costs are within the limits of a clipped gap | what you wanted checked |
| `id` | 64 lower-case hex digits, or `not-shown-without-id` | the commitment ID you passed with `-id`, which every stage also computed from the files | the ID that you hold |
| `bits` | a whole number | the bits you asked for with `-bits`, or 128 without the option. It is a requirement, not a security level | 128 |
| `work` | 0 to 32 | 0 if every round reaches the parameter file's target without proof of work. Otherwise, the largest number of bits of work that the file declares for any kind of round | 0, if you decided beforehand not to count proof of work |
| `work-rounds` | a whole number | the number of rounds that reach the target only with their work. It is 0 if and only if `work=0` | |
| `least-round` | two decimal places | the bits of the weakest round, counting the declared work | |
| `least-round-without-work` | two decimal places | the bits of the weakest round without work | |
| `commitment-assumed-rounds` | a whole number | how many closeness rounds of the commitment's own files (the key file and the leaf file) rest not on the bound within a third of the distance but on [the theorem on codes](../../docs/limits.md#what-the-theorem-assumes). The request's own rounds are not counted here | nothing fixed, since the count differs from file to file: the sum, over the two lines of the same output that begin `accounting: one-time files:`, of the rounds that they count on the theorem on codes |
| `record` | `1` | the parameter file says `record 1`: the weight tables have OOD answers, and the binding of the weights rests on [the theorem on codes](../../docs/limits.md#what-the-theorem-assumes) | `1`. If you have decided not to rely on that theorem, accept no reply |
| `r`, `e`, `T` | whole numbers | the last two words of your request file, and the number of tokens in it | your own r and e |
| `tokens` | `<first>..<last>` or `none` | the checked tokens, counted from 0: r + 1 to min(e, T - 1). `none` (only when r = T - 1) means that no token of the request was checked | the reply tokens that you wanted checked |
| `y` | `yes` or `no` | whether the verdict also covers y (e = T). The token itself is not shown | `yes`, if you wanted y checked |
| `policy` | `exact-match` or `clipped-gap` | the kind of policy that the parameter file sets | the kind that you expect: `exact-match` if you passed no `-policy` |
| `C`, `theta`, `cap`, `s`, `sh` | whole numbers | the policy's numbers, as the check used them. Under exact match `C`, `theta` and `cap` are fixed (`C` = `theta` = 2^`sh`, `cap` = 0); `sh` itself is 7 or 8, and every example has 8 | under a clipped gap, the five numbers that you named with `-policy`; under exact match, `sh` = 8 |
| `sT`, `F` | whole numbers | two more lines of the parameter file, as the check used them | the line's own `sh` minus 4, and 24 |
| `above-clip` | a whole number | see [below](#the-field-above-clip) | |
| `leaf` | a whole number | the number of the leaf file that the proof names and the call checked | |
| `stages` | `key,leaf,request` | a constant, which shows that the line comes from a whole call | |
| `dw` | `8` | the number of words in a digest, a build-time constant | |
| `build` | 64 hex digits | the digest of the source code the program was built from. It does not cover the compiler or its flags. A tampered binary can print any digest here, so this field does not prove which build you ran | the build that you trust |

- **Status, claim and verdict go together.** With `status=0` and `claim=next-token` or
  `claim=exact-match`, the last line is `YES`. With `status=0` and `claim=policy`, it is
  `WITHIN LIMITS: ` followed by the policy's kind and `C=` to `sh=` with the same values. With
  `status=14`, it is `NOT-A-VERIFICATION`. The other fields are filled in even then. But the program
  skips four of its late checks on work and policy before printing such a line, so do not rely on
  any of that line's fields. `claim=next-token` can appear together with `policy=clipped-gap`, in
  which case y is checked as the top choice and no token of the request is checked under the policy.
- **What a program must check.** Exit code 0 alone is no reason to accept. For example, a call on a
  request with r = T - 1 ends with a `YES` about one token and says nothing about a reply. A call that exits with code 0 and prints `YES` rests
  the binding of the weights on the theorem on codes; the `commitment:` line says so. Of the four
  closing lines, only the `commitment:` line and the field `record=1` of the `result-v1` line show
  it. These pages claim no binding of the weights for the programs' own commitment
  ([binding](../../docs/glossary.md#binding)). Accept only if
  the exit code is 0, the last line is the verdict you expect, and every field with an entry in the
  table's last column passes that comparison. Parse the line the way `integrations/check_proof.py`
  does. Split standard output on line feeds only, and require exactly one line that begins
  `result-v1 `, which must be the second-to-last line. Match that line as a whole against the form
  you expect, not only the fields you look for. If any line begins `TEST BUILD:`, accept nothing.
- **When it is printed.** Only a whole call that is about to exit with code 0 or 14 prints it, and
  `-q` does not suppress it. The same calls print the `commitment:` line. No other line of standard
  output begins with `result-v1`, and at exit codes 0 and 14 the word appears nowhere else in the
  output. No printed line begins with text from a file. As a second safeguard, a stage's output that
  contains the word is not printed. The call then ends in `NO` if a stage rejected, and otherwise in
  `NOT CHECKED` (`internal-result-line`, unless another stage failed first). Look for `result-v1` followed by a space,
  because the stages' lines that begin `result:` are something else.
- **What it does not show.** No field names the source for the request's own closeness rounds,
  because `commitment-assumed-rounds` covers only the key file and the leaf file. The `accounting:`
  lines and a planning call (`-plan accounting`) say how many closeness rounds rest on each source
  ([The error accounting](../README.md#the-error-accounting)).
- **What it reveals.** The line shows r and e and nothing else from your request: no token, and no
  digest of tokens. If you forward a call's output and want to keep r and e private, leave out this
  line and the line that begins `covered:`. Two other outputs also leak something about r and e. If
  you pass `-next-token` but (r, e) is not (T - 1, T), the `request` stage's `result:` line and the
  closing line for that stage say `next-token`. Without `-policy`, if the parameter file sets a
  clipped gap and r is below T - 1, the call ends in `NO: policy` before any stage.

## The line on proof of work

The line reads either `proof of work: none counted` or
`proof of work: up to <n> bits counted in <m> rounds; with it the least round has <a> bits, without it <b> bits`
(`round` where m is 1). In the second form:

- n is the largest number of bits that the parameter file declares for any kind of round, and is at
  most 32.
- m is the number of rounds that reach the target only with their work.
- a is the bits of the round that is weakest when work is counted.
- b is the bits of the round that is weakest when it is not. The two can be different rounds.

Each is a single round's figure, and neither is a security level. A whole call prints both figures
in the `result-v1` line too: `least-round=`, with the work, and `least-round-without-work=`. A
part-call has no `result-v1` line: where its line reads `none counted`, the figures are only in its
`accounting:` lines. `integrations/check_proof.py` gives both in its own `result:` line. The printed
bits are truncated to two decimal places and are not the numbers that were compared. To find out
whether work was counted, test `work=0`. `proof of work: none counted` means only that no round
relies on proof of work, and says nothing about what else the bounds rest on. [Point
6](../../docs/verifying.md#when-to-accept) says what this line must read if you do not accept proof
of work.

## The field `above-clip=`

Under a clipped gap, a token that costs more than `C` can be any token at all ([Acceptance
policies](../../docs/verifying.md#acceptance-policies)). The field `above-clip=` in the `result-v1`
line says how many such tokens the limits allow. Let n be the number of checked tokens in the
request, and m = 2^`sh`.

- If `C` is above m, the field is min(`cap`, n, floor(n (`theta` - m) / (`C` - m))), where the
  quotient is taken as 0 if `theta` is m or less.
- Otherwise it is min(`cap`, n).
- Under exact match it is 0.

For illustration, m = 256, n = 100, `C` = 100000, `theta` = 40000 and `cap` = 2 give min(2, 100,
floor(100 × 39744 / 99744)) = min(2, 100, 39) = 2.

## Exit codes

Treat the reply as unproved after any exit code other than 0, whether or not the code is listed
here.

| Code | Last line | Meaning, and what to do |
|---|---|---|
| 0 | `YES`, or `WITHIN LIMITS: clipped-gap C=<C> theta=<theta> cap=<cap> s=<s> sh=<sh>` | The proof was accepted, under exact match or under the clipped-gap policy you named. Even so, accept the reply only if the whole [acceptance rule](../../docs/verifying.md#when-to-accept) holds. |
| 1 | `NO: <name>` | The proof was rejected, or one of your own files, your machine or the program itself is at fault ([the names](#rejection-names)). The name, or the line before it, says which check failed. |
| 2 | the usage text, whose last line says what was not accepted, or one sentence about `-id` or `-policy`, on standard error; or `input-unreadable-<role>: <the file> cannot be read: <reason>`, on standard output | The command line was wrong, or a file you passed is missing or unreadable. There is no verdict. Fix the call. |
| 3 | `PART PASSED: ...` | The result of a part-call (`-key`, `-leaf`), not a verdict on a proof. Do not piece a verdict together from part-calls, or from results you saved from earlier calls. A false statement about the weights can pass a single stage, so all three stages must pass on the same files, in one whole call. |
| 4 | `PLAN ONLY: no proof was checked` | The result of a planning call (`-plan`), not a verdict on a proof. |
| 5 | `NOT CHECKED: <name>` | The verifier could not finish its check, so nothing was verified. It is neither an acceptance nor a rejection. See below. |
| 14 | `NOT-A-VERIFICATION` | No check failed, but the call was not a valid verification. See below. |
| 15 | none | The output could not be written to standard output. Treat the call as if you had never made it. |

`cp_verify` does not limit its running time or memory use, so set limits yourself
([details](../../docs/limits.md#hostile-files)).

**Which ending wins.** If several apply, the first of these wins:

1. a bad command line (2);
2. an unreadable file (2) or a rejection before the stages (1), whichever of the two the call hits
   first;
3. a rejection by a stage (1): if one stage rejects and another is killed, the call ends in `NO`;
4. `NOT CHECKED` (5);
5. `NOT-A-VERIFICATION` (14);
6. an acceptance (0).

There are three exceptions. A few of the late internal checks run only where the call would
otherwise be accepted, so they come after `NOT-A-VERIFICATION`. Exit code 15 replaces any result
that the call returns at its end. And the program itself can end in `NOT CHECKED` outside the
stages, for example when it runs out of memory or cannot start a thread. Section 14.3 of
[docs/format/FORMAT.md](../../docs/format/FORMAT.md) has the full table.

**Exit code 1 in place of 2.** A call stops at its first fault, and a whole call checks the lengths
of the request file and the token file before it opens them or the proof file. So if either of the
two has the wrong length, the call ends in `NO: statement-range` or `NO: statement-required`, with
exit code 1, even if one of these three files is missing or unreadable. That `NO` is about the
length: no proof has been read.

**Exit code 5.** After `internal-stage` or `internal-exception-parFor`, the machine may be to blame
(it may have run out of memory, for example), so try again. But the fault may be in the program
itself, and `cp_verify` cannot distinguish between the two, so if the call ends the same way again,
please report it. Any other name will recur with the same files, so please report it
([SECURITY.md](../../SECURITY.md)). Whatever the name, `NOT CHECKED` is not an acceptance.

- **Some internal errors can appear without a fault in the program.** An `internal-` name normally
  means a bug in `cp_verify`. Three exceptions are known
  ([docs/format/FORMAT.md](../../docs/format/FORMAT.md), section 17.2). Public files with more than
  4096 stretches of padding in `ranges` end a call in `NOT CHECKED`. So can a file of several
  gigabytes, which is read whole before its length can be tested. And for rare values of its
  challenges, a proof that the protocol's verifier would accept can raise at least these eight:
  `internal-f-table-read`, `internal-t2f-untyped-<tree>`, `internal-t2f-untiled`,
  `internal-sym-rowvec`, `internal-f-tie-weights`, `internal-g-free-slot`, `internal-a-free-slot`
  and `internal-h-free-slot`, each preceded by its sub-proof's prefix. We have also not ruled out
  that a crafted file can trigger an `internal-` error. Either way, the call ends in `NOT CHECKED`
  (exit code 5), which is neither an acceptance nor a rejection. One more rare value of the
  challenges ends the call in `NO: coin-zero` instead: an honest proof is then rejected, so ask for
  a new proof.
- **`internal-stage` can be the program's fault, not only the machine's.** A stage ends in `abort()`
  when one of the library's guards fails, when a check in the C++ library or one from the hardening
  flags fails, on an uncaught exception and, in some places, when it runs out of memory. The parent
  process sees signal 6 in every case. Whatever ended the stage writes a line to standard error.

**Exit code 14.** The lines after `---- the call ----` give the cause.

- *Your command.* `-id` is missing, `-bits` is below 128, `-next-token` was passed with a request
  that does not ask only about y, or your `cp_verify` is a test build (the first line it prints says
  so). For a missing `-id` the line begins `the call: not the standard parameter form: id-named`,
  although the fault is in the command. Fix the command using only an ID and values that you held
  before you sent the prompt. Do not take anything from the provider's files or from this output. If
  you hold none, treat the reply as unproved.
- *Your rule on proof of work.* You passed `-no-proof-of-work`, and some round would miss the
  parameter file's target without its proof of work (the line begins
  `the call: not the standard parameter form: no-proof-of-work`). The call did what you asked, so do
  not drop the option to get a `YES`. By [point 6](../../docs/verifying.md#when-to-accept) you do
  not accept this proof. Every parameter file under `prover/examples/` counts proof of work, so among
  the settings in this repository there is no commitment you could accept in its place.
- *The provider's files.* The parameters are not in the [standard parameter
  form](../../docs/glossary.md#standard-parameter-form).
  `verifier/docs/parameter-file.md` lists the form's conditions, and the output names each one that
  is not met. One of them concerns the test on the committed table: `two-codes`. Treat the reply as unproved.

## Rejection names

This table explains the common rejection names. `verifier/src/vf_rejection_names.txt` lists them
all. For what the others mean, see section 13.8, "Every rejection by name", of
[docs/format/FORMAT.md](../../docs/format/FORMAT.md). A `NO` is not always the provider's fault, and
even an honest proof can be rejected.

| Name | Cause |
|---|---|
| `model-statement` | the files belong to a different commitment ID. Nothing in them was checked, and their own ID is not printed. Altered configuration files can also be rejected earlier, under a name such as `plan-layout` or `policy-call-line` |
| `proof-refused` | the proof file failed a check |
| `reply-differs` | the proof is about a different token y from the one in your token file |
| `cp-entry-missing` | the leaf file that the proof names is missing or unreadable |
| `statement-range`, `statement-required`, `token-range` | usually your own request file or token file is malformed: the length is wrong, r < e <= T does not hold, the token file does not hold 0 even though e is below T, or a token id is at or above the vocabulary size. `statement-required` can also come from the parameter file |
| `table-digest-<name>` | your machine's floating-point library computes one of the four public tables (silu, sigmoid, rope, exp) differently, so its digest does not match the one in the parameter file. The proof may be honest |
| `version-unknown (policy)` | the word on the first line of the parameter file is not this program's. The blank and the brackets are part of the name. If another file holds another word, the name is `cp-proof-head` (the proof file), `cp-key-head` (`key.bin`) or `entry-refused` (the leaf file) |
| `cp-bits-above-digest` | you asked for more than 128 bits with `-bits` |
| a name with `internal-` | the program or the machine failed before the stages started. This says nothing about the files. Treat it like exit code 5 |
| `plan-layout` | `layout.txt` is missing from the configuration directory, cannot be parsed as a layout, or does not match the proof's arrays |
| `policy` | one of the three causes listed below the table |

**The name `policy`.** The line before the `NO` says which cause applies:

- The parameter file's policy is not the one you named, or is of neither kind.
- You did not name a policy, but a token in the request is checked under a clipped gap.
- The limits you named are meaningless: `C` or `theta` is below 2^`sh`, `theta` is above `C`, or the
  limits could not reject any token.

The command line alone can cause this rejection, before any file is opened. If the file's policy is
not the one you expected, treat the reply as unproved.

## The request file and the token file

Both files consist of 32-bit little-endian words. The request file holds the T token ids, then r,
then e. The token file holds one word: y or 0. In the example in
[docs/verifying.md](../../docs/verifying.md#the-request-and-the-token-file) (range 9:14), with T =
64, the request file has 66 words: 10 prompt ids, 5 reply ids, the [filler
token](../../docs/glossary.md#filler) 49 times, then 9, then 14.

The filler tokens are part of the request, so you and the provider must pad with the same token. The
protocol does not say how the two of you agree on it, but the shipped tools on both sides pad with
token 0. Any token id below the vocabulary size is allowed, but with some filler tokens a request
may have no proof at all.

† Not on the released programs. What it ran on, and how far that speaks for them: [what was run on these programs, and what was not](../../docs/limits.md#what-was-run-on-these-programs).
