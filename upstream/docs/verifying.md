# Verifying a proof

This page is for a client who wants to check the proof of a reply. It says how to check the proof,
when to accept the reply, and what an acceptance means. Start before you send the prompt: some
values must be in your hands by then.

A provider commits once to a model's weights and publishes a **commitment ID**, 64 hex digits long.
It can then attach a proof to a reply. Under exact match, a `YES` is meant to show that each checked
token is the top choice of the committed integer model at its position, up to the error bound ([what
it shows](#what-a-yes-means); [how far you can rely on it](#what-a-yes-does-not-say)). The verifier,
`cp_verify`, never sees what you sent. So exit code 0 means nothing until every point of [the
acceptance rule](#when-to-accept) holds. Some of the 128 bits a check requires (a per-round target
the program computes, not a security level) come from proof of work, at the parameter files under `prover/examples/`, and the verdict does not say so. If you do not accept that, you accept no proof made under those files ([point 6](#when-to-accept)).

## Check a proof

1. Before you send the prompt, collect [what you need](#what-you-need-and-where-it-must-come-from):
   the commitment ID, the model's end-token ids and its tokenizer. Unless the policy is exact match,
   you also need the policy values you will accept. Never take any of these from the server, from a
   program's output or from files that came with a proof. If one is missing, stop and ask for it.
2. Choose [where to work](#keep-the-check-honest), build the verifier `cp_verify` there, and
   download the public files. `cp_verify` runs on a CPU.
3. Send the prompt as token ids and keep the token ids of the reply. [Build the request file and the
   token file](#the-request-and-the-token-file) from them.
4. Download the proof.
5. [Run `cp_verify`](#run-the-check) with your own values, or let the [check
   script](#with-the-check-script) make the call and most of the comparisons. Either way, set limits
   on time and memory yourself ([hostile files](limits.md#hostile-files)).
6. Accept the reply only if every point of the [acceptance rule](#when-to-accept) holds. Otherwise
   treat it as unproved.

## What you need, and where it must come from

A provider that can choose a value after seeing your prompt can fit that value to the answer it
wants. So get the first four items before you send the prompt, and not through the server you are
checking.

- **The commitment ID.** Get it from the party that vouches for the model, that is, the one you hold
  responsible for it: the provider, which computes and publishes the ID, or a third party. Get it
  over a channel that the server you are checking does not control and, where you can, from a place
  that shows every client the same ID ([why](limits.md#who-chooses-the-model)). The provider's own
  page can be that place, but only as far as it is public and others watch it. An ID that reaches
  you through the server you are checking, alone or with the files and the proof, will not do: then
  a `YES` shows only that they fit together. Always compare the whole ID. If the provider gives you
  a new commitment ID: no proof links it to the old one. [What holds under one ID, what forces a new
  one, and what you can do](../guide/verify-a-proof.md#when-the-commitment-id-changes).
- **The tokenizer and the chat template** published with the model.
- **The end-token ids.** They are in the model's published `generation_config.json`. Neither the ID
  nor a proof covers them.
- **The policy values,** if the policy is clipped gap rather than exact match. They are lines in the
  published parameter file, which your commitment ID covers completely. Read them there when you get
  the ID, and decide then whether to accept that policy ([which lines, and how to
  decide](#acceptance-policies)). Do not take them from files that arrive with a response.
- **The public files.** The key file and the leaf files may come from anyone, the provider included:
  `cp_verify` checks them on every call. Hold the three configuration files as you hold the ID: the ID
  covers them by design, but the soundness theorem on `verify` is stated at given public files and does not
  itself say that one ID cannot verify under two sets of them. That the key file and the leaf files
  may come from anyone rests on a [technical report that is not public and not
  refereed](glossary.md#technical-report): on an argument about the protocol only, whose adversary
  makes them with the hash in hand, and which does not cover the commitment ID as the programs form
  it, which also hashes the out-of-domain (OOD) seeds. The three configuration files (`policy.txt`, the parameter file with
  the acceptance policy; `KL.bin`, the scales; `layout.txt`) are taken as fixed before the hash is
  drawn; a provider that picks them with the hash in hand is covered only by a lemma whose hypotheses
  are not shown for the files that `cp_verify` takes.
  No result in Lean covers public files that a provider picks (second point of [What an accepted
  proof does not tell you](#what-a-yes-does-not-say)). Whoever serves you the leaf files
  learns which one you ask for, and when.
  - The configuration directory holds `policy.txt` (the parameter file), `KL.bin` and `layout.txt`.
    The commitment ID covers all three completely. The parameter file determines what an acceptance
    rests on, so read it once for each commitment, even when it is the default ([its
    lines](../verifier/docs/parameter-file.md); [which setting is it?](#which-setting)).
  - The public key directory holds `key.bin` and the [leaf files](glossary.md#leaf-file) `leaf/<k>.bin`. A [proof file](glossary.md#proof-file) begins with the number k of the leaf file that it needs. The ID covers only a part of `key.bin` ([what the ID does not
    cover](limits.md#what-the-id-does-not-cover)).
- **The proof.** To make it, the provider needs your tokens and the range. It responds with either a
  proof file and a token file, or a notice that the request failed, with no reason given. How you
  send the tokens and download the response depends on the provider's service. For the example front
  end, [integrations/README.md](../integrations/README.md#a-reply-of-your-own) has the commands,
  with the example client and without it.

## Keep the check honest

Anyone who can write to the files that the check reads can decide what you see. If you are the only
user of the machine, work in any directory of your own. On a shared machine, create a private
directory first:

```
mktemp -d /tmp/check.XXXXXXXX
```

The command prints the name of a new directory that only your account can read or write. No program
of this repository removes that directory, but many systems clear `/tmp` on reboot or after a few
days, so keep a copy of your record of the ID elsewhere. Do not change the permissions of a
directory you share with others.

1. **Keep everything in that directory:** the repository, the verifier you build, your record of the
   commitment ID and the other values, and every file you pass to the verifier.
2. **Build the verifier yourself,** from sources you trust. A program that came with the files can
   print anything. From the root of the repository, run:

```
verifier/build.sh
```

   This builds only `cp_verify` and puts it in `bin/`. On success the exit code is 0, and the last
   line of output begins `THE VERIFIER IS BUILT:`. A line that begins `build.sh: note:` names a
   directory that another account may be able to change. The build continues, but on a shared
   machine do not rely on it ([build notes](../verifier/README.md#build)). Run every command in this
   document from the repository root.
3. **Look for symbolic links** (`ls -lR`) in any directory that someone else packed. `cp_verify`
   follows links, so the name `key.bin` or `leaf/<k>.bin` could lead to another file. If you find a
   link, do not use that directory.

## The request and the token file

A request holds a window of exactly T tokens: the prompt, then the reply, then [filler
tokens](glossary.md#filler). T = 2^`lt`, where `lt` is a value set in `policy.txt`. At [full
size](glossary.md#full-size), T is 1,024. Positions are counted from 0.

The request also holds a range r:e. For each position t from r to e - 1, the proof is about token
t + 1, given tokens 0 to t. **To check a whole reply of R tokens (its end token included) after a
prompt of P tokens, set r = P - 1 and e = P + R - 1.** For example, a prompt of 10 tokens and a
reply of 5 occupy positions 0 to 14, and the range is 9:14. With r = P, the reply's first token
would go unchecked.

Alongside the request file there is a token file, which holds one token, y. If the prompt and the
reply fit in the window, e is less than T, y is unused, and the file must hold 0 (four zero bytes).
If they add up to T + 1 tokens, the reply's last token goes there as y, and e = T. A request with
r = T - 1 and e = T asks only about y. Under either policy, y must be the model's top choice.

A proof matches exactly one request file: all T tokens, r and e. So you must pad with the same token
as the provider. The tools on both sides pad with token 0 ([the format of both
files](../verifier/docs/output.md#the-request-file-and-the-token-file)).

This command builds the request file for a prompt and a reply that fit in the window. It never
builds a request with e = T. `prover/tools/make_request.py` does: given a file of your T tokens as
its third argument, and no range, it writes r = T - 1 and e = T. The call that
[operators-duties.md](operators-duties.md#from-nothing-to-the-first-proof), step 16, prints is for a
trial: it names no file of tokens, so its request holds synthetic tokens, not your prompt. Run the
command below on your own machine:

```
python3 integrations/spool.py make-request T PROMPT_FILE REPLY_FILE END_TOKEN OUTPUT_FILE
```

- `PROMPT_FILE` and `REPLY_FILE` are text files of decimal token ids, separated by blanks, line ends
  or commas.
- `END_TOKEN` is the id of one end token, or `none` if your own length limit stopped the reply. The
  command checks that the reply ends with that token and does not contain it earlier, but it does
  not add the token.
- `OUTPUT_FILE` must not exist yet. Its directory, and every directory above it, must be one that no
  other account can change; otherwise the command writes nothing and exits with code 2. A directory
  from `mktemp -d` qualifies.

A second command, `python3 driver/spool.py request`, also writes a request file. It reads
token ids as little-endian 32-bit words and appends the end token if the reply does not end with it,
so neither command can stand in for the other ([driver/README.md](../driver/README.md)).

The command prints the range, `9:14` in the example. It writes only the request file. For such a
request the token file is four zero bytes
(`python3 -I -c 'import sys; open(sys.argv[1], "xb").write(bytes(4))' TOKEN_FILE` writes them, and
fails if that name is taken). The response includes a token file as well. Do not pass it to
`cp_verify`: [the check script](#with-the-check-script) reads it only to compare it with your value.

If the server left the end token out of the reply, you can add it to `REPLY_FILE` yourself. But a
provider that proves replies only as it served them will not prove a token you added. In that case,
require that the served reply itself ends with an end token.

## Run the check

To try the call on a stored proof first, and to see a rejection, follow [the quick
start](quickstart.md#1-cpu-check-a-proof).

This command checks a reply of your own under exact match. Replace the words in capitals:

```
bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID
```

Run `echo $?` straight afterwards to see the exit code.

`cp_verify` does not limit its running time or its memory, so set limits yourself ([hostile
files](limits.md#hostile-files)), no lower than an honest check needs on your machine
([guide/performance.md](../guide/performance.md)). In bash, this form allows
each process 8 GiB of address space, and the whole call ten minutes:

```
( ulimit -v 8388608; timeout 600 bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID )
```

Under these two limits the stored full-size proof (default setting, synthetic weights) was
accepted. With 2 GiB the same call ended in `NOT CHECKED: internal-stage`, with exit code 5, and
with five seconds in exit code 124 and no verdict (each tried once, not on the released programs;
the limits set with `prlimit`, not in the form printed here). Treat
a call that hangs or is killed as not accepted.

Put options after the five paths, because an option before the paths is read as a path. A client
needs only these ([the others](../verifier/README.md#run)):

- `-id` takes your commitment ID, all 64 hex digits. Without it, `cp_verify` accepts nothing. At
  best it prints `NOT-A-VERIFICATION` and exits with code 14.
- `-policy clipped-gap,C,THETA,CAP,S,SH` names the policy you expect, if it is not exact match.
  Every value must equal the corresponding line of `policy.txt`; otherwise the proof is rejected
  with `NO: policy`.
- `-no-proof-of-work` is for a client who accepts no counted proof of work (point 6): if any round
  reaches the parameter file's target only with its work counted, the call ends in
  `NOT-A-VERIFICATION` at best. At every parameter file under `prover/examples/` for which a
  planning call can be made, some round does. Under a provider's own file that counts none, an
  acceptance would still rest on that file's source for its closeness rounds and on numbers that
  nothing checks at the round (row SO-4 in [the table of all rows](limits-all-rows.tsv)).
- `-j N` sets the number of threads (1 to 256; the default is 1).
- Do not pass `-bits`. Without it the target is 128 bits, and a call that asks for fewer ends in
  `NOT-A-VERIFICATION`.

## When to accept

Accept a reply only if all eight points below hold. In any other case the reply is not proved. That
includes any other exit code, a rejection, a missing `-id`, or a field on the `covered:` line that
differs from your value.

1. **Your verifier.** You built `cp_verify` yourself from the source in this repository, or got it
   from a source you trust just as much ([Keep the check honest](#keep-the-check-honest)).
2. **Your files.** You built the request file from the prompt you sent and the reply you received,
   and you wrote the token file yourself. You did not use any request file, token file or ID that
   came with the proof. `cp_verify` accepts the tokens before the checked ones exactly as the
   request file gives them, so they must be the ones you sent. Whatever uses the reply after the
   check gets exactly the prompt and reply tokens you put in the request file, and the y you put in
   the token file.
3. **Your ID.** You got the commitment ID before you sent the prompt, from the [party that vouches
   for the model](#what-you-need-and-where-it-must-come-from), over a channel that the server you
   are checking does not control. You passed it with `-id`.
4. **The verdict.** The exit code is 0, and the last line of standard output is exactly `YES`. If
   you named a clipped-gap policy, the last line begins `WITHIN LIMITS: clipped-gap` instead, unless
   your request asks only about y, in which case it is still `YES`. You did not pass `-bits`. Do not
   search the output for a word such as `YES`, because other lines can contain it.
5. **What was checked.** The line that begins `covered:` matches values you already hold:
   - `r=` and `e=` are yours.
   - `tokens=` lists the reply tokens you wanted checked. `tokens=none` means that only y was
     checked; expect it only when r = T - 1, where the prompt fills all T positions of the window.
   - If you wanted y checked, the line shows `y=yes`.
   - `policy=` is the policy you expect. Under a clipped gap, which [proves less than exact
     match](#acceptance-policies), the policy values on the line are the ones you named. Under exact
     match, ignore `C`, `theta` and `cap`, because they are fixed: `C` = `theta` = 2^`sh`, and
     `cap` = 0. `sh` itself is 7 or 8, and every example has 8; require 8: no theorem or written
     argument covers 7 (row SO-6.3.1.1 in [the table of all rows](limits-all-rows.tsv)).
6. **Proof of work.** Some of the bits the check counts come from proof of work, at the parameter files under `prover/examples/`, and you decided beforehand whether to accept that. Unless you pass `-no-proof-of-work`,
   counting proof of work changes neither the verdict nor the exit code, so read the line that
   begins `proof of work:` ([its numbers](../verifier/docs/output.md#the-line-on-proof-of-work)). If
   you do not accept it, that line must read `proof of work: none counted`; no parameter file under
   `prover/examples/` yields that line, so you accept no proof made under any of them.
   Whichever you decide, you rely on things that are not proved here:
   - With the work, the assumption that it may count for the programs' hash chains ([what counting
     the work assumes](limits.md#how-the-programs-count)).
   - With or without the work, [the theorem on codes](glossary.md#theorem-on-codes), which the
     programs' accounting takes as given; a statement on codes held to follow from it is proved in
     the Lean library ([docs/format/FORMAT.md](format/FORMAT.md), 13.7, condition 4). At the default
     setting every closeness round rests on that theorem (the planner counts none on [the bound within a third of the distance](glossary.md#proved-lemma)), and with them the binding of the weights ([what the bounds
     rest on](limits.md#what-the-theorem-assumes)). That the library's statement follows from the
     paper's, and that these programs' parameters meet the theorem's conditions, rest on reading,
     not on a proof, and the theorem is used with no bits to spare.
   - The closeness rounds' error bounds also rest on further steps argued in a technical report that
     is not public and not refereed.

   What Lean proves of the Lean verifier is that any commitment, with its committer's hash queries,
   settles at the given public files one set of weights or none before any proof (index binding); of
   the programs' own commitment no such theorem is proved ([more](glossary.md#binding)).
7. **The commitment line.** Read the line that begins `commitment:`. It says
   that the table has a record (the programs' word for published out-of-domain (OOD) answers, which
   are masked sums of the weights) and that the binding of the weights rests on [the theorem on
   codes](limits.md#what-the-theorem-assumes), in these words:
   `; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given`.
   Refuse if the line is missing or lacks these words. If you do not take that theorem as given, you accept no proof; say so to whoever relies on your check. The exit code and the last line do not show what the binding of the weights rests on. This line
   says what the error bounds for the key file and the leaf file ("the one-time files") rest on, and
   nothing about a request's own rounds ([the closing lines](../verifier/docs/output.md#the-closing-lines)).
8. **The end of the reply.** If you need the complete reply, its last token (the last checked token,
   or y if e = T) is one of your end-token ids, and no earlier checked token is. `cp_verify` does
   not know any end tokens, so a reply that was cut short verifies as a shorter reply. Under a
   clipped gap an end token is scored like any other token, so a false end is limited only as much
   as any other wrong token (row SO-29 in [the table of all rows](limits-all-rows.tsv)).

**Where to find these lines.** A whole call that exits with code 0 or 14 ends with four lines,
always in this order: the `commitment:` line, the `proof of work:` line, the `result-v1` line and
the verdict. Find the `covered:` line by its first word, not by counting lines from the end. The
`result-v1` line repeats what `covered:` shows, in [named fields for
programs](../verifier/README.md#the-line-for-programs). `NOT-A-VERIFICATION`, `PART PASSED`,
`PLAN ONLY`, a `STAGE PASSED` line, and exit code 0 from another program are not acceptances.

For the stored proof in `examples/accepted-proof`, at the default setting, the call that [the quick
start](quickstart.md#1-cpu-check-a-proof) prints ends with these lines. The proof is for a small
model, and its request covers 3 tokens and y. Dots mark what is left out. The stored example was not made by the released programs; this sample is what the released `cp_verify` prints on it: the row `verifier.call_accounting.examples` of [SUITE.md](../SUITE.md) runs that call.

```text
commitment ID: 79a5a2bc...
id: named by the caller and equal
h_par=0adb76ee...
weight hiding: NOT CHECKED BY THIS CALL (it rests on how the commitment was made)
covered: r=60 e=64 T=64 C=256 theta=256 cap=0 s=16 sh=8 positions=60..63 tokens=61..63 y=yes next-token=no policy=exact-match above-clip=0 (...)
stage key: work: 96.23 + 32 (...)
stage leaf: work: 96.50 + 32 (...)
stage request: work: 100.11 + 28 (...)
commitment: 49 closeness rounds of the one-time files rest on another source than the bound within a third of the distance; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given
proof of work: up to 32 bits counted in 38 rounds; with it the least round has 128.00 bits, without it 96.23 bits
result-v1 status=0 claim=exact-match id=79a5a2bc... bits=128 work=32 work-rounds=38 least-round=128.00 least-round-without-work=96.23 commitment-assumed-rounds=49 record=1 r=60 e=64 T=64 tokens=61..63 y=yes policy=exact-match (...)
YES
```

For the range 9:14 from the earlier example, the `covered:` line shows `r=9 e=14`,
`positions=9..13`, `tokens=10..14` and `y=no`. The line `proof of work:` gives two figures in bits.
Each is a single round's figure, and neither is a security level. A round's bits are not the chance
of a forgery: that chance grows with every hash computed, the verifier's own included, and even the
figure without the work still rests on results that are cited, one of them taken as given, and on
the hash taken as a random oracle. The figures in bits on the earlier `accounting:` lines are a
calculation, not a security level, and no theorem is about the three programs ([what they are
worth](limits.md#the-soundness-figure)). [The closing
lines](../verifier/docs/output.md#the-closing-lines) explains the sample's other lines and shows a
rejected call.

**Compare, never copy.** Apart from the exit code, the verdict and the lines on the commitment and
on proof of work, use the output only to compare it with values you already hold. Never copy an ID,
a file name or a number from the output into a later call, and never base a decision on one.

## Exit codes

Read the exit code together with the last line of standard output. Only code 0 can lead to
acceptance. For any other code, listed here or not, treat the reply as unproved.

| Code | Last line | Meaning, and what to do |
|---|---|---|
| 0 | `YES`, or `WITHIN LIMITS: clipped-gap C=<C> theta=<theta> cap=<cap> s=<s> sh=<sh>` | The proof was accepted, under exact match or under the clipped-gap policy you named. Accept the reply only if the whole [acceptance rule](#when-to-accept) holds. |
| 1 | `NO: <name>` | The proof was rejected, or one of your own files, your machine or the program itself is at fault ([the names](../verifier/docs/output.md#rejection-names)). A `NO` is not always the provider's fault, and an honest proof can be rejected. |
| 14 | `NOT-A-VERIFICATION` | No check failed, but the call was not a valid verification. The lines after `---- the call ----` give the cause, which is your command, the provider's files, or your own `-no-proof-of-work` ([the causes](../verifier/docs/output.md#exit-codes)). If you fix the command, use only values you had before you sent the prompt, and do not drop `-no-proof-of-work` to get a `YES`. If you have none, or if the cause is in the provider's files, treat the reply as unproved. |
| 5 | `NOT CHECKED: <name>` | The verifier could not finish its check, so nothing was verified. It is neither an acceptance nor a rejection. If the name is `internal-stage` or `internal-exception-parFor`, the machine may have run out of memory, so retry once. If you get the same result, or any other name, please report it ([SECURITY.md](../SECURITY.md)). A malicious provider may be able to turn a `NO` into it, so it says nothing in the provider's favour. |
| 2, 3, 4, 15 | [see the reference](../verifier/docs/output.md#exit-codes) | No verdict on a proof: a bad command line or an unreadable file (2), a part-call (3), a planning call (4), or output that could not be written (15). Never piece a verdict together from part-calls, or from results you saved from earlier calls. |

## With the check script

`integrations/check_proof.py` makes the call and most of the comparisons for you. You pass your own
values as options: the commitment ID, the range, the end tokens, y if the request asks about it, and
the policy. The script first checks the files against your ID, then runs `cp_verify` and compares
the `result-v1` line with your values. [integrations/README.md](../integrations/README.md) documents
its options and output, and row SO-66 in [the table of all rows](limits-all-rows.tsv) says how the
script is tested.

- It cannot compare the tokens before the range, so build the request file yourself (point 2 of the
  rule).
- It exits with code 0 (`result: outcome=verified` or `outcome=within-limits`), 1 (rejected), 2 (an
  error) or 14 (not a verification).
- By default it exits with code 0 even if proof of work was counted. In that case the `work=` field
  and the last line say so. With `--no-proof-of-work` it exits with code 14 instead (point 6).
- Its `result:` line has `record=1`, and its last line repeats the `commitment:` line (point 7).
- Under a clipped gap its `--C` and `--s` must equal the parameter file's values, but its `--theta`
  and `--cap` are upper limits on them, and it accepts only `sh` = 8. `cp_verify -policy`, by
  contrast, requires all five values to be equal.
- Under every policy it accepts only `sh` 8, `sT` 4 and `F` 24; `cp_verify` itself also admits `sh`
  7 (row SO-6.3.1.1).

<a id="what-a-yes-means"></a>

## What an accepted proof tells you

Under exact match, a `YES` is meant to show that each checked token is the top choice of the
committed integer model at its position, up to [the error bound](limits.md#the-soundness-figure)
([in full](limits.md#what-a-yes-covers)). The committed model is the [integer
model](glossary.md#integer-model) that the commitment and your configuration files define. The
second point of the list below says how far that is proved.

For a clipped gap, see [Acceptance policies](#acceptance-policies). If you pass an accepting answer
on, pass the range of checked tokens and the policy with it, and never a number of bits that the
verifier prints as a proved level.

<a id="what-a-yes-does-not-say"></a>

## What an accepted proof does not tell you

[limits.md](limits.md) describes most of these limits in full.

- **Whose model it is, or whether it is a sensible one.** The provider alone chooses the weights,
  the scales and every setting, so an accepted proof is only as good as your knowledge of whose ID
  it is ([who chooses the model](limits.md#who-chooses-the-model)).
- **That the soundness theorems apply to the programs.** Lean proves its own verifier, `verify`, sound at exact
  match, under stated conditions ([what the Lean proofs
  cover](limits.md#what-the-lean-proofs-cover)); `lean-verify` runs its executable form. Nothing
  proves that the programs carry out the protocol that the theorems are about ([no theorem is about
  `cp_commit`, `cp_prove` or `cp_verify`](limits.md#which-verifier-has-a-theorem)). That soundness
  holds with any key file or leaf file that passes the checks, whoever made it, is argued only for
  the protocol, in a technical report that is not public and not refereed ([what the ID does not
  cover](limits.md#what-the-id-does-not-cover); [row LIM-2](limits-all-rows.tsv)). The programs are
  known to differ from that protocol, and the argument does not cover three points of the programs:
  what a proof's first hash takes in, the labels of two of a proof's chains, and how the commitment
  ID is formed. No parameter file, those in this repository included, is known to meet all its hypotheses: a file can pass every check of `cp_verify` and still
  violate one ([how the programs count](limits.md#how-the-programs-count)).
- **Whether proof of work was counted, or what the error bounds rest on.** The verdict and the exit
  code do not show these ([points 6 and 7](#when-to-accept)).
- **Which engine produced the tokens,** or how they were sampled.
- **Anything about your prompt.** `cp_verify` takes tokens 0 to r from your request file as they are
  ([what is covered](limits.md#what-a-yes-covers)).
- **Anything about a token after the range,** or about y if e is less than T.
- **Anything about text.** The proof is about token ids, not about wording or detokenisation.
- **Whether the reply is complete.** Check that yourself (point 8).
- **Why a request failed.** A provider's "failed" notice takes the place of a proof. It is not a
  rejection.
- **Whether the proof was made for you, or recently.** A proof does not name a client, a time or a
  nonce, and accepted proofs are not unique. A valid proof verifies for anyone who holds the same
  request file, ID and configuration. If freshness matters, put your own nonce in the prompt, and
  keep the request, the reply and the proof private. `verify`'s executable form has accepted parsed
  proof files changed, for the same statement, at entries it does not read (by reading it holds of
  `verify` as it is; seen in runs over Lean files that differ from this repository's); none such is
  known to pass `cp_verify`. For `cp_verify`, no one has searched for a change that would still be
  accepted for the same statement ([no freshness](limits.md#no-freshness); what was tried: [at the
  default setting](limits.md#few-runs), and rows SO-19.2 and SO-19.2.1 in [the table of all
  rows](limits-all-rows.tsv)).
- **Anything about weight hiding.** No verifier checks whether the provider masked the proof or the
  public files ([where weight hiding stands](limits.md#weight-hiding)).

## Acceptance policies

| Policy | Name in `-policy` and `policy=` | What an accepted proof is meant to show about the checked tokens |
|---|---|---|
| exact match | `exact-match` | each is the top choice of the committed integer model |
| clipped gap | `clipped-gap` | their costs (defined below) meet the limits you named |

Under a clipped gap, a call that accepts the proof ends in `WITHIN LIMITS`, not `YES`. The exception
is a request that asks only about y: it ends in `YES` under either policy, because no token inside
the window is checked.

A clipped-gap acceptance says only that the checked tokens' costs meet your limits. It does not say
that the reply is the model's. What it says about a single token depends on your limits, and can be
nothing. Even limits fitted to the honest model say little about which of two nearby models served a
reply ([what such an acceptance tells you, and what it rests on](limits.md#within-limits)).

**How the limits work.** Each checked token has a cost: cost = gap + m, where m = 2^`sh`. The gap is
the distance between the token's logit and the largest logit, plus 1 if a token with a lower id has
the largest logit. So only the top choice has gap 0 and costs m. An accepted proof means two things:

- the costs, each capped at `C`, average at most `theta`;
- at most `cap` tokens cost more than `C`.

`s` is the exponent the provider states for the model's logits. `cp_verify` compares it with yours,
but no check confirms that it is true.

**What the limits do not give you.**

- **A token that costs more than `C` can be any token at all.** The `above-clip=` field on the
  `covered:` line says how many such tokens the limits allow ([how it is
  computed](../verifier/docs/output.md#the-field-above-clip)). That allowance grows with the number
  of checked tokens, up to `cap`, and it is the provider that sets the reply's length.
- **The other tokens can be wrong too.** For illustration, take m = 256, 100 checked tokens, `C` =
  100000, `theta` = 40000 and `cap` = 0. Then `above-clip=` is 0, yet all 100 tokens can trail the
  top choice by 39,744 units each. Where the committed model's logits lie close together, even a
  token that costs `C` or less can be any token, and no verifier can tell.
- **The provider writes the values** into the parameter file (`C`, `theta`, `cap`, `s` and `sh`),
  and `-policy` only confirms that they equal yours. With large values, almost any tokens are
  accepted. The project does not recommend any values, and the limits in this repository's examples
  say nothing about thresholds for a real model. Decide before you send any prompt whether you can
  defend the published values for your use and for the longest reply you accept. If you cannot, or
  if the files hold different values, treat the reply as unproved. Read also the final norm's two
  exponents, the parameter file's lines `MF` and `KF`: `cp_verify` prints them in no line. How much
  a clipped-gap policy excludes depends on their difference as well as on the policy's three numbers
  ([why](limits.md#within-limits)).

<a id="which-setting"></a>

## Which setting is it?

The first 32 lines of a parameter file, from `version` to `table`, fix which results an acceptance
rests on and how much proof of work is counted. The rest describes the model and the acceptance
policy, and a provider's tools rewrite some of it for each model, so compare just these 32 lines,
not the whole file. To get their SHA-256, run `head -n 32 CONFIG_DIR/policy.txt | sha256sum` and
ignore the trailing ` -`. The table lists parameter files under `prover/examples/` by setting and
size:

| Setting | Size | File under `prover/examples/` | SHA-256 of its first 32 lines |
|---|---|---|---|
| default | full size | `target_full_f2/policy.txt`, `target_full_f2_gap/policy.txt` | `98ba1ef1a36978d347cb10e89125a8f64297ce68af81e8df1f57255ca08e5484` |
| default | small | `target_small_f2/policy.txt`, `target_small_f2_gap/policy.txt`, `target_small_f2_roomy/policy.txt` | `b21e4564a809f9d0f70a585460cfcb6990d5151b0fe270c1273da5c30a6f0ef5` |
| default | small, another model | `target_small_file_f2/policy.txt` | `3776afcd121fc2a98834052a8f31c448a7fc2d0e18b5217022a6aa24c13a2e36` |

A `YES` under one of these rests on what [docs/limits.md](limits.md#what-the-theorem-assumes) says
of that setting. With any other 32 lines, it rests on that file's own numbers, so read the
`accounting:` and `proof of work:` lines and do not rely on the `YES` by itself. See also [what a
`YES` rests on at the files in the table](../verifier/README.md#what-a-yes-does-not-say).

`cp_verify` prints the same `YES` at every size of model. At a small example it says of that small
committed model what it says at full size of the full-size one, and no more.

No theorem on soundness reaches a call of `cp_verify`. `lean-verify` evaluates, inside Lean, a form of
`verify` that is proved sound, trusting Lean's evaluator and compiler, the helper library and the
script ([tools/lean-verify/README.md](../tools/lean-verify/README.md)). The theorem's scope is one
format and one size: the format's default setting at full size. That the library's written copy of
the full-size public files is the one in this repository is held by no theorem (a program beside the
library, `lean/scripts/check_stored_examples.py`, compares them from outside). What is proved:
[guide/what-is-proved.md](../guide/what-is-proved.md).

What this check cannot tell you:

- The 32 lines also include the sizes of the proof system's codes, so the same setting has a
  different SHA-256 at every size. The list is no general test of a setting. For a real model, the
  full-size row is the one that matters (row SO-6.5 in [the table of all
  rows](limits-all-rows.tsv)).
- The 32 lines say nothing about the model's shape and arithmetic, or about the acceptance policy;
  those come later in the file. `cp_verify` also tests no-overflow conditions on the public
  parameters, which are meant to ensure that an equation that a proof checks modulo the prime also
  holds over the integers. That they suffice for the programs' equations is not proved
  ([limits.md](limits.md#which-verifier-has-a-theorem), row SO-9.3). The figures in bits also depend
  on the model's shape lines, so the same 32 lines with a different shape need not give the figures
  that docs/limits.md quotes.
- The commitment ID already covers the whole parameter file, so this check tells you which account
  applies, not whether the file is genuine.
- The comparison of the first 32 lines covers `policy.txt` alone. The commitment ID covers `KL.bin`
  and `layout.txt` too, but no list here says which forms of them go with which account, and
  `cp_verify` accepts forms of all three files on which the technical report states nothing
  ([limits.md](limits.md#how-the-programs-count)).

<a id="cost-and-privacy"></a>

## Running time and privacy

[guide/performance.md](../guide/performance.md) gives the time and memory a full-size check needs.
Keep the output and both files of a response as private as the request and
the reply: the `covered:` line, for example, shows r and e ([what anyone holding these files
learns](what-is-not-hidden.md)).
