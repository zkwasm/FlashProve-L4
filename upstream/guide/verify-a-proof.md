# Verify a proof

A provider publishes a commitment ID for a model's weights. `cp_verify`, the C++ verifier that
clients run, checks each proof against it, but never sees what you sent. So exit code 0 means
nothing until [the rule](#when-to-accept) holds.

## What you need first

Get the commitment ID [from a party you trust](../docs/limits.md#who-chooses-the-model), before you
send your prompt, over a channel that the server you check does not control. Never take it from that
server or from files sent with a proof.

The proof file comes from the provider; how you fetch it depends on its service.

Get these before the prompt, not from the server you check:

- The tokenizer and chat template published with the model.
- The end-token ids in its published `generation_config.json`.
- If you accept clipped gap, the weaker of the two acceptance policies, the lines `C`, `theta`,
  `cap`, `s`, `sh` of the published parameter file.

The public files are the three configuration files (`policy.txt`, which is the parameter file,
`KL.bin` and `layout.txt`) and the public key directory: `key.bin` and `leaf/<k>.bin`; the proof file says which
leaf file it needs. [The key file and the leaf files may come from
anyone](../docs/verifying.md#what-you-need-and-where-it-must-come-from), the provider included:
`cp_verify` checks them on every call. Hold the three configuration files as you hold the ID: the ID
covers them by design, but the soundness theorem on `verify` is stated at given public files and does not itself
say that one ID cannot verify under two sets of them. That the key file and the leaf files may come
from anyone rests on an argument in a technical report that is not public and
not refereed, for a protocol from which the programs differ in places. That claim is argued only in a weaker sense, which does not cover the commitment ID. Passing those checks says nothing about the
values themselves: a scale file within the admitted range can switch off model blocks, up to all of
them (worked out by hand; [What is proved](what-is-proved.md)).

## Build the verifier and request

Run the commands on this page from the repository's root. Build `cp_verify` yourself, as in [route 1
of Getting started](getting-started.md#route-1-check-a-stored-proof):

```
verifier/build.sh
```

It ends with `THE VERIFIER IS BUILT:` and the program's path, exit code 0. A long first line that
begins `build.sh: note:` is a warning, not an error.

A [request file](../docs/verifying.md#the-request-and-the-token-file) holds exactly T tokens of
prompt, reply and filler, and a range r:e of positions, counted from 0. T is 2 to the power `lt`,
from `policy.txt`: 1,024 at full size. For a prompt of P tokens and a reply of R, end token
included, use `r=P-1` and `e=P+R-1`.

```
python3 integrations/spool.py make-request T PROMPT_FILE REPLY_FILE END_TOKEN OUTPUT_FILE
```

For a prompt and reply within T tokens, this reads two files of decimal token ids and an end-token
id, or `none` if your length limit stopped the reply. It prints the range, `9:14` for `P=10`, `R=5`.
`OUTPUT_FILE` must not exist yet. Its directory, and every directory above it, must be one that no
other account can change; otherwise the command writes nothing and exits with code 2. A directory
from `mktemp -d` qualifies.

The token file holds `y`, the token after the T tokens: here 0, as four zero bytes that you write
yourself:

```
python3 -I -c 'import sys; open(sys.argv[1], "xb").write(bytes(4))' TOKEN_FILE
```

## Run the check

`cp_verify` limits neither its time nor its memory, and a [hostile
file](../docs/limits.md#hostile-files) can exhaust both. This bash call allows 8 GiB per process and
ten minutes:

```
( ulimit -v 8388608; timeout 600 bin/cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID )
```

`-id` takes all 64 hex digits. Without it, the call ends at best in `NOT-A-VERIFICATION`. For
clipped gap, add `-policy clipped-gap,C,THETA,CAP,S,SH`. A call that hangs or is killed counts as
not accepted.

`integrations/check_proof.py` makes most of the comparisons of [the rule](#when-to-accept) and then
runs `cp_verify` ([integrations/README.md](../integrations/README.md)). It takes the proof with its
token file beside it: the two files share the request's ID as their name, with the endings `.proof`
and `.y`. It sets no limit on time or memory, so put it
inside the same `ulimit` and `timeout`.

## Last line and exit code

- Code 0, `YES` or `WITHIN LIMITS: clipped-gap …`: accepted. Now apply the rule.
- Code 1, `NO: <name>`: rejected, or your files, machine or the program are at fault. An honest
  proof can be rejected.
- Code 14, `NOT-A-VERIFICATION`: no check failed, but the call was invalid. The lines after
  `---- the call ----` say why. Fix it only with values you held before the prompt.
- Code 5, `NOT CHECKED: <name>`: unfinished, neither accepted nor rejected. After `internal-stage`
  or `internal-exception-parFor`, memory may have run out: retry once. Report a repeat, or another
  name, through [SECURITY.md](../SECURITY.md).
- Codes 2, 3, 4 and 15 say nothing about a proof. After any code other than 0, treat the reply as
  unproved.

## When to accept

Accept a reply only if all eight hold.

1. Your verifier: built by you from this release's source, or one you trust as much.
2. Your files: request and token files built by you from the token ids you sent and received.
3. Your ID: you got the commitment ID before you sent the prompt, from [the party that vouches for the model](../docs/verifying.md#what-you-need-and-where-it-must-come-from), over a channel that the server you check does not control, and passed it with `-id`. You hold the three configuration files the same way.
4. The last line of standard output: exactly `YES`, or beginning `WITHIN LIMITS: clipped-gap` if you
   named that policy, with exit code 0. Never search the output for a word.
5. The line `covered:`: `r=`, `e=` and `policy=` are yours, `sh=8` (no theorem or written argument
   covers 7), and `tokens=` lists the reply tokens you wanted.
6. Proof of work: the line `proof of work:` shows what you decided beforehand to accept. For every
   example in this repository, that has to be `up to 32 bits`, or no proof passes. Its bits are not
   a security level.
7. The line `commitment:`: present, with the clause on the record (below).
8. The reply's end, if you need the complete reply: its last token is one of your end-token ids, and
   no earlier checked token is.

Proof of work is a hash search that the provider does before some rounds. Under the parameter file
of every example in this repository, [some rounds meet their error bound only with it
counted](../docs/limits.md#how-the-programs-count), which rests on an assumption. Every example
here counts up to 32 bits. With `-no-proof-of-work`, no proof under them is accepted.

The programs call the out-of-domain (OOD) answers (masked sums of the weights) "the record". You should see the line `commitment:` end with a
clause beginning `; the table has a record:`. Its other words name what the programs' own reckoning
of the commitment rests on ([lean/TERMS.md](../lean/TERMS.md)). The line does not say that anything is missing.

Under a parameter file other than the default setting's, a `YES` also rests on that file's own numbers.

`cp_verify` knows no end tokens, so a reply stopped short still passes. Every ID a call prints is
computed from its files: compare it, never pass it back as `-id`.

## What acceptance means

Under exact match, [a `YES` is meant to show](../docs/limits.md#what-a-yes-covers) that each checked
token is the committed integer model's top choice at its position, given every token before it, up
to the error bound. That is the model computed in integers, not the floating-point one.

The printed error bounds [rest on assumptions](../docs/limits.md#what-the-theorem-assumes), none
proved for the programs. Soundness is proved of the verifier written in Lean, at exact match. Nothing
is proved of
the C++ verifier that the examples and most users run, of the CUDA prover, or of the committer, and
no theorem says that they run the protocol the Lean verifier checks: that rests on tests ([What is
proved, and of what](what-is-proved.md)). [Known limits](known-limits.md) has the bad results on soundness.

`WITHIN LIMITS` [says only](../docs/limits.md#within-limits) that the checked tokens' costs together
meet the limits that the client named, not that the reply is the model's.
Under clipped gap, on a real model, 28 of 32 replies with one token swapped for the model's second
choice met limits fitted to honest replies: scored with the integer-model package, not proved, and not on the released programs.

## What it leaves out

- Whose model it is, or whether it is sensible: the provider alone chooses the weights and settings.
- Anything about your prompt, which `cp_verify` takes as the request file gives it, or about later
  tokens, text, engine or sampling.
- Anything about [weight hiding](weight-hiding.md). No verifier checks whether the provider masked
  the proof or the public files.

See [the rule in full](../docs/verifying.md#when-to-accept), [the limits on
soundness](../docs/limits.md#soundness) and [Known limits](known-limits.md).

## When the commitment ID changes

**Under one commitment ID, each proof is checked against the same first table of weights.** A
commitment is a first table that encodes the weights, and copies of it ("leaf tables"), on which
proofs are made. Each copy's public file, `leaf/<k>.bin`, holds a link proof that it encodes the
same weights as the first table; `cp_verify` checks it on every call. All copies share one ID. (That
is the design: [what is proved of it](what-is-proved.md).)

**Between two IDs there is no proof.** Nothing public shows that two commitment IDs stand for the
same weights: you have only the provider's word. A renewal looks exactly like a switch to another
model. (The exception: public weights committed under a published seed, as in `examples/maverick`:
anyone who has the weight files and makes the calibration set can make that commitment again and compare the ID.)

**If you are given a new ID:**

1. Get it as you got the first: from a party you trust, over a channel that the server you check
   does not control.
2. Compare the new configuration files with the old ones. A provider that only renews can keep all
   three byte for byte; a change shows what it changed. By design, nothing else in the public files
   of two IDs can be compared: each commitment has randomness of its own.
3. Keep some prompts with their replies and the proofs you accepted, and send the same prompts under
   the new ID: that compares the two at those prompts only. At exact match, where two accepted
   replies to the same prompt differ at a token that both reach, the weights or the settings differ,
   unless a false proof was accepted: a theorem bounds that chance for the Lean verifier, not for
   `cp_verify` ([what is proved](what-is-proved.md)). A shorter reply may be the start of a longer
   one at the same weights. Under clipped gap, different replies can pass at the same weights, so a
   difference there says nothing by itself.
4. From a new ID alone, do not conclude that the model has changed; from equal files and replies, do
   not conclude that it has not.

**What is known to force a new ID:** a copy that is used up; a second GPU; any change to a configuration file,
such as another acceptance policy, window length or scales; the provider's own count of batches lost
or in doubt; a move of the service to another machine; a version of the programs that forms the ID
or the files otherwise. `cp_commit` makes all copies in one call and cannot add one later.

**How long an ID lasts.** One copy serves 688,127 batches (of 1 to 64 requests each) at full size
and the default setting, by the programs' own calculation; no run has come near it. A commitment
ends when its first copy is full. By arithmetic from three short sessions, one copy is about a year
and a quarter for one GPU that never pauses, at 1 request per batch; longer with fuller batches. It
is not a lower bound: a batch that fails still counts. `cp_prove` counts and stops, because a copy's
masks cover only so many batches: the limit is there for weight hiding, not for how a proof is
checked. `cp_verify` cannot see how many batches a copy has served, so a proof from a copy past its
limit would be checked like any other.

## The verifier that Lean proves sound

To get the answer of the Lean verifier, `verify`, run [`lean-verify`](../tools/lean-verify/README.md),
which runs its executable form inside Lean. It takes a full-size proof only, not the small stored
examples, and a run takes tens of minutes. The theorem is for exact match, and the Lean verifier does
not test that the policy says so: the lines `C`, `theta`, `cap` of the parameter file must read 256,
256, 0. A run trusts Lean's evaluator, compiler and runtime, the helper library and the C compiler that builds it, and the file that drives the run.
What is proved of it, and what an accept by it means: [What is proved](what-is-proved.md).
