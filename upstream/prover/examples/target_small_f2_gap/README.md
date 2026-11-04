# target_small_f2_gap: a small example with a clipped-gap acceptance policy

The other example configurations (the folders `target_*`) set the exact
match, all but `target_full_f2_gap`: a proof states that every covered token is the committed
integer model's first choice. This one sets a *clipped gap*. It is for a provider who
serves a floating-point model and afterwards proves, in one forward pass of the committed integer
model over prompt and reply, that the reply's tokens lie near that model's choices.

All weights are synthetic (`cp_commit -weights made-up-blocks`). The limits below fit this example
only. They say nothing about a real model's thresholds. What the tests of hiding covered:
[../TARGET-KITS.md](../TARGET-KITS.md#weight-hiding).

## The files

| file | what it is |
|---|---|
| `policy.txt` | `../target_small_f2_roomy/policy.txt` with three other lines: `C 9244130`, `theta 323343`, `cap 1`. All other lines are that file's; one of them, `s 16`, is a round number that the file states: no equation ties it to the model (verifier/docs/parameter-file.md) |
| `KL.bin` | quantization scales under which the integer model does not saturate on these synthetic weights (below) |
| `layout.txt` | `../target_small_f2_roomy/layout.txt`, byte for byte |
| `reply.json` | one prompt of 16 token ids and the reply of 32 ids that the floating-point model gave to it; not among the fitted replies |
| `other_prompt.json` | the same reply after another prompt: a control, which misses the limits |

**Why other scales.** A floating-point model has no clamps and no table ends, so it is close to the
integer model only where the integer model does not saturate. On 8 windows of 64 made-up tokens, with
the floating-point model in float32 under Hugging Face transformers on a CPU, the two models' first
choices agree at 501 of 512 positions under these scales, at 50 under `target_small_f2_roomy`'s and at
23 under `target_small_f2`'s, where 17 % and 43 % of the gate's keys lie past the end of the SiLU
table. The script that counted these five figures is not in this repository;
`integrations/vllm/integer_model/tests/test_float_pair.py` repeats the comparison in part, for these
scales and for `target_small_f2`'s.

## What the policy says

A covered token's *cost* is its logit gap plus 2^sh = 256: the largest logit at its position, minus
its own, plus 1 if a token with the largest logit has a lower id ([docs/model.md](../../../docs/model.md),
section 8). The integer model's own choice costs 256. A proof under this file states that the
covered tokens' costs, each counted as at most `C`, have a mean of at most `theta`, and that at most
`cap` of them exceed `C`. It does not state which model served the reply: see "Most replies of a nearby
model pass", in the last section.

## How the limits were fitted

No GPU is needed. The runs told in this section used these scripts, not in every case as these sources have them. `try_limits.py` took 56 seconds at 16 threads of one machine, six minutes on a busy one and 106
seconds at 16 threads of another. On the CPU of one GPU node, also at 16 threads, it had not ended
after 18 minutes; the cause is not known.

From the tree's top, with the package, transformers and safetensors installed
(`integrations/vllm/integer_model/README.md`, "Install and quick start"):

```sh
D=$(mktemp -d)/pair && echo "D=$D" &&
python3 -m integer_model_vllm.example small_float_pair --directory "$D" --repository "$PWD" &&
integrations/vllm/integer_model/examples/try_limits.py "$D"
```

The first line draws a new folder, below one that only you can write to. The second writes the
integer model of this configuration and the same model as a float32 checkpoint. The third decodes 128
greedy replies of up to 32 tokens from the checkpoint with Hugging Face transformers, in bfloat16 on
the CPU, to prompts of 16 random token ids. It fits the three lines
on the first 64 replies with `integrations/vllm/integer_model/examples/fit_limits.py`
(`C`: the 99.9th percentile of all costs;
`theta`: the largest clipped mean; `cap`: the most costs above `C` in one reply) and counts how many
other replies meet them: the other 64 as they are, after another prompt and with tokens replaced; the
replies of a nearby model (the last section says what that is); and the integer model's own replies to
eight of the prompts. Its standard output on the machine where this file's lines were made:

```
C 9244130
theta 323343
cap 1
held-out replies: 63 of 64 meet the limits
each after the next reply's prompt: 0 of 64 meet the limits
1 of the tokens replaced by a random id: 3 of 64 meet the limits
1 of the tokens replaced by the floating-point model's second choice: 5 of 64 meet the limits
4 of the tokens replaced by a random id: 0 of 63 meet the limits
4 of the tokens replaced by the floating-point model's second choice: 0 of 63 meet the limits
16 of the tokens replaced by a random id: 0 of 61 meet the limits
16 of the tokens replaced by the floating-point model's second choice: 0 of 61 meet the limits
replies of a nearby model, which applies the router's score twice: 47 of 64 meet the limits
the integer model's own greedy replies: 8 of 8 meet the limits
```

That run used PyTorch 2.9.1 and transformers 4.57.6. Of the 2,035 reply tokens that were fitted, 86
are not the integer model's first choice. Other runs with the same versions, with another form of the script, had
85 or 86, the same three lines and the same first eight counts. Six of the 128
replies end early, at the end token, so fewer than 64 have 4 or 16 tokens to replace.

Floating-point results differ between machines and libraries, so another machine can decode other
replies and fit other limits: nothing is broken if yours does. This folder's three lines were fitted
under the versions above, on a CPU without bfloat16 hardware. Under PyTorch 2.13.0 and transformers
5.17.0 a second CPU without such hardware printed the thirteen lines above to the letter. Under
these versions a CPU with such hardware gave `C 6654538`, `theta 282031` and `cap 1`; 64 of 64
held-out replies met them, 0 of 64 after the next reply's prompt, and 46 of 64 of the nearby
model's. A further machine, under transformers 4.57.6, counted 62 of 64 held-out replies and 42 of
64 of the nearby model's (by a report alone). With `--dtype float32` two machines, each under its
own versions, gave the same numbers as each other, with `--replies 64` and, in another pair of
machines, with the default 128 replies. Limits are a provider's own choice, fitted on its own
replies from its own machines.

## Prove and check a reply

This needs one H100 and the programs, built as [docs/quickstart.md](../../../docs/quickstart.md),
"3. GPU: a small model", says. Type the blocks of this section in one shell, from the tree's top:
the later ones use what the earlier ones set, and `${W:?}` stops a line if `W` is not set. `mktemp`
draws its folder below `TMPDIR` if that is set, and `cp_prove` refuses (`cp-isk`) a name for `-isk` that
does not begin with `/`, and a path on which a part is a symbolic link or a directory in which others
than you or root can write (a sticky one, like `/tmp`, passes). So unset a `TMPDIR` that names such a
place.

```sh
T=$(mktemp -d) && W="$T/gap_example" && echo "W=$W" # a new folder, outside the tree, below one that only you can write to
prover/examples/run_target_example.sh target_small_f2_gap "$W"
```

This commits to the synthetic weights under this configuration, and proves and checks one request for
the next token. That request covers no token of its own, so the limits play no part in it. Now make a
request from `reply.json`: the prompt, the reply, token 0 up to the window's 64 tokens, then r = 15 and
e = 47, so that the proof covers the 32 tokens of the reply. (The function is that of
`integrations/vllm/integer_model/README.md`, which says more about the file.)

```sh
request() { python3 -I -c 'import json, struct, sys
one, T = json.load(open(sys.argv[1])), int(sys.argv[2])
p, a = one["prompt_token_ids"], one["reply_token_ids"]
assert p and a and len(p) + len(a) <= T + 1, "prompt and reply need a token each, and together at most T + 1"
words = struct.pack("<%dI" % (T + 2), *(p + a + [0] * T)[:T], len(p) - 1, len(p) + len(a) - 1)
open(sys.argv[3], "xb").write(words)' "$1" "$2" "$3"; }
B=prover/build/hc    # where prover/build.sh has put cp_commit, cp_prove and cp_verify
id=$(sed -n "s/^commitment ID: //p" "${W:?}/cp_commit.log" | tr -d ' ')
mkdir "${W:?}/in2"
request prover/examples/target_small_f2_gap/reply.json 64 "${W:?}/in2/reply.req"
env HOME="${W:?}/home" TMPDIR="${W:?}/tmp" $B/cp_prove -isk "${W:?}/isk" -in "${W:?}/in2" -out "${W:?}/out2"
$B/cp_verify "${W:?}/config" "${W:?}/ivk" "${W:?}/in2/reply.req" "${W:?}/out2/reply.y" "${W:?}/out2/reply.proof" -id "${id:?}" \
  -policy clipped-gap,9244130,323343,1,16,8
```

Four of the reply's 32 tokens are not the integer model's first choice. Their costs are 446,700,
438,154, 56,748 and 470,490, the other 28 cost 256 each, and the mean is about 44,352, below `theta`. So
`cp_prove` should make a proof, and `cp_verify` should accept it: exit status 0 and the last line
`WITHIN LIMITS: clipped-gap C=9244130 theta=323343 cap=1 s=16 sh=8`, which is no `YES`
(verifier/README.md, "Last line and exit status"). The client names the policy itself, with
`-policy`: without the option no call that covers a token under this file is accepted.

The control has the same reply after another prompt. 29 of its 32 costs exceed `C`, so `cp_prove`
should make no proof for it:

```sh
mkdir "${W:?}/in3"
request prover/examples/target_small_f2_gap/other_prompt.json 64 "${W:?}/in3/other.req"
env HOME="${W:?}/home" TMPDIR="${W:?}/tmp" $B/cp_prove -isk "${W:?}/isk" -in "${W:?}/in3" -out "${W:?}/out3"
```

**What has been run.** Every run was at this small size, on the synthetic weights, with a seed that
`cp_commit` drew, with the model, the scales and the three limits of this folder. At this configuration 38 rounds reach the target of 128 bits only because up to 32 bits of proof of work are counted in them; without that work the least round has 96.23 bits (../TARGET-KITS.md, "What the accounting says and does not say"). On the released
programs both audits with the provider's secrets ran under this folder's policy: they show that the
masks are there as derived, which is no search for a leak (docs/limits-all-rows.tsv, row H13.8).
On the released programs the four blocks of this section also ran once, on one H100, with no line
changed: the request for the next token ended `YES`; `cp_verify` ended `reply.json`'s check with
status 0 and the last line `WITHIN LIMITS: clipped-gap C=9244130 theta=323343 cap=1 s=16 sh=8`;
`other_prompt.json` got `other.failed` and no proof; and `score_reply.py` said "meets the policy",
then "fails the policy". The further calls and the three starts told below were not made on the released programs. The
runs in the two bullets below were on [other programs](../../../docs/glossary.md#other-programs).

- **Three further calls of `cp_verify` on a proof that those programs made for `reply.json`** each ended with status 1: without `-policy`,
  `NO: policy`; with `theta` one lower in `-policy`, `NO: policy`; with the other request, `NO:
  proof-refused`.
- **Three starts that `cp_prove` should refuse** were refused (`cp-isk`, status 1, no proof): with
  `-isk` as a relative name, through a symbolic link, and below a directory of mode 777.

The same two verdicts can be had without a GPU, on the folder `$D` that the first block of "How the limits were
fitted" makes, in the same shell (the last line begins "meets the policy", then "fails the policy"):

```sh
integrations/vllm/integer_model/examples/score_reply.py "${D:?}/model" \
  118,506,483,466,241,336,339,174,175,177,162,310,245,130,250,452 \
  46,316,408,491,195,74,74,74,74,155,338,162,305,155,372,408,408,146,418,251,415,408,68,281,146,307,391,304,400,230,36,216
integrations/vllm/integer_model/examples/score_reply.py "${D:?}/model" \
  167,429,128,18,329,250,68,11,246,414,187,129,428,386,238,101 \
  46,316,408,491,195,74,74,74,74,155,338,162,305,155,372,408,408,146,418,251,415,408,68,281,146,307,391,304,400,230,36,216
```

## What the counts show, and what they do not

- The limits admit nearly all honest replies that they were not fitted on, and no reply to another
  prompt.
- They are no exact match: some replies with one changed token pass. A client learns only that the
  costs lie within the limits.
- In the worst case one of 32 covered tokens can be any token at all: with the integer model's first
  choices for the other 31, the mean is (9,244,130 + 31 x 256) / 32 = 289,127, below `theta`.
  `cp_verify` prints this as `above-clip=1`. In the counts above a reply was not decoded again after a
  change, so they are not that worst case.
- A changed token also changes the context in which every later token is scored, so one change can
  raise many costs.
- **Most replies of a nearby model pass.** The nearby model has the same weights. In the four layers
  with experts, where the router's score (between 0 and 1) scales the chosen expert's input, it scales
  that expert's output too, so the expert counts for less. It is no smaller or cheaper model. Decoded
  like the floating-point model, to the same 64 prompts that were not fitted, 60 of its 64 replies
  differ from that model's, and 13 % of its tokens are not the integer model's first choice (3 % for
  the floating-point model). Yet 47 of its 64 replies meet the limits, against 63 of 64; in float32,
  54 against 64. So a reply that meets fitted limits says little about which of two nearby models
  served it; one that misses them says much (17 of 64 against 1 of 64). Most of the looseness is that
  `theta` is the mean of the worst fitted reply, 27 times the fitted replies' median. With `theta` at a
  lower share of the fitted replies (`fit_limits.py --share`) fewer of the nearby model's replies
  pass, and fewer honest ones: by the script's rule, in bfloat16, at share 95: 39 and 60 of 64; at
  90: 18 and 58; at 80: 10 and 56. These replies were scored, not proved; it is one alteration, on
  synthetic weights, with one draw of prompts; two runs gave the same counts. The count of 47 is also
  `try_limits.py`'s own line, above; the scripts that made the other counts are not in this repository. Under
  a looser policy of round numbers all 64 replies of this nearby model passed.
- The prompts are random ids and the model is synthetic. Nothing here measures a real model, another
  engine, another data type or longer replies.
- The three numbers are the provider's own statement. No proof shows how they were fitted.
- The three numbers are public, and they are a function of the weights and of the fitted replies.
  Under a policy other than the exact match each proof or refusal also tells something about the
  costs, so a provider with hidden weights builds the requests itself.
