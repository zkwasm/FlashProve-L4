# integrations/

Scripts to check a proof made by FlashProve, and to get such proofs from a vLLM server.

A **provider** commits to a model, publishes the **commitment ID** (64 hex digits that identify one commitment to weights and settings) and runs the server. A **client** sends prompts to the server and checks the proofs of its replies.

**Status:** research artifact: a working example, not hardened for production. [vllm/README.md](vllm/README.md#status) says what has run, not on the released programs, and [LIMITS.md](LIMITS.md) everything that the scripts do not do.

## Words

- **Integer model**: the fixed-point model that is committed and proved.
- **Acceptance policy**: the rule that the checked tokens must meet. Under exact match, the default, each checked token is the model's greedy choice. Under clipped gap the checked tokens' scores must lie, on average, within limits that you accept. That says much less than exact match.
- **Request file**: the token IDs of prompt and reply, and the range of tokens to check.
- **Next token**: the token that follows the last token of the request file. It is not in that file: it comes in a file of its own, and you state it with `--next-token`.
- **Verifier**: `cp_verify`, a C++ program ([what its yes is worth](#what-a-pass-means)).
- **Public key directory**: what the provider publishes for one commitment: `key.bin` and the leaf files `leaf/K.bin` ([what these are](vllm/REFERENCE.md#fetch-the-proof)).
- **Proving service**: `cp_prove`, the provider's program that makes the proofs. A **front end**, here the middleware, passes it requests through a **spool directory**.

## What is here

- `check_proof.py` checks one proof: it checks that the proof's files match the values that you give it, then runs the verifier. It needs only the standard library. Read this first.
- `ask_and_check.py`, the example client, asks a server for a reply, fetches the proof and runs `check_proof.py`.
- `proof_middleware.py` runs inside `vllm serve`. It turns served replies into proof requests, has `cp_verify` check every proof of a batch (narrow exceptions, none open to a client: [LIMITS.md](LIMITS.md#narrow-exceptions)) before any response of it leaves, and serves the proofs. Its commands `init`, `check`, `clean`, `hold` and `lift-hold` are for the provider. It is an example: [what it leaves undone](LIMITS.md#what-the-integration-reveals).
- `spool.py` is the library through which a front end and the proving service exchange files. Its command `make-request` writes a client's request file; `sweep` deletes responses older than a given number of days.
- `vllm/end_to_end/end_to_end.py`, the demo, runs all of these with vLLM and `cp_prove` on one machine. `vllm/integer_model/` lets vLLM run the integer model.
- `tests/` has test files, most of them named after the program that they test, and `test_round_trip.py` for all together. `dummy_*.py` and `fake_cp_verify.py` are stand-ins; `recorded_vllm/` contains response bodies that vLLM's own routes produced around a fake engine: every value in them is a dummy.
- Documents: this file (check a proof), [`LIMITS.md`](LIMITS.md), [`vllm/README.md`](vllm/README.md) (guide and demo) and [`vllm/REFERENCE.md`](vllm/REFERENCE.md).

`pip install ./integrations`, in a virtual environment, installs them as `flashprove-check-proof`, `flashprove-ask-and-check`, `flashprove-middleware-admin` and `flashprove-spool`. The installed commands do not know where the verifier is: pass `--verifier PATH`. For `pip install ./integrations`, pip fetches the package's build tool (`setuptools`, 77 or later), of no fixed version and without a hash, and runs it under the account that types the command (pip's documented way).

## Check a proof

You need Linux, g++ 13 or later and Python 3.10 or later (3.13 for `vllm/`). Run every command from the repository root.

**Do not run any of this under `sudo` or as root.** The programs, scripts and tests need no root rights, and as root a fault or a mistake reaches further ([what can go wrong](LIMITS.md#under-sudo)). Two exceptions: the tests marked `root_only`, which are skipped without root ([Tests](#tests)), and two steps of an operator's set-up, which start none of these scripts ([vllm/REFERENCE.md, Part 2](vllm/REFERENCE.md#part-2-operators)).

### The shipped example

No network and no GPU are needed:

```sh
verifier/build.sh
python3 integrations/check_proof.py examples/accepted-proof/example.proof examples/accepted-proof/example.req \
  --public-key-dir examples/accepted-proof/ivk \
  --parameter-file examples/accepted-proof/config/policy.txt --scales-file examples/accepted-proof/config/KL.bin \
  --commitment-id 79a5a2bc7a469b40df73dedf40abeb740671e8ae8077738bd3743897aeb22183 \
  --range 60:64 --next-token 199 --end-token none
```

You should see exit code 0 and a last line that begins `check_proof: verified: exact match:`. The script needs the token file `example.y` beside `example.proof`, and compares it with `--next-token`. Before that, `cp_verify` prints diagnostics that can look like failures. Only the last two lines, `result:` and `check_proof:`, say how the check ended. The script leaves out the verifier's detailed notes on what its figures rest on; `--show-all` shows them.

This pass shows only that the files fit together. The example comes with its commitment ID and request, and its model is synthetic, so the pass says nothing about any model that anyone serves. [examples/README.md](../examples/README.md) lists the other examples.

### The full-size example

The same check on a proof for a model of Llama 4 Maverick's shape, with synthetic weights. Its request holds a prompt of 24 tokens and then 1,000 tokens that the integer model chose itself, each its greedy choice after the tokens before it:

```sh
python3 integrations/check_proof.py examples/full-size/example.proof examples/full-size/example.req \
  --public-key-dir examples/full-size/ivk \
  --parameter-file examples/full-size/config/policy.txt --scales-file examples/full-size/config/KL.bin \
  --commitment-id 7a285919b43961861d9839530c26725accf26aa6d74e3c2afa31b9dc6bcd18fe \
  --range 23:1024 --next-token 52599 --end-token none
```

You should see exit code 0 and a last line that begins `check_proof: verified: exact match:`. The request has 1,024 tokens, so `23:1024` checks tokens 24 to 1023 and the next token: 1,001 tokens in one proof ([positions](#a-reply-of-your-own)). Like the small example's, this pass shows only that the files fit together. What such a check asks of a machine: [the example's own page](../examples/full-size/README.md#try-it).

### A reply of your own

A check is worth only what you bring to it. Five things must never come from the server that you are checking:

1. the commitment ID, which you got before you sent the request, from a source that does not depend on that server; the place where the commitment is published is one;
2. under clipped gap, the limits that you accept;
3. the token IDs of prompt and reply. Build the request file yourself from the tokens that you sent and received: the proof is about the checked tokens given the earlier ones, and it does not check where those came from;
4. the model's end-token IDs, from its published `generation_config.json`;
5. the verifier, built by you.

Never take a commitment ID or a limit from a line that the verifier or a script prints. The key file and the leaf files may come from anyone, the provider included: `cp_verify` checks them on every call. Hold the three configuration files as you hold the ID: the ID covers them by design, but the soundness theorem on `verify` is stated at given public files and does not itself say that one ID cannot verify under two sets of them. [REFERENCE.md](vllm/REFERENCE.md#what-you-must-bring-yourself) says what else is not proved about them. The commands: [with the example client](vllm/REFERENCE.md#check-the-proof), or [without it](vllm/REFERENCE.md#without-the-example-client).

`--end-token none` is for the examples, whose tokens are arbitrary and do not end with an end token, or for a reply that your own length limit stopped. With `none` the reply's end is not checked, and a reply that was cut short can end `verified`. If you have no end-token IDs, stop and ask.

**Positions.** `--range START:END` is not a Python slice. Counting from 0, it checks the tokens at positions START + 1 to END, each given the tokens before it. If END is the number of tokens in the request file, the last of these is the next token, which the script's `checked_tokens=` and the other pages do not count among the [checked tokens](../docs/glossary.md#checked-token). So `60:64` on 64 tokens checks tokens 61 to 63 and the next token. `--reply PROMPT_LEN:REPLY_LEN` gives the same range by lengths: `--reply 61:4` is `--range 60:64`.

### What a pass means

A pass means that the verifier accepted a proof of this statement: under the integer model that was committed under your commitment ID, the checked tokens, each given the tokens before it, meet the acceptance policy.

- A pass does not tell you which model was committed, how closely it follows the original, or anything outside the checked range.
- Under clipped gap the last line says `within limits`, not `verified`. That bounds the mean of clipped costs, and the number of checked tokens that cost more than `C`. The last line gives that bound: up to so many of the checked tokens can be any token at all. Each of the others costs at most `C`, which rules out no token if `C` is large or where the model's scores lie close together, and no verifier sees that. And it does not show that the reply is the model's own.
- No theorem covers `cp_verify` or the Python scripts in this folder ([LIMITS.md](LIMITS.md#what-a-pass-does-not-show-and-what-a-client-must-bring)).

[docs/verifying.md](../docs/verifying.md) has the whole rule of when to accept.

### What the script adds to the verifier

- It passes your commitment ID to the verifier, and rejects configuration files, or a `key.bin`, that give another commitment ID ([what the ID covers in `key.bin`](vllm/REFERENCE.md#what-you-must-bring-yourself)).
- It rejects a checked range, next token or policy that is not yours.
- It requires the checked tokens to end at their first end token.
- It gives the verifier private copies of the configuration files, the request and the proof ([details](vllm/REFERENCE.md#check-the-proof)).
- Run from a checkout, it starts itself again in Python's isolated mode, and so does the example client. From then on `PYTHONPATH`, your user site-packages and the current directory do not decide which modules they import ([what that leaves open](LIMITS.md#smaller-limits)).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | passed; `claim=` says which claim `cp_verify` checked: `exact-match`, `policy` (your clipped-gap limits), or `next-token` (the next token alone was checked); every pass is under a commitment with published out-of-domain (OOD) answers (masked sums of the weights; the programs call them the record), and, where its parameter file counts it, with counted proof of work (see below) |
| 1 | rejected, files not as stated, or no proof made |
| 2 | error, no verdict: usage, environment, missing file, Ctrl-C |
| 14 | not a verification: for example a parameter file outside the verifier's standard form, a test build of the verifier, or counted proof of work under `--no-proof-of-work` (see below); never a pass |
| 24 | `ask_and_check.py` only: passed, but the server chose the prompt token IDs |
| 75 | `ask_and_check.py` only: try again (proof not ready or unchecked, server unreachable) |

Before its last line `check_proof.py` prints a `result:` line. At a pass it has all fields; for the shipped example: `result: outcome=verified commitment_id=... range=60:64 ... claim=exact-match ...`. Otherwise it is only `result: outcome=WORD`, for example `result: outcome=rejected`.

A program should take a check as a pass only if all three conditions hold:

1. The exit code is 0.
2. On standard output only, the line directly before the last begins `result: `, and its first field is `outcome=verified` or `outcome=within-limits`. An exit code 0 without such a line, as after `--help`, is an error, not a pass.
3. The fields of that line are what the program meant to ask:
   - `commitment_id=` and `range=` have its own values. The script has checked them against its command line; checking again catches a wrapper that passed the wrong variable or read another call's output.
   - `claim=` has a value that the program accepts; the strongest is `exact-match`.
   - `end_token=` is one of its end-token IDs, or `not_checked` only if its own length limit stopped the reply.
   - `work=` has a value that it accepts (only 0 under `--no-proof-of-work`; see below).
   - `commitment_assumed_rounds=` has a value that it accepts (how many [closeness rounds](../docs/glossary.md#closeness-round) of the commitment's [one-time files](../docs/glossary.md#one-time-file) rest on another source than [the bound within a third of the distance](../docs/glossary.md#proved-lemma); never 0 at a pass; see below).
   - `record=1` is there, as at every pass. The field stands for what the verifier's sentence on the commitment says: the binding of the weights rests on the theorem on codes, taken as given (see below).

**Proof of work.** A commitment's parameter file can count [proof of work](../docs/glossary.md#proof-of-work) toward its target, in bits: hashing work that the proof shows was spent, and that a cheater must spend again on every try. The default settings count it.

At every pass you see both figures, with the work and without it. The `result:` line has the work's bits as `work=` (0 if none), then `least_round=` and `least_round_without_work=`. Where `work=0`, read `least_round_without_work=`. The last line ends in one of two ways:

- where work is counted: `proof of work: up to <n> bits counted in <m> rounds; with it the least round has <a> bits, without it <b> bits (cp_verify's own figures, not a security level; --show-all has its caveats)` (for one round: `1 round`)
- where none is counted: `proof of work: none counted; the least round has <b> bits (cp_verify's own figure, not a security level; --show-all has its caveats)`

`<a>` and `<b>` are the two figures. Each is a single round's figure, and neither is a security level. The script computes nothing here: it repeats what the verifier prints. Decide beforehand whether you count such work; if you do not, pass `--no-proof-of-work`:

- **Without the option** the script reports and does not decide: a pass with counted proof of work ends with exit code 0.
- **With the option** a pass with counted proof of work ends with exit code 14. So under a commitment at the default settings no proof passes: a pass needs a commitment under a parameter file that counts none.

**What the commitment rests on.** Just before the words on proof of work, the last line repeats the verifier's line `commitment:` whole: what it says of the commitment's [one-time files](../docs/glossary.md#one-time-file). Every commitment has published [OOD answers](../docs/glossary.md#record) (the parameter file's line `record 1`), and the verifier's line ends `; the table has a record: ...`. The words after `record:` say what the binding of the weights rests on: a theorem from the literature, which the programs' accounting takes as given. This project has verified no proof of it in its paper's own form; Lean's library proves a Lean statement written to follow it; that the two say the same has been checked by reading alone ([lean/THEOREM.md](../lean/THEOREM.md), "What it rests on"; the terms: [lean/TERMS.md](../lean/TERMS.md)) ([what that means for a yes](../docs/limits.md#what-the-theorem-assumes); in full: rows LIM-14.1 to LIM-14.4 of [the table of all rows](../docs/limits-all-rows.tsv)).

At the default setting the same theorem, used with no bits to spare, carries the programs' bound for every accepted request too, not the binding alone; and both lean on counted proof of work (above; docs/limits.md says what each is worth). Under a provider's own parameter file the accounting lines say which rounds rest on it (`--show-all`). The provider's files choose that, not you.

Every pass (exit code 0) has the verifier's words on that theorem in its last line and `record=1` on its `result:` line: at every pass the binding of the weights, in the programs' own accounting, rests on that theorem, taken as given.

If you have decided beforehand not to rely on that theorem, accept no pass: no option of the script or of the verifier changes what the binding of the weights rests on. The line `commitment:` is about the commitment's own files only: it does not say what a request's own rounds rest on (`--show-all` has the verifier's account of that).

The verifier's line has one form: `commitment: <n> closeness rounds of the one-time files rest on another source than the bound within a third of the distance; the table has a record: ...` (for one round: `1 closeness round of the one-time files rests`). A verifier that passes with any other line `commitment:` is not this repository's: the script exits with code 2. What else a yes rests on: [docs/limits.md](../docs/limits.md#the-soundness-figure). What to decide beforehand: [when to accept](../docs/verifying.md#when-to-accept).

`ask_and_check.py` applies part of these rules itself, and has outcomes of its own: [REFERENCE.md](vllm/REFERENCE.md#check-the-proof).

## Tests

```sh
python3 -m venv .venv-tests && . .venv-tests/bin/activate &&
  python -m pip install pytest &&
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider integrations/tests
```

On Ubuntu the first line needs `python3-venv`. It leaves `.venv-tests/` in the clone, which holds nothing of yours and can be deleted. Without `bin/cp_verify` (`verifier/build.sh`) and the examples under `examples/` the tests run on stand-ins only, and only `check_proof.py` ever meets a real verifier in them. Where pytest's temporary directory lies below a place that another account can change ([which](LIMITS.md#the-look-above)), all these tests are skipped: the run ends with exit code 77 and a last line that says why, and what to set. Where that directory's path has a character other than letters, digits and / . _ -, those of the demo's tests that need a work directory there are skipped; where a directory on the way to it is missing or cannot be examined, no test is run (exit code 4, with a line that says what to set).

The tests marked `root_only` want a true root; under any other account they are skipped. They are an exception to the note under [Check a proof](#check-a-proof), not to run anything here as root: they are for whoever wishes to try what the scripts do as root. Most need root to give files and folders to another owner (a uid number: no account is made); one opens a socket on 127.0.0.1 as another user; the others test a check that is made only for root. All that they make, change or remove lies below pytest's temporary directory. To run them, from the repository's root, with the environment made above: `sudo .venv-tests/bin/python -B -m pytest -q -p no:cacheprovider -m root_only integrations/tests`. Leave pytest's temporary directory where that line puts it (`/tmp/pytest-of-root`, which stays behind, as root's), below root's folders: with one below a folder of your own making (`--basetemp`, `PYTEST_DEBUG_TEMPROOT`, `TMPDIR`, `TEMP` or `TMP`), nine of them are skipped, with a reason that says so: those nine have then not run.

No test here makes a real proof of a reply that came through the middleware, or runs the middleware inside vLLM's own server. The demo is designed to do both, on a GPU.
