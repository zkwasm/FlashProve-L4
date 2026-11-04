# FlashProve for vLLM

`integrations/proof_middleware.py`, the example front end, is an ASGI middleware for an unmodified vLLM server: it turns served replies into proof requests, has `cp_verify` check every proof of a batch (narrow exceptions, none open to a client: [LIMITS.md](../LIMITS.md#narrow-exceptions)) before any response of it leaves, and serves the proofs. The client fetches the proof and checks it against the commitment ID, 64 hex digits that identify one commitment to the weights. The weights themselves are not sent.

```text
        1 request                      2 request file
client -------------------> vLLM +     -------------> spool     <--> proving service
       <------------------- middleware <------------- directory      (cp_prove)
        3 reply + proof ID             4 response
        5 GET /plugins/proof/jobs/ID

6 middleware: the verifier (cp_verify) on every proof of the batch, before it is on offer
7 client: check_proof.py + the verifier (cp_verify)
```

The spool directory is where the middleware and the proving service exchange files.

<a id="status"></a>
**Status:** research artifact: a working example, not hardened for production. The runs and figures in this paragraph were not on the released programs. At full size, on a public 4-bit conversion of Llama 4 Maverick, vLLM served the integer model, and in each run all 16 replies got a proof that passed its check. At full size without vLLM, 178 requests made from replies that had been served beforehand were put to `cp_prove` under a clipped gap: it made a proof for exactly the 152 that the integer model package's scores had forecast, among them 127 of 128 greedy replies that stock vLLM had served from Maverick's 16-bit checkpoint, by that job's own output. Both ways of serving ran end to end through a live vLLM at toy size, on synthetic weights. Among the limits: at full size, ordinary serving has never run as one service, since its replies were stored first and proved afterwards; serving the integer model is slow (0.60 to 0.62 s for each token after a 3-token prompt, one sequence at a time, on one GPU); and prompts go in as token IDs, since no real tokenizer has been tested. [LIMITS.md](../LIMITS.md#what-has-never-been-run) tells each run, and [LIMITS.md](../LIMITS.md) this integration's limits for clients and for providers.

## What gets a proof

A request gets a proof only if:

- it is a `POST` to `/v1/completions` or `/v1/chat/completions` with the header `x-proof-policy: NAME`;
- it sets `"temperature": 0` and `"return_token_ids": true`;
- its only other fields are `model`, `prompt`, `messages`, `max_tokens`, `max_completion_tokens` and `stream`.

A request with any other field, such as `stop`, is served as usual, without an error. Its proof ID leads to a job in the state `not_provable`, never to a proof. Other traffic reaches vLLM untouched and gets no proof.

The reply's `x-proof-id` header is the proof ID, with which the client [fetches the proof](REFERENCE.md#fetch-the-proof).

Send the prompt as token IDs. If you send text or chat messages, the server chooses the token IDs and nothing checks how, so `integrations/ask_and_check.py` exits with 24 at best, never 0.

## Two ways to serve

| | vLLM runs | The proof shows |
|---|---|---|
| **Ordinary serving** | the ordinary floating-point model; the proving service runs prompt and reply through the committed integer model in one prefill pass | against that model, the reply's tokens score on average within limits that the client accepts (`clipped-gap`, after [Token-DiFR](https://arxiv.org/abs/2511.20621)) |
| **Integer serving** (stricter, slower) | the committed integer (fixed-point) model | each reply token is that model's greedy choice given the tokens before it (`exact`) |

Both are for greedy replies (temperature 0): the middleware asks for a proof only for a request at temperature 0. A proof itself says nothing about sampling: the prover gets token IDs and a range, with no seed and no temperature ([what a `YES` covers](../../docs/limits.md#what-a-yes-covers)). The integer model package's seeded score of a sampled reply is a score, not a verification: no program of this repository proves or verifies it, and no Lean theorem or security claim covers it ([OPERATING.md](integer_model/OPERATING.md)). For replies that were served another way, see [stored replies](REFERENCE.md#prove-replies-that-were-served-earlier). Before you choose, as a provider with secret weights, read what each policy gives away: [which policy](../../docs/limits.md#which-policy).

## Quick start

The demo commits to a small synthetic model, serves eight prompts, then proves all eight replies in one batch and checks each. A changed reply must get no proof and be rejected. The demo's weights follow from a public rule, so there is nothing secret in them to hide, and a success shows only that the parts fit together.

You need x86-64 Linux on a CPU with the SHA extensions, one H100 that nothing else uses, CUDA 13.2 Update 2 (a later toolkit is not supported), gcc and g++ 13, and Python 3.13 ([the prover's whole list](../../prover/docs/build.md#what-you-need)). Published security advisories cover some of the pinned packages, vLLM 0.29.0 among them: before you expose a server, read [the checklist](REFERENCE.md#the-operators-checklist).

**Do not run any of this under `sudo` or as root.** The programs, scripts and tests need no root rights (but for the tests marked `root_only`), and as root a fault or a mistake reaches further ([what can go wrong](../LIMITS.md#under-sudo)). An operator's set-up has two steps as root, which make directories, copy in three configuration files and write a config file, and start none of these scripts ([Part 2](REFERENCE.md#part-2-operators)).

From the repository root. The first line clears variables that the build scripts would otherwise take from your environment. For the last `pip` line, pip fetches a build tool of no fixed version, without a hash ([README.md](../README.md#what-is-here)). The last two lines run the demo twice, first with ordinary serving, then with integer serving, each in a new directory below a private one:

```sh
unset OUT BUILD GPU_RUN
verifier/build.sh
prover/build.sh
python3.13 -m venv .venv && . .venv/bin/activate &&
  python -m pip install --require-hashes -r integrations/vllm/requirements.txt &&
  python -m pip install --no-deps --no-build-isolation integrations/vllm/integer_model &&
  python -m pip install ./integrations
W="$(mktemp -d)" && echo "$W"
python3 integrations/vllm/end_to_end/end_to_end.py "${W:?first run the line with mktemp}/float" --serve float
python3 integrations/vllm/end_to_end/end_to_end.py "${W:?first run the line with mktemp}/integer" --serve integer
```

| Exit code | Last line |
|---|---|
| 0 | `RESULT: every step gave the expected result` |
| 0 | `RESULT: every step that ran gave the expected result; not checked: ...` |
| 1 | `RESULT: failed`; a line above it shows which step |
| 2 | none: usage or environment error |
| 130 | none: stopped by Ctrl-C (143 after SIGTERM, 129 after SIGHUP) |

**The work directory.** Give each run a new path. The script stops if the path exists; otherwise it makes the directory with mode 0700 and writes the logs to its `logs/`. The full path must:

- be named by a path with no symbolic link in it;
- contain only letters, digits and `/ . _ -`, and at most 380 characters;
- be a new name in a directory that exists;
- lie below directories that only you or root can change, as `mktemp -d` gives you. A sticky one such as `/tmp` is fine ([Shared machines](../../docs/operators-duties.md#shared-machines)). If you run it under `sudo` anyway, the account that ran `sudo` counts as you ([what that costs](../LIMITS.md#under-sudo)).

Two cases count as one that others can change: a directory that its group can write, also your own with mode 0775 under umask 002; and a `/tmp` without the sticky bit, such as a Kubernetes `emptyDir`. In a rootless container whose `/` or `/tmp` shows an unmapped owner, the script refuses every work directory below it. Where a rule is broken, the script stops with exit code 2, before it creates anything.

The script examines its work directory only. vLLM itself runs with your `HOME` and `TMPDIR`, and with any cache variable that you have set (such as `XDG_CACHE_HOME`). The script does not examine those places, and vLLM and its libraries may write caches there that they load back (not examined). Only you or root should be able to write in them, or in a directory above them ([what counts](../LIMITS.md#the-look-above)). With `TMPDIR` unset they usually use `/tmp`: its sticky bit keeps other accounts from removing or renaming your files there, not from making a directory beforehand under a name that a library expects, and whether these libraries guard against that has not been examined. On a machine that you share, set `TMPDIR` to a directory that only you can write in.

Only the demo's synthetic model has its secret key directory inside the work directory; below `/tmp` that is harmless only for that model ([for another: never in a file system in memory, which `/tmp` often is](../../docs/operators-duties.md#keys-seeds-and-machines)). A commitment of your own keeps its secret key directory where `cp_commit` made it, on a local disk: the demo neither makes, moves nor copies it.

**The second row of the table above.** With these builds a run in which every step passes ends in it (`not checked: ...`): `cp_prove`'s default build prints no cause when it makes no proof, so the demo cannot confirm why the changed reply got none. To have that checked too, build the programs with the test options (`prover/build.sh tests` puts them into `prover/build/hc/test_build`; the ordinary ones are in `prover/build/hc`) and pass `--build-dir prover/build/hc/test_build`. A run that gets through all its steps then prints, above its result, a note that this is not the build that is meant for serving. Keep that build to synthetic weights: never start it on a real commitment's secret key directory.

**Options.** Without `--serve` the script uses `--serve float`, or `--serve integer` if the installed integer model package cannot pair the two models. `--help` lists the other options: those for your own commitment, `--stop-vllm-to-prove` (one GPU for a full-size model), `--vllm-gpu` and `--prover-gpu` (a GPU for each side: `cp_prove` does not start on one with more than 2 GiB in use), `--proof-timeout` (the time limit in seconds for each run of a program other than vLLM; default 3600) and `--cp-prove-options` (extra options for `cp_prove`). Before you run the demo on a commitment of your own, read [what that costs](REFERENCE.md#the-demo-on-your-own-commitment).

**Stopping.** Ctrl-C, a time-out or a failed step stops the programs that the script started, with signals. vLLM and the prover's programs then get up to 15 minutes to exit: on the signal `cp_commit` removes the files that it has made, nearly all overwritten with zeros first ([what that is worth, and which are not](../../prover/docs/ends-in-short.md)), and `cp_prove` wipes its secrets once the GPU work that is running has ended. Whether 15 minutes are enough at full size has not been measured. Ctrl-C during that wait kills them at once (after a time-out or a failed step it takes two), and then they wipe and remove nothing. A process that one of these programs started and that outlives it gets a signal only if `/proc` shows that process, or if `/proc` cannot be listed at all; a process that `/proc` hides can be left running. The script itself deletes nothing.

## Your vLLM 0.29.0 server

[REFERENCE.md, Part 2](REFERENCE.md#part-2-operators) has the eight steps with their commands. In short:

- build the programs and, as the account that will run vLLM and `cp_prove`, make a commitment to the integer model (its parameter file sets the policy);
- create the directories and write the middleware's config file;
- run `proof_middleware.py init`, then start the proving service, `cp_prove`;
- start vLLM with `--middleware proof_middleware.ProofMiddleware`;
- check one proof as a client, and publish what clients need.

The demo caps vLLM's memory so that its small model and the prover share one GPU. A server that proves while it serves a full-size model needs one more GPU than vLLM uses, only for `cp_prove`. It also needs host memory for both at once ([how much](REFERENCE.md#host-memory)); at full size the two have not been run side by side through this integration.

The example front end does not meet [duty 11](../../docs/limits.md#the-conditions) (what a front end must do). Before you expose a server, read [LIMITS.md](../LIMITS.md), and in [REFERENCE.md, Part 3](REFERENCE.md#part-3-security-model-and-limits) the security model (who trusts what) and the operator's checklist.

## Checking a proof, and tests

[integrations/README.md](../README.md#check-a-proof) shows how to check a proof, what a pass means and how a program reads the result; [its last section](../README.md#tests) runs the tests.
