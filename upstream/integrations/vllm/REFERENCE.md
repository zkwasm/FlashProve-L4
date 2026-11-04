# vLLM integration reference

This document is the reference for `proof_middleware.py`, the example front end: an ASGI middleware for an unmodified vLLM server, which turns served replies into proof requests and serves the proofs. Part 1 is for clients: the routes, the example client `ask_and_check.py` and the check script. Part 2 is for operators: how to set a server up. Part 3 has the security model and the operator's checklist. Part 4 has two other routes: [proving replies that were served earlier](#prove-replies-that-were-served-earlier), without the middleware, and [the demo on a commitment of your own](#the-demo-on-your-own-commitment).

**Status:** research artifact: [README.md](README.md#status) says what has run. The limits are in [LIMITS.md](../LIMITS.md). Where a sentence here says that something was read, it comes from source code, not from a run.

## Two ways of serving

[README.md](README.md#two-ways-to-serve) describes both. In **ordinary serving** the policy is clipped gap, a score that follows Token-DiFR (arXiv:2511.20621): at a pass the `result:` line has `claim=policy`, and the last line begins `check_proof: within limits:`. In **integer serving** the policy is exact match: at a pass the `result:` line has `claim=exact-match`, and the last line begins `check_proof: verified:`. What a program may take for a pass: [integrations/README.md, Exit codes](../README.md#exit-codes).

# Part 1: Clients

## Ask for a proof

Add the header `x-proof-policy` and two body fields to a `POST /v1/completions` or `POST /v1/chat/completions`:

```sh
curl -sS -D - "http://127.0.0.1:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -H "x-proof-policy: POLICY" \
  -d '{"model": "MODEL_NAME", "prompt": [1, 2, 3], "max_tokens": 32, "temperature": 0, "return_token_ids": true}'
```

You should see vLLM's normal `200` reply with one more header, `x-proof-id`: 64 hex digits. Keep it and the reply's token IDs, `choices[0].token_ids`. The reply also repeats the prompt's IDs, as `prompt_token_ids` (in `choices[0]` for completions, at the top level for chat): compare them with the IDs that you sent, and build the request file from your own. A server that counts its reply's first tokens as prompt narrows the checked range.

Anyone who has the proof ID can fetch or delete the proof; where the provider has given vLLM an API key, that key is needed too, but for the cases in [the operator's checklist](#the-operators-checklist). The server keeps the proof until a `DELETE`, the provider's clean-up, or the end of its policy (the policy's name is removed or gets a new commitment).

### Which requests can get a proof

Many vLLM fields let the client force tokens, and if forced tokens could get a proof, whether they did would reveal something about the weights ([which policy](../../docs/limits.md#which-policy)). So the body may contain only these, each once:

- `temperature`: required, must be `0`
- `return_token_ids`: required, must be `true`
- `prompt`: one string or one list of token IDs
- `messages`: objects with exactly `role` and `content`, both strings
- `model`: optional; must be the policy's `served_model`
- `max_tokens`, `max_completion_tokens`, `stream`: optional

The reply must also:

- be a single, non-empty choice with a finish reason;
- fit, with the prompt, in the policy's `max_total_tokens`;
- have a body of at most 16 MiB;
- arrive while neither the proving queue (`max_waiting`) nor the store of responses (`max_stored`) is full, and while the policy is not [on hold](#start-the-proving-service).

A request that breaks one of these rules **is still served**. If vLLM answers `200`, the reply carries a proof ID, and the job under that ID ([Fetch the proof](#fetch-the-proof)) has the state `not_provable`, with the reason in `detail`.

An unknown or closed policy name, or the header on another route or method, gets a `400` before vLLM runs, and a body larger than 1 MiB a `413`. Where the server has an API key, a call with the header but without the right key gets a `401` before anything else is looked at.

## Fetch the proof

If the provider has given vLLM an API key, each of these routes needs the header `Authorization: Bearer KEY` ([the operator's checklist](#the-operators-checklist)). All clients share the key or keys, so between clients the proof ID is the only guard.

The example client sends your key (`VLLM_API_KEY` in its environment) on every call, but only over `https://` or to this machine, named by its loopback address, as in `http://127.0.0.1:8000`. The name `localhost` does not count: it can resolve to an address that another local account holds. The loopback address is no proof of who listens either: the client does not look at that, so another local account that takes the port while vLLM is down gets the key with the client's first call. For any other `http://` server the client sends nothing and exits with 2. Without a key it applies no such rule ([LIMITS.md](../LIMITS.md#smaller-limits)).

Under `/plugins/proof/`:

- `GET policies`: each policy's `served_model`, `commitment_id`, `max_total_tokens`, `kind` (`exact` or `clipped-gap`), `limits` and `closed` (200). For comparison only: never take your commitment ID or your limits from here ([why](#what-you-must-bring-yourself))
- `GET jobs/PROOF_ID`: the job as JSON: `state`, `policy`, `request` (`total_tokens`, `prompt_tokens`, `reply_tokens`, `padding_token`); once proved, `files` (200, 404, 500, 503)
- `GET jobs/PROOF_ID/proof.bin`, `.../y.bin`: the proof file; the next-token file (200, 404, 500, 503)
- `DELETE jobs/PROOF_ID`: the job's last state; deletes the job and the server's copy of the response (200, 404, 409, 500, 503)
- `GET published/POLICY/key.bin`: `key.bin` from the public key directory (200, 404, 500)
- `GET published/POLICY/layout.txt`: the layout file, from next to the policy's parameter file (200, 404, 500)
- `GET published/POLICY/leaf/K.bin`: the leaf file of leaf table K (a further weight table for the same weights, with its own key and masks; not a leaf of a Merkle tree); the proof file contains K (200, 404, 500)

- **401** (any route): the server has an API key, and the call did not send the right one.
- **404**: unknown proof ID, file or route. A reply that was cut off, or whose job could not be saved, has no job.
- **409**: the job is still `waiting`, so it was not deleted.
- **500**: a fault that the provider must fix. Keep the proof ID and ask again later.
- **503**: the provider has put this policy's proofs on hold; ask again later.
- A `proof.bin` that ends short of its `Content-Length` failed the server's check on the way: discard it.

States:

- `waiting`: queued, being proved, proved but not yet final (the service still has the request in its `work/`), or not yet checked by the server; poll again, with back-off
- `proved`: `files` lists the two files and their sizes
- `no_proof`: the tokens do not meet the policy, the proving service failed, or the batch was withheld; the server does not report which
- `not_provable`: the request was never sent to the proving service; `detail` has the reason
- `expired`: the server no longer has this proof; send the prompt again

Fetch the proof as soon as the job reads `proved`: after a loss of power on the provider's machine such a job can read `waiting` again and end as `no_proof`. This happens because of the disk; it says nothing against the proof.

A job can stay `waiting` until `clean` deletes it. This happens if its response was deleted before the first poll, if a response without a proof was deleted, or if the service has stopped for good. Stop polling after a time of your choice.

## Check the proof

`ask_and_check.py` does all of the above and then runs the check script, [`integrations/check_proof.py`](../check_proof.py), which calls the verifier, `cp_verify`. From the repository root:

```sh
verifier/build.sh
OUT="$(mktemp -d)" && echo "$OUT"
python3 integrations/ask_and_check.py \
  --server "http://127.0.0.1:8000" --model "MODEL_NAME" --policy-name "POLICY" \
  --commitment-id "COMMITMENT_ID" \
  --public-key-dir "PUBLIC_KEY_DIR" \
  --parameter-file "PARAMETER_FILE" --scales-file "SCALES_FILE" \
  --end-token "END_TOKEN_IDS" \
  --prompt-ids "PROMPT_TOKEN_IDS" \
  --out-dir "${OUT:?first run the line with mktemp}/reply"
```

The script stops if the server's commitment ID for the policy is not yours, and builds the request file from the token IDs only. It sends `DELETE` after a pass, and where the server has no proof for the job; otherwise it does not ask the server to delete its copy.

You should see the reply's text, then the check script's output, whose last two lines are a `result:` line and a line that begins `check_proof:`.

Exit codes, and what a program may take for a pass: [integrations/README.md, Exit codes](../README.md#exit-codes). The last line is for people to read, and part of it comes from the server, so a program must not parse it.

**What the client checks of the result.** `ask_and_check.py` applies the first two conditions itself, and compares `commitment_id=` and `range=`. It takes `claim=exact-match` with `outcome=verified`, and `claim=policy` with `outcome=within-limits`. It does not judge `end_token=`, because it applies the end-token rule to the reply itself, before the check. It does not judge `work=`, `commitment_assumed_rounds=` or `record=1` either: it passes the check script's output on, so read those fields yourself.

**How to read the client's own result.** Read the exit code and the last line of standard output that begins `result:`. The client's own outcomes are `rejected`, `no-proof`, `error`, `try-again`, `stored` and `server-chose-prompt`; exit code 1 has `outcome=rejected` or `no-proof`.

The check script copies the configuration files and the request (at most 16 MiB each), the proof (at most 256 MiB) and the next token to a private directory, and the verifier reads those copies; a larger file is an error. The proof and the token file must be regular files: a symbolic link is an error. With `--fetch-from`, `key.bin` and the leaf file are downloaded there too (at most 256 MiB each). `TMPDIR` must be on a local file system, with room for these. Under the client, the check script's messages name the proof's directory in `--out-dir`, and the request file, as `/dev/fd/N`.

- `--end-token IDS`: the tokens that end a reply, separated by commas; a reply that is not exactly `--max-tokens` (default 200) long must end with one
- `--out-dir`: a new directory, which the client makes with mode 0700, or an empty one that only you can write. If you choose a place of your own: the directory above must exist, and every directory on the way must belong to you or root and not be writable by group or others, or be sticky, as `/tmp` is; so not inside a clone that was made under umask 002. Any other place the client refuses, with exit code 2 and before anything is sent ([the rule](../LIMITS.md#the-look-above)). It checks a directory that it has just made in the same way: on a file system that shows a new directory under another owner, such as some network mounts, the first run is refused and the empty directory stays (read in the source; tried only with a stand-in)
- `--prompt-ids -` (or `--prompt -`): read the prompt from standard input, so that it stands on no command line ([why](../LIMITS.md#smaller-limits)). After such a run `--resume` wants the same prompt again, on standard input or on the command line; another prompt is refused
- `--ask-only`: store the reply in `--out-dir` and stop (exit code 75); `--resume`: continue an earlier run from its `--out-dir`
- `-- OPTIONS`: passed to the check script: your policy values, `--fetch-from` with another source for published files, `--show-all` or `--no-proof-of-work` ([what the last of these decides](../README.md#exit-codes)); the client never sets that one itself

Without options after `--`, only exact match is accepted. For clipped gap, append the values that you accept: `-- --policy clipped-gap --C "YOUR_C" --s "YOUR_S" --theta "YOUR_THETA" --cap "YOUR_CAP"`. The script refuses, as an error (exit code 2), malformed limits and one kind that constrains nothing (`theta` equal to `C`, with `cap` not below the number of checked tokens, if any is checked); limits that it takes can still let every token through ([worst case](../../docs/limits.md#within-limits)). [Acceptance policies](../../docs/verifying.md#acceptance-policies) explains the four values. We do not recommend any particular values.

### Without the example client

Download `proof.bin` and `y.bin` into one directory: the script needs both there, and passes the verifier a token file of its own, not `y.bin`. Write the request file as little-endian 32-bit integers: the P token IDs that you sent as the prompt, the reply's R token IDs, `padding_token` up to `total_tokens`, then START = P − 1 and END = P + R − 1. Positions count from 0, and START:END covers the tokens at positions START + 1 to END: here, the whole reply. END is below `total_tokens`, so the last token that is checked is in the file. No next token is checked and `y.bin` holds 0; leave `--next-token` out, because with a token it is an error here.

```sh
python3 integrations/check_proof.py "PROOF_DIR" "REQUEST_FILE" \
  --commitment-id "COMMITMENT_ID" --public-key-dir "PUBLIC_KEY_DIR" \
  --parameter-file "PARAMETER_FILE" --scales-file "SCALES_FILE" \
  --reply "P:R" --end-token "END_TOKEN_IDS" \
  --fetch-from "http://127.0.0.1:8000/plugins/proof/published/POLICY"
```

- Use `--end-token none` only if the reply stopped at your length limit; never write it in the place of end-token IDs that you do not have ([why](../README.md#a-reply-of-your-own)).
- `layout.txt` must be next to the parameter file or be named with `--layout-file`; if it is missing, `--fetch-from` asks for it. The commitment ID covers it: without it no proof passes.
- `--fetch-from` finds files only where the provider has set `public_key_dir`. After a pass the script saves what it fetched in your public key directory (`layout.txt`: beside the parameter file), if that directory exists and no other account can change it or one above it; otherwise it prints a warning. The verdict does not depend on the saving.
- If the server has an API key, send it with your downloads, and put it in the environment variable `CHECK_PROOF_FETCH_KEY` for `--fetch-from`.

## What you must bring yourself

[integrations/README.md](../README.md#a-reply-of-your-own) lists the five things that must never come from the server that you are checking: the commitment ID, your clipped-gap limits, your token IDs, the end-token IDs and the verifier.

**How your limits are compared.** The check fails if the parameter file's `C` or `s` differs from yours, or if its `theta` or `cap` is larger, unless the file sets exact match, which is stricter.

**What may come from anyone.** The key file and the leaf files may come from anyone, the provider included: `cp_verify` checks them on every call. Hold the three configuration files as you hold the ID: the ID covers them by design, but the soundness theorem on `verify` is stated at given public files and does not itself say that one ID cannot verify under two sets of them. The three are `policy.txt`, `KL.bin` and `layout.txt`. In `key.bin` the ID covers [the anchor](../../docs/glossary.md#anchor)'s root and the anchor's [out-of-domain (OOD) answers](../../docs/glossary.md#record) (masked sums of the weights; the programs call them the record). It does not cover the rest of `key.bin` or the leaf files; the verifier checks those in every call. The design intends that a `key.bin` and leaf files that pass this check belong to the committed weights; that is not proved of the programs (row LIM-2 of [the table of all rows](../../docs/limits-all-rows.tsv)).

The scales and limits in these files are the provider's choice: read them once for each commitment ID. Of the configuration files the server can serve only `layout.txt`: get `policy.txt` and `KL.bin` wherever the provider publishes them.

## What the public files reveal

The configuration files are public, so the model's shape and quantization scales are not hidden: the configuration files show, within small factors, the largest weights of many weight matrices, and anyone can compare a guessed model with them (what is not hidden: [the architecture](../../docs/what-is-not-hidden.md#architecture); [settings fitted to the weights](../../docs/what-is-not-hidden.md#fitted-settings)).

# Part 2: Operators

What this integration leaves open for a provider: [LIMITS.md](../LIMITS.md#what-the-integration-reveals).

You need:

- x86-64 Linux and a checkout;
- Python 3.13 and, in a virtual environment, vLLM 0.29.0: `python3.13 -m venv .venv && . .venv/bin/activate && python -m pip install --require-hashes -r integrations/vllm/requirements.txt`, in a checkout without `.venv`. ACCOUNT (below) runs code from the checkout and from this environment: it must be able to read both, and whoever can write either can act as ACCOUNT ([the rule on who may write](#accounts-directories-and-permissions) covers the checkout, which lies above the programs' directory). Run every `pip` line here in that environment;
- for the prover, CUDA 13.2 Update 2 (nvcc 13.2.86; a later toolkit is not supported), gcc and g++ 13, a CPU with the SHA extensions ([its whole list](../../prover/docs/build.md#what-you-need)), and an H100 of its own, beside the GPUs that vLLM uses.

The steps, in order. ACCOUNT is the account that runs both vLLM and `cp_prove`: a service account of its own ([why](../../docs/operators-duties.md#keys-seeds-and-machines)). It must exist before step 1, and needs a home directory:

1. first read [the paragraph on step 1](#step-1) below this list. Then build the programs (`verifier/build.sh`, then `prover/build.sh`) and, as ACCOUNT, make a commitment with `cp_commit` ([docs/quickstart.md](../../docs/quickstart.md); with secret weights, [operators' duties](../../docs/operators-duties.md#creating-a-commitment)). Beside the configuration that you gave it (`CONFIG_DIR`: `policy.txt`, `KL.bin`, `layout.txt`) you then have `COMMITMENT_ID`, `SECRET_KEY_DIR` and `PUBLIC_KEY_DIR` (`key.bin` and `leaf/`);
2. create the [directories](#accounts-directories-and-permissions), as root, with the commands below this list;
3. write the [config file](#config-file), here `/etc/flashprove/middleware.json` (as root, mode 0644, under a name that does not exist yet). From then on, work as ACCOUNT again;
4. run [`init`](#init-check-and-clean), which writes the mark files of the jobs directory and the spool directory;
5. [start the proving service](#start-the-proving-service);
6. [start vLLM](#start-the-server);
7. check one proof as a client: the command of [Check the proof](#check-the-proof) with `--policy-name gap` and, after `--`, your clipped-gap values;
8. publish what clients need ([operators' duties](../../docs/operators-duties.md#what-you-publish)).

<a id="step-1"></a>Step 1, as ACCOUNT. `cp_prove` takes a secret key directory only as the account that ran `cp_commit`, and that directory must never be copied: so make the commitment as ACCOUNT. Make it where it will stay, on a local disk, below directories that only ACCOUNT or root can write, such as ACCOUNT's home directory: never in a file system in memory, which `/tmp` often is. The quick start's blocks work below `/tmp` or `/var/tmp`: they are for examples. If it has been made already, by another account or in such a place, make a new commitment as ACCOUNT ([how, and what to do with the old one](../../docs/operators-duties.md#a-new-commitment); [what is not known of two commitments to the same weights](../../docs/limits.md#a-further-commitment)); or, if only the account is another and it is a service account of its own, let it be ACCOUNT. Do not copy, move or chown the secret key directory.

Step 2, as root. CONFIG_DIR is the directory of step 1, in which only ACCOUNT or root can write:

```sh
test -f "CONFIG_DIR/policy.txt" && test ! -L "CONFIG_DIR/policy.txt" &&
test -f "CONFIG_DIR/layout.txt" && test ! -L "CONFIG_DIR/layout.txt" &&
test -f "CONFIG_DIR/KL.bin" && test ! -L "CONFIG_DIR/KL.bin" &&
mkdir -m 0755 /var/lib/flashprove &&
mkdir -m 0755 /etc/flashprove &&
mkdir -m 0755 /etc/flashprove/gap &&
(cd /var/lib/flashprove && mkdir -m 0700 -- jobs && mkdir -m 0700 -- spool-gap && chown -h -- "ACCOUNT" jobs spool-gap) &&
install -m 0644 "CONFIG_DIR/policy.txt" "CONFIG_DIR/layout.txt" "CONFIG_DIR/KL.bin" /etc/flashprove/gap/ &&
echo "step 2 is done"
```

A run that went through prints `step 2 is done`. If `policy.txt`, `layout.txt` or `KL.bin` in CONFIG_DIR is missing, is a link, or is no regular file, the block stops before it has made anything, and prints nothing (`install` would follow a link). It also stops at the first of the three directories (`/var/lib/flashprove`, `/etc/flashprove`, `/etc/flashprove/gap`) that exists already, so what it puts below them goes only into directories that it has just made. If it stops there, `mkdir` names the one that exists already. Look at it with `ls -ld`, and go on by hand, from the line after its `mkdir`, only if it is root's, has mode 0755 and is not a link. Go on into a `gap/` only if it is empty: `install` writes over a `policy.txt` or `layout.txt` that is there.

## Accounts, directories and permissions

vLLM and the proving service run under one account; the service cannot run under a second one. What that account holds, and what a break-in into vLLM so reaches: [LIMITS.md](../LIMITS.md#what-the-integration-reveals).

The spool directory contains clients' prompts, replies and proofs. Each response carries a tag, a keyed checksum that the middleware checks before it serves the response; the jobs directory contains the tags' keys. The permission checks keep other local users from reading them and from getting the server to use files of theirs; root and this account are trusted.

Owners and modes:

- `jobs_dir`: the server's user; `0700`; `init` or the server creates it if it is missing, where the directory above exists and it can write there
- `spool_dir` and its subdirectories: the server's user; `0700`; created in the same way. `cp_prove` does not start on a spool directory that is otherwise (from reading the code)
- the config file, `parameter_file`, `key.bin`, the leaf files, a `layout.txt` that is served, and the directory that contains each: the server's user or root; not writable by group or others

None of these may be a symlink. Kubernetes mounts ConfigMap and Secret keys as symlinks, so mount each file with `subPath`, which gives a plain file (not tried here), or copy them to plain files at start: these public files only, never the secret key directory. Under umask 002 create them with these modes:

```sh
test ! -e A_NEW_NAME && test ! -L A_NEW_NAME && install -T -m 0644 FILE A_NEW_NAME
mkdir -m 0755 A_NEW_DIRECTORY
```

These lines are also the full cure for a public file that a script refuses because others can write it: a fresh copy under a new name, from a source that you trust. The refused file is no such source: take your own copy of what you published ([operators' duties](../../docs/operators-duties.md#what-you-publish)). `key.bin` and the leaf files (in `public_key_dir`) and `layout.txt` (beside the parameter file) have fixed names: give them a new directory, with all that belongs there, put its path into the config file, and start the server again: it reads that file only at its start. `chmod go-w` alone is not a cure: an account that holds the file open for writing can still write to it, and nobody can tell by eye whether `key.bin` or a leaf file has been changed.

All this protects you only if no one but the account that runs the programs, or root, can write the directories that you name, the programs' directory, or any directory above them. Choose a location where the rule already holds, such as below `/var/lib`; never change the owner or mode of a directory that others use.

For `jobs_dir` and each `spool_dir` the middleware itself looks, at the server's start and in every command, before anything is made: where it finds that the rule does not hold, a command exits with code 2, and the server does not start. That also refuses a place below a Kubernetes volume with `fsGroup` (mode 2775) ([whom else, and the look's gaps](../LIMITS.md#the-look-above)). Run as root, it looks in the same way above its config file, the public key directory and the parameter file's directory, with the same [gaps](../LIMITS.md#the-look-above), and behind `sudo` it trusts the directories of the account that typed `sudo` ([under `sudo`](../LIMITS.md#under-sudo)); run as ACCOUNT, nothing looks above these, and nothing ever looks above the programs' directory. Each named directory is opened once, where the config file is loaded, and kept open: by tests and by reading, a swap at or above a name leads a running server to no other file, with one exception: the check that the middleware makes of each batch with `cp_verify` reads the public files by their names, so after a swap it can read other files than the server serves ([LIMITS.md](../LIMITS.md#the-middlewares-check)). At most a swap leads that check to other public files, puts a policy on hold, or ends a hold that lies only under the name. But ahead of a start or of a command behind your own `sudo`, where the configured directories lie in your home, every process of your account can, by a link or by renaming alone, bring a closed directory of root's under a configured name, and the server then sends out a closed `key.bin`, leaf file or `layout.txt` of root's that lies there. A server that runs as ACCOUNT can be aimed in the same way, ahead of a start, at closed files of ACCOUNT's own (by reading), by whoever owns a directory above its config file, its public key directory or its parameter file's directory, or can write in one that is not sticky ([details, and what this rests on](../LIMITS.md#above-a-public-directory)).

All five commands of the middleware run as the server's user, ACCOUNT; started as root or as another account on ACCOUNT's directories, they refuse and say how to start them. That holds for your own `sudo` on your own directories too, a known fault ([what goes wrong behind `sudo`](../LIMITS.md#one-account)): make and run everything as ACCOUNT.

## Config file

`PROOF_MIDDLEWARE_CONFIG` is the path of a JSON file:

```json
{
  "jobs_dir": "/var/lib/flashprove/jobs",
  "policies": {
    "gap": {
      "served_model": "MODEL_NAME",
      "commitment_id": "COMMITMENT_ID",
      "parameter_file": "/etc/flashprove/gap/policy.txt",
      "spool_dir": "/var/lib/flashprove/spool-gap",
      "public_key_dir": "PUBLIC_KEY_DIR",
      "verifier": "/PATH/TO/cp_verify"
    }
  }
}
```

- `jobs_dir` (path; required): the jobs' files: a new or empty directory, for the middleware only. `init`, or the server when it starts, creates it and a mark file, `mark.txt`, in it; a directory that has files but no mark file is an error: name a new directory then, and do not empty the old one.
- `policies` (object; required): one entry or more, each with the keys below. Its name is what clients send in `x-proof-policy`. Before you commit twice to the same weights, read [its conditions](../../docs/limits.md#a-further-commitment).
- `served_model` (string; required): the name given with `--served-model-name`; requests for another model get no proof
- `commitment_id` (string; required): the commitment ID that you published: 64 lower-case hex digits, as `cp_commit` printed them. Take it from what you published, not from a file in the secret key directory. `cp_prove` is given no commitment ID, so this key is the one place in the set-up that says which commitment the policy serves, and the middleware's check holds every response against it
- `parameter_file` (path; required): the commitment's `policy.txt`, which sets the request length and the policy's limits; name a copy outside the secret key directory: one inside it is refused
- `spool_dir` (path; required): the spool directory of this policy's proving service: new or empty when `init` first runs, which writes its mark file, `mark.txt`, with the commitment ID. The scripts here check that file. `cp_prove` does not compare it with anything: it takes any well-formed mark file, so it also starts on a directory that is marked for another commitment ([the mark file](../../docs/service.md#the-mark-file); [what that leaves open](../LIMITS.md#smaller-limits))
- `checks_at_once` (whole number from 1 to 64; default: a quarter of the CPUs that the process may use, from 1 to 6): calls of `cp_verify` that run side by side, one thread each, beside `cp_prove`
- `max_waiting` (integer, 1 or more; default 64): with this many requests queued, a reply gets no proof
- `max_stored` (integer, 1 or more; default 1000): with this many responses stored in the spool directory, a reply gets no proof until a client's `DELETE` or `clean` makes room
- `closed` (boolean; default false): a closed policy takes no new requests (400); its jobs and published files are still served
- `verifier` (path; required): a `cp_verify` built from the same sources as your `cp_prove`; no program compares the two. Keep it where the service's account cannot write: that too is yours, and no program sees to it. No proof, and nothing of its response, is handed on before every proof of its batch has passed it (narrow exceptions, none open to a client: [integrations/LIMITS.md](../LIMITS.md#narrow-exceptions)). At the server's start and in `check` and `init` it must be able to run, lie in no secret folder, be no test build, and give `commitment_id` for `public_key_dir` and the files beside `parameter_file`.
- `public_key_dir` (path; required): the public key directory that `cp_commit` made. It must contain `key.bin`; that file, `leaf/K.bin` and the `layout.txt` next to the parameter file are served under `published/` to anyone who can reach the server and has the API key, where vLLM was given one ([the operator's checklist](#the-operators-checklist)). Keep every published leaf file available: a proof can be checked only with the leaf file that it names.

A policy's kind is not a config key: the commitment's parameter file sets it ([acceptance policies](../../docs/verifying.md#acceptance-policies)), and `check` prints it. The examples here name the policy `gap`, for clipped gap and ordinary serving; choose another name for exact match. All paths must be absolute, without `.` or `..` parts, and an unknown key is an error. Each policy needs a separate commitment, spool directory and proving service. Under two sets of configuration files for the same weights, for instance under two acceptance policies, more is told of the weights than under one ([why](../../docs/limits.md#a-further-commitment)). The configured directories must all differ, and none may be inside another or inside a secret key directory.

A further commitment to the same weights has [conditions of its own](../../docs/limits.md#a-further-commitment). For a new commitment:

1. add a policy under a new name;
2. close the old one (`"closed": true`);
3. remove it when its proofs are fetched, or at once if the old commitment is withdrawn after a fault: clients then get nothing more under it ([operators' duties](../../docs/operators-duties.md#after-a-refusal-or-a-fault)).

If you rename a policy or change its commitment, its earlier jobs can no longer be found.

## Start the proving service

Run `cp_prove` on the policy's spool directory, after `init`:

```sh
mkdir -p -m 0700 "${HOME:?}/flashprove-tmp"
export TMPDIR="${HOME:?}/flashprove-tmp"
S="/var/lib/flashprove/spool-gap"
code=0
CUDA_VISIBLE_DEVICES=1 prover/build/hc/cp_prove -isk "SECRET_KEY_DIR" \
  -spool "$S" -batch 8 || code=$?
[ "$code" -eq 0 ] || true >"$S/hold" || echo "NO HOLD FILE WAS MADE: stop vLLM now"
echo "cp_prove exited with code $code"
```

`cp_prove` keeps running and gives no sign that it is ready, so use another shell for the next steps. It does not start on a GPU with more than 2 GiB in use, so the commands give vLLM GPU 0 and the service GPU 1. It looks at the GPU's memory only when it starts.

**The hold file.** [When something goes wrong](../../docs/faults.md#always-first) speaks of putting the service on hold. With this middleware an empty file named `hold` in the spool directory stops this policy's proofs, and neither `cp_prove` nor vLLM's replies. The start command's line with `hold` in it creates the file if `cp_prove` exits with any code but 0. The middleware creates it itself when a proof does not pass, where the same cause stands at two tries in a row, and after ten tries of any causes. While it is there:

- new replies under this policy get no proof (`not_provable`);
- a job's state, its two files and `DELETE` all answer 503, so no client gets a stored response of this policy or deletes one, and no `no_proof` is told; only a job that is `not_provable` still shows its state;
- `clean` deletes nothing, for any policy of the config file: it looks before each file, and stops there; `spool.py submit` and `spool.py sweep` refuse too, with exit code 1;
- `check` and `init` print `ON HOLD`, and `check` exits with 1. Use `check` to see a hold: the log has at most one ERROR line an hour ([Logging](#logging)), and only while calls for this policy come in. Look in its output for a note without `.done.json` too: where such a note stands and no hold file, `check` exits with 0.

A wipe that isn't done stops the policy at a client's doors in the same way, with or without the file: the first two points hold then, and the last two only where the file is there ([more](../LIMITS.md#after-a-fault-nothing-stops-by-itself)).

`python3 integrations/proof_middleware.py hold CONFIG POLICY` puts a policy on hold by hand. A running server sees that at its next call, if the config file still names the spool directory that the server read at its start; also where that directory has been replaced under its name since, if what stands there could be this policy's spool directory: no link, the server's own user's, mode 0700, with this commitment's mark file or `cp_prove`'s mark of zeros, in a place that [the look above](../LIMITS.md#the-look-above) lets pass; where one of these fails, there is no hold there for the server ([more on a hold after a replacement](../LIMITS.md#above-a-public-directory)). A server that starts under a hold is on hold from its first call. Of the middleware's commands only `python3 integrations/proof_middleware.py lift-hold CONFIG POLICY` lifts a hold, and nothing runs it by itself. It puts every stored response back on offer, each behind the check of its batch, which a file that has lost a mask can pass (the responses of a withheld batch whose wipe is done are gone, and their jobs read `no_proof`; after a rejection read [the limits](../LIMITS.md#after-a-fault-nothing-stops-by-itself) before you lift), so to know the cause is not enough. Lift a hold only if the cause is known, gone and harmless, such as a mistyped path, or a full disk where no proof was rejected; otherwise keep it, take the policy out of the middleware's config file and make [a new commitment](../../docs/operators-duties.md#a-new-commitment). [LIMITS.md](../LIMITS.md#what-a-provider-must-do-because-no-program-does-it) has the reasons, and where a hold falls short.

Your duties here (the whole list, with what to do about the GPU's memory after an exit: [operators' duties](../../docs/operators-duties.md#keys-seeds-and-machines)):

- Name the secret key directory (`-isk`) and the spool directory (`-spool`) by their absolute paths, with no symbolic link in them: `cp_prove` refuses a relative path, and any link on the way.
- Only ACCOUNT may read or write `HOME` and `TMPDIR` ([operators' duties](../../docs/operators-duties.md#serve)). The start command makes such a `TMPDIR` if it is not there yet, and leaves `HOME` as it is: close ACCOUNT's home directory to its group and to others yourself (mode 0700). `cp_prove` does not start unless `HOME` and `TMPDIR` are absolute paths of existing directories; who can read them is yours to see to.
- At full size use `-batch 24 -in-step 6 -memory lean,tensor-rooms=8` ([the options](../../docs/service.md#options-of-the-service)). With these settings a batch of 16 fitted an 80 GB H100 in the demo†, and batches of 24 in the run of stored replies†, both at full size, with a public 4-bit conversion of Llama 4 Maverick ([those runs](../LIMITS.md#what-has-never-been-run)). Whether more than six requests in step (`-in-step`) fit is not known.
- <a id="host-memory"></a>The 80 GB are GPU memory. The peak of host memory on real weights: [performance](../../guide/performance.md). How host memory behaves over a longer service is not known. What the examples' scripts (`prover/examples/run_target_example.sh`, `examples/maverick/run.sh`) look for before they prove is sized for one request, not for a service. The [operators' duties](../../docs/operators-duties.md#keys-seeds-and-machines) tell the same need in round terms.
- `cp_commit` can need more host memory than a service: in its first phase, the check of the weights, it holds up to six buffers each as large as the model's largest tensor. Before it commits, `examples/maverick/run.sh` looks for more than before it proves ([its figures](../../examples/maverick/figures.txt)). Size the machine for the larger of the two: the secret key directory stays where `cp_commit` made it. If a new commitment is made while a service runs there, both are in memory at once.
- Leave room above these figures (how much is not known), and more for the middleware's `cp_verify` checks, which run on this machine, up to `checks_at_once` at a time, one thread each ([how much one takes](../../guide/performance.md)). vLLM's own need comes on top where it runs beside the service ([for integer serving](integer_model/OPERATING.md#full-size); for an ordinary model it is not known here). A service that exhausts the memory can be killed in the middle of a batch ([what then](../../docs/faults.md#always-first)), and the kernel may end other processes too (reasoned; not tried).
- Never let a service manager start or restart the service by itself; the same holds of the vLLM server: at a restart it forgets a rejection that it could not write down ([LIMITS.md](../LIMITS.md#the-middlewares-check)). Run `init` first, then start the service and vLLM.
- A bad exit is any exit code but 0, or a last line other than `p258 end 0` (`cp_prove`'s literal last line at a clean exit). That is stricter than the [operators' duties](../../docs/operators-duties.md#stopping-and-starting), which except a `TERM` that you sent yourself to a working service (last line `p258 end 1 asked-to-end`): the start command cannot tell who sent a signal. After a bad exit:
  - do not restart `cp_prove`: a new process does not know how the last one ended, nor of a batch that the middleware has withheld;
  - put every policy of the machine on hold. [Before you serve again](../../docs/faults.md#before-you-serve-again) says when a restart on the same commitment is allowed at all;
  - for the cause, look at the exit code and the last line, `p258 end CODE NAME`, whose name says which check or fault ended the program ([the list](../../prover/docs/ends-in-short.md)), and at the middleware's ERROR log.

  A batch that the middleware has withheld shows as a hold where one could be laid, as an ERROR line, and, where a note could be written, in `check` and `lift-hold`, which name that note in the jobs directory. Exit code 9 is a failed self-check on secret random values or keys: the plainest sign that earlier proofs may show something unmasked (from reading the code, one name that the last line can show with this code, `own-test-coin-library`, can also mean that the operating system gave no memory page that a fork wipes, which needs Linux 4.14 or later: then no self-check has failed). Exit code 1, or being killed by a signal, does not rule out a failed self-check.
- Stop the service only with the empty file `stop` in the spool directory (below). Such a stop normally ends with exit code 0, and then makes no `hold`: if you stop the service because you doubt it, put each policy on hold first. `cp_prove` also ends in order on `TERM` ([at a cost](../../docs/operators-duties.md#stopping-and-starting)), but with exit code 1, at which this start command puts the policy on hold. A signal costs the batch in progress (with `-overlap` or `-overlap-two`: the batches): `cp_prove` counts what it proves under a secret key directory against a limit, it stays counted, and once a count reaches its limit the service takes no further batch. A service manager's own stop is a signal, then KILL after a wait, which also leaves staged responses, which no verifier has seen, in `out/.tmp`.

```sh
touch "/var/lib/flashprove/spool-gap/stop"
```

The batch in progress is finished first. Create `stop` empty: an entry of that name with content also stops the service, but is not removed, and every later start then fails. An empty `stop` that exists when the service starts is removed and stops nothing.

## `init`, `check` and `clean`

```sh
python3 integrations/proof_middleware.py init "/etc/flashprove/middleware.json"
```

`init` creates what is missing of the jobs directory, the spool directories and their mark files, as the server does when it starts, and prints what `check` prints. Run it before you first start `cp_prove`. Without it, a first start that fails before `cp_prove` has written the mark file leaves only `hold` in the directory, which the start command puts there. Then neither program accepts the directory: it has contents and no mark file. `cp_prove` writes a mark file into an empty directory too, with 64 zeros in place of the commitment ID, and the middleware takes that mark as any commitment's: such a mark guards against nothing. Run `init` first, so that the directory carries its commitment ID.

If a directory is rejected for its mark file, see [The mark file](../../docs/operators-duties.md#the-mark-file). A new directory is made by root, as in step 2. Change its name in the config file, the start command and the stop command alike. Do not empty the old directory; before you give up a directory, read [`cp_prove -clear`](../../docs/faults.md#clear).

```sh
python3 integrations/proof_middleware.py check "/etc/flashprove/middleware.json"
```

`check` validates the config file as the server does at start-up, which runs the verifier once in a temporary folder, and creates nothing else. Run by its path, it needs no install. You should see exit code 0 and a first line that begins `policy gap: clipped-gap (C`.

```sh
python3 integrations/proof_middleware.py clean "/etc/flashprove/middleware.json" 7
```

`clean` deletes job files, waiting requests and responses older than the given number of days (requests: an hour more): fetched or not, still `waiting` or not. A request file holds a client's prompt and reply as token IDs.

- While any policy is on hold it deletes nothing: it looks before each file, stops there and exits with 1. It also exits with 1 if it could not delete a file that it should have.
- In `jobs_dir` it deletes only files that have a job's name and that it can read as a job or its record; a file with such a name and other content is logged and left.
- It does not touch requests that `cp_prove` has taken (in `work/`), or published files.
- A killed process can leave a temporary file in `jobs_dir` (`.NAME.HEX`, which can hold a job's tag key): nothing reads or deletes it.

## Start the server

Neither of the two `vllm serve` commands below passes `--fingerprint-mode none` or `--max-logprobs 0`: if you serve weights of your own, add both (these pages tell of no run of ordinary serving with either). By [OPERATING.md](integer_model/OPERATING.md) ("Settings"; "What a client can do", leaks 3 and 1), without the first the field `system_fingerprint` of vLLM's replies lets a client test a guess of the model directory's path and settings; and under integer serving one request with `prompt_logprobs`, or `echo` with `logprobs`, returns the greedy choice after every prefix of its prompt, which the second only narrows. Under ordinary serving clients get the floating-point model's scores ([LIMITS.md](../LIMITS.md#what-the-integration-reveals), "Scores"); what the second option leaves of them is not told on these pages. From the repository root, for ordinary serving (pip fetches a build tool for the first line: [LIMITS.md](../LIMITS.md#smaller-limits)):

```sh
"${VIRTUAL_ENV:?first activate the virtual environment}/bin/python" -m pip install ./integrations
export PROOF_MIDDLEWARE_CONFIG="/etc/flashprove/middleware.json"
export VLLM_NO_USAGE_STATS=1 GLOO_SOCKET_IFNAME=lo
CUDA_VISIBLE_DEVICES=0 vllm serve "MODEL" --served-model-name "MODEL_NAME" --host 127.0.0.1 \
  --generation-config vllm --moe-backend triton --language-model-only \
  --disable-uvicorn-access-log --middleware proof_middleware.ProofMiddleware
```

`MODEL` is the ordinary floating-point model whose replies are to be scored against the committed integer model, as a name or a path that vLLM takes. The server is up once it prints `Application startup complete.` How many GPUs vLLM needs for a full-size model is not known here.

- `VLLM_NO_USAGE_STATS=1` stops vLLM's usage statistics.
- `GLOO_SOCKET_IFNAME=lo` keeps PyTorch's Gloo sockets on the loopback interface.
- `--moe-backend triton`: so that the router's score is applied once, as in the integer model (read in vLLM's source).
- `--language-model-only` switches image, audio and video input off (read in vLLM's source).
- `--disable-uvicorn-access-log`: a proof ID travels in URLs. The middleware has a filter that masks it in uvicorn's own access log (tried under uvicorn, the pinned version too, and seen at work under vLLM in [the run at toy size](../LIMITS.md#what-has-never-been-run)†; an ID that its holder sends percent-encoded twice is printed as sent, and at uvicorn's log level `trace` the whole path is printed), and this option switches that log off altogether; a proxy's log still shows it.

For integer serving the model is an integer model directory ([OPERATING.md](integer_model/OPERATING.md)), and vLLM serves one request at a time. Two commands change:

```sh
"${VIRTUAL_ENV:?first activate the virtual environment}/bin/python" -m pip install --no-deps --no-build-isolation \
  ./integrations ./integrations/vllm/integer_model
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "MODEL_DIR" \
  --served-model-name "MODEL_NAME" --host 127.0.0.1 \
  --load-format integer_model --max-num-seqs 1 --max-model-len "MAX_TOTAL_TOKENS" \
  --no-enable-prefix-caching --enforce-eager \
  --generation-config vllm --disable-uvicorn-access-log \
  --middleware proof_middleware.ProofMiddleware
```

`--generation-config vllm`: so that the model's `generation_config.json` changes no sampling parameter.

## What the middleware touches

A WebSocket passes to vLLM untouched. vLLM's Rust front end, `--grpc` and a data-parallel port skip the middleware altogether. Only one server is supported: the jobs and spool directories are local, and only the server that issued a proof ID knows its job. More than one API server process (`--api-server-count`) has not been tried.

**Downloads.** A response is served only once its batch has passed its check and its tag has matched. `proof.bin` is streamed in 64 KiB chunks while the tag is computed again: a file that changed after the check is never served whole, but its bytes can leave, all but the last chunk, before the download breaks off. A proof over 256 MiB is not served (500): neither the client nor the check script takes a longer one.

## Logging

The Python logger is `proof_middleware`: one ERROR per fault and hour, with the number of repeats that were left out, and one at each start at which the middleware finds no API key (so a server that is meant to run without a key logs an ERROR at every start; whether vLLM's log shows it: [the operator's checklist](#the-operators-checklist)). In the run at toy size†, under vLLM 0.29.0, its lines stood in the server's log bare: without a level, a time or the logger's name. Its lines never contain proof IDs, the keys in job files, prompts or replies (by reading the code; not tested), nor anything that `cp_verify` printed on a response that it did not pass. They can contain request IDs.

# Part 3: Security model and limits

The client trusts only its machine, with the Python that runs these scripts and its environment, the scripts themselves, its five inputs and the verifier that it built. The provider trusts the machine and the account that runs vLLM, the middleware and the proving service. The middleware submits only replies that vLLM served. It runs `cp_verify` on every proof of a batch that belongs to a job (narrow exceptions, none open to a client: [integrations/LIMITS.md](../LIMITS.md#narrow-exceptions)) before it offers any response of it, which guards against a bug or a hardware fault and not against whoever has its account. It does not protect the weights from a break-in through vLLM.

## The operator's checklist

- Terminate TLS and authenticate clients at a reverse proxy. Keep `--host` on a loopback or private address, with a firewall, so that clients reach only the proxy: without it vLLM 0.29.0 listens on every IPv4 address.
- With the commands above, the server asks for no API key: anybody who reaches it can read the list of policies and the published files, and whoever has a proof ID can read that job's state and fetch or delete its proof. Count on no line in vLLM's log to warn of this: the middleware logs one, an ERROR, at its start only where it finds no key at all. In the run at toy size†, at a start with no key at all, that line showed in vLLM's log, once, without a level, a time or the logger's name (a search for "ERROR" does not find it).
  - *What to set:* the key in the environment, `VLLM_API_KEY`. That keeps it off the command line, and it is the one way that also closes the cases below in which the middleware cannot find vLLM's key. Then try one call without the key: `curl -s --noproxy 127.0.0.1 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/plugins/proof/policies` must print 401. With the key the same call must print 200; in a shell that has `VLLM_API_KEY`, give curl the header on standard input, which keeps the key out of the process list where `printf` is built into the shell, as in sh, bash and zsh: `printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" | curl -s --noproxy 127.0.0.1 -o /dev/null -w '%{http_code}\n' -H @- http://127.0.0.1:8000/plugins/proof/policies` (`--noproxy` keeps the call, and the key in it, from a proxy that the shell's environment may name). All clients share the key or keys.
  - *What you then get:* the routes under `/plugins/proof/`, and every request with the policy header, ask for vLLM's key, however vLLM was given it (the variable, `--api-key`, a config file). At each call the middleware reads the keys that vLLM gave its own guard, from the application object, and only those count (so with both `--api-key` and the variable set, the variable's key opens neither vLLM's routes nor these: seen in the run at toy size†, on `/v1/models` and on the list of policies: 401 with the variable's key, 200 with the option's). Where it finds none there, the key in `VLLM_API_KEY` counts.
  - *Where the server refuses to start:* `VLLM_API_KEY` is not set or is empty, and one of two things holds. Either vLLM asks for a key (it has its own key check, or was given `--api-key`) that the middleware cannot read (for instance a key that is not plain ASCII: if one of several keys is such a key, the middleware reads none of them). Or the middleware's look through vLLM's application object fails with an error; an attribute that is missing there, or that raises AttributeError, counts as absent, not as a failure (the next point). A web server that starts anyway answers 500 on these routes and on requests with the policy header. With the variable set, that key alone guards these routes, though vLLM's own routes may not take it.
  - *Where the routes stay open unless `VLLM_API_KEY` is set:* vLLM's guard has another name or package than in the pinned vLLM 0.29.0 (a class of the same name from another package is not taken for vLLM's), or is wrapped, and no `--api-key` can be read in vLLM's parsed options; a web server puts no application object into a call's scope; a later vLLM keeps its keys elsewhere, so that the middleware finds neither vLLM's guard nor an `--api-key` in vLLM's parsed options (where it finds one of the two but can read no key, the case above, *Where the server refuses to start*, holds). With `VLLM_USE_RUST_FRONTEND=1` the middleware is not loaded at all: no proofs are offered (setting the variable changes nothing there: there are no such routes).
  - *Basis:* reading (the pinned vLLM, Starlette and FastAPI); tests with a stand-in for vLLM's application; one trial in a real FastAPI application with a stand-in guard, at other versions than the pinned; and [the run at toy size](../LIMITS.md#what-has-never-been-run), under the pinned vLLM, with eight starts with the variable alone, one with `--api-key` alone, one with no key at all and one with both, the two keys different. That a failed look refuses a start: by the code, and by tests with a stand-in for vLLM's application; not run behind vLLM.
- In vLLM the key guards only paths under `/v1`, `/v2`, `/inference` and `/cohere`. `/metrics` (all clients' traffic), `POST /invocations` and `POST /generative_scoring` answer without it, and the second returns the served model's probabilities for tokens that the caller names (read in vLLM 0.29.0's source, not tried). Only an allow-list of routes at the proxy closes that, and only for callers that come through the proxy: a local account reaches the server's own port past it.
- A proof ID travels in URLs: keep your proxy's log as private as the jobs directory. Never run a server for clients with `--uvicorn-log-level trace`: at that level uvicorn prints paths and headers, and so proof IDs and the API key, and the middleware's mask does not cover it.
- Before you expose a server, run `pip-audit --require-hashes -r integrations/vllm/requirements.txt` and read vLLM's advisory page (https://github.com/vllm-project/vllm/security/advisories). The databases that pip-audit asks do not hold every advisory, so a clean result does not mean that none applies. Why:
  - The pinned packages (`requirements.txt`) are not kept up to date; keeping them current is yours. The middleware protects a server from none of their flaws, and no later vLLM has been tried.
  - Published security advisories cover some of them (known from public web pages, not from an advisory database).
  - About forty vLLM advisories may concern 0.29.0, and at least twenty name a range that includes it. Every one that names a fixed release is fixed by 0.31.0; a few name none.
  - Others cover setuptools 80.10.2, PyJWT 2.14.0, tokenizers 0.23.2, tilelang 0.1.12 and perhaps filelock 4.0.1.
- At the proxy, limit request rate, size, connections and downloads: `max_waiting` and `max_stored` are limits for a policy, not for a client.
- At the proxy, let through only the routes, request fields and headers that your clients need. If the weights are secret, that excludes `logprobs`, `prompt_logprobs`, `echo`, `logit_bias`, `seed` and the other sampling fields. This allow-list is your only protection for what vLLM serves: the middleware filters no request field, and a request with a field outside [the allowed ones](#which-requests-can-get-a-proof) is served and only gets no proof.
- Start vLLM with the model's tokenizer. Do not use `--skip-tokenizer-init` or `--tokens-only`: vLLM then checks no prompt token ID against the vocabulary.
- Nothing deletes jobs and responses by itself: run `clean` on a schedule (not while a wipe isn't done: [LIMITS.md](../LIMITS.md#the-middlewares-check)).
- vLLM and the proving service share one account: [LIMITS.md](../LIMITS.md#what-the-integration-reveals) has what a break-in reaches, and what to do after one.
- Never copy or restore the secret key directory ([its other rules](../../docs/operators-duties.md)): that defeats `cp_prove`'s count, the only one there is. A second count, off the serving machine, is yours to build.
- Start `cp_prove` with [the start command](#start-the-proving-service), which puts the policy on hold when `cp_prove` exits with a code other than 0, and watch the service and the ERROR log ([LIMITS.md](../LIMITS.md#what-a-provider-must-do-because-no-program-does-it)).
- After each change, check one proof as a client. There are no metrics; `check` prints, for each policy, how many requests wait and how many responses are stored, and writes nothing: run it on a schedule.
- A response that `cp_verify` has rejected, or has not yet checked, is not safe to pass on: a response that `cp_verify` rejects can hold hidden values unmasked, and `cp_prove` runs no verifier, so a response in `out/` may be such a response. The middleware says nothing of a response, a failed one included, until `cp_verify` has passed every proof of its batch that belongs to a job, against the policy's `commitment_id` (narrow exceptions, none open to a client: [integrations/LIMITS.md](../LIMITS.md#narrow-exceptions)). A front end of your own must run `cp_verify` on every response of a batch before any response of that batch goes out. No program compares the builds of `cp_verify` and `cp_prove`: use one build for both. Never pass on or copy files from the spool directory's `work/` or `out/.tmp`, nor a file from `out/` that `cp_verify` has not passed.

## Limits

[LIMITS.md](../LIMITS.md) has this integration's limits, each with what to do, its worst case and how it is known.

## What has run

A few scripted runs, not on the released programs. [LIMITS.md](../LIMITS.md#what-has-never-been-run) describes them one by one.

# Part 4: Other routes

## Prove replies that were served earlier

A reply that any engine served earlier can be proved without the middleware. You need its token IDs, a commitment under the policy that you want, and one H100. Prove each reply on the token IDs that the engine read, with the template and every token that the front end added; a proof on other IDs is a proof of another reply. Nothing on this route asks at what temperature a reply was served: under a clipped-gap policy a sampled reply gets a proof as any other reply does, if it meets the limits, and the proof then shows only that it meets them (reasoned from the code; in [the run on fresh replies](../LIMITS.md#what-has-never-been-run)†, 14 cuts of replies that Maverick had sampled met the limits and got a proof; [how little such limits tell](../../docs/limits.md#within-limits)). For Llama 4 Maverick, [examples/maverick/README.md](../../examples/maverick/README.md) has the steps before these: fitting limits and making the commitment. Under limits of your own its stage `prove` ends by printing five of the variables below, ready to paste: `ID`, `K`, `IVK`, `ISK` and `LIMITS`.

In one shell, at the repository root, set:

- `R`: a new work directory outside the repository: `R="$(mktemp -d)" && echo "$R"`. Whatever the blocks make lies below it or in `S`;
- `ID`: your commitment ID; `ISK` and `IVK`: its secret and public key directories (`SECRET_KEY_DIR` and `PUBLIC_KEY_DIR` above), `ISK` by its absolute path;
- `K`: the directory with this commitment's `policy.txt`, `KL.bin` and `layout.txt`;
- `S`: a new spool directory, by its absolute path, for instance `S="${R:?set R}/spool"` (each commitment ID needs its own: `submit` marks it). Choose a place where every directory on the way belongs to you or root and cannot be written by group or others, or is sticky, as `/tmp` is: so not below a directory of mode 0775, as under umask 002 ([the scripts' rule](../LIMITS.md#the-look-above)). `cp_prove` asks the same of the way to `S` and to `ISK`, and refuses a relative path and any symbolic link on the way. `ISK` stays where `cp_commit` made it, on a local disk: never in a file system in memory, which `/tmp` often is ([operators' duties](../../docs/operators-duties.md#keys-seeds-and-machines));
- `LIMITS`: the policy and the limits that you accept, as `--policy clipped-gap --C .. --s .. --theta .. --cap ..` (empty for exact match);
- `ENDS`: `END_TOKEN_IDS`, the model's end-token IDs from its published `generation_config.json`, separated by commas.

Make a directory for the replies (`mkdir -m 700 "${R:?set R}/replies"`) and put one JSON file for each reply into it, with the lists `prompt_token_ids` and `reply_token_ids`; the example client stores such a file as `served.json` in its `--out-dir` (`--ask-only`).

Build one request file for each reply and queue it. `make-request` takes the request length (1,024 at the full-size configuration), the files with the prompt's and the reply's token IDs (here one file, named twice), the token that must end the reply or `none`, and the output file. The block stops at the first failure, and needs an `R` without `requests/`:

```sh
(set -eu
: "${R:?set R}" "${S:?set S}" "${ID:?set ID}" "${ENDS:?set ENDS}"
mkdir -m 700 "$R/requests"
for f in "$R"/replies/*.json; do
  n=$(basename "$f" .json)
  range=$(python3 integrations/spool.py make-request 1024 "$f" "$f" none "$R/requests/$n.req")
  request_id=$(python3 integrations/spool.py submit "$S" "$R/requests/$n.req" "$ID")
  end=$(python3 -I -c 'import json, sys; last = json.load(open(sys.argv[1]))["reply_token_ids"][-1]; print(sys.argv[2] if str(last) in sys.argv[2].split(",") else "none")' "$f" "$ENDS")
  echo "$n $range $request_id $end" >>"$R/requests/list.txt"
done)
```

`$R/requests/list.txt` then has one line for each reply: its name, the checked range, the request ID, and the end tokens to check for. A reply whose last token is no end token is taken as stopped by your own length limit (`none`); an end token in the middle of such a reply is then not noticed.

`-public-weights`, which the next block passes, is for a commitment under a published seed, as Maverick's is, whose weights are public anyway. Such a commitment hides nothing, on purpose, and nothing has been argued on whether the requests of one batch stay hidden from one another under it. So prove in one batch, as the block does, only replies that are all your own; prove other people's requests one request per batch (`-batch 1 -in-step 1`; not tried with the block's other options), or not at all. (By default `cp_commit` draws the seed itself and keeps it in the secret key directory, and no verifier needs it.) With secret weights, start `cp_prove` as in [Start the proving service](#start-the-proving-service).

Prove, on a GPU that nothing else uses. `cp_prove` wants both `HOME` and `TMPDIR` set ([the rule](#start-the-proving-service)): a login shell sets `HOME`, and the block gives `cp_prove` a new `TMPDIR` below `R`:

```sh
tmp=$(mktemp -d "${R:?set R}/tmp.XXXXXX") &&
TMPDIR="$tmp" prover/build/hc/cp_prove -isk "${ISK:?set ISK}" -public-weights examples/maverick/public-seed.bin \
  -spool "${S:?set S}" -batch 24 -in-step 6 -memory lean,tensor-rooms=8 -end-when-empty
```

A batch takes the oldest waiting requests, at most 24, and does not wait for a full one (from reading the code).

Check each response, before you hand any of them to anybody:

```sh
while read -r n range request_id end; do
  code=0
  python3 integrations/check_proof.py "${S:?set S}/out/$request_id.proof" "$R/requests/$n.req" --range "$range" --end-token "$end" \
    --commitment-id "${ID:?set ID}" --public-key-dir "${IVK:?set IVK}" --parameter-file "${K:?set K}/policy.txt" --scales-file "$K/KL.bin" \
    $LIMITS >"$R/requests/$n.check.txt" 2>&1 || code=$?
  echo "$n: exit code $code: $(tail -n 1 "$R/requests/$n.check.txt")"
done <"${R:?set R}/requests/list.txt"
```

You should see one line for each reply:

- Exit code 0: `check_proof: within limits: ...` (with an empty `LIMITS`: `check_proof: verified: ...`).
- Exit code 1: the proof was rejected, or `the proving service made no proof for this request`. In the first case stop: `cp_prove` runs no verifier on what it makes, so this check is the first that the proof meets, and, where the weights are secret, a proof that `cp_verify` rejects can hold hidden values unmasked. Give it to nobody, keep it and `$R/requests/N.check.txt` as secret as the secret key directory, and prove nothing more on that machine until the cause is known ([after a refusal or a fault](../../docs/operators-duties.md#after-a-refusal-or-a-fault)). In the second case `$S/out/` has the empty file `REQUEST_ID.failed` for it and no proof: either the reply does not meet the limits, or the prover failed; `cp_prove` does not say which. The loop goes on after such a reply.
- Exit code 2 is a fault to look into: see `$R/requests/N.check.txt`.

Nothing here deletes a file: each run of the second block leaves a directory `$R/tmp.XXXXXX` behind. All that the blocks make lies below `R` and `S`.

- The limits are lines of `policy.txt`, fixed when the commitment is made, and the commitment ID covers them: limits of your own give a commitment ID of your own. At a pass `check_proof.py` prints the policy's values (`C`, `theta`, `cap`, `s`) in its `result:` line, so each file `$R/requests/N.check.txt` holds them. They derive from the weights and become public with `policy.txt`: keep such logs to yourself until the commitment is published.
- Limits mean something only if they come from the party that vouches for the model. Limits that a provider fitted on its own replies are, to a client, the provider's claim about its service: no verifier checks how they were chosen, whether those replies were honest, or whether they resemble the reply that is checked now. `within limits` under them says only that this reply's costs on the committed model are within those limits. Take `C`, `theta` and `cap`, like the commitment ID, from the party that you trust to vouch for the model, and judge what they allow. Where the weights are public, compare them with a fit of your own on replies that you trust.
- Someone must have served the ordinary model. Replies that the integer model served have gap 0 and pass any limits: that shows that the steps work, not what the score is worth.
- As printed, these steps let the caller choose the checked tokens. Under clipped gap let your front end ask for proofs only of replies that you served yourself, over ranges that it sets ([the conditions, duty 11](../../docs/limits.md#the-conditions)). The example middleware falls short of this ([LIMITS.md](../LIMITS.md#what-the-integration-reveals)).

These steps have run at full size twice† ([LIMITS.md](../LIMITS.md#what-has-never-been-run)), each time with `-public-weights` left out of the second block, since the commitment was under a drawn seed. The demo on a commitment of your own has run only in integer serving under exact match ([the runs at full size](../LIMITS.md#what-has-never-been-run)).

## The demo on your own commitment

To serve and prove in one go, run the demo script with your own commitment and `--serve float --policy clipped-gap --limits C,THETA,CAP,S --stop-vllm-to-prove`. At full size add `--batch 24 --cp-prove-options="-in-step 6 -memory lean,tensor-rooms=8"`, the settings of [Start the proving service](#start-the-proving-service): by itself the script starts `cp_prove` without `-memory`. `--start-timeout`, whose default is 3600 seconds, holds for each start of vLLM, and with `--stop-vllm-to-prove` there are two. By itself the script passes vLLM neither `--fingerprint-mode none` nor `--max-logprobs 0`: add both at the end of the line, after `--` ([why](#start-the-server)), as the runs at full size did, in integer serving.

At full size the demo has run only with `--serve integer --policy exact` ([the runs at full size](../LIMITS.md#what-has-never-been-run): that section lists the options that it got there besides, four for vLLM among them; we have not tried whether it starts at full size without those four). In those runs vLLM was ready only after 20 to 25 minutes, which the default of `--start-timeout` leaves room for; a slower disk or fewer CPUs may need more. It has not been run at full size without `-memory`, nor with `--serve float` or under a clipped gap: so never as printed above.

`cp_prove` counts what it proves under a secret key directory against a limit, so:

- Stop your proving service before you give the script its secret key directory. `cp_prove` locks that directory, and a second one is refused at its start where one running operating system sees the same directory (read in its source, not tried through the demo). Do not lean on the lock: it sees no other machine or guest system, and no copy. Never give the script a copy or a restored backup of such a directory: a copy counts for itself, so their batches would not be counted together.
- Ctrl-C, a time-out or a failed step stops `cp_prove` with a signal. That costs the batch in progress, which stays counted, and can leave staged responses, which no verifier has seen, in the spool directory's `out/.tmp`. A second Ctrl-C kills it before it has wiped anything.
- The work directory holds prompts and replies: keep it private.

† Not on the released programs: [which kinds of result these are](../../docs/limits.md#on-other-programs).
