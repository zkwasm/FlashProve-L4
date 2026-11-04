# Operating the integer model

This page is the manual for this package: the *integer model*, which is what a proof is about, and a
*plug-in* that lets vLLM 0.29.0, and no other version, load and serve it. The package makes no
proofs: `cp_prove` does, outside vLLM ([guide/serving.md](../../../guide/serving.md)). The terms:
[guide/terms.md](../../../guide/terms.md).

- **You serve a floating-point model with stock vLLM and prove the integer model's pass over prompt
  and reply** (the expected way): read "What a served reply means", "Score a reply", "Fit the limits
  of a clipped-gap policy", "What a client can do" and "Threat model".
- **You serve the integer model itself:** read everything, and the list below first.

Commands run from this directory, in the environment of [README.md](README.md), "Install and quick
start".

**Before you serve the integer model to other people:**

1. Run the check in "Check that float64 products are exact on your GPU" on the serving GPU.
2. Switch on both checks of "Verify what was loaded".
3. Pass the settings in "Settings", and none of the ones it lists under "Do not use these".
4. Keep the server on loopback, behind a proxy that allows only the routes you serve ("What a client
   can do").
5. Mount the model directory read-only.

## Full size

"Full size" is Llama 4 Maverick's shape with a *window* of 1,024 tokens for prompt and reply
together: the example configuration in `prover/examples/target_full_f2`. These figures are from runs
with Maverick's real 4-bit files on one H100 (11 CPU threads where recorded), on other sources of
the package ("What has been run").

| resource | at full size |
|---|---|
| free host memory | 238 GB, most of which it pins (locks in RAM) |
| GPU memory | 36 GB at the peak, 26.3 GB of it resident (computed) |
| routed experts, in host memory | about 223 GB in 4-bit files; 412 GB in Q8_0, 825 GB in F16 or BF16 (computed; never loaded) |
| start-up | 22 to 25 minutes with every weight tensor verified (five starts) |
| decoding | 0.60 to 0.62 s for each token after a 3-token prompt |

Before it allocates anything, the loader raises `NotEnoughMemory` unless free host memory holds the
weights plus 1/32 of their size. `"check_free_memory": false` among the `"weight_settings"` in
`integer_model.json` switches this off. The check runs once: memory that other processes take during
the minutes of loading is not seen, and the kernel may then kill this process or others.

```sh
examples/serve.sh full_size
```

This serves synthetic weights. "Application startup complete." appears after about 5 minutes. Then,
in a second shell:

```sh
examples/request.sh full_size
```

`choices[0].token_ids` should be `[42470, 62623, 33076, 45768]` (not confirmed by a run at this
size).

## The files of a model

- `policy.txt`: the parameter file, with the model's shape and every other public parameter.
- `KL.bin`: each layer's quantization scales.
- `integer_model.json`: which weights to load, and how.
- `config.json`, `tokenizer.json`, `tokenizer_config.json`: what vLLM itself reads.
- `*.gguf`: the weights as quantized blocks or 16-bit numbers, in llama.cpp's file format, GGUF.
- `wsums.txt`: the sums file, with the SHA-256 of each committed tensor's integers.
- `fold.txt`: two settings of the conversion from blocks to integers
  ([docs/model.md](../../../docs/model.md), "11 From the model's files to the integers").

Despite its name, `policy.txt` holds all of the model's public parameters, the acceptance policy
among them. The files of the first four lines form the *model directory*. `wsums.txt` and `fold.txt`
are in the *set-up directory*.

`integer_model.json` for GGUF files, the model's first file first:

```json
{
  "weights": "blocks",
  "weight_settings": {
    "blocks": "gguf",
    "setup_folder": "/path/to/the/set-up/directory",
    "shards": ["/path/to/first.gguf", "/path/to/second.gguf"],
    "verify_sums": true
  }
}
```

For synthetic weights, `"weights"` is `"synthetic"`, or `"blocks"` is `"synthetic"` without the two
paths. The loader refuses every other combination, and a file in which an entry is written twice.
Without digests or reference copies, either top entry can also be given in
`--model-loader-extra-config`, where it replaces the file's entry whole.

## Serve from GGUF files

Run the steps at small size first (needs g++):

```sh
examples/serve.sh small_gguf
```

This makes two small synthetic GGUF files (70 MB), reference copies and a model directory in a new
temporary directory, then calls `examples/serve_gguf.sh`. TMPDIR must allow programs to run, and
`g++` gets only PATH from your environment. The log has two lines with "Verified:", and "Application
startup complete." after about 2 minutes. Then, in a second shell:

```sh
VLLM_API_KEY=example examples/request.sh small_gguf
```

With the example's synthetic weights, `choices[0].token_ids` is `[93, 45, 59, 289]`.

**Llama 4 Maverick.** This repository contains no file of Meta's. [docs/quickstart.md, "4. Llama 4
Maverick: the full-size model"](../../../docs/quickstart.md#4-llama-4-maverick-the-full-size-model)
has the download (huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF, directory
UD-Q4_K_XL: 232 GB, a third party's 4-bit conversion) and the code that makes the commitment. The
steps:

1. When the commitment is made, copy `policy.txt`, `KL.bin`, `wsums.txt` and `fold.txt` into a
   private directory `K`: your reference copies.
2. Write `K/integer_model.json` as above, with absolute paths.
3. Let `M` be a new directory that only you can write to (use `mktemp -d`: `vllm_files` refuses a
   directory that group or others can write). Copy `policy.txt`, `KL.bin` and `integer_model.json`
   into `M`, and run `python3 -m integer_model_vllm.example vllm_files M ENDS` (`ENDS`: the
   publisher's `eos_token_id`, separated by commas).
4. Run `VLLM_API_KEY=KEY examples/serve_gguf.sh K M NAME`. `M` has a word-list tokenizer ("w7" is id
   7, an unknown word is id 0, no chat template), so send token ids: `VLLM_API_KEY=KEY
   examples/request.sh NAME 17,400,3`.

`examples/serve_gguf.sh` makes its working directory in TMPDIR (or /tmp if TMPDIR is unset) and
stops with status 2 if another account could change that folder or any folder above it. If it stops,
set TMPDIR to a folder of your own below folders that only you or root can change. It does not check
`K` or `M`: keep both, and the folders above them, closed to other accounts yourself.

If the start-up stops with `unknown entry "reference_directory"`, the installed package is not this
checkout's: install it again.

End the server with an interrupt (Ctrl-C) or with SIGTERM. Do not send SIGHUP to its first process
alone: in one run vLLM's engine ran on and held the GPU, until SIGTERM to the process group ended
it. SIGTERM and SIGHUP were tried on the example `small_synthetic` alone, each on an idle server,
and the interrupt on `small_gguf` too, all on other sources of the package ("What has been run"). A
terminal that closes, and a signal during the load or a request, were not tried.

## Verify what was loaded

The GGUF example switches on two checks that are off by default. A mismatch stops the start-up.

1. **The five small files against your reference copies** in `K`, outside the model directory.
   `examples/serve_gguf.sh` passes the path of `K` as `"reference_directory"`, and
   `"reference_files": 5`, in `--model-loader-extra-config`, and the loader takes the digests of the
   copies itself, in memory (seen in three runs at small size, on synthetic weights: no digest on a
   command line or in the log; in one of them the script did not pass `"reference_files": 5`; two
   ran on other sources of the package, and none of the three is known to have run on these
   sources). `K` must hold regular files of their own, not links to the files that they check, and
   must stay as it is until the log has the line below. If you have digests and no copies,
   `"sha256_file"` takes the absolute path of a file outside the model directory, at most 64 KiB,
   one JSON object, that only you can read or write (chmod 600) and that nothing removes, and
   `"sha256"` takes the digests themselves on the command line, which shows them in the process list
   and the log. Give only one of the three. Log: "Verified: these files have the given SHA-256
   digests: ...", with the files' names.
2. **Every weight tensor against `wsums.txt`**, with `"verify_sums": true` in `integer_model.json`.
   Log: "Verified: every weight tensor has the SHA-256 that wsums.txt lists for it." At full size
   this hashes 800 GB of integers on one CPU core, which makes loading several times as long.

A check that is off logs a WARNING with "not verified". Use both: check 1 covers the small files,
among them `wsums.txt`, on which check 2 relies. Not covered:

- **The files that vLLM reads:** `config.json`, `generation_config.json`, tokenizer, chat template.
  Mount the model directory read-only.
- **The prover's own checks of the GGUF files.** The prover applies further limits, so weight files
  that load here can be rejected by the prover, and then get no proof.
- **The link to the commitment.** This package compares no file with the commitment or the prover's
  copies. It checks only part of `policy.txt`: not every rule of the verifier. So it accepts some
  parameter files that prover and verifier reject.

## Settings

The scripts in `examples/` pass these, except as the rows say. The first four rows are required.

| setting | why |
|---|---|
| `--load-format integer_model` | selects this package's loader (`small_float_pair` passes neither this nor `--max-num-seqs 1`, `--no-enable-prefix-caching` or `VLLM_USE_V2_MODEL_RUNNER=1`) |
| `--max-num-seqs 1`, `--max-model-len` up to 1,024 | one sequence, within the window |
| `--no-enable-prefix-caching` | position 0 marks a new sequence |
| `--enforce-eager` | Python runs at every step |
| `--generation-config vllm`, `--served-model-name` | no sampling defaults from the model directory; no path in replies (`/v1/models` still shows the path: close it at the proxy) |
| `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_NO_USAGE_STATS=1` | vLLM's V2 model runner; no usage statistics sent |
| `--host 127.0.0.1`, `GLOO_SOCKET_IFNAME=lo` | HTTP and PyTorch's Gloo sockets on loopback |
| `PYTHONSAFEPATH=1` | Python does not put the caller's directory on its path at start-up |
| a new working directory, `TMPDIR`, `HOME` | nothing of the caller's directory is at hand, and caches stay in a folder of the server's own (`examples/serve.sh` sets `HOME` alone) |
| `PYTHONNOUSERSITE=1` | `examples/serve.sh` only: Python reads no user site under `HOME` |
| `--fingerprint-mode none`, `--max-logprobs 0` | `examples/serve_gguf.sh` only: narrow leaks 1 and 3 of "What a client can do" |

The command that `python3 -m integer_model_vllm.example` prints after "To serve it:" does not set
`HOME` or `PYTHONNOUSERSITE`: set them yourself, or use `examples/serve.sh`. The scripts pass on to
vLLM everything in your environment that they neither set nor refuse, and under `sudo` whatever the
sudo configuration keeps of it. Variables such as `LD_PRELOAD`, `PYTHONHOME`, `XDG_*` or `VLLM_*`
can change what runs or where files go, so start from a clean environment.

`examples/serve_gguf.sh` takes a fourth argument, the value of `--max-model-len` (1,024 if omitted),
then pairs `--middleware MODULE.CLASS` (run once, at small size, with a request that asked for no
proof, on other sources of the package), and rejects every other option. The module must be
installed or reached through an absolute entry in `PYTHONPATH` (for this repository's middleware,
the absolute path of `integrations/`).

Do not use these:

| setting | effect |
|---|---|
| `--trust-remote-code` | the server runs code that a file in the model directory points to |
| `VLLM_SERVER_DEV_MODE` | refused by the scripts, with exit status 2; adds routes that skip the API key, show the server's settings (the digests, if given as `"sha256"`) and replace weights |
| `VLLM_MODEL_REDIRECT_PATH` | refused by the scripts, with exit status 2; can send the server to another model directory |
| `VLLM_LOGGING_LEVEL` | refused by the scripts, with exit status 2; at `WARNING`, hides the "Verified" lines; at `ERROR`, the warnings too |
| `VLLM_CONFIGURE_LOGGING`, `VLLM_LOGGING_CONFIG_PATH` | by their names, not tried: may keep the "Verified" lines out of the log; the scripts do not refuse them |
| a second `--model-loader-extra-config` | replaces the first, with its digests, its `"sha256_file"` or its `"reference_directory"` |
| the model directory as working directory | a file there can be imported before any check |
| empty or relative `PYTHONPATH` entries | refused by the scripts, with exit status 2; otherwise a file that such an entry reaches can be imported before any check, despite `PYTHONSAFEPATH=1` |
| `--prefill-context-parallel-size` | unsupported, and not checked |

## As root or under sudo

Do not run these scripts under `sudo` or as root. They need no root rights, and as root they run the
checkout's and its environment's code with root's rights. Run them as the account that owns the
checkout. Under `sudo` a script stops with status 2 and one line that says why, or (the Python
scripts, in some cases, by reading) with Python's own error for a missing package and status 1, or,
where it finds what it needs, goes on as root without saying so
(`python3 -m integer_model_vllm.example` needs nothing but for `small_float_pair`, so it goes on).
This was tried with stand-ins for vLLM, without a GPU. Even when it stops, root has run files that
the checkout's owner can change.

**If root it must be,** as in a container, make a checkout and an environment of root's own, below
directories that only root can write. Otherwise give root the environment's `bin` and no more of
your `PATH`: `sudo env "PATH=$VIRTUAL_ENV/bin:SAFE" COMMAND`, with system directories alone for SAFE
(for example `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`).

Whoever can write in the checkout, or replace a directory above it, then acts as root, so let nobody
else write there, and keep files such as `torch.py` out of the working directory. Files that root
makes stay root's, in /tmp and in the checkout. The Python example scripts overwrite nothing, but
they follow a link in place of the directory that you give them, or of one above it. Where a
script's message advises a `pip install`, run it as the environment's owner: under `sudo` its build
steps run as root. More: [LIMITS.md](../../LIMITS.md#under-sudo) and [operators'
duties](../../../docs/operators-duties.md#under-sudo).

Start `examples/serve.sh` and `examples/serve_gguf.sh` by a path with a slash; under a bare name
they refuse to start.

## What a client can do

**Put an allow-list of routes at a proxy,** with only what you serve, such as `/v1/completions`.
`VLLM_API_KEY` does not guard every route of vLLM: `/metrics` (all clients' traffic), `/invocations`
and `/generative_scoring` answer without it, the last two from the model (read in vLLM 0.29.0's
source, not run). A filter on request fields does not close them, and a local account reaches the
server's own port past the proxy. The proof middleware ([../README.md](../README.md)) passes a
request that asks for no proof unchanged.

**Forced tokens.** Sampling settings have no effect, but these force tokens that the model did not
choose, and such a reply gets no proof: `allowed_token_ids`, `bad_words`, `min_tokens`, structured
outputs, a thinking budget, all but the first sequence of a beam search, a `logit_bias` of about 1e9
or more on the route /inference/v1/generate, logits processors on the server, and a
`repetition_penalty` so small that float32 logits overflow (not tried with vLLM). The server logs
one warning per sequence, but can miss the last token or a beam search.

**Token ids outside the vocabulary.** The plug-in starts only with a tokenizer whose `max_token_id`
is below the vocabulary size (not with `--skip-tokenizer-init` or `--tokens-only`), and stops the
server if such an id still reaches it. This is tested with ids in plain prompts only; ids in
`logit_bias` or `allowed_token_ids` are not covered.

**Unexpected input that reaches the plug-in stops the engine**, with exit code 0: watch the log for
"EngineCore encountered a fatal error."

**A client can learn about the weights.** This matters where the weights must stay hidden. Leak 1
and the fingerprint of leak 3 are vLLM's own; leak 2, and how reply times depend on the weights, are
this plug-in's. The list may be incomplete, and no leak is bounded:

1. *Many answers per request.* With `prompt_logprobs`, or `echo` with `logprobs`, one request
   returns the greedy choice after every prefix of its prompt. `--max-logprobs 0` only narrows this.
2. *Where a value first leaves its range.* When a range condition fails
   ([docs/model.md](../../../docs/model.md), "9 Range conditions"), the reply ends with an
   end-of-sequence token that the model did not choose, and gets no proof (with `ignore_eos` or
   `min_tokens` it runs on, and its further tokens mean nothing). One request tests every prefix of
   its prompt at once, and tries are unlimited. The committing program's worst-case check of the
   weights is meant to exclude such prompts; that it does is not proved, and a commitment does not
   show that it passed.
3. *Timing and fingerprint.* Reply times depend on the weights and on other clients' requests.
   `system_fingerprint` lets a client test a guess of the model directory's path and settings,
   unless `--fingerprint-mode none` empties it.

## Threat model

Anyone who can write the server's files, links on their paths, command line or environment controls
the server and can learn the weights. The two checks catch mix-ups, not attackers. So:

- **Keep the model, weight and set-up directories closed to other accounts.** Someone who can only
  alter the weight files still learns one bit of their choosing about the weights from each start-up
  (whether it is rejected), and more from the error, which lists every tensor that differs from
  `wsums.txt`. A link to another model's files shows which tensors the two share.
- **Keep no other model's GGUF files where the server's account can read them.** The loader follows
  the paths in `integer_model.json` with the server's rights, so it can serve files that someone who
  can only alter the weight files cannot read.
- **Keep `wsums.txt`, its digest and `fold.txt` as secret as the weights**
  ([docs/operators-duties.md](../../../docs/operators-duties.md#commands), step 8). With them anyone
  can test a guess at the weights, or whether a candidate model is the served one. Use
  `"reference_directory"` ("Verify what was loaded").
- **Keep the server's working directory as secret as the weights.** `examples/serve_gguf.sh` makes a
  new one (mode 700) at each start, and it stays after the server has ended, with vLLM's caches,
  which may depend on the weights. The script prints its path on the line that begins "The server's
  working directory, TMPDIR and HOME:". Once the server has ended, you can remove that one directory
  by that exact path, with no wildcard and no guessed name.
- **Keep what this package computes on your own machine.** Choices, logit gaps, costs, and with
  `record=` every layer's arrays and the logits are derived from the weights and are not masked. The
  tokens that you serve, and `C`, `theta` and `cap` in `policy.txt`, are handed out by design. What
  is hidden, and what is not: [guide/weight-hiding.md](../../../guide/weight-hiding.md).

## What a served reply means

**A reply that the plug-in served.** By design, each token is the integer model's greedy choice on
the loaded files, if no setting forces tokens, the log has no error or warning for the reply, and
the GPU passes the check in "Check that float64 products are exact on your GPU". Only a proof,
checked against the commitment ID, shows which model that was.

A correct reply can still get no proof: the prover needs the range conditions at every position of
its window, padding included, and the server evaluates fewer.

**A reply that another engine served, proved under a clipped-gap policy.** Such a policy also
accepts replies near the greedy one, within thresholds that the provider sets. A proof says only
that the tokens meet the policy on the committed model: another model's reply can get one too
("Score a reply" gives counts).

**What Lean's theorems cover.**

- Lean's soundness theorem is about the Lean verifier, not `cp_verify`, and holds at exact match
  only, in the random-oracle model, for public files in scope (one proof format, at full size) fixed
  before the hash function is drawn ([lean/THEOREM.md](../../../lean/THEOREM.md);
  [guide/what-is-proved.md](../../../guide/what-is-proved.md)). `cp_verify` is the C++ verifier that
  users run.
- For any other policy Lean proves, again of the Lean verifier and not of `cp_verify`, index binding
  ([lean/TERMS.md](../../../lean/TERMS.md)) for "the output meets the policy". By itself that bars
  nothing against a committer who commits to weights under which every output meets the policy.
- Some policies bar nothing. By a lemma on the policy's definition, not about a verifier, at two of
  the library's sets of public files, a policy with `theta` = `C` and `cap` at least n is met by
  every output with n tokens in the window on which no range condition fails, whatever the weights
  (`lean/Johnson/Goal/Files/MeetsPolicyAtFiles.lean`). At some other settings of the lines `MF` and
  `KF` more policies are so (`lean/Policy/Front/MeetsPolicyIdleAtScales.lean`). `C` 25856, `theta`
  512 and `cap` 1, the numbers of `examples/full-size-clipped-gap`, are no such policy, which says
  nothing of a given model: under weights that are all 0 every such output meets them.
- Exact match is a premise on `policy.txt`, not a test inside the Lean verifier. So read the lines
  `C`, `theta` and `cap` yourself. A token's cost is 256 plus its gap, so 256, 256 and 0 are exact
  match ([docs/limits.md](../../../docs/limits.md#within-limits)).

## Score a reply

`examples/score_reply.py` scores a reply that another engine served, in one forward pass. It prints
each reply token's logit gap, 0 for a greedy choice, and ends with "meets the policy" (exit code 0)
or "fails the policy" (exit code 1), for the clipped-gap policy that the lines `C`, `theta` and
`cap` of `policy.txt` set: each token's cost is its gap plus a constant; the mean of the costs, each
counted as at most `C`, must be at most `theta`, and at most `cap` costs may exceed `C`. It makes no
proof.

Where the script and a proof differ:

- A reply token after a full window must be the model's first choice in a proof. The script scores
  it like any other token.
- With `theta` equal to `C` and `cap` above 0 the mean bounds nothing. `cp_commit`, which makes the
  commitment, and `cp_prove` refuse such a parameter file (read in the code, not run); the script
  says "meets" wherever at most `cap` costs exceed `C`.

To try both halves, serve a small synthetic model in floating point with stock vLLM:

```sh
examples/serve.sh small_float_pair
```

After "Application startup complete.", in a second shell:

```sh
examples/request.sh small_float_pair
```

Then score the reply that vLLM served, with its token ids, separated by commas, in place of
`REPLY_IDS`:

```sh
directory=$(python3 -m integer_model_vllm.example small_float_pair --quiet)
examples/score_reply.py "$directory/model" 17,400,3 REPLY_IDS
```

To see how the example's policy was fitted and what it accepts, run these in one shell. They need
PyTorch, transformers and safetensors, and no GPU (`--device cuda` of `try_limits.py` uses one). The
script, `try_limits.py`, took about 1 to 6 minutes on a CPU (on other sources of the package, the
script's among them; on one machine with 16 threads it did not end within 18 minutes, and the cause
is not known):

```sh
D=$(mktemp -d)/pair && echo "D=$D" &&
python3 -m integer_model_vllm.example small_float_pair --directory "$D" &&
examples/try_limits.py "$D"
```

`prover/examples/target_small_f2_gap/README.md` explains the output. A nearby model that applies the
router's score twice met these limits with 47 of its 64 greedy replies, against 63 of 64 of the
example's own model (in bfloat16), though only 4 of its replies were that model's token for token.
These are scores, not proofs. So on this one small synthetic model a reply that meets limits says
little about which model made it, and one that misses them says much.

**Seeded replies: a score, not a verification.** With `--seed SEED`, `examples/score_reply.py`
scores a reply that a server sampled at temperature 1 under that seed, by adding the seed's public
noise (docs/model.md, "12 The noise of a seeded score") to the integer model's logits. No program of
this repository proves or verifies this score, and no Lean theorem covers it. Keep limits fitted to
it out of a parameter file that you commit to, and keep them as secret as the weights, unless you
mean to publish them. The noise aims to match vLLM 0.29.0's V2 model runner, without top-k, top-p or
min-p; a server whose noise follows another rule fails as a wrong seed does. If prompt and reply
together have one token more than the window, the last token is scored on the logits alone, although
its line says "seeded choice".

**On Maverick's real weights.** The score refused nearly all replies of a smaller sibling model and
every reply set after another prompt. It did not tell Maverick's own replies under other sampling
settings, or a reply with one token replaced, from honest ones: these often passed. The integer
model, on Maverick's public 4-bit weights, not checked against their sums, scored stored replies
that vLLM 0.29.0 had served, of up to 500 tokens, to 96 prompts and 136 further ones of 524
WikiText-2 tokens. Greedy limits were fitted on 48 greedy replies of one engine (16-bit or FP8),
seeded limits on 48 replies that the 16-bit engine had sampled at temperature 1, and each was tried
on other prompts at the default threshold, the worst fitted reply. A pair of counts gives 16-bit and
FP8 limits unless its cell says otherwise. There were no proofs, each count is from one run, and no
one tried to deceive. The replies were scored on other sources of the package ("What has been run");
by a comparison of the code, function by function, those sources compute the same forward pass,
costs and fit as these sources. As the runs were set up, the runs' table of scales is the one that
`examples/maverick/run.sh` fits, and the lines of their parameter file that state the model have the
same names and values as the fitted ones. The runs' own output does not name the files that they
read.

| replies scored | pass greedy limits | pass seeded limits |
|---|---|---|
| honest (greedy; under their own seed) | 91 to 95 of 96; 134 to 135 of 136 | 94 of 96; 134 and 133 of 136 (per fit) |
| Llama 4 Scout | 4 and 2 of 96 | 0 of 96 |
| set after another prompt | 0 of 96 | |
| under the next seed | | 1 of 96; 1 and 1 of 136 (per fit) |
| Maverick at temperature 0.7 | 86 of 96 (16-bit) | 91 of 96 |
| Maverick at top-p 0.9 | 85 of 96 (16-bit) | 88 of 96 |
| Maverick at temperature 1 | 83 and 40 of 136 (two 16-bit fits) | as honest |
| Maverick, greedy | as honest | 20 of 96 (16-bit engine), 9 of 96 (FP8 engine) |
| one token replaced by the model's second choice | 28 and 24 of 32 | 30 of 32 |
| one token replaced at random | 27 and 21 of 32 | 27 of 32 |

- *A tighter threshold costs honest replies.* With the third-worst fitted reply as threshold, none
  of Scout's replies passes and 30 of 96 at temperature 0.7 do (53 under the FP8 limits), but the
  16-bit limits then refuse 5, 11 and 17 of 96 honest replies of their own engine and of two
  servings of one FP8 checkpoint.
- *Much hangs on the fit.* Scout's counts and the seeded counts of 96 add two sets of limits, where
  a provider publishes one; with so few replies Scout's true rate could be about 1 to 10 in 100. The
  looseness hangs on the one worst fitted reply: of two seeded sets fitted on 68 replies, one let 58
  and 48 of 96 greedy replies (16-bit and FP8 engine), 9 of 96 under the next seed (and 3 of 68 to
  further prompts), 1 of 96 of Scout's and all 96 at temperature 0.7 and at top-p 0.9 pass, and the
  other refused 8 of 96 honest replies. With the looser set's fourth-worst fitted reply as
  threshold, the greedy replies fall to 11 and 4 of 96 (16-bit; FP8), the next seed's to 0 of 96 and
  1 of 68, and Scout's to 0 of 96, but 79 and 77 of 96 at temperature 0.7 and with top-p 0.9 still
  pass.
- *Exact match.* Under it none of these replies, honest or not, would get a proof: it is for replies
  that the integer model itself served.
- *Not scored,* because the two runs behind these counts ended at their time limits: seven of the
  eight sets of replies with 4 or 16 tokens replaced; seeded replies with 4 or 16 tokens replaced;
  sampled replies of the FP8 engine; the same prompts under other seeds.

So the seeded score told a wrong seed and the one smaller sibling apart. It does not show at which
settings a reply was sampled. A seed that the server picks after it has seen the reply shows less
than one that the client named: the server can try seeds.

## Fit the limits of a clipped-gap policy

A provider that serves a floating-point model, and proves the integer model's pass over prompt and
reply, chooses the lines `C`, `theta` and `cap` of `policy.txt`: wide enough for its own replies,
and no wider. This repository has no such limits for a real model. `examples/fit_limits.py` fits
them to replies that you served:

```sh
examples/fit_limits.py MODEL_DIRECTORY replies/*.json --device cuda
```

`MODEL_DIRECTORY` is the one that `score_reply.py` takes. Each file holds one JSON object with the
lists `"prompt_token_ids"` and `"reply_token_ids"`. Neither script says which weights it loaded, and
limits fitted on synthetic weights look like any others, so before you publish limits, check that
`integer_model.json` names your GGUF files. A `C` wider than your own replies need lets a client who
chooses the checked tokens measure a wider band of scores.

**The model directory for Llama 4 Maverick.** Run these lines after the stage `fit` of
[examples/maverick/run.sh](../../../examples/maverick/run.sh), at the top of the repository, with
`FIT` set to the full path of its directory `fit.N`. They read the path of the weights directory
(the GGUF files with `wsums.txt` and `fold.txt`) from `fit.N/weights-directory.txt`, and stop with
"No *.gguf file in the weights directory: …" if that directory holds none.

```sh
D=$(mktemp -d) && cp "$FIT/config/policy.txt" "$FIT/config/KL.bin" "$D" &&
python3 -I -c 'import glob, json, sys
settings = {"blocks": "gguf", "setup_folder": sys.argv[1], "verify_sums": True}
settings["shards"] = sorted(glob.glob(glob.escape(sys.argv[1]) + "/*.gguf"))
if not settings["shards"]:
    sys.exit("No *.gguf file in the weights directory: " + repr(sys.argv[1]))
print(json.dumps({"weights": "blocks", "weight_settings": settings}))' \
  "$(cat "$FIT/weights-directory.txt")" >"$D/integer_model.json"
```

`$D` is then the model directory. The loader warns that the weights' paths lead out of it: that
warning is expected here.

**What it prints.** Standard output gets the three lines, `C`, `theta` and `cap`, each with its
number, and nothing else. Put them in place of the three in `policy.txt` before you make the
commitment: the commitment ID covers that file, so other limits need another commitment. Two
commitments to the same weights under different configuration files, for instance under two
acceptance policies, tell more of the weights together than either does alone
([why](../../../docs/limits.md#a-further-commitment)). Standard error says how many replies meet the
limits, and how many of each half meet limits fitted on the other half alone: those counts are the
script's only guide to how your further replies will fare.

**Which replies to give.** At least 32 greedy replies (temperature 0) of the engine and settings
that will serve, to prompts like those that you expect, each whole: up to your own length limit, or
with the token that ended it as its last id. With `--end-token IDS` the script leaves out a reply
that has such a token earlier; it cannot tell whether a reply was cut short.

**Options.** `--share 90` sets `theta` so that 90 % of the replies meet it: tighter limits reject
more of your own replies and more of others'. `--C N` sets the clip. `--swapped` also scores each
reply after the next reply's prompt; a low count there says nothing of a reply with a few tokens
changed. `--seeded` fits limits for the seeded score: give replies sampled at temperature 1, without
top-k, top-p or min-p, each with its integer seed under the key `"seed"`. `--next-seed` also scores
each reply under its seed + 1.

**Cost at full size.** In one of the runs behind the table under "Full size", on other sources of
the package, loading took 4 minutes without verifying and 21 with, a pass over a whole window took
11 s, and host memory was 234 GB. So 100 replies take about 23 minutes, or 40 with verification, as
the lines above set it (computed from these figures).

**What the limits do not show.**

- They are a function of the weights and of the replies, and they are published with the parameter
  file. Together they hold at most 74 bits, at a window of 1,024 tokens; what these bits reveal
  about secret weights is not known. Under a score policy the answers themselves tell something
  about the scores: item 2 of
  [docs/what-is-not-hidden.md](../../../docs/what-is-not-hidden.md#served-or-failed).
- `theta` bounds a mean, not each token.
- To a client, limits that a provider fitted on its own replies are the provider's claim: no
  verifier checks how they were chosen or whether the replies were honest. A pass says only that
  this reply's costs on the committed model lie within those three numbers. Take `C`, `theta` and
  `cap`, like the commitment ID, from the party you trust to vouch for the model.

## Check that float64 products are exact on your GPU

PyTorch has no int64 matrix product on CUDA ([issue
44428](https://github.com/pytorch/pytorch/issues/44428)), so the model multiplies in float64. That
is exact only if the library forms ordinary inner products in true float64. The package cannot see
this, and environment variables can make cuBLAS emulate float64 ([NVIDIA Technical
Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/);
[cuBLAS manual](https://docs.nvidia.com/cuda/cublas/), "Floating Point Emulation"), whose results
then need not comply with IEEE 754.

```sh
INTEGER_MODEL_TEST_DEVICE=cuda pytest -rs tests/test_arithmetic_edges.py
```

Every test should pass, and none be skipped. If one fails or is skipped, do not serve from that GPU:
the served tokens may not be the committed model's. Four tests are skipped where less than 8 GB of
GPU memory are free, so run this before a server takes the GPU, in the server's environment, and
again whenever GPU, PyTorch, CUDA or environment change.

## What has been run

The package was compared with the prover and with Lean's definition of the model, at the sizes
below. No difference was found in what was compared. These are tests of agreement, not proofs, and
none says anything about soundness or weight hiding.

*These sources* are the files of this repository. *Other sources of the package* are sources of this
package that differ from them; they are not part of the repository. What a run on them found tells
nothing certain about this package. *Other programs* (test builds and test programs among them) are
programs that are not the released programs: [which those
are](../../../docs/limits.md#what-was-run-on-these-programs), and [what these pages tell of the
others](../../../docs/limits.md#on-other-programs).

In the table below, the results of the first two rows were got on other programs and on other
sources of the package. In the third row, the comparison with the recorded values is the test row
`integer_model.cpu` of [SUITE.md](../../../SUITE.md), which passes on these sources, and the
comparison with Lean's definition ran on these sources too. The results of the fourth row were got
on other sources of the package (their `noise.py` has the same code as this repository's). The test
counts are in [tests/README.md](tests/README.md).

| what was compared | size, weights | result |
|---|---|---|
| replies served from the package, proved under exact match | full, public 4-bit Maverick | 16 of 16 greedy replies to made-up prompts, in each of four runs (three of them told in [integrations/LIMITS.md](../../LIMITS.md)) |
| `cp_prove` against the package's forecast, clipped gap | full, public 4-bit Maverick | 178 of 178 as forecast (152 proofs, 26 none), one run; among them, by that job's own output, 127 of 128 greedy replies that stock vLLM served from the 16-bit checkpoint got a proof; by scores alone, these loose limits also pass 7 of 96 greedy replies of a smaller model of the family ([integrations/LIMITS.md](../../LIMITS.md#what-has-never-been-run)) |
| value by value, against a separate C++ implementation (tests/README.md, "The recorded values") and Lean's definition of the model (tests/lean_vectors/README.md) | small sizes; two layers at Maverick's sizes | all equal; the prover prints no intermediate value, so it took no part, and nothing was compared at full size |
| the seeded noise against vLLM's sampler | 2-layer model with random weights, one H100 | 1,984 of 1,984 positions under the V2 model runner, 0 of 960 under the other; on three real models each served token led the twenty tokens that vLLM lists, under the noise of its request's seed, at about 500 positions each; in part of these runs the served token was not the likeliest at 116, 141 and 140 positions: all of these fit under the request's seed, and 13, 10 and 16 under the next seed; that is necessary, not enough, as the other tokens were not compared |

- *Not tried:* with the prover, a request at which a range condition fails; a real tokenizer (send
  prompts as token ids); with this package, a real model's files with 8- or 16-bit tensors; at full
  size, ordinary serving under a clipped gap through the proof middleware (integrations/LIMITS.md);
  for the noise, other GPUs, versions or temperatures, top-k, top-p, speculative decoding and noise
  in float64 (vLLM's `use_fp64_gumbel`).
- *What bounds all of it.* What was compared with the prover is proof or no proof, and greedy
  choices. `cp_prove`'s answer without a proof is an empty file, whatever the cause. On synthetic
  weights an agreement of greedy choices, however long, says little about attention, routing or the
  scales: there the first choice is all but fixed by the position's own token. All sides were
  written from one document, the package and the programs by the same authors, so their agreement is
  no independent check, and an error in that document passes.
- *What a disagreement would mean* (by reasoning, not by a test). The package takes no part in
  making or checking a proof, so a fault in it cannot by itself make a verifier accept a false
  proof. But served replies would lose their proofs, fitted limits would be off, and the scored
  counts under "Score a reply" would not carry over to proofs. A fault in the programs would mean
  that their proofs are about another function than these pages and Lean's theorems describe.
