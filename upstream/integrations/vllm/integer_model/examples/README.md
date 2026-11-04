# Examples

Run these from `integrations/vllm/integer_model`, in the environment of
[../README.md](../README.md), "Install and quick start". The examples that `serve.sh` creates have
synthetic weights. In an ordinary run no script overwrites or removes a file of yours: every file is
made new, in a new directory whose path it prints or in one that you name (not under `sudo`:
[OPERATING.md](../OPERATING.md), "As root or under sudo").

A first run, in two shells:

```sh
examples/serve.sh small_synthetic     # wait for "Application startup complete."
examples/request.sh small_synthetic   # token_ids: [365, 287, 374, 290] (synthetic weights)
```

With GGUF files the server asks for an API key; the example's is `example`:

```sh
examples/serve.sh small_gguf
VLLM_API_KEY=example examples/request.sh small_gguf      # [93, 45, 59, 289] (synthetic weights)
```

| command | what it does | needs |
|---|---|---|
| `examples/serve.sh small_synthetic` | serves a small model on 127.0.0.1:8000, until Ctrl-C | vLLM, a GPU |
| `examples/serve.sh full_size` | the same at the shape of Llama 4 Maverick | vLLM, a GPU, 238 GB of free host memory |
| `examples/serve.sh small_gguf` | serves a small model from GGUF files, with its five small files and every weight tensor verified at start-up | vLLM, a GPU, g++ |
| `examples/serve.sh small_float_pair` | serves a small model in floating point with stock vLLM, and creates its integer counterpart, on which `score_reply.py` scores the replies | vLLM, a GPU |
| `examples/request.sh NAME TOKEN_IDS` | sends one request to the running server; both arguments can be omitted | curl |
| `VLLM_API_KEY=KEY examples/serve_gguf.sh REFERENCE_DIRECTORY MODEL_DIRECTORY NAME` | serves a model from your GGUF files ([OPERATING.md](../OPERATING.md), "Serve from GGUF files"; its options: "Settings") | vLLM, a GPU |
| `examples/score_reply.py MODEL_DIRECTORY PROMPT_IDS REPLY_IDS` | scores a reply that another engine served, in one forward pass; with `--seed SEED`, a reply sampled at temperature 1 under that seed. A score, not a verification | PyTorch |
| `examples/fit_limits.py MODEL_DIRECTORY REPLY_FILE ...` | fits the lines `C`, `theta` and `cap` of a clipped-gap policy to at least 32 replies that your own engine served, in one forward pass for each; with `--seeded`, to replies sampled at temperature 1, each under a seed of its own | PyTorch |
| `examples/try_limits.py DIRECTORY` | repeats on a CPU how the float pair's policy was fitted, on the example that `small_float_pair --directory` made in `DIRECTORY` ([OPERATING.md](../OPERATING.md), "Score a reply", prints the lines), and counts what the limits accept; from one minute to more than 18 minutes, by machine | PyTorch, transformers, safetensors |

What `score_reply.py`, `fit_limits.py` and `try_limits.py` print or write (logit gaps, costs,
replies, and the fitted `C`, `theta` and `cap`) comes from the weights and is not masked: keep it on
your own machine. `policy.txt` publishes `C`, `theta` and `cap` by design
([OPERATING.md](../OPERATING.md), "Threat model").

`python3 -m integer_model_vllm.example NAME` creates an example's files without serving them, and
prints the command that would. It needs neither vLLM nor a GPU; `small_float_pair` needs PyTorch
and safetensors, `small_gguf` needs g++.

`examples/find_or_give_root_up.sh` is not meant to be run by hand: the scripts start it where they
do not find `vllm` or a Python package that they need.

When you stop a server with Ctrl-C, vLLM 0.29.0 often logs an ERROR with a traceback
(`EngineDeadError`) while it shuts down; the exit code is still 0, and nothing is left running.
