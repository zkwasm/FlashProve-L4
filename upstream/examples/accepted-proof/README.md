# A small proof that the verifier accepts

One proof for a small synthetic model, with the public files that the verifier needs to check it. You build only the verifier, which takes Linux and g++ 13 or later. No GPU is needed.

## Try it

From the repository's top directory. Here the commitment ID after `-id` comes with the example; in a real check it must not ([below](#what-the-yes-says)).

```sh
verifier/build.sh
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk \
  examples/accepted-proof/example.req examples/accepted-proof/example.y \
  examples/accepted-proof/example.proof \
  -id 79a5a2bc7a469b40df73dedf40abeb740671e8ae8077738bd3743897aeb22183 -q
```

The call takes under two seconds (one call in each of three runs, each on 24 CPUs of a shared host; not on the released programs), prints its checks line by line (`-q` leaves some out) and ends with exit code 0. Its last four lines:

```
commitment: 49 closeness rounds of the one-time files rest on another source than the bound within a third of the distance; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given
proof of work: up to 32 bits counted in 38 rounds; with it the least round has 128.00 bits, without it 96.23 bits
result-v1 status=0 claim=exact-match id=79a5a2bc7a469b40df73dedf40abeb740671e8ae8077738bd3743897aeb22183 bits=128 work=32 work-rounds=38 least-round=128.00 least-round-without-work=96.23 commitment-assumed-rounds=49 record=1 r=60 e=64 T=64 tokens=61..63 y=yes policy=exact-match C=256 theta=256 cap=0 s=16 sh=8 sT=4 F=24 above-clip=0 leaf=0 stages=key,leaf,request dw=8 build=<64 hex digits>
YES
```

The last line is the answer.

The line `proof of work:` says whether the verifier's own count of each round's strength includes proof of work, and how much. Its two figures are the weakest round's bits with the work counted and without it. Each is a single round's figure, which covers a cheating prover's single try, not repeated tries, and neither is a security level. The printed bits are the C++ programs' own accounting, not a theorem. `bits=128` in the line `result-v1` is a requirement, the bits that the call asks for; the weakest round meets it only with up to 32 bits of proof of work counted by assumption, and with no margin. The proved figure is the Lean theorem's, about 124 bits per hash query, with a floor of about 2^-101, and it is of the verifier written in Lean, not of `cp_verify`, and at full size only: not at this example's ([guide/what-is-proved.md](../../guide/what-is-proved.md)).

The line `commitment:` says on what the binding of the weights rests. Its number counts the [closeness rounds](../../docs/glossary.md#closeness-round) of the [one-time files](../../docs/glossary.md#one-time-file), the key file and the leaf file (`ivk/key.bin`, `ivk/leaf/0.bin`) that do not rest on [the bound within a third of the distance](../../docs/glossary.md#proved-lemma). "The table has a record" says that the committed tables have published [out-of-domain (OOD) answers](../../docs/glossary.md#record) (masked sums of the weights), by the parameter file's line `record 1`.

The bounds that the programs print for the truth of an accepted request and for the binding of the weights rest on the theorem on codes named in the line `commitment:`, taken as given. At the default setting they do so with no bits to spare, and lean on counted proof of work in the rounds that draw positions. They also rest on BLAKE2b, taken as a random oracle, and on an argument on paper, in a technical report that is not public and not refereed, whose conditions no complete set of parameters of any size is known to meet. Neither bound is proved of the programs. [docs/limits.md](../../docs/limits.md#what-the-theorem-assumes) gives all that these bounds rest on, and what has been checked of it and what has not.

The line `result-v1` is for programs: [verifier/docs/output.md](../../verifier/docs/output.md) gives each field. Its `build=` is a digest of the source files that `cp_verify` is built from, not of the compiler or its flags; the block above shows `<64 hex digits>` in its place.

To see the verifier say no, go on to [../rejected-proofs/](../rejected-proofs/README.md).

## What is here

| path | what it is |
|---|---|
| `config/` | the configuration: `policy.txt` (the parameter file), `KL.bin` (the quantization scales), `layout.txt` (where each array lies in a proof) |
| `ivk/` | the public key directory: `key.bin` and `leaf/0.bin` |
| `example.req` | the request: 64 tokens, then r = 60 and e = 64, as 32-bit little-endian words. The proof is about tokens r + 1 to e, counted from 0 |
| `example.y` | token 64, the one after the request: 199 |
| `example.proof` | the proof |
| `ID.txt` | the commitment ID: 64 hex digits that stand for the weights and the configuration |
| `SHA256SUMS` | digests of the other files: `(cd examples/accepted-proof && sha256sum -c SHA256SUMS)` |

## What the YES says

Tokens 0 to 60 of the request are the prompt. The proof states that tokens 61, 62 and 63 and the token after the request are each the token that the committed model ranks first after the tokens before it. This is the exact-match policy, which the verifier expects unless you name another. The line `covered:` near the end repeats which tokens the answer is about: `tokens=61..63`, and `y=yes` for the token in `example.y`. (`next-token=no` says only that the request does not ask for that token alone.)

The model is synthetic: 8 layers, 4 routed experts, 64 tokens in a request, a vocabulary of 512, weights from a generator. The commitment ID and the request came in the same package as the proof. So this `YES` shows that the verifier builds and runs on your machine and that these files fit together. It says nothing about any model that anyone serves.

In real use, two things are your part:

- Get the commitment ID before you send the prompt, from the party that vouches for the model, not through the server you are checking; best from where the commitment is published. Never take it from files that come with a proof.
- Build the request file yourself from the prompt that you sent and the reply that you received, with r and e set to what you want checked ([docs/verifying.md](../../docs/verifying.md#what-a-yes-does-not-say)). The proof is tied to the whole request file, so a changed token is rejected; but nothing checks that the file holds your prompt and your reply.

`integrations/check_proof.py` compares the commitment ID, the range and the end of the reply for you ([integrations/README.md](../../integrations/README.md)).

[docs/verifying.md](../../docs/verifying.md#what-a-yes-means) describes a real check, and [guide/known-limits.md](../../guide/known-limits.md) what a `YES` does not show.

## How the files were made

At the configuration [prover/examples/target_small_f2_roomy](../../prover/examples/target_small_f2_roomy): `config/` holds copies of its three files, and [THIRD_PARTY.md](../../THIRD_PARTY.md) lists on which of Meta's models they may depend. `example.req` holds the made-up tokens that [prover/tools/make_request.py](../../prover/tools/make_request.py) writes at places 0 to 60, then the model's next three tokens, computed on a CPU with the [integer model package](../../integrations/vllm/integer_model), with r = 60 and e = 64. On one H100, in a new directory of one's own (such as `W=$(mktemp -d) && cd "$W"`) that holds `config/` and, alone in the directory `in/`, `example.req`:

```sh
cp_commit -weights made-up-blocks -config config -M 1 -ivk ivk -isk "$PWD/isk"
mkdir home tmp
HOME="$PWD/home" TMPDIR="$PWD/tmp" cp_prove -isk "$PWD/isk" -in "$PWD/in" -out "$PWD/out"
```

The new directory matters: `cp_commit` rejects `-isk` (`refused (cp-commit-path)`), and `cp_prove` rejects `-isk` (`refused (cp-isk)`) and `-out` (`refused (cp-spool)`), if group or others can write to a directory above them that has no sticky bit, as in a checkout made under `umask 002`. `ivk` is the public key directory, and `isk` the provider's secret one, which `cp_prove` takes by its whole path only. `-M 1` asks for one stored copy of the weight table, `leaf/0.bin`. `cp_prove` refuses to start unless `HOME` and `TMPDIR` each name an existing directory by its whole path, written with letters, digits and `/ . _ -` alone; most shells leave `TMPDIR` unset. The stored files were made by calls like these on [other programs](../../docs/glossary.md#other-programs), not the released programs, so they show nothing of the released prover; `prover/build.sh` built the programs that made them, as it builds by default. The secret directory of this example is not part of the repository. We cannot show that no copy of it exists; the weights are synthetic, so nothing depends on it. `cp_commit` draws a fresh seed each time, so a run of your own gives another commitment ID, other files in `ivk/` and another proof, for the same model and the same tokens. [prover/examples/run_target_example.sh](../../prover/examples/run_target_example.sh) makes these calls for you, with another request than this example's: the configuration's own `own_choices.req` (r = 15, e = 64).
