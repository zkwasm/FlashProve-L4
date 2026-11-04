# Altered files that the verifier rejects

No altered files are stored here. One script alters the files of [../accepted-proof/](../accepted-proof/README.md), or of [../accepted-proof-clipped-gap/](../accepted-proof-clipped-gap/README.md), one at a time, as the rows of a small table say, and runs the verifier on each altered set. No GPU is needed.

## Try it

From the repository's top directory, after `verifier/build.sh`:

```sh
W=$(mktemp -d) && COPIES="$W/altered-copies" && echo "COPIES=$COPIES"
python3 examples/rejected-proofs/try_altered_copies.py bin/cp_verify "${COPIES:?first run the line with mktemp}"
```

The second argument names a new directory, so run both lines for every run. The script creates it, writes a few copies of the example's files there (about 16 MB) and leaves them for you to look at; its first line says where. If the directory exists, or lies below a directory in which another account could replace it, the script stops before it runs anything. It writes nowhere else and deletes nothing.

## What you see

After less than a minute:

```
The altered copies are written to /tmp/tmp.XXXXXXXXXX/altered-copies
the files as they are                  exit code 0   YES
    proof of work: up to 32 bits counted in 38 rounds; with it the least round has 128.00 bits, without it 96.23 bits
    by the verifier's own count, the weakest round has 128.00 bits with proof of work counted and the weakest without it has 96.23; no figure here is a security level
one bit of the proof changed           exit code 1   NO: proof-refused
the proof's last byte cut off          exit code 1   NO: proof-refused
one byte added to the proof            exit code 1   NO: proof-refused
one bit of ivk/key.bin changed         exit code 1   NO: key-refused
one bit of ivk/leaf/0.bin changed      exit code 1   NO: entry-refused
a prompt token changed                 exit code 1   NO: proof-refused
a checked reply token changed          exit code 1   NO: proof-refused
another token after the request        exit code 1   NO: reply-differs
another scale in KL.bin                exit code 1   NO: model-statement
another commitment ID                  exit code 1   NO: model-statement
10 altered sets: all rejected
```

The first call checks the files as they are; the two lines beneath it repeat what the verifier printed on proof of work, with its two figures ([../accepted-proof/](../accepted-proof/README.md) explains them). Each is a single round's figure, and neither is a security level. Each later call differs from the first in one file or value. A rejection is exit code 1 and a last line `NO: ` with the name of the check that failed; [verifier/src/vf_rejection_names.txt](../../verifier/src/vf_rejection_names.txt) lists every name, and [verifier/README.md](../../verifier/README.md#last-line-and-exit-status) explains the last line and the exit code. The script ends with exit code 0 if the first call passed and every other call was rejected by the check that the output above names. Otherwise the line of that call says what went wrong, and the exit code is 3 if the verifier accepted an altered set; please report that as described in [SECURITY.md](../../SECURITY.md).

A longer table, [every_kind_of_item.tsv](every_kind_of_item.tsv), has 79 altered copies, meant to cover each sort of item in the files (a tag, a root, an element of a message, an answer, a nonce, a salt, a digest and so on), not every field of every round. A few rows are for `example.req` and `example.y`. Each row has a few words on what stands at its place. Give the table to the script as a third argument; that takes a few minutes, and the copies that it writes take about 183 MB. Its rows name places by byte numbers or as the file's end, so they fit the stored files alone. The names of rounds and items in its descriptions, such as `SEND seed` or `CS`, are those of docs/format/.

## Add a row

[altered_copies.tsv](altered_copies.tsv) has one row for each altered set: which file, where, what change, and the last line that the verifier must print. The comment lines at its top explain the columns. Add a row and run the script again, or give the script a table of your own as a third argument. Every proof has other bytes, and most of its fields lie at other places, so name a place by a word (`middle`, `token 0`, a field's name) where you can.

## The other small example

The script takes one option after its other arguments.

`--example accepted-proof-clipped-gap` alters the files of [../accepted-proof-clipped-gap/](../accepted-proof-clipped-gap/README.md) instead, as the rows of [altered_copies_clipped_gap.tsv](altered_copies_clipped_gap.tsv) say. Every call then names that example's policy with `-policy`, and the first call must end with `WITHIN LIMITS:` and the policy.

The test `verifier.rejected_proofs` in [SUITE.md](../../SUITE.md) runs the script on both examples.

## What this shows

That the verifier notices these changes to these files. It is a demonstration, not a test of soundness: a cheating prover would not flip bits in an honest proof. What is known about soundness is in [docs/limits.md](../../docs/limits.md).
