#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Scores a reply that any engine served, by the integer model's logit gaps.

usage: examples/score_reply.py MODEL_DIRECTORY PROMPT_IDS REPLY_IDS [--policy FILE] [--device NAME]
                               [--seed SEED]
       Token ids are separated by commas. The exit code is 0 if the reply meets the policy, 1 if
       not or without a score, 2 for usage or an unusable policy.

One forward pass gives each reply token's logit gap (docs/model.md, "8 Embedding, head and output"),
0 for the greedy choice only (with --seed, the seeded choice). In the parameter file (policy.txt),
lines C, theta, cap and sh set a clipped-gap acceptance policy: each token's cost, gap + 2^sh, is
clipped to C, the mean clipped cost must be at most theta, and at most cap costs may exceed C.

The script proves nothing: a proof also needs the range conditions at every position of its window,
padding included. Nothing compares a --policy file with the model's policy.txt.

A reply token that follows a full window of T tokens is not scored by the policy in a proof: it must
be the model's first choice. This script scores it like any other token, so at that one place the
script's verdict and a proof's statement can part.

Limit: the gaps and costs that this script prints are functions of the weights. With secret weights,
publish none of them.
"""

if __name__ == "__main__":
    # Where no virtual environment is active and a package named below is not found,
    # examples/find_or_give_root_up.sh takes over: it starts this program again with
    # .venv/bin/python3 beside the package, or, as root, stops and says what works.
    import os
    import sys

    def find_or_hand_over(*names: str) -> None:
        """Asks the finders that Python has loaded, which runs none of the packages' code. Starts
        the helper only if this file, at its place in a checkout, is the program that was started,
        and only if the helper and its folder have the owner and group of this file and its folder
        and no write permission for the group that these lack, and others may write in none of them.
        """
        finders = [finder for finder in sys.meta_path if hasattr(finder, "find_spec")]
        missing = [n for n in names if not any(f.find_spec(n, None) for f in finders)]
        tried = os.environ.get("INTEGER_MODEL_ENVIRONMENT_TRIED")
        typed = getattr(sys, "orig_argv", None)  # since Python 3.10
        if not missing or tried or not typed or sys.prefix != sys.base_prefix:
            return
        this = os.path.realpath(globals().get("__file__") or os.devnull)
        folder = os.path.dirname(this)
        package, name = os.path.split(folder)
        examples = os.path.join(package, "examples")
        helper = os.path.join(examples, "find_or_give_root_up.sh")
        placed = package.endswith("/integrations/vllm/integer_model")
        placed = placed and name in ("examples", "integer_model_vllm")
        if not placed or os.path.realpath(sys.argv[0]) != this:
            return
        try:
            pairs = [(os.stat(this), os.lstat(helper)), (os.stat(folder), os.lstat(examples))]
        except OSError:
            return
        regular = pairs[0][1].st_mode & 0o170000 == 0o100000  # a file, not a link to one
        if regular and all(
            (mine.st_uid, mine.st_gid) == (its.st_uid, its.st_gid)
            and not (mine.st_mode | its.st_mode) & 0o002
            and not its.st_mode & 0o020 & ~mine.st_mode
            for mine, its in pairs
        ):
            words = ["python3", missing[0], sys.executable, sys.argv[0], *typed[1:]]
            os.execv("/bin/sh", ["/bin/sh", helper, *words])

    find_or_hand_over("integer_model")
    del find_or_hand_over

import argparse
import sys
from pathlib import Path

import integer_model


def token_ids(text: str) -> list[int]:
    return [int(token) for token in text.split(",")]


parser = argparse.ArgumentParser(
    description="Scores a reply that any engine served, by the integer model's logit gaps.",
    epilog="Limit: the gaps and costs that this script prints are functions of the weights. With "
    "secret weights, publish none of them.",
)
parser.add_argument("model_directory", type=Path)
parser.add_argument("prompt", type=token_ids)
parser.add_argument("reply", type=token_ids)
parser.add_argument(
    "--policy",
    type=Path,
    help="the parameter file with C, theta and cap (default: the model's policy.txt)",
)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--seed",
    type=int,
    help="the seed that the request specified, for a reply sampled at temperature 1: the gaps are "
    "then taken from the logits plus the seed's public noise (integer_model/noise.py), except at "
    "the last position of the window. For scoring alone: no proof is made under a seed",
)
given = parser.parse_args()

policy_file = given.policy or given.model_directory / "policy.txt"
try:
    lines = dict(line.split(maxsplit=1) for line in policy_file.read_text().splitlines())
    C, theta, cap, sh, sT = (int(lines[name]) for name in ("C", "theta", "cap", "sh", "sT"))
except (OSError, KeyError, ValueError) as error:
    parser.error(
        f"{policy_file} is not a parameter file with the lines C, theta, cap, sh, sT: {error}"
    )
# Only under sT = sh - 4 (both lines of the parameter file) is a token's cost its gap plus 2^sh.
cost_is_gap_plus_offset = sT == sh - 4
if not cost_is_gap_plus_offset:
    parser.error(
        f"{policy_file} sets a log-probability bound (line sT). This script scores exact-match "
        "and clipped-gap policies only."
    )

tokens = given.prompt + given.reply
try:
    model = integer_model.load_from_folder(given.model_directory, given.device)
    sequence = model.new_sequence()
    # Position i scores token i + 1, so the last token is not fed.
    choices, gaps = model.extend(sequence, tokens[:-1], gap_for=tokens[1:], seed=given.seed)
    model.free(sequence)
except (
    OSError,
    ValueError,
    MemoryError,
    integer_model.WindowFull,
    integer_model.RangeConditionFailed,
) as error:
    sys.exit(f"No score: {error}")

first = len(given.prompt) - 1
reply_gaps = [int(gap) for gap in gaps[first:]]
kind = "greedy" if given.seed is None else "seeded"
for token, choice, gap in zip(given.reply, choices[first:], reply_gaps):
    print(f"token {token}: gap {gap}" + (f" ({kind} choice: {int(choice)})" if gap else ""))
costs = [gap + (1 << sh) for gap in reply_gaps]
total, above = sum(min(cost, C) for cost in costs), sum(cost > C for cost in costs)
print(f"mean clipped cost: {total / len(costs)} (at most theta = {theta})")
print(f"costs above C = {C}: {above} (at most cap = {cap})")
meets = total <= theta * len(costs) and above <= cap
# Whoever holds the weights can run this script. Its verdict is the holder's own word, not a
# proof that cp_verify checks.
print(f"{'meets' if meets else 'fails'} the policy (a score, not a verification)")
sys.exit(0 if meets else 1)
