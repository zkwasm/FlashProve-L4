#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Fits the limits of a clipped-gap acceptance policy to replies that your own engine served.

usage: examples/fit_limits.py MODEL_DIRECTORY REPLY_FILE ...
       (options: --share, --C, --end-token, --swapped, --seeded, --next-seed, --device)
       The exit code is 0 if the three lines were printed, 1 if not, 2 for usage.

Each REPLY_FILE contains one JSON object with the lists "prompt_token_ids" and "reply_token_ids";
other keys are ignored. Give at least 32 greedy replies from the engine that you will serve with, to
prompts like those that you expect. (A floating-point engine computes nearly, not exactly, what the
integer model computes.)

One forward pass for each reply gives each reply token's cost, its logit gap + 2^sh, as in
integrations/vllm/integer_model/examples/score_reply.py. Each pass covers the window of a proof: the
prompt, the reply, then token 0 up to T tokens (whole_reply() in integrations/spool.py). The
parameter file, policy.txt, sets sh and lt; T = 2^lt. A reply whose window fails a range condition
is left out and reported on standard error: the prover makes no proof for it.

A reply meets the limits if its clipped mean, the mean of its costs with each counted as at most C,
is at most theta, and at most cap of its costs exceed C. Standard output gets three lines for the
parameter file and nothing else:

    C      the 99.9th percentile of all N costs (the ceil(0.999 N)-th smallest). If that is 2^sh,
           the cost of the integer model's own choice: the smallest cost above 2^sh, if any. Or --C.
    theta  the smallest integer >= the clipped means of --share percent of replies (default: all)
    cap    the largest number of costs above C in any reply, regardless of --share

Source: the 99.9th percentile for C is from the DiFR paper (https://arxiv.org/abs/2511.20621),
section 5.2, which clips scores at a percentile of the honest training split. Like its section 4.1,
the script takes its threshold from a calibration set of honest runs.

Standard error gets the rest: what was read, how many replies meet the limits, the same rule fitted
on each half and tested on the other, and the limits as options, on lines that begin
"check_proof.py: ", "end_to_end.py: " and "cp_verify: ".

No limits are printed if theta would equal C with cap above 0: the mean would then bound nothing.

With --seeded the replies are not greedy: each was sampled at temperature 1 under a seed, and its
REPLY_FILE also has that seed, an integer, under the key "seed". A token's gap is then taken from
the logits plus the seed's public noise (integrations/vllm/integer_model/integer_model/noise.py),
as with score_reply.py's --seed. Such limits are for scoring alone: no program proves or verifies
a seeded policy, so the three lines of options are not printed. Do not commit to a parameter file
that holds them: the prover would hold greedy gaps to them. Without --seeded a key "seed" is
ignored, and the script says so.

Limit: the three numbers are a function of the weights and of these replies, and they are public
(with --seeded: nothing publishes them; keep them as secret as the weights, unless you mean to
publish them).
Limit: a reply that meets them need not come from your engine, or from greedy decoding (with
--seeded: from sampling at temperature 1).
Limit: they suit replies like those given. Other lengths, languages or prompts can need others.
Limit: the numbers are your own statement about your engine. No proof shows how they were fitted or
on which replies: a client that accepts a pass within them trusts you for them as it trusts you for
the commitment ID, and the pass says only that the reply's costs on the committed model lie within
them.
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
import json
import sys
from pathlib import Path
from typing import NoReturn

import integer_model

FEWEST = 32  # each half then has 16 replies


def token_ids(text: str) -> list[int]:
    return [int(token) for token in text.split(",")]


def say(line: str) -> None:
    print(line, file=sys.stderr)


def stop(reason: str) -> NoReturn:
    sys.exit(f"No limits: {reason}")


def clipped_mean(costs: list[int], C: int) -> int:
    """Rounded up."""
    return -(-sum(min(cost, C) for cost in costs) // len(costs))


def fit(replies: list[list[int]], own_choice: int, share: int, C: int | None) -> tuple:
    """Return (C, theta, cap), fitted to the given replies' costs; see the module's docstring."""
    if C is None:
        ordered = sorted(cost for costs in replies for cost in costs)
        C = ordered[-(-999 * len(ordered) // 1000) - 1]
        if C == own_choice:
            C = min((cost for cost in ordered if cost > own_choice), default=own_choice)
    means = sorted(clipped_mean(costs, C) for costs in replies)
    theta = means[-(-share * len(means) // 100) - 1]
    return C, theta, max(sum(cost > C for cost in costs) for costs in replies)


def meets(costs: list[int] | None, C: int, theta: int, cap: int) -> bool:
    if costs is None:
        return False
    return clipped_mean(costs, C) <= theta and sum(cost > C for cost in costs) <= cap


parser = argparse.ArgumentParser(
    description="Fits the limits of a clipped-gap acceptance policy to replies that you served.",
    epilog="Limit: the three numbers are a function of the weights and of these replies, and they "
    "are public (with --seeded: nothing publishes them; keep them as secret as the weights, unless "
    "you mean to publish them). Limit: a reply that meets them need not come from your engine, or "
    "from greedy decoding (with --seeded: from sampling at temperature 1). Limit: they suit "
    "replies like those given. Other lengths, languages or prompts can need others. Limit: the "
    "numbers are your own statement about your engine. No proof shows how they were fitted or on "
    "which replies: a client that accepts a pass within them trusts you for them as it trusts you "
    "for the commitment ID, and the pass says only that the reply's costs on the committed model "
    "lie within them.",
)
parser.add_argument(
    "model_directory",
    type=Path,
    help="the integer model: the directory with policy.txt, KL.bin and integer_model.json",
)
parser.add_argument(
    "reply_files",
    type=Path,
    nargs="+",
    metavar="reply_file",
    help='a JSON object with the lists "prompt_token_ids" and "reply_token_ids". At least 32 files',
)
parser.add_argument(
    "--share",
    type=int,
    default=100,
    metavar="PERCENT",
    help="theta admits the clipped mean of this percentage of the replies, from 50 to 100 (default "
    "100). A lower percentage rejects more of your own replies, and more of others'",
)
parser.add_argument("--C", type=int, metavar="N", help="the clip, in place of the fitted one")
parser.add_argument(
    "--end-token",
    type=token_ids,
    default=[],
    metavar="IDS",
    help="the token ids that end a reply, separated by commas. A reply that has one of them before "
    "its last token is then left out: no request can be made of it",
)
parser.add_argument(
    "--swapped",
    action="store_true",
    help="also score each reply after the next reply's prompt, in one more pass for each, and "
    "count how many meet the limits",
)
parser.add_argument(
    "--seeded",
    action="store_true",
    help='the replies were sampled at temperature 1, each under the integer "seed" of its file: '
    "the gaps are taken from the logits plus that seed's public noise. For scoring alone: no "
    "proof is made under a seed",
)
parser.add_argument(
    "--next-seed",
    action="store_true",
    help="with --seeded: also score each reply under its seed + 1, in one more pass for each, and "
    "count how many meet the limits",
)
parser.add_argument(
    "--device", default="cpu", help="the PyTorch device for the model: cpu (default) or cuda"
)
given = parser.parse_args()
if not 50 <= given.share <= 100:
    parser.error("--share must be from 50 to 100")
if given.next_seed and not given.seeded:
    parser.error("--next-seed needs --seeded")

# These checks need no weights, so they come before the weights are loaded.
policy_file = given.model_directory / "policy.txt"
try:
    parameters = integer_model.Parameters.read(policy_file)
    lines = dict(line.split(maxsplit=1) for line in policy_file.read_text().splitlines())
except (OSError, ValueError) as error:
    stop(str(error))
T, V, s, sh = parameters.T, parameters.V, parameters.s, int(lines["sh"])
if int(lines["sT"]) != sh - 4:
    parser.error(
        f"{policy_file} sets a log-probability bound (line sT). "
        "This script fits the clipped-gap policy only."
    )
if given.seeded and s > 40:
    parser.error(f"{policy_file}: with --seeded, s must be at most 40. It is {s}.")
own_choice = 1 << sh
if given.C is not None and not own_choice <= given.C < 1 << 32:
    parser.error(f"--C must be from {own_choice} (2^sh) to 4294967295")
if len(given.reply_files) < FEWEST:
    stop(f"{len(given.reply_files)} replies are too few. Give at least {FEWEST}.")

replies, seeds_ignored = [], 0
for file in given.reply_files:
    try:
        one = json.loads(file.read_text())
        prompt, reply = list(one["prompt_token_ids"]), list(one["reply_token_ids"])
        seed = one["seed"] if given.seeded else None
        seeds_ignored += not given.seeded and "seed" in one
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        stop(
            f'{file} is not a JSON object with the lists "prompt_token_ids" and "reply_token_ids"'
            f"{' and the integer seed' if given.seeded else ''} ({type(error).__name__})."
        )
    if given.seeded and not (type(seed) is int and -(1 << 63) <= seed < 1 << 64):
        stop(f'{file}: "seed" must be an integer that fits 64 bits.')
    if not all(type(token) is int and 0 <= token < V for token in prompt + reply):
        stop(f"{file}: every token id must be an integer from 0 to {V - 1}.")
    if not prompt or not reply or len(prompt) + len(reply) > T:
        stop(f"{file}: prompt and reply need a token each, and together at most {T}.")
    replies.append((file, prompt, reply, seed))
if len({(tuple(prompt), tuple(reply)) for _, prompt, reply, _ in replies}) < len(replies):
    stop("two files hold the same prompt and reply. Give each reply once.")
if seeds_ignored:
    say(
        f'{seeds_ignored} of the files have a "seed", which is ignored without --seeded: their '
        "replies are taken for greedy ones."
    )

try:
    model = integer_model.load_from_folder(given.model_directory, given.device)
except (OSError, ValueError, MemoryError) as error:
    stop(str(error))


def costs_of(prompt: list[int], reply: list[int], seed: int | None) -> list[int] | None:
    """The reply tokens' costs, or None if a range condition fails in the window."""
    window = prompt + reply + [0] * (T - len(prompt) - len(reply))
    sequence = model.new_sequence()
    try:
        # The position of token i scores token i + 1. The last entry is not used.
        _, gaps = model.extend(sequence, window, gap_for=window[1:] + [0], seed=seed)
    except integer_model.RangeConditionFailed:
        return None
    finally:
        model.free(sequence)
    first = len(prompt) - 1  # the position that scores the reply's first token
    return [int(gap) + own_choice for gap in gaps[first : first + len(reply)]]


kept = []
for file, prompt, reply, seed in replies:
    if set(reply[:-1]) & set(given.end_token):
        say(f"left out: {file}: an end token comes before the reply's last token")
    elif (costs := costs_of(prompt, reply, seed)) is None:
        say(f"left out: {file}: a range condition fails in its window")
    else:
        kept.append((prompt, reply, seed, costs))
say(f"replies read: {len(replies)}; left out: {len(replies) - len(kept)}")
if len(kept) < FEWEST:
    stop(f"{len(kept)} replies are too few. Give at least {FEWEST}.")
scored = [costs for _, _, _, costs in kept]
others = sum(cost > own_choice for costs in scored for cost in costs)
say(f"reply tokens: {sum(map(len, scored))}; not the integer model's own choice: {others}")
if others == 0:
    say("Every token is the integer model's own choice, so only such replies meet these limits.")

C, theta, cap = fit(scored, own_choice, given.share, given.C)
if not C < 1 << 32:
    stop(f"the fitted C, {C}, is above 4294967295, the largest that a parameter file can hold.")
if theta == C and cap > 0:
    stop(
        f"theta would equal C, {C}, with cap {cap}: the mean would bound nothing. Give longer "
        "replies, or a larger --C."
    )
say(f"of these replies, {sum(meets(costs, C, theta, cap) for costs in scored)} meet the limits")
half = len(scored) // 2
for name, fitted, tried in (
    ("first", scored[:half], scored[half:]),
    ("second", scored[half:], scored[:half]),
):
    passed = sum(meets(costs, *fit(fitted, own_choice, given.share, given.C)) for costs in tried)
    say(f"fitted on the {name} half alone: {passed} of the other {len(tried)} replies meet them")
if given.swapped:
    pairs = [
        (kept[(i + 1) % len(kept)][0], reply, seed) for i, (_, reply, seed, _) in enumerate(kept)
    ]
    wrong = [costs_of(*pair) for pair in pairs if len(pair[0]) + len(pair[1]) <= T]
    passed = sum(meets(costs, C, theta, cap) for costs in wrong)
    say(f"each reply after the next reply's prompt: {passed} of {len(wrong)} meet the limits")
if given.next_seed:
    # Modulo 2^64, as noise() takes a seed, so that the largest seed has a next one too.
    wrong = [costs_of(prompt, reply, (seed + 1) % (1 << 64)) for prompt, reply, seed, _ in kept]
    passed = sum(meets(costs, C, theta, cap) for costs in wrong)
    say(f"each reply under its seed + 1: {passed} of {len(wrong)} meet the limits")
if given.seeded:
    say("For scoring alone (score_reply.py --seed): no program proves or verifies a seeded policy.")
    say("Do not commit to a parameter file that holds these limits.")
    say(
        "The limits are a function of the weights and of these replies. Nothing publishes them: "
        "keep them as secret as the weights, unless you mean to publish them."
    )
else:
    say(f"check_proof.py: --policy clipped-gap --C {C} --s {s} --theta {theta} --cap {cap}")
    say(f"end_to_end.py: --policy clipped-gap --limits {C},{theta},{cap},{s}")
    # cp_verify compares the kind too, and calls these three values `exact-match`.
    kind = "exact-match" if (C, theta, cap) == (own_choice, own_choice, 0) else "clipped-gap"
    say(f"cp_verify: -policy {kind},{C},{theta},{cap},{s},{sh}")
    say("The limits are a function of the weights and of these replies, and they are public.")
print(f"C {C}\ntheta {theta}\ncap {cap}")
