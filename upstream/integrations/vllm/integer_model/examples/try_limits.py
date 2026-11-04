#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Fits a clipped-gap acceptance policy on the example small_float_pair and counts what it accepts.

usage: examples/try_limits.py DIRECTORY [--dtype float32] [--replies 64] [--device cuda]
       DIRECTORY is what `python3 -m integer_model_vllm.example small_float_pair --directory
       DIRECTORY` created. The exit code is 0 if all counts were printed, 1 if not, 2 for usage or
       a refused DIRECTORY.

The example has one small synthetic model twice: float_model/, in floating point, and model/, in
integers. This script plays a provider who serves the first and proves replies on the second.

1. Hugging Face transformers decodes 128 greedy replies from float_model/ on the CPU, in bfloat16,
   to prompts of 16 random token ids. A reply ends at the end token or after 32 tokens.
2. examples/fit_limits.py fits C, theta and cap on the first 64 replies.
3. The integer model scores the other 64 under these limits: as they are; each after the next
   reply's prompt; each with 1, 4 or 16 tokens replaced, by random ids or by the floating-point
   model's second choice at that position. Then it scores the replies of a nearby floating-point
   model to the same 64 prompts: one that multiplies the router's score into the routed expert's
   output as well as its input. Last come the integer model's own greedy replies to eight of the
   prompts, whose tokens all cost the least.

Prompts and replacements come from random.Random(2026), so every run draws the same. The replies can
differ between machines, and the limits with them. The replies are written to
DIRECTORY/replies/fitted and DIRECTORY/replies/held_out, in the form that fit_limits.py reads.

Limit: the weights are synthetic and the prompts are random ids. The counts show how the policy
behaves on this example. They say nothing about a real model's thresholds.
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

    find_or_hand_over("torch", "transformers", "integer_model")
    del find_or_hand_over

import argparse
import json
import os
import random
import re
import stat
import subprocess
import sys
from pathlib import Path

import torch
import transformers

import integer_model

PROMPT_TOKENS, REPLY_TOKENS = 16, 32


# fmt: off
# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")
# fmt: on


parser = argparse.ArgumentParser(
    description="Fits a clipped-gap acceptance policy on the example small_float_pair, on the CPU, "
    "and counts which other replies meet it.",
    epilog="Limit: the weights are synthetic and the prompts are random ids. The counts say "
    "nothing about a real model's thresholds.",
)
parser.add_argument(
    "directory",
    type=Path,
    help='what "python3 -m integer_model_vllm.example small_float_pair --directory DIRECTORY" '
    "created. The replies are written to DIRECTORY/replies, which must not exist yet",
)
parser.add_argument(
    "--dtype",
    default="bfloat16",
    choices=["bfloat16", "float32"],
    help="the type in which the floating-point model decodes (default bfloat16)",
)
parser.add_argument(
    "--replies",
    type=int,
    default=128,
    choices=[64, 128],
    help="how many replies are decoded (default 128). The first half is fitted, the second scored",
)
parser.add_argument(
    "--device",
    default="cpu",
    choices=["cpu", "cuda"],
    help="the PyTorch device for the integer model: cpu (default) or cuda. The floating-point "
    "model stays on the CPU",
)
given = parser.parse_args()
# The replies go into DIRECTORY: another account must not be able to change it or a folder above it.
# The look is always made (the routine refuses a missing DIRECTORY too), at the path as given and at
# its real path, and the script goes on by that real path, once it is sure that it did not turn.
real, shown = os.path.realpath(given.directory), printable(str(given.directory))
for place in (str(given.directory), real):
    found = others_can_change(place)
    if found and found[1] == "does not exist":
        missing = printable(found[0])
        words = f"try_limits.py: {missing} does not exist. Nothing was read or made."
        print(words, file=sys.stderr)
        sys.exit(2)
    if found:
        words = refusal("directory", str(given.directory), *found)
        words += ". Here the directory must hold the example: make the example again, with"
        words += " --directory, below a folder that only you can write in, and name that one."
        print("try_limits.py: " + words, file=sys.stderr)
        sys.exit(2)
if os.path.realpath(real) != real:  # checked from the top: no link can come now
    print(f"try_limits.py: {shown} changed while it was looked at.", file=sys.stderr)
    sys.exit(2)
given.directory = Path(real)
REPLIES = given.replies
if os.path.lexists(given.directory / "replies"):  # a link whose target is missing counts too
    parser.error(
        f"{printable(str(given.directory / 'replies'))} exists already, and nothing is overwritten."
        " Create the example in a new directory and name that one."
    )

try:
    model = integer_model.load_from_folder(given.directory / "model", given.device)
    policy_text = (given.directory / "model" / "policy.txt").read_text()
    # An absolute path that exists, and no look at a model hub: nothing is fetched from a network.
    floats = transformers.AutoModelForCausalLM.from_pretrained(
        (given.directory / "float_model").resolve(strict=True),
        local_files_only=True,
        attn_implementation="eager",
        dtype=getattr(torch, given.dtype),
    ).eval()
except (OSError, ValueError, MemoryError, RuntimeError, AssertionError) as error:
    sys.exit(f"No counts: {printable(str(error))}")  # the last two: no GPU for --device cuda
# The last count but one alters the layers with experts, which it finds by transformers' own names.
# Another version may name them otherwise, so that is looked at now, before minutes of work.
try:
    with_experts = [
        layer.feed_forward
        for layer in floats.model.layers
        if hasattr(layer.feed_forward, "router") and hasattr(layer.feed_forward, "experts")
    ]
except AttributeError:
    with_experts = []
if len(with_experts) != sum(map(model.parameters.has_experts, range(model.parameters.L))):
    sys.exit(
        "No counts: this version of transformers names the parts of a layer with experts "
        "otherwise than this script expects, so it cannot make the nearby model."
    )
T, end_token = model.parameters.T, model.parameters.V - 1  # the example's end token is the last id
own_choice = 1 << int(dict(line.split(maxsplit=1) for line in policy_text.splitlines())["sh"])
draw = random.Random(2026)


def decode(some_prompts: list[list[int]]) -> list[list[int]]:
    """Each prompt with the floating-point model's REPLY_TOKENS greedy choices after it."""
    sequences = [list(prompt) for prompt in some_prompts]
    with torch.no_grad():
        for _ in range(REPLY_TOKENS):
            choices = floats(torch.tensor(sequences)).logits[:, -1].float().argmax(dim=1).tolist()
            for sequence, choice in zip(sequences, choices):
                sequence.append(choice)
    return sequences


def replies_in(sequences: list[list[int]]) -> list[list[int]]:
    """The replies, each up to its first end token."""
    whole = [sequence[PROMPT_TOKENS:] for sequence in sequences]
    return [r[: r.index(end_token) + 1] if end_token in r else r for r in whole]


prompts = [[draw.randrange(end_token) for _ in range(PROMPT_TOKENS)] for _ in range(REPLIES)]
sequences = decode(prompts)
with torch.no_grad():  # the second choice after each prefix of prompt and reply
    second = floats(torch.tensor(sequences)).logits.float().topk(2, dim=2).indices[:, :, 1].tolist()
replies = replies_in(sequences)

fitted, held_out = range(REPLIES // 2), range(REPLIES // 2, REPLIES)
LEFT = (
    "nothing is overwritten. Whatever was written stays in {}. To run this script again, create the"
    " example in a new directory and name that one."
)
try:
    # Each level new (mkdir() without parents refuses one that stands), and not writable by group or
    # others whatever the umask, so that nobody else can swap a folder below DIRECTORY.
    (given.directory / "replies").mkdir(mode=0o755)
    for name, numbers in (("fitted", fitted), ("held_out", held_out)):
        (given.directory / "replies" / name).mkdir(mode=0o755)
        for i in numbers:
            one = {"prompt_token_ids": prompts[i], "reply_token_ids": replies[i]}
            with open(given.directory / "replies" / name / f"{i:03d}.json", "x") as file:
                file.write(json.dumps(one) + "\n")
except FileExistsError as error:
    left = LEFT.format(printable(str(given.directory / "replies")))
    sys.exit(f"try_limits.py: {printable(str(error.filename))} exists already; {left}")
except OSError as error:
    left = LEFT.format(printable(str(given.directory / "replies")))
    sys.exit(f"try_limits.py: {printable(str(error.filename))}: {error.strerror}; {left}")
fit = subprocess.run(
    [
        sys.executable,
        Path(__file__).with_name("fit_limits.py"),
        given.directory / "model",
        *sorted((given.directory / "replies" / "fitted").iterdir()),
        "--device",
        given.device,
    ],
    stdout=subprocess.PIPE,
    text=True,
)
if fit.returncode:
    sys.exit(1)
print(fit.stdout, end="")
C, theta, cap = (int(line.split()[1]) for line in fit.stdout.splitlines())


def meets(prompt: list[int], reply: list[int]) -> bool:
    """Whether the reply meets the limits, in the window of a proof (as in fit_limits.py)."""
    window = prompt + reply + [0] * (T - len(prompt) - len(reply))
    sequence = model.new_sequence()
    try:
        _, gaps = model.extend(sequence, window, gap_for=window[1:] + [0])
    except integer_model.RangeConditionFailed:
        return False
    finally:
        model.free(sequence)
    first = len(prompt) - 1
    costs = [int(gap) + own_choice for gap in gaps[first : first + len(reply)]]
    return sum(min(c, C) for c in costs) <= theta * len(costs) and sum(c > C for c in costs) <= cap


def count(what: str, pairs: list[tuple[list[int], list[int]]]) -> None:
    met = sum(meets(prompt, reply) for prompt, reply in pairs)
    print(f"{what}: {met} of {len(pairs)} meet the limits", flush=True)


count("held-out replies", [(prompts[i], replies[i]) for i in held_out])
count(
    "each after the next reply's prompt",
    [(prompts[i + 1 if i + 1 < REPLIES else held_out[0]], replies[i]) for i in held_out],
)
for replaced in (1, 4, 16):
    for kind in ("a random id", "the floating-point model's second choice"):
        pairs = []
        for i in held_out:
            reply = list(replies[i])
            if len(reply) < replaced:
                continue
            for j in draw.sample(range(len(reply)), replaced):
                if kind == "a random id":
                    reply[j] = draw.choice([id_ for id_ in range(end_token) if id_ != reply[j]])
                else:
                    reply[j] = second[i][PROMPT_TOKENS - 1 + j]
            pairs.append((prompts[i], reply))
        count(f"{replaced} of the tokens replaced by {kind}", pairs)

# A nearby model: Llama 4 multiplies the router's score into the routed expert's input, and this one
# multiplies it into that expert's output too. The names router and experts, what the router gives
# (the scores first: a row for each token, a column for each expert) and the order of the experts'
# rows (expert by expert, each with all tokens) are those of transformers' Llama4TextMoe.
scaled = 0  # how often a layer's output was scaled
for block in with_experts:
    seen: dict[str, torch.Tensor] = {}

    def keep(_module, _inputs, output, seen=seen):
        seen["scores"] = output[0]

    def scale(_module, _inputs, output, seen=seen):
        global scaled
        factor = seen["scores"].transpose(0, 1).reshape(-1, 1)
        # A score is a sigmoid's value. An expert that a token did not choose has the score 0, and
        # so the input 0 and the output 0. Scores and rows that are paired otherwise fail this.
        if not (
            len(factor) == len(output)
            and bool(((factor >= 0) & (factor <= 1)).all())
            and bool((factor == 0).any())
            and not bool(output[factor[:, 0] == 0].any())
        ):
            raise RuntimeError("the router's scores and the experts' rows do not pair as expected")
        scaled += 1
        return output * factor

    block.router.register_forward_hook(keep)
    block.experts.register_forward_hook(scale)
try:
    others = replies_in(decode([prompts[i] for i in held_out]))
    if scaled != REPLY_TOKENS * len(with_experts):
        raise RuntimeError(f"{scaled} outputs were scaled")
    if others == [replies[i] for i in held_out]:
        raise RuntimeError("its replies are the floating-point model's own")
except (AttributeError, IndexError, KeyError, RuntimeError, TypeError) as error:
    sys.exit(f"No count for the nearby model: this version of transformers differs: {error!r}")
count(
    "replies of a nearby model, which applies the router's score twice",
    [(prompts[i], reply) for i, reply in zip(held_out, others)],
)
own = []
for i in held_out[:8]:  # token by token, which is slow; eight show it
    sequence = model.new_sequence()
    reply = [int(model.extend(sequence, prompts[i])[-1])]
    while len(reply) < REPLY_TOKENS and reply[-1] != end_token:
        reply.append(int(model.extend(sequence, reply[-1:])[0]))
    model.free(sequence)
    own.append((prompts[i], reply))
count("the integer model's own greedy replies", own)
