# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of examples/fit_limits.py and examples/try_limits.py, at small size on the CPU. They need
neither vLLM nor a GPU.

Served replies come from two engines that compute nearly what the integer model computes. One is a
stand-in: the logits of the tests' smallest model (16 tokens in a window, 200 in the vocabulary)
plus a random error, decoded greedily. The other is the floating-point model of the example
small_float_pair under Hugging Face transformers, which is installed with vLLM.
"""

import json
import math
import os
import random
import re
import shutil
import sys
from fractions import Fraction

import pytest
import torch
from test_examples import PACKAGE, make_example, run

import integer_model
from integer_model.noise import TEMPERATURE_ONE, noise

SCRIPT = PACKAGE / "examples" / "fit_limits.py"
NUMBER, T = 40, 16
OWN_CHOICE = 256  # 2^sh: the cost of the integer model's own choice


def prompts(number, V=200, lengths=(4,)):
    return [
        [((37 + 2 * j) * i + 11 + 5 * j) % V for i in range(lengths[j % len(lengths)])]
        for j in range(number)
    ]


def write(directory, replies):
    """One file for each pair of prompt and reply. Returns the files."""
    directory.mkdir()
    files = [directory / f"{number:04}.json" for number in range(len(replies))]
    for file, (prompt, reply) in zip(files, replies):
        file.write_text(
            json.dumps({"prompt_token_ids": prompt, "reply_token_ids": reply, "more": "ignored"})
        )
    return files


def fit(*words, status=0, start=(sys.executable, SCRIPT)):
    """Runs the script. Returns the lines of standard output and of standard error."""
    done = run(*start, *words, env={**os.environ, "PYTHONPATH": str(PACKAGE)})
    assert done.returncode == status and "Traceback" not in done.stderr, done.stderr
    return done.stdout.splitlines(), done.stderr.splitlines()


def rule(costs, share=100, clip=None):
    """C (the clip), theta and cap by the rule in the script's docstring, in exact fractions."""
    pooled = sorted(cost for reply in costs for cost in reply)
    if clip is None:
        clip = pooled[math.ceil(Fraction(999, 1000) * len(pooled)) - 1]
        if clip == OWN_CHOICE:
            clip = min((cost for cost in pooled if cost > OWN_CHOICE), default=OWN_CHOICE)
    means = sorted(Fraction(sum(min(c, clip) for c in reply), len(reply)) for reply in costs)
    theta = math.ceil(means[math.ceil(Fraction(share, 100) * len(means)) - 1])
    return clip, theta, max(sum(c > clip for c in reply) for reply in costs)


def meets(costs, C, theta, cap):
    """The rule of examples/score_reply.py."""
    return sum(min(c, C) for c in costs) <= theta * len(costs) and sum(c > C for c in costs) <= cap


def costs_of(model, prompt, reply):
    tokens = prompt + reply
    sequence = model.new_sequence()
    _, gaps = model.extend(sequence, tokens[:-1], gap_for=tokens[1:])
    model.free(sequence)
    return [int(gap) + OWN_CHOICE for gap in gaps[len(prompt) - 1 :]]


def swapped(model, pairs, T=T):
    """Each reply's costs after the next reply's prompt, where the two fit into a window."""
    moved = [(pairs[(i + 1) % len(pairs)][0], reply) for i, (_, reply) in enumerate(pairs)]
    return [costs_of(model, prompt, reply) for prompt, reply in moved if len(prompt + reply) <= T]


def expected(costs, wrong=None, **options):
    """What the script prints for costs that are not all 256: standard output, standard error."""
    C, theta, cap = limits = rule(costs, **options)
    half, others = len(costs) // 2, sum(c > OWN_CHOICE for reply in costs for c in reply)
    first = sum(meets(reply, *rule(costs[:half], **options)) for reply in costs[half:])
    second = sum(meets(reply, *rule(costs[half:], **options)) for reply in costs[:half])
    return [f"C {C}", f"theta {theta}", f"cap {cap}"], [
        f"replies read: {len(costs)}; left out: 0",
        f"reply tokens: {sum(map(len, costs))}; not the integer model's own choice: {others}",
        f"of these replies, {sum(meets(reply, *limits) for reply in costs)} meet the limits",
        f"fitted on the first half alone: {first} of the other {len(costs) - half} replies "
        "meet them",
        f"fitted on the second half alone: {second} of the other {half} replies meet them",
        *(
            []
            if wrong is None
            else [
                "each reply after the next reply's prompt: "
                f"{sum(meets(reply, *limits) for reply in wrong)} of {len(wrong)} meet the limits"
            ]
        ),
        # Users paste the options of these three lines, so their form is fixed.
        f"check_proof.py: --policy clipped-gap --C {C} --s 16 --theta {theta} --cap {cap}",
        f"end_to_end.py: --policy clipped-gap --limits {C},{theta},{cap},16",
        f"cp_verify: -policy {'exact-match' if limits == (256, 256, 0) else 'clipped-gap'},"
        f"{C},{theta},{cap},16,8",
        "The limits are a function of the weights and of these replies, and they are public.",
    ]


@pytest.fixture(scope="module")
def directory(tmp_path_factory):
    """A model directory for the tests' smallest configuration, with synthetic weights."""
    new = tmp_path_factory.mktemp("tiny")
    for name in ("policy.txt", "KL.bin"):
        shutil.copy(PACKAGE / "tests" / "configurations" / "tiny_without_clamps" / name, new)
    (new / "integer_model.json").write_text('{"weights": "synthetic"}\n')
    return new


@pytest.fixture(scope="module")
def model(directory):
    return integer_model.load_from_folder(directory, "cpu")


def decode(model, prompt, length, error=None):
    """A greedy reply: the integer model's own or, with error, from its logits plus error()."""
    sequence, reply, fed = model.new_sequence(), [], prompt
    for _ in range(length):
        kept = {}
        model.extend(sequence, fed, record=lambda name, _, values: kept.update({name: values}))
        logits = kept["lam"][-1].double()
        fed = [int((logits if error is None else logits + error(logits.shape)).argmax())]
        reply += fed
    model.free(sequence)
    return reply


@pytest.fixture(scope="module")
def served(model):
    """Pairs of prompt and reply from the stand-in engine. The prompts have 2 to 8 tokens and the
    replies 8, so some fill the window. Two longer replies fill it too, and do not fit into it after
    the next prompt.
    """
    generator = torch.Generator().manual_seed(1)

    def error(shape):
        return 1.5e6 * torch.randn(shape, generator=generator, dtype=torch.float64)

    return [
        (prompt, decode(model, prompt, T - len(prompt) if j in (7, 30) else 8, error))
        for j, prompt in enumerate(prompts(NUMBER, lengths=(2, 3, 4, 5, 6, 7, 8)))
    ]


@pytest.fixture(scope="module")
def files(served, tmp_path_factory):
    return write(tmp_path_factory.mktemp("served") / "replies", served)


@pytest.fixture(scope="module")
def costs(model, served):
    return [costs_of(model, prompt, reply) for prompt, reply in served]


def test_the_replies_of_these_tests_have_what_the_tests_need(model, served, costs):
    others = sum(cost > OWN_CHOICE for reply in costs for cost in reply)
    assert 10 < others < sum(map(len, costs)) // 2, "The stand-in should differ in some tokens."
    assert sum(len(prompt + reply) == T for prompt, reply in served) > 2
    assert len(swapped(model, served)) == NUMBER - 2
    halves = [rule(costs[: NUMBER // 2]), rule(costs[NUMBER // 2 :])]
    assert halves[0] != halves[1]


def test_the_script_as_the_page_calls_it(directory, model, served, files, costs, tmp_path):
    """By its path, as in OPERATING.md: it must be executable. PATH starts with the directory of
    pytest's interpreter, so that the script's `env python3` finds that one.
    """
    path = os.pathsep.join([os.path.dirname(sys.executable), os.environ["PATH"]])
    done = run(
        SCRIPT,
        directory,
        *files,
        "--swapped",
        env={**os.environ, "PATH": path, "PYTHONPATH": str(PACKAGE)},
    )
    wrong = swapped(model, served)
    assert (done.stdout.splitlines(), done.stderr.splitlines()) == expected(costs, wrong)
    C, theta, cap = rule(costs)
    # With fewer than 1,000 costs the percentile is the largest cost, so none is above the clip.
    assert C == max(max(reply) for reply in costs) and cap == 0
    assert all(meets(reply, C, theta, cap) for reply in costs)
    assert not all(meets(reply, C, theta - 1, cap) for reply in costs)
    assert sum(meets(reply, C, theta, cap) for reply in wrong) < NUMBER // 8

    # examples/score_reply.py agrees, with the three lines in a copy of the parameter file.
    fitted = dict(line.split() for line in done.stdout.splitlines())
    policy = tmp_path / "policy.txt"
    policy.write_text(
        "".join(
            f"{name} {fitted.get(name, value)}\n"
            for name, value in (
                line.split(maxsplit=1)
                for line in (directory / "policy.txt").read_text().splitlines()
            )
        )
    )
    chosen = max(range(8, 28), key=lambda i: sum(costs[i]))
    for prompt, status, verdict in (
        (served[chosen][0], 0, "meets the policy (a score, not a verification)"),
        (served[chosen + 1][0], 1, "fails the policy (a score, not a verification)"),
    ):
        assert meets(costs_of(model, prompt, served[chosen][1]), C, theta, cap) == (status == 0)
        words = (",".join(map(str, ids)) for ids in (prompt, served[chosen][1]))
        start = (sys.executable, PACKAGE / "examples" / "score_reply.py")
        assert fit(directory, *words, "--policy", policy, status=status, start=start)[0][-1] == (
            verdict
        )

    # --help ends with the docstring's four limits, word for word.
    printed = " ".join(word for line in fit("--help")[0] for word in line.split())
    assert "trusts you for them as it trusts you for the commitment ID" in printed
    assert printed.endswith(
        " ".join(SCRIPT.read_text().split('"""')[1].partition("Limit: ")[2].split())
    )
    # So does score_reply.py's, with its one limit.
    printed = " ".join(word for line in fit("--help", start=start)[0] for word in line.split())
    assert printed.endswith("With secret weights, publish none of them.")
    assert printed.endswith(
        " ".join(start[1].read_text().split('"""')[1].partition("Limit: ")[2].split())
    )


@pytest.mark.parametrize("share", [50, 77, 90])  # 77 % of 40 and of 20 replies are no whole numbers
def test_a_lower_share(share, directory, files, costs):
    assert fit(directory, *files, "--share", share) == expected(costs, share=share)
    C, theta, cap = rule(costs, share=share)
    needed = math.ceil(NUMBER * share / 100)
    assert needed <= sum(meets(reply, C, theta, cap) for reply in costs) < NUMBER
    assert sum(meets(reply, C, theta - 1, cap) for reply in costs) < needed


def test_a_given_clip(directory, files, costs):
    clip = sorted(cost for reply in costs for cost in reply)[-4]  # three costs are above it
    assert fit(directory, *files, "--C", clip) == expected(costs, clip=clip)
    C, theta, cap = rule(costs, clip=clip)
    assert 1 <= cap <= 3 and not all(meets(reply, C, theta, cap - 1) for reply in costs)
    assert fit(directory, *files, "--C", (1 << 32) - 1)[0][0] == "C 4294967295"


def test_above_a_thousand_costs_the_clip_is_below_the_largest(directory, model, tmp_path):
    """Random replies, which no engine served: only the arithmetic of C and cap is under test."""
    draw = random.Random(2)
    pairs = [(prompt, [draw.randrange(200) for _ in range(11)]) for prompt in prompts(100)]
    costs = [costs_of(model, prompt, reply) for prompt, reply in pairs]
    assert fit(directory, *write(tmp_path / "replies", pairs)) == expected(costs)
    pooled = sorted(cost for reply in costs for cost in reply)
    assert (
        len(pooled) == 1100 and rule(costs)[::2] == (pooled[1098], 1) and pooled[1098] < pooled[-1]
    )


def test_the_integer_models_own_replies(directory, model, tmp_path):
    pairs = [(prompt, decode(model, prompt, 11)) for prompt in prompts(100)]
    files = write(tmp_path / "own", pairs)
    # With a clip above 2^sh the lines have the form of a clipped-gap policy. They accept the same
    # replies: a mean of 256 leaves no room for a cost above 256.
    for words, clip in (([], 256), (["--C", "257"], 257), (["--C", "1000"], 1000)):
        output, said = fit(directory, *files[:32], *words)
        assert output == [f"C {clip}", "theta 256", "cap 0"]
        assert said[2] == (
            "Every token is the integer model's own choice, so only such replies meet these limits."
        )
        kind = "exact-match" if clip == 256 else "clipped-gap"  # cp_verify compares the kind too
        assert said[-2] == f"cp_verify: -policy {kind},{clip},256,0,16,8"
    # One other token among 1,100: the percentile is 2^sh, and C is that token's cost.
    pairs[50] = (pairs[50][0], pairs[50][1][:-1] + [(pairs[50][1][-1] + 1) % 200])
    costs = [costs_of(model, prompt, reply) for prompt, reply in pairs]
    assert sum(cost > OWN_CHOICE for reply in costs for cost in reply) == 1
    assert fit(directory, *write(tmp_path / "but_one", pairs)) == expected(costs)
    assert rule(costs) == (costs[50][-1], -(-(10 * OWN_CHOICE + costs[50][-1]) // 11), 0)


def test_theta_never_equals_the_clip_with_a_cap_above_0(directory, files, costs, tmp_path):
    """Under such limits the mean bounds nothing; cp_verify rejects them for short requests."""
    most = max(sum(cost > OWN_CHOICE for cost in reply) for reply in costs)
    output, said = fit(directory, *files, "--C", OWN_CHOICE, status=1)
    assert output == [] and said[-1] == (
        f"No limits: theta would equal C, 256, with cap {most}: the mean would bound nothing. Give "
        "longer replies, or a larger --C."
    )
    # The rule itself leads there if the cost of a reply's only token is above the clip. With
    # cap 0, theta may equal C.
    draw = random.Random(3)
    pairs = [([j % 200, j // 200], [draw.randrange(200)]) for j in range(1100)]
    output, said = fit(directory, *write(tmp_path / "many", pairs), status=1)
    assert output == [] and re.fullmatch(
        r"No limits: theta would equal C, \d+, with cap 1: the mean would bound nothing\. .*",
        said[-1],
    )
    C, theta, cap = fit(directory, *write(tmp_path / "fewer", pairs[:900]))[0]
    assert theta.split()[1] == C.split()[1] and cap == "cap 0"


# Runs the script with another IntegerModel.extend. It takes whole windows only, padded with token 0
# from position 12 on. A range condition fails at token 199. FACTOR, if set, multiplies the gaps.
WITH_ANOTHER_EXTEND = """
import os, runpy, sys, integer_model
extend = integer_model.IntegerModel.extend
def other(self, sequence, tokens, **more):
    assert len(tokens) == 16 and tokens[12:] == [0] * 4
    if 199 in tokens:
        raise integer_model.RangeConditionFailed(tokens.index(199), 7)
    choices, gaps = extend(self, sequence, tokens, **more)
    return choices, gaps * int(os.environ.get("FACTOR", 1))
integer_model.IntegerModel.extend = other
del sys.argv[0]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_replies_that_are_left_out(directory, served, tmp_path, monkeypatch):
    """No window of these replies fails a range condition, so the test makes one fail."""
    pairs = [(p, r[: 12 - len(p)]) for p, r in served if 199 not in p + r][:34]
    # Two ids that no reply has serve as the end tokens.
    end, other_end = sorted(set(range(199)) - {token for _, reply in pairs for token in reply})[:2]
    pairs[3] = (pairs[3][0], pairs[3][1][:-1] + [199])
    pairs[5] = (pairs[5][0], [end] + pairs[5][1][1:])  # an end token, then more tokens
    pairs[6] = (pairs[6][0], pairs[6][1][:-1] + [other_end])  # as the last token it is in order
    files = write(tmp_path / "replies", pairs)
    # python3 -c puts the current directory first on its import path, so the test changes to its own
    # temporary directory.
    monkeypatch.chdir(tmp_path)
    start = (sys.executable, "-c", WITH_ANOTHER_EXTEND, SCRIPT)
    ends = f"{end},{other_end}"
    output, said = fit(directory, *files, "--end-token", ends, start=start)
    assert len(output) == 3 and said[:3] == [
        f"left out: {files[3]}: a range condition fails in its window",
        f"left out: {files[5]}: an end token comes before the reply's last token",
        "replies read: 34; left out: 2",
    ]
    output, said = fit(directory, *files[:33], "--end-token", ends, status=1, start=start)
    assert output == [] and said[-1] == "No limits: 31 replies are too few. Give at least 32."

    monkeypatch.setenv("FACTOR", "1000")  # the largest cost is then above 2^32
    output, said = fit(directory, *files, status=1, start=start)
    assert output == [] and re.fullmatch(
        r"No limits: the fitted C, \d{10,}, is above 4294967295, the largest that a parameter file "
        r"can hold\.",
        said[-1],
    )


def test_what_the_script_rejects(directory, served, files, tmp_path):
    def changed(name, **fields):
        """The files, with a changed copy in place of the first."""
        prompt, reply = served[0]
        new = tmp_path / f"{name}.json"
        new.write_text(json.dumps({"prompt_token_ids": prompt, "reply_token_ids": reply} | fields))
        return [new, *files[1:]]

    for name, text in (("text", "17,400,3"), ("deep", "[" * 200_000)):
        (tmp_path / name).write_text(text)
    # A model directory without weights: each of these is rejected before the weights are loaded.
    early = tmp_path / "early"
    early.mkdir()
    for name in ("policy.txt", "KL.bin"):
        shutil.copy(directory / name, early)
    ids = "every token id must be an integer from 0 to 199."
    for words, reason in (
        (files[:31], "31 replies are too few. Give at least 32."),
        ([*files, files[0]], "two files hold the same prompt and reply. Give each reply once."),
        ([tmp_path / "text", *files[1:]], f"{tmp_path / 'text'} is not a JSON object with the "),
        ([tmp_path / "deep", *files[1:]], f"{tmp_path / 'deep'} is not a JSON object with the "),
        ([tmp_path / "none", *files[1:]], f"{tmp_path / 'none'} is not a JSON object with the "),
        (changed("no_reply", reply_token_ids=None), f"{tmp_path / 'no_reply.json'} is not a JSON "),
        (changed("empty", reply_token_ids=[]), "prompt and reply need a token each, and together"),
        (changed("long", prompt_token_ids=[1] * 9), "need a token each, and together at most 16."),
        (changed("large", reply_token_ids=[200]), ids),
        (changed("negative", prompt_token_ids=[-1]), ids),
        (changed("words", reply_token_ids=["1"]), ids),
        (changed("true", reply_token_ids=[True]), ids),
        (changed("lists", prompt_token_ids=[[1]]), ids),
    ):
        output, said = fit(early, *words, status=1)
        assert output == [] and said[-1].startswith("No limits: ") and reason in said[-1], said
    output, said = fit(early, *files, status=1)
    assert output == [] and "integer_model.json" in said[-1]

    lines = (directory / "policy.txt").read_text()
    for name, old, new, status in (
        ("log_probability", "\nsT 4\n", "\nsT 16\n", 2),
        ("without_s", "\ns 16\n", "\n", 1),
        ("negative_sh", "\nsh 8\nsT 4\n", "\nsh -1\nsT -5\n", 1),
        ("huge_sh", "\nsh 8\nsT 4\n", "\nsh 40000000000\nsT 39999999996\n", 1),
    ):
        other = tmp_path / name
        other.mkdir()
        assert lines.count(old) == 1
        (other / "policy.txt").write_text(lines.replace(old, new))
        shutil.copy(directory / "KL.bin", other)
        output, said = fit(other, *files, status=status)
        assert output == [] and f"{other / 'policy.txt'}" in said[-1], said
    for words in (["--share", "49"], ["--share", "101"], ["--C", "255"], ["--C", str(1 << 32)], []):
        output, said = fit(directory, *(files if words else []), *words, status=2)
        assert output == [] and said[0].startswith("usage: ")


# Seeded replies


def seeded_costs_of(model, prompt, reply, seed):
    tokens = prompt + reply
    sequence = model.new_sequence()
    _, gaps = model.extend(sequence, tokens[:-1], gap_for=tokens[1:], seed=seed)
    model.free(sequence)
    return [int(gap) + OWN_CHOICE for gap in gaps[len(prompt) - 1 :]]


@pytest.fixture(scope="module")
def fine_directory(directory, tmp_path_factory):
    """The model directory with s 21 for s 16. The line moves no logit; it states their unit, in
    which the noise is computed. This model's logits lie millions of units apart, so at s 16, where
    the noise is at most 1,635,339, the noise would hardly ever decide a token.
    """
    new = tmp_path_factory.mktemp("fine") / "model"
    shutil.copytree(directory, new, copy_function=shutil.copyfile)  # writable, whatever the modes
    lines = (directory / "policy.txt").read_text()
    assert lines.count("\ns 16\n") == 1
    (new / "policy.txt").write_text(lines.replace("\ns 16\n", "\ns 21\n"))
    return new


@pytest.fixture(scope="module")
def fine_model(fine_directory):
    return integer_model.load_from_folder(fine_directory, "cpu")


@pytest.fixture(scope="module")
def sampled(fine_model):
    """Prompt, reply and seed from a stand-in engine that samples at temperature 1 under the seed:
    at each position it takes the argmax of the integer model's logits, plus a random error, plus
    the seed's public noise.
    """
    model = fine_model
    generator, p = torch.Generator().manual_seed(2), model.parameters
    found = []
    for j, prompt in enumerate(prompts(NUMBER, lengths=(2, 3, 4, 5, 6, 7, 8))):
        seed = ((1 << 64) - 1, -5, 0)[j] if j < 3 else 1000 + j
        sequence, reply, fed = model.new_sequence(), [], prompt
        for _ in range(8):
            kept = {}
            model.extend(sequence, fed, record=lambda name, _, values: kept.update({name: values}))
            position = sequence.length - 1
            z = kept["lam"][-1].double() + noise(seed, [position], p.V, TEMPERATURE_ONE, p.s)[0]
            fed = [int((z + 1e5 * torch.randn(z.shape, generator=generator).double()).argmax())]
            reply += fed
        model.free(sequence)
        found.append((prompt, reply, seed))
    return found


def write_seeded(directory, replies):
    files = write(directory, [(prompt, reply) for prompt, reply, _ in replies])
    for file, (_, _, seed) in zip(files, replies):
        file.write_text(json.dumps(json.loads(file.read_text()) | {"seed": seed}))
    return files


def test_seeded_replies(directory, fine_directory, fine_model, sampled, tmp_path):
    model = fine_model
    files = write_seeded(tmp_path / "sampled", sampled)
    costs = [seeded_costs_of(model, *one) for one in sampled]
    moved = [
        (sampled[(i + 1) % NUMBER][0], reply, seed) for i, (_, reply, seed) in enumerate(sampled)
    ]
    after_the_next_prompt = [seeded_costs_of(model, *one) for one in moved]
    under_the_next_seed = [
        seeded_costs_of(model, prompt, reply, (seed + 1) % (1 << 64))
        for prompt, reply, seed in sampled
    ]
    limits = rule(costs)
    output, said = expected(costs, after_the_next_prompt)
    passed = sum(meets(reply, *limits) for reply in under_the_next_seed)
    said[-4:] = [
        f"each reply under its seed + 1: {passed} of {NUMBER} meet the limits",
        "For scoring alone (score_reply.py --seed): no program proves or verifies a seeded policy.",
        "Do not commit to a parameter file that holds these limits.",
        "The limits are a function of the weights and of these replies. Nothing publishes them: "
        "keep them as secret as the weights, unless you mean to publish them.",
    ]
    assert fit(fine_directory, *files, "--seeded", "--swapped", "--next-seed") == (output, said)
    # The replies show something: the noise decides many tokens, and the controls mostly fail.
    plain = [costs_of(model, prompt, reply) for prompt, reply, _ in sampled]
    assert sum(c > OWN_CHOICE for reply in plain for c in reply) > 3 * sum(
        c > OWN_CHOICE for reply in costs for c in reply
    )
    assert all(meets(reply, *limits) for reply in costs)
    assert passed < NUMBER // 4
    assert sum(meets(reply, *limits) for reply in after_the_next_prompt) < NUMBER // 4

    # Without --seeded the key "seed" is ignored, and the script says so first.
    greedy_output, greedy_said = expected(plain)
    warning = (
        f'{NUMBER} of the files have a "seed", which is ignored without --seeded: their replies '
        "are taken for greedy ones."
    )
    assert fit(directory, *files) == (greedy_output, [warning, *greedy_said])

    # examples/score_reply.py --seed agrees, with the three lines in a copy of the parameter file.
    fitted = dict(line.split() for line in output)
    policy = tmp_path / "policy.txt"
    policy.write_text(
        "".join(
            f"{name} {fitted.get(name, value)}\n"
            for name, value in (
                line.split(maxsplit=1)
                for line in (fine_directory / "policy.txt").read_text().splitlines()
            )
        )
    )
    chosen = next(i for i, reply in enumerate(under_the_next_seed) if not meets(reply, *limits))
    prompt, reply, seed = sampled[chosen]
    words = [",".join(map(str, ids)) for ids in (prompt, reply)]
    start = (sys.executable, PACKAGE / "examples" / "score_reply.py")
    for given, status, verdict in (
        (seed, 0, "meets the policy (a score, not a verification)"),
        ((seed + 1) % (1 << 64), 1, "fails the policy (a score, not a verification)"),
    ):
        options = ("--policy", policy, f"--seed={given}")
        assert fit(fine_directory, *words, *options, status=status, start=start)[0][-1] == verdict


def test_what_the_script_rejects_of_seeded_replies(directory, sampled, tmp_path):
    files = write_seeded(tmp_path / "sampled", sampled)

    def changed(name, seed):
        new = tmp_path / f"{name}.json"
        fields = json.loads(files[0].read_text())
        del fields["seed"]
        new.write_text(json.dumps(fields if seed is None else fields | {"seed": seed}))
        return [new, *files[1:]]

    size = '"seed" must be an integer that fits 64 bits.'
    for name, seed, reason in (
        ("none", None, "and the integer seed (KeyError)."),
        ("large", 1 << 64, size),
        ("small", -(1 << 63) - 1, size),
        ("true", True, size),
        ("text", "7", size),
        ("fraction", 7.0, size),
    ):
        output, said = fit(directory, *changed(name, seed), "--seeded", status=1)
        assert output == [] and said[-1].startswith("No limits: ") and reason in said[-1], said
    output, said = fit(directory, *files, "--next-seed", status=2)
    assert output == [] and said[-1].endswith("--next-seed needs --seeded")
    other = tmp_path / "fine_unit"
    other.mkdir()
    lines = (directory / "policy.txt").read_text()
    assert lines.count("\ns 16\n") == 1
    (other / "policy.txt").write_text(lines.replace("\ns 16\n", "\ns 41\n"))
    shutil.copy(directory / "KL.bin", other)
    output, said = fit(other, *files, "--seeded", status=2)
    assert output == [] and said[-1].endswith("with --seeded, s must be at most 40. It is 41.")


def test_the_lines_of_the_page_make_a_model_directory(tmp_path):
    """OPERATING.md's lines for Llama 4 Maverick, on the files of the example small_gguf, arranged
    like the directory fit.N of that page: config/, and weights-directory.txt, which names the
    directory that holds the GGUF files, wsums.txt and fold.txt.
    """
    assert make_example("small_gguf", "--directory", tmp_path / "new").returncode == 0
    weights, fitted = tmp_path / "with [brackets]" / "weights", tmp_path / "fit.1"
    shutil.copytree(tmp_path / "new" / "weights", weights)
    assert {"wsums.txt", "fold.txt"} < {file.name for file in weights.iterdir()}
    (fitted / "config").mkdir(parents=True)
    (fitted / "weights-directory.txt").write_text(f"{weights}\n")
    for name in ("policy.txt", "KL.bin"):
        shutil.copy(tmp_path / "new" / "reference" / name, fitted / "config")
    (block,) = re.findall(
        r"```sh\n(D=\$\(mktemp -d\) .*?integer_model\.json\"\n)```",
        (PACKAGE / "OPERATING.md").read_text(),
        flags=re.DOTALL,
    )
    variables = {**os.environ, "FIT": str(fitted), "TMPDIR": str(tmp_path)}
    done = run("sh", "-c", block + 'echo "$D"', env=variables, cwd=tmp_path)
    assert done.returncode == 0 and done.stderr == "", done.stderr
    model = integer_model.load_from_folder(done.stdout.strip(), "cpu")
    sequence = model.new_sequence()
    _, gaps = model.extend(sequence, [17, 400, 3, 93, 45, 59], gap_for=[400, 3, 93, 45, 59, 289])
    assert gaps[2:].tolist() == [0, 0, 0, 0]  # the example's reply (examples/request.sh)

    # A directory without GGUF files stops the lines, rather than give a model without weights.
    (fitted / "weights-directory.txt").write_text(f"{fitted}\n")
    done = run("sh", "-c", block, env=variables, cwd=tmp_path)
    assert done.returncode != 0 and "No *.gguf file in the weights directory" in done.stderr


def test_replies_of_the_float_pair(tmp_path):
    """examples/try_limits.py, which fits limits to replies of the floating-point model and counts
    which other replies meet them.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")
    top = tmp_path / "new"
    assert make_example("small_float_pair", "--directory", top).returncode == 0
    start = (sys.executable, PACKAGE / "examples" / "try_limits.py")
    # In float32, in which every CPU multiplies fast; the script's default is bfloat16. The test
    # asserts ranges of the counts, so it is not tied to one type.
    output, said = fit(top, "--replies", "64", "--dtype", "float32", start=start)
    for half in ("fitted", "held_out"):
        assert len(list((top / "replies" / half).iterdir())) == 32
    # replies/ and its two folders are not writable by group or others, whatever the umask.
    for folder in (top / "replies", top / "replies" / "fitted", top / "replies" / "held_out"):
        assert folder.stat().st_mode & 0o022 == 0
    assert [line.split()[0] for line in output[:3]] == ["C", "theta", "cap"]
    assert "replies read: 32; left out: 0" in said  # fit_limits.py's own lines
    counts = re.findall(r"^(.+): (\d+) of (\d+) meet the limits$", "\n".join(output[3:]), re.M)
    assert len(counts) == len(output) - 3
    kinds = ("a random id", "the floating-point model's second choice")
    assert [count[0] for count in counts] == [
        "held-out replies",
        "each after the next reply's prompt",
        *(f"{n} of the tokens replaced by {kind}" for n in (1, 4, 16) for kind in kinds),
        "replies of a nearby model, which applies the router's score twice",
        "the integer model's own greedy replies",
    ]
    met, of = ([int(count[i]) for count in counts] for i in (1, 2))
    assert of[:4] == [32] * 4 and of[-1] == 8 and all(28 <= n <= 32 for n in of[:-1])
    assert met[0] >= 28 and met[1] <= 2 and met[6:8] == [0, 0] and met[-1] == 8
    # Many replies of the nearby model meet the limits: this is a limit of the policy. The script
    # itself stops if that model's replies are the floating-point model's own.
    assert met[1] < met[8] <= met[0]
    assert max(met[2:4]) < met[0] and max(met[4:6]) <= max(met[2:4])
    # A second run on the same directory overwrites nothing, and a directory without the example
    # is rejected.
    assert "exists already" in fit(top, status=2, start=start)[1][-1]
    assert fit(tmp_path, status=1, start=start)[1][-1].startswith("No counts: ")


# A transformers.py that loads no model, and records in the file $SEEN how it was asked for one.
STAND_IN = """import json, os, types
class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(path, **options):
        seen = {"path": str(path), "local_files_only": options.get("local_files_only")}
        with open(os.environ["SEEN"], "x") as f:
            json.dump(seen, f)
        if os.environ["STAND_IN"] == "loads nothing":
            raise OSError("the stand-in loads nothing")
        # A model in whose layers nothing is called router.
        layer = types.SimpleNamespace(feed_forward=types.SimpleNamespace(experts=None))
        new = types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer] * 8))
        new.eval = lambda: new
        return new
"""


def try_limits_with_a_stand_in(tmp_path, kind):
    """Runs examples/try_limits.py with STAND_IN as transformers, on the example small_float_pair,
    named by a relative path. The run must fail before it writes a reply. Returns the last line of
    standard error and what STAND_IN recorded.
    """
    pytest.importorskip("safetensors")
    assert make_example("small_float_pair", "--directory", tmp_path / "new").returncode == 0
    (tmp_path / "stand_in").mkdir()
    (tmp_path / "stand_in" / "transformers.py").write_text(STAND_IN)
    path = os.pathsep.join([str(tmp_path / "stand_in"), str(PACKAGE)])
    seen = tmp_path / "seen.json"
    variables = {**os.environ, "PYTHONPATH": path, "SEEN": str(seen), "STAND_IN": kind}
    script = PACKAGE / "examples" / "try_limits.py"
    done = run(sys.executable, script, "new", env=variables, cwd=tmp_path)
    assert done.returncode == 1 and done.stdout == "", done.stderr
    assert not (tmp_path / "new" / "replies").exists()
    return done.stderr.splitlines()[-1], json.loads(seen.read_text())


def test_try_limits_asks_no_model_hub(tmp_path):
    """The script asks for the floating-point model with local_files_only, and by an absolute path
    that exists, which no library can take for a model's name on a hub.
    """
    last, asked = try_limits_with_a_stand_in(tmp_path, "loads nothing")
    assert last == "No counts: the stand-in loads nothing"
    assert asked == {
        "path": str((tmp_path / "new" / "float_model").resolve()),
        "local_files_only": True,
    }


def test_try_limits_stops_if_it_cannot_make_the_nearby_model(tmp_path):
    """Under a version of transformers that names no part of a layer `router`, the script could
    register no hook, and would count the floating-point model's own replies as the nearby model's.
    """
    last, _ = try_limits_with_a_stand_in(tmp_path, "other names")
    assert last.startswith("No counts: this version of transformers names the parts of a layer")
