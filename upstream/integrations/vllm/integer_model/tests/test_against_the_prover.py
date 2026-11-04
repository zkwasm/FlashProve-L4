# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Compares IntegerModel with the GPU prover: cp_commit, cp_prove and cp_verify.

The programs have no option that prints an intermediate value of the forward pass, so the tests compare whether a
request gets a proof and, if so, the token after its window. A request is a window of T tokens and a range of positions.
cp_prove proves it only if the window's tokens after these positions (the checked tokens) meet the
acceptance policy: the lines C, theta and cap of the parameter file (policy.txt), explained in
integrations/vllm/integer_model/examples/score_reply.py. So IntegerModel says which requests meet a
policy and which miss it by the least amount, and cp_prove must prove exactly those that meet it.

All weights are synthetic. The first two tests take them from a rule that both sides know
(`-weights made-up-blocks`). The third takes them from GGUF files, in four forms: the 4-bit storage
types, Q8_0, F16 and BF16. The build, and these tests' limits:
integrations/vllm/integer_model/tests/README.md, "The comparison with the GPU prover".
"""

import os
import pathlib
import random
import re
import shutil
import struct
import subprocess
import sys

import pytest

from block_tools import TREE, ending
from example_models import with_lines
from integer_model import IntegerModel, Parameters, RangeConditionFailed
from integer_model.block_weights import BlockWeights, GgufBlocks

# cp_commit's worst-case check is meant to rule out that a prompt drives a value of the forward pass
# out of its range. This example passes it; target_small_f2 does not.
EXAMPLE = TREE / "prover" / "examples" / "target_small_f2_roomy"
# The repository's example for a model in files: 256 tokens, 4 layers, exact match. model.txt holds
# the numbers from which make_model (prover/tests/weights_blocks/make_model.cpp) writes the model.
FILE_EXAMPLE = TREE / "prover" / "examples" / "target_small_file_f2"

pytestmark = [pytest.mark.prover, pytest.mark.gpu]


@pytest.fixture(scope="module")
def model() -> IntegerModel:
    """The example's model, on the CPU: cp_commit and cp_prove need the GPU to themselves."""
    parameters = Parameters.read(EXAMPLE / "policy.txt")
    return IntegerModel(parameters, BlockWeights(parameters, "cpu", blocks="synthetic"), "cpu")


# A program that runs longer than TIME_LIMIT seconds is sent no signal: SIGTERM ends cp_commit at
# once, whatever it is computing. The test session ends instead, naming the program and its pid, so
# that no further test starts a program while that one is running. A program's output goes to a
# file, so that the program can go on writing after pytest has ended.
TIME_LIMIT = 3600


class Prover:
    """A commitment to an example's weights, in the new directory `work`. `policy` replaces lines
    of the example's acceptance policy. The policy is part of what is committed to, so each policy
    needs its own commitment. `weights` is what follows cp_commit's option -weights: the words
    made-up-blocks, or a directory with a model's files.
    """

    def __init__(
        self,
        work: pathlib.Path,
        example: pathlib.Path = EXAMPLE,
        weights: str | pathlib.Path = "made-up-blocks",
        **policy: int,
    ):
        self.build = pathlib.Path(os.environ["INTEGER_MODEL_PROVER_BUILD"]).resolve()
        self.work, self.config = work, work / "config"
        for new in (work, self.config, work / "home", work / "tmp", work / "in"):
            new.mkdir()
        shutil.copy(example / "KL.bin", self.config)
        text = with_lines(policy)((example / "policy.txt").read_text())
        (self.config / "policy.txt").write_text(text)
        if (example / "layout.txt").exists():
            shutil.copy(example / "layout.txt", self.config)
        else:
            # FILE_EXAMPLE has no layout file: prover/examples/TARGET-KITS.md says how one is made.
            # A planning call of cp_verify ends with exit code 4.
            (work / "arrays.txt").write_text(
                self.run("cp_verify", "-plan", "arrays", self.config, "128", exit_code=4)
            )
            pack = [sys.executable, "-B", TREE / "prover" / "tools" / "pack_layout.py"]
            packed = subprocess.run(
                [*pack, "--arrays", work / "arrays.txt", "17", "4"],
                check=True,
                capture_output=True,
                text=True,
            )
            (self.config / "layout.txt").write_text(packed.stdout)
        value = {name: int(number) for name, number in re.findall(r"(?m)^(\w+) (\d+)$", text)}
        C, theta, cap, s, sh = (value[name] for name in ("C", "theta", "cap", "s", "sh"))
        # The two kinds below hold for the step table (sT = sh - 4), which the example has.
        assert value["sT"] + 4 == sh
        kind = "exact-match" if C == theta == 1 << sh and cap == 0 else "clipped-gap"
        self.policy = f"{kind},{C},{theta},{cap},{s},{sh}"
        # cp_verify's last line for a proof that covers tokens of the request (verifier/README.md,
        # "Last line and exit status"): YES under exact match, else a WITHIN LIMITS line, which is
        # no YES.
        self.passed = "YES"
        if kind != "exact-match":
            self.passed = f"WITHIN LIMITS: {kind} C={C} theta={theta} cap={cap} s={s} sh={sh}"
        self.commit_said = self.run(
            *("cp_commit", "-weights", weights, "-config", self.config, "-M", "1"),
            *("-ivk", work / "ivk", "-isk", work / "isk"),
        )
        (self.id,) = re.findall(r"(?m)^commitment ID: ([0-9a-f]{64})$", self.commit_said)

    def run(self, program: str, *arguments, exit_code: int = 0) -> str:
        """Runs one of the three programs, requires the exit code, and returns what it printed.

        If the program runs longer than TIME_LIMIT, the test session ends.
        """
        # cp_prove requires that HOME and TMPDIR name existing directories.
        directories = {"HOME": str(self.work / "home"), "TMPDIR": str(self.work / "tmp")}
        output = self.work / f"{program}.out"  # a file, not a pipe: see TIME_LIMIT
        with open(output, "w") as out:
            process = subprocess.Popen(
                [self.build / program, *arguments],
                env=os.environ | directories,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        try:
            returncode = process.wait(timeout=TIME_LIMIT)
        except subprocess.TimeoutExpired:
            pytest.exit(
                f"{program} still runs after {TIME_LIMIT} s (pid {process.pid}); no signal was sent"
                f" to it, and no further test starts. What it printed: {output}. It holds the GPU"
                " until it ends by itself (nvidia-smi shows it); ending a program while its GPU"
                " kernel runs can make the card's driver take the whole machine out of use.",
                returncode=2,
            )
        said = output.read_text()
        assert returncode == exit_code, (
            f"{program} {ending(returncode)}. Last it printed {said.splitlines()[-5:]}."
        )
        return said

    def prove(self, **windows: list[int]) -> dict[str, int | None]:
        """Has cp_prove answer one request for each window, all in one batch. Call it once.

        Each request's range is every position: a proof states that tokens 1 to T - 1 of the window
        meet the policy, and gives the model's greedy choice after the window. The result has that
        token, once cp_verify's last line is self.passed (YES under exact match, the `WITHIN LIMITS:
        ..` line under a clipped-gap policy), or None if cp_prove made no proof.
        """
        for name, window in windows.items():
            T = len(window)
            request = struct.pack(f"<{T + 2}I", *window, 0, T)  # the tokens, then the range
            (self.work / "in" / f"{name}.req").write_bytes(request)
        out = self.work / "out"
        self.run(
            *("cp_prove", "-isk", self.work / "isk", "-in", self.work / "in", "-out", out),
        )
        answers = {}
        for name in windows:
            if (out / f"{name}.failed").exists():
                answers[name] = None
                continue
            said = self.run(
                *("cp_verify", self.config, self.work / "ivk", self.work / "in" / f"{name}.req"),
                *(out / f"{name}.y", out / f"{name}.proof", "-id", self.id, "-policy", self.policy),
            )
            assert said.splitlines()[-1] == self.passed
            (answers[name],) = struct.unpack("<I", (out / f"{name}.y").read_bytes())
        return answers


def greedy_window(model: IntegerModel, first: int) -> tuple[list[int], int]:
    """A window of T tokens that begins with `first` and goes on with the model's greedy choices,
    and the model's choice after the window.
    """
    window, sequence = [first], model.new_sequence()
    try:
        for _ in range(model.parameters.T):
            window.append(int(model.extend(sequence, window[-1:])))
    finally:
        model.free(sequence)
    *window, following = window
    return window, following


def test_greedy_choices_are_proved_under_exact_match_and_one_other_token_is_not(model, tmp_path):
    """The example's policy is exact match: each checked token must be the model's greedy choice."""
    V = model.parameters.V
    window, following = greedy_window(model, random.Random("exact match").randrange(V))
    # Earlier choices do not depend on the last token, so it alone misses the policy.
    changed = window[:-1] + [(window[-1] + 1) % V]
    answers = Prover(tmp_path / "exact").prove(greedy=window, changed=changed)
    assert answers == {"greedy": following, "changed": None}


def test_the_limits_of_a_clipped_gap_policy_are_where_the_gaps_put_them(model, tmp_path):
    """A window of random tokens is proved under the tightest theta and cap that IntegerModel's gaps
    allow, and under neither theta - 1 nor cap - 1.

    A checked token's cost is its logit gap plus 2^sh. The costs, each clipped to C, must sum to at
    most theta times their number, and at most cap may exceed C. So the first two cases place
    cp_prove's sum within T - 1 of IntegerModel's, where a single gap is tens of millions, and the
    first and third fix the number of costs above C.
    """
    T, V = model.parameters.T, model.parameters.V
    draw = random.Random("clipped gap")
    window, sequence = [draw.randrange(V) for _ in range(T)], model.new_sequence()
    choices, gaps = model.extend(sequence, window, gap_for=window[1:] + [0])
    model.free(sequence)
    (sh,) = re.findall(r"(?m)^sh (\d+)$", (EXAMPLE / "policy.txt").read_text())
    costs = sorted(gap + (1 << int(sh)) for gap in gaps.tolist()[:-1])  # no token follows the last
    C = costs[len(costs) * 3 // 4]  # clips about a quarter of them
    cap = sum(cost > C for cost in costs)
    theta = -(-sum(min(cost, C) for cost in costs) // len(costs))
    assert 0 < cap and costs[0] < C
    for name, limits, answer in (
        ("met", {"theta": theta, "cap": cap}, int(choices[-1])),
        ("mean_missed", {"theta": theta - 1, "cap": cap}, None),
        ("count_missed", {"theta": theta, "cap": cap - 1}, None),
    ):
        answers = Prover(tmp_path / name, C=C, **limits).prove(window=window)
        assert answers == {"window": answer}, name


# The forms in which make_model writes FILE_EXAMPLE's model, each with make_model's last arguments.
# Without --types the files hold a mix of Q4_K, Q5_K, Q6_K and F32. With it, the embedding, the head
# and every matrix but the router have the named type, and hold the first form's values, encoded
# anew; that loses precision, so the integers differ from form to form.
FORMS = {
    "4_bit": [],
    "Q8_0": ["--types", "q8_0"],
    "F16": ["--types", "f16"],
    "BF16": ["--types", "bf16"],
}


class FileModels:
    """FILE_EXAMPLE's model in each form: its files, and one greedy window. A form's files and
    window are made when first asked for, and at most one model is loaded at a time.
    """

    def __init__(self, make_model: pathlib.Path, top: pathlib.Path):
        self.make_model, self.top, self.windows = make_model, top, {}

    def directory(self, form: str) -> pathlib.Path:
        """The directory with the form's GGUF files, wsums.txt and fold.txt."""
        directory = self.top / form
        if not directory.exists():
            numbers = (FILE_EXAMPLE / "model.txt").read_text().split()
            subprocess.run(
                [self.make_model, directory, *numbers, *FORMS[form]],
                check=True,
                capture_output=True,
            )
        return directory

    def load(self, form: str) -> IntegerModel:
        """The model on the CPU. Loading compares the SHA-256 of each tensor's integers with
        wsums.txt, and fails if one differs.
        """
        parameters = Parameters.read(FILE_EXAMPLE / "policy.txt")
        directory = self.directory(form)
        files = GgufBlocks(parameters, directory, sorted(directory.glob("*.gguf")))
        try:
            weights = BlockWeights(parameters, "cpu", blocks=files, verify_sums=True)
        finally:
            files.close()
        return IntegerModel(parameters, weights, "cpu")

    def window(self, form: str) -> tuple[list[int], int]:
        """greedy_window() of the form's model. If a range condition fails on the way, cp_prove
        would make no proof, so another first token is drawn, at most twice.
        """
        if form not in self.windows:
            model, draw = self.load(form), random.Random("files")
            for _ in range(3):
                try:
                    self.windows[form] = greedy_window(model, draw.randrange(model.parameters.V))
                    break
                except RangeConditionFailed:
                    continue
            else:
                pytest.fail(f"{form}: a range condition failed in each of three greedy windows.")
        return self.windows[form]


@pytest.fixture(scope="module")
def file_models(programs, tmp_path_factory) -> FileModels:
    return FileModels(programs / "make_model", tmp_path_factory.mktemp("file_models"))


@pytest.mark.parametrize("form", FORMS)
def test_a_model_in_files_has_the_same_choices_here_and_in_the_prover(form, file_models, tmp_path):
    """cp_commit and this package hold each tensor's integers to the same sums file. Under exact
    match cp_prove then proves the window that this package continued greedily, and gives this
    package's next token. It makes no proof if the last token is changed, and none for the greedy
    window of the same model stored in another form. That last request is what depends on how a
    storage type is decoded: every form begins with the same first token.
    """
    window, following = file_models.window(form)
    names = list(FORMS)
    other_form = names[(names.index(form) + 1) % len(names)]
    other, _ = file_models.window(other_form)
    assert other != window, f"The forms {form} and {other_form} give the same window."
    model = file_models.load(form)
    sequence = model.new_sequence()
    _, gaps = model.extend(sequence, other, gap_for=other[1:] + [0])
    model.free(sequence)
    assert gaps[:-1].any()  # by this package, the other form's window misses exact match here
    changed = window[:-1] + [(window[-1] + 1) % model.parameters.V]
    del model
    prover = Prover(tmp_path / "files", FILE_EXAMPLE, file_models.directory(form))
    counts = re.findall(
        r"(?m)^cp_commit: (\d+) of (\d+) tensors' integers have the sums of wsums\.txt$",
        prover.commit_said,
    )
    assert len(counts) == 1 and counts[0][0] == counts[0][1] != "0", counts
    answers = prover.prove(greedy=window, changed=changed, other=other)
    assert answers == {"greedy": following, "changed": None, "other": None}
