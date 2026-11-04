# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""IntegerModel against the values that a second implementation computed.

For two synthetic models, variants of them and several windows, the tests compare
IntegerModel.extend() with the recorded values (recorded_values.py, which states their limits): the
named intermediate values of docs/model.md at every layer and position, the cached keys and values,
the chosen experts, the logits, the greedy choices and the logit gaps. Each window is fed in one
call, one token per call, and in chunks of random lengths. Symbols are those of docs/model.md.

With the examples' scales, no range condition fails in these windows, the clamps of k and v never
take effect, and no two logits tie at the top. test_arithmetic_edges.py reaches those cases.

On a window of random tokens, only test_against_the_prover.py, which needs the GPU programs, checks
the output head against a prover (cp_prove): by the sum of the clipped logit gaps, not gap by gap.
"""

import pathlib
import struct

import numpy
import pytest
import torch

import recorded_values
from example_models import (
    PARAMETER_FILES,
    load_model,
    recorded_forward,
    splits,
    windows,
    with_line,
    with_lines,
)
from integer_model import IntegerModel, WindowFull, arithmetic
from integer_model.parameters import Parameters
from integer_model.synthetic_weights import SyntheticWeights
from integer_model.tables import silu_table, table_digest
from recorded_values import assert_model_equals_recorded, difference

CASES = [
    pytest.param(model_name, window, split, id=f"{model_name}-{window_name}-{split_name}")
    for model_name in PARAMETER_FILES
    for window_name, window in windows(model_name).items()
    for split_name, split in splits(model_name).items()
]


@pytest.mark.parametrize("model_name, window, split", CASES)
def test_every_integer_equals_the_recorded_one(model_name, window, split, device):
    assert_model_equals_recorded(
        load_model(model_name, device), window, split, recorded_forward(model_name, window)
    )


# The interval from which write_other_scales() draws each shift. Each contains the example model's
# value, except that target_small_f2's kd, 9, is below its interval. All are within what
# Parameters.read() accepts, and narrow enough that no range condition fails in
# test_scales_that_differ_in_every_layer().
SCALE_INTERVALS = {
    "tiny_without_clamps": dict(
        kq=(25, 30),
        kk=(26, 31),
        kv=(12, 17),
        ko=(14, 19),
        kg=(14, 19),
        ku=(14, 19),
        kd=(15, 20),
        sh=(4, 12),
        kr=(14, 19),
        kgR=(14, 19),
        kuR=(14, 19),
    ),
    "target_small_f2": dict(
        kq=(25, 32),
        kk=(25, 32),
        kv=(14, 21),
        ko=(5, 12),
        kg=(7, 14),
        ku=(13, 20),
        kd=(10, 17),
        sh=(4, 12),
        kr=(11, 14),
        kgR=(7, 10),
        kuR=(14, 17),
    ),
}


def write_other_scales(
    model_name: str, heads: dict[str, int], folder: pathlib.Path, seed: int
) -> pathlib.Path:
    """Writes to `folder` a parameter file (policy.txt) and quantization scales (KL.bin). Returns
    the parameter file's path.

    Each shift has a different value in every layer, and the 2 L + 1 norms have different pairs
    (M, K). The regularizers eps1, eps2 and epsF differ, and each is at least 2^32, so both of its
    32-bit lines (_lo and _hi) count. The seed of the synthetic weights, the flags rp and the shape
    are those of the example model, except the numbers of query heads and key-value heads (the
    lines H and G), which `heads` may replace.
    """
    random = numpy.random.default_rng(seed)
    p = Parameters.read(PARAMETER_FILES[model_name])
    expert_layers = [layer for layer in range(p.L) if p.has_experts(layer)]
    drawn = {}
    for word, (lowest, highest) in SCALE_INTERVALS[model_name].items():
        count = len(expert_layers) if word in ("kr", "kgR", "kuR") else p.L
        drawn[word] = [
            int(value) for value in random.permutation(numpy.arange(lowest, highest + 1))[:count]
        ]
    norms = [(K + difference, K) for K in range(26, 33) for difference in (12, 13, 14)]
    norms = [norms[i] for i in random.permutation(len(norms))[: 2 * p.L + 1]]
    words = []
    for layer in range(p.L):
        words += [drawn[word][layer] for word in ("kq", "kk", "kv", "ko", "kg", "ku", "kd")]
        words += [*norms[2 * layer], *norms[2 * layer + 1], drawn["sh"][layer], p.layers[layer].rp]
    for number in range(len(expert_layers)):
        words += [drawn[word][number] for word in ("kr", "kgR", "kuR")]
    (folder / "KL.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))

    lines = heads | {"MF": norms[-1][0], "KF": norms[-1][1]}
    for name, high_word in zip(
        ("eps1", "eps2", "epsF"), random.permutation(numpy.arange(200, 2000))
    ):
        lines |= {name + "_lo": int(random.integers(1, 1 << 32)), name + "_hi": int(high_word)}
    (folder / "policy.txt").write_text(with_lines(lines)(PARAMETER_FILES[model_name].read_text()))
    return folder / "policy.txt"


@pytest.mark.parametrize(
    "model_name, heads",
    [
        pytest.param("tiny_without_clamps", {}, id="tiny_without_clamps"),
        pytest.param("target_small_f2", {}, id="target_small_f2"),
        # Five query heads per key-value head, as in target_full_f2, and one. The example models
        # have two. With ten heads the queries are wider than the residual stream: dq = 160, d = 64.
        pytest.param(
            "tiny_without_clamps",
            {"H": 10, "G": 2},
            id="tiny_without_clamps-10_query_heads-2_key_value_heads",
        ),
        pytest.param(
            "tiny_without_clamps",
            {"H": 4, "G": 4},
            id="tiny_without_clamps-4_query_heads-4_key_value_heads",
        ),
    ],
)
def test_scales_that_differ_in_every_layer(model_name, heads, tmp_path, device):
    """In the example models all layers have the same shifts and norm exponents, and every eps is 1.

    With the scales of write_other_scales(), a model that reads a scale of the wrong layer or of the
    wrong norm computes other integers. The smaller shifts also make the clamps of q, k and v take
    effect.
    """
    parameter_file = write_other_scales(model_name, heads, tmp_path, seed=5)
    p = Parameters.read(parameter_file)
    for word in ("kq", "kk", "kv", "ko", "kg", "ku", "kd", "sh"):
        assert len({getattr(k, word) for k in p.layers}) == p.L
    assert (
        len({(k.M1, k.K1) for k in p.layers} | {(k.M2, k.K2) for k in p.layers} | {(p.MF, p.KF)})
        == 2 * p.L + 1
    )
    assert min(p.eps1, p.eps2, p.epsF) >= 1 << 32
    assert len({p.eps1, p.eps2, p.epsF}) == 3

    model = IntegerModel.load(parameter_file, SyntheticWeights(p, device), device)
    window = windows(model_name)["random_0"]
    case = recorded_values.forward(parameter_file, tmp_path / "KL.bin", window)
    arrays = [
        assert_model_equals_recorded(model, window, split, case)
        for split in splits(model_name).values()
    ][0]
    for name in ("qw", "kw", "vw"):
        largest = max(numpy.abs(arrays[(name, layer)]).max() for layer in range(p.L))
        assert largest > 2047, (
            f"No value of {name} exceeds 2,047 in this window, so the test does not exercise its "
            "clamp."
        )


@pytest.mark.parametrize("eg, smax", [(7, 2047), (6, 1023), (5, 511), (4, 255), (7, 255)])
def test_other_silu_tables(eg, smax, tmp_path, device):
    """The lines eg and smax change the SiLU table, and so the model. The example models have
    neither line.

    The parameter file must state the table's digest. The test computes it with this package's own
    functions, so it tests the model's integers, not the digest (test_parameter_file.py tests the
    digest).
    """
    example = PARAMETER_FILES["tiny_without_clamps"]
    text = with_line(example.read_text(), "tsilu0", f"eg {eg}\nsmax {smax}\ntsilu0 0\n")
    for i, word in enumerate(table_digest([entry + 4096 for entry in silu_table(eg, smax)])):
        text = with_line(text, f"tsilu{i}", f"tsilu{i} {word}\n")
    (tmp_path / "policy.txt").write_text(text)
    (tmp_path / "KL.bin").write_bytes((example.parent / "KL.bin").read_bytes())
    p = Parameters.read(tmp_path / "policy.txt")
    assert (p.eg, p.smax) == (eg, smax)
    model = IntegerModel.load(tmp_path / "policy.txt", SyntheticWeights(p, device), device)
    window = windows("tiny_without_clamps")["random_3"]
    case = recorded_values.forward(tmp_path / "policy.txt", tmp_path / "KL.bin", window)
    for split in splits("tiny_without_clamps").values():
        assert_model_equals_recorded(model, window, split, case)
    usual = recorded_forward("tiny_without_clamps", window)
    assert case.arrays[("s", 0)] != usual.arrays[("s", 0)], (
        "The other table has changed no activation in this window."
    )


def test_products_that_are_cut_into_slices(device, monkeypatch):
    """arithmetic.exact_product() converts a large weight matrix to float64 in slices of rows. Here
    every weight matrix but the router's needs several.
    """
    monkeypatch.setattr(arithmetic, "_SLICE_ELEMENTS", 1 << 9)
    window = windows("tiny_without_clamps")["random_1"]
    assert_model_equals_recorded(
        load_model("tiny_without_clamps", device),
        window,
        [len(window)],
        recorded_forward("tiny_without_clamps", window),
    )


@pytest.mark.parametrize("model_name", PARAMETER_FILES)
def test_default_dtype_does_not_matter(model_name, device):
    """A server may load its models with another default floating-point type. The integers must not
    depend on it.
    """
    window = windows(model_name)["random_1"]
    case = recorded_forward(model_name, window)
    parameter_file = PARAMETER_FILES[model_name]
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = IntegerModel.load(
            parameter_file, SyntheticWeights(Parameters.read(parameter_file), device), device
        )
        assert_model_equals_recorded(model, window, splits(model_name)["random_chunks_0"], case)
    finally:
        torch.set_default_dtype(default)


class NarrowEmbeddings(SyntheticWeights):
    """The synthetic weights, with the embedding's rows as 32-bit integers, which hold its 20-bit
    entries.
    """

    def embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        return super().embedding_rows(token_ids).to(torch.int32)


def test_embedding_rows_of_a_narrower_type(device):
    """The residual stream needs 64 bits. A weight source that returns narrower embeddings must not
    make it overflow.
    """
    parameter_file = PARAMETER_FILES["target_small_f2"]
    window = windows("target_small_f2")["random_0"]
    model = IntegerModel.load(
        parameter_file, NarrowEmbeddings(Parameters.read(parameter_file), device), device
    )
    arrays = assert_model_equals_recorded(
        model, window, [len(window)], recorded_forward("target_small_f2", window)
    )
    assert max(numpy.abs(arrays[("x", layer)]).max() for layer in range(1, 8)) >= 1 << 31


@pytest.mark.parametrize("model_name", PARAMETER_FILES)
def test_two_sequences_do_not_disturb_each_other(model_name, device):
    """Two windows are fed in alternating chunks to two sequences of one model."""
    model = load_model(model_name, device)
    two_windows = [windows(model_name)[name] for name in ("random_0", "random_1")]
    split = splits(model_name)["random_chunks_0"]
    sequences = [model.new_sequence(), model.new_sequence()]
    choices = [[], []]
    start = 0
    for length in split:
        for i in (0, 1):
            choices[i].append(
                model.extend(sequences[i], list(two_windows[i][start : start + length]))
            )
        start += length
    for i in (0, 1):
        expected = recorded_forward(model_name, two_windows[i]).whole
        assert expected[("defined", None)], (
            "This test needs two windows at which every range condition holds."
        )
        message = difference(
            "the returned greedy choice",
            None,
            expected[("first_choice", None)],
            torch.cat(choices[i]).numpy(),
        )
        assert message is None, message
        model.free(sequences[i])


@pytest.mark.parametrize("model_name", PARAMETER_FILES)
def test_a_call_past_the_window_is_rejected_and_changes_nothing(model_name, device):
    model = load_model(model_name, device)
    window = windows(model_name)["random_0"]
    T = len(window)
    sequence = model.new_sequence()
    model.extend(sequence, list(window[: T - 2]))
    with pytest.raises(WindowFull, match=f"holds {T} tokens. The sequence has {T - 2}"):
        model.extend(sequence, list(window[T - 2 :]) + [0])
    # The rejected call has fed nothing, so the window's last two tokens still fit.
    choices = model.extend(sequence, list(window[T - 2 :]))
    expected = recorded_forward(model_name, window).whole
    assert expected[("defined", None)], (
        "This test needs a window at which every range condition holds."
    )
    message = difference(
        "the returned greedy choice",
        None,
        expected[("first_choice", None)][T - 2 :],
        choices.numpy(),
    )
    assert message is None, message
    with pytest.raises(WindowFull, match=f"holds {T} tokens. The sequence has {T},"):
        model.extend(sequence, [0])
    model.free(sequence)


def test_synthetic_weights_equal_the_recorded_ones():
    """Every synthetic weight of tiny_without_clamps, with the embedding and the output matrix."""
    parameter_file = PARAMETER_FILES["tiny_without_clamps"]
    p = Parameters.read(parameter_file)
    source = SyntheticWeights(p)
    case = recorded_values.weights(parameter_file, parameter_file.parent / "KL.bin")
    for name, layer in case.arrays:
        if name == "Emb":
            found = source.embedding_rows(torch.arange(p.V)).numpy()
        else:
            matrix_name, _, expert = name.partition(".")
            # The recorded matrices are indexed [input][output], a weight source's [output][input].
            found = (
                source.matrix(matrix_name, layer, int(expert) if expert else None)
                .numpy()
                .astype("int64")
                .T
            )
        message = case.difference(name, layer, found)
        assert message is None, message


def test_token_ids_must_be_integers(device):
    """Floats and booleans among the token ids are rejected, not rounded or read as integers."""
    model = load_model("tiny_without_clamps", device)
    sequence = model.new_sequence()
    expected = model.extend(sequence, [3, 1, 2], gap_for=[1, 2, 0])
    model.free(sequence)
    accepted = [
        (3, 1, 2),
        [numpy.int64(3), numpy.int32(1), 2],
        numpy.array([3, 1, 2]),
        numpy.array([3, 1, 2], dtype=numpy.uint8),
        numpy.array([3, 1, 2], dtype="<u4"),  # the type of the words in a prover's token file
        numpy.array([3, 1, 2], dtype=numpy.uint16),
        numpy.array([3, 1, 2], dtype=numpy.uint64),
        *(
            torch.tensor([3, 1, 2], dtype=dtype)
            for dtype in (torch.uint16, torch.uint32, torch.uint64)
        ),
        *(
            torch.tensor([3, 1, 2], dtype=dtype)
            for dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
        ),
    ]
    for token_ids in accepted:
        sequence = model.new_sequence()
        found = model.extend(sequence, token_ids, gap_for=[1, 2, 0])
        model.free(sequence)
        assert found[0].tolist() == expected[0].tolist()
        assert found[1].tolist() == expected[1].tolist()
    not_integers = [
        [1.5],
        [1, 2.0],
        [True],
        [1, True],
        (False, 2),
        ["a"],
        [None],
        torch.tensor([1.0]),
        torch.tensor([True]),
        numpy.array([1.0]),
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.tensor([1 + 0j]),
    ]
    outside = [
        # As a signed 64-bit integer this would be negative.
        numpy.array([(1 << 63) + 5], dtype=numpy.uint64),
        numpy.array([(1 << 32) - 1], dtype=numpy.uint32),
        [-1],
        [model.parameters.V],
    ]
    not_a_list = [[], [[1, 2]]]
    for reason, rejected in (
        ("must be integers", not_integers),
        ("A token id is outside", outside),
        ("must be a one-dimensional sequence of at least one id", not_a_list),
    ):
        for token_ids in rejected:
            sequence = model.new_sequence()
            with pytest.raises(ValueError, match=reason):
                model.extend(sequence, token_ids)
            # A token per gap_for entry (at least one), so that only gap_for can be at fault.
            with pytest.raises(ValueError, match=reason):
                model.extend(sequence, [1] * max(len(token_ids), 1), gap_for=token_ids)
            assert sequence.length == 0, "A rejected call must feed nothing."
            model.free(sequence)
