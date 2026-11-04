# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of loading a model from GGUF files with the module integer_model.block_weights.

make_model (prover/tests/weights_blocks/make_model.cpp) writes a small model in all four storage
types, with wsums.txt, the checksum file, which lists the SHA-256 of each tensor's integers as the
prover decodes them (docs/model.md, "11.4 From a decoded value to an integer"). With its option
--types it writes the same model with every matrix but the router in Q8_0, F16 or BF16. A model
directory's five small files are integer_model.json, policy.txt and KL.bin, and wsums.txt and
fold.txt in its set-up directory (`setup_folder` in the code; here model/). A `shard` is one of the
GGUF files.

In order: the loaded weights against wsums.txt; the digests given for the five small files; the
module's command line; GGUF headers that make_model does not write; malformed GGUF files, wsums.txt
and fold.txt; then shapes, rows and value ranges that the loader rejects.
"""

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys

import pytest
import torch

from block_tools import (
    FILE_MODEL,
    FILE_MODEL_TENSORS,
    PACKAGE,
    SMALL_EXAMPLE,
    copy_of,
    load_files,
    open_descriptors,
    open_files,
    returned_digests,
    run_main,
    served_folder,
    sums_of,
)
from integer_model import block_weights
from integer_model.block_weights import (
    BF16,
    BLOCK_BYTES,
    F16,
    F32,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    BlockWeights,
    GgufBlocks,
    SyntheticBlocks,
    WeightsRejected,
    all_tensors,
)
from integer_model.model import load_from_folder
from integer_model.parameters import ParameterError, Parameters


def test_every_tensor_of_a_model_in_files_has_the_digest_in_the_sums_file(file_model, file_weights):
    sums = sums_of(file_model)
    assert {name: digest for name, (_, digest) in file_weights.digests.items()} == sums
    assert len(sums) == FILE_MODEL_TENSORS


def test_the_model_in_files_has_all_four_storage_types(file_model):
    source = open_files(file_model)
    tensors = all_tensors(Parameters.read(file_model / "policy.txt"))
    assert {source.storage_type(name, layer) for name, layer in tensors} == {F32, Q4_K, Q5_K, Q6_K}
    source.close()


@pytest.mark.parametrize("type_name", ["Q8_0", "F16", "BF16"])
def test_a_model_in_files_of_a_wider_type_has_the_digests_in_its_sums_file(
    programs, file_model, tmp_path, device, type_name
):
    """The embedding, the head and every matrix but the router have the type, as in a public model
    file of 8 or 16 bits. make_model computes the sums file with the prover's rule (fr::weight).
    """
    shape = [FILE_MODEL[word] for word in ("L", "H", "G", "f", "fe", "NE", "V")]
    subprocess.run(
        [programs / "make_model", tmp_path / "model", *map(str, shape), "30"]
        + ["--types", type_name.lower()],
        check=True,
        capture_output=True,
    )
    parameters = Parameters.read(file_model / "policy.txt")
    source = open_files(file_model, tmp_path / "model")
    types = {key: source.storage_type(*key) for key in all_tensors(parameters)}
    source.close()
    wider = {"Q8_0": Q8_0, "F16": F16, "BF16": BF16}[type_name]
    assert types == {key: F32 if key[0] == "Wr" else wider for key in types}
    assert len({name for name, _ in types}) == 13
    lines = (tmp_path / "model" / "wsums.txt").read_text().splitlines()[:-1]
    sums = {line.split()[0]: line.split()[4] for line in lines}
    assert len(sums) == FILE_MODEL_TENSORS and sums != sums_of(file_model)
    # vLLM loads a model while PyTorch's default floating-point type is the served model's, for
    # example bfloat16.
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        weights = load_files(file_model, device, tmp_path / "model", verify_sums=True)
    finally:
        torch.set_default_dtype(before)
    assert {name: digest for name, (_, digest) in weights.digests.items()} == sums
    assert returned_digests(weights, parameters, device) == sums
    # The last tensor of the second file has the wider type, so this also tests the size that is
    # computed for it.
    second = sorted((tmp_path / "model").glob("*.gguf"))[1]
    os.truncate(second, second.stat().st_size - 64)
    with pytest.raises(WeightsRejected, match="extends past the end of its file"):
        open_files(file_model, tmp_path / "model")


def test_what_is_returned_after_loading_matches_the_sums_file(file_model, file_weights, device):
    """BlockWeights computes its digests while it loads; this test hashes what matrix() and
    embedding_rows() return afterwards.
    """
    assert returned_digests(
        file_weights, Parameters.read(file_model / "policy.txt"), device
    ) == sums_of(file_model)


def test_experts_that_fill_several_slabs_are_returned_unchanged(file_model, device, monkeypatch):
    """The routed experts' blocks live in host memory in slabs, 1 GiB each by default. One expert's
    tensor takes 737,280 or 901,120 bytes, so a slab of 2 MB holds two.
    """
    monkeypatch.setattr(block_weights._HostSlabs, "SLAB_BYTES", 2_000_000)
    weights = load_files(file_model, device)
    assert weights.host_bytes == 12 * 2_000_000
    assert returned_digests(weights, Parameters.read(file_model / "policy.txt"), device) == sums_of(
        file_model
    )
    # Less than one expert's tensor of type Q5_K.
    monkeypatch.setattr(block_weights._HostSlabs, "SLAB_BYTES", 800_000)
    with pytest.raises(
        WeightsRejected,
        match="blk.1.ffn_down_exps.weight: One expert's tensor takes more than 800,000 bytes",
    ) as failure:
        load_files(file_model, device)
    # The message gives the limit, not the size that the file's header gives.
    assert "901,120" not in str(failure.value)


def test_embedding_rows_come_in_the_order_of_the_token_ids(file_weights, device):
    every_row = file_weights.embedding_rows(torch.arange(FILE_MODEL["V"], device=device))
    token_ids = torch.tensor([95, 0, 7, 7, 31], device=device)
    assert torch.equal(file_weights.embedding_rows(token_ids), every_row[token_ids])
    for outside in (-1, FILE_MODEL["V"]):
        with pytest.raises(IndexError, match="A token id is outside"):
            file_weights.embedding_rows(torch.tensor([0, outside], device=device))


def change_one_code(model_folder: pathlib.Path) -> None:
    """Flips one bit of a code in the last tensor of the model's second file,
    blk.7.ffn_down_exps.weight.
    """
    shard = sorted(model_folder.glob("*.gguf"))[1]
    data = bytearray(shard.read_bytes())
    data[-100] ^= 1
    shard.write_bytes(data)


def test_the_five_digests_cover_no_weight_file(file_model, programs, tmp_path, device):
    """The five small files' digests do not cover the GGUF files: another model's files of the same
    shape load with them, unless "verify_sums" is true.
    """
    other = tmp_path / "other"
    shape = [FILE_MODEL[word] for word in ("L", "H", "G", "f", "fe", "NE", "V")]
    seed = 31  # the small model's is 30 (conftest.py)
    subprocess.run(
        [programs / "make_model", other, *map(str, shape), str(seed)],
        check=True,
        capture_output=True,
    )
    for verify_sums in (False, True):
        folder, given = served_folder(
            file_model, tmp_path / str(verify_sums), verify_sums=verify_sums
        )
        own = load_from_folder(folder, device, {"sha256": given})
        for shard in sorted(other.glob("*.gguf")):
            shutil.copy(shard, folder / "model" / shard.name)
        if verify_sums:
            with pytest.raises(WeightsRejected, match="differs from wsums.txt"):
                load_from_folder(folder, device, {"sha256": given})
        else:
            loaded = load_from_folder(folder, device, {"sha256": given})
            assert loaded.weights.files_not_compared
            assert not torch.equal(loaded.weights.matrix("Wo", 0), own.weights.matrix("Wo", 0))


def test_verify_sums_rejects_a_model_that_differs_from_the_sums_file(file_model, tmp_path, device):
    load_files(file_model, device, verify_sums=True)
    model = copy_of(file_model, tmp_path)
    # wsums.txt was made with the embedding's exponent 11.
    (model / "fold.txt").write_text("eres 10\nqkrows asis\n")
    with pytest.raises(WeightsRejected, match="token_embd.weight differs from wsums.txt"):
        load_files(file_model, device, model, verify_sums=True)
    (model / "fold.txt").write_text("eres 11\nqkrows asis\n")
    change_one_code(model)
    with pytest.raises(
        WeightsRejected, match="wsums.txt: blk.7.ffn_down_exps.weight differs from wsums.txt.$"
    ):
        load_files(file_model, device, model, verify_sums=True)
    assert load_files(file_model, device, hash_integers=True).files_not_compared
    with pytest.raises(ValueError, match="verify_sums needs a model in GGUF files"):
        BlockWeights(
            Parameters.read(SMALL_EXAMPLE / "policy.txt"),
            device,
            blocks="synthetic",
            verify_sums=True,
        )


def test_verify_sums_asks_for_the_embeddings_line_before_anything_is_decoded(
    file_model, tmp_path, device, monkeypatch
):
    """Without verify_sums, a checksum file without the embedding's line is accepted."""
    model = copy_of(file_model, tmp_path)
    lines = (model / "wsums.txt").read_text().splitlines()
    kept = [line for line in lines if not line.startswith("token_embd.weight ")]
    kept[-1] = f"END {len(kept) - 1}"
    (model / "wsums.txt").write_text("".join(line + "\n" for line in kept))
    load_files(file_model, device, model)

    def not_decoded(*arguments):
        raise AssertionError("decoding began")

    for method in ("blocks", "multipliers"):
        monkeypatch.setattr(GgufBlocks, method, not_decoded)
    with pytest.raises(WeightsRejected, match="wsums.txt has no line for the tensor token_embd"):
        load_files(file_model, device, model, verify_sums=True)


def with_exponent_0(sums_file: bytes) -> bytes:
    """The checksum file with exponent 0, which the reader accepts, for the first layer's output
    projection.
    """
    lines = [line.split() for line in sums_file.decode().splitlines()]
    changed = [
        line[:2] + ["0"] + line[3:] if line[0] == "blk.0.attn_output.weight" else line
        for line in lines
    ]
    assert changed != lines
    return "".join(" ".join(line) + "\n" for line in changed).encode()


def with_embedding_exponent_10(fold_file: bytes) -> bytes:
    assert b"eres 11" in fold_file
    return fold_file.replace(b"eres 11", b"eres 10")


def test_the_files_of_the_setup_folder_are_compared_with_the_given_digests(
    file_model, tmp_path, device
):
    """With the five files' digests, one matrix equals that of a load without them, and a changed
    file stops the load.
    """
    folder, given = served_folder(file_model, tmp_path)
    plain, compared = (
        load_from_folder(folder, device),
        load_from_folder(folder, device, {"sha256": given}),
    )
    assert torch.equal(compared.weights.matrix("Wo", 0), plain.weights.matrix("Wo", 0))
    for name, change in (("wsums.txt", with_exponent_0), ("fold.txt", with_embedding_exponent_10)):
        whole = (folder / "model" / name).read_bytes()
        (folder / "model" / name).write_bytes(change(whole))
        found = hashlib.sha256(change(whole)).hexdigest()
        with pytest.raises(ParameterError) as failure:
            load_from_folder(folder, device, {"sha256": given})
        assert f"{folder / 'model' / name} is not the file" in str(failure.value)
        assert found not in str(failure.value)
        assert given[name] not in str(failure.value)
        if name == "wsums.txt":
            changed = load_from_folder(folder, device)
            assert not torch.equal(changed.weights.matrix("Wo", 0), plain.weights.matrix("Wo", 0))
        (folder / "model" / name).write_bytes(whole)
    # Another set-up directory. Options that give digests cannot name it, and a rewritten
    # integer_model.json is rejected.
    (folder / "other").mkdir()
    (folder / "other" / "wsums.txt").write_bytes(
        with_exponent_0((folder / "model" / "wsums.txt").read_bytes())
    )
    shutil.copy(folder / "model" / "fold.txt", folder / "other")
    description = json.loads((folder / "integer_model.json").read_text())
    description["weight_settings"]["setup_folder"] = "other"
    with pytest.raises(ParameterError, match='with "sha256", no other option is allowed'):
        load_from_folder(folder, device, {"sha256": given, **description})
    (folder / "integer_model.json").write_text(json.dumps(description))
    with pytest.raises(ParameterError, match="integer_model.json is not the file"):
        load_from_folder(folder, device, {"sha256": given})
    # The digests are compared with the files that are read, whichever set-up directory the
    # description names.
    rewritten = hashlib.sha256((folder / "integer_model.json").read_bytes()).hexdigest()
    with pytest.raises(ParameterError, match="other/wsums.txt is not the file"):
        load_from_folder(folder, device, {"sha256": {**given, "integer_model.json": rewritten}})


@pytest.mark.parametrize("verify_sums", (False, True))
def test_each_of_the_five_files_is_compared_beside_gguf_files(
    file_model, tmp_path, device, verify_sums
):
    """Each file in turn is changed, emptied, and replaced by a symbolic link to a changed copy. The
    digest stops each load.
    """
    folder, given = served_folder(file_model, tmp_path, verify_sums=verify_sums)
    changes = {
        "integer_model.json": lambda whole: whole + b" ",
        "policy.txt": lambda whole: whole.replace(b"\neps1_lo 1\n", b"\neps1_lo 2\n"),
        # Replaces the first layer's first shift, 29, by 28 (0x1c).
        "KL.bin": lambda whole: b"\x1c" + whole[1:],
        "model/wsums.txt": with_exponent_0,
        "model/fold.txt": with_embedding_exponent_10,
    }
    for number, (name, change) in enumerate(changes.items()):
        file, changed, copy = (
            folder / name,
            tmp_path / f"changed_{number}",
            tmp_path / f"with_a_link_{number}",
        )
        whole = file.read_bytes()
        changed.write_bytes(change(whole))
        assert changed.read_bytes() != whole
        for other in (changed.read_bytes(), b""):
            file.write_bytes(other)
            with pytest.raises(
                ParameterError, match=f"{file} is not the file whose SHA-256 was given"
            ):
                load_from_folder(folder, device, {"sha256": given})
        file.write_bytes(whole)
        # Hard links take no disk space. No two files have the same name, so the pattern leaves out
        # only this file.
        shutil.copytree(
            folder, copy, ignore=shutil.ignore_patterns(file.name), copy_function=os.link
        )
        (copy / name).symlink_to(changed)
        with pytest.raises(
            ParameterError, match=f"{copy / name} is not the file whose SHA-256 was given"
        ):
            load_from_folder(copy, device, {"sha256": given})
    assert load_from_folder(folder, device, {"sha256": given}).weights.files_not_compared == (
        not verify_sums
    )


def test_the_digests_of_the_setup_folder_are_given_with_gguf_files_and_only_then(
    file_model, tmp_path, device
):
    folder, given = served_folder(file_model, tmp_path)
    three = {name: given[name] for name in ("integer_model.json", "policy.txt", "KL.bin")}
    with pytest.raises(
        ParameterError, match='the weights are in GGUF files, so "sha256" must also have'
    ):
        load_from_folder(folder, device, {"sha256": three})
    with pytest.raises(
        ParameterError, match='and for weights in GGUF files also "wsums.txt", "fold.txt"\\.'
    ):
        load_from_folder(folder, device, {"sha256": {**three, "fold.txt": given["fold.txt"]}})
    parameters, shards = Parameters.read(folder / "policy.txt"), sorted(folder.glob("model/*.gguf"))
    with pytest.raises(ParameterError, match='entries: "wsums.txt", "fold.txt"\\.'):
        GgufBlocks(parameters, folder / "model", shards, given)
    with pytest.raises(ValueError, match='sha256 needs blocks = "gguf"'):
        BlockWeights(
            Parameters.read(SMALL_EXAMPLE / "policy.txt"), device, blocks="synthetic", sha256={}
        )
    with pytest.raises(ValueError, match='sha256 needs blocks = "gguf"'):
        load_files(
            file_model, device, sha256={name: given[name] for name in ("wsums.txt", "fold.txt")}
        )
    description = json.loads((folder / "integer_model.json").read_text())
    description["weight_settings"]["sha256"] = {
        name: given[name] for name in ("wsums.txt", "fold.txt")
    }
    (folder / "integer_model.json").write_text(json.dumps(description))
    with pytest.raises(ParameterError, match='"weight_settings" has no entry "sha256"'):
        load_from_folder(folder, device)
    # Synthetic blocks, here at the small example's shape, are of the kind "blocks" too, but have no
    # set-up directory.
    for name in ("policy.txt", "KL.bin"):  # copyfile: writable, whatever the files' modes
        shutil.copyfile(SMALL_EXAMPLE / name, folder / name)
    (folder / "integer_model.json").write_text(
        '{"weights": "blocks", "weight_settings": {"blocks": "synthetic"}}'
    )
    three = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in three}
    with pytest.raises(
        ParameterError, match='the weights are not in GGUF files, so "sha256" must not have'
    ):
        load_from_folder(folder, device, {"sha256": {**given, **three}})
    assert not load_from_folder(folder, device, {"sha256": three}).weights.files_not_compared


def test_without_digests_a_warning_names_the_paths_that_leave_the_model_directory(
    file_model, tmp_path, device, caplog
):
    """A symbolic link counts by its target. No warning with digests or if every path is inside."""

    def warnings(folder, options=None) -> list[str]:
        caplog.clear()
        load_from_folder(folder, device, options)
        return [
            record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
        ]

    folder, given = served_folder(file_model, tmp_path)
    assert warnings(folder) == []
    description = json.loads((folder / "integer_model.json").read_text())
    first, second = description["weight_settings"]["shards"]
    name = pathlib.Path(second).name
    (tmp_path / "elsewhere").mkdir()
    shutil.copy(folder / second, tmp_path / "elsewhere" / name)
    (folder / "link").symlink_to(tmp_path / "elsewhere")
    for path in (f"../elsewhere/{name}", f"link/{name}"):
        description["weight_settings"]["shards"] = [first, path]
        text = json.dumps(description)
        (folder / "integer_model.json").write_text(text)
        (message,) = warnings(folder)
        assert f"lead out of the model directory {folder}: {path!r}. " in message
        digest = hashlib.sha256(text.encode()).hexdigest()
        assert warnings(folder, {"sha256": {**given, "integer_model.json": digest}}) == []


def test_each_of_the_five_files_is_read_once(file_model, tmp_path, device, monkeypatch):
    """Each file holds its bytes only during the package's call of Path.read_bytes and is empty
    otherwise, so a load that parsed any other reading would fail.
    """
    folder, given = served_folder(file_model, tmp_path)
    names = ("integer_model.json", "policy.txt", "KL.bin", "model/wsums.txt", "model/fold.txt")
    files = [folder / name for name in names]
    right = {file: file.read_bytes() for file in files}
    read_bytes, readings = pathlib.Path.read_bytes, []

    def read_the_right_bytes(path):
        readings.append(path.name)
        path.write_bytes(right[path])
        try:
            return read_bytes(path)
        finally:
            path.write_bytes(b"")

    for file in files:
        file.write_bytes(b"")
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_the_right_bytes)
    load_from_folder(folder, device, {"sha256": given})
    assert readings == ["integer_model.json", "policy.txt", "KL.bin", "wsums.txt", "fold.txt"]


def test_digest_faults_names_every_kind_of_difference():
    digests = {"a": (256, "1" * 64), "b": (256, "2" * 64), "c": (256, "3" * 64)}
    sums = {name: (elements, 12, digest) for name, (elements, digest) in digests.items()}
    assert block_weights.digest_faults(digests, sums) == []
    exchanged = sums | {"a": sums["b"], "b": sums["a"]}
    assert block_weights.digest_faults(digests, exchanged) == [
        "a differs from wsums.txt",
        "b differs from wsums.txt",
    ]
    del sums["c"]
    assert block_weights.digest_faults(digests, sums | {"d": (256, 12, "3" * 64)}) == [
        "c has no line in wsums.txt",
        "line 3 of wsums.txt lists a tensor that is not a committed tensor of the model",
    ]


@pytest.mark.skipif(
    not pathlib.Path("/proc/self/fd").is_dir(), reason="this system has no /proc/self/fd"
)
def test_files_that_the_loader_opened_are_closed_again(file_model, tmp_path, device):
    model = copy_of(file_model, tmp_path)
    settings = {
        "blocks": "gguf",
        "setup_folder": model,
        "shards": [str(path) for path in sorted(model.glob("*.gguf"))],
    }
    before = open_descriptors()
    BlockWeights(Parameters.read(file_model / "policy.txt"), device, **settings)
    assert open_descriptors() == before
    change_one_code(model)
    with pytest.raises(WeightsRejected, match="blk.7.ffn_down_exps.weight differs from wsums.txt"):
        BlockWeights(
            Parameters.read(file_model / "policy.txt"), device, **settings, verify_sums=True
        )
    assert open_descriptors() == before


def test_the_program_reports_a_match_a_mismatch_and_a_model_that_cannot_be_loaded(
    file_model, tmp_path, device, monkeypatch, capsys
):
    command = [
        sys.executable,
        "-m",
        "integer_model.block_weights",
        file_model / "policy.txt",
        "--device",
        device,
    ]
    command += ["--fold", file_model / "model"]
    for shard in sorted((file_model / "model").glob("*.gguf")):
        command += ["--shard", shard]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(command, cwd=PACKAGE, env=environment, capture_output=True, text=True)
    assert (done.returncode, done.stdout) == (
        0,
        "MATCH: all 74 tensors have the SHA-256 listed in wsums.txt.\n",
    )

    def run(device=device, **arguments):
        return run_main(monkeypatch, capsys, file_model / "policy.txt", device, **arguments)

    model = copy_of(file_model, tmp_path)
    shards = sorted(model.glob("*.gguf"))
    change_one_code(model)
    status, out, _ = run(fold=model, shards=shards)
    assert status == 1
    assert out.startswith("MISMATCH")
    assert "blk.7.ffn_down_exps.weight differs from wsums.txt\n" in out

    # A model that cannot be loaded gives one line on stderr and exit code 2, not a mismatch's 1.
    status, out, err = run(fold=model, shards=shards, device="no_such_device")
    assert (status, out) == (2, "")
    assert err.startswith("error: RuntimeError: ")
    assert err.count("\n") == 1
    status, out, err = run(fold=model, shards=[shards[0], model / "absent.gguf"])
    assert (status, out) == (2, "")
    assert err.startswith("error: ")
    assert "absent.gguf" in err
    shards[1].write_bytes(shards[1].read_bytes()[:-64])
    status, out, err = run(fold=model, shards=shards)
    assert (status, out) == (2, "")
    assert err.startswith("error: blk.7.ffn_down_exps.weight: the tensor extends past")
    for arguments in ({"shards": shards}, {"fold": model}):
        status, out, err = run(**arguments)
        assert (status, out) == (2, "")
        assert "--fold and --shard go together" in err

    status, out, _ = run_main(monkeypatch, capsys, SMALL_EXAMPLE / "policy.txt", device)
    assert status == 0
    assert re.fullmatch(
        "74 tensors. SHA-256 over the lines that --lines prints: [0-9a-f]{64}\n", out
    )


# GGUF headers that make_model does not write.


def gguf_string(text: str) -> bytes:
    return struct.pack("<Q", len(text.encode())) + text.encode()


def gguf_array(element_type: int, count: int, elements: bytes) -> bytes:
    return struct.pack("<IQ", element_type, count) + elements


def take_apart(data: bytes) -> tuple[list, list, bytes]:
    """Splits a file that make_model wrote into metadata pairs, tensor table and tensor data.

    A pair is [key, value type, the value's bytes]. make_model writes values of three types only: 4
    (u32), 6 (f32) and 8 (string). An entry of the table is [name, list of dimensions, storage type,
    offset]. The first 24 bytes are "GGUF" (0x46554747), the version, 3, and the two counts.
    """
    tensor_count, pair_count = struct.unpack_from("<QQ", data, 8)
    at = 24

    def take(code):
        nonlocal at
        values = struct.unpack_from("<" + code, data, at)
        at += struct.calcsize("<" + code)
        return values

    def take_string():
        nonlocal at
        (length,) = take("Q")
        at += length
        return data[at - length : at].decode()

    pairs, tensors = [], []
    for _ in range(pair_count):
        key, (value_type,) = take_string(), take("I")
        start = at
        take_string() if value_type == 8 else take("I")
        pairs.append([key, value_type, data[start:at]])
    for _ in range(tensor_count):
        name, (axes,) = take_string(), take("I")
        dims, (storage_type, offset) = list(take(f"{axes}Q")), take("IQ")
        tensors.append([name, dims, storage_type, offset])
    return pairs, tensors, data[(at + 31) // 32 * 32 :]


def put_together(pairs: list, tensors: list, tensor_data: bytes, alignment: int = 32) -> bytes:
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(pairs))
    for key, value_type, value in pairs:
        header += gguf_string(key) + struct.pack("<I", value_type) + value
    for name, dims, storage_type, offset in tensors:
        header += gguf_string(name) + struct.pack(
            f"<I{len(dims)}QIQ", len(dims), *dims, storage_type, offset
        )
    return header + bytes(-len(header) % alignment) + tensor_data


def test_a_header_as_public_models_have_it_is_read(file_model, tmp_path, device):
    """Public models have arrays (the tokenizer's) and more value types in their metadata, usually
    omit the alignment and the head dimension, and state the model's metadata in the first file
    only. A file may also hold a tensor that the model does not use. The one here has the type Q8_0
    and a row of 96 elements, which is whole in the format's blocks of 32 elements but not in blocks
    of 256.
    """
    model = copy_of(file_model, tmp_path)
    first, second = sorted(model.glob("*.gguf"))
    pairs, tensors, tensor_data = take_apart(first.read_bytes())
    assert put_together(pairs, tensors, tensor_data) == first.read_bytes()
    more = [
        [
            "tokenizer.ggml.tokens",
            9,
            gguf_array(8, 3, gguf_string("a") + gguf_string("") + gguf_string("xyz")),
        ],
        ["tokenizer.ggml.token_type", 9, gguf_array(5, 5, struct.pack("<5i", 1, 2, 3, -1, 0))],
        ["tokenizer.ggml.scores", 9, gguf_array(6, 3, struct.pack("<3f", 0.5, -1.0, 0.0))],
        ["an.empty.array", 9, gguf_array(4, 0, b"")],
        ["a.bool", 7, b"\x01"],
        ["an.i8", 1, b"\xff"],
        ["a.u16", 2, struct.pack("<H", 7)],
        ["an.i64", 11, struct.pack("<q", -7)],
        ["a.u64", 10, struct.pack("<Q", 7)],
        ["an.f64", 12, struct.pack("<d", 0.7)],
    ]
    kept = [
        pair
        for pair in pairs
        if pair[0] not in ("general.alignment", "llama4.attention.key_length")
    ]
    unused = ["unused", [96], Q8_0, 0]
    first.write_bytes(put_together(more + kept, tensors + [unused], tensor_data))
    _, tensors, tensor_data = take_apart(second.read_bytes())
    second.write_bytes(put_together([["split.no", 2, struct.pack("<H", 1)]], tensors, tensor_data))
    load_files(file_model, device, model, verify_sums=True)


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def entry(entries: list, name: str) -> list:
    return next(entry for entry in entries if entry[0] == name)


# A fault is a pair: the short name of the file to damage ("first" and "second" are the GGUF files),
# and a function from all four files' bytes, by short name, to that file's faulty bytes.


def header(change):
    """A fault in the first file's header. change alters the metadata pairs and the tensor table in
    place.
    """

    def faulty(whole):
        pairs, tensors, tensor_data = take_apart(whole["first"])
        change(pairs, tensors)
        return put_together(pairs, tensors, tensor_data)

    return "first", faulty


def new_pair(value_type: int, value: bytes):
    return header(lambda pairs, tensors: pairs.insert(0, ["a", value_type, value]))


def other_pair(key: str, value_type: int, value: bytes):
    def change(pairs, tensors):
        entry(pairs, key)[1:] = [value_type, value]

    return header(change)


def new_tensor(dims: list[int], storage_type: int):
    return header(lambda pairs, tensors: tensors.append(["unused", dims, storage_type, 0]))


def other_tensor(name: str, dims=None, storage_type=None, offset=None):
    def change(pairs, tensors):
        tensor = entry(tensors, name)
        tensor[1:] = [
            new if new is not None else old
            for old, new in zip(tensor[1:], (dims, storage_type, offset))
        ]

    return header(change)


def move_second_tensor(pairs, tensors):
    tensors[1][3] += 8


def give_second_tensor_five_axes(pairs, tensors):
    tensors[1][1] = [5120, 1, 1, 1, 1]


def sums(change):
    """A fault in wsums.txt. change gets the file's lines and, of the first line, the name, the
    number of elements, the exponent and the rest (the width in bits and the SHA-256). It returns
    the new lines.
    """

    def faulty(whole):
        lines = whole["wsums.txt"].decode().splitlines()
        name, elements, exponent, rest = lines[0].split(maxsplit=3)
        assert len(exponent) == 2
        return "\n".join(change(lines, name, elements, exponent, rest)).encode()

    return "wsums.txt", faulty


def fold(text: bytes):
    return "fold.txt", lambda whole: text


NORM = "blk.0.attn_norm.weight"
NO_ARCHITECTURE = "do not name the model's architecture"
TOO_MANY = r"more than 2\^20 tensors or metadata pairs"
NO_ALIGNMENT = "general.alignment is not a power of two"
FOLD_LINES = "fold.txt must have the lines"
# Written into wsums.txt by two faults below. No message may repeat what the loader read there.
PLANTED_NAME, PLANTED_SIZE = "planted_tensor_name", 987654
MALFORMED_FILES = {
    "other_first_bytes": (
        "does not start with the bytes GGUF",
        "first",
        lambda whole: b"GGUG" + whole["first"][4:],
    ),
    "version_2": (
        "not a GGUF file of version 3",
        "first",
        lambda whole: whole["first"][:4] + u32(2) + whole["first"][8:],
    ),
    "too_many_tensors": (
        TOO_MANY,
        "first",
        lambda whole: whole["first"][:8] + u64(2**20 + 1) + whole["first"][16:],
    ),
    "too_many_pairs": (
        TOO_MANY,
        "first",
        lambda whole: whole["first"][:16] + u64(2**20 + 1) + whole["first"][24:],
    ),
    "cut_in_the_header": ("ends inside its header", "first", lambda whole: whole["first"][:1000]),
    "cut_in_the_last_tensor": (
        "extends past the end of its file",
        "second",
        lambda whole: whole["second"][:-64],
    ),
    "the_same_file_twice": (
        "an earlier file has the same name",
        "second",
        lambda whole: whole["first"],
    ),
    "no_architecture": (NO_ARCHITECTURE, *header(lambda pairs, tensors: pairs.pop(0))),
    "architecture_is_a_number": (NO_ARCHITECTURE, *other_pair("general.architecture", 4, u32(7))),
    "architecture_is_empty": (
        NO_ARCHITECTURE,
        *other_pair("general.architecture", 8, gguf_string("")),
    ),
    "unknown_value_type": ("has an unknown type", *new_pair(13, u32(0))),
    "string_too_long": (r"longer than 2\^20 bytes", *new_pair(8, u64(2**20 + 1) + b"x")),
    "array_of_2_to_the_60_u64": (
        "ends inside its header",
        *new_pair(9, gguf_array(10, 2**60, b"")),
    ),
    "array_of_2_to_the_64_less_1_u8": (
        "ends inside its header",
        *new_pair(9, gguf_array(0, 2**64 - 1, b"")),
    ),
    # The bytes that follow are read as strings until one of them cannot be one.
    "array_of_2_to_the_63_strings": (
        r"ends inside its header|longer than 2\^20 bytes",
        *new_pair(9, gguf_array(8, 2**63, b"")),
    ),
    "alignment_48": (NO_ALIGNMENT, *other_pair("general.alignment", 4, u32(48))),
    "alignment_0": (NO_ALIGNMENT, *other_pair("general.alignment", 4, u32(0))),
    "alignment_is_a_string": (NO_ALIGNMENT, *other_pair("general.alignment", 8, gguf_string("32"))),
    "tensor_twice": (
        "an earlier entry has the same name",
        *header(lambda pairs, tensors: tensors.append(list(tensors[0]))),
    ),
    "offset_not_aligned": ("not a multiple of the alignment", *header(move_second_tensor)),
    "axis_of_length_0": (r"is 0 or longer than 2\^30", *new_tensor([5120, 0], F32)),
    "axis_too_long": (r"is 0 or longer than 2\^30", *new_tensor([5120, 2**30 + 1], F32)),
    "five_axes": ("does not have 1 to 4 axes", *header(give_second_tensor_five_axes)),
    "row_of_100_elements": ("not a multiple of 256 long", *new_tensor([100], Q4_K)),
    "row_of_100_elements_of_q8_0": ("not a multiple of 32 long", *new_tensor([100], Q8_0)),
    "unsupported_storage_type": (
        "the storage type is not supported",
        *other_tensor("blk.0.attn_q.weight", storage_type=2),  # 2 is the format's type Q4_0
    ),
    "norm_of_another_type": (
        NORM + " must be a tensor of 5,120",
        *other_tensor(NORM, storage_type=1),
    ),
    "norm_of_another_length": (
        NORM + " must be a tensor of 5,120",
        *other_tensor(NORM, dims=[2560]),
    ),
    "norm_past_the_end": (NORM + ": the tensor extends past", *other_tensor(NORM, offset=2**40)),
    "no_final_norm": (
        "output_norm.weight must be a tensor",
        *header(lambda pairs, tensors: tensors.remove(entry(tensors, "output_norm.weight"))),
    ),
    "sums_without_END": ("last line must be END", *sums(lambda lines, *first: lines[:-1])),
    "sums_END_counts_one_too_many": (
        "last line must be END",
        *sums(lambda lines, *first: lines[1:]),
    ),
    "sums_lack_a_tensor": (
        "wsums.txt has no line for the tensor ",
        *sums(lambda lines, *first: lines[1:-1] + ["END 73"]),
    ),
    "sums_list_a_tensor_twice": (
        "an earlier line lists the same tensor",
        *sums(lambda lines, *first: [lines[0]] + lines),
    ),
    "sums_list_an_unknown_tensor": (
        "the model's files have no tensor of this name",
        *sums(
            lambda lines, name, elements, exponent, rest: [
                f"{PLANTED_NAME} {elements} {exponent} {rest}"
            ]
            + lines[1:]
        ),
    ),
    "sums_state_another_size": (
        "the number of elements is not that of the tensor",
        *sums(
            lambda lines, name, elements, exponent, rest: [
                f"{name} {PLANTED_SIZE} {exponent} {rest}"
            ]
            + lines[1:]
        ),
    ),
    # Python's int() would read a number with an underscore as the same number.
    "sums_underscore_in_the_size": (
        "must have five fields",
        *sums(
            lambda lines, name, elements, exponent, rest: [
                f"{name} {elements[0]}_{elements[1:]} {exponent} {rest}"
            ]
            + lines[1:]
        ),
    ),
    "sums_underscore_in_the_exponent": (
        "must have five fields",
        *sums(
            lambda lines, name, elements, exponent, rest: [
                f"{name} {elements} {exponent[0]}_{exponent[1:]} {rest}"
            ]
            + lines[1:]
        ),
    ),
    "sums_are_not_text": (
        "have no tensor",
        "wsums.txt",
        lambda whole: b"\xff\xfe" + whole["wsums.txt"],
    ),
    "fold_lacks_a_line": (FOLD_LINES, *fold(b"eres 11\n")),
    "fold_has_a_line_twice": (FOLD_LINES, *fold(b"eres 11\neres 11\nqkrows asis\n")),
    "fold_exponent_is_no_number": (FOLD_LINES, *fold(b"eres x\nqkrows asis\n")),
    "fold_underscore_in_the_exponent": (FOLD_LINES, *fold(b"eres 1_1\nqkrows asis\n")),
    "fold_is_not_text": (FOLD_LINES, *fold(b"\xff eres 11\nqkrows asis\n")),
    # cp_commit and cp_prove cut a word of fold.txt after 63 bytes and end it at a zero byte. In the
    # next entry they would read exponent 11; a reader that takes 64 bytes as one word would read 9.
    "fold_word_of_64_bytes": (
        FOLD_LINES,
        *fold(b"qkrows asis " + b"x" * 64 + b" eres 11 z eres 9\n"),
    ),
    "fold_zero_byte_in_a_name": (FOLD_LINES, *fold(b"eres\x00x 11\nqkrows asis\n")),
    "fold_zero_byte_in_a_value": (FOLD_LINES, *fold(b"eres 11\x009\nqkrows asis\n")),
}


@pytest.fixture(scope="module")
def files_to_change(file_model, tmp_path_factory):
    """A copy of the model in files: its directory, the paths of its four files by short name, and
    their bytes.
    """
    model = copy_of(file_model, tmp_path_factory.mktemp("files_to_change"))
    first, second = sorted(model.glob("*.gguf"))
    paths = {
        "first": first,
        "second": second,
        "wsums.txt": model / "wsums.txt",
        "fold.txt": model / "fold.txt",
    }
    open_files(file_model, model).close()  # unchanged, the copy is accepted
    return model, paths, {name: path.read_bytes() for name, path in paths.items()}


@pytest.mark.parametrize("fault", MALFORMED_FILES)
def test_a_malformed_file_is_rejected(fault, file_model, files_to_change):
    pattern, name, faulty = MALFORMED_FILES[fault]
    model, paths, whole = files_to_change
    paths[name].write_bytes(faulty(whole))
    try:
        with pytest.raises(WeightsRejected, match=pattern) as failure:
            open_files(file_model, model)
    finally:
        paths[name].write_bytes(whole[name])
    for planted in (PLANTED_NAME, str(PLANTED_SIZE), f"{PLANTED_SIZE:,}"):
        assert planted not in str(failure.value)


# cp_commit and cp_prove split only at ASCII's six white-space characters. str.split also splits at
# the bytes in the first three texts, so it would pair their words differently and read exponent 9.
@pytest.mark.parametrize(
    "text",
    [
        b"qkrows asis x\x1cy eres 9 z eres 11\n",
        b"qkrows asis x\xc2\xa0y eres 9 z eres 11\n",
        b"qkrows asis x\xc2\x85y eres 9 z eres 11\n",
        b"a " + b"x" * 63 + b"\neres 11\nqkrows asis\n",
        b"eres\x0b11\x0cqkrows\tasis\r\n",
        b"eres 11 qkrows asis x",
    ],
    ids=[
        "x1c",
        "no_break_space",
        "next_line",
        "word_of_63_bytes",
        "all_six_white_spaces",
        "odd_word_at_the_end",
    ],
)
def test_fold_txt_is_cut_into_words_as_the_provers_cut_it(text, file_model, files_to_change):
    model, paths, whole = files_to_change
    paths["fold.txt"].write_bytes(text)
    try:
        files = open_files(file_model, model)
        assert files.exponent("Emb", None) == 11
        files.close()
    finally:
        paths["fold.txt"].write_bytes(whole["fold.txt"])


def test_a_message_prints_no_name_from_a_header(file_model, files_to_change):
    """After a damaged header, what the reader takes for a tensor's name can be weight bytes.

    Here the header counts one tensor too many and the tensor data begin right after the table, so
    the reader takes their first bytes for one more entry: a name, then five axes.
    """
    model, paths, whole = files_to_change
    pairs, tensors, tensor_data = take_apart(whole["first"])
    for length in range(32):
        filler = ["filler", 8, gguf_string("x" * length)]
        table = put_together(pairs + [filler], tensors, b"", alignment=1)
        if len(table) % 32 == 0:
            break
    data = u64(8) + b"WEIGHTS!" + u32(5)
    faulty = table[:8] + u64(len(tensors) + 1) + table[16:] + data + tensor_data[len(data) :]
    paths["first"].write_bytes(faulty)
    try:
        with pytest.raises(WeightsRejected, match="does not have 1 to 4 axes") as failure:
            open_files(file_model, model)
    finally:
        paths["first"].write_bytes(whole["first"])
    assert "WEIGHTS!" not in str(failure.value)


def test_the_alignment_that_a_file_states_is_used(file_model, tmp_path, device):
    """The first file states alignment 16, and its header ends where alignment 32 would start the
    tensor data 16 bytes later.
    """
    model = copy_of(file_model, tmp_path)
    first = sorted(model.glob("*.gguf"))[0]
    pairs, tensors, tensor_data = take_apart(first.read_bytes())
    next(pair for pair in pairs if pair[0] == "general.alignment")[2] = struct.pack("<I", 16)
    for length in range(32):
        data = put_together(
            pairs + [["filler", 8, gguf_string("x" * length)]], tensors, tensor_data, alignment=16
        )
        if (len(data) - len(tensor_data)) % 32 == 16:
            break
    first.write_bytes(data)
    load_files(file_model, device, model, verify_sums=True)


def test_a_model_whose_shape_differs_from_the_parameter_file_is_rejected(file_model):
    fitting = Parameters.read(file_model / "policy.txt")
    for change in (
        {"L": 6, "layers": fitting.layers[:6]},
        {"f": 1024, "fe": 512},
        {"V": 97},
        {"NE": 0, "router_clamp": False, "gate_clamp": False},
        {"NE": 4},
        {"n_h": 4},
        {"n_g": 2},
        {"d": 512},
    ):
        parameters = dataclasses.replace(fitting, **change)
        with pytest.raises(WeightsRejected, match="parameter file"):
            GgufBlocks(
                parameters, file_model / "model", sorted((file_model / "model").glob("*.gguf"))
            )


def test_rows_outside_a_tensor_are_not_read(file_model):
    source = open_files(file_model)
    assert source.blocks("Emb", None, range(95, 96)).shape == (20, BLOCK_BYTES[Q6_K])
    for rows in (range(95, 97), range(-1, 1), range(0, 4, 2)):
        with pytest.raises(IndexError, match="token_embd.weight has no rows"):
            source.blocks("Emb", None, rows)
    source.close()


class ChangedSource:
    """A block source that forwards everything to another, except the attributes a test replaces."""

    def __init__(self, source, **replaced):
        self._source = source
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._source, name)


def test_an_exponent_or_multiplier_outside_the_accepted_range_is_rejected(device):
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    for exponent in (-1, 41):
        source = ChangedSource(
            SyntheticBlocks(parameters, device), exponent=lambda name, layer: exponent
        )
        with pytest.raises(WeightsRejected, match="exponent"):
            BlockWeights(parameters, device, blocks=source)
    for magnitude in (2.0**-65, 2.0**65, float("inf"), float("nan")):
        multipliers = torch.ones(parameters.d)
        multipliers[100] = magnitude
        source = ChangedSource(
            SyntheticBlocks(parameters, device), multipliers=lambda name, layer: multipliers
        )
        with pytest.raises(WeightsRejected, match="multiplier"):
            BlockWeights(parameters, device, blocks=source)


def test_an_exponent_above_24_is_rejected_where_the_prover_keeps_blocks(device):
    """cp_prove keeps a layer's matrix of type Q4_K, Q5_K, Q8_0, F16 or BF16, other than the router,
    as its blocks, and takes at most 24 for its exponent; cp_commit refuses an exponent above 24
    too. The bound does not apply to the router, even when its type is given as Q4_K (it then fails
    later, for its bytes).
    """
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    synthetic = SyntheticBlocks(parameters, device)
    for storage_type in (Q4_K, Q5_K):
        source = ChangedSource(
            synthetic,
            exponent=lambda name, layer: 25
            if synthetic.storage_type(name, layer) == storage_type
            else 12,
        )
        with pytest.raises(WeightsRejected, match="so its exponent must be at most 24"):
            BlockWeights(parameters, device, blocks=source)
    # The synthetic model has no matrix of a wider type, so Wq's type is given as one.
    for storage_type in (Q8_0, F16, BF16):
        source = ChangedSource(
            synthetic,
            exponent=lambda name, layer: 25 if name == "Wq" else 12,
            storage_type=lambda name, layer: (
                storage_type if name == "Wq" else synthetic.storage_type(name, layer)
            ),
        )
        with pytest.raises(WeightsRejected, match="so its exponent must be at most 24"):
            BlockWeights(parameters, device, blocks=source)
    source = ChangedSource(
        synthetic,
        exponent=lambda name, layer: 25 if name == "Wr" else 12,
        storage_type=lambda name, layer: (
            Q4_K if name == "Wr" else synthetic.storage_type(name, layer)
        ),
    )
    with pytest.raises(ValueError) as failure:  # its F32 bytes are no Q4_K blocks
        BlockWeights(parameters, device, blocks=source)
    assert "at most 24" not in str(failure.value)


def test_an_out_of_range_weight_is_rejected_when_the_model_is_loaded(device, monkeypatch):
    """At exponent 12 the synthetic integers stay within a matrix's limit, 4,095:
    prover/weights_blocks/made_up_blocks.h keeps 4,096 |value| below 3,673 and the multipliers
    below 1.09375. At exponent 13 the integers double, and some exceed the limit. An embedding
    entry of -2^18 becomes -2^30 at exponent 12; the embedding's integers must stay below 2^30 in
    magnitude.
    """
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")
    synthetic = SyntheticBlocks(parameters, device)
    source = ChangedSource(synthetic, exponent=lambda name, layer: 13)
    with pytest.raises(WeightsRejected, match="blk.0.attn_q.weight"):
        BlockWeights(parameters, device, blocks=source)
    source = ChangedSource(
        synthetic, exponent=lambda name, layer: 13 if (name, layer) == ("Wdr", 7) else 12
    )
    with pytest.raises(WeightsRejected, match="blk.7.ffn_down_exps.weight"):
        BlockWeights(parameters, device, blocks=source)

    def with_one_large_value(tensor, row):
        def blocks(name, layer, rows):
            found = synthetic.blocks(name, layer, rows)
            if (name, layer) == tensor and row in rows:
                first_block = (row - rows.start) * (found.shape[0] // len(rows))
                if name == "Emb":
                    found.view(torch.float32)[first_block, 3] = -(2.0**18)
                else:
                    # The block's scale h (its first float16 number) becomes 1, and its sub-block
                    # scales and minima 63. Its values then reach about 63 times its largest code,
                    # and its integers 4,096 times that.
                    found[first_block, 0:2] = torch.tensor(
                        [0x00, 0x3C], dtype=torch.uint8, device=found.device
                    )
                    found[first_block, 4:16] = 0xFF
            return found

        return ChangedSource(synthetic, blocks=blocks)

    # Small decoding steps, so that the rows below are not in a tensor's first step.
    monkeypatch.setattr(BlockWeights, "WEIGHTS_AT_ONCE", 37 * 512)
    last_expert = (parameters.NE - 1) * parameters.d + 500
    for tensor, row, file_name in (
        (("Emb", None), 100, "token_embd.weight"),
        (("Wo", 2), 500, "blk.2.attn_output.weight"),
        (("Wdr", 7), last_expert, "blk.7.ffn_down_exps.weight"),
    ):
        with pytest.raises(WeightsRejected, match=file_name):
            BlockWeights(parameters, device, blocks=with_one_large_value(tensor, row))
