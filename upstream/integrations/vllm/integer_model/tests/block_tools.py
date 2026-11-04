# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Paths and helpers shared by the tests, chiefly those of integer_model.block_weights."""

import hashlib
import json
import pathlib
import shutil
import sys

import torch

from integer_model import block_weights
from integer_model.block_weights import (
    BlockWeights,
    GgufBlocks,
    all_tensors,
    tensor_name,
    tensor_shape,
)
from integer_model.parameters import Parameters

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
TREE = PACKAGE.parents[2]
# Two of the repository's example configurations: 8 layers of width 512 with windows of 64 tokens,
# and the shape of Llama 4 Maverick.
SMALL_EXAMPLE = TREE / "prover" / "examples" / "target_small_f2"
FULL_EXAMPLE = TREE / "prover" / "examples" / "target_full_f2"

# The shape of the small model in GGUF files (fixture file_model in conftest.py). The conversion
# rule from quantized blocks to integers is defined only for width 5,120 and head dimension 128.
# The model has 8 layers because it borrows SMALL_EXAMPLE's quantization scales (KL.bin).
FILE_MODEL = {"L": 8, "d": 5120, "H": 2, "G": 1, "dh": 128, "f": 512, "fe": 256, "NE": 2, "V": 96}
# Four dense layers of 7 tensors, four expert layers of 11, and the embedding and head.
FILE_MODEL_TENSORS = 4 * 7 + 4 * 11 + 2


def ending(returncode: int) -> str:
    """How a program ended, from subprocess's returncode."""
    if returncode < 0:
        return f"was killed by signal {-returncode}"
    return f"ended with exit code {returncode}"


def copy_of(file_model, tmp_path) -> pathlib.Path:
    """A copy of file_model's subdirectory `model`, for a test to change."""
    shutil.copytree(file_model / "model", tmp_path / "model")
    return tmp_path / "model"


def open_files(file_model, model_folder=None) -> GgufBlocks:
    model_folder = model_folder or file_model / "model"
    return GgufBlocks(
        Parameters.read(file_model / "policy.txt"),
        model_folder,
        sorted(model_folder.glob("*.gguf")),
    )


def load_files(file_model, device, model_folder=None, **settings) -> BlockWeights:
    """Loads the small model (model_folder, if given, replaces `model`) and closes the files."""
    source = open_files(file_model, model_folder)
    try:
        return BlockWeights(
            Parameters.read(file_model / "policy.txt"), device, blocks=source, **settings
        )
    finally:
        source.close()


def sums_of(file_model) -> dict[str, str]:
    lines = [line.split() for line in (file_model / "model" / "wsums.txt").read_text().splitlines()]
    return {line[0]: line[4] for line in lines if line[0] != "END"}


def sha256_of(integers: torch.Tensor, dtype: torch.dtype) -> str:
    return hashlib.sha256(integers.to(dtype).cpu().numpy()).hexdigest()


def returned_digests(weights: BlockWeights, parameters: Parameters, device) -> dict[str, str]:
    """For each committed tensor, under its name as in sums_of(): the SHA-256 of what matrix() or
    embedding_rows() returns, as int16 (int32 for the embedding), a routed tensor's experts one
    after another.
    """
    digests = {}
    for name, layer in all_tensors(parameters):
        experts, outputs, inputs = tensor_shape(parameters, name, layer)
        if name == "Emb":
            returned = weights.embedding_rows(torch.arange(parameters.V, device=device))
            assert returned.dtype == torch.int64
        elif experts == 1:
            returned = weights.matrix(name, layer)
        else:
            returned = torch.cat([weights.matrix(name, layer, expert) for expert in range(experts)])
        assert returned.shape == (experts * outputs, inputs)
        assert returned.device.type == torch.device(device).type
        digests[tensor_name(parameters, name, layer)] = sha256_of(
            returned, torch.int32 if name == "Emb" else torch.int16
        )
    return digests


def served_folder(file_model, tmp_path, **settings) -> tuple[pathlib.Path, dict[str, str]]:
    """A copy of the small model as a directory for load_from_folder(), and the option "sha256"
    for its five small files.

    settings are further entries of "weight_settings".
    """
    folder = tmp_path / "served"
    shutil.copytree(file_model, folder)
    shards = sorted(str(path.relative_to(folder)) for path in folder.glob("model/*.gguf"))
    settings = {"blocks": "gguf", "setup_folder": "model", "shards": shards, **settings}
    (folder / "integer_model.json").write_text(
        json.dumps({"weights": "blocks", "weight_settings": settings})
    )
    files = ("integer_model.json", "policy.txt", "KL.bin", "model/wsums.txt", "model/fold.txt")
    return folder, {
        file.split("/")[-1]: hashlib.sha256((folder / file).read_bytes()).hexdigest()
        for file in files
    }


def open_descriptors() -> int:
    return len(list(pathlib.Path("/proc/self/fd").iterdir()))


def run_main(
    monkeypatch, capsys, parameter_file, device, fold=None, shards=(), options=()
) -> tuple[int, str, str]:
    """Calls block_weights.main() with a command line; returns its exit code, stdout and stderr."""
    arguments = ["block_weights", str(parameter_file), "--device", device, *options]
    arguments += ["--fold", str(fold)] if fold else []
    for shard in shards:
        arguments += ["--shard", str(shard)]
    monkeypatch.setattr(sys, "argv", arguments)
    try:
        status = block_weights.main()
    except SystemExit as stop:  # argparse ends the program in this way
        status = stop.code
    printed = capsys.readouterr()
    return status, printed.out, printed.err
