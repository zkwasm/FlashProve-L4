# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests of the check of free host memory that BlockWeights makes before it loads the weights."""

import functools
import json
import math
import pathlib

import pytest
import torch

from block_tools import (
    FULL_EXAMPLE,
    SMALL_EXAMPLE,
    open_descriptors,
    open_files,
    run_main,
    served_folder,
)
from integer_model import block_weights
from integer_model.block_weights import (
    BlockWeights,
    GgufBlocks,
    NotEnoughMemory,
    SyntheticBlocks,
    free_host_memory,
    host_bytes_needed,
)
from integer_model.model import load_from_folder
from integer_model.parameters import ParameterError, Parameters

GIB = 2**30
# A machine with 100 GiB available. /proc/meminfo counts in KiB and writes "kB".
MACHINE = {
    "proc/meminfo": f"MemTotal: {200 * 2**20} kB\nMemFree: 0 kB\nMemAvailable: {100 * 2**20} kB\n"
}
# memory.limit_in_bytes of a v1 cgroup without a limit, with pages of 4 KiB: 2^63 - 4096.
V1_NO_LIMIT = 9223372036854771712


def cgroup(directory: str, limit, usage, inactive=None, active=None, high=None, v1=False) -> dict:
    """The files of one fake cgroup, as {path: contents}. `inactive` and `active` are the bytes of
    inactive_file and active_file in memory.stat; if both are None there is no memory.stat.
    """
    limit_file, usage_file = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
        if v1
        else ("memory.max", "memory.current")
    )
    files = {f"{directory}/{limit_file}": f"{limit}\n", f"{directory}/{usage_file}": f"{usage}\n"}
    if high is not None:
        files[f"{directory}/memory.high"] = f"{high}\n"
    if inactive is not None or active is not None:
        # v1 also has the lines without "total_", which leave out the child cgroups.
        # free_host_memory() must not use them.
        lines = "inactive_file 1\nactive_file 1\n" if v1 else ""
        prefix = "total_" if v1 else ""
        files[f"{directory}/memory.stat"] = (
            f"anon 5\n{lines}{prefix}inactive_file {inactive or 0}\n{prefix}active_file "
            f"{active or 0}\n"
        )
    return files


# Fake systems: (files, free, source). free_host_memory() must return `free` bytes and a text that
# ends with `source`, or None if `free` is None. "proc" stands for /proc and "cgroups" for
# /sys/fs/cgroup. Contents None make a directory, which cannot be read as a file.
SYSTEMS = {
    "no_cgroups": (MACHINE, 100 * GIB, "MemAvailable in proc/meminfo"),
    "container_with_a_limit": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", 64 * GIB, 30 * GIB, 10 * GIB, 7 * GIB),
        51 * GIB,
        "memory.max - memory.current + file cache in cgroups",
    ),
    "v2_without_a_limit": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", "max", 30 * GIB, 10 * GIB, high="max"),
        100 * GIB,
        "proc/meminfo",
    ),
    "limit_above_what_the_machine_has": (
        MACHINE | {"proc/self/cgroup": "0::/\n"} | cgroup("cgroups", 320 * GIB, 30 * GIB),
        100 * GIB,
        "proc/meminfo",
    ),
    "memory_high_below_memory_max": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", 64 * GIB, 30 * GIB, high=40 * GIB),
        10 * GIB,
        "memory.high - memory.current + file cache in cgroups",
    ),
    "memory_high_alone": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", "max", 30 * GIB, high=40 * GIB),
        10 * GIB,
        "memory.high - memory.current + file cache in cgroups",
    ),
    "memory_high_above_memory_max": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", 64 * GIB, 30 * GIB, high=80 * GIB),
        34 * GIB,
        "memory.max - memory.current + file cache in cgroups",
    ),
    "limit_on_a_parent": (
        MACHINE
        | {"proc/self/cgroup": "0::/a/b\n"}
        | cgroup("cgroups/a", 50 * GIB, 10 * GIB)
        | cgroup("cgroups/a/b", "max", GIB),
        40 * GIB,
        "in cgroups/a",
    ),
    "lower_limit_on_the_own_cgroup": (
        MACHINE
        | {"proc/self/cgroup": "0::/a/b\n"}
        | cgroup("cgroups/a", 50 * GIB, 10 * GIB)
        | cgroup("cgroups/a/b", 20 * GIB, 5 * GIB),
        15 * GIB,
        "in cgroups/a/b",
    ),
    "another_cgroup_is_not_read": (
        MACHINE
        | {"proc/self/cgroup": "0::/a\n"}
        | cgroup("cgroups/a", 50 * GIB, 10 * GIB)
        | cgroup("cgroups/c", 2 * GIB, GIB),
        40 * GIB,
        "in cgroups/a",
    ),
    "cgroup_of_another_controller_is_not_read": (
        MACHINE
        | {"proc/self/cgroup": "5:cpu:/c\n0::/a\n"}
        | cgroup("cgroups/a", 50 * GIB, 10 * GIB)
        | cgroup("cgroups/c", 2 * GIB, GIB),
        40 * GIB,
        "in cgroups/a",
    ),
    "cgroup_not_visible_under_its_name": (
        MACHINE | {"proc/self/cgroup": "0::/pods/pod1/c1\n"} | cgroup("cgroups", 64 * GIB, 4 * GIB),
        60 * GIB,
        "in cgroups",
    ),
    "cgroup_outside_the_namespace": (
        MACHINE | {"proc/self/cgroup": "0::/../other\n"} | cgroup("other", 2 * GIB, GIB),
        100 * GIB,
        "proc/meminfo",
    ),
    "colon_in_the_name": (
        MACHINE | {"proc/self/cgroup": "0::/a:b\n"} | cgroup("cgroups/a:b", 50 * GIB, 10 * GIB),
        40 * GIB,
        "in cgroups/a:b",
    ),
    "form_feed_in_the_name": (
        MACHINE
        | {"proc/self/cgroup": "0::/a\x0cb\n"}
        | cgroup("cgroups/a\x0cb", 50 * GIB, 10 * GIB),
        40 * GIB,
        "in cgroups/a\x0cb",
    ),
    "name_that_is_not_utf_8": (
        MACHINE
        | {"proc/self/cgroup": b"0::/a\xff\n"}
        | cgroup("cgroups/a\udcff", 50 * GIB, 10 * GIB),
        40 * GIB,
        "in cgroups/a\udcff",
    ),
    "zero_byte_in_the_name": (
        MACHINE | {"proc/self/cgroup": "0::/a\x00b\n"} | cgroup("cgroups", 50 * GIB, 10 * GIB),
        40 * GIB,
        "in cgroups",
    ),
    "usage_above_the_limit": (
        MACHINE | {"proc/self/cgroup": "0::/\n"} | cgroup("cgroups", 10 * GIB, 12 * GIB, GIB),
        0,
        "in cgroups",
    ),
    "cache_above_the_usage": (
        MACHINE
        | {"proc/self/cgroup": "0::/\n"}
        | cgroup("cgroups", 10 * GIB, 4 * GIB, 5 * GIB, 3 * GIB),
        10 * GIB,
        "in cgroups",
    ),
    "v1_with_a_limit": (
        MACHINE
        | {"proc/self/cgroup": "5:cpu,cpuacct:/c\n4:memory:/a\n0::/\n"}
        | cgroup("cgroups/memory/a", 32 * GIB, 8 * GIB, 2 * GIB, 3 * GIB, v1=True)
        | cgroup("cgroups/memory/c", 2 * GIB, GIB, v1=True),
        29 * GIB,
        "memory.limit_in_bytes - memory.usage_in_bytes + file cache in cgroups/memory/a",
    ),
    "v1_mounted_with_another_controller": (
        MACHINE
        | {"proc/self/cgroup": "4:cpu,memory:/a\n"}
        | cgroup("cgroups/memory/a", 32 * GIB, 8 * GIB, v1=True),
        24 * GIB,
        "in cgroups/memory/a",
    ),
    "v1_without_a_limit": (
        MACHINE
        | {"proc/self/cgroup": "4:memory:/a\n"}
        | cgroup("cgroups/memory/a", V1_NO_LIMIT, 8 * GIB, v1=True),
        100 * GIB,
        "proc/meminfo",
    ),
    "v1_in_a_container": (
        MACHINE
        | {"proc/self/cgroup": "4:memory:/docker/0123\n"}
        | cgroup("cgroups/memory", 16 * GIB, GIB, v1=True),
        15 * GIB,
        "in cgroups/memory",
    ),
    "only_a_cgroup_can_be_read": (cgroup("cgroups", 16 * GIB, GIB), 15 * GIB, "in cgroups"),
    "only_a_cgroup_without_a_limit": (
        cgroup("cgroups/memory", V1_NO_LIMIT, GIB, v1=True),
        None,
        "",
    ),
    "nothing": ({}, None, ""),
    "kernel_without_memavailable": (
        {"proc/meminfo": "MemTotal: 1024 kB\nMemFree: 512 kB\n"},
        None,
        "",
    ),
    "directories_in_place_of_files": (
        {
            "proc/meminfo": None,
            "proc/self/cgroup": None,
            "cgroups/memory.max": None,
            "cgroups/memory.current": None,
        },
        None,
        "",
    ),
    "nonsense": (
        {
            "proc/meminfo": "MemAvailable: plenty kB\n",
            "proc/self/cgroup": b"\xff\xfe\x00:::\n0:\n\n::",
        }
        | cgroup("cgroups", -5, "1e9")
        | cgroup("cgroups/memory", "9" * 5000, 0, v1=True),
        None,
        "",
    ),
    "meminfo_that_is_not_text": ({"proc/meminfo": b"\xff\nMemAvailable: 1024 kB\n"}, None, ""),
    "unreadable_usage": (
        MACHINE | {"proc/self/cgroup": "0::/\n"} | cgroup("cgroups", 10 * GIB, ""),
        100 * GIB,
        "proc/meminfo",
    ),
    "unreadable_cache": (
        {"proc/self/cgroup": "0::/\n", "cgroups/memory.stat": "inactive_file much\n"}
        | cgroup("cgroups", 10 * GIB, 4 * GIB),
        6 * GIB,
        "in cgroups",
    ),
}


def fake_system(directory: pathlib.Path, files: dict) -> tuple[pathlib.Path, pathlib.Path]:
    """Writes the files below `directory`. Returns its stand-ins for /proc and /sys/fs/cgroup."""
    for name, contents in ({"proc": None, "cgroups": None} | files).items():
        if contents is None:
            (directory / name).mkdir(parents=True, exist_ok=True)
        else:
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_bytes(
                contents if isinstance(contents, bytes) else contents.encode()
            )
    return directory / "proc", directory / "cgroups"


@pytest.mark.parametrize("name", SYSTEMS)
def test_free_host_memory_returns_the_smallest_estimate(name, tmp_path):
    files, free, source = SYSTEMS[name]
    found = free_host_memory(*fake_system(tmp_path, files))
    if free is None:
        assert found is None
    else:
        assert found[0] == free
        assert found[1].replace(f"{tmp_path}/", "").endswith(source)


def use_fake_system(monkeypatch, tmp_path, files: dict) -> None:
    directories = fake_system(tmp_path / "system", files)
    monkeypatch.setattr(
        block_weights, "free_host_memory", functools.partial(free_host_memory, *directories)
    )


@pytest.fixture
def nothing_may_be_allocated(monkeypatch):
    """Makes a load that is not rejected fail at once; at full size it could use up the machine's
    memory. torch.empty may make only empty tensors, and SyntheticBlocks and GgufBlocks return no
    blocks or multipliers. Memory taken in other ways is not caught.
    """
    empty = torch.empty

    def only_empty_tensors(*size, **settings):
        assert math.prod(size[0] if isinstance(size[0], (tuple, list)) else size) == 0, (
            "a tensor was allocated"
        )
        return empty(*size, **settings)

    def not_returned(*arguments):
        raise AssertionError("loading began")

    monkeypatch.setattr(torch, "empty", only_empty_tensors)
    for source in (SyntheticBlocks, GgufBlocks):
        monkeypatch.setattr(source, "blocks", not_returned)
        monkeypatch.setattr(source, "multipliers", not_returned)


def full_size_weights(device, pinned: bool) -> int:
    """The host memory, in bytes, of the synthetic full-size model's weights, computed here without
    host_bytes_needed().

    Routed experts: 24 layers x 128 experts x 3 tensors of 8,192 x 5,120 weights, in blocks of 256
    weights. Each expert number has 72 tensors across the 24 layers: 63 are Q4_K (144 bytes a block)
    and 9 are Q5_K (176 bytes). Pinned, the experts fill 211 slabs of 1 GiB, the pieces in which
    _HostSlabs allocates their memory. The embedding is F32. On a CPU the other matrices, as int16,
    are in host memory too.
    """
    experts = 211 * GIB if pinned else 128 * (8192 * 5120 // 256) * (63 * 144 + 9 * 176)
    attention = 48 * 2 * (5120 * 5120 + 1024 * 5120)
    dense_and_shared = 24 * 3 * 16384 * 5120 + 24 * 3 * 8192 * 5120
    others = attention + dense_and_shared + 24 * 128 * 5120 + 202048 * 5120
    return experts + 4 * 202048 * 5120 + (2 * others if torch.device(device).type == "cpu" else 0)


def test_host_bytes_needed_at_full_size():
    parameters = Parameters.read(FULL_EXAMPLE / "policy.txt")
    source = SyntheticBlocks(parameters)
    for device in ("cpu", "cuda"):
        for pinned in (True, False):
            assert host_bytes_needed(
                parameters, source, torch.device(device), pinned
            ) == full_size_weights(device, pinned)
    assert full_size_weights("cuda", True) == 230_697_467_904


@pytest.mark.parametrize("pinned", [True, False])
def test_a_full_size_model_is_rejected_before_anything_is_allocated(
    pinned, device, tmp_path, monkeypatch, nothing_may_be_allocated
):
    """The machine has 1,999 GiB free. The container may use 320 - 120 + 3 + 2 = 205 GiB."""
    files = {"proc/meminfo": f"MemAvailable: {1999 * 2**20} kB\n", "proc/self/cgroup": "0::/\n"}
    use_fake_system(
        monkeypatch, tmp_path, files | cgroup("cgroups", 320 * GIB, 120 * GIB, 3 * GIB, 2 * GIB)
    )
    with pytest.raises(NotEnoughMemory) as rejection:
        BlockWeights(
            Parameters.read(FULL_EXAMPLE / "policy.txt"),
            device,
            blocks="synthetic",
            pin_memory=pinned,
        )
    weights = full_size_weights(device, pinned)
    assert str(rejection.value) == (
        "Not enough free host memory to load the weights: need "
        f"{(weights + weights // 32) / 1e9:.1f} GB "
        f"({weights / 1e9:.1f} GB for the weights, {weights // 32 / 1e9:.1f} GB for the rest of "
        "the process), found "
        f"220.1 GB (memory.max - memory.current + file cache in {tmp_path}/system/cgroups). "
        "Nothing was allocated. Use "
        "a machine or container with more free memory; if you know that more is free than was "
        'found, skip this check: add "check_free_memory": false to the entries of '
        '"weight_settings" in integer_model.json or, with python -m integer_model.block_weights, '
        "pass --no-memory-check."
    )


def test_the_weights_and_the_allowance_must_fit(
    device, tmp_path, monkeypatch, nothing_may_be_allocated
):
    parameters = Parameters.read(FULL_EXAMPLE / "policy.txt")
    needed = full_size_weights(device, True) * 33 // 32
    use_fake_system(monkeypatch, tmp_path / "enough", cgroup("cgroups", needed, 0))
    with pytest.raises(AssertionError, match="loading began"):
        BlockWeights(parameters, device, blocks="synthetic", pin_memory=True)
    use_fake_system(monkeypatch, tmp_path / "one_byte_short", cgroup("cgroups", needed - 1, 0))
    with pytest.raises(NotEnoughMemory, match="Not enough free host memory"):
        BlockWeights(parameters, device, blocks="synthetic", pin_memory=True)


@pytest.mark.skipif(
    not pathlib.Path("/proc/self/fd").is_dir(), reason="this system has no /proc/self/fd"
)
def test_a_rejected_model_in_files_has_its_files_closed(
    file_model, device, tmp_path, monkeypatch, nothing_may_be_allocated
):
    use_fake_system(monkeypatch, tmp_path, {"proc/meminfo": "MemAvailable: 1024 kB\n"})
    shards = [str(path) for path in sorted((file_model / "model").glob("*.gguf"))]
    settings = dict(blocks="gguf", setup_folder=file_model / "model", shards=shards)
    before = open_descriptors()
    with pytest.raises(
        NotEnoughMemory, match=r"MB for the rest of the process\), found 1.0 MB \(MemAvailable in "
    ):
        BlockWeights(Parameters.read(file_model / "policy.txt"), device, **settings)
    assert open_descriptors() == before


def test_bytes_for_counts_the_slabs_that_allocate_creates(monkeypatch):
    monkeypatch.setattr(block_weights._HostSlabs, "SLAB_BYTES", 100)
    for sizes in ([], [0], [100], [50, 50, 1], [60, 50, 40, 10, 100, 0, 1], [30] * 7):
        slabs = block_weights._HostSlabs(pinned=False)
        for nbytes in sizes:
            slabs.allocate(nbytes)
        assert block_weights._HostSlabs.bytes_for(sizes) == 100 * slabs.slabs
    # allocate() would reject the third request, and bytes_for() stops there.
    assert block_weights._HostSlabs.bytes_for([60, 50, 101, 60]) == 200


@pytest.mark.parametrize("kind", ["synthetic", "files"])
def test_host_bytes_needed_is_what_a_load_holds(kind, file_model, device, monkeypatch):
    monkeypatch.setattr(block_weights._HostSlabs, "SLAB_BYTES", 2_000_000)
    parameters = Parameters.read(
        (SMALL_EXAMPLE if kind == "synthetic" else file_model) / "policy.txt"
    )
    source = SyntheticBlocks(parameters, device) if kind == "synthetic" else open_files(file_model)
    weights = BlockWeights(parameters, device, blocks=source, check_free_memory=False)
    written = sum(blocks.numel() for blocks in weights._on_host.values())
    others = weights._embedding.numel()
    if torch.device(device).type == "cpu":
        others += sum(2 * matrix.numel() for matrix in weights._on_device.values())
    assert 0 < written < weights.host_bytes
    assert (
        host_bytes_needed(parameters, source, torch.device(device), True)
        == weights.host_bytes + others
    )
    assert host_bytes_needed(parameters, source, torch.device(device), False) == written + others
    if kind == "files":
        source.close()


def test_the_check_changes_nothing_in_a_model_that_fits_and_can_be_skipped(
    file_model, device, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(block_weights._HostSlabs, "SLAB_BYTES", 2_000_000)
    parameters = Parameters.read(SMALL_EXAMPLE / "policy.txt")

    def digests(**settings):
        return BlockWeights(
            parameters, device, blocks="synthetic", hash_integers=True, **settings
        ).digests

    unchecked = digests(check_free_memory=False)
    use_fake_system(monkeypatch, tmp_path / "enough", MACHINE)
    assert digests() == unchecked
    use_fake_system(monkeypatch, tmp_path / "nothing_to_read", {})
    assert digests() == unchecked
    too_little = {"proc/meminfo": "MemAvailable: 0 kB\n"}
    use_fake_system(monkeypatch, tmp_path / "too_little", too_little)
    with pytest.raises(NotEnoughMemory, match="Not enough free host memory"):
        digests()
    assert digests(check_free_memory=False) == unchecked

    # The setting in integer_model.json, then the command line, on a system with too little free
    # memory.
    use_fake_system(monkeypatch, tmp_path / "too_little_for_the_loads", too_little)
    directory, _ = served_folder(file_model, tmp_path)
    description = json.loads((directory / "integer_model.json").read_text())

    def load_with(weights="blocks", **changed):
        settings = (description["weight_settings"] if weights == "blocks" else {}) | changed
        (directory / "integer_model.json").write_text(
            json.dumps({"weights": weights, "weight_settings": settings})
        )
        return load_from_folder(directory, device)

    with pytest.raises(NotEnoughMemory, match="Not enough free host memory"):
        load_with()
    with pytest.raises(NotEnoughMemory, match="Not enough free host memory"):
        load_with(check_free_memory=True)
    assert load_with(check_free_memory=False).weights.host_bytes == 12 * 2_000_000
    with pytest.raises(ParameterError, match='the setting "check_free_memory" must be a boolean'):
        load_with(check_free_memory="no")
    with pytest.raises(
        ParameterError, match='"weight_settings" has no entry "check_free_memory" for these weights'
    ):
        load_with(weights="synthetic", check_free_memory=False)

    status, out, err = run_main(monkeypatch, capsys, SMALL_EXAMPLE / "policy.txt", device)
    assert (status, out) == (2, "")
    assert err.startswith("error: NotEnoughMemory: Not enough free host memory")
    status, out, _ = run_main(
        monkeypatch, capsys, SMALL_EXAMPLE / "policy.txt", device, options=["--no-memory-check"]
    )
    assert status == 0
    assert out.startswith("74 tensors.")
