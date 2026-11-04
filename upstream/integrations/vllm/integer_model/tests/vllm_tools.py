# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the tests with vLLM."""

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile

import stand_in_integer_model as stand_in

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGES = os.path.dirname(HERE)

# These must be set before vllm is imported. "cpu", set only without a GPU, selects vLLM's CPU
# worker, which cpu_only.py lets start from a wheel built for GPUs.
ENVIRONMENT = {
    "VLLM_TARGET_DEVICE": "cpu",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

HAS_GPU = (
    bool(shutil.which("nvidia-smi"))
    and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
)


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


# The stand-in model (stand_in_integer_model.py) never chooses the last two ids, 49 and 48.
VOCABULARY_SIZE, WINDOW_LENGTH, END_TOKEN, OUT_OF_RANGE_TOKEN, FAULTY_TOKEN = 50, 32, 7, 49, 48
STAND_IN = {
    "stand_in": True,
    "window_length": WINDOW_LENGTH,
    "out_of_range_token": OUT_OF_RANGE_TOKEN,
    "faulty_token": FAULTY_TOKEN,
}


def expected(prompt, longest=WINDOW_LENGTH, window=WINDOW_LENGTH):
    """The reply that vLLM must serve: the stand-in's, up to the end token or a length limit."""
    return stand_in.continuation(
        prompt, VOCABULARY_SIZE, [END_TOKEN], min(longest, window - len(prompt))
    )


# Prompts of several lengths. The replies to the first four end with the end token, and those to
# the last two reach the window's end.
PROMPTS = [
    [t] for t in range(VOCABULARY_SIZE) if expected([t])[-1] == END_TOKEN and len(expected([t])) > 2
][:2] + [
    [1, 2, 3],
    [0],
    list(range(10, 30)),
    [5] * (WINDOW_LENGTH - 1),
]


def set_up_environment(*folders):
    """Sets the variables of ENVIRONMENT that are unset (VLLM_TARGET_DEVICE only without a GPU), and
    puts the directories `folders` first on the import path of this process and its children.

    The children matter because vLLM inspects a model class in a new process and runs its engine in
    another.
    """
    for name, value in ENVIRONMENT.items():
        if name != "VLLM_TARGET_DEVICE" or not HAS_GPU:
            os.environ.setdefault(name, value)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        list(folders) + [p for p in [os.environ.get("PYTHONPATH")] if p]
    )
    sys.path[:0] = folders


def model_folder(
    folder, vocabulary_size=VOCABULARY_SIZE, window_length=WINDOW_LENGTH, end_token=END_TOKEN
):
    """Creates a new directory with a synthetic config.json and tokenizer, and returns it."""
    if importlib.util.find_spec("integer_model_vllm") is None:
        sys.path.append(PACKAGES)  # neither installed nor on the path: take the checkout's
    example = importlib.import_module("integer_model_vllm.example")
    os.makedirs(folder)
    example.write_vllm_files(folder, vocabulary_size, window_length, end_token)
    return folder


def install(work):
    """Builds the package's wheel, unpacks it into <work>/installed/site, and returns that path.

    With that directory on PYTHONPATH, each of vLLM's processes finds the plug-in through its entry
    point, as after an installation. The tests' own plug-ins are modules of this directory (HERE),
    so the caller puts HERE on PYTHONPATH too. Fails if <work>/installed exists.
    """
    work = os.path.abspath(os.path.join(work, "installed"))
    os.makedirs(work)
    source, site = os.path.join(work, "source"), os.path.join(work, "site")
    os.makedirs(source)
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copy(os.path.join(PACKAGES, name), source)
    for directory in ("integer_model", "integer_model_vllm", "licenses"):
        shutil.copytree(os.path.join(PACKAGES, directory), os.path.join(source, directory))
    build = (
        "import sys; from setuptools import build_meta; print(build_meta.build_wheel(sys.argv[1]))"
    )
    wheel = subprocess.run(
        [sys.executable, "-c", build, work], cwd=source, capture_output=True, text=True, check=True
    )
    with zipfile.ZipFile(os.path.join(work, wheel.stdout.split()[-1])) as f:
        f.extractall(site)
    # Entry points for two modules of tests/. stand_in_integer_model.register() makes
    # integer_model.load_from_folder() return the stand-in if the options have "stand_in";
    # cpu_only.register() gets an entry point only without a GPU.
    plug_ins = {"stand_in": "stand_in_integer_model"}
    if not HAS_GPU:
        plug_ins["cpu_only"] = "cpu_only"
    for name, module in plug_ins.items():
        os.makedirs(os.path.join(site, f"{name}-0.dist-info"))
        for file, text in (
            ("METADATA", f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: 0\n"),
            ("entry_points.txt", f"[vllm.general_plugins]\n{name} = {module}:register\n"),
        ):
            with open(os.path.join(site, f"{name}-0.dist-info", file), "w") as f:
                f.write(text)
    return site


def shut_down(llm):
    """Shuts down a vllm.LLM engine, so that the same process can start another one.

    The two calls are internals of vLLM 0.29.0, and vLLM's tests shut an engine down with them
    (Apache License 2.0).
    """
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    llm.llm_engine.engine_core.shutdown()
    cleanup_dist_env_and_memory()


def engine_arguments(folder, **changes):
    """vLLM's engine arguments: the stand-in in place of the model, and the settings that
    integrations/vllm/integer_model/OPERATING.md, "Settings", lists as required.
    """
    return {
        "model": folder,
        "load_format": "integer_model",
        "model_loader_extra_config": dict(STAND_IN),
        "max_num_seqs": 1,
        "max_model_len": WINDOW_LENGTH,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        **changes,
    }
