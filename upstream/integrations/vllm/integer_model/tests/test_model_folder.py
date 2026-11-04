# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests of load_from_folder on a model directory (`folder` in identifiers): integer_model.json, the
options that replace its entries, and the SHA-256 digests with which a caller pins the files.
"""

import hashlib
import itertools
import json
import os
import pathlib
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest

from block_tools import SMALL_EXAMPLE
from example_models import with_line, with_lines
from integer_model import ParameterError, load_from_folder
from integer_model.block_weights import BlockWeights, SyntheticBlocks
from integer_model.parameters import Parameters
from integer_model_vllm.example import SYNTHETIC
from integer_model_vllm.reference import digests_of_reference_copies, replace_reference_directory


def blocks_description(**settings) -> str:
    return json.dumps({"weights": "blocks", "weight_settings": settings})


MODEL_DESCRIPTIONS_TO_REJECT = {
    "not_json": "{",
    "not_text": b"\xff\xfe\x00",
    "not_an_object": '["synthetic"]',
    "no_weights": "{}",
    "unknown_weights": '{"weights": "random"}',
    "unknown_entry": '{"weights": "synthetic", "device": "cpu"}',
    "settings_not_an_object": '{"weights": "synthetic", "weight_settings": []}',
    "setting_for_synthetic_weights": (
        '{"weights": "synthetic", "weight_settings": {"blocks": "synthetic"}}'
    ),
    "unknown_setting": blocks_description(folder="/tmp"),
    "setting_of_the_wrong_type": blocks_description(verify_sums="yes"),
    "sums_to_verify_for_synthetic_blocks": blocks_description(blocks="synthetic", verify_sums=True),
    "shard_that_is_no_string": blocks_description(blocks="gguf", setup_folder="s", shards=[1]),
    "unknown_kind_of_blocks": blocks_description(blocks="real"),
    "gguf_without_a_setup_folder": blocks_description(blocks="gguf", shards=["a.gguf"]),
    "gguf_without_shards": blocks_description(blocks="gguf", setup_folder="s", shards=[]),
}


# A small model, for this package's tests only.
TINY = pathlib.Path(__file__).parent / "configurations" / "tiny_without_clamps"


def model_folder(folder: pathlib.Path, description: str | bytes) -> pathlib.Path:
    """Fills `folder` for load_from_folder: the small model's parameter file (policy.txt) and
    quantization scales (KL.bin), and the given contents of integer_model.json.
    """
    (folder / "policy.txt").write_bytes((TINY / "policy.txt").read_bytes())
    (folder / "KL.bin").write_bytes((TINY / "KL.bin").read_bytes())
    (folder / "integer_model.json").write_bytes(
        description.encode("ascii") if isinstance(description, str) else description
    )
    return folder


@pytest.mark.parametrize("name", MODEL_DESCRIPTIONS_TO_REJECT)
def test_malformed_model_descriptions_are_rejected(name, tmp_path, device):
    with pytest.raises(ParameterError, match="integer_model.json"):
        load_from_folder(model_folder(tmp_path, MODEL_DESCRIPTIONS_TO_REJECT[name]), device)


def test_an_unknown_name_is_quoted_on_one_line(tmp_path, device):
    """The message escapes an unknown name in integer_model.json, so it cannot start a new line."""
    name = 'a"b\nc'
    for description in (
        {"weights": "synthetic", name: 1},
        {"weights": "blocks", "weight_settings": {name: 1}},
        {"weights": "blocks", "weight_settings": {"blocks": name}},  # here it is a value
    ):
        with pytest.raises(ParameterError) as failure:
            load_from_folder(model_folder(tmp_path, json.dumps(description)), device)
        assert '"a\\"b\\nc"' in str(failure.value)
        assert "\n" not in str(failure.value)


def test_synthetic_blocks_load_only_where_the_settings_say_synthetic(tmp_path, device):
    """If "blocks" had a default, synthetic blocks could silently load in place of GGUF files that
    the operator named. The folder has the small example's shape, at which synthetic blocks load, so
    that only a refusal stops a load. Digests, three or five, make no difference.
    """
    folder = model_folder(tmp_path, "{}")
    for name in ("policy.txt", "KL.bin"):
        shutil.copyfile(SMALL_EXAMPLE / name, folder / name)
    parameters = Parameters.read(folder / "policy.txt")

    def refused(description: str, fault: str):  # the message must begin with the file's path
        (folder / "integer_model.json").write_text(description)
        three = sha256_of_the_files(folder)
        five = three | dict.fromkeys(("wsums.txt", "fold.txt"), "0" * 64)
        for options in (None, {"sha256": three}, {"sha256": five}):
            with pytest.raises(ParameterError, match="^[^:]*integer_model.json: " + fault):
                load_from_folder(folder, device, options)

    for _, description in SYNTHETIC.values():  # what the synthetic examples write is accepted
        (folder / "integer_model.json").write_text(json.dumps(description))
        load_from_folder(folder, device)
    named = {"setup_folder": "weights", "shards": ["weights/1.gguf"]}
    in_files = blocks_description(blocks="gguf", **named)
    unsaid = r'"weight_settings" needs an entry "blocks": "gguf" .* in policy.txt\)\.'
    from_options = "integer_model.json, with the options that replace its entries: " + unsaid
    whole_entry = r' The option "weight_settings" \(.* --model-loader-extra-config\) replaces the'
    for settings in (named, {"shards": []}, {"check_free_memory": False}, {}):
        refused(blocks_description(**settings), unsaid + "$")  # $: nothing about an option follows
        # Given as an option, the settings replace the whole entry of a file that says "gguf".
        (folder / "integer_model.json").write_text(in_files)
        with pytest.raises(ParameterError, match=from_options + whole_entry):
            load_from_folder(folder, device, {"weight_settings": settings})
        fault = "setup_folder and shards need" if "shards" in settings else "blocks must be"
        with pytest.raises(ValueError, match=fault):
            BlockWeights(parameters, device, folder=folder, **settings)
    refused('{"weights": "blocks"}', unsaid + "$")
    (folder / "integer_model.json").write_text('{"weights": "synthetic"}')
    with pytest.raises(ParameterError, match=from_options + "$"):
        load_from_folder(folder, device, {"weights": "blocks"})
    (folder / "integer_model.json").write_text(in_files)
    with pytest.raises(ParameterError, match=r'"blocks" is "real" but must be .* policy.txt\)\.$'):
        load_from_folder(folder, device, {"weight_settings": {"blocks": "real"}})
    with pytest.raises(ParameterError, match='"blocks": "gguf" needs a .*' + whole_entry):
        load_from_folder(folder, device, {"weight_settings": {"blocks": "gguf"}})
    # Of an entry that is written twice, Python's JSON reader would on its own keep the second.
    for description, name in (
        (in_files[:-1] + ', "weight_settings": {"blocks": "synthetic"}}', '"weight_settings"'),
        (in_files[:-1] + ', "weight_settings": {"check_free_memory": false}}', '"weight_settings"'),
        (blocks_description(blocks="gguf")[:-2] + ', "blocks": "synthetic"}}', '"blocks"'),
        ('{"weights": "blocks", "weights": "synthetic"}', '"weights"'),
        ('{"a\\nb": 1, "a\\nb": 1}', r'"a\\nb"'),  # a line break in the name stays escaped
    ):
        refused(description, f"the entry {name} is written more than once in the same JSON object")
    # The same name in two objects is not "twice": this file gets the message on an unknown setting.
    refused(blocks_description(weights="blocks"), '"weight_settings" has no entry "weights"')
    for settings in (
        named,
        {"setup_folder": "weights"},
        {"shards": ["weights/1.gguf"]},
        {"shards": []},  # what a script writes when its pattern matches no file
    ):
        refused(
            blocks_description(blocks="synthetic", **settings),
            '"blocks": "synthetic" does not go with "setup_folder" or "shards"',
        )
        for blocks in ("synthetic", SyntheticBlocks(parameters, device)):
            with pytest.raises(ValueError, match='setup_folder and shards need blocks = "gguf"'):
                BlockWeights(parameters, device, blocks=blocks, verify_sums=True, **settings)


ABSENT_WEIGHT_FILES = json.dumps(
    {
        "weights": "blocks",
        "weight_settings": {
            "blocks": "gguf",
            "setup_folder": "no_such_folder",
            "shards": ["no_such_file.gguf"],
        },
    }
)


def test_a_wrong_table_digest_is_rejected_before_a_weight_file_is_opened(tmp_path, device):
    folder = model_folder(tmp_path, ABSENT_WEIGHT_FILES)
    # With the right digest the load fails only at the missing weight files.
    with pytest.raises(FileNotFoundError, match="no_such_f"):
        load_from_folder(folder, device)
    text = (folder / "policy.txt").read_text()
    # The lines texp0 to texp7 state the softmax table's digest.
    stated = int(re.search(r"^texp7 (\d+)$", text, flags=re.MULTILINE)[1])
    (folder / "policy.txt").write_text(with_line(text, "texp7", f"texp7 {stated ^ 1}\n"))
    with pytest.raises(ParameterError, match=r"\(lines texp0 to texp7\)"):
        load_from_folder(folder, device)


def test_a_shape_whose_sums_could_round_is_rejected_before_a_weight_file_is_opened(
    tmp_path, device
):
    """At a hidden width of f = 32,768 the down projection's sums could reach 2^53: bound (e) of
    integer_model.model._assert_arithmetic_is_exact. fe changes too: f must be 2 fe.
    """
    folder = model_folder(tmp_path, ABSENT_WEIGHT_FILES)
    text = (folder / "policy.txt").read_text()
    (folder / "policy.txt").write_text(with_lines({"f": 32768, "fe": 16384})(text))
    with pytest.raises(
        ParameterError, match="cannot evaluate a model of this shape exactly: bound \\(e\\)"
    ):
        load_from_folder(folder, device)


def test_options_replace_entries_of_the_model_description(tmp_path, device):
    """Options are checked like the file's entries, and replace them. With the option {"weights":
    "synthetic"} in place of the file's "blocks", the model makes the same top choices on a random
    window as one whose file says "synthetic".
    """
    folder = model_folder(tmp_path, json.dumps({"weights": "blocks", "weight_settings": {}}))
    with pytest.raises(ParameterError, match='"weights" must be "synthetic" or "blocks"'):
        load_from_folder(folder, device, {"weights": "random"})
    with pytest.raises(ParameterError, match="The options .* expected a JSON object"):
        load_from_folder(folder, device, ["synthetic"])
    model = load_from_folder(folder, device, {"weights": "synthetic"})
    (tmp_path / "described").mkdir()
    described = load_from_folder(
        model_folder(tmp_path / "described", '{"weights": "synthetic"}'), device
    )
    window = random.Random(2).choices(range(model.parameters.V), k=model.parameters.T)
    choices = [m.extend(m.new_sequence(), window).tolist() for m in (model, described)]
    assert choices[0] == choices[1]


def with_another_seed(whole: bytes) -> bytes:
    """The parameter file with seed_lo one higher: another seed for the synthetic weights."""
    line = re.search(rb"\nseed_lo (\d+)\n", whole)
    assert line
    return whole.replace(line[0], b"\nseed_lo %d\n" % (int(line[1]) + 1), 1)


def with_another_first_shift(whole: bytes) -> bytes:
    """The scales with the first layer's first shift changed by one."""
    return bytes([whole[0] ^ 1]) + whole[1:]


def sha256_of_the_files(folder: pathlib.Path) -> dict[str, str]:
    """The option "sha256" for the directory's three files as they are now."""
    names = ("integer_model.json", "policy.txt", "KL.bin")
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}


def test_the_digests_can_come_from_a_file_outside_the_folder(tmp_path, device):
    """ "sha256_file" names a file that holds what "sha256" would, so that the digests stand on no
    command line and in no log.
    """
    (tmp_path / "model").mkdir()
    folder = model_folder(tmp_path / "model", '{"weights": "synthetic"}')
    given = sha256_of_the_files(folder)
    digests = tmp_path / "digests.json"
    digests.write_text(json.dumps(given))
    digests.chmod(0o600)  # others may neither read nor write it
    options = {"sha256_file": str(digests)}
    assert load_from_folder(folder, device, options).parameters == (
        load_from_folder(folder, device).parameters
    )
    assert options == {"sha256_file": str(digests)}  # the caller's dictionary stays as it was
    digests.write_text(json.dumps({**given, "KL.bin": "0" * 64}))
    with pytest.raises(ParameterError, match="KL.bin is not the file"):
        load_from_folder(folder, device, options)
    (folder / "digests.json").write_text(json.dumps(given))
    (tmp_path / "link.json").symlink_to(digests)
    (tmp_path / "not_json").write_text("{")
    (tmp_path / "null").write_text("null")  # JSON, but no object: it must not load unverified
    os.mkfifo(tmp_path / "fifo")  # without a writer: opened without O_NONBLOCK it would hang
    (tmp_path / "big.json").write_bytes(b" " * 65537)
    for name in ("not_json", "null", "big.json"):
        (tmp_path / name).chmod(0o600)
    (tmp_path / "readable.json").write_text(json.dumps(given))
    (tmp_path / "readable.json").chmod(0o644)
    (tmp_path / "into_the_folder").symlink_to(folder)  # a link outside that leads inside
    for wrong, fault in (
        ({"sha256_file": "digests.json"}, "must be the absolute path"),
        ({"sha256_file": str(folder / "digests.json")}, "lies in the model directory"),
        ({"sha256_file": str(digests), "sha256": given}, 'give "sha256" or "sha256_file"'),
        ({"sha256_file": str(tmp_path)}, "is not a regular file"),
        ({"sha256_file": str(tmp_path / "fifo")}, "is not a regular file"),
        ({"sha256_file": str(tmp_path / "link.json")}, "cannot read"),
        ({"sha256_file": str(tmp_path / "not_json")}, "does not hold one JSON object"),
        ({"sha256_file": str(tmp_path / "null")}, "does not hold one JSON object"),
        ({"sha256_file": str(tmp_path / "big.json")}, "is larger than 64 KiB"),
        ({"sha256_file": str(tmp_path / "readable.json")}, "can be read or written by other"),
        (
            {"sha256_file": str(tmp_path / "into_the_folder" / "digests.json")},
            "lies in the model directory",
        ),
    ):
        with pytest.raises(ParameterError, match=re.escape(fault)):
            load_from_folder(folder, device, wrong)
    # The model directory given through a link: the file inside it is still found inside.
    with pytest.raises(ParameterError, match="lies in the model directory"):
        options = {"sha256_file": str(folder / "digests.json")}
        load_from_folder(tmp_path / "into_the_folder", device, options)


def test_the_digests_can_be_taken_from_reference_copies_outside_the_folder(tmp_path, device):
    """The vLLM loader's option "reference_directory" names a directory with copies of the three
    files: integer_model_vllm/reference.py replaces it by their digests, as "sha256", and writes
    nothing. (With weights in GGUF files: test_examples.py, on the example small_gguf.)
    """
    (tmp_path / "model").mkdir()
    folder = model_folder(tmp_path / "model", '{"weights": "synthetic"}')
    names = ("integer_model.json", "policy.txt", "KL.bin")
    count = itertools.count()

    def copies_but(name=None, change=None):
        """The option, for a new directory with the three copies; change(path) replaces name's."""
        copies = tmp_path / f"copies_{next(count)}"
        copies.mkdir()
        for each in names:
            change(copies / each) if each == name else shutil.copy(folder / each, copies)
        return {"reference_directory": str(copies)}

    def load(options):
        """As the vLLM loader does: on a copy of the options, the entry replaced, then the load."""
        options = dict(options)
        replace_reference_directory(options, folder)
        assert "reference_directory" not in options
        return load_from_folder(folder, device, options)

    options = copies_but()
    before = sorted(tmp_path.rglob("*"))
    assert load(options).parameters == load_from_folder(folder, device).parameters
    assert digests_of_reference_copies(*options.values(), folder) == sha256_of_the_files(folder)
    assert sorted(tmp_path.rglob("*")) == before  # nothing was written
    assert load({}).parameters == load(options).parameters  # without the entry, nothing is done
    with pytest.raises(ParameterError, match='unknown entry "reference_directory"'):
        load_from_folder(folder, device, options)  # the option is the vLLM loader's alone
    for name in names:  # each of the three is compared
        changed = copies_but(name, lambda path: path.write_bytes(b"another\n"))
        with pytest.raises(ParameterError, match=f"{name} is not the file"):
            load(changed)
    (tmp_path / "into_the_folder").symlink_to(folder)  # a link outside that leads inside
    shutil.copytree(options["reference_directory"], folder / "copies")
    refused = (
        ({"reference_directory": "copies_0"}, "must be the absolute path of a directory"),
        ({"reference_directory": 7}, "must be the absolute path of a directory"),
        ({"reference_directory": "/a\0b"}, "must be the absolute path of a directory"),
        ({"reference_directory": str(folder)}, "lies in the model directory"),
        ({"reference_directory": str(folder / "copies")}, "lies in the model directory"),
        ({"reference_directory": str(tmp_path / "into_the_folder")}, "lies in the model directory"),
        ({**options, "sha256": sha256_of_the_files(folder)}, 'give only one of "sha256", "sha256_'),
        ({**options, "sha256_file": "/x"}, 'give only one of "sha256", "sha256_file"'),
        ({**options, "weights": "synthetic"}, "no other option is allowed"),
        *(
            ({**options, "reference_files": n}, "must be the number 5")
            for n in (3, 4, "5", [5], 5.0, True, None)
        ),
        ({"reference_files": 5}, '"reference_files" needs "reference_directory"'),
        # all five demanded: that the two do not lie there does not make it three
        ({**options, "reference_files": 5}, "wsums.txt': No such file"),
        ({"reference_directory": str(tmp_path / "none")}, "integer_model.json': No such file"),
        (copies_but("KL.bin", lambda path: None), "KL.bin': No such file"),
        (copies_but("KL.bin", lambda path: path.mkdir()), "KL.bin' is not a regular file"),
        # without a writer: opened without O_NONBLOCK it would hang
        (copies_but("policy.txt", os.mkfifo), "policy.txt' is not a regular file"),
        # the model's own file under a second name would be compared with itself
        (copies_but("KL.bin", lambda path: path.symlink_to(folder / "KL.bin")), "are one file"),
        (copies_but("KL.bin", lambda path: path.hardlink_to(folder / "KL.bin")), "are one file"),
    )

    def held_up(*_):
        raise AssertionError("the load was held up for a minute, as by a FIFO without a writer")

    earlier = signal.signal(signal.SIGALRM, held_up)  # so that a hang fails this test alone
    try:
        for wrong, fault in refused:
            signal.alarm(60)
            with pytest.raises(ParameterError, match=re.escape(fault)):
                load(wrong)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, earlier)
    # A link to a copy that is a file of its own is followed.
    first = pathlib.Path(options["reference_directory"], "policy.txt")
    load(copies_but("policy.txt", lambda path: path.symlink_to(first)))
    # One of the two alone makes four digests, which no load takes. A link to nowhere counts too.
    four, dangling = copies_but(), copies_but()
    pathlib.Path(*four.values(), "wsums.txt").symlink_to(first)
    with pytest.raises(ParameterError, match="must be a JSON object with exactly these entries"):
        load(four)
    pathlib.Path(*dangling.values(), "fold.txt").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ParameterError, match="fold.txt': No such file"):
        load(dangling)
    # With wsums.txt and fold.txt there too, all five count, and these weights, which are in no
    # GGUF files, are refused: so examples/serve_gguf.sh, which asks for all five, serves none such.
    for name in ("wsums.txt", "fold.txt"):
        first.with_name(name).symlink_to(first)
    with pytest.raises(ParameterError, match='"sha256" must not have "wsums.txt" and "fold.txt"'):
        load(options)
    if os.geteuid() != 0:  # root reads everything
        closed = copies_but("KL.bin", lambda path: path.touch(mode=0o000))
        with pytest.raises(ParameterError, match="cannot read .*KL.bin': Permission denied"):
            load(closed)


# A program that takes the command line of "vllm serve" and runs integer_model_vllm/loader.py
# itself: small stand-ins take the place of the three modules of vLLM that the loader imports.
LOADER_ALONE = textwrap.dedent(
    """
        import contextlib, json, sys, types

        def module(name, **entries):
            sys.modules[name] = types.ModuleType(name)
            vars(sys.modules[name]).update(entries)

        for name in ("vllm", "vllm.model_executor", "vllm.utils"):
            module(name)
        module("vllm.model_executor.model_loader", register_model_loader=lambda name: lambda c: c)
        module("vllm.model_executor.model_loader.base_loader", BaseModelLoader=object)
        module("vllm.utils.torch_utils", set_default_torch_dtype=lambda t: contextlib.nullcontext())
        from integer_model_vllm.loader import IntegerModelLoader

        loader = IntegerModelLoader()
        options = json.loads(sys.argv[sys.argv.index("--model-loader-extra-config") + 1])
        loader.load_config = types.SimpleNamespace(model_loader_extra_config=options)
        model = types.SimpleNamespace(device="cpu", attach=lambda loaded: print("attached"))
        directory = sys.argv[sys.argv.index("serve") + 1]
        loader.load_weights(model, types.SimpleNamespace(model=directory))
    """
)
PACKAGE = pathlib.Path(__file__).resolve().parents[1]


def test_the_vllm_loader_takes_the_reference_directory(tmp_path):
    """integer_model_vllm/loader.py itself, in a process of its own (LOADER_ALONE): it replaces the
    option and loads, and at a file that differs from its copy the refusal names the reference
    directory.
    """
    (tmp_path / "model").mkdir()
    folder = model_folder(tmp_path / "model", '{"weights": "synthetic"}')
    shutil.copytree(folder, tmp_path / "copies")
    options = json.dumps({"reference_directory": str(tmp_path / "copies")})
    command = [sys.executable, "-B", "-P", "-c", LOADER_ALONE, "serve", str(folder)]
    command += ["--model-loader-extra-config", options]  # -P: no import from the caller's folder

    def run():
        environment = {**os.environ, "PYTHONPATH": str(PACKAGE)}
        return subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)

    done = run()
    assert (done.returncode, done.stdout) == (0, "attached\n"), done.stderr
    (tmp_path / "copies" / "KL.bin").write_bytes(b"another\n")
    done = run()
    assert done.returncode != 0 and done.stdout == ""
    assert "KL.bin is not the file whose SHA-256 was given" in done.stderr
    assert (
        'With the option "reference_directory", the SHA-256 digests are those of the copies in '
        f"{str(tmp_path / 'copies')!r}: compare a file that is refused for its digest with its "
        "copy there. Those of wsums.txt and fold.txt are among them if the two lie there or "
        '"reference_files" is 5, and go only with weights in GGUF files.' in done.stderr
    )


def test_serve_gguf_sh_serves_no_model_whose_weights_are_in_no_gguf_files(tmp_path):
    """examples/serve_gguf.sh itself, with LOADER_ALONE in the place of vllm: all five copies lie in
    the reference directory, and its integer_model.json, like the model's, puts the weights in no
    GGUF files: the script's look passes, and the load is refused. It is refused too if wsums.txt
    and fold.txt are taken out of the directory between the look and the load: the script has
    demanded all five.
    """
    for name in ("model", "bin", "tmp", "aside"):
        (tmp_path / name).mkdir(mode=0o700)
    folder = model_folder(tmp_path / "model", '{"weights": "synthetic"}')
    shutil.copytree(folder, tmp_path / "copies")
    for name in ("wsums.txt", "fold.txt"):
        (tmp_path / "copies" / name).symlink_to("policy.txt")
    (tmp_path / "bin" / "loader_alone.py").write_text(LOADER_ALONE)
    python = shlex.quote(sys.executable)
    (tmp_path / "bin" / "vllm").write_text(
        "#!/bin/sh\n"
        '[ -z "${FROM-}" ] || mv -- "$FROM/wsums.txt" "$FROM/fold.txt" "$FROM/../aside" || exit 9\n'
        f'exec {python} -B "$(dirname -- "$0")/loader_alone.py" "$@"\n'
    )
    (tmp_path / "bin" / "vllm").chmod(0o755)
    paths = [tmp_path / "bin", pathlib.Path(sys.executable).parent, os.environ["PATH"]]

    def serve(**more):
        done = subprocess.run(
            [PACKAGE / "examples" / "serve_gguf.sh", "copies", "model", "name"],
            env={
                "PATH": os.pathsep.join(map(str, paths)),
                "PYTHONPATH": str(PACKAGE),
                "TMPDIR": str(tmp_path / "tmp"),
                "VLLM_API_KEY": "k",
                **more,
            },
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode not in (0, 9) and "attached" not in done.stdout
        return done.stderr

    refusal = serve()
    assert 'so "sha256" must not have "wsums.txt" and "fold.txt".' in refusal
    assert "and go only with weights in GGUF files." in refusal
    refusal = serve(FROM=str(tmp_path / "copies"))
    assert sorted(path.name for path in (tmp_path / "aside").iterdir()) == ["fold.txt", "wsums.txt"]
    assert f"cannot read '{tmp_path}/copies/wsums.txt': No such file or directory." in refusal


def test_the_three_files_are_compared_with_the_given_digests(tmp_path, device):
    """With the files' digests the load gives the same parameters and top choices as without them.
    A changed file or one changed digit of a digest stops it.
    """
    folder = model_folder(tmp_path, '{"weights": "synthetic"}')
    given = sha256_of_the_files(folder)
    plain, compared = (
        load_from_folder(folder, device),
        load_from_folder(folder, device, {"sha256": given}),
    )
    window = random.Random(2).choices(range(plain.parameters.V), k=plain.parameters.T)
    assert compared.parameters == plain.parameters
    options = {"sha256": given}
    load_from_folder(folder, device, options)
    # The load must not change `options`: the vLLM loader (IntegerModelLoader) reads them again.
    assert options == {"sha256": given}
    choices = [model.extend(model.new_sequence(), window).tolist() for model in (plain, compared)]
    assert choices[0] == choices[1]
    # Without digests each changed file still loads, with other parameters.
    for name, change in (("policy.txt", with_another_seed), ("KL.bin", with_another_first_shift)):
        whole = (folder / name).read_bytes()
        (folder / name).write_bytes(change(whole))
        found = sha256_of_the_files(folder)[name]
        assert load_from_folder(folder, device).parameters != plain.parameters
        with pytest.raises(ParameterError) as failure:
            load_from_folder(folder, device, {"sha256": given})
        assert f"{folder / name} is not the file" in str(failure.value)
        # The message states neither digest: logs are read more widely than these files.
        assert found not in str(failure.value)
        assert given[name] not in str(failure.value)
        # Nor does it state a part of a digest, or the beginning of the unchanged file.
        beside_the_path = str(failure.value).replace(str(folder), "")
        assert not re.search("[0-9a-f]{8}", beside_the_path)
        assert whole[:8].decode("latin-1") not in beside_the_path
        (folder / name).write_bytes(whole)
        # The first digit counts, and so does the last.
        for at in (0, 63):
            one_digit_changed = (
                given[name][:at] + ("0" if given[name][at] != "0" else "1") + given[name][at + 1 :]
            )
            with pytest.raises(ParameterError, match=f"{name} is not the file"):
                load_from_folder(folder, device, {"sha256": {**given, name: one_digit_changed}})
    # The keys "policy.txt" and "KL.bin" stand for the two files, whatever these are called.
    renamed = tmp_path / "under_other_names"
    renamed.mkdir()
    (renamed / "p.txt").write_bytes((folder / "policy.txt").read_bytes())
    (renamed / "scales.bin").write_bytes((folder / "KL.bin").read_bytes())
    two = {name: given[name] for name in ("policy.txt", "KL.bin")}
    assert Parameters.read(renamed / "p.txt", renamed / "scales.bin", two) == plain.parameters
    with pytest.raises(ParameterError, match="p.txt is not the file"):
        exchanged = {"policy.txt": two["KL.bin"], "KL.bin": two["policy.txt"]}
        Parameters.read(renamed / "p.txt", renamed / "scales.bin", exchanged)
    with pytest.raises(ParameterError, match='entries: "policy.txt", "KL.bin"\\.'):
        Parameters.read(renamed / "p.txt", renamed / "scales.bin", given)


def test_the_model_description_is_compared_before_it_is_parsed(tmp_path, device):
    """A rewritten integer_model.json stops a load with digests, whatever it states."""
    folder = model_folder(tmp_path, '{"weights": "synthetic"}')
    given = sha256_of_the_files(folder)
    # The first is a description that loads, and the second is not JSON.
    for rewritten in ('{"weights": "synthetic", "weight_settings": {}}', "{"):
        model_folder(tmp_path, rewritten)
        found = sha256_of_the_files(folder)["integer_model.json"]
        with pytest.raises(ParameterError) as failure:
            load_from_folder(folder, device, {"sha256": given})
        assert "integer_model.json is not the file" in str(failure.value)
        assert "not a JSON file" not in str(failure.value)
        assert found not in str(failure.value)
        assert given["integer_model.json"] not in str(failure.value)
    model_folder(tmp_path, '{"weights": "synthetic", "weight_settings": {}}')
    load_from_folder(folder, device)


def test_with_digests_the_options_have_no_other_entry(tmp_path, device):
    """Otherwise an option could replace an entry of the compared file, and the digests would still
    match.
    """
    folder = model_folder(tmp_path, '{"weights": "synthetic"}')
    given = sha256_of_the_files(folder)
    for others in (
        {"weights": "synthetic"},
        {"weight_settings": {}},
        {"device": "cpu"},
        {"a": 1, "b": 2},
    ):
        names = ", ".join(f'"{name}"' for name in others)
        with pytest.raises(ParameterError, match=rf"no other option is allowed \(found {names}\)"):
            load_from_folder(folder, device, {"sha256": given, **others})
        # The options are checked before any file is read.
        with pytest.raises(ParameterError, match="no other option is allowed"):
            load_from_folder(tmp_path / "no_such_folder", device, {"sha256": given, **others})
    # "sha256": None counts as no digests.
    load_from_folder(folder, device, {"sha256": None, "weights": "synthetic"})


def test_malformed_digests_are_rejected(tmp_path, device):
    """The digests must be all three, with two more only for weights in GGUF files, in the form in
    which sha256sum prints them, and from outside the model directory.
    """
    folder = model_folder(tmp_path, '{"weights": "synthetic"}')
    given = sha256_of_the_files(folder)
    for malformed in (
        {},
        {"policy.txt": given["policy.txt"]},
        {name: given[name] for name in ("policy.txt", "KL.bin")},
        {**given, "config.json": given["KL.bin"]},
        {**given, "wsums.txt": given["KL.bin"]},
        {**given, "KL.bin": None},
        {**given, "KL.bin": given["KL.bin"].upper()},
        {**given, "KL.bin": given["KL.bin"][:-1]},
        {**given, "KL.bin": given["KL.bin"] + "\n"},
        {**given, "KL.bin": int(given["KL.bin"], 16)},
        list(given.values()),
        ["policy.txt", "KL.bin"],
        [],
        given["policy.txt"],
        "",
        0,
        False,
    ):
        with pytest.raises(
            ParameterError, match='entries: "integer_model.json", "policy.txt", "KL.bin", and for'
        ):
            load_from_folder(folder, device, {"sha256": malformed})
    five = {**given, "wsums.txt": given["KL.bin"], "fold.txt": given["KL.bin"]}
    with pytest.raises(
        ParameterError, match='the weights are not in GGUF files, so "sha256" must not have'
    ):
        load_from_folder(folder, device, {"sha256": five})
    for stated in (given, None):
        model_folder(tmp_path, json.dumps({"weights": "synthetic", "sha256": stated}))
        with pytest.raises(ParameterError, match='"sha256" is not allowed here'):
            load_from_folder(folder, device)


def test_the_bytes_that_are_compared_are_the_bytes_that_are_parsed(tmp_path, device, monkeypatch):
    """Each file holds its right bytes only during the package's call of Path.read_bytes, so a load
    that parsed any other reading would fail or give another model.
    """
    folder = model_folder(tmp_path, '{"weights": "synthetic"}')
    given, plain = sha256_of_the_files(folder), load_from_folder(folder, device)
    right = {name: (folder / name).read_bytes() for name in given}
    other = {
        "integer_model.json": b"{",
        "policy.txt": with_another_seed(right["policy.txt"]),
        "KL.bin": with_another_first_shift(right["KL.bin"]),
    }
    read_bytes, readings = pathlib.Path.read_bytes, []

    def read_the_right_bytes(path):
        assert path.parent == folder
        readings.append(path.name)
        path.write_bytes(right[path.name])
        try:
            return read_bytes(path)
        finally:
            path.write_bytes(other[path.name])

    for name in other:
        (folder / name).write_bytes(other[name])
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_the_right_bytes)
    assert load_from_folder(folder, device, {"sha256": given}).parameters == plain.parameters
    assert readings == ["integer_model.json", "policy.txt", "KL.bin"]


def test_the_parameter_file_is_read_and_parsed_before_the_scales(tmp_path, device):
    """So a malformed parameter file is reported as such, even if the scales are missing."""
    (tmp_path / "integer_model.json").write_bytes(b'{"weights": "synthetic"}')
    (tmp_path / "policy.txt").write_bytes(b"magic\n")
    assert not (tmp_path / "KL.bin").exists()
    with pytest.raises(ParameterError, match="policy.txt, line 1: "):
        load_from_folder(tmp_path, device)
