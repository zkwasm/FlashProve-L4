# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests, in this order, of integer_model_vllm/example.py with its command vllm_files and of
examples/serve.sh, serve_gguf.sh, request.sh and score_reply.py; last, of how paths are printed.

They need neither vLLM nor a GPU: vllm and curl are fake programs. Paths in comments start at
integrations/vllm/integer_model.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import pytest

import integer_model
from integer_model.block_weights import BlockWeights
from integer_model_vllm import example
from integer_model_vllm.reference import replace_reference_directory

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
VARIABLES = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "PYTHONSAFEPATH": "1",
}


def run(*command, **options):
    return subprocess.run(
        [str(word) for word in command], capture_output=True, text=True, **options
    )


def fake_program(path, script):
    """Writes a shell script that stands in for a program. Skips the test if the script cannot
    run, because a PATH search could then find the real program.
    """
    path.write_text(script)
    path.chmod(0o755)
    try:
        run(path)
    except PermissionError:
        pytest.skip(
            f"{path.parent} does not let programs run (mounted noexec?). First make a folder on a "
            "file system that does: dir=$(mktemp -d -p DIR), where DIR is a folder of yours there. "
            "If that succeeds, run pytest again "
            'with your other arguments and --basetemp="${dir:?}/t" (pytest empties the folder that '
            "--basetemp names)."
        )


def make_example(*arguments):
    """Runs example.py. For small_gguf with --directory D, TMPDIR is a new private folder beside D,
    so that the folder for make_model lies below the test's own folder, not in /tmp."""
    environment = dict(os.environ)
    if arguments[0] == "small_gguf" and "--directory" in arguments:
        beside = pathlib.Path(arguments[arguments.index("--directory") + 1]).parent / "tmp.make"
        beside.mkdir(mode=0o700, exist_ok=True)
        environment["TMPDIR"] = str(beside)
    return run(
        sys.executable, "-m", "integer_model_vllm.example", *arguments, cwd=PACKAGE, env=environment
    )


@pytest.mark.parametrize(
    "name, configuration, window_length",
    [("small_synthetic", "target_small_f2", 64), ("full_size", "target_full_f2", 1024)],
)
def test_an_example_with_synthetic_weights(name, configuration, window_length, tmp_path):
    done = make_example(name, "--directory", tmp_path / "new")
    assert (done.returncode, done.stdout) == (0, f"{tmp_path / 'new'}\n")
    model = tmp_path / "new" / "model"
    assert [path.name for path in (tmp_path / "new").iterdir()] == ["model"]
    assert sorted(path.name for path in model.iterdir()) == [
        "KL.bin",
        "config.json",
        "integer_model.json",
        "policy.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for file in ("policy.txt", "KL.bin"):
        assert (model / file).read_bytes() == (
            REPOSITORY / "prover" / "examples" / configuration / file
        ).read_bytes()
    assert (
        json.loads((model / "config.json").read_text())["max_position_embeddings"] == window_length
    )
    assert f"vllm serve {model} --host 127.0.0.1 --served-model-name {name} " in done.stderr


@pytest.mark.parametrize("end_token", [49, 7, 0, [7, 9]])
def test_the_tokenizer_has_one_base_word_fewer_than_the_vocabulary(end_token, tmp_path):
    """vLLM accepts prompt ids up to max(largest id, number of base words): both are 49, not 50."""
    example.write_vllm_files(tmp_path, 50, 32, end_token)
    first_end = end_token if isinstance(end_token, int) else end_token[0]
    words = ["<end>" if i == first_end else f"w{i}" for i in range(50)]
    tokenizer = json.loads((tmp_path / "tokenizer.json").read_text())
    assert tokenizer["model"]["vocab"] == {word: i for i, word in enumerate(words[:49])}
    (added,) = tokenizer["added_tokens"]
    assert (added["id"], added["content"], added["special"]) == (49, words[49], first_end == 49)
    assert tokenizer["model"]["unk_token"] == words[0]
    assert json.loads((tmp_path / "config.json").read_text())["eos_token_id"] == end_token
    transformers = pytest.importorskip("transformers")
    loaded = transformers.AutoTokenizer.from_pretrained(tmp_path)
    assert (loaded.vocab_size, max(loaded.get_vocab().values())) == (49, 49)
    assert loaded.convert_ids_to_tokens(list(range(50))) == words
    assert loaded(" ".join(words))["input_ids"] == list(range(50))
    assert loaded.eos_token_id == first_end


def test_vllm_files_for_a_model_directory_of_ones_own(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    missing = make_example("vllm_files", model, "7")
    assert missing.returncode != 0
    assert "Cannot read the parameter file" in missing.stderr
    shutil.copy(REPOSITORY / "prover" / "examples" / "target_small_f2" / "policy.txt", model)
    for words, message in (
        ((model,), "usage: "),
        ((model, "7,"), "usage: "),
        ((model, "-1"), "usage: "),
        ((model, "7", "9"), "usage: "),
        ((model, "7,512"), "must be below the vocabulary size, 512"),
    ):
        done = make_example("vllm_files", *words)
        assert done.returncode != 0, words
        assert message in done.stderr, words
        assert os.listdir(model) == ["policy.txt"]
    done = make_example("vllm_files", model, "7,9")
    assert (done.returncode, done.stderr) == (0, "")
    assert "vocab_size 512, max_position_embeddings 64, eos_token_id 7,9." in done.stdout
    assert sorted(os.listdir(model)) == [
        "config.json",
        "policy.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    config = json.loads((model / "config.json").read_text())
    assert (config["vocab_size"], config["max_position_embeddings"]) == (512, 64)
    assert config["eos_token_id"] == [7, 9]
    before = {name: (model / name).read_bytes() for name in os.listdir(model)}
    again = make_example("vllm_files", model, "7,9")
    assert again.returncode != 0
    assert (
        "has config.json, tokenizer.json, tokenizer_config.json already. Nothing was written"
        in again.stderr
    )
    assert {name: (model / name).read_bytes() for name in os.listdir(model)} == before
    other = tmp_path / "with_a_tokenizer"
    other.mkdir()
    for name in ("policy.txt", "tokenizer.json"):
        shutil.copy(model / name, other)
    third = make_example("vllm_files", other, "7,9")
    assert third.returncode != 0
    assert "has tokenizer.json already. Nothing was written" in third.stderr
    assert sorted(os.listdir(other)) == ["policy.txt", "tokenizer.json"]


def greedy_reply(model):
    """The model's first four tokens after examples/request.sh's default prompt."""
    sequence = model.new_sequence()
    reply = [int(model.extend(sequence, [17, 400, 3])[-1])]
    for _ in range(3):
        reply.append(int(model.extend(sequence, reply[-1:])[-1]))
    return reply


def test_the_small_example_gives_the_reply_that_request_sh_names(tmp_path):
    example.main(["small_synthetic", "--directory", str(tmp_path / "new"), "--quiet"])
    assert greedy_reply(integer_model.load_from_folder(tmp_path / "new" / "model", "cpu")) == [
        365,
        287,
        374,
        290,
    ]


def test_an_existing_directory_is_rejected_and_unchanged(tmp_path):
    (tmp_path / "theirs").mkdir()
    (tmp_path / "theirs" / "config.json").write_text("theirs")
    for directory in (tmp_path / "theirs", tmp_path / "theirs" / "config.json"):
        done = make_example("small_synthetic", "--directory", directory)
        assert done.returncode != 0
        assert done.stdout == ""
        assert done.stderr.startswith(f"Cannot create {directory}: ")
        assert "Traceback" not in done.stderr
    # A missing parent is refused by the look itself, which is always made.
    done = make_example("small_synthetic", "--directory", tmp_path / "no" / "parent")
    assert (done.returncode, done.stdout) == (2, "")
    assert "does not exist" in done.stderr and "Traceback" not in done.stderr
    assert [path.name for path in tmp_path.iterdir()] == ["theirs"]
    assert [path.name for path in (tmp_path / "theirs").iterdir()] == ["config.json"]
    assert (tmp_path / "theirs" / "config.json").read_text() == "theirs"


def test_a_directory_below_one_that_others_can_change_is_refused(tmp_path):
    """The new directory's parent, and every folder above it, must be out of reach of other
    accounts: a folder that its group or others can write in, without the sticky bit, is refused
    before anything is created. So is such a temporary directory, named by TMPDIR.
    """
    open_folder = tmp_path / "open"
    open_folder.mkdir()
    open_folder.chmod(0o777)
    for done in (
        make_example("small_synthetic", "--directory", open_folder / "new"),
        run(
            *(sys.executable, "-m", "integer_model_vllm.example", "small_synthetic"),
            cwd=PACKAGE,
            env={**os.environ, "TMPDIR": str(open_folder)},
        ),
    ):
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr.startswith("integer_model_vllm.example: the ")
        assert (
            f"lies below {open_folder}, which another account may be able to change (its group or "
            "others can write in it, and it has no sticky bit)"
        ) in done.stderr
    # The last call, on TMPDIR, also says what to do.
    assert done.stderr.endswith(" Set TMPDIR to such a folder, or give --directory.\n")
    assert list(open_folder.iterdir()) == []
    open_folder.chmod(0o1777)  # with the sticky bit, as /tmp has it
    assert make_example("small_synthetic", "--directory", open_folder / "new").returncode == 0


def test_try_limits_refuses_a_directory_that_others_can_change(tmp_path):
    """examples/try_limits.py writes its replies into DIRECTORY. A DIRECTORY that its group or
    others can write in, without the sticky bit, is refused before anything is written.
    """
    pytest.importorskip("transformers")  # the script imports it before it looks
    open_folder = tmp_path / "open"
    open_folder.mkdir()
    open_folder.chmod(0o777)
    done = run(
        sys.executable,
        PACKAGE / "examples" / "try_limits.py",
        open_folder,
        env={**os.environ, "PYTHONPATH": str(PACKAGE)},
    )
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr.startswith(
        f"try_limits.py: the directory {open_folder} lies below {open_folder}, which another "
        "account may be able to change (its group or others can write in it, and it has no "
        "sticky bit)"
    )
    assert "Here the directory must hold the example: make the example again" in done.stderr
    assert list(open_folder.iterdir()) == []


def test_try_limits_refuses_a_directory_that_does_not_stand(tmp_path):
    """try_limits.py makes its look always: a DIRECTORY that does not exist is refused by it, status
    2, before anything is loaded or made."""
    pytest.importorskip("transformers")  # try_limits.py imports it first
    done = run(
        sys.executable,
        PACKAGE / "examples" / "try_limits.py",
        tmp_path / "gone",
        env={**os.environ, "PYTHONPATH": str(PACKAGE)},
    )
    assert done.returncode == 2 and "does not exist" in done.stderr, done.stderr
    assert list(tmp_path.iterdir()) == []


def test_the_new_directory_is_not_writable_by_others_under_any_umask(tmp_path):
    """Under umask 002, as for many users with a private group, example.py still makes --directory
    with no write for group and others, so that examples/try_limits.py, the next command in
    integrations/vllm/integer_model/OPERATING.md, accepts it.
    """
    new = tmp_path / "new"
    done = run(
        *("sh", "-c", 'umask 002 && exec "$@"', "sh", sys.executable, "-m"),
        *("integer_model_vllm.example", "small_synthetic", "--directory", new, "--quiet"),
        cwd=PACKAGE,
    )
    assert done.returncode == 0, done.stderr
    assert new.stat().st_mode & 0o022 == 0
    assert example.others_can_change(str(new)) is None


def test_a_test_session_below_an_open_place_stands_aside(tmp_path):
    """tests/conftest.py looks at pytest's base temporary directory with example.py's routine. Below
    a folder that its group or others can write in, without the sticky bit, no test runs, the
    exit status is 77, and the last line says why, as the tree's test runner expects.
    """
    open_folder = tmp_path / "open"
    open_folder.mkdir()
    open_folder.chmod(0o777)
    done = run(
        *(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
        *(f"--basetemp={open_folder / 'base'}", "tests/test_model_folder.py"),
        cwd=PACKAGE,
    )
    assert done.returncode == 77, done.stdout[-2000:]
    assert done.stdout.splitlines()[-1] == (
        f"stood aside: below an open place: {open_folder} (its group or others can write in it, "
        "and it has no sticky bit)"
    )
    assert " passed" not in done.stdout


@pytest.mark.parametrize("planted", ["policy.txt", "KL.bin", "integer_model.json"])
@pytest.mark.parametrize("kind", ["link", "file"])
def test_the_example_overwrites_nothing_put_after_its_mkdir(tmp_path, monkeypatch, planted, kind):
    """After the example's own mkdir of model/, one name that it writes is taken by a link to a file
    elsewhere or by a plain file: it refuses to write there, and nothing of theirs changes.
    """
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("not the example's")
    mkdir = pathlib.Path.mkdir

    def mkdir_and_plant(self, *args, **kwargs):
        mkdir(self, *args, **kwargs)
        if self.name == "model":
            if kind == "link":
                (self / planted).symlink_to(elsewhere)
            else:
                (self / planted).write_text("planted")

    monkeypatch.setattr(pathlib.Path, "mkdir", mkdir_and_plant)
    new = tmp_path / "new"
    with pytest.raises(SystemExit, match=f"{planted} exists already; nothing is overwritten"):
        example.main(["small_synthetic", "--directory", str(new), "--quiet"])
    assert elsewhere.read_text() == "not the example's"
    if kind == "link":
        assert (new / "model" / planted).is_symlink()
    else:
        assert (new / "model" / planted).read_text() == "planted"


def test_a_refusal_names_the_file_also_where_the_error_carries_no_name(tmp_path, monkeypatch):
    """float_export.py raises FileExistsError with a message alone; the sentence still names the
    file.
    """

    def refused(*arguments):
        raise FileExistsError("/some/where/config.json exists already.")

    monkeypatch.setattr(example, "make_synthetic", refused)
    with pytest.raises(SystemExit) as stop:
        example.main(["small_synthetic", "--directory", str(tmp_path / "new"), "--quiet"])
    assert str(stop.value.code).startswith(
        "/some/where/config.json exists already; nothing is overwritten."
    )


def test_make_model_is_built_and_run_in_a_folder_of_its_own(tmp_path, monkeypatch):
    """small_gguf builds make_model in a new folder below TMPDIR, runs it there by that path, and
    removes both afterwards: a make_model put into the example's own directory is never run.
    """
    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")  # no compiler runs here
    (tmp_path / "tmp").mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("DEPENDENCIES_OUTPUT", str(tmp_path / "deps"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    calls = []

    def run_nothing(arguments, **options):
        calls.append((arguments, options))
        if len(calls) == 2:  # make_model itself: stop here, as if it had failed
            raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(example.subprocess, "run", run_nothing)
    with pytest.raises(SystemExit, match="make_model failed"):
        example.main(["small_gguf", "--directory", str(tmp_path / "new"), "--quiet"])
    (compiler, compiling), (program, running) = calls
    built = pathlib.Path(compiler[compiler.index("-o") + 1])
    assert built.name == "make_model" and built.parent.name.startswith("make_model.")
    assert built.parent.parent == tmp_path / "tmp" and compiling["cwd"] == built.parent
    assert pathlib.Path(program[0]) == built
    # g++'s own temporary files, and make_model's, lie in that closed folder too, and neither gets
    # more of the caller's environment than PATH (DEPENDENCIES_OUTPUT, set here, would lead g++).
    assert compiling["env"]["TMPDIR"] == running["env"]["TMPDIR"] == str(built.parent)
    assert set(compiling["env"]) == set(running["env"]) == {"PATH", "TMPDIR", "LC_ALL"}
    assert not built.parent.exists()  # removed again
    assert list((tmp_path / "tmp").iterdir()) == []


def test_the_build_folder_is_refused_below_a_place_that_others_can_change(tmp_path, monkeypatch):
    """Below a folder that others can write in, or under sudo below the sudoing account's own
    folder, make_model is not built: the run stops with status 2 and leaves nothing there."""
    monkeypatch.delenv("SUDO_UID", raising=False)  # sudo sets it; the first pass needs it unset
    (tmp_path / "open" / "t").mkdir(parents=True)
    (tmp_path / "open").chmod(0o777)
    places = [(tmp_path / "open" / "t", {})]
    if os.geteuid() == 0:  # the account that ran sudo is not trusted here
        (tmp_path / "theirs" / "t").mkdir(parents=True)
        os.chown(tmp_path / "theirs", 54321, 54321)
        places.append((tmp_path / "theirs" / "t", {"SUDO_UID": "54321"}))
    for place, variables in places:
        monkeypatch.setenv("TMPDIR", str(place))
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(tempfile, "tempdir", None)
        with pytest.raises(SystemExit) as stop:
            example.build_folder()
        assert stop.value.code == 2
        assert list(place.iterdir()) == []
        assert os.environ.get("SUDO_UID") == variables.get("SUDO_UID")  # put back as it was


def test_small_gguf_refused_for_its_build_folder_leaves_nothing(tmp_path, monkeypatch):
    """build_folder() is made and looked at before the example's directory: its refusal leaves
    neither behind."""
    (tmp_path / "open" / "t").mkdir(parents=True)
    (tmp_path / "open").chmod(0o777)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "open" / "t"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")  # no compiler runs here
    with pytest.raises(SystemExit) as stop:
        example.main(["small_gguf", "--directory", str(tmp_path / "new"), "--quiet"])
    assert stop.value.code == 2
    assert not (tmp_path / "new").exists() and list((tmp_path / "open" / "t").iterdir()) == []


def test_make_model_that_cannot_run_there_is_named(tmp_path, monkeypatch):
    """A folder for temporary files on a file system that lets no program run (noexec): a sentence
    that says what to do, and make_model's folder removed again."""
    (tmp_path / "tmp").mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")
    calls = []

    def run_nothing(arguments, **options):
        calls.append(arguments)
        if len(calls) == 2:
            raise PermissionError(13, "Permission denied", str(arguments[0]))

    monkeypatch.setattr(example.subprocess, "run", run_nothing)
    with pytest.raises(SystemExit, match="mounted noexec"):
        example.main(["small_gguf", "--directory", str(tmp_path / "new"), "--quiet"])
    assert list((tmp_path / "tmp").iterdir()) == []


def test_the_folder_for_temporary_files_is_refused_below_another_accounts(
    tmp_path, monkeypatch, capsys
):
    """The example refuses a folder for temporary files below a folder that others can write in,
    and, as root with SUDO_UID set, below a folder of that account, as examples/serve_gguf.sh does:
    root would make the example there. (The second half needs root; the test runs either way, so
    that the suite's count of skips does not grow.)"""
    places = [tmp_path / "open" / "t"]
    places[0].mkdir(parents=True)
    (tmp_path / "open").chmod(0o777)
    if os.geteuid() == 0:  # the account that ran sudo is not trusted here
        places.append(tmp_path / "theirs" / "t")
        places[1].mkdir(parents=True)
        os.chown(tmp_path / "theirs", 54321, 54321)
        monkeypatch.setenv("SUDO_UID", "54321")
    for place in places:
        monkeypatch.setenv("TMPDIR", str(place))
        monkeypatch.setattr(tempfile, "tempdir", None)  # gettempdir() keeps what it found first
        with pytest.raises(SystemExit) as stop:
            example.main(["small_synthetic", "--quiet"])
        assert stop.value.code == 2
        assert str(place) in capsys.readouterr().err  # the refusal names this place
        assert list(place.iterdir()) == []
    if os.geteuid() == 0:
        assert os.environ["SUDO_UID"] == "54321"


@pytest.mark.parametrize("example_name", ["small_synthetic", "small_gguf"])
def test_a_folder_for_temporary_files_that_is_a_file_ends_in_a_sentence(
    example_name, tmp_path, monkeypatch, capsys
):
    """TMPDIR that names a file: a sentence and status 2, no traceback, nothing made."""
    (tmp_path / "a_file").write_text("")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "a_file"))
    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")  # no compiler runs here
    with pytest.raises(SystemExit) as stop:
        example.main([example_name, "--quiet"])
    assert stop.value.code not in (0, None)
    said = capsys.readouterr().err + str(stop.value.code)
    assert str(tmp_path / "a_file") in said and "Traceback" not in said
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a_file"]


@pytest.mark.parametrize("example_name", ["small_synthetic", "small_gguf"])
def test_a_folder_for_temporary_files_that_cannot_be_written_ends_in_a_sentence(
    example_name, tmp_path, monkeypatch, capsys
):
    """mkdtemp() refused (a TMPDIR that this account cannot write in, stood in for by a function,
    since root can write anywhere): a sentence and status 2, no traceback; and a TMPDIR two levels
    below a missing folder is named whole."""
    (tmp_path / "tmp").mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")

    def refused(**options):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(example.tempfile, "mkdtemp", refused)
    with pytest.raises(SystemExit) as stop:
        example.main([example_name, "--quiet"])
    assert stop.value.code == 2
    said = capsys.readouterr().err
    assert f"cannot make a folder in TMPDIR, {tmp_path / 'tmp'}: Permission denied" in said
    monkeypatch.setenv("TMPDIR", str(tmp_path / "gone" / "deeper"))
    with pytest.raises(SystemExit):
        example.main([example_name, "--quiet"])
    assert f"TMPDIR, {tmp_path / 'gone' / 'deeper'}, does not exist" in capsys.readouterr().err


def test_vllm_files_takes_a_dangling_link_for_a_file_that_stands(tmp_path):
    """A link whose target is missing, at one of vllm_files' three names, is refused before any file
    is written, as a file would be (os.path.lexists)."""
    model = tmp_path / "model"
    model.mkdir()
    shutil.copy(PACKAGE / "tests" / "configurations" / "tiny_without_clamps" / "policy.txt", model)
    (model / "tokenizer.json").symlink_to(tmp_path / "nowhere")
    done = make_example("vllm_files", model, "7")
    assert done.returncode != 0 and "Traceback" not in done.stderr
    assert "has tokenizer.json already. Nothing was written" in done.stderr
    assert sorted(path.name for path in model.iterdir()) == ["policy.txt", "tokenizer.json"]
    assert not (tmp_path / "nowhere").exists()


def test_try_limits_takes_a_dangling_link_at_replies_for_one_that_stands(tmp_path):
    """try_limits.py refuses a DIRECTORY whose replies/ is a link to nowhere, before any work."""
    pytest.importorskip("transformers")  # try_limits.py imports it first
    (tmp_path / "d" / "replies").parent.mkdir()
    (tmp_path / "d" / "replies").symlink_to(tmp_path / "nowhere")
    done = run(
        sys.executable,
        PACKAGE / "examples" / "try_limits.py",
        tmp_path / "d",
        env={**os.environ, "PYTHONPATH": str(PACKAGE)},
    )
    assert done.returncode == 2 and "exists already, and nothing is overwritten" in done.stderr
    assert not (tmp_path / "nowhere").exists()


def test_a_missing_folder_for_temporary_files_is_named_plainly(tmp_path, monkeypatch, capsys):
    """A TMPDIR that does not exist: a plain sentence, status 2, nothing made."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "gone"))
    with pytest.raises(SystemExit) as stop:
        example.main(["small_synthetic", "--quiet"])
    assert stop.value.code == 2
    assert f"TMPDIR, {tmp_path / 'gone'}, does not exist." in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_vllm_files_names_a_model_directory_that_does_not_exist(tmp_path):
    done = make_example("vllm_files", tmp_path / "gone", "7")
    assert done.returncode == 2
    assert (
        done.stderr
        == f"integer_model_vllm.example: the model directory {tmp_path / 'gone'} does not exist.\n"
    )
    done = make_example("vllm_files", tmp_path / "gone" / "deeper", "7")
    assert f"does not exist: {tmp_path / 'gone'} is missing." in done.stderr


def test_vllm_files_writes_by_the_real_path(tmp_path):
    """vllm_files given the model directory through a link of the caller's own: the three files are
    written by the real path. (The second look, at the real path, and the check that it did not
    turn guard against a race that no test here makes; this one only shows that the road stays.)"""
    model = tmp_path / "model"
    model.mkdir()
    shutil.copy(PACKAGE / "tests" / "configurations" / "tiny_without_clamps" / "policy.txt", model)
    (tmp_path / "link").symlink_to(model)
    done = make_example("vllm_files", tmp_path / "link", "7")
    assert done.returncode == 0, done.stderr
    assert f"into {model}:" in done.stdout
    assert (model / "config.json").is_file()


def test_vllm_files_refuses_a_model_directory_that_others_can_change(tmp_path):
    """vllm_files writes three files into the model directory that it is given; such a directory
    that its group or others can write in, without the sticky bit, is refused before any is written.
    """
    model = tmp_path / "model"
    model.mkdir()
    shutil.copy(PACKAGE / "tests" / "configurations" / "tiny_without_clamps" / "policy.txt", model)
    model.chmod(0o777)
    done = make_example("vllm_files", model, "7")
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr.startswith(f"integer_model_vllm.example: the model directory {model} lies ")
    assert "Here the directory must hold the model's files" in done.stderr
    assert [path.name for path in model.iterdir()] == ["policy.txt"]
    model.chmod(0o755)
    assert make_example("vllm_files", model, "7").returncode == 0


def test_without_a_directory_the_example_creates_a_temporary_one(tmp_path):
    done = run(
        sys.executable, "-m", "integer_model_vllm.example", "small_synthetic", "--quiet",
        cwd=PACKAGE, env={**os.environ, "TMPDIR": str(tmp_path)},
    )  # fmt: skip
    assert (done.returncode, done.stderr) == (0, "")
    assert [str(path) for path in tmp_path.iterdir()] == [done.stdout.strip()]


def test_only_the_checkout_that_holds_the_package_is_found(tmp_path):
    """A look-alike checkout above the current directory or above a copy of the package is not
    taken for the repository.
    """

    def look_alike(top):
        (top / "prover" / "examples").mkdir(parents=True)
        shutil.copytree(
            REPOSITORY / "prover" / "examples" / "target_small_f2",
            top / "prover" / "examples" / "target_small_f2",
        )

    def create(site):
        shutil.copytree(PACKAGE / "integer_model_vllm", site / "integer_model_vllm")
        command = [sys.executable, "-m", "integer_model_vllm.example", "small_synthetic", "--quiet"]
        environment = {**os.environ, "PYTHONPATH": str(site), "TMPDIR": str(tmp_path / "new")}
        return run(*command, cwd=tmp_path / "below" / "deeper", env=environment)

    look_alike(tmp_path)
    (tmp_path / "below" / "deeper").mkdir(parents=True)
    (tmp_path / "new").mkdir()
    done = create(tmp_path / "elsewhere" / "site")
    assert done.returncode != 0
    assert done.stdout == ""
    assert "--repository" in done.stderr
    assert list((tmp_path / "new").iterdir()) == []
    # A copy in a virtual environment in the checkout's package directory, as after the install in
    # integrations/vllm/integer_model/README.md, "Install and quick start", finds the checkout.
    look_alike(tmp_path / "checkout")
    done = create(
        tmp_path / "checkout" / "integrations" / "vllm" / "integer_model" / "venv" / "site"
    )
    assert (done.returncode, done.stderr) == (0, "")
    assert [str(path) for path in (tmp_path / "new").iterdir()] == [done.stdout.strip()]


def test_a_failure_reports_where_the_incomplete_example_is(tmp_path):
    """Also checks that g++ runs in make_model's own new folder (build_folder()), so that an empty
    CPATH entry (the current directory) finds none of the caller's headers, and that the folder is
    removed again.
    """
    (tmp_path / "bin").mkdir()
    fake_program(tmp_path / "bin" / "g++", '#!/bin/sh\npwd > "$0.directory"\nexit 1\n')
    (tmp_path / "tmp").mkdir(mode=0o700)
    environment = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    environment["TMPDIR"] = str(tmp_path / "tmp")  # make_model's folder below the test's own
    command = [
        sys.executable,
        "-m",
        "integer_model_vllm.example",
        "small_gguf",
        "--directory",
        tmp_path / "new",
    ]
    done = run(*command, cwd=PACKAGE, env=environment)
    assert done.returncode != 0
    assert done.stdout == ""
    assert "Traceback" not in done.stderr
    assert (
        done.stderr
        == f"g++ failed. The example is incomplete. What was written is in {tmp_path / 'new'}.\n"
    )
    directory = pathlib.Path((tmp_path / "bin" / "g++.directory").read_text().strip())
    assert directory.name.startswith("make_model.") and not directory.exists()


@pytest.mark.parametrize("absent", ["safetensors", "torch"])
def test_the_float_pair_without_safetensors_says_what_to_install(tmp_path, monkeypatch, absent):
    """example.main() exits before it creates the directory, also without PyTorch."""
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *more: None if name == absent else find_spec(name, *more),
    )
    with pytest.raises(SystemExit, match=f"needs the package {absent}"):
        example.main(["small_float_pair", "--directory", str(tmp_path / "new"), "--quiet"])
    assert not (tmp_path / "new").exists()


@pytest.fixture
def programs_that_print(tmp_path):
    """An environment with fake "vllm" and "curl" that print their arguments, variables and
    directory, and their standard input if an argument is "@-".
    """
    (tmp_path / "bin").mkdir()
    (tmp_path / "tmp").mkdir()
    for program in ("vllm", "curl"):
        fake_program(
            tmp_path / "bin" / program,
            f'#!/bin/sh\nprintf "%s\\n" {program} "$@" "its standard input:"\n'
            'case " $* " in *" @- "*) cat ;; esac\necho "its environment:"\nenv -0\npwd\n',
        )
    paths = [tmp_path / "bin", pathlib.Path(sys.executable).parent, os.environ["PATH"]]
    return {
        "PATH": os.pathsep.join(map(str, paths)),
        "PYTHONPATH": str(PACKAGE),
        "PYTHONDONTWRITEBYTECODE": "1",  # keeps __pycache__ out of the checkout
        "TMPDIR": str(tmp_path / "tmp"),
        "HOME": "/nonexistent",
    }


def printed(done):
    """A fake program's command line, and a dict of its variables, directory and standard input."""
    assert done.returncode == 0, done.stderr
    before, _, environment = done.stdout.partition("its environment:\n")
    lines = before.splitlines()
    start = min(lines.index(name) for name in ("vllm", "curl") if name in lines)
    middle = lines.index("its standard input:")
    # env -0 ends each variable with a zero byte, so a value may contain newlines.
    *pairs, directory = environment.split("\0")
    variables = dict(pair.split("=", 1) for pair in pairs)
    variables["its directory"] = directory.removesuffix("\n")
    variables["its standard input"] = "\n".join(lines[middle + 1 :])
    return lines[start:middle], variables


@pytest.mark.parametrize("name, window_length", [("small_synthetic", 64), ("full_size", 1024)])
def test_serve_sh_runs_the_command_that_the_example_prints(
    name, window_length, programs_that_print, tmp_path
):
    arguments, variables = printed(
        run(PACKAGE / "examples" / "serve.sh", name, env=programs_that_print, cwd=tmp_path)
    )
    (directory,) = (tmp_path / "tmp").iterdir()
    assert arguments == ["vllm", "serve", str(directory / "model")] + shlex.split(
        f"--host 127.0.0.1 --served-model-name {name} --load-format integer_model --max-num-seqs 1 "
        f"--max-model-len {window_length} --no-enable-prefix-caching --enforce-eager "
        "--generation-config vllm"
    )
    assert VARIABLES.items() <= variables.items()
    # vLLM's caches and user site go below the example's own directory, not the caller's HOME.
    assert variables["HOME"] == str(directory) and variables["PYTHONNOUSERSITE"] == "1"
    hint = make_example(name, "--directory", tmp_path / "second").stderr.splitlines()[-1]
    words = shlex.split(hint.replace(str(tmp_path / "second"), str(directory)))
    assert words[:4] == [f"{key}={variables[key]}" for key in example.ENVIRONMENT]
    assert words[4:] == arguments


def test_serve_sh_small_gguf_passes_the_digests_of_the_reference_copies(
    programs_that_print, tmp_path
):
    if not shutil.which("g++"):
        pytest.skip("needs g++, which builds the program that writes the GGUF files")
    done = run(
        PACKAGE / "examples" / "serve.sh", "small_gguf", env=programs_that_print, cwd=tmp_path
    )
    arguments, variables = printed(done)
    directory = pathlib.Path(
        done.stdout.splitlines()[0].removeprefix("The example's files are in ")
    )
    # The test writes into this directory below, so it must be under tmp_path.
    assert directory.resolve().is_relative_to((tmp_path / "tmp").resolve())
    # make_model is built and run in a folder of its own, and removed (build_folder()).
    assert sorted(path.name for path in directory.iterdir()) == ["model", "reference", "weights"]
    assert arguments[:3] == ["vllm", "serve", str(directory / "model")]
    assert arguments[3:-1] == shlex.split(
        "--host 127.0.0.1 --served-model-name small_gguf --load-format integer_model "
        "--max-num-seqs 1 --max-model-len 64 --no-enable-prefix-caching --enforce-eager "
        "--generation-config vllm "
        "--fingerprint-mode none --max-logprobs 0 --model-loader-extra-config"
    )
    assert {"tokenizer.json", "tokenizer_config.json"} <= set(os.listdir(directory / "model"))
    assert variables["VLLM_API_KEY"] == "example"
    assert variables["PYTHONSAFEPATH"] == "1"
    assert VARIABLES.items() <= variables.items()
    work = pathlib.Path(variables["TMPDIR"])
    assert variables["HOME"] == variables["its directory"] == str(work)
    assert str(work) in done.stdout.splitlines()[1]
    assert work.parent == tmp_path / "tmp"
    assert work.stat().st_mode & 0o777 == (directory / "reference").stat().st_mode & 0o777 == 0o700
    # vLLM gets the reference directory's path. No digest stands among the arguments, which every
    # account can read, and the script has written none: its working directory is empty.
    assert list(work.iterdir()) == []
    options = json.loads(arguments[-1])
    assert options == {"reference_directory": str(directory / "reference"), "reference_files": 5}
    assert not any(re.search("[0-9a-f]{64}", word) for word in arguments)
    assert not any(re.search("[0-9a-f]{64}", value) for value in [*variables.values(), done.stdout])
    # The vLLM loader replaces the path by the five digests, in memory, and loads with them.
    given = dict(options)
    replace_reference_directory(options, directory / "model")
    assert options == {
        "sha256": {
            name: hashlib.sha256((directory / "reference" / name).read_bytes()).hexdigest()
            for name in example.REFERENCE_FILES
        }
    }

    model = integer_model.load_from_folder(directory / "model", "cpu", options)
    assert isinstance(model.weights, BlockWeights)
    assert model.weights.sums_verified
    assert len(model.weights.digests) == 74
    assert greedy_reply(model) == [93, 45, 59, 289]  # as examples/request.sh states
    for changed in (directory / "model" / "KL.bin", directory / "weights" / "fold.txt"):
        changed.write_bytes(changed.read_bytes() + b"\n")
        with pytest.raises(integer_model.ParameterError, match=f"{changed.name} is not the file"):
            integer_model.load_from_folder(directory / "model", "cpu", options)
        shutil.copy(directory / "reference" / changed.name, changed)
    # With weights in GGUF files the two files of the set-up directory need copies of their own too.
    copy = directory / "reference" / "wsums.txt"
    copy.unlink()
    with pytest.raises(integer_model.ParameterError, match="wsums.txt': No such file"):
        replace_reference_directory(dict(given), directory / "model")
    copy.hardlink_to(directory / "weights" / "wsums.txt")
    with pytest.raises(integer_model.ParameterError, match="are one file under two names"):
        replace_reference_directory(dict(given), directory / "model")


def test_serve_sh_takes_a_relative_tmpdir_from_the_callers_directory(programs_that_print, tmp_path):
    """serve.sh enters the package's folder to start example.py: a relative TMPDIR still names a
    folder below the caller's directory."""
    (tmp_path / "rel").mkdir(mode=0o700)
    environment = {**programs_that_print, "TMPDIR": "rel"}
    arguments, _ = printed(
        run(PACKAGE / "examples" / "serve.sh", "small_synthetic", env=environment, cwd=tmp_path)
    )
    assert pathlib.Path(arguments[2]).parent.parent == tmp_path / "rel"


def test_serve_sh_imports_nothing_from_the_callers_directory(programs_that_print, tmp_path):
    (tmp_path / "integer_model_vllm").mkdir()
    (tmp_path / "integer_model_vllm" / "__init__.py").touch()
    (tmp_path / "integer_model_vllm" / "example.py").write_text('print("planted")\n')
    # example.py imports json. If serve.sh appended the caller's empty PYTHONPATH to its own entry,
    # leaving an empty entry (the current directory), and ran python3 in the caller's directory,
    # this json.py would shadow the standard library's.
    (tmp_path / "json.py").write_text('raise SystemExit("planted json.py was imported")\n')
    (tmp_path / "link.sh").symlink_to(PACKAGE / "examples" / "serve.sh")
    # An empty PYTHONPATH adds nothing to Python's path, and an absolute entry does not depend on
    # the current directory. serve.sh rejects an empty or relative entry (tested below).
    environment = {**programs_that_print, "PYTHONPATH": ""}
    for script in (PACKAGE / "examples" / "serve.sh", tmp_path / "link.sh"):
        for variables in (environment, {**environment, "PYTHONPATH": "/nonexistent"}):
            arguments, seen = printed(run(script, "small_synthetic", env=variables, cwd=tmp_path))
            assert pathlib.Path(arguments[2]).parent.parent == tmp_path / "tmp"
            # vLLM starts in the example's new directory, not in the caller's.
            assert seen["its directory"] == str(pathlib.Path(arguments[2]).parent)
    # This fake python3 ignores PYTHONSAFEPATH, as Python 3.10 and older do.
    python3 = tmp_path / "bin" / "python3"
    python3.write_text(
        f'#!/bin/sh\nunset PYTHONSAFEPATH\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    python3.chmod(0o755)
    arguments, _ = printed(
        run(PACKAGE / "examples" / "serve.sh", "small_synthetic", env=environment, cwd=tmp_path)
    )
    assert pathlib.Path(arguments[2]).parent.parent == tmp_path / "tmp"


def test_serve_sh_stops_unless_it_knows_its_own_place(programs_that_print, tmp_path):
    """serve.sh imports and starts nothing when copied outside a checkout, even with a package
    integer_model_vllm in the caller's directory. It takes realpath from the system's directories
    ahead of any on the caller's PATH."""
    (tmp_path / "integer_model_vllm").mkdir()
    (tmp_path / "integer_model_vllm" / "__init__.py").touch()
    (tmp_path / "integer_model_vllm" / "example.py").write_text('print("planted")\n')
    script = PACKAGE / "examples" / "serve.sh"
    shutil.copy(script, tmp_path / "serve.sh")
    done = run(tmp_path / "serve.sh", "small_synthetic", env=programs_that_print, cwd=tmp_path)
    assert (done.returncode, done.stdout) == (1, "")
    assert "by its path inside a checkout of FlashProve" in done.stderr
    assert list((tmp_path / "tmp").iterdir()) == []
    fake_program(tmp_path / "bin" / "realpath", "#!/bin/sh\necho /planted/examples/serve.sh\n")
    done = run(script, "small_synthetic", env=programs_that_print, cwd=tmp_path)
    arguments, _ = printed(done)
    assert arguments[:2] == ["vllm", "serve"]


def test_the_scripts_refuse_settings_that_undo_their_precautions(programs_that_print, tmp_path):
    """If one of three vLLM variables is set, even to the empty string, serve.sh and serve_gguf.sh
    stop before they create anything or start vLLM. Why: OPERATING.md, "Settings".
    """
    (tmp_path / "model").mkdir()
    (tmp_path / "reference").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "reference" / name).write_text(name)
    examples = PACKAGE / "examples"
    for variable in ("VLLM_SERVER_DEV_MODE", "VLLM_MODEL_REDIRECT_PATH", "VLLM_LOGGING_LEVEL"):
        environment = {**programs_that_print, "VLLM_API_KEY": "k", variable: ""}
        for command in (
            [examples / "serve.sh", "small_synthetic"],
            [examples / "serve_gguf.sh", "reference", "model", "name"],
        ):
            done = run(*command, env=environment, cwd=tmp_path)
            assert (done.returncode, done.stdout) == (2, ""), (variable, command)
            assert done.stderr == (
                f"{command[0]}: {variable} is set, which can undo this script's precautions. "
                "Unset it.\n"
            )
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize(
    "python_path", [":/a", "/a::/b", "/a:", ".", "..", "../src", "src", "/a:src"]
)
def test_the_scripts_refuse_an_empty_or_relative_entry_of_pythonpath(
    python_path, programs_that_print, tmp_path
):
    """An empty or relative PYTHONPATH entry can name a directory that the scripts do not choose,
    despite PYTHONSAFEPATH=1: ".." names the parent of the directory in which vLLM starts. Both
    scripts stop before they create anything or start vLLM.
    """
    (tmp_path / "model").mkdir()
    (tmp_path / "reference").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "reference" / name).write_text(name)
    examples = PACKAGE / "examples"
    environment = {**programs_that_print, "VLLM_API_KEY": "k", "PYTHONPATH": python_path}
    for command in (
        [examples / "serve.sh", "small_synthetic"],
        [examples / "serve_gguf.sh", "reference", "model", "name"],
    ):
        done = run(*command, env=environment, cwd=tmp_path)
        assert (done.returncode, done.stdout) == (2, ""), command
        assert done.stderr == (
            f"{command[0]}: PYTHONPATH has an empty or relative entry; use absolute paths, "
            "or unset it.\n"
        )
    assert list((tmp_path / "tmp").iterdir()) == []


def test_the_scripts_take_an_absolute_or_absent_pythonpath(programs_that_print, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "reference").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "reference" / name).write_text(name)
    examples = PACKAGE / "examples"
    for python_path in (None, "", "/a", "/a:/b"):
        environment = {**programs_that_print, "VLLM_API_KEY": "k"}
        if python_path is None:
            del environment["PYTHONPATH"]
        else:
            environment["PYTHONPATH"] = python_path
        for command in (
            [examples / "serve.sh", "small_synthetic"],
            [examples / "serve_gguf.sh", "reference", "model", "name"],
        ):
            arguments, _ = printed(run(*command, env=environment, cwd=tmp_path))
            assert arguments[:2] == ["vllm", "serve"], (python_path, command)


def test_the_scripts_say_how_they_are_called(programs_that_print, tmp_path):
    examples = PACKAGE / "examples"
    gguf = examples / "serve_gguf.sh"
    for command in (
        [examples / "serve.sh"],
        [examples / "serve.sh", "other"],
        [gguf, "a", "b"],
        [gguf, "a", "", "c"],
    ):
        done = run(*command, env={**programs_that_print, "VLLM_API_KEY": "k"}, cwd=tmp_path)
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr.startswith("usage: ")
    done = run(examples / "serve_gguf.sh", "a", "b", "c", env=programs_that_print, cwd=tmp_path)
    assert done.returncode != 0
    assert "VLLM_API_KEY" in done.stderr
    assert done.stdout == ""
    # A missing reference copy, each of the five in turn, stops the script before vLLM starts.
    (tmp_path / "model").mkdir()
    (tmp_path / "incomplete").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "incomplete" / name).touch()
    command = [examples / "serve_gguf.sh", "incomplete", "model", "name"]
    for name in example.REFERENCE_FILES:
        (tmp_path / "incomplete" / name).rename(tmp_path / "aside")
        done = run(*command, env={**programs_that_print, "VLLM_API_KEY": "k"}, cwd=tmp_path)
        assert (done.returncode, done.stdout) == (1, "")
        copy = tmp_path / "incomplete" / name
        assert done.stderr == f"{gguf}: cannot read the reference copy {copy}\n"
        (tmp_path / "aside").rename(tmp_path / "incomplete" / name)
    # So does a model directory that cannot be entered.
    command = [examples / "serve_gguf.sh", "incomplete", "missing", "name"]
    done = run(*command, env={**programs_that_print, "VLLM_API_KEY": "k"}, cwd=tmp_path)
    assert (done.returncode, done.stdout) == (1, "")
    assert "cannot enter the model directory missing" in done.stderr
    assert list((tmp_path / "tmp").iterdir()) == []


def test_serve_gguf_sh_takes_relative_paths(programs_that_print, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "reference").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "reference" / name).write_text(name)
    environment = {**programs_that_print, "VLLM_API_KEY": "k", "TMPDIR": "tmp"}
    done = run(
        PACKAGE / "examples" / "serve_gguf.sh",
        "reference",
        "model",
        "name",
        env=environment,
        cwd=tmp_path,
    )
    arguments, variables = printed(done)
    assert arguments[:3] == ["vllm", "serve", str(tmp_path / "model")]
    assert arguments[arguments.index("--max-model-len") + 1] == "1024"
    assert json.loads(arguments[-1]) == {
        "reference_directory": str(tmp_path / "reference"),
        "reference_files": 5,
    }
    (work,) = (tmp_path / "tmp").iterdir()
    assert list(work.iterdir()) == []
    # A link and then ..: the path goes on as it was given, so the system finds the same directory
    # for the loader as for the script. A directory named - is a path like any other.
    shutil.copytree(tmp_path / "reference", tmp_path / "far" / "reference")
    (tmp_path / "far" / "below").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "far" / "below")
    shutil.copytree(tmp_path / "reference", tmp_path / "-")
    for typed, found in (
        ("link/../reference", tmp_path / "far" / "reference"),
        ("-", tmp_path / "-"),
    ):
        script = PACKAGE / "examples" / "serve_gguf.sh"
        others, _ = printed(run(script, typed, "model", "name", env=environment, cwd=tmp_path))
        wanted = {"reference_directory": f"{tmp_path}/{typed}", "reference_files": 5}
        assert json.loads(others[-1]) == wanted and os.path.samefile(tmp_path / typed, found)
    assert variables["TMPDIR"] == variables["HOME"] == variables["its directory"] == str(work)


def test_serve_gguf_sh_refuses_a_path_with_a_quote_a_backslash_or_a_control_character(
    programs_that_print, tmp_path
):
    """serve_gguf.sh writes the reference directory's path into JSON as it is, and prints its new
    directory's path as it is, so either path with a quote, a backslash or a control character is
    refused, before anything is made. A relative path brings the caller's directory into it. A ..
    part in TMPDIR is resolved before mktemp (the script goes on by the real path), so that case is
    taken.
    """
    script = PACKAGE / "examples" / "serve_gguf.sh"

    def folders(top):
        for folder in (top / "model", top / "reference", top / "tmp"):
            folder.mkdir(parents=True)
        for name in example.REFERENCE_FILES:
            (top / "reference" / name).write_text(name)
        return top

    good = folders(tmp_path / "good")
    for bad in ('c"d', "c\\d", "c\nd", "c\x85d"):  # \x85: a control character beyond ASCII
        top = folders(tmp_path / bad)
        environment = {**programs_that_print, "VLLM_API_KEY": "k", "TMPDIR": "tmp"}
        done = run(script, good / "reference", "model", "name", env=environment, cwd=top)
        assert done.returncode == 2, done.stderr  # refused by the check, before mktemp
        assert "has a quote, a backslash or a control character in its path" in done.stderr
        assert list((top / "tmp").iterdir()) == []
        if bad == "c\x85d":  # JSON holds it as it is
            continue
        environment["TMPDIR"] = str(good / "tmp")
        done = run(script, "reference", "model", "name", env=environment, cwd=top)
        assert (done.returncode, done.stdout) == (2, ""), done.stderr
        assert "which this script cannot hand on to vLLM. Nothing was made." in done.stderr
        assert list((good / "tmp").iterdir()) == []
    # A newline at the end of the name: refused, and not taken for the folder beside it.
    (good / "reference\n").mkdir()
    for name in example.REFERENCE_FILES:
        (good / "reference\n" / name).write_text(name)
    done = run(script, "reference\n", "model", "name", env=environment, cwd=good)
    assert (done.returncode, done.stdout) == (2, ""), done.stderr
    # By a link whose own path has no such character, that folder is taken: the path goes to vLLM as
    # it was given, so it still leads there, and not to the folder beside it.
    (good / "link").symlink_to(good / "reference\n")
    arguments, _ = printed(run(script, "link", "model", "name", env=environment, cwd=good))
    assert json.loads(arguments[-1]) == dict(reference_directory=f"{good}/link", reference_files=5)
    top = folders(tmp_path / "dots")
    environment = {**programs_that_print, "VLLM_API_KEY": "k", "TMPDIR": f"{top}/tmp/../tmp"}
    _, variables = printed(run(script, "reference", "model", "name", env=environment, cwd=top))
    assert pathlib.Path(variables["TMPDIR"]).parent == top / "tmp"


def test_serve_gguf_sh_passes_on_middleware_and_no_other_option(programs_that_print, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "reference").mkdir()
    for name in example.REFERENCE_FILES:
        (tmp_path / "reference" / name).write_text(name)
    command = [PACKAGE / "examples" / "serve_gguf.sh", "reference", "model", "name"]
    environment = {**programs_that_print, "VLLM_API_KEY": "k"}
    for more in (
        ["--host", "0.0.0.0"],
        ["64", "--max-logprobs", "5"],
        ["64", "65"],
        ["--middleware"],
        ["--middleware", ""],
        ["--middleware", "a b"],
        ["--middleware", "--host"],
        ["--middleware=a.B"],
        ["--middleware", "a.B", "--model-loader-extra-config", "{}"],
        ["--middleware", "a.B", "--middleware"],
    ):
        done = run(*command, *more, env=environment, cwd=tmp_path)
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr.startswith("usage: ")
    assert list((tmp_path / "tmp").iterdir()) == []
    for length, pairs in (
        ([], ["--middleware", "a.b.C"]),
        (["64"], ["--middleware", "a.B", "--middleware", "c_d.E2"]),
    ):
        arguments, _ = printed(run(*command, *length, *pairs, env=environment, cwd=tmp_path))
        # The pairs come before the script's own options, so they cannot override any of them.
        assert arguments[3 : len(pairs) + 4] == [*pairs, "--host"]
        assert arguments[arguments.index("--max-model-len") + 1] == (length or ["1024"])[0]


def test_request_sh(programs_that_print, tmp_path):
    script = PACKAGE / "examples" / "request.sh"
    arguments, _ = printed(run(script, env=programs_that_print, cwd=tmp_path))
    assert "Authorization" not in " ".join(arguments)
    assert json.loads(arguments[arguments.index("-d") + 1]) == {
        "model": "small_synthetic",
        "prompt": [17, 400, 3],
        "max_tokens": 4,
        "temperature": 0,
        "return_token_ids": True,
    }
    environment = {**programs_that_print, "VLLM_API_KEY": "example"}
    arguments, variables = printed(run(script, "small_gguf", "1,2", env=environment, cwd=tmp_path))
    # The key reaches curl on standard input, not among the arguments, which the process list shows.
    assert arguments[-2:] == ["-H", "@-"]
    assert "example" not in " ".join(arguments)
    assert variables["its standard input"] == "Authorization: Bearer example"
    assert json.loads(arguments[arguments.index("-d") + 1])["prompt"] == [1, 2]
    for words in (
        ["small_synthetic", '1],"max_tokens":9,"x":[2'],
        ['a"b'],
        ["small_synthetic", "1 2"],
        ["small_synthetic", "a,b"],
        ["small_synthetic", "1,,2"],
        ["small_synthetic", "1", "2"],
    ):
        done = run(script, *words, env=programs_that_print, cwd=tmp_path)
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr.startswith("usage: ")


def test_score_reply(tmp_path):
    example.main(["small_synthetic", "--directory", str(tmp_path / "new"), "--quiet"])
    # The model's parameter file (policy.txt) with a clipped-gap policy in place of exact match.
    clipped_gap = tmp_path / "policy.clipped_gap.txt"
    limits = {"C": "100000", "theta": "40000", "cap": "2"}  # invented limits, for this test only
    clipped_gap.write_text(
        "".join(
            f"{name} {limits.get(name, rest)}\n"
            for name, rest in (
                line.split(maxsplit=1)
                for line in (tmp_path / "new" / "model" / "policy.txt").read_text().splitlines()
            )
        )
    )

    def score(reply, *options, status=0):
        command = [
            sys.executable,
            PACKAGE / "examples" / "score_reply.py",
            tmp_path / "new" / "model",
            "17,400,3",
            reply,
        ]
        done = run(*command, *options, env={**os.environ, "PYTHONPATH": str(PACKAGE)})
        assert done.returncode == status, done.stderr
        assert "Traceback" not in done.stderr, done.stderr
        return (done.stdout or done.stderr).splitlines()

    lines = score("365,287,374,290")
    assert lines[:4] == [f"token {token}: gap 0" for token in (365, 287, 374, 290)]
    assert lines[4:] == [
        "mean clipped cost: 256.0 (at most theta = 256)",
        "costs above C = 256: 0 (at most cap = 0)",
        "meets the policy (a score, not a verification)",
    ]
    lines = score("365,287,374,291", status=1)  # under exact match, one other token fails
    assert lines[3].startswith("token 291: gap ")
    assert lines[3].endswith(" (greedy choice: 290)")
    assert lines[-1] == "fails the policy (a score, not a verification)"
    lines = score("365,287,374,291", "--policy", clipped_gap)  # (3 * 256 + 100000) / 4
    assert lines[4:] == [
        "mean clipped cost: 25192.0 (at most theta = 40000)",
        "costs above C = 100000: 1 (at most cap = 2)",
        "meets the policy (a score, not a verification)",
    ]
    assert score("365,9999", status=1)[0].startswith("No score: ")  # an id outside the vocabulary
    assert "is not a parameter file" in score("365", "--policy", tmp_path / "none", status=2)[-1]


ODD = "gone\n\x1b[2Ja line of its own"  # a name with a newline and an escape character
ODD_AS_PRINTED = "gone\\n\\x1b[2Ja line of its own"


def is_one_line_with_the_name_escaped(said):
    return ODD_AS_PRINTED in said and "\x1b" not in said and said.find("\n") == len(said) - 1


def test_a_path_that_does_not_exist_is_printed_with_its_control_characters_escaped(
    tmp_path, monkeypatch, capsys
):
    """A newline or an escape character in a path must not reach the terminal as it is: what follows
    it could pass for another line of the program's, or move the cursor."""
    for words in (
        ("vllm_files", tmp_path / ODD, "7"),
        ("vllm_files", tmp_path / ODD / "deeper", "7"),
        ("small_synthetic", "--directory", tmp_path / ODD / "new"),
    ):
        done = make_example(*words)
        assert done.returncode == 2 and is_one_line_with_the_name_escaped(done.stderr), done.stderr
    monkeypatch.setenv("TMPDIR", str(tmp_path / ODD))
    for call in (lambda: example.main(["small_synthetic", "--quiet"]), example.build_folder):
        with pytest.raises(SystemExit) as stop:
            call()
        said = capsys.readouterr().err
        assert stop.value.code == 2 and is_one_line_with_the_name_escaped(said), said
    assert list(tmp_path.iterdir()) == []


def test_try_limits_prints_a_directory_that_does_not_exist_escaped(tmp_path):
    pytest.importorskip("transformers")  # try_limits.py imports it first
    done = run(
        sys.executable,
        PACKAGE / "examples" / "try_limits.py",
        tmp_path / ODD,
        env={**os.environ, "PYTHONPATH": str(PACKAGE)},
    )
    assert done.returncode == 2 and is_one_line_with_the_name_escaped(done.stderr), done.stderr
    assert list(tmp_path.iterdir()) == []


def last_words(call, capsys):
    """Everything that `call` prints before it ends the run, with the words that it ends on."""
    with pytest.raises(SystemExit) as stop:
        call()
    printed, code = capsys.readouterr(), stop.value.code
    return printed.out + printed.err + (f"{code}\n" if isinstance(code, str) else "")


def turn_the_real_path(patch, folder):
    """Each place that looks at a folder asks for its real path four times: for itself, in each of
    the routine's two looks, and once more to see that it has not turned. For `folder`, which is
    its own real path, the fourth answer is another."""
    real, calls = os.path.realpath, []

    def realpath(path, **options):
        calls.extend([path] if os.fspath(path) == str(folder) else [])
        return real(path, **options) + ("x" if calls[3:] == [path] else "")

    patch.setattr(os.path, "realpath", realpath)


def refuse(error):
    """A function that raises `error`, whatever it is called with."""

    def refused(*arguments, **options):
        raise error

    return refused


def test_vllm_files_prints_the_model_directory_escaped_in_every_sentence(
    tmp_path, monkeypatch, capsys
):
    """Each refusal of vllm_files in turn, then its report of what it wrote, on one directory."""
    model = tmp_path / ODD
    model.mkdir()

    def said(text=None, patch=None):
        if text is not None:
            (model / "policy.txt").write_text(text)
        with monkeypatch.context() as context:
            if patch:
                patch(context)
            return last_words(lambda: example.main(["vllm_files", str(model), "7"]), capsys)

    stands = FileExistsError(17, "File exists", str(model / "config.json"))
    for words, given in (
        ("Cannot read the parameter file", {}),
        ("is not a parameter file", {"text": "x\n"}),
        ("must be at least 2", {"text": "V 1\nlt 5\n"}),
        (
            "changed while it was looked at",
            {"text": "V 50\nlt 5\n", "patch": lambda context: turn_the_real_path(context, model)},
        ),
        (
            "stay in",
            {"patch": lambda context: context.setattr(example, "write_vllm_files", refuse(stands))},
        ),
    ):
        assert words in (all_said := said(**given)), all_said
        assert is_one_line_with_the_name_escaped(all_said), all_said
    assert os.listdir(model) == ["policy.txt"]
    example.main(["vllm_files", str(model), "7"])
    wrote = capsys.readouterr().out
    assert wrote.startswith("Wrote ") and is_one_line_with_the_name_escaped(wrote), wrote
    assert "already. Nothing was written" in (all_said := said()), all_said
    assert is_one_line_with_the_name_escaped(all_said), all_said


def test_the_example_prints_the_folder_for_temporary_files_escaped_in_every_sentence(
    tmp_path, monkeypatch, capsys
):
    """Two sentences, each at two places: where small_gguf's folder for make_model is made, and
    where an example without --directory is."""
    (tmp_path / ODD).mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(tmp_path / ODD))
    denied = refuse(PermissionError(13, "Permission denied"))
    for call in (example.build_folder, lambda: example.main(["small_synthetic", "--quiet"])):
        for words, patch in (
            ("changed while it was looked at", lambda c: turn_the_real_path(c, tmp_path / ODD)),
            ("cannot make a folder", lambda c: c.setattr(example.tempfile, "mkdtemp", denied)),
        ):
            with monkeypatch.context() as context:
                patch(context)
                all_said = last_words(call, capsys)
            assert words in all_said and is_one_line_with_the_name_escaped(all_said), all_said
    assert list((tmp_path / ODD).iterdir()) == []


def test_the_example_prints_its_directory_escaped_in_every_sentence(tmp_path, monkeypatch, capsys):
    """The sentences that name the new directory, a file in it, a program that failed, or the
    checkout. Two lines are meant to have the path as it is and are not tried here: the new
    directory's path on standard output, which scripts read, and the command to serve the example.
    """
    odd = tmp_path / ODD
    odd.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(odd))

    def make(directory, name="small_synthetic"):
        return last_words(lambda: example.main([name, "--directory", str(directory)]), capsys)

    all_said = last_words(lambda: example.find_repository(str(odd)), capsys)
    assert "is not the top" in all_said and is_one_line_with_the_name_escaped(all_said), all_said
    assert make(tmp_path / "none" / "new") == (
        f"integer_model_vllm.example: Cannot create {tmp_path / 'none' / 'new'}: "
        f"{tmp_path / 'none'} does not exist. Nothing was made.\n"
    )
    assert "File exists" in (all_said := make(odd)), all_said
    assert is_one_line_with_the_name_escaped(all_said), all_said
    for number, (words, error) in enumerate(
        (
            (" failed. ", subprocess.CalledProcessError(1, [str(odd)])),
            ("nothing is overwritten", FileExistsError(17, "File exists", str(odd / "a"))),
            ("nothing is overwritten", FileExistsError(f"{odd / 'a'} exists already.")),
            ("No space left", OSError(28, "No space left on device", str(odd / "a"))),
        )
    ):
        # First into a directory with a plain name, where the sentence's other path alone is odd.
        with monkeypatch.context() as context:
            context.setattr(example, "make_synthetic", refuse(error))
            first, second = make(tmp_path / f"new{number}"), make(odd / f"new{number}")
        for all_said in (first, second):
            assert words in all_said and is_one_line_with_the_name_escaped(all_said), all_said
        assert second.count(ODD_AS_PRINTED) == 2, second
    calls = []

    def cannot_run_the_second(arguments, **options):  # the first is g++, the second make_model
        calls.append(arguments)
        if len(calls) % 2 == 0:
            raise PermissionError(13, "Permission denied", str(arguments[0]))

    monkeypatch.setattr(example.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setattr(example.subprocess, "run", cannot_run_the_second)
    for new in (tmp_path / "plain", odd / "new"):  # make_model's folder lies in TMPDIR either way
        assert "noexec" in (all_said := make(new, "small_gguf")), all_said
        assert is_one_line_with_the_name_escaped(all_said), all_said
    assert all_said.count(ODD_AS_PRINTED) == 2, all_said


# Two modules that stand in for transformers and for this package when examples/try_limits.py runs,
# so that it reaches in seconds the sentences that it prints after its models have loaded. The first
# also brings the fault that $FAULT names, if it is one of its three; the script imports it before
# it looks at DIRECTORY.
STAND_IN_FOR_TRANSFORMERS = """import os, pathlib, sys, types, torch
if os.environ["FAULT"] == "the real path turns":  # at the fourth question about DIRECTORY, the last
    real, calls = os.path.realpath, []
    def realpath(path, **options):
        calls.extend([path] if os.fspath(path) == sys.argv[1] else [])
        return real(path, **options) + ("x" if calls[3:] == [path] else "")
    os.path.realpath = realpath
def stands(self, **options):
    raise FileExistsError(17, "File exists", str(self))
def full(self, **options):
    raise OSError(28, "No space left on device", str(self))
if os.environ["FAULT"] in ("stands", "full"):
    pathlib.Path.mkdir = {"stands": stands, "full": full}[os.environ["FAULT"]]
class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(path, **options):
        new = lambda ids: types.SimpleNamespace(logits=torch.zeros(*ids.shape, 8))
        new.model, new.eval = types.SimpleNamespace(layers=[]), lambda: new
        return new
"""
STAND_IN_FOR_THE_PACKAGE = """import types
def load_from_folder(folder, device):
    parameters = types.SimpleNamespace(L=0, T=64, V=8, has_experts=lambda layer: False)
    return types.SimpleNamespace(parameters=parameters)
"""


@pytest.mark.parametrize(
    "fault, words, status, paths",
    [
        ("the real path turns", "changed while it was looked at", 2, 1),
        ("replies/ is there", "exists already, and nothing is overwritten", 2, 1),
        ("the model is none", "No counts: ", 1, 1),
        ("stands", "exists already; nothing is overwritten", 1, 2),
        ("full", "No space left on device; nothing is overwritten", 1, 2),
    ],
)
def test_try_limits_prints_its_directory_escaped_in_every_sentence(
    fault, words, status, paths, tmp_path
):
    pytest.importorskip("torch")
    path = [tmp_path / "stand_in"]
    path[0].mkdir()
    (tmp_path / "stand_in" / "transformers.py").write_text(STAND_IN_FOR_TRANSFORMERS)
    for name in ("model", "float_model") + (("replies",) if fault == "replies/ is there" else ()):
        (tmp_path / ODD / name).mkdir(parents=True)
    (tmp_path / ODD / "model" / "policy.txt").write_text("sh 8\n")
    if fault == "the model is none":  # the package itself words the refusal, with the path in it
        (tmp_path / ODD / "model" / "integer_model.json").write_text("no JSON\n")
        path.append(PACKAGE)
    else:
        (tmp_path / "stand_in" / "integer_model.py").write_text(STAND_IN_FOR_THE_PACKAGE)
    done = run(
        sys.executable,
        PACKAGE / "examples" / "try_limits.py",
        tmp_path / ODD,
        "--replies",
        "64",
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, path)), "FAULT": fault},
    )
    last = done.stderr.splitlines(keepends=True)[-1:]  # argparse puts its usage before a refusal
    assert (done.returncode, done.stdout) == (status, ""), done.stderr
    assert words in done.stderr and "\x1b" not in done.stderr, done.stderr
    assert is_one_line_with_the_name_escaped("".join(last)), done.stderr
    assert last[0].count(ODD_AS_PRINTED) == paths, done.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
