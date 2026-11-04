# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests for dummy_service.py, a stand-in for the proving service (cp_prove -spool).

They check its responses, when it starts and stops, and that it deletes nothing through a symbolic
link or outside its spool directory. The service accepts only a spool directory below
tempfile.gettempdir(), so the tests set TMPDIR to tmp_path for it, or patch that function.
"""

import os
import select
import signal
import subprocess
import sys
import time

import pytest

import dummy_service
import spool

SCRIPT = os.path.join(os.path.dirname(__file__), "dummy_service.py")
CHILD_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p not in ("", os.getcwd()))}  # pytest's import path, without the caller's folder
COMMITMENT_ID = "ab" * 32
TOKENS = 8
REQUEST_SIZE = 4 * (TOKENS + 2) + spool.NONCE_BYTES
NOT_A_REQUEST = bytes(REQUEST_SIZE)  # the tests' names for it are not the hash of its bytes


@pytest.fixture(autouse=True)
def with_the_default_sigchld():
    """Under an ignored SIGCHLD, which can be inherited, the kernel reaps a test's children as soon
    as they exit: their exit codes are lost, and a PID that a test still holds can pass to another
    process."""
    set_handler = signal.signal
    before = set_handler(signal.SIGCHLD, signal.SIG_DFL)
    yield
    set_handler(signal.SIGCHLD, before)


# This fixture's service takes its first batch only once two requests are waiting (--first-take 2).
@pytest.fixture
def service(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(mode=0o700)
    # The front end must write its mark file (mark.txt) before the service creates subdirectories.
    spool.make_private_dirs(str(spool_dir), "in", commitment_id=COMMITMENT_ID)
    argv = [sys.executable, "-B", SCRIPT, str(spool_dir), "--tokens", str(TOKENS)]
    argv += ["--batch", "2", "--first-take", "2"]
    env = {**CHILD_ENV, "TMPDIR": str(tmp_path)}
    process = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL)
    yield str(spool_dir)
    if process.poll() is None:
        process.kill()
    process.wait()


def submit(spool_dir, first_token, start=2, end=5, change=None):
    """Submit a request whose tokens count up from first_token; token `change`, if any, is 1 larger.

    Return (request ID, tag key); spool.read_response() takes both."""
    tokens = list(range(first_token, first_token + TOKENS))
    if change is not None:
        tokens[change] += 1
    request = spool.make_request(tokens, start, end)
    return spool.submit(spool_dir, request, COMMITMENT_ID, with_key=True)


# Fails unless every request has a response within 20 seconds.
def wait_for_all(spool_dir, request_ids):
    assert spool.wait_for_responses(spool_dir, request_ids, 20) == set(request_ids)


def test_responses_pass_the_front_ends_checks(service):
    submitted = [submit(service, token) for token in (1, 2, 3)]
    wait_for_all(service, [request_id for request_id, _ in submitted])
    for request_id, key in submitted:
        response = spool.read_response(service, request_id, key=key)
        assert (response.failed, response.faults) == (False, [])
        # the length of dummy_proof()'s result
        assert response.files["proof"]["size"] == 200
    assert os.listdir(os.path.join(service, "work")) == []


@pytest.mark.parametrize(
    "change, failed", [(2, True), (3, True), (5, True), (1, False), (6, False)]
)
def test_no_proof_if_a_checked_token_does_not_count_up(service, change, failed):
    # With start=2, end=5, positions 2 to 4 check tokens 3 to 5, each against the token before it.
    submitted = [submit(service, 1, change=change), submit(service, 2)]
    wait_for_all(service, [request_id for request_id, _ in submitted])
    responses = [spool.read_response(service, request_id, key=key) for request_id, key in submitted]
    assert [response.failed for response in responses] == [failed, False]
    assert [response.faults for response in responses] == [[], []]


@pytest.mark.parametrize("end, next_token", [(TOKENS, 1 + TOKENS), (TOKENS - 1, 0)])
def test_the_next_token_is_given_if_the_checked_range_reaches_the_end(service, end, next_token):
    request_id = submit(service, 1, end=end)[0]
    wait_for_all(service, [request_id, submit(service, 2)[0]])
    assert spool.read_words(os.path.join(service, "out", request_id + ".y")) == [next_token]


def test_the_first_batch_waits_for_first_take_requests(service):
    with open(os.path.join(service, "in", "cd" * 32 + ".req"), "xb") as f:
        f.write(bytes(REQUEST_SIZE + 4))  # of another length, so it does not count
    request_id, _ = submit(service, 1)
    assert not spool.wait_for_responses(service, [request_id], 0.5)
    other, _ = submit(service, 2)
    wait_for_all(service, [request_id, other])
    times = {spool.read_response(service, name).tick_ns for name in (request_id, other)}
    assert len(times) == 1  # one batch


@pytest.mark.parametrize(
    "size, stays",
    [(REQUEST_SIZE, False), (REQUEST_SIZE - 4, True), (REQUEST_SIZE + 4, True), (0, True)],
)
def test_a_file_that_is_not_a_request_is_deleted_if_it_has_a_requests_length(service, size, stays):
    good = [submit(service, token)[0] for token in (1, 2)]
    wait_for_all(service, good)
    with open(os.path.join(service, "in", "cd" * 32 + ".req"), "xb") as f:
        f.write(bytes(size))  # not the content that the name is the hash of
    last, _ = submit(service, 3)
    wait_for_all(service, [last])
    left = [name for name in os.listdir(os.path.join(service, "in")) if name.endswith(".req")]
    assert left == (["cd" * 32 + ".req"] if stays else [])
    assert "cd" * 32 not in spool.finished_requests(service)


def run_script(tmp_path, *argv):
    env = {**CHILD_ENV, "TMPDIR": str(tmp_path)}
    argv = [sys.executable, "-B", SCRIPT, *argv]
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=20)


# A spool directory with the front end's mark file and one waiting request.
@pytest.fixture
def used_spool(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(mode=0o700)
    submit(str(spool_dir), 1)
    return spool_dir


@pytest.fixture
def start(tmp_path, used_spool):
    """Return a function that starts the service and waits for its first line.

    By then the service has deleted an empty file "stop" from before, so a later one ends it."""
    started = []

    def start_service():
        argv = [sys.executable, "-B", SCRIPT, str(used_spool), "--tokens", str(TOKENS)]
        env = {**CHILD_ENV, "TMPDIR": str(tmp_path)}
        started.append(subprocess.Popen(argv, env=env, stdout=subprocess.PIPE))
        assert started[-1].stdout.readline().startswith(b"dummy_service.py: a stand-in")
        return started[-1]

    yield start_service
    for process in started:
        if process.poll() is None:
            process.kill()
        process.communicate()


# For make_foreign_stop(): entries named "stop" that are not an empty regular file.
FOREIGN = ["not empty", "directory", "link to nothing", "link to empty"]


def make_foreign_stop(spool_dir, kind):
    if kind == "not empty":  # linked once it has content: the service must never see it empty
        (spool_dir.parent / "not empty").write_text("x")
        os.link(spool_dir.parent / "not empty", spool_dir / "stop")
    elif kind == "directory":
        (spool_dir / "stop").mkdir()
    elif kind == "link to nothing":
        (spool_dir / "stop").symlink_to(spool_dir.parent / "nowhere")
    else:
        (spool_dir.parent / "empty").touch()
        (spool_dir / "stop").symlink_to(spool_dir.parent / "empty")


def test_stop_ends_the_running_service(start, used_spool):
    process = start()
    spool.request_stop(str(used_spool), COMMITMENT_ID)
    assert process.wait(20) == 0
    assert not (used_spool / "stop").exists()


def test_an_empty_stop_found_at_the_start_is_deleted_and_stops_nothing(start, used_spool):
    spool.request_stop(str(used_spool), COMMITMENT_ID)
    process = start()
    assert not (used_spool / "stop").exists()
    request_id, _ = submit(str(used_spool), 2)
    wait_for_all(str(used_spool), [request_id])
    assert process.poll() is None


@pytest.mark.parametrize("kind", FOREIGN)
def test_another_entry_named_stop_ends_the_running_service_and_stays(start, used_spool, kind):
    process = start()
    make_foreign_stop(used_spool, kind)
    assert process.wait(20) == 0
    assert os.path.lexists(used_spool / "stop")


@pytest.mark.parametrize("kind", FOREIGN)
def test_does_not_start_with_another_entry_named_stop(tmp_path, used_spool, kind):
    make_foreign_stop(used_spool, kind)
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS))
    assert (got.returncode, got.stdout) == (1, "")
    assert (
        got.stderr == "dummy_service.py: error: %s/stop is not an empty regular file\n" % used_spool
    )
    assert os.path.lexists(used_spool / "stop")
    assert any(name.endswith(".req") for name in os.listdir(used_spool / "in"))


# A service that started here would delete both files, the empty "stop" and the invalid request.
def test_does_not_start_in_a_directory_with_entries_and_no_mark_file(tmp_path):
    (tmp_path / "spool" / "in").mkdir(mode=0o700, parents=True)
    (tmp_path / "spool").chmod(0o700)
    (tmp_path / "spool" / "stop").touch()
    (tmp_path / "spool" / "in" / ("cd" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    argv = [str(tmp_path / "spool"), "--tokens", str(TOKENS), "--end-when-empty"]
    got = run_script(tmp_path, *argv)
    assert (got.returncode, got.stdout) == (1, "")
    assert "%s is not a spool directory: it has no mark file" % (tmp_path / "spool") in got.stderr
    assert sorted(os.listdir(tmp_path / "spool")) == ["in", "stop"]
    assert os.listdir(tmp_path / "spool" / "in") == ["cd" * 32 + ".req"]


def test_a_front_end_takes_a_directory_that_the_service_used_first(tmp_path):
    (tmp_path / "spool").mkdir(mode=0o700)
    argv = [str(tmp_path / "spool"), "--tokens", str(TOKENS), "--end-when-empty"]
    assert run_script(tmp_path, *argv).returncode == 0
    assert sorted(os.listdir(tmp_path / "spool")) == ["in", "mark.txt", "out", "work"]
    assert (tmp_path / "spool" / "mark.txt").read_text().endswith("commitment %s\n" % ("0" * 64))
    request_id = submit(str(tmp_path / "spool"), 1)[0]
    assert (tmp_path / "spool" / "in" / (request_id + ".req")).is_file()


def test_deletes_nothing_in_a_directory_that_is_open_to_other_users(tmp_path, used_spool):
    (used_spool / "stop").touch()
    used_spool.chmod(0o755)
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS))
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.startswith("dummy_service.py: error: %s " % used_spool)
    assert "is not your own private directory" in got.stderr
    assert (used_spool / "stop").exists()


def test_end_when_empty_exits_once_no_request_is_waiting(tmp_path, used_spool):
    request_id, _ = submit(str(used_spool), 2)
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS), "--end-when-empty")
    assert got.returncode == 0
    assert len(spool.finished_requests(str(used_spool))) == 2
    argv = [str(used_spool), "--tokens", "8", "--end-when-empty"]
    got = run_script(tmp_path, *argv, "--first-take", "2")
    assert got.returncode == 2 and "--first-take cannot be used with --end-when-empty" in got.stderr


@pytest.mark.parametrize("path", ["/", "/var/spool", "{tmp}", "{tmp}/", "{tmp}/../elsewhere"])
def test_rejects_a_directory_that_is_not_below_the_temporary_directory(tmp_path, path):
    got = run_script(tmp_path, path.format(tmp=tmp_path), "--tokens", "8")
    assert got.returncode == 2
    assert "SPOOL_DIR must be a directory below %s, not " % tmp_path in got.stderr


# -OO removes the docstring from which the service takes its description.
def test_help_works_under_python_oo():
    argv = [sys.executable, "-OO", "-B", SCRIPT, "--help"]
    got = subprocess.run(argv, capture_output=True, text=True, env=CHILD_ENV)
    assert (got.returncode, got.stderr) == (0, "") and got.stdout.startswith("usage: ")


def test_rejects_a_directory_that_does_not_exist(tmp_path):
    got = run_script(tmp_path, str(tmp_path / "none"), "--tokens", "8")
    assert got.returncode == 2
    assert "SPOOL_DIR %s is not a directory" % (tmp_path / "none") in got.stderr


@pytest.mark.parametrize("option", ["--tokens", "--batch", "--first-take"])
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_rejects_numbers_that_are_not_positive(tmp_path, option, value):
    (tmp_path / "spool").mkdir()
    argv = [str(tmp_path / "spool"), "--tokens", "8", option, value]
    got = run_script(tmp_path, *(argv[:1] + argv[3:] if option == "--tokens" else argv))
    assert got.returncode == 2
    assert "is not a positive integer" in got.stderr


def test_a_spool_directory_of_any_commitment_is_served(tmp_path, used_spool):
    # The mark names a commitment, and the service knows none.
    assert (used_spool / "mark.txt").read_text().endswith("commitment %s\n" % COMMITMENT_ID)
    argv = [str(used_spool), "--tokens", str(TOKENS), "--end-when-empty"]
    assert run_script(tmp_path, *argv).returncode == 0
    assert os.listdir(used_spool / "in") in ([], [".tmp"])


def test_a_mark_file_of_another_form_is_refused_and_left_as_it_is(tmp_path, used_spool):
    # The service would delete the file stop, and make work/ and out/.
    (used_spool / "stop").touch()
    (used_spool / "mark.txt").chmod(0o600)
    (used_spool / "mark.txt").write_bytes(b"cp-folder 1\nkind out\ncommitment %s\n" % (b"0" * 64))
    before = sorted(str(path) for path in used_spool.rglob("*"))
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stdout) == (1, "")
    assert "its mark file (mark.txt) is not that of a spool directory" in got.stderr
    assert sorted(str(path) for path in used_spool.rglob("*")) == before


def test_a_request_that_has_a_response_is_deleted_and_not_answered_again(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(mode=0o700)
    request_id, _ = submit(str(spool_dir), 1)
    (spool_dir / "out").mkdir(mode=0o700)
    (spool_dir / "out" / (request_id + ".proof")).write_text("from an earlier run")
    got = run_script(tmp_path, str(spool_dir), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stderr) == (0, "")
    assert sorted(os.listdir(spool_dir / "out")) == [".tmp", request_id + ".proof"]
    assert (spool_dir / "out" / (request_id + ".proof")).read_text() == "from an earlier run"
    assert os.listdir(spool_dir / "in") == [".tmp"]
    line = dummy_service.REMOVED % (request_id + ".req", "its ID has an entry in out/")
    assert got.stdout.splitlines()[1:] == [line]


def test_a_line_is_printed_for_a_deleted_file(tmp_path, used_spool):
    (used_spool / "in" / ("cd" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stderr) == (0, "")
    line = dummy_service.REMOVED % ("cd" * 32 + ".req", "its name is not the hash of its bytes")
    assert got.stdout.splitlines()[1:] == [line]
    assert os.listdir(used_spool / "in") == [".tmp"]


def test_a_file_that_cannot_be_read_is_deleted(tmp_path, used_spool, monkeypatch, capsys):
    (name,) = (name for name in os.listdir(used_spool / "in") if name.endswith(".req"))
    monkeypatch.setattr(dummy_service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(dummy_service, "read_request", lambda *_: None)
    args = dummy_service.parse_args([str(used_spool), "--tokens", str(TOKENS)])
    with spool.open_private_dir(str(used_spool)) as top:
        assert dummy_service.take(args, dummy_service.Directories(top, str(used_spool))) == {}
    line = dummy_service.REMOVED % (name, "it could not be read")
    assert capsys.readouterr().out.splitlines() == [line]
    assert os.listdir(used_spool / "in") == [".tmp"]


@pytest.mark.parametrize("change", [None, 3])  # 3: the service prints that the request failed
def test_goes_on_when_nobody_reads_its_output_any_more(start, used_spool, change):
    process = start()
    process.stdout.close()
    process.stdout = None  # the fixture's communicate() then only waits
    (used_spool / "in" / ("cd" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    wait_for_all(str(used_spool), [submit(str(used_spool), 2, change=change)[0]])
    assert not (used_spool / "in" / ("cd" * 32 + ".req")).exists()
    spool.request_stop(str(used_spool), COMMITMENT_ID)
    assert process.wait(20) == 0


# As if the service were started on the spool directory of a commitment with longer requests.
def test_a_request_of_another_length_is_left_alone(tmp_path, used_spool):
    request = spool.make_request(range(2 * TOKENS), 2, 5)
    name = spool.submit(str(used_spool), request, COMMITMENT_ID) + ".req"
    before = (used_spool / "in" / name).read_bytes()
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stderr, len(got.stdout.splitlines())) == (0, "", 1)
    assert sorted(os.listdir(used_spool / "in")) == [".tmp", name]
    assert (used_spool / "in" / name).read_bytes() == before
    assert len(spool.finished_requests(str(used_spool))) == 1  # used_spool's own request


def test_does_not_start_below_a_directory_that_others_can_change(tmp_path, monkeypatch):
    (tmp_path / "open").mkdir()
    (tmp_path / "open").chmod(0o777)
    (tmp_path / "open" / "spool").mkdir(mode=0o700)
    (tmp_path / "open" / "spool" / "stop").touch()
    got = run_script(
        tmp_path, str(tmp_path / "open" / "spool"), "--tokens", "8", "--end-when-empty"
    )
    assert got.returncode == 1
    assert got.stderr.startswith(
        "dummy_service.py: error: the spool directory %s lies below %s, which another account may "
        "be able to change (its group or others can write in it, and it has no sticky bit)"
        % tuple(
            spool.printable(str(path)) for path in (tmp_path / "open" / "spool", tmp_path / "open")
        )
    )
    assert os.listdir(tmp_path / "open" / "spool") == ["stop"]
    # main() makes this check itself: it does not rely on the one in spool.open_private_dir().
    monkeypatch.setattr(dummy_service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(spool, "open_private_dir", lambda path: pytest.fail("it was opened"))
    assert dummy_service.main([str(tmp_path / "open" / "spool"), "--tokens", "8"]) == 1


@pytest.mark.parametrize("name", ["in", "in/.tmp", "out", "out/.tmp", "work"])
def test_does_not_start_if_a_subdirectory_is_a_link(tmp_path, name):
    (tmp_path / "spool").mkdir(mode=0o700)
    # Create the mark file: without it the service rejects the directory before it sees the link.
    with spool.open_private_dir(str(tmp_path / "spool")) as top:
        spool.check_mark(top, COMMITMENT_ID, create=True)
    (tmp_path / "spool" / name).parent.mkdir(mode=0o700, exist_ok=True)
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    (tmp_path / "elsewhere" / ("cd" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    (tmp_path / "spool" / name).symlink_to(tmp_path / "elsewhere")
    got = run_script(tmp_path, str(tmp_path / "spool"), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stdout) == (1, "")
    assert "%s/%s is a link or not a directory" % (tmp_path / "spool", name) in got.stderr
    assert os.listdir(tmp_path / "elsewhere") == ["cd" * 32 + ".req"]


@pytest.mark.parametrize("mode", [0o770, 0o750, 0o705])
def test_does_not_start_if_a_subdirectory_is_open_to_others(tmp_path, used_spool, mode):
    (used_spool / "in").chmod(mode)
    got = run_script(tmp_path, str(used_spool), "--tokens", str(TOKENS), "--end-when-empty")
    assert (got.returncode, got.stdout) == (1, "")
    assert "%s/in belongs to another user, or is open to others" % used_spool in got.stderr
    assert any(name.endswith(".req") for name in os.listdir(used_spool / "in"))


def test_a_link_put_in_the_place_of_a_subdirectory_later_is_not_followed(start, used_spool):
    elsewhere = used_spool.parent / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (elsewhere / ("cd" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    process = start()
    (used_spool / "in").rename(used_spool / "moved")
    (used_spool / "in").symlink_to(elsewhere)
    # The service still reads the directory that it opened, under its new name.
    (used_spool / "moved" / ("ef" * 32 + ".req")).write_bytes(NOT_A_REQUEST)
    deadline = time.monotonic() + 20
    while (used_spool / "moved" / ("ef" * 32 + ".req")).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not (used_spool / "moved" / ("ef" * 32 + ".req")).exists()
    assert os.listdir(elsewhere) == ["cd" * 32 + ".req"]
    assert process.poll() is None


def test_does_not_start_if_the_path_leads_elsewhere_when_it_is_opened(
    tmp_path, monkeypatch, capsys
):
    for above in ("above", "elsewhere"):  # not group-writable, even under a umask of 002
        (tmp_path / above).mkdir(mode=0o700)
        (tmp_path / above / "spool").mkdir(mode=0o700)
    (tmp_path / "elsewhere" / "spool" / "stop").touch()
    parse_args = dummy_service.parse_args

    def parse_then_swap(argv):
        args = parse_args(argv)
        (tmp_path / "above").rename(tmp_path / "moved")
        (tmp_path / "above").symlink_to(tmp_path / "elsewhere")
        return args

    monkeypatch.setattr(dummy_service, "parse_args", parse_then_swap)
    monkeypatch.setattr(dummy_service.tempfile, "gettempdir", lambda: str(tmp_path))
    # If the service did start, --end-when-empty would make main() return 0, not serve forever.
    argv = [str(tmp_path / "above" / "spool"), "--tokens", "8", "--end-when-empty"]
    assert dummy_service.main(argv) == 1
    assert "spool no longer leads to the same directory" in capsys.readouterr().err
    assert os.listdir(tmp_path / "elsewhere" / "spool") == ["stop"]


def test_rejects_the_root_directory_as_the_temporary_directory(tmp_path, monkeypatch, capsys):
    (tmp_path / "spool").mkdir(mode=0o700)
    monkeypatch.setattr(dummy_service.tempfile, "gettempdir", lambda: "/")
    with pytest.raises(SystemExit) as stopped:
        dummy_service.parse_args([str(tmp_path / "spool"), "--tokens", "8"])
    assert stopped.value.code == 2
    assert "SPOOL_DIR must be a directory below /, not " in capsys.readouterr().err


def test_a_symbolic_link_in_in_is_left_alone(service, tmp_path):
    (tmp_path / "target").write_bytes(NOT_A_REQUEST)
    good = [submit(service, token)[0] for token in (1, 2)]
    wait_for_all(service, good)
    os.symlink(tmp_path / "target", os.path.join(service, "in", "cd" * 32 + ".req"))
    last, _ = submit(service, 3)
    wait_for_all(service, [last])
    assert os.path.islink(os.path.join(service, "in", "cd" * 32 + ".req"))
    assert (tmp_path / "target").read_bytes() == NOT_A_REQUEST


def test_exits_when_the_process_that_started_it_is_gone(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(mode=0o700)
    # The service inherits the pipe's write end: its exit shows as end of file.
    read_end, write_end = os.pipe()
    starter = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-B', %r, %r, *%r], pass_fds=[%d], "
        "stdout=subprocess.PIPE); assert p.stdout.readline(); time.sleep(0.5)"
        % (SCRIPT, str(spool_dir), ("--tokens", "8"), write_end)
    )
    env = {**CHILD_ENV, "TMPDIR": str(tmp_path)}
    subprocess.run([sys.executable, "-I", "-c", starter], env=env, pass_fds=[write_end], timeout=20, check=True)
    os.close(write_end)
    readable, _, _ = select.select([read_end], [], [], 20)
    assert readable and os.read(read_end, 1) == b""
    os.close(read_end)
