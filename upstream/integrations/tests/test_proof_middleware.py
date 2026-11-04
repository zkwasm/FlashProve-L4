# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of proof_middleware.py. They need only pytest: no vLLM, no GPU, no build.

vLLM is replaced by an ASGI application that replays the response bodies in recorded_vllm/, which
vLLM 0.29.0's routes produced around a fake engine, with this middleware loaded and the policy
header set. The proving service (cp_prove) is replaced by Setup.respond(), which writes a response's
files with a dummy proof. So the tests show nothing about a real engine or a real proof.

Words: a job is the middleware's file about one served reply; a client asks for it by its proof ID.
A response is what the proving service writes into out/ for one request: the files y (the next
token), proof and tag, or failed and tag. The tag is an HMAC, under a key in the job file, that ties
the response to the request ID. A job's record (".checked") is written when a proof's tag has
matched, so that later polls do not read the proof again. A mark file (mark.txt) states what its
directory is for."""

import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import dataclasses
import errno
import fcntl
import gc
import io
import itertools
import json
import logging
import os
import pickle
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time

import pytest

import proof_middleware
from proof_middleware import ConfigError, ProofMiddleware, load_config, spool

RECORDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_vllm")

COMMITMENT_ID = "ab" * 32
OTHER_UID = 12346 if os.environ.get("SUDO_UID") == "12345" else 12345  # an account nothing trusts


def as_root(test):
    """Mark a test that only root can run, because it gives files or directories to other accounts:
    skipped for anybody else, and selected or left out as a group by the marker root_only."""
    skip = pytest.mark.skipif(os.geteuid() != 0, reason="only root can give a file away")
    return pytest.mark.root_only(skip(test))


PROMPT, REPLY = (
    [1, 2, 3, 4, 5],
    [
        11,
        12,
        13,
        7,
    ],
)  # the token IDs in completion.json and completion_stream.txt
ASK = {"x-proof-policy": "exact"}
FIELDS = {
    "model": "dummy",
    "prompt": PROMPT,
    "max_tokens": 8,
    "temperature": 0,
    "return_token_ids": True,
}
JOBS, PUBLISHED = "/plugins/proof/jobs/", "/plugins/proof/published/"
# What a command adds where permission is denied, or a directory is another account's.
DENIED = (
    " (if you are not the account that runs the vLLM server, run this as that account: sudo -u"
    " ACCOUNT ...)"
)
# What the hold command adds to each error that it prints itself; argparse's usage errors lack it.
NOT_HELD = (
    ". NOTHING WAS PUT ON HOLD. Mend that and run this command again. If you can't, stop every vLLM"
    " server that serves this policy, and start none before this command has put the policy on hold"
    " (what a stopped server remembered of a rejected batch is lost: before that hold is lifted,"
    " see integrations/LIMITS.md#the-middlewares-check)"
)
# policy.txt's lines: exact match (C, theta, cap, sT), requests of 2^lt = 16 tokens, V token IDs.
PARAMETERS = {
    "lt": 4,
    "V": 200,
    "C": 256,
    "theta": 256,
    "cap": 0,
    "s": 15,
    "sh": 8,
    "sT": 4,
    "F": 24,
}


class FakeVllm:
    """An ASGI application that answers every HTTP call with one recorded response and records the
    request bodies."""

    def __init__(self, name="completion.json"):
        with open(os.path.join(RECORDED, name), "rb") as f:
            body = f.read()
        self.chunks = (
            [body]
            if name.endswith(".json")
            else [event + b"\n\n" for event in body.split(b"\n\n") if event] + [b""]
        )
        self.bodies, self.status = [], 200

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        self.bodies.append((await receive())["body"])
        await send(
            {
                "type": "http.response.start",
                "status": self.status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        for number, chunk in enumerate(self.chunks, 1):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": number < len(self.chunks),
                }
            )


def http_scope(method, path, headers=(), root_path=""):
    pairs = headers.items() if isinstance(headers, dict) else headers
    return {
        "type": "http",
        "method": method,
        "path": root_path + path,
        "root_path": root_path,
        "headers": [(name.encode(), value.encode()) for name, value in pairs],
    }


def in_chunks(*chunks):
    """Return a web server's messages for a chunked request body."""
    return [
        {"type": "http.request", "body": chunk, "more_body": number < len(chunks)}
        for number, chunk in enumerate(chunks, 1)
    ]


def call(app, method, path, headers=(), body=b"", root_path="", send=None):
    """Call an ASGI application once. Return the status (None if nothing was sent), the headers and
    the body.

    body: bytes, or the messages that receive() returns. send: replaces the function that collects
    the HTTP response, so that the result is (None, {}, b"")."""
    scope = http_scope(method, path, headers, root_path)
    requests = list(body) if isinstance(body, list) else in_chunks(body)
    status, got, chunks = [], {}, []

    async def receive():
        return requests.pop(0) if requests else await asyncio.Event().wait()

    async def collect(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
            got.update((name.decode(), value.decode()) for name, value in message["headers"])
        else:
            chunks.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send or collect))
    return status[0] if status else None, got, b"".join(chunks)


async def stays_connected():
    """A receive() for a silent, connected client."""
    await asyncio.Event().wait()


def error_of(result, status):
    """Return the message of an error response, which must have this status and the form of vLLM's
    errors."""
    assert result[0] == status, result
    error = json.loads(result[2])["error"]
    assert sorted(error) == ["code", "message", "param", "type"] and error["code"] == status
    return error["message"]


def count_reads(monkeypatch, path):
    """Return a list that receives the length of each later os.read() from this file (a download's
    reads are not counted)."""
    inode, read, lengths = os.stat(path).st_ino, os.read, []

    def counting_read(fd, length):
        data = read(fd, length)
        if os.fstat(fd).st_ino == inode:
            lengths.append(len(data))
        return data

    monkeypatch.setattr(os, "read", counting_read)
    return lengths


RUN_THE_CHECK, PLAN_ID = proof_middleware.run_check, proof_middleware.plan_id


def RUN_CHECK(policy, job, y, path, folder):
    """run_check() with a jobs directory given by its path."""
    jobs = os.open(folder, os.O_RDONLY)
    try:
        return RUN_THE_CHECK(policy, job, y, path, jobs)
    finally:
        os.close(jobs)


WAIT_S_OF_THE_MODULE = proof_middleware.WAIT_S


@pytest.fixture(autouse=True)
def every_proof_passes(monkeypatch, tmp_path_factory):
    """The tests that are not about the check of a batch get a check that passes, with no verifier."""
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.PASSED)
    monkeypatch.setattr(proof_middleware, "plan_id", lambda policy: policy.commitment_id)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    monkeypatch.setattr(proof_middleware, "RETRY_S", 0)
    monkeypatch.setattr(proof_middleware, "MEMORY_DIR", str(tmp_path_factory.mktemp("memory")))


class Setup:
    """A temporary directory with a config file, and the middleware around a fake vLLM."""

    def __init__(self, root):
        self.root = str(root)
        os.chmod(self.root, 0o700)
        self.entry = {
            "served_model": "dummy",
            "commitment_id": COMMITMENT_ID,
            "parameter_file": self.path("policy.txt"),
            "spool_dir": self.path("spool"),
            "public_key_dir": self.root + "-key",
            "verifier": os.path.join(self.root + "-key", "cp_verify"),  # see every_proof_passes
        }
        self.config = {"jobs_dir": self.path("jobs"), "policies": {"exact": self.entry}}
        os.mkdir(self.root + "-key", 0o700)
        with open(os.path.join(self.root + "-key", "key.bin"), "wb") as f:
            f.write(b"dummy key.bin")
        with open(self.entry["verifier"], "w") as f:  # as far as plan_id() asks
            f.write("#!/bin/sh\necho 'commitment ID: %s'\nexit 4\n" % COMMITMENT_ID)
        os.chmod(self.entry["verifier"], 0o755)
        self.ask_headers = dict(ASK)
        self.write_parameters()

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def write_parameters(self, name="policy.txt", **changes):
        with open(self.path(name), "w") as f:
            f.writelines("%s %d\n" % line for line in {**PARAMETERS, **changes}.items())

    def write_config(self):
        with open(self.path("config.json"), "w") as f:
            json.dump(self.config, f)
        return self.path("config.json")

    def app(self, recorded="completion.json"):
        self.vllm = FakeVllm(recorded)
        app = ProofMiddleware(self.vllm, config=self.write_config())
        assert app.fault is None, app.fault
        return app

    def ask(self, app, route="/v1/completions", **changes):
        """Ask for a reply with a proof. Return the proof ID."""
        status, headers, _ = call(
            app, "POST", route, self.ask_headers, json.dumps({**FIELDS, **changes}).encode()
        )
        assert status == 200
        return headers["x-proof-id"]

    def state(self, app, proof_id):
        status, _, body = call(app, "GET", JOBS + proof_id)
        assert status == 200, body
        return json.loads(body)

    def job_path(self, proof_id, suffix=".json"):
        """Return the path of a job file or (".checked") its record."""
        return self.path("jobs", proof_middleware.job_file(proof_id, suffix))

    def proved(self, app, proof=b"dummy proof", poll=True):
        """Ask for a proof and respond. With poll, also ask for the state, which checks the tag and
        writes the record. Return the proof ID and the proof file's path."""
        proof_id = self.ask(app)
        (name,) = self.respond(proof)
        if poll:
            assert self.state(app, proof_id)["state"] == "proved"
        return proof_id, self.path("spool", "out", name + ".proof")

    def jobs(self):
        """Return the jobs directory's sorted listing without mark.txt, which must exist."""
        names = sorted(os.listdir(self.path("jobs")))
        names.remove("mark.txt")
        return names

    def files(self):
        """Return jobs() and out/'s sorted listing."""
        return self.jobs(), sorted(os.listdir(self.path("spool", "out")))

    def requests(self):
        """Return {request ID: bytes} for the waiting requests."""
        found = {}
        for name in os.listdir(self.path("spool", "in")):
            if name.endswith(".req"):
                with open(self.path("spool", "in", name), "rb") as f:
                    found[name.removesuffix(".req")] = f.read()
        return found

    def respond(self, proof=b"dummy proof", failed=False):
        """Respond to every waiting request (failed: without a proof); return the request IDs."""
        os.makedirs(self.path("spool", "out"), mode=0o700, exist_ok=True)
        answered = self.requests()
        for name, data in answered.items():
            key, y = spool.tag_key(data), bytes(4)
            files = (
                {"tag": spool.failed_tag(key, name), "failed": b""}
                if failed
                else {"y": y, "tag": spool.response_tag(key, name, y, proof), "proof": proof}
            )
            for suffix, content in files.items():
                with open(self.path("spool", "out", "%s.%s" % (name, suffix)), "wb") as f:
                    f.write(content)
            self.set_one_time(name)
            os.remove(self.path("spool", "in", name + ".req"))
        return list(answered)

    def set_one_time(self, name):
        """Give a response's files one modification time, as the proving service does: here one in
        2001, which is old for `clean`."""
        for file in os.listdir(self.path("spool", "out")):
            if file.startswith(name):
                os.utime(
                    self.path("spool", "out", file), ns=(10**18, 10**18), follow_symlinks=False
                )

    def public_key_dir(self):
        self.entry["public_key_dir"] = self.path("public")
        os.makedirs(self.path("public", "leaf"))
        for name in ("key.bin", "leaf/0.bin"):
            with open(self.path("public", name), "wb") as f:
                f.write(b"dummy " + name.encode())


@pytest.fixture(autouse=True)
def umask_022():
    """The middleware rejects files that their group can write, so the tests' files must not depend
    on the caller's umask."""
    before = os.umask(0o022)
    yield
    os.umask(before)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


# ---- Serving and fetching ----


@pytest.mark.parametrize(
    "route, recorded, changes",
    [
        ("/v1/completions", "completion.json", {}),
        ("/v1/completions", "completion_stream.txt", {"stream": True}),
        (
            "/v1/chat/completions",
            "chat.json",
            {"prompt": "", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "/v1/chat/completions",
            "chat_stream.txt",
            {"prompt": "", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        ),
    ],
)
def test_served_reply_is_submitted_and_its_proof_can_be_fetched(setup, route, recorded, changes):
    app = setup.app(recorded)
    proof_id = setup.ask(app, route, **changes)
    state = setup.state(app, proof_id)
    assert state["state"] == "waiting" and state["policy"] == "exact"
    (request,) = setup.requests().values()
    prompt_tokens = state["request"]["prompt_tokens"]
    assert state["request"] == {
        "total_tokens": 16,
        "prompt_tokens": prompt_tokens,
        "reply_tokens": 4,
        "padding_token": 0,
    }
    assert len(request) == 4 * (16 + 2) + spool.NONCE_BYTES
    if route == "/v1/completions":
        assert request[: -spool.NONCE_BYTES] == spool.whole_reply(PROMPT, REPLY, 16)
    setup.respond(b"dummy proof")
    assert setup.state(app, proof_id)["files"] == {"y.bin": 4, "proof.bin": 11}
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[2] == b"dummy proof"
    assert call(app, "GET", JOBS + proof_id + "/y.bin")[2] == bytes(4)


def test_reply_is_passed_on_unchanged(setup):
    app = setup.app()
    with_proof = call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())
    without = call(app, "POST", "/v1/completions", {}, json.dumps(FIELDS).encode())
    assert with_proof[2] == without[2] == b"".join(setup.vllm.chunks)
    assert "x-proof-id" in with_proof[1] and "x-proof-id" not in without[1]
    assert setup.vllm.bodies == [json.dumps(FIELDS).encode()] * 2
    assert len(setup.requests()) == 1


def test_no_reason_is_given_when_the_prover_issues_no_proof(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    setup.respond(failed=True)
    assert setup.state(app, proof_id) | {"request": None} == {
        "state": "no_proof",
        "policy": "exact",
        "request": None,
        "detail": proof_middleware.DECLINED,
    }
    error_of(call(app, "GET", JOBS + proof_id + "/proof.bin"), 404)


def test_delete_removes_the_job_and_the_response(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    assert "nothing was deleted" in error_of(call(app, "DELETE", JOBS + proof_id), 409)
    setup.respond()
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert setup.jobs() == os.listdir(setup.path("spool", "out")) == []
    error_of(call(app, "GET", JOBS + proof_id), 404)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("poll", [False, True])
def test_job_waits_while_its_request_is_in_work(setup, failed, poll):
    app = setup.app()
    proof_id = setup.ask(app)
    (request_id,) = setup.respond(failed=failed)
    final = "no_proof" if failed else "proved"
    assert not poll or setup.state(app, proof_id)["state"] == final
    before = setup.files()
    os.mkdir(setup.path("spool", "work"), 0o700)
    open(setup.path("spool", "work", request_id + ".req"), "w").close()
    assert sorted(setup.state(app, proof_id)) == ["policy", "request", "state"]
    assert setup.state(app, proof_id)["state"] == "waiting"
    for file in ("y.bin", "proof.bin"):
        error_of(call(app, "GET", JOBS + proof_id + "/" + file), 404)
    error_of(call(app, "DELETE", JOBS + proof_id), 409)
    assert setup.files() == before
    os.remove(setup.path("spool", "work", request_id + ".req"))
    assert setup.state(app, proof_id)["state"] == final


# Put the policy "exact" on hold and return the path of the entry that does it. A policy is on hold
# while its spool directory has an entry "hold", of any kind: the middleware then submits nothing
# for it, and neither serves nor deletes its stored responses.
def make_hold(setup, kind):
    path = setup.path("spool", "hold")
    if kind == "file":
        open(path, "w").close()
    elif kind == "directory":
        os.mkdir(path)
    elif kind == "pipe":
        os.mkfifo(path)
    else:
        os.symlink("nowhere", path)
    return path


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
@pytest.mark.parametrize("poll", [False, True])
def test_policy_on_hold_submits_nothing_and_serves_no_response(setup, caplog, kind, poll):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=poll)
    hold, before = make_hold(setup, kind), setup.files()
    for method, path in (("GET", ""), ("GET", "/y.bin"), ("GET", "/proof.bin"), ("DELETE", "")):
        assert "on hold" in error_of(call(app, method, JOBS + proof_id + path), 503)
    # Four calls, one logged line: a hold is logged at most once an hour.
    assert len(caplog.messages) == 1 and hold in caplog.messages[0]
    state = setup.state(app, setup.ask(app))
    assert state["state"] == "not_provable" and "on hold" in state["detail"]
    assert setup.requests() == {} and setup.files()[1] == before[1]
    os.rmdir(hold) if kind == "directory" else os.remove(hold)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"dummy proof")


def test_hold_and_lift_hold_commands(setup, capsys):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    config, path = setup.path("config.json"), setup.path("spool", "hold")
    assert proof_middleware.main(["lift-hold", config, "exact"]) == 1
    assert capsys.readouterr().err == "error: the policy exact is not on hold\n"
    # The second run finds the hold in place.
    for expected in ("Created %s: the policy exact is on hold.\n", "%s exists).\n"):
        assert proof_middleware.main(["hold", config, "exact"]) == 0
        assert capsys.readouterr().out.endswith(expected % path)
        assert os.stat(path).st_size == 0 and os.stat(path).st_mode & 0o777 == 0o600
    error_of(call(app, "GET", JOBS + proof_id), 503)
    before = setup.files()
    assert proof_middleware.main(["lift-hold", config, "exact"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(
        "Removing %s: the policy exact goes off hold. Its stored responses (now 1) are on offer"
        " again, also those of a cp_prove that exited badly.\nThat is safe only if the cause of"
        " the hold is known, is gone, and was harmless (a mistyped path, a full disk)." % path
    )
    assert "Otherwise run the hold command now and make a new commitment" in printed
    assert not os.path.lexists(path) and setup.files() == before
    assert setup.state(app, proof_id)["state"] == "proved"


def test_hold_command_syncs_the_spool_directory(setup, monkeypatch):
    setup.app()
    synced, path = [], setup.path("spool", "hold")
    monkeypatch.setattr(
        os, "fsync", lambda fd: synced.append((os.fstat(fd).st_ino, os.path.exists(path)))
    )
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    assert synced == [(os.stat(setup.path("spool")).st_ino, True)]  # after the file is made


@pytest.mark.parametrize("call, which", [("close", 1), ("fsync", 1), ("close", 2)])
def test_hold_command_says_that_the_hold_is_there_if_only_the_sync_fails(
    setup, monkeypatch, capsys, call, which
):
    real, path, calls = getattr(os, call), setup.path("spool", "hold"), []

    # `which`: 1 for the file's close or the sync, 2 for the directory's close.
    def fails_once_the_file_is_there(fd):
        real(fd)
        calls.append(os.path.exists(path))
        if sum(calls) == which:
            monkeypatch.undo()
            raise OSError(5, "Input/output error")

    setup.app()
    monkeypatch.setattr(os, call, fails_once_the_file_is_there)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    printed = capsys.readouterr()
    assert printed.err == "" and printed.out == (
        "Created %s: the policy exact is on hold.\nwarning: closing or syncing it failed"
        " (Input/output error), so a loss of power may undo it\n" % path
    )
    assert os.path.exists(path)


def test_hold_command_gives_no_warning_for_a_hold_that_it_did_not_make(setup, monkeypatch, capsys):
    real, config = os.close, setup.app().config
    path = make_hold(setup, "file")

    def fails_at_the_spool_directory(fd):
        directory = os.fstat(fd).st_ino == os.stat(setup.path("spool")).st_ino
        real(fd)
        if directory:
            monkeypatch.undo()
            raise OSError(5, "Input/output error")

    # load_config() closes the spool directory too: it is replaced so that the fault meets hold().
    monkeypatch.setattr(proof_middleware, "load_config", lambda *arguments, **more: config)
    monkeypatch.setattr(os, "close", fails_at_the_spool_directory)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    printed = capsys.readouterr()
    assert printed.out == "The policy exact is already on hold (%s exists).\n" % path
    assert printed.err == ""


def test_hold_command_says_so_if_it_cannot_make_the_file(setup, monkeypatch, capsys):
    real = os.open

    def no_hold(path, *arguments, **more):
        if path == "hold":
            raise PermissionError(1, "Operation not permitted", path)
        return real(path, *arguments, **more)

    setup.app()
    monkeypatch.setattr(os, "open", no_hold)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: [Errno 1] Operation not permitted: '%s'%s\n"
        % (setup.path("spool", "hold"), NOT_HELD)
    )
    assert not os.path.lexists(setup.path("spool", "hold"))


def test_hold_command_speaks_on_standard_error_if_standard_output_cannot_be_written(setup, capsys):
    class FullDisk:
        def write(self, text):
            raise OSError(28, "No space left on device")

    # On standard error say() prints ASCII only, so the path's non-ASCII letter is escaped.
    setup.entry["spool_dir"] = setup.path("sp\u00e9ol")
    setup.app()
    with contextlib.redirect_stdout(FullDisk()):
        assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    assert capsys.readouterr().err == (
        "Created %s: the policy exact is on hold.\n" % setup.path("sp\\xe9ol", "hold")
    )


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), ValueError("dummy")])
def test_hold_command_does_not_deny_the_hold_if_it_cannot_print(setup, monkeypatch, capsys, error):
    def no_print(*arguments, **more):
        raise error

    setup.app()
    monkeypatch.setattr("builtins.print", no_print)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    assert os.path.exists(setup.path("spool", "hold"))


def test_clean_exits_with_1_under_a_hold_even_if_nothing_is_old(setup, capsys):
    setup.ask(setup.app())
    make_hold(setup, "file")
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 1
    assert capsys.readouterr().err.startswith("error: nothing was deleted: the policy exact is on")


def test_hold_command_marks_an_empty_spool_directory_first(setup):
    # Only an empty directory gets a mark file: with a hold written first it would never get one,
    # and init would reject it.
    os.mkdir(setup.path("spool"), 0o700)
    assert proof_middleware.main(["hold", setup.write_config(), "exact"]) == 0
    assert sorted(os.listdir(setup.path("spool"))) == ["hold", "mark.txt"]
    assert proof_middleware.main(["init", setup.path("config.json")]) == 0


@pytest.mark.parametrize("kind", ["directory", "symlink", "file with content", "pipe"])
def test_lift_hold_removes_only_an_empty_regular_file(setup, capsys, kind):
    setup.app()
    path = make_hold(setup, "file" if kind == "file with content" else kind)
    if kind == "file with content":
        with open(path, "w") as f:
            f.write("x")
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and "is not an empty regular file" in printed.err
    assert os.path.lexists(path)


def fail(*arguments, **more):
    raise OSError("dummy")


@pytest.mark.parametrize(
    "command, end",
    [
        ("hold", NOT_HELD + "\n"),
        ("lift-hold", ". Nothing was removed\n"),
    ],
)
def test_hold_commands_reject_an_unknown_policy_and_a_missing_directory(
    setup, capsys, monkeypatch, command, end
):
    before = sorted(os.listdir(setup.root) + ["config.json"])
    assert proof_middleware.main([command, setup.write_config(), "exact"]) == 2
    assert capsys.readouterr().err.endswith("spool directory does not exist: run init" + end)
    assert sorted(os.listdir(setup.root)) == before
    setup.app()
    before = snapshot(setup.path("spool"))
    assert proof_middleware.main([command, setup.path("config.json"), "spool"]) == 2
    assert capsys.readouterr().err.endswith("has no policy 'spool' (it has: exact)" + end)
    assert proof_middleware.main([command, setup.path("missing.json"), "exact"]) == 2
    assert capsys.readouterr().err.endswith("No such file or directory" + end)
    monkeypatch.setattr(spool, "check_mark", fail)
    assert proof_middleware.main([command, setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err.endswith("dummy" + end)
    assert snapshot(setup.path("spool")) == before


@pytest.mark.parametrize("command", [proof_middleware.hold, proof_middleware.lift_hold])
def test_hold_commands_check_the_mark_file_themselves(setup, command):
    # Called without main(), whose load_config() would reject the directory first.
    policy = setup.app().config.policies["exact"]
    path = make_hold(setup, "file")
    write_mark_of_another_commitment(setup)
    before = snapshot(setup.path("spool"))
    with pytest.raises(spool.UnsafePath, match="is not a spool directory"):
        command(policy)
    assert snapshot(setup.path("spool")) == before and os.path.exists(path)


def test_hold_commands_act_on_the_named_policy(setup):
    setup.config["policies"]["other"] = {**setup.entry, "spool_dir": setup.path("other")}
    setup.app()
    config = setup.path("config.json")
    for name, directory in (("other", "other"), ("exact", "spool")):
        assert proof_middleware.main(["hold", config, name]) == 0
        assert os.path.exists(setup.path(directory, "hold"))
    assert proof_middleware.main(["lift-hold", config, "other"]) == 0
    assert os.path.exists(setup.path("spool", "hold"))
    assert not os.path.exists(setup.path("other", "hold"))


def test_lift_hold_keeps_the_hold_if_the_responses_cannot_be_counted(setup, monkeypatch, capsys):
    setup.app()
    path = make_hold(setup, "file")
    monkeypatch.setattr(proof_middleware, "waiting_and_stored", fail)
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == "error: dummy. Nothing was removed\n"
    assert os.path.exists(path)


def test_lift_hold_does_not_deny_the_removal_if_only_the_closing_fails(setup, monkeypatch, capsys):
    setup.app()
    real, path = os.close, make_hold(setup, "file")

    def fails_once_the_file_is_gone(fd):
        real(fd)
        if not os.path.lexists(path):
            monkeypatch.undo()
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "close", fails_once_the_file_is_gone)
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 0
    printed = capsys.readouterr()
    assert printed.out.startswith("Removing ") and printed.err == "" and not os.path.lexists(path)


@pytest.mark.parametrize(
    "number, words", [(1, "Operation not permitted"), (30, "Read-only file system")]
)
def test_lift_hold_says_so_if_the_removal_is_refused(setup, monkeypatch, capsys, number, words):
    def no_unlink(name, **arguments):
        raise OSError(number, words, name)

    setup.app()
    path = make_hold(setup, "file")
    monkeypatch.setattr(os, "unlink", no_unlink)
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    printed = capsys.readouterr()
    assert printed.out.startswith("Removing ") and os.path.exists(path)
    assert printed.err == "error: [Errno %d] %s: '%s'. Nothing was removed\n" % (
        number,
        words,
        path,
    )


def test_lift_hold_keeps_the_hold_if_its_words_reach_nobody(setup, capsys):
    class FullDisk:
        def write(self, text):
            return len(text)

        def flush(self):
            raise OSError(28, "No space left on device")

    setup.app()
    path = make_hold(setup, "file")
    with contextlib.redirect_stdout(FullDisk()):
        assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: lift-hold cannot write to its standard output ([Errno 28] No space left on device)."
        " Nothing was removed\n"
    )
    assert os.path.exists(path)


@pytest.mark.parametrize("command, code", [("hold", 0), ("lift-hold", 2)])
def test_hold_commands_end_as_told_if_neither_stream_can_be_written(setup, command, code):
    class FullDisk:
        def write(self, text):
            raise OSError(28, "No space left on device")

    setup.app()
    path = make_hold(setup, "file")
    with contextlib.redirect_stdout(FullDisk()), contextlib.redirect_stderr(FullDisk()):
        assert proof_middleware.main([command, setup.path("config.json"), "exact"]) == code
        assert sys.stdout is None and sys.stderr is None  # or Python would exit with 120
    assert os.path.exists(path)


def test_lift_hold_says_what_to_do_if_the_encoding_cannot_show_the_path(setup, capsys):
    setup.entry["spool_dir"] = setup.path("sp\u00e9ol")
    setup.app()
    open(setup.path("sp\u00e9ol", "hold"), "w").close()
    with contextlib.redirect_stdout(io.TextIOWrapper(io.BytesIO(), encoding="ascii")):
        assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: lift-hold cannot write to its standard output (its encoding cannot show the path:"
        " set PYTHONIOENCODING=utf-8:backslashreplace). Nothing was removed\n"
    )
    assert os.path.exists(setup.path("sp\u00e9ol", "hold"))


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full")
@pytest.mark.parametrize("command, code", [("hold", 0), ("lift-hold", 2)])
def test_hold_commands_as_programs_with_both_streams_on_a_full_device(setup, command, code):
    # Run as a process, where Python's own exit code 120 for unwritten output would show.
    setup.app()
    path = make_hold(setup, "file")
    with open("/dev/full", "w") as full:
        done = subprocess.run(
            [sys.executable, "-B", proof_middleware.__file__, command, setup.path("config.json")]
            + ["exact"],
            stdout=full,
            stderr=full,
        )
    assert done.returncode == code and os.path.exists(path)


def test_lift_hold_keeps_the_hold_if_standard_output_is_closed(setup, capsys):
    setup.app()
    path = make_hold(setup, "file")
    with contextlib.redirect_stdout(None):
        assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: lift-hold cannot write to its standard output (it is closed). Nothing was removed\n"
    )
    assert os.path.exists(path)


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"), ValueError("dummy")])
def test_lift_hold_keeps_the_hold_if_it_cannot_say_what_it_does(setup, monkeypatch, capsys, error):
    real = print

    def no_standard_output(*arguments, **more):
        if "file" not in more:
            raise error
        real(*arguments, **more)

    setup.app()
    path = make_hold(setup, "file")
    monkeypatch.setattr("builtins.print", no_standard_output)
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err.endswith(". Nothing was removed\n") and os.path.exists(path)


def test_hold_that_cannot_be_looked_at_counts_as_a_hold(setup, monkeypatch, capsys):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    for directory in ("jobs", "spool/out"):
        for name in os.listdir(setup.path(directory)):
            os.utime(setup.path(directory, name), (0, 0))
    before, real = setup.files(), os.stat

    def no_look_at_hold(path, **arguments):
        if path == "hold":
            raise OSError(5, "Input/output error")
        return real(path, **arguments)

    monkeypatch.setattr(os, "stat", no_look_at_hold)
    # Nothing is served, deleted or submitted, but the statuses are a fault's (500), not a hold's
    # (503), and the new reply's job is not stored (404).
    for path in ("", "/y.bin", "/proof.bin"):
        error_of(call(app, "GET", JOBS + proof_id + path), 500)
    error_of(call(app, "DELETE", JOBS + proof_id), 500)
    error_of(call(app, "GET", JOBS + setup.ask(app)), 404)
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 1
    assert setup.requests() == {} and setup.files() == before


def test_server_that_starts_under_a_hold_is_on_hold(setup):
    proof_id, _ = setup.proved(setup.app())
    hold, before = make_hold(setup, "file"), setup.files()
    app = setup.app()
    for path in ("", "/proof.bin"):
        assert "on hold" in error_of(call(app, "GET", JOBS + proof_id + path), 503)
    assert setup.state(app, setup.ask(app))["state"] == "not_provable"
    assert setup.requests() == {} and setup.files()[1] == before[1] and os.path.exists(hold)


def test_hold_stops_one_policy_only_and_is_logged_at_a_submit(setup, caplog):
    setup.config["policies"]["other"] = {**setup.entry, "spool_dir": setup.path("other")}
    app = setup.app()
    make_hold(setup, "file")
    setup.ask(app)
    assert len(caplog.messages) == 1 and "exact is on hold" in caplog.messages[0]
    setup.ask_headers["x-proof-policy"] = "other"
    assert setup.state(app, setup.ask(app))["state"] == "waiting"
    os.remove(setup.path("spool", "hold"))
    open(setup.path("other", "hold"), "w").close()
    setup.ask_headers["x-proof-policy"] = "exact"
    proof_id, _ = setup.proved(app)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] == 200


def test_check_init_clean_and_a_start_leave_a_hold_and_delete_nothing_under_it(setup, capsys):
    setup.config["policies"]["other"] = {**setup.entry, "spool_dir": setup.path("other")}
    app = setup.app()
    setup.proved(app)
    # The hold is on "other". The job and the response, made old enough for `clean`, belong to
    # "exact": `clean` deletes nothing of any policy.
    open(setup.path("other", "hold"), "w").close()
    for directory in ("jobs", "spool/out", "other"):
        for name in os.listdir(setup.path(directory)):
            os.utime(setup.path(directory, name), (0, 0))
    before = [snapshot(setup.path(name)) for name in ("jobs", "spool/out", "other")]
    on_hold = (
        "the policy other is on hold (%s exists). The lift-hold command removes it, which puts"
        " every stored response back on offer, faulty ones too, each behind the check of its batch,"
        " which a file that has lost a mask can pass: read integrations/vllm/REFERENCE.md first.\n"
    ) % setup.path("other", "hold")
    for command, code in (("check", 1), ("init", 0)):
        assert proof_middleware.main([command, setup.path("config.json")]) == code
        printed = capsys.readouterr()
        assert "policy other (ON HOLD: %s exists): " % setup.path("other", "hold") in printed.out
        assert printed.err == ("error: " + on_hold) * code
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 1
    assert capsys.readouterr().err == "error: nothing was deleted: " + on_hold
    setup.app()
    assert [snapshot(setup.path(name)) for name in ("jobs", "spool/out", "other")] == before


def test_job_of_a_streamed_reply_is_stored_before_its_last_event_is_sent(setup):
    app, stored = setup.app("completion_stream.txt"), []

    async def send(message):
        if message.get("body", b"").startswith(b"data: [DONE]"):
            stored.append(len(setup.jobs()))

    call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode(), send=send)
    assert stored == [1]


def test_no_response_of_the_proof_routes_may_be_cached(setup):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    for path in ("/plugins/proof/policies", JOBS + proof_id, JOBS + proof_id + "/proof.bin", JOBS):
        assert call(app, "GET", path)[1]["cache-control"] == "no-store"


def test_proof_id_is_masked_in_the_access_log_of_uvicorn(setup, caplog):
    proof_id = setup.ask(setup.app())
    caplog.set_level(logging.INFO, "uvicorn.access")
    paths = (
        JOBS + proof_id,
        "/proxy" + JOBS + proof_id + "/proof.bin?x=1",
        JOBS + proof_id.upper(),
        "/plugins/proof//jobs/" + proof_id,
        "/plugins/proof/jobs?id=" + proof_id,
        JOBS + proof_id + "a",
        JOBS + proof_id[:-1],
        JOBS + proof_id + proof_id,
        "/plugins/proof/jobs?id=" + "".join("%%%02x" % ord(digit) for digit in proof_id),
        "/plugins/proof/jobs?id=" + "".join("%%%02X" % ord(digit) for digit in proof_id.upper()),
        "/plugins/proof/jobs?id=" + proof_id[:20] + "%%%02x" % ord(proof_id[20]) + proof_id[21:],
    )
    # uvicorn does not run here: the test itself logs each path to uvicorn's logger, once as an
    # argument and once inside the message. In the last three paths all digits of the proof ID, or
    # one of them, are percent-encoded.
    for path in paths:
        logging.getLogger("uvicorn.access").info("%s %s %d", "GET", path, 200)
        logging.getLogger("uvicorn.access").info("GET " + path)
    assert proof_id[:32] not in caplog.text.lower() and "%" not in caplog.text
    assert len(caplog.messages) == 22
    assert caplog.messages[0] == "GET /plugins/proof/jobs/(hex digits) 200"
    assert caplog.messages[3] == "GET /proxy/plugins/proof/jobs/(hex digits)/proof.bin?x=1"
    # A run of fewer than 32 hex digits is not masked. "%30" counts as one digit; of "%0a", which
    # encodes no hex digit, "0a" counts as two, so the last run begins there.
    logging.getLogger("uvicorn.access").info("%s %s %s", "0" * 31, "0" * 32, "%0d%0a" + "%30" * 32)
    assert caplog.messages[22] == "0" * 31 + " (hex digits) %0d%(hex digits)"  # nothing is decoded


def test_delete_fails_and_says_so_if_the_job_file_cannot_be_deleted(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=True)
    before, real = setup.files(), os.unlink

    def keeps_the_job_file(name, **arguments):
        if name.endswith(".json"):
            raise PermissionError(1, "Operation not permitted", name)
        real(name, **arguments)

    monkeypatch.setattr(os, "unlink", keeps_the_job_file)
    assert error_of(call(app, "DELETE", JOBS + proof_id), 500) == proof_middleware.JOB_FAULT
    assert setup.files() == before


def test_delete_succeeds_if_only_the_record_cannot_be_deleted(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=True)
    real = os.unlink

    def keeps_the_record(name, **arguments):
        if name.endswith(".checked"):
            raise PermissionError(1, "Operation not permitted", name)
        real(name, **arguments)

    monkeypatch.setattr(os, "unlink", keeps_the_record)
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    error_of(call(app, "GET", JOBS + proof_id), 404)
    assert [name[64:] for name in os.listdir(setup.path("jobs")) if len(name) > 64] == [".checked"]


def test_delete_succeeds_if_only_the_record_cannot_be_read(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=True)
    real = proof_middleware.read_json

    def fails_once_the_job_file_is_gone(jobs, name):
        if name.endswith(".checked") and not os.path.exists(setup.job_path(proof_id)):
            raise OSError(5, "Input/output error", name)
        return real(jobs, name)

    monkeypatch.setattr(proof_middleware, "read_json", fails_once_the_job_file_is_gone)
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    error_of(call(app, "GET", JOBS + proof_id), 404)
    assert [name[64:] for name in os.listdir(setup.path("jobs")) if len(name) > 64] == [".checked"]
    assert setup.files()[1] == []


def test_unknown_proof_id_and_unknown_routes_get_404(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    for path in (
        JOBS + "0" * 64,
        JOBS + proof_id.upper(),
        JOBS + proof_id + "/tag.bin",
        JOBS + proof_id + "/../x",
        JOBS,
        "/plugins/proof/x",
    ):
        error_of(call(app, "GET", path), 404)
    error_of(call(app, "POST", JOBS + proof_id), 404)


def test_proof_id_is_in_no_file_name_or_file(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    (name,) = setup.jobs()
    with open(setup.path("jobs", name)) as f:
        assert proof_id not in name and proof_id not in f.read()
    assert os.stat(setup.path("jobs", name)).st_mode & 0o777 == 0o600


def test_reply_that_is_not_200_gets_no_proof_id_and_no_job(setup):
    app = setup.app()
    setup.vllm.status = 400
    status, headers, body = call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())
    assert (status, body) == (400, b"".join(setup.vllm.chunks)) and "x-proof-id" not in headers
    assert setup.jobs() == [] and os.listdir(setup.path("spool")) == ["mark.txt"]


def test_existing_job_file_is_not_replaced(setup):
    load_config(setup.write_config())
    with spool.open_private_dir(setup.path("jobs")) as jobs:
        name = proof_middleware.job_file("0" * 64)
        proof_middleware.write_file(jobs, name, b"first")
        with pytest.raises(FileExistsError):
            proof_middleware.write_file(jobs, name, b"second")
    assert setup.jobs() == [name]
    with open(setup.job_path("0" * 64)) as f:
        assert f.read() == "first"


@pytest.mark.parametrize("tag_key", [None, "", 7, "AB" * 32, "ab" * 33])
def test_job_file_with_a_malformed_tag_key_is_ignored(setup, caplog, tag_key):
    app = setup.app()
    proof_id = setup.ask(app)
    setup.respond()
    with open(setup.job_path(proof_id)) as f:
        job = json.load(f)
    with open(setup.job_path(proof_id), "w") as f:
        json.dump(job | {"tag_key": tag_key}, f)
    before = setup.files()
    for method, file in (("GET", ""), ("GET", "/proof.bin"), ("DELETE", "")):
        error_of(call(app, method, JOBS + proof_id + file), 404)
    assert setup.files() == before and caplog.text == ""


@pytest.mark.parametrize("poll", [False, True])
def test_delete_removes_the_job_then_its_record_then_the_response(setup, monkeypatch, poll):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, poll=poll)
    unlink, order = os.unlink, []
    monkeypatch.setattr(
        os, "unlink", lambda path, **where: unlink(path, **where) or order.append(path)
    )
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    order = [path for path in order if not path.startswith(".")]  # ignore temporary files
    assert order[:3] == [
        proof_middleware.job_file(proof_id),
        proof_middleware.job_file(proof_id, ".checked"),
        os.path.basename(proof_path),
    ]
    assert setup.files() == ([], [])


def test_clean_removes_jobs_and_records_before_responses_and_counts_each_job_once(
    setup, monkeypatch
):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    orphan = setup.job_path("0" * 64, ".checked")  # a record whose job is gone
    shutil.copy(setup.job_path(proof_id, ".checked"), orphan)
    for path in (setup.job_path(proof_id), setup.job_path(proof_id, ".checked"), orphan):
        os.utime(path, (0, 0))
    unlink, order = os.unlink, []
    monkeypatch.setattr(
        os, "unlink", lambda path, **where: order.append(path) or unlink(path, **where)
    )
    # 2 jobs (one with its record, one orphan record), 0 requests, 1 response,
    # 0 files that could not be deleted
    assert proof_middleware.clean(app.config, 1) == (2, 0, 1, 0)
    assert sorted(path.partition(".")[2] for path in order[:3]) == ["checked", "checked", "json"]
    assert os.path.basename(proof_path) in order[3:] and setup.files() == ([], [])


@pytest.mark.parametrize("error", [OSError(5, "dummy"), ValueError("dummy")])
def test_delete_succeeds_when_deleting_the_response_fails(setup, monkeypatch, caplog, error):
    app = setup.app()
    proof_id = setup.ask(app)
    setup.respond()

    def fail(*arguments):
        raise error

    monkeypatch.setattr(spool, "delete_response_at", fail)
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert setup.jobs() == [] and len(caplog.records) == 1
    assert "was not deleted" in caplog.text
    error_of(call(app, "GET", JOBS + proof_id), 404)


# ---- The tag ----


def tamper_change_a_byte(setup, names):
    with open(setup.path("spool", "out", names[0] + ".proof"), "r+b") as f:
        f.write(b"D")


def tamper_swap_two_responses(setup, names):
    for suffix in ("y", "tag", "proof"):
        one, other = (setup.path("spool", "out", "%s.%s" % (name, suffix)) for name in names)
        os.rename(one, one + ".x"), os.rename(other, one), os.rename(one + ".x", other)


def tamper_remove_the_tag(setup, names):
    os.remove(setup.path("spool", "out", names[0] + ".tag"))


def tamper_turn_a_proof_into_a_failure(setup, names):
    os.remove(setup.path("spool", "out", names[0] + ".y"))
    os.remove(setup.path("spool", "out", names[0] + ".proof"))
    open(setup.path("spool", "out", names[0] + ".failed"), "w").close()


def tamper_replace_the_proof_by_a_symlink(setup, names):
    os.rename(setup.path("spool", "out", names[0] + ".proof"), setup.path("elsewhere"))
    os.symlink(setup.path("elsewhere"), setup.path("spool", "out", names[0] + ".proof"))


@pytest.mark.parametrize(
    "tamper",
    [
        tamper_change_a_byte,
        tamper_swap_two_responses,
        tamper_remove_the_tag,
        tamper_turn_a_proof_into_a_failure,
        tamper_replace_the_proof_by_a_symlink,
    ],
)
def test_response_is_served_only_if_its_tag_matches(setup, tamper, caplog):
    app = setup.app()
    proof_ids = [setup.ask(app), setup.ask(app)]
    names = setup.respond()
    tamper(setup, names)
    for name in names:
        setup.set_one_time(name)
    statuses = {}
    for proof_id in proof_ids:
        with open(setup.job_path(proof_id)) as f:
            request_id = json.load(f)["request_id"]
        statuses[request_id] = call(app, "GET", JOBS + proof_id + "/proof.bin")[0]
    assert 200 not in statuses.values()


def test_no_route_serves_the_tag_or_its_key(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    (name,) = setup.respond()
    error_of(call(app, "GET", JOBS + proof_id + "/tag.bin"), 404)
    with (
        open(setup.path("spool", "out", name + ".tag"), "rb") as f,
        open(setup.job_path(proof_id)) as g,
    ):
        secrets = [f.read().hex(), json.load(g)["tag_key"]]
    assert not any(secret in json.dumps(setup.state(app, proof_id)) for secret in secrets)


@pytest.mark.parametrize("proof", [b"", b"12345678901"])
def test_proof_of_a_disallowed_size_is_not_read(setup, monkeypatch, caplog, proof):
    monkeypatch.setattr(proof_middleware, "MAX_PROOF_BYTES", 10)
    app = setup.app()
    proof_id, proof_path = setup.proved(app, proof, poll=False)
    lengths = count_reads(monkeypatch, proof_path)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert lengths == [] and "bytes" not in caplog.text
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))  # the second try; nothing wiped
    assert os.path.exists(proof_path)


@pytest.mark.parametrize("tampered", [False, True])
def test_delete_without_an_earlier_poll_checks_the_tag_first(setup, tampered):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, poll=False)
    if tampered:
        tamper_change_a_byte(setup, [os.path.basename(proof_path).removesuffix(".proof")])
        setup.set_one_time("")
    if tampered:
        assert call(app, "DELETE", JOBS + proof_id)[0] == 409
        assert os.path.exists(setup.job_path(proof_id))
    else:
        assert json.loads(call(app, "DELETE", JOBS + proof_id)[2])["state"] == "proved"
        assert setup.files() == ([], [])


# ---- Records ----


def test_first_poll_of_a_proved_job_writes_a_record(setup):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, poll=False)
    record = setup.job_path(proof_id, ".checked")
    assert not os.path.exists(record)
    setup.state(app, proof_id)
    info = os.stat(proof_path)
    with open(record) as f, open(setup.job_path(proof_id)) as g:
        text, tag_key = f.read(), json.load(g)["tag_key"]
    assert json.loads(text) == {
        "y": "00000000",
        "proof": [info.st_size, info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns],
        "verified": True,
    }
    assert os.stat(record).st_mode & 0o777 == 0o600 and len(setup.jobs()) == 2
    assert proof_id not in record and proof_id not in text and tag_key not in text


def test_polls_read_the_proof_once(setup, monkeypatch):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, b"0123456789", poll=False)
    monkeypatch.setattr(proof_middleware, "CHUNK", 4)
    lengths = count_reads(monkeypatch, proof_path)
    for _ in range(5):
        assert setup.state(app, proof_id)["state"] == "proved"
        assert call(app, "GET", JOBS + proof_id + "/y.bin")[0] == 200
    assert lengths == [4, 4, 2, 0]


def test_y_that_changes_during_the_check_is_not_served(setup, monkeypatch):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, poll=False)
    read, y_path = proof_middleware.TagCheck.read, proof_path.removesuffix("proof") + "y"

    def read_and_change_y(check):
        with open(y_path, "r+b") as f:
            f.write(b"evil")
        return read(check)

    monkeypatch.setattr(proof_middleware.TagCheck, "read", read_and_change_y)
    assert call(app, "GET", JOBS + proof_id + "/y.bin")[::2] == (200, bytes(4))
    with open(setup.job_path(proof_id, ".checked")) as f:
        assert json.load(f)["y"] == "00000000"


def test_delete_of_a_checked_job_does_not_read_the_proof(setup, monkeypatch):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    lengths = count_reads(monkeypatch, proof_path)
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert lengths == [] and setup.files() == ([], [])


def test_two_first_polls_at_once_start_one_check(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    checks = []

    def slow_check(*_):
        checks.append(1)
        time.sleep(0.2)
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", slow_check)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        states = list(pool.map(lambda _: setup.state(app, proof_id)["state"], range(2)))
    assert states == ["proved", "proved"] and len(checks) == 1
    names = [proof_middleware.job_file(proof_id, suffix) for suffix in (".json", ".checked")]
    assert setup.files()[0] == sorted(names)


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        "[]",
        '{"y": "00000000"}',
        '{"y": 0, "proof": [11, 1, 1, 1, 1]}',
        '{"y": "0000000", "proof": [11, 1, 1, 1, 1]}',
        '{"y": "00000000", "proof": [11, 1, 1, 1]}',
        '{"y": "00000000", "proof": [11, 1, 1, 1, "1"]}',
        '{"y": "00000000", "proof": [11, 1, 1, 1, true]}',
        '{"y": "00000000", "proof": [11, 1, 1, 1, 1]}',  # tag matched, never verified
    ],
)
def test_malformed_record_counts_for_nothing_and_is_kept(setup, caplog, text):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    with open(setup.job_path(proof_id, ".checked"), "w") as f:
        f.write(text)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert "stands in the way" in caplog.text
    with open(setup.job_path(proof_id, ".checked")) as f:
        assert f.read() == text


def test_leftover_temporary_file_is_harmless_and_kept(setup):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    names = [proof_middleware.job_file(proof_id, suffix) for suffix in (".json", ".checked")]
    left = [setup.path("jobs", ".%s.%s" % (name, "0" * 16)) for name in names]
    for path in left:
        open(path, "w").close()
        os.utime(path, (0, 0))
    assert setup.state(app, proof_id)["state"] == "proved"
    assert proof_middleware.clean(app.config, 1) == (0, 0, 1, 0)  # the old response
    assert len(setup.jobs()) == 4 and all(map(os.path.exists, left))


def test_response_without_a_proof_gets_no_record(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    (name,) = setup.respond(failed=True)
    assert setup.state(app, proof_id)["state"] == "no_proof"
    assert setup.jobs() == [proof_middleware.job_file(proof_id)]
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(bytes(32))
    setup.set_one_time(name)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))  # its own poll asks for the check


def test_job_of_another_commitment_under_the_same_policy_name_is_unknown(setup, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    not_provable = setup.ask(app, temperature=1)
    before, earlier = setup.files(), dict(setup.entry)
    routes = (("GET", ""), ("GET", "/y.bin"), ("GET", "/proof.bin"), ("DELETE", ""))
    setup.entry.update(commitment_id="cd" * 32, spool_dir=setup.path("spool-2"))
    app = setup.app()
    for method, file in routes:
        error_of(call(app, method, JOBS + proof_id + file), 404)
    assert setup.state(app, not_provable)["state"] == "not_provable"
    assert setup.files() == before and caplog.text == ""
    setup.entry.update(earlier)
    app = setup.app()
    assert setup.state(app, proof_id)["state"] == "proved"
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"dummy proof")
    with open(setup.job_path(proof_id)) as f:
        job = json.load(f)
    assert job.pop("commitment_id") == COMMITMENT_ID
    with open(setup.job_path(proof_id), "w") as f:  # a job file without a commitment ID
        json.dump(job, f)
    for method, file in routes:
        error_of(call(app, method, JOBS + proof_id + file), 404)
    assert setup.files() == before


def test_delete_during_a_first_check_leaves_no_record(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)

    def delete_the_job(*_):
        os.unlink(setup.job_path(proof_id))
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", delete_the_job)
    call(app, "GET", JOBS + proof_id)
    assert setup.files()[0] == []
    error_of(call(app, "GET", JOBS + proof_id), 404)


def test_responses_that_are_swapped_after_their_checks_are_not_served(setup):
    app = setup.app()
    proof_ids = [setup.ask(app), setup.ask(app)]
    names = setup.respond()
    assert [setup.state(app, proof_id)["state"] for proof_id in proof_ids] == ["proved"] * 2
    tamper_swap_two_responses(setup, names)
    setup.set_one_time("")
    for proof_id in proof_ids:
        for file in ("/y.bin", "/proof.bin"):
            assert call(app, "GET", JOBS + proof_id + file)[0] != 200


def put_another_proof(setup, proof_id, proof_path, proof, honest):
    """Replace a response's proof file, as a second response to the request does. honest: with the
    matching tag file."""
    with open(proof_path + ".new", "wb") as f:
        f.write(proof)
    os.replace(proof_path + ".new", proof_path)
    if honest:
        name = os.path.basename(proof_path).removesuffix(".proof")
        with open(setup.job_path(proof_id)) as f:
            key = bytes.fromhex(json.load(f)["tag_key"])
        with open(proof_path.removesuffix("proof") + "tag", "wb") as f:
            f.write(spool.response_tag(key, name, bytes(4), proof))
    setup.set_one_time("")


def test_second_response_to_a_request_is_checked_like_a_first_one(setup, caplog):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    with open(setup.job_path(proof_id, ".checked")) as f:
        first_record = json.load(f)
    put_another_proof(setup, proof_id, proof_path, b"another dummy proof", honest=True)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"another dummy proof")
    with open(setup.job_path(proof_id, ".checked")) as f:
        second_record = json.load(f)
    assert second_record["proof"][0] == 19 and second_record["proof"][2] != first_record["proof"][2]
    assert setup.state(app, proof_id)["files"] == {"y.bin": 4, "proof.bin": 19}
    assert caplog.text == ""


@pytest.mark.parametrize("proof", [b"DUMMY PROOF", b"dummy proof and more"])
@pytest.mark.parametrize("method, file", [("GET", "/proof.bin"), ("DELETE", "")])
def test_second_response_with_a_wrong_tag_is_not_served(setup, caplog, proof, method, file):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    put_another_proof(setup, proof_id, proof_path, proof, honest=False)
    assert call(app, method, JOBS + proof_id + file)[0] == (404 if method == "GET" else 409)
    assert not os.path.exists(setup.job_path(proof_id, ".checked"))


def change_in_place(proof_path):
    """Change a proof file's first byte; size and modification time stay."""
    with open(proof_path, "r+b") as f:
        f.write(b"D")
    os.utime(proof_path, ns=(10**18, 10**18))


def test_proof_that_is_changed_in_place_after_its_check_is_not_served(setup, caplog):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    # Wait for the file system's clock to advance, so that the status-change time shows the change.
    open(setup.path("scratch"), "w").close()
    deadline = time.monotonic() + 10
    while os.stat(setup.path("scratch")).st_ctime_ns <= os.stat(proof_path).st_ctime_ns:
        assert time.monotonic() < deadline, "the file system's clock does not move"
        time.sleep(0.001)
        os.utime(setup.path("scratch"))
    change_in_place(proof_path)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] == 404
    assert not os.path.exists(setup.job_path(proof_id, ".checked"))


def test_download_is_cut_after_a_change_that_the_record_cannot_show(setup, monkeypatch, caplog):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    with open(setup.job_path(proof_id, ".checked")) as f:
        recorded = json.load(f)["proof"]
    change_in_place(proof_path)
    # Simulate a write too quick for the file times to show.
    monkeypatch.setattr(proof_middleware, "identity", lambda fd: recorded)
    assert setup.state(app, proof_id)["state"] == "proved"
    events = []
    with pytest.raises(proof_middleware.ResponseCut):
        download(app, proof_id, events)
    assert events == [("head", 11)]  # its one chunk is held back
    assert len(caplog.records) == 1 and "was broken off" in caplog.text


def remove_the_response(setup, proof_path):
    for suffix in ("proof", "tag", "y"):
        os.remove(proof_path.removesuffix("proof") + suffix)


def remove_out(setup, proof_path):
    os.rename(setup.path("spool", "out"), setup.path("elsewhere"))


@pytest.mark.parametrize("remove", [remove_the_response, remove_out])
def test_checked_job_whose_response_is_gone_is_expired(setup, caplog, remove):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    request, jobs = setup.state(app, proof_id)["request"], setup.jobs()
    remove(setup, proof_path)
    for _ in range(3):
        assert setup.state(app, proof_id) == {
            "state": "expired",
            "policy": "exact",
            "request": request,
            "detail": proof_middleware.EXPIRED,
        }
        for file in ("/y.bin", "/proof.bin"):
            assert "no such file" in error_of(call(app, "GET", JOBS + proof_id + file), 404)
    assert setup.jobs() == jobs and len(jobs) == 2 and caplog.text == ""
    assert json.loads(call(app, "DELETE", JOBS + proof_id)[2])["state"] == "expired"
    assert setup.jobs() == []
    error_of(call(app, "GET", JOBS + proof_id), 404)


def test_delete_of_an_expired_job_removes_what_is_left_of_its_response(setup):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    os.remove(proof_path)  # as an interrupted deletion leaves it
    assert len(setup.files()[1]) == 2
    state = setup.state(app, proof_id)
    assert state["state"] == "expired" and "does not have it now" in state["detail"]
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert setup.files() == ([], [])


def test_new_response_to_an_expired_job_is_checked_and_served(setup):
    app = setup.app()
    proof_id, proof_path = setup.proved(app)
    remove_the_response(setup, proof_path)
    assert setup.state(app, proof_id)["state"] == "expired"
    with open(proof_path.removesuffix("proof") + "y", "wb") as f:
        f.write(bytes(4))
    put_another_proof(setup, proof_id, proof_path, b"another dummy proof", honest=True)
    assert setup.state(app, proof_id)["files"] == {"y.bin": 4, "proof.bin": 19}
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"another dummy proof")


# ---- Downloads ----


def download(app, proof_id, events):
    """Download proof.bin, appending ("head", length) and ("send", chunk) to events."""

    async def send(message):
        if message["type"] == "http.response.start":
            events.append(("head", int(dict(message["headers"])[b"content-length"])))
        else:
            events.append(("send", message["body"]))

    call(app, "GET", JOBS + proof_id + "/proof.bin", send=send)


def test_proof_is_streamed_with_at_most_two_chunks_in_memory(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, b"0123456789")
    monkeypatch.setattr(proof_middleware, "CHUNK", 4)
    read, events = proof_middleware.TagCheck.read, []

    def noting_read(check):
        events.append(("read", read(check)))
        return events[-1][1]

    monkeypatch.setattr(proof_middleware.TagCheck, "read", noting_read)
    download(app, proof_id, events)
    assert events == [
        ("head", 10),
        ("read", b"0123"),
        ("read", b"4567"),
        ("send", b"0123"),
        ("read", b"89"),
        ("send", b"4567"),
        ("read", b""),
        ("send", b"89"),
    ]


def overwrite_the_first_byte(path):
    with open(path, "r+b") as f:
        f.write(b"x")


def truncate_to_six_bytes(path):
    os.truncate(path, 6)


@pytest.mark.parametrize(
    "proof, change, sent",
    [
        (b"0123456789", overwrite_the_first_byte, [b"x123", b"4567"]),
        (b"0123", overwrite_the_first_byte, []),
        (b"0123456789", truncate_to_six_bytes, [b"0123"]),
    ],
)
def test_proof_that_changes_during_its_download_is_not_sent_whole(
    setup, monkeypatch, caplog, proof, change, sent
):
    app = setup.app()
    proof_id, proof_path = setup.proved(app, proof)
    monkeypatch.setattr(proof_middleware, "CHUNK", 4)
    read, events = proof_middleware.TagCheck.read, []

    def change_and_read(check):
        if events == [("head", len(proof))]:  # after the head, before any chunk is sent
            change(proof_path)
        return read(check)

    monkeypatch.setattr(proof_middleware.TagCheck, "read", change_and_read)
    with pytest.raises(proof_middleware.ResponseCut):
        download(app, proof_id, events)
    # The last chunk is held back.
    assert events == [("head", len(proof)), *(("send", chunk) for chunk in sent)]
    assert len(caplog.records) == 1 and "was broken off" in caplog.text
    assert len(os.listdir(setup.path("spool", "out"))) == 3


def test_many_downloads_at_once(setup):
    app = setup.app()
    proof_id, _ = setup.proved(app, os.urandom(100000), poll=False)
    with concurrent.futures.ThreadPoolExecutor(20) as pool:
        results = list(
            pool.map(lambda _: call(app, "GET", JOBS + proof_id + "/proof.bin"), range(20))
        )
    assert {status for status, _, _ in results} == {200}
    assert len({body for _, _, body in results}) == 1 and len(results[0][2]) == 100000


@pytest.mark.filterwarnings("ignore::ResourceWarning")  # a cancelled call drops a file object
@pytest.mark.parametrize("leaves", ["while the files are opened", "while a chunk is sent"])
def test_client_that_leaves_during_a_download_leaves_no_open_file(setup, monkeypatch, leaves):
    app = setup.app()
    proof_id, _ = setup.proved(app, b"0123456789")
    monkeypatch.setattr(proof_middleware, "CHUNK", 4)
    job_state, reached, go_on = app.job_state, threading.Event(), threading.Event()

    def slow_job_state(*arguments):
        reached.set()
        go_on.wait(10)
        return job_state(*arguments)

    async def send(message):  # a client that stops reading after the head
        if message["type"] == "http.response.body":
            reached.set()
            await asyncio.Event().wait()

    async def leave():
        task = asyncio.ensure_future(
            app.job_route("GET", proof_id, "proof.bin", stays_connected, send)
        )
        await asyncio.to_thread(reached.wait, 10)
        task.cancel()
        await asyncio.wait([task])
        go_on.set()
        return task.cancelled()

    if leaves == "while the files are opened":
        monkeypatch.setattr(app, "job_state", slow_job_state)
    gc.collect()
    before = len(os.listdir("/dev/fd"))
    assert asyncio.run(leave())  # asyncio.run() waits for the thread too
    gc.collect()
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.parametrize("file", ["the proof", "a published file"])
def test_reading_stops_when_the_client_has_disconnected(setup, monkeypatch, caplog, file):
    setup.public_key_dir()
    with open(setup.path("public", "key.bin"), "wb") as f:
        f.write(bytes(40))
    app = setup.app()
    proof_id, _ = setup.proved(app, bytes(40))
    monkeypatch.setattr(proof_middleware, "CHUNK", 4)
    files = []

    class CountingFile(io.FileIO):
        reads = 0

        def read(self, size=-1):
            self.reads += 1
            return super().read(size)

    def counting_open(file, mode="r", buffering=-1):
        if buffering:  # the two files sent in chunks are unbuffered
            return open(file, mode, buffering)
        files.append(CountingFile(file))
        return files[-1]

    monkeypatch.setattr(proof_middleware, "open", counting_open, raising=False)
    path = JOBS + proof_id + "/proof.bin" if file == "the proof" else PUBLISHED + "exact/key.bin"
    # The web server reports the disconnect, and takes what is sent all the same, as uvicorn does.
    status, headers, body = call(
        app, "GET", path, body=[*in_chunks(b""), {"type": "http.disconnect"}]
    )
    assert (status, headers["content-length"]) == (200, "40") and len(body) <= 8
    assert len(files) == 1 and files[0].reads <= 2 and files[0].closed and caplog.text == ""


@pytest.mark.parametrize("sender", ["stream_proof", "send_file"])
def test_file_stays_open_until_a_read_in_flight_has_finished(setup, monkeypatch, sender):
    setup.public_key_dir()
    app = setup.app()
    proof_id, _ = setup.proved(app)
    reading, go_on, files = threading.Event(), threading.Event(), []

    class SlowFile(io.FileIO):
        def read(self, size=-1):
            files.append(self)
            reading.set()
            go_on.wait(10)
            return super().read(size)

    def keep_open(check, fds):
        check.file = SlowFile(fds.pop("proof"))

    async def send(message):
        pass

    async def leave():
        if sender == "stream_proof":
            call_of_the_client = app.job_route("GET", proof_id, "proof.bin", stays_connected, send)
        else:
            file = SlowFile(setup.path("public", "key.bin"))
            call_of_the_client = proof_middleware.send_file(stays_connected, send, file)
        task = asyncio.ensure_future(call_of_the_client)
        await asyncio.to_thread(reading.wait, 10)
        task.cancel()
        await asyncio.wait([task])
        closed_at_once = files[0].closed
        go_on.set()
        for _ in range(1000):
            if files[0].closed:
                break
            await asyncio.sleep(0.01)
        return task.cancelled(), closed_at_once, files[0].closed

    monkeypatch.setattr(proof_middleware.TagCheck, "keep_open", keep_open)
    assert asyncio.run(leave()) == (True, False, True)


# ---- What a proof request may contain ----


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"logit_bias": {"5": 100}}, "has the field 'logit_bias'"),
        ({"seed": 1}, "has the field 'seed'"),
        ({"temperature": 1}, "decoded greedily"),
        ({"temperature": None}, "decoded greedily"),
        ({"temperature": False}, "decoded greedily"),
        ({"temperature": "0"}, "decoded greedily"),
        ({"return_token_ids": False}, "return_token_ids"),
        ({"model": "another"}, "only the served model dummy"),
        ({"prompt": [[1, 2], [3]]}, "one list of token IDs"),
        ({"messages": [{"role": "user", "content": "x", "name": "y"}]}, '"role" and "content"'),
        (
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
            '"role" and "content"',
        ),
    ],
)
def test_request_that_could_steer_the_reply_gets_no_proof(setup, changes, expected):
    app = setup.app()
    state = setup.state(app, setup.ask(app, **changes))
    assert state["state"] == "not_provable" and expected in state["detail"]
    assert not os.path.exists(setup.path("spool", "in"))


def test_field_that_occurs_twice_gets_no_proof(setup):
    app = setup.app()
    body = json.dumps(FIELDS)[:-1].encode() + b', "temperature": 0}'
    state = setup.state(app, call(app, "POST", "/v1/completions", ASK, body)[1]["x-proof-id"])
    assert state["state"] == "not_provable" and "twice" in state["detail"]


def test_request_without_a_temperature_gets_no_proof(setup):
    app = setup.app()
    body = json.dumps({name: value for name, value in FIELDS.items() if name != "temperature"})
    proof_id = call(app, "POST", "/v1/completions", ASK, body.encode())[1]["x-proof-id"]
    assert "decoded greedily" in setup.state(app, proof_id)["detail"]


@pytest.mark.parametrize("temperature", [0.0, -0.0])
def test_temperature_zero_may_be_a_float(setup, temperature):
    app = setup.app()
    assert setup.state(app, setup.ask(app, temperature=temperature))["state"] == "waiting"


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda r: r["choices"][0].update(finish_reason=None), "before vLLM gave a finish reason"),
        (lambda r: r["choices"].append({**r["choices"][0], "index": 1}), "returned several"),
        (lambda r: r.update(error={"message": "dummy"}), "reported an error"),
    ],
)
def test_reply_that_is_not_one_finished_choice_gets_no_proof(setup, change, expected):
    app = setup.app()
    reply = json.loads(setup.vllm.chunks[0])
    change(reply)
    setup.vllm.chunks = [json.dumps(reply).encode()]
    state = setup.state(app, setup.ask(app))
    assert state["state"] == "not_provable" and expected in state["detail"]
    assert not os.path.exists(setup.path("spool", "in"))


@pytest.mark.parametrize("name", ["seed", "top_p", "top_k", "min_p", "repetition_penalty"])
def test_sampling_field_gets_no_proof(setup, name):
    app = setup.app()
    state = setup.state(app, setup.ask(app, **{name: 1}))
    assert state["state"] == "not_provable" and "has the field %r" % name in state["detail"]


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b'"text"', b""])
def test_request_body_that_is_no_json_object_gets_no_proof(setup, body):
    app = setup.app()
    state = setup.state(app, call(app, "POST", "/v1/completions", ASK, body)[1]["x-proof-id"])
    assert state["state"] == "not_provable" and "not one JSON object" in state["detail"]
    assert setup.vllm.bodies == [body]


def test_prompt_and_reply_must_fit_into_one_request_file(setup):
    app = setup.app()
    reply = json.loads(setup.vllm.chunks[0])
    for reply_tokens, expected in (
        (16 - len(PROMPT), "waiting"),
        (17 - len(PROMPT), "not_provable"),
    ):
        reply["choices"][0]["token_ids"] = [7] * reply_tokens
        setup.vllm.chunks = [json.dumps(reply).encode()]
        state = setup.state(app, setup.ask(app))
        assert state["state"] == expected
    assert "at most 16 tokens together. These have 17." in state["detail"]
    assert len(setup.requests()) == 1


def test_reply_that_is_too_long_to_copy_gets_no_proof(setup, monkeypatch):
    app = setup.app("completion_stream.txt")
    monkeypatch.setattr(proof_middleware, "MAX_REPLY_BYTES", len(b"".join(setup.vllm.chunks)) - 1)
    status, headers, body = call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())
    assert (status, body) == (200, b"".join(setup.vllm.chunks))
    state = setup.state(app, headers["x-proof-id"])
    assert state["state"] == "not_provable" and "The reply is longer than" in state["detail"]


def test_full_queue_or_a_token_outside_the_vocabulary_gets_no_proof(setup):
    setup.entry["max_waiting"] = 1
    app = setup.app()
    assert setup.state(app, setup.ask(app))["state"] == "waiting"
    assert "queue is full" in setup.state(app, setup.ask(app))["detail"]
    setup.write_parameters(lt=4, V=12)
    assert "vocabulary of 12 tokens" in setup.state(app2 := setup.app(), setup.ask(app2))["detail"]


def test_reply_gets_no_proof_while_too_many_responses_are_stored(setup):
    setup.entry["max_stored"] = 2
    app = setup.app()
    first = setup.ask(app)
    setup.respond()  # three files, which count as one response
    assert setup.state(app, setup.ask(app))["state"] == "waiting"
    setup.respond(failed=True)  # a response without a proof counts too
    state = setup.state(app, setup.ask(app))
    assert state["state"] == "not_provable" and "stores the most responses" in state["detail"]
    assert setup.requests() == {}
    assert call(app, "DELETE", JOBS + first)[0] == 200
    assert setup.state(app, setup.ask(app))["state"] == "waiting"


def test_deleting_a_job_that_is_not_provable(setup):
    app = setup.app()
    proof_id = setup.ask(app, temperature=1)
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    error_of(call(app, "GET", JOBS + proof_id), 404)


def test_proof_request_that_is_too_long_never_reaches_vllm(setup):
    app = setup.app()
    fits = json.dumps(FIELDS).encode().ljust(proof_middleware.MAX_REQUEST_BYTES)
    assert "at most 1 MiB" in error_of(call(app, "POST", "/v1/completions", ASK, fits + b" "), 413)
    assert setup.vllm.bodies == [] and setup.jobs() == []
    assert call(app, "POST", "/v1/completions", ASK, fits)[0] == 200 and setup.vllm.bodies == [fits]


def test_proof_request_in_several_chunks_reaches_vllm_and_the_checks_in_one_piece(setup):
    app = setup.app()
    allowed, forbidden = json.dumps(FIELDS).encode(), json.dumps({**FIELDS, "seed": 1}).encode()
    states = []
    for body in (allowed, forbidden):
        cut = body.rindex(b'"') - 2  # for the second body, inside the name "seed"
        messages = in_chunks(body[:10], body[10:cut], body[cut:])
        proof_id = call(app, "POST", "/v1/completions", ASK, messages)[1]["x-proof-id"]
        states.append(setup.state(app, proof_id))
    assert setup.vllm.bodies == [allowed, forbidden]
    assert [state["state"] for state in states] == ["waiting", "not_provable"]
    assert "has the field 'seed'" in states[1]["detail"]


def test_client_that_leaves_in_the_middle_of_a_proof_request(setup):
    app = setup.app()
    first = {"type": "http.request", "body": b'{"model": "dummy",', "more_body": True}
    messages = [first, {"type": "http.disconnect"}]
    assert call(app, "POST", "/v1/completions", ASK, messages)[0] is None
    assert setup.vllm.bodies == []
    assert setup.jobs() == [] and os.listdir(setup.path("spool")) == ["mark.txt"]


@pytest.mark.parametrize("name", ["X-Proof-Policy", "X-PROOF-POLICY"])
def test_case_of_the_header_name_does_not_matter(setup, name):
    app = setup.app()
    result = call(app, "POST", "/v1/completions", {name: "exact"}, json.dumps(FIELDS).encode())
    assert "x-proof-id" in result[1] and len(setup.requests()) == 1
    error_of(call(app, "GET", "/v1/models", {name: "exact"}), 400)


def test_root_path_is_removed_before_the_route_is_compared(setup):
    app = setup.app()
    assert b'"policies"' in call(app, "GET", "/plugins/proof/policies", root_path="/proxy")[2]
    body = json.dumps(FIELDS).encode()
    assert "x-proof-id" in call(app, "POST", "/v1/completions", ASK, body, root_path="/proxy")[1]
    error_of(call(app, "POST", "/proxy/v1/completions", ASK, body), 400)


@pytest.mark.parametrize(
    "method, path, headers",
    [
        ("GET", "/v1/models", ASK),
        ("POST", "/v1/completions", {"x-proof-policy": "another"}),
        ("POST", "/v1/completions", [("x-proof-policy", "exact"), ("x-proof-policy", "exact")]),
    ],
)
def test_policy_header_that_cannot_be_served(setup, method, path, headers):
    app = setup.app()
    error_of(call(app, method, path, headers, json.dumps(FIELDS).encode()), 400)
    assert setup.vllm.bodies == []


def test_policies_route(setup):
    assert json.loads(call(setup.app(), "GET", "/plugins/proof/policies")[2]) == {
        "policies": {
            "exact": {
                "served_model": "dummy",
                "commitment_id": COMMITMENT_ID,
                "max_total_tokens": 16,
                "kind": "exact",
                "limits": {"C": 256, "theta": 256, "cap": 0, "s": 15},
                "closed": False,
            }
        }
    }


def test_closed_policy_takes_no_new_request_and_serves_what_it_has(setup, capsys):
    setup.public_key_dir()
    proof_id, _ = setup.proved(setup.app(), poll=False)
    setup.entry["closed"] = True
    app, before = setup.app(), setup.files()
    result = call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())
    assert "that is not closed" in error_of(result, 400)
    assert setup.vllm.bodies == [] and setup.requests() == {} and setup.files() == before
    policies = json.loads(call(app, "GET", "/plugins/proof/policies")[2])["policies"]
    assert policies["exact"]["closed"] is True
    assert proof_middleware.main(["check", setup.path("config.json")]) == 0
    assert "policy exact (closed): exact (" in capsys.readouterr().out
    assert call(app, "GET", PUBLISHED + "exact/key.bin")[::2] == (200, b"dummy key.bin")
    assert setup.state(app, proof_id)["state"] == "proved"
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"dummy proof")
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200 and setup.files() == ([], [])


@pytest.mark.parametrize("limits", [{"C": 512, "theta": 300, "cap": 3}, {"C": 512}, {"C": 300, "cap": 1}])
def test_policy_with_other_limits_is_a_clipped_gap_policy(setup, limits):
    setup.write_parameters(**limits)
    policy = json.loads(call(setup.app(), "GET", "/plugins/proof/policies")[2])["policies"]["exact"]
    assert policy["kind"] == "clipped-gap"
    assert policy["limits"] == {"C": 256, "theta": 256, "cap": 0, "s": 15, **limits}


def test_each_policy_has_its_vocabulary(setup):
    setup.write_parameters(V=12)
    setup.write_parameters("larger.txt", V=200)
    setup.config["policies"]["larger"] = {
        **setup.entry,
        "parameter_file": setup.path("larger.txt"),
        "spool_dir": setup.path("spool-larger"),
    }
    app, states = setup.app(), {}
    for name in ("exact", "larger"):
        headers = {"x-proof-policy": name}
        proof_id = call(app, "POST", "/v1/completions", headers, json.dumps(FIELDS).encode())[1]
        states[name] = setup.state(app, proof_id["x-proof-id"])["state"]
    assert states == {"exact": "not_provable", "larger": "waiting"}


# ---- Other calls ----


def reaches_vllm_untouched(app, scope):
    """Return whether the middleware calls vLLM once, with this scope and the web server's receive
    and send, which are not callable here: reading the body would fail."""
    receive, send, reached = object(), object(), []

    async def vllm(*arguments):
        reached.append(arguments)

    app.app = vllm
    asyncio.run(app(scope, receive, send))
    given = (scope, receive, send)
    return len(reached) == 1 and all(got is sent for got, sent in zip(reached[0], given))


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("GET", "/metrics"),
        ("GET", "/"),
        ("DELETE", "/any/route"),
        ("GET", "/plugins/proof"),
    ],
)
def test_call_without_the_header_reaches_vllm_untouched(setup, method, path):
    headers = {"x-proof-policy-2": "exact", "content-length": "9999999"}
    assert reaches_vllm_untouched(setup.app(), http_scope(method, path, headers))
    assert setup.jobs() == [] and os.listdir(setup.path("spool")) == ["mark.txt"]


@pytest.mark.parametrize(
    "body",
    [
        b" " * (proof_middleware.MAX_REQUEST_BYTES + 1),
        b'{"prompt": [1, 200, -1]}',
        b'{"tools": [{"parameters": {"properties": {"address_line_1": {}}}}]}',
        b"not JSON",
    ],
)
def test_body_of_a_call_without_the_header_is_left_to_vllm(setup, body):
    app = setup.app()
    assert call(app, "POST", "/v1/chat/completions", {}, body)[2] == b"".join(setup.vllm.chunks)
    assert setup.vllm.bodies == [body]


@pytest.mark.parametrize("scope_type", ["websocket", "lifespan", "another"])
def test_scope_that_is_not_http_reaches_vllm_untouched(setup, scope_type):
    assert reaches_vllm_untouched(setup.app(), {"type": scope_type, "path": "/v1/realtime"})


# ---- The API key ----

KEY = "dummy-key"


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("authorization", "Bearer dummy-kez")],
        [("authorization", "Bearer dummy-ke")],
        [("authorization", "Bearer dummy-key2")],
        [("authorization", "Bearer  dummy-key")],
        [("authorization", "Bearer")],
        [("authorization", "Bearer ")],
        [("authorization", "dummy-key")],
        [("authorization", "Basic dummy-key")],
        [("authorization", "Bearer dummy-key"), ("authorization", "Bearer other")],
        [("x-api-key", "dummy-key")],
    ],
)
def test_proof_routes_need_the_api_key_of_a_server_that_has_one(setup, monkeypatch, headers):
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    setup.ask_headers["authorization"] = "Bearer " + KEY
    setup.public_key_dir()
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    before = setup.files()
    for method, route in (
        ("GET", "policies"),
        ("GET", "jobs/" + proof_id),
        ("GET", "jobs/" + proof_id + "/proof.bin"),
        ("DELETE", "jobs/" + proof_id),
        ("GET", "published/exact/key.bin"),
        ("GET", "no/such/route"),
    ):
        message = error_of(call(app, method, "/plugins/proof/" + route, headers), 401)
        assert "Authorization: Bearer KEY" in message and KEY not in message
    assert setup.files() == before


@pytest.mark.parametrize("name, scheme", [("authorization", "Bearer"), ("Authorization", "bearer")])
def test_proof_routes_take_the_api_key(setup, monkeypatch, name, scheme):
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    setup.ask_headers["authorization"] = "Bearer " + KEY
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    headers = {name: "%s %s" % (scheme, KEY)}
    assert call(app, "GET", "/plugins/proof/policies", headers)[0] == 200
    assert call(app, "GET", JOBS + proof_id + "/proof.bin", headers)[2] == b"dummy proof"
    assert call(app, "DELETE", JOBS + proof_id, headers)[0] == 200
    error_of(call(app, "GET", "/plugins/proof/no/such/route", headers), 404)


def test_call_without_the_api_key_cannot_test_a_policy_name(setup, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    setup.config["policies"]["shut"] = {**setup.entry, "spool_dir": setup.path("s"), "closed": True}
    app, body = setup.app(), json.dumps(FIELDS).encode()
    for key, expected in ((None, [401, 401, 401, 401]), (KEY, [200, 400, 400, 400])):
        headers = {"authorization": "Bearer " + key} if key else {}
        results = [
            call(app, method, "/v1/completions", {**headers, "x-proof-policy": name}, body)
            for method, name in (
                ("POST", "exact"),
                ("POST", "shut"),
                ("POST", "none"),
                ("GET", "exact"),
            )
        ]
        assert [status for status, _, _ in results] == expected
        assert key or len({body for _, _, body in results}) == 1 and setup.vllm.bodies == []


@pytest.mark.parametrize("key", [None, "", KEY])
def test_server_logs_an_error_at_startup_if_the_proof_routes_need_no_key(
    setup, monkeypatch, caplog, key
):
    if key is not None:
        monkeypatch.setenv("VLLM_API_KEY", key)
    app = setup.app()
    assert lifespan(app) == [] and caplog.text.count("need no API key") == (0 if key else 1)
    # As an error: at starts under vLLM 0.29.0 the line did not appear in the log as a warning.
    told = [record for record in caplog.records if "need no API key" in record.getMessage()]
    assert all(record.levelno == logging.ERROR for record in told)
    assert call(app, "GET", "/plugins/proof/policies")[0] == (401 if key else 200)


def test_api_key_need_not_be_utf8(setup, monkeypatch):
    monkeypatch.setitem(os.environb, b"VLLM_API_KEY", b"dummy \xff")
    app, sent = setup.app(), []

    async def send(message):
        sent.append(message.get("status"))

    for key in (b"dummy \xfe", b"dummy \xff"):
        scope = http_scope("GET", "/plugins/proof/policies")
        scope["headers"] = [(b"authorization", b"Bearer " + key)]
        asyncio.run(app(scope, stays_connected, send))
    assert sent == [401, None, 200, None]


# ---- Published files ----


def test_only_key_and_leaf_files_are_published(setup):
    setup.public_key_dir()
    with open(setup.path("public", "other.bin"), "w") as f:
        f.write("not published")
    app = setup.app()
    assert call(app, "GET", PUBLISHED + "exact/key.bin")[2] == b"dummy key.bin"
    assert call(app, "GET", PUBLISHED + "exact/leaf/0.bin")[2] == b"dummy leaf/0.bin"
    for path in (
        "exact/leaf/1.bin",
        "exact/other.bin",
        "exact/leaf/00.bin",
        "exact/leaf/../key.bin",
        "another/key.bin",
        "exact/",
    ):
        error_of(call(app, "GET", PUBLISHED + path), 404)


def test_published_file_that_others_can_write_is_not_served(setup, caplog):
    setup.public_key_dir()
    app = setup.app()
    os.chmod(setup.path("public", "key.bin"), 0o666)
    error_of(call(app, "GET", PUBLISHED + "exact/key.bin"), 500)
    assert "writable by other users" in caplog.text


def test_published_file_that_becomes_shorter_while_it_is_sent_breaks_the_response(setup, caplog):
    setup.public_key_dir()
    app = setup.app()

    async def send(message):
        if message["type"] == "http.response.start":
            os.truncate(setup.path("public", "key.bin"), 1)

    with pytest.raises(proof_middleware.ResponseCut):
        call(app, "GET", PUBLISHED + "exact/key.bin", send=send)
    assert "became shorter" in caplog.text


# ---- layout.txt ----


def write_layout(setup):
    with open(setup.path("layout.txt"), "wb") as f:
        f.write(b"dummy layout\n")


def test_layout_file_next_to_the_parameter_file_is_published(setup):
    setup.public_key_dir()
    write_layout(setup)
    open(setup.path("public", "notes.txt"), "w").close()
    app = setup.app()
    assert call(app, "GET", PUBLISHED + "exact/layout.txt")[::2] == (200, b"dummy layout\n")
    for path in (
        "exact/notes.txt",
        "exact/leaf/layout.txt",
        "exact/Layout.txt",
        "exact/layout.txt/",
        "exact/layout.bin",
        "exact/policy.txt",
        "another/layout.txt",
    ):
        error_of(call(app, "GET", PUBLISHED + path), 404)


def test_layout_file_is_not_served_if_it_is_missing(setup):
    setup.public_key_dir()
    error_of(call(setup.app(), "GET", PUBLISHED + "exact/layout.txt"), 404)


def turn_into_a_symlink(path):
    os.rename(path, path + ".real")
    os.symlink(path + ".real", path)


def make_writable_by_others(path):
    os.chmod(path, 0o666)


@pytest.mark.parametrize(
    "change, expected",
    [(turn_into_a_symlink, "symlink"), (make_writable_by_others, "writable by other users")],
)
def test_layout_file_that_cannot_be_trusted(setup, caplog, change, expected):
    setup.public_key_dir()
    write_layout(setup)
    app = setup.app()
    change(setup.path("layout.txt"))
    result = call(app, "GET", PUBLISHED + "exact/layout.txt")  # changed while the server runs
    assert error_of(result, 500) == proof_middleware.SERVER_FAULT and expected in caplog.text
    with pytest.raises(ConfigError, match="layout.txt.*" + expected):  # found at the next startup
        load_config(setup.path("config.json"))


def test_check_says_whether_the_layout_file_is_served(setup, capsys):
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    assert "; layout.txt not served\n" in capsys.readouterr().out
    write_layout(setup)
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    assert "; layout.txt served from %s\n" % setup.path("layout.txt") in capsys.readouterr().out


# ---- The check of a batch ----


def spoil_the_tag(setup, request_id):
    with open(setup.path("spool", "out", request_id + ".tag"), "r+b") as f:
        first = f.read(1)[0]
        f.seek(0)
        f.write(bytes([first ^ 1]))  # a fixed byte would be the right one once in 256 runs


def two_in_a_batch(setup, app, failed=False):
    """Return the proof IDs of two jobs, and their request IDs in the same order, of one batch."""
    ids = [setup.ask(app), setup.ask(app)]
    names = sorted(setup.respond())
    by_proof = {}
    for proof_id in ids:
        with open(setup.job_path(proof_id)) as f:
            by_proof[proof_id] = json.load(f)["request_id"]
    assert sorted(by_proof.values()) == names
    return ids, [by_proof[proof_id] for proof_id in ids]


def proof_path(setup, request_id):
    return setup.path("spool", "out", request_id + ".proof")


def test_nothing_of_a_batch_is_reported_until_all_its_proofs_have_passed(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    go, seen = threading.Event(), []

    def check(policy, job, y, path, jobs_dir):
        seen.append(job["request_id"])
        assert go.wait(10)
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["waiting", "waiting"]
    error_of(call(app, "GET", JOBS + ids[0] + "/proof.bin"), 404)
    go.set()
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["proved", "proved"]
    assert len(seen) == 2 and len(set(seen)) == 2


def test_a_failed_response_waits_for_the_proofs_of_its_batch(setup, monkeypatch):
    app = setup.app()
    proved, failed = setup.ask(app), setup.ask(app)
    names = setup.respond()
    with open(setup.job_path(failed)) as f:
        request_id = json.load(f)["request_id"]
    key = bytes.fromhex(json.loads(open(setup.job_path(failed)).read())["tag_key"])
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (request_id, suffix)))
    with open(setup.path("spool", "out", request_id + ".tag"), "wb") as f:
        f.write(spool.failed_tag(key, request_id))
    with open(setup.path("spool", "out", request_id + ".failed"), "wb"):
        pass
    assert request_id in names
    setup.set_one_time(request_id)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, failed)["state"] == "waiting"
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.PASSED)
    assert setup.state(app, failed)["state"] == "no_proof"
    assert setup.state(app, proved)["state"] == "proved"


def test_a_failed_response_costs_its_batch_nothing(setup, monkeypatch):
    app = setup.app()
    good, failed = setup.ask(app), setup.ask(app)
    with open(setup.job_path(failed)) as f:
        job = json.load(f)
    name, key = job["request_id"], bytes.fromhex(job["tag_key"])
    setup.respond()
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (name, suffix)))
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(spool.failed_tag(key, name))
    open(setup.path("spool", "out", name + ".failed"), "wb").close()
    setup.set_one_time(name)
    assert setup.state(app, good)["state"] == "proved"
    assert setup.state(app, failed)["state"] == "no_proof"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_a_batch_with_no_proof_needs_no_check(setup, monkeypatch):
    app = setup.app()
    proof_id = setup.ask(app)
    setup.respond(failed=True)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: 1 / 0)
    assert setup.state(app, proof_id)["state"] == "no_proof"


def test_a_rejected_proof_fails_its_batch_and_holds_the_policy(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    bad = requests[1]
    seen_at_deletion, delete = {}, spool.delete_response_at

    def note_and_delete(out, request_id):
        with open(proof_path(setup, request_id), "rb") as f, open(
            setup.path("spool", "out", request_id + ".y"), "rb"
        ) as g:
            seen_at_deletion[request_id] = f.read() + g.read()
        return delete(out, request_id)

    def check(policy, job, y, path, jobs_dir):
        return proof_middleware.REJECTED if job["request_id"] == bad else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    monkeypatch.setattr(spool, "delete_response_at", note_and_delete)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    for proof_id in ids:
        error_of(call(app, "GET", JOBS + proof_id), 503)
    assert sorted(seen_at_deletion) == sorted(requests)
    assert not any(name.endswith(".checked") for name in os.listdir(setup.path("jobs")))
    assert all(set(data) == {0} for data in seen_at_deletion.values())
    assert not any(name.endswith((".proof", ".y")) for name in os.listdir(setup.path("spool", "out")))
    assert "did not pass" in caplog.text
    assert not any(proof_id in caplog.text for proof_id in ids)
    # Neither a lifted hold nor a new process brings anything back.
    os.remove(setup.path("spool", "hold"))
    for one in (app, setup.app()):
        for proof_id in ids:
            state = json.loads(call(one, "GET", JOBS + proof_id)[2])
            assert state["state"] == "no_proof" and state["detail"] == proof_middleware.DECLINED


def test_the_rejected_proof_goes_last(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    order, replace = [], proof_middleware.replace_by_failed

    def note(out, request_id, job, tick_ns):
        assert os.path.exists(setup.path("spool", "hold"))
        order.append(request_id)
        replace(out, request_id, job, tick_ns)

    def check(policy, job, y, path, jobs_dir):
        return proof_middleware.REJECTED if job["request_id"] == requests[0] else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", note)
    call(app, "GET", JOBS + ids[0])
    assert order == [requests[1], requests[0]]


@pytest.mark.parametrize("outcome", [None, "anything else"])
def test_a_check_without_an_answer_leaves_the_batch_as_it_is(setup, monkeypatch, caplog, outcome):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    before = setup.files()
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: outcome)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert setup.files() == before and not os.path.exists(setup.path("spool", "hold"))
    assert "could not be checked" in caplog.text
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.PASSED)
    assert setup.state(app, ids[0])["state"] == "proved"


def test_a_fault_around_the_check_ends_in_a_hold_and_wipes_nothing(setup, monkeypatch, caplog):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    before = setup.files()
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert setup.state(app, ids[0])["state"] == "waiting"
    error_of(call(app, "GET", JOBS + ids[0]), 503)
    assert setup.files() == before and "on hold until a person has looked" in caplog.text


def test_a_proof_without_a_job_keeps_its_batch_from_passing(setup, monkeypatch, caplog):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    os.remove(setup.job_path(ids[1]))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert "1 without an answer" in caplog.text and "has no job" in caplog.text


def test_a_proof_with_a_wrong_tag_keeps_its_batch_from_passing(setup, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    spoil_the_tag(setup, requests[1])
    setup.set_one_time(requests[1])
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


def test_a_proof_with_a_fault_keeps_its_batch_from_passing(setup, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    y = setup.path("spool", "out", requests[1] + ".y")
    os.utime(y, ns=(7, 7))  # its files no longer have one time
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


@pytest.mark.parametrize("bad", [0, 1])
def test_a_rejection_beside_a_check_without_an_answer_still_counts(setup, monkeypatch, caplog, bad):
    setup.entry["checks_at_once"] = 1  # in turn, each order once
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)

    def check(policy, job, y, path, jobs_dir):
        return proof_middleware.REJECTED if job["request_id"] == requests[bad] else None

    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(name.endswith(".proof") for name in os.listdir(setup.path("spool", "out")))
    assert "did not pass" in caplog.text


def test_cp_verify_without_a_verdict_ends_in_a_hold_and_wipes_nothing(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert all(os.path.exists(proof_path(setup, name)) for name in requests)


def test_a_fault_around_the_check_is_told_once(setup, monkeypatch, caplog):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    for _ in range(3):
        call(app, "GET", JOBS + ids[0])
    assert caplog.text.count("A later call tries again") == 1
    assert caplog.text.count("on hold until a person has looked") == 1
    os.remove(setup.path("spool", "hold"))
    call(app, "GET", JOBS + ids[0])
    assert caplog.text.count("on hold until a person has looked") == 2  # each new hold is told


def test_a_response_replaced_by_a_failed_one_has_one_time_and_a_right_tag(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    for request_id in requests:
        times = {
            os.stat(setup.path("spool", "out", "%s.%s" % (request_id, suffix))).st_mtime_ns
            for suffix in ("tag", "failed")
        }
        assert times == {10**18}  # the batch's own time
    assert sorted(x.rsplit(".")[-1] for x in os.listdir(setup.path("spool", "out")) if x[0] != ".") == [
        "failed", "failed", "tag", "tag",
    ]  # fmt: skip


def test_a_response_that_comes_after_its_batch_passed_is_checked_itself(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    late = setup.ask(app)
    setup.respond()  # the same modification time as the first
    checked = []

    def check(policy, job, y, path, jobs_dir):
        checked.append(job["request_id"])
        return proof_middleware.REJECTED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, late)["state"] == "waiting"
    assert len(checked) == 1


def test_a_response_that_replaces_a_checked_one_is_checked_itself(setup, monkeypatch):
    app = setup.app()
    proof_id, path = setup.proved(app)
    with open(setup.job_path(proof_id)) as f:
        job = json.load(f)
    request_id, key, y = job["request_id"], bytes.fromhex(job["tag_key"]), bytes(4)
    os.remove(path)
    with open(path, "wb") as f:
        f.write(b"another proof")
    with open(setup.path("spool", "out", request_id + ".tag"), "wb") as f:
        f.write(spool.response_tag(key, request_id, y, b"another proof"))
    setup.set_one_time(request_id)  # the same time as the response that it replaces
    checked = []
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: checked.append(1))
    assert setup.state(app, proof_id)["state"] == "waiting" and checked


def test_a_malformed_member_ends_in_a_hold_and_wipes_nothing(setup, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    spoil_the_tag(setup, requests[1])
    setup.set_one_time(requests[1])
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert all(os.path.exists(proof_path(setup, name)) for name in requests)
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))
    assert "looks malformed" in caplog.text


def test_a_member_caught_between_two_renames_costs_one_try(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    path = proof_path(setup, requests[1])
    os.rename(path, path + ".aside")  # cp_prove has named its y and tag, not yet its proof
    assert setup.state(app, ids[0])["state"] == "waiting"
    os.rename(path + ".aside", path)
    assert setup.state(app, ids[0])["state"] == "proved"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_deletes_of_failed_jobs_beside_their_polls_lay_no_hold(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "RETRY_S", 0)
    setup.entry["max_waiting"] = 500
    app = setup.app()
    failed = [setup.ask(app) for _ in range(19)]
    setup.respond(failed=True)
    others = [setup.ask(app) for _ in range(5)]
    setup.respond()
    deadline = time.monotonic() + 20
    while setup.state(app, others[0])["state"] != "proved":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            call(app, "GET", JOBS + failed[0])

    poller = threading.Thread(target=poll)
    poller.start()
    for proof_id in failed[1:]:
        call(app, "DELETE", JOBS + proof_id)
    stop.set()
    poller.join()
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, others[-1])["state"] == "proved"


def test_deletes_beside_the_polls_of_a_failed_mate_lay_no_hold(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "RETRY_S", 0)
    setup.entry["max_waiting"] = 500
    app = setup.app()
    failed = setup.ask(app)
    setup.respond(failed=True)
    ids = [setup.ask(app) for _ in range(40)]
    setup.respond()
    deadline = time.monotonic() + 20
    while setup.state(app, ids[0])["state"] != "proved":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            call(app, "GET", JOBS + failed)

    pollers = [threading.Thread(target=poll) for _ in range(4)]
    for poller in pollers:
        poller.start()
    time.sleep(0.2)
    for proof_id in ids[:-5]:
        call(app, "DELETE", JOBS + proof_id)
    time.sleep(0.5)
    stop.set()
    for poller in pollers:
        poller.join()
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, ids[-1])["state"] == "proved"


def test_two_batches_side_by_side_are_judged_apart(setup, monkeypatch):
    app = setup.app()
    first = setup.ask(app)
    (name,) = setup.respond()
    os.utime(setup.path("spool", "out", name + ".tag"), ns=(7 * 10**17, 7 * 10**17))
    for suffix in ("y", "proof"):
        os.utime(setup.path("spool", "out", "%s.%s" % (name, suffix)), ns=(7 * 10**17, 7 * 10**17))
    second = setup.ask(app)
    setup.respond()

    def check(policy, job, y, path, jobs_dir):
        return proof_middleware.REJECTED if job["request_id"] == name else proof_middleware.PASSED

    assert setup.state(app, second)["state"] == "proved"
    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, first)["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    with open(setup.path("spool", "out", name + ".failed"), "rb"):
        pass


def test_a_record_of_another_file_opens_no_door(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    other = {"y": "00000000", "proof": [1, 2, 3, 4, 5], "verified": True}

    def lay_a_record_of_another_file(*_):
        with open(setup.job_path(proof_id, ".checked"), "w") as f:
            json.dump(other, f)
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", lay_a_record_of_another_file)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] == 404


def test_a_clean_leaves_no_proof_without_its_job(setup, capsys):
    app = setup.app()
    proof_id, path = setup.proved(app)
    os.utime(setup.job_path(proof_id), (0, 0))  # the job is old, its response is not
    os.utime(setup.job_path(proof_id, ".checked"), (0, 0))
    for suffix in ("proof", "y", "tag"):
        now = time.time()
        os.utime(path.removesuffix("proof") + suffix, (now, now))
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 0
    assert os.path.exists(setup.job_path(proof_id))


def test_a_failed_job_does_not_wait_for_a_proof_that_has_no_job(setup, monkeypatch):
    app = setup.app()
    failed, orphan = setup.ask(app), setup.ask(app)
    with open(setup.job_path(failed)) as f:
        job = json.load(f)
    name, key = job["request_id"], bytes.fromhex(job["tag_key"])
    setup.respond()
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (name, suffix)))
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(spool.failed_tag(key, name))
    open(setup.path("spool", "out", name + ".failed"), "wb").close()
    setup.set_one_time(name)
    os.remove(setup.job_path(orphan))
    for _ in range(3):
        assert setup.state(app, failed)["state"] == "no_proof"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_a_failed_jobs_poll_reads_no_proof(setup, monkeypatch):
    app = setup.app()
    ids = [setup.ask(app), setup.ask(app)]
    names = setup.respond()
    with open(setup.job_path(ids[0])) as f:
        job = json.load(f)
    name, key = job["request_id"], bytes.fromhex(job["tag_key"])
    other = [one for one in names if one != name][0]
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (name, suffix)))
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(spool.failed_tag(key, name))
    open(setup.path("spool", "out", name + ".failed"), "wb").close()
    setup.set_one_time(name)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "RETRY_S", 1000)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    monkeypatch.setattr(proof_middleware.BatchChecks, "ask", lambda *_: None)
    lengths = count_reads(monkeypatch, proof_path(setup, other))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert lengths == []


@pytest.mark.parametrize("first", [0, 1])
def test_a_response_with_two_times_holds_its_mates_back_whoever_polls_first(setup, first):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.utime(proof_path(setup, requests[1]), ns=(7, 7))
    assert setup.state(app, ids[first])["state"] == "waiting"
    assert setup.state(app, ids[first])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(name.endswith(".checked") for name in os.listdir(setup.path("jobs")))


def test_the_child_gets_none_of_the_servers_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "QZXJKV")
    code, printed = proof_middleware.run_to_the_end(["/bin/sh", "-c", "env"], str(tmp_path))
    assert code == 0 and b"QZXJKV" not in printed and b"TMPDIR=" in printed


def test_a_mate_with_another_time_on_its_proof_alone_is_a_member(setup):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.utime(proof_path(setup, requests[1]), ns=(7, 7))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


def test_a_response_on_its_way_out_is_no_malformed_member(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    read = proof_middleware.BatchChecks.jobs_of

    def job_read_just_before_a_delete(jobs, policy):
        found = read(jobs, policy)
        os.remove(setup.job_path(ids[1]))
        os.remove(setup.path("spool", "out", requests[1] + ".y"))
        monkeypatch.setattr(proof_middleware.BatchChecks, "jobs_of", staticmethod(read))
        return found

    monkeypatch.setattr(
        proof_middleware.BatchChecks, "jobs_of", staticmethod(job_read_just_before_a_delete)
    )
    assert setup.state(app, ids[0])["state"] == "proved"
    assert not os.path.exists(setup.path("spool", "hold"))
    assert "response %s got no verdict" % requests[1] in caplog.text


def test_a_response_whose_request_is_not_final_is_no_answer(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.makedirs(setup.path("spool", "work"), exist_ok=True)
    open(setup.path("spool", "work", requests[1] + ".req"), "w").close()
    os.remove(proof_path(setup, requests[1]))  # the next start is taking it back
    for _ in range(3):  # no try is counted
        assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))
    assert all(not x.endswith(".checked") for x in os.listdir(setup.path("jobs")))


def test_a_childs_descriptor_does_not_hold_the_lock(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    monkeypatch.setattr(proof_middleware, "MEMORY_DIR", setup.path("not there"))
    monkeypatch.setattr(proof_middleware, "run_check", RUN_THE_CHECK)
    seen, ran = [], []

    def child(words, work, fds):
        below = int(work.split("/")[4])  # the folder lies below this descriptor
        ran.append(below in fds and int(words[4].split("/")[4]) in fds)  # so does the proof
        with contextlib.suppress(BlockingIOError):  # the check holds the lock on another open
            fcntl.flock(below, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append("holds the lock")
        return 2, b""

    monkeypatch.setattr(proof_middleware, "run_to_the_end", child)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert ran == [True] and seen == []


def test_a_wipe_goes_past_a_response_that_sticks_to_the_rejected_one(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    replace, done = proof_middleware.replace_by_failed, []

    def first_sticks(out, request_id, job, tick_ns):
        done.append(request_id)
        if len(done) == 1:
            raise OSError("it sticks")
        replace(out, request_id, job, tick_ns)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", first_sticks)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert len(done) == 2 and "is not done for these" in caplog.text and done[0] in caplog.text


def test_a_try_that_passes_sets_the_count_back(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, ids[0])["state"] == "waiting"
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.PASSED)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, late)["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_a_wipe_that_cannot_zero_a_file_says_so_and_the_note_stays(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)

    def no_zeros(*_):
        raise OSError(errno.EIO, "no")

    monkeypatch.setattr(proof_middleware, "overwrite", no_zeros)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert "is not done for these responses" in caplog.text
    assert os.path.exists(setup.path("jobs", "withheld.exact.%d.json" % 10**18))
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_the_hold_asks_for_the_same_look_at_both_tries(setup):
    app = setup.app()
    ids = [setup.ask(app) for _ in range(3)]
    names = setup.respond()
    paths = [proof_path(setup, name) for name in names[1:]]
    os.rename(paths[0], paths[0] + ".aside")
    assert setup.state(app, ids[0])["state"] == "waiting"
    os.rename(paths[0] + ".aside", paths[0])
    os.rename(paths[1], paths[1] + ".aside")
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_two_tries_without_an_answer_hold_whichever_check_it_was(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    turn = itertools.count()

    def in_turns(policy, job, *_):
        failing = requests[next(turn) // 2 % 2]
        return None if job["request_id"] == failing else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", in_turns)
    for _ in range(2):
        assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_death_behind_the_note_serves_nothing_and_the_next_start_holds(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()

    def dies(_):
        raise SystemExit("it dies here")

    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "lay_hold", dies)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    app.batches.ask(app.config.policies["exact"], 10**18)
    app.batches.threads[("exact", 10**18)].join(30)
    assert not os.path.exists(setup.path("spool", "hold"))
    assert call(app, "GET", JOBS + ids[0] + "/proof.bin")[0] != 200
    monkeypatch.undo()
    monkeypatch.setattr(proof_middleware, "plan_id", lambda policy: policy.commitment_id)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    again = setup.app()
    deadline = time.monotonic() + 10
    while any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out"))):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert os.path.exists(setup.path("spool", "hold"))
    assert "goes on, behind a hold" in caplog.text
    del again


def test_a_note_names_nothing_but_request_ids_below_out(setup):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    outside = setup.path("spool", "keep.proof")
    with open(outside, "wb") as f:
        f.write(b"keep")
    note = {"policy": "exact", "time": 10**18, "proofs": ["../keep"], "rejected": ["../keep"]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    with open(outside, "rb") as f:
        assert f.read() == b"keep"


def test_a_finishing_without_a_hold_does_not_say_that_it_has_one(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)

    def no_room(_):
        raise OSError(errno.ENOSPC, "no room")

    monkeypatch.setattr(proof_middleware, "lay_hold", no_room)
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    assert "the wipe of the batch" in caplog.text and "behind a hold" not in caplog.text


def test_a_record_that_cannot_be_unlinked_is_named_and_stops_no_wipe(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    unlink = os.unlink

    def stuck(name, **arguments):
        if str(name).endswith(".checked"):
            raise OSError(errno.EIO, "no")
        unlink(name, **arguments)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(proof_middleware.os, "unlink", stuck)
    call(app, "GET", JOBS + late)
    assert "is not done for these responses" in caplog.text
    assert requests[0] in caplog.text


def test_a_check_without_an_answer_is_named_by_its_request(tmp_path, setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert "no answer for response " + requests[0] in caplog.text
    assert "remove out/%s.*" % requests[0] in caplog.text


def test_a_pass_clears_the_tries_of_any_cause(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "MOST_TRIES", 2)
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    path = proof_path(setup, requests[1])
    os.rename(path, path + ".aside")
    assert setup.state(app, ids[0])["state"] == "waiting"
    os.rename(path + ".aside", path)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    (name,) = setup.respond()
    os.remove(setup.path("spool", "out", name + ".y"))
    assert setup.state(app, late)["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_the_cap_is_ten_tries():
    assert proof_middleware.MOST_TRIES == 10


def test_the_note_comes_before_the_hold(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    hold, seen = proof_middleware.lay_hold, []

    def look(top):
        seen.append(os.path.exists(setup.path("jobs", "withheld.exact.%d.json" % 10**18)))
        hold(top)

    monkeypatch.setattr(proof_middleware, "lay_hold", look)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    setup.state(app, proof_id)
    assert seen[0] is True


def test_a_note_that_cannot_be_read_back_counts_as_not_written(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    read, hold, seen, fault = proof_middleware.read_file, proof_middleware.lay_hold, [], []

    def no_read(directory, name):
        if fault and name.startswith("withheld."):
            raise OSError(errno.EIO, "no")
        return read(directory, name)

    def rejected_as_the_fault_sets_in(*_):
        fault.append(1)
        return proof_middleware.REJECTED

    def look(top):
        seen.append(os.path.exists(setup.job_path(ids[0], ".checked")))
        hold(top)

    monkeypatch.setattr(proof_middleware, "read_file", no_read)
    monkeypatch.setattr(proof_middleware, "lay_hold", look)
    monkeypatch.setattr(proof_middleware, "run_check", rejected_as_the_fault_sets_in)
    call(app, "GET", JOBS + late)
    assert seen and not seen[0]  # the records have gone, and the hold lies


def test_a_note_that_stands_but_cannot_be_read_stops_a_record_and_ends_in_a_hold(
    setup, monkeypatch, caplog
):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)
    read = proof_middleware.read_file

    def no_read(directory, name):
        if name.startswith("withheld."):
            raise OSError(errno.EIO, "no")
        return read(directory, name)

    monkeypatch.setattr(proof_middleware, "read_file", no_read)
    error_of(call(app, "GET", JOBS + ids[0]), 503)  # the record stands; it does not count
    assert call(app, "GET", JOBS + ids[0] + "/proof.bin")[0] != 200
    assert call(app, "DELETE", JOBS + ids[1])[0] == 503
    assert os.path.exists(proof_path(setup, requests[0]))  # nothing is decided on it
    assert "can't be read or is not whole. Do not remove it" in caplog.text
    monkeypatch.setattr(proof_middleware, "read_file", read)
    os.remove(setup.path("spool", "hold"))
    call(app, "GET", JOBS + ids[0])
    assert not os.path.exists(proof_path(setup, requests[0]))  # it reads again: the wipe goes on


def raising_on_the_road_of_a_no(road, number, monkeypatch):
    """Make what is on the road from a NO to its hold raise a fault on the note's name."""

    def fault(*_, **__):
        raise OSError(number, os.strerror(number))

    def on_the_note(real):
        def wrapped(directory, name, *rest, **keywords):
            if name.startswith("withheld."):
                fault()
            return real(directory, name, *rest, **keywords)

        return wrapped

    if road == "write_note itself":
        monkeypatch.setattr(proof_middleware.BatchChecks, "write_note", staticmethod(fault))
    for call_name in road.split(" and "):
        if hasattr(proof_middleware, call_name):
            real = getattr(proof_middleware, call_name)
            monkeypatch.setattr(proof_middleware, call_name, on_the_note(real))


@pytest.mark.parametrize("number", [errno.EIO, errno.EMFILE, errno.EACCES, errno.ENOSPC])
@pytest.mark.parametrize(
    "road", ["write_file", "read_file", "write_file and read_file", "write_note itself"]
)
def test_whatever_befalls_the_note_no_mate_goes_out_and_a_hold_lies(
    setup, monkeypatch, caplog, road, number
):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    armed = []

    def rejected(*_):
        if not armed:
            armed.append(1)
            raising_on_the_road_of_a_no(road, number, monkeypatch)
        return proof_middleware.REJECTED

    monkeypatch.setattr(proof_middleware, "run_check", rejected)
    call(app, "GET", JOBS + late)
    assert os.path.exists(setup.path("spool", "hold"))
    assert all(not os.path.exists(setup.job_path(proof_id, ".checked")) for proof_id in ids)
    if road == "write_note itself":
        assert "could not be written and read back" in caplog.text
    for _ in range(2):  # meanwhile, and behind a lift
        for proof_id in ids:
            assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] != 200
            found = call(app, "GET", JOBS + proof_id)
            assert found[0] != 200 or json.loads(found[2])["state"] != "proved"
        monkeypatch.undo()
        monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
        monkeypatch.setattr(proof_middleware.BatchChecks, "ask", lambda *_: None)
        with contextlib.suppress(FileNotFoundError):
            os.remove(setup.path("spool", "hold"))


def test_a_finishing_closes_no_note_on_a_list_it_made_up(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    name = "withheld.exact.%d.json" % 10**18
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", name), "w") as f:
        json.dump(note, f)
    read, reads = proof_middleware.read_file, []

    def second_read_fails(directory, file):
        if file == name:
            reads.append(1)
            if len(reads) > 2:  # read_note reads a note twice
                raise OSError(errno.EIO, "no")
        return read(directory, file)

    def no_members(*_, **__):
        raise OSError(errno.EIO, "no")

    monkeypatch.setattr(proof_middleware, "read_file", second_read_fails)
    monkeypatch.setattr(proof_middleware.BatchChecks, "members", no_members)
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    assert all(not os.path.exists(setup.job_path(proof_id, ".checked")) for proof_id in ids)


def test_a_note_that_cannot_be_read_outranks_a_record_where_the_gate_was_passed(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w").close()
    monkeypatch.setattr(proof_middleware, "is_held", lambda *_, **__: False)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware.BatchChecks, "ask", lambda *_: None)
    assert setup.state(app, proof_id)["state"] == "waiting"


def test_a_fresh_wipe_closes_no_note_that_it_cannot_read_back(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    read, armed = proof_middleware.read_file, []

    def no_read(directory, name):
        if armed and name.startswith("withheld."):
            raise OSError(errno.EIO, "no")
        return read(directory, name)

    def rejected(*_):
        armed.append(1)
        return proof_middleware.REJECTED

    monkeypatch.setattr(proof_middleware, "read_file", no_read)
    monkeypatch.setattr(proof_middleware, "run_check", rejected)
    call(app, "GET", JOBS + proof_id)
    assert os.path.exists(setup.path("jobs", "withheld.exact.%d.json" % 10**18))


def test_a_finishing_lays_its_hold_before_it_reads_the_batch(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)
    members, seen = proof_middleware.BatchChecks.members, []

    def look(*arguments, **keywords):
        seen.append(os.path.exists(setup.path("spool", "hold")))
        return members(*arguments, **keywords)

    monkeypatch.setattr(proof_middleware.BatchChecks, "members", staticmethod(look))
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    assert seen and seen[0]


def test_a_note_outranks_a_record_by_its_request_where_the_gate_was_passed(setup, monkeypatch):
    app = setup.app()
    proof_id, path = setup.proved(app)
    with open(setup.job_path(proof_id)) as f:
        name = json.load(f)["request_id"]
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump({"policy": "exact", "time": 10**18, "proofs": [name], "rejected": []}, f)
    monkeypatch.setattr(proof_middleware, "is_held", lambda *_: False)  # a note laid just behind it
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware.BatchChecks, "ask", lambda *_: None)
    assert setup.state(app, proof_id)["state"] == "waiting"


def test_a_failed_job_waits_for_a_proof_whose_job_cannot_be_read(setup, monkeypatch):
    app = setup.app()
    failed = setup.ask(app)
    setup.respond(failed=True)
    other = setup.ask(app)
    setup.respond()
    read, name = proof_middleware.read_json, proof_middleware.job_file(other)

    def not_that_one(directory, file):
        if file == name:
            raise OSError(errno.EMFILE, "no")
        return read(directory, file)

    monkeypatch.setattr(proof_middleware, "read_json", not_that_one)
    assert setup.state(app, failed)["state"] == "waiting"


def test_a_hold_is_tried_though_a_response_of_the_wipe_sticks(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    hold, tries = proof_middleware.lay_hold, []

    def not_at_first(top):
        tries.append(1)
        if len(tries) <= 2:
            raise OSError(errno.ENOSPC, "no room")
        hold(top)

    def sticks(*_):
        raise OSError(errno.EIO, "no")

    monkeypatch.setattr(proof_middleware, "lay_hold", not_at_first)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", sticks)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    call(app, "GET", JOBS + ids[0])
    assert os.path.exists(setup.path("spool", "hold"))


def test_a_job_that_cannot_be_read_is_no_job_that_is_gone(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    read, name = proof_middleware.read_json, proof_middleware.job_file(ids[1])

    def not_that_one(directory, file):
        if file == name:
            raise OSError(errno.EMFILE, "no")
        return read(directory, file)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    note = setup.path("jobs", "withheld.exact.%d.json" % 10**18)
    with open(note, "w") as f:
        json.dump({"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}, f)
    monkeypatch.setattr(proof_middleware, "read_json", not_that_one)
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    assert os.path.exists(note)  # its client still has its failed response to come
    monkeypatch.setattr(proof_middleware, "read_json", read)
    proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
    os.remove(setup.path("spool", "hold"))
    assert setup.state(app, ids[1])["state"] == "no_proof"


def test_a_server_fault_is_told_in_the_systems_words(setup, monkeypatch, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)

    def full(*_):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(proof_middleware.BatchChecks, "sweep", full)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert "Too many open files" in caplog.text


def answers_to_a_client(setup, app, proof_id):
    """What each of a client's calls answers now: for a job, and for two new requests."""
    found = [call(app, "GET", JOBS + proof_id), call(app, "DELETE", JOBS + proof_id)]
    for name in ("proof.bin", "y.bin"):
        found.append(call(app, "GET", JOBS + proof_id + "/" + name))
    for extra in ({}, {"temperature": 1}):  # one that has a reason of its own
        found.append(json.loads(call(app, "GET", JOBS + setup.ask(app, **extra))[2]))
    return found


@pytest.mark.parametrize("state", ["a NO's hold with its note done", "a note and a hold", "a note alone"])
def test_a_client_cannot_tell_what_stopped_a_policy(setup, monkeypatch, state):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    monkeypatch.setattr(proof_middleware.BatchChecks, "ask", lambda *_: None)
    hold = setup.path("spool", "hold")
    open(hold, "w").close()
    by_an_operator = answers_to_a_client(setup, app, proof_id)
    os.remove(hold)
    note = setup.path("jobs", "withheld.exact.1.json")
    if state != "a note alone":
        open(hold, "w").close()
    if state == "a NO's hold with its note done":
        note = setup.path("jobs", "withheld.exact.1.done.json")
    with open(note, "w") as f:
        json.dump({"policy": "exact", "time": 1, "proofs": [], "rejected": []}, f)
    assert answers_to_a_client(setup, app, proof_id) == by_an_operator
    assert "/" not in by_an_operator[-2]["detail"]  # none of the provider's words


@pytest.mark.parametrize("text", ["", "not json", '{"policy": "exact"}'])
def test_a_note_that_is_not_whole_counts_as_unreadable(setup, text):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        f.write(text)
    error_of(call(app, "GET", JOBS + proof_id), 503)
    setup.app()  # a new server starts all the same


def test_without_room_for_the_note_the_records_go_at_once(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    write, seen = proof_middleware.write_file, []

    def no_room(jobs, name, data):
        if name.startswith("withheld."):
            raise OSError(errno.ENOSPC, "no room")
        write(jobs, name, data)

    hold = proof_middleware.lay_hold

    def look(top):
        seen.append(os.path.exists(setup.job_path(ids[0], ".checked")))
        hold(top)

    monkeypatch.setattr(proof_middleware, "write_file", no_room)
    monkeypatch.setattr(proof_middleware, "lay_hold", look)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    setup.state(app, late)
    assert seen and not seen[0]


def test_y_and_tag_alone_polled_by_their_own_client_end_in_a_hold(setup):
    app = setup.app()
    proof_id, path = setup.proved(app, poll=False)
    os.remove(path)
    for _ in range(2):
        assert setup.state(app, proof_id)["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))


def test_a_request_that_stays_in_work_is_told_of_at_last(setup, monkeypatch, caplog):
    monkeypatch.setattr(proof_middleware, "NOT_FINAL_TRIES", 2)
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.makedirs(setup.path("spool", "work"), exist_ok=True)
    open(setup.path("spool", "work", requests[1] + ".req"), "w").close()
    for _ in range(3):
        assert setup.state(app, ids[0])["state"] == "waiting"
    assert "stay in in/ or work/" in caplog.text


def test_the_note_stands_while_the_other_checks_of_the_batch_run(setup, monkeypatch):
    setup.entry["checks_at_once"] = 2
    app = setup.app()
    ids = batch_of(setup, app, 2)
    go, both, lock, seen = threading.Event(), threading.Barrier(2), threading.Lock(), []
    stood, seen_note = threading.Event(), []

    def check(policy, job, *_):
        with lock:
            seen.append(1)
            first = len(seen) == 1
        both.wait(30)
        if first:
            return proof_middleware.REJECTED
        deadline = time.monotonic() + 20  # the first ends and tells; this one is still running
        hold = setup.path("spool", "hold")
        while not os.path.exists(hold) and time.monotonic() < deadline:
            time.sleep(0.01)
        seen_note.append(os.path.exists(note) and os.path.exists(hold))
        stood.set()
        go.wait(30)
        return proof_middleware.PASSED

    note = setup.path("jobs", "withheld.exact.%d.json" % 10**18)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "run_check", check)
    setup.state(app, ids[0])
    assert stood.wait(30) and seen_note == [True]
    go.set()
    app.batches.threads[("exact", 10**18)].join(30)


def test_the_way_out_for_a_proof_without_a_job_is_told(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.remove(setup.job_path(ids[1]))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert "remove out/%s.* or put its job back" % requests[1] in caplog.text


def test_a_hold_that_cannot_be_laid_at_a_fault_is_not_told_as_laid(setup, monkeypatch, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)

    def no_room(_):
        raise OSError(errno.ENOSPC, "no room")

    monkeypatch.setattr(proof_middleware, "lay_hold", no_room)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    for _ in range(2):
        assert setup.state(app, proof_id)["state"] == "waiting"
    assert "the hold could not be laid" in caplog.text
    assert "on hold until a person has looked" not in caplog.text


def test_a_job_file_that_cannot_be_read_stops_no_wipe(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    read, name = proof_middleware.read_json, proof_middleware.job_file(ids[1])

    def unreadable(directory, file):
        if file == name:
            raise OSError(errno.EIO, "no")
        return read(directory, file)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(proof_middleware, "read_json", unreadable)
    setup.state(app, ids[0])
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_the_finishing_of_a_wipe_needs_no_listing_of_its_members(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)

    def breaks(*_, **__):
        raise OSError(errno.EIO, "no")

    monkeypatch.setattr(proof_middleware.BatchChecks, "members", breaks)
    checks = proof_middleware.BatchChecks(app.config)
    checks.run(app.config.policies["exact"], 10**18)
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_no_room_through_a_whole_wipe_and_then_room_again(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)

    def no_room(*_, **__):
        raise OSError(errno.ENOSPC, "no room")

    with monkeypatch.context() as full:
        full.setattr(proof_middleware, "lay_failed", no_room)
        full.setattr(proof_middleware, "write_file", no_room)
        full.setattr(proof_middleware, "lay_hold", no_room)
        full.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
        call(app, "GET", JOBS + ids[0])
        error_of(call(app, "GET", JOBS + ids[1]), 503)
    app.batches.not_before.clear()
    error_of(call(app, "GET", JOBS + ids[0]), 503)
    assert os.path.exists(setup.path("spool", "hold"))
    os.remove(setup.path("spool", "hold"))
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["no_proof"] * 2


def test_a_clean_leaves_a_note_whose_wipe_is_not_done(setup):
    app = setup.app()
    note = setup.path("jobs", "withheld.exact.%d.json" % 10**18)
    with open(note, "w") as f:
        json.dump({"policy": "exact", "time": 10**18, "proofs": [], "rejected": []}, f)
    os.utime(note, (0, 0))
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 0
    assert os.path.exists(note)


def test_a_note_stays_until_a_hold_lies_and_stops_the_policy_meanwhile(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    hold = proof_middleware.lay_hold

    def no_room(top):
        raise OSError(errno.ENOSPC, "no room")

    monkeypatch.setattr(proof_middleware, "lay_hold", no_room)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    note = setup.path("jobs", "withheld.exact.%d.json" % 10**18)
    assert os.path.exists(note)
    error_of(call(app, "GET", JOBS + ids[1]), 503)  # not no_proof: nobody has looked yet
    waiting = setup.requests()
    new = setup.ask(app)
    assert json.loads(call(app, "GET", JOBS + new)[2]).get("state") != "waiting"
    assert setup.requests() == waiting  # a new request gets no proof
    monkeypatch.setattr(proof_middleware, "lay_hold", hold)
    call(app, "GET", JOBS + ids[0])
    assert os.path.exists(setup.path("spool", "hold")) and not os.path.exists(note)


@pytest.mark.parametrize("folder", ["in", "work"])
def test_a_request_in_either_folder_is_not_final(setup, folder):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.makedirs(setup.path("spool", folder), 0o700, exist_ok=True)
    open(setup.path("spool", folder, requests[1] + ".req"), "w").close()
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


def test_a_finishing_takes_the_rejected_proof_last_by_its_note(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    for last in requests:
        note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": [last]}
        order, replace = [], proof_middleware.replace_by_failed

        def noting(out, request_id, job, tick_ns):
            assert os.path.exists(setup.path("spool", "hold"))  # the hold comes before the wipe
            order.append(request_id)

        monkeypatch.setattr(proof_middleware, "replace_by_failed", noting)
        name = setup.path("jobs", "withheld.exact.%d.json" % 10**18)
        with open(name, "w") as f:
            json.dump(note, f)
        proof_middleware.BatchChecks(app.config).run(app.config.policies["exact"], 10**18)
        assert len(order) == 2 and order[-1] == last
        done = name.removesuffix("json") + "done.json"
        if os.path.exists(done):
            os.remove(done)
        monkeypatch.setattr(proof_middleware, "replace_by_failed", replace)
        os.remove(setup.path("spool", "hold"))


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_the_hold_comes_though_the_records_raise_what_no_one_expects(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    unlink = os.unlink

    def no_memory(name, **arguments):
        if str(name).endswith(".checked"):
            raise MemoryError
        unlink(name, **arguments)

    def note_gone(*_):
        raise MemoryError

    monkeypatch.setattr(proof_middleware.BatchChecks, "write_note", staticmethod(note_gone))
    monkeypatch.setattr(proof_middleware.os, "unlink", no_memory)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    call(app, "GET", JOBS + late)
    assert os.path.exists(setup.path("spool", "hold"))


def nothing_can_be_written(monkeypatch):
    def read_only(*_, **__):
        raise OSError(errno.EROFS, "Read-only file system")

    unlink = os.unlink

    def no_unlink(name, **arguments):
        read_only()

    monkeypatch.setattr(proof_middleware, "write_file", read_only)
    monkeypatch.setattr(proof_middleware, "lay_hold", read_only)
    monkeypatch.setattr(proof_middleware, "overwrite", read_only)
    monkeypatch.setattr(proof_middleware.os, "unlink", no_unlink)


def test_a_disk_that_turns_read_only_at_a_no_stops_the_policy_in_memory(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    with monkeypatch.context() as fault:
        armed = []

        def rejected(*_):
            if not armed:
                armed.append(1)
                nothing_can_be_written(fault)
            return proof_middleware.REJECTED

        fault.setattr(proof_middleware, "run_check", rejected)
        call(app, "GET", JOBS + late)
        for proof_id in ids:
            error_of(call(app, "GET", JOBS + proof_id), 503)
            assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] == 503
        assert "Only this process's memory is sure to stop the policy" in caplog.text
    # The disk is healed: the mark finishes what it knows, and goes.
    app.batches.not_before.clear()
    error_of(call(app, "GET", JOBS + ids[0]), 503)
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    assert app.batches.remembered == {}
    os.remove(setup.path("spool", "hold"))
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["no_proof"] * 2


def test_a_no_once_seen_stands_though_the_verifier_would_pass_on_a_healed_disk(setup, monkeypatch):
    app = setup.app()
    ids = [setup.ask(app) for _ in range(3)]
    setup.respond()
    asked = []

    def no_once_then_yes(policy, job, *_):
        asked.append(job["request_id"])
        return proof_middleware.PASSED

    with monkeypatch.context() as fault:
        armed = []

        def first_a_no(*_):
            if not armed:
                armed.append(1)
                nothing_can_be_written(fault)
                return proof_middleware.REJECTED
            return proof_middleware.PASSED

        fault.setattr(proof_middleware, "run_check", first_a_no)
        call(app, "GET", JOBS + ids[0])
    monkeypatch.setattr(proof_middleware, "run_check", no_once_then_yes)
    app.batches.not_before.clear()
    error_of(call(app, "GET", JOBS + ids[1]), 503)
    assert asked == []  # the verifier isn't asked again
    assert os.path.exists(setup.path("spool", "hold"))
    assert app.batches.remembered == {}
    os.remove(setup.path("spool", "hold"))
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["no_proof"] * 3
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_a_jobs_directory_that_fails_for_a_while_at_a_no_serves_nothing_behind_a_lift(
    setup, monkeypatch
):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()

    def fault(*_, **__):
        raise OSError(errno.EIO, "no")

    with monkeypatch.context() as failing:
        armed = []

        def rejected(*_):
            if not armed:
                armed.append(1)
                failing.setattr(proof_middleware, "write_file", fault)
                failing.setattr(proof_middleware, "read_file", fault)
                failing.setattr(proof_middleware.BatchChecks, "jobs_of", staticmethod(fault))
            return proof_middleware.REJECTED

        failing.setattr(proof_middleware, "run_check", rejected)
        call(app, "GET", JOBS + late)
    assert os.path.exists(setup.path("spool", "hold"))
    app.batches.not_before.clear()
    error_of(call(app, "GET", JOBS + ids[0]), 503)  # the fault has passed: the wipe goes on
    os.remove(setup.path("spool", "hold"))
    for proof_id in ids:
        assert setup.state(app, proof_id)["state"] == "no_proof"
        assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] != 200


def test_a_mark_goes_only_with_the_wipe_not_with_the_hold(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    sticks = []

    def stuck(*_):
        sticks.append(1)
        raise OSError(errno.EIO, "no")

    monkeypatch.setattr(proof_middleware, "replace_by_failed", stuck)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    call(app, "GET", JOBS + ids[0])
    assert sticks and os.path.exists(setup.path("spool", "hold"))
    assert app.batches.remembered != {}


def test_a_hold_that_cannot_be_laid_is_told(setup, monkeypatch, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)

    def no_room(_):
        raise OSError(errno.ENOSPC, "no room")

    monkeypatch.setattr(proof_middleware, "lay_hold", no_room)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert "the hold could not be laid" in caplog.text
    assert "on hold" not in caplog.text


def test_a_mates_stale_record_goes_at_the_check(setup, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    for name in requests:
        os.chmod(proof_path(setup, name), 0o400)  # the identity has its times: as a restore does
    for _ in range(2):
        setup.state(app, ids[0])
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, ids[1])["state"] == "proved"
    assert "stands in the way" not in caplog.text


def test_a_clean_keeps_a_job_whose_request_is_in_work(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    with open(setup.job_path(proof_id)) as f:
        request_id = json.load(f)["request_id"]
    os.makedirs(setup.path("spool", "work"), 0o700, exist_ok=True)
    os.chmod(setup.path("spool", "work"), 0o700)
    open(setup.path("spool", "work", request_id + ".req"), "w").close()
    os.utime(setup.job_path(proof_id), (0, 0))
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 0
    assert os.path.exists(setup.job_path(proof_id))


def test_a_finished_wipe_turns_its_note(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    setup.state(app, proof_id)
    assert not os.path.exists(setup.path("jobs", "withheld.exact.%d.json" % 10**18))
    assert os.path.exists(setup.path("jobs", "withheld.exact.%d.done.json" % 10**18))


def test_a_check_without_an_answer_counts_for_nothing_if_the_job_has_gone(setup, monkeypatch, caplog):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)

    def taken_away(policy, job, *_):
        if job["request_id"] == requests[1]:
            os.remove(setup.job_path(ids[1]))
            return None
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", taken_away)
    assert setup.state(app, ids[0])["state"] == "proved"
    assert "response %s got no verdict" % requests[1] in caplog.text


@pytest.mark.parametrize(
    "last, name",
    [
        ("check_proof: error: the verifier /v does not exist or is not executable", "the verifier is missing"),
        ("check_proof: error: /c/layout.txt does not exist or is not a regular file", "a file is missing"),
        ("check_proof: error: layout.txt is missing and could not be fetched: x", "a file is missing"),
        ("check_proof: error: cp_verify exited with status 5 and the last line", "NOT CHECKED"),
        ("check_proof: rejected: the public key directory and configuration files", "another commitment ID"),
    ],
)  # fmt: skip
def test_each_end_without_an_answer_has_its_name_in_the_log(tmp_path, monkeypatch, caplog, last, name):
    monkeypatch.setattr(proof_middleware, "run_to_the_end", lambda *_: (2, (last + "\n").encode()))
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    assert RUN_CHECK(policy, job, bytes(4), __file__, str(tmp_path)) is None
    assert name in caplog.text


def test_the_records_go_before_any_proof(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    assert setup.state(app, ids[0])["state"] == "proved"
    late = setup.ask(app)
    setup.respond()
    replace, seen = proof_middleware.replace_by_failed, []

    def note(out, request_id, job, tick_ns):
        seen.append(any(x.endswith(".checked") for x in os.listdir(setup.path("jobs"))))
        replace(out, request_id, job, tick_ns)

    monkeypatch.setattr(proof_middleware, "replace_by_failed", note)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    setup.state(app, late)
    assert seen and not any(seen)


def test_looks_that_take_turns_end_in_a_hold_at_last(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "MOST_TRIES", 4)
    app = setup.app()
    ids = [setup.ask(app) for _ in range(3)]
    names = setup.respond()
    paths = [proof_path(setup, name) for name in names[1:]]
    os.rename(paths[0], paths[0] + ".aside")
    for turn in range(3):
        assert setup.state(app, ids[0])["state"] == "waiting"
        os.rename(paths[turn % 2] + ".aside", paths[turn % 2])
        os.rename(paths[(turn + 1) % 2], paths[(turn + 1) % 2] + ".aside")
    assert not os.path.exists(setup.path("spool", "hold"))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))


def test_a_no_reaches_the_batch_tied_to_it_by_a_response_with_two_times(setup, monkeypatch):
    app = setup.app()
    ids = [setup.ask(app) for _ in range(3)]
    tied, rejected, other = setup.respond()
    for name, when in ((other, 7), (tied, 7)):
        for suffix in ("y", "tag", "proof") if name == other else ("proof",):
            os.utime(setup.path("spool", "out", "%s.%s" % (name, suffix)), ns=(when, when))

    def check(policy, job, *_):
        return proof_middleware.REJECTED if job["request_id"] == rejected else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    for proof_id in ids:
        call(app, "GET", JOBS + proof_id)
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_no_signal_goes_to_a_child_that_something_else_has_reaped(tmp_path, monkeypatch):
    sent, looks = [], []

    def gone_after_the_first_look(*_):
        looks.append(1)
        if len(looks) > 1:
            raise ChildProcessError
        return None

    monkeypatch.setattr(proof_middleware.os, "killpg", lambda _, number: sent.append(number))
    monkeypatch.setattr(proof_middleware.os, "waitid", gone_after_the_first_look)
    monkeypatch.setattr(proof_middleware, "CHECK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(proof_middleware, "STOP_S", 0.1)
    script = ["/bin/sh", "-c", "sleep 1"]
    assert proof_middleware.run_to_the_end(script, str(tmp_path)) == (None, b"")
    assert sent == [signal.SIGTERM]


def test_every_record_is_tried_though_one_is_in_the_way(setup, caplog):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    for proof_id in ids:
        with open(setup.job_path(proof_id, ".checked"), "w") as f:
            f.write("not a record")
    setup.state(app, ids[0])
    assert caplog.text.count("in the jobs directory stands in the way") == 2


def test_a_job_whose_response_has_gone_finishes_the_wipe_of_its_note(setup):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)
    for suffix in ("proof", "y", "tag"):
        os.remove(setup.path("spool", "out", "%s.%s" % (requests[1], suffix)))
    call(app, "GET", JOBS + ids[1])
    assert not os.path.exists(proof_path(setup, requests[0]))


def test_no_check_is_spent_on_a_batch_that_is_not_final(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.makedirs(setup.path("spool", "work"), exist_ok=True)
    open(setup.path("spool", "work", requests[1] + ".req"), "w").close()
    calls = []
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: calls.append(1))
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert calls == []


def test_a_response_replaced_under_its_record_is_judged_anew(setup, monkeypatch):
    app = setup.app()
    proof_id, path = setup.proved(app)
    late = setup.ask(app)
    setup.respond()
    put_another_proof(setup, proof_id, path, b"another proof", honest=True)
    with open(setup.job_path(proof_id)) as f:
        replaced = json.load(f)["request_id"]

    def check(policy, job, *_):
        no = job["request_id"] == replaced
        return proof_middleware.REJECTED if no else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    setup.state(app, late)
    assert os.path.exists(setup.path("spool", "hold"))


def test_a_no_beside_a_proof_without_a_job_still_counts(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    os.remove(setup.job_path(ids[1]))
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_no_byte_of_a_proof_reaches_the_log_at_a_no(setup, monkeypatch, caplog):
    app = setup.app()
    secret = b"QZXJKVWPGHQZXJKV"
    proof_id, _ = setup.proved(app, secret, poll=False)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert "did not pass" in caplog.text
    for start in range(len(secret) - 4):
        piece = secret[start : start + 5]
        assert piece.decode() not in caplog.text and piece.hex() not in caplog.text


def lay_a_spoiled_failed(setup, name, key):
    """A failed response whose tag is wrong."""
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (name, suffix)))
    open(setup.path("spool", "out", name + ".failed"), "wb").close()
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(bytes(32))


def lay_a_failed_with_bytes(setup, name, key):
    for suffix in ("proof", "y"):
        os.remove(setup.path("spool", "out", "%s.%s" % (name, suffix)))
    with open(setup.path("spool", "out", name + ".failed"), "wb") as f:
        f.write(b"bytes")
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(spool.failed_tag(key, name))


def lay_y_and_tag_alone(setup, name, key):
    os.remove(setup.path("spool", "out", name + ".proof"))


@pytest.mark.parametrize(
    "spoil", [lay_a_spoiled_failed, lay_a_failed_with_bytes, lay_y_and_tag_alone]
)
def test_a_mate_that_is_no_well_formed_response_keeps_its_batch_from_passing(setup, spoil):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    with open(setup.job_path(ids[1])) as f:
        key = bytes.fromhex(json.load(f)["tag_key"])
    spoil(setup, requests[1], key)
    setup.set_one_time(requests[1])
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


def test_no_record_if_a_worker_ends_without_an_answer(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    start = proof_middleware.start_worker
    monkeypatch.setattr(proof_middleware, "start_worker", lambda _: start(lambda: None))
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert not os.path.exists(setup.job_path(proof_id, ".checked"))


def test_an_overwritten_file_keeps_its_batchs_time(setup):
    app = setup.app()
    _, path = setup.proved(app, poll=False)
    out = os.open(setup.path("spool", "out"), os.O_RDONLY)
    proof_middleware.overwrite(out, os.path.basename(path), 10**18)
    os.close(out)
    assert os.stat(path).st_mtime_ns == 10**18


def test_a_no_stands_if_the_pool_breaks(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)

    start, started = proof_middleware.start_worker, []

    def breaks(target):
        if started:
            raise RuntimeError("can't start new thread")
        started.append(1)
        return start(target)

    setup.config["policies"]["exact"]["checks_at_once"] = 2
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    monkeypatch.setattr(proof_middleware, "start_worker", breaks)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_the_wipe_goes_on_without_room_for_the_note(setup, monkeypatch):
    app = setup.app()
    ids, _ = two_in_a_batch(setup, app)
    write = proof_middleware.write_file

    def no_room(jobs, name, data):
        if name.startswith("withheld."):
            raise OSError(errno.ENOSPC, "no room")
        write(jobs, name, data)

    monkeypatch.setattr(proof_middleware, "write_file", no_room)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))


def test_no_byte_of_a_proof_reaches_the_log(setup, monkeypatch, caplog):
    app = setup.app()
    secret = b"QZXJKVWPGHQZXJKV"
    proof_id, path = setup.proved(app, secret, poll=False)
    for outcome in (None, None, proof_middleware.REJECTED):
        monkeypatch.setattr(proof_middleware, "run_check", lambda *_: outcome)
        with contextlib.suppress(Exception):
            setup.state(app, proof_id)
    for start in range(len(secret) - 4):
        piece = secret[start : start + 5]
        assert piece.decode() not in caplog.text and piece.hex() not in caplog.text


def test_a_call_does_not_wait_for_a_check_by_default():
    assert WAIT_S_OF_THE_MODULE == 0


def test_the_child_gets_no_standard_input_and_its_standard_error_is_not_read(tmp_path, monkeypatch):
    seen, popen = {}, subprocess.Popen

    def noting(*arguments, **options):
        seen.update(options)
        return popen(*arguments, **options)

    monkeypatch.setattr(proof_middleware.subprocess, "Popen", noting)
    script = "echo on-stdout; echo on-stderr >&2"
    code, printed = proof_middleware.run_to_the_end(["/bin/sh", "-c", script], str(tmp_path))
    assert (code, printed) == (0, b"on-stdout\n") and seen["stdin"] == subprocess.DEVNULL


def test_a_proof_beside_a_failed_file_counts_as_a_defect(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    with open(setup.path("spool", "out", requests[1] + ".failed"), "wb"):
        pass
    setup.set_one_time(requests[1])
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.job_path(ids[0], ".checked"))


def test_a_check_runs_under_no_hold(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    checks = proof_middleware.BatchChecks(setup_config := load_config(setup.path("config.json")))
    called = []
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: called.append(1))
    open(setup.path("spool", "hold"), "w").close()
    checks.run(setup_config.policies["exact"], 10**18)
    assert called == []


def test_a_batch_is_not_checked_again_before_its_pause_is_over(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    called = []
    monkeypatch.setattr(proof_middleware, "RETRY_S", 1000)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: called.append(1))
    for _ in range(3):
        assert setup.state(app, proof_id)["state"] == "waiting"
    assert len(called) == 1


def test_the_proof_is_overwritten_before_y(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    names, overwrite = [], proof_middleware.overwrite

    def noting(out, name, tick_ns):
        names.append(name.rpartition(".")[2])
        overwrite(out, name, tick_ns)

    monkeypatch.setattr(proof_middleware, "overwrite", noting)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    setup.state(app, proof_id)
    assert names == ["proof", "y"]


def test_a_child_has_a_limit_of_its_own(tmp_path):
    code, printed = proof_middleware.run_to_the_end(["/bin/sh", "-c", "ulimit -t"], str(tmp_path))
    assert (code, printed) == (0, b"%d\n" % proof_middleware.cpu_limit())


def test_the_cpu_limit_counts_every_thread(monkeypatch):
    monkeypatch.setattr(proof_middleware, "CHECK_THREADS", 4)
    assert proof_middleware.cpu_limit() == 4 * proof_middleware.CHECK_TIMEOUT_S


def test_no_core_dump_from_the_child(tmp_path):
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if hard == 0:
        pytest.skip("core dumps are off here")
    resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))
    code, printed = proof_middleware.run_to_the_end(["/bin/sh", "-c", "ulimit -c"], str(tmp_path))
    resource.setrlimit(resource.RLIMIT_CORE, (soft, hard))
    assert (code, printed) == (0, b"0\n")


def test_memory_without_room_for_the_copy_is_passed_over(tmp_path, monkeypatch):
    usage = collections.namedtuple("Usage", "free")
    monkeypatch.setattr(proof_middleware.shutil, "disk_usage", lambda _: usage(10))
    jobs = os.open(tmp_path, os.O_RDONLY)
    with proof_middleware.check_folder(jobs, 100) as work:
        assert os.path.dirname(work) == proof_middleware.kept_path(jobs)
    os.close(jobs)


def test_nothing_that_cp_verify_said_of_a_proof_reaches_the_log(tmp_path, monkeypatch, caplog):
    line = "check_proof: error: cp_verify exited with status 3 and the last line 'QZXJKV WP', which"
    monkeypatch.setattr(proof_middleware, "run_to_the_end", lambda *_: (2, (line + "\n").encode()))
    (tmp_path / "p.proof").write_bytes(b"x")
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    assert RUN_CHECK(policy, job, bytes(4), str(tmp_path / "p.proof"), str(tmp_path)) is None
    assert "another exit of the verifier" in caplog.text and "QZXJKV" not in caplog.text


def test_the_cause_of_a_fault_around_the_check_is_told(tmp_path, monkeypatch, caplog):
    line = "check_proof: error: the leaf file leaf/3.bin, which the proof needs, is missing"
    monkeypatch.setattr(proof_middleware, "run_to_the_end", lambda *_: (2, (line + "\n").encode()))
    (tmp_path / "p.proof").write_bytes(b"x")
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    assert RUN_CHECK(policy, job, bytes(4), str(tmp_path / "p.proof"), str(tmp_path)) is None
    assert "a leaf file is missing" in caplog.text and "leaf/3" not in caplog.text


def test_a_death_in_the_wipe_is_finished_for_every_job(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    delete, deaths = spool.delete_response_at, []

    def dies_behind_the_delete(out, request_id):
        delete(out, request_id)
        if not deaths:
            deaths.append(1)
            raise OSError("it dies here")

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(spool, "delete_response_at", dies_behind_the_delete)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert deaths == [1] and os.path.exists(setup.path("jobs", "withheld.exact.%d.json" % 10**18))
    monkeypatch.setattr(spool, "delete_response_at", delete)
    os.remove(setup.path("spool", "hold"))
    gone = [ids[n] for n in (0, 1) if not os.path.exists(proof_path(setup, requests[n]))]
    for proof_id in ids:  # one has no response any more: its call finishes the work
        call(app, "GET", JOBS + proof_id)
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    assert os.path.exists(setup.path("spool", "hold"))  # it comes back once: a person should know
    os.remove(setup.path("spool", "hold"))
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["no_proof"] * 2


def batch_of(setup, app, count):
    ids = [setup.ask(app) for _ in range(count)]
    setup.respond()
    return ids


def test_no_more_checks_run_at_once_than_the_config_says(setup, monkeypatch):
    setup.entry["checks_at_once"] = 2
    app = setup.app()
    ids = batch_of(setup, app, 6)
    lock, running, most = threading.Lock(), [0], [0]

    def check(*_):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, ids[0])["state"] == "proved"
    assert most[0] == 2


def test_no_record_is_written_before_every_check_has_ended(setup, monkeypatch):
    setup.entry["checks_at_once"] = 3
    app = setup.app()
    ids = batch_of(setup, app, 3)
    go, ended, lock, started = threading.Event(), [], threading.Lock(), []

    def check(policy, job, *_):
        with lock:
            started.append(1)
            last = len(started) == 3
        if last:
            assert go.wait(10)
        with lock:
            ended.append(job["request_id"])
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "run_check", check)
    setup.state(app, ids[0])
    deadline = time.monotonic() + 5
    while len(ended) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert not any(name.endswith(".checked") for name in os.listdir(setup.path("jobs")))
    go.set()
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    assert setup.state(app, ids[0])["state"] == "proved"


@pytest.mark.parametrize("last", [False, True])
def test_a_no_wins_whichever_check_ends_first(setup, monkeypatch, last):
    setup.entry["checks_at_once"] = 3
    app = setup.app()
    ids = batch_of(setup, app, 3)
    with open(setup.job_path(ids[1])) as f:
        bad = json.load(f)["request_id"]

    def check(policy, job, *_):
        time.sleep((0.2 if last else 0) if job["request_id"] == bad else (0 if last else 0.2))
        return proof_middleware.REJECTED if job["request_id"] == bad else proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not any(x.endswith((".proof", ".checked")) for x in os.listdir(setup.path("spool", "out")))
    assert not any(x.endswith(".checked") for x in os.listdir(setup.path("jobs")))


def test_after_a_no_no_further_check_starts_and_the_operator_is_told_at_once(setup, monkeypatch, caplog):
    setup.entry["checks_at_once"] = 2
    app = setup.app()
    ids = batch_of(setup, app, 6)
    seen, go, both, lock = [], threading.Event(), threading.Barrier(2), threading.Lock()

    def check(policy, job, *_):
        with lock:
            seen.append(1)
            place = len(seen)
        if place > 2:
            return proof_middleware.PASSED
        first = place == 1
        both.wait(5)
        if first:
            return proof_middleware.REJECTED
        assert go.wait(5)
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    monkeypatch.setattr(proof_middleware, "run_check", check)
    setup.state(app, ids[0])
    deadline = time.monotonic() + 5
    while not os.path.exists(setup.path("spool", "hold")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert os.path.exists(setup.path("spool", "hold")) and "did not pass" in caplog.text
    assert any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))  # not yet
    assert caplog.text.count("did not pass") == 1
    go.set()
    deadline = time.monotonic() + 5
    while any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out"))):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert len(seen) <= 3


def test_a_batchs_tries_count_once_for_all_its_proofs(setup, monkeypatch):
    setup.entry["checks_at_once"] = 3
    app = setup.app()
    ids = batch_of(setup, app, 3)
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: None)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert not os.path.exists(setup.path("spool", "hold"))


def test_a_check_that_raises_gives_no_answer_and_its_mates_are_waited_for(setup, monkeypatch, caplog):
    setup.entry["checks_at_once"] = 2
    app = setup.app()
    ids = batch_of(setup, app, 2)
    ended, lock = [], threading.Lock()

    def check(policy, job, *_):
        with lock:
            first = not ended
            ended.append(1 if first else 2)
        if first:
            raise RuntimeError("a fault")
        time.sleep(0.1)
        return proof_middleware.PASSED

    monkeypatch.setattr(proof_middleware, "run_check", check)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert ended == [1, 2] and "RuntimeError" in caplog.text
    assert not any(name.endswith(".checked") for name in os.listdir(setup.path("jobs")))


def test_room_in_memory_is_asked_for_every_check_at_once(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(proof_middleware, "check_folder", lambda _, size: seen.append(size) or 1 / 0)
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    (tmp_path / "p").write_bytes(bytes(7))
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {}, "x", 3)
    with pytest.raises(ZeroDivisionError):
        RUN_CHECK(policy, job, bytes(4), str(tmp_path / "p"), str(tmp_path))
    assert seen == [21]


@pytest.mark.parametrize("value", [0, -1, True, "2", 1.5])
def test_checks_at_once_is_a_whole_number_from_1_on(setup, value):
    setup.entry["checks_at_once"] = value
    with pytest.raises(ConfigError, match="checks_at_once"):
        load_config(setup.write_config())


def test_checks_at_once_has_the_drivers_default(setup, monkeypatch):
    for cpus, expected in ((1, 1), (8, 2), (16, 4), (64, 6)):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(cpus)))
        assert load_config(setup.write_config()).policies["exact"].checks_at_once == expected


def test_a_note_outranks_a_mates_record(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    late = setup.ask(app)
    setup.respond()
    with open(setup.job_path(proof_id)) as f:
        name = json.load(f)["request_id"]
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump({"policy": "exact", "time": 10**18, "proofs": [name], "rejected": []}, f)
    error_of(call(app, "GET", JOBS + proof_id), 503)
    os.remove(setup.path("spool", "hold"))
    assert setup.state(app, proof_id)["state"] == "no_proof"
    assert setup.state(app, late)["state"] == "no_proof"


POLICY_OF_CHECK = dataclasses.make_dataclass(
    "PolicyOfCheck",
    [("commitment_id", str), ("public_key_dir", str), ("parameter_file", str), ("verifier", str),
     ("max_total_tokens", int), ("kind", str), ("limits", dict), ("name", str, "exact"),
     ("checks_at_once", int, 1)],
)  # fmt: skip


@pytest.mark.parametrize(
    "code, lines, expected",
    [
        (0, ["result: outcome=verified commitment_id=%s range=1:2 claim=exact-match" % COMMITMENT_ID, "x"], "passed"),
        (0, ["result: outcome=within-limits commitment_id=%s range=1:2" % COMMITMENT_ID, "x"], "passed"),
        (0, ["result: outcome=verified commitment_id=%s range=1:3" % COMMITMENT_ID, "x"], None),
        (0, ["result: outcome=verified commitment_id=%s range=1:2" % ("cd" * 32), "x"], None),
        (0, ["x"], None),  # a pass that says nothing: a lost exit code looks like this
        (1, ["result: outcome=verified commitment_id=%s range=1:2" % COMMITMENT_ID, "x"], None),
        (1, ["result: outcome=rejected", "check_proof: rejected: cp_verify printed NO: proof-refused"], "rejected"),
        (0, ["result: outcome=rejected", "check_proof: rejected: cp_verify printed NO: proof-refused"], "rejected"),
        (1, ["result: outcome=rejected", "check_proof: rejected: cp_verify printed NO: entry-refused: it rejected the leaf file"], None),
        (1, ["result: outcome=rejected", "check_proof: rejected: cp_verify rejected key.bin or a configuration file"], None),
        (2, ["result: outcome=error", "check_proof: error: no such file"], None),
        (14, ["result: outcome=not a verification", "x"], None),
        (2, ["result: outcome=error", "check_proof: error: cp_verify exited with status 5 and the last line"], None),
        (1, ["result: outcome=rejected", "check_proof: rejected: the proof file does not start with the header of this format"], None),
        (2, ["result: outcome=error", "check_proof: error: cp_verify was killed by signal 9"], None),
        (1, ["result: outcome=rejected", "check_proof: rejected: the response is about another next token than the request"], None),
        (1, ["result: outcome=rejected", "check_proof: rejected: the proving service made no proof"], None),
    ],
)  # fmt: skip
def test_what_run_check_takes_for_a_pass_and_for_a_rejection(tmp_path, monkeypatch, code, lines, expected):
    def run(words, *_):
        assert words[1:3] == ["-I", "-B"] and words[words.index("--threads") + 1] == "1"
        return code, ("\n".join(lines) + "\n").encode()

    monkeypatch.setattr(proof_middleware, "run_to_the_end", run)
    request = struct.pack("<6I", 1, 2, 3, 4, 1, 2)
    job = {"request_bytes": request.hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    assert RUN_CHECK(policy, job, bytes(4), __file__, str(tmp_path)) == expected


def test_a_check_that_does_not_end_gives_no_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "run_to_the_end", lambda *_: (None, b""))
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    assert RUN_CHECK(policy, job, bytes(4), __file__, str(tmp_path)) is None


def test_what_a_program_started_goes_even_if_the_program_ends_at_the_first_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "CHECK_TIMEOUT_S", 0.5)
    alive = tmp_path / "alive"
    script = "trap 'exit 0' TERM; (trap '' TERM; exec sleep 30) & echo $! > %s; wait" % alive
    assert proof_middleware.run_to_the_end(["/bin/sh", "-c", script], str(tmp_path)) == (None, b"")
    child = int(alive.read_text())
    for _ in range(200):
        with contextlib.suppress(FileNotFoundError), open("/proc/%d/stat" % child) as f:
            if f.read().rpartition(")")[2].split()[0] == "Z":
                return
            time.sleep(0.01)
            continue
        return
    raise AssertionError("the program's child lives on")


def test_a_program_that_ignores_the_first_signal_goes_at_the_second(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "CHECK_TIMEOUT_S", 0.3)
    monkeypatch.setattr(proof_middleware, "STOP_S", 0.3)
    started = time.monotonic()
    script = "trap '' TERM; sleep 30"
    assert proof_middleware.run_to_the_end(["/bin/sh", "-c", script], str(tmp_path)) == (None, b"")
    assert time.monotonic() - started < 5


def test_a_wipe_that_broke_off_goes_on_under_the_hold(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    replace, broke = proof_middleware.replace_by_failed, []

    def breaks_once(*arguments):
        if not broke:
            broke.append(1)
            raise OSError("it breaks off here")
        replace(*arguments)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", breaks_once)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    error_of(call(app, "GET", JOBS + ids[0]), 503)  # the hold stands
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    assert os.path.exists(setup.path("spool", "hold"))


def test_a_start_finishes_a_withheld_batch(setup, monkeypatch):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    note = {"policy": "exact", "time": 10**18, "proofs": sorted(requests), "rejected": requests[:1]}
    with open(setup.path("jobs", "withheld.exact.%d.json" % 10**18), "w") as f:
        json.dump(note, f)
    setup.app()
    deadline = time.monotonic() + 5
    while any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out"))):
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_a_strangers_entry_does_not_stop_the_sweep(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    jobs = os.open(setup.path("jobs"), os.O_RDONLY)
    prefix = proof_middleware.work_prefix(jobs)
    os.close(jobs)
    memory = proof_middleware.MEMORY_DIR
    os.symlink("/", os.path.join(memory, prefix + "a"))
    os.mkdir(os.path.join(memory, prefix + "b"))
    assert setup.state(app, proof_id)["state"] == "proved"
    assert os.listdir(memory) == [prefix + "a"]


def test_a_program_that_runs_over_the_time_limit_goes_with_what_it_started(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "CHECK_TIMEOUT_S", 0.5)
    alive = tmp_path / "alive"
    script = "sleep 30 & echo $! > %s; wait" % alive
    assert proof_middleware.run_to_the_end(["/bin/sh", "-c", script], str(tmp_path)) == (None, b"")
    child = int(alive.read_text())

    def is_alive():  # a killed process stays a zombie until its parent, here perhaps nobody, asks
        with contextlib.suppress(FileNotFoundError):
            with open("/proc/%d/stat" % child) as f:
                return f.read().rpartition(")")[2].split()[0] != "Z"
        return False

    for _ in range(200):
        if not is_alive():
            break
        time.sleep(0.01)
    assert not is_alive()


def test_a_next_token_outside_the_range_must_be_zero(setup, monkeypatch):
    app = setup.app()
    proof_id = setup.ask(app)
    (name,) = setup.respond()
    with open(setup.job_path(proof_id)) as f:
        key = bytes.fromhex(json.load(f)["tag_key"])
    y = b"\1\0\0\0"
    with open(setup.path("spool", "out", name + ".y"), "wb") as f:
        f.write(y)
    with open(setup.path("spool", "out", name + ".tag"), "wb") as f:
        f.write(spool.response_tag(key, name, y, b"dummy proof"))
    setup.set_one_time(name)
    seen, delete = [], spool.delete_response_at

    def note_and_delete(out, request_id):
        with open(setup.path("spool", "out", request_id + ".y"), "rb") as f:
            seen.append(f.read())
        return delete(out, request_id)

    monkeypatch.setattr(spool, "delete_response_at", note_and_delete)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    assert not os.path.exists(proof_path(setup, name))
    assert seen == [bytes(4)]  # overwritten first


def test_a_commitment_id_that_the_files_do_not_give_is_found_at_the_start(setup, monkeypatch):
    monkeypatch.setattr(proof_middleware, "plan_id", lambda policy: "cd" * 32)
    with pytest.raises(ConfigError, match="another commitment ID"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))
    assert "another commitment ID" in ProofMiddleware(FakeVllm(), config=setup.write_config()).fault
    assert proof_middleware.main(["check", setup.write_config()]) == 2


def test_a_verifier_that_others_can_change_or_that_cannot_run_is_refused(setup):
    os.chmod(setup.entry["verifier"], 0o777)
    with pytest.raises(ConfigError, match="writable by other users"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))
    os.chmod(setup.entry["verifier"], 0o644)
    with pytest.raises(ConfigError, match="cannot run"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))


def test_a_test_build_of_cp_verify_is_refused_at_the_start(setup):
    with open(setup.entry["verifier"], "w") as f:
        f.write("#!/bin/sh\necho 'TEST BUILD: x'\necho 'commitment ID: %s'\nexit 4\n" % COMMITMENT_ID)
    policy = load_config(setup.write_config()).policies["exact"]
    assert PLAN_ID(policy) is None


def test_limits_with_which_a_short_reply_could_never_pass_are_refused_at_the_start(setup):
    setup.write_parameters(C=1000, theta=1000, cap=5)
    with pytest.raises(ConfigError, match="theta equals C"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))


def test_a_verifier_below_a_secret_folder_is_refused(setup):
    secret = setup.root + "-secret"
    os.makedirs(os.path.join(secret, "below"), 0o700)
    open(os.path.join(secret, spool.SECRET_KEY_FILES[0]), "w").close()
    setup.entry["verifier"] = shutil.copy(setup.entry["verifier"], os.path.join(secret, "below"))
    with pytest.raises(ConfigError, match="secret"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))


def test_a_verifier_in_a_secret_folder_is_refused(setup):
    secret = setup.root + "-secret"
    os.mkdir(secret, 0o700)
    open(os.path.join(secret, spool.SECRET_KEY_FILES[0]), "w").close()
    setup.entry["verifier"] = shutil.copy(setup.entry["verifier"], secret)
    with pytest.raises(ConfigError, match="secret"):
        proof_middleware.check_verifiers(load_config(setup.write_config()))


def test_a_fault_in_the_checks_thread_is_told(setup, monkeypatch, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)

    def breaks(*_):
        raise RuntimeError("a fault")

    monkeypatch.setattr(proof_middleware.BatchChecks, "members", breaks)
    assert setup.state(app, proof_id)["state"] == "waiting"
    assert "could not be checked: RuntimeError" in caplog.text


def test_hold_needs_no_verifier(setup):
    setup.app()
    os.remove(setup.entry["verifier"])
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    assert os.path.exists(setup.path("spool", "hold"))


def test_at_a_rejection_what_a_mate_already_has_goes_too(setup, monkeypatch):
    app = setup.app()
    first, _ = setup.proved(app)
    late = setup.ask(app)
    setup.respond()
    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    assert setup.state(app, late)["state"] == "waiting"
    assert not os.path.exists(setup.job_path(first, ".checked"))
    os.remove(setup.path("spool", "hold"))
    assert setup.state(app, first)["state"] == "no_proof"


def test_a_withheld_batch_leaves_a_note_and_a_crash_is_finished_by_the_next_check(setup, monkeypatch, capsys):
    app = setup.app()
    ids, requests = two_in_a_batch(setup, app)
    replace, calls = proof_middleware.replace_by_failed, []

    def dies_after_the_first(*arguments):
        if calls:
            raise OSError("no space")
        calls.append(1)
        replace(*arguments)

    monkeypatch.setattr(proof_middleware, "run_check", lambda *_: proof_middleware.REJECTED)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", dies_after_the_first)
    assert setup.state(app, ids[0])["state"] == "waiting"
    assert os.path.exists(setup.path("spool", "hold"))
    (note,) = [x for x in os.listdir(setup.path("jobs")) if x.startswith("withheld.")]
    with open(setup.path("jobs", note)) as f:
        assert sorted(json.load(f)["proofs"]) == sorted(requests)
    monkeypatch.setattr(proof_middleware, "replace_by_failed", replace)
    os.remove(setup.path("spool", "hold"))
    for proof_id in ids:
        call(app, "GET", JOBS + proof_id)
    assert not any(x.endswith(".proof") for x in os.listdir(setup.path("spool", "out")))
    os.remove(setup.path("spool", "hold"))
    assert [setup.state(app, proof_id)["state"] for proof_id in ids] == ["no_proof"] * 2
    assert proof_middleware.main(["check", setup.path("config.json")]) in (0, 1)
    assert "A batch was withheld" in capsys.readouterr().out


def test_one_check_at_a_time_under_the_lock_on_the_jobs_directory(setup, monkeypatch):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    fd = os.open(setup.path("jobs"), os.O_RDONLY)
    proof_middleware.fcntl.flock(fd, proof_middleware.fcntl.LOCK_EX)
    try:
        assert setup.state(app, proof_id)["state"] == "waiting"
        time.sleep(0.2)
        assert setup.state(app, proof_id)["state"] == "waiting"
    finally:
        os.close(fd)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 10)
    assert setup.state(app, proof_id)["state"] == "proved"


def test_what_a_dead_check_left_is_swept(setup):
    app = setup.app()
    proof_id, _ = setup.proved(app, poll=False)
    jobs = os.open(setup.path("jobs"), os.O_RDONLY)
    left = setup.path("jobs", proof_middleware.work_prefix(jobs) + "x")
    os.close(jobs)
    other = os.path.join(setup.path("jobs"), proof_middleware.CHECK_WORK + "another")
    os.mkdir(left)
    os.mkdir(other)
    assert setup.state(app, proof_id)["state"] == "proved"
    assert not os.path.exists(left) and os.path.exists(other)


def test_a_check_goes_by_what_the_server_holds_not_by_a_name(setup, monkeypatch):
    app = setup.app()
    proof_id, path = setup.proved(app, poll=False)
    monkeypatch.setattr(proof_middleware, "MEMORY_DIR", setup.path("not there"))
    seen = {}

    def child(words, work, fds):
        seen.update(proof=open(words[4], "rb").read(), work=os.listdir(work))
        return 2, b""

    monkeypatch.setattr(proof_middleware, "run_check", RUN_THE_CHECK)
    monkeypatch.setattr(proof_middleware, "run_to_the_end", child)
    for name in ("jobs", "spool"):  # a directory of the same name takes their place
        os.rename(setup.path(name), setup.path(name + ".aside"))
        os.mkdir(setup.path(name), 0o700)
    checks = proof_middleware.BatchChecks(app.config)
    checks.run(app.config.policies["exact"], 10**18)
    assert seen == {"proof": b"dummy proof", "work": ["request.bin"]}
    assert os.listdir(setup.path("jobs")) == [] and os.listdir(setup.path("spool")) == []


def test_a_child_gets_the_descriptors_that_its_paths_go_through(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY)
    (tmp_path / "a file").write_text("there")
    script = 'cat "/proc/self/fd/%d/a file"' % fd
    try:
        word = ["/bin/sh", "-c", script]
        assert proof_middleware.run_to_the_end(word, str(tmp_path), (fd,)) == (0, b"there")
        assert proof_middleware.run_to_the_end(word, str(tmp_path))[0] != 0
    finally:
        os.close(fd)


def test_memory_that_other_accounts_can_change_is_passed_over(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "others_can_change", lambda _: ("a place", "a reason"))
    jobs = os.open(tmp_path, os.O_RDONLY)
    with proof_middleware.check_folder(jobs, 1) as work:
        assert os.path.dirname(work) == proof_middleware.kept_path(jobs)
    os.close(jobs)


def test_a_note_of_a_withheld_batch_stops_a_record_at_once(setup, monkeypatch):
    app = setup.app()
    proof_id, path = setup.proved(app)
    monkeypatch.setattr(proof_middleware, "WAIT_S", 0)
    name = os.path.basename(path).removesuffix(".proof")
    with open(setup.path("jobs", "withheld.exact.%d.json" % 7), "w") as f:  # another time: tied
        json.dump({"policy": "exact", "time": 7, "proofs": [name], "rejected": []}, f)
    error_of(call(app, "GET", JOBS + proof_id), 503)
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[0] != 200


def test_a_check_goes_on_in_the_jobs_directory_if_memory_cannot_be_had(tmp_path, monkeypatch):
    monkeypatch.setattr(proof_middleware, "MEMORY_DIR", str(tmp_path / "not there"))
    jobs = os.open(tmp_path, os.O_RDONLY)
    with proof_middleware.check_folder(jobs, 1) as work:
        assert os.path.dirname(work) == proof_middleware.kept_path(jobs) and os.path.isdir(work)
        assert os.listdir(tmp_path) == [os.path.basename(work)]
    assert os.listdir(tmp_path) == []
    os.close(jobs)


def test_a_checks_folder_is_in_memory_or_in_the_jobs_directory(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(proof_middleware, "run_to_the_end", lambda words, work, *_: seen.append(work) or (2, b""))
    job = {"request_bytes": struct.pack("<6I", 1, 2, 3, 4, 1, 2).hex()}
    policy = POLICY_OF_CHECK(COMMITMENT_ID, "/k", "/c/policy.txt", "/v", 4, "exact", {})
    RUN_CHECK(policy, job, bytes(4), __file__, str(tmp_path))
    above = os.path.dirname(seen[0])
    assert above == proof_middleware.MEMORY_DIR  # the copy of a proof stays off the disk


@pytest.mark.parametrize("status, found", [(4, COMMITMENT_ID), (0, None), (1, None)])
def test_plan_id_takes_one_line_and_exit_code_4(setup, status, found):
    with open(setup.entry["verifier"], "w") as f:
        f.write("#!/bin/sh\necho 'commitment ID: %s'\nexit %d\n" % (COMMITMENT_ID, status))
    for name in ("KL.bin", "layout.txt"):
        open(setup.path(name), "wb").close()
    policy = load_config(setup.write_config()).policies["exact"]
    assert PLAN_ID(policy) == found


@pytest.mark.parametrize("missing", ["KL.bin", "layout.txt"])
def test_a_configuration_that_is_not_whole_is_found_at_the_start(setup, missing):
    for name in {"KL.bin", "layout.txt"} - {missing}:
        open(setup.path(name), "wb").close()
    policy = load_config(setup.write_config()).policies["exact"]
    assert PLAN_ID(policy) is None


def test_a_job_holds_its_request_and_serves_it_to_nobody(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    with open(setup.job_path(proof_id)) as f:
        job = json.load(f)
    (request,) = setup.requests().values()
    assert bytes.fromhex(job["request_bytes"]) == request[: -spool.NONCE_BYTES]
    assert job["request_bytes"] not in json.dumps(setup.state(app, proof_id))


# ---- The config file ----


def test_minimal_config_creates_private_directories(setup):
    config = load_config(setup.write_config())
    assert config.policies["exact"].max_waiting == proof_middleware.DEFAULT_MAX_WAITING
    assert config.policies["exact"].max_stored == proof_middleware.DEFAULT_MAX_STORED
    assert config.policies["exact"].public_key_dir == setup.root + "-key"
    for path in (config.jobs_dir, config.policies["exact"].spool_dir):
        assert os.stat(path).st_mode & 0o777 == 0o700 and os.listdir(path) == ["mark.txt"]


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda c, e: c.pop("jobs_dir"), 'needs the key "jobs_dir"'),
        (lambda c, e: c.update(jobs_folder="/x"), "unknown key 'jobs_folder'"),
        (lambda c, e: c.update(policies={}), "at least one policy"),
        (lambda c, e: c.update(policies={"a b": e}), "policy name"),
        (lambda c, e: e.pop("served_model"), 'needs the key "served_model"'),
        (lambda c, e: e.update(commitment_id="AB" * 32), "64 lower-case hex digits"),
        (lambda c, e: e.update(max_waiting=0), '"max_waiting" must be a whole number'),
        (lambda c, e: e.update(max_waiting=True), '"max_waiting" must be a whole number'),
        (lambda c, e: e.update(max_stored=0), '"max_stored" must be a whole number'),
        (lambda c, e: e.update(max_stored="9"), '"max_stored" must be a whole number'),
        (lambda c, e: e.update(closed=1), '"closed" must be true or false'),
        (lambda c, e: e.update(spool_dir="spool"), "not an absolute path"),
        (lambda c, e: e.update(public_key_dir=""), "not an absolute path"),
        (lambda c, e: e.update(spool_dir=e["spool_dir"] + "/../spool"), "not an absolute path"),
        (lambda c, e: c.update(jobs_dir=c["jobs_dir"] + "/."), "not an absolute path"),
        (lambda c, e: e.update(parameter_file="policy.txt"), "not an absolute path"),
        (lambda c, e: e.update(parameter_file=e["parameter_file"] + ".none"), "No such file"),
        (lambda c, e: e.update(spool_dir=c["jobs_dir"]), "the same as, or inside"),
        (lambda c, e: e.update(spool_dir=c["jobs_dir"] + "/spool"), "the same as, or inside"),
        (lambda c, e: e.update(public_key_dir=e["spool_dir"] + "/out"), "the same as, or inside"),
    ],
)
def test_faulty_config_is_rejected_and_creates_nothing(setup, change, expected):
    change(setup.config, setup.entry)
    before = sorted(os.listdir(setup.root) + ["config.json"])
    with pytest.raises(ConfigError, match=expected):
        load_config(setup.write_config())
    assert sorted(os.listdir(setup.root)) == before


def test_nothing_is_marked_if_a_later_policy_is_faulty(setup):
    for name in ("spool", "other"):
        os.mkdir(setup.path(name), 0o700)
    open(setup.path("other", "a file"), "w").close()
    setup.config["policies"]["later"] = {**setup.entry, "spool_dir": setup.path("other")}
    with pytest.raises(ConfigError, match="mark.txt"):
        load_config(setup.write_config())
    assert os.listdir(setup.path("spool")) == [] and not os.path.exists(setup.path("jobs"))


@pytest.mark.parametrize("key", ["jobs_dir", "spool_dir"])
@pytest.mark.parametrize("below", ["new", "new/deeper"])
@pytest.mark.parametrize(
    "mode, safe", [(0o777, False), (0o775, False), (0o1777, True), (0o755, True)]
)
def test_directory_below_one_that_others_can_change_is_rejected(setup, key, below, mode, safe):
    os.mkdir(setup.path("above"))
    os.chmod(setup.path("above"), mode)
    (setup.config if key == "jobs_dir" else setup.entry)[key] = setup.path("above", below)
    if safe and below == "new":
        load_config(setup.write_config())
        return
    expected = "which another account may be able to change"
    if safe:  # only the last directory of a path is made
        expected = re.escape(setup.path("above", "new") + " does not exist: create it")
    # `denied`: main() passes the hint to run the command as the vLLM server's account.
    with pytest.raises(ConfigError, match=expected) as caught:
        load_config(setup.write_config(), denied=" (dummy hint)")
    assert "dummy hint" not in str(caught.value)  # the account is not what is wrong here
    assert os.listdir(setup.path("above")) == []
    assert not os.path.exists(setup.path("jobs")) and not os.path.exists(setup.path("spool"))


@as_root
def test_directory_below_another_account_s_is_rejected(setup):
    os.mkdir(setup.path("above"), 0o755)
    os.chown(setup.path("above"), OTHER_UID, OTHER_UID)
    setup.config["jobs_dir"] = setup.path("above", "jobs")
    with pytest.raises(ConfigError, match="owned by another account") as caught:
        load_config(setup.write_config(), denied=" (dummy hint)")
    assert str(caught.value).endswith("; nothing was run (dummy hint)")
    assert os.listdir(setup.path("above")) == []


def test_the_middleware_s_own_look_above_refuses(setup, monkeypatch):
    # spool.py calls its own. Root's first look is at the config file's directory.
    def look(path):
        return ("/dummy", spool.OPEN) if path == setup.root else None

    os.mkdir(setup.path("holder"))
    config = shutil.move(setup.write_config(), setup.path("holder"))
    monkeypatch.setattr(proof_middleware, "others_can_change", look)
    with pytest.raises(ConfigError, match="the jobs directory .* lies below /dummy, which another"):
        load_config(config)
    assert sorted(os.listdir(setup.root)) == ["holder", "policy.txt"]


@pytest.mark.parametrize("kind", ["file", "pipe", "symlink"])
def test_parent_that_is_not_a_directory_is_rejected(setup, kind):
    path = setup.path("parent")
    if kind == "file":
        open(path, "w").close()
    elif kind == "pipe":
        os.mkfifo(path)
    else:
        os.symlink(path, path)
    setup.config["jobs_dir"] = setup.path("parent", "jobs")
    with pytest.raises(ConfigError) as caught:
        load_config(setup.write_config())
    assert "create it" not in str(caught.value)
    assert sorted(os.listdir(setup.root)) == ["config.json", "parent", "policy.txt"]


def test_directory_of_another_account_gets_the_plain_message(setup, monkeypatch, capsys):
    def anothers(path, **arguments):  # what spool.py says to an account that is not root
        raise spool.UnsafePath("%s is owned by uid 5, not by the current user" % path)

    setup.app()
    monkeypatch.setattr(spool, "open_private_dir", anothers)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: %s: %s is owned by uid 5, not by the current user%s%s\n"
        % (setup.path("config.json"), setup.path("jobs"), DENIED, NOT_HELD)
    )


def test_no_proof_is_served_that_the_check_script_would_not_take():
    assert proof_middleware.MAX_PROOF_BYTES == proof_middleware.check_proof.MAX_FETCHED_BYTES


def test_parent_that_cannot_be_seen_is_not_called_missing(setup, monkeypatch):
    real, parent = os.lstat, setup.path("above")

    def no_look(path, **arguments):
        if path == parent:
            raise PermissionError(13, "Permission denied", path)
        return real(path, **arguments)

    setup.config["jobs_dir"] = setup.path("above", "jobs")
    monkeypatch.setattr(proof_middleware, "others_can_change", lambda path: None)
    monkeypatch.setattr(os, "lstat", no_look)
    with pytest.raises(ConfigError) as caught:
        load_config(setup.write_config(), denied=" (dummy hint)")
    assert str(caught.value).endswith(parent + ": Permission denied (dummy hint)")


def test_config_file_in_another_account_s_directory_gets_the_hint(setup, monkeypatch):
    def anothers(*arguments):  # what spool.py says to an account that is not root
        raise spool.UnsafePath("%s is owned by uid 5, not by root or the current user" % setup.root)

    monkeypatch.setattr(spool, "open_public_file", anothers)
    with pytest.raises(ConfigError) as caught:
        load_config(setup.write_config(), denied=" (dummy hint)")
    assert str(caught.value).endswith("not by root or the current user (dummy hint)")


def test_the_look_above_is_the_one_in_the_spool_module_beside_the_middleware():
    assert proof_middleware.others_can_change is spool.others_can_change
    assert os.path.dirname(spool.__file__) == os.path.dirname(proof_middleware.__file__)


def test_the_look_above_comes_before_anything_is_created(setup, monkeypatch):
    looked = []

    def look(path):
        looked.append((path, sorted(os.listdir(setup.root))))

    monkeypatch.setattr(proof_middleware, "others_can_change", look)
    load_config(setup.write_config())
    assert looked[:2] == [(setup.root, ["config.json", "policy.txt"])] * 2


def test_fault_of_the_config_file_itself_names_it_once(setup):
    path = setup.write_config()
    os.chmod(path, 0o664)
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert str(caught.value).startswith(path + " is writable by other users (mode 0664)")


def test_config_file_that_is_nested_too_deeply_is_rejected(setup):
    with open(setup.path("config.json"), "w") as f:
        f.write("[" * 200000)
    with pytest.raises(ConfigError, match="config.json"):
        load_config(setup.path("config.json"))


@pytest.mark.parametrize("changes", [{"lt": 3}, {"lt": 17}, {"V": 0}, {"sT": 5}])
def test_unusable_parameter_file_is_rejected(setup, changes):
    setup.write_parameters(**changes)
    with pytest.raises(ConfigError, match="policy.txt"):
        load_config(setup.write_config())


def test_parameter_file_may_have_the_programs_other_kinds_of_line(setup):
    with open(setup.path("policy.txt")) as f:
        lines = f.read()
    with open(setup.path("policy.txt"), "w") as f:
        f.write(
            "version H811\nwork positions 20 28 32\ngroup link 1 18 17\nranges-part 14\n" + lines
        )
    policy = load_config(setup.write_config()).policies["exact"]
    assert (policy.kind, policy.max_total_tokens, policy.vocabulary) == ("exact", 16, 200)


@pytest.mark.parametrize("temperature", [255, 256])
def test_parameter_file_with_a_temperature_is_rejected(setup, temperature):
    setup.write_parameters(temp=temperature)
    with pytest.raises(ConfigError, match='policy.txt has a line "temp", so it sets no supported'):
        load_config(setup.write_config())


@pytest.mark.parametrize("key", ["jobs_dir", "spool_dir", "public_key_dir"])
@pytest.mark.parametrize("below", ["", "inside", "inside/deeper"])
@pytest.mark.parametrize("file", ["seed.bin", "isk.txt"])
def test_secret_key_directory_is_rejected(setup, key, below, file):
    os.mkdir(setup.path("secret"), 0o700)
    for name in (file, "key.bin"):
        open(setup.path("secret", name), "w").close()
    (setup.config if key == "jobs_dir" else setup.entry)[key] = setup.path("secret", below).rstrip(
        "/"
    )
    with pytest.raises(ConfigError, match="secret key directory"):
        load_config(setup.write_config())
    assert sorted(os.listdir(setup.path("secret"))) == sorted([file, "key.bin"])


@pytest.mark.parametrize("key", ["jobs_dir", "spool_dir", "public_key_dir"])
def test_parent_symlink_into_a_secret_key_directory_is_rejected(setup, key):
    os.makedirs(setup.path("secret", "inside"), 0o700)
    open(setup.path("secret", "seed.bin"), "w").close()
    os.symlink(setup.path("secret", "inside"), setup.path("link"))
    (setup.config if key == "jobs_dir" else setup.entry)[key] = setup.path("link", "new")
    before = sorted(os.listdir(setup.root) + ["config.json"])
    with pytest.raises(ConfigError, match="secret key directory"):
        load_config(setup.write_config())
    assert sorted(os.listdir(setup.root)) == before
    assert os.listdir(setup.path("secret", "inside")) == []


@pytest.mark.parametrize("secret", ["jobs", "spool", "public"])
def test_secret_key_directory_that_is_moved_into_place_later_is_rejected_at_each_call(
    setup, caplog, secret
):
    setup.public_key_dir()
    app = setup.app()
    proof_id = setup.ask(app)
    open(setup.path(secret, "seed.bin"), "w").close()
    before = {name: sorted(os.listdir(setup.path(name))) for name in ("jobs", "spool", "public")}
    if secret == "public":
        error_of(call(app, "GET", PUBLISHED + "exact/key.bin"), 500)
    else:
        error_of(call(app, "GET", JOBS + proof_id), 500)
        error_of(call(app, "GET", JOBS + setup.ask(app)), 500 if secret == "jobs" else 404)
    assert {name: sorted(os.listdir(setup.path(name))) for name in before} == before
    assert len(setup.requests()) == 1 and "secret key directory" in caplog.text


def make_open_to_others(path):
    os.mkdir(path, 0o700)
    os.chmod(path, 0o750)


def make_symlink(path):
    os.mkdir(path + ".real", 0o700)
    os.symlink(path + ".real", path)


def make_file(path):
    open(path, "w").close()


@pytest.mark.parametrize("key", ["jobs", "spool"])
@pytest.mark.parametrize(
    "make, expected",
    [
        (make_open_to_others, "accessible to other users"),
        (make_symlink, "symlink"),
        (make_file, "not a directory"),
    ],
)
def test_private_directory_must_be_a_real_directory_closed_to_others(setup, key, make, expected):
    make(setup.path(key))
    with pytest.raises(ConfigError, match=expected):
        load_config(setup.write_config())


@pytest.mark.parametrize("ending", ["/", "//"])
def test_symlink_is_rejected_however_its_path_is_spelled(setup, ending):
    make_symlink(setup.path("jobs"))
    setup.config["jobs_dir"] += ending
    with pytest.raises(ConfigError, match="symlink"):
        load_config(setup.write_config())


@pytest.mark.parametrize("name", ["config.json", "policy.txt"])
def test_file_that_others_can_write_is_not_trusted(setup, name):
    setup.write_config()
    os.chmod(setup.path(name), 0o664)
    with pytest.raises(ConfigError, match="writable by other users"):
        load_config(setup.path("config.json"))


def test_parent_directory_may_be_a_symlink_of_this_account(setup):
    os.mkdir(setup.path("volume"), 0o755)
    os.symlink(setup.path("volume"), setup.path("link"))
    setup.config["jobs_dir"], setup.entry["spool_dir"] = (
        setup.path("link", "jobs"),
        setup.path("link", "spool"),
    )
    app = setup.app()
    assert setup.state(app, setup.ask(app))["state"] == "waiting"


def test_directory_that_is_opened_to_others_later_is_not_used(setup, caplog):
    app = setup.app()
    proof_id = setup.ask(app)
    os.chmod(setup.path("jobs"), 0o755)
    error_of(call(app, "GET", JOBS + proof_id), 500)
    assert call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())[2] == b"".join(
        setup.vllm.chunks
    )
    assert len(setup.jobs()) == 1 and "accessible to other users" in caplog.text


# ---- A spool directory's mark file ----


def test_startup_and_a_submit_mark_an_empty_spool_directory(setup):
    app = setup.app()
    for expected in (["mark.txt"], ["in", "mark.txt"]):
        assert sorted(os.listdir(setup.path("spool"))) == expected
        with open(setup.path("spool", "mark.txt")) as f:
            assert f.read() == "cp-folder 1\nkind spool\ncommitment %s\n" % COMMITMENT_ID
        os.remove(setup.path("spool", "mark.txt"))
        if expected == ["mark.txt"]:
            setup.ask(app)


@pytest.mark.parametrize(
    "mark",
    [
        None,
        "cp-folder 1\nkind spool\ncommitment %s\n" % ("cd" * 32),
        "cp-folder 1\nkind out\ncommitment %s\n" % COMMITMENT_ID,
    ],
)
def test_spool_directory_of_another_commitment_is_rejected(setup, mark, caplog):
    app = setup.app()
    os.mkdir(setup.path("spool", "in"), 0o700)
    os.remove(setup.path("spool", "mark.txt"))
    if mark:
        with open(setup.path("spool", "mark.txt"), "w") as f:
            f.write(mark)
    with pytest.raises(ConfigError, match="is not a spool directory"):
        load_config(setup.path("config.json"))
    error_of(call(app, "GET", JOBS + setup.ask(app)), 404)
    assert (
        os.listdir(setup.path("spool", "in")) == setup.jobs() == []
        and "is not a spool directory" in caplog.text
    )


def write_mark_of_another_commitment(setup):
    with open(setup.path("spool", "mark.txt"), "w") as f:
        f.write("cp-folder 1\nkind spool\ncommitment %s\n" % ("cd" * 32))


def test_mark_file_that_changes_after_startup_stops_the_job_routes(setup, caplog):
    app = setup.app()
    proof_id = setup.ask(app)
    setup.respond()
    write_mark_of_another_commitment(setup)
    before = setup.files()
    for method, file in (("GET", ""), ("GET", "/proof.bin"), ("DELETE", "")):
        assert (
            error_of(call(app, method, JOBS + proof_id + file), 500) == proof_middleware.JOB_FAULT
        )
    assert setup.files() == before and "is not a spool directory" in caplog.text


def test_clean_deletes_nothing_if_a_mark_file_is_for_another_commitment(setup, capsys):
    app = setup.app()
    os.utime(setup.job_path(setup.ask(app)), (0, 0))
    setup.respond()
    write_mark_of_another_commitment(setup)
    before = setup.files()
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 2
    assert setup.files() == before and "is not a spool directory" in capsys.readouterr().err


def test_spool_directory_with_only_an_unfinished_mark_counts_as_empty(setup):
    os.mkdir(setup.path("spool"), 0o700)
    open(setup.path("spool", "mark.txt.making.0123456789abcdef"), "w").close()
    app = setup.app()
    assert setup.state(app, setup.ask(app))["state"] == "waiting"
    assert "mark.txt" in os.listdir(setup.path("spool"))


# ---- The jobs directory ----

JOBS_MARK = b"cp-folder 1\nkind jobs\n"


def snapshot(directory):
    """Return {name: (mode, modification time, content)} for a directory."""
    found = {}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        info, content = os.lstat(path), None
        if stat.S_ISLNK(info.st_mode):
            content = os.readlink(path)
        elif stat.S_ISREG(info.st_mode):
            with open(path, "rb") as f:
                content = f.read()
        found[name] = info.st_mode, info.st_mtime_ns, content
    return found


def write_old_files(directory, files):
    """Create {name: content} files with an old modification time."""
    os.makedirs(directory, mode=0o700, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(directory, name), "xb") as f:
            f.write(content)
        os.utime(os.path.join(directory, name), (0, 0))


@pytest.mark.parametrize("found", ["missing", "empty", "with an unfinished mark file"])
def test_server_marks_a_new_jobs_directory_at_startup(setup, found):
    if found != "missing":
        os.mkdir(setup.path("jobs"), 0o700)
    if found == "with an unfinished mark file":
        open(setup.path("jobs", ".mark.txt.0123456789abcdef"), "w").close()
    app = setup.app()
    with open(setup.path("jobs", "mark.txt"), "rb") as f:
        assert f.read() == JOBS_MARK
    assert os.stat(setup.path("jobs", "mark.txt")).st_mode & 0o777 == 0o600
    assert setup.state(app, setup.ask(app))["state"] == "waiting"
    setup.app()
    assert len(setup.jobs()) == (1 if found != "with an unfinished mark file" else 2)


def test_servers_that_start_at_once_write_one_mark_file(setup, monkeypatch):
    write_file, both_found_it_empty = proof_middleware.write_file, threading.Barrier(2)

    def write_together(*arguments):
        both_found_it_empty.wait(10)
        write_file(*arguments)

    monkeypatch.setattr(proof_middleware, "write_file", write_together)
    path = setup.write_config()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert len(list(pool.map(load_config, [path, path]))) == 2
    assert setup.jobs() == []


# A well-formed job: in the next test only the missing mark file protects it from `clean`.
JOB = json.dumps({"policy": "exact", "not_provable": "dummy"}).encode()


@pytest.mark.parametrize(
    "files",
    [{"notes.txt": b"mine", "0" * 64 + ".json": JOB}, {".profile": b"mine"}, {".mark.txt": b""}],
)
def test_directory_with_files_and_no_mark_file_is_not_used_as_the_jobs_directory(
    setup, capsys, files
):
    write_old_files(setup.path("jobs"), files)
    before = snapshot(setup.path("jobs"))
    expected = setup.path("jobs") + " cannot be used as the jobs directory: it has no mark file"
    with pytest.raises(ConfigError, match=expected):
        load_config(setup.write_config())
    for arguments in (["check"], ["clean", "1"]):
        assert proof_middleware.main([arguments[0], setup.path("config.json"), *arguments[1:]]) == 2
        assert expected in capsys.readouterr().err
    assert snapshot(setup.path("jobs")) == before and not os.path.exists(setup.path("spool"))


def write_bytes(content):
    def write(path):
        with open(path, "wb") as f:
            f.write(content)

    return write


def link_to_a_true_mark_file(path):
    write_bytes(JOBS_MARK)(path + ".elsewhere")
    os.symlink(path + ".elsewhere", path)


@pytest.mark.parametrize(
    "make, expected",
    [
        (write_bytes(b""), "is for something else"),
        (write_bytes(JOBS_MARK[:-1]), "is for something else"),
        (write_bytes(JOBS_MARK + b"more\n"), "is for something else"),
        (
            write_bytes(b"cp-folder 1\nkind spool\ncommitment %s\n" % COMMITMENT_ID.encode()),
            "is for something else",
        ),
        (os.mkdir, "is for something else"),
        (os.mkfifo, "is for something else"),
        (link_to_a_true_mark_file, "is for something else"),
    ],
)
def test_jobs_directory_with_a_wrong_mark_file_is_rejected(setup, make, expected):
    os.mkdir(setup.path("jobs"), 0o700)
    make(setup.path("jobs", "mark.txt"))
    before = snapshot(setup.path("jobs"))
    with pytest.raises(ConfigError, match=expected):
        load_config(setup.write_config())
    assert snapshot(setup.path("jobs")) == before


def test_check_and_clean_write_nothing_into_an_empty_jobs_directory(setup):
    os.mkdir(setup.path("jobs"), 0o700)
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 0
    assert os.listdir(setup.path("jobs")) == []


@pytest.mark.parametrize("arguments", [{}, {"allow_empty": True}, {"mark": True}])
def test_open_jobs_passes_on_an_error_raised_inside(setup, arguments):
    os.mkdir(setup.path("jobs"), 0o700)
    if not arguments:
        write_bytes(JOBS_MARK)(setup.path("jobs", "mark.txt"))
    config = load_config(setup.write_config(), create_dirs=False)
    with pytest.raises(OSError, match="dummy"):
        with proof_middleware.open_jobs(config, **arguments):
            raise OSError(5, "dummy")


@pytest.mark.parametrize("emptied", [False, True])
def test_jobs_directory_that_loses_its_mark_file_after_startup_is_not_used(setup, caplog, emptied):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    os.remove(setup.path("jobs", "mark.txt"))
    if emptied:
        for name in os.listdir(setup.path("jobs")):
            os.remove(setup.path("jobs", name))
    before = snapshot(setup.path("jobs")), snapshot(setup.path("spool", "out"))
    for method, file in (("GET", ""), ("GET", "/proof.bin"), ("DELETE", "")):
        assert (
            error_of(call(app, method, JOBS + proof_id + file), 500) == proof_middleware.JOB_FAULT
        )
    setup.ask(app)
    assert (snapshot(setup.path("jobs")), snapshot(setup.path("spool", "out"))) == before
    assert setup.requests() == {} and "it has no mark file" in caplog.text


def test_clean_deletes_nothing_if_the_jobs_directory_has_lost_its_mark_file(setup):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    os.utime(setup.job_path(proof_id), (0, 0))
    os.remove(setup.path("jobs", "mark.txt"))
    before = snapshot(setup.path("jobs")), snapshot(setup.path("spool", "out"))
    with pytest.raises(spool.UnsafePath, match="it has no mark file"):
        proof_middleware.clean(app.config, 1)
    assert (snapshot(setup.path("jobs")), snapshot(setup.path("spool", "out"))) == before


def test_clean_keeps_every_file_that_is_not_a_job_file_or_a_record(setup, caplog):
    app = setup.app()
    proof_id, _ = setup.proved(app)
    for suffix in (".json", ".checked"):
        os.utime(setup.job_path(proof_id, suffix), (0, 0))
    record = snapshot(setup.path("jobs"))[proof_middleware.job_file(proof_id, ".checked")][2]
    other_names = {
        "notes.txt": JOB,
        "0" * 64: JOB,
        "0" * 63 + ".json": JOB,
        "0" * 65 + ".json": JOB,
        "A" * 64 + ".json": JOB,
        "g" * 64 + ".json": JOB,
        "0" * 64 + ".JSON": JOB,
        "0" * 64 + ".json~": JOB,
        "0" * 64 + ".json.bak": JOB,
        "0" * 64 + ".json\n": JOB,
        "0" * 64 + ".txt": JOB,
        "0" * 64 + ".proof": JOB,
        "0" * 64 + ".checked.old": record,
        "x" + "0" * 64 + ".checked": record,
        ".%s.json.%s" % ("0" * 64, "0" * 16): JOB,  # the form of a temporary name
    }
    same_names = {
        "1" * 64 + ".json": b"",
        "2" * 64 + ".json": b'{"name": "my own data"}',
        "3" * 64 + ".json": b"\x00\xff binary",
        "4" * 64 + ".json": record,  # a record under a job file's name
        "5" * 64 + ".json": JOB + b" " * proof_middleware.MAX_JOB_BYTES + b"and my own data",
        "6" * 64 + ".checked": b"[1, 2]",
        "7" * 64 + ".checked": JOB,  # a job under a record's name
    }
    write_old_files(setup.path("jobs"), {**other_names, **same_names})
    write_old_files(setup.path("elsewhere"), {"precious.json": JOB})
    specials = ["8" * 64 + ".json", "9" * 64 + ".json", "a" * 64 + ".checked"]
    os.symlink(setup.path("elsewhere", "precious.json"), setup.path("jobs", specials[0]))
    os.mkdir(setup.path("jobs", specials[1]))
    os.mkfifo(setup.path("jobs", specials[2]))
    for name in specials:
        os.utime(setup.path("jobs", name), (0, 0), follow_symlinks=False)
    before = snapshot(setup.path("jobs"))
    for suffix in (".json", ".checked"):
        del before[proof_middleware.job_file(proof_id, suffix)]

    assert proof_middleware.clean(app.config, 1) == (1, 0, 1, 0)  # the one job, and its response
    assert snapshot(setup.path("jobs")) == before
    assert snapshot(setup.path("elsewhere")) == {"precious.json": (0o100644, 0, JOB)}
    # Exactly the files with a job's or record's name are reported.
    reported = sorted(message.split(" was not deleted: ")[0] for message in caplog.messages)
    assert reported == sorted(setup.path("jobs", name) for name in (*same_names, *specials))
    assert {record.levelname for record in caplog.records} == {"WARNING"}


def test_delete_keeps_a_file_that_is_not_a_record(setup):
    app = setup.app()
    proof_id = setup.ask(app, temperature=1)  # not provable, so no call writes a record
    write_old_files(
        setup.path("jobs"), {proof_middleware.job_file(proof_id, ".checked"): b"my own data"}
    )
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert setup.jobs() == [proof_middleware.job_file(proof_id, ".checked")]


# ---- Failures ----


def lifespan(app):
    sent = []

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        sent.append(message)

    asyncio.run(app({"type": "lifespan"}, receive, send))
    return sent


def test_faulty_config_stops_the_server_at_startup(setup, caplog):
    setup.config["jobs_dir"] = "jobs"
    app = ProofMiddleware(FakeVllm(), config=setup.write_config())
    (message,) = lifespan(app)
    assert (
        message["type"] == "lifespan.startup.failed"
        and "not an absolute path" in message["message"]
    )
    # A web server that serves anyway: no proofs, and vLLM works as it does without the middleware.
    for method, path, headers in (
        ("POST", "/v1/completions", ASK),
        ("GET", "/plugins/proof/policies", {}),
        ("GET", JOBS + "0" * 64, {}),
    ):
        assert error_of(call(app, method, path, headers), 500) == proof_middleware.SERVER_FAULT
    assert app.app.bodies == [] and "not an absolute path" in caplog.text
    for scope in (http_scope("POST", "/v1/completions"), {"type": "websocket", "path": "/x"}):
        assert reaches_vllm_untouched(app, scope)


def test_config_path_comes_from_the_environment(setup, monkeypatch):
    monkeypatch.setenv("PROOF_MIDDLEWARE_CONFIG", setup.write_config())
    assert ProofMiddleware(FakeVllm()).fault is None
    monkeypatch.delenv("PROOF_MIDDLEWARE_CONFIG")
    assert "PROOF_MIDDLEWARE_CONFIG" in ProofMiddleware(FakeVllm()).fault


def test_text_of_an_internal_error_does_not_reach_the_client(setup, monkeypatch, caplog):
    app = setup.app()
    monkeypatch.setattr(app, "job_state", lambda *arguments: 1 / 0)
    assert error_of(call(app, "GET", JOBS + "0" * 64), 500) == proof_middleware.SERVER_FAULT
    assert "ZeroDivisionError" in caplog.text


def test_failure_to_store_a_job_leaves_the_reply_intact(setup, monkeypatch, caplog):
    app = setup.app("completion_stream.txt")
    monkeypatch.setattr(app, "store_job", lambda *arguments: 1 / 0)
    status, headers, body = call(app, "POST", "/v1/completions", ASK, json.dumps(FIELDS).encode())
    assert (status, body) == (200, b"".join(setup.vllm.chunks))
    error_of(call(app, "GET", JOBS + headers["x-proof-id"]), 404)
    assert "was not stored" in caplog.text


@pytest.mark.parametrize("error", [RuntimeError("dummy"), asyncio.CancelledError()])
@pytest.mark.parametrize(
    "source", ["vllm", "send, in vllm's reply", "send, in the middleware's route"]
)
def test_exception_from_vllm_or_the_web_server_passes_through(setup, caplog, source, error):
    app = setup.app()

    async def fail(*arguments):
        raise error

    if source == "vllm":
        app.app = fail
    with pytest.raises(type(error)) as raised:
        if source == "send, in the middleware's route":
            call(app, "GET", "/plugins/proof/policies", send=fail)
        else:
            body = json.dumps(FIELDS).encode()
            call(app, "POST", "/v1/completions", ASK, body, send=None if source == "vllm" else fail)
    passed_on = raised.value
    if sys.version_info < (3, 11) and isinstance(error, asyncio.CancelledError):
        # Python 3.10's asyncio.run() raises a new CancelledError, with the first as its context.
        passed_on = passed_on.__context__
    assert passed_on is error and caplog.text == ""


def test_removed_spool_directory_is_a_fault_of_the_server(setup, caplog):
    app = setup.app()
    proof_id = setup.ask(app)
    shutil.rmtree(setup.path("spool"))
    os.mkdir(setup.path("spool"), 0o700)  # the server does not see what now has the name
    for method, file in (("GET", ""), ("GET", "/proof.bin"), ("DELETE", "")):
        assert (
            error_of(call(app, method, JOBS + proof_id + file), 500) == proof_middleware.JOB_FAULT
        )
    assert len(setup.jobs()) == 1 and "No such file or directory" in caplog.text
    error_of(call(app, "GET", JOBS + setup.ask(app)), 404)  # the reply is served all the same
    assert "its proof job was not stored" in caplog.text and os.listdir(setup.path("spool")) == []


def test_spool_directory_that_is_moved_aside_is_still_the_server_s(setup, caplog):
    app = setup.app()
    proof_id = setup.ask(app)
    os.rename(setup.path("spool"), setup.path("elsewhere"))
    os.mkdir(setup.path("spool"), 0o700)
    other = setup.ask(app)
    assert len(os.listdir(setup.path("elsewhere", "in"))) == 3  # two requests and .tmp
    for one in (proof_id, other):
        assert setup.state(app, one)["state"] == "waiting"
    assert os.listdir(setup.path("spool")) == [] and not caplog.records


def test_repeated_error_is_logged_once_an_hour(setup, caplog):
    app = setup.app()
    proof_id = setup.ask(app)
    os.chmod(setup.path("jobs"), 0o755)
    for calls in (5, 2, 1):  # in three different hours
        for _ in range(calls):
            error_of(call(app, "GET", JOBS + proof_id), 500)
        app.last_logged = {
            key: (last - 3600, repeats) for key, (last, repeats) in app.last_logged.items()
        }
    assert [message.rpartition(": ")[2] for message in caplog.messages] == ["0)", "4)", "1)"]
    assert caplog.messages[0].endswith("; repeats not logged since the last line: 0)")


# ---- The commands ----


def test_check_prints_a_summary(setup, capsys):
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    printed = capsys.readouterr().out
    assert (
        "policy exact: exact (C 256, theta 256, cap 0, s 15)" in printed
        and COMMITMENT_ID in printed
    )


def test_check_counts_waiting_requests_and_stored_responses(setup, capsys):
    expected = "requests waiting: %d (max_waiting 64); responses stored: %d (max_stored 1000);"
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    assert expected % (0, 0) in capsys.readouterr().out
    app = setup.app()
    for failed in (False, True):
        setup.ask(app)
        setup.respond(failed=failed)
    setup.ask(app)
    before = snapshot(setup.path("spool")), snapshot(setup.path("jobs"))
    assert proof_middleware.main(["check", setup.path("config.json")]) == 0
    assert expected % (1, 2) in capsys.readouterr().out
    assert (snapshot(setup.path("spool")), snapshot(setup.path("jobs"))) == before


def test_check_reports_a_directory_that_changes_while_it_runs(setup, monkeypatch, capsys):
    def changed(top):
        raise spool.UnsafePath("dummy")

    setup.app()
    monkeypatch.setattr(proof_middleware, "waiting_and_stored", changed)
    assert proof_middleware.main(["check", setup.path("config.json")]) == 2
    assert capsys.readouterr().err == "error: dummy\n"


def test_check_creates_nothing_and_marks_missing_directories(setup, capsys):
    before = sorted(os.listdir(setup.root) + ["config.json"])
    assert proof_middleware.main(["check", setup.write_config()]) == 0
    printed = capsys.readouterr().out
    assert sorted(os.listdir(setup.root)) == before
    for name in ("jobs", "spool"):
        assert setup.path(name) + " (missing: init or the server creates it)" in printed


def test_init_creates_and_marks_the_directories(setup, capsys):
    for _ in range(2):
        assert proof_middleware.main(["init", setup.write_config()]) == 0
        printed = capsys.readouterr().out
        assert printed.startswith("policy exact: ") and "(missing: " not in printed
        for name in ("jobs", "spool"):
            assert os.listdir(setup.path(name)) == ["mark.txt"]
    for name in ("in", "work", "out"):  # as cp_prove creates them
        os.mkdir(setup.path("spool", name), 0o700)
    assert setup.state(setup.app(), setup.ask(setup.app()))["state"] == "waiting"


def test_permission_error_in_a_command_names_the_right_account(setup, monkeypatch, capsys):
    def no_access(path, **arguments):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(spool, "open_private_dir", no_access)
    assert ProofMiddleware(FakeVllm(), config=setup.write_config()).fault.endswith("denied")
    for command in (["check"], ["init"], ["clean", "1"]):
        assert proof_middleware.main([command[0], setup.path("config.json"), *command[1:]]) == 2
        assert capsys.readouterr().err == (
            "error: %s: %s: Permission denied (if you are not the account that runs the vLLM"
            " server, run this as that account: sudo -u ACCOUNT ...)\n"
            % (setup.path("config.json"), setup.path("jobs"))
        )


def test_config_file_that_cannot_be_opened_names_the_right_account(setup, monkeypatch, capsys):
    def no_access(*arguments):
        raise PermissionError(13, "Permission denied", path)  # spool.py gives the whole path

    path = setup.write_config()
    monkeypatch.setattr(spool, "open_public_file", no_access)
    assert proof_middleware.main(["hold", path, "exact"]) == 2
    assert capsys.readouterr().err == (
        "error: %s: Permission denied%s%s\n" % (path, DENIED, NOT_HELD)
    )


def test_check_reports_a_faulty_config(setup, capsys):
    setup.entry["commitment_id"] = "x"
    assert proof_middleware.main(["check", setup.write_config()]) == 2
    captured = capsys.readouterr()
    assert (
        captured.out == "" and captured.err.startswith("error: ") and captured.err.count("\n") == 1
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["check"],
        ["clean", "config.json"],
        ["clean", "config.json", "0"],
        ["clean", "config.json", "1.5"],
        ["remove", "x"],
    ],
)
def test_usage_error(arguments, capsys):
    with pytest.raises(SystemExit) as stopped:
        proof_middleware.main(arguments)
    assert stopped.value.code == 2 and capsys.readouterr().err.startswith("usage: ")


def test_clean_deletes_only_old_jobs_requests_and_responses(setup, capsys):
    app = setup.app()
    old, new = setup.ask(app), None
    setup.respond()
    waiting = setup.ask(app)
    os.mkdir(setup.path("spool", "work"), 0o700)
    # Old files that `clean` leaves because of their name or directory.
    others = [
        setup.path("jobs", "notes.txt"),
        setup.path("spool", "out", "notes.proof"),
        setup.path("spool", "out", "0" * 64 + ".other"),
        setup.path("spool", "in", "notes.req"),
        setup.path("spool", "in", "0" * 64 + ".other"),
        setup.path("spool", "in", "2" * 64),
        setup.path("spool", "in", ".tmp", "notes"),
        setup.path("spool", "in", ".tmp", "2" * 64 + ".req"),
        setup.path("spool", "work", "0" * 64 + ".req"),
    ]
    unfinished = setup.path("spool", "in", ".tmp", "1" * 64)
    for path in (*others, unfinished):
        open(path, "w").close()
    long_ago = time.time() - 3 * 86400
    for directory in ("jobs", "spool/out", "spool/in", "spool/in/.tmp", "spool/work"):
        for name in os.listdir(setup.path(directory)):
            os.utime(setup.path(directory, name), (long_ago, long_ago), follow_symlinks=False)
    new = setup.ask(app)
    assert proof_middleware.main(["clean", setup.path("config.json"), "4"]) == 0
    assert setup.state(app, old)["state"] == "proved"
    assert proof_middleware.main(["clean", setup.path("config.json"), "2"]) == 0
    # The jobs of `old` and `waiting`; the request of `waiting` and the `unfinished` one; the
    # response to `old`.
    assert capsys.readouterr().out.endswith(
        "Deleted 2 jobs, 2 requests and 1 response older than 2 days.\n"
    )
    for proof_id in (old, waiting):
        error_of(call(app, "GET", JOBS + proof_id), 404)
    assert setup.state(app, new)["state"] == "waiting"
    assert len(set(setup.requests()) - {"notes"}) == 1 and not os.path.exists(unfinished)
    assert all(map(os.path.exists, others))


@pytest.mark.parametrize("name, directory", [("exact", "spool"), ("other", "other")])
def test_clean_stops_at_a_hold_that_appears_while_it_runs(
    setup, monkeypatch, capsys, name, directory
):
    setup.config["policies"]["other"] = {**setup.entry, "spool_dir": setup.path("other")}
    app = setup.app()
    setup.proved(app)
    for old in ("jobs", "spool/out"):
        for file in os.listdir(setup.path(old)):
            os.utime(setup.path(old, file), (0, 0))
    unlink, responses = os.unlink, setup.files()[1]

    def unlink_and_hold(file, **arguments):
        unlink(file, **arguments)
        open(setup.path(directory, "hold"), "w").close()

    monkeypatch.setattr(os, "unlink", unlink_and_hold)
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.startswith(
        "error: clean stopped early; some files are already deleted: the policy %s is on hold ("
        % name
    )
    # Of the job file and its record, one was deleted before the hold was there.
    assert len(setup.jobs()) == 1 and setup.files()[1] == responses


def test_clean_leaves_the_request_of_a_job_that_it_leaves(setup):
    app = setup.app()
    proof_id = setup.ask(app)
    (request_id,) = setup.requests()
    # A job is deleted if it is older than `limit`, a request only if it is an hour older still.
    limit = time.time() - 86400
    os.utime(setup.path("spool", "in", request_id + ".req"), (limit - 10, limit - 10))
    os.utime(setup.job_path(proof_id), (limit + 10, limit + 10))
    assert proof_middleware.clean(app.config, 1) == (0, 0, 0, 0)
    os.utime(setup.job_path(proof_id), (limit - 5, limit - 5))
    assert proof_middleware.clean(app.config, 1) == (0, 0, 0, 0)  # its request stays
    os.utime(setup.path("spool", "in", request_id + ".req"), (limit - 3700, limit - 3700))
    assert proof_middleware.clean(app.config, 1) == (1, 1, 0, 0)


# A file that is gone before `clean` deletes it (FileNotFoundError) is no failure: exit code 0.
@pytest.mark.parametrize("error, code", [(PermissionError, 1), (FileNotFoundError, 0)])
def test_clean_exits_with_1_if_a_file_cannot_be_deleted(
    setup, monkeypatch, capsys, caplog, error, code
):
    app = setup.app()
    setup.proved(app)
    for directory in ("jobs", "spool/out"):
        for name in os.listdir(setup.path(directory)):
            os.utime(setup.path(directory, name), (0, 0))
    unlink = os.unlink

    def unlink_but_no_tag(name, **arguments):
        if name.endswith(".tag"):
            raise error(1, "Operation not permitted", name)
        unlink(name, **arguments)

    monkeypatch.setattr(os, "unlink", unlink_but_no_tag)
    assert proof_middleware.main(["clean", setup.path("config.json"), "1"]) == code
    printed = capsys.readouterr()
    # The response counts as deleted although its tag file stays.
    assert printed.out == "Deleted 1 job, 0 requests and 1 response older than 1 day.\n"
    assert printed.err == "error: 1 file could not be deleted: see above\n" * code
    assert len(caplog.messages) == code
    assert not code or caplog.messages[0].endswith(".tag was not deleted: Operation not permitted")
    assert [name[64:] for name in setup.files()[1]] == [".tag"]


def test_command_that_is_run_from_a_checkout_leaves_no_bytecode_there(setup):
    os.mkdir(setup.path("checkout"))
    for name in ("KL.bin", "layout.txt"):  # the real plan_id() runs in the command
        open(setup.path(name), "wb").close()
    source = os.path.dirname(proof_middleware.__file__)
    for name in ("proof_middleware.py", "check_proof.py", "spool.py"):
        shutil.copy(os.path.join(source, name), setup.path("checkout"))
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("PYTHON")
    }

    def run(*arguments):
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=setup.path("checkout"),
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    done = run("proof_middleware.py", "check", setup.write_config())
    assert done.returncode == 0, done.stderr
    assert len(os.listdir(setup.path("checkout"))) == 3
    # Importing the middleware, as vLLM does, leaves sys.dont_write_bytecode unchanged.
    done = run("-c", "import sys, proof_middleware; print(sys.dont_write_bytecode)")
    assert done.stdout == "False\n", done.stderr


def test_clean_creates_no_directory(setup, capsys):
    before = sorted(os.listdir(setup.root) + ["config.json"])
    assert proof_middleware.main(["clean", setup.write_config(), "1"]) == 0
    assert sorted(os.listdir(setup.root)) == before
    assert (
        capsys.readouterr().out == "Deleted 0 jobs, 0 requests and 0 responses older than 1 day.\n"
    )


def test_clean_removes_a_symlink_and_not_its_target(setup):
    app = setup.app()
    setup.ask(app)
    os.mkdir(setup.path("spool", "out"), 0o700)
    target, link = setup.path("precious"), setup.path("spool", "out", "0" * 64 + ".proof")
    open(target, "w").close()
    os.symlink(target, link)
    os.utime(link, (0, 0), follow_symlinks=False)
    assert proof_middleware.clean(load_config(setup.path("config.json")), 1) == (0, 0, 1, 0)
    assert os.path.exists(target) and not os.path.lexists(link)


# ---- Run by root, in another account's directories ----


@pytest.fixture
def no_sudo(monkeypatch, tmp_path):
    take_sudo_uid_out_or_skip(monkeypatch, tmp_path)


def give(uid, *paths):
    """Give files or directories, with all that lies in them, to an account."""
    for path in paths:
        os.chown(path, uid, uid)
        for folder, folders, files in os.walk(path):
            for name in folders + files:
                os.chown(os.path.join(folder, name), uid, uid, follow_symlinks=False)


@as_root
def test_init_takes_root_s_config_file_in_root_s_directory_in_the_sudo_caller_s_home(
    setup, no_sudo, monkeypatch
):
    # "sudo mkdir ~/srv", root's config file in it. The look above that file
    # trusts the account that ran sudo, so that account's home on the way is no ground to refuse.
    monkeypatch.setenv("SUDO_UID", str(OTHER_UID))
    mine = tree(setup, home_of(setup, OTHER_UID), "srv")
    other = tree(setup, "elsewhere")
    mine.entry["public_key_dir"] = other.path("public")
    mine.entry["parameter_file"] = other.path("parameters", "policy.txt")
    mine.entry["spool_dir"], mine.config["jobs_dir"] = other.path("spool"), other.path("jobs")
    assert proof_middleware.main(["init", mine.write_config()]) == 0


def test_an_account_that_is_not_root_gets_no_new_look(monkeypatch):
    def look(*arguments):
        raise AssertionError("looked")

    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() or 1)
    monkeypatch.setattr(proof_middleware, "others_can_change", look)
    proof_middleware.check_above_config("/nonexistent/config.json")
    proof_middleware.check_above_public("/nonexistent/public")


@as_root
@pytest.mark.parametrize("mode", [0o775, 0o2775, 0o777, "another account's"])
def test_root_s_config_file_below_a_folder_that_others_can_change_is_refused(
    setup, no_sudo, capsys, mode
):
    os.makedirs(setup.path("shared", "etc"), 0o755)
    config = shutil.move(setup.write_config(), setup.path("shared", "etc"))
    assert proof_middleware.main(["check", config]) == 0
    why = "its group or others can write in it, and it has no sticky bit"
    if mode == "another account's":
        os.chown(setup.path("shared"), OTHER_UID, OTHER_UID)
        why = "owned by another account"
    else:
        os.chmod(setup.path("shared"), mode)
    with open(config, "w") as f:
        f.write("dummy secret, and no JSON")  # no byte of it is read
    before = listing(setup.root)
    capsys.readouterr()
    assert proof_middleware.main(["init", config]) == 2
    assert capsys.readouterr().err == (  # and no hint behind it: nothing is denied to root
        "error: %s: it lies in or below %s, which another account may be able to change (%s): that"
        " account could put another file in its place, and so choose in which directories root"
        " works. Lay the config file in a directory of root's, below directories of root's, such as"
        " the one in step 3 of integrations/vllm/REFERENCE.md. Whatever the config file names in or"
        " below the directory that another account may change is refused next, for the same reason."
        " For each such jobs or spool directory, name a new one below directories of root's (init"
        " makes it; start the proving service on a new spool directory). Copy every other such file"
        " or directory there as root, from a source that you trust (do not move it: a moved file"
        " keeps its owner), and name the copy in the config file\n"
        % (config, setup.path("shared"), why)
    )
    assert ProofMiddleware(FakeVllm(), config=config).fault and listing(setup.root) == before


def published(setup, which):
    """Lay the three kinds of published file as cp_commit lays its own: closed, in closed
    directories. Return one file's path and its route."""
    setup.public_key_dir()
    os.mkdir(setup.path("parameters"))
    setup.entry["parameter_file"] = shutil.copy(setup.path("policy.txt"), setup.path("parameters"))
    with open(setup.path("parameters", "layout.txt"), "w") as f:
        f.write("dummy layout\n")
    for top in (setup.path("public"), setup.path("parameters")):
        for folder, _, files in os.walk(top):
            os.chmod(folder, 0o700)
            for name in files:
                os.chmod(os.path.join(folder, name), 0o600)
    folder = "parameters" if which == "layout.txt" else "public"
    return setup.path(folder, *which.split("/")), PUBLISHED + "exact/" + which


# ---- A process works where it looked ----


def tree(setup, *parts, text=b"honest "):
    """Make a Setup in a new directory below setup.root, with the three kinds of published file,
    each beginning with text."""
    os.makedirs(setup.path(*parts))
    one = Setup(setup.path(*parts))
    published(one, "key.bin")
    for name in ("public/key.bin", "public/leaf/0.bin", "parameters/layout.txt"):
        with open(one.path(name), "wb") as f:
            f.write(text + name.encode())
    return one


def listing(top):
    """Return everything below a directory: each path with its owner, its mode, a file's size, and
    the time of the last change to its content or, for a directory, to its entries."""
    found = {}
    for folder, folders, files in os.walk(top):
        for name in folders + files:
            info = os.lstat(os.path.join(folder, name))
            found[os.path.relpath(os.path.join(folder, name), top)] = (
                info.st_uid,
                info.st_mode,
                info.st_size if stat.S_ISREG(info.st_mode) else None,
                info.st_mtime_ns,
            )
    return found


def swap(old, new, how):
    """Move a directory aside and bring another under its name: by a link or by renaming."""
    os.rename(old, old + ".aside")
    how(new, old)
    return old if how is os.rename else new


SWAPS = pytest.mark.parametrize("how", [os.symlink, os.rename])
ROUTES = ("key.bin", "leaf/0.bin", "layout.txt")


@SWAPS
@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_swap_at_or_above_a_name_moves_no_running_server(setup, caplog, how, level):
    honest, decoy = tree(setup, "a", "b"), tree(setup, "decoy", "b", text=b"DECOY ")
    app = honest.app()
    decoy.proved(decoy.app())  # another service, alive
    proof_id, _ = honest.proved(app)
    caplog.clear()
    before = listing(decoy.root)
    # Links are tried at levels 1 and 2 only: at level 3, each named directory itself, both cases
    # rename (a link there would not be followed), so the link's case repeats the renaming's.
    if level == 3:
        now = [
            swap(honest.path(name), decoy.path(name), os.rename)
            for name in ("jobs", "spool", "public", "parameters")
        ]
        before = {name: v for name, v in before.items() if "/" in name}
    else:
        at = swap(
            *((honest.root, decoy.root) if level == 1 else (setup.path("a"), setup.path("decoy"))),
            how,
        )
        now = [at if level == 1 else os.path.join(at, "b")]
    for which in ROUTES:
        status, _, body = call(app, "GET", PUBLISHED + "exact/" + which)
        assert status == 200 and body.startswith(b"honest "), which
    assert honest.state(app, proof_id)["state"] == "proved"
    assert call(app, "GET", JOBS + proof_id + "/proof.bin")[::2] == (200, b"dummy proof")
    assert honest.state(app, honest.ask(app))["state"] == "waiting"
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    after = {
        os.path.join(os.path.basename(top) if level == 3 else "", name): v
        for top in now
        for name, v in listing(top).items()
    }
    assert after == before and not caplog.records  # nothing is turned in the other


@SWAPS
@pytest.mark.parametrize("command", ["clean", "hold", "lift-hold"])
def test_a_swap_inside_one_command_reaches_nothing(setup, monkeypatch, how, command):
    honest, decoy = tree(setup, "a", "b"), tree(setup, "decoy", "b", text=b"DECOY ")
    for one in (honest, decoy):
        one.proved(one.app())  # a job, its record and a response, old enough for clean
        for name in one.jobs():
            os.utime(one.path("jobs", name), ns=(10**18, 10**18))
        if command == "lift-hold":
            open(one.path("spool", "hold"), "w").close()
    before, name = listing(decoy.root), command.replace("-", "_")
    real = getattr(proof_middleware, name)

    def swap_first(*arguments):
        swap(honest.root, decoy.root, how)
        return real(*arguments)

    monkeypatch.setattr(proof_middleware, name, swap_first)
    typed = [command, honest.path("config.json"), *(["1"] if command == "clean" else ["exact"])]
    assert proof_middleware.main(typed) == 0
    assert listing(honest.root if how is os.rename else decoy.root) == before
    aside = honest.root + ".aside"
    if command == "clean":
        assert os.listdir(os.path.join(aside, "jobs")) == ["mark.txt"]
        assert os.listdir(os.path.join(aside, "spool", "out")) == []
    else:
        assert os.path.exists(os.path.join(aside, "spool", "hold")) == (command == "hold")


def test_a_hold_typed_behind_a_swap_still_stops_the_running_server(setup, caplog):
    # The command acts on what has the name now; the server works in what had it at its start.
    app = setup.app()
    proof_id, _ = setup.proved(app)
    os.rename(setup.path("spool"), setup.path("elsewhere"))
    os.mkdir(setup.path("spool"), 0o700)
    assert setup.state(app, proof_id)["state"] == "proved"
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0
    assert proof_middleware.main(["check", setup.path("config.json")]) == 1  # it sees this one
    assert os.listdir(setup.path("spool")) != [] and "hold" not in os.listdir(
        setup.path("elsewhere")
    )
    error_of(call(app, "GET", JOBS + proof_id), 503)
    waiting = len(os.listdir(setup.path("elsewhere", "in")))
    assert setup.state(app, setup.ask(app))["state"] == "not_provable"
    assert len(os.listdir(setup.path("elsewhere", "in"))) == waiting
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 0
    assert setup.state(app, proof_id)["state"] == "proved"
    open(setup.path("elsewhere", "hold"), "w").close()  # one in the server's own directory too
    app.last_logged.clear()  # a hold is logged at most once an hour
    caplog.clear()
    error_of(call(app, "GET", JOBS + proof_id), 503)
    assert not os.path.lexists(setup.path("spool", "hold"))  # the path that the line names
    assert "If %s exists, that is a hold." % setup.path("spool", "hold") in caplog.text
    assert "The server still has the old directory open, and a hold in it counts" in caplog.text
    assert "Neither lift-hold nor check sees that hold; a restart ends it" in caplog.text
    assert proof_middleware.main(["lift-hold", setup.path("config.json"), "exact"]) == 1
    assert proof_middleware.main(["check", setup.path("config.json")]) == 0  # 1 under a hold
    error_of(call(app, "GET", JOBS + proof_id), 503)  # neither command sees or reaches it
    again = setup.app()  # a restart does: it works in the new directory
    assert setup.state(again, setup.ask(again))["state"] == "waiting"


def test_clean_and_check_see_a_hold_behind_a_replacement_too(setup):
    setup.proved(setup.app())
    config = load_config(setup.path("config.json"))
    os.rename(setup.path("spool"), setup.path("elsewhere"))
    os.mkdir(setup.path("spool"), 0o700)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 0  # marks it too
    before = listing(setup.root)
    with pytest.raises(proof_middleware.OnHold, match="nothing was deleted"):
        proof_middleware.clean(config, 1)
    assert listing(setup.root) == before
    assert "(ON HOLD: " in proof_middleware.describe(config)[0]
    assert [policy.name for policy in proof_middleware.policies_on_hold(config)] == ["exact"]


HOLDS_THAT_DO_NOT_COUNT = {
    "no mark file": lambda path: os.unlink(os.path.join(path, "mark.txt")),
    "another commitment's mark file": lambda path: write_mark(path, "cd" * 32),
    "open to others": lambda path: os.chmod(path, 0o755),
}


def write_mark(path, commitment_id):
    os.unlink(os.path.join(path, "mark.txt"))
    os.rename(os.path.join(path, "hold"), path + ".hold")  # only an empty directory is marked
    with spool.open_private_dir(path) as top:
        spool.check_mark(top, commitment_id, create=True)
    os.rename(path + ".hold", os.path.join(path, "hold"))


@pytest.mark.parametrize("at", ["the name", "above the name"])
@pytest.mark.parametrize("why", [None, *HOLDS_THAT_DO_NOT_COUNT])
def test_a_hold_under_the_name_counts_only_in_what_could_be_the_spool_directory(setup, at, why):
    # Else whoever can aim the name would learn whether some other directory has an entry "hold".
    one, other = tree(setup, "a", "b"), tree(setup, "other", "b")
    app = one.app()
    proof_id, _ = one.proved(app)
    for command in (["init"], ["hold", "exact"]):  # makes and marks it; lays the hold
        assert proof_middleware.main([command[0], other.write_config(), *command[1:]]) == 0
    if why:
        HOLDS_THAT_DO_NOT_COUNT[why](other.path("spool"))
    if at == "the name":  # a link there is not followed
        swap(one.path("spool"), other.path("spool"), os.symlink)
    else:
        swap(one.root, other.root, os.symlink)
    if why or at == "the name":
        assert one.state(app, proof_id)["state"] == "proved"
    else:
        error_of(call(app, "GET", JOBS + proof_id), 503)


@as_root
def test_another_account_s_hold_is_not_seen_through_the_name(setup, no_sudo):
    one, other = tree(setup, "a", "b"), tree(setup, "other", "b")
    app = one.app()
    proof_id, _ = one.proved(app)
    for command in (["init"], ["hold", "exact"]):
        assert proof_middleware.main([command[0], other.write_config(), *command[1:]]) == 0
    give(OTHER_UID, other.root)
    swap(one.root, other.root, os.symlink)
    assert one.state(app, proof_id)["state"] == "proved"


@pytest.mark.parametrize("name", ["public", "parameters"])
def test_a_replaced_public_directory_is_seen_at_the_next_start(setup, name):
    honest, other = tree(setup, "a"), tree(setup, "b", text=b"other ")
    app = honest.app()
    swap(honest.path(name), other.path(name), os.rename)
    which = "key.bin" if name == "public" else "layout.txt"
    assert call(app, "GET", PUBLISHED + "exact/" + which)[2].startswith(b"honest ")
    assert call(honest.app(), "GET", PUBLISHED + "exact/" + which)[2].startswith(b"other ")
    shutil.rmtree(honest.path(name) + ".aside")
    assert call(app, "GET", PUBLISHED + "exact/" + which)[0] == 404


@pytest.mark.parametrize("spoil", ["writable by others", make_symlink, make_file])
def test_a_leaf_folder_that_cannot_be_trusted_is_found_at_the_start(setup, spoil):
    honest = tree(setup, "a")
    load_config(honest.write_config())
    if spoil == "writable by others":
        os.chmod(honest.path("public", "leaf"), 0o777)
    else:
        shutil.move(honest.path("public", "leaf"), honest.path("aside"))
        spoil(honest.path("public", "leaf"))
    with pytest.raises(ConfigError, match=re.escape(honest.path("public", "leaf"))):
        load_config(honest.write_config())


def test_leaf_files_that_come_later_are_served(setup):
    honest = tree(setup, "a")
    shutil.rmtree(honest.path("public", "leaf"))
    app = honest.app()
    assert call(app, "GET", PUBLISHED + "exact/leaf/7.bin")[0] == 404
    os.mkdir(honest.path("public", "leaf"))
    write_bytes(b"dummy leaf")(honest.path("public", "leaf", "7.bin"))
    assert call(app, "GET", PUBLISHED + "exact/leaf/7.bin")[::2] == (200, b"dummy leaf")


def test_no_descriptor_outlives_its_config(setup):
    published(setup, "key.bin")
    config = setup.write_config()
    load_config(config)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(50):
        load_config(config)
    kept = load_config(config)
    assert len(os.listdir("/proc/self/fd")) == before + 4  # jobs, spool, public key, parameters
    policy = kept.policies["exact"]
    del kept
    assert len(os.listdir("/proc/self/fd")) == before + 4  # a Policy needs them too
    del policy
    assert len(os.listdir("/proc/self/fd")) == before
    setup.entry["commitment_id"] = "dummy"
    for _ in range(3):
        with pytest.raises(ConfigError):
            load_config(setup.write_config())
    os.chmod(setup.path("public", "key.bin"), 0o666)  # a fault that is met when all are open
    setup.entry["commitment_id"] = COMMITMENT_ID
    with pytest.raises(ConfigError) as caught:
        load_config(setup.write_config())
    assert len(os.listdir("/proc/self/fd")) == before and caught  # also while the exception lives


@pytest.mark.parametrize("where", ["spool", "jobs"])
def test_a_directory_that_also_holds_the_parameter_file_is_opened_once(setup, monkeypatch, where):
    # A second open by the same path, by the rules for a public directory, could meet another one.
    load_config(setup.write_config())  # makes and marks both
    setup.entry["parameter_file"] = shutil.copy(setup.path("policy.txt"), setup.path(where))
    opened, real = [], spool.open_public_dir

    def noting(path, *more):
        opened.append(path)
        return real(path, *more)

    monkeypatch.setattr(spool, "open_public_dir", noting)
    before = len(os.listdir("/proc/self/fd"))
    config = load_config(setup.write_config())
    assert setup.path(where) not in opened
    assert len(os.listdir("/proc/self/fd")) == before + 3 and len(config.kept.fds) == 3


def test_two_names_of_one_directory_are_kept_apart(setup):
    # Through a link the spool directory has a second path: each path is opened by its own rules.
    load_config(setup.write_config())
    shutil.copy(setup.path("policy.txt"), setup.path("spool"))
    os.symlink(setup.root, setup.path("link"))
    setup.entry["parameter_file"] = setup.path("link", "spool", "policy.txt")
    kept = load_config(setup.write_config()).kept
    private, public = kept[setup.path("spool")], kept[setup.path("link", "spool")]
    assert private != public and os.path.sameopenfile(private, public) and len(kept.fds) == 4


def test_a_parameter_file_in_a_spool_directory_that_does_not_exist(setup):
    setup.entry["parameter_file"] = setup.path("spool", "policy.txt")
    with pytest.raises(ConfigError, match="%s: No such file or directory" % setup.path("spool")):
        load_config(setup.write_config())
    assert sorted(os.listdir(setup.root)) == ["config.json", "policy.txt"]


def test_each_named_directory_is_opened_by_its_path_once_and_no_file_below_by_its_path(
    setup, monkeypatch
):
    published(setup, "key.bin")
    config, events = setup.write_config(), []
    load_config(config)  # makes what is missing

    def noting(what, real):
        def wrapper(path, *more, **options):
            if os.path.isabs(path):
                events.append((what, path))
            return real(path, *more, **options)

        return wrapper

    for what in ("open_private_dir", "open_public_dir", "open_public_file"):
        monkeypatch.setattr(spool, what, noting(what, getattr(spool, what)))
    look = proof_middleware.others_can_change
    monkeypatch.setattr(
        proof_middleware,
        "others_can_change",
        lambda path: events.append(("look", path)) or look(path),
    )
    app = ProofMiddleware(FakeVllm(), config=config)
    opened = [event for event in events if event[0] != "look"]
    assert sorted(opened) == sorted([
        ("open_private_dir", setup.path("jobs")),
        ("open_private_dir", setup.path("spool")),
        ("open_public_dir", setup.root),  # by open_public_file(), for the config file
        ("open_public_dir", setup.root + "-key"),  # by open_public_file(), for the verifier
        ("open_public_dir", setup.path("parameters")),
        ("open_public_dir", setup.path("public")),
        ("open_public_file", config),
        ("open_public_file", setup.entry["verifier"]),
    ])  # fmt: skip
    if not os.geteuid():  # the look above a public directory comes behind its open
        for name in ("public", "parameters"):
            at = events.index(("open_public_dir", setup.path(name)))
            assert ("look", setup.root) in events[at:] and events[at + 1] == ("look", setup.root)
    # Behind the start. (Only spool.py's three functions are watched; the program itself opens
    # nothing by a path.) The one open by a path is the look for a hold: is_on_hold().
    del events[:]
    proof_id, _ = setup.proved(app)
    for route in (
        JOBS + proof_id,
        JOBS + proof_id + "/proof.bin",
        *(PUBLISHED + "exact/" + r for r in ROUTES),
    ):
        assert call(app, "GET", route)[0] == 200
    assert call(app, "DELETE", JOBS + proof_id)[0] == 200
    assert set(events) == {("open_private_dir", setup.path("spool"))}


def test_no_path_is_kept_twice(setup):
    kept = proof_middleware.Kept()
    kept.add(setup.root, spool.open_public_dir(setup.root))
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ValueError, match="is named twice"):
        kept.add(setup.root, spool.open_public_dir(setup.root))
    assert len(os.listdir("/proc/self/fd")) == before


def test_a_copy_of_a_config_closes_nothing(setup):
    config = load_config(setup.write_config())
    copies = [copy.copy(config), copy.deepcopy(config), dataclasses.asdict(config)]
    assert copies[0].kept is copies[1].kept is copies[2]["kept"] is config.kept
    del copies
    gc.collect()
    with proof_middleware.open_jobs(config), proof_middleware.open_spool(config.policies["exact"]):
        pass
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(config)


def test_no_context_manager_of_spool_py_s_is_left_open(setup):
    # At the interpreter's end it would be closed with a traceback, if the server's object lives
    # until then.
    published(setup, "key.bin")
    app = setup.app()
    left = [one for one in gc.get_objects() if type(one).__name__ == "generator"]
    assert app.fault is None and "_open_dir" not in [one.__name__ for one in left]


def test_no_exception_leaves_the_constructor(setup, monkeypatch):
    def fails(*arguments, **more):
        raise TypeError("dummy fault")

    monkeypatch.setattr(proof_middleware, "load_config", fails)
    app = ProofMiddleware(FakeVllm(), config=setup.write_config())
    assert app.fault == "dummy fault"
    error_of(call(app, "GET", "/plugins/proof/policies"), 500)


# ---- Run by root: the way to a directory ----


def home_of(setup, uid):
    """Make an account's home directory, and return its path."""
    os.mkdir(setup.path("home"), 0o755)
    give(uid, setup.path("home"))
    return setup.path("home")


def test_a_parameter_file_inside_a_secret_key_directory_is_refused(setup):
    os.makedirs(setup.path("secret", "config"))
    write_bytes(b"")(setup.path("secret", "seed.bin"))
    setup.entry["parameter_file"] = shutil.copy(
        setup.path("policy.txt"), setup.path("secret", "config")
    )
    with pytest.raises(
        ConfigError,
        match="is, or is inside, a secret key directory .%s contains seed.bin.: name a directory"
        " outside it .for a parameter file: a copy of the file there.$" % setup.path("secret"),
    ):
        load_config(setup.write_config())


@as_root
@pytest.mark.parametrize("name", ["public", "parameters"])
def test_root_s_own_public_directory_in_another_account_s_folder_is_refused(setup, no_sudo, name):
    published(setup, "key.bin")
    os.mkdir(setup.path("folder"), 0o755)
    moved = shutil.move(setup.path(name), setup.path("folder"))
    key = "public_key_dir" if name == "public" else "parameter_file"
    setup.entry[key] = moved if name == "public" else os.path.join(moved, "policy.txt")
    load_config(setup.write_config())
    os.chown(setup.path("folder"), OTHER_UID, OTHER_UID)
    with pytest.raises(ConfigError) as caught:
        load_config(setup.write_config(), denied=" (dummy hint)")
    assert "lies below %s, which another account" % setup.path("folder") in str(caught.value)
    assert str(caught.value).endswith(  # no hint behind it: it would lead past the look
        "Publish copies, made as root from a source that you trust, in a new directory below"
        " folders that belong to root and that only root can write, and name it (for a parameter"
        " file: the copy) in the config file"
    )
    os.chown(setup.path("folder"), 0, 0)
    os.chmod(setup.path("folder"), 0o775)  # as a clone under umask 002: refused too, for root
    with pytest.raises(ConfigError, match="its group or others can write in it"):
        load_config(setup.write_config())


@pytest.mark.skipif(os.geteuid() == 0, reason="for root the look is made")
def test_for_another_runner_nothing_looks_above_a_public_directory(setup):
    published(setup, "key.bin")
    os.mkdir(setup.path("folder"))
    setup.entry["public_key_dir"] = shutil.move(setup.path("public"), setup.path("folder"))
    os.chmod(setup.path("folder"), 0o775)  # as a clone under umask 002
    load_config(setup.write_config())


@as_root
def test_behind_one_s_own_sudo_what_has_the_name_at_a_start_is_served(setup, no_sudo, monkeypatch):
    # One case of what integrations/LIMITS.md tells as left behind one's own sudo, ahead of a start.
    monkeypatch.setenv("SUDO_UID", str(OTHER_UID))
    home = home_of(setup, OTHER_UID)
    honest, closed = tree(setup, "home", "up"), tree(setup, "home", "other", text=b"DECOY ")
    app = honest.app()
    swap(honest.path("public"), closed.path("public"), os.rename)
    assert call(app, "GET", PUBLISHED + "exact/key.bin")[2].startswith(b"honest ")
    assert call(honest.app(), "GET", PUBLISHED + "exact/key.bin")[2].startswith(b"DECOY ")
    assert os.stat(home).st_uid == OTHER_UID


# ---- Run by root: files really given to another owner. A hold and the published files ----


def take_sudo_uid_out_or_skip(monkeypatch, tmp_path):
    """For the fixture no_sudo: take SUDO_UID out of the environment. Run by root, the middleware
    then refuses a place below a directory that another account may be able to change: skip the test
    if pytest's temporary directory is such a place."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    found = None if os.geteuid() else spool.others_can_change(str(tmp_path))
    if found:
        pytest.skip(
            "pytest's temporary directory lies in or below %s, which another account may be able to"
            " change: let it lie below folders of root's (it follows --basetemp,"
            " PYTEST_DEBUG_TEMPROOT, TMPDIR, TEMP and TMP)" % found[0]
        )


# The next two try on a real change of owner what two tests further up, with nearly the same names,
# try for every account on a stand-in that raises a copy of spool.py's message.


@as_root
def test_directory_given_to_another_account_gets_the_plain_message(setup, capsys):
    setup.app()
    os.chown(setup.path("jobs"), OTHER_UID, OTHER_UID)
    assert proof_middleware.main(["hold", setup.path("config.json"), "exact"]) == 2
    printed = capsys.readouterr().err
    assert (
        "jobs is owned by uid %d, not by the current user (if you are not the account that runs"
        " the vLLM server, run this as that account: sudo -u ACCOUNT ...)%s\n"
        % (OTHER_UID, NOT_HELD)
    ) in printed
    assert "may be able to change" not in printed


@as_root
def test_config_file_in_a_directory_given_to_another_account_gets_the_hint(setup):
    path = setup.write_config()
    os.chown(setup.root, OTHER_UID, OTHER_UID)
    with pytest.raises(ConfigError) as caught:
        load_config(path, denied=" (dummy hint)")
    assert str(caught.value).endswith("not by root or the current user (dummy hint)")


def test_a_hold_does_not_stop_the_published_files(setup):
    setup.public_key_dir()
    write_layout(setup)
    app = setup.app()
    proof_id, _ = setup.proved(app)
    make_hold(setup, "file")
    error_of(call(app, "GET", JOBS + proof_id), 503)
    for name in ("key.bin", "leaf/0.bin", "layout.txt"):
        assert call(app, "GET", PUBLISHED + "exact/" + name)[0] == 200


# ---- The API key, however vLLM got it ----


class Bag:
    """An object with the given attributes."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def entry_of(module, kwargs):
    """An entry as Starlette keeps it, for a class of the name of vLLM's guard in this module."""
    return Bag(cls=type("AuthenticationMiddleware", (), {"__module__": module}), kwargs=kwargs)


class InVllm:
    """Hands each scope on with a stand-in for vLLM's application in it, as Starlette does with the
    true one. guard: the keywords of the entry of vLLM's own guard, or None for no such entry;
    options: what --api-key left in vLLM's parsed options; others: further entries."""

    def __init__(self, app, guard, options=None, others=()):
        entries = [Bag(cls=ProofMiddleware, args=(), kwargs={}), *others]
        if guard is not None:
            entries.append(entry_of("vllm.entrypoints.dummy", guard))
        self.app = app
        self.vllm = Bag(user_middleware=entries, state=Bag(args=Bag(api_key=options)))

    async def __call__(self, scope, receive, send):
        await self.app({**scope, "app": self.vllm}, receive, send)


def every_route(proof_id):
    """Each kind of call that the middleware answers itself or looks at before vLLM does."""
    own = (
        ("GET", "policies"),
        ("GET", "published/exact/key.bin"),
        ("GET", "published/exact/layout.txt"),
        ("GET", "published/exact/leaf/0.bin"),
        ("GET", "jobs/" + proof_id),
        ("GET", "jobs/" + proof_id + "/proof.bin"),
        ("GET", "jobs/" + proof_id + "/y.bin"),
        ("DELETE", "jobs/" + proof_id),
        ("HEAD", "jobs/" + proof_id),
        ("OPTIONS", "policies"),  # vLLM's own guard lets this method by
        ("GET", "no/such/route"),
    )
    return [(method, "/plugins/proof/" + route, {}) for method, route in own] + [
        ("POST", "/v1/completions", ASK),
        ("POST", "/v1/chat/completions", ASK),
        ("GET", "/v1/completions", ASK),
        ("GET", "/v1/models", ASK),
    ]


@pytest.mark.parametrize("sent", [None, "Bearer other", "Basic first-key", "Bearer first-ke"])
def test_every_route_asks_for_a_key_that_vllm_got_without_the_variable(setup, caplog, sent):
    setup.public_key_dir()
    inner = setup.app()
    proof_id, _ = setup.proved(inner, poll=False)  # before a key is asked for
    app, before = InVllm(inner, {"tokens": ["first-key", "second-key"]}), setup.files()
    asked = len(setup.vllm.bodies)
    caplog.set_level(logging.DEBUG)
    assert lifespan(app) == [] and "need no API key" not in caplog.text
    for method, path, headers in every_route(proof_id):
        if sent:
            headers = {**headers, "authorization": sent}
        for root_path in ("", "/behind/a/proxy"):
            result = call(app, method, path, headers, json.dumps(FIELDS).encode(), root_path)
            message = error_of(result, 401)
            assert "Authorization: Bearer KEY" in message and "-key" not in message
    assert setup.files() == before and len(setup.vllm.bodies) == asked
    assert "-key" not in caplog.text


@pytest.mark.parametrize("key", ["first-key", "second-key"])
def test_each_key_that_vllm_got_opens_the_routes(setup, key):
    setup.public_key_dir()
    inner = setup.app()
    proof_id, _ = setup.proved(inner, poll=False)
    app = InVllm(inner, {"tokens": ["first-key", "second-key"]})
    headers = {"authorization": "Bearer " + key}
    assert call(app, "GET", "/plugins/proof/policies", headers)[0] == 200
    assert call(app, "GET", "/plugins/proof/published/exact/key.bin", headers)[0] == 200
    assert json.loads(call(app, "GET", JOBS + proof_id, headers)[2])["state"] == "proved"
    assert call(app, "GET", JOBS + proof_id + "/proof.bin", headers)[2] == b"dummy proof"
    assert call(app, "DELETE", JOBS + proof_id, headers)[0] == 200
    body = json.dumps(FIELDS).encode()
    assert call(app, "POST", "/v1/completions", {**ASK, **headers}, body)[0] == 200


def test_beside_a_key_of_vllm_s_own_the_variable_s_does_not_count(setup, monkeypatch):
    # As in vLLM itself, which then takes the command line's keys alone.
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    app = InVllm(setup.app(), {"tokens": ["first-key"]}, ["first-key"])
    for key, status in ((KEY, 401), ("first-key", 200), ("other", 401), ("", 401)):
        headers = {"authorization": "Bearer " + key}
        assert call(app, "GET", "/plugins/proof/policies", headers)[0] == status


UNREADABLE = [
    ({}, None),  # the guard's entry has no such keyword
    ({"tokens": None}, None),
    ({"tokens": []}, None),
    ({"tokens": "first-key"}, None),  # not a list: each letter would count as a key
    ({"tokens": ["first-key", b"second-key"]}, None),
    ({"tokens": ["first-key", ""]}, None),  # an empty key would fit "Bearer "
    ({"tokens": ["first-key\udcff"]}, None),  # a byte of a command line that is not UTF-8
    ({"tokens": ["first-k\xe9y"]}, None),  # sent as UTF-8 or as Latin-1?
    (None, ["first-key"]),  # no entry of that name, and yet --api-key was given
    (None, "first-key"),
    (0, None),  # the entry's keywords are no dictionary
]


@pytest.mark.parametrize("guard, options", UNREADABLE)
def test_keys_of_vllm_s_that_cannot_be_read_close_the_server(setup, caplog, guard, options):
    setup.public_key_dir()
    inner = setup.app()
    proof_id, _ = setup.proved(inner, poll=False)
    before, asked = setup.files(), len(setup.vllm.bodies)
    caplog.set_level(logging.DEBUG)
    # A web server without the lifespan protocol: every call fails, whatever key it carries.
    for method, path, headers in every_route(proof_id):
        for sent in ({}, {"authorization": "Bearer first-key"}, {"authorization": "Bearer "}):
            result = call(InVllm(inner, guard, options), method, path, {**headers, **sent})
            assert error_of(result, 500) == proof_middleware.SERVER_FAULT
    assert setup.files() == before and len(setup.vllm.bodies) == asked
    assert "first-k" not in caplog.text
    # With it, the server does not start, and is told what to do.
    (message,) = lifespan(InVllm(inner, guard, options))
    assert message["type"] == "lifespan.startup.failed"
    assert "cannot read that key" in message["message"] and "Set VLLM_API_KEY" in message["message"]
    assert "first-k" not in message["message"] and "udcff" not in message["message"]


@pytest.mark.parametrize("guard, options", UNREADABLE)
def test_the_variable_guards_the_routes_where_vllm_s_keys_cannot_be_read(
    setup, monkeypatch, guard, options
):
    monkeypatch.setenv("VLLM_API_KEY", KEY)  # as the refusal at the start advises
    app = InVllm(setup.app(), guard, options)
    assert lifespan(app) == []
    for key, status in ((KEY, 200), ("first-key", 401), ("f", 401), ("", 401)):
        headers = {"authorization": "Bearer " + key}
        assert call(app, "GET", "/plugins/proof/policies", headers)[0] == status
    assert call(app, "GET", "/plugins/proof/policies")[0] == 401


ODD = [
    entry_of("starlette.middleware.authentication", {"backend": None}),  # Starlette's, same name
    entry_of("vllm_ext.auth", {"tokens": ["first-key"]}),  # another package than vllm
    Bag(cls=Bag(), kwargs={}),  # an object without a name, as functools.partial() gives
    Bag(cls=lifespan, kwargs=None),
    ("no", "such", "attributes"),
    None,
]


@pytest.mark.parametrize("options", [None, [], [""]])  # vLLM drops an empty key
def test_vllm_without_a_key_leaves_the_routes_open_and_the_start_says_so(setup, caplog, options):
    app = InVllm(setup.app(), None, options, ODD)
    assert lifespan(app) == [] and caplog.text.count("found no key of vLLM's") == 1
    assert call(app, "GET", "/plugins/proof/policies")[0] == 200


def test_other_entries_on_vllm_s_application_do_not_matter(setup, monkeypatch):
    app = InVllm(setup.app(), {"tokens": ["first-key"]}, ["first-key"], ODD)
    assert lifespan(app) == [] and call(app, "GET", "/plugins/proof/policies")[0] == 401
    headers = {"authorization": "Bearer first-key"}
    assert call(app, "GET", "/plugins/proof/policies", headers)[0] == 200
    for vllm in (Bag(), Bag(user_middleware=None, state=None), None, "no application"):
        app.vllm = vllm  # nothing to read there: the variable alone, as without an application
        assert lifespan(app) == [] and call(app, "GET", "/plugins/proof/policies")[0] == 200
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    app.app = setup.app()
    assert lifespan(app) == [] and call(app, "GET", "/plugins/proof/policies")[0] == 401


@pytest.mark.parametrize("key, status", [("first-key", 200), ("second-key", 200), ("other", 401)])
def test_a_sent_key_is_compared_with_every_key(setup, monkeypatch, key, status):
    # Whichever fits: otherwise the time of an answer could tell where in the list the key stands.
    app, compared = InVllm(setup.app(), {"tokens": ["first-key", "second-key", "third-key"]}), []
    real = proof_middleware.hmac.compare_digest

    def counting(sent, one):
        compared.append(one)
        return real(sent, one)

    monkeypatch.setattr(proof_middleware.hmac, "compare_digest", counting)
    headers = {"authorization": "Bearer " + key}
    assert call(app, "GET", "/plugins/proof/policies", headers)[0] == status
    assert compared == [b"first-key", b"second-key", b"third-key"]


class Raising:
    """An application whose list of entries cannot be read. The exception's text holds a key."""

    state = None

    @property
    def user_middleware(self):
        raise OSError("first-key")


class RaisingOptions:
    """Parsed options whose key cannot be read. The exception's text holds a key."""

    @property
    def api_key(self):
        raise RuntimeError("first-key")


class RaisingEntry:
    """An entry that cannot be read. The exception's text holds a key."""

    @property
    def cls(self):
        raise RuntimeError("first-key")


@pytest.mark.parametrize(
    "vllm",
    [
        Raising(),
        Bag(user_middleware=5),
        Bag(user_middleware=[], state=Bag(args=RaisingOptions())),
        # A key read from a first entry does not count once a later one cannot be read.
        Bag(user_middleware=[entry_of("vllm.x", {"tokens": ["first-key"]}), RaisingEntry()]),
    ],
)
def test_an_application_that_cannot_be_looked_through_closes_the_server(
    setup, monkeypatch, caplog, vllm
):
    caplog.set_level(logging.DEBUG)
    app = InVllm(setup.app(), None)
    app.vllm = vllm
    result = call(app, "GET", "/plugins/proof/policies", {"authorization": "Bearer first-key"})
    assert error_of(result, 500) == proof_middleware.SERVER_FAULT
    (message,) = lifespan(app)
    assert message["type"] == "lifespan.startup.failed" and "Set VLLM_API_KEY" in message["message"]
    assert "first-key" not in message["message"] + caplog.text
    monkeypatch.setenv("VLLM_API_KEY", KEY)  # as the refusal advises
    app.app = setup.app()
    assert lifespan(app) == [] and call(app, "GET", "/plugins/proof/policies")[0] == 401
    headers = {"authorization": "Bearer " + KEY}
    assert call(app, "GET", "/plugins/proof/policies", headers)[0] == 200


def order_faults(source):
    """Returns what breaks, in the text of proof_middleware.py, the order below, as sentences.

    A batch's check lists out/ once, and nothing in out/ says how many responses a batch has. That
    listing is whole for two reasons: a client's call starts no check while its own request is in
    work/, and cp_prove names every response of a batch before the first of their requests leaves
    work/ (a row on prover/spool.h in prover/tests/guards_stand.tsv holds that second part). So on
    the path of a client's call the is_in_work() test must come before everything that can start a
    check, and no other path may lead there. An ask for a note's time is left out: a note's batch
    has been listed already.
    """
    import ast

    names = ("ask", "check_response", "batch_waits", "finish", "check_batch")

    def uses(node):
        """Every mention of those names and of run() on self, called or handed on as a value."""
        calls = {id(c.func): c for c in ast.walk(node) if isinstance(c, ast.Call)}
        found = []
        for a in ast.walk(node):
            if isinstance(a, ast.Attribute) and (
                a.attr in names
                or (a.attr == "run" and ast.unparse(a.value) in ("self", "self.batches"))
            ):
                call = calls.get(id(a))
                if a.attr != "ask" or call is None or ast.unparse(call.args[1]) != "note['time']":
                    found.append(a)
        return found

    tree = ast.parse(source)
    test = ast.unparse(ast.parse('if is_in_work(top, job["request_id"]):\n    return state, None'))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
    routines = [n for n in ast.walk(tree) if isinstance(n, kinds)]
    found = [
        (routine, block)
        for routine in routines
        for block in ast.walk(routine)
        if isinstance(block, ast.With) and test in [ast.unparse(s) for s in block.body]
    ]
    if len(found) != 1:
        return ["the is_in_work() test stands in the body of %d with statements" % len(found)]
    client_path, block = found[0]
    at = [ast.unparse(s) for s in block.body].index(test)
    after = {id(u) for s in block.body[at + 1 :] for u in uses(s)}
    allowed = (
        ("check_response", "ask"),
        ("check_response", "batch_waits"),
        ("batch_waits", "ask"),
        ("ask", "run"),
        ("run", "check_batch"),
    )
    faults, seen = [], set()
    for routine in routines:
        for use in uses(routine):
            seen.add(id(use))
            if routine is client_path and id(use) not in after:
                faults.append("%s stands before the is_in_work() test" % use.attr)
            elif routine is not client_path and (routine.name, use.attr) not in allowed:
                faults.append("%s() uses %s with no is_in_work() test" % (routine.name, use.attr))
    faults += ["%s stands outside any routine" % u.attr for u in uses(tree) if id(u) not in seen]
    return faults


def test_no_check_starts_while_the_clients_request_is_in_work():
    with open(proof_middleware.__file__, encoding="utf-8") as f:
        source = f.read()
    assert order_faults(source) == []
    test = (
        '            if is_in_work(top, job["request_id"]):\n                return state, None\n'
    )
    ask = "self.batches.ask(policy, 0)\n"
    after_the_branch = "            # File names do not show that a response"
    after_the_check = "            if fault is WAITING:\n                return state, None\n            if fault:\n"
    docstring_end = (
        '        A failure is only logged: the job is gone, so the DELETE has succeeded."""\n'
    )
    async_head = "    async def own_route(self, method, route, receive, send):\n"
    for text in (test, after_the_branch, after_the_check, docstring_end, async_head):
        assert source.count(text) == 1, text
    without = source.replace(test, "")
    ahead = "            %s\n" + test
    plants = {
        "the test is gone": without,
        "the test comes after the branch for no response": without.replace(
            after_the_branch, test + after_the_branch
        ),
        "the test comes after the check": without.replace(after_the_check, test + after_the_check),
        "the test returns nothing": source.replace(
            test, test.replace("return state, None", "pass")
        ),
        "an ask comes before the test": source.replace(test, "            " + ask + test),
        "an ask stands in another routine": source.replace(
            docstring_end, docstring_end + "        " + ask
        ),
        "an ask stands in an async routine": source.replace(
            async_head, async_head + "        " + ask
        ),
        "an ask is handed on as a value": source.replace(
            test, ahead % "threading.Thread(target=self.batches.ask, args=(policy, 0)).start()"
        ),
        "run() comes before the test": source.replace(test, ahead % "self.batches.run(policy, 0)"),
        "check_batch() comes before the test": source.replace(
            test, ahead % "self.batches.check_batch(jobs, policy, 0)"
        ),
        "an ask stands outside any routine": source + "\nhanded_on = BatchChecks.ask\n",
    }
    for what, text in plants.items():
        assert text != source and order_faults(text), what
