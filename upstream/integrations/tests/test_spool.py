# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests for integrations/spool.py.

No proving service runs, and no proof is made or verified: the tests write the service's files
themselves. They create files only under pytest's tmp_path. As root, two tests change the owner of
tmp_path or of a file in it to uid 12345, and one switches to uid 65534 in a child process."""

import errno
import fcntl
import hashlib
import json
import os
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
import spool

SCRIPT = Path(spool.__file__)
COMMITMENT = "c0" * 32
OTHER_COMMITMENT = "d1" * 32
MARK = b"cp-folder 1\nkind spool\ncommitment %s\n" % COMMITMENT.encode()
REQUEST = spool.make_request([1, 2, 3, 4], 3, 4)
# A response's request ID and tag key, and the bytes of ID.y (the next token, 7) and ID.proof.
ID = "ab" * 32
KEY = bytes(range(32))
Y, PROOF = b"\x07\0\0\0", b"not a real proof"
NO_KEY = "the tag was not compared: no key was given"


def mode(path):
    """Return the mode without the set-group-ID bit, which a new directory inherits."""
    return stat.S_IMODE(os.lstat(path).st_mode) & ~stat.S_ISGID


def listing(path):
    return sorted(str(p.relative_to(path)) for p in Path(path).rglob("*"))


@pytest.fixture
def spool_dir(tmp_path):
    """A spool directory with COMMITMENT's mark file and an empty out/."""
    path = tmp_path / "spool"
    spool.make_private_dirs(path, "out", commitment_id=COMMITMENT)
    return path


def write_response(spool_dir, files, tag="right", request_id=ID, mtime_s=1000):
    """Write a response: out/ID.SUFFIX for each item of files, {suffix: bytes}, and out/ID.tag.

    tag: "right" for the tag under KEY, None for no tag file, or the tag file's bytes. All files get
    the same modification time, as read_response() requires."""
    for suffix, data in files.items():
        path = spool_dir / "out" / ("%s.%s" % (request_id, suffix))
        path.write_bytes(data)
        os.utime(path, (mtime_s, mtime_s))
    if tag == "right":
        if "proof" in files:
            tag = spool.response_tag(KEY, request_id, files["y"], files["proof"])
        else:
            tag = spool.failed_tag(KEY, request_id)
    if tag is not None:
        path = spool_dir / "out" / (request_id + ".tag")
        path.write_bytes(tag)
        os.utime(path, (mtime_s, mtime_s))


# ---- The wire format ----


# The service computes IDs, keys and tags by the same rules (prover/prover/spool.h):
# change both together.
def test_wire_format_vector():
    nonce = bytes(range(32))
    request = spool.make_request([1, 2, 3, 4], 3, 4)
    request_id, key = spool.request_id(request + nonce), spool.tag_key(request + nonce)
    proof = b"CPPF" + bytes(8)
    assert request_id == "ab493ee38f5a7d99895c25e21f09f859cadc67f6ef019551ddfef83c0839b7ea"
    assert key.hex() == "0d7ebcaad6480f3ea3c94598cdc432ffd67b77120f26e57d632ad6daf83acb71"
    assert spool.tag_key(nonce) == key
    tag = spool.response_tag(key, request_id, b"\x05\0\0\0", proof)
    assert tag.hex() == "a14d410dc8587b05070e06b95c481de0ba6790a6af4acb07cc218b05ef7efbd7"
    assert (
        spool.failed_tag(key, request_id).hex()
        == "185184cb79ce32943a4b856bde20140e3c6fca6baf99fe7e6a2cb22a490d8038"
    )
    in_pieces = spool.tag_hmac(key, request_id, b"\x05\0\0\0", len(proof))
    in_pieces.update(proof[:5])
    in_pieces.update(proof[5:])
    assert in_pieces.digest() == tag
    with pytest.raises(ValueError):
        spool.tag_key(nonce[:31])


def test_make_request_bytes():
    assert spool.make_request([7, 8, 9], 0, 2) == struct.pack("<5I", 7, 8, 9, 0, 2)
    assert spool.make_request([7, 8, 9], 2, 3) == struct.pack("<5I", 7, 8, 9, 2, 3)


@pytest.mark.parametrize(
    "tokens, start, end",
    [([7, 8], 1, 1), ([7, 8], 0, 3), ([7, 8], -1, 1), ([-1, 8], 0, 1), ([1 << 32], 0, 1)],
)
def test_make_request_rejects_bad_values(tokens, start, end):
    with pytest.raises(ValueError):
        spool.make_request(tokens, start, end)


def test_whole_reply_covers_the_reply():
    assert spool.whole_reply([7, 8], [1, 2, 3], 8) == struct.pack(
        "<10I", 7, 8, 1, 2, 3, 0, 0, 0, 1, 4
    )
    assert spool.whole_reply([7, 8], [1, 2, 3], 8, fill=5)[20:32] == struct.pack("<3I", 5, 5, 5)
    assert spool.whole_reply([7], [1], 2) == struct.pack("<4I", 7, 1, 0, 1)


def test_whole_reply_must_end_with_the_end_token():
    for reply in ([1, 2, 3], []):
        with pytest.raises(ValueError, match="does not end with the end token 99"):
            spool.whole_reply([7, 8], reply, 8, 99)
    assert spool.whole_reply([7, 8], [1, 2, 99], 8, 99) == struct.pack(
        "<10I", 7, 8, 1, 2, 99, 0, 0, 0, 1, 4
    )
    assert spool.whole_reply([99, 8], [1, 99], 8, 99)[:8] == struct.pack("<2I", 99, 8)


@pytest.mark.parametrize("reply", [[1, 99, 3], [99, 2, 99], [99, 99]])
def test_whole_reply_rejects_an_early_end_token(reply):
    with pytest.raises(ValueError, match="end token 99 before its last token"):
        spool.whole_reply([7, 8], reply, 8, 99)


@pytest.mark.parametrize(
    "prompt, reply, end_token, words",
    [
        ([], [1], None, "at least one token"),
        ([7], [], None, "at least one token"),
        ([7, 8], [1], None, "do not fit"),
        ([7], [1, 99], 99, "do not fit"),
    ],
)
def test_whole_reply_rejects_empty_or_oversized_input(prompt, reply, end_token, words):
    with pytest.raises(ValueError, match=words):
        spool.whole_reply(prompt, reply, 2, end_token)


def test_request_fields():
    assert spool.request_fields(spool.make_request([7, 8, 9], 1, 3)) == ([7, 8, 9], 1, 3)
    for data in (b"", bytes(8), bytes(13)):
        with pytest.raises(ValueError, match="not a request"):
            spool.request_fields(data)


def test_read_words(tmp_path):
    (tmp_path / "words").write_bytes(spool.pack([1, 2, 1 << 31]))
    assert spool.read_words(tmp_path / "words") == [1, 2, 1 << 31]
    (tmp_path / "odd").write_bytes(bytes(5))
    with pytest.raises(ValueError, match="not a whole number"):
        spool.read_words(tmp_path / "odd")


# ---- submit() ----


def test_submit_creates_request_atomically(tmp_path):
    path = tmp_path / "spool"
    request_id = spool.submit(path, REQUEST, COMMITMENT)
    assert listing(path) == ["in", "in/.tmp", "in/%s.req" % request_id, "mark.txt"]
    data = (path / "in" / (request_id + ".req")).read_bytes()
    assert data[:-32] == REQUEST and spool.request_id(data) == request_id
    assert [mode(path), mode(path / "in"), mode(path / "in" / ".tmp")] == [0o700] * 3
    assert mode(path / "in" / (request_id + ".req")) == 0o600


def test_submit_writes_and_syncs_before_the_final_name_appears(spool_dir, monkeypatch):
    events, link, fsync = [], os.link, os.fsync

    def recording_fsync(fd):
        events.append("fsync")
        return fsync(fd)

    def recording_link(src, dst, **kwargs):
        events.append("link " + dst[-4:])
        return link(src, dst, **kwargs)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "link", recording_link)
    spool.submit(spool_dir, REQUEST, COMMITMENT)
    assert events == ["fsync", "link .req"]


def test_submit_never_replaces_a_request(spool_dir, monkeypatch):
    monkeypatch.setattr(os, "urandom", lambda n: bytes(n))
    request_id = spool.submit(spool_dir, REQUEST, COMMITMENT)
    stored = spool_dir / "in" / (request_id + ".req")
    before = stored.stat()
    with pytest.raises(FileExistsError):
        spool.submit(spool_dir, REQUEST, COMMITMENT)
    assert (stored.stat().st_ino, stored.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert os.listdir(spool_dir / "in" / ".tmp") == []


def test_submit_that_fails_leaves_no_file(spool_dir, monkeypatch):
    def fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="No space left"):
        spool.submit(spool_dir, REQUEST, COMMITMENT)
    assert listing(spool_dir) == ["in", "in/.tmp", "mark.txt", "out"]


def test_submit_returns_the_tag_key_on_request(spool_dir):
    request_id, key = spool.submit(spool_dir, [1, 2, 3, 4, 3, 4], COMMITMENT, with_key=True)
    data = (spool_dir / "in" / (request_id + ".req")).read_bytes()
    assert data[:-32] == REQUEST and key == spool.tag_key(data)
    assert spool.submit(spool_dir, REQUEST, COMMITMENT) != request_id


# ---- Directories and files that are safe to use ----


def test_private_dir_is_created_with_mode_0700(tmp_path):
    with spool.open_private_dir(str(tmp_path / "new"), create=True) as fd:
        assert mode(tmp_path / "new") == 0o700
        with spool.open_private_dir("below", fd, create=True):
            pass
    assert mode(tmp_path / "new" / "below") == 0o700
    with pytest.raises(FileNotFoundError):
        with spool.open_private_dir(str(tmp_path / "missing")):
            pass


@pytest.mark.parametrize("bits", [0o750, 0o705, 0o720, 0o701, 0o777])
def test_private_dir_rejects_access_for_others(tmp_path, bits):
    (tmp_path / "open").mkdir()
    os.chmod(tmp_path / "open", bits)
    with pytest.raises(spool.UnsafePath, match="accessible to other users"):
        spool.submit(tmp_path / "open", REQUEST, COMMITMENT)
    assert listing(tmp_path / "open") == []


# geteuid() is replaced after the spool directory is opened. With a path, open_private_dir() would
# raise spool.OpenAbove first: the directories above would be another account's too, unless root's.
def test_private_dir_rejects_another_owner(spool_dir, monkeypatch):
    (spool_dir / "in").mkdir(mode=0o700)
    with spool.open_private_dir(str(spool_dir)) as spool_fd:
        monkeypatch.setattr(os, "geteuid", lambda: os.stat(spool_dir).st_uid + 1)
        with pytest.raises(spool.UnsafePath, match="not by the current user"):
            with spool.open_private_dir("in", spool_fd):
                pass


def reject_noatime(monkeypatch):
    """Make fcntl reject O_NOATIME, as the kernel does for another user's file."""
    real = fcntl.fcntl

    def rejecting(fd, command, argument=0):
        if command == fcntl.F_SETFL and argument & os.O_NOATIME:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        return real(fd, command, argument)

    monkeypatch.setattr(fcntl, "fcntl", rejecting)


@pytest.fixture
def public_file(tmp_path, monkeypatch):
    """A file of the current user, who is not root: root's files skip the O_NOATIME check."""
    os.chmod(tmp_path, 0o755)
    (tmp_path / "config").write_bytes(b"x")
    os.chmod(tmp_path / "config", 0o644)
    as_root = os.geteuid() == 0
    if as_root:
        try:
            os.chown(tmp_path / "config", 12345, -1)
        except OSError:
            pytest.skip("the owner cannot be changed here")
        os.chown(tmp_path, 12345, -1)
        monkeypatch.setattr(os, "geteuid", lambda: 12345)
    yield str(tmp_path / "config")
    if as_root:
        os.chown(tmp_path, 0, -1)


def test_owner_that_only_looks_like_the_current_user_is_rejected(spool_dir, monkeypatch):
    # Like another user's directory in a user namespace that does not map its owner.
    reject_noatime(monkeypatch)
    with pytest.raises(spool.UnsafePath, match="not owned by the current user"):
        spool.submit(spool_dir, REQUEST, COMMITMENT)


def test_public_owner_that_only_looks_like_the_current_user_is_rejected(public_file, monkeypatch):
    os.close(spool.open_public_file(public_file))
    reject_noatime(monkeypatch)
    with pytest.raises(spool.UnsafePath, match="not owned by the current user"):
        spool.open_public_file(public_file)


def test_read_only_file_system_is_accepted(spool_dir, monkeypatch):
    write_response(spool_dir, {"y": Y, "proof": PROOF})

    def read_only(*_, **__):
        raise OSError(errno.EROFS, os.strerror(errno.EROFS))

    real_open = os.open

    def open_for_reading(path, flags, *more, **still_more):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            read_only()
        return real_open(path, flags, *more, **still_more)

    monkeypatch.setattr(os, "open", open_for_reading)
    for name in "fchmod chmod mkdir rmdir link symlink unlink rename replace utime".split():
        monkeypatch.setattr(os, name, read_only)
    assert spool.read_response(spool_dir, ID, key=KEY).faults == []


def test_kernel_does_not_let_another_user_set_o_noatime():
    code = (  # a user who is not root checks "/", which root owns
        "import os, spool\n"
        "if os.geteuid() == 0:\n"
        "    os.setgroups([]), os.setgid(65534), os.setuid(65534)\n"
        "spool._confirm_owner(os.open('/', os.O_RDONLY), '/')\n"
    )
    if os.stat("/").st_uid != 0:
        pytest.skip('"/" is not owned by root here')
    path = "import sys; sys.path.insert(0, %r)\n" % str(SCRIPT.parent)  # -I drops PYTHONPATH
    words = [sys.executable, "-I", "-B", "-c", path + code]
    done = subprocess.run(words, capture_output=True, text=True)
    if "setg" in done.stderr or "setuid" in done.stderr:
        pytest.skip("root cannot change to another user here")
    assert done.stderr.splitlines()[-1] == "spool.UnsafePath: / is not owned by the current user"


def test_checks_change_nothing_in_a_directory_that_is_rejected(empty_dir):
    (empty_dir / "a file").write_text("not a spool directory")
    before = os.stat(empty_dir)
    time.sleep(0.05)  # longer than a timestamp's granularity
    with pytest.raises(spool.UnsafePath, match="Nothing in it was written or removed"):
        spool.submit(empty_dir, REQUEST, COMMITMENT)
    after = os.stat(empty_dir)
    assert (after.st_mode, after.st_ctime_ns, after.st_mtime_ns) == (
        before.st_mode,
        before.st_ctime_ns,
        before.st_mtime_ns,
    )
    # _confirm_owner() sets O_NOATIME on the descriptor and must restore the flags.
    with spool.open_private_dir(empty_dir) as fd:
        assert not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NOATIME


@pytest.mark.parametrize("slash", ["", "/", "//", "/."])
def test_private_dir_rejects_a_symlink(spool_dir, tmp_path, slash):
    os.symlink(spool_dir, tmp_path / "link")
    with pytest.raises(spool.UnsafePath, match="symlink"):
        spool.submit(str(tmp_path / "link") + slash, REQUEST, COMMITMENT)
    assert listing(spool_dir) == ["mark.txt", "out"]


def test_private_dir_rejects_a_file(tmp_path):
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(spool.UnsafePath, match="not a directory"):
        spool.submit(tmp_path / "file", REQUEST, COMMITMENT)


@pytest.mark.parametrize("name", spool.SECRET_KEY_FILES)
def test_private_dir_rejects_a_secret_key_directory(tmp_path, name):
    (tmp_path / "secret").mkdir(mode=0o700)
    (tmp_path / "secret" / name).write_bytes(b"")
    with pytest.raises(spool.UnsafePath, match="secret key directory"):
        spool.submit(tmp_path / "secret", REQUEST, COMMITMENT)
    assert listing(tmp_path / "secret") == [name]


def test_submit_rejects_a_subdirectory_that_is_not_private(spool_dir):
    (spool_dir / "in").mkdir(mode=0o755)
    os.chmod(spool_dir / "in", 0o755)
    with pytest.raises(spool.UnsafePath, match="accessible to other users"):
        spool.submit(spool_dir, REQUEST, COMMITMENT)


def test_directory_below_one_that_others_can_change_is_refused(tmp_path):
    (tmp_path / "shared").mkdir()
    os.chmod(tmp_path / "shared", 0o777)
    with pytest.raises(
        spool.OpenAbove, match="lies below %s, which another" % (tmp_path / "shared")
    ):
        spool.submit(tmp_path / "shared" / "spool", REQUEST, COMMITMENT)
    assert listing(tmp_path / "shared") == []
    os.chmod(tmp_path / "shared", 0o1777)  # sticky, like /tmp
    spool.submit(tmp_path / "shared" / "spool", REQUEST, COMMITMENT)


@pytest.mark.parametrize("kind", ["missing", "a file", "a dangling link"])
def test_only_the_last_directory_of_a_path_is_created(tmp_path, kind):
    if kind == "a file":
        (tmp_path / "above").write_text("")
    elif kind == "a dangling link":
        (tmp_path / "above").symlink_to("nothing")
    before = listing(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not create it.: '%s'" % (tmp_path / "above")):
        spool.submit(tmp_path / "above" / "spool", REQUEST, COMMITMENT)
    assert listing(tmp_path) == before


def test_public_directory_which_is_only_read_may_lie_below_an_open_one(tmp_path):
    (tmp_path / "shared" / "public").mkdir(parents=True)
    os.chmod(tmp_path / "shared" / "public", 0o755)
    os.chmod(tmp_path / "shared", 0o777)
    with spool.open_public_dir(str(tmp_path / "shared" / "public")):
        pass


def test_directory_below_a_link_of_ones_own_is_used_by_its_resolved_path(tmp_path):
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "link").symlink_to("real")
    spool.submit(tmp_path / "link" / "spool", REQUEST, COMMITMENT)
    assert (tmp_path / "real" / "spool" / "mark.txt").is_file()
    os.chmod(tmp_path / "real" / "spool", 0o755)
    with pytest.raises(spool.UnsafePath, match="^%s is accessible" % (tmp_path / "real" / "spool")):
        spool.submit(tmp_path / "link" / "spool", REQUEST, COMMITMENT)


def test_commands_refuse_a_place_below_a_directory_that_others_can_change(tmp_path):
    (tmp_path / "shared").mkdir()
    os.chmod(tmp_path / "shared", 0o777)
    (tmp_path / "request").write_bytes(REQUEST)
    (tmp_path / "tokens.json").write_text('{"prompt_token_ids": [1, 2], "reply_token_ids": [3]}')
    tokens = tmp_path / "tokens.json"
    for words in (
        ("submit", tmp_path / "shared" / "spool", tmp_path / "request", COMMITMENT),
        ("sweep", tmp_path / "shared" / "spool", COMMITMENT, "1"),
        ("make-request", 6, tokens, tokens, 3, tmp_path / "shared" / "r"),
    ):
        code, out, err = run(*words)
        assert (code, out) == (2, "") and err.startswith("spool.py: the "), words
        assert err.endswith("; nothing was run\n") and len(err.splitlines()) == 1
    assert listing(tmp_path / "shared") == []
    code, out, err = run("make-request", 6, tokens, tokens, 3, str(tmp_path) + "/new/")
    assert (code, out) == (1, "") and "names a directory" in err and not (tmp_path / "new").exists()


def test_name_that_something_else_has_for_a_moment_while_the_directory_is_made(
    tmp_path, monkeypatch
):
    def taken(path, mode, dir_fd=None):
        raise FileExistsError(
            errno.EEXIST, "File exists", path
        )  # as if an entry had just been there

    monkeypatch.setattr(os, "mkdir", taken)
    words = "%s could not be created: something else had that name" % (tmp_path / "new")
    with pytest.raises(spool.UnsafePath, match=words):
        with spool.open_private_dir(tmp_path / "new", create=True):
            pass
    assert not (tmp_path / "new").exists()


def test_make_request_looks_above_its_own_output_file(tmp_path):
    (tmp_path / "tokens.json").write_text('{"prompt_token_ids": [1, 2], "reply_token_ids": [3]}')
    words = ("make-request", 6, tmp_path / "tokens.json", tmp_path / "tokens.json", 3)
    code, out, err = run(*words, tmp_path / "none" / "r")
    assert (code, out) == (1, "") and "%s: Not an existing directory" % (tmp_path / "none") in err
    (tmp_path / "shared").mkdir()
    os.chmod(tmp_path / "shared", 0o777)
    code, out, err = run(*words, tmp_path / "shared" / "r")
    assert code == 2 and err.startswith("spool.py: the file %s lies " % (tmp_path / "shared" / "r"))
    code, out, err = run("sweep", tmp_path / "shared" / "spool", COMMITMENT, "1")
    assert err.startswith("spool.py: the directory %s lies " % (tmp_path / "shared" / "spool"))
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "link").symlink_to("real")
    assert run(*words, tmp_path / "link" / "r")[0] == 0
    taken = "error: %s already exists (choose a new name)\n" % (tmp_path / "real" / "r")
    assert run(*words, tmp_path / "link" / "r") == (1, "", taken)


def test_a_name_in_an_open_directory_must_be_a_single_name(spool_dir):
    with spool.open_private_dir(str(spool_dir)) as fd:
        for name in ("..", "out/..", "/tmp", ""):
            with pytest.raises(ValueError):
                with spool.open_private_dir(name, fd):
                    pass


def test_public_file(tmp_path):
    os.chmod(tmp_path, 0o755)
    (tmp_path / "config").write_bytes(b"x")
    os.chmod(tmp_path / "config", 0o644)
    fd = spool.open_public_file(str(tmp_path / "config"))
    assert os.read(fd, 9) == b"x"
    os.close(fd)
    with spool.open_public_dir(str(tmp_path)) as dir_fd:
        os.close(spool.open_public_file("config", dir_fd))


def test_public_file_rejections(tmp_path):
    os.chmod(tmp_path, 0o755)
    for name in ("good", "writable"):
        (tmp_path / name).write_bytes(b"x")
        os.chmod(tmp_path / name, 0o644)
    os.chmod(tmp_path / "writable", 0o664)
    os.symlink("good", tmp_path / "link")
    os.mkfifo(tmp_path / "fifo", 0o644)
    for name, words in (
        (
            "writable",
            r"writable by other users \(mode 0[0-7]+\): use a new copy under a new name, from a "
            "source that you trust; or, if it is yours, run chmod go-w on it, and "
            "then look at what is in it",
        ),
        ("link", "symlink"),
        ("fifo", "not a regular file"),
    ):
        with pytest.raises(spool.UnsafePath, match=words):
            spool.open_public_file(str(tmp_path / name))
    for bits in (0o777, 0o1777, 0o775):  # also sticky, like /tmp
        os.chmod(tmp_path, bits)
        with pytest.raises(spool.UnsafePath, match="writable by other users") as caught:
            spool.open_public_file(str(tmp_path / "good"))
        # The advice to chmod a directory goes to its owner, but never to root.
        assert str(caught.value).endswith(spool.CHMOD_ADVICE) == (os.geteuid() != 0)
        assert ("run chmod" in str(caught.value)) == (os.geteuid() != 0)


@pytest.mark.parametrize("caller, advised", [(0, False), (12345, True), (54321, False)])
def test_chmod_is_advised_for_a_directory_only_to_its_owner_who_is_not_root(
    tmp_path, monkeypatch, caller, advised
):
    """Caller 54321 stands for an account, not root, that checks a directory owned by root."""
    os.chmod(tmp_path, 0o1777)
    fd = os.open(tmp_path, os.O_RDONLY)
    real = os.fstat(fd)
    owner = 0 if caller == 54321 else caller
    seen = os.stat_result((real.st_mode, real.st_ino, real.st_dev, 1, owner, 0, 0, 0, 0, 0))
    monkeypatch.setattr(spool.os, "fstat", lambda _: seen)
    monkeypatch.setattr(spool.os, "geteuid", lambda: caller)
    with pytest.raises(spool.UnsafePath, match=r"writable by other users \(mode 1777\)") as caught:
        spool._check_not_writable_by_others(fd, str(tmp_path))
    monkeypatch.undo()
    os.close(fd)
    assert str(caught.value).endswith(spool.CHMOD_ADVICE) == advised
    assert ("run chmod" in str(caught.value)) == advised  # in no other sentence either


def test_public_file_rejects_another_owner(tmp_path, monkeypatch):
    os.chmod(tmp_path, 0o755)
    (tmp_path / "config").write_bytes(b"x")
    os.chmod(tmp_path / "config", 0o644)
    if os.geteuid() == 0:  # root's files are accepted for every user
        try:
            os.chown(tmp_path / "config", 12345, -1)
        except OSError:
            pytest.skip("the owner cannot be changed here")
    else:
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(spool.UnsafePath, match="not by root or the current user"):
        spool.open_public_file(str(tmp_path / "config"))


# ---- The mark file ----


def check_mark(path, commitment_id, create=False):
    with spool.open_private_dir(str(path)) as fd:
        spool.check_mark(fd, commitment_id, create)


@pytest.fixture
def empty_dir(tmp_path):
    (tmp_path / "empty").mkdir(mode=0o700)
    return tmp_path / "empty"


def rejected(path, commitment_id, create=False, words=""):
    """Assert that check_mark() rejects the directory and neither adds nor removes a name in it."""
    before = listing(path)
    with pytest.raises(spool.UnsafePath, match=words) as caught:
        check_mark(path, commitment_id, create)
    assert "is not a spool directory" in str(caught.value) and str(path) in str(caught.value)
    assert listing(path) == before


def test_mark_is_created_in_an_empty_directory(empty_dir, monkeypatch):
    synced, fsync = [], os.fsync

    def recording_fsync(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    check_mark(empty_dir, COMMITMENT, create=True)
    # The mark file is synced first, then its directory.
    assert synced == [False, True]
    assert listing(empty_dir) == ["mark.txt"]
    assert (empty_dir / "mark.txt").read_bytes() == MARK and len(MARK) == 99
    assert mode(empty_dir / "mark.txt") == 0o600


def test_mark_that_matches_is_accepted(spool_dir):
    check_mark(spool_dir, COMMITMENT)
    check_mark(spool_dir, COMMITMENT, create=True)
    assert (spool_dir / "mark.txt").read_bytes() == MARK


def test_mark_of_64_zeros_is_every_commitments(spool_dir):
    (spool_dir / "mark.txt").chmod(0o600)
    (spool_dir / "mark.txt").write_bytes(MARK.replace(COMMITMENT.encode(), b"0" * 64))
    check_mark(spool_dir, COMMITMENT)
    check_mark(spool_dir, OTHER_COMMITMENT, create=True)
    check_mark(spool_dir, None)


def test_a_commitment_of_64_zeros_takes_no_mark_of_a_real_id(spool_dir):
    rejected(spool_dir, spool.ANY_COMMITMENT, words="is for commitment %s" % COMMITMENT)


def test_mark_of_another_commitment_is_rejected(spool_dir):
    words = "of commitment %s: .* is for commitment %s" % (OTHER_COMMITMENT, COMMITMENT)
    rejected(spool_dir, OTHER_COMMITMENT, create=True, words=words)
    with pytest.raises(spool.UnsafePath, match="is not a spool directory"):
        spool.submit(spool_dir, REQUEST, OTHER_COMMITMENT)
    assert listing(spool_dir) == ["mark.txt", "out"]


def test_missing_mark_in_a_directory_that_is_not_empty_is_rejected(empty_dir):
    (empty_dir / "in").mkdir()
    rejected(empty_dir, COMMITMENT, create=True, words="has no mark file")


def test_missing_mark_is_not_created_unless_asked(empty_dir):
    rejected(empty_dir, COMMITMENT, words="has no mark file")


def test_leftover_temporary_mark_counts_as_empty(empty_dir):
    (empty_dir / "mark.txt.making.0123").write_bytes(b"cp-fol")
    check_mark(empty_dir, COMMITMENT, create=True)
    assert listing(empty_dir) == ["mark.txt", "mark.txt.making.0123"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        MARK[:-1],
        MARK + b"\n",
        MARK.replace(b"spool", b"out"),
        MARK.replace(b"c0", b"C0"),
        MARK.replace(b"\n", b"\r\n"),
    ],
)
def test_malformed_mark_is_rejected(empty_dir, data):
    (empty_dir / "mark.txt").write_bytes(data)
    rejected(empty_dir, COMMITMENT, create=True, words="not that of a spool directory")
    rejected(empty_dir, None, words="not that of a spool directory")


def test_mark_that_is_not_a_regular_file_is_rejected(empty_dir, tmp_path):
    (tmp_path / "elsewhere").write_bytes(MARK)
    os.symlink(tmp_path / "elsewhere", empty_dir / "mark.txt")
    rejected(empty_dir, COMMITMENT, create=True, words="symlink")
    os.unlink(empty_dir / "mark.txt")
    (empty_dir / "mark.txt").mkdir()
    rejected(empty_dir, COMMITMENT, create=True, words="not that of a spool directory")


@pytest.mark.parametrize("winner, accepted", [(COMMITMENT, True), (OTHER_COMMITMENT, False)])
def test_with_two_creators_the_first_mark_counts(empty_dir, monkeypatch, winner, accepted):
    link = os.link

    def link_after_the_other(src, dst, **kw):
        (empty_dir / "mark.txt").write_bytes(MARK.replace(COMMITMENT.encode(), winner.encode()))
        return link(src, dst, **kw)

    monkeypatch.setattr(os, "link", link_after_the_other)
    if accepted:
        check_mark(empty_dir, COMMITMENT, create=True)
    else:
        with pytest.raises(spool.UnsafePath, match="is for commitment " + winner):
            check_mark(empty_dir, COMMITMENT, create=True)
    assert listing(empty_dir) == ["mark.txt"]


def test_mark_that_appears_before_the_directory_is_listed_counts(empty_dir, monkeypatch):
    listdir = os.listdir

    def listdir_after_the_other(path):
        (empty_dir / "mark.txt").write_bytes(MARK)
        return listdir(path)

    monkeypatch.setattr(os, "listdir", listdir_after_the_other)
    check_mark(empty_dir, COMMITMENT, create=True)


def test_without_an_id_only_the_format_is_checked(spool_dir, empty_dir):
    check_mark(spool_dir, None)
    rejected(empty_dir, None, create=True, words="is not a spool directory: it has no mark file")
    words = (
        r"it has no mark file \(mark.txt\): check the path. .* To start anew, "
        "name a new directory as the spool directory; do not empty this one. Nothing in it was"
    )
    rejected(empty_dir, None, create=True, words=words)


@pytest.mark.parametrize(
    "commitment_id", ["", "c0" * 31, "C0" * 32, COMMITMENT + "\n", "../" + "c" * 61]
)
def test_malformed_commitment_id(tmp_path, commitment_id):
    with pytest.raises(ValueError, match="not a commitment ID"):
        spool.submit(tmp_path / "spool", REQUEST, commitment_id)
    assert listing(tmp_path) == []


def test_submit_needs_a_commitment_id(spool_dir):
    with pytest.raises(ValueError, match="needs a commitment ID"):
        spool.submit(spool_dir, REQUEST, None)
    assert listing(spool_dir) == ["mark.txt", "out"]


def test_is_unused(empty_dir):
    with spool.open_private_dir(str(empty_dir)) as fd:
        assert spool.is_unused(fd)
        (empty_dir / "mark.txt.making.0123").write_bytes(b"")
        assert spool.is_unused(fd)
        (empty_dir / "mark.txt").write_bytes(MARK)
        assert not spool.is_unused(fd)


def test_errors_carry_the_full_path(spool_dir):
    (spool_dir / "in").mkdir(mode=0o700)
    (spool_dir / "in" / ".tmp").write_bytes(b"")
    with pytest.raises(spool.UnsafePath, match=str(spool_dir / "in" / ".tmp")):
        spool.submit(spool_dir, REQUEST, COMMITMENT)
    os.unlink(spool_dir / "in" / ".tmp")
    request_id = spool.request_id(REQUEST + bytes(32))
    (spool_dir / "in" / (request_id + ".req")).write_bytes(b"")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "urandom", lambda n: bytes(n))
        with pytest.raises(FileExistsError, match=str(spool_dir / "in" / request_id)):
            spool.submit(spool_dir, REQUEST, COMMITMENT)


# ---- make_private_dirs() and request_stop() ----


def test_make_private_dirs_nests_and_marks_first(tmp_path, monkeypatch):
    seen, mkdir = [], os.mkdir

    def recording_mkdir(name, *args, **kwargs):
        seen.append((str(name), listing(tmp_path)))
        return mkdir(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", recording_mkdir)
    spool.make_private_dirs(tmp_path / "spool", "in", ".tmp", commitment_id=COMMITMENT)
    assert listing(tmp_path / "spool") == ["in", "in/.tmp", "mark.txt"]
    assert seen[1] == ("in", ["spool", "spool/mark.txt"])


@pytest.mark.parametrize(
    "arguments", [{"commitment_id": None}, {"commitment_id": COMMITMENT, "create_mark": False}]
)
def test_make_private_dirs_creates_nothing_that_cannot_get_a_mark(tmp_path, arguments):
    with pytest.raises(FileNotFoundError):
        spool.make_private_dirs(tmp_path / "spool", "out", **arguments)
    assert listing(tmp_path) == []


def test_request_stop(spool_dir):
    spool.request_stop(spool_dir)
    spool.request_stop(spool_dir, COMMITMENT)
    assert (spool_dir / "stop").read_bytes() == b""


def test_request_stop_rejects_another_entry_of_that_name(spool_dir):
    (spool_dir / "stop").write_text("x")
    with pytest.raises(spool.UnsafePath, match="stop is not an empty regular file"):
        spool.request_stop(spool_dir)
    (spool_dir / "stop").unlink()
    (spool_dir / "stop").mkdir()
    with pytest.raises(spool.UnsafePath, match="stop is not an empty regular file"):
        spool.request_stop(spool_dir)


def test_request_stop_asks_again_if_the_service_deletes_an_earlier_stop_file(
    spool_dir, monkeypatch
):
    real_stat, real_open = os.stat, os.open

    def stat_too_late(name, **more):
        if name == "stop":
            os.unlink(name, dir_fd=more["dir_fd"])  # the service deleted it first
        return real_stat(name, **more)

    monkeypatch.setattr(os, "stat", stat_too_late)
    (spool_dir / "stop").touch()
    spool.request_stop(spool_dir)
    assert (spool_dir / "stop").read_bytes() == b""

    def open_too_late(name, *more, **still_more):
        if name == "stop":
            (spool_dir / "stop").touch()  # another caller created it first
        return real_open(name, *more, **still_more)

    monkeypatch.setattr(os, "open", open_too_late)
    with pytest.raises(OSError, match="deleted twice while it was being checked"):
        spool.request_stop(spool_dir)


def test_request_stop_needs_the_mark(spool_dir, empty_dir):
    for path, commitment_id in (
        (empty_dir, None),
        (empty_dir, COMMITMENT),
        (spool_dir, OTHER_COMMITMENT),
    ):
        before = listing(path)
        with pytest.raises(spool.UnsafePath, match="is not a spool directory"):
            spool.request_stop(path, commitment_id)
        assert listing(path) == before


# ---- Responses ----


def test_read_response_with_a_proof(spool_dir):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    response = spool.read_response(spool_dir, ID, key=KEY)
    assert response == (
        ID,
        False,
        1000 * 10**9,
        {
            "y": {"size": 4, "sha256": hashlib.sha256(Y).hexdigest()},
            "proof": {
                "size": len(PROOF),
                "sha256": spool.sha256_file(spool.response_path(spool_dir, ID, "proof")),
            },
        },
        [],
    )
    assert response._fields == ("id", "failed", "tick_ns", "files", "faults")
    without = spool.read_response(spool_dir, ID, hashes=False, key=KEY)
    assert (without.files, without.faults) == (
        {"y": {"size": 4}, "proof": {"size": len(PROOF)}},
        [],
    )


def test_read_response_without_a_proof(spool_dir):
    write_response(spool_dir, {"failed": b""})
    response = spool.read_response(spool_dir, ID, key=KEY)
    assert (response.failed, response.tick_ns, response.faults) == (True, 1000 * 10**9, [])
    assert response.files == {"failed": {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()}}


@pytest.mark.parametrize("files", [{}, {"y": Y}])
def test_read_response_before_the_response_is_complete(spool_dir, files):
    write_response(spool_dir, files, tag=bytes(32) if files else None)
    assert spool.read_response(spool_dir, ID, key=KEY) == (
        ID,
        False,
        0,
        {},
        ["there is no response"],
    )
    assert spool.finished_requests(spool_dir) == set()


def test_read_response_without_an_out_directory(tmp_path):
    assert spool.read_response(tmp_path, ID).faults == ["there is no response"]
    assert spool.finished_requests(tmp_path) == set()


@pytest.mark.parametrize(
    "files, tag, fault",
    [
        (
            {"y": Y, "proof": PROOF, "failed": b""},
            "right",
            "unexpected set of files: y, tag, proof, failed",
        ),
        ({"proof": PROOF}, bytes(32), "unexpected set of files: tag, proof"),
        ({"y": Y, "proof": PROOF}, None, "unexpected set of files: y, proof"),
        ({"failed": b""}, None, "unexpected set of files: failed"),
        ({"y": Y + b"\0", "proof": PROOF}, "right", "y has 5 bytes"),
        ({"y": Y, "proof": PROOF}, bytes(31), "tag has 31 bytes"),
        ({"failed": b"x"}, "right", "failed has 1 bytes"),
    ],
)
def test_read_response_faults(spool_dir, files, tag, fault):
    write_response(spool_dir, files, tag)
    assert spool.read_response(spool_dir, ID, key=KEY).faults == [fault]
    assert spool.read_response(spool_dir, ID).faults == [fault, NO_KEY]


def test_read_response_rejects_different_modification_times(spool_dir):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    os.utime(spool_dir / "out" / (ID + ".y"), (1001, 1001))
    response = spool.read_response(spool_dir, ID, key=KEY)
    assert response.faults == ["the files have different modification times"]
    assert response.tick_ns == 1000 * 10**9


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo", "socket"])
def test_read_response_rejects_files_that_are_not_regular(spool_dir, tmp_path, monkeypatch, kind):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    path = spool_dir / "out" / (ID + ".proof")
    os.rename(path, tmp_path / "proof")
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        os.symlink(tmp_path / "proof", path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        monkeypatch.chdir(path.parent)  # a socket's path must be short
        with socket.socket(socket.AF_UNIX) as s:
            s.bind(path.name)
    response = spool.read_response(spool_dir, ID, key=KEY)
    assert response.faults == ["proof is not a regular file"] and "proof" not in response.files
    with spool.open_private_dir(str(spool_dir / "out")) as out_fd:
        with spool.open_response_at(out_fd, ID) as (response, fds):
            assert sorted(fds) == ["tag", "y"]


def test_read_response_rejects_a_directory_that_is_not_private(spool_dir):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    for path in (spool_dir / "out", spool_dir):
        os.chmod(path, 0o755)
        with pytest.raises(spool.UnsafePath, match="accessible to other users"):
            spool.read_response(spool_dir, ID)
        os.chmod(path, 0o700)
    assert spool.read_response(spool_dir, ID, key=KEY).faults == []


def test_response_files_are_opened_last_published_first_and_deleted_in_that_order(
    spool_dir, monkeypatch
):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    calls, os_open, unlink = [], os.open, os.unlink

    def recording_open(path, *args, **kwargs):
        calls.append("open " + str(path).rpartition(".")[2])
        return os_open(path, *args, **kwargs)

    def recording_unlink(path, **kwargs):
        calls.append("unlink " + path.rpartition(".")[2])
        return unlink(path, **kwargs)

    with spool.open_private_dir(str(spool_dir / "out")) as out_fd:
        monkeypatch.setattr(os, "open", recording_open)
        monkeypatch.setattr(os, "unlink", recording_unlink)
        with spool.open_response_at(out_fd, ID):
            pass
        spool.delete_response_at(out_fd, ID)
    assert calls[:4] == ["open failed", "open proof", "open tag", "open y"]
    assert calls[4:] == ["unlink proof", "unlink failed", "unlink tag", "unlink y"]


@pytest.mark.parametrize("files", [{"y": Y, "proof": PROOF}, {"failed": b""}])
def test_read_response_compares_the_tag_and_a_missing_key_is_a_fault(spool_dir, files):
    write_response(spool_dir, files, tag=bytes(32))
    fault = "the tag does not match the request ID and the response's bytes"
    assert spool.read_response(spool_dir, ID, key=KEY).faults == [fault]
    assert spool.read_response(spool_dir, ID, hashes=False, key=KEY).faults != []
    assert spool.read_response(spool_dir, ID).faults == [NO_KEY]


def test_read_response_detects_swapped_responses(spool_dir):
    other = "cd" * 32
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    write_response(spool_dir, {"y": Y, "proof": b"another client's proof"}, request_id=other)
    for suffix in ("y", "tag", "proof"):
        a, b = (spool.response_path(spool_dir, name, suffix) for name in (ID, other))
        os.rename(a, a + ".x")
        os.rename(b, a)
        os.rename(a + ".x", b)
    assert spool.read_response(spool_dir, ID, key=KEY).faults != []
    assert spool.read_response(spool_dir, other, key=KEY).faults != []


@pytest.mark.parametrize("name", ["", "../in/x", "ab" * 31, "AB" * 32, ID + "\n", ID + ".proof"])
def test_a_request_id_must_be_64_hex_digits(spool_dir, name):
    with spool.open_private_dir(str(spool_dir / "out")) as out_fd:
        for call in (
            lambda: spool.read_response(spool_dir, name),
            lambda: spool.response_path(spool_dir, name, "proof"),
            lambda: spool.open_response_at(out_fd, name).__enter__(),
            lambda: spool.delete_response_at(out_fd, name),
        ):
            with pytest.raises(ValueError, match="not a request ID"):
                call()


def test_response_path(spool_dir):
    assert spool.response_path("s", ID, "tag") == "s/out/%s.tag" % ID
    with pytest.raises(ValueError):
        spool.response_path("s", ID, "req")


def test_finished_requests_and_waiting(spool_dir, monkeypatch):
    other, third = "cd" * 32, "ef" * 32
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    write_response(spool_dir, {"failed": b""}, request_id=other)
    (spool_dir / "out" / "notes.proof").write_bytes(b"")
    assert spool.finished_requests(spool_dir) == {ID, other}
    assert spool.wait_for_responses(spool_dir, [ID, other], timeout_s=60) == {ID, other}
    assert spool.wait_for_responses(spool_dir, [ID, third], timeout_s=0.1, poll_s=0.01) == {ID}
    monkeypatch.setattr(spool.time, "sleep", lambda s: pytest.fail("it went on waiting"))
    assert (
        spool.wait_for_responses(spool_dir, [third], timeout_s=60, keep_waiting=lambda: False)
        == set()
    )


def test_open_response_at(spool_dir):
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    with spool.open_private_dir(str(spool_dir / "out")) as out_fd:
        with spool.open_response_at(out_fd, ID) as (response, fds):
            assert (response.failed, response.faults) == (False, [])
            assert response.files == {"y": {"size": 4}, "proof": {"size": len(PROOF)}}
            assert sorted(fds) == ["proof", "tag", "y"] and os.read(fds["proof"], 99) == PROOF
            kept = fds.pop("y")
        assert os.read(kept, 9) == Y
        os.close(kept)
        with pytest.raises(OSError):
            os.fstat(fds["proof"])
        with spool.open_response_at(out_fd, "cd" * 32) as (response, fds):
            assert (response, fds) == (None, {})


def test_delete_response_at(spool_dir, tmp_path):
    other = "cd" * 32
    write_response(spool_dir, {"y": Y, "proof": PROOF})
    write_response(spool_dir, {"failed": b""}, request_id=other)
    (tmp_path / "target").write_bytes(b"kept")
    with spool.open_private_dir(str(spool_dir / "out")) as out_fd:
        spool.delete_response_at(out_fd, ID)
        spool.delete_response_at(out_fd, ID)
        assert sorted(os.listdir(out_fd)) == [other + ".failed", other + ".tag"]
        os.symlink(tmp_path / "target", spool_dir / "out" / (other + ".y"))
        (spool_dir / "out" / (other + ".proof")).mkdir()
        with pytest.raises(OSError, match=other + ".proof"):
            spool.delete_response_at(out_fd, other)
        assert os.listdir(out_fd) == [other + ".proof"]
    assert (tmp_path / "target").read_bytes() == b"kept"


# ---- sweep() ----


def test_sweep_by_age(spool_dir):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    write_response(spool_dir, {"failed": b""}, request_id="cd" * 32, mtime_s=int(time.time()))
    assert spool.sweep(spool_dir, COMMITMENT, days=1) == 1 and not (spool_dir / "in").exists()
    assert [name for name in listing(spool_dir / "out") if name.startswith(ID)] == []
    assert len(listing(spool_dir / "out")) == 2
    # True equals 1, but sweep() rejects a bool.
    for days in (0, None, 0.5, True):
        with pytest.raises(ValueError, match="1 or more"):
            spool.sweep(spool_dir, COMMITMENT, days=days)


def test_sweep_leaves_other_names_and_a_name_that_vanishes(spool_dir, monkeypatch):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    # Equally old files whose names are not a response's (a request ID is lower-case hex).
    for name in ("notes.proof", ID.upper() + ".proof", ID + ".proof.bak"):
        (spool_dir / "out" / name).write_bytes(b"")
        os.utime(spool_dir / "out" / name, (1000, 1000))
    real = os.stat

    # As if ID.proof had been deleted between sweep()'s os.listdir() and its os.stat().
    def stat_of_a_name_that_has_gone(name, **more):
        if str(name) == ID + ".proof":
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", name)
        return real(name, **more)

    before = listing(spool_dir)
    monkeypatch.setattr(spool.os, "stat", stat_of_a_name_that_has_gone)
    assert spool.sweep(spool_dir, COMMITMENT, days=1) == 0 and listing(spool_dir) == before
    monkeypatch.undo()
    assert spool.sweep(spool_dir, COMMITMENT, days=1) == 1
    assert sorted(set(before) - set(listing(spool_dir))) == [
        "out/%s.%s" % (ID, suffix) for suffix in ("proof", "tag", "y")
    ]


def test_sweep_checks_the_directory_out(spool_dir):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    (spool_dir / "out").chmod(0o755)
    before = listing(spool_dir)
    with pytest.raises(spool.UnsafePath, match="mode 0755"):
        spool.sweep(spool_dir, COMMITMENT, days=1)
    assert listing(spool_dir) == before


def test_subdirectories_are_checked_too(spool_dir):
    spool.submit(spool_dir, REQUEST, COMMITMENT)
    (spool_dir / "in" / ".tmp").chmod(0o755)
    before = listing(spool_dir)
    with pytest.raises(spool.UnsafePath, match="mode 0755"):
        spool.submit(spool_dir, REQUEST, COMMITMENT)
    with pytest.raises(spool.UnsafePath, match="mode 0755"):
        spool.make_private_dirs(spool_dir, "in", ".tmp", commitment_id=COMMITMENT)
    assert listing(spool_dir) == before


# An entry named "hold", of any kind, puts a spool directory on hold: see spool.is_on_hold().
@pytest.mark.parametrize("kind", ["file", "directory", "dangling link"])
def test_nothing_is_submitted_or_swept_while_the_directory_is_on_hold(spool_dir, kind):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    hold = spool_dir / "hold"
    {"file": hold.touch, "directory": hold.mkdir}.get(kind, lambda: hold.symlink_to("nothing"))()
    before = listing(spool_dir)
    with pytest.raises(spool.OnHold, match="hold exists.: nothing was deleted"):
        spool.sweep(spool_dir, COMMITMENT, days=1)
    with pytest.raises(spool.OnHold, match="hold exists.: nothing was submitted"):
        spool.submit(spool_dir, REQUEST, COMMITMENT)
    with pytest.raises(spool.OnHold, match="hold exists.: no response was read"):
        spool.read_response(spool_dir, ID, key=KEY)
    with pytest.raises(spool.OnHold, match="hold exists.: no response was listed"):
        spool.finished_requests(spool_dir)
    with pytest.raises(spool.OnHold, match="hold exists.: no response was listed"):
        spool.wait_for_responses(spool_dir, [ID], timeout_s=60)
    code, out, err = run("sweep", spool_dir, COMMITMENT, "1")
    assert (code, out) == (1, "") and err.startswith("error: the spool directory is on hold (")
    assert listing(spool_dir) == before
    hold.rmdir() if kind == "directory" else hold.unlink()
    assert spool.sweep(spool_dir, COMMITMENT, days=1) == 1


def test_look_for_the_hold_that_fails_is_not_taken_as_no_hold(spool_dir, monkeypatch):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    before = listing(spool_dir)
    real, holds = {"stat": os.stat, "lstat": os.lstat}, os.stat(spool_dir)

    # is_on_hold() looks with os.stat(), finished_requests() with os.lstat(): both fail for "hold"
    # in the spool directory, not for a directory of that name higher up.
    def failing(which):
        def look(path, **more):
            above, name = os.path.split(os.fspath(path))
            if name == "hold":
                found = real["stat"](above) if above else os.fstat(more["dir_fd"])
                if (found.st_dev, found.st_ino) == (holds.st_dev, holds.st_ino):
                    raise OSError(errno.EIO, "Input/output error")
            return real[which](path, **more)

        return look

    for which in real:
        monkeypatch.setattr(spool.os, which, failing(which))
    with spool.open_private_dir(str(spool_dir)) as spool_fd:
        with pytest.raises(OSError, match="Input/output error"):
            spool.is_on_hold(spool_fd)
    for call in (
        lambda: spool.read_response(spool_dir, ID, key=KEY),
        lambda: spool.finished_requests(spool_dir),
        lambda: spool.wait_for_responses(spool_dir, [ID], timeout_s=60),
        lambda: spool.submit(spool_dir, REQUEST, COMMITMENT),
        lambda: spool.sweep(spool_dir, COMMITMENT, days=1),
    ):
        with pytest.raises(OSError, match="Input/output error"):
            call()
    monkeypatch.undo()
    assert listing(spool_dir) == before


def test_hold_that_comes_while_a_script_waits_ends_the_waiting(spool_dir):
    looks = []

    def put_on_hold_at_the_second_look():
        looks.append(None)
        if len(looks) == 2:
            (spool_dir / "hold").touch()
        return True

    with pytest.raises(spool.OnHold, match="no response was listed"):
        spool.wait_for_responses(
            spool_dir, [ID], 60, poll_s=0.01, keep_waiting=put_on_hold_at_the_second_look
        )
    assert len(looks) == 2


def test_hold_is_told_also_where_there_is_no_directory_out(empty_dir):
    spool.make_private_dirs(empty_dir, commitment_id=COMMITMENT)
    (empty_dir / "hold").touch()
    with pytest.raises(spool.OnHold, match="no response was read"):
        spool.read_response(empty_dir, ID, key=KEY)


def test_hold_that_comes_during_a_sweep_stops_it(spool_dir, monkeypatch):
    for request_id in (ID, "cd" * 32):
        write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, request_id=request_id)
    real = spool.delete_response_at

    # The hold appears as the first response is deleted; sweep() checks for it before each one.
    def delete_and_put_on_hold(out_fd, request_id):
        (spool_dir / "hold").touch()
        return real(out_fd, request_id)

    monkeypatch.setattr(spool, "delete_response_at", delete_and_put_on_hold)
    with pytest.raises(spool.OnHold, match="nothing more was deleted"):
        spool.sweep(spool_dir, COMMITMENT, days=1)
    assert len([name for name in listing(spool_dir / "out") if name.endswith(".proof")]) == 1


def test_message_after_a_fault_in_marking_does_not_say_that_nothing_was_written(
    empty_dir, monkeypatch
):
    real = os.fsync

    # mark.txt has already been created when its directory is synced.
    def fsync_that_fails_on_a_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "Input/output error")
        real(fd)

    monkeypatch.setattr(spool.os, "fsync", fsync_that_fails_on_a_directory)
    with pytest.raises(spool.UnsafePath, match="cannot be read or created") as caught:
        spool.submit(empty_dir, REQUEST, COMMITMENT)
    assert "Nothing in it was written" not in str(caught.value)


def test_sweep_skips_a_request_that_the_service_is_working_on(spool_dir):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    for directory in ("in", "work"):
        (spool_dir / directory).mkdir(mode=0o700)
        (spool_dir / directory / (ID + ".req")).write_bytes(b"")
        before = listing(spool_dir)
        assert spool.sweep(spool_dir, COMMITMENT, days=1) == 0
        assert listing(spool_dir) == before
        (spool_dir / directory / (ID + ".req")).unlink()
    assert spool.sweep(spool_dir, COMMITMENT, days=1) == 1


@pytest.mark.parametrize("mark", [None, MARK.replace(b"c0", b"d1")])
def test_sweep_checks_the_mark_file_before_it_deletes(spool_dir, mark):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"}, mtime_s=1000)
    (spool_dir / "mark.txt").unlink()
    if mark:
        (spool_dir / "mark.txt").write_bytes(mark)
    before = listing(spool_dir)
    with pytest.raises(spool.UnsafePath, match="is not a spool directory"):
        spool.sweep(spool_dir, COMMITMENT, days=1)
    assert listing(spool_dir) == before
    with pytest.raises(ValueError, match="commitment ID"):
        spool.sweep(spool_dir, None, days=1)


# ---- The command line ----


def run(*arguments):
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)], capture_output=True, text=True
    )
    return done.returncode, done.stdout, done.stderr


def test_command_submit(tmp_path):
    (tmp_path / "request").write_bytes(REQUEST)
    code, out, err = run("submit", tmp_path / "spool", tmp_path / "request", COMMITMENT)
    assert (code, err) == (0, "") and spool.REQUEST_ID.match(out.strip())
    assert (tmp_path / "spool" / "in" / (out.strip() + ".req")).read_bytes()[:-32] == REQUEST
    assert run("submit", tmp_path / "spool", tmp_path / "request", COMMITMENT.upper())[0] == 0
    code, out, err = run("submit", tmp_path / "spool", tmp_path / "request", OTHER_COMMITMENT)
    assert (
        (code, out) == (1, "")
        and err.startswith("error: ")
        and "is not a spool directory" in err
        and err.count("\n") == 1
    )
    code, out, err = run("submit", tmp_path / "spool", tmp_path / "missing", COMMITMENT)
    assert (code, out, err) == (
        1,
        "",
        "error: %s: No such file or directory\n" % (tmp_path / "missing"),
    )


def test_command_sweep(spool_dir):
    write_response(spool_dir, {"y": bytes(4), "proof": b"p"})
    assert run("sweep", spool_dir, COMMITMENT, "1") == (0, "1\n", "")
    assert run("sweep", spool_dir, COMMITMENT.upper(), "1") == (0, "0\n", "")
    assert run("sweep", spool_dir, COMMITMENT)[0] == 2
    assert run("sweep", spool_dir, OTHER_COMMITMENT, "1")[0] == 1
    spool_dir.chmod(0o750)
    code, out, err = run("sweep", spool_dir, COMMITMENT, "1")
    assert (code, out) == (1, "") and err.startswith("error: ") and "mode 0750" in err
    for days in ("0", "0.5", "-1", "nan", "inf", "x"):
        assert run("sweep", spool_dir, COMMITMENT, days)[0] == 2


def test_command_says_how_to_go_on_at_another_account_s_directory(tmp_path, monkeypatch, capsys):
    theirs = tmp_path / "theirs"
    theirs.mkdir(mode=0o700)
    (tmp_path / "request").write_bytes(REQUEST)
    refusal = "%s is owned by uid 12345, not by the current user" % theirs
    told = "error: %s%s\n" % (refusal, spool.AS_THAT_ACCOUNT)
    assert "without sudo if it is your own, else sudo -u ACCOUNT ..." in told
    if os.geteuid() == 0:
        os.chown(theirs, 12345, -1)
        assert run("submit", theirs, tmp_path / "request", COMMITMENT) == (1, "", told)
        assert run("sweep", theirs, COMMITMENT, "1") == (1, "", told)
        assert listing(theirs) == []

    def refuse(*arguments):
        raise spool.UnsafePath(words)

    monkeypatch.setattr(spool, "sweep", refuse)
    for words, expected in ((refusal, told), ("another refusal", "error: another refusal\n")):
        assert spool.main(["sweep", str(theirs), COMMITMENT, "1"]) == 1
        assert capsys.readouterr() == ("", expected)


def test_command_make_request(tmp_path):
    (tmp_path / "prompt").write_text("7 8\n")
    (tmp_path / "reply").write_text("[1, 2,\n3]\n")
    files = (tmp_path / "prompt", tmp_path / "reply")
    code, out, err = run("make-request", 8, *files, 99, tmp_path / "a")
    assert (code, out) == (1, "") and "does not end with the end token 99" in err
    assert run("make-request", 8, *files, 3, tmp_path / "a") == (0, "1:4\n", "")
    assert stat.S_IMODE((tmp_path / "a").stat().st_mode) == 0o600
    assert (tmp_path / "a").read_bytes() == spool.whole_reply([7, 8], [1, 2, 3], 8, 3)
    assert run("make-request", 8, *files, "none", tmp_path / "b") == (0, "1:4\n", "")
    taken = run("make-request", 8, *files, "none", tmp_path / "a")
    assert taken == (1, "", "error: %s already exists (choose a new name)\n" % (tmp_path / "a"))
    assert (tmp_path / "a").read_bytes() == spool.whole_reply([7, 8], [1, 2, 3], 8, 3)
    code, out, err = run("make-request", 4, *files, "none", tmp_path / "c")
    assert (code, out) == (1, "") and err.startswith("error: ") and "do not fit" in err
    assert not (tmp_path / "c").exists()
    # "²".isdigit() is true, but int() rejects it.
    code, out, err = run("make-request", 8, *files, "²", tmp_path / "c")
    assert code == 2 and 'neither a token ID nor "none"' in err
    (tmp_path / "reply").write_bytes(spool.pack([1, 2, 3]))
    code, out, err = run("make-request", 8, *files, "none", tmp_path / "c")
    assert (code, out) == (1, "") and "does not contain token IDs" in err


def test_command_make_request_from_a_json_object(tmp_path):
    served = {"reply_token_ids": [1, 2, 3], "prompt_token_ids": [7, 8], "text": "\u00e9", "n": 1}
    file = tmp_path / "served.json"
    file.write_text(json.dumps(served, indent=1, ensure_ascii=False), encoding="utf-8")
    assert run("make-request", 8, file, file, 3, tmp_path / "a") == (0, "1:4\n", "")
    assert (tmp_path / "a").read_bytes() == spool.whole_reply([7, 8], [1, 2, 3], 8, 3)
    (tmp_path / "prompt").write_text("9")  # one may be a plain list
    assert run("make-request", 8, tmp_path / "prompt", file, 3, tmp_path / "b") == (0, "0:3\n", "")
    assert (tmp_path / "b").read_bytes() == spool.whole_reply([9], [1, 2, 3], 8, 3)
    for text in (
        "{",
        '{"prompt_token_ids": [7, 8]}',
        '{"prompt_token_ids": [7, 8], "reply_token_ids": "123"}',
        '{"prompt_token_ids": [7, 8], "reply_token_ids": [1, 2.0]}',
        '{"prompt_token_ids": [7, 8], "reply_token_ids": [1, true]}',
        '{"prompt_token_ids": [7, 8], "reply_token_ids": {"0": 1}}',
        '{"prompt_token_ids": [7, 8], "reply_token_ids": ' + "[" * 100000,
    ):
        file.write_text(text)
        code, out, err = run("make-request", 8, tmp_path / "prompt", file, "none", tmp_path / "c")
        message = 'served.json is not a JSON object with a list of integers "reply_token_ids"\n'
        assert (code, out) == (1, "") and err.endswith(message), text
    file.write_text('{"prompt_token_ids": [7, -1], "reply_token_ids": [4294967296]}')
    code, out, err = run("make-request", 8, file, file, "none", tmp_path / "c")
    assert (code, out) == (1, "") and "must be integers from 0 to 4294967295" in err
    assert not (tmp_path / "c").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("submit",),
        ("submit", "a", "b"),
        ("unknown", "a", "b"),
        ("make-request", "8", "p", "r", "x"),
        ("make-request", "8", "p", "r", "no", "x"),
        ("make-request", "8", "p", "r", "²", "x"),
    ],
)
def test_command_usage_errors(arguments):
    code, out, err = run(*arguments)
    assert (code, out) == (2, "") and err.startswith("usage: ") and len(err.splitlines()) <= 5


@pytest.mark.parametrize(
    "arguments",
    [("--help",), ("submit", "--help"), ("sweep", "--help"), ("make-request", "--help")],
)
def test_command_help_is_short(arguments):
    code, out, err = run(*arguments)
    assert (code, err) == (0, "") and len(out.splitlines()) <= 60
