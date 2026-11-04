#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The front end's side of a spool directory, where it and the proving service exchange files.

The front end is the program that takes clients' requests, such as a server's middleware.

    in/ID.req               a request file that waits for the proving service (cp_prove)
    work/ID.req             the same file while the service works on it
    out/ID.y, .proof, .tag  the response: next token, proof, tag; or out/ID.failed, .tag: no proof
    stop                    an empty file that asks the service to exit
    mark.txt                the mark file: that this is a spool directory, and, if a front end
                            made it first, of which commitment
    hold                    a sign: submit nothing, hand out no response: is_on_hold()

The service checks no finished proof, and a proof that cp_verify rejects can hold values that are
not masked. So whatever reads a response here hands it on only after cp_verify has passed every
proof of its batch. proof_middleware.py does that; these functions do not.

A script calls make_request() or whole_reply(), submit(with_key=True), wait_for_responses() and
read_response(key=key). A front end opens the directory once and works relative to the descriptor:
open_private_dir(), check_mark(), submit_at(), open_response_at(), delete_response_at(). The front
end and the service run under one account.

From a shell: python3 integrations/spool.py --help. Needs Linux and only the standard library."""

import argparse
import collections
import contextlib
import errno
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import struct
import sys
import time

RESPONSE_FILES = ("y", "proof")  # what a client downloads
FAILED = "failed"
TAG = "tag"
TAG_BYTES = 32
NONCE_BYTES = 32
REQUEST_ID = re.compile(r"[0-9a-f]{64}\Z")
COMMITMENT_ID = re.compile(r"[0-9a-f]{64}\Z")

# One response, as read_response() and open_response_at() return it.
#   failed   True if the response is ID.failed: the service made no proof
#   tick_ns  the modification time of ID.proof or ID.failed: when the service released the batch
#   files    {suffix: {"size": bytes, "sha256": hex digits}}, without the tag. ID.y is the next
#            token as a little-endian 32-bit word; 0 if the checked range ends earlier
#   faults   what is wrong, as strings. Use the response only if there is none, the tag was
#            compared and cp_verify has passed every proof of its batch
Response = collections.namedtuple("Response", "id failed tick_ns files faults")


class UnsafePath(Exception):
    """A directory or file has the wrong type, owner, permissions or mark file."""


class OpenAbove(UnsafePath):
    """Another account may be able to change a directory above the one that was named."""


class OnHold(Exception):
    """The spool directory is on hold: see is_on_hold()."""


def _check_request_id(request_id):
    # A client's string such as "../in/x" would name a file outside out/.
    if not REQUEST_ID.match(request_id):
        raise ValueError("%r is not a request ID (64 lower-case hex digits)" % (request_id,))


def _check_commitment_id(commitment_id):
    if commitment_id is not None and not COMMITMENT_ID.match(commitment_id):
        raise ValueError("%r is not a commitment ID (64 lower-case hex digits)" % (commitment_id,))


# ---- Requests ----
#
# A request file is the bytes of make_request() followed by a nonce of 32 random bytes, which
# submit() adds; the verifier gets the request without it. The file's name is its request ID, a hash
# of its bytes, so only someone who knows them all, or can list the spool's directories, can name
# the response. The service gives no response to a file whose name is not that hash or whose length
# is not that of its commitment's requests.
#
# The service computes the same IDs, keys and tags (prover/prover/spool.h). The "p258-spool-"
# strings are fixed labels that both sides share; they separate the uses of one hash function, and
# the number 258 has no meaning.


def pack(words):
    """Return a list of numbers as bytes, each as a little-endian 32-bit word."""
    return struct.pack("<%dI" % len(words), *words)


def read_words(path):
    """Return the little-endian 32-bit words of a file as a list. Raise ValueError if the file's
    length is not a whole number of 32-bit words."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4:
        raise ValueError("%s: %d bytes is not a whole number of 32-bit words" % (path, len(data)))
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def make_request(tokens, start, end):
    """Return the bytes of a request, without its nonce: the token IDs, then start and end.

    At position t the model reads tokens 0 to t, and the proof checks token t + 1. The checked
    range is the positions start to end - 1; if end == len(tokens), the last one checks the next
    token, which the response reports in ID.y. A proof covers only the checked range, so start
    and end have no defaults. Raise ValueError unless 0 <= start < end <= len(tokens) and every
    token ID, start and end is an integer from 0 to 4294967295. If tokens cannot be iterated, raise
    TypeError instead; a start or end that is not an integer can raise TypeError or
    OverflowError."""
    tokens = list(tokens)
    if not 0 <= start < end <= len(tokens):
        raise ValueError(
            "the range %d:%d does not fit a request of %d tokens" % (start, end, len(tokens))
        )
    try:
        return pack(tokens + [start, end])
    except struct.error:
        raise ValueError("token IDs, start and end must be integers from 0 to 4294967295") from None


def request_fields(data):
    """Return (token IDs, start, end) of a request's bytes, without its nonce. Raise ValueError
    unless their length is a multiple of 4 and at least 12 bytes."""
    if len(data) % 4 or len(data) < 12:
        raise ValueError("%d bytes are not a request" % len(data))
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    return words[:-2], words[-2], words[-1]


def whole_reply(prompt, reply, length, end_token=None, fill=0):
    """Return the bytes of a request, without its nonce, whose checked range is the whole reply.

    The prompt and the reply are padded with 'fill' to 'length' tokens, which must be 2**lt,
    where lt is a line of the parameter file (policy.txt).

    A proof does not show where a reply should end: a truncated reply verifies as a shorter
    reply. So pass the model's end token: the reply must end with it and not contain it earlier
    (ValueError), and the checked range then covers the position at which the reply ends. Omit
    end_token only if your own length limit stopped the reply."""
    reply = list(reply)
    if end_token is not None and end_token in reply[:-1]:
        raise ValueError("the reply contains the end token %d before its last token" % end_token)
    if end_token is not None and reply[-1:] != [end_token]:
        raise ValueError(
            "the reply does not end with the end token %d (add it to the reply if the server left "
            "it out; name no end token if your own length limit stopped the reply)" % end_token
        )
    n, m = len(prompt), len(reply)
    if n == 0 or m == 0:
        raise ValueError("the prompt and the reply must each have at least one token")
    if n + m > length:
        raise ValueError(
            "a prompt of %d and a reply of %d tokens do not fit a request of %d tokens"
            % (n, m, length)
        )
    tokens = list(prompt) + reply + [fill] * (length - n - m)
    return make_request(tokens, n - 1, n + m - 1)


def request_id(data):
    """Return the request ID of a request file's bytes (the request, then the nonce)."""
    return hashlib.sha256(b"p258-spool-req-1" + data).hexdigest()


# ---- Tags ----
#
# The service writes ID.tag next to every response: an HMAC-SHA-256 over the request ID and the
# response's bytes, under a key derived from the request's nonce. With it the front end detects a
# response stored under another request's ID before it sends one client's data to another. The tag
# does not show that a proof is valid, and does not protect against anyone who can read the request
# file, which contains the nonce.


def tag_key(data):
    """Return the 32-byte key of a request's tag, from the request file's bytes or just the nonce.

    Store it with the request ID. Never log it. Raise ValueError if data has fewer than 32 bytes."""
    if len(data) < NONCE_BYTES:
        raise ValueError(
            "expected a request file's bytes or its 32-byte nonce, got %d bytes" % len(data)
        )
    return hmac.new(bytes(data[-NONCE_BYTES:]), b"p258-spool-tag-1", hashlib.sha256).digest()


def tag_hmac(key, request_id, y, proof_size):
    """Return the HMAC object of response_tag(), fed with everything before the proof.

    For a long proof: update() with each chunk, check that proof_size bytes were fed, then
    digest()."""
    head = bytes.fromhex(request_id) + struct.pack("<2Q", len(y), proof_size) + y
    return hmac.new(key, head, hashlib.sha256)


def response_tag(key, request_id, y, proof):
    """Return the tag of a response with a proof.

    The HMAC covers the request ID's 32 raw bytes, the lengths of y and of the proof as
    little-endian 64-bit words, y, and the proof."""
    mac = tag_hmac(key, request_id, y, len(proof))
    mac.update(proof)
    return mac.digest()


def failed_tag(key, request_id):
    """Return the tag of a response without a proof.

    The HMAC covers the request ID's 32 raw bytes, 16 zero bytes and the ASCII string "failed".
    It cannot collide with response_tag(): an input that starts with the same 48 bytes has both
    lengths 0 and ends there."""
    return hmac.new(key, bytes.fromhex(request_id) + bytes(16) + b"failed", hashlib.sha256).digest()


# ---- Directories and files that are safe to use ----
#
# Threat model: other local users must not read clients' prompts, replies and proofs, or make the
# front end use their files. Root and the front end's account are trusted; under sudo, see
# trusted_owners().
#
# A directory for clients' data that is named by a path is refused if another account may be able to
# change a directory above it: others_can_change(). That is one look by path, which can be raced: so
# the directory is then opened once and checked on its descriptor, and everything in it is named
# relative to that: what was checked is what is used. Only the last directory of a path is created.

# cp_commit writes these files into a secret key directory only.
SECRET_KEY_FILES = ("seed.bin", "isk.txt")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# O_NONBLOCK: opening a FIFO must not block.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


# fmt: off

# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")

# fmt: on


def _full_path(name, dir_fd):
    if dir_fd is None:
        return name
    try:
        return os.path.normpath(os.path.join(os.readlink("/proc/self/fd/%d" % dir_fd), name))
    except OSError:
        return name


@contextlib.contextmanager
def _errors_with_full_path(name, dir_fd):
    try:
        yield
    except OSError as error:
        raise OSError(error.errno, error.strerror, _full_path(name, dir_fd)) from None


def _open(name, flags, dir_fd):
    """os.open() for reading. A symlink raises UnsafePath; other errors carry the full path."""
    if dir_fd is not None and ("/" in name or name in ("", "..")):
        raise ValueError("%r is not a single file name" % (name,))
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as error:
        path = _full_path(name, dir_fd)
        if flags & os.O_DIRECTORY and error.errno in (errno.ENOTDIR, errno.ELOOP):
            raise UnsafePath("%s is a symlink or not a directory" % path) from None
        if error.errno == errno.ELOOP:
            raise UnsafePath("%s is a symlink" % path) from None
        raise OSError(error.errno, error.strerror, path) from None


def is_secret_key_dir(dir_fd):
    """Return whether an open directory has a file that only a secret key directory has."""
    for name in SECRET_KEY_FILES:
        with contextlib.suppress(OSError):
            os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            return True
    return False


def _confirm_owner(fd, path):
    """Raise UnsafePath if a file or directory only appears to be owned by the current user.

    A user namespace shows every unmapped owner as the same uid, which can be the current user's.
    The local kernel lets only the real owner, or a process privileged over that uid, set
    O_NOATIME on a descriptor, which changes nothing in the file system."""
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NOATIME)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    except PermissionError:
        raise UnsafePath("%s is not owned by the current user" % path) from None
    except OSError as error:
        raise OSError(error.errno, error.strerror, path) from None


def _check_private(fd, path):
    """Raise UnsafePath unless only the current user has access to an open directory."""
    info = os.fstat(fd)
    if info.st_uid != os.geteuid():
        raise UnsafePath("%s is owned by uid %d, not by the current user" % (path, info.st_uid))
    if info.st_mode & 0o077:
        raise UnsafePath(
            "%s is accessible to other users (mode %04o, not 0700): choose a new directory"
            % (path, stat.S_IMODE(info.st_mode))
        )
    _confirm_owner(fd, path)


# Others may have written to the file or directory already, and what passes this check is trusted.
# chmod does not stop a process that already has a file open for writing: so a new copy is surer.
CHMOD_ADVICE = ": if it is yours, run chmod go-w on it, and then look at what is in it"
COPY_ADVICE = (
    ": use a new copy under a new name, from a source that you trust; or, if it is yours, run "
    "chmod go-w on it, and then look at what is in it"
)


def _check_not_writable_by_others(fd, path):
    """Raise UnsafePath unless only root and the current user can change the file or directory."""
    info = os.fstat(fd)
    if info.st_uid not in (0, os.geteuid()):
        raise UnsafePath(
            "%s is owned by uid %d, not by root or the current user" % (path, info.st_uid)
        )
    if info.st_mode & 0o022:
        # Never this advice on a directory for root, to whom /tmp or a shared disk's mount point is
        # "its own": obeyed, it would shut every other account out.
        advice = CHMOD_ADVICE if info.st_uid == os.geteuid() != 0 else ""
        if stat.S_ISREG(info.st_mode):
            advice = COPY_ADVICE
        raise UnsafePath(
            "%s is writable by other users (mode %04o)%s"
            % (path, stat.S_IMODE(info.st_mode), advice)
        )
    if info.st_uid != 0:
        _confirm_owner(fd, path)


def _need_parent(path):
    """Raise FileNotFoundError unless the directory above an absolute path exists."""
    parent = os.path.dirname(path)
    try:
        missing = not stat.S_ISDIR(os.stat(parent).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        missing = True
    except OSError:
        missing = False  # left to others_can_change() and to the open that follows
    if missing:
        words = "Not an existing directory (this program does not create it)"
        raise FileNotFoundError(errno.ENOENT, words, parent)


@contextlib.contextmanager
def _open_dir(path, dir_fd, create, check):
    if dir_fd is None:
        path = os.path.abspath(path)  # after a trailing "/" or "/.", O_NOFOLLOW would not apply
        if check is _check_private:  # written to; a public directory is only read
            _need_parent(path)
            # Not in a helper, here and in main(): a static check of this repository's scripts
            # wants the look and its refusal in the function that writes.
            found = others_can_change(os.path.dirname(path))
            if found:
                raise OpenAbove(refusal("directory", path, *found))
            path = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
    try:
        fd = _open(path, _DIR_FLAGS, dir_fd)
    except FileNotFoundError:
        if not create:
            raise
        with contextlib.suppress(FileExistsError), _errors_with_full_path(path, dir_fd):
            os.mkdir(path, 0o700, dir_fd=dir_fd)
        try:
            fd = _open(path, _DIR_FLAGS, dir_fd)
        except FileNotFoundError:  # mkdir met an entry of that name, which has gone again
            raise UnsafePath(
                "%s could not be created: something else had that name for a moment and is gone "
                "again, so another program is at work there. Try again, or name another directory"
                % _full_path(path, dir_fd)
            ) from None
    try:
        check(fd, _full_path(path, dir_fd))
        if is_secret_key_dir(fd):
            raise UnsafePath("%s is a secret key directory" % _full_path(path, dir_fd))
        yield fd
    finally:
        os.close(fd)


def open_private_dir(path, dir_fd=None, create=False):
    """Context manager: open a directory for clients' data and yield its descriptor.

    path is a path or, with dir_fd, one name in that directory. Raise UnsafePath if it is a
    symlink, is not owned by the current user, is accessible to group or others, or is a secret
    key directory; OpenAbove if another account may be able to change a directory above a path.
    With create, a missing directory is made with mode 0700, but not the one above it; without
    create, FileNotFoundError."""
    return _open_dir(path, dir_fd, create, _check_private)


def open_public_dir(path, dir_fd=None):
    """Context manager: open a directory of files that anybody may read, and yield its descriptor.

    Raise UnsafePath if it is a symlink, is not owned by root or the current user, is writable by
    group or others, or is a secret key directory."""
    return _open_dir(path, dir_fd, False, _check_not_writable_by_others)


def open_public_file(path, dir_fd=None):
    """Open a file that the program trusts or serves to everyone; return a descriptor for reading.

    path is a path, whose directory must pass open_public_dir(), or, with dir_fd, one name in a
    directory from open_public_dir() or open_private_dir(). Raise UnsafePath unless it is a
    regular file, not a symlink, owned by root or the current user, and not writable by group or
    others. The checks are made on the returned descriptor."""
    if dir_fd is None:
        directory, name = os.path.split(os.path.abspath(path))
        with open_public_dir(directory) as opened:
            return open_public_file(name, opened)
    fd = _open(path, _FILE_FLAGS, dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafePath("%s is not a regular file" % _full_path(path, dir_fd))
        _check_not_writable_by_others(fd, _full_path(path, dir_fd))
    except BaseException:
        os.close(fd)
        raise
    return fd


def _publish(data, temporary, tmp_fd, name, dir_fd):
    """Create the file 'name' in dir_fd, with mode 0600, so that it appears complete or not at all.

    As in maildir delivery (D. J. Bernstein, "Using maildir format"; maildir(5)), data is written
    and synced as 'temporary' in tmp_fd, then linked: link() fails where rename() would replace."""
    with _errors_with_full_path(temporary, tmp_fd):
        fd = os.open(temporary, _NEW_FILE_FLAGS, 0o600, dir_fd=tmp_fd)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with _errors_with_full_path(name, dir_fd):
            os.link(temporary, name, src_dir_fd=tmp_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=tmp_fd)


# ---- The mark file ----
#
# mark.txt at the top of a spool directory is three lines, 99 bytes:
#     cp-folder 1
#     kind spool
#     commitment <the commitment ID, or 64 zeros>
# The first program that uses an empty directory creates it before it creates anything else: the
# front end, with its commitment ID, or the proving service, which knows no commitment ID and
# leaves that line all zeros. That mark fits every commitment, and it stays: where the service came
# first, any front end, submit() and sweep() take the directory, whichever commitment they name,
# and sweep() deletes there. The service takes every mark of this form. Nothing changes or deletes
# the file. So it catches a front end that was given the wrong directory only where a front end
# wrote the mark. It does not protect against someone who can write in the directory.

MARK_FILE = "mark.txt"
ANY_COMMITMENT = "0" * 64  # the mark of a directory that the proving service used first
HOLD_FILE = "hold"  # see is_on_hold()
_MARK_TEMPORARY = MARK_FILE + ".making."  # plus a unique suffix
_MARK = re.compile(rb"cp-folder 1\nkind spool\ncommitment ([0-9a-f]{64})\n")


def _read_mark(spool_fd):
    """Return the start of the mark file; b"" if it is not a regular file; None if it is missing."""
    try:
        fd = _open(MARK_FILE, _FILE_FLAGS, spool_fd)
    except FileNotFoundError:
        return None
    try:
        return os.read(fd, 256) if stat.S_ISREG(os.fstat(fd).st_mode) else b""
    finally:
        os.close(fd)


def _create_mark(spool_fd, commitment_id):
    mark = b"cp-folder 1\nkind spool\ncommitment %s\n" % commitment_id.encode()
    with contextlib.suppress(FileExistsError):  # another process created the file first
        _publish(mark, _MARK_TEMPORARY + os.urandom(8).hex(), spool_fd, MARK_FILE, spool_fd)
    os.fsync(spool_fd)


def is_unused(spool_fd):
    """Return whether an open directory is empty, not counting unfinished mark files."""
    return all(name.startswith(_MARK_TEMPORARY) for name in os.listdir(spool_fd))


def check_mark(spool_fd, commitment_id, create=False):
    """Raise UnsafePath unless an open spool directory has a mark file that fits this commitment.

    A mark of ANY_COMMITMENT is every commitment's. Call this before anything is written or deleted
    in the directory. With create, an unused directory gets a mark file: pass it only where the
    caller takes the directory as its spool directory, not in a call that only reads. If
    commitment_id is None, only the file's format is checked, and none is created."""
    _check_commitment_id(commitment_id)
    try:
        found = _read_mark(spool_fd)
        if found is None and create and commitment_id is not None:
            if is_unused(spool_fd):
                _create_mark(spool_fd, commitment_id)
            found = _read_mark(spool_fd)
    except (OSError, UnsafePath) as error:
        reason = getattr(error, "strerror", None) or error
        problem = "its mark file (mark.txt) cannot be read or created (%s)" % reason
        untouched = ""  # the file may have been created before the fault
    else:
        untouched = ". Nothing in it was written or removed"
        match = _MARK.fullmatch(found or b"")
        if match and (commitment_id is None or match.group(1).decode() in (commitment_id, ANY_COMMITMENT)):
            return
        if found is None:
            problem = (
                "it has no mark file (mark.txt): check the path. The first program that uses a "
                "spool directory writes that file, while the directory is still empty. To start "
                "anew, name a new directory as the spool directory; do not empty this one"
            )
        elif match:
            problem = "its mark file (mark.txt) is for commitment %s" % match.group(1).decode()
        else:
            problem = "its mark file (mark.txt) is not that of a spool directory"
    wanted = " of commitment %s" % commitment_id if commitment_id else ""
    raise UnsafePath(
        "%s is not a spool directory%s: %s%s"
        % (_full_path(".", spool_fd), wanted, problem, untouched)
    )


# ---- Requests and responses, relative to an open directory ----


def is_on_hold(spool_fd):
    """Return whether an open spool directory has an entry "hold", of any kind.

    An operator creates it, for example after the proving service has failed: responses that it
    made earlier may then be faulty too and must not go out, and somebody who looks for the cause
    may need the files. While it is there, submit(), sweep(), finished_requests(),
    wait_for_responses() and read_response() raise OnHold. The functions that take an open
    directory do not look for it, nor do response_path() and sha256_file(), nor does the proving
    service: a front end calls this itself before it submits a request or hands out a response."""
    try:
        os.stat(HOLD_FILE, dir_fd=spool_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _on_hold(path, untouched):
    return OnHold(
        "the spool directory is on hold (%s exists): %s. Read integrations/vllm/REFERENCE.md "
        "before the hold is lifted" % (path, untouched)
    )


def _check_hold(spool_fd, untouched):
    if is_on_hold(spool_fd):
        raise _on_hold(_full_path(HOLD_FILE, spool_fd), untouched)


def submit_at(spool_fd, request):
    """Submit one request to an open spool directory, and return its request ID and its tag key.

    spool_fd comes from open_private_dir(), after check_mark(). request is the bytes of
    make_request(), or the list of its numbers. in/ and in/.tmp/ are created if missing. The
    service never sees part of a request, and no file is ever replaced."""
    data = bytes(request) if isinstance(request, (bytes, bytearray)) else pack(request)
    data += os.urandom(NONCE_BYTES)
    name = request_id(data)
    with open_private_dir("in", spool_fd, create=True) as in_fd:
        with open_private_dir(".tmp", in_fd, create=True) as tmp_fd:
            _publish(data, name, tmp_fd, name + ".req", in_fd)
    return name, tag_key(data)


def _open_response_files(out_fd, request_id, fds):
    """Open a response's files into fds, {suffix: descriptor}. Return the suffixes that exist."""
    present = []
    # The service publishes .y first, then .tag, then .proof or .failed. They are opened in the
    # reverse order, so that a response that is being published looks absent, never incomplete.
    for suffix in (FAILED, "proof", TAG, "y"):
        try:
            fds[suffix] = _open("%s.%s" % (request_id, suffix), _FILE_FLAGS, out_fd)
        except FileNotFoundError:
            continue
        except UnsafePath:  # a symlink
            pass
        except OSError as error:
            if error.errno != errno.ENXIO:  # ENXIO: a socket
                raise
        present.append(suffix)  # also without a descriptor: _inspect_response() reports it
    return present


def _inspect_response(request_id, present, fds):
    """Return the Response for the files that exist. Close descriptors of non-regular files."""
    failed = FAILED in present
    files, faults, times = {}, [], []
    if set(present) != ({TAG, FAILED} if failed else {"y", TAG, "proof"}):
        faults.append("unexpected set of files: %s" % ", ".join(reversed(present)))
    for suffix in reversed(present):
        info = os.fstat(fds[suffix]) if suffix in fds else None
        if info is None or not stat.S_ISREG(info.st_mode):
            faults.append("%s is not a regular file" % suffix)
            if suffix in fds:
                os.close(fds.pop(suffix))
            continue
        if info.st_size != {"y": 4, TAG: TAG_BYTES, FAILED: 0}.get(suffix, info.st_size):
            faults.append("%s has %d bytes" % (suffix, info.st_size))
        times.append(info.st_mtime_ns)
        if suffix != TAG:
            files[suffix] = {"size": info.st_size}
    if len(set(times)) > 1:
        faults.append("the files have different modification times")
    return Response(request_id, failed, times[-1] if times else 0, files, faults)


@contextlib.contextmanager
def open_response_at(out_fd, request_id):
    """Context manager: open the files of one response in an open out/.

    Yield the Response, without hashes, and {suffix: descriptor} for its regular files, the tag
    included; or (None, {}) if there is no response yet. To keep a descriptor after the block,
    pop it from the dictionary.

    Warning: the tag is not compared here. Until the caller has compared it with the bytes read
    from these descriptors (tag_hmac(), failed_tag()), the response may be another request's."""
    _check_request_id(request_id)
    fds = {}
    try:
        present = _open_response_files(out_fd, request_id, fds)
        if "proof" in present or FAILED in present:
            yield _inspect_response(request_id, present, fds), fds
        else:
            yield None, {}
    finally:
        for fd in fds.values():
            os.close(fd)


def delete_response_at(out_fd, request_id):
    """Delete the files of one response from an open out/. Return whether there was one.

    The file that completes a response is deleted first, so that no reader finds a response that
    looks complete and is not. Names are only unlinked: a symlink is removed, not followed. If a
    name cannot be unlinked, the others are still tried, and the first OSError is raised at the
    end."""
    _check_request_id(request_id)
    failure, found = None, False
    for suffix in ("proof", FAILED, TAG, "y"):
        name = "%s.%s" % (request_id, suffix)
        try:
            os.unlink(name, dir_fd=out_fd)
            found = True
        except FileNotFoundError:
            pass
        except OSError as error:
            failure = failure or OSError(error.errno, error.strerror, _full_path(name, out_fd))
    if failure:
        raise failure
    return found


def _is_pending(spool_fd, request_id):
    """Return whether a request is still in in/ or work/."""
    for directory in ("in", "work"):
        name = "%s/%s.req" % (directory, request_id)
        with contextlib.suppress(FileNotFoundError), _errors_with_full_path(name, spool_fd):
            os.stat(name, dir_fd=spool_fd, follow_symlinks=False)
            return True
    return False


def _delete_old(spool_fd, out_fd, days):
    deleted = 0
    for name in os.listdir(out_fd):
        request_id, _, suffix = name.partition(".")
        if suffix in ("proof", FAILED) and REQUEST_ID.match(request_id):
            try:
                info = os.stat(name, dir_fd=out_fd, follow_symlinks=False)
            except FileNotFoundError:  # deleted meanwhile
                continue
            old = info.st_mtime < time.time() - 86400 * days
            if old and not _is_pending(spool_fd, request_id):
                _check_hold(spool_fd, "nothing more was deleted")
                deleted += delete_response_at(out_fd, request_id)
    return deleted


def sweep(spool_dir, commitment_id, days):
    """Delete the responses that are more than 'days' days old. Return how many.

    The service may be running. A request still in in/ or work/ is left for the next sweep. A
    directory whose mark fits every commitment (see the mark file) is swept too. The sweep ends
    with an OSError at the first response with a file that cannot be deleted. Raise
    OnHold, and delete nothing, while the directory is on hold."""
    if commitment_id is None:
        raise ValueError("sweep() needs a commitment ID")
    if type(days) is not int or days < 1:
        raise ValueError("days must be a whole number, 1 or more")
    with open_private_dir(spool_dir) as spool_fd, open_private_dir("out", spool_fd) as out_fd:
        check_mark(spool_fd, commitment_id)
        _check_hold(spool_fd, "nothing was deleted")
        return _delete_old(spool_fd, out_fd, days)  # looks again before each response


# ---- The same by path, for scripts ----
#
# In every call these functions check the spool directory's owner and mode, and that it is no
# symlink; those that write or delete also check its mark file. Four check none of these:
# finished_requests(), wait_for_responses(), response_path(), sha256_file().


@contextlib.contextmanager
def _open_spool_dir(spool_dir, commitment_id, create):
    _check_commitment_id(commitment_id)
    # Without commitment_id a new directory could get no mark file and would be rejected, so it is
    # not created.
    with open_private_dir(spool_dir, create=create and commitment_id is not None) as spool_fd:
        check_mark(spool_fd, commitment_id, create)
        yield spool_fd


def make_private_dirs(spool_dir, *subdirs, commitment_id, create_mark=True):
    """Create a spool directory and subdirs, each inside the one before, with mode 0700.

    check_mark() runs before the first subdirectory is created. A missing spool directory is
    created only if it can get a mark file: create_mark is true and commitment_id is not None."""
    with contextlib.ExitStack() as stack:
        fd = stack.enter_context(_open_spool_dir(spool_dir, commitment_id, create_mark))
        for name in subdirs:
            fd = stack.enter_context(open_private_dir(name, fd, create=True))


def submit(spool_dir, request, commitment_id, with_key=False):
    """Submit one request, and return its request ID, under which the response will appear.

    A missing spool directory is created, and an empty one gets commitment_id's mark file. With
    with_key, return (request ID, tag key), for read_response(). Raise OnHold while the directory
    is on hold. See submit_at()."""
    if commitment_id is None:
        raise ValueError("submit() needs a commitment ID")
    with _open_spool_dir(spool_dir, commitment_id, create=True) as spool_fd:
        _check_hold(spool_fd, "nothing was submitted")
        name, key = submit_at(spool_fd, request)
    return (name, key) if with_key else name


def request_stop(spool_dir, commitment_id=None):
    """Ask the service to stop: it takes no further batch, finishes the current one and exits.

    Creates the empty file "stop". A running service exits when any entry has that name, deletes
    only an empty regular file, and will not start while another kind of entry has it;
    request_stop() then raises UnsafePath. A service that finds an empty stop file when it starts
    deletes it and keeps running, so call this once the service is running. Raises OSError (EAGAIN)
    if an existing stop file is deleted twice while it is checked.
    """
    path = os.path.join(spool_dir, "stop")
    with _open_spool_dir(spool_dir, commitment_id, create=False) as spool_fd:
        for _ in range(2):  # again if the service deleted an earlier stop file
            try:
                os.close(os.open("stop", _NEW_FILE_FLAGS, 0o600, dir_fd=spool_fd))
                return
            except FileExistsError:  # already requested
                with contextlib.suppress(FileNotFoundError):
                    info = os.stat("stop", dir_fd=spool_fd, follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode) and not info.st_size:
                        return
                    raise UnsafePath("%s is not an empty regular file" % path) from None
    message = "an existing stop file was deleted twice while it was being checked; try again"
    raise OSError(errno.EAGAIN, message, path)


def finished_requests(spool_dir):
    """Return the set of request IDs that have a response now. Raise OnHold: see is_on_hold()."""
    hold = os.path.join(spool_dir, HOLD_FILE)
    # Any other error is raised: it does not show that there is no hold.
    with contextlib.suppress(FileNotFoundError):
        os.lstat(hold)
        raise _on_hold(hold, "no response was listed")
    try:
        names = os.listdir(os.path.join(spool_dir, "out"))
    except FileNotFoundError:
        return set()
    finished = set()
    for name in names:
        stem, _, suffix = name.partition(".")
        if suffix in ("proof", FAILED) and REQUEST_ID.match(stem):
            finished.add(stem)
    return finished


def wait_for_responses(spool_dir, request_ids, timeout_s, poll_s=0.05, keep_waiting=lambda: True):
    """Wait until all of request_ids have a response, timeout_s has passed, or keep_waiting() is
    false.

    Return the set of those that have a response. Raise OnHold, also if the hold comes meanwhile."""
    request_ids, deadline = set(request_ids), time.monotonic() + timeout_s
    while True:
        last_look = time.monotonic() >= deadline or not keep_waiting()
        finished = request_ids & finished_requests(spool_dir)
        if finished == request_ids or last_look:
            return finished
        time.sleep(poll_s)


def response_path(spool_dir, request_id, suffix):
    """Return the path of a response file: suffix "y", "proof", "failed" or "tag"."""
    _check_request_id(request_id)
    if suffix not in RESPONSE_FILES + (FAILED, TAG):
        raise ValueError("%r is not the suffix of a response's file" % (suffix,))
    return os.path.join(spool_dir, "out", "%s.%s" % (request_id, suffix))


def _read_response_files(response, fds, hashes, key):
    """Read each file once: add SHA-256 hashes to response.files, and a fault for a wrong tag."""
    digests = {suffix: hashlib.sha256() for suffix in response.files}
    if response.failed:
        expected = None if key is None else failed_tag(key, response.id)
    else:
        y = os.read(fds["y"], 4)
        digests["y"].update(y)
        mac = (
            None if key is None else tag_hmac(key, response.id, y, response.files["proof"]["size"])
        )
        sinks = [digests["proof"]] if hashes else []
        if mac is not None:
            sinks.append(mac)
        while sinks and (chunk := os.read(fds["proof"], 8 << 20)):
            for sink in sinks:
                sink.update(chunk)
        expected = None if mac is None else mac.digest()
    if hashes:
        for suffix, digest in digests.items():
            response.files[suffix]["sha256"] = digest.hexdigest()
    if expected is not None and not hmac.compare_digest(os.read(fds[TAG], TAG_BYTES), expected):
        response.faults.append("the tag does not match the request ID and the response's bytes")


def read_response(spool_dir, request_id, hashes=True, key=None):
    """Read one response, check it, and return it as a Response.

    faults is empty only if the files are exactly ID.y (4 bytes), ID.tag (32 bytes) and ID.proof,
    or exactly ID.tag and an empty ID.failed; all are regular files with the same modification
    time; and the tag matches key, from submit(with_key=True). Without key the tag is not
    compared, which is a fault too: the response may be another request's. A missing response is
    a fault, not an exception. hashes=False leaves the SHA-256 out.

    The hash and the tag come from one read of each file: a caller that later reads a file by its
    path must compare it with this hash. Raise OnHold: see is_on_hold()."""
    _check_request_id(request_id)
    try:
        with open_private_dir(spool_dir) as spool_fd:
            _check_hold(spool_fd, "no response was read")
            with open_private_dir("out", spool_fd) as out_fd:
                with open_response_at(out_fd, request_id) as (response, fds):
                    if response is not None:
                        if not response.faults:
                            _read_response_files(response, fds, hashes, key)
                        if key is None:
                            response.faults.append("the tag was not compared: no key was given")
                        return response
    except FileNotFoundError:
        pass
    return Response(request_id, False, 0, {}, ["there is no response"])


def sha256_file(path):
    """Return the SHA-256 of a file as hex digits."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# ---- Command line ----


def _end_token(text):
    if text == "none":
        return None
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError('%r is neither a token ID nor "none"' % text)
    return int(text)


def _days(text):
    if not re.fullmatch("[1-9][0-9]{0,4}", text, re.ASCII):
        raise argparse.ArgumentTypeError("%r is not a whole number of days, 1 or more" % text)
    return int(text)


def _read_token_ids(path, key):
    """Return the token IDs in a file: decimal numbers, a JSON list, or an object's list 'key'."""
    with open(path, encoding="ascii", errors="replace") as f:
        text = f.read().strip()
    if text.startswith("{"):
        try:
            ids = json.loads(text)[key]
        except (ValueError, KeyError, RecursionError):
            ids = None
        if type(ids) is not list or not all(type(number) is int for number in ids):
            raise ValueError('%s is not a JSON object with a list of integers "%s"' % (path, key))
        return ids
    words = re.split(r"[\s,]+", text.removeprefix("[").removesuffix("]").strip())
    if not all(re.fullmatch("[0-9]{1,10}", word) for word in words):
        raise ValueError("%s does not contain token IDs as decimal numbers" % path)
    return [int(word) for word in words]


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build a request file, submit one to a proving service's spool directory, or "
        "delete old responses there."
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    text = "submit a request file and print its request ID"
    command = commands.add_parser(
        "submit",
        help=text,
        description=text,
        epilog="The response appears as SPOOL_DIR/out/ID.proof and ID.y, or as ID.failed if there "
        "is no proof. Check it with integrations/check_proof.py.",
    )
    command.add_argument("spool_dir", metavar="SPOOL_DIR", help="created if it is missing")
    command.add_argument("request_file", metavar="REQUEST_FILE", help="a file from make-request")
    command.add_argument(
        "commitment_id", metavar="COMMITMENT_ID", help="of the service's commitment (64 hex digits)"
    )

    text = "delete old responses, but none whose request is still waiting; print their number"
    command = commands.add_parser("sweep", help=text, description=text)
    command.add_argument("spool_dir", metavar="SPOOL_DIR")
    command.add_argument(
        "commitment_id", metavar="COMMITMENT_ID", help="of the service's commitment (64 hex digits)"
    )
    text = "delete the responses that are older than this many days (a whole number, 1 or more)"
    command.add_argument("days", metavar="DAYS", type=_days, help=text)

    text = (
        "write the request file for a prompt and its whole reply, "
        "and print the checked range as START:END"
    )
    command = commands.add_parser(
        "make-request",
        help=text,
        description=text,
        epilog="PROMPT_FILE and REPLY_FILE are text files with token IDs as decimal numbers, "
        "separated by spaces, commas or newlines; a JSON list works too. Or name one file twice: a "
        'JSON object with the lists "prompt_token_ids" and "reply_token_ids", such as the '
        "served.json that integrations/ask_and_check.py stores. "
        "integrations/check_proof.py takes OUTPUT_FILE as its REQUEST_FILE and START:END as "
        "its --range.",
    )
    command.add_argument(
        "length",
        metavar="T",
        type=int,
        help="the number of tokens in a request: 2**lt, where lt is a line of the commitment's "
        "parameter file (policy.txt)",
    )
    command.add_argument("prompt_file", metavar="PROMPT_FILE")
    command.add_argument("reply_file", metavar="REPLY_FILE")
    command.add_argument(
        "end_token",
        metavar="END_TOKEN",
        type=_end_token,
        help="the ID of the token that ends a reply: the reply must end with it; "
        'or "none" if your own length limit stopped the reply',
    )
    command.add_argument("output_file", metavar="OUTPUT_FILE", help="must not exist")
    return parser.parse_args(argv)


# Follows the refusal of a directory that belongs to another account.
AS_THAT_ACCOUNT = (
    " (run this as that account: without sudo if it is your own, else sudo -u ACCOUNT ... If a "
    "service was started in this directory under sudo, that fails too: integrations/LIMITS.md says "
    "which way works behind sudo. If you do not know that account, name another directory)"
)


def main(argv=None):
    """Run the command line.

    Exit codes: 0 success; 1 failure; 2 usage error, or a place that OpenAbove refuses."""
    args = _parse_args(argv)
    try:
        if args.command == "submit":
            with open(args.request_file, "rb") as f:
                print(submit(args.spool_dir, f.read(), args.commitment_id.lower()))
        elif args.command == "sweep":
            print(sweep(args.spool_dir, args.commitment_id.lower(), args.days))
        else:
            prompt = _read_token_ids(args.prompt_file, "prompt_token_ids")
            reply = _read_token_ids(args.reply_file, "reply_token_ids")
            request = whole_reply(prompt, reply, args.length, args.end_token)
            if not os.path.basename(args.output_file):
                raise ValueError("OUTPUT_FILE %s names a directory" % args.output_file)
            output_file = os.path.abspath(args.output_file)
            _need_parent(output_file)
            found = others_can_change(os.path.dirname(output_file))
            if found:
                raise OpenAbove(refusal("file", output_file, *found))
            output_file = os.path.join(
                os.path.realpath(os.path.dirname(output_file)), os.path.basename(output_file)
            )
            # Owner only: the file contains a client's prompt and reply.
            with open(os.open(output_file, _NEW_FILE_FLAGS, 0o600), "wb") as f:
                f.write(request)
            print("%d:%d" % request_fields(request)[1:])
    except OpenAbove as error:
        print("spool.py: %s" % error, file=sys.stderr)
        return 2
    except FileExistsError as error:
        print("error: %s already exists (choose a new name)" % error.filename, file=sys.stderr)
        return 1
    except OSError as error:
        print(
            "error: %s%s" % ("%s: " % error.filename if error.filename else "", error.strerror),
            file=sys.stderr,
        )
        return 1
    except (UnsafePath, OnHold, ValueError) as error:
        hint = AS_THAT_ACCOUNT if str(error).endswith("not by the current user") else ""
        print("error: %s%s" % (error, hint), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
