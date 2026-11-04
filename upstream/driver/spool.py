# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Builds a request (the client's part: request(), whole_reply()), and handles a proving service's
spool directory from the front end's side: submit a request (lay()), wait for the response
(wait_for()), read it (read_member()), and stop the service (stop()). The spool directory is the
directory through which requests reach the service and responses leave it. This module needs only
the standard library.

No client may reach the spool directory. A front end stands between the clients and the directory,
and the part of it that reads and writes there runs under the service's own account. The service is
the prover program cp_prove, run with the option -spool <dir>; its code for the directory is
prover/prover/spool.h. docs/service.md, "What your program must do", lists a front end's steps;
driver/README.md lists further duties. The first of them: the service checks no finished proof, and
a proof that cp_verify rejects can hold values that are not masked, so a front end hands a response
on only after cp_verify has passed every proof of its batch. read_member() does not do that.

A request is submitted as a message is delivered to a maildir: it is written and synced under a
temporary name and then linked to its final name (D. J. Bernstein, "Using maildir format"; the
manual page maildir(5), "How a message is delivered").

From a shell, after "python3 driver/spool.py":
  lay <spool> <request file>
      submits the file (the bytes of request()) and prints the id.
  request <T> <prompt file> <reply file> <end token's id, or none> <file to write>
      writes the bytes of whole_reply() and prints r:e. The two input files hold tokens as
      little-endian 32-bit words. The file to write must not exist; it is created for its owner
      alone (mode 0600).

A front end that serves clients should not trust the path to the spool directory: an account that
can write in a directory on that path can put another directory there. lay(), read_member() and the
other functions that take a path do not walk it one directory at a time without following links, so
they would not notice. Such a front end must bring its own functions for that: this module has
none.

Only the shell command "lay" checks the path first. It rejects, with exit code 2, a spool directory
below a place that another account may be able to change (others_can_change()). The check and the
submission are two steps, each by path, not one walk by descriptors. A relative path is taken from
the working directory's real path, so a symbolic link through which the caller came there is not
checked."""

import collections
import errno
import hashlib
import hmac
import os
import re
import stat
import struct
import sys
import time

# The files of the spool directory. <id> is request_id() of the request file's bytes; the functions
# below take it as the parameter `name`.
#   in/<id>.req      a request: the bytes of request(), then a nonce of 32 random bytes
#   work/<id>.req    the same file while the service works on it
#   out/<id>.y       the token file of a response with a proof: one 32-bit word (see request())
#   out/<id>.tag     in every response: 32 bytes for the front end (see "The tag" below)
#   out/<id>.proof   the proof file. It is named last, so where it exists the response is complete
#                    (one exception: see the comment on the order of opening in read_member())
#   out/<id>.failed  the response to a request that gets no proof: an empty file, and no .y
#   stop             an empty file that asks the service to stop (see stop())
# .y, .proof and .failed get their names by a rename, so none is seen half written. All files of the
# responses of a batch (the requests that the service proves together) have one modification time,
# the time of the batch's release.
MEMBER_FILES = ("y", "proof")  # the endings of the two files of a response that a client downloads
FAILED = "failed"
ID = re.compile(r"[0-9a-f]{64}\Z")
NONCE_BYTES = 32

# One response (`member` in identifiers) as read_member() found it. failed: it is the file
# <id>.failed. tick_ns: the modification time of <id>.proof (or .failed); if that file is missing,
# of the last regular file found, or 0. files: {ending: {"size": bytes, "sha256": hex}}. faults: a
# message for each check of read_member() that failed; usually empty.
Member = collections.namedtuple("Member", "id failed tick_ns files faults")


def pack(words):
    """Returns a list of numbers as bytes, each number as a little-endian 32-bit word."""
    return struct.pack("<%dI" % len(words), *words)


def request(tokens, first, end):
    """Returns the bytes of a request without its nonce: the T tokens, then r = first and e = end.

    T is 2^lt, with `lt` from the parameter file (`policy.txt`). The proof is about positions r to
    e - 1, the request's range. At position t the model reads tokens 0 to t, and the proof checks
    token t + 1. With e = T the last position checks the token after the request, which the prover
    announces in <id>.y; with e < T that file holds 0. r and e have no default, because a proof says
    nothing about a token that no position of the range checks. The values and their number are not
    checked here: the service answers a request without r < e <= T, or with a token outside the
    vocabulary, with <id>.failed, and leaves one that does not have T tokens unanswered (see
    request_id()). For a reply after a prompt, use whole_reply()."""
    return pack(list(tokens) + [first, end])


def whole_reply(prompt, reply, length, end_token=None, fill=0):
    """Returns the bytes of a request, without its nonce, whose range checks the whole reply.

    The request holds the prompt's n tokens, the reply's m tokens, and the token 'fill' up to
    'length' tokens (T: see request()), then r = n - 1 and e = n + m - 1. So the range
    reaches from the prompt's last token, where the reply's first token is checked, to the position
    where the reply's last token is checked.

    A proof does not know where a reply should end: a reply that was cut short verifies as a shorter
    reply. 'end_token' is the id of a token that ends a reply in the model. If the reply does not
    end with it, it is appended, and the range has one more position. Under the exact-match policy
    (in the parameter file C = theta = 2^sh, cap 0, sT = sh - 4, lines that
    verifier/docs/parameter-file.md explains: the one policy under which the verifier program
    cp_verify says YES about a request's tokens without the option -policy) an accepted proof then
    states of that position too that its token, the end token, is the model's top choice (the token
    with the largest logit). That a provider who cut off the end of the reply gets no such proof
    accepted is the design's aim; nothing is proved about these programs. A reply that contains
    'end_token' before its last token raises ValueError: the model's reply would have ended there.
    Omit 'end_token' only if you set the length limit that stopped the reply yourself.

    An empty prompt, an empty reply without 'end_token', and a prompt and reply that together, with
    an appended end token, have more than 'length' tokens raise ValueError too."""
    reply = list(reply) + ([end_token] if end_token is not None and list(reply)[-1:] != [end_token] else [])
    if end_token is not None and end_token in reply[:-1]:
        raise ValueError("the reply contains the end token %d before its last token" % end_token)
    n, m = len(prompt), len(reply)
    if not (n >= 1 and m >= 1 and n + m <= length):
        raise ValueError("a prompt of %d and a reply of %d tokens do not fit a request of %d tokens" % (n, m, length))
    return request(list(prompt) + reply + [fill] * (length - n - m), n - 1, n + m - 1)


def request_id(data):
    """Returns the id of a request file's bytes (request, then nonce).

    No one chooses an id: it is the SHA-256 of a label and the file's bytes, 64 hex digits, as
    Spool::id_of_bytes() in prover/prover/spool.h. The service leaves a file whose length is not a
    request's where it is, and removes one of that length whose name is not that hash. Neither gets
    a response. So only someone who knows a request's bytes, including the random ones, can compute
    the name of its response."""
    return hashlib.sha256(b"p258-spool-req-1" + data).hexdigest()


# ---- The tag: the file <id>.tag beside every response ----
#
# The tag is an HMAC over the id and the response's bytes (tag_of(), tag_of_failed()). Without it,
# an account that cannot read a response but can exchange names in out/ could bring a front end to
# hand one client's response to another. Only those who have the request file's nonce can form the
# tag's key (tag_key()), so the tag does not protect against the service's own account, or against
# anyone else who can read that file. It says nothing on whether a proof is right. It is for the
# front end: give it to no client.

TAG = "tag"  # the ending of the file. It is not among MEMBER_FILES, which are what a client downloads
TAG_BYTES = 32


def tag_key(data):
    """Returns the key of a request's tags, 32 bytes, as Spool::tag_key_of() in
    prover/prover/spool.h. data: the request file's bytes, or the nonce alone, which is their last
    32 bytes. Keep the key with the id and forget the nonce. The key belongs in no log, message or
    exception."""
    if len(data) < NONCE_BYTES:
        raise ValueError("a request file ends with a nonce of %d bytes" % NONCE_BYTES)
    return hmac.new(bytes(data[-NONCE_BYTES:]), b"p258-spool-tag-1", hashlib.sha256).digest()


def tag_begun(key, name, y, proof_bytes):
    """Returns tag_of()'s HMAC object, fed with everything before the proof. proof_bytes is the
    proof's length. For a caller that reads a long proof piece by piece: it calls update() with each
    piece, checks that it has fed exactly proof_bytes bytes, and takes digest()."""
    return hmac.new(key, bytes.fromhex(name) + struct.pack("<2Q", len(y), proof_bytes) + y, hashlib.sha256)


def tag_of(key, name, y, proof):
    """Returns the tag of a response with a proof: HMAC-SHA-256 under the key over the id's 32 raw
    bytes, the lengths of y and of the proof as little-endian 64-bit words, y's bytes and the
    proof's bytes."""
    h = tag_begun(key, name, y, len(proof))
    h.update(proof)
    return h.digest()


def tag_of_failed(key, name):
    """Returns the tag of the response "failed": the same HMAC over the id's 32 raw bytes, two
    lengths of 0, and the word "failed". No message of tag_of() with both lengths 0 is as long, so
    the two functions never hash the same message."""
    return hmac.new(key, bytes.fromhex(name) + bytes(16) + b"failed", hashlib.sha256).digest()


def lay_tag(spool, name, tag, ns):
    """Makes <id>.tag in out/ directly under its final name, as the service does, with the
    modification time ns. It fails if the name exists, and it does not sync. For a test's stand-in
    for the service."""
    with open(os.open(path_of(spool, name, TAG), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600), "wb") as f:
        f.write(tag)
        f.flush()
        os.utime(f.fileno(), ns=(ns, ns))


def home(spool):
    """Returns the one absolute path by which this module names the spool directory, for its checks
    and for every file that it opens there. Checks on one form of the path and opens by another
    could reach two directories: for "link/../spool" os.path.abspath() drops "link/.." as text,
    while the system first follows the link. The result has no trailing slash: with one, lstat()
    would follow a link."""
    return os.path.abspath(spool)


def account_that_ran_sudo():
    """Returns the user ID of the account that ran sudo, or None.

    The ID is read from the environment variable SUDO_UID, and only if this process is root by both
    its real and its effective user ID: under a set-user-ID program the caller could set the
    variable to anything. The variable's text must be the ID exactly as it is printed in decimal (no
    sign, blank or leading 0). The ID must not be 0 and must be below 2^32 - 1: that number names no
    account, and chown reads it as "leave unchanged"."""
    s = os.environ.get("SUDO_UID", "")
    strict = len(s) <= 10 and s.isascii() and s.isdigit() and str(int(s)) == s and 0 < int(s) < 2**32 - 1  # the length first: int() refuses a text of thousands of digits
    return int(s) if os.geteuid() == 0 and os.getuid() == 0 and strict else None


def not_the_users(folder, found, owners):
    """Returns the message of the PermissionError that private_folders() raises for an existing
    folder: its path, the fault (with the owner and mode, if it is a directory), the rule, and what
    to do. found: the folder's os.lstat(); owners: the user IDs that may own it. The wording follows
    the service's own message (Spool::not_the_users() in prover/prover/spool.h). Root is never told
    to chmod, as in cure() below: on a mistyped path that would shut every other account out of a
    folder that they use."""
    rule = ("The spool directory and each folder in it must be a directory and no symbolic link and belong to the user who runs this "
            "program (you are %d), with mode 0700; under sudo the spool directory itself may also belong to the account that ran sudo. " % os.geteuid())
    if stat.S_ISLNK(found.st_mode):
        fault, advice = "is a symbolic link", "Name the directory itself, not a link to it"
    elif not stat.S_ISDIR(found.st_mode):
        fault, advice = "is not a directory", "Name a spool directory of your own"
    elif found.st_uid in owners:
        fault = "(owner %d, mode %04o) is open to its group or to others" % (found.st_uid, stat.S_IMODE(found.st_mode))
        advice = "Its owner is right and only its mode is at fault: find out who else could read or write there; then " + (
            "close it (chmod 700)" if os.geteuid() else "name a new directory, made with mode 0700")
    else:
        fault = "(owner %d, mode %04o) belongs to another user" % (found.st_uid, stat.S_IMODE(found.st_mode))
        advice = ("Run this program as that owner, or name a spool directory of your own. What a run under sudo makes belongs to root: "
                  "so run the service and the client both with sudo, or neither")
    return "%s %s. %s%s. Nothing was written" % (printable(folder), fault, rule, advice)


def private_folders(spool, *below):
    """Creates the spool directory and the folders 'below' it, each inside the one before, for their
    owner alone (mode 0700).

    An existing folder is left as it is, but it must be a folder and no symbolic link, belong to
    this user and have no permission for its group or for others (the service demands all three as
    well, and does not start without them). Anything else raises PermissionError, and nothing is
    written in it: an account that made it first could read or replace what passes through it, and
    anyone who can list it sees the requests' ids and when each response comes. The folders above
    the spool directory are the operator's to guard; missing ones are created with the default mode.
    A process that only reads responses should call private_folders(spool, "out") once, and one
    that only creates the stop file private_folders(spool), before it trusts what it finds there.

    Under sudo the spool directory itself, and it alone, may also belong to the account that ran
    sudo (account_that_ran_sudo()), so that a directory which that account made beforehand can be
    used; the service allows the same. The directory must still have no permission for its group or
    for others. That account is trusted here as root is: as the directory's owner it can rename or
    replace in/ while a request is written. What a run under sudo creates belongs to root, in/ and
    the request files too, and an account that is not root can neither enter nor read them. So run
    the service and the program that submits requests both with sudo, or neither.

    A limit on every comparison of an owner here, in read_member() and in the service's own demands:
    in a user namespace the kernel shows every owner that the namespace does not map as one and the
    same uid (the overflow uid, 65534 unless it was changed). A process that sees itself as that
    uid, in a namespace that leaves any uid out, cannot tell its own files from another account's.
    """
    folder = home(spool)
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    for name in ("",) + below:  # one by one: os.makedirs() would give the mode to the last folder only
        folder = os.path.join(folder, name) if name else folder
        try:
            os.mkdir(folder, 0o700)
        except FileExistsError:
            found = os.lstat(folder)
            owners = (os.geteuid(),) if name else (os.geteuid(), account_that_ran_sudo())  # no name: the spool directory itself
            if not stat.S_ISDIR(found.st_mode) or found.st_uid not in owners or found.st_mode & 0o077:
                raise PermissionError(not_the_users(folder, found, owners)) from None


def lay(spool, words, with_key=False):
    """Submits one request, given as the bytes of request() or as the list of its numbers (the
    tokens, r, e), with a fresh random nonce. Returns its id, under which the response will appear;
    with_key=True: the id and the key of its tags (tag_key()), for read_member(). It never replaces
    an existing file.

    The spool directory must exist: otherwise lay() raises FileNotFoundError and writes nothing, so
    that a mistyped path does not become a new folder that no service reads, where the request would
    wait for ever. in/ and in/.tmp/ are created if missing, by private_folders(), which can raise
    PermissionError.

    Keep the id with the request's owner, and give the response to no one else. The verifier gets
    the request without the nonce: the bytes of request()."""
    data = (bytes(words) if isinstance(words, (bytes, bytearray)) else pack(words)) + os.urandom(NONCE_BYTES)
    name = request_id(data)
    if not os.path.lexists(home(spool)):
        raise FileNotFoundError(
            "%s does not exist: a request is submitted to an existing spool directory only. Check the path, or make the folder "
            "first, for its owner alone (mkdir -m 700)" % home(spool)
        )
    private_folders(spool, "in", ".tmp")
    unfinished = os.path.join(home(spool), "in", ".tmp", name)
    descriptor = os.open(unfinished, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(unfinished, os.path.join(home(spool), "in", name + ".req"))  # fails if the name exists; rename() would replace it
    finally:
        os.unlink(unfinished)
    return (name, tag_key(data)) if with_key else name


def answered(spool):
    """Returns the ids that have a response now. It only lists the names in out/ and checks nothing
    about the files: anyone who can write there can make an id appear. Treat a response as the
    service's only after read_member() has reported no fault on it."""
    try:
        listed = os.listdir(os.path.join(home(spool), "out"))
    except FileNotFoundError:
        return set()
    return {name for name, _, ending in (x.partition(".") for x in listed) if ending in ("proof", FAILED) and ID.match(name)}


def path_of(spool, name, ending):
    """Returns the path of one file of a response. name must be an id and ending one of "y",
    "proof", "failed" and "tag": any other string raises ValueError, because it could name a file
    outside out/."""
    if not ID.match(name) or ending not in MEMBER_FILES + (FAILED, TAG):
        raise ValueError("%r and %r are not the id and the ending of a response's file" % (name, ending))
    return os.path.join(home(spool), "out", name + "." + ending)


def wait_for(spool, names, limit_s, poll_s=0.05, go_on=lambda: True):
    """Waits until every one of these ids has a response, limit_s seconds have passed, or go_on() is
    false (for example because the service has ended). Returns those of the ids that have a
    response."""
    names, deadline = set(names), time.monotonic() + limit_s
    while True:
        last_look = time.monotonic() >= deadline or not go_on()
        have = names & answered(spool)
        if have == names or last_look:
            return have
        time.sleep(poll_s)


def read_member(spool, name, sums=True, key=None):
    """Reads one response and checks what the service promises about every response: there are
    exactly <id>.y, of 4 bytes, <id>.tag, of 32 bytes, and <id>.proof, or exactly <id>.tag and an
    empty <id>.failed; each is a regular file with one link; the files have one modification time;
    the tag belongs to this user and has the mode 0600. Each broken promise is an entry of
    Member.faults; a response that is missing or not yet complete counts as one. sums=False skips
    the SHA-256. key: the key that lay(with_key=True) returned for this id. With it the tag is
    compared with the id and the response's bytes; without it only the tag's form is checked, and
    the response may be another request's. The tag is not among Member.files.

    Each file is opened once, without following a link. All checks are made on that descriptor, and
    the SHA-256 and the comparison come from one reading of it. Names in out/ may change afterwards:
    a caller that then reads a file by its path must compare what it reads with the SHA-256 that it
    got here. A file that the system does not let this process open raises the OSError.

    out/ itself is not checked here: if it is a symbolic link or another user's folder, it is used
    all the same. A forged response then shows only in the tag's owner and mode, in each file's type
    and link count and, with key, in the tag's value. These are reported in Member.faults (a file
    that cannot be opened raises, as above), so a caller must read the faults: that answered() lists
    an id says nothing. private_folders(spool, "out") raises PermissionError for such an out/ and
    creates a missing one; the service does not start with such an out/.

    name must be an id, 64 lower-case hex digits, as answered() returns it. Any other string raises
    ValueError: a string such as "../in/x", which a front end might pass on from its own client,
    would name files outside out/."""
    if not ID.match(name):
        raise ValueError("%r is not the id of a response (64 lower-case hex digits)" % (name,))
    order, small = ("y", TAG, "proof", FAILED), {"y": 4, TAG: TAG_BYTES}
    opened, about, faults, read, ticks, held, fits = {}, {}, [], {}, [], {}, None
    try:
        # The service names .y first, then .tag, and .proof or .failed last. The names are opened in
        # the reverse order, so that a response that gets its names between two opens is seen
        # without its .proof (not there yet) and never with .proof alone or without its tag
        # (damaged). One exception: a service that starts removes the response of every request that
        # it finds left in work/ (after a power loss that response may be truncated), .y first, so
        # that a call at that moment can find .proof without .y.
        for ending in order[::-1]:
            try:  # O_NONBLOCK: to open a named pipe would otherwise wait for a writer
                opened[ending] = os.open(path_of(spool, name, ending), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            except FileNotFoundError:
                continue
            except OSError as why:
                if why.errno not in (errno.ELOOP, errno.ENXIO):  # a symbolic link, a socket
                    raise
                about[ending] = None
                continue
            about[ending] = os.fstat(opened[ending])
        found = [e for e in order if e in about]
        failed = FAILED in found
        if found != ([TAG, FAILED] if failed else list(order[:3])):
            faults.append("the response has the files %s, not exactly %s or exactly %s" % (", ".join(found or ["none"]), ", ".join(order[:3]), ", ".join((TAG, FAILED))))
        for ending in found:
            s = about[ending]
            if s is None or not stat.S_ISREG(s.st_mode):
                faults.append("%s is not a plain file" % ending)
                continue
            if s.st_nlink != 1:
                faults.append("%s has %d links" % (ending, s.st_nlink))
            if s.st_size != {FAILED: 0, **small}.get(ending, s.st_size):
                faults.append("%s has %d bytes" % (ending, s.st_size))
            ticks.append(s.st_mtime_ns)
            if ending == TAG:
                if s.st_uid != os.geteuid():
                    faults.append("%s belongs to user %d, not to user %d, who reads it" % (ending, s.st_uid, os.geteuid()))
                if stat.S_IMODE(s.st_mode) != 0o600:
                    faults.append("%s has the mode %04o, not 0600" % (ending, stat.S_IMODE(s.st_mode)))
            else:
                read[ending] = {"size": s.st_size}
            # Each file is read at most once. y and the tag are kept if their lengths are right. The
            # proof is read after them, so its pieces can go into the HMAC as they are read.
            begun = tag_begun(key, name, held["y"], s.st_size) if key is not None and ending == "proof" and len(held) == 2 else None
            if s.st_size == small.get(ending):
                held[ending] = os.read(opened[ending], s.st_size + 1)
                digest, count = hashlib.sha256(held[ending]), len(held[ending])
            elif sums or begun:
                digest, count = hashlib.sha256(), 0
                while piece := os.read(opened[ending], 8 << 20):
                    for h in [digest] * sums + [begun] * bool(begun):
                        h.update(piece)
                    count += len(piece)
            else:
                continue
            if count != s.st_size:
                faults.append("%s changed its length while it was read" % ending)
            if sums and ending != TAG:
                read[ending]["sha256"] = digest.hexdigest()
            if begun:
                fits = hmac.compare_digest(held[TAG], begun.digest())
        if len(set(ticks)) > 1:
            faults.append("the files do not have one modification time")
        if key is not None and failed and TAG in held:
            fits = hmac.compare_digest(held[TAG], tag_of_failed(key, name))
        # If a file that the comparison needs is missing or malformed, no comparison was made (fits
        # is None), and a fault above says why.
        if key is not None and not fits and (fits is not None or not faults):
            faults.append("the tag is not that of this id and these bytes")
        return Member(name, failed, ticks[-1] if ticks else 0, read, faults)
    finally:
        for descriptor in opened.values():
            os.close(descriptor)


# Returns the SHA-256 of the file at path, in hex, as in Member.files.
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(8 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def stop(spool):
    """Asks the service to stop, by the empty file `stop`: it takes no further batch, finishes what
    it has taken, removes the file and, if nothing has failed, exits with code 0. A service that
    starts removes a `stop` file that is already there, and does not obey it. A symbolic link under
    the stop file's name is not followed: it raises OSError, and nothing is created."""
    os.close(os.open(os.path.join(home(spool), "stop"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666))


# A check that no other account can change the path to a directory. The constants and functions
# from here to cure() have the same code as in run_suite.py. A script that creates a directory
# whose path it was given, or works below one, copies this block whole and changes nothing in it.
# The block needs `import os`, `import re` and `import stat`. Its printable() stays even where the
# script has one of its own.
#
# Call others_can_change() on the existing parent of the directory to be created, or on the
# directory itself if it exists. If it returns (place, why), print
# "<script>: " + refusal(<what>, <path>, place, why) on standard error and exit with code 2.
# run_suite.py adds a clause on an option of its own to that message, and has other messages for a
# place that is missing or cannot be examined.
# If others_can_change() returns None, use the parent's resolved path from then on.
#
# run_suite.py also rejects the directory that its environment variable BUILD names if that has a
# default access control list. That check is not in this block, so nothing here returns `ACL`.

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


def words_in(path):
    """Returns a file's little-endian 32-bit words, followed by None if the length is not a multiple
    of 4. The None makes pack() raise struct.error, so that such a file gives no request."""
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<%dI" % (len(data) // 4), data[: len(data) // 4 * 4])) + [None] * bool(len(data) % 4)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "lay":
        with open(sys.argv[3], "rb") as request_file:
            request_bytes = request_file.read()
        try:
            # Checks the directories above the spool directory, on the path that lay() uses
            # (home()). lay() itself checks the spool directory.
            found = others_can_change(os.path.dirname(home(sys.argv[2])))
            if found and found[1] == GONE:  # exit code 1, in the words of `lay`'s own message
                raise FileNotFoundError("%s does not exist: a request is submitted to an existing spool directory only" % printable(home(sys.argv[2])))
            if found:
                print("spool.py: " + refusal("spool directory", sys.argv[2], *found), file=sys.stderr)
                sys.exit(2)
            print(lay(sys.argv[2], request_bytes))
        except (FileNotFoundError, PermissionError) as fault:
            sys.exit("spool.py: %s" % fault)
    elif (len(sys.argv) == 7 and sys.argv[1] == "request" and sys.argv[2].isdigit()
          and (sys.argv[5] == "none" or sys.argv[5].isdigit())):  # fmt: skip
        laid = whole_reply(words_in(sys.argv[3]), words_in(sys.argv[4]), int(sys.argv[2]),
                           None if sys.argv[5] == "none" else int(sys.argv[5]))  # fmt: skip
        # a new file, for its owner alone whatever the umask: it holds a prompt and its reply
        with open(os.open(sys.argv[6], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600), "wb") as request_file:
            request_file.write(laid)
        print("%d:%d" % struct.unpack("<2I", laid[-8:]))
    else:
        sys.exit("usage: spool.py lay <spool> <request file: the tokens, then r and e, as little-endian 32-bit words>\n"
                 "       spool.py request <T> <prompt file> <reply file> <end token's id, or none> <request file to write>")
