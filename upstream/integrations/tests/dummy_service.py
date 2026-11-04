#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for the proving service (cp_prove -spool), for tests. It proves nothing.

Started by test_dummy_service.py or test_round_trip.py, which the runner starts; by hand: below a
directory of your own. (Runner: run_suite.py.)

It follows the protocol of a spool directory (spool.py). A request whose checked tokens count up
(next_token()) gets a dummy proof, any other an empty ID.failed. As with cp_prove, any entry named
"stop" ends it. It has no lock and does not answer what an earlier process left in work/. It exits
with code 1 if its parent process exits.

It removes names only in the directories that it holds open, never a link's target: a file ID.req
in in/ of a request's length that cannot be read, whose name is not the hash of its bytes or whose
ID has an entry in out/, with a printed line; a request in work/ once it is answered; an empty file
"stop". Like cp_prove, it leaves a file ID.req of another length in in/: it may be another
commitment's request. Between its look at an entry and the removal, only the same user could change
the entry. It accepts only a directory below tempfile.gettempdir(), which follows TMPDIR, and only
an empty one, in which it writes a mark file (mark.txt) that fits every commitment, or one with a
mark file of the right form, whatever commitment it names. cp_prove also takes, and marks, a
directory that holds only waiting requests in in/, an empty out/ and an empty file stop. The
stand-in cannot tell a real spool directory there from a test's.
"""

import argparse
import contextlib
import errno
import hashlib
import os
import re
import stat
import struct
import sys
import tempfile
import time

if __name__ == "__main__":  # run by path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spool

POLL_S = 0.02
REQUEST_NAME = re.compile(r"[0-9a-f]{64}\.req")  # other names in in/ are left alone
REMOVED = "removed in/%s unanswered: %s"  # the name, the cause


def positive_int(text):
    if not re.fullmatch(r"[1-9]\d{0,8}", text):
        raise argparse.ArgumentTypeError("%r is not a positive integer" % text)
    return int(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0], allow_abbrev=False)
    p.add_argument(
        "spool_dir", metavar="SPOOL_DIR", help="a directory below the temporary directory"
    )
    count = {"type": positive_int, "metavar": "N"}
    p.add_argument("--tokens", required=True, help="tokens in a request (T = 2**lt)", **count)
    p.add_argument("--batch", default=8, help="the most requests in a batch (default: 8)", **count)
    p.add_argument(
        "--first-take",
        default=1,
        help="take the first batch only when N requests are waiting, so that a test's batches do "
        "not depend on the machine's speed (default: 1)",
        **count,
    )
    p.add_argument(
        "--end-when-empty",
        action="store_true",
        help="exit with code 0 when no request is waiting, like cp_prove -end-when-empty",
    )
    args = p.parse_args(argv)
    if args.end_when_empty and args.first_take > 1:
        p.error("--first-take cannot be used with --end-when-empty")
    args.spool_dir = os.path.realpath(args.spool_dir)
    temporary = os.path.realpath(tempfile.gettempdir())
    if not args.spool_dir.startswith(temporary + os.sep):
        p.error("SPOOL_DIR must be a directory below %s, not %s" % (temporary, args.spool_dir))
    if not os.path.isdir(args.spool_dir):
        p.error("SPOOL_DIR %s is not a directory" % args.spool_dir)
    return args


def open_subdirectory(parent, name, shown):
    """Make a directory if it is missing, and return a descriptor of it.

    Raise spool.UnsafePath, as cp_prove refuses to start, if it is a link, belongs to another user
    or gives its group or others any access."""
    with contextlib.suppress(FileExistsError):
        os.mkdir(name, 0o700, dir_fd=parent)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except OSError as error:
        if error.errno not in (errno.ELOOP, errno.ENOTDIR):
            raise
        raise spool.UnsafePath("%s is a link or not a directory" % shown) from None
    found = os.fstat(fd)
    if found.st_uid != os.geteuid() or found.st_mode & 0o077:
        raise spool.UnsafePath("%s belongs to another user, or is open to others" % shown)
    return fd


class Directories:
    """Descriptors of the spool directory and its subdirectories.

    Every file is reached through them, so a link that is later put in a directory's place cannot
    lead the service to other files."""

    def __init__(self, top, shown):
        self.top = top
        self.incoming = open_subdirectory(top, "in", shown + "/in")
        os.close(open_subdirectory(self.incoming, ".tmp", shown + "/in/.tmp"))
        self.work = open_subdirectory(top, "work", shown + "/work")
        self.out = open_subdirectory(top, "out", shown + "/out")
        self.tmp = open_subdirectory(self.out, ".tmp", shown + "/out/.tmp")


def move(name, source, target):
    """Move a file to another directory, but raise FileExistsError if the name is taken there."""
    os.link(name, name, src_dir_fd=source, dst_dir_fd=target, follow_symlinks=False)
    os.unlink(name, dir_fd=source)


def read_request(name, directory, size):
    """Return the content of a regular file of this size, or None."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except OSError:
        return None
    with open(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = f.read(size + 1)
    return data if len(data) == size else None


def next_token(token):
    """Return the next token of the stand-in for the committed model, which counts up."""
    return (token + 1) & 0xFFFFFFFF


def dummy_proof(request):
    """Return 200 bytes: a proof file's header for the parameter file's version H821 and leaf table
    0; it contains request's hex SHA-256."""
    digest = hashlib.sha256(request).hexdigest().encode()
    return (b"CPPFH821" + struct.pack("<I", 0) + b"dummy " + digest).ljust(200, b"\0")


def say(line):
    """Print a line. If that fails, as when nobody reads the output any more, send the output to
    os.devnull from then on: the service finishes its batch and keeps its usual exit code. If that
    switch fails as well (os.devnull cannot be opened because no file descriptor is left, say), the
    error ends the service like any other OSError: main() prints a line on stderr and returns 1, and
    as the line that could not be printed still cannot be flushed at exit, Python then ends the
    process with status 120."""
    try:
        print(line, flush=True)
    except OSError:
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, sys.stdout.fileno())
        os.close(null)


def request_size(args):
    """Return the length of a request file: the tokens, two more words and the nonce."""
    return 4 * (args.tokens + 2) + spool.NONCE_BYTES


def waiting(args, directories):
    """Return the regular files ID.req of a request's length in in/, oldest first.

    Other entries are left alone."""
    found = []
    for name in os.listdir(directories.incoming):
        if REQUEST_NAME.fullmatch(name):
            entry = os.stat(name, dir_fd=directories.incoming, follow_symlinks=False)
            if stat.S_ISREG(entry.st_mode) and entry.st_size == request_size(args):
                found.append((entry.st_mtime_ns, name))
    return [name for _, name in sorted(found)]


def take(args, directories):
    """Move up to one batch of requests from in/ to work/. Return {request ID: content}.

    Delete a file that cannot be read or whose name is not the hash of its bytes, and a request
    whose ID has an entry in out/, and print a line for each."""
    taken = {}
    for name in waiting(args, directories)[: args.batch]:
        data = read_request(name, directories.incoming, request_size(args))
        if data is None:
            cause = "it could not be read"
        elif spool.request_id(data) != name[:-4]:
            cause = "its name is not the hash of its bytes"
        elif any(entry.startswith(name[:-3]) for entry in os.listdir(directories.out)):
            cause = "its ID has an entry in out/"
        else:
            move(name, directories.incoming, directories.work)
            taken[name[:-4]] = data
            continue
        os.remove(name, dir_fd=directories.incoming)
        say(REMOVED % (name, cause))
    return taken


def response_files(request_id, data):
    """Return the response to a request file's content, as (suffix, content) pairs."""
    request, key = data[: -spool.NONCE_BYTES], spool.tag_key(data)
    *tokens, start, end = struct.unpack("<%dI" % (len(request) // 4), request)
    # Position t checks token t + 1 (spool.make_request()).
    if any(tokens[t + 1] != next_token(tokens[t]) for t in range(start, min(end, len(tokens) - 1))):
        # The part of cp_prove's line that end_to_end.py reads; 26: the policy is not met.
        say("request %s: failed: cause 26" % request_id)
        return (spool.TAG, spool.failed_tag(key, request_id)), (spool.FAILED, b"")
    y = struct.pack("<I", next_token(tokens[-1]) if end == len(tokens) else 0)
    proof = dummy_proof(request)
    return ("y", y), (spool.TAG, spool.response_tag(key, request_id, y, proof)), ("proof", proof)


def respond(taken, directories):
    """Write one batch's responses, all with one modification time, and delete its requests."""
    now = time.time_ns()
    for request_id, data in taken.items():
        # The file that completes a response gets its final name last.
        for suffix, content in response_files(request_id, data):
            name = request_id + "." + suffix
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            with open(os.open(name, flags, 0o600, dir_fd=directories.tmp), "wb") as f:
                f.write(content)
                f.flush()
                os.utime(f.fileno(), ns=(now, now))
            move(name, directories.tmp, directories.out)
    # Only now, as cp_prove does: a request that has left work/ has its whole batch in out/.
    for request_id in taken:
        os.remove(request_id + ".req", dir_fd=directories.work)


def stop_entry(top):
    """Return None if "stop" is missing, True if it was an empty regular file (now deleted), else
    False."""
    try:
        found = os.stat("stop", dir_fd=top, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISREG(found.st_mode) and found.st_size == 0:
        os.remove("stop", dir_fd=top)
        return True
    return False


def serve(args, directories):
    parent = os.getppid()
    started = False
    while os.getppid() == parent:
        if stop_entry(directories.top) is not None:
            return 0
        count = len(waiting(args, directories))
        if count >= (1 if started else args.first_take):
            started = True
            respond(take(args, directories), directories)
        elif args.end_when_empty and not count:
            return 0
        else:
            time.sleep(POLL_S)
    return 1


def main(argv=None):
    args = parse_args(argv)
    os.umask(0o077)
    is_now = os.lstat(args.spool_dir)  # spool.open_private_dir() looks again, at what it has opened
    if is_now.st_uid != os.geteuid() or is_now.st_mode & 0o077:
        sys.exit("dummy_service.py: error: %s is not your own private directory" % args.spool_dir)
    try:
        found = spool.others_can_change(os.path.dirname(args.spool_dir))
        if found:  # spool.open_private_dir() looks too; here it is seen without opening spool.py
            raise spool.UnsafePath(spool.refusal("spool directory", args.spool_dir, *found))
        with spool.open_private_dir(args.spool_dir) as top:  # checks its owner and mode
            # A directory above may have been replaced by a link since parse_args() looked.
            if os.readlink("/proc/self/fd/%d" % top) != args.spool_dir:
                raise spool.UnsafePath("%s no longer leads to the same directory" % args.spool_dir)
            # Before anything else is made or removed.
            spool.check_mark(top, spool.ANY_COMMITMENT if spool.is_unused(top) else None, create=True)
            directories = Directories(top, args.spool_dir)
            if stop_entry(top) is False:  # an empty one is a leftover, and is deleted
                raise spool.UnsafePath("%s/stop is not an empty regular file" % args.spool_dir)
            # Printed after that: a later "stop" is not a leftover.
            print("dummy_service.py: a stand-in for tests; it proves nothing", flush=True)
            return serve(args, directories)
    except (OSError, spool.UnsafePath) as error:
        print("dummy_service.py: error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
