#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Example client: get a reply and its proof from a vLLM server, then check the proof.

The server must run proof_middleware.py. check_proof.py gets a request file that is built only from
the command line and the token IDs.

A proof covers token IDs, not text, and the script does not compare the two. With --prompt or
--chat the server also chooses the prompt's token IDs and where the reply begins, so the best exit
code is 24, never 0.

Threat model: the server and other local users are untrusted; root and the user's own account are
trusted. The script refuses an --out-dir below a directory that another account may be able to
change; under sudo, the account that ran sudo does not count as another (LIMITS.md says what that
costs). It then opens the directories above --out-dir one by one, judges each again once it is open,
and holds --out-dir open: every file in it is named relative to that descriptor, and check_proof.py
gets the request and the proof's directory as open descriptors. So a user who renames a directory on
the way, or puts a link in its place, leads no file of a run elsewhere once the directories are
open, and changes nothing that the run checks. Until then, the account that ran sudo can still
choose the directory that the run uses: another in a directory of its own or, by a link that it
lays on the way before the run resolves the path, any that passes both looks, root's own too.
Between two runs such a user can put another directory under the name: --resume takes it only if
it is the caller's own and not writable by others. The script needs Linux (O_PATH; /dev/fd).
"""

# ruff: noqa: E402 (the other imports must come after isolate())
import posix  # built in: under Python 3.10 with -S, a file on PYTHONPATH could stand in for os
import sys


def isolate():
    """Start the program again in Python's isolated mode (-I), unless it already runs in it.

    PYTHON* variables, the user's site-packages and the current directory then do not decide what
    is imported below: check_proof.py and spool.py are loaded by path from this script's real
    directory, and everything else comes from the interpreter's own directories. Not covered: what
    Python itself takes from those three while it first starts (for example a sitecustomize.py on
    PYTHONPATH, or with -m what runpy imports), which only "python3 -I" on the command line
    excludes; and the installed command, whose wrapper imports this module. Of the other options
    only -B and -S are kept. If Python cannot be started again, print "result: outcome=error" and
    an error line, and exit with status 2.
    """
    if sys.flags.isolated:
        return
    kept = ["-B"] * sys.flags.dont_write_bytecode + ["-S"] * sys.flags.no_site
    try:
        posix.execv(sys.executable, [sys.executable, "-I", *kept, __file__, *sys.argv[1:]])
    except (OSError, ValueError) as failure:  # ValueError: sys.executable is empty
        reason = getattr(failure, "strerror", None) or failure
        print("result: outcome=error", flush=True)
        message = "error: cannot start %r in isolated mode: %s"
        print(message % (sys.executable, reason), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    isolate()
    # Leave no __pycache__ in a checkout. Not set on import: the setting is process-wide.
    sys.dont_write_bytecode = True

import argparse
import collections
import contextlib
import errno
import functools
import hashlib
import http.client
import importlib.util
import json
import os
import re
import signal
import ssl
import stat
import struct
import subprocess
import time
import urllib.error
import urllib.request

try:
    if __name__ == "__main__":
        # Load the two neighbours by path. Their directory is not put on sys.path: for an installed
        # copy it is site-packages, which must not come before the standard library.
        for name in ("check_proof", "spool"):
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), name + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            sys.modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules[name])
    import check_proof
    import spool
    from check_proof import (
        GONE,
        MAX_FETCHED_BYTES,
        OPEN,
        THEIRS,
        NoRedirects,
        one_line,
        others_can_change,
        policy_fault,
        refusal,
        trusted_owners,
    )
except BaseException as error:  # whatever a neighbour raises while it loads, sys.exit() too
    print("result: outcome=error", flush=True)
    if isinstance(error, KeyboardInterrupt):
        print("error: interrupted (run the same command again)", file=sys.stderr)
        sys.exit(2)
    advice = "install them with pip install ./integrations"
    if __name__ == "__main__":
        advice = "they must be in this script's directory, unchanged"
    message = "error: cannot import check_proof.py and spool.py (%s: %r): %s"
    print(message % (type(error).__name__, str(error), advice), file=sys.stderr)
    sys.exit(2)

CHECK_SCRIPT = check_proof.__file__

# These must agree with proof_middleware.py.
PREFIX = "/plugins/proof/"
POLICY_HEADER, ID_HEADER = "x-proof-policy", "x-proof-id"
DOWNLOADS = ("y.bin", "proof.bin")
PADDING = 0  # the token ID that fills a request file after the reply
POLICY_NAME = "[A-Za-z0-9_-]{1,64}"

VERIFIED, REJECTED, ERROR, SERVER_CHOSE_PROMPT, TRY_AGAIN = 0, 1, 2, 24, 75
# check_proof.py's exit codes, each with the values of outcome= in its "result:" line.
OUTCOMES = {
    VERIFIED: ("verified", "within-limits"),
    REJECTED: ("rejected", "no-proof"),
    ERROR: ("error",),
    14: ("not-a-verification",),
}
# claim= beside each outcome= of a pass. No "next-token": such a pass checks no token of the reply.
CLAIMS = {"verified": "exact-match", "within-limits": "policy"}
# outcome= in this script's own "result:" line, by exit code. With --ask-only: STORED.
OWN_OUTCOMES = {
    REJECTED: "rejected",
    ERROR: "error",
    SERVER_CHOSE_PROMPT: "server-chose-prompt",
    TRY_AGAIN: "try-again",
}
STORED = "stored"
SERVER_CHOSE_PROMPT_WARNING = (
    "ask_and_check.py: warning: the server chose the prompt's token IDs and where the reply begins,"
    " so even a pass below is not a full verification (use --prompt-ids)"
)
DESCRIPTION = """Ask a vLLM server that runs proof_middleware.py for a reply and its proof, and
check the proof with check_proof.py. Take --commitment-id from where the
commitment is published, never from this server. The reply must end with one of
--end-token, unless it is exactly --max-tokens tokens long. ps shows the command
line to other local users: to keep the prompt off it, give it as "-"."""
EXIT_CODES = """exit codes:
   0  verified, or within limits (a pass under the clipped-gap limits that you
      named after "--"); in check_proof.py's result line, claim= says what was
      shown; record=1, commitment_assumed_rounds= and work= say what it rests on
   1  rejected: check_proof.py rejected the proof, the prover gave no proof or
      the server no longer has it, or what the server sent contradicts the
      command line (commitment ID, prompt IDs, end-token rule, request length)
   2  usage or environment error; a closed policy; a redirect, a 4xx status, a
      TLS error while connecting or something malformed from the server; the
      check failed to run, was killed or gave no consistent result; or the
      script was interrupted (Ctrl-C)
  14  not a verification: for example, a test build of the verifier; or counted
      proof of work under --no-proof-of-work
  24  instead of 0: the proof passed, but for prompt token IDs that the
      server chose (--prompt, --chat): not a full verification
  75  try again: the proof is not ready or, with --ask-only, not checked yet;
      the server was not reached, broke off, or sent this script a 5xx status
The last stdout line that begins with "result:" has the outcome: outcome=WORD.
Only --help and a usage error that this script finds itself print no such line."""
# Without the options that this script sets itself; the caller may override --fetch-from.
PASSED_ON_HELP = """options of check_proof.py that may follow "--" (see check_proof.py --help):
  --policy --C --theta --cap --s
  --fetch-from --layout-file --threads --show-all --no-proof-of-work"""
PASSED_ON = " ".join(PASSED_ON_HELP.splitlines()[1:]).split()
WITHOUT_VALUE = ("--show-all", "--no-proof-of-work")
ENVIRONMENT_HELP = """environment variable:
  VLLM_API_KEY  the API key; sent only over https or to this machine"""

# socket time-outs, in seconds
TIMEOUT_S, REPLY_TIMEOUT_S = 60, 3600
LARGEST_JSON, LARGEST_FILE = 64 << 20, MAX_FETCHED_BYTES
STDIN = "-"  # as the value of --prompt-ids or --prompt: the prompt is on standard input
LARGEST_PROMPT = 1 << 20  # bytes on standard input
PROOF_ID = "[0-9a-f]{64}"


class TryAgain(Exception):
    """A temporary failure: main() exits with TRY_AGAIN."""


def fail(code, message, *values, outcome=None, start="error: "):
    """Print this script's "result:" line and one error line, and exit."""
    with contextlib.suppress(OSError):
        # Flush, so that the error line comes last in a shared file.
        print("result: outcome=%s" % (outcome or OWN_OUTCOMES[code]), flush=True)
    print(start + message % values, file=sys.stderr)
    sys.exit(code)


def new_opener(args):
    """Return the opener for calls to --server.

    The API key must reach only --server: no redirects and, for plain http with a key, no proxy.
    """
    handlers = [NoRedirects]
    if args.api_key and not check_proof.HTTPS.match(args.server):
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def new_request(args, path, data=None, headers=None, method=None):
    headers = {"Content-Type": "application/json", **(headers or {})}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    return urllib.request.Request(args.server.rstrip("/") + path, data, headers, method=method)


def hide_proof_id(text):
    """Anyone who has the proof ID can fetch the proof, and error lines end up in logs."""
    return re.sub(PROOF_ID, "<proof ID>", text)


def http_call(args, path, body=None, headers=None, timeout=TIMEOUT_S):
    """Send a GET, or a POST with a JSON body, and return the open response.

    Exit on a redirect or a 4xx status. Raise TryAgain on 5xx, or if the server cannot be reached.
    """
    request = new_request(args, path, None if body is None else json.dumps(body).encode(), headers)
    route = "%s %s" % (request.get_method(), hide_proof_id(path))
    try:
        return args.opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            message = "the server returned a redirect (status %d) for %s, which is not followed"
            fail(ERROR, message, error.code, route)
        text = read(error, 2000).decode(errors="replace")
        with contextlib.suppress(ValueError, KeyError, TypeError):
            text = json.loads(text)["error"]["message"]  # vLLM's error format
        text = one_line(hide_proof_id(str(text)))
        message = "the server returned status %d for %s: %s" % (error.code, route, text)
        if error.code >= 500:
            raise TryAgain(message)
        fail(ERROR, "%s%s", message, " (check VLLM_API_KEY)" * (error.code == 401))
    except OSError as error:
        reason = getattr(error, "reason", error)
        message = "cannot reach %s for %s: %s" % (args.server, route, one_line(reason))
        closed_early = (ssl.SSLEOFError, ssl.SSLZeroReturnError)
        if isinstance(reason, ssl.SSLError) and not isinstance(reason, closed_early):
            fail(ERROR, "%s", message)  # retrying does not fix a wrong scheme or certificate
        raise TryAgain(message)


def read(response, size):
    """Read at most size bytes from the server. Raise TryAgain if the connection fails.

    Every read from the server must go through here or http_call().
    """
    try:
        return response.read(size)
    except (OSError, http.client.HTTPException) as error:
        raise TryAgain("the connection to the server failed: %s" % one_line(error))


def read_json(response):
    with response:
        data = read(response, LARGEST_JSON)
        if len(data) < LARGEST_JSON and getattr(response, "length", 0):  # bytes still to come
            raise TryAgain("the connection to the server failed: the body ended early")
        try:
            return json.loads(data)
        except (ValueError, RecursionError):
            limit = LARGEST_JSON >> 20
            fail(ERROR, "the server did not send JSON, or sent more than %d MiB", limit)


def compare_commitment_id(args):
    """Exit unless the server's commitment ID for the policy is the caller's and the policy is open.

    Called only before a prompt is sent: a closed policy still serves its proofs.
    """
    policies = read_json(http_call(args, PREFIX + "policies"))
    policies = policies.get("policies") if isinstance(policies, dict) else None
    if not isinstance(policies, dict) or not isinstance(policies.get(args.policy_name), dict):
        names = one_line(", ".join(map(str, policies))) if isinstance(policies, dict) else ""
        message = "the server has no policy named %s (it has: %s)"
        fail(ERROR, message, args.policy_name, names or "none")
    theirs = policies[args.policy_name].get("commitment_id")
    if not isinstance(theirs, str) or not theirs:
        fail(REJECTED, "the server publishes no commitment ID for the policy %s", args.policy_name)
    if theirs != args.commitment_id:
        # The server's ID is not shown: a caller who pastes it into --commitment-id accepts whatever
        # model the server has committed to.
        message = (
            "the server's commitment ID for the policy %s differs from --commitment-id; no prompt"
            " was sent (compare --policy-name and --commitment-id with where the commitment is"
            " published)"
        )
        fail(REJECTED, message, args.policy_name)
    if policies[args.policy_name].get("closed") is True:
        fail(ERROR, "the policy %s is closed: it takes no new requests", args.policy_name)


def request_body(args):
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        prompt = {"messages": messages, "max_completion_tokens": args.max_tokens}
    else:
        prompt = args.prompt if args.prompt_ids is None else args.prompt_ids
        prompt = {"prompt": prompt, "max_tokens": args.max_tokens}
    # The middleware makes a proof only for a request at temperature 0.
    return {"model": args.model, **prompt, "temperature": 0, "return_token_ids": True}


def is_token_ids(value):
    if not isinstance(value, list) or not value:
        return False
    return all(type(token) is int and 0 <= token < 1 << 32 for token in value)


def is_well_formed(served):
    return (
        isinstance(served, dict)
        and is_token_ids(served.get("prompt_token_ids"))
        and is_token_ids(served.get("reply_token_ids"))
        and isinstance(served.get("proof_id"), str)
        and bool(re.fullmatch(PROOF_ID, served["proof_id"]))
        and isinstance(served.get("text", 0), (str, type(None)))
    )


def ask(args):
    """Send the prompt. Return the proof ID, token IDs and text as a dictionary."""
    headers = {POLICY_HEADER: args.policy_name}
    path = "/v1/chat/completions" if args.chat else "/v1/completions"
    response = http_call(args, path, request_body(args), headers, REPLY_TIMEOUT_S)
    proof_id, body = response.headers.get(ID_HEADER), read_json(response)
    if not re.fullmatch(PROOF_ID, proof_id or ""):
        message = "the reply has no valid %s header (does the server run proof_middleware.py?)"
        fail(ERROR, message, ID_HEADER)
    try:
        choice = body["choices"][0]
        served = {
            "proof_id": proof_id,
            "prompt_token_ids": (body if args.chat else choice)["prompt_token_ids"],
            "reply_token_ids": choice["token_ids"],
            "text": choice["message"]["content"] if args.chat else choice["text"],
        }
    except (KeyError, IndexError, TypeError):
        served = None
    if not is_well_formed(served):
        fail(ERROR, "the reply has no usable token IDs or text")
    if args.prompt_ids is not None and served["prompt_token_ids"] != args.prompt_ids:
        fail(REJECTED, "the server returned other prompt token IDs than were sent")
    return served


def wait_for_proof(args, proof_id):
    """Poll the job for at most --wait seconds while it is "waiting". Return the job."""
    deadline, pause = time.monotonic() + args.wait, 0.25
    while True:
        job = read_json(http_call(args, PREFIX + "jobs/" + proof_id))
        states = ("waiting", "proved", "no_proof", "not_provable", "expired")
        if not isinstance(job, dict) or job.get("state") not in states:
            shown = one_line(hide_proof_id(json.dumps(job)[:2000]))
            fail(ERROR, "the server's description of the job is unusable: %s", shown)
        if job["state"] != "waiting" or time.monotonic() >= deadline:
            return job
        time.sleep(pause)
        pause = min(2 * pause, 10)


def download_proof(args, proof_id, proof):
    """Download the proof's files into the Directory 'proof'.

    A file gets its final name only when complete, so a download can be resumed. If a download
    fails, its temporary file is deleted. A killed run's temporary files are kept.
    """
    for name in DOWNLOADS:  # fixed names, never the server's
        if proof.has(name):
            continue
        response = http_call(args, PREFIX + "jobs/%s/%s" % (proof_id, name))
        with response, proof.create_temporary(name) as f:
            try:
                announced = response.headers.get("content-length") or ""
                length = int(announced) if re.fullmatch("[0-9]{1,10}", announced) else -1
                if not 0 <= length <= LARGEST_FILE:
                    message = "the server sent no Content-Length header, or one over %d MiB, for %s"
                    fail(ERROR, message, LARGEST_FILE >> 20, name)
                while f.tell() < length:  # never more than the announced length
                    piece = read(response, min(1 << 20, length - f.tell()))
                    if not piece:
                        raise TryAgain("the download of %s ended early" % name)
                    f.write(piece)
                f.flush()
                proof.link(f.name, name)
            finally:
                proof.unlink(f.name)


def delete_servers_copy(args, proof_id):
    """Ask the server to delete the job and its proof, ignoring failures and interrupts."""
    request = new_request(args, PREFIX + "jobs/" + proof_id, method="DELETE")
    with contextlib.suppress(OSError, http.client.HTTPException, KeyboardInterrupt):
        args.opener.open(request, timeout=TIMEOUT_S).close()


# The files in --out-dir:
#   served.json    what the server sent, and the options that defined the request
#   proof/         y.bin and proof.bin (NAME.part.RANDOM during a download)
#   no_proof.txt   instead of proof/: the server's reason
#   request.bin    a record of the request file, written once


# This script never overwrites a file, and deletes only files that the same run created.
NEW_FILE = os.O_RDWR | os.O_CREAT | os.O_EXCL
DIRECTORY = os.O_DIRECTORY | os.O_NOFOLLOW


class NotPrivate(Exception):
    """Why other users could change a directory."""


class OwnedByAnother(NotPrivate):
    """The directory belongs, or may belong, to another account."""


class IsSymlink(NotPrivate):
    """The name is a symbolic link."""


def whole_paths(method):
    """Make a method of Directory put whole paths into its OSError, for the message."""

    @functools.wraps(method)
    def wrapped(self, *names, **options):
        try:
            return method(self, *names, **options)
        except OSError as error:
            error.filename = error.filename and self.path_of(error.filename)
            error.filename2 = error.filename2 and self.path_of(error.filename2)
            raise

    return wrapped


class Directory:
    """A directory that the script holds open.

    Its files are created, read, linked and deleted by name relative to the descriptor, never by
    path. So a user who renames the directory or one above it, or puts a link in its place, leads
    none of that elsewhere. 'path' is for messages only.
    """

    def __init__(self, fd, path):
        self.fd, self.path = fd, path

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)

    def path_of(self, name):
        return os.path.join(self.path, name)

    def names(self):
        return os.listdir(self.fd)

    @whole_paths
    def has(self, name):
        """Return whether the name exists, even as a dangling symlink."""
        try:
            os.lstat(name, dir_fd=self.fd)
        except FileNotFoundError:
            return False
        return True

    @whole_paths
    def create(self, name, mode="wb"):
        """Create a file that only the current user can read and write.

        Raise FileExistsError if the name exists, even as a symlink.
        """
        return open(name, mode, opener=lambda _, __: os.open(name, NEW_FILE, 0o600, dir_fd=self.fd))

    def create_temporary(self, name, mode="wb"):
        return self.create("%s.part.%s" % (name, os.urandom(8).hex()), mode)

    @whole_paths
    def open_stored(self, name, mode="r"):
        """Open a regular file that an earlier run stored.

        Raise OSError for a symlink, a FIFO, etc.
        """
        if not stat.S_ISREG(os.lstat(name, dir_fd=self.fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file (choose a new --out-dir)", name)
        # If the file is swapped after lstat(): O_NOFOLLOW refuses a symlink, and O_NONBLOCK keeps a
        # FIFO from blocking the open.
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        return open(os.open(name, flags, dir_fd=self.fd), mode)

    @whole_paths
    def link(self, name, new):
        """Give a file a second name. Unlike a rename, it never overwrites: FileExistsError."""
        os.link(name, new, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)

    @whole_paths
    def unlink(self, name):
        os.unlink(name, dir_fd=self.fd)

    @whole_paths
    def make_directory(self, name):
        """Make a directory with mode 0700, unless the name exists."""
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=self.fd)

    @whole_paths
    def open_private(self, name):
        """Open the directory 'name' in this one, and return it as a Directory.

        Raise NotPrivate unless it is a real directory (not a symlink), owned by the user who runs
        the script (even if that is root), and not writable by its group or by others. What is
        judged is the open directory. If it may not be opened, raise OwnedByAnother where lstat()
        shows another owner or is refused too, and let the PermissionError pass where it is the
        user's own.
        """
        mine, owner_words = os.geteuid(), "its owner is uid %d, and this script runs as uid %d"
        try:
            fd = os.open(name, os.O_RDONLY | DIRECTORY, dir_fd=self.fd)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.ELOOP, errno.ENOTDIR):
                raise
            try:
                found = os.lstat(name, dir_fd=self.fd)  # for the words only
            except PermissionError:  # the directory that holds 'name' cannot be searched
                words = "Permission denied: it may be another account's"
                raise OwnedByAnother(words) from None
            if error.errno == errno.EACCES and found.st_uid == mine:
                raise
            if error.errno == errno.EACCES:
                raise OwnedByAnother(owner_words % (found.st_uid, mine)) from None
            if stat.S_ISLNK(found.st_mode):
                raise IsSymlink("it is a symlink") from None
            raise NotPrivate("it is not a directory") from None
        found, problem = os.fstat(fd), None
        if found.st_uid != mine:
            problem = OwnedByAnother(owner_words % (found.st_uid, mine))
        elif found.st_mode & 0o022:
            mode = stat.S_IMODE(found.st_mode)
            problem = NotPrivate("its group or other users can write to it (mode %04o)" % mode)
        if problem:
            os.close(fd)
            raise problem
        return Directory(fd, os.path.normpath(self.path_of(name)))


def keep_record(directory, part, request):
    """Hard-link the request file 'part' as request.bin. An earlier run's record is kept."""
    try:
        directory.link(part, "request.bin")
    except FileExistsError:
        words = "is not the request that was checked in this run"
        try:
            with directory.open_stored("request.bin", "rb") as f:
                same = f.read(len(request) + 1) == request
        except OSError:  # for example, this account may not read it, or it is a link
            same = False
            words = "was not compared with the request that was checked in this run, because this"
            words += " account cannot read it as a regular file"
        if not same:
            message = "note: %s %s; it was left as it is"
            print(message % (directory.path_of("request.bin"), words), file=sys.stderr)


def refuse(path, place, why):
    """Exit with the sentence for an --out-dir below a place that another account may change."""
    fail(ERROR, "%s", refusal("--out-dir", path, place, why), start="ask_and_check.py: ")


# What cannot_open() says of a directory that was there at the first look.
CHANGES = {
    errno.ENOENT: "it no longer exists",
    errno.ENOTDIR: "it is now a link or not a directory",
    errno.ELOOP: "it is now a link or not a directory",
}


def cannot_open(path, place, error):
    """Exit because open_above() cannot open 'place', a directory above --out-dir."""
    names = one_line(path, None), one_line(place, None)
    if error.errno not in CHANGES:
        message = "--out-dir %s cannot be used: %s cannot be opened: %s (choose a new --out-dir)"
        fail(ERROR, message, *names, error.strerror)
    message = (
        "--out-dir %s cannot be used: %s has changed since it was looked at: %s (run the command"
        " again, or choose a new --out-dir)"
    )
    fail(ERROR, message, *names, CHANGES[error.errno])


def open_above(parent, path):
    """Open 'parent', the directory above --out-dir, and return it as a Directory.

    'parent' must have no symbolic link in it: the caller resolves them. From "/" down, each
    directory is opened relative to the one before, without following a link, and judged once it is
    open, by the rule of others_can_change(): its owner is one that trusted_owners() names, and its
    group and others cannot write in it unless it has the sticky bit. So what is opened is what was
    judged, whatever another account renames meanwhile.
    """
    flags = os.O_PATH | DIRECTORY  # a directory above may allow this user to pass, not to read
    fd, walked = os.open(os.sep, flags), os.sep
    try:
        for name in [*filter(None, parent.split(os.sep)), None]:
            found = os.fstat(fd)
            if found.st_uid not in trusted_owners():
                refuse(path, walked, THEIRS)
            if found.st_mode & 0o022 and not found.st_mode & stat.S_ISVTX:
                refuse(path, walked, OPEN)
            if name is not None:
                walked = os.path.join(walked, name)
                try:
                    below = os.open(name, flags, dir_fd=fd)
                except OSError as error:
                    cannot_open(path, walked, error)
                os.close(fd)
                fd = below
    except BaseException:
        os.close(fd)
        raise
    return Directory(fd, parent)


def open_out_dir(path, resume):
    """Create --out-dir with mode 0700, or take an existing private directory. Return it open.

    Before that, exit if a directory above it is missing, cannot be examined, or may be changed by
    another account. The symbolic links above it are resolved once.
    """
    parent, name = os.path.split(path)
    real = os.path.realpath(parent)  # before the look: open_above() refuses a link that comes later
    found = others_can_change(parent)  # Directory.open_private() judges --out-dir itself
    if found and found[1] == GONE:
        message = "--out-dir %s cannot be used: %s does not exist"
        fail(ERROR, message, one_line(path, None), one_line(found[0], None))
    if found:
        fail(ERROR, "%s", refusal("--out-dir", path, *found), start="ask_and_check.py: ")
    with open_above(real, os.path.join(real, name)) as above:
        return create_or_open(above, name or os.curdir, resume)


def create_or_open(above, name, resume):
    if not resume:
        above.make_directory(name)  # if it exists, first_run() checks that it is empty
    try:
        return above.open_private(name)
    except FileNotFoundError:
        if not resume:
            message = "--out-dir %s disappeared before it could be opened (run the command again)"
            fail(ERROR, message, above.path_of(name))
        message = (
            "--out-dir %s does not exist, so there is nothing to resume (run the command without"
            " --resume)"
        )
        fail(ERROR, message, above.path_of(name))
    except (NotPrivate, PermissionError) as problem:
        advice = "choose a new --out-dir, without --resume" if resume else "choose a new directory"
        if isinstance(problem, IsSymlink):
            advice = "name the directory itself, not a link to it"
        elif resume and isinstance(problem, OwnedByAnother):
            advice = (
                "resume as the account that started this run: with sudo if it was started with"
                " sudo, without sudo if not"
            )
        reason = getattr(problem, "strerror", None) or problem
        message = "--out-dir %s cannot be used: %s (%s)"
        fail(ERROR, message, os.path.normpath(above.path_of(name)), reason, advice)


def first_run(args, directory, sent):
    """Compare the commitment IDs, send the prompt and store what the server sent. Return it."""
    if directory.names():  # before anything is created
        message = (
            "--out-dir %s is not empty (use --resume to continue an earlier run, or choose a new"
            " directory)"
        )
        fail(ERROR, message, directory.path)
    part = "served.json.part"
    in_use = "%s exists: another run is using the directory (choose a new --out-dir)"
    try:
        # Before the server is called: reserves the directory, and shows that a reply can be stored.
        f = directory.create(part, "w")
    except FileExistsError:
        fail(ERROR, in_use, directory.path_of(part))
    try:
        with f:
            try:
                directory.link(part, part + ".link")  # served.json needs a hard link: try one
            except FileExistsError:
                fail(ERROR, in_use, directory.path_of(part + ".link"))
            except OSError:
                message = (
                    "--out-dir %s is on a file system without hard links (choose another directory)"
                )
                fail(ERROR, message, directory.path)
            directory.unlink(part + ".link")
            compare_commitment_id(args)
            served = ask(args)
            json.dump({**served, "sent": sent}, f)
        directory.link(part, "served.json")
        directory.unlink(part)
    except BaseException:
        # Empty the directory, so that the command can be repeated. Once served.json exists, delete
        # nothing: this run may have removed 'part' already, and another run created it since.
        if not directory.has("served.json"):
            with contextlib.suppress(OSError):
                directory.unlink(part)
        raise
    return served


def load_served(directory, sent, prompt_ids):
    """Return what an earlier run stored in served.json.

    The stored options serve only to detect a command line that asks for something else.
    """
    served_json = directory.path_of("served.json")
    try:
        with directory.open_stored("served.json") as f:
            # json.dump() escapes non-ASCII characters, so the file can be several times larger
            # than the reply.
            served = json.loads(f.read(4 * LARGEST_JSON))
    except FileNotFoundError:
        message = (
            "%s does not exist, so there is nothing to resume (choose a new --out-dir, without"
            " --resume)"
        )
        fail(ERROR, message, served_json)
    except (ValueError, RecursionError):
        served = None
    if not is_well_formed(served) or "sent" not in served:
        message = "%s is damaged or was not written by this script (choose a new --out-dir)"
        fail(ERROR, message, served_json)
    same_prompt = prompt_ids is None or served["prompt_token_ids"] == prompt_ids
    if served["sent"] != sent or not same_prompt:
        message = (
            "%s contains the reply to another request: the prompt, --chat, --max-tokens, --model or"
            " --policy-name differs (choose a new --out-dir)"
        )
        fail(ERROR, message, served_json)
    return served


def request_length(parameter_file):
    """Return the number of tokens in a request file: 2^lt, with lt from the parameter file.

    Exit unless check_proof.py supports the file's policy and lt is from 4 to 16, as the verifier
    requires.
    """
    with open(parameter_file, "rb") as f:
        try:
            values = check_proof.parse_policy(f.read(1 << 20))
        except check_proof.PolicyError as error:
            message = "the parameter file %s cannot be used: %s"
            fail(ERROR, message, parameter_file, one_line(error))
    fault = policy_fault(values)
    if fault:
        fail(
            ERROR,
            "the parameter file %s cannot be used with check_proof.py: it %s",
            parameter_file,
            fault,
        )
    lt = values.get("lt", "")
    if not re.fullmatch("[4-9]|1[0-6]", lt):
        message = 'the parameter file %s needs a line "lt N" with N from 4 to 16'
        fail(ERROR, message, parameter_file)
    return 1 << int(lt)


def breaks_end_token_rule(reply, max_tokens, end_tokens):
    """Return how the reply breaks the end-token rule, or None.

    A truncated reply is still a valid shorter reply, so the end is checked here. A reply must end
    with an end token, unless it is exactly --max-tokens long: the caller's limit may then have
    stopped it. No end token may come before the last token.
    """
    for position, token in enumerate(reply[:-1]):
        if token in end_tokens:
            message = (
                "position %d of the reply (counting from 0) is the end token %d, and the reply"
                " continues after it"
            )
            return message % (position, token)
    if len(reply) != max_tokens and reply[-1] not in end_tokens:
        message = "the reply has %d tokens, not --max-tokens (%d), and ends with no end token"
        return message % (len(reply), max_tokens)
    return None


def open_proof_dir(args, directory, make=False):
    """Return proof/ in --out-dir as a Directory that stays open until main() ends.

    If it does not exist, make it, or return None.
    """
    if make:
        directory.make_directory("proof")
    try:
        return args.held.enter_context(directory.open_private("proof"))
    except FileNotFoundError:
        if make:
            message = "%s disappeared before it could be opened (%s)"
            fail(ERROR, message, directory.path_of("proof"), how_to_continue(args))
        return None
    except NotPrivate as problem:
        message = "%s cannot be used: %s (choose a new --out-dir)"
        fail(ERROR, message, directory.path_of("proof"), problem)


def get_proof(args, directory, proof_id):
    """Return the Directory with the proof's files. Exit if there is no proof, or none yet."""
    proof = open_proof_dir(args, directory)
    if proof and all(proof.has(name) for name in DOWNLOADS):
        return proof
    if directory.has("no_proof.txt"):
        with directory.open_stored("no_proof.txt") as f:
            reason = f.read(2000)
    else:
        job = wait_for_proof(args, proof_id)
        if job["state"] == "waiting":
            raise TryAgain("the proof is not ready after %g seconds" % args.wait)
        if job["state"] == "proved":
            proof = proof or open_proof_dir(args, directory, make=True)
            download_proof(args, proof_id, proof)
            return proof
        reason = one_line(hide_proof_id(str(job.get("detail") or "none given")[:2000]))
        # If the reason cannot be stored, the next run asks the server again.
        with contextlib.suppress(OSError), directory.create("no_proof.txt", "w") as f:
            f.write(reason)
        delete_servers_copy(args, proof_id)
    message = "there is no proof; the server's reason: %s"
    fail(REJECTED, message, one_line(reason), outcome="no-proof")


def fetches_from_server(passed_on):
    """Return whether check_proof.py fetches from --server: no --fetch-from follows "--"."""
    return not any(arg.split("=")[0] == "--fetch-from" for arg in passed_on)


def published_url(args):
    """Return the URL from which check_proof.py fetches missing files."""
    return args.server.rstrip("/") + PREFIX + "published/" + args.policy_name


def check_command(args, passed_on, proof_dir, request_file, prompt, reply):
    """Return the command line for check_proof.py.

    Every value is the caller's, except the token counts and "--end-token none".
    """
    if fetches_from_server(passed_on):
        passed_on = [*passed_on, "--fetch-from", published_url(args)]
    if args.verifier:
        passed_on = [*passed_on, "--verifier", args.verifier]
    stopped_by_limit = len(reply) == args.max_tokens and reply[-1] not in args.end_token
    fixed = {
        "--commitment-id": args.commitment_id,
        "--public-key-dir": args.public_key_dir,
        "--parameter-file": args.parameter_file,
        "--scales-file": args.scales_file,
        "--reply": "%d:%d" % (len(prompt), len(reply)),
        "--end-token": "none" if stopped_by_limit else ",".join(map(str, args.end_token)),
    }
    fixed = [arg for pair in fixed.items() for arg in pair]
    # -I ignores PYTHON* variables, the user's site-packages and the script's directory.
    return [sys.executable, "-I", CHECK_SCRIPT, proof_dir, request_file, *passed_on, *fixed]


def result_fields(last_two):
    """Return the fields of the "result:" line that check_proof.py prints before its last line.

    Return None if that line is no "result:" line, and {} if it is malformed, does not begin
    "result: outcome=", repeats a field, or the last line is a "result:" line too.
    """
    if len(last_two) < 2 or not last_two[0].startswith(b"result: "):
        return None
    try:
        line = last_two[0].decode()
        fields = check_proof.parse_result(line)
    except ValueError:
        return {}
    well_formed = line.startswith("result: outcome=") and len(fields) == len(line.split()) - 1
    return fields if well_formed and not last_two[1].startswith(b"result:") else {}


def why_no_result(code, fields, own):
    """Return why check_proof.py's exit code and "result:" line are not a valid result, or None.

    A pass counts only if both report it and the line has this run's values ('own').
    """
    if code < 0:
        return "check_proof.py was killed by signal %d" % -code
    if code not in OUTCOMES:
        return "check_proof.py exited with the unexpected code %d" % code
    if code == ERROR and fields is None:
        return None  # a usage error
    fits = fields is not None and fields.get("outcome") in OUTCOMES[code]
    if fits and code == VERIFIED:
        fits = fields.get("claim") == CLAIMS[fields["outcome"]]
        fits = fits and all(fields.get(name) == value for name, value in own.items())
    if not fits:
        return "check_proof.py's exit code (%d) and its result line disagree" % code
    return None


def check(args, passed_on, directory, proof, request, prompt, reply):
    """Write the request file, and run check_proof.py on it and the proof.

    Return the exit code and the fields of the "result:" line.
    """
    if args.show_checked:
        message = (
            "checked: the reply's token IDs %s after a prompt of %d token IDs; SHA-256 of the"
            " request file: %s"
        )
        ids, digest = ",".join(map(str, reply)), hashlib.sha256(request).hexdigest()
        print(message % (ids, len(prompt), digest))
    sys.stdout.flush()
    environment = dict(os.environ)
    for name in ("VLLM_API_KEY", check_proof.FETCH_KEY):
        environment.pop(name, None)
    if args.api_key and fetches_from_server(passed_on):  # never sent to another source
        environment[check_proof.FETCH_KEY] = args.api_key
    with directory.create_temporary("request.bin", "w+b") as f:
        try:
            f.write(request)
            f.flush()
            f.seek(0)
            keep_record(directory, f.name, request)
        finally:
            directory.unlink(f.name)
        # check_proof.py gets the open file and the open directory, not their names: a user who can
        # rename a directory above them cannot swap in another request or proof. request.bin is only
        # a record.
        open_fds, last_two = (proof.fd, f.fileno()), collections.deque(maxlen=2)
        names = ("/dev/fd/%d" % fd for fd in open_fds)
        command = check_command(args, passed_on, *names, prompt, reply)
        options = {"env": environment, "stdout": subprocess.PIPE, "pass_fds": open_fds}
        with subprocess.Popen(command, **options) as child:
            try:
                for line in child.stdout:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    last_two.append(line)
            except KeyboardInterrupt:
                stop_check(child)
                raise
    if last_two and not last_two[-1].endswith(b"\n"):
        print(flush=True)
    return child.returncode, result_fields(last_two)


def stop_check(child):
    """Pass an interrupt on to check_proof.py, which cleans up, and wait for it.

    check_proof.py ignores the second SIGINT that it gets this way after Ctrl-C in a terminal.
    """
    child.stdout.close()  # check_proof.py must not block on a full pipe
    child.send_signal(signal.SIGINT)
    child.wait()


def finish(args, code, fields, own, proof_id):
    """Turn check_proof.py's exit code and "result:" line into this script's exit code."""
    problem = why_no_result(code, fields, own)
    if problem or code == ERROR:
        # so that the last "result:" line matches the exit code
        code = ERROR
        print("result: outcome=%s" % OWN_OUTCOMES[code], flush=True)
    if problem:
        print("error: %s, so there is no result" % problem, file=sys.stderr)
    passed = code == VERIFIED
    if passed and args.prompt_ids is None:
        code = SERVER_CHOSE_PROMPT
        print("result: outcome=%s" % OWN_OUTCOMES[code])
        message = (
            "ask_and_check.py: not a full verification (exit code %d): the server chose the"
            " prompt's token IDs and where the reply begins; use --prompt-ids"
        )
        # Print now: the server can delay the call below.
        print(message % code, flush=True)
    if passed:
        delete_servers_copy(args, proof_id)
    else:
        print("note: the server was not asked to delete its copy of the proof.", file=sys.stderr)
    return code


def token_ids(text):
    ids = None
    if re.fullmatch("[0-9]{1,10}(,[0-9]{1,10})*", text):
        ids = [int(one) for one in text.split(",")]
    if not is_token_ids(ids):
        message = "expected whole numbers below 2^32, separated by commas, without spaces"
        raise argparse.ArgumentTypeError(message)
    return ids


def token_ids_or_stdin(text):
    return text if text == STDIN else token_ids(text)


def read_prompt(args):
    """Replace a prompt that was given as "-" with standard input, minus one trailing newline."""
    if STDIN not in (args.prompt_ids, args.prompt):
        return
    cannot_read = "cannot read the prompt from standard input: %s"
    if sys.stdin is None:
        fail(ERROR, cannot_read, "it is closed")
    try:
        # Only the end of input ends the prompt: one read can return a part of it.
        fd, data = sys.stdin.fileno(), bytearray()
        while len(data) <= LARGEST_PROMPT:
            piece = os.read(fd, LARGEST_PROMPT + 1 - len(data))
            if not piece:
                break
            data += piece
    except BlockingIOError:  # more may come, or not
        fail(ERROR, cannot_read, "it is in non-blocking mode (give the prompt from a file: < FILE)")
    except OSError as error:
        fail(ERROR, cannot_read, error.strerror or "it has no file descriptor")
    if len(data) > LARGEST_PROMPT:
        fail(ERROR, "the prompt on standard input is longer than %d MiB", LARGEST_PROMPT >> 20)
    try:
        text = data.decode().removesuffix("\n")
    except UnicodeDecodeError:
        fail(ERROR, "the prompt on standard input is not UTF-8")
    if not text:
        fail(ERROR, "the prompt on standard input is empty")
    if args.prompt == STDIN:
        args.prompt = text
        return
    try:
        args.prompt_ids = token_ids(text)
    except argparse.ArgumentTypeError as error:
        fail(ERROR, "--prompt-ids on standard input: %s", error)


def commitment_id(text):
    if not re.fullmatch(check_proof.COMMITMENT_ID, text.lower()):
        raise argparse.ArgumentTypeError("expected 64 hex digits")
    return text.lower()


def at_least(minimum, kind):
    """Return an argparse type: a finite number of at least minimum. The returned function raises
    argparse.ArgumentTypeError for any other text.
    """

    def convert(text):
        with contextlib.suppress(ValueError):
            value = kind(text)
            if minimum <= value < float("inf"):
                return value
        words = "whole number" if kind is int else "number"
        raise argparse.ArgumentTypeError("expected a %s of at least %d" % (words, minimum))

    return convert


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=26)


def build_parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options] [-- options for check_proof.py]",
        description=DESCRIPTION,
        epilog="\n\n".join((PASSED_ON_HELP, ENVIRONMENT_HELP, EXIT_CODES)),
        formatter_class=HelpFormatter,
        allow_abbrev=False,
        add_help=False,
    )
    required = parser.add_argument_group("required")
    for name, metavar, kind, text in (
        ("--server", "URL", str, "the vLLM server, for example http://127.0.0.1:8000"),
        ("--model", "NAME", str, "the model's name on the server"),
        ("--policy-name", "NAME", str, "the name of one of the server's policies"),
        ("--commitment-id", "HEX", commitment_id, "64 hex digits, from a source that you trust"),
        ("--public-key-dir", "DIR", str, "the public key directory (key.bin, leaf/)"),
        ("--parameter-file", "FILE", str, "the parameter file (policy.txt)"),
        ("--scales-file", "FILE", str, "the scales file (KL.bin)"),
        ("--out-dir", "DIR", str, "a new or empty directory that only you can write to"),
        ("--end-token", "IDS", token_ids, "the token IDs that end a reply, separated by commas"),
    ):
        required.add_argument(name, metavar=metavar, type=kind, required=True, help=text)
    prompt = required.add_mutually_exclusive_group(required=True)
    text = "the prompt as token IDs, separated by commas"
    prompt.add_argument("--prompt-ids", metavar="IDS", type=token_ids_or_stdin, help=text)
    text = (
        "or the prompt as text (exit code 24 at best); for either, - reads the prompt from standard"
        " input"
    )
    prompt.add_argument("--prompt", metavar="TEXT", help=text)
    optional = parser.add_argument_group("options")
    optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
    seconds = at_least(0, float)
    for name, metavar, kind, default, text in (
        (
            "--max-tokens",
            "N",
            at_least(1, int),
            200,
            "the limit on the reply's tokens (default: 200)",
        ),
        ("--wait", "SECONDS", seconds, 3600, "how long to wait for the proof (default: 3600)"),
        ("--verifier", "PATH", str, None, "the cp_verify program (default: check_proof.py's)"),
    ):
        optional.add_argument(name, metavar=metavar, type=kind, default=default, help=text)
    for name, text in (
        ("--chat", "send --prompt as a chat message"),
        ("--ask-only", "store the reply and stop (exit code 75)"),
        ("--resume", "continue an earlier run from its --out-dir"),
        ("--show-checked", "print the checked token IDs and the request's hash"),
    ):
        optional.add_argument(name, action="store_true", help=text)
    return parser


def parse_args(argv):
    """Return this script's options, and the arguments after "--" for check_proof.py. Raise
    SystemExit on wrong usage (status 2, after argparse prints why) and after --help alone
    (status 0).
    """
    split = argv.index("--") if "--" in argv else len(argv)
    own, passed_on = argv[:split], argv[split + 1 :]
    parser = build_parser()
    # --help exits with code 0, like "verified". argparse also reads "-hx" as -h.
    if len(argv) > 1 and any(re.match("-h|--help(=|$)", arg) for arg in own):
        parser.error("--help must be the only argument")
    args = parser.parse_args(own)
    for part in (own, passed_on):  # argparse would silently use the last value
        names = [arg.split("=")[0] for arg in part if arg.startswith("--")]
        for name in names:
            if names.count(name) > 1:
                parser.error("%s given more than once" % name)
    # Only allowed options, each with its value if it takes one: none that this script sets can be
    # given again.
    rest = list(passed_on)
    while rest:
        arg = rest.pop(0)
        if arg.split("=")[0] not in PASSED_ON:
            message = (
                '%s cannot follow "--" (see --help; this script\'s own options go before "--")'
            )
            parser.error(message % arg.split("=")[0])
        if arg.split("=")[0] in WITHOUT_VALUE:
            if "=" in arg:
                parser.error("%s takes no value" % arg.split("=")[0])
        elif "=" not in arg and (not rest or rest.pop(0).startswith("--")):
            parser.error("%s needs a value" % arg)
    if not re.fullmatch(POLICY_NAME, args.policy_name):
        parser.error("--policy-name must be 1 to 64 letters, digits, - or _")
    if not re.match("https?://[^/]", args.server, re.I | re.A):
        parser.error("--server must be a URL that begins with http:// or https://")
    # Only from the environment: ps shows a command line to other users.
    args.api_key = os.environ.get("VLLM_API_KEY")
    if not re.fullmatch("[ -~]*", args.api_key or ""):  # a secret: not shown
        parser.error(
            "VLLM_API_KEY contains a character that is not printable ASCII, such as a line ending"
        )
    if args.api_key and not check_proof.HTTPS.match(args.server):
        if not check_proof.is_this_machine(args.server):
            parser.error(
                "VLLM_API_KEY is set, so a plain http --server must be 127.0.0.1 or [::1], not a"
                " name such as localhost (otherwise use https or an SSH tunnel, or unset"
                " VLLM_API_KEY)"
            )
    if args.chat and args.prompt is None:
        parser.error("--chat needs --prompt, not --prompt-ids")
    if not args.out_dir:
        parser.error("--out-dir is empty")
    # abspath() removes a trailing "/" or "/.": the last name in the path is then --out-dir itself.
    args.out_dir = os.path.abspath(args.out_dir)
    for path in (args.public_key_dir, args.parameter_file, args.scales_file):
        if not os.path.exists(path):
            parser.error("%s does not exist" % path)
    return args, passed_on


def how_to_continue(args):
    stored = args.directory is not None and args.directory.has("served.json")
    return "run the same command again" + " with --resume" * stored


def run(args, passed_on):
    if not hasattr(os, "O_PATH"):
        fail(ERROR, "this script needs Linux: it opens directories with O_PATH")
    read_prompt(args)  # before anything is created
    args.opener = new_opener(args)
    length = request_length(args.parameter_file)  # before the server is called
    directory = args.directory = args.held.enter_context(open_out_dir(args.out_dir, args.resume))
    served_json = directory.path_of("served.json")
    names = ("prompt", "prompt_ids", "chat", "max_tokens", "model", "policy_name")
    sent = {name: getattr(args, name) for name in names}
    if args.resume:
        served = load_served(directory, sent, args.prompt_ids)
        print("Resuming from %s. No new request is sent." % served_json)
    else:
        served = first_run(args, directory, sent)
    # The caller's IDs, if any: ask() and load_served() have compared them with the server's.
    prompt = served["prompt_token_ids"] if args.prompt_ids is None else args.prompt_ids
    reply = served["reply_token_ids"]
    print("reply (the server's text, unchecked): %s" % one_line(served["text"] or "", 4000))
    counts = (len(prompt), len(reply), served_json)
    print("prompt: %d tokens; reply: %d tokens; stored in %s" % counts)
    problem = breaks_end_token_rule(reply, args.max_tokens, args.end_token)
    if problem:
        fail(REJECTED, "%s; nothing was checked", problem)
    if args.ask_only:
        print("result: outcome=%s" % STORED)
        print(
            "ask_and_check.py: reply stored; not checked yet (run the same command again with"
            " --resume, without --ask-only)"
        )
        return TRY_AGAIN
    proof = get_proof(args, directory, served["proof_id"])
    try:
        request = spool.whole_reply(prompt, reply, length, fill=PADDING)
    except ValueError as error:
        fail(REJECTED, "%s; nothing was checked", error)
    if args.prompt_ids is None:
        # before check_proof.py can print "verified"
        print(SERVER_CHOSE_PROMPT_WARNING)
    code, fields = check(args, passed_on, directory, proof, request, prompt, reply)
    start_end = struct.unpack_from("<2I", request, 4 * length)
    own = {"commitment_id": args.commitment_id, "range": "%d:%d" % start_end}
    return finish(args, code, fields, own, served["proof_id"])


def fill_standard_descriptors():
    """Open /dev/null on each of the descriptors 0 to 2 that the caller has closed.

    Otherwise a directory that the script holds open could get such a number, and check_proof.py
    would inherit it as its standard input or standard error. Where descriptor 2 was closed, Python
    has set sys.stderr to None, and print() would write the messages to standard output: so
    sys.stderr is opened on /dev/null too.
    """
    try:
        fd = os.open(os.devnull, os.O_RDWR)
        while fd <= 2:
            os.set_inheritable(fd, True)  # like a shell's "< /dev/null": check_proof.py gets it too
            fd = os.open(os.devnull, os.O_RDWR)
    except OSError as error:
        fail(ERROR, "cannot open %s: %s", os.devnull, error.strerror)
    os.close(fd)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def main(argv=None):
    fill_standard_descriptors()
    if sys.stdout is None:
        message = (
            "standard output is closed, but the result is printed there (run the command with"
            " standard output open)"
        )
        fail(ERROR, message)
    for stream in (sys.stdout, sys.stderr):
        # A character that the terminal cannot encode must not stop the script.
        stream.reconfigure(errors="backslashreplace")
    # If the caller ignores SIGCHLD, subprocess reports exit code 0 for every child.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    args, passed_on = parse_args(sys.argv[1:] if argv is None else list(argv))
    args.directory = None  # --out-dir, once it is open
    with contextlib.ExitStack() as args.held:  # closes the directories that the run holds open
        try:
            return run(args, passed_on)
        except TryAgain as error:
            fail(TRY_AGAIN, "%s (%s)", error, how_to_continue(args))
        except KeyboardInterrupt:
            if sys.stderr.isatty():
                print(file=sys.stderr)  # after the terminal's "^C"
            fail(ERROR, "interrupted (%s)", how_to_continue(args))
        except OSError as error:  # a local file or program: the server's failures became TryAgain
            # os.link() puts the new name in filename2.
            name = error.filename2 or error.filename
            fail(ERROR, "%s%s", "%s: " % name if name else "", error.strerror)
        except Exception as error:  # no traceback: it could print the server's strings unescaped
            message = "the server sent something unusable, or this script has a bug: %s: %s"
            fail(ERROR, message, type(error).__name__, one_line(error))


if __name__ == "__main__":
    sys.exit(main())
