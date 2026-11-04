#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""ASGI middleware that gets proofs for the replies of an unmodified vLLM server.

    vllm serve MODEL --disable-uvicorn-access-log --middleware proof_middleware.ProofMiddleware

Install it with "pip install ./integrations", and set PROOF_MIDDLEWARE_CONFIG to the config file's
path. For a POST to /v1/completions or /v1/chat/completions with the header "x-proof-policy: NAME",
it copies the token IDs from vLLM's reply, writes a request file into the proving service's spool
directory, and adds the header "x-proof-id". The client fetches the proof from
/plugins/proof/jobs/PROOF_ID, which is served only after cp_verify has passed every proof of its
batch that has a job. Every other call passes to vLLM unchanged. All state is in files, but for
the counts of a check's tries and, where it can't be written, a rejection's note (remembered).

Threat model: a client must not fetch another client's proof or choose the tokens that are proved.
Other local users are not trusted (spool.py); root and the server's user are.
When a process loads the config file, it opens each directory that the file names, by its path, and
keeps it open; later it works below these open directories (Kept), and a check's child too, by
kept_path(). That is meant to keep whoever can
change a directory at or above one of the names, by a link or by renaming, from leading a process
that has loaded its config file to other files. Such a change can still bring a hold, or end one
that lies only under the name (is_on_hold()). For what this leaves open, above all before a server
starts under sudo, see integrations/LIMITS.md#above-a-public-directory.

Run as a program, it has the commands check, init, clean, hold and lift-hold (--help). See
integrations/vllm/REFERENCE.md."""

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import errno
import fcntl
import functools
import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

if __name__ == "__main__":
    # Write no __pycache__ into a checkout. Not on import: the setting is process-wide.
    sys.dont_write_bytecode = True

import check_proof
import spool
from spool import others_can_change, refusal

# a prefix of the form that vLLM's documentation on endpoint plug-ins recommends
PREFIX = "/plugins/proof/"
POLICY_HEADER, ID_HEADER = "x-proof-policy", "x-proof-id"
CONFIG_VARIABLE = "PROOF_MIDDLEWARE_CONFIG"
API_KEY_VARIABLE = "VLLM_API_KEY"  # defined by vLLM
# The package and the name of the class with which vLLM 0.29.0 guards its own routes.
VLLM_GUARD = ("vllm", "AuthenticationMiddleware")
SERVING_ROUTES = ("/v1/completions", "/v1/chat/completions")
# The only body fields of a request whose reply is proved. An allow-list, with no option to extend
# it: many other fields let a client force the reply's tokens (structured outputs,
# allowed_token_ids, logit_bias, bad_words, min_tokens, and whatever later versions of vLLM add),
# and whether chosen tokens get a proof reveals something about hidden weights.
ALLOWED_FIELDS = (
    "model",
    "prompt",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "stream",
    "return_token_ids",
)
# Pads a request file after the reply. No proof covers the padding.
PADDING = 0
DOWNLOADS = {suffix + ".bin": suffix for suffix in spool.RESPONSE_FILES}
PROOF_SUFFIX = spool.RESPONSE_FILES[-1]

POLICY_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
JOB_ROUTE = re.compile(r"jobs/([0-9a-f]{64})(?:/([a-z]+\.bin))?")
# A leaf file's number is a 32-bit word: at most ten digits, no sign, no leading zero.
PUBLISHED_ROUTE = re.compile(
    r"published/(%s)/(?:(key\.bin|layout\.txt)|leaf/((?:0|[1-9][0-9]{0,9})\.bin))"
    % POLICY_NAME.pattern
)
# mark.txt in the jobs directory. Its kind differs from a spool directory's (spool.py), so neither
# directory passes as the other.
JOBS_MARK = b"cp-folder 1\nkind jobs\n"
# The only names that clean considers.
JOB_FILE = re.compile(r"[0-9a-f]{64}\.(?:json|checked)")
WITHHELD_FILE = re.compile(r"withheld\.[A-Za-z0-9_-]{1,64}\.[0-9]+(?:\.done)?\.json")
WITHHELD_DONE = re.compile(r"withheld\.[A-Za-z0-9_-]{1,64}\.[0-9]+\.done\.json")
REQUEST_FILE = re.compile(r"[0-9a-f]{64}\.req")
UNFINISHED_REQUEST = re.compile(r"[0-9a-f]{64}")  # in in/.tmp/
RESPONSE_FILE = re.compile(
    r"([0-9a-f]{64})\.(?:%s)" % "|".join((*spool.RESPONSE_FILES, spool.FAILED, spool.TAG))
)

DEFAULT_MAX_WAITING = 64  # requests waiting in in/ before replies get no proof
DEFAULT_MAX_STORED = 1000  # responses stored in out/, likewise
MAX_REQUEST_BYTES = 1 << 20  # a longer proof request is rejected
MAX_REPLY_BYTES = 16 << 20  # a longer reply gets no proof
MAX_JOB_BYTES = 1 << 20  # a job holds its request: up to 2^16 tokens of 4 bytes, in hex
MAX_PARAMETER_FILE_BYTES = 1 << 20
MAX_PROOF_BYTES = check_proof.MAX_FETCHED_BYTES  # a longer proof is not read: no client takes it
CHUNK = 1 << 16

# The prover gives no reason when it issues no proof, because a reason can depend on the hidden
# weights.
DECLINED = (
    "The prover issued no proof for this reply. Either the reply's tokens do not meet the policy"
    " under the committed model, or the prover failed. The prover does not say which."
)
EXPIRED = (
    "The server had the response to this job and does not have it now. If the state stays so, the"
    " response was deleted: ask again with a new request."
)
SERVER_FAULT = "The server has a fault that its provider must look into. Ask again later."
JOB_FAULT = (
    "The server has a fault that its provider must look into. Keep the proof"
    " ID and ask again later."
)
# ---- The check of a batch ----
#
# cp_prove releases what it has staged, and nothing in it tests a finished proof. A proof that
# cp_verify rejects can hold values that are not masked, and a machine that made one may have
# spoiled its neighbours too. So the middleware says nothing of a response, a failed one included,
# until every proof of its batch has passed check_proof.py, the check that a client runs. A batch is
# the responses with one modification time. cp_prove names all of a batch's files before it removes
# the first of its requests from work/, so a job that has left work/ has its whole batch in out/.
# Two ends only. A verdict on a whole response (cp_verify's NO, a y that no check reads):
# every proof of the batch is overwritten and replaced by a failed response, and the policy goes on
# hold. All else (no verdict, a time limit, a fault of the set-up, a member that looks malformed,
# which can be a file on its way in or out): if the same cause stands at the next try, or after
# MOST_TRIES of any causes, the policy goes on hold, and nothing is wiped: behind lift-hold the
# batch is checked anew.
WAIT_S = 0  # how long a call waits for a check before it answers that the job is waiting
CHECK_TIMEOUT_S = 1800  # for one proof; a check that takes longer gives no answer
CHECK_WORK = ".check-"  # the start of a check's temporary folder's name
MEMORY_DIR = "/dev/shm"  # where that folder lies if it can: it holds a copy of the proof
NOT_FINAL_TRIES = 10  # from this try on, a batch that waits for the service counts as no answer
MOST_TRIES = 10  # a batch without an answer this many tries goes on hold, whatever the causes
RETRY_S = 60  # a batch without an answer is checked again no sooner than this
STOP_S = 5  # from SIGTERM to SIGKILL for a check that has run out of time
CHECK_THREADS = 1  # for each call of cp_verify
# What run_check() returns. None: no answer: nothing is said of the response.
PASSED, REJECTED = "passed", "rejected"
SKIPPED = "skipped"  # a check not started, because another has said NO
# check_proof.py's last line where cp_verify itself has rejected a proof.
CP_VERIFY_NO = "check_proof: rejected: cp_verify printed NO: "
# Other last lines, and the constant name that the log gives each: nothing of check_proof.py's or
# cp_verify's words goes to the log, which so holds no byte of a response. No answer of this kind
# says anything of a response: the policy goes on hold, and an operator looks.
OTHER_ENDS = (
    ("check_proof: rejected: cp_verify printed NO: ", "the verifier rejected the leaf file"),
    ("check_proof: error: cp_verify was killed by signal", "the verifier was killed by a signal"),
    ("check_proof: error: cp_verify exited with status 5 ", "the verifier ended NOT CHECKED"),
    ("check_proof: not a verification: ", "not a verification"),
    ("check_proof: rejected: the public key directory", "another commitment ID"),
    ("check_proof: rejected: the proof file does not start", "not a proof of this format"),
    ("check_proof: error: the leaf file", "a leaf file is missing"),
    ("check_proof: error: the verifier ", "the verifier is missing"),
    ("check_proof: rejected: cp_verify rejected key.bin", "key.bin or a configuration file"),
    ("check_proof: error: cp_verify exited", "another exit of the verifier"),
)

# While a policy's spool directory has an entry "hold" (spool.is_on_hold()), the middleware submits
# nothing, and serves no stored response. cp_prove neither writes nor reads it: whatever starts
# cp_prove creates it after a bad exit ("check --help"), and the check of a batch creates it at a
# rejection, before it clears that batch, and where a batch stays without a verdict for the same
# cause at two tries in a row, or for MOST_TRIES tries of any causes (no_answer()).
ON_HOLD = "The provider has put this policy's proofs on hold. Ask again later."
NO_PROOF_ON_HOLD = "The provider has put this policy's proofs on hold, so this reply gets no proof."
BEFORE_REMOVING_A_HOLD = (
    "The lift-hold command removes it, which puts every stored response back on offer, faulty ones"
    " too, each behind the check of its batch, which a file that has lost a mask can pass: read"
    " integrations/vllm/REFERENCE.md first"
)
WHEN_LIFTING_IS_SAFE = (
    "safe only if the cause of the hold is known, is gone, and was harmless (a mistyped path, a"
    " full disk). Otherwise %s and make a new commitment (integrations/vllm/REFERENCE.md)."
)
# In place of a result: the server cannot use its files, or a response is malformed or fails its tag
# check.
FAULTY = object()
WAITING = object()  # in place of a fault: the response's batch has not passed its check yet

log = logging.getLogger("proof_middleware")


class MaskProofIds(logging.Filter):
    """Replaces every run of 32 or more hex digits in a log record's strings with "(hex digits)".

    Anyone who has a proof ID, 64 hex digits, can fetch the proof, and uvicorn's access log prints
    each call's path, also one with a mistyped or cut-off ID."""

    def filter(self, record):
        def mask(value):
            return LONG_HEX.sub("(hex digits)", value) if isinstance(value, str) else value

        record.msg = mask(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(map(mask, record.args))
        return True


# A digit as it is or percent-encoded: uvicorn prints a query string as it was sent.
LONG_HEX = re.compile("(?:%3[0-9]|%[46][1-6]|[0-9A-Fa-f]){32,}")
MASK_PROOF_IDS = MaskProofIds()


class ConfigError(ValueError):
    """The config file, or something that it lists, cannot be used. The message is for the
    provider."""


class ClientError(Exception):
    """A call that gets an HTTP error response. The message is for the client."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class NotProvable(ValueError):
    """A served reply that is not submitted to the prover. The message is for the client."""


class ResponseCut(Exception):
    """Breaks the connection when a response that has started cannot or must not be completed."""


class OnHold(NotProvable):
    """A policy is on hold. submit() raises it with a message for the client, clean() with one for
    the provider."""


# ---- The config file ----


class Kept:
    """The directories that a config file names, each kept open: {path: descriptor}.

    load_config() opens each by its path, checks it on that descriptor, and keeps it. The looks
    around that one open go by the path too: above the directory and, inside it, the walk that
    refuses a secret key directory (reject_secret_key_dir()). Every later use of its files opens "."
    below the descriptor. That is meant to keep whoever can change a directory at or above one of
    the names, by a link or by renaming, from leading a server elsewhere between two calls, or a
    command elsewhere between its checks and its act. Two later opens go by the path: a check's
    child reads the public files by their names, and there is the look for a hold under the name.
    Through that look, a change under the name can bring a hold, or end one
    that lies only there (is_on_hold()). describe() and lift_hold() also ask by the path whether a
    directory exists: the answer chooses only printed words and an exit code. The descriptors are
    closed when the last Config or Policy that refers to this object is gone."""

    def __init__(self):
        self.fds, self.counted = {}, set()

    def add(self, path, opened):
        """Keep a copy of the descriptor that one of spool.py's context managers, which opens path,
        yields. Not the context manager, left open: what is still open when the interpreter ends is
        closed too late to find the os module, with a traceback on standard error."""
        if path in self.fds:
            raise ValueError("%s is named twice" % path)  # each is opened once, by one set of rules
        with opened as fd:
            self.fds[path] = os.dup(fd)  # not passed on across exec
        # A file system might report a link count of 0 for every directory. Then the count cannot
        # tell that the directory was removed (__getitem__()). No such file system has been tried.
        if os.fstat(self.fds[path]).st_nlink:
            self.counted.add(path)
        return self.fds[path]

    def __contains__(self, path):
        return path in self.fds

    def __getitem__(self, path):
        """Return the descriptor. A directory that did not exist when load_config() looked is
        missing for as long as this object lives, and so is one that was removed since: nothing can
        be made in it, and no other process finds it."""
        if path not in self.fds or (path in self.counted and not os.fstat(self.fds[path]).st_nlink):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return self.fds[path]

    def close(self, close=os.close):
        # Emptied first: a closed descriptor's number can soon be another file's.
        fds, self.fds = self.fds, {}
        for fd in fds.values():
            close(fd)

    __del__ = close

    def __deepcopy__(self, memo=None):
        """A copy is this object: a second one with the same numbers would close them at its end,
        under a Config that still uses them. dataclasses.asdict() copies so."""
        return self

    __copy__ = __deepcopy__

    def __reduce__(self):
        raise TypeError("open directories cannot be pickled")


@dataclasses.dataclass(frozen=True)
class Policy:
    """One acceptance policy and the spool directory of its proving service."""

    name: str
    served_model: str
    commitment_id: str
    spool_dir: str
    max_waiting: int
    max_stored: int
    public_key_dir: str  # key.bin and leaf/: served to clients, and read by the check
    kind: str  # "exact" or "clipped-gap"
    limits: dict  # the parameter file's C, theta, cap and s
    # T: the tokens in a request file, so the limit for a prompt plus its reply
    max_total_tokens: int
    vocabulary: int  # V, the number of token IDs
    closed: bool = False  # no new requests; jobs and published files are still served
    layout_file: str | None = None  # layout.txt next to the parameter file, if any
    kept: Kept = dataclasses.field(default=None, compare=False, repr=False)  # its Config's
    parameter_file: str = ""
    verifier: str = ""  # cp_verify; that the service's account cannot change it is the provider's
    checks_at_once: int = 1  # calls of cp_verify that run side by side, one thread each


@dataclasses.dataclass(frozen=True)
class Config:
    jobs_dir: str
    policies: dict  # {name: Policy}
    kept: Kept = dataclasses.field(default=None, compare=False, repr=False)


def load_config(path, create_dirs=True, denied=""):
    """Read and check the config file and everything that it lists. Return a Config, or raise
    ConfigError. denied: a hint to append if permission is denied.

    With create_dirs, and only if every check here passes, create what is missing: the jobs
    directory,
    the spool directories (not their parents), and the mark file of each (only in an empty
    directory)."""
    if not path:
        raise ConfigError(
            "no config file (set the environment variable %s to its path)" % CONFIG_VARIABLE
        )
    kept = Kept()
    try:
        with open(spool.open_public_file(os.path.abspath(path)), "rb") as f:
            check_above_config(os.path.abspath(path))  # before a byte is read
            data = json.load(f)
        config = check_dirs(*parse_config(data), kept, create_dirs)
    except (spool.UnsafePath, OSError, ValueError, RecursionError) as error:
        kept.close()
        if isinstance(error, OSError) and error.filename == os.path.abspath(path):
            error = error.strerror + (denied if isinstance(error, PermissionError) else "")
        elif isinstance(error, OSError) and error.filename:
            hint = denied if isinstance(error, PermissionError) else ""
            error = "%s: %s%s" % (error.filename, error.strerror, hint)
        elif str(error).endswith("the current user") or (
            isinstance(error, spool.OpenAbove) and "(owned by another account)" in str(error)
        ):  # what the wrong account gets where nothing is denied to it
            error = "%s%s" % (error, denied)
        if str(error).startswith(os.path.abspath(path) + " "):  # it names the config file itself
            raise ConfigError(str(error)) from None
        raise ConfigError("%s: %s" % (path, error)) from None
    return config


def check_above_config(path):
    """Run by root: raise spool.UnsafePath if another account may be able to change the directory
    that holds the config file, or one above: it could put another file under its name, and the file
    says where root works. For another runner nothing looks above that directory."""
    if os.geteuid():
        return
    found = others_can_change(os.path.dirname(path))
    if found:  # not spool.OpenAbove, which would get load_config()'s hint: it does not fit root
        raise spool.UnsafePath(
            "it lies in or below %s, which another account may be able to change (%s): that account"
            " could put another file in its place, and so choose in which directories root works."
            " Lay the config file in a directory of root's, below directories of root's, such as"
            " the one in step 3 of integrations/vllm/REFERENCE.md. Whatever the config file names"
            " in or below the directory that another account may change is refused next, for the"
            " same reason. For each such jobs or spool directory, name a new one below"
            " directories of root's (init makes it; start the proving service on a new spool"
            " directory). Copy every other such file or directory there as root, from a source that"
            " you trust (do not move it: a moved file keeps its owner), and name the copy in the"
            " config file" % (spool.printable(found[0]), spool.whose(*found))
        )


def require_keys(entry, required, optional, where):
    """Raise ConfigError unless a JSON value is an object with all required keys and no unknown
    ones."""
    if not isinstance(entry, dict):
        raise ConfigError("%s must be a JSON object" % where)
    for key in required:
        if key not in entry:
            raise ConfigError('%s needs the key "%s"' % (where, key))
    for key in entry:
        if key not in (*required, *optional):
            raise ConfigError(
                "%s has the unknown key %r (known: %s)"
                % (where, key, ", ".join((*required, *optional)))
            )


def parse_config(data):
    """Check the config file's JSON value. It reads no file. Return the jobs directory's
    path and {name: a policy's entry}, with every path normalized. Raise ConfigError if a key is
    missing or unknown, a policy's name or a value has the wrong type or form, or a path is not
    absolute or has "." or ".." as one of its parts."""
    require_keys(data, ("jobs_dir", "policies"), (), "the config file")
    if not isinstance(data["jobs_dir"], str):
        raise ConfigError('"jobs_dir" must be a path')
    if not isinstance(data["policies"], dict) or not data["policies"]:
        raise ConfigError('"policies" must be an object with at least one policy')
    entries = {name: parse_entry(name, entry) for name, entry in data["policies"].items()}
    return normal_path(data["jobs_dir"]), entries


def parse_entry(name, entry):
    if not POLICY_NAME.fullmatch(name):
        raise ConfigError("the policy name %r must be 1 to 64 letters, digits, - or _" % name)
    where = "policy %s" % name
    strings = (
        "served_model",
        "commitment_id",
        "parameter_file",
        "spool_dir",
        "public_key_dir",
        "verifier",
    )
    optional = ("max_waiting", "max_stored", "closed", "checks_at_once")
    require_keys(entry, strings, optional, where)
    for key in strings:
        if not isinstance(entry[key], str):
            raise ConfigError('%s: "%s" must be a string' % (where, key))
    for key in ("max_waiting", "max_stored"):
        if type(entry.get(key, 1)) is not int or entry.get(key, 1) < 1:
            raise ConfigError('%s: "%s" must be a whole number, 1 or more' % (where, key))
    if type(entry.get("closed", False)) is not bool:
        raise ConfigError('%s: "closed" must be true or false' % where)
    if not re.fullmatch(check_proof.COMMITMENT_ID, entry["commitment_id"]):
        raise ConfigError('%s: "commitment_id" must be 64 lower-case hex digits' % where)
    return {
        **entry,
        "parameter_file": normal_path(entry["parameter_file"]),
        "spool_dir": normal_path(entry["spool_dir"]),
        "public_key_dir": normal_path(entry["public_key_dir"]),
        "verifier": normal_path(entry["verifier"]),
        "checks_at_once": checks_at_once(entry, where),
    }


def checks_at_once(entry, where):
    """Return how many checks of a batch run side by side: as configured, else a quarter of the CPUs
    that this process may use, at least 1 and at most 6, as driver/measure_rate.py does."""
    value = entry.get("checks_at_once", max(1, min(6, len(os.sched_getaffinity(0)) // 4)))
    if type(value) is not int or not 1 <= value <= 64:
        raise ConfigError('%s: "checks_at_once" must be a whole number from 1 to 64' % where)
    return value


def read_policy(name, entry, kept):
    """Read a policy's parameter file, below the kept descriptor of its directory. Return the
    Policy. Raise ConfigError if the file is longer than MAX_PARAMETER_FILE_BYTES, is not in the
    strict form of check_proof.parse_policy(), has no line lt from 4 to 16 or no line V of 1 or
    more, or sets no supported policy; OSError or spool.UnsafePath if it cannot be opened."""
    path = entry["parameter_file"]
    top = kept[os.path.dirname(path)]
    try:
        with open(spool.open_public_file(os.path.basename(path), top), "rb") as f:
            data = f.read(MAX_PARAMETER_FILE_BYTES + 1)
        if len(data) > MAX_PARAMETER_FILE_BYTES:
            raise ConfigError("%s is longer than %d bytes" % (path, MAX_PARAMETER_FILE_BYTES))
        values = check_proof.parse_policy(data)
    except check_proof.PolicyError as error:
        raise ConfigError("%s: %s" % (path, error)) from None
    # The verifier accepts lt 4 to 16; a request file has 2^lt tokens.
    if not re.fullmatch("[4-9]|1[0-6]", values.get("lt", "")) or int(values.get("V", 0)) < 1:
        raise ConfigError(
            "%s needs the lines lt, a whole number from 4 to 16, and V, 1 or more" % path
        )
    layout_file = None
    with contextlib.suppress(FileNotFoundError):
        os.stat("layout.txt", dir_fd=top, follow_symlinks=False)
        layout_file = os.path.join(os.path.dirname(path), "layout.txt")
        os.close(spool.open_public_file("layout.txt", top))
    fault = check_proof.policy_fault(values)
    if fault:
        raise ConfigError(
            "%s %s, so it sets no supported policy (exact, clipped-gap)" % (path, fault)
        )
    return Policy(
        name,
        entry["served_model"],
        entry["commitment_id"],
        entry["spool_dir"],
        entry.get("max_waiting", DEFAULT_MAX_WAITING),
        entry.get("max_stored", DEFAULT_MAX_STORED),
        entry["public_key_dir"],
        check_proof.policy_kind(values),
        {key: int(values[key]) for key in check_proof.POLICY_LIMITS},
        1 << int(values["lt"]),
        int(values["V"]),
        entry.get("closed", False),
        layout_file,
        kept,
        path,
        entry["verifier"],
        entry["checks_at_once"],
    )


def normal_path(path):
    """Return a configured path, normalized. Raise ConfigError unless it is absolute and has no "."
    or "..": after a symlink, the kernel resolves ".." differently from os.path.normpath()."""
    if not os.path.isabs(path) or {".", ".."} & set(path.split("/")):
        raise ConfigError('%s is not an absolute path without "." and ".."' % (path or '""'))
    return os.path.normpath(path)


def check_dirs(jobs_dir, entries, kept, create):
    """Open the directories that a config file names, each once (Kept), check them, and read the
    parameter files. Return the Config. With create, then create what load_config() describes.
    jobs_dir, entries: from parse_config(). Raise ConfigError if two of the jobs, spool and public
    key directories are the same or one is inside another, if a jobs or spool directory is missing
    and has no parent directory, or if read_policy() or reject_secret_key_dir() finds a fault;
    spool.UnsafePath and OSError from the checks of the directories pass through."""
    private = [jobs_dir, *(entry["spool_dir"] for entry in entries.values())]
    for path in private:  # first: nothing is opened or made below a place that others can change
        check_above("jobs directory" if path == jobs_dir else "spool directory", path)
    # Several policies of one commitment may share one.
    public = list(dict.fromkeys(entry["public_key_dir"] for entry in entries.values()))
    # The parameter files' directories. Several policies may share one.
    holders = list(
        dict.fromkeys(os.path.dirname(entry["parameter_file"]) for entry in entries.values())
    )
    for path in private + public + holders:
        reject_secret_key_dir(path)
    # Real paths, because a parent directory may be a symlink.
    real = [os.path.realpath(path).rstrip("/").split("/") for path in private + public]
    for (one, outer), (other, inner) in itertools.permutations(zip(private + public, real), 2):
        if inner[: len(outer)] == outer:
            raise ConfigError(
                "%s is the same as, or inside, %s (the jobs directory, each spool directory and"
                " each public key directory must be separate)" % (other, one)
            )
    for path in private:
        if not os.path.lexists(path) and not os.path.isdir(os.path.dirname(path)):
            parent, fault = os.path.dirname(path), "is not a directory"
            try:
                os.lstat(parent)  # "permission denied" is reported as that
            except FileNotFoundError:
                fault = "does not exist: create it, or correct the path"
            raise ConfigError("%s cannot be made: %s %s" % (path, parent, fault))
    for path in private:
        with contextlib.suppress(FileNotFoundError):
            kept.add(path, spool.open_private_dir(path))
    for path in dict.fromkeys(public + holders):
        # A parameter file may lie in the jobs or a spool directory. That one is opened above, as a
        # private directory, and not again: a second open by the same path could meet another.
        if path not in private:
            kept.add(path, spool.open_public_dir(path))
        check_above_public(path)
    policies = {name: read_policy(name, entry, kept) for name, entry in entries.items()}
    config = Config(jobs_dir, policies, kept)
    for path in public:
        os.close(spool.open_public_file("key.bin", kept[path]))
        with contextlib.suppress(FileNotFoundError), spool.open_public_dir("leaf", kept[path]):
            pass
    with contextlib.suppress(FileNotFoundError), open_jobs(config, allow_empty=True):
        pass
    for policy in policies.values():
        # Check the existing parts of a spool directory now, not at the first call.
        with contextlib.suppress(FileNotFoundError), open_spool(policy) as top:
            with contextlib.suppress(FileNotFoundError), spool.open_private_dir("in", top) as queue:
                with spool.open_private_dir(".tmp", queue):
                    pass
            with contextlib.suppress(FileNotFoundError), spool.open_private_dir("out", top):
                pass
    if not create:
        return config
    for path in (*sorted(private[1:]), jobs_dir):
        if path not in kept:
            kept.add(path, spool.open_private_dir(path, create=True))
    for policy in policies.values():
        with open_spool(policy, mark=True):
            pass
    with open_jobs(config, mark=True):
        pass
    return config


def check_above_public(path):
    """Run by root: raise spool.UnsafePath if another account may be able to change a directory
    above a public key directory or a parameter file's directory: it could put another directory in
    its place. Made behind the open of that directory. For another runner nothing looks above these
    directories: the look also refuses places with which nothing need be wrong
    (integrations/LIMITS.md)."""
    if os.geteuid():
        return
    found = others_can_change(os.path.dirname(path))
    if found:  # not spool.OpenAbove: load_config()'s hint, to run this as ACCOUNT, leads past it
        raise spool.UnsafePath(
            "%s lies below %s, which another account may be able to change (%s): it could put"
            " another directory in its place, and choose what is served as your published files and"
            " which limits this server applies. Publish copies, made as root from a source that you"
            " trust, in a new directory below folders that belong to root and that only root can"
            " write, and name it (for a parameter file: the copy) in the config file"
            % (path, spool.printable(found[0]), spool.whose(*found))
        )


def check_above(what, path):
    """Raise spool.OpenAbove if another account may be able to change a directory above path: that
    account could then put another directory in its place. The look starts at the deepest parent
    that exists. It is called for the jobs and spool directories: spool.py looks again wherever it
    opens one of these by its path, and checks the opened directory itself; an open below a kept
    descriptor (Kept) looks at nothing above."""
    existing = os.path.dirname(path)
    while not os.path.lexists(existing):
        existing = os.path.dirname(existing)
    found = others_can_change(existing)
    if found:
        raise spool.OpenAbove(refusal(what, path, *found))


def reject_secret_key_dir(path):
    """Raise ConfigError if a configured directory is, or is inside, a secret key directory.

    A mistyped path can point to one, and its key.bin would let it pass as a public key directory.
    Parents are checked as written and with symlinks resolved; spool.py re-checks only the directory
    itself, at each open."""
    for above in (path, os.path.realpath(path)):
        while True:
            for name in spool.SECRET_KEY_FILES:
                if os.path.lexists(os.path.join(above, name)):
                    raise ConfigError(
                        "%s is, or is inside, a secret key directory (%s contains %s): name a"
                        " directory outside it (for a parameter file: a copy of the file there)"
                        % (path, above, name)
                    )
            if above == os.path.dirname(above):
                break
            above = os.path.dirname(above)


# ---- What a proof request may contain ----


class Pairs(list):
    """A JSON object as (name, value) pairs, so that a repeated name can be detected."""


def check_request(body, served_model):
    """Raise NotProvable unless a request body leaves the choice of every reply token to the served
    model."""
    try:
        pairs = json.loads(body, object_pairs_hook=Pairs)
    except (ValueError, RecursionError):
        pairs = None
    if not isinstance(pairs, Pairs):
        raise NotProvable("The request body is not one JSON object, so the reply gets no proof.")
    names, fields = [name for name, _ in pairs], dict(pairs)
    for name in names:
        if name not in ALLOWED_FIELDS:
            raise NotProvable(
                "The request has the field %.40r, so the reply gets no proof. A request that asks"
                " for a proof under this policy may have only these fields: %s."
                % (name, ", ".join(ALLOWED_FIELDS))
            )
        if names.count(name) > 1:
            raise NotProvable(
                "The request has the field %r twice, so the reply gets no proof." % name
            )
    if type(fields.get("temperature")) not in (int, float) or fields["temperature"] != 0:
        raise NotProvable(
            'Only a reply that was decoded greedily gets a proof. Set "temperature": 0 in the'
            " request body."
        )
    if fields.get("return_token_ids") is not True:
        raise NotProvable(
            'A proof needs the token IDs. Set "return_token_ids": true in the request body.'
        )
    if fields.get("model", served_model) != served_model:
        raise NotProvable(
            'This policy covers only the served model %s. Set "model" to that name, or leave it'
            " out." % served_model
        )
    prompt, messages = fields.get("prompt", ""), fields.get("messages", [])
    if not isinstance(prompt, str) and not (
        type(prompt) is list and all(type(token) is int for token in prompt)
    ):
        raise NotProvable(
            'One proof covers one reply. Send "prompt" as one string or as one list of token IDs.'
        )
    plain = type(messages) is list and all(
        isinstance(one, Pairs)
        and sorted(name for name, _ in one) == ["content", "role"]
        and all(isinstance(part, str) for _, part in one)
        for one in messages
    )
    if not plain:
        raise NotProvable(
            'Each message of a request that asks for a proof may have only "role" and "content",'
            " both strings. Other parts, such as images, influence the reply in a way that no proof"
            " covers."
        )


def served_tokens(body):
    """Return the prompt's and the reply's token IDs from a body that vLLM sent with status 200.

    The body is one JSON object, or server-sent events with one JSON object per "data:" line. vLLM
    puts the prompt's IDs in the choice (completions) or at the top level (chat), in a stream only
    in the first chunk; each chunk has the reply's new IDs. Raise NotProvable if vLLM reported an
    error, sent more than one reply, sent bytes that are not UTF-8 or JSON that cannot be read, or
    sent token IDs but no finish reason."""
    prompt, reply, finished = [], [], False
    try:
        text = body.decode()
        if text.lstrip().startswith("{"):
            objects = [json.loads(text)]
        else:
            # Split at line feeds only: str.splitlines() also splits at characters that JSON allows
            # unescaped in a string.
            events = (line[5:].strip() for line in text.split("\n") if line.startswith("data:"))
            objects = [json.loads(event) for event in events if event != "[DONE]"]
        for one in objects:
            if "error" in one:
                raise NotProvable(
                    "vLLM reported an error while it streamed this reply, so the reply is"
                    " incomplete."
                )
            prompt = one.get("prompt_token_ids") or prompt
            for choice in one.get("choices", ()):
                if choice["index"] != 0:
                    raise NotProvable(
                        "One proof covers one reply, and this call returned several. Send one"
                        " prompt."
                    )
                prompt = choice.get("prompt_token_ids") or prompt
                reply += choice.get("token_ids") or []
                finished = finished or choice.get("finish_reason") is not None
    except NotProvable:
        raise
    except (ValueError, KeyError, TypeError, AttributeError):
        raise NotProvable(
            "The proof middleware could not read vLLM's reply, so it gets no proof. Tell the"
            " provider: a layer between vLLM and the middleware that compresses replies can cause"
            " this."
        ) from None
    if (prompt or reply) and not finished:
        raise NotProvable("The reply ended before vLLM gave a finish reason, so it is incomplete.")
    return prompt, reply


# ---- Jobs and the spool directory ----


@contextlib.contextmanager
def open_spool(policy, mark=False):
    """Open a policy's spool directory, check its mark file against the commitment ID, and yield the
    descriptor. mark: write the mark file into an empty directory.

    What cp_prove does with that file: integrations/LIMITS.md."""
    with spool.open_private_dir(".", policy.kept[policy.spool_dir]) as top:
        if mark or not spool.is_unused(top):
            spool.check_mark(top, policy.commitment_id, create=mark)
        yield top


def submit(policy, prompt, reply, held=None):
    """Submit one served reply, as two lists of token IDs, to a policy's proving service.

    Return the request ID, which names the response's files in out/, and the key of the response's
    tag. Raise NotProvable, or spool.UnsafePath or OSError if the directory cannot be used.

    The proof covers every token of the reply and none of the prompt. No end token is appended,
    because vLLM includes the token that ended a reply in the reply's IDs. A proof does not show
    where a reply should end: see spool.whole_reply(). A client that checks the proof rebuilds these
    bytes from its IDs."""
    if not prompt or not reply:
        raise NotProvable(
            'The reply carries no token IDs. Set "return_token_ids": true in the request body.'
            " An empty reply has none either."
        )
    if not all(
        type(token) is int and 0 <= token < policy.vocabulary for token in (*prompt, *reply)
    ):
        raise NotProvable(
            "The prompt or the reply holds something that is not a token ID of the model's"
            " vocabulary of %d tokens." % policy.vocabulary
        )
    if len(prompt) + len(reply) > policy.max_total_tokens:
        raise NotProvable(
            "This prover covers a prompt and a reply of at most %d tokens together. These have %d."
            % (policy.max_total_tokens, len(prompt) + len(reply))
        )
    with open_spool(policy, mark=True) as top:
        if is_on_hold(policy, top) or (held and held(top)):
            raise OnHold(NO_PROOF_ON_HOLD)
        waiting, stored = waiting_and_stored(top)
        if waiting >= policy.max_waiting:
            raise NotProvable(
                "The prover's queue is full, so this reply gets no proof. Ask again with a later"
                " request."
            )
        # Undeleted responses would fill the disk, and the proving service stops when it cannot
        # write.
        if stored >= policy.max_stored:
            raise NotProvable(
                "The server already stores the most responses that its provider allows, so this"
                " reply gets no proof. Ask again later with a new request."
            )
        request = spool.whole_reply(prompt, reply, policy.max_total_tokens, fill=PADDING)
        return spool.submit_at(top, request)


def hold_path(policy):
    return os.path.join(policy.spool_dir, spool.HOLD_FILE)


def is_held(policy, top, jobs, batches):
    """Return whether a client of a policy is to be refused: on hold, or with a wipe to finish (a
    note on disk or in this process's memory); a client can't tell which. The provider's commands
    go by the hold file alone; check and lift-hold also list the notes on disk."""
    return is_on_hold(policy, top) or any(
        note.get("policy") == policy.name for note in batches.all_notes(jobs)
    )


def is_on_hold(policy, top):
    """Return whether a policy is on hold. top is its open spool directory.

    A hold is also looked for under the configured path. If the directory was replaced since this
    process started, a hold that the hold command lays lands in the new one, and must still stop
    this process. What stands under the name counts only if it could be this policy's spool
    directory: it passes spool.open_private_dir() and has a mark file that fits this commitment,
    which a mark that the service wrote does for every commitment. Else whoever
    can aim the name would learn whether some other directory has an entry "hold"."""
    if spool.is_on_hold(top):
        return True
    try:
        with spool.open_private_dir(policy.spool_dir) as named:
            spool.check_mark(named, policy.commitment_id)
            return spool.is_on_hold(named)
    except (spool.UnsafePath, OSError):
        return False


def policies_on_hold(config):
    found = []
    for policy in config.policies.values():
        with contextlib.suppress(FileNotFoundError), open_spool(policy) as top:
            if is_on_hold(policy, top):
                found.append(policy)
    return found


def hold_text(policy):
    return "the policy %s is on hold (%s exists). %s" % (
        policy.name,
        hold_path(policy),
        BEFORE_REMOVING_A_HOLD,
    )


def waiting_and_stored(top):
    """Return the numbers of requests waiting in in/ and of responses stored in out/. A missing
    directory counts as empty."""

    def count(name, suffixes):
        with contextlib.suppress(FileNotFoundError), spool.open_private_dir(name, top) as directory:
            return sum(file.endswith(suffixes) for file in os.listdir(directory))
        return 0

    return count("in", ".req"), count("out", (".%s" % PROOF_SUFFIX, ".%s" % spool.FAILED))


@contextlib.contextmanager
def open_jobs(config, allow_empty=False, mark=False):
    """Open the jobs directory and yield its descriptor.

    The middleware deletes files there, and "jobs_dir" may name the wrong directory. So a directory
    without the mark file raises spool.UnsafePath before anything is written or deleted.
    allow_empty: accept an empty directory too. mark: write the mark file into an empty one."""
    path = config.jobs_dir
    with spool.open_private_dir(".", config.kept[path]) as jobs:
        problem = check_jobs_mark(jobs, allow_empty, mark)
        if problem:
            raise spool.UnsafePath(
                "%s cannot be used as the jobs directory: %s. Nothing in it was changed (set"
                ' "jobs_dir" to a new or empty directory; the init command or the server\'s start'
                " creates the mark file there)" % (path, problem)
            )
        yield jobs


def check_jobs_mark(jobs, allow_empty, create):
    """Return what is wrong with the jobs directory's mark file, or None."""
    try:
        mark = read_file(jobs, spool.MARK_FILE)
        if mark is None and (allow_empty or create):
            # Ignore temporary files from an interrupted write of the mark file.
            if all(name.startswith(".%s." % spool.MARK_FILE) for name in os.listdir(jobs)):
                if not create:
                    return None
                with contextlib.suppress(FileExistsError):  # another process was faster
                    write_file(jobs, spool.MARK_FILE, JOBS_MARK)
            mark = read_file(jobs, spool.MARK_FILE)  # written by this call or by another process
    except OSError as error:
        return "its mark file (mark.txt) cannot be read or created (%s)" % error.strerror
    if mark is None:
        return "it has no mark file (mark.txt)"
    return None if mark == JOBS_MARK else "its mark file (mark.txt) is for something else"


def job_file(proof_id, suffix=".json"):
    """Return the name of a job file (".json") or its record (".checked"): the SHA-256 of the proof
    ID.

    Anyone who has the ID can fetch the proof, so it is written to no file or file name. Names are
    never built from characters that a client sent."""
    return hashlib.sha256(proof_id.encode()).hexdigest() + suffix


def write_file(jobs, name, data):
    """Create a file in the jobs directory, complete or not at all. Raise FileExistsError if the
    name is taken: nothing is replaced."""
    temporary = ".%s.%s" % (name, secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600, dir_fd=jobs)
    try:
        with open(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # or a power failure could leave an empty file
        os.link(temporary, name, src_dir_fd=jobs, dst_dir_fd=jobs, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=jobs)


def read_file(jobs, name):
    """Return a file's bytes; None if it is missing; b"" if it is not a regular file or is longer
    than MAX_JOB_BYTES."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=jobs)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno != errno.ELOOP:
            raise
        return b""  # the name is a symlink
    try:
        data = os.read(fd, MAX_JOB_BYTES + 1) if stat.S_ISREG(os.fstat(fd).st_mode) else b""
    finally:
        os.close(fd)
    return data if len(data) <= MAX_JOB_BYTES else b""


def read_json(jobs, name):
    """Return the JSON object in a file of the jobs directory, or None if there is none."""
    try:
        value = json.loads(read_file(jobs, name) or b"")
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def new_note(policy, tick_ns, members, rejected):
    return {"policy": policy.name, "time": tick_ns, "proofs": sorted(members), "rejected": rejected}


def read_note(jobs, name):
    """Return a note of a withheld batch, or None if there is none. One that stands but can't be
    read is a note with "unreadable": it may name anything, so each caller takes its safe side."""
    with contextlib.suppress(OSError):
        if read_file(jobs, name) is None:
            return None
        found = read_json(jobs, name)
        if found is not None and {"policy", "time", "proofs", "rejected"} <= set(found):
            return found
    _, policy, tick_ns = name.split(".")[:3]
    return {"unreadable": True, "policy": policy, "time": int(tick_ns), "proofs": []}


def note_covers(note, policy, request_id):
    """Return whether a note stops a request of a policy. An unreadable one stops the policy's."""
    if note.get("unreadable"):
        return note["policy"] == policy.name
    return request_id in note.get("proofs", ())


def is_job(value):
    if value is None or not isinstance(value.get("policy"), str):
        return False
    submitted = (
        isinstance(value.get("request_id"), str)
        and spool.REQUEST_ID.match(value["request_id"])
        and isinstance(value.get("tag_key"), str)
        and re.fullmatch("[0-9a-f]{64}", value["tag_key"])
        and "request" in value
        and isinstance(value.get("request_bytes"), str)
        and re.fullmatch("(?:[0-9a-f]{2})+", value["request_bytes"])
    )
    return bool(submitted) or "not_provable" in value


def is_record(value):
    if value is None or not isinstance(value.get("y"), str) or value.get("verified") is not True:
        return False
    proof = value.get("proof")
    return bool(
        re.fullmatch("(?:[0-9a-f]{2})+", value["y"])
        and isinstance(proof, list)
        and len(proof) == 5
        and all(type(number) is int for number in proof)
    )


JOB_FORMS = {".json": is_job, ".checked": is_record}


def read_job(jobs, proof_id):
    """Return the job for a proof ID as a dictionary, or None if there is no well-formed one."""
    job = read_json(jobs, job_file(proof_id))
    return job if is_job(job) else None


def read_record(jobs, proof_id):
    """Return a job's record, or None if there is no well-formed one.

    It is written when the response's tag has matched and its whole batch has passed the check:
    {"y": the first file's bytes in hex, "proof": identity() of the proof file, "verified": true}.
    While that identity() stays the same, the state and y.bin are served without reading the
    proof."""
    record = read_json(jobs, job_file(proof_id, ".checked"))
    return record if is_record(record) else None


def delete_job(jobs, proof_id):
    """Delete a job file, then its record, each only if well-formed. Call it before deleting the
    response, so that no part of a half-deleted response can be fetched."""
    for suffix, is_well_formed in JOB_FORMS.items():
        name = job_file(proof_id, suffix)
        # Missing: another call was faster. Once the job file is gone, the job is deleted: a record
        # that cannot be read or deleted stays for clean.
        with contextlib.suppress(FileNotFoundError if suffix == ".json" else OSError):
            if is_well_formed(read_json(jobs, name)):
                os.unlink(name, dir_fd=jobs)


def identity(fd):
    """Return what shows that a proof file was replaced or modified: size, device, inode,
    modification time and status-change time.

    It saves reading the file again. It misses a change that leaves all five the same, such as a
    write within one tick of the file system's clock; stream_proof() guards against that."""
    info = os.fstat(fd)
    return [info.st_size, info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]


@dataclasses.dataclass(repr=False)
class TagCheck:
    """Checks that a response with a proof answers a job's request, complete and unchanged.

    The tag file must contain the HMAC, under the job's key, of the request ID and the response's
    bytes. Feed the proof through read(), then ask matches(). Never log or raise the key or a tag:
    with a tag for bytes of their choice, someone who can write in out/ could get them served."""

    proof: int
    size: int
    identity: list  # taken before the proof is read
    y: bytes
    tag: bytes
    key: bytes
    request_id: str
    left: int = 0  # bytes not read yet
    digest: object = None
    # Set by keep_open(); closes the descriptor when the TagCheck is dropped.
    file: object = None

    @classmethod
    def start(cls, job, response, fds):
        """response, fds: from spool.open_response_at(); response.faults is empty."""
        first, last = spool.RESPONSE_FILES
        found = identity(fds[last])
        check = cls(
            proof=fds[last],
            size=found[0],
            identity=found,
            y=os.read(fds[first], response.files[first]["size"] + 1),
            tag=os.read(fds[spool.TAG], spool.TAG_BYTES + 1),
            key=bytes.fromhex(job["tag_key"]),
            request_id=job["request_id"],
        )
        check.rewind()
        return check

    def rewind(self):
        """Seek back to the proof file's start. y and the tag are not read again."""
        os.lseek(self.proof, 0, os.SEEK_SET)
        self.left = self.size
        self.digest = spool.tag_hmac(self.key, self.request_id, self.y, self.size)

    def read(self):
        """Return the next chunk of the proof file, or b"" at its end."""
        size = min(CHUNK, self.left)
        # After close() the file object raises; a bare descriptor number could by then be another
        # file's.
        chunk = self.file.read(size) if self.file else os.read(self.proof, size)
        self.digest.update(chunk)
        self.left -= len(chunk)
        return chunk

    def matches(self):
        return not self.left and hmac.compare_digest(self.tag, self.digest.digest())

    def inputs(self):
        """Return a hash of what matches() depends on, with the proof file's identity() for its
        bytes.

        So a proof file repaired in place within one tick of the file system's clock after a failed
        check keeps its hash, and still counts as failed."""
        inputs = (self.identity, self.y, self.tag, self.key)
        return hashlib.sha256(repr(inputs).encode()).digest()

    def keep_open(self, fds):
        """Take the proof's descriptor out of open_response_at()'s dictionary, so that it stays
        open."""
        self.file = open(fds.pop(PROOF_SUFFIX), "rb", buffering=0)


def clean(config, days):
    """Delete job files, records, waiting requests and response files older than 'days' days ("clean
    --help"). Return the numbers of jobs, requests and responses deleted, and of files that could
    not be deleted.

    A file that cannot be deleted, or has a job's or record's name but is not one, is kept and
    logged. In a spool directory a symlink is removed, not followed. Jobs go first, so that none
    points to a half-deleted response, but for a job whose response or request stays: a proof
    without a job would hold back its batch. cp_prove passes over a request that its submitter has
    removed. Raise OnHold if a policy is on hold, at the start or before a deletion: a person is
    then looking for a cause, and may need these files."""
    oldest, deleted, failed, tops = time.time() - 86400 * days, set(), [], []
    young = set()  # request IDs with a response file that stays: their jobs stay too

    def stays(folder, name, opened):
        with contextlib.suppress(FileNotFoundError):  # moved on: then it is in the next folder
            about = os.stat(name, dir_fd=opened, follow_symlinks=False)
            return folder == "work" or about.st_mtime >= oldest - 3600
        return True

    def stop_at_a_hold():
        for policy, top in tops:
            if is_on_hold(policy, top):
                done = "clean stopped early; some files are already deleted"
                raise OnHold(
                    "%s: %s" % (done if deleted else "nothing was deleted", hold_text(policy))
                )

    def sweep(kind, directory, shown, pattern, forms=None):
        # A request must be an hour older, so that its job, which is written just after it, is
        # deleted in the same run and does not wait for a request that is gone.
        limit = oldest - 3600 * (kind == "request")
        for name in filter(pattern.fullmatch, os.listdir(directory)):
            try:
                if os.stat(name, dir_fd=directory, follow_symlinks=False).st_mtime >= limit:
                    continue
                job = read_json(directory, name[:64] + ".json") if forms else None
                if job and job.get("request_id") in young:
                    continue
                if forms and not forms[name[64:]](read_json(directory, name)):
                    log.warning(
                        "%s was not deleted: it has the name of a job file or record but is not one"
                        " (if it is yours, move it out of the jobs directory)",
                        os.path.join(shown, name),
                    )
                    continue
                stop_at_a_hold()
                os.unlink(name, dir_fd=directory)
                deleted.add((kind, name[:64]))  # the files under one ID count once
            except FileNotFoundError:
                continue  # a client's DELETE was faster
            except OSError as error:
                log.error("%s was not deleted: %s", os.path.join(shown, name), error.strerror)
                failed.append(name)

    with contextlib.ExitStack() as stack:
        for policy in config.policies.values():
            with contextlib.suppress(FileNotFoundError):
                tops.append((policy, stack.enter_context(open_spool(policy))))
        stop_at_a_hold()
        for _, top in tops:
            with contextlib.suppress(FileNotFoundError), spool.open_private_dir("out", top) as out:
                for name in filter(RESPONSE_FILE.fullmatch, os.listdir(out)):
                    with contextlib.suppress(FileNotFoundError):
                        if os.stat(name, dir_fd=out, follow_symlinks=False).st_mtime >= oldest:
                            young.add(name[:64])
            for folder in ("in", "work"):  # a request that stays: a proof of it may come
                with contextlib.suppress(FileNotFoundError):
                    with spool.open_private_dir(folder, top) as opened:
                        for name in os.listdir(opened):
                            if stays(folder, name, opened):
                                young.add(name[:64])
        with contextlib.suppress(FileNotFoundError):
            with open_jobs(config, allow_empty=True) as jobs:
                sweep("job", jobs, config.jobs_dir, JOB_FILE, JOB_FORMS)
                sweep("withheld", jobs, config.jobs_dir, WITHHELD_DONE)
        for (policy, top), (kind, path, pattern) in itertools.product(
            tops,
            (
                ("request", "in", REQUEST_FILE),
                ("request", "in/.tmp", UNFINISHED_REQUEST),
                ("response", "out", RESPONSE_FILE),
            ),
        ):
            with contextlib.suppress(FileNotFoundError), contextlib.ExitStack() as parts:
                directory = top
                for part in path.split("/"):
                    directory = parts.enter_context(spool.open_private_dir(part, directory))
                sweep(kind, directory, os.path.join(policy.spool_dir, path), pattern)
    counts = collections.Counter(kind for kind, _ in deleted)
    return counts["job"], counts["request"], counts["response"], len(failed)


def config_folder(policy, work):
    """Lay copies of a policy's configuration files in a new folder below work; return its path."""
    folder = os.path.join(work, "config")
    os.mkdir(folder)
    shutil.copyfile(policy.parameter_file, os.path.join(folder, "policy.txt"))
    above = os.path.dirname(policy.parameter_file)
    for name in ("KL.bin", "layout.txt"):
        shutil.copyfile(os.path.join(above, name), os.path.join(folder, name))
    return folder


def check_verifiers(config):
    """Raise ConfigError unless each policy's verifier can run, lies where the config's other public
    files may, and gives its public files the policy's commitment ID. At a server's start and in
    the commands check and init: a wrong ID or path is then found before the first batch."""
    for policy in config.policies.values():
        path = policy.verifier
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise ConfigError('policy %s: "verifier": %s cannot run' % (policy.name, path))
        try:
            reject_secret_key_dir(os.path.dirname(path))
            os.close(spool.open_public_file(path))
        except (spool.UnsafePath, OSError) as error:
            raise ConfigError('policy %s: "verifier": %s' % (policy.name, error)) from None
        if plan_id(policy) != policy.commitment_id:
            raise ConfigError(
                "policy %s: the verifier %s gives these public files and this parameter file"
                " another commitment ID than the config file, or none; or it is a test build, or"
                " KL.bin or layout.txt is missing beside the parameter file" % (policy.name, path)
            )
        limits = policy.limits
        if policy.kind == "clipped-gap" and limits["theta"] == limits["C"] and limits["cap"] > 0:
            raise ConfigError(
                "policy %s: theta equals C: check_proof.py takes no such limits for a short reply,"
                " so that its batch could never pass" % policy.name
            )


def plan_id(policy):
    """Return the commitment ID that the policy's verifier computes from its public files, or None.

    Asked once, at the start: a config file with a wrong ID or a wrong public key directory is then
    found before the first batch, which the check would otherwise withhold."""
    key = os.path.join(policy.public_key_dir, "key.bin")
    with tempfile.TemporaryDirectory(prefix="plan-id-") as work:
        try:
            words = [policy.verifier, "-plan", "id", config_folder(policy, work), key]
            done = subprocess.run(words, cwd=work, env={}, capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
    printed = done.stdout.decode(errors="replace")
    found = re.findall(r"^commitment ID: ([0-9a-f]{64})$", printed, re.M)
    if check_proof.TEST_BUILD in printed:  # its NO says nothing of a proof
        return None
    return found[0] if done.returncode == 4 and len(found) == 1 else None


def kept_path(fd):
    """Return a path to an open directory. It reaches what the descriptor holds, whatever stands
    under the directory's name by now, in this process and in a child that gets the descriptor."""
    return "/proc/self/fd/%d" % fd


def is_own_folder(path):
    """Return whether path is a directory of this account, and no link."""
    about = os.lstat(path)
    return stat.S_ISDIR(about.st_mode) and about.st_uid == os.geteuid()


@contextlib.contextmanager
def second_open(fd):
    """Yield a descriptor of an open directory that shares no lock with fd: a child that outlives
    this server must not keep the check's lock."""
    second = os.open(kept_path(fd), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        yield second
    finally:
        os.close(second)


def work_prefix(jobs):
    """Return the start of the name of a check's temporary folder: its own for each jobs directory,
    so that the next check can sweep what a dead process left, and nothing else."""
    about = os.fstat(jobs)
    return "%s%x.%x-" % (CHECK_WORK, about.st_dev, about.st_ino)


@contextlib.contextmanager
def check_folder(jobs, size):
    """Yield a new private folder: in memory if that can be had, is closed to other accounts and has
    room for copies of size bytes, else below the open jobs directory."""
    prefix = work_prefix(jobs)
    try:
        if others_can_change(MEMORY_DIR):  # another account could take the folder's name
            raise OSError(errno.EACCES, "open to other accounts")
        if shutil.disk_usage(MEMORY_DIR).free < 2 * size:
            raise OSError(errno.ENOSPC, "no room")
        work = tempfile.mkdtemp(prefix=prefix, dir=MEMORY_DIR)
    except OSError:
        name = prefix + secrets.token_hex(8)
        with spool.open_private_dir(name, jobs, create=True):
            work = os.path.join(kept_path(jobs), name)
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def has_ended(child):
    """Return whether a child has ended or is gone; it stays unreaped."""
    try:
        return bool(os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT))
    except ChildProcessError:
        return True


def is_unreaped(child):
    """Return whether a child is still ours to signal: running, or ended and not reaped."""
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return False
    return True


def cpu_limit():
    """Return the CPU seconds that a check may use: threads count together."""
    return math.ceil(CHECK_TIMEOUT_S * CHECK_THREADS)


def run_to_the_end(words, work, fds=()):
    """Run a program in a session of its own, with no core dump, which could hold a spoiled proof,
    with these descriptors, and with a limit of its own on CPU time, which holds even where this
    server dies first; return its exit code and standard output. At the time limit ask it to end,
    then kill the session, so that what it started goes too, and return (None, b"")."""
    child = subprocess.Popen(
        ["/bin/sh", "-c", 'ulimit -c 0; ulimit -t %d; exec "$@"' % cpu_limit(), "sh", *words],
        cwd=work,
        env={"TMPDIR": work},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=fds,
    )
    try:
        printed = child.communicate(timeout=CHECK_TIMEOUT_S)[0]
    except subprocess.TimeoutExpired:
        # A group's number is sure only while its leader is unreaped. Where something else has
        # reaped the child no signal goes out, and what it started lives on to its CPU limit.
        # WNOWAIT looks without reaping.
        if is_unreaped(child):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + STOP_S
            while time.monotonic() < deadline and not has_ended(child):
                time.sleep(0.05)
            if is_unreaped(child):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)  # what it started, if it ended by itself
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(STOP_S)
        child.stdout.close()
        return None, b""
    return child.returncode, printed


def run_check(policy, job, y, proof_path, jobs):
    """Run check_proof.py on one response, in a process of its own. proof_path goes by kept_path()
    and jobs is a descriptor: the directory is the one that the server holds, and the record ties
    the file by its identity. Return PASSED, REJECTED, or None for anything else, which says
    nothing of the response: no verdict, a mistake in the files, a verifier that is missing, the
    time limit.

    A pass takes what integrations/README.md asks of a program: the exit code, which alone does not
    suffice, the line result: and its fields. Only cp_verify's own NO on the proof counts as a
    rejection: a wrong path, or a new parameter file under the old name, must not cost a batch.
    What the child prints goes no further than
    this function: at a rejection it can tell of the file."""
    request = bytes.fromhex(job["request_bytes"])
    start, end = struct.unpack("<2I", request[-8:])
    checked = "%d:%d" % (start, end)
    room = os.stat(proof_path).st_size * policy.checks_at_once
    with second_open(jobs) as jobs, check_folder(jobs, room) as work:
        request_file = os.path.join(work, "request.bin")
        with open(request_file, "xb") as f:
            f.write(request)
        words = [sys.executable, "-I", "-B", check_proof.__file__, proof_path, request_file]
        words += ["--commitment-id", policy.commitment_id, "--range", checked]
        words += ["--threads", str(CHECK_THREADS)]
        words += ["--end-token", "none", "--public-key-dir", policy.public_key_dir]
        words += ["--parameter-file", policy.parameter_file, "--verifier", policy.verifier]
        words += ["--scales-file", os.path.join(os.path.dirname(policy.parameter_file), "KL.bin")]
        if end == policy.max_total_tokens:
            words += ["--next-token", str(struct.unpack("<I", y)[0])]
        if policy.kind == "clipped-gap":
            words += ["--policy", "clipped-gap"]
            for name in check_proof.POLICY_LIMITS:
                words += ["--" + name, str(policy.limits[name])]
        out = re.match(r"/proc/self/fd/([0-9]+)/", proof_path)
        fds = {int(out[1])} if out else set()
        if work.startswith(kept_path(jobs) + "/"):
            fds.add(jobs)
        code, printed = run_to_the_end(words, work, tuple(sorted(fds)))
    if code is None:  # no timing of a check stands yet: this says nothing of the response
        log.error("Policy %s: a check gave no answer: the time limit", policy.name)
        return None
    lines = printed.decode(errors="replace").strip("\n").split("\n")
    last = lines[-1]
    if last.startswith(CP_VERIFY_NO) and check_proof.REJECTED_LEAF_FILE not in last:
        return REJECTED
    fields = lines[-2].split() if len(lines) > 1 else []
    if code == 0 and fields[:2] in (
        ["result:", "outcome=verified"],
        ["result:", "outcome=within-limits"],
    ):
        wanted = {"commitment_id=" + policy.commitment_id, "range=" + checked}
        if wanted <= set(fields):
            return PASSED
    cause = next((name for start, name in OTHER_ENDS if last.startswith(start)), None)
    if cause is None:
        missing = " is missing" in last or "does not exist" in last
        cause = "a file is missing" if missing else "another end"
    log.error("Policy %s: a check gave no answer: %s", policy.name, cause)
    return None


def lay_hold(top):
    """Create the empty file hold in a spool directory, if it is not there."""
    with contextlib.suppress(FileExistsError):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        os.close(os.open(spool.HOLD_FILE, flags, 0o600, dir_fd=top))
    os.fsync(top)


def overwrite(out, name, tick_ns):
    """Write zeros over a file in out/, if it is a file there, and give it back its batch's time:
    a death behind this must not make a batch of its own of it."""
    with contextlib.suppress(FileNotFoundError):
        flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        try:
            fd = os.open(name, flags, dir_fd=out)
        except OSError as error:
            if error.errno == errno.ELOOP:  # a link: nothing to zero; the delete takes it
                return
            raise
        with open(fd, "r+b", buffering=0) as f:
            left = os.fstat(fd).st_size if stat.S_ISREG(os.fstat(fd).st_mode) else 0
            while left:
                left -= f.write(bytes(min(left, CHUNK)))
            os.utime(fd, ns=(tick_ns, tick_ns))
            os.fsync(fd)


def replace_by_failed(out, request_id, job, tick_ns):
    """Overwrite and delete a response. With its job, lay in its place the failed response that the
    job's key allows, with the batch's time on both files. Can be done again after a crash."""
    fault = None
    for suffix in reversed(spool.RESPONSE_FILES):  # the proof first
        try:
            overwrite(out, "%s.%s" % (request_id, suffix), tick_ns)
        except OSError as error:  # not zeroed: the caller tells; the delete comes all the same
            fault = error
    spool.delete_response_at(out, request_id)  # what it can, and then raises
    if job is not None:
        lay_failed(out, request_id, job, tick_ns)
    if fault:
        raise fault


def lay_failed(out, request_id, job, tick_ns):
    """Lay the failed response that a job's key allows, with its batch's time on both files."""
    tag = spool.failed_tag(bytes.fromhex(job["tag_key"]), request_id)
    for suffix, data in ((spool.TAG, tag), (spool.FAILED, b"")):
        name = "%s.%s" % (request_id, suffix)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
        with open(os.open(name, flags, 0o600, dir_fd=out), "wb") as f:
            f.write(data)
            f.flush()
            os.utime(f.fileno(), ns=(tick_ns, tick_ns))
            os.fsync(f.fileno())


def is_well_formed_failure(job, request_id, fds):
    """Return whether an open failed response has its right tag."""
    tag = os.read(fds[spool.TAG], spool.TAG_BYTES + 1)
    key = bytes.fromhex(job["tag_key"])
    return hmac.compare_digest(tag, spool.failed_tag(key, request_id))


def start_worker(target):
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    return worker


def withheld_file(policy, tick_ns):
    return "withheld.%s.%d.json" % (policy.name, tick_ns)


class JobsFound(dict):
    unread = 0


class BatchChecks:
    """The checks of one process. ask() starts one, and waits up to WAIT_S for its end."""

    def __init__(self, config):
        self.config, self.lock, self.threads = config, threading.Lock(), {}
        # By (policy name, tick_ns): what had no answer at the last try; tries while the service
        # was at work on a member; whether the operator was told.
        self.faults, self.waits, self.told = {}, collections.Counter(), set()
        self.tries = collections.Counter()  # tries without an answer, whatever their causes
        # By (policy name, tick_ns): the notes of this process's NOs, from before anything is
        # written until the hold lies and the wipe is done: where nothing can be written, they alone
        # stop the policy. They die with the process, and no other process sees them.
        self.remembered = {}
        self.not_before = {}  # {(policy name, tick_ns): time.monotonic() of the next try}

    def ask(self, policy, tick_ns):
        """Start the check of a batch, if it is due and not running; wait up to WAIT_S for it."""
        key = policy.name, tick_ns
        with self.lock:
            thread = self.threads.get(key)
            if thread is None or not thread.is_alive():
                if time.monotonic() < self.not_before.get(key, 0):
                    return
                thread = threading.Thread(target=self.run, args=(policy, tick_ns), daemon=True)
                self.threads = {key: one for key, one in self.threads.items() if one.is_alive()}
                self.threads[key] = thread
                thread.start()
        thread.join(WAIT_S)

    def finish_all(self):
        """At a start: finish every withheld batch that a process that died left half done."""
        with contextlib.suppress(spool.UnsafePath, OSError), open_jobs(self.config) as jobs:
            for note in self.notes(jobs):
                policy = self.config.policies.get(note.get("policy"))
                if policy is not None:
                    self.ask(policy, note["time"])

    def all_notes(self, jobs):
        """Return the notes of unfinished wipes, on disk and in this process's memory."""
        return self.notes(jobs) + list(self.remembered.values())

    @staticmethod
    def notes(jobs):
        """Return the notes of withheld batches whose wipe is not done."""
        names = filter(WITHHELD_FILE.fullmatch, os.listdir(jobs))
        return [read_note(jobs, name) or {} for name in names if not name.endswith(".done.json")]

    def finish(self, jobs, policy, request_id):
        """Ask for the check of a withheld batch that names this request: a process may have died
        between deleting a response and laying the failed one."""
        for note in self.notes(jobs):
            if note.get("policy") == policy.name and note_covers(note, policy, request_id):
                self.ask(policy, note["time"])

    def run(self, policy, tick_ns):
        try:
            with open_jobs(self.config) as jobs:
                fcntl.flock(jobs, fcntl.LOCK_EX)  # one check at a time, in all processes
                self.sweep(jobs)
                self.check_batch(jobs, policy, tick_ns)
        except Exception as error:  # in a thread of its own: nothing else would see it
            name = type(error).__name__
            self.no_answer(policy, tick_ns, getattr(error, "strerror", None) or name, [name])

    def sweep(self, jobs):
        """Remove what a process that died in a check left. Runs under the lock."""
        prefix = work_prefix(jobs)
        for above in (MEMORY_DIR, kept_path(jobs)):
            with contextlib.suppress(OSError):
                for name in os.listdir(above):
                    path = os.path.join(above, name)
                    with contextlib.suppress(OSError):  # gone by now: the next one
                        if name.startswith(prefix) and is_own_folder(path):
                            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def jobs_of(jobs, policy):
        """Return {request ID: (the job file's name without ".json", job)} for a policy's proofs,
        with unread: how many job files could not be read, so that no request counts as having
        no job for sure."""
        found = JobsFound()
        for name in os.listdir(jobs):
            is_a_job = JOB_FILE.fullmatch(name) and name.endswith(".json")
            job = None
            try:
                job = read_json(jobs, name) if is_a_job else None
            except OSError:  # one file that can't be read stops no batch
                found.unread += 1
            if is_job(job) and "not_provable" not in job and job["policy"] == policy.name:
                found[job["request_id"]] = name[:-5], job
        return found

    @staticmethod
    def members(jobs, out, policy, tick_ns, read=True):
        """Return what must pass of a batch: {request ID: (stem, job, record, kept)}. With read
        false no proof is read, and a record is None.

        Every response with a file of this time belongs to it, but for a well-formed failed one
        with its right tag, or with no job to compare it: that is cp_prove's own answer, and costs
        its mates nothing. stem is the job file's name without ".json"; kept: it has its record
        already; record what the check writes if it passes. None for a record: it cannot pass: it
        has no job, a fault, a size out of range or a wrong tag."""
        jobs_of = BatchChecks.jobs_of(jobs, policy)
        times = collections.defaultdict(set)  # {request ID: the times of its files}
        for name in os.listdir(out):
            request_id, _, suffix = name.partition(".")
            if suffix in (PROOF_SUFFIX, "y", spool.TAG, spool.FAILED):
                with contextlib.suppress(FileNotFoundError):
                    about = os.stat(name, dir_fd=out, follow_symlinks=False)
                    if spool.REQUEST_ID.match(request_id):
                        times[request_id].add(about.st_mtime_ns)
        # A response whose files bear two times ties the two batches together.
        ticks, ids = {tick_ns}, set()
        while True:
            more = {one for one, its_times in times.items() if its_times & ticks} - ids
            if not more:
                break
            ids |= more
            ticks = ticks.union(*(times[one] for one in more))
        found = JobsFound()
        found.unread = jobs_of.unread
        for request_id in sorted(ids):
            stem, job = jobs_of.get(request_id, (None, None))
            with spool.open_response_at(out, request_id) as (response, fds):
                if response is None or response.faults:
                    found[request_id] = stem, job, None, False
                elif response.failed:
                    well_formed = job and is_well_formed_failure(job, request_id, fds)
                    if not well_formed and (job or jobs_of.unread):
                        found[request_id] = stem, job, None, False
                elif job is None:
                    found[request_id] = None, None, None, False
                else:
                    found[request_id] = BatchChecks.member(jobs, stem, job, response, fds, read)
            # DELETE takes the job first, then the files: with the job gone by now, what looked
            # malformed is a response on its way out.
            if job and request_id in found and found[request_id][2] is None:
                if not is_job(read_json(jobs, stem + ".json")):
                    del found[request_id]
                    if read:
                        BatchChecks.tell_of_a_job_gone(out, policy, request_id)
        return found

    @staticmethod
    def tell_of_a_job_gone(out, policy, request_id):
        """A client's DELETE leaves no proof for long. One that stays was passed over unjudged."""
        with contextlib.suppress(FileNotFoundError):
            os.stat(request_id + "." + PROOF_SUFFIX, dir_fd=out, follow_symlinks=False)
            log.warning(
                "Policy %s: response %s got no verdict, because its job is gone (by a client's"
                " DELETE, the clean command, or by hand); the rest of its batch goes on without it."
                " Its proof was still in out/: if it stays there, remove the response's files by"
                " hand (first see integrations/LIMITS.md#the-middlewares-check)",
                policy.name,
                request_id,
            )

    @staticmethod
    def member(jobs, stem, job, response, fds, read):
        kept = read_json(jobs, stem + ".checked")
        if is_record(kept) and is_recorded(kept, response, fds):
            return stem, job, kept, True
        if not read:
            return stem, job, None, False
        if is_record(kept):  # of a file that is no longer there: it must not stand in the way
            with contextlib.suppress(FileNotFoundError):
                os.unlink(stem + ".checked", dir_fd=jobs)
        check = TagCheck.start(job, response, fds)
        if not 0 < check.size <= MAX_PROOF_BYTES:
            return stem, job, None, False
        while check.read():
            pass
        if not check.matches():
            return stem, job, None, False
        return stem, job, {"y": check.y.hex(), "proof": check.identity, "verified": True}, False

    def check_batch(self, jobs, policy, tick_ns):
        with open_spool(policy) as top, spool.open_private_dir("out", top) as out:
            earlier = read_note(jobs, withheld_file(policy, tick_ns))
            earlier = earlier or self.remembered.get((policy.name, tick_ns))
            if earlier and earlier.get("unreadable"):  # nothing can be decided on that
                why = (
                    "its note %s in the jobs directory can't be read or is not whole. Do not remove"
                    " it: without it, proofs of this rejected batch could be served again. Make it"
                    " readable (lift-hold alone won't do); if it is not whole, see"
                    " integrations/LIMITS.md#the-middlewares-check" % withheld_file(policy, tick_ns)
                )
                return self.no_answer(policy, tick_ns, why, ["the note"])
            if earlier:  # a wipe is not done: finish it by its note, on disk or in memory
                BatchChecks.try_to_hold(policy, top)
                members = {}
                with contextlib.suppress(Exception):  # the note alone can drive the wipe
                    members = self.members(jobs, out, policy, tick_ns)
                rejected = earlier["rejected"]
                return self.withhold(jobs, policy, tick_ns, top, out, members, rejected, earlier)
            if spool.is_on_hold(top):  # a person is looking for a cause
                return
            members = self.members(jobs, out, policy, tick_ns)
            rejected, unanswered, not_final = [], [], []
            try:
                lists = rejected, unanswered, not_final
                self.judge(jobs, out, policy, tick_ns, top, members, *lists)
            except Exception as error:  # what has been rejected so far stands
                unanswered.append(type(error).__name__)
            if rejected:
                self.withhold(jobs, policy, tick_ns, top, out, members, rejected)
            elif unanswered:
                why = "%d without an answer, first %s" % (len(unanswered), unanswered[0])
                self.no_answer(policy, tick_ns, why, unanswered)
            elif not_final:  # the service is at work on them: not yet, unless that lasts
                self.waits[policy.name, tick_ns] += 1
                self.not_before[policy.name, tick_ns] = time.monotonic() + RETRY_S
                if self.waits[policy.name, tick_ns] >= NOT_FINAL_TRIES:
                    why = (
                        "%d of its requests stay in in/ or work/: see that cp_prove runs, or take"
                        " them out, first %s" % (len(not_final), not_final[0])
                    )
                    self.no_answer(policy, tick_ns, why, not_final)
            elif not all(
                [
                    self.write_record(jobs, stem, record)
                    for stem, _, record, kept in members.values()
                    if not kept
                ]
            ):
                self.no_answer(policy, tick_ns, "a file stands in the way", ["in the way"])
            else:
                self.faults.pop((policy.name, tick_ns), None)
                self.tries.pop((policy.name, tick_ns), None)

    def judge(self, jobs, out, policy, tick_ns, top, members, rejected, unanswered, not_final):
        """Run the checks of a batch, side by side, and fill the three lists; with not_final filled
        none runs. When it returns or raises, every check has ended, and what was rejected so far
        stands in rejected.

        Workers only wait for their children, and note a NO: all the rest is decided behind them,
        in check_batch(). At the first NO the operator is told and the hold laid, and no further
        check starts."""
        due = []
        for request_id, (stem, job, record, kept) in members.items():
            if is_not_final(top, request_id):
                not_final.append(request_id)
            elif job is None:
                unanswered.append(
                    "response %s, which has no job, unless one of them can't be read: mend that,"
                    " or remove out/%s.* or put its job back, then lift-hold"
                    % (request_id, request_id)
                )
            elif record is None:
                unanswered.append(
                    "response %s, which looks malformed: if it stays so, remove out/%s.* and its"
                    " job, whose client then has no proof, and lift-hold" % (request_id, request_id)
                )
            elif not kept:
                due.append((request_id, stem, job, bytes.fromhex(record["y"])))
        if not_final:  # the batch can't pass yet: no check is spent on it
            return
        no_seen, lock = threading.Event(), threading.Lock()

        def one(item):
            request_id, stem, job, y = item
            if no_seen.is_set():
                return request_id, SKIPPED
            path = os.path.join(kept_path(out), request_id + "." + PROOF_SUFFIX)
            try:
                end = struct.unpack("<I", bytes.fromhex(job["request_bytes"])[-4:])[0]
                if end != policy.max_total_tokens and y != bytes(4):
                    outcome = REJECTED  # the next token is outside the range: no check reads y
                else:
                    outcome = run_check(policy, job, y, path, jobs)
            except Exception as error:  # in a worker: nothing else would see it
                log.error("Policy %s: a check ended with %s", policy.name, type(error).__name__)
                outcome = None
            if outcome is None and not is_job(read_json(jobs, stem + ".json")):
                BatchChecks.tell_of_a_job_gone(out, policy, request_id)
                return request_id, SKIPPED  # its job went meanwhile
            if outcome == REJECTED:
                with lock:
                    first = not no_seen.is_set()
                    no_seen.set()
                    rejected.append(request_id)
                if first:
                    note = new_note(policy, tick_ns, members, [request_id])
                    self.remembered[policy.name, tick_ns] = note
                    self.tell_of_a_no(jobs, policy, tick_ns, top, members, [request_id])
            return request_id, outcome

        waiting, workers, answered = collections.deque(due), [], []

        def work():
            while True:
                try:
                    item = waiting.popleft()
                except IndexError:
                    return
                request_id, outcome = one(item)
                answered.append(request_id)
                if outcome not in (PASSED, REJECTED, SKIPPED):
                    unanswered.append("a check without an answer")  # any two in a row: the hold
                    log.error(
                        "Policy %s: no answer for response %s: mend the cause named above; if the"
                        " file itself is at fault, remove out/%s.* and its job, for it can hold"
                        " values that are not masked; then lift-hold",
                        policy.name,
                        request_id,
                        request_id,
                    )

        try:  # daemons: a server that ends does not wait for the rest of a batch
            for _ in range(min(policy.checks_at_once, len(due))):
                workers.append(start_worker(work))
        finally:
            for worker in workers:
                worker.join()
        if len(answered) != len(due):  # no worker could be started, or one died
            unanswered.append("a check that did not end")

    @staticmethod
    def tell_of_a_no(jobs, policy, tick_ns, top, members, rejected):
        """Log, note, hold, in this order. The log first: whatever fails next, the operator is told.
        The note next: once it stands no record of the batch counts, whatever dies; a death before
        it leaves what a death before the verdict leaves, and the next check finds the NO again.
        Where it can't be written, the records go at once."""
        standing = False
        try:
            log.error(
                "Policy %s: a response of the batch of time %d did not pass. Such a proof can hold"
                " values that are not masked. The batch will be answered as failed. See"
                ' docs/faults.md, "Always first"',
                policy.name,
                tick_ns,
            )
            BatchChecks.write_note(jobs, policy, tick_ns, members, rejected)
            written = read_note(jobs, withheld_file(policy, tick_ns))
            standing = written is not None and not written.get("unreadable")
        except Exception as error:
            log.error(
                "Policy %s: the note of the rejected batch could not be written and read back (%s)."
                " So the server now tries to remove the batch's *.checked files from the jobs"
                " directory: without them none of its proofs that passed earlier is served",
                policy.name,
                type(error).__name__,
            )
        finally:  # whatever happened to the note, these two come
            try:
                if not standing:
                    for stem, _, _, _ in members.values():
                        if stem is not None:
                            with contextlib.suppress(OSError):
                                os.unlink(stem + ".checked", dir_fd=jobs)
            finally:
                if not BatchChecks.try_to_hold(policy, top) and not standing:
                    log.error(
                        "Policy %s: the rejection could not be written down: no whole note could be"
                        " read back from the jobs directory, and the hold could not be laid. Only"
                        " this process's memory is sure to stop the policy: another server process,"
                        " or this one after a restart, may go on serving it, and with it those of"
                        " the batch's proofs that had passed. What to do now, and before you stop"
                        " or restart a server: integrations/LIMITS.md#the-middlewares-check",
                        policy.name,
                    )

    @staticmethod
    def try_to_hold(policy, top):
        """Return whether the hold lies."""
        try:
            lay_hold(top)
        except OSError as error:
            log.error("Policy %s: the hold could not be laid: %s", policy.name, error.strerror)
            return False
        return True

    @staticmethod
    def write_note(jobs, policy, tick_ns, members, rejected):
        """Write down what is to go, and return it. Without room the wipe goes on without it."""
        note = new_note(policy, tick_ns, members, rejected)
        with contextlib.suppress(OSError):
            write_file(jobs, withheld_file(policy, tick_ns), json.dumps(note).encode())
            os.fsync(jobs)
        written = read_note(jobs, withheld_file(policy, tick_ns))
        return note if written is None or written.get("unreadable") else written

    @staticmethod
    def write_record(jobs, stem, record):
        """Write a job's record, and leave none if the job was deleted meanwhile. Return False if
        something else stands under its name."""
        name = stem + ".checked"
        try:
            write_file(jobs, name, json.dumps(record).encode())
        except FileExistsError:  # a file with that name that is no record of this response
            log.error("The file %s in the jobs directory stands in the way: move it away", name)
            return False
        if not is_job(read_json(jobs, stem + ".json")):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=jobs)
        return True

    def withhold(self, jobs, policy, tick_ns, top, out, members, rejected, earlier=None):
        """Clear a batch by its note. If the process dies on the way, the next check goes on from
        the note on disk; a note in memory alone is lost. Every record of the batch goes before any
        proof, and a rejected proof goes last. A lifted hold comes back here: the wipe isn't done,
        and a person should know."""
        note = earlier or BatchChecks.write_note(jobs, policy, tick_ns, members, rejected)
        held = BatchChecks.try_to_hold(policy, top)
        if earlier:
            log.error(
                "Policy %s: the wipe of the batch of time %d goes on%s",
                policy.name,
                tick_ns,
                ", behind a hold" if held else "",
            )
        rejected = list(filter(spool.REQUEST_ID.match, rejected))  # names below out/
        going = filter(spool.REQUEST_ID.match, set(note["proofs"]) | set(members))
        going = sorted(set(going) - set(rejected))
        jobs_of = BatchChecks.jobs_of(jobs, policy)
        stuck = []
        for request_id in going + rejected:
            stem = jobs_of.get(request_id, (None, None))[0]
            if stem is not None:
                try:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(stem + ".checked", dir_fd=jobs)
                except OSError:
                    stuck.append(request_id)
        for request_id in going + rejected:  # one that sticks stops neither the rest nor the last
            try:
                job = jobs_of.get(request_id, (None, None))[1]
                replace_by_failed(out, request_id, job, tick_ns)
                if job is None and jobs_of.unread:  # it may have a job: no failed response yet
                    stuck.append(request_id)
            except OSError:
                stuck.append(request_id)
        held = BatchChecks.try_to_hold(policy, top)  # the deletes may have made room
        if stuck:
            log.error(
                "Policy %s: the wipe of the withheld batch of time %d is not done for these"
                " responses: %s. Remove by hand what is left of their .proof and .y files in out/"
                " (first see integrations/LIMITS.md#the-middlewares-check):"
                " it can hold values that are not masked. If none is left, see that the *.checked"
                " files in the jobs directory can be removed, that every job file there can be"
                " read, and that out/ has room. A later call then goes on with the wipe",
                policy.name,
                tick_ns,
                " ".join(stuck),
            )
            self.not_before[policy.name, tick_ns] = time.monotonic() + RETRY_S
            return
        if not held:
            self.not_before[policy.name, tick_ns] = time.monotonic() + RETRY_S
            return  # the note stands, and stops the policy until a hold does
        name = withheld_file(policy, tick_ns)
        if earlier is None and read_note(jobs, name) != note:
            self.not_before[policy.name, tick_ns] = time.monotonic() + RETRY_S
            return  # no note is closed that this pass has not read
        self.remembered.pop((policy.name, tick_ns), None)  # the hold lies and the wipe is done
        try:
            os.rename(name, name[: -len("json")] + "done.json", src_dir_fd=jobs, dst_dir_fd=jobs)
        except FileNotFoundError:  # it was in memory alone
            pass
        except OSError as error:  # the next check does it over, to no harm
            log.error("Policy %s: the note %s stays: %s", policy.name, name, error.strerror)
            self.not_before[policy.name, tick_ns] = time.monotonic() + RETRY_S

    def no_answer(self, policy, tick_ns, why, causes):
        """Tell of a batch without an answer. Nothing of it leaves and nothing is wiped. If one of
        causes stood at the last try too, or at MOST_TRIES of any causes, the policy goes on hold:
        its clients get an answer, and a person looks. A file on its way in or out is gone by
        then."""
        key = policy.name, tick_ns
        self.not_before[key] = time.monotonic() + RETRY_S
        self.tries[key] += 1
        held = bool(self.faults.get(key, set()) & set(causes)) or self.tries[key] >= MOST_TRIES
        self.faults[key] = set(causes)
        if held:
            self.tries.pop(key, None)
            try:
                with open_spool(policy) as top:
                    lay_hold(top)
            except (spool.UnsafePath, OSError) as error:
                log.error("Policy %s: the hold could not be laid: %s", policy.name, error)
                held = False
        if not held and key in self.told:
            return
        self.told.add(key)
        log.error(
            "Policy %s: the batch of time %d could not be checked: %s. %s",
            policy.name,
            tick_ns,
            why,
            "The policy is on hold until a person has looked: mend the cause, then lift-hold, and a"
            " later call takes the batch up again"
            if held
            else "A later call tries again",
        )


# ---- The ASGI application ----


# Follows the pure ASGI middleware pattern of Starlette's documentation (BSD-3-Clause) and of vLLM's
# middleware (Apache-2.0).
class ProofMiddleware:
    """The ASGI application that vLLM wraps around its application.

    vLLM passes only the inner application; 'config' is for callers that do not use the environment
    variable."""

    def __init__(self, app, config=None):
        self.app, self.fault, self.last_logged = app, None, {}
        # For a server started without --disable-uvicorn-access-log. It changes uvicorn's logger for
        # the whole process, once. uvicorn's log level "trace" still prints paths, headers and keys.
        logging.getLogger("uvicorn.access").addFilter(MASK_PROOF_IDS)
        self.api_key = os.environb.get(API_KEY_VARIABLE.encode(), b"")
        try:
            self.config = load_config(config or os.environ.get(CONFIG_VARIABLE))
            check_verifiers(self.config)
        except Exception as error:
            # __call__ reports it, because a web server may swallow an exception raised here. Of
            # whatever kind: behind an exception uvicorn 0.53.0 serves on, and this method would run
            # again at every call, so that the directories were opened anew with nobody looking on.
            self.fault = str(error) or type(error).__name__
            return
        self.batches = BatchChecks(self.config)
        self.batches.finish_all()
        published = (
            "served_model",
            "commitment_id",
            "max_total_tokens",
            "kind",
            "limits",
            "closed",
        )
        self.table = {
            "policies": {
                name: {key: getattr(policy, key) for key in published}
                for name, policy in self.config.policies.items()
            }
        }

    def log_hourly(self, key, message, *args, **kwargs):
        """Log an error at most once an hour per key, with the number of repeats left out since the
        last line.

        Never raises: a failure to log must not change what a client gets."""
        try:
            now = time.monotonic()
            last, repeats = self.last_logged.get(key, (-3600, 0))
            if now - last < 3600:
                self.last_logged[key] = last, repeats + 1
                return
            self.last_logged[key] = now, 0
            message += " (logged at most once an hour; repeats not logged since the last line: %d)"
            log.error(message, *args, repeats, **kwargs)
        except Exception:
            pass

    def api_keys(self, scope):
        """Return the API keys, one of which a call must carry. If there is none, no key is needed.

        They are the keys that vLLM itself asks for, however it got them (--api-key, a --config
        file, VLLM_API_KEY). vLLM 0.29.0 gives them to its guard as the keyword "tokens", a list of
        strings; Starlette keeps the guard's entry on the application and puts the application into
        every scope. Where none can be read there (for example in a scope without an application, or
        if a key is not plain ASCII), the key is the one in VLLM_API_KEY.

        Raise ValueError if there is a sign that vLLM asks for a key (its guard's entry, or
        --api-key among its options), none can be read and the variable is not set: the routes would
        be open on a server that asks for a key. An application that raises while it is looked
        through counts as such a sign, except for an AttributeError from an attribute that is asked
        for: that attribute counts as missing."""
        app, keys = scope.get("app"), []
        try:
            # vLLM 0.29.0 also keeps its parsed options there: a second sign that it asks for a key.
            # vLLM drops empty keys, so a list of empty strings is no sign.
            given = getattr(getattr(getattr(app, "state", None), "args", None), "api_key", None)
            guarded = any(given) if isinstance(given, list) else bool(given)
            for entry in getattr(app, "user_middleware", None) or ():
                cls, kwargs = getattr(entry, "cls", None), getattr(entry, "kwargs", None)
                package = str(getattr(cls, "__module__", "")).split(".")[0]
                if (package, getattr(cls, "__name__", "")) == VLLM_GUARD:
                    guarded = True
                    tokens = kwargs.get("tokens") if isinstance(kwargs, dict) else None
                    if isinstance(tokens, list) and all(
                        isinstance(one, str) and one and one.isascii() for one in tokens
                    ):
                        keys += [one.encode() for one in tokens]
        except Exception:  # not passed on or logged: its text could hold a key
            guarded, keys = True, []
        if not keys and self.api_key:
            keys = [self.api_key]
        if guarded and not keys:
            raise ValueError(
                "vLLM seems to ask for an API key (it has its own key check, or was given"
                " --api-key, or its application could not be looked through), and the proof"
                " middleware cannot read that key from vLLM's application, so the routes under %s"
                " would be open. Set %s in the server's environment to one of the keys that vLLM"
                " was given, or, if it was given none, to a new key (vLLM then asks for it too):"
                " these routes then ask for that one" % (PREFIX, API_KEY_VARIABLE)
            )
        return keys

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan" and not self.fault:
            try:
                keys = self.api_keys(scope)
            except Exception as error:
                self.fault = str(error) or type(error).__name__
        if self.fault and scope["type"] == "lifespan":
            await receive()  # "lifespan.startup"
            # Sending this makes the web server log the message and exit.
            return await send(
                {"type": "lifespan.startup.failed", "message": "proof middleware: " + self.fault}
            )
        if scope["type"] == "lifespan" and not keys:
            # An error, not a warning: at a start under vLLM 0.29.0 a message logged here appeared
            # in the server's log as an error (one run), and did not as a warning (four runs). Why
            # is not known. The message's present wording has not been run there.
            log.error(
                "The routes under %s need no API key: %s is not set or is empty, and the proof"
                " middleware found no key of vLLM's on vLLM's application, where vLLM 0.29.0 keeps"
                " them. If vLLM was given a key all the same, these routes may be open without it"
                " (the middleware looks again at each call): make one call to them without a key;"
                " status 401 means that they ask for one. To require a key in every case, set %s in"
                " the server's environment, to the key that vLLM was given or to one of them",
                PREFIX,
                API_KEY_VARIABLE,
                API_KEY_VARIABLE,
            )
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # The path without a proxy's root path, as in vLLM's authenticate.py (Apache-2.0).
        path = scope["path"].removeprefix(scope.get("root_path", ""))
        asked = [
            value for name, value in scope["headers"] if name.lower() == POLICY_HEADER.encode()
        ]
        if not asked and not path.startswith(PREFIX):
            return await self.app(scope, receive, send)
        # An exception from the middleware's code must not reach the enclosing layer, which in vLLM
        # sends its text to the client. It is logged here, and the client gets a fixed message, or a
        # broken connection if the response has started. vLLM's exceptions, and those of the web
        # server's send(), pass through.
        started, from_send = False, None

        async def tracking_send(message):
            nonlocal started, from_send
            started = started or message["type"] == "http.response.start"
            try:
                await send(message)
            except Exception as error:
                from_send = error
                raise

        try:
            onward = await self.before_vllm(scope, path, asked, receive, tracking_send)
        except ClientError as error:
            return await send_error(send, error.status, str(error))
        except Exception as error:
            if error is from_send or isinstance(error, ResponseCut):
                raise
            self.log_hourly(
                ("call", cause(error)),
                "An HTTP call failed in the proof middleware",
                exc_info=error,
            )
            if started:
                raise ResponseCut("the proof middleware could not finish a response") from None
            return await send_error(send, 500, SERVER_FAULT)
        if onward is not None:
            await self.app(scope, *onward)

    async def before_vllm(self, scope, path, asked, receive, send):
        """Handle a call to the middleware's routes, or a call with the policy header until vLLM
        takes over.

        Return the receive and send functions to call vLLM with, or None if the call is answered or
        the client has left. Raise ClientError if the configuration failed (status 500), the call
        lacks the server's API key (401), one of the middleware's own routes refuses it, or a call
        with the policy header is not a POST to a serving route, names no open policy (both 400) or
        has a body longer than MAX_REQUEST_BYTES (413)."""
        # Reached if the web server serves although startup failed, or has no lifespan protocol.
        if self.fault:
            self.log_hourly(
                "config",
                "The proof middleware serves no proofs, because its configuration failed: %s",
                self.fault,
            )
            raise ClientError(500, SERVER_FAULT)
        # First, so that a stranger cannot test a policy's name.
        self.check_api_key(scope)
        if path.startswith(PREFIX):
            return await self.own_route(scope["method"], path[len(PREFIX) :], receive, send)
        if scope["method"] != "POST" or path not in SERVING_ROUTES:
            raise ClientError(
                400,
                "Proofs are offered for POST %s and POST %s only. Drop the %s header to use this"
                " route." % (*SERVING_ROUTES, POLICY_HEADER),
            )
        policy = self.config.policies.get(asked[0].decode("latin-1")) if len(asked) == 1 else None
        if policy is None or policy.closed:
            raise ClientError(
                400,
                "The %s header must name one policy that this server offers and that is not closed."
                " GET %spolicies lists them." % (POLICY_HEADER, PREFIX),
            )
        # Keep the whole body: check_request() examines it after the reply has been served.
        body = await read_body(receive)
        if body is None:
            return None
        replay = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive_again():
            return replay.pop() if replay else await receive()

        return receive_again, self.copying_send(send, policy, body)

    def check_api_key(self, scope):
        """Raise ClientError unless a call carries one of the server's API keys, if it has any. vLLM
        checks the key only on vLLM's routes.

        All clients share the keys, so they only keep out strangers; the proof ID keeps one client
        from another's proof. api_keys()'s ValueError passes through: the call gets status 500."""
        keys = self.api_keys(scope)
        if not keys:
            return
        sent = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        # The value is split, and its scheme compared, as in vLLM's authenticate.py (Apache-2.0).
        scheme, _, key = sent[0].partition(b" ") if len(sent) == 1 else (b"", b"", b"")
        # A list, not a generator: every key is compared, whichever fits.
        if scheme.lower() != b"bearer" or not any([hmac.compare_digest(key, one) for one in keys]):
            raise ClientError(
                401,
                "This route needs the server's API key. Send it in the header"
                ' "Authorization: Bearer KEY".',
            )

    def copying_send(self, send, policy, request_body):
        """Return a send function that adds the proof ID to vLLM's reply, copies the reply as it
        passes through, and stores the job."""
        # The ID is generated before the reply exists, because a streamed reply's headers are sent
        # first.
        proof_id, chunks, size = secrets.token_hex(32), None, 0

        async def copy_and_send(message):
            nonlocal chunks, size
            # An exception here would break vLLM's reply. If the middleware's part fails, the
            # message is passed on unchanged and no job is stored, so the proof ID is unknown to the
            # server.
            try:
                if message["type"] == "http.response.start" and message["status"] == 200:
                    chunks = []
                    header = (ID_HEADER.encode(), proof_id.encode())
                    message = {**message, "headers": [*message.get("headers", ()), header]}
                elif message["type"] == "http.response.body" and chunks is not None:
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size <= MAX_REPLY_BYTES:
                        chunks.append(chunk)
                    else:
                        chunks.clear()
                    # Store the job before the client can see the reply's end (a stream's "[DONE]"
                    # event, or else the last chunk). The shield lets the call finish if the client
                    # disconnects.
                    if chunk.startswith(b"data: [DONE]") or not message.get("more_body", False):
                        reply = b"".join(chunks) if size <= MAX_REPLY_BYTES else None
                        chunks = None
                        storing = asyncio.to_thread(
                            self.store_job, proof_id, policy, request_body, reply
                        )
                        await asyncio.shield(storing)
            except Exception as error:
                chunks = None
                self.log_unstored(policy, error)
            await send(message)

        return copy_and_send

    def log_unstored(self, policy, error):
        self.log_hourly(
            ("store", cause(error)),
            "A reply was served under policy %s, but its proof job was not stored, so its proof ID"
            " is unknown",
            policy.name,
            exc_info=error,
        )

    def store_job(self, proof_id, policy, request_body, reply_body):
        """Run submit_and_write_job() and log its failures: if the client has left, nothing awaits
        the result."""
        try:
            self.submit_and_write_job(proof_id, policy, request_body, reply_body)
        except Exception as error:
            self.log_unstored(policy, error)

    def submit_and_write_job(self, proof_id, policy, request_body, reply_body):
        job = {"policy": policy.name, "commitment_id": policy.commitment_id}
        # First the jobs directory: without it the proof could never be fetched.
        with open_jobs(self.config) as jobs:
            try:
                check_request(request_body, policy.served_model)
                if reply_body is None:
                    raise NotProvable(
                        "The reply is longer than %d MiB, so it gets no proof."
                        % (MAX_REPLY_BYTES >> 20)
                    )
                prompt, reply = served_tokens(reply_body)
                held = functools.partial(is_held, policy, jobs=jobs, batches=self.batches)
                job["request_id"], key = submit(policy, prompt, reply, held)
                job["tag_key"] = key.hex()  # never served or logged
                request = spool.whole_reply(prompt, reply, policy.max_total_tokens, fill=PADDING)
                job["request_bytes"] = request.hex()  # for the check of its batch; never served
                job["request"] = {
                    "total_tokens": policy.max_total_tokens,
                    "prompt_tokens": len(prompt),
                    "reply_tokens": len(reply),
                    "padding_token": PADDING,
                }
            except NotProvable as error:
                job["not_provable"] = str(error)
                if isinstance(error, OnHold):
                    self.log_hold(policy)
            write_file(jobs, job_file(proof_id), json.dumps(job).encode())

    def job_state(self, proof_id, delete=False, wanted=None):
        """Return what the job route reports for a proof ID, and the file 'wanted' (a name in
        DOWNLOADS) or None: y.bin as bytes, proof.bin as a TagCheck.

        The first value is None for an unknown ID; FAULTY, after logging the reason, on a fault.
        Nothing is then deleted. With delete, a job that is not waiting is then deleted with its
        response. Raise ClientError (503) if the policy is on hold."""
        try:
            with open_jobs(self.config) as jobs:
                return self.job_state_in(jobs, proof_id, delete, DOWNLOADS.get(wanted))
        except (spool.UnsafePath, OSError) as error:
            message = "A call for a proof job failed: %s"
            if isinstance(error, FileNotFoundError):
                message += ". A restart creates a missing directory, if its parent exists"
            self.log_hourly(("job", cause(error)), message, error)
            return FAULTY, None

    def job_state_in(self, jobs, proof_id, delete, wanted):
        job = read_job(jobs, proof_id)
        # A job stores its policy's name; the directory is the one that the config file gives it
        # now.
        policy = self.config.policies.get(job["policy"]) if job else None
        if policy is not None and policy.commitment_id != job.get("commitment_id"):
            # The name now belongs to another commitment, or the job records none.
            policy = None
        if job and "not_provable" in job:
            if delete:
                delete_job(jobs, proof_id)
            state = {
                "state": "not_provable",
                "policy": job["policy"],
                "detail": job["not_provable"],
            }
            return state, None
        if policy is None:
            return None, None
        state = {"state": "waiting", "policy": policy.name, "request": job["request"]}
        record = read_record(jobs, proof_id)
        with contextlib.ExitStack() as stack:
            top = stack.enter_context(open_spool(policy))
            if is_held(policy, top, jobs, self.batches):
                for note in self.batches.all_notes(jobs):  # a wipe that broke off goes on
                    if note.get("policy") == policy.name:
                        self.batches.ask(policy, note["time"])
                self.log_hold(policy)
                raise ClientError(503, ON_HOLD)
            try:
                out = stack.enter_context(spool.open_private_dir("out", top))
            except FileNotFoundError:
                out = None  # the service has not created out/ yet
            response, fds = None, {}
            if out is not None:
                response, fds = stack.enter_context(spool.open_response_at(out, job["request_id"]))
            # Asked after the files are open: the request is in work/ from before they appear until
            # they are final.
            if is_in_work(top, job["request_id"]):
                return state, None
            if (response is None or response.faults) and read_job(jobs, proof_id) is None:
                return None, None  # a concurrent DELETE is removing the response
            if record and response is None:
                state.update(state="expired", detail=EXPIRED)
                if delete:
                    delete_job(jobs, proof_id)
                    self.delete_response(out, job["request_id"])
                return state, None
            if record and not is_recorded(record, response, fds):
                # A restarted proving service can replace a response. The record is the earlier
                # one's, so this response is checked like a first one.
                with contextlib.suppress(FileNotFoundError):  # another call was faster
                    os.unlink(job_file(proof_id, ".checked"), dir_fd=jobs)
                record = None
            if response is None:
                self.batches.finish(jobs, policy, job["request_id"])
                for tick_ns in times_of(out, job["request_id"]) if out is not None else ():
                    self.batches.ask(policy, tick_ns)  # files, but no response
                return state, None
            # File names do not show that a response belongs to this job. Nothing about it is
            # reported, not even whether there is a proof, unless its tag matches.
            fault, check = self.check_response(
                jobs, proof_id, policy, job, record, out, response, fds
            )
            if fault is WAITING:
                return state, None
            if fault:
                self.log_hourly(
                    ("response", policy.name),
                    "The response %s.* is not served, and its client got status 500: %s. Nothing"
                    " was deleted. See the proving service's log",
                    os.path.join(policy.spool_dir, "out", job["request_id"]),
                    fault,
                )
                return FAULTY, None
            if check is None:
                state.update(state="no_proof", detail=DECLINED)
            else:
                # y comes from the record: the y read by this call is not what was checked.
                record = read_record(jobs, proof_id)
                if record is None:  # a withheld batch took it meanwhile
                    return state, None
                state = proved(state, record)
            if delete:
                delete_job(jobs, proof_id)
                self.delete_response(out, job["request_id"])
            if check is None or not wanted:
                return state, None
            if wanted != PROOF_SUFFIX:
                return state, bytes.fromhex(record["y"])
            check.keep_open(fds)
            return state, check

    def log_hold(self, policy):
        self.log_hourly(
            ("hold", policy.name),
            "The policy %s is on hold: the server submits no new request to the proving service,"
            " and serves no stored response; the routes of a job that was"
            " submitted answer with status 503. vLLM's replies and the published files are still"
            " served, and a job whose reply was not submitted still shows its state. If %s exists,"
            " that is a hold. %s. If that hold file does not exist, or the policy stays on hold"
            " after lift-hold, there are two more causes. One: the wipe of a rejected batch is not"
            ' done. Then the log has a line "did not pass" for this policy, or check lists a file'
            " withheld.POLICY.TIME.json whose name does not end in .done.json. A server process's"
            " memory may be all that stops the policy: before you stop or restart a server, read"
            " integrations/LIMITS.md#the-middlewares-check. Two, only if neither sign is there and"
            " that hold file does not exist: the spool directory has been replaced since this"
            " server started. The server still has the old directory open, and a hold in it counts."
            " Neither lift-hold nor check sees that hold; a restart ends it: the restarted server"
            " works in the new directory, and any stored responses there are back on offer, as"
            " after lift-hold",
            policy.name,
            hold_path(policy),
            BEFORE_REMOVING_A_HOLD,
        )

    def delete_response(self, out, request_id):
        """Delete what exists of a deleted job's response. out: the open out/, or None.

        A failure is only logged: the job is gone, so the DELETE has succeeded."""
        try:
            if out is not None:
                spool.delete_response_at(out, request_id)
        except Exception as error:
            self.log_hourly(
                ("delete", cause(error)),
                "A file of a deleted job's response was not deleted: %s: %s. It stays until the"
                " clean command deletes it",
                type(error).__name__,
                error,
            )

    def check_response(self, jobs, proof_id, policy, job, record, out, response, fds):
        """Return what is wrong with a job's response, or None, or WAITING while its batch has not
        passed; and a TagCheck at the proof's start, or None if there is no proof.

        A response with no record is not read here and nothing is said of it: the check of its
        batch compares its tag, and writes the record. With a record, stream_proof() computes the
        tag again. Where a note names this request, nothing counts but the note."""
        for note in self.batches.all_notes(jobs):
            if note_covers(note, policy, job["request_id"]):
                self.batches.ask(policy, note["time"])
                return WAITING, None
        if response.failed:
            if response.faults or not is_well_formed_failure(job, job["request_id"], fds):
                self.batches.ask(policy, response.tick_ns)
                return WAITING, None
            if self.batch_waits(jobs, policy, response.tick_ns, out):
                return WAITING, None
            return None, None
        if not record:
            self.batches.ask(policy, response.tick_ns)
            found = read_record(jobs, proof_id)
            if found is None or not is_recorded(found, response, fds):
                return WAITING, None
        if response.faults:
            return "; ".join(response.faults), None
        return None, TagCheck.start(job, response, fds)

    def batch_waits(self, jobs, policy, tick_ns, out):
        """Return whether a proof of this batch still lacks a record, after asking for the check."""
        def unchecked():  # a proof with no job can never be judged: nothing waits for it
            members = BatchChecks.members(jobs, out, policy, tick_ns, read=False)
            return any((job or members.unread) and not kept for _, job, _, kept in members.values())

        if not unchecked():
            return False
        self.batches.ask(policy, tick_ns)
        return unchecked()

    async def own_route(self, method, route, receive, send):
        job, published = JOB_ROUTE.fullmatch(route), PUBLISHED_ROUTE.fullmatch(route)
        if method == "GET" and route == "policies":
            return await send_json(send, 200, self.table)
        if method == "GET" and published:
            return await self.send_published(published, receive, send)
        if not job or method not in ("GET", "DELETE") or method == "DELETE" and job[2]:
            message = (
                "There is no such route. The routes are GET {0}policies, GET and DELETE"
                " {0}jobs/PROOF_ID, GET {0}jobs/PROOF_ID/FILE, and GET {0}published/POLICY/ with"
                " key.bin, layout.txt or leaf/NUMBER.bin. A proof ID is 64 lower-case hex digits."
            )
            raise ClientError(404, message.format(PREFIX))
        await self.job_route(method, job[1], job[2], receive, send)

    async def job_route(self, method, proof_id, file_name, receive, send):
        state, file = await asyncio.to_thread(
            self.job_state, proof_id, method == "DELETE", file_name
        )
        if state is FAULTY:
            raise ClientError(500, JOB_FAULT)
        if state is None:
            raise ClientError(
                404,
                "No proof job exists under this ID. An ID is valid from the end of its reply"
                " until DELETE, the provider's clean-up, or the end of its policy (the name"
                " dropped, or a new commitment under it). A reply that was cut off, or whose job"
                " the server failed to store, has no job.",
            )
        if method == "DELETE" and state["state"] == "waiting":
            raise ClientError(
                409,
                "The job has no answer yet, so nothing was deleted. Send DELETE again"
                " later.",
            )
        if file_name is None:
            return await send_json(send, 200, state)
        if file is None:
            raise ClientError(
                404,
                "There is no such file. GET %sjobs/PROOF_ID lists the files once the state is"
                ' "proved".' % PREFIX,
            )
        if isinstance(file, bytes):
            await send_head(send, 200, b"application/octet-stream", len(file))
            return await send({"type": "http.response.body", "body": file})
        await self.stream_proof(receive, send, file, state["policy"])

    async def stream_proof(self, receive, send, check, policy_name):
        """Send a proof file as the body of a 200 response, and close it.

        The tag is computed again from the bytes sent, and the last chunk is sent only if it
        matches. Otherwise the response stops short of its announced length, so that no client
        mistakes changed bytes for a complete proof."""
        reading, gone = None, asyncio.ensure_future(client_has_left(receive))
        try:
            await send_head(send, 200, b"application/octet-stream", check.size)
            held = b""
            while True:
                if gone.done():
                    return
                reading = asyncio.ensure_future(asyncio.to_thread(check.read))
                chunk = await asyncio.shield(reading)
                if not chunk:
                    break
                if held:
                    await send({"type": "http.response.body", "body": held, "more_body": True})
                held = chunk
            if not check.matches():
                self.log_hourly(
                    ("download", policy_name),
                    "A download under policy %s was broken off: the proof file changed after its"
                    " tag had been checked. See the proving service's log",
                    policy_name,
                )
                raise ResponseCut("a proof file changed while it was sent")
            await send({"type": "http.response.body", "body": held})
        finally:
            gone.cancel()
            close_after(reading, check.file)

    def open_published(self, policy, key_file, leaf_file):
        """Open key.bin, layout.txt or one leaf file of a policy. Return a file object; None if
        there is none; FAULTY, after logging, if it cannot be served."""
        path = policy.public_key_dir
        try:
            if key_file == "layout.txt":
                if policy.layout_file is None:
                    return None
                path = os.path.dirname(policy.layout_file)
            with spool.open_public_dir(".", policy.kept[path]) as top:
                if key_file:
                    return open(spool.open_public_file(key_file, top), "rb", buffering=0)
                try:
                    with spool.open_public_dir("leaf", top) as leaf:
                        return open(spool.open_public_file(leaf_file, leaf), "rb", buffering=0)
                except FileNotFoundError:
                    # leaf/ or the leaf file need not exist, and anyone can ask for any number.
                    return None
        except (spool.UnsafePath, OSError) as error:
            self.log_hourly(
                ("published", cause(error)),
                "A published file of policy %s cannot be served: %s",
                policy.name,
                error,
            )
            return None if isinstance(error, FileNotFoundError) else FAULTY

    async def send_published(self, match, receive, send):
        """Serve key.bin, layout.txt or one leaf file. They are public, so no proof ID is needed;
        PUBLISHED_ROUTE allows no other name. Raise ClientError with status 404 if this server
        publishes no such file, and with status 500 if the file cannot be served."""
        policy, file = self.config.policies.get(match[1]), None
        if policy is not None:
            # A file object, not a bare descriptor: it closes the file if this call is cancelled
            # while the thread runs.
            file = await asyncio.to_thread(self.open_published, policy, match[2], match[3])
        if file is FAULTY:
            raise ClientError(500, SERVER_FAULT)
        if file is None:
            raise ClientError(
                404,
                "This server has no such published file. Check the policy's name (GET %spolicies"
                " lists the names) and the file's name. A provider may also publish key.bin,"
                " layout.txt and leaf/NUMBER.bin elsewhere." % PREFIX,
            )
        await send_file(receive, send, file)


def is_in_work(top, request_id):
    """Return whether cp_prove still has a request in work/. Until it leaves, the response is not
    final: the service's next start withdraws it and responds "failed"."""
    try:
        os.stat("work/%s.req" % request_id, dir_fd=top, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def times_of(out, request_id):
    """Return the modification times of the response files of a request that are there."""
    found = set()
    for suffix in (PROOF_SUFFIX, "y", spool.TAG, spool.FAILED):
        with contextlib.suppress(FileNotFoundError):
            name = "%s.%s" % (request_id, suffix)
            found.add(os.stat(name, dir_fd=out, follow_symlinks=False).st_mtime_ns)
    return found


def is_not_final(top, request_id):
    """Return whether a request is still in in/ or work/: the service's next start can take its
    response back file by file."""
    for folder in ("in", "work"):
        with contextlib.suppress(FileNotFoundError):
            os.stat("%s/%s.req" % (folder, request_id), dir_fd=top, follow_symlinks=False)
            return True
    return False


def is_recorded(record, response, fds):
    """Return whether an open response is the one that the record describes."""
    if response is None or response.faults or response.failed:
        return False
    return identity(fds[PROOF_SUFFIX]) == record["proof"]


def proved(state, record):
    """Return a job's state once its batch has passed."""
    sizes = dict(zip(DOWNLOADS, (len(record["y"]) // 2, record["proof"][0])))
    return {**state, "state": "proved", "files": sizes}


def cause(error):
    """Return a key for log_hourly(): the error's text without file names, so that clients cannot
    multiply the lines by asking for many files."""
    if isinstance(error, (spool.UnsafePath, OSError)):
        return re.sub(r"[0-9a-f]{64}|[0-9]+\.bin", "*", str(error))
    return type(error).__name__


async def read_body(receive):
    """Return the whole body of an HTTP request, or None if the client has left. Raise ClientError
    if the body is too long.

    Follows read_body in uvicorn's documentation (BSD-3-Clause)."""
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return None
        body += message.get("body", b"")
        if len(body) > MAX_REQUEST_BYTES:
            raise ClientError(
                413,
                "The body of a request that asks for a proof may have at most %d MiB."
                % (MAX_REQUEST_BYTES >> 20),
            )
        if not message.get("more_body", False):
            return bytes(body)


async def client_has_left(receive):
    """Return when the web server reports a disconnect, or receive() fails.

    A web server may silently drop what is sent to such a client, so a sender of long files runs
    this concurrently and stops reading once it returns."""
    with contextlib.suppress(Exception):
        while (await receive())["type"] != "http.disconnect":
            pass


async def send_file(receive, send, file):
    """Send an open file as the body of a 200 response, and close it.

    At most the announced length is sent. If the file has become shorter, an OSError breaks the
    response, so that it does not look complete."""
    reading, gone = None, asyncio.ensure_future(client_has_left(receive))
    try:
        left = os.fstat(file.fileno()).st_size
        await send_head(send, 200, b"application/octet-stream", left)
        while left:
            if gone.done():
                return
            reading = asyncio.ensure_future(asyncio.to_thread(file.read, min(CHUNK, left)))
            chunk = await asyncio.shield(reading)
            if not chunk:
                raise OSError("a published file became shorter while it was sent")
            left -= len(chunk)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})
    finally:
        gone.cancel()
        close_after(reading, file)


def close_after(reading, file):
    """Close a file, but not while a thread still reads it after a cancelled call ('reading' is not
    done): the descriptor's number could be reused, and the thread would read another file."""

    def close(_=None):
        # The file was only read, so a failure to close it must not break a complete response.
        with contextlib.suppress(OSError):
            file.close()

    if reading is not None and not reading.done():
        reading.add_done_callback(close)
    else:
        close()


async def send_head(send, status, content_type, length):
    headers = [
        (b"content-type", content_type),
        (b"content-length", b"%d" % length),
        (b"cache-control", b"no-store"),
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})


async def send_json(send, status, value):
    data = json.dumps(value).encode()
    await send_head(send, status, b"application/json", len(data))
    await send({"type": "http.response.body", "body": data})


async def send_error(send, status, message):
    """Send an error in the form of vLLM's error responses (Apache-2.0)."""
    kind = {
        400: "BadRequestError",
        401: "AuthenticationError",
        404: "NotFoundError",
        409: "ConflictError",
        413: "RequestEntityTooLargeError",
        500: "InternalServerError",
        503: "ServiceUnavailableError",
    }[status]
    await send_json(
        send, status, {"error": {"message": message, "type": kind, "param": None, "code": status}}
    )


# ---- The provider's commands ----

EXAMPLE_CONFIG = """a minimal config file (all paths absolute):
  {"jobs_dir": "/PATH/TO/jobs",
   "policies": {"exact": {"served_model": "MODEL",
                          "commitment_id": "64 HEX DIGITS",
                          "parameter_file": "/PATH/TO/policy.txt",
                          "spool_dir": "/PATH/TO/spool",
                          "public_key_dir": "/PATH/TO/ivk",
                          "verifier": "/PATH/TO/cp_verify"}}}
"jobs_dir" and each "spool_dir": name a new or empty directory whose parent exists. The init
  command, or the server at its start, creates each directory and its mark file, mark.txt. Run
  init before cp_prove first starts: cp_prove does not create its spool directory.
"public_key_dir": the directory with key.bin and leaf/, which are served to clients, as is
  layout.txt from next to the parameter file. KL.bin lies next to the parameter file too.
"verifier": cp_verify; keep it where the service's account cannot change it. Every proof of a
  batch passes it before any response of that batch is handed on. If one is rejected, the batch
  is answered as failed, and the policy goes on hold.
optional keys of a policy:
  "max_waiting" (default %d): requests that may wait for the prover before replies get no proof
  "max_stored" (default %d): responses that may be stored in the spool directory before replies
    get no proof. A client's DELETE and the clean command make room.
  "closed": true for a policy that takes no new requests; its proofs are still served
  "checks_at_once": calls of cp_verify that run side by side, one thread each, beside cp_prove on
    the same machine (default: a quarter of the CPUs that this process may use, from 1 to 6)
to put a policy on hold: create the empty file "hold" in its spool directory. While it exists,
  replies get no proof, no stored response is served (503), and clean deletes nothing. It does
  not stop cp_prove. Create it whenever cp_prove exits with a code other than 0 (by hand: the
  hold command). Of these commands only lift-hold removes it. That puts every stored response
  back on offer, faulty ones too, each behind the check of its batch, which a file that has lost a
  mask can pass: read integrations/vllm/REFERENCE.md first.
""" % (DEFAULT_MAX_WAITING, DEFAULT_MAX_STORED)


def describe(config):
    def shown(path):
        return path if os.path.isdir(path) else path + " (missing: init or the server creates it)"

    lines = []
    for policy in config.policies.values():
        waiting, stored, on_hold = 0, 0, False
        with contextlib.suppress(FileNotFoundError), open_spool(policy) as top:
            (waiting, stored), on_hold = waiting_and_stored(top), is_on_hold(policy, top)
        lines.append(
            "policy %s%s: %s (%s); served model %s; at most %d tokens in a prompt and its reply;"
            " commitment ID %s; spool directory %s; requests waiting: %d (max_waiting %d);"
            " responses stored: %d (max_stored %d); key.bin and leaf files %s; layout.txt %s"
            % (
                policy.name,
                " (closed)" * policy.closed + " (ON HOLD: %s exists)" % hold_path(policy) * on_hold,
                policy.kind,
                ", ".join("%s %d" % limit for limit in policy.limits.items()),
                policy.served_model,
                policy.max_total_tokens,
                policy.commitment_id,
                shown(policy.spool_dir),
                waiting,
                policy.max_waiting,
                stored,
                policy.max_stored,
                "served from %s" % policy.public_key_dir,
                "served from %s" % policy.layout_file if policy.layout_file else "not served",
            )
        )
    lines.append("jobs directory: %s" % shown(config.jobs_dir))
    lines.append(
        "The config file is valid, and each verifier gives its public files the config's commitment"
        " ID. Not checked: that a proving service runs on each spool directory."
    )
    return lines + withheld_lines(config)


def withheld_lines(config):
    """Return a line for each batch that the check of a batch withheld."""
    with contextlib.suppress(FileNotFoundError), open_jobs(config, allow_empty=True) as jobs:
        return [
            "A batch was withheld: %s in the jobs directory names its requests, and the log says"
            ' why: see docs/faults.md, "Always first".' % name
            for name in sorted(filter(WITHHELD_FILE.fullmatch, os.listdir(jobs)))
        ]
    return []


def mute(name):
    """Give up sys.stdout or sys.stderr, which cannot be written: otherwise Python exits with 120
    for the text that is left in it."""
    setattr(sys, name, None)


def say(text):
    """Print on standard output or, if that cannot be written, in ASCII on standard error; else stay
    silent."""
    for name in ("stdout", "stderr"):
        try:
            if getattr(sys, name) is not None:
                print(text, file=getattr(sys, name), flush=True)
                return
        except (OSError, ValueError):
            mute(name)
        text = text.encode("ascii", "backslashreplace").decode()


def hold(policy):
    """The hold command: print what was done, and return the exit code.

    Once the hold is there, no later fault may make the command look as if it had failed."""
    path, made, there, warning = hold_path(policy), False, False, ""
    try:
        with open_spool(policy, mark=True) as top:
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                created = os.open(spool.HOLD_FILE, flags, 0o600, dir_fd=top)
                made = True
            except FileExistsError:
                pass
            except OSError as error:
                raise OSError(error.errno, error.strerror, path)
            there = True
            if made:
                os.close(created)
                os.fsync(top)
    except OSError as error:
        if not there:
            raise
        warning = (
            "\nwarning: closing or syncing it failed (%s), so a loss of power may undo it"
            % error.strerror
        )
    if made:
        say("Created %s: the policy %s is on hold.%s" % (path, policy.name, warning))
    else:
        say("The policy %s is already on hold (%s exists)." % (policy.name, path))
    return 0


def lift_hold(policy):
    """The lift-hold command: print what it does, do it, and return the exit code.

    Everything that can fail comes before the removal, the printing too: a hold must not go without
    a word."""
    path, removed = hold_path(policy), False
    try:
        with open_spool(policy) as top:
            words = (
                "Removing %s: the policy %s goes off hold. Its stored responses (now %d) are on"
                " offer again, also those of a cp_prove that exited badly.\nThat is %s What clients"
                " fetch meanwhile cannot be taken back."
            ) % (
                path,
                policy.name,
                waiting_and_stored(top)[1],
                WHEN_LIFTING_IS_SAFE % "run the hold command now",
            )
            found = os.stat(spool.HOLD_FILE, dir_fd=top, follow_symlinks=False)
            if not stat.S_ISREG(found.st_mode) or found.st_size:
                print(
                    "error: %s is not an empty regular file, which is all that these commands"
                    " create, so it was not removed: find out what put it there" % path,
                    file=sys.stderr,
                )
                return 1
            try:
                if sys.stdout is None:
                    raise ValueError("it is closed")
                print(words, flush=True)
            except (OSError, ValueError) as error:
                mute("stdout")
                if isinstance(error, UnicodeEncodeError):
                    error = (
                        "its encoding cannot show the path: set"
                        " PYTHONIOENCODING=utf-8:backslashreplace"
                    )
                raise ValueError("lift-hold cannot write to its standard output (%s)" % error)
            try:
                os.unlink(spool.HOLD_FILE, dir_fd=top)
            except OSError as error:
                raise OSError(error.errno, error.strerror, path)
            removed = True
    except OSError as error:
        if isinstance(error, FileNotFoundError) and os.path.isdir(policy.spool_dir):
            print("error: the policy %s is not on hold" % policy.name, file=sys.stderr)
            return 1
        if not removed:  # else only the closing of the directory failed
            raise
    return 0


def whole_days(text):
    if not re.fullmatch("[1-9][0-9]{0,4}", text):
        raise argparse.ArgumentTypeError("must be a whole number from 1 to 99999")
    return int(text)


def main(argv=None):
    """Run a provider's command and return its exit code (see the epilog). On a usage error argparse
    exits itself."""
    parser = argparse.ArgumentParser(
        prog="proof_middleware.py",
        allow_abbrev=False,
        description=(
            "Provider's commands for the proof middleware. Run them as the user that runs the vLLM"
            " server: the jobs directory and the spool directories belong only to that user."
        ),
        epilog=(
            "exit codes: 0 done; 1 a policy is on hold (check, clean), clean did not delete all"
            " that it should, or lift-hold found no hold that it may remove; 2 a usage error, or"
            " the command could not do its work: the message says why; 120 (Python's own) the"
            " command's output could not be written"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    check = commands.add_parser(
        "check",
        help="validate a config file and print a summary",
        epilog=EXAMPLE_CONFIG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Validate a config file as the server does at startup, and print a summary, with\n"
            "the numbers of waiting requests and of stored responses for each policy.\n"
            "Nothing is created."
        ),
    )
    init = commands.add_parser(
        "init",
        help="create the jobs directory and the spool directories",
        description=(
            "Create what is missing of the jobs directory, the spool directories and their mark"
            " files (mark.txt), as the server does at its start. The parent of each directory must"
            " exist. Run it before cp_prove is first started on a spool directory: cp_prove does"
            " not create that directory, and this command writes a mark file only into an empty"
            " one."
        ),
    )
    cleaner = commands.add_parser(
        "clean",
        help="delete old jobs, requests and responses",
        description=(
            "Delete job files, waiting requests, and the prover's responses, fetched or not, that"
            " are older than DAYS days (requests: an hour more). Only these files are deleted: in"
            " the jobs directory, files named 64 HEX DIGITS.json or .checked whose content is a job"
            " or a record of the middleware, and notes withheld.POLICY.TIME.done.json; in each"
            " spool directory, files named 64 HEX DIGITS.req"
            " in in/, 64 HEX DIGITS in in/.tmp/, and 64 HEX DIGITS.y, .proof, .tag or .failed in"
            " out/. Nothing is deleted in a jobs or spool directory that has no mark file"
            " (mark.txt). A job that is still waiting at that age is deleted too, unless its"
            " response or its request stays. Not touched:"
            " requests that cp_prove has taken (in work/), and published files. Nothing is deleted"
            ' while a policy is on hold (the file "hold" in its spool directory): clean looks'
            " before each file, and stops there."
        ),
    )
    holder = commands.add_parser(
        "hold",
        help="put a policy on hold",
        description=(
            'Create the empty file "hold" in a policy\'s spool directory. While it exists, replies'
            " under that policy get no proof, no stored response is served (503), and clean deletes"
            " nothing. It does not stop cp_prove. Run it if cp_prove has exited with a code other"
            " than 0 and nothing has created the file."
        ),
    )
    lifter = commands.add_parser(
        "lift-hold",
        help="take a policy off hold",
        description=(
            'Remove the file "hold" from a policy\'s spool directory, if it is an empty regular'
            " file. That puts every stored response of the policy back on offer, also those of a"
            " cp_prove that exited badly. It is %s Nothing runs this command by itself."
            % (WHEN_LIFTING_IS_SAFE % "keep the hold")
        ),
    )
    for command in (check, init, cleaner, holder, lifter):
        command.add_argument("config", metavar="CONFIG", help="the middleware's config file")
    cleaner.add_argument("days", metavar="DAYS", type=whole_days, help="the age in days, 1 or more")
    for command in (holder, lifter):
        command.add_argument("policy", metavar="POLICY", help="the policy's name in CONFIG")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s")
    undone = {
        "hold": (
            ". NOTHING WAS PUT ON HOLD. Mend that and run this command again. If you can't, stop"
            " every vLLM server that serves this policy, and start none before this command has"
            " put the policy on hold (what a stopped server remembered of a rejected batch is lost:"
            " before that hold is lifted, see integrations/LIMITS.md#the-middlewares-check)"
        ),
        "lift-hold": ". Nothing was removed",
    }.get(args.command, "")
    try:
        config = load_config(
            args.config,
            create_dirs=args.command == "init",
            denied=(
                " (if you are not the account that runs the vLLM server, run this as that account:"
                " sudo -u ACCOUNT ...)"
            ),
        )
        if args.command in ("check", "init"):
            check_verifiers(config)
    except ConfigError as error:
        print("error: %s%s" % (error, undone), file=sys.stderr)
        return 2
    if args.command in ("hold", "lift-hold"):
        if args.policy not in config.policies:
            print(
                "error: %s has no policy %r (it has: %s)%s"
                % (args.config, args.policy, ", ".join(config.policies), undone),
                file=sys.stderr,
            )
            return 2
        try:
            if args.command == "lift-hold":
                for line in withheld_lines(config):
                    say(line)
            return (hold if args.command == "hold" else lift_hold)(config.policies[args.policy])
        except FileNotFoundError:
            error = "the policy's spool directory does not exist: run init"
        except (spool.UnsafePath, OSError, ValueError) as fault:  # ValueError: it cannot be printed
            error = str(fault).encode("ascii", "backslashreplace").decode()
        try:
            print("error: %s%s" % (error, undone), file=sys.stderr, flush=True)
        except (OSError, ValueError):
            mute("stderr")
        return 2
    if args.command != "clean":
        try:
            print("\n".join(describe(config)))
            for policy in policies_on_hold(config) if args.command == "check" else ():
                print("error: %s." % hold_text(policy), file=sys.stderr)
                return 1
        except (spool.UnsafePath, OSError) as error:  # a directory has changed since load_config()
            print("error: %s" % error, file=sys.stderr)
            return 2
        return 0

    def counted(number, noun):
        return "%d %s%s" % (number, noun, "" if number == 1 else "s")

    try:
        jobs, requests, responses, failed = clean(config, args.days)
        print(
            "Deleted %s, %s and %s older than %s."
            % (
                counted(jobs, "job"),
                counted(requests, "request"),
                counted(responses, "response"),
                counted(args.days, "day"),
            )
        )
        if failed:
            print(
                "error: %s could not be deleted: see above" % counted(failed, "file"),
                file=sys.stderr,
            )
            return 1
    except OnHold as error:
        print("error: %s." % error, file=sys.stderr)
        return 1
    except (spool.UnsafePath, OSError) as error:
        print(
            "error: clean stopped early; some files may already be deleted: %s" % error,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
