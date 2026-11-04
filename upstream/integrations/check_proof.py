#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Check the proof for one LLM reply against values that you state yourself.

The verifier program cp_verify checks the proof. The script checks that the proof is about your
request: your commitment ID, checked range, end token and acceptance policy. A check exits with 0
only if everything passes; see --help.

Take the commitment ID from a source that you trust, not from the server that sent the proof, and
build cp_verify yourself: anyone who can write to this script's directory or to that program
controls the result.

cp_verify runs in a private temporary directory, so that no other local user can swap a file between
the script's read and the verifier's. The configuration files, the request and the proof are copied
there, and fetched files are written there; cp_verify gets relative names for them. Your own key.bin
and leaf file are not copied: the script does not read them. TMPDIR must be on a local file system,
where a read returns what was just written.
"""

# ruff: noqa: E402 (the other imports must come after isolate())
import posix  # built in: under Python 3.10 with -S, a file on PYTHONPATH could stand in for os
import sys


def isolate():
    """Start the program again in Python's isolated mode (-I), unless it already runs in it.

    PYTHON* variables, the user's site-packages and this script's directory then do not decide what
    is imported below. Not covered: what Python itself takes from those three while it first starts
    (for example a sitecustomize.py on PYTHONPATH), which only "python3 -I" on the command line
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
        print("result: outcome=error")
        text = "check_proof: error: cannot start Python (%s) in isolated mode: %s"
        print(text % (sys.executable, reason))
        sys.exit(2)  # EXIT_ERROR


if __name__ == "__main__":
    isolate()

import argparse
import contextlib
import errno
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import urllib.error
import urllib.request

EXIT_VERIFIED = 0
EXIT_REJECTED = 1  # also: no proof, or files not as stated
EXIT_ERROR = 2
EXIT_NOT_A_VERIFICATION = 14  # as in cp_verify

# The words after "check_proof: " in the last line. A pass under the caller's clipped-gap limits has
# the plain yes's exit code but its own words: it bounds an average and a count of gaps, not each
# token.
WITHIN_LIMITS_WORDS = "within limits"
LAST_LINE = {
    EXIT_VERIFIED: "verified",
    EXIT_REJECTED: "rejected",
    EXIT_ERROR: "error",
    EXIT_NOT_A_VERIFICATION: "not a verification",
}

# outcome= in the "result:" line, for programs, which should not parse the last line: see EPILOG.
OUTCOMES = {
    EXIT_VERIFIED: "verified",
    EXIT_REJECTED: "rejected",
    EXIT_ERROR: "error",
    EXIT_NOT_A_VERIFICATION: "not-a-verification",
}
WITHIN_LIMITS_OUTCOME = "within-limits"
NO_PROOF_OUTCOME = "no-proof"  # the proving service made no proof
NO_PASS = "result: outcome=%s"  # the whole line unless the check passes

REPOSITORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
VERIFIER = os.path.join(REPOSITORY, "bin", "cp_verify")
HOW_TO_BUILD = "to build it, run verifier/build.sh in the repository's root"

COMMITMENT_ID = "[0-9a-f]{64}"
# Environment variable: a bearer key for fetches from a URL. Its name is specific to this script, so
# that no other program's key goes to that server.
FETCH_KEY = "CHECK_PROOF_FETCH_KEY"
API_KEYS = (FETCH_KEY, "VLLM_API_KEY")  # not handed on to cp_verify
# re.A: with only re.I, the letter U+017F would match "s".
URL, HTTPS = re.compile("https?://", re.I | re.A), re.compile("https://", re.I | re.A)
THIS_MACHINE = re.compile(
    r"https?://(127\.0\.0\.1|\[::1\])(:[0-9]{1,5})?(/[!-\[\]-~]*)?", re.I | re.A
)
POLICY_LIMITS = ("C", "theta", "cap", "s")
POLICY_LINES = POLICY_LIMITS + ("sh", "sT", "F")
SH, F = 8, 24  # this script supports no other values
OFFSET = 1 << SH  # the cost of a token whose gap is 0
GREEDY = "the model's greedy choice"  # the token whose gap is 0
# Lines of the parameter file (`policy.txt`) that hold a name and then names or several numbers.
LIST_LINES = ("call", "callA", "callH", "group", "stages", "table", "work")
PROOF_HEADER = b"CPPF"  # then the parameter file's version word and the leaf number
# Passed to cp_verify as -bits: the target of its own error accounting, which it does not call a
# security level.
BITS = 128
MAX_SMALL_FILE_BYTES = 16 << 20  # a configuration file or a request
MAX_FETCHED_BYTES = 256 << 20  # a fetched key.bin or leaf file; the proof

# cp_verify's exit code and last line show that a run passed, not what it was about. That is in the
# "result-v1" line, directly before the last. The whole line is compared: leaf= with the proof's
# header, commitment-assumed-rounds= and the fields on proof of work with the two lines before it,
# most others with the caller's values, request and parameter file. record=1 and stages= are fixed
# texts, and stages= does not show what ran. Of build and, if no work is counted, least-round and
# least-round-without-work only the form is checked.
RESULT_V1_START = "result-v1 "
RESULT_V1 = (
    "result-v1 status=%d claim=%s id=%s bits=%d work=%s work-rounds=%s "
    "least-round=%s least-round-without-work=%s commitment-assumed-rounds=%s record=1 "
    "r=%d e=%d T=%d tokens=%s y=%s policy=%s C=%s theta=%s cap=%s s=%s sh=%s sT=%s F=%s "
    "above-clip=%d leaf=%d stages=key,leaf,request dw=8 build=([0-9a-f]{64})"
)
YES = "YES"
# Directly before "result-v1" come two lines: one on the commitment's one-time files (key.bin and
# the leaf file), one on proof of work. The first must match ON_THE_COMMITMENT, with at least one
# round, and end with a clause that begins with RECORD_CLAUSE, and the parameter file must say
# "record 1".
ON_THE_COMMITMENT = (
    "commitment: (1 closeness round of the one-time files rests|"
    "(?:[2-9]|[1-9][0-9]+) closeness rounds of the one-time files rest) "
    "on another source than the bound within a third of the distance"
)
# What follows RECORD_CLAUSE says what the weights' binding rests on. Those words are
# cp_verify's: the script's last line repeats them as printed, and adds none of its own. They must
# be visible ASCII characters with single blanks between them, few enough for the last line.
RECORD_CLAUSE = "; the table has a record: "
RECORD_WORDS, MAX_RECORD_WORDS = "[!-~]+(?: [!-~]+)*", 180
NO_WORK = "proof of work: none counted"
# The script's last line ends with cp_verify's sentences on the commitment and on proof of work.
# Where work is counted, the second has both figures; where none is, the script adds the
# "result-v1" line's least-round-without-work=.
ONE_FIGURE = (
    "the least round has %s bits (cp_verify's own figure, not a security level; --show-all has its "
    "caveats)"
)
BOTH_FIGURES = "(cp_verify's own figures, not a security level; --show-all has its caveats)"
WORK = (
    "proof of work: up to ([1-9][0-9]*) bits counted in (1 round|[1-9][0-9]* rounds); "
    r"with it the least round has ([0-9]+\.[0-9][0-9]) bits, without it ([0-9]+\.[0-9][0-9]) bits"
)
WITHIN_LIMITS = "WITHIN LIMITS: clipped-gap C=%d theta=%d cap=%d s=%d sh=%d"
STATUS_WITHIN_LIMITS = 0  # as for a plain yes
NO = r"NO: \S.*"
NOT_A_VERIFICATION = "NOT-A-VERIFICATION"
TEST_BUILD = "TEST BUILD:"  # first line of a test build, whose runs never count
ID_FIELD = "commitment ID: "  # from "-plan id"
# These only decide what is shown (`covered:` lists the checked tokens).
CLOSING_HEAD = "---- the call ----"
# Left out unless --show-all is given: cp_verify's account of its error figures, of what they rest
# on and of what is not proved; and the line for programs, most of which is in the "result:" line.
LEFT_OUT = ("accounting:", "result-v1 ")
LEFT_OUT_NOTE = (
    "(%d more line%s not shown: the verifier's own account of what its figures rest on, with its "
    "caveats%s; to see all, pass --show-all)"
)
WITH_THE_FILES_NUMBERS = ("covered:", "result-v1 ", "WITHIN LIMITS:")  # shown only at a pass
# Endings of cp_verify, other than a pass, whose closing lines are shown: exit code and last line.
SHOWN_ENDINGS = {
    1: NO,
    2: r"input-unreadable-\S.*",
    5: r"NOT CHECKED: \S.*",
    14: NOT_A_VERIFICATION,
}
LEAF_FILE_REJECTED = "refused by the stage leaf(: .*)?"  # adds a hint to a rejection

PARAMETER_COPY = os.path.join("config", "policy.txt")
SCALES_COPY = os.path.join("config", "KL.bin")
LAYOUT_COPY = os.path.join("config", "layout.txt")
KEY_COPY = os.path.join("ivk", "key.bin")
PROOF_COPY = "proof.bin"


REJECTED_LEAF_FILE = ": it rejected the leaf file"
# A verifier that rejects may print the commitment ID that it computed from the files, and a caller
# who pastes it into --commitment-id accepts those files. So HIDDEN_ID replaces every run of 64 or
# more hex digits, except the caller's ID, in what cp_verify prints unless the script accepts its
# pass.
HIDDEN_ID = "(64 hex digits, not shown)"


class CheckFailed(Exception):
    """The check did not pass. outcome replaces OUTCOMES[code] in the "result:" line."""

    def __init__(self, code, text, outcome=None):
        super().__init__(text)
        self.code, self.text, self.outcome = code, text, outcome


class Stopped(BaseException):
    """A signal asked the script to stop. Not an Exception, so that no handler swallows it."""


# True once the check is ending or a signal has stopped it: stop() then raises nothing, so that the
# clean-up and the last line are not interrupted.
ENDING = [False]


def rejected(text):
    return CheckFailed(EXIT_REJECTED, text)


def error(text):
    return CheckFailed(EXIT_ERROR, text)


def not_a_verification(text):
    return CheckFailed(EXIT_NOT_A_VERIFICATION, text)


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


def one_line(text, limit=300):
    """Return text as one line that cannot easily imitate the script's other output.

    Visible characters can still push words to the start of a terminal row, so read the verdict only
    from the exit code and the last line. INVISIBLE holds characters that str.isprintable() accepts
    but that look like blanks.
    """
    text = str(text)
    head = text if limit is None else text[: 4 * limit]
    escaped = (c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in head)
    line = re.sub(" {2,}", " ", "".join(escaped)).strip()
    if limit is not None and (len(line) > limit or len(head) < len(text)):
        line = line[:limit] + " ... (cut)"
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    return line.encode(encoding, "backslashreplace").decode(encoding)


# ---- The parameter file and its acceptance policy ----


class PolicyError(Exception):
    """The parameter file is not in parse_policy()'s strict form."""


def parse_policy(data):
    """Return the "name number" lines of a parameter file (bytes) as a dict of strings.

    cp_verify parses the same bytes. Raise PolicyError unless the file is in a form that no two
    parsers read differently: printable ASCII, lines ending in a newline, single spaces, no repeated
    name, plain decimals below 2**32. The line version holds a word of four letters or digits.
    """
    if not data.endswith(b"\n"):
        raise PolicyError("it is empty or does not end with a newline")
    values, lists = {}, set()
    for number, line in enumerate(data[:-1].split(b"\n"), 1):
        if not all(0x20 <= byte <= 0x7E for byte in line):
            raise PolicyError("line %d has a byte that is not printable ASCII" % number)
        words = line.decode("ascii").split(" ")
        if "" in words:
            raise PolicyError("line %d is empty or has extra spaces" % number)
        if words[0] in LIST_LINES:
            if len(words) < 3 or tuple(words[:2]) in lists:
                raise PolicyError("line %d is a malformed or repeated list line" % number)
            lists.add(tuple(words[:2]))
        elif words[0] in values:
            raise PolicyError("line %d repeats an earlier name" % number)
        elif words[0] == "version":
            if len(words) != 2 or not re.fullmatch("[0-9A-Za-z]{4}", words[1]):
                raise PolicyError(
                    "line %d is not a version word of four letters or digits" % number
                )
            values["version"] = words[1]
        elif len(words) != 2 or not re.fullmatch("0|[1-9][0-9]{0,9}", words[1]):
            raise PolicyError("line %d is not a name and a plain decimal number" % number)
        elif int(words[1]) >= 2**32:
            raise PolicyError("line %d has a number that is not below 2**32" % number)
        else:
            values[words[0]] = words[1]
    return values


def is_exact_match(values):
    """Return whether C, theta, cap and sT are those of exact match: every checked gap is 0."""
    limits = [int(values[name]) for name in ("C", "theta", "cap", "sT")]
    return limits == [OFFSET, OFFSET, 0, SH - 4]


def policy_fault(values):
    """Return why this script does not support a parameter file's policy lines, or None."""
    missing = [name for name in POLICY_LINES if name not in values]
    if "temp" in values:
        return 'has a line "temp"'
    if missing:
        return "has no line %s" % missing[0]
    if [int(values[name]) for name in ("sh", "sT", "F")] != [SH, SH - 4, F]:
        return "has sh, sT or F other than %d, %d and %d" % (SH, SH - 4, F)
    return None


def policy_kind(values):
    """Return "exact" or "clipped-gap", or None if policy_fault() finds a fault."""
    if policy_fault(values):
        return None
    return "exact" if is_exact_match(values) else "clipped-gap"


def policy_numbers(values):
    return tuple(int(values[name]) for name in ("C", "theta", "cap", "s", "sh"))


def above_clip_bound(checked, values):
    """Return cp_verify's bound on how many of the checked tokens can cost more than C."""
    clip, mean, cap = (int(values[name]) for name in ("C", "theta", "cap"))
    bounds = [cap, checked]
    if clip > OFFSET:
        bounds.append(checked * max(mean - OFFSET, 0) // (clip - OFFSET))
    return min(bounds)


def describe_policy(values, checked, tokens):
    """Return what a pass shows about the checked tokens under this policy."""
    if is_exact_match(values):
        return "exact match: every checked token (%s) is %s" % (tokens, GREEDY)
    # Proved: the values min(gap + OFFSET, C) add up to at most theta * checked, and at most cap
    # tokens have gap + OFFSET > C.
    clip = max(int(values["C"]) - OFFSET, 0)
    total = max(int(values["theta"]) - OFFSET, 0) * checked
    above = above_clip_bound(checked, values)
    if not clip:
        return (
            "clipped gap: up to %d of the checked tokens (%s) can be any token; the others are %s"
            % (above, tokens, GREEDY)
        )
    text = (
        "clipped gap: the logit gaps of the checked tokens (%s), each clipped to %d, add up to at "
        "most %d, in the integer model's units (C and theta count costs: a cost is a gap plus %d); "
        % (tokens, clip, total, OFFSET)
    )
    if not above:
        return text + "no gap can be above the clip"
    if above == 1:
        return text + "at most one gap can be above the clip, and such a token can be any token"
    return text + "up to %d gaps can be above the clip, and such a token can be any token" % above


# ---- Files ----


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the caller gets an HTTPError.

    The handlers are overridden, not redirect_request(): they parse the Location header first and
    can raise an error with the server's text.
    """

    def http_error_302(self, *_):
        return None

    # The chained assignment is as in CPython's HTTPRedirectHandler (PSF-2.0).
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def is_this_machine(url):
    """Return whether the host in a URL is the address 127.0.0.1 or [::1], with an optional port.

    Not the name localhost: it can resolve to either, and another local user may listen on the one
    that the server does not use. Not a user name, second colon or escape: URL parsers disagree on
    where the host then ends.
    """
    return THIS_MACHINE.fullmatch(url) is not None


def fetch(source, name, max_bytes, destination):
    """Copy source/name (a URL or a directory) to the new file destination.

    Raise OSError if it cannot be read or has more than max_bytes.
    """
    if URL.match(source):
        # An error's text can contain the server's words: pass on only the status code, the
        # operating system's message or the exception's type.
        handlers = [NoRedirects]
        if os.environ.get(FETCH_KEY) and not HTTPS.match(source):
            if not is_this_machine(source):
                raise OSError(
                    "$%s is set, and plain http may go only to 127.0.0.1 or [::1]" % FETCH_KEY
                )
            # Connect directly: a proxy from the environment would see the key. With https a proxy
            # sees only a tunnel.
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        if os.environ.get(FETCH_KEY):
            opener.addheaders.append(("Authorization", "Bearer " + os.environ[FETCH_KEY]))
        try:
            with opener.open(source.rstrip("/") + "/" + name, timeout=120) as reply:
                data = reply.read(max_bytes + 1)
        except urllib.error.HTTPError as failure:
            raise OSError("HTTP status %d" % failure.code) from None
        except Exception as failure:
            reason = getattr(failure, "reason", None)
            inner = reason if isinstance(reason, Exception) else failure
            known = isinstance(inner, OSError) and inner.errno
            raise OSError(inner.strerror if known else type(inner).__name__) from None
    else:
        if not os.path.isfile(os.path.join(source, name)):
            raise OSError("no such regular file")
        with open(os.path.join(source, name), "rb") as f:
            data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise OSError("longer than the %d bytes that it can have" % max_bytes)
    with open(destination, "xb") as f:
        f.write(data)


def copy_local(path, max_bytes, destination):
    """Copy path to the new file destination.

    Raise CheckFailed if path cannot be read or exceeds max_bytes, so that no file fills TMPDIR.
    """
    try:
        fetch(*os.path.split(path), max_bytes, destination)
    except OSError as failure:
        raise error("%s: %s" % (path, one_line(failure.strerror or failure)))


def link_new_file(path, directory, name):
    """Copy path to the new file 'name' in the open directory (a descriptor), whole or not at all.

    Raise OSError if another user replaced the part file ('name' may be theirs) or 'name' exists.
    """
    part = "%s.%s.part" % (name, os.urandom(8).hex())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with open(os.open(part, flags, 0o600, dir_fd=directory), "wb") as to:
        try:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, to)
            to.flush()
            os.fchmod(to.fileno(), 0o644)  # a published file
            # Unlike rename, link fails if the name exists.
            os.link(part, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            if not os.path.samestat(
                os.stat(name, dir_fd=directory, follow_symlinks=False), os.fstat(to.fileno())
            ):
                raise OSError("another user has put something else under that name")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(part, dir_fd=directory)


# Said only of a directory that is the caller's own, and never to root: see cure().
OWN_DIRECTORY = ": if it is yours, run chmod go-w on it, and then look at what is in it"


def save_fetched(path, destination, in_leaf):
    """Store a fetched file that cp_verify has checked, never overwriting a file.

    Another local user must not make the script create a file elsewhere or change another file's
    mode. So nothing is saved in or below a directory that another account may be able to change
    (others_can_change()), and the directory that the caller named must exist. Failing to save only
    prints a warning: the check has passed. in_leaf: the file goes into leaf/, which the script
    appends to that directory.
    """
    directory, name = os.path.split(destination)
    # The caller may name a directory through a symlink. leaf/ is created, if missing, and opened
    # relative to that directory's descriptor, without following a symlink; its owner and mode are
    # read from its own descriptor.
    named = os.path.dirname(directory) if in_leaf else directory
    try:
        found = others_can_change(named)
    except ValueError:  # int() in trusted_owners(), as root: more digits than Python converts
        reason = "the environment variable SUDO_UID is not a user ID: unset it"
        print("warning: did not save %s: %s" % (one_line(destination, None), reason))
        return
    if found:
        reason = "another account may be able to change %s (%s)"
        if found[1] == GONE:
            reason = (
                "the directory %s is missing (%s): create it (mkdir -m 755) to keep fetched files"
            )
        elif found[1] == UNSEEN:
            reason = "the directory %s cannot be used (%s)"
        reason %= (one_line(found[0], None), whose(*found))
        if cure(*found) != "N":  # the caller's own, and the caller is not root: see cure()
            reason += OWN_DIRECTORY
        print("warning: did not save %s: %s" % (one_line(destination, None), reason))
        return
    try:
        opened = os.open(os.path.realpath(named), os.O_RDONLY | os.O_DIRECTORY)
        try:
            if in_leaf:
                with contextlib.suppress(FileExistsError):
                    os.mkdir("leaf", 0o755, dir_fd=opened)
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                opened, above = os.open("leaf", flags, dir_fd=opened), opened
                os.close(above)
                about = os.fstat(opened)
                if about.st_uid not in trusted_owners():
                    raise OSError(
                        "the directory leaf belongs to another account: name another "
                        "--public-key-dir"
                    )
                if about.st_mode & 0o022:
                    mine = about.st_uid == os.geteuid() != 0  # as in cure(): never for root
                    raise OSError(
                        "the directory leaf can be written by its group or by other accounts"
                        + (OWN_DIRECTORY if mine else "")
                    )
            link_new_file(path, opened, name)
        finally:
            os.close(opened)
        print("saved the fetched file %s" % one_line(destination, None))
    except OSError as failure:
        reason = one_line(failure.strerror or failure)
        if in_leaf and os.path.islink(directory):  # reported as ENOTDIR, which would mislead
            reason = "the directory leaf is a symlink"
        print("warning: did not save %s: %s" % (one_line(destination, None), reason))


class PrivateDirectory:
    """Context manager: create a temporary directory and make it the working directory.

    A user who can rename entries in TMPDIR could swap in another directory, so it is checked and
    entered through an open descriptor, and removed through a descriptor of its parent directory.
    chmod, which only the owner (or root) may do, confirms the owner even where a user namespace
    shows every unmapped user as one ID.
    """

    DIRECTORIES = ("config", "ivk", os.path.join("ivk", "leaf"))

    def __enter__(self):
        self.files, self.previous, self.identity, self.parent = [], None, None, None
        self.path = tempfile.mkdtemp(prefix="check_proof-")
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(descriptor)
                if info.st_uid != os.geteuid() or os.listdir(descriptor):
                    raise error("the temporary directory %s was replaced" % self.path)
                os.fchmod(descriptor, 0o700)
                with contextlib.suppress(OSError):
                    self.previous = os.open(os.curdir, os.O_RDONLY | os.O_DIRECTORY)
                os.fchdir(descriptor)
                self.identity = (info.st_dev, info.st_ino)
                with contextlib.suppress(OSError):  # then the emptied directory stays
                    self.parent = os.open(os.pardir, getattr(os, "O_PATH", 0) | os.O_DIRECTORY)
            finally:
                os.close(descriptor)
            for name in self.DIRECTORIES:
                os.mkdir(name, 0o700)
        except BaseException:
            self.__exit__()
            raise
        return self

    def new_file(self, *parts):
        """Return the relative name of a file to create; __exit__() deletes it."""
        self.files.append(os.path.join(*parts))
        return self.files[-1]

    def __exit__(self, *_):
        ENDING[0] = True
        here = os.stat(os.curdir)
        if (here.st_dev, here.st_ino) == self.identity:  # delete only what this run created
            for name in self.files:
                with contextlib.suppress(OSError):
                    os.unlink(name)
            for name in reversed(self.DIRECTORIES):
                with contextlib.suppress(OSError):
                    os.rmdir(name)
            if self.previous is None:
                os.chdir(os.sep)
        if self.previous is not None:
            os.fchdir(self.previous)
            os.close(self.previous)
        if self.parent is not None:
            name = os.path.basename(self.path)
            with contextlib.suppress(OSError):
                entry = os.stat(name, dir_fd=self.parent, follow_symlinks=False)
                if (entry.st_dev, entry.st_ino) == self.identity:  # still this run's directory
                    os.rmdir(name, dir_fd=self.parent)
            os.close(self.parent)


def expected_result_v1(args, values, request, leaf_number, commitment_line, work_line):
    """Return the pattern of the "result-v1" line of a pass, or None if the two lines before it are
    not cp_verify's lines on the commitment and on proof of work under this parameter file.
    """
    commitment_line, _, words = commitment_line.partition(RECORD_CLAUSE)
    found = re.fullmatch(ON_THE_COMMITMENT, commitment_line)
    if values.get("record") != "1" or not found:
        return None
    if (
        RECORD_CLAUSE.rstrip() in words
        or len(words) > MAX_RECORD_WORDS
        or not re.fullmatch(RECORD_WORDS, words)
    ):
        return None  # the marker a second time, no clause or words, too many, or not RECORD_WORDS
    assumed = found[1].split()[0]
    work = ("0", "0", "[0-9]+\\.[0-9][0-9]", "[0-9]+\\.[0-9][0-9]")
    if work_line != NO_WORK:
        found = re.fullmatch(WORK, work_line)
        if not found or found[2] == "1 rounds":
            return None
        work = (found[1], found[2].split()[0], re.escape(found[3]), re.escape(found[4]))
    first, last = request.start + 1, request.start + request.checked
    return RESULT_V1 % (
        0,
        claim(args, values, request),
        args.commitment_id,
        BITS,
        *work,
        assumed,
        request.start,
        request.end,
        request.length,
        r"%d\.\.%d" % (first, last) if request.checked else "none",
        "yes" if request.reaches_end else "no",
        "exact-match" if is_exact_match(values) else "clipped-gap",
        *(values[name] for name in POLICY_LINES),
        above_clip_bound(request.checked, values),
        leaf_number,
    )


# ---- The command line ----


class Once(argparse.Action):
    """Store a value and reject a second one, so that an unquoted value cannot replace an option."""

    def __call__(self, parser, namespace, value, option_string=None):
        seen = namespace.__dict__.setdefault("options_seen", set())
        if self.dest in seen:
            parser.error("%s given more than once (an unquoted value?)" % option_string)
        seen.add(self.dest)
        setattr(namespace, self.dest, value)


def whole_number(text):
    if not re.fullmatch("[0-9]{1,10}", text) or int(text) >= 2**32:
        raise argparse.ArgumentTypeError("%r is not a whole number below 2**32" % text)
    return int(text)


def next_token(text):
    return None if text == "none" else whole_number(text)


def number_pair(text):
    if not re.fullmatch("[0-9]{1,10}:[0-9]{1,10}", text):
        raise argparse.ArgumentTypeError("%r is not of the form N:M" % text)
    return tuple(map(int, text.split(":")))


def end_tokens(text):
    if text == "none":
        return frozenset()
    if not re.fullmatch("[0-9]{1,10}(,[0-9]{1,10})*", text):
        raise argparse.ArgumentTypeError("%r is neither token IDs nor the word none" % text)
    return frozenset(map(int, text.split(",")))


def commitment_id(text):
    if not re.fullmatch(COMMITMENT_ID, text.lower()):
        raise argparse.ArgumentTypeError("a commitment ID must be 64 hex digits")
    return text.lower()


USAGE = """%(prog)s RESPONSE REQUEST_FILE --commitment-id ID
           --public-key-dir DIR --parameter-file FILE --scales-file FILE
           (--range START:END | --reply PROMPT_LEN:REPLY_LEN)
           --end-token IDS|none [options]"""

EPILOG = """exit codes:
  0   "verified" the verifier's plain yes, about what you stated
  0   "within limits" a pass under the clipped-gap limits that you named; weaker
      than a plain yes: it bounds an average and a count, not each token
  1   "rejected" by the verifier; or no proof; or the files are not as stated
  2   "error" a missing file or program, a failed verifier run; wrong usage
  14  "not a verification" for example a test build of the verifier; or counted
      proof of work under --no-proof-of-work
Except after wrong usage or --help, the last two lines are "result: FIELDS" and
"check_proof: QUOTED WORDS: DETAILS". For a program a pass is exit code 0 with
outcome=verified or within-limits, nothing less; claim= tells what passed,
work= how many bits of proof of work cp_verify counted; record=1 stands on
every pass. The last line ends with what the pass rests on: in it, cp_verify's
line "commitment:" says what the binding of the weights rests on. A pass for
which work was counted is a pass too, unless you pass --no-proof-of-work.
least_round= and least_round_without_work= are cp_verify's own figures with and
without the work that the parameter file declares, which can be more than is
counted: so where work=0, read least_round_without_work=.
Both are rounded down to two decimals for printing, and neither is a security
level. Do not compare them with a target: cp_verify makes that comparison
itself, on the exact values. --show-all shows its caveats on them.
SIGTERM, SIGHUP and SIGINT stop the check (exit code 2), but not its clean-up
or last print: if its output is blocked then, use SIGKILL.
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="check_proof.py",
        usage=USAGE,
        description="Check the proof for one LLM reply against values that you state yourself.",
        epilog=EPILOG,
        allow_abbrev=False,
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "response",
        metavar="RESPONSE",
        help="ID.proof next to ID.y, as in SPOOL_DIR/out/, or a directory with proof.bin and y.bin",
    )
    parser.add_argument(
        "request_file",
        metavar="REQUEST_FILE",
        help="the request that you built yourself, for example with spool.py make-request",
    )

    def add(group, name, metavar, text, **more):
        group.add_argument(name, metavar=metavar, help=text, action=Once, **more)

    group = parser.add_argument_group(
        "what the proof must be about",
        "Tokens are counted from 0. The range START:END checks tokens START+1 to END.\n"
        "If END is the number of tokens in the request, the last of them is the next\n"
        "token, the one after the request.",
    )
    add(
        group, "--commitment-id", "ID", "64 hex digits, from a source you trust", type=commitment_id
    )
    text = "the checked range, as spool.py make-request prints it"
    add(group, "--range", "START:END", text, type=number_pair)
    text = "the same as two lengths in tokens; the reply's length includes its end token"
    add(group, "--reply", "PROMPT_LEN:REPLY_LEN", text, type=number_pair)
    text = (
        "token IDs that end a reply: the last checked token must be one of them, and no earlier "
        "one may be; none skips this check"
    )
    add(group, "--end-token", "IDS|none", text, type=end_tokens)
    text = (
        "the token that the server returned after the last token of the request; needed only if "
        "the range reaches the end of the request"
    )
    add(group, "--next-token", "TOKEN|none", text, type=next_token)

    group = parser.add_argument_group(
        "acceptance policy (default: exact match)",
        "A token's cost is its logit gap to the model's greedy choice plus %d. Under\n"
        "clipped-gap the costs, each clipped to C, average at most theta, and at most\n"
        "cap tokens cost more than C. s is the unit in which C is counted. Malformed\n"
        "limits are an error; so is theta = C with cap >= the checked tokens, if any.\n"
        "Whatever limits you give, a parameter file with exact match (C = theta = %d,\n"
        "cap 0) passes, and so does a range that checks the next token alone." % (OFFSET, OFFSET),
    )
    text = "exact: each checked token is the greedy choice"
    group.add_argument("--policy", choices=("exact", "clipped-gap"), help=text, action=Once)
    add(group, "--C", "N", "clipped-gap: the parameter file's C must equal this", type=whole_number)
    add(group, "--s", "N", "clipped-gap: the parameter file's s must equal this", type=whole_number)
    text = "clipped-gap: limit for the file's theta (default %d)" % OFFSET
    add(group, "--theta", "N", text, type=whole_number)
    add(group, "--cap", "N", "clipped-gap: limit for the file's cap (default 0)", type=whole_number)

    group = parser.add_argument_group("files that the provider publishes for the commitment")
    text = "the directory with key.bin and the leaf files (leaf/K.bin; the proof names K)"
    add(group, "--public-key-dir", "DIR", text)
    add(group, "--parameter-file", "FILE", "policy.txt: proof parameters and the acceptance policy")
    add(group, "--scales-file", "FILE", "KL.bin: the quantization scales")
    add(group, "--layout-file", "FILE", "layout.txt (default: next to the parameter file)")
    text = (
        "where to get published files that are missing; they are saved once verified; a URL gets "
        "$%s, if set, as a bearer key" % FETCH_KEY
    )
    add(group, "--fetch-from", "URL|DIR", text)

    group = parser.add_argument_group(
        "the verifier",
        "cp_verify, a C++ program, checks the proof. No theorem covers that program.",
    )
    add(
        group,
        "--verifier",
        "PATH",
        "cp_verify (default: where this repository's build puts it)",
    )
    add(group, "--threads", "N", "threads for cp_verify, 1 to 256 (default 4)", type=whole_number)
    text = (
        "also show cp_verify's long account of what its figures rest on, with its caveats, and, at "
        "a pass, its line for programs"
    )
    group.add_argument("--show-all", action="store_true", help=text)
    text = (
        'accept no pass for which cp_verify counted proof of work (its line "proof of work:"); '
        "with this option the exit code is then 14"
    )
    group.add_argument("--no-proof-of-work", action="store_true", help=text)
    text = "show this message; only on its own"
    parser.add_argument("-h", "--help", action="store_true", help=text)
    return parser


def find_verifier(parser, args):
    """Set args.verifier. Raise CheckFailed, not a usage error, if the program is missing."""
    if args.verifier is not None and os.sep not in args.verifier:  # PATH is not searched
        parser.error("--verifier needs a path, for example ./cp_verify")
    args.verifier = args.verifier or VERIFIER
    if not (os.path.isfile(args.verifier) and os.access(args.verifier, os.X_OK)):
        raise error(
            "the verifier %s does not exist or is not executable (%s; or pass --verifier PATH)"
            % (args.verifier, HOW_TO_BUILD)
        )


def parse_args(argv):
    """Parse and check the command line, and fill in the defaults.

    Raise SystemExit on wrong usage, CheckFailed on a missing verifier program.
    """
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    # Help exits with 0, like a verified proof, so it must be the sole argument: a file named "-h"
    # must not end a check with 0.
    if argv in (["-h"], ["--help"]):
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(argv)
    if args.help:
        parser.error("--help must be the only argument")
    args.policy = args.policy or "exact"
    needed = "--commitment-id --public-key-dir --parameter-file --scales-file --end-token".split()
    if (args.range is None) == (args.reply is None):
        parser.error("give exactly one of --range and --reply")
    if args.reply is not None:
        if not all(args.reply):
            parser.error("--reply needs a prompt and a reply of at least one token each")
        args.range = (args.reply[0] - 1, args.reply[0] + args.reply[1] - 1)
    missing = [name for name in needed if getattr(args, name[2:].replace("-", "_")) is None]
    if missing:
        parser.error("missing: " + ", ".join(missing))
    limits = [name for name in POLICY_LIMITS if getattr(args, name) is not None]
    if args.policy == "exact" and limits:
        parser.error("--%s needs --policy clipped-gap" % limits[0])
    if args.policy == "clipped-gap" and (args.C is None or args.s is None):
        parser.error("--policy clipped-gap needs --C and --s")
    args.theta = OFFSET if args.theta is None else args.theta
    args.cap = args.cap or 0
    args.threads = 4 if args.threads is None else args.threads
    if not 1 <= args.threads <= 256:
        parser.error("--threads takes 1 to 256")
    source = args.fetch_from or ""
    if os.environ.get(FETCH_KEY) and URL.match(source) and not HTTPS.match(source):
        if not is_this_machine(source):
            parser.error(
                "$%s is set: --fetch-from must be https, or http://127.0.0.1 or http://[::1] (not "
                "the name localhost)" % FETCH_KEY
            )
    find_verifier(parser, args)
    args.layout_is_named = args.layout_file is not None
    if not args.layout_is_named:
        args.layout_file = os.path.join(os.path.dirname(args.parameter_file), "layout.txt")
    for name in (
        "response",
        "public_key_dir",
        "parameter_file",
        "scales_file",
        "layout_file",
        "verifier",
        "fetch_from",
    ):
        value = getattr(args, name)  # the check changes the working directory
        if value == "":
            parser.error("an empty path was given")
        if value and not (name == "fetch_from" and URL.match(value)):
            setattr(args, name, os.path.abspath(value))
    return args


# ---- The check ----


def run(words):
    """Run cp_verify with the caller's environment, but without the API keys."""
    environment = {name: value for name, value in os.environ.items() if name not in API_KEYS}
    return subprocess.run(words, capture_output=True, env=environment)


def hide_other_ids(line, own):
    """Return line with HIDDEN_ID for every run of 64 or more hex digits except own."""
    return re.sub(
        "[0-9A-Fa-f]{64,}", lambda found: found[0] if found[0] == own else HIDDEN_ID, line
    )


def pass_on_stderr(done, own=None):
    for line in done.stderr.decode(errors="replace").splitlines():
        line = hide_other_ids(line, own) if own else line
        print(one_line(line, None), file=sys.stderr)


def copy_configuration(args, work, fetched):
    """Copy the three configuration files into the private directory.

    The script and cp_verify read only these copies. Return the parameter file's values. Raise
    CheckFailed with EXIT_ERROR if a file is not a regular file, is missing and cannot be fetched,
    cannot be read or is too long, or if the parameter file is not in the strict form of
    parse_policy().
    """
    copies = (
        (PARAMETER_COPY, args.parameter_file, True),
        (SCALES_COPY, args.scales_file, True),
        # Not fetched if the caller named it: a fetch asks for layout.txt, not for that name.
        (LAYOUT_COPY, args.layout_file, not args.layout_is_named),
    )
    for copy, path, can_be_fetched in copies:
        name = os.path.basename(copy)
        if os.path.isfile(path):
            copy_local(path, MAX_SMALL_FILE_BYTES, work.new_file(copy))
        elif args.fetch_from and not os.path.lexists(path) and can_be_fetched:
            try:
                fetch(args.fetch_from, name, MAX_SMALL_FILE_BYTES, work.new_file(copy))
            except OSError as failure:
                reason = one_line(failure)
                raise error("%s is missing and could not be fetched: %s" % (name, reason))
            fetched.append((copy, path))
        else:
            raise error("%s does not exist or is not a regular file" % path)
    try:
        with open(PARAMETER_COPY, "rb") as f:
            return parse_policy(f.read())
    except PolicyError as failure:
        raise error("the parameter file is not in the strict form required: %s" % failure)


def provide_key(args, work, fetched):
    """Put key.bin, the caller's or a fetched one, into the private directory. Raise CheckFailed
    with EXIT_ERROR if the caller's key.bin is not a regular file, or is missing and none can be
    fetched.
    """
    key = os.path.join(args.public_key_dir, "key.bin")
    if os.path.isfile(key):
        os.symlink(key, work.new_file(KEY_COPY))
    elif args.fetch_from and not os.path.lexists(key):
        try:
            fetch(args.fetch_from, "key.bin", MAX_FETCHED_BYTES, work.new_file(KEY_COPY))
        except OSError as failure:
            reason = one_line(failure)
            raise error("key.bin is missing and could not be fetched: %s" % reason)
        fetched.append((KEY_COPY, key))
    else:
        raise error("%s does not exist or is not a regular file" % key)


def provide_leaf_file(args, work, fetched, number):
    """Put leaf/<number>.bin into the private directory. Return its source, or None if missing."""
    name = os.path.join("leaf", "%d.bin" % number)
    copy, local = os.path.join("ivk", name), os.path.join(args.public_key_dir, name)
    if os.path.isfile(local):
        os.symlink(local, work.new_file(copy))
        return "found locally"
    if args.fetch_from and not os.path.lexists(local):
        try:
            fetch(args.fetch_from, name, MAX_FETCHED_BYTES, work.new_file(copy))
        except OSError as failure:
            print("warning: could not fetch %s: %s" % (name, one_line(failure)))
            return None
        fetched.append((copy, local))
        return "fetched"
    return None


def commitment_id_of_files(args):
    """Return the commitment ID that cp_verify computes from the configuration files and key.bin.
    Raise CheckFailed with EXIT_REJECTED if cp_verify rejects key.bin or a configuration file, and
    otherwise with EXIT_ERROR unless it exits with status 4 and prints exactly one commitment ID.
    """
    plan = run([args.verifier, "-plan", "id", "config", KEY_COPY])
    lines = plan.stdout.decode(errors="replace").strip("\n").split("\n")
    found = [line for line in lines if re.fullmatch(ID_FIELD + COMMITMENT_ID, line)]
    if plan.returncode == 1 and re.fullmatch(NO, lines[-1]):
        shown = one_line(hide_other_ids(" ".join(lines[-1].split()[:2]), args.commitment_id))
        raise rejected("cp_verify rejected key.bin or a configuration file: %s" % shown)
    if plan.returncode != 4 or len(found) != 1:
        raise error(
            "cp_verify -plan id exited with status %d; it should exit with 4 and print exactly one "
            'line "commitment ID: ..." (is it built from this repository?)' % plan.returncode
        )
    return found[0][len(ID_FIELD) :]


class Request:
    """The caller's request file, checked against the caller's own values."""

    def __init__(self, args, data, token_in_response):
        if len(data) % 4 or len(data) < 12:
            raise error("the request file does not contain token IDs followed by START and END")
        self.data, self.length = data, len(data) // 4 - 2
        self.start, self.end = struct.unpack_from("<2I", data, 4 * self.length)
        if not self.start < self.end <= self.length:
            raise error(
                "the request file has START=%d and END=%d; it needs START < END <= %d"
                % (self.start, self.end, self.length)
            )
        if args.range != (self.start, self.end):
            raise rejected(
                "the request file's checked range is %d:%d, but you expect %d:%d"
                % (self.start, self.end, *args.range)
            )
        self.reaches_end = self.end == self.length
        if self.reaches_end != (args.next_token is not None):
            raise error(
                "--next-token is needed if, and only if, the range reaches the request's end"
            )
        if self.reaches_end and args.next_token != token_in_response:
            # Do not show the response's token: its file may be a hard link to some other file.
            raise rejected(
                "the response is about another next token than the %d that you expect"
                % args.next_token
            )
        # Number of request tokens under the acceptance policy.
        self.checked = min(self.end, self.length - 1) - self.start
        self.last = args.next_token if self.reaches_end else self.token(self.end)
        if args.end_token:
            self.check_end_token(args.end_token)

    def token(self, number):
        return struct.unpack_from("<I", self.data, 4 * number)[0]

    def check_end_token(self, allowed):
        if self.last not in allowed:
            raise rejected("the last checked token is %d, not one of your end tokens" % self.last)
        # Tokens up to START are the prompt, which may contain the end tokens of earlier turns.
        for number in range(self.start + 1, self.end):
            if self.token(number) in allowed:
                raise rejected("token %d of the request is an end token inside the reply" % number)


def check_own_limits(args, request):
    """Raise CheckFailed if the caller's limits are malformed or, in one case, constrain nothing.

    That case: a token is checked, theta equals C, and cap is not below the number of checked
    tokens. cp_verify never sees the caller's limits, so a mistake in them would go unnoticed."""
    if args.policy != "clipped-gap":
        return
    if (
        min(args.C, args.theta) < OFFSET
        or args.theta > args.C
        or args.cap > request.length
        or (args.theta == args.C and args.cap >= request.checked > 0)
    ):
        needed = "%d <= theta <= C and cap <= %d" % (OFFSET, request.length)
        if request.checked:
            needed += ", and theta < C or cap < %d, the number of checked tokens" % request.checked
        raise error(
            "your limits C=%d theta=%d cap=%d cannot be used: limits need %s"
            % (args.C, args.theta, args.cap, needed)
        )


def check_limits(args, values, request):
    """Raise CheckFailed unless the parameter file's policy is within the caller's limits."""
    check_own_limits(args, request)
    fault = policy_fault(values)
    if fault:
        raise rejected("the parameter file %s: this script does not support that" % fault)
    if not request.checked or is_exact_match(values):  # stricter than any clipped-gap policy
        return
    if args.policy == "exact":
        raise rejected(
            "the parameter file has a clipped-gap policy, but you asked for exact match (if a "
            "source that you trust for this model gave you limits, pass them with --policy "
            "clipped-gap; never take them from the file)"
        )
    # theta and cap are upper limits: tokens that meet (C, theta, cap) also meet every larger theta
    # and cap at the same C. The same is not true of C.
    clip, mean, cap, scale = (int(values[name]) for name in POLICY_LIMITS)
    faults = (
        ["C differs"] * (clip != args.C)
        + ["theta is above your limit"] * (mean > args.theta)
        + ["cap is above your limit"] * (cap > args.cap)
        + ["s differs"] * (scale != args.s)
    )
    if faults:
        # The file's numbers are not shown: a caller who copies them as limits accepts any file.
        raise rejected(
            "the parameter file's policy is outside what you accept (%s): C=%d and s=%d with theta "
            "up to %d and cap up to %d; never take limits from the file"
            % ("; ".join(faults), args.C, args.s, args.theta, args.cap)
        )


def open_regular(path):
    """Open a file that comes from somebody else.

    Raise CheckFailed if it cannot be opened, is not a regular file, or is a symbolic link, which
    could point at a file of the caller's. With O_NONBLOCK, opening a FIFO does not block.
    """
    problem = "%s does not exist or is not a regular file (a symbolic link does not count)" % path
    try:
        f = open(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC), "rb")
    except OSError as failure:
        if failure.errno not in (errno.ENOENT, errno.ELOOP, errno.ENXIO):  # ENXIO: a socket
            problem = "%s: %s" % (path, failure.strerror)
        raise error(problem) from None
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
        f.close()
        raise error(problem)
    return f


def read_response(args, work, values):
    """Copy the proof into the private directory, so that cp_verify reads the same bytes as the
    script. Return the copy's name, the response's next token, and the proof's leaf number.
    Raise CheckFailed with EXIT_REJECTED if the service made no proof, the parameter file has no
    line version, or the proof file does not start with this format's header; with EXIT_ERROR if
    the proof or the next token's file cannot be opened as a regular file, the next token's file
    is not exactly one 32-bit word, or the proof is longer than MAX_FETCHED_BYTES.
    """
    if args.response.endswith(".proof") and not os.path.isdir(args.response):
        stem = args.response[: -len(".proof")]
        proof, token_file, failed = args.response, stem + ".y", stem + ".failed"
    else:
        names = ("proof.bin", "y.bin", "failed")
        proof, token_file, failed = (os.path.join(args.response, name) for name in names)
    if os.path.exists(failed):
        text = "the proving service made no proof for this request"
        raise CheckFailed(EXIT_REJECTED, text, NO_PROOF_OUTCOME)
    with open_regular(token_file) as f:
        token = f.read(5)
    if len(token) != 4:
        raise error("%s does not contain exactly one 32-bit word" % token_file)
    if "version" not in values:
        raise rejected("the parameter file has no line version")
    with open_regular(proof) as f, open(work.new_file(PROOF_COPY), "xb") as copy:
        header = chunk = f.read(12)
        while chunk:
            copy.write(chunk)
            if copy.tell() > MAX_FETCHED_BYTES:  # so that no file fills TMPDIR
                raise error(
                    "%s: longer than the %d bytes that it can have" % (proof, MAX_FETCHED_BYTES)
                )
            chunk = f.read(1 << 20)
    if len(header) != 12 or header[:8] != PROOF_HEADER + values["version"].encode():
        raise rejected("the proof file does not start with the header of this proof format")
    return PROOF_COPY, struct.unpack("<I", token)[0], struct.unpack("<I", header[8:])[0]


def named_policy(args, values, request):
    """Return the "-policy" arguments for cp_verify, and the last line of a pass under them.

    cp_verify applies a clipped-gap policy only if the call names it. Return ([], None) if no token
    of the request is checked, or under exact match. The same if the caller did not ask for the
    policy: cp_verify then does not accept the file, and verify_response() has already stopped.
    """
    numbers = policy_numbers(values)
    if args.policy != "clipped-gap" or not request.checked or is_exact_match(values):
        return [], None
    words = ["-policy", ",".join(["clipped-gap"] + [str(number) for number in numbers])]
    return words, WITHIN_LIMITS % numbers


def run_cp_verify(args, values, request, proof, leaf_number, leaf_source):
    """Run cp_verify. Raise CheckFailed unless it passes and the pass is about the caller's values.

    Return cp_verify's lines on the commitment and on proof of work, joined, with a closing clause
    on their figures, and the fields of its "result-v1" line.
    """
    words = [args.verifier, "config", "ivk", "request.bin", "y.bin", proof]
    words += ["-id", args.commitment_id, "-bits", str(BITS), "-j", str(args.threads), "-q"]
    policy_words, within_limits = named_policy(args, values, request)
    words += policy_words
    expected_status, expected_last = 0, YES
    if policy_words:
        expected_status, expected_last = STATUS_WITHIN_LIMITS, within_limits
    done = run(words)
    # Split at "\n" only, so that no other character can start a line.
    lines = done.stdout.decode(errors="replace").strip("\n").split("\n")
    last = lines[-1]
    for_programs = [line for line in lines if line.startswith(RESULT_V1_START)]
    rests_on = lines[-4:-2] if len(lines) > 3 else ["", ""]
    pattern = expected_result_v1(args, values, request, leaf_number, *rests_on)
    problem = None
    if last != expected_last:
        problem = "its last line does not go with this call"
    elif len(for_programs) != 1 or lines[-2] != for_programs[0]:
        problem = 'it did not print one line "result-v1" directly before its last line'
    elif not pattern:
        problem = (
            'its lines on the commitment and on proof of work, directly before "result-v1", are '
            "not those of a pass under this parameter file"
        )
    elif not re.fullmatch(pattern, lines[-2]):
        problem = 'its line "result-v1" is not the one that goes with this call'
    failure = None
    if done.returncode != expected_status:
        failure = no_pass(args, done.returncode, lines, leaf_number, leaf_source)
    elif any(line.startswith(TEST_BUILD) for line in lines):
        failure = not_a_verification("the verifier is a test build, and none of its runs counts")
    elif problem:
        failure = error(
            "cp_verify exited with status %d, but %s (is it built from this repository?)"
            % (done.returncode, problem)
        )
    elif rests_on[1] != NO_WORK and args.no_proof_of_work:
        failure = not_a_verification(
            "cp_verify passed, but printed %r, and you passed --no-proof-of-work" % rests_on[1]
        )
    pass_on_stderr(done, args.commitment_id if failure else None)
    closing = lines[lines.index(CLOSING_HEAD) + 1 :] if CLOSING_HEAD in lines else lines[-1:]
    if not failure:
        closing = closing[:-1]  # a passing last line is never shown
    elif re.fullmatch(SHOWN_ENDINGS.get(done.returncode, "(?!)"), last):
        # See check_limits() and HIDDEN_ID.
        closing = [line for line in closing if not line.startswith(WITH_THE_FILES_NUMBERS)]
        closing = [hide_other_ids(line, args.commitment_id) for line in closing]
    else:
        # Any other ending, such as a pass that the script does not accept or an exit code that does
        # not go with the last line: the output may be cut short, or may not count.
        closing = []
    shown = [line for line in closing if args.show_all or not line.startswith(LEFT_OUT)]
    for line in shown:
        print("cp_verify: " + one_line(line, None))
    left_out = len(closing) - len(shown)
    if left_out:
        more = "" if failure else ", and its line for programs"
        note = LEFT_OUT_NOTE % (left_out, " is" if left_out == 1 else "s are", more)
        print("cp_verify: " + note)
    if failure:
        raise failure
    fields = parse_result(lines[-2])
    if rests_on[1] == NO_WORK:  # a file can declare work that no round needs: not that figure
        return "; ".join([*rests_on, ONE_FIGURE % fields["least-round-without-work"]]), fields
    return "; ".join(rests_on) + " " + BOTH_FIGURES, fields


def no_pass(args, status, lines, leaf_number, leaf_source):
    """Return the CheckFailed for a run of cp_verify that did not pass.

    An exit code counts only if the last line matches it.
    """
    shown = one_line(hide_other_ids(" ".join(lines[-1].split()[:2]), args.commitment_id))
    if status == 1 and re.fullmatch(NO, lines[-1]):
        hint = ""
        if any(re.fullmatch(LEAF_FILE_REJECTED, line) for line in lines):
            hint = "%s leaf/%d.bin, not the proof" % (REJECTED_LEAF_FILE, leaf_number)
            if leaf_source == "found locally":
                hint += " (get the published files again, into a new --public-key-dir)"
        return rejected("cp_verify printed %s%s" % (shown, hint))
    if status == 14 and lines[-1] == NOT_A_VERIFICATION:
        return not_a_verification("cp_verify printed " + NOT_A_VERIFICATION)
    if status < 0:
        return error("cp_verify was killed by signal %d" % -status)
    text = "cp_verify exited with status %d" % status
    if lines[-1] != YES and not lines[-1].startswith(WITH_THE_FILES_NUMBERS):  # not a pass's words
        text += " and the last line %r" % shown
    return error(text + ", which is not a verdict on this call")


def describe(args, values, request):
    """Return one sentence on what a pass shows."""
    first, last = request.start + 1, request.start + request.checked
    tokens = "token %d" % first if first == last else "tokens %d to %d" % (first, last)
    next_token = "the next token, %s, is %s" % (args.next_token, GREEDY)
    if not request.checked:
        text = next_token + "; no token of the request itself is checked"
    else:
        text = describe_policy(values, request.checked, tokens + " of the request")
        if request.reaches_end:
            text += "; " + next_token
    if not args.end_token:
        return text + "; the end of the reply was not checked"
    text += "; the last checked token is the end token %d" % request.last
    # With theta = OFFSET and no token above the clip, every gap is 0.
    all_greedy = int(values["theta"]) == OFFSET and not above_clip_bound(request.checked, values)
    if not all_greedy and not request.reaches_end:
        text += " (this policy does not show that it is %s)" % GREEDY
    return text


def claim(args, values, request):
    """Return what a pass shows: "exact-match", "next-token" (no other token) or "policy"."""
    if named_policy(args, values, request)[0]:
        return "policy"
    return "exact-match" if request.checked else "next-token"


def parse_result(line):
    """Return the fields of a "result:" or "result-v1" line as a dictionary."""
    return dict(word.split("=", 1) for word in line.split()[1:])


def result_line(outcome, args, values, request, leaf_number, verifier_fields):
    """Return the "result:" line for a pass.

    leaf=K names the file leaf/K.bin that was read; the prover chooses K.
    """
    limits = "exact"
    if args.policy == "clipped-gap":
        limits = "C:%d,theta:%d,cap:%d,s:%d" % (args.C, args.theta, args.cap, args.s)
    fields = {
        "outcome": outcome,
        "commitment_id": args.commitment_id,
        "range": "%d:%d" % (request.start, request.end),
        "checked_tokens": request.checked,
        "end_token": request.last if args.end_token else "not_checked",
        "claim": claim(args, values, request),
        "policy": policy_kind(values) if request.checked else "none",
        "next_token": args.next_token if request.reaches_end else "none",
        **{name: values[name] for name in POLICY_LIMITS},
        "limits": "not_compared" if not request.checked else limits,
        "request_tokens": request.length,
        "leaf": leaf_number,
        "commitment_assumed_rounds": verifier_fields["commitment-assumed-rounds"],
        "record": verifier_fields["record"],
        "work": verifier_fields["work"],
        "least_round": verifier_fields["least-round"],
        "least_round_without_work": verifier_fields["least-round-without-work"],
        "build": verifier_fields["build"],
    }
    return "result: " + " ".join("%s=%s" % field for field in fields.items())


def verify_response(args, work, values, fetched, request_bytes):
    """Compare the response and the request with the caller's values, and run cp_verify.

    Fetched files are saved after a pass. Raise CheckFailed unless the response passes; a leaf
    file that the proof needs and that is neither found nor fetched gives EXIT_ERROR.
    """
    proof, token_in_response, leaf_number = read_response(args, work, values)
    request = Request(args, request_bytes, token_in_response)
    check_limits(args, values, request)
    with open(work.new_file("request.bin"), "xb") as f:
        f.write(request_bytes)
    with open(work.new_file("y.bin"), "xb") as f:  # the caller's own token, not the response's
        f.write(struct.pack("<I", args.next_token or 0))
    leaf_source = provide_leaf_file(args, work, fetched, leaf_number)
    if leaf_source is None:
        hint = "" if args.fetch_from else " (get it from the provider, or pass --fetch-from)"
        raise error(
            "the leaf file leaf/%d.bin, which the proof needs, is missing%s" % (leaf_number, hint)
        )
    print("leaf file leaf/%d.bin: %s" % (leaf_number, leaf_source))
    found = run_cp_verify(args, values, request, proof, leaf_number, leaf_source)
    for copy, destination in fetched:
        save_fetched(copy, destination, os.path.dirname(copy) == os.path.join("ivk", "leaf"))
    return (request, leaf_number, *found)


def check(args):
    """Run the check. Raise CheckFailed unless it passes."""
    print("verifier: %s" % one_line(args.verifier, None))
    if not os.path.isfile(args.request_file):
        raise error("%s does not exist or is not a regular file" % args.request_file)
    with open(args.request_file, "rb") as f:  # read once, before the working directory changes
        request_bytes = f.read(MAX_SMALL_FILE_BYTES + 1)
    if len(request_bytes) > MAX_SMALL_FILE_BYTES:
        raise error(
            "%s: longer than the %d bytes that it can have"
            % (args.request_file, MAX_SMALL_FILE_BYTES)
        )
    with PrivateDirectory() as work:
        # Pairs: a fetched file's private copy, and the caller's path for it.
        fetched = []
        values = copy_configuration(args, work, fetched)
        provide_key(args, work, fetched)
        if commitment_id_of_files(args) != args.commitment_id:
            raise rejected(  # their ID is not shown: see HIDDEN_ID
                "the public key directory and configuration files have a commitment ID other than "
                "the one you gave"
            )
        request, leaf_number, rests_on, verifier_fields = verify_response(
            args, work, values, fetched, request_bytes
        )
    words, outcome = LAST_LINE[EXIT_VERIFIED], OUTCOMES[EXIT_VERIFIED]
    if named_policy(args, values, request)[0]:
        words, outcome = WITHIN_LIMITS_WORDS, WITHIN_LIMITS_OUTCOME
    text = "%s; %s" % (describe(args, values, request), rests_on)
    result = result_line(outcome, args, values, request, leaf_number, verifier_fields)
    return EXIT_VERIFIED, words, text, result


def main(argv=None):
    """Run the command line, print the result, and return the exit code.

    Not thread-safe: the check changes the working directory.
    """
    ENDING[0] = False
    try:
        try:
            args = parse_args(argv)
        except SystemExit as usage:
            ENDING[0] = True
            return usage.code if isinstance(usage.code, int) else EXIT_ERROR
        code, words, text, result = check(args)
    except Stopped as stopped:
        # The verifier is killed and the private directory deleted. Processes that cp_verify itself
        # started keep running until they finish.
        return stopped_by(stopped)
    except CheckFailed as failed:
        code, words, text = failed.code, LAST_LINE.get(failed.code, "error"), failed.text
        result = NO_PASS % (failed.outcome or OUTCOMES.get(failed.code, OUTCOMES[EXIT_ERROR]))
    except Exception as failure:  # no failure may look like a verdict on the proof
        code, words, result = EXIT_ERROR, LAST_LINE[EXIT_ERROR], NO_PASS % OUTCOMES[EXIT_ERROR]
        text = "%s: %s" % (type(failure).__name__, failure)
    ENDING[0] = True
    return code if print_last_lines(result, words, text) else EXIT_ERROR


def print_last_lines(result, words, text, start=""):
    """Print the "result:" line and the last line in one call. Return whether that worked.

    Nothing is printed after a failure, so no other outcome's lines follow a pass's "result:" line.
    """
    last = "check_proof: %s: %s" % (words, one_line(text, 1100))
    try:
        print("%s%s\n%s" % (start, result, last), flush=True)
    except Exception:
        with contextlib.suppress(Exception):  # or the interpreter's final flush fails
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return False
    return True


def stopped_by(stopped):
    """Print the last two lines after a signal. Return EXIT_ERROR."""
    ENDING[0] = True
    # The signal may have interrupted a printed line, so start a new one.
    result, words = NO_PASS % OUTCOMES[EXIT_ERROR], LAST_LINE[EXIT_ERROR]
    print_last_lines(result, words, "stopped by %s" % stopped, start="\n")
    return EXIT_ERROR


def stop(number, _frame):
    if not ENDING[0]:
        ENDING[0] = True
        raise Stopped(signal.Signals(number).name)


def command():
    """Program entry point, also of the installed command."""
    try:
        # With an inherited, ignored SIGCHLD the verifier's exit code would be lost and read as 0.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        # Otherwise SIGTERM or SIGHUP would leave the verifier running and the private directory in
        # place, and SIGINT would end in a traceback without a last line.
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGHUP, stop)
        signal.signal(signal.SIGINT, stop)
        code = main()
    except Stopped as stopped:
        code = stopped_by(stopped)
    sys.exit(code)


if __name__ == "__main__":
    command()
