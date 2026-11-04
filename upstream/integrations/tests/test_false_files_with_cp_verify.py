# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Run integrations/check_proof.py with the real cp_verify on altered copies of the example files.

The honest files must pass, and no altered case may. The cases also check the exit code, outcome= in
the "result:" line and the start of the last line, so that a rejection (1) and an error (2) are
distinguished. All are skipped if bin/cp_verify or examples/accepted-proof/example.proof is missing.
The examples are small proofs on synthetic weights.
"""

import functools
import http.server
import os
import pathlib
import re
import struct
import subprocess
import sys
import threading

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "integrations" / "check_proof.py"
VERIFIER = REPOSITORY / "bin" / "cp_verify"
EXAMPLE = REPOSITORY / "examples" / "accepted-proof"
OTHER_EXAMPLE = REPOSITORY / "examples" / "accepted-proof-clipped-gap"  # another commitment
PROOF, KEY = "example.proof", "ivk/key.bin"
# example.req: TOKENS token IDs, then START and END of the checked range. example.y: the next token
# (number TOKENS), used only if END == TOKENS. All are 32-bit little-endian words. The checked
# tokens are START + 1 to END, counted from 0.
# The values below are stand-ins, replaced by the example's own if example.proof is there.
TOKENS, START, END, NEXT_TOKEN, LEAF_NUMBER = 4, 1, 4, 0, 0
if (EXAMPLE / PROOF).is_file():
    TOKENS = (EXAMPLE / "example.req").stat().st_size // 4 - 2
    START, END = struct.unpack("<2I", (EXAMPLE / "example.req").read_bytes()[-8:])
    (NEXT_TOKEN,) = struct.unpack("<I", (EXAMPLE / "example.y").read_bytes())
    (LEAF_NUMBER,) = struct.unpack("<I", (EXAMPLE / PROOF).read_bytes()[8:12])
LEAF_NAME = "%d.bin" % LEAF_NUMBER
LEAF = "ivk/leaf/" + LEAF_NAME
OTHER_NUMBER = LEAF_NUMBER + 1  # a leaf file that the example does not have
while (EXAMPLE / "ivk" / "leaf" / ("%d.bin" % OTHER_NUMBER)).exists():
    OTHER_NUMBER += 1

PASSED = (0, "verified")
REJECTED, NO_PROOF, ERROR = (1, "rejected"), (1, "no-proof"), (2, "error")
# How the last line starts, as regular expressions.
BY_CP_VERIFY = "rejected: cp_verify printed NO: "
NOT_BY_THE_ID = BY_CP_VERIFY + "(?!model-statement)"  # not by cp_verify's comparison of the ID
LEAF_FILE = BY_CP_VERIFY + r"\S+: it rejected the leaf file leaf/%d\.bin, not the proof"
OTHER_ID = "rejected: the public key directory and configuration files have a commitment ID other "
NO_HEADER = "rejected: the proof file does not start with the header of this proof format"
NO_LEAF_FILE = r"error: the leaf file leaf/%d\.bin, which the proof needs, is missing"

pytestmark = pytest.mark.skipif(
    not (VERIFIER.is_file() and (EXAMPLE / PROOF).is_file()),
    reason="bin/cp_verify is not built, or the example's files are missing",
)


def read(name):
    return (EXAMPLE / name).read_bytes()


def read_other(name):
    """Read a file of the other honest example, which has another commitment."""
    if not (OTHER_EXAMPLE / name).is_file():
        pytest.skip("examples/accepted-proof-clipped-gap/ is missing")
    return (OTHER_EXAMPLE / name).read_bytes()


def other_leaf_number():
    return struct.unpack("<I", read_other(PROOF)[8:12])[0]


def read_other_leaf_file():
    return read_other("ivk/leaf/%d.bin" % other_leaf_number())


def flipped(data, place):
    """Return data with one bit of data[place] changed. A negative place counts from the end."""
    place %= len(data)
    return data[:place] + bytes([data[place] ^ 1]) + data[place + 1 :]


@functools.lru_cache(maxsize=None)
def behind_the_id():
    """Return the place in key.bin of the first byte after the part that the commitment ID covers.

    After key.bin's 8-byte head come the table's root and the record: the ID covers these. The next
    field is the one-time chain's salt. cp_verify's planning call lists the fields.
    """
    words = [str(VERIFIER), "-plan", "request", "config/policy.txt", "config/KL.bin", "128"]
    words += ["-layout", "config/layout.txt", "-prompt", "example.req", "-reply", "example.y"]
    done = subprocess.run([*words, "-fields", "-q"], cwd=EXAMPLE, capture_output=True, timeout=600)
    field = r"^field: key at ([0-9]+) [0-9]+ salt: one-time chain: chain salt$"
    places = re.findall(field, done.stdout.decode(), re.MULTILINE)
    assert done.returncode == 4 and len(places) == 1, (done.returncode, places)
    return int(places[0])


def other_parameter_file():
    """Return the example's parameter file with seed_lo raised by 1, so that it keeps the strict
    form that the script's parse_policy() requires: outside that form the script reports an error
    (2), not a rejection (1).
    """
    text = read("config/policy.txt")
    seed = re.search(rb"(?m)^seed_lo (\d+)$", text)
    return text[: seed.start(1)] + b"%d" % (int(seed[1]) + 1) + text[seed.end(1) :]


# The example's proof with another leaf number, which is the last word of its 12-byte header.
def with_leaf_number(number):
    return read(PROOF)[:8] + struct.pack("<I", number) + read(PROOF)[12:]


class Files:
    """The files and values of one call: the example's, until a case replaces some."""

    def __init__(self, directory):
        self.directory = directory
        self.response = EXAMPLE / PROOF
        self.request = EXAMPLE / "example.req"
        self.key_directory = EXAMPLE / "ivk"
        self.configuration = EXAMPLE / "config"
        self.commitment_id = read("ID.txt").decode().strip()
        self.range, self.fetch_from = (START, END), None
        self.next_token = NEXT_TOKEN if END == TOKENS else None

    def write(self, name, data):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def state(self, **values):
        """Set commitment_id, range, next_token or response for the call."""
        vars(self).update(values)
        return self

    def new_response(self, proof=None, token=None):
        """Use a copy of the response with these bytes as the proof or the token file."""
        self.write("response/example.y", read("example.y") if token is None else token)
        self.response = self.write(
            "response/example.proof", read(PROOF) if proof is None else proof
        )
        return self

    def new_request(self, places=(), start=START, end=END):
        """Use a copy of the request with another token at these places, and with this range."""
        words = list(struct.unpack("<%dI" % TOKENS, read("example.req")[: 4 * TOKENS]))
        for place in places:
            words[place] = 2 if words[place] == 1 else 1
        self.request = self.write("request", struct.pack("<%dI" % (TOKENS + 2), *words, start, end))
        return self.state(range=(start, end), next_token=NEXT_TOKEN if end == TOKENS else None)

    def new_key_directory(self, key=True, leaf=True, leaf_number=LEAF_NUMBER):
        """Use a copy of the public key directory (ivk). True: the example's file; None: no file."""
        self.key_directory = self.directory / "ivk"
        (self.key_directory / "leaf").mkdir(parents=True)
        if key is not None:
            self.write("ivk/key.bin", read(KEY) if key is True else key)
        if leaf is not None:
            self.write("ivk/leaf/%d.bin" % leaf_number, read(LEAF) if leaf is True else leaf)
        return self

    def new_parameter_file(self):
        """Use a copy of the configuration with one number of the parameter file changed."""
        for path in (EXAMPLE / "config").iterdir():  # the bytes alone: the modes may be read-only
            self.write("config/" + path.name, path.read_bytes())
        self.configuration = self.directory / "config"
        self.write("config/policy.txt", other_parameter_file())
        return self

    def fetch_leaf_file(self, data):
        """Use a key directory without a leaf file. Publish these bytes (None: nothing) as that
        file, in the directory that --fetch-from names or, in one test, a local server serves.
        """
        if self.key_directory == EXAMPLE / "ivk":
            self.new_key_directory(leaf=None)
        self.fetch_from = self.directory / "published"
        self.fetch_from.mkdir(exist_ok=True)
        (self.fetch_from / "leaf" / LEAF_NAME).unlink(missing_ok=True)
        if data is not None:
            self.write("published/leaf/" + LEAF_NAME, data)
        return self

    def fetch_all_files(self, altered=None):
        """Start without the published files, and publish them all, one altered if named."""
        self.new_key_directory(key=None, leaf=None)
        self.configuration = self.directory / "config"
        self.configuration.mkdir()
        for name in ("config/policy.txt", "config/KL.bin", "config/layout.txt", KEY, LEAF):
            data = read(name)
            if name == altered == "config/policy.txt":
                data = other_parameter_file()
            elif name == altered:
                data = flipped(data, len(data) // 2)
            self.write("published/" + name.split("/", 1)[1], data)
        self.fetch_from = self.directory / "published"
        return self

    def own_commitment_id(self):
        """Pass the commitment ID that cp_verify computes from these files."""
        words = [VERIFIER, "-plan", "id", self.configuration, self.key_directory / "key.bin"]
        printed = subprocess.run(words, capture_output=True, text=True, timeout=600).stdout
        found = re.search(r"(?m)^commitment ID: ([0-9a-f]{64})$", printed)
        return self.state(commitment_id=found[1])

    def saved(self):
        """Return the names of the configuration's files and of the key directory's *.bin files."""
        found = [*self.configuration.iterdir(), *self.key_directory.rglob("*.bin")]
        return sorted(path.name for path in found)

    def check(self):
        """Run the script. Return its exit code, outcome= and the last line's text."""
        (self.directory / "tmp").mkdir(exist_ok=True)
        words = [sys.executable, "-B", str(SCRIPT), str(self.response), str(self.request)]
        words += ["--public-key-dir", str(self.key_directory)]
        words += ["--parameter-file", str(self.configuration / "policy.txt")]
        words += ["--scales-file", str(self.configuration / "KL.bin")]
        words += ["--commitment-id", self.commitment_id, "--range", "%d:%d" % self.range]
        if self.next_token is not None:
            words += ["--next-token", str(self.next_token)]
        if self.fetch_from is not None:
            words += ["--fetch-from", str(self.fetch_from)]
        words += ["--end-token", "none"]
        environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(self.directory / "tmp")}
        # Without it the script, under sudo below that account's folder, would save no fetched file.
        if "SUDO_UID" in os.environ:
            environment["SUDO_UID"] = os.environ["SUDO_UID"]
        done = subprocess.run(words, capture_output=True, text=True, env=environment, timeout=600)
        assert "Traceback" not in done.stderr and not list((self.directory / "tmp").iterdir())
        lines = done.stdout.splitlines()
        assert sum(line.startswith("result: ") for line in lines) == 1
        outcome = re.fullmatch(r"result: outcome=([a-z-]+)( .*)?", lines[-2])
        assert outcome and bool(outcome[2]) == (done.returncode == 0), lines[-2]
        return done.returncode, outcome[1], lines[-1].removeprefix("check_proof: ")


@pytest.fixture
def files(tmp_path):
    return Files(tmp_path)


def test_honest_files_pass(files):
    code, outcome, last = files.check()
    assert (code, outcome) == PASSED
    assert last.startswith("verified: ")


def test_honest_response_in_a_directory_passes(files):
    files.write("response/proof.bin", read(PROOF))
    files.response = files.write("response/y.bin", read("example.y")).parent
    assert files.check()[:2] == PASSED


def test_honest_fetched_leaf_file_passes_and_is_saved(files):
    assert files.fetch_leaf_file(read(LEAF)).check()[:2] == PASSED
    assert (files.key_directory / "leaf" / LEAF_NAME).read_bytes() == read(LEAF)


def test_all_published_files_are_fetched_and_saved(files):
    assert files.fetch_all_files().check()[:2] == PASSED
    assert files.saved() == sorted([LEAF_NAME, "KL.bin", "key.bin", "layout.txt", "policy.txt"])


# Each case: what it alters, and how the last line starts.
REJECTED_CASES = {
    # The caller states a value that differs from the honest files: cp_verify never gets the proof.
    "the caller expects another next token": (
        lambda files: files.state(next_token=NEXT_TOKEN + 1),
        "rejected: the response is about another next token than the %d that" % (NEXT_TOKEN + 1),
    ),
    "the caller expects another range": (
        lambda files: files.state(range=(START - 1, END)),
        "rejected: the request file's checked range is %d:%d, but you expect %d:%d"
        % (START, END, START - 1, END),
    ),
    "the caller gives another commitment ID": (
        lambda files: files.state(commitment_id="ab" * 32),
        OTHER_ID,
    ),
    # The request is altered, the proof is not, and the caller states the altered values.
    "a token of the prompt": (lambda files: files.new_request([0]), BY_CP_VERIFY),
    "a checked token": (lambda files: files.new_request([START + 1]), BY_CP_VERIFY),
    "the range starts earlier": (lambda files: files.new_request(start=START - 1), BY_CP_VERIFY),
    "the range starts later": (lambda files: files.new_request(start=START + 1), BY_CP_VERIFY),
    "the range ends earlier": (lambda files: files.new_request(end=END - 1), BY_CP_VERIFY),
    # The response is altered.
    "the next token, in the response and in the call": (
        lambda files: files.new_response(token=struct.pack("<I", NEXT_TOKEN + 1)).state(
            next_token=NEXT_TOKEN + 1
        ),
        BY_CP_VERIFY,
    ),
    "the proof's first word": (
        lambda files: files.new_response(flipped(read(PROOF), 0)),
        NO_HEADER,
    ),
    "the proof's version word": (
        lambda files: files.new_response(flipped(read(PROOF), 4)),
        NO_HEADER,
    ),
    "the proof's first byte after the header": (
        lambda files: files.new_response(flipped(read(PROOF), 12)),
        BY_CP_VERIFY,
    ),
    "the proof's middle byte": (
        lambda files: files.new_response(flipped(read(PROOF), len(read(PROOF)) // 2)),
        BY_CP_VERIFY,
    ),
    "the proof's last byte": (
        lambda files: files.new_response(flipped(read(PROOF), -1)),
        BY_CP_VERIFY,
    ),
    "the proof is short": (lambda files: files.new_response(read(PROOF)[:-4]), BY_CP_VERIFY),
    "the proof is long": (lambda files: files.new_response(read(PROOF) + bytes(4)), BY_CP_VERIFY),
    "the proof is its header alone": (
        lambda files: files.new_response(read(PROOF)[:12]),
        BY_CP_VERIFY,
    ),
    # The public key directory is altered.
    "the leaf file is altered": (
        lambda files: files.new_key_directory(leaf=flipped(read(LEAF), 300)),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "the same leaf file, under another number in the proof and in its name": (
        lambda files: files.new_response(with_leaf_number(OTHER_NUMBER)).new_key_directory(
            leaf_number=OTHER_NUMBER
        ),
        BY_CP_VERIFY,
    ),
    "the leaf file is altered, under another number": (
        lambda files: files.new_response(with_leaf_number(OTHER_NUMBER)).new_key_directory(
            leaf=flipped(read(LEAF), 300), leaf_number=OTHER_NUMBER
        ),
        LEAF_FILE % OTHER_NUMBER,
    ),
    "the leaf file is short": (
        lambda files: files.new_key_directory(leaf=read(LEAF)[:-4]),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "the leaf file is long": (
        lambda files: files.new_key_directory(leaf=read(LEAF) + bytes(4)),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "key.bin is altered": (
        lambda files: files.new_key_directory(key=flipped(read(KEY), behind_the_id())),
        NOT_BY_THE_ID,
    ),
    "key.bin is altered where the commitment ID covers it": (
        lambda files: files.new_key_directory(key=flipped(read(KEY), behind_the_id() - 1)),
        OTHER_ID,
    ),
    "key.bin's last byte": (
        lambda files: files.new_key_directory(key=flipped(read(KEY), -1)),
        BY_CP_VERIFY,
    ),
    "key.bin is short": (lambda files: files.new_key_directory(key=read(KEY)[:-4]), BY_CP_VERIFY),
    "key.bin and the leaf file are exchanged": (
        lambda files: files.new_key_directory(key=read(LEAF), leaf=read(KEY)),
        "rejected: cp_verify rejected key.bin or a configuration file: NO: ",
    ),
    # The configuration is altered.
    "another parameter file": (lambda files: files.new_parameter_file(), OTHER_ID),
    "another parameter file, under its own commitment ID": (
        lambda files: files.new_parameter_file().own_commitment_id(),
        NOT_BY_THE_ID,
    ),
    # Honest files of two commitments are mixed.
    "another commitment's key.bin": (
        lambda files: files.new_key_directory(key=read_other(KEY)),
        OTHER_ID,
    ),
    "another commitment's leaf file": (
        lambda files: files.new_key_directory(leaf=read_other_leaf_file()),
        LEAF_FILE % LEAF_NUMBER,
    ),
    # The example's leaf file is stored under the number that the other proof names: with no file
    # of that number the script would report an error (2), not a rejection.
    "another commitment's proof": (
        lambda files: files.new_response(read_other(PROOF)).new_key_directory(
            leaf_number=other_leaf_number()
        ),
        BY_CP_VERIFY,
    ),
    # A wrong leaf file is fetched. It is not saved.
    "a fetched leaf file is another commitment's": (
        lambda files: files.fetch_leaf_file(read_other_leaf_file()),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "a fetched leaf file is altered": (
        lambda files: files.fetch_leaf_file(flipped(read(LEAF), 300)),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "a fetched leaf file is short": (
        lambda files: files.fetch_leaf_file(read(LEAF)[:-4]),
        LEAF_FILE % LEAF_NUMBER,
    ),
    "a fetched leaf file is long": (
        lambda files: files.fetch_leaf_file(read(LEAF) + bytes(4)),
        LEAF_FILE % LEAF_NUMBER,
    ),
}
# Cases that are skipped if the example's range does not allow them.
# NEED_A_RANGE: a range of at least two positions.
NEED_A_RANGE = ("a checked token", "the range starts later", "the range ends earlier")
NEED_A_POSITION_BEFORE = ("the caller expects another range", "the range starts earlier")
NEED_THE_NEXT_TOKEN = (
    "the caller expects another next token",
    "the next token, in the response and in the call",
)

# A needed file is missing, so the script cannot reach a verdict. With no response, the message
# names the token file (nothing.y): the script's read_response() opens it before the proof.
ERROR_CASES = {
    "no response": (
        lambda files: files.state(response=files.directory / "nothing.proof"),
        r"error: \S+/nothing\.y does not exist or is not a regular file",
    ),
    "no key.bin": (
        lambda files: files.new_key_directory(key=None),
        r"error: \S+/ivk/key\.bin does not exist or is not a regular file",
    ),
    "no leaf file": (lambda files: files.new_key_directory(leaf=None), NO_LEAF_FILE % LEAF_NUMBER),
    "the proof names a leaf file that is not there": (
        lambda files: files.new_response(with_leaf_number(OTHER_NUMBER)),
        NO_LEAF_FILE % OTHER_NUMBER,
    ),
    "the proof names a very high leaf number": (
        lambda files: files.new_response(with_leaf_number(4000000000)),
        NO_LEAF_FILE % 4000000000,
    ),
    "the leaf file to fetch is not there": (
        lambda files: files.fetch_leaf_file(None),
        NO_LEAF_FILE % LEAF_NUMBER,
    ),
}


@pytest.mark.parametrize("case", REJECTED_CASES)
def test_false_files_are_rejected(files, case):
    if case in NEED_A_RANGE and END - START < 2:
        pytest.skip("the example's range has one position")
    if case in NEED_A_POSITION_BEFORE and START == 0:
        pytest.skip("the example's range starts at the request's first position")
    if case in NEED_THE_NEXT_TOKEN and END != TOKENS:
        pytest.skip("the example's range does not reach the request's end")
    alter, last_line = REJECTED_CASES[case]
    code, outcome, last = alter(files).check()
    assert (code, outcome) == REJECTED and re.match(last_line, last), last
    if files.fetch_from is not None:
        assert not list((files.key_directory / "leaf").iterdir())


@pytest.mark.parametrize("case", ERROR_CASES)
def test_missing_files_are_errors(files, case):
    alter, last_line = ERROR_CASES[case]
    code, outcome, last = alter(files).check()
    assert (code, outcome) == ERROR and re.match(last_line, last), last


def test_marker_that_the_service_made_no_proof(files):
    files.response = files.write("response/example.failed", b"").with_suffix(".proof")
    code, outcome, last = files.check()
    assert (code, outcome) == NO_PROOF
    assert last == "rejected: the proving service made no proof for this request"


@pytest.mark.parametrize(
    "altered, expected, last_line",
    [
        ("config/policy.txt", REJECTED, "rejected: "),
        ("config/KL.bin", REJECTED, OTHER_ID),
        ("config/layout.txt", REJECTED, "rejected: "),
        (KEY, REJECTED, BY_CP_VERIFY),
    ],
    ids=["policy.txt", "KL.bin", "layout.txt", "key.bin"],
)
def test_nothing_is_saved_if_a_fetched_file_is_altered(files, altered, expected, last_line):
    code, outcome, last = files.fetch_all_files(altered).check()
    assert (code, outcome) == expected and last.startswith(last_line), last
    assert files.saved() == []


def test_honest_leaf_file_is_fetched_after_a_false_one_was_rejected(files):
    assert files.fetch_leaf_file(flipped(read(LEAF), 300)).check()[:2] == REJECTED
    assert files.fetch_leaf_file(read(LEAF)).check()[:2] == PASSED
    assert (files.key_directory / "leaf" / LEAF_NAME).read_bytes() == read(LEAF)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


@pytest.mark.parametrize(
    "published, expected, saved",
    [("honest", PASSED, True), ("altered", REJECTED, False), ("nothing", ERROR, False)],
    ids=["honest", "altered", "nothing"],
)
def test_leaf_file_fetched_from_a_server(files, published, expected, saved):
    leaf = {"honest": read(LEAF), "altered": flipped(read(LEAF), 300), "nothing": None}[published]
    handler = functools.partial(QuietHandler, directory=str(files.fetch_leaf_file(leaf).fetch_from))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            files.fetch_from = "http://127.0.0.1:%d" % server.server_address[1]
            assert files.check()[:2] == expected
        finally:
            server.shutdown()
    assert (files.key_directory / "leaf" / LEAF_NAME).exists() == saved
