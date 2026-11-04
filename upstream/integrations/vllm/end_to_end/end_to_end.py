#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Run a provider and a client end to end on one machine.

The script starts vLLM with the proof middleware, sends prompts with the example client, runs the
proving service (cp_prove), and lets the client fetch and check each proof. As a negative control,
it changes one reply, which must get no proof and be rejected (skipped under clipped-gap with your
own commitment). Without the options for your own commitment, it first commits to a small synthetic
model. See integrations/vllm/README.md.

With --serve float (ordinary serving), each proof is one prefill pass of the integer model over the
prompt and the reply, under clipped-gap by default. With --serve integer, vLLM runs the integer
model itself.

The run writes its files to WORK_DIR. cp_prove also updates its batch counter in --secret-key-dir,
and vLLM uses its usual temporary and cache directories.
"""

import argparse
import contextlib
import dataclasses
import errno
import http.client
import importlib.metadata
import importlib.util
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
import urllib.request

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if __name__ == "__main__":
    # No __pycache__ in a "pip install -e" checkout.
    sys.dont_write_bytecode = True
try:
    import check_proof
    import proof_middleware
    import spool

    NOT_INSTALLED = None
except ModuleNotFoundError as error:  # parse_args() reports it after handling --help
    NOT_INSTALLED = error.name

PYTHON = (sys.executable, "-P", "-m")  # -P: import nothing from the current directory

VLLM_VERSION = "0.29.0"
SERVED_MODEL = "demo"
POLICY_NAME = "demo"
DEFAULT_POLICY = {"integer": "exact", "float": "clipped-gap"}

# Where verifier/build.sh and prover/build.sh put the programs.
VERIFIER = os.path.join("bin", "cp_verify")
BUILD_DIR = os.path.join("prover", "build", "hc")
DEMO_CONFIGURATION = "target_small_f2_roomy"  # under prover/examples/; for exact match
DEMO_WEIGHTS = "made-up-blocks"  # cp_commit -weights: synthetic weights
DEMO_MODEL_WEIGHTS = {"weights": "blocks", "weight_settings": {"blocks": "synthetic"}}
DEMO_PROMPT_LENGTHS = (1, 3, 8, 16, 32, 41, 56, 63)
DEMO_KV_CACHE_BYTES = 32 << 20  # enough for 4,096 tokens

# Looked for in other programs' logs.
RESULT = "result: "  # check_proof.py, the example client: fields, outcome= first
CHECK_PROOF = "check_proof: "
COMMIT_ID_LINE = "commitment ID: "  # cp_commit
CAUSE_LINE = r"request %s: failed: cause (\d+)\b"  # cp_prove, only if built with test options
POLICY_CAUSE = "26"  # "the tokens do not meet the policy"
GPU_IN_USE = "refused (gpu-in-use)"  # cp_prove: its GPU has over 2 GiB in use
SETTING = "setting: "  # cp_prove: what it was compiled with; the last part is the kind of build
SERVING_BUILD = "a serving build"
# The "result:" line does not say which program rejected; verifier_rejected() looks for this:
VERIFIER_NO = CHECK_PROOF + "rejected: cp_verify printed NO: "
STORED = 75  # the client's exit code with --ask-only
STORED_OUTCOME = "stored"  # and its outcome=: it exits with the same code when told to try again

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
PROC = "/proc"
STOP_GRACE_S = 20  # from SIGTERM to SIGKILL
# From SIGTERM to SIGKILL for the programs that use a GPU (vLLM, cp_commit, cp_prove): a generous
# bound, not a measurement. A kill during GPU work can disturb a card's other users. And on SIGTERM
# cp_commit removes the secret files that it has made, and cp_prove wipes its secrets once the GPU
# work that is under way has ended.
GPU_GRACE_S = 900
KILL_WAIT_S = 10  # for an exit after SIGKILL
STOP_TO_PROVE_S = 300  # for vLLM's exit, which releases pinned memory


class StepError(Exception):
    """A step did not give the expected result."""


class Interrupted(BaseException):
    """The script received SIGINT, SIGTERM or SIGHUP."""


@dataclasses.dataclass
class Commitment:
    commitment_id: str
    public_key_dir: str
    secret_key_dir: str
    parameter_file: str
    scales_file: str
    model_dir: str
    prompts_file: str
    end_token: str
    limits: tuple = ()  # C, theta, cap, s that the client accepts


def say(text):
    print(text, flush=True)


class Programs:
    """The programs that this script starts, each in a new process group and with a log file.

    A group gets signals only before its program is reaped: until then the kernel cannot give the
    PID, which is also the group's ID, to another process.
    """

    def __init__(self, log_dir, log_fd=None):
        # Under an ignored SIGCHLD, which can be inherited, the kernel reaps every child at once:
        # its exit code is lost, and its PID can become another process's.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        self.log_dir = log_dir
        # Logs go through this descriptor: renaming a directory above cannot move them.
        self.log_fd = os.open(log_dir, DIR_FLAGS) if log_fd is None else log_fd
        self.running = []
        self.starting = False  # hold interrupts back until the program is in self.running
        self.held_back = None  # the run's first signal, which sets the exit code
        self.quiet = False  # hold interrupts back
        self.signals = 0
        self.kill_now = False

    def on_signal(self, number, frame):
        self.signals += 1
        self.held_back = self.held_back or number
        # Ctrl-C after an earlier signal: the user insists. Other signals do not count: one event,
        # such as a service manager's stop, can bring both SIGTERM and SIGHUP.
        if number == signal.SIGINT and self.signals > 1:
            self.kill_now = True
        if not (self.quiet or self.starting):
            self.quiet = True
            raise Interrupted(self.held_back)

    def log_path(self, name):
        return os.path.join(self.log_dir, name + ".log")

    def log_to_read(self, name):
        """Return a path to a log through the descriptor."""
        return "/proc/self/fd/%d/%s.log" % (self.log_fd, name)

    def start(self, name, argv, env=None, grace_s=None):  # grace_s=None: STOP_GRACE_S
        self.starting = True
        try:
            # A new file, never an existing one or a link.
            fd = os.open(
                name + ".log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.log_fd
            )
            with open(fd, "wb") as log:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    process_group=0,
                )
            process.grace_s = grace_s
            self.running.append(process)
        except FileExistsError:
            raise StepError("the log file %s already exists" % self.log_path(name)) from None
        except OSError as error:
            raise StepError("cannot start %s: %s" % (argv[0], error.strerror)) from None
        finally:
            self.starting = False
        if self.held_back is not None:
            self.quiet = True
            raise Interrupted(self.held_back)
        return process

    @staticmethod
    def exited(process):
        """Return whether a program has exited, without reaping it (WNOWAIT): every later signal to
        its group relies on that, because only an unreaped program keeps its PID."""
        if process.returncode is not None:
            return True
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        return os.waitid(os.P_PID, process.pid, flags) is not None

    def reap(self, process):
        """Kill what is left of a program's group, reap it and return its exit code, or None after
        KILL_WAIT_S. If /proc cannot be listed, send SIGKILL whether or not anything is left."""
        living = self.living_groups()
        if living is None or process.pid in living or not self.exited(process):
            os.killpg(process.pid, signal.SIGKILL)
        self.running.remove(process)
        try:
            return process.wait(KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):  # a closed standard output must not end a stop
                say(
                    "warning: %s (PID %d) has not exited after SIGKILL"
                    % (process.args[0], process.pid)
                )
            return None

    def run(self, name, argv, timeout_s, env=None, expected=(0,), grace_s=None):
        """Run a program and return its exit code.

        Raise StepError for a code not in expected (None: any code is), or, having stopped every
        program, after timeout_s seconds.
        """
        process = self.start(name, argv, env, grace_s)
        deadline = time.monotonic() + timeout_s
        while not self.exited(process):
            if time.monotonic() > deadline:
                say(
                    "   %s has not finished within %d s: stopping every program."
                    % (name, timeout_s)
                )
                self.stop_all()
                raise StepError(
                    "%s did not finish within %d s; see %s" % (name, timeout_s, self.log_path(name))
                )
            time.sleep(0.05)
        self.wait_for_groups([process])  # for what it may have left behind
        code = self.reap(process)
        if expected is not None and code not in expected:
            raise StepError(
                "%s exited with code %s, expected %s; see %s"
                % (name, code, " or ".join(map(str, expected)), self.log_path(name))
            )
        return code

    @staticmethod
    def living_groups():
        """Return the IDs of the process groups with a process that has not exited, or None if /proc
        cannot be listed.

        vLLM's own tests (Apache-2.0) look at /proc for the same purpose: tests/utils.py."""
        groups = set()
        try:
            entries = os.listdir(PROC)
        except OSError:
            return None
        for pid in filter(str.isdigit, entries):
            try:
                with open("%s/%s/stat" % (PROC, pid), "rb") as f:  # a name can hold any bytes
                    # After the name: state (Z: exited), parent, group.
                    state, _, group = f.read().rpartition(b")")[2].split()[:3]
                if state != b"Z":
                    groups.add(int(group))
            except (OSError, ValueError):  # it has exited, or the line has another form
                continue
        return groups

    def alive(self, processes):
        """Return the programs that have not exited, or whose group has a process that has not."""
        living = self.living_groups() or set()
        return [p for p in processes if p.pid in living or not self.exited(p)]

    def wait_for_groups(self, processes):
        """Send SIGTERM to the programs' groups that are alive, and wait until every process of
        them has exited, for at most the longest of the programs' graces, or until kill_now."""
        graces = [process.grace_s for process in processes if process.grace_s is not None]
        began, grace_s, told = time.monotonic(), max([STOP_GRACE_S] + graces), False
        for process in self.alive(processes):
            os.killpg(process.pid, signal.SIGTERM)
        while True:
            alive = self.alive(processes)
            waited = time.monotonic() - began
            if not alive or self.kill_now or waited > grace_s:
                return
            if waited > STOP_GRACE_S and not told:
                told = True
                names = ", ".join(os.path.basename(process.args[0]) for process in alive)
                with contextlib.suppress(OSError):  # a closed standard output must not end the stop
                    say(
                        "   Still waiting for %s to exit after SIGTERM, for up to %d s. %s kills "
                        "them at once, and then they wipe and remove nothing."
                        % (names, grace_s, "Ctrl-C" if self.signals else "Ctrl-C, twice,")
                    )
            time.sleep(0.1)

    def stop_and_wait(self, timeout_s):
        """Send SIGTERM to every running program and wait until its whole group has exited.

        Raise StepError after timeout_s seconds. Unlike in stop_all(), an interrupt ends the run."""
        for process in self.running:
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while self.alive(self.running):
            if time.monotonic() > deadline:
                names = ", ".join(process.args[0] for process in self.running)
                raise StepError("%s did not exit within %d s after SIGTERM" % (names, timeout_s))
            time.sleep(0.2)
        for process in list(self.running):
            self.reap(process)

    def stop_all(self):
        """Stop every running program: SIGTERM, a wait until its whole group has exited, and SIGKILL
        only for what is left after the longest of their graces. (If /proc cannot be listed, the
        wait is for the programs themselves, and what they left behind is killed at once.)

        An interrupt is raised at the end, unless quiet was already set. Ctrl-C after an earlier
        signal ends the wait, and SIGKILL follows at once.

        The steps are those of vLLM's own tests (Apache-2.0): tests/utils.py, which also looks at
        /proc for what is left of the group, and tests/v1/kv_connector/rocm_pd_accuracy_utils.py.
        The source archive of vLLM 0.29.0 does not contain the second file.
        """
        was_quiet, self.quiet = self.quiet, True
        self.wait_for_groups(self.running)
        for process in list(self.running):
            self.reap(process)
        self.quiet = was_quiet
        if self.held_back is not None and not was_quiet:
            self.quiet = True
            raise Interrupted(self.held_back)


def copy_to_new_file(source, target):
    """Copy a file. Raise FileExistsError if target exists, also as a link."""
    with open(source, "rb") as f, open(target, "xb") as g:
        shutil.copyfileobj(f, g)


def first_link(path):
    """Return " (LINK)" for the first symbolic link on the way along path, as typed, or "".

    A relative path starts at the working directory's resolved path."""
    at = os.sep if path.startswith(os.sep) else os.getcwd()
    for name in filter(None, path.split(os.sep)):
        at = os.path.join(at, name)
        if os.path.islink(at):
            return " (%s)" % spool.printable(at)
    return ""


def trusted_owners():
    """Return the user IDs whose directories may be above WORK_DIR: root's, this user's and, for
    root with SUDO_UID set, that account's, as in cp_prove: so a WORK_DIR in one's own home
    directory is not refused here under "sudo".

    The price: if a directory above WORK_DIR belongs to that account, the account can at any time
    during the run put a link or another directory in WORK_DIR's place, and root then writes the
    run's files, and reads its results, wherever the account chooses. SUDO_UID stays set in a root
    shell from "sudo -s" and in whatever is started from it, and whoever starts the script as root
    can set it: unset it unless that account may act as root anyway."""
    owners = {0, os.geteuid()}
    named = os.environ.get("SUDO_UID", "")  # taken only if written as sudo writes it: "1000"
    root = os.getuid() == os.geteuid() == 0  # as sudo sets both, and as cp_prove asks
    if root and re.fullmatch("[1-9][0-9]{0,9}", named) and int(named) < 2**32 - 1:
        owners.add(int(named))
    return owners


def trusted_like_root(path):
    """Return (directory, uid) for the first directory above path that neither root nor this user
    owns, or None. After make_work_dir(), its owner can only be the account in SUDO_UID."""
    above = os.sep
    for name in path.strip(os.sep).split(os.sep):
        with contextlib.suppress(OSError):
            uid = os.lstat(above).st_uid
            if uid not in (0, os.geteuid()):
                return above, uid
        above = os.path.join(above, name)
    return None


def fault_of(found):
    """Return what is wrong with a directory above WORK_DIR, by its os.stat(), or None.

    Another user can replace the entries of a directory that none of trusted_owners() owns, and of
    one that group or others can write and that is not sticky. Not covered: a file system that
    another user serves (FUSE with allow_other) can report any owner and mode."""
    if not stat.S_ISDIR(found.st_mode):
        return " is a link or not a directory"
    if found.st_uid not in trusted_owners():
        return " belongs to another user (uid %d)" % found.st_uid
    if found.st_mode & 0o022 and not found.st_mode & stat.S_ISVTX:
        mode = stat.S_IMODE(found.st_mode)
        return " can be written by its group or by others (mode %04o) and is not sticky" % mode
    return None


def fault_in(error):
    """Return what is wrong with a directory above WORK_DIR that gave an OSError."""
    if error.errno in (errno.ELOOP, errno.ENOTDIR):
        return " is a link or not a directory"
    return " does not exist" if error.errno == errno.ENOENT else ": " + error.strerror


def fault_above(path):
    """Return the first directory above path, from "/" down, with a fault, and the fault, or None.

    This look by path is for an early refusal; make_work_dir() looks again at each step."""
    above = os.sep
    for name in path.strip(os.sep).split(os.sep):
        try:
            fault = fault_of(os.lstat(above))
        except OSError as error:
            fault = fault_in(error)
        if fault:
            return above + fault
        above = os.path.join(above, name)
    return None


def make_work_dir(path):
    """Make the new directory path with mode 0700 and return a descriptor of it.

    From "/" down, each directory above it is opened without following a link, relative to the one
    before, which has no fault_of(): so no user outside trusted_owners() can lead a step elsewhere,
    then or later."""
    flags = DIR_FLAGS | os.O_PATH  # a directory above may allow this user to pass, not to read
    *names, last = path.strip(os.sep).split(os.sep)
    fd, walked = os.open(os.sep, flags), os.sep
    try:
        for name in [*names, None]:
            fault = fault_of(os.fstat(fd))
            if fault:
                raise OSError(None, walked + fault)
            if name is not None:
                walked = os.path.join(walked, name)
                try:
                    below = os.open(name, flags, dir_fd=fd)
                except OSError as error:
                    raise OSError(None, walked + fault_in(error)) from None
                os.close(fd)
                fd = below
        os.mkdir(last, 0o700, dir_fd=fd)
        return os.open(last, DIR_FLAGS, dir_fd=fd)
    finally:
        os.close(fd)


def last_line(path, prefix=""):
    """Return the last line of a file that starts with prefix, or ""."""
    with open(path, errors="replace") as f:
        lines = [line for line in f.read().splitlines() if line.startswith(prefix)]
    return lines[-1] if lines else ""


def gpu_env(gpu, **more):
    visible = {} if gpu is None else {"CUDA_VISIBLE_DEVICES": gpu}
    return {**os.environ, **visible, **more}


def vllm_program():
    return os.path.join(os.path.dirname(sys.executable), "vllm")


def positive_int(text):
    if not re.fullmatch(r"[1-9]\d{0,8}", text):
        raise argparse.ArgumentTypeError("%r is not a positive integer" % text)
    return int(text)


OWN_OPTIONS = (
    "commitment_id",
    "public_key_dir",
    "secret_key_dir",
    "parameter_file",
    "scales_file",
    "model_dir",
    "prompts_file",
    "end_token",
)


SET_BY_THE_SCRIPT = ("--host", "--port", "--served-model-name", "--max-model-len", "--middleware")
CP_PROVE_SET_BY_THE_SCRIPT = "isk spool batch end-when-empty".split()


def build_parser():
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        allow_abbrev=False,
        usage="%(prog)s WORK_DIR [options] [-- options for vllm serve]",
        epilog="exit codes: 0 every step that ran gave the expected result (the last line names "
        "what was not checked); 1 a step failed; 2 usage or environment error; 128 + N stopped by "
        "signal N (130 for Ctrl-C). WORK_DIR/logs/ has one log file per program.",
    )
    p.add_argument("work_dir", metavar="WORK_DIR", help="a new directory for the run's files")
    p.add_argument(
        "--serve",
        choices=("float", "integer"),
        help="which model vLLM runs: float: the ordinary floating-point model (ordinary serving; "
        "the default); integer: the integer model that is proved (integer serving; the demo's "
        "default if the installed integer model package has no integer_model_vllm.float_pair)",
    )
    p.add_argument(
        "--policy",
        choices=("exact", "clipped-gap"),
        help="the acceptance policy (default: exact with --serve integer, clipped-gap "
        "with --serve float)",
    )
    p.add_argument(
        "--build-dir",
        default=os.path.join(REPO, BUILD_DIR),
        metavar="DIR",
        help="the directory with cp_prove, cp_commit and the cp_verify that cp_commit runs "
        "(default: %s)" % BUILD_DIR,
    )
    p.add_argument(
        "--verifier",
        default=os.path.join(REPO, VERIFIER),
        metavar="PATH",
        help="the client's cp_verify (default: %s)" % VERIFIER,
    )
    count = {"type": positive_int, "metavar": "N"}
    p.add_argument(
        "--max-tokens", default=16, help="the most tokens in a reply (default: 16)", **count
    )
    p.add_argument(
        "--batch", default=8, help="cp_prove's -batch: requests per batch (default: 8)", **count
    )
    p.add_argument(
        "--cp-prove-options",
        default="",
        metavar="OPTIONS",
        help="extra options for cp_prove, as one argument after an equals sign: "
        '--cp-prove-options="-memory lean". The script passes them on unchecked; on a model\'s '
        "files cp_prove itself takes neither -overlap, -overlap-two nor a test option",
    )
    p.add_argument("--vllm-gpu", metavar="ID", help="CUDA_VISIBLE_DEVICES for vLLM")
    p.add_argument(
        "--prover-gpu",
        metavar="ID",
        help="CUDA_VISIBLE_DEVICES for cp_commit and cp_prove; cp_prove does not start on a GPU "
        "with more than 2 GiB in use, so give a full-size model another GPU, or pass "
        "--stop-vllm-to-prove",
    )
    p.add_argument(
        "--stop-vllm-to-prove",
        action="store_true",
        help="stop vLLM whenever cp_prove runs, and start it again to serve the proofs: for a "
        "full-size model on one GPU",
    )
    seconds = {"type": positive_int, "metavar": "SECONDS"}
    p.add_argument(
        "--start-timeout",
        default=3600,
        help="time limit for each start of vLLM (default: 3600)",
        **seconds,
    )
    p.add_argument(
        "--proof-timeout",
        default=3600,
        help="time limit for each run of any other program that the script starts (default: 3600)",
        **seconds,
    )
    g = p.add_argument_group(
        "your own commitment: give the first eight options together, or none "
        "to run the synthetic demo"
    )
    g.add_argument("--commitment-id", metavar="HEX", help="the published commitment ID")
    g.add_argument(
        "--public-key-dir",
        metavar="DIR",
        help="the public key directory (cp_commit's -ivk), with key.bin and leaf/",
    )
    g.add_argument(
        "--secret-key-dir",
        metavar="DIR",
        help="the secret key directory (cp_commit's -isk), by its full path from /, with no "
        "symbolic link on the way. cp_prove counts its batches there. "
        "Never give the directory of a running service, or a copy or restored backup of one: "
        "with such a directory a leaf table can pass its batch limit unnoticed "
        '(docs/limits.md, "What a provider must do")',
    )
    g.add_argument(
        "--parameter-file",
        metavar="FILE",
        help="the parameter file (policy.txt), with the layout file (layout.txt) beside it",
    )
    g.add_argument("--scales-file", metavar="FILE", help="the scales file (KL.bin)")
    g.add_argument(
        "--model-dir",
        metavar="DIR",
        help="the model directory for vllm serve, with config.json and tokenizer.json: the "
        "floating-point model with --serve float, the integer model with --serve integer",
    )
    g.add_argument(
        "--prompts-file",
        metavar="FILE",
        help="one prompt per line, as token IDs separated by commas",
    )
    g.add_argument(
        "--end-token", metavar="IDS", help="the token IDs that end a reply, separated by commas"
    )
    g.add_argument(
        "--limits",
        metavar="C,THETA,CAP,S",
        help="with --policy clipped-gap: the limits that the client accepts",
    )
    return p


def parse_args(argv=None):
    p = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:cut])
    if NOT_INSTALLED:
        message = "error: %s is not installed (pip install %s/integrations)\n"
        p.exit(2, message % (NOT_INSTALLED, REPO))
    args.vllm_options = argv[cut + 1 :]
    for option in args.vllm_options:
        if option.split("=")[0].replace("_", "-") in SET_BY_THE_SCRIPT:
            p.error("%s cannot follow --: the script sets it" % option.split("=")[0])
    try:
        args.cp_prove_options = shlex.split(args.cp_prove_options)
    except ValueError as error:
        p.error("--cp-prove-options: %s" % error)
    for option in args.cp_prove_options:
        name = option.split("=")[0]
        if name.startswith("-") and name.lstrip("-") in CP_PROVE_SET_BY_THE_SCRIPT:
            p.error(
                "--cp-prove-options cannot contain %s: the script sets it" % spool.printable(option)
            )
    given = [name for name in OWN_OPTIONS if getattr(args, name) is not None]
    args.demo = not given
    args.builds = set()  # the kinds of build that cp_prove's logs name: prove() fills it
    if args.serve is None and args.policy == "exact":
        p.error("with --policy exact, also pass --serve integer")
    args.fell_back = args.serve is None and args.demo and not has_float_pair()
    args.serve = args.serve or ("integer" if args.fell_back else "float")
    args.policy = args.policy or DEFAULT_POLICY[args.serve]
    if given and len(given) < len(OWN_OPTIONS):
        missing = [name for name in OWN_OPTIONS if name not in given]
        p.error("also needed: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.demo and args.serve == "float" and args.policy == "exact":
        p.error(
            "the demo's ordinary serving (--serve float) needs --policy clipped-gap; for exact "
            "match pass --serve integer"
        )
    if args.demo and args.limits:
        p.error("--limits applies only with your own commitment")
    if args.limits and args.policy != "clipped-gap":
        p.error("--limits applies only with --policy clipped-gap")
    if not args.demo:
        args.commitment_id = args.commitment_id.lower()
        if not spool.COMMITMENT_ID.match(args.commitment_id):
            p.error("--commitment-id must be 64 hex digits")
        if not re.fullmatch(r"\d+(,\d+)*", args.end_token):
            p.error("--end-token must be token IDs separated by commas")
        if args.policy == "clipped-gap" and not re.fullmatch(r"\d+(,\d+){3}", args.limits or ""):
            p.error(
                "--policy clipped-gap (the default with --serve float) needs --limits C,THETA,CAP,S"
            )
    if os.path.lexists(args.work_dir):  # as given: a dangling symbolic link counts too
        p.error("%s already exists; choose a new directory" % spool.printable(args.work_dir))
    # Next to the path as given, where check_proof.py looks too.
    args.layout_file = args.parameter_file and os.path.join(
        os.path.dirname(os.path.abspath(args.parameter_file)), "layout.txt"
    )
    for name in ("work_dir", "build_dir", "verifier", *OWN_OPTIONS[1:7]):
        value = getattr(args, name)
        shown = "WORK_DIR" if name == "work_dir" else "--" + name.replace("_", "-")
        if value == "":
            p.error(shown + " must not be empty")
        if value is None:
            continue
        # Like cp_prove's -isk. A relative name is made whole from the working directory's resolved
        # path: a link on the way to that directory would go unseen.
        if name == "secret_key_dir" and not os.path.isabs(value):
            p.error('--secret-key-dir must be a full path from /, for example "$PWD/isk"')
        setattr(args, name, os.path.realpath(value))
        # Another user's link could lead the run's files, or cp_prove's count of what it has proved,
        # to a place of their choice. cp_prove refuses a link on the way to -isk: it must not get a
        # path from which the link has gone. The messages name the link, not where it leads: a
        # person in a hurry would copy that path.
        named = os.sep + os.path.abspath(value).lstrip(os.sep)  # abspath() keeps a leading "//"
        if name == "work_dir" and args.work_dir != named:
            p.error(
                "WORK_DIR leads through a symbolic link%s: name a place that has no link on its way"
                % first_link(value)
            )
        if name == "secret_key_dir" and args.secret_key_dir != named:
            p.error(
                "--secret-key-dir leads through a symbolic link%s: name the directory by its own "
                "path, with no link on the way; do not copy or move it for this" % first_link(value)
            )
    if not re.fullmatch(r"[A-Za-z0-9/._-]{1,380}", args.work_dir):
        p.error(
            "WORK_DIR's full path, %s, may contain only letters, digits and / . _ - and at most "
            "380 characters: cp_prove accepts no other HOME or TMPDIR, and the script puts both "
            "there" % spool.printable(args.work_dir)
        )
    return args


def has_float_pair():
    try:
        return importlib.util.find_spec("integer_model_vllm.float_pair") is not None
    except ModuleNotFoundError:  # integer_model_vllm itself cannot be imported
        return False


def check_prerequisites(args):
    """Return what is missing for the run, as sentences. Create nothing."""
    errors = []
    if sys.version_info < (3, 11):
        errors.append("this script needs Python 3.11 or later")
    fault = fault_above(args.work_dir)
    if fault:
        errors.append(
            "WORK_DIR cannot be used: %s. So that no other user can lead this run's files "
            "elsewhere, choose a new name in an existing directory, below directories that only "
            "you or root can change, such as your home directory or a sticky /tmp. For example: "
            'W="$(mktemp -d -p /tmp)", and then "${W:?}/run" as WORK_DIR.' % fault
        )
    # cp_commit checks what it makes with the cp_verify in the same directory.
    programs = ["cp_prove"] + ["cp_commit", "cp_verify"] * args.demo
    for path in [os.path.join(args.build_dir, name) for name in programs] + [args.verifier]:
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            errors.append(
                "%s is missing or not executable (build it as described in "
                "integrations/vllm/README.md)" % path
            )
    for name in OWN_OPTIONS[1:7]:
        path = getattr(args, name)
        if path is not None and not os.path.exists(path):
            errors.append("%s does not exist" % path)
    if args.secret_key_dir and os.path.isdir(args.secret_key_dir):
        if not os.path.isfile(os.path.join(args.secret_key_dir, "isk.txt")):
            errors.append(
                "%s has no isk.txt, so it is not a secret key directory" % args.secret_key_dir
            )
    if args.parameter_file and os.path.isfile(args.parameter_file):
        if not os.path.isfile(args.layout_file):
            errors.append(
                "%s is missing: put the layout.txt of the configuration that you committed to "
                "beside the parameter file" % args.layout_file
            )
    if args.model_dir and os.path.isdir(args.model_dir):
        if not os.path.isfile(os.path.join(args.model_dir, "tokenizer.json")):
            errors.append(
                "%s has no tokenizer.json; vLLM needs a tokenizer to check prompt token IDs"
                % args.model_dir
            )
    if importlib.util.find_spec("ask_and_check") is None:
        errors.append("ask_and_check is not installed (pip install %s/integrations)" % REPO)
    try:
        installed = importlib.metadata.version("vllm")
        if installed != VLLM_VERSION:
            errors.append(
                "vLLM %s is installed; this script needs %s (pip install -r "
                "integrations/vllm/requirements.txt)" % (installed, VLLM_VERSION)
            )
    except importlib.metadata.PackageNotFoundError:
        errors.append("vLLM is not installed (pip install -r integrations/vllm/requirements.txt)")
    plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
    needs_package = args.serve == "integer" or args.demo
    if needs_package and not any(e.name == "integer_model" for e in plugins):
        errors.append(
            "the integer model package is not installed (pip install --no-deps "
            "--no-build-isolation integrations/vllm/integer_model)"
        )
    elif args.demo and args.policy == "clipped-gap" and not has_float_pair():
        errors.append(
            "the installed integer model package has no integer_model_vllm.float_pair, which the "
            "demo needs for clipped-gap; with this package the demo runs only with --serve integer "
            "and --policy exact"
        )
    if not os.path.isfile(vllm_program()):
        errors.append(
            "there is no vllm program next to %s (install vLLM into this Python "
            "environment)" % sys.executable
        )
    return errors


def read_parameters(path):
    """Return (T, V) from a parameter file: tokens per request, vocabulary size."""
    try:
        with open(path, "rb") as f:
            values = check_proof.parse_policy(f.read())
        return 1 << int(values["lt"]), int(values["V"])
    except (OSError, KeyError, check_proof.PolicyError) as error:
        why = "it has no line %s" % error if isinstance(error, KeyError) else str(error)
        raise StepError("cannot use the parameter file %s: %s" % (path, why)) from None


def copy_configuration(args, parameter_file, scales_file, layout_file):
    """Copy the configuration files to the new directory WORK_DIR/configuration and return it.

    The middleware rejects a parameter file that another user can write, so the copies are private
    (main() sets umask 077). check_proof.py looks for layout.txt next to the parameter file.
    """
    directory = os.path.join(args.work_dir, "configuration")
    os.mkdir(directory)
    files = {"policy.txt": parameter_file, "KL.bin": scales_file, "layout.txt": layout_file}
    for name, source in files.items():
        target = os.path.join(directory, name)
        try:
            copy_to_new_file(source, target)
        except OSError as error:
            raise StepError("cannot copy %s to %s: %s" % (source, target, error.strerror)) from None
    return directory


def demo_configuration(args):
    """Return WORK_DIR/configuration with the demo's configuration files."""
    examples, name = os.path.join(REPO, "prover", "examples"), DEMO_CONFIGURATION
    if args.policy == "clipped-gap":
        from integer_model_vllm import float_pair

        # That configuration's limits are fitted to the demo's two models, which its scales keep
        # close to each other.
        name = float_pair.CONFIGURATION
        if not os.path.isdir(os.path.join(examples, name)):
            raise StepError(
                "the installed integer model package names the configuration %s, which is not in "
                "%s: install this repository's package (pip install --no-deps "
                "--no-build-isolation integrations/vllm/integer_model)" % (name, examples)
            )
    files = ("policy.txt", "KL.bin", "layout.txt")
    return copy_configuration(args, *(os.path.join(examples, name, file) for file in files))


def prover_env(args):
    # cp_prove wants both to be existing directories.
    home, tmp = (os.path.join(args.work_dir, name) for name in ("home", "tmp"))
    return gpu_env(args.prover_gpu, HOME=home, TMPDIR=tmp)


def write_demo_model(args, configuration, first_prompt):
    """Write the demo's model directory. Return its path and the end token."""
    import integer_model
    from integer_model_vllm.example import write_vllm_files

    model_dir = os.path.join(args.work_dir, "integer_model")
    os.mkdir(model_dir)
    for name in ("policy.txt", "KL.bin"):
        copy_to_new_file(os.path.join(configuration, name), os.path.join(model_dir, name))
    with open(os.path.join(model_dir, "integer_model.json"), "x") as f:
        json.dump(DEMO_MODEL_WEIGHTS, f)
    model = integer_model.load_from_folder(model_dir, "cpu")
    # A synthetic model has no end token: declare one.
    if args.serve == "integer":
        # The third token of the first prompt's greedy reply, which then ends there.
        sequence = model.new_sequence()
        token = int(model.extend(sequence, first_prompt)[-1])
        for _ in range(2):
            token = int(model.extend(sequence, [token])[-1])
        model.free(sequence)
    else:
        # The last token ID.
        token = model.parameters.V - 1
    # This also writes a tokenizer, without which vLLM does not check a prompt's token IDs.
    write_vllm_files(model_dir, model.parameters.V, model.parameters.T, token)
    if args.serve == "integer":
        return model_dir, token
    from integer_model.float_export import write_float_model

    float_model_dir = os.path.join(args.work_dir, "float_model")
    os.mkdir(float_model_dir)
    write_float_model(model.parameters, model.weights, float_model_dir, token)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        copy_to_new_file(os.path.join(model_dir, name), os.path.join(float_model_dir, name))
    return float_model_dir, token


def prepare_demo(args, programs):
    configuration = demo_configuration(args)
    say("1. Commit to synthetic weights, with the configuration files in %s." % configuration)
    public_key_dir = os.path.join(args.work_dir, "public_key")
    secret_key_dir = os.path.join(args.work_dir, "secret", "key")
    os.mkdir(os.path.dirname(secret_key_dir), 0o700)
    argv = [os.path.join(args.build_dir, "cp_commit"), "-weights", DEMO_WEIGHTS]
    argv += ["-config", configuration, "-ivk", public_key_dir, "-isk", secret_key_dir]
    argv += ["-M", "2"]  # two leaf tables: the client must fetch the right leaf file
    programs.run("cp_commit", argv, args.proof_timeout, prover_env(args), grace_s=GPU_GRACE_S)
    line = last_line(programs.log_to_read("cp_commit"), COMMIT_ID_LINE)
    commitment_id = line[len(COMMIT_ID_LINE) :]
    if not spool.COMMITMENT_ID.match(commitment_id):
        raise StepError(
            'cp_commit printed no commitment ID (expected a line "%s" and 64 lowercase hex '
            "digits); see %s" % (COMMIT_ID_LINE, programs.log_path("cp_commit"))
        )
    say("   commitment ID " + commitment_id)

    vocabulary = read_parameters(os.path.join(configuration, "policy.txt"))[1]
    tokens = [(31 * i + 7) % vocabulary for i in range(max(DEMO_PROMPT_LENGTHS))]
    prompts_file = os.path.join(args.work_dir, "prompts.txt")
    with open(prompts_file, "x") as f:
        for length in DEMO_PROMPT_LENGTHS:
            f.write(",".join(map(str, tokens[:length])) + "\n")
    model_dir, end_token = write_demo_model(
        args, configuration, list(tokens[: DEMO_PROMPT_LENGTHS[0]])
    )
    with open(os.path.join(configuration, "policy.txt"), "rb") as f:
        values = check_proof.parse_policy(f.read())
    return Commitment(
        commitment_id=commitment_id,
        public_key_dir=public_key_dir,
        secret_key_dir=secret_key_dir,
        parameter_file=os.path.join(configuration, "policy.txt"),
        scales_file=os.path.join(configuration, "KL.bin"),
        model_dir=model_dir,
        prompts_file=prompts_file,
        end_token=str(end_token),
        limits=tuple(values[name] for name in ("C", "theta", "cap", "s")),
    )


def write_middleware_config(args, own):
    policy = {
        "served_model": SERVED_MODEL,
        "commitment_id": own.commitment_id,
        "parameter_file": own.parameter_file,
        "spool_dir": os.path.join(args.work_dir, "spool"),
        "public_key_dir": own.public_key_dir,
        "verifier": args.verifier,
    }
    config = {"jobs_dir": os.path.join(args.work_dir, "jobs"), "policies": {POLICY_NAME: policy}}
    path = os.path.join(args.work_dir, "middleware.json")
    with open(path, "x") as f:
        json.dump(config, f, indent=1)
    return path


def listener_uid(port):
    """Return the user ID of the socket that listens on 127.0.0.1:port, or None."""
    # An address is shown as its four bytes, read as one native-endian integer.
    address = "%08X:%04X" % (struct.unpack("=I", socket.inet_aton("127.0.0.1"))[0], port)
    with open("/proc/net/tcp") as f:
        for line in f.readlines()[1:]:
            fields = line.split()  # 1: local address; 3: state, 0A is listening; 7: user ID
            if fields[1] == address and fields[3] == "0A":
                return int(fields[7])
    return None


def check_listener(url, log):
    """Raise StepError unless a socket of this user listens at url, "http://127.0.0.1:PORT".

    The port is free between the script's probe and vLLM's bind, and again if vLLM exits. Another
    user's program that binds it then should get neither the API key nor a prompt. Not covered: a
    vLLM that exits while the client runs; the prompt's token IDs, which ps shows to every user."""
    port = int(url.rpartition(":")[2])
    if listener_uid(port) != os.geteuid():
        raise StepError(
            "no socket of this user listens on 127.0.0.1:%d: another program took the port before "
            "vLLM did, or vLLM has exited (see %s; to try again, run with a new WORK_DIR)"
            % (port, log)
        )


@dataclasses.dataclass
class Server:
    url: str
    api_key: str
    log: str


def start_vllm(args, own, programs, config, tokens_per_request, name="vllm"):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url, api_key = "http://127.0.0.1:%d" % port, secrets.token_hex(16)
    argv = [vllm_program(), "serve", own.model_dir, "--served-model-name", SERVED_MODEL]
    argv += ["--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(tokens_per_request)]
    argv += ["--generation-config", "vllm", "--enforce-eager"]
    argv += ["--disable-uvicorn-access-log"]  # it would print URLs, which hold proof IDs
    env = {
        "VLLM_API_KEY": api_key,
        proof_middleware.CONFIG_VARIABLE: config,
        "VLLM_ENGINE_READY_TIMEOUT_S": str(args.start_timeout),
        "VLLM_NO_USAGE_STATS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GLOO_SOCKET_IFNAME": "lo",
        "PYTHONSAFEPATH": "1",
    }
    if args.serve == "integer":
        argv += ["--load-format", "integer_model", "--max-num-seqs", "1"]
        # vLLM needs this share of the GPU's memory free, and uses none of it for the integer model.
        argv += ["--no-enable-prefix-caching", "--gpu-memory-utilization", "0.1"]
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    else:
        # With vLLM's Triton kernels for experts, the router's score multiplies the expert's input
        # once, as in the integer model. An empty VLLM_PLUGINS loads no plug-in.
        argv += ["--dtype", "bfloat16", "--moe-backend", "triton"]
        env["VLLM_PLUGINS"] = ""
        if args.demo:
            # cp_prove, which may share the GPU, does not start with over 2 GiB in use. Without the
            # second option, vLLM fills its share with KV cache.
            argv += ["--gpu-memory-utilization", "0.1"]
            argv += ["--kv-cache-memory-bytes", str(DEMO_KV_CACHE_BYTES)]
    argv += ["--middleware", "proof_middleware.ProofMiddleware", *args.vllm_options]
    process, log = (
        programs.start(name, argv, gpu_env(args.vllm_gpu, **env), GPU_GRACE_S),
        programs.log_path(name),
    )
    deadline = time.monotonic() + args.start_timeout
    # The loop has the structure of the one in vLLM's own tests (Apache-2.0): tests/utils.py
    while True:
        if programs.exited(process):
            raise StepError("vLLM exited during start-up; see " + log)
        if time.monotonic() > deadline:
            raise StepError(
                "vLLM did not answer /health within %d s; see %s. If the log shows that it was "
                "still loading the model, pass a larger --start-timeout."
                % (args.start_timeout, log)
            )
        if listener_uid(port) is not None:
            check_listener(url, log)  # before the first call to the port
            with contextlib.suppress(OSError, http.client.HTTPException):
                urllib.request.urlopen(url + "/health", timeout=10).close()
                return Server(url, api_key, log)
        time.sleep(1)


def check_options(args, own):
    options = ["--commitment-id", own.commitment_id, "--parameter-file", own.parameter_file]
    options += ["--public-key-dir", os.path.join(args.work_dir, "client", "public_key")]
    options += ["--scales-file", own.scales_file, "--verifier", args.verifier]
    return options


def client_options(args, own, url, number, prompt, max_tokens):
    options = ["--server", url, "--model", SERVED_MODEL, "--policy-name", POLICY_NAME]
    options += ["--prompt-ids", ",".join(map(str, prompt)), "--max-tokens", str(max_tokens)]
    options += ["--end-token", own.end_token]
    options += ["--out-dir", os.path.join(args.work_dir, "client", str(number))]
    return options + check_options(args, own)


def policy_options(args, own):
    if args.policy == "exact":
        return []
    c, theta, cap, s = own.limits
    return ["--policy", "clipped-gap", "--C", c, "--theta", theta, "--cap", cap, "--s", s]


def put_on_hold(args):
    """Create the empty file "hold" in the spool directory, after a bad end of cp_prove.

    While it exists, the middleware submits nothing and serves no stored response."""
    path = os.path.join(args.work_dir, "spool", "hold")
    try:
        with contextlib.suppress(FileExistsError):
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        say("   cp_prove has not exited with code 0: proofs are on hold while %s exists" % path)
    except OSError as error:  # so that prove() still raises the error on cp_prove's exit
        with contextlib.suppress(OSError):  # standard output may be what failed
            say(
                "   cp_prove has not exited with code 0; cannot create %s: %s"
                % (path, error.strerror)
            )


def check_the_id(args, own):
    """Raise StepError unless cp_verify computes your commitment ID from your public files.

    This says nothing of --secret-key-dir: a secret key directory of another commitment is found
    only by the client's check of the first proof."""
    words = [args.verifier, "-plan", "id", os.path.dirname(own.parameter_file)]
    words += [os.path.join(own.public_key_dir, "key.bin")]
    try:
        done = subprocess.run(words, capture_output=True, timeout=300)
        status, printed = done.returncode, done.stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        status, printed = None, ""
    found = re.findall(r"^commitment ID: ([0-9a-f]{64})$", printed, re.M)
    if status != 4 or found != [own.commitment_id]:
        raise StepError(
            "cp_verify gives these public files another commitment ID than --commitment-id, or "
            "none: see the options, or run: %s" % shlex.join(words)
        )


def prove(args, own, programs, name):
    argv = [os.path.join(args.build_dir, "cp_prove"), "-isk", own.secret_key_dir]
    argv += ["-spool", os.path.join(args.work_dir, "spool")]
    argv += ["-batch", str(args.batch), "-end-when-empty"]
    argv += args.cp_prove_options
    code = None
    try:
        code = programs.run(
            name, argv, args.proof_timeout, prover_env(args), expected=None, grace_s=GPU_GRACE_S
        )
    finally:
        if code != 0:
            put_on_hold(args)
    log = programs.log_path(name)
    args.builds.add(last_line(programs.log_to_read(name), SETTING).rpartition("; ")[2])
    if code is None:
        raise StepError("%s did not exit after SIGKILL; see %s" % (name, log))
    shared = args.vllm_gpu == args.prover_gpu and not args.stop_vllm_to_prove
    gpu_in_use = last_line(programs.log_to_read(name), GPU_IN_USE)
    if code != 0 and gpu_in_use and shared:
        raise StepError(
            "cp_prove did not start: more than 2 GiB of the GPU's memory were in use, and vLLM "
            "runs on the same GPU. Give them different GPUs with --vllm-gpu and --prover-gpu, or "
            "pass --stop-vllm-to-prove; see " + log
        )
    if code != 0 and gpu_in_use:
        raise StepError(
            "cp_prove did not start: more than 2 GiB of its GPU's memory were in use. Give it a "
            "GPU that nothing else uses (--prover-gpu); see " + log
        )
    if code != 0:
        # Once it holds a secret, cp_prove's default build gives only a name, in its last line.
        last = last_line(programs.log_to_read(name))
        printable = "".join(c if c.isprintable() else "?" for c in last[:200])
        said = 'its last line: "%s"' % printable if last else "it printed nothing"
        message = "%s exited with code %d, expected 0; %s; see %s" % (name, code, said, log)
        if shared:
            message += (
                ". If that gives no reason: one common cause is more than 2 GiB of its GPU's "
                "memory in use, for example by vLLM. Give cp_prove its own GPU with --prover-gpu, "
                "or pass --stop-vllm-to-prove."
            )
        raise StepError(message)


def read_outcome(log):
    """Return outcome= of the last "result:" line in a log file, or ""."""
    fields = last_line(log, RESULT).split()[1:]
    if not fields or not fields[0].startswith("outcome="):
        return ""
    return fields[0].removeprefix("outcome=")


def has_passed(args, code, outcome):
    """Return whether the exit code and outcome= of a check fit each other and the policy."""
    expected = check_proof.OUTCOMES[check_proof.EXIT_VERIFIED]
    if args.policy == "clipped-gap":
        expected = check_proof.WITHIN_LIMITS_OUTCOME
    return (code, outcome) == (check_proof.EXIT_VERIFIED, expected)


def serve_and_check(args, own, programs, start, prompts, tokens_per_request):
    """Steps 2 to 5. start(name) starts vLLM, logging to name.log. Return the served replies."""
    kind = "integer" if args.serve == "integer" else "floating-point"
    say("2. Start vLLM %s with the proof middleware, serving the %s model." % (VLLM_VERSION, kind))
    server = start("vllm")
    client_keys = os.path.join(args.work_dir, "client", "public_key")
    os.mkdir(client_keys, 0o700)
    copy_to_new_file(
        os.path.join(own.public_key_dir, "key.bin"), os.path.join(client_keys, "key.bin")
    )

    def run_client(name, i, *more, expected):
        most = min(args.max_tokens, tokens_per_request - len(prompts[i]))
        options = client_options(args, own, server.url, i, prompts[i], most)
        env = {**os.environ, "VLLM_API_KEY": server.api_key}
        check_listener(server.url, server.log)
        argv = [*PYTHON, "ask_and_check", *options, *more]
        return programs.run(name, argv, args.proof_timeout, env, expected)

    say("3. Send %d prompts and ask for a proof of each reply." % len(prompts))
    served = []
    for i in range(len(prompts)):
        name = "ask_%d" % i
        run_client(name, i, "--ask-only", expected=(STORED,))
        if read_outcome(programs.log_to_read(name)) != STORED_OUTCOME:
            raise StepError(
                "the client stored no reply to prompt %d; see %s and, for the server's side, %s"
                % (i, programs.log_path(name), server.log)
            )
        with open(os.path.join(args.work_dir, "client", str(i), "served.json")) as f:
            served.append(json.load(f))
        say("   reply %d: tokens: %d" % (i, len(served[-1]["reply_token_ids"])))
    say("4. Run the proving service on the queued requests.")
    if args.cp_prove_options:
        say("   extra options for cp_prove: " + shlex.join(args.cp_prove_options))
    if args.stop_vllm_to_prove:
        say("   vLLM is stopped before that, and started again afterwards.")
        programs.stop_and_wait(STOP_TO_PROVE_S)
    prove(args, own, programs, "cp_prove")
    if args.stop_vllm_to_prove:
        server = start("vllm_again")
    say("5. Fetch and check each proof (policy: %s)." % args.policy)
    count = 0
    for i in range(len(prompts)):
        name = "check_%d" % i
        code = run_client(name, i, "--resume", "--", *policy_options(args, own), expected=None)
        line = last_line(programs.log_to_read(name), CHECK_PROOF)
        count += has_passed(args, code, read_outcome(programs.log_to_read(name)))
        say("   reply %d: %s" % (i, line or "no result; see " + programs.log_path(name)))
    say("   %d of %d proofs passed" % (count, len(prompts)))
    if count < len(prompts):
        raise StepError("not every proof passed; the logs are in " + programs.log_dir)
    return served


def changed_reply(policy, reply, vocabulary):
    """Return reply with its last (exact) or every (clipped-gap) token replaced by the next ID."""
    if policy == "exact":
        return reply[:-1] + [(reply[-1] + 1) % vocabulary]
    return [(token + 1) % vocabulary for token in reply]


def prover_declined(response, log_text):
    """Check the response to the changed reply. Return (passed, sentence); passed is None if there
    is no proof and no cause.

    The cause must be the policy: another, such as a GPU fault, shows nothing about the policy. Only
    a cp_prove that was built with test options prints it.
    """
    if response.faults:
        return False, "unexpected: " + "; ".join(response.faults)
    if not response.failed:
        return False, "unexpected: it made a proof"
    causes = re.findall(CAUSE_LINE % response.id, log_text)
    if not causes:
        return None, "no proof; its log does not give the cause, so the cause was not checked"
    if causes != [POLICY_CAUSE]:
        return False, "no proof, but for a cause other than the policy: " + ", ".join(causes)
    return True, "no proof, because the tokens do not meet the policy, as expected"


def verifier_rejected(code, line):
    """Return whether cp_verify rejected the proof itself, not a leaf file."""
    of_the_proof = check_proof.REJECTED_LEAF_FILE not in line
    return code == 1 and line.startswith(VERIFIER_NO) and of_the_proof


def negative_control(args, own, programs, config, served, tokens_per_request, vocabulary):
    """Step 6. Return whether the changed reply got no proof and was rejected, and what could not
    be checked."""
    say("6. Negative control: change a reply; it must get no proof and be rejected.")
    # A reply that fills the request would need check_proof.py's --next-token: use another.
    fitting = [
        i
        for i, s in enumerate(served)
        if len(s["prompt_token_ids"]) + len(s["reply_token_ids"]) < tokens_per_request
    ]
    if not fitting:
        say("   skipped: every reply fills its request")
        return True, ["the negative control"]
    number = fitting[0]
    prompt, reply = served[number]["prompt_token_ids"], served[number]["reply_token_ids"]
    changed = changed_reply(args.policy, reply, vocabulary)
    policy = proof_middleware.load_config(config, create_dirs=False).policies[POLICY_NAME]
    request_id, key = proof_middleware.submit(policy, prompt, changed)
    if args.stop_vllm_to_prove:
        programs.stop_and_wait(STOP_TO_PROVE_S)  # nothing below needs the server
    prove(args, own, programs, "cp_prove_control")
    with open(programs.log_to_read("cp_prove_control"), errors="replace") as f:
        declined, sentence = prover_declined(
            spool.read_response(policy.spool_dir, request_id, hashes=False, key=key), f.read()
        )
    say("   the proving service: " + sentence)

    def check(name, tokens):
        request_file = os.path.join(args.work_dir, "client", name + ".bin")
        request = spool.whole_reply(
            prompt, tokens, tokens_per_request, fill=proof_middleware.PADDING
        )
        with open(request_file, "xb") as f:
            f.write(request)
        proof_dir = os.path.join(args.work_dir, "client", str(number), "proof")
        options = [proof_dir, request_file, "--end-token", "none"]
        options += ["--reply", "%d:%d" % (len(prompt), len(tokens))]
        options += check_options(args, own) + policy_options(args, own)
        # -I: ignore the current directory and PYTHON* variables.
        argv = [sys.executable, "-I", check_proof.__file__, *options]
        code = programs.run(name, argv, args.proof_timeout, None, None)
        return code, programs.log_to_read(name)

    # The two calls differ only in the request file, so the rejection is about the changed tokens.
    rejected = False
    code, log = check("control_unchanged", reply)
    if has_passed(args, code, read_outcome(log)):
        code, log = check("control_changed", changed)
        rejected = verifier_rejected(code, last_line(log, CHECK_PROOF))
        if rejected:
            say(
                "   check_proof.py, with the same proof and the changed reply: rejected, as "
                "expected"
            )
        else:
            say(
                "   check_proof.py: unexpected: cp_verify did not reject the changed reply; see "
                "control_changed.log"
            )
    else:
        say(
            "   check_proof.py: unexpected: the unchanged reply did not pass; see "
            "control_unchanged.log"
        )
    unchecked = ["why the changed reply got no proof"] if declined is None else []
    return declined is not False and rejected, unchecked


def read_prompts(path, tokens_per_request, vocabulary):
    with open(path) as f:
        lines = [line for line in f.read().split("\n") if line]
    if not lines or not all(re.fullmatch(r"\d+(,\d+)*", line) for line in lines):
        raise StepError(
            "%s must contain one prompt per line, as token IDs separated by commas" % path
        )
    prompts = [[int(token) for token in line.split(",")] for line in lines]
    if not all(len(p) < tokens_per_request and max(p) < vocabulary for p in prompts):
        raise StepError(
            "each prompt in %s must have fewer than %d token IDs, each below %d"
            % (path, tokens_per_request, vocabulary)
        )
    return prompts


def closing_notes(args):
    """Return the limits that a run states at its end."""
    notes = []
    if args.demo:
        notes.append(
            "the demo's model is synthetic: a pass says nothing about any model that anyone "
            "serves, and nothing about weight hiding."
        )
    if args.policy == "clipped-gap":
        notes.append(
            "the clipped-gap policy bounds a mean and a count over the checked tokens, not each "
            "token. C and theta mean something only against the committed model's logit scale."
        )
    for kind in sorted(args.builds - {SERVING_BUILD}):
        # A cp_prove left in BUILD_DIR can be a build for tests alone.
        if kind:
            said = 'is not the build that is meant for serving: by its log it is "%s".'
            said %= spool.printable(kind[:100])
        else:
            said = 'did not say what kind of build it is (its log has no "setting:" line).'
        notes.append(
            "cp_prove in %s %s The build meant for serving is made by prover/build.sh run "
            "without an argument." % (args.build_dir, said)
        )
    notes.append(
        "The masking is built in and on by default; that it is enough to hide the weights is not "
        'proved for the programs. See docs/limits.md, "Weight hiding".'
    )
    return notes


def run_steps(args, programs):
    """Run all steps. Return whether the negative control passed, and what could not be checked.
    Any other failure raises StepError."""
    if args.fell_back:
        say(
            "Note: the installed integer model package has no integer_model_vllm.float_pair, which "
            "ordinary serving in the demo needs, so this run uses integer serving and exact match "
            "(--serve integer)."
        )
    if args.demo:
        own = prepare_demo(args, programs)
    else:
        say("1. Use your own commitment.")
        own = Commitment(
            **{name: getattr(args, name) for name in OWN_OPTIONS},
            limits=tuple((args.limits or "").split(",")),
        )
        directory = copy_configuration(args, own.parameter_file, own.scales_file, args.layout_file)
        own.parameter_file = os.path.join(directory, "policy.txt")
        own.scales_file = os.path.join(directory, "KL.bin")
        check_the_id(args, own)
    tokens_per_request, vocabulary = read_parameters(own.parameter_file)
    prompts = read_prompts(own.prompts_file, tokens_per_request, vocabulary)
    config = write_middleware_config(args, own)

    def start(name):
        return start_vllm(args, own, programs, config, tokens_per_request, name)

    served = serve_and_check(args, own, programs, start, prompts, tokens_per_request)
    if args.demo or args.policy == "exact":
        passed, unchecked = negative_control(
            args, own, programs, config, served, tokens_per_request, vocabulary
        )
    else:  # on a real model, a changed reply can still meet a clipped-gap policy
        say("6. Negative control: skipped under clipped-gap with your own commitment.")
        passed, unchecked = True, ["the negative control"]
    say("7. Stop vLLM." if programs.running else "7. vLLM is already stopped.")
    programs.stop_all()
    for note in closing_notes(args):
        say("Note: " + note)
    return passed, unchecked


def main(argv=None):
    args = parse_args(argv)
    errors = check_prerequisites(args)
    if errors:
        print("\n".join("error: " + text for text in errors), file=sys.stderr)
        return 2
    os.umask(0o077)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Keep the client's requests to 127.0.0.1 away from a proxy.
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    work_fd = None
    try:
        work_fd = make_work_dir(args.work_dir)
        made = os.fstat(work_fd)
        if made.st_uid != os.geteuid() or made.st_mode & 0o077:
            raise OSError(None, "it is not this user's private directory")
        for name in ("logs", "spool", "jobs", "client", "home", "tmp"):
            os.mkdir(name, 0o700, dir_fd=work_fd)
        log_fd = os.open("logs", DIR_FLAGS, dir_fd=work_fd)
    except OSError as error:
        print("error: cannot create %s: %s" % (args.work_dir, error.strerror), file=sys.stderr)
        return 2
    finally:
        if work_fd is not None:
            os.close(work_fd)
    trusted = trusted_like_root(args.work_dir)
    if trusted:
        say(
            "Note: %s belongs to the account in SUDO_UID (uid %d), and this run trusts that "
            "account like root: it could lead the run's files elsewhere. To rule that out, choose "
            "a WORK_DIR with only root's directories above it. For example: "
            'W="$(sudo mktemp -d -p /tmp)", and then "${W:?}/run" as WORK_DIR.' % trusted
        )
    programs = Programs(os.path.join(args.work_dir, "logs"), log_fd)
    passed, unchecked = False, []
    try:
        try:
            try:
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    # nohup ignores SIGHUP: leave it ignored
                    if signal.getsignal(number) != signal.SIG_IGN:
                        signal.signal(number, programs.on_signal)
                passed, unchecked = run_steps(args, programs)
            except StepError as error:
                print("error: %s" % error, file=sys.stderr)
            except Exception as error:
                print("error: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        finally:
            programs.quiet = True  # an earlier interrupt is caught below
    except Interrupted as interrupt:
        with contextlib.suppress(OSError):  # after SIGHUP the terminal may be gone
            print("interrupted; stopping the programs that this run started", file=sys.stderr)
        return 128 + interrupt.args[0]
    finally:
        programs.stop_all()
        os.close(log_fd)
    # An interrupt during the last stop_all(), or one that was lost because an error was raised
    # while it passed.
    if programs.held_back is not None:
        return 128 + programs.held_back
    if not passed:
        say("RESULT: failed")
    elif unchecked:
        say(
            "RESULT: every step that ran gave the expected result; not checked: "
            + "; ".join(unchecked)
        )
    else:
        say("RESULT: every step gave the expected result")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
