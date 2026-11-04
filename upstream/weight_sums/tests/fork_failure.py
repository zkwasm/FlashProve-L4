#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Started by weight_sums/tests/run_tests.sh, which the runner starts; by hand: below a directory of your own.
# Runner: run_suite.py. <new folder> goes there: the test does not check the directories above it.
"""Test that weight_sums.py fails cleanly when the system cannot start its second worker process.

usage: fork_failure.py <new folder> <path of weight_sums.py>

The script, weight_sums/weight_sums.py, hashes a model's tensors in worker processes, no more
workers than tensors. The test runs it with --workers 2 on a valid GGUF file (a model shard) with
two tensors, in an interpreter in which the second call of os.fork fails, once with EAGAIN (process
limit reached) and once with ENOMEM (out of memory). The script must print its error message, exit
with status 26 within the time limit, and leave no new file in the folder and no live process in its
process group. The first worker is already running then, and an exit that waited for it would hang:
the script exits without waiting, and the worker exits once the script has exited.

The test creates <new folder> and leaves it there; it signals only its own runs' process groups.
The exit status is 0 if both runs end as required, 1 if not, and 2 for a wrong command line.
"""

import os
import signal
import struct
import subprocess
import sys
import threading
import time

SECONDS = 60  # a run that takes longer is killed

# Runs before the script in the same interpreter: the second os.fork fails with errno sys.argv[1].
# This works because weight_sums.py, in convert_all(), names multiprocessing's start method `fork`.
BEFORE = """
import errno, os, runpy, sys
inner, calls, number = os.fork, [0], getattr(errno, sys.argv[1])
def fork():
    calls[0] += 1
    if calls[0] == 2:
        raise OSError(number, os.strerror(number))
    return inner()
os.fork = fork
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def string(text):
    return struct.pack("<Q", len(text)) + text


def shard():
    """Return a GGUF file's bytes: two F32 tensors of four elements each, which need no norm tensor.

    The comment at the top of ggml/include/gguf.h describes the format (llama.cpp as carried in llama-cpp-python 0.3.7,
    whose changelog names the commit 794fe23f29fb40104975c91fe19f23798f7c726e).
    """
    # Value types: 8 string, 4 uint32. Tensor type 0: F32.
    pairs = [
        string(b"general.architecture") + struct.pack("<I", 8) + string(b"llama4"),
        string(b"llama4.embedding_length") + struct.pack("<II", 4, 5120),
        string(b"llama4.block_count") + struct.pack("<II", 4, 1),
    ]
    head = b"GGUF" + struct.pack("<IQQ", 3, 2, len(pairs)) + b"".join(pairs)
    for number, name in enumerate((b"blk.0.attn_output.weight", b"blk.0.ffn_down.weight")):
        head += string(name) + struct.pack("<IQIQ", 1, 4, 0, 32 * number)
    head += bytes(-len(head) % 32)
    return head + 2 * (struct.pack("<4f", 1, -2, 3, 0.5) + bytes(16))


def members(group, leader):
    """Return the IDs of the group's live processes other than its leader, read from /proc."""
    found = []
    for name in os.listdir("/proc"):
        if name.isdigit() and int(name) != leader:
            try:
                with open("/proc/%s/stat" % name, encoding="utf-8", errors="replace") as f:
                    state, _, pgrp = f.read().rsplit(")", 1)[1].split()[:3]  # state, parent, group
                if int(pgrp) == group and state != "Z":  # skip zombies: exited, not yet reaped
                    found.append(int(name))
            except (OSError, ValueError, IndexError):
                pass
    return found


def ended_within(process, seconds):
    """Wait up to `seconds` for the process to exit, without reaping it, and return whether it did.
    Until it is reaped, its ID, which is also its group's, cannot be reused, so a signal to that
    group reaches only this run's processes."""
    until = time.monotonic() + seconds
    while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
        if time.monotonic() >= until:
            return False
        time.sleep(0.05)
    return True


def main(words):
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # an inherited SIG_IGN would auto-reap the script
    if len(words) != 2:
        print(__doc__.split("\n")[2], file=sys.stderr)
        return 2
    folder, script = words
    os.mkdir(folder, 0o700)  # fails if the name is taken; closed to other users
    with open(os.path.join(folder, "two.gguf"), "wb") as file:
        file.write(shard())
    wrong = []
    for error in ("EAGAIN", "ENOMEM"):
        # The script gets its own process group, so that its worker can be found and killed with it.
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", BEFORE, error, script, "--workers", "2", "--out", "out.txt", "two.gguf"],
            cwd=folder, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        got = {}
        readers = [threading.Thread(target=lambda n, p: got.__setitem__(n, p.read()), args=(n, p), daemon=True)
                   for n, p in (("printed", process.stdout), ("messages", process.stderr))]  # fmt: skip
        for reader in readers:
            reader.start()
        # As in run_suite.py: wait without reaping, signal the group, then reap. See ended_within().
        start = time.monotonic()
        ended = ended_within(process, SECONDS)
        for reader in readers:  # until end of file on both pipes, when no process holds them open
            reader.join(max(0, SECONDS - (time.monotonic() - start)) if ended else 0)
        left = members(process.pid, process.pid)
        os.killpg(process.pid, signal.SIGKILL)  # unconditional: a scan of /proc can miss a process
        process.wait()
        for reader in readers:
            reader.join(10)
        printed, messages = got.get("printed", b""), got.get("messages", b"")
        if not ended:
            wrong.append(f"{error}: it did not end within {SECONDS} seconds")
            continue
        if left:
            wrong.append(f"{error}: a process was left")
        if process.returncode != 26:
            wrong.append(f"{error}: the exit status is {process.returncode} and not 26")
        if b"cannot start the worker processes" not in messages or b"Traceback" in messages:
            wrong.append(f"{error}: standard error is not as expected: {messages.decode(errors='replace')}")
        if printed:
            wrong.append(f"{error}: something is on standard output")
        if sorted(os.listdir(folder)) != ["two.gguf"]:
            wrong.append(f"{error}: the folder holds {sorted(os.listdir(folder))}")
    for line in wrong:
        print("fork_failure.py: " + line)
    print(f"fork_failure.py: 2 runs, {len(wrong)} things not as expected")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
