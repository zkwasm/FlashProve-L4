#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own, since the test does not check the
# directories above <new work directory>. (Runner: run_suite.py. No row of SUITE.md starts this
# test.)
"""Tests that cp_commit rejects a GPU that another program is using. cp_commit sizes its work by the
GPU's free memory, so either program can run out of memory midway. If more than 2 GiB are in use at
its start, its own share included, it exits with status 2 ("refused (gpu-in-use)") before it draws
a seed or writes a file, except under -test-seed (test build only).

    prover/tests/gpu_in_use.py <build directory> <configuration directory> <new work directory>

<build directory> contains cp_commit from the production build (prover/build.sh) and
test_build/cp_commit from the test build (prover/build.sh tests), each with cp_verify next to it.
<configuration directory> is prover/examples/target_small_f2_roomy. The weights are synthetic.
Needs libcuda.so.1 and a GPU that nothing else uses during the test: if more than 1 GiB is in use
at the start, the test runs nothing.
Prints "gpu_in_use: 4 scenes, all as expected" and exits 0, or says what differs and exits 1.
No process is killed: the other program on the GPU is this test's child, which exits when its
standard input closes; the driver then frees its memory.

The cases (`scene` in code and output), by printed name. The child must outlive cp_commit.
  beside_4_gib               cp_commit, which draws its seed, while the child holds 4 GiB: exit
                             status 2; the last line on standard error begins
                             "cp_commit: refused (gpu-in-use): "; no key directory (-ivk, -isk)
                             exists, finished or unfinished (.making)
  beside_4_gib_a_tests_seed  test_build/cp_commit -test-seed, next to 4 GiB: makes its commitment
                             and does not print the rejection name
  beside_256_mib             the first command, next to 256 MiB: the same outcome
  alone                      the first command, no child: the same

Not shown: that no seed was drawn before the rejection (the first case checks only that none of the
four directory names exists afterwards); whether the test seed or the test build lets the second
case go on (no case runs the test build next to 4 GiB with a drawn seed); cp_prove's rejection of
the same name; a process that comes to the GPU after the check, or that has not yet taken its
memory; the limit of 2 GiB itself, which no case comes near.
"""

import os
import subprocess
import sys

GIB = 1 << 30
REFUSAL = "cp_commit: refused (gpu-in-use): "  # prover/hash_commitment/hf_commit.cu
# The child prints how many bytes of GPU memory are in use, its own share included. If argv[1] is
# not 0, it allocates argv[1] bytes first, and does not exit until its standard input closes.
CHILD = """
import ctypes, sys
cuda = ctypes.CDLL("libcuda.so.1")
device, context, held = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
free, total, take = ctypes.c_size_t(), ctypes.c_size_t(), int(sys.argv[1])
if (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0)
        or cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) or cuda.cuCtxSetCurrent(context)):
    sys.exit("the GPU cannot be opened")
if take and cuda.cuMemAlloc_v2(ctypes.byref(held), ctypes.c_size_t(take)):
    sys.exit("%d bytes of the GPU's memory cannot be allocated" % take)
if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)):
    sys.exit("the GPU's free memory cannot be read")
print(total.value - free.value, flush=True)
if take:
    sys.stdin.read()
"""


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    build, config, work = (os.path.abspath(x) for x in sys.argv[1:])
    child = [sys.executable, "-I", "-c", CHILD]
    looked = subprocess.run(child + ["0"], stdout=subprocess.PIPE, text=True)
    if looked.returncode or int(looked.stdout) > GIB:
        sys.exit("gpu_in_use: this test needs a GPU that nothing else uses, and %s: nothing was run"
                 % ("the GPU's memory cannot be read" if looked.returncode
                    else "%.1f GB of this one's memory are in use" % (int(looked.stdout) / 1e9)))  # fmt: skip
    os.makedirs(work, mode=0o700)  # it fails if the directory exists
    with open(os.path.join(work, "seed.bin"), "wb") as f:
        f.write(bytes(range(1, 33)))  # made up, and none of the seeds that are written in the source
    differs = []

    def scene(name, held, program, refused, *more):
        """Runs `program` while a child holds `held` bytes of GPU memory, and checks the outcome."""
        ivk, isk = os.path.join(work, name + ".ivk"), os.path.join(work, name + ".isk")
        holder, before, beside = None, "0", False
        if held:
            holder = subprocess.Popen(child + [str(held)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            before = holder.stdout.readline().strip()
            if not before.isdigit():
                holder.wait()
                sys.exit("gpu_in_use: %s: the child could not allocate %d bytes of the GPU's memory" % (name, held))
        try:
            ran = subprocess.run([os.path.join(build, program), "-weights", "made-up-blocks", "-config", config, "-M", "1",
                                  "-ivk", ivk, "-isk", isk, *more], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # fmt: skip
        finally:
            if holder:
                beside = holder.poll() is None  # the child was still there when cp_commit ended
                holder.stdin.close()
                holder.wait()
        stand = [x for x in (ivk, isk, ivk + ".making", isk + ".making") if os.path.lexists(x)]
        errors = ran.stderr.splitlines()
        if refused:
            good = ran.returncode == 2 and bool(errors) and errors[-1].startswith(REFUSAL) and not stand
        else:
            good = ran.returncode == 0 and stand == [ivk, isk] and "(gpu-in-use)" not in ran.stdout + ran.stderr
        good = good and (not held or beside)
        print("%s %s: %s; exit status %d; %s"
              % ("as expected" if good else "DIFFERS", name, "the child saw %d MiB in use%s" % (int(before) >> 20, "" if beside else " and ended too early") if held else "no child",
                 ran.returncode, errors[-1][:200] if refused and errors else "%d of the four names exist" % len(stand)))  # fmt: skip
        if not good:
            differs.append(name)
            print(ran.stdout[-600:] + ran.stderr[-600:])

    scene("beside_4_gib", 4 * GIB, "cp_commit", True)
    scene("beside_4_gib_a_tests_seed", 4 * GIB, "test_build/cp_commit", False, "-test-seed", os.path.join(work, "seed.bin"))
    scene("beside_256_mib", GIB // 4, "cp_commit", False)
    scene("alone", 0, "cp_commit", False)
    print("gpu_in_use: 4 scenes, " + ("these differ: " + ", ".join(differs) if differs else "all as expected"))
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
