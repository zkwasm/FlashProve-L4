#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Checks the CUDA compiler's version. This repository requires nvcc 13.2.86 (CUDA 13.2 Update 2).
# An older one fails this check; scripts that compile GPU code run the check and stop if it fails.
#
# Usage: prover/check_nvcc_version.sh [NVCC]
#   NVCC   the compiler to check; default: the nvcc found on PATH
# Exit code 0: the compiler reports version 13.2.86 (no output) or a later one (one line on
# standard error: a later release passes this check and is not supported).
# Exit code 1, one line on standard error: no such program is found, "nvcc --version" fails or does
# not print exactly one line "Cuda compilation tools, release X.Y, VX.Y.Z", or X.Y.Z is older.
#
# Why this version. The CUDA 13.2 Update 2 release notes ("Resolved Issues", "CUDA Compiler")
# tell of a fault in the compiler that came in with CUDA 12.8 and that this release repairs: where
# threads diverge inside a branch that is itself divergent, a register can end up with a wrong value.
# An affected kernel can compute a wrong result without any error message. The component table in
# the same notes lists nvcc 13.2.86.
#
# "nvcc --version" runs in an environment that contains only PATH, so that NVCC_PREPEND_FLAGS and
# NVCC_APPEND_FLAGS, whose contents nvcc adds to its command line, do not reach the compiler.
# tools() in prover/build.sh runs the command in the same way when it writes a build record.
#
# Limitations: the reported version is not verified. A compiler that passes this check may have
# other bugs. The same release notes list open ones under "Known Issues".
set -u
nvcc=${1:-nvcc}
fail() {
  echo "check_nvcc_version.sh: cannot build: this repository requires nvcc 13.2.86 (CUDA 13.2 Update 2), but $1." \
    "Install that toolkit and put its bin directory first on PATH." >&2
  exit 1
}
path=$(command -v -- "$nvcc") || fail "no program named $nvcc was found (PATH is $PATH)"
output=$(env -i "PATH=$PATH" "$path" --version 2>&1) || fail "$path --version failed"
version=$(sed -n 's/^Cuda compilation tools, release [0-9.]*, V\([0-9.]*\)$/\1/p' <<<"$output")
[[ $version =~ ^([0-9]{1,4})\.([0-9]{1,4})\.([0-9]{1,6})$ ]] ||
  fail "$path --version does not print exactly one line of the form \"Cuda compilation tools, release X.Y, VX.Y.Z\""
major=$((10#${BASH_REMATCH[1]})) minor=$((10#${BASH_REMATCH[2]})) patch=$((10#${BASH_REMATCH[3]}))
if ((major < 13 || (major == 13 && (minor < 2 || (minor == 2 && patch < 86))))); then
  fail "$path reports version $version; compilers from CUDA 12.8 up to that release have a published bug that can silently produce wrong results"
fi
if ((major != 13 || minor != 2 || patch != 86)); then
  echo "check_nvcc_version.sh: note: $path reports version $version; this repository has been built and tried with 13.2.86. A later release is let through, but it is not supported (prover/docs/build.md)." >&2
fi
exit 0
