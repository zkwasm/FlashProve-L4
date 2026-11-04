// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Two fault-injection hooks of the committer program cp_commit. Each is a test-only environment
// variable, read with TESTENV (prover/common/setting.h), that changes the integers committed to.
//
//   FORMF_PLANTED_INTEGER=<unit>:<value>[:past-the-scan]   one committed integer has another value
//   FORMF_GENERATED_VALUES=<a>[:<b>[:<unit>]]              generated integers are a, or a or b
//
// A production build (macro SERVING_BUILD) compiles nothing of the hooks, not even their names and
// messages: there PlantedInteger and GeneratedValues are structs whose functions do nothing.
// prover/build.sh does not install a production program that contains a line of
// prover/tests/serving_build_words.txt, which lists these strings.
//
// cp_commit exits with code 2 if any test variable, as defined in prover/common/test_words.h, is
// set, unless it is a test build that runs with -test-seed (a seed read from a file) on synthetic
// weights (-weights generated or made-up-blocks). The output of such a run is only for tests. The
// hooks do not check this: their constructors read the variables, so cp_commit_main()
// (prover/hash_commitment/hf_commit.cu) checks before it constructs the objects. Other test
// variables of cp_commit are read in prover/weights_commitment/cert_doors.h or, like
// ROWS_COUNT_FAULT, in the files where they act.
//
// Include this file after prover/hash_commitment/hf_commit_walk.h, which defines Piece, and after a
// definition of the CUDA error-check macro CK (prover/weights_commitment/common.cuh).
#pragma once
#include "common/setting.h"
#include "hash_commitment/hc_format.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hf {
// FORMF_PLANTED_INTEGER=<unit>:<value> replaces the first integer (a quantized weight) of unit
// <unit>: index 0 of its first tensor in the model file's order, which is input 0 of output row 0
// of expert 0. A unit (Piece::unit) is an entry of the layout, with one or two tensors.
// PinnedWeights::walk_on() (hf_commit_weights.cuh) calls lay() in every pass over the weights
// (`walk` in identifiers), so that, without :past-the-scan, every pass gets the same model.
// make_plan() (hf_commit.cu) rejects the hook, exit code 2, unless its plan keeps all of the
// committed matrix in GPU memory. In the other plan k_slab_cells() (hf_commit_windows.cuh) builds
// the matrix's rows from the stored tensors, without a pass; it applies the value too, but the
// hook is rejected there all the same.
//
// cp_commit's range scan (run by check_the_weights() in hf_commit.cu) rejects a value outside the
// range that the parameter file (policy.txt) declares for the unit's kind of tensor (for the
// embedding: above 2^30 - 1 in magnitude): exit code 16, no commitment written. With :past-the-scan
// the scan's pass (`checking_walk`) does not get the value, so an out-of-range value passes the
// scan and is committed to, as by a malicious committer. The run is then expected to fail later: at
// one of cp_commit's own checks while it builds the key file (key.bin), or when the verifier
// program cp_verify, which cp_commit runs with -key, rejects that file's range proof. No test in
// this repository sets :past-the-scan. The other hook for a malicious committer (commit_lie.cuh),
// empty in this repository, is not called in the scan's pass either.
#ifdef SERVING_BUILD
struct PlantedInteger {
  PlantedInteger() {} // user-provided, so a const data member of this type needs no initializer
  void lay(const Piece &, bool, bool, size_t, cudaStream_t) const {}
  bool acts_on(size_t) const { return false; }
};
#else
struct PlantedInteger {
  bool on = false, past_the_scan = false;
  size_t unit = 0;
  long long value = 0;
  // lay() copies from these fields asynchronously, so the object must outlive the copies.
  int8_t digit[2] = {0, 0};
  int16_t narrow = 0;
  int32_t wide = 0;
  PlantedInteger() {
    const char *word = TESTENV("FORMF_PLANTED_INTEGER");
    if (!word) return;
    char *rest = nullptr, *end = nullptr;
    unit = strtoull(word, &rest, 10);
    if (rest != word && *rest == ':') value = strtoll(rest + 1, &end, 10);
    if (!end || end == rest + 1 || (*end && strcmp(end, ":past-the-scan")) || value < INT32_MIN || value > INT32_MAX) {
      fprintf(stderr, "cp_commit: FORMF_PLANTED_INTEGER takes <unit>:<value> or <unit>:<value>:past-the-scan, and the "
                      "value has at most 32 bits\n");
      exit(2);
    }
    past_the_scan = *end != 0;
    // The byte planes Piece::lo and Piece::hi contain balanced digits, each from -128 to 127:
    // value = digit[0] + 256 * digit[1]. So lay() allows an int16 unit no value above 32767 - 128.
    // Its lower limit, -32768 + 128, is stricter than the digits require.
    const long long low = ((value + 128) % 256 + 256) % 256 - 128;
    digit[0] = (int8_t)low, digit[1] = (int8_t)((value - low) / 256);
    narrow = (int16_t)value, wide = (int32_t)value;
    on = true;
  }
  // True if the hook changes an integer of this unit. PinnedWeights::blocks() then returns false,
  // because the hook does not change the stored bytes that blocks() would return.
  bool acts_on(size_t unit_) const { return on && unit_ == unit; }
  // Overwrites the integer in GPU memory. Call it for each Piece of a pass, after the work that
  // fills it is queued on `stream` and before the Visit callback. is_wide: int32, not int16.
  void lay(const Piece &piece, bool is_wide, bool checking_walk, size_t units, cudaStream_t stream) const {
    if (!on) return;
    if (unit >= units) {
      fprintf(stderr, "cp_commit: FORMF_PLANTED_INTEGER: the layout has %zu units\n", units);
      exit(2);
    }
    if (piece.unit != unit || piece.first != 0 || (checking_walk && past_the_scan)) return;
    if (!is_wide && (value < -32768 + 128 || value > 32767 - 128)) {
      fprintf(stderr, "cp_commit: FORMF_PLANTED_INTEGER: unit %zu holds integers of 16 bits\n", unit);
      exit(2);
    }
    if (piece.lo) {
      CK(cudaMemcpyAsync((void *)piece.lo, &digit[0], 1, cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync((void *)piece.hi, &digit[1], 1, cudaMemcpyHostToDevice, stream));
    } else if (is_wide)
      CK(cudaMemcpyAsync((void *)piece.w0, &wide, 4, cudaMemcpyHostToDevice, stream));
    else
      CK(cudaMemcpyAsync((void *)piece.w0, &narrow, 2, cudaMemcpyHostToDevice, stream));
  }
};
#endif

// FORMF_GENERATED_VALUES=<a>[:<b>[:<unit>]], only with -weights generated: every integer of the
// named unit, or of all units, is a; with b it is a or b, chosen by one bit of a hash of its index
// and unit. The hook acts once, before the generator's integers are stored in host memory, so every
// later reader gets the same model. It is for tests that check that runs with one layout and very
// different weights (all equal; two values) make the same kernel launches and copies; no test in
// this repository sets it. The range scan rejects out-of-range values as for FORMF_PLANTED_INTEGER.
#ifdef SERVING_BUILD
struct GeneratedValues {
  GeneratedValues() {}
  void hold(bool) const {}
  void lay(size_t, void *, size_t, bool) const {}
};
#else
// The loop form is public: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops",
// NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// The hash is the multiplier of Fibonacci hashing (D. E. Knuth, "The Art of Computer Programming",
// volume 3) and a shift: no more is asked of it than that both values occur.
template <class T>
__global__ void k_generated_values(T *out, size_t n, long long a, long long b, uint64_t salt) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    out[i] = (T)((((i + salt) * 0x9E3779B97F4A7C15ull) >> 40 & 1) ? b : a);
}
struct GeneratedValues {
  bool on = false, one_unit = false;
  long long a = 0, b = 0;
  size_t unit = 0;
  GeneratedValues() {
    const char *word = TESTENV("FORMF_GENERATED_VALUES");
    if (!word) return;
    char *end = nullptr;
    a = b = strtoll(word, &end, 10);
    bool good = end != word;
    if (good && *end == ':') {
      const char *second = end + 1;
      b = strtoll(second, &end, 10);
      good = end != second;
    }
    if (good && *end == ':') {
      const char *third = end + 1;
      unit = strtoull(third, &end, 10), one_unit = true;
      good = end != third && *third >= '0' && *third <= '9';
    }
    if (!good || *end || a < INT32_MIN || a > INT32_MAX || b < INT32_MIN || b > INT32_MAX) {
      fprintf(stderr, "cp_commit: FORMF_GENERATED_VALUES takes <a>, <a>:<b> or <a>:<b>:<unit>, and a value has at most 32 bits\n");
      exit(2);
    }
    on = true;
  }
  // Exits with code 2 if the hook is on without -weights generated.
  void hold(bool generator) const {
    if (!on || generator) return;
    fprintf(stderr, "cp_commit: FORMF_GENERATED_VALUES goes with -weights generated alone\n");
    exit(2);
  }
  // Overwrites n integers at `out` (GPU), on cudaStreamPerThread: launch the generator there first.
  void lay(size_t unit_, void *out, size_t n, bool is_wide) const {
    if (!on || (one_unit && unit_ != unit)) return;
    if (!is_wide && (a < -32768 || a > 32767 || b < -32768 || b > 32767)) {
      fprintf(stderr, "cp_commit: FORMF_GENERATED_VALUES: unit %zu holds integers of 16 bits\n", unit_);
      exit(2);
    }
    if (is_wide)
      k_generated_values<int><<<8192, 256, 0, cudaStreamPerThread>>>((int *)out, n, a, b, (uint64_t)unit_ << 40);
    else
      k_generated_values<short><<<8192, 256, 0, cudaStreamPerThread>>>((short *)out, n, a, b, (uint64_t)unit_ << 40);
    CK(cudaGetLastError());
  }
};
#endif
} // namespace hf
