// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Common header of the programs cp_commit (prover/hash_commitment/hf_commit.cu) and cp_prove
// (prover/hash_commitment/hf_keeper.cu, the translation unit that owns the weight table, which is
// the committed encoding of all the weights): GPU headers, small helpers, and host-side headers
// inside namespace rf. Its include order matters.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include "common/numa_pin.h"
#include "common/zeroing_heap.h"
#include "common/common.h"
#include "common/element_from_hash.h"
// Replaces prover/common/common.h's CK from here on: a failed call leads to exit(5), so that
// cp_commit's exit code is not mistaken for one that reports on the data. In cp_prove the exit
// handler end_on_exit() (prover/prover/ends.h) catches the 5: it sets the name "cuda-call" for the
// program's last line, if no name is set yet, and ends the process through stop(), which chooses
// the exit code.
#undef CK
#define CK(x)                                                                                                          \
  do {                                                                                                                 \
    cudaError_t e_ = (x);                                                                                              \
    if (e_ != cudaSuccess) {                                                                                           \
      fprintf(stderr, "GPU fault: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                         \
      exit(5);                                                                                                         \
    }                                                                                                                  \
  } while (0)
#include "hiding/product_test.cuh"
#include "common/row_pass_core.cuh"
#include "common/blas.h"
#include "weights_blocks/decode.cuh"
#include "common/hash_tree.cuh"
#include <algorithm>
#include <map>
typedef int8_t i8;
typedef uint16_t u16;
// The loop form is public: the grid-stride loop: M. Harris,
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#define GSL(i, n)                                                                                                      \
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)
// Grid size for a GSL kernel; 0, an invalid grid, if n is 0.
static int gsl_blocks_of(size_t n, int t = 256, int cap = 1 << 20) {
  size_t b = (n + t - 1) / t;
  return (int)std::min<size_t>(b, cap);
}
// Writes the eq table of the point z[0..n) into o[0..2^n); z[0] goes with the index's top bit.
__global__ void k_eq(const T *z, int n, T *o) {
  GSL(x, (size_t)1 << n) {
    T v = F::one();
    for (int i = 0; i < n; i++) {
      T r = z[i];
      v = F::mul(v, ((x >> (n - 1 - i)) & 1) ? r : F::sub(F::one(), r));
    }
    o[x] = v;
  }
}
static u32 *g_th[2];
static size_t g_th_bytes[2];
// Reallocates treehash()'s two scratch buffers; a size of 0 only frees the buffer. For a list of n
// words the first must hold ceil(n / THPW) digests of 4 * DW bytes, the second the next level's.
// DW (a digest's length in 32-bit words), THPW = 16 DW and the domain-separation numbers TH_LIST
// and TH_UPPER are defined in prover/common/setting.h.
static void treehash_scratch(size_t bytes0, size_t bytes1) {
  const size_t want[2] = {bytes0, bytes1};
  for (int p = 0; p < 2; p++) {
    // treehash() may have left digests of secret data in the old buffer.
    if (g_th[p]) {
      CK(cudaMemset(g_th[p], 0, g_th_bytes[p]));
      cudaFree(g_th[p]);
    }
    g_th[p] = nullptr;
    if (want[p]) CK(cudaMalloc(&g_th[p], want[p]));
    g_th_bytes[p] = want[p];
  }
}
// Returns the tree hash of the n >= DW words at `in` (GPU memory); a shorter list has none, as in
// the verifier, and here leads to exit(2). The digest is in a scratch buffer until the next call.
// The launches go to the calling thread's default stream (prover/build.sh compiles with
// --default-stream per-thread) and are neither checked nor waited for.
// The list's chunks of THPW words are hashed under TH_LIST, and each level of digests under
// TH_UPPER, until a level is one digest. A list of DW words is hashed once too. Only
// prover/tests/tree_hash_test.cu calls this treehash(); cp_commit and cp_prove only allocate its
// scratch buffers, with treehash_scratch(). The treehash() that cp_prove's main translation unit
// (prover/prover/cp_prove.cu) calls is in prover/prover/base.cuh.
static const u32 *treehash(const u32 *in, size_t n) {
  if (n < DW) {
    fprintf(stderr,
            "internal error: the tree hash of %zu words was requested; it is defined for %d words or more; the program "
            "ends\n",
            n, DW);
    exit(2);
  }
  int p = 0;
  u32 num = TH_LIST;
  for (;;) {
    const size_t nc = (n + THPW - 1) / THPW;
    if (nc * 4 * DW > g_th_bytes[p]) {
      fprintf(stderr, "treehash: the scratch is too small for a level of %zu words\n", n);
      exit(4);
    }
    k_th<<<(int)((nc + 127) / 128), 128>>>(in, n, g_th[p], num);
    in = g_th[p];
    p ^= 1;
    n = nc * DW;
    if (nc == 1) return in;
    num = TH_UPPER;
  }
}

#include "hiding/token_sumcheck.cuh"
#include "hiding/trees.cuh"
#include "hiding/eqs.cuh"
#include "weights_commitment/kernels.cuh"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <initializer_list>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
// Namespace rf holds what the two host-side headers below define (field arithmetic, the tree hash
// TH(); the layout of the committed weight matrix W). Files of
// prover/hash_commitment and prover/weights_commitment reopen it for their own code.
// Some names in the host-side headers also exist at global scope, hence namespace rf. What these
// headers include that does not belong in rf must already be included above.
namespace rf {
#include "common/host_common.h"
}
namespace rf {
#include "common/weight_layout.h"
} // namespace rf
