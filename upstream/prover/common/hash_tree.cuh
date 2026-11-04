// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// GPU code for the row commitments of prover/hiding/. tables() returns the twiddle factors of the
// encoding transform, whose layers are in prover/common/code_layers.cuh. They must equal those of
// the CPU reference, namespace row_commit_ref in prover/common/hash_tree_ref.h: a test not part of this
// repository compares the tables. node() computes a vertex of a Merkle tree, with put_digest();
// nothing calls it, and no test compares its digests with any other program's.
// H(number, t, x; words) takes the steps of earlier_hash_h_digest() of prover/common/hash.h, with num
// = tree_f(number, t) and x as the word c1 of every call: number separates the hash's uses, t is the
// number of the tree's kind (see TREE_UW there), and x is the computed node's place, its heap number
// (root 1; children of x: 2 x, 2 x + 1).
// DW, SW and BLKW are the lengths of a digest, a hash state and a hash block in 32-bit words
// (prover/common/setting.h).
// Nothing here generates randomness.
#pragma once
#include <cuda_runtime.h>
#include <map>
#include "hiding/place.cuh"
#include "common/hash_tree_ref.h"
namespace row_tree {
// If the CUDA call x fails: prints the error, records "cuda-call" as the reason unless one is
// recorded already (name_the_end(), prover/common/end_name.h), and exits with code 1.
#define ROW_TREE_CK(x)                                                                                                      \
  do {                                                                                                                 \
    cudaError_t e_ = (x);                                                                                              \
    if (e_ != cudaSuccess) {                                                                                           \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                                  \
      name_the_end("cuda-call");                                                                                       \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)
// The field's modulus,
// p = 2^64 - 2^32 + 1 (prover/field/field_names.h).
constexpr u64 P = FIELD_P;
typedef uint16_t u16;
// Writes the digest, the first DW 32-bit words of the hash state cv, to out, which must be aligned
// to 8 bytes (not checked).
// The idea is public: a vectorized store through a recast pointer: "CUDA Pro Tip: Increase
// Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// uint2: a built-in vector type, aligned to 8 bytes (NVIDIA, CUDA C++ Programming Guide, "Built-in
// Vector Types").
// See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions", "Built-in Types",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// The compiler hint is public: #pragma unroll: the same page of the CUDA Programming Guide, "#pragma unroll"; once
// the loop is unrolled, each subscript of the caller's cv[] is known at compile time, so cv[] can stay in registers:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
__device__ __forceinline__ void put_digest(u32 *out, const u32 cv[SW]) {
#pragma unroll
  for (int i = 0; i < DW / 2; i++) ((uint2 *)out)[i] = make_uint2(cv[2 * i], cv[2 * i + 1]);
}
// Computes the parent of two nodes of a tree: out = H(number, t, x; l | r), which takes one call of
// earlier_hash_c(), the hash's compression function (prover/common/hash.h). f = tree_f(number, t); x is the
// parent's place. l, r and out must be aligned to 8 bytes (not checked).
// The idea is public: vectorized loads through a recast pointer, for which the data must be aligned: "CUDA Pro Tip:
// Increase Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The compiler hint is public: #pragma unroll on loops over a small per-thread array: the CUDA
// Programming Guide and Milakov's post (cited at put_digest()).
__device__ __forceinline__ void node(const u32 *l, const u32 *r, u32 *out, u32 f, u32 x) {
  static_assert(2 * DW <= BLKW && DW % 2 == 0, "two digests must fit in one block, and a digest is read and written in "
                                               "pairs of words");
  u32 m[BLKW], cv[SW];
#pragma unroll
  for (int i = 0; i < DW / 2; i++) {
    const uint2 a = ((const uint2 *)l)[i], b = ((const uint2 *)r)[i];
    m[2 * i] = a.x, m[2 * i + 1] = a.y, m[DW + 2 * i] = b.x, m[DW + 2 * i + 1] = b.y;
  }
#pragma unroll
  for (int i = 2 * DW; i < BLKW; i++) m[i] = 0;
  earlier_hash_initial_state(cv);
  earlier_hash_c(cv, m, 4 * 2 * DW, x, 2 * DW, f, true);
  put_digest(out, cv);
}
// Roots of unity: root[b] = r^(2^(31 - b)), of order 2^(b + 1), for r = 7^(5 (2^32 - 1)).
struct Squares {
  u64 root[24];
};
// Fills tw with the n = 2^ln twiddle factors for plain Reed-Solomon on the coset 7 H (H: the n-th
// roots of unity), as row_commit_ref::Code::init() (prover/common/hash_tree_ref.h) defines them. Entry 0 is
// 0. Entry 2^d + hh belongs to layer i = ln - 1 - d of the transform and is z^(2^i), where z is the
// evaluation point of position hh 2^(i + 1). Every entry is canonical, so the table equals init()'s
// byte for byte.
__global__ void k_tables(const Squares g, int ln, u64 *__restrict__ tw) {
  const size_t n = (size_t)1 << ln, t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= n) return;
  u64 w = 0;
  if (t) {
    const int d = 63 - __clzll((long long)t), i = ln - 1 - d;
    const size_t hh = t - ((size_t)1 << d);
    w = 7;
    for (int ip = i + 1; ip < ln; ip++)
      if ((hh >> (ip - i - 1)) & 1) w = field_mul(w, g.root[ip]);
    for (int a = 0; a < i; a++) w = field_mul(w, w);
  }
  tw[t] = w;
}
// Returns the twiddle factors for code length 2^ln (1 <= ln <= 24), in GPU memory. The first call
// for a length in each translation unit builds a table, which is never freed; on return it is
// complete for every stream. A first call allocates and waits for the calling thread's default
// stream (prover/build.sh passes --default-stream per-thread): never make it while that stream is
// being captured or while another thread is in tables(). Any other ln: exit code 4.
// The table is allocated through hz::alloc_hook (by default cudaMalloc; a program can set its own
// allocator) as hz::OWN, memory that no other buffer reuses (prover/hiding/place.cuh).
static const u64 *tables(int ln) {
  static std::map<int, u64 *> cache;
  auto it = cache.find(ln);
  if (it != cache.end()) return it->second;
  if (ln < 1 || ln > 24) {
    printf("row_tree: internal error: a table of length 2^%d was requested; lengths 2^1 to 2^24 are supported\n", ln);
    exit(4);
  }
  Squares squares{};
  u64 root = row_commit_ref::power(7, 5 * 4294967295ull); // of order 2^32
  for (int b = 31; b >= 0; b--, root = row_commit_ref::mul(root, root))
    if (b < 24) squares.root[b] = root;
  if (squares.root[0] != P - 1 || root != 1) {
    printf("row_tree: the transform's generator does not have the expected order\n");
    exit(1);
  }
  const size_t n = (size_t)1 << ln;
  u64 *d;
  ROW_TREE_CK(hz::alloc_hook((void **)&d, n * 8, hz::OWN));
  // The launch is checked by its return value: cudaGetLastError() gives the thread's last error,
  // which can be an earlier, tolerated one (NVIDIA, CUDA Runtime API reference, Error Handling).
  void *arguments[] = {&squares, &ln, &d};
  ROW_TREE_CK(cudaLaunchKernel((const void *)k_tables, dim3((unsigned)((n + 255) / 256)), dim3(256), arguments, 0, 0));
  ROW_TREE_CK(cudaStreamSynchronize(0)); // waits for the launch above, which went to the default stream
  cache[ln] = d;
  return d;
}
} // namespace row_tree
