// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The rounds over the features (SCH, described in prover/head/claims.cuh) in the output head's
// sumcheck over token positions and features. Include after prover/hiding/token_sumcheck.cuh.
#pragma once
/* The lam rounds over the feature index kk, highest bit first. The transcript *tr (`Tr`,
   prover/hiding/transcript.cuh) holds the state of a token sumcheck, which must have
   TscShape::more == lam and have run its `before` rounds over the token positions. A and B (2^lam
   elements each) are folded in place; at the end A[0] and B[0] are the two end values: A and B at
   the lam challenges. Round i's polynomial is g(X) = (1 / H) sum_(x < H) A(X, x) B(X, x), where
   H = 2^(lam - 1 - i): each bit still free has the weight 1/2. `sc_door::round`
   (prover/hiding/sumcheck_door.cuh) gets g(0) and the coefficient of X^2, sends the round and
   returns the challenge coins[i]. Launch one block of 1024 threads (not checked).
   Also not checked: the sumcheck's running claim must be 2^-lam sum_kk A[kk] B[kk];
   `sc_door::round` computes the coefficient of X from the claim. */
// Block reduction by warp shuffles and shared memory, as in k_tsc_msg()
// (prover/hiding/token_sumcheck.cuh), here two sums at a time. The idea is public:
// J. Luitjens, "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014, "Block Reduce"
// and "Interleaving Multiple Reductions",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/. Here thread 0 adds the
// warps' sums in a loop; in the post the first warp reduces them.
// The idea is public: one thread's value (r_all) broadcast through shared memory: Luitjens, "Block
// Reduce" (cited above); M. Harris, "Optimizing CUDA", NVIDIA Developer Technology, slides
// headed "S05: High Performance Computing with CUDA", slide "Take Advantage of Shared Memory" (the
// address runs over two lines),
// https://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/Cours09/CUDA/
// SC07_CUDA_5_Optimization_Harris.pdf.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void __launch_bounds__(1024) k_head_rounds(T *A, T *B, int lam, int before, Tr *tr, T *coins) {
  __shared__ T sh[2][32], r_all;
  Tr t;
  T half = F::zero(); // used by thread 0 only
  if (threadIdx.x == 0) t = *tr, half = F::inv(F::small(2));
  for (int i = 0; i < lam; i++) {
    const int H = 1 << (lam - 1 - i);
    T g0 = F::zero(), g2 = F::zero();
    for (int x = threadIdx.x; x < H; x += 1024) {
      const T a0 = A[x], a1 = A[x + H], b0 = B[x], b1 = B[x + H];
      g0 = F::add(g0, F::mul(a0, b0));
      g2 = F::add(g2, F::mul(F::sub(a1, a0), F::sub(b1, b0)));
    }
    for (int s = 16; s; s >>= 1) {
      g0 = F::add(g0, shfl_down<F>(g0, s));
      g2 = F::add(g2, shfl_down<F>(g2, s));
    }
    if ((threadIdx.x & 31) == 0) {
      sh[0][threadIdx.x >> 5] = g0;
      sh[1][threadIdx.x >> 5] = g2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T q[2] = {F::zero(), F::zero()}, weight = F::one();
      for (int w = 0; w < 32; w++) {
        q[0] = F::add(q[0], sh[0][w]);
        q[1] = F::add(q[1], sh[1][w]);
      }
      for (int b = 0; b < lam - 1 - i; b++) weight = F::mul(weight, half);
      q[0] = F::mul(q[0], weight);
      q[1] = F::mul(q[1], weight);
      r_all = coins[i] =
          sc_door::round<sc_door::Form::EQ2, sc_door::Draw::COIN>(sc_door::Site::TOKEN, before + i, t, q);
    }
    __syncthreads();
    const T r = r_all;
    for (int x = threadIdx.x; x < H; x += 1024) {
      A[x] = F::add(A[x], F::mul(r, F::sub(A[x + H], A[x])));
      B[x] = F::add(B[x], F::mul(r, F::sub(B[x + H], B[x])));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *tr = t;
}
