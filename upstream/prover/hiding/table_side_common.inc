// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Shared code for the first three sumcheck rounds of a product layer on a lookup's table side:
// field arithmetic, per-block sums and the kernel k_end. The rounds sum eq(rho, i) L(i) R(i) over i
// and bind the top three bits of i. Included by prover/hiding/table_side.cuh, which describes them.
// Public source: NVIDIA, CUB documentation (https://nvidia.github.io/cccl/cub/): cub::BlockReduce,
// cub::WarpReduce; for the arithmetic, the works that prover/hiding/pt_field.h cites.
namespace tside {
constexpr int TB = 256;    // threads per block
constexpr int MAXV = 8;    // the largest number of partial sums that a block writes in a round
constexpr int MAXB = 1056; // the largest grid of a round's kernel; Ctx::part has room for this many blocks
// Arithmetic on T (`Ext`, prover/common/common.h), an element of the cubic extension field.
// Forwards to F, the traits struct `FE` (prover/hiding/pt_field.h).
struct PF {
  static __device__ __forceinline__ T add(T a, T b) { return F::add(a, b); }
  static __device__ __forceinline__ T sub(T a, T b) { return F::sub(a, b); }
  static __device__ __forceinline__ T mul(T a, T b) { return F::mul(a, b); }
  static __device__ __forceinline__ T mulb(T a, u64 s) { return F::mul(a, F::small(s)); } // s below the prime
};
__device__ __forceinline__ T zero() {
  return {0, 0, 0};
}
// Returns a times y. y must not be LLONG_MIN, whose negation overflows (not checked).
__device__ __forceinline__ T muli(T a, long long y) {
  const T m = PF::mulb(a, (u64)(y < 0 ? -y : y));
  return y < 0 ? PF::sub(zero(), m) : m;
}
// Returns lo + r (hi - lo).
__device__ __forceinline__ T lerp(T r, T lo, T hi) {
  return PF::add(lo, PF::mul(r, PF::sub(hi, lo)));
}
struct Add {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const { return PF::add(a, b); }
};
// State stored between a layer's three rounds: the challenges so far, r, and E[h] = eq(r, h), where
// the first challenge belongs to the top bit of h.
struct Rs {
  T r[3], E[8];
};
// Scratch memory for these rounds. ctx_at() expects ctx_elements() elements of T (not checked).
struct Ctx {
  T *part = nullptr; // MAXB x MAXV partial sums
  Rs *rs = nullptr;
};
static inline size_t ctx_elements() {
  return (size_t)MAXB * MAXV + sizeof(Rs) / sizeof(T) + 8;
}
static inline Ctx ctx_at(T *room) {
  Ctx c;
  c.part = room;
  c.rs = reinterpret_cast<Rs *>(room + (size_t)MAXB * MAXV);
  return c;
}
// Writes the block's sum of each v[q] to part[blockIdx.x * NV + q].
// The block must have TB threads, and all of them must call it.
// The idea is public: a small count as a template parameter: NVIDIA, CUTLASS documentation,
// "Programming Guidelines", "Templates",
// https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/programming_guidelines.html.
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// each thread's short v is then read at fixed q only and may live in registers: M. Milakov, "Fast
// Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int NV> __device__ __forceinline__ void block_sums(const T (&v)[NV], T *part) {
  typedef cub::BlockReduce<T, TB> BR;
  __shared__ typename BR::TempStorage tmp;
#pragma unroll
  for (int q = 0; q < NV; q++) {
    const T s = BR(tmp).Reduce(v[q], Add());
    if (threadIdx.x == 0) part[(size_t)blockIdx.x * NV + q] = s;
    __syncthreads();
  }
}
// Finishes round rd < 3 of a layer. `part` has nblk blocks' partial sums of v0_t = sum w lo_L lo_R
// and v1_t = sum w (hi_L - lo_L)(hi_R - lo_R) (w, lo, hi: table_side_kernels.inc), for each value
// t < nt = 4 >> rd of the top bits that this round leaves unbound. Added over the blocks, then over
// t with weights eq((rho[iz2], rho[iz3]), t) in round 0, eq(rho[iz3], t) in round 1, they are g(0)
// and the X^2 coefficient of psc::Form::EQ2's quadratic g. sc_door::round() (sumcheck_door.cuh)
// sends the masked message formed from them on the transcript st->tr; it traps if no sumcheck was
// begun there. The challenge goes to ch->r[j + rd] and into *rs; round 2 copies rs->E to k8.
// The idea is public: the GPU derives the Fiat-Shamir challenge, so that no data goes back to the
// host in a round: "Hardware-friendliness of HyperPlonk, Part2", a note on HackMD
// under the account Ingonyama, https://hackmd.io/@Ingonyama/Hardware-Friendliness-HyperPlonk-Part2.
// The idea is public: __syncwarp(), not __syncthreads(), for shared data that only one warp uses:
// NVIDIA, "CUDA C++ Best Practices Guide", "Shared Memory in Matrix Multiplication (C=AB)",
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The idea is public: __syncwarp() before and after an in-place store: "Using CUDA Warp-Level
// Primitives", a post on NVIDIA's developer blog, "Warp Synchronization",
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/.
// The compiler hints are public: __launch_bounds__: the Best Practices Guide, "Occupancy";
// #pragma unroll with a constant: the CUDA Programming Guide, cited at block_sums().
__global__ void __launch_bounds__(TB)
    k_end(const T *part, int nblk, int rd, St *st, Chal<F> *ch, int j, int iz2, int iz3, T *out, T *k8, Rs *rs) {
  typedef cub::WarpReduce<T> WR;
  __shared__ typename WR::TempStorage tmp[TB / 32];
  __shared__ T S[MAXV], Q[8], R, Eo[8];
  TR_IN_SHARED(trs);
  const int nt = 4 >> rd, nv = nt * 2, t = threadIdx.x;
  // S[q] is the sum of part[b * nv + q] over the blocks b < nblk.
#pragma unroll 1
  for (int q = t >> 5; q < nv; q += TB / 32) {
    T x = zero();
#pragma unroll 4
    for (int b = t & 31; b < nblk; b += 32) x = PF::add(x, part[(size_t)b * nv + q]);
    x = WR(tmp[t >> 5]).Reduce(x, Add());
    if ((t & 31) == 0) S[q] = x;
  }
  __syncthreads();
  if (t >= 32) return;
  const T z2 = ch->rho[iz2], z3 = ch->rho[iz3];
  sc_door::tr_copy_by_lanes(&trs, &st->tr, t, 32);
  if (t < nt) {
    Q[t] = S[t * 2];
    Q[4 + t] = S[t * 2 + 1];
  }
  if (rd && t < (1 << rd)) Eo[t] = rs->E[t];
  __syncwarp();
#pragma unroll 1
  for (int w = nt >> 1, s = 0; w; w >>= 1, s++) {
    const bool on = t < 2 * w;
    const int q = t >= w ? 4 : 0, x = t >= w ? t - w : t;
    T y = zero();
    if (on) y = lerp(s ? z2 : z3, Q[q + 2 * x], Q[q + 2 * x + 1]);
    __syncwarp();
    if (on) Q[q + x] = y;
    __syncwarp();
  }
  if (t == 0) {
    const T q[2] = {Q[0], Q[4]};
    const T r = sc_door::round<sc_door::Form::EQ2, sc_door::Draw::COIN>(sc_door::Site::LAYER_TABLE, j + rd, trs, q);
    R = r;
    ch->r[j + rd] = r;
    rs->r[rd] = r;
  }
  __syncwarp();
  sc_door::tr_copy_by_lanes(&st->tr, &trs, t, 32);
  if (t < (2 << rd)) {
    const T f = (t & 1) ? R : PF::sub(F::one(), R), e = rd ? PF::mul(Eo[t >> 1], f) : f;
    rs->E[t] = e;
    if (rd == 2) k8[t] = e;
  }
}
// Launches k_end (one block of TB threads) at the highest launch priority. k_end ignores `out`.
static inline void end_round(const Ctx &x, int nblk, int rd, St *st, Chal<F> *cd, int j, int iz2, int iz3, T *out,
                             T *k8, cudaStream_t stream) {
  pt_hi(k_end, 1, TB, 0, stream, (const T *)x.part, nblk, rd, st, cd, j, iz2, iz3, out, k8, x.rs);
  CK(cudaGetLastError());
}
} // namespace tside
