// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The worst-case check of the committer program cp_commit: can any prompt drive a value of the
// forward pass out of its range? The prover program cp_prove fails a request in which a value
// leaves the range that its cells (committed small integers) can represent: see QUOTIENT_TOP,
// ROUTER_TOP, SQUARES_BITS, STREAM_TOP and HEAD_LOW_BITS. The lines rcl, gcl and xl of the
// parameter file (policy.txt) widen some of these ranges. Whether a request fails depends on the
// weights and the prompt together, so the outcome, served or failed, tells a client something about
// the weights (docs/what-is-not-hidden.md, "What a client learns"). From the committed integers and
// public parameters only, this check computes upper bounds on these values over all prompts
// (scales/worst-case-bounds.md, "The bounds of the worst-case check"). The bounds alone do not show
// that no request fails on a range: that also needs the programs to make no range test other than
// those bounded here, and not all of that is established. Every range is still proved in every
// request: the check is about the honest prover's arithmetic only.
// Callers use struct Check: read(), begin(), take() for each unit, table() for the verdict, end().
// After begin(), a caller must call end() before it returns: end() zeroes GPU buffers; stop() and
// a failed CK() exit without it. A unit (`UnitW`) is a tensor of kind X; take() lists X's values.
// Every number below is an upper bound: sums of integers are exact; other steps are rounded up or
// carry an error bound. The inequalities are stated without their proofs.
// 1. (scales/worst-case-bounds.md, "The ball and the box after a norm";
//    scales/worst-case-bounds.md, "The bound on one row's sum".)
//    A norm (public M, K, eps) maps a row x of the residual stream to y_k = round(x_k R / 2^K),
//    with R = round(2^M / sigma) and sigma = isqrt(sum x^2 + eps), and to a_k = clamp(y_k, +-CM). If sum x^2 < 2^59,
//      ||a||_2 <= B = 2^(M-K) (1 + 1/isqrt(eps)) + (isqrt(2^59 + eps) + 2) / 2^(K+1) + sqrt(d) / 2,   |a_k| <= CM.
//    So a prompt chooses a point of a public ball cut with a public box, and a matrix row w gives a sum of at most
//      max { w.a : ||a||_2 <= B, |a_k| <= CM } <= tau B^2 / (2 CM) + CM sum_k phi(|w_k|),  for any tau > 0,
//      phi(v) = v^2 / (2 tau) if v <= tau, else v - tau / 2,      where CM is CELL_TOP
//    (the Lagrange dual, tight at the best tau). The least over a grid of tau, the ball only and the box only is taken.
//    This bounds q, k, v, router, gate and up. RoPE rotates two rows by integers (c, s), c^2 + s^2 <= 4096^2 + 5794.
//    A routed expert reads round(g a), with a router weight 0 <= g <= 1: no cell is larger than a's. Same ball and box.
// 2. The residual stream (scales/worst-case-bounds.md, "The length of the residual stream"). Each
//    of its 2 L additions reads a normed row, so its size does not depend on x (L layers; H heads
//    of dh cells share G key-value groups; d is the stream's width):
//      ||x||_2 <= X_0 + the sum of the additions' bounds Delta,   X_0 = the length of the embedding's longest row.
//    After attention, each head's output is a rounded convex combination of its group's v rows, so
//      Delta_o <= ||Wo|| sqrt((H/G) sum_g (V_g + sqrt(dh)/2)^2) / 2^ko + sqrt(d)/2,   V_g >= ||v in group g||_2.
//    After a feed-forward block, h_j = s_j u_j, |u_j| <= U_j, |s_j| <= S_j, the gate table's largest output on the keys
//    that gate row j can reach. So ||h||_2 <= min( B ||diag(S) Wu|| / 2^ku + ||S||_2 / 2, sqrt(sum_j (S_j U_j)^2) ) and
//      Delta_d <= (||Wd|| ||h||_2 of the shared expert + max over experts e of ||Wd_e|| ||h_e||_2) / 2^kd + sqrt(d)/2.
//    A token goes to one expert, so the experts contribute a maximum, not a sum. If ||x||_2 < 2^29.5 at every norm, no
//    sum of squares reaches 2^59 and no cell of the stream leaves four limbs. (Five limbs, under xl 5: see lines().)
// 3. ||.|| is the largest singular value. The check uses ||W||^(2^(k+1)) <= || (W^T W)^(2^k) ||_F
//    (Gram iteration, Delattre et al., arXiv:2305.16173): one Gram matrix and k <= SQUARINGS
//    squarings in binary64 (scales/worst-case-bounds.md, "The largest singular value of a matrix").
//    A product with inner length m has, in any order of summation, ||computed(A^T A) - A^T A||_F <=
//    gamma ||A||_F^2, gamma = m 2^-53 / (1 - m 2^-53) (Higham, Accuracy and Stability of Numerical
//    Algorithms). Twice that is added at each level. This assumes that the library computes the
//    classical product in IEEE binary64. To request that arithmetic, the check sets the math mode
//    CUBLAS_PEDANTIC_MATH (cuBLAS manual, cited at begin()), and two guards, product_holds()
//    and gram_is_exact(), test the assumption. If a guard fails, no line of the table fits.
// Not bounded here, because public sizes rule out an overflow
// (scales/worst-case-bounds.md, "What the bounds do not cover"): the lookup multiplicities and
// attention's division. cp_prove rejects a parameter file that would allow either.
// The table has one line per site (a range condition at one point of the forward pass) and layer: the largest bound
// over the site's rows and experts, never a row's index. Identifiers and printed lines say "word"
// for a parameter, "layer words" for the quantization scales (KL.bin) and "add" for an addition.
// check_the_weights() (prover/hash_commitment/hf_commit.cu) calls take() in the same pass that
// scans the cells' ranges. Public source: RFC 1071 (Braden, Borman, Partridge, 'Computing the
// Internet Checksum', September 1988, section 2) describes one fetch of data for checksum and copy.
#pragma once
#include <array>
#include <cublas_v2.h>
#include "common/blas.h"
namespace rf {
namespace worst_case {
// ---------------------------------------------------------------------------------------------------------------------
// Host arithmetic, rounded up. The arguments are not negative. Each operation rounds to the nearest
// binary64 value, so the next value above its result is at least the true result.
static inline double up(double x) {
  return std::nextafter(x, INFINITY);
}
static inline double add_up(double a, double b) {
  return up(a + b);
}
static inline double mul_up(double a, double b) {
  return up(a * b);
}
static inline double div_up(double a, double b) { // b is exact, or a lower bound
  return up(a / b);
}
static inline double sqrt_up(double a) {
  return up(std::sqrt(a));
}
static inline double whole_up(u64 v) {
  return v < (1ull << 53) ? (double)v : up((double)v);
}
static inline u64 isqrt(u64 v) {
  u64 r = (u64)std::sqrt((double)v);
  while (r * r > v) r--;
  while ((r + 1) * (r + 1) <= v) r++;
  return r;
}
// round(A / 2^k), k >= 1: the largest |round(s / 2^k)| over the sums s with |s| <= A.
static inline u64 quotient(u64 A, int k) {
  return (A + (1ull << (k - 1))) >> k;
}
// The prover's kernels have these range ends and table rules as literals, so no compiler catches a
// mismatch: change both together. A test that is not part of this repository compares the texts.
// margin_digits(), HEAD_BIG: prover/head/forward.cuh. Cost: the acceptance policy's charge for a logit gap.
constexpr int WEIGHT_MOST = 4095;        // |w| <= WEIGHT_MOST for a matrix's integers
constexpr int QUOTIENT_TOP = 32639;      // q, k, v, gate, up and a norm's quotient lie in QUOTIENT_LOW .. QUOTIENT_TOP
constexpr int QUOTIENT_LOW = -32768;
constexpr int ROUTER_TOP = 16383;        // a router logit lies in -ROUTER_TOP - 1 .. ROUTER_TOP
constexpr int CELL_TOP = 2047;           // a normed cell and a v cell are clamped to +-CELL_TOP (cells of 12 bits)
constexpr int SQUARES_BITS = 59;         // a norm's sum of squares stays below 2^SQUARES_BITS
constexpr long long STREAM_TOP = 2139062143; // a cell of the residual stream lies in -2,147,483,647 .. STREAM_TOP
constexpr int GATE_KEY_BITS = 8;         // the default of the line eg; the gate table's key c means z = c / 2^eg
constexpr int GATE_OUT_STEPS = 64;       // the table clamps z to -GATE_OUT_STEPS .. GATE_OUT_STEPS - 2^-eg and returns
                                         // round(GATE_OUT_STEPS silu(z)), clamped to +-smax (default: GATE_TOPS[0])
constexpr int LAST_STREAM_BITS = 30;     // a cell of the stream after the last layer stays below 2^LAST_STREAM_BITS
                                         // (no program tests this; SQUARES_BITS implies it)
constexpr int HEAD_LOW_BITS = 7;         // a margin of the output head stays below (2^16 + 1) 2^(HEAD_LOW_BITS + 15),
                                         // the range of its three digits (`margin_digits`) if policy.txt's
                                         // line sh is 7; sh is 7 or 8, so one verdict covers both
constexpr int HEAD_LIFT_BITS = 37;       // a position adds at most 2^HEAD_LIFT_BITS + 2^32 to its margins: HEAD_BIG
                                         // if its cost > policy.txt's C, else the cost (32 bits)
// The values of tau that a row's bound tries: every integer up to 67, then steps of 3 %, up to
// WEIGHT_MOST. The dual bound is flat around its minimum: at a tau that is off by a factor r it
// is at most (r + 1/r) / 2 times the minimum, so a step of 3 % costs less than 2 parts in 10,000.
constexpr int BINS_MOST = 250;
static inline std::vector<int> tau_grid() {
  std::vector<int> edge = {0};
  while (edge.back() < WEIGHT_MOST) edge.push_back(std::min(WEIGHT_MOST, std::max(edge.back() + 1, edge.back() * 103 / 100)));
  return edge; // for b >= 1, bin b contains the weights with edge[b - 1] < |w| <= edge[b]
}
constexpr int SQUARINGS = 4;
// The gate table's output is clamped to +-smax, a line of the parameter file; without that line smax is 4,095
// (64 in real units). The stream's bound is also computed under each lower value that smax can
// have, to show what it would gain. Only the parameter file's own value counts for the verdict.
constexpr int N_TOPS = 5;
constexpr int GATE_TOPS[N_TOPS] = {4095, 2047, 1023, 511, 255};
using ByTop = std::array<double, N_TOPS>;
// Whether the forward pass reads the parameter file's lines eg and smax, rcl, dcl. read() rejects a
// parameter file that sets a line which the forward pass does not read.
constexpr bool GATE_WORDS_SERVED = true, ROUTER_CLAMP_SERVED = true, DOWN_CLAMP_SERVED = false;
constexpr bool GATE_CLAMP_SERVED = true, WIDE_STREAM_SERVED = true; // the lines gcl and xl, by the same rule
// Under the line xl 5 of the parameter file a cell of the stream has five limbs (its digits at
// radix 256), and a norm's sum of squares plus its eps stays below 2^WIDE_SQUARES_BITS.
constexpr long long WIDE_STREAM_TOP = 547599908735; // 127 (256^5 - 1) / 255
constexpr int WIDE_SQUARES_BITS = 78;
// Under gcl 1 a gate's wide key is l + 256 h, where l is a limb in -128 .. 127 and -16,384 <= h <= 16,383.
constexpr int WIDE_GATE_TOP = 256 * 16383 + 127;
// ---------------------------------------------------------------------------------------------------------------------
// Kernels. A tensor is stored as in its file: the input index varies fastest, then the output row,
// then the expert. GSL(i, n) (common.cuh) is a grid-stride loop over i < n.

// For each row w of n0 weights: bound[row] is an integer at or above every |w.a| with ||a||_2^2 <=
// ball2 and |a_k| <= box, and squares[row] = sum w^2. Sets *fault if a weight exceeds WEIGHT_MOST
// in absolute value. `edges` is tau_grid(), of BINS + 1 entries; bins[v] is the bin of |w| = v.
// Requires 256 threads per block and BINS <= BINS_MOST (begin() checks that). sum1 (32 bits) needs
// n0 < 2^20: not checked. The bound is the least of the dual bound (top comment, 1) at each
// positive tau in `edges`, the bound by the box only and the bound by the ball only. The weights
// and tau are integers, and the factor 1/(2 tau) is taken out of the sum, so a 64-bit integer
// represents it exactly. The few closing floating-point steps are each rounded upwards, before the
// result is cut down to an integer, as w.a is one.
// Public source: a sum accumulated exactly in fixed point and rounded only at the end, upwards for
// an upper bound, is described by Kulisch and Bohlender ("High Speed Associative Accumulation of
// Floating-point Numbers and Floating-point Intervals", Reliable Computing 23, 2016, pp. 141-153,
// section 4), there for sums of floating-point numbers and intervals in a long register.
// The idea is public: a per-block histogram in shared memory: N. Sakharnykh, "GPU Pro Tip: Fast
// Histograms Using Shared Atomics on Maxwell", NVIDIA Technical Blog, 2015,
// https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/.
// The idea is public: candidates evaluated from a histogram's bins: H. Zhang, S. Si, C.-J. Hsieh,
// "GPU-acceleration for Large-scale Tree Boosting", arXiv:1706.08359, sections 3.3 and 4,
// https://arxiv.org/pdf/1706.08359.
// The idea is public: lookup tables staged in shared memory: A. Di Biagio, A. Barenghi, G. Agosta,
// G. Pelosi, "Design of a Parallel AES for Graphics Hardware using the CUDA framework", section 3.3
// and Figure 4, https://barenghi.faculty.polimi.it/files/IPDPS2009.pdf.
// The idea is public: shared memory filled by the program itself: NVIDIA, "CUDA Programming Guide",
// page "Writing SIMT Kernels", "Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
// The idea is public: tree reduction with sequential addressing (here of a minimum): M. Harris,
// "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology, slides "Parallel Reduction:
// Sequential Addressing" and "Reduction #3: Sequential Addressing",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// The loop form is public: the block-stride loop: shown in a listing on the page "Writing SIMT
// Kernels" of the CUDA Programming Guide (cited above), "Distributed Shared Memory".
// Related published results for the three bounds of which the least is taken.
// The bound by the box only is the bound of: Colbert et al., "A2Q: Accumulator-Aware Quantization with Guaranteed
// Overflow Avoidance", arXiv:2308.13504, section 3.2, eq. (11).
// The bound by the ball only uses the same idea, there in an approximate form, here as an inequality with the
// rounding terms: Salmani et al., "SLaNC: Static LayerNorm Calibration", arXiv:2410.10553, section 3.2.
// The bounds at a tau use the fact that a feasible dual point gives a guaranteed bound: E. Wong and J. Z. Kolter,
// "Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope", Proceedings of the 35th
// International Conference on Machine Learning (ICML 2018), PMLR 80, arXiv:1711.00851, section 3.2. There a lower
// bound on a minimum, for a ReLU network around one input; here an upper bound on one inner product, for all prompts.
__global__ void k_row_bounds(const short *w, size_t rows, int n0, double ball2, int box, const int *edges, int BINS,
                             const unsigned char *bins, u64 *bound, u64 *squares, unsigned *fault) {
  __shared__ unsigned count[BINS_MOST + 1], sum1[BINS_MOST + 1];
  __shared__ unsigned long long sum2[BINS_MOST + 1];
  __shared__ double least[256];
  __shared__ int edge[BINS_MOST + 1];
  __shared__ unsigned char bin_of[WEIGHT_MOST + 1];
  for (int b = threadIdx.x; b <= BINS; b += blockDim.x) edge[b] = edges[b];
  for (int v = threadIdx.x; v <= WEIGHT_MOST; v += blockDim.x) bin_of[v] = bins[v];
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    for (int b = threadIdx.x; b <= BINS; b += blockDim.x) count[b] = sum1[b] = 0, sum2[b] = 0;
    __syncthreads();
    const short *r = w + row * (size_t)n0;
    for (int k = threadIdx.x; k < n0; k += blockDim.x) {
      int a = abs((int)r[k]);
      if (a > WEIGHT_MOST) {
        atomicOr(fault, 1u);
        a = WEIGHT_MOST;
      }
      if (!a) continue;
      const int b = bin_of[a];
      atomicAdd(&count[b], 1u);
      atomicAdd(&sum1[b], (unsigned)a);
      atomicAdd(&sum2[b], (unsigned long long)(a * a));
    }
    __syncthreads();
    // Thread b in 1 .. BINS computes the bound at tau = edge[b]. Thread 0 computes the two bounds
    // without a tau: by the box only and by the ball only.
    double mine = INFINITY;
    const int b = threadIdx.x;
    if (b >= 1 && b <= BINS) {
      long long above = 0, above1 = 0, below2 = 0;
      for (int c = 1; c <= BINS; c++) {
        if (c <= b)
          below2 += (long long)sum2[c];
        else
          above += count[c], above1 += sum1[c];
      }
      const long long tau = edge[b];
      // [tau^2 ball2 / box + box (sum_{|w| <= tau} w^2 + sum_{|w| > tau} (2 tau |w| - tau^2))] / (2 tau)
      const long long inner = below2 + 2 * tau * above1 - tau * tau * above; // exact, and not negative
      mine = __ddiv_ru(__dadd_ru(__ddiv_ru(__dmul_ru((double)(tau * tau), ball2), (double)box),
                                 __dmul_ru((double)box, __ll2double_ru(inner))),
                       (double)(2 * tau));
    } else if (b == 0) {
      long long all1 = 0, all2 = 0; // sum |w| and sum w^2 over the row, exact
      for (int c = 1; c <= BINS; c++) all1 += sum1[c], all2 += (long long)sum2[c];
      mine = fmin(__dmul_ru((double)box, __ll2double_ru(all1)), __dsqrt_ru(__dmul_ru(ball2, __ll2double_ru(all2))));
      // Public source: a by-product written out for a later step: cuDNN's
      // cudnnBatchNormalizationForwardTraining() (NVIDIA, cuDNN API Reference, version 8.7.0)
      // likewise writes statistics that it computes anyway to optional output buffers. Here the
      // by-product is the row's sum of squares, which take() (case 0) uses.
      squares[row] = (u64)all2;
    }
    least[threadIdx.x] = mine;
    __syncthreads();
    for (int half = 128; half; half >>= 1) {
      if (threadIdx.x < half) least[threadIdx.x] = fmin(least[threadIdx.x], least[threadIdx.x + half]);
      __syncthreads();
    }
    if (!threadIdx.x) bound[row] = (u64)floor(least[0]); // the sum w.a is an integer
    __syncthreads();
  }
}
// dot[i] = (row 2 i) . (row 2 i + 1), exact. Requires 256 threads per block.
// The idea is public: tree reduction in shared memory with sequential addressing: M. Harris's
// slides on reduction (cited at k_row_bounds()).
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels"
// (cited at k_row_bounds()).
__global__ void k_pair_dots(const short *w, size_t pairs, int n0, long long *dot) {
  __shared__ long long part[256];
  for (size_t i = blockIdx.x; i < pairs; i += gridDim.x) {
    const short *r = w + 2 * i * (size_t)n0;
    long long s = 0;
    for (int k = threadIdx.x; k < n0; k += blockDim.x) s += (long long)r[k] * r[n0 + k];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int half = 128; half; half >>= 1) {
      if (threadIdx.x < half) part[threadIdx.x] += part[threadIdx.x + half];
      __syncthreads();
    }
    if (!threadIdx.x) dot[i] = part[0];
    __syncthreads();
  }
}
// squares[row] >= the sum of squares of a row of 32-bit integers. Requires 256 threads per block.
// The idea is public: tree reduction in shared memory with sequential addressing: M. Harris's
// slides on reduction (cited at k_row_bounds()).
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels"
// (cited at k_row_bounds()).
__global__ void k_wide_row_squares(const int *w, size_t rows, int n0, double *squares) {
  __shared__ double part[256];
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int *r = w + row * (size_t)n0;
    double s = 0;
    for (int k = threadIdx.x; k < n0; k += blockDim.x) s = __dadd_ru(s, __dmul_ru((double)r[k], (double)r[k]));
    part[threadIdx.x] = s;
    __syncthreads();
    for (int half = 128; half; half >>= 1) {
      if (threadIdx.x < half) part[threadIdx.x] = __dadd_ru(part[threadIdx.x], part[threadIdx.x + half]);
      __syncthreads();
    }
    if (!threadIdx.x) squares[row] = part[0];
    __syncthreads();
  }
}
// out = the rows x n0 matrix w in binary64, with row j multiplied by scale[j] if `scale` is
// given. Both factors must be integers below 2^13 in absolute value: then each product is exact.
__global__ void k_to_double(const short *w, size_t rows, int n0, const double *scale, double *out) {
  GSL(i, rows * (size_t)n0) out[i] = scale ? scale[i / n0] * (double)w[i] : (double)w[i];
}
// sum of part[0 .. gridDim.x) = sum of the squares of a[0 .. n), rounded to nearest. Requires 256 threads per block.
__global__ void k_sum_squares(const double *a, size_t n, double *part) {
  __shared__ double sum[256];
  double s = 0;
  GSL(i, n) s += a[i] * a[i];
  sum[threadIdx.x] = s;
  __syncthreads();
  for (int half = 128; half; half >>= 1) {
    if (threadIdx.x < half) sum[threadIdx.x] += sum[threadIdx.x + half];
    __syncthreads();
  }
  if (!threadIdx.x) part[blockIdx.x] = sum[0];
}
// y[j] = sum_i a[i + ld j] x[i] for j < columns: the transposed matrix times x. Requires 256 threads per block.
// The idea is public: tree reduction in shared memory with sequential addressing: M. Harris's
// slides on reduction (cited at k_row_bounds()).
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels"
// (cited at k_row_bounds()).
__global__ void k_transposed_times(const double *a, size_t ld, size_t length, size_t columns, const double *x,
                                   double *y) {
  __shared__ double sum[256];
  for (size_t j = blockIdx.x; j < columns; j += gridDim.x) {
    double s = 0;
    for (size_t i = threadIdx.x; i < length; i += blockDim.x) s += a[i + ld * j] * x[i];
    sum[threadIdx.x] = s;
    __syncthreads();
    for (int half = 128; half; half >>= 1) {
      if (threadIdx.x < half) sum[threadIdx.x] += sum[threadIdx.x + half];
      __syncthreads();
    }
    if (!threadIdx.x) y[j] = sum[0];
    __syncthreads();
  }
}
// y[i] += sum_j a[i + ld j] x[j], i < length. y must contain zeros; gridDim.x blockDim.x >= length.
__global__ void k_times(const double *a, size_t ld, size_t length, size_t columns, const double *x, double *y) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= length) return;
  double s = 0;
  for (size_t j = blockIdx.y; j < columns; j += gridDim.y) s += a[i + ld * j] * x[j];
  atomicAdd(&y[i], s);
}
// sum[j], squares[j] = the sum of row j of the rows x n0 matrix w, and of its squares. Requires 256 threads per block.
// The idea is public: tree reduction in shared memory with sequential addressing: M. Harris's
// slides on reduction (cited at k_row_bounds()).
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels"
// (cited at k_row_bounds()).
__global__ void k_row_sums(const short *w, size_t rows, int n0, long long *sum, u64 *squares) {
  __shared__ long long part[256], part2[256];
  for (size_t j = blockIdx.x; j < rows; j += gridDim.x) {
    long long s = 0, s2 = 0;
    for (int i = threadIdx.x; i < n0; i += blockDim.x) {
      const long long v = w[j * (size_t)n0 + i];
      s += v, s2 += v * v;
    }
    part[threadIdx.x] = s, part2[threadIdx.x] = s2;
    __syncthreads();
    for (int half = 128; half; half >>= 1) {
      if (threadIdx.x < half) part[threadIdx.x] += part[threadIdx.x + half], part2[threadIdx.x] += part2[threadIdx.x + half];
      __syncthreads();
    }
    if (!threadIdx.x) sum[j] = part[0], squares[j] = (u64)part2[0];
    __syncthreads();
  }
}
// sum[i] = the sum of column i.
__global__ void k_column_sums(const short *w, size_t rows, int n0, long long *sum) {
  GSL(i, (size_t)n0) {
    long long s = 0;
    for (size_t j = 0; j < rows; j++) s += w[j * (size_t)n0 + i];
    sum[i] = s;
  }
}
// For the n x n matrix g times `back` (a power of two): adds the sum of all entries to out[0] and
// the trace to out[1], both modulo 2^64, and sets out[2] if an entry is not an integer below 2^53
// in absolute value. out must contain zeros.
// Public source: exact integer products in binary64, below 2^53: J.-G. Dumas, P. Giorgi, C. Pernet,
// "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK packages", arXiv
// cs/0601133 (version dated August 22, 2018), introduction and section 2.1.2 "Ring homomorphism and
// delayed reduction"; https://arxiv.org/abs/cs/0601133 (the libraries FFLAS and FFPACK).
// Public source: a result's distance from an integer: Great Internet Mersenne Prime Search,
// "Mathematics and Research Strategy" (last updated 16 Oct 2024), passage on the Lucas-Lehmer
// test's floating-point FFT; https://www.mersenne.org/various/math.php. GIMPS allows 0.4; here 0.
// Public sources for the sum of all entries: see gram_is_exact().
__global__ void k_whole_sums(const double *g, size_t n, double back, unsigned long long *out) {
  unsigned long long all = 0, trace = 0;
  bool odd = false;
  GSL(i, n * n) {
    const double v = g[i] * back;
    odd = odd || !(fabs(v) < 9007199254740992.0) || v != rint(v);
    const unsigned long long word = (unsigned long long)(long long)v;
    all += word;
    if (i / n == i % n) trace += word;
  }
  atomicAdd(&out[0], all);
  atomicAdd(&out[1], trace);
  if (odd) atomicOr(&out[2], 1ull);
}
// x[i] = +1 or -1, by a fixed rule that does not depend on the matrices.
// Public sources for a probe vector of signs: see product_holds().
__global__ void k_signs(double *x, size_t n) {
  GSL(i, n) x[i] = ::mix64(0x7369676E73ull + i) & 1 ? 1.0 : -1.0;
}
// Writes the synthetic model's tensor of kind X (UnitW::X, see take()) in layer l, in a file's
// element order, by the formulas of g5::cellWm() (kernels.cuh). `second` selects k in kind 0.
template <class Out>
__global__ void k_generated(::g5::SM S, int X, long long l, bool second, size_t n, long long n0, long long rows,
                            Out *out) {
  const bool experts = S.moe && (l & 1);
  GSL(i, n) {
    const long long in = (long long)(i % n0), row = (long long)(i / n0 % rows), e = (long long)(i / n0 / rows);
    long long w = 0;
    switch (X) {
    case 0:
      w = second ? ::g5::v13m(S.seed, 3, (l * S.d + in) * S.dkv + row) : ::g5::v13m(S.seed, 2, (l * S.d + in) * S.dq + row);
      break;
    case 1:
      w = ::g5::made_up(S, 4, (l * S.d + in) * S.dkv + row);
      break;
    case 2:
      w = ::g5::made_up(S, 5, (l * S.dq + in) * S.d + row);
      break;
    case 3:
    case 5:
      w = ::g5::made_up(S, 6, (l * S.d + in) * (experts ? S.fe : S.f) + row);
      break;
    case 4:
    case 6:
      w = ::g5::made_up(S, 7, (l * S.d + in) * (experts ? S.fe : S.f) + row);
      break;
    case 7:
      w = ::g5::made_up(S, 10, ((l * S.NE + e) * S.d + in) * S.fe + row);
      break;
    case 8:
      w = ::g5::made_up(S, 11, ((l * S.NE + e) * S.d + in) * S.fe + row);
      break;
    case 9:
      w = ::g5::made_up(S, 8, (l * S.f + in) * S.d + row);
      break;
    case 10:
      w = ::g5::made_up(S, 12, ((l * S.NE + e) * S.fe + in) * S.d + row);
      break;
    case 11:
      w = ::g5::made_up(S, 9, (l * S.d + in) * S.NE + row);
      break;
    case 12:
      w = ::g5::v13m(S.seed, 13, row * S.d + in, 20);
      break;
    default:
      w = ::g5::v13m(S.seed, 14, in * S.V + row);
    }
    out[i] = (Out)w;
  }
}
// ---------------------------------------------------------------------------------------------------------------------
// One layer's scales (KL.bin): shifts k.., two norms' M and K, attention's shift sh, RoPE flag rp.
struct LayerWords {
  int kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, sh, rp, kr = 0, kgR = 0, kuR = 0;
};
// One line of the table.
struct Line {
  std::string site; // what is bounded
  int layer;        // L: after the last layer
  std::string word; // the scale parameter that moves the site's range; "-" if none does
  int value;        // that parameter's value; 0 where 'word' is "e_res" or "-"
  double bound, limit; // the value stays at or below 'bound'; it must stay at or below 'limit'
  int shortfall;       // the least b >= 0 with bound <= limit 2^b
};
constexpr int NO_BOUND = 99; // the shortfall of a bound that is not a number, is beyond every range, or is not trusted
static inline int shortfall_of(double bound, double limit) {
  int b = 0;
  while (b < NO_BOUND && !(bound <= std::ldexp(limit, b))) b++; // written so that a bound that is NaN never fits
  return b;
}
struct Check {
  ShapeW sh;
  int H = 0, G = 0, box = 0, MF = 0, KF = 0;
  // Optional lines of the parameter file, for sites that no shift brings into range. An absent line means its default.
  int eg = GATE_KEY_BITS; // the gate table's key c means c / 2^eg and is clamped to -2^(6 + eg) .. 2^(6 + eg) - 1
  int smax = GATE_TOPS[0]; // the gate table's output is clamped to +-smax
  int rcl = 0;      // 1: the proof clamps a router logit, and only its wide form, of 19 bits, can leave a range
  int dcl = 0;      // a > 0: the proof clamps each cell of the addition after the feed-forward block below 2^(11 + a),
                    // and only its wide form, of 16 + a bits, can leave a range
  int gcl = 0;      // 1: the proof clamps a gate's key, and only its wide form, of 23 bits, can leave a range
  int xl = 4;       // the number of limbs of a stream cell: 4, or 5 with a wider sum of squares
  int top_of(int c) const { return std::min(GATE_TOPS[c], smax); } // the c-th value of smax that is evaluated
  u64 eps1 = 0, eps2 = 0, epsF = 0;
  std::vector<LayerWords> words;
  std::vector<int> gate_most; // gate_most[g]: the gate table's largest absolute output on the keys -g .. g
  // What take() records, by layer. Bounds on sums are integers; norms are rounded up.
  struct Layer {
    u64 q = 0, k = 0, v = 0, router = 0, gate = 0, up = 0, gate_routed = 0, up_routed = 0; // bound on the largest |sum|
    double attention_in = 0;                // ||all heads' outputs||_2
    double v_norm = 0, o_norm = 0;          // the largest ||Wv of a group||, ||Wo||
    ByTop hidden{};                         // ||h||_2 of the dense block or the shared expert
    double down_norm = 0;                   // and its ||Wd||
    double gated_up_norm = 0, up_norm = 0;  // its ||diag(S) Wu|| and ||Wu||
    ByTop routed{};                         // the largest ||Wd_e|| ||h_e||_2 over the experts e
    // `hidden` (by ||Wu|| only) and `routed` with all S_j = top_of(c). They do not depend on which
    // keys a row of Wg can reach, so they can be scaled to a larger ball (lines(), under a WhatIf).
    ByTop hidden_at_top{}, routed_at_top{};
    double rows = 0, rows_routed = 0;       // the number of entries of h
    // the largest of each over the experts
    double routed_hidden = 0, routed_down_norm = 0, routed_gated_up_norm = 0, routed_up_norm = 0;
    int gate_top = 0, gate_top_routed = 0;  // the largest S_j
  };
  std::vector<Layer> layer;
  double embedding = 0; // X_0
  bool embedding_seen = false;
  size_t units_seen = 0;
  // State kept from a gate's unit to its up unit, and from the experts' up unit to their down unit.
  std::vector<double> S;
  std::vector<ByTop> hidden_of_expert, hidden_at_top_of_expert;
  // Buffers and handles on the GPU.
  cublasHandle_t blas = nullptr;
  void *blas_room = nullptr; // the workspace of cuBLAS
  double *matrix = nullptr, *gram[2] = {nullptr, nullptr}, *part = nullptr, *scale = nullptr, *vectors = nullptr;
  size_t matrix_cells = 0, gram_cells = 0, scale_cells = 0, rows_cap = 0, longest = 0;
  u64 *bound = nullptr, *squares = nullptr;
  int *edges = nullptr, bins = 0;      // tau_grid() on the GPU, and its number of bins
  unsigned char *bin_of = nullptr;     // the bin of each absolute value of a weight
  long long *dots = nullptr;
  unsigned *fault = nullptr;
  double seconds_rows = 0, seconds_norms = 0;
  size_t products = 0, norms = 0;
  // Why the check can give no bound, once that is so: a cuBLAS product has failed one of the two
  // guards, or a matrix has a weight beyond WEIGHT_MOST. From then on no line of the table fits.
  const char *why_no_bound = nullptr;

  static void stop(const char *why) {
    fprintf(stderr, "cp_commit: the worst-case check: %s\n", why);
    exit(2);
  }
  bool experts(int l) const { return sh.moeMode && moeW(l); }

  // PO is the parameter file; `file` is KL.bin as 32-bit words, the copy that the caller digests.
  // Returns false, with the reason in `why`, for parameters that the check does not support.
  bool read(const PolC &PO, const ShapeW &shape, const std::vector<u32> &file, std::string &why) {
    sh = shape;
    H = PO["H"], G = PO["G"];
    if ((PO["prof"] & 15) != 4 || !(PO["prof"] >> 4))
      return why = "it supports only profile 4 with the embedding and the head (prof 20)", false;
    if (PO.m.count("cw") && PO["cw"] != 12) return why = "it supports only 12-bit cells (no line cw, or cw 12)", false;
    if (PO["cs"] != 12 || PO["ut"] != 1) return why = "it supports only cs 12 and ut 1", false;
    box = CELL_TOP;
    // As in the prover, a line that is present must have one of the non-default values `admitted`.
    auto optional = [&](const char *name, int absent, std::initializer_list<int> admitted, int &word) {
      word = PO.m.count(name) ? (int)PO[name] : absent;
      return !PO.m.count(name) || std::find(admitted.begin(), admitted.end(), word) != admitted.end();
    };
    if (!optional("eg", GATE_KEY_BITS, {7, 6, 5, 4}, eg) || !optional("smax", GATE_TOPS[0], {2047, 1023, 511, 255}, smax) ||
        !optional("rcl", 0, {1}, rcl) || !optional("dcl", 0, {5, 6, 7}, dcl) || !optional("gcl", 0, {1}, gcl) ||
        !optional("xl", 4, {5}, xl))
      return why = "the parameter file's eg, smax, rcl, dcl, gcl or xl has a value that the check does not "
                   "support", false;
    if ((!GATE_WORDS_SERVED && (eg != GATE_KEY_BITS || smax != GATE_TOPS[0])) || (!ROUTER_CLAMP_SERVED && rcl) ||
        (!DOWN_CLAMP_SERVED && dcl) || (!GATE_CLAMP_SERVED && gcl) || (!WIDE_STREAM_SERVED && xl != 4))
      return why = "the parameter file has a line eg, smax, rcl, dcl, gcl or xl that the prover's forward pass does "
                   "not read: a bound computed with it would describe a different forward pass from the one that is "
                   "proved",
             false;
    auto wide = [&](const char *lo, const char *hi) {
      return PO[lo] | (u64)PO[hi] << 32;
    };
    eps1 = wide("eps1_lo", "eps1_hi"), eps2 = wide("eps2_lo", "eps2_hi"), epsF = wide("epsF_lo", "epsF_hi");
    MF = PO["MF"], KF = PO["KF"];
    size_t with_experts = 0;
    for (int l = 0; l < sh.L; l++) with_experts += experts(l);
    if (file.size() != 13 * (size_t)sh.L + 3 * with_experts)
      return why = "the file of layer words (KL.bin) does not have 13 words per layer and 3 more for each layer with experts", false;
    words.resize(sh.L);
    size_t extra = 13 * (size_t)sh.L;
    for (int l = 0; l < sh.L; l++) {
      const u32 *w = &file[13 * (size_t)l];
      LayerWords &x = words[l];
      x.kq = w[0], x.kk = w[1], x.kv = w[2], x.ko = w[3], x.kg = w[4], x.ku = w[5], x.kd = w[6], x.M1 = w[7],
      x.K1 = w[8], x.M2 = w[9], x.K2 = w[10], x.sh = w[11], x.rp = w[12];
      if (experts(l)) x.kr = file[extra], x.kgR = file[extra + 1], x.kuR = file[extra + 2], extra += 3;
      for (int k : {x.kq, x.kk, x.kv, x.ko, x.kg, x.ku, x.kd, experts(l) ? x.kr : 1, experts(l) ? x.kgR : 1,
                    experts(l) ? x.kuR : 1})
        if (k < 1 || k > 32) return why = "a shift lies outside 1 to 32", false;
      if (w[11] < 4 || w[11] > 12) return why = "attention's shift (word 11) lies outside 4 to 12", false;
      if (w[12] > 1) return why = "a RoPE flag is neither 0 nor 1", false;
    }
    // With five limbs the verifier requires eps < 2^48, as stated in
    // verifier/docs/parameter-file.md, "The optional lines on the model's arithmetic".
    const int eps_bits = xl == 5 ? 48 : SQUARES_BITS;
    auto norm_fits = [eps_bits](int M, int K, u64 eps) {
      return M >= 24 && M <= 47 && K >= 17 && K <= 32 && K >= M - 14 && eps >= 1 && !(eps >> eps_bits);
    };
    bool norms_fit = norm_fits(MF, KF, epsF);
    for (auto &x : words) norms_fit = norms_fit && norm_fits(x.M1, x.K1, eps1) && norm_fits(x.M2, x.K2, eps2);
    if (!norms_fit)
      return why = "a norm's parameters are outside 24 <= M <= 47, 17 <= K <= 32, K >= M - 14, 1 <= eps < 2^59 (2^48 "
                   "under xl 5)",
             false;
    lay_the_gate_table();
    layer.assign(sh.L, Layer());
    return true;
  }
  // Computes gate_most by the rule of the prover's own table (prover/prover/inputs.cuh). One is
  // added to each output, so that a table computed with another library's exp() is covered.
  // Public source: libraries do not round exp() alike: V. Innocente, P. Zimmermann, "Accuracy of
  // Mathematical Functions in Single, Double, Double Extended, and Quadruple Precision", 2022.
  // Public source: a computed function widened by one unit in the last place: pyintval, User guide,
  // https://pyintval.readthedocs.io/en/stable/guide.html, "Correct rounding and directed
  // endpoints". Here the value is rounded to an integer after exp(), so the unit is 1.
  // Public source for a running maximum: G. E. Blelloch, "Prefix Sums and Their Applications",
  // School of Computer Science, Carnegie Mellon University, 1.3 (a max-prescan). Here step g takes
  // in the absolute outputs at the keys g and -g, so that entry g covers the keys -g .. g.
  void lay_the_gate_table() {
    gate_most.assign(-QUOTIENT_LOW + 1, 0);
    for (int g = 0; g <= -QUOTIENT_LOW; g++) {
      int most = g ? gate_most[g - 1] : 0;
      for (int c : {g, -g}) {
        const int key_top = (GATE_OUT_STEPS << eg) - 1;
        c = c < -key_top - 1 ? -key_top - 1 : c > key_top ? key_top : c;
        const double z = std::ldexp((double)c, -eg), s = std::fabs(z / (1 + std::exp(-z)) * GATE_OUT_STEPS);
        most = std::max(most, std::min(smax, (int)std::floor(s + 0.5) + 1));
      }
      gate_most[g] = most;
    }
  }
  // The radius B after a norm (top comment, 1) or, with one_cell, a bound on one unclamped |y_k|.
  double ball(int M, int K, u64 eps, bool one_cell = false, int root_bits = 0) const {
    const double first = mul_up(std::ldexp(1.0, M - K), add_up(1.0, div_up(1.0, (double)isqrt(eps))));
    // root >= sigma + 1, from the sum of squares' range or from the bound 2^root_bits on ||x||_2
    if (!root_bits && xl == 5) root_bits = WIDE_SQUARES_BITS / 2; // sigma^2 <= the sum of squares + eps < 2^78
    const double root = root_bits ? add_up(std::ldexp(1.0, root_bits), whole_up(isqrt(eps) + 2))
                                  : whole_up(isqrt((1ull << SQUARES_BITS) + eps) + 2);
    const double second = std::ldexp(root, -(K + 1));
    return add_up(add_up(first, second), one_cell ? 0.5 : div_up(sqrt_up((double)sh.d), 2.0));
  }
  static double squared_whole(double B) { // an integer at or above B^2
    return std::ceil(mul_up(B, B));
  }

  void begin() {
    CK(cudaMalloc(&blas_room, blas::BLAS_WORKSPACE));
    CK(cudaMemset(blas_room, 0, blas::BLAS_WORKSPACE));
    blas = blas::blas(cudaStreamPerThread, blas_room); // this host thread's handle; nothing else uses it before end()
    // Public source: CUBLAS_PEDANTIC_MATH is a documented mode of cuBLAS (NVIDIA, cuBLAS
    // documentation, 'cublasMath_t'), meant there for examining numerical behaviour and finding
    // errors. It is set here because the error bound assumes binary64 throughout each product.
    if (cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS)
      stop("cuBLAS did not accept the pedantic math mode (CUBLAS_PEDANTIC_MATH)");
    const size_t widest = std::max(std::max(sh.f, sh.fe), std::max(sh.dq, sh.d));
    matrix_cells = widest * sh.d;
    longest = widest;
    const size_t side = std::max(sh.d, sh.dh);
    gram_cells = side * side;
    rows_cap = std::max({sh.NE * sh.fe, sh.f, sh.dq, sh.dkv, sh.V, sh.d}); // the most rows of a matrix in take()
    scale_cells = std::max(sh.NE * sh.fe, sh.f);
    CK(cudaMalloc(&matrix, matrix_cells * 8));
    for (auto &g : gram) CK(cudaMalloc(&g, gram_cells * 8));
    CK(cudaMalloc(&part, 1024 * 8));
    CK(cudaMalloc(&scale, scale_cells * 8));
    CK(cudaMalloc(&vectors, 4 * longest * 8));
    k_signs<<<64, 256>>>(vectors, longest);
    CK(cudaMalloc(&bound, rows_cap * 8));
    CK(cudaMalloc(&squares, rows_cap * 8));
    CK(cudaMalloc(&dots, rows_cap * 8));
    CK(cudaMalloc(&fault, 4));
    CK(cudaMemset(fault, 0, 4));
    const std::vector<int> grid = tau_grid();
    bins = (int)grid.size() - 1;
    if (bins > BINS_MOST) stop("the grid of tau has more entries than planned");
    std::vector<unsigned char> of(WEIGHT_MOST + 1, 0);
    for (int b = 1; b <= bins; b++)
      for (int v = grid[b - 1] + 1; v <= grid[b]; v++) of[v] = (unsigned char)b;
    edges = ::g5::up(grid.data(), grid.size());
    bin_of = ::g5::up(of.data(), of.size());
  }
  // Zeroes the GPU buffers of weights and of values computed from them (not `fault`); frees all.
  void end() {
    for (auto p : {std::make_pair((void *)matrix, matrix_cells * 8), std::make_pair((void *)gram[0], gram_cells * 8),
                   std::make_pair((void *)gram[1], gram_cells * 8), std::make_pair((void *)part, (size_t)1024 * 8),
                   std::make_pair((void *)scale, scale_cells * 8), std::make_pair((void *)vectors, 4 * longest * 8),
                   std::make_pair((void *)bound, rows_cap * 8),
                   std::make_pair((void *)squares, rows_cap * 8), std::make_pair((void *)dots, rows_cap * 8)}) {
      CK(cudaMemset(p.first, 0, p.second));
      cudaFree(p.first);
    }
    cudaFree(fault);
    cudaFree(edges);
    cudaFree(bin_of);
    if (cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
      stop("cuBLAS did not accept the default math mode (CUBLAS_DEFAULT_MATH)");
    CK(cudaMemset(blas_room, 0, blas::BLAS_WORKSPACE));
    cudaFree(blas_room);
    blas_room = nullptr;
    blas = nullptr;
  }
  struct Clock {
    double &into;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit Clock(double &s) : into(s) {}
    ~Clock() { into += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
  };

  // Runs k_row_bounds() with ball radius B; b and s receive the rows' bounds and sums of squares.
  void row_bounds(const short *w, size_t rows, size_t n0, double B, std::vector<u64> &b, std::vector<u64> &s) {
    Clock clock(seconds_rows);
    k_row_bounds<<<(unsigned)std::min<size_t>(rows, 132 * 8), 256>>>(w, rows, (int)n0, squared_whole(B), box, edges, bins,
                                                                    bin_of, bound, squares, fault);
    b.resize(rows), s.resize(rows);
    CK(cudaMemcpy(b.data(), bound, rows * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.data(), squares, rows * 8, cudaMemcpyDeviceToHost));
    unsigned f;
    CK(cudaMemcpy(&f, fault, 4, cudaMemcpyDeviceToHost));
    if (f) why_no_bound = "a matrix has a weight beyond 4,095 in absolute value"; // no source of weights produces one
  }
  // Returns an upper bound on ||a||_F. Kernel and host together apply fewer than 2^27 additions to
  // any term, each with a relative error of at most 2^-53: the factor 1 + 2^-20 covers them.
  double frobenius(const double *a, size_t n) {
    k_sum_squares<<<1024, 256>>>(a, n, part);
    double h[1024], s = 0;
    CK(cudaMemcpy(h, part, sizeof h, cudaMemcpyDeviceToHost));
    for (double x : h) s += x;
    if (!std::isfinite(s)) stop("a matrix product overflowed binary64");
    return sqrt_up(mul_up(s, 1.0 + std::ldexp(1.0, -20)));
  }
  // Guard (top comment, 3) against a shorter format or loosely emulated binary64. C is cuBLAS's
  // result for alpha P^T P, E its error, r the sign vector of k_signs(). Returns whether
  // ||C r - alpha P^T (P r)||_2, by this file's kernels, is within the limit. `a` is a `length` x
  // `columns` array, first index fastest; P is that matrix, or its transpose if `transposed`.
  // `scaled` >= sqrt(alpha) ||P||_F; C is n x n. Not checked: length, columns <= `longest`.
  // ||E r||_2 is ||E||_F in the root mean square over sign vectors r, so the limit is binary64's
  // bound on ||E||_F: gamma scaled^2, with gamma taken as (m + 64) 2^-53, m = max(length, columns),
  // which exceeds the top comment's gamma if m < 2^29. operator_norm() allows for twice the limit.
  // Binary64 on both sides should pass; 24-bit arithmetic or a result rounded to 24 bits should
  // not. In a direct test of this guard, one correct 640 x 640 product passed, and the same rounded
  // to 24 bits did not; 24-bit arithmetic has not been tried. A false alarm is safe: no site fits.
  // Public source: Compare guarded acceptance in conformity assessment: Joint Committee for Guides
  // in Metrology, JCGM 106:2012, Evaluation of measurement data: The role of measurement
  // uncertainty in conformity assessment, 8.3.2. Here the guard's limit is the acceptance limit and
  // twice that limit the tolerance limit; no probability is computed.
  // Public source: a result check by one probe vector: M. Turmon, R. Granat, D. S. Katz, J. Z. Lou,
  // 'Tests and Tolerances for High-Performance Software-Implemented Fault Detection', IEEE
  // Transactions on Computers, 2003; sections 1, 2 and 3 (Result 1, Corollary 2),
  // https://dus.jpl.nasa.gov/home/turmon/Papers/abft-ieee-tc.pdf; they cite Freivalds. They take
  // the vector of all ones, multiply the limit by the probe's norm and tune its constant; here the
  // probe is a fixed vector of signs, the limit is the Frobenius-norm bound itself, and what is
  // looked for is a product computed in a shorter format.
  // Public source: the mean of ||E r||_2^2 over sign vectors is ||E||_F^2: Z. Bujanovic, D.
  // Kressner, 'Norm and trace estimation with random rank-one vectors', arXiv:2004.06433 (text
  // dated August 11, 2020), sections 1 and 2, https://arxiv.org/abs/2004.06433; they credit
  // Hutchinson. Here r is one fixed vector: the mean says nothing about this r.
  bool product_holds(const double *a, size_t length, size_t columns, bool transposed, double alpha, const double *C,
                     size_t n, double scaled) {
    double *r = vectors, *t = r + longest, *z = t + longest, *c = z + longest;
    CK(cudaMemsetAsync(t, 0, 3 * longest * 8, cudaStreamPerThread));
    auto times = [&](const double *m, size_t len, size_t cols, const double *x, double *y) {
      k_times<<<dim3((unsigned)((len + 255) / 256), 64), 256>>>(m, len, len, cols, x, y);
    };
    auto transposed_times = [&](const double *m, size_t len, size_t cols, const double *x, double *y) {
      k_transposed_times<<<(unsigned)std::min<size_t>(cols, 132 * 8), 256>>>(m, len, len, cols, x, y);
    };
    if (transposed) { // P = a^T: P r = a^T r, P^T (P r) = a (a^T r)
      transposed_times(a, length, columns, r, t);
      times(a, length, columns, t, z);
    } else {
      times(a, length, columns, r, t);
      transposed_times(a, length, columns, t, z);
    }
    times(C, n, n, r, c);
    std::vector<double> hz(n), hc(n);
    CK(cudaMemcpy(hz.data(), z, n * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), c, n * 8, cudaMemcpyDeviceToHost));
    double gap = 0;
    for (size_t i = 0; i < n; i++) gap += (hc[i] - alpha * hz[i]) * (hc[i] - alpha * hz[i]);
    const double limit = std::ldexp((double)(std::max(length, columns) + 64), -53) * scaled * scaled;
    return std::sqrt(gap) <= limit; // also false if the gap is not a number
  }
  // Guard (top comment, 3): compares g, cuBLAS's Gram matrix of a weight matrix (no row_scale),
  // with the weights, in integers. `back`: 1 / alpha. A true entry is a sum of products below 2^24.
  // Up to 2^29 such products stay below 2^53 in sum, so true binary64 gives every entry exactly, in
  // any order of summation. Three tests, exact modulo 2^64: every entry is an integer; the trace is
  // the sum of the weights' squares; all entries sum to ||W 1||^2 (W^T W) or ||W^T 1||^2 (W W^T).
  // An error that changes none of the three is not noticed.
  // Public source: the checksum (e^T H)(W e) of HW: K.-H. Huang, J. A. Abraham, "Algorithm-Based
  // Fault Tolerance for Matrix Operations", IEEE Transactions on Computers C-33(6), June 1984, pp.
  // 518-528, section IV, Theorem 4.1; https://doi.org/10.1109/TC.1984.1676475. They also take the
  // sums of integer matrices modulo a power of two.
  // Public source: the checksum in this form: C. Peltekis, G. Dimitrakopoulos, "GCN-ABFT: Low-Cost
  // Online Error Checking for Graph Convolutional Networks", arXiv 2412.18534v1, section II.B
  // "Error checking GCN layers with baseline ABFT"; https://arxiv.org/abs/2412.18534. Both aim at
  // hardware faults; here, at a product not computed in true binary64.
  // Public sources for exact integer products and for the integer test: see k_whole_sums().
  bool gram_is_exact(const short *w, size_t rows, size_t n0, const double *g, double back) {
    const bool of_columns = n0 <= rows; // g = W^T W
    unsigned long long *out = (unsigned long long *)bound;
    CK(cudaMemsetAsync(out, 0, 24, cudaStreamPerThread));
    k_whole_sums<<<1024, 256>>>(g, std::min(rows, n0), back, out);
    unsigned long long said[3], all = 0, trace = 0;
    CK(cudaMemcpy(said, out, 24, cudaMemcpyDeviceToHost));
    k_row_sums<<<(unsigned)std::min<size_t>(rows, 132 * 8), 256>>>(w, rows, (int)n0, dots, squares);
    std::vector<long long> sum(rows);
    std::vector<u64> square(rows);
    CK(cudaMemcpy(sum.data(), dots, rows * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(square.data(), squares, rows * 8, cudaMemcpyDeviceToHost));
    for (u64 s : square) trace += s;
    if (!of_columns) {
      k_column_sums<<<64, 256>>>(w, rows, (int)n0, dots);
      sum.resize(n0);
      CK(cudaMemcpy(sum.data(), dots, n0 * 8, cudaMemcpyDeviceToHost));
    }
    for (long long s : sum) all += (unsigned long long)s * (unsigned long long)s;
    return !said[2] && said[0] == all && said[1] == trace;
  }
  // Returns an upper bound on the largest singular value of w (rows x n0), row j scaled by
  // row_scale[j] (on the GPU) if given. Stops early if it is at most `enough`, a level gains under
  // 1/256, or a product fails a guard: that sets why_no_bound, and only earlier products count.
  // Power iteration approaches the value from below, so it proves no bound. After k squarings
  // this bound is, up to rounding, at most a factor n^(1/2^(k+2)) above the true value for an
  // n x n Gram matrix.
  // Source for the inequality (the top comment, 3): Delattre, Barthélemy, Araujo, Allauzen,
  // "Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration", ICML 2023
  // (PMLR 202), arXiv:2305.16173, Theorem 3, first item, with W for the paper's G and t = k + 2.
  // The bound on the rounding error is not from that paper.
  double operator_norm(const short *w, size_t rows, size_t n0, const double *row_scale = nullptr, double enough = 0) {
    Clock clock(seconds_norms);
    norms++;
    if (rows * n0 > matrix_cells || std::min(rows, n0) * std::min(rows, n0) > gram_cells) stop("a matrix is larger than planned");
    if (sh.d == 256 && sh.dh == 32) {
      // The padded small model has exact zero experts and an identity layer.
      // Check zeros in the integer data, before outward rounding introduces a
      // subnormal Frobenius bound whose Gram scaling would overflow.
      // Nonzero sentinels make a missing copy decline this shortcut.
      std::vector<short> integers(rows * n0, 1);
      CK(cudaMemcpy(integers.data(), w, integers.size() * sizeof(short), cudaMemcpyDeviceToHost));
      std::vector<double> factors;
      if (row_scale) {
        factors.assign(rows, NAN);
        CK(cudaMemcpy(factors.data(), row_scale, rows * sizeof(double), cudaMemcpyDeviceToHost));
      }
      bool zero = true;
      for (size_t i = 0; i < integers.size(); ++i) {
        const double s = row_scale ? factors[i / n0] : 1;
        if (!std::isfinite(s) || (s != 0 && integers[i] != 0)) { zero = false; break; }
      }
      if (zero) return 0; // ||0||_2 = 0; every nonzero matrix keeps the original guards.
    }
    k_to_double<<<8192, 256>>>(w, rows, (int)n0, row_scale, matrix);
    const int n = (int)std::min(rows, n0);
    // A_0 is the matrix; A_(j+1) is cuBLAS's Gram matrix of A_j / 2^e_j, a product of inner length
    // m_j. f_j >= ||A_j||_F (for zeros about 2.2e-162, not 0).
    double f[SQUARINGS + 2];
    int e[SQUARINGS + 2];
    size_t m[SQUARINGS + 2];
    const double *at = matrix;
    double best = INFINITY;
    for (int level = 0;; level++) {
      f[level] = frobenius(at, level ? (size_t)n * n : rows * n0);
      if (f[level] == 0) return 0;
      // f == 0 above: only if subnormals count as 0. Back from this level: ||A_level|| <= f_level,
      // ||A_j||^2 <= 4^e_j (||A_(j+1)|| + 2 gamma (f_j / 2^e_j)^2); on the 2, see product_holds().
      // ||A||^2 = ||A^T A|| holds for every matrix, so the computed matrices need not
      // be symmetric.
      double a = f[level];
      for (int j = level - 1; j >= 0; j--) {
        const double scaled = std::ldexp(f[j], -e[j]);
        const double error = mul_up(std::ldexp((double)(m[j] + 64), -52), mul_up(scaled, scaled));
        a = std::ldexp(sqrt_up(add_up(a, error)), e[j]);
      }
      const bool little = best < mul_up(a, 1.0 + 1.0 / 256);
      best = std::min(best, a);
      if (level == SQUARINGS + 1 || best <= enough || little) return best;
      e[level] = std::ilogb(f[level]) + 1; // f / 2^e lies in [1/2, 1); for zeros e is -536, and alpha below is infinite
      m[level] = level ? (size_t)n : std::max(rows, n0);
      const double alpha = std::ldexp(1.0, -2 * e[level]), zero = 0;
      double *to = gram[level & 1];
      cublasStatus_t said;
      if (level) // A^T A
        said = cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, at, n, at, n, &zero, to, n);
      else if (n0 <= rows) // cuBLAS reads the array as the n0 x rows matrix W^T: W^T W
        said = cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, (int)rows, &alpha, at, (int)n0, at, (int)n0, &zero, to, n);
      else // W W^T
        said = cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, (int)n0, &alpha, at, (int)n0, at, (int)n0, &zero, to, n);
      if (said != CUBLAS_STATUS_SUCCESS) stop("a matrix product failed");
      const double scaled = std::ldexp(f[level], -e[level]);
      if (!(level ? product_holds(at, n, n, false, alpha, to, n, scaled)
                  : product_holds(at, n0, rows, n0 <= rows, alpha, to, n, scaled)))
        return why_no_bound = "the matrix library's product is not a product in binary64, or the matrix or a slice of it holds only zeros", best;
      if (!level && !row_scale && !gram_is_exact(w, rows, n0, to, std::ldexp(1.0, 2 * e[0])))
        return why_no_bound = "the matrix library's Gram matrix does not match the weights", best;
      products++;
      at = to;
    }
  }
  static u64 largest(const std::vector<u64> &v) {
    return *std::max_element(v.begin(), v.end());
  }

  // Bounds ||h||_2 of a feed-forward block's rows first .. first + n - 1 (top comment, 2).
  // up_matrix starts at row `first`; its row bounds b, and S, start at row 0. Under smax = `top`,
  // S_j becomes min(S_j, top), and ||diag(S) Wu|| <= min(its unclamped value, top ||Wu||).
  ByTop hidden_norm(const short *up_matrix, const std::vector<u64> &b, size_t first, size_t n, double B, int ku,
                    double &gated_up_norm, double &up_norm, ByTop &at_top) {
    gated_up_norm = operator_norm(up_matrix, n, sh.d, scale + first);
    up_norm = operator_norm(up_matrix, n, sh.d);
    ByTop hidden;
    for (int c = 0; c < N_TOPS; c++) {
      const double top = top_of(c);
      double cells = 0, gates = 0;
      for (size_t j = first; j < first + n; j++) {
        const double s = std::min(S[j], top), h = s * (double)std::min<u64>(-QUOTIENT_LOW, quotient(b[j], ku)); // exact: below 2^27
        cells = add_up(cells, mul_up(h, h));
        gates = add_up(gates, s * s);
      }
      const double norm = std::min(gated_up_norm, mul_up(top, up_norm));
      hidden[c] = std::min(add_up(std::ldexp(mul_up(B, norm), -ku), div_up(sqrt_up(gates), 2.0)), sqrt_up(cells));
      at_top[c] = add_up(std::ldexp(mul_up(B, mul_up(top, up_norm)), -ku), div_up(mul_up(top, sqrt_up((double)n)), 2.0));
    }
    return hidden;
  }
  // Sets S from a matrix Wg's row bounds b, on the host and on the GPU; returns the largest S_j.
  int gate_outputs(const std::vector<u64> &b, int kg) {
    S.resize(b.size());
    int top = 0;
    for (size_t j = 0; j < b.size(); j++) {
      const int s = gate_most[std::min<u64>(-QUOTIENT_LOW, quotient(b[j], kg))];
      S[j] = s, top = std::max(top, s);
    }
    CK(cudaMemcpy(scale, S.data(), S.size() * 8, cudaMemcpyHostToDevice));
    return top;
  }
  // Processes one unit: the tensor w0 of kind u.X (listed below) in layer u.l, on the GPU as
  // integers in its file's element order (int16; embedding: int32). Kind 0: w0 = q, w1 = k. Units
  // must arrive in layoutW()'s order (prover/common/weight_layout.h, which also defines `UnitW`).
  // u.X: 0 q and k, 1 v, 2 Wo, 3 Wg, 4 Wu (shared expert: 5, 6; routed: 7, 8), 9 Wd (routed: 10),
  // 11 router, 12 embedding, 13 output head.
  void take(const UnitW &u, const void *w0, const void *w1) {
    units_seen++;
    const short *w = (const short *)w0;
    const size_t d = sh.d;
    std::vector<u64> b, s;
    if (u.X == 12) {
      Clock clock(seconds_rows);
      double *rows = (double *)squares;
      k_wide_row_squares<<<132 * 8, 256>>>((const int *)w0, sh.V, (int)d, rows);
      std::vector<double> h(sh.V);
      CK(cudaMemcpy(h.data(), rows, sh.V * 8, cudaMemcpyDeviceToHost));
      embedding = sqrt_up(*std::max_element(h.begin(), h.end()));
      embedding_seen = true;
      return;
    }
    if (u.X == 13) return; // lines() bounds the output head's margins from public sizes
    if (u.X == 0 && u.l % 8 == 0) { // a progress line: the check takes minutes
      printf("cp_commit: the worst-case check: at layer %d of %d, %.0f s in the bounds so far\n", u.l, sh.L,
             seconds_rows + seconds_norms);
      fflush(stdout);
    }
    Layer &y = layer[u.l];
    const LayerWords &x = words[u.l];
    const double B1 = ball(x.M1, x.K1, eps1), B2 = ball(x.M2, x.K2, eps2);
    switch (u.X) {
    case 0: {
      // The sums (A0, A1) of rows 2 i and 2 i + 1 become (c A0 - s A1, s A0 + c A1), where c =
      // round(4096 cos) and s = round(4096 sin), or 4096 and 0 without RoPE. Each is at most
      // sqrt(c^2 + s^2) N, where N = sqrt(A0^2 + A1^2) <= min(B times the rows' largest singular
      // value, sqrt(b0^2 + b1^2)).
      // Public source: a rotary rotation keeps each pair's length: Wu et al., "PolarQuant:
      // Leveraging Polar Transformation for Key Cache Quantization and Decoding Acceleration",
      // arXiv:2502.00527, section 1; here the scale comes from rounded (c, s): not exactly 4096.
      // Public source: ||Ah||_2 <= the largest singular value of A times ||h||_2: Miyato et al.,
      // "Spectral Normalization for Generative Adversarial Networks", arXiv:1802.05957, equation 6;
      // here A is the two rows, and the value comes from their 2 x 2 Gram matrix.
      // Public source: |round(a e^(j t))| <= a + 1/sqrt(2): D. F. G. Coelho, R. J. Cintra,
      // "Discrete Fourier Transform Approximations Based on the Cooley-Tukey Radix-2 Algorithm"
      // (arXiv:2402.16225), inequality (10), times a. Its square at a = 4096 is 4096^2 + 4096
      // sqrt(2) + 1/2, rounded up here to 4096^2 + 5794.
      const double turn = sqrt_up(4096.0 * 4096.0 + 5794.0);
      for (int second = 0; second < 2; second++) {
        const short *t = second ? (const short *)w1 : w;
        const size_t rows = second ? sh.dkv : sh.dq;
        row_bounds(t, rows, d, B1, b, s);
        std::vector<long long> dot(rows / 2);
        k_pair_dots<<<(unsigned)std::min<size_t>(rows / 2, 132 * 8), 256>>>(t, rows / 2, (int)d, dots);
        CK(cudaMemcpy(dot.data(), dots, rows / 2 * 8, cudaMemcpyDeviceToHost));
        u64 most = 0;
        for (size_t i = 0; i < rows / 2; i++) {
          if (!x.rp) {
            most = std::max(most, 4096 * std::max(b[2 * i], b[2 * i + 1]));
            continue;
          }
          const double n0 = (double)s[2 * i], n1 = (double)s[2 * i + 1], p = std::fabs((double)dot[i]); // exact: below 2^53
          const double gap = std::fabs(n0 - n1);
          const double top = div_up(add_up(n0 + n1, sqrt_up(add_up(mul_up(gap, gap), mul_up(4.0, mul_up(p, p))))), 2.0);
          const double by_ball = mul_up(B1, sqrt_up(top));
          const double b0 = whole_up(b[2 * i]), b1 = whole_up(b[2 * i + 1]);
          const double by_rows = sqrt_up(add_up(mul_up(b0, b0), mul_up(b1, b1)));
          most = std::max(most, (u64)std::floor(mul_up(turn, std::min(by_ball, by_rows))));
        }
        (second ? y.k : y.q) = most;
      }
      break;
    }
    case 1: {
      // y.attention_in is the root in Delta_o (top comment, 2), with V_g the lesser of
      // B ||Wv_g|| / 2^kv + sqrt(dh)/2 and the length of the cell bounds min(CM, round(b_j/2^kv)).
      // Public source: a root of a sum over heads of squares: Kim, Papamakarios and Mnih, The
      // Lipschitz Constant of Self-Attention, Proceedings of the 38th International Conference on
      // Machine Learning, PMLR 139, 2021, Theorem 3.2.
      // Public source: a head's output is a convex combination of its value rows (there for a
      // Jacobian): Emadi, Exact Attention Sensitivity and the Geometry of Transformer Stability,
      // arXiv:2602.18849, section 2.1 and appendix C.1.
      // Both are written for one value matrix per head. Here the H/G heads of a key-value group
      // read the same slice, so its term counts H/G times, and the rounding terms are added.
      row_bounds(w, sh.dkv, d, B1, b, s);
      y.v = largest(b);
      const double half = div_up(sqrt_up((double)sh.dh), 2.0);
      double sum = 0;
      for (int g = 0; g < G; g++) {
        const double norm = operator_norm(w + (size_t)g * sh.dh * d, sh.dh, d);
        y.v_norm = std::max(y.v_norm, norm);
        double cells = 0;
        for (size_t j = (size_t)g * sh.dh; j < (size_t)(g + 1) * sh.dh; j++) {
          const double c = (double)std::min<u64>(box, quotient(b[j], x.kv));
          cells = add_up(cells, c * c);
        }
        const double V = std::min(sqrt_up(cells), add_up(std::ldexp(mul_up(B1, norm), -x.kv), half));
        const double with_rounding = add_up(V, half);
        sum = add_up(sum, mul_up(with_rounding, with_rounding));
      }
      y.attention_in = sqrt_up(mul_up((double)(H / G), sum));
      break;
    }
    case 2:
      y.o_norm = operator_norm(w, d, sh.dq);
      break;
    case 3:
    case 5:
      row_bounds(w, u.X == 3 ? sh.f : sh.fe, d, B2, b, s);
      y.gate = largest(b);
      y.gate_top = gate_outputs(b, x.kg);
      break;
    case 4:
    case 6:
      row_bounds(w, u.X == 4 ? sh.f : sh.fe, d, B2, b, s);
      if (S.size() != b.size()) stop("an up matrix arrived without its gate matrix");
      y.up = largest(b);
      y.hidden = hidden_norm(w, b, 0, b.size(), B2, x.ku, y.gated_up_norm, y.up_norm, y.hidden_at_top);
      y.rows = (double)b.size();
      break;
    case 7:
      row_bounds(w, sh.NE * sh.fe, d, B2, b, s);
      y.gate_routed = largest(b);
      y.gate_top_routed = gate_outputs(b, x.kgR);
      break;
    case 8:
      row_bounds(w, sh.NE * sh.fe, d, B2, b, s);
      if (S.size() != b.size()) stop("the experts' up matrices arrived without their gate matrices");
      y.up_routed = largest(b);
      hidden_of_expert.assign(sh.NE, ByTop{});
      hidden_at_top_of_expert.assign(sh.NE, ByTop{});
      y.rows_routed = (double)sh.fe;
      for (size_t e = 0; e < sh.NE; e++) {
        double gated, plain;
        hidden_of_expert[e] = hidden_norm(w + e * sh.fe * d, b, e * sh.fe, sh.fe, B2, x.kuR, gated, plain, hidden_at_top_of_expert[e]);
        y.routed_gated_up_norm = std::max(y.routed_gated_up_norm, gated);
        y.routed_up_norm = std::max(y.routed_up_norm, plain);
        y.routed_hidden = std::max(y.routed_hidden, hidden_of_expert[e][0]);
      }
      break;
    case 9:
      y.down_norm = operator_norm(w, d, experts(u.l) ? sh.fe : sh.f);
      break;
    case 10: {
      // Only the largest ||Wd_e|| ||h_e||_2 is used. In descending order of ||h_e||_2, ||Wd_e|| is
      // refined only while its product could be the largest: routed_down_norm may be coarse.
      if (hidden_of_expert.size() != sh.NE) stop("the experts' down matrices arrived without their up matrices");
      std::vector<size_t> order(sh.NE);
      for (size_t e = 0; e < sh.NE; e++) order[e] = e;
      std::sort(order.begin(), order.end(),
                [&](size_t a, size_t c) { return hidden_of_expert[a][0] > hidden_of_expert[c][0]; });
      for (size_t e : order) {
        const ByTop &h = hidden_of_expert[e], &t = hidden_at_top_of_expert[e];
        double enough = INFINITY; // a norm at or below this raises no entry of y.routed or y.routed_at_top
        for (int c = 0; c < N_TOPS; c++) {
          if (h[c] > 0) enough = std::min(enough, y.routed[c] / h[c] * (1 - 1e-9));
          if (t[c] > 0) enough = std::min(enough, y.routed_at_top[c] / t[c] * (1 - 1e-9));
        }
        const double norm = operator_norm(w + e * d * sh.fe, d, sh.fe, nullptr, enough);
        for (int c = 0; c < N_TOPS; c++) {
          y.routed[c] = std::max(y.routed[c], mul_up(norm, h[c]));
          y.routed_at_top[c] = std::max(y.routed_at_top[c], mul_up(norm, t[c]));
        }
        y.routed_down_norm = std::max(y.routed_down_norm, norm);
      }
      hidden_of_expert.clear(), hidden_at_top_of_expert.clear();
      break;
    }
    case 11:
      row_bounds(w, sh.NE, d, B2, b, s);
      y.router = largest(b);
      break;
    default:
      stop("a unit of an unknown kind");
    }
    if (u.X != 3 && u.X != 5 && u.X != 7) S.clear();
  }
  // Hypothetical parameters: lines() recomputes the table under them from what take() recorded, to
  // show what a change would gain without a second pass over the weights. Every line remains an
  // upper bound, but may be coarser than a run under them would give. They never count for the
  // verdict. eres_down lowers e_res, the stream's number of fractional bits.
  struct WhatIf {
    int eg = GATE_KEY_BITS, smax = GATE_TOPS[0], rcl = 0, gcl = 0, eres_down = 0;
    int xl = 4;          // the limbs of a stream cell; with 5, a norm's sum of squares is assumed wide enough for them
    int root_bits = 0;   // with xl 5: the stream's assumed length is 2^root_bits, the least that closes (settled())
    std::string text;
  };
  // Parses a list like "eg=4,xl=5"; false for another form or value. Accepted: eg 4 to 8; smax in
  // GATE_TOPS, at most the file's; rcl, gcl 0 or 1; eres-down <= min(10, e_res); xl 4, 5.
  bool parse(const char *text, WhatIf &set) const {
    set.eg = eg, set.smax = smax, set.rcl = rcl, set.gcl = gcl, set.xl = xl, set.text = text;
    std::string all = text;
    for (size_t at = 0; at <= all.size();) {
      const size_t comma = std::min(all.find(',', at), all.size()), equals = all.find('=', at);
      if (equals == std::string::npos || equals > comma) return false;
      const std::string name = all.substr(at, equals - at), digits = all.substr(equals + 1, comma - equals - 1);
      if (digits.empty() || digits.size() > 4 || digits.find_first_not_of("0123456789") != std::string::npos) return false;
      const int value = atoi(digits.c_str());
      if (name == "eg" && value >= 4 && value <= 8)
        set.eg = value;
      else if (name == "smax" && std::find(GATE_TOPS, GATE_TOPS + N_TOPS, value) != GATE_TOPS + N_TOPS && value <= smax)
        set.smax = value;
      else if (name == "rcl" && value <= 1)
        set.rcl = value;
      else if (name == "gcl" && value <= 1)
        set.gcl = value;
      else if (name == "eres-down" && value <= 10)
        set.eres_down = value;
      else if (name == "xl" && (value == 4 || value == 5))
        set.xl = value;
      else
        return false;
      at = comma + 1;
    }
    return !set.eres_down || stream_step() >= set.eres_down;
  }
  // Returns e_res: the e in 0 .. 24 for which all three norms' eps equal
  // floor((2 d 4^e + 100,000) / 200,000), the rule in scales/words_to_policy.py and the integer
  // form in scales/README.md, "The epsilon of a norm". Else -1.
  int stream_step() const {
    for (int e = 0; e <= 24; e++)
      if ((2 * (u64)sh.d * (1ull << (2 * e)) + 100000) / 200000 == eps1 && eps1 == eps2 && eps1 == epsF) return e;
    return -1;
  }
  // Whether every bound on the stream's length is below 2^bits.
  static bool closes(const std::vector<Line> &t, int bits) {
    for (const Line &x : t)
      if (!x.site.rfind("stream", 0) && !(x.bound < std::ldexp(1.0, bits))) return false;
    return true;
  }
  // For a WhatIf's xl 5 under the parameter file's xl 4. A longer stream makes each ball larger
  // (the norm's R is rounded), and the balls bound the stream. So a length 2^n is assumed; it
  // closes if the bound that follows is below 2^n. Returns `set` with root_bits the smallest such n
  // in 30 .. 37, or 38. 2^38 is within five limbs (2^38.99) and the sum of squares (length 2^39).
  WhatIf settled(WhatIf set) const {
    for (set.root_bits = 30; set.root_bits < 38 && !closes(lines(set), set.root_bits);) set.root_bits++;
    return set;
  }
  std::vector<Line> lines(int c = 0) const {
    WhatIf set;
    set.eg = eg, set.smax = top_of(c), set.rcl = rcl, set.gcl = gcl, set.xl = xl;
    return lines(set);
  }
  // The table's lines, in forward-pass order. Call it after every unit: only table() checks that.
  //
  // With the parameter file's eg, e_res and xl the lines use what take() recorded. Otherwise:
  //   - a smaller e_res (so a smaller eps) or five limbs make each ball larger by a factor rho. The
  //     larger ball cut with the box is inside rho times the old one, so rho times an old
  //     bound on a sum is valid; a quotient's cell y grows to at most rho |y| + 1;
  //   - the keys over which S_j is a maximum change by a rounding, so every S_j becomes smax;
  //   - the fitted shifts follow: kg and kgR rise as eg falls, ko and kd as e_res falls; the
  //     embedding is rounded again, for which one step per cell, sqrt(d) in all, is added.
  // The bounds are computed from the weights: cp_commit prints the verdict and no bound, and writes
  // the table only to the file of -worst-case-table, for its owner only.
  //
  // Public sources: a star-shaped set cut by a growing ball: Plan, Vershynin and Yudovina,
  // High-dimensional estimation with geometric constraints, 2014, https://arxiv.org/abs/1404.3749,
  // proof of Corollary 8.3 (there, for the set K - K in an estimator's error bound and as a
  // well-known fact: a linear form's maximum over the cut set, divided by the radius, does not grow
  // with the radius); the same step is in Bartlett, Bousquet and Mendelson, Local Rademacher
  // complexities, The Annals of Statistics 33 (4), 2005, proof of Lemma 3.4.
  // Here the set is the box.
  // Public source: an output shift from power-of-two scales: Arm, Converting a Neural Network for
  // Arm Cortex-M with CMSIS-NN, 'Compute the layer shifts'.
  // Public source: in general M = S1 S2 / S3: Jacob et al., Quantization and Training of Neural
  // Networks for Efficient Integer-Arithmetic-Only Inference, 2017,
  // https://arxiv.org/abs/1712.05877, equation (5). Both state it for converting a network once;
  // here a hypothetical change of an output's scale moves the fitted shifts by as many bits.
  // Public source: rounding to a grid of step s moves a value by at most s/2: Nagel et al., A White
  // Paper on Neural Network Quantization, 2021, https://arxiv.org/abs/2106.08295, section 2.2.
  // Public source: a vector of m cells moves by at most sqrt(m) s/2 in Euclidean length: Boufounos
  // et al., Quantization and Compressive Sensing, 2014, https://arxiv.org/abs/1405.1194, section
  // 3.1. Here one whole step per cell covers the new rounding and the rounding already in the
  // stored integers.
  std::vector<Line> lines(const WhatIf &asked) const {
    if (asked.xl == 5 && xl != 5 && !asked.root_bits) {
      const WhatIf set = settled(asked);
      std::vector<Line> t = lines(set);
      // If no assumed length closes, the balls rest on an unconfirmed length: no line may fit.
      if (!closes(t, set.root_bits))
        for (Line &x : t) x.shortfall = std::max(x.shortfall, 1);
      return t;
    }
    const WhatIf &set = asked;
    const int wide = set.root_bits;
    int c = 0;
    while (c + 1 < N_TOPS && top_of(c) > set.smax) c++;
    const int down = set.eres_down, coarser = eg - set.eg;
    const bool scaled = down || coarser || wide;
    const u64 eps_then = down ? (2 * (u64)sh.d * (1ull << (2 * (stream_step() - down))) + 100000) / 200000 : 0;
    const u64 e1 = down ? eps_then : eps1, e2 = down ? eps_then : eps2, eF = down ? eps_then : epsF;
    std::vector<Line> t;
    auto line = [&](const char *site, int l, const char *word, int value, double bound, double limit) {
      t.push_back({site, l, word, value, bound, limit, shortfall_of(bound, limit)});
    };
    // round(A / 2^k) <= top means A <= (2 top + 1) 2^(k-1) - 1; the lower ends are further out.
    static_assert(-QUOTIENT_LOW > QUOTIENT_TOP, "the upper end is the nearer one");
    auto sum = [&](const char *site, int l, const char *word, int k, u64 A, int top, double rho) {
      line(site, l, word, k, mul_up(rho, whole_up(A)), std::ldexp((double)(2 * top + 1), k - 1) - 1);
    };
    // ||x||_2 <= floor(sqrt(2^59 - 1)) keeps the sum of squares below 2^59 and every cell inside
    // four limbs. Under the parameter file's own xl 5 the cell's upper end is the stricter limit:
    // a row of at most that length has a sum of squares below 2^77.99, and eps is below 2^59.
    static_assert((double)WIDE_STREAM_TOP * (double)WIDE_STREAM_TOP + 576460752303423488.0 < 302231454903657293676544.0, "2^78");
    // In a WhatIf with five limbs the length 2^wide is only assumed, to bound the
    // balls (settled()). The lines compare the bounds with the format's limit.
    const double stream_most = wide || xl == 5 ? (double)WIDE_STREAM_TOP : (double)isqrt((1ull << SQUARES_BITS) - 1);
    static_assert((1ull << SQUARES_BITS) <= (unsigned long long)STREAM_TOP * STREAM_TOP, "a row that short has no cell beyond STREAM_TOP");
    const double half = div_up(sqrt_up((double)sh.d), 2.0);
    double X = down ? add_up(std::ldexp(embedding, -down), sqrt_up((double)sh.d)) : embedding;
    for (int l = 0; l < sh.L; l++) {
      const Layer &y = layer[l];
      const LayerWords &x = words[l];
      const bool grows = down || wide;
      const double rho1 = grows ? std::max(1.0, div_up(ball(x.M1, x.K1, e1, false, wide), ball(x.M1, x.K1, eps1))) : 1.0;
      const double rho2 = grows ? std::max(1.0, div_up(ball(x.M2, x.K2, e2, false, wide), ball(x.M2, x.K2, eps2))) : 1.0;
      line("stream at norm 1", l, "e_res", 0, X, stream_most);
      line("norm 1 quotient", l, "K1", x.K1, ball(x.M1, x.K1, e1, true, wide), QUOTIENT_TOP);
      sum("q", l, "kq", x.kq, y.q, QUOTIENT_TOP, rho1);
      sum("k", l, "kk", x.kk, y.k, QUOTIENT_TOP, rho1);
      sum("v", l, "kv", x.kv, y.v, QUOTIENT_TOP, rho1);
      const double heads = grows ? add_up(mul_up(rho1, y.attention_in), sqrt_up((double)sh.dq)) : y.attention_in;
      X = add_up(X, add_up(std::ldexp(mul_up(y.o_norm, heads), -(x.ko + down)), half));
      line("stream at norm 2", l, "e_res", 0, X, stream_most);
      line("norm 2 quotient", l, "K2", x.K2, ball(x.M2, x.K2, e2, true, wide), QUOTIENT_TOP);
      // Under rcl 1 the wide logit is 8 y + w, where y is a quotient of two limbs and 0 <= w < 8.
      if (experts(l)) sum("router", l, "kr", x.kr, y.router, set.rcl ? 8 * QUOTIENT_TOP + 7 : ROUTER_TOP, rho2);
      const int gate_top = set.gcl ? WIDE_GATE_TOP : QUOTIENT_TOP;
      sum(experts(l) ? "gate, shared" : "gate", l, "kg", x.kg + coarser, y.gate, gate_top, rho2);
      sum(experts(l) ? "up, shared" : "up", l, "ku", x.ku, y.up, QUOTIENT_TOP, rho2);
      if (experts(l)) {
        sum("gate, routed", l, "kgR", x.kgR + coarser, y.gate_routed, gate_top, rho2);
        sum("up, routed", l, "kuR", x.kuR, y.up_routed, QUOTIENT_TOP, rho2);
      }
      const double own = scaled ? mul_up(rho2, y.hidden_at_top[c]) : y.hidden[c];
      const double of_experts = scaled ? mul_up(rho2, y.routed_at_top[c]) : y.routed[c];
      double add = add_up(std::ldexp(add_up(mul_up(y.down_norm, own), of_experts), -(x.kd + down)), half);
      if (dcl) { // the wide addition is 2^dcl y + w, y a quotient of two limbs; no cell exceeds the addition's length
        line("down quotient", l, "kd", x.kd, add, std::ldexp((double)QUOTIENT_TOP + 1, dcl) - 1);
        add = std::min(add, mul_up(sqrt_up((double)sh.d), std::ldexp(1.0, 11 + dcl)));
      }
      X = add_up(X, add);
    }
    line("stream at the last norm", sh.L, "e_res", 0, X, stream_most);
    // No cell exceeds the row's length.
    line("last stream, a cell", sh.L, "e_res", 0, X,
         wide || xl == 5 ? (double)WIDE_STREAM_TOP : std::ldexp(1.0, LAST_STREAM_BITS) - 1);
    line("last norm quotient", sh.L, "KF", KF, ball(MF, KF, eF, true, wide), QUOTIENT_TOP);
    // A margin is a difference of two logits, minus 0 or 1, plus what its position adds: at most
    // 2 d CM 4,095 + 1 + 2^37 + 2^32 in absolute value. The head's |w| <= 4,095 is not tested here.
    line("head margin", sh.L, "-", 0,
         2.0 * (double)sh.d * box * WEIGHT_MOST + 1 + std::ldexp(1.0, HEAD_LIFT_BITS) + std::ldexp(1.0, 32),
         std::ldexp(65537.0, HEAD_LOW_BITS + 15) - 1);
    // If a guard failed or a weight was out of range (why_no_bound), no bound is trusted: every shortfall is NO_BOUND.
    if (why_no_bound)
      for (Line &x : t) x.shortfall = NO_BOUND;
    return t;
  }
  static double log2_up(double v) { // only for printing: rounded up to three decimal places
    return v > 0 ? std::ceil(std::log2(v) * 1000 + 1e-6) / 1000 : 0;
  }
  // One line per site: its worst layer, and how many layers are short by 1, 2, 3, 4 or more bits.
  static void by_site(FILE *to, const std::vector<Line> &t) {
    std::vector<std::string> order;
    std::map<std::string, const Line *> worst;
    std::map<std::string, std::array<int, 5>> layers_short;
    for (const Line &x : t) {
      if (!worst.count(x.site)) order.push_back(x.site), worst[x.site] = &x;
      const Line *&held = worst[x.site];
      if (x.bound / x.limit > held->bound / held->limit) held = &x;
      layers_short[x.site][std::min(x.shortfall, 4)]++;
    }
    fprintf(to, "%-26s %6s %6s %6s %11s %11s %10s %11s   %s\n", "site", "layer", "param", "value", "log2 bound",
            "log2 limit", "bits over", "bits short", "layers short by 1, 2, 3, 4 or more bits");
    for (const std::string &site : order) {
      const Line &x = *worst[site];
      const std::array<int, 5> &n = layers_short[site];
      fprintf(to, "%-26s %6d %6s %6d %11.3f %11.3f %+10.3f %11d   %d %d %d %d\n", site.c_str(), x.layer, x.word.c_str(), x.value,
              log2_up(x.bound), std::log2(x.limit + 1), log2_up(x.bound) - std::log2(x.limit + 1), x.shortfall, n[1], n[2],
              n[3], n[4]);
    }
  }
  // Writes the whole table to `out`, each site's worst layer to `brief`, and to both the closing
  // lines and a table by site for each of `sets`. Returns the verdict: whether every bounded site fits.
  bool table(FILE *out, FILE *brief, size_t units_in_the_layout, const std::vector<WhatIf> &sets = {}) const {
    if (units_seen != units_in_the_layout || !embedding_seen) stop("not every unit of the layout was processed");
    const std::vector<Line> t = lines();
    bool fits = true;
    for (const Line &x : t) fits = fits && !x.shortfall;
    if (out) {
      fprintf(out, "worst-case 1\nd %zu\n", sh.d);
      fprintf(out, "# site | layer | parameter | its value | log2 bound | log2 limit | whole bits short\n");
      for (const Line &x : t)
        fprintf(out, "site | %s | %d | %s | %d | %.3f | %.3f | %d\n", x.site.c_str(), x.layer, x.word.c_str(), x.value,
                log2_up(x.bound), std::log2(x.limit + 1), x.shortfall);
      fprintf(out, "# layer | log2 of: the addition after attention, ||Wo||, largest ||Wv of a group||, ||heads' "
                   "outputs|| | the addition after the feed-forward block, ||Wd||, ||diag(S) Wu||, ||Wu||, ||h||, "
                   "largest S | over the experts, the largest of each: ||Wd_e|| ||h_e||, ||Wd_e||, ||diag(S) Wu_e||, "
                   "||Wu_e||, ||h_e||, S\n");
      for (int l = 0; l < sh.L; l++) {
        const Layer &y = layer[l];
        const LayerWords &x = words[l];
        fprintf(out, "layer | %d | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f %.3f %d | %.3f %.3f %.3f %.3f %.3f %d\n", l,
                log2_up(std::ldexp(mul_up(y.o_norm, y.attention_in), -x.ko)), log2_up(y.o_norm), log2_up(y.v_norm),
                log2_up(y.attention_in),
                log2_up(std::ldexp(add_up(mul_up(y.down_norm, y.hidden[0]), y.routed[0]), -x.kd)), log2_up(y.down_norm),
                log2_up(y.gated_up_norm), log2_up(y.up_norm), log2_up(y.hidden[0]), y.gate_top, log2_up(y.routed[0]),
                log2_up(y.routed_down_norm), log2_up(y.routed_gated_up_norm), log2_up(y.routed_up_norm),
                log2_up(y.routed_hidden), y.gate_top_routed);
      }
      fprintf(out, "embedding | %.3f\n", log2_up(embedding));
    }
    // Prints where the stream's bound comes from, and what other parameters would gain. The
    // stream's bound is largest at the last norm, since every addition's bound is added to it.
    auto closing = [&](FILE *to) {
      double after_attention = 0, dense = 0, shared = 0, routed = 0, last = 0;
      std::vector<std::pair<double, int>> adds;
      for (int l = 0; l < sh.L; l++) {
        const Layer &y = layer[l];
        const LayerWords &x = words[l];
        const double own = std::ldexp(mul_up(y.down_norm, y.hidden[0]), -x.kd), of_experts = std::ldexp(y.routed[0], -x.kd);
        after_attention += std::ldexp(mul_up(y.o_norm, y.attention_in), -x.ko);
        (experts(l) ? shared : dense) += own;
        routed += of_experts;
        adds.push_back({own + of_experts, l});
      }
      for (const Line &x : t)
        if (x.site == "stream at the last norm") last = x.bound;
      fprintf(to, "# split | log2 of the parts of the stream's bound at the last norm: the embedding | the additions "
                  "after attention | the feed-forward additions of layers without experts | of the shared parts | of "
                  "the routed parts\n");
      fprintf(to, "split | %.3f | %.3f | %.3f | %.3f | %.3f\n", log2_up(embedding), log2_up(after_attention), log2_up(dense),
              log2_up(shared), log2_up(routed));
      std::sort(adds.rbegin(), adds.rend());
      fprintf(to, "# largest | the five largest feed-forward additions: layer, log2\nlargest");
      for (size_t i = 0; i < adds.size() && i < 5; i++) fprintf(to, " | %d %.3f", adds[i].second, log2_up(adds[i].first));
      fprintf(to, "\n# needs | whole bits needed for the stream's bound: for the absolute value of a cell | for a "
                  "norm's sum of squares\n");
      fprintf(to, "needs | %d | %d\n", (int)std::ceil(std::log2(last) + 1e-9), (int)std::ceil(2 * std::log2(last) + 1e-9));
      fprintf(to, "# what if | the limit on the gate table's output (the parameter file's smax) | log2 of the stream's "
                  "bound at the last norm | whole bits short\n");
      for (int c = 0; c < N_TOPS; c++)
        for (const Line &x : lines(c))
          if (x.site == "stream at the last norm" && (!c || top_of(c) < top_of(c - 1)))
            fprintf(to, "what if | %d%s | %.3f | %d\n", top_of(c), c ? "" : " (the parameter file's)", log2_up(x.bound),
                    x.shortfall);
      for (const WhatIf &set : sets) {
        const std::vector<Line> under = lines(set);
        bool all = true;
        for (const Line &x : under) all = all && !x.shortfall;
        fprintf(to, "under other words | %s | eg %d smax %d rcl %d gcl %d e_res %d xl %d | %s\n", set.text.c_str(), set.eg,
                set.smax, set.rcl, set.gcl, stream_step() - set.eres_down, set.xl,
                all ? "every site would fit" : "NOT every site would fit");
        if (set.xl == 5 && xl != 5) {
          const WhatIf with = settled(set);
          if (closes(lines(with), with.root_bits))
            fprintf(to, "# (to bound the balls after the norms, the Euclidean length of a row of the stream was "
                        "assumed to be below 2^%d, and its bound confirms this; the limits below are the format's)\n",
                    with.root_bits);
          else
            fprintf(to, "# (no assumed Euclidean length of a row of the stream up to 2^38 is confirmed by the bound "
                        "that follows from it: the balls after the norms are not bounded, and every line below counts "
                        "as short)\n");
        }
        by_site(to, under);
      }
    };
    if (out) {
      closing(out);
      fprintf(out, "verdict %s\n", fits ? "fits" : "short");
    }
    if (brief) {
      fprintf(brief, "worst case over all prompts: for each site, its worst layer\n");
      by_site(brief, t);
      closing(brief);
    }
    return fits;
  }
};
} // namespace worst_case
} // namespace rf
