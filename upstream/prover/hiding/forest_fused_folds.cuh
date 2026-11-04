// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Lookup forest (the product trees of a group of lookup calls): folds of the layers nearest the
// leaves, computed from the lookup keys on the tensor cores. All sums are exact: fold() and
// third_fold() write the same elements as their namesakes in prover/hiding/forest_keys.cuh
// (namespace fk), which multiply in limbs. While `forest_fused` is set, prover/hiding/forest.cuh
// calls them instead, fold() only for nodes of 2 or 4 leaves (LV 1 or 2). Neither checks dT or the
// keys' alignment. Include after prover/hiding/forest_fused.cuh.
//
// A tree of N = 2^dT keys (16-bit row numbers of a public table) has leaves alpha - tau(k): alpha
// is its challenge, tau(k) the table's value for key k (in a range table, k itself). A node is the
// product of D leaves, sum_i (-1)^i e_i alpha^(D - i), where e_i is the i-th elementary symmetric
// polynomial of their values. A fold replaces the nodes at index c of eight slabs (equal parts of a
// layer's table) by a weighted sum, with extension-field constants K (`T`: 3 coordinates mod
// p = 2^64 - 2^32 + 1):
//   out(c) = K_0 + sum over (a, i >= 1) of e_i(node c of slab a) K_(a,i).
// Here the sum is a product of byte matrices. The first holds the e_i unchanged: a 32-bit word is
// four bytes of the summed dimension. The second holds, for byte b of a word with constant K, the
// 24 signed digits (bytes from -128 to 127) of K 2^(8 b) mod p, eight for each coordinate. A row of
// the product is one output's 24 sums, which from_sums() joins into the canonical coordinates.
// Public sources. Products of numbers cut into bytes on an integer matrix unit, joined by powers of
// 256 and reduced: [TensorFHE], [Ozaki-int8] (cited in prover/hiding/gram_products.cuh). The
// instruction, here with unsigned bytes in the first matrix, and its fragments: [mma]
// (prover/hiding/forest_fused.cuh). A weighted sum of vectors as a product with a constant matrix:
// no source was looked for.
#pragma once
namespace ff {
// d += a b as in multiply() (forest_fused.cuh), with unsigned bytes in a: 16 rows of a times 8 rows
// of b over one step, 32 bytes, of the summed dimension.
// But for the operand types (u8 for s8), the asm template and the order of its operands are the
// same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.
__device__ __forceinline__ void multiply_unsigned(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Warps in a block of k_fold_range() and k_fold64_range(); both rely on it and do not check it.
constexpr int FOLD_WARPS = 4;
// A tile of sums in shared memory, in 32-bit words: 16 rows (outputs) of 24 sums. Four words of
// padding shift the next tile by four banks, which spreads the reads of from_sums() over the banks.
constexpr int FOLD_TILE = 16 * 24 + 4;
// The word of the second matrix that thread (g, t4), lane 4 g + t4, holds for a word of the first
// with constant K and weight 2^shift: byte b < 4 is digit g of coordinate j of K 2^(shift + 8 b).
__device__ __forceinline__ u32 constant_word(const T &K, int j, int shift, int g) {
  u32 w = 0;
#pragma unroll
  for (int b = 0; b < 4; b++)
    w |= (u32)((grm::signed_digits(grm::gram_times_2_to(coord(K, j), (shift + 8 * b) % grm::TURNS)) >> (8 * g)) & 255) << (8 * b);
  return w;
}
// Stores a tile's sums: sum[j][e] is that of row g + 8 (e >> 1) with digit 2 t4 + (e & 1) of
// coordinate j.
__device__ __forceinline__ void put_sums(int *tile, const int (&sum)[3][4], int g, int t4) {
#pragma unroll
  for (int j = 0; j < 3; j++)
#pragma unroll
    for (int h = 0; h < 2; h++)
      *reinterpret_cast<int2 *>(tile + (g + 8 * h) * 24 + 8 * j + 2 * t4) = make_int2(sum[j][2 * h], sum[j][2 * h + 1]);
}
// Each sum of an output adds `bytes` products of an unsigned byte and a signed digit of a constant,
// one for each byte of the output's words. from_sums() needs |sum| < 2^24.
constexpr long long FOLD_SUM_MOST = 1ll << 24;
__host__ __device__ constexpr bool fold_sum_fits(long long bytes) {
  return bytes * BYTE_MOST * DIGIT_MOST < FOLD_SUM_MOST;
}
static_assert(FOLD_SUM_MOST * ((1ll << 24) + (1ll << 16) + (1ll << 8) + 1) < TURN_MOST, "from_sums() turns two sums of four such sums, shifted by 0 to 24 bits");
// Returns start + sum over m < 8 of 2^(8 m) s[m] mod p, canonical. Needs |s[m]| < FOLD_SUM_MOST, a
// canonical start, and s 16-byte aligned; FOLD_TILE and a row's 24 sums are multiples of four words
// for that.
__device__ __forceinline__ u64 from_sums(const int *s, u64 start) {
  const int4 x = *reinterpret_cast<const int4 *>(s), y = *reinterpret_cast<const int4 *>(s + 4);
  const long long lo = (long long)x.x + ((long long)x.y << 8) + ((long long)x.z << 16) + ((long long)x.w << 24),
                  hi = (long long)y.x + ((long long)y.y << 8) + ((long long)y.z << 16) + ((long long)y.w << 24);
  return gf_add(start, gf_add(grm::signed_turn(lo, 0), grm::signed_turn(hi, 32)));
}
// A node of D = 2^LV 16-bit keys has coefficients e_1 .. e_D of 17 and 32 bits (D = 2) or 18, 35,
// 50 and 64 bits (D = 4). 32-bit word w is bits shift(w) to shift(w) + 31 of coefficient of(w).
template <int LV> struct NodeWords;
template <> struct NodeWords<1> {
  static constexpr int WORDS = 2;
  static __host__ __device__ constexpr int of(int w) { return w + 1; }
  static __host__ __device__ constexpr int shift(int) { return 0; }
};
template <> struct NodeWords<2> {
  static constexpr int WORDS = 7;
  static __host__ __device__ constexpr int of(int w) { return (w + 1) / 2 + 1; }
  static __host__ __device__ constexpr int shift(int w) { return w && !(w & 1) ? 32 : 0; }
};
// As fk::k_fold_poly<LV, fk::RangeSrc>: Lo[c] = sum_a k8[a] L_a(c), Ro[c] = sum_a k8[a] R_a(c), for
// c < SC = 2^(dT - LV - 4) and slabs L_a, R_a of the layer's two tables. A warp takes 64 outputs of
// one side as four tiles of 16 rows; row r of tile x is output 4 r + x, so that a thread reads four
// keys as one 64-bit word. Step w of the summed dimension is word w of the eight slabs' nodes.
template <int LV>
__global__ void __launch_bounds__(32 * FOLD_WARPS)
    k_fold_range(const uint16_t *__restrict__ keys, int dT, const T *alpha, const T *__restrict__ k8, T *__restrict__ Lo,
                 T *__restrict__ Ro) {
  typedef NodeWords<LV> W;
  constexpr int D = 1 << LV;
  static_assert(fold_sum_fits(W::WORDS * 8 * 4), "an output's sums run over the words of eight slabs' nodes, four bytes each");
  __shared__ T constant[8][D + 1]; // [a][i]: (-1)^i alpha^(D - i) k8[a]
  __shared__ __align__(16) int room[FOLD_WARPS][4 * FOLD_TILE];
  if (threadIdx.x < 8) {
    T u = k8[threadIdx.x];
    const T alpha_here = *alpha;
    for (int i = D; i >= 0; i--) {
      constant[threadIdx.x][i] = (i & 1) ? F::sub(F::zero(), u) : u;
      u = F::mul(u, alpha_here);
    }
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  u32 b[W::WORDS][3][2];
#pragma unroll
  for (int w = 0; w < W::WORDS; w++)
#pragma unroll
    for (int j = 0; j < 3; j++)
#pragma unroll
      for (int h = 0; h < 2; h++) b[w][j][h] = constant_word(constant[t4 + 4 * h][W::of(w)], j, W::shift(w), g);
  T start = F::zero();
  for (int a = 0; a < 8; a++) start = F::add(start, constant[a][0]);
  const int d = dT - LV;
  const size_t SC = (size_t)1 << (d - 4), batches = 2 * SC / 64;
  int *mine = room[warp];
  for (size_t batch = blockIdx.x * (size_t)FOLD_WARPS + warp; batch < batches; batch += (size_t)gridDim.x * FOLD_WARPS) {
    const bool second = batch >= SC / 64;
    const size_t c0 = (second ? batch - SC / 64 : batch) * 64;
    uint2 q[2][2][D]; // [slab t4 + 4 s][rows g + 8 r][leaf]
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int j = 0; j < D; j++)
          q[s][r][j] = fk::four_keys(keys + ((size_t)((second ? 8 : 0) + t4 + 4 * s) << (d - 4)) + c0 + 4 * (g + 8 * r) + ((size_t)j << d));
#pragma unroll
    for (int x = 0; x < 4; x++) {
      u64 cf[2][2][D];
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          u64 t[D];
#pragma unroll
          for (int j = 0; j < D; j++) t[j] = fk::key_of(q[s][r][j], x);
          fk::Coefficients<D, true>::of(t, cf[s][r]);
        }
      int sum[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
      for (int w = 0; w < W::WORDS; w++) {
        const int i = W::of(w) - 1, sh = W::shift(w);
        const u32 a[4] = {(u32)(cf[0][0][i] >> sh), (u32)(cf[0][1][i] >> sh), (u32)(cf[1][0][i] >> sh), (u32)(cf[1][1][i] >> sh)};
#pragma unroll
        for (int j = 0; j < 3; j++) multiply_unsigned(sum[j], a, b[w][j]);
      }
      put_sums(mine + x * FOLD_TILE, sum, g, t4);
    }
    __syncwarp();
    u64 *out = reinterpret_cast<u64 *>((second ? Ro : Lo) + c0);
#pragma unroll
    for (int u = 0; u < 6; u++) {
      const int id = lane + 32 * u, c = id / 3, j = id % 3; // output c0 + c is row c >> 2 of tile c & 3
      out[id] = from_sums(mine + (c & 3) * FOLD_TILE + (c >> 2) * 24 + 8 * j, coord(start, j));
    }
    __syncwarp();
  }
}
// As grm::k_fold64<grm::Key> (gram_products.cuh): U(r) = alpha - sum_(j < 64) K[j] key(side + 64 r
// + j), r < rows = N / 128, written at (r & 7) CO + (r >> 3), CO = N / 1024, to UL (side 0) and UR
// (side N / 2). It is the leaves' weighted sum only if the 64 weights K add up to 1, which is not
// checked. A warp takes 64 outputs. An output's keys are read unchanged, two to a 32-bit word, as
// eight 128-bit words; step s is 32-bit word s of each.
__global__ void __launch_bounds__(32 * FOLD_WARPS)
    k_fold64_range(const uint4 *__restrict__ key, size_t rows, size_t CO, const T *__restrict__ K, const T *__restrict__ alpha,
                   T *__restrict__ UL, T *__restrict__ UR) {
  static_assert(fold_sum_fits(64 * 2), "an output's sums run over 64 keys of two bytes");
  __shared__ __align__(16) int room[FOLD_WARPS][4 * FOLD_TILE];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  u32 b[4][3][2];
#pragma unroll
  for (int s = 0; s < 4; s++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      // 32-bit word s of 128-bit word t4 + 4 h holds keys 8 (t4 + 4 h) + 2 s and the next; their
      // weights enter with a minus sign.
      const int first = 8 * (t4 + 4 * h) + 2 * s;
      const T k0 = F::sub(F::zero(), K[first]), k1 = F::sub(F::zero(), K[first + 1]);
#pragma unroll
      for (int j = 0; j < 3; j++)
        b[s][j][h] = (constant_word(k0, j, 0, g) & 0xFFFFu) | (constant_word(k1, j, 0, g) << 16);
    }
  const T start = *alpha;
  int *mine = room[warp];
  const size_t batches = 2 * rows / 64;
  for (size_t batch = blockIdx.x * (size_t)FOLD_WARPS + warp; batch < batches; batch += (size_t)gridDim.x * FOLD_WARPS) {
    const uint4 *from = key + batch * 64 * 8; // the second side's keys follow the first's
#pragma unroll
    for (int x = 0; x < 4; x++) {
      uint4 q[2][2]; // [word t4 + 4 h][row g + 8 r]
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int r = 0; r < 2; r++) q[h][r] = __ldg(from + (size_t)(16 * x + g + 8 * r) * 8 + t4 + 4 * h);
      int sum[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
      for (int s = 0; s < 4; s++) {
        auto word = [&](const uint4 &v) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; };
        const u32 a[4] = {word(q[0][0]), word(q[0][1]), word(q[1][0]), word(q[1][1])};
#pragma unroll
        for (int j = 0; j < 3; j++) multiply_unsigned(sum[j], a, b[s][j]);
      }
      put_sums(mine + x * FOLD_TILE, sum, g, t4);
    }
    __syncwarp();
    const bool second = batch >= rows / 64;
    const size_t r0 = (second ? batch - rows / 64 : batch) * 64; // a multiple of 64: outputs r0 + 8 c + a, c < 8, a < 8
    u64 *out = reinterpret_cast<u64 *>(second ? UR : UL);
#pragma unroll
    for (int u = 0; u < 6; u++) {
      const int id = lane + 32 * u, e = id / 3, j = id % 3, a = e >> 3, c = e & 7, r = 8 * c + a;
      out[((size_t)a * CO + (r0 >> 3) + c) * 3 + j] = from_sums(mine + (r >> 4) * FOLD_TILE + (r & 15) * 24 + 8 * j, coord(start, j));
    }
    __syncwarp();
  }
}
inline int fold_blocks(size_t batches) {
  return (int)std::min<size_t>((batches + FOLD_WARPS - 1) / FOLD_WARPS, (size_t)g_sms * 8);
}
// fk::fold() for a range table. LV must be 1 or 2: it is not checked, and any other runs as 2.
inline void fold_range(int LV, const fk::Tree &t, int dT, const T *alpha, const T *k8, T *Lo, T *Ro, cudaStream_t s) {
  const int blocks = fold_blocks((size_t)1 << (dT - LV - 9));
  if (LV == 1)
    k_fold_range<1><<<blocks, 32 * FOLD_WARPS, 0, s>>>(t.keys, dT, alpha, k8, Lo, Ro);
  else
    k_fold_range<2><<<blocks, 32 * FOLD_WARPS, 0, s>>>(t.keys, dT, alpha, k8, Lo, Ro);
  CK(cudaGetLastError());
}
// fk::third_fold() for a range table.
inline void third_fold_range(const fk::Tree &t, int dT, const T *alpha, const T *K64, T *UL, T *UR, cudaStream_t s) {
  const size_t N = (size_t)1 << dT;
  k_fold64_range<<<fold_blocks(N / 4096), 32 * FOLD_WARPS, 0, s>>>((const uint4 *)t.keys, N / 128, N / 1024, K64, alpha, UL, UR);
  CK(cudaGetLastError());
}
// ---- The same folds for function tables (every fk::Kind but RANGE). A value tau(k) has up to 61
// bits, and a node's coefficients are 64-bit words that stand for residues mod p
// (fk::Coefficients<D, false>). A fold is linear in them mod p, so any representative gives the
// same output. A block has 16 warps, so that a table that fk::Form<S> copies to shared memory
// serves many; a warp takes 32 outputs, and its two tiles follow fk::Form<S>'s bytes there.
constexpr int VALUE_WARPS = 16;
template <class S> inline size_t value_fold_bytes(const S &src) {
  return rounded16(fk::Form<S>::shared(src)) + (size_t)VALUE_WARPS * 2 * FOLD_TILE * sizeof(int);
}
// k_fold_range() for a function table, as fk::k_fold_poly<LV, S>. Row r of tile x < 2 is output
// 2 r + x of the warp's 32. Step w is word w & 1 of coefficient (w >> 1) + 1 of the slabs' nodes.
// The idea is public: a vectorized load of two elements: M. Harris, "Mixed-Precision Programming
// with CUDA 8", NVIDIA Technical Blog, 2016, "FP16 types and intrinsics",
// https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/. Here the elements are
// integers, two neighbouring 16-bit keys.
template <int LV, class S>
__global__ void __launch_bounds__(32 * VALUE_WARPS, 1)
    k_fold_values(const uint16_t *__restrict__ keys, int dT, const T *alpha, const S src, const T *__restrict__ k8,
                  T *__restrict__ Lo, T *__restrict__ Ro) {
  constexpr int D = 1 << LV, WORDS = 2 * D;
  static_assert(fold_sum_fits(WORDS * 8 * 4), "an output's sums run over the words of eight slabs' nodes, four bytes each");
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ T constant[8][D + 1]; // [a][i]: (-1)^i alpha^(D - i) k8[a]
  if (threadIdx.x < 8) {
    T u = k8[threadIdx.x];
    const T alpha_here = *alpha;
    for (int i = D; i >= 0; i--) {
      constant[threadIdx.x][i] = (i & 1) ? F::sub(F::zero(), u) : u;
      u = F::mul(u, alpha_here);
    }
  }
  fk::Form<S> f;
  f.init_values(src, staged);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  int *mine = reinterpret_cast<int *>(staged + (fk::Form<S>::shared(src) + 15) / 16 * 16) + warp * 2 * FOLD_TILE;
  u32 b[WORDS][3][2];
#pragma unroll
  for (int w = 0; w < WORDS; w++)
#pragma unroll
    for (int j = 0; j < 3; j++)
#pragma unroll
      for (int h = 0; h < 2; h++) b[w][j][h] = constant_word(constant[t4 + 4 * h][(w >> 1) + 1], j, 32 * (w & 1), g);
  T start = F::zero();
  for (int a = 0; a < 8; a++) start = F::add(start, constant[a][0]);
  const int d = dT - LV;
  const size_t SC = (size_t)1 << (d - 4), batches = 2 * SC / 32;
  for (size_t batch = blockIdx.x * (size_t)VALUE_WARPS + warp; batch < batches; batch += (size_t)gridDim.x * VALUE_WARPS) {
    const bool second = batch >= SC / 32;
    const size_t c0 = (second ? batch - SC / 32 : batch) * 32;
    u32 q[2][2][D]; // [slab t4 + 4 s][rows g + 8 r][leaf]: two neighbouring keys
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int j = 0; j < D; j++)
          q[s][r][j] = __ldg(reinterpret_cast<const u32 *>(keys + ((size_t)((second ? 8 : 0) + t4 + 4 * s) << (d - 4)) + c0 +
                                                           2 * (g + 8 * r) + ((size_t)j << d)));
#pragma unroll
    for (int x = 0; x < 2; x++) {
      u64 cf[2][2][D];
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          u64 t[D];
#pragma unroll
          for (int j = 0; j < D; j++) t[j] = f.value(x ? q[s][r][j] >> 16 : q[s][r][j] & 0xFFFFu);
          fk::Coefficients<D, false>::of(t, cf[s][r]);
        }
      int sum[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
      for (int w = 0; w < WORDS; w++) {
        const int i = w >> 1, sh = 32 * (w & 1);
        const u32 a[4] = {(u32)(cf[0][0][i] >> sh), (u32)(cf[0][1][i] >> sh), (u32)(cf[1][0][i] >> sh), (u32)(cf[1][1][i] >> sh)};
#pragma unroll
        for (int j = 0; j < 3; j++) multiply_unsigned(sum[j], a, b[w][j]);
      }
      put_sums(mine + x * FOLD_TILE, sum, g, t4);
    }
    __syncwarp();
    u64 *out = reinterpret_cast<u64 *>((second ? Ro : Lo) + c0);
#pragma unroll
    for (int u = 0; u < 3; u++) {
      const int id = lane + 32 * u, c = id / 3, j = id % 3; // output c0 + c is row c >> 1 of tile c & 1
      out[id] = from_sums(mine + (c & 1) * FOLD_TILE + (c >> 1) * 24 + 8 * j, coord(start, j));
    }
    __syncwarp();
  }
}
// k_fold64_range() with tau(key) for key, as grm::k_fold64() and fk::k_fold64_rows(). A value is
// two 32-bit words, so an output takes 16 steps: step s is word s & 1 of the value of key s >> 1 of
// each 128-bit word of keys. The constants, too many for registers, are in shared memory, as
// constants[step][coordinate][lane].
template <class S>
__global__ void __launch_bounds__(32 * VALUE_WARPS, 1)
    k_fold64_values(const uint4 *__restrict__ key, size_t rows, size_t CO, const S src, const T *__restrict__ K,
                    const T *__restrict__ alpha, T *__restrict__ UL, T *__restrict__ UR) {
  static_assert(fold_sum_fits(64 * 8), "an output's sums run over 64 values of eight bytes");
  extern __shared__ __align__(16) unsigned char staged[];
  __shared__ uint2 constants[16][3][32];
  fk::Form<S> f;
  f.init_values(src, staged);
  for (int x = threadIdx.x; x < 16 * 3 * 32; x += blockDim.x) {
    const int s = x / 96, j = x / 32 % 3, l = x & 31;
    const T k0 = F::sub(F::zero(), K[8 * (l & 3) + (s >> 1)]), k1 = F::sub(F::zero(), K[8 * ((l & 3) + 4) + (s >> 1)]);
    constants[s][j][l] = make_uint2(constant_word(k0, j, 32 * (s & 1), l >> 2), constant_word(k1, j, 32 * (s & 1), l >> 2));
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  int *mine = reinterpret_cast<int *>(staged + (fk::Form<S>::shared(src) + 15) / 16 * 16) + warp * 2 * FOLD_TILE;
  const T start = *alpha;
  const size_t batches = 2 * rows / 32;
  for (size_t batch = blockIdx.x * (size_t)VALUE_WARPS + warp; batch < batches; batch += (size_t)gridDim.x * VALUE_WARPS) {
    const uint4 *from = key + batch * 32 * 8;
#pragma unroll
    for (int x = 0; x < 2; x++) {
      uint4 q[2][2]; // [word t4 + 4 h][row g + 8 r]
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int r = 0; r < 2; r++) q[h][r] = __ldg(from + (size_t)(16 * x + g + 8 * r) * 8 + t4 + 4 * h);
      int sum[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const u64 v[4] = {f.value(grm::key_of(q[0][0], e)), f.value(grm::key_of(q[0][1], e)), f.value(grm::key_of(q[1][0], e)),
                          f.value(grm::key_of(q[1][1], e))};
#pragma unroll
        for (int half = 0; half < 2; half++) {
          const u32 a[4] = {(u32)(v[0] >> (32 * half)), (u32)(v[1] >> (32 * half)), (u32)(v[2] >> (32 * half)), (u32)(v[3] >> (32 * half))};
#pragma unroll
          for (int j = 0; j < 3; j++) {
            const uint2 c = constants[2 * e + half][j][lane];
            const u32 bb[2] = {c.x, c.y};
            multiply_unsigned(sum[j], a, bb);
          }
        }
      }
      put_sums(mine + x * FOLD_TILE, sum, g, t4);
    }
    __syncwarp();
    const bool second = batch >= rows / 32;
    const size_t r0 = (second ? batch - rows / 32 : batch) * 32; // outputs r0 + 8 c + a, c < 4, a < 8
    u64 *out = reinterpret_cast<u64 *>(second ? UR : UL);
#pragma unroll
    for (int u = 0; u < 3; u++) {
      const int id = lane + 32 * u, e = id / 3, j = id % 3, a = e >> 2, c = e & 3, r = 8 * c + a;
      out[((size_t)a * CO + (r0 >> 3) + c) * 3 + j] = from_sums(mine + (r >> 4) * FOLD_TILE + (r & 15) * 24 + 8 * j, coord(start, j));
    }
    __syncwarp();
  }
}
// The first call for a kernel that needs over 48 KB of shared memory raises its limit to `bytes`.
// That call must be outside a stream capture (forest_prove_graph() in forest.cuh runs a proof
// directly before it captures one).
template <class Kernel> inline void wants_shared(Kernel kernel, size_t bytes) {
  static std::mutex m;
  static std::set<const void *> asked;
  std::lock_guard<std::mutex> lock(m);
  if (bytes > 49152 && asked.insert((const void *)kernel).second)
    CK_ATTRIBUTE(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}
// fk::fold() on the tensor cores, LV 1 or 2, any table. Needs dT >= LV + 10 and t.keys 8-byte
// aligned; neither is checked. For a function table, another LV exits with code 1.
inline void fold(int LV, const fk::Tree &t, int dT, const T *alpha, const T *k8, T *Lo, T *Ro, cudaStream_t s) {
  if (t.tab.kind == fk::RANGE) return fold_range(LV, t, dT, alpha, k8, Lo, Ro, s);
  fk::with_src(t.tab, [&](auto src) {
    typedef decltype(src) S;
    if constexpr (!std::is_same<S, fk::RangeSrc>::value)
      fk::with_constant<1, 2>(LV, [&](auto lv) {
        auto kernel = k_fold_values<decltype(lv)::value, S>;
        const size_t bytes = value_fold_bytes(src), batches = (size_t)1 << (dT - LV - 8);
        wants_shared(kernel, bytes);
        kernel<<<(int)std::min<size_t>((batches + VALUE_WARPS - 1) / VALUE_WARPS, g_sms), 32 * VALUE_WARPS, bytes, s>>>(t.keys, dT, alpha, src, k8, Lo, Ro);
        CK(cudaGetLastError());
      });
  });
}
// fk::third_fold() on the tensor cores, any table. Not checked: dT >= 13, t.keys 16-byte aligned,
// K64 adding up to 1.
inline void third_fold(const fk::Tree &t, int dT, const T *alpha, const T *K64, T *UL, T *UR, cudaStream_t s) {
  if (t.tab.kind == fk::RANGE) return third_fold_range(t, dT, alpha, K64, UL, UR, s);
  const size_t N = (size_t)1 << dT;
  fk::with_src(t.tab, [&](auto src) {
    typedef decltype(src) S;
    if constexpr (!std::is_same<S, fk::RangeSrc>::value) {
      auto kernel = k_fold64_values<S>;
      const size_t bytes = value_fold_bytes(src), batches = N / 2048;
      wants_shared(kernel, bytes);
      kernel<<<(int)std::min<size_t>((batches + VALUE_WARPS - 1) / VALUE_WARPS, g_sms), 32 * VALUE_WARPS, bytes, s>>>(
          (const uint4 *)t.keys, N / 128, N / 1024, src, K64, alpha, UL, UR);
      CK(cudaGetLastError());
    }
  });
}
} // namespace ff
