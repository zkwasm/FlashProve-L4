// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Computes T_a for a weight claim (a claim on the committed weight matrix W) in CClaim's mode 2.
// T_a has one entry per column of W: the sum over a unit's rows of coefficient times cell.
// A unit is a tensor's rows in W; a cell is a signed byte. Mode 0 (kG_Tcells()) finds each cell
// through offset tables. Mode 2 is for two-limb units (a weight is a low and a high byte) whose
// cells are in segments of adjacent cells: a thread loads four cells at once. Included by
// prover/head/claims_lib.cuh, which defines index groups (CGrp) and their classes. Nothing here
// checks that CMap::setup_run() accepted the unit, or that `cells` is aligned to 8 bytes (16 if
// rem >= 4); CEnd::tables() tests only the alignment before it calls kG_T_segments().
//
// Indexing. A cell's index in the class-'L' group is (kh << takeK) | kl. Its offset in the unit's
// other group or groups is (hi << z) | (c << rem) | low; setup_run() accepts only rem = 0 and
// 2 to 8. The pair (kl, c) selects the entry of T_a to which the cell contributes. An entry's
// cells are numbered e = (seg << rem) | low, where seg = (limb * leftK + kh) * o + hi numbers its
// segments of 2^rem adjacent cells (kh < leftK, hi < o). The entry is the sum over e of
//   cell e * gam(e), where gam(e) = 256^limb * gk[kh] * gw[hi * 2^rem + low].
// Each coordinate of gam(e) is cut into two signed 32-bit halves (iprep()'s limbs). A thread adds
// up half times cell in 64-bit integers and reduces each sum once, at the end (kG_joined()).
// prover/field/lazy_signed.cuh allows terms * largest |cell| <= 2^IMAC_ROOM_BITS = 2^29;
// setup_run() accepts at most 2^16 cells per entry, and |cell| <= 2^7: a product of at most 2^23.
// Public source: products accumulated, reduced only before an overflow: J-G. Dumas, P. Giorgi,
// C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK packages"
// (arXiv:cs/0601133), section 2.1.
// Public source: shared memory, used in kG_Tsegs(): NVIDIA, CUDA C++ Programming Guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/): "Shared Memory".
// See also: NVIDIA, CUDA Programming Guide, "Writing SIMT Kernels", "Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
#pragma once
#include "field/field_lazy.cuh"
// T, the prover's element type, and blk::E (blocks/field/field.h) have the same three coordinates.
__host__ __device__ __forceinline__ blk::E kG_e(const T &v) {
  return blk::E{{blk::F{coord(v, 0)}, blk::F{coord(v, 1)}, blk::F{coord(v, 2)}}};
}
__host__ __device__ __forceinline__ T kG_t(const blk::E &e) {
  T v;
  for (int t = 0; t < EDEG; t++) set_coord(v, t, e.c[t].v);
  return v;
}
// Fills kG_Tsegs()'s tables. parts[((e / 4) * 2 * EDEG + 2 * t + h) * 4 + e % 4] is half h
// (0 low, 1 high) of coordinate t of gam(e), so that one int4 of `parts` has the same half for
// four consecutive e. segOff[seg] is the offset, in cells, of segment seg < nseg from the
// entry's first cell. GRID_LOOP (prover/prover/forward.cuh) is a grid-stride loop.
__global__ void kG_segtab(int nseg, size_t limbStride, size_t strideK, int takeK, int leftK, int o, int z, int rem,
                          const T *gk, const T *gw, u32 *segOff, int *parts) {
  GRID_LOOP(e, (size_t)nseg << rem) {
    const u32 seg = (u32)(e >> rem);
    blk::E g = blk::lazy::mul(kG_e(gk[(seg / o) % leftK]), kG_e(gw[((seg % o) << rem) + (u32)(e & ((1u << rem) - 1))]));
    if (seg / o >= (u32)leftK) g = blk::lazy::shifted<8>(g); // the second limb's bytes weigh 256
    const blk::lazy::IPrep limbs = blk::lazy::iprep(g);
    for (int t = 0; t < EDEG; t++) {
      parts[((e >> 2) * (2 * EDEG) + 2 * t) * 4 + (e & 3)] = limbs.l[t];
      parts[((e >> 2) * (2 * EDEG) + 2 * t + 1) * 4 + (e & 3)] = limbs.h[t];
    }
  }
  GRID_LOOP(id, nseg) {
    const u32 seg = (u32)id, kh = (seg / o) % leftK;
    segOff[seg] = (u32)(seg / o / leftK * limbStride + ((size_t)kh << takeK) * strideK + ((seg % o) << z));
  }
}
// Returns the element whose coordinate t is acc[2 * t] + 2^ILIMB_BITS * acc[2 * t + 1].
__device__ __forceinline__ blk::E kG_joined(const long long (&acc)[2 * EDEG]) {
  blk::lazy::IAcc a;
  for (int t = 0; t < EDEG; t++) a.l[t] = acc[2 * t], a.h[t] = acc[2 * t + 1];
  return blk::lazy::ired(a);
}
// Writes partial sums: T_a[index] is the sum over s < gridDim.y of part[s * K + index]. Entry
// r = (kl << zc) | c, for r < nrows, has the index colTab[c] | (kl << cbK), where c has
// zc = z - rem bits and cbK is the 'L' group's first column bit. colTab[c] is ~0 if c has no
// column: entry r is then not written. ONE must be (rem == 0). From each segment of its partial
// sum, a thread loads four adjacent cells: of four entries with ONE, else of one. If rem > 0, an
// entry's 2^(rem - 2) threads, all in one block, store their sums in `sh`, and the first of them
// adds these. Launch (ceil(nrows * 2^rem / 1024), number of partial sums) blocks of 256 threads.
// Public sources for the techniques of this routine.
// The idea is public: several outputs per thread (thread coarsening), with ONE: Wen-mei W. Hwu,
// David Kirk, "Advanced Algorithmic Techniques for GPUs, Lecture 4: Thread Coarsening and more on
// Tiling/Blocking", PASI Summer School, Chile, January 5-7, 2011, slides "Thread Coarsening" and
// "Outline of Technique", https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf.
// The idea is public: aligned vector loads through a cast pointer (int4, without ONE): "CUDA Pro
// Tip: Increase Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/;
// for bytes, four cells in one int load (there: a histogram's input, to suit the G80 GPU):
// V. Podlozhnyuk, "Histogram calculation in CUDA", NVIDIA, November 2007,
// "Implementation of histogram64",
// https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/histogram64/doc/histogram.pdf.
// The compiler hints are public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; #pragma unroll, without
// and with a count: NVIDIA, "CUDA Programming Guide", page "C/C++ Language Extensions",
// "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// unrolling fixes each index into acc, a thread's few sums, so registers can hold them: M. Milakov,
// "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <bool ONE>
__global__ void __launch_bounds__(256) kG_Tsegs(const i8 *cells, size_t strideK, int cbK, int zc, int rem, size_t nrows,
                                                int nseg, const u32 *__restrict__ segOff, const int *__restrict__ parts,
                                                const u32 *__restrict__ colTab, size_t K, T *part) {
  __shared__ u64 sh[256][EDEG];
  const size_t p = (blockIdx.x * (size_t)256 + threadIdx.x) * 4;
  const bool live = p < (nrows << rem);
  const int per = (nseg + gridDim.y - 1) / gridDim.y, s0 = blockIdx.y * per, s1 = min(nseg, s0 + per);
  const size_t r = live ? p >> rem : 0, zmask = ((size_t)1 << zc) - 1;
  const u32 low = live ? (u32)(p & ((1u << rem) - 1)) : 0;
  const i8 *at = cells + (r >> zc) * strideK + ((r & zmask) << rem) + low;
  T *mine = part + (size_t)blockIdx.y * K;
  if constexpr (ONE) {
    long long acc[4][2 * EDEG];
    for (int i = 0; i < 4; i++)
      for (int v = 0; v < 2 * EDEG; v++) acc[i][v] = 0;
    if (live)
#pragma unroll 4
      for (int seg = s0; seg < s1; seg++) {
        const int x4 = *(const int *)(at + segOff[seg]);
        const int *w = parts + (size_t)(seg >> 2) * (2 * EDEG) * 4 + (seg & 3);
#pragma unroll
        for (int v = 0; v < 2 * EDEG; v++) {
          const int c = w[4 * v];
#pragma unroll
          for (int i = 0; i < 4; i++) acc[i][v] += (long long)(int)(i8)(x4 >> (8 * i)) * c;
        }
      }
    if (live)
      for (int i = 0; i < 4; i++) {
        const T v = kG_t(kG_joined(acc[i]));
        const u32 ct = colTab[(r + i) & zmask];
        if (ct != ~0u) mine[(size_t)ct | (((r + i) >> zc) << cbK)] = v;
      }
  } else {
    long long acc[2 * EDEG];
    for (int v = 0; v < 2 * EDEG; v++) acc[v] = 0;
    if (live)
#pragma unroll 4
      for (int seg = s0; seg < s1; seg++) {
        const int x4 = *(const int *)(at + segOff[seg]);
        const int x0 = (int)(i8)x4, x1 = (int)(i8)(x4 >> 8), x2 = (int)(i8)(x4 >> 16), x3 = x4 >> 24;
        const int4 *w = (const int4 *)parts + ((((size_t)seg << rem) + low) >> 2) * (2 * EDEG);
#pragma unroll
        for (int v = 0; v < 2 * EDEG; v++) {
          const int4 c = w[v];
          acc[v] += (long long)x0 * c.x + (long long)x1 * c.y + (long long)x2 * c.z + (long long)x3 * c.w;
        }
      }
    const blk::E own = kG_joined(acc);
    for (int t = 0; t < EDEG; t++) sh[threadIdx.x][t] = own.c[t].v;
    __syncthreads();
    if (live && low == 0) {
      T v;
      for (int t = 0; t < EDEG; t++) {
        blk::F s = blk::F::zero();
        for (int i = 0; i < (1 << (rem - 2)); i++) s += blk::F{sh[threadIdx.x + i][t]};
        set_coord(v, t, s.v);
      }
      const u32 ct = colTab[r & zmask];
      if (ct != ~0u) mine[(size_t)ct | ((r >> zc) << cbK)] = v;
    }
  }
}
// out[index] = the sum over s < ns of part[s * K + index], for the index in T_a of each entry
// r < nrows that has one.
__global__ void kG_Tsegsum(const T *part, int ns, size_t K, size_t nrows, int zc, int cbK, const u32 *colTab, T *out) {
  GRID_LOOP(r, nrows) {
    const u32 ct = colTab[r & (((size_t)1 << zc) - 1)];
    if (ct == ~0u) continue;
    const size_t at = (size_t)ct | ((r >> zc) << cbK);
    T a = part[at];
    for (int s = 1; s < ns; s++) a = F::add(a, part[(size_t)s * K + at]);
    out[at] = a;
  }
}
// Computes T_a on stream st for a claim on unit u. m is u's map and k = u.g[m.ikk] its 'L' group.
// nrows must be 2^(k.take + m.z - m.rem), the number of pairs (kl, c); this is not checked.
// Writes at most nrows of the K entries at `out`, which the caller must have zeroed. gk and gw hold
// gam(e)'s factors. CEnd's scratch buffers, whose sizes are not checked: segOff (one word per
// segment), gb (16-byte aligned; 8 * EDEG words for every four cells of an entry, rounded up),
// `part` (ns * K elements).
static inline void kG_T_segments(const i8 *cells, const CUnit &u, const CGrp &k, const CMap &m, size_t nrows, int ns,
                                 const T *gk, const T *gw, u32 *segOff, u32 *gb, T *part, size_t K, T *out,
                                 cudaStream_t st) {
  const int nseg = 2 * (int)k.left * m.o, slices = nseg >= 64 ? ns : 1;
  kG_segtab<<<grid_blocks((size_t)nseg << m.rem), 256, 0, st>>>(nseg, u.limbStride, k.stride, k.take, (int)k.left, m.o, m.z,
                                                         m.rem, gk, gw, segOff, (int *)gb);
  CK(cudaGetLastError());
  const dim3 grid((unsigned)(((nrows << m.rem) + 1023) / 1024), (unsigned)slices);
  T *to = slices > 1 ? part : out;
  if (m.rem == 0)
    kG_Tsegs<true><<<grid, 256, 0, st>>>(cells, k.stride, k.cb0, m.z - m.rem, m.rem, nrows, nseg, segOff,
                                         (const int *)gb, m.colTab, K, to);
  else
    kG_Tsegs<false><<<grid, 256, 0, st>>>(cells, k.stride, k.cb0, m.z - m.rem, m.rem, nrows, nseg, segOff,
                                          (const int *)gb, m.colTab, K, to);
  CK(cudaGetLastError());
  if (slices > 1) {
    kG_Tsegsum<<<grid_blocks(nrows), 256, 0, st>>>(part, slices, K, nrows, m.z - m.rem, k.cb0, m.colTab, out);
    CK(cudaGetLastError());
  }
}
