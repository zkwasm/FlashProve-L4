// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Groups the T tokens of a mixture-of-experts (MoE) layer by routed expert (top-1 routing: one of
// the NE experts per token), so that each expert's matrix products run over its own tokens, not
// over all T (prover/weights_blocks/experts.cuh). Source for this grouping:
// Gale, Narayanan, Young, Zaharia, "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts",
// arXiv:2211.15841, Figure 1, step 2. The cells (committed small integers), lookup keys and range
// flags (ForwardBuffers::err) are meant to be as if every expert ran on all tokens and each token kept its
// expert's result. This equivalence has not been traced through the tables below, nor through the
// kernels that form the matrix products. k_gate_up() (prover/weights_blocks/experts.cuh) sets the
// routed experts' range flags only for the tokens that the tables list. ForwardPass::ffn_moe()
// (prover/prover/forward.cuh) takes this path only if wblocks is set, T % 64 == 0, fe % 128 == 0,
// d % 128 == 0 and NE <= 1024 (fe: features per expert; d: the model's width).
#pragma once
#include "common/index_guard.cuh"
#include "common/lie.cuh"
#include "prover/forward_ffn.cuh"
// The grouping's index tables. bind() places them in one GPU buffer of ints(T, NE) ints, and
// k_moe_prep() writes them. A token's rank is its index in the order sorted by expert; its slot is
// its index once each expert's run is padded to a multiple of 64. cnt, sbase, rbase: each expert's
// token count, first slot and first rank. rank_of[t]: token t's rank. tok_of[s]: slot s's token, or
// -1. Every kernel of the prover that reads a table (test programs' kernels not counted) checks an
// entry where it uses it as an index or as a count (prover/common/index_guard.cuh).
// ExpertsWork::fetch() (prover/weights_blocks/experts.cuh) checks the host's copy of cnt, sbase and
// rbase. A failed check stops nothing: it sets the failure bit INDEX, and a request with a failure
// bit set gets no proof.
struct MoeTables {
  int T, NE, S; // S = T + 64 * NE is an upper bound on the number of slots
  int *cnt, *sbase, *rbase, *rank_of, *tok_of, *te, *tc,
      *tr; // te, tc, tr, per group of 32 slots: its expert (-1: none), its number of tokens, its first slot's rank
  static size_t ints(int T, int NE) { return (size_t)3 * NE + T + (T + 64 * NE) + 3 * ((T + 64 * NE) / 32) + 64; }
  void bind(int *p, int T_, int NE_) {
    T = T_;
    NE = NE_;
    S = T + 64 * NE;
    cnt = p;
    sbase = cnt + NE;
    rbase = sbase + NE;
    rank_of = rbase + NE;
    tok_of = rank_of + T;
    te = tok_of + S;
    tc = te + S / 32;
    tr = tc + S / 32;
  }
  // Returns te[q], q unchecked. A value outside [-1, NE) gives -1 and sets the failure bit INDEX.
  // The idea is public: reading a table through __ldg(): NVIDIA, Kepler Tuning Guide,
  // "Memory Throughput", part "Read-Only Data Cache",
  // https://docs.nvidia.com/cuda/kepler-tuning-guide/.
  __device__ __forceinline__ int expert_of(int q) const {
    return (int)index_below((long long)__ldg(te + q) + 1, NE + 1, INDEX_MOE_TABLE) - 1;
  }
};
namespace ffn_moe_kernels {
// Sorts the tokens by expert (es[t] is token t's) and writes all of m's tables. The order within a
// run is not deterministic. An es[t] outside [0, NE) counts as expert 0 and sets the bit INDEX.
// Launch: one block of 1024 threads.
// The idea is public: padded runs and a table of each block's expert: vLLM's moe_align_block_size()
// "Aligns the token distribution across experts to be compatible with block size for matrix
// multiplication" and returns sorted_token_ids, "the sorted token indices according to their
// allocated expert", and expert_ids, "the assigned expert index for each block":
// vLLM's API documentation,
// https://docs.vllm.ai/en/v0.10.1.1/api/vllm/model_executor/layers/fused_moe/moe_align_block_size.html.
// Here each token has one expert, a run is padded to a multiple of 64 slots, and a table entry
// stands for 32 slots. An empty slot or entry holds -1; in vLLM a padding position holds a token
// number that does not exist, and -1 marks an expert of another GPU.
__global__ void __launch_bounds__(1024) k_moe_prep(MoeTables m, const int *es) {
  __shared__ int c[1024], sb[1024], rb[1024];
  const int u = threadIdx.x;
  for (int e = u; e < m.NE; e += 1024) c[e] = 0;
  __syncthreads();
  for (int t = u; t < m.T; t += 1024) atomicAdd(&c[index_below(es[t], m.NE, INDEX_EXPERT)], 1);
  __syncthreads();
  if (u == 0) {
    int s = 0, r = 0;
    for (int e = 0; e < m.NE; e++) {
      sb[e] = s;
      rb[e] = r;
      s += (c[e] + 63) / 64 * 64;
      r += c[e];
    }
  }
  __syncthreads();
  for (int e = u; e < m.NE; e += 1024) {
    m.cnt[e] = c[e];
    m.sbase[e] = sb[e];
    m.rbase[e] = rb[e];
  }
  for (int i = u; i < m.S; i += 1024) m.tok_of[i] = -1;
  for (int q = u; q < m.S / 32; q += 1024) {
    m.te[q] = -1;
    m.tc[q] = 0;
    m.tr[q] = 0;
  }
  __syncthreads();
  for (int e = u; e < m.NE; e += 1024)
    for (int i = 0; i * 32 < c[e]; i++) {
      const int q = sb[e] / 32 + i;
      m.te[q] = e;
      m.tc[q] = min(32, c[e] - i * 32);
      m.tr[q] = rb[e] + i * 32;
    }
  __syncthreads();
  for (int e = u; e < m.NE; e += 1024) c[e] = 0;
  __syncthreads();
  for (int t = u; t < m.T; t += 1024) {
    const int e = index_below(es[t], m.NE, INDEX_EXPERT), i = atomicAdd(&c[e], 1);
    m.rank_of[t] = rb[e] + i;
    // es is read twice, once in each loop. If an entry changed in between, this expert's
    // second count could exceed its first, and sb[e] + i could exceed the number of slots.
    m.tok_of[index_below(sb[e] + i, m.S, INDEX_MOE_TABLE)] = t;
  }
}
// Copies the first 16 * nb16 bytes of each token's row of src to its slot's row of dst; an empty
// slot gets zeros, so dst needs m.S rows. lds and ldd are the distances between rows, in bytes.
// The idea is public: vectorized loads and stores of 128 bits: "CUDA Pro Tip: Increase Performance
// with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_moe_rows(MoeTables m, const i8_ *src, size_t lds, i8_ *dst, size_t ldd, int nb16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)m.S * nb16;
       i += (size_t)gridDim.x * blockDim.x) {
    const int sl = (int)(i / nb16), c = (int)(i % nb16), t = m.tok_of[sl];
    *(uint4 *)(dst + (size_t)sl * ldd + c * 16) =
        t >= 0 ? *(const uint4 *)(src + (size_t)index_below(t, m.T, INDEX_MOE_TABLE) * lds + c * 16)
               : make_uint4(0, 0, 0, 0);
  }
}
// Writes in token order the cells and lookup keys of the routed experts' features: what k_epi_gate_up
// (prover/prover/forward.cuh) writes only if `proof` is set. tq, t1, t2, th: arrays [jl < n][rank]
// from gate_up() (prover/weights_blocks/experts.cuh); th is read only if gc.on is set. The cell
// pointers, KHs and KHu point to the kernel's first feature (fe in ffn_moe()), so jl indexes them;
// Pos2 maps the feature's number j (first feature + jl) to a key's column.
// lie::at() (prover/common/lie.cuh) is a fault-injection hook for tests; in this repository it is
// the identity: it returns its fourth argument.
// The loop form is public: the grid-stride loop (cited at k_moe_rows()).
// A thread takes W adjacent tokens of a feature; LAUNCH_BY_FOUR (prover/prover/forward_by_four.cuh)
// chooses W. The ranks need not be adjacent, so what is indexed by rank is loaded token by token.
template <int W = 1>
__global__ void k_moe_gu_out(MoeTables m, const u32 *tq, const u32 *t1, const u32 *t2, int n, int j0, int kg, int ku,
                             int hg, int hu, const short *Ts, i8_ *gi, i8_ *sc, i8_ *uc, i8_ *rg, i8_ *ru, size_t st2,
                             u16_ *kG, u16_ *kU, u16_ *kGH, u16_ *kUH, u16_ *kUT, u16_ *kF, Pos2 pG, Pos2 pU, Pos2 pGH,
                             Pos2 pUH, Pos2 pUT, Pos2 pF, u16_ *KHs, u16_ *KHu, GateClamp gc, const u16_ *th) {
  const int T = m.T;
  const Dig Dg{kg, hg}, Du{ku, hu};
  for (size_t z = blockIdx.x * (size_t)blockDim.x + threadIdx.x; z < (size_t)T * n / W;
       z += (size_t)gridDim.x * blockDim.x) {
    const size_t i = W * z;
    const int t = (int)(i % T), jl = (int)(i / T), j = j0 + jl;
    i64 qg[W], qu[W], s[W], gh[W];
    i8_ g0[W], g1[W], u0[W], u1[W];
    u16_ lo_g[W], lo_u[W], hi_g[W], hi_u[W], of_u[W], of_g[W], of_s[W];
#pragma unroll
    for (int e = 0; e < W; e++) {
      const size_t o = (size_t)jl * T + index_below(m.rank_of[t + e], T, INDEX_MOE_TABLE);
      const u32 Q = tq[o], r1 = t1[o], r2 = t2[o];
      qg[e] = (short)(Q & 65535u), qu[e] = (short)(Q >> 16), s[e] = lie::at(lie::SILU, j, t + e, (i64)Ts[qg[e] + 32768]);
      g0[e] = (i8_)Dg.c0(r1), g1[e] = (i8_)Dg.c1(r1), u0[e] = (i8_)Du.c0(r2), u1[e] = (i8_)Du.c1(r2);
      lo_g[e] = (u16_)Dg.lo(r1), lo_u[e] = (u16_)Du.lo(r2), hi_g[e] = (u16_)Dg.hi(r1), hi_u[e] = (u16_)Du.hi(r2);
      of_u[e] = (u16_)(qu[e] + 32768), of_g[e] = (u16_)(qg[e] + 32768), of_s[e] = (u16_)(s[e] + 32768);
      gh[e] = gc.on ? (short)th[o] : 0;
    }
    write_limbs_wide<W>(qg, 2, gi + i, st2);
    write_limbs_wide<W>(s, 2, sc + i, st2);
    write_limbs_wide<W>(qu, 2, uc + i, st2);
    if (Dg.one() || Dg.two()) st_w<W>(rg + i, g0);
    if (Dg.two()) st_w<W>(rg + i + st2, g1);
    if (Du.one() || Du.two()) st_w<W>(ru + i, u0);
    if (Du.two()) st_w<W>(ru + i + st2, u1);
    st_w<W>(kG + (size_t)pG.at(j) * T + t, lo_g);
    st_w<W>(kU + (size_t)pU.at(j) * T + t, lo_u);
    if (Dg.hasHi()) st_w<W>(kGH + (size_t)pGH.at(j) * T + t, hi_g);
    if (Du.hasHi()) st_w<W>(kUH + (size_t)pUH.at(j) * T + t, hi_u);
    st_w<W>(kUT + (size_t)pUT.at(j) * T + t, of_u);
    st_w<W>(kF + (size_t)pF.at(j) * T + t, of_g);
    st_w<W>(KHs + i, of_s);
    st_w<W>(KHu + i, of_u);
    if (!gc.on) continue;
    write_limbs_wide<W>(gh, 2, gc.gh + i, gc.st);
#pragma unroll
    for (int e = 0; e < W; e++) of_u[e] = (u16_)(gh[e] + 49152), of_g[e] = (u16_)(((int)qg[e] & 255) ^ 128);
    st_w<W>(gc.kCG + (size_t)gc.pCG.at(j) * T + t, of_u);
    st_w<W>(gc.kG0 + (size_t)gc.pG0.at(j) * T + t, of_g);
  }
}
} // namespace ffn_moe_kernels
// Launches k_moe_prep() on stream st; call m.bind() first. Exits with code 1 if NE > 1024, the size
// of the kernel's shared arrays, or if T is not a multiple of 64.
static inline void launch_moe_prep(MoeTables &m, const int *es, cudaStream_t st) {
  if (m.NE > 1024 || m.T % 64) {
    printf("launch_moe_prep: NE <= 1,024, T %% 64 == 0\n");
    exit(1);
  }
  ffn_moe_kernels::k_moe_prep<<<1, 1024, 0, st>>>(m, es);
}
// Launches k_moe_rows() on stream st. `bytes` is the number of bytes copied per row. It must be a
// multiple of 16, and the rows of src and dst must be 16-byte aligned; neither is checked.
static inline void launch_moe_rows(const MoeTables &m, const int8_t *src, size_t lds, int8_t *dst, size_t ldd, int bytes,
                                cudaStream_t st) {
  ffn_moe_kernels::k_moe_rows<<<2048, 256, 0, st>>>(m, src, lds, dst, ldd, bytes / 16);
}
// Launches k_moe_gu_out() on stream st, with W = 4 or 1.
static inline void launch_moe_gu_out(const MoeTables &m, const u32 *tq, const u32 *t1, const u32 *t2, int n, int j0, int kg,
                                  int ku, int hg, int hu, const short *Ts, int8_t *gi, int8_t *sc, int8_t *uc,
                                  int8_t *rg, int8_t *ru, size_t st2, uint16_t *kG, uint16_t *kU, uint16_t *kGH,
                                  uint16_t *kUH, uint16_t *kUT, uint16_t *kF, Pos2 pG, Pos2 pU, Pos2 pGH, Pos2 pUH,
                                  Pos2 pUT, Pos2 pF, uint16_t *KHs, uint16_t *KHu, cudaStream_t st,
                                  GateClamp gc = GateClamp(), const uint16_t *th = nullptr) {
  LAUNCH_BY_FOUR(ffn_moe_kernels::k_moe_gu_out, m.T, (size_t)m.T * n, st, //
                 m, tq, t1, t2, n, j0, kg, ku, hg, hu, Ts, gi, sc, uc, rg, ru, st2, kG, kU, kGH, kUH, kUT, kF, pG, pU,
                 pGH, pUH, pUT, pF, KHs, KHu, gc, th);
}
