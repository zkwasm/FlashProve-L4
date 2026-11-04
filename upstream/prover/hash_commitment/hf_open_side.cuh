// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The callbacks (HfSeams; "Open_F's seams" in messages) by which hf::HfKeeper
// (prover/hash_commitment/hf_keeper.h) opens the weight table T: they fold its rows into one row u
// and compute its columns at drawn positions. Contracts: prover/hash_commitment/hf_open_api.h. They
// get Merkle nodes of T (HfHeld::nodes), not T, so make_u takes one pass over the weights W ("walk"
// in identifiers and messages) and make_columns one or more. A caller fills an rf::HfSide and calls
// hf_side_seams(): rf::Unit (prover/hash_commitment/hf_keeper.cu) in the prover program cp_prove,
// hf::HfOnceTable (prover/hash_commitment/hf_keeper_once.cuh) in the committer program cp_commit.
// The two differ in who brings W (`walk`) and in `fibres_a_walk` (0 in rf::Unit: two passes).
// hf_side_record() is no callback: cp_commit calls it to compute a table's record, in one pass.
// Layout. W has m0 rows and k = 2^lk columns; a unit (rf::UnitW) is a range of its rows. M is W
// as R = 2^b long rows; T[r], of 2^ln positions, is the codeword of M's row r. lcl = lk - b.
// A row of M has a message of i0 entries and nm masks, its random tail.
// Entry y 2^lcl + c' of row r's message is W[y][r 2^lcl + c'], c' < 2^lcl; entries from
// l = m0 2^lcl on are 0, up to the pads. A table has a record: published sums of each row's
// weights, each masked by a pad. The last P.pads() entries of each message are the row's pads,
// which are random values (prover/hash_commitment/hf_pads.cuh). blocks/code/code.h
// defines where the message and the masks are among the row's KD coefficients: with
// blk::Code{ln, i0, nm}, message entry j is coefficient message_at() + j and mask j is coefficient
// tail_at() + j: the masks from 0 and the message after them, from nm. u is indexed by coefficient.
// symbol(x) = message_scale(x) sum_j message[j] basis_j(x) + tail_scale(x) sum_j tail[j] basis_j(x)
// (j from 0 in each sum), and basis_j(x) is the product of the factors f_i(x) over j's set bits i.
// make_u: u[i] = sum over r of fold[r] M[r][i], and omega = W eq(rw), in one pass (rf::HfFold).
//   Then the pads' part, summed on the host (hf_side_u_pads()), and the masks' part
//   (hf_mask_fold()). Neither reads a weight. Each has a self-check, [own-test-pads] and
//   [own-test-fold] ("own test" in messages and identifiers). A failure prints own-test-pads or
//   own-test-fold and raises the flag OWN_TEST. u is in held.u_host, pinned memory (10.7 GB at full
//   size), or, if null, on the GPU.
// make_columns: a run is the 2^lcl positions h 2^lcl + a, a < 2^lcl. On a run the factors
//   f_(lcl + j) are constant; hi(y) is their product over the set bits j of y. So g[c] = sum over y
//   of hi(y) W[y][c] holds, for every row r, the 2^lcl numbers whose transform of size 2^lcl gives
//   the data's part of row r's symbols on the whole run; k_join_runs() joins the masks' share to
//   them. A drawn position's fibre is the 2^s positions under its node of level s, the lowest level
//   of T's Merkle tree that the prover keeps: one run if s = lcl (normally cp_commit's default),
//   part of a run if s < lcl, 2^(s - lcl) runs otherwise. A portion is at most fibres_a_walk
//   positions, to bound GPU memory: one pass (rf::HfColumns), hf_mask_columns(), the pads' share
//   (hf::pads_into_runs(), from a GPU copy of the pads; [own-test-pads]), the transforms,
//   and hf::show_columns(), which self-checks the fibres and stores the columns' words on the host.
// Buffers (m = lcl). Runs: run[((z R + r) << m) + e] = T[r][h_z 2^m + e], h_z the number of run z;
// those of position I[a] are z = a per .. a per + per - 1, per = hf_side_runs(), in rising order of
// h_z. Fibres: fib[((a R + r) << s) + x] = T[r][(I[a] >> s << s) + x].
// Memory. Every buffer (`room` in names and messages) is allocated under short_lived(false): at a
// call of walk() an opening may hold nothing in borrowed memory (prover/hash_commitment/hf_walk.h).
// Hidden values: u, omega, g, the masks' array g2, the fibres, the pads on the host and on the GPU,
// the pads' part of u, and any scratch buffer that depends on the weights, the masks or the pads.
// Each is zeroed here as soon as its use ends (a u on the GPU: at forget()), except omega and
// held.u_host, left to their owners. If walk() throws, the callback zeroes only its host copy of
// the pads (hf::TablePads's destructor); on exit() it zeroes nothing. rf::Unit zeroes u_host and
// both memory pools in leaf_end() and forget(); if a CUDA call fails in its forget(), it returns 1
// and the pools may stay unzeroed. The columns' words are hidden until show() returns them;
// forget() or a failed self-check zeroes them.
// Include prover/weights_commitment/lent_memory.h, which routes cudaMalloc and cudaFree to the
// opening's memory pools, and prover/weights_commitment/common.cuh before this file.
#pragma once
#include "hash_commitment/hf_code.cuh"
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_own.cuh"
#include "hash_commitment/hf_mask_ref.h"
#include "hash_commitment/hf_open_api.h"
#include "hash_commitment/hf_pads.cuh"
#include "hash_commitment/hf_params.h"
#include "hash_commitment/hf_passes.cuh"
#include "hash_commitment/hf_tree.cuh"
#include "hash_commitment/hf_walk.h"
#include "code/code.h" // blocks/: where a row's message and its masks are among its coefficients
#include "iop/foldeq.h" // blocks/: a stored table's shape, name and record
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace hfs {
// Copies runs into fibres (layouts: top of file). The loop form is public: the grid-stride loop:
// M. Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", NVIDIA developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void k_fibres_of_runs(const u64 *__restrict__ run, const u64 *__restrict__ I, int count, u32 R, int m, int s,
                                 int per, u64 *__restrict__ fib) {
  const size_t all = ((size_t)count * R) << s;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < all; t += (size_t)gridDim.x * blockDim.x) {
    const u64 x = t & (((u64)1 << s) - 1);
    const size_t ar = t >> s, a = ar / R, r = ar % R;
    const u64 first = I[a] >> s << s, pos = first + x;
    const size_t z = a * (size_t)per + (size_t)((pos >> m) - (first >> m));
    fib[t] = run[((z * R + r) << m) + (pos & (((u64)1 << m) - 1))];
  }
}
// The two scales of blocks/code/code.h at every position of some runs: at position h[z] 2^m + e,
// scale[2 ((z << m) + e)] = basis_(data_at) and scale[2 ((z << m) + e) + 1] = basis_(masks_at).
// nf: the number of factors computed, <= hf::MOST_LN; 2^nf must exceed both indices. Public values.
__global__ void k_run_scales(hf::Domain D, int m, const u64 *__restrict__ h, int runs, u64 data_at, u64 masks_at, int nf,
                             u64 *__restrict__ scale) {
  const size_t all = (size_t)runs << m;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < all; t += (size_t)gridDim.x * blockDim.x) {
    u64 f[hf::MOST_LN];
    hf::factors(D, (h[t >> m] << m) + (t & (((u64)1 << m) - 1)), nf, f);
    scale[2 * t] = hf::factor_product(f, data_at), scale[2 * t + 1] = hf::factor_product(f, masks_at);
  }
}
// data[((z R + r) << m) + e] becomes row r's symbol at position e of run z: the data's sum times
// its scale, plus the masks' sum times theirs.
__global__ void k_join_runs(u64 *__restrict__ data, const u64 *__restrict__ masks, const u64 *__restrict__ scale, int runs,
                            u32 R, int m) {
  const size_t all = ((size_t)runs * R) << m;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < all; t += (size_t)gridDim.x * blockDim.x) {
    const size_t at = (((t >> m) / R) << m) + (t & (((size_t)1 << m) - 1));
    const u64 d = hf::code_mul(scale[2 * at], data[t]);
    data[t] = hf::code_add(d, hf::code_mul(scale[2 * at + 1], masks[t]));
  }
}
} // namespace hfs

namespace rf {
[[noreturn]] static inline void hf_side_internal(const char *what) {
  fprintf(stderr, "Open_F's seams: internal error: %s\n", what);
  exit(4);
}

struct HfSide {
  // ---- Filled by the caller. Y: W's units. held.seed ("key" below): key of T's masks, salts, pads
  const hf::HfParams *P = nullptr;
  const hf::HfOpen *mode = nullptr; // read here: tcols, c and once
  const LayW *Y = nullptr;
  HfHeld held;
  std::function<void(bool)> short_lived;
  HfWalk walk;           // brings every unit of W once, and returns with the GPU idle (hf_walk.h)
  int fibres_a_walk = 0; // the most drawn positions per pass; 0: all of a call of `make_columns`
  // The cuBLAS handle for hf_mask_columns(). Null: the calling thread's own (cp_commit). rf::Unit
  // gives its service thread's, so that the opening's thread need not create one, which allocates
  // from the driver; the owner must not use the handle while hf_mask_columns() runs.
  cublasHandle_t blas = nullptr;
  bool lines = false;    // each step's time on stdout: times and public numbers only (cp_commit)
  // ---- The callbacks' state, from make_u() to forget().
  Ext *u = nullptr;       // KD elements: held.u_host, or on the GPU
  bool u_is_ours = false; // u was allocated here, on the GPU
  int tcols = -1;         // of the last make_columns(); -1: none
  int end = hf::OWN_GOOD; // the first self-check of make_columns that failed; a failed [own-test-pads] overrides it
  int failed = hf::OWN_GOOD; // the same, kept through `forget`; a failed [own-test-pads] in make_u sets it too
  int fold_failed = hf::OWN_GOOD; // [own-test-fold]'s failure code, kept likewise; not read here
  std::vector<uint32_t> words; // host: the drawn columns' words, tcols * column_words
};

// Where a row's message and masks start, and how make_columns() gets a run's symbols from them.
// By blk::Code's layout (top of file) the message starts at nm, an odd number
// (hf_table_params.h). Its scale changes from position to position of a run, and a run has at
// least two (lcl >= 1), so the shares cannot be added first: each is summed over its own index
// from 0 and transformed; k_join_runs() joins them as in symbol(x).
struct HfSideRow {
  uint64_t data_at = 0, masks_at = 0;
};
static inline HfSideRow hf_side_row(const hf::HfParams &P) {
  const blk::Code code{(unsigned)P.ln, (size_t)P.i0, (size_t)P.nm};
  if (code.dim() != P.KD) {
    fprintf(stderr, "Open_F's seams: internal error: a row of the weight table has KD = %llu coefficients by its parameters and %llu by the code "
                    "of length 2^%d with a message of %llu entries and %llu masks\n",
            (unsigned long long)P.KD, (unsigned long long)code.dim(), P.ln, (unsigned long long)P.i0, (unsigned long long)P.nm);
    exit(4);
  }
  HfSideRow w;
  w.data_at = code.message_at(), w.masks_at = code.tail_at();
  return w;
}
// The bytes of g2, the masks' array, and of the scales.
static inline size_t hf_side_second_bytes(const hf::HfParams &P, int runs) {
  return hcp::up512((size_t)runs * P.k * 8) + hcp::up512(((size_t)runs * 16) << P.lcl);
}

// ---- Sizes for the caller.
// HfHeld::u_host must hold KD elements of E (the extension field, Ext): the code's dimension.
static inline size_t hf_side_u_elements(const hf::HfParams &P) {
  return (size_t)P.KD;
}
// The number of runs that cover one fibre.
static inline int hf_side_runs(const hf::HfParams &P, int s) {
  return s > P.lcl ? 1 << (s - P.lcl) : 1;
}
static inline int hf_side_portion(const hf::HfOpen &mode, int fibres_a_walk, int tcols) {
  (void)mode;
  return fibres_a_walk > 0 ? std::min(fibres_a_walk, tcols) : tcols;
}
// The passes over the weights in an opening: one for u, and one for each portion. mode.tcols >= 1.
static inline int hf_side_walks(const hf::HfParams &, const hf::HfOpen &mode, int fibres_a_walk = 0) {
  const int portion = hf_side_portion(mode, fibres_a_walk, mode.tcols);
  return 1 + (mode.tcols + portion - 1) / portion;
}
// How many 32-bit words show() writes: 2^c top-layer digests (depth c of T's tree), tcols columns.
static inline size_t hf_side_shown_words(const hf::HfParams &P, const hf::HfOpen &mode, int tcols) {
  return ((size_t)DW << mode.c) + (size_t)tcols * hf::column_words((u32)P.R, P.ln, mode.c);
}
// Scratch for the masks' shares. A larger buffer means fewer, larger steps and the same results.
static inline size_t hf_side_fold_room_bytes(const hf::HfParams &P) {
  return hcp::up512(std::min((size_t)32 << 20, sizeof(Ext) * ((size_t)P.nm + 8) + 1024));
}
// runs: a portion's positions times hf_side_runs(). At full size, say 48 runs get 1 GiB (14 rows of
// masks at a time); 1,622 runs, whose coefficients' digits alone take 0.85 GB, get 4 GiB (28 rows).
static inline size_t hf_side_columns_room_bytes(const hf::HfParams &P, int runs) {
  const size_t most = (size_t)(runs > 256 ? 4 : 1) << 30;
  return hcp::up512(std::max(2 * hf_mask_columns_room_least(P.lcl, runs),
                             std::min(most, (size_t)P.R * 32 * ((size_t)P.nm + 4096) + (size_t)runs * 4096 * 64)));
}
// An upper bound on the GPU bytes that the callbacks hold at any time ("slab": the pool that
// short_lived(false) selects). s: the lowest kept level. u_on_the_gpu: the caller gives no u_host.
static inline size_t hf_side_slab_bytes(const hf::HfParams &P, const LayW &Y, const hf::HfOpen &mode, int s,
                                        int fibres_a_walk = 0, bool u_on_the_gpu = false) {
  const int portion = hf_side_portion(mode, fibres_a_walk, mode.tcols), per = hf_side_runs(P, s), runs = portion * per;
  const size_t held_u = u_on_the_gpu ? hcp::up512(hf_side_u_elements(P) * sizeof(Ext)) : 0;
  const size_t fold = hcp::up512(P.R * sizeof(Ext)) + hcp::up512(sizeof(Ext) << P.lcl) + HfFold::room_bytes(Y, P.b) + 2048;
  const size_t masks_u = hcp::up512(P.R * sizeof(Ext)) + hf_side_fold_room_bytes(P);
  const size_t lists = hcp::up512((size_t)runs * (P.lc - P.lcl) * 8) + hcp::up512((size_t)runs * 8) +
                       hcp::up512((size_t)portion * 8) + hcp::up512(((size_t)runs * 8) << P.lcl);
  const size_t g = hcp::up512((size_t)runs * P.k * 8) + hf_side_second_bytes(P, runs);
  const size_t pads = hcp::up512(hf::pads_card_bytes(P));
  const size_t fibres = s == P.lcl ? 0 : hcp::up512((((size_t)portion * P.R) << s) * 8);
  const size_t after = std::max(hf_side_columns_room_bytes(P, runs),
                                fibres + hcp::up512(hf::show_room_bytes(portion, (u32)P.R, s)));
  const size_t columns = lists + g + pads + std::max(HfColumns::room_bytes(runs) + 4096, after);
  return held_u + std::max(std::max(fold, masks_u), columns) + 8192;
}
// Makes sure the calling thread has a cuBLAS handle (memory from the driver, for good). Unless
// mode.once (cp_commit, which serves no batch), it also runs once, on zeros or no data, the passes'
// kernels, except the decoder of quantized blocks, and each size of product of a pass of mode.tcols
// runs, to load their code before any batch. 48 runs at full size: 0.9 GB (computed), then freed.
// It does the same with the kernel that adds the pads to the runs.
static inline void hf_side_warm_up(const LayW &Y, const hf::HfOpen &mode) {
  blas::blas(cudaStreamPerThread, nullptr, 0);
  if (mode.once) return;
  if (mode.tcols < 1) hf_side_internal("warm_up() is given a mode without columns: was it ever derived? (the new format's programs do not fill P.batch and P.once)");
  HfFold::warm_up();
  HfColumns::warm_up(Y, mode.tcols);
  // Checked for local memory, then launched with 0 runs: every thread returns at once.
  hf::pads_kernels_ready();
  hf::k_pads_into_runs<<<1, 256, 0, cudaStreamPerThread>>>(nullptr, 0, 0, 0, 0, nullptr, 0, 0, nullptr);
  CK(cudaGetLastError());
  CK(cudaStreamSynchronize(cudaStreamPerThread));
}

// ---- The masks as an HfMaskFill (hf_passes.cuh). hf::masks_of_rows() generates blocks of eight
// masks, so a rectangle must begin and end at a multiple of eight. Indices from nm on give 0.
static inline HfMaskFill hf_side_masks(const hf_key32 *key, uint64_t nm) {
  const uint64_t limit = hf::mask_limit(key);
  return [=](uint32_t r0, uint32_t nr, uint64_t j0, uint64_t nj, u64 *out, cudaStream_t st) {
    if (j0 % 8 || nj % 8) hf_side_internal("a rectangle of masks does not begin and end at a block of the generator");
    hf::masks_of_rows(key, limit, r0, nr, (uint32_t)(j0 / 8), nj / 8, nm, out, nj, st);
  };
}

// eq(point, x) for all x < 2^n; point[0] goes with the top bit of x. Host; public values.
static inline std::vector<Ext> hf_side_eq(const Ext *point, int n) {
  std::vector<Ext> eq((size_t)1 << n);
  eq[0] = ext_base(1);
  for (int j = 0; j < n; j++) { // after step j: eq over the first j + 1 coordinates, on j + 1 bits
    Ext one_less{};
    for (int q = 0; q < EDEG; q++) set_coord(one_less, q, gf_sub(q ? 0 : 1, coord(point[j], q)));
    for (size_t x = (size_t)1 << j; x--;) {
      const Ext e = eq[x];
      eq[2 * x] = k3_mul(e, one_less), eq[2 * x + 1] = k3_mul(e, point[j]);
    }
  }
  return eq;
}
// If lines is set: waits for this thread's stream; prints the time since creation or the last call.
struct HfSideLap {
  const bool lines;
  std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
  explicit HfSideLap(bool lines_) : lines(lines_) {}
  void operator()(const char *step) {
    if (!lines) return;
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    const auto now = std::chrono::steady_clock::now();
    printf("Open_F's seams: %s: %.1f s\n", step, std::chrono::duration<double>(now - last).count());
    fflush(stdout);
    last = now;
  }
};
static inline void hf_side_zero_free(void *p, size_t bytes) {
  if (!p) return;
  CK(cudaMemsetAsync(p, 0, bytes, cudaStreamPerThread));
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  cudaFree(p);
}
static inline HfUnits hf_side_units(HfPass *pass) {
  HfUnits units;
  units.cells = [=](const int8_t *cells, size_t ld, size_t rows, size_t first_row, cudaStream_t st) {
    pass->cells(cells, ld, rows, first_row, st);
  };
  units.blocks = [=](const UnitW &u, const void *blocks, int type, const float *gains, float two_e, cudaStream_t st) {
    pass->blocks(u, blocks, type, gains, two_e, st);
  };
  return units;
}
// [own-test-fold]: compares the masks' part of u = sum over r of coef[r] M[r] with the same sum
// over masks that the plain host generator (hf_mask_ref.h) computes from held.key_again, the key's
// second copy. It looks at a few indices only: eight spread evenly over the masks, and those of
// hf_mask_fold_places(), among them both ends of each chunk of 'portion' indices of hf_mask_fold().
// coef: host. v_masks: GPU or host. Returns OWN_GOOD, OWN_MASKS, OWN_KEY_FILE (copy null or zero).
static inline int hf_side_fold_own_test(const hf::HfParams &P, const HfHeld &held, const Ext *coef, const Ext *v_masks,
                                        uint64_t portion) {
  constexpr int PLACES = 8;
  uint8_t bytes[32];
  if (!hf::key_again_bytes(held.key_again, bytes)) return hf::OWN_KEY_FILE;
  const uint64_t limit = hf::mask_limit(held.seed);
  int end = hf::OWN_GOOD;
  std::vector<uint64_t> spread;
  for (int x = 0; x < PLACES; x++) spread.push_back((P.nm - 1) / (PLACES - 1) * x);
  for (const uint64_t j : hf_mask_fold_places(portion, P.nm, spread)) {
    Ext sum{}, have;
    for (uint32_t r = 0; r < (uint32_t)P.R; r++) sum = k3_add(sum, k3_mul(coef[r], ext_base(hf_mask_ref::mask(bytes, r, j, limit))));
    CK(cudaMemcpy(&have, v_masks + j, sizeof(Ext), cudaMemcpyDefault));
    for (int q = 0; q < EDEG; q++)
      if (coord(sum, q) != coord(have, q)) end = hf::OWN_MASKS;
    sum = have = Ext{};
  }
  explicit_bzero(bytes, 32);
  return end;
}

// The masks' part of u under o's key, to v_masks (nm elements), self-checked. eg: fold on the GPU.
static inline void hf_side_u_masks(HfSide &o, const Ext *fold, const Ext *eg, Ext *v_masks, HfSideLap &lap) {
  const hf::HfParams &P = *o.P;
  const cudaStream_t st = cudaStreamPerThread;
  const size_t all = hf_side_fold_room_bytes(P);
  void *room = nullptr;
  CK(cudaMalloc(&room, all));
  hf_mask_fold(o.held.seed, hf::mask_limit(o.held.seed), (uint32_t)P.R, P.nm, eg, v_masks, room, all, st);
  hf_side_zero_free(room, all);
  lap("the masks' part of u");
  const int end = hf_side_fold_own_test(P, o.held, fold, v_masks, hf_mask_fold_made_portion(all));
  lap("the own test on it");
  CK(cudaStreamSynchronize(st));
  if (end != hf::OWN_GOOD)
    fprintf(stderr, "%s: the masks' part of the folded rows is not what a second making of the masks under the key's second copy gives; "
                    "the message is withheld\n", HF_OWN_TEST_FOLD);
  if (end != hf::OWN_GOOD) o.fold_failed = end;
  raise_flags(end == hf::OWN_GOOD ? 0 : OWN_TEST);
}

// ---- The pads of a table.
// make_u and make_columns each derive the pads again, into a separate buffer, and erase it before
// they return. So if one buffer is overwritten or erased too early, u and the columns disagree: do
// not merge the two derivations (prover/hash_commitment/hf_pads.cuh gives the reason in full). Each
// callback runs [own-test-pads] when its buffers are filled and again after their last use.

// Prints the line of a failed [own-test-pads] and stores the failure code. `then` ends the line: a
// sentence on what follows for the caller's work.
static inline void hf_side_pads_failed(HfSide &o, int end, const char *then) {
  fprintf(stderr, "%s: a self-check failed: the random pads of the weight table's rows, as they lie in memory, are not what a second computation from the table's key "
                  "gives. This is a fault of the machine or of the program. %s\n", hf::own_stop(end), then);
  o.failed = end;
}
// The pads' part of u = sum over r of fold[r] M[r], to v_pads (P.pads() elements, GPU or host: u
// from the first pad's coefficient on). It is summed on the host, copied to v_pads, read back and
// compared. fold: host. Returns false, with OWN_TEST raised and a line printed, if a self-check has
// failed: the caller must not use what v_pads then holds.
static inline bool hf_side_u_pads(HfSide &o, const Ext *fold, Ext *v_pads) {
  const hf::HfParams &P = *o.P;
  const cudaStream_t st = cudaStreamPerThread;
  hf::TablePads pads;
  int end = hf::take_table_pads(P, o.held.seed, o.held.key_again, hf::PADS_FOR_THE_FOLD, pads);
  if (end == hf::OWN_GOOD) {
    std::vector<Ext> folded = hf::pads_fold(pads, P, fold), arrived(folded.size());
    const size_t bytes = folded.size() * sizeof(Ext);
    CK(cudaMemcpyAsync(v_pads, folded.data(), bytes, cudaMemcpyDefault, st));
    CK(cudaMemcpyAsync(arrived.data(), v_pads, bytes, cudaMemcpyDefault, st));
    CK(cudaStreamSynchronize(st));
    if (memcmp(arrived.data(), folded.data(), bytes)) end = hf::OWN_PADS;
    explicit_bzero(folded.data(), bytes), explicit_bzero(arrived.data(), bytes);
    if (end == hf::OWN_GOOD) end = hf::pads_own_test(P, o.held.seed, o.held.key_again, pads);
  }
  pads.wipe();
  if (end != hf::OWN_GOOD) hf_side_pads_failed(o, end, "The folded rows are erased, and what is made from this opening of the table is withheld.");
  raise_flags(end == hf::OWN_GOOD ? 0 : OWN_TEST);
  return end == hf::OWN_GOOD;
}

// ---- make_u. Reads only rw's last lcl coordinates: omega = W eq(rw) if fold = eq(rw's first b).
static inline void hf_side_make_u(HfSide &o, const Ext *fold, const Ext *rw, Ext *omega) {
  const hf::HfParams &P = *o.P;
  const cudaStream_t st = cudaStreamPerThread;
  const size_t KD = hf_side_u_elements(P);
  const HfSideRow row = hf_side_row(P);
  if (o.u) hf_side_internal("make_u() was called twice");
  if (!fold || !rw || !omega) hf_side_internal("make_u() needs the fold's coefficients, the point rw and room for omega");
  o.short_lived(false);
  if (o.held.u_host) {
    if (o.held.u_host_elements < KD) hf_side_internal("the host's room for u is too small");
    o.u = o.held.u_host;
  } else {
    CK(cudaMalloc(&o.u, KD * sizeof(Ext)));
    o.u_is_ours = true;
  }
  Ext *eg = nullptr, *el = nullptr;
  CK(cudaMalloc(&eg, P.R * sizeof(Ext)));
  CK(cudaMemcpyAsync(eg, fold, P.R * sizeof(Ext), cudaMemcpyHostToDevice, st));
  const std::vector<Ext> low = hf_side_eq(rw + P.b, P.lcl); // the coordinates of c's low bits
  CK(cudaMalloc(&el, low.size() * sizeof(Ext)));
  CK(cudaMemcpyAsync(el, low.data(), low.size() * sizeof(Ext), cudaMemcpyHostToDevice, st));
  CK(cudaStreamSynchronize(st));
  HfSideLap lap(o.lines);
  {
    HfFold pass;
    pass.begin(*o.Y, P.b, eg, el, o.u + row.data_at, omega);
    o.walk(hf_side_units(&pass));
    o.short_lived(false);
    pass.finish(st);
    pass.wipe(st);
    CK(cudaStreamSynchronize(st));
    pass.end();
  }
  cudaFree(el); // public values
  lap("the walk for u and omega");
  // The message's zeros after the data, up to the pads; the pads' part; the masks' part.
  if (P.pad_entry(0) > P.l) CK(cudaMemsetAsync(o.u + row.data_at + P.l, 0, (size_t)(P.pad_entry(0) - P.l) * sizeof(Ext), st));
  if (!hf_side_u_pads(o, fold, o.u + row.data_at + P.pad_entry(0))) { // without its pads, nothing of the table is returned
    CK(cudaMemsetAsync(o.u, 0, KD * sizeof(Ext), st));
    CK(cudaMemsetAsync(omega, 0, P.m0 * sizeof(Ext), st));
    CK(cudaStreamSynchronize(st));
    cudaFree(eg); // public values
    return;
  }
  hf_side_u_masks(o, fold, eg, o.u + row.masks_at, lap);
  cudaFree(eg); // public values
}

// ---- u_cells. Copies u[first .. first + count - 1] to out (GPU), 0 from KD on; does not wait.
static inline void hf_side_u_cells(HfSide &o, uint64_t first, uint64_t count, Ext *out) {
  const uint64_t KD = hf_side_u_elements(*o.P);
  if (!o.u) hf_side_internal("u_cells() before make_u()");
  if (first + count > ((uint64_t)1 << o.P->lc)) hf_side_internal("u_cells(): an index of 2^lc or more");
  const uint64_t have = first < KD ? std::min(count, KD - first) : 0;
  if (have) CK(cudaMemcpyAsync(out, o.u + first, have * sizeof(Ext), cudaMemcpyDefault, cudaStreamPerThread));
  if (have < count) CK(cudaMemsetAsync(out + have, 0, (count - have) * sizeof(Ext), cudaStreamPerThread));
}

// Turns the shares into the rows' symbols on 'runs' runs, in g. g holds the data's share, g2 the
// masks' (zeroed here after its use). h, tw: the runs'
// numbers and transform tables. scale: scratch of 2 runs 2^lcl words.
static inline void hf_side_symbols(const hf::HfParams &P, const HfSideRow &row, const hf::Domain &D, const u64 *h,
                                   const u64 *tw, int runs, u64 *g, u64 *g2, u64 *scale, cudaStream_t st) {
  const int m = P.lcl;
  const u32 R = (u32)P.R;
  hf::short_runs_encode(g, tw, m, (int)R, runs, false, g, st);
  hf::short_runs_encode(g2, tw, m, (int)R, runs, false, g2, st);
  hfs::k_run_scales<<<hfp::grid((size_t)runs << m), 256, 0, st>>>(D, m, h, runs, row.data_at, row.masks_at, P.lc, scale);
  CK(cudaGetLastError());
  hfs::k_join_runs<<<hfp::grid(((size_t)runs * R) << m), 256, 0, st>>>(g, g2, scale, runs, R, m);
  CK(cudaGetLastError());
  CK(cudaMemsetAsync(g2, 0, (size_t)runs * P.k * 8, st));
}

// ---- make_columns. I: host; tcols positions as drawn, repeats kept.
static inline void hf_side_make_columns(HfSide &o, const uint64_t *I, int tcols) {
  const hf::HfParams &P = *o.P;
  const hf::HfOpen &M = *o.mode;
  const cudaStream_t st = cudaStreamPerThread;
  const int m = P.lcl, s = o.held.s, nb = P.lc - P.lcl, per = hf_side_runs(P, s);
  const u32 R = (u32)P.R;
  if (tcols < 1) hf_side_internal("make_columns() needs a position");
  if (m > hf::MOST_SHORT) hf_side_internal("lk - b is larger than the longest short run");
  const HfSideRow row = hf_side_row(P);
  if (s < 0 || s > P.ln - M.c) hf_side_internal("the lowest kept level lies above the top layer of T's tree");
  for (int a = 0; a < tcols; a++)
    if (I[a] >> P.ln) hf_side_internal("a drawn position is 2^ln or more");
  const size_t column = hf::column_words(R, P.ln, M.c);
  o.tcols = tcols, o.end = hf::OWN_GOOD;
  o.words.assign((size_t)tcols * column, 0u);
  const hf::Domain D = hf::domain(P.ln);
  const int portion = hf_side_portion(M, o.fibres_a_walk, tcols), most = portion * per;
  o.short_lived(false);
  u64 *hv, *h, *pos, *tw, *g;
  CK(cudaMalloc(&hv, (size_t)most * nb * 8));
  CK(cudaMalloc(&h, (size_t)most * 8));
  CK(cudaMalloc(&pos, (size_t)portion * 8));
  CK(cudaMalloc(&tw, ((size_t)most * 8) << m));
  CK(cudaMalloc(&g, (size_t)most * P.k * 8));
  u64 *g2 = nullptr, *scale = nullptr; // the masks' share, zero between uses; scales
  CK(cudaMalloc(&g2, (size_t)most * P.k * 8));
  CK(cudaMemsetAsync(g2, 0, (size_t)most * P.k * 8, st));
  CK(cudaMalloc(&scale, ((size_t)most * 16) << m));
  // This callback's derivation of the pads (see "The pads of a table" above) and a GPU copy for the
  // kernel below. The copy is in the opening's memory pool:
  // if walk() throws, it is zeroed with the pool (prover/hash_commitment/hf_walk.h), and the host
  // buffer by its destructor.
  hf::TablePads pads;
  u64 *pads_card = nullptr;
  hf::pads_kernels_ready(); // it can end the process, so it runs before any pad is derived
  CK(cudaMalloc(&pads_card, hf::pads_card_bytes(P)));
  int pads_end = hf::take_table_pads(P, o.held.seed, o.held.key_again, hf::PADS_FOR_THE_COLUMNS, pads);
  if (pads_end == hf::OWN_GOOD) pads_end = hf::pads_to_card(pads, P, hf::PADS_FOR_THE_COLUMNS, pads_card, st);
  o.end = pads_end;
  for (int a0 = 0; a0 < tcols && o.end == hf::OWN_GOOD; a0 += portion) {
    const int count = std::min(portion, tcols - a0), runs = count * per;
    // The runs' numbers, and on each run the factors f_(lcl + j) of the bits of y.
    std::vector<u64> run_number((size_t)runs), first_position((size_t)runs), high((size_t)runs * nb);
    for (int a = 0; a < count; a++)
      for (int z = 0; z < per; z++)
        run_number[(size_t)a * per + z] = s > m ? ((I[a0 + a] >> s) << (s - m)) + (u64)z : I[a0 + a] >> m;
    for (int z = 0; z < runs; z++) first_position[z] = run_number[z] << m;
    const std::vector<u64> factors = hf::position_factors_host(D, first_position.data(), runs, P.lc);
    for (int z = 0; z < runs; z++)
      for (int j = 0; j < nb; j++) high[(size_t)z * nb + j] = factors[(size_t)z * P.lc + m + j];
    CK(cudaMemcpyAsync(hv, high.data(), high.size() * 8, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(h, run_number.data(), run_number.size() * 8, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(pos, I + a0, (size_t)count * 8, cudaMemcpyHostToDevice, st));
    hf::run_tables(D, m, h, runs, tw, st);
    CK(cudaStreamSynchronize(st));
    HfSideLap lap(o.lines);
    {
      HfColumns pass;
      pass.begin(*o.Y, runs, hv, nb, g);
      CK(cudaStreamSynchronize(st)); // g is zeroed on this stream; the units come on the caller's
      o.walk(hf_side_units(&pass));
      o.short_lived(false);
      pass.finish(st);
      pass.wipe(st);
      CK(cudaStreamSynchronize(st));
      pass.end();
    }
    lap("a walk for the drawn columns");
    {
      const size_t all = hf_side_columns_room_bytes(P, runs);
      void *room = nullptr;
      CK(cudaMalloc(&room, all));
      // The masks' share has the array g2 and its own index from 0.
      // If a run holds a block of eight masks, the kernel that writes their digits generates them.
      if (m >= 3)
        hf_mask_columns(nullptr, R, m, P.nm, hv, nb, 0, runs, g2, room, all, st, o.held.seed, hf::mask_limit(o.held.seed),
                        o.blas);
      else
        hf_mask_columns(hf_side_masks(o.held.seed, P.nm), R, m, P.nm, hv, nb, 0, runs, g2, room, all, st, nullptr, 0, o.blas);
      hf_side_zero_free(room, all);
    }
    // A pad is an entry of the row's message, so its share is added to g, like the data's.
    hf::pads_into_runs(pads_card, P, hv, nb, runs, g, st);
    lap("the masks' share of them");
    hf_side_symbols(P, row, D, h, tw, runs, g, g2, scale, st);
    u64 *fib = g;
    const size_t fib_bytes = (((size_t)count * R) << s) * 8;
    if (s != m) {
      CK(cudaMalloc(&fib, fib_bytes));
      hfs::k_fibres_of_runs<<<hfp::grid(fib_bytes / 8), 256, 0, st>>>(g, pos, count, R, m, s, per, fib);
      CK(cudaGetLastError());
    }
    const size_t room_bytes = hf::show_room_bytes(count, R, s);
    void *room = nullptr;
    CK(cudaMalloc(&room, room_bytes));
    int which = -1;
    o.end = hf::show_columns(o.held.seed, o.held.key_again, I + a0, count, fib, R, P.ln, s, M.c, hf::TreeTags(P.configuration, HF_NUM_F, (u32)P.R), o.held.nodes,
                             o.words.data() + (size_t)a0 * column, room, room_bytes, st, &which, a0 == 0);
    if (o.end != hf::OWN_GOOD)
      fprintf(stderr, "%s: an own test on the drawn columns of the weight table has failed%s; nothing of them is shown\n",
              hf::own_stop(o.end), o.end == hf::OWN_NODE ? ": a node of the lowest kept level, made again, is not the kept one" : "");
    hf_side_zero_free(room, room_bytes);
    if (fib != g) hf_side_zero_free(fib, fib_bytes);
    CK(cudaMemsetAsync(g, 0, (size_t)runs * P.k * 8, st));
    CK(cudaStreamSynchronize(st));
    lap("their transforms, leaves, nodes and own tests");
  }
  // Both buffers again, after their last use and before `show` can return a word.
  if (pads_end == hf::OWN_GOOD) pads_end = hf::pads_own_test(P, o.held.seed, o.held.key_again, pads);
  if (pads_end == hf::OWN_GOOD) pads_end = hf::pads_card_test(pads, P, pads_card, st);
  hf_side_zero_free(pads_card, hf::pads_card_bytes(P));
  pads.wipe();
  if (pads_end != hf::OWN_GOOD) { // reported instead of a failed check of the nodes, which wrong pads cause
    hf_side_pads_failed(o, pads_end, "No column of the table is shown, and what is made from this opening of the table is withheld.");
    o.end = pads_end;
  }
  if (o.end != hf::OWN_GOOD) std::fill(o.words.begin(), o.words.end(), 0u); // earlier portions' too
  if (o.end != hf::OWN_GOOD) o.failed = o.end;
  cudaFree(g), cudaFree(g2); // zeroed above
  cudaFree(scale);           // public values
  cudaFree(tw), cudaFree(pos), cudaFree(h), cudaFree(hv); // public values
}

// ---- show. Writes the words if make_columns' self-checks passed; else zeros, and raises OWN_TEST.
static inline void hf_side_show(HfSide &o, uint32_t *out) {
  const hf::HfParams &P = *o.P;
  if (o.tcols < 1) hf_side_internal("show() before make_columns()");
  const size_t top = (size_t)DW << o.mode->c;
  if (o.words.size() != (size_t)o.tcols * hf::column_words((u32)P.R, P.ln, o.mode->c))
    hf_side_internal("the columns' words do not have the planned length");
  memset(out, 0, hf_side_shown_words(P, *o.mode, o.tcols) * 4);
  raise_flags(o.end == hf::OWN_GOOD ? 0 : OWN_TEST);
  if (o.end != hf::OWN_GOOD) return;
  hf::top_layer(o.held.nodes, o.mode->c, out);
  memcpy(out + top, o.words.data(), o.words.size() * 4);
}

// ---- forget. Zeroes and frees a u on the GPU; zeroes the words. Callable at any time, repeatedly.
static inline void hf_side_forget(HfSide &o) {
  if (o.u && o.u_is_ours) // a u_host is its owner's to zero
    hf_side_zero_free(o.u, hf_side_u_elements(*o.P) * sizeof(Ext));
  o.u = nullptr, o.u_is_ours = false;
  volatile uint32_t *w = o.words.data();
  for (size_t i = 0; i < o.words.size(); i++) w[i] = 0;
  o.words.clear();
  o.tcols = -1, o.end = hf::OWN_GOOD;
}

// ---- The table's record (blocks/iop/foldeq.h).
// Defined in docs/format/FORMAT.md, R4.8, and docs/format/09-key-and-leaf-files.md, R13.8.
// The most GPU memory that hf_side_record() allocates, each buffer rounded up to 512 bytes.
static inline size_t hf_side_record_bytes(const hf::HfParams &P) {
  const size_t sums = (size_t)P.pads();
  return hcp::up512(sums * P.m0 * 8) + hcp::up512(sums * P.k * 8) + HfColumns::room_bytes((int)sums) + 4096;
}
// Computes the record of o's table: s = shape.answers answers for each row r, each equal to
// blk::record_value() of the row's message M[r]:
//     answer[r s + i] = beta_i pad(r, i) + the sum over e < P.l of eq(e; z_i) M[r][e],
//     pad(r, i) = pad[r][3 i] + pad[r][3 i + 1] X + pad[r][3 i + 2] X^2.
// `name` must be this table's name (blk::index_name()): the seeds (z_i, beta_i) are derived from
// it, and answers under two names with one set of pads would reveal exact linear equations on the
// model's weights. `pads`: the caller must derive them for this call only and run [own-test-pads]
// before and after it (prover/hash_commitment/hf_pads.cuh); only their number is checked here.
// It takes one pass over the weights. With e = j 2^lcl + cl,
// eq(e; z) = eq(cl; z_low) eq(j; z_high). For q < 3, rf::HfColumns computes
//     G_q[c] = the sum over j of (coordinate q of eq(j; z_high)) W[j][c],
// and the host adds the sum over cl of eq(cl; z_low) (G_0 + G_1 X + G_2 X^2)[r 2^lcl + cl].
// An answer starts on the host as beta_i pad(r, i), so a failed launch gives a wrong answer and
// never an answer without its pad. G and the pass's buffers are hidden values and are zeroed here.
static inline blk::IndexRecord hf_side_record(HfSide &o, const blk::IndexShape &shape, const std::vector<blk::u8> &name, const hf::TablePads &pads) {
  const hf::HfParams &P = *o.P;
  const cudaStream_t st = cudaStreamPerThread;
  const unsigned m = (unsigned)P.lcl, bits = shape.cube_bits(), s = shape.answers;
  blk::IndexRecord record;
  if (!(shape.code == blk::Code{(unsigned)P.ln, (size_t)P.i0, (size_t)P.nm}) || shape.v != (unsigned)P.b || s != (unsigned)P.answers || o.Y->m0 != P.m0 || o.Y->k != P.k)
    hf_side_internal("a record is asked for under another shape or layout than the table's");
  if (!o.held.seed || pads.host.size() != P.R * P.pads()) hf_side_internal("a record's answers are made only under a key, with all the table's pads at hand");
  if (bits < m || ((P.m0 - 1) >> (bits - m)) || shape.pad_place(0, 0) != P.pad_entry(0) || P.l > P.pad_entry(0))
    hf_side_internal("the point of a record's sample or the pads' place does not fit the rows of the weight table");
  const std::vector<blk::SampleSeed> seeds = blk::record_seeds(shape, name);
  const int sums = (int)P.pads();
  // values[(3 i + q) m0 + j] = coordinate q of eq(j; the coordinates of z_i from lcl on).
  std::vector<u64> values((size_t)sums * P.m0);
  {
    std::vector<blk::E> high((size_t)1 << (bits - m));
    for (unsigned i = 0; i < s; i++) {
      blk::eq_table(seeds[i].z.data() + m, bits - m, high.data());
      for (size_t j = 0; j < P.m0; j++)
        for (unsigned q = 0; q < blk::E_DEGREE; q++) values[((size_t)blk::E_DEGREE * i + q) * P.m0 + j] = high[j].c[q].v;
    }
  }
  // All host memory is allocated before the pass: a failed allocation ends the process, and until
  // then nothing has been computed from the weights.
  const size_t g_bytes = (size_t)sums * P.k * 8;
  std::vector<blk::E> low((size_t)1 << m);
  std::vector<u64> G((size_t)sums * P.k);
  CK(cudaHostRegister(G.data(), g_bytes, cudaHostRegisterDefault)); // page-locked: G cannot be paged out, and the copy from the GPU needs no staging buffer
  record.y.resize(P.R * s);
  o.short_lived(false);
  u64 *factors, *g;
  CK(cudaMalloc(&factors, values.size() * 8));
  CK(cudaMalloc(&g, g_bytes));
  CK(cudaMemcpyAsync(factors, values.data(), values.size() * 8, cudaMemcpyHostToDevice, st));
  CK(cudaStreamSynchronize(st));
  HfSideLap lap(o.lines);
  {
    HfColumns pass;
    pass.begin_values(*o.Y, sums, factors, g);
    CK(cudaStreamSynchronize(st)); // g is zeroed on this thread's stream; the units arrive on the caller's
    o.walk(hf_side_units(&pass));
    o.short_lived(false);
    pass.finish(st);
    pass.wipe(st);
    CK(cudaStreamSynchronize(st));
    pass.end();
  }
  lap("a walk for the record's answers");
  // G is a hidden value in host memory: until it is zeroed below, no call can end the process and
  // leave it there.
  const cudaError_t fetched = cudaMemcpy(G.data(), g, g_bytes, cudaMemcpyDeviceToHost);
  if (fetched != cudaSuccess) explicit_bzero(G.data(), g_bytes);
  CK(fetched);
  for (unsigned i = 0; i < s; i++) {
    blk::eq_table(seeds[i].z.data(), m, low.data());
    const u64 *G0 = G.data() + (size_t)blk::E_DEGREE * i * P.k, *G1 = G0 + P.k, *G2 = G1 + P.k;
    for (size_t r = 0; r < P.R; r++) {
      const u64 *p = pads.host.data() + r * P.pads() + (size_t)blk::E_DEGREE * i;
      blk::E &answer = record.y[r * s + i];
      answer = seeds[i].beta * blk::E::from(blk::F{p[0]}, blk::F{p[1]}, blk::F{p[2]});
      for (size_t cl = 0; cl < low.size(); cl++) {
        const size_t c = (r << m) + cl;
        answer += low[cl] * blk::E::from(blk::F{G0[c]}, blk::F{G1[c]}, blk::F{G2[c]});
      }
    }
  }
  explicit_bzero(G.data(), g_bytes);
  hf_side_zero_free(g, g_bytes);
  CK(cudaHostUnregister(G.data()));
  cudaFree(factors); // public values
  lap("the record's answers from the walk's sums");
  return record;
}

// Returns the callbacks on o, which must outlive them.
static inline HfSeams hf_side_seams(HfSide &o) {
  if (!o.P || !o.mode || !o.Y || !o.walk || !o.short_lived || !o.held.nodes || !o.held.seed != !o.held.key_again)
    hf_side_internal("the seams need the parameters, the mode, the layout, a walk, short_lived, the kept nodes, and a key "
                     "with its second copy or neither");
  if (!o.held.seed) hf_side_internal("a table is opened without its key, so without its rows' pads");
  HfSeams seams;
  seams.short_lived = o.short_lived;
  seams.make_u = [&o](const ::Ext *fold, const ::Ext *rw, ::Ext *omega) { hf_side_make_u(o, fold, rw, omega); };
  seams.u_cells = [&o](uint64_t first, uint64_t count, ::Ext *out) { hf_side_u_cells(o, first, count, out); };
  seams.make_columns = [&o](const uint64_t *I, int tcols) { hf_side_make_columns(o, I, tcols); };
  seams.show = [&o](uint32_t *out) { hf_side_show(o, out); };
  seams.forget = [&o]() { hf_side_forget(o); };
  return seams;
}
} // namespace rf
