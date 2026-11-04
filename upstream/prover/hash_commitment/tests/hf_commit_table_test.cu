// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A GPU test, at toy sizes, of how the committer program cp_commit builds a weight table's Merkle
// tree: make_table() (prover/hash_commitment/hf_commit_table.cuh) and rows_of_panel()
// (prover/hash_commitment/hf_commit_cells.cuh). Each shape, HfParams::toy(0) to toy(5), is run on
// random cells (signed bytes for weights) with a key, from which masks (encoded with each row's
// cells), salts and pads are derived. The keys come from a fixed-seed generator and hide nothing.
// Three of the shapes are run again, with one (the run above once more), two and four stored
// answers. Checks: (1) to (6)
// below, and tree()'s. The implementations that it compares with use the same field arithmetic and
// the same BLAKE2b compression function and initialization vector as the GPU code, so the test
// cannot catch a fault in either. A pass shows agreement at these sizes, not hiding.
//
// Build, from prover/:  ./build.sh hc_unit_test tests/hf_commit_table_test
// On success the last line is "hf_commit_table_test: all <n> cases agree", exit code 0. A failed
// check prints a line with DIFFERS; the exit code is then 1.
// With `small` as its only argument the test runs only toy(0): under Compute Sanitizer's racecheck
// the entire test did not finish within 2,500 s on an otherwise idle H100; without a tool it takes
// 47 s. Kernel variants, launch shapes and branches that only toy(1) to toy(5) reach are then not
// run under the tool. The test that runs `small` under racecheck is not part of this repository,
// and it watches only the first 400 launches (docs/limits-all-rows.tsv, row T3).
#include "hash_commitment/hf_commit_table.cuh"
#include "hash_commitment/hf_code_ref.h"
#include "hash_commitment/hf_mask_ref.h"
#include "hash_commitment/hf_tree_ref.h"
#include <chrono>
#include <cstring>
#include <random>
#include <unistd.h>
using namespace hf;

static int cases = 0, bad = 0;
static void say(bool good, const char *what, const std::string &where) {
  cases++;
  if (good) return;
  bad++;
  printf("DIFFERS: %s: %s\n", what, where.c_str());
}
// All of M, the R rows of l cells that a table encodes, in GPU memory.
struct RandomCells : RowCells {
  const HfParams &P;
  size_t stride;
  int8_t *M = nullptr;
  RandomCells(const HfParams &P_, const std::vector<int8_t> &host, size_t stride_) : P(P_), stride(stride_) {
    CK(cudaMalloc((void **)&M, host.size()));
    CK(cudaMemcpy(M, host.data(), host.size(), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
  }
  const int8_t *rows(uint32_t r0, int, size_t *s, cudaStream_t) override { return *s = stride, M + r0 * stride; }
  int most() const override { return (int)P.R; }
  size_t bytes() const override { return P.R * stride; }
  void release() override {
    if (M) CK(cudaFree(M));
    M = nullptr;
  }
};
// The key's second copy: make_table() compares some of its masks with this copy's [own-test-masks].
static uint8_t key_again[32];
static void write_key(const hf_key32 &key) {
  memcpy(key_again, key.w, 32);
}
static std::vector<u32> tree(const HfParams &P, int lp, int s, const hf_key32 *key, const std::vector<int8_t> &cells,
                             size_t stride, int np = 1) {
  std::vector<u32> heap(kept_digests(P.ln, s) * DW, 0xA5A5A5A5u);
  TablePlan plan;
  plan.lp = lp, plan.s = s, plan.np = np, plan.lines = false;
  plan.key_again = key_again;
  const bool passed = make_table(P, plan, key, [&](int) { return new RandomCells(P, cells, stride); }, heap.data());
  say(passed, "make_table() does not say that the masks' own test has passed", "");
  return heap;
}
static void one_shape(int which, const HfParams &P, std::mt19937_64 &rng) {
  const std::string name = "toy " + std::to_string(which) + ", " + std::to_string(P.answers) + " stored answers";
  const size_t stride = ((size_t)P.l + 255) & ~(size_t)255, rows16 = (P.R + 15) & ~(size_t)15;
  std::vector<int8_t> cells(rows16 * stride, 77); // padding of 77 (i >= l, r >= R) must not matter
  for (size_t r = 0; r < P.R; r++)
    for (size_t i = 0; i < P.l; i++) cells[r * stride + i] = (int8_t)(rng() & 255);
  hf_key32 key{};
  for (u32 &w : key.w) w = (u32)rng();
  uint8_t key_bytes[32];
  memcpy(key_bytes, key.w, 32);
  write_key(key);
  const hf_key32 *k = &key;
  const u64 KD = P.i0 + P.nm;
  // (1) make_table() works in parts of 2^lp positions, np at a time: neither may change a digest.
  const std::vector<u32> heap = tree(P, P.ln > 28 ? 28 : P.ln, 0, k, cells, stride);
  for (int lp : {P.ln - 1, P.ln - 3, (P.ln + 1) / 2, 5, 1})
    if (lp >= 1 && P.ln - lp <= 12)
      say(tree(P, lp, 0, k, cells, stride) == heap, "parts of another size give another tree",
          name + ", lp " + std::to_string(lp));
  for (int np : {2, 3})
    say(tree(P, P.ln - 3, 0, k, cells, stride, np) == heap, "parts side by side give another tree",
        name + ", np " + std::to_string(np));
  // (2) Leaves, at ten positions only, against host reference code (`hfr`,
  // prover/hash_commitment/hf_code_ref.h: a symbol by its defining sum).
  std::vector<u64> row(KD), column(P.R);
  std::vector<u64> at = {0, 1, P.n - 1, P.n / 2, P.n / 2 - 1};
  for (int x = 0; x < 5; x++) at.push_back(rng() % P.n);
  std::vector<std::vector<u64>> columns(at.size(), std::vector<u64>(P.R));
  for (size_t r = 0; r < P.R; r++) {
    // A row's KD coefficients, placed without blk::Code, which make_table() uses:
    // nm masks, l cells, zeros.
    const u64 data_at = P.nm, mask_at = 0;
    for (u64 i = 0; i < KD; i++) {
      row[i] = 0;
      if (i >= data_at && i - data_at < P.l) {
        const int c = cells[r * stride + (i - data_at)];
        row[i] = c >= 0 ? (u64)c : HF_P - (u64)(-c);
      }
      if (i >= mask_at && i - mask_at < P.nm) row[i] = hf_mask_ref::mask(key_bytes, (u32)r, i - mask_at, hf_mask_ref::FULL);
    }
    // With s answers, a row's pads are the last 3 s of the i0 entries from data_at on.
    for (u64 j = 0; j < 3 * (u64)P.answers; j++)
      row[data_at + P.i0 - 3 * (u64)P.answers + j] = hf_mask_ref::pad(key_bytes, (u32)r, j, hf_mask_ref::FULL);
    for (size_t x = 0; x < at.size(); x++) columns[x][r] = hfr::enc_at(P.ln, row.data(), KD, at[x]);
  }
  for (size_t x = 0; x < at.size(); x++) {
    u64 salt[5];
    for (int sigma = 0; sigma < 5; sigma++) salt[sigma] = hf_mask_ref::salt(key_bytes, at[x], sigma);
    u32 leaf[DW];
    hf_tree_ref::leaf(leaf, columns[x].data(), (u32)P.R, salt, P.ln, HF_NUM_F, at[x], P.configuration);
    say(!memcmp(leaf, heap.data() + kept_at(P.ln, 0, at[x]) * DW, DW * 4), "a leaf is not that of the defining sums",
        name + ", position " + std::to_string(at[x]));
  }
  // (3) Every level above the leaves, recomputed on the host from the GPU's leaves.
  const std::vector<u32> leaves(heap.begin() + kept_at(P.ln, 0, 0) * DW, heap.end());
  const auto level = hf_tree_ref::whole(leaves, P.ln, P.configuration);
  for (int l = 1; l <= P.ln; l++)
    say(!memcmp(level[l].data(), heap.data() + kept_at(P.ln, l, 0) * DW, level[l].size() * 4),
        "a level is not that of the plain tree", name + ", level " + std::to_string(l));
  // (4) A tree kept only from level s up has the full tree's digests.
  for (int s : {P.lcl, 3, P.ln})
    if (s <= P.ln) {
      const std::vector<u32> kept = tree(P, std::max(s, std::min(P.ln, 9)), s, k, cells, stride);
      say(!memcmp(kept.data() + DW, heap.data() + DW, (kept.size() - DW) * 4), "a tree kept from a higher level differs",
          name + ", s " + std::to_string(s));
    }
}
// (5) On toy(0), toy(1) and toy(5): rows_of_panel() against the layout's rule, HfParams::row_of()
// and coef_of().
static void panels(const HfParams &P, std::mt19937_64 &rng) {
  const size_t rows = 37, pitch = 48, y0 = P.m0 > 100 ? 61 : 0, stride = ((size_t)P.l + 255) & ~(size_t)255;
  std::vector<int8_t> pan(P.k * pitch), out(P.R * stride, 0), want(P.R * stride, 0);
  for (auto &b : pan) b = (int8_t)(rng() & 255);
  for (size_t c = 0; c < P.k; c++)
    for (size_t a = 0; a < rows; a++) want[P.row_of(y0 + a, c) * stride + P.coef_of(y0 + a, c)] = pan[c * pitch + a];
  int8_t *p, *o;
  CK(cudaMalloc((void **)&p, pan.size()));
  CK(cudaMalloc((void **)&o, out.size()));
  CK(cudaMemcpy(p, pan.data(), pan.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(o, 0, out.size()));
  CK(cudaDeviceSynchronize());
  rows_of_panel(P, p, pitch, rows, y0, 0, (u32)P.R, o, stride, cudaStreamPerThread);
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  CK(cudaMemcpy(out.data(), o, out.size(), cudaMemcpyDeviceToHost));
  say(out == want, "rows_of_panel() is not the layout's rule", "fb " + std::to_string(P.b));
  CK(cudaFree(p));
  CK(cudaFree(o));
}
// (6) A toy shape with `answers` answers in its record: every row has pads.
static HfParams with_answers(int which, int answers) {
  HfParams P = HfParams::toy(which);
  P.answers = answers;
  const std::string why = P.derive();
  say(why.empty(), "a toy shape with these answers is not admitted", why);
  return P;
}
int main(int argc, char **argv) {
  std::mt19937_64 rng(20261010);
  const bool small = argc == 2 && !strcmp(argv[1], "small");
  for (int which : {0, 1, 2, 3, 4, 5})
    if (!small || !which)
      one_shape(which, HfParams::toy(which), rng);
  for (int which : {0, 1, 5})
    if (!small || !which)
      one_shape(which, with_answers(which, which == 1 ? 2 : which == 5 ? 4 : 1), rng);
  for (int which : {0, 1, 5})
    if (!small || !which) panels(HfParams::toy(which), rng);
  if (bad) return printf("hf_commit_table_test: %d of %d cases DIFFER\n", bad, cases), 1;
  printf("hf_commit_table_test: all %d cases agree\n", cases);
  return 0;
}
