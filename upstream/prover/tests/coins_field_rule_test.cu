// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the rule by which a 64-bit word of the generator of private randomness (`coin` in
// identifiers) becomes a cell, one element of the field modulo p = 2^64 - 2^32 + 1: the word's low
// FIELD_BITS bits (p's length in bits), or a retry if these are not below p (the word is rejected;
// printed: refused). One try is field_from_word() in prover/field/field_names.h; hz::coin4() and
// hz::coin_again() in prover/hiding/trees.cuh apply it on the GPU, prover/hiding/coins_by_value.cuh
// on the host. Cells are used for blinding, so each must be exactly uniform below p. A rule made
// for another prime can compile, agree on GPU and host, pass every honest run and still draw from a
// small part of the field: a 61-bit mask leaves out seven eighths of it, and p used as a mask
// leaves bits 1 to 31 of every cell at 0. Under the fixed public test key of hz::key_test():
//   1. 2^24 cells of hz::fill(): all below p, each bit set in half, within six standard deviations.
//   2. The host gives the same cells. Every fifth cell equals cell_by_reading(), this file's form
//      of the rule without field_from_word().
//   3. hz::coin_again() by itself gives the first retry that is below p; at rejected words found in
//      the stream, hz::fill() and the host give that retry's cell.
// One H100. From prover/: tests/unit.sh both tests/coins_field_rule_test.cu <directory>
#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "hiding/coins_by_value.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
static int failed = 0, lines = 0;
static void line(bool good, const char *what, unsigned long long a = 0, unsigned long long b = 0) {
  printf("%s (%llu, %llu): %s\n", what, a, b, good ? "ok" : "FAILS");
  failed += !good, lines++;
}
// This file's form of the rule: the low FIELD_BITS bits of two 32-bit words, taken if below p.
static bool taken(const u32 *o, int pair, u64 &cell) {
  const u64 word = ((u64)o[2 * pair + 1] << 32) | o[2 * pair];
  cell = FIELD_BITS == 64 ? word : word & (((u64)1 << (FIELD_BITS & 63)) - 1);
  return cell < FIELD_P;
}
// Cell i of the stream, by taken(): pair i mod 4 of call i / 4 of the generator (coin_output() in
// prover/common/coin_chacha.h; a call gives four cells, its lanes). Retry k = 1, 2, ... has the
// retry word RETRYQ k + (i mod 4) and takes pair 0. The tested code calls the same coin_output():
// the generator is not tested here.
static u64 cell_by_reading(const hz::CoinK &key, u32 stream, u64 i, int *tries = nullptr) {
  u32 o[COINOUTW];
  u64 cell;
  coin_output(o, key.k, stream, i >> 2, 0);
  int n = 1;
  if (!taken(o, (int)(i & 3), cell))
    for (u32 k = 1;; k++) {
      coin_output(o, key.k, stream, i >> 2, RETRYQ * k + (u32)(i & 3));
      n++;
      if (taken(o, 0, cell)) break;
    }
  if (tries) *tries = n;
  return cell;
}
__global__ void k_again(const __grid_constant__ hz::CoinK ck, u32 stream, u64 bk0, int n, u64 *out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = hz::coin_again(ck, stream, bk0 + (t >> 2), t & 3);
}
// Writes the numbers of the cells from cell 4 bk0 on whose first word is rejected; at most `room`.
// Searches `calls` calls and adds the number of all such cells to `*count`, which must start at 0.
__global__ void k_refused(const __grid_constant__ hz::CoinK ck, u32 stream, u64 bk0, u64 calls, unsigned long long *found, unsigned *count, unsigned room) {
  for (u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; t < calls; t += (u64)gridDim.x * blockDim.x) {
    u32 o[COINOUTW];
    hz::coin_words(ck, stream, bk0 + t, o);
    for (int q = 0; q < 4; q++) {
      u64 v;
      if (!field_from_word(((u64)o[2 * q + 1] << 32) | o[2 * q], v)) {
        const unsigned at = atomicAdd(count, 1u);
        if (at < room) found[at] = 4 * (bk0 + t) + q;
      }
    }
  }
}
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coins_field_rule_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  printf("coins_field_rule_test: BLK_FIELD %d, p = %llu, %d bits\n", BLK_FIELD, (unsigned long long)FIELD_P, FIELD_BITS);
  hz::key_test();
  hz::set_attempt(5, 6, 2);
  const u32 stream = hz::ATTEMPT_STREAMS_FROM + 41;
  hz::CoinK key = hz::coink(stream);
  { // 1 and 2
    const size_t n = (size_t)1 << 24;
    u64 *gpu;
    CK(cudaMalloc(&gpu, 8 * n));
    hz::fill(gpu, stream, 0, n);
    std::vector<u64> got(n);
    CK(cudaMemcpy(got.data(), gpu, 8 * n, cudaMemcpyDeviceToHost));
    CK(cudaFree(gpu));
    size_t above = 0, ones[64] = {}, high = 0, high_top = 0, other = 0;
    for (u64 v : got) {
      above += v >= FIELD_P;
      for (int b = 0; b < 64; b++) ones[b] += v >> b & 1;
    }
    line(!above, "every cell is below p: cells, above", n, above);
    const double six = 6 * std::sqrt((double)n) / 2;
    int off = 0, worst = -1;
    double worst_by = 0;
    for (int b = 0; b < 64; b++) {
      const double by = std::fabs((double)ones[b] - (b < FIELD_BITS ? n / 2.0 : 0.0));
      if (by > (b < FIELD_BITS ? six : 0.0)) off++;
      if (by > worst_by) worst_by = by, worst = b;
    }
    line(!off, "each of the field's bits is set in half of the cells, within six standard deviations, and no higher bit in any: bits off, the bit furthest off", off, worst);
    // Checks the top eighth of the range: a quarter of the cells should have the two bits below the
    // top set; half of those, all three.
    for (u64 v : got)
      if ((v >> (FIELD_BITS - 3) & 3) == 3) high++, high_top += v >> (FIELD_BITS - 1) & 1;
    line(std::fabs((double)high - n / 4.0) <= 6 * std::sqrt(n * 3.0 / 16) && std::fabs((double)high_top - high / 2.0) <= 6 * std::sqrt((double)high) / 2, "the top eighth of the words is drawn as often as any: cells with the two bits below the top set, with the top bit too", high, high_top);
    for (size_t i = 0; i < n; i += 5) other += got[i] != cell_by_reading(key, stream, i); // a step of 5 visits all four lanes of a call equally often
    line(!other, "a cell is the generator's word, by this file's reading: differing", other);
    line(!rf::hf_cells_differ_on_threads(key, stream, 0, got.data(), n), "the host's second derivation gives the same cells");
  }
  { // 3: coin_again() by itself
    const int n = 4096;
    const u64 bk0 = (u64)1 << 40;
    u64 *gpu;
    CK(cudaMalloc(&gpu, 8 * n));
    k_again<<<(n + 255) / 256, 256>>>(key, stream, bk0, n, gpu);
    CK(cudaGetLastError());
    std::vector<u64> got(n);
    CK(cudaMemcpy(got.data(), gpu, 8 * n, cudaMemcpyDeviceToHost));
    CK(cudaFree(gpu));
    size_t bad = 0;
    for (int t = 0; t < n; t++) {
      u32 o[COINOUTW];
      u64 cell = 0;
      for (u32 k = 1;; k++) {
        coin_output(o, key.k, stream, bk0 + (t >> 2), RETRYQ * k + (u32)(t & 3));
        if (taken(o, 0, cell)) break;
      }
      bad += got[t] != cell;
    }
    line(!bad, "coin_again() gives the first retry's word that is below p: calls, wrong", n, bad);
  }
  // 3, in the stream. About one word in 2^32 is rejected: k_refused() should find about eight in
  // 2^35 words.
  {
    const u64 bk0 = (u64)1 << 24, calls = (u64)1 << 33; // after the cells of 1
    const unsigned room = 64;
    unsigned long long *found;
    unsigned *n_found;
    CK(cudaMalloc(&found, 8 * room));
    CK(cudaMalloc(&n_found, 4));
    CK(cudaMemset(n_found, 0, 4));
    k_refused<<<132 * 8, 256>>>(key, stream, bk0, calls, found, n_found, room);
    CK(cudaGetLastError());
    unsigned n = 0;
    CK(cudaMemcpy(&n, n_found, 4, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> at(room);
    CK(cudaMemcpy(at.data(), found, 8 * room, cudaMemcpyDeviceToHost));
    line(n >= 1 && n <= room, "refused words among 2^35 (about eight are expected): found", n);
    size_t bad = 0;
    for (unsigned x = 0; x < n && x < room; x++) {
      const u64 i = at[x], first = i & ~(u64)3;
      int tries = 0;
      const u64 want = cell_by_reading(key, stream, i, &tries);
      u64 *gpu, got[4];
      CK(cudaMalloc(&gpu, 32));
      // If a call had two rejected words (not expected), its four cells would be drawn twice here.
      hz::fill(gpu, stream, first, 4); // `fill` ends the process on a second draw
      CK(cudaMemcpy(got, gpu, 32, cudaMemcpyDeviceToHost));
      CK(cudaFree(gpu));
      const bool good = tries >= 2 && want < FIELD_P && got[i & 3] == want && rf::hf_coin_on_the_host(key, stream, i) == want &&
                        !rf::hf_cells_differ(key, stream, first, got, 4);
      bad += !good;
      printf("  cell %llu: its word is refused; the cell comes from try %d; fill() and the host's derivation %s\n", (unsigned long long)i, tries, good ? "give it" : "DIFFER");
    }
    line(!bad, "at every refused word fill() and the host's second derivation give the retry's cell: wrong", bad);
    CK(cudaFree(found)); CK(cudaFree(n_found));
  }
  hz::forget(key);
  if (failed) printf("%d of %d lines fail\n", failed, lines);
  else printf("all %d lines ok\n", lines);
  return failed ? 1 : 0;
}
