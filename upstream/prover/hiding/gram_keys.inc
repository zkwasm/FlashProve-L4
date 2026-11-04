// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// KeyPasses runs the first three Gram passes of a product layer for PtPL::prove(), its only caller.
// The layer's entries, alpha - val.value(key), are computed from the N 16-bit keys, not stored;
// alpha is E.st->alpha. val is grm::Key (layer A: a key's value is the key) or grm::Tau (layer F: a
// function table's row value). prover/hiding/product_test.cuh includes this file after struct Eng;
// prover/hiding/gram_products.cuh defines the terms.
//
// A pass computes G[a][a'] = sum over indices c of eq(c) * L_a(c) * R_a'(c), where L_a and R_a'
// are eighths of the layer's two sides. From G the caller's k_g2msgT runs three sumcheck rounds.
// Entries are written as bytes (digits) to two matrices, one column per index, whose product gives
// 32-bit sums per piece of a chunk (2^lc indices). 2^lc must divide the number of indices: a
// remainder is skipped silently.
//
// Public source: see prover/hiding/gram_products.cuh.
struct KeyPasses {
  // Set by the caller before each pass: eq(c) = E.eqhi[c >> lc] * E.eqlo[c % 2^lc].
  Eng &E;
  const uint16_t *keys;
  size_t N;
  // Receives the sums per chunk.
  void *scA;
  T *G80, *K64;
  cudaStream_t stream;
  // Computes the sums of a layer of SC indices whose digit matrices have `wide` and `narrow`
  // rows. digits(c0, n, A, B, ld) writes the digits of n indices from index c0 on, which need not
  // begin a chunk; join(C, pieces, nc, c0) combines the sums of nc chunks from chunk c0 on.
  template <class D, class J> void lib_wide(int wide, int narrow, size_t SC, int lc, const D &digits, const J &join) {
    E.gram_slab_in(2); // pass 1: the calling code keeps nothing in E.S[0] and E.S[1]
    const size_t chunk = (size_t)1 << lc;
    const int nchunks = (int)(SC >> lc), k = grm::Slab::piece(chunk, SC), pieces = (int)(chunk / k);
    const size_t ld = E.gram_slab.columns(wide + narrow, k);
    const int per = (int)std::max<size_t>(1, E.gram_slab.sums_hold(wide * narrow) / pieces);
    if (!ld || E.gram_slab.sums_hold(wide * narrow) < (size_t)pieces) {
      printf("internal error: product test: the memory of a Gram pass is too small for one piece's digits or one "
             "chunk's sums\n");
      fflush(stdout);
      exit(1);
    }
    grm::i8 *A = E.gram_slab.A, *B = A + (size_t)wide * ld;
    for (int c0 = 0; c0 < nchunks; c0 += per) {
      const int nc = std::min(per, nchunks - c0);
      const size_t n = nc * chunk;
      for (size_t x = 0; x < n; x += ld) {
        const int some = (int)std::min(ld, n - x);
        digits(c0 * chunk + x, some, A, B, ld);
        grm::product(A, B, E.gram_slab.C + x / k * wide * narrow, wide, narrow, k, some / k, ld, k, stream);
      }
      join(E.gram_slab.C, pieces, nc, c0);
    }
  }
  // Pass 1 sums over val.value(key), not entries; the caller's k_finA80 computes G from its
  // 80 sums, G80.
  template <class V> void lib_a1(const V &val, int lc) {
    const size_t SC = N / 16;
    T *Gq = (T *)scA;
    lib_wide(
        grm::WIDE, grm::narrow<V>(), SC, lc,
        [&](size_t c0, int n, grm::i8 *A, grm::i8 *B, size_t ld) {
          const int grid = std::min((n + 127) / 128, g_sms * 8);
          grm::k_digits_1<V><<<grid, 256, 0, stream>>>(keys, SC, c0, n, 1 << lc, E.eqlo, A, B, ld, val);
          CK(cudaGetLastError());
        },
        [&](const int *C, int pieces, int nc, int c0) {
          grm::k_join_1<V><<<(nc * 80 + 3) / 4, 128, 0, stream>>>(C, pieces, nc, E.eqhi + c0, Gq + (size_t)c0 * 80);
          CK(cudaGetLastError());
        });
    t3k::addchunks<<<80, 128, 0, stream>>>(Gq, G80, (int)(SC >> lc), 80);
    CK(cudaGetLastError());
  }
  // Computes G into E.G for eighths of CO entries; Gq receives each chunk's 64 sums. digits(c0, n)
  // writes the digits of n indices from c0 on to E.gram_slab, which the caller placed by
  // E.gram_slab_in(). Not checked: that a chunk fits there.
  template <class D> void lib_gram(size_t CO, int lc, const D &digits, T *Gq) {
    const size_t chunk = (size_t)1 << lc, ld = E.gram_slab.cap;
    const int nchunks = (int)(CO >> lc), k = grm::Slab::piece(chunk, CO), pieces = (int)(chunk / k),
              per = E.gram_slab.chunks(chunk, k);
    for (int c0 = 0; c0 < nchunks; c0 += per) {
      const int nc = std::min(per, nchunks - c0), n = (int)(nc * chunk);
      digits(c0 * chunk, n);
      grm::product(E.gram_slab.A, E.gram_slab.B, E.gram_slab.C, grm::ROWS, grm::ROWS, k, nc * pieces, ld, k, stream);
      grm::join(E.gram_slab.C, pieces, nc, E.eqhi + c0, Gq + (size_t)c0 * 64, stream);
    }
    grm::k_sum<<<64, 128, 0, stream>>>(Gq, nchunks, E.G, nullptr, 0, 0, 0);
    CK(cudaGetLastError());
  }
  // Pass 2 reads E.k8, the eq weights of pass 1's three sumcheck challenges.
  template <class V> void lib_a2(const V &val, int lc) {
    E.gram_slab_in(2);
    const size_t ld = E.gram_slab.cap;
    kg3::k_prepeq<<<((1 << lc) + 255) / 256, 256, 0, stream>>>(E.eqlo, 1 << lc, E.eqprep);
    CK(cudaGetLastError());
    lib_gram(
        N / 128, lc,
        [&](size_t c0, int n) {
          grm::k_digits_2<V><<<std::min((n + 127) / 128, g_sms * 8), 256, 0, stream>>>(
              (const uint4 *)keys, N / 16, c0, n, 1 << lc, E.eqprep, E.k8, &E.st->alpha, E.gram_slab.A, E.gram_slab.B, ld,
              val);
          CK(cudaGetLastError());
        },
        (T *)scA);
  }
  // Pass 3 folds the entries into UL and UR with K64, the eq weights of the six earlier
  // challenges.
  template <class V> void lib_a3(const V &val, int lc, T *UL, T *UR) {
    E.gram_slab_in(0); // UL and UR are in E.S[1]
    const size_t ld = E.gram_slab.cap, CO = N / 1024;
    grm::k_fold64<V><<<(int)std::min<size_t>((N / 64 + 31) / 32, (size_t)g_sms * 8), 256, 0, stream>>>(
        (const uint4 *)keys, N / 128, CO, K64, &E.st->alpha, UL, UR, val);
    CK(cudaGetLastError());
    kg3::k_prepeq<<<((1 << lc) + 255) / 256, 256, 0, stream>>>(E.eqlo, 1 << lc, E.eqprep);
    CK(cudaGetLastError());
    lib_gram(
        CO, lc,
        [&](size_t c0, int n) {
          grm::k_digits<<<std::min((n + 127) / 128, g_sms * 8), 256, 0, stream>>>(UL, UR, CO, c0, n, 1 << lc, E.eqprep,
                                                                                  E.gram_slab.A, E.gram_slab.B, ld);
          CK(cudaGetLastError());
        },
        (T *)scA);
  }
  // tab8 is grm::Tau's byte table, or nullptr for grm::Key; PtPL::prove() passes nullptr.
  void a1(const u8 *tab8, int lc) { tab8 ? lib_a1(grm::Tau{tab8}, lc) : lib_a1(grm::Key{}, lc); }
  void a2(const u8 *tab8, int lc) { tab8 ? lib_a2(grm::Tau{tab8}, lc) : lib_a2(grm::Key{}, lc); }
  void a3(const u8 *tab8, int lc, T *UL, T *UR) {
    tab8 ? lib_a3(grm::Tau{tab8}, lc, UL, UR) : lib_a3(grm::Key{}, lc, UL, UR);
  }
};
