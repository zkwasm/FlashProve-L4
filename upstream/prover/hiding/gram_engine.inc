// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Members of struct Eng (prover/hiding/product_test.cuh) for Gram passes. A Gram pass runs three
// sumcheck rounds of a lookup argument's product layer from G[a][b] = sum_c eq(c) L_a(c) R_b(c),
// where L_a, R_b are eighths (slabs) of its sides L, R; prover/hiding/gram_products.cuh defines the
// terms. grm::Slab is not a slab but the memory of one pass's digits (the entries' signed bytes)
// and sums. Both derive from hidden values. Eng does not zero them: the code that installs the
// allocator `pt_alloc` must. cp_prove takes them from GPU memory that it zeroes before each
// request and after each batch pass (prover/prover/memory_as_set_up.cuh), and in its exit handler.
// Public source: see prover/hiding/gram_products.cuh.
grm::Slab gram_slab; // the memory of this Eng's current pass: see `gram_slab_in`
// `big`: entries in one side of the largest stored layer (in GPU memory). `room`
// (Eng::init() passes sc) must have gram_slab_bytes(big) bytes and be 256-byte aligned, as
// grm::product() assumes (not checked).
static size_t gram_slab_bytes(size_t big) {
  return grm::Slab::bytes(grm::Slab::cap_for(big));
}
void gram_slab_lay(void *room, size_t big) {
  gram_slab.lay(room, grm::Slab::cap_for(big));
}
// Places the next pass's digit matrices and sums in the work buffer sc. `free`: which of the
// buffers S[0] and S[1] that pass may overwrite: 0, 1, or 2 for both, which are adjacent. If
// matrices twice as wide as in sc fit there, and their sums in sc, the matrices go there instead,
// with as many columns (a power of 2) as both allow, for fewer and larger launches. A pass must
// itself write what it reads in either memory: PT_FREE_MEMORY() lets a test overwrite both first.
// Public source for reuse of memory no longer needed: T. Chen, B. Xu, C. Zhang, C. Guestrin,
// "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174, section 3 (memory sharing).
// Public source for larger launches: NVIDIA, "GPU Performance Background User's Guide".
void gram_slab_in(int free) {
  T *room = S[free == 1];
  const size_t bytes = free == 2 ? 2 * s_bytes : s_bytes;
  PT_FREE_MEMORY(room, bytes, false);
  PT_FREE_MEMORY(sc, scb, true);
  size_t cap = own_cap;
  while (4 * grm::ROWS * cap <= bytes && grm::Slab::pieces_for(2 * cap) * grm::SUMS * sizeof(int) <= scb) cap *= 2;
  if (cap == own_cap) {
    gram_slab.lay(sc, own_cap);
    return;
  }
  gram_slab.cap = cap;
  gram_slab.A = reinterpret_cast<grm::i8 *>(room);
  gram_slab.B = gram_slab.A + grm::ROWS * cap;
  gram_slab.C = static_cast<int *>(sc);
}
// Runs one Gram pass of a stored layer: rounds j to j + 2. Call gram_slab_in() first; the sumcheck
// must have begun (k_layer_begin or k_leaves_begin). L, R: the sides, 8 slabs of SC indices each.
// `chunk` must divide SC, be a multiple of 4 and at most 2^15, and Gcb must hold 64 SC / chunk
// elements (not checked). eq(c) = lo[c % chunk] * hi[c / chunk]. Leaves the rounds' challenges in
// cd->r and eq(challenges, a) in k8[a]. `o`: ignored. dka, dlk, da: see grm::k_sum.
void lib_pass(const T *L, const T *R, size_t SC, size_t chunk, const M3L *lo, const T *hi, Chal<F> *cd, int j, T *o,
              int dka, int dlk, int da, cudaStream_t stream) {
  const int nchunks = (int)(SC / chunk), k = grm::Slab::piece(chunk, SC), pieces = (int)(chunk / k),
            per = gram_slab.chunks(chunk, k);
  const size_t ld = gram_slab.cap;
  for (int c0 = 0; c0 < nchunks; c0 += per) {
    const int nc = std::min(per, nchunks - c0), n = (int)(nc * chunk);
    grm::k_digits<<<std::min((n + 127) / 128, g_sms * 8), 256, 0, stream>>>(L, R, SC, c0 * chunk, n, (int)chunk, lo,
                                                                            gram_slab.A, gram_slab.B, ld);
    CK(cudaGetLastError());
    grm::product(gram_slab.A, gram_slab.B, gram_slab.C, grm::ROWS, grm::ROWS, k, nc * pieces, ld, k, stream);
    grm::join(gram_slab.C, pieces, nc, hi + c0, Gcb + (size_t)c0 * 64, stream);
  }
  pt_hi(grm::k_sum, 64, 128, 0, stream, Gcb, nchunks, G, cd, dka, dlk, da);
  pt_hi(k_g2msgT, 1, 64, 0, stream, G, st, cd, j, j + 1, j + 2, o, k8);
}
