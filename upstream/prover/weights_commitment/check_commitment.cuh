// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The range scan of the integers that the committer program cp_commit commits to. cp_commit runs it
// before it draws or reads its seed. If an integer is outside the range of its kind (limit_of()),
// cp_commit exits with an error and writes no commitment, so that no such commitment is published
// by mistake. No proof shows that the integers are those of a given model: the verifier takes that
// as given. The scan reads cells of the committed matrix W, which check_the_weights() in
// prover/hash_commitment/hf_commit.cu computes, not the source of the weights. So a unit's padding
// cells are scanned too. An honest committer sets these to 0; the scan does not test that. A cell
// is a signed byte, a digit of an integer. A unit is W's rows for one kind of weight in one layer.
#pragma once
namespace rf {
namespace check_commitment {
// The width in bits of a committed integer of kind X (12: the embedding), a constant of the proof
// format. The verifier program cp_verify rejects a parameter file that gives a kind another width
// (`policy-weight-kind`, `policy-weight-width-emb`). Its no-overflow conditions assume width 13.
static u32 format_width(int X) {
  return X == 12 ? 21u : 13u;
}
// The greatest absolute value that the scan accepts for an integer of kind X. The declared limit is
// 2^(wb - 1) - 1, where wb is the kind's width line in the parameter file (`policy.txt`; `policy`
// in identifiers and messages). Exits with code 2 if wb is missing or is not format_width(X), with
// code 4 if X is not 0 to 13. For every kind but the embedding, cp_verify's no-overflow conditions
// and the range proof (blocks/frame/wf_units.h) in the key file, `ivk/key.bin`, also allow
// -2^(wb - 1). For the embedding the conditions need only that each of an integer's four digits is
// small, which every signed byte is. If as_declared is false, the embedding's limit is 2^30 - 1.
static long long limit_of(const PolC &policy, int X, bool as_declared) {
  if (X == 12 && !as_declared) return (1ll << 30) - 1;
  static const char *const word[14] = {"wb_qk", "wb_v",  "wb_o",  "wb_g",   "wb_u", "wb_gs",  "wb_us",
                                       "wb_gr", "wb_ur", "wb_dn", "wb_dnr", "wb_r", "wb_emb", "wb_head"};
  if (X < 0 || X >= 14) {
    fprintf(stderr, "cp_commit: internal error: a unit of kind %d has no width line\n", X);
    exit(4);
  }
  const u32 bits = policy[word[X]];
  if (bits != format_width(X)) {
    fprintf(stderr,
            "cp_commit: refused (policy-weight-width): the policy's %s is %u, but in this proof format an integer "
            "of that kind has %u bits, and the verifier accepts no other width\n",
            word[X], bits, format_width(X));
    exit(2);
  }
  return (1ll << (bits - 1)) - 1;
}
// cp_commit calls this before it loads the weights, so that a wrong width ends the run early. Exits
// with code 2 if limit_of() rejects the width line of a kind in the layout (`LayW`,
// prover/common/weight_layout.h), or if the parameter file has a line `cw`.
static void hold_the_widths(const PolC &policy, const LayW &layout) {
  for (const UnitW &u : layout.U) (void)limit_of(policy, u.X, true);
  if (policy.m.count("cw")) {
    fprintf(stderr, "cp_commit: refused (policy-weight-width): the parameter file has a line cw, which this proof "
                    "format does not have: remove it\n");
    exit(2);
  }
}
// found[0]: the number of integers outside -limit .. limit. found[1]: the greatest absolute value.
// found[2]: the least index i outside. The caller sets `found` to {0, 0, ~0} before the launch.
// `cells`: consecutive rows of W, as k columns of `pitch` bytes. The unit begins at row `first` of
// `cells` and has `rows` rows for each digit. Digit l of integer i is in column i / rows, row
// first + l * rows + i % rows. Integer i is the sum over l of digit l * 256^l (balanced base 256).
// Public source: an encoder that checks its output: Xiph.Org, FLAC API documentation (libFLAC),
// module 'FLAC/stream_encoder.h: stream encoder interface', FLAC__stream_encoder_set_verify(). Here
// each integer is tested against its range instead of against the source.
// Public source: validate after transforming: MITRE, Common Weakness Enumeration 4.20, CWE-179
// 'Incorrect Behavior Order: Early Validation', 'Extended Description' (about untrusted input).
// Privatization of a maximum. The idea is public: private copies, merged at the end: M. Bertini,
// "Parallel Computing", slides "CUDA: atomic operations, privatization, algorithms", slide
// "Privatization", https://www.micc.unifi.it/bertini/download/parallel/2016-2017/14_gpu_cuda_5.pdf.
// The idea is public: a scalar per thread, one atomic update: T. Mattson, "A 'Hands-on'
// Introduction to OpenMP", slide "Example: Using an atomic to remove impact of false sharing",
// https://www.openmp.org/wp-content/uploads/Intro_To_OpenMP_Mattson.pdf. Their examples are a
// histogram and a sum.
// The loop form is public: the grid-stride loop: M. Harris,
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_scan(const i8 *cells, size_t pitch, size_t first, size_t rows, int digits, size_t k, long long limit,
                       unsigned long long *found) {
  unsigned long long greatest = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < rows * k; i += (size_t)gridDim.x * blockDim.x) {
    const i8 *digit = cells + (i / rows) * pitch + first + i % rows;
    long long w = 0;
    for (int l = digits - 1; l >= 0; l--) w = w * 256 + digit[(size_t)l * rows];
    const unsigned long long a = w < 0 ? -w : w;
    greatest = a > greatest ? a : greatest;
    if (a > (unsigned long long)limit) {
      atomicAdd(found, 1ull);
      atomicMin(found + 2, (unsigned long long)i);
    }
  }
  atomicMax(found + 1, greatest);
}
struct Scan {
  unsigned long long *found = nullptr; // three words on the GPU: see k_scan
  size_t places = 0, outside = 0, units_outside = 0;
  bool as_declared = true;      // true: the embedding too is compared with its width line. cp_commit sets it to false
  // cp_commit sets `say` to true only for synthetic weights under a test seed.
  bool say = true;              // false: no count, position or size is printed
  bool embedding_wider = false; // the embedding exceeds the range of its width line
  void begin() { CK(cudaMalloc(&found, 24)); }
  // Scans unit u (`index` in the layout). A failed launch finds nothing: callers must check for it.
  void unit(const PolC &policy, const UnitW &u, size_t index, const i8 *cells, size_t pitch, size_t first, size_t k) {
    const unsigned long long start[3] = {0, 0, ~0ull};
    unsigned long long got[3];
    const size_t rows = u.rows / u.nlw;
    const long long limit = limit_of(policy, u.X, as_declared);
    CK(cudaMemcpyAsync(found, start, 24, cudaMemcpyHostToDevice, cudaStreamPerThread));
    k_scan<<<4096, 256, 0, cudaStreamPerThread>>>(cells, pitch, first, rows, u.nlw, k, limit, found);
    CK(cudaMemcpyAsync(got, found, 24, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    places += rows * k;
    if (u.X == 12) embedding_wider = got[1] > (unsigned long long)limit_of(policy, 12, true);
    if (!got[0]) return;
    outside += got[0];
    if (units_outside++ < 8 && say) // at most eight lines, even if every unit of a model is out of range
      printf("cp_commit: unit %zu (kind %d, layer %d): %llu integers lie outside -%lld .. %lld; the greatest "
             "absolute value is %llu; the first is in row %llu, column %llu of W\n",
             index, u.X, u.l, got[0], limit, limit, got[1], (unsigned long long)(u.off + got[2] % rows),
             (unsigned long long)(got[2] / rows));
  }
};
} // namespace check_commitment
} // namespace rf
