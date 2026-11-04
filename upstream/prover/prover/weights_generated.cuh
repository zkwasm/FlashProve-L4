// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Synthetic weights (`cp_commit -weights generated`): each is one value of a numbered stream of the
// seeded generator synth_weight() (forward.cuh), so none is stored. `cp_prove`'s GeneratedWeights writes
// committed cells (cell: one balanced base-256 limb of a weight) and token embeddings. The layers'
// forward pass generates its matrices through ForwardPass::wg() and must use the same streams and indexes.
#pragma once
#include "common/lie.cuh"
#include "prover/weights.h"

// Returns limb 0 of w, in [-128, 127], or limb 1, (w - limb 0) / 256; w must fit in two limbs.
__device__ __forceinline__ i8 balanced_limb(int w, int limb) {
  const int b0 = ((w + 128) & 255) - 128;
  return (i8)(limb ? (w - b0) >> 8 : b0);
}
// Maps cell r of one limb of a kind > 0 (kind: tensor type, weights.h) to an index in its stream.
// Mode 0: layers from first_layer on, per_layer cells each. Mode 1: every second layer. Mode 2,
// kind 9 (down) with experts: per_layer cells cover two layers, the dense one's dense_part first.
struct GeneratedKind {
  int kind = 0, mode = 0, first_layer = 0, d = 0, dh = 0, H = 0, G = 0, bits = 13;
  u64 stream = 0;
  size_t per_limb = 1, per_layer = 1, dense_part = 1;
};
// Writes `words` 32-bit words of 4 cells from cell `start`. Kind 0 is [layer][in][lane][slot < 64].
__global__ void k_generated_cells(u64 seed, GeneratedKind w, size_t start, size_t words, u32 *out) {
  GRID_LOOP(i4, words) {
    const size_t I = start + 4 * i4;
    const int limb = (int)(I / w.per_limb);
    const size_t r0 = I % w.per_limb;
    u32 pack = 0;
    u64 ix0 = 0;
    if (w.kind) {
      if (w.mode == 0)
        ix0 = (u64)w.first_layer * w.per_layer + r0;
      else if (w.mode == 1)
        ix0 = r0 + (u64)(w.first_layer + r0 / w.per_layer) * w.per_layer;
      else {
        const size_t a = r0 / w.per_layer, rem = r0 % w.per_layer;
        ix0 = rem < w.dense_part ? (u64)(w.first_layer + 2 * a) * w.dense_part + rem
                                 : (u64)(w.first_layer + 2 * a + 1) * w.dense_part + rem - w.dense_part;
      }
    }
    for (int b = 0; b < 4; b++) {
      int v;
      if (w.kind)
        v = synth_weight(seed, w.stream, ix0 + b, w.bits);
      else {
        const size_t r = r0 + b;
        const int slot = (int)(r % 64);
        size_t q = r / 64;
        const int lane = (int)(q % w.dh);
        q /= w.dh;
        const int in = (int)(q % w.d), layer = w.first_layer + (int)(q / w.d);
        v = slot < w.H ? synth_weight(seed, 2, ((u64)layer * w.d + in) * (w.H * w.dh) + slot * w.dh + lane, 13)
            : slot < w.H + w.G
                ? synth_weight(seed, 3, ((u64)layer * w.d + in) * (w.G * w.dh) + (slot - w.H) * w.dh + lane, 13)
                : 0;
      }
      pack |= (u32)(u8)balanced_limb(v, limb) << (8 * b);
    }
    out[i4] = pack;
  }
}
__device__ __forceinline__ u32 embedding_limbs_packed(const long long *v4, int limb) {
  u32 pack = 0;
  for (int b = 0; b < 4; b++) {
    long long v = v4[b];
    int byte = 0;
    for (int q = 0; q <= limb; q++) {
      byte = (int)(((v + 128) & 255) - 128);
      v = (v - byte) >> 8;
    }
    pack |= (u32)(u8)byte << (8 * b);
  }
  return pack;
}
__global__ void k_generated_embed(const int *tok, u64 seed, int T, int d, int xl, ResidualInt *x0, i8 *cells) {
  GRID_LOOP(x, (size_t)T * d) {
    const int t = (int)(x % T), k = (int)(x / T);
    const int v = lie::at(lie::STREAM, k, t, (int)synth_weight(seed, 13, (size_t)tok[t] * d + k, 20)); // lie::at(): a test hook, the identity in this repository (prover/common/lie.cuh)
    x0[x] = v;
    write_limbs(v, xl, cells + x, (size_t)d * T);
  }
}
__global__ void k_generated_embedding_cells(u64 seed, size_t entries, size_t start, size_t words, u32 *out) {
  GRID_LOOP(i4, words) {
    const size_t I = start + 4 * i4;
    long long v4[4];
    for (int b = 0; b < 4; b++) v4[b] = synth_weight(seed, 13, (I + b) % entries, 20);
    out[i4] = embedding_limbs_packed(v4, (int)(I / entries));
  }
}

// Kind 0: q, k. 1: v. 2: o. 3 to 8: gating, up (dense, shared, routed). 9, 10: down. 11: router.
struct GeneratedWeights : Weights {
  u64 seed = 0;
  int Lam = 1, d = 0, H = 0, G = 0, dh = 0, f = 0, fe = 0, NE = 0, V = 0;
  // Must equal LayerShape::cw (layout.h), used by ForwardPass::wg(); not checked. `cp_prove` sets neither. At 14,
  // weights of kind 1 to 11 have 16 bits, not 13 (synth_weight_bits(), forward.cuh); kind 0 keeps 13.
  int cw = 12;
  bool experts = false; // every second layer has experts

  bool planes(int, int, ForwardPass::WP &) const override { return false; }
  // Not checked: out, bytes / parts, bytes / 2 and each layer's number of cells are multiples of 4.
  void cells(int kind, int group, size_t bytes, int part, int parts, i8 *out) const override {
    GeneratedKind q;
    q.kind = kind;
    q.d = d;
    q.dh = dh;
    q.H = H;
    q.G = G;
    q.per_limb = bytes / 2;
    const int l0 = group * Lam;
    q.first_layer = l0;
    const size_t d_ = d, f_ = f, fe_ = fe, NE_ = NE, dq = (size_t)H * dh, dkv = (size_t)G * dh;
    auto second_layers = [&](u64 stream, size_t per_layer) {
      q.stream = stream;
      q.per_layer = per_layer;
      q.mode = 1;
      q.first_layer = l0 + 1;
    };
    if (kind == 1) {
      q.stream = 4;
      q.per_layer = d_ * dkv;
    } else if (kind == 2) {
      q.stream = 5;
      q.per_layer = dq * d_;
    } else if (kind == 3 || kind == 4) {
      q.stream = kind == 3 ? 6 : 7;
      q.per_layer = d_ * f_;
      q.mode = experts ? 1 : 0;
    } else if (kind == 5 || kind == 6) {
      second_layers(kind == 5 ? 6 : 7, d_ * fe_);
    } else if (kind == 7 || kind == 8) {
      second_layers(kind == 7 ? 10 : 11, NE_ * d_ * fe_);
    } else if (kind == 9) {
      q.stream = 8;
      if (experts) {
        q.mode = 2;
        q.per_layer = (f_ + fe_) * d_;
        q.dense_part = f_ * d_;
      } else
        q.per_layer = f_ * d_;
    } else if (kind == 10) {
      second_layers(12, NE_ * fe_ * d_);
    } else if (kind == 11) {
      second_layers(9, d_ * NE_);
    }
    q.bits = synth_weight_bits(q.stream, cw);
    const size_t n = bytes / parts;
    k_generated_cells<<<grid_blocks(n / 4), 256>>>(seed, q, (size_t)part * n, n / 4, (u32 *)out);
  }
  void embed(const int *tokens, int T, int xl, ResidualInt *x0, i8 *x0_cells) const override {
    k_generated_embed<<<grid_blocks((size_t)T * d), 256>>>(tokens, seed, T, d, xl, x0, x0_cells);
  }
  void embedding_cells(size_t start, size_t words, u32 *out) const override {
    k_generated_embedding_cells<<<grid_blocks(words), 256>>>(seed, (size_t)V * d, start, words, out);
  }
  const int16_t *head() const override { return nullptr; }
};
