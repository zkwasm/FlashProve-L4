// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Implements BlockWeights: fill() stores a model on the GPU as quantized blocks, planes of limbs
// (signed bytes) and committed cells. An int `stream` or `st` is a tensor number (block_weights.h).
#pragma once
#include "common/end_name.h"
#include "common/index_guard.cuh"
#include "common/lie.cuh"
#include "weights_blocks/block_weights.h"
#include "weights_blocks/decode.cuh"
#include "weights_blocks/fold_setup.cuh"
#include "weights_blocks/row_pass_blocks.cuh"
#include <cstring>

namespace wb {

// Writes q's and k's cells (kind 0) of layer `a` of a layer group as [input][lane of a head][slot].
// The 64 head slots: q's H heads, k's G heads, then zeros. per_limb: one limb's cells in the group.
__global__ void k_qk_cells(const int16_t *q, const int16_t *k, int a, int d, int dh, int H, int G, size_t per_limb,
                           i8 *cells) {
  GRID_LOOP(i, (size_t)d * dh * 64) {
    const int slot = (int)(i % 64);
    const size_t r = i / 64;
    const int lane = (int)(r % dh), in = (int)(r / dh);
    const int v = slot < H       ? (int)q[(size_t)(slot * dh + lane) * d + in]
                  : slot < H + G ? (int)k[(size_t)((slot - H) * dh + lane) * d + in]
                                 : 0;
    write_limbs(v, 2, cells + (size_t)a * d * dh * 64 + i, per_limb);
  }
}
// Writes the embedding's cells [limb < 4][token][k] from the file's bytes, with no gain (a norm's
// multiplier per input column). Sets *over if an entry is not finite or reaches 2^30 in magnitude.
__global__ void k_embedding_cells(int type, const uint8_t *bytes, size_t V, int d, int e, i8 *cells, unsigned *over) {
  const int block = fr::block_bytes(type);
  GRID_LOOP(i, V * d) {
    const int w = fr::weight(type, bytes + (i >> 8) * block, (int)(i & 255), 1.f, e);
    if (w >= 1 << 30 || w <= -(1 << 30)) atomicOr(over, 1u);
    write_limbs(w, 4, cells + i, V * d);
  }
}
// Writes x0[k][t] = Emb[tokens[t]][k] and its limbs. An invalid token counts as 0 and is flagged.
__global__ void k_embed_from_cells(const int *tokens, const i8 *cells, size_t entries, int T, int d, int xl, ResidualInt *x0,
                                   i8 *x0_cells) {
  GRID_LOOP(x, (size_t)T * d) {
    const int t = (int)(x % T), k = (int)(x / T);
    const size_t at = (size_t)index_below(tokens[t], (unsigned)(entries / d), INDEX_TOKEN) * d + k;
    int v = 0;
    for (int limb = 3; limb >= 0; limb--) v = v * 256 + (int)cells[(size_t)limb * entries + at];
    v = lie::at(lie::STREAM, k, t, v); // a test hook (common/lie.cuh): the identity in this repository
    x0[x] = v;
    write_limbs(v, xl, x0_cells + x, (size_t)d * T);
  }
}

inline void *Placement::place(size_t bytes, int, Form) {
  void *p = nullptr;
  CK(fork_alloc(&p, bytes, Place::Own));
  return p;
}

// Rows of a stream's tensor in a layer; for the routed streams 10 to 12, the rows of one expert.
inline size_t Shape::rows(int stream, int layer) const {
  const int ff = has_experts(layer) ? fe : f;
  switch (stream) {
  case 2:
    return (size_t)H * dh;
  case 3:
  case 4:
    return (size_t)G * dh;
  case 6:
  case 7:
    return ff;
  case 9:
    return NE;
  case 10:
  case 11:
    return fe;
  case 13:
  case 14:
    return V;
  default: // streams 5, 8, 12
    return d;
  }
}
inline int Shape::inputs(int stream, int layer) const {
  return stream == 5 ? H * dh : stream == 8 ? (has_experts(layer) ? fe : f) : stream == 12 ? fe : d;
}

// A layer group's cells form one array per `kind`. Returns its stream; kind 0 (streams 2 and 3): 2.
static inline int stream_of_kind(int kind) {
  static const int stream[12] = {2, 4, 5, 6, 7, 6, 7, 10, 11, 8, 12, 9};
  return stream[kind];
}
// Whether a layer has a matrix of this kind.
static inline bool kind_in_layer(const Shape &s, int kind, int layer) {
  if (kind == 3 || kind == 4) return !s.has_experts(layer);
  if ((kind >= 5 && kind <= 8) || kind == 10 || kind == 11) return s.has_experts(layer);
  return true;
}

inline bool BlockWeights::all_blocks(int kind, int group) const {
  if (kind == 0 || kind == 11) return false;
  for (int l = group * shape.Lam; l < (group + 1) * shape.Lam; l++)
    if (kind_in_layer(shape, kind, l) && !blocks(stream_of_kind(kind), l)) return false;
  return true;
}

inline size_t BlockWeights::cells_per_limb(int kind, int group) const {
  const int st = stream_of_kind(kind);
  size_t n = 0;
  for (int l = group * shape.Lam; l < (group + 1) * shape.Lam; l++)
    if (kind_in_layer(shape, kind, l))
      n += kind == 0 ? (size_t)shape.d * shape.dh * 64
                     : shape.rows(st, l) * shape.inputs(st, l) * (st >= 10 ? shape.NE : 1);
  return n;
}

// Whether two device arrays of n floats have the same bits. Null equals only null.
static inline bool same_floats(const float *a, const float *b, size_t n) {
  if (a == b || !a || !b) return a == b;
  std::vector<float> x(n), y(n);
  CK(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost));
  return !memcmp(x.data(), y.data(), n * 4);
}

// Sets bit 1 (value 2) of *over if a gain is neither zero nor of magnitude 2^-64 to 2^64.
__global__ void k_gain_range(const float *gain, int n, unsigned *over) {
  GRID_LOOP(j, n) {
    const float g = fabsf(gain[j]);
    if (g != 0.f && !(g >= 0x1p-64f && g <= 0x1p64f)) atomicOr(over, 2u);
  }
}

// Reads each tensor and checks its shape, type and range. Stores a layer's tensor as blocks or, if
// stays_blocks() is false, as planes. Stores each kind's cells, but a kind other than 0 and 11 gets
// none in a layer group where all of its tensors are blocks, and the routed tensors (streams 10 to
// 12) never get any. The embedding is stored as cells only, the output head as 16-bit integers.
inline void BlockWeights::fill(BlockSource &source, Placement &where, cudaStream_t stream) {
  const Shape &s = shape;
  auto stop = [](const std::string &what, int st, int l) {
    static const char *names[15] = {"",    "",     "q",      "k",           "v",         "o",           "gate",
                                    "up",  "down", "router", "routed gate", "routed up", "routed down", "embedding",
                                    "head"};
    printf("weights: %s%s: %s\n", names[st], st < 13 ? (" of layer " + std::to_string(l)).c_str() : "", what.c_str());
    name_the_end("cp-weights-files"); // the end's name for the last line, as in GgufSource::stop()
    exit(3);
  };
  auto read = [&](int st, int l, uint8_t *into, size_t room, uint8_t *keep) {
    if (between_tensors) between_tensors();
    if (s.inputs(st, l) % 256) stop("its number of input columns must be a multiple of 256", st, l);
    // The source gets the CUDA stream that routed_filled(), group_filled() and end_filled() get. A
    // placement that reuses a layer's memory for another layer's tensors relies on stream order.
    const FileTensor t = source.tensor(st, l, into, room, stream, keep);
    const size_t rows = s.rows(st, l) * (st >= 10 && st <= 12 ? s.NE : 1);
    if (!t.bytes) stop("the model has no such tensor", st, l);
    if (t.rows != rows || t.inputs != s.inputs(st, l))
      stop("the model has " + std::to_string(t.rows) + " rows of " + std::to_string(t.inputs) +
               " inputs, but the parameter file's shape requires " + std::to_string(rows) + " rows of " +
               std::to_string(s.inputs(st, l)),
           st, l);
    if (t.type != source.type(st, l))
      stop("internal error: the weight source reported two different types for it", st, l);
    if (t.type < Q4_K || t.type > BF16) stop("its type is not one of Q4_K, Q5_K, Q6_K, F32, Q8_0, F16 and BF16", st, l);
    return t;
  };
  if (s.layers % s.Lam) printf("weights: %d layers cannot be divided into layer groups of %d\n", s.layers, s.Lam);
  if (s.experts && s.Lam % 2)
    printf("weights: in a model with experts a layer group needs an even number of layers, "
           "not %d\n",
           s.Lam);
  if (s.H + s.G > 64) printf("weights: q and k have %d heads together; at most 64 are supported\n", s.H + s.G);
  if (s.layers % s.Lam || (s.experts && s.Lam % 2) || s.H + s.G > 64) {
    name_the_end("cp-weights-files");
    exit(3);
  }
  auto exponent = [&](int st, int l) {
    const float scale = source.scale(st, l);
    const int e = scale >= 1.f && scale <= 0x1p40f ? ilogbf(scale) : -1;
    if (e < 0 || ldexpf(1.f, e) != scale) stop("its exponent must be a whole number from 0 to 40", st, l);
    return e;
  };
  int16_t *q16 = nullptr, *k16 = nullptr;
  unsigned *over = nullptr;
  CK(fork_alloc(&q16, (size_t)s.H * s.dh * s.d * 2, Place::Own));
  CK(fork_alloc(&k16, (size_t)s.G * s.dh * s.d * 2, Place::Own));
  const size_t flags = (size_t)16 * s.layers; // one word per (layer, stream); embedding and head: layer 0
  CK(fork_alloc(&over, flags * 4, Place::Own));
  CK(cudaMemsetAsync(over, 0, flags * 4, stream));

  const std::vector<int> kinds =
      s.experts ? std::vector<int>{0, 1, 2, 3, 4, 5, 6, 9, 11} : std::vector<int>{0, 1, 2, 3, 4, 9};
  // Whether a tensor of streams 2 to 12 stays as the file's blocks.
  // Public source for the choice between the forms: see block_weights.h, Form.
  auto stays_blocks = [&](int st, int l) {
    return st >= 10 || (st != 9 && source.as_blocks(st, l));
  };
  // Announces to the placement every place(), place_routed() and place_end() call below, in order,
  // before the first. Sizes follow from the shape and the tensors' types only: no tensor is read.
  // Public sources for announcing beforehand: see prover/prover/weight_stream.cuh, will_place().
  for (int g = 0; g < s.layers / s.Lam; g++) {
    const int l0 = g * s.Lam, l1 = l0 + s.Lam;
    for (int l = l0; l < l1; l++)
      for (int st = 2; st <= (s.has_experts(l) ? 12 : 8); st++) {
        if (!stays_blocks(st, l)) continue;
        Blocks b;
        b.type = source.type(st, l), b.rows = s.rows(st, l), b.inputs = s.inputs(st, l);
        if (st >= 10)
          where.will_place(b.expert_bytes() * s.NE, g, Form::RoutedBlocks, l);
        else
          where.will_place(b.expert_bytes(), g, Form::Blocks, -1);
      }
    for (int l = l0; l < l1; l++)
      for (int st = 2; st <= (s.has_experts(l) ? 9 : 8); st++)
        if (!stays_blocks(st, l)) where.will_place(2 * s.rows(st, l) * s.inputs(st, l), g, Form::Planes, -1);
    for (int kind : kinds) {
      bool all = kind != 0 && kind != 11; // what `all_blocks` returns once the blocks are placed
      for (int l = l0; l < l1; l++) all = all && (!kind_in_layer(s, kind, l) || stays_blocks(stream_of_kind(kind), l));
      if (!all) where.will_place(2 * cells_per_limb(kind, g), g, Form::Cells, -1);
    }
  }
  where.will_place((size_t)4 * s.V * s.d, -1, Form::Cells, -1);
  where.will_place((size_t)s.V * s.d * 2, -1, Form::Planes, -1);

  for (int g = 0; g < s.layers / s.Lam; g++) {
    const int l0 = g * s.Lam, l1 = l0 + s.Lam;
    for (int l = l0; l < l1; l++)
      for (int st = 2; st <= (s.has_experts(l) ? 12 : 8); st++) {
        if (!stays_blocks(st, l)) continue;
        Blocks b;
        b.type = source.type(st, l);
        if (!decoded_from_blocks(b.type) || (b.type == Q5_K && (st == 10 || st == 11)))
          stop("it is stored as blocks, so its type must be Q4_K, Q8_0, F16 or BF16, or Q5_K unless it is routed gate "
               "or routed up",
               st, l);
        b.rows = s.rows(st, l), b.inputs = s.inputs(st, l);
        if (b.rows % 4) stop("it is stored as blocks, so its number of rows must be a multiple of 4", st, l);
        b.gain = source.gain(st, l), b.scale = source.scale(st, l);
        b.bytes = (const uint8_t *)(st >= 10 ? where.place_routed(b.expert_bytes() * s.NE, g, l)
                                             : where.place(b.expert_bytes(), g, Form::Blocks));
        blocks_[{st, l}] = b;
      }
    // A layer's tensors of streams 10 and 11 read the gains of the same norm. If the source gives
    // them different gains, the program exits with code 3.
    for (int l = l0; l < l1; l++)
      if (s.has_experts(l) && !same_floats(blocks_[{10, l}].gain, blocks_[{11, l}].gain, s.d))
        stop("it must read the same gains as routed gate", 11, l);
    for (int l = l0; l < l1; l++)
      for (int st = 2; st <= (s.has_experts(l) ? 9 : 8); st++) {
        if (blocks(st, l)) continue;
        Planes p;
        p.rows = s.rows(st, l), p.inputs = s.inputs(st, l);
        i8 *w = (i8 *)where.place(2 * p.rows * p.inputs, g, Form::Planes);
        p.w0 = w, p.w1 = w + p.rows * p.inputs;
        planes_[{st, l}] = p;
      }
    // Public source for keeping the cells as a second copy of weights that are also stored as
    // planes or blocks: see block_weights.h, cells_in_place().
    for (int kind : kinds)
      if (!all_blocks(kind, g)) cells_[{kind, g}] = (i8 *)where.place(2 * cells_per_limb(kind, g), g, Form::Cells);

    std::map<int, size_t> done; // per kind: cells of one limb written so far
    for (int l = l0; l < l1; l++) {
      for (int st = 2; st <= (s.has_experts(l) ? 12 : 8); st++) {
        const Blocks *b = blocks(st, l);
        const size_t kept = b ? b->expert_bytes() * (st >= 10 ? s.NE : 1) : 0;
        // Blocks are the file's own bytes. `twin` is the placement's host copy of them, or null if
        // it keeps none or the source cannot fill one. The source writes them there and to the GPU.
        uint8_t *twin = b && source.keeps_on_host() ? where.host_twin(b->bytes, kept) : nullptr;
        const FileTensor t = b ? read(st, l, (uint8_t *)b->bytes, kept, twin) : read(st, l, nullptr, 0, nullptr);
        const float *gain = source.gain(st, l);
        const int e = exponent(st, l);
        if (b && e > 24) stop("it is stored as blocks, so its exponent must be at most 24", st, l);
        unsigned *flag = over + 16 * l + st;
        fold_range(t.type, t.bytes, t.rows, t.inputs, gain, e, flag, stream);
        if (gain) k_gain_range<<<grid_blocks(t.inputs), 256, 0, stream>>>(gain, t.inputs, flag);
        if (st >= 10) continue;
        if (!b) {
          const Planes &p = planes_[{st, l}];
          fold_planes(t.type, t.bytes, t.rows, t.inputs, gain, e, (i8 *)p.w0, (i8 *)p.w1, t.inputs, stream);
        }
        if (st == 2 || st == 3) {
          fold_i16(t.type, t.bytes, t.rows, t.inputs, gain, e, st == 2 ? q16 : k16, flag, stream);
          if (st == 3)
            k_qk_cells<<<grid_blocks((size_t)s.d * s.dh * 64), 256, 0, stream>>>(q16, k16, l - l0, s.d, s.dh, s.H, s.G,
                                                                          cells_per_limb(0, g), cells_[{0, g}]);
          continue;
        }
        const int kind = st == 4   ? 1
                         : st == 5 ? 2
                         : st == 6 ? (s.has_experts(l) ? 5 : 3)
                         : st == 7 ? (s.has_experts(l) ? 6 : 4)
                         : st == 8 ? 9
                                   : 11;
        if (all_blocks(kind, g)) continue;
        i8 *lo = cells_[{kind, g}] + done[kind];
        fold_cells(t.type, t.bytes, t.rows, t.inputs, gain, e, lo, lo + cells_per_limb(kind, g), t.rows, stream);
        done[kind] += t.rows * t.inputs;
      }
      if (s.has_experts(l)) where.routed_filled(l, stream);
    }
    where.group_filled(g, stream);
  }

  embedding_ = (i8 *)where.place_end((size_t)4 * s.V * s.d, 0);
  {
    const FileTensor t = read(13, 0, nullptr, 0, nullptr);
    k_embedding_cells<<<grid_blocks((size_t)s.V * s.d), 256, 0, stream>>>(t.type, t.bytes, (size_t)s.V, s.d, exponent(13, 0),
                                                                   embedding_, over + 13);
  }
  where.end_filled(0, stream);
  head_ = (int16_t *)where.place_end((size_t)s.V * s.d * 2, 1);
  {
    const FileTensor t = read(14, 0, nullptr, 0, nullptr);
    const float *gain = source.gain(14, 0);
    fold_i16(t.type, t.bytes, t.rows, t.inputs, gain, exponent(14, 0), head_, over + 14, stream);
    if (gain) k_gain_range<<<grid_blocks(t.inputs), 256, 0, stream>>>(gain, t.inputs, over + 14);
  }
  where.end_filled(1, stream);
  where.group_filled(-1, stream);

  std::vector<unsigned> found(flags);
  CK(cudaGetLastError());
  CK(cudaMemcpyAsync(found.data(), over, flags * 4, cudaMemcpyDeviceToHost, stream));
  CK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < flags; i++) {
    if (found[i] & 2)
      stop("one of its multipliers (a gain of its norm times the constant c_0) is neither zero nor of "
           "magnitude 2^-64 to 2^64",
           i % 16, i / 16);
    if (found[i])
      stop("a weight is not finite or lies outside its range (at most 4095 in magnitude; in the embedding 2^30 - 1); "
           "its exponent may be too large",
           i % 16, i / 16);
  }
  CK(fork_free(q16));
  CK(fork_free(k16));
  CK(fork_free(over));
  source.all_read();
}

inline const Blocks *BlockWeights::blocks(int stream, int layer) const {
  auto it = blocks_.find({stream, layer});
  return it == blocks_.end() ? nullptr : &it->second;
}
inline const Planes *BlockWeights::tensor_planes(int stream, int layer) const {
  auto it = planes_.find({stream, layer});
  return it == planes_.end() ? nullptr : &it->second;
}
inline const i8 *BlockWeights::cells_in_place(int kind, int group) const {
  auto it = cells_.find({kind, group});
  return it == cells_.end() ? nullptr : it->second;
}

// On cudaStreamPerThread, writes part `part` of `parts` equal parts of the cells of (kind, group)
// to `out`. `bytes` must be the number of all these cells (cells_number() tests it) and a multiple
// of `parts`; neither is checked here. Stored cells (cells_in_place()) are copied. Otherwise all
// cells are decoded from the blocks: then `parts` must be 1 and the kind not routed (7, 8 or 10,
// whose cells are never written), or the program prints an internal error and exits with code 2.
inline void BlockWeights::cells(int kind, int group, size_t bytes, int part, int parts, i8 *out) const {
  const size_t n = bytes / parts;
  if (const i8 *c = cells_in_place(kind, group)) {
    CK(cudaMemcpyAsync(out, c + (size_t)part * n, n, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    return;
  }
  const int st = stream_of_kind(kind);
  if (st >= 10) {
    printf("internal error: the cells of routed kind %d were requested, but the cells of a routed kind are never "
           "written\n",
           kind);
    exit(2);
  }
  if (parts != 1) {
    printf("internal error: kind %d is stored as blocks, so its cells can be written in one part only\n", kind);
    exit(2);
  }
  i8 *lo = out;
  for (int l = group * shape.Lam; l < (group + 1) * shape.Lam; l++) {
    if (!kind_in_layer(shape, kind, l)) continue;
    const Blocks &b = *blocks(st, l);
    blocks_to_cells(b.bytes, (int)b.rows, b.inputs / 256, lo, lo + bytes / 2, b.rows, cudaStreamPerThread, b.gain,
                    b.scale, b.type);
    lo += b.rows * b.inputs;
  }
}
inline void BlockWeights::embed(const int *tokens, int T, int xl, ResidualInt *x0, i8 *x0_cells) const {
  k_embed_from_cells<<<grid_blocks((size_t)T * shape.d), 256>>>(tokens, embedding_, (size_t)shape.V * shape.d, T, shape.d, xl,
                                                         x0, x0_cells);
}
inline void BlockWeights::embedding_cells(size_t start, size_t words, u32 *out) const {
  CK(cudaMemcpyAsync(out, embedding_ + start, words * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}

inline void BlockWeights::write_planes(int stream, int layer, int expert, i8 *w0, i8 *w1, cudaStream_t s) const {
  if (const Blocks *b = blocks(stream, layer)) {
    const uint8_t *bytes = b->bytes + expert * b->expert_bytes();
    blocks_to_planes(bytes, (int)b->rows, b->inputs / 256, w0, w1, b->inputs, s, b->gain, b->scale, b->type);
    return;
  }
  const Planes &p = *tensor_planes(stream, layer);
  CK(cudaMemcpyAsync(w0, p.w0, p.rows * p.inputs, cudaMemcpyDeviceToDevice, s));
  CK(cudaMemcpyAsync(w1, p.w1, p.rows * p.inputs, cudaMemcpyDeviceToDevice, s));
}

inline void BlockWeights::bind(ForwardPass &forward) const {
  forward.wplanes = [this](int stream, int layer, int expert, i8 *w0, i8 *w1, cudaStream_t s) {
    write_planes(stream, layer, expert, w0, w1, s);
  };
  forward.wblocks = [this](int stream, int layer) {
    return blocks(stream, layer);
  };
}

inline size_t BlockWeights::row_pass_scratch() const {
  size_t most = 0; // maximum over each kind in layer 0 (no experts) and layer 1 (experts, if any)
  for (int kind = 0; kind < 12; kind++)
    for (int l = 0; l < std::min(shape.layers, 2); l++)
      if (kind_in_layer(shape, kind, l)) {
        const int st = stream_of_kind(kind);
        most = std::max(most, rowpass1_q4k_scratch((int)shape.rows(st, l), shape.inputs(st, l) / 256,
                                                   st >= 10 ? shape.NE : 1));
      }
  return most;
}
inline void BlockWeights::row_pass(int kind, int group, int cut, const T *phi, T *table, void *scratch,
                                   cudaStream_t stream, int only_layer) const {
  const int st = stream_of_kind(kind), experts = st >= 10 ? shape.NE : 1;
  size_t all = 0, before = 0; // input columns of this kind in the group, counted once per expert
  for (int l = group * shape.Lam; l < (group + 1) * shape.Lam; l++)
    if (kind_in_layer(shape, kind, l)) all += (size_t)experts * shape.inputs(st, l);
  for (int l = group * shape.Lam; l < (group + 1) * shape.Lam; l++) {
    if (!kind_in_layer(shape, kind, l)) continue;
    const Blocks &b = *blocks(st, l);
    const int per_row = b.inputs / 256, run = (int)b.rows >> cut;
    for (int r = 0; r < (1 << cut) && (only_layer < 0 || only_layer == l); r++)
      rowpass1_q4k(b.bytes + (size_t)r * run * per_row * block_bytes(b.type), run, per_row, 0, experts * per_row, phi,
                   table + (before << cut) + r, table + ((all + before) << cut) + r, scratch, stream, b.type,
                   (int)b.rows, 1 << cut, b.gain, b.scale);
    before += (size_t)experts * b.inputs;
  }
}

} // namespace wb
