// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Interfaces to a model stored as quantized blocks (GGUF): BlockWeights::fill() reads a BlockSource
// into a Placement's GPU memory. A cell is a committed byte, a weight's limb. An int `stream` is a
// tensor's number: 2 q, 3 k, 4 v, 5 o, 6 gate, 7 up, 8 down (in a layer with experts: the shared expert's), 9 router,
// 10 to 12 the routed experts' counterparts of 6 to 8, 13 embedding, 14 output head.
#pragma once
#include "prover/weights.h"
#include "weights_blocks/blocks.h"
#include <functional>

namespace wb {

// One tensor's bytes on the GPU, as in the file; `type` is a BlockType (blocks.h). Row o is
// output o. A routed tensor's experts follow one another, so rows = experts * rows of one expert.
struct FileTensor {
  const uint8_t *bytes = nullptr;
  int type = Q4_K;
  size_t rows = 0;
  int inputs = 0; // elements in a row, a multiple of 256
};

// Supplies a model's tensors. It is used only at set-up, never while a request is proved.
struct BlockSource {
  virtual ~BlockSource() {}
  virtual int type(int stream, int layer) = 0;
  // If true, a matrix of streams 2 to 8 is stored as Form::Blocks; decoded_from_blocks() must hold for
  // its type.
  virtual bool as_blocks(int stream, int layer) { return decoded_from_blocks(type(stream, layer)); }
  // Whether tensor() accepts a `keep` that is not null: `room` bytes of pinned host memory, passed
  // with `into`, that receive the same bytes and must not change until `reader`'s work has run.
  virtual bool keeps_on_host() { return false; }
  // Returns the tensor's bytes: in `into`, if it is not null (the process exits if they exceed
  // `room` bytes); otherwise in the source's buffer until the next such call, which first waits
  // for the work queued on `reader` so far. Work queued on `reader` after the call sees the bytes.
  virtual FileTensor tensor(int stream, int layer, uint8_t *into, size_t room, cudaStream_t reader, uint8_t *keep) = 0;
  // The gains: multipliers gamma_j of the conversion to committed integers (fold_rule.h), one float
  // per input, on the GPU, 16-byte aligned (not checked); null if the input is not a norm's output.
  virtual const float *gain(int stream, int layer) = 0;
  // 2^e, where e is the tensor's exponent in the same conversion: 0 to 40, at most 24 for Blocks.
  virtual float scale(int stream, int layer) = 0;
  // Called when no further tensor will be requested: the source may release its buffers and close
  // its files. The gains stay valid for the life of the process.
  virtual void all_read() {}
};

// The forms in which a layer group's weights (Shape::Lam consecutive layers) are stored on the GPU.
// Blocks: the file's bytes. Planes: each weight's limbs w0, w1 (w = w0 + 256 w1), [output][input].
// Cells: the same limbs in the committed cells' layout (cells_in_place()). RoutedBlocks has the
// format of Blocks; it marks the routed tensors for a Placement that stores them elsewhere.
// Public source: a raw or a compressed form for each projection is described by Nikulin ("Unweight:
// Lossless MLP Weight Compression for LLM Inference", Cloudflare Technical Report Cf-TR-2026.04.v1,
// 2026, sections 1.3, 5 and 6.2, https://research.cloudflare.com/papers/unweight-2026.pdf); here
// the compressed form is the file's quantized blocks (`stays_blocks` in fill(); as_blocks()).
enum class Form { Blocks, Planes, Cells, RoutedBlocks };

// Decides where on the GPU the forms are stored. The default allocates per call and never frees.
struct Placement {
  virtual ~Placement() {}
  // Before its first call of place(), place_routed() or place_end(), fill() announces each of these
  // calls here, in order, with its size. A routed tensor has the form RoutedBlocks and its layer,
  // everything else layer -1. The embedding's cells (Cells) and the head (Planes) have group -1.
  virtual void will_place(size_t bytes, int group, Form form, int layer) {}
  // Returns 256-byte aligned device memory. `group` is the layer group, or -1 for the embedding's
  // cells and the head. Within a group all Blocks and RoutedBlocks are requested first, then all
  // Planes, then all Cells. Only then is the group's first tensor read.
  virtual void *place(size_t bytes, int group, Form form);
  // Returns the memory for one routed tensor of `layer`, aligned like place()'s. A layer's three
  // routed tensors are requested one after the other, in the order of their streams (10, 11, 12),
  // after that layer's Blocks.
  virtual void *place_routed(size_t bytes, int group, int layer) { return place(bytes, group, Form::RoutedBlocks); }
  // Returns pinned host memory for a copy of the `bytes` bytes at `placed`, or null if the
  // placement keeps none. `placed` is from place() (form Blocks) or place_routed(), for the group
  // or layer being filled, after the entire group is placed. The result becomes tensor()'s `keep`.
  virtual uint8_t *host_twin(const void *placed, size_t bytes) { return nullptr; }
  // Called once all work that fills and checks the routed tensors of `layer` is queued on the CUDA
  // stream `stream`, and before any tensor of a later layer is read, because a placement may use
  // the same memory for the routed tensors of different layers of one group.
  virtual void routed_filled(int layer, cudaStream_t stream) {}
  // Returns the memory for the embedding's cells (`which` 0) or the head (`which` 1). end_filled()
  // is called once the work that fills it is queued on `stream`: for the cells, before the head's
  // memory is requested, because a placement may use the same memory for both.
  virtual void *place_end(size_t bytes, int which) { return place(bytes, -1, which ? Form::Planes : Form::Cells); }
  virtual void end_filled(int which, cudaStream_t stream) {}
  // Called once all work that fills the group's forms is queued on `stream`. The groups come in
  // ascending order, then -1. Nothing of the next group is requested before this call returns, so
  // the call may wait until the stream's work has run.
  virtual void group_filled(int group, cudaStream_t stream) {}
};
// H heads of q, G of k, G of v, dh wide. Feed-forward width f; with experts, fe. NE routed experts.
struct Shape {
  int layers = 0, Lam = 1, d = 0, H = 0, G = 0, dh = 0, f = 0, fe = 0, NE = 0, V = 0;
  bool experts = false; // true: every odd layer has experts
  bool has_experts(int layer) const { return experts && (layer & 1); }
  // Output rows (of one expert) and input columns of a stream's tensor in a layer.
  size_t rows(int stream, int layer) const;
  int inputs(int stream, int layer) const;
};

// Implements Weights (prover/prover/weights.h). `kind` of a group's cells, by stream: 0: 2 and 3;
// 1: 4; 2: 5; 3, 4: 6, 7 without experts; 5, 6: 6, 7 with; 7, 8: 10, 11; 9: 8; 10: 12; 11: 9.
struct BlockWeights : Weights {
  Shape shape;
  // If set, fill() calls it before it reads each tensor; the function may end the process.
  std::function<void()> between_tensors;

  // Set `shape` first. Exits with code 3 on an unsupported shape or tensor (size, type, scale,
  // gains) and on a weight that is not finite or exceeds 4095 in magnitude (embedding: 2^30 - 1).
  void fill(BlockSource &source, Placement &where, cudaStream_t stream = cudaStreamPerThread);

  // Each returns null if the tensor is not in that form. Streams 10 to 12 are Blocks, 9 never.
  const Blocks *blocks(int stream, int layer) const;
  const Planes *tensor_planes(int stream, int layer) const;
  // A group's cells of a kind, [limb][layer][input][output]; null if `kind` > 0 and all its tensors
  // there are blocks. In kind 0, [output] is [index in a head][64 heads: q's H, k's G, zeros].
  // If null, cells() decodes the blocks, or exits (code 2) if `parts` != 1 or `kind` is 7, 8 or 10.
  // Public source: NVIDIA's Transformer Engine documentation ("Performance Considerations",
  // "Handling transposes") describes a second, transposed tensor for a second matrix product, in
  // FP8 training on Hopper GPUs; here the proof's pass over the committed weights reads the cells.
  const i8 *cells_in_place(int kind, int group) const override;
  const i8 *embedding_in_place() const override { return embedding_; } // [limb < 4][token][k]
  bool lays_out_cells(int kind) const override { return kind != 7 && kind != 8 && kind != 10; }
  bool cells_number(int kind, int group, size_t bytes) const override {
    return bytes == 2 * cells_per_limb(kind, group);
  }
  bool has_row_pass(int kind, int group) const override { return all_blocks(kind, group); }

  bool planes(int, int, ForwardPass::WP &) const override { return false; }
  void cells(int kind, int group, size_t bytes, int part, int parts, i8 *out) const override;
  void embed(const int *tokens, int T, int xl, ResidualInt *x0, i8 *x0_cells) const override;
  void embedding_cells(size_t start, size_t words, u32 *out) const override;
  const int16_t *head() const override { return head_; }

  // Writes the two byte planes [output][input] of a tensor of streams 2 to 12 (routed: of one
  // expert), decoded from its blocks or copied from its planes. It must exist (not checked).
  void write_planes(int stream, int layer, int expert, i8 *w0, i8 *w1, cudaStream_t s) const;
  void bind(ForwardPass &forward) const override;
  void row_pass(int kind, int group, int cut, const T *phi, T *table, void *scratch, cudaStream_t stream,
                int only_layer = -1) const override;
  size_t row_pass_scratch() const override;

private:
  bool all_blocks(int kind, int group) const; // false for kinds 0 and 11
  size_t cells_per_limb(int kind, int group) const;
  std::map<std::pair<int, int>, Blocks> blocks_;
  std::map<std::pair<int, int>, Planes> planes_;
  std::map<std::pair<int, int>, i8 *> cells_;
  i8 *embedding_ = nullptr;
  int16_t *head_ = nullptr;
};

} // namespace wb
