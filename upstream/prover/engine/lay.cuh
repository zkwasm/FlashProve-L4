// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Writes a sub-proof's arrays from GPU memory into the main tables of the oracle messages of a
// segment (`link` in identifiers): the GPU form of req_tables_host() in blocks/iop/req_arrays.h.
// A sub-proof is the argument for one part of the forward pass; its arrays are the named matrices
// that it commits to; a segment is its part of a request's proof. An oracle message is rows that
// are encoded and sent as Merkle roots: main rows (its main table), mask rows of random values, or
// both. array_element() defines an element's value, laid_element() its row and `place` (index in
// the row); all other entries are zero, so the same proof results. Draws no private randomness.
//
// These tables are most of what a request holds in GPU memory, so each row gets the narrowest kind
// of cell (`Cell`, prover/engine/encode.cuh) that holds every array with a block in it, or 64-bit
// words (WORD); any other kind is narrow. A block (`LaidBlock`, blocks/iop/laid.h) is an aligned
// rectangle of an array; a band is a run of rows of one kind. The pad is the random element that
// masks the answer to an out-of-domain sample; a sampled message without mask rows has it in its
// main table. Only SEND, the step that sends the message, draws it: send_prove()
// (blocks/iop/send.h) ends the process unless the pad's entries are zero, then writes fresh random
// values there by Engine::set_entries(), which on the GPU requires WORD cells.
#pragma once
#include <vector>

#include "engine/device.cuh"
#include "engine/engine_gpu.cuh"
#include "iop/req_arrays.h"

namespace blk {

// The narrowest cell that holds every element of an array.
inline Cell cell_for(const ArraySource &s) {
  if (s.marks) return Cell::WORD; // marked rows have 2^15 subtracted: values from -2^15 to 2^16 - 1
  switch (s.cells) {
  case ArrayCells::BITS:
  case ArrayCells::UBYTES: return Cell::U8;
  case ArrayCells::BYTES:
  case ArrayCells::INT16: return Cell::I16;
  case ArrayCells::UINT16: return Cell::U16;
  case ArrayCells::SYMBOLS: return s.symbol_bits == S24_BITS ? Cell::S24 : Cell::WORD;
  default: return Cell::WORD;
  }
}
// The narrowest cell that holds what either holds.
inline Cell cell_for_both(Cell a, Cell b) {
  if (a == b) return a;
  if (a == Cell::U8 && (b == Cell::U16 || b == Cell::I16)) return b;
  if (b == Cell::U8 && (a == Cell::U16 || a == Cell::I16)) return a;
  return Cell::WORD;
}

// Limits a table to `most` bands: card_bands() and EngineGpu::adopt() end the process on more than
// MOST_BANDS. While it has more runs of rows of one kind, the shortest run of narrow rows is
// widened to WORD, which holds whatever a narrow cell holds.
inline void widen_to_bands(std::vector<Cell> &kind, size_t most = MOST_BANDS) {
  BLK_REQUIRE(most >= 1, "table: at least one band");
  for (;;) {
    size_t runs = kind.empty() ? 0 : 1;
    for (size_t row = 1; row < kind.size(); row++) runs += kind[row] != kind[row - 1];
    if (runs <= most) return; // in the end all rows are words: one run
    size_t shortest = 0, shortest_rows = kind.size() + 1;
    for (size_t row = 0; row < kind.size();) {
      size_t end = row;
      while (end < kind.size() && kind[end] == kind[row]) end++;
      if (kind[row] != Cell::WORD && end - row < shortest_rows) shortest = row, shortest_rows = end - row;
      row = end;
    }
    for (size_t row = shortest; row < shortest + shortest_rows; row++) kind[row] = Cell::WORD;
  }
}

// One thread per element of a `LaidBlock`, written in its row's kind of cell. at: the base address
// for `source`. deg: cells per entry (`main_deg`): 1 over the field F, 3 over its extension E.
static __global__ void k_lay_block(const void *at, ArraySource source, LaidBlock block, unsigned deg, CardBands table) {
  const u64 x = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (x >> (block.bits_i + block.bits_c)) return;
  const LaidElement e = laid_element(block, x);
  const unsigned band = table.band_of(e.row);
  const u64 cell = (e.row - table.first_row[band]) * table.row_cells + e.place * deg;
  void *cells = const_cast<void *>(table.cells[band]);
  if (table.cell[band] == Cell::S24) { // SYMBOLS at S24_BITS (`cell_for`): three digits, unchanged
    for (unsigned k = 0; k < 3; k++) ((signed char *)cells)[3 * cell + k] = (signed char)array_symbol(at, source, e.i, e.c, k);
    return;
  }
  F out[3];
  array_element(at, source, e.i, e.c, out);
  // A narrow cell stores the integer, which fits (cell_for()). A negative integer n is the element
  // P + n: subtracting P in 64 bits leaves n in two's complement, whose low bits are the cell.
  switch (table.cell[band]) {
  case Cell::WORD:
    for (unsigned d = 0; d < deg; d++) ((u64 *)cells)[cell + d] = out[d].v;
    break;
  case Cell::U8: ((u8 *)cells)[cell] = (u8)out[0].v; break;
  case Cell::U16: ((unsigned short *)cells)[cell] = (unsigned short)out[0].v; break;
  default: ((short *)cells)[cell] = (short)(unsigned short)(out[0].v > P / 2 ? out[0].v - P : out[0].v); break;
  }
}
// Makes the tables in GPU memory (`Card` in identifiers) and owns their bands: EngineGpu::adopt()
// does not copy them, and they stay allocated until give_back() or this object's destruction.
class CardTables {
public:
  CardTables(EngineGpu &engine, cudaStream_t stream = 0) : engine_(engine), stream_(stream) {}
  // Returns the main table of each oracle message of commit point `moment` (a numbered prover
  // message of the segment), in order; NO_ID for a message without main rows. As req_tables_host(),
  // with `at` and the sources' `marks` pointing into GPU memory. A test that is not part of this
  // repository compares the proofs made with the two, byte for byte, on
  // small synthetic requests. Waits for no kernel, so `stream` must keep order with the default
  // stream, the engine's, and the work that wrote the arrays: not checked. Of the layout only the
  // blocks' rows are checked: it must have passed req_layout_fault() (blocks/iop/req_rows.h), or
  // k_lay_block() can write outside a band.
  std::vector<TableId> lay(const Codes &codes, const ReqShape &shape, const ReqLayout &layout, const ArraySources &sources, size_t link, unsigned moment,
                           const std::vector<const void *> &at) {
    const char *fault = req_sources_fault(shape, layout, sources); // `cell_for` and `k_lay_block` rely on valid sources
    BLK_REQUIRE(!fault, fault ? fault : "");
    if (held_.size() <= link) held_.resize(link + 1);
    std::vector<TableId> main;
    const std::vector<ReqMessage> &messages = shape.link[link].messages;
    for (size_t m = 0; m < messages.size(); m++) {
      const OracleShape &o = messages[m].shape;
      if (messages[m].moment != moment) continue;
      if (!o.main_rows) {
        main.push_back(NO_ID);
        continue;
      }
      const u64 len = codes.base[o.code].ell;
      const std::vector<PadPlace> pad = o.mask_rows ? std::vector<PadPlace>() : Send::pad(codes, o);
      const unsigned row_bits = ceil_log2(o.main_rows);
      const std::vector<Band> bands = take(link, kinds(layout, sources, link, m, o, pad, row_bits), len * o.main_deg);
      const CardBands table = card_bands(bands, len * o.main_deg);
      for (size_t a = 0; a < layout.link[link].size(); a++) {
        const ReqArray &array = layout.link[link][a];
        if (array.message != m) continue;
        const ArraySource &s = sources[link][a];
        BLK_REQUIRE(s.pointer < at.size(), "request: an array is read at a pointer that the moment does not pass");
        for (const LaidBlock &b : array.laid.blocks) {
          const u64 elements = u64(1) << (b.bits_i + b.bits_c);
          k_lay_block<<<blocks_for(elements, 256), 256, 0, stream_>>>(at[s.pointer], s, b, o.main_deg, table);
          ENGINE_CK(cudaGetLastError());
        }
      }
      main.push_back(engine_.adopt(bands, o.main_deg, o.main_rows, len)); // the pad's entries are zero: SEND draws the pad (see the top of this file)
    }
    return main;
  }
  // Frees the segment's tables. Call it only after Engine::drop() of each of them: not checked.
  void give_back(size_t link) {
    if (link < held_.size()) held_[link].clear();
  }
  // The kind of cell in which lay() writes each row of message m of a segment.
  static std::vector<Cell> row_kinds(const Codes &codes, const ReqShape &shape, const ReqLayout &layout, const ArraySources &sources, size_t link, size_t m) {
    const OracleShape &o = shape.link[link].messages[m].shape;
    return kinds(layout, sources, link, m, o, o.mask_rows ? std::vector<PadPlace>() : Send::pad(codes, o), ceil_log2(o.main_rows));
  }
  // Bytes of GPU memory that lay() allocates for the main table of message m of a segment.
  static size_t table_bytes(const Codes &codes, const ReqShape &shape, const ReqLayout &layout, const ArraySources &sources, size_t link, size_t m) {
    const OracleShape &o = shape.link[link].messages[m].shape;
    if (!o.main_rows) return 0;
    size_t sum = 0;
    for (Cell kind : row_kinds(codes, shape, layout, sources, link, m)) sum += codes.base[o.code].ell * o.main_deg * cell_bytes(kind);
    return sum;
  }
  // Bytes of GPU memory that the segment's tables occupy now.
  size_t bytes(size_t link) const {
    size_t sum = 0;
    if (link < held_.size())
      for (const Card<u8> &band : held_[link]) sum += band.size();
    return sum;
  }

private:
  // As row_kinds(). Rows over E (`main_deg` 3) and the pad's row are WORD.
  static std::vector<Cell> kinds(const ReqLayout &layout, const ArraySources &sources, size_t link, size_t m, const OracleShape &o,
                                 const std::vector<PadPlace> &pad, unsigned row_bits) {
    std::vector<Cell> kind(o.main_rows, Cell::WORD);
    std::vector<bool> met(o.main_rows, false);
    for (size_t a = 0; a < layout.link[link].size(); a++) {
      if (layout.link[link][a].message != m) continue;
      const Cell cell = cell_for(sources[link][a]);
      for (const LaidBlock &b : layout.link[link][a].laid.blocks)
        for (u64 row = b.first_row; row < b.first_row + (u64(1) << b.row_bits); row++) {
          BLK_REQUIRE(row < o.main_rows, "request: a block is laid beyond the message's rows");
          kind[row] = met[row] ? cell_for_both(kind[row], cell) : cell;
          met[row] = true;
        }
    }
    if (o.main_deg != 1) return std::vector<Cell>(o.main_rows, Cell::WORD); // after the loop, which checks that each block is within the rows
    for (const PadPlace &p : pad) { // SEND will write field elements here, in place
      const u64 row = p.index & ((u64(1) << row_bits) - 1);
      kind[row] = Cell::WORD, met[row] = true;
    }
    // A row with no kind yet holds zeros: it takes the kind of the row before, else of the next.
    for (size_t row = 1; row < kind.size(); row++)
      if (!met[row] && met[row - 1]) kind[row] = kind[row - 1], met[row] = true;
    for (size_t row = kind.size(); row-- > 1;)
      if (!met[row - 1] && met[row]) kind[row - 1] = kind[row], met[row - 1] = true;
    widen_to_bands(kind);
    return kind;
  }
  // Allocates and zeroes GPU memory for the rows: one band for each run of rows of one kind.
  std::vector<Band> take(size_t link, const std::vector<Cell> &kind, u64 row_cells) {
    std::vector<Band> bands;
    for (size_t row = 0; row < kind.size();) {
      size_t end = row;
      while (end < kind.size() && kind[end] == kind[row]) end++;
      const size_t bytes = (end - row) * row_cells * cell_bytes(kind[row]);
      held_[link].emplace_back();
      held_[link].back().resize(bytes);
      ENGINE_CK(cudaMemsetAsync(held_[link].back().get(), 0, bytes, stream_));
      bands.push_back(Band{held_[link].back().get(), kind[row], end - row});
      row = end;
    }
    return bands;
  }

  EngineGpu &engine_;
  cudaStream_t stream_;
  std::vector<std::vector<Card<u8>>> held_; // by link: the bands of its tables
};

} // namespace blk
