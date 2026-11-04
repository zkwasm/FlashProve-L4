// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the arrays that a sub-proof (of one part of the computation) commits to are read from its
// working memory and put into tables.
//
// A sub-proof computes with narrow cells (bits, bytes, 16-bit numbers, field elements) in whatever
// order suits its kernels. It commits to arrays: matrices over the field F or its cubic extension E
// (req_rows.h). An ArraySource defines how element (r, c) of an array is made from cells, so that
// arrays are read in place: column-major, with gaps between rows, or packed from three cells.
// Nothing here checks that a source's cells lie inside the memory passed.
// In identifiers, `link` is a segment (one sub-proof's part of a request's proof) and `moment` a
// commit point (a prover message of the segment that carries oracle messages; numbered from 0).
#pragma once
#include <vector>

#include "base.h"
#include "engine/engine.h"
#include "iop/req.h"
#include "iop/req_rows.h"
#include "iop/send.h"

namespace blk {

// The storage format of an array's cells.
enum class ArrayCells {
  BITS,      // 0 or 1, eight to a byte, the lowest bit first
  BYTES,     // signed
  UBYTES,    // unsigned
  INT16,     // signed, little-endian
  UINT16,    // unsigned, little-endian
  WORDS,     // an element of F as a u64 below the modulus (not checked)
  ELEMENTS,  // an element of E: three neighbouring words. A cell is a word
  SYMBOLS    // three neighbouring signed bytes b0, b1, b2: the element b0 + 2^s b1 + 2^(2s) b2
};

// Element (r, c) starts offset + r * row_stride + c * column_stride cells after the chosen pointer.
// A cell at distance cells_in_a_column or more from its column's first cell is absent: it is not
// read and counts as 0.
struct ArraySource {
  ArrayCells cells = ArrayCells::WORDS;
  u64 offset = 0, row_stride = 0, column_stride = 1;  // in cells
  u64 cells_in_a_column = ~u64(0);
  unsigned pointer = 0;             // which of the commit point's pointers
  unsigned symbol_bits = 20;        // s of SYMBOLS
  // UINT16 only. If set: one byte for each row, in the same memory (host or GPU) as the cells. In a
  // row whose byte is not 0 the cells are stored with 2^15 added: the element is the cell - 2^15.
  const u8 *marks = nullptr;
};

BLK_HD inline i64 array_cell(const void *at, ArrayCells kind, u64 i) {
  const u8 *b = static_cast<const u8 *>(at);
  switch (kind) {
    case ArrayCells::BITS: return (b[i >> 3] >> (i & 7)) & 1;
    case ArrayCells::UBYTES: return b[i];
    case ArrayCells::INT16: return (i64)(short)(unsigned short)(b[2 * i] | (b[2 * i + 1] << 8));
    case ArrayCells::UINT16: return b[2 * i] | (b[2 * i + 1] << 8);
    default: return (i64)(signed char)b[i];
  }
}

// SYMBOLS: digit k (0, 1, 2) of element (r, c): a signed byte, or 0 where the cell is absent.
BLK_HD inline i64 array_symbol(const void *at, const ArraySource &s, u64 r, u64 c, unsigned k) {
  return r * s.row_stride + k < s.cells_in_a_column ? array_cell(at, ArrayCells::BYTES, s.offset + r * s.row_stride + c * s.column_stride + k) : 0;
}

// Defines element (r, c) for host and GPU code. Writes it to out[0]; for ELEMENTS, to out[0 .. 3).
BLK_HD inline void array_element(const void *at, const ArraySource &s, u64 r, u64 c, F *out) {
  const u64 i = s.offset + r * s.row_stride + c * s.column_stride, down = r * s.row_stride;
  if (s.cells == ArrayCells::WORDS || s.cells == ArrayCells::ELEMENTS) {
    const u64 *w = static_cast<const u64 *>(at);
    for (unsigned d = 0; d < (s.cells == ArrayCells::WORDS ? 1u : 3u); d++) out[d] = down + d < s.cells_in_a_column ? F{w[i + d]} : F::zero();
  } else if (s.cells == ArrayCells::SYMBOLS) {
    i64 v = 0;
    for (unsigned k = 0; k < 3; k++) v += array_symbol(at, s, r, c, k) * ((i64)1 << (s.symbol_bits * k));
    out[0] = from_i64(v);
  } else {
    out[0] = down < s.cells_in_a_column ? from_i64(array_cell(at, s.cells, i) - (s.marks && s.marks[r] ? 32768 : 0)) : F::zero();
  }
}

// Element x of a LaidBlock (laid.h): array element (i, c), at index `place` of table row `row`.
struct LaidElement {
  u64 i, c, row, place;
};
BLK_HD inline LaidElement laid_element(const LaidBlock &b, u64 x) {
  LaidElement e;
  const u64 low = x & ((u64(1) << (b.i_is_low ? b.bits_i : b.bits_c)) - 1), high = x >> (b.i_is_low ? b.bits_i : b.bits_c);
  e.i = b.first_i + (b.i_is_low ? low : high), e.c = b.first_c + (b.i_is_low ? high : low);
  e.row = b.first_row + (x >> b.place_bits), e.place = b.first_place + (x & ((u64(1) << b.place_bits) - 1));
  return e;
}

// sources[l][a] is the source of the array layout.link[l][a].
using ArraySources = std::vector<std::vector<ArraySource>>;

// Null, or what is wrong with the sources. ELEMENTS sources, and no others, fill main rows over E.
// The layout must have passed req_layout_fault().
inline const char *req_sources_fault(const ReqShape &shape, const ReqLayout &layout, const ArraySources &sources) {
  if (sources.size() != layout.link.size()) return "the sources have another number of links than the layout";
  for (size_t l = 0; l < sources.size(); l++) {
    if (sources[l].size() != layout.link[l].size()) return "a link has another number of sources than of arrays";
    for (size_t a = 0; a < sources[l].size(); a++) {
      const bool over_E = shape.link[l].messages[layout.link[l][a].message].shape.main_deg == E_DEGREE;
      if (over_E != (sources[l][a].cells == ArrayCells::ELEMENTS)) return "elements of E and rows over E go together";
      if (sources[l][a].marks && sources[l][a].cells != ArrayCells::UINT16) return "only unsigned 16-bit cells can have marked rows";
      if (sources[l][a].symbol_bits > 20) return "three symbols' digits do not fit 62 bits";
    }
  }
  return nullptr;
}

// The main tables of the oracle messages of commit point `moment` of segment `link`, in order
// (NO_ID for one without main rows). `at`: the commit point's pointers, into host memory. The
// sources must have passed req_sources_fault(). Entries that no array covers stay 0, also those of
// Send::pad() (the pad: a random element that masks a sample's answer): send_prove() requires 0
// there and draws every pad. Only tests call this; on the GPU: CardTables::lay() in
// prover/engine/lay.cuh.
inline std::vector<TableId> req_tables_host(Engine &engine, const Codes &codes, const ReqShape &shape, const ReqLayout &layout,
                                            const ArraySources &sources, size_t link, unsigned moment, const std::vector<const void *> &at) {
  std::vector<TableId> main;
  const std::vector<ReqMessage> &messages = shape.link[link].messages;
  for (size_t m = 0; m < messages.size(); m++) {
    const OracleShape &o = messages[m].shape;
    if (messages[m].moment != moment) continue;
    if (!o.main_rows) {
      main.push_back(NO_ID);
      continue;
    }
    Table t(o.main_deg, o.main_rows, codes.base[o.code].ell);
    for (size_t a = 0; a < layout.link[link].size(); a++) {
      const ReqArray &array = layout.link[link][a];
      if (array.message != m) continue;
      const ArraySource &s = sources[link][a];
      BLK_REQUIRE(s.pointer < at.size(), "request: an array is read at a pointer that the moment does not pass");
      for (const LaidBlock &b : array.laid.blocks)
        for (u64 x = 0; x < u64(1) << (b.bits_i + b.bits_c); x++) {
          const LaidElement e = laid_element(b, x);
          array_element(at[s.pointer], s, e.i, e.c, &t.a[(e.row * t.len + e.place) * t.deg]);
        }
    }
    main.push_back(engine.put(t));
  }
  return main;
}

}  // namespace blk
