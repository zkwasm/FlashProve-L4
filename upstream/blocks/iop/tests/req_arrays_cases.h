// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture: the synthetic requests of blocks/iop/tests/req_rows_cases.h, each matrix stored as
// a sub-proof (one part of a request's proof) stores its arrays: in 8-, 16- or 64-bit cells, in
// several orders (`ArraySource`, blocks/iop/req_arrays.h). The prover computes its claims' values
// from matrix values set here by written-out formulas instead of array_element(), to check it.
#pragma once
#include "iop/req_arrays.h"
#include "iop/tests/req_rows_cases.h"

namespace blk {

// By array name (rows_toy_layout()). `pointer`: the array's index in lay()'s `at`, one address for
// each array of a commit point (`moment`: a prover message of a segment, `link`).
inline ArraySources arrays_toy_sources(const ReqShape &shape, const ReqLayout &layout) {
  ArraySources sources;
  for (size_t l = 0; l < layout.link.size(); l++) {
    sources.emplace_back();
    std::vector<unsigned> pointers(shape.link[l].moments(), 0);
    for (const ReqArray &a : layout.link[l]) {
      ArraySource s;
      s.pointer = pointers[shape.link[l].messages[a.message].moment]++;
      if (a.name == "A") {  // 3-byte symbols, column after column; a column's last byte is absent
        s.cells = ArrayCells::SYMBOLS, s.row_stride = 3, s.column_stride = s.cells_in_a_column = 3 * a.laid.size_i - 1, s.symbol_bits = 20;
      } else if (a.name == "B") {  // after five unused cells
        s.cells = ArrayCells::UINT16, s.offset = 5, s.row_stride = a.laid.size_c;
      } else if (a.name == "K") {  // row after row, with a gap of three cells between rows
        s.cells = ArrayCells::INT16, s.row_stride = a.laid.size_c + 3;
      } else if (a.name == "R") {
        s.cells = ArrayCells::ELEMENTS, s.column_stride = 3;
      } else {  // "D": column after column
        s.cells = ArrayCells::UBYTES, s.row_stride = 1, s.column_stride = a.laid.size_i;
      }
      sources.back().push_back(s);
    }
  }
  return sources;
}

struct ArraysSubProver : RowsSubProver {
  ArraySources sources;
  std::vector<std::vector<std::vector<u8>>> bytes;  // [link][array]: bytes at the array's pointer
  ArraysSubProver(const ReqWorld &world, const ReqLayout &lay, u64 s, Engine &e, Coins &c, const E &o)
      : RowsSubProver(world, lay, s, e, c, o), sources(arrays_toy_sources(world.shape, lay)), bytes(world.shape.link.size()) {}

  // Builds the tables from host memory. Virtual: GPU tests build from GPU memory, or build nothing.
  virtual std::vector<TableId> lay(size_t l, unsigned moment, const std::vector<const void *> &at) {
    return req_tables_host(engine, w.codes, w.shape, layout, sources, l, moment, at);
  }
  std::vector<TableId> tables(size_t l, unsigned moment) override {
    matrix[l].resize(layout.link[l].size()), bytes[l].resize(layout.link[l].size());
    std::vector<const void *> at;
    for (size_t a = 0; a < layout.link[l].size(); a++) {
      const ReqArray &array = layout.link[l][a];
      if (w.shape.link[l].messages[array.message].moment != moment) continue;
      const u64 R = array.laid.size_i, K = array.laid.size_c;
      std::vector<u8> &b = bytes[l][a];
      std::vector<E> &v = matrix[l][a];
      v.assign(R * K, E::zero());
      auto i16 = [&](u64 cell) { return (i64)(b[2 * cell] | (b[2 * cell + 1] << 8)); };
      if (array.name == "A") {
        const u64 m = 3 * R - 1;
        b.resize(m * K);
        data.bytes(b.data(), b.size());
        for (u64 c = 0; c < K; c++)
          for (u64 y = 0; y < m; y++) v[(y / 3) * K + c] += E::from(from_i64((i64)(signed char)b[c * m + y] * ((i64)1 << (20 * (y % 3)))));
      } else if (array.name == "B") {
        b.resize(2 * (5 + K));
        data.bytes(b.data(), b.size());
        for (u64 c = 0; c < K; c++) v[c] = E::from(from_u64((u64)i16(5 + c)));
      } else if (array.name == "K") {
        b.resize(2 * R * (K + 3));
        data.bytes(b.data(), b.size());
        for (u64 r = 0; r < R; r++)
          for (u64 c = 0; c < K; c++) v[r * K + c] = E::from(from_i64(i16(r * (K + 3) + c) >= 32768 ? i16(r * (K + 3) + c) - 65536 : i16(r * (K + 3) + c)));
      } else if (array.name == "R") {
        b.resize(24 * K);
        for (u64 c = 0; c < K; c++) {
          v[c] = data.e();
          for (unsigned d = 0; d < 3; d++) word_to_bytes(v[c].c[d].v, &b[8 * (3 * c + d)]);
        }
      } else {
        b.resize(R * K);
        data.bytes(b.data(), b.size());
        for (u64 r = 0; r < R; r++)
          for (u64 c = 0; c < K; c++) v[r * K + c] = E::from(from_u64(b[c * R + r]));
      }
      at.push_back(b.data());
    }
    return lay(l, moment, at);
  }
};

template <class SubProver = ArraysSubProver>
inline std::vector<u8> arrays_prove(const ReqWorld &w, const ReqLayout &layout, u64 seed, const E &off, Engine &engine) {
  return rows_prove<SubProver>(w, layout, seed, off, engine);
}

}  // namespace blk
