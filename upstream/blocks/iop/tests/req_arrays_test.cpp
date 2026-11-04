// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/iop/req_arrays.h, which reads arrays in a prover's memory. Part 1 checks
// array_element() on fixed bytes for each cell type except ArrayCells::ELEMENTS (in part 2) and
// ArrayCells::WORDS (not tested here). Part 2 proves synthetic requests
// (blocks/iop/tests/req_arrays_cases.h). An accepted proof shows that three things agree: each
// equation's value (`target`), computed without array_element(); the tables, which
// req_tables_host() fills using array_element(); the verifier's linear forms, from layout and
// weights. A false `target` must be rejected.
#include "iop/tests/req_arrays_cases.h"

using namespace blk;

int main() {
  {
    const u8 b[8] = {0xA5, 0xFF, 0x00, 0x80, 0x7F, 0x01, 0xFE, 0x03};
    F x[3];
    auto one = [&](ArraySource s, u64 r, u64 c) {
      array_element(b, s, r, c, x);
      return x[0];
    };
    ArraySource s;
    s.cells = ArrayCells::BITS;
    CHECK(one(s, 0, 0) == F::one() && one(s, 0, 1) == F::zero() && one(s, 0, 2) == F::one() && one(s, 0, 7) == F::one() && one(s, 0, 16) == F::zero(), "bits");
    s.cells = ArrayCells::BYTES;
    CHECK(one(s, 0, 1) == from_i64(-1) && one(s, 0, 3) == from_i64(-128) && one(s, 0, 4) == from_u64(127), "signed bytes");
    s.cells = ArrayCells::UBYTES;
    CHECK(one(s, 0, 1) == from_u64(255) && one(s, 0, 3) == from_u64(128), "unsigned bytes");
    s.cells = ArrayCells::INT16;
    CHECK(one(s, 0, 0) == from_i64(-91) && one(s, 0, 1) == from_i64(-32768) && one(s, 0, 2) == from_u64(0x017F), "signed 16 bits");
    s.cells = ArrayCells::UINT16;
    CHECK(one(s, 0, 0) == from_u64(0xFFA5) && one(s, 0, 1) == from_u64(0x8000), "unsigned 16 bits");
    const u8 marks[2] = {0, 1};
    s.marks = marks, s.row_stride = 2;
    CHECK(one(s, 0, 0) == from_u64(0xFFA5) && one(s, 1, 0) == from_i64(0x017F - 32768) && one(s, 1, 1) == from_i64(0x03FE - 32768), "marked rows");
    s = ArraySource();
    s.cells = ArrayCells::SYMBOLS, s.row_stride = 3, s.column_stride = 4, s.cells_in_a_column = 4, s.symbol_bits = 20;
    CHECK(one(s, 0, 0) == from_i64(-91 - (i64(1) << 20)) && one(s, 1, 0) == from_i64(-128) && one(s, 0, 1) == from_i64(127 + (i64(1) << 20) - 2 * (i64(1) << 40)) &&
              one(s, 1, 1) == from_u64(3),
          "symbols, the last of a column short");
    s = ArraySource();
    s.cells = ArrayCells::UBYTES, s.offset = 1, s.row_stride = 2, s.column_stride = 1, s.cells_in_a_column = 3;
    CHECK(one(s, 1, 1) == from_u64(0x7F) && one(s, 2, 0) == F::zero(), "offset, strides, and a cell beyond its column's end");
  }
  for (const ReqCase &t : rows_toy_cases()) {
    const ReqWorld w = req_toy_world(t, 1);
    const ReqLayout layout = rows_toy_layout(w);
    CHECK(!req_sources_fault(w.shape, layout, arrays_toy_sources(w.shape, layout)), "lawful sources are found faulty");
    EngineCpu engine, again;
    CHECK(rows_verify(w, layout, 600, arrays_prove(w, layout, 600, E::zero(), engine), false), "an honest proof is refused");
    CHECK(!rows_verify(w, layout, 600, arrays_prove(w, layout, 600, E::one(), again), false), "a false target is accepted");
  }
  std::printf("req_arrays_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
