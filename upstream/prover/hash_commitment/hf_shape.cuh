// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// HfShape::from_policy() derives from the parameter file (`policy.txt`) the model's dimensions,
// the layout of the committed weight matrix W, and the weight commitment's parameters.
// A unit is W's rows for the embedding, the output head, or one layer's weights of one kind.
// Include after prover/weights_commitment/common.cuh and prover/weights_commitment/arguments.h.
#pragma once
#include "hash_commitment/hf_params.h"
namespace rf {
struct HfShape {
  // lk: log2 of the number of W's columns, the last number of the line `table`; hb: the error
  // accounting's target, in bits, the line `sec`; eh: nonzero if the embedding and the output head
  // are in W.
  int lk = 0, hb = 0, eh = 0;
  ShapeW sh;
  LayW Y;                  // W's layout; Y.mW == Y.m0
  size_t k = 0, m0 = 0;    // columns; rows
  vector<::g5::UM> um;     // the units, as the kernels read them
  ::g5::UM *umD = nullptr; // the same on the GPU
  // The model's dimensions, for the kernels that write W's cells.
  ::g5::SM model{};
  ::hf::HfParams P; // the weight commitment's parameters
  int from_policy(const PolC &PO);
};
// Fills the fields from the parameter file PO; the caller then sets model.seed (the synthetic
// weights' seed) and P.configuration. Sets the globals D (prover/common/host_common.h), of which m0
// is a multiple, and, through HfParams::from_policy(), `s8::now` (prover/common/state8.h). Call it
// once per object: it appends to um. Returns 0, or 2 with a reason on stderr. PO[name] exits with
// code 2 if the name is missing; a failed CUDA call exits with code 5.
// from_policy() does not check that the lines D, d, dh, G, H and f (and fe if NE > 0) are at
// least 1. PolC::read() and PolC::take() do; PolC::read_form(), which only tests call, does not.
// With a 0, layoutW() (prover/common/weight_layout.h) can loop forever or divide by zero.
int HfShape::from_policy(const PolC &PO) {
  D = PO["D"];
  const int p = PO["prof"] & 15;
  eh = PO["prof"] >> 4;
  lk = (int)PO.file.frame.block_log;
  hb = (int)PO.file.frame.sec;
  if (lk < ELKW_LEAST || lk > ELKW_MOST) {
    fprintf(stderr, "the last number of the parameter file's line table must be from %d to %d\n", ELKW_LEAST, ELKW_MOST);
    return 2;
  }
  for (const char *line : {"rkW", "mkW", "lmW", "cw", "lbzW", "tauW_W", "hbc", "lxb", "ln1", "ln2", "mkc", "er1", "oiG"})
    if (PO.m.count(line)) {
      fprintf(stderr, "the parameter file has a line %s, which the flat commitment does not have: remove it\n",
              line);
      return 2;
    }
  sh.L = PO["L"];
  sh.d = PO["d"];
  sh.dh = PO["dh"];
  sh.dkv = (size_t)PO["G"] * PO["dh"];
  sh.dq = (size_t)PO["H"] * PO["dh"];
  sh.f = PO["f"];
  sh.fe = PO["fe"];
  sh.NE = PO["NE"];
  sh.V = eh ? PO["V"] : 0;
  sh.moeMode = (p >= 3 && PO["NE"] > 0) ? 1 : 0;
  // If a token index were not entirely in W's column index, the coefficients on W's rows in a
  // request's weight claims would show which tokens the request contains.
  if (eh && (clgW(sh.V) < 1 || clgW(sh.V) > lk)) {
    fprintf(
        stderr,
        "the last number of the parameter file's line table is too small, or V is below 2: a token index must fit entirely in the column index "
        "(w-list-token-factor)\n");
    return 2;
  }
  Y = layoutW(sh, lk, D, 0, eh ? 1 : 0); // rR = 0: no rows after the m0
  k = Y.k;
  m0 = Y.m0;
  if (m0 > ((size_t)1 << ELMW)) {
    fprintf(stderr, "the model is too large for this parameter file: W would have more than 2^%d rows\n", ELMW);
    return 2;
  }
  if (int rc = P.from_policy(PO.file.frame, m0)) return rc;
  // UM has space for three groups (GrpW), the most that layoutW() gives a unit.
  for (auto &u : Y.U) {
    ::g5::UM d{};
    d.off = u.off;
    d.rows = u.rows;
    d.X = u.X;
    d.l = u.l;
    d.nlw = u.nlw;
    d.ncb = u.ncb;
    d.ng = (int)u.g.size();
    for (size_t i = 0; i < u.g.size(); i++) {
      d.cls[i] = u.g[i].cls;
      d.z[i] = u.g[i].z;
      d.take[i] = u.g[i].take;
      d.cb0[i] = u.g[i].cb0;
      d.left[i] = u.g[i].left;
      d.R[i] = u.g[i].R;
    }
    um.push_back(d);
  }
  umD = ::g5::up(um.data(), um.size());
  model.d = sh.d;
  model.H = PO["H"];
  model.Gk = PO["G"];
  model.dh = sh.dh;
  model.dq = sh.dq;
  model.dkv = sh.dkv;
  model.f = sh.f;
  model.fe = sh.fe;
  model.NE = sh.NE;
  model.V = sh.V;
  model.moe = sh.moeMode;
  return 0;
}
} // namespace rf
