// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Where each weight is stored in W, the matrix of weight digits that the program cp_commit commits
// to (layoutW()), and a claim's coefficients on W's rows (gammaW()). The verifier's
// implementation is verifier/src/vf_weights_layout.h; verifier/tests/weights_twin_test.cpp compares the
// two on small shapes. The includer first declares `vector`, std::min(), u32, u64, SW, E, E0, E1,
// ebase(), esub() and emul().
#pragma once
// One index of a unit's tensor (an index group; unit: see UnitW), of extent R. z: R's trailing
// zero bits ('A': 0). The index gives `take` bits to W's column index, from column bit cb0 up, and
// leaves `left` values to the row index. Of its z low bits, class 'S' gives the lowest, class 'P'
// the highest; class 'A' gives the entire index or nothing.
struct GrpW {
  const char *nm = "";
  size_t R = 1;
  char cls = 'S';
  int z = 0, take = 0, cb0 = 0;
  size_t left = 1;
};
// A unit: one layer's tensors of kind X, or the embedding or output head (listed by
// unit_tensors(), prover/weights_commitment/fold_source.cuh). nlw: base-256 digits per weight.
// The unit occupies `rows` rows of W from row `off`, in the columns below 2^ncb.
struct UnitW {
  int X = 0, l = 0, nlw = 2, ncb = 0;
  vector<GrpW> g;
  size_t rows = 0, off = 0;
};
// hdR: extent of kind 0's head index: query heads, key heads, zeros (k_cellsT(),
// prover/weights_commitment/kernels.cuh).
// NE: routed experts in a MoE layer (the parameter file's line `NE`); read only for MoE layers.
struct ShapeW {
  int moeMode = 1; // 1: odd layers are MoE layers; 0: no MoE layer
  int L = 0;
  size_t d = 0, dh = 0, hdR = 64, dkv = 0, dq = 0, f = 0, fe = 0, NE = 0, V = 0;
};
// W's layout, as layoutW() returns it: k = 2^lk columns, the units U, and the row counts m0 and
// mW (defined at layoutW()).
struct LayW {
  int lk = 0;
  size_t k = 0, m0 = 0, mW = 0;
  vector<UnitW> U;
};
// ceil(log2 R)
static int clgW(size_t R) {
  int b = 0;
  while (((size_t)1 << b) < R) b++;
  return b;
}
// R must not be 0 (not checked).
static int tzW(size_t R) {
  int z = 0;
  while (!((R >> z) & 1)) z++;
  return z;
}
// Shares out up to lk column bits among u's groups, in the listed order.
static void lay_unitW(UnitW &u, int lk) {
  int left = lk, cb = 0;
  for (auto &g : u.g) {
    g.z = g.cls == 'A' ? 0 : tzW(g.R);
    g.take = 0;
    g.cb0 = cb;
    if (g.cls == 'A') {
      const int b = clgW(g.R);
      if (b <= left) g.take = b;
      g.left = g.take || g.R == 1 ? 1 : g.R;
      if (g.R == 1) g.take = 0;
    } else {
      g.take = std::min(g.z, left);
      g.left = g.R >> g.take;
    }
    left -= g.take;
    cb += g.take;
  }
  u.ncb = cb;
  u.rows = u.nlw;
  for (auto &g : u.g) u.rows *= g.left;
}
static bool moeW(int l) {
  return l & 1;
}
// Places the units over 2^lk columns; the embedding and the output head (X = 12, 13) only if eh
// is nonzero. m0: the sum of the units' rows, rounded up to a multiple of Dd. mW = m0 + rR * Dd;
// the programs pass rR = 0 (HfShape::from_policy(), prover/hash_commitment/hf_shape.cuh;
// member_layout_of_W(), prover/prover/request_member.cuh).
static LayW layoutW(const ShapeW &s, int lk, int Dd, int rR = 2, int eh = 1) {
  LayW Y;
  Y.lk = lk;
  Y.k = (size_t)1 << lk;
  auto add = [&](int X, int l, int nlw, vector<GrpW> g) {
    UnitW u;
    u.X = X;
    u.l = l;
    u.nlw = nlw;
    u.g = g;
    lay_unitW(u, lk);
    Y.U.push_back(u);
  };
  auto G = [](const char *n, size_t R, char c) {
    GrpW g;
    g.nm = n;
    g.R = R;
    g.cls = c;
    return g;
  };
  for (int l = 0; l < s.L; l++) {
    const bool M = s.moeMode && moeW(l);
    add(0, l, 2, {G("hd", s.hdR, 'A'), G("cd", s.dh, 'A'), G("kk", s.d, 'S')});
    add(1, l, 2, {G("kk", s.d, 'S'), G("x", s.dkv, 'P')});
    add(2, l, 2, {G("kk", s.dq, 'S'), G("x", s.d, 'P')});
    if (!M) {
      add(3, l, 2, {G("kk", s.d, 'S'), G("x", s.f, 'P')});
      add(4, l, 2, {G("kk", s.d, 'S'), G("x", s.f, 'P')});
    } else {
      add(5, l, 2, {G("kk", s.d, 'S'), G("x", s.fe, 'P')});
      add(6, l, 2, {G("kk", s.d, 'S'), G("x", s.fe, 'P')});
      add(7, l, 2, {G("kk", s.d, 'S'), G("e", s.NE, 'S'), G("x", s.fe, 'P')});
      add(8, l, 2, {G("kk", s.d, 'S'), G("e", s.NE, 'S'), G("x", s.fe, 'P')});
    }
    add(9, l, 2, {G("kk", M ? s.fe : s.f, 'S'), G("x", s.d, 'P')});
    if (M) {
      add(10, l, 2, {G("kk", s.fe, 'S'), G("e", s.NE, 'S'), G("x", s.d, 'P')});
      add(11, l, 2, {G("kk", s.d, 'S'), G("x", s.NE, 'P')});
    }
  }
  if (eh) {
    add(12, s.L, 4, {G("v", s.V, 'A'), G("x", s.d, 'P')});
    add(13, s.L, 2, {G("x", s.V, 'A'), G("kk", s.d, 'S')});
  }
  size_t off = 0;
  for (auto &u : Y.U) {
    u.off = off;
    off += u.rows;
  }
  Y.m0 = (off + Dd - 1) / Dd * Dd;
  Y.mW = Y.m0 + (size_t)rR * Dd;
  return Y;
}
// A claim's factor on one group. Classes 'S' and 'P' (R = o * 2^z, o odd): at index hi * 2^z + lo
// its value is HV[hi] * eq(pt, lo); HV has o elements, pt has z; pt[i] is the coordinate of lo's
// bit z - 1 - i. Class 'A': at index i its value is AV[i] (R elements, read only if `take` is 0).
struct FacW {
  vector<E> HV, pt, AV;
};
// A claim on unit LayW::U[ui]: a sum of the unit's weights, each multiplied by the factors f[i]
// at its indices.
struct ClaimW {
  int ui = 0;
  vector<FacW> f;
};
static E eq1_W(E z, int bit) {
  return bit ? z : esub(E1, z);
}
// Claim a's coefficients on u's rows. Coefficient r (W's row u.off + r) is 256^j (j: the
// row's digit) times each group's factor at the part of its index that the row fixes. In r,
// g[0]'s part varies fastest, the digit slowest.
// Not checked: the lengths of a.f and its vectors.
static vector<E> gammaW(const UnitW &u, const ClaimW &a) {
  vector<E> t(1, E1);
  {
    vector<E> n;
    u64 p = 1;
    for (int l = 0; l < u.nlw; l++) {
      n.push_back(ebase(p));
      p *= 256;
    }
    t = n;
  }
  for (size_t i = u.g.size(); i-- > 0;) {
    const GrpW &g = u.g[i];
    const FacW &f = a.f[i];
    vector<E> w(g.left, E1);
    if (g.cls == 'A') {
      if (!g.take)
        for (size_t x = 0; x < g.left; x++) w[x] = f.AV[x];
    } else {
      const int rem = g.z - g.take;
      for (size_t x = 0; x < g.left; x++) {
        E r = f.HV[x >> rem];
        for (int b = 0; b < rem; b++) {
          const int lob = g.cls == 'S' ? g.take + b : b;
          r = emul(r, eq1_W(f.pt[g.z - 1 - lob], (x >> b) & 1));
        }
        w[x] = r;
      }
    }
    vector<E> n(t.size() * g.left);
    for (size_t q = 0; q < t.size(); q++)
      for (size_t x = 0; x < g.left; x++) n[q * g.left + x] = emul(t[q], w[x]);
    t.swap(n);
  }
  return t;
}
// Nothing uses MergeW.
struct MergeW {
  vector<E> rho, rm, co, T;
  E fv = E0;
  u32 sm[SW] = {0};
};
