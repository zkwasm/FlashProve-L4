// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The GPU prover's state for the layer sub-proofs (`fork` in identifiers), each the argument for
// one part of the forward pass: one Fork for each layer group (Prover::FK), set up by
// Prover::setup_g_forks(). The sub-proofs share many GPU buffers. A "slot" indexes one masked
// element of Prover::MSG; a "call" is a lookup call; `T` (prover/hiding/transcript.cuh) is an
// extension-field element.
#pragma once
#include "prover/hidden_kernels.cuh"
#include "head/stack.cuh"
#include "prover/plan_rounds.h"
// The type of Fork::h. S: the slots' mask entries. W: the linear claims. U[3] ("U3" elsewhere): the
// committed array of the reads (vec[3], nv[3] elements), inner products of committed cells with
// challenge-dependent vectors; no other index of U, vec, nv or LU is used. HT: the approximate
// range check on the cells ("smallness test" in the plan's names).
struct HFk {
  int NCL = 0, NV = 0;
  size_t slTS = 0, slPL2 = 0, slVE = 0, slMSC = 0;
  size_t nEndT = 0, nEndS = 0; // how many end values Fork::ts and the batched sumcheck (GS) send
  // For plan_rounds(); `where` is the sub-proof's name in the plan (blk::Plan: the rounds of a
  // protocol run). The verifier's counterpart: HPlan2::rounds (verifier/src/vf_layers_plan.inc).
  blk::LookupSize look;
  blk::LayerRounds own;
  std::string where;
  std::vector<RowJob> job2[2];
  gs::Fork GS;
  HiddenTail HT;
  hz::Tree S;
  hz::TreeN U[5];
  hw::Walk W;
  // slPT: the first slot of the calls' lookup; slPZ: of the range check's lookup.
  size_t ns = 0, nmsg = 0, slPT = 0, slVW = 0, slWPL = 0, slPV = 0, slTV = 0, slPL = 0, slPZ = 0,
         slSCH = 0, nv[5] = {};
  T *vec[5] = {}, *sc = nullptr,
    *rrS = nullptr /* the batched sumcheck's challenges, in the order of its rounds */;
  // roots (at word 3 * DW, for U[3]) and zs (for ZZ and BZ, see plan_rounds()) each receive DW
  // words, a digest's length, in place of a root. They are zeros: the code that commits to these
  // arrays returns no root to a sub-proof (ch::Host::words, prover/hiding/moments.cuh).
  u32 *roots = nullptr, *zs = nullptr;
  int lamS = 0;
  he::Eqs EQ;
  hd::Door door; // the claims' one equation, the argument of `hd::close` (prover/hiding/door.cuh)
  // Coefficients of that equation, cut from Lall (nLall elements): LU[3] on the reads, Lrow[r] on
  // the range check's virtual row r (HiddenTail::vrows: a combination of a committed array's rows).
  std::vector<T *> Lrow;
  T *LU[5] = {}, *Lall = nullptr, *one = nullptr;
  size_t nLall = 0, oLate = 0;
  int nu = 0;
  size_t R2 = 0;
  std::vector<T *> dst;
  struct XT {
    int ln = 0, t = 0, w = 0;
    size_t n = 0;
    T *bet = nullptr;
  };
  u32 *live = nullptr;
  // Adds to p the rounds between commit point `moment` and the next. 0, the first: the calls'
  // lookup and the sub-proof's own rounds. 1, the reads: the range check's random matrix. 2, its
  // arrays ZZ (digits) and BZ (multiplicity bits): its lookup on the digits, then the challenge
  // whose powers batch the claims.
  void plan_rounds(blk::Plan &p, unsigned moment) const {
    if (moment == 0) {
      blk::plan_lookup(p, where + ", look-up", look);
      blk::plan_layer_own(p, where, own);
      return;
    }
    HT.plan_rounds(p, where + ", smallness test", moment - 1);
    // Further powers of that challenge batch the range check's equations.
    if (moment == 2) blk::plan_bat_sigma(p, where + ", claims", HT.batched_rows(W.nclaim));
  }
};
// n matrices of Fork::M, the matrices of committed cells, from M[first] on.
// WeightKind::lm uses only n.
struct LM {
  int first = -1, n = 0;
};
struct WeightKind {
  int kind;
  LM lm;
  size_t L, bytes;
  i8 *buf;
  T *phi, *tab;
  size_t ntab;
  std::vector<RowJob> jobs;
};
// TSC(zt): the sumcheck over tokens at zt = Fork::zx_last(lt), for the calls in Prover::tscCalls:
// sum_t eq(zt, t) sum_m co[m] Fv[m][t] Gv[m][t], m < NTm. Fv, Gv: a term's first and second factor.
struct TokenSumcheck {
  Tsc eng;
  TscEnd end; // its last round, in an extra variable Y for the end values' masks, then these values
  int NTm;
  T *Fv, *Gv, *co, *msg, *yb;
};
struct Fork {
  T *tkh = nullptr, *ZE = nullptr, *pk = nullptr, *phiE = nullptr;
  std::vector<RowJob> jobE;
  ForwardPass g;
  std::vector<MatD> M;
  LM af, ad, ar, b, qa, xg, xn;
  // The filler cells (padding) at the end of the buffers that hold af, ad, ar and qa: first cell
  // and count. All layer sub-proofs share these four buffers, so every Fork's list names the same
  // cells. A request gets no proof unless each run finds them all 0 (Prover::finish_request()).
  std::vector<std::pair<const i8 *, size_t>> filler;
  std::vector<WeightKind> wk;
  int NR;
  // One object each for all layer sub-proofs. F: the calls' lookup forest
  // (prover/hiding/forest.cuh), which runs them as one lookup. FG: its cached CUDA graphs. Product
  // layers outside the lookups: WP, the norms' sums of squares; HL, the feed-forward blocks' gated
  // products; HL2, the routed experts' alone (null if the model has none).
  Forest *F = nullptr;
  PtGraph *FG = nullptr;
  PtPL *HL, *HL2 = nullptr;
  PtWPL *WP;
  Tsc tV;
  TscEnd tVend;
  std::vector<TokenSumcheck> ts; // none or one
  // The end values of TSC(zt)'s second factors that the batched sumcheck reads as scalars: GPU
  // addresses. The map's key is the name of one of a call's lists of lookup keys, as on the call's
  // line in the parameter file ("N1", "N2" or "XR"; `seg` in identifiers), then a number: for N1
  // and N2 the layer's index lam in its group, for XR lam / 2.
  std::map<std::string, const T *> kept;

  u32 *pub = nullptr; // each tree's number of padding keys, by tree number
  std::vector<std::vector<size_t>> padded; // per call (host): what `forest_padded_from` returns
  int n[MAX_CALLS], lJ[MAX_CALLS];
  T *sPT, *sWT, *sVW, *sHV, *sWPL, *sPV, *sTV, *sHH, *sPL, *sW,
      *sHH2 = nullptr, *sPL2 = nullptr, *rhoH2 = nullptr, *etD2 = nullptr, *ewX = nullptr;
  u16 *KH2 = nullptr;

  size_t nWT, nW;
  std::vector<RowJob> jobs, jobsB;
  size_t short_jobs_from = 0; // `jobs` from this index on make the tables at token points (`SPn`)
  std::map<std::string, T *> phiP;
  std::vector<std::pair<std::string, int>> LPn, SPn;
  // el[c]: eq(zk(c)'s first ll coordinates, .). ele[c]: el[c] times ForestClaims::e[c], the call's
  // weight in the forest's join.
  T *phiB[3], *ec[MAX_CALLS], *el[MAX_CALLS], *ele[MAX_CALLS], *eV, *etD, *PV, *gam, *rhoV, *rhoH, *omV, *A, *FV,
      *GV, *gnp, *ewN[2], *AH, *lfc, *ehq, *Cm, *halves /* 2^-lPd */,
      *start /* the sum that a product layer outside the lookups claims */;
  // g_token_sumchecks()'s int8 matrix products: Cg[i] (32-bit sums) = aT[i] (byte limb i of an
  // integer matrix) times Bl (field elements' signed byte limbs). i > 1: only with experts.
  i8 *Bl, *aT[4];
  int *Cg[4];
  char *blas_room = nullptr; // cuBLAS's workspace, valid only from `middle_begin` to `late_begin`
  char *zfold_work = nullptr; // for the range check's combinations; valid only from `late_begin` on
  long long *Ws;
  std::string lists;
  HFk *h = nullptr;
  const T *zk(int c) const { return F->claims->zk[c]; }
  // zk(c), call c's evaluation point, has ll layer coordinates, then lJ[c] column coordinates (from
  // zj(c)), then lt token coordinates (from zt(c)). Every zk(c) ends with ForestClaims::zx, so all
  // calls share zx's last lt coordinates, the point zt ("z" in SPn), and its last la (Prover::la)
  // coordinates, the point "Z" in LPn.
  const T *zx_last(int coordinates) const { return F->claims->zx + (F->plan.dT - coordinates); }
  auto zj(int c) const { return zk(c) + g.L.s.ll; }
  auto zt(int c) const { return zj(c) + lJ[c]; }
};
// Launch with one thread.
__global__ void k_sety(u32 *hdr, int at, const int *y) {
  hdr[at] = (u32)*y;
}
#include "head/hidden_fork.cuh"
static double epoch() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
