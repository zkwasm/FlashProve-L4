// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* How `cp_verify` plans and checks the segments of a request's proof. A segment (`link` in names
 * and output) is one sub-proof's (`fork`) part of the proof's first hash chain (Fiat-Shamir
 * transcript). The order is G_0, A_0, G_1, A_1, ..., H (G_g: layer group g; A_g: its attention; H:
 * the output head). Format: docs/format/FORMAT.md, 13.4, steps 5 to 7. The frame
 * (`blk::ReqVerifier`, blocks/) handles commitments, samples, positions, closings and the last
 * proximity test. Included inside `runStage` (cp_verify.cpp); uses its locals. buildLinksH makes
 * the plan from the sub-proofs' plans, `layout.txt` and `policy.txt`. runLinksH checks each
 * segment: the frame's first call, then the sub-proof's function, which calls the frame at each
 * later commit point (`moment` in names) via `rootInH` and at its end via `doorH`
 * (vf_subproof_segment.h). */
// One segment. tl: approximate range check. Fp->trees: arrays; Tp->nt: more on each.
struct LinkH {
  string name;
  u32 gid = 0; // the sub-proof's public identifier
  int kind = 0; // 0 layers, 1 attention, 2 the head
  u64 g = 0;
  const HTailF2 *tl = nullptr;
  const HFork *Fp = nullptr;
  const Tail2 *Tp = nullptr;
  u32 ns = 0; // scalars the sub-proof sends: it uses that many entries of the segment's mask row
  bool fv = false; // has a pending claim: a linear claim (`row`) on the weight table, tested later
  const std::function<void(blk::Plan &, unsigned)> *rounds = nullptr; // rounds after a commit point
  // arrayOf[t]: the segment that commits to Fp->trees[t], and its name there.
  vector<std::pair<size_t, string>> arrayOf;
};
vector<LinkH> linksH;
blk::Subproofs subH;      // what the frame asks of the sub-proofs
blk::Sized sizedH;        // sizes and codes, from subH and the parameter file
blk::ReqLayout layoutH;   // where each array's elements lie
// blk::configure()'s result: sizes, every chain's plan, the configuration files' digest.
std::unique_ptr<blk::Configured> confH;
// Prints `name: why`; refuse() (vf_common.h) then prints `NO: name` and exits.
auto refuseWhyH = [&](const char *name, const std::string &why) {
  printf("%s: %s\n", name, why.c_str());
  refuse(name);
};
// Builds linksH and, from it, the four objects above. Returns false after printing the list of the
// segments' arrays, if the call asked for it (`-plan arrays`, or `-plan request` with `-arrays`).
auto buildLinksH = [&]() -> bool {
  for (u64 g = 0; g < NGr; g++) {
    LinkH a;
    a.name = "G" + std::to_string(g);
    a.gid = (u32)(1 + g);
    a.kind = 0;
    a.g = g;
    a.tl = &hidPlans2[g].tl;
    a.Fp = &hidPlans2[g].Fp;
    a.Tp = &hidPlans2[g].Tp;
    a.ns = hidPlans2[g].ns;
    a.fv = true;
    a.rounds = &hidPlans2[g].rounds;
    linksH.push_back(a);
    LinkH b;
    b.name = "A" + std::to_string(g);
    b.gid = (u32)(2 + NGr + g);
    b.kind = 1;
    b.g = g;
    b.tl = &AH2[g].seam.tl;
    b.Fp = &AH2[g].seam.Fp;
    b.Tp = &AH2[g].seam.Tp;
    b.ns = AH2[g].ns;
    b.fv = false;
    b.rounds = &AH2[g].rounds;
    linksH.push_back(b);
  }
  {
    LinkH c;
    c.name = "H";
    c.gid = (u32)(1 + NGr);
    c.kind = 2;
    c.tl = &PH2.tl;
    c.Fp = &PH2.Fp;
    c.Tp = &PH2.Tp;
    c.ns = PH2.ns;
    c.fv = true;
    c.rounds = &PH2.rounds;
    linksH.push_back(c);
  }
  // Exactly one segment is the head's, and it is the last: the frame takes the next token (`reply`)
  // right before the last segment's first message. All sub-proofs' identifiers differ.
  {
    size_t heads = 0;
    std::set<u32> gids;
    for (size_t i = 0; i < linksH.size(); i++) {
      if (linksH[i].kind == 2) {
        heads++;
        if (i + 1 != linksH.size()) refuse("plan-head");
      }
      if (!gids.insert(linksH[i].gid).second || !linksH[i].rounds || !*linksH[i].rounds) refuse("plan-link");
    }
    if (heads != 1u) refuse("plan-head");
  }
  const size_t NL = linksH.size();
  if (NL < 2 || NL > 64) refuse("plan-shape");
  auto layersLink = [&](u64 g) {
    return (size_t)(2 * g);
  };
  /* Each segment's arrays by commit point, named and ordered as the prover declares them
   * (prover/prover/chain_frame.cuh, chain_moments.cuh) and as the layout file numbers a point's
   * arrays. Names and order (pkMat: read by no other sub-proof; pxMat: read by others):
   * docs/format/FORMAT.md, 13.4, step 6. */
  vector<blk::FrameLink> declared;
  for (size_t l = 0; l < NL; l++) {
    LinkH &X = linksH[l];
    const HTailF2 &tl = *X.tl;
    X.arrayOf.assign(X.Tp->nt.size(), {~(size_t)0, string()});
    blk::FrameLink D;
    D.name = X.name;
    D.moment.emplace_back();
    auto names = [&](int tree, size_t born, const string &name) {
      if (tree < 0 || (size_t)tree >= X.arrayOf.size() || !X.arrayOf[(size_t)tree].second.empty()) refuse("plan-link");
      X.arrayOf[(size_t)tree] = {born, name};
    };
    auto own = [&](int tree, const string &name) {
      names(tree, l, name);
      blk::FrameArray a;
      a.name = name, a.rows = X.Fp->trees[(size_t)tree].R, a.columns = X.Fp->trees[(size_t)tree].K;
      a.over_E = X.Tp->nt[(size_t)tree].ec == (u32)CE;
      D.moment.back().push_back(a);
    };
    if (tl.pkMat.empty() || tl.ix.PK.size() != tl.pkMat.size() || tl.ix.TX.size() != tl.pxMat.size() || tl.ix.FT.size() != tl.plFor.size())
      refuse("plan-link");
    for (size_t i = 0; i < tl.pkMat.size(); i++) own(tl.ix.PK[i], "M" + std::to_string(tl.pkMat[i]));
    for (size_t i = 0; i < tl.pxMat.size(); i++) own(tl.ix.TX[i], "M" + std::to_string(tl.pxMat[i]));
    for (int n = 0; n < 4; n++)
      if (tl.ix.U[n] >= 0) {
        D.moment.emplace_back();
        own(tl.ix.U[n], "U" + std::to_string(n + 1));
      }
    D.moment.emplace_back();
    own(tl.ix.ZZ, "ZZ");
    own(tl.ix.BZ, "BZ");
    /* Matrices that an earlier sub-proof, the owner, committed to. The reader's record `pl` (the
     * plans compared it with the owner's) has the owner's number among the layer sub-proofs and the
     * matrix's index among its "M" arrays. */
    for (size_t i = 0; i < tl.ix.FT.size(); i++) {
      const PlaceF &pl = tl.plFor[i];
      if (pl.owner < 0 || (u64)pl.owner >= NGr || layersLink((u64)pl.owner) >= l) refuse("plan-fold-source");
      const size_t born = layersLink((u64)pl.owner);
      const HTailF2 &O = *linksH[born].tl;
      if (pl.rootIndex < (int)O.pkMat.size() || (size_t)pl.rootIndex >= O.pkMat.size() + O.pxMat.size()) refuse("plan-fold-source");
      const size_t mu = O.pxMat[(size_t)pl.rootIndex - O.pkMat.size()];
      const HTree &mine = X.Fp->trees[(size_t)tl.ix.FT[i]], &theirs = linksH[born].Fp->trees[(size_t)O.gtTree[mu]];
      if (mine.R != theirs.R || mine.K != theirs.K) refuse("plan-fold-source");
      names(tl.ix.FT[i], born, "M" + std::to_string(mu));
      // Kept up to this reader: blk::req_frame() requires it of the layout file.
      bool found = false;
      for (blk::FrameArray &a : declared[born].moment[0])
        if (a.name == "M" + std::to_string(mu)) a.life = std::max<unsigned>(a.life, (unsigned)(l - born)), found = true;
      if (!found) refuse("plan-fold-source");
    }
    for (auto &a : X.arrayOf)
      if (a.second.empty()) refuse("plan-link");
    declared.push_back(std::move(D));
  }
  /* For prover/tools/pack_layout.py --arrays. Fields: verifier/docs/plan-data.md, "`array:`"; group
   * 1 holds the arrays that no later segment reads. */
  if (in.arraysAsked) {
    for (size_t l = 0; l < NL; l++)
      for (size_t m = 0; m < declared[l].moment.size(); m++) {
        unsigned kept = 0;
        for (size_t a = 0; a < declared[l].moment[m].size(); a++) {
          const blk::FrameArray &A = declared[l].moment[m][a];
          printf("array: %zu %zu %u %c %u %zu %llu %llu %c %s\n", l + 1, m + 1, A.life ? 1 + ++kept : 1, A.life ? 'B' : A.over_E ? 'W' : 'N', A.life, a,
                 (unsigned long long)A.rows, (unsigned long long)A.columns, A.name[0] == 'M' ? 'r' : 'c', A.name.c_str());
        }
      }
    return false;
  }
  // No default layout exists; prover/tools/pack_layout.py writes one.
  if (!layoutP) refuse("plan-layout");
  blk::LayoutLines lines;
  if (const char *fault = blk::layout_read(*layoutP, lines)) refuseWhyH("plan-layout", fault);
  if (const char *fault = blk::req_frame(declared, lines, PS.frame.answers, subH.request, layoutH)) refuseWhyH("plan-layout", fault);
  subH.request.link.back().plain = 1; // one unmasked scalar: the next token
  subH.open_rows.assign(NL, 1);       // a sub-proof ends in one equation
  subH.plan = [&](blk::Plan &p, size_t link, unsigned moment) { (*linksH.at(link).rounds)(p, moment); };
  size_t onTable = 0;
  for (auto &X : linksH) onTable += X.fv;
  if (onTable != NGr + 1) refuse("plan-link");
  /* For the committer's range proof (`wf`). Of a four-limb unit (`nlw`) it bounds bytes, not
   * numbers: only the embedding (X = 12) may be one. */
  for (const WsUnit &u : wsLay.U)
    if (u.nlw == 4 && u.X != 12) refuse("internal-weight-limbs");
  if (const char *fault = blk::wf_layout_fault(wsLay)) refuseWhyH("internal-weight-limbs", fault);
  subH.ranges = blk::wf_shape_of_layout(wsLay, PS.frame.block_log); // log2(weight matrix's columns)
  /* The key file's range proof must show what a request assumes: a weight's width (WEIGHT_WIDTH_BITS, WEIGHT_BOUND: the
   * no-overflow conditions), and weight = low + WS_SECOND_DIGIT * high limb (`wsGammaAdd`). */
  static_assert(WEIGHT_WIDTH_BITS == blk::WF_WEIGHT_BITS && WEIGHT_BOUND == 1u << (blk::WF_WEIGHT_BITS - 1) && WS_SECOND_DIGIT == blk::WF_SECOND_DIGIT && blk::WF_BYTE_BITS == 8,
                "a weight's width in the request's conditions is not the width that a key proves");
  subH.pending_rows = onTable;
  sizedH = blk::sized(PS.frame, subH);
  if (!sizedH.refused.empty()) refuseWhyH("parameters-not-admitted", sizedH.refused);
  if (const char *fault = blk::req_layout_fault(sizedH.proof_codes, sizedH.request, layoutH)) {
    // Prints each array whose blocks differ in size from the plan.
    for (size_t l = 0; l < NL; l++)
      for (const blk::ReqArray &a : layoutH.link[l]) {
        u128 held = 0;
        for (const blk::LaidBlock &b : a.laid.blocks) held += (u128)1 << (b.bits_i + b.bits_c);
        if (held != (u128)a.laid.size_i * a.laid.size_c)
          printf("plan-layout: link %zu (%s), array %s: %llu rows of %llu columns by the plan; the file's %zu blocks hold %llu elements\n", l + 1,
                 linksH[l].name.c_str(), a.name.c_str(), (unsigned long long)a.laid.size_i, (unsigned long long)a.laid.size_c, a.laid.blocks.size(),
                 (unsigned long long)held);
      }
    refuseWhyH("plan-layout", fault);
  }
  // Checked here: where one of these fails, blocks/ ends the process, and a file's content can
  // cause that. The index-mask message (the pending claims' masks) has a free entry, counted as in
  // blk::Send::free_mask_entries(), for each claim and for what batching them takes. Each segment's
  // mask row has entries for its scalars; each array that it reads is in the layout and still kept.
  {
    const blk::OracleShape &s = sizedH.batch.input.shape;
    const size_t ell = sizedH.proof_codes.mask.at(s.mask_code).ell;
    if (!s.mask_rows || ell < s.answers ||
        s.mask_rows * (ell - s.answers) + (s.mask_rows - 1) < onTable + (onTable > 1 ? blk::batt_masks(sizedH.batch.S.k_col()) : 0))
      refuse("plan-mask-row");
  }
  const size_t maskRow = sizedH.proof_codes.mask[sizedH.request.mask_code].ell;
  for (size_t l = 0; l < NL; l++) {
    if ((size_t)linksH[l].ns + sizedH.request.answers > maskRow) refuse("plan-mask-row");
    for (auto &a : linksH[l].arrayOf) {
      const blk::ReqArray *at = layoutH.find(a.first, a.second);
      if (!at || a.first + sizedH.request.link[a.first].messages[at->message].life < l) refuse("plan-layout");
    }
  }
  /* `blk::configure` takes the shapes from a plain function, `stageShapesOf` (cp_verify.cpp), which
   * cannot see this function's locals. So `made` passes through the file-scope stageShapesMade: set
   * before the call, cleared after; put no code between. One `runStage` at a time per process. */
  blk::Shapes made;
  made.sub = subH, made.layout = layoutH;
  for (auto &X : linksH) made.identifiers.push_back(X.gid);
  // The request's range checks, for the conditions for hiding that blk::configure() tests: one per
  // matrix of each segment, read from an earlier one or not. Sizes are in cells (signed bytes).
  for (auto &X : linksH)
    for (auto &M : X.tl->mats) {
      blk::RangeCheck c;
      c.rows = M.m, c.columns = M.k, c.combinations = (u64)TAU, c.bound = 129, c.range = (u64)1 << LBZ;
      c.digit_bits[0] = T2D0, c.digit_bits[1] = T2D1;
      made.range_checks.push_back(c);
    }
  if (stageShapesMade) refuse("internal-shapes-order");
  stageShapesMade = &made;
  confH.reset(new blk::Configured(blk::configure({polB, *layoutP, klBytesH}, stageShapesOf)));
  stageShapesMade = nullptr;
  if (!confH->refused().empty()) refuseWhyH("parameters-not-admitted", confH->refused());
  sizedH = confH->sized(); // so later code and `blk::check_proof` use one computation of the sizes
  return true;
};
// Returns one line per segment, for printing.
auto planLineH = [&]() {
  string o;
  char b[256];
  for (size_t l = 0; l < linksH.size(); l++) {
    const LinkH &X = linksH[l];
    size_t rowsF = 0, rowsE = 0, kept = 0, blocks = 0;
    for (const blk::ReqMessage &m : sizedH.request.link[l].messages) {
      (m.shape.main_deg == 1 ? rowsF : rowsE) += m.shape.main_rows;
      if (m.life) kept += m.shape.main_rows;
    }
    for (const blk::ReqArray &a : layoutH.link[l]) blocks += a.laid.blocks.size();
    snprintf(b, sizeof b,
             "link %zu: sub-proof %s, identifier %u; %u commitments; rows over F %zu (kept for later links %zu), over E %zu; arrays %zu in %zu blocks; "
             "%u scalars\n",
             l + 1, X.name.c_str(), X.gid, sizedH.request.link[l].moments(), rowsF, kept, rowsE, layoutH.link[l].size(), blocks, X.ns);
    o += b;
  }
  return o;
};
// One sub-proof's pending claim (vf_weights_pending_claim.h), masked by the index-mask message's next free entry.
auto pendingRowH = [&](blk::VerifierContext &V, blk::OracleId indexMasks, const WsForkOut &o, size_t claims, const blk::LinShape &S) {
  WsPending p = wsPending(wsLay, o, claims, S);
  return blk::member_pending_row(std::move(p.g), std::move(p.psi), V.st.masks(indexMasks).take(), p.beta, p.target);
};
/* Checks a proof's segments in turn, for `blk::check_proof` (blocks/frame/check.h). A sub-proof's
 * function ends the process on a fault (`refuse`). Returns false if the frame's `first` or `finish`
 * failed. */
double linksEndH = 0; // when `runLinksH` finished, for a printed timing
auto runLinksH = [&](blk::VerifierContext &V, const blk::RequestPublic &x, blk::LinksResult &out) -> bool {
  if (!namingDone) refuse("internal-naming-first"); // set once format and commitment are identified
  mb::ChainCheck check(V, mb::ChainShape{x.shape, x.layout});
  innerThreads = nthreads < 1 ? 1 : nthreads;
  for (size_t l = 0; l < linksH.size(); l++) {
    const LinkH &X = linksH[l];
    const double tFirst = now();
    if (!check.first()) return false;
    const double sFirst = now() - tFirst;
    if (!l) {
      if (check.frame().at_start().size() != 1) refuse("plan-link");
      out.index_masks = check.frame().at_start()[0];
      // The index-mask message needs a free entry per pending claim and those that batching them
      // takes. The number follows from the parameter file; blocks/ ends the process on too few.
      size_t rows = 0;
      for (auto &Y : linksH) rows += Y.fv;
      if (V.st.masks(out.index_masks).left() < rows + (rows > 1 ? blk::batt_masks(V.ch.get(x.lin.shape).k_col()) : 0)) refuse("plan-mask-row");
    }
    if (check.frame().plain().size() != (X.kind == 2 ? 1u : 0u)) refuse("plan-head");
    if (X.kind == 2) out.reply = check.frame().plain()[0], out.replied = true;
    LinkCtx C;
    C.check = &check;
    C.V = &V;
    C.link = l;
    C.arrayOf = &X.arrayOf;
    C.layout = &V.ch.get(x.layout);
    currentSegment = &C;
    const double t0 = now();
    if (X.kind == 0)
      runForkHidG2(X.g);
    else if (X.kind == 2) {
      runForkHidH2();
      forkRan[NGr] = 1;
    } else {
      AH2[X.g].run("A" + std::to_string(X.g) + ":");
      forkRan[NGr + 1 + X.g] = 1;
      char ln[256];
      snprintf(ln, sizeof ln, "read: sub-proof A_%llu: %u claims listed (tested at the chain's end), %u slots  %.3f s\n", (unsigned long long)X.g, AH2[X.g].nClaim, AH2[X.g].ns, now() - t0);
      forkLine[NGr + 1 + X.g] = ln;
    }
    currentSegment = nullptr;
    if (!C.handed || !C.done) refuse("plan-link"); // the function did not close its segment
    if (X.fv)
      out.pending.push_back(pendingRowH(V, out.index_masks, forkOut[X.kind == 2 ? NGr : X.g], X.kind == 2 ? PH2.NCL : hidPlans2[X.g].NCL,
                                        V.ch.get(x.lin.shape)));
    const double sAll = now() - t0;
    if (!quiet)
      printf("%s   its equation: %zu terms of rank one, %zu written-out weights in all; %zu virtual rows had no product form and were cut row by row\n"
             "   seconds: the frame's first() %.3f | the sub-proof's own checks %.3f | the frame's later() %.3f | the equation's terms made %.3f | the frame's close() %.3f\n",
             forkLine[X.kind == 0 ? X.g : X.kind == 2 ? NGr : NGr + 1 + X.g].c_str(), C.terms, C.written, C.rowsCut, sFirst, sAll - C.sLater - C.sDoor - C.sClose, C.sLater, C.sDoor,
             C.sClose);
  }
  const double tFinish = now();
  const bool finished = check.finish();
  linksEndH = now();
  if (!quiet) printf("seconds: the links' last proximity test (the frame's finish()) %.3f\n", linksEndH - tFinish);
  return finished;
};
