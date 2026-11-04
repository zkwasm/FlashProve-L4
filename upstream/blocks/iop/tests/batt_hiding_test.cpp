// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// An exact test, at toy size: do the scalars sent while a request's claims on the committed matrix
// are batched reveal anything about the matrix? Each segment of the proof (`link` in identifiers)
// batches its J claims (`rows`) into one by BATT with the end value sent (blocks/iop/batt_shown.h),
// masked by an entry c of the request's index-mask message (its oracle message of masks for such
// claims); BATT (blocks/iop/batt.h) batches several segments' claims into one. No verifier, honest
// run or comparison of bytes can detect a mask entry (`coins`: a random element of the field E,
// added into sent scalars) that was not drawn, was drawn once for two uses, or is missing. This
// test can.
// Covered, if the matrix has 2^k_col columns: each segment's J masked claim values (`target`) and
// 3 + 2 k_col scalars; BATT's 1 + 2 k_col. Not covered: the hash chain (challenges that depend on
// the messages); roots, samples' answers, openings, the oracle message W (blocks/iop/lin.h) and all
// that follows; whether a program draws its mask entries uniformly; the prover `cp_prove`, which
// does not call batt_shown_prove() (only tests and blocks/frame/handover_proof.cpp, a tool that
// writes toy proof files, do).
#include <functional>

#include "engine/sumcheck_cpu.h"
#include "field/field.h"
#include "iop/batt_shown.h"
#include "toy_support.h"

using namespace blk;

// With the verifier's messages fixed, what the prover sends is an affine map of its mask entries:
//     sent = A coins + b(matrix).
// A is found column by column; that the map is affine is tried at one random point. If the rank of
// A equals the number of scalars, the scalars are uniform over all tuples whatever the matrix is:
// they reveal nothing. If not, matrices M and M' can be distinguished as soon as b(M) - b(M') is
// outside the span of A's columns; each case checks this for one random pair.
// In the plain cases, every case must show full rank and as many scalars as mask entries, and the
// verifier's functions must accept and return true claims.
// Three negative controls set a fault in the prover's mask entries. The verifier accepts as before
// (that is the point); a control passes only if it does, the rank is short and two matrices are
// distinguished, and the program returns 0 only if all three pass.
enum Control {
  NONE,  // the plain cases
  // segments 0 and 1 have one value at their entries c; cases of one segment are skipped
  ONE_C_FOR_TWO,
  // segment 0's hiding polynomial has no linear term in the extra variable Y (blocks/iop/psc.h):
  // Y's round sends c times the challenge eps, unmasked
  NO_Y_COEFFICIENT,
  // segment 0 is proved again, with fresh masks of its own and other challenges, under the entry c
  // that it has used
  LINK_RUN_AGAIN
};
static const char *const CONTROL_WORDS[] = {"", "one value at two segments' entries c", "no linear term in Y", "a segment proved again under its entry c"};

static E random_E(TestRng &rng) { return E::from(from_u64(rng.word()), from_u64(rng.word()), from_u64(rng.word())); }
static std::vector<E> random_Es(TestRng &rng, size_t n) {
  std::vector<E> v(n);
  for (E &x : v) x = random_E(rng);
  return v;
}

// ---- a channel that takes the verifier's messages from a list ----
class Listed {
 public:
  explicit Listed(const std::vector<E> &challenges) : list_(challenges) {}
  struct Draw {
    Listed *ch;
    void require_rows(size_t) {}
    E draw() { return ch->next(); }
    E draw_nonzero() {
      const E x = ch->next();
      BLK_REQUIRE(!(x == E::zero()), "the test's list holds 0 where 0 is not allowed");
      return x;
    }
  };
  E close_epsilon(const std::string &) { return next(); }
  Draw close_algebraic(const std::string &) { return Draw{this}; }
  void close_empty(const std::string &) {}
  size_t used() const { return at_; }

 private:
  E next() {
    BLK_REQUIRE(at_ < list_.size(), "the test's list of challenges is too short");
    return list_[at_++];
  }
  const std::vector<E> &list_;
  size_t at_ = 0;
};
class ListedProver : public Listed {
 public:
  using Listed::Listed;
  void send(const E &x) { tape.push_back(x); }
  void send_hiding_sum(const E &x) { tape.push_back(x); }
  std::vector<E> tape;
};
class ListedVerifier : public Listed {
 public:
  ListedVerifier(const std::vector<E> &challenges, const std::vector<E> &tape) : Listed(challenges), tape_(tape) {}
  bool recv(E &x) {
    if (at_ >= tape_.size()) return good_ = false;
    x = tape_[at_++];
    return true;
  }
  bool recv_hiding_sum(E &x) { return recv(x); }
  bool ok() const { return good_; }
  bool at_end() const { return at_ == tape_.size(); }

 private:
  const std::vector<E> &tape_;
  size_t at_ = 0;
  bool good_ = true;
};

// ---- the case ----
// The matrix has `matrix_rows` rows; their weights `g` have k_row variables.
struct Case {
  unsigned k_row, k_col;
  size_t matrix_rows, links, rows;  // rows: J, in each link
};
using Matrix = std::vector<std::vector<E>>;
struct Forms {  // public: the same for every matrix
  Case c;
  Control control = NONE;
  std::vector<std::vector<Tensor>> g;  // by link * J + j
  std::vector<std::vector<E>> psi;
  std::vector<E> challenges;
  size_t runs() const {
    if (control == LINK_RUN_AGAIN) return c.links + 1;  // the last run is link 0 once more
    return c.links;
  }
  size_t link_of(size_t run) const { return run < c.links ? run : 0; }
  size_t own_coins() const { return c.rows + batt_shown_own_masks(c.k_col); }
  size_t index_coins() const { return c.links + (c.links > 1 ? batt_masks(c.k_col) : 0); }
  size_t coins() const { return index_coins() + runs() * own_coins(); }
};
static Matrix random_matrix(const Case &c, TestRng &rng) {
  Matrix M;
  for (size_t y = 0; y < c.matrix_rows; y++) M.push_back(random_Es(rng, size_t(1) << c.k_col));
  return M;
}
// W = g^T M
static std::vector<E> weights_times_matrix(const Case &c, const Matrix &M, const std::vector<Tensor> &g) {
  std::vector<E> w(size_t(1) << c.k_col, E::zero());
  for (size_t y = 0; y < c.matrix_rows; y++) {
    E a = E::zero();
    for (const Tensor &t : g) a += t.at(y);
    for (size_t col = 0; col < w.size(); col++) w[col] += a * M[y][col];
  }
  return w;
}
// Splits `coins` into mask tables: table 0 is the index-mask message's (an entry c for each
// segment, then BATT's masks), table 1 + r is run r's own mask row (its J claim values' masks, then
// the hiding polynomial's coefficients in hp_place()'s order: blocks/iop/hp.h).
static std::vector<std::vector<E>> tables(const Forms &f, const std::vector<E> &coins) {
  std::vector<std::vector<E>> t(1 + f.runs());
  size_t at = 0;
  t[0].assign(coins.begin(), coins.begin() + f.index_coins()), at = f.index_coins();
  for (size_t r = 0; r < f.runs(); r++, at += f.own_coins()) t[1 + r].assign(coins.begin() + at, coins.begin() + at + f.own_coins());
  if (f.control == ONE_C_FOR_TWO) t[0][1] = t[0][0];
  if (f.control == NO_Y_COEFFICIENT) t[1][f.c.rows + 2 * f.c.k_col + 1] = E::zero();  // the last coefficient: the linear one of Y
  return t;
}
// One `MaskLedger` (it gives out each entry once) for each mask table.
struct Ledgers {
  std::vector<std::unique_ptr<MaskLedger>> of;
  explicit Ledgers(const Forms &f) {
    of.emplace_back(new MaskLedger(SendAccess::make(0, 0, u32(f.index_coins()))));
    for (size_t r = 0; r < f.runs(); r++) of.emplace_back(new MaskLedger(SendAccess::make(u32(1 + r), 0, u32(f.own_coins()))));
  }
};
static BattSumcheck cpu_sumcheck(const Case &c, const Matrix &M, const std::vector<TensorRow> &rows) {
  std::vector<std::vector<E>> maps;
  for (const TensorRow &r : rows) maps.push_back(weights_times_matrix(c, M, r.g));
  return [maps](const Summand<E> &S, const std::vector<Weight<E>> &vars, const std::vector<E> &, const std::vector<std::vector<E>> &) {
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(S, vars, maps));
  };
}
// A new `MaskLedger` for the index-mask message, so that segment 0's second run takes the same
// entry c as the first, as in a program that proves a segment again and keeps the message.
static MaskLedger again_from_the_start(const Forms &f) { return MaskLedger(SendAccess::make(0, 0, u32(f.index_coins()))); }

// Every scalar that the prover sends, in order.
static std::vector<E> sent(const Forms &f, const Matrix &M, const std::vector<E> &coins) {
  const std::vector<std::vector<E>> t = tables(f, coins);
  const MaskValues<E> values = [&t](u32 oracle, u32 index) { return t[oracle][index]; };
  ListedProver ch(f.challenges);
  Ledgers ledger(f);
  std::vector<TensorRow> pending;
  for (size_t r = 0; r < f.runs(); r++) {
    const size_t l = f.link_of(r);
    std::vector<TensorRow> rows;
    for (size_t j = 0; j < f.c.rows; j++) {
      TensorRow row;
      row.g = f.g[l * f.c.rows + j], row.psi = f.psi[l * f.c.rows + j];
      MaskSlot slot = ledger.of[1 + r]->take();
      const std::vector<E> w = weights_times_matrix(f.c, M, row.g);
      row.target = mle_eval(w.data(), w.size(), row.psi.data(), f.c.k_col) + values(slot.oracle(), slot.index());
      row.masks.push_back({std::move(slot), E::one()});
      ch.send(row.target);
      rows.push_back(std::move(row));
    }
    const BattSumcheck heavy = cpu_sumcheck(f.c, M, rows);
    if (r < f.c.links) {
      pending.push_back(batt_shown_prove(ch, std::move(rows), *ledger.of[1 + r], *ledger.of[0], values, heavy).row);
    } else {
      MaskLedger kept = again_from_the_start(f);
      (void)batt_shown_prove(ch, std::move(rows), *ledger.of[1 + r], kept, values, heavy);
    }
    if (r + 1 == f.c.links && f.c.links > 1) {
      const BattSumcheck at_the_end = cpu_sumcheck(f.c, M, pending);
      (void)batt_prove(ch, std::move(pending), *ledger.of[0], values, at_the_end);
    }
  }
  return ch.tape;
}
// The verifier's functions on the same challenges; every claim that they return is checked against
// the matrix and the mask tables.
static bool accepted(const Forms &f, const Matrix &M, const std::vector<E> &coins, const std::vector<E> &tape) {
  const std::vector<std::vector<E>> t = tables(f, coins);
  const MaskValues<E> values = [&t](u32 oracle, u32 index) { return t[oracle][index]; };
  auto holds = [&](const TensorRow &r) {
    const std::vector<E> w = weights_times_matrix(f.c, M, r.g);
    return mle_eval(w.data(), w.size(), r.psi.data(), f.c.k_col) + mask_form(r.masks, values) == r.target;
  };
  ListedVerifier ch(f.challenges, tape);
  Ledgers ledger(f);
  std::vector<TensorRow> pending;
  for (size_t r = 0; r < f.runs(); r++) {
    const size_t l = f.link_of(r);
    std::vector<TensorRow> rows;
    for (size_t j = 0; j < f.c.rows; j++) {
      TensorRow row;
      row.g = f.g[l * f.c.rows + j], row.psi = f.psi[l * f.c.rows + j];
      if (!ch.recv(row.target)) return false;
      row.masks.push_back({ledger.of[1 + r]->take(), E::one()});
      rows.push_back(std::move(row));
    }
    BattShown out;
    MaskLedger kept = again_from_the_start(f);
    if (!batt_shown_verify(ch, std::move(rows), *ledger.of[1 + r], r < f.c.links ? *ledger.of[0] : kept, out)) return false;
    if (!(mask_form(out.own_masks, values) == out.own_target) || !holds(out.row)) return false;
    if (r < f.c.links) pending.push_back(std::move(out.row));
    if (r + 1 == f.c.links && f.c.links > 1) {
      TensorRow one;
      if (!batt_verify(ch, std::move(pending), *ledger.of[0], one) || !holds(one)) return false;
    }
  }
  return ch.at_end();
}

// ---- linear algebra over E ----
// Returns the rank of `columns`. *in_span: whether `extra` is in their span.
static size_t rank_of(std::vector<std::vector<E>> columns, const std::vector<E> *extra = nullptr, bool *in_span = nullptr) {
  const size_t n = columns.empty() ? 0 : columns[0].size();
  std::vector<E> x = extra ? *extra : std::vector<E>(n, E::zero());
  size_t rank = 0;
  for (size_t row = 0; row < n && rank < columns.size(); row++) {
    size_t pivot = rank;
    while (pivot < columns.size() && columns[pivot][row] == E::zero()) pivot++;
    if (pivot == columns.size()) continue;
    std::swap(columns[rank], columns[pivot]);
    const E scale = inv(columns[rank][row]);
    auto clear = [&](std::vector<E> &v) {
      const E factor = v[row] * scale;
      if (factor == E::zero()) return;
      for (size_t i = 0; i < n; i++) v[i] += -(factor * columns[rank][i]);
    };
    for (size_t j = rank + 1; j < columns.size(); j++) clear(columns[j]);
    clear(x);
    rank++;
  }
  if (in_span) {
    *in_span = true;
    for (const E &e : x) *in_span = *in_span && e == E::zero();
  }
  return rank;
}
static std::vector<E> minus(const std::vector<E> &a, const std::vector<E> &b) {
  std::vector<E> d(a.size());
  for (size_t i = 0; i < a.size(); i++) d[i] = a[i] + -b[i];
  return d;
}

static const Case CASES[] = {{2, 1, 3, 2, 1}, {2, 2, 4, 2, 2}, {3, 3, 5, 3, 2}, {1, 4, 2, 2, 3}, {2, 2, 3, 1, 2}};

// Plays the cases under `control`; every play starts from the same seed. Returns how many cases
// were played, or 0 after printing what is not as it must be.
static int play(Control control) {
  TestRng rng(20261016);
  int cases = 0, played = 0;
  for (const Case &c : CASES) {
    cases++;
    if (control == ONE_C_FOR_TWO && c.links < 2) continue;
    Forms f;
    f.c = c, f.control = control;
    for (size_t i = 0; i < c.links * c.rows; i++) {
      const std::vector<E> z = random_Es(rng, c.k_row);
      f.g.push_back({Tensor::eq(z.data(), c.k_row, random_E(rng))});
      f.psi.push_back(random_Es(rng, c.k_col));
    }
    f.challenges = random_Es(rng, (f.runs() + 1) * (c.k_col + 3));
    const Matrix M = random_matrix(c, rng), other = random_matrix(c, rng);
    const std::vector<E> base = random_Es(rng, f.coins());
    const std::vector<E> at_base = sent(f, M, base);
    const size_t scalars = at_base.size();
    if (scalars != f.runs() * (c.rows + batt_shown_own_masks(c.k_col) + 1) + (c.links > 1 ? batt_masks(c.k_col) : 0)) return printf("case %d: %zu scalars are sent\n", cases, scalars), 0;
    if (!accepted(f, M, base, at_base)) return printf("case %d: the verifier's routines refuse, or a row is false\n", cases), 0;
    std::vector<std::vector<E>> A;
    for (size_t j = 0; j < f.coins(); j++) {
      std::vector<E> moved = base;
      moved[j] += E::one();
      A.push_back(minus(sent(f, M, moved), at_base));
    }
    {  // is the map affine, and is A the same for the other matrix?
      const std::vector<E> step = random_Es(rng, f.coins());
      std::vector<E> moved = base, foretold(scalars, E::zero());
      for (size_t j = 0; j < f.coins(); j++) {
        moved[j] += step[j];
        for (size_t i = 0; i < scalars; i++) foretold[i] += step[j] * A[j][i];
      }
      if (minus(sent(f, M, moved), at_base) != foretold) return printf("case %d: what is sent is no affine map of the coins\n", cases), 0;
      if (minus(sent(f, other, moved), sent(f, other, base)) != foretold) return printf("case %d: the linear part hangs on the matrix\n", cases), 0;
    }
    const std::vector<E> between = minus(sent(f, other, base), at_base);  // b(other) - b(M)
    bool in_span = false;
    const size_t rank = rank_of(A, &between, &in_span);
    played++;
    if (control != NONE) {
      if (rank == scalars) return printf("case %d: THE CONTROL IS NOT NOTICED: the rank is full\n", cases), 0;
      if (in_span) return printf("case %d: THE CONTROL IS NOT NOTICED: the rank is %zu of %zu, but the two matrices are not told apart\n", cases, rank, scalars), 0;
      continue;
    }
    if (f.coins() != scalars) return printf("case %d: %zu coins for %zu scalars\n", cases, f.coins(), scalars), 0;
    if (rank != scalars || !in_span) return printf("case %d: THE SCALARS SHOW SOMETHING OF THE MATRIX: rank %zu of %zu\n", cases, rank, scalars), 0;
  }
  if (!played) return printf("no case was played\n"), 0;
  return played;
}

int main() {
  const int cases = play(NONE);
  if (!cases) return 1;
  int controls = 0, noticed = 0;
  std::string summary;
  for (Control control : {ONE_C_FOR_TWO, NO_Y_COEFFICIENT, LINK_RUN_AGAIN}) {
    const int played = play(control);
    controls++, noticed += played > 0;
    if (!played) printf("the control, %s: NOT AS FORETOLD (how: the line before this one)\n", CONTROL_WORDS[control]);
    summary += std::string(summary.empty() ? "" : "; ") + CONTROL_WORDS[control] + ": " + std::to_string(played) + " cases of " + std::to_string(cases);
  }
  if (noticed != controls) return printf("FAILED: %d of %d controls noticed\n", noticed, controls), 1;
  printf("ok: %d cases: as many independent scalars as coins: the scalars are uniform whatever the matrix; %d of %d controls noticed: the verifier accepts, the rank is short and two matrices are told apart (%s)\n", cases,
         noticed, controls, summary.c_str());
  return 0;
}
