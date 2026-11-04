// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests of blocks/iop/tensor.h. A Tensor is the vector
//   v[j] = scalar * prod_t (bit t of j ? hi(t) : lo(t)),
// times, if it has a dense factor, an entry of a slice of a stored vector. A Form is a sum of
// Tensors, single entries and other parts. An operation on a Tensor or a Form must do what the same
// operation does to the written-out vector (all 2^vars entries). Form::add_to() writes vectors out;
// main() checks it against at(), and at() against the definition of a Tensor's entry.
// Not tested here: times(), add_product(), Tensor::product(), rotate(), set_held(), with_table(),
// the values of Tensor::eq(), some accessors, and the BLK_REQUIRE checks, which end the process.
#include <cstdio>

#include "iop/tensor.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// Test data (SplitMix64, Steele, Lea, Flood, OOPSLA 2014).
static u64 state = 7;
static u64 word() {
  u64 z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static E element() {
  E x;
  for (F &c : x.c)
    while (!F_from_word(word(), c)) {
    }
  return x;
}
static std::vector<E> elements(size_t n) {
  std::vector<E> v(n);
  for (E &x : v) x = element();
  return v;
}

static std::vector<E> written(const Form &f, unsigned vars) {
  std::vector<E> v(size_t(1) << vars, E::zero());
  f.add_to(v.data(), vars);
  return v;
}
// The vector with its first variable fixed at g.
static std::vector<E> fixed_once(const std::vector<E> &v, const E &g) {
  std::vector<E> r(v.size() / 2);
  for (size_t i = 0; i < r.size(); i++) r[i] = v[2 * i] + g * (v[2 * i + 1] - v[2 * i]);
  return r;
}
// What insert_choice(at, count, value) must make of v: `count` choice variables before variable
// `at`; entries whose new index bits are not `value` are 0.
static std::vector<E> with_choice(const std::vector<E> &v, unsigned at, unsigned count, u64 value) {
  std::vector<E> r(v.size() << count, E::zero());
  const u64 low = (u64(1) << at) - 1;
  for (u64 j = 0; j < v.size(); j++) r[(j & low) | (value << at) | ((j & ~low) << count)] = v[j];
  return r;
}

// ---- Tests of a Tensor's memory layout: picking variables (pair (1, 0) or (0, 1), as choice
// variables have) stored as bits, the other pairs in a list that copies share, each from its own
// offset. In the walks of layout_walks() each Tensor has a plain twin (`Plain`): its scalar and
// pairs as raw data, on which each step is redone. After every step same() compares the two, so no
// expected value comes from the structure under test.
struct Plain {
  E scalar = E::one();
  std::vector<E> lo, hi;
  E at(u64 j) const {
    E r = scalar;
    for (size_t t = 0; t < lo.size(); t++) r = r * (((j >> t) & 1) ? hi[t] : lo[t]);
    return r;
  }
};
static bool same(const Tensor &t, const Plain &p) {
  if (t.vars() != p.lo.size()) return false;
  for (unsigned i = 0; i < t.vars(); i++)
    if (!(t.lo(i) == p.lo[i]) || !(t.hi(i) == p.hi[i])) return false;
  std::vector<E> all(size_t(1) << t.vars(), E::zero());
  t.add_to(all.data());
  for (u64 j = 0; j < all.size(); j++)
    if (!(all[j] == p.at(j)) || !(t.at(j) == p.at(j))) return false;
  // For every cut, entry j = scalar * part(0, cut)[low bits of j] * part(cut, rest)[high bits].
  for (unsigned cut = 0; cut <= t.vars(); cut++) {
    const std::vector<E> low = t.part(0, cut), high = t.part(cut, t.vars() - cut);
    for (u64 j = 0; j < all.size(); j++)
      if (!(t.scalar * low[j & ((u64(1) << cut) - 1)] * high[j >> cut] == p.at(j))) return false;
  }
  return true;
}
// A random pair (lo, hi), usually a special case: picking, with a 0 or a 1, lo = 1 - hi, or in F
// (the base field of E). Tensor::add_to() branches on several of these.
static void a_pair(E &l, E &h) {
  switch (word() % 14) {
  case 9: h = element(), l = E::one() - h; break;
  case 10: l = E::from(element().c[0]), h = E::from(element().c[0]); break;
  case 11: l = E::one(), h = E::from(element().c[0]); break;
  case 12: h = E::from(element().c[0]), l = E::one() - h; break;
  case 13: l = E::from(element().c[0]), h = element(); break;
  case 0: l = E::one(), h = E::zero(); break;
  case 1: l = E::zero(), h = E::one(); break;
  case 2: l = h = E::zero(); break;
  case 3: l = h = E::one(); break;
  case 4: l = element(), h = E::zero(); break;
  case 5: l = E::zero(), h = element(); break;
  case 6: l = E::one(), h = element(); break;
  default: l = element(), h = element();
  }
}
static void layout_walks() {
  for (int walk = 0; walk < 300; walk++) {
    std::vector<Tensor> t(1);
    std::vector<Plain> p(1);
    t[0].scalar = p[0].scalar = element();
    Tensor::Picked kept;  // `fix_low`'s cache, as in `Form::fix_low`: valid for one point and count
    for (int step = 0; step < 24; step++) {
      const size_t x = word() % t.size();
      const unsigned vars = t[x].vars();
      switch (word() % 6) {
      case 0:
      case 1:  // a push: also after a copy, and after variables were fixed
        if (vars < 9) {
          E l, h;
          a_pair(l, h);
          t[x].push(l, h), p[x].lo.push_back(l), p[x].hi.push_back(h);
        }
        break;
      case 2:  // 0 to 2 choice variables, inserted anywhere
        if (vars < 8) {
          const unsigned at = (unsigned)(word() % (vars + 1)), count = (unsigned)(word() % 3);
          const u64 value = word() & ((u64(1) << count) - 1);
          t[x].insert_choice(at, count, value);
          for (unsigned i = 0; i < count; i++) {
            const bool one = (value >> i) & 1;
            p[x].lo.insert(p[x].lo.begin() + at + i, one ? E::zero() : E::one()), p[x].hi.insert(p[x].hi.begin() + at + i, one ? E::one() : E::zero());
          }
        }
        break;
      case 3: {  // all tensors with enough variables are fixed at one point, sharing `kept`
        const unsigned count = (unsigned)(word() % 4);
        std::vector<E> g = elements(count);
        for (E &c : g)
          if (word() % 4 == 0) c = word() % 2 ? E::one() : E::zero();
        kept = Tensor::Picked();
        for (size_t y = 0; y < t.size(); y++) {
          if (t[y].vars() < count) continue;
          t[y].fix_low(g.data(), count, kept);
          for (unsigned i = 0; i < count; i++) p[y].scalar = p[y].scalar * (p[y].lo[i] + g[i] * (p[y].hi[i] - p[y].lo[i]));
          p[y].lo.erase(p[y].lo.begin(), p[y].lo.begin() + count), p[y].hi.erase(p[y].hi.begin(), p[y].hi.begin() + count);
        }
        break;
      }
      case 4:  // a copy, which shares the list of pairs
        if (t.size() < 5) t.push_back(t[x]), p.push_back(p[x]);
        break;
      default:  // one tensor is erased; the others keep the shared list
        if (t.size() > 1) t.erase(t.begin() + x), p.erase(p.begin() + x);
      }
      bool all = true;
      for (size_t y = 0; y < t.size(); y++) all = all && same(t[y], p[y]);
      CHECK(all, "after a step of a walk a tensor is not its plain twin");
      if (!all) return;
    }
    for (size_t y = 0; y < t.size(); y++) {  // the multilinear extension, from the twin's vector
      const std::vector<E> x = elements(t[y].vars());
      std::vector<E> v(size_t(1) << t[y].vars());
      for (u64 j = 0; j < v.size(); j++) v[j] = p[y].at(j);
      for (const E &g : x) v = fixed_once(v, g);
      CHECK(t[y].mle(x.data()) == v[0], "a tensor's multilinear extension is not its plain twin's");
    }
  }
  // Boundary cases of insert_choice()'s shifts: 63 variables, the most allowed.
  const unsigned ends[][3] = {{0, 63, 0}, {0, 1, 62}, {62, 1, 62}, {31, 32, 31}, {0, 0, 63}, {63, 0, 63}, {1, 62, 1}, {30, 3, 60}};  // at, count, variables before
  for (const auto &e : ends) {
    Tensor t;
    u64 index = 0;
    for (unsigned i = 0; i < e[2]; i++) t.push(i % 2 ? E::zero() : E::one(), i % 2 ? E::one() : E::zero()), index |= u64(i % 2) << i;
    const u64 value = e[1] ? 0x5A5A5A5A5A5A5A5Aull & ((u64(1) << e[1]) - 1) : 0, low = (u64(1) << e[0]) - 1;
    t.insert_choice(e[0], e[1], value);
    const u64 want = (index & low) | (value << e[0]) | ((index & ~low) << e[1]);
    CHECK(t.vars() == e[1] + e[2] && t.at(want) == E::one() && is_zero(t.at(want ^ 1)) && is_zero(t.at(want ^ (u64(1) << (t.vars() - 1)))), "choice variables at an end are not where they belong");
  }
}
// With 4,096 entries or more, merge_entries() first adds them up in a hash table. The result must
// be the list that sorting and adding gives: distinct indices in rising order, sums of 0 kept.
static void many_entries() {
  // 513, 2,049 and 8,193 distinct indices: one more than half the table's size, at which it grows.
  // This test does not catch an entry that is put into a wrong free slot, and need not: it comes
  // out as a second entry of its index, and the sorting and adding after the table joins the two.
  for (int turn = 0; turn < 8; turn++)
  for (u64 distinct : {u64(1), u64(3), u64(513), u64(700), u64(2049), u64(5000), u64(8193), u64(20000)}) {
    Form f;
    std::vector<std::pair<u64, E>> want;
    std::vector<u64> index;
    for (u64 i = 0; i < distinct; i++) index.push_back((word() >> 20) | (i << 44));  // all distinct
    for (u64 i = 0; i < 20000; i++) f.add_entry(index[i < distinct ? i : word() % distinct], i % 97 == 0 ? E::zero() : element());
    f.add_entry(index[0], element());
    f.add_entry(index[0], E::zero() - f.entries.back().second);  // a pair that cancels
    want = f.entries;
    std::stable_sort(want.begin(), want.end(), [](const std::pair<u64, E> &a, const std::pair<u64, E> &b) { return a.first < b.first; });
    size_t n = 0;
    for (size_t i = 0; i < want.size(); i++) {
      if (n && want[n - 1].first == want[i].first) want[n - 1].second += want[i].second;
      else want[n++] = want[i];
    }
    want.resize(n);
    f.merge_entries();
    CHECK(f.entries == want, "many entries are not merged to the list that sorting and adding gives");
  }
}
// add(Form &&, x) must give what add(const Form &, x) gives, into an empty and into a filled form,
// hints included. A hint (`Held`) names parts that add up to a multiple of a vector that the
// prover's engine has (blocks/engine/engine.h: it keeps a prover's large tables).
// Also fix_low() with a factor.
static void moved_and_scaled() {
  for (int filled = 0; filled < 2; filled++) {
    const unsigned vars = 5;
    auto some = [&](int tensors, int entries) {
      Form f;
      for (int i = 0; i < tensors; i++) f.add(Tensor::eq(elements(vars).data(), vars, element()));
      for (int i = 0; i < entries; i++) f.add_entry(word() % 32, element());
      return f;
    };
    Form a = filled ? some(2, 3) : Form(), b = a, other = some(3, 4);
    other.held.push_back(Held{7, element(), 1, 3, 0, 2, 0, 0});
    const E x = element();
    a.add(other, x);
    Form dies = other;
    b.add(std::move(dies), x);
    CHECK(written(a, vars) == written(b, vars), "a form that is moved in gives another vector than one that is copied in");
    CHECK(a.held.size() == 1 && b.held.size() == 1 && a.held[0].from == b.held[0].from && a.held[0].to == b.held[0].to && a.held[0].entries_from == b.held[0].entries_from &&
              a.held[0].entries_to == b.held[0].entries_to && a.held[0].scale == b.held[0].scale,
          "a hint of a form that is moved in names other parts than one that is copied in");
    for (unsigned count : {0u, 2u, 5u}) {
      const std::vector<E> g = elements(count);
      const E times = element();
      Form one = a, two = a;
      one.fix_low(g.data(), count), one.scale(times);
      two.fix_low(g.data(), count, times);
      CHECK(written(one, vars - count) == written(two, vars - count), "fixing with a factor is not fixing and then scaling");
    }
  }
}
// Form::fix_low() folds a dense factor's slice once for all tensors that share it (same vector,
// first entry and dense variables, which index the slice, in the same order). The form must equal
// the sum of its tensors fixed alone, over several steps, and a tensor that differs in any of the
// three must not be taken to share it.
static void shared_slices() {
  const auto B = std::make_shared<const std::vector<E>>(elements(40)), C = std::make_shared<const std::vector<E>>(elements(40));
  Form f;
  std::vector<Tensor> alone;
  for (int i = 0; i < 12; i++) {
    Tensor t;
    t.scalar = element();
    t.push_dense(2);  // variables 0, 1
    E l, h;
    a_pair(l, h), t.push(l, h);
    t.push_dense(2);  // variables 3, 4
    a_pair(l, h), t.push(l, h);
    if (i % 4 == 3) std::swap(t.dense_vars[0], t.dense_vars[2]);  // same variables in another order
    t.set_values(i % 5 == 4 ? C : B, i % 3 == 2 ? 7 : 0);
    f.add(t), alone.push_back(t);
  }
  unsigned vars = 6;
  for (unsigned count : {1u, 3u, 2u}) {
    const std::vector<E> g = elements(count);
    f.fix_low(g.data(), count);
    Form apart;
    for (Tensor &t : alone) t.fix_low(g.data(), count), apart.add(t);
    vars -= count;
    CHECK(written(f, vars) == written(apart, vars), "tensors that share a written-out slice are not fixed as each would be alone");
  }
}

// A tensor over vars variables of which some, in a random order, index a slice of a shared vector.
static Tensor random_tensor(unsigned vars, unsigned dense, const std::shared_ptr<const std::vector<E>> &shared) {
  Tensor v;
  v.scalar = element();
  for (unsigned t = 0; t < vars; t++) v.push(element(), element());
  std::vector<unsigned> all;
  for (unsigned t = 0; t < vars; t++) all.push_back(t);
  for (unsigned i = 0; i < dense; i++) {
    const size_t pick = word() % all.size();
    v.dense_vars.push_back(all[pick]);
    v.set_pair(all[pick], E::one(), E::one());
    all.erase(all.begin() + pick);
  }
  if (dense) v.set_values(shared, word() % (shared->size() - (size_t(1) << dense) + 1));
  return v;
}

// Slices of many lengths in one form, long and short in turn. Form::fix_low() folds them one after
// the other in one work buffer, which still contains what an earlier, longer slice left there.
// The form must equal the sum of its tensors fixed alone, over several steps, down to slices of
// one entry.
static void slices_of_many_lengths() {
  const auto B = std::make_shared<const std::vector<E>>(elements(200));
  for (int trial = 0; trial < 50; trial++) {
    const unsigned vars = 7;
    Form f;
    std::vector<Tensor> alone;
    for (unsigned dense : {6u, 1u, 5u, 2u, 7u, 3u, 1u, 4u, 6u}) {
      Tensor t = random_tensor(vars, dense, B);
      f.add(t), alone.push_back(t);
    }
    unsigned left = vars;
    for (unsigned count : {1u, 2u, 1u, 3u}) {
      const std::vector<E> g = elements(count);
      f.fix_low(g.data(), count);
      Form apart;
      for (Tensor &t : alone) t.fix_low(g.data(), count), apart.add(t);
      left -= count;
      CHECK(written(f, left) == written(apart, left), "slices of many lengths in one form are not folded as each would be alone");
    }
  }
}

int main() {
  const auto shared = std::make_shared<const std::vector<E>>(elements(100));
  for (int trial = 0; trial < 200; trial++) {
    const unsigned vars = 1 + word() % 6;
    Form f;
    for (unsigned i = 0; i < 3; i++) f.add(random_tensor(vars, word() % (vars + 1), shared));
    const u64 twice = word() % (u64(1) << vars);
    f.add_entry(twice, element()), f.add_entry(word() % (u64(1) << vars), element()), f.add_entry(twice, element());
    const std::vector<E> v = written(f, vars);
    CHECK(f.fits(vars) && !f.fits(vars + 1), "fits() is wrong");
    Form beyond;
    beyond.add_entry(u64(1) << vars, E::one());
    CHECK(!beyond.fits(vars) && beyond.fits(vars + 1), "fits() does not see an entry beyond the length");

    // at(), for the form and for each tensor by its definition
    for (u64 j = 0; j < v.size(); j++) CHECK(f.at(j) == v[j], "at() differs from the written-out vector");
    for (const Tensor &t : f.tensors)
      for (u64 j = 0; j < v.size(); j++) {
        E x = t.scalar;
        u64 d = 0;
        for (unsigned b = 0; b < vars; b++) x = x * (((j >> b) & 1) ? t.hi(b) : t.lo(b));
        for (size_t i = 0; i < t.dense_vars.size(); i++) d |= ((j >> t.dense_vars[i]) & 1) << i;
        if (t.has_dense()) x = x * (*shared)[t.first + d];
        CHECK(t.at(j) == x, "a tensor's entry is not its definition");
      }

    // fix_low() and mle()
    const std::vector<E> g = elements(vars);
    std::vector<E> w = v;
    for (unsigned count = 0; count <= vars; count++) {
      Form h = f;
      h.fix_low(g.data(), count);
      CHECK(written(h, vars - count) == w, "fix_low() differs from fixing the written-out vector");
      if (count < vars) w = fixed_once(w, g[count]);
    }
    CHECK(f.mle(g.data(), vars) == w[0], "mle() differs");
    for (const Tensor &t : f.tensors) {
      Form one;
      one.add(t);
      CHECK(t.mle(g.data()) == one.mle(g.data(), vars), "a tensor's mle() differs");
    }
    // in two steps as in one
    const unsigned cut = word() % (vars + 1);
    Form two = f;
    two.fix_low(g.data(), cut), two.fix_low(g.data() + cut, vars - cut);
    CHECK(two.at(0) == w[0], "fixing in two steps differs from fixing at once");

    // insert_choice()
    const unsigned at = word() % (vars + 1), count = word() % 3;
    const u64 value = word() % (u64(1) << count);
    Form c = f;
    c.insert_choice(at, count, value);
    CHECK(written(c, vars + count) == with_choice(v, at, count, value), "insert_choice() differs");

    // scale(), add()
    const E x = element();
    Form s = f, sum = f;
    s.scale(x), sum.add(f, x);
    const std::vector<E> sv = written(s, vars), sumv = written(sum, vars);
    for (u64 j = 0; j < v.size(); j++) CHECK(sv[j] == x * v[j] && sumv[j] == v[j] + x * v[j], "scale() or add() differs");

    // A hint stays true under scale(), add() and insert_choice(); fix_low() drops it. `outer` gets
    // two hints: on all of f, and on the last two tensors and the first entry of a second copy.
    Form hinted = f;
    hinted.held.push_back(Held{5, E::one(), 0, f.tensors.size(), 0, f.entries.size(), 0, 0});
    Form part_of_f;
    part_of_f.tensors.assign(f.tensors.begin() + 1, f.tensors.end()), part_of_f.entries.assign(f.entries.begin(), f.entries.begin() + 1);
    const std::vector<E> part_written = written(part_of_f, vars);
    Form second = f;
    second.held.push_back(Held{6, E::one(), 1, f.tensors.size(), 0, 1, 0, 0});
    Form outer;
    outer.add(random_tensor(vars, 0, shared));
    outer.add_entry(0, element());
    outer.add(hinted, x);
    outer.add(second, g[0]);
    outer.insert_choice(at, count, value);
    outer.scale(g[0]);
    outer.insert_choice(0, 1, 1);
    CHECK(outer.held.size() == 2, "a hint was lost");
    for (const Held &h : outer.held) {
      Form part;
      part.tensors.assign(outer.tensors.begin() + h.from, outer.tensors.begin() + h.to);
      part.entries.assign(outer.entries.begin() + h.entries_from, outer.entries.begin() + h.entries_to);
      const std::vector<E> pv = written(part, vars + count + 1);
      for (u64 j = 0; j < pv.size(); j++)
        CHECK(held_at(h, h.vector == 5 ? v : part_written, j, vars + count + 1) == pv[j], "a hint no longer says what its parts add up to");
    }
    outer.fix_low(g.data(), 1);
    CHECK(outer.held.empty(), "a hint outlived fix_low()");
  }

  // A frozen part (`FrozenPart`: a shared, unchanging Form) must mean what its contents, copied
  // out, mean: the same steps on a Form (`plain`) and on its frozen copy (`cold`), compared written
  // out. Also frozen twice, and fix_low() over choice variables only (it stays frozen) and beyond
  // them (it is copied out).
  for (int round = 0; round < 20; round++) {
    const unsigned vars = 5;
    const auto shared = std::make_shared<const std::vector<E>>(elements(64));
    Form plain;
    for (int i = 0; i < 3; i++) plain.add(random_tensor(vars, i == 2 ? 2 : 0, shared));
    for (int i = 0; i < 4; i++) plain.add_entry(word() % (u64(1) << vars), element());
    Form cold = plain;
    cold.freeze(vars);
    CHECK(cold.tensors.empty() && cold.entries.empty() && cold.frozen.size() == 1 && !cold.empty() && cold.fits(vars) && !cold.fits(vars + 1), "freeze() left parts behind, or the frozen part's length is not looked at");
    unsigned now = vars;
    auto same = [&](const char *what) {
      CHECK(written(plain, now) == written(cold, now) && written(cold.flattened(), now) == written(plain, now), what);
      for (int i = 0; i < 5; i++) {
        const u64 j = word() % (u64(1) << now);
        CHECK(plain.at(j) == cold.at(j), "an entry of a frozen form differs");
      }
    };
    same("a frozen form differs from its parts");
    const E s = element();
    plain.scale(s), cold.scale(s);
    same("scale() on a frozen form");
    plain.insert_choice(0, 2, 2), cold.insert_choice(0, 2, 2), now += 2;
    same("insert_choice() on a frozen form");
    {  // more parts beside it
      Tensor more = random_tensor(now, 0, shared);
      plain.add(more), cold.add(more);
      const u64 j = word() % (u64(1) << now);
      const E e = element();
      plain.add_entry(j, e), cold.add_entry(j, e);
      same("parts beside a frozen part");
    }
    const std::vector<E> g = elements(4);
    const E times = round % 2 ? element() : E::one();  // `fix_low`'s factor, also on frozen parts
    plain.fix_low(g.data(), 2, times), cold.fix_low(g.data(), 2, times), now -= 2;
    CHECK(cold.frozen.size() == 1, "a frozen part was copied out although only choice variables were fixed");
    same("fix_low() at the choice variables of a frozen part");
    cold.freeze(now);
    CHECK(cold.frozen.size() == 1 && cold.frozen[0].form->frozen.size() == 1, "the second freeze does not hold the first");
    same("a form frozen twice");
    Form sum_plain, sum_cold;
    const E x = element();
    sum_plain.add(plain, x), sum_cold.add(cold, x);
    {  // the moving overload of `add` gives the same, into an empty form and into a filled one
      Form moved_into_empty, moved_into_filled = sum_cold, twice = sum_plain;
      moved_into_empty.add(Form(cold), x), moved_into_filled.add(Form(cold), x), twice.add(plain, x);
      CHECK(written(moved_into_empty, now) == written(sum_plain, now) && moved_into_empty.frozen.size() == cold.frozen.size(), "add(Form &&) of a frozen form into an empty one");
      CHECK(written(moved_into_filled, now) == written(twice, now), "add(Form &&) of a frozen form into a filled one");
    }
    plain = sum_plain, cold = sum_cold;
    same("add() of a frozen form");
    plain.insert_choice(1, 1, 1), cold.insert_choice(1, 1, 1), now += 1;
    same("a choice variable in the middle");
    const std::vector<E> point = elements(now);
    CHECK(plain.mle(point.data(), now) == cold.mle(point.data(), now), "mle() of a frozen form");
    plain.fix_low(g.data(), 3, times), cold.fix_low(g.data(), 3, times), now -= 3;
    CHECK(cold.frozen.empty(), "a frozen part was kept although variables of its own were fixed");
    same("fix_low() beyond the choice variables of a frozen part");
  }

  // An engine's term (`EngineTerm`: a multiple of a vector that only a prover's engine has) must
  // keep its meaning under scale(), add() and insert_choice().
  {
    const unsigned vars = 4;
    const std::vector<E> w = elements(size_t(1) << vars);
    Form f;
    f.engine_terms.push_back(EngineTerm{9, E::one(), 0, 0});
    const E s = element(), x = element();
    f.scale(s);
    f.insert_choice(2, 2, 1);
    Form sum;
    sum.add_entry(5, element());
    sum.add(f, x);
    sum.insert_choice(0, 1, 1);
    std::vector<E> want = w;
    for (E &e : want) e = e * s * x;
    want = with_choice(with_choice(want, 2, 2, 1), 0, 1, 1);
    CHECK(sum.engine_terms.size() == 1 && !sum.empty() && sum.fits(vars + 3) && !sum.fits(2), "an engine's term was lost, or its choices are not looked at");
    for (u64 j = 0; j < want.size(); j++) CHECK(engine_term_at(sum.engine_terms[0], w, j, vars + 3) == want[j], "an engine's term no longer says what it said");
    // At an index that the term's choice variables do not pick, the term's entry is 0 and at() must
    // return the other parts' entry. (Elsewhere it ends the process: tested in fold_base_test.cpp.)
    Form written_alone = sum;
    written_alone.engine_terms.clear();
    size_t outside = 0;
    for (u64 j = 0; j < want.size(); j++) {
      if ((j & sum.engine_terms[0].choice_mask) == sum.engine_terms[0].choice_value) continue;
      outside++;
      CHECK(is_zero(want[j]) && sum.at(j) == written_alone.at(j), "at() outside an engine's term's choices is not the written parts' entry");
    }
    CHECK(outside == want.size() - w.size(), "the indices outside an engine's term's choices are not all but its vector's");
  }

  // Many single entries, several on one index: fix_low() must give what fixing the written-out
  // vector gives, in both of its branches for the low bits' weights, and entries that meet on one
  // index must become one.
  for (unsigned count : {1u, 3u, 13u}) {
    const unsigned vars = 14;
    Form f;
    for (int i = 0; i < 600; i++) f.add_entry(word() % (u64(1) << vars) & ~u64(i % 3 ? 0 : 0x30), element());
    std::vector<E> w = written(f, vars);
    const std::vector<E> g = elements(count);
    for (unsigned t = 0; t < count; t++) w = fixed_once(w, g[t]);
    f.fix_low(g.data(), count);
    CHECK(written(f, vars - count) == w, "fix_low() on many entries differs from fixing the written-out vector");
    bool distinct = true;
    for (size_t i = 1; i < f.entries.size(); i++) distinct = distinct && f.entries[i - 1].first < f.entries[i].first;
    CHECK(distinct && f.entries.size() <= size_t(1) << (vars - count), "entries on one index were not merged");
  }

  // The dense factor's purpose: weights A[i] * B[c] on a matrix, A of product form, B arbitrary,
  // placed in a block:
  // element x = c * 2^a + i goes to row r0 + (x >> bj), place j0 + (x mod 2^bj), in a table of 2^k rows.
  {
    const unsigned a = 2, e = 3, bj = 3, br = a + e - bj, k = 3, places = 4;
    const u64 r0 = 4, j0 = 8;
    const std::vector<E> z = elements(a);
    const auto B = std::make_shared<const std::vector<E>>(elements(20));
    const u64 first = 5;
    Tensor t;                           // variables in index order: row bits, then place bits
    t.push_dense(br);                   // their order in the factor's index is set further down
    t.insert_choice(br, k - br, r0 >> br);
    for (unsigned i = 0; i < a; i++) t.push(E::one() - z[i], z[i]);
    t.push_dense(bj - a);
    t.insert_choice(t.vars(), places - bj, j0 >> bj);
    // c = (place bits a .. bj-1, then the br row bits): the place bits are the low bits of c.
    std::vector<unsigned> order(t.dense_vars.begin() + br, t.dense_vars.end());
    order.insert(order.end(), t.dense_vars.begin(), t.dense_vars.begin() + br);
    t.dense_vars = order;
    t.set_values(B, first);
    std::vector<E> want(size_t(1) << (k + places), E::zero()), got(want.size(), E::zero());
    for (u64 c = 0; c < (u64(1) << e); c++)
      for (u64 i = 0; i < (u64(1) << a); i++) {
        const u64 x = (c << a) + i;
        E A = E::one();
        for (unsigned b = 0; b < a; b++) A = A * (((i >> b) & 1) ? z[b] : E::one() - z[b]);
        want[(r0 + (x >> bj)) + ((j0 + (x & ((u64(1) << bj) - 1))) << k)] = A * (*B)[first + c];
      }
    t.add_to(got.data());
    CHECK(got == want, "a block of a matrix is not laid as it should be");
  }

  layout_walks();
  many_entries();
  moved_and_scaled();
  shared_slices();
  slices_of_many_lengths();

  std::printf("tensor_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
