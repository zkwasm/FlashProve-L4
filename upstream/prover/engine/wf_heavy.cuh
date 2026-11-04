// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The GPU side of the prover in WF, the committer's range proof, by lookups, for the entries of the
// committed matrix M (2^k_col columns). Terms (blocks/iop/wf.h): a lookup claims that every key,
//   key(place, column) = offset + sum over digits q of digit_weight[q] * M[row(place, q)][column],
// is a row number of a table 0 .. 2^d - 1. Its places come in parts of 2^log_places places. Runs
// assign matrix rows to the covered places; any other place is padding and has the padding_key.
// The keys need not fit in GPU memory: a key source writes them piece by piece, as often as asked.
//   WfKeySource   interface: writes the keys of a piece (some places of a part) to GPU memory
//   WfMatrixKeys  a WfKeySource for a matrix of signed bytes in GPU memory; only tests use it
//   WfGpu         the WfHeavy: in one walk (pass over a lookup's keys) counts how often each table
//                 row is a covered place's key (multiplicities); its LkpProver is an LkpPieces
//   WfProducts    the MemberHeavy (blocks/iop/member.h) for the pending rows (linear claims on M)
// Hidden values: the matrix, the keys, the counts. Every GPU buffer of this file that has contained
// keys or counts is zeroed before it is freed. What multiplicities() returns is the caller's to
// erase. One object serves one run. All work runs on the calling thread's default stream.
// Tested by prover/engine/tests/wf_heavy_test.cu.
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "engine/device.cuh"
#include "engine/lkp_pieces.cuh"
#include "engine/member_heavy.cuh"
#include "iop/wf.h"

namespace blk {

// The committer's implementation is rf::WfWindows (prover/hash_commitment/hf_wf_keys.cuh).
class WfKeySource {
public:
  virtual ~WfKeySource() = default;
  // Writes the piece's keys of lookup number `lookup`: keys[column + 2^k_col * local place], on the
  // calling thread's default stream, as for LkpKeySource::lay() (prover/engine/lkp_pieces.cuh).
  // Local place (g | p), g of piece.closed bits, is the part's place (g | piece.number | p), high
  // bits first. A key outside its table is written as 0 and recorded for all_fit().
  virtual void lay(size_t lookup, const LkpPiece &piece, Key16 *keys) = 0;
  // True if no key written at a covered place since the last call was outside its table. Waits
  // until every lay() so far has finished.
  virtual bool all_fit() = 0;
};

// WfRun and WfLookup's key map, as wf_k_keys() takes them.
// The runs are in a `Card`, a GPU buffer (prover/engine/device.cuh).
struct WfRunOnCard {
  u64 first_place, first_row, places, stride;
};
struct WfKeyMap {
  long long offset = 0, digit_weight[4] = {};
  unsigned digits = 0, d = 0;
  Key16 padding_key = 0, unused[3] = {};  // unused: as in DenseFactor (prover/engine/evaluate.cuh)
};
static_assert(sizeof(WfKeyMap) == sizeof(WfKeyMap::offset) + sizeof(WfKeyMap::digit_weight) + sizeof(WfKeyMap::digits) + sizeof(WfKeyMap::d) + sizeof(WfKeyMap::padding_key) + sizeof(WfKeyMap::unused),
              "WfKeyMap must have no padding bytes: resize `unused`");
// Writes one piece's keys (layout: WfKeySource::lay()). runs: the part's, by rising first_place.
// number_bits: log2 of the number of pieces in the part.
__global__ void wf_k_keys(const signed char *__restrict__ matrix, u64 row_step, u64 column_step, unsigned k_col, const WfRunOnCard *__restrict__ runs, unsigned n_runs, const WfKeyMap map,
                          unsigned number_bits, const LkpPiece piece, Key16 *__restrict__ keys, unsigned *unfit) {
  const unsigned low = piece.log_places - piece.closed;
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < u64(1) << (piece.log_places + k_col); i += (u64)gridDim.x * blockDim.x) {
    const u64 local = i >> k_col, column = i & ((u64(1) << k_col) - 1);
    const u64 place = ((((local >> low) << number_bits) | piece.number) << low) | (local & ((u64(1) << low) - 1));
    unsigned a = 0, b = n_runs;  // the last run that begins at or before the place
    while (a + 1 < b) {
      const unsigned m = (a + b) / 2;
      if (runs[m].first_place <= place) a = m; else b = m;
    }
    Key16 key = map.padding_key;
    if (n_runs && runs[a].first_place <= place && place - runs[a].first_place < runs[a].places) {
      long long k = map.offset;
      const u64 row = runs[a].first_row + (place - runs[a].first_place);
      for (unsigned q = 0; q < map.digits; q++) k += map.digit_weight[q] * matrix[(row + q * runs[a].stride) * row_step + column * column_step];
      if (k < 0 || k >> map.d) k = 0, atomicOr(unfit, 1u);
      key = (Key16)k;
    }
    keys[i] = key;
  }
}
class WfMatrixKeys : public WfKeySource {
public:
  // matrix: in GPU memory, entry (row, column) at row * row_step + column * column_step; row_step 0
  // means 2^k_col. The caller owns it; it must outlive this object.
  WfMatrixKeys(const WfPublic &x, const WfMatrix &m, const signed char *matrix, u64 row_step = 0, u64 column_step = 1)
      : k_col_(m.k_col), matrix_(matrix), row_step_(row_step ? row_step : u64(1) << m.k_col), column_step_(column_step), unfit_(1) {
    ENGINE_CK(cudaMemset(unfit_.get(), 0, sizeof(unsigned)));
    for (const WfLookup &l : x.lookups) {
      One o;
      BLK_REQUIRE(l.digit_weight.size() >= 1 && l.digit_weight.size() <= 4 && l.d <= 16, "WF on the GPU: one to four digits, a table of at most 2^16 rows");
      o.map.offset = small(l.offset), o.map.digits = (unsigned)l.digit_weight.size(), o.map.d = l.d, o.map.padding_key = (Key16)small(l.padding_key);
      for (size_t q = 0; q < l.digit_weight.size(); q++) o.map.digit_weight[q] = small(l.digit_weight[q]);
      for (const WfPart &p : l.parts) {
        std::vector<WfRunOnCard> runs;
        for (const WfRun &r : p.runs) runs.push_back(WfRunOnCard{r.first_place, r.first_row, r.places, r.stride});
        o.runs.emplace_back();
        o.runs.back().from_host(runs.data(), runs.size());
        o.log_places.push_back(p.log_places);
      }
      lookups_.push_back(std::move(o));
    }
  }
  void lay(size_t lookup, const LkpPiece &piece, Key16 *keys) override {
    const One &o = lookups_[lookup];
    const u64 all = u64(1) << (piece.log_places + k_col_);
    wf_k_keys<<<(unsigned)std::min<u64>(1 << 16, (all + 255) / 256), 256>>>(matrix_, row_step_, column_step_, k_col_, o.runs[piece.part].get(), (unsigned)o.runs[piece.part].size(), o.map,
                                                                            o.log_places[piece.part] - piece.log_places, piece, keys, unfit_.get());
    ENGINE_CK(cudaGetLastError());
  }
  bool all_fit() override {
    unsigned word = 1;
    ENGINE_CK(cudaMemcpy(&word, unfit_.get(), sizeof word, cudaMemcpyDeviceToHost));
    ENGINE_CK(cudaMemset(unfit_.get(), 0, sizeof(unsigned)));
    return !word;
  }

private:
  static long long small(F a) {
    BLK_REQUIRE(a.v >> 32 == 0, "WF on the GPU: offsets, digit weights and the padding key are small numbers that are not negative");
    return (long long)a.v;
  }
  struct One {
    WfKeyMap map;  // all its bytes go to the GPU as a kernel argument, unused digit weights too
    std::vector<Card<WfRunOnCard>> runs;  // by part
    std::vector<unsigned> log_places;
  };
  unsigned k_col_;
  const signed char *matrix_;
  u64 row_step_, column_step_;
  Card<unsigned> unfit_;
  std::vector<One> lookups_;
};

// count[key] += how often the key occurs among n keys. A thread block counts in shared memory
// first: 2^d counters of 32 bits, d <= 13.
// Counting in shared memory with atomic additions: NVIDIA Technical Blog, "GPU Pro Tip: Fast Histograms
// Using Shared Atomics on Maxwell". For the one atomic addition a bin see prover/hiding/count_keys16.cuh's head.
// Not checked: every key < 2^d; a block sees fewer than 2^32 keys.
__global__ void wf_k_count(const Key16 *__restrict__ keys, u64 n, unsigned d, unsigned long long *count) {
  extern __shared__ unsigned bins[];
  for (unsigned t = threadIdx.x; t < 1u << d; t += blockDim.x) bins[t] = 0;
  __syncthreads();
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) atomicAdd(&bins[keys[i]], 1u);
  __syncthreads();
  for (unsigned t = threadIdx.x; t < 1u << d; t += blockDim.x)
    if (bins[t]) atomicAdd(&count[t], (unsigned long long)bins[t]);
}

// out[column] += sum over the piece's local places p of
//   e[the part's place of p] * keys[column + 2^k_col * p]. One thread per column.
__global__ void wf_k_products(const Key16 *__restrict__ keys, unsigned k_col, const E *__restrict__ e, unsigned number_bits, const LkpPiece piece, E *__restrict__ out) {
  const unsigned low = piece.log_places - piece.closed;
  for (u64 column = blockIdx.x * (u64)blockDim.x + threadIdx.x; column < u64(1) << k_col; column += (u64)gridDim.x * blockDim.x) {
    E sum = out[column];
    for (u64 local = 0; local < u64(1) << piece.log_places; local++) {
      const u64 place = ((((local >> low) << number_bits) | piece.number) << low) | (local & ((u64(1) << low) - 1));
      sum += e[place] * from_u64(keys[(local << k_col) + column]);
    }
    out[column] = sum;
  }
}
// wf_k_products_fast() forms the same sum without a field multiplication for each key (lazy::SPrep,
// prover/field/field_lazy.cuh). A thread takes four adjacent columns and one of the
// 2^WF_PRODUCT_SPLIT equal slices of the local places (blockIdx.y); its partial sums (shares) go
// to part[blockIdx.y * 2^k_col + column]. Not checked here: gridDim.y = 2^WF_PRODUCT_SPLIT; keys is
// aligned to 8 bytes; k_col >= 2; WF_PRODUCT_SPLIT <= piece.log_places <= 16 + WF_PRODUCT_SPLIT, so
// that (terms of a sum) x (greatest key) <= 2^(piece.log_places - WF_PRODUCT_SPLIT) * 2^16 <= 2^32,
// SPrep's capacity.
constexpr unsigned WF_PRODUCT_SPLIT = 3;
__global__ void wf_k_prepared(const E *e, u64 n, lazy::SPrep *out) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) out[i] = lazy::sprep(e[i]);
}
__global__ void __launch_bounds__(128) wf_k_products_fast(const Key16 *__restrict__ keys, unsigned k_col, const lazy::SPrep *__restrict__ e, unsigned number_bits, const LkpPiece piece,
                                                          E *__restrict__ part) {
  const unsigned low = piece.log_places - piece.closed;
  const u64 run = u64(1) << (piece.log_places - WF_PRODUCT_SPLIT), from = blockIdx.y * run;
  for (u64 four = blockIdx.x * (u64)blockDim.x + threadIdx.x; four < u64(1) << (k_col - 2); four += (u64)gridDim.x * blockDim.x) {
    lazy::SAcc sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) sum[i] = lazy::szero();
    for (u64 local = from; local < from + run; local++) {
      const u64 place = ((((local >> low) << number_bits) | piece.number) << low) | (local & ((u64(1) << low) - 1));
      const lazy::SPrep w = e[place];
      const ushort4 k = *(const ushort4 *)(keys + (local << k_col) + 4 * four);
      lazy::smac(sum[0], w, k.x), lazy::smac(sum[1], w, k.y), lazy::smac(sum[2], w, k.z), lazy::smac(sum[3], w, k.w);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) part[((u64)blockIdx.y << k_col) + 4 * four + i] = lazy::sred(sum[i]);
  }
}
__global__ void wf_k_shares_added(const E *part, u64 columns, E *out) {
  for (u64 c = blockIdx.x * (u64)blockDim.x + threadIdx.x; c < columns; c += (u64)gridDim.x * blockDim.x) {
    E sum = out[c];
    for (unsigned y = 0; y < 1u << WF_PRODUCT_SPLIT; y++) sum += part[y * columns + c];
    out[c] = sum;
  }
}
// Adds the partial sums of the slices whose local places have top bit 0 to out[column], the others
// to out[columns + column].
__global__ void wf_k_halves_added(const E *part, u64 columns, E *out) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < 2 * columns; i += (u64)gridDim.x * blockDim.x) {
    const u64 h = i / columns, c = i % columns;
    E sum = out[i];
    for (unsigned y = 0; y < 1u << (WF_PRODUCT_SPLIT - 1); y++) sum += part[((h << (WF_PRODUCT_SPLIT - 1)) + y) * columns + c];
    out[i] = sum;
  }
}
__global__ void wf_k_of_halves(const E *half, u64 columns, const E a, const E b, const E x, E *out) {
  for (u64 c = blockIdx.x * (u64)blockDim.x + threadIdx.x; c < columns; c += (u64)gridDim.x * blockDim.x) out[c] = a * half[c] + b * half[columns + c] - x;
}
__global__ void wf_k_less(E *out, u64 n, const E x) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) out[i] -= x;
}

/* Like `wf_k_count`, for keys of a two-digit lookup; also counts each key's two digits as keys
   of a one-digit lookup: digits[digit + digit_offset] += 1 for each. The digits follow from
   key = offset + low + base * high with -base / 2 <= low < base / 2, since they are signed bytes
   (`two_digit_partner`). Shared memory: 2^d + 2^d_digits counters. */
__global__ void wf_k_count_both(const Key16 *__restrict__ keys, u64 n, unsigned d, unsigned long long *count, int offset, int base, unsigned d_digits, int digit_offset,
                                unsigned long long *digits) {
  extern __shared__ unsigned bins[];
  unsigned *digit_bins = bins + (1u << d);
  for (unsigned t = threadIdx.x; t < (1u << d) + (1u << d_digits); t += blockDim.x) bins[t] = 0;
  __syncthreads();
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
    const int key = keys[i], w = key - offset, low = ((w + base / 2) & (base - 1)) - base / 2, high = (w - low) / base;  // base is a power of two
    atomicAdd(&bins[key], 1u);
    atomicAdd(&digit_bins[low + digit_offset], 1u), atomicAdd(&digit_bins[high + digit_offset], 1u);
  }
  __syncthreads();
  for (unsigned t = threadIdx.x; t < 1u << d; t += blockDim.x)
    if (bins[t]) atomicAdd(&count[t], (unsigned long long)bins[t]);
  for (unsigned t = threadIdx.x; t < 1u << d_digits; t += blockDim.x)
    if (digit_bins[t]) atomicAdd(&digits[t], (unsigned long long)digit_bins[t]);
}

// What WfGpu stores of a part for WfProducts. The lookup argument leaves a claim on the part's keys
// (the key claim) at a point z = (columns | low bits of the place | top bit of the place), low
// coordinates first. All coordinates but the last are known when the sumcheck of the part's last
// key level (of the product tree over its keys) ends; then these sums are formed:
//   half[h][column] = sum over places with top bit h of low[low bits of place] * key[place, column]
struct WfPartSums {
  std::vector<E> low;  // eq(low coordinates, .): 2^(log_places - 1) entries
  Card<E> half;        // 2 * 2^k_col
};

// GPU memory, in bytes, that a WfGpu takes besides its key source's: the largest lookup's
// LkpPieces, every part's WfPartSums::half, and one part_sums() call's partial sums. Not counted:
// part_sums()'s weights by place (e, prepared); the counting walk's buffers, which are smaller than
// an LkpPieces and freed before wf_prove() asks for one; WfProducts' buffers.
inline u64 wf_card_needs(const WfPublic &x, const WfMatrix &m, const LkpCut &cut) {
  u64 engine = 0, parts = 0;
  for (const LkpPublic &p : wf_lookups(x, m)) engine = std::max(engine, lkp_card_needs(p.part_bits(), m.k_col, cut).bytes()), parts += p.part_bits().size();
  return engine + ((2 * parts + (u64(1) << WF_PRODUCT_SPLIT)) << m.k_col) * sizeof(E);
}

// cut.plain: none of the shortcuts below (two_digit_partner(), counts_by_part(), part_sums()).
class WfGpu : public WfHeavy {
public:
  WfGpu(const WfPublic &x, const WfMatrix &m, WfKeySource &keys, const LkpCut &cut = LkpCut())
      : x_(x), m_(m), fixed_(wf_lookups(x, m)), keys_(keys), cut_(cut), count_(x.lookups.size()), by_part_(x.lookups.size()), sums_(x.lookups.size()) {}
  // Temporary host copies of counts in multiplicities() and prover() are not zeroed.
  ~WfGpu() override {
    for (std::vector<u64> &c : count_) std::fill(c.begin(), c.end(), 0);  // (a compiler may drop this; the counts are also in the oracle message)
    by_part_.push_back(std::move(derived_));  // zeroed with the others
    for (std::vector<std::vector<u64>> &l : by_part_)
      for (std::vector<u64> &c : l) {
        volatile u64 *p = c.data();  // finer than what is sent: each part's own counts
        for (size_t i = 0; i < c.size(); i++) p[i] = 0;
      }
    for (std::vector<WfPartSums> &l : sums_)
      for (WfPartSums &o : l)
        if (o.half.size()) ENGINE_CK(cudaMemset(o.half.get(), 0, o.half.size() * sizeof(E)));
  }
  // False if a key was outside its table: wf_prove() then returns false before it sends anything.
  bool multiplicities(size_t lookup, std::vector<u64> &count) override {
    if (!cut_.plain && !count_[lookup].empty()) return count = count_[lookup], true;  // already counted, during the other lookup's call
    const WfLookup &l = x_.lookups[lookup];
    // Where part j of a one-digit lookup covers the same matrix rows as part j of a two-digit
    // lookup (same_entries()), the two-digit keys also give the one-digit counts. So a two-digit
    // lookup that is not counted yet is counted first, and the other then asks for the keys of its
    // remaining parts only.
    const size_t pair = cut_.plain ? lookup : two_digit_partner(lookup);
    if (pair != lookup && count_[pair].empty()) {
      std::vector<u64> unused;
      derive_for_ = lookup;
      const bool fits = multiplicities(pair, unused);
      derive_for_ = NONE;
      if (!fits) derived_.clear(), derived_of_ = NONE;  // a key outside its table was written as 0: count this lookup by a separate walk
    }
    const std::chrono::steady_clock::time_point then = std::chrono::steady_clock::now();
    BLK_REQUIRE(l.d <= 13, "WF on the GPU: the counting kernel has 2^13 bins");
    Card<unsigned long long> bins(size_t(1) << l.d);
    ENGINE_CK(cudaMemset(bins.get(), 0, bins.size() * 8));
    unsigned most = 0;
    for (const WfPart &p : l.parts) most = std::max(most, lkp_piece_of(p.log_places + m_.k_col, m_.k_col, cut_).log_places);
    Card<Key16> room(size_t(1) << (most + m_.k_col));
    u64 padding = 0;
    std::vector<std::vector<u64>> by_part(l.parts.size(), std::vector<u64>(bins.size(), 0));
    const WfLookup *digits = derive_for_ == NONE ? nullptr : &x_.lookups[derive_for_];
    Card<unsigned long long> digit_bins(digits ? size_t(1) << digits->d : 0);
    if (digits) {
      ENGINE_CK(cudaMemset(digit_bins.get(), 0, digit_bins.size() * 8));
      derived_.assign(digits->parts.size(), std::vector<u64>());
    }
    for (size_t part = 0; part < l.parts.size(); part++) {
      const WfPart &p = l.parts[part];
      if (derive_for_ == NONE && part < derived_.size() && !derived_[part].empty() && derived_of_ == lookup) {  // counted with the partner's walk
        by_part[part] = derived_[part];
        continue;
      }
      const bool both = digits && same_entries(*digits, l, part);
      // The same pieces as LkpPieces will ask for, so that the source cuts a part in one way only.
      LkpPiece piece = lkp_piece_of(p.log_places + m_.k_col, m_.k_col, cut_);
      piece.part = (unsigned)part;
      const u64 n = u64(1) << (piece.log_places + m_.k_col);
      for (u64 q = 0; q < u64(1) << (p.log_places - piece.log_places); q++) {
        piece.number = q;
        room.spent();
        keys_.lay(lookup, piece, room.get());
        if (both)
          wf_k_count_both<<<(unsigned)std::min<u64>(1024, (n + 255) / 256), 256, (sizeof(unsigned) << l.d) + (sizeof(unsigned) << digits->d)>>>(
              room.get(), n, l.d, bins.get(), (int)l.offset.v, (int)l.digit_weight[1].v, digits->d, (int)digits->offset.v, digit_bins.get());
        else
          wf_k_count<<<(unsigned)std::min<u64>(1024, (n + 255) / 256), 256, sizeof(unsigned) << l.d>>>(room.get(), n, l.d, bins.get());
        ENGINE_CK(cudaGetLastError());
      }
      if (both) {
        derived_[part].assign(digit_bins.size(), 0);
        ENGINE_CK(cudaMemcpy(derived_[part].data(), digit_bins.get(), digit_bins.size() * 8, cudaMemcpyDeviceToHost));
        ENGINE_CK(cudaMemset(digit_bins.get(), 0, digit_bins.size() * 8));
      }
      padding += ((u64(1) << p.log_places) - p.places()) << m_.k_col;
      static_assert(sizeof(unsigned long long) == sizeof(u64), "counters of 64 bits");
      ENGINE_CK(cudaMemcpy(by_part[part].data(), bins.get(), bins.size() * 8, cudaMemcpyDeviceToHost));
      ENGINE_CK(cudaMemset(bins.get(), 0, bins.size() * 8));
    }
    count.assign(bins.size(), 0);
    for (const std::vector<u64> &c : by_part)
      for (size_t t = 0; t < c.size(); t++) count[t] += c[t];
    ENGINE_CK(cudaMemset(room.get(), 0, room.size() * sizeof(Key16)));
    if (!keys_.all_fit()) return std::fill(count.begin(), count.end(), 0), false;
    BLK_REQUIRE(count[l.padding_key.v] >= padding, "WF on the GPU: fewer keys of padding were laid than the parts have places of padding");
    count[l.padding_key.v] -= padding;
    if (digits) derived_of_ = derive_for_;
    walks_saved += derive_for_ == NONE && derived_of_ == lookup;
    count_[lookup] = count, by_part_[lookup] = std::move(by_part);
    report_.counting += std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count();
    return true;
  }
  LkpProver &prover(size_t lookup) override {
    if (count_[lookup].empty()) {
      std::vector<u64> count;
      BLK_REQUIRE(multiplicities(lookup, count), "WF: a key in no row of its table");
    }
    const LkpPublic &p = fixed_[lookup];
    std::vector<E> bits(size_t(1) << (p.mu + p.d), E::zero());
    for (size_t t = 0; t < count_[lookup].size(); t++)
      for (size_t j = 0; j < p.planes(); j++)
        if ((count_[lookup][t] >> j) & 1) bits[j + (t << p.mu)] = E::one();
    book();
    now_.reset();  // the one before gives its memory back first
    from_.reset(new OneLookup(keys_, lookup));
    now_.reset(new LkpPieces(p, m_.k_col, *from_, std::move(bits), cut_));
    if (!cut_.plain) {  // plain: LkpPieces walks for the roots, WfProducts for the sums
      now_->counts_by_part(by_part_[lookup]);
      sums_[lookup].resize(x_.lookups[lookup].parts.size());
      now_->at_part_end([this, lookup](unsigned part, const std::vector<E> &by_variable, Key16 *room) { part_sums(lookup, part, by_variable, room); });
    }
    return *now_;
  }
  const LkpPieces *at_hand() const { return now_.get(); }  // for a report
  size_t walks_saved = 0;                                  // for a test: lookups whose partner's walk was made to count their digits too
  // The part's stored sums, or null. This object owns them.
  const WfPartSums *sums(size_t lookup, size_t part) const { return lookup < sums_.size() && part < sums_[lookup].size() && sums_[lookup][part].half.size() ? &sums_[lookup][part] : nullptr; }
  // Totals over all lookups so far: LkpPieces' passes and seconds by kind of work; counting, sums:
  // seconds in the counting walks and in part_sums().
  struct Report {
    size_t passes = 0;
    double base = 0, matrix = 0, store = 0, memory = 0, counting = 0, sums = 0;
  };
  Report report() {
    book();
    return report_;
  }
  // Frees the current lookup's LkpPieces and keeps the parts' sums.
  void rest() {
    book();
    now_.reset();
  }
  // Frees all GPU memory of this object; the parts' sums are zeroed first.
  void done() {
    book();
    now_.reset();
    for (std::vector<WfPartSums> &l : sums_)
      for (WfPartSums &o : l)
        if (o.half.size()) {
          ENGINE_CK(cudaMemset(o.half.get(), 0, o.half.size() * sizeof(E)));
          o.half.release();
        }
  }

private:
  static constexpr size_t NONE = ~size_t(0);
  // A two-digit lookup whose keys may determine this lookup's keys (same_entries() decides part by
  // part), or `lookup` if there is none.
  size_t two_digit_partner(size_t lookup) const {
    const WfLookup &l = x_.lookups[lookup];
    // d = 8: the matrix's entries are signed bytes by their type (the key source's duty; not
    // checked here), so a two-digit key splits into two entries in one way only. (With a narrower
    // range an entry could lie outside it and the key still fit.)
    if (l.digit_weight.size() != 1 || !(l.digit_weight[0] == F::one()) || l.d != 8 || !(l.offset == from_u64(128))) return lookup;
    for (size_t i = 0; i < x_.lookups.size(); i++) {
      const WfLookup &o = x_.lookups[i];
      if (o.digit_weight.size() == 2 && o.digit_weight[0] == F::one() && o.digit_weight[1] == from_u64(u64(1) << l.d) && o.d <= 13 && o.offset.v >> 15 == 0) return i;
    }
    return lookup;
  }
  // True if part `part` of both lookups has no padding and, for each run of `two`, `digits` covers
  // digit 0's rows at the same places and digit 1's rows 2^(log_places of `two`) places higher.
  static bool same_entries(const WfLookup &digits, const WfLookup &two, size_t part) {
    if (part >= digits.parts.size() || part >= two.parts.size()) return false;
    const WfPart &a = digits.parts[part], &b = two.parts[part];
    if (a.log_places != b.log_places + 1 || b.places() != u64(1) << b.log_places || a.places() != u64(1) << a.log_places || a.runs.size() != 2 * b.runs.size()) return false;
    const u64 half = u64(1) << b.log_places;
    for (size_t r = 0; r < b.runs.size(); r++)
      for (u64 q = 0; q < 2; q++) {
        const WfRun &x = a.runs[q * b.runs.size() + r], &y = b.runs[r];
        if (x.first_place != q * half + y.first_place || x.first_row != y.first_row + q * y.stride || x.places != y.places) return false;
      }
    return true;
  }
  // derive_for_: the one-digit lookup that the walk in progress also counts, or NONE.
  size_t derive_for_ = NONE, derived_of_ = NONE;
  std::vector<std::vector<u64>> derived_;  // by part of lookup derived_of_: its counts, where the partner's walk gave them
  void book() {  // moves the current LkpPieces' passes and seconds into report_
    if (!now_) return;
    report_.passes += now_->passes, report_.base += now_->s_base, report_.matrix += now_->s_matrix, report_.store += now_->s_store, report_.memory += now_->s_memory;
    now_->passes = 0, now_->s_base = now_->s_matrix = now_->s_store = now_->s_memory = 0;
  }
  // Fills sums_[lookup][part]. LkpPieces runs it at the part's end (at_part_end()): the keys are
  // requested once more while the source may still have the part's data loaded.
  void part_sums(size_t lookup, unsigned part, const std::vector<E> &by_variable, Key16 *room) {
    const WfPart &p = x_.lookups[lookup].parts[part];
    LkpPiece piece = lkp_piece_of(p.log_places + m_.k_col, m_.k_col, cut_);
    const unsigned more = p.log_places - piece.log_places;
    // Stores nothing (WfProducts then walks the part itself) unless the top bit of a local place is
    // the top bit of the place and wf_k_products_fast()'s preconditions hold.
    if (p.log_places < 1 || (more && !piece.closed) || m_.k_col < 2 || piece.log_places < WF_PRODUCT_SPLIT || piece.log_places > 16 + WF_PRODUCT_SPLIT) return;
    const std::chrono::steady_clock::time_point then = std::chrono::steady_clock::now();
    WfPartSums &o = sums_[lookup][part];
    o.low.assign(1, E::one());
    for (unsigned t = 0; t + 1 < p.log_places; t++) {
      const E z = by_variable[m_.k_col + t];
      const size_t half = o.low.size();
      o.low.resize(2 * half);
      for (size_t i = 0; i < half; i++) o.low[half + i] = o.low[i] * z, o.low[i] = o.low[i] - o.low[half + i];
    }
    const u64 columns = u64(1) << m_.k_col;
    std::vector<E> twice(o.low);
    twice.insert(twice.end(), o.low.begin(), o.low.end());  // by place: the same weights in both halves
    Card<E> e;
    e.from_host(twice.data(), twice.size());
    Card<lazy::SPrep> prepared(twice.size());
    left_unzeroed(e, "the equality weights, by place, of the challenges that the part's last level has bound: a function of the verifier's challenges alone");
    left_unzeroed(prepared, "the same weights of the verifier's challenges, laid out for the sums' kernel");
    wf_k_prepared<<<(unsigned)std::min<u64>(1 << 12, (twice.size() + 255) / 256), 256>>>(e.get(), twice.size(), prepared.get());
    Card<E> shares(columns << WF_PRODUCT_SPLIT);
    o.half.resize(2 * columns);
    ENGINE_CK(cudaMemset(o.half.get(), 0, 2 * columns * sizeof(E)));
    piece.part = part;
    for (u64 q = 0; q < u64(1) << more; q++) {
      piece.number = q;
      shares.spent(), card_fill(room, sizeof(Key16) << (piece.log_places + m_.k_col), 0); // (`room` is borrowed: a bare pointer)
      keys_.lay(lookup, piece, room);
      wf_k_products_fast<<<dim3((unsigned)std::min<u64>(1 << 14, (columns / 4 + 127) / 128), 1u << WF_PRODUCT_SPLIT), 128>>>(room, m_.k_col, prepared.get(), more, piece, shares.get());
      wf_k_halves_added<<<(unsigned)std::min<u64>(1 << 12, (columns + 255) / 256), 256>>>(shares.get(), columns, o.half.get());
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaMemset(shares.get(), 0, shares.size() * sizeof(E)));  // (`room` is its owner's to zero)
    ENGINE_CK(cudaDeviceSynchronize());
    report_.sums += std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count();
  }
  Report report_;
  class OneLookup : public LkpKeySource {
  public:
    OneLookup(WfKeySource &keys, size_t lookup) : keys_(keys), lookup_(lookup) {}
    void lay(const LkpPiece &piece, Key16 *keys) override { keys_.lay(lookup_, piece, keys); }

  private:
    WfKeySource &keys_;
    size_t lookup_;
  };
  const WfPublic x_;
  const WfMatrix m_;
  const std::vector<LkpPublic> fixed_;
  WfKeySource &keys_;
  const LkpCut cut_;
  // prover() forms the multiplicities' bits (planes) from count_. Protected: a test's subclass
  // alters count_ together with the counts that it returns.
protected:
  std::vector<std::vector<u64>> count_;  // by lookup, row: what `multiplicities` has returned
private:
  std::vector<std::vector<std::vector<u64>>> by_part_;  // by lookup, part, row: all keys of the part, those of padding too
  std::vector<std::vector<WfPartSums>> sums_;           // by lookup, part
  std::unique_ptr<OneLookup> from_;
  std::unique_ptr<LkpPieces> now_;
};

/* Forms W_j = g_j^T M for the pending rows that WF leaves on the matrix, one for each part of each
   lookup in turn (`wf_pending_row`, blocks/iop/wf.h). With e[place] = eq(place coordinates of the
   key claim's point, place), g_j is e[place] * digit_weight[q] on matrix row(place, q) at the
   covered places and 0 elsewhere. So, by the definition of a key,
       W_j[column] = sum over the covered places of e[place] * (key[place, column] - offset),
   and the matrix itself is not needed, only the part's keys once more. `begin` reads e off the
   weights that it is given, at the row of each place's first digit, and aborts if any row has
   another weight than this form gives it. */
class WfProducts : public MemberHeavy {
public:
  // kept: the run's WfGpu, or null. A part without stored sums there has its keys requested once
  // more. Neither kept nor keys is used after begin().
  WfProducts(const WfPublic &x, const WfMatrix &m, WfKeySource &keys, const LkpCut &cut = LkpCut(), const WfGpu *kept = nullptr) : x_(x), m_(m), keys_(keys), cut_(cut), kept_(kept) {}
  void begin(const std::vector<std::vector<Tensor>> &g) override {
    size_t parts = 0;
    for (const WfLookup &l : x_.lookups) parts += l.parts.size();
    BLK_REQUIRE(g.size() == parts && !inner_, "WF's products: not one row for each part of each look-up, or a second run");
    inner_.reset(new MemberHeavyGpu(m_.k_col, parts));
    const u64 columns = u64(1) << m_.k_col;
    std::vector<E> by_row(m_.rows, E::zero());
    size_t j = 0;
    for (size_t i = 0; i < x_.lookups.size(); i++)
      for (size_t part = 0; part < x_.lookups[i].parts.size(); part++, j++) {
        const WfLookup &l = x_.lookups[i];
        const WfPart &p = l.parts[part];
        std::vector<u64> touched;
        for (const Tensor &t : g[j]) spread(t, by_row, touched);
        std::vector<E> e(size_t(1) << p.log_places, E::zero());
        E covered = E::zero();
        for (const WfRun &r : p.runs)
          for (u64 k = 0; k < r.places; k++) {
            const E w = by_row[r.first_row + k] * inv(l.digit_weight[0]);
            e[r.first_place + k] = w, covered += w;
            for (size_t q = 0; q < l.digit_weight.size(); q++) {
              E &at = by_row[r.first_row + k + q * r.stride];
              BLK_REQUIRE(at == w * l.digit_weight[q], "WF's products: a row's weight is not that of WF's pending row");
              at = E::zero();
            }
          }
        for (u64 row : touched) BLK_REQUIRE(by_row[row] == E::zero(), "WF's products: a weight on a row that the part does not cover");
        if (const WfPartSums *o = kept_ ? kept_->sums(i, part) : nullptr) {
          // e[place] = factor[top bit of place] * low[low bits of place], factor = (1 - tau, tau),
          // tau the claim's last coordinate. A factor is read off one covered place, then checked
          // at all of them.
          const size_t half = o->low.size();
          BLK_REQUIRE(2 * half == e.size(), "WF's products: the kept sums are another part's");
          E factor[2] = {E::zero(), E::zero()};
          bool known[2] = {false, false};
          E uncovered = E::zero();  // sum of the weights at padding, whose keys the sums include
          std::vector<bool> is_covered(e.size(), false);
          for (const WfRun &r : p.runs)
            for (u64 k = 0; k < r.places; k++) is_covered[r.first_place + k] = true;
          for (size_t place = 0; place < e.size(); place++) {
            const int h = place >= half;
            if (is_covered[place] && !known[h] && !(o->low[place - h * half] == E::zero())) factor[h] = e[place] * inv(o->low[place - h * half]), known[h] = true;
          }
          if (known[0] != known[1]) factor[known[0] ? 1 : 0] = E::one() - factor[known[0] ? 0 : 1], known[0] = known[1] = true;
          BLK_REQUIRE(known[0] && factor[0] + factor[1] == E::one(), "WF's products: the weights are not an equality vector's");
          for (size_t place = 0; place < e.size(); place++) {
            const E w = factor[place >= half] * o->low[place % half];
            if (is_covered[place])
              BLK_REQUIRE(e[place] == w, "WF's products: the row's point is not the one at which the part's sums were kept");
            else
              uncovered += w;
          }
          wf_k_of_halves<<<(unsigned)std::min<u64>(1 << 12, (columns + 255) / 256), 256>>>(o->half.get(), columns, factor[0], factor[1], covered * l.offset + uncovered * l.padding_key,
                                                                                          inner_->table(j));
          ENGINE_CK(cudaGetLastError());
          continue;
        }
        Card<E> e_card;
        e_card.from_host(e.data(), e.size());
        LkpPiece piece = lkp_piece_of(p.log_places + m_.k_col, m_.k_col, cut_);
        piece.part = (unsigned)part;
        Card<Key16> room(size_t(1) << (piece.log_places + m_.k_col));
        const bool fast = !cut_.plain && m_.k_col >= 2 && piece.log_places >= WF_PRODUCT_SPLIT && piece.log_places <= 16 + WF_PRODUCT_SPLIT;
        Card<lazy::SPrep> e_prepared(fast ? e.size() : 0);
        left_unzeroed(e_card, "the pending row's weights by place: a function of the claim's point and public shapes, which the verifier forms too");
        left_unzeroed(e_prepared, "the same weights of the claim's point, laid out for the sums' kernel");
        Card<E> shares(fast ? columns << WF_PRODUCT_SPLIT : 0);
        if (fast) wf_k_prepared<<<(unsigned)std::min<u64>(1 << 12, (e.size() + 255) / 256), 256>>>(e_card.get(), e.size(), e_prepared.get());
        E *out = inner_->table(j);
        ENGINE_CK(cudaMemset(out, 0, columns * sizeof(E)));
        for (u64 q = 0; q < u64(1) << (p.log_places - piece.log_places); q++) {
          piece.number = q;
          room.spent(), shares.spent();
          keys_.lay(i, piece, room.get());
          if (fast) {
            wf_k_products_fast<<<dim3((unsigned)std::min<u64>(1 << 14, (columns / 4 + 127) / 128), 1u << WF_PRODUCT_SPLIT), 128>>>(room.get(), m_.k_col, e_prepared.get(),
                                                                                                                      p.log_places - piece.log_places, piece, shares.get());
            wf_k_shares_added<<<(unsigned)std::min<u64>(1 << 12, (columns + 255) / 256), 256>>>(shares.get(), columns, out);
          } else {
            wf_k_products<<<(unsigned)std::min<u64>(1 << 16, (columns + 127) / 128), 128>>>(room.get(), m_.k_col, e_card.get(), p.log_places - piece.log_places, piece, out);
          }
          ENGINE_CK(cudaGetLastError());
        }
        wf_k_less<<<(unsigned)std::min<u64>(1 << 16, (columns + 255) / 256), 256>>>(out, columns, covered * l.offset);
        ENGINE_CK(cudaGetLastError());
        ENGINE_CK(cudaMemset(room.get(), 0, room.size() * sizeof(Key16)));
        if (fast) ENGINE_CK(cudaMemset(shares.get(), 0, shares.size() * sizeof(E)));
      }
    inner_->begin(g);
    kept_ = nullptr;
    if (after_begin_) after_begin_();
  }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &chi, const std::vector<std::vector<E>> &psi) override {
    return inner_->sumcheck(s, vars, chi, psi);
  }
  std::vector<E> combine(const std::vector<E> &a) override { return inner_->combine(a); }
  // `routine` runs at the end of begin(). The kernels that form the W_j are then queued on the
  // default stream, not necessarily finished. The caller can free the GPU memory that served the
  // keys (the run's WfGpu, the key source), in stream order or after waiting, before its next step
  // needs it.
  void after_begin(std::function<void()> routine) { after_begin_ = std::move(routine); }

private:
  // Adds the tensor's entries to by_row. Visits only the indices at which the tensor is not 0 by
  // its form: a variable with the pair (1, 0) or (0, 1) fixes a bit.
  static void spread(const Tensor &t, std::vector<E> &by_row, std::vector<u64> &touched) {
    const u64 base = t.pick_value();
    std::vector<unsigned> open;
    for (unsigned v = 0; v < t.vars(); v++)
      if (!t.picks(v)) open.push_back(v);
    BLK_REQUIRE(open.size() < 40, "WF's products: a weight vector that is not cut into blocks of rows");
    for (u64 i = 0; i < u64(1) << open.size(); i++) {
      u64 row = base;
      for (size_t b = 0; b < open.size(); b++) row |= ((i >> b) & 1) << open[b];
      const E w = t.at(row);
      if (row >= by_row.size()) {
        BLK_REQUIRE(w == E::zero(), "WF's products: a weight behind the matrix's last row");
        continue;
      }
      by_row[row] += w, touched.push_back(row);
    }
  }
  const WfPublic x_;
  const WfMatrix m_;
  WfKeySource &keys_;
  const LkpCut cut_;
  const WfGpu *kept_;
  std::function<void()> after_begin_;
  std::unique_ptr<MemberHeavyGpu> inner_;
};

}  // namespace blk
