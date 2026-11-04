// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Builds and opens the Merkle tree over a weight table T: R rows of 2^ln symbols (field elements).
// It is a stored tree (docs/format/FORMAT.md, R7.4): built once and used by many proofs.
//   building a tree (the committer program cp_commit): chain_rows(), chain_leaves(), nodes_up()
//   the stored upper levels (host):                    kept_rebuild(), top_layer()
//   opening columns:                                   show_columns()
// Symbols are u64 in canonical form; this is not checked. Pointers are to GPU memory unless a
// comment says host. No function allocates GPU memory, and only those that return data to the host
// wait for the GPU. On a failed CUDA call the macro CK calls exit(1), or exit(5) as redefined in
// prover/weights_commitment/common.cuh. A failed precondition that this file itself checks leads to
// exit(4), and so does a missing or all-zero key: key_for_a_launch(), prover/hash_commitment/hf_mask.cuh.
// Hidden values: the symbols and salts of columns that are not opened ("shown" in names and
// messages), and the columns' hash states (see below). The owner of a buffer that contains them
// must erase it. Outside such buffers, they and the table's key exist here in local variables and
// kernel arguments, which are not erased. Digests (tree leaves and nodes) are not hidden values;
// even so, show_columns() and top_layer() output no digests but the siblings on opened columns'
// paths and the top layer (the 2^c nodes at depth c).
#pragma once
#include "common/hash.h"
#include "common/refusal_test.h"
#include "hash/hash.h" // ../blocks
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hf_mask.cuh" // hf_salt5()
#include "hash_commitment/hf_mask_own.cuh" // [own-test-salt]; the OWN_ codes and their names
#include <cstring>
#include <thread>
#include <vector>
#ifdef __CUDACC__
#include "common/common.h" // CK
#include "common/lie_stops.h"             // hook for a test build with a malicious prover
#include "hash_commitment/commit_lie.cuh" // same for a committer; both false in this repository
#endif

namespace hf {
// H(number, tree, index, start; bytes) is the hash of blocks/hash/hash.h: BLAKE2b as specified, on
// a 16-byte label (number, tree, index) followed by `bytes`; start goes into BLAKE2b's parameter
// block. number says what is hashed (`blk::Domain`). cfg is the configuration digest, a hash of
// the public configuration files (docs/format/FORMAT.md, R4.9).
// Levels count from the tree leaves:
//   node(0, I) = H(174, num, 2^ln + I, cfg; T[0][I] .. T[R - 1][I], 8 bytes each | 40-byte salt)
//   node(l, y) = H(166, 0, 0, cfg; node(l - 1, 2 * y) | node(l - 1, 2 * y + 1)), l >= 1
// node(0, I) is tree leaf I, the hash of the column at position I; its label also contains R and
// the salt's length. node(ln, 0) is the root. docs/format/FORMAT.md calls a node a vertex.
// R must be 8 or a multiple of 16; then no zero fill precedes the salt (R7.3 there). In this file
// only chain_rows() checks this, on each call's rows.
// The hashed string starts with the label's two 8-byte words, so 128-byte block b contains symbols
// 16 * b - 2 .. 16 * b + 13 of the column. The committer computes 16 rows at a time (a slab), so
// between slabs a column's hash state is the chaining value and the slab's last two symbols, which
// begin the next block. A tree leaf costs R / 16 + 1 compressions: one per slab, one for the salt's
// block. Identifiers call this running hash a `chain`; it is not the protocol's hash chain.
//
// Public source: leaf and node hashed under different numbers: RFC 6962 (Laurie, Langley, Kasper,
// "Certificate Transparency", 2013), section 2.1.
// Public source: a leaf's hash covers its tree's number and its index: RFC 8391 (Huelsing, Butin,
// Gazdag, Rijneveld, Mohaisen, "XMSS: eXtended Merkle Signature Scheme", 2018), section 2.5.
// Public source: an opening's top layer (top_layer()): Cryptology ePrint Archive 2017/933
// (Aumasson, Endignoux, "Improving Stateless Hash-Based Signatures"), section 2.1 with Figure 1
// (section 3.5 recalls it in one sentence).
// Public source: a related form (Merkle cap): Cryptology ePrint Archive 2021/281 (Chiesa, Yogev,
// "Subquadratic SNARGs in the Random Oracle Model"), section 2.2.

// DW: the 32-bit words of a digest, 8.
constexpr int TDL = DW / 2;  // the long words (64 bits) of a digest
constexpr int TCL = 8;       // the long words of a chaining value
constexpr int TSL = TCL + 2; // the long words of a column's running state
static_assert(DW == 8 && SLABR == 16 && SALTW == 2 * HF_SALT_SYMBOLS && 8 * HF_SALT_SYMBOLS == 40 && blk::SLAB_SYMBOLS == 16 && blk::LABEL_WORDS == 2,
              "a digest of 32 bytes; a slab is a block; a salt of 40 bytes; a label of two words");

// The tags of the tree's two hashes: tree leaves and nodes. A tag (`blk::Tag`): a hash's number,
// tree and start.
struct TreeTags {
  blk::Tag leaf, vertex;
  // configuration: the configuration digest. num: the tree's number, HF_NUM_F (0) for every weight
  // table. R: symbols per column. `blk::Tag` rejects an all-zero digest, num >= 2^16 and R >= 2^24
  // through BLK_REQUIRE (blocks/base.h), which ends the process.
  TreeTags(const blk::Digest &configuration, u32 num, u32 R)
      : leaf(blk::Domain::LEAF_STORED_MAIN, num, configuration, R, 8 * HF_SALT_SYMBOLS), vertex(blk::Domain::TREE_NODE, 0, configuration) {}
  TreeTags(const u32 configuration[8], u32 num, u32 R) : TreeTags(as_digest(configuration), num, R) {} // HfTableParams::configuration

private:
  static blk::Digest as_digest(const u32 w[8]) {
    blk::Digest d;
    for (int i = 0; i < 8; i++) d.w[i] = w[i];
    return d;
  }
};
HD constexpr u32 slabs_of(u32 R) {
  return (R + SLABR - 1) / SLABR;
}

// ---- 1. A column's hash state and its tree leaf ----

// A buffer of symbols (`sym`) is row-major: row r of column j is sym[r * stride + j]. Each thread
// has one column, so the threads of a warp read adjacent words.

// Starts a column's hash state: the chaining value cv and, in `open`, the two words that begin the
// next block, here the label.
HD void chain_begin(u64 cv[TCL], u64 open[2], const blk::Tag &tag, int ln, u64 I) {
  blk::leaf_begin(cv, tag);
  open[0] = tag.word, open[1] = ((u64)1 << ln) + I;
}
// Absorbs `slabs` whole slabs, numbered from b0, into one column's state. sym points at the
// column's row 16 * b0.
HD void chain_steps(u64 cv[TCL], u64 open[2], const u64 *sym, u64 stride, u32 b0, u32 slabs) {
  for (u32 b = 0; b < slabs; b++) {
    u64 m[SLABR];
    m[0] = open[0], m[1] = open[1];
    F2B_UNROLL
    for (int x = 0; x < SLABR - 2; x++) m[2 + x] = sym[(SLABR * b + x) * stride];
    open[0] = sym[(SLABR * b + SLABR - 2) * stride], open[1] = sym[(SLABR * b + SLABR - 1) * stride];
    blk::leaf_step(cv, m, b0 + b);
  }
}
// Writes the column's tree leaf to out, from its state after the last slab; cv is overwritten.
// few: eight words, never null. They are hashed only if R = 8: they are then the column's symbols,
// because such a column has no whole slab.
HD void leaf_end(u32 *out, u64 cv[TCL], const u64 open[2], const u64 few[8], const u64 salt[HF_SALT_SYMBOLS], u32 R) {
  u64 m[SLABR];
  F2B_UNROLL
  for (int i = 0; i < SLABR; i++) { // every index is known to the compiler, so that m stays in registers
    const u64 behind_slabs = i < 2 ? open[i & 1] : i < 2 + HF_SALT_SYMBOLS ? salt[i < 2 + HF_SALT_SYMBOLS && i >= 2 ? i - 2 : 0] : 0;
    const u64 behind_few = i < 2 ? open[i & 1] : i < 10 ? few[i >= 2 && i < 10 ? i - 2 : 0] : i < 10 + HF_SALT_SYMBOLS ? salt[i >= 10 && i < 10 + HF_SALT_SYMBOLS ? i - 10 : 0] : 0;
    m[i] = R == 8 ? behind_few : behind_slabs;
  }
  const blk::Digest d = blk::leaf_last(cv, m, (u32)blk::leaf_bytes(R, 8 * HF_SALT_SYMBOLS));
  F2B_UNROLL
  for (int i = 0; i < DW; i++) out[i] = d.w[i];
}
// The size in bytes of `count` columns' states. Long word w of column j's state is
// state[w * count + j]: cv, then `open`. With R = 8, cv's place holds the column's symbols: nothing
// has been hashed yet.
static inline size_t chain_bytes(u64 count) {
  return (size_t)TSL * 8 * count;
}
// Thread j < count: chain_steps() on column j's stored state; if rows is 8, it stores the eight
// rows instead, unhashed. With b0 = 0 it starts a new state and does not read the old one. I: the
// column's position.
HD void chain_thread(u64 *state, u64 count, const u64 *sym, u64 stride, u32 b0, u32 slabs, u32 rows, const blk::Tag &tag, int ln, u64 I, u64 j) {
  u64 cv[TCL], open[2];
  if (b0 == 0)
    chain_begin(cv, open, tag, ln, I);
  else {
    F2B_UNROLL
    for (int w = 0; w < TCL; w++) cv[w] = state[w * count + j];
    open[0] = state[TCL * count + j], open[1] = state[(TCL + 1) * count + j];
  }
  if (rows == 8) {
    F2B_UNROLL
    for (int w = 0; w < 8; w++) cv[w] = sym[j + w * stride];
  } else
    chain_steps(cv, open, sym + j, stride, b0, slabs);
  F2B_UNROLL
  for (int w = 0; w < TCL; w++) state[w * count + j] = cv[w];
  state[TCL * count + j] = open[0], state[(TCL + 1) * count + j] = open[1];
}
// The position of a launch's column j: I0 + j if base is null. Otherwise the columns come in runs
// of 2^lrun adjacent positions, and run a starts at base[a].
HD u64 run_position(const u64 *base, int lrun, u64 I0, u64 j) {
  return base ? base[j >> lrun] + (j & (((u64)1 << lrun) - 1)) : I0 + j;
}
// Thread j < count: writes column j's tree leaf to leaves + j * DW. The column's state must have
// absorbed all rows. key: the table's ChaCha20 key, from which hf_salt5() (hf_mask.cuh) derives the
// salt.
HD void chain_leaf_thread(u32 *leaves, const u64 *state, u64 count, const hf_key32 &key, const u64 *base, int lrun, u64 I0, u32 R,
                          const blk::Tag &tag, u64 j) {
  u64 cv[TCL], few[8], open[2] = {state[TCL * count + j], state[(TCL + 1) * count + j]};
  F2B_UNROLL
  for (int w = 0; w < TCL; w++) cv[w] = few[w] = state[w * count + j];
  if (R == 8) blk::leaf_begin(cv, tag);
  const u64 I = run_position(base, lrun, I0, j);
  u64 salt[HF_SALT_SYMBOLS];
  hf_salt5(key, I, salt);
  leaf_end(leaves + j * DW, cv, open, few, salt, R); // few for every R: a choice between it and null at run time would keep the array in local memory
}
// Thread j < count: the tree leaf of column x = j mod 2^lrun of fibre a = j >> lrun, hashed in one
// go, so no state is stored. A fibre is a run of 2^lrun adjacent columns (section 4). base[a]: the
// fibre's first position.
//   sym:   [fibre][row][column]: row r of the column is sym[((a * R + r) << lrun) + x]
//   heaps: one binary heap of 2^(lrun + 1) digests per fibre; the tree leaf is digest 2^lrun + x
//   salts: salts + 5 * j receives the salt that was hashed; fibre_take_thread() reads it there
HD void fibre_leaf_thread(u32 *heaps, u64 *salts, const u64 *sym, const hf_key32 &key, const u64 *base, int lrun, u32 R, int ln,
                          const blk::Tag &tag, u64 j) {
  const u64 a = j >> lrun, x = j & (((u64)1 << lrun) - 1), I = base[a] + x, stride = (u64)1 << lrun;
  const u64 *column = sym + ((a * R) << lrun) + x;
  u64 cv[TCL], open[2], few[8] = {};
  chain_begin(cv, open, tag, ln, I);
  if (R == 8) {
    F2B_UNROLL
    for (int w = 0; w < 8; w++) few[w] = column[w * stride];
  } else
    chain_steps(cv, open, column, stride, 0, R / SLABR);
  u64 salt[HF_SALT_SYMBOLS];
  hf_salt5(key, I, salt);
  F2B_UNROLL
  for (int i = 0; i < HF_SALT_SYMBOLS; i++) salts[HF_SALT_SYMBOLS * j + i] = salt[i];
  leaf_end(heaps + ((a << (lrun + 1)) + ((u64)1 << lrun) + x) * DW, cv, open, few, salt, R);
}

// ---- 2. Nodes ----

// out = the parent of left and right; out may alias either. Neither the node's level nor its index
// is hashed.
HD void tree_node(u32 *out, const u32 *left, const u32 *right, const blk::Tag &tag) {
  blk::Digest l, r;
  F2B_UNROLL
  for (int i = 0; i < DW; i++) l.w[i] = left[i], r.w[i] = right[i];
  const blk::Digest d = blk::vertex(tag, l, r);
  F2B_UNROLL
  for (int i = 0; i < DW; i++) out[i] = d.w[i];
}
HD void nodes_up_thread(u32 *out, const u32 *in, const blk::Tag &tag, u64 j) {
  tree_node(out + j * DW, in + 2 * j * DW, in + (2 * j + 1) * DW, tag);
}
// Thread t < fibres * 2^depth: one digest at depth `depth` of a fibre's heap, from its two
// children. lrun: the heaps' depth.
HD void fibre_node_thread(u32 *heaps, int lrun, int depth, const blk::Tag &tag, u64 t) {
  const u64 a = t >> depth, x = t & (((u64)1 << depth) - 1), at = ((u64)1 << depth) + x;
  u32 *heap = heaps + (a << (lrun + 1)) * DW;
  tree_node(heap + at * DW, heap + 2 * at * DW, heap + (2 * at + 1) * DW, tag);
}

// ---- 3. The stored ("kept") nodes (host) ----

// The prover stores the tree's levels s .. ln in host memory as a binary heap of 2^(ln - s + 1)
// digests (HfKept, hf_commit_api.h): digest 2^(ln - l) + y is node(l, y), so digest 1 is the root;
// digest 0 is all zero. The file hcnodes.bin contains only level s, the heap's second half.

static inline u64 kept_at(int ln, int l, u64 y) { // the digest's number in the heap; s <= l <= ln
  return ((u64)1 << (ln - l)) + y;
}
static inline size_t kept_digests(int ln, int s) {
  return (size_t)2 << (ln - s);
}
// heap: host. Recomputes every level above `from` (s <= from <= ln) and zeroes digest 0. threads:
// the host threads that share a large level. Unit::load_the_nodes() (hf_keeper.cu) calls it with
// from = s, on the level read from hcnodes.bin, and then compares the root with the root given to
// Unit::leaf_begin(), so that every stored node on a path hashes up to that root.
// make_table() (hf_commit_table.cuh) calls it with from = the top level of the parts in which it
// builds the tree.
static inline void kept_rebuild(u32 *heap, int ln, int from, const TreeTags &tags, unsigned threads = 16) {
  const blk::Tag tag = tags.vertex;
  for (int l = from + 1; l <= ln; l++) {
    const u64 count = (u64)1 << (ln - l);
    u32 *out = heap + kept_at(ln, l, 0) * DW;
    const u32 *in = heap + kept_at(ln, l - 1, 0) * DW;
    const unsigned n = count < 4096 || threads < 2 ? 1 : threads;
    if (n == 1) {
      for (u64 j = 0; j < count; j++) nodes_up_thread(out, in, tag, j);
      continue;
    }
    std::vector<std::thread> hands;
    for (unsigned h = 0; h < n; h++)
      hands.emplace_back([=] {
        for (u64 j = count * h / n; j < count * (h + 1) / n; j++) nodes_up_thread(out, in, tag, j);
      });
    for (auto &hand : hands) hand.join();
  }
  for (int i = 0; i < DW; i++) heap[i] = 0;
}
// Copies the top layer, the 2^c digests node(ln - c, y), to out (host). An opened column's path
// ends below it. Requires ln - c >= s; this is not checked here.
static inline void top_layer(const u32 *heap, int c, u32 *out) {
  memcpy(out, heap + ((size_t)1 << c) * DW, ((size_t)1 << c) * DW * 4);
}
// The number of words of one opened column as show_columns() writes it: R symbols (two words each,
// low first), the salt (SALTW words) and the ln - c siblings below the top layer, lowest first.
static inline size_t column_words(u32 R, int ln, int c) {
  return 2 * (size_t)R + SALTW + (size_t)DW * (ln - c);
}

// ---- 4. Opened columns, from their fibres ----

// Levels below s are not stored. To open the column at position I, the caller recomputes its fibre:
// the 2^s columns under node(s, I >> s). The code below hashes the fibre's tree leaves and the s
// levels above them, compares the resulting node(s, I >> s) with the stored one (a self-check of
// the prover), and takes the rest of the path from the stored nodes.
// Square brackets, as in [hc-own-node], mark the name under which a failed self-check ("own test"
// in identifiers) is reported; own_stop() (hf_mask_own.cuh) maps the OWN_ codes to these names.

// A fibre's result (`take` in identifiers), which the GPU returns to the host, in long words: the
// opened column's R symbols, its 5 salt symbols, its s siblings inside the fibre, lowest level
// first, and the fibre's top, node(s, I >> s).
static inline size_t fibre_longs(u32 R, int s) {
  return (size_t)R + HF_SALT_SYMBOLS + (size_t)(s + 1) * TDL;
}
// Thread t < fibres * fibre_longs(R, s): writes long word t of these results. pos[a]: the position
// opened in fibre a.
HD void fibre_take_thread(u64 *take, const u32 *heaps, const u64 *salts, const u64 *sym, const u64 *pos, int s, u32 R,
                          u64 t) {
  const u64 per = (u64)R + HF_SALT_SYMBOLS + (u64)(s + 1) * TDL, a = t / per, w = t % per;
  const u64 I = pos[a], x = I & (((u64)1 << s) - 1);
  if (w < R) {
    take[t] = sym[((a * R + w) << s) + x];
  } else if (w < (u64)R + HF_SALT_SYMBOLS) {
    take[t] = salts[HF_SALT_SYMBOLS * ((a << s) + x) + (w - R)]; // the salt that the leaf took in
  } else {
    const u64 d = (w - R - HF_SALT_SYMBOLS) / TDL, part = (w - R - HF_SALT_SYMBOLS) % TDL;
    const u64 at = d < (u64)s ? ((((u64)1 << s) + x) >> d) ^ 1 : 1;
    const u32 *from = heaps + ((a << (s + 1)) + at) * DW + 2 * part;
    take[t] = from[0] | (u64)from[1] << 32;
  }
}
// The size in bytes of the GPU scratch buffer (`room`) of show_columns() for `fibres` fibres. The
// buffer must start at a multiple of 8 bytes.
static inline size_t show_room_bytes(int fibres, u32 R, int s) {
  return (size_t)fibres * (16 + ((size_t)2 << s) * DW * 4 + ((size_t)HF_SALT_SYMBOLS << s) * 8 + fibre_longs(R, s) * 8);
}

#ifdef __CUDACC__
// The kernels only call the
// ..._thread() functions above,
// which also compile for the host.
// Each kernel computes its index for blocks of 128 threads and must be launched so, with
// tree_blocks() blocks; this is not checked.
// The compiler hints of the kernels below are public.
// __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// __grid_constant__ for the key: cited in prover/hash_commitment/hc_chacha.cuh.
// Unrolled loops over small per-thread arrays (F2B_UNROLL): cited in prover/common/blake2b_f.h.
__global__ void __launch_bounds__(128)
    k_chain(u64 *state, u64 count, const u64 *__restrict__ sym, u64 stride, u32 b0, u32 slabs, u32 rows, const __grid_constant__ blk::Tag tag, int ln,
            u64 I0) {
  const u64 j = blockIdx.x * (u64)128 + threadIdx.x;
  if (j < count) chain_thread(state, count, sym, stride, b0, slabs, rows, tag, ln, I0 + j, j);
}
__global__ void __launch_bounds__(128)
    k_chain_leaves(u32 *leaves, const u64 *__restrict__ state, u64 count, const __grid_constant__ hf_key32 key,
                   const u64 *__restrict__ base, int lrun, u64 I0, u32 R, const __grid_constant__ blk::Tag tag) {
  const u64 j = blockIdx.x * (u64)128 + threadIdx.x;
  if (j < count) chain_leaf_thread(leaves, state, count, key, base, lrun, I0, R, tag, j);
}
__global__ void __launch_bounds__(128) k_nodes_up(u32 *out, const u32 *__restrict__ in, const __grid_constant__ blk::Tag tag, u64 count) {
  const u64 j = blockIdx.x * (u64)128 + threadIdx.x;
  if (j < count) nodes_up_thread(out, in, tag, j);
}
__global__ void __launch_bounds__(128)
    k_fibre_leaves(u32 *heaps, u64 *salts, const u64 *__restrict__ sym, const __grid_constant__ hf_key32 key,
                   const u64 *__restrict__ base, int lrun, u32 R, int ln, const __grid_constant__ blk::Tag tag, u64 count) {
  const u64 j = blockIdx.x * (u64)128 + threadIdx.x;
  if (j < count) fibre_leaf_thread(heaps, salts, sym, key, base, lrun, R, ln, tag, j);
}
__global__ void __launch_bounds__(128) k_fibre_nodes(u32 *heaps, int lrun, int depth, const __grid_constant__ blk::Tag tag, u64 count) {
  const u64 t = blockIdx.x * (u64)128 + threadIdx.x;
  if (t < count) fibre_node_thread(heaps, lrun, depth, tag, t);
}
__global__ void __launch_bounds__(128)
    k_fibre_take(u64 *take, const u32 *__restrict__ heaps, const u64 *__restrict__ salts, const u64 *__restrict__ sym,
                 const u64 *__restrict__ pos, int s, u32 R, u64 count) {
  const u64 t = blockIdx.x * (u64)128 + threadIdx.x;
  if (t < count) fibre_take_thread(take, heaps, salts, sym, pos, s, R, t);
}
static inline unsigned tree_blocks(u64 threads) {
  return (unsigned)((threads + 127) / 128);
}

// Absorbs `rows` rows, from row 16 * b0 on, into the states of columns I0 .. I0 + count - 1. Row x
// of them is at sym + x * stride. rows: a multiple of 16, or 8 with b0 = 0 for a table of eight
// rows. b0 = 0 starts new states. state: chain_bytes(count) bytes. Later calls must follow in the
// order of the rows, with the same state and count; this is not checked.
static inline void chain_rows(u64 *state, u64 count, const u64 *sym, u64 stride, u32 b0, u32 rows, const TreeTags &tags, int ln, u64 I0,
                              cudaStream_t st) {
  if (!count || !rows) return;
  if (!(rows % SLABR == 0 || (rows == 8 && b0 == 0))) {
    fprintf(stderr, "internal: the rows of T's tree come 16 at a time, or are 8 in all\n");
    exit(4);
  }
  k_chain<<<tree_blocks(count), 128, 0, st>>>(state, count, sym, stride, b0, rows / SLABR, rows, tags.leaf, ln, I0);
  CK(cudaGetLastError());
}
// After the last chain_rows(): writes tree leaf I0 + j to leaves + j * DW, j < count. count and I0
// must be the ones given to chain_rows(), where I0 entered the label; here it selects the salt.
// This is not checked. key: host.
static inline void chain_leaves(const u64 *state, u64 count, const hf_key32 *key, u64 I0, u32 R, const TreeTags &tags, u32 *leaves, cudaStream_t st) {
  if (!count) return;
  k_chain_leaves<<<tree_blocks(count), 128, 0, st>>>(leaves, state, count, key_for_a_launch(key), nullptr, 0, I0, R, tags.leaf);
  CK(cudaGetLastError());
}
// One level of the tree, `count` nodes, from the level below it. out and in must not overlap.
static inline void nodes_up(const TreeTags &tags, const u32 *in, u64 count, u32 *out, cudaStream_t st) {
  if (!count) return;
  k_nodes_up<<<tree_blocks(count), 128, 0, st>>>(out, in, tags.vertex, count);
  CK(cudaGetLastError());
}

// The first part of show_columns(): hashes the fibres and returns the host's copy of their results.
// salts_end receives the result of the self-check [own-test-salt], salts_own_test(): every salt
// that a tree leaf of the fibres absorbed, opened or not, must equal a second derivation from
// key_again. strike: read only by a statement that never runs. key, key_again, I: host.
static inline std::vector<u64> fibres_taken(const hf_key32 *key, const uint8_t *key_again, const u64 *I, int fibres,
                                            const u64 *fib, u32 R, int ln, int s, const TreeTags &tags, void *room,
                                            size_t room_bytes, cudaStream_t st, int &salts_end, bool strike = false) {
  const bool no_second_making = !key_again; // a failure, not a skip
  if (room_bytes < show_room_bytes(fibres, R, s) || ((uintptr_t)room & 7)) {
    fprintf(stderr, "internal: the room for the shown columns' trees is too small or not aligned\n");
    exit(4);
  }
  std::vector<u64> lists(2 * (size_t)fibres); // the positions, then the fibres' first positions
  for (int a = 0; a < fibres; a++) lists[a] = I[a], lists[fibres + a] = I[a] >> s << s;
  u64 *pos = (u64 *)room, *base = pos + fibres, *take = base + fibres;
  u64 *salts = take + (size_t)fibres * fibre_longs(R, s);
  u32 *heaps = (u32 *)(salts + ((size_t)fibres * HF_SALT_SYMBOLS << s));
  // show_columns() outputs words that k_fibre_take writes into this buffer, and only the fibres'
  // tops among them are compared with stored values. So the buffer is zeroed first: a word that a
  // faulty GPU fails to write must not contain stale data.
  CK(cudaMemsetAsync(room, 0, show_room_bytes(fibres, R, s), st));
  CK(cudaMemcpyAsync(pos, lists.data(), lists.size() * 8, cudaMemcpyHostToDevice, st));
  const hf_key32 k = key_for_a_launch(key);
  const u64 columns = (u64)fibres << s;
  k_fibre_leaves<<<tree_blocks(columns), 128, 0, st>>>(heaps, salts, fib, k, base, s, R, ln, tags.leaf, columns);
  CK(cudaGetLastError());
  for (int depth = s - 1; depth >= 0; depth--) {
    const u64 nodes = (u64)fibres << depth;
    k_fibre_nodes<<<tree_blocks(nodes), 128, 0, st>>>(heaps, s, depth, tags.vertex, nodes);
    CK(cudaGetLastError());
  }
  // Behind `refusal_test::strikes()`, a stub of a test hook that returns false:
  // this block never runs. It would overwrite the first word of fibre 0's recomputed
  // top, so that the comparison with the stored node must fail [hc-own-node].
  // prover/tests/one_changed_weight.sh makes that comparison fail by a changed weight.
  if (strike && ::refusal_test::strikes("root", 1)) {
    static const u32 another_word = 0x524F4F54u; // "ROOT"
    CK(cudaMemcpyAsync(heaps + DW, &another_word, 4, cudaMemcpyHostToDevice, st));
  }
  std::vector<u64> taken((size_t)fibres * fibre_longs(R, s));
  k_fibre_take<<<tree_blocks(taken.size()), 128, 0, st>>>(take, heaps, salts, fib, pos, s, R, taken.size());
  CK(cudaGetLastError());
  std::vector<u64> salts_home((size_t)columns * HF_SALT_SYMBOLS);
  CK(cudaMemcpyAsync(taken.data(), take, taken.size() * 8, cudaMemcpyDeviceToHost, st));
  CK(cudaMemcpyAsync(salts_home.data(), salts, salts_home.size() * 8, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  salts_end = no_second_making ? (int)OWN_KEY_FILE : salts_own_test(key_again, salts_home.data(), (size_t)columns, [&](size_t j) {
    return lists[fibres + (j >> s)] + (j & (((u64)1 << s) - 1));
  });
  volatile u64 *wipe = salts_home.data(); // the salts of columns that are not shown are hidden
  for (size_t w = 0; w < salts_home.size(); w++) wipe[w] = 0;
  return taken;
}
// Whether a fibre's recomputed top equals the stored node(s, I >> s). taken_a: host, the fibre's
// result. heap: host. The comparison has no early exit.
static inline bool fibre_top_is_kept(const u64 *taken_a, u32 R, int ln, int s, u64 I, const u32 *heap) {
  const u64 *top = taken_a + R + HF_SALT_SYMBOLS + (size_t)s * TDL;
  const u32 *kept = heap + kept_at(ln, s, I >> s) * DW;
  u64 differs = 0;
  for (int w = 0; w < TDL; w++) differs |= top[w] ^ (kept[2 * w] | (u64)kept[2 * w + 1] << 32);
  return !differs;
}

// Opens the columns at the drawn positions I[0 .. columns - 1] (host; repeats allowed) from their
// fibres, which the caller has recomputed in GPU memory:
//   fib[((a * R + r) << s) + x] = T[r][(I[a] >> s << s) + x], a < columns, r < R, x < 2^s.
// columns must be at least 1: unlike chain_rows(), this code does not skip a launch of 0 blocks.
// Every I[a] must be below 2^ln. Neither is checked.
// Writes column_words(R, ln, c) words for each a to out (host), complete on return. heap: host, the
// stored nodes. room: GPU, show_room_bytes(columns, R, s) bytes; it receives the salts of all the
// fibres' columns, unopened ones too, and its final zeroing is queued on st, not waited for.
// key, key_again: host; the table's ChaCha20 key and a second copy of its 32 bytes, stored
// separately.
// Two self-checks run before any column is written: [hc-own-node] (fibre_top_is_kept()) and
// [own-test-salt] (fibres_taken()). The first guards hiding: it is to keep columns of a table other
// than the committed one inside the process. Returns OWN_GOOD (0) if both pass. Otherwise out is
// all zero words, and it returns OWN_SALTS, or OWN_KEY_FILE (key_again is null or 32 zero bytes),
// or, if the salt check has passed, OWN_NODE; *which (host), if not null, gets the first fibre whose
// node differs. A failed salt check keeps its code because a wrong
// salt also changes its tree leaf and the nodes above it, and cp_prove reacts more strictly to a
// failed check on random values than to a stored node that differs. Unit::keeps_fault()
// (prover/hash_commitment/hf_keeper.cu) treats OWN_NODE, but no salt code, as a fault in what the
// prover stores. Unit::own_test_fault() reports the salt codes; cp_prove then withholds every batch
// whose release has not begun (Prover::a_self_check_failed(), prover/prover/prover.h).
// A caller that opens one prover message's columns in several calls (portions) must release none of
// out before the last portion has returned 0, and must zero the earlier portions if one has not.
// first_portion: whether I[0] is a message's first drawn position; read only by a statement in
// fibres_taken() that never runs.
static inline int show_columns(const hf_key32 *key, const uint8_t *key_again, const u64 *I, int columns, const u64 *fib,
                               u32 R, int ln, int s, int c, const TreeTags &tags, const u32 *heap, u32 *out, void *room,
                               size_t room_bytes, cudaStream_t st, int *which = nullptr, bool first_portion = true) {
  const size_t words = column_words(R, ln, c), longs = fibre_longs(R, s);
  memset(out, 0, (size_t)columns * words * 4);
  if (ln - c < s) {
    fprintf(stderr, "internal: the top layer of T's tree lies below the lowest kept level\n");
    exit(4);
  }
  int bad = OWN_GOOD;
  std::vector<u64> taken =
      fibres_taken(key, key_again, I, columns, fib, R, ln, s, tags, room, room_bytes, st, bad, first_portion);
  for (int a = columns - 1; a >= 0; a--)
    if (!fibre_top_is_kept(&taken[a * longs], R, ln, s, I[a], heap) && !commit_lie::passes("own-node") && // hook for a malicious committer's test build: false in this repository
        !lie::passes_in_another_unit("own-node")) {                                                      // the same for a malicious prover's
      if (bad == OWN_GOOD) bad = OWN_NODE; // a failed salt check keeps its code: see above
      if (which) *which = a;
    }
  CK(cudaMemsetAsync(room, 0, show_room_bytes(columns, R, s), st));
  if (!bad)
    for (int a = 0; a < columns; a++) {
      u32 *o = out + a * words;
      const u64 *t = &taken[a * longs];
      for (size_t w = 0; w < (size_t)R + HF_SALT_SYMBOLS + (size_t)s * TDL; w++)
        o[2 * w] = (u32)t[w], o[2 * w + 1] = (u32)(t[w] >> 32);
      o += 2 * (size_t)R + SALTW + (size_t)s * DW;
      for (int l = s; l < ln - c; l++, o += DW) memcpy(o, heap + kept_at(ln, l, (I[a] >> l) ^ 1) * DW, DW * 4);
    }
  volatile u64 *wipe = taken.data(); // on a failed test these are symbols that were not shown
  for (size_t w = 0; w < taken.size(); w++) wipe[w] = 0;
  return bad;
}
#endif // __CUDACC__
} // namespace hf
