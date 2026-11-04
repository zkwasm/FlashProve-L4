// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Constants of the weight commitment's format for the two GPU programs, the committer `cp_commit`
// and the prover `cp_prove`, second half: limits, tag words, message layouts, labels of private
// randomness (`coins` in identifiers), file names and error names for what is made for one batch,
// for one link or once for a commitment (the key file's proofs). hf_table_format.h, included here,
// is the first half: the format as a whole and the weight table T (the weights, masked and encoded,
// under one Merkle tree). hf_format.h includes both. The section numbers run across both, so this
// half's numbers have gaps.
// The two programs use some limits of section 1 and HF_OWN_TEST_LINK. They write none of the
// messages and files that sections 2, 6, 9 and 11 describe, and they stop under none of the names
// in section 10.
// The anchor is a commitment's first weight table. A leaf table is a further weight table of the
// same weights, with its own masks. A link is a message about one leaf table and the anchor
// (section 11). A member is a request of a batch or, in the one-time message, the range member,
// which takes a request's place.
// The file needs no CUDA and no namespace: every name begins with HF_, Hf or hf_.
// Terms: docs/glossary.md.
#pragma once
#include "hash_commitment/hf_table_format.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

// ---- 1. Fixed sizes and limits.
// A "line" is a named number, not a line of the parameter file, `policy.txt`. leafd and lNmax are
// both log2 of its line `slots`, the most requests in a batch; fun is log2 of ZU's code length
// (section 6). lwr, an argument of HfRangePlan::make() (prover/hash_commitment/hf_range_plan.h), is
// log2 of the places in a lookup call of that plan.
// The binding tree is a Merkle tree in which each member has a leaf, of depth leafd in a batch and
// 0 in the one-time message.
// HfParams::derive() (prover/hash_commitment/hf_opening_params.h) and make() return an error text for
// a value outside these limits. No code uses HF_ZU_ROWS_MOST.
constexpr int HF_LEAFD_MOST = 8;                   // line leafd: a batch's binding tree's depth
constexpr int HF_LNMAX_MOST = 16;                  // line lNmax
constexpr int HF_ZU_ROWS_MOST = 4096;              // ZU's code rows
constexpr int HF_FUN_MOST = 24;                    // line fun
constexpr int HF_LWR_LEAST = 1, HF_LWR_MOST = 28;  // line lwr

// ---- 2. Tag words: the first word of a message and of lists of words hashed for it. "The list
// under 121" is the list hashed with domain-separation number 121 (section 4).
constexpr uint32_t HF_OPEN_MAGIC = 0x464F504Eu;   // "FOPN": first word of a batch's shared message
constexpr uint32_t HF_ONCE_MAGIC = 0x464F4E43u;   // "FONC": of the one-time message
constexpr uint32_t HF_CLAIMS_MAGIC = 0x46434C4Du; // "FCLM": a request's claims.bin, list under 121
constexpr uint32_t HF_MEMBER_MAGIC = 0x46434C4Fu; // "FCLO": the same for the range member
constexpr uint32_t HF_LINK_MAGIC = 0x464C4E4Bu;   // "FLNK": a link, its start list, h_link's list
static_assert(HF_OPEN_MAGIC == 1179603022u && HF_ONCE_MAGIC == 1179602499u && HF_CLAIMS_MAGIC == 1178815565u &&
                  HF_MEMBER_MAGIC == 1178815567u,
              "the four tag words' decimal forms");

// ---- 4. Domain-separation numbers. A member's messages and the binding tree have the numbers 121 to 124. 126 is the start
// of a link (section 11). 127 is a hash rule for testing a proof-of-work nonce that the proof files' format does not use:
// the format tests nonces by another rule; no program's source has this one (tests/s8_model.py). 128, 129 and 140 are not in use.
// No code uses these two constants.
constexpr uint32_t HF_NUM_S = 120;       // the start of the shared message (section 6 below)
constexpr uint32_t HF_SAMPLE_KAPPA = 12; // claim n's weight: kappa_n = sample(12; seed sk; n, 0)

// ---- 6. The shared message. No program in this tree writes it. It has two modes: a batch's, and
// the one-time message (`once`), which has no step of its own. ROOT is the binding tree's root. ZU
// commits to u, the fold of T's rows. ZV commits to W eq(rw), for the weight matrix W and a
// challenge point rw of lkW coordinates. The closing is the argument for the claims below, on
// ZU and ZV. S commits to the closing's pads, its random masking values. tcols is the number of
// T's opened columns. The parts, in order:
//   TAG, ver, tcols | ROOT | root ZU | T's top layer, 2^c digests | tcols columns (R symbols, 5
//   salt symbols, fln - c siblings) | root S | root ZV | the closing
// The transcript starts from H^v(120; the words in HfStartAt's order).
// prover/hash_commitment/hf_table_format.h, section 5, defines h_card, h_0, a table's record and,
// at HF_REC_H_PAR, h_par.
// Four orders matter: ROOT comes before rw; rw before roots ZU and ZV; both roots before rZ, a
// second challenge point; root ZU before sigma_x, the seed of the opened positions.
constexpr int HF_MSG_HEAD_WORDS = 3, HF_MSG_ROOT = 3;
enum HfStartAt : int {
  HF_START_TAG = 0,
  HF_START_VERSION = 1,
  HF_START_TCOLS = 2,
  HF_START_SHF = 3,
  HF_START_H = HF_START_SHF + HF_DW,      // h_card in a batch, h_0 in the one-time message
  HF_START_H_PAR = HF_START_H + HF_DW,
  HF_START_ZEROS = HF_START_H_PAR + HF_DW,                // HF_START_PAD zero words
  HF_START_PAD = hf_pad(HF_START_ZEROS),                  // a count, not an offset
  HF_START_ROOT = HF_START_ZEROS + HF_START_PAD,          // the only words that come from the file
  HF_START_WORDS = HF_START_ROOT + HF_DW
};
static_assert(HF_DW == 8 ? HF_START_ZEROS == 27 && HF_START_PAD == 5 && HF_START_ROOT == 32 && HF_START_WORDS == 40
                         : HF_START_PAD == 0 && HF_START_ROOT == 33 && HF_START_WORDS == 43,
              "F93: under 120 off 27, pad 5, 40 words; F38: 43 words, no zero word");
static_assert(HF_DW == 8 ? hf_pad(3 + 2 * HF_DW) == 0 : hf_pad(3 + 2 * HF_DW) == 9, "F93: under 121 off 19, pad 0; built: off 23, pad 9");
// The shared closing's claims, in both modes: (C.a) for a < tcols, then (Z). hf_claims() counts them.
constexpr uint32_t hf_claims(uint32_t tcols) { return tcols + 1; }
static_assert(hf_claims(48) == 49 && hf_claims(1612) == 1613, "S1");

// ---- 8. Labels of the prover's private randomness.
// A random value's label is sixteen words (prover/common/coin_chacha.h), which name a stream in a
// context. A context has a counter, a nonce, the version word (word 10), an object (word 11) and a
// member's number (word 12). No verifier sees these numbers.
// In the key file and in each leaf file, engine_coins() (prover/hiding/engine_coins.cuh) sets word
// 10 in the labels of the generator of random values.
// Word 11, the object: shared message 6 (one-time 22), a member's first message 7 (23), its second message 8 (24).
// Outside static_asserts, no code uses these constants. static_asserts in
// prover/hiding/long_rows_words.h keep the objects of its enum `Object` different from these and from HF_LINK_OBJECT.
// The counter: a batch takes 2 B + 2 values from 1 on, B = 2^leafd, so `before` in the formulas
// below is 0 in every batch; j is a member's number. Every process key has one use (the key file's
// proofs, one link or one batch), and every context's nonce is 0. ZU, ZV, S and the binding tree
// share the shared message's one context. Creating T starts no context.
// HF_STREAM_SALTS gives the salts and spare words of ROOT's leaves and, in object 38 (a link), the
// start's salt.
constexpr uint32_t HF_COINS_OPEN = 6, HF_COINS_PART1 = 7, HF_COINS_PART2 = 8, HF_COINS_ONCE = 16;
constexpr uint32_t hf_coins_object(uint32_t base, bool once) { return base + (once ? HF_COINS_ONCE : 0); }
constexpr uint64_t hf_counter_part1(uint64_t before, uint64_t j) { return before + 1 + j; }
constexpr uint64_t hf_counter_open(uint64_t before, uint64_t B) { return before + B + 1; }
constexpr uint64_t hf_counter_part2(uint64_t before, uint64_t B, uint64_t j) { return before + B + 2 + j; }
constexpr uint64_t hf_counter_empty_batch(uint64_t before, uint64_t B) { return before + 2 * B + 2; }
constexpr uint64_t hf_counters_of_a_batch(uint64_t B) { return 2 * B + 2; }
constexpr uint32_t HF_STREAMS = 4u * (4096 + 64 * 60); // 31,744
constexpr uint32_t HF_STREAM_S = HF_STREAMS,           // S (shared message, first message)
    HF_STREAM_UW = HF_STREAMS + 4,                     // UW (first message)
    HF_STREAM_ZV = HF_STREAMS + 16,                    // ZV (shared message), ZF (second message)
    HF_STREAM_SALTS = HF_STREAMS + 20,                 // salts of ROOT's leaves, or of a start
    HF_STREAM_PLACES = HF_STREAMS + 24,                // the permutation of ROOT's leaves
    HF_STREAM_ZU = HF_STREAMS + 88;                    // ZU
static_assert(HF_STREAM_ZV == 31760 && HF_STREAM_SALTS == 31764 && HF_STREAM_PLACES == 31768 && HF_STREAM_ZU == 31832 &&
                  HF_STREAM_UW == 31748,
              "FN7's and the built decimal forms");

// ---- 9. File names. No code uses these four constants.
constexpr const char *HF_OPEN_FILE = "open.bin";     // the shared message, in a response or key.bin
constexpr const char *HF_CLAIMS_FILE = "claims.bin"; // m<j>/claims.bin: a member's opening messages
constexpr const char *HF_PATH_FILE = "path.bin";     // m<j>/path.bin: its path in the binding tree
constexpr const char *HF_ONCE_FOLDER = "cert";

// ---- 10. The names under which a prover stops. No code uses them.
constexpr const char *HF_STOP_ZV_READERS = "hc-zv-readers";       // one more set of positions on ZV than it has blinding for
// Names of self-checks ("own tests") that no code in this tree makes: each constant occurs only
// here.
constexpr const char *HF_OWN_TEST_COLUMNS = "own-test-columns"; // a check of (C.a)
// A check that ZU's root, recomputed for the closing, is the one that was sent.
constexpr const char *HF_OWN_TEST_ZU_ROOT = "own-test-zu-root";
// The remaining names stand for comparisons that no code makes: of private random values, one by
// one, with a second derivation of them. (The programs' comparison samples four values of each draw:
// EngineCoins::hold(), prover/hiding/engine_coins.cuh.)
// HF_OWN_TEST_ZU_COINS: the values of a table of ZU's shape. HF_OWN_TEST_OPEN_COINS: the values of
// the binding tree, of the pad tree S and of ZV, and the host's and the GPU's copies of S's pads.
constexpr const char *HF_OWN_TEST_ZU_COINS = "own-test-zu-coins";
constexpr const char *HF_OWN_TEST_OPEN_COINS = "own-test-open-coins";
// The private random values of the pad tree S of a link's closing, and the copy of its pads with
// which the GPU masks.
constexpr const char *HF_OWN_TEST_LINK_PADS = "own-test-link-pads";
// The private random values of the range member's trees SM_R, UW_R and ZF_R.
constexpr const char *HF_OWN_TEST_MEMBER_COINS = "own-test-member-coins";
// The salt at the start of a link.
constexpr const char *HF_OWN_TEST_START_SALT = "own-test-start-salt";

// ---- 11. The link. No program in this tree writes it: the LINK proof in a leaf file,
// `ivk/leaf/<k>.bin` (blocks/iop/link.h), is another message. The link is to show that a leaf table
// and the anchor encode the same coefficients below fi0, the number of a row's coefficients that
// are not masks. Everywhere the anchor (A) comes first, then the leaf table (C, for "copy"). The
// parts, in order:
//   TAG, ver, q, Q | the salt of the start, DW words | root ZUA | root ZUC | the anchor's top layer
//   and q columns | the leaf table's top layer and Q columns | root S | the closing, on ZUA, ZUC
// The transcript starts from H^v(126; TAG, h_card of the anchor, h_0 of the leaf table, the salt),
// with no zero word, and ends in the state e_L.
// The claims: (A.a) at a < q, (C.a) at q + a, a < Q, (E) at q + Q.
// A leaf table's record has the anchor's form, with HF_RECORD_COPY_WORD as word 0.
constexpr uint32_t HF_NUM_LINK = 126;
constexpr int HF_LINK_HEAD_WORDS = 4, HF_LINK_START_WORDS = 1 + 3 * HF_DW, HF_H_LINK_WORDS = 1 + HF_DW;
static_assert(hf_pad(1 + HF_DW) == 0 && HF_LINK_START_WORDS <= 32, "both digests and the salt lie inside the first block");
constexpr uint32_t HF_STREAM_ZUC = HF_STREAM_ZU + 4; // ZUA has ZU's own stream, 31,832
static_assert(HF_STREAM_ZUC == 31836, "the stream number of ZUC is 31,836");
// The link's blinding context: object 6 + 32, member 0, counter 1, word 10 the version word.
constexpr uint32_t HF_COINS_LINK = 32, HF_LINK_OBJECT = HF_COINS_OPEN + HF_COINS_LINK;
constexpr uint64_t HF_LINK_COUNTER = 1;
static_assert(HF_LINK_OBJECT == 38, "the object number of a link is 38");
constexpr const char *HF_LINK_FILE = "link.bin";
constexpr const char *HF_OWN_TEST_LINK = "own-test-link"; // a leaf table has the anchor's root
