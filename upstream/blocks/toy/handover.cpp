// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Writes the proof of one toy run (small synthetic vectors, no model) as files, with values that
// this library's verifier computes on the way, so that a second implementation of the verifier
// (toy_check.py, links_check.py in tests/new/second/) can check the same bytes and compare values.
//   handover <folder> <product|tiny|links|tiny-links>
//            [work bits on draws of positions, 1 to 30 [key of the coins [short-masks]]]
// Files, arguments: see main(). <folder> must be an existing directory. The tool overwrites
// nothing: at the first name that exists it exits with code 2 and leaves what it has written, part
// of a set. Use a new directory.
// Nothing is hidden in these files: the key (default 1) is public. It determines the witness
// (`ToyCoins`) and, with a digest of the run's inputs, the prover's private randomness (`coins`):
// run_coins_that_hide_nothing() seeds the generator of blocks/coins/run_coins.h from the key and
// takes no bytes from the system, so a run repeats byte for byte.
// The exit code is 0 only if the honest proof is accepted and every altered copy rejected
// (`altered copies, 0 accepted` in output). Verdicts come from toy_verify_file() or
// segments_verify_file(), given only the public file and a proof.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#define BLK_REPEATABLE_COINS_HIDE_NOTHING 1 // runs of this tool are compared byte for byte
#include "coins/tests/repeatable_coins.h"
#include "engine/engine_cpu.h"
#include "toy/segments.h"
#include "toy/toy_coins.h"

using namespace blk;

namespace {

// Neighbouring items of the proof with the same kind and label (`what`).
struct Field {
  size_t at = 0, bytes = 0, count = 0;
  char kind = 'B';
  bool opened = false; // part of an opening: the answer to a draw of positions
  std::string what;
};

const char *kind_words(char kind) {
  switch (kind) {
  case 'F': return "element of F";
  case 'E': return "element of E";
  case 'D': return "digest";
  case 'W': return "word of 64 bits";
  default: return "bytes";
  }
}

// Labels every item that the prover wrote and joins neighbours into fields. `proof` lists items
// only if Writer::keep_list() (blocks/iop/message.h) was called before proving. Outside an
// opening's two marks, a digest at byte 0 is the chain's random bytes, plain bytes of a nonce's
// length are taken for a nonce, and the remaining items are matched in order to those messages of
// the plan (the list of the run's rounds and messages: blocks/numbers/plan.h) that enter the hash
// chain (the Fiat-Shamir transcript).
std::vector<Field> fields_of(const Writer &proof, const Planned &plan) {
  struct InChain {
    std::string what;
    u64 bytes;
  };
  std::vector<InChain> planned;
  for (const PlanMessage &m : plan->messages_)
    if (m.in_chain)
      for (u64 c = 0; c < m.count; c++) planned.push_back(InChain{m.name + (m.where.empty() ? "" : " (" + m.where + ")"), m.bytes});
  size_t next = 0, mark = 0;
  u64 used = 0;
  std::string opening;
  std::vector<Field> fields;
  for (const Writer::Item &item : proof.items()) {
    for (; mark < proof.marks().size() && proof.marks()[mark].at <= item.at; mark++)
      opening = proof.marks()[mark].words == "end of the opening" ? "" : proof.marks()[mark].words;
    Field f;
    f.at = item.at, f.bytes = item.bytes, f.count = 1, f.kind = item.kind;
    if (!opening.empty()) {
      f.opened = true;
      f.what = opening + (item.kind == 'F' ? ": symbols of a column" : item.kind == 'B' ? ": the salt of a leaf" : ": digests of the paths");
    } else if (item.at == 0 && item.kind == 'D') {
      f.what = "the chain's random bytes";
    } else if (item.kind == 'B' && item.bytes == plan->set.nonce_bytes) {
      f.what = "nonce";
    } else {
      while (next < planned.size() && used == planned[next].bytes) next++, used = 0;
      BLK_REQUIRE(next < planned.size() && used + item.bytes <= planned[next].bytes, "handover: an item that the plan's messages do not hold");
      f.what = planned[next].what;
      used += item.bytes;
    }
    if (!fields.empty() && fields.back().what == f.what && fields.back().kind == f.kind && fields.back().at + fields.back().bytes == f.at)
      fields.back().bytes += f.bytes, fields.back().count++;
    else
      fields.push_back(f);
  }
  return fields;
}

void write_file(const std::string &path, const u8 *p, size_t n) {
  std::FILE *out = std::fopen(path.c_str(), "wbx");  // x: fails if the name exists, a symlink too
  if (!out && errno == EEXIST) std::printf("%s exists already, and this tool overwrites nothing: name another folder\n", path.c_str()), std::exit(2);
  if (!out || std::fwrite(p, 1, n, out) != n || std::fclose(out)) std::printf("cannot write %s\n", path.c_str()), std::exit(2);
}
void write_file(const std::string &path, const std::string &text) { write_file(path, reinterpret_cast<const u8 *>(text.data()), text.size()); }

std::string hex(const u8 *p, size_t n) {
  static const char digit[] = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; i++) s += digit[p[i] >> 4], s += digit[p[i] & 15];
  return s;
}
std::string hex(const Digest &d) {
  u8 b[DIGEST_BYTES];
  digest_to_bytes(d, b);
  return hex(b, sizeof b);
}
// An element of E, the cubic extension of the field F, as its three coordinates in decimal.
std::string decimal(const E &x) {
  u8 b[E_BYTES];
  E_to_bytes(x, b);
  std::string s;
  for (size_t c = 0; c < E_BYTES; c += 8) {
    u64 w = 0;
    for (int i = 0; i < 8; i++) w |= (u64)b[c + i] << (8 * i);
    s += (c ? " " : "") + std::to_string(w);
  }
  return s;
}

// A round kind's number in the round's query (its input to the hash).
ChainKind kind_in_query(RoundKind k) {
  switch (k) {
  case RoundKind::ALGEBRAIC: return ChainKind::ALGEBRAIC;
  case RoundKind::EPSILON: return ChainKind::EPSILON;
  case RoundKind::SEED: return ChainKind::SEED;
  case RoundKind::POSITIONS: return ChainKind::POSITIONS;
  case RoundKind::BITS: return ChainKind::BITS;
  case RoundKind::SEAL: return ChainKind::SEAL;
  case RoundKind::EMPTY: return ChainKind::EMPTY;
  default: break;
  }
  BLK_REQUIRE(false, "handover: an entry of the plan that is no round of a chain");
  return ChainKind::EMPTY;
}

// The texts of <run>.digests.tsv and <run>.chain.tsv. chain_lists() replays the hash chain outside
// the channel (`VerifierRounds`): the same `Chain`, but each round's message is cut out of the
// proof's bytes by the plan's `message_bytes`. Of the proof it checks only these lengths and the
// proof of work.
struct ChainLists {
  std::string digests, rounds;
};
ChainLists chain_lists(u64 label, const Binder &binder, const Digest &configuration, const Planned &plan, const std::vector<u8> &proof,
                       const std::vector<Field> &fields) {
  std::vector<u8> sent; // all that enters a query, in order
  std::vector<u64> nonces;
  for (const Field &f : fields) {
    if (f.opened || f.at == 0) continue; // an opening; the chain's random bytes
    if (f.what == "nonce") {
      u64 n = 0;
      for (size_t i = 0; i < f.bytes; i++) n |= (u64)proof[f.at + i] << (8 * i);
      nonces.push_back(n);
    } else {
      sent.insert(sent.end(), proof.begin() + (long)f.at, proof.begin() + (long)(f.at + f.bytes));
    }
  }
  FirstQuery q;
  q.label = label, q.parameters = binder.parameters_digest(), q.statement = binder.statement_bytes();
  digest_from_bytes(proof.data(), q.salt);
  Chain chain(q);
  ChainLists out;
  out.digests = "what\tbytes\nthe configuration's digest\t" + hex(configuration) + "\nthe parameters' digest\t" + hex(q.parameters) +
                "\nthe chain's random bytes\t" + hex(q.salt) + "\nthe run's value\t" + hex(first_query_run(q)) + "\nthe chain's first value\t" +
                hex(chain.value()) + "\n";
  out.rounds = "round\tname\tkind\twork bits\tnonce\tthe chain's value before\tthe prover's message\tcall 0\tthe chain's value after\t"
               "elements of E drawn\tpositions drawn\n";
  size_t at = 0, nonce_at = 0;
  for (const PlanRound &r : plan->rounds_) {
    if (r.kind == RoundKind::RECORD) continue; // a stored table's own sample: no round of the chain
    BLK_REQUIRE(at + r.message_bytes <= sent.size() && (r.work_bits == 0 || nonce_at < nonces.size()), "handover: the proof holds less than the plan's rounds");
    const u64 nonce = r.work_bits ? nonces[nonce_at++] : 0;
    const Digest before = chain.value();
    ChainAbsorb a = chain.message();
    a.push(sent.data() + at, r.message_bytes);
    const ChainPrefix p = Chain::prefix(a, kind_in_query(r.kind), r.work_bits);
    const ChainCall first = chain_call(p, nonce, 0);
    u8 call[64];
    for (int i = 0; i < 64; i++) call[i] = (u8)(first.w[i / 8] >> (8 * (i % 8)));
    BLK_REQUIRE(chain.step(p, nonce, r.work_bits), "handover: a nonce of the accepted proof has not its work");
    std::string elements, positions;
    auto element = [&](bool nonzero) { elements += (elements.empty() ? "" : "; ") + decimal(nonzero ? chain.draw_E_nonzero() : chain.draw_E()); };
    for (u64 i = 0; i < r.draw_positions; i++) positions += (i ? " " : "") + std::to_string(chain.draw_position(r.log_n));
    if (r.kind == RoundKind::SEED) { // each seed: a point's coordinates, then a nonzero factor
      BLK_REQUIRE(r.draw_nonzero > 0 && r.draw_E % r.draw_nonzero == 0, "handover: a round of seeds without whole seeds");
      for (u64 s = 0; s < r.draw_nonzero; s++)
        for (u64 i = 0; i < r.draw_E / r.draw_nonzero; i++) element(i + 1 == r.draw_E / r.draw_nonzero);
    } else {
      for (u64 i = 0; i < r.draw_E; i++) element(r.draw_nonzero > 0);
    }
    out.rounds += std::to_string(chain.rounds()) + "\t" + r.name + "\t" + std::to_string((unsigned)kind_in_query(r.kind)) + "\t" +
                  std::to_string(r.work_bits) + "\t" + std::to_string(nonce) + "\t" + hex(before) + "\t" + hex(sent.data() + at, r.message_bytes) +
                  "\t" + hex(call, sizeof call) + "\t" + hex(chain.value()) + "\t" + elements + "\t" + positions + "\n";
    at += r.message_bytes;
  }
  BLK_REQUIRE(at == sent.size() && nonce_at == nonces.size(), "handover: the proof holds more than the plan's rounds");
  return out;
}

// The text of <run>.codes.tsv: the codes, then the stages of a proximity test, which is meant to
// show that rows are close to codewords.
std::string codes_list(const Codes &codes, const std::vector<IoppStage> &stages) {
  std::string text = "list\tnumber\tlog2 of the length\tentries of a message\tentries of the random tail\n";
  for (int mask = 0; mask < 2; mask++) {
    const std::vector<Code> &list = mask ? codes.mask : codes.base;
    for (size_t i = 0; i < list.size(); i++)
      text += std::string(mask ? "mask" : "base") + "\t" + std::to_string(i) + "\t" + std::to_string(list[i].log_n) + "\t" + std::to_string(list[i].ell) + "\t" +
              std::to_string(list[i].r) + "\n";
  }
  // The last column is computed here from the codes: the message entries of a stage's mask rows
  // (of random values), less those of the previous stage's two folded mask rows and its three
  // random tails.
  text += "stage\tits base code's number\tits mask code's number\tk\th\tentries that fill its mask rows up (follows from the codes)\n";
  for (size_t i = 0; i < stages.size(); i++) {
    std::string left = "-";
    if (i > 0) {
      const Code &c = codes.base[stages[i - 1].code], &z = codes.mask[stages[i - 1].mask_code];
      left = std::to_string(((u64)codes.mask[stages[i].mask_code].ell << stages[i].h) - (2 * (u64)z.ell + c.r + 2 * (u64)z.r));
    }
    text += std::to_string(i + 1) + "\t" + std::to_string(stages[i].code) + "\t" + std::to_string(stages[i].mask_code) + "\t" + std::to_string(stages[i].k) + "\t" +
            std::to_string(stages[i].h) + "\t" + left + "\n";
  }
  return text;
}

const char *round_kind(RoundKind k) {
  switch (k) {
  case RoundKind::ALGEBRAIC: return "elements of E";
  case RoundKind::EPSILON: return "the element that joins a hiding sum";
  case RoundKind::SEED: return "seeds of samples";
  case RoundKind::POSITIONS: return "positions";
  case RoundKind::BITS: return "entries -1, 0, 1";
  case RoundKind::SEAL: return "seal";
  case RoundKind::EMPTY: return "nothing";
  default: return "other";
  }
}

// Guards against an empty shell variable in the argument ("$W", "$W/"), with which the tool would
// write into the root of the file system. Returns null, or what is wrong.
static const char *folder_fault(const char *given) {
  std::error_code e;
  return !*given                                        ? "the folder's name is empty"
         : !std::filesystem::is_directory(given, e)     ? "this is no folder: make one first"
         : std::filesystem::equivalent(given, "/", e)   ? "this is the root of the file system"
                                                        : nullptr;
}
} // namespace

// Writes into <folder>, where <run> is the second argument:
//   <run>.configuration    the run's one public file: its lines fix the setting, codes and shapes
//   <run>.proof            the honest proof
//   <run>.parameters       the parameters of the chain's first query, as hashed (`Binder`,
//                          blocks/chain/public.h)
//   <run>.parameters.hex   the same bytes in hexadecimal
//   <run>.statement        the first query's statement: empty for these runs
//   <run>.plan.tsv         every round of the plan
//   <run>.fields.tsv       every field of the proof
//   <run>.codes.tsv        the codes, and the stages of the (last) proximity test
//   <run>.digests.tsv      the digests at the chain's start
//   <run>.chain.tsv        every round's query and answer
//   <run>.targets.tsv      values that the verifier notes on the honest proof, such as a linear
//                          claim's value (`target`) after a block of the proximity test
//   <run>.altered-NN.proof altered copies; <run>.altered.tsv: what was changed in each, and this
//                          verifier's reason for rejecting it, or ACCEPTED
// `product` is the run of blocks/toy/product.h (the entrywise product of two vectors), `links` that
// of blocks/toy/segments.h (three such products, each a segment: `link` in identifiers). `tiny` and
// `tiny-links` are these runs at a size for reading by eye: the same protocol blocks in the same
// order, vectors of 16 (32) entries, short codes, a few positions in a list.
// The target is the setting's 128 bits, but 10 bits in a tiny run: each round's error by this
// library's own accounting, times 2^-w for w bits of proof of work, must be at most 2^-target. A
// tiny run reaches no security level: its public file says so in the first line and has the 10 in
// the line `target`. The verifier's check that the plan meets the target is the same.
// The work bits are the w of rounds that draw positions (default 20; tiny runs 6). Mask rows are
// as long as main rows unless `short-masks` is given; tiny runs ignore it.
int main(int argc, char **argv) {
  if (argc < 3 || (std::strcmp(argv[2], "product") && std::strcmp(argv[2], "tiny") && std::strcmp(argv[2], "links") && std::strcmp(argv[2], "tiny-links"))) {
    std::printf("usage: handover <folder> <product|tiny|links|tiny-links> [work bits on draws of positions, 1 to 30 [key of the coins [short-masks]]]\n");
    return 2;
  }
  if (const char *fault = folder_fault(argv[1])) {
    std::printf("\"%s\": %s\n", argv[1], fault);
    return 2;
  }
  if (argc > 3 && (std::atoi(argv[3]) < 1 || std::atoi(argv[3]) > 30)) {
    std::printf("work bits (the third word) must be 1 to 30: each further bit doubles the prover's search\n");
    return 2;
  }
  const std::string name = argv[2], stem = std::string(argv[1]) + "/" + name;
  const bool links = name == "links" || name == "tiny-links", tiny = name == "tiny" || name == "tiny-links";
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  if (tiny) set.target = 10; // bits: no security level; the run's public file says so
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, argc > 3 ? (unsigned)std::atoi(argv[3]) : tiny ? 6 : 20);
  coins_own_test();
  const u64 key = argc > 4 ? (u64)std::atoll(argv[4]) : 1;
  ToyCoins made_up(key); // the witness, and nothing else
  const bool short_masks = argc > 5 && !std::strcmp(argv[5], "short-masks");
  const std::vector<StageChoice> further = {StageChoice{1, 1, 14, short_masks ? 12u : 14u}, StageChoice{0, 0, 14, 14}};
  const ToyRun product = tiny ? toy_size(set, 4, 7, 7, {StageChoice{1, 1, 8, 8}, StageChoice{0, 0, 9, 9}},
                                         "product, tiny edition: a test size that reaches no security level")
                              : toy_size(set, 12, 15, short_masks ? 11 : 15, further);
  const SegmentsRun chain = tiny ? segments_size(set, 3, 6, 9, 9, {StageChoice{1, 1, 9, 9}, StageChoice{0, 0, 10, 10}},
                                                 "links, tiny edition: a test size that reaches no security level")
                                 : segments_size(set, 3, 12, 15, short_masks ? 11 : 15, further);
  const Planned &plan = links ? chain.plan : product.plan;
  if (!plan->holds()) {
    std::printf("no such parameters: %s\n", plan->why_not().c_str());
    return 2;
  }

  EngineCpu engine;
  Writer proof;
  proof.keep_list();
  Binder binder(links ? SEGMENTS_LABEL : TOY_LABEL, plan);
  std::vector<std::pair<std::string, E>> notes, *listen = nullptr; // filled while 'listen' points at it
  std::function<bool(const std::vector<u8> &, const char **)> verify;
  if (links) {
    std::vector<F> x(size_t(1) << chain.vars());
    std::vector<std::vector<F>> b(chain.links(), x);
    made_up.fill(x.data(), x.size());
    for (std::vector<F> &v : b) made_up.fill(v.data(), v.size());
    CoinInputs all;
    all.add("the public file", chain.text).add("x", x.data(), x.size());
    for (const std::vector<F> &v : b) all.add("b", v.data(), v.size());
    const std::unique_ptr<StreamCoins> coins = run_coins_that_hide_nothing<StreamCoins>(key, all.finish());
    segments_prove(chain, x, b, engine, *coins, proof);
    segments_bind(binder, chain);
    verify = [&](const std::vector<u8> &p, const char **why) { return segments_verify_file(chain.text, p.data(), p.size(), why, listen); };
  } else {
    Table abc(1, 3, product.shape.codes.base[0].ell);
    made_up.fill(abc.a.data(), 2 * abc.len);
    for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
    const Digest inputs = CoinInputs().add("the public file", product.shape.text).add("the witness", abc.a.data(), abc.a.size()).finish();
    const std::unique_ptr<StreamCoins> coins = run_coins_that_hide_nothing<StreamCoins>(key, inputs);
    toy_prove(product, abc, engine, *coins, proof);
    toy_bind(binder, product.shape, product.stages);
    verify = [&](const std::vector<u8> &p, const char **why) { return toy_verify_file(product.shape.text, p.data(), p.size(), why, listen); };
  }
  const std::vector<u8> honest = proof.bytes();
  const char *why = nullptr;
  listen = &notes;
  if (!verify(honest, &why)) {
    std::printf("FAILED: the honest proof is refused: %s\n", why);
    return 1;
  }
  listen = nullptr;
  write_file(stem + ".configuration", links ? chain.text : product.shape.text);
  write_file(stem + ".proof", honest.data(), honest.size());
  write_file(stem + ".parameters", binder.parameter_bytes().data(), binder.parameter_bytes().size());
  write_file(stem + ".statement", binder.statement_bytes().data(), binder.statement_bytes().size());
  write_file(stem + ".parameters.hex", hex(binder.parameter_bytes().data(), binder.parameter_bytes().size()) + "\n");
  {
    std::string list = "what\tthe element's three coordinates\n";
    for (const auto &n : notes) list += n.first + "\t" + decimal(n.second) + "\n";
    write_file(stem + ".targets.tsv", list);
  }

  // Rounds are counted from 1, as in a round's query. The last two columns follow from the others:
  // the largest number of 64-bit words that the verifier's message can use, and the 32-byte answer
  // blocks that hold them.
  std::string text = "round\tname\twhere\tverifier's message\tits kind's number in the query\tbytes of the prover's message\troots\tanswers\telements of E drawn\tof these not 0\t"
                     "positions drawn\tlog2 of their range\tentries drawn\twork bits\trows batched\tmost words of the verifier's message\tits blocks\n";
  for (size_t i = 0; i < plan->rounds_.size(); i++) {
    const PlanRound &r = plan->rounds_[i];
    text += std::to_string(i + 1) + "\t" + r.name + "\t" + r.where + "\t" + round_kind(r.kind) + "\t" +
            (r.kind == RoundKind::RECORD ? std::string("-") : std::to_string((unsigned)kind_in_query(r.kind))); // a stored table's own sample is no round of this chain
    for (u64 x : {(u64)r.message_bytes, (u64)r.roots, (u64)r.answers, (u64)r.draw_E, (u64)r.draw_nonzero, (u64)r.draw_positions, (u64)r.log_n,
                  (u64)r.draw_trits, (u64)r.work_bits, (u64)r.rows, most_words(r), (most_words(r) + 3) / 4})
      text += "\t" + std::to_string(x);
    text += "\n";
  }
  write_file(stem + ".plan.tsv", text);

  const std::vector<Field> fields = fields_of(proof, plan);
  text = "place\tbytes\tkind\thow many\twhat\n";
  for (const Field &f : fields)
    text += std::to_string(f.at) + "\t" + std::to_string(f.bytes) + "\t" + kind_words(f.kind) + "\t" + std::to_string(f.count) + "\t" + f.what + "\n";
  write_file(stem + ".fields.tsv", text);
  write_file(stem + ".codes.tsv", links ? codes_list(chain.codes, chain.req.last) : codes_list(product.shape.codes, product.stages));
  const ChainLists lists = chain_lists(links ? SEGMENTS_LABEL : TOY_LABEL, binder, configuration_digest({links ? chain.text : product.shape.text}), plan, honest, fields);
  write_file(stem + ".digests.tsv", lists.digests);
  write_file(stem + ".chain.tsv", lists.rounds);

  // One altered copy for each entry of `wants` (one bit flipped), then an opened symbol that is no
  // canonical element, and two wrong lengths.
  struct Want {
    const char *words;
    std::function<bool(const Field &)> is;
  };
  auto holds = [](const char *part) { return [part](const Field &f) { return !f.opened && f.what.find(part) != std::string::npos; }; };
  auto opened = [](char kind) { return [kind](const Field &f) { return f.opened && f.kind == kind; }; };
  const std::vector<Want> wants = {
      {"the chain's random bytes", holds("random bytes")},  {"a root", holds("root")},
      {"the answer to a sample", holds("sample answers")},  {"the sum of a hiding polynomial", holds("hiding sum")},
      {"a coefficient of a round polynomial", holds("round polynomial")}, {"a nonce", holds("nonce")},
      {"the word shown in the last step", holds("masked message")},       {"an opened symbol", opened('F')},
      {"the salt of an opened leaf", opened('B')},          {"a digest of a path", opened('D')},
  };
  text = "copy\twhat was changed\tplace\tthis verifier's words\n";
  int number = 0, accepted = 0;
  auto judge = [&](const std::vector<u8> &copy, const std::string &words, size_t at) {
    char tail[32];
    std::snprintf(tail, sizeof tail, ".altered-%02d.proof", ++number);
    write_file(stem + tail, copy.data(), copy.size());
    const char *reason = nullptr;
    if (verify(copy, &reason)) accepted++, reason = "ACCEPTED";
    text += std::to_string(number) + "\t" + words + "\t" + std::to_string(at) + "\t" + reason + "\n";
  };
  for (const Want &w : wants) {
    const Field *found = nullptr;
    for (const Field &f : fields)
      if (!found && w.is(f)) found = &f;
    if (!found) {
      std::printf("FAILED: the proof has no field of the kind «%s»\n", w.words);
      return 1;
    }
    std::vector<u8> copy = honest;
    copy[found->at] ^= 1;
    judge(copy, std::string(w.words) + ": lowest bit of its first byte flipped (" + found->what + ")", found->at);
  }
  for (const Field &f : fields)
    if (f.opened && f.kind == 'F') {
      std::vector<u8> copy = honest;
      for (size_t i = 0; i < F_BYTES; i++) copy[f.at + i] = 0xff; // 2^64 - 1 is above the prime
      judge(copy, "an opened symbol: all its bits set, which is no element's canonical form", f.at);
      break;
    }
  judge(std::vector<u8>(honest.begin(), honest.end() - 1), "the last byte cut off", honest.size() - 1);
  {
    std::vector<u8> copy = honest;
    copy.push_back(0);
    judge(copy, "a zero byte added at the end", honest.size());
  }
  write_file(stem + ".altered.tsv", text);
  std::printf("%s: %zu bytes accepted, %zu fields, %zu rounds; %d altered copies, %d accepted\n", name.c_str(), honest.size(), fields.size(),
              plan->rounds_.size(), number, accepted);
  return accepted ? 1 : 0;
}
