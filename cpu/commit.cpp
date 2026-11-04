// Copyright 2026 Anthropic PBC (MemberCpu adapted from blocks/iop/tests/member_world.h).
// SPDX-License-Identifier: Apache-2.0
// CPU commitment prototype. Input is a public, row-major signed-byte weight matrix,
// already laid out as the verifier's W. No model importer or inference is provided.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "cp_configured.h"
#include "coins/run_coins.h"
#include "engine/engine_cpu.h"
#include "engine/sumcheck_cpu.h"
#include "frame/check_key.h"
#include "frame/check_entry.h"
#include "iop/index_cpu.h"

namespace cpu {
using namespace blk;
static std::vector<u8> read(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
static void write(const std::string &path, const std::vector<u8> &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!out) throw std::runtime_error("cannot write " + path);
}
static F digit(u8 byte) {
  const int value = byte < 128 ? byte : int(byte) - 256;
  return value < 0 ? F::zero() - from_u64(-value) : from_u64(value);
}

// Same matrix products and sumcheck as the reference fixture, with supplied
// weight bytes instead of its generated LinMatrix. No test-only interfaces.
class MemberCpu : public MemberHeavy {
 public:
  MemberCpu(const LinShape &shape, const std::vector<u8> &matrix) : shape_(shape), matrix_(matrix) {}
  void begin(const std::vector<std::vector<Tensor>> &claims) override {
    products_.clear();
    const size_t columns = size_t(1) << shape_.k_col();
    for (const auto &claim : claims) {
      std::vector<E> g(size_t(1) << shape_.k_row(), E::zero());
      for (const Tensor &t : claim) t.add_to(g.data());
      std::vector<E> w(columns, E::zero());
      for (u64 row = 0; row < shape_.matrix_rows; row++)
        for (size_t col = 0; col < columns; col++)
          w[col] += g[row] * digit(matrix_[row * columns + col]);
      products_.push_back(std::move(w));
    }
  }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars,
      const std::vector<E> &, const std::vector<std::vector<E>> &) override {
    return std::make_unique<SumcheckCpu<E>>(s, vars, products_);
  }
  std::vector<E> combine(const std::vector<E> &a) override {
    std::vector<E> w(size_t(1) << shape_.k_col(), E::zero());
    for (size_t j = 0; j < a.size(); j++)
      for (size_t col = 0; col < w.size(); col++) w[col] += a[j] * products_[j][col];
    return w;
  }
 private:
  const LinShape &shape_;
  const std::vector<u8> &matrix_;
  std::vector<std::vector<E>> products_;
};

static int commit(const Configured &c, const std::string &matrix_path, const std::string &out) {
  const LinShape &s = c.sized().once.S;
  const size_t columns = size_t(1) << s.k_col();
  const IndexShape &index = s.fold.index;
  const u128 encoded = u128(index.rows()) * index.code.n() * sizeof(F);
  std::cout << "cpu-shape rows=" << s.matrix_rows << " columns=" << columns
            << " encoded-bytes-per-table=" << static_cast<u64>(encoded) << std::endl;
  if (matrix_path == "--inspect") return 0;
  // IndexCpu materializes full codewords. Bound this prototype before allocation.
  if (encoded > (u128(512) << 20) || u128(s.matrix_rows) * columns > (u128(64) << 20))
    throw std::runtime_error("configuration exceeds this in-memory CPU prototype's table/matrix limit");
  const std::vector<u8> matrix = read(matrix_path);
  if (matrix.size() != s.matrix_rows * columns) throw std::runtime_error("matrix byte length differs from configured W");
  if (!std::filesystem::create_directory(out)) throw std::runtime_error("output must be a new directory");
  std::filesystem::create_directory(out + "/leaf");
  CoinInputs inputs;
  const Digest input_digest = inputs.title("cpu-commit-prototype").add("configuration", c.configuration())
      .add("matrix", matrix.data(), matrix.size()).finish();
  coins_own_test();
  auto coins = run_coins_without_a_seed(input_digest);
  EngineCpu engine;
  const NonceSearch search = [&engine](const ChainPrefix &prefix, unsigned bits) { return engine.search_nonce(prefix, bits); };
  Table rows(1, index.rows(), index.plain());
  for (u64 row = 0; row < rows.rows; row++)
    for (u64 y = 0; y < s.matrix_rows; y++)
      for (u64 j = 0; j < (u64(1) << s.k_lo); j++)
        rows.set(row, j + (y << s.k_lo), E::from(digit(matrix[y * columns + j + (row << s.k_lo)])));
  std::cout << "cpu-commit: encoding anchor" << std::endl;
  IndexCpu anchor(index, rows, *coins, c.configuration());
  WfCpu ranges(c.sized().ranges.x, wf_matrix_of(s), [&](u64 row, u64 col) { return digit(matrix[row * columns + col]); });
  MemberCpu products(s, matrix);
  Writer key;
  std::cout << "cpu-commit: proving key" << std::endl;
  if (!make_key(c, anchor, ranges, products, engine, *coins, search, key)) throw std::runtime_error("key generation refused matrix or column budget");
  TestedTable tested;
  const char *why = nullptr;
  Reader key_reader(key.bytes().data(), key.bytes().size());
  if (check_key(c, key_reader, tested, why) == ProofVerdict::NO) throw std::runtime_error(why ? why : "key rejected");
  write(out + "/key.bin", key.bytes());
  std::cout << "cpu-commit: encoding leaf" << std::endl;
  IndexCpu leaf(c.sized().link.S.table, rows, *coins, c.configuration());
  Writer entry;
  std::cout << "cpu-commit: proving link" << std::endl;
  if (!make_entry(c, anchor, leaf, engine, *coins, search, entry)) throw std::runtime_error("link generation refused column budget");
  TestedTable leaf_tested;
  Reader entry_reader(entry.bytes().data(), entry.bytes().size());
  if (check_entry(c, tested, entry_reader, leaf_tested, why) == ProofVerdict::NO) throw std::runtime_error(why ? why : "leaf rejected");
  write(out + "/leaf/0.bin", entry.bytes());
  const auto record = anchor.record();
  const Digest id = published_id(c.configuration(), anchor.name(), record_seeds(index, anchor.name()), record.y);
  const auto bytes = digest_bytes(id);
  std::string hex;
  for (u8 b : bytes) { char part[3]; std::snprintf(part, sizeof part, "%02x", b); hex += part; }
  write(out + "/ID.txt", std::vector<u8>(hex.begin(), hex.end()));
  std::cout << "commitment ID: " << hex << '\n';
  std::cout << "CPU KEY AND LEAF GENERATED; NO MODEL INFERENCE OR REQUEST PROOF" << std::endl;
  return 0;
}
} // namespace cpu

int main(int argc, char **argv) {
  if (argc != 4) { std::cerr << "usage: cpu_commit CONFIG MATRIX.bin|--inspect NEW_OUTPUT\n"; return 2; }
  try {
    const std::string config = argv[1];
    const auto p = cpu::read(config + "/policy.txt"), l = cpu::read(config + "/layout.txt"), k = cpu::read(config + "/KL.bin");
    if (k.size() % 4) throw std::runtime_error("KL.bin is not whole 32-bit words");
    std::vector<uint32_t> words(k.size() / 4);
    for (size_t i = 0; i < words.size(); i++)
      for (unsigned j = 0; j < 4; j++) words[i] |= uint32_t(k[4 * i + j]) << (8 * j);
    return cp_with_configured(std::string(p.begin(), p.end()), std::string(l.begin(), l.end()), words, 128, true,
      [&](const blk::Configured &c) { return cpu::commit(c, argv[2], argv[3]); });
  } catch (const std::exception &e) { std::cerr << "cpu_commit: " << e.what() << '\n'; return 2; }
}
