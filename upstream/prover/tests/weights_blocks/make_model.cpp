// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// The encoders to Q8_0, F16 and BF16 (wider(), to_binary16(), to_bfloat16()) are adapted from gguf-py 0.19.0
// (gguf/quants.py): see their comments. gguf-py is under the MIT License. That project's notice:
// MIT License
//
// Copyright (c) 2023 Georgi Gerganov
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The encoder to Q8_0 is also adapted from ggml (llama.cpp): ggml/src/ggml-quants.c, quantize_row_q8_0_ref, which does
// the same steps. ggml is under the MIT License. That project's notice, which applies to that part:
// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Terms: docs/glossary.md.

// A CPU test tool that writes a small synthetic model (`made_up` in identifiers) as the files that
// the programs cp_commit and cp_prove read: two GGUF files (shards), wsums.txt (each committed
// tensor's exponent and digest) and fold.txt (the embedding's exponent, plus the required line
// `qkrows asis`). Every byte is a function of the arguments. The tool creates <new dir>, which must
// not exist, and overwrites no file. Exit code: 0, or 2 (usage, or stop()). After a failure stop()
// removes the files that this run has created. It removes the directory too, unless main() did not
// accept it when it opened it, or its path no longer names it.
// The tool follows the file layout, the block formats and the rule FOLD
// (prover/weights_blocks/fold_rule.h) as this source tree describes them. The metadata keys are
// those of a llama4 GGUF file. FOLD converts a stored weight v to its committed integer,
// round(2^e * gamma * v): e is the tensor's exponent, gamma the input column's multiplier.
//
// With the last two arguments "--types q8_0", "--types f16" or "--types bf16" the embedding, the head and every matrix
// but the router are stored in that one type, as in a model file of 8 or 16 bits, and the router and the norms stay
// F32. The values are those of the default model, encoded anew (wider()). That loses precision, so the committed
// integers differ from form to form. F16 and BF16 truncate every value, and in Q8_0 a run's largest magnitude does not
// grow (its scale is truncated; a smaller value can round up by half a step), so the same parameter file and
// quantization scales are meant to fit all four forms. Without these arguments the files hold a mix of Q4_K, Q5_K, Q6_K
// and F32 (main()).
#include "weights_commitment/sha256.h"
#include "weights_blocks/fold_rule.h"
#include "weights_blocks/made_up_blocks.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// D: width of the residual stream; DH: of an attention head. Type: the block formats, numbered as
// in fr::block_bytes(); FILE_TYPE: GGUF's numbers.
static const int D = 5120, DH = 128, EMBEDDING_EXPONENT = 11;
enum Type { Q4_K = 0, Q5_K = 1, Q6_K = 2, F32 = 3, Q8_0 = 4, F16 = 5, BF16 = 6 }; // the numbers of fold_rule.h
// The types' numbers in a file. Public source: the package gguf 0.19.0, gguf/constants.py, GGMLQuantizationType.
static const uint32_t FILE_TYPE[7] = {12, 13, 14, 0, 8, 1, 30};

// What this run has created: the directory (its path, and a descriptor that stays open) and the
// names of the files in it. All files are created and removed through the descriptor, so they are
// in the directory that main() opened, whatever has happened to its path since.
static std::string made_directory;
static int made_descriptor = -1;
static std::vector<std::string> made;

// Whether the path still names the directory that this run made.
static bool the_path_names_it() {
  struct stat held, named;
  return made_descriptor >= 0 && !fstat(made_descriptor, &held) && !lstat(made_directory.c_str(), &named) &&
         held.st_dev == named.st_dev && held.st_ino == named.st_ino;
}

[[noreturn]] static void stop(const std::string &what) {
  printf("make_model: %s\n", what.c_str());
  for (size_t i = made.size(); i-- > 0;) unlinkat(made_descriptor, made[i].c_str(), 0);
  // The directory can be removed only by its path. Something else could still take the path between
  // the look and the call; rmdir() removes nothing but an empty directory.
  if (the_path_names_it()) rmdir(made_directory.c_str());
  exit(2);
}

// Creates the file `name` in the directory that this run made, for writing. Stops the run if anything
// has the name, a symbolic link included. Permissions are those that fopen() gives.
static FILE *create(const std::string &name) {
  const int flags = O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC;
  const int descriptor = openat(made_descriptor, name.c_str(), flags, 0666);
  if (descriptor < 0) stop(made_directory + "/" + name + " cannot be created: " + strerror(errno));
  made.push_back(name);
  FILE *f = fdopen(descriptor, "wb");
  if (!f) stop(made_directory + "/" + name + " cannot be opened for writing: " + strerror(errno));
  return f;
}

// Calls work(i) for every i < n in parallel; work must be thread-safe.
template <class Work> static void in_parallel(size_t n, Work work) {
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency() / 2); k++)
    threads.emplace_back([&] {
      for (size_t i; (i = next++) < n;) work(i);
    });
  for (auto &th : threads) th.join();
}

// `stream`: the kind of tensor, numbered in main(); 20 to 22 are the norms' gains, not committed.
// `inputs`: a row's length, a multiple of 256 (main() checks H, f, fe).
struct Tensor {
  std::string name;
  int type = F32, stream = 0, layer = 0;
  uint64_t inputs = 0, rows = 1, experts = 1;
  std::string norm;    // the norm whose gains FOLD multiplies into the weights; empty if none
  uint32_t c_word = 0; // FOLD's constant for that norm, as binary32 bits
  bool committed = true;
  uint64_t cells() const { return inputs * rows * experts; }
  uint64_t bytes() const { return cells() / 256 * fr::block_bytes(type); }
};

static float from_bits(uint32_t b) {
  float f;
  memcpy(&f, &b, 4);
  return f;
}
// One binary32 product, rounded once; volatile prevents extra precision.
static float times(float a, float b) {
  volatile float x = a, y = b, p = x * y;
  return p;
}

// The token whose embedding row ("the long row") has entries about 2^18 times as large as the other
// rows'; -1 means none. A value of the residual stream has four limbs (base-256 digits), or five
// with the line `xl 5` in the parameter file (policy.txt). With four, the row's sum of squares
// exceeds what a norm can represent: a prompt with that token leaves the model's ranges and has no
// proof.
static long long_row = -1;

// The token whose embedding row ("the aimed row") has in every column the sign of row aimed_at_row
// of aimed_at, a matrix that reads the residual stream after a norm; -1 means none. A norm
// preserves signs (its gains are positive), so the product with that matrix row is a sum of
// magnitudes, about 50 times what a random row gives. The entries are as large as the long row's,
// so that the row outweighs what the layers add to the residual stream. A norm can represent their
// sum of squares only with five limbs.
static long aimed_token = -1, aimed_at_row = 0;
static std::string aimed_at;

// The matrix in which row copied_from overwrites row copied_to ("the copied row"); an empty name
// means none. The two rows then give equal outputs at every token position: two equal logits, or
// two experts with equal router logits.
static std::string copied_in;
static long copied_from = 0, copied_to = 0;

// Writes one Q6_K block of the aimed row: the block's binary16 scale d is 1.5 x 2^5, every 8-bit
// scale is 64, and every code is +18 or -18.
template <class Negative> static void aimed_block(uint8_t *b, Negative negative) {
  memset(b, 0, 192);
  memset(b + 192, 64, 16);
  b[208] = 0, b[209] = (20u << 2) | 2u;
  for (int el = 0; el < 256; el++) {
    const int h = el >> 7, t = (el >> 5) & 3, c = el & 31, code = negative(el) ? 32 - 18 : 32 + 18;
    b[64 * h + 32 * (t & 1) + c] |= (uint8_t)((code & 15) << (4 * (t >> 1)));
    b[128 + 32 * h + c] |= (uint8_t)((code >> 4) << (2 * t));
  }
}

// Writes a Q6_K block with pseudo-random codes and scales. |d| is in [2^-13, 2^-12), or in
// [2^5, 2^6) in the long row.
static void q6_block(uint64_t seed, const Tensor &t, uint64_t row, uint64_t blk, uint8_t *out) {
  for (int i = 0; i < 26; i++) {
    const uint64_t w = made_up::word(seed, t.stream, t.layer, 0, row & 65535, blk, i) ^ made_up::mix(row >> 16);
    memcpy(out + 8 * i, &w, 8);
  }
  const uint64_t r = made_up::mix(seed + 77 + row * 256 + blk);
  const bool is_long = t.name == "token_embd.weight" && (long)row == long_row;
  const uint32_t d = (uint32_t)(r & 0x8000) | ((is_long ? 20u : 2u) << 10) | (uint32_t)((r >> 16) & 1023);
  out[208] = (uint8_t)d, out[209] = (uint8_t)(d >> 8);
}

// The tensor's GGUF bytes, before main() writes an aimed or copied row.
static std::vector<uint8_t> bytes_of(uint64_t seed, const Tensor &t) {
  std::vector<uint8_t> b(t.bytes());
  const uint64_t per_row = t.inputs / 256, rows = t.rows * t.experts;
  const int size = fr::block_bytes(t.type);
  auto part = [&](uint64_t r0, uint64_t r1) {
    for (uint64_t r = r0; r < r1; r++)
      for (uint64_t k = 0; k < per_row; k++) {
        uint8_t *out = b.data() + (r * per_row + k) * size;
        if (t.type <= Q5_K) {
          made_up::block(t.type, seed, t.stream, t.layer, r / t.rows, r % t.rows, k, out);
        } else if (t.type == Q6_K) {
          q6_block(seed, t, r, k, out);
        } else {
          for (int i = 0; i < 256; i++) {
            const uint64_t w = made_up::mix(seed + ((uint64_t)t.stream << 50) + ((uint64_t)t.layer << 40) +
                                            (r * per_row + k) * 256 + i);
            // Gains are in [0.5, 2), except those of the norm before attention, which are in
            // [1/8, 1/2) to keep attention scores small. Router weights: [-1/16, 1/16).
            const float v = t.committed
                                ? ((float)(w & 0xFFFFFF) - 8388608.f) * 0x1p-27f
                                : from_bits((t.stream == 20 ? 0x3E000000u : 0x3F000000u) + (uint32_t)(w & 0xFFFFFF));
            memcpy(out + 4 * i, &v, 4);
          }
        }
      }
  };
  const uint64_t pieces = (rows + 255) / 256;
  in_parallel(pieces, [&](size_t i) { part(i * 256, std::min(rows, (i + 1) * 256)); });
  return b;
}

// The two functions below truncate, so that no magnitude grows. A tensor's largest magnitude then stays just below
// where it was, and with it, but for a rare case, the tensor's exponent under FOLD.
// Both are adapted from gguf-py 0.19.0, gguf/quants.py (F16: the module's quantize(); BF16: BF16.quantize_blocks).
// Returns the pattern of v truncated towards zero to binary16 (IEEE 754: a sign bit, 5 exponent bits with bias 15, 10
// fraction bits); an infinity if |v| >= 65536. v must not be a NaN.
// With u = 2^ulp the distance between neighbouring binary16 numbers at |v|, and q = floor(|v| / u), the pattern's low
// 15 bits are (ulp + 24) * 1024 + q: for a subnormal result ulp is -24 and the pattern is q itself; for a normal one q
// is from 1024 to 2047, and its leading bit adds 1 to the exponent field.
static uint16_t to_binary16(float v) {
  const uint16_t sign = std::signbit(v) ? 0x8000 : 0;
  const float a = fabsf(v);
  if (a >= 65536.f) return sign | 0x7C00;
  if (a == 0.f) return sign;
  int exponent = 0; // a = m * 2^exponent with 0.5 <= m < 1
  frexpf(a, &exponent);
  const int ulp = std::max(exponent - 1, -14) - 10;
  return sign | (uint16_t)((ulp + 24) * 1024 + (int)floor(ldexp((double)a, -ulp)));
}
// Returns the pattern of v truncated towards zero to bfloat16: the upper 16 bits of its binary32 pattern (Hugging Face,
// "GGUF", https://huggingface.co/docs/hub/gguf: "16-bit shortened version of the 32-bit IEEE 754 single-precision
// floating-point number").
static uint16_t to_bfloat16(float v) {
  uint32_t n;
  memcpy(&n, &v, 4);
  return (uint16_t)(n >> 16);
}

// Returns the tensor's values, decoded from b by fold_rule.h, encoded in the type 'to': Q8_0, F16 or BF16.
// Q8_0: each run of 32 values gets the scale d = (their largest magnitude) / 127, stored as binary16, and the codes
// x / d, rounded with ties away from zero. Adapted from the llama.cpp project's public code: gguf-py 0.19.0,
// gguf/quants.py (Q8_0.quantize_blocks), and ggml, ggml/src/ggml-quants.c (quantize_row_q8_0_ref), which do the same
// steps; both round d to the nearest binary16 number, where this routine truncates it. The files are test input, so
// nothing rests on the encoder being exact.
static std::vector<uint8_t> wider(const Tensor &t, const std::vector<uint8_t> &b, int to) {
  const int size = fr::block_bytes(t.type), wide = fr::block_bytes(to);
  std::vector<uint8_t> out(t.cells() / 256 * wide);
  in_parallel(t.cells() / 256, [&](size_t k) {
    float v[256];
    for (int el = 0; el < 256; el++) v[el] = fr::value(t.type, b.data() + k * size, el);
    uint8_t *o = out.data() + k * wide;
    for (int el = 0; el < 256 && to != Q8_0; el++) {
      const uint16_t h = to == F16 ? to_binary16(v[el]) : to_bfloat16(v[el]);
      o[2 * el] = (uint8_t)h, o[2 * el + 1] = (uint8_t)(h >> 8);
    }
    for (int j = 0; j < 8 && to == Q8_0; j++) {
      float largest = 0.f;
      for (int i = 0; i < 32; i++) largest = std::max(largest, fabsf(v[32 * j + i]));
      const float d = largest / 127.f, inverse = d ? 1.f / d : 0.f;
      const uint16_t h = to_binary16(d);
      o[34 * j] = (uint8_t)h, o[34 * j + 1] = (uint8_t)(h >> 8);
      for (int i = 0; i < 32; i++) o[34 * j + 2 + i] = (uint8_t)(int8_t)roundf(times(v[32 * j + i], inverse));
    }
  });
  return out;
}

// Returns the tensor's line of wsums.txt: name, number of elements, exponent e, width of the
// integers in bits, and the SHA-256 of the integers (little-endian; 2 bytes each, 4 in the
// embedding). An empty gamma means 1. Returns an empty string if the generated values leave the
// range that FOLD allows.
static std::string sums_line(const Tensor &t, const std::vector<uint8_t> &b, const std::vector<float> &gamma) {
  const int size = fr::block_bytes(t.type);
  const bool embedding = t.name == "token_embd.weight";
  auto gain = [&](uint64_t i) {
    return gamma.empty() ? 1.f : gamma[i % t.inputs];
  };
  int e = EMBEDDING_EXPONENT;
  if (!embedding) {
    float largest = 0.f;
    for (uint64_t i = 0; i < t.cells(); i++)
      largest = std::max(largest, fabsf(times(gain(i), fr::value(t.type, b.data() + (i >> 8) * size, (int)(i & 255)))));
    if (!(largest > 0.f) || !std::isfinite(largest)) return "";
    // The largest e with round(2^e * largest) <= 4095.
    for (e = 101; e >= -100 && nearbyint(ldexp((double)largest, e)) > 4095.0; e--) {
    }
    if (e > 100 || e < -100) return "";
  }
  Sha256 sum;
  int lowest = 0, highest = 0;
  std::vector<uint8_t> piece;
  for (uint64_t i = 0; i < t.cells(); i++) {
    const int w = fr::weight(t.type, b.data() + (i >> 8) * size, (int)(i & 255), gain(i), e);
    if (w == INT_MIN || abs(w) > (embedding ? (1 << 30) - 1 : 4095)) return "";
    lowest = std::min(lowest, w), highest = std::max(highest, w);
    for (int k = 0; k < (embedding ? 4 : 2); k++) piece.push_back((uint8_t)((uint32_t)w >> (8 * k)));
    if (piece.size() >= (1u << 20) || i + 1 == t.cells()) {
      sum.update(piece.data(), piece.size());
      piece.clear();
    }
  }
  int bits = 1;
  while (lowest < -(1 << (bits - 1)) || highest >= 1 << (bits - 1)) bits++;
  return t.name + " " + std::to_string(t.cells()) + " " + std::to_string(e) + " " + std::to_string(bits) + " " +
         sum.hex();
}

struct Shape {
  int layers, H, G, f, fe, experts, V;
};

struct Writer {
  std::vector<uint8_t> head;
  template <class X> void put(X x) { head.insert(head.end(), (uint8_t *)&x, (uint8_t *)&x + sizeof(X)); }
  void text(const std::string &s) {
    put<uint64_t>(s.size());
    head.insert(head.end(), s.begin(), s.end());
  }
  int pairs = 0;
  // A metadata pair is key, value type (4: u32, 6: f32, 8: string), value.
  void number(const std::string &key, uint32_t value) { text(key), put<uint32_t>(4), put<uint32_t>(value), pairs++; }
  void real(const std::string &key, float value) { text(key), put<uint32_t>(6), put<float>(value), pairs++; }
  void word(const std::string &key, const std::string &value) { text(key), put<uint32_t>(8), text(value), pairs++; }
};

// Writes one shard: header, metadata, the table of tensors, then the tensors' bytes, each at a
// multiple of 32 bytes.
static void write_shard(const std::string &name, const Shape &s, const std::vector<Tensor> &tensors,
                        const std::map<std::string, std::vector<uint8_t>> &bytes) {
  Writer w;
  w.put<uint32_t>(0x46554747u); // "GGUF"
  w.put<uint32_t>(3);
  w.put<uint64_t>(tensors.size());
  const size_t pairs_at = w.head.size();
  w.put<uint64_t>(0);
  w.word("general.architecture", "llama4");
  w.word("general.name", "made-up");
  w.number("general.alignment", 32);
  w.number("llama4.embedding_length", D);
  w.number("llama4.block_count", s.layers);
  w.number("llama4.attention.head_count", s.H);
  w.number("llama4.attention.head_count_kv", s.G);
  w.number("llama4.attention.key_length", DH);
  w.number("llama4.expert_count", s.experts);
  w.number("llama4.feed_forward_length", s.f);
  w.number("llama4.expert_feed_forward_length", s.fe);
  w.number("llama4.vocab_size", s.V);
  w.real("llama4.attention.layer_norm_rms_epsilon", 1e-5f);
  w.real("llama4.rope.freq_base", 500000.f);
  const uint64_t pairs = w.pairs;
  memcpy(w.head.data() + pairs_at, &pairs, 8);
  uint64_t offset = 0;
  for (const Tensor &t : tensors) {
    w.text(t.name);
    const int axes = t.experts > 1 ? 3 : t.rows > 1 ? 2 : 1;
    w.put<uint32_t>(axes);
    w.put<uint64_t>(t.inputs);
    if (axes > 1) w.put<uint64_t>(t.rows);
    if (axes > 2) w.put<uint64_t>(t.experts);
    w.put<uint32_t>(FILE_TYPE[t.type]);
    w.put<uint64_t>(offset);
    offset += (t.bytes() + 31) / 32 * 32;
  }
  w.head.resize((w.head.size() + 31) / 32 * 32);
  FILE *f = create(name);
  bool good = fwrite(w.head.data(), 1, w.head.size(), f) == w.head.size();
  static const uint8_t padding[32] = {};
  for (const Tensor &t : tensors) {
    const std::vector<uint8_t> &b = bytes.at(t.name);
    good = good && fwrite(b.data(), 1, b.size(), f) == b.size();
    good = good && fwrite(padding, 1, (32 - b.size() % 32) % 32, f) == (32 - b.size() % 32) % 32;
  }
  if (fclose(f) || !good) stop(made_directory + "/" + name + " could not be written completely");
}

static void write_text(const std::string &name, const std::string &text) {
  FILE *f = create(name);
  const bool good = fwrite(text.data(), 1, text.size(), f) == text.size();
  if (fclose(f) || !good) stop(made_directory + "/" + name + " could not be written completely");
}

int main(int argc, char **argv) {
  int types = -1; // the type after --types, or -1
  if (argc >= 2 && !strcmp(argv[argc - 1], "--types")) stop("--types needs q8_0, f16 or bf16 after it");
  if (argc >= 3 && !strcmp(argv[argc - 2], "--types")) {
    const std::string word = argv[argc - 1];
    types = word == "q8_0" ? (int)Q8_0 : word == "f16" ? (int)F16 : word == "bf16" ? (int)BF16 : -1;
    if (types < 0) stop("--types takes q8_0, f16 or bf16, not \"" + word + "\"");
    argc -= 2;
  }
  if (argc != 10 && argc != 11) {
    printf("usage: make_model <new dir> <layers> <heads H> <kv heads G> <ffn width f> <expert ffn width fe> "
           "<experts in every second layer, 0 for none> <vocabulary V> <seed> [<a token whose row of the embedding is made "
           "too long for the model's ranges> | <token>:<a matrix that reads the stream, as blk.0.attn_v>:<its row, at which "
           "the token's row of the embedding is aimed> | <a committed matrix, as output>:<its row a>=<its row b, over which "
           "row a is copied>] [--types q8_0|f16|bf16: the type of the embedding, the head and every matrix but the "
           "router]\n");
    return 2;
  }
  // Trailing slashes are removed: with one, open() and lstat() would follow a symbolic link at the name.
  std::string dir = argv[1];
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  static const char *names[8] = {"layers", "H", "G", "f", "fe", "experts", "V", "seed"};
  // The bounds keep layer and expert numbers below 2^10, row numbers below 2^16 and blocks per row
  // below 2^8, as made_up::word() requires (prover/weights_blocks/made_up_blocks.h). V may be
  // larger because q6_block() also mixes in the high bits of the row number.
  static const long least[8] = {1, 2, 1, 256, 256, 0, 2, 0},
                    most[8] = {1023, 510, 510, 65280, 65280, 1023, 1 << 24, LONG_MAX};
  long number[8];
  for (int i = 0; i < 8; i++) {
    char *end = argv[2 + i];
    errno = 0;
    number[i] = strtol(argv[2 + i], &end, 10);
    if (!*argv[2 + i] || *end || errno || number[i] < least[i] || number[i] > most[i])
      stop(std::string(names[i]) + " must be a whole number from " + std::to_string(least[i]) + " to " +
           std::to_string(most[i]) + ", not \"" + argv[2 + i] + "\"");
  }
  const int layers = number[0], H = number[1], G = number[2], f = number[3], fe = number[4], experts = number[5];
  const int V = number[6];
  const uint64_t seed = number[7];
  if (H % 2 || H % G) stop("H must be even and a multiple of G");
  if (f % 256 || fe % 256) stop("f and fe must be multiples of 256");
  if (experts == 1) stop("experts must be 0, or 2 or more");
  if (argc == 11 && strchr(argv[10], '=')) {
    const std::string last = argv[10];
    const size_t colon = last.rfind(':'), equals = last.find('=');
    char *end = nullptr;
    errno = 0;
    if (colon == std::string::npos || !colon || equals < colon + 2 || equals + 1 == last.size())
      stop("a copied row is written as <matrix>:<row a>=<row b>");
    copied_in = last.substr(0, colon) + ".weight";
    copied_from = strtol(last.c_str() + colon + 1, &end, 10);
    if (*end != '=' || errno) stop("a copied row is written as <matrix>:<row a>=<row b>");
    copied_to = strtol(last.c_str() + equals + 1, &end, 10);
    if (*end || errno) stop("a copied row is written as <matrix>:<row a>=<row b>");
  } else if (argc == 11) {
    char *end = argv[10];
    errno = 0;
    long_row = strtol(argv[10], &end, 10);
    if (!*argv[10] || errno || long_row < 0 || long_row >= V) stop("the token in the last argument must be a whole number from 0 to V - 1");
    if (*end == ':') {
      const std::string rest = end + 1;
      const size_t colon = rest.rfind(':');
      if (colon == std::string::npos || colon + 1 == rest.size()) stop("an aimed row is written as <token>:<matrix>:<row>");
      aimed_token = long_row, long_row = -1, aimed_at = rest.substr(0, colon) + ".weight";
      aimed_at_row = strtol(rest.c_str() + colon + 1, &end, 10);
      if (*end || errno) stop("an aimed row is written as <token>:<matrix>:<row>");
    } else if (*end)
      stop("the last argument is a token, or <token>:<matrix>:<row>");
  }
  if (mkdir(dir.c_str(), 0700)) stop(dir + " cannot be created: " + strerror(errno));
  // The new directory is opened once and held. What is opened must be owned by this user, with no
  // access for group or others. If it is not, something else may have taken the path, or the file
  // system does not keep permissions: nothing is written, and what is at the path stays. Not
  // noticed: another such directory of this user, put at the path between mkdir() and open().
  struct stat opened;
  const int descriptor = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (descriptor < 0) stop(dir + " was created but cannot be opened: " + strerror(errno) + "; it is left as it is");
  if (fstat(descriptor, &opened) || opened.st_uid != geteuid() || (opened.st_mode & 077))
    stop(dir + " was created, but what is there is not owned by this user with no access for group or others; "
               "it is left as it is");
  made_directory = dir, made_descriptor = descriptor;

  // gamma is a norm's gain times one of these constants (binary32 bits): sqrt(D)/64, or for attn_q
  // sqrt(D)/64/sqrt(DH).
  const uint32_t C = 0x3F8F1BBDu, C_Q = 0x3DCA62C2u;
  std::vector<Tensor> all;
  auto add = [&](const std::string &name, int stream, int layer, int type, uint64_t inputs, uint64_t rows, uint64_t n,
                 const std::string &norm, uint32_t c_word) {
    Tensor t;
    t.name = name + ".weight", t.stream = stream, t.layer = layer, t.type = type, t.inputs = inputs, t.rows = rows;
    t.experts = n, t.norm = norm.empty() ? "" : norm + ".weight", t.c_word = c_word, t.committed = stream < 20;
    all.push_back(t);
  };
  // One matrix of each layer (streams 2 to 8) is Q6_K, so that the layers' matrices include that
  // type too.
  auto type = [&](int stream, int layer) {
    return (layer + stream) % 7 == 3 ? (int)Q6_K : made_up::type_of(stream, layer);
  };
  add("token_embd", 13, 0, Q6_K, D, V, 1, "", 0);
  add("output_norm", 22, 0, F32, D, 1, 1, "", 0);
  add("output", 14, 0, Q6_K, D, V, 1, "output_norm", C);
  for (int l = 0; l < layers; l++) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const bool routed = experts > 0 && (l & 1);
    const std::string shared = routed ? "_shexp" : "";
    const uint64_t width = routed ? fe : f;
    add(p + "attn_norm", 20, l, F32, D, 1, 1, "", 0);
    add(p + "ffn_norm", 21, l, F32, D, 1, 1, "", 0);
    add(p + "attn_q", 2, l, type(2, l), D, (uint64_t)H * DH, 1, p + "attn_norm", C_Q);
    add(p + "attn_k", 3, l, type(3, l), D, (uint64_t)G * DH, 1, p + "attn_norm", C);
    add(p + "attn_v", 4, l, type(4, l), D, (uint64_t)G * DH, 1, p + "attn_norm", C);
    add(p + "attn_output", 5, l, type(5, l), (uint64_t)H * DH, D, 1, "", 0);
    add(p + "ffn_gate" + shared, 6, l, type(6, l), D, width, 1, p + "ffn_norm", C);
    add(p + "ffn_up" + shared, 7, l, type(7, l), D, width, 1, p + "ffn_norm", C);
    add(p + "ffn_down" + shared, 8, l, type(8, l), width, D, 1, "", 0);
    if (!routed) continue;
    add(p + "ffn_gate_inp", 9, l, F32, D, experts, 1, p + "ffn_norm", C);
    add(p + "ffn_gate_exps", 10, l, Q4_K, D, fe, experts, p + "ffn_norm", C);
    add(p + "ffn_up_exps", 11, l, Q4_K, D, fe, experts, p + "ffn_norm", C);
    add(p + "ffn_down_exps", 12, l, made_up::type_of(12, l), fe, D, experts, "", 0);
  }

  std::map<std::string, std::vector<uint8_t>> bytes;
  for (const Tensor &t : all) bytes[t.name] = bytes_of(seed, t);
  if (aimed_token >= 0) {
    auto target = std::find_if(all.begin(), all.end(), [&](const Tensor &t) { return t.name == aimed_at; });
    if (target == all.end() || !target->committed || target->inputs != (uint64_t)D || target->norm.empty() ||
        aimed_at_row < 0 || (uint64_t)aimed_at_row >= target->rows * target->experts)
      stop("the aimed row needs a committed matrix that reads the residual stream after a norm, and one of its rows");
    const int size = fr::block_bytes(target->type);
    const uint8_t *from = bytes.at(aimed_at).data() + (uint64_t)aimed_at_row * (D / 256) * size;
    uint8_t *row = bytes.at("token_embd.weight").data() + (uint64_t)aimed_token * (D / 256) * fr::block_bytes(Q6_K);
    for (int k = 0; k < D / 256; k++)
      aimed_block(row + k * fr::block_bytes(Q6_K),
                  [&](int el) { return fr::value(target->type, from + k * size, el) < 0.f; });
  }
  if (!copied_in.empty()) {
    auto target = std::find_if(all.begin(), all.end(), [&](const Tensor &t) { return t.name == copied_in; });
    if (target == all.end() || !target->committed || std::min(copied_from, copied_to) < 0 ||
        (uint64_t)std::max(copied_from, copied_to) >= target->rows * target->experts)
      stop("the copied row needs a committed matrix and two of its rows");
    const uint64_t row_bytes = target->inputs / 256 * fr::block_bytes(target->type);
    uint8_t *rows = bytes.at(copied_in).data();
    memcpy(rows + copied_to * row_bytes, rows + copied_from * row_bytes, row_bytes);
  }
  for (Tensor &t : all)
    if (types >= 0 && t.committed && t.stream != 9) {
      bytes[t.name] = wider(t, bytes.at(t.name), types);
      t.type = types;
    }
  std::vector<Tensor> committed;
  for (const Tensor &t : all)
    if (t.committed) committed.push_back(t);
  std::vector<std::string> lines(committed.size());
  in_parallel(committed.size(), [&](size_t i) {
    const Tensor &t = committed[i];
    std::vector<float> gamma;
    if (!t.norm.empty()) {
      gamma.resize(D);
      memcpy(gamma.data(), bytes.at(t.norm).data(), (size_t)D * 4);
      for (float &g : gamma) g = times(g, from_bits(t.c_word));
    }
    lines[i] = sums_line(t, bytes.at(t.name), gamma);
  });
  for (size_t i = 0; i < lines.size(); i++)
    if (lines[i].empty())
      stop("internal error: " + committed[i].name + " leaves the range that the rule FOLD allows; try another seed");
  std::sort(lines.begin(), lines.end());

  // The first shard contains the tensors outside the layers and the first half of the layers,
  // rounded up. So with one layer the second shard contains no tensor.
  std::vector<Tensor> shard[2];
  for (const Tensor &t : all) shard[t.name.rfind("blk.", 0) == 0 && t.layer >= (layers + 1) / 2].push_back(t);
  const Shape shape{layers, H, G, f, fe, experts, V};
  write_shard("made-00001-of-00002.gguf", shape, shard[0], bytes);
  write_shard("made-00002-of-00002.gguf", shape, shard[1], bytes);
  std::string sums;
  for (const std::string &line : lines) sums += line + "\n";
  sums += "END " + std::to_string(lines.size()) + "\n";
  write_text("wsums.txt", sums);
  write_text("fold.txt", "eres " + std::to_string(EMBEDDING_EXPONENT) + "\nqkrows asis\n");
  if (!the_path_names_it())
    stop(dir + " no longer names the directory that this run made; its files are removed, the empty directory stays");
  printf("wrote %zu tensors, %zu of them committed, to %s\n", all.size(), lines.size(), dir.c_str());
  return 0;
}
