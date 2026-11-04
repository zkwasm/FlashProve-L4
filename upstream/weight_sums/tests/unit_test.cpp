// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Tests four parts of weight_sums/weight_sums.cpp on their own: its SHA-256, its widening of
// binary16 numbers to binary32, its rounding RNE(2^e * p) (to the nearest integer, ties to even),
// and which error counts when two tensors each have one. That program converts a model's stored
// weights to integers and writes their SHA-256 digests to a file. The test takes no arguments and
// exits with status 0 if every check passes, otherwise with status 1. It writes one file, in a
// temporary directory next to its executable, and removes both. It includes the program's source
// without main() (WEIGHT_SUMS_WITHOUT_MAIN), so one command builds it, from the top of the
// repository, with the options of weight_sums/build.sh except -Wall and -Wextra:
//   g++ -std=c++17 -O2 -fno-ipa-modref -ffp-contract=off -pthread -o unit_test weight_sums/tests/unit_test.cpp &&
//     ./unit_test

#define WEIGHT_SUMS_WITHOUT_MAIN
#include "../weight_sums.cpp"

#include <random>

namespace {

using namespace weight_sums;

long g_checks = 0;
long g_failures = 0;

// Counts one check. For each of the first 10 checks that fail, it prints what describe() returns.
template <typename Describe> void check(bool passed, Describe describe) {
  ++g_checks;
  if (!passed && ++g_failures <= 10) std::fprintf(stderr, "unit_test: failed: %s\n", describe().c_str());
}

// ---------------------------------------------------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------------------------------------------------

// Returns the hex digest of a message that is fed to Sha256 in pieces. The pieces' lengths cycle
// through `lengths` until the message ends, so `lengths` must contain a non-zero length.
std::string digest_in_pieces(const std::string &message, const std::vector<std::size_t> &lengths) {
  Sha256 hash;
  std::size_t fed = 0;
  for (std::size_t turn = 0; fed < message.size(); ++turn) {
    const std::size_t length = std::min(lengths[turn % lengths.size()], message.size() - fed);
    hash.update(reinterpret_cast<const std::uint8_t *>(message.data()) + fed, length);
    fed += length;
  }
  return to_hex(hash.finish());
}

// Published digests. All six are on the page "Test vectors for SHA-1, SHA-2 and SHA-3" of DI Management
// (di-mgt.com.au/sha_testvectors.html). "abc", the message of 56 bytes and one million times "a" are also TEST1,
// TEST2_1 and TEST3 of the test driver in RFC 6234, "US Secure Hash Algorithms", section 8.5. The same three were
// examples in earlier editions of FIPS 180, the Secure Hash Standard.
void test_sha256_published() {
  const struct {
    std::string message;
    const char *digest;
  } published[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
       "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
       "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
      {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
  };
  // Each of these five messages is fed in pieces of mixed lengths, some of them empty, and in equal
  // pieces: short, around one and two 64-byte blocks, longer, and the whole message at once.
  std::vector<std::vector<std::size_t>> ways = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {64, 1, 63, 65, 128, 7}, {0, 5, 0, 64, 0, 200}};
  for (const std::size_t length : {1, 3, 55, 56, 63, 64, 65, 127, 128, 129, 1000, 4096, 65536, 1000000}) {
    ways.push_back({length});
  }
  for (const auto &vector : published) {
    for (const std::vector<std::size_t> &lengths : ways) {
      check(digest_in_pieces(vector.message, lengths) == vector.digest, [&] {
        std::string text = "the SHA-256 of the published message of " + std::to_string(vector.message.size()) +
                           " bytes is wrong when the message is fed in pieces of these lengths, repeated:";
        for (const std::size_t length : lengths) text += " " + std::to_string(length);
        return text;
      });
    }
  }

  // The sixth message has 2^30 bytes, so its length in bits does not fit in 32 bits. A real model gives hash inputs of
  // this length and longer. The message is a block of 64 bytes, repeated 16,777,216 times.
  const std::string block = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno";
  std::string many;
  for (int k = 0; k < 1024; ++k) many += block;
  Sha256 hash;
  for (int k = 0; k < 16777216 / 1024; ++k)
    hash.update(reinterpret_cast<const std::uint8_t *>(many.data()), many.size());
  check(to_hex(hash.finish()) == "50e72a0e26442fe2552dc3938ac58658228c0cbfb1d2ca872ae435266fcd055e",
        [] { return std::string("the SHA-256 of the published message of 2^30 bytes is wrong"); });
}

// Checks the padding at every number of bytes that can be left over in the last block. For each
// length from 0 to 200 the test pads the message itself, as RFC 6234, section 4.1, states the rule:
// a byte 0x80, then zero bytes until the length is 56 modulo 64, then the length of the message in
// bits as 8 bytes, the most significant first. The padded message fills whole blocks, so after it
// hash_value(), which adds no padding, must equal finish() of the unpadded message.
void test_sha256_padding() {
  for (std::size_t length = 0; length <= 200; ++length) {
    std::vector<std::uint8_t> message(length);
    for (std::size_t i = 0; i < length; ++i) message[i] = static_cast<std::uint8_t>(7 * i + length);
    std::vector<std::uint8_t> padded = message;
    padded.push_back(0x80);
    while (padded.size() % 64 != 56) padded.push_back(0);
    for (int k = 7; k >= 0; --k) padded.push_back(static_cast<std::uint8_t>((8 * length) >> (8 * k)));
    Sha256 by_finish, by_hand;
    if (length != 0) by_finish.update(message.data(), length);
    by_hand.update(padded.data(), padded.size());
    check(by_finish.finish() == by_hand.hash_value(),
          [&] { return "Sha256::finish pads a message of " + std::to_string(length) + " bytes wrongly"; });
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Widening binary16 to binary32
// ---------------------------------------------------------------------------------------------------------------------

std::uint32_t bits_of(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

std::string hex(std::uint32_t bits) {
  char text[16];
  std::snprintf(text, sizeof text, "0x%x", static_cast<unsigned>(bits));
  return text;
}

// Compares widen_binary16() on all 65,536 bit patterns with the value that IEEE 754 gives to a
// binary16 number with sign s, exponent field x and fraction field f: (-1)^s * f * 2^-24 for x = 0,
// (-1)^s * (1024 + f) * 2^(x - 25) for x from 1 to 30, an infinity for x = 31 and f = 0, and a NaN
// for x = 31 and any other f. std::ldexp forms these values exactly. For a NaN the test checks
// std::isnan only, not the bits.
void test_widening() {
  for (std::uint32_t word = 0; word < 65536; ++word) {
    const bool negative = (word >> 15) != 0;
    const int x = static_cast<int>((word >> 10) & 31);
    const int f = static_cast<int>(word & 1023);
    const float widened = widen_binary16(static_cast<std::uint16_t>(word));
    bool right;
    if (x == 31 && f != 0) {
      right = std::isnan(widened);
    } else {
      float size;
      if (x == 31) {
        size = std::numeric_limits<float>::infinity();
      } else if (x == 0) {
        size = std::ldexp(static_cast<float>(f), -24);
      } else {
        size = std::ldexp(static_cast<float>(1024 + f), x - 25);
      }
      right = bits_of(widened) == bits_of(negative ? -size : size); // the bit patterns tell -0 from +0
    }
    check(right, [&] { return "widen_binary16 gives a wrong value for the bit pattern " + hex(word); });
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------------------------------------------------

// Gives |p| = significand * 2^exponent for the finite binary32 number p with these bits.
void take_apart(std::uint32_t bits, std::uint64_t &significand, int &exponent) {
  const int field = static_cast<int>((bits >> 23) & 255);
  significand = bits & 0x7FFFFF;
  exponent = -149;
  if (field != 0) {
    significand |= 0x800000;
    exponent = field - 150;
  }
}

// RNE(2^e * p) in integer arithmetic only, for the finite p with these bits and |2^e * p| < 2^62.
std::int64_t round_with_integers(std::uint32_t bits, int e) {
  std::uint64_t significand;
  int exponent;
  take_apart(bits, significand, exponent);
  const int shift = exponent + e; // 2^e * |p| = significand * 2^shift
  std::uint64_t size;
  if (shift >= 0) {
    size = significand << shift;
  } else if (shift < -25) {
    size = 0; // significand < 2^24, so 2^e * |p| < 1/2
  } else {
    const std::uint64_t unit = std::uint64_t{1} << -shift;
    const std::uint64_t whole = significand / unit;
    const std::uint64_t twice_the_rest = 2 * (significand % unit);
    const bool up = twice_the_rest > unit || (twice_the_rest == unit && whole % 2 == 1);
    size = whole + (up ? 1 : 0);
  }
  return (bits >> 31) != 0 ? -static_cast<std::int64_t>(size) : static_cast<std::int64_t>(size);
}

// Compares round_scaled() with round_with_integers() on twenty million pairs (p, e): a binary32
// number and the exponent of its scale. e is from -161 to 161, which covers every e that the
// program uses. The pairs rotate through five cases (`kind`):
//   0  a normal p with a random significand, and 2^e * |p| between 2^-7 and 2^30 where binary32 has such a p;
//   1  an exact tie, where binary32 has one: 2^e * |p| is an integer plus one half;
//   2  a p made as for kind 1, then moved one binary32 step up or down;
//   3  a subnormal p, or 0;
//   4  any finite p, with e drawn so that 2^e * |p| stays below 2^62.
// For kind 0 to 3, about one pair in eight has e = -161, -101, -100, 100, 101 or 161: the range's
// ends, the bounds of a tensor's exponent and the values just outside. The sign of p is random.
void test_rounding() {
  constexpr int kPairs = 20000000;
  std::mt19937_64 random(1); // the C++ standard fixes this generator's output, so every run has the same pairs
  const auto below = [&](std::uint64_t n) {
    return random() % n;
  };
  for (int pair = 0; pair < kPairs; ++pair) {
    const int kind = pair % 5;
    int e = static_cast<int>(below(323)) - 161;
    if (below(8) == 0) e = std::array<int, 6>{-161, -101, -100, 100, 101, 161}[below(6)];
    std::uint32_t bits;
    if (kind <= 2) {
      // A normal p, for which 2^e * |p| = significand * 2^shift. The exponent field is chosen to give the wanted shift
      // and is then limited to 2 .. 253, so that one step up or down leaves p normal and finite.
      const int wanted = kind == 0 ? static_cast<int>(below(37)) - 30 : -1 - static_cast<int>(below(24));
      const int field = std::clamp(wanted - e + 150, 2, 253);
      const int shift = field - 150 + e;
      std::uint32_t fraction = static_cast<std::uint32_t>(below(1u << 23));
      if (kind != 0 && shift >= -24 && shift <= -1) {
        // Clear the bits below one half and set the bit of one half. At shift -24 that bit is the implicit leading 1.
        const std::uint32_t half = 1u << (-shift - 1);
        fraction = ((fraction & ~(2 * half - 1)) | half) & 0x7FFFFF;
      }
      bits = static_cast<std::uint32_t>(field) << 23 | fraction;
      if (kind == 2) bits = below(2) == 0 ? bits + 1 : bits - 1;
    } else if (kind == 3) {
      bits = static_cast<std::uint32_t>(below(1u << 23));
    } else {
      bits = static_cast<std::uint32_t>(below(255u << 23));
      std::uint64_t significand;
      int exponent;
      take_apart(bits, significand, exponent);
      e = static_cast<int>(below(static_cast<std::uint64_t>(std::min(161, 38 - exponent) + 162))) - 161;
    }
    bits |= static_cast<std::uint32_t>(below(2)) << 31;

    const double rounded = round_scaled(float_from_bits(bits), power_of_two(e));
    check(std::fabs(rounded) < 0x1p62 && static_cast<std::int64_t>(rounded) == round_with_integers(bits, e), [&] {
      return "round_scaled differs from integer arithmetic for the binary32 bit pattern " + hex(bits) +
             " and e = " + std::to_string(e);
    });
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Which error counts
// ---------------------------------------------------------------------------------------------------------------------

// The exit status of the error that convert_all() throws, or 0 if none. 11 is the program's default
// for --eres, the embedding's exponent.
int status_of(const std::vector<Job> &jobs, std::size_t threads) {
  try {
    convert_all(jobs, 11, threads);
  } catch (const Failure &failure) {
    return static_cast<int>(failure.status);
  } catch (const std::bad_alloc &) {
    return static_cast<int>(Status::memory);
  }
  return 0;
}

// The destructor removes the file `path`, then its parent directory if that is empty. The parent
// must be a directory that this test made.
struct TemporaryFile {
  std::filesystem::path path;
  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(path.parent_path(), ignored);
  }
};

// If two tensors have different errors, convert_all() must report the error of the tensor that is
// earlier in its list of jobs, whichever thread finds its error first. One tensor is a million
// zeros, so its error (Status::bound, 25) is found only after all of it is read. The other starts
// with a NaN, so its error (Status::not_finite, 24) is found at once. Each order runs 50 times on
// each of 1, 2, 3 and 8 threads. Both tensors are in a temporary 4 MB file. The test writes it into
// a new directory, which mkdtemp() makes in `folder` under an unused name and with access for this
// user only, so it overwrites no file. It removes both, unless the run is killed.
void test_which_error_counts(const std::filesystem::path &folder) {
  std::string own = (folder / "unit_test.XXXXXX").string();
  const bool made = ::mkdtemp(own.data()) != nullptr;
  check(made, [&] { return "the test cannot make a folder in " + folder.string() + ": " + std::strerror(errno); });
  if (!made) return;
  const TemporaryFile file{std::filesystem::path(own) / "tensors"};
  std::vector<float> values(1000000 + 256, 0.0f);
  values[1000000] = std::numeric_limits<float>::quiet_NaN();
  std::ofstream stream(file.path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(values.data()), 4 * values.size());
  stream.close();
  check(stream.good(), [&] { return "the test cannot write its file " + file.path.string(); });
  if (!stream.good()) return;

  Tensor zeros, nan;
  zeros.name = "zeros";
  zeros.elements = 1000000;
  nan.name = "nan";
  nan.elements = 256;
  nan.position = 4 * zeros.elements;
  zeros.type = nan.type = kTypeF32;
  zeros.path = nan.path = file.path.string();
  Job slow, fast;
  slow.tensor = &zeros;
  fast.tensor = &nan;

  for (int run = 0; run < 50; ++run) {
    for (const std::size_t threads : {1, 2, 3, 8}) {
      const int slow_first = status_of({slow, fast}, threads);
      const int fast_first = status_of({fast, slow}, threads);
      check(slow_first == 25 && fast_first == 24, [&] {
        return "with errors in two tensors the first tensor's error must count: expected the statuses 25 and 24, got " +
               std::to_string(slow_first) + " and " + std::to_string(fast_first) +
               " (number of threads: " + std::to_string(threads) + ")";
      });
    }
  }
}

} // namespace

int main(int, char **argv) {
  test_sha256_published();
  test_sha256_padding();
  test_widening();
  test_rounding();
  test_which_error_counts(std::filesystem::path(argv[0]).parent_path()); // next to this program
  std::fprintf(stderr, "unit_test: %ld of %ld checks failed%s\n", g_failures, g_checks,
               g_failures > 10 ? "; the first 10 are listed above" : "");
  return g_failures == 0 ? 0 : 1;
}
