// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shared code of the verifier cp_verify, including refuse() (rejection), parFor() (a parallel
// loop), file readers, the fields F_p and E, the format's constants, the standard parameter form
// (`Served`), acceptance policies, the last lines of a stage (a child process), the transcript
// `Tr`. Globals have no lock: set them outside parFor().
// Limits, assumptions, contracts and formats:
// verifier/docs/source-notes/vf_common.h.md. A key in brackets names an entry there.
#pragma once
#include "vf_earlier_hash.h"
#include "vf_accounting_record.h"
#include "frame/version.h"
#include "hash/tweak.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <system_error>
#include <atomic>
#include <mutex>
#include <map>
#include <array>
#include <memory>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cctype>
using std::string;
using std::vector;
typedef unsigned __int128 u128;
#include "base.h"
#include "field/field.h"
namespace blk {
class VerifierRounds;
}
struct Refusal {
  string name;
};
// While set, refuse() throws a `Refusal` instead of printing and exiting. parFor() sets it around
// its jobs, to report the rejection of the lowest job index; tests set it to catch a rejection.
static thread_local bool refuseThrows = false;
[[noreturn]] static void refuse(const string &bare) {
  if (refuseThrows) throw Refusal{bare.empty() ? string("internal-refusal-name") : bare};
  // Limits and notes: [L:66e592269867]
  printf("NO: %s\n", bare.c_str());
  fflush(stdout);
  exit(1);
}
// Threads that parFor() may use when this thread calls it.
static thread_local int innerThreads = 1;
// Runs fn(i) for all i < n on up to innerThreads threads: fn must be thread-safe. Every job runs,
// even after a rejection or a std::exception (its name: internal-exception-parFor); refuse() then
// gets the name from the lowest such i.
template <class Fn> static void parFor(u64 n, Fn fn) {
  const int own = innerThreads < 1 ? 1 : innerThreads;
  if (own <= 1 || n < 2) {
    const bool was1 = refuseThrows;
    u64 bad1 = ~0ull;
    string badName1;
    refuseThrows = true;
    for (u64 i = 0; i < n; i++) {
      try {
        fn(i);
      } catch (Refusal &r) {
        if (bad1 == ~0ull) {
          bad1 = i;
          badName1 = r.name;
        }
      } catch (std::exception &) {
        if (bad1 == ~0ull) {
          bad1 = i;
          badName1 = "internal-exception-parFor";
        }
      }
    }
    refuseThrows = was1;
    if (bad1 != ~0ull) refuse(badName1);
    return;
  }
  std::atomic<u64> next(0);
  std::mutex mu;
  u64 bad = ~0ull;
  string badName;
  const bool was = refuseThrows;
  const int nt = (int)std::min<u64>((u64)own, n);
  auto work = [&]() {
    refuseThrows = true;
    const int keep = innerThreads;
    innerThreads = 1;
    for (;;) {
      const u64 i = next++;
      if (i >= n) break;
      try {
        fn(i);
      } catch (Refusal &r) {
        std::lock_guard<std::mutex> l(mu);
        if (i < bad) {
          bad = i;
          badName = r.name;
        }
      } catch (std::exception &) {
        std::lock_guard<std::mutex> l(mu);
        if (i < bad) {
          bad = i;
          badName = "internal-exception-parFor";
        }
      }
    }
    innerThreads = keep;
  };
  // If a thread cannot start, the others take all jobs.
  vector<std::thread> th;
  th.reserve((size_t)nt);
  try {
    for (int t = 1; t < nt; t++) th.emplace_back(work);
  } catch (std::system_error &) {
  } catch (std::bad_alloc &) {
  }
  work();
  for (auto &t : th) t.join();
  refuseThrows = was;
  if (bad != ~0ull) refuse(badName);
}
#ifndef ENTRIES_PER_PARALLEL_JOB
#define ENTRIES_PER_PARALLEL_JOB ((u64)16384)
#endif
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Only regular files are read: a directory reports a huge length, a device or pipe may never end,
// and opening a named pipe with no writer would block. The path is checked before the open (links
// are followed) and the open file after it, because the path may have been replaced in between.
static bool pathIsRegularFile(const char *path) {
  struct stat sb;
  return !stat(path, &sb) && S_ISREG(sb.st_mode);
}
static bool openFileIsRegular(FILE *fp) {
  struct stat sb;
  return !fstat(fileno(fp), &sb) && S_ISREG(sb.st_mode);
}
// Read each file once, then parse, hash and print from those bytes: a second read could differ.
static bool readWholeFile(const char *path, string &out) {
  if (!pathIsRegularFile(path)) return false;
  FILE *fp = fopen(path, "rb");
  if (!fp) return false;
  if (!openFileIsRegular(fp)) {
    fclose(fp);
    return false;
  }
  out.clear();
  char b[65536];
  size_t n;
  while ((n = fread(b, 1, sizeof b, fp)) > 0) out.append(b, n);
  const bool ok = !ferror(fp);
  fclose(fp);
  return ok;
}
// Iterates over the lines of a text file's bytes. next() copies the next line, with its newline if
// any, into buf as a C string and returns 1; or 0 at the end, 2 for a line over cap - 1 bytes, 3
// for a zero byte in it. After 2 or 3 the caller must reject: parsed and hashed bytes could differ.
struct LineIterator {
  const string &s;
  size_t p = 0;
  explicit LineIterator(const string &x) : s(x) {}
  int next(char *buf, size_t cap) {
    if (p >= s.size()) return 0;
    const size_t e = s.find('\n', p), n = (e == string::npos ? s.size() : e + 1) - p;
    if (n > cap - 1) return 2;
    memcpy(buf, s.data() + p, n);
    buf[n] = 0;
    p += n;
    return memchr(buf, 0, n) ? 3 : 1;
  }
};
// True if l, a line as LineIterator::next() returns it, has the canonical form of a value line of the
// parameter file (policy.txt): a name (printable ASCII, no space), one space, the value in decimal
// with no sign or leading zero, one newline. So the parsed names and values determine the bytes,
// and a hash of the bytes covers the values.
static bool wordLineForm(const char *l) {
  size_t i = 0;
  while (l[i] > ' ' && l[i] < 127) i++;
  if (!i || l[i] != ' ') return false;
  i++;
  if (l[i] < '0' || l[i] > '9' || (l[i] == '0' && l[i + 1] != '\n')) return false;
  const size_t d0 = i;
  unsigned long long v = 0;
  while (l[i] >= '0' && l[i] <= '9') {
    if (i - d0 >= 10) return false;
    v = 10 * v + (unsigned long long)(l[i] - '0');
    i++;
  }
  if (v > 0xFFFFFFFFull) return false;
  return l[i] == '\n' && l[i + 1] == 0;
}
// True if l has more than 19 digits in a row.
static bool digitsBeyondScan(const char *l) {
  for (size_t run = 0; *l; l++) {
    run = *l >= '0' && *l <= '9' ? run + 1 : 0;
    if (run > 19) return true;
  }
  return false;
}
// readWords() reads a file of at least 2 * BYTES_PER_PARALLEL_READ bytes in parallel, in blocks of this size.
#ifndef BYTES_PER_PARALLEL_READ
// Can be defined smaller at compile time, so that small files are read the way large files are.
#define BYTES_PER_PARALLEL_READ ((size_t)1 << 26)
#endif
// Asks the kernel, in parallel, to map a buffer's pages before they are written. It changes no
// byte. Where the system call is missing or fails, the pages are mapped as they are written.
static void makePagesResident(void *p, size_t bytes) {
#ifdef MADV_POPULATE_WRITE
  const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
  char *a = (char *)(((uintptr_t)p + pg - 1) / pg * pg), *e = (char *)p + bytes;
  if (a >= e) return;
  const size_t len = (size_t)(e - a) / pg * pg;
  parFor((len + BYTES_PER_PARALLEL_READ - 1) / BYTES_PER_PARALLEL_READ, [&](u64 b) {
    const size_t o = b * BYTES_PER_PARALLEL_READ;
    (void)!madvise(a + o, std::min(BYTES_PER_PARALLEL_READ, len - o), MADV_POPULATE_WRITE);
  });
#else
  (void)p;
  (void)bytes;
#endif
}
static bool readBlocks(int fd, void *to, size_t bytes) {
  std::atomic<bool> ok(true);
  parFor((bytes + BYTES_PER_PARALLEL_READ - 1) / BYTES_PER_PARALLEL_READ, [&](u64 b) {
    size_t o = b * BYTES_PER_PARALLEL_READ;
    const size_t e = std::min(bytes, o + BYTES_PER_PARALLEL_READ);
    while (o < e) {
      const ssize_t g = pread(fd, (char *)to + o, e - o, (off_t)o);
      if (g < 0 && errno == EINTR) continue;
      if (g <= 0) {
        ok = false;
        return;
      }
      o += (size_t)g;
    }
  });
  return ok;
}
// Reads little-endian 32-bit words; `odd`: the length is no multiple of 4.
static bool readWords(const char *path, vector<u32> &out, bool &odd) {
  if (!pathIsRegularFile(path)) return false;
  FILE *fp = fopen(path, "rb");
  if (!fp) return false;
  if (!openFileIsRegular(fp)) {
    fclose(fp);
    return false;
  }
  const long n = fseek(fp, 0, SEEK_END) ? -1 : ftell(fp);
  if (n < 0 || fseek(fp, 0, SEEK_SET)) {
    fclose(fp);
    return false;
  }
  odd = n % 4 != 0;
  const size_t nw = (size_t)n / 4;
  const bool large = 4 * nw >= 2 * BYTES_PER_PARALLEL_READ;
  if (large) {
    out.clear();
    out.reserve(nw);
    makePagesResident(out.data(), 4 * nw);
    errno = 0;
  }
  out.resize(nw);
  size_t got = out.empty() ? 0
               : large     ? (readBlocks(fileno(fp), out.data(), 4 * nw) ? nw : 0)
                           : fread(out.data(), 4, out.size(), fp);
  fclose(fp);
  // The file is read directly into `out`, so a large file is not copied a second time. The last
  // n mod 4 bytes are not read: the caller must reject a file for which `odd` is set.
  if (got != out.size()) return false;
  const u32 one = 1;
  if (*(const u8 *)&one != 1)
    for (auto &v : out) {
      const u8 *b = (const u8 *)&v;
      v = (u32)b[0] | ((u32)b[1] << 8) | ((u32)b[2] << 16) | ((u32)b[3] << 24);
    }
  return true;
}
// The least l >= 0 with 2^l >= v. Not checked: v <= 2^63; above that the shift is undefined.
static int clog2(u64 v) {
  int l = 0;
  while ((1ull << l) < v) l++;
  return l;
}
// Exits with status 2, like a bad command line: an input file could not be opened or read in full.
// No stage has started, so no proof is rejected; a readable file of wrong length or content is
// rejected (status 1) where it is checked. The caller must clear errno before the read (0: not a
// regular file, or a short read). The file is named by role: cp_verify prints no text from argv.
[[noreturn]] static void unreadableInput(const char *role) {
  const int why = errno;
  static const char *const names[][2] = {{"policy", "policy.txt"}, {"KL", "KL.bin"}, {"layout", "layout.txt"}, {"key", "key.bin"}, {"entry", "the entry file"},
                                         {"request", "the request file"}, {"y", "the y file"}, {"proof", "the proof file"}};
  const char *what = "an input file";
  for (auto &n : names)
    if (!strcmp(n[0], role)) what = n[1];
  printf("input-unreadable-%s: %s cannot be read: %s\n", role, what, why ? strerror(why) : "it is not a regular file, or the read came up short");
  fflush(stdout);
  exit(2);
}
static void mustReadWords(const char *path, vector<u32> &out, bool &odd, const char *role) {
  errno = 0;
  if (!readWords(path, out, odd)) unreadableInput(role);
}
static void needBytes(const char *path, string &out, const char *role) {
  errno = 0;
  if (!readWholeFile(path, out)) unreadableInput(role);
}
// Rejects a text file of 2^32 bytes or more.
static void rejectLongTextFile(u64 bytes) {
  if (bytes >> 32) refuse("close-list-long");
}
#define THRESHOLD_BITS_TEXT "128"
#define REQUESTED_BITS_RULE_TEXT "largest-round-error"
#define REQUESTED_BITS_RULE_USAGE_TEXT                                                                                                   \
  "Every round's error must be at most 2^-asked, proof of work\n"                                                       \
  "    counted (the line \"" WORKLINE "..\" says how much); half a digest's bits are 128, so a call above 128 is refused."
/* By convention a digest of 2 b bits counts as b; -bits above b is rejected (cp-bits-above-digest).
   The compiled bound is stricter, and only printed (the line "accounting: as one number .."): see
   COLLISION_CONSTANT and `compiled_level` in blocks/numbers/accounting.h. */
static constexpr double DIGEST_HALF_BITS = 16.0 * DW;
static string hex64(const u32 *w) {
  char b[HEXW + 1];
  for (int i = 0; i < (int)DW; i++) snprintf(b + 8 * i, 9, "%08x", w[i]);
  return b;
}
/* True if `name` is meant for a program or machine fault: "internal-" first, or after a sub-proof's
   prefix ("G0:"). The parent tests a stage's last line (NOT CHECKED); outside a stage, `refuse`
   ends NO. Rare challenges can raise some: verifier/README.md, "Last line and exit status". */
[[maybe_unused]] static bool internalName(const string &name) {
  size_t i = 0;
  while (i < name.size() && isalnum((unsigned char)name[i])) i++;
  return name.compare(i && i < name.size() && name[i] == ':' ? i + 1 : 0, 9, "internal-") == 0;
}
// Exit statuses of cp_verify: 0 YES or WITHIN LIMITS; 1 a rejection (NO: <name>); 2 a bad command
// line, or an input file that cannot be read (unreadableInput()); 3 PART PASSED (a part-call, -key
// or -leaf, found no fault); 4 a planning call (-plan); 5 NOT CHECKED (the verifier itself failed);
// 14 NOT-A-VERIFICATION; 15 standard output could not be written.
// decimal() parses a number from the command line: 1 to 7 decimal digits and nothing else, no
// leading zero except in "0" itself, at most `max`. Otherwise it returns false; `out` is unchanged.
static bool decimal(const char *arg, u32 max, u32 &out) {
  const size_t n = strlen(arg);
  if (n < 1 || n > 7 || (n > 1 && arg[0] == '0')) return false;
  u32 v = 0;
  for (size_t i = 0; i < n; i++) {
    if (arg[i] < '0' || arg[i] > '9') return false;
    v = 10 * v + (u32)(arg[i] - '0');
  }
  if (v > max) return false;
  out = v;
  return true;
}
static bool parseRequestedBits(const char *arg, double &mb) {
  u32 v = 0;
  if (!decimal(arg, 1000000u, v)) return false;
  mb = (double)v;
  return true;
}
// The SplitMix64 mixer; not cryptographic.
// Limits and notes: [L:ef3e3f5b882a]
static u64 mix64(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// F_p, p = 2^64 - 2^32 + 1; E = F_p[x] / (x^3 - FIELD_W).
static const u64 FIELD_P = 0xFFFFFFFF00000001ull;
static constexpr int FIELD_BITS = 64;
static constexpr u64 FIELD_W = 2;
static_assert(FIELD_P == blk::P && FIELD_BITS == (int)blk::F_BITS && FIELD_W == blk::NR, "the verifier's own typing of the field is that of ../../blocks");
static_assert((~(u128)0 - blk::P) / ((u128)(blk::P - 1) * (blk::P - 1)) >= 1, "a product of two canonical words and one more word fit into 128 bits");
#define STATIC_ALWAYS_INLINE static inline __attribute__((always_inline))
// The canonical value mod p of any v < 2^128.
STATIC_ALWAYS_INLINE u64 fred(u128 v) {
  return blk::BaseField::reduce128((u64)(v >> 64), (u64)v);
}
STATIC_ALWAYS_INLINE u64 fredW(u128 v) {
  return fred(v);
}
STATIC_ALWAYS_INLINE u64 fadd(u64 a, u64 b) {
  return blk::BaseField::add(a, b);
}
STATIC_ALWAYS_INLINE u64 fsub(u64 a, u64 b) {
  return blk::BaseField::sub(a, b);
}
STATIC_ALWAYS_INLINE u64 fmul(u64 a, u64 b) {
  return fred((u128)a * b);
}
// a + x1 y1 (+ x2 y2 (+ x3 y3)), canonical. a, x1, x2, x3: any 64-bit words; y1, y2, y3: below p.
// One reduction per product: a u128 holds one product of two canonical words and one more word.
STATIC_ALWAYS_INLINE u64 fmad1(u64 a, u64 x1, u64 y1) {
  return fredW((u128)x1 * y1 + a);
}
STATIC_ALWAYS_INLINE u64 fmad2(u64 a, u64 x1, u64 y1, u64 x2, u64 y2) {
  return fadd(fmad1(a, x1, y1), fmul(x2, y2));
}
STATIC_ALWAYS_INLINE u64 fmad3(u64 a, u64 x1, u64 y1, u64 x2, u64 y2, u64 x3, u64 y3) {
  return fadd(fmad2(a, x1, y1, x2, y2), fmul(x3, y3));
}
static u64 fpow(u64 a, u64 e) {
  u64 r = 1;
  while (e) {
    if (e & 1) r = fmul(r, a);
    a = fmul(a, a);
    e >>= 1;
  }
  return r;
}
// v mod p.
static u64 fint(i64 v) {
  const u64 r = (v < 0 ? 0 - (u64)v : (u64)v) % FIELD_P;
  return v < 0 && r ? FIELD_P - r : r;
}
static_assert(DW == 8, "this format has digests of 8 words");
static constexpr u32 PROGRAM_VERSION_WORD = blk::SERVED_VERSION;
static constexpr bool versionWordAdmitted(u32 v) {
  return v == PROGRAM_VERSION_WORD;
}
// Limits and notes: [L:fefe5f20a906]
static_assert(!versionWordAdmitted(1213018161u) && !versionWordAdmitted(1213018162u) && !versionWordAdmitted(1212367669u) &&
                  !versionWordAdmitted(0x48434338u) && !versionWordAdmitted(0x48463031u) && !versionWordAdmitted(0x48463032u) &&
                  !versionWordAdmitted(2147469570u) && !versionWordAdmitted(2147465474u) && !versionWordAdmitted(2147461378u) &&
                  !versionWordAdmitted(2147457282u) && !versionWordAdmitted(2135734530u) &&
                  !versionWordAdmitted(2147473666u) && !versionWordAdmitted(1212362801u) && !versionWordAdmitted(1212367665u) &&
                  !versionWordAdmitted(2147408130u) && !versionWordAdmitted(2147277058u) && !versionWordAdmitted(2146228482u) &&
                  !versionWordAdmitted(2144131330u) && !versionWordAdmitted(2135742722u) && !versionWordAdmitted(2118965506u) &&
                  !versionWordAdmitted(1045223682u) && !versionWordAdmitted(508352770u) && !versionWordAdmitted(239917314u) &&
                  !versionWordAdmitted(105699586u) && !versionWordAdmitted(38590722u) && !versionWordAdmitted(6402u),
              "the version word of another format is accepted");
// admitWord() sets runVersionWord on the first admitted v; then it accepts only that word.
// Make the first call while no parFor() runs.
[[maybe_unused]] static u32 runVersionWord = 0;
[[maybe_unused]] static bool runVersionWordFixed = false;
[[maybe_unused]] static bool admitWord(u32 v) {
  if (runVersionWordFixed) return v == runVersionWord;
  if (!versionWordAdmitted(v)) return false;
  runVersionWord = v;
  runVersionWordFixed = true;
  return true;
}
// `has`: the parameter file has a line `version`; v: its word.
[[maybe_unused]] static void checkVersionLine(bool has, u64 v) {
  if (has && v <= 0xFFFFFFFFull && versionWordAdmitted((u32)v)) return;
  refuse("version-unknown (policy)");
}
static constexpr int CE = 3; // Components of an element of the challenge field E = F_p[x] / (x^3 - 2).
// Words of an element: component 0 low, then high, then component 1 low, then high, and so on.
static constexpr size_t WPE = 2 * CE;
static constexpr int LE = FIELD_BITS * CE; // lg |E|, rounded up; read by verifier.field_setting_test alone
// 1 / |E| > 2^-LE: 2^LE / |E| = (2^FIELD_BITS / p)^3 is 1 + 2^-30.4.
// TAU: the parameter file's `tau`.
// Set it once, on the main thread, while no parFor() runs and before anything is sized by it.
// Limits and notes: [L:84d37b91c124]
static int TAU = 160;
// Accepted tau: a multiple of 32; 2^a or 2^a + 2^b (one or two blocks of rows); at most 256 (no ground is recorded
// here; the hash chain needs no cap).
static bool tauServed(u64 tau) {
  if (tau < 32 || tau > 256 || tau % 32) return false;
  u64 first = 32;
  while (2 * first <= tau) first *= 2;
  return !((tau - first) & (tau - first - 1));
}
// The fewest requested bits with which a call can pass.
static constexpr double BAR = DW == 8 ? 128.0 : 136.0;
static_assert(BAR <= DIGEST_HALF_BITS, "a digest has at least twice as many bits as the threshold asks");
static constexpr u32 textNumber(const char *s) {
  u32 v = 0;
  for (; *s >= '0' && *s <= '9'; s++) v = 10 * v + (u32)(*s - '0');
  return v;
}
static_assert(textNumber(THRESHOLD_BITS_TEXT) == (u32)BAR, "the number in printed text is that of the constant");
static_assert(sizeof(long) == 8 && sizeof(size_t) == 8,
              "a file's length is stored in a long");
// Exit status and last line of a run where no check that ran failed but which is no verification.
// Limits and notes: [L:047ce97e2642]
static constexpr int NOT_A_VERIFICATION_STATUS = 14;
static const char *const NOT_A_VERIFICATION_TEXT = "NOT-A-VERIFICATION";
static_assert(NOT_A_VERIFICATION_STATUS == 14, "the READMEs, the page and the tests contain the number 14: change them with it");
static_assert(NOT_A_VERIFICATION_STATUS != 0 && NOT_A_VERIFICATION_STATUS != 1 && NOT_A_VERIFICATION_STATUS != 2 && NOT_A_VERIFICATION_STATUS != 3 &&
                  NOT_A_VERIFICATION_STATUS != 4 && NOT_A_VERIFICATION_STATUS != 5 && NOT_A_VERIFICATION_STATUS != 6 && NOT_A_VERIFICATION_STATUS != 7 &&
                  NOT_A_VERIFICATION_STATUS != 8 && NOT_A_VERIFICATION_STATUS != 9 && NOT_A_VERIFICATION_STATUS != 10 && NOT_A_VERIFICATION_STATUS != 11 &&
                  NOT_A_VERIFICATION_STATUS != 12 && NOT_A_VERIFICATION_STATUS != 15 && NOT_A_VERIFICATION_STATUS != 99 && NOT_A_VERIFICATION_STATUS < 126,
              "it must differ from this program's other exit statuses and from numbers that a "
              "calling program is likely to use as exit statuses of its own");
#define IDLINE "commitment ID: "
#define STAGEIDLINE "computed " IDLINE
// Limits and notes: [L:a19bbe35a24b]
#define WORKLINE "proof of work: "
// Limits and notes: [L:c33c302c9004]
#define RESULTWORD "result-v1"
// Begins a whole call's line before WORKLINE (status 0 or 14): how many key and entry rounds (not a
// request's) rest on another `CountSource` (blocks/numbers/setting.h) than THIRD, the bound within a
// third of the distance.
#define COMMITMENTLINE "commitment: "
#define ONE_TIME_LINE_START "accounting: one-time files: " // printed under -q too
#define RECORDCLAUSE "; the table has a record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given" // ends COMMITMENTLINE
// A whole call without -id prints this in place of the commitment ID: a caller who copied the
// files' ID into -id would accept whatever commitment the files belong to.
static const char *const IDHIDDEN = "not-shown-without-id";
// Exit status and start of the last line of a whole call if every check passed and a token of the
// request was checked under a policy other than exact-match, named by -policy; its kind and numbers
// follow. It is no YES: it bounds the tokens' costs and does not say that each is the model's top
// choice. The status is 0, as for YES, so that a shell sees a pass; the last line tells them apart.
static constexpr int WITHINSTATUS = 0;
static const char *const WITHIN = "WITHIN LIMITS: ";
// LBZ: a request's range checks bound Z, a random combination of a matrix's cells, by beta_z =
// 2^LBZ. They are to show |cell| < Ty = 2 beta_z, which the no-overflow conditions assume.
// LBZW: the condition "J-E" assumes each limb of an embedding weight below 2^(LBZW + 1).
static constexpr int LBZ = 17, // beta_z = 2^LBZ for request data
    LBZW = 20;
// A lookup forest: the product trees of a group of lookup calls.
static constexpr int LIDB_SEALED = 22, LIDB_MAX = 30;
// A tree of a request's lookup forests has at most 2^LIDB keys; the parameter file's optional line
// "lidb" (23 .. 30) sets it. It determines a plan: blk::configuration_digest() covers it, and must.
static int LIDB = LIDB_SEALED;
// forestShape() (vf_lookup_argument.h) caps a tree's depth dT at LIDB and at each call's n (2^n keys).
// NMINH: the optional line "nminH" of the parameter file, or 0 if absent. A lookup call of the
// output head has at least 2^nminH keys; without the line, at least 2^nmin. The line is valid only
// right after "lidb", with nmin < nminH <= NMINH_MAX and nminH >= 2. The same digest covers it.
static constexpr int NMINH_MAX = 28;
static int NMINH = 0;
// Bits of work, line "work positions": a request's own rounds, a batch's shared, the committer's.
[[maybe_unused]] static int POW = 0, POWS = 0, POWC = 0;
static_assert(LIDB_MAX < 32, "a count has 32 bits, so an identity must have fewer than 2^32 keys");
// A test build: one with a switch of the library that is for tests alone (blocks/fence.h).
// No run of a test build is accepted: goodEnding() is false in it, whatever the run has found.
#ifdef BLK_NOT_FOR_SERVING
static constexpr bool TEST_BUILD = true;
#else
static constexpr bool TEST_BUILD = false;
#endif
[[maybe_unused]] static void testBuildLine() {
  if (TEST_BUILD) printf("TEST BUILD: a switch of the library for a test or for a malicious prover is set\n");
}
// The standard parameter form: conditions on the public inputs (the last three, also on an option)
// needed for YES or WITHIN LIMITS. A run on inputs that miss one, such as a test's altered copy of
// a small example configuration, ends at best with NOT-A-VERIFICATION; a stage's second-to-last
// line ("result:") lists the conditions that it finds unmet. A field is true if met.
struct Served {
  bool experts;        // NE above 0
  bool fiveLimbs;      // line xl
  bool clamps;         // lines rcl, gcl
  bool promptSource;   // toksrc 1
  bool placementTable; // the configuration directory contains a layout file
  bool tableDigests;   // the 4 DW lines for the digests of the four public tables
  bool noRangeWords;   // neither line eg nor smax
  bool threeBytes;     // three bytes for a remainder of the division in attention
  bool ringDegree;     // D = 2,048: the committed matrix's rows are rounded up to a multiple of D
  bool oneShift;       // one attention shift per layer group: sh in KL.bin (quantization scales)
  bool onePlacement;   // the layout places qa and the residual stream alike in all layer groups
  bool softmaxTable;   // the two parameters of the softmax table: S = 12, rx = 8
  bool rotaryTable;    // rotaryServed()
  bool normShifts;     // no norm's right shift is above its scale: K <= M in KL.bin, KF <= MF
  bool twoCodes;       // the folded row and Omega have two different codes: SERVED_TWO_CODES
  bool nextToken;      // only under -next-token: r = T - 1 and e = T, so the proof is about y only
  bool exactMatch;     // exact-match; not needed under -policy, in a planning call or if r = T - 1
  bool policyNamed;    // only under -policy
};
static constexpr const char *SERVED_LAYOUT = "placement-table";
/* How round errors are counted. The parameter file's author is not the relying party. So every
   line that lets the author choose something laxer than what a standard YES means needs a
   condition of the standard form. The lines work are an exception, by design: their bits count
   toward 128 (no condition reads them; least-round= and least-round-without-work= show the
   split). Key and leaf rest on the theorem on codes, taken as
   given ("accounting: one-time files:" and "commitment:" show it; they also lean on rowcost 1,
   setting.h, and one more list size for the repeated mask code, blocks.h). A YES does
   not say on which source the proof's closeness rounds rest; "accounting:" does. Nobody has shown
   that there is no other such line. WORK_BITS_MOST caps the work at 32 bits; WORKLINE and
   "result-v1" show it;
   -no-proof-of-work (check_proof.py's too) lets no call pass if a round needs work. That g bits of
   work cut a round's error by 2^-g is assumed for this program's hash chains; see "accounting:". */
/* The line of the accounting that says on which bounds the proximity rounds of a key file's or a
   leaf file's plan rest. The stages key and leaf print it (oneTimeLineH in cp_verify.cpp) for their
   own file, the planning calls request and accounting for both (not id, not arrays). lemma,
   assumed: the two counts in blk::Accounting::closeness, which follow from policy.txt's lines group
   once-.. and table. */
static string oneTimeLine(const char *what, unsigned long long lemma, unsigned long long assumed) {
  return string(ONE_TIME_LINE_START "the plan of ") + what + ": " + std::to_string(assumed) + " closeness rounds rest on the theorem on codes (ePrint 2025/2055, Thm 4.6), taken as given, " +
         std::to_string(lemma) + " on the bound within a third of the distance; the table has a record";
}
/* A condition on which form of the test on the committed table runs. If the parameter file gives
   the folded row and Omega (two of the test's messages) the same code length and message length,
   the programs test both in one group and omit the split step, which divides one linear check
   between two groups (blocks/frame/lin_frame.h, blocks/iop/lin.h). The bounds argued on paper for
   this protocol are for the test with two codes and the split, and say nothing of the other form.
   No attack on that form is known, and nobody has searched for one. All three stages check the
   condition, on a batch's lines (group folded, omega) and the key's (once-folded, once-omega). */
static constexpr const char *SERVED_TWO_CODES =
    "two-codes (group folded and group omega or group once-folded and group once-omega have the same code length and message length. The test on the committed table then runs without its "
    "split step and the bounds argued on paper for this protocol do not cover that form. See verifier/docs/parameter-file.md)";
static constexpr unsigned WORK_BITS_MOST = 32;
// No ", ", "; " or ": " in any condition's name: servedForm() and lastLine() separate by them.
static constexpr const char *SERVED_EXACT =
    "exact-match (the request covers tokens of its own under another policy than C = theta = 2^sh with cap = 0 and "
    "sT = sh - 4 and a pass could say little. See -policy)";
static constexpr const char *SERVED_NAMED =
    "policy-named (the option -policy names another kind of policy or another C or theta or cap or s or sh than the "
    "parameter file sets. The line covered shows what the file sets)";
// Exact match: every checked token is the model's top choice. Other values in the parameter file
// can mean less, down to nothing: theta = C with cap = T accepts any tokens.
static constexpr bool acceptancePolicyIsExactMatch(u64 C, u64 theta, u64 cap, u64 sh, u64 sT) {
  return sh < 63 && C == theta && theta == (1ull << sh) && cap == 0 && sT + 4 == sh;
}
// The kind of acceptance policy that the lines set. Under the step table (sT = sh - 4), which
// bounds logit gaps: exact-match or, with any other C, theta or cap, clipped-gap. Under a table of
// exponentials: log-probability if sT = s, else another-table. The provider states s, the exponent
// of the model's logits; no equation ties it to the model. So log-probability only names sT = s:
// the table bounds the tokens' negative log-probabilities at temperature 1 only if s is right.
static constexpr const char *policyKind(u64 C, u64 theta, u64 cap, u64 s, u64 sh, u64 sT) {
  return acceptancePolicyIsExactMatch(C, theta, cap, sh, sT) ? "exact-match"
         : sT + 4 == sh                      ? "clipped-gap"
         : sT == s                           ? "log-probability"
                                             : "another-table";
}
static_assert(policyKind(256, 256, 0, 15, 8, 4)[0] == 'e' && policyKind(257, 256, 0, 15, 8, 4)[0] == 'c' &&
                  policyKind(256, 256, 1, 15, 8, 4)[0] == 'c' && policyKind(256, 256, 0, 4, 8, 4)[0] == 'e' &&
                  policyKind(300, 299, 0, 4, 8, 4)[0] == 'c' && policyKind(300, 299, 0, 15, 8, 15)[0] == 'l' &&
                  policyKind(256, 256, 0, 15, 8, 15)[0] == 'l' && policyKind(300, 299, 0, 15, 8, 14)[0] == 'a',
              "the four kinds of acceptance policy, by their first letters");
// An upper bound on how many of n checked tokens can have a cost above C.
// Such a token can be any token. The policy allows at most cap of them and, if C > 2^sh, at most
// n (theta - 2^sh) / (C - 2^sh): the costs, each counted as at most C, sum to at most theta n, and
// every cost is at least 2^sh under every table (a cost is at least the logit gap plus 2^sh).
// This does not bound the tokens that can be any token: how much a cost of at most C rules out
// depends on the model's logits, which this program does not see. Under the step table the bound is
// reached whatever the model, if the lines allow a proof at all: by that many arbitrary tokens and,
// for the rest, the model's top choices at 2^sh each.
static constexpr u64 aboveClip(u64 n, u64 C, u64 theta, u64 cap, u64 sh) {
  const u64 least = sh < 63 ? 1ull << sh : ~0ull;
  const u64 k = cap < n ? cap : n;
  if (C <= least) return k;
  const u128 room = theta > least ? (u128)n * (theta - least) / (C - least) : 0;
  return room < k ? (u64)room : k;
}
static_assert(aboveClip(32, 256, 256, 0, 8) == 0 && aboveClip(32, 256, 256, 64, 8) == 32 &&
                  aboveClip(32, 256, 256, 31, 8) == 31 && aboveClip(0, 256, 256, 64, 8) == 0 &&
                  aboveClip(32, 100000, 40000, 64, 8) == 12 && aboveClip(3, 100000, 40000, 64, 8) == 1 &&
                  aboveClip(2, 100000, 40000, 64, 8) == 0 && aboveClip(32, 100000, 40000, 3, 8) == 3 &&
                  aboveClip(32, 100000000, 4000000, 0, 8) == 0 && aboveClip(10, 300, 299, 10, 8) == 9 &&
                  aboveClip(10, 300, 200, 10, 8) == 0 && aboveClip(10, 200, 200, 4, 8) == 4,
              "the bound on the covered tokens whose cost is above C");
// Nonzero if a result under the policy that -policy names would mean nothing, by its numbers alone.
// A token costs at least 2^sh. 1: C < 2^sh (every cost counts as C); 2: theta < 2^sh (no request
// can meet it); 3: theta > C (no parameter file may set that); 4: theta = C and 0 < n <= cap (no
// token is ruled out). n: the number of checked tokens of the request; y is not counted.
static constexpr int policyEmpty(u64 n, u64 C, u64 theta, u64 cap, u64 sh) {
  const u64 least = sh < 63 ? 1ull << sh : ~0ull;
  return C < least ? 1 : theta < least ? 2 : theta > C ? 3 : theta == C && n && n <= cap ? 4 : 0;
}
static_assert(policyEmpty(1, 256, 256, 0, 8) == 0 && policyEmpty(0, 255, 255, 0, 8) == 1 && policyEmpty(0, 300, 255, 0, 8) == 2 &&
                  policyEmpty(0, 256, 257, 0, 8) == 3 && policyEmpty(3, 256, 256, 3, 8) == 4 && policyEmpty(4, 256, 256, 3, 8) == 0 &&
                  policyEmpty(0, 256, 256, 3, 8) == 0 && policyEmpty(500, 4096, 300, 10, 8) == 0 && policyEmpty(1, 128, 128, 0, 7) == 0 &&
                  policyEmpty(1, 128, 128, 0, 8) == 1,
              "the named policies that say nothing");
// What the caller expects, given by -policy: a kind (exact-match or clipped-gap) and values for the
// lines C, theta, cap, s and sh. Both imply sT = sh - 4, so these determine every line of the
// policy except F, which is always 24. In a whole call cpPolicyHeld() (cp_verify.cpp) rejects a
// parameter file with another sT before any stage starts. The run compares the kind and the five
// numbers with what the parameter file sets, each by equality (`is`).
struct PolicyNamed {
  bool given = false;
  string kind;
  u64 C = 0, theta = 0, cap = 0, s = 0, sh = 0;
  bool is(u64 fC, u64 fTheta, u64 fCap, u64 fS, u64 fSh, u64 fST) const {
    return kind == policyKind(fC, fTheta, fCap, fS, fSh, fST) && C == fC && theta == fTheta && cap == fCap &&
           s == fS && sh == fSh;
  }
};
static constexpr const char *POLICY_OPTION_FORM_TEXT =
    "-policy requires <kind>,<C>,<theta>,<cap>,<s>,<sh>: exact-match or clipped-gap and five whole "
    "numbers of at most ten digits without leading zeros, with commas and no blanks between them";
// Parses the argument of -policy (POLICY_OPTION_FORM_TEXT) into p. Returns false for any other text, without
// setting p.given; other fields of p may then have been written.
static bool policyNamedRead(const char *text, PolicyNamed &p) {
  const string t = text;
  const size_t comma = t.find(',');
  if (comma == string::npos) return false;
  p.kind = t.substr(0, comma);
  if (p.kind != "exact-match" && p.kind != "clipped-gap") return false;
  u64 *const number[5] = {&p.C, &p.theta, &p.cap, &p.s, &p.sh};
  size_t i = comma + 1;
  for (int k = 0; k < 5; k++) {
    const size_t first = i;
    for (*number[k] = 0; i < t.size() && t[i] >= '0' && t[i] <= '9'; i++)
      *number[k] = 10 * *number[k] + (u64)(t[i] - '0');
    if (i == first || i - first > 10 || (t[first] == '0' && i - first > 1)) return false;
    if (k < 4 ? i >= t.size() || t[i++] != ',' : i != t.size()) return false;
  }
  return p.given = true;
}
static constexpr bool rotaryServed(u64 theta0, u64 dh, u64 lt) {
  return (theta0 == 10000 || theta0 == 500000) && dh <= 128 && lt <= 10;
}
static string servedForm(const Served &s) {
  const std::pair<bool, const char *> clause[] = {{s.experts, "routed-experts"},
                                                  {s.fiveLimbs, "five-limbs"},
                                                  {s.clamps, "both-clamps"},
                                                  {s.promptSource, "prompt-source"},
                                                  {s.placementTable, SERVED_LAYOUT},
                                                  {s.tableDigests, "table-digests"},
                                                  {s.noRangeWords, "no-range-words"},
                                                  {s.threeBytes, "three-bytes"},
                                                  {s.ringDegree, "ring-degree"},
                                                  {s.oneShift, "one-shift"},
                                                  {s.onePlacement, "one-placement"},
                                                  {s.softmaxTable, "softmax-table"},
                                                  {s.rotaryTable, "rotary-table"},
                                                  {s.normShifts, "norm-shifts (in KL.bin or in the lines MF and KF a norm's shift K is above its scale M and its output may be 0 for every input)"},
                                                  {s.twoCodes, SERVED_TWO_CODES},
                                                  {s.nextToken, "next-token (under -next-token the request must have r = T - 1 and e = T)"},
                                                  {s.exactMatch, SERVED_EXACT},
                                                  {s.policyNamed, SERVED_NAMED}};
  string missed;
  for (auto &c : clause)
    if (!c.first) missed += (missed.empty() ? "" : ", ") + string(c.second);
  return missed;
}
// Decides whether a stage that passed its checks ends with its passing line or NOT-A-VERIFICATION
// (lastLine()). No parameter has a default, so the assertion below covers every call. The test
// build (TEST_BUILD) is not a parameter, so that no caller can leave it out.
static constexpr bool goodEnding(bool belowBar, bool notServed) { return !TEST_BUILD && !belowBar && !notServed; }
static_assert(goodEnding(false, false) == !TEST_BUILD && !goodEnding(true, false) && !goodEnding(false, true) && !goodEnding(true, true),
              "a run may be accepted only if no cause applies: fix goodEnding, not this table");
/* Records that a run needed proof of work. blocks/numbers compares every round's error with the
   parameter file's target twice, exactly: with the round's bits of work and without.
   Limits and notes: [L:32a702f1e759] */
struct WorkNote {
  bool needed = false;
  double bits = 0;
  u32 work = 0;
  string where;
};
static WorkNote stageWorkNote;
[[maybe_unused]] static void noteWorkNeeded(double bits, u32 work, const string &where) {
  if (stageWorkNote.needed && !(bits < stageWorkNote.bits)) return;
  stageWorkNote = WorkNote{true, bits, work, where};
}
[[maybe_unused]] static string workNoteLine() {
  if (!stageWorkNote.needed) return "";
  char b[64];
  snprintf(b, sizeof b, "work: %.2f + %u (", floor(100.0 * stageWorkNote.bits) / 100.0, stageWorkNote.work);
  return b + std::string("one round's error without its work, in bits, + its bits of work; ") + stageWorkNote.where + ")\n";
}
/* Reported to the parent: the error accounting of the stage's file (stage request: all three). */
static blk::Accounting acctNote;
// acctNote with this process's count of hash calls, complete once every hashing thread is joined.
[[maybe_unused]] static const blk::Accounting &acctCounted() {
  acctNote.counted = true;
  acctNote.calls = f2b::calls_total();
#if BLK_COUNT_HASH_CALLS
  acctNote.calls += blk::hash_calls_counted();
#endif
  return acctNote;
}
static constexpr int STAGE_PASSED_WITH_WORK_STATUS = 12;
// Prints a stage's last lines with one printf and returns the exit status, so that the two cannot
// disagree. The line "result:" has the caller's fields and a note; the last line is `good`, or
// NOT-A-VERIFICATION. `good` and goodStatus are used only if no cause applies (goodEnding());
// otherwise that note names every cause that applies. `fields` begins with a space; `note` is the
// caller's part of the printed note; `form` lists the unmet conditions as servedForm() does, or
// is empty. The fields pow=, pows=, powc= follow the caller's. Before these lines comes the
// accounting as text (acctLinesH(), vf_accounting_record.h), for the parent. If the run needed work, the
// line "work: .." comes directly before "result:" in both cases, and `good`, which must begin with
// "STAGE PASSED: ", becomes "STAGE PASSED WITH WORK: .." (status STAGE_PASSED_WITH_WORK_STATUS).
static int lastLine(bool belowBar, const string &form, int lidb, int nminH, const char *good, int goodStatus, const char *fields,
                      const char *note) {
  if (lidb < LIDB_SEALED || lidb > LIDB_MAX || nminH < 0 || nminH > NMINH_MAX) refuse("internal-last-line");
  // A stage with no accounting prints no record. If it passes, the parent ends with NOT CHECKED.
  const string &missed = form;
  // The note lists two optional lines that the standard parameter form does not cover, where the
  // parameter file has them.
  const string set = (lidb != LIDB_SEALED ? "lidb " + std::to_string(lidb) + "; " : "") + (nminH ? "nminH " + std::to_string(nminH) + "; " : "");
  const char *caps = set.c_str();
  const string all = string(fields) + " pow=" + std::to_string(POW) + " pows=" + std::to_string(POWS) +
                     " powc=" + std::to_string(POWC);
  const string work = acctLinesH(acctCounted()) + workNoteLine();
  if (goodEnding(belowBar, !missed.empty())) {
    static const char PASSED[] = "STAGE PASSED";
    if (stageWorkNote.needed && strncmp(good, "STAGE PASSED: ", 14)) refuse("internal-last-line");
    const string word = stageWorkNote.needed ? string(PASSED) + " WITH WORK" + (good + sizeof PASSED - 1) : string(good);
    printf("%sresult:%s (%s; %sno security level is claimed)\n%s\n", work.c_str(), all.c_str(), note, caps, word.c_str());
    return stageWorkNote.needed ? STAGE_PASSED_WITH_WORK_STATUS : goodStatus;
  }
  printf("%sresult:%s (not a verification: %s%s%s%s; %sno security level is claimed)\n%s\n", work.c_str(),
         all.c_str(), TEST_BUILD ? "the program is a test build (its first lines say which); " : "",
         belowBar ? "fewer than " THRESHOLD_BITS_TEXT " bits were requested; " : "",
         missed.empty() ? "" : ("not the standard parameter form: " + missed + "; ").c_str(), note, caps, NOT_A_VERIFICATION_TEXT);
  return NOT_A_VERIFICATION_STATUS;
}
// Every component must be below p: the arithmetic relies on that; no constructor checks it
// (blocks/ checks a proof's values as it reads them).
// Limits and notes: [L:2a77dd5f777f]
struct Ex {
  u64 c[CE];
  Ex() {
    for (int i = 0; i < CE; i++) c[i] = 0;
  }
  Ex(u64 c0, u64 c1, u64 c2) {
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
  }
};
static inline u64 eC(const Ex &x, int i) {
  return x.c[i];
}
static inline void eSet(Ex &x, int i, u64 v) {
  x.c[i] = v;
}
// eB(v): v < p as (v, 0, 0).
static Ex eB(u64 v) {
  Ex x;
  eSet(x, 0, v);
  return x;
}
static Ex eZ() {
  return Ex();
}
static Ex eO() {
  return eB(1);
}
static Ex eI(i64 v) {
  return eB(fint(v));
}
STATIC_ALWAYS_INLINE Ex operator+(Ex x, Ex y) {
  Ex r;
  for (int i = 0; i < CE; i++) eSet(r, i, fadd(eC(x, i), eC(y, i)));
  return r;
}
STATIC_ALWAYS_INLINE Ex operator-(Ex x, Ex y) {
  Ex r;
  for (int i = 0; i < CE; i++) eSet(r, i, fsub(eC(x, i), eC(y, i)));
  return r;
}
STATIC_ALWAYS_INLINE Ex eS(Ex x, u64 s) {
  Ex r;
  for (int i = 0; i < CE; i++) eSet(r, i, fmul(eC(x, i), s));
  return r;
}
STATIC_ALWAYS_INLINE bool operator==(Ex x, Ex y) {
  for (int i = 0; i < CE; i++)
    if (eC(x, i) != eC(y, i)) return false;
  return true;
}
STATIC_ALWAYS_INLINE bool isZ(Ex x) {
  u64 o = 0;
  for (int i = 0; i < CE; i++) o |= eC(x, i);
  return !o;
}
// E = F_p[x] / (x^3 - w), w = FIELD_W:
// (a, b, c)(a', b', c') = (a a' + w (b c' + c b'), a b' + b a' + w c c', a c' + b b' + c a').
// A u128 holds only one product of components, so E's product is blocks/field/field.h's.
STATIC_ALWAYS_INLINE Ex operator*(Ex x, Ex y) {
  const blk::E r = blk::E{{blk::F{x.c[0]}, blk::F{x.c[1]}, blk::F{x.c[2]}}} * blk::E{{blk::F{y.c[0]}, blk::F{y.c[1]}, blk::F{y.c[2]}}};
  return Ex(r.c[0].v, r.c[1].v, r.c[2].v);
}
// For x = 0 the result is 0.
// Limits and notes: [L:2b73e1f3d01f]
static Ex eInv(Ex x) {
  const u64 w = FIELD_W, a = x.c[0], b = x.c[1], c = x.c[2], A = fsub(fmul(a, a), fmul(w, fmul(b, c))),
            B = fsub(fmul(w, fmul(c, c)), fmul(a, b)), C = fsub(fmul(b, b), fmul(a, c)),
            N = fadd(fmul(a, A), fmul(w, fadd(fmul(b, C), fmul(c, B)))), ni = fpow(N, FIELD_P - 2);
  return Ex(fmul(A, ni), fmul(B, ni), fmul(C, ni));
}
static Ex ePow(Ex x, u64 e) {
  Ex r = eO();
  while (e) {
    if (e & 1) r = r * x;
    x = x * x;
    e >>= 1;
  }
  return r;
}
// TH, the tree hash over the verifier's own hash H (vf_earlier_hash.h). The "number" of a hash call
// is its domain-separation number.
// One level: each chunk of THPW words of w[0 .. n), the last perhaps shorter, becomes the digest
// H(num; chunk) in nx.
static void thLevel(u32 num, const u32 *w, size_t n, vector<u32> &nx) {
  const size_t nc = (n + THPW - 1) / THPW, blk = 1 << 12;
  nx.resize(DW * nc);
  parFor((nc + blk - 1) / blk, [&](u64 b) {
    const size_t e = nc < (b + 1) * blk ? nc : (b + 1) * blk;
    for (size_t i = b * blk; i < e; i++)
      earlier_hash_h_digest(nx.data() + DW * i, num, w + THPW * i, n - THPW * i < THPW ? n - THPW * i : THPW);
  });
}
static void TH(u32 out[DW], const u32 *w, size_t n) {
  if (n < DW) refuse("internal-th");
  vector<u32> cur, nx;
  thLevel(10, w, n, cur);
  while (cur.size() > DW) {
    thLevel(11, cur.data(), cur.size(), nx);
    cur.swap(nx);
  }
  if (cur.size() != DW) refuse("internal-th");
  memcpy(out, cur.data(), 4 * DW);
}
// The sub-proofs' handle on the channel of the proof's first hash chain (blocks/chain/rounds.h):
// every message that a sub-proof reads and every challenge that it draws goes through it, by the
// routines of vf_transcript_calls.h.
struct Tr {
  blk::VerifierRounds &one;
};
// Returns t with t[c] = eq(z, c) for all c < 2^len; z[0] belongs to the top bit of c.
static vector<Ex> eqTab(const Ex *z, int len) {
  vector<Ex> t(1, eO());
  for (int v = 0; v < len; v++) {
    const size_t nt = t.size();
    vector<Ex> n2(2 * nt);
    // Plain pointers skip the index checks of -D_GLIBCXX_ASSERTIONS: c < nt; n2 has 2 nt entries.
    const Ex zv = z[v];
    const Ex *a = t.data();
    Ex *d = n2.data();
    parFor((nt + ENTRIES_PER_PARALLEL_JOB - 1) / ENTRIES_PER_PARALLEL_JOB, [&](u64 b) {
      for (size_t c = b * ENTRIES_PER_PARALLEL_JOB; c < nt && c < (b + 1) * ENTRIES_PER_PARALLEL_JOB; c++) {
        Ex hi = a[c] * zv;
        d[2 * c + 1] = hi;
        d[2 * c] = a[c] - hi;
      }
    });
    t.swap(n2);
  }
  return t;
}

// Entry c of eqTab(z, len), without the table.
static Ex eqOne(const Ex *z, int len, u64 c) {
  if (len < 0 || len > 63 || (c >> len)) refuse("internal-eq-index");
  Ex t = eO();
  for (int v = 0; v < len; v++) {
    const Ex hi = t * z[v];
    t = ((c >> (len - 1 - v)) & 1) ? hi : t - hi;
  }
  return t;
}
// Round j of a product layer's sumcheck binds index bit o[j].
static vector<int> orderDefault(int mu) {
  vector<int> o(mu);
  for (int j = 0; j < mu; j++) o[j] = mu - 1 - j;
  return o;
}
// The range order: 2, 1, 0, 5, 4, 3, 8, 7, 6, then mu - 1 down to 9. Not checked here: mu >= 9 (in
// cp_verify.cpp: shape-product-layer). vf_layers_check.inc (`fork`: a sub-proof) uses it for two layers.
static vector<int> orderRange(int mu) {
  vector<int> o = {2, 1, 0, 5, 4, 3, 8, 7, 6};
  for (int v = mu - 1; v >= 9; v--) o.push_back(v);
  return o;
}
// Limits and notes: [L:2315e951af98]
struct Call {
  string name;
  int n = 0, iota = 0, k = 0, kappa = 0, fn = 0;
  u64 t0 = 0; // the table row that padding keys point to
  const vector<u64> *tau = nullptr;
  vector<u64> pub;
  vector<vector<std::pair<u64, u64>>>
      heavy; // per identity, rows with separate counts ("heavy rows"): pairs (row, count); or none
  Ex alpha;
  vector<Ex> Z, Y0, zk, zb[2];
  Ex vk, vb[2];
  u64 K() const { return 1ull << kappa; }
  u64 I() const { return 1ull << iota; }
  u64 tauAt(u64 t) const { return tau ? (*tau)[t] : t; }
};
