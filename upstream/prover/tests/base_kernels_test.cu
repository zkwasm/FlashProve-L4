// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Compares k_bg, k_limb9, k_hh3, k_gn1 and times_2_to() of prover/prover/base.cuh, byte for byte on
// synthetic inputs, with plain forms (namespace plain, same parameters): sums by full products of
// two field elements (F::mul), digits one byte per thread. The kernels take cheaper steps, such as
// F::smac (an element times a small integer, reduced later). In the prover k_bg and k_gn1 compute
// sums for the norms' sumchecks, k_limb9 cuts field coordinates into signed bytes for integer
// matrix products, and k_hh3 sums the feed-forward blocks' gated products. First small, irregular
// shapes, which reach every branch in the kernels' own code (of k_hh3's two conditions for 8-byte
// loads only one ever fails: see hh3()), then the full size (d = 5,120, groups of 4 layers).
// `T`: extension-field element, EDEG = 3 coordinates below the prime p = 2^64 - 2^32 + 1 (`FIELD_P`);
// `F`: its arithmetic; ext_from_signed(): an integer as an element. Limbs and digits: signed bytes, base 256.
//   prover/tests/unit.sh both tests/base_kernels_test.cu <directory>   one H100, 6 GB of its memory
//   "both": build and run; unit.sh resolves relative paths from prover/.
// Each check prints a line that ends in "ok" or "FAILS"; the exit code is 0 if none fails, else 1
// (also after a CUDA error). With the argument "time" it also prints how long both forms take at
// the full size; with "small" it skips the full size (for the sanitizers; unused in SUITE.md).
#include "prover/base.cuh"

namespace plain {
// A[(i Pd + k) T_ + t] = sum over l < xl of gam[0]^l limb_l(xst[(i d + k) T_ + t]), 0 if k >= d.
// HV[i Pd + k] = sum over t of eV[t] A[..]^2.
// i < Lam numbers the norm instances; Pd >= d is the number of rows k that each has in A and HV.
// Assumed and not checked: xl is 4 or 5, T_ = 1 << lt, blocks of 256 threads.
__global__ void k_bg(int xl, const long long *xst, int T_, int lt, int d, int Pd, int Lam, const T *gam, const T *eV,
                     T *A, T *HV) {
  __shared__ T sh[8];
  const T g1 = gam[0], g2 = F::mul(g1, g1), g3 = F::mul(g2, g1), g4 = F::mul(g2, g2);
  for (size_t q = blockIdx.x; q < (size_t)Lam * Pd; q += gridDim.x) {
    const int i = (int)(q / Pd), k = (int)(q % Pd);
    T acc = F::zero();
    for (int t = threadIdx.x; t < T_; t += 256) {
      T b = F::zero();
      if (k < d) {
        long long v = xst[((size_t)i * d + k) * T_ + t];
        int l[5];
        for (int c = 0; c < 5; c++) { // with four limbs the fifth is not read
          l[c] = (int)(((v + 128) & 255) - 128);
          v = (v - l[c]) >> 8;
        }
        b = F::add(F::add(ext_from_signed(l[0]), F::mul(g1, ext_from_signed(l[1]))), F::add(F::mul(g2, ext_from_signed(l[2])), F::mul(g3, ext_from_signed(l[3]))));
        if (xl == 5) b = F::add(b, F::mul(g4, ext_from_signed(l[4])));
        acc = F::add(acc, F::mul(eV[t], F::mul(b, b)));
      }
      A[(q << lt) + t] = b;
    }
    for (int s = 16; s; s >>= 1) {
      acc = F::add(acc, shfl_down<F>(acc, s));
    }
    if ((threadIdx.x & 31) == 0) {
      sh[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T v = F::zero();
      for (int w = 0; w < 8; w++) {
        v = F::add(v, sh[w]);
      }
      HV[q] = v;
    }
    __syncthreads();
  }
}
// Bl[((v EDEG + comp) 8 + lb) d + kk] = digit lb < 8 of coordinate comp of Cm[v d + kk].
// Assumed and not checked: the launch has at least 8 EDEG n threads, one for each byte of Bl.
__global__ void k_limb9(const T *Cm, size_t n /* V d */, int d, i8 *Bl) {
  const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (x < n * EDEG * 8) {
    const size_t kk = x % d, row = x / d, v = row / (8 * EDEG);
    const int lb = (int)(row % 8), comp = (int)(row / 8 % EDEG);
    const u64 c = coord(Cm[v * d + kk], comp);
    // Signed byte digits (-128 to 127), lowest first. Eight of them reach up to 0x7F7F7F7F7F7F7F7F;
    // a larger coordinate is split as c - p, which is negative. `number` holds c - p modulo 2^64,
    // which determines all eight digits.
    long long number = c > 0x7F7F7F7F7F7F7F7Full ? (long long)(c - FIELD_P) : (long long)c;
    int digit = 0;
    for (int i = 0; i <= lb; i++) {
      digit = (int)(((number + 128) & 255) - 128);
      number = (number - digit) >> 8;
    }
    Bl[x] = (i8)digit;
  }
}
// o[i] = x[i] times 2^(i mod turns), by i mod turns doublings.
__global__ void k_turns(const T *x, size_t n, int turns, T *o) {
  GSL(i, n) {
    T y = x[i];
    for (int s = 0; s < (int)(i % turns); s++) y = F::add(y, y);
    o[i] = y;
  }
}
// Adds to *off how many coordinates of Cm are not the sum of 256^lb times digit lb in Bl, modulo p.
__global__ void k_digits_give_the_value(const T *Cm, size_t n, int d, const i8 *Bl, unsigned long long *off) {
  GSL(x, n * EDEG) {
    const size_t at = x / EDEG, kk = at % d, v = at / d;
    const int comp = (int)(x % EDEG);
    T sum = F::zero(), power = F::one();
    for (int lb = 0; lb < 8; lb++) {
      sum = F::add(sum, F::mul(power, ext_from_signed(Bl[((v * EDEG + comp) * 8 + lb) * d + kk])));
      power = F::mul(power, ext_base(256));
    }
    if (!F::eq(sum, ext_base(coord(Cm[at], comp)))) atomicAdd(off, 1ull);
  }
}
// Hh[q] = sum over t < NT of et[t] s u for q < Lam f, where s = KH[row NT + t] - 32768 (in the
// prover the SiLU table's entry), u = KH[(Lam Pf + row) NT + t] - 32768 (the up projection) and
// row = (q / f) Pf + q % f: Lam layers of f rows, Pf apart. hh3() calls KH's entries keys.
// Assumed and not checked: blocks of 256 threads.
__global__ void k_hh3(const u16 *KH, size_t NT, int Lam, int f, int Pf, const T *et, T *Hh) {
  __shared__ T sh[8];
  const size_t half = (size_t)Lam * Pf;
  for (size_t q = blockIdx.x; q < (size_t)Lam * f; q += gridDim.x) {
    const size_t row = (q / f) * Pf + q % f;
    T acc = F::zero();
    for (size_t t = threadIdx.x; t < NT; t += 256) {
      long long s = (int)KH[row * NT + t] - 32768, u = (int)KH[(half + row) * NT + t] - 32768;
      acc = F::add(acc, F::mul(et[t], ext_from_signed(s * u)));
    }
    for (int o = 16; o; o >>= 1) {
      acc = F::add(acc, shfl_down<F>(acc, o));
    }
    if ((threadIdx.x & 31) == 0) {
      sh[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T v = F::zero();
      for (int w = 0; w < 8; w++) {
        v = F::add(v, sh[w]);
      }
      Hh[q] = v;
    }
    __syncthreads();
  }
}
// part[z T_ + t] = sum of ew[k] x[k T_ + t], k in the z-th of gridDim.z ranges below d.
// Assumed and not checked: the launch's x dimension covers all t < T_.
__global__ void k_gn1(const long long *x, int T_, int d, const T *ew, T *part) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, per = (d + gridDim.z - 1) / gridDim.z, k0 = blockIdx.z * per,
            k1 = min(d, k0 + per);
  if (t >= T_) {
    return;
  }
  T a = F::zero();
  for (int k = k0; k < k1; k++) {
    a = F::add(a, F::mul(ew[k], ext_from_signed(x[(size_t)k * T_ + t])));
  }
  part[(size_t)blockIdx.z * T_ + t] = a;
}
} // namespace plain
namespace {
int failed = 0, checks = 0;
bool timed = false;
void check(bool ok, const std::string &what) {
  printf("%s: %s\n", what.c_str(), ok ? "ok" : "FAILS");
  failed += !ok, checks++;
}
__host__ __device__ u64 mix(u64 z) { // one step of SplitMix64 (THIRD_PARTY.md, "SplitMix64")
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__global__ void k_turns(const T *x, size_t n, int turns, T *o) {
  GSL(i, n) o[i] = times_2_to(x[i], (int)(i % turns));
}
// Field elements: pseudo-random coordinates (field_from_word(), prover/field/field_names.h); about
// 6 in 61 are edge[] modulo p: 0, p - 1, 1, the largest number that eight signed byte digits reach,
// the next number, and 2^63. All six are below p, so `% FIELD_P` changes none of them.
__global__ void k_elements(T *p, size_t n, u64 salt) {
  GSL(i, n) {
    T e{};
    for (int c = 0; c < EDEG; c++) {
      const u64 z = mix(salt + i * EDEG + c);
      constexpr u64 edge[6] = {0, FIELD_P - 1, 1, 0x7F7F7F7F7F7F7F7Full, 0x7F7F7F7F7F7F7F80ull, 1ull << 63};
      u64 w = 0;
      if (z % 61 < 6) w = edge[z % 61] % FIELD_P;
      else
        for (u64 again = z; !field_from_word(mix(again), w); again++) {
        }
      set_coord(e, c, w);
    }
    p[i] = e;
  }
}
// Integers from -most to most, and now and then one of the two ends.
__global__ void k_integers(long long *p, size_t n, u64 salt, long long most) {
  GSL(i, n) {
    const u64 z = mix(salt + i);
    p[i] = z % 53 == 0 ? most : z % 53 == 1 ? -most : (long long)((z >> 1) % (2 * (u64)most + 1)) - most;
  }
}
__global__ void k_keys(u16 *p, size_t n, u64 salt) {
  GSL(i, n) {
    const u64 z = mix(salt + i);
    p[i] = z % 53 == 0 ? 0 : z % 53 == 1 ? 65535 : (u16)(z >> 7);
  }
}
__global__ void k_compare(const unsigned char *a, const unsigned char *b, size_t n, unsigned long long *differ) {
  unsigned long long d = 0;
  GSL(i, n) d += a[i] != b[i];
  atomicAdd(differ, d);
}
// Device memory: n elements and 16 guard bytes, all set to `fill`. The caller frees it.
template <class X> X *room(size_t n, int fill) {
  X *p;
  CK(cudaMalloc(&p, n * sizeof(X) + 16));
  CK(cudaMemsetAsync(p, fill, n * sizeof(X) + 16, cudaStreamPerThread));
  return p;
}
// Counts the differing bytes of a and b. The tests fill a with 0x5A and b with 0xA5, so a byte that
// neither form writes differs. behind_too needs these fills: it adds changes in the 16 guard bytes.
template <class X> unsigned long long differ(const X *a, const X *b, size_t n, bool behind_too = true) {
  static unsigned long long *out = room<unsigned long long>(1, 0);
  unsigned long long got;
  CK(cudaMemsetAsync(out, 0, 8, cudaStreamPerThread));
  k_compare<<<1024, 256>>>((const unsigned char *)a, (const unsigned char *)b, n * sizeof(X), out);
  CK(cudaMemcpyAsync(&got, out, 8, cudaMemcpyDeviceToHost, cudaStreamPerThread));
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  if (behind_too) { // neither form may write past its output: the fill bytes must be intact
    unsigned char x[16], y[16];
    CK(cudaMemcpy(x, a + n, 16, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y, b + n, 16, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 16; i++) got += x[i] != 0x5A || y[i] != 0xA5;
  }
  return got;
}
// Runs both forms; if `large` and "time" was given, runs each again and prints how long it took.
template <class Plain, class Kernel> void both(const char *name, bool large, Plain plain_form, Kernel kernel) {
  plain_form(), kernel();
  CK(cudaGetLastError());
  if (!timed || !large) return;
  cudaEvent_t e[3];
  for (cudaEvent_t &x : e) CK(cudaEventCreate(&x));
  CK(cudaEventRecord(e[0], cudaStreamPerThread));
  plain_form();
  CK(cudaEventRecord(e[1], cudaStreamPerThread));
  kernel();
  CK(cudaEventRecord(e[2], cudaStreamPerThread));
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  float of_plain, of_kernel;
  CK(cudaEventElapsedTime(&of_plain, e[0], e[1]));
  CK(cudaEventElapsedTime(&of_kernel, e[1], e[2]));
  printf("time: %s: the plain form %.0f us, the kernel %.0f us\n", name, 1000 * of_plain, 1000 * of_kernel);
  for (cudaEvent_t &x : e) CK(cudaEventDestroy(x));
}
u64 salt = 1;
// Inputs above the range of xl limbs lose the same carry in both forms.
unsigned long long bg(int xl, int lt, int d, int Pd, int Lam, int grid, bool large) {
  const int T_ = 1 << lt;
  const size_t n = (size_t)Lam * d * T_, m = (size_t)Lam * Pd;
  long long *x = room<long long>(n, 0);
  T *gam = room<T>(1, 0), *eV = room<T>(T_, 0), *A0 = room<T>(m << lt, 0x5A), *A1 = room<T>(m << lt, 0xA5),
    *H0 = room<T>(m, 0x5A), *H1 = room<T>(m, 0xA5);
  k_integers<<<1024, 256>>>(x, n, salt++, (1ll << (8 * xl - 1)) - 129); // above xl limbs' range
  k_elements<<<1, 32>>>(gam, 1, salt++), k_elements<<<64, 256>>>(eV, T_, salt++);
  both(
      "k_bg", large, [&] { plain::k_bg<<<grid, 256>>>(xl, x, T_, lt, d, Pd, Lam, gam, eV, A0, H0); },
      [&] { k_bg<<<grid, 256>>>(xl, x, T_, lt, d, Pd, Lam, gam, eV, A1, H1); });
  const unsigned long long off = differ(A0, A1, m << lt) + differ(H0, H1, m);
  for (void *p : std::initializer_list<void *>{x, gam, eV, A0, A1, H0, H1}) CK(cudaFree(p));
  return off;
}
// off_by = 1 misaligns Bl, which sends k_limb9 to its byte-by-byte branch; off_by = 4 leaves Bl the
// 4-byte alignment that the other branch needs, and no more. The digits of k_limb9 must also give
// their coordinate modulo p (plain::k_digits_give_the_value).
unsigned long long limb9(size_t V, int d, size_t off_by, bool large) {
  const size_t n = V * d, bytes = n * EDEG * 8;
  T *Cm = room<T>(n, 0);
  i8 *B0 = room<i8>(bytes + off_by, 0x5A), *B1 = room<i8>(bytes + off_by, 0xA5);
  k_elements<<<1024, 256>>>(Cm, n, salt++);
  both(
      "k_limb9", large, [&] { plain::k_limb9<<<(unsigned)((bytes + 255) / 256), 256>>>(Cm, n, d, B0 + off_by); },
      [&] { k_limb9<<<std::max(1u, limb9_blocks(n)), 256>>>(Cm, n, d, B1 + off_by); });
  unsigned long long off = differ(B0 + off_by, B1 + off_by, bytes), *wrong = room<unsigned long long>(1, 0), by_value = 0;
  plain::k_digits_give_the_value<<<1024, 256>>>(Cm, n, d, B1 + off_by, wrong);
  CK(cudaMemcpy(&by_value, wrong, 8, cudaMemcpyDeviceToHost));
  for (void *p : std::initializer_list<void *>{Cm, B0, B1, wrong}) CK(cudaFree(p));
  return off + by_value;
}
// times_2_to() against doublings (plain::k_turns).
unsigned long long turns(size_t n, int turns) {
  T *x = room<T>(n, 0), *o0 = room<T>(n, 0x5A), *o1 = room<T>(n, 0xA5);
  k_elements<<<1024, 256>>>(x, n, salt++);
  plain::k_turns<<<1024, 256>>>(x, n, turns, o0), k_turns<<<1024, 256>>>(x, n, turns, o1);
  const unsigned long long off = differ(o0, o1, n);
  for (void *p : std::initializer_list<void *>{x, o0, o1}) CK(cudaFree(p));
  return off;
}
// cudaMalloc() aligns KH: k_hh3's alignment test never fails here, and only NT = 101 reaches its
// branch without 8-byte loads.
unsigned long long hh3(size_t NT, int Lam, int f, int Pf, int grid, bool large) {
  const size_t keys = (size_t)2 * Lam * Pf * NT, rows = (size_t)Lam * f;
  u16 *KH = room<u16>(keys, 0);
  T *et = room<T>(NT, 0), *H0 = room<T>(rows, 0x5A), *H1 = room<T>(rows, 0xA5);
  k_keys<<<1024, 256>>>(KH, keys, salt++), k_elements<<<64, 256>>>(et, NT, salt++);
  both(
      "k_hh3", large, [&] { plain::k_hh3<<<grid, 256>>>(KH, NT, Lam, f, Pf, et, H0); },
      [&] { k_hh3<<<grid, 256>>>(KH, NT, Lam, f, Pf, et, H1); });
  const unsigned long long off = differ(H0, H1, rows);
  for (void *p : std::initializer_list<void *>{KH, et, H0, H1}) CK(cudaFree(p));
  return off;
}
// rows: blockDim.y of k_gn1's launch, 1 to 4.
unsigned long long gn1(int T_, int d, int slices, long long most, bool large, int rows = 1) {
  const size_t n = (size_t)d * T_;
  long long *x = room<long long>(n, 0);
  T *ew = room<T>(d, 0), *P0 = room<T>((size_t)slices * T_, 0x5A), *P1 = room<T>((size_t)slices * T_, 0xA5);
  k_integers<<<1024, 256>>>(x, n, salt++, most), k_elements<<<64, 256>>>(ew, d, salt++);
  const dim3 grid((T_ + 127) / 128, 1, slices);
  both(
      "k_gn1", large, [&] { plain::k_gn1<<<grid, 128>>>(x, T_, d, ew, P0); },
      [&] { k_gn1<<<grid, dim3(128, rows)>>>(x, T_, d, ew, P1); });
  const unsigned long long off = differ(P0, P1, (size_t)slices * T_);
  for (void *p : std::initializer_list<void *>{x, ew, P0, P1}) CK(cudaFree(p));
  return off;
}
} // namespace
int main(int argc, char **argv) {
  timed = argc > 1 && !strcmp(argv[1], "time");
  unsigned long long off = 0;
  int cases = 0;
  for (int xl : {4, 5})
    for (int lt : {5, 8, 10})
      for (int d : {1, 7, 40})
        for (int Lam : {1, 3}) off += bg(xl, lt, d, d + (d & 3) * 3, Lam, 1 + (cases * 7) % 50, false), cases++;
  check(!off, "k_bg: " + std::to_string(cases) + " small shapes, " + std::to_string(off) + " bytes differ");
  off = cases = 0;
  for (size_t V : {1, 3, 128})
    for (int d : {4, 5, 6, 64, 100, 257})
      for (size_t by : {0, 1, 4}) off += limb9(V, d, by, false), cases++;
  check(!off, "k_limb9: " + std::to_string(cases) + " small shapes, " + std::to_string(off) +
                  " bytes differ or coordinates are not what their digits give");
  off = turns((size_t)1 << 20, 400); // 400 is past the period of 2^s, grm::TURNS = 192
  check(!off, "times_2_to: exponents 0 to 399 against doublings, " + std::to_string(off) + " bytes differ");
  off = cases = 0;
  for (size_t NT : {4, 32, 100, 101, 128, 1024, 4096})
    for (int f : {1, 3, 4, 37})
      for (int Lam : {1, 2}) off += hh3(NT, Lam, f, f + (f & 1) * 5, 1 + (cases * 5) % 30, false), cases++;
  check(!off, "k_hh3: " + std::to_string(cases) + " small shapes, " + std::to_string(off) + " bytes differ");
  off = cases = 0;
  for (int T_ : {1, 100, 128, 300})
    for (int d : {1, 31, 257, 700, 9000})
      for (int slices : {1, 3, 32})
        for (long long most : {1ll, (1ll << 39), (1ll << 60)})
          off += gn1(T_, d, slices, most, false, 1 + cases % 4), cases++;
  check(!off, "k_gn1: " + std::to_string(cases) + " small shapes, " + std::to_string(off) + " bytes differ");
  // Negative control: differ() must count exactly one changed byte.
  {
    i8 *a = room<i8>(1000, 0x11), *b = room<i8>(1000, 0x11);
    CK(cudaMemsetAsync(b + 500, 0x12, 1, cudaStreamPerThread));
    check(differ(a, b, 1000, false) == 1, "negative control: one changed byte is detected");
    CK(cudaFree(a));
    CK(cudaFree(b));
  }
  if (argc > 1 && !strcmp(argv[1], "small")) {
    printf("%d of %d checks failed\n", failed, checks);
    return failed ? 1 : 0;
  }
  // As launched for 1,024 tokens in prover/prover/layer_prove.cuh (the layers) and
  // prover/head/hidden_fork.cuh (the output head).
  check(!bg(5, 10, 5120, 8192, 8, 16384, true), "k_bg at the full size (five limbs, eight instances)");
  check(!limb9(128, 5120, 0, true), "k_limb9 at the full size (128 rows of 5,120)");
  check(!hh3(1024, 4, 16384, 16384, 8192, true), "k_hh3 at the full size (four layers of 16,384 rows)");
  check(!gn1(1024, 5120, 32, 1ll << 39, true), "k_gn1 at the full size, the head's launch");
  check(!gn1(1024, 5120, 32, 1ll << 39, true, 4), "k_gn1 at the full size, the layers' launch");
  printf("%d of %d checks failed\n", failed, checks);
  return failed ? 1 : 0;
}
