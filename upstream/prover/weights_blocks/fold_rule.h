// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are adapted from ggml (llama.cpp), which is under the MIT License. That project's notice, which
// applies to those parts:
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
//
// Parts of this file follow gguf-py (gguf/quants.py), which is under the MIT License. That project's notice:
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

// Converts a weight stored in a Q4_K, Q5_K, Q6_K, F32, Q8_0, F16 or BF16 block to its committed
// integer w (`fold` in identifiers). CPU and GPU code share the file so that both compute the same
// w. RN rounds to the nearest binary32 value; mul_rn() and sub_rn() round each step by itself. v is
// the decoded value; gamma_j is input column j's multiplier, or 1 if none; e is the tensor's
// exponent. fold() scales gamma_j by 2^e; the rule in weight_sums/README.md, "Which models, and the
// rule", scales RN(gamma_j v). The two orders agree unless a binary32 result overflows or is
// subnormal, and are not proved equal on all inputs.
// For the wider types Q8_0, F16 and BF16, value8() and value16() give v.
// The rule for converting the stored weights to integers grew out of VerInf's (J. Petrie. VerInf. Source code and
// README, github.com/JamesPetrie/VerInf, commit 89903c96 (28 August 2026); MIT License, Copyright (c) 2026 James
// Petrie; THIRD_PARTY.md, "VerInf"): as far as we can tell, the rule only, and no text but two local names in value6().
// The decoding in this file is adapted from ggml's public code (sub_scale() from get_scale_min_k4), except value16()'s
// own lines, which were written from the formats' description. The byte layout of the block formats Q4_K, Q5_K, Q6_K
// and Q8_0, and the rule that decodes them, are those of ggml, the tensor library of the llama.cpp project
// (github.com/ggml-org/llama.cpp; MIT licence, "Copyright (c) 2023-2024 The ggml authors"; the notice is in NOTICE and
// THIRD_PARTY.md): ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K) and ggml/src/ggml-quants.c
// (get_scale_min_k4; dequantize_row_q4_K, _q5_K, _q6_K); for Q8_0, ggml-quants.h at tag b2261 (see value8()); the 12
// scale bytes are drawn as a diagram on the project's wiki page "Tensor Encoding Schemes". The order of the arithmetic
// steps, and the form of code() and value6(), follow the same project's Python reader, gguf-py (gguf/quants.py;
// THIRD_PARTY.md, "gguf-py").
#pragma once
#include <stdint.h>
#include <math.h>
#include <limits.h>
#if defined(__CUDACC__)
#define FR_HD __host__ __device__
#else
#define FR_HD
#endif
namespace fr { // the fold rule described above; prover/hiding/forest.cuh also opens namespace fr
// Widens a binary16 bit pattern exactly to binary32.
FR_HD inline float half_bits(uint32_t h) {
  const int s = (h >> 15) & 1, ex = (h >> 10) & 31, ma = h & 1023;
  float v;
  if (ex == 31)
    v = ma ? NAN : INFINITY;
  else if (ex == 0)
    v = ldexpf((float)ma, -24);
  else
    v = ldexpf((float)(1024 + ma), ex - 25);
  return s ? -v : v;
}
// mul_rn(a, b) and sub_rn(a, b) return the product and the difference of two binary32 numbers, each
// rounded to nearest on its own: the operation is not fused with another one and not computed in a
// wider format. GPU code uses the rounding-mode intrinsics, which the compiler never fuses. The
// idea is public: see value_q45() in prover/weights_commitment/fold_weights.cuh. GPU code must not
// use a volatile local here: the compiler can keep such a local in the kernel's stack frame, so
// decoded weights would stay in local memory, which no caller can wipe
// (prover/common/local_memory.cuh). CPU code uses volatile temporaries.
FR_HD inline float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  volatile float x = a, y = b;
  volatile float p = x * y;
  return p;
#endif
}
FR_HD inline float sub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  volatile float x = a, y = b;
  volatile float p = x - y;
  return p;
#endif
}
// Reads the 6-bit scale *sc and minimum *m of sub-block j (0 to 7) from the 12 packed scale bytes.
FR_HD inline void sub_scale(const uint8_t *s12, int j, int *sc, int *m) {
  if (j < 4) {
    *sc = s12[j] & 63;
    *m = s12[j + 4] & 63;
  } else {
    *sc = (s12[j + 4] & 15) | ((s12[j - 4] >> 6) << 4);
    *m = (s12[j + 4] >> 4) | ((s12[j] >> 6) << 4);
  }
}
// Returns the code of element el (0 to 255) of block b: 4 bits for Q4_K (ty 0), 5 for Q5_K (ty 1).
FR_HD inline int code(int ty, const uint8_t *b, int el) {
  const int j = el >> 5, i = el & 31;
  const uint8_t by = b[(ty ? 48 : 16) + 32 * (j >> 1) + i];
  int q = (j & 1) ? (by >> 4) : (by & 15);
  if (ty) q |= ((b[16 + i] >> j) & 1) << 4;
  return q;
}
// Returns the decoded value of element el of a Q6_K block. Its 210 bytes: ql[128], qh[64], signed
// sc[16], d (binary16). v = RN(RN(d sc) q), q = (lo | hi << 4) - 32; both products are exact.
FR_HD inline float value6(const uint8_t *b, int el) {
  const int h = el >> 7, t = (el >> 5) & 3, c = el & 31;
  const int lo = (b[64 * h + 32 * (t & 1) + c] >> (4 * (t >> 1))) & 15, hi = (b[128 + 32 * h + c] >> (2 * t)) & 3,
            q = (lo | (hi << 4)) - 32, s = (int)(int8_t)b[192 + (el >> 4)];
  const float d = half_bits((uint32_t)b[208] | ((uint32_t)b[209] << 8));
  const float fs = (float)s, fq = (float)q;
  const float dl = mul_rn(d, fs);
  const float v = mul_rn(dl, fq);
  return v;
}
// Returns the decoded value of element el (0 to 255) of eight Q8_0 blocks that follow one another at b. A Q8_0 block
// has 34 bytes: the scale d (binary16), then 32 signed codes q. v = RN(d q). The product is exact in binary32, because
// d has at most 11 significant bits and q at most 8. Like the rest of this file's decoding, this routine is adapted
// from ggml's public code. Public source for the layout and the rule: llama.cpp (github.com/ggml-org/llama.cpp),
// ggml-quants.h at tag b2261, read in a third party's public mirror, not in the project's own repository:
// "#define QK8_0 32", and the structure block_q8_0, whose two members are "ggml_fp16_t d;" and "int8_t  qs[QK8_0];"; the
// project's wiki page "Tensor Encoding Schemes" and Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf): "Each
// block has 32 weights", "w = q * block_scale".
FR_HD inline float value8(const uint8_t *b, int el) {
  const uint8_t *s = b + 34 * (el >> 5);
  const float d = half_bits((uint32_t)s[0] | ((uint32_t)s[1] << 8)), fq = (float)(int)(int8_t)s[2 + (el & 31)];
  const float v = mul_rn(d, fq);
  return v;
}
// Returns element el of 256 little-endian 16-bit values at b, widened exactly to binary32: binary16 values, or, if
// bf16 is set, bfloat16 values, which are the upper 16 bits of a binary32 value. Written from the formats' description:
// Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf): F16, "16-bit standard IEEE 754 half-precision
// floating-point number"; BF16, "16-bit shortened version of the 32-bit IEEE 754 single-precision floating-point
// number". For binary16 it calls this file's half_bits(), of which the head's sentence on the decoding holds.
FR_HD inline float value16(const uint8_t *b, int el, bool bf16) {
  const uint32_t h = (uint32_t)b[2 * el] | ((uint32_t)b[2 * el + 1] << 8);
  if (!bf16) return half_bits(h);
  union {
    uint32_t u;
    float f;
  } c;
  c.u = h << 16;
  return c.f;
}
// A block's size in bytes. ty: 0 Q4_K, 1 Q5_K, 2 Q6_K, 3 F32 (256 little-endian binary32 values),
// 4 Q8_0 (eight blocks of 34 bytes), 5 F16, 6 BF16 (256 values of two bytes). 0: ty is none of these.
FR_HD inline int block_bytes(int ty) {
  return ty == 0 ? 144 : ty == 1 ? 176 : ty == 2 ? 210 : ty == 3 ? 1024 : ty == 4 ? 272 : ty == 5 || ty == 6 ? 512 : 0;
}
// Returns the decoded value v of element el; el is not checked. ty is as in block_bytes(); for any
// other ty it reads nothing and returns a NaN, so that fold() returns INT_MIN. For Q4_K and Q5_K, with
// the block's d and dmin, the sub-block's sc and m, the code q: v = RN(RN(RN(d sc) q) - RN(dmin m)).
FR_HD inline float value(int ty, const uint8_t *b, int el) {
  if (ty == 2) return value6(b, el);
  if (ty == 4) return value8(b, el);
  if (ty == 5 || ty == 6) return value16(b, el, ty == 6);
  if (ty < 0 || ty > 6) return NAN;
  if (ty == 3) {
    union {
      uint32_t u;
      float f;
    } c;
    const uint8_t *p = b + 4 * el;
    c.u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return c.f;
  }
  int sc, m;
  sub_scale(b + 4, el >> 5, &sc, &m);
  const float d = half_bits((uint32_t)b[0] | ((uint32_t)b[1] << 8)),
              dn = half_bits((uint32_t)b[2] | ((uint32_t)b[3] << 8));
  const float fs = (float)sc, fm = (float)m, fq = (float)code(ty, b, el);
  const float ds = mul_rn(d, fs);
  const float dm = mul_rn(dn, fm);
  const float t = mul_rn(ds, fq);
  const float v = sub_rn(t, dm);
  return v;
}
// Returns w (g is gamma_j): p = RN(RN(2^e g) v), rounded to the nearest integer, ties to even, or
// INT_MIN if p is not finite or |p| >= 2147483000 (just under 2^31); the caller must check for it.
FR_HD inline int fold(float v, float g, int e) {
  const float G = ldexpf(g, e);
  const float p = mul_rn(G, v);
  const double x = (double)p;
  if (!(x > -2147483000.0 && x < 2147483000.0)) return INT_MIN;
  double f = floor(x);
  const double r = x - f;
  if (r > 0.5 || (r == 0.5 && fmod(f, 2.0) != 0.0)) f += 1.0;
  return (int)f;
}
FR_HD inline int weight(int ty, const uint8_t *b, int el, float g, int e) {
  return fold(value(ty, b, el), g, e);
}
// Splits w into its balanced low limb b0 and the rest b1: w = b0 + 256 * b1 with b0 in [-128, 127].
FR_HD inline void limbs(int w, int *b0, int *b1) {
  *b0 = ((w + 128) & 255) - 128;
  *b1 = (w - *b0) / 256;
}
} // namespace fr
