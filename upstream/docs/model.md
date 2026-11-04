# The integer model

This document defines Model, the function that FlashProve proves. A proof refers to a commitment ID,
which identifies the public parameters and the committed integer weights, and to a request of T
token ids. Its statement ([protocol.md](protocol.md)) is about the logits and the output token that
Model computes from those weights on those tokens in one forward pass.

Model is Llama 4 Maverick's text transformer in integer arithmetic: integer weights, fixed-point
activations with fitted power-of-two scales, and lookup tables for the non-linear functions. It
approximates the floating-point model and is not equal to it
([limits.md](limits.md#the-integer-model-is-not-the-floating-point-model)).

This document is the reference. Two programs implement it:

- `integrations/vllm/integer_model/integer_model/model.py`. Its variables are this document's
  symbols, and `IntegerModel.extend(..., record=f)` reports intermediate values under the names set
  in code font here, such as `qw`. It accepts fewer inputs (section 9). The Python files named below
  are in its directory.
- `lean/IntegerModel/Model.lean` ([the terms for what Lean proves](../lean/TERMS.md)). Each of its
  names has this document's symbol in a comment. It also takes shorter inputs (section 8).

The GPU prover is written to compute Model too. It has no option that prints an intermediate value
of its forward pass, so it is compared with `model.py` by which requests get a proof and by greedy
choices. In every comparison that `integrations/vllm/integer_model/OPERATING.md`, "What has been
run", reports, the two agreed (not on the released programs).
`integrations/vllm/integer_model/tests/lean_vectors/` writes test vectors from `model.py` and
compares `Model.lean` with it, value by value, on made-up weights at small sizes: every compared
value was equal. These are tests, not proofs.

## 1 Notation

Every quantity is an unbounded integer unless stated otherwise, so sums and products are exact.
Indices start at 0. floor rounds towards minus infinity, and a mod b is in 0 .. b - 1.

    rq(a, k)    = floor((a + 2^(k-1)) / 2^k), k >= 1    divide by 2^k, round to nearest, ties up
    clamp(a; b) = min(max(a, -b), b)
    isqrt(a)    = the largest r with r^2 <= a
    argmax      = the least index at which the maximum is attained (the first arg-max)
    low8(a)     = ((a + 128) mod 256) - 128              the balanced low byte, in -128 .. 127
    [P]         = 1 if P holds, else 0

A weight matrix W of shape a x b has the input index first: W[k][j] with k < a and j < b. For a
vector x of a entries, (x W)[j] = sum over k of x[k] W[k][j]. (`WeightSource.matrix` in `weights.py`
returns the transpose.)

The activations a, m, v and aF, and the queries and keys, are clamped to 12 bits with sign (cw = 12,
a constant of the format): clamp(x; 2,047).

Letters such as S, s, u, m, e and sc are local to a section.

## 2 Shape parameters

All parameters are public. The parameter file, `policy.txt`, states them one "name value" pair per
line, after its lines on the proof system
([verifier/docs/parameter-file.md](../verifier/docs/parameter-file.md) has its grammar). A value
written as two lines `x_lo` and `x_hi` is x_lo + 2^32 x_hi.

| symbol | line | meaning | full size |
|---|---|---|---|
| T = 2^lt | `lt` | number of token positions | 1,024 |
| L, d | `L`, `d` | number of layers; width of the residual stream | 48; 5,120 |
| n_h, n_g | `H`, `G` | query heads; key-value heads (n_g divides n_h) | 40; 8 |
| dh | `dh` | head dimension (even); dq = n_h dh, dkv = n_g dh | 128 |
| f, fe | `f`, `fe` | hidden units of a dense feed-forward layer; of one expert (f = 2 fe if NE > 0) | 16,384; 8,192 |
| NE, V | `NE`, `V` | number of routed experts; vocabulary size | 128; 202,048 |
| cs, theta0 | `cs`, `theta0` | scale exponent and base of the rotary tables | 12; 500,000 |
| expS, rx | `S`, `rx` | scale and step exponents of the softmax table | 12; 8 |
| eg, smax | `eg`, `smax` | key step exponent and bound of the SiLU table (omitted: 8; 4,095) | 8; 4,095 |

If NE > 0, the odd layers are mixture-of-experts layers and the even layers are dense. If NE = 0,
all layers are dense.

Three more lines, `rcl 1`, `gcl 1` and `xl 5`, select the router's clamp, the gate's clamp and the
wider range of the residual stream. Sections 5 to 9 describe Model with all three, as Model.lean
does; section 10 says what changes without them. The line `ut 1` states that an up projection is
limited to 16 bits (condition 7 of section 9); the verifier admits no other value, and the GPU
prover rejects any other value at its start.

## 3 Weights and quantization scales

**Weights.** The committed weights are these integer matrices. There are no bias vectors and no norm
gains (section 11 multiplies the gains into the matrices).

| matrix | shape | where |
|---|---|---|
| Wq; Wk, Wv; Wo | d x dq; d x dkv; dq x d | every layer |
| Wg, Wu; Wd | d x f; f x d | dense layer |
| Wg, Wu; Wd | d x fe; fe x d | mixture-of-experts layer: the shared expert |
| Wr | d x NE | mixture-of-experts layer: the router |
| Wgr_e, Wur_e; Wdr_e | d x fe; fe x d | mixture-of-experts layer: routed expert e < NE |
| Emb; Wh | V x d; d x V | whole model: embedding; output matrix |

In a vector of heads, v[h][c], entry c of head h, is entry h dh + c. The outputs of Wq, Wk and Wv
and the inputs of Wo are such vectors.

The formulas hold for any integers. A commitment's [key file](glossary.md#key-file) shows that every
entry outside Emb is in -4,096 .. 4,095, and that an entry of Emb is a number of four signed bytes
in base 256, so in -2,155,905,152 .. 2,139,062,143; it shows no narrower range. The programs that
make or load the weights accept less of Emb: entries below 2^30 in absolute value (section 11.4).
The parameter file's line `wb_emb 21` is a constant of the format: the verifier compares it with 21
and uses it in no test.

**Quantization scales.** The file `KL.bin` holds unsigned little-endian 32-bit words. First, 13
words for each layer, layers in increasing order of the layer number, in this order:

- kq, kk, kv, ko, kg, ku, kd: right shifts after the products with Wq, Wk, Wv, Wo, Wg, Wu, Wd (kq
  and kk act after the rotary step and also remove its factor 2^cs);
- M1, K1, M2, K2: exponents of the layer's two norms;
- sh: the right shift that forms a softmax key;
- rp: 1 if the layer applies the rotary encoding, else 0. In Maverick rp = 0 where l mod 4 = 3. The
  model of the Lean soundness theorem fixes this pattern.

After all of these, 3 words for each mixture-of-experts layer, in increasing order of the layer
number: kr, kgR, kuR, the right shifts after the products with Wr, Wgr and Wur.

The parameter file holds `MF` and `KF`, the final norm's exponents, and the regularizers eps1, eps2
and epsF (lines `eps1_lo`, `eps1_hi` and so on) of every layer's first norm, every layer's second
norm and the final norm. (Model.lean allows a separate eps1 and eps2 for each layer.)

Model needs every shift and every eps to be at least 1. The programs accept narrower ranges
(`parameters.py`; for the verifier, [protocol.md](protocol.md)), which are not part of Model.

## 4 Public tables

Model reads five tables of integers. Each entry results from a fixed sequence of IEEE 754 float64
operations, each rounded once, with correctly rounded exp, cos, sin and pow. Conversions of integers
to float64, and multiplications and divisions by powers of two, are exact. Nothing is fused,
reordered or evaluated in higher precision.

- rha(v) = floor(v + 0.5) if v >= 0, else -floor(-v + 0.5), with a float64 addition: ties away from
  zero.
- rne(v) = v rounded to the nearest integer, ties to even.

1. Rotary tables C and S, for t < T and i < dh / 2: b = (-2 i) / dh; w = pow(theta0, b); x = t * w;
   C(t, i) = rha(2^cs * cos(x)); S(t, i) = rha(2^cs * sin(x)).
2. Softmax table EX, for u = 0, 1, ..: EX[u] = rne(2^expS * exp(-u / 2^rx)). u0 is the least u with
   EX[u] = 0, and the table ends at u0 - 1.
3. SiLU table Ts, for keys c = -32,768 .. 32,767: c' = min(max(c, -64 * 2^eg), 64 * 2^eg - 1);
   z = c' / 2^eg; g = 1 + exp(-z); s = z / g; Ts[c] = clamp(rha(64 * s); smax).
4. Sigmoid table Sig, for the same keys: g = 1 + exp(-c / 256); q = 4096 / g; Sig[c] = 16 rha(q).
   Model reads it only at -16,384 .. 16,383.

Ts is what the float64 steps give, not the rounded real function: for large z, 1 + exp(-z) rounds to
1, which moves some entries up by one. At eg = 8 and smax = 4,095 that is 1,744 entries, each one
unit above the rounded real function. A library whose pow, cos and sin are not correctly rounded can
give other rotary entries where an entry lies close to a rounding tie. For theta0 in {10,000;
500,000}, dh in {2, 4, 8, 16, 32, 64, 128}, T <= 1,024 and cs = 12, every entry lies at least
9.32e-8 from a rounding tie, and a library must be off by at least 2,559 units in the last place to
change one (computed by `scales/tests/table_ties.py`, not proved); the [standard parameter
form](../verifier/docs/parameter-file.md#the-standard-parameter-form) admits only this set.
Elsewhere such a library can differ: at theta0 = 3,675,548,936, dh = 128, cs = 12 and T >= 690,
C(689, 1) is -548 with correctly rounded functions and -547 with a common library.

**Digests.** The parameter file states four digests, each as eight lines `NAME0` to `NAME7`. A
program must stop if its tables have other digests. To digest a list of n 32-bit words, write each
word as 4 little-endian bytes and hash them with unkeyed BLAKE2b (64-byte output), with one change:
in every call of the compression function the high half of the 128-bit byte counter, which BLAKE2b
leaves at 0, is 72 + 2^32 n (72 marks the hash of a public table; `verifier/HASHES.md`, "H(tag;
x)"). Read the first 32 bytes of the result as eight little-endian words. Standard libraries give no
access to the counter; `tables.py` writes the chain out.

- `tsilu`: Ts[c] + 4,096 for all keys in increasing order.
- `tsig`: Sig[c] for c = -16,384 .. 16,383, then clamp(r; 127) + 128 for r = -16,384 .. 16,383 (a
  proof's lookup table for gc of section 7). Without `gcl`: Sig[c] for all keys.
- `trope`: all of C, as C(t, i) + 4,096 at index t dh / 2 + i, then all of S likewise.
- `texp`: EX[0] .. EX[u0 - 1].

## 5 The norm

N(x; M, K, eps) maps a row x of d stream values to a row nr. It is a root-mean-square norm without
gains.

    Vx    = sum over k of x[k]^2
    sigma = isqrt(Vx + eps)
    Rc    = floor((2^M + floor(sigma / 2)) / sigma)
    w[k]  = rq(x[k] Rc, K)
    nr[k] = clamp(w[k]; 2,047)

Range conditions (section 9): 8 on x[k], 1 on Vx + eps, 2 on w[k]. Vx can exceed 64 bits.

sigma is the root of the sum of squares, not of their mean: the factor sqrt(d) is in the weights
(section 11.3) and the scales.

## 6 Attention

x_t (`x`) is the stream row that enters layer l at position t < T. The scales are layer l's.

1. **Norm.** a_t = N(x_t; M1, K1, eps1).
2. **Queries and keys.** Let A[j] = (a_t Wq)[j] for j < dq and A[dq + j] = (a_t Wk)[j] for j < dkv:
   n_h query heads, then n_g key heads. If rp = 1, let C = C(t, i) and S = S(t, i); if rp = 0, let
   C = 2^cs and S = 0. In each head, for i < dh / 2, let A0 and A1 be the head's entries 2 i and
   2 i + 1:

       p0 = C A0 - S A1        the new entry 2 i
       p1 = S A0 + C A1        the new entry 2 i + 1

   The wide value is `qw` = rq(p, kq) in a query head and `kw` = rq(p, kk) in a key head. Then
   q = clamp(qw; 2,047) and k = clamp(kw; 2,047). Range condition 3 is on qw and kw.
3. **Values.** The wide value is `vw`[j] = rq((a_t Wv)[j], kv), and v = clamp(vw; 2,047). Range
   condition 4 is on vw.
4. **Causal attention.** Query head h reads key-value head g = floor(h / (n_h / n_g)). At position
   i, with j <= i and c < dh:

       s_ij      = sum over c of q_i[h][c] k_j[g][c]
       m_i       = max over j of s_ij
       u_ij      = floor((m_i - s_ij) / 2^sh)
       e_ij      = EX[u_ij] if u_ij < u0, else 0
       Z_i       = sum over j of e_ij
       acc_i[c]  = sum over j of e_ij v_j[g][c]
       o_i[h][c] = floor((2 acc_i[c] + Z_i) / (2 Z_i))

   Z_i >= EX[0] > 0, because u = 0 at the largest score. This step has no range condition.
5. **Output projection and residual.** ad[k] = rq((o_t Wo)[k], ko), and x'_t = x_t + ad (`x_mid`).
   Neither o nor ad is clamped.

The scale 1 / sqrt(dh) is in Wq (section 11.3).

Absent from Model: attention within chunks of 8,192 tokens on the rotary layers, and the query
factor 1 + 0.1 log(1 + floor((t + 1) / 8,192)) on the others, which change nothing for T < 8,192
(`parameters.py` accepts T <= 1,024); and the normalization of queries and keys, which is off in
Maverick.

## 7 Feed-forward

For one position, this section maps x', the stream row after attention, to x'', the row that enters
layer l + 1. Let m = N(x'; M2, K2, eps2).

**A block of hidden units** reads a vector z of d entries through a gate matrix G and an up matrix
U, with shifts k_g and k_u. For hidden unit j:

    gw  = rq((z G)[j], k_g)                                       the wide gate key
    g0  = low8(gw),   gh = (gw - g0) / 256,   gc = clamp(gh; 127)
    key = 256 gc + g0
    s   = Ts[key]
    u   = rq((z U)[j], k_u)

Range conditions: 6 on gh, 7 on u. Where gc differs from gh, Ts is constant, so the clamp only
saturates.

**Dense layer.** One block of f units with (z, G, U, k_g, k_u) = (m, Wg, Wu, kg, ku). Then

    ad[k] = rq(sum over j < f of s[j] u[j] Wd[j][k], kd),   x'' = x' + ad

**Mixture-of-experts layer.**

1. Router. For e < NE:

       lw    = rq((m Wr)[e], kr)                the wide router logit
       ly    = floor(lw / 8),   l3 = lw mod 8
       lr[e] = 8 clamp(ly; 2,047) + l3

   Range condition 5 is on ly. So lr = lw for -16,376 <= lw <= 16,383; beyond, only l3 orders the
   values, so the map is not monotone at its two ends.
2. Choice. e* (`e_star`) is the argmax of lr: one routed expert, ties to the lowest index.
3. Scale. sc = Sig[lr[e*]] and xr[k] = rq(sc m[k], 16). The scale multiplies the routed expert's
   input, not its output.
4. Shared expert. A block of fe units with (z, G, U, k_g, k_u) = (m, Wg, Wu, kg, ku). It gives s[j]
   and u[j] for j < fe.
5. Routed expert. A block of fe units with (z, G, U, k_g, k_u) = (xr, Wgr_e*, Wur_e*, kgR, kuR). It
   gives s[fe + j] and u[fe + j] for j < fe.
6. Down projection and residual. The two sums are added before the one shift:

       ad[k] = rq(sum over j < fe of (s[j] u[j] Wd[j][k] + s[fe + j] u[fe + j] Wdr_e*[j][k]), kd)
       x''   = x' + ad

## 8 Embedding, head and output

The input is a sequence tok_0 .. tok_(T-1) of token ids below V.

    x^0_t   = Emb[tok_t]
    x^(l+1) = layer l applied to x^l (sections 6, 7)     for l < L
    aF_t    = N(x^L_t; MF, KF, epsF)                     at every t < T
    lam_t   = aF_t Wh                                    the V logits at position t
    Model(tok) = y = argmax of lam_(T-1)                 ties to the lowest token id

**Logit gap.** With tok_T = y,

    gap_t = max over v < V of (lam_t[v] - lam_t[tok_(t+1)] + [v < tok_(t+1)])

It is 0 exactly if tok_(t+1) is the argmax of lam_t. It is not a value of Model; a proof's statement
uses it.

**Length and padding.** Model takes exactly T tokens, a *window*. A request built from a shorter
text is padded to T, and the padding counts like any other tokens, also in section 9. No padding
token is fixed here; this repository's vLLM integration uses id 0. In a padded request y follows the
padding; which positions a proof covers is part of its statement ([protocol.md](protocol.md)).

**Prefixes.** Only step 4 of section 6 reads other positions, and only j <= i. So every value at
position t depends only on tok_0 .. tok_t. `model` in Model.lean takes n <= T tokens, returns the
argmax of lam_(n-1), and asks for the range conditions at these n positions; at n = T it is Model.

## 9 Range conditions

Model is a partial function. It is defined exactly at the inputs (its domain) at which conditions
1 to 8 hold, with the intervals of the column "defined", at all T positions, in all layers, at all 2
L + 1 norms.

| # | value | section | defined | accepted by the provers and model.py |
|---|---|---|---|---|
| 1 | Vx + eps | 5 | < 2^78 | the same |
| 2 | a norm's quotient w[k] | 5 | -32,768 .. 32,767 | -32,768 .. 32,639 in `model.py`; the GPU prover tests no interval here, and its test of M and K at start-up takes the place of one |
| 3 | a wide query or key entry, qw or kw | 6 | -32,768 .. 32,767 | -32,768 .. 32,639 |
| 4 | a wide value entry vw | 6 | -32,768 .. 32,767 | -32,768 .. 32,639 |
| 5 | the high part ly of a wide router logit | 7 | -32,768 .. 32,767 | -32,768 .. 32,639 |
| 6 | the high part gh of a wide gate key | 7 | -16,384 .. 16,383 | the same |
| 7 | an up projection u | 7 | -32,768 .. 32,767 | -32,768 .. 32,639 |
| 8 | x[k]: a stream value that a norm reads | 5 | \|x[k]\| < 2^39, which follows from condition 1 | \|x[k]\| <= 547,599,908,735 |

The last column is narrower because the provers write these values in limbs (signed bytes): two
limbs end at 32,639 and five at 547,599,908,735. Between the two columns Model is defined, but the
provers make no proof (the request gets the response `failed`) and `IntegerModel.extend` raises
`RangeConditionFailed`. Model.lean has the column "defined". Attention's rounded division (section
6, step 4) has a remainder that the provers write in three bytes; it fits, since a parameter file
with lt + expS + 1 > 24 is rejected (argued from the sizes, not proved). Before it commits,
`cp_commit` bounds, from the weights and scales alone, what any prompt can drive these values to;
why those bounds should hold is argued, not proved
([scales/worst-case-bounds.md](../scales/worst-case-bounds.md)).

At layer 0's first norm x is a row of Emb. Every number of four signed bytes meets condition 8, and
a whole row of them meets condition 1 if d <= 65,024. So the range conditions do not narrow Emb:
Model is defined on every embedding that can be committed, as far as that norm goes, and no
condition of the verifier stands against a proof there.

The verifier, `cp_verify`, refuses a parameter file unless every norm has 24 <= M <= 47,
17 <= K <= 32 and K >= M - 14. It admits K > M, as does `parameters.py`, but such a file is not in
the standard parameter form (the norm's output may be 0 for every input), and no call under it ends
with `YES`. With five limbs, the provers also reject K = M - 14 for K = 17 to 23, and so does
`parameters.py`. At these seven pairs a norm's quotient can leave the range that the provers accept
while every other range condition holds (the table above). Up to 32,767 the model is defined, so a
proof there would prove the model's own reply; no shipped prover makes one. Above 32,767 the model
is undefined, and no proof satisfies the relation.

## 10 Parameters that change the function

| line | otherwise | effect |
|---|---|---|
| `xl 5`: a stream value has five limbs | four limbs; condition 1 is Vx < 2^59, without eps, and implies condition 8 | same values, smaller domain |
| `gcl 1` | key = gw; condition 6 is on gw: -32,768 .. 32,767 (provers: .. 32,639) | same values, smaller domain |
| `rcl 1` | lr = lw; condition 5 is on lw: -16,384 .. 16,383 | smaller domain; same values, except that the clamped form gives lw + 8 for lw <= -16,377 |
| `eg`, `smax` | eg 7, 6, 5 or 4; smax 2,047, 1,023, 511 or 255 | another Ts, so another function |

## 11 From the model's files to the integers

This section defines how the matrices of section 3 are made from a model's GGUF files. No proof
checks this conversion: a proof is about the committed integers.
[quickstart.md](quickstart.md#4-llama-4-maverick-the-full-size-model), "4. Llama 4 Maverick: the
full-size model", says where its route downloads Maverick's public weight files from.

RN32 rounds to the nearest float32 value and RNE to the nearest integer, both with ties to even.
Every RN32 below is one float32 operation; nothing is fused.

### 11.1 The container

A model is one or more little-endian GGUF files of version 3. Each file has a header, metadata and a
tensor table, and each tensor is wholly in one file.

- The first file is the one that holds the model's metadata. They give the architecture's name
  (`general.architecture`) and, under that prefix, d (`embedding_length`), L (`block_count`) and dh
  (`attention.key_length`; 128 if absent). The programs that apply the conversion (`weight_sums/`,
  `cp_commit`, `cp_prove` and `block_weights.py`) do not test the architecture's name: they take any
  name as the prefix. Only the floating-point pass under `scales/` refuses a name other than
  `llama4`.
- A file's alignment is that file's `general.alignment`; 32 if absent. Its tensor data start at the
  next multiple of the alignment after the tensor table, and the table's offsets count from there.
- Element (i0, i1, i2) of a tensor has index i0 + dims[0] (i1 + dims[1] i2). For a weight matrix i0
  is the input index, i1 the output index and i2 the expert.

### 11.2 Stored blocks and their values

Seven storage types are read, and a committed tensor of any other type is rejected: F32, F16 and
BF16 (one little-endian float32, float16 or bfloat16 number per element), ggml's block type Q8_0,
which holds each run of 32 elements in one block, and its block types Q4_K, Q5_K and Q6_K, which
hold each run of 256 elements in one block. A block never crosses from one row (one value of i1 and
i2) into the next, so a row has a whole number of blocks. h and h_min are little-endian float16
numbers. A bfloat16 number is the upper 16 bits of a float32 number whose lower 16 bits are 0.

- **Q4_K**, 144 bytes. Bytes 0-1: h. 2-3: h_min. 4-15: twelve bytes b with eight pairs (sc, mn). For
  j < 4: sc = b[j] mod 64, mn = b[j+4] mod 64. For j >= 4: sc = (b[j+4] mod 16) + 16
  floor(b[j-4] / 64), mn = floor(b[j+4] / 16) + 16 floor(b[j] / 64). 16-143: 4-bit codes. Sub-block
  j (elements 32 j .. 32 j + 31) uses pair j and the 32 bytes from 16 + 32 floor(j / 2); element i's
  code c is in byte i of these, the low half for even j and the high half for odd j.
- **Q5_K**, 176 bytes. h, h_min and the pairs as in Q4_K. 16-47: qh. 48-175: the codes' low four
  bits, arranged like Q4_K's codes. Bit j of qh[i] is the fifth bit of element i of sub-block j.
- **Q6_K**, 210 bytes. 0-127: ql. 128-191: qh. 192-207: sixteen signed 8-bit scales sc, one per 16
  elements. 208-209: h. Element 128 a + 32 t + i (a < 2, t < 4, i < 32) has its 4 low bits in
  ql[64 a + 32 (t mod 2) + i], the low half for t < 2 and the high half otherwise, and its 2 high
  bits in bits 2 t and 2 t + 1 of qh[32 a + i]. The code c is this 6-bit number minus 32.
- **Q8_0**, 34 bytes. Bytes 0-1: h. 2-33: thirty-two signed 8-bit codes; element i's code c, from
  -128 to 127, is in byte 2 + i.

With h and h_min widened to float32, an element's decoded value is

    Q4_K, Q5_K:   v = RN32( RN32( RN32(h sc) c ) - RN32(h_min mn) )
    Q6_K:         v = RN32( RN32(h sc) c )
    Q8_0:         v = RN32(h c)
    F32:          v = the stored value
    F16, BF16:    v = the stored value, widened to float32

In the last three lines nothing is rounded. The product h c is exact in float32, because h has at
most 11 significant bits and c at most 8. Every float16 and every bfloat16 number is a float32
number, so the widening is exact: a subnormal float16 number becomes a normal float32 number, a
bfloat16 number keeps its bits, subnormal or not, and an infinity or a NaN stays one. A stored
float16, bfloat16 or float32 number, or an h or h_min, that is an infinity or a NaN gives a decoded
value that is not finite, which section 11.4 refuses.

`block_weights.py` has read no real model's file with tensors in 8 or 16 bits (Q8_0, F16, BF16). In
this repository one test runs the GPU code for them,
`integrations/vllm/integer_model/tests/test_against_the_prover.py`, on one small synthetic model; no
test here has a 4-bit file with one such tensor.

### 11.3 Which tensor is which matrix

| tensor name | matrix of section 3 | reads the gains of | constant c_0 |
|---|---|---|---|
| `blk.l.attn_q.weight` | Wq | `blk.l.attn_norm` | RN32(sqrt(d)/64/sqrt(dh)) |
| `blk.l.attn_k.weight`, `blk.l.attn_v.weight` | Wk, Wv | `blk.l.attn_norm` | RN32(sqrt(d)/64) |
| `blk.l.attn_output.weight` | Wo | none | |
| `blk.l.ffn_gate.weight`, `ffn_up` | Wg, Wu of a dense layer | `blk.l.ffn_norm` | RN32(sqrt(d)/64) |
| `blk.l.ffn_down.weight` | Wd of a dense layer | none | |
| `blk.l.ffn_gate_inp.weight` | Wr | `blk.l.ffn_norm` | RN32(sqrt(d)/64) |
| `blk.l.ffn_gate_shexp.weight`, `ffn_up_shexp` | the shared expert's Wg, Wu | `blk.l.ffn_norm` | RN32(sqrt(d)/64) |
| `blk.l.ffn_down_shexp.weight` | the shared expert's Wd | none | |
| `blk.l.ffn_gate_exps.weight`, `ffn_up_exps` | Wgr_e, Wur_e, e = i2 | `blk.l.ffn_norm` | RN32(sqrt(d)/64) |
| `blk.l.ffn_down_exps.weight` | Wdr_e, e = i2 | none | |
| `token_embd.weight` | Emb | none | |
| `output.weight` | Wh | `output_norm` | RN32(sqrt(d)/64) |

Here l is the layer's number, `ffn_up` is short for `blk.l.ffn_up.weight`, and a gain vector's name
ends in `.weight` too.

W[k][j] comes from the element (i0, i1) = (k, j), and Emb[tok][k] from the element (k, tok). The
outputs of Wq and Wk keep the file's order, so the rotary pairs of section 6 are neighbouring rows
of the file.

The constants are the float32 bit patterns 0x3F8F1BBD and, for Wq, 0x3DCA62C2. They are right for
d = 5,120 and dh = 128 only, and files with other values are rejected. A gain vector is an F32
tensor of d elements and is not committed. No weight is taken from a tensor with another name.

### 11.4 From a decoded value to an integer

**A matrix.** If it reads gains g with the constant c_0, input index k has the multiplier gamma_k =
RN32(g_k c_0), and an element's product is pi = RN32(gamma_k v). Otherwise pi = v. Let mx be the
largest |pi| in the whole tensor, all experts together. The tensor's exponent e is the largest
integer with RNE(2^e mx) <= 4,095; a tensor with mx = 0 is rejected (`scales/README.md` also uses
mx, for a recorded maximum). The element's integer is

    w = RNE(2^e pi)

where 2^e pi is exact, so |w| <= 4,095.

**The embedding** reads no gains, and its exponent e_res is an input of the conversion, chosen with
the scales: w = RNE(2^e_res v). A result of 2^30 or more in absolute value is rejected.

**Refusals.** Every program that applies the rule refuses an infinity or a NaN among the decoded
weights, the gains or their products. The exponent's range differs: `weight_sums/` takes -100 to
100; `cp_commit`, `cp_prove` and `block_weights.py` take 0 to 40, and at most 24 for a layer's
matrix other than the router if its type is Q4_K, Q5_K, Q8_0, F16 or BF16. `weight_sums/` refuses a
multiplier gamma_k that is not 0 but below 2^-100; `cp_commit`, `cp_prove` and `block_weights.py`
take one only if it is 0 or between 2^-64 and 2^64. The converter under `scales/` writes no
parameter file whose embedding needs more than 21 bits, although the rule allows any value below
2^30 in absolute value. `weight_sums/`, `cp_commit`, `cp_prove` and `block_weights.py` refuse a
tensor whose rows are not whole blocks: a multiple of 256 elements for Q4_K, Q5_K and Q6_K, and of
32 for Q8_0. `weight_sums/` and `cp_commit` read all seven storage types, but `cp_prove` takes a
routed expert's gate and up matrices only as Q4_K, Q8_0, F16 or BF16, takes its down matrix as one
of these or as Q5_K, needs rows of a multiple of 256 elements in every matrix, and needs a number of
rows that is a multiple of 4 in every matrix that it keeps as blocks (the routed experts' matrices,
and a layer's other matrices but the router if their type is Q4_K, Q5_K, Q8_0, F16 or BF16).
`block_weights.py` needs rows of a multiple of 256 elements in every matrix too. `cp_commit` tests
that row length first for Q8_0, F16 and BF16 but not for F32, and tests neither the routed matrices'
types nor the numbers of rows, so it can commit to files that `cp_prove` refuses at its start (by
reading, not run).

**Which checks see the whole integer.** A matrix's integers are kept in 16 bits and the embedding's
in 32. If the exponent in `wsums.txt` is wrong for the file, if the file has changed since the
exponents were chosen, or if e_res is too large, RNE(2^e pi) can lie far outside that width. At
least four checks test the whole integer, and refuse the weights if it is out of range (above 4,095
in absolute value; in the embedding 2^30 or more): the programs in `weight_sums/`, the checking pass
of `cp_commit` (exit code 3), `cp_prove` on every matrix as it loads the weights, and
`block_weights.py` as it loads them (it raises `WeightsRejected`). What comes behind them sees only
the kept bits: the SHA-256 sums in `wsums.txt`, and `cp_commit`'s range scan and worst-case check.

The programs in `weight_sums/` compute e by this rule and record it in `wsums.txt`, with a SHA-256
of each tensor's integers; `fold.txt` records e_res. `cp_commit`, `cp_prove` and `block_weights.py`
read the exponents there. Keep `wsums.txt` as secret as the weights: each of its hashes lets anyone
test a guess of a tensor's integers (`weight_sums/README.md`). The fit of the scales is defined in
[scales/README.md, "The fit of the scales"](../scales/README.md#fit), which calls the norms' output
sites n1, n2 and nF (the values a_t, m and aF of sections 6 to 8) and calls expS "S".

One difference of method is known. `cp_prove`, for every tensor and all seven storage types, and
`cp_commit`, for those keys of its range proof that it makes with kernels that decode stored blocks
(tensors of type Q4_K, Q5_K, Q8_0, F16 or BF16, and not all of these), multiply 2^e into the float32
multiplier before the one rounded product; `cp_commit`'s check of the weights, its range proof's
keys for all other weights and `block_weights.py` round the product first and then scale it exactly,
as above. The two give the same integers unless a float32 product overflows or is subnormal; they
are not proved equal on all inputs.

`cp_prove` does not compare the sums of `wsums.txt`, so nothing compares its integers with the
committed ones when it loads the weights. For Q8_0, F16 and BF16, whose decoding is written three
times in the CUDA sources, no test in this repository does either. The comparison comes with each
proof: above every column that is opened, `cp_prove` makes the commitment's stored tree node again
from the weights and the seed, and releases nothing unless the two agree (the end `hc-own-node`). So
it notices a run on other weights only where that changes a node above an opened column. By
reasoning, not by a test, a difference would show as a proof that is not made or is refused, not as
a false proof that is accepted.

## 12 The noise of a seeded score

This section is no part of Model, of a proof's statement or of `Model.lean`. No program of this
repository proves or verifies anything about it. `IntegerModel.extend(..., seed=...)` and two
example scripts use it to score a reply that a server sampled at temperature 1 under a seed that
the request names (`integrations/vllm/integer_model/OPERATING.md`, "Seeded replies").

A server can sample by Gumbel-max sampling: it adds noise to every logit and takes the largest sum.
Here the noise is a public function nz of the seed, the position and the token id, in the logits'
unit, 2^-s (the line `s` of [the parameter file](../verifier/docs/parameter-file.md), at most 40
here):

    z_t[v]      = lam_t[v] + nz_t[v]    for t < T - 1;     z_(T-1) = lam_(T-1)
    the seeded choice at t = argmax of z_t                 ties to the lowest token id
    the seeded gap_t = max over v < V of (z_t[v] - z_t[tok_(t+1)] + [v < tok_(t+1)])

t is the position whose logits choose token t + 1. Every position but the window's last has noise,
the prompt's too. The last, T - 1, has none. No token of the window is scored there: token t + 1 is
scored at position t, so the window's last token is scored at T - 2. What T - 1 chooses is y,
Model's output (section 8), which so stays as defined there. The seed is an integer of 64 bits; a
negative one is taken modulo 2^64.

**The rule.** With temp = 256, the temperature 1 in units of 1/256:

1. K is the first word of Philox4x32-10 with the key (the seed's low 32 bits, its high 32 bits) and
   the counter (t, 0, 0, 0). W is the first word with the key (K, 0) and the counter (v, 0, 0, 0).
2. W selects one of 65,536 slots. Let x = W, or W with all 32 bits inverted if W >= 2^31; side = 1
   if x >= 2^30, else 0; k = 2^31 - 1 - x if side = 1, else x; e the bit length of k; m the ten bits
   below k's leading bit, missing bits being 0. The slot is 2^15 side + 2^10 e + m.
3. G[slot] = floor((g(a) + g(b)) / 2), where a and b are the smallest and the largest x of the slot.
   A slot that no x reaches has G = 0.
4. nz_t[v] = floor(temp G[slot] / 2^(56 - s)). It is never negative. At s = 15 it is at most
   817,669.

**The function g.** g(x) is -ln(-ln(1 - u)) + 5 ln 2 in units of 2^-48, for u about x / 2^31, in
these integer steps:

    r24(v)  = v rounded to 24 significant bits, ties to even
    u       = max(r24(r24(x) (2^24 - 1)), 2^24 - 1)                     u in units of 2^-55
    lb(y)   = n 2^48 + f, the binary logarithm of y >= 1 in units of 2^-48, where n + 1 is the bit
              length of y, and f collects 48 bits: start with w = y 2^(63 - n); 48 times, let
              q = w^2 and b = floor(q / 2^127), append the bit b to f, and let
              w = floor(q / 2^(63 + b))
    ln2(a)  = floor(195,103,586,505,167 a / 2^48)                       a ln 2
    g(x)    = ln2(60 2^48 - lb(u))                                      if u < 2^39
    g(x)    = ln2(53 2^48 - lb(ln2(55 2^48 - lb(2^55 - u))))            otherwise

Below 2^39, that is u < 2^-16, -ln(1 - u) is taken as u. The constant 5 ln 2 keeps g positive; it is
the same for every token, so it cancels in a difference of two noise values, up to one unit of 2^-s.
The SHA-256 of the 65,536 entries of G, as little-endian 64-bit words, is
090e3da3a772ee5a5427e0fc4c78d2179e98d3e8bfc40d255245fb9ab2340e85; `noise.py` computes G and stops if
the digest differs.

**What the rule follows.** Scoring a sampled token against the logits plus the seed's noise, with
the temperature multiplying the noise, is Token-DiFR's (Karvonen et al., arXiv:2511.20621, sections
3.2 and 4.2). Philox4x32-10 is defined in J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
"Parallel Random Numbers: As Easy as 1, 2, 3", SC '11 (2011), section 4.3. Steps 1 and 3, and the x
of step 2, aim to match the noise that the V2 sampler of vLLM 0.29.0 generates for a request's seed,
so that an unmodified server can serve what is scored; their sources are
vllm/v1/worker/gpu/sample/gumbel.py (Apache-2.0) and triton/language/random.py of Triton 3.7.1
(MIT). There u is a float32 and the noise is -log(-log1p(-u)) in float32; here u is the same number,
computed in integers, and the two logarithms are one lookup. At temperature 1 the table is within
0.002 of that float32 noise. How far the match has been tried:
`integrations/vllm/integer_model/OPERATING.md`, "What has been run".

## Sources

[PUBLISHED-WORKS.md](../PUBLISHED-WORKS.md#works-that-the-files-cite), "Works that the files cite",
lists public work that the files of this repository name. The Llama 4 specifics follow Hugging Face
transformers 5.17.0 (its src/transformers/models/llama4/modeling_llama4.py and
configuration_llama4.py). The GGUF container, the block formats and their decoding are ggml's
(github.com/ggml-org/llama.cpp, MIT licence; notice in `THIRD_PARTY.md`), as in its Python package
gguf 0.19.0. The choice of the public quantized weight files, and the rule from which the conversion
grew, come from J. Petrie's VerInf (github.com/JamesPetrie/VerInf, commit 89903c96 of 28 August
2026; MIT License, Copyright (c) 2026 James Petrie).
