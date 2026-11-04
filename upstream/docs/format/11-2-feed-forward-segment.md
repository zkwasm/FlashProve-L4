*Part of [the file format](FORMAT.md), [section 11](11-sub-proofs.md). Rule numbers and terms are
those of that page.*

### 11.2 A feed-forward segment

This part is written from the programs' source. It lists, in order, every round of the sub-proof of
a feed-forward segment, with what the prover sends and what is sampled. The kinds of step are those
of 11.1, and the frame's rounds of a segment are R14.4.

*What the segment is about.* The model has
routed experts if the line `NE` is above 0. The model's L layers are cut into groups of Lam = 2^ll
neighbouring layers. Group g has one feed-forward segment. Its word in the item `links' identifiers`
(R14.3; in this name a link is a segment) is 1 + g. The segment is about everything in the group's
layers except the attention function (11.3). That is: the two norms of each layer; the projections
to queries, keys and values, with the rotary tables; the projection of attention's output; the
router; the gated feed-forward block, with its shared and routed experts; and the residual stream
from the group's input to its output. All of it is integer arithmetic, with rounding shifts that the
scale file gives for each layer. With routed experts, the odd layers of the model are the layers
that have them. The segment of group 0 is also about the statement that the first residual stream is
the embedding of the request's tokens. The segment's claims on the weights are merged into one row
on the stored table (11.1, the merge; 11.5).

*Sizes.* From the parameter file: lt, L, ll, d, H, G, dh, f, fe, NE, la, the line `xl` (XL = 5 limbs
of a residual value if `xl 5` is present, else 4) and lkW (11.1). This part also writes these lines
of the parameter file (8.6) by their names alone: nmin; tau (11.1); the matrices' sizes lmaf, NAf,
lmad, NAd, lmar, lmb, lmq, lmx; and the seven digit forms hq, hv, ho, hgD, huD, hdD, hr (second
half, a rounding site). Derived: T = 2^lt; Lam = 2^ll; LamM = Lam / 2 if the model has routed
experts, else 0 in every count below; NI = 2 Lam, the number of norms of the group; lNI = ll + 1;
lPd, lPf and lfe, the least numbers with 2^lPd >= d, 2^lPf >= f, 2^lfe >= fe; ls = la - lt. A norm
has NS = 2 XL - 1 sums of limb products, NPr row products, NO shown row values and a row of NR
digits: NPr = 7, NO = 4 and NR = 99 with five limbs; NPr = 3, NO = 3 and NR = 74 with four. E0 is 1
in the segment of group 0 and 0 in every other.

*The arrays that the segment commits to.* The names are those of the item `layout` (R14.5). The
segment's matrices of cells have a fixed list, and matrix number mu of the list is committed in
packed form (11.1) as the array `M` followed by mu.

| mu | matrix | cells | cell rows |
|---|---|---|---|
| 0 .. NAf - 1 | af, in NAf parts | most intermediate values of the layers | 2^lmaf rows each |
| then NAd | ad, in NAd parts | the norms' clamped outputs, the routed experts' input | 2^lmad rows each |
| then 1 | ar | the router's values and the norms' rows of digits | 2^lmar rows |
| then 1 | b | the planes of the lookup's multiplicities | 2^lmb rows |
| then 1 | qa | queries, keys, values, attention's output | 2^lmq rows |
| then 1 | xbO | the residual stream that leaves the group | 2^lmx rows |
| last | xbI | the residual stream that enters the group | 2^lmx rows |

*The columns.* A matrix in P parts of 2^lm rows that holds n cells for each token has ceil(n / (P
2^(lm - lt))) columns in each part. n is: for af, 6 Lam f + (hgD + huD) Lam f + XL (2 Lam - 1) d +
(8 + ho + hdD) Lam d + (2 + hq) Lam (H + G) dh + (2 + hv) Lam G dh, and 2 Lam f more with `gcl 1`;
for ad, 4 Lam d + 2 LamM d; for ar, 2 Lam NR + (3 + hr) LamM NE + 3 LamM, and 3 LamM NE more with
`rcl 1`; for qa, 4 Lam (H + G) dh; for xbO and xbI, XL d. b has NC 2^(kT + 5 - lmb) columns (11.1).
In 11.3, m_qa = 2^lmq is the number of cell rows of qa and k_qa its number of columns.

| moment | arrays, in the order of the layout's numbering |
|---|---|
| 0 | the matrices that no other segment reads, by increasing mu: af, ad, ar, b, and in group 0 xbI; then those that another segment reads: qa (read by the group's attention segment, 11.3), xbO (read by the next feed-forward segment, or by the output head) |
| 1 | `U3`: one vector over E (11.1) |
| 2 | `ZZ`, `BZ`: the smallness test's digits and the planes of its lookup |

From group 1 on, xbI is not committed by this segment: it is the array of xbO of the segment of
group g - 1, which the frame keeps up to this segment. The mask row and the frame's own messages of
moment 0 are R14.4.

*The rounds, in order.* A row in italics is the frame's.

| no. | step | rounds | scalars sent | sampled |
|---|---|---|---|---|
| | *moment 0* | *`SEND seed`; then `SEND answer` in the chain's first segment, `LOC` in a later one (R14.4)* | | |
| 1 | the lookup of the segment's call list | 11.1, steps 1 to 9 | 187 + (v'^2 + 4 v') + (d'^2 + 14 d') + (1 + 2 d' + 5) + (2 (5 + d') + 4), where v' and d' are the lookup's v and dL; the second and third terms are the levels of the key side and of the table side | 11.1 |
| 2 | the norms' token point | `ZERO` | 0 | lt elements |
| 3 | group 0 only: the embedding's point, in two rounds | `ZERO`, `ZERO` | 0 | ls + lt elements; then lPd - ls elements |
| 4 | the sums of limb products, hidden values sent alone; their batching, n = NS | `BAT-sigma` | NS NI | 1 element |
| 5 | a point on the norms' numbers | `JOIN` | 0 | lNI elements |
| 6 | the sums of squares: a product level with lNI + lPd + lt variables, default order | `PSC epsilon`, `PSC round` .., `PSC Y`, `PSC values`, `LINE` | 2 (lNI + lPd + lt) + 5 | as for a product level |
| 7 | the row products, hidden values sent alone; their batching, n = NPr NI | `BAT-sigma` | NPr NI | 1 element |
| 8 | the first token sumcheck, at the point of step 2: d_Y = 2, NO NI end values | `PSC epsilon`, lt times `PSC round`, `PSC Y`, `PSC values` | 1 + (1 + 2 lt + 2 + NO NI) | eps, lt challenges, beta |
| 9 | the second token sumcheck, at the last lt coordinates of the lookup's key point: d_Y = 1, 2 Lam + LamM (1 + NE) end values | as step 8 | 1 + (1 + 2 lt + 1 + 2 Lam + LamM (1 + NE)) | eps, lt challenges, beta |
| 10 | a batching, n = 2 | `BAT-sigma` | 0 | 1 element |
| 11 | the batched inner products: a sumcheck with lamS plain rounds of degree 2, d_Y = 2, 3 + LamM + NCL - E0 end values | `PSC epsilon`, lamS times `PSC round`, `PSC Y`, `PSC values` | 1 + 2 lamS + 2 + (3 + LamM + NCL - E0) | eps, lamS challenges, beta |
| 12 | the gated products: a product level with ll + lPf + lt variables, range order | as step 6 | 2 (ll + lPf + lt) + 5 | as for a product level |
| 13 | with routed experts, and only if ll > 1: a point on the numbers of the layers that have them | `JOIN` | 0 | ll - 1 elements |
| 14 | with routed experts: their gated products: a product level with ll - 1 + lfe + lt variables, range order | as step 6 | 2 (ll - 1 + lfe + lt) + 5 | as for a product level |
| 15 | group 0 only: the embedding's value, a hidden value sent alone | none of its own: it is in the message of step 16's `BAT-sigma` | 1 | |
| 16 | the merge, n = NCL | `BAT-sigma`, `PSC epsilon`, lkW times `PSC round`, `PSC Y`, `PSC values` | 2 lkW + 3 | 1 element; eps, lkW challenges, beta |
| | *moment 1: `U3`* | *`SEND seed`, `SEND answer` (R14.4 (d))* | | |
| 17 | the smallness test's matrix | `RNG` | 0 | (the number of columns of all matrices of the list, xbI among them in every group) x tau entries |
| | *moment 2: `ZZ`, `BZ`* | *`SEND seed`, `SEND answer`* | | |
| 18 | the smallness test's lookup on the digits | 11.1, steps 1 to 9, with NC = 2 and (v'', d'') = (v_Z, 11) of 11.1 | as for a lookup | as for a lookup |
| 19 | the batching of all claims and equations, n = NCm + Q | `BAT-sigma` | 0 | 1 element chi |
| | *the segment is closed* | *`BAT`, `FOLD epsilon`, `FOLD round` .. (R14.4)* | | |

Steps 9 and 10 are always run. The condition for them is that some call of the list has one of the
calls' segments (the call list, below) N1, N2, QK, or with routed experts G, U, XR, D, DD, MR, and
every admissible call list has N1, N2 and QK.

*In which message the scalars are.* The NS NI scalars of step 4 are the prover's message of that
step's `BAT-sigma`. The NPr NI scalars of step 7 are the message of that step's `BAT-sigma`. The
claimed sum of steps 8 and 9 is in the message of the step's `PSC epsilon`, before the hiding
polynomial's sum. The rounds of steps 2, 3, 5, 10, 13, 17 and 19 have an empty prover's message, and
so has the `BAT-sigma` of step 16 outside group 0.

*The end values, in the order sent.* Step 8: for each norm, by number, its NO row values. Step 9:
one value for each of the 2 Lam norms; then one for each layer with routed experts; then, layer by
layer, the NE entries of the selector, which marks the one routed expert of a token (second half,
line 17). Step 11: one value A0; one value for the gated products; one for each layer with routed
experts; one value B0; then the values of the weight claims in the order of the list below, without
the embedding's.

*Two derived numbers.* lamS is the least number with 2^lamS >= S. Write P(x) for x rounded up to a
power of two. Then S = Lam (4 P(d) + P(H dh) + P(f) + 2), and with routed experts S has in addition
LamM (P(d) + 1 + 2 NE) + LamM NE (2 P(d) + fe). NCL = 6 Lam + 4 LamM + E0 is the number of weight
claims.

*The weight claims, in order.* The calls are gone through twice, each time in the order of the list,
and a call's segments in the order of its line. First pass: each of the calls' segments G, U, V, O,
D gives one claim for each layer of the group, by layer, and LR gives one for each layer with routed
experts. Second pass: QK gives one claim for each layer, and with routed experts each of G, U, D
gives one for each layer that has them. In group 0 the embedding's claim is the last.

*The claims, in order.* Claim number c gets chi^c (11.1, with its exception). NCm is the sum of the
counts. The second half calls row n of this table "group n".

| no. | claims | count |
|---|---|---|
| 1 | of the lookup of step 1, on mask entries alone: the tie, the key levels, the table levels, the bits | 31 + v' + d' + 1 |
| 2 | of step 6: the level's own; its two end values are equal; what the first is | 3 |
| 3 | the claimed sum of step 8 is the batch of the row products; step 8's own | 2 |
| 4 | the end values of step 8 are what `U3` gives, by norm and within a norm in the order sent | NO NI |
| 5 | the carries of the sums of limb products, by norm | NS NI |
| 6 | the lines of each norm's row of digits, by norm: 8 with five limbs, 5 with four | 8 NI or 5 NI |
| 7 | the end values of step 9 are what `U3` gives, in the order sent | 2 Lam + LamM (1 + NE) |
| 8 | of step 12: the level's own; what its two end values are | 3 |
| 9 | with routed experts, of step 14: the same | 3 |
| 10 | with routed experts: the selector's entries sum to 1, by layer | LamM |
| 11 | of the lookup of step 1, on its planes: the leaves; the planes at the bits' point; the number of keys | 3 |
| 12 | group 0 only: what the embedding's value is | 1 |
| 13 | of the smallness test, in the order of 11.1, with (v'', d'') = (v_Z, 11) of 11.1, 36 + v'' + d'' + NM + NX in all: 31 + v'' + d'' + 1 of its lookup on mask entries alone, that lookup's join and its three on the planes, one for each matrix, one for each read | 36 + v'' + d'' + NM + NX |
| 14 | step 11's own; what A0 is; what B0 is | 3 |
| 15 | the merge's own | 1 |

The lookup of step 1 has no claim of its own for its join, and step 9 has none of its own either:
both are inside the start form of step 11. NM is the number of matrices of the list. NX is the
number of reads (11.1): the sum, over the parts of `U3`, of the number of the matrix's cell rows
divided by the length of the part's weight vector. Q, how many numbers the smallness test's
equations take, is the sum over the matrices of tauP times (their packed rows rounded up to a power
of two), plus NX tauP (11.1).

*The parts of `U3`, in order.* A part of `U3` is defined in 11.1. A **long** part has a weight
vector of 2^la entries and a **short** one of T entries. In order: af at five points (four without
routed experts); qa at one; xbI at three, and in group 0 at a fourth, the embedding's; xbO at one:
these are long. Then ad at two points; ar at four; qa at one: these are short. Last, three parts on
b, with weight vectors of 2^lmb entries.

*What the verifier takes from the public files.* From the scale file, for each layer, 32-bit words
in this order: the shifts kq, kk, kv, ko, kg, ku, kd; M1, K1, M2, K2 of the two norms; sh, which
attention uses; rp, which is 1 if the layer has rotary tables. After all layers come, for each layer
with routed experts, kr, kgR, kuR. From the parameter file: the shape; eps1 and eps2; the digit
forms hq, hv, ho, hgD, huD, hdD, hr; the matrices' sizes; the call lines; the lines `iota_0` to
`iota_7`, each of which must equal n_c - dT of its call, or 0 if the list has no call of that
number; and the options `xl`, `rcl`, `gcl`, `eg`, `smax`, `lidb`. A verifier rejects the files,
before it reads a proof, unless the tests of section 13 on these numbers hold, the ranges and the
no-overflow conditions among them.

*The call list.* A line `call NAME KIND SEGMENT ..` names a call, its table's kind and the call's
segments. **A call's segment** is a named group of the call's key columns. It is not a segment of a
request, and this section always writes it in full. The kinds are R8 and R16 (ranges of 8 and 16
bits), FC (the clamp), FL (the gate function) and FS (the sigmoid).

*The calls' segments.* h stands for a digit form: the call's segment is absent if h = 0, of kind R8
if h = 1 and of kind R16 if h = 2. A call's segment marked R has keys only in the layers with routed
experts; in the other layers of the group its columns are padding.

| calls' segments | kind | length | present |
|---|---|---|---|
| G, U, UT | R16 | f | always |
| QK | R16 | (H + G) dh | always |
| QKH | by hq | (H + G) dh | hq > 0 |
| N1, N1H, N2, N2H, O, D | R16 | d | always |
| V | R16 | G dh | always |
| XR (R) | R16 | d | with routed experts |
| LR, LT, DD (R) | R16 | NE | with routed experts |
| GH; UH | by hgD; by huD | f | form > 0 |
| OH; DH | by ho; by hdD | d | form > 0 |
| VH | by hv | G dh | hv > 0 |
| ROW | R8 | 2 NR | always |
| LH (R) | by hr | NE | with routed experts, hr > 0 |
| SEL (R) | R8 | NE | with routed experts |
| L3 (R) | R8 | NE | with `rcl 1` |
| G0 | R8 | f | with `gcl 1` |
| CN1, CN2 | FC | d | always |
| CQK | FC | (H + G) dh | always |
| CV | FC | G dh | always |
| CLR (R) | FC | NE | with `rcl 1` |
| MR (R) | FS | 1 | with routed experts |
| CG | FS | f | with `gcl 1` |
| F | FL | f | always |

A call of kind R8 has k_c = 8; every other call has k_c = 16. The lines `rcl 1` and `gcl 1` are
admitted only with routed experts.

*The tests on the list, and the key bits.* A verifier rejects (`policy-call-list`) a list with more
than 8 calls, with a line without a call's segment, with two calls of one name, or with a call named
V, a, b or E. (In `cp_verify` a line of the file without a call's segment never reaches this test:
it has fewer than four tokens, and the tests on a line's form reject it first, under
`policy-call-line`.) Under the same name it rejects a list in which some call's segment that the
parameters call for is missing, is listed twice, or is in a call of another kind. The same holds for
a list that names a call's segment which the parameters do not call for, or a name that is not a
call's segment at all. The tests on the form of a call line come before these, under the name
`policy-call-line` (section 13): among them, a call's name has at most 8 characters. Call c has n_c
= ll + lJ + lt key bits. The length of each of the call's segments is cut into powers of two. The
pieces of all the call's segments are sorted by decreasing size, and pieces of equal size keep their
order. They take columns one after the other. lJ is the least number such that 2^lJ columns are
enough for them, and it is at least nmin - ll - lt.

*The table FS with the line `gcl 1`.* Row t < 32768 is t + 2^17 x 16 round(4096 sigmoid((t - 16384)
/ 256)). Row t >= 32768 is t + 2^28 (clamp(t - 49152, -127 .. 127) + 128) + 2^40.

*A worked example.* With lt = 10, ll = 2, d = 5120, H = 40, G = 8, dh = 128, f = 16384, fe = 8192,
NE = 128, la = 16, nmin = 20, `lidb 28`, `xl 5`, `rcl 1`, `gcl 1`, lkW = 18, hq = 2 and the other
six digit forms 1, and the eight calls (G U), (N1 N1H N2 N2H O D V), (QK QKH XR LR LT DD), (GH UH OH
DH VH ROW LH SEL L3 G0), (CN1 CN2 CQK CV CLR), (MR CG), (F), (UT): the calls have 27, 27, 27, 28,
27, 27, 26, 26 key bits, so dT = 26, NT = 16, v' = 30, d' = 19. With tau = 128, lmaf = 16, NAf = 3,
lmad = 15, NAd = 1, lmar = 13 and lmb = lmq = lmx = 16: mz = 499,712, so v'' = 27. The scalars are:
step 1: 1,930; step 4: 72; step 6: 57; step 7: 56; step 8: 56; step 9: 289; step 11: 86, with lamS =
23 and NCL - E0 = 32; step 12: 57; step 14: 53; step 16: 39; step 18: 1,363. These are the values of
`target_full_f2/policy.txt`, the full-size example configuration (section 8).

*Easy to get wrong.*

1. The merge (step 16) comes before the root of `U3` (moment 1). In the output head it comes after
   the smallness test (11.4).
2. The start form of step 11 names entries of `U3`, which is committed later. Only its constant and
   the scalars that were sent before step 11 enter the running claim (11.1).
3. The product level of step 6 uses the default order; those of steps 12 and 14 use the range order.
   The first round of step 11 binds the highest bit.
4. In step 6 the lPd coordinates of the feature bits are the constant 1/2: the rounds are equality
   rounds like the others.
5. Step 10 batches two rows: row 0 gets the coefficient 1, row 1 the sampled element. Step 13 is
   absent if ll = 1.
6. The segment of group 0 has four things more: it commits to xbI itself, and it has step 3, step 15
   and one more part of `U3`. Its NCL, its NCm and its number of scalars are larger.
7. Two tests on content are not claims, and every verifier makes both. No verifier accepts a proof
   in which eps of step 9 is 0 (the programs reject). And none accepts if, after the last claim, the
   coefficient under a scalar's number other than that of fv' is 0 in the sum of all claims (the
   programs end with `NOT CHECKED`). Both are in the second half, "Two tests beside the claims", and
   in 13.9, events 1 and 9.

*What this part does not specify.* This first half gives the rounds, the counts and the orders; the
second half gives the integer computation, what each cell, each scalar and each entry of `U3` holds,
the key of every call's segment, every claim as an equation, the weight claims, and the tests on the
public files with the no-overflow conditions. The two halves do not define which limbs a prover
writes for a value that can be written in limbs in more than one way, or what is in the features
that belong to no tensor: a verifier needs neither. They give no argument that the claims imply the
integer computation, that the indices beyond the true width do no harm, or that an equation in the
field holds in the integers.

#### 11.2, second half: cells, keys and claims

This part is written from the programs' source. The first half of 11.2 gives the rounds of a
feed-forward segment and counts its scalars and claims. This half says what each committed cell and
each scalar is, how the keys of the lookup are formed, and gives every claim as an equation. A reader needs the first half of 11.2, and from 11.1: a
hidden value and its shown value, a form, the claim that a sumcheck leaves, the lookup with its own
claims, the smallness test with the entries of `U3`, and the order of bits. How a weight claim
reaches the stored table is in 11.5.

*What is public and what is hidden.* Public are the three public files, the request's tokens, every
sampled value and every scalar as it was sent. From these a verifier computes every coefficient and
every constant below. Hidden are the cells of the committed matrices, the entries of `U3`, the mask
row's entries and the weights. A verifier is not sent a hidden value itself (11.1). A claim is a
form in hidden values, and the verifier computes the form's coefficients and its constant (11.1).

##### Notation

All numbers of the first half keep their letters: lt, T, L, ll, Lam, LamM, NI, lNI, d, H, G, dh, f,
fe, NE, la, ls, lPd, lPf, lfe, XL, NS, NPr, NO, NR, E0, NCL, lamS, lkW; the shifts and exponents of
the scale file kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, kr, kgR, kuR; the digit forms hq, hv, ho,
hgD, huD, hdD, hr; and of the lookup of step 1: NC, n_c, k_c, lJ, dT, kT, cK, NT, rK, zx, zy. "Step
n" is a row of the first half's table of rounds, and "group n" a row of its table of claims. A group
of layers is always named by its number g. The name of a call's segment is written in capitals, as
in the first half. G, F, V, D, U and O are such names where the text speaks of calls' segments or of
panels, and in a table's column of them. Elsewhere G is the number of key heads, F is the field, `V`
and `D` are lines of the parameter file, and D_j is a digit.

| letter | meaning |
|---|---|
| g; lam; l | the group's number; a layer's number inside the group, 0 <= lam < Lam; the layer's number in the model, l = g Lam + lam. The scale file's words are those of layer l |
| lamM | for an odd lam, in a model with routed experts: lamM = (lam - 1) / 2, the layer's number among the group's LamM layers with routed experts |
| t | a token's position, 0 <= t < T |
| i | a norm's number in the group, 0 <= i < NI. Norm i = 2 lam + n is the first (n = 0) or the second (n = 1) norm of layer lam. M(i), K(i), eps(i) are M1, K1, `eps1` of layer l for n = 0 and M2, K2, `eps2` for n = 1 |
| sg | a stream's number, 0 <= sg <= 2 Lam |
| j; k2; x; e | a feature of a stream; the index that an inner product runs over; an index inside a call's segment; an expert's number, e < NE |
| lb | a limb's number |
| dq, dkv, nqk, ldh | dq = H dh, which must equal d; dkv = G dh; nqk = dq + dkv; 2^ldh = dh |
| Pd, Pf, le | 2^lPd; 2^lPf; clog2(NE) |
| lift(k) | for a shift k: 16 - k if k < 16, else 0 |
| rsh(A, k) | floor((A + 2^(k - 1)) / 2^k), the rounding shift of the integer A by k bits |
| clamp(y, B) | y if -B <= y <= B; -B if y is below; B if y is above |
| eq(z, n) | the equality weight of the index n at the point z, the coordinate of the highest bit first (11.1). At a point without coordinates it is 1 |
| a name with a prime, and without | for a scalar that has a name here and the number n: the name alone, as Cl_8, is the form hidden(n) of 11.1; the name with a prime, as Cl'_8, is the public element shown(n), the scalar as it was sent. The prime follows the name's letters and comes before an index. v' and d' without an index are the first half's numbers of levels. The primes of e', x' and lb' mark a second index, not a shown value |
| U3[n] | entry n of the vector `U3`, a variable of the forms (11.1) |
| C_n; Left(C_n; S0; 0) | the start form of the sumcheck of step n; the form that this sumcheck leaves (11.1), made with the sumcheck's own numbers, sampled values and last running claim. S0 is a public element: the summand at the sampled values, computed from the end values as they were sent; at step 9 it is the constant 0 (the start form of step 11, below). The kept form is 0 in every sumcheck of this part |
| alpha, theta, e_c, C_end, tab_c, t0_c, padkey_c, tree0(c), npad(y), eqlow | of the lookup of step 1, as in 11.1: the point of comparison; the shift; the weight of call c's trees at zy; the form of the key side's leaves at (zx, zy); the rows' values of call c's table; its padding row; tab_c(t0_c) + c theta; the call's first tree; the number of padding positions in tree y. eqlow is not the lookup's: it is 11.1's equality weight on a range of bits, taken here on the challenges of step 11 (below, after the table of the points) |
| a | the number of a weight claim, as in 11.5. Under the heading on a norm's row and in groups 3 and 4, a and b are two numbers of the row. In sw_SEG(a, b), SL(X; a, b; x0) and Qg(a, b), a and b are the bounds of a range of indices |
| Wm_X(..) | a committed weight of the unit of kind X of layer l (11.5), with its indices in the order of the unit's index groups |

A sum over an empty range is 0. The coefficients and constants of a form are elements of E. An
integer constant, also a negative one, stands for its image in F.

##### What the segment computes

This is the integer computation whose values the prover commits to. Each numbered line is a relation
between committed values and committed weights, and the keys and claims below are written for these
relations. The names in code marks are the tensors of the next heading.

*The residual stream.* The group has 2 Lam + 1 streams str_sg, each with d integers for each token.
str_0 enters the group. str_(2 lam) enters layer lam, str_(2 lam + 1) is the stream after the
layer's attention block and str_(2 lam + 2) the stream after its feed-forward block. str_(2 Lam)
leaves the group. If g = 0, str_0(j, t) = Wm_12(tok_t, j): the embedding of tok_t, the request's
token at position t.

*A norm.* Norm i reads str_i.

1. Vsq_i(t) = the sum over j < d of str_i(j, t)^2.
2. sig_i(t) = floor(sqrt(Vsq_i(t) + eps(i))).
3. R_i(t) = floor((2^M(i) + floor(sig_i(t) / 2)) / sig_i(t)).
4. The wide output is rsh(str_i(j, t) R_i(t), K(i)). It is `aw`(lam, j, t) for n = 0 and `mw`(lam,
   j, t) for n = 1.
5. The output is `an` = clamp(`aw`, 2047) for n = 0 and `mn` = clamp(`mw`, 2047) for n = 1.

*Queries, keys and values.* hd is a head's number: the H query heads come first, then the G key
heads. ch < dh is a channel, and x = hd dh + ch.

6. pr(hd, ch, t) = the sum over k2 < d of Wm_0(hd, ch, k2) `an`(lam, k2, t).
7. Let Co and Si be the cosine and the sine of the rotary tables (11.1) of layer l for position t
   and pair pa < dh / 2. Then Aqk(hd, 2 pa, t) = Co pr(hd, 2 pa, t) - Si pr(hd, 2 pa + 1, t), and
   Aqk(hd, 2 pa + 1, t) = Si pr(hd, 2 pa, t) + Co pr(hd, 2 pa + 1, t).
8. `qkw`(lam, x, t) = rsh(Aqk(hd, ch, t), kq) for x < dq, and rsh(Aqk(hd, ch, t), kk) for dq <= x <
   nqk.
9. `q`(lam, x, t) = clamp(`qkw`(lam, x, t), 2047) for x < dq, and `kx`(lam, x - dq, t) =
   clamp(`qkw`(lam, x, t), 2047) for dq <= x < nqk.
10. `vw`(lam, x, t) = rsh(the sum over k2 < d of Wm_1(k2, x) `an`(lam, k2, t), kv) for x < dkv, and
    `v` = clamp(`vw`, 2047).

*Attention's output.* `o`(lam, k2, t), k2 < dq, is what the attention segment is about (11.3).

11. str_(2 lam + 1)(j, t) = str_(2 lam)(j, t) + rsh(the sum over k2 < dq of Wm_2(k2, j) `o`(lam, k2,
    t), ko).

*The feed-forward block of a layer without routed experts.* x < f.

12. gw(lam, x, t) = rsh(the sum over k2 < d of Wm_3(k2, x) `mn`(lam, k2, t), kg), and `up`(lam, x,
    t) = rsh(the sum over k2 < d of Wm_4(k2, x) `mn`(lam, k2, t), ku), with -2^15 <= `up` < 2^15.
13. `sl`(lam, x, t) is the value of the gate function's table (11.1) at gw. Without `gcl 1` the
    cells hold `gi` = gw. With `gcl 1` they hold gw = `g0` + 256 `gh` with -128 <= `g0` <= 127, and
    `gc` = clamp(`gh`, 127), and the table is read at `g0` + 256 `gc`.
14. hh(lam, x, t) = `sl`(lam, x, t) `up`(lam, x, t). It is not committed.
15. str_(2 lam + 2)(j, t) = str_(2 lam + 1)(j, t) + rsh(the sum over k2 < f of Wm_9(k2, j) hh(lam,
    k2, t), kd).

*The feed-forward block of a layer with routed experts.* Here f = 2 fe. The features x < fe belong
to the shared expert, and the features fe <= x < f to the one routed expert that the router selects.

16. lw(e, t) = rsh(the sum over k2 < d of Wm_11(k2, e) `mn`(lam, k2, t), kr). Without `rcl 1`, `lr`
    = lw. With `rcl 1`, lw = 8 `ly` + `l3` with 0 <= `l3` <= 7, and `lr` = 8 clamp(`ly`, 2047) +
    `l3`. In both cases -2^14 <= `lr` < 2^14.
17. `sel`(e, t) is 0 or 1, and the sum over e of `sel`(e, t) is 1. With mr(t) = the sum over e of
    `sel`(e, t) `lr`(e, t): for every e, mr(t) - `lr`(e, t) - (the sum over e' > e of `sel`(e', t))
    is at least 0. So the selected expert is the first one that has the largest `lr`.
18. `sc`(t) = 16 round(4096 sigmoid(mr(t) / 256)), by the sigmoid's table (11.1).
19. `xr`(lamM, j, t) = rsh(`sc`(t) `mn`(lam, j, t), 16).
20. For x < fe, line 12 holds with Wm_5 and Wm_6 instead of Wm_3 and Wm_4. For fe <= x < f: gw =
    rsh(the sum over e and over k2 < d of `sel`(e, t) Wm_7(k2, e, x - fe) `xr`(lamM, k2, t), kgR),
    and `up` = rsh(the same sum with Wm_8, kuR). Lines 13 and 14 hold for all x < f.
21. str_(2 lam + 2)(j, t) = str_(2 lam + 1)(j, t) + rsh(A, kd), with A = the sum over k2 < fe of
    Wm_9(k2, j) hh(lam, k2, t), plus the sum over e and over k2 < fe of `sel`(e, t) Wm_10(k2, e, j)
    hh(lam, fe + k2, t).

*A rounding site.* Every rsh(A, k) above is committed as a quotient Q and a remainder rem = A +
2^(k - 1) - 2^k Q with 0 <= rem < 2^k. A site has a **digit form** h. The parameter file gives it
(hq for line 8, hv for 10, ho for 11, hgD and huD for the two of 12 and 20, hdD for 15 and 21, hr
for 16). The sites of line 4 have h = 2, and the site of line 19 has none. With h = 0 nothing of the
remainder is committed, and k is at most 16. With h = 1 or 2 the cells hold rh, the remainder's part
from bit 16 on minus an offset: rem = low + 2^16 (rh + ofs_h) with 0 <= low < 2^16, ofs_1 = 128 and
ofs_2 = 32896 = 128 + 256 x 128. rh has h limbs. k is at most 24 for h = 1 and at most 32 for h = 2.

##### Tensors and cells

*Features.* The matrices af, ad, ar, qa, xbI and xbO (first half) hold **features**. A feature is a
run of T cells, one for each token. Take a matrix in np parts of 2^lm cell rows and k columns each
(np is NAf for af, NAd for ad, and 1 for the others; k is the first half's number of columns), and
let fr = 2^(lm - lt). Feature number ft of the matrix is in part floor(ft / (k fr)), in column
floor(ft / fr) mod k of that part, and its cell for token t is in cell row (ft mod fr) T + t. So a
part holds k fr features.

*Tensors.* A **tensor** is a named family of integers in one matrix. It has nl limbs, cnt instances
and J features for each instance. Its first feature is OFF, the sum of nl cnt J over the tensors
before it in the same matrix, in the order of the table below; a tensor with nl = 0 is absent. Limb
lb of the tensor's entry for instance ix and feature j is the feature

      feat(X, lb, ix, j) = OFF_X + (lb cnt_X + ix) J_X + j

and the tensor's value is X(ix, j, t) = the sum over lb < nl of 256^lb times the cell of feat(X, lb,
ix, j) for token t. Cells are integers of either sign. The features from the last tensor's end up to
np k fr belong to no tensor, and every claim of this part gives them the coefficient 0.

| matrix | tensor | nl | cnt | J | holds (the lines are those of "What the segment computes") |
|---|---|---|---|---|---|
| af | `g0`, then `gc`, with `gcl 1` | 1 each | Lam | f | line 13 |
| af | `gi`, without `gcl 1` | 2 | Lam | f | line 13 |
| af | `sl` | 2 | Lam | f | line 13 |
| af | `up` | 2 | Lam | f | lines 12, 20 |
| af | `rg` | hgD | Lam | f | rh of the site of gw |
| af | `ru` | huD | Lam | f | rh of the site of `up` |
| af | `xs` | XL | 2 Lam - 1 | d | instance ix is the stream str_(ix + 1) |
| af | `ro` | ho | Lam | d | rh of the site of line 11 |
| af | `rd` | hdD | Lam | d | rh of the site of lines 15, 21 |
| af | `aw` | 2 | Lam | d | line 4, n = 0 |
| af | `ra` | 2 | Lam | d | rh of that site |
| af | `mw` | 2 | Lam | d | line 4, n = 1 |
| af | `rm` | 2 | Lam | d | rh of that site |
| af | `qkw` | 2 | Lam | nqk | line 8 |
| af | `rqk` | hq | Lam | nqk | rh of that site |
| af | `vw` | 2 | Lam | dkv | line 10 |
| af | `rv` | hv | Lam | dkv | rh of that site |
| af | `gh`, with `gcl 1` | 2 | Lam | f | line 13 |
| ad | `an` | 2 | Lam | d | line 5 |
| ad | `mn` | 2 | Lam | d | line 5 |
| ad | `xr`, with routed experts | 2 | LamM | d | line 19 |
| ar | `lr`, with routed experts | 2 | LamM | NE | line 16 |
| ar | `rl`, with routed experts | hr | LamM | NE | rh of the site of line 16 |
| ar | `sel`, with routed experts | 1 | LamM | NE | line 17 |
| ar | `sc`, with routed experts | 3 | LamM | 1 | line 18 |
| ar | `row` | 1 | 2 Lam | NR | instance i is norm i's row of digits (below) |
| ar | `ly`, then `l3`, with `rcl 1` | 2; 1 | LamM | NE | line 16 |
| qa | `q` | 2 | Lam | dq | line 9 |
| qa | `kx` | 2 | Lam | dkv | line 9 |
| qa | `v` | 2 | Lam | dkv | line 10 |
| qa | `o` | 2 | Lam | dq | attention's output |
| xbI; xbO | the stream str_0; str_(2 Lam) | XL | 1 | d | OFF = 0 |

An instance of a tensor with cnt = Lam is a layer lam. An instance of a tensor with cnt = LamM is a
layer with routed experts, lamM. The matrix b holds the planes of the lookup's multiplicities
(11.1). A verifier rejects the parameter file unless 2^ls divides d, f, dkv and nqk, and fe with
routed experts; so the first feature of every limb of every instance of a tensor in af, qa, xbI and
xbO is a multiple of 2^ls. It also rejects unless lmaf, lmq and lmx are at least la.

##### The sampled values and the points

A point is written with the coordinate of the highest bit first (11.1). (y, z) is the point y
followed by the point z; 11.1 also writes (y | z). "The last n of z" is the point of the last n
coordinates of z. The elements of steps 2, 3, 5 and 13 are their points' coordinates in the order in
which they are sampled, so the first sampled element is the coordinate of the highest bit.

| name | what it is | coordinates |
|---|---|---|
| zk_c | the key point of call c (11.1) | n_c = ll + lJ_c + lt |
| zl_c, zj_c | the first ll coordinates of zk_c; the next lJ_c | ll; lJ_c |
| zt | the lookup's token point: the last lt of zx, which are the last lt of every zk_c | lt |
| Zk | the last la of zx. Zk = (the last ls of zj_c, zt), for every call c: it is not a call's key point zk_c but the last la of each of them. A capital Z marks a point of la coordinates | la |
| PV | step 2 | lt |
| ZE = (zsE, ztE); tkh | if g = 0, step 3: the first round's ls and then lt elements; the second round's elements | ls + lt; lPd - ls |
| gam | step 4's element | |
| rhoI | step 5 | lNI |
| r6 | step 6's challenges, each as the coordinate of the variable that its round binds | lNI + lPd + lt |
| ZV | the last la of r6 | la |
| omV | step 7's element | |
| rtV | step 8's challenges | lt |
| rt; eps_9 | step 9's challenges; the element of its `PSC epsilon` | lt |
| Zr | (the first ls of Zk, rt) | la |
| om_1 | step 10's element | |
| rs_n, 0 <= n < lamS | step 11's challenges: rs_n is the coordinate of bit n. The first round samples rs_(lamS - 1) | |
| r12; Za | step 12's challenges, as for r6; the last la of r12 | ll + lPf + lt; la |
| rhoM | step 13; without coordinates if ll = 1 | ll - 1 |
| r14; Zb | step 14's challenges, as for r6; the last la of r14 | ll - 1 + lfe + lt; la |

eqlow(lo, hi, n) = eq((rs_(hi - 1), .., rs_lo), n) is 11.1's, on the challenges of step 11. The
element xi that the round `LINE` of steps 6, 12 and 14 samples enters no formula, and these three
levels pass on no start form.

##### The parts of U3, and the two reads

*The entries.* A part of `U3` belongs to a matrix mu of the list and a weight vector phi of len
entries, and it has nb = m_mu / len blocks (11.1). The part's entry number col nb + bk is the sum
over r < len of phi[r] times the cell in cell row bk len + r and column col of the matrix, for col <
k_mu and bk < nb. A matrix in np parts has np parts of `U3` for each weight vector, one after the
other. They are named here by one name, "matrix@point", and at(name) is the number in `U3` of the
first entry of the first of them.

| no. | name | weight vector | entries |
|---|---|---|---|
| 1 to 5 | af@Zk, af@ZV, af@Zr, af@Za, and with routed experts af@Zb | long | NAf k_af 2^(lmaf - la) each |
| 6 | qa@Zk | long | k_qa 2^(lmq - la) |
| 7 to 9 | xbI@Zk, xbI@ZV, xbI@Zr | long | k_x 2^(lmx - la) each |
| 10 | xbI@ZE, only if g = 0 | long | k_x 2^(lmx - la) |
| 11 | xbO@Zk | long | k_x 2^(lmx - la) |
| 12, 13 | ad@zt, ad@rt | short | NAd k_ad 2^(lmad - lt) each |
| 14 to 17 | ar@zt, ar@PV, ar@rtV, ar@rt | short | k_ar 2^(lmar - lt) each |
| 18 | qa@zt | short | k_qa 2^(lmq - lt) |
| 19, 20, 21 | b@0, b@1, b@pin | of 2^lmb entries (11.1: the two weight vectors of the planes' points, then the weights 2^(r mod 32)) | k_b = NC 2^(kT + 5 - lmb) each |

k_af, k_ad, k_ar, k_qa and k_x are the numbers of columns of a part of af, of ad, of ar, of qa and
of xbI and xbO. at(name) is the sum of the numbers of entries of all names before it, in the order
of the table; a name that is absent counts 0. The **short** weight vector at a point z of lt
coordinates is phi[t] = eq(z, t). The **long** weight vector at a point z of la coordinates is
phi[c2 T + t] = eq(the first ls of z, c2) eq(the last lt of z, t), for c2 < 2^ls. By the placement
of the features, entry ft of a short name is the sum over t of eq(z, t) times feature ft's cell for
token t. Entry n of a long name is the same sum over the 2^ls features 2^ls n + c2, each with the
weight eq(the first ls of z, c2).

*The reads.* Every claim of this part names entries of `U3` through one of these forms. mat(X) is
the matrix of the tensor X.

      RS(X, z, ix, j)   = the sum over lb < nl_X of 256^lb U3[ at(mat(X)@z) + feat(X, lb, ix, j) ]
      RL(X, z, ix, jh)  = the sum over lb < nl_X of 256^lb U3[ at(mat(X)@z) + feat(X, lb, ix, 0) / 2^ls + jh ]
      RX(sg, z, jh)     = the sum over lb < XL of 256^lb U3[ at(xbI@z) + lb d / 2^ls + jh ]     for sg = 0
                        = the same with xbO@z                                                  for sg = 2 Lam
                        = RL(xs, z, sg - 1, jh)                                                otherwise

RS is used with a short name and RL and RX with a long one. RS(X, z, ix, j) stands for the sum over
t of eq(z, t) X(ix, j, t). RL(X, z, ix, jh) stands for the sum over c2 < 2^ls and t of eq(the first
ls of z, c2) eq(the last lt of z, t) X(ix, 2^ls jh + c2, t), and RX for the same sum on the stream
str_sg.

##### The scalars by name

The numbers of the scalars follow from the order in which they are sent (11.1). This table names
those that the claims of this part use. The scalars of the sumchecks' rounds, and all scalars of
steps 1 and 18, are 11.1's.

| step | names, in the order sent | what the hidden value is |
|---|---|---|
| 4 | W_(i,s): by norm i, within a norm by s = 0, .., NS - 1 | the sum over t of eq(PV, t) times the sum over j < d and over the ordered pairs (lb, lb') with lb + lb' = s of (limb lb of str_i(j, t)) (limb lb' of str_i(j, t)) |
| 6 | vL_6, vR_6 | both: the sum over i, j, t of eq(r6, (i Pd + j) T + t) times the sum over lb of gam^lb (limb lb of str_i(j, t)) |
| 7 | P_(i,r): by norm i, within a norm by r = 0, .., NPr - 1 | the sum over t of eq(PV, t) times product r of norm i at token t (the norm's row, below) |
| 8 | Cl_8; then the end values rv_(i,k): by norm i, within a norm by k = 0, .., NO - 1 | Cl_8: the sum over i and r of omV^(NPr i + r) P_(i,r). rv_(i,k): with five limbs a, b, Rlo, Rhi of norm i's row at rtV; with four limbs sig, Rlo, Rhi |
| 9 | Cl_9; then the end values Rt_i, i < 2 Lam; sct_lamM, lamM < LamM; selt_(lamM,e), by lamM and within it by e | Cl_9: below, with the terms. The others: R of norm i's row, `sc` and `sel` at rt |
| 11 | the end values A0; H1; H2_lamM, lamM < LamM; B0; v_a, a = 0, .., NCL - E0 - 1 | below, with the panels |
| 12 | vL_12, vR_12 | minus `sl`(lam, x, t) and minus `up`(lam, x, t) at r12, under the index (lam Pf + x) T + t (group 8) |
| 14 | vL_14, vR_14 | minus `sl`(2 lamM + 1, fe + x, t) and minus `up` of the same, x < fe, at r14, under the index (lamM fe + x) T + t (group 9) |
| 15 | v_E, which is v_a for a = NCL - 1 | the embedding's weight claim |
| 16 | fv' | 11.5 |

##### A norm's row of digits

Instance i of the tensor `row` is the row of norm i: NR cells for each token. Cell j holds the digit
minus 128, so the digit is D_j = cell j + 128. At a point z of lt coordinates (z is PV, rtV, rt or
zt) the digit is the form

      D_j(i, z) = U3[ at(ar@z) + OFF_row + i NR + j ] + 128

and every number of the row is a form in the same way. Write dig(n0, n) = the sum over j < n of
256^j D_(n0 + j). The call's segment ROW has the key mul_j D_j in a range of 8 bits. By the source's
comment D_j is then below 256 / mul_j, given a bound on the cell that keeps the key below p. mul_j
is 1 except where the table says otherwise.

| number | with five limbs (NR = 99) | with four limbs (NR = 74) | what it is |
|---|---|---|---|
| Vsq | Vlo = dig(0, 6), Vhi = dig(6, 4); Vsq = Vlo + 2^48 Vhi is not formed | Vsq = dig(0, 8) | the sum of squares, line 1 |
| c_s | c_0 = 0; c_s = dig(10 + 6 (s - 1), 6) - 2^47 for s = 1, .., 8; c_9 = D_9 | c_0 = 0; c_s = dig(8 + 6 (s - 1), 6) - 2^47 for s = 1, .., 6; c_7 = D_7 | the carry into byte s of Vsq |
| s0 | D_59 | D_45 | the lowest bit of sig |
| a, b | a = 2 D_58 + D_59 + 2^8 D_60 + 2^16 D_61; b = D_62 + 2^8 D_63 | none | the two pieces of sig |
| sig | a + 2^24 b | 2 D_44 + D_45 + 2^8 D_46 + 2^16 D_47 + 2^24 D_48 | line 2 |
| l1; h1 | dig(64, 5); dig(69, 5) | dig(49, 4); dig(53, 4) | Vsq + eps(i) - sig^2; 2 sig - l1 |
| Blo; Bhi | dig(74, 3); dig(77, 2) | none | (a b) mod 2^23; floor(a b / 2^23) |
| kap | D_79 - 4 | none | the carry from the low 48 bits of the line LE1 into LE2 (group 6) |
| Rlo; Rhi; R | dig(80, 3); dig(83, 3); Rlo + 2^24 Rhi | dig(57, 3); dig(60, 3); Rlo + 2^24 Rhi | line 3 |
| q1 | dig(86, 3) | none | Rlo b + Rhi a |
| p1 | none | dig(63, 3) | Rhi sig |
| rr; rq | dig(89, 5); dig(94, 5) | dig(66, 4); dig(70, 4) | 2^M(i) + floor(sig / 2) - R sig; sig - 1 - rr |
| mul_j other than 1 | mul_9 = 4, mul_58 = 2, mul_59 = 128, mul_63 = 2, mul_76 = 2, mul_79 = 32 | mul_7 = 32, mul_44 = 2, mul_45 = 128, mul_48 = 4 | |

*The products of a norm.* With five limbs there are seven: P0 = a^2, P1 = a b, P2 = b^2, P3 = Rlo a,
P4 = Rlo b, P5 = Rhi a, P6 = Rhi b. With four limbs there are three: P0 = sig^2, P1 = Rlo sig, P2 =
Rhi sig. Each is a product of two numbers of one token's row. The scalar P_(i,r) of step 7 is
product r of norm i, summed over the tokens with the weights eq(PV, t).

##### The keys of the lookup of step 1

*A key's index.* Call c has 2^lJ_c columns for each layer of the group. The key of layer lam, column
jc and token t has the index (lam 2^lJ_c + jc) T + t among the call's 2^(n_c) keys.

*The columns of a call's segment.* The first half gives the rule; in full it is this. The length n
of each of the call's segments is written in binary. Each 1 bit, from the highest down, gives a
piece of that size, and the pieces cover the indices 0, .., n - 1 in this order. The pieces of all
the call's segments are listed one call's segment after the other, in the order of the call's line.
The list is sorted by decreasing size, and pieces of equal size keep their order. The pieces then
take consecutive columns from column 0 on. pos_SEG(x) is the column of index x of the call's segment
SEG. A column that no piece takes is padding in every layer. A column of a call's segment marked R
(first half) is padding in the layers with an even lam.

*The weights.* For the call c that holds the call's segment SEG, of length n:

      el_c[lam]    = eq(zl_c, lam)
      ew_SEG[x]    = eq(zj_c, pos_SEG(x))                                                        x < n
      ewh_SEG[xh]  = eq(the first lJ_c - ls of zj_c, floor(pos_SEG(2^ls xh) / 2^ls))             xh < n / 2^ls, if 2^ls divides n
      sw_SEG(a, b) = the sum over a <= x < b of ew_SEG[x];    sw_SEG = sw_SEG(0, n)

*The keys.* The table gives the key of index x of each of the calls' segments, for layer lam and
token t. All tensors are taken at (lam, x, t), or at (lamM, x, t) if their cnt is LamM; `sc`, which
has one feature, is taken at (lamM, 0, t) for every x. The key must be a row's value of the call's
table (11.1): for a range of n bits that is 0 <= key < 2^n. Two shapes occur several times:

      SK(A, Q, k, h, rh) = 2^lift(k) (A + 2^(k - 1) - 2^k Q) - 2^16 (rh + ofs_h)        the last term only if h > 0
      HK(rh, k, h)       = 2^(top_h - max(k, 16)) (rh + ofs_h)                           top_1 = 24, top_2 = 32

| call's segment | key |
|---|---|
| G | SK(A, gw, k, hgD, `rg`), with A and k of line 12 or 20, and gw = `gi`, or `g0` + 256 `gh` with `gcl 1` |
| U | SK(A, `up`, k, huD, `ru`), with A and k of line 12 or 20 |
| GH; UH | HK(`rg`, k, hgD); HK(`ru`, k, huD), with the same k |
| UT | `up` + 2^15 |
| QK | SK(Aqk, `qkw`, k, hq, `rqk`), k = kq for x < dq and kk for x >= dq |
| QKH | HK(`rqk`, k, hq), with the same k |
| N1; N2 | SK(str_(2 lam) R_(2 lam), `aw`, K1, 2, `ra`); SK(str_(2 lam + 1) R_(2 lam + 1), `mw`, K2, 2, `rm`) |
| N1H; N2H | HK(`ra`, K1, 2); HK(`rm`, K2, 2) |
| O | SK(A, str_(2 lam + 1) - str_(2 lam), ko, ho, `ro`), A of line 11 |
| D | SK(A, str_(2 lam + 2) - str_(2 lam + 1), kd, hdD, `rd`), A of line 15 or 21 |
| V | SK(A, `vw`, kv, hv, `rv`), A of line 10 |
| OH; DH; VH | HK(`ro`, ko, ho); HK(`rd`, kd, hdD); HK(`rv`, kv, hv) |
| ROW | for x = n NR + j, n < 2, j < NR: mul_j D_j of norm 2 lam + n |
| CN1; CN2 | (`aw` + 2^15) + 2^28 (`an` + 2^11); (`mw` + 2^15) + 2^28 (`mn` + 2^11) |
| CQK | (`qkw` + 2^15) + 2^28 (y + 2^11), y = `q`(lam, x, t) for x < dq and `kx`(lam, x - dq, t) for x >= dq |
| CV | (`vw` + 2^15) + 2^28 (`v` + 2^11) |
| F | (y + 2^15) + 2^28 (`sl` + 2^12), y = `gi`, or `g0` + 256 `gc` with `gcl 1` |
| G0 | `g0` + 128 |
| CG | (`gh` + 49152) + 2^28 (`gc` + 128) + 2^40 |
| XR | `sc` `mn` + 2^15 - 2^16 `xr` |
| LR | SK(A, lw, kr, hr, `rl`), A of line 16, and lw = `lr`, or 8 `ly` + `l3` with `rcl 1` |
| LH | HK(`rl`, kr, hr) |
| LT | 2 (`lr` + 2^14) |
| DD | mr - `lr`(lamM, x, t) - the sum over x' > x of `sel`(lamM, x', t) |
| SEL | 128 `sel` |
| MR | (mr + 2^15) + 2^17 `sc`; with `gcl 1`: (mr + 2^14) + 2^17 `sc` |
| L3 | 32 `l3` |
| CLR | (`ly` + 2^15) + 2^25 (`lr` - `l3`) + 2^39 |

*Three pieces of a key.* A key is the sum of three pieces. The **line piece** is affine in committed
cells. The **inner products** are sums of a weight times one vector over the tokens. The vector is a
committed tensor, or hh in D. They are: 2^lift(k) A in the calls' segments G and U, for all x in a
layer without routed experts and for x < fe in a layer with them; 2^lift(k) A in V, O and LR; and in
D, 2^lift(kd) times the sum with Wm_9. The **terms** are products of two vectors over the tokens:
str_i R_i in N1 and N2; 2^lift(kq) Aqk in QK; `sc` `mn` in XR; mr in DD and MR; and with routed
experts 2^lift(k) A for fe <= x < f in G and U, and in D the sum with Wm_10 times 2^lift(kd). A
verifier rejects the scale file unless kq = kk or both are at least 16, so lift(kq) = lift(kk).

*The line pieces as forms.* LINE_SEG(lam) is the sum over x < n of ew_SEG[x] times the line piece of
the key at zt. It is formed from these, where the weights are those of SEG, and ix is the tensor's
instance: lam, or lamM if its cnt is LamM:

      SL(X; a, b; x0) = the sum over a / 2^ls <= xh < b / 2^ls of ewh_SEG[xh] RL(X, Zk, lam, xh - x0 / 2^ls)
      SL(X)           = SL(X; 0, n; 0),    SL(X; a, b) = SL(X; a, b; 0)
      SX(sg)          = the sum over xh < d / 2^ls of ewh_SEG[xh] RX(sg, Zk, xh)
      SS(X)           = the sum over x < n of ew_SEG[x] RS(X, zt, ix, x)
      SITE(sw, Q, k, h, rh) = 2^lift(k) (2^(k - 1) sw - 2^k Q) - 2^16 (ofs_h sw + rh)        the last term only if h > 0
      HIGH(sw, rh, k, h)    = 2^(top_h - max(k, 16)) (ofs_h sw + rh)

In SITE and HIGH, sw is a number and Q and rh are forms. Below, sw alone is sw_SEG, sw(a, b) is
sw_SEG(a, b), and Qg(a, b) = SL(`gi`; a, b), or SL(`g0`; a, b) + 256 SL(`gh`; a, b) with `gcl 1`.

| call's segment | LINE_SEG(lam) |
|---|---|
| G, layer without routed experts | SITE(sw, Qg(0, f), kg, hgD, SL(`rg`)) |
| G, layer with routed experts | SITE(sw(0, fe), Qg(0, fe), kg, hgD, SL(`rg`; 0, fe)) + SITE(sw(fe, f), Qg(fe, f), kgR, hgD, SL(`rg`; fe, f)) |
| U | the same with SL(`up`; a, b) for Qg(a, b), ku, kuR, huD and `ru` |
| GH, layer without routed experts | HIGH(sw, SL(`rg`), kg, hgD) |
| GH, layer with routed experts | HIGH(sw(0, fe), SL(`rg`; 0, fe), kg, hgD) + HIGH(sw(fe, f), SL(`rg`; fe, f), kgR, hgD) |
| UH | the same with `ru`, ku, kuR, huD |
| UT | SL(`up`) + 2^15 sw |
| QK | SITE(sw(0, dq), SL(`qkw`; 0, dq), kq, hq, SL(`rqk`; 0, dq)) + SITE(sw(dq, nqk), SL(`qkw`; dq, nqk), kk, hq, SL(`rqk`; dq, nqk)) |
| QKH | HIGH(sw(0, dq), SL(`rqk`; 0, dq), kq, hq) + HIGH(sw(dq, nqk), SL(`rqk`; dq, nqk), kk, hq) |
| N1; N2 | SITE(sw, SL(`aw`), K1, 2, SL(`ra`)); SITE(sw, SL(`mw`), K2, 2, SL(`rm`)) |
| N1H; N2H | HIGH(sw, SL(`ra`), K1, 2); HIGH(sw, SL(`rm`), K2, 2) |
| O | SITE(sw, SX(2 lam + 1) - SX(2 lam), ko, ho, SL(`ro`)) |
| D | SITE(sw, SX(2 lam + 2) - SX(2 lam + 1), kd, hdD, SL(`rd`)) |
| V | SITE(sw, SL(`vw`), kv, hv, SL(`rv`)) |
| OH; DH; VH | HIGH(sw, SL(`ro`), ko, ho); HIGH(sw, SL(`rd`), kd, hdD); HIGH(sw, SL(`rv`), kv, hv) |
| ROW | the sum over x < 2 NR of mul_(x mod NR) ew[x] D_(x mod NR)(2 lam + floor(x / NR), zt) |
| CN1; CN2 | SL(`aw`) + 2^28 SS(`an`) + (2^15 + 2^39) sw; SL(`mw`) + 2^28 SS(`mn`) + (2^15 + 2^39) sw |
| CQK | SL(`qkw`) + 2^28 (SL(`q`; 0, dq) + SL(`kx`; dq, nqk; dq)) + (2^15 + 2^39) sw |
| CV | SL(`vw`) + 2^28 SL(`v`) + (2^15 + 2^39) sw |
| F | SL(`gi`) + 2^28 SL(`sl`) + (2^15 + 2^40) sw; with `gcl 1`, SL(`g0`) + 256 SL(`gc`) instead of SL(`gi`) |
| G0 | SL(`g0`) + 128 sw |
| CG | SL(`gh`) + 2^28 SL(`gc`) + (49152 + 2^35 + 2^40) sw |
| XR | 2^15 sw - 2^16 SS(`xr`) |
| LR | SITE(sw, SS(`lr`), kr, hr, SS(`rl`)); with `rcl 1`, 8 SS(`ly`) + SS(`l3`) instead of SS(`lr`) |
| LH | HIGH(sw, SS(`rl`), kr, hr) |
| LT | 2 SS(`lr`) + 2^15 sw |
| DD | minus the sum over x < NE of ew[x] (RS(`lr`, zt, lamM, x) + the sum over x < x' < NE of RS(`sel`, zt, lamM, x')) |
| SEL | 128 SS(`sel`) |
| MR | ew[0] (2^15 + 2^17 RS(`sc`, zt, lamM, 0)); with `gcl 1`, 2^14 instead of 2^15 |
| L3 | 32 SS(`l3`) |
| CLR | SS(`ly`) + 2^25 (SS(`lr`) - SS(`l3`)) + (2^15 + 2^39) sw |

So the tensors of af, qa, xbI and xbO are read through the long parts at Zk, and those of ad and ar
through the short parts at zt. The forms of a call's segment marked R are formed for odd lam only.

##### The start forms and the equality points of the sumchecks

| step | start form C_n | the point of its equality rounds |
|---|---|---|
| 6 | 2^(-lPd) times the sum over i < NI and s < NS of eq(rhoI, i) gam^s W_(i,s) | (rhoI, then lPd times the constant 1/2, PV) |
| 8 | Cl_8 | PV |
| 9 | Cl_9 | zt |
| 11 | C_11, below | none: plain rounds |
| 12 | H1 | (rs_(ll + lPf - 1), .., rs_0, zt) |
| 14 | the sum over lamM < LamM of eq(rhoM, lamM) H2_lamM | (rhoM, rs_(lfe - 1), .., rs_0, rt) |
| 16 | the merge's start form (11.1), on the NCL values v_a in the order of the weight claims | none: plain rounds |

1/2 and 2^(-lPd) are inverses in F. In steps 6, 12 and 14 the variable of bit n has the coordinate
number (number of variables - 1 - n) of the point, whatever the order of the rounds.

*The start form of step 11.* It holds the lookup's join, the claimed sum of step 9, and what step 9
leaves:

      C_11 = the sum over c < NC of e_c JP_c
             + the sum over NT <= y < 2^rK of eq(zy, y) (alpha - padkey_0)
             - Cl_9 - C_end + om_1 KT
      JP_c = alpha - c theta - tab_c(t0_c) PADW_c - the sum over lam < Lam of el_c[lam] (the sum over the call's segments SEG of LINE_SEG(lam))
      PADW_c = the sum over lam < Lam of el_c[lam] (the sum over the columns jc < 2^lJ_c that are padding in layer lam of eq(zj_c, jc))
      KT   = - (1 / eps_9) Left(C_9; 0; 0)

In JP_c a call's segment marked R is left out for even lam. Left(C_9; 0; 0) is the form that 11.1
gives for step 9 with S0 = 0 and the kept form 0; it is not added as a claim of its own. It names
Cl_9 and the mask entries of step 9's hiding polynomial. A verifier gives no acceptance if eps_9 = 0
("Two tests beside the claims", below).

*The running claim of step 11 starts from the public part of C_11* (11.1): the constant of C_11,
plus each coefficient of a hidden value times its shown value. The entries of `U3` count as 0, and
so do the mask entries of step 9's hiding polynomial, whose shown value is 0 (11.1): the scalars
that were sent under their numbers are not put in.

*What C_11 stands for.* With KF_c the key vector of call c at zk_c, the lookup's join (11.1) is: the
sum over c of e_c (alpha - c theta - KF_c), plus the second line above, minus C_end, is 0. KF_c is
tab_c(t0_c) PADW_c plus the sum over lam of el_c[lam] times the three pieces of the keys. Cl_9 is
the sum of all terms at zt: the sum over t of eq(zt, t), over c of e_c, over lam of el_c[lam], over
the call's segments and over x of ew_SEG[x] times the term of the key. KT is the same sum with each
term's two factors taken at rt one by one: the sum over t of eq(rt, t) times the first factor, times
the same sum for the second. A factor that step 9 shows (R, `sc`, `sel`) enters as it was sent: Rt',
sct', selt'. So C_11 is (the inner products of all keys at zt) + om_1 (the terms at rt), and step 11
writes both as one sum of products.

*What 11.1 leaves to the host of a lookup.* The lookup of step 1 sends the segment's first scalars:
jLK = 0. Its claim LK0 has the claim number 0 (group 1). Its claim LP0 has the number that is the
sum of the counts of groups 1 to 10 (group 11), where TB_0, TB_1 and TP are given. Its join LJ is
not a claim: C_11 holds it. The padding positions of call c are the T keys of every column jc that
is padding in layer lam. Column jc of layer lam is in tree y = tree0(c) + floor((lam 2^lJ_c + jc) /
2^(dT - lt)), so npad(y) is T times the number of such pairs (lam, jc) of the call of tree y. The
key form KF_c is not a form in committed entries here, because the keys hold products with weights;
the three pieces above are used instead of it.

##### The panels of step 11

Step 11 is a sumcheck for the statement C_11 = the sum over n < 2^lamS of SA[n] SB[n]. The two
vectors SA and SB are made of **panels**. A panel belongs to one of two passes (first: inner
products at zt; second: terms at rt). It has the weight e_c of the call that its call's segment is
in, ni rows, ny columns, and a coefficient wt_ri for each row. Outside the panels, and in a panel's
entries with ri >= ni or yv >= ny, SA SB is 0.

*The list of the panels.* First pass: the calls that hold one of the calls' segments G, U, V, O, D,
LR, in the order of the call list; within a call, the call's segments in the order of its line.
Second pass: the calls that hold one of N1, N2, QK, or with routed experts G, U, XR, D, DD, MR, in
the same way. A row number ri stands for the layer lam = ri if the table's ni is Lam, and for lam =
2 ri + 1, lamM = ri, if it is LamM. In a selector panel, whose ni is NE, ri is the number of an
expert; the column "kind" decides, also where NE is the same number as Lam or LamM. el is el_c of
the panel's call. A panel whose second factor is "weights" has weight claims instead. They take the
next numbers a in the order of this list. A panel has one claim for each row, by increasing ri; a
selector panel has one claim in all.

| pass | call's segment | kind | ni | ny | first factor FA(ri, yv) | second factor FB(ri, yv) | wt_ri |
|---|---|---|---|---|---|---|---|
| 1 | G; U | weights | Lam | d | RS(`mn`, zt, lam, yv) | weights | el[lam] 2^lift(kg); el[lam] 2^lift(ku) |
| 1 | V | weights | Lam | d | RS(`an`, zt, lam, yv) | weights | el[lam] 2^lift(kv) |
| 1 | O | weights | Lam | dq | RS(`o`, zt, lam, yv) | weights | el[lam] 2^lift(ko) |
| 1 | D | weights, first factor not sent | Lam | f | the sum over t of eq(zt, t) hh(lam, yv, t), known through H1 alone | weights | el[lam] 2^lift(kd) |
| 1 | LR | weights | LamM | d | RS(`mn`, zt, lam, yv) | weights | el[lam] 2^lift(kr) |
| 2 | N1; N2 | pair | Lam | 1 | the number Rt'_(2 lam); Rt'_(2 lam + 1) | the sum over xh < d / 2^ls of ewh_SEG[xh] RX(2 lam + n, Zr, xh), n = 0; 1 | el[lam] |
| 2 | QK | weights | Lam | d | RS(`an`, rt, lam, yv) | weights | el[lam] 2^lift(kq) |
| 2 | XR | pair | LamM | 1 | the number sct'_lamM | the sum over x < d of ew_XR[x] RS(`mn`, rt, lam, x) | el[lam] |
| 2 | DD; MR | pair | LamM | NE | the number selt'_(lamM,yv) | RS(`lr`, rt, lamM, yv) | el[lam] sw_DD; el[lam] ew_MR[0] |
| 2 | G; U, with routed experts: one panel for each lamM, by increasing lamM | selector | NE | d | selt'_(lamM,ri) RS(`xr`, rt, lamM, yv) | weights | one for the panel: el[2 lamM + 1] 2^lift(kgR); el[2 lamM + 1] 2^lift(kuR) |
| 2 | D, with routed experts: one panel for each lamM | selector, column factor not sent | NE | fe | selt'_(lamM,ri) times the sum over t of eq(rt, t) hh(2 lamM + 1, fe + yv, t), known through H2_lamM alone | weights | one for the panel: el[2 lamM + 1] 2^lift(kd) |

The shifts are those of layer l = g Lam + lam. In the second pass a call's other segments give no
panel.

*The placement.* A panel has wy = clog2(ny) column bits and w = clog2(ni) + wy bits in all. The
panels are sorted by decreasing w, and panels of equal w keep the order of the list. off is the sum
of 2^w over the panels before it in the sorted order. Entry (ri, yv) of a panel has the index off +
ri 2^wy + yv. lamS is clog2 of the sum of all 2^w (first half).

*The entries of SA and SB.* Let n = off + ri 2^wy + yv, and om = 1 in the first pass, om = om_1 in
the second. In the D panel of the first pass SA[n] = FA(ri, yv) and SB[n] = e_c wt_ri FB(ri, yv). In
every other panel SA[n] = om e_c wt_ri FA(ri, yv) and SB[n] = FB(ri, yv); a selector panel has one
wt for all rows. Where FB is "weights", take the panel's weight claim for row ri (a selector panel
has one claim). FB(ri, yv) is the sum over the weights of its unit that have the index yv in the
index group `kk`, and in a selector panel the index ri in the index group `e`. Each weight is
multiplied by the weights that the claim's other factors give to its other indices. FB(ri, yv) is 0
if yv is not below the extent of `kk`.

*At the challenges.* For each panel let

      sb    = eqlow(w, lamS, off / 2^w)
      EY[yv] = eqlow(0, wy, yv),     EI[ri] = eqlow(wy, w, ri)
      co    = sb om e_c,           with om = 1 for a panel of the first pass and om = om_1 for one of the second
      s1    = the sum over ri < NE of selt'_(lamM,ri) EI[ri]          for a selector panel

v0 is the number of the panel's first weight claim. The verifier computes two numbers As and Bs and
two forms Alin and Blin as sums over all panels:

| kind of panel | adds to As | adds to Alin | adds to Bs | adds to Blin |
|---|---|---|---|---|
| weights | | the sum over ri, yv of co wt_ri EI[ri] EY[yv] FA(ri, yv) | the sum over ri of sb EI[ri] v'_(v0 + ri) | |
| weights, first factor not sent (D, first pass) | sb H1' | | the sum over ri of sb EI[ri] e_c wt_ri v'_(v0 + ri) | |
| pair | | the constant: the sum over ri, yv of co wt_ri EI[ri] EY[yv] FA(ri, yv) | | the sum over ri, yv of sb EI[ri] EY[yv] FB(ri, yv) |
| selector | | the sum over yv of co wt s1 EY[yv] RS(`xr`, rt, lamM, yv) | sb v'_v0 | |
| selector, column factor not sent (D, second pass) | co wt s1 H2'_lamM | | sb v'_v0 | |

To these sums As adds A0' and Bs adds B0'. The sums run over ri < ni and yv < ny. So A0 is all of SA
at the challenges that is a form in `U3`, and B0 is the same for SB. H1 is the extension of the D
panel's first factor at (rs_(ll + lPf - 1), .., rs_0). H2_lamM is the extension of the column factor
at (rs_(lfe - 1), .., rs_0). The summand of step 11 at the challenges is As Bs.

##### The weight claims

Claim a has a unit, named by its kind X and its layer, a value v_a, and one factor for each of the
unit's index groups (11.5). The order of the claims is the order of the panels' list, not the sorted
one. It is the order of the first half's paragraph "The weight claims, in order".

*Two kinds of factor* for an index group of class low or high with extent o 2^z, o odd (11.5):

      FR(lo0, nbit): pt[n] = rs_(lo0 + z - 1 - n) for n < z;              HV[hi] = eqlow(lo0 + z, lo0 + nbit, hi) for hi < o
      FE(SEG, x0)  : pt[n] = coordinate number lJ_c - z + n of zj_c;      HV[hi] = eq(the first lJ_c - z of zj_c, floor(pos_SEG(x0 + 2^z hi) / 2^z))

c is the call that holds SEG. FE gives the index x of the group the weight ew_SEG[x0 + x]. FR gives
the index its equality weight at the nbit challenges rs_lo0, .., rs_(lo0 + nbit - 1).

| from | unit: kind X, layer | factors, by the name of the index group |
|---|---|---|
| first pass, G; U | X = 3; 4 in a layer without routed experts, 5; 6 in a layer with them; layer l | `kk`: FR(0, lPd). `x`: FE(SEG, 0) |
| first pass, V | 1, l | `kk`: FR(0, lPd). `x`: FE(V, 0) |
| first pass, O | 2, l | `kk`: FR(0, clog2(dq)). `x`: FE(O, 0) |
| first pass, D | 9, l | `kk`: FR(0, lPf). `x`: FE(D, 0) |
| first pass, LR | 11, l, for odd lam | `kk`: FR(0, lPd). `x`: FE(LR, 0) |
| second pass, QK | 0, l | `hd`: AV = AH. `cd`: AV = rot. `kk`: FR(0, lPd) |
| second pass, G; U | 7; 8, layer g Lam + 2 lamM + 1 | `kk`: FR(0, lPd). `e`: FR(lPd, le). `x`: FE(SEG, fe) |
| second pass, D | 10, layer g Lam + 2 lamM + 1 | `kk`: FR(0, lfe). `e`: FR(lfe, le). `x`: FE(D, 0) |
| if g = 0: the embedding | 12, layer L | `v`: AV = tokw. `x`: pt and HV from pnt, below |

In a layer with routed experts the group `kk` of the unit of kind 9 has the extent fe. FR(0, lPf)
then has z = lfe and HV[0] = 1 - rs_lfe: the claim reads the columns yv < fe of the D panel alone.
In the same way the group `x` of the kinds 5 and 6 has the extent fe, and FE(SEG, 0) covers x < fe.

*The factors of the queries and keys.* Let c be the call that holds QK.

      AH[hd]  = eq(the first lJ_c - ldh of zj_c, pos_QK(hd dh) / dh)   for hd < H + G;      AH[hd] = 0 for H + G <= hd < 64
      lf[ch]  = eq(the last ldh of zj_c, ch)                            for ch < dh
      Ct[pa]  = the sum over t < T of eq(rt, t) Co(l, t, pa),     St[pa] = the sum over t < T of eq(rt, t) Si(l, t, pa)
      rot[2 pa]     = Ct[pa] lf[2 pa] + St[pa] lf[2 pa + 1]
      rot[2 pa + 1] = Ct[pa] lf[2 pa + 1] - St[pa] lf[2 pa]

Co and Si are the rotary tables' integers (11.1), for pa < dh / 2. In a layer whose word rp is 0,
Ct[pa] = 2^12 and St[pa] = 0.

*The factors of the embedding.* tokw[vv] = the sum of eq(ztE, t) over the positions t < T with tok_t
= vv, for vv below the line `V`. Let pnt = (tkh, zsE), a point of lPd coordinates. For the group
`x`, of extent d = o 2^z: pt[n] = coordinate number lPd - z + n of pnt, and HV[hi] = eq(the first
lPd - z of pnt, hi).

##### The claims, group by group

Each line below is one claim, a form that must be 0, and the lines are in the order of the claims'
numbers. The groups and their counts are those of the first half's table. A number of a norm's row
with the arguments (i, z) is the form of the heading "A norm's row of digits" at the point z.

*Group 1*, 31 + v' + d' + 1 claims: the claims that the sumchecks of the lookup of step 1 leave, in
11.1's order: the 31 of the tie, the v' of the key side's levels, the d' of the table side's levels,
the one of the bits.

*Group 2*, 3 claims.

1. Left(C_6; vL'_6 vR'_6; 0) = 0.
2. vL_6 - vR_6 = 0.
3. vL_6 - the sum over i < NI, kh < d / 2^ls and lb < XL of eq(the first lNI + lPd - ls of r6, i Pd
   / 2^ls + kh) gam^lb ent(i, lb, kh) = 0. Here ent(0, lb, kh) = U3[at(xbI@ZV) + lb d / 2^ls + kh],
   and for i >= 1, ent(i, lb, kh) = U3[at(af@ZV) + feat(`xs`, lb, i - 1, 0) / 2^ls + kh].

*Group 3*, 2 claims.

1. Cl_8 - the sum over i < NI and r < NPr of omV^(NPr i + r) P_(i,r) = 0.
2. Left(C_8; S0; 0) = 0. With five limbs, and a', b', lo', hi' for rv'_(i,0), rv'_(i,1), rv'_(i,2),
   rv'_(i,3), S0 is the sum over i of omV^(7 i) a' a' + omV^(7 i + 1) a' b' + omV^(7 i + 2) b' b' +
   omV^(7 i + 3) lo' a' + omV^(7 i + 4) lo' b' + omV^(7 i + 5) hi' a' + omV^(7 i + 6) hi' b'. With
   four limbs, and s', lo', hi' for rv'_(i,0), rv'_(i,1), rv'_(i,2), S0 is the sum over i of omV^(3
   i) s' s' + omV^(3 i + 1) lo' s' + omV^(3 i + 2) hi' s'.

*Group 4*, NO NI claims, by i and within i by k: rv_(i,k) - y_k(i, rtV) = 0, where (y_0, .., y_3) =
(a, b, Rlo, Rhi) with five limbs and (y_0, y_1, y_2) = (sig, Rlo, Rhi) with four.

*Group 5*, NS NI claims, by i and within i by s = 0, .., NS - 1: W_(i,s) + c_s(i, PV) - D_s(i, PV) -
256 c_(s+1)(i, PV) = 0.

*Group 6*, 8 NI or 5 NI claims, by i. All numbers of the row are taken at (i, PV).

With five limbs, in this order:

1. (LE0) P_(i,1) - Blo - 2^23 Bhi = 0.
2. (LE1) Vlo + eps(i) - P_(i,0) - 2^25 Blo - l1 + 2^48 kap = 0.
3. (LE2) Vhi - Bhi - P_(i,2) - kap = 0.
4. (LE3) l1 + h1 - 2 sig = 0.
5. (LR1) P_(i,4) + P_(i,5) - q1 = 0.
6. (LR2) P_(i,6) = 0.
7. (LR3) 2^(M(i) + 1) + sig - s0 - 2 P_(i,3) - 2^25 q1 - 2 rr = 0.
8. (LR4) rr + rq - sig + 1 = 0.

With four limbs, in this order:

1. Vsq + eps(i) - P_(i,0) - l1 = 0.
2. l1 + h1 - 2 sig = 0.
3. P_(i,2) - p1 = 0.
4. 2^(M(i) + 1) + sig - s0 - 2 P_(i,1) - 2^25 p1 - 2 rr = 0.
5. rr + rq - sig + 1 = 0.

*Group 7*, 2 Lam + LamM (1 + NE) claims, in the order in which step 9 sends its end values.

1. For i < 2 Lam: Rt_i - R(i, rt) = 0.
2. For lamM < LamM: sct_lamM - RS(`sc`, rt, lamM, 0) = 0.
3. For lamM < LamM and within it e < NE: selt_(lamM,e) - RS(`sel`, rt, lamM, e) = 0.

*Group 8*, 3 claims. Note the plus signs: the two end values are minus the tensors' values.

1. Left(C_12; vL'_12 vR'_12; 0) = 0.
2. vL_12 + the sum over lam < Lam and kh < f / 2^ls of eq(the first ll + lPf - ls of r12, lam Pf /
   2^ls + kh) RL(`sl`, Za, lam, kh) = 0.
3. vR_12 + the same sum with RL(`up`, Za, lam, kh) = 0.

*Group 9*, with routed experts, 3 claims.

1. Left(C_14; vL'_14 vR'_14; 0) = 0.
2. vL_14 + the sum over lamM < LamM and kh < fe / 2^ls of eq(the first ll - 1 + lfe - ls of r14,
   lamM fe / 2^ls + kh) RL(`sl`, Zb, 2 lamM + 1, fe / 2^ls + kh) = 0.
3. vR_14 + the same sum with RL(`up`, Zb, 2 lamM + 1, fe / 2^ls + kh) = 0.

*Group 10*, with routed experts, LamM claims, by lamM: the sum over e < NE of RS(`sel`, zt, lamM,
e) - 1 = 0. This statement on every token is tested at the one point zt and has no round of its own
in the error accounting (13.6; 17.1, item 3).

*Group 11*, 3 claims: the three claims of the lookup of step 1 on its planes, in 11.1's order (the
leaves; the planes at the bits' point; the number of keys). 11.1 writes them with the sums of a
column of b under three weight vectors. For column col < k_b these are TB_0(col) = U3[at(b@0) +
col], TB_1(col) = U3[at(b@1) + col] and TP(col) = U3[at(b@pin) + col].

*Group 12*, only if g = 0, 1 claim: the sum over kh < d / 2^ls of eq(tkh, kh) RX(0, ZE, kh) - v_E
= 0.

*Group 13*, 36 + v'' + d'' + NM + NX claims: the smallness test's, in 11.1's order. Its parts of
`U3` are those of the table of names above, in that order.

*Group 14*, 3 claims.

1. Left(C_11; As Bs; 0) = 0.
2. A0 - Alin = 0.
3. B0 - Blin = 0.

*Group 15*, 1 claim: Left(C_16; fv'; 0) = 0, the merge's (11.1, 11.5).

*Two tests beside the claims.* These are the two of this part; the smallness test has its own (11.1;
13.9, events 2 to 8). Both are tests on content of every verifier (13.9, events 1 and 9). First: a
verifier gives no acceptance if eps_9 = 0, because the claim is divided by eps_9; the programs
reject (`coin-zero`). Second: after all claims are added up with their powers of chi, a verifier
looks at the coefficient of each hidden value, and gives no acceptance if the coefficient under a
scalar's number other than that of fv' is 0. In that case the programs end the call under the name
`internal-g-free-slot`, after the prefix `G`g`:`. They do the same if the coefficient under the
number of fv' is not 0, which no proof can cause. Such a call ends `NOT CHECKED` (14.3). The source
gives as the purpose of the second test that every scalar is in some claim. It adds that the test
guards hiding: that every scalar is in a claim is what lets whoever verifies a proof before it is handed on see an end value that was sent
without its mask. What a verifier with two answers does, and how often an honest proof meets either
event, is in 13.9.

*What cannot fail.* The programs compare many of the numbers of this part with a second computation
of the same number: the lookup's sizes, the length of `U3`, lamS, the numbers of weight claims, of
scalars and of claims. They also compare the columns that the calls' segments list, which the forms
are meant to weigh, with the padding's numbers npad(y) (the source says of this test that a slip
inside a form is not seen by it: `verifier/src/vf_layers_check.inc`), and the last la coordinates of every
zk_c with those of zx. They end the call under names that begin with `internal-`. Under
`plan-tensor-layout` they reject at three places where the claims on the norms are formed
(`verifier/src/vf_layers_check.inc`): there they compare the numbers of limbs of two tensors, and the
numbers of sums and of end values of a norm, with what the number of limbs of a value of the stream
gives. They also reject under `hid-profile-segment-` followed by the name of a call's segment for
which they have no form LINE_SEG. All of this follows from numbers that the earlier tests on the
public files have fixed: by this page's reading of the source no public file that has passed those
tests, and no proof, can make one of these tests fail, and a second verifier needs none of them.
This holds for these places alone. Under `plan-tensor-layout` the programs also make tests on the
public files, which every verifier makes (11.3; 13.8, the row of that name). Under the same name
shared routines that this segment calls too compare the number of a point's coordinates, or the
place of a head's features in a tensor, with the program's own sizes (`verifier/src/vf_attention_shape.h`);
this part says nothing of those. The second of the two tests above has a name that begins with
`internal-` too, but it depends on sampled values, and so on a proof's bytes, and a second verifier
needs it (13.9).

##### The public numbers that the formulas take

| from | what | used in |
|---|---|---|
| the scale file | for layer l the 13 words kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, sh, rp, in this order: word number 13 l + n. After the 13 L words come, for each odd layer l of a model with routed experts, the three words kr, kgR, kuR: word number 13 L + 3 floor(l / 2) + n | the shifts in SITE, HIGH and wt_ri; M(i) in group 6; rp in Ct and St. sh is attention's (11.3) |
| the parameter file | `eps1` = `eps1_lo` + 2^32 `eps1_hi`, and `eps2` in the same way | group 6 |
| the parameter file | the shape, the digit forms, the matrices' sizes, the call lines, `xl`, `rcl`, `gcl`, `eg`, `smax`, `lidb`, `theta0_lo`, `theta0_hi` | the tensors, the columns pos_SEG, the tables |
| the tables (11.1) | tab_c(t0_c), the padding key's value; Co and Si | JP_c; rot |
| the request | tok_t, t < T | tokw, if g = 0 |

The constants of this part, with their names in the source: 2047 (`CLAMP`), the clamp's bound; 15
(`LBU`), the exponent in the key of UT; 128 and 32896 (`o1`, `o2`), the offsets ofs_1 and ofs_2;
12, the only admitted value of the line `cs`, the exponent of the rotary tables; 64, the extent of
the index group `hd`.

##### The tests on the public numbers

A verifier makes these tests on the public files, before it reads a proof, and rejects under the
name given (section 13). They are listed here because the formulas above rest on them.

A call ends with an acceptance only if, beyond these tests, the conditions of the standard parameter
form are met (13.7). Among them, on this segment's numbers, are `routed-experts` (NE is above 0),
`five-limbs` (the line `xl 5`), `both-clamps` (the lines `rcl 1` and `gcl 1`), `no-range-words`
(neither the line `eg` nor the line `smax`) and `norm-shifts` (K1 <= M1 and K2 <= M2 in every
layer). So this page also writes out forms on which no call ends with an acceptance: four limbs,
`NE 0`, a missing `rcl 1` or `gcl 1`, a line `eg` or `smax`, and a shift K above M.

*The shape.*

| name | a verifier rejects unless |
|---|---|
| `shape-words` | 4 <= lt <= 16; 1 <= ll <= 6; 1 <= d, f <= 2^20; nmin <= 28; and with routed experts: fe is a power of two, f = 2 fe, and not (ll = 1 and fe = 1) |
| `shape-L` | L >= 1 and Lam divides L |
| `shape-la` | lt < la <= 16; 2^ls divides d, f, dkv, nqk, and fe with routed experts; lmq >= la; lmaf >= la |
| `shape-heads` | H, G >= 1; dh is a power of two, at least 2; H dh = d; H + G <= 64 |
| `shape-NE` | with routed experts: NE is a power of two from 2 to 4096 |
| `policy-norm` | 1 <= `eps1`, `eps2` < 2^59 |
| `policy-cs`; `policy-ut` | the line `cs` is 12; the line `ut` is 1 |
| `policy-forms`; `header-absent-class` | every digit form is at most 2; hr = 0 without routed experts |
| `policy-gcl`; `policy-rcl` | `gcl 1`, `rcl 1` only with routed experts |
| `shape-request-matrix` | lt <= lmaf, lmad, lmar, lmq <= 24; 1 <= NAf, NAd <= 64; la <= lmx <= 24 |
| `shape-lmb` | lmb <= 24 |
| `forest-planes` | 5 <= lmb <= kT + 5 (11.1). The number of columns k_b = NC 2^(kT + 5 - lmb) rests on it |
| `shape-call-` and the call's name | max(10, nmin, la) <= n_c <= 44; lJ_c >= ls; the line `iota_`c equals n_c - dT |
| `header-absent-call` | the lines `iota_`c with c from NC to 7 are 0 |
| `forest-depth-G` | dT >= la |
| `shape-product-layer` | lNI + lPd + lt >= 9; ll + lPf + lt >= 9; with routed experts ll - 1 + lfe + lt >= 9. Why: the range order of steps 12 and 14 names the bits 0 to 8. The programs make the same test for step 6 |
| `shape-theta0` | theta0 is not 0 |

*The scale file.*

| name | a verifier rejects unless |
|---|---|
| `KL-length` | the file has 13 L words, and 3 L / 2 more with routed experts |
| `KL-shift-range` | every shift k with the digit form h has 1 <= k <= 16 for h = 0, 24 for h = 1, 32 for h = 2. The digit forms are: kg and kgR: hgD; ku and kuR: huD; kd: hdD; kq and kk: hq; kv: hv; ko: ho; kr: hr |
| `KL-kq-kk` | kq = kk, or both are at least 16 |
| `KL-norm` | for (M, K) = (M1, K1) and (M2, K2): 24 <= M <= 47, 17 <= K <= 32 and K + 14 >= M |
| `KL-rp` | rp is 0 or 1 |

*The no-overflow conditions.* A claim or a key's membership in a table is an equation in the field.
The source gives as the purpose of these conditions that an equation which the proof tests in the
field also holds over the integers, and it says that this is not proved
(`verifier/docs/parameter-file.md`). The
conditions do not establish the bounds that they use. They take them as given, and by the comments
in the source (`cp_verify.cpp`, at the conditions) the bounds have different origins. Ty is assumed
of the smallness test. BC on `an`, `mn` and `v`, the bound Qb of `gate` and the 2^15 in `down` come
from other lookups of this segment. lsg, and with it the bound A of `norm` and the bounds SP and Qb
of `o` and `down`, comes from the norms' rows of digits. BC on `o` comes from the attention segment
(11.3): `o` has no clamp, and no lookup of this segment bounds it. BW comes from the key file
(below). For BC on `xr`, which acc takes at kgR and kuR, the source names no origin. Of
`clamp-keys`, `MR` and `one-hot` the source says that they use the bounds on the cells only, and of
`lr` with `rcl 1` that only Ty2 bounds `ly`. The conditions use 2^60, or 2^59, whatever the field;
the programs of this repository can be built for field 2 alone (R1.2). The rejection name is
`no-wrap-` followed by the name in the table. The numbers are: Ty = 2^18, the bound on a cell that
the conditions assume of the smallness test (the source's `TyReq`), which is twice that test's
offset 2^17 (`LBZ` = 17; 11.1); Ty2 = 257 Ty and Ty3 = 65793 Ty, the bounds on a value of two and
of three limbs; BW = 2^12, the bound on a weight (`WEIGHT_BOUND`), which does not hold for a weight of the
embedding, of four digits: the no-overflow condition of the one equation of this part that reads the
embedding is `no-wrap-H-J-E`, among the head's (11.4, second half, condition 12; 13.8); BC = 2^11
(`CLAMPED_CELL_BOUND`), the bound on `an`, `mn`, `v`, `xr` and `o`; acc = d BC BW; lsg = 39 with five limbs and 30
with four.

A rounding site with the name nm, the shift k, the digit form h, a bound A on its sum, a span SP and
a bound Qb on its quotient has two conditions:

      U-nm:   SP 2^max(k, 16) <= 2^60
      W-nm:   (A + 2^(k - 1) + Qb 2^k) 2^lift(k) + add_h < 2^60,     add_0 = 0,  add_1 = 2^16 (Ty + 128),  add_2 = 2^16 (Ty2 + 32896)

| nm | for each layer, k | h | A | SP | Qb |
|---|---|---|---|---|---|
| `gate` | kg; and kgR in a layer with routed experts | hgD | acc | 2^16; with `gcl 1` 2 Qb + 1 | 2^15; with `gcl 1` 128 + 2^22 |
| `up` | ku; and kuR in a layer with routed experts | huD | acc | 2^16 | 2^15 |
| `norm` | K1; K2 | 2 | 2^M1 + 2^(lsg - 1); 2^M2 + 2^(lsg - 1) | 2^16 | 2^15 |
| `down` | kd | hdD | f x 4096 x 2^15 x BW | 2^(lsg + 2) | 2^(lsg + 1) |
| `qk` | kq; kk | hq | 2 x 2^12 x acc | 2^16 | 2^15 |
| `o` | ko | ho | dq BC BW | 2^(lsg + 2) | 2^(lsg + 1) |
| `v` | kv | hv | acc | 2^16 | 2^15 |
| `lr` | kr, in a layer with routed experts | hr | acc | 2^15; with `rcl 1` 16 Ty2 + 15 | 2^14; with `rcl 1` 8 Ty2 + 7 |

In `down`, 4096 is 1 + 4095 (`SMAXTOP`): 4095 is the bound on the gate function's values if the
line `smax` is absent, and every value of the line is smaller. 2^15 is the bound on `up`.

| name | when | condition |
|---|---|---|
| `guard-radix-clamp` | each layer with routed experts, with `rcl 1` | floor((acc + 2^(kr - 1)) / 2^(kr + 3)) + 1 + 2^15 + 2^16 < 2^25 |
| `clamp-keys` | the same | 32 Ty < 2^60 and (Ty2 + 2^15) + 2^25 (Ty2 + Ty) + 2^39 < 2^60 |
| `U-xr` | with routed experts | (2 Ty2 + 1) 2^16 <= 2^60 |
| `W-xr` | with routed experts | 2^16 BC + 2^15 + 2^16 Ty2 < 2^60 |
| `guard-radix-FS` | with routed experts | 2^15 + 2^14 + 2^17 Ty3 < 2^59 |
| `guard-radix-gate-clamp` | with `gcl 1` | (Ty2 + 49152) + 2^28 (Ty + 128) + 2^40 < 2^59 and Ty2 + 49152 + 65536 < 2^28 |
| `guard-flag-gate-clamp` | with `gcl 1` | 2^40 + 49152 > Ty2 + 32767 + 2^33 |
| `MR` | with routed experts | NE Ty Ty2 + Ty2 + NE Ty < 2^60 |
| `one-hot` | with routed experts | NE Ty + 1 < 2^60 |
| `linear` | always | 2^16 (Ty2 + 32896) < 2^60 |
| `guard-radix` | always | (Ty2 + 32768) + 2^28 (Ty2 + 4096) < 2^59 and Ty2 + 32768 + 65536 < 2^28 |
| `ledger` | always | XL d Ty^2 < 2^53 |
| `ledger-eps` | with five limbs | `eps1`, `eps2` and the final norm's epsF (11.4) are below 2^48 |

The condition `no-wrap-pin-forest` of the lookup is in 11.1. A verifier also rejects
(`policy-weight-kind`) unless each of the twelve lines `wb_qk` to `wb_r` of a kind that the model
has is 13. BW = 2^(13 - 1) comes from this number. It rejects (`header-absent-kind`) unless the line
of a kind that the model lacks is 0. The bound on the weights is not shown by a request's proof: it
is what a key file's item `ranges` is about (11.5).

##### What the optional lines change

| line | cells | calls' segments | claims and scalars |
|---|---|---|---|
| `xl 5` (absent: four limbs) | the streams have 5 limbs instead of 4; a norm's row has 99 cells instead of 74 | ROW has 198 keys for each layer and token instead of 148, with other multipliers | NS = 9, NPr = 7, NO = 4 instead of 7, 3, 3; group 6 has 8 lines for each norm instead of 5 |
| `gcl 1` | `g0`, `gc` instead of `gi`, in the same features; also `gh`, at the end of af | also G0 and CG; the quotient in G and the argument in F change; the table FS has two halves, and MR's constant is 2^14 | none |
| `rcl 1` | also `ly` and `l3`, at the end of ar | also L3 and CLR; the quotient in LR changes | none |
| `eg`, `smax` | none | the rows of the gate function's table (11.1) | none |
| `NE 0` | no `xr`, `lr`, `rl`, `sel`, `sc` | none of the calls' segments marked R | no steps 13 and 14, no af@Zb, no groups 9 and 10; LamM counts as 0 |
| a digit form 0 | the site's tensor rh is absent | its high call's segment (QKH, GH, UH, OH, DH, VH, LH) is absent | none |

##### Indices beyond the true width

Several index ranges are rounded up to a power of two. This is what the forms do there; this page
gives no argument from it.

1. *Features of a matrix that belong to no tensor* are in the parts of `U3` like all others. No form
   of this part names their entries.
2. *Step 6.* The level's index is (i Pd + j) T + t. Claim 3 of group 2 has terms for j < d alone, so
   it takes the squared vector as 0 for d <= j < Pd.
3. *Steps 12 and 14.* In the same way the index of step 12 has room for Pf features, and the claims
   of group 8 have terms for the features below f alone. In step 14 fe is a power of two.
4. *Step 11.* A panel's entries with ri >= ni or yv >= ny, and the indices after the last panel, get
   no term in As, Bs, Alin or Blin.
5. *Key columns* that no call's segment takes, and the columns of a call's segment marked R in a
   layer with an even lam, count as padding keys. Their number in each tree, npad(y), enters the
   lookup's factor a_pad and its count of keys (11.1).
6. *Weights.* AH is 0 for the head numbers from H + G on. The other factors give a weight to the
   true indices alone (11.5).

##### Easy to get wrong

1. The claims of groups 8 and 9 have plus signs; claim 3 of group 2 has a minus sign.
2. Group 2 reads single limbs of the streams, each under its power of gam. Every other read of a
   stream joins the limbs with the powers of 256.
3. A tensor of af, qa, xbI or xbO is read through a long part, whose entries are numbered by blocks
   of 2^ls features. A tensor of ad or ar is read through a short part, whose entries are numbered
   by features. qa has both: `q`, `kx` and `v` are read at Zk through the long part, `o` at zt
   through the short one.
4. The norms' rows are read at four points: at zt for the keys of ROW, at PV for groups 5 and 6, at
   rtV for group 4 and at rt for group 7.
5. The D panel of the first pass has its weights e_c wt_ri on the side of Bs. All other panels have
   them on the side of As and Alin.
6. The numbers of the weight claims follow the list of the panels before it is sorted. The offsets
   follow the sorted list.
7. The panel of QK takes lift(kq) for queries and keys alike. LINE_QK takes kq for x < dq and kk for
   x >= dq.
8. In a layer with routed experts the first-pass panels of G, U and D are about the shared expert:
   the units 5, 6 and 9, with the shifts kg, ku and kd. The routed expert has the shifts kgR and
   kuR, and kd again.
9. KT is a form, not a number. It brings Cl_9 and mask entries into C_11, and C_11 also has the
   term - Cl_9 of its own.
10. The constant 128 of a digit D_j needs no weight, because the weights eq(z, t) of a short part
    add up to 1. In LINE_ROW it is multiplied by mul_j ew[x].

##### What this half does not specify

It does not define which limbs a prover writes for a value that can be written in limbs in more than
one way, or what is in the features that belong to no tensor. It does not define Left, the lookup's
own claims, the claims on the planes or the smallness test's claims: these are 11.1's. It does not
define how a model's tensors become the committed integers Wm_X (11.5). It gives no argument that
the claims imply the relations of "What the segment computes".
