*Part of [the file format](FORMAT.md), [section 11](11-sub-proofs.md). Rule numbers and terms are
those of that page.*

### 11.4 The output head

This part is written from the programs' source. It lists, in order, every round of the sub-proof of
the last segment of a request's first chain, the output head, with what the prover sends and what is
sampled. The kinds of step are those of 11.1, the frame's rounds of a segment are R14.4, and the
next token's round is R14.8.

*What the segment is about.* The segment takes the residual stream that leaves the last group of
layers. It is about the final norm at every position, the product of the normalized stream with the
head's weights, and the relation between the resulting logits and the request's tokens that the
acceptance policy states. The request is T tokens and two positions r and e with r < e <= T, and the
next token is one token y. Position t is a checked position if r <= t < e (12.2). The token that it
is about is token t + 1 of the request, a checked token, or at position T - 1 the next token y. A
checked position with t + 1 < T has a **cost** (12.5); the policy's lines `C`, `theta` and `cap`
bound the costs. If position T - 1 is checked, y must be the index of the largest logit at that
position, the lowest index on ties. If e < T, no checked position is about y, and y is 0. The
segment's claims on the head's weights are merged into one row on the stored table (11.1, the merge;
11.5). The segment's word in the item `links' identifiers` (R14.3) is 1 + NG, where NG = L / Lam is
the number of groups of Lam = 2^ll layers (11.2; 13.4, step 5).

*Sizes.* lt, d, la, ls, lPd, XL, NS, NPr, NO, NR and lkW are those of 11.2. V is the number of the
parameter file's line `V`, the number of vocabulary entries. S2 = 2^ls. JV is V rounded up to a
multiple of S2, and JVa = 2^(w - 1), where w is the least number with 2^w >= JV.

*The arrays that the segment commits to.* The names are those of the item `layout` (R14.5). Matrix
number mu of the segment's list is committed in packed form (11.1) as the array `M` followed by mu.

| mu | matrix | cells | rows of cells |
|---|---|---|---|
| 0 .. NHf - 1 | hf, in NHf parts | the final norm's wide output and remainder (2 limbs x d each); for each vocabulary entry two digits of a margin (2 limbs and 1 limb x JV) and a value of the table of exponentials (3 cells x JV) | 2^lmhf each |
| then 1 | hd | the final norm's clamped output (2 limbs x d) | 2^lmhd |
| then 1 | hr | the final norm's row of digits and 21 further digits for each position | 2^lmhr |
| then 1 | bH | the planes of the lookup's multiplicities | 2^lmbH |
| last | xbL | the residual stream that enters the head | 2^lmx |

*The columns.* The columns follow as in 11.2, with n = 4 d + 6 JV for hf, 2 d for hd and NR + 21 for
hr. bH has NC 2^(kT + 5 - lmbH) columns.

| moment | arrays, in the order of the layout's numbering |
|---|---|
| 0 | hf, hd, hr, bH, by increasing mu. No other segment reads them |
| 1 | `U3`: one vector over E (11.1) |
| 2 | `ZZ`, `BZ`: the smallness test's digits and the planes of its lookup |

xbL is not committed by this segment: it is the array of xbO of the last feed-forward segment
(11.2), which the frame keeps up to this segment.

*The next token.* The frame receives the next token as one scalar that is not masked, in the round
`REPLY` (the name means the next token), directly before the segment's first `SEND seed` (R14.8). It
has no number among the sub-proof's scalars. The sub-proof rejects the proof unless the scalar's
second and third coordinates are 0 and its first is below V. The first coordinate is y, and the
sub-proof uses it as the token of position T - 1, if that position is checked. The comparison with
the next token that the client was given is the frame's (R14.8), and how a token becomes an element
is R15.7.

*The rounds, in order.* A row in italics is the frame's.

| no. | step | rounds | scalars sent | sampled |
|---|---|---|---|---|
| | *the next token* | *`REPLY` (R14.8)* | *1, not masked* | *nothing* |
| | *moment 0* | *`SEND seed`, `LOC` (R14.4)* | | |
| 1 | the lookup of the head's call list | 11.1, steps 1 to 9 | as for a lookup, where v' and d' are its v and dL | 11.1 |
| 2 | the norm's token point | `ZERO` | 0 | lt elements |
| 3 | the sums of limb products, hidden values sent alone; their batching, n = NS | `BAT-sigma` | NS | 1 element |
| 4 | the sum of squares: a product level with lPd + lt variables, default order | `PSC epsilon`, `PSC round` .., `PSC Y`, `PSC values`, `LINE` | 2 (lPd + lt) + 5 | as for a product level |
| 5 | the row products, hidden values sent alone; their batching, n = NPr | `BAT-sigma` | NPr | 1 element |
| 6 | the first token sumcheck, at the point of step 2: d_Y = 2, NO end values | `PSC epsilon`, lt times `PSC round`, `PSC Y`, `PSC values` | 1 + (1 + 2 lt + 2 + NO) | eps, lt challenges, beta |
| 7 | the second token sumcheck (the stream times the reciprocal), at the last lt coordinates of the lookup's key point: d_Y = 2, 2 end values | as step 6 | 1 + (1 + 2 lt + 2 + 2) | eps, lt challenges, beta |
| 8 | the third token sumcheck (the activations times the head's weights), at the same lt coordinates, with lPd further equality rounds at the constant 1/2: d_Y = 2, 2 end values | `PSC epsilon`, lt + lPd times `PSC round`, `PSC Y`, `PSC values` | 1 + (1 + 2 (lt + lPd) + 2 + 2) | eps, lt + lPd challenges, beta |
| | *moment 1: `U3`* | *`SEND seed`, `SEND answer` (R14.4 (d))* | | |
| 9 | the smallness test's matrix | `RNG` | 0 | (the number of columns of all matrices of the list, xbL among them) x tau entries |
| | *moment 2: `ZZ`, `BZ`* | *`SEND seed`, `SEND answer`* | | |
| 10 | the smallness test's lookup on the digits | 11.1, steps 1 to 9, with NC = 2 and (v'', d'') = (v_Z, 11) of 11.1 | as for a lookup | as for a lookup |
| 11 | the merge, n = 1 | `BAT-sigma`, `PSC epsilon`, lkW times `PSC round`, `PSC Y`, `PSC values` | 2 lkW + 3 | 1 element; eps, lkW challenges, beta |
| 12 | the batching of all claims and equations, n = NCm + Q | `BAT-sigma` | 0 | 1 element chi |
| | *the segment is closed* | *`BAT`, `FOLD epsilon`, `FOLD round` .. (R14.4)* | | |

*In which message the scalars are.* The NS scalars of step 3 and the NPr scalars of step 5 are the
prover's message of the step's `BAT-sigma`. The claimed sum of steps 6, 7 and 8 is in the message of
the step's `PSC epsilon`, before the hiding polynomial's sum. The rounds of steps 2, 9 and 12, and
the `BAT-sigma` of step 11, have an empty prover's message.

*The rounds of step 8.* The first lt rounds bind the bits of the position, the highest first.
Further round number i, from 0, binds bit lPd - 1 - i of the feature's index. Because the features
are summed plainly through equality rounds at 1/2, the claimed sum that is sent is 2^(-lPd) times
the sum that the lookup's join needs.

*The end values, in the order sent.* Step 6: the NO row values of the final norm. Step 7: the
reciprocal; the weighted stream. Step 8: the activations' value; then the value of the one weight
claim, which is the value that the merge starts from.

*The claims, in order.* Claim number c gets chi^c (11.1, with its exception). NCm is the sum of the
counts.

| no. | claims | count |
|---|---|---|
| 1 | of the lookup of step 1, on mask entries alone: the tie, the key levels, the table levels, the bits | 31 + v' + d' + 1 |
| 2 | of step 4: the level's own; its two end values are equal; what the first is | 3 |
| 3 | the claimed sum of step 6 is the batch of the row products; step 6's own | 2 |
| 4 | the end values of step 6 are what `U3` gives, in the order sent | NO |
| 5 | the carries of the sums of limb products | NS |
| 6 | the lines of the norm's row of digits: 8 with five limbs, 5 with four | 8 or 5 |
| 7 | step 7's own; what its two end values are | 3 |
| 8 | the join of the lookup of step 1 | 1 |
| 9 | the acceptance policy's four equations: the sum of the costs; the number of positions above the clip, which is the line `C`; a cost is at most the clip; the table values of a position add up to at most 2^F, with F the line `F` | 4 |
| 10 | of the lookup of step 1, on its planes: the leaves; the planes at the bits' point; the number of keys | 3 |
| 11 | of the smallness test, in the order of 11.1, with (v'', d'') = (v_Z, 11) of 11.1 | 36 + v'' + d'' + NM + NX |
| 12 | step 8's own; what its first end value is | 2 |
| 13 | the merge's own | 1 |

NM = NHf + 4. NX = 2 NHf 2^(lmhf - la) + 2 x 2^(lmhd - lt) + 6 x 2^(lmhr - lt) + 3 + 2 x 2^(lmx -
la). Q is formed as in 11.2.

*The parts of `U3`, in order.* Long and short parts are those of 11.2. phiC is the vector that has
the equality weights of the lookup's token point at the positions that have a cost, and 0 elsewhere.

| no. | matrix | long or short | weight vector or point |
|---|---|---|---|
| 1 | hf | long | the last la coordinates of the lookup's key point |
| 2 | hf | long | 1 on a block's S2 features, phiC on the positions |
| 3 | xbL | long | the last la coordinates of step 4's point |
| 4 | xbL | long | the ls coordinates before the last lt of the lookup's key point, then step 7's challenges |
| 5 | hd | short | the lookup's token point |
| 6 | hd | short | the first lt challenges of step 8 |
| 7 to 10 | hr | short | the lookup's token point; the point of step 2; step 6's challenges; step 7's challenges |
| 11 | hr | short | phiC |
| 12 | hr | short | 1 at the positions that have a cost, 0 elsewhere |
| 13 to 15 | bH | 2^lmbH entries | the three of a lookup's planes |

So the request's r and e enter three weight vectors. They change no length and no order.

*The call list.* A line `callH NAME KIND SEGMENT ..` names a call. The kinds are R8, R16, FC (11.2)
and FE, the table of exponentials. The calls' segments (11.2) are NF and NFH (R16, d keys for each
position), HD0 and HD1 (R16, JV), ROW (R8, NR + 21), HD2 (R8, JV), CNF (FC, d), HE0 (FE, JVa) and
HE1 (FE, JV - JVa). A verifier rejects a list with more than 6 calls and applies the other tests of
11.2, but not the one on the names V, a, b and E. Without such lines the list is (NF NFH HD0),
(HD1), (ROW HD2), (HE0), (HE1), (CNF); R15.6 has these six lines whole, with the calls' names and
kinds, which are hashed into the statement. Call c has n_c = lJ + lt key bits, with lJ formed as in
11.2 and at least nminH - lt, or nmin - lt if the line `nminH` is absent. Each line `iotaH_0` to
`iotaH_5` must equal n_c - dT of its call, and 0 if the list has no call of that number (second
half, the tests on the public numbers).

*The table of exponentials.* Row u, for u = 0 to 65535, is u + 2^16 EX[u]. It is made with integers
alone: g_0 = 2^(F + 17); EX[u] = ceil(g_u / 2^17); g_(u+1) = ceil(g_u N / 2^32). F is the line `F`,
which is 24; sh and sT are the lines of these names. N is 484 if sT + 4 = sh. Otherwise sh + 4 <= sT
<= sh + 10, and N is entry number sT - sh - 4 of the list 4034748383, 4162825045, 4228380000,
4261543596, 4278222806, 4286586875, 4290775040. The padding key of a call of kind FE is row 0. With
N = 484 the values EX[u] are 2^F, 2, 1, 1, ..: this table is the **step table**. Where the second
half sets "a table of exponentials" against the step table, it means a table with sT >= sh + 4. A
whole call takes the step table alone: a verifier rejects a parameter file in which sT + 4 is not
sh, as `cp_verify` does in every whole call that reaches this comparison, before its stages, with
the name `policy` (12.5); 13.8, in the second row `policy` of the table "Before the stages", says
which whole calls do not reach it or leave it without a result: each of them is rejected under
another name, by an earlier test or, by the source, by a stage. The other seven tables are given
because the stages' tests and formulas have them.

*What the verifier takes from the public files and the call.* From the parameter file: MF, KF and
epsF of the final norm; `toksrc`, which must be 1; the matrices' sizes lmhf, NHf, lmhd, lmhr, lmbH;
the policy's lines `C`, `theta`, `cap`, `s`, `sh`, `sT`, `F`; the call lines. From the call: the
request's T + 2 words and the next token's word. A verifier rejects unless every token and y are
below V, r < e <= T, and y = 0 where e < T. It rejects the parameter file, before it reads a proof,
unless `theta` <= `C`, `cap` <= T, `cap` < 2^16, `F` = 24, `s` <= 63, `sh` is 7 or 8, `sT` is as
above, and 2 V <= 2^F. It also rejects the file unless the other tests on these numbers hold, the
ranges and the no-overflow conditions among them: they are in the second half, and section 13 has
their names.

*A worked example.* With lt = 10, d = 5120, V = 202048, la = 16, `lidb 28`, `nminH 27`, `xl 5`, lkW
= 18 and the six calls (NF NFH HD0), (HD1), (ROW HD2), (HE0), (HE1), (CNF): JV = 202048 and JVa =
131072; the calls have 28, 28, 28, 27, 27, 27 key bits, so dT = 27, NT = 9, v' = 31, d' = 19. With
tau = 128, lmhf = 16, NHf = 4, lmhd = 14, lmhr = 13, lmbH = 15 and lmx = 16: mz = 385,024, so v''
= 27. The scalars are: step 1: 1,995; step 3: 9; step 4: 51; step 5: 7; step 6: 28; step 7: 26; step
8: 52; step 10: 1,363; step 11: 39. These are the values of `target_full_f2/policy.txt`, the
full-size example configuration (section 8).

*Easy to get wrong.*

1. The merge (step 11) comes after the smallness test, directly before the last round. In a
   feed-forward segment it comes before the root of `U3` (11.2).
2. The merge has one claim, and its `BAT-sigma` is a round like any other: an element is sampled and
   only its power 0 is used.
3. The next token is not a scalar of the sub-proof. It has a round of its own before the segment's
   first message, has no number among the scalars and has no mask.
4. Step 8 is one sumcheck with lt + lPd rounds, not two sumchecks. All its rounds are equality
   rounds, so the constant coefficient is the one that is not sent.
5. Here the lookup's join is a claim of its own (number 8). In a feed-forward segment it is inside a
   start form.
6. The columns of the entries v with V <= v < JV are padding, and so are the columns that no call's
   segment takes. The number of padding keys does not depend on the request.

*What this part does not specify.* This first half gives the rounds, the counts and the orders; the
second half gives what each cell, each scalar and each entry of `U3` holds, the key of every call's
segment, every claim as an equation, the weight claim and the no-overflow conditions. The two halves
do not define the honest prover's rule for a position's cost and its hidden numbers, or its choice
of digits and limbs where several fit: a verifier needs none of them. They give no argument that the
claims imply the integer function, that the cells beyond the true sizes do no harm, or that an
equation in the field holds in the integers.

#### 11.4, second half: cells, keys and claims

This part is written from the programs' source. The first half of 11.4 gives the rounds, the counts
and the orders. This half gives what the numbers are: the integers that the head computes; what each
cell, each scalar and each entry of `U3` holds; the key of every lookup call; every claim as an
equation, in the order of the claims; the weight claim; and the no-overflow conditions. It needs
from 11.1: a hidden value and the scalar as sent; a form; the claim that a sumcheck leaves; the
lookup's own claims, its join and its claims on the planes; the smallness test and an entry of `U3`.
It needs from 11.5 the factors of a claim on the weights.

*Public and hidden.* **Public** is what a verifier computes from the public files, the request, the
bytes of the proof and the sampled values: every size, point, weight vector, table and coefficient
below. **Hidden** are the cells, the entries of `U3` and the hidden values of the scalars. A
verifier never computes anything hidden. It forms each claim as a form: a public constant, plus
public coefficients on hidden values of scalars and on entries of `U3` (11.1). All arithmetic of the
claims is in E; an integer in a claim stands for its image in the field, and a negative integer -n
for the field's -n.

##### Notation of this half

| letter | meaning | |
|---|---|---|
| T, lt, d, V, la, ls, S2, lPd, XL, NS, NPr, NO, NR, lkW, JV, JVa, NHf, lmhf, lmhd, lmhr, lmbH, lmx, NC, dT, v', d' | as in the first half, 11.1 and 11.2 | public |
| NRH | NR + 21: the digits of the head's row (120 with five limbs, 95 with four) | public |
| `MF`, `KF`; epsF | the numbers of the final norm's lines of these names, always in code marks (KF_c is another thing: a key form); epsF = `epsF_lo` + 2^32 `epsF_hi` | public |
| `C`, `theta`, `cap`, `s`, `sh`, `sT`, `F` | the numbers of the acceptance policy's lines of these names; always in code marks, also inside a formula; `C` is also called the clip | public |
| BIG | 2^37 (the source's `BIGH`) | public |
| offE | 128 (2^24 - 1) / 255 + 1 = 8,421,505 | public |
| r, e, y | the request's two positions and the next token (first half) | public |
| on_t | 1 if r <= t < e, else 0: position t is a checked position (12.2) | public |
| co_t | 1 if on_t = 1 and t + 1 < T, else 0: position t has a cost | public |
| RC | the number of positions with a cost: min(e, T - 1) - r | public |
| win_t | the token that a position with on_t = 1 is about: token t + 1 of the request, a checked token, if t + 1 < T; at t = T - 1 the next token, the y that is sent in the round `REPLY` | public |
| a real pair | a pair (v, t) of a vocabulary entry v < V and a position t with on_t = 1 | public |
| [..] | 1 if the statement in the brackets is true, else 0 | |
| eq(z, i) | the equality weight of the index i at the point z, the coordinate of the highest bit first (11.1, the order of bits) | public |
| X and X' | for a scalar with the name X and the number j: X is its hidden value, hidden(j) of 11.1, and X' the scalar as sent, shown(j). The prime comes after the letters and before the index: EV'_0 | X hidden, X' public |
| C_n; Left(C_n; S0; 0) | the start form of the sumcheck of step n; the form that this sumcheck leaves (11.1), made with the scalars and sampled values of step n, where S0 is the summand on the end values as sent. No sumcheck of the head's own has a kept form (the lookup's leaves have one, 11.1) | a form |
| KF_c, npad(y), TB_0, TB_1, TP, LJ | what the lookup of 11.1 takes from its host: the key form of call c, the number of padding keys of tree y, the three column sums of the planes' matrix; LJ is the join. y in npad(y) and in "tree y" is 11.1's letter for a tree's number, not the next token | |
| lb | the number of a limb, from 0 | |
| cell_M(f, t) | the cell of feature f and position t of the matrix M (hf, hd, hr, xbL) | hidden |
| part_n[i] | entry number i, from 0, of part number n of `U3`, n = 1 .. 15 as in the first half's table of the parts. A part on hf counts its NHf matrices as one run; 11.1 counts them as NHf parts | hidden |
| str, Vsq, sig, R, aw, rh, rl, act, lam, Mg, K_0, K_1, K_2, d1, d2, ex, Wh, UH | the head's integers (below) | hidden |
| x | the number of an entry of a call's segment, from 0 | public |
| gc_t, gb_t, gu_t, gk_t, gz_t, gs_t | the six hidden numbers of a position (the source's c, b, u, k, z, zs) | hidden |
| D_j, mul_j, Vlo, Vhi, c_s, a, b, s0, l1, h1, Blo, Bhi, kap, Rlo, Rhi, q1, p1, rr, rq | the digits, their multipliers and the fields of the norm's row, with the letters of 11.2 | hidden; mul_j public |
| S0 and s0; SH and `sh` | they differ by capitals alone: S0 is a sumcheck's summand and s0 the lowest bit of sig; SH is a constant of the no-overflow conditions and `sh` a line of the policy | |

The steps are those of the first half's table of the rounds, and the groups of claims those of its
table of the claims.

##### What the head computes, in integers

All of this is hidden. t is a position below T, k a feature below d, v a vocabulary entry below V.
floor rounds toward minus infinity.

1. *The stream.* str[t][k] is the value of the residual stream that leaves the last group of layers.
   It comes in XL limbs at the radix 256: str[t][k] = the sum over lb < XL of 256^lb cell_xbL(lb d +
   k, t).
2. *The final norm.* Vsq_t = the sum over k < d of str[t][k]^2. sig_t is the largest integer whose
   square is at most Vsq_t + epsF. R_t = floor((2^`MF` + floor(sig_t / 2)) / sig_t).
3. *The wide output.* aw[t][k] = floor((str[t][k] R_t + 2^(`KF` - 1)) / 2^`KF`). The remainder
   str[t][k] R_t + 2^(`KF` - 1) - 2^`KF` aw[t][k] is 2^16 rh[t][k] + rl[t][k], with 0 <= rl < 2^16
   and 0 <= rh < 2^(`KF` - 16).
4. *The activations.* act[t][k] = aw[t][k] clamped to -2047 .. 2047. The clamp's table has an
   argument only for -32768 <= aw <= 32767.
5. *The logits.* lam_t[v] = the sum over k < d of act[t][k] Wh[k][v]. Wh[k][v] is the committed
   weight of unit 13 that has the index v in the group named x and the index k in the group named kk
   (11.5).
6. *The margins.* At a real pair (v, t) the **margin** (the source's `D`) is Mg[t][v] =
   lam_t[win_t] + co_t (gc_t + BIG gb_t) - lam_t[v] - [v < win_t] - Pub[t][v]. Pub[t][v] is 0 in
   this format. The **logit gap** of a position is the gap of 12.5: the largest, over v < V, of
   lam_t[v] - lam_t[win_t] + [v < win_t]. It is how far the token's logit is below the position's
   largest, plus 1 where that largest is reached at an index below win_t.
7. *The three digits of a margin.* Mg[t][v] = K_2[t][v] + 2^`sh` K_1[t][v] + 2^(`sh` + 15)
   K_0[t][v], with 0 <= K_2 < 2^8, 0 <= K_1 < 2^16 and 0 <= K_0 < 2^16. K_2 is the low digit, K_1
   the middle digit, K_0 the top digit. The weights of K_1 and K_0 overlap in one bit, and with `sh`
   = 7 those of K_2 and K_1 too, so the digits of a margin are not unique; the prover chooses them.
   At a position with a cost, equation (Z) below can hold only if K_1 is at least 1 at every v < V.
8. *The table values.* ex[t][v] = EX[K_1[t][v]], with the table EX of the first half.
9. *The six hidden numbers* of a position with a cost are integers that are not negative. They stand
   for the position's cost (12.5): by the source's comment an honest prover has either gc_t <= `C`
   with gb_t = 0, or gc_t = 0 with gb_t = 1, and this page does not define its rule further (the end
   of this half). By the argument of the source's comments (below), the claims enforce inequalities
   alone: gc_t + BIG gb_t is at least the cost, gc_t is at most `C`, and the two sums (B2) and (B3)
   hold. So gb_t is above 0 wherever the cost is above `C`. But any gc_t from the cost up to `C`
   satisfies the claims as well, and so does a gb_t above 0 at a position whose cost is at most `C`.
   gk_t = `C` - gc_t. gs_t = 2^`F` - the sum over v < V of ex[t][v]. gu_t and gz_t fill the two sums
   (B2) and (B3) below up to their public values.

The embedding of the request's tokens is not in this segment: it is in the feed-forward segment of
group 0 (11.2). No gain vector of the norm appears: this page does not define how a model's tensors
become the committed integers (11.5).

*At a pair (v, t) that is not real* (v >= V, or on_t = 0) there is no margin. The cells of its two
lower digits hold 0, and its table value is EX[0] = 2^`F` (the keys, below).

*The next token.* Position T - 1 never has a cost. If on_(T-1) = 1, that is e = T, its margins are
Mg[T - 1][v] = lam_(T-1)[y] - lam_(T-1)[v] - [v < y], with the y of the round `REPLY` (first half).
The test on the scalar's form has the rejection name `header-y` (section 13). If e < T no formula of
this half reads y.

##### Where a cell is

A matrix holds n features for each position (the first half, the columns). Take a matrix that is cut
into np matrices of the segment's list, each with 2^lm cell rows and k_mu columns. Let i = f T + t.
The cell of feature f and position t is in matrix number floor(i / (k_mu 2^lm)) of the np, in column
floor(i / 2^lm) mod k_mu, in cell row i mod 2^lm. So a column holds 2^(lm - lt) whole features, one
after the other, each with its T positions in order. np is NHf for hf and 1 for hd, hr and xbL. A
feature from n on, up to the last column's end, has cells that no claim but the smallness test's
reads.

| matrix | feature | the cell holds | so that |
|---|---|---|---|
| xbL | lb d + k, lb < XL | limb lb of str[t][k] | str = the sum of 256^lb times limb lb |
| hf | lb d + k, lb < 2 | limb lb of aw[t][k] | aw = limb 0 + 256 limb 1 |
| hf | 2 d + lb d + k, lb < 2 | limb lb of rh[t][k] - 32896 | rh = 32896 + limb 0 + 256 limb 1 |
| hf | 4 d + lb JV + v, lb < 2, v < JV | limb lb of d1[t][v] | K_1 = 32896 + d1 at a real pair; d1 = limb 0 + 256 limb 1 |
| hf | 4 d + 2 JV + v, v < JV | d2[t][v] | K_2 = 128 + d2 at a real pair |
| hf | 4 d + 3 JV + lb JV + v, lb < 3, v < JV | limb lb of ex[t][v] - offE | ex = offE + limb 0 + 256 limb 1 + 65536 limb 2, at every pair (v, t) |
| hd | lb d + k, lb < 2 | limb lb of act[t][k] | act = limb 0 + 256 limb 1 |
| hr | j < NRH | digit j of the position's row, minus 128 | D_j = 128 + the cell |
| bH | | the planes of the lookup's multiplicities (11.1) | |

32896 = 128 (1 + 256) and offE - 1 = 128 (1 + 256 + 65536): a limb of rh, of K_1, of K_2 and of ex -
1 is a byte minus 128. The source's names for the first features of hf are `oAw` = 0, `oRa` = 2 d,
`oD1` = 4 d, `oD2` = 4 d + 2 JV and `oE` = 4 d + 3 JV. A claim reads only the sums in the last
column of this table. What is meant to bound a single cell is the smallness test (11.1; a comment in
`verifier/src/cp_verify.cpp`, at `TyReq`) and, for hr, the lookup on ROW
(`verifier/src/vf_norm_digit_rows.h`).

##### The row of digits

Position t has one row of NRH digits D_0, .., D_(NRH - 1). The first NR are the final norm's, with
the same fields as a layer's norm (11.2); the last 21 are the six hidden numbers. dig(j, n) stands
for D_j + 256 D_(j+1) + .. + 256^(n-1) D_(j+n-1). The lookup on ROW has the key mul_j D_j in the
range of 8 bits, so D_j < 256 / mul_j, given a bound on the cell that keeps the key below p (by a
comment in `verifier/src/vf_norm_digit_rows.h`). mul_j is 1 except where the tables say otherwise.

*With five limbs* (NR = 99):

| digits | field | what an honest row holds |
|---|---|---|
| 0 .. 9 | Vlo = dig(0, 6), Vhi = dig(6, 4); mul_9 = 4 | the bytes of Vsq_t: Vsq = Vlo + 2^48 Vhi |
| 10 .. 57 | c_s = dig(10 + 6 (s - 1), 6) - 2^47 for s = 1 .. 8; c_0 = 0; c_9 = D_9 | the carry into byte s of the sum of 256^s W_s[t] |
| 58 .. 63 | a = 2 D_58 + D_59 + 2^8 D_60 + 2^16 D_61; b = D_62 + 2^8 D_63; s0 = D_59; sig = a + 2^24 b; mul_58 = 2, mul_59 = 128, mul_63 = 2 | sig_t, with s0 its lowest bit |
| 64 .. 68 | l1 = dig(64, 5) | Vsq + epsF - sig^2 |
| 69 .. 73 | h1 = dig(69, 5) | 2 sig - l1 |
| 74 .. 76 | Blo = dig(74, 3); mul_76 = 2 | (a b) mod 2^23 |
| 77, 78 | Bhi = dig(77, 2) | floor(a b / 2^23) |
| 79 | kap = D_79 - 4; mul_79 = 32 | the carry from the low 48 bits of line LE1 into LE2 |
| 80 .. 82 | Rlo = dig(80, 3) | R_t mod 2^24 |
| 83 .. 85 | Rhi = dig(83, 3); R = Rlo + 2^24 Rhi | floor(R_t / 2^24) |
| 86 .. 88 | q1 = dig(86, 3) | Rlo b + Rhi a |
| 89 .. 93 | rr = dig(89, 5) | 2^`MF` + floor(sig / 2) - R sig |
| 94 .. 98 | rq = dig(94, 5) | sig - 1 - rr |

*With four limbs* (NR = 74):

| digits | field | what an honest row holds |
|---|---|---|
| 0 .. 7 | Vsq = dig(0, 8); mul_7 = 32 | Vsq_t |
| 8 .. 43 | c_s = dig(8 + 6 (s - 1), 6) - 2^47 for s = 1 .. 6; c_0 = 0; c_7 = D_7 | the carries |
| 44 .. 48 | sig = 2 D_44 + D_45 + 2^8 D_46 + 2^16 D_47 + 2^24 D_48; s0 = D_45; mul_44 = 2, mul_45 = 128, mul_48 = 4 | sig_t |
| 49 .. 52; 53 .. 56 | l1 = dig(49, 4); h1 = dig(53, 4) | Vsq + epsF - sig^2; 2 sig - l1 |
| 57 .. 59; 60 .. 62 | Rlo = dig(57, 3); Rhi = dig(60, 3); R = Rlo + 2^24 Rhi | R_t |
| 63 .. 65 | p1 = dig(63, 3) | Rhi sig |
| 66 .. 69; 70 .. 73 | rr = dig(66, 4); rq = dig(70, 4) | as with five limbs |

*The 21 further digits*, all with mul_j = 1:

| number i | hidden number | digits | |
|---|---|---|---|
| 0 | gc_t | NR + 0 .. NR + 4 | dig(NR, 5) |
| 1 | gb_t | NR + 5 | D_(NR+5) |
| 2 | gu_t | NR + 6 .. NR + 10 | dig(NR + 6, 5) |
| 3 | gk_t | NR + 11 .. NR + 15 | dig(NR + 11, 5) |
| 4 | gz_t | NR + 16, NR + 17 | dig(NR + 16, 2) |
| 5 | gs_t | NR + 18 .. NR + 20 | dig(NR + 18, 3) |

dg0_i and dgn_i stand for the first digit after NR and the number of digits of hidden number i:
(dg0_i, dgn_i) = (0, 5), (5, 1), (6, 5), (11, 5), (16, 2), (18, 3). At a position without a cost no
claim but the lookup on ROW and the smallness test reads these 21 cells.

*A row read through a part of `U3`.* The claims never read a row at one position. They read it under
a weight vector, through one of the parts 7 to 12. For n = 7, 8, 9, 10 the weights add up to 1, and
D_j(n) = part_n[j] + 128 is the form of digit j. A field with (n) after its name, as Rlo(9), is the
field's formula above on the forms D_j(n); its constants (2^47, 4) stay as they are.

##### Cells and indices beyond the true sizes

Several index ranges are rounded up. What is there, and what reads it:

1. *Features k from d to 2^lPd - 1.* They have no cell. Every sum over features in a claim runs over
   k < d (items 3, 18 and 26 of the claims, and the factor kk of the weight claim), so the
   multilinear extensions that steps 4 and 8 are about are 0 there.
2. *Vocabulary entries v from V to JV - 1.* They have cells in hf (d1, d2, ex). The calls' segments
   HD0, HD1, HD2, HE0 and HE1 read them, and their keys count as padding keys: an honest prover's
   cells there give the call's padding key (0 in d1 and d2, EX[0] in ex). Equation (Z) takes their
   table values out again as (JV - V) 2^`F`. The weight claim's factor AV has V entries and reads no
   weight for them.
3. *Feature numbers of a matrix after its last feature n - 1*, up to the end of the last column:
   only the smallness test reads these cells.
4. *Columns of a call that no call's segment takes.* No cell belongs to them: the key form puts the
   padding key in, with the weight padW_c.
5. *Positions with on_t = 0.* All cells exist and all keys are formed. The final norm, the wide
   output and the activations are tied there as at every position. The margins' cells are read
   without their offsets.

##### The scalars of the head's own steps

The lookups, the smallness test and the merge name their own scalars (11.1). The head's own steps
send these. A scalar's number follows from the order sent: j_n is the number of the first scalar of
step n; j_3 is the number of scalars of step 1, and m_4 = lPd + lt.

| name | step | a scalar's number | the hidden value is |
|---|---|---|---|
| W_s, s < NS | 3 | j_3 + s | the sum over t of eq(pV, t) W_s[t], where W_s[t] = the sum over k < d and over the pairs lb + lb' = s, lb, lb' < XL, of cell_xbL(lb d + k, t) cell_xbL(lb' d + k, t) |
| vL_4, vR_4 | 4, end values | j_4 + 2 m_4 + 3 and j_4 + 2 m_4 + 4, with j_4 = j_3 + NS | both: the value at the point rW of the multilinear extension of XG[k T + t] = the sum over lb < XL of gam^lb cell_xbL(lb d + k, t), which is 0 for d <= k < 2^lPd |
| P_i, i < NPr | 5 | j_5 + i, with j_5 = j_4 + 2 m_4 + 5 | the sum over t of eq(pV, t) times row product i at position t (below) |
| Cl_V | 6, the claimed sum | j_6 = j_5 + NPr | the sum over i < NPr of om^i P_i |
| EV_j, j < NO | 6, end values | j_6 + 2 lt + 4 + j | the sum over t of eq(rV, t) times a, b, Rlo, Rhi of position t (j = 0, 1, 2, 3); with four limbs sig, Rlo, Rhi (j = 0, 1, 2) |
| Cl_N | 7, the claimed sum | j_7 = j_6 + 2 lt + 4 + NO | the sum over t of eq(zt, t) R_t SX[t], where SX[t] = the sum over k < d of ew_NF[k] str[t][k] |
| EN_0, EN_1 | 7, end values | j_7 + 2 lt + 4 and j_7 + 2 lt + 5 | the sums over t of eq(rN, t) R_t and of eq(rN, t) SX[t] |
| Cl_H | 8, the claimed sum | j_8 = j_7 + 2 lt + 6 | 2^(-lPd) times the sum over t and over v < V of eq(zt, t) on_t ew_HD0[v] (lam_t[win_t] - lam_t[v]), which is 2^(-lPd) times the sum over t and k < d of eq(zt, t) act[t][k] UH[k][t] |
| EH_0, EH_1 | 8, end values | j_8 + 2 (lt + lPd) + 4 and j_8 + 2 (lt + lPd) + 5 | EH_0: the sum over k < d and t of eq(rsP, k) eq(rH, t) act[t][k]. EH_1: the same sum with UH[k][t] instead of act[t][k], where UH[k][t] = on_t (wx_HD0 Wh[k][win_t] - the sum over v < V of ew_HD0[v] Wh[k][v]); it is the value of the weight claim (below) |

The smallness test's lookup (step 10) begins at the number j_8 + 2 (lt + lPd) + 6, and the merge
(step 11) takes the last 2 lkW + 3 numbers. The row products are, with five limbs: P_0 = a^2, P_1 =
a b, P_2 = b^2, P_3 = Rlo a, P_4 = Rlo b, P_5 = Rhi a, P_6 = Rhi b. With four limbs: P_0 = sig^2,
P_1 = Rlo sig, P_2 = Rhi sig.

##### Points, batching elements and public weights

A point is written with the coordinate of the highest bit first (11.1). All of this is public.

| name | coordinates | what it is |
|---|---|---|
| zk_c = (zj_c, zt) | lJ_c + lt | the key point of call c (11.1). The key of column j and position t of call c has the index j T + t, so the first lJ_c coordinates, zj_c, belong to the column and the last lt, zt, to the position |
| zt | lt | the lookup's token point; the same for all calls, since dT >= la |
| zl | ls | the ls coordinates before the last lt of every key point; the same for all calls. (zl, zt) are the last la coordinates of the zx of 11.1 |
| pV | lt | step 2's elements, in the order sampled |
| gam | | the element of step 3's `BAT-sigma` |
| rW | lPd + lt | step 4's challenges, each as the coordinate of the bit that its round binds, which is the order of the rounds: the first lPd belong to the feature k, the last lt to the position. The equality point of step 4 is lPd times 1/2, then pV |
| om | | the element of step 5's `BAT-sigma` |
| rV; rN | lt; lt | the challenges of step 6; of step 7, in the order of the rounds |
| rH | lt | the first lt challenges of step 8, in the order of the rounds |
| rs_i, i < lPd; rsP | lPd | rs_i is the challenge of the round of step 8 that binds bit i of the feature: further round number n gives rs_(lPd - 1 - n). rsP = (rs_(lPd - 1), .., rs_0), which is these challenges in the order of the rounds |

The element of step 4's `LINE` is sampled and enters no formula.

*A call's columns.* pos_sg(x) is the column that entry x < n_sg of the call's segment sg takes in
its call. The rule is that of 11.2 (the tests on the list, and the key bits). The length n_sg is cut
into powers of two, the largest first, so that the first piece holds the entries from 0 on. The
pieces of all the call's segments are sorted by decreasing size. Pieces of equal size keep the order
in which they were cut, the call's segments in the order of the line. The sorted pieces take columns
one after the other from column 0, and inside a piece the entries keep their order. J_c = 2^(lJ_c)
is the number of the call's columns.

| name | for | value |
|---|---|---|
| ew_sg[x] | a call's segment sg of call c, x < n_sg | eq(zj_c, pos_sg(x)) |
| wsum_sg | | the sum of ew_sg[x] over all x < n_sg |
| wx_sg | HD0, HD1, HD2, HE0, HE1 | the sum of ew_sg[x] over the x whose vocabulary entry is below V. Entry x stands for the vocabulary entry x; in HE1 for JVa + x |
| ewh_sg[i] | a call's segment whose length is a multiple of S2, i < n_sg / S2 | eq(the first lJ_c - ls coordinates of zj_c, pos_sg(S2 i) / S2). So ew_sg[S2 i + j] = ewh_sg[i] eq(zl, j) for j < S2 |
| padW_c | call c | the sum of eq(zj_c, j) over the columns j < J_c that no call's segment takes |
| pad_c | call c | the padding key of call c, tab_c(t0_c) of 11.1: 0 for R8 and R16; 32768 + 2^39 for FC; 2^(16 + `F`) for FE |
| ON | | the sum of eq(zt, t) over the t with on_t = 1 |
| phiC[t] | t < T | eq(zt, t) if co_t = 1, else 0 |
| ERC | | the sum of phiC[t] over t |
| rhoC[t] | t < T | co_t |
| place_sg | HD0 | the sum over the t with on_t = 1 of eq(zt, t) times the sum of ew_sg[v] over v < win_t (the source's `placeTermH`) |
| pub_sg | HD0 | 0 (the source's `publicTermH`: the sum of ew_sg[v] eq(zt, t) Pub[t][v] over the real pairs) |
| iw | | the inverse of 2^(`sh` + 15) in the field |

*The padding keys of a tree.* The lookup needs npad(y) for each tree y. Call c has 2^(n_c - dT)
neighbouring trees (11.1). The i-th of them, from 0, is tree y = tree0(c) + i and holds the columns
j with floor(j / 2^(dT - lt)) = i. Its npad is T for each of these columns that no call's segment
takes, and T for each that a call's segment HD0, HD1, HD2, HE0 or HE1 takes for a vocabulary entry
from V on. A key at a position with on_t = 0 in the column of an entry below V is not padding. So
these numbers do not depend on the request.

##### The parts of `U3`

at_n is the number of entries of the parts before part n, and part_n[i] = U3[at_n + i], where U3[n]
is entry n of the vector (11.1). An entry of `U3` as a variable of a form, and the claims that tie
it to the cells, are in 11.1 (the smallness test). What an honest entry holds:

- a long part, with the weight vector phi of 2^la entries: part_n[i] = the sum over j < S2 and t < T
  of phi[j T + t] cell_M(S2 i + j, t). Entry i belongs to the **block** of the S2 features S2 i, ..,
  S2 i + S2 - 1.
- a short part, with the weight vector phi of T entries: part_n[i] = the sum over t < T of phi[t]
  cell_M(i, t). Entry i belongs to feature i.

This numbering agrees with 11.1's (entry c B + b of a matrix, the matrices of a cut matrix by
increasing mu) because of the order of the cells (above).

| n | matrix M | entries | phi |
|---|---|---|---|
| 1 | hf | NHf k_hf 2^(lmhf - la) | phi[j T + t] = eq(zl, j) eq(zt, t) |
| 2 | hf | as part 1 | phi[j T + t] = phiC[t] |
| 3 | xbL | k_x 2^(lmx - la) | phi[j T + t] = eq(rWl, j) eq(rWt, t), where rWt is the last lt coordinates of rW and rWl the ls coordinates before them |
| 4 | xbL | as part 3 | phi[j T + t] = eq(zl, j) eq(rN, t) |
| 5 | hd | k_hd 2^(lmhd - lt) | eq(zt, t) |
| 6 | hd | as part 5 | eq(rH, t) |
| 7 | hr | k_hr 2^(lmhr - lt) | eq(zt, t) |
| 8 | hr | as part 7 | eq(pV, t) |
| 9 | hr | as part 7 | eq(rV, t) |
| 10 | hr | as part 7 | eq(rN, t) |
| 11 | hr | as part 7 | phiC[t] |
| 12 | hr | as part 7 | rhoC[t] |
| 13, 14, 15 | bH | its number of columns, NC 2^(kT + 5 - lmbH), each | the three weight vectors of a lookup's planes, of 2^lmbH entries (11.1): phiB_0, at the leaves' point; phiB_1, at the bits' point; phiP, 2^(row mod 32) |

k_hf, k_hd, k_hr and k_x are the numbers of columns of one matrix of hf, of hd, of hr and of xbL
(the first half, the columns). `U3` has the sum of these numbers of entries and no more.

*Three short forms.* They are forms in entries of `U3`.

- SL(sg, f0, nl, fs) = the sum over i < n_sg / S2 and lb < nl of 256^lb ewh_sg[i] part_1[(f0 + lb
  fs) / S2 + i]. It is the sum, over the entries x of the call's segment sg and the positions t, of
  ew_sg[x] eq(zt, t) times the number that hf holds in nl limbs in the features f0 + lb fs + x.
  Every f0 and fs below is a multiple of S2.
- HN(n, i), for n = 11 or 12 and a hidden number i = 0 .. 5: the sum over lb < dgn_i of 256^lb
  (part_n[NR + dg0_i + lb] + 128 ws_n), with ws_11 = ERC and ws_12 = RC. It is the sum over t of
  phiC[t], or of rhoC[t], times hidden number i of position t.
- A2(n, k), for n = 5 or 6 and k < d: part_n[k] + 256 part_n[d + k]. It is act[.][k] under the
  part's weights.

##### The nine calls' segments: table, length and key

The key of a call's segment is an integer for each entry x and each position t; it must be a row of
the call's table. The tables are those of 11.1 (public tables) and of the first half (the table of
exponentials). k stands for a feature, v for a vocabulary entry, j for a digit's number.

| call's segment | kind: table | length n_sg | the key at (x, t) |
|---|---|---|---|
| NF | R16: the range of 16 bits | d | rl[t][k] = str[t][k] R_t + 2^(`KF` - 1) - 2^`KF` aw[t][k] - 2^16 rh[t][k] |
| NFH | R16 | d | 2^(32 - `KF`) rh[t][k] |
| HD0 | R16 | JV | K_0[t][v] = iw (Mg[t][v] - 2^`sh` K_1[t][v] - K_2[t][v]) at a real pair; - iw (2^`sh` d1[t][v] + d2[t][v]) at any other pair |
| HD1 | R16 | JV | K_1[t][v] = 32896 + d1[t][v] at a real pair; d1[t][v] at any other pair |
| ROW | R8: the range of 8 bits | NRH | mul_j D_j of position t |
| HD2 | R8 | JV | K_2[t][v] = 128 + d2[t][v] at a real pair; d2[t][v] at any other pair |
| CNF | FC: the clamp | d | (aw[t][k] + 32768) + 2^28 (act[t][k] + 2048) |
| HE0 | FE: the table of exponentials | JVa | the key of HD1 at (v, t), plus 2^16 ex[t][v], for v = x |
| HE1 | FE | JV - JVa | the same for v = JVa + x |

K_0 has no cell: its key is formed in the field, with iw. The keys of HD0, HD1 and HD2 in the
columns of the entries v >= V count as padding keys of their calls; the padding key there is 0.
Those of HE0 and HE1 count as padding keys too; the padding key there is 2^(16 + `F`). All of them
are read from the cells all the same.

##### The key forms

The lookup's join needs, for each call c, the key form KF_c: the form of the call's key vector at
its key point zk_c (11.1). KF_c = pad_c padW_c + [NF is in call c] Cl_N + the sum of KS_sg over the
call's segments sg of call c. The nine forms KS_sg are:

1. KS_NF = wsum_NF (2^(`KF` - 1) - 32896 2^16) - 2^`KF` SL(NF, 0, 2, d) - 2^16 SL(NF, 2 d, 2, d).
   The key's term str R is in Cl_N.
2. KS_NFH = 2^(32 - `KF`) (32896 wsum_NFH + SL(NFH, 2 d, 2, d)).
3. KS_HD1 = 32896 wx_HD1 ON + SL(HD1, 4 d, 2, JV).
4. KS_HD2 = 128 wx_HD2 ON + SL(HD2, 4 d + 2 JV, 1, JV).
5. KS_HD0 = iw (2^lPd Cl_H + wx_HD0 (HN(11, 0) + BIG HN(11, 1)) - place_HD0 - pub_HD0 - 2^`sh`
   (32896 wx_HD0 ON + SL(HD0, 4 d, 2, JV)) - (128 wx_HD0 ON + SL(HD0, 4 d + 2 JV, 1, JV))).
6. KS_HE0 = 32896 wx_HE0 ON + SL(HE0, 4 d, 2, JV) + 2^16 (offE wsum_HE0 + SL(HE0, 4 d + 3 JV, 3,
   JV)).
7. KS_HE1 = 32896 wx_HE1 ON + SL(HE1, 4 d + JVa, 2, JV) + 2^16 (offE wsum_HE1 + SL(HE1, 4 d + 3 JV +
   JVa, 3, JV)).
8. KS_ROW = the sum over j < NRH of mul_j ew_ROW[j] (part_7[j] + 128).
9. KS_CNF = SL(CNF, 0, 2, d) + 2^28 (the sum over k < d of ew_CNF[k] A2(5, k)) + wsum_CNF (2^15 +
   2^39).

In 3 to 7 an offset that exists only at the real pairs has the weight wx_sg ON; the offset offE
exists at every pair and has the weight wsum_sg, since the weights eq(zt, t) add up to 1. SL reads
the cells at every pair. In 5 the two hidden numbers are read under phiC, which holds the positions'
weights and is 0 where there is no cost, and get the columns' weight wx_HD0.

##### The claims, in order

Each item is one claim, a form that must be 0, or as many claims as the item says. The groups and
their counts are those of the first half's table of the claims. The claims take the numbers 0, 1, 2,
.. in the order written, and the claim with the number n has the coefficient chi^n (11.1). The
numbers 1 to 27 before the items only name the items; a claim's number, from 0, follows from the
table. NL is 8 with five limbs and 5 with four.

| group | its first claim has the number |
|---|---|
| 1 | 0 |
| 2 | cn_2 = 32 + v' + d' |
| 3 | cn_2 + 3 |
| 4 | cn_2 + 5 |
| 5 | cn_2 + 5 + NO |
| 6 | cn_2 + 5 + NO + NS |
| 7 | cn_7 = cn_2 + 5 + NO + NS + NL |
| 8 | cn_7 + 3 |
| 9 | cn_7 + 4 |
| 10 | cn_7 + 8 |
| 11 | cn_7 + 11 |
| 12 | cn_12 = cn_7 + 11 + 36 + v'' + d'' + NM + NX |
| 13 | cn_12 + 2; so NCm = cn_12 + 3 |

In the names of 11.1, for the lookup of step 1: its first scalar has the number jLK = 0; LK0 has the
claim number 0, LJ the number cn_7 + 3 and LP0 the number cn_7 + 8. The smallness test's lookup has
jLK = j_8 + 2 (lt + lPd) + 6 (the scalars, above), and the smallness test's first claim has the
number cn_7 + 11.

*Group 1: the lookup of step 1, on mask entries alone.* 31 + v' + d' + 1 claims, as in 11.1.

*Group 2: the sum of squares.*

1. Left(C_4; vL'_4 vR'_4; 0) = 0, with the start form C_4 = 2^(-lPd) times the sum over s < NS of
   gam^s W_s.
2. vL_4 - vR_4 = 0.
3. vL_4 - the sum over i < d / S2 and lb < XL of eq(the first lPd - ls coordinates of rW, i) gam^lb
   part_3[lb d / S2 + i] = 0.

*Group 3: the row products.*

4. Cl_V - the sum over i < NPr of om^i P_i = 0.
5. Left(C_6; S0; 0) = 0, with C_6 = Cl_V. With five limbs S0 = EV'_0^2 + om EV'_0 EV'_1 + om^2
   EV'_1^2 + om^3 EV'_2 EV'_0 + om^4 EV'_2 EV'_1 + om^5 EV'_3 EV'_0 + om^6 EV'_3 EV'_1. With four
   limbs S0 = EV'_0^2 + om EV'_1 EV'_0 + om^2 EV'_2 EV'_0.

*Group 4: what the end values of step 6 are.* The row is read through part 9.

6. With five limbs, four claims: EV_0 - a(9) = 0; EV_1 - b(9) = 0; EV_2 - Rlo(9) = 0; EV_3 - Rhi(9)
   = 0. With four limbs, three: EV_0 - sig(9) = 0; EV_1 - Rlo(9) = 0; EV_2 - Rhi(9) = 0.

*Group 5: the carries.* The row is read through part 8, here and in group 6.

7. For s = 0, .., NS - 1, one claim each: W_s + c_s(8) - D_s(8) - 256 c_(s+1)(8) = 0.

*Group 6: the lines of the norm's row.* With five limbs eight claims, in this order (all fields read
through part 8):

8. LE0: P_1 - Blo - 2^23 Bhi = 0.
9. LE1: Vlo + epsF - P_0 - 2^25 Blo - l1 + 2^48 kap = 0.
10. LE2: Vhi - Bhi - P_2 - kap = 0.
11. LE3: l1 + h1 - 2 sig = 0.
12. LR1: P_4 + P_5 - q1 = 0.
13. LR2: P_6 = 0.
14. LR3: 2^(`MF` + 1) + sig - s0 - 2 P_3 - 2^25 q1 - 2 rr = 0.
15. LR4: rr + rq - sig + 1 = 0.

With four limbs five claims, in this order: Vsq + epsF - P_0 - l1 = 0; l1 + h1 - 2 sig = 0; P_2 - p1
= 0; 2^(`MF` + 1) + sig - s0 - 2 P_1 - 2^25 p1 - 2 rr = 0; rr + rq - sig + 1 = 0.

*Group 7: the stream times the reciprocal.*

16. Left(C_7; EN'_0 EN'_1; 0) = 0, with C_7 = Cl_N.
17. EN_0 - R(10) = 0.
18. EN_1 - the sum over i < d / S2 and lb < XL of 256^lb ewh_NF[i] part_4[lb d / S2 + i] = 0.

*Group 8: the join of the lookup of step 1.*

19. LJ = 0: the join of 11.1 on the key forms KF_c, c = 0, .., NC - 1, in the order of the call
    list.

*Group 9: the four equations of the acceptance policy.*

20. (B2): HN(12, 0) + `C` HN(12, 1) + HN(12, 2) - `theta` RC = 0. In integers: the sum over the
    positions with a cost of gc_t + `C` gb_t + gu_t is `theta` RC.
21. (B3): HN(12, 1) + HN(12, 4) - capR = 0, where capR is `cap` if RC > 0 and 0 if RC = 0. In
    integers: the sum over the positions with a cost of gb_t + gz_t is capR.
22. (B4): HN(11, 0) + HN(11, 3) - `C` ERC = 0. It is the sum over t of phiC[t] (gc_t + gk_t - `C`).
23. (Z): HN(11, 5) + offE JV ERC - (JV - V + 1) 2^`F` ERC + the sum over lb < 3 and i < JV / S2 of
    256^lb part_2[(4 d + 3 JV + lb JV) / S2 + i] = 0. It is the sum over t of phiC[t] (gs_t + the
    sum over v < JV of ex[t][v] - (JV - V) 2^`F` - 2^`F`).

*Group 10: the lookup of step 1, on its planes.*

24. The three claims of 11.1 on the planes, in 11.1's order: the leaves; the planes at the bits'
    point; the number of keys. They are formed with TB_0(i) = part_13[i], TB_1(i) = part_14[i] and
    TP(i) = part_15[i] for each column i of bH.

*Group 11: the smallness test.* 36 + v'' + d'' + NM + NX claims, in the order of 11.1, on the
segment's list of matrices and on the parts above, which are 2 NHf + 13 parts in 11.1's count.

*Group 12: the logits.*

25. Left(C_8; EH'_0 EH'_1; 0) = 0, with C_8 = Cl_H.
26. EH_0 - the sum over k < d of eq(rsP, k) A2(6, k) = 0.

*Group 13: the merge.*

27. The merge's claim (11.1): Left(C; fv'; 0) = 0 with the start form C = EH_1.

No claim says what EH_1 is: it is the value of the weight claim, and the pending row on the stored
table is about it (11.5).

*The equality points of the sumchecks.* Step 4: lPd times 1/2, then pV. Step 6: pV. Step 7: zt. Step
8: zt, then lPd times 1/2. They enter Left(..) as 11.1 says. The variables of each sumcheck are
counted in the order of its rounds.

*A test on the sum of the claims.* After the last claim the programs compute the coefficient that
each hidden(j) has in the sum of all claims, each under its power of chi. They end the call with the
name `internal-h-free-slot`, after the prefix `H:`, unless the coefficient is 0 for the number of
the merge's end value fv' and differs from 0 for each other number j of the segment. The
coefficients are public: they are made of sampled values and public numbers alone. The test that no
coefficient but that of fv' is 0 is not a self-check but a test on content of every verifier: where
another scalar's coefficient is 0, a verifier gives no acceptance (13.9, event 9, which also has
what is known of its chance). The source's comment calls the test a self-check of the verifier's own
list of claims, not of the proof. It adds that the test guards hiding: that every scalar is in a
claim is what lets whoever verifies a proof before it is handed on see an end value that was sent without its mask.

##### The weight claim

The head has one claim on the weights, on unit 13 (11.5), which is the last unit. Its value is EH_1.
Unit 13 has the index groups (x, V, A) and (kk, d, low); write d = o 2^z with o odd. The two factors
are public:

- the group named x: AV[v] = wx_HD0 psiH[v] - sR ew_HD0[v] for v < V, where psiH[v] is the sum of
  eq(rH, t) over the positions t with on_t = 1 and win_t = v, and sR is the sum of eq(rH, t) over
  all t with on_t = 1;
- the group named kk: pt = (rs_(z-1), .., rs_0), so that rs_i is the coordinate of bit i of lo, and
  HV[hi] = eq((rs_(lPd - 1), .., rs_z), hi) for hi < o.

So the claim says: the sum over v < V and k < d of AV[v] eq(rsP, k) Wh[k][v] = EH_1. The merge has
this one claim; its batching element is sampled, and only its power 0 is used. A verifier rejects
the parameter file unless clog2(V) <= lkW.

##### The tests on the public numbers

A verifier makes these tests on the public files alone, before it reads a proof, and rejects if one
fails. The rejection names are in code marks (section 13).

*Ranges.*

| test | rejection name |
|---|---|
| 24 <= `MF` <= 47; `MF` <= `KF` + 14; 17 <= `KF` <= 32; 1 <= epsF < 2^59 (with five limbs the narrower bound epsF < 2^48 of `no-wrap-ledger-eps`, below, is the one that binds) | `policy-norm-F` |
| the line `toksrc` is 1 | `policy-toksrc` |
| `theta` <= `C`; `cap` <= T; `cap` < 2^16; `F` = 24; `s` <= 63; `sh` is 7 or 8; `sT` + 4 = `sh` or `sh` + 4 <= `sT` <= `sh` + 10; S2 divides JVa; 2 V <= 2^`F` | `policy-gap-words` |
| la <= lmhf <= 24; 1 <= NHf <= 64; lt <= lmhd <= 24; lt <= lmhr <= 24; lmbH <= 24 | `shape-request-matrix-H` |
| the tests on the call list (first half) | `policy-call-list-H` |
| for each call: n_c >= 10; n_c >= nminH (nmin without that line); n_c <= 44; n_c >= la; lJ_c >= ls; the line `iotaH_`c equals n_c - dT | `shape-call-H-` and the call's name |
| dT >= la | `forest-depth-H` |
| 5 <= lmbH <= kT + 5 | `forest-planes` |
| the lines `iotaH_`c of the numbers c from NC to 5 are 0 | `header-absent-call-H` |
| lPd + lt >= 9 | `shape-product-layer-H` |
| clog2(V) <= lkW | `hid-vocab-cols` |
| the segment has fewer than 2^30 scalars | `hid-h-slots` |

The tests of a lookup on its own numbers, the one on padding among them, are in 11.1. So are the
tests of the smallness test on its list of matrices, which the head's plan makes too; among them
`hid-k-range`, at most 8191 columns for each matrix, binds the head's own sizes. This part does not
list the programs' self-checks in this segment: their names are in 13.8. Three further names in the
head's source are not in this part either, though they do not begin with `internal-` and so end a
stage with a rejection: `plan-link` where the segment's run begins, `plan-tensor-layout` inside the
run, and `hid-planes-` with a call's name. What they test follows from the public
files and the program's own walk; by this page's reading of the source no public file that has
passed the earlier tests, and no proof, can make one of them fail, and a second verifier needs none
of them. This holds for these places alone: under `plan-link` and `plan-tensor-layout` the programs
also make tests on the public files, which every verifier makes (11.1, third piece; 11.3; 13.8, the
rows of these two names). The test on the optional line `nminH` itself (`policy-nminH`) is in 13.8
too.

The formulas of this half also assume three tests of other parts: S2 divides d (`shape-la`, 11.2); 2
<= V <= 2^24 (`shape-V`, section 13); the line `wb_head` is 13 (`policy-weight-kind`, section 13),
to which BW = 2^12 belongs.

Why (the source's reasons): a cost clipped at `C` is at most `C`, and there are at most T positions,
so a larger `theta` or `cap` says no more. gz_t has two digits, so `cap` is below 2^16. With 2 V <=
2^`F`, under every table a position whose middle digits are all above 0 can have a sum of table
values of at most 2^`F`. With dT >= la all calls share zt and zl, at which part 1 is read for all of
them.

A call ends with an acceptance only if, beyond these tests, the conditions of the standard parameter
form hold (13.7). Those on the head's numbers are `five-limbs` (so with four limbs no call ends with
an acceptance), `norm-shifts` (`KF` <= `MF`), `exact-match` and `policy-named`. The last two are
parts (b) and (c) of condition 11 of 13.7. That condition also has its part (d): a call whose
`-policy` names numbers that exclude nothing is rejected with the name `policy` (14.3, case 4).
`next-token` is a condition of the command `cp_verify` alone (14.4). One test is narrower than the
row `policy-gap-words`: a whole call is rejected with the name `policy`, before the stages, if
`sT` + 4 is not `sh`, whether or not a token of the request is checked (12.5; 14.3 has the four
cases of that name, and says where the comparison stops without a result: by the source a stage then
rejects the call, under another name). So no whole call is judged under a table with `sT` >=
`sh` + 4. `cp_commit` and `cp_prove` reject such a parameter file at their start
(`prover-only-policy-kind`).

*The no-overflow conditions.* Each is an inequality between integers, computed without overflow (the
programs use 128 bits). The bound 2^60 is the same whatever the field; the programs of this
repository can be built for field 2 alone (R1.2). The constants are: Ty = 2^18, the bound on a cell
that the conditions assume of the smallness test (the source's `TyReq`); Ty2 = 257 Ty; TyX = Ty
(256^XL - 1) / 255, which is 4,311,810,305 Ty with five limbs and 16,843,009 Ty with four; BW =
2^12, the bound on a weight of the head (`WEIGHT_BOUND`); BC = 2^11, the bound on an activation (`CLAMPED_CELL_BOUND`); SH
= 2 d BC BW + 1; Be = 65793 Ty + offE. The rejection name is `no-wrap-H-` followed by the name in
the table.

| no. | name | condition |
|---|---|---|
| 1 | `U-NF` | 2^(16 + `KF`) <= 2^60 |
| 2 | `W-NF` | 2^`MF` + 2^38 + 2^(`KF` - 1) + 2^(15 + `KF`) + 2^16 (Ty2 + 32896) < 2^60; with four limbs 2^29 instead of 2^38 |
| 3 | `W-HD0` | 2^8 + 2^(`sh` + 16) + 2^(`sh` + 31) <= 2^40 |
| 4 | `W-HD0` | SH + 2^40 + 2^8 BIG + 2^`sh` (Ty2 + 32896) + Ty + 128 < 2^60 |
| 5 | `three-digits` | SH <= BIG, and SH + BIG + 2^32 <= (2^16 + 1) 2^(`sh` + 15) |
| 6 | `three-digits` | BIG >= SH + 2^(`sh` + 15) |
| 7 | `HE` | 2^16 (Be + 2^`F` + 1) + 2^16 < 2^60 |
| 8 | `Z` | JV 2^`F` + 2^24 + (JV - V + 1) 2^`F` < 2^60 |
| 9 | `E2` | T (2^40 + 2^8 `C` + 2^40) + `theta` T < 2^60 |
| 10 | `E3` | T (2^8 + 2^16) + `cap` < 2^60 |
| 11 | `E4` | 2^40 + 2^40 + `C` < 2^60 |
| 12 | `J-E` | TyX + 16,843,009 2^21 < 2^60 |
| 13 | `guard-radix` | (Ty2 + 32768) + 2^28 (Ty2 + 4096) < 2^59, and Ty2 + 32768 + 65536 < 2^28 |

They are tested in this order. In 13 the number 4096 is the larger of BC and 4096. Condition 12 is
about the embedding (11.2): 2^21 bounds each of the four digits of one of its weights (it is
2^(`LBZW` + 1), with the source's `LBZW` = 20). Two conditions of the feed-forward segments'
list (11.2) are about the final norm too: XL d Ty^2 < 2^53 (`no-wrap-ledger`) and, with five limbs,
epsF < 2^48 (`no-wrap-ledger-eps`).

*What each condition is for, as the source gives it.*

- 1: the source gives no purpose at this place. Conditions 1 and 2 are the two conditions of a
  rounding site of 11.2, `U-` and `W-`, with the shift `KF`, the span 2^16, the bound 2^15 on the
  quotient and the digit form 2. Of `U-` the source says there: two quotients within the span differ
  by less than the span, and the span times 2^`KF` is at most 2^60, which is below the field's
  prime; so they give different values in the field, and the site's relation fixes one quotient.
  With `KF` <= 32 condition 1 always holds.
- 2: the key of NF, with |str R| <= 2^`MF` + sig / 2 and sig below 2^39 (2^30 with four limbs).
- 3: a margin made of its three digits is below 2^40.
- 4: both sides of the field equation 2^(`sh` + 15) K_0 = Mg - 2^`sh` K_1 - K_2 are integers within
  this bound, so they are equal as integers.
- 5: an honest margin, with gc <= `C` < 2^32 and gb = 0, or gc = 0 and gb = 1, is at least 0 for
  every logit gap and fits the three digits.
- 6: at a position above the clip the middle digit of every margin is in the part of the table where
  EX is 1. That is the source's comment at the condition. The digits of a margin are not unique (the
  three digits of a margin, above), so by this page's reading of the source the condition gives
  less: at such a position, if the logits keep the bounds that these conditions assume, every margin
  is at least 2^(`sh` + 15), so its middle digit can be chosen in the rows u >= 2^15, where EX is 1.
- 7: a key of HE0 or HE1 equals a row u + 2^16 EX[u] as integers, which gives K_1 = u and ex =
  EX[u].
- 8: equation (Z).
- 9, 10, 11: equations (B2), (B3), (B4).
- 12: the embedding (above).
- 13: the packed key of CNF.

The bounds BW and 2^21 on the weights are assumptions of these conditions: a request's proof shows
nothing of them (the key file's item `ranges`, 11.5).

##### The kind of an acceptance policy

Two public numbers are derived from the policy's lines. They enter no claim; the programs print them
in the line `covered:` and use the first for the rejection `policy` (12.5) and for two conditions of
the standard parameter form (13.7, condition 11).

*The kind* is one of four words (12.5 has the same four). A whole call takes the first two alone:
under the other two it is rejected with the name `policy`, and the option `-policy` can name only
the first two. The four are:

1. `exact-match` if `C` = `theta` = 2^`sh`, `cap` = 0 and `sT` + 4 = `sh`;
2. otherwise `clipped-gap` if `sT` + 4 = `sh`;
3. otherwise `log-probability` if `sT` = `s`;
4. otherwise `another-table`.

The line `s` is in no equation: it is the provider's statement of the exponent of the model's logits
(an integer logit is a real logit times 2^`s`).

*The number above the clip*, for n = RC tokens: let least = 2^`sh` and most = min(`cap`, n). If `C`
<= least the number is most. Otherwise let room = floor(n (`theta` - least) / (`C` - least)) if
`theta` > least, and room = 0 if not; the number is min(room, most). The source gives it as an upper
bound on the positions whose cost is above `C`, and says that a token whose cost is above `C` can be
any token. It says that the number is not a bound on the tokens that can be any token: how much a
cost of at most `C` rules out depends on the model's logits, which a verifier does not see. And it
says that under the step table the number is reached whatever the model, if the lines admit a proof
at all: so many tokens of any kind, and the model's top choices, at 2^`sh` each, for the rest.

##### What the claims are for, as the source's comments argue

This page states no property of the protocol. What follows is the argument of the comments in the
programs' source, in their terms, with this half's letters. The source states what the segment is
about as follows. A position's gap is its logit gap (the margins, above); it is 0 if the token's
logit is the largest, with the lowest index on ties. Over the RC positions with a cost, the sum of
min(gap + 2^`sh`, `C`) is at most `theta` RC. At most `cap` of these positions have gap + 2^`sh`
above `C`. If `sT` >= `sh` + 4, the same holds for the tokens' negative log-probabilities at the
exponent `sT`; the comments say so, but no whole call is judged under such a table (the kind of a
policy, above). If e = T, y is the index of the largest logit of position T - 1. The comments take
for granted that the claims hold, that the lookups and the smallness test say of the cells what 11.1
gives as their statements, and that the no-overflow conditions hold.

1. *The bracket.* The term [v < win_t] in a margin "makes the first of equal logits the largest".
2. *Pairs that are not real.* There the three keys are d1, d2 and - (2^`sh` d1 + d2) / 2^(`sh` +
   15), "which are in their lookup tables only if all three are 0". With d1 = 0 the key of HE0 or
   HE1 is a row of the table only if ex is EX[0].
3. *The six numbers.* All six are at least 0 only by the lookup on their digits (ROW).
4. *(B2), (B3).* They are plain sums over the RC positions with a cost. *(B4)* says gc_t + gk_t =
   `C` at each such position, tested at the point zt, which is sampled after the cells were
   committed. So gc_t <= `C`.
5. *(Z)* says, at each position with a cost and at the same point, that gs_t and the position's V
   table values add up to 2^`F`. gs_t is at least 0, so the values add up to at most 2^`F`. EX[0] =
   2^`F` and every row of the table is at least 1, so no middle digit K_1 of the position is 0:
   gc_t + BIG gb_t is at least the logit gap plus 2^`sh`.
6. *Under a table of exponentials* (`sT` >= `sh` + 4): ex is at least 2^`F` exp(- K_1 2^`sh` /
   2^`sT`), and 2^`sh` K_1 is at most the margin. So (a step that the comments leave out) the sum
   over v < V of exp((lam_t[v] - lam_t[win_t] - gc_t - BIG gb_t) / 2^`sT`) is at most 1. Hence
   gc_t + BIG gb_t is at least the token's negative log-probability at the exponent `sT`, in units
   of the integer logits. *Under the step table* (`sT` + 4 = `sh`; the values EX[u] are 2^`F`, 2, 1,
   1, ..) equation (Z) says only that no middle digit is 0, and the bound is on the logit gaps, each
   plus 2^`sh`.
7. *The sum.* A position with gb_t = 0 has a cost of at most gc_t, by its margins and (Z); a
   position with gb_t above 0 is charged at least `C`. Hence the costs, each clipped at `C`, add up
   to at most `theta` RC, and at most `cap` positions have a cost above `C`.
8. *The next token.* Position T - 1 has no cost, so its margins have no hidden number: y must be the
   index of the largest logit, the lowest index on ties.
9. *The exact match.* The source calls `exact-match` the policy under which every checked token is
   the top choice of its position, and says that under other values of these lines an acceptance can
   say less, down to nothing: `theta` = `C` with `cap` = T admits any tokens.

(B4) and (Z) are statements on every position with a cost. Each is tested at one point, zt, and has
no round of its own: 13.6 has what the source says of the rounds that carry them, and what this page
does not settle.

##### Easy to get wrong in this half

1. *Offsets only at the real pairs.* The offsets 32896 and 128 of a margin's two lower digits have
   the weight wx_sg ON. The offset offE of a table value has the weight wsum_sg. The offsets of NF,
   NFH, CNF and ROW exist at every position.
2. *Three weights of one call's segment.* wsum_sg runs over all n_sg entries, wx_sg over the entries
   below V, and padW_c over the columns that no call's segment takes. The columns of the entries
   from V on are in wsum_sg, are not in wx_sg and are not in padW_c.
3. *HE1 is shifted.* Its entry x is the vocabulary entry JVa + x, in the features read and in
   wx_HE1.
4. *2^lPd and 2^(-lPd).* The scalar Cl_H holds 2^(-lPd) times the sum, and KS_HD0 multiplies it by
   2^lPd. The start form of step 4 has the factor 2^(-lPd), and no claim takes it back: the features
   are summed at the coordinate 1/2.
5. *The same cells under several weights.* d1 is read by HD1, by HD0 and by HE0 or HE1, each time
   through part 1 with the weights ewh of the reading call's segment. aw is read by NF and CNF, rh
   by NF and NFH.
6. *128 times the sum of the weights.* A digit read through parts 7 to 10 gets 128; through part 11
   it gets 128 ERC; through part 12 it gets 128 RC.
7. *capR is 0 if no position has a cost.*
8. *y comes from the proof.* win_(T-1) is the first coordinate of the scalar of the round `REPLY`,
   not the next token of the call. The frame compares the two (R14.8).
9. *The order of groups 3 to 6.* Groups 3 and 4 (the row products, steps 5 and 6) come before group
   5 (the carries), although the scalars W_s of group 5 were sent first, in step 3. Inside group 6
   the order is LE0, LE1, LE2, LE3, LR1, LR2, LR3, LR4.
10. *pV is used as sampled.* The lt elements of step 2's `ZERO` are the coordinates of pV in the
    order sampled, the first for the highest bit. The two points of a lookup that are sampled in one
    round (11.1: the elements of `JOIN` and of the lookup's `ZERO`) are sampled with the lowest
    bit's coordinate first and are used in the reverse order.

##### What this half does not specify

It does not define the honest prover's rule for a position's cost and its hidden numbers, the choice
of a margin's digits where several fit, or the choice of limbs where several give one number: a
verifier needs none of them. It does not argue that the claims imply the integer function; the
passage above only reports the source's comments. The forms of Left(..), of the join, of the planes'
claims, of the smallness test's claims and of the pending row are in 11.1 and 11.5.
