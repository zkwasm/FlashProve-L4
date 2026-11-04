*Part of [the file format](FORMAT.md), [section 11](11-sub-proofs.md). Rule numbers and terms are
those of that page.*

### 11.3 An attention segment

This part is written from the programs' source. It gives the order of an attention segment: which
arrays the prover commits to, and every round of the segment by name, with what the prover sends in
it and what is sampled. It needs 11.1 (the kinds of step), R5.1 and R5.2 (a round and its query),
R6.2 to R6.9 (how a verifier's message is sampled), R8.2 to R8.7 (a chain in a file) and R14.4 (the
rounds that every segment has). A segment, an oracle message and a root are used as R14.4 and R7.12
use them. The word "group" has two uses in this first half; the second half has a third, the ten
groups of claims. "Group g" and "the group" mean a group of Lam neighbouring layers (11.2). "A
segment's group" means the group of oracle messages of that segment (R14.4); it occurs only in rows
2 and 25 of the table of the segment's rounds and in the paragraph after that table. The names of
the rounds, with their kinds, are in R14.11.

An **attention segment** is the segment of a request's first chain that comes directly after the
feed-forward segment of group g (the order of the segments: the head of section 11). So an attention
segment is never the first segment of a chain and never the last. The segment's word in the item `links' identifiers` (R14.3)
is 2 + NG + g, where L is the number of the model's layers and NG = L / Lam is the number of groups
of layers (13.4, step 5).

#### What the segment is about

The feed-forward segment of group g has committed to an array qa that holds, for each of the group's
layers, the integer tensors q, kx and v (queries, keys and values), which go into the attention
function, and o, which comes out of it. The attention segment is meant to show that o is the model's
integer attention function of q, kx and v, with the causal mask. Its steps are:

1. Each score is the inner product of a row of q and a row of kx.
2. Each row's maximum is reached at an index that the prover commits to.
3. Each weight is read from a public table of the exponential function at the maximum less the
   score.
4. The denominator of a row is the sum of its weights.
5. o is the rounded quotient of the weighted sum of v by the denominator, with remainders in their
   ranges.

An attention segment reads no weights of the model, and it leaves no row on the stored table.

#### Numbers

All numbers of this part follow from the three public files. The lines named are lines of the
parameter file.

| symbol | meaning |
|---|---|
| lt, T | line `lt`; T = 2^lt is the number of tokens of a request |
| lB, lnT | line `lB`; lnT = lt - lB |
| H, G, lgk, HA | lines `H` and `G`: the number of query heads and the number of key-value heads; G = 2^lgk; HA = H / G |
| dh, ldh | line `dh`: the width of a head; dh = 2^ldh |
| ll, Lam | line `ll`; Lam = 2^ll is the number of layers of a group |
| NP | Lam HA |
| la, nmin, kFD, lmS, NAs, lmh, lmo, lmbA, tau | the lines of these names |
| kFM | line `kFM`: it is hashed like every line, and no test and no formula of a verifier reads it |
| lidb | line `lidb`; 22 if the file does not have the line |
| sh(lam) | the shift of layer lam of the group, from the scale file; nsh is the number of different shifts in the group. Under the standard parameter form all layers of a group have the same shift (13.7, `one-shift`), so nsh = 1 and each line `callA` gives one call |
| m_qa, k_qa | the number of cell rows and the number of columns of qa: m_qa = 2^lmq, with the line `lmq`, and k_qa is given in 11.2 (the columns) |
| s | line `answers`: the number of answers to each out-of-domain sample; the parser holds it at 1 (8.1, test 14) |
| nM, nD | nM = lgk + 2 lt - 1; nD = lgk + lt + lB. nD is the number with which section 13's test `af-shape-lmS` compares lmS |
| clog2(x) | the least l with 2^l not below x |

#### The lookup calls

The lines `callA` of the parameter file list the segment's lookup calls. A line is a name, a kind
(`KR`, `KFD`, `R16` or `WH`) and the call's segments (11.2), one or more. Each of the seven calls'
segments `RM`, `RD`, `FM`, `FD`, `WL`, `WH`, `IX` is in exactly one line, and in a line of its kind:
`RM` and `RD` in `KR`; `FM` and `FD` in `KFD`; `WL` and `IX` in `R16`; `WH` in `WH`. The **expanded
list** is made line by line: a line of kind `KR` or `KFD` gives one call for each different shift of
the group, in increasing order, and any other line gives one call. NC is the number of calls in the
expanded list; it is at least 4, and a verifier rejects a list with more than 12.

A call's keys come in pieces. For each of the call's segments, in the line's order, for each layer
lam of the group (for the kinds `KR` and `KFD`: only the layers whose shift is the call's), and for
each a below HA, there is one piece of 2^nb keys. nb is nM for `RM` and `FM`; nD for `RD` and `FD`;
1 + lgk + lt + ldh for `WL`; lgk + lt + ldh for `WH`; lgk + lt for `IX`. With tot_c the sum of the
pieces' sizes, call c has n_c = max(nmin, clog2(tot_c)) key bits. Its table has 2^k_c rows, with k_c
= kFD for the kind `KFD` and 16 for the other kinds. The tables' rows are in 11.1.

A **unit** is a pair of a call and one of the call's segments. The units of `KR` are taken call by
call in the order of the expanded list, and within a call `RM` before `RD`, whatever the line's
order. The units of `KFD` are taken in the same way, `FM` before `FD`. There are nKR = 2 nsh units
of `KR` and nKF = 2 nsh units of `KFD`. A unit of `KFD` has np pieces: HA times the number of the
group's layers that have the call's shift. Over all units of `KFD` these add up to 2 NP.

#### The arrays

The prover commits to the segment's arrays at the moments 0, 1 and 2 (11.1). The segment's own four
matrices of cells have the names as, ah, ao and bA; in "the parts of as" or "plane x of as" the last
word is the first of these names. A matrix of m cell rows and k columns is committed as an array of
ceil(m / 3) rows and k columns over F: one element holds three cell rows of one column. The names
are those of the item `layout` (R14.5). The layout file says in which oracle message an array is
placed.

| moment | name | what it holds | number of cell rows m | number of columns k |
|---|---|---|---|---|
| 0 | `M0` .. `M`(NAs - 1) | the parts of as: five cells for each score | 2^lmS each | kAs / NAs each |
| 0 | `M`(NAs) | ah: each row's maximum and denominator, four cells each, and the lt bits of its index | 2^lmh | fillH / 2^(lmh - lt) |
| 0 | `M`(NAs + 1) | ao: the two remainders of each division | 2^lmo | fillO / 2^(lmo - lt) |
| 0 | `M`(NAs + 2) | bA: the multiplicities of the segment's lookup, as 32 planes of bits | 2^lmbA | kbA |
| 1 | `U3` | partial evaluations of the five matrices as, ah, ao, bA and qa | one row of nW elements of E | |
| 2 | `ZZ` | the two digits of every combination of the smallness test | tau rows of 2 mz elements of F | |
| 2 | `BZ` | the multiplicities of the lookup on the digits, as planes of bits | 32 rows of 2^11 elements of F | |

Here kAs = NAs ceil(5 NP (2^nM + 2^nD) / (NAs 2^lmS)). fillH is 8 Lam H + Lam lt H, rounded up to a
multiple of 2^(lmh - lt). fillO is 2 Lam H dh, rounded up to a multiple of 2^(lmo - lt). kbA = NC
2^(kT + 5 - lmbA), with kT of the segment's lookup (11.1). mz = NAs 2^lmS + 2^lmh + 2^lmo + 2^lmbA +
m_qa. nW = 3 kAs + (nKF + 6) fillH + fillO / 2^(la - lt) + (2 nKF + 5) k_qa m_qa / 2^la + 3 kbA.
`U3`, `ZZ` and `BZ` are not packed: the table gives their rows and columns as they are committed.

The segment also reads qa, which the feed-forward segment of group g has committed to. That array is
kept up to this segment and is sent again in this segment's first prover's message (R14.4).

#### Two runs of rounds that come again and again

A scalar is defined in 11.1. In the tables, the column "sampled" gives the number of elements of E
that are sampled, unless it says otherwise. A scalar belongs to the message of the round that is
closed next. So a scalar that is sent outside a sumcheck has no round of its own: it is at the start
of the message of the next round.

A **sumcheck** [k_1, .., k_n | d_Y | N] is the sumcheck of 11.1 with n variables: the i-th round
polynomial has degree k_i, so k_i scalars are sent; d_Y is the degree in Y; N is the number of end
values. A sumcheck [k_1, .., k_n] has neither `PSC Y` nor `PSC values`.

| round | times | the prover's message | sampled |
|---|---|---|---|
| `PSC epsilon` | 1 | the scalars that are waiting, if any; then 1 scalar, the last item of the message (R8.4) | 1 |
| `PSC round` | n | k_i scalars in the i-th | 1 |
| `PSC Y` | 1 | d_Y scalars | 1, not 0 (R6.5) |
| `PSC values` | 1 | N scalars | nothing |

n can be 0. A product level with n variables (11.1) is the sumcheck [2, .., 2 | 2 | 2] with n
entries 2, and after it one round `LINE` with an empty message, in which 1 element is sampled.

In the sumchecks of rows 4 to 15 of the segment's table every `PSC round` is a plain round (11.1):
the linear coefficient is the one that is not sent. In a product level and in the sumchecks of a
lookup every `PSC round` is an equality round: the constant is the one that is not sent.

A lookup (v, dL) is the lookup of 11.1 with these two numbers. Both lookups of an attention segment
have more than one call, so both begin with `LKP shift`. For an attention segment's own calls a
verifier also rejects the parameter file unless dT is at least lmS and at least lt + ldh, and unless
the line `iotaA_0` is 32 + dT.

#### The rounds of the segment

(v_A, d_A) are the numbers v and dL of the lookup on the expanded list, and (v_Z, 11) those of the
lookup on the digits (11.1).

| # | rounds | the prover's message | sampled |
|---|---|---|---|
| 1 | `SEND seed` | the roots of the trees of the segment's first prover's message (R7.12, R14.4), 32 bytes each, with no length before them (R1.5) | the s seeds of an out-of-domain sample (R6.9) |
| 2 | `LOC` | s answers to the out-of-domain sample for each oracle message of that prover's message that gets one (R8.3): the fresh mask of the segment's fold, which is in that prover's message too, gets none (section 10) | the positions of the group of oracle messages of the feed-forward segment of group g. After the round, in the file: the openings (R8.7) |
| 3 | the lookup (v_A, d_A) | | |
| 4 | for each unit of `KFD`, in order: the sumcheck [3, .., 3 \| 2 \| 2 np], with lgk + 2 lnT - 1 + ldh variables for `FM` and lgk + lnT + ldh for `FD` | before the sumcheck's own scalar, in its `PSC epsilon`: 7 scalars. In the first unit's, before these 7: 3 scalars for each unit of `KR`, 3 nKR in all | |
| 5 | the sumcheck [3, .., 3 \| 2 \| 2 NP], with lgk + lt variables | before its own scalar: 5 + lt + 2 scalars | |
| 6 | `BAT-sigma`, for 2 rows | empty | 1 |
| 7 | the sumcheck [3, .., 3 \| 2 \| 2 NP], with lgk + 2 lnT - 1 + lB variables | before its own scalar: 2 scalars | |
| 8 | the sumcheck [3, .., 3 \| 2 \| 2 NP], with lgk + lnT + lB variables | | |
| 9 | `ZERO` | empty | 2 lgk + 2 lt + 4 |
| 10 | `BAT-sigma`, for NP rows | NP scalars | 1 |
| 11 | the sumcheck [3, .., 3 \| 2 \| 2 NP], with lt + lgk + ldh variables | | |
| 12 | `BAT-sigma`, for NP rows | empty | 1 |
| 13 | the sumcheck [2, .., 2, lt + 2, .., lt + 2, lt + 1, .., lt + 1 \| lt + 1 \| Lam + NP lt], with lt entries 2, then lgk entries lt + 2, then lt entries lt + 1 | | |
| 14 | `BAT-sigma`, for NP rows | empty | 1 |
| 15 | the sumcheck [3, .., 3 \| 2 \| 2 NP], with lt + lgk + 4 variables | | |
| 16 | `SEND seed` | the roots of the trees of moment 1: `U3` | the s seeds of an out-of-domain sample |
| 17 | `SEND answer` | s answers to the out-of-domain sample for each oracle message of the moment | nothing |
| 18 | `RNG` | empty | K tau matrix entries (R6.7), with K = kAs + fillH / 2^(lmh - lt) + fillO / 2^(lmo - lt) + kbA + k_qa |
| 19 | `SEND seed` | the roots of the trees of moment 2: `ZZ` and `BZ` | the s seeds of an out-of-domain sample |
| 20 | `SEND answer` | s answers to the out-of-domain sample for each oracle message of the moment | nothing |
| 21 | the lookup (v_Z, 11) | | |
| 22 | `BAT-sigma`, for NCm + Q rows | empty | 1, the element chi |
| 23 | `BAT` | empty | 1 |
| 24 | `FOLD epsilon` | 1 scalar | 1 |
| 25 | `FOLD round`, once for each row variable of the segment's group | 2 scalars | 1 |

Rows 1, 2, 16, 17, 19, 20 and 23 to 25 are the rounds that every segment has (R14.4); they are
listed here so that the order is complete. The positions of this segment's own group are sampled in
the `LOC` of the next segment. A round with proof-of-work bits has its nonce after its message
(R8.2); which rounds have such bits is in the plan.

The matrix entries of row 18 are sampled together (R6.7), in this order: matrix by matrix in the
order as (its parts), ah, ao, bA, qa; within a matrix column by column; tau entries for each column.
The columns of qa are among them, although another segment has committed to qa.

*The number of rows of row 22.* A plan's entry for a round that batches rows holds their number
(R4.3), so it is hashed. The number is NCm + Q. NCm is the number of the segment's claims, NX is the
number of its reads, and Q is how many numbers the equations of its smallness test take (11.1). With
LL(v, dL) = 32 + v + dL:

      NX = 3 NAs + (nKF + 6) 2^(lmh - lt) + 2^(lmo - la) + (2 nKF + 5) m_qa / 2^la + 3
      NCm = LL(v_A, d_A) + 3 nKR + 6 nKF + 4 NP + (5 + lt + 2 NP) + 1 + (1 + 4 NP) + (4 NP + Lam + NP lt) + 3 + (nKF + 6)
          + LL(v_Z, 11) + 4 + (NAs + 4) + NX
      Q = tauP (NX + the sum over the NAs + 4 matrices of 2^clog2(ceil(m / 3))),   tauP = 2^clog2(tau)

*The number of scalars.* The segment's scalars are those of rows 3 to 15 and 21, numbered from 0 in
the order sent. The test on their number is in 11.1 (the room in the mask row).

#### A worked example

With lt = 10, lB = 6, H = 40, G = 8, dh = 128, ll = 2, nmin = 20, `lidb 28`, kFD = 16, tau = 128,
lmS = 16, NAs = 1, lmh = 13, lmo = 16, lmbA = 14, lmq = 16, one shift in the group (nsh = 1) and the
four calls (`RM` `RD`), (`FM` `FD`), (`WL` `IX`), (`WH`): NP = 20, nM = 22 and nD = 19; the calls
have 27, 27, 26, 25 key bits, so dT = 25, NT = 11, v_A = 29, d_A = 18, and the line `iotaA_0` is 57;
mz = 221,184, so v_Z = 26. The scalars are: row 3: 1,812; row 4: 199; row 5: 99; row 7: 93; row 8:
82; row 10: 20; row 11: 103; row 13: 382; row 15: 94; row 21: 1,306; 4,190 in all. These are the
values of `target_full_f2/policy.txt`, the full-size example configuration (section 8).

#### Easy to get wrong

1. *Which scalars are in which round.* Values that are sent outside a sumcheck have no round. The 3
   nKR scalars of the units of `KR` are in the `PSC epsilon` of the first unit of `KFD`, before that
   unit's 7. The NP scalars of row 10 are the message of a `BAT-sigma`. A round that follows a round
   in which something was sampled, with nothing sent between, has an empty message, and it asks its
   query like any other round.

2. *The order of the bits.* In the programs a point's coordinate 0 belongs to the highest bit of an
   index. The 5 elements of `JOIN` and the 5 + dL elements of a lookup's `ZERO` are sampled with the
   lowest bit's coordinate first, and are used in the reverse order. A product level binds its
   variables from the highest bit down. The last product level of the key side (step 5 of a lookup),
   with v - 1 variables, has another order, which is given in 11.1. The number of rounds does not
   show this; the meaning of each challenge does.

3. *The order of the claims is part of the protocol.* No round of a sumcheck tests an equation: the
   coefficient that is not sent is computed from the round's equation. What is left of every check
   is a claim. Claim number c, counted from 0 in a fixed order, is multiplied by the c-th power of
   chi, the element of row 22. The claims of the smallness test on a matrix or on a read are the
   exception (11.1): each takes a number, but it is its equations that are multiplied, equation
   number j by the power NCm + j. The sum is one equation, which the segment passes on to its
   closing (rows 23 to 25). A verifier that adds the same claims in another order tests another
   equation.

#### What this part does not specify

This first half gives the order of the rounds and the counts; the second half gives the integer
function, what each cell, each scalar and each entry of `U3` holds, the key of every lookup call,
every claim as an equation in the order of the claims, and the tests on the public files with the
no-overflow conditions. The two halves do not define how an honest prover cuts a value into limbs or
a weight into its two cells, or what it writes into the committed cells that have no meaning: a
verifier needs none of these. They give no argument that the claims imply the integer function, that
the cells beyond the sizes in use do no harm, or that an equation in the field holds in the
integers.

#### The cells, the keys and the claims

This part is written from the programs' source. It is the second half of 11.3. The first half gives
the order and the size of every message of an attention segment. This half gives what the messages
mean: the integer function, the cell that holds each number, the key of every lookup call, every
scalar by name, and every claim as an equation, in the order in which the claims are numbered. It
needs the first half and the three pieces of 11.1: the first (the rounds of the steps that all
sub-proofs share, the public tables, the order of bits), the second (values, forms and claims; the
lookup) and the third (the smallness test). From the second and third it takes four things by name:
a hidden value and its mask entry; the claim that a sumcheck leaves; the lookup's own claims, its
join and its three claims on the planes of bA; and an entry of a part of `U3` with the reads that
tie it to the cells.

Everything that this half calls public is computed by the verifier from the three public files, from
sampled values and from the scalars as they were sent (11.1, second piece). Everything else is
hidden: the verifier knows it only as a scalar under a mask or as an entry of a committed array, and
it never compares two such values itself. It only adds claims to the segment's one equation (11.1).

##### Notation

The symbols of the first half's table "Numbers" and of 11.1 keep their meaning. g stays the group's
number. "Row n" without more means row n of the first half's table "The rounds of the segment".
These are added:

| symbol | meaning |
|---|---|
| Bt, nT, hP | Bt = 2^lB is the side of a tile; nT = 2^lnT = T / Bt is the number of tile rows; hP = nT / 2 |
| ls, S2 | ls = la - lt; S2 = 2^ls |
| dq, dkv | dq = H dh; dkv = G dh |
| wcH, wcO, wcQ | 2^(lmh - lt), 2^(lmo - lt) and m_qa / T: how many features one column of ah, ao, qa holds |
| fillQ | k_qa wcQ |
| Sx, rx | the lines `S` and `rx` |
| Tt[u], u0 | the table of the exponential function and its length (11.1, public tables); 12.6 and section 13 call it the softmax table |
| Smax | dh x 2047^2. 2047 is the bound of the clamp of q, kx and v (the source's `CLAMP`) |
| C_sh, rho, nHD | for a shift sh: C_sh = u0 2^sh; rho = 65536 - 2^sh; nHD = ceil((4 Smax + 1) / rho) (as in 11.1) |
| Om_sh | 2 Smax + C_sh |
| lam | a layer's number in its group, 0 .. Lam - 1; it is layer g Lam + lam of the model. Its shift sh(lam) is that layer's word sh in the scale file (11.2) |
| hk, a, h | hk, 0 .. G - 1, is a key-value head; a, 0 .. HA - 1, is a query head's number among those of hk; h = hk HA + a is the query head |
| b | a layer lam and a number a as one number, b = lam HA + a, 0 .. NP - 1 |
| i, j | token positions: i of the query, j of the key |
| I, J, il, jl | i = I Bt + il and j = J Bt + jl, with il, jl below Bt: tile row, tile column, and the positions inside a tile |
| ch | a channel of a head, 0 .. dh - 1 |
| x | the number of a plane of as, 0 .. 4 |
| l | the number of a limb |
| k | the number of a bit of a position, 0 .. lt - 1; bit 0 is the lowest. (With an index, k_0, .., k_(n-1) are the degrees of a sumcheck's rounds, which the first half counts from 1, and k_c, k_qa are as in the first half) |
| pair | always a pair (i, j) of token positions |
| z[m .. n) | the coordinates number m, .., n - 1 of the point z, counted from 0 |
| (y, z) | the point whose coordinates are those of y and then those of z |
| eq(z, n) | the equality weight of the index n at the point z (11.1, the order of bits): coordinate 0 belongs to the highest bit of n. eq of a point without coordinates is 1 |
| eqp(y, z) | for two points of equal length: the product over s of y_s z_s + (1 - y_s)(1 - z_s) |
| ones(z) | the product of all coordinates of z |
| id(z) | for a point of n coordinates: the sum over s of 2^(n - 1 - s) z_s |
| LE(y, z) | for two points of n coordinates: LE_n = 1; LE_s = (1 - y_s) z_s + (y_s z_s + (1 - y_s)(1 - z_s)) LE_(s+1) for s = n - 1, .., 0; LE(y, z) = LE_0. On bits it is 1 if the number y is at most the number z, and 0 otherwise |
| shown(X) and [X] | for a scalar with the name X and the number j: shown(X) is shown(j), the scalar as it was sent, and [X] is hidden(j) = shown(j) - factor(j) mask(j), with factor(j) = 1 for a value sent alone and the sumcheck's beta for an end value (11.1, second piece) |
| U(p; n) | entry number n, from 0, of the part p of `U3`: U3[at(p) + n], where at(p) is the sum of the lengths of the parts before p (11.1, third piece). It is a hidden entry of a committed array |

A sum without a stated range runs over all values of its index. All arithmetic in a claim is in E.
An integer in a claim means its image in F; a negative integer -n means the element -n of F.

##### What the segment computes

The function is stated for one layer lam of the group, with its shift sh = sh(lam), and one query
head h = hk HA + a. All numbers are integers. q, kx, v and o are tensors of qa: q[lam, h, i, ch] the
queries, kx[lam, hk, j, ch] the keys, v[lam, hk, j, ch] the values, o[lam, h, i, ch] the output.

1. *Where q, kx and v come from.* The attention segment does not test it. The projections, the
   rotary tables and the clamp of q, kx and v to -2047 .. 2047 belong to the feed-forward segment of
   the group, which commits to qa (11.2: the calls' segments `CQK` and `CV`). kx is the key after
   the rotary tables and the clamp. What becomes of o is also in 11.2.
2. *Scores.* sc(i, j) = the sum over ch of q[lam, h, i, ch] kx[lam, hk, j, ch], for j <= i. With q
   and kx in the clamp's range, sc is between -Smax and Smax.
3. *The maximum.* Mx(i) = sc(i, ib(i)) for a position ib(i) <= i, and Mx(i) >= sc(i, j) for all j <=
   i.
4. *The weights.* With Df(i, j) = Mx(i) - sc(i, j) and t = floor(Df(i, j) / 2^sh): ew(i, j) = Tt[t]
   if t < u0, and ew(i, j) = 0 otherwise. For j > i in a tile above the diagonal there is no weight;
   in a tile on the diagonal the weight is 0 (the causal mask, below).
5. *The denominator.* Zd(i) = the sum over j <= i of ew(i, j).
6. *The weighted sum.* acc(i, ch) = the sum over j <= i of ew(i, j) v[lam, hk, j, ch].
7. *The rounding.* o[lam, h, i, ch] is the integer for which rem = 2 acc(i, ch) + Zd(i) - 2 Zd(i)
   o[lam, h, i, ch] satisfies 0 <= rem <= 2 Zd(i) - 1. That is acc / Zd rounded to the nearest
   integer, a half rounded up.

No floating-point number enters but through the public table Tt (11.1).

##### Tiles and their addresses

The T x T pairs (i, j) of one head are cut into nT x nT tiles of Bt x Bt pairs. Tile (I, J) holds
the pairs with tile row I and tile column J. Cells exist for two **regions**:

- **Region D**, the nT tiles on the diagonal, J = I. All Bt^2 pairs of such a tile have cells, those
  with jl > il too.
- **Region M**, the tiles below the diagonal, J < I. A tile of region M has an **address** (P, sg),
  with P = 0 .. hP - 1 and sg = 0 .. nT - 2. The address (P, sg) is the tile (I, J) = (P, sg) if sg
  < P, and the tile (I, J) = (nT - 1 - P, sg - P) otherwise. So the addresses with the number P hold
  the P tiles of tile row P and then the nT - 1 - P tiles of tile row nT - 1 - P. Every tile with J
  < I has exactly one address: it is (I, J) if I < hP, and (nT - 1 - I, J + nT - 1 - I) otherwise.
  The address (P, nT - 1) is **empty**: it has cells, but no tile.

Tiles with J > I have no cells. Write I(P, sg) and J(P, sg) for the tile of an address. For a point
(yP, ys) of lnT - 1 and lnT coordinates, two public vectors of nT entries are used again and again:

      WI(yP, ys)[I'] = the sum of eq(yP, P) eq(ys, sg) over the addresses (P, sg), sg <= nT - 2, with I(P, sg) = I'
      WJ(yP, ys)[J'] = the sum of eq(yP, P) eq(ys, sg) over the addresses (P, sg), sg <= nT - 2, with J(P, sg) = J'

The **index of a pair in its region** has the highest bits first: in region M it is n = (((hk hP +
P) nT + sg) Bt + il) Bt + jl, a number of nM bits; in region D it is n = ((hk nT + I) Bt + il) Bt +
jl, a number of nD bits.

##### The arrays and what each cell holds

The segment commits to seven kinds of array: as (in NAs parts), ah, ao, bA, `U3`, `ZZ` and `BZ`; and
it reads qa. Their sizes are in the first half. as, ah, ao, bA and qa are matrices of small
integers, the cells, and are packed three cell rows to an element (11.1). This heading says which
cell holds what. Many cells hold a byte less 128, so that the cell is a small number of either sign;
the claims add the 128 again.

*as: five cells for each pair.* The five **planes** of as hold, for each pair:

| plane x | the cell holds | meaning |
|---|---|---|
| 0 | (rm mod 256) - 128 | rm is the pair's remainder, 0 .. 65535 |
| 1 | floor(rm / 256) - 128 | |
| 2 | fl | the flag: 1 if rm >= 2^sh, and 0 otherwise |
| 3 | c3 | ew = (c3 + 128) + 256 c4 is the pair's weight |
| 4 | c4 | |

Write c0, .., c4 for the five cells of a pair. Only (c3 + 128) + 256 c4 enters the claims; this page
does not define how a prover cuts ew into c3 and c4. The cells of as are numbered through all parts.
Plane x of as, for the pair with the index n, layer lam and number a, has the **cell number**

      in region M:   N = ((x Lam + lam) HA + a) 2^nM + n
      in region D:   N = 5 NP 2^nM + ((x Lam + lam) HA + a) 2^nD + n

Cell number N is in cell row N mod 2^lmS of column floor(N / 2^lmS) of as. Column number col of as
is column col mod (kAs / NAs) of part floor(col / (kAs / NAs)), that is of the array `M`(that part's
number). No claim of this part reads a cell whose number is 5 NP (2^nM + 2^nD) or more.

*ah, ao and qa: features.* Each of these three matrices holds, for every token, a list of numbers,
the **features**, numbered from 0. With wc = wcH, wcO or wcQ, the cell of feature f and token
position pos is in column floor(f / wc) and in cell row (f mod wc) T + pos. A **tensor** in such a
matrix is given by four numbers (off, nl, cnt, nf): nl limbs, each of cnt layers of nf features,
from feature off on (nf is the source's J). Limb l of the tensor's entry number f' of layer lam is
feature

      off + (l cnt + lam) nf + f'

(the source's `tenCell`). cnt is Lam for every tensor of this part. The **value** of an entry is the
sum over l of 256^l times the cell of limb l. Only this sum enters the claims. This page does not
define how a prover cuts a value into limbs.

| matrix | tensor | off | nl | nf | entry number f' | value |
|---|---|---|---|---|---|---|
| ah | Mx | 0 | 4 | H | h | Mx(i) at token i |
| ah | Zd | 4 Lam H | 4 | H | h | Zd(i) at token i |
| ah | ibit | 8 Lam H | 1 | lt H | k H + h, for k = 0 .. lt - 1 | bit k of ib(i) at token i; bit 0 is the lowest |
| ao | rh | 0 | 1 | dq | h dh + ch | floor(rem / 65536) - 128 at token i |
| ao | rhp | Lam dq | 1 | dq | h dh + ch | floor((2 Zd(i) - 1 - rem) / 65536) - 128 at token i |
| qa | q | 0 | 2 | dq | h dh + ch | q[lam, h, i, ch] |
| qa | kx | 2 Lam dq | 2 | dkv | hk dh + ch | kx[lam, hk, j, ch] |
| qa | v | 2 Lam dq + 2 Lam dkv | 2 | dkv | hk dh + ch | v[lam, hk, j, ch] |
| qa | o | 2 Lam dq + 4 Lam dkv | 2 | dq | h dh + ch | o[lam, h, i, ch] |

ah has 8 Lam H + Lam lt H features in use, ao has 2 Lam dq, qa has 4 Lam (dq + dkv). The features
from there up to fillH, fillO, fillQ are read by no claim of this part. A verifier rejects the
public files (`internal-qa-layout`, `plan-tensor-layout`) unless these numbers for qa are those by
which the feed-forward segment places q, kx, v and o (11.2).

*A block.* A **block** of ao or qa is the 2^la = S2 T cell rows of S2 neighbouring features f = S2
bl, .., S2 bl + S2 - 1 in one column; bl is the block's number. Inside a block the cell of feature
S2 bl + lo and token position pos is row lo T + pos (the source's `blockRow`). S2 divides wcO, wcQ
and dh, so a block never crosses a column or a head.

*bA.* The multiplicities of the segment's lookup, in bits: 11.1 (where the multiplicities are), with
lm = lmbA.

*`U3`.* One row of nW elements of E, made of parts; the list of its parts is below. *`ZZ` and `BZ`.*
The digits of the smallness test and the multiplicities of its lookup: 11.1.

##### The calls: tables, keys and lengths

The expanded list of calls, the pieces and the numbers n_c and k_c are in the first half; the rows
of the four kinds of table are in 11.1 (public tables). A call's keys are never sent. A key is an
integer that is made of cells. The lookup states that every key of call c is a row of the table of
call c.

| the call's segment | kind of table | a key's index inside its piece, highest bits first | key bits nb | the key |
|---|---|---|---|---|
| `RM` | `KR`, for sh(lam) | (hk, P, sg, il, jl) | nM | (c0 + 128) + 2^20 (c1 + 128) + 2^40 c2, of the pair in region M |
| `RD` | `KR`, for sh(lam) | (hk, I, il, jl) | nD | the same, of the pair in region D |
| `FM` | `KFD`, for sh(lam) | (hk, P, sg, il, jl) | nM | Df - (c0 + 128) - 256 (c1 + 128) + 2^32 c2 + 2^33 ((c3 + 128) + 256 c4), with Df as below |
| `FD` | `KFD`, for sh(lam) | (hk, I, il, jl) | nD | the same |
| `WL` | `R16` | (sl, hk, i, ch), sl one bit | 1 + lgk + lt + ldh | for sl = 0: rem - 65536 (rh + 128); for sl = 1: 2 Zd(i) - 1 - rem - 65536 (rhp + 128) |
| `WH` | `WH` | (hk, i, ch) | lgk + lt + ldh | (rh + 128) + 2^20 (rhp + 128) |
| `IX` | `R16` | (hk, i) | lgk + lt | 2^(16 - lt) (i - ib(i)), with ib(i) = the sum over k of 2^k ibit_k |

Each piece belongs to one layer lam and one number a, and h = hk HA + a. In the keys of `WL`, rem =
2 acc(i, ch) + Zd(i) - 2 Zd(i) o[lam, h, i, ch]. In the keys of `FM` and `FD`:

      region M, sg <= nT - 2:   Df = Mx(I Bt + il) - sc(I Bt + il, J Bt + jl), with (I, J) the tile of the address (P, sg)
      region M, sg = nT - 1:    Df = C_sh                                      (the empty address)
      region D, jl <= il:       Df = Mx(I Bt + il) - sc(I Bt + il, I Bt + jl)
      region D, jl > il:        Df = Mx(I Bt + il) - sc(I Bt + il, I Bt + jl) + Om_sh      (the causal mask)

Here sc(i, j) is the inner product of step 2 also where j > i.

*What the tables say of a key* (by this page's reading of the tables' rows in the source,
`verifier/src/vf_attention_shape.h`; it describes the code and is not an argument: each step from a row to a
statement on integers needs the seven no-overflow conditions below, and this page does not prove
that they suffice). A row of `KR` fixes c0 + 128 and c1 + 128 as the two bytes of a number rm below
65536, and c2 as the flag of rm >= 2^sh. A row t < u0 of `KFD` is t 2^sh + 2^33 Tt[t]: it fits a key
with flag 0, Df - rm = t 2^sh and ew = Tt[t]. A row t >= u0 has the term 2^32 and no term 2^33: it
fits a key with flag 1, ew = 0 and Df - rm equal to C_sh - 2^sh plus a multiple of rho, or plus such
a multiple and rho / 2. Then 2^sh <= rm < 65536, so Df is at least C_sh. A key of `WL` is in 0 ..
65535 and the bytes of `WH` are in 0 .. 255, so rem and 2 Zd - 1 - rem are each in 0 .. 2^24 - 1. A
key of `IX` in 0 .. 65535 gives 0 <= i - ib(i) < T.

*How a call's keys are ordered.* List the call's pieces by the call's segment, in the order of the
line `callA`; within it by lam, increasing (for the kinds `KR` and `KFD` only the layers with the
call's shift); within it by a, increasing. Sort the list by decreasing nb, and keep the order of
pieces of equal nb. Then place the pieces one after the other from key index 0. off is a piece's
first key index; it is a multiple of 2^nb. The key with the index n inside its piece is key number
off + n of the call. The keys from tot_c on are padding (11.1): row 0 of the table, which is the
number 0, for `KR`, `R16` and `WH`; row u0 of `KFD`, which is C_sh - 2^sh + 2^32.

*A piece's weight and a unit's inner point.* Let zk_c be the key point of call c (11.1), of n_c
coordinates. The **weight** of a piece of call c is the public element

      wp = eq(zk_c[0 .. n_c - nb), off / 2^nb)

The **inner point** of a unit, that is of one of the call's segments in call c (first half), is the
last nb coordinates of zk_c. A sum "over the pieces" of a unit takes them by lam, then by a,
whatever their order in the call; LW is the sum of their weights wp.

##### The points

Every point below is public. A point's coordinate 0 belongs to the highest bit of the index that it
stands for. The challenges of a sumcheck are taken in the order of its rounds: the challenge of the
first `PSC round` is coordinate 0. So every sumcheck of rows 4 to 15 binds the highest bit of its
first index first. The elements of the round `ZERO` of row 9 are used in the order in which they are
sampled, without a reversal.

*From the lookup (row 3).* theta (`LKP shift`), alpha (`LKP`), the points zx and zy, the key point
zk_c and the weight e_c of each call, the form C_end of the key side's leaves, and the two points
zb_0 and zb_1 at which the planes of bA are read: all as the second piece of 11.1 defines them.

*Inner points.* They are cut as follows; the numbers in brackets are numbers of coordinates.

| the call's segment | inner point | its pieces' weights |
|---|---|---|
| `RM`, `RD`, for each unit u of `KR` | Z_u [nM or nD] | wp, LW_u |
| `FM`, for each unit u of `KFD` with `FM` | Z_u = (zg_u [lgk], zP_u [lnT - 1], zs_u [lnT], zi_u [lB], zj_u [lB]) | wp, LW_u |
| `FD`, for each unit u of `KFD` with `FD` | Z_u = (zg_u [lgk], zI_u [lnT], zi_u [lB], zj_u [lB]) | wp, LW_u |
| `WL` | (zsl [1], zgW [lgk], ziW [lt], zcW [ldh]); ziW = (zihi [lnT], zilo [lB]) | wpW_b, LWW |
| `IX` | (zgX [lgk], ziX [lt]) | wpX_b, LWX |
| `WH` | (zgH [lgk], ziH [lt], zcH [ldh]) | wpH_b, LWH |

`WL`, `IX` and `WH` have NP pieces each, one for each b = lam HA + a. A unit u of `KR` or `KFD` has
the pieces of the layers with its call's shift; they are numbered b' = 0, 1, .., np - 1 by (lam, a).
dT >= lmS and dT >= lt + ldh (first half). So the last lmS coordinates of every Z_u are the last lmS
coordinates of zx. And (ziW, zcW) = (ziH, zcH) = the last lt + ldh coordinates of zx.

*Sampled in rows 6 to 14.*

| row | round | names |
|---|---|---|
| 6 | `BAT-sigma`, 2 rows | gam: the sampled element itself (row 0 has the coefficient 1, row 1 has gam) |
| 9 | `ZERO` | in the order sampled: tg [lgk], ti [lt], tk [4], tbg [lgk], tbi [lt] |
| 10 | `BAT-sigma`, NP rows | om_b is the b-th power of the sampled element |
| 12 | `BAT-sigma`, NP rows | om4_b is the b-th power of the sampled element |
| 14 | `BAT-sigma`, NP rows | om5_b is the b-th power of the sampled element |

*The challenges of the sumchecks.*

| row | sumcheck | its challenges, in the order of the rounds |
|---|---|---|
| 4 | K_u, u a unit with `FM` | (rg_u [lgk], rP_u [lnT - 1], rs_u [lnT], rc_u [ldh]) |
| 4 | K_u, u a unit with `FD` | (rg_u [lgk], rI_u [lnT], rc_u [ldh]) |
| 5 | SE | (rgE [lgk], riE [lt]) |
| 7 | VM | (rgM [lgk], rPM [lnT - 1], rsM [lnT], rjM [lB]) |
| 8 | VD | (rgD [lgk], rID [lnT], rjD [lB]) |
| 11 | SF | (riF [lt], rgF [lgk], rcF [ldh]) |
| 13 | SD | (pj [lt], pg [lgk], pi [lt]) |
| 15 | HB | (ri2 [lt], rg2 [lgk], rk [4]) |

Two more points are put together from these: ptM = (rgM, rPM, rsM, zilo, rjM), of nM coordinates,
and ptD = (rgD, rID, zilo, rjD), of nD coordinates.

##### The scalars by name

The scalars of rows 3 to 15 in the order sent, which is the order of their numbers. "Alone" means a
hidden value sent alone (11.1). The last column says what an honest prover's hidden value is.

| row | scalars, in order | kind | the hidden value |
|---|---|---|---|
| 3 | the lookup's | 11.1 | 11.1 |
| 4 | for each unit u of `KR`, in order: R_(u,0), R_(u,1), R_(u,2) | alone | R_(u,x) = the sum over the unit's pieces of wp times the sum over n of eq(Z_u, n) c_x(lam, a, n): plane x of as at the inner point |
| 4 | then for each unit u of `KFD`, in order: PV_(u,0), .., PV_(u,4), MV_u, SV_u | alone | PV_(u,x): plane x of as at Z_u, by the formula of R_(u,x). MV_u = the sum over the unit's pieces of wp times the sum over n of eq(Z_u, n) Mx(i of the pair n); SV_u the same with sc(i, j) of the pair n, also where jl > il. At the empty address both summands are 0 |
| 4 | directly after SV_u: the scalars of K_u | sumcheck [3, .., 3 \| 2 \| 2 np] | end values AE_(u,0), BE_(u,0), AE_(u,1), BE_(u,1), .. |
| 5 | ACC, ZO, ZC, RH, RHP | alone | ACC, ZO, RH and RHP are acc, Zd o, rh and rhp at (zgW, ziW, zcW); ZC is Zd at (zgW, ziW). Each is summed over the pieces of `WL` with the weights wpW_b |
| 5 | IB_0, .., IB_(lt - 1) | alone | ibit_k at (zgX, ziX), summed over the pieces of `IX` with the weights wpX_b |
| 5 | RH2, RHP2 | alone | rh, rhp at (zgH, ziH, zcH), summed over the pieces of `WH` with the weights wpH_b |
| 5 | the scalars of SE | sumcheck [3, .. \| 2 \| 2 NP] | end values AE_(SE,b), BE_(SE,b) |
| 7 | CL0, CL1 | alone | the sums of VM and of VD |
| 7 | the scalars of VM | sumcheck | end values AE_(VM,b), BE_(VM,b) |
| 8 | the scalars of VD | sumcheck | end values AE_(VD,b), BE_(VD,b) |
| 10 | M6_0, .., M6_(NP - 1) | alone | M6_b = the sum over hk and i of eq(tg, hk) eq(ti, i) Mx(i) of layer lam and head hk HA + a |
| 11 | the scalars of SF | sumcheck | end values AE_(SF,b), BE_(SF,b) |
| 13 | the scalars of SD | sumcheck | end values KC_0, .., KC_(Lam - 1), then IBP_(b,k) as number Lam + b lt + k among the end values |
| 15 | the scalars of HB | sumcheck | end values AE_(HB,b), BE_(HB,b) |

In every sumcheck but SD, the end values come as AE_0, BE_0, AE_1, BE_1, ..: AE_b is number 2 b and
BE_b number 2 b + 1 among the end values.

##### The parts of `U3`

`U3` is made of these parts, one after the other from entry 0, in this order. A part of `U3` has a
matrix and a public weight vector phi of L entries (11.1). The smallness test's reads tie U(p; n) to
the cells (11.1, third piece). What the entry must be is:

- for a part on as: L = 2^lmS; n is a column of as, counted through all NAs parts; U(p; n) = the sum
  over the cell rows r of phi[r] cell(r, n). In the terms of 11.1 these are NAs parts of `U3`, one
  for each part of as, side by side; here they are one part of kAs entries.
- for a part on ah: L = T; n is a feature; U(p; n) = the sum over pos < T of phi[pos] cell(feature
  n, token position pos).
- for a part on ao or qa: L = 2^la; n is a block; U(p; n) = the sum over lo < S2 and pos < T of
  phi[lo T + pos] cell(feature S2 n + lo, token position pos).
- for a part on bA: L = 2^lmbA; n is a column; U(p; n) = the sum over the cell rows r of phi[r]
  cell(r, n).

Three ways to make a weight vector are used. EQ(z) is the vector with EQ(z)[n] = eq(z, n). For a
vector Wv of nT entries and a point z of lB coordinates, tok(Wv, z) has T entries: tok(Wv, z)[I Bt +
il] = Wv[I] eq(z, il). For a point zc of ldh coordinates and a vector tv of T entries, lng(zc, tv)
has 2^la entries: lng(zc, tv)[lo T + pos] = eq(zc[ldh - ls .. ldh), lo) tv[pos].

| no. | part | matrix | entries | weight vector phi |
|---|---|---|---|---|
| 1 | tA | as | kAs | EQ(the last lmS coordinates of zx) |
| 2 | tVM | as | kAs | EQ(the last lmS coordinates of ptM) |
| 3 | tVD | as | kAs | EQ(the last lmS coordinates of ptD) |
| 4 | tM_u, for each unit u of `KFD`, in order | ah | fillH | `FM`: tok(WI(zP_u, zs_u), zi_u); `FD`: tok(EQ(zI_u), zi_u) |
| 5 | hW | ah | fillH | EQ(ziW) |
| 6 | hX | ah | fillH | EQ(ziX) |
| 7 | hE | ah | fillH | EQ(riE) |
| 8 | hT | ah | fillH | EQ(ti) |
| 9 | hPi | ah | fillH | EQ(pi) |
| 10 | hR2 | ah | fillH | EQ(ri2) |
| 11 | tO | ao | fillO / S2 | lng(zcW, EQ(ziW)) |
| 12 | for each unit u of `KFD`, in order: tQ_u, then tK_u | qa | fillQ / S2 each | `FM`: tQ_u: lng(rc_u, tok(WI(rP_u, rs_u), zi_u)); tK_u: lng(rc_u, tok(WJ(rP_u, rs_u), zj_u)). `FD`: tQ_u: lng(rc_u, tok(EQ(rI_u), zi_u)); tK_u: lng(rc_u, tok(EQ(rI_u), zj_u)) |
| 13 | qO | qa | fillQ / S2 | lng(zcW, EQ(riE)) |
| 14 | qVM | qa | fillQ / S2 | lng(zcW, tok(WJ(rPM, rsM), rjM)) |
| 15 | qVD | qa | fillQ / S2 | lng(zcW, tok(EQ(rID), rjD)) |
| 16 | qF | qa | fillQ / S2 | lng(rcF, EQ(riF)) |
| 17 | qKx | qa | fillQ / S2 | lng(rcF, EQ(pj)) |
| 18 | tB_0 | bA | kbA | phiB_0, the lookup's weight vector for the planes of bA at zb_0 (11.1, second piece) |
| 19 | tB_1 | bA | kbA | phiB_1, the same at zb_1 |
| 20 | tP | bA | kbA | phiP: phi[r] = 2^(r mod 32) |

The lengths add up to nW. All weight vectors are fixed before the root of `U3` is sent (row 16). The
order of this table is also the order of the reads within a matrix (11.1, the order of the smallness
test's claims).

##### Four readings

The claims read `U3` only through these four affine forms in its entries.

      PL(p; x, lam, a; Z, M) = the sum over n < 2^(nM - lmS) of eq(Z[0 .. nM - lmS), n) U(p; ((x Lam + lam) HA + a) 2^(nM - lmS) + n)
      PL(p; x, lam, a; Z, D) = the sum over n < 2^(nD - lmS) of eq(Z[0 .. nD - lmS), n) U(p; 5 NP 2^(nM - lmS) + ((x Lam + lam) HA + a) 2^(nD - lmS) + n)
      LV(p; Tn; lam, f') = the sum over l < nl of 256^l U(p; off + (l Lam + lam) nf + f')
      XL(p; Tn; lam, hd; zc) = the sum over n < 2^(ldh - ls) of eq(zc[0 .. ldh - ls), n) times the sum over l < nl of 256^l U(p; (off + (l Lam + lam) nf + hd dh) / S2 + n)

PL reads plane x of as, for a piece of region M or D, from a part p on as, at a point Z of nM or nD
coordinates whose last lmS coordinates are those of the part's weight vector. LV reads entry f' of
the tensor Tn = (off, nl, Lam, nf) of ah from a part p on ah. XL reads head hd of the tensor Tn of
ao or qa from a part p on that matrix, at the channel point zc; the division by S2 is exact. For a
point z of lgk coordinates and a form F(hk), OG(z; F) = the sum over hk < G of eq(z, hk) F(hk).

##### The eight kinds of sumcheck of rows 4 to 15

The eight kinds are K_u for a unit with `FM`, K_u for a unit with `FD`, SE, VM, VD, SF, SD and HB.
Each is a sumcheck with a hiding sum (11.1) whose rounds are all plain. All but SD have the same
shape: the sum over b and over the points of a cube of W_b A_b B_b, with W_b public and multilinear
and A_b, B_b hidden and multilinear, so degree 3 in each variable. The verifier needs only three
things of a sumcheck: its start form, the public weights at the challenge point, and S0, the summand
at the challenge point computed on the end values as they were sent. For the sumchecks of this shape

      S0 = the sum over b of W_b shown(AE_b) shown(BE_b)

with W_b from the table. The columns A_b and B_b say what the end values AE_b and BE_b are the
values of, at the challenge point; the claims below tie them to the cells.

| sumcheck | start form | cube, highest bits first | W_b at the challenge point | A_b | B_b |
|---|---|---|---|---|---|
| K_u, `FM` | [SV_u] | (hk, P, sg, ch) | wp_b eqp(zg_u, rg_u) eqp(zP_u, rP_u) (eqp(zs_u, rs_u) - ones(zs_u) ones(rs_u)) | the sum over il of eq(zi_u, il) q[lam, h, I(P, sg) Bt + il, ch] | the sum over jl of eq(zj_u, jl) kx[lam, hk, J(P, sg) Bt + jl, ch] |
| K_u, `FD` | [SV_u] | (hk, I, ch) | wp_b eqp(zg_u, rg_u) eqp(zI_u, rI_u) | the sum over il of eq(zi_u, il) q[lam, h, I Bt + il, ch] | the sum over jl of eq(zj_u, jl) kx[lam, hk, I Bt + jl, ch] |
| SE | [ZO] | (hk, i) | wpW_b eqp(zgW, rgE) eqp(ziW, riE) | Zd(i) of (lam, h) | the sum over ch of eq(zcW, ch) o[lam, h, i, ch] |
| VM | [CL0] | (hk, P, sg, jl) | wpW_b eqp(zgW, rgM) times the sum over I' < nT of eq(zihi, I') WI(rPM, rsM)[I'] | the sum over il of eq(zilo, il) ew of the pair (hk, P, sg, il, jl) of region M | gam + the sum over ch of eq(zcW, ch) v[lam, hk, J(P, sg) Bt + jl, ch] |
| VD | [CL1] | (hk, I, jl) | wpW_b eqp(zgW, rgD) eqp(zihi, rID) | the sum over il of eq(zilo, il) ew of the pair (hk, I, il, jl) of region D | gam + the sum over ch of eq(zcW, ch) v[lam, hk, I Bt + jl, ch] |
| SF | the sum over b of om_b [M6_b] | (i, hk, ch) | om_b eqp(tg, rgF) eqp(ti, riF) | q[lam, h, i, ch] | kx[lam, hk, ib(i), ch], with ib of (lam, h) |
| HB | 0 | (i, hk, k), k of 4 bits | om5_b eqp(tbi, ri2) eqp(tbg, rg2) eqp(tk, rk) | ibit_k at token i of (lam, h) for k < lt, and 0 for k >= lt | A_b - 1 |

In K_u the index b runs over the unit's np pieces, and (lam, a) are the piece's. In the others b =
lam HA + a runs to NP - 1. In the weight of K_u for `FM` the term ones(zs_u) ones(rs_u) takes the
empty address out of the sum. At the empty address A_b and B_b of K_u are 0, because WI and WJ leave
that address out; B_b of VM is gam there; A_b of VM is what the cells of the empty address hold, and
the weight of VM is 0 there.

*The sumcheck SD.* It starts from the sum over b of om4_b [BE_(SF,b)], the second end values of SF.
It is about the sum, over (j, hk, i), of

      eq(riF, i) eq(rgF, hk) times the sum over b of om4_b KC_lam(j, hk) times the product over k < lt of eq1(ibit_k(lam, h, i), bit k of j)

with lam = floor(b / HA), h = hk HA + (b mod HA), KC_lam(j, hk) = the sum over ch of eq(rcF, ch)
kx[lam, hk, j, ch], and eq1(s, y) = s y + (1 - s)(1 - y). The product is 1 exactly if j = ib(i). Its
rounds bind the lt bits of j (degree 2), then the lgk bits of hk (degree lt + 2), then the lt bits
of i (degree lt + 1). With y_k = pj[lt - 1 - k], the coordinate of bit k of j,

      S0 = eqp(riF, pi) eqp(rgF, pg) times the sum over b of om4_b shown(KC_(floor(b / HA))) times the product over k < lt of (shown(IBP_(b,k)) (2 y_k - 1) + 1 - y_k)

*The claim that a sumcheck of this part leaves.* It is Left(C; S0; 0) = 0 of the second piece of
11.1, with the start form C and the S0 given above and the kept form 0. Write LEFT(name) for this
claim. For a sumcheck with n plain rounds of the degrees k_0, .., k_(n-1) and then the variable Y it
reads as follows. j0 is the number of the sumcheck's first scalar, the hiding polynomial's sum.
sk(l) = k_0 + .. + k_(l-1); here l is a variable's number, as in 11.1. gamma_0, .., gamma_(n-1) are
the challenges in the order of the rounds; gamma_l is not the element gam of row 6. eps and beta are
as in 11.1. cur is the running claim after the round `PSC Y`, that is the value at beta of that
round's polynomial.

      the sum over l < n and t = 1 .. k_l of  gamma_l^t (gamma_(l+1) gamma_(l+2) .. gamma_(n-1)) mask(j0 + sk(l) + t - 1)
      + the sum over t = 0 .. d_Y of  beta^t mask(j0 + sk(n) + t)
      + eps (gamma_0 gamma_1 .. gamma_(n-1)) (pub(C) - C)
      + eps S0 - cur  =  0

pub(C) is C with every [X] replaced by shown(X). So pub(C) - C holds mask entries alone: each [X] of
the start form gives its coefficient times the mask entry of the number of X, and times beta of the
earlier sumcheck if X is an end value. The first end value has the number jE = j0 + 1 + sk(n) + d_Y.

##### The claims, in order

Claim number 0 is the first. Every line of the list is one claim "form = 0", and a line "for each
.." gives its claims in the stated order. The list has ten groups of claims, group 1 to group 10
(these are not groups of layers). The count of each group is at its head. The sum of the counts is
NCm (first half).

*Group 1: the lookup's own claims.* 32 + v_A + d_A claims, LK0 to LK(31 + v_A + d_A) of the second
piece of 11.1: 31 of the tie, v_A of the key side's product levels, d_A of the table side's product
levels, 1 of the sumcheck "the bits are bits", in this order.

*Group 2: the planes of as for the units of `KR`.* 3 nKR claims. For each call of kind `KR` in the
order of the expanded list; within it for the unit with `RM`, then the unit with `RD`; within it for
x = 0, 1, 2:

      [R_(u,x)] - the sum over the unit's pieces of wp PL(tA; x, lam, a; Z_u, reg)  =  0

reg is M for `RM` and D for `RD`.

*Group 3: the units of `KFD`.* 7 nKF + 4 NP claims. For each call of kind `KFD` in the order of the
expanded list; within it for the unit u with `FM`, then the unit with `FD`:

1. for x = 0, .., 4:  [PV_(u,x)] - the sum over the unit's pieces of wp PL(tA; x, lam, a; Z_u, reg)
   = 0
2. [MV_u] - the sum over the unit's pieces of wp OG(zg_u; hk -> LV(tM_u; Mx; lam, hk HA + a)) = 0
3. LEFT(K_u)
4. for each piece b' = 0, .., np - 1, two claims, in this order:
   - [AE_(u,b')] - OG(rg_u; hk -> XL(tQ_u; q; lam, hk HA + a; rc_u)) = 0
   - [BE_(u,b')] - OG(rg_u; hk -> XL(tK_u; kx; lam, hk; rc_u)) = 0

*Group 4: the division.* 6 + lt + 2 NP claims, in this order:

1. [ZC] - the sum over b of wpW_b OG(zgW; hk -> LV(hW; Zd; lam, hk HA + a)) = 0
2. [RH] - the sum over b of wpW_b OG(zgW; hk -> XL(tO; rh; lam, hk HA + a; zcW)) = 0
3. [RHP] - the sum over b of wpW_b OG(zgW; hk -> XL(tO; rhp; lam, hk HA + a; zcW)) = 0
4. for k = 0, .., lt - 1:  [IB_k] - the sum over b of wpX_b OG(zgX; hk -> LV(hX; ibit; lam, k H + hk
   HA + a)) = 0
5. [RH2] - the sum over b of wpH_b OG(zgH; hk -> XL(tO; rh; lam, hk HA + a; zcH)) = 0
6. [RHP2] - the sum over b of wpH_b OG(zgH; hk -> XL(tO; rhp; lam, hk HA + a; zcH)) = 0
7. LEFT(SE)
8. for b = 0, .., NP - 1, two claims, in this order:
   - [AE_(SE,b)] - OG(rgE; hk -> LV(hE; Zd; lam, hk HA + a)) = 0
   - [BE_(SE,b)] - OG(rgE; hk -> XL(qO; o; lam, hk HA + a; zcW)) = 0

*Group 5: the join.* 1 claim, LJ of the second piece of 11.1. It ties the keys of all calls to the
lookup's key side:

      the sum over the calls c of e_c (alpha - c theta - KF_c)  +  (alpha - padkey_0) times the sum over y = NT .. 2^rK - 1 of eq(zy, y)  -  C_end  =  0

c is a call's number in the expanded list, from 0, and padkey_0 is the padding key of call 0. KF_c,
the **key form** of call c, is the extension of the call's key vector at zk_c, as an affine form in
hidden values:

- c of kind `KR`:  KF_c = the sum over the call's units u of  ([R_(u,0)] + 128 LW_u) + 2^20
  ([R_(u,1)] + 128 LW_u) + 2^40 [R_(u,2)]
- c of kind `KFD`, with the shift sh:  KF_c = (1 - wall_c)(C_sh - 2^sh + 2^32) + the sum over the
  call's units u of

      DP_u - ([PV_(u,0)] + 128 LW_u) - 256 ([PV_(u,1)] + 128 LW_u) + 2^32 [PV_(u,2)] + 2^33 ([PV_(u,3)] + 128 LW_u + 256 [PV_(u,4)])
      DP_u = [MV_u] - [SV_u] + C_sh LW_u ones(zs_u)                       for `FM`
      DP_u = [MV_u] - [SV_u] + Om_sh LW_u (1 - LE(zj_u, zi_u))            for `FD`

  wall_c is the sum of the weights wp of all pieces of the call.
- c of kind `R16`:  KF_c = tWL, if the call has `WL`, plus tIX, if it has `IX`, with

      REM = 2 [ACC] + [ZC] - 2 [ZO]
      tw0 = REM - 65536 ([RH] + 128 LWW)
      tw1 = 2 [ZC] - LWW - REM - 65536 ([RHP] + 128 LWW)
      tWL = (1 - zsl) tw0 + zsl tw1
      tIX = 2^(16 - lt) (LWX id(ziX) - the sum over k < lt of 2^k [IB_k])

- c of kind `WH`:  KF_c = ([RH2] + 128 LWH) + 2^20 ([RHP2] + 128 LWH)

The key forms of `KR`, `R16` and `WH` have no term for the padding, because their padding key is 0.

*Group 6: the weighted sum and the denominator.* 3 + 4 NP claims, in this order:

1. [CL0] + [CL1] - [ACC] - gam [ZC] = 0
2. LEFT(VM)
3. for b = 0, .., NP - 1, two claims, in this order:
   - [AE_(VM,b)] - PL(tVM; 3, lam, a; ptM, M) - 128 - 256 PL(tVM; 4, lam, a; ptM, M) = 0
   - [BE_(VM,b)] - gam - OG(rgM; hk -> XL(qVM; v; lam, hk; zcW)) = 0
4. LEFT(VD)
5. for b = 0, .., NP - 1, two claims, in this order:
   - [AE_(VD,b)] - PL(tVD; 3, lam, a; ptD, D) - 128 - 256 PL(tVD; 4, lam, a; ptD, D) = 0
   - [BE_(VD,b)] - gam - OG(rgD; hk -> XL(qVD; v; lam, hk; zcW)) = 0

*Group 7: the maximum is a score of its row.* 2 + 2 NP + Lam + NP lt claims, in this order:

1. for b = 0, .., NP - 1:  [M6_b] - OG(tg; hk -> LV(hT; Mx; lam, hk HA + a)) = 0
2. LEFT(SF)
3. for b = 0, .., NP - 1:  [AE_(SF,b)] - OG(rgF; hk -> XL(qF; q; lam, hk HA + a; rcF)) = 0
4. LEFT(SD)
5. for lam = 0, .., Lam - 1:  [KC_lam] - OG(pg; hk -> XL(qKx; kx; lam, hk; rcF)) = 0
6. for b = 0, .., NP - 1, and within it for k = 0, .., lt - 1:  [IBP_(b,k)] - OG(pg; hk -> LV(hPi;
   ibit; lam, k H + hk HA + a)) = 0

The end values BE_(SF,b) have no claim on cells: they are the start form of SD.

*Group 8: the index's cells are bits.* 1 + 2 NP claims, in this order:

1. LEFT(HB)
2. for b = 0, .., NP - 1, two claims, in this order:
   - [BE_(HB,b)] - [AE_(HB,b)] + 1 = 0
   - [AE_(HB,b)] - the sum over k < lt of eq(rk, k) OG(rg2; hk -> LV(hR2; ibit; lam, k H + hk HA +
     a)) = 0

*Group 9: the lookup's three claims on the planes of bA.* 3 claims, LP0, LP1 and LP2 of the second
piece of 11.1. LP0 is the claim that the sumcheck over the table side's leaves has left, with the
value of the planes of bA at zb_0 read from tB_0. LP1 is on the value of the planes of bA at zb_1,
which the sumcheck "the bits are bits" has shown; that value is read from tB_1. LP2 is that all
multiplicities, read from tP, add up to the number of keys that are not padding, 2^v_A less all
padding keys. The three forms that the lookup takes from its host are TB_0(col) = U(tB_0; col),
TB_1(col) = U(tB_1; col) and TP(col) = U(tP; col), for each column col of bA.

*Group 10: the smallness test.* (32 + v_Z + 11) + 4 + (NAs + 4) + NX claims, in the order of 11.1,
on the list of matrices as (its parts), ah, ao, bA, qa and on the parts of `U3` in the order of the
table above.

In groups 4, 6, 7 and 8, lam = floor(b / HA) and a = b mod HA.

*The count.* Without the nKF + 6 claims LEFT(..), the groups 1 to 9 have (32 + v_A + d_A) + 3 nKR +
(6 nKF + 4 NP) + (5 + lt + 2 NP) + 1 + (1 + 4 NP) + (4 NP + Lam + NP lt) + 3 claims: these are the
terms of the first half's formula for NCm, in its order. Groups 7 and 8 together are its term 4 NP +
Lam + NP lt.

*What the lookup and the smallness test take from this part.* The segment's own lookup is the first
thing that sends scalars, so jLK = 0. Its calls are the expanded list; its tables are those of 11.1;
KF_c, TB_0, TB_1 and TP are given above. The padding positions of call c are the key numbers tot_c,
.., 2^(n_c) - 1, one run at the end. So tree number m of call c, m = 0, .., 2^(n_c - dT) - 1, which
is tree y = tree0(c) + m, has npad(y) = min(2^dT, max(0, (m + 1) 2^dT - tot_c)) padding positions.
Among the segment's claims, LK0 has the number 0; LJ has the number nLJ = (32 + v_A + d_A) + 3 nKR +
(7 nKF + 4 NP) + (6 + lt + 2 NP); LP0 has the number nLJ + 1 + (3 + 4 NP) + (2 + 2 NP + Lam + NP
lt) + (1 + 2 NP). The smallness test's first claim has the number n0 = that number + 3. The lookup
on the digits has as its jLK the number of the scalars of rows 3 to 15.

*The programs' self-checks.* The programs make some tests that do not depend on the proof, their
self-checks: they compare two ways in which the same program computes one public thing. By this
page's reading of the source no public file that has passed the earlier tests can make one of them
fail either, and a verifier that is written from this page need not have them.

- Before the join: every call has exactly one key form (`af-unclosed-call`; the name does not begin
  with `internal-`, so a call of `cp_verify` that ended under it would end with a rejection and not
  with `NOT CHECKED`); the padding key of every call that is not of kind `KFD` is 0
  (`internal-af-padding-key`); and the key indices that no key form weighs are, tree by tree, the
  numbers of padding keys that the lookup uses (`internal-forest-padding-places`).
- While the parts of `U3` are listed: each part has as many entries as its matrix has columns times
  blocks (`internal-a-table-length`), the parts add up to nW (`internal-af-W-length`), and the
  shared coordinates of the inner points are equal to those of zx (`internal-a-merged-point`).
- After the last scalar: the number of scalars read is the planned one (`internal-slot-count`).
- After the last claim: the number of claims is NCm (`internal-a-claim-count`), and every scalar's
  number has a coefficient other than 0 in the sum of all claims (`internal-a-free-slot`). The
  outcome of the last test alone depends on sampled values. It is not a self-check but a test on
  content of every verifier: where a scalar's coefficient is 0, a verifier gives no acceptance
  (13.9, event 9, which also has what is known of its chance). The source's comment
  (`verifier/src/vf_attention_check.h`) calls it a self-check: "every message slot must be in a claim" (a
  message slot is a scalar's number). Read so, it is a test of the program's own list of claims, not
  of the proof. By the same comment the coefficient is 0 if no claim has the number, and otherwise
  only with a probability for which the comment gives a bound; this second case is the one that
  depends on sampled values. The comment adds that the test guards hiding: that every scalar is in a
  claim is what lets whoever verifies a proof before it is handed on see an end value that was sent without its mask. An attention segment has
  a smallness test like every segment (11.1), so events 2 to 8 of 13.9, which are that test's, can
  happen in it too, and a verifier gives no acceptance at them either.

Further self-checks, of which a second verifier needs none, since by this page's reading of the
source no public file that has passed the earlier tests, and no proof, can make one fail: `hid-hb`
where the segment's plan is made (the line `sec` less the first number of the line `work positions`
is not 0; a parameter file in which that number is not below `sec` has been rejected before),
`hid-a-owner-uses`, `hid-a-weight-matrix`, `internal-a-TX-place`, `internal-a-tree-list`,
`plan-link` where the segment's run begins, `internal-a-Ty` (Ty is 2^18), `internal-a-claim-count` a
first time, where the plan is made (the smallness test's list has NAs + 4 matrices),
`internal-af-calls`, `internal-sc3-` and the programs' name of a sumcheck (the call's name for K_u;
`e`, `VPM`, `VPD`, `f` and `h` for SE, VM, VD, SF and HB), `internal-psc-order`,
`internal-psc-degree`, `plan-tensor-layout` (while the weight vectors and the readings are formed),
`internal-f-draws`, `internal-af-rows-booked`, `internal-a-calls-booked`. This holds
for an attention segment's plan and run alone and, where the list gives a place, for that place
alone. Three of the names without a place, `internal-psc-order`, `internal-psc-degree` and
`internal-f-draws`, are also raised from code that other segments run; this part
says nothing of them there. Under `plan-link` and `plan-tensor-layout` the programs also make tests
on the public files, which every verifier makes (11.1, third piece; the table of this part's tests
on the public files; 13.8, the rows of these two names).

Thirteen names of this part are printed after the sub-proof's prefix (13.8), which for an attention
segment is `A`, the group's number g in decimal digits and `:`. They are `af-unclosed-call`,
`internal-af-padding-key`, `internal-a-table-length`, `internal-a-merged-point`,
`internal-slot-count`, `internal-a-claim-count` after the last claim, `internal-a-free-slot`,
`internal-a-table-index` (below), `internal-psc-order`, `internal-psc-degree`, `internal-f-draws`
(but for two places in the smallness test, where it is printed without a prefix),
`internal-af-rows-booked` and `internal-a-calls-booked`. The other names of this part are printed
without a prefix.

##### What is public in a claim

Every coefficient of a hidden value or of an entry of `U3` in the claims above is public, and so is
every term without a hidden value. The public terms come from four sources. (1) The 128 that a cell
leaves out: the extension of the constant 128 over a unit's pieces at the key point is 128 LW. The
same holds for the -1 in the second key of `WL` (the term - LWW) and for the position i in the key
of `IX` (the term LWX id(ziX)). In the claims on AE_(VM,b) and AE_(VD,b) the sumcheck runs over one
piece, so the constant is 128 itself. (2) The constants of the keys of `FM` and `FD`: C_sh at the
empty address, whose indicator has the extension ones(zs_u), and Om_sh at the pairs with jl > il,
whose indicator has the extension 1 - LE(zj_u, zi_u). (3) The padding: (1 - wall_c) times the
padding key in the key form of `KFD`, and the term with padkey_0 in the join. (4) What a sumcheck
leaves: eps S0 - cur.

##### Where the causal mask is

No claim has the mask as its only subject. Four things together make it, and all four are needed.

1. *Tiles above the diagonal do not exist.* Region M has addresses only for tiles with J < I, and
   region D only the tiles with J = I. So no score, weight or key exists for a pair in a tile with
   J > I. The map from addresses to tiles enters the claims through the public vectors WI and WJ: in
   tM_u, tQ_u, tK_u, qVM and in the weight of VM.
2. *Inside a tile on the diagonal* all Bt^2 pairs have cells. The key of `FD` adds Om_sh = 2 Smax +
   C_sh to Df where jl > il. In the key form this is the public term Om_sh LW_u (1 - LE(zj_u, zi_u))
   of DP_u. The source's comments call Om the causal mask's margin (`verifier/src/vf_attention_shape.h`) and
   say that it is added where j > i (`verifier/src/vf_attention_check.h`, at the claims for `KFD`). That it
   forces the weight of these cells to 0 is this page's reading of the source. In numbers: if q and
   kx are in the clamp's range (11.2) and Mx(i) is a score of its row (groups 7 and 8), then Mx - sc
   is at least -2 Smax, so Df is at least C_sh. A row t < u0 of `KFD` fits only a key with flag 0,
   so with rm < 2^sh (the key of `RD`), and with Df - rm = t 2^sh <= C_sh - 2^sh, so with Df < C_sh.
   So only rows t >= u0 fit, and they have the weight 0. This step from an equation in the field to
   one in the integers is what N2, N3 and N4 are for. This page does not prove it.
3. *The sums take every cell of both regions.* VD runs over all jl of a diagonal tile, and VM over
   all addresses but the empty one. So Zd and acc are sums over the cells' weights, and a pair with
   j > i adds what its cells hold, which point 2 is meant to fix to 0.
4. *The maximum's position.* The key of `IX`, 2^(16 - lt) (i - ib(i)), must be a row of the range
   table of 16 bits. By this page's reading of the source its purpose is that ib(i) <= i; the
   source's comment at this key gives no purpose. Of Mx(i), which group 7 ties to sc(i, ib(i)), the
   comments in `verifier/src/vf_attention_check.h`, which write m and ix for Mx and ib, say that it is the
   row's maximum and that it is attained at that index. That the two together make Mx(i) a score of
   an allowed pair is this page's wording.

The empty address is treated like a masked pair: its Df is the constant C_sh, its cells are read by
the keys of `RM` and `FM`, and the weights of K_u and VM leave it out.

##### Cells that fit

A verifier does not need this heading. It gives, for each Df, one choice of cells of as that fits
the tables; this follows from the tables' rows. If Df < C_sh (and Df >= 0): rm = Df mod 2^sh, fl =
0, ew = Tt[floor(Df / 2^sh)]; the key is row floor(Df / 2^sh) of `KFD`. If Df >= C_sh: fl = 1, ew =
0 and rm = 2^sh + ((Df - C_sh) mod rho); the key is row u0 + floor((Df - C_sh) / rho). With q and kx
in the clamp's range Df is at most 4 Smax + C_sh, so that row is below u0 + nHD. The rows from u0 +
nHD on, at most nHD of them, are not needed for this choice. They are rows like the others: for some
Df >= C_sh a second rm fits through them, with the same fl and ew. At the empty address Df = C_sh,
so rm = 2^sh, fl = 1, ew = 0.

##### Cells and entries that no claim of this part reads

The sizes are rounded up, so some committed cells have no meaning. No claim of groups 1 to 9 reads:
a cell of as with a number from 5 NP (2^nM + 2^nD) on; a feature of ah from 8 Lam H + Lam lt H on,
of ao from 2 Lam dq on, of qa from 4 Lam (dq + dkv) on; an entry of a part of `U3` that belongs to
such a cell. The smallness test covers these cells like all others. The programs' verifier ends the
call under the name `internal-a-table-index` (`NOT CHECKED`, 14.3) if one of its own claims names an
entry at or after the end of a part of `U3`; it has no test that an index is below the number of
entries in use.

##### The tests on the public files

All these tests use the three public files alone. A verifier makes them before it reads a proof, and
it rejects with the name given. They are listed in the order in which the programs make them for one
group of layers: the rows from `af-policy-wd` to `plan-tensor-layout` are made group by group, all
of them for one group before the first of them for the next, and the last test of the first row is
made there too, after `af-policy-wd`. The tests of 11.1 on the lookup's numbers come after
`af-shape-call-`, and the tests of 11.1 on the smallness test's list come between `hid-a-fold-list`
and `hid-a-slots`. Ty = 2^18 is the bound that the conditions assume for the absolute value of a
committed cell (twice the range 2^17 of the smallness test's combinations); Ty2 = 257 Ty and Ty4 =
16,843,009 Ty bound a value of two and of four limbs. TS = T 2^Sx.

*Shape and policy.*

| name | a verifier rejects unless |
|---|---|
| `policy-call-list-A` | there are from 1 to 12 lines `callA`; every kind is `KR`, `KFD`, `R16` or `WH`; no two lines have the same name; every further word is one of `RM`, `RD`, `FM`, `FD`, `WL`, `WH`, `IX`; and, for every group, the expanded list has at most 12 calls |
| `af-shape-heads` | G is a power of two |
| `shape-request-matrix-A` | lt <= lmh <= 24; lt <= lmo <= 24; lmbA <= 24; lmS <= 24; 1 <= NAs <= 64 |
| `af-policy-wd` | Sx < 32 |
| `af-shape-tiles` | 2 <= lt <= 16 and 0 <= lB <= lt - 2 |
| `af-shape-heads` | H >= 1 and G divides H |
| `af-shape-sh` | the group has exactly Lam shifts (in `cp_verify` this cannot fail: the program takes exactly Lam words of the scale file for a group) |
| `af-shape-la` | 0 <= la - lt <= ldh and la <= 16 |
| `af-policy-softmax` | 1 <= Sx <= 14; 1 <= rx <= 12; lt + Sx + 1 <= 24 (the source: a remainder of the division has three bytes); and Tt reaches 0 at some u0 <= 2^20 + 1 |
| `af-shape-lmS` | 1 <= lmS <= nD |
| `af-policy-kF` | 8 <= kFD <= 16 |
| `af-policy-sh` | 4 <= sh <= 12, for each shift sh of the group |
| `af-policy-kF` | u0 + nHD <= 2^kFD <= u0 + 2 nHD, for each shift |
| `af-policy-C` | C_sh + 65536 <= 2^24, for each shift |
| `af-no-wrap-N3`, `af-no-wrap-N4` | below, for each shift; the five tests on a shift (this row and the three before it) are made shift by shift, in increasing order |
| `af-no-wrap-N1`, `-N2`, `-N5`, `-N6`, `-N9` | below, in this order |
| `af-policy-call-list` | every line `callA` has at least one call's segment; each of the seven is in exactly one line, and in a line of its kind (first half) |
| `af-policy-nmin` | nmin >= 10 |
| `af-shape-la` | 2^la <= 2^lmo; 2^la <= m_qa; S2 divides wcO and wcQ |
| `af-policy-call-list` | the expanded list has at most 12 calls (it cannot fail after the last test of the first row) |
| `af-shape-call-` and the call's name | 10 <= n_c <= 44 |
| `forest-depth-A` | dT >= lmS and dT >= lt + ldh |
| `forest-planes` | 5 <= lmbA <= kT + 5 |
| `forest-only-padding-` and the call's name | not all keys of a call are padding (11.1) |
| `hid-a-fold-list` | NAs + 4 <= 256 (it cannot fail after `shape-request-matrix-A`) |
| `hid-a-slots` | the segment has fewer than 2^30 scalars |
| `policy-iotaA` | the line `iotaA_0` is 32 + dT, for every group |
| `internal-qa-layout`, `plan-tensor-layout` | fillQ, the number of features of qa in use, the four tensors of qa, Lam, dq, dkv and ls are those of the feed-forward segment's own layout, and S2 divides dh |
| `table-digest-exp` | if the lines `texp0` to `texp7` are present: the digest of Tt is theirs (11.1, 12.6) |

*No overflow.* There are seven conditions. The names N7 and N8 are not in use. The two constants
2^60 and 2^61 - 1 are written into the conditions as numbers: they are the same whatever the field,
and for field 2, the one field for which the programs of this repository can be built (R1.2), they
are below half the field's prime and below the prime.

| name | the condition | a reading: the equation that the terms fit |
|---|---|---|
| `af-no-wrap-N1` | Ty4 + Smax < 2^60 | Mx(i), of four limbs, = sc(i, ib(i)) |
| `af-no-wrap-N2` | 2^40 Ty + 2^20 (Ty + 128) + Ty + 128 + 2^41 < 2^61 - 1, and Ty + 128 + 256 < 2^20 | a key of `RM`, `RD` or `WH` = a row of its table; the second half is for the radix 2^20 |
| `af-no-wrap-N3` | 6 Smax + 2 C_sh + 3 x 65536 + 2^32 + 2^33 (Ty2 + 128 + 2^Sx) < 2^61 - 1 | a key of `FM` or `FD` = a row of `KFD` |
| `af-no-wrap-N4` | 6 Smax + 2 C_sh + 3 x 65536 < 2^32 | the part below 2^32 of such a key and of such a row |
| `af-no-wrap-N5` | TS + Ty4 < 2^60 | Zd(i), of four limbs, = the sum of the weights |
| `af-no-wrap-N6` | 2 x 2047 TS + TS + 2 TS Ty2 + 2 TS + 1 + 65536 (Ty + 128) + 65536 < 2^60 | a key of `WL` = a number below 65536. The source: the first term is for v, a clamp's output; the term 2 TS Ty2 is for o, a rounded mean of values of v, which has no clamp; the term 2 TS + 1 is for the second key, which is 2 Zd - 1 less the first key's remainder |
| `af-no-wrap-N9` | 2^(16 - lt) x 2 T < 2^60 | a key of `IX` = a number below 65536 |

The third column is a reading of the terms. The source states a purpose in words only for N6, in a
comment at the condition. In general, the head of `verifier/src/vf_attention_shape.h` calls them no-overflow
conditions against wrap-around in the field, and `verifier/docs/parameter-file.md` says of a
no-overflow condition that it is "meant to ensure that an equation that the proof checks modulo p
also holds over the integers", and adds that this is not proved. The conditions assume two
things. q, kx and v are between -2047 and 2047: the feed-forward segment's clamp is meant to give
this. Every cell is below Ty in absolute value: the smallness test is meant to give this. One bound
has no lookup of its own: no lookup has o as its key, in this segment or in the feed-forward
segment. Two comments in the source speak of it. The one at N6 (`verifier/src/vf_attention_shape.h`) calls o "a
rounded mean of values of v (no clamp)". The one at the feed-forward segment's conditions
(`verifier/src/cp_verify.cpp`) says that the attention sub-proof "is meant to show that o is a
rounded mean of values of v, which keeps v's bound", and that the site "o" relies on that; by this
page's reading of the source this is the condition `no-wrap-W-o`, which takes BC (11.2) as the bound
of o. Neither comment shows the bound, this page does not show it, and this page knows no test that
shows it.

##### What the parameter file can vary

The same text holds for every admitted parameter file. Beyond the sizes that are in the formulas,
four things change the list of scalars and claims:

- the lines `callA`: which calls' segments share a call, and in which order. This moves pieces, key
  points and the numbers c in the join. The order of the words within one line does not change the
  order of the units. The order of the lines does: units are taken call by call, so if `RM` and
  `RD`, or `FM` and `FD`, are in two lines, the unit of the earlier line comes first;
- the number nsh of different shifts in a group (each line of kind `KR` or `KFD` gives nsh calls; a
  unit then has only the pieces of its shift's layers);
- NAs (the parts of as);
- `lidb` (through dT).

With more than one shift, the name of an expanded call is the line's name followed by the shift in
decimal digits; with one shift it is the line's name.

##### Easy to get wrong

1. *Two orders of pieces.* A piece's offset off, and so its weight wp, follows the placing: the
   line's order of the call's segments, then lam, then a, then the stable sort by size. The sums
   over a unit's pieces, the end values of a sumcheck and the claims take the pieces by (lam, a).
2. *Units against calls.* The scalars of row 4 and the sumchecks K_u follow the units: `RM` before
   `RD` and `FM` before `FD` within a call, whatever the line says. The parts tM_u, tQ_u, tK_u
   follow the same order of units.
3. *Which point reads which part.* hW, hX and tO are read at coordinates of the lookup's key points;
   hE and qO at riE; qVM and qVD have the channel point zcW but token weights from VM's and VD's
   challenges; qF and qKx share the channel point rcF.
4. *The low ls coordinates of a channel point are in the weight vector, the others in the claim.* XL
   takes zc[0 .. ldh - ls) and lng takes zc[ldh - ls .. ldh).
5. *Bit k of the index.* IB_k and IBP_(b,k) have the weight 2^k, bit 0 the lowest, but pj's
   coordinate 0 belongs to the highest bit of j: y_k = pj[lt - 1 - k]. In HB, k is an index of 4
   bits with coordinate 0 of rk and tk for its highest bit.
6. *LE takes (zj_u, zi_u) in this order*, and the term is 1 - LE.
7. *The order (i, hk, ch) of SF and (i, hk, k) of HB* differs from the order (hk, i) of SE and of
   the keys.
8. *ZC has no channel.* [ZC] is Zd at (zgW, ziW); it enters REM beside [ACC] and [ZO], which are
   values at (zgW, ziW, zcW).
9. *BE_(SF,b) has no claim of its own*, and in group 8 the claim [BE] - [AE] + 1 = 0 comes before
   the claim on [AE].
