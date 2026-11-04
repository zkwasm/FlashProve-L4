*Part of [the file format](FORMAT.md), [section 11](11-sub-proofs.md). Rule numbers and terms are
those of that page.*

### 11.5 How the weights are placed in the stored table

This part is written from the programs' source. The rules of this page define a stored table's
shape, name and tree (R4.10, R7.4, R13.8); they do not define how a model's weights are placed in a
table. This part gives the placement that the three programs `cp_commit`, `cp_prove` and `cp_verify`
use. It has four steps: the matrix of digits; its rows, unit by unit; the position of every digit;
and how a segment's claims on the weights become one row on the table. It needs R15.2 (the lines of
the parameter file), R14.4 (the segments of a request's first chain), from 11.1 a scalar, the
sumcheck with a hiding sum, clog2 and the two orders of bits, and from 11.3 the short form [k_1, ..,
k_n | d_Y | N] of a sumcheck.

#### The matrix W

The committed weights are integers. Each is written in **digits** d_0, .., d_(nd - 1): its value is
d_0 + 256 d_1 + .. + 256^(nd - 1) d_(nd - 1), with nd = 4 digits for the embedding and nd = 2 for
every other tensor. All digits of all weights are the cells of one matrix, the **committed matrix**,
with m0 rows and 2^lk columns. In this part alone it is written W; sections 9 and 13.9 write M; the
W of R14.4 (e) and of section 9 is another thing, a vector with one entry for each of this matrix's
columns. What a key file states of the cells is: every cell is from -128 to 127, and every weight of
two digits is from -4096 to 4095 (the item `ranges`, R13.14).

*From W to the table.* The line `table b n i m e w` of the parameter file (R15.2) gives lk = w and
the number 2^b of the table's rows. Let k_lo = w - b. (Sections 9 and 10 write k_col for lk and v
for b; 11.1 writes lkW for lk.) The cell in row y and column c of W, with c = j + 2^k_lo r and j
below 2^k_lo, is entry j + 2^k_lo y of the message of the table's row r. So the high b bits of a
column's number choose the table's row, and a row of the table holds 2^k_lo columns of W, one row of
the matrix after the other. A verifier rejects the public files unless m0 2^k_lo entries fit a row's
message beside its pads.

#### The numbers that decide the placement

They are all lines of the parameter file: `L` (layers), `d`, `dh`, `H`, `G`, `f`, `fe`, `NE`
(experts), `V` (tokens of the vocabulary), `D`, and w of the line `table`. dq = H dh and dkv
= G dh. Layer l has experts if NE is not 0 and l is odd (12.3). No line of the layout file and no word
of the scale file enters: the layout file places the arrays of a request's segments (R14.5), not the
weights.

#### Units

The rows of W are cut into **units**. A unit is a run of consecutive rows that holds one kind of
tensor of one layer. It has a kind X, a layer l, a number of digits nd and an ordered list of two or
three **index groups**. An index group, in this part also just a group, is one index of the tensor:
it has a name, an extent M and a class, which says how the index is split between the column's
number and the row's number.

| X | tensor | groups, in order: (name, extent, class) | nd | in |
|---|---|---|---|---|
| 0 | query and key | (hd, 64, A), (cd, dh, A), (kk, d, low) | 2 | every layer |
| 1 | value | (kk, d, low), (x, dkv, high) | 2 | every layer |
| 2 | attention output | (kk, dq, low), (x, d, high) | 2 | every layer |
| 3, 4 | gate, up | (kk, d, low), (x, f, high) | 2 | layers without experts |
| 5, 6 | shared expert's gate, up | (kk, d, low), (x, fe, high) | 2 | layers with experts |
| 7, 8 | routed experts' gate, up | (kk, d, low), (e, NE, low), (x, fe, high) | 2 | layers with experts |
| 9 | down, or shared expert's down | (kk, f, low), (x, d, high); with experts (kk, fe, low), (x, d, high) | 2 | every layer |
| 10 | routed experts' down | (kk, fe, low), (e, NE, low), (x, d, high) | 2 | layers with experts |
| 11 | router | (kk, d, low), (x, NE, high) | 2 | layers with experts |
| 12 | embedding | (v, V, A), (x, d, high) | 4 | once, after the layers |
| 13 | output head | (x, V, A), (kk, d, low) | 2 | once, after the embedding |

*The order of the units.* For l = 0, .., L - 1: the kinds 0, 1, 2; then 3, 4, 9 in a layer without
experts, or 5, 6, 7, 8, 9, 10, 11 in a layer with experts. After the layers come 12 and 13. The
kinds 12 and 13 count as layer L. The extent 64 of hd is a constant.

#### Column bits

The lk bits of a column's number are given to a unit's groups in the groups' order. Let left be the
number of bits that are not given out, at first lk, and cb the number that are, at first 0. For each
group:

- Classes low and high: write M = o 2^z with o odd. The group takes t = min(z, left) bits.
- Class A: the group takes t = clog2(M) bits if that is not above left, and t = 0 otherwise. With t
  above 0 its whole index goes to the columns.
- The group's bits are the column bits cb0, .., cb0 + t - 1, with cb0 = cb. Then left becomes left -
  t and cb becomes cb + t.
- What the group leaves to the rows is lambda = M / 2^t for the classes low and high. For class A it
  is 1 if t is above 0, and M otherwise.

The unit uses ncb = cb column bits. Its number of rows is nd times the product of its groups'
numbers lambda.

#### The position of a digit

Take a unit whose first row is off. A cell of the unit is named by a digit's number q below nd and,
for each group g, an index v_g below M_g. Each index is split into a column part col_g below 2^t and
a row part rho_g below lambda:

- class low (the lowest t bits go to the columns): col = v mod 2^t, rho = floor(v / 2^t);
- class high (of the low z bits, the highest t go to the columns): with z' = z - t, col = floor(v /
  2^z') mod 2^t, rho = floor(v / 2^z) 2^z' + (v mod 2^z');
- class A: (col, rho) = (v, 0) if t is above 0, and (0, v) otherwise.

The cell is in

      column  c = the sum over the groups of col_g 2^(cb0 of g)
      row     y = off + rho_0 + lambda_0 (rho_1 + lambda_1 (rho_2 + lambda_2 q))

For a unit of two groups read lambda_2 = 1 and rho_2 = 0. The first group's row part runs fastest
and the digit's number slowest: a unit holds all its digits 0, then all its digits 1. The columns
from 2^ncb on belong to no cell of the unit, and neither do the columns in which the part of a class
A group is M or more.

*Offsets and m0.* The first row of a unit is the sum of the rows of the units before it. m0 is the
sum of all units' rows, rounded up to a multiple of `D`. m0 is the number that sections 8 and 9 call
the matrix's rows.

*What a verifier rejects* (the names are in 13.8):

- w below 1 or above 20;
- `D` other than 2048 or 32;
- L below 1 or above 4096;
- one of d, dh, dkv, dq, f below 1 or above 2^24, and, if some layer has experts, the same for fe
  and NE;
- V below 1 or above 2^24;
- a unit, a running sum or m0 above 2^22 rows;
- clog2(V) below 1, that is V = 1, or above w. Why: so the class A index group of the kinds 12 and
  13 always goes to the columns. `cp_verify` asks the first half as V below 2 (`shape-V`) and the
  second under the name `hid-vocab-cols`.

#### From a segment's claims to a row on the table

The segment of each group of layers and the segment of the head end with claims on the weights; an
attention segment has none (11.3). Here and below, eq(point, index) is the equality weight of the
index at the point, the coordinate of the highest bit first (11.1), except in the pending row. A
**weight claim** a names a unit and has one **factor** for each of the unit's index groups, and a
value v_a, which the segment has shown before as a scalar under a mask. In this part "claim" alone
means a weight claim, and not a claim in the sense of 11.1. A factor of class low or high is a list
HV of o elements of E and a point pt of z coordinates. It gives the index hi 2^z + lo the weight
HV[hi] eq(pt, lo). The first coordinate of pt belongs to the highest bit of lo. A factor of class A
is a list AV of M elements: it gives the index v the weight AV[v]. The claim says that the sum over
the unit's weights, each times the product of its indices' weights, is v_a. A segment has from 1 to
4096 claims.

*The merge* joins a segment's claims. Its rounds, with NA the number of claims:

| rounds | the prover's message | sampled |
|---|---|---|
| `BAT-sigma`, for NA rows | the scalars that wait, if any | 1: x. Claim a, counted from 0, has the coefficient x^a |
| the sumcheck [2, .., 2 \| 1 \| 1], with lk entries 2, all rounds of the plain form (11.1); its start form is the sum over a of x^a v_a | | its lk challenges are the point rm, the coordinate of the highest column bit first; the element of its `PSC Y` is beta; its one end value is fv' |

Write X_s = rm[lk - 1 - s] for the coordinate of column bit s, and eq1(u, y) = u y + (1 - u)(1 - y).
For claim a let

      e_a = the product over the index groups of class low or high, and over i below t, of eq1(pt of the index group's i-th taken bit, X_(cb0 + i))
            times the product over the index groups of class A with t above 0 of  the sum over v below M of AV[v] eq((X_(cb0 + t - 1), .., X_cb0), v)
            times the product over s from ncb to lk - 1 of (1 - X_s)

The i-th taken bit of an index group, counted from the lowest, is bit i of lo for class low and bit
z - t + i for class high. The **row weights** of the segment are a vector Gamma over the rows of W,
0 outside the units that its claims name. Claim a adds to Gamma[y], for every row y of its unit,

      x^a e_a 256^q times the product over the index groups of r_g(rho_g)

with q and the rho_g as in the formula for y. Here:

- classes low and high: r_g(rho) = HV[hi] eq(pl, lo'), where rho = hi 2^(z - t) + lo' and pl holds
  the coordinates of pt for the bits that are not taken, the highest first;
- class A with t = 0: r_g(rho) = AV[rho];
- class A with t above 0: r_g = 1.

The segment leaves one **pending row** on the table (the item `pending rows`, R14.3, counts them):

      the sum over y and c of Gamma[y] eq(psi, c) W[y][c]  +  beta c_mask  =  fv'

psi is rm in the reverse order, the coordinate of column bit 0 first, and eq(psi, c) is the frame's:
coordinate s belongs to bit s of c (11.1). c_mask is an entry of the request's index-mask message
(R14.4 (a)). The segments with claims take its entries one after the other, in the order of the
segments. The sumcheck's own closing equation, with fv' as it was sent, is the last equation of the
segment's sub-proof. It is on the sub-proof's own mask entries, and it does not read the mask entry
that has the number of fv' (11.1). A request with NG groups of layers has NG + 1 pending rows. How
they are joined into one and tested on the table is R14.4 (e), R14.9 and R14.10. (13.9 writes this
row with w, M, c and v.)

#### Worked numbers

| | the small example configuration (`target_small_f2/policy.txt`) | the full-size example configuration (`target_full_f2/policy.txt`) |
|---|---|---|
| `L`, `d`, `dh`, `H`, `G` | 8, 512, 128, 4, 2 | 48, 5120, 128, 40, 8 |
| `f`, `fe`, `NE`, `V`, `D` | 1024, 512, 4, 512, 2048 | 16384, 8192, 128, 202048, 2048 |
| b and w of the line `table` | 8, 11 | 11, 18 |
| units | 66 | 386 |
| rows of the first three units (kinds 0, 1, 2) | 4096, 128, 256 | 320, 40, 200 |
| first row and rows of the embedding | 57,352 and 512 | 3,045,360 and 20,480 |
| first row and rows of the head | 57,864 and 256 | 3,065,840 and 10,240 |
| the sum of the units' rows | 58,120 | 3,076,080 |
| m0 | 59,392 | 3,076,096 |

In the small example configuration every unit uses all 11 column bits. In the full-size example
configuration the router's units (kind 11) use 17 of the 18 column bits, and every other unit uses
all 18. There V is not a power of two, so in the kinds 12 and 13 the columns whose part for the
token is from V on belong to no weight.

#### Easy to get wrong

1. *Low and high.* A group of class low gives its lowest bits to the columns. A group of class high
   gives the highest of its z low bits, not the highest bits of the index. If M is not a power of
   two, the odd part o always stays in the rows.
2. *Three orders of the column bits.* rm has the highest column bit first, psi the lowest first, and
   the point of a class A group is taken from the group's highest bit down.
3. *Two cuts of a column's number.* The groups cut it by cb0 and t. The table cuts it at k_lo: the
   low k_lo bits are a position inside a row of the table, the high b bits are the row. The two cuts
   have nothing to do with each other.
4. *The groups run the other way in a row's number.* The first group of the list is the fastest in
   y, and the digit's number is the slowest.
5. *Rows after the last unit.* m0 is rounded up, so W can have rows that belong to no unit. No claim
   reads them; the key file's statement on bytes covers them.

#### What this part does not specify

This part does not define:

- how a model's tensors become the committed integers;
- which entry of a tensor an index of a group is;
- what is in the cells that belong to no weight (columns from 2^ncb on; column parts from M on of a
  class A group; rows after the last unit);
- which claims a segment of layers or of the head makes and what their factors are (11.2, 11.4);
- the digits that a committer chooses for a weight, where more than one list of digits gives it.
  Where every digit is from -128 to 127, only one list gives it, with two digits as with four. That
  every cell is in this range is what a key file's lookup on bytes is meant to show of the anchor,
  the commitment's first stored table (R13.14). `cp_verify` checks a proof on a leaf table, never on
  the anchor (13.2), and a leaf file has no such lookup on its leaf table: the file's link is meant
  to show that the leaf table's rows are the anchor's (section 9's head; R13.10).
