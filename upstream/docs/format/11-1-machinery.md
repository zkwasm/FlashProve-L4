*Part of [the file format](FORMAT.md), [section 11](11-sub-proofs.md). Rule numbers and terms are
those of that page.*

### 11.1 The machinery

This part is written from the programs' source. It describes the kinds of step that the sub-proofs
of all segments share, each as a sequence of rounds: the round's name, what the prover sends in it,
and what is sampled. Parts 11.2 to 11.4 then list the steps of each kind of segment in order. 11.1
has three pieces. This first piece gives the rounds. The second piece, from the heading "Values,
forms and claims: what this piece gives" on, gives their content. The third piece, from the heading
"The smallness test: values, equations and claims" on, does the same for the smallness test. The
protocol page, `docs/protocol.md`, calls this test the range check, and says that it is not the
range proof of its section 6.1. A reader needs the rounds of a chain (R5.1 to R5.4), sampling (R6.2
to R6.7), the order of the items in a prover's message (R8.3, R8.4), the frame's rounds of a segment
(R14.4) and the list of the rounds' names (R14.11). The letters lt, la, d, dh and NE are the numbers
of the parameter file's lines of these names (8.6), and T is 2^lt.

*Where a sub-proof's rounds are.* The sub-proof of a segment runs on the request's first chain,
between the frame's rounds of that segment (R14.4 (c)). It has no first query and no starting value
of its own. A **moment** of a segment is a point at which its prover commits to arrays (R12.12).
Moment 0 is the segment's first prover's message. Each later moment is a pair of the frame's rounds,
`SEND seed` and `SEND answer` (R14.4 (d)). The tables of this section name the frame's rounds only
where the order of a sub-proof's rounds depends on them.

*Scalars and their numbers.* A **scalar** is one element of E, 24 bytes (R1.4). Apart from the roots
and the answers to out-of-domain samples in the frame's rounds, everything that a sub-proof sends is
scalars. The scalars of a sub-proof are numbered from 0 in the order in which they are sent: this is
**a scalar's number**. (The programs' rejection names with `free-slot`, section 13, call a scalar's
number a slot.) The segment's first prover's message commits to a **mask row**, and the mask row has
an entry for each number. Every hidden value is sent under a mask. A **hidden value sent alone** is
the value plus the mask entry of its number. An **end value** of a sumcheck is the value plus beta
times the mask entry of its number. beta is what that sumcheck samples in its round `PSC Y`. The
number of scalars of a segment is public and follows from the public files. A verifier rejects a
proof that ends before the last scalar.

*Scalars do not close a round.* The prover's message of a round is everything that was sent after
the round before it, in the order sent. So a hidden value that is sent alone directly before a
sumcheck is in the message of that sumcheck's `PSC epsilon`, before the hiding polynomial's sum,
which is the last item (R8.4). A round whose step sends nothing has an empty prover's message, and
it asks its query all the same (R5.1).

*Nothing is tested during the rounds.* While a sub-proof's rounds run, a verifier rejects only for
form: a coordinate of a scalar that is not below p, missing bytes, a failed proof-of-work test.
There are two exceptions (11.2, step 9; 11.4, the next token). Every check on content becomes a
**claim**: an affine form in hidden values and in entries of committed arrays that must be 0. The
claims of a segment have a fixed order and are numbered from 0. The segment's last round of its own,
a `BAT-sigma`, samples one element chi, and claim number i gets the coefficient chi^i. The claims of
the smallness test on a matrix or on a read are the exception: each takes a number, and its
equations get powers of their own (below). The sum of all claims is one equation, which the frame
tests with the segment's group of oracle messages. So the order of the claims is part of the
protocol. After chi is sampled, a verifier also tests that certain weights of that equation are not
0 (third piece of 11.1; 13.9).

*Steps of one round.*

| step | round | prover's message | sampled |
|---|---|---|---|
| a point of n coordinates | `ZERO` | only scalars that are waiting; as a rule none | n elements of E, each by R6.4 |
| a point of n coordinates that joins 2^n claims of one shape into one | `JOIN` | as above | n elements of E |
| the coordinate that joins the two end values of a product level | `LINE` | nothing | 1 element of E |
| a batching of n rows | `BAT-sigma` | only scalars that are waiting | 1 element x of E; row number i, from 0, gets the coefficient x^i |

The number n of a `BAT-sigma` is the last of the eleven words of the round's entry in the plan
(R4.3). The round is run, and x is sampled, also where n is 1.

*The sumcheck with a hiding sum.* It is the step for the statement that a sum over the points of a
cube equals a hidden value, which is given as an affine form, the **start form**. It runs on a
hiding polynomial plus eps times the true summand. Its parameters are: for each variable the form of
its round and the degree k of its round polynomial; d_Y, the degree of the summand in the values
that are shown at the end; and n, the number of end values. R13.16 describes the same run of rounds
for the frame, in other letters. The g of R13.16 is k + 1, the number of coefficients, and its D is
d_Y. Its hiding sum is the hiding polynomial's sum, and a level of its product tree is a product
level.

| round | how often | prover's message | sampled |
|---|---|---|---|
| `PSC epsilon` | once | scalars that are waiting, then 1 scalar: the hiding polynomial's sum | 1 element eps of E, which can be 0 |
| `PSC round` | once for each variable | k scalars: the coefficients of the round polynomial g, lowest degree first, without the one that the table below names | 1 element of E, the variable's challenge |
| `PSC Y` | once, if n > 0 | d_Y scalars: the coefficients of degree 1 to d_Y of a polynomial in one further variable Y | 1 element beta of E that is not 0 (R6.5) |
| `PSC values` | once, if n > 0 | n scalars: the end values | nothing |

A sumcheck sends 1 + (the sum of the k) + d_Y + n scalars. A sumcheck can have no variable at all;
then `PSC Y` follows `PSC epsilon` directly. A sumcheck whose summand is affine in everything that
it reads shows no value: it has neither `PSC Y` nor `PSC values`, and it ends with its last
`PSC round`.

*The forms of a round, and the coefficient that is not sent.* Write cur for the running claim and g
= g_0 + g_1 X + .. + g_k X^k. The round's equation fixes one coefficient, and that one is not sent:
the verifier computes it. So no round can fail. After the challenge r is sampled, the running claim
becomes g(r). The running claim starts as (the hiding polynomial's sum) + eps t, where t is the
start form's constant plus, for each scalar's number that the form names, its coefficient times the
shown value of that number. For a hidden value sent alone and for an end value, the **shown value**
is the scalar as it was sent. It is 0 for the number of a hiding polynomial's sum and for the number
of a coefficient that was sent in a `PSC round` or in `PSC Y` (the second piece of 11.1).

| form | the round's equation | not sent | computed as |
|---|---|---|---|
| equality, at the coordinate z | (1 - z) g(0) + z g(1) = cur | the constant g_0 | g_0 = cur - z (g_1 + .. + g_k) |
| plain | g(0) + g(1) = cur | the linear coefficient g_1 | g_1 = cur - 2 g_0 - (g_2 + .. + g_k) |
| the round `PSC Y` | g(0) = cur | the constant g_0 | g_0 = cur |

In an equality round, z is the variable's coordinate of a point that is fixed before the sumcheck
begins. A plain sum over a variable is sometimes written as an equality round at the constant z =
1/2, with the start form divided by 2, once for each such round. Parts 11.2 and 11.4 say where. Each
such sumcheck (one in a feed-forward segment, two in the output head) has lPd such rounds and the
factor 2^(-lPd), where lPd is the least number with 2^lPd >= d. In feed-forward segments and in the
output head every round has degree 2, so 2 scalars, except five rounds of degree 1 in each lookup.
The forms and degrees of an attention segment's rounds are in 11.3.

*A product level with m variables.* It is a sumcheck with m equality rounds of degree 2, d_Y = 2 and
2 end values, followed by one round `LINE`. It sends 2 m + 5 scalars. Its rounds are `PSC epsilon`,
m times `PSC round`, `PSC Y`, `PSC values`, `LINE`. With m = 0 it has no `PSC round`. The point of
the next level down has m + 1 coordinates: the element of `LINE` for the new highest bit, then the m
challenges, each as the coordinate of its variable. A product level of a segment binds the variables
in one of two orders; the last key level of a lookup has a third (below). The **default order** goes
from the highest bit down to bit 0. The **range order** binds the bits 2, 1, 0, 5, 4, 3, 8, 7, 6 and
then the other bits from the highest down to bit 9. A challenge is the coordinate of the variable
that its round binds, whatever the order of the rounds.

*A token sumcheck.* It is 1 hidden value sent alone, the claimed sum, and then a sumcheck that
starts from it, with one equality round of degree 2 for each of the lt bits of a token's position,
the highest bit first. The claimed sum and the hiding polynomial's sum are one prover's message, in
this order. It sends 1 + (1 + 2 lt + d_Y + n) scalars, and 2 more for each further round that a
segment appends.

*A lookup.* All calls of a segment's list of calls run as one lookup. The smallness test's two calls
run as a second one (below), so every segment has two lookups. Call number c, in the order of the
segment's list of calls, has 2^(n_c) keys and a public table of 2^(k_c) rows. dT is the least of the
line `lidb` (22 if the line is absent) and all n_c. kT is the largest k_c. NC is the number of
calls, at most 16, and cK the least number with 2^cK >= NC. NT is the sum of the 2^(n_c - dT), at
most 256, and rK the least number with 2^rK >= NT. v = dT + rK and dL = kT + cK. A verifier rejects
a parameter file unless dT >= 10, v < 32 and 2^(dL + 32) < p. The keys are never sent. The
multiplicities are committed in bits, in 32 planes. Some key positions are **padding**. Which ones
is public. A padding key is a fixed row of its call's table, and it is not counted in the
multiplicities. The source argues, and does not prove, that padding which is a row of the table lets
in no key that is in no row (`blocks/iop/lkp.h`, in the comment that begins "Padding."). The
argument there is that one side of the lookup's equation of products is then a product of factors
alpha minus a row (alpha is the point of comparison, step 2), whatever the number of padding
positions is. By that argument a false number can make an honest proof fail, or force more keys to
be that row, but lets in no key from outside the table. That comment gives this as the second of two
safeguards, and says why there are two: a lookup whose padding was described loosely, by a number in
the public inputs and by positions in a list of the caller's that had one position fewer, accepted a
key that is in no row, in a test of that block. It also says what a test that is not part of this
repository shows: an injected surplus of padding with a free value lets a key from outside the table
pass, and a surplus with a row of the table does not.
The first safeguard of that comment is that the padding is in the public inputs once, as runs of
positions, from which its number is computed. The lookups of the sub-proofs do not have it: there
the padding is described twice, and a self-check compares the two descriptions (the second piece of
11.1, "The padding is described twice", with the limit of that self-check). Of a difference between
the two the source says that it could not let in a key from outside a table, since a padding key is
a row of its call's table (`verifier/src/vf_lookup_argument.h`); that rests on the argument above,
which is not proved. The tables are listed below, under public tables. The padding row is row 0 of a
range table, of the table of exponentials of the output head and of attention's tables `KR` and
`WH`; row u0 of `KFD`; and row 32768 of the clamp, the gate function and the sigmoid. A verifier
rejects a parameter file under which all keys of a call are padding. The other tests on these
numbers are in section 13.

| no. | step | rounds | scalars | sampled |
|---|---|---|---|---|
| 1 | the shift, only if NC > 1 | `LKP shift` | 0 | 1 element |
| 2 | the point of comparison | `LKP` | 0 | 1 element |
| 3 | the tie of the two sides' roots: 31 sumchecks without a variable, d_Y = 3, with 3 end values in the first and 2 in each other | 31 times `PSC epsilon`, `PSC Y`, `PSC values` | 7 + 30 x 6 = 187 | 31 times eps and beta |
| 4 | the point that joins the 32 planes' roots | `JOIN` | 0 | 5 elements, the coordinate of the lowest bit first |
| 5 | the key side: product levels with m = 0, 1, .., v - 1 variables | as for a product level | the sum of 2 m + 5 | as for a product level |
| 6 | the table side: product levels with m = 5, 6, .., 4 + dL variables | as for a product level | the sum of 2 m + 5 | as for a product level |
| 7 | the table side's leaves: a sumcheck with dL equality rounds of degree 2 and then 5 equality rounds of degree 1, which shows no value | `PSC epsilon`, 5 + dL times `PSC round` | 1 + 2 dL + 5 | eps and 5 + dL challenges |
| 8 | a fresh point | `ZERO` | 0 | 5 + dL elements, the coordinate of the lowest bit first |
| 9 | the bits are bits: a sumcheck with 5 + dL equality rounds of degree 2, d_Y = 2, 1 end value | `PSC epsilon`, 5 + dL times `PSC round`, `PSC Y`, `PSC values` | 2 (5 + dL) + 4 | eps, 5 + dL challenges, beta |

All levels use the default order, except the last level of the key side, m = v - 1. It binds the
bits rK + 2, rK + 1, rK, rK + 5, rK + 4, rK + 3, rK + 8, rK + 7, rK + 6, then the bits from m - 1
down to rK + 9, then the bits from rK - 1 down to 0. A key's index has the number of its tree in its
low rK bits and its index inside the tree in the dT bits above. A row's index in steps 7 to 9 has
the plane's number j in its low 5 bits, the call's number c in the cK bits above, and the row t of
the call's table in the kT bits above these. The trees are given to the calls by size, the largest
calls first, calls of one size in the order of the list; call c has 2^(n_c - dT) neighbouring trees.
The trees from NT on hold padding alone, with the padding key of call 0. In the table, the numbers c
from NC on hold the last call's table again, and a table with fewer than 2^kT rows is repeated.

*The key point.* After the last level of step 5 the point has v coordinates, the highest bit's
first. The first dT of them, zx, belong to the index inside a tree, and the last rK, zy, to the
tree's number. The **key point of call c** has n_c coordinates: the last n_c - dT coordinates of zy,
and then zx. So the last dT coordinates are the same for all calls. Where 11.2 and 11.4 speak of the
last lt or the last la coordinates of the lookup's key point, they mean the last coordinates of zx;
its last lt coordinates are the lookup's **token point**. A verifier rejects the parameter file
unless dT >= la in a feed-forward segment and in the output head.

*Where the multiplicities are.* The bit (t, c, j) is bit j of the multiplicity of row t of call c,
which is the number of keys of call c that are counted at row t. Each key that is not padding is
counted once, at one row whose value it equals; padding keys are not counted. Where a value is in
several rows (a table with fewer than 2^kT rows is repeated), a verifier accepts every choice of the
row. The bits are the cells of a committed matrix of 2^lm rows and NC 2^(kT + 5 - lm) columns, with
5 <= lm <= kT + 5. lm is the line `lmb`, `lmbA` or `lmbH`, and the matrix is called b in a
feed-forward segment (11.2), bA in an attention segment (11.3) and bH in the head (11.4). (The array
`BZ`, which holds the multiplicities of the smallness test's own lookup, is placed by another rule,
which the third piece of 11.1 gives.) With S = (c 2^kT + t) 32 + j, the bit is in column floor(S /
2^lm) and row S mod 2^lm. The matrices b, bA and bH are packed like every matrix of a segment's list
(the smallness test, below); `BZ` is not packed. The matrix, every array from which the keys are
read, and the mask row are committed before step 1, and the answers to their out-of-domain samples
are given before step 1. They belong to an earlier moment of the segment, or to an earlier segment.

*The merge.* A segment that makes claims on the committed weights ends them with a merge. It is one
`BAT-sigma` whose n is the number of weight claims, and then a sumcheck with lkW plain rounds of
degree 2, d_Y = 1 and 1 end value. lkW is the last number of the parameter file's line `table`. The
first round binds the highest column bit. The merge sends 2 lkW + 3 scalars. Its end value is masked
by an entry of the request's index-mask message (R14.4 (a)) and not by the segment's mask row: the
segments that have a merge take these entries in the order of the segments. The source says what
soundness rests on here (`blocks/iop/batt_shown.h`, in the comment before `batt_shown_prove()`; the
header states this step in the library's form): the index-mask message, with its root and the answer
to its out-of-domain sample, comes before the merge's element beta, because a prover that could
choose the entry after beta could fit the end value to a false claim. By that header's head and the
same comment no program includes the header: `cp_verify` runs the step's messages by a text of its
own, `verifier/src/vf_weights_merge.h`, and `cp_prove` by kernels of its own. So the header's tests,
and its malicious provers, which are not part of this repository, test a twin of what the programs
run, and by themselves say nothing of the programs' texts. By R14.4 (a) that message is sent, and
its out-of-domain sample is answered, in the first segment's first two rounds, so before every
merge. The end value has a number like every scalar, and the mask row's entry of that number masks
nothing.

*The smallness test.* It is the step for two statements: the cells which a segment's packed arrays
hold are small integers, and the entries of the segment's vector `U3` are sums of these cells. A
segment commits to its cells in packed form: three cell rows 3 i, 3 i + 1, 3 i + 2 of a column are
one element of F, with the radix 2^20. The test's steps are:

| no. | step | rounds | prover's message | sampled |
|---|---|---|---|---|
| 1 | the matrix | `RNG` | nothing | (the number of columns of all matrices of the segment's list) x tau entries in {-1, 0, 1}, by R6.7 |
| 2 | the moment of the digits: the arrays `ZZ` (digits) and `BZ` (the planes of step 3) | `SEND seed`, `SEND answer` | roots; then the answers to the out-of-domain samples (R14.4 (d)) | seeds; nothing |
| 3 | a lookup with two calls: the low digits, of 10 bits, and the high digits, of 8 bits | as for a lookup with NC = 2, so with `LKP shift` | as for a lookup | as for a lookup |

The entries of step 1 come matrix by matrix in the order of the segment's list of matrices, within a
matrix column by column, with tau entries for each column. tau is the parameter file's line `tau`;
the tests on it are below. In every segment `RNG` comes directly after the `SEND answer` of the
moment of `U3`. The test's equations have no round of their own. Equation number j gets the
coefficient chi^(NCm + j) of the segment's last `BAT-sigma`, where NCm is the number of the
segment's claims. How the equations are numbered is said below, with the order of the test's claims.

*The lookup on the digits.* It has two calls, with tables of 2^10 and 2^8 rows, so its dL is 11.
Each of the two calls has n_Z = max(10, clog2(tau) + clog2(mz)) key bits. clog2(x) is the least l
with 2^l not below x, and mz is the number of cell rows of all matrices of the segment's list
together (below). By the rule above its v is v_Z = n_Z + 1. A verifier rejects the parameter file
unless tau is a multiple of 32 from 32 to 256 and is a power of two or the sum of two powers of two.
It also rejects unless tau is the number of combinations that the plan derives: the line `sec` less
the proof-of-work bits that the line `work smallness` gives to a request's own rounds. (The source's
formula has one more term, for requests that share a session; no parameter file sets their number,
so the term is 0, and by the source the programs' bound on such a round is for one request alone,
not jointly for the requests of a batch (`blocks/numbers/setting.h`): 8.4, item 21.)

*The arrays of a segment's smallness test.* A segment has a list of **matrices** of small integers,
the cells, in a fixed order. Some of them the segment commits to itself; the others an earlier
segment has committed to, and this segment reads them. Matrix number mu of the list has m_mu cell
rows and k_mu columns. Let mz be the sum of m_mu over all matrices of the list, those of earlier
segments among them. A verifier rejects the public files unless the list has fewer than 256
matrices, every m_mu is at least 1, every k_mu is from 1 to 8191, and 2 mz is below 2^32.

| name in the item `layout` | moment | number of rows | number of columns | over |
|---|---|---|---|---|
| `M`mu, for each matrix mu that the segment itself commits to | 0: the matrices that no other segment reads, in the list's order; then those that another segment reads, in the list's order | ceil(m_mu / 3) | k_mu | F |
| `U3` | 1 | 1 | the length of the vector | E |
| `ZZ` | 2 | tau | 2 mz | F |
| `BZ` | 2, after `ZZ` | 32 | 2^11 | F |

In `M`mu the element in row i and column c holds the cell rows 3 i, 3 i + 1 and 3 i + 2 of column c,
as cell(3 i) + 2^20 cell(3 i + 1) + 2^40 cell(3 i + 2), read in F. A cell row from m_mu on counts
as 0. In `ZZ`, row u belongs to combination u of the matrix of the round `RNG`. Column off(mu) + i
holds the low digit, of 10 bits, and column mz + off(mu) + i the high digit, of 8 bits, that belong
to cell row i of matrix mu; off(mu) is the number of cell rows of the matrices before mu together.
The two digits together are the combination's value plus 2^17. `U3` is made of parts, one after the
other. A **part of `U3`** belongs to one matrix of the list and one public weight vector. It holds
one sum over E for every column of the matrix and every block of as many cell rows as the weight
vector has entries. The entry of column c and block b of a part with B blocks is number c B + b of
the part. Where 11.2 and 11.4 name a matrix that is cut into several matrices of the list (af, ad,
hf), each point or weight vector gives one part of `U3` for each of these, by increasing mu. A
matrix that an earlier segment has committed to has no array in this segment; the segment reads the
earlier segment's array `M`mu', where mu' is the matrix's number in the earlier segment's list. That
array is kept until then.

*The order of the smallness test's claims.* The claims of a segment's smallness test are added in
this order. The steps are those of a lookup (above), (v_Z, 11) are the numbers of the lookup on the
digits, and NM is the number of matrices of the list.

1. The claims that the lookup's sumchecks leave: 31 for step 3 of the lookup, in order; v_Z for the
   product levels of step 5; 11 for the product levels of step 6; 1 for step 9.
2. One claim that ties the lookup's keys to the digits in `ZZ`.
3. Three claims that read `BZ`: the claim that the sumcheck of step 7 leaves; the value that step 9
   has shown; and that all multiplicities add up to the number of keys that are not padding.
4. One claim for each matrix, mu = 0, .., NM - 1, those of earlier segments among them: it ties the
   digits in `ZZ` to the array of the matrix.
5. One claim for each read. A **read** is a pair of a part of `U3` and one block of cell rows of
   that part's matrix. NX is the number of reads. The reads are taken matrix by matrix in the list's
   order; within a matrix by part of `U3`, in the order of the segment's list of these parts (in an
   attention segment: the order in which they are placed in `U3`); within a part of `U3` by
   increasing block.

That makes (32 + v_Z + 11) + 4 + NM + NX claims. Each takes one number in the segment's count of
claims. The claims of 1 to 3 are multiplied by the power of chi that their number gives. A claim of
4 or 5 holds many equations, which have numbers of their own. Let tauP be tau rounded up to a power
of two. Let first_mu be the sum, over the matrices before mu, of tauP times (their packed rows
rounded up to a power of two); a matrix has ceil(m_mu / 3) packed rows. Packed row i of matrix mu
and combination u have the number first_mu + i tauP + u. After all matrices, read number q and
combination u have the number first_NM + q tauP + u. Numbers that name no equation stay unused, and
they count in the n of the segment's last `BAT-sigma`. So the equations take Q = first_NM + NX tauP
numbers. In an attention segment the claims of the smallness test are the last of the segment's
claims.

*The room in the mask row.* The number of a segment's scalars plus the line `answers` must not be
above the number of entries of a message of the request's mask code; a verifier rejects the public
files otherwise (`plan-mask-row`). Why plus `answers`: the oracle message that holds the mask row
gets an out-of-domain sample, and its pads are the last entries of that row, one for each answer
(13.9), so the scalars' entries must end before them.

*The last round of a sub-proof.* It is a `BAT-sigma` whose n is the number of the segment's claims
plus the numbers that the smallness test's equations take. After it the frame closes the segment
(R14.4).

*Public tables.* A call's table is public. A range table has no listed rows: row t is the number t.
The other tables of feed-forward segments and of the output head have 65,536 rows each, and a row
packs an argument and a function's value into one number below p. The rotary tables are public too,
but they are not tables of a call. In the table below, eg, smax, `S`, `rx` and kFD are lines of the
parameter file (8.6); eg is 8 and smax is 4095 if the line is absent; and sh and rp are words of a
layer's record in the scale file (13.4, step 2).

| table | row t | how the values are made |
|---|---|---|
| ranges of 8, 10 and 16 bits | t | nothing to make |
| the clamp | t + 2^28 (clamp(t - 32768, -2047 .. 2047) + 2048) | integers |
| the gate function (12.6 calls it SiLU) | t + 2^28 (s(t) + 4096), s(t) = round(64 silu(c / 2^eg)) clamped to -smax .. smax, c = t - 32768 clamped to -64 x 2^eg .. 64 x 2^eg - 1 | floating point (exp) |
| the sigmoid | t + 2^17 x 16 round(4096 sigmoid((t - 32768) / 256)); with the line `gcl 1` the table has two halves (11.2) | floating point (exp) |
| the rotary tables (12.6: the rotary table) | round(2^12 cos(a)) and round(2^12 sin(a)), a = t theta0^(-2 i / dh), for position t below T and pair i below dh / 2, with theta0 = `theta0_lo` + 2^32 `theta0_hi`; a layer whose word rp is 0 has the cosine 2^12 and the sine 0 | floating point (pow, cos, sin) |
| the exponentials of the output head | u + 2^16 EX[u] (11.4) | integers, by a recurrence |
| `KR`, for the shift sh (attention) | (t mod 256) + 2^20 floor(t / 256), and 2^40 more if t >= 2^sh; 65,536 rows | integers |
| `WH` (attention) | (t mod 256) + 2^20 floor(t / 256); 65,536 rows | integers |
| the exponential function (attention; 12.6 calls it the softmax table) | Tt[u] = 2^S exp(-u / 2^rx), rounded, for u = 0, 1, .., with the lines `S` and `rx`; u0 is the first u with Tt[u] = 0 | floating point (exp) |
| `KFD`, for the shift sh (attention) | 2^kFD rows. With C_sh = u0 2^sh, rho = 65536 - 2^sh and nHD = ceil((4 dh 2047^2 + 1) / rho): t 2^sh + 2^33 Tt[t] for t < u0; C_sh - 2^sh + rho (t - u0) + 2^32 for u0 <= t < u0 + nHD; C_sh - 2^sh + rho (t - u0 - nHD) + rho / 2 + 2^32 from there on | from Tt |

The programs compute four tables in double-precision floating point: the gate function, the sigmoid
and the rotary tables, with halves rounded away from zero, and attention's table of the exponential
function, with halves rounded to the even integer. The format does not define these values by
integer arithmetic. A second verifier computes the four tables to the bit as the programs do, or
takes them as data (what then fixes them: R15.3). The parameter file can hold their digests, in the
lines `tsilu0` to `tsilu7`, `tsig0` to `tsig7`, `trope0` to `trope7` and `texp0` to `texp7`. Either
all 32 lines are present or none, and a verifier rejects a file that has them unless NE is above 0. If they are present, a verifier rejects unless the digest of each table equals
the file's. Each line holds one 32-bit word of a digest of 32 bytes, lowest word first. The four
lists of words and their digest are defined in 12.6.

*The order of bits.* Two conventions meet at the border between the sub-proofs and the frame. Inside
a sub-proof a point is written with the coordinate of the highest bit first. The equality weight of
the index i at the point (z_0, .., z_(n-1)) is a product over s = 0, .., n - 1. Its factor number s
is z_s if bit n - 1 - s of i is 1, and 1 - z_s otherwise. In the frame, variable s belongs to bit s,
the lowest bit first, and the entry in row b and column c of a table of 2^k rows has the index b +
2^k c. Three things follow. The points of steps 4 and 8 of a lookup are sampled lowest bit first and
then reversed for use inside the sub-proof. The weights of a claim on the rows of an array are
passed to the frame with the point reversed. The merge's point is sampled highest column bit first,
and the frame's row on the stored table takes it reversed.

*Easy to get wrong.*

1. The order of bits, as above, and the range order of some product levels. A verifier that binds
   the variables in the default order everywhere reads the same bytes and samples the same
   challenges, but it forms other claims, and it rejects honest proofs.
2. Sumchecks are not all alike. The coefficient that is not sent is the constant in an equality
   round and the linear one in a plain round. The leaves' sumcheck of a lookup has no `PSC Y` and no
   `PSC values`. The tie's sumchecks have no `PSC round`. A `BAT-sigma` for one row is a round like
   any other.
3. The point that joins the planes' roots (step 4 of a lookup) is sampled in a round named `JOIN`,
   and the fresh point of step 8 in a round named `ZERO`.
4. A hidden value that is sent alone opens no round of its own. It waits for the next round.

*What this part does not specify.* This first piece of 11.1 gives the rounds of the shared steps:
the order and the size of every message, and the number of the values that are sampled in every
round. The second and the third piece give their content: what each scalar is, and every claim of a
sumcheck, of a lookup, of the merge and of the smallness test as a formula. How a segment forms the
keys of its lookup from committed cells is in the second halves of 11.2 to 11.4. 11.1 does not give
the arguments why these claims imply the statements that the steps are for, and it does not define
an honest prover's choice where a verifier accepts several values. 13.9 says in words which row a
sumcheck with a hiding sum leaves and which mask entries it takes.

#### Values, forms and claims: what this piece gives

The first piece of 11.1, above, gives the rounds of the steps that all sub-proofs share. This second
piece gives their content: what each sent scalar is, what a verifier computes from it, and every
claim that these steps leave, as a formula and in order. It covers hidden values and forms, the one
equation of a segment, the sumcheck with a hiding sum in all its uses, the lookup, and the merge.
The third piece does the same for the smallness test. Parts 11.2 to 11.4 give what is particular to
a kind of segment: the start forms, the summands and the key forms. A reader needs the first piece
of 11.1, the field E and its arithmetic (section 2), the item `layout` (R14.5) and 13.9, which says
how the frame tests the equation that a segment leaves.

All arithmetic is in E. An integer or an element of F in a formula means its image in E: the element
whose first coordinate is that number modulo p and whose other two coordinates are 0. **Public**
means: a verifier computes it from the public files, the request, the bytes of the proof and the
sampled values. **Hidden** means: a verifier is not sent the value itself; it computes only with a
form in which the value occurs.

*Notation.* eq(z, i) is the equality weight of 11.1 (the order of bits): for a point z = (z_0, ..,
z_(n-1)) and an index i below 2^n it is the product over s = 0, .., n - 1 of z_s if bit n - 1 - s of
i is 1, and of 1 - z_s otherwise. With n = 0 it is 1. Every point in this piece is written with the
coordinate of the highest bit first, unless the text says otherwise. (x | y) is an index whose high
bits are x and whose low bits are y; the same sign joins the coordinates of a point. clog2(x) is the
least integer l with 2^l >= x. A **map** on n bits is a vector a(i), i < 2^n, of hidden or public
values. Its value at a point z of n coordinates is a(z) = sum over i < 2^n of eq(z, i) a(i); this is
the one polynomial of degree at most 1 in each coordinate that agrees with a on the points of zeros
and ones. If the a(i) are committed cells, a(z) is a form. The letter n has three uses, and the text
says each time which: the number of coordinates of a point, the number of end values of a sumcheck,
and the number of rows of a `BAT-sigma`.

| sign | meaning | public or hidden |
|---|---|---|
| j | a scalar's number (11.1) | public |
| ns | the number of the segment's scalars | public |
| mask(j) | entry number j of the segment's mask row | hidden |
| shown(j), factor(j) | the two elements that the verifier keeps for the number j (below) | public |
| hidden(j) | shown(j) - factor(j) mask(j): the hidden value of the number j | hidden |
| A{w} | a combined row: the array A under the row weights w; A{w}[col] is its entry for the column col | w public, the entry hidden |
| f, pub(f) | a form; its public part | the coefficients and pub(f) public, the value hidden |
| chi | the element of the segment's last `BAT-sigma` (11.1) | public |
| NCm | the number of the segment's claims (11.1) | public |
| C | the start form of a sumcheck | a form |
| j0 | the number of a sumcheck's first scalar | public |
| hs | the hiding polynomial's sum, the scalar with the number j0 | public |
| eps, beta | what a sumcheck samples in `PSC epsilon` and in `PSC Y` | public |
| NV | the number of a sumcheck's variables, the variable Y counted | public |
| l | the number of a variable, from 0, in the order in which the rounds are run; Y, if there is one, is the last | public |
| k_l, gamma_l | the degree of the round polynomial of variable l; the element sampled in its round | public |
| cur | the running claim of a sumcheck (11.1) | public |
| psi_l, tail_l, PP | weights of the hiding polynomial at the sampled values (below) | public |
| u(l, deg) | the coefficient of degree deg of the hiding polynomial's part in variable l | hidden: a mask entry |
| jE | the number of a sumcheck's first end value | public |
| S0, K | the summand's public part on the shown end values; its kept form | S0 public, K a form |
| Left(C; S0; K) | the form that a sumcheck leaves | a form |
| xi | the element sampled in a round `LINE` | public |
| theta, alpha | a lookup's shift (`LKP shift`) and its point of comparison (`LKP`) | public |
| c | the number of a call in the segment's list of calls, from 0 | public |
| n_c, k_c, dT, kT, NC, cK, NT, rK, v, dL, lm | a lookup's sizes, as in 11.1 | public |
| the other signs of a lookup | in the table under the lookup's heading below | |

*In the source's names*, for a reader who has the source. One pair of names clashes. This page's chi
is `theta` in `HFork`, the state of a sub-proof, and its powers are what `kap()` gives; this
page's theta, the lookup's shift, is `theta` in `HForest`. A scalar's number is a slot. A claim is
what `lin` adds. A combined row is a virtual row, `HRow`: `coef` holds the row weights w, and `L`
the weights on the columns. shown(j), factor(j) and the coefficient of hidden(j) in the segment's
one equation are `vp`, `mf` and `LS`. A sumcheck's start form is `start` in `Psc`, and Left(C; S0;
K) is what `pscRow` returns.

##### Hidden values, forms and the one equation of a segment

*What the verifier keeps for a scalar's number.* Each of the ns numbers of a segment gets one pair
(shown(j), factor(j)), once. Which pair depends only on the kind of the scalar that has the number,
and the kind is public.

| the scalar with the number j | shown(j) | factor(j) | so hidden(j) is |
|---|---|---|---|
| a hidden value sent alone | the scalar | 1 | the scalar - mask(j) |
| an end value of a sumcheck (for the merge's, see the merge below) | the scalar | the beta of that sumcheck | the scalar - beta mask(j) |
| the hiding polynomial's sum, or a coefficient sent in a `PSC round` or in `PSC Y` | 0 | -1 | mask(j) |

The third line is easy to get wrong. The sum and the coefficients of the round polynomials are used
as they are sent, in the running claim, and no mask is taken off them. Their numbers serve a second
purpose: the mask entries of these numbers are the coefficients of the sumcheck's hiding polynomial,
in an order of their own (below). So for such a number hidden(j) is the mask entry, and it has
nothing to do with the scalar that was sent under j.

*A combined row.* An array A of the item `layout` (R14.5) has R rows and NK columns. A **combined
row** A{w} is the array with a public vector w of R **row weights** in E. Its entry for the column
col is A{w}[col] = sum over r < R of w(r) A[r][col]. For an array over F the cell A[r][col] is read
in E. Each part of this section says which combined rows its claims read and what their row weights
are. The row weights can depend on every sampled value of the segment, chi among them. A vector that
is stored as an array of NK columns has its entry x in row floor(x / NK) and column x mod NK.

*Three shapes of row weights.* Most row weights have one of three shapes, and the parts of this
section name them so. (1) **Equality weights on a block of rows**: for a point z of n coordinates
and a first row r0 that is a multiple of 2^n, w(r0 + r) = eq(z, r) for r < 2^n, and w is 0 on all
other rows. (2) **Ones**: w(r) = 1 for every row. (3) **A product over the bits** of the row's
number: with elements lo_q and hi_q for each bit q, bit 0 first, w(r) is the product over q of hi_q
if bit q of r is 1, and of lo_q otherwise. The powers of a base x are of this shape, with lo_q = 1
and hi_q = x^(2^q), so that w(r) = x^r.

*A form.* A **form** f is a sum of a constant, of terms on hidden values and of terms on entries of
combined rows: f = const(f) + sum over j of f[j] hidden(j) + sum over the entries of f[A{w}, col]
A{w}[col]. The constant const(f) and the coefficients f[j] and f[A{w}, col] are public elements of
E, and all but finitely many are 0. Forms are added, and multiplied by a public element, term by
term. The form whose only term is hidden(j), with the coefficient 1, is written hidden(j). The
**public part** of a form is pub(f) = const(f) + sum over j of f[j] shown(j): each hidden(j) is
replaced by shown(j), and the terms on arrays are left out. So

pub(f) - f = sum over j of f[j] factor(j) mask(j) - sum over the entries of f[A{w}, col] A{w}[col].

pub(f) is computed only after every number that f names has its pair. The arrays that f names need
not be committed by then (11.2 has such a start form). The source gives its reason at that start
form (`verifier/src/vf_layers_check.inc`): every coefficient of the form is fixed by then, and the entries
of `U3` that it names are committed later and are functions of the cells. By this page's reading
(the source states the reason in a comment and proves nothing), a formal statement on such a
sumcheck asks this of the order: before eps is sampled the form's coefficients are fixed, the cells
are committed (11.2, the table of moments), and the points that define the tables of `U3` that the
start form and the summand name are drawn, or those entries are not yet functions of the cells
alone. It does not ask that everything that the form names is committed by then. From reading the
code, the verifier meets the third condition; no test or self-check enforces it.

*A claim* is a form that must have the value 0. The claims of a segment are numbered from 0 in the
order that the segment's part gives. No claim is tested by itself.

*The one equation.* Let cl_0, .., cl_(NCm - 1) be the segment's claims, and chi the element of the
segment's last `BAT-sigma`. The verifier adds up chi^i cl_i, i = 0, .., NCm - 1, term by term:

1. for each combined row A{w} and each column col: L(A{w})[col] = sum over i of chi^i cl_i[A{w},
   col];
2. for each number j < ns: LS(j) = sum over i of chi^i cl_i[j];
3. c0 = sum over i of chi^i const(cl_i).

The claims of the smallness test on a matrix or on a read add to the same three sums under the
powers of chi that 11.1 gives to their equations. Their own numbers i are counted in NCm, but the
three sums above have no term chi^i for them. The segment's equation is

sum over the combined rows A{w} of sum over col < NK of L(A{w})[col] A{w}[col] + sum over j < ns of
LS(j) hidden(j) + c0 = 0.

With hidden(j) = shown(j) - factor(j) mask(j) it becomes an equation on committed entries with a
public right side, and in this shape the sub-proof passes it to the frame:

sum over the combined rows A{w} of sum over r < R, col < NK of w(r) L(A{w})[col] A[r][col] + sum
over j < ns of MC(j) mask(j) = TG, where MC(j) = - LS(j) factor(j) and TG = - (c0 + sum over j < ns
of LS(j) shown(j)).

This is the equation of 13.9 (at a segment's end). Each combined row gives a term on its array,
which is named by the segment that committed to it and by its name in that segment's list of the
item `layout`. The term has the scale 1, the weights w on the array's rows and the weights L(A{w})
on its columns. MC(0), .., MC(ns - 1) are the coefficients on the first ns entries of the segment's
mask row, and TG is the target. Some combined rows have a w that is not a product over the bits of
the row's number. The programs pass such a row as one term for each row r with w(r) not 0. The
term's scale is w(r), and its weight on rows is 1 at r and 0 elsewhere. That is the same equation.
The n of the segment's last `BAT-sigma` is NCm plus the numbers that the smallness test's equations
take (11.1).

*What cannot fail.* The programs end some runs under names that begin with `internal-` (13.8). Five
of the causes are: (1) a number gets two pairs; (2) a number of the segment has no pair at the end;
(3) a form's public part is asked for before a number that it names has its pair; (4) a claim's
number is used twice or not at all; (5) a claim is added after the equation was passed on. All this
depends on the public files alone, so no proof can cause it, and by this page's reading of the
source no public file that has passed the earlier tests can cause it either. A second verifier needs
none of these five tests. Of the further tests under such names that depend on the public files
alone this page says nothing in general: that no proof can cause one of them does not show that no
admitted public file can. Where a part of section 11 says of such a test that a verifier rejects
(`internal-qa-layout` in 11.3, for example), every verifier makes it. Where a part says of tests
that it names that a second verifier needs none of them (the second halves of 11.2, 11.3 and 11.4),
its ground is not that no proof can cause them: it is that, by this page's reading of the source, no
public file that has passed the earlier tests can make them fail either. Some tests under names with
`internal-` depend on sampled values, and so on a proof's bytes: the four groups of events in the
third piece of 11.1 (sampled values at which a verifier gives no acceptance), and the test on the
scalars' coefficients in the second halves of 11.2, 11.3 and 11.4. Every verifier needs those: 13.9
has them in one table.

##### The sumcheck with a hiding sum: values, and the claim that it leaves

*What defines a sumcheck.* A use of the sumcheck of 11.1 is defined by six things: (1) the start
form C; (2) for each variable, in the order of the rounds, the form of its round (equality at a
coordinate z_l, or plain) and the degree k_l; (3) d_Y and the number n of end values, both 0 or both
at least 1; (4) which map each end value is a value of; (5) S0, a public function of the sampled
values and of the end values as they were sent; (6) the kept form K, which is the form 0 in most
uses. The tables below give (1) to (6) for the uses that all sub-proofs share, and parts 11.2 to
11.4 for their own.

*The statement.* Let phi_l be the way in which variable l is summed: phi_l(g) = (1 - z_l) g(0) + z_l
g(1) for an equality round, phi_l(g) = g(0) + g(1) for a plain round. The sumcheck is about a
polynomial H(X_0, X_1, ..), the **summand**, which is a public function of the point and of the
values of some maps at the point. The statement is: phi_0, phi_1, .. applied to H, each in its
variable, give the value of C. The equality weights are in the phi_l and are not a factor of H.

*The variable Y.* If n > 0, the sumcheck has one further variable Y, the last one, with phi(g) =
g(0); it is treated as an equality round at z = 0 of degree d_Y, and its sampled element is beta. So
NV is the number of `PSC round` rounds plus 1 if n > 0, and then k_(NV-1) = d_Y and gamma_(NV-1) =
beta. There is no sumcheck with NV = 0.

*The numbers of its scalars.* Let j0 be the number of the first scalar, and sk(l) = k_0 + .. +
k_(l-1), so sk(0) = 0. The hiding polynomial's sum has the number j0. The i-th scalar, from 0, that
is sent in the round of variable l has the number j0 + 1 + sk(l) + i. The first end value has the
number jE = j0 + 1 + sk(NV), and end value number i, from 0, has jE + i. A hidden value that is sent
alone before the sumcheck (the claimed sum of a token sumcheck) has the number j0 - 1; it does not
belong to the sumcheck's numbers.

*What the verifier computes during the rounds.*

1. It computes pub(C). It reads hs, samples eps in `PSC epsilon`, and sets cur = hs + eps pub(C).
2. For each variable l = 0, 1, .. in the order of the rounds, Y last: it reads the k_l scalars, the
   coefficients of g = g_0 + g_1 X + .. + g_(k_l) X^(k_l), lowest degree first, without the one that
   is not sent. It computes that one from cur by the table of 11.1 (for Y: g_0 = cur). It samples
   gamma_l (for Y: beta, which is not 0). It sets cur = g(gamma_l).
3. If n > 0, it reads the n end values, and sets shown(jE + i) to end value i and factor(jE + i) =
   beta. In `PSC values` nothing is sampled.
4. For every number j from j0 to j0 + sk(NV), it sets shown(j) = 0 and factor(j) = -1.

From here on cur means its last value.

*The hiding polynomial.* It is U(X_0, .., X_(NV-1)) = sum over l < NV of u_l(X_l) times the product
over i > l of psi_i(X_i). Here psi_i(X) = 1 for an equality round and for Y, and psi_i(X) = X for a
plain round, so that phi_i(psi_i) = 1. u_l(X) = sum over deg of u(l, deg) X^deg, with deg = 1, ..,
k_l for l < NV - 1 and deg = 0, .., k_l for the last variable l = NV - 1. That makes 1 + sk(NV)
coefficients. They are mask entries:

u(l, deg) = mask(j0 + pos(l, deg)), with pos(l, deg) = sk(l) + deg - 1 for l < NV - 1, and pos(NV -
1, deg) = sk(NV - 1) + deg.

So the coefficients come by variable, in the order of the rounds, and within a variable by
increasing degree. They fill the numbers j0, .., j0 + sk(NV), which are the numbers of hs and of the
sent coefficients. A mask entry and the scalar that was sent under its number do not belong
together: if NV > 1, mask(j0) is u(0, 1), and it is not a mask of hs. Only the last variable has a
constant.

*What an honest prover sends.* hs is phi_0, .., phi_(NV-1) applied to U. Write val_i for the value
of shown map i at a point. Let H' be H with every val_i replaced by val_i + Y mask(jE + i), plus the
term (pub(C) - C) times the product of all psi_i(X_i). The polynomial g of variable l is U + eps H'
with X_i = gamma_i for i < l, X_l = X, and phi_i applied for i > l. End value i is val_i at the
point (gamma_0, gamma_1, ..) plus beta mask(jE + i), so hidden(jE + i) is the value of map i at that
point.

*The claim that a sumcheck leaves.* Let psi_l stand for psi_l(gamma_l): 1 for an equality round and
for Y, gamma_l for a plain round. Let tail_l be the product of psi_i over i = l + 1, .., NV - 1 (so
tail_(NV-1) = 1), and PP the product of all psi_l, l < NV. Then

Left(C; S0; K) = sum over l < NV, over the degrees deg of variable l, of gamma_l^deg tail_l
mask(j0 + pos(l, deg)) + eps PP (pub(C) - C) + eps K + eps S0 - cur.

The claim is Left(C; S0; K) = 0. It is the only test of the sumcheck. As a rule it is added as one
of the segment's claims. A segment can instead put the form Left(..) into the start form of a later
sumcheck (11.2, step 9); then it takes no claim number. S0 + K is the summand at the sampled point
with the end values put in as they were sent: S0 holds the terms that are public then, and K = sum
over i of ka_i (the value of kept map i at the point, as a form) holds the terms in maps whose value
is not shown, with public factors ka_i. (13.9 writes A_0 for S0 and A_i for ka_i.) With the formula
for pub(C) - C above, the claim's parts are:

| part of the form | value |
|---|---|
| constant | eps S0 - cur + eps const(K) + eps PP sum over j of C[j] shown(j) |
| on hidden(j0 + pos(l, deg)), which is a mask entry | gamma_l^deg tail_l |
| on hidden(j), for each j that C names | - eps PP C[j] |
| on A{w}[col], for each entry that C names | - eps PP C[A{w}, col] |
| on what K names | eps times K's coefficient |

If all rounds are equality rounds, every tail_l and PP are 1. In a sumcheck of plain rounds tail_l =
gamma_(l+1) gamma_(l+2) .. over the plain rounds after l, and PP is the product of all their
challenges.

*For each end value, a second claim.* Left(..) says nothing on what an end value is. A step that
shows an end value goes on in one of two ways. Either hidden(jE + i) enters the start form of a
later sumcheck. Or a claim of its own says: hidden(jE + i) - (the map's value at the point, as a
form) = 0. Each use below says which. The merge's end value is the one exception (below).

*The uses that the sub-proofs share.* z is the point of the step, gamma the sampled values. The
names in the last column are the routines of the programs' source, for a reader who has it.

| use | C | rounds, in order | d_Y, n | end values | S0 | K | in the programs' source |
|---|---|---|---|---|---|---|---|
| a product level with m variables, at the point z | given | m equality rounds of degree 2; round l is at z_(m - 1 - bit(l)), where bit(l) is the bit that it binds | 2, 2 | vL, vR | shown(jE) shown(jE + 1) | 0 | `readProductLevelRounds` |
| a token sumcheck at the point z of lt coordinates | hidden(j0 - 1), the claimed sum | lt equality rounds of degree 2; round l is at z_l; then the rounds that the segment appends (11.4, step 8: lPd equality rounds at 1/2; further round i binds bit lPd - 1 - i) | the segment's | the segment's | the segment's | the segment's | `tscRounds`, `tscIn` |
| a sumcheck of lam plain rounds of degree 2 (11.2, step 11) | the segment's | round l binds bit lam - 1 - l | the segment's | the segment's | the segment's | the segment's | `stackRounds` |
| attention's sumcheck over nb products, nv variables | the segment's | nv plain rounds of degree 3 | 2, 2 nb | the two factors of product 0, then those of product 1, and so on | sum over i < nb of W_i shown(jE + 2 i) shown(jE + 2 i + 1), W_i the public weight of product i at the sampled point | 0 | `sc3In`, `sc3Row` |
| attention's sumcheck with a degree for each round | the segment's | plain rounds of the degrees k_0, k_1, .., each from 1 to 64 | the segment's | the segment's | the segment's | the segment's | `scdIn` |
| the steps of a lookup | below | | | | | | `forestPad` |
| the merge | below | | | | | | `wsForkMerge` |

*A product level.* A **product tree** over 2^D leaves N_D(i), i < 2^D, has the levels N_m(i) =
N_(m+1)(i) N_(m+1)(i + 2^m) for m = D - 1, .., 0 and i < 2^m. So a vertex is the product of the two
vertices of the level below whose indices are its own index with a new highest bit, 0 or 1, and
N_0(0), the **root**, is the product of all leaves. The product level with m variables has a point z
of m coordinates and a start form C that stands for sum over i < 2^m of eq(z, i) N_m(i). Its summand
is N_(m+1)(0 | X) N_(m+1)(1 | X), each factor read as a map on the m low bits. Its end values are
vL, of the half with the new bit 0, and vR, of the half with the new bit 1. Let bit(0), bit(1), ..
be the order in which the bits are bound (11.1: the default order, the range order, or the order of
a lookup's last key level). Round l is an equality round at z_(m - 1 - bit(l)), and its challenge
gamma_l becomes the coordinate r_(m - 1 - bit(l)) of a point r of m coordinates. After `PSC values`
the round `LINE` samples xi. The level leaves:

1. the claim Left(C; shown(jE) shown(jE + 1); 0) = 0, in which the variables of the hiding
   polynomial are counted in the order of the rounds, not of the bits;
2. the point of the next level, z' = (xi, r_0, .., r_(m-1)), of m + 1 coordinates;
3. the start form of the next level, C' = (1 - xi) hidden(jE) + xi hidden(jE + 1).

With m = 0 the point z is empty, the sumcheck has the variable Y alone, and z' = (xi). A product
level outside a lookup runs the same rounds, `LINE` among them, and leaves claim 1. Its two factors
are maps that the segment's part names (for a sum of squares, one map twice), not the halves of a
tree. Its xi enters no formula: no z' and no C' are formed. The segment's part gives two further
claims, which say what hidden(jE) and hidden(jE + 1) are at the point r.

*Equality weights on a range of bits.* Take lam rounds of one sumcheck in which round l binds bit
lam - 1 - l of an index: the plain rounds of 11.2, step 11, or the further equality rounds of 11.4,
step 8. Write rs_q for the challenge of the round that binds bit q. For 0 <= lo <= hi <= lam and an
index i below 2^(hi - lo), eqlow(lo, hi, i) is the product over s < hi - lo of rs_(lo + s) if bit s
of i is 1, and of 1 - rs_(lo + s) otherwise. It equals eq((rs_(hi-1), .., rs_lo), i). (In the
programs' source: `eqLow`, `eqLowAt`.)

##### The lookup: what is looked up, the values, and the claims

The text under this heading follows the nine steps of a lookup in 11.1 and uses its sizes: n_c, k_c,
dT, kT, NC, cK, NT, rK, v = dT + rK, dL = kT + cK, lm. Under this heading j is the number of a
plane, from 0 to 31, as in 11.1; the number of a scalar has a name that begins with j (jZ, jBit).
G_j and A_L are values of the lookup, not the segments G_g and A_g of this section's head. phiB_e
and phiP are row weights, not the phi_l of a sumcheck. The letter e alone is 0 or 1; e_c is a weight
of call c. The segment whose sub-proof runs the lookup is called its **host**; the smallness test is
a host too.

| sign | meaning | public or hidden |
|---|---|---|
| tab_c(t) | the value of row t of call c's table, a number below p, t < 2^(k_c); for a range table tab_c(t) = t (11.1, public tables) | public |
| t0_c | the padding row of call c (11.1) | public |
| key_c(i) | key number i of call c, i < 2^(n_c); at a padding position it counts as tab_c(t0_c) | hidden |
| iota_c | n_c - dT; call c has 2^(iota_c) trees | public |
| tree0(c) | the number of call c's first tree | public |
| call(y) | the call that tree y belongs to, y < NT | public |
| npad(y) | the number of padding positions in tree y, y < NT | public |
| NPAD | all padding keys: (2^rK - NT) 2^dT + sum over y < NT of npad(y) | public |
| theta | the shift; 0 if NC = 1, where no `LKP shift` is run | public |
| alpha | the point of comparison | public |
| padkey_c | tab_c(t0_c) + c theta | public |
| a_pad | the padding's factor (step 2) | public |
| b(t, c, j) | bit j of the multiplicity m(t, c), t < 2^kT, c < NC. Each key of call c that is not padding is counted once, at one row t with tab_c(t mod 2^(k_c)) equal to the key; m(t, c) is the number of keys counted at t. Where a value occurs in several rows (a table with k_c < kT is repeated), a verifier accepts every choice of the row | hidden |
| Z | the root of the key side's product tree | hidden |
| G_j | the root of plane j's product tree | hidden |
| P_s | G_31 for s = 31, and P_(s+1)^2 G_s for s = 30, .., 0 | hidden |
| jLK | the number of the lookup's first scalar | public |
| jZ, jP_s, jG_j, jBit | the numbers of the end values for Z, P_s, G_j and of step 9's end value | public |
| zK_m, CK_m | the point and the start form of the key side's level with m variables | public; a form |
| zT_m, CT_m | the point and the start form of the table side's level with m variables, m = 5, .., 4 + dL; zT_(5 + dL) and CT_(5 + dL) are what the last level passes on | public; a form |
| zx, zy, zk_c | the parts of zK_v, and the key point of call c (11.1) | public |
| C_end | CK_v: the form of the key side's leaves at (zx \| zy) | a form |
| e_c | the weight of call c's trees at zy | public |
| KF_c | the key form of call c: key_c at zk_c, as a form; the host gives it | a form |
| zb_0, zb_1 | the two points (t \| c \| j), of 5 + dL coordinates each, at which the planes are read | public |
| T_c | call c's table at zb_0 | public |
| A_L | the factor of the planes' value in the leaves' summand | public |
| PL_0, PL_1 | the planes' value at zb_0 and at zb_1, as forms | forms |
| TB_0(col), TB_1(col), TP(col) | three weighted column sums of the planes' matrix, as forms; the host gives them | forms |
| LK0 .., LP0, LP1, LP2, LJ | the lookup's claims (below) | forms |

*The trees and the calls.* Sort the calls by decreasing iota_c; calls with equal iota_c keep the
order of the list. Going through the sorted calls, give each call c the next 2^(iota_c) numbers:
tree0(c) is the first of them, and call(y) = c for y = tree0(c), .., tree0(c) + 2^(iota_c) - 1.
tree0(c) is a multiple of 2^(iota_c). Key number i of call c is in tree y = tree0(c) + floor(i /
2^dT), at the index x = i mod 2^dT inside the tree.

*The key side's leaves.* The key side is one product tree over 2^v leaves. The leaf with the index
(x | y), that is x 2^rK + y, with x < 2^dT and y < 2^rK, is

- alpha - c theta - key_c((y - tree0(c)) 2^dT + x), with c = call(y), for y < NT;
- alpha - padkey_0 for y >= NT.

At a padding position the key is tab_c(t0_c), so the leaf is alpha - padkey_c. Which positions of a
call are padding is the host's to say, and npad(y) counts them tree by tree.

*The table side's leaves.* The table side has 2^(5 + dL) leaves. The leaf with the index (t | c |
j), that is (t 2^cK + c) 32 + j, with t < 2^kT, c < 2^cK, j < 32, is 1 + b(t, c, j) (alpha - 1 -
rowval(t, c)). Here rowval(t, c) = tab_c'(t mod 2^(k_c')) + c' theta with c' = min(c, NC - 1), and
b(t, c, j) = 0 for c >= NC. So a leaf is alpha - rowval(t, c) where the bit is 1, and 1 where it
is 0. G_j is the product of the leaves (t | c | j) over all t and c. The 32 values G_j are the level
with 5 variables of the table side; there is no level above it.

*The relation.* The lookup is about the equation Z = a_pad times the product over j < 32 of
G_j^(2^j). Why: the programs' source gives as the purpose of the shift that a key of one call cannot
pass as a row of another call's table.

*Tests on the sizes.* They read the public files alone. A verifier rejects, with the name in code
marks (NAME is the call's name), unless all of this holds:

- For each call: 5 <= k_c <= 16, t0_c < 2^(k_c), 0 <= n_c <= 62, and a table that has listed rows
  has 2^(k_c) of them (`shape-pt-words-NAME`; the programs also test under this name a number of
  their own for the kind of the call's function, which no file sets); every listed row's value is
  below p (`forest-row-value-NAME`); n_c >= 10 (`hid-call-floor`); the call has 32 planes and no row
  that is counted separately (`hid-planes-NAME`; in this format both hold of every call, so no file
  can cause this rejection).
- For the group: 1 <= NC <= 16 (`forest-calls`); dT >= 10 (`forest-depth`); NT <= 256
  (`forest-trees`); v < 32 (`forest-keys`); 2^(dL + 32) < p (`no-wrap-pin-forest`).
- For the padding: npad(y) <= 2^dT for every y < NT (`pt-pub-forest`); for each call the sum of
  npad(y) over its trees is below 2^(n_c) (`forest-only-padding-NAME`).
- For the planes' matrix: 5 <= lm <= kT + 5 (`forest-planes`).

Why: the programs' source gives as the purpose of v < 32 that a multiplicity of 32 bits has room for
all keys, and of 2^(dL + 32) < p that the sum of all multiplicities cannot wrap around in the field.

*The numbers of the lookup's scalars.* The lookup sends 187 + (v^2 + 4 v) + (dL^2 + 14 dL) + (2 dL +
6) + (2 dL + 14) scalars, with neighbouring numbers from jLK on.

| sumcheck | its j0 | its jE |
|---|---|---|
| step 3, tie step s = 0 | jLK | jLK + 4 |
| step 3, tie step s = 1, .., 30 | jLK + 7 + 6 (s - 1) | j0 + 4 |
| step 5, key level m = 0, .., v - 1 | jLK + 187 + m^2 + 4 m | j0 + 2 m + 3 |
| step 6, table level m = 5, .., 4 + dL | jLK + 187 + v^2 + 4 v + m^2 + 4 m - 45 | j0 + 2 m + 3 |
| step 7, the leaves | jLK + 187 + v^2 + 4 v + dL^2 + 14 dL | none |
| step 9, the bits | the leaves' j0 + 2 dL + 6 | j0 + 2 dL + 13 |

*Steps 1 and 2: theta, alpha, a_pad.* If NC > 1, theta is the element of `LKP shift`; else theta
= 0. alpha is the element of `LKP`. Then

a_pad = (alpha - padkey_0)^((2^rK - NT) 2^dT) times the product over y < NT of (alpha -
padkey_call(y))^(npad(y)).

*Step 3: the tie.* It has 31 sumchecks without a `PSC round`, s = 0, .., 30. Each has the variable Y
alone, with d_Y = 3, so NV = 1, and its hiding polynomial has the four coefficients mask(j0), ..,
mask(j0 + 3), of the degrees 0 to 3 in Y.

| s | start form C | end values, in the order sent | their numbers | S0 |
|---|---|---|---|---|
| 0 | 0 | Z, P_1, G_0 | jZ = jLK + 4, jP_1 = jLK + 5, jG_0 = jLK + 6 | shown(jZ) - a_pad shown(jP_1)^2 shown(jG_0) |
| 1, .., 30 | hidden(jP_s) | P_(s+1), G_s | jP_(s+1) = j0 + 4, jG_s = j0 + 5 | shown(jP_(s+1))^2 shown(jG_s) |

G_31 is P_31, so jG_31 = jP_31. K is 0 in all of them. Written out, with beta and eps those of step
s, g_1, g_2, g_3 the three scalars of its `PSC Y`, and cur = hs + eps pub(C) + g_1 beta + g_2
beta^2 + g_3 beta^3, the claim of tie step s is

mask(j0) + beta mask(j0 + 1) + beta^2 mask(j0 + 2) + beta^3 mask(j0 + 3) + eps (pub(C) - C) + eps
S0 - cur = 0.

For s = 0, pub(C) - C is 0. For s >= 1 it is shown(jP_s) - hidden(jP_s), which is the beta of step
s - 1 times mask(jP_s).

*Step 4: the planes' roots at one point.* `JOIN` samples 5 elements, the coordinate of bit 0 of j
first. zT_5 is these five in reverse order, the coordinate of bit 4 first. CT_5 = sum over j < 32 of
eq(zT_5, j) hidden(jG_j).

*Step 5: the key side.* zK_0 is the empty point and CK_0 = hidden(jZ). For m = 0, .., v - 1 the
product level with m variables runs at the point zK_m with the start form CK_m, as described above
under a product level. It gives zK_(m+1) = (xi, r_0, .., r_(m-1)) and CK_(m+1) = (1 - xi)
hidden(jE) + xi hidden(jE + 1), with xi, r and jE those of level m. The levels m < v - 1 bind the
bits m - 1, m - 2, .., 0. The level m = v - 1 binds the bits rK + 2, rK + 1, rK, rK + 5, rK + 4,
rK + 3, rK + 8, rK + 7, rK + 6, then v - 2, v - 3, .., rK + 9, then rK - 1, .., 0. After the last
level: zx is the first dT coordinates of zK_v, zy the last rK, and C_end = CK_v.

*Step 6: the table side.* For m = 5, .., 4 + dL the product level with m variables runs at zT_m with
the start form CT_m, in the default order, and gives zT_(m+1) and CT_(m+1) in the same way. zT_(5 +
dL) has 5 + dL coordinates: first the kT of t, the highest bit first, then the cK of c, then the 5
of j.

*Step 7: the leaves.* It is a sumcheck with the start form CT_(5 + dL) and 5 + dL equality rounds.
Round l, l = 0, .., 4 + dL, is at coordinate number l of zT_(5 + dL), so the highest bit of t is
bound first. It has degree 2 for l < dL and degree 1 for l >= dL. It has no variable Y, so NV = 5 +
dL, and the last variable, bit 0 of j, is the one whose part of the hiding polynomial has a
constant: u(4 + dL, 0) = mask(j0 + 2 dL + 4) and u(4 + dL, 1) = mask(j0 + 2 dL + 5). zb_0 =
(gamma_0, .., gamma_(4 + dL)). Its summand is 1 + (the planes at the point) (alpha - 1 - (the rows
at the point)), so S0 = 1 and K = A_L PL_0, with A_L and PL_0 as below.

*Step 8 and step 9: the bits.* `ZERO` samples 5 + dL elements, the coordinate of bit 0 of j first;
zq is these in reverse order. Step 9 is a sumcheck with the start form 0 and 5 + dL equality rounds
of degree 2, round l at coordinate number l of zq; d_Y = 2; one end value, with the number jBit.
zb_1 = (gamma_0, .., gamma_(4 + dL)), without beta. Its summand is b^2 - b, so S0 = shown(jBit)^2 -
shown(jBit) and K = 0. hidden(jBit) stands for the planes' value at zb_1.

*What the verifier computes after step 9.* Split zb_e, e = 0 or 1, into zbt_e, its first kT
coordinates, zbc_e, the next cK, and zbj_e, the last 5.

1. For each call c: h_c = rK - iota_c, and e_c = eq((zy_0, .., zy_(h_c - 1)), tree0(c) /
   2^(iota_c)). With h_c = 0, e_c = 1.
2. For each call c: zk_c = (zy_(h_c), .., zy_(rK - 1), zx_0, .., zx_(dT - 1)), of n_c coordinates.
3. For each call c: let zu be the last k_c coordinates of zbt_0. T_c = sum over t < 2^(k_c) of
   eq(zu, t) tab_c(t). For a range table that is T_c = sum over i < k_c of 2^(k_c - 1 - i) zu_i.
4. A_L = alpha - 1 - sum over c < 2^cK of eq(zbc_0, c) (T_c' + c' theta), with c' = min(c, NC - 1).

*The planes' value as a form.* For e = 0 and 1,

PL_e = sum over t < 2^kT, c < NC, j < 32 of eq(zb_e, (t | c | j)) b(t, c, j).

The numbers c from NC on have no term. Some hosts keep the planes in a matrix MB of 2^lm rows and NC
2^wd columns, wd = kT + 5 - lm. By 11.1 the bit (t, c, j) is in row (t mod 2^(lm - 5)) 32 + j and
column c 2^wd + floor(t / 2^(lm - 5)). There the sum is taken in two steps. The weights on the rows
are

- phiB_e(row) = eq((the last lm - 5 coordinates of zbt_e | zbj_e), row), for row < 2^lm;
- phiP(row) = 2^(row mod 32).

The host gives, as forms, TB_e(col) = sum over row < 2^lm of phiB_e(row) MB[row][col] and TP(col) =
sum over row < 2^lm of phiP(row) MB[row][col], for each column col < NC 2^wd. Then

PL_e = sum over c < NC, th < 2^wd of eq(zbc_e, c) eq((the first wd coordinates of zbt_e), th) TB_e(c
2^wd + th).

In feed-forward and attention segments and in the output head, TB_0, TB_1 and TP are three parts of
`U3` on the matrix b, bA or bH, with the weight vectors phiB_0, phiB_1, phiP. The smallness test
reads its planes from `BZ` directly, by a rule of its own (the third piece of 11.1).

*The claims on mask entries alone.* There are 31 + v + dL + 1 of them. A host adds them as one run
of neighbouring claim numbers, in this order. The column no. counts from the first of them.

| no. | name | claim |
|---|---|---|
| 0 | LK0 | tie step 0: Left(0; shown(jZ) - a_pad shown(jP_1)^2 shown(jG_0); 0) = 0 |
| s = 1, .., 30 | LKs | tie step s: Left(hidden(jP_s); shown(jP_(s+1))^2 shown(jG_s); 0) = 0 |
| 31 + m, m = 0, .., v - 1 | LK(31 + m) | key level m: Left(CK_m; shown(jE) shown(jE + 1); 0) = 0, jE that of the level |
| 26 + v + m, m = 5, .., 4 + dL | LK(26 + v + m) | table level m: Left(CT_m; shown(jE) shown(jE + 1); 0) = 0, jE that of the level |
| 31 + v + dL | LK(31 + v + dL) | the bits: Left(0; shown(jBit)^2 - shown(jBit); 0) = 0 |

Each Left(..) is formed with j0, eps, the gamma_l and cur of its own sumcheck. All start forms here
name numbers of scalars alone, so these claims have no term on an array.

*The three claims on the planes.* A host adds them as one run of three neighbouring claim numbers,
in this order.

| no. | name | claim |
|---|---|---|
| 0 | LP0, the leaves | Left(CT_(5 + dL); 1; A_L PL_0) = 0, with j0, eps, the gamma_l and cur of step 7 |
| 1 | LP1, the planes at the bits' point | hidden(jBit) - PL_1 = 0 |
| 2 | LP2, the number of keys | sum over col < NC 2^wd of TP(col) - (2^v - NPAD) = 0; that is, sum over t, c < NC, j of 2^j b(t, c, j) = 2^v - NPAD |

Written out, with eps_L the eps of step 7, LP0 is: sum over l < 5 + dL and the degrees deg of
variable l of gamma_l^deg mask(j0 + pos(l, deg)) + eps_L (pub(CT_(5 + dL)) - CT_(5 + dL)) + eps_L
A_L PL_0 + eps_L - cur = 0. Here CT_(5 + dL) = (1 - xi) hidden(jE) + xi hidden(jE + 1), with xi and
jE those of the table level m = 4 + dL.

*The join.* It ties the key side's leaves to the calls' keys. It is a claim, and it is not the round
`JOIN` of step 4. For each call the host gives the **key form** KF_c, a form that stands for sum
over i < 2^(n_c) of eq(zk_c, i) key_c(i), with tab_c(t0_c) as the key at every padding position. The
join is the form

LJ = sum over c < NC of e_c (alpha - c theta - KF_c) + (alpha - padkey_0) times the sum over y = NT,
.., 2^rK - 1 of eq(zy, y) - C_end,

and the claim is LJ = 0. It is one claim. The output head and an attention segment add it as a claim
of its own. The smallness test adds the same claim, written on `ZZ`. A feed-forward segment has no
claim for it: there the join is inside the start form of a later sumcheck, and the terms of the key
forms that are products are what that sumcheck sums (11.2). The host's part says at which claim
numbers the three groups are.

*The padding is described twice.* The numbers npad(y) enter a_pad and NPAD. The positions at which a
host's key form puts the padding key in must be the same in number, tree by tree. Both follow from
the public files. The programs count the positions again where the key forms are made and end the
run under the name `internal-forest-padding-places` if the two differ. No proof can cause that. The
source states the limit of this test (`verifier/src/vf_lookup_argument.h`): the positions are counted by
loops of the host's own beside its key forms, not inside them, and in a feed-forward segment and in
the output head both descriptions go back to the lists of columns of the calls' segments. They are
two descriptions from one source, and a slip inside a form is not seen. By the same comment the
routine that closes a sub-proof's list of claims (`settle`) does not ask whether the comparison was
made.

*What must be fixed before step 1.* The programs' source states three duties of the host
(`verifier/src/vf_lookup_argument.h`, at the head), and says why: the lookup's own code enters none of the
values below into the chain, and a prover that can choose one after a challenge is sampled proves a
false lookup. The third duty is that all the lookup's claims are added (above, the three groups):
the programs end the call under a name `internal-claim-not-added-..` (`NOT CHECKED`, 14.3) unless
each group is marked as added, and by the same comment a mark does not see what was added, if
anything. The first two are about the order. First, the following are committed, and their
out-of-domain samples answered, before theta is sampled (before alpha, if NC = 1):

- the planes;
- every array from which the keys are read;
- the mask row.

Second, the following must be fixed by what the chain holds before the same round:

- the order and number of the calls;
- n_c, k_c, the table and t0_c of each call;
- the line `lidb`;
- every npad(y);
- lm, which committed matrix the planes are, in which row and column of it each bit is, and that it
  has no column for a number c from NC on;
- what each key form reads;
- the root and the length of the mask row;
- jLK;
- the number of claims that come before the lookup's first;
- that every call has 32 planes and no row that is counted separately;
- the field and the hash.

All of them but the root are computed from the public files. The source warns that the chain's plan
does not fix them: a lookup enters the plan with v, dL, the number of planes and whether theta is
sampled. So (dT, rK) = (10, 2) and (11, 1), or (kT, cK) = (6, 1) and (5, 2), have one plan but other
orders of rounds, other weights and other tables. That comment says what must be fixed and not what
fixes it. By this page's reading the digest of the public files in the chain's first query (the item
`configuration`) fixes it.

##### The merge: values and claim

*What goes in.* A segment with claims on the committed weights has NA weight claims, in the order of
its part (11.2, 11.4; NA is 11.5's letter, and 11.2 writes NCL for it). Weight claim a, a = 0, ..,
NA - 1, has a hidden value hidden(jV_a), which 11.5 writes v_a: an end value of an earlier sumcheck,
or a hidden value sent alone. All of them are sent before the merge's `BAT-sigma`.

*The rounds.* `BAT-sigma`, with n = NA, samples one element x, also if NA = 1. (x is the letter of
11.1 and 11.5 for this element; under the lookup's heading x was an index inside a tree.) The start
form is C = sum over a < NA of x^a hidden(jV_a). The sumcheck has lkW plain rounds of degree 2, then
d_Y = 1 and one end value (11.5 writes lk for lkW). Write rm_l for the challenge of round l; rm_0 is
the coordinate of the highest column bit of the stored table. The whole summand counts as one map fv
of the column's index, and this map is shown. The end value fv' has the number jFv = j0 + 2 lkW + 2.
What fv is, and the row on the stored table that says so, are in 11.5 and 13.9.

*The claim.* S0 = shown(jFv) = fv' and K = 0. The merge's claim is Left(C; fv'; 0) = 0. In a
feed-forward segment and in the output head it is the segment's last claim. Here NV = lkW + 1, psi_l
= rm_l for l < lkW and 1 for Y, tail_l = rm_(l+1) rm_(l+2) .. rm_(lkW - 1), and PP = rm_0 rm_1 ..
rm_(lkW - 1). Written out:

sum over l < lkW of tail_l (rm_l mask(j0 + 2 l) + rm_l^2 mask(j0 + 2 l + 1)) + mask(j0 + 2 lkW) +
beta mask(j0 + 2 lkW + 1) + eps PP (pub(C) - C) + eps fv' - cur = 0, with pub(C) - C = sum over a <
NA of x^a factor(jV_a) mask(jV_a).

In a plain round the two scalars sent are g_0 and g_2, and g_1 = cur - 2 g_0 - g_2 (11.1).

*What goes out.* No claim of the segment names hidden(jFv): its coefficient LS(jFv) is 0, so MC(jFv)
= 0, and mask(jFv) masks nothing. The statement fv(rm) + beta cm = fv' leaves the sub-proof as a row
on the stored table, where cm is an entry of the request's index-mask message (11.1; 13.9, after
each segment with claims on the weights, writes c for it). The merge passes on four things for that
row: fv', beta, the point (rm_0, .., rm_(lkW - 1)), which the frame takes in reverse order, and x,
whose powers enter the row's weights (11.5).

##### Easy to get wrong in this piece

1. The mask entries with the numbers of hs and of the sent coefficients are the coefficients of the
   hiding polynomial, at the positions pos(l, deg). They do not mask the scalars that were sent
   under these numbers.
2. pub(C) leaves out the terms of C on arrays. These terms come back in the claim, inside eps PP
   (pub(C) - C).
3. S0 has no factor eq(z, gamma). The equality weights are in the way the variables are summed. So
   for an honest prover the last cur is U at the sampled values plus eps times H' at the sampled
   values, with no equality factor.
4. The variables of a hiding polynomial are counted in the order of the rounds. In a level that uses
   the range order, or the order of a lookup's last key level, this is not the order of the bits.
   The point that the level passes on is in the order of the bits.
5. With plain rounds tail_l and PP are products of challenges. With equality rounds alone they
   are 1.
6. The index of a leaf of the key side is (x | y): the tree's number y is in the low bits. The index
   of a key inside its call is (the tree's number inside the call | x): there it is in the high
   bits. So zk_c has coordinates of zy first and then zx.
7. For a number c from NC on, the table side has the last call's rows with the last call's shift,
   (NC - 1) theta. The planes have nothing there.
8. A table with k_c < kT is read at the last k_c coordinates of zbt_0.
9. G_31 has no end value of its own: jG_31 = jP_31, the first end value of tie step 30.
10. The trees from NT on hold the padding key of call 0, whichever call has the highest tree
    numbers.
11. The leaves' sumcheck has S0 = 1, not 0, and its claim is formed with the three claims on the
    planes, not with the claims on mask entries alone.

#### The smallness test: values, equations and claims

The first piece of 11.1 gives the rounds of the smallness test, the sizes of its arrays and the
number and order of its claims. This piece gives five things: what each cell of the arrays holds;
which sampled entry multiplies what; how the keys of the lookup on the digits are formed; every
equation, with its number and its power of chi; and every claim, as a formula. It needs the first
and the second piece of 11.1. The second piece defines a hidden value, a form, a combined row of an
array, the claim that a sumcheck leaves, and the claims of a lookup. Parts 11.2 to 11.4 give, for
each kind of segment, the list of matrices and the parts of `U3` with their weight vectors.

*What is public and what is hidden.* Public (second piece of 11.1): every size, index and number of
this piece; the sampled matrix Theta; chi; the weight vectors phi_f; all weights on rows and
columns. Hidden: the cells, and every entry of the arrays `M`mu, `U3`, `ZZ` and `BZ`. In this piece
the verifier is sent no entry of these arrays. It only forms the coefficients with which the entries
enter the segment's one equation (second piece of 11.1).

##### Notation

The letters tau, tauP, NM, NX, NCm, Q, mz, m_mu, k_mu, off(mu), first_mu, n_Z, v_Z, chi and clog2
are those of the first piece of 11.1. The letters hidden(j), A{w}[col], Left(C; S0; K), alpha,
theta, a_pad, npad(y), NPAD, jLK, e_c, key_c(i), KF_c, zx, zy, zk_c, C_end, CT_(5 + dL), zb_0, zb_1,
T_c, A_L, eps_L, jBit, PL_0, PL_1, LK, LJ and LP are those of the second piece of 11.1. eq(z, i) is
the equality weight of the index i at the point z, with the coordinate of the highest bit first
(11.1, the order of bits). z[s] is coordinate number s, from 0, of a point z, which the second piece
writes z_s. Theta, with a capital, is the sampled matrix; theta is the lookup's shift. b alone is a
block of cell rows; b(t, c, j) is a bit; the matrix b is 11.2's.

| sign | meaning | public |
|---|---|---|
| mu, nu | the number of a matrix in the segment's list, 0 .. NM - 1 | yes |
| i | a cell row of a matrix, 0 .. m_mu - 1 | yes |
| i' | a packed row of a matrix, 0 .. mp_mu - 1 | yes |
| mp_mu | ceil(m_mu / 3), the number of packed rows of matrix mu | yes |
| c | a column of a matrix, 0 .. k_mu - 1 | yes |
| u | a combination, 0 .. tau - 1 | yes |
| crow | off(mu) + i, the number of cell row i of matrix mu among all cell rows of the list, 0 .. mz - 1 | yes |
| col0(mu) | the sum of k_nu over the matrices nu before mu: the number of the matrix's first column among all columns of the list | yes |
| cell(mu, i, c) | the cell in row i and column c of matrix mu: an integer | no |
| M_mu[i'][c] | the entry in row i' and column c of the array of matrix mu, an element of F | no |
| U3[n] | entry n of the vector `U3`, an element of E | no |
| ZZ[u][col], BZ[row][col] | the entries of the arrays `ZZ` and `BZ`, elements of F | no |
| Theta_mu[c][u] | the sampled entry in {-1, 0, 1} of column c of matrix mu and combination u | yes |
| comb(mu, i, u) | the sum over c < k_mu of Theta_mu[c][u] cell(mu, i, c): combination u of cell row i, an integer | no |
| val(u, crow) | ZZ[u][crow] + 2^10 ZZ[u][mz + crow] - 2^17: the value that the two digits of (u, crow) stand for | no |
| f | the number of a part of `U3`, in the order of the segment's list of parts, from 0 | yes |
| mat(f), len(f), phi_f, at(f), nb(f) | of part f: its matrix; the length of its weight vector; the weight vector, len(f) elements of E; the number of its first entry in `U3`; nb(f) = m_mat(f) / len(f), its number of blocks | yes |
| b | a block of a part, 0 .. nb(f) - 1 | yes |
| q | the number of a read, 0 .. NX - 1 | yes |
| pw | the row weights pw(u) = chi^u on the tau rows of `ZZ` | yes |
| rw_mu | the row weights rw_mu(i') = chi^(tauP i') on the mp_mu rows of the array of matrix mu | yes |
| cellw_mu(i) | rw_mu(floor(i / 3)) 2^(20 (i mod 3)): the weight of cell row i in the claim of matrix mu | yes |
| colw_mu(c) | the sum over u < tau of chi^u Theta_mu[c][u]: the weight of column c | yes |
| spw | the sum over u < tau of chi^u | yes |
| lc | clog2(mz) | yes |
| tau0, tau1 | tau0 is the largest power of two that is not above tau; tau1 = tau - tau0, which is 0 or a power of two | yes |
| n0 | the number of claims that the segment has added before the first claim of the smallness test | yes |
| nJ | n0 + 43 + v_Z, the number of the join among the segment's claims | yes |
| r | an index into a weight vector phi_f, 0 .. len(f) - 1 | yes |
| s | the number of a coordinate of a point, or of a bit | yes |
| x | a key number of a call, or the number of an equation | yes |
| blk | a block of rows of `ZZ`, 0 or 1 (the lookup on the digits, below), with R_blk, row0_blk, a_blk, h_blk, zk_top,blk, zk_row,blk, zk_col and the row weights kw_blk | yes |
| X | (2 t + c) 32 + j, the index of a bit of a multiplicity; here, and only in (t, c, j), c is a call's number, as in the second piece of 11.1 | yes |
| bw_0, bw_1 | row weights on the 32 rows of `BZ` (below) | yes |
| KF_0, KF_1 | the key forms of the two calls (second piece of 11.1), which this piece gives (below) | a form (coefficients public, value hidden) |
| EP(mu, i', u), ER(f, b, u) | the equation of a packed row; the equation of a read (below) | a form (coefficients public, value hidden) |
| TIE_mu, READ_q | the claim of a matrix; the claim of a read (below) | a form (coefficients public, value hidden) |

All powers of chi are taken in E, and chi^0 is 1. An integer stands for its residue modulo p, as an
element of F, and an element x of F stands for the element (x, 0, 0) of E.

##### Constants

| value | what it is | name in the source |
|---|---|---|
| 3 | cell rows in one packed row | `LANES` |
| 20; 2^20 | the radix of the packing, as a logarithm and as a number | `LRAD`; `RADB` |
| 17 | the logarithm of the offset 2^17 of a combination | `LBZ` |
| 10 | bits of the low digit | `T2D0` |
| 8 | bits of the high digit, 17 + 1 - 10 | `T2D1` |
| 8191 | the largest number of columns of a matrix | `KCAP` |
| 32 | rows of `BZ`, and planes of a multiplicity | the 32 in `addTreesF` |
| tau | the number of combinations, the parameter file's line `tau` | `TAU`, `T2TAU` |
| `R10z`, `R8d` | the programs' names of the two calls of the lookup on the digits: call 0, the low digits, and call 1, the high digits | `piecesF` |

##### The cells, the arrays of the matrices, `U3` and `ZZ`

*The array of a matrix.* The array of matrix mu has mp_mu rows and k_mu columns over F. Its entry is
M_mu[i'][c] = cell(mu, 3 i', c) + 2^20 cell(mu, 3 i' + 1, c) + 2^40 cell(mu, 3 i' + 2, c), read in
F, where a cell row from m_mu on counts as 0. A negative cell enters as a negative integer, so the
sum is formed over the integers and then reduced modulo p. Packed row i' is row i' of the array, and
column c is column c of the array. (The source's routine `placeP` is written for J packed rows in
one row of the array; the programs use J = 1 alone.) A matrix that the segment commits to itself has
the name `M`mu in the segment's layout. For a matrix that an earlier segment has committed to, M_mu
is that segment's array `M`mu', where mu' is the matrix's number in the earlier segment's list; it
is read whole, with the same m and k, from its first column on.

*The vector `U3`.* It is one row of elements of E. The parts of `U3` follow one another without a
gap: at(0) = 0 and at(f + 1) = at(f) + k_mat(f) nb(f), and the length of `U3` is the sum of k_mat(f)
nb(f) over all parts. The entry of part f for column c and block b is U3[at(f) + c nb(f) + b]. What
it is meant to be: the sum over r < len(f) of phi_f[r] cell(mat(f), b len(f) + r, c). In a form,
U3[n] is named as it is: `U3` has one row, so its only combined row has the row weight 1, and U3[n]
is that row's column n. Every segment has `U3` alone. (The source has room for vectors U1 and U2
too.)

*The array `ZZ`.* It has tau rows and 2 mz columns over F. Take the cell row with the number crow =
off(mu) + i, and a combination u. ZZ[u][crow] is the low digit and ZZ[u][mz + crow] the high digit
of comb(mu, i, u) + 2^17. So that number is ZZ[u][crow] + 2^10 ZZ[u][mz + crow]. The low digit is
from 0 to 2^10 - 1, the high digit from 0 to 2^8 - 1. So val(u, crow) is meant to be comb(mu, i, u).
Digits exist only if -2^17 <= comb(mu, i, u) < 2^17: the combination's **window**.

##### Where a bit of a multiplicity is in `BZ`

`BZ` has 32 rows and 2^11 columns over F. It holds the bits b(t, c, j) of the multiplicities of the
lookup on the digits (second piece of 11.1). Here t < 2^10 is a row of the table, c < 2 a call and j
< 32 a plane. In this paragraph c is a call's number, not a column. Call 1's table has 2^8 rows and
is repeated four times: row t of call 1 has the value t mod 2^8 (second piece of 11.1). A high digit
can be counted in any one of its four rows; no claim tells them apart. Let X = (2 t + c) 32 + j, a
number of 16 bits: t is its top ten bits, c the next bit and j its low five bits. Then b(t, c, j) =
BZ[floor(X / 2^11)][X mod 2^11]. So the row of `BZ` is floor(t / 32), the top five bits of t, and
the column is (2 (t mod 32) + c) 32 + j. This is not the rule of the matrices b, bA and bH (11.1,
where the multiplicities are). `BZ` is not packed and is not a matrix of the list.

##### The sampled matrix

The round `RNG` samples (the sum of all k_mu) tau entries by R6.7, as one sequence. Entry number
(col0(mu) + c) tau + u of the sequence, counted from 0, is Theta_mu[c][u]. So the entries come
matrix by matrix in the list's order, the matrices of earlier segments among them, within a matrix
column by column, and within a column by increasing u. Theta_mu[c][u] multiplies column c of matrix
mu in combination u. The same entry multiplies every cell row of that column, the packed entries
M_mu[i'][c] of that column, and the entries of `U3` that belong to that column in every part on
matrix mu.

*What comes before the round.* The arrays whose cells the test is about are committed before `RNG`,
and the answers to their out-of-domain samples are given before it. They are the segment's own
arrays `M`mu, `U3`, and the arrays of earlier segments; these last are sent again in the segment's
first prover's message (R14.4). The list of parts with its weight vectors is fixed too (below). `ZZ`
and `BZ` are committed after `RNG` (first piece of 11.1, step 2). By the source's comments the
round's error 2^-tau rests on this order: a prover that knows the matrix before it sends `U3` could
add to a read a vector that the matrix maps to 0, and every equation would be met all the same,
although the read is then not a sum of small cells. A verifier needs no test of its own for this
order. It follows from the order of the rounds in the plan, and from the table of moments in the
first piece of 11.1 ("The arrays of a segment's smallness test"), which is part of the format: the
layout file places an array inside its moment and cannot move it to another moment. So a second
verifier takes an array's moment from that table and never from the layout file. The programs test
the order all the same, directly before the matrix is sampled, and end the call under the name
`internal-f-matrix-order`, which is printed after the sub-proof's prefix (13.8); the source counts a
failure as a fault of the program, so such a call ends `NOT CHECKED` (14.3).

##### The lookup on the digits

*Its calls.* The lookup has NC = 2 calls, in this order. Call 0 (`R10z`) has the range table of 2^10
rows, tab_0(t) = t, and its keys are the low digits. Call 1 (`R8d`) has the range table of 2^8 rows,
tab_1(t) = t, and its keys are the high digits. The padding row of both is t0 = 0, so before the
shift a padding key is 0.

*The keys.* Each call has 2^(n_Z) key numbers, n_Z = max(10, clog2(tau) + lc). For u < tau and crow
< mz: key_0(u 2^lc + crow) = ZZ[u][crow] and key_1(u 2^lc + crow) = ZZ[u][mz + crow]. Every other
key number of either call is a padding position: the numbers u 2^lc + x with u < tau and mz <= x <
2^lc, and all numbers from tau 2^lc on. So each call has tau mz keys that are not padding and
2^(n_Z) - tau mz padding positions. Nothing of `ZZ` is read at a padding position.

*Its numbers.* With the rules of 11.1: kT = 10, cK = 1, dL = 11; dT = min(lidb, n_Z), where lidb is
the line `lidb`, or 22 if the line is absent; iota_0 = iota_1 = n_Z - dT; NT = 2^(n_Z - dT + 1); rK
= n_Z - dT + 1; v_Z = dT + rK = n_Z + 1. Call 0 has the trees 0 .. 2^(n_Z - dT) - 1, and call 1 the
trees from 2^(n_Z - dT) to NT - 1. NT is 2^rK, so there is no tree of padding alone. Key number x of
a call is in the call's tree number floor(x / 2^dT), at the index x mod 2^dT inside the tree, and
npad(y) is the number of padding positions that fall in tree y.

*What follows for the values of the second piece of 11.1.*

1. a_pad = alpha^(2^(n_Z) - tau mz) (alpha - theta)^(2^(n_Z) - tau mz).
2. e_0 = 1 - zy[0] and e_1 = zy[0], where zy[0] is the first of the rK coordinates of zy.
3. Both calls have the same key point zk = zk_0 = zk_1, of n_Z coordinates: zy[1], .., zy[rK - 1],
   and then the dT coordinates of zx.
4. T_0 = the sum over s = 0, .., 9 of 2^(9 - s) zb_0[s], and T_1 = the sum over s = 0, .., 7 of
   2^(7 - s) zb_0[2 + s]. zb_0 and zb_1 have 16 coordinates: ten of t, one of c, five of j.
5. A_L = alpha - 1 - ((1 - zb_0[10]) T_0 + zb_0[10] (T_1 + theta)).
6. NPAD = 2 (2^(n_Z) - tau mz), so the number of keys that are not padding, 2^(v_Z) - NPAD, is 2 tau
   mz.
7. jLK, the number of the lookup's first scalar, is the number of scalars that the segment has sent
   before the round `RNG`.

*The key forms.* KF_0 = the sum over u < tau and crow < mz of eq(zk, u 2^lc + crow) ZZ[u][crow].
KF_1 = the same sum with ZZ[u][mz + crow]. The index u 2^lc + crow is read as a number of n_Z bits.
A padding position adds nothing, because its key tab_c(t0_c) is 0.

*The same sums as combined rows.* The weight eq(zk, u 2^lc + crow) is a product of a factor that
depends on u and a factor that depends on crow, once the rows are cut into blocks whose size is a
power of two. Block 0 is the rows 0 .. tau0 - 1. Block 1, only if tau1 > 0, is the rows tau0 ..
tau - 1. For block blk (0 or 1) write R_blk for its number of rows (tau0 or tau1), row0_blk for its
first row (0 or tau0), a_blk = log2(R_blk) and h_blk = n_Z - a_blk - lc. Cut zk into three runs:
zk_top,blk, its first h_blk coordinates; zk_row,blk, the next a_blk; zk_col, the last lc, which is
the same for both blocks. Then for a row u of block blk: eq(zk, u 2^lc + crow) = eq(zk_top,blk,
row0_blk / R_blk) eq(zk_row,blk, u - row0_blk) eq(zk_col, crow). The number row0_blk / R_blk is 0
for block 0 and tau0 / tau1 for block 1. Let kw_blk be the row weights on `ZZ` with kw_blk(u) =
eq(zk_row,blk, u - row0_blk) on the rows of block blk and 0 on all other rows. Then KF_0 = the sum
over the blocks blk of eq(zk_top,blk, row0_blk / R_blk) times the sum over crow < mz of eq(zk_col,
crow) ZZ{kw_blk}[crow], and KF_1 is the same with ZZ{kw_blk}[mz + crow].

*The planes at their two points.* The test reads its planes from `BZ` directly: it has no TB_0, TB_1
and TP (second piece of 11.1). PL_0 = the sum over X < 2^16 of eq(zb_0, X) BZ[floor(X / 2^11)][X mod
2^11], and PL_1 is the same sum with zb_1. Let bw_0 be the row weights on the 32 rows of `BZ` with
bw_0(row) = eq(the first 5 coordinates of zb_0, row), and bw_1 the same with zb_1. Then PL_0 = the
sum over col < 2^11 of eq(the last 11 coordinates of zb_0, col) BZ{bw_0}[col], and PL_1 = the sum
over col < 2^11 of eq(the last 11 coordinates of zb_1, col) BZ{bw_1}[col].

##### The equations

The test has two families of equations over E. Each is linear in the entries of the committed
arrays, with public coefficients. No equation is tested alone: all of them enter the segment's one
equation, each under a power of chi.

*The equation of a packed row.* For each matrix mu, each packed row i' < mp_mu and each combination
u < tau, with crow = off(mu) + 3 i':

      EP(mu, i', u):   val(u, crow) + 2^20 val(u, crow + 1) + 2^40 val(u, crow + 2) - (the sum over c < k_mu of Theta_mu[c][u] M_mu[i'][c]) = 0.

A term val(u, crow + 1) or val(u, crow + 2) whose cell row 3 i' + 1 or 3 i' + 2 is not below m_mu is
left out, with its constant -2^17. (The source calls the sum of the equations of matrix mu the tie
of matrix mu.)

*The equation of a read.* A read is a pair (f, b) of a part of `U3` and a block (11.1). For each
read and each combination u < tau, with mu = mat(f) and crow = off(mu) + b len(f):

      ER(f, b, u):   (the sum over r < len(f) of phi_f[r] val(u, crow + r)) - (the sum over c < k_mu of Theta_mu[c][u] U3[at(f) + c nb(f) + b]) = 0.

*The numbers of the reads.* The reads are numbered q = 0, .., NX - 1 in this order: by matrix mu =
0, .., NM - 1; within a matrix by the parts f with mat(f) = mu, by increasing f; within a part by
increasing b. So NX is the sum of nb(f) over all parts.

*The numbers of the equations.* EP(mu, i', u) has the number first_mu + i' tauP + u, with first_0 =
0 and first_(mu + 1) = first_mu + tauP 2^clog2(mp_mu). ER of read number q and combination u has the
number first_NM + q tauP + u. The numbers with u from tau to tauP - 1, and those with i' from mp_mu
to 2^clog2(mp_mu) - 1, name no equation. The equations take Q = first_NM + NX tauP numbers.

*The powers.* Equation number x gets the coefficient chi^(NCm + x). NCm is the number of all claims
of the segment, those of the smallness test among them. So the powers chi^0 .. chi^(NCm - 1) belong
to claims and the powers chi^NCm .. chi^(NCm + Q - 1) to equations.

##### The claims, in order

The test adds 47 + v_Z + NM + NX claims to the segment's list, one after the other, at the position
in that list that 11.2 to 11.4 give. The first has the number n0. A claim is a form that must be 0.

| number | claim | the form | coefficient in the segment's one equation |
|---|---|---|---|
| n0 .. n0 + 42 + v_Z | LK0 .. LK(42 + v_Z): the lookup's claims on mask entries alone: 31 of the tie of the two sides' roots, v_Z of the key side's levels, 11 of the table side's levels, 1 of the bits | second piece of 11.1, with the numbers and a_pad of this piece | chi^(its number) |
| n0 + 43 + v_Z | LJ, the join | e_0 (alpha - KF_0) + e_1 (alpha - theta - KF_1) - C_end | chi^(its number) |
| n0 + 44 + v_Z | LP0, the leaves | Left(CT_16; 1; A_L PL_0): CT_16 is CT_(5 + dL) with dL = 11, the form that the last table level passes on. It is the claim that the sumcheck of step 7 of the lookup leaves, with S0 = 1; PL_0 has the coefficient eps_L A_L in it | chi^(its number) |
| n0 + 45 + v_Z | LP1, the planes at the bits' point | hidden(jBit) - PL_1 | chi^(its number) |
| n0 + 46 + v_Z | LP2, the number of keys | (the sum over row < 32 and col < 2^11 of 2^(col mod 32) BZ[row][col]) - 2 tau mz | chi^(its number) |
| n0 + 47 + v_Z + mu, for mu = 0, .., NM - 1 | TIE_mu | below | chi^(NCm + first_mu) |
| n0 + 47 + v_Z + NM + q, for q = 0, .., NX - 1 | READ_q | below | chi^(NCm + first_NM + q tauP) |

*The join has no term for empty trees.* The general join (second piece of 11.1) has a sum over the
tree numbers from NT to 2^rK - 1. Here it is empty.

*The claim of a matrix.* TIE_mu is the sum over i' < mp_mu and u < tau of chi^(i' tauP + u) EP(mu,
i', u). Written on combined rows:

      TIE_mu = (the sum over i < m_mu of cellw_mu(i) (ZZ{pw}[off(mu) + i] + 2^10 ZZ{pw}[mz + off(mu) + i])) - 2^17 spw (the sum over i < m_mu of cellw_mu(i)) - (the sum over c < k_mu of colw_mu(c) M_mu{rw_mu}[c]).

*The claim of a read.* For read number q = (f, b), with mu = mat(f), READ_q is the sum over u < tau
of chi^u ER(f, b, u). Written on combined rows:

      READ_q = (the sum over r < len(f) of phi_f[r] (ZZ{pw}[off(mu) + b len(f) + r] + 2^10 ZZ{pw}[mz + off(mu) + b len(f) + r])) - 2^17 spw (the sum over r < len(f) of phi_f[r]) - (the sum over c < k_mu of colw_mu(c) U3[at(f) + c nb(f) + b]).

*The number that such a claim takes is not its power.* TIE_mu and READ_q each take one number in the
count of the segment's claims, so the claims that the segment adds after the test have higher
numbers. But the power chi^(that number) multiplies nothing: the claim's coefficient is the one in
the table. (The programs multiply the form by chi^(NCm - number + x), where x is the number of the
claim's first equation, and then by chi^number like every claim.)

*A matrix of an earlier segment* has its claim TIE_mu, its columns in Theta and its reads like every
other matrix of the list. Each segment that has the matrix in its list samples entries of its own
for it and has digits of its own for it.

##### The weights that the test puts on the arrays

The segment's one equation holds, for each array, combined rows with weights on their columns
(second piece of 11.1). The table gives what the claims of the smallness test add. Two lines of this
table with the same array and the same row weights can be kept as one combined row, with their
column weights added.

| array | row weights | column | weight that the test adds |
|---|---|---|---|
| `ZZ` | kw_blk, for each block blk | crow, for crow < mz | - chi^nJ e_0 eq(zk_top,blk, row0_blk / R_blk) eq(zk_col, crow) |
| `ZZ` | kw_blk, for each block blk: a combined row of its own in the programs, apart from the one above | mz + crow | - chi^nJ e_1 eq(zk_top,blk, row0_blk / R_blk) eq(zk_col, crow) |
| `ZZ` | pw (the claims READ_q) | off(mu) + i | the sum over the reads q = (f, b) with mat(f) = mu and b = floor(i / len(f)) of chi^(NCm + first_NM + q tauP) phi_f[i mod len(f)] |
| `ZZ` | pw (the claims READ_q) | mz + off(mu) + i | 2^10 times the weight of column off(mu) + i |
| `ZZ` | pw (the claims TIE_mu) | off(mu) + i | chi^(NCm + first_mu) cellw_mu(i) |
| `ZZ` | pw (the claims TIE_mu) | mz + off(mu) + i | 2^10 chi^(NCm + first_mu) cellw_mu(i) |
| `BZ` | bw_0 | col | chi^(nJ + 1) eps_L A_L eq(the last 11 coordinates of zb_0, col) |
| `BZ` | bw_1 | col | - chi^(nJ + 2) eq(the last 11 coordinates of zb_1, col) |
| `BZ` | 1 on every row | col | chi^(nJ + 3) 2^(col mod 32) |
| the array of matrix mu | rw_mu | c | - chi^(NCm + first_mu) colw_mu(c) |
| `U3` | 1 on its one row | at(f) + c nb(f) + b | - chi^(NCm + first_NM + q tauP) colw_mat(f)(c), q the number of the read (f, b); the segment's other claims add to the same columns |

The constants that the test adds to the equation's constant are:

1. from LJ: chi^nJ (e_0 alpha + e_1 (alpha - theta));
2. from LP2: - chi^(nJ + 3) 2 tau mz;
3. from TIE_mu, for each matrix: - chi^(NCm + first_mu) 2^17 spw (the sum over i < m_mu of
   cellw_mu(i));
4. from READ_q, for each read: - chi^(NCm + first_NM + q tauP) 2^17 spw (the sum over r < len(f) of
   phi_f[r]);
5. the constants of LK, C_end, LP0 and hidden(jBit), which follow from the second piece of 11.1.

*In the source's names.* `powersF` makes pw (`vt` for the reads and `vp` for the ties: one vector
under two names), rw_mu (`rowX`), the vector `loX` = (1), and first_mu (`eqFirst`). `foldSums` makes
colw_mu (`gamV` for the reads and `gamX` for the ties: equal). `tieLeft` makes cellw_mu: wl[i] =
Hi[floor(floor(i / 3) / J)] Lo[floor(i / 3) mod J] 2^(20 (i mod 3)), with Hi = `rowX`, Lo = `loX`
and J = 1. `tieRight` makes the column weights of the array of a matrix: wr[c J + x] = gam[c] Lo[x],
so wr[c] = colw_mu(c). `leadF` makes the factor that moves a claim from the power of its number to
the power of its first equation.

*The shapes of the row weights.* The table names each shape as the second piece of 11.1 does. It
also writes the weights out bit by bit, bit 0 first, as the frame takes them (11.1, the order of
bits). A bit's pair (x, y) means: the factor is x if the bit is 0 and y if it is 1.

| row weights | shape (second piece of 11.1) | the pair of bit s |
|---|---|---|
| pw | the powers of chi: a product over the bits | bit s of u has the pair (1, chi^(2^s)), for s < clog2(tau) |
| rw_mu | the powers of chi^tauP: a product over the bits | bit s of i' has the pair (1, chi^(tauP 2^s)), for s < clog2(mp_mu) |
| kw_blk | equality weights on a block of rows, with the point zk_row,blk and the first row row0_blk | the low a_blk bits of u have the pairs (1 - z, z) of the coordinates of zk_row,blk, the last coordinate for bit 0; each higher bit s has the pair (1, 0) if bit s of row0_blk is 0 and (0, 1) if it is 1 |
| bw_0, bw_1 | equality weights on the block of all 32 rows | the five bits have the pairs (1 - z, z) of the first five coordinates of the point, the fifth coordinate for bit 0 |
| the weights of LP2 on `BZ`; the weight on `U3` | ones | (1, 1) for every bit |

For tau = 96, 160 or 192, and for every mp_mu that is not a power of two, the products of pw and of
rw_mu are also defined, and are not 0, at row numbers from the array's number of rows up to the next
power of two. They multiply nothing: a combined row is a sum over the array's own rows (second piece
of 11.1), and the frame puts a term's weights only on the blocks of the array in the item `layout`
(R14.5), each of which is inside the array.

##### Tests on the public numbers

These tests read no proof. The first piece of 11.1 states most of them in words; the table gives
each with the programs' rejection name. A name that ends in a hyphen is followed by the programs'
name of the matrix or of the part.

| a verifier rejects unless | rejection name |
|---|---|
| tau is a multiple of 32, 32 <= tau <= 256, and tau1 is 0 or a power of two | `hid-tau` |
| tau is not below the line `sec` | `policy-tau-hb` |
| tau is the number of combinations that the plan derives: the line `sec` less the proof-of-work bits that the line `work smallness` gives to a request's own rounds (11.1; the source's number has one more term, which is 0: 8.4, item 21). With the test above, that number of bits is 0 and tau equals `sec` | `plan-tau` |
| NM < 256, and at least one matrix of the list is the segment's own and is read by no other segment | `plan-link` |
| m_mu >= 1 | `shape-matrix-` |
| 1 <= k_mu <= 8191 | `hid-k-range` |
| no matrix of the list is a weight matrix | `t2f-weight-class` |
| 2 mz < 2^32 | `t2f-columns-range` |
| a matrix of an earlier segment has that segment's m and k and begins at its first column | `shape-foreign-` |
| a matrix of an earlier segment belongs to a feed-forward segment that comes before this one, is one of the matrices that that segment lists as read by others, and its array has the same numbers of rows and columns in both segments | `plan-fold-source` |
| n_Z - dT <= 7, so NT <= 256 | `forest-trees` |
| v_Z < 32, so n_Z <= 30 | `forest-keys` |
| len(f) >= 1, len(f) divides m_mat(f), and phi_f has len(f) entries | `shape-functional-` |

The other tests of a lookup (11.1) hold here for every tau and mz: dT >= 10, because n_Z >= 10 and
lidb >= 22; 2^(11 + 32) < p; and no call is all padding, because tau mz >= 1.

*A condition of 8.4 that reads these sizes.* One of the ten conditions on hiding that 8.4, item 23,
names and 16.1 defines, the one with the name `range-checks` (16.1, condition 7), is about the
matrices of this test. For each matrix of each segment's list, those of earlier segments among them,
it forms the number floor(14426 x 2^34 / (10000 x 2^14 k_mu)) - 1 - clog2(m_mu) - clog2(tau). It
takes the least of these numbers over the whole request, subtracts clog2 of how many such numbers
there are and three more terms that follow from the public files, and compares the result with the
line `sec`. If the result is below `sec`, `cp_verify` ends with `parameters-not-admitted`. 16.1,
condition 7, gives the condition whole, with the three further terms.

*What the parts must cover.* The parts of `U3` fill the vector from entry 0 to its last entry,
without a gap and without an overlap, and every entry of `U3` belongs to exactly one read. Every
matrix of the list has at least one part, so every cell row is in at least one read. These hold by
the lists of 11.2 to 11.4. `cp_verify` tests them in every run, under names that begin with
`internal-` (section 14).

*The scalars.* The test sends no scalar but those of its lookup (11.1). No other claim of the
segment names a scalar of that lookup.

*When the list of parts is fixed.* The weight vectors phi_f are functions of the public files, the
request, and what is sent and sampled before the round `RNG`. The list of parts, with all phi_f,
does not change after that round.

##### Sampled values at which a verifier gives no acceptance

After all claims are added, a verifier tests that no weight which should bind a committed entry
is 0. The value of a sampled element can make such a weight 0. At each of the events below a
verifier gives no acceptance: 13.9 says so for every verifier, says what `cp_verify` does under the
names in brackets, and gives the chance of each event. The events are:

1. colw_mu(c) = 0 for some column c of some matrix (`internal-f-table-read`).
2. A column weight is 0 in the segment's one equation, all claims of the segment counted
   (`internal-t2f-untyped-`, followed by the programs' name of the array). The columns are:
    - `U3`: any entry.
    - `ZZ` under kw_blk: a column crow for call 0, mz + crow for call 1.
    - `ZZ` under pw: any of the 2 mz columns, separately for the claims READ_q and TIE_mu.
    - `BZ`: any of the 2^11 columns under bw_0, under bw_1 or under the weights 1.
    - The array of a matrix: any column.
3. chi^nJ e_0 or chi^nJ e_1 is 0, or eq(zk, u 2^lc + crow) is 0 for some u < tau and some crow < mz
   (`internal-t2f-untiled` or `internal-t2f-untyped-`: the programs make the test of event 2 first,
   so where a column weight of `ZZ` is 0 with it, the call ends under that test's name; 13.9, events
   4 and 6). The programs look at the first row of each block with every column, and at every row
   with the first column. That is enough, because the weight is a product of a factor of the row and
   a factor of the column.
4. Two of the row weights rw_mu(i') of one matrix are equal, or one is 0 (`internal-sym-rowvec`); a
   weight chi^(NCm + first_mu) cellw_mu(i) is 0, which is so at chi = 0 alone
   (`internal-f-tie-weights`, or the name of a test that the programs make before it: 13.9, event
   4).

The weights of the claims READ_q and those of the claims TIE_mu under pw are tested separately, as
the programs keep them separate.

##### The purpose that the source states

The source states the purpose of this test at the head of `blocks/iop/rng.h`, where the test is
given in a general form. It is: for each matrix of the list there is one matrix of integers that has
the committed packing M_mu and the committed entries of `U3`. Each of these integers is above -2^18
and below 2^18. The bound is twice the offset 2^17. The test does not state that a cell is a byte.
This page makes no statement of its own on what the test shows. Comments further down in that header
state three limits of it. First, on what the round's error counts (the comment before
`rng_theta()`): 2^-N for N combinations is the chance for one column, the first that no column of
small integers explains, which is fixed before Theta is sampled (beyond unique decoding, the comment
says, that rests on the answered out-of-domain samples, whose round has its own error). A union
bound over all columns, N - log2(columns) bits, is printed beside it; the comment calls it the same
event counted more coarsely, no cover for a prover that is not bound to one object, and no knowledge
error, and says that no verdict depends on that line. Second, in the same comment: the count is that
of this protocol's technical report, which states it as adapted from Lemma 2.5 of the first work
that the comment cites, ePrint 2020/1448, and argues it anew, for cells that are sent packed and
through reads, with the combinations written in digits that a lookup checks; by one comparison its
argument has the shape of the proof of that work's Lemma A.1; and the argument for this error has
not been compared with Lemma 3.4 of ePrint 2021/1397, which the comment names for the distribution
of Theta's entries. By row SO-17.6.1 of `docs/limits-all-rows.tsv` one unchecked reading of that
paper finds the step, at least in part, in its section 3.4, so how much of the step is this
project's own is not known. Third, the comment before `rng_lookup()` reports a false acceptance:
with a second description of where the digits are in a caller's hands, a cell of 1,000 has been
accepted under the bound 32. It is the control of a test of that block that is not part of this
repository, which injects the second description on both sides. `cp_verify` compiles nothing of that
header: the sub-proofs run a second implementation of the test, of which the library's tests say
nothing (17.5, item 25). What the plan counts for the round `RNG` is in 13.9.

*The relations among the constants.* `cp_verify` and `cp_commit` do not build unless these hold (the assertions are in the verifier's source, of which `cp_prove` compiles nothing); the source names them (f1) and (f2).

1. (f1) 2^20 >= 4 x 2^17 - 1, and (4 x 2^17 - 2)(1 + 2^20 + 2^40) < p. The source gives as the
   reason: a packed entry then has at most one unpacking into small integers.
2. (f1) A packed row has exactly three cell rows. With four, the second relation of 1 would read (4
   x 2^17 - 2)(1 + 2^20 + 2^40 + 2^60) < p, which is false in both fields. The source's reason (the
   message of an assertion in `verifier/src/vf_smallness_packing.h`): with four, (4 x 2^17 - 2) 2^60 is above p.
   By the comment above the assertion, that product is above 2^78.
3. (f2) (8191 + 1) 2^14 2^7 <= 2^34. The source gives as the reason: it ties the largest number of
   columns to the offset, so that an honest prover's combination leaves its window only with a small
   chance.
4. Both digits have at least 5 bits, the number of bits of a plane's number, and 2^(10 + 8) = 2 x
   2^17.

*The honest prover.* In the programs a cell is a signed byte. Even so, a combination of such cells
can fall outside -2^17 .. 2^17 - 1. Then it has no digits, and there is no honest proof with these
sampled entries. This failure depends on the cells and on the sampled matrix, and the source counts
its chance in the error of hiding (`blocks/iop/rng.h`, in the comment before `rng_prove()`; 16.1,
condition 7).

##### Easy to get wrong

1. `BZ` is placed by the top five bits of X = (t, c, j), not by the rule of the matrices b, bA and
   bH.
2. The high digits begin at column mz of `ZZ`, not at a power of two. The key numbers do use a power
   of two: key number u 2^lc + crow.
3. The offset 2^17 is subtracted once for each cell row that exists. In the last packed row of a
   matrix whose m_mu is not a multiple of 3, one or two terms are absent with their offsets.
4. One entry Theta_mu[c][u] serves the packed rows and all reads of the column. Nothing is sampled
   separately for the reads.
5. The step from one packed row or read to the next is tauP, not tau, and a matrix takes tauP
   2^clog2(mp_mu) numbers, not tauP mp_mu.
6. The exponent of an equation's power starts at NCm, the number of all claims of the segment, also
   of those that the segment adds after the test (11.2, 11.4).
7. The claims TIE_mu and READ_q take numbers whose powers of chi multiply nothing.
8. The reads are ordered by matrix first, then by part. The entries of `U3` are ordered by part
   first.
9. In `U3` the column's index is the slow one: entry at(f) + c nb(f) + b.
