*Part of [the file format](FORMAT.md). Rule numbers and terms are those of that page.*

## 9. A key file and a leaf file, round by round

This section gives, for a key file and a leaf file, the first query of each chain and then every
round: its name, its kind, the prover's message and what the verifier samples. Section 7 (R13.1 to
R13.3) says where the bytes of the two files are: heads, roots, records, the order of the chains.
The section needs sections 2 to 8. It begins with the types whose canonical bytes (R4.2) the first
queries hold; section 10 uses the same types, and 9.9 has those that only section 10 uses.

A **key file** comes with a commitment. It holds the root of the **anchor**, the commitment's first
stored table, and the committer's proof that the table is well formed. A **leaf file** comes with a
**leaf table**, a further stored table with the same rows under fresh randomness, and is meant to
show that the leaf table's rows are the anchor's. A key file has three chains: the **one-time
chain**, the **shared chain** and the **second chain**. The shared chain and the second chain are
those of a proof (section 10), under labels of their own (R13.4), for a batch of one request, in
which the committer takes the request's part; so their items keep the names that speak of a request.
A leaf file has one chain, the **link's chain**; the **link** is the tie between the anchor and a
leaf table.

What the acceptance of a key file is meant to say, whether or not its proof of work had to be
counted (section 14). By the programs' source it says the three things that follow (the source
counts the second and the third as one: `blocks/frame/check_key.h`, at its head); this page proves
none of them (section 15). The committed word is close to one code word. Every entry of that word's
matrix is in the range which the public parameters name for single entries. Every further lookup
which the public parameters name holds on the rows which they name. No verifier judges whether these
are the right ranges and lookups for a model. No line of a public file names a range's size. Which
rows each lookup reaches follows from counts and words of the public files, through a fixed routine
of the programs. The size of a lookup's table, its offset, its digits' weights, its padding key and
its first bounds are constants of that routine (R13.14 lists them). All of it is bound, as the item
`ranges` among the parameters of the first query of the key file's one-time chain. But the
configuration's digest is a digest of the public files' bytes, and the commitment ID (R4.11) is a
hash of that digest, of the anchor's name and of its record's seeds and stored answers: neither
holds that routine. So anyone who relies on a commitment ID has to judge the public files and the
programs' routine, or the bytes of the key file's first query.

What this section defines, and what it does not. It defines the first queries of the four chains,
the rounds of the leaf file, and, for the rounds of the key file's three chains, the names, the
length of each message and the elements that are sampled. It does not define what the steps of the
key file's chains prove, what each element of their messages is, what the words of their items mean
beyond what R13.16 to R13.19 take from them, or the checks on the items. Section 13 lists what a
verifier does. The mark "(from the programs' source)" says that a passage is needed by a second
verifier and is taken from the source of the programs in this repository. After the last full stop
of a paragraph or a list item it holds for the whole of it; inside a sentence it holds for that
sentence alone.

Terms from earlier sections: word, element, digest, list (R1.1 to R1.7); chain, parameters,
statement, item (R4.1); plan and entry (R4.3); random bytes and run's value (R4.5); stored table and
record (R4.6, R4.8); configuration (R4.9); a table's name (R4.10); round and kind (R5.1, R5.2); seal
(R6.8); the draws and the list (R6.6); column (R7.1); short position (R7.2); leaf class (R7.12); a
tree's place and number (R7.13); binding tree (R7.18). From a later part: the **matrix** M, whose
entries are the digits of all committed weights (11.5). It has 2^k_col columns (R13.13). 11.5 writes
W for the matrix, m0 for the number of its rows and lk for k_col; in this section W is an oracle
message.

In this section s is the number of answers to an out-of-domain sample, the plan's sixth word (R4.3).
(R7.3 uses the same letter for the number of bytes of a salt.) F is the base field and E the
extension field (R1.2, R1.4). eq(p; z) is the equality polynomial: the product, over all coordinates
z_t of z, t from 0, of z_t if bit t of the number p is 1 and of 1 - z_t if it is 0; bit 0 is the
lowest.

An **oracle message** is a set of rows that the prover encodes and sends in a tree (R7.1): **main
rows** over F or E under a base code, and **mask rows** over E under a mask code. An oracle message
**gets an out-of-domain sample** if its notice (R12.3) has a number of answers above 0: after its
root the verifier samples the seeds, one for each answer (R6.9, R12.9), and the prover sends the
answers. A **group** is a set of oracle messages under one base code and one mask code that one
proximity test (R13.19) checks together. A **switch** is the step of a proximity test that sends the
next stage's oracle message: the rounds `SEND seed` and `CS`.

The oracle messages of a key file and a leaf file, by the names that the rules use. The **index-mask
message** has mask rows alone. The **multiplicities** say how often each row of a lookup's table
occurs among the lookup's keys; they are sent in bits, and **plane** j is bit j of all of them. W is
g^T M for the row weights g that the lookups leave: one entry for each column of the matrix M. The
**folded row** of a stored table is the combination of its 2^v rows (v of R4.10) with the weights
eq(r; gamma) for a sampled point gamma, encoded again as an oracle message (section 13 writes G_u
for this oracle message). Omega has one entry for each row of the matrix: that row combined with the
equality weights of the shared point (`LIN point`). The **shared split's mask message** has mask
rows alone; it is sent where the folded row and Omega are under different base codes, and belongs to
both their groups. A **request's own group** holds the index-mask message and W, and in the second
chain Omega too (8.4, item 18; 13.9 calls it the own group); the test on a request's own rows is the
proximity test on this group. (from the programs' source)

The **pad** of an oracle message's out-of-domain sample is in the message's last mask row or, if it
has no mask rows, in its last main row: the last s entries of that row's message, one for each
answer in order; in a main row over F the last 3 s entries, three for each answer. (from the
programs' source)

Where a vector is in an oracle message's main rows. The folded row, Omega, W and a leaf file's two
folded rows each hold one vector of elements of E. Let l be the number of entries of a message of
the oracle message's base code, which is a power of two, and R the number of its main rows. For each
of these oracle messages an item has a word that is 1 if the rows' index is in the high bits (R13.9,
R13.13). Where the word is 1, entry j of the vector is entry j mod l of main row j div l: a row
holds l consecutive entries of the vector, and the vector takes only the rows that it needs. Where
the word is 0, R is a power of two, and entry j of the vector is entry j div R of main row j mod R.
In both cases the vector begins at its entry 0 and ends before the first entry of the pad, whatever
s is. A verifier rejects a shape in which it does not, as the programs do (under
`parameters-not-admitted`: 8.4, item 23, step 3). It also rejects a shape in which the row that
holds the pad has fewer entries than the pad; there the programs have a guard and no rejection
(section 18, the row of a pad; 17.2, item 2). An honest prover sets the entries that the vector
leaves free, the pad apart, to 0. (from the programs' source)

What the vectors hold. Omega's vector has one entry for each row of the matrix, by the row's number;
W's has one for each of the 2^k_col columns, by the column's number. The vector of a folded row has
two parts. Write (a, l_T, r_T) for the tables' code (R4.10). The fold of the rows' messages, l_T
entries, a record's pads included, is at the entries 0 to l_T - 1. The fold of the rows' random
tails, r_T entries, begins at l_T rounded up to a multiple of the least power of two that is at
least r_T; the entries between the two parts are 0. (from the programs' source)

The **free entries** of an oracle message's mask rows are the entries from which later steps take
their masks. Let the message have m mask rows of e entries each, and let 2^q be the least power of
two that is at least m. Entry c of row r has the number r + 2^q c (`toys/1-product.md` writes 2^b
for this 2^q, and calls c a place and the number an index); a number whose low q bits are m or more
names no entry and is passed over. The free entries are those with a number below that of the pad's
first entry, which is entry e - s of row m - 1: they are m (e - s) + (m - 1). They are taken by
increasing number, each once: entry 0 of row 0, entry 0 of row 1, and so on to row m - 1, then entry
1 of row 0. "The next free entry" (13.9) is the next in this order. (from the programs' source)

### 9.1 Three things that are easy to get wrong

1. Which coefficient a `PSC round` leaves out, and in which order the variables are bound, differs
   between the sumchecks of one chain. In every sumcheck of a lookup (R13.17) the constant
   coefficient is left out, and the variables are bound from the highest down; in the last level of
   the product tree of each of the lookup's parts the variables of the lookup's first bounds come
   first. In the sumcheck after `BAT-sigma` (R13.18) the linear coefficient is left out, and the
   variables are bound from the lowest up. In a `FOLD round` the linear coefficient is left out.
   R13.16 has the rule. (from the programs' source)

2. The shared chain of a key file has no random bytes in the file. The binding tree's root is the
   leaf hash of the one-time chain's seal at slot 0, with the salt from the file. It is used twice:
   as the second item of the shared chain's statement, and as the 32 bytes that are used instead of
   that chain's random bytes (R13.6). The second chain continues the one-time chain's list of trees
   and adopts one tree of the shared chain, which keeps in its leaves the number that it had there
   and the shared chain's run's value (R13.7). The table in 9.7 lists all trees of a key file by
   chain and place.

3. A sumcheck with a hiding sum has no check that can fail. The verifier uses the round's equation
   to fill in the coefficient that was not sent (R13.16). A false message shows only later, in a
   **linear row**: a linear equation with a public target on the entries of committed oracle
   messages (13.9), which a proximity test checks. These rows name entries of mask rows, and which
   entry a row names follows from the order of the protocol alone. In each oracle message whose
   out-of-domain sample has been answered, they are its free entries, used once each in the order
   given at the start of this section. A sumcheck takes as many entries as it sends elements. It
   takes them when it begins: first those of the hiding polynomial, variable by variable in the
   order of the rounds and by rising degree, from degree 1 for every variable but the last and from
   degree 0 for the last (Y, if there is a round in Y); then one for each end value. All entries of
   a lookup come from the oracle message of the multiplicities, with one exception: the two entries
   for the end values of the last level of the product tree of each of the lookup's parts come from
   the index-mask message. The sumcheck after `BAT-sigma` takes its entries from the index-mask
   message too. (from the programs' source)

### 9.2 Types

- **R12.1** A **code** is three words: a, where its length is 2^a; the number of entries of a
  message; the number of entries of the random tail.

  The programs, which can be built for field 2 alone (R1.2), reject three words as a code unless a
  is from 1 to 60, the message has at least one entry, and message and tail together have at most
  2^a and at most 2^32 entries. (from the programs' source)

- **R12.2** The **codes** of a protocol are the list of its base codes and then the list of its mask
  codes (R1.7). A code is named by its number in its list, from 0. The word 2^32 - 1 names no code.
  A code enters its list when the first stage that uses it is entered, stage 1 first. A code that is
  in the list already is not entered again: equal triples are in the list once.

- **R12.3** An oracle message's **notice** is seven words: the number of its base code; the number
  of its mask code; the number of its main rows; 1 if the main rows are over F, 3 if they are over
  E; the number of its mask rows, which are always over E; the number of answers to its
  out-of-domain sample; its leaf class. One shape has one encoding. Where a message has no mask
  rows, the word of its mask code is 2^32 - 1. Where it has no main rows, the word of its base code
  is 2^32 - 1 and the fourth word is 1. R7.12 says what the leaf class does.

- **R12.4** A **stage** of a proximity test is four words: the number of its base code; the number
  of its mask code; k, where it has 2^k main rows; h, where it has 2^h mask rows. How many entries
  fill a stage's mask rows up follows from these and the codes, and is not a word.

- **R12.5** The **choices** of a proximity test are one word: whether the masks of the stages from
  the second on are sent with the switch that makes the stage. If the word is 1, this page says that
  masks ride. The number of answers to an out-of-domain sample is not one of the choices: it is the
  sixth of the plan's eight words (R4.3). A chain of segments holds it a second time (R12.14, item
  4; 17.2, item 9).

- **R12.11** A code's points and its symbols, in field 2. A code (a, l, r) of R12.1 with a <= 32
  lives on the coset 7 H, where H is the subgroup of order 2^a of the units of F. The shift is 7 at
  every length. Let w = 7^(5 (2^32 - 1)), an element of order 2^32 (as a word: 0xE2AF70635744FD26).
  The **point of position x** is z_x = 7 times the product, over the bits i of x that are 1, of
  w^(2^(31 - i)), the root of unity of order 2^(i + 1). So position 0 has the point 7, position 1
  the point -7, and the order of the points is the bit-reversed one. The **symbol at x** of a row
  with the message m_0, .., m_(l-1) and the random tail t_0, .., t_(r-1) is the value at z_x of the
  polynomial whose lowest r coefficients are the tail and whose next l are the message: the sum of
  t_j z_x^j over j < r, and of m_j z_x^(r + j) over j < l. A row over E is encoded coordinate by
  coordinate. A mask code is a code like any other, with its own a. No numbered rule covers lengths
  above 2^32, and this page does not define the points of field 1. No line of the parameter file
  admits a length above 2^32 (8.1, tests 22, 29 and 36; by tests 24 and 30 a mask code is not longer
  than its base code). The programs' source has points for longer codes too, on several cosets of
  the subgroup of order 2^32 (`blocks/code/cosets.h`). By a comment in `blocks/frame/parameters.h`,
  key files, leaf files and proofs for a longer code of the stored tables are untested.

### 9.3 Round names and kinds

- **R12.8** An entry's text (R4.3) is the round's name alone. The names are ASCII. These names are
  common to the protocols of the format; each protocol uses some of them. Each has one kind (R5.2)
  wherever it is used:

| name | kind | | name | kind |
|---|---|---|---|---|
| `SEND seed` | 3 | | `BAT` | 1 |
| `SEND answer` | 7 | | `FOLD epsilon` | 2 |
| `ZERO` | 1 | | `FOLD round` | 1 |
| `PSC epsilon` | 2 | | `CS` | 4 |
| `PSC round` | 1 | | `BASE lambda` | 1 |
| `PSC Y` | 1, its one element not 0 | | `BASE` | 4 |
| `PSC values` | 7 | | `LOC` | 4, and no element after the positions |

- **R13.11** Names and kinds of the rounds that the two files add to those of R12.8. Kind 1: `LKP`,
  `JOIN`, `LINE`, `BAT-sigma`, `LIN point`, `LIN zeta`, `LINK point`, `LINK phi`. Kind 4: `LINK`;
  and `LOC`, which in the shared chain has an empty prover's message, samples the positions at which
  the stored table is opened, and reads no tree of the chain's own. Kind 6: `SEAL`. Kind 7: `SPLIT`.
  A prover's message can hold a root beside other items: the roots come first, in the order of R8.6.
  No program sorts a message: its items are in the order in which they were sent (11.1), and the
  roots come first because in every such message of the three files they are sent first (from the
  programs' source).

A round has a nonce in the file exactly if its entry in the plan has proof-of-work bits above 0
(R5.3, R8.2). All four chains of this section are planned with the committer's proof-of-work bits,
the third number of each line of the parameter file that sets proof of work (section 8). Which
rounds these are depends on the parameter file alone. Of its five lines on proof of work,
`work positions` governs `LOC`, `LINK`, `CS` and `BASE`, and `work closeness` governs `FOLD round`,
`BASE lambda`, `LIN point` and `LINK point`. 8.4, item
3, has the rounds of all five lines. In the parameter file `record.f2.txt` only the line
`work positions` has numbers above 0. (from the programs' source)

### 9.4 Two patterns of rounds: a sumcheck with a hiding sum, and a proximity test

- **R12.6** The example protocol "product" (the toy run `product`: R4.9) is the smallest protocol
  that has both patterns, and R12.9 uses it to show them. The prover knows three vectors a, b, c
  over F with a_i b_i = c_i for all i; the protocol is meant to show that and to hide the vectors.
  No key file, leaf file or proof file holds a chain of this protocol. Its chain has the label 1 and
  random bytes by case (a) of R4.5. After the version's item and the plan (R4.3) its parameters are
  five items, in this order and under these names: `configuration` (R4.9); `codes` (R12.2);
  `the oracle message a, b, c` (a notice: R12.3); `the stages of the proximity test` (a list of
  stages: R12.4); `the choices of the proximity test` (R12.5). Its statement is empty. One oracle
  message is in no item: the fresh mask of the first fold, which is sent with a, b, c; its shape
  follows from the words of stage 1 (R12.10).

- **R12.9** The rounds of "product", and how each message is cut into items (R8.8). L is the
  logarithm of the vectors' length. The proximity test has stages 1, 2, .., S; stage i has the
  number k_i of R12.4. Every item but a root is an element of E.

| rounds | name | the prover's message | the verifier's message | after it in the file |
|---|---|---|---|---|
| 1 | `SEND seed` | the root of tree 0 | s seeds | |
| 1 | `SEND answer` | the s answers to the out-of-domain sample | nothing | |
| 1 | `ZERO` | nothing | L elements | |
| 1 | `PSC epsilon` | the sum of the hiding polynomial | one element | |
| L | `PSC round` | two coefficients of the round's polynomial, the linear and the quadratic one | one element | |
| 1 | `PSC Y` | the same two, for the further variable | one element that is not 0 | |
| 1 | `PSC values` | the two end values | nothing | |
| 1 | `BAT` | nothing | one element | |
| for each stage i < S: 1 | `FOLD epsilon` | the sum of the hiding polynomial | one element | |
| k_i | `FOLD round` | two coefficients, the constant and the quadratic one | one element | |
| 1 | `SEND seed` | the root of tree i | s seeds | |
| 1 | `CS` | the s answers to the out-of-domain sample | the positions, then one element | the nonce; the opening of tree i - 1 |
| at the end: 1 | `BASE lambda` | the target of the masking word | one element | |
| 1 | `BASE` | the masked main message; the masked mask message; the masked main tail; the masked mask tail | the positions, and no element | the nonce; the opening of tree S - 1 |

  The numbers of coordinates, positions, rows and entries are the plan's and the codes'. The
  **seeds** of an out-of-domain sample are s, one for each of the s answers (R6.9): each is the
  coordinates of a point, each by R6.4, and then one element that is not 0, by R6.5. This section
  does not define the checks that the verifier makes on these items.

  The four parts of the message of `BASE` have l, l', r and r' elements of E, where (.., l, r) is
  the last stage's base code and (.., l', r') its mask code. (from the programs' source)

- **R12.10** What a stage's tree holds beside the stage's rows. Two kinds of oracle message are in
  no item of the first query. Their shapes follow from a stage's words (R12.4), and neither gets an
  out-of-domain sample (0 answers). The **fresh mask of the fold at a stage** has no main rows and
  2^h mask rows over E under the stage's mask code. It is sent in the prover's message that holds
  its stage's rows, unless R13.19 gives it a tree of its own. At stage 1 that is the message with
  the oracle messages that are tested (in "product": a, b, c). At a later stage it is the switch
  that makes the stage, if the word of R12.5 is 1. The **masking word of the last step** has one
  main row over E under the last stage's base code and one mask row under its mask code, and is sent
  with the switch that makes the last stage, if that word is 1 (where it is 0: R13.19). Where all
  have one leaf class, each joins its stage's tree by R7.12. By R7.1 the columns then have these
  widths, in elements of F. Tree 0 of "product": 3 for a, b, c, 3 for their mask row, 3 for the
  fresh mask: 9. The tree of a stage i with 1 < i < S: 3 x 2^k for its main rows, 3 x 2^h for its
  mask rows, 3 x 2^h for the fresh mask. The last stage has k = h = 0. The last tree: 3 for its main
  row and 3 for the masking word's, then 3 for its mask row and 3 for the masking word's: 12.

- **R13.16** A sumcheck with a hiding sum, in general. (Where R13.16 to R13.19 name no message for a
  round, the prover's message is empty.) R12.9 has such a sumcheck for one shape. In general it is
  given by a list of numbers g, one for each variable that it binds; by whether it has a round in Y;
  and, if it has one, by a degree D in Y and a number of end values. The round's polynomial of a
  variable has degree below g, so g coefficients, of which g - 1 are sent, since the check fixes one
  (R4.3). The polynomial in Y has degree D, so D + 1 coefficients, of which D are sent. Its rounds
  are: `PSC epsilon`, whose message is one element, the hiding sum; for each g of the list one
  `PSC round`, whose message is g - 1 elements; and, only if it has a round in Y, two more: `PSC Y`,
  whose message is D elements, and in which the one element that is sampled must not be 0, and
  `PSC values`, whose message is the end values. The verifier samples one element in every one of
  these rounds but `PSC values`, where it samples nothing. So a sumcheck without a round in Y ends
  with a `PSC round`, or, if it binds no variable, with `PSC epsilon`, and has no end values. The
  programs do not run that last shape: at a sumcheck that binds no variable and has no round in Y a
  guard of the library ends the call, on the prover's side and on the verifier's (17.2, item 1; from
  the programs' source). A **product tree** of L levels with u further variables is, for l = 0 to
  L - 1: a sumcheck with l + u variables, g = 3 for each, D = 2 and two end values; and then one
  round `LINE`, one element sampled.

  Which coefficient is left out. The coefficients that are sent come lowest degree first. Which one
  is left out depends on how the variable is summed. If the variable is summed over {0, 1} under the
  factor of an equality polynomial at a point z, the round's polynomial q counts as (1 - z) q(0) + z
  q(1). Then the constant coefficient is left out, and the verifier sets q_0 = claim - z (q_1 +
  q_2 + ..). The variable Y counts as q(0), which is the same rule with z = 0: the constant is left
  out and is set to the claim. If the variable is a plain sum, q(0) + q(1), the linear coefficient
  is left out, and the verifier sets q_1 = claim - 2 q_0 - q_2 - .. . The first claim is the hiding
  sum plus the element of `PSC epsilon` times the target of the statement that is proved; each later
  claim is the value of the round's polynomial before it at that round's element. So none of these
  equations can fail. (from the programs' source)

  After `LINE`, the sampled element becomes the last coordinate of the point of the next level's
  claim. (from the programs' source)

### 9.5 A stored table: pads and record

- **R13.8** Where a stored table's pads are, and what an answer of its record is. No word says where the pads are. Their positions follow, by a fixed rule of the
  format, from two words of the shape: the number l of entries of a row's message and the number s
  of stored answers for each row. The last 3 s entries of a row's message are its **pads**; the l -
  3 s entries before them are **plain**. The pad of answer i and coordinate t, with i < s and t < 3,
  is at the position l - 3 s + 3 i + t. The pads are counted in l; the entries of a row's random
  tail are not. A verifier rejects a shape with 3 s >= l. With (z_i, beta_i) the i-th seed of R4.8
  and m the row's message, the row's i-th answer is the sum over the positions p of the plain
  entries of eq(p; z_i) m[p], plus beta_i times the element of E whose three coordinates are that
  answer's three pads. So an answer reads its own pads and no other answer's, and the point's
  weights are on the plain entries alone. Every claim that a later proof makes on the stored table
  reaches the plain entries alone. What a record is for, by the comments of the programs' source,
  which this page repeats and does not prove: the distances go beyond half the minimum distance, and a committed word
  can be close to several tables. There a root alone does not name the weights: the table is the
  close table that agrees with the record's stored answers, which is why these are in the statement
  of every chain that speaks of the table (R4.6) and in the commitment ID (R4.11) (from the
  programs' source: `blocks/iop/foldeq.h`).

  This rule and R13.15 write s for the shape's fifth word, the number of stored answers for each
  row. The programs set it to the plan's sixth word, so the
  two numbers that are written s are equal. The programs also reject a shape whose three code words
  are not a code (R12.1), a shape with v above 40, and a shape whose salt has a length that R7.3
  does not admit. (from the programs' source)

  The stored answers are in the file after the stored table's root, 2^v s elements of
  E, by row and within a row by answer, with no word for their number (R13.2, R13.3). A verifier
  rejects an answer that is not canonical (R1.4). The answers enter the statement of every chain
  that speaks of the stored table; a second chain has them only through the two seals of its
  statement (R4.6). A verifier computes the seeds itself (R4.8). Beyond hashing the answers, it
  first uses them in the shared chain and in the link's chain. There, for each i < s, the i-th
  answers of all rows are combined with the weights eq(r; gamma) of the fold point gamma. The sum is
  the target of one linear row on the oracle message that holds the folded row. (from the programs'
  source)

### 9.6 The four first queries

- **R13.4** The labels of the four chains. (A chain's label is defined at the start of section 4; it
  is not a hash's label.) The one-time chain has the label 0x46524d02, the shared chain 0x4c494e01,
  the second chain 0x4c494e02, the link's chain 0x4c4e4b01. None of them is a label of a proof
  file's chains (R14.3, R14.9, R14.10).

- **R13.5** The four first queries. After the version's item and the plan (R4.3) come these
  parameter items, in this order, and the statement has these items, in this order. An item that
  holds a stored table is R4.6's: the root as a string of bytes, then the list of the record's
  stored answers.

| chain | parameter items | statement's items |
|---|---|---|
| one-time | `configuration`; `index-mask message`; `ranges`; `binding tree`; `codes`; `the test on the table` | `table` |
| shared | `configuration`; `codes`; `the test on the table` | `table`; `the binding tree's root` |
| second | `configuration` | `the seal of an earlier chain`, twice (R4.7): the one-time chain's seal, then the shared chain's |
| the link's | `configuration`; `codes`; `the link` | `the first table`; `the further table` |

  `codes` is R12.2's. The tables' own code is in `codes` in none of the four chains: it is the code
  of no stage. It is encoded by its own three words at the beginning of the tables' shape. That
  shape is the beginning of `the link` and of `the test on the table`. `index-mask message` is a
  notice (R12.3); `binding tree` is two words, the depth of the tree and the number of bytes of a
  leaf's salt; `the link` is R13.9's. `the test on the table` is R13.13's and `ranges` R13.14's. In
  the link's chain the item `the first table` holds the anchor and the item `the further table` the
  leaf table. The anchor's item holds the root and the stored answers that the verifier has kept
  from the key file that it checked, not bytes of the leaf file.

  The key file's three chains use one list of codes, and the values of `codes`,
  `the test on the table` and `table` are the same, byte for byte, in the one-time chain and in the
  shared chain: a verifier that takes the chains' first queries as inputs compares the three values
  and rejects if one differs, as in a proof (R14.9). A leaf file has a list of codes of its own. The
  tests enter their stages in this order: the test on the folded row; the test on Omega, if it is
  made; the test on a request's own rows; the test of `ranges`. In a key file the depth of the
  binding tree is 0. (from the programs' source)

  Digests. The values of `configuration`, `the binding tree's root` and
  `the seal of an earlier chain` are digests, encoded bare (R1.5): 32 bytes with no length of their
  own, after the length that every item has before its value (R4.1). A root in a prover's message is
  32 bare bytes too. The one exception is a stored table's root inside a table's item, which is a
  string of bytes: the word 32 and then the 32 bytes (R4.6).

  Random bytes (R4.5). The one-time chain, the second chain and the link's chain have case (a):
  their first 32 bytes in the file. The shared chain has case (b) (R13.6).

- **R13.15** The entry `RECORD` of a plan (R4.3). The plan of the
  chain that follows a stored table's root in a file holds one entry under the name `RECORD`, as its
  first entry. These plans are two: the plan of a key file's one-time chain, for the anchor, and the
  plan of a link's chain, for the leaf table alone. The plans of a key file's shared and second
  chains hold none, and a link's plan holds none for the anchor, whose out-of-domain sample is
  counted in the key file. The entry's eleven words, in R4.3's order, are: 1; 0 proof-of-work bits;
  0 bytes of a message, 0 roots and 0 answers to an out-of-domain sample among them; s (c + 1)
  elements of E that are sampled, of which s must not be 0, where c is the least number with 2^c not
  below the number l of entries of a row's message; and 0 for the positions, for k, for the matrix
  entries and for the rows. It is not a round: no query of the chain belongs to it, the chain's
  count of rounds passes over it, and the sampling itself is done in R4.8's own chain. Why it is in
  the plan: so that the error of the stored table's out-of-domain sample is counted with the file's.
  It is hashed with the plan, so it changes the parameters' digest. The stored answers' bytes in the
  file (R13.2, R13.3) are not counted in any of its words.

- **R13.9** The item `the link` is these words:
  1. the tables' shape, six words in R4.10's order (a; the number of entries of a row's message; of
     its tail; v; the number of stored answers for each row; the number of bytes of a leaf's salt);
  2. nu_0;
  3. nu_j (R7.19);
  4. the number of the group's code in its list;
  5. the number of its mask code in its list;
  6. R, the number of main rows of an oracle message that holds a folded row;
  7. a word that is 1 if the rows' index is in the high bits;
  8. the leaf class;
  9. a word that is 1 if the fold's mask is sent ahead;
  10. the later stages of the proximity test, a list by R12.4;
  11. the choices' word (R12.5).

  Items 1 to 9 are 14 words. The anchor and all leaf tables have one shape.

- **R13.13** The item `the test on the table`. An oracle message's **shape** is a notice (R12.3)
  without its last word: six words. The item is, in this order:
  1. the tables' shape: six words, in R4.10's order;
  2. the shape of the oracle message that holds the folded row;
  3. a word that is 1 if that message's rows' index is in the high bits;
  4. its leaf class;
  5. the word on the stored table's columns: the number of positions that the round `LOC` samples on
     them (with repeats: R6.6);
  6. k_lo;
  7. the number of rows of the matrix;
  8. the shape of Omega;
  9. the shape of W;
  10. two words on the high bits: Omega's, then W's;
  11. the shape of the shared split's mask message;
  12. the shape of a request's split's mask message;
  13. four leaf classes: of Omega, of the shared split's message, of W, of a request's split's
      message;
  14. the mask code of the shared group;
  15. the mask code of a request's own group;
  16. a word that is 1 if a fold's mask is sent ahead;
  17. four proximity tests, each a list of stages by R12.4: on the folded row, on Omega, on a
      request's own rows, on a request's own rows with Omega;
  18. the choices' word (R12.5).

  Items 1 to 16 are 50 words. The shared group of item 14 stands for the groups that the shared
  chain tests, the folded row's and Omega's, which have one mask code (from the programs' source).
  8.4, item 18, says how the shapes follow from the parameter file. R13.19 gives the values that the
  programs write for the leaf classes and for the words of items 3, 10, 16 and 18. A message that
  does not exist has the word 2^32 - 1 for both codes and no rows: its six words are 2^32 - 1,
  2^32 - 1, 0, 1, 0, 1, the last of them 1 whatever the number of answers to an out-of-domain sample
  is. A test that is not made is the empty list. One shape is encoded although no verifier reads it
  there. Where the folded row's and Omega's base codes are one, no split is made and the shared
  split's mask message is not sent (R13.19). The programs encode it there too, with its mask code
  and one mask row: 2^32 - 1, the mask code, 0, 1, 1, s. Its sixth word is the number of answers s, and not the constant 1 of a message that does not exist (the parsers admit `answers 1` alone, 8.1, so in every file both words are 1). This page does not define another
  encoding for that case. The matrix has 2^(k_lo + v) columns; this section writes k_col for k_lo +
  v. A verifier rejects an item in which (the number of rows of the matrix) * 2^k_lo is above the
  number of plain entries of a row's message (R13.8), which is l - 3 s. This is the first of the further conditions of 8.4, item 23; 16.1, condition 6, says what
  depends on it (from the programs' source: `blocks/iop/lin.h`).

- **R13.14** The item `ranges` is: the list of the lookups; the notice of the oracle message that
  holds the multiplicities; its number among the oracle messages of the chain; the stages of the
  proximity test on that message, a list by R12.4; and the choices' word. A **lookup** is: d; mu;
  live; the list of the digits' weights, elements of F; the offset and the padding key, elements of
  F; the list of its parts; and a list of words, the **first bounds**: the numbers of the variables
  that the last level of each part's product tree binds first (R13.17). A **part** of a lookup is
  one word, lp, where the part has 2^lp **cells**, and the list of its runs. A **run** is four
  words: its first cell, its first row, the number of its cells, its stride. A lookup's name is not
  among the bytes.

  A chain's oracle messages are numbered from 0 in the order of sending. The number of the
  multiplicities' message is 1: it is the second oracle message of the one-time chain, after the
  index-mask message. (from the programs' source)

  What the words mean. A lookup says that every key is in the lookup's table 0, 1, .., 2^d - 1. The
  key at a cell and a column is the offset plus the sum, over the digits q, of the digit's weight
  times the matrix's entry in the row of (cell, q) and that column. Cell number (first cell + p) of
  a run reads, for digit q, the row (first row + p + q x stride). A cell that no run covers is
  padding: it holds the padding key and is not counted. A part's keys have the numbers column +
  2^k_col x cell. (from the programs' source)

  The lookups of a model's matrix. The programs make two lookups, in this order. The first, on
  bytes: d = 8; one digit of weight 1; offset 128. The second, on weights: d = 13; two digits of
  weights 1 and 256; offset 4096. In both the padding key is 0, and the first bounds are 2, 1, 0, 5,
  4, 3, 8, 7, 6 if k_col is at least 9 and the empty list otherwise. The matrix's rows come in units
  of two or four digits, one after the other from row 0. The units, their order, their numbers of
  digits and of rows, and the matrix's number of rows (m0) are those of 11.5: they follow from the
  parameter file, and no line of the layout file enters. A unit of D digits and n rows holds n / D
  numbers, and digit q of number i is in the unit's row q (n / D) + i. The rows are taken in
  **pairs**. The pairs are, in order: first, for each unit of two digits, wherever it is among the
  units, digits 0 and 1 of each of its numbers; then, for each unit of four digits, digits 0 and 1
  of each number and then digits 2 and 3 of each; then the rows after the last unit, in pairs of
  neighbours. The second lookup takes the pairs of the units of two digits, one cell for each pair.
  The first takes all pairs and makes each of a pair's two rows a key of its own. So of a unit of
  four digits a key file's lookups say that each of its cells is from -128 to 127, and they say
  nothing of its numbers. The library takes from the calling program which units have four digits.
  (from the programs' source)

  A lookup's parts and planes. Let N be the number of the parameter file's line `ranges-part`
  (section 8, which writes L for it). The programs reject N above the logarithm, rounded up, of the
  matrix's rows. A part of the first lookup has lp = N and a part of the second lp = N - 1, so that
  each part reads 2^N rows. The pairs are cut into parts in their order; a run of pairs that crosses
  a part's border is cut there; the last part can be partly covered. In the first lookup row q of
  pair i of a part is at the cell q 2^(lp - 1) + i, and its runs have the stride 0. In the second a
  run's stride is the distance from a number's first digit to its second, n / 2. Before the cut,
  each unit of two digits is one run of pairs, each unit of four digits is two (digits 0 and 1;
  digits 2 and 3), and each pair of rows after the last unit is a run of its own; neighbouring runs
  are never joined. A part's runs are in the order of the pairs; in the first lookup all runs of the
  pairs' first rows come first, and then all runs of their second rows. The number w of planes is
  the least number that is at least 2, exceeds lp + k_col for every part, and has 2^w above the
  number of all keys of the lookup, padding included. Then mu is the logarithm of w, rounded up, and
  at least 1; live is 0 if w = 2^mu, and w otherwise. (from the programs' source)

  What the programs reject in `ranges`, beside R13.17's list. These are the programs' tests on the
  item, on its stages and on the matrix's units. They reject each case of the list below, for which
  this paragraph's mark holds too. The programs can be built for field 2 alone (R1.2): the list's 64
  bits and its bound of 60 on k are that field's. (from the programs' source)
  - An item without a lookup.
  - A matrix without rows, a matrix for which the logarithm of the number of rows, rounded up, is
    above 40, or k_col above 40.
  - A lookup with d + w not below the length in bits of the field's prime, which is 64; without a
    digit or without a part; with a digit of weight 0; with a padding key not below 2^d; with a
    number twice among its first bounds, or a first bound b with b + 1 not below lp + k_col for some
    part.
  - A part with lp above 40, with lp + k_col below 1 or not below w, or without a covered cell.
  - A lookup's run without a cell; runs that overlap, are not in rising order, or reach beyond their
    part or beyond the matrix's rows, the last digit's last row included; a stride other than 0
    where a key has one digit; the stride 0 where it has several.
  - A lookup whose keys, padding included, number 2^w or more.
  - An item in which no lookup with one digit of weight 1 covers every row of the matrix.
  - A multiplicities' message that has another number than 1; that has not main rows over F, mask
    rows and an out-of-domain sample; that names a code which is not in `codes`; for which a block
    of planes (below) is longer than a main row; that has not exactly the main rows which the
    multiplicities fill; whose mask rows have no room for the lookups' entries; or whose codes are
    not those of the first stage of the item's proximity test.
  - In that test, beside what R13.19 says of every test: a stage that names a code which is not in
    `codes`, or three words that are not a code (R12.1); a base or mask code whose message has a
    number of entries that is not a power of two; a mask code that is longer than the stage's base
    code; k above 60, or h above k; a stage before the last with k = 0; a last stage with k other
    than 0; a stage before the last whose mask code's message has fewer than 2 k + 1 - h entries.
    For a stage after the first, write (.., l, ..) and (.., l', ..) for its base and mask code and
    (.., l_0, r_0) and (.., l'_0, r'_0) for those of the stage before: the programs reject it if 2^k
    l is not l_0, if 2^h l' is below 2 l'_0 + r_0 + 2 r'_0 + s, or if s is above l'.
  - Of the units (11.5): no unit; an odd number of rows of the matrix, or 2^40 rows or more; a unit
    that has not two or four digits, has no row, or has a number of rows that its digits do not
    divide; a unit that begins before the end of the unit before it, or reaches beyond the matrix's
    rows; a gap between two units, or before the first. The library rejects these cases.
    `cp_verify`, which derives the units from the parameter file by a routine of its own, ends at
    them with `NOT CHECKED`, under the name `internal-weight-limbs`, and not with a rejection (17.2,
    item 1); it ends in the same way at a unit of four digits that is not the embedding's.

  A lookup with more than 4096 stretches of padding is not among these rejections (a stretch is the
  cells that a part's runs leave uncovered between them, before the first or after the last): none
  of these tests counts the stretches. A guard of the library ends the call when the lookup is run,
  and `cp_verify` then ends with `NOT CHECKED` (17.2, item 2). (from the programs' source)

  Why the covering rule and the rule on the weight 0: without them a key file would be accepted for
  a matrix with entries that no lookup reads. (from the programs' source)

  Where the multiplicities are. The oracle message's main rows hold one bit for each entry, and a
  main row has as many entries as its base code's message: none is kept free, since the pad is in a
  mask row. The **live planes** 0, .., w - 1 of a lookup are cut into aligned blocks of plane
  numbers, the largest first: a block begins at a multiple of its size. A block of 2^b planes is a
  vector of 2^(b + d) bits, with bit (plane - first plane of the block) + 2^b x (row of the lookup's
  table). The blocks of all lookups are sorted by their numbers of bits, the largest first; blocks
  with equal numbers keep the order of the lookups and of their blocks. They are placed one after
  the other into the main rows; a block that does not fit into what is left of a row begins the next
  row. (from the programs' source)

  The index-mask message has no main rows. Its mask rows are the least number, at most 4096, whose
  free entries hold two entries for each part of each lookup and, if there is more than one part in
  all, 1 + 2 k_col more. (By the count at the start of this section, m mask rows of e entries each
  have m (e - s) + (m - 1) free entries.) Its mask code is that of a request's own group, and its
  leaf class is 0. (from the programs' source)

### 9.7 The key file's three chains

- **R13.18** The rounds of a key file's one-time chain, in this order:
  1. `SEND seed` and `SEND answer` for the index-mask message;
  2. the same two for the multiplicities;
  3. the lookups of `ranges`, in their order, each by R13.17;
  4. the proximity test whose stages are in `ranges`, by R13.19;
  5. if the lookups of `ranges` have more than one part in all, `BAT-sigma`, one element sampled,
     and a sumcheck with k_col variables of g = 3, without a round in Y (where there is one part in
     all, neither is there);
  6. `SEND seed` and `SEND answer` for W;
  7. `SEAL`.

  With s answers to an out-of-domain sample, here and in R13.19: the message of a `SEND answer` is s
  elements for each oracle message that gets an out-of-domain sample, the message of a `CS` is s
  elements, and s of the elements that a `SEND seed` samples must not be 0.

  The message of a `SEND seed` is the roots of the trees that R7.12 makes of the oracle messages
  sent. The same s seeds serve all oracle messages of the prover's message that get an out-of-domain
  sample. Each seed's point has as many coordinates as the largest of them needs, and each uses the
  first coordinates that it needs. A message with main rows alone needs the logarithm of the number
  of its main rows, rounded up, plus the logarithm, rounded up, of the number of entries of its base
  code's message. A message with mask rows alone needs the same two numbers of its mask rows and
  mask code. A message with both needs the greater of the two sums, plus 1. Answers come message by
  message in the order sent, and within a message by answer (R8.3). (from the programs' source)

  The sumcheck after `BAT-sigma` sums each variable plainly, so by R13.16 each `PSC round` sends the
  constant and the quadratic coefficient. Round j binds column variable j, from the lowest up. The
  linear rows (9.1, item 3) that it joins are in the order of the lookups and, within a lookup, of
  the parts; the j-th of them, from 0, has the j-th power of the sampled element as its factor.
  (from the programs' source)

  How many linear rows (9.1, item 3) a round batches is the last of the eleven words of its entry
  (R4.3). In a key file the `BAT` of the test of `ranges` has s rows, for the multiplicities'
  out-of-domain sample, and for each lookup (w - 1) + (P - 1) + d + 4 + the sum of lp + k_col over
  its parts, where P is the number of the lookup's parts and w, d and lp are R13.17's. `BAT-sigma`
  has the number of parts of all lookups together. The `BAT` rounds of the shared chain and of the
  second chain, and every `CS`, have the numbers of 10.6. (from the programs' source)

- **R13.17** The rounds of one lookup. Let d, mu and live be its first three words (R13.14), let w
  be live, or 2^mu if live is 0, and let its parts have 2^(lp + k_col) keys each, where lp is the
  part's word and the matrix has 2^k_col columns (R13.13). In this order:
  1. `LKP`, one element sampled;
  2. w - 1 sumchecks without a variable, D = 3, the first with three end values and the others with
     two;
  3. for each part but the first, one sumcheck without a variable, D = 2, with two end values;
  4. `JOIN`, mu elements sampled;
  5. for each part, a product tree of lp + k_col levels with no further variable;
  6. a product tree of d levels with mu further variables;
  7. a sumcheck with d variables of g = 3 and then mu variables of g = 2, without a round in Y;
  8. `ZERO`, mu + d elements sampled;
  9. a sumcheck with mu + d variables of g = 3, D = 2 and one end value.

  The programs reject d below 1, mu outside 1 to 6, live not below 2^mu, so that "all planes live"
  is encoded as 0 and never as 2^mu, and w outside 2 to 62; the routine that makes `ranges` from the
  public files rejects a lookup of more than 256 parts, and the routine that sizes the item rejects
  d above 40 (this last clause: from the programs' source). The programs also reject d + w not below
  the length in bits of the field's prime, which is 64 (it is in R13.14's list). So 2^d (2^w - 1) is
  below the field's prime: a multiplicity is a number of w bits, the lookup's table has 2^d entries,
  and the sum of all multiplicities cannot wrap around in the field. The routine that plans a
  lookup's rounds tests that product against the prime itself (the sentences from "The programs also
  reject" on: from the programs' source).

  All variables of these sumchecks are summed under an equality polynomial, so by R13.16 each
  `PSC round` leaves out the constant coefficient. The variables of a level are numbered from 0, and
  variable t belongs to coordinate t of the claim's point. Round j of a sumcheck binds the highest
  variable that is not bound. One exception: in the last level of each part's product tree the
  variables of the lookup's first bounds are bound first, in the list's order, and then the others
  from the highest down. After a sumcheck each sampled element is put back at its variable's
  coordinate, so a point's coordinates keep their numbers whatever the order of the rounds. In the
  product tree of the lookup's table and in the two sumchecks after it the mu plane variables are
  the low ones and the d variables of the lookup's table the high ones. The elements of `JOIN` and
  of `ZERO` are the coordinates 0, 1, .. of their points, in the order sampled. (from the programs'
  source)

- **R13.6** From the one-time chain to the shared chain. The one-time chain's last round is `SEAL`
  (R6.8). Its seal is the one leaf of a binding tree of depth 0 (R7.18): slot 0, with the salt that
  is in the file after the word 0. A tree of depth 0 has no vertex but its leaf, so that leaf's
  digest is the binding tree's root. It does two things in the shared chain: it is the value of the
  statement's second item, and it is used instead of the chain's 32 random bytes in the run's value
  (R4.5). So the shared chain has no random bytes of its own in the file, and every byte of it
  depends on the one-time chain's seal.

  A verifier rejects a key file whose word before the salt is not 0. (from the programs' source)

- **R13.19** A proximity test, and the rounds of a key file's shared and second chains. A
  **proximity test** is given by a list of stages (code, mask code, k, h), which in R13.13 and
  R13.14 holds the first stage too (the lists of R12.14 and R13.10 do not), and by two truth values:
  whether its first fold's mask has a tree of its own, and whether masks ride. It is: `BAT`, one
  element sampled; for each stage but the last `FOLD epsilon`, k rounds `FOLD round`, `SEND seed`
  with the root of the next stage's tree, and `CS`; then `BASE lambda` and `BASE`; messages and
  sampling as in R12.9. A list of fewer than two stages, or with k = 0 before the last stage, is not
  a test of the format.

  Three messages can begin with a root beside what R12.9 gives them. The message of the first
  stage's `FOLD epsilon` does if the first fold's mask has a tree of its own, and the message of a
  later stage's `FOLD epsilon` if masks do not ride: the root of a tree that holds that fold's fresh
  mask alone. If masks do not ride, the message of `BASE lambda` begins with a root as well: of a
  tree that holds the masking word of the last step alone, one main row over E under the last
  stage's base code and one mask row under its mask code.

  Which words say so. Masks ride if the choices' word (R12.5) of the item that holds the test's
  stages is 1: for the test of `ranges` that item's own, for the others that of
  `the test on the table`. The first fold's mask:
  - In the test of `ranges` it is sent with the multiplicities, in their tree, if that item's
    choices' word is 1. It has a tree of its own if the word is 0.
  - In the test on the folded row, or on the folded row and Omega together, and in the test on a
    request's own rows it has a tree of its own unless the word of `the test on the table` on a mask
    sent ahead is 1.
  - In the test on Omega it always has a tree of its own.

  The programs always write 1 for the choices' words, for the words on a mask sent ahead and for the
  words on the high bits, in `ranges`, in `the test on the table` and in `the link`. So in their
  files the only such root is in the first `FOLD epsilon` of the test on Omega. The choices' word of
  a proof's item `request's shape` (R14.3) is 1 in their files too. The leaf classes that the
  programs write are: 0 for a key file's index-mask message, for the multiplicities, for the folded
  row and for W; 1 for Omega; 2 for the shared split's mask message; 3 for a request's split's mask
  message; and 0 in `the link`. (from the programs' source)

  The shared chain, in this order:
  1. `LIN point`, v + k_lo elements sampled;
  2. `SEND seed` and `SEND answer` for the folded row;
  3. `LOC` (R13.11);
  4. `SEND seed` and `SEND answer` for Omega;
  5. `LIN zeta`, as many elements sampled as the logarithm of the matrix's rows, rounded up;
  6. if the folded row's and Omega's base codes differ: `SPLIT`, whose message is one element, then
     the test on the folded row, then the test on Omega; if they are one code: one test on both,
     which is the first of the four lists, and the second list is empty;
  7. `SEAL`.

  If the two base codes differ, the shared split's mask message is sent with Omega in the same two
  messages: `SEND seed` then has two roots, Omega's tree first (the two leaf classes differ), and
  `SEND answer` the answers of both. That depends on the two codes alone and not on that message's
  shape in the item (R13.13).

  In `LIN point`, `LIN zeta`, `LINK point` and `LINK phi` the element that is sampled i-th, i from
  0, is coordinate i of the round's point, and coordinate i belongs to bit i of the index that the
  point runs over (eq, at the start of this section). Of the elements of `LIN point`, the first k_lo
  belong to the columns within a row of the stored table, and the last v are the **fold point**
  gamma of the stored table's 2^v rows: coordinate k_lo + i belongs to bit i of the row's number.
  The elements of `LIN zeta` belong to the bits of the number of a row of the matrix. `LOC` samples
  as many positions as the word of `the test on the table` on the stored table's columns says, in
  [0, 2^a), where a is the first word of the tables' shape (R4.10). After its nonce comes the stored
  table's opening (R8.7): a column is 2^v elements of F wide, with the salt of the tables' shape, in
  a stored tree (R7.4), under the root in the item `table`. (from the programs' source)

  The second chain is the test on a request's own rows. It has no other round and no `SEAL`; the
  file ends with it (R13.2). Its own first query holds neither stages nor codes (R13.5): it takes
  the third list of stages of `the test on the table`, and `codes`, from the first query of the
  chain that it continues, the one-time chain (from the programs' source). In a key file W's base
  code is Omega's, a request's split's mask message does not exist and the fourth list is empty.
  This page does not define parameters in which these two codes differ, and the programs reject
  them.

  The record has no round of its own in the three chains and no item in a message.
  The number of linear rows in the plan's entry for the `BAT` of the test on the folded row holds s
  rows for it. (from the programs' source)

- **R13.7** The second chain continues the one-time chain (R7.13 (a)): it begins with that chain's
  whole list of trees. It then adopts one tree of the shared chain (R7.13 (c)), which takes the next
  place and keeps, in its leaves, the number that it had in the shared chain and the shared chain's
  run's value as its start. The trees that the second chain sends take the places after it, and
  their numbers are those places. It has random bytes and a run's value of its own.

  The adopted tree is Omega's. (from the programs' source)

- **R13.12** Which trees are read at a list. A step of a protocol reads a set of oracle messages at
  a list of positions. The trees that are opened are all trees that hold one of them, each once, by
  increasing place (R8.7). A leaf is opened whole, so the set must hold every oracle message of each
  of these trees. A tree that is as deep as the logarithm of the list's range is opened at the list.
  A shallower tree, of depth b, is opened at the **short list**: the set of the short positions of
  the list's positions (R7.2), sorted, each once. A tree can be read at several lists, of one chain
  or of two: then R7.14 holds.

The trees of a key file, by chain and place, in the programs' files (all words named in R13.19 under
"Which words say so" are 1, and the folded row's and Omega's base codes differ). n(T) is the number
of stages of the test T: the test of `ranges`, on the folded row, on Omega, or of the second chain.
The widths follow from R7.1 and R12.10; in a width, h is the h of the first stage of the test that
the row names. (from the programs' source)

| chain | places | tree | width of a column, in elements of F | opened at |
|---|---|---|---|---|
| one-time | 0 | the index-mask message; as deep as its mask code | 3 x the number of its mask rows | the first list of the second chain's test, as a short list where the mask code is shorter than the test's first base code |
| one-time | 1 | the multiplicities, with the fresh mask of the first fold of the test of `ranges` | the number of their main rows + 3 x the number of their mask rows + 3 x 2^h | the first list of that test |
| one-time | 2 to n(`ranges`) | the later stages of that test, one tree each | R12.10's | the next lists of that test, one each |
| one-time | n(`ranges`) + 1 | W, with the fresh mask of the first fold of the second chain's test | 3 x the number of its main rows + 3 x the number of its mask rows + 3 x 2^h | the first list of the second chain's test |
| shared | none | the stored table | 2^v | the list of `LOC` |
| shared | 0 | the folded row, with the fresh mask of the first fold of its test | 3 x the number of its main rows + 3 x 2^h | the first list of the test on the folded row |
| shared | 1 | Omega, alone | 3 x the number of its main rows | the first list of the test on Omega; and in the second chain |
| shared | 2 | the shared split's mask message; as deep as its mask code | 3 | the first lists of both tests of the chain, each as a short list where the mask code is shorter than the test's first base code |
| shared | the next n(folded row) - 1 | the later stages of the test on the folded row | R12.10's | the next lists of that test |
| shared | the next 1 | the fresh mask of the first fold of the test on Omega, alone; as deep as its mask code | 3 x 2^h | the first list of the test on Omega, as a short list where the mask code is shorter than the base code |
| shared | the next n(Omega) - 1 | the later stages of the test on Omega | R12.10's | the next lists of that test |
| second | 0 to n(`ranges`) + 1 | the one-time chain's trees | as above | as above |
| second | n(`ranges`) + 2 | Omega's tree, adopted | as above | the first list of the second chain's test |
| second | the next n(second chain) - 1 | the later stages of the second chain's test | R12.10's | the next lists of that test |

If the two base codes are one, the shared chain has place 0, place 1 (Omega, alone) and the later
stages of the one test; its first list opens places 0 and 1. (from the programs' source)

### 9.8 The leaf file's chain

- **R13.10** The rounds of the link's chain. The names, kinds and items are those of R12.8 and R12.9
  unless this rule says otherwise. Every item but a root is an element of E.

| rounds | name | kind | the prover's message | the verifier's message | after it in the file |
|---|---|---|---|---|---|
| 1 | `LINK point` | 1 | nothing | v elements | |
| 1 | `SEND seed` | 3 | the root of tree 0 | s seeds | |
| 1 | `SEND answer` | 7 | the answers to the out-of-domain sample for the anchor's folded row, then for the leaf table's, s each | nothing | |
| 1 | `LINK` | 4 | nothing | nu_0 + nu_j positions in [0, 2^a) for the tables' code, and no element | the nonce; the anchor's opening; the leaf table's (R7.19, R8.7) |
| 1 | `LINK phi` | 1 | nothing | c elements, as the plan says | |
| 1 | `BAT` | 1 | nothing | one element | |
| then | the stages of a proximity test | | as in R12.9 from its first `FOLD epsilon` on | | the j-th round `CS` or `BASE`, a round that samples positions, opens tree j - 1 |

  The first stage is not written in `the link`: it is the group itself, with k = 1 + the logarithm
  of R, rounded up, and h = 0. Tree 0 holds the two folded rows' oracle messages, the anchor's
  before the leaf table's, R main rows over E each, and, where the word of `the link` on a mask sent
  ahead is 1, the fresh mask of the group's fold: its columns are then 3 (2 R + 1) elements of F
  wide. Where that word is 0, tree 0 holds the two messages alone, 3 x 2 R wide, and the mask is
  sent with the fold's own first message. The later trees are R12.10's. A column of a stored table
  is 2^v elements of F wide, with the salt of the shape.

  The v elements of `LINK point` are one fold point for both tables: the i-th, from 0, belongs to
  bit i of the number of a table's row (R13.19). The number c of elements of `LINK phi` is the least
  number with 2^c not below the number l of entries of a row's message; the i-th belongs to bit i of
  the number of an entry of the folded message. The number of linear rows in the plan's entry for
  `BAT` (R4.3) is nu_0 + nu_j + 1 + 4 s: one row for each position that `LINK` samples, the one row
  that compares the two folded rows, s rows for the out-of-domain sample of each of the two
  oracle messages, and s rows for the record of each of the two tables. The chain has no `SEAL`; the file ends with it (R13.3). The library's routine for
  a leaf file (`check_entry`) rejects, before it reads a byte of the file, an anchor that has passed
  no check: one whose key file was rejected or was never checked (for `cp_verify`, which calls
  `check_entry_if_tested` on a table read from the key file without its proofs, this test shows
  nothing: `blocks/chain/closed.h`, "Not checked"; 13.6, 13.7). By the routine's comment
  (`blocks/frame/check_entry.h`) it does not check that this table is the anchor: it also takes a
  leaf table that an earlier call of `check_entry` has accepted. Before it reads the chain, a
  verifier looks at what it has kept of the anchor (R13.5) and at what it has read of the leaf
  table. It rejects the leaf file if either has another shape than the tables' shape in `the link`,
  or was made under another configuration's digest than the chain's, or has a record that has not
  one stored answer for each row and seed. (from the programs' source)

  The leaf table has a root of its own, so a name, seeds and stored answers of its
  own (R4.8, R4.10). The plan begins with the entry `RECORD` (R13.15). The two folded rows are compared on the plain
  entries alone, l - 3 s entries (R13.8). (from the programs' source)

### 9.9 A chain of segments

These rules define the types and the frame of rounds of a request's first chain. Section 10 uses
them, with the changes of R14.4, which also moves the trees' numbers; a key file and a leaf file do
not. A segment is one piece of a request's proof. Hashed and printed names call it a link: the item
`links' identifiers`, the parameter lines `group link` and `stages link`, and the name of the
example protocol "links".

- **R12.12** A **message of a segment** is six words: the number of its main rows; 1 if they are
  over F or if there are none, 3 if they are over E; the number of its mask rows; its leaf class;
  its **moment**, the number, from 0, of the prover's message of its segment in which it is sent;
  its **life**, the number of later segments for which its main rows are kept. Its codes and the
  number of answers to its out-of-domain sample are not words of its own: they are the chain's
  (R12.14).

  Every message of a segment gets an out-of-domain sample: the programs reject a chain of segments
  whose number of answers is below 1. The library also rejects, among others: a message without
  rows, or with more than 2^20 main rows or more than 2^20 mask rows; a message with a life above 0
  that has no main rows, has mask rows, or has a moment other than 0; a segment without a message of
  moment 0, or whose messages do not come by increasing moment, or that leaves a moment out; a
  message of segment j, of L segments, with a life f for which j + f is above L; and a first segment
  in which no message has both main rows and the life 0. In `cp_verify` these tests end under the
  rejection name `parameters-not-admitted`; but `cp_verify` reads the layout file first, and rejects
  what it finds there under `plan-layout` (13.8). So, from files, these cases end under
  `parameters-not-admitted`: more than 2^20 and fewer than 2^32 main rows, a message with a life
  above 0 at a moment other than 0, and j + f above L with f below 2^20. These end under
  `plan-layout`: a message with 2^32 rows or more, or with a life of 2^20 or more, which the reader
  of the layout file rejects line by line; a message without rows; a message with a life above 0
  that has mask rows, which can only be the first message of a segment; messages that do not come by
  increasing moment; and a moment without a message, the moment 0 too, whose arrays then have no
  block. That is so of a moment that has an array: a moment with no array and no message passes the
  reading of the layout file. It ends under `parameters-not-admitted` if it is the moment 0 or if a
  later moment of its segment has a message. This page does not say under what a last moment of that
  kind ends, or whether a sub-proof has such a moment. A layout file cannot express the other cases.
  (from the programs' source)

- **R12.13** A **segment** is one word and a list: how many plain scalars are sent in a round of
  their own right before the segment's first prover's message; and the segment's messages (R12.12),
  by moment.

- **R12.14** A **chain of segments** is, in this order:
  1. two words, the number of the base code and the number of the mask code, which all segments'
     groups share;
  2. the list of the segments (R12.13);
  3. the list of the notices (R12.3) of the oracle messages that are sent in the very first prover's
     message and belong to no segment's group;
  4. the number of answers to the out-of-domain sample of a carrying message (defined below this
     list);
  5. the list of the stages of the last proximity test from the second on (R12.4): the first stage
     is the group of the last carrying message, and its numbers follow by R12.16;
  6. the word of R12.5.

  The number of answers is the plan's sixth word as well.

  Parameters in which that number and the plan's sixth word differ are not parameters of the format.
  The **carrying message** of a group m > 1 is an oracle message of two main rows over E under the
  chain's base code, without mask rows, which is sent first in the group's tree (R12.16, R12.18).
  (from the programs' source)

- **R12.16** The groups of a chain of segments. A chain of L segments has L + 1 groups: one for each
  segment, and the group of the last proximity test. Group m has one tree, tree m - 1, and two
  numbers k and h, which follow from the item of R12.14 and are not words of their own. One tree for
  each group, with that number, is the case of the example protocol "links" (R12.17). In a proof a
  group can have several trees and the numbers are others: R14.4 (a) and (d) give them. A tree's
  number is hashed into its leaves (R7.13). Write up(r) for the least power of two that is at least
  r, and 0 for r = 0. A message of segment j with life f > 0 is **kept**: its main rows are sent
  again in the trees of groups j + 1 to j + f, and they count in group j + f alone. The main rows
  that count in group m are: 2, for the carrying message, if m > 1; up(r) for each kept message of r
  main rows whose life ends in group m; and up(r) for each message of segment m with life 0. The
  mask rows that count are up(r) for each message of segment m with r mask rows. Then h is the
  logarithm, rounded up, of the mask rows that count (0 if there are none or one), and k is the
  greatest of three numbers: the logarithm, rounded up, of the main rows that count; h; and 1. The
  last group has no segment's messages.

- **R12.15** The example protocol "links" (the toy run `links`: R4.9) is a chain of segments, each
  of which is meant to show a product of vectors and passes one vector on to the next. R12.17 and
  R12.18 use it to show the frame of rounds. No key file, leaf file or proof file holds a chain of
  this protocol. Its chain has the label 2 and random bytes by case (a) of R4.5. After the version's
  item and the plan (R4.3) its parameters are three items, in this order: `configuration` (R4.9),
  `codes` (R12.2), `the chain of links` (the chain of segments: R12.14). Its statement is empty. Its
  public file is encoded as R12.7's, with these lines, in this order: a first line of free text that
  begins with `run links`; `format`; `target`; `work-on-positions`; `salt-bytes`; `nonce-bytes`;
  `position-bytes`; `samples`; `links`, the number of segments; `log-length`; `log-n`; `mask-log-n`;
  and a line `stage` with four numbers for each later stage of the last proximity test. Its rounds,
  and how each message is cut into items, are R12.16 to R12.18.

- **R12.17** The rounds of "links", and how each message is cut into items. The names, kinds and
  items are those of R12.8 and R12.9 unless this rule says otherwise. V is the number of variables
  of a segment's vectors: the public file's `log-length` less 1, since a vector fills the lower half
  of a row. s is the number of answers (R12.14). Every item but a root is an element of E. The rows
  from `ZERO` to `PSC values` are the sub-proof of this example; in a proof other sub-proofs come
  instead of them (section 10). The table is read segment by segment: segment m has its two rounds
  of the first or the second pair of rows, then the rows from `ZERO` to `FOLD round`; then segment
  m + 1 begins.

| rounds | name | the prover's message | the verifier's message | after it in the file |
|---|---|---|---|---|
| for segment 1: 1 | `SEND seed` | the root of tree 0 | s seeds | |
| 1 | `SEND answer` | s answers for each oracle message of tree 0 that gets an out-of-domain sample, in the order of sending | nothing | |
| for each segment m > 1: 1 | `SEND seed` | the root of tree m - 1 | s seeds | |
| 1 | `LOC` | s answers for each oracle message of tree m - 1 that gets an out-of-domain sample, in the order of sending | the positions, and no element | the nonce; the opening of tree m - 2 |
| then for each segment: 1 | `ZERO` | nothing | V elements | |
| 1; V; 1; 1 | `PSC epsilon`; `PSC round`; `PSC Y`; `PSC values` | as in R12.9 | as in R12.9 | |
| 1 | `BAT` | nothing | one element | |
| 1; k of the segment's group | `FOLD epsilon`; `FOLD round` | as in R12.9 | as in R12.9 | |
| after the last segment: 1 | `SEND seed` | the root of tree L | s seeds | |
| 1 | `LOC` | s answers for the carrying message | the positions, and no element | the nonce; the opening of tree L - 1 |
| 1 | `BAT` | nothing | one element | |
| then | the stages of the last proximity test | as in R12.9 from its first `FOLD epsilon` on: stage 1 has the last group's k, the later stages those of R12.14's list; the trees go on from L + 1 | | the j-th round `CS` or `BASE`, a round that samples positions, opens tree L - 1 + j |

  The oracle messages of a group's tree that get an out-of-domain sample are, in the order of
  sending: the carrying message, if m > 1; the kept messages that are sent again, the one that was
  first sent earliest coming first, and of two that were first sent in one prover's message the one
  that comes first in their segment's list (R12.13; from the programs' source); and the messages of
  segment m. So in "links" the (i + 1)-th round that samples positions in the chain opens tree i,
  and no other round does; in a proof this does not hold, and neither do the trees' numbers in the
  table above (R14.4 (a), (d)). The numbers of coordinates, positions, rows and entries are the
  plan's and the codes'. This section does not define what the carrying message holds, or the checks
  that the verifier makes on these items.

- **R12.18** What is in a tree of "links". All oracle messages of a group have the leaf class 0 in
  this example, so by R7.12 they join one tree. Its columns hold, by R7.1, first the main rows in
  the order of R12.17's list of the messages that get an out-of-domain sample, then the mask rows of
  segment m's messages, then the fresh mask of the group's fold, which has 2^h mask rows over E,
  gets no out-of-domain sample and is sent in the same prover's message (R12.10). In elements of F a
  column has this width: 6 for the carrying message, if m > 1; r or 3 r for a message of r main rows
  over F or over E; 3 for each mask row; 3 x 2^h for the fresh mask. The trees of the later stages
  are R12.10's.

- **R12.7** The public file of the example protocol "product", whose digest by R4.9 is the run's
  configuration. No key file, leaf file or proof file depends on it. It is a text of lines, in the
  bytes 0x20 to 0x7E and the line feed (0x0A); each line ends with a line feed. The first line is
  free text that begins with `run product`; what follows on it says which variant of the run it is.
  Every other line is a name and numbers, separated by one space. The lines are, in this order: the
  first; `format` and the four characters of the version word; `target` and the number of bits;
  `work-on-positions`; `salt-bytes`; `nonce-bytes`; `position-bytes`; `samples`; `log-length`;
  `log-n`; `mask-log-n`; and then, for each stage of the proximity test after the first, a line
  `stage` with four numbers: k, h, the logarithm of the base code's length and that of the mask
  code's. A verifier that is defined from this page takes the parameters' bytes as its public input,
  and compares their item `configuration` with the digest of this text (that item is a value of the
  hash: R4.9, section 18). This page does not define how the codes, the stages and the plan follow
  from the text. The programs' own verifier of the run takes the text and the proof and nothing
  else, sizes and plans the run itself, and rejects a text that is not encoded in this way (of the
  first line it tests only that it begins with `run `; from the programs' source): for this run it
  is there, and not on this page, that the second half of R0.5 is met.
