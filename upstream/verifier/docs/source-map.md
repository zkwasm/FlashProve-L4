# Source map of the verifier

What each file of `verifier/src/` holds, for whoever reads or changes the code. Most files' own head comments say more. The programs have one field, p = 2^64 - 2^32 + 1; `build.sh --field` accepts only 2 and changes nothing.

## Reading the names

The names in these sources are short, and many are built from a few words of this project's own. The tables below define those words once; [docs/glossary.md](../../docs/glossary.md) has the terms of the protocol. Where the Lean port of this verifier (in `lean/OnBytes/`) has the same thing, it mostly has the same name, so the tables serve both texts; some of what the verifier's sources hold has no counterpart in the port. The last table lists the names that are known to differ.

### Words that the names are made of

| word | in the names of `verifier/src/` |
|---|---|
| claim | the statement that an affine form (a constant, plus weighted entries of virtual rows, plus weighted hidden values) is 0; a sub-proof adds up all its claims, each under its own weight, into one equation |
| slot | the number of one masked value that a sub-proof sends, counted in the order of the file; the mask entry with the same number goes with it |
| cell | one committed integer of a matrix; the smallness test is to show that every cell is small |
| array | R rows by K columns that a sub-proof commits to or reads (in names: *tree*) |
| virtual row | a weighted sum of an array's rows; a claim reads single entries of it |
| segment | the part of a request's proof that belongs to one sub-proof (in names: *link*) |
| chain, channel | a chain is one Fiat-Shamir transcript: a sequence of hashes; the channel (`blk::VerifierRounds`) is the object through which the verifier reads a chain's messages and draws its challenges |
| call | two things: one run of `cp_verify`; and a lookup call: 2^n keys, each to be found in a public table of 2^k rows |
| stage | one of the three parts of a `cp_verify` call, each a child process: `key`, `leaf`, `request` |
| statement | the words that a request's proof is about (in names: *st*, *header*) |
| scan | the strict scan: the one pass over the parameter file's lines after the leading ones; it accepts one order and one spelling |
| smallness test | the test that ends a sub-proof; the comments call it the approximate range check (in names: *tail*) |
| plane | in a lookup: bit j of every table row's multiplicity, for j below 32 |
| digit row | a norm's row of byte digits, one row for each norm and token (in names: *led*; in rejection names: *ledger*) |

### Parts of names

Each line speaks of the names that it lists, and of no others.

| part | in these names | what it means there |
|---|---|---|
| forest | `HForest`, every `forestX`, `FOREST_JOIN_SECTION`, `FOREST_PLANE_SECTIONS` | the lookups of one group of lookup calls (the calls of one sub-proof, or of its smallness test), run together as one lookup; nothing to do with Merkle trees |
| fork, Fork | `HFork`, `AFork`, `aforkShapeHid`, `WsForkOut`, `wsForkX` | a sub-proof: the argument for one part of the forward pass (a layer group, its attention, or the output head); afork, AFork: the attention's |
| link, Link | `LinkCtx`, `linkRefused`; *link* in comments and output | a segment; not a symbolic link, and not LINK, the leaf file's proof in `blocks/` |
| tail, Tail | `Tail2`, `HTailF2` | the smallness test |
| served, Served, SERVED_ | `Served`, `servedForm`, `rotaryServed`, the four `SERVED_X` | of the standard parameter form: conditions on the public inputs and on some options of the call, without which no run ends YES or WITHIN LIMITS; `SERVED_X` is the printed name of condition X. In `tauServed` and `blk::SERVED_VERSION`: what this program takes |
| tr | the ten `trX` | the transcript as a sub-proof's routines use it: `trX` reads values of the prover's message, draws challenges, or closes a round |
| chain | the ten `chainX` that have a `trX` | the same call on the channel; each `trX` calls its `chainX` |
| Answer | `trAnswer`, `trAnswerNotZero`, `chainAnswer`, `chainAnswerNotZero` | a challenge that the verifier draws (the chain's answer to the prover's message); never something that the prover sends. |
| Empty | `trEmpty`, `chainEmpty` | it is the verifier's message that is empty: the prover has sent its message, and the verifier answers with no challenge; the chain still makes its hash query |
| Sum | `trSum`, `chainSum` | reads one element, the announced sum of a PSC's hiding polynomial |
| Row, Rows | `pscRow`, `sc3Row`, `trBatchRows`, `chainBatchRows` | a linear claim, claims. In `HRow`, `Row5`, `blockRow`, `blockRows`, `rowWeights` a row is a row |
| Batch | `trBatch`, `trBatchOpen`, `trBatchRows` and their `chainX`; `t2Batch` | the round that batches linear claims. `trBatch` draws x and returns x^0 .. x^(n-1); `trBatchOpen` returns x alone; `trBatchRows` states afterwards how many claims x has batched, and the process ends unless that is the planned number. Nothing is opened. `t2Batch` makes that draw after a sub-proof's last message |
| NoBatching | `chainNoBatching` | ends the run if a draw names the batching round, which only `chainBatch` and `chainBatchOpen` may draw |
| Held, Ask, Whole, Agree, Fill, Units | `chainHeld`, `groupsBaseHeld`, `forestPubAsk`, `askedWhole`, `tensorsAgree`, `tensorsFill`, `tokenFactorUnits` | these seven return nothing and reject, or end the run, unless the thing named is in order. Not `readWholeFile`, `fillL` |
| pol, Pol | `PolScan`, `polScan`, `polPrescan`, `polSkeleton`, `polLayW`, `polLayout`, `polWord` | the parameter file, policy.txt; neither a polynomial nor the acceptance policy |
| policy, Policy | `policyKind`, `PolicyNamed`, `policyNamedRead`, `POLICY_OPTION_FORM_TEXT`, `acceptancePolicyIsExactMatch` | the acceptance policy. But in `cpPolicyWord`, in the role "policy" and in every rejection name `policy-..`: the parameter file |
| word, Word |  | three things. The format's version word, four characters read as one number: `PROGRAM_VERSION_WORD`, `runVersionWord`, `runVersionWordFixed`, `versionWordAdmitted`, `admitWord`, `cpPolicyWord`. A value line of the parameter file: `polWord`, `wordLineForm`. 32-bit words of a file or of the statement: `mustReadWords`, `statementLengthWords`, `TargetBitsWord` |
| psc, Psc | `Psc`, `pscBegin`, `pscRound`, `pscEnds`, `pscRow` | PSC, the sumcheck with a hiding polynomial (also called the masked sumcheck) |
| tsc | `tscIn`, `tscRounds`, `tscSlots`; the type is `TokenSumcheckState` | the token sumcheck: a PSC whose sum runs over the token positions. `tscRounds` reads one without its end: the caller runs further rounds and the end itself |
| sc3, scd | `sc3In`, `sc3A`, `sc3B`, `sc3Row`, `sc3Slots`, `scdIn`; the types are `Degree3SumcheckState`, `VariedDegreeSumcheckState` | the attention sub-proof's two sumchecks: SC3, whose rounds all have degree 3, and SCD, with a degree of its own for each round |
| In | `tscIn`, `sc3In`, `scdIn`, `roundIn`; not `rootInH` | reads that part of the proof and draws its challenges; it tests no sum: the claim that it leaves does |
| ws, Ws, WS_ | `WsMerged`, `WsPending`, `wsPending`, `WS_ROWS_MOST_LOG2`, `WS_SECOND_DIGIT`; `WsLay`, `WsGrp`, `WsUnit` .. | the committed matrix W of the weights: its layout, and a sub-proof's claims on it |
| acct, ACCT_ | `acctNote`, `acctCounted`, `ACCT_GROUPS_MOST` | the error accounting: a stage's record of its rounds' error estimates (the program's own; no security level) |
| par, Par | `hParOf`, `hParDigest` | the three configuration files (parameter file, KL.bin, layout file); h_par is the digest of the three. Not `parFor`, a parallel loop |
| cp, Cp | `cpMain`, `cpPlanMain`, `cpStagesRun`, `cpUsage`, `cpLineOf`, `cpNotChecked`, `cpCovered`, `cpPolicyWord`, `CpId`, `CpStage` | of the program cp_verify itself: its command line, its three stages, its last lines |
| Req | `blk::ReqLayout`, `ReqEquation` .. | REQ, the outer structure of one request's own proof, a sequence of segments (`blocks/`) |
| tau |  | three things. `TAU`, `tauServed`, `T2TAU`: the number of random combinations in the smallness test (the parameter file's line `tau`). `Call::tau`, `tauAt`, `tauFC` .., `setTau`: a public table's row values. `ProductLevel::tau`: a challenge |
| Slots | `layerSlots`, `sc3Slots`, `tscSlots`, `sumcheckSlots`, `sumcheckSlotsDegrees` | how many slots that part of a proof takes |
| Elems, Lins, Object(s) |  | `forestElems`: a count of slots. `forestLins`: a count of claims; `forestPlaneLins` adds three claims. `forestObjects`, `layerObject`: add claims |
| Pub | `forestPubAsk`, `forestPubRun` | the public counts of padding keys, one for each tree of keys that belongs to a call |
| Tie, Join | `forestTie`, `forestJoin`, `forestJoinAdded` | the tie: the equation between the roots of the lookup's two sides; the join: the claim that links its key side to the calls' key vectors. `tieLeft`, `tieRight` are another tie, the smallness test's |
| Added | `forestJoinAdded`, `forestPlanesAdded` | sets a mark that the caller has added these claims; the sub-proof's last check ends the run if a mark is missing; the routine cannot see what was added |
| Laid | `forestPaddingLaid` | compares the caller's own count of padding places, tree by tree, with the public counts, and ends the run if they differ; it sets no mark, and nothing tests that it was called |
| SECTION, SECTIONS | `FOREST_JOIN_SECTION`, `FOREST_PLANE_SECTIONS`; `HFork::section` | labels of groups of claims, for tests' reports alone |
| layer | `layerSlots`, `layerObject` | a product level (docs/format/11-1-machinery.md): the sumcheck for a sum of products of two vectors; a level of a lookup's product tree is one, and so are the norms' sums of squares and the gated products; not a layer of the model |
| Place | `PlaceF`, `samePlaceF` | the placement record of a matrix that one sub-proof commits to and another reads: where it lies |
| block | `blockRow`, `blockRows` | a run of adjacent features of a committed matrix, T cells each; neither a layer group, nor a step of the protocol, nor a piece of a file (`readBlocks`) |
| long, hi | `longPoint`, `longPhi`, `hiTab` | of such a run: the point inside it, the weights of its cells, the weights over the coordinates above it |
| tokenFactor | `tokenFactorTaken`, `tokenFactorUnits` | the token (vocabulary) index as column bits of W |
| group |  | several things. `groupsBaseHeld`, `NGr`: layer groups (the format's limit is 28). `ACCT_GROUPS_MOST`: groups of rounds in the error accounting. In comments on lookups: the calls that run as one lookup (see forest). In `blocks/` and the glossary: oracle messages under one code |
| plan, Plan |  | two things. The plan of a chain (`blk::Plan`, `forestPlan`): the list of its rounds, fixed before the run. A sub-proof's plan (`HPlan2` ..): its arrays and counts, made from the public files without reading a proof. `cpPlanMain`: the calls `-plan` |
| led, Led, LED | `LEDG`, `LEDNE`, `LedK`, every `ledX` | the digit rows |
| A, B | `sc3A`, `sc3B` | the two hidden factors A_b and B_b of SC3's summand: their end values |
| PhiB, PhiP | `forestPhiB`, `forestPhiP` | the weights phi over the rows of the planes' matrix in the column sums TB and TP |
| F at a name's end | `PlaceF`, `samePlaceF`, `eqPlainF`, and the smallness test's routines (`shapeF`, `tileSelfTestF` ..) | of the smallness test's packing and placement |
| L | `fillL`, `trimL` | the vectors L: the one equation's coefficients on the entries of each virtual row |
| W | `polLayW` | one word of the statement |
| H at a name's start | `HFork`, `HTree`, `HRow`, `HForest`, `HFun`, `HPiece`, `HCallZ`, `HStack` | of the engine that the sub-proofs share (`vf_subproof_engine.h` and the headers that it includes) |
| H at a name's end |  | mostly the output head's (`ledNRH`, `NMINH`, `polCallsH`, `lmbH`) |
| leaf, child, copy; entry | in comments and output | a further stored copy of the committed table; its file. Not `HForest::leaves` (a product tree's leaves) nor `gateTableEntry`, `sigmoidTableEntry` (an entry of a public table) |
| refuse | `refuse()`, `linkRefused` | reject |
| asked, bar, BAR | `askedWhole`, `askedBelowBar`, `BAR` | the requested bits (`-bits`); the threshold on them |

### Names that do not say what the routine does

Many of these names are shared with the Lean port. Read them by this table, not by their words.

| name | what it does |
|---|---|
| `lastLine` | prints the last lines of a stage and returns the stage's exit status |
| `servedForm` | returns the names of those of `Served`'s 18 conditions that are not met, or "". Two more conditions of the form are tested elsewhere (see `Served`), so "" does not say that the whole form is met |
| `tauServed` | returns whether the number of combinations (line `tau`) is one of the values that this program takes; `polLayout` rejects (`hid-tau`) if not. It is no condition of the standard parameter form |
| `aboveClip` | returns an upper bound on how many checked tokens can have a cost above the clip C; it tests nothing; each such token may be any token, and it does not bound the tokens that may be any token |
| `goodEnding` | says whether a stage that has passed its checks may print its passing line, or must end NOT-A-VERIFICATION |
| `needBytes` | reads a whole input file, or prints `input-unreadable-..` and ends the call with status 2 (its brother for 32-bit words is `mustReadWords`) |
| `trAnswer`, `chainAnswer` | close the prover's message and draw one challenge; they reject (`chain-refused`) on a fault in the proof's bytes or a missing or failing nonce. The prover answers nothing here |
| `trBatchOpen` | draws the element whose powers batch the linear claims, and returns it; nothing is opened |
| `trBatchRows` | tells the chain how many claims that element has batched; the process ends unless that is the planned number. The "rows" are claims |
| `layerSlots` | the number of slots of a product level; not of a layer of the model |
| `layerObject` | adds a product level's claim to the sub-proof |
| `pscRow` | returns the one linear claim that a PSC leaves; no row of an array. The caller must add it: nothing checks that |
| `sc3Row` | forms the one linear claim that an SC3 leaves, and stores it |
| `samePlaceF` | compares two placement records and returns the label of the first difference, or "" (also "P3" if the owner's columns run past its array); it does not return true or false |
| `roundValues` | the number of scalars that a round sends |
| `forestElems` | the number of slots of a group's lookup |
| `forestLins` | the number of claims that `forestObjects` adds |
| `forestPad` | runs all messages of a group's lookup and returns its state; on the way it rejects bad sizes and padding counts. It pads nothing (the lookup's padding keys are another matter) |
| `forestPlaneCols` | the number of columns of the planes' matrix; it also rejects a size that does not fit, and is called twice for that alone |
| `aforkShapeHid` | computes from public parameters, reading no proof, the sizes and padding counts of the attention sub-proof's one lookup; rejects bad sizes |
| `polWord` | searches the parameter file's text, apart from the strict scan, for the first line "name value"; sets the value and returns true, or returns false; it rejects nothing |
| `polLayout` | runs the tests that follow the scan, on the statement's words together (it rejects), sets `TAU`, and returns the layout of the layer sub-proofs |
| `polSkeleton` | sizes the empty statement, writes the few words that are known before the scan, and fixes the run's version word (it rejects another) |
| `cpCovered` | forms (does not print) the line "covered:" and says of which kind the parameter file's acceptance policy is; YES, WITHIN LIMITS or NOT CHECKED depends on its result |
| `cpPolicyWord` | returns the version word on the parameter file's first line and rejects any word but this program's; it has nothing to do with the acceptance policy |
| `linkRefused` | ends the run after the checker of a request's chain has returned false: prints the chain's reason, if any, and rejects (`chain-refused`) |
| `lastOf` | the last k coordinates of a point; rejects a k out of range |
| `rootInH` | receives the roots of the segment's next commit point, where the prover commits to further arrays; rejects (`plan-link`) unless the caller names exactly the layout's arrays there |
| `doorH` | hands the sub-proof's one equation to the chain and closes the segment |
| `ledNS` | the number of a norm's sums of limb products (the format's NS); these are no part of the digit row |
| `ledLines5` | the eight equations of a five-limb digit row that come after the nine carry equations; `ledLines` gives their number, 8 or 5 |
| `eqPlainF` | eq(z, idx) as a plain product, for the smallness test's self-test, which runs whenever a sub-proof is checked |
| `internalName` | says whether a rejection's name is one meant for a fault of the program ("internal-..", also behind a sub-proof's prefix: "G0:internal-..") and not of a proof |
| `blockRows` | the weights of the cells of one run of adjacent features |
| `WS_SECOND_DIGIT` | the radix, 256, in which a weight is stored as digits; no digit |

### The short names

Constants and variables in capitals, short type names, and five functions that give counts.

| name | what it is |
|---|---|
| `Aff` | An affine form in hidden things: terms on entries of virtual rows (e), terms on hidden slot values (s), and a constant c |
| `BAR` | The threshold: the fewest requested bits (-bits) with which a call can pass; 128, the only value that passes (fewer ends NOT-A-VERIFICATION at best, more is refused). A `double` here; the provers' sources have an `int` of the same name and value (`prover/common/setting.h`) |
| `BIGH` | The number 2^37 (the page's BIG) that the prover adds b times (b > 0 is charged C; `aboveClip` bounds how many positions can have it) to every logit margin of a token position whose cost is above the clip C |
| `CLAMP` | The clamp's bound 2047: lookup table FC clamps to -2047 .. 2047 |
| `CpId` | What cpMain passes to cpStagesRun: the caller's commitment ID (-id), the printed digest h_par, and three of the call's options (-policy given, -show-stages, -no-proof-of-work) |
| `CpStage` | A record of one stage of a call (key, leaf or request) for the parent to start as a child process: its name, its number of threads and the function that runs it |
| `Ent` | One term of an affine form: coef times entry idx of virtual row `row` of array `tree` |
| `EqRun` | An iterator that gives lead * eq(zp, idx) at consecutive indices idx, rebuilding only the factors from the highest bit that changed down; only the tiling self-test uses it |
| `EXC` | The number of cells (base-256 digits), 3, that hold one value ex of the exponential table FE |
| `GCL` | 1 if the parameter file has the line "gcl 1": the gating value, the key at which table SILU is read, is clamped; else 0 |
| `HCallZ` | A lookup call (Call) together with the pieces of arrays that its keys come from |
| `HFork` | The state of one sub-proof: its arrays, its slots (shown values, mask factors), and the one equation into which all claims are summed |
| `HFun` | A functional: a weight vector phi of length L that is applied to every run of L cells of each column of matrix M, with the place `at` of the table of the results in the committed vector that `tree` names: 0 means U1 and 2 means U3. `tree` is not an index into `HFork::trees`; any other value ends the stage under the name `internal-t2f-functional-<name>` |
| `HPiece` | One piece of a lookup call: a block of Rp rows by Cp columns of an array, and the 2^b keys from `off` on that it takes |
| `HRow` | A virtual row of an array: the vector sum_r coef[r] * (row r), with L, the one equation's coefficients on its entries |
| `HStack` | The layer sub-proof's batched masked sumcheck: lam rounds without an equality factor, and their challenges. The output head's sub-proof uses only lam and rs |
| `HTree` | An array of R rows and K columns that a sub-proof commits to or reads; no Merkle tree |
| `KCAP` | The largest number of columns of a matrix of the smallness test: 8191; above it "hid-k-range" |
| `LANES` | Cell rows in one packed row: up to 3 cells of a column are packed into one field element |
| `LBU` | The exponent 15 in the key of lookup UT: the key of u, the up projection's output, is u + 2^15, a row of a table of 2^16 rows, so -2^15 <= u < 2^15 |
| `LBZ` | The logarithm 17 of beta_z = 2^17: the smallness test is to bound each random combination Z of a matrix's cells by it (Z + 2^17 has 18 bits); Ty = 2^18 is the bound on \|cell\| that the no-overflow conditions assume and that the smallness test is meant to show |
| `LBZW` | One less than the logarithm of the bound that the head's no-overflow condition J-E assumes on each base-256 digit of an embedding weight: the bound is 2^(20 + 1) |
| `LEDG` | For each of the six hidden numbers c, b, u, k, z, zs in the output head's digit row: its offset after the norm's digits and its length in digits |
| `LedK` | A template that turns a field constant x into the type T in which a digit row is written: an affine form in the sub-proofs, or a field element |
| `LEDNE` | How many more digits, 21, the output head's digit row has after the final norm's |
| `LRAD` | The radix of the packing as a logarithm: 20 |
| `Owed` | A claim that a function leaves for its caller to add: its name and whether it has been added; settle() rejects if not |
| `PlaceF` | Holds where a matrix lies that one sub-proof commits to and another reads: owner, array index, shape, columns, cell rows. Owner's and reader's records must be equal. |
| `POW` | Bits of proof of work before codeword positions are drawn in a request's own rounds: first number of the parameter file's line "work positions" |
| `POWC` | The same in the committer's rounds: third number of the line "work positions" |
| `POWS` | The same in a batch's shared rounds: second number of the line "work positions" |
| `Psc` | The state of one masked sumcheck (PSC): the form that the sum must equal, eps, the running claim, its first slot, the variables so far and their challenges |
| `RADB` | The radix of the packing as a number: B = 2^20 |
| `RCL` | 1 if the parameter file has the line "rcl 1": the router logits are clamped before the routing reads them; else 0 |
| `Row5` | The five-limb digit row of a norm: its 99 digits and the numbers made of them (V's two parts Vlo and Vhi, carries, sigma = a + 2^24 b, R, remainders), as values of type T |
| `Served` | The 18 conditions of the standard parameter form that the stage `request` tests, a field each. Two more have no field: id-named, no-proof-of-work (parameter-file.md) |
| `SMAX` | The largest absolute value of an entry of table SILU: values are clamped to +-SMAX; parameter-file line smax, 4095 if absent |
| `SMAXTOP` | The largest absolute value in table SILU, 4095, which the line smax can only lower |
| `StOwn` | What the statement is built from on the caller's side: the request's words, its tokens, r, e, y; four lines of the parameter file (found, value); and which words of the statement have been written |
| `TAU` | tau: the number of random combinations in the smallness test of a request's sub-proofs; the parameter file's line tau |
| `ledNS`, `ledNR`, `ledNP`, `ledNO`, `ledNRH` | functions: the counts that the format's pages call NS, NR, NPr, NO and NRH: a norm's sums of limb products; its digit row's digits, hidden products and end values; the digits of the output head's digit row |

### Where the Lean port's name differs

Where the Lean port has the thing, its name is mostly the C++ name. Some C++ constants in capitals are Lean definitions in small or mixed letters: the table lists each; a constant that is not listed has the same letters there. There are two differences by form. The lookup's routines `forestX` are `Forest.x` in Lean (not `forestPlan`: see the table). What the parameter file sets is a global variable here and a field of a structure (`Settings`, `PolScan`) there. The table lists the other names that are known to differ. Of the names that are alike not every pair has been compared, and many names of these sources have no counterpart in Lean and are not listed.

| C++ name | Lean's name | what it is |
|---|---|---|
| `acceptancePolicyIsExactMatch` | `exactMatch` | True if the acceptance policy's lines C, theta, cap, sh, sT are those of exact match: every checked token is the model's top choice |
| `BAR` | `bar` | see "The short names" |
| `BIGH` | `bigH` | see "The short names" |
| `CLAMP` | `clamp` | see "The short names" |
| `earlier_hash_h_digest` | `secondHashFunction` | The digest of H(num; w) of verifier/HASHES.md: the first 8 words of the state after hashing a list of 32-bit words under a domain-separation number |
| `earlier_hash_h_state` | `secondHashFunctionState` | The 16-word state of H(num; w), with l = L, the list's length in words |
| `EXC` | `exc` | see "The short names" |
| `forestPlan` | `Params.planForest` | Appends the rounds of a group's lookup to the plan p under the label `where`; hands on only v, d, 5 (2^5 bit positions of a multiplicity) and whether a shift is drawn. |
| `gateTableInputFractionBits` | `EG` | The fraction bits of a key of table SILU: the step between keys is 2^-EG; parameter-file line eg, 8 if absent |
| `GCL` | `PolScan.gcl` | see "The short names" |
| `HCallZ` | `CallZ` | see "The short names" |
| `HFork` | `Fork` | see "The short names" |
| `hParDigest` | `hPar` | A local in runStage: the digest h_par of the three configuration files, only printed. Kin, no twin: Lean's `hPar` computes the digest (here: `hParOf`) |
| `HPiece` | `Piece` | see "The short names" |
| `HStack` | `Stack` | see "The short names" |
| `KCAP` | `kcap` | see "The short names" |
| `LANES` | `lanes` | see "The short names" |
| `LBZ` | `lbz` | see "The short names" |
| `LBZW` | `lbzw` | see "The short names" |
| `LEDG` | `ledG` | see "The short names" |
| `LEDNE` | `ledNE` | see "The short names" |
| `LRAD` | `lrad` | see "The short names" |
| `POW` | `Settings.pow` | see "The short names" |
| `POWC` | `Settings.powC` | see "The short names" |
| `POWS` | `Settings.powS` | see "The short names" |
| `RADB` | `radb` | see "The short names" |
| `RCL` | `PolScan.rcl` | see "The short names" |
| `RECORDCLAUSE` | `recordClause` | The words that end the line "commitment:". The programs, their files and the format say "the record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say "the record's seeds" for its OOD seeds ([glossary](../../docs/glossary.md#record)). |
| `residualStreamLimbs` | `XL` | Limbs (base-256 digits) of a value of the residual stream: 4, or 5 with the parameter file's line "xl 5" |
| `roundName` | `RoundName.name` | Maps a kind of round (enum RoundName) to the text by which the hash-chain channel compares the round with the plan. Three kinds give "JOIN". |
| `TAU` | `Settings.tau` | see "The short names" |
| `TokenSumcheckState` | `Tsc` | The state and result of one token sumcheck (TSC): the slot of the claimed sum Cl, the slot of the first end value, the challenges, the masked sumcheck inside, the claim it leaves |
| `WEIGHT_BOUND` | `BW` | The bound 2^12 of the range -2^12 .. 2^12 - 1 of a committed weight of any kind but the embedding; shown by the key file's range proof |
| `WEIGHT_WIDTH_BITS` | `WB` | The width of a committed weight in bits, 13: every kind but the embedding; the lines wb_X must equal it |
| `statedTarget` (its member `hb`) | `Settings.hb` | the statement's word hb: the target in bits (the parameter file's line `sec`) |

## The split

What a key, an entry and a proof are **outside** the sub-proofs is implemented once, for prover and verifier, in `blocks/`: hash chains, trees, codes, samples, positions, proximity tests, the test on the committed table, the order of each file, the plan of every round and the error accounting. `verifier/src/` holds the rest: the command line and the three stages, the parsing of the parameter file's lines after the frame's lines, the model's shape and its no-overflow conditions, the three kinds of sub-proof, and the interface (below: the seam) between these and `blocks/`.

## The files

Notes on `vf_common.h` stand in [`source-notes/vf_common.h.md`](source-notes/vf_common.h.md), under the keys that its comments name. Where a comment of another file leaves a format to a page, it names the page and the place on it: a heading, a paragraph's first words, a section's number, or a step or row of a table.

### Main program and common parts

| file | contents |
|---|---|
| `cp_verify.cpp` | `main()` reads the command line and every file once, tests the heads, starts the three stages, each in a child process (`cpStagesRun`), and prints the call's last line. Each stage is one run of `runStage`, which runs only once in a process: it parses the parameter file, makes the sub-proofs' plans, has `blk::configure()` make all sizes and plans, and then runs as the stage key (`blk::check_key`), leaf (`blk::check_entry_if_tested`) or request (`blk::check_proof_if_tested`). `cpPlanMain` reads the planning calls; `cpUsage` prints the usage text |
| `cp_configured.h` | the entry point for another program (committer, prover) that links `cp_verify.cpp`, compiled as a unit: it leads from the three configuration files to a `blk::Configured`, by the same lines of code that a stage runs |
| `vf_input_files.h` | the reader of the configuration folder; the heads of `key.bin` and of a proof |
| `vf_common.h` | what does not depend on what is proved: the fields, rejections, `parFor`, the readers of files, the version word, the threshold, the conditions of the standard parameter form, status 14 and a stage's last lines, test builds, this program's own transcript object and tree hash TH, the record of one lookup call |
| `vf_accounting_record.h` | how a stage's record of its error accounting reaches the parent process as text. The accounting itself and the one routine that prints it are `blocks/numbers/accounting.h`. Its numbers are the program's own estimate and no security level |
| `vf_rejection_names.txt` | the classes of refusal names. It is not a source of the program, but it is in this folder and so enters the source digest `build=`. An entry is a whole name or the fixed beginning of names that end in a variable part. A rejection prints `NO: <name>`, under `-q` too |

### This program's own hash

This is a second hash mode of our own, built on BLAKE2b's compression function. It hashes public inputs: the public files and tables, a request's tokens with its two numbers, the statement's words. Some of these digests are only printed or compared. Others go on into a proof: nine of them enter the first query of a proof's first chain as bytes, among the statement's words. The request file's words also enter that query under the chains' hash (`blocks/hash`) itself, as an item of their own: the tokens, then the two numbers. So another request (other tokens, or another range of checked tokens) gives another first query under the chains' hash, and so does a statement that differs in any other word, or another configuration: the statement's words, all but the next token's (the next token enters the chain in a round of its own), and a digest of every byte of the three configuration files made with the chains' hash, are in the first query too (read in the sources). [../../blocks/hash/MODE.md](../../blocks/hash/MODE.md) ("2. What is assumed") says what soundness then asks of the second hash mode, and on what condition. What a proof's chains themselves hash with is `blocks/hash`. See [../HASHES.md](../HASHES.md).

| file | contents |
|---|---|
| `vf_digest_words.h` | a digest has 8 words of 32 bits |
| `vf_earlier_hash_f.h` | F, the compression function of BLAKE2b (RFC 7693) |
| `vf_earlier_hash.h` | one call of F with a domain-separation number and a length; the hash H of a list of words |
| `vf_widths_and_bounds.h` | the integer types; the widths of the hash and of a few hashed lists; the width of a committed weight and two bounds of the no-overflow conditions |

### The seam to `blocks/`

| file | contents |
|---|---|
| `vf_request_chain.h` | the pass over a request's links, as calls that a sub-proof's routine makes itself (`mb::ChainCheck`) |
| `vf_segments.inc` | from the sub-proofs' plans and the layout file to the shapes that `blk::configure()` takes (`buildLinksH`); the links of one proof (`runLinksH`) |
| `vf_transcript_channel.h`, `vf_transcript_calls.h` | the transcript as the sub-proofs' routines use it, on the channel of the proof's hash chain: every read and every draw is compared with the next planned round, and a mismatch ends the process |
| `vf_subproof_segment.h` | a sub-proof's list of arrays; the element that batches its claims; `rootInH` at each later commitment; `doorH`, which hands the chain the sub-proof's one equation |
| `vf_weights_layout.h` | the layout of W, and a sub-proof's claims on it |
| `vf_weights_merge.h`, `vf_weights_pending_claim.h` | the merge of a sub-proof's weight claims, and the row that it leaves on the committed table |

### The sub-proofs

| file | contents |
|---|---|
| `vf_layers_plan.inc`, `vf_output_head_plan.inc` | the plans of the layer sub-proofs and of the head; included inside `runStage()` |
| `vf_layers_check.inc`, `vf_output_head_check.inc` | the checks of one block of layers and of the output head; included inside `runStage()` |
| `vf_attention_check.h`, `vf_attention_shape.h` | the check of the attention sub-proof; its shape, call lists and no-overflow tests, and the index routines (`tenCell` and its neighbours) by which every sub-proof finds a tensor's entry in a shared matrix |
| `vf_subproof_engine.h` | the engine that the sub-proofs share: hidden values under mask entries, affine claims on them, the arrays that the claims read, the weighted sum of all claims |
| `vf_sumcheck_psc.h`, `vf_sumcheck_receive.h`, `vf_sumcheck_token.h`, `vf_sumcheck_attention.h` | the masked sum-check (PSC); the file through whose routine its rounds must be received (all but the round in Y); the token sum-check; the attention sub-proof's two kinds of sum-check |
| `vf_lookup_helpers.h`, `vf_lookup_argument.h` | the look-ups on hidden values; `blocks/iop/lkp.h` states the argument |
| `vf_norm_digit_rows.h` | the row of byte digits of a norm when a value of the residual stream has five limbs (line `xl 5`): the row, the multipliers with which a lookup checks the range of its digits, and its 9 + 8 linear equations. The sources call the row the "ledger" of a norm; it has nothing to do with soundness errors |
| `vf_smallness_test.h`, `vf_smallness_test_members.inc`, `vf_smallness_matrices.h`, `vf_smallness_packing.h` | the smallness test, which shows that every committed cell is a small integer; the records of its matrices; the placement record of a matrix that one sub-proof commits to and another reads |
| `prover/prover/plan_rounds.h`, `prover/attn/plan_rounds.h`, `prover/head/plan_rounds.h` | not in this folder: the rounds of each kind of sub-proof as the proof's plan lists them, one source for prover and verifier |

### For tests alone

| file | contents |
|---|---|
| `../tests/coefficient_vectors.h`, `../tests/claims_report.h` | no part of `cp_verify`: two headers that checkers under `verifier/tests/` include: the verifier's coefficient vectors compared with a prover's test files; which claims of a sub-proof are false on a prover's test data |

A build with one of the library's switches for a test (`blocks/fence.h`; `vf_common.h`, `TEST_BUILD`) never ends a run with an accepting word, and does not compile under `-DSERVING_BUILD`, which `build.sh` passes. That holds of a switch set on the command line or before `blocks/base.h` is first included: `blocks/fence.h`'s own limit.
