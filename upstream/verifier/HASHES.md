# The verifier's own hash: F, C, H and TH, and what `cp_verify` hashes with them

## What this hash is for

`cp_verify` ("the program") holds two hashes.

- **What a proof's chains and trees hash is not defined here.** It is BLAKE2b as specified, on labelled strings: `blocks/hash/` (its `README.md` and `MODE.md`), to the byte in `docs/format/FORMAT.md`, sections 2 and 3. The commitment ID (printed in the line `commitment ID: `; the value that `-id` takes) is `blk::published_id` in `blocks/frame/heads.h`, and the digest of the three configuration files that every chain binds (`configuration=`) and the seal of a request's first chain (`dj=`) are the blocks' too.
- **The program's own hash**, which this page defines, is a second hash mode of our own, built on BLAKE2b's compression function: `verifier/src/vf_earlier_hash_f.h` (F), `vf_earlier_hash.h` (C, H) and `vf_common.h` (TH). In a run of the program it makes only the digests of this table.

| Digest | Of what | Function | What is done with it |
|---|---|---|---|
| in the statement: the request's words, `KL.bin` | the T + 2 words of the request file; all words of `KL.bin` | TH | words of the statement |
| in the statement: four public tables (SiLU; sigmoid, for a model with experts, else eight zero words; RoPE (rotary); exponential (for the softmax)) | each table's entries, one word each, as `cp_verify.cpp` lays them | H under 72 | words of the statement; compared with the parameter file's lines `tsilu0` .. `texp7` if the file has them (`table-digest-silu`, `-sigmoid`, `-rope`, `-exp`) |
| in the statement: the lists of look-up calls (layers, head, attention) | the text of the lines `call ..`, `callH ..` and `callA ..`, one list each, four bytes to a word, the first byte lowest | H under 73 | words of the statement |

The program builds the statement itself, from its caller's files, and never reads one. The statement's bytes, with the word of y as 0, are part of the first query of a proof's first chain, so the digests in it are bound by the blocks' hash like its other words. The request file itself, its T + 2 words byte for byte, is a further item of that query (`the request file's words`): so every challenge of that chain hangs on the tokens and the two numbers r and e through the blocks' hash alone, whatever the digest of them in the statement is worth (../docs/format/10-proof-file.md, R14.3, the item of that name; ../docs/format/FORMAT.md, R15.3). The program's own hash is used there as the fixed, public function that it is, not as a second random oracle; that it does not disturb the blocks' hash's standing as a random oracle is a weak assumption about the pair (../docs/format/FORMAT.md, R15.3, 12.6). The program computes the four tables with the platform's floating-point library; the parameter file's digests are there so that which function is proved does not depend on the last bit of a library's result. What then hangs on the program's own hash: ../docs/format/FORMAT.md, R15.3, and ../blocks/hash/MODE.md, "2. What is assumed".

The three configuration files are the parameter file (`policy.txt`), the quantization scales (`KL.bin`) and the layout file (`layout.txt`).

## Notation

A **word** has 32 bits, a **long word** 64. Two words make a long word, the low word first. Lists are indexed from 0; "‖" joins lists. A **state** has 16 words: what a call returns and what a further call takes as its chaining value. A **digest** has 8 words, the first eight of a state: what is printed or put into a list. A **block** has 32 words. The width of a digest is a constant of the build (`vf_digest_words.h`), never a value of a run or a line of a file.

Every call takes a **domain-separation number**, a word that keeps the uses of the hash apart. "H under 10" is H with the number 10. The sources also call it `num`, tag or flag; on this page "the flag" is always the input "last" of F.

## F and C

**F** stands for BLAKE2b's compression function. It is defined in section 3.2 of RFC 7693 ("The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)", M-J. Saarinen (ed.) and J-P. Aumasson, November 2015); the tables IV and SIGMA are those of that document's sections 2.6 and 2.7. It has twelve rounds, and its four rotation amounts are 32, 24, 16 and 63. On this page it is written F(h, m, t0, t1, last): h is the chaining value, eight long words; m is one block, sixteen long words; t0 and t1 are the two counter words, lower and upper; last is the flag that marks a final block. The code for F in `vf_earlier_hash_f.h` is adapted from the C reference implementation that the BLAKE2 team publishes; a comment at the top of that file lists what was taken over and what was changed.

**C**(cv, m, c, l, f, last) = F(cv, m, c, f + 2^32 · l, last), with the state cv, the block m and the result read as long words. c is a counter of 64 bits; l is a length; f is the call's number. Every number that the programs use is below 2^16.

**IV'** is the state from which BLAKE2b-512 starts, IV with its first long word xor 0000000001010040. As words:

```
f2bdc948 6a09e667 84caa73b bb67ae85 fe94f82b 3c6ef372 5f1d36f1 a54ff53a
ade682d1 510e527f 2b3e6c1f 9b05688c fb41bd6b 1f83d9ab 137e2179 5be0cd19
```

A **bare call** is C(cv, m, c0 + 2^32 · c1, 0, f, not last). Number 3 is used both for a bare call and for H, in tests and in the worked values below. Such uses are kept apart only by the other inputs of the call.

## H(tag; x)

x is a list of L words, 0 ≤ L < 2^32.

1. cv = IV'.
2. Split x into blocks of 32 words; at least one block; missing entries are 0.
3. For every block but the last: cv = C(cv, block k, 128 · (k + 1), L, tag, not last). For the last: cv = C(cv, the block, 4 · L, L, tag, last).
4. The last cv is the state of H; its first eight words are the digest. "H(tag; x)" means the digest.

For a reader who knows BLAKE2b:

- First state, zero padding, the one block of an empty input, the counter (bytes fed so far) and the flag on the last call are those of BLAKE2b-512 (RFC 7693, section 3.3, nn = 64, no key).
- The one difference: BLAKE2b leaves the high half of its counter at 0. Here it holds the tag (low word) and L (high word), in every call. So H under any tag is not BLAKE2b-512, and no library's `blake2b(...)` computes it. With that half forced to 0 the same steps are BLAKE2b-512 of the list's bytes, each word's lowest byte first: a good first test of an implementation.
- A digest is a truncation of the 64-byte result. It is not BLAKE2b with a shorter output, which starts from another state.
- L is the length of the whole list and the same in every call. Only L and the counter tell a list from the same list followed by zeros. 4 · L can exceed 2^32: the counter has 64 bits.

## TH(x), the tree hash

x is a list of at least eight words. (The bound is the program's, not the construction's; a shorter list is refused as `internal-th`.)

1. Split x into pieces of 128 words; the last may be shorter. Replace each piece by H(10; piece), with L that piece's length. Join the digests in order.
2. While more than eight words are left, do the same to them under 11: a piece is then sixteen digests.
3. The eight words that are left are TH(x).

So every list is hashed at least once: TH of eight words is H(10; x), not x. 10 is used for pieces of the list itself and for nothing else, 11 for pieces of digests and for nothing else, and H feeds its number and its list's length into every compression. Hence a collision of TH, between lists of whatever lengths, yields a collision of H under 10 or 11.

## The numbers that the sources pass

| Number | Kind of call | Use | In a run of the program |
|---|---|---|---|
| 10, 11 | H | TH: the first level, every level above | yes |
| 72, 73 | H | a public table; a list of look-up calls | yes |
| 127 | H | the test of a nonce: H(127; the state's digest ‖ the nonce's two words); the low g bits of word 0 must be 0 | no: no program's source passes it (one test does, on known answers: `blocks/hash/tests/older_test.cpp`) |

The sources hold further numbers that no program passes: 103 to 105 and 133 (`prover/hiding/long_rows_words.h`) and 130 to 132 (`blocks/hash/older.h`). Only tests and tools use them. `prover/hiding/coins_own.cuh` passes 0, which is plain BLAKE2b-512.

The numbers reserved for this hash are 3 to 16, 64 to 67, 72, 73, 75, 83, 84, 90 to 110, 112, 113, 120 to 127, 130 to 139, 240 and 241. Give none of them a new use, and search the sources before you take a new number. The blocks' own numbers run from 160 to below 240, between these (`blocks/hash/domains.h`). Test programs and `hash_vectors.txt` also call H and C under other numbers, on synthetic lists (a bare call under 3, H under 107 and, once, under 2147483651); none of that enters a proof.

## Files and digits

The request file and the y file (the third and fourth arguments of a call on a proof; in `-plan request` the files after `-prompt` and `-reply`) are sequences of 32-bit little-endian words: the bytes 04 03 02 01 are the word 0x01020304. An empty file, or one whose length is no multiple of 4, is refused (`statement-required`).

A digest is printed as its eight words in order, each as eight lower-case hexadecimal digits, most significant first (C's `%08x`): 64 digits. That is not the order in which a BLAKE2b library prints its bytes.

## h_par

h_par is no digest of this hash. It is the digest of the three configuration files that every chain binds (`blk::configuration_digest` in `blocks/frame/heads.h`), printed word by word, so its digits differ from those of the line `configuration=`. It is sent nowhere, and no verifier compares it with anything; cp_commit writes it into isk.txt's line `ranges` when its worst-case check has passed, and cp_prove compares that line with its own digest of the three files that it has read, and refuses to start if they differ or the line is absent (`ranges-not-guarded`; under `seed test`, in test builds, it does not ask for the line). What ties a proof to the files, byte for byte, is `configuration=`.

## Worked values

Words are hexadecimal. The lists are chosen to show one compression and two levels. Their first word, 0x4D425048, is a constant that spells "MBPH" in ASCII, and nothing reads it here.

**One compression:** TH of the eleven words below: one piece, one block: H(10; x) with L = 11.

```
x    4d425048 0000000b 0000000b 00000016 00000021 0000002c 00000037 00000042 0000004d 00000058 00000063
c = 44 (4 L, the last call)   l = 11   f = 10   the flag is set:   t0 = 000000000000002c   t1 = 0000000b0000000a
state  0ed406bd d49186df 14e2a071 4541176c 73b6fbf2 b22b8a14 94e626f2 d2fb95a6
       32dccade c899de38 64636296 fa2c0b4a 3924b896 82e069b8 f255022a 5ccba6bc
TH = 0ed406bdd49186df14e2a0714541176c73b6fbf2b22b8a1494e626f2d2fb95a6
```

**Two levels:** TH of 202 words: 0x4D425048, 202, then 1, 2, .. 200: pieces of 128 and 74 words, then one H under 11 over the 16 words.

```
H(10; x[0..127])    9e0af6e5b842f2a992ad18d909b09e1e86eed911e9787c001d14a4b940bb70d9
H(10; x[128..201])  3fb30f532605ec037b4c40d1227bd4e8c5708c0793b3e6ce8290043eb3c78c6d
TH = 2025410fdf19dd1f53d64a938171ebddc2da8a19771dbcb6b7ed90e77d071744
```

"1..n" is the list 1, 2, .. n.

| What | Value |
|---|---|
| F(IV', m, 3, 0, last), m's first long word 0000000000636261, the rest 0: BLAKE2b-512("abc"), RFC 7693, Appendix A, printed as a library prints it, byte by byte * | `ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923` |
| C(IV', the block 0, 1, .. 31, c0 = 1, c1 = 2, number 3): a bare call's state * | `f5727365c6b1178ebb489e21a739d5edde6eb7157ce3c623ac9f86185c77d0d724e40b1b5c884aceaea66bf5a7d65570f06c1b009e7a6b62ada026b4cbb7ec9d` |
| H(10; the empty list) * | `642033eebd8188a6b2b6dba3ad4e81de70b03c7d9bc86eec189b6da891098fc0` |
| H(10; the one word 7) | `9199d8e76712b9d5a45ca6ffa7333389d9332b0a830f4274cc71888ff4dd0629` |
| H(10; the two words 7, 0) | `5270ece847c0c6e6215c895e003b513dc694c460d5091150f1a79692658fcb04` |
| H(3; 1..32): one full block | `3d11700a6c1b3f588c605790c666049f2a3b0711fb1cdb03fc8cba0a088f932b` |
| H(3; 1..33): a second block of one word | `63879cbc0f65a48f466bbb5507fb9c4244a169c026ffae47fb3f0fa72bd88212` |
| TH(1..8) = H(10; 1..8) | `2c9ad89066617790478f356ad1401491956327cfb43ec51194e7ad7eb36bdbba` |
| TH(1..136): pieces of 128 and 8 words, then 16 words under 11 | `b790a9d3eb4b5173034538aa7d2ddd3ab7359b67c77b274469f8de7e5f0044b2` |
| TH(1..2049): 17 pieces, then 136 words and then 16, both under 11 | `3323f7b96604ef89cbe4e3b4ae6641bb73ccc22b581b898b679d62882b68a13e` |

The steps of H do not give the row of "abc", even with the counter's high half forced to 0: a list of words always has a multiple of 4 bytes. The row tests F and IV' alone.

## What checks these values

- The rows marked * are checked against cp_verify's own headers by a test that is not part of this repository.
- `verifier/tests/known_answers/format_hash.py`, on the F of `blake2b_f.py` beside it, is a writing of C, H and TH in Python that imports no code of the program (its F has the same outside source as the program's: the BLAKE2 team's reference implementation); it prints `hash_vectors.txt`, and the suite's row `verifier.format_hash` has it make all 57 lines again and compare. `blocks/hash/older.h` is a second writing of H, TH and the bare call in C++, and its test, `blocks/hash/tests/older_test.cpp`, reads the same file.
- Three more rows (H(10; the one word 7), H(10; the two words 7, 0) and H(3; 1..32)), and the row of the empty list again, are known answers in `scales/tests/test_policy_tables.py`. That test checks `scales/policy_tables.py`, a separate writing of H in Python, and not the program.
- **The other four rows are checked by no test in this repository.** Every value on this page comes from this page's rules alone, by a separate short Python program.
