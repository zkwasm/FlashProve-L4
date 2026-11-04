# hash/: BLAKE2b on labelled strings

All public data that blocks/ hashes goes through this folder, on the host and on a GPU. Every hash is BLAKE2b as specified (sequential, no key) on a string that begins with a label of 16 bytes. Where a use has a start, which is a digest, that digest is BLAKE2b's salt and personalization. A digest has 256 bits. A prover's private randomness does not come from this folder; the made-up coins of tests and toy runs (hashes of a seed and a counter) do, under a domain-separation number of their own. The rule, to the byte, is in docs/format/FORMAT.md, rules R2 and R3 (its section 3).

| file | what |
|---|---|
| `tweak.h` | the macros for code that is compiled for host and GPU; the counter and the flag of a call of the compression function |
| `blake2b_f.h` | BLAKE2b's compression function; its first chaining value, counter and flag |
| `domains.h` | the domain-separation numbers (160 and up), and which of them have a start |
| `hash.h` | the interface |
| `MODE.md` | for a reader who checks soundness and hiding: what is assumed and on what grounds, costs, why leaf, salt and round are laid out as they are, what it does not decide |
| `work_search.cuh` | the search for a round's proof-of-work nonce on a GPU |
| `lanes.cuh` | one compression, and one call of a round, computed by four lanes of a warp; for a chain that runs inside a kernel |
| `older.h` | a second hash mode of our own, built on BLAKE2b's compression function, byte for byte. In blocks/ only its own tests call it: `hash.h` hashes the weight table's tree (numbers 174 and 166) and the digests that name its files. MODE.md says what rests on that mode |

## The calls

| call | what it computes | compressions |
|---|---|---|
| `Tag(domain, tree)`; for a tree's vertices `Tag(domain, 0, start)`; for its leaves `Tag(domain, tree, start, symbols, salt bytes)` | the tag of a hash; a `Domain` converts to one implicitly | 0 |
| `Hasher(tag, index)`, `Hasher(tag, index, start)`, `push`, `finish` | the hash of any bytes | `hash_calls(bytes)` = ceil((16 + bytes) / 128) |
| `vertex(tag, left, right)`; `node(tag, index, left, right)`, `run_value(label, parameters, random bytes)` | a vertex of a tree: its two children, from a digest of the public parameters; the hash of two digests under another number | 1 |
| `leaf(tag, index, symbols, count, salt, salt bytes)`, `leaf(tag, index, digest, salt, salt bytes)`; on whole blocks `leaf_begin`, `leaf_step`, `leaf_last` | a leaf: a column's symbols, then its salt, whole in the last block (`leaf_salt_at`, `leaf_bytes`, which count the label) | ceil((16 + 8 count + salt bytes) / 128) |
| `ChainAbsorb(rho, round)`, `push`, `prefix()` | what the calls of a round share: the message, whose last word of 8 bytes leaves its bits from 16 up free | the message's whole blocks |
| `chain_call(prefix, nonce, j)`, `chain_call_cells` | call j of the round: 64 bytes, two blocks of its answer (`chain_block`) | 1; 2 if the message's last word ends a block |
| `work_test(word, bits)`, `chain_work_passes(prefix, nonce, bits)` | whether a word's low bits are 0; the test of a round's work on call 0 | 0; as call 0 |
| `test_coins(seed, counter)` | 32 made-up bytes | 1 |

A block has 128 bytes, which are 16 words of 8 bytes (`SLAB_SYMBOLS`); a chaining value has 64 bytes. The first block of every string begins with the label's two words, so it holds 14 symbols of a leaf. If the prover's message in a round has up to 96 bytes, each call of that round is one compression, and none comes before the calls.

## Tests

`make -C blocks OUT="$D" test`, with `D=$(mktemp -d)`, builds and runs all tests of blocks/ (89), this folder's `tests/*.cpp` among them. The comment at the top of blocks/Makefile says how to run chosen tests only (`TESTS=`). This folder's tests:

- `functions_test.cpp`: RFC 7693's value for "abc", and digests from Python's hashlib at several lengths, with and without a key;
- `mode_test.cpp`: compares the calls with a second implementation of BLAKE2b's pass over the blocks and of its parameter block (that implementation uses the library's own F and IV), and with the known answers in `mode_vectors.txt`, each line of which is one call of Python's hashlib (`mode_vectors.py`); inputs that must give different digests; the work test; calls out of bounds. Not tested: the bound on a hash's length (it takes 4 GB of input);
- `calls_test.cpp`: counts the host's calls of the compression function and compares them with the column "compressions" above; eight calls of that table it never runs (the comment at its top names them);
- `older_test.cpp`: the second hash mode's vector files (lists, tree hash, leaves, nodes), and `table_tree_vectors.txt`, leaves and nodes of the weight table's tree under that mode, computed by host code that is not in this repository.

No test in this repository compares `work_search.cuh` or `lanes.cuh` with the host code.

A call out of bounds ends the process with an error message. On a GPU it ends the kernel without one (a later CUDA call reports an unspecified launch failure): make the same call on the host to read the error message.
