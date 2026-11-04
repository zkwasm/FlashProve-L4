# Small parameter sets that tests pin

The three folders in the table below are inputs of tests in this repository, not examples. Many tests name a place in a table, a digit, a count of values or the number of a round, and such numbers depend on the parameter file. So these files change
only together with the tests that name such numbers.

| folder | what it is |
|---|---|
| `pinned` | a small model's parameter file: the first 32 lines, which set the proof system, are those of the default setting's small form (`prover/examples/target_small_f2/policy.txt`: proof of work 20, 28, 32); `s 15` |
| `pinned_no_proof_of_work` | the same file with the line `work positions 0 0 0`: a setting for tests alone, in which no program searches for a nonce |
| `pinned_no_proof_of_work_roomy` | `pinned_no_proof_of_work` with roomier scales (`KL.bin`; the line `MF 43` for `MF 44`), on which `cp_commit`'s worst-case check passes, so that it can be committed under a drawn seed |

No test kills or signals a program at a parameter file that asks for proof of work: the search for a nonce is one GPU kernel that ends only when a nonce passes or after 2^63 - 1 tries, and what is left behind when the process is killed inside it has not been tried.

All three have 15 in the line `s`. Where that number comes from is told in `THIRD_PARTY.md`, under "What depends on Meta's models".

None of them is the default setting's small form itself: `pinned` differs from `prover/examples/target_small_f2` in one line, `s 15` for `s 16`. Tests with cheating and fault-injected copies of the programs (not in this repository), and the verifier's tests on real proofs, run at these files, with some exceptions (docs/limits-all-rows.tsv, row SO-15.5). So no test of the test table shows a cheating prover or committer refused at the very files of the default's small form. The verifier's tests on real proofs (the suite's rows that need `setup.target.other_parameters_worlds` or its twin `.work`) run at `pinned` and `pinned_no_proof_of_work`, on proofs that a test build makes there under a test seed on synthetic weights.
