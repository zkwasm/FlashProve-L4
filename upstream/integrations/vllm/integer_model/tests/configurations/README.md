# Configurations for the tests

`tiny_without_clamps/` holds the parameter file (`policy.txt`) and the quantization scales
(`KL.bin`) of a model that runs in milliseconds: 4 layers, width 64, 8 experts, a vocabulary of 200
tokens and windows of 16 tokens. The file has none of the lines `rcl`, `gcl` and `xl`, so the
router's logits and the gate's keys are not clamped, and a value of the residual stream has four
limbs ([docs/model.md](../../../../../docs/model.md), "10 Parameters that change the function").
Every configuration under `prover/examples/` has all three lines.

The configuration is for tests alone: this package's, and two others in this repository that read
`policy.txt` as a parameter file (`scales/tests/test_policy_tables.py`,
`verifier/tests/forest_lines.py`). `cp_commit` and `cp_prove` refuse it, because they need `D 2048`
and a head dimension of at least 32. Its lines on the proof system, from `version` to `table`, are
those of `prover/examples/target_small_f2/policy.txt` with other values for `slots` and `table`; this
package checks their form, and the value of the four lines that take one value.

`KL.bin` is, byte for byte, the file of a small example of this project that is not in this
repository. The lines of `policy.txt` from the line `prof` onward come from that example's file, and
differ from it in four things. `tau` is 128, following the line `sec 128`. `s` is 16, a round number
that the file simply states: no equation ties it to the model, as `verifier/docs/parameter-file.md`
says. The 32 words of the tables' digests are computed for this file. Lines that the format does not
have are absent. It is not known who chose each shift count, exponent of a scale or threshold in the
two files, or how. `THIRD_PARTY.md` at the top of the tree ("What depends on Meta's models") lists
which of Meta's models some of them and `KL.bin` are computed from or may depend on. By its kind
none of these numbers is a weight of a trained model. Whether ranges of scales that were fitted to a
trained model had a part in the fields of `KL.bin` is not known. The tests generate their weights
from the seed in `policy.txt`.
