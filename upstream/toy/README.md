# toy/

Two small files that tests use. You do not need either of them to build or run the programs.

| file | what it is for | how to run it |
|---|---|---|
| `quick_start_test.py` | Runs the GPU route of [docs/quickstart.md](../docs/quickstart.md#3-gpu-a-small-model) as printed, except for two lines (see the script's head), in a fresh copy of the repository, and checks that it ends as the page says. It needs an H100 and the CUDA compiler. Most of its time goes to the build. | `W=$(mktemp -d) && python3 -B toy/quick_start_test.py --gpu "$W/play"` |
| `mask_rules.py` | A separate Python implementation of the generator for the weight table's masks and salts, built on the ChaCha20 block function (RFC 8439, section 2.3). It shares no code with the prover. Run by itself, it checks the RFC's test vector and its own list of known answers. Nothing imports it, but `prover/tests/masking_cells_test.cpp` reads these known answers from it as text. | `python3 -B toy/mask_rules.py` |

`SUITE.md` lists the tests that use these files: `gpu.quick_start`, `toy.mask_rules` and `prover.masking_cells_test`.
