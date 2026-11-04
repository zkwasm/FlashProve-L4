# frame/: the format's frame

This folder leads from a parameter file to the plans (`Plan`) of the three files that a verifier reads. `FORMAT.md` describes the format in words.

| file | what |
|---|---|
| `version.h` | the one version word: format, field, hash |
| `parameters.h` | the frame's lines of the parameter file, in a struct; the one strict parser, for prover and verifier |
| `derive.h` | the numbers that have no line of their own: distance, positions, random coordinates and stages, for a request's links and for the committed table; the solvers that `lin_frame.h`, `link_frame.h` and `wf_frame.h` use |
| `req_frame.h` | from the links' arrays and a layout file to the shape of a request |
| `lin_frame.h` | the groups, codes and plans of the test on the committed table |
| `link_frame.h` | the same for the link between two stored copies of the table |
| `files.h` | `sized()`: all of the above for one parameter file; `file_key`, `file_entry`, `file_proof`: a finished plan for each hash chain; `merged()`: one plan for printing, which `configure()` also uses to reject a file and for the accounting of work |
| `heads.h` | the heads of the files; the digest of the configuration; the labels of the frame's chains |
| `configure.h` | `configure(files, shapes)`: the one routine that leads from an instance's public files to the digest that every chain binds, and to everything that is derived from these files (sizes, the plans of key, entry and proof, the fixed values). No other code can make its result. The code that makes a proof and the code that checks one take plans, codes, shapes and the digest from that result and from nowhere else |
| `check.h` | the order of a whole proof file on the blocks' verifiers, without a model: `check_proof()`, with three verdicts. Also the one place that builds the first queries of a proof's three chains |
| `check_key.h`, `check_entry.h` | the same for a key and for an entry, each with the routine that makes the file |
| `wf_frame.h`, `wf_units.h` | the numbers of the committer's proof of the ranges; what it looks up, from a layout's units |
| `hiding_conditions.h` | what hiding requires of a parameter file: it compares the derived sizes with a separate count (from the plans' draws of positions and the shapes); a serving program calls it before anything that depends on a secret is shown or stored |
| `prints.h` | what a planning call prints. The word «link» has two senses in what is printed: in `group: link` and `FOLD round (link #)` it is one link of a request's chain, which the paper calls a segment; in `in a link`, `folded tables of a link` and `LINK (link)` it is the link between two stored copies of the table, the paper's link. The kinds of rounds are listed in `numbers/round_names.h`; the version word (H821) is explained in `version.h` |
| `fields.h` | every field of a file with its place, length and class |
| `cp_frame.cpp` | the planning calls: `cp_frame -plan id\|key\|leaf\|request FILE [-rounds] [-fields]` (`leaf`: an entry). **A good planning call ends with exit status 4, not 0**, since no proof was checked. The statuses: 4, planned, and every round holds at the file's target (`-plan id` prints the version word and the digest alone, looks at no round and ends with 4 too); 6, planned, and some round does not hold; 7, planned, every round holds, but the file misses a condition of hiding; 1, the file is rejected or cannot be read; 2, wrong usage. (`numbers/cp_plan` ends a good call with 0.) |
| `examples/` | one parameter file, `record.f2.txt` (the table's code has length 2^32): the first 32 lines of `prover/examples/target_full_f2/policy.txt`. It has counted proof of work on the rounds that draw positions: three CPU tests here only size and plan this file, and `cp_commit`, `cp_prove` and `cp_verify` cannot take it, because it has no lines on a model; nothing in the programs refuses counted proof of work. The programs, their files and the format say "the record" for a table's published out-of-domain (OOD) answers (masked sums of the weights). They say "the record's seeds" for its OOD seeds ([glossary](../../docs/glossary.md#record)). |
| `tests/` | `frame_test`: parser, heads, derived numbers, plans, fields. `check_test`: whole proof files at toy lengths and the real target, from a parameter file to `check_proof()`. `hiding_conditions_test`. The four `*_frame_test` (`lin`, `link`, `req`, `wf`): runs of single parts on these plans. |

The example holds the frame's lines alone. A whole parameter file goes on with the statement's lines and the sub-proofs' lines.
