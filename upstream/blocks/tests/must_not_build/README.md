# Sources that must not build

Each source file here is one that the compiler must reject. Most of them try to get round the two types of `numbers/plan.h`, which decide who may add a round to a plan and who may change what has been added. Eight try to get round the types of `iop/foldeq.h` that say whether a committed table has passed its test.

The line `// MUST NOT BUILD ..` at the top of a file gives the words that the compiler's message must contain, so that a file does not count as rejected because of a typing mistake. It also gives any further compiler option for the build. With `-DCONTROL` the same file builds. What the control changes depends on the file: it defines a test-only macro that opens the type (`BLK_PLANS_BY_HAND_FOR_A_TEST`), where that is the difference; it omits the option that the top line names, where it names one; otherwise it leaves out the offending lines.

`tests/new/run_new.py`, stage `fences`, compiles every file both ways. No Makefile target builds them.

A file with several offending calls is rejected as long as one of them is. `numbers/tests/plan_types_test.cpp` tests each call by itself.

The eight files on a tested table are `a_tested_table_made_by_hand`, `a_tested_tables_distances_written`, `an_untested_reader_for_check_proof`, `an_untested_reader_for_check_entry`, `an_unproved_table_for_check_proof`, `an_unproved_table_turned_into_a_tested_one`, `a_result_if_tested_taken_for_a_verdict` and `a_result_if_tested_taken_for_a_bool`. In the routines that the checks of whole files call, the same mistakes end the process at run time; `iop/tests/lin_test.cpp` and `link_test.cpp` have a case for each.

Two pass-keys, objects that only named routines can construct, have no such source in this repository: the pass-key for `switch_admit()` (`iop/cs.h`), and `LinBegun`, which says that a batch's columns were counted against what the table's random tails cover (`iop/lin.h`). An empty class whose private constructor is defaulted at its declaration is an aggregate under C++17, so empty braces would construct one.

Seven files are about something else. Six are named `a_tests_word_said_late_..`. `fence.h` checks a list of the tests' macros, but it does not see a macro that a source defines after the first include of `fence.h`. For the six test-only macros of `blocks/` that these files define late, the header that reads the macro expands the fence's marker itself, so the unit does not compile with `SERVING_BUILD`. The header that reads the plans' macro does the same; its file is `the_word_behind_base_h`. Not guarded in this way: the macros that `prover/engine/` reads. The seventh file is `a_guards_hook_that_may_throw`: the routine that a program installs in `blk::at_a_failed_guard` (`base.h`) cannot be one that may throw.

Six files are further attempts on the plan's types: `copy_alter_enter_assign`, `enter_under_another_setting`, `the_door_by_a_pointer_to_member`, `append_and_go_on`, `a_reference_kept_from_close` and `a_whole_plan_by_hand`. Two further cases are tested at run time: a draw of positions whose error bound is set to 0 (`numbers/tests/numbers_test.cpp`), and a plan that is changed after something has been made from it: another plan is assigned to the variable after a `Cursor` was made on it (`plan_types_test.cpp`), or a round is added after a `Binder` was made on it (`chain/tests/public_test.cpp`).

No row of SUITE.md runs the stage `fences`. Nothing in this repository records how many sources are in this folder, so a file that is lost is not missed.
