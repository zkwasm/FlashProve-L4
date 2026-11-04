# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A pytest plugin that writes the models and prompts of tests/test_arithmetic_edges.py as test
vectors (README.md beside this file). run.py uses it:

    VECTORS_OUT=OUT/vectors pytest -p edge_cases_plugin tests/test_arithmetic_edges.py -k five_limbs

The tests run as they are, and the plugin changes no test's outcome. It wraps three functions that
the tests call, lets each do its work, and then writes a case with the same parameters, weights and
tokens: for each window that a test compares with the recorded values, for each window that a test
shows to fail, and for each row that test_norm_at_the_ends_of_its_ranges feeds. It passes over a
model without the parameter lines rcl 1, gcl 1 and xl 5. A fault of its own while it writes is kept
and printed at the end of the session, not raised. Before any test it stops, with a message, unless
VECTORS_OUT is a directory as run.py makes it.

How it finds the kind of a failing window (README.md, "expected"): it runs the package once more
with the interval -32,768 .. 32,639 widened to -32,768 .. 32,767, the column "defined" of
docs/model.md, section 9. If the package then gives a token, the kind is "package refuses". If it
still refuses at a condition other than 8, the kind is "none". Condition 8 is decided in
make_vectors.make_case.
"""

import os
import pathlib
import re
import stat

import pytest
import torch

import chosen_model
import integer_model.model as package_model
import make_vectors
from integer_model import RangeConditionFailed


def the_folder_that_run_py_made() -> pathlib.Path:
    """VECTORS_OUT, if it looks like run.py's: it holds make_vectors' directory `cases`, and it lies
    in a directory of the caller's own that is closed to group and others, as tempfile.mkdtemp makes
    one. So no other account, root aside, can have put anything there. The directories above are
    conftest.py's to look at."""
    out = pathlib.Path(os.environ["VECTORS_OUT"])
    try:
        above = out.parent.lstat()
        own = stat.S_ISDIR(above.st_mode) and above.st_uid == os.geteuid()
        fine = own and (above.st_mode & 0o077) == 0 and (out / "cases").is_dir()
    except OSError:
        fine = False
    if not fine:
        raise SystemExit(
            f'VECTORS_OUT is {out}: it must hold a directory "cases" and lie in a directory of '
            "your own that is closed to group and others, as the one that run.py makes"
        )
    return out


OUT = the_folder_that_run_py_made()
state = {"test": "", "count": 0, "cases": [], "writing": False, "norm": {}, "faults": [], "over": 0}
plain_extend = package_model.IntegerModel.extend
plain_assert_equals_recorded = chosen_model.ChosenModel.assert_equals_recorded


def has_the_three_lines(p) -> bool:
    return bool(p.router_clamp and p.gate_clamp and p.NX == 5)


def never_raises(write):
    def guarded(*given):
        state["writing"] = True
        try:
            write(*given)
        except (Exception, SystemExit) as fault:  # make_vectors stops with SystemExit
            state["faults"].append(f"{state['test']}: {type(fault).__name__}: {fault}")
        finally:
            state["writing"] = False

    return guarded


def write_case(what: str, p, weights, runs: list) -> None:
    state["count"] += 1
    name = f"edge.{state['test']}.{state['count']:03d}"
    what = f"{what} of {state['test']} (tests/test_arithmetic_edges.py)"
    state["cases"].append(make_vectors.make_case(OUT, name, what, p, weights, runs))


def failing_window(chosen, rows) -> make_vectors.Run:
    window = [int(t) for t in chosen.window]
    narrow = package_model._TWO_LIMBS
    package_model._TWO_LIMBS = (-32768, 32767)
    try:
        model = chosen.model(rows)
        sequence = model.new_sequence()
        try:
            token = int(plain_extend(model, sequence, chosen.window)[-1])
            return make_vectors.Run(window, "package refuses", token)
        except RangeConditionFailed as failure:
            if failure.condition == 8:
                return make_vectors.Run(window, "package refuses or none")
            return make_vectors.Run(window, "none", undefined_from=failure.position)
        finally:
            model.free(sequence)
    finally:
        package_model._TWO_LIMBS = narrow


@never_raises
def write_window(chosen, rows, fails: bool) -> None:
    if not has_the_three_lines(chosen.parameters):
        state["over"] += 1
        return
    run = (
        failing_window(chosen, rows) if fails else make_vectors.Run([int(t) for t in chosen.window])
    )
    write_case("a window", chosen.parameters, chosen.model(rows).weights, [run])


def assert_equals_recorded_and_write(self, rows):
    found = plain_assert_equals_recorded(self, rows)
    write_window(self, rows, False)
    return found


def extend_and_note(self, seq, token_ids, *more, **named):
    """test_norm_at_the_ends_of_its_ranges feeds each chosen row as the one token of a new sequence.
    These calls are noted, and at the test's end each model becomes one case."""
    if state["test"].startswith("norm_at_the_ends") and not state["writing"] and seq.length == 0:
        noted = state["norm"].setdefault(id(self.weights), (self.parameters, self.weights, []))
        noted[2].append(tuple(int(t) for t in token_ids))
    return plain_extend(self, seq, token_ids, *more, **named)


@never_raises
def write_the_norm_tests_rows(noted: dict) -> None:
    for p, weights, prompts in noted.values():
        if not has_the_three_lines(p):
            state["over"] += 1
            continue
        runs, model = [], package_model.IntegerModel(p, weights, "cpu")
        for prompt in dict.fromkeys(prompts):
            sequence = model.new_sequence()
            try:
                plain_extend(model, sequence, list(prompt))
                runs.append(make_vectors.Run(list(prompt)))
            except RangeConditionFailed as failure:
                # All weights are 0 here, so every norm reads the token's own row. Where only the
                # package's condition 8 fails, the model is defined if condition 1 holds of that
                # row. Condition 2 holds under every parameter file that the package accepts, by
                # the test's own argument.
                assert failure.condition in (1, 8), failure.condition
                row = [int(x) for x in weights.embedding_rows(torch.tensor(list(prompt)))[0]]
                inside = sum(x * x for x in row) + max(p.eps1, p.eps2, p.epsF) < 1 << 78
                defined = failure.condition == 8 and inside
                runs.append(
                    make_vectors.Run(list(prompt), "package refuses" if defined else "none")
                )
            finally:
                model.free(sequence)
        what = "all weights are 0, and each run is one token, whose embedding is a row"
        write_case(what, p, weights, runs)


package_model.IntegerModel.extend = extend_and_note
chosen_model.ChosenModel.assert_equals_recorded = assert_equals_recorded_and_write


def pytest_collection_modifyitems(items):
    for module in {item.module for item in items}:
        fails = getattr(module, "assert_fails_as_recorded", None)
        if fails and not getattr(fails, "writes", False):

            def fails_and_write(chosen, rows, position, condition, layer, fails=fails):
                fails(chosen, rows, position, condition, layer)
                write_window(chosen, rows, True)

            fails_and_write.writes = True
            module.assert_fails_as_recorded = fails_and_write


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    name = item.name.replace("test_", "", 1)
    state["test"], state["count"] = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-"), 0


def pytest_runtest_teardown(item):
    noted, state["norm"] = state["norm"], {}
    write_the_norm_tests_rows(noted)


def pytest_sessionfinish(session):
    runs = sum(len(case["runs"]) for case in state["cases"])
    print(
        f"\nedge_cases_plugin: {len(state['cases'])} cases written, {runs} runs; {state['over']} "
        f"models passed over; {len(state['faults'])} faults of the plugin's own"
    )
    for fault in state["faults"]:
        print("edge_cases_plugin: FAULT:", fault)
