#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Compares Lean's definition of the integer model with this Python package, value by value.

    python3 integrations/vllm/integer_model/tests/lean_vectors/run.py [--large] [--vectors-only]

The script makes a new temporary folder, called OUT here, and prints its path. It writes only below
OUT and removes nothing. README.md beside this file says what the comparison shows, what it does not
show, and what it needs. The steps:

1. make_vectors.py and edge_cases_plugin.py run the package on made-up weights, on the CPU, and
   write the inputs and the reported values below OUT/vectors.
2. Every listed file must have the SHA-256 that is listed for it. --vectors-only stops here, and
   needs no Lean.
3. CompareVectors.lean is compiled, together with three unchanged files of lean/, into
   OUT/build/compare_vectors.
4. compare_vectors must find no difference in the vectors.
5. On every damaged copy it must find one, and the first that it prints must be at the value that
   the copy's DAMAGE.json names.

The exit status is 0 only if every step ended as it must. The programs' output is kept in OUT/*.txt.
A program that has not ended after four hours is stopped, and the script with it. If the script gets
SIGTERM, it stops the program that it is waiting for and ends. In both cases a program that this one
has started in turn, such as the C compiler, is not stopped.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
LEAN = PACKAGE.parent.parent.parent / "lean"
# What is compared, as in README.md, so that a case or a value that drops out does not go unseen:
# compare_vectors' closing line without and with --large, and the number of damaged copies.
CLOSING = {
    False: "103 cases, 614 runs, 30761 arrays, 66598727 values compared; 0 runs differ",
    True: "105 cases, 616 runs, 30815 arrays, 67367156 values compared; 0 runs differ",
}
DAMAGED = 146
# Of the cases' settings: how many pass the Lean library's settingsInRange, and how many pass its
# clauses on the layers that the model has. The case that --large adds passes neither.
IN_RANGE, OWN_LAYERS_IN_RANGE = 1, 68
# The three files of lean/ that are compiled, and the SHA-256 of each in the run that README.md
# reports.
MODULES = {
    "IntegerModel/Model": "200e5c0a24e4207ef77de0c2294030553a98a732fec32416e4ad58bdba8a4144",
    "Inst/Model/TablesData": "39157e38578ff807a1b41ad412a0f98f7d2dca5c6daf1039aa4ee144e627d347",
    "VerifiableInference/ModelWeights": (
        "8a01031138552eaa5f0ca2c3b94df85040d3999e2cccd890e5c495cfc012da3c"
    ),
}


# No program that the script starts may take longer than this, in seconds: several times as long as
# the whole script took in the runs that README.md reports.
TIME_LIMIT = 4 * 3600


def run(words: list, screen: pathlib.Path, **named) -> int:
    """Runs a command with its output kept in `screen`, and returns its exit status."""
    with open(screen, "a") as kept:
        return subprocess.run(
            [str(w) for w in words],
            stdout=kept,
            stderr=subprocess.STDOUT,
            timeout=TIME_LIMIT,
            **named,
        ).returncode


def keep(path: pathlib.Path, text: str) -> None:
    path.write_text(text)


def last_line(screen: pathlib.Path) -> str:
    lines = screen.read_text().strip().splitlines()
    return lines[-1] if lines else "(no output)"


def make(out: pathlib.Path, large: bool) -> bool:
    print("1. making the vectors, which takes some minutes", flush=True)
    vectors = out / "vectors"
    paths = os.pathsep.join(str(p) for p in (HERE, PACKAGE, PACKAGE / "tests"))
    environment = os.environ | {
        "PYTHONPATH": paths,
        "PYTHONDONTWRITEBYTECODE": "1",
        "VECTORS_OUT": str(vectors),
    }
    # -P: Python must not put the current directory, which may be anybody's, on its path.
    call = f"import make_vectors; make_vectors.make({str(vectors)!r}, large={large})"
    if run([sys.executable, "-P", "-c", call], out / "make.txt", env=environment, cwd=PACKAGE):
        print(f"1. make_vectors.py failed: {last_line(out / 'make.txt')}")
        return False
    words = [sys.executable, *"-P -m pytest -q -s -p no:cacheprovider -p edge_cases_plugin".split()]
    words += [f"--basetemp={out / 'pytest'}", "tests/test_arithmetic_edges.py", "-k", "five_limbs"]
    status = run(words, out / "edge-tests.txt", env=environment, cwd=PACKAGE)
    screen = (out / "edge-tests.txt").read_text()
    told = [line for line in screen.splitlines() if line.startswith("edge_cases_plugin:")]
    print("1.", "; ".join(told) or "the plugin printed nothing")
    if status:
        print(f"1. the edge tests failed: see {out / 'edge-tests.txt'}, which ends:")
        print(f"   {last_line(out / 'edge-tests.txt')}")
    # The counts are those of README.md, so that a test that is renamed does not drop out unseen.
    wanted = "39 cases written, 307 runs; 0 models passed over; 0 faults of the plugin's own"
    return status == 0 and told == ["edge_cases_plugin: " + wanted]


def build(out: pathlib.Path) -> bool:
    """Compiles with the compiler that lean/lean-toolchain names: elan picks it by the directory.
    None of the files imports Mathlib, so neither lake nor the library's build is needed."""
    made = out / "build"
    made.mkdir()
    environment = os.environ | {"LEAN_PATH": str(made)}
    sources = [(LEAN / f"{module}.lean", made / module) for module in MODULES]
    sources.append((HERE / "CompareVectors.lean", made / "CompareVectors"))
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        root = LEAN if source.is_relative_to(LEAN) else HERE
        words = ["lean", f"--root={root}", "-o", f"{target}.olean", "-c", f"{target}.c", source]
        if run(words, out / "build.txt", env=environment, cwd=LEAN):
            print(f"3. lean failed on {source}: see {out / 'build.txt'}")
            return False
    words = ["leanc", "-O3", "-o", made / "compare_vectors", *(f"{t}.c" for _, t in sources)]
    if run(words, out / "build.txt", cwd=LEAN):
        print(f"3. leanc failed: see {out / 'build.txt'}")
        return False
    print(f"3. built {made / 'compare_vectors'}")
    return True


def check_sums(top: pathlib.Path) -> tuple:
    """Reads every listed file below `top`, its damaged copies left out. Returns the files that have
    another SHA-256 than listed, the number of files, and one SHA-256 over all their names and sums,
    by which two runs' vectors can be compared."""
    wrong, lines = [], []
    for word in ("case.json", "tables.json", "function.json"):
        for listing in top.rglob(word):
            if "damaged" in listing.relative_to(top).parts:
                continue
            described = json.loads(listing.read_text())
            runs = described.get("runs", [])
            for entry in described["arrays"] + [a for r in runs for a in r["arrays"]]:
                file = listing.parent / entry["file"]
                found = hashlib.sha256(file.read_bytes()).hexdigest()
                lines.append(f"{found}  {file.relative_to(top)}\n")
                if found != entry["sha256"]:
                    wrong.append(str(file))
    return wrong, len(lines), hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def deep_stack() -> None:
    """Model.lean's sums and its argmax recurse as deep as a vector is long, 202,048 for the full
    vocabulary, which overflows the usual 8 MB of stack. This allows 1 GB."""
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    soft = 1 << 30 if hard == resource.RLIM_INFINITY else min(1 << 30, hard)
    resource.setrlimit(resource.RLIMIT_STACK, (soft, hard))


def compare(out: pathlib.Path, top: pathlib.Path) -> tuple:
    done = subprocess.run(
        [str(out / "build" / "compare_vectors"), str(top)],
        capture_output=True,
        text=True,
        cwd=out,
        preexec_fn=deep_stack,
        timeout=TIME_LIMIT,
    )
    return done.returncode, done.stdout + done.stderr


def at_the_named_value(named: str, first: str) -> bool:
    """Is the first printed difference at the value that DAMAGE.json names?"""
    if named == "the expected token":
        return first.startswith("DIFFERS") and "expected: token" in first
    if named.startswith("the expected outcome: "):
        words = named.removeprefix("the expected outcome: ")
        return first.startswith("DIFFERS") and f": expected: {words}" in first
    if named.startswith("the settings file "):
        return first.startswith(f"DIFFERS settings {named.removeprefix('the settings file ')}:")
    if named == "positions_compared":
        return "/case.json: positions_compared is " in first
    name, _, place = named.split("; or ")[-1].partition(", ")
    return f"/{name}.bin:" in first and (not place or f"({place})" in first)


def compare_damaged(out: pathlib.Path) -> bool:
    copies = sorted(p for p in (out / "vectors" / "damaged").iterdir() if p.is_dir())
    caught = placed = 0
    lines = []
    for copy in copies:
        damage = json.loads((copy / "DAMAGE.json").read_text())
        status, screen = compare(out, copy)
        first = next((line.strip() for line in screen.splitlines() if "DIFFERS" in line), "")
        seen = status == 1 and bool(first)
        if damage["kind"] == "stale-sum":
            seen = seen and bool(check_sums(copy)[0])
        there = seen and at_the_named_value(damage["a_comparison_must_fail_at"], first)
        caught += seen
        placed += there
        lines.append(
            f"{'caught' if seen else 'MISSED'}, {'at' if there else 'NOT AT'} the named value "
            f"({damage['a_comparison_must_fail_at']}): {copy.name}: status {status}: {first}\n"
        )
    keep(out / "compare-damaged.txt", "".join(lines))
    print(
        f"5. the damaged copies: {caught} of {len(copies)} caught, {placed} of {len(copies)} at "
        "the value that DAMAGE.json names"
    )
    return caught == placed == len(copies) == DAMAGED


def main() -> int:
    parser = argparse.ArgumentParser(description="See README.md beside this script.")
    parser.add_argument(
        "--large",
        action="store_true",
        help="add two layers at Llama 4 Maverick's sizes and the argmax over the full vocabulary; "
        "takes several GB of memory (README.md says how much), which the script does not limit",
    )
    parser.add_argument(
        "--vectors-only",
        action="store_true",
        help="stop when the vectors are made and their sums are checked; needs no Lean",
    )
    given = parser.parse_args()
    if not given.vectors_only:
        if not (shutil.which("lean") and shutil.which("leanc")):
            print("lean and leanc are not on the PATH: see README.md beside this script")
            return 2
        wanted = (LEAN / "lean-toolchain").read_text().strip().rpartition(":v")[2]
        version = subprocess.run(
            ["lean", "--version"], cwd=LEAN, capture_output=True, text=True, timeout=TIME_LIMIT
        )
        if f"version {wanted}," not in version.stdout:
            print(f"lean/lean-toolchain names {wanted}, but lean --version says: {version.stdout}")
            return 2
        print(f"lean: {version.stdout.strip()}, as lean/lean-toolchain names", flush=True)
        for module, reported in MODULES.items():
            found = hashlib.sha256((LEAN / f"{module}.lean").read_bytes()).hexdigest()
            same = "the file" if found == reported else "NOT the file"
            print(f"lean: {module}.lean has the SHA-256 {found}:", flush=True)
            print(f"      {same} of the run that README.md reports; the comparison runs either way")
        sys.stdout.flush()
    out = pathlib.Path(tempfile.mkdtemp(prefix="lean_vectors."))
    print(f"Everything is written below the new folder {out}", flush=True)
    if not make(out, given.large):
        return 1
    wrong, files, digest = check_sums(out / "vectors")
    print(f"2. {len(wrong)} of {files} listed files differ from their listed SHA-256")
    print(f"   the SHA-256 over all their names and sums: {digest}")
    if given.vectors_only:
        return 1 if wrong else 0
    if not build(out):
        return 1
    print("4. comparing, which takes some minutes", flush=True)
    status, screen = compare(out, out / "vectors")
    keep(out / "compare.txt", screen)
    print(f"4. the vectors: exit status {status}: {last_line(out / 'compare.txt')}")
    whole = last_line(out / "compare.txt") == CLOSING[given.large]
    if not whole:
        print(f"4. that line was to be: {CLOSING[given.large]}")
    tables = re.search(
        r"^same +tables .*: the library's lookupTables, entry for entry;", screen, re.M
    )
    if not tables:
        print("4. the library's five tables were not compared, or differ")
    valid = "same    settings target_full_f2.json: expected: settingsAreValid true; Lean: true;"
    if valid not in screen:
        print("4. settingsAreValid was not evaluated on the full-size settings, or is false")
    ranges = re.search(r"^settings in range, of \d+ cases: (\d+) by .*, (\d+) by .*$", screen, re.M)
    print(f"4. {ranges[0] if ranges else 'the settings in range were not counted'}")
    if not ranges or ranges.groups() != (str(IN_RANGE), str(OWN_LAYERS_IN_RANGE)):
        print(f"4. the two counts were to be {IN_RANGE} and {OWN_LAYERS_IN_RANGE}")
        whole = False
    fine = compare_damaged(out) and status == 0 and not wrong and bool(tables) and valid in screen
    fine = fine and whole
    print("All steps ended as they must." if fine else "NOT all steps ended as they must.")
    return 0 if fine else 1


def stop(number, frame):
    """Raises SystemExit where the script is waiting, at which subprocess.run ends the program that
    it started, though not what that program has started in turn."""
    print("Stopped by SIGTERM. NOT all steps ended as they must.")
    sys.exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired as late:
        print(f"Stopped: after {TIME_LIMIT // 3600} hours this program had not ended:")
        print("   " + " ".join(late.cmd))
        print("NOT all steps ended as they must.")
        sys.exit(1)
