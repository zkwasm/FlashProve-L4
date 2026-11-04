# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Writes test vectors from the integer model, so that a second implementation of docs/model.md can
be compared with this package value by value.

run.py calls make(); README.md describes the files.

The module chooses settings, weights and prompts, runs IntegerModel.extend on a CPU, and writes down
the inputs and what the package reported. All weights are made up. It computes no reported value
itself. It does two pieces of the model's arithmetic: a row's sum of squares, to sort the runs in
which the package's range condition 8 fails (make_case), and the product and shift by which one
output is chosen at a range condition's upper end (upper_end_cases). There it also widens the
package's interval for the length of a call (with_the_interval_widened).

It needs the package's directory and its tests/ on Python's path.
"""

import dataclasses
import hashlib
import itertools
import json
import pathlib
import shutil
import struct

import numpy
import torch

import integer_model.model as package_model
from example_models import PARAMETER_FILES, with_lines
from integer_model import IntegerModel, Parameters, RangeConditionFailed
from integer_model.arithmetic import first_argmax
from integer_model.parameters import LayerScales
from integer_model.synthetic_weights import (
    _EMBEDDING_STREAM,
    SyntheticWeights,
    matrix_layout,
    stream_entries,
)
from integer_model.tables import make_tables

FORMAT = 1
TYPES = {"int16": "<i2", "int32": "<i4", "int64": "<i8"}
SMALL = PARAMETER_FILES["target_small_f2"]
FULL_SIZE = SMALL.parent.parent / "target_full_f2" / "policy.txt"
# The largest stream value that the package admits (range condition 8, with five limbs).
LARGEST_STREAM_VALUE = 547_599_908_735


def write(path: pathlib.Path, data: bytes) -> None:
    """Every file that this module writes is written here. All paths lie below the directory that
    make() has made, and are built from this module's and the edge tests' own names."""
    path.write_bytes(data)


def write_json(path: pathlib.Path, value: dict) -> None:
    write(path, (json.dumps(value, indent=1) + "\n").encode())


def write_array(folder: pathlib.Path, file: str, array: numpy.ndarray, kind: str) -> dict:
    """Writes one array as a raw file and returns the file's part of its entry in "arrays"."""
    limits = numpy.iinfo(TYPES[kind])
    if array.size and (int(array.min()) < limits.min or int(array.max()) > limits.max):
        raise SystemExit(f"{file}: a value does not fit {kind}")
    data = numpy.ascontiguousarray(array.astype(TYPES[kind])).tobytes()
    write(folder / file, data)
    return {
        "shape": list(array.shape),
        "type": kind,
        "file": file,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def label_of(name: str, layer: int | None, expert: int | None = None) -> str:
    """An array's name in the files, for example L3.E5.Wgr, L3.x or lam."""
    return (
        ("" if layer is None else f"L{layer}.") + ("" if expert is None else f"E{expert}.") + name
    )


# Weight sources. Like the package's, each has matrix(name, layer, expert), which returns a matrix
# with the output index first, embedding_rows(token_ids) and device.


class Seeded:
    """The package's synthetic weights, made one matrix at a time and not kept, so that a layer at
    full size fits a small machine's memory."""

    def __init__(self, parameters: Parameters):
        self.device, self.p = torch.device("cpu"), parameters

    def matrix(self, name, layer=None, expert=None):
        stream, inputs, outputs, start = matrix_layout(self.p, name, layer, expert)
        by_input = torch.empty((inputs, outputs), dtype=torch.int16)
        step = max(1, (1 << 22) // outputs)
        for first in range(0, inputs, step):
            last = min(first + step, inputs)
            index = torch.arange(start + first * outputs, start + last * outputs)
            entries = stream_entries(self.p.seed, stream, index, 13)
            by_input[first:last] = entries.reshape(last - first, outputs).to(torch.int16)
        return by_input.T.contiguous()

    def embedding_rows(self, token_ids):
        index = token_ids[:, None] * self.p.d + torch.arange(self.p.d)
        return stream_entries(self.p.seed, _EMBEDDING_STREAM, index, 20)


def all_matrices(p: Parameters) -> list:
    """The (name, layer, expert) of every weight matrix of a model."""
    keys = []
    for layer in range(p.L):
        keys += [(name, layer, None) for name in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")]
        if p.has_experts(layer):
            keys.append(("Wr", layer, None))
            keys += [(n, layer, e) for e in range(p.NE) for n in ("Wgr", "Wur", "Wdr")]
    return [*keys, ("Wh", None, None)]


def check_seeded(p: Parameters) -> None:
    """Stops unless Seeded gives the same numbers as the package's SyntheticWeights."""
    mine, theirs, tokens = Seeded(p), SyntheticWeights(p, "cpu"), torch.arange(p.V)
    for key in all_matrices(p):
        if not torch.equal(mine.matrix(*key), theirs.matrix(*key)):
            raise SystemExit(f"Seeded differs from SyntheticWeights at {key}")
    if not torch.equal(mine.embedding_rows(tokens), theirs.embedding_rows(tokens)):
        raise SystemExit("Seeded differs from SyntheticWeights in the embedding")


class Changed:
    """Another source's weights with some matrices changed. `changes` maps (name, layer, expert) to
    a function on a copy of the matrix; `rows`, if given, is a function on (rows, token_ids)."""

    def __init__(self, inner, changes: dict, rows=None):
        self.inner, self.device, self.changes, self.rows = inner, inner.device, changes, rows

    def matrix(self, name, layer=None, expert=None):
        w = self.inner.matrix(name, layer, expert)
        change = self.changes.get((name, layer, expert))
        return change(w.clone()) if change else w

    def embedding_rows(self, token_ids):
        rows = self.inner.embedding_rows(token_ids)
        return self.rows(rows.clone(), token_ids) if self.rows else rows


class Written:
    """Hands on another source's weights, and writes each matrix the first time it is asked for."""

    def __init__(self, inner, folder: pathlib.Path):
        self.inner, self.device, self.folder, self.arrays = inner, inner.device, folder, {}

    def matrix(self, name, layer=None, expert=None):
        w = self.inner.matrix(name, layer, expert)
        if (name, layer, expert) not in self.arrays:
            label = label_of(name, layer, expert)
            self.arrays[name, layer, expert] = {
                "name": label,
                "kind": "weight",
                "symbol": name,
                "layer": layer,
                "expert": expert,
                "order": "input first",
            } | write_array(self.folder, label + ".bin", w.T.cpu().numpy(), "int16")
        return w

    def embedding_rows(self, token_ids):
        return self.inner.embedding_rows(token_ids)


# Settings and tables.


def parameters_from(out: pathlib.Path, name: str, base: pathlib.Path, lines: dict, scales: list):
    """The package's reading of a shipped parameter file with some lines changed and with the given
    layers' quantization scales. The changed files are kept in parameters/NAME."""
    folder = out / "parameters" / name
    folder.mkdir()
    text = with_lines({**lines, "L": len(scales)})(base.read_text())
    write(folder / "policy.txt", text.encode())
    words = [word for k in scales for word in dataclasses.astuple(k)[:13]]
    words += [w for layer, k in enumerate(scales) if layer % 2 == 1 for w in (k.kr, k.kgR, k.kuR)]
    write(folder / "KL.bin", struct.pack(f"<{len(words)}I", *words))
    return Parameters.read(folder / "policy.txt")


def settings_of(p: Parameters) -> dict:
    names = "T L d n_h n_g dh f fe NE V cs eps1 eps2 epsF MF KF".split()
    return {n: getattr(p, n) for n in names} | {"layers": [dataclasses.asdict(k) for k in p.layers]}


def write_tables(out: pathlib.Path, p: Parameters) -> str:
    """Writes the five lookup tables of p, as the package holds them once it has checked their
    digests, unless they are there already. Returns the directory's name below tables/."""
    made_with = {n: getattr(p, n) for n in "T dh cs theta0 expS rx eg smax".split()}
    ident = ".".join(f"{n.removesuffix('0')}{v}" for n, v in made_with.items())
    folder = out / "tables" / ident
    if folder.exists():
        return ident
    folder.mkdir()
    t = make_tables(p, torch.device("cpu"))
    arrays = [
        {"name": name, "symbol": symbol, "lowest_key": lowest}
        | write_array(folder, name + ".bin", tensor.numpy(), "int32")
        for name, symbol, tensor, lowest in (
            ("cosine", "C", t.cos, None),
            ("sine", "S", t.sin, None),
            # The package appends one 0, which it reads for every key from u0 on.
            ("exponential", "EX", t.softmax[:-1], 0),
            ("silu", "Ts", t.silu, -32768),
            ("sigmoid", "Sig", t.sigmoid, -16384),
        )
    ]
    full = Parameters.read(FULL_SIZE)
    described = {
        "format": FORMAT,
        "parameters": made_with,
        "digests_in_the_parameter_file": {n: list(words) for n, words in p.table_digests.items()},
        "compare_with_the_librarys_tables": p.table_digests == full.table_digests
        and all(getattr(full, n) == v for n, v in made_with.items()),
        "arrays": arrays,
    }
    write_json(folder / "tables.json", described)
    return ident


# Cases.


def reported(p: Parameters, weights, prompt: list, change_tables=None, model=None) -> tuple:
    """Runs the package on one prompt, in a new sequence. Returns what it reported, as a list of
    (name, layer, tensor) in the order of evaluation, and the last position's greedy choice or the
    failure. `model`: an IntegerModel of p and weights, if the caller has one already."""
    model, seen = model or IntegerModel(p, weights, "cpu"), []
    if change_tables:
        change_tables(model.tables)
    sequence = model.new_sequence()
    try:
        choices = model.extend(
            sequence,
            prompt,
            record=lambda n, layer, tensor: seen.append((n, layer, tensor.clone())),
        )
        return seen, int(choices[-1])
    except RangeConditionFailed as failure:
        return seen, failure
    finally:
        model.free(sequence)


def condition_1_fails_too(p: Parameters, seen: list, position: int) -> bool:
    """For a run in which the package's condition 8 failed at `position`. The package reports a
    stream row before the norm that reads it tests it, so the first reported row there with too
    large an entry is the one that failed. If its squares and eps reach 2^78, condition 1 fails as
    well."""
    eps_of = {"x": p.eps1, "x_mid": p.eps2, "x_last": p.epsF}
    for name, _, tensor in seen:
        row = [int(v) for v in tensor[position]] if name in eps_of else []
        if any(abs(v) > LARGEST_STREAM_VALUE for v in row):
            return sum(v * v for v in row) + eps_of[name] >= 1 << 78
    return False


@dataclasses.dataclass
class Run:
    prompt: list
    # What the case's author expects: "token"; "none"; "package refuses"; or "package refuses or
    # none", which make_case decides where condition 8 failed.
    kind: str = "token"
    # For "package refuses": the token with the interval widened, if it is known.
    token: int | None = None
    # For "none": the first position at which the model is undefined, if that is not the position
    # at which the package fails first.
    undefined_from: int | None = None
    # For "package refuses": what the package reported with the interval widened, as reported()
    # returns it, if these values are to be compared at all positions.
    widened_values: list | None = None


def make_case(
    out: pathlib.Path,
    name: str,
    what: str,
    p: Parameters,
    weights,
    runs: list,
    whole_model: bool = True,
) -> dict:
    """Runs the package on one model and some prompts and writes cases/NAME.

    `runs`: a list of Run. README.md, "expected", says what the kinds mean. It stops if the
    package does otherwise than the run's kind says.

    `whole_model`: write every matrix and embedding row, not only those that a run asked for.
    """
    if not (p.router_clamp and p.gate_clamp and p.NX == 5):
        raise SystemExit(f"{name}: the vectors are for the model with rcl 1, gcl 1 and xl 5")
    folder = out / "cases" / name
    folder.mkdir()
    written = Written(weights, folder)
    model = IntegerModel(p, written, "cpu")
    described = []
    for number, run in enumerate(runs):
        prompt, kind = run.prompt, run.kind
        seen, result = reported(p, written, prompt, model=model)
        if isinstance(result, int):
            expected, compared = {"kind": "token", "token": result}, len(prompt)
        else:
            if kind == "token":
                raise SystemExit(
                    f"{name}, run {number}: a token was expected; the package: {result}"
                )
            if kind == "package refuses or none" and result.condition == 8:
                if condition_1_fails_too(p, seen, result.position):
                    kind = "none"
            expected = {"kind": kind, "condition": result.condition, "position": result.position}
            if kind == "none":
                first = run.undefined_from
                expected["undefined_from"] = result.position if first is None else first
            if run.token is not None:
                expected["token_with_the_interval_widened"] = run.token
            compared = result.position
            if run.widened_values is not None:
                expected["values_with_the_interval_widened"] = True
                seen, compared = run.widened_values, len(prompt)
        if expected["kind"] != kind or kind not in ("token", "none", "package refuses"):
            raise SystemExit(f"{name}, run {number}: {kind!r} was expected; the package: {result}")
        (folder / f"run{number:03d}").mkdir()
        arrays = [
            {"name": label_of(n, layer), "kind": "intermediate", "symbol": n, "layer": layer}
            | write_array(
                folder,
                f"run{number:03d}/{label_of(n, layer)}.bin",
                tensor[:compared].numpy(),
                "int64",
            )
            for n, layer, tensor in (seen if compared else [])
        ]
        described.append(
            {
                "prompt": [int(t) for t in prompt],
                "expected": expected,
                "positions_compared": compared,
                "arrays": arrays,
            }
        )
    if whole_model:
        for key in all_matrices(p):
            written.matrix(*key)
    used = sorted({t for run in described for t in run["prompt"]})
    tokens = torch.arange(p.V) if whole_model else torch.tensor(used)
    embedding = {
        "name": "Emb",
        "kind": "embedding",
        "symbol": "Emb",
        "rows_are_tokens": "all" if whole_model else used,
    } | write_array(folder, "Emb.bin", weights.embedding_rows(tokens).numpy(), "int64")
    case = {
        "format": FORMAT,
        "name": name,
        "what": what,
        "parameter_lines": {"rcl": 1, "gcl": 1, "xl": 5},
        "settings": settings_of(p),
        "tables": write_tables(out, p),
        "every_matrix_is_written": whole_model,
        "arrays": [*written.arrays.values(), embedding],
        "runs": described,
    }
    write_json(folder / "case.json", case)
    print(f"case {name}: {len(described)} runs; the first: {described[0]['expected']}", flush=True)
    return case


# Damaged copies.


class Damage:
    """Makes copies of one good case with one run, each with one deliberate error, as
    damaged/NAME--KIND (README.md, "Damaged copies").

    Where an input is damaged, the package is run on the damaged input first. The names of the
    reported values that then differ go into DAMAGE.json, and the run stops if there is none.
    """

    def __init__(self, out: pathlib.Path, case: dict, p: Parameters, weights):
        self.out, self.case, self.p, self.weights = out, case, p, weights
        self.name, self.prompt = case["name"], case["runs"][0]["prompt"]
        self.good = self.values(p, weights)

    def values(self, p: Parameters, weights, change_tables=None) -> dict:
        seen, _ = reported(p, weights, self.prompt, change_tables)
        return {label_of(n, layer): tensor for n, layer, tensor in seen}

    def differing(self, other: dict) -> list:
        """The names whose values differ from the good run's, in the order of evaluation. If the
        package refuses the damaged input, its values are still true up to the first that fails a
        range condition, so the first name stands."""
        return [n for n, v in self.good.items() if n in other and not v.equal(other[n])]

    def copy(self, kind: str, more: tuple = ()) -> tuple:
        """A new copy. Returns its top directory, the case's directory and the tables' directory.
        `more`: further directories of the vectors to copy."""
        top = self.out / "damaged" / f"{self.name}--{kind}"
        parts = (f"cases/{self.name}", f"tables/{self.case['tables']}", *more)
        for part in parts:
            shutil.copytree(self.out / part, top / part)
        return top, top / parts[0], top / parts[1]

    def describe(self, kind: str, top: pathlib.Path, what: dict, differing: list, at: str = ""):
        if not (differing or at):
            raise SystemExit(f"{self.name}--{kind}: no reported value shows the damage")
        write_json(
            top / "DAMAGE.json",
            {"kind": kind, "case": self.name}
            | what
            | {
                "a_comparison_must_fail_at": at or differing[0],
                "the_package_on_the_damaged_input_differs_in": differing,
            },
        )
        print(f"damaged {self.name}--{kind}: must fail at {at or differing[0]}", flush=True)

    @staticmethod
    def add_one(listing: pathlib.Path, label: str, index: int, keep_sum: bool = False) -> dict:
        """Adds 1 to one number of a listed array. The listed SHA-256 follows unless keep_sum."""
        described = json.loads(listing.read_text())
        listed = described["arrays"] + [a for r in described.get("runs", []) for a in r["arrays"]]
        entry = next(a for a in listed if a["name"] == label)
        values = numpy.fromfile(listing.parent / entry["file"], dtype=TYPES[entry["type"]])
        values[index] += 1
        write(listing.parent / entry["file"], values.tobytes())
        if not keep_sum:
            entry["sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
            write_json(listing, described)
        now = int(values[index])
        return {"file": entry["file"], "flat_index": index, "was": now - 1, "is": now}

    def changed_settings(self, kind: str, change, other: Parameters) -> None:
        """A copy whose case.json has the settings changed by `change`. `other`: the same change
        made to the package's parameters."""
        top, folder, _ = self.copy(kind)
        case = json.loads((folder / "case.json").read_text())
        change(case["settings"])
        write_json(folder / "case.json", case)
        differing = self.differing(self.values(other, self.weights))
        self.describe(kind, top, {"file": "case.json"}, differing)

    def exchanged_settings(self) -> None:
        """Two settings that a reader could take for one another are exchanged, wherever they
        differ."""
        p = self.p

        def with_layer(layer: int, **values) -> Parameters:
            k = dataclasses.replace(p.layers[layer], **values)
            return dataclasses.replace(p, layers=(*p.layers[:layer], k, *p.layers[layer + 1 :]))

        pairs = [("M1", "M2"), ("K1", "K2"), ("kq", "kk"), ("kg", "ku")]
        routed = [("kgR", "kuR"), ("kg", "kgR"), ("ku", "kuR")]
        for layer, k in enumerate(p.layers):
            for one, two in pairs + (routed if p.has_experts(layer) else []):
                a, b = getattr(k, one), getattr(k, two)
                if a != b:

                    def change(s, layer=layer, one=one, two=two, a=a, b=b):
                        s["layers"][layer] |= {one: b, two: a}

                    self.changed_settings(
                        f"exchanged.L{layer}.{one}-{two}",
                        change,
                        with_layer(layer, **{one: b, two: a}),
                    )
        for one, two in (("eps1", "eps2"), ("eps2", "epsF"), ("eps1", "epsF")):
            a, b = getattr(p, one), getattr(p, two)
            if a != b:
                self.changed_settings(
                    f"exchanged.{one}-{two}",
                    lambda s, one=one, two=two, a=a, b=b: s.update({one: b, two: a}),
                    dataclasses.replace(p, **{one: b, two: a}),
                )
        if p.L >= 3 and p.layers[0] != p.layers[2]:

            def two_layers(s):
                s["layers"][0], s["layers"][2] = s["layers"][2], s["layers"][0]

            layers = (p.layers[2], p.layers[1], p.layers[0], *p.layers[3:])
            self.changed_settings(
                "exchanged.layers-0-2", two_layers, dataclasses.replace(p, layers=layers)
            )
        for one, two in (("MF", "M1"), ("KF", "K1")):  # the final norm's and layer 0's first norm's
            a, b = getattr(p, one), getattr(p.layers[0], two)
            if a != b:

                def final_and_first(s, one=one, two=two, a=a, b=b):
                    s[one], s["layers"][0][two] = b, a

                self.changed_settings(
                    f"exchanged.{one}-L0.{two}",
                    final_and_first,
                    dataclasses.replace(with_layer(0, **{two: a}), **{one: b}),
                )

    def other_outcome(self, expected: dict) -> None:
        """The run expects another outcome than the true one, and lists no values."""
        kind = "outcome." + expected["kind"].replace(" ", "-")
        top, folder, _ = self.copy(kind)
        case = json.loads((folder / "case.json").read_text())
        case["runs"][0] |= {"expected": expected, "positions_compared": 0, "arrays": []}
        write_json(folder / "case.json", case)
        word = "none" if expected["kind"] == "none" else "defined"
        self.describe(kind, top, {"file": "case.json"}, [], "the expected outcome: " + word)

    def other_expected(self, kind: str, at: str, **changed) -> None:
        """Changes entries of the run's "expected". The listed values stay."""
        top, folder, _ = self.copy(kind)
        case = json.loads((folder / "case.json").read_text())
        case["runs"][0]["expected"] |= changed
        write_json(folder / "case.json", case)
        self.describe(kind, top, {"file": "case.json"} | changed, [], at)

    def one_position_fewer(self) -> None:
        """Every listed array has lost its last position, and positions_compared says so, as if the
        maker had dropped it. All values that are left are right."""
        top, folder, _ = self.copy("positions-compared")
        case = json.loads((folder / "case.json").read_text())
        run = case["runs"][0]
        run["positions_compared"] -= 1
        for entry in run["arrays"]:
            values = numpy.fromfile(folder / entry["file"], dtype=TYPES[entry["type"]])
            values = values.reshape(entry["shape"])[:-1]
            entry |= write_array(folder, entry["file"], values, entry["type"])
        write_json(folder / "case.json", case)
        self.describe("positions-compared", top, {"file": "case.json"}, [], "positions_compared")

    def settings_file(self) -> None:
        """The full-size settings with kd = 20 in one layer, still listed as valid."""
        top, _, _ = self.copy("settings-file", ("settings",))
        file = top / "settings" / "target_full_f2.json"
        described = json.loads(file.read_text())
        described["settings"]["layers"][0]["kd"] = 20
        write_json(file, described)
        at = "the settings file target_full_f2.json"
        self.describe("settings-file", top, {"file": "settings/target_full_f2.json"}, [], at)

    def full_size_table(self, table: str, index: int) -> None:
        """One entry of the tables that are compared with the library's is off by one."""
        full = Parameters.read(FULL_SIZE)
        ident = write_tables(self.out, full)
        top, _, _ = self.copy("full-size-table." + table, (f"tables/{ident}",))
        what = self.add_one(top / "tables" / ident / "tables.json", table, index)
        columns = full.dh // 2 if table in ("cosine", "sine") else 1
        at = f"{table}, position {index // columns}, entry {index % columns}"
        self.describe("full-size-table." + table, top, what, [], at)

    def one_number_off_by_one(self) -> None:
        p, weights, prompt = self.p, self.weights, self.prompt
        # One weight of each of three matrices. A rounding shift absorbs most single steps, so the
        # places are tried from the middle on until a reported value shows the change.
        for key in (("Wq", 0, None), ("Wd", 0, None), ("Wh", None, None)):
            matrix = weights.matrix(*key)
            outputs, inputs = matrix.shape
            for step in range(outputs * inputs):
                j, k = divmod((outputs * inputs // 2 + step) % (outputs * inputs), inputs)
                if int(matrix[j, k]) >= 4095:
                    continue

                def plus_one(w, j=j, k=k):
                    w[j, k] += 1
                    return w

                differing = self.differing(self.values(p, Changed(weights, {key: plus_one})))
                if differing:
                    kind = "weight." + label_of(key[0], key[1])
                    top, folder, _ = self.copy(kind)
                    what = self.add_one(folder / "case.json", kind[7:], k * outputs + j)
                    self.describe(kind, top, what | {"places_tried": step + 1}, differing)
                    break

        def other_row(rows, token_ids):
            rows[token_ids == prompt[0], 0] += 1
            return rows

        top, folder, _ = self.copy("embedding")
        self.describe(
            "embedding",
            top,
            self.add_one(folder / "case.json", "Emb", prompt[0] * p.d),
            self.differing(self.values(p, Changed(weights, {}, other_row))),
        )
        # Table entries that the run reads: SiLU's at layer 0's first key, the exponential's at 0,
        # the sigmoid's at layer 1's first chosen logit, the first sine of position 1 that shows.
        # tables.json keeps the parameter file's digests, which do not fit the damaged table.
        logit = self.good["L1.lr"][0, int(self.good["L1.e_star"].reshape(-1)[0])]
        for table, field, places in (
            ("silu", "silu", [int(self.good["L0.key"][0, 0]) + 32768]),
            ("exponential", "softmax", [0]),
            ("sigmoid", "sigmoid", [int(logit) + 16384]),
            ("sine", "sin", range(p.dh // 2, p.dh)),
        ):
            for index in places:

                def change(t, field=field, index=index):
                    getattr(t, field).view(-1)[index] += 1

                differing = self.differing(self.values(p, weights, change))
                if differing:
                    break
            top, _, tables = self.copy("table." + table)
            what = self.add_one(tables / "tables.json", table, index)
            self.describe("table." + table, top, what, differing)
        k = dataclasses.replace(p.layers[0], kv=p.layers[0].kv + 1)
        self.changed_settings(
            "setting.L0.kv",
            lambda s: s["layers"][0].update(kv=k.kv),
            dataclasses.replace(p, layers=(k, *p.layers[1:])),
        )
        # Expected values: one intermediate value, the token, and a file whose listed sum is stale.
        top, folder, _ = self.copy("intermediate.L1.m")
        what = self.add_one(folder / "case.json", "L1.m", p.d + 3)
        self.describe("intermediate.L1.m", top, what, [], "L1.m, position 1, entry 3")
        top, folder, _ = self.copy("token")
        case = json.loads((folder / "case.json").read_text())
        expected = case["runs"][0]["expected"]
        expected["token"] = (expected["token"] + 1) % p.V
        write_json(folder / "case.json", case)
        self.describe("token", top, {"file": "case.json"}, [], "the expected token")
        top, folder, _ = self.copy("stale-sum")
        what = self.add_one(folder / "case.json", "L0.x_mid", 5, keep_sum=True)
        at = "the file's SHA-256; or L0.x_mid, position 0, entry 5"
        self.describe("stale-sum", top, what | {"sum_left_as_it_was": True}, [], at)


# The cases.


# Three eps that are as large as the rows' sums of squares, so that each one shows in the norms.
LARGE_EPS = {"eps1": 300_000_000_001, "eps2": 700_000_000_003, "epsF": 1_100_000_000_007}


def eps_lines(eps: dict) -> dict:
    """The parameter file's lines for the three eps."""
    low = {key + "_lo": value & 0xFFFFFFFF for key, value in eps.items()}
    return low | {key + "_hi": value >> 32 for key, value in eps.items()}


def narrow(out, name: str, L=2, d=16, H=1, G=1, f=8, NE=4, seed=1, eps=None, **scales):
    """A narrow model on target_small_f2's tables (T = 64, dh = 128, V = 64), with quantization
    scales that suit 13-bit weights at these widths. Every fourth layer has no rotary step. The
    scales given here lie in the ranges that the Lean library's theorems ask for
    (layerSettingsInRange, lean/VerifiableInference/ModelWeights.lean)."""
    scale = dict(kq=28, kk=28, kv=14, ko=8, kg=10, ku=12, kd=19, M1=44, K1=32, M2=44, K2=32, sh=12)
    scale |= dict(rp=1, kr=14, kgR=9, kuR=12) | scales
    dense = {"kr": 0, "kgR": 0, "kuR": 0}
    layers = []
    for layer in range(L):
        rotary = {"rp": 0} if layer % 4 == 3 else {}
        layers.append(LayerScales(**scale | rotary | ({} if layer % 2 else dense)))
    lines = {"d": d, "H": H, "G": G, "f": f, "fe": f // 2, "NE": NE, "V": 64, "seed_lo": seed}
    return parameters_from(out, name, SMALL, lines | {"KF": 32} | eps_lines(eps or {}), layers)


def unlike(out: pathlib.Path, name: str, L=2, d=16, H=1, f=8, NE=4, seed=5) -> Parameters:
    """A narrow model in which no two settings that a reader could take for one another are equal.
    The three eps are as large as the rows' sums of squares, so that each one shows. The norms'
    exponents are such that the package also accepts each pair with its M or its K exchanged.
    These scales lie outside the ranges that the Lean library's theorems ask for: three of the four
    kd are above 19, and the layers of a group of four have different sh."""
    # fmt: off
    layers = [
        dict(M1=44, K1=31, M2=43, K2=30, kq=28, kk=27, kv=14, ko=8, kg=10, ku=12, kd=20, sh=12,
             rp=1, kr=0, kgR=0, kuR=0),
        dict(M1=42, K1=30, M2=44, K2=32, kq=27, kk=29, kv=13, ko=9, kg=11, ku=13, kd=21, sh=11,
             rp=1, kr=14, kgR=9, kuR=12),
        dict(M1=43, K1=30, M2=41, K2=29, kq=29, kk=28, kv=15, ko=7, kg=9, ku=11, kd=19, sh=10,
             rp=1, kr=0, kgR=0, kuR=0),
        dict(M1=45, K1=32, M2=44, K2=31, kq=26, kk=27, kv=12, ko=10, kg=12, ku=14, kd=22, sh=12,
             rp=0, kr=13, kgR=10, kuR=13),
    ]
    # fmt: on
    lines = {"d": d, "H": H, "G": 1, "f": f, "fe": f // 2, "NE": NE, "V": 64, "seed_lo": seed}
    lines |= {"MF": 45, "KF": 32} | eps_lines(LARGE_EPS)
    return parameters_from(out, name, SMALL, lines, [LayerScales(**k) for k in layers[:L]])


def copied_output(source: int, target: int):
    """Output `target` of a matrix gets output `source`'s weights, so the two tie on every input."""

    def change(w):
        w[target] = w[source]
        return w

    return change


def small_cases(out: pathlib.Path, wanted) -> None:
    def case(name: str, what: str, p: Parameters, prompt: list, weights=None) -> dict | None:
        if wanted(name):
            return make_case(out, name, what, p, weights or Seeded(p), [Run(prompt)])

    p = narrow(out, "two-layers")
    grouped = narrow(
        out, "grouped-heads", L=4, d=32, H=4, G=2, f=16, NE=8, seed=2, ku=13, kuR=13, kv=15
    )
    check_seeded(p)
    check_seeded(grouped)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    made = case(
        "whole.2-layers.8-tokens",
        "a dense layer and a layer with experts, seeded weights, a prompt of 8 tokens",
        p,
        prompt,
    )
    if made:
        damage = Damage(out, made, p, Seeded(p))
        damage.exchanged_settings()
        damage.one_number_off_by_one()
        damage.one_position_fewer()
    made = case("whole.2-layers.1-token", "the same model, a prompt of one token", p, [7])
    if made:
        damage = Damage(out, made, p, Seeded(p))
        for table, index in (
            ("cosine", 64),
            ("sine", 64),
            ("exponential", 2306),
            ("silu", 40000),
            ("sigmoid", 20000),
        ):
            damage.full_size_table(table, index)
        damage.settings_file()
    case(
        "whole.2-layers.T-tokens",
        "the same model, a prompt of T = 64 tokens",
        p,
        [(5 * t + 3) % 64 for t in range(64)],
    )
    case(
        "whole.4-layers.grouped-heads",
        "four layers, the last without rotary step; 4 query heads on 2 key-value heads",
        grouped,
        [(7 * t + 1) % 64 for t in range(12)],
    )
    case(
        "whole.48-layers",
        "48 narrow layers, the number for which the Lean library's settingsInRange asks",
        narrow(out, "48-layers", L=48, seed=4),
        [11, 2, 35, 8, 50, 23],
    )
    case(
        "whole.2-layers.saturated",
        "scales under which most normed rows, values and router scales sit at their clamps' ends",
        narrow(out, "saturated", seed=3, K1=30, K2=30, kq=26, kk=26, kv=12, kr=11, sh=8),
        [9, 8, 7, 6, 5, 4],
    )
    what = (
        "no two settings that could be taken for one another are equal: the norms' exponents and "
        "eps, kq and kk, kg and ku, the shared and the routed expert's shifts, one layer's and the "
        "next's"
    )
    for name, q, tokens in (
        ("unlike.2-layers", unlike(out, "unlike-2"), [2, 7, 1, 8, 2, 8]),
        (
            "unlike.4-layers",
            unlike(out, "unlike-4", L=4, d=32, H=2, f=16, NE=8, seed=6),
            [3, 14, 15, 9, 26, 5],
        ),
    ):
        made = case(name, what, q, tokens)
        if made:
            Damage(out, made, q, Seeded(q)).exchanged_settings()
    # Ties. A plain run says which token comes first, and which experts. A copy of the winner's
    # weights under a neighbouring index then ties with it, and the lower index must win.
    seen, first = reported(p, Seeded(p), prompt)
    chosen = next(tensor for n, layer, tensor in seen if n == "e_star").tolist()
    for word, other in (("below", first - 1), ("above", first + 1)):
        if 0 <= other < p.V:
            made = case(
                f"tie.tokens.{word}",
                f"token {other} has token {first}'s column of Wh, so their logits tie at every "
                "position: the lower id wins",
                p,
                prompt,
                Changed(Seeded(p), {("Wh", None, None): copied_output(first, other)}),
            )
            if made and made["runs"][0]["expected"]["token"] != min(first, other):
                raise SystemExit(f"tie.tokens.{word}: the package did not choose the lower id")
    for word, winner, other in (
        ("below", max(chosen), max(chosen) - 1),
        ("above", min(chosen), min(chosen) + 1),
    ):
        if 0 <= other < p.NE:
            case(
                f"tie.experts.{word}",
                f"expert {other} has expert {winner}'s column of Wr in layer 1, so their router "
                "logits tie at every position: where they come first, the lower index wins",
                p,
                prompt,
                Changed(Seeded(p), {("Wr", 1, None): copied_output(winner, other)}),
            )


def seeded_cases(out: pathlib.Path, wanted) -> None:
    """27 models with seeded weights throughout, of two to four layers, three widths and three
    arrangements of heads, each with ten prompts of 8 to 64 tokens. The hidden units, the experts
    and the softmax shift each take three values, laid so that each value meets each number of
    layers, each width and each arrangement of heads three times; a third of the models, laid in
    the same way, have the large eps. Not every combination occurs.
    Four of the models come a second time, with the first prompt alone, and these get damaged
    copies."""
    shapes = itertools.product((2, 3, 4), (16, 32, 64), ((1, 1), (2, 1), (4, 2)))
    for number, (L, d, (H, G)) in enumerate(shapes):
        a, b, c = number // 9, number // 3 % 3, number % 3  # the places of L, d and the heads
        f, NE = (8, 16, 32)[(a + b + c) % 3], (4, 8, 16)[(a + b + 2 * c) % 3]
        name = f"seeded.{number:02d}.{L}-layers.d{d}.{H}-{G}-heads.f{f}.{NE}-experts"
        if not wanted(name):
            continue
        p = narrow(
            out,
            name,
            L=L,
            d=d,
            H=H,
            G=G,
            f=f,
            NE=NE,
            seed=100 + number,
            eps=LARGE_EPS if (a + 2 * b + c) % 3 == 1 else None,
            sh=(10, 11, 12)[(2 * a + b + c) % 3],
        )
        runs = []
        # Odd steps, so that no token comes twice in a prompt.
        for i, step in enumerate((7, 11, 13, 17, 19, 23, 29, 31, 37, 41)):
            length = (8, 12, 16, 24, 32, 48, 64)[(number + 3 * i) % 7]
            runs.append(Run([(step * t + number + i) % 64 for t in range(length)]))
        make_case(out, name, "seeded weights throughout; ten prompts", p, Seeded(p), runs)
        if number % 7 == 3:
            what = "the same model with its first prompt alone, for the damaged copies"
            made = make_case(out, name + ".one-prompt", what, p, Seeded(p), runs[:1])
            damage = Damage(out, made, p, Seeded(p))
            damage.exchanged_settings()
            damage.one_number_off_by_one()


def with_the_interval_widened(p: Parameters, weights, prompt: list) -> tuple:
    """reported(), with the package's interval -32,768 .. 32,639 widened to -32,768 .. 32,767, the
    column "defined" of docs/model.md, section 9."""
    narrow_interval = package_model._TWO_LIMBS
    package_model._TWO_LIMBS = (-32768, 32767)
    try:
        return reported(p, weights, prompt)
    finally:
        package_model._TWO_LIMBS = narrow_interval


def weights_for(x: list, goal: int, slack: int) -> list:
    """13-bit weights w with |x[0] w[0] + x[1] w[1] + .. - goal| <= slack."""
    w, left = [0] * len(x), goal
    for k in sorted(range(len(x)), key=lambda k: -abs(x[k])):
        if x[k]:
            w[k] = max(-4095, min(4095, round(left / x[k])))
            left -= w[k] * x[k]
    if abs(left) > slack:
        raise SystemExit(f"no weights found for the sum {goal}: {left} is left")
    return w


def upper_end_cases(out: pathlib.Path, wanted) -> None:
    """At range conditions 3, 4, 5 and 7 the package stops at 32,639 and the model is defined up to
    32,767. In a seeded model, output 0 of one matrix is chosen so that the condition's value at a
    prompt of one token is 32,639, 32,640, 32,767 or 32,768. The shift behind the matrix is lowered
    until 13-bit weights reach that far, and all other weights that are read under that shift are
    made smaller by as much.

    The expected outcome is the package's own: as it is, and with its interval widened. At 32,640
    and 32,767 the values that the package then reports are written and compared too. The run stops
    unless the package fails at the condition that is aimed at and, widened, reports the wanted
    value at that entry (ly, the value of condition 5, is not reported)."""
    shape = dict(L=2, d=64, H=1, G=1, f=16, NE=4, seed=77)
    prompt = [5]  # at position 0 the rotary step changes nothing: the cosine is 2^cs, the sine 0
    plain = narrow(out, "upper-end.plain", **shape)
    seen, _ = reported(plain, Seeded(plain), prompt)
    sites = (
        # condition, matrix, its layer, the vector that it reads, its shift, the value's name
        (3, "Wq", 0, "a", "kq", "qw"),
        (4, "Wv", 0, "a", "kv", "vw"),
        (5, "Wr", 1, "m", "kr", None),  # ly is not reported
        (7, "Wu", 0, "m", "ku", "u"),
    )
    for condition, matrix, layer, reads, shift, told in sites:
        x = [int(v) for v in next(t for n, at, t in seen if (n, at) == (reads, layer))[0]]
        before = getattr(plain.layers[layer], shift)
        # qw is the product times 2^cs, shifted by kq. ly is the shifted product divided by 8.
        spare, times = (plain.cs, 1) if condition == 3 else (0, 8) if condition == 5 else (0, 1)
        k = before
        while 5 * (32768 * times << (k - spare)) > 4 * 4095 * sum(abs(v) for v in x):
            k -= 1
        name = f"upper-end.condition-{condition}"
        p = narrow(out, name, **shape, **{shift: k}, **({"kk": k} if condition == 3 else {}))
        for value in (32639, 32640, 32767, 32768):
            if not wanted(f"{name}.{value}"):
                continue
            # Of the eight lw with one ly, the highest; at 32,768 the lowest.
            scaled = value * 8 + (0 if value == 32768 else 7) if condition == 5 else value
            column = weights_for(x, scaled << (k - spare), (1 << (k - spare - 1)) - 1)

            def smaller(w, by=before - k):
                return w >> by

            def chosen(w, by=before - k, column=column):
                w = w >> by
                w[0] = torch.tensor(column, dtype=w.dtype)
                return w

            # narrow() gives both layers the lowered shift, and kk follows kq. Only layer 1 has a
            # router.
            others = {"Wq": ("Wq", "Wk"), "Wr": ()}.get(matrix, (matrix,))
            changes = {(n, at, None): smaller for n in others for at in range(p.L)}
            weights = Changed(Seeded(p), changes | {(matrix, layer, None): chosen})
            _, as_it_is = reported(p, weights, prompt)
            wide_seen, widened = with_the_interval_widened(p, weights, prompt)
            if isinstance(as_it_is, int):
                run = Run(prompt)
            elif isinstance(widened, int):
                run = Run(prompt, "package refuses", widened, widened_values=wide_seen)
            else:
                run = Run(prompt, "none", undefined_from=widened.position)
            due = {32639: "token", 32768: "none"}.get(value, "package refuses")
            if run.kind != due:
                raise SystemExit(f"{name}.{value}: the outcome is {run.kind!r}, not {due!r}")
            for failure in (as_it_is, widened):
                if not isinstance(failure, int) and failure.condition != condition:
                    raise SystemExit(f"{name}.{value}: condition {failure.condition} fails")
            found = [int(t[0, 0]) for n, at, t in wide_seen if (n, at) == (told, layer)]
            if told and value < 32768 and found != [value]:
                raise SystemExit(f"{name}.{value}: the package reports {found}")
            what = (
                f"seeded weights, but that output 0 of {matrix} in layer {layer} is chosen so that "
                f"the value of range condition {condition} is {value:,}"
            )
            made = make_case(out, f"{name}.{value}", what, p, weights, [run])
            if condition == 4 and value == 32767:
                none = {"kind": "none", "condition": 4, "position": 0, "undefined_from": 0}
                damage = Damage(out, made, p, weights)
                damage.other_outcome(none)
                top, folder, _ = damage.copy("intermediate.L0.vw")
                changed = damage.add_one(folder / "case.json", "L0.vw", 0)
                at = "L0.vw, position 0, entry 0"
                damage.describe("intermediate.L0.vw", top, changed, [], at)
                other = (widened + 1) % p.V
                at = f"the expected outcome: defined, token {other}"
                damage.other_expected("widened-token", at, token_with_the_interval_widened=other)
            if condition == 4 and value == 32768:
                defined = {"kind": "package refuses", "condition": 4, "position": 0}
                Damage(out, made, p, weights).other_outcome(defined)
                # The same weights, with that token at position 5 of eight. In the damaged copies
                # the model is said to be undefined from one position too early or too late.
                longer = [9, 3, 12, 40, 21, *prompt, 7, 30]
                _, widened = with_the_interval_widened(p, weights, longer)
                if isinstance(widened, int) or widened.position != 5:
                    raise SystemExit(f"{name}.{value}: position 5 was to fail, not: {widened}")
                run = Run(longer, "none", undefined_from=5)
                what += ", at position 5 of a prompt of eight tokens"
                made = make_case(out, f"{name}.{value}.position-5", what, p, weights, [run])
                damage = Damage(out, made, p, weights)
                for word, told, lean in (
                    ("early", 4, "defined before it, STILL defined with it"),
                    ("late", 6, "NOT defined before it, undefined with it"),
                ):
                    at = f"the expected outcome: none, from position {told} on; Lean: none, {lean}"
                    damage.other_expected(f"undefined-from.too-{word}", at, undefined_from=told)


def large_case(out: pathlib.Path) -> None:
    """Two layers at Llama 4 Maverick's sizes, with the vocabulary cut to 4,096. Under the example's
    sh = 8 attention is all but one-hot on these weights. Layer 1 gets sh = 12, the largest that the
    verifier admits, so that several positions share the softmax's weight there."""
    shipped = Parameters.read(FULL_SIZE)
    layers = [shipped.layers[0], dataclasses.replace(shipped.layers[1], sh=12)]
    p = parameters_from(out, "maverick-sizes", FULL_SIZE, {"V": 4096}, layers)
    make_case(
        out,
        "maverick-sizes.2-layers.3-tokens",
        "layer 0 (dense) and layer 1 (128 experts) at Llama 4 Maverick's sizes, seeded weights; "
        "the vocabulary cut to 4,096; only the experts that a position chose are written",
        p,
        Seeded(p),
        [Run([17, 4000, 911])],
        whole_model=False,
    )


def first_argmax_at_full_width(out: pathlib.Path, V: int = 202_048) -> None:
    """The package's first_argmax on six rows of V logits. At this width the output matrix itself is
    too large for a small machine, so the choice is tried apart from the product."""
    folder = out / "functions" / f"first_argmax.V{V}"
    folder.mkdir()

    def drawn(row: int) -> torch.Tensor:  # seeded numbers of up to 40 bits, with sign
        return stream_entries(7, 14, torch.arange(row * V, (row + 1) * V), 40)

    rows = {"seeded numbers": drawn(0)}
    row = drawn(1)
    row[V - 1] = int(row.max()) + 1
    rows["the largest at the last index alone"] = row
    row = drawn(2)
    row[0] = int(row.max()) + 1
    rows["the largest at index 0 alone"] = row
    row = drawn(3)
    row[[1234, 100_000, V - 1]] = int(row.max()) + 1
    rows["three equal at the top: 1,234; 100,000; the last index"] = row
    rows["all equal, negative"] = torch.full((V,), -5, dtype=torch.int64)
    row = -drawn(4).abs() - 2
    row[[V - 2, V - 1]] = -1
    rows["all negative, the last two equal at the top"] = row
    lam = torch.stack(list(rows.values()))
    choices = first_argmax(lam)
    described = {
        "format": FORMAT,
        "function": "first_argmax",
        "V": V,
        "rows": list(rows),
        "arrays": [
            {"name": "lam", "kind": "input", "symbol": "lam"}
            | write_array(folder, "lam.bin", lam.numpy(), "int64"),
            {"name": "first_choice", "kind": "expected", "symbol": "first_choice"}
            | write_array(folder, "first_choice.bin", choices.numpy(), "int64"),
        ],
    }
    write_json(folder / "function.json", described)
    print(f"function {folder.name}: {choices.tolist()}", flush=True)


def make(out: str, large: bool = False, only: str = "") -> None:
    """Writes the vectors into the new directory `out`. `only`: just the cases whose names contain
    this word. `large`: also two layers at Llama 4 Maverick's sizes (1.8 GB of files) and the argmax
    over the full vocabulary."""
    out = pathlib.Path(out)
    out.mkdir()  # fails if it exists
    # Later directories are made without their parents (a damaged copy's by copytree, which first
    # reads its source below `out`), so that a run whose `out` has been removed stops and does not
    # make it anew.
    for part in ("cases", "functions", "parameters", "settings", "tables"):
        (out / part).mkdir()
    write_tables(out, Parameters.read(FULL_SIZE))
    # The library's theorems are stated at settings that pass its test settingsAreValid. The
    # full-size parameter file's are meant to pass it. The small example's have other sizes, and
    # the third file has the full sizes with one shift, kd, above the 19 that the test allows.
    full, small = settings_of(Parameters.read(FULL_SIZE)), settings_of(Parameters.read(SMALL))
    out_of_range = json.loads(json.dumps(full))
    out_of_range["layers"][0]["kd"] = 20
    for name, settings, valid in (
        ("target_full_f2", full, True),
        ("target_full_f2.kd-20", out_of_range, False),
        ("target_small_f2", small, False),
    ):
        described = {"format": FORMAT, "settingsAreValid": valid, "settings": settings}
        write_json(out / "settings" / f"{name}.json", described)
    small_cases(out, lambda name: only in name)
    seeded_cases(out, lambda name: only in name)
    upper_end_cases(out, lambda name: only in name)
    if large:
        if only in "maverick-sizes.2-layers.3-tokens":
            large_case(out)
        if only in "first_argmax":
            first_argmax_at_full_width(out)
