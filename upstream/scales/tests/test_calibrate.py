#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests of scales/float_forward.py and scales/calibrate.py.

usage: [SMALL_MODEL=<folder>] python3 scales/tests/test_calibrate.py

float_forward.py is a Llama 4 text model's float32 forward pass, and calibrate.py runs it to fit the
model's quantization scales.

Most cases need no model file. They work on a tiny model with pseudo-random weights (RandomWeights):

- The forward pass is compared at each site (a named point in it) with reference(), a second
  evaluation in numpy and binary64 that calls nothing of float_forward.py. It uses complex numbers
  for the rotary encoding. Like the pass, it follows the steps of the text model of Hugging Face's library transformers,
  version 5.17.0 (src/transformers/models/llama4/modeling_llama4.py; Apache License, version 2.0), and its rotary step
  is adapted from that file's apply_rotary_emb. The text at the top of float_forward.py has the full credit.
- calibrate.py is run on the tiny model, written as a GGUF file with the public package gguf, and
  on faulty inputs.

If the environment variable SMALL_MODEL names a folder written by the test tool make_model
(prover/tests/weights_blocks/make_model.cpp), called as
"make_model <folder> $(cat prover/examples/target_small_file_f2/model.txt)", one more case fits
that model's scales and compares them with the example configuration
prover/examples/target_small_file_f2.

The cases write only below temporary folders, which main() removes.

Exit status 0: every case passed. 1: a case failed. 77: the cases that ran passed, and the last
line says what did not run. If nothing ran, that line begins with "not run" (a Python package is
missing) or with "stood aside": the temporary folders would lie below a folder that another
account may be able to change, and calibrate.py rejects every --out below such a folder.

RandomWeights and reference() can be imported and used without the rest.
"""

import contextlib
import inspect
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile

try:
    import gguf
    import numpy
    import torch
except ImportError as fault:
    if __name__ != "__main__":
        raise
    print(f"not run: python3 lacks the package {fault.name} here")
    sys.exit(77)

SCALES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.dirname(SCALES)
sys.path.insert(0, SCALES)
sys.dont_write_bytecode = True  # the test leaves no __pycache__ folder in the tree; its child processes get -B

import calibrate  # noqa: E402
import fit_scales  # noqa: E402
import float_forward  # noqa: E402

# The exit statuses that the cases expect of calibrate.py. INADMISSIBLE: files written, a scale out
# of range. FAULT: the run failed.
DONE, INADMISSIBLE, FAULT = 0, 1, 2

# The tiny model's shape, with the keys of gguf_weights.shape_words(): d width, dh head dimension,
# L layers, H query heads, G key and value heads, f and fe hidden widths (dense layer, one expert),
# NE routed experts, V vocabulary size. `moe` (layers with experts) and `rope` (layers with rotary
# encoding) are what shape_words() gives for four layers.
TINY = {"d": 32, "dh": 8, "L": 4, "H": 4, "G": 2, "f": 48, "fe": 24, "NE": 4, "V": 20, "eps": 1e-05, "theta": 500000.0,
        "moe": [1, 3], "rope": [0, 1, 2]}  # fmt: skip
DENSE = {**TINY, "fe": 0, "NE": 0, "moe": []}  # the same without experts


def tensor_shapes(shape):
    """Return {name: dimensions in torch's order} for all tensors of a model of this shape."""
    d, f, fe, experts = shape["d"], shape["f"], shape["fe"], shape["NE"]
    queries, keys = shape["H"] * shape["dh"], shape["G"] * shape["dh"]
    shapes = {"token_embd.weight": (shape["V"], d), "output_norm.weight": (d,), "output.weight": (shape["V"], d)}
    for layer in range(shape["L"]):
        kinds = {"attn_norm": (d,), "attn_q": (queries, d), "attn_k": (keys, d), "attn_v": (keys, d),
                 "attn_output": (d, queries), "ffn_norm": (d,)}  # fmt: skip
        if layer in shape["moe"]:
            kinds.update({"ffn_gate_inp": (experts, d), "ffn_gate_shexp": (fe, d), "ffn_up_shexp": (fe, d),
                          "ffn_down_shexp": (d, fe), "ffn_gate_exps": (experts, fe, d), "ffn_up_exps": (experts, fe, d),
                          "ffn_down_exps": (experts, d, fe)})  # fmt: skip
        else:
            kinds.update({"ffn_gate": (f, d), "ffn_up": (f, d), "ffn_down": (d, f)})
        shapes.update({f"blk.{layer}.{kind}.weight": dims for kind, dims in kinds.items()})
    return shapes


class RandomWeights:
    """Pseudo-random weights of a given shape, with names() and tensor() as in gguf_weights.Model.

    A matrix entry is normal with the variance 1 / (number of inputs), an embedding entry is
    standard normal, and a norm's gain is near 1. `arrays` has the weights as float32 numpy arrays.
    `asked` lists the calls of tensor() as (name, expert), and `devices` is the set of devices that
    they named.
    """

    def __init__(self, shape, seed):
        draw = numpy.random.default_rng(seed).standard_normal
        self.arrays, self.asked, self.devices = {}, [], set()
        for name, dims in sorted(tensor_shapes(shape).items()):
            if len(dims) == 1:
                values = 1 + 0.1 * draw(dims)
            else:
                values = draw(dims) if name == "token_embd.weight" else draw(dims) / math.sqrt(dims[-1])
            self.arrays[name] = values.astype(numpy.float32)

    def names(self):
        return sorted(self.arrays)

    def tensor(self, name, expert=None, device="cpu"):
        self.asked.append((name, expert))
        self.devices.add(str(device))
        array = self.arrays[name] if expert is None else self.arrays[name][expert]
        return torch.from_numpy(array.copy()).to(device)


def reference(weights, shape, window, period=8192):
    """Evaluate one window of token ids in binary64.

    The result is {(site, layer): array}, with one row per token, and {layer: each token's routed
    expert}. The final norm's output, with its gain, is under the key ("result", L). `period` is the
    number of positions after which the query factor, which scales the queries of layers without
    rotary encoding, rises.
    """
    d, dh, heads, groups = shape["d"], shape["dh"], shape["H"], shape["G"]
    tokens = len(window)
    sites, experts = {}, {}

    def norm(rows):
        return numpy.stack([row / math.sqrt(sum(value * value for value in row) / d + shape["eps"]) for row in rows])

    def block(z, gate, up, down):
        g, u = gate @ z, up @ z
        return down @ (g / (1 + numpy.exp(-g)) * u), g, u

    x = weights.arrays["token_embd.weight"].astype(numpy.float64)[window]
    for layer in range(shape["L"]):

        def w(kind):
            return weights.arrays[f"blk.{layer}.{kind}.weight"].astype(numpy.float64)

        sites["n1", layer] = norm(x)
        a = sites["n1", layer] * w("attn_norm")
        q = (a @ w("attn_q").T).reshape(tokens, heads, dh)
        k = (a @ w("attn_k").T).reshape(tokens, groups, dh)
        v = (a @ w("attn_v").T).reshape(tokens, groups, dh)
        for t in range(tokens):
            if layer in shape["rope"]:
                turn = numpy.exp(1j * t * shape["theta"] ** (-numpy.arange(0, dh, 2) / dh))
                for rows in (q, k):
                    pairs = (rows[t, :, 0::2] + 1j * rows[t, :, 1::2]) * turn
                    rows[t, :, 0::2], rows[t, :, 1::2] = pairs.real, pairs.imag
            else:
                q[t] *= 1 + 0.1 * math.log(1 + (t + 1) // period)
        sites["q", layer], sites["k", layer], sites["v", layer] = q, k, v
        o = numpy.zeros((tokens, heads, dh))
        for t in range(tokens):
            for head in range(heads):
                group = head // (heads // groups)
                shares = numpy.exp([q[t, head] @ k[j, group] / math.sqrt(dh) for j in range(t + 1)])
                o[t, head] = sum(share * v[j, group] for j, share in enumerate(shares)) / shares.sum()
        x = x + o.reshape(tokens, heads * dh) @ w("attn_output").T

        sites["n2", layer] = norm(x)
        h = sites["n2", layer] * w("ffn_norm")
        rows = {site: [] for site in ("out", "gate", "u", "router", "gate_routed", "uR")}
        experts[layer] = []
        for z in h:
            if layer in shape["moe"]:
                out, g, u = block(z, w("ffn_gate_shexp"), w("ffn_up_shexp"), w("ffn_down_shexp"))
                logits = w("ffn_gate_inp") @ z
                expert = min(e for e in range(shape["NE"]) if logits[e] == max(logits))
                scaled = z / (1 + math.exp(-logits[expert]))
                routed, g_routed, u_routed = block(scaled, w("ffn_gate_exps")[expert], w("ffn_up_exps")[expert],
                                                   w("ffn_down_exps")[expert])  # fmt: skip
                out = out + routed
                experts[layer].append(expert)
                for site, row in (("router", logits), ("gate_routed", g_routed), ("uR", u_routed)):
                    rows[site].append(row)
            else:
                out, g, u = block(z, w("ffn_gate"), w("ffn_up"), w("ffn_down"))
            for site, row in (("out", out), ("gate", g), ("u", u)):
                rows[site].append(row)
        x = x + numpy.stack(rows.pop("out"))
        sites.update({(site, layer): numpy.stack(of_site) for site, of_site in rows.items() if of_site})
        sites["stream", layer] = x
    sites["nF", shape["L"]] = norm(x)
    sites["result", shape["L"]] = norm(x) * weights.arrays["output_norm.weight"].astype(numpy.float64)
    return sites, experts


def reference_of_all(weights, shape, windows, period=8192):
    """Return {(site, layer): array} for several windows, rows in float_forward.forward()'s order.

    At the two sites of the routed experts the rows are grouped by expert, in ascending order. At
    every other site the first dimension counts the windows.
    """
    single = [reference(weights, shape, window, period) for window in windows]
    together = {}
    for place in single[0][0]:
        stacked = numpy.stack([sites[place] for sites, _ in single])
        if place[0] in ("gate_routed", "uR"):
            chosen = numpy.array([experts[place[1]] for _, experts in single]).reshape(-1)
            stacked = stacked.reshape(len(chosen), -1)[numpy.argsort(chosen, kind="stable")]
        together[place] = stacked
    return together


def run_forward(weights, shape, windows, **options):
    """Run float_forward.forward(). Return {(site, layer): all reported rows, as one binary64
    array}, with its result under the key ("result", L)."""
    seen = {}

    def hook(site, layer, tensor):
        seen.setdefault((site, layer), []).append(tensor.clone())

    seen["result", shape["L"]] = [float_forward.forward(weights, shape, windows, hook, **options)]
    return {place: torch.cat(parts).double().cpu().numpy() for place, parts in seen.items()}


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def check_same_sites(seen, expected):
    check(set(seen) == set(expected), f"the hook saw the places {sorted(seen)}, not {sorted(expected)}")
    for place, rows in expected.items():
        check(seen[place].shape == rows.shape, f"{place}: shape {seen[place].shape}, not {rows.shape}")
        check(numpy.allclose(seen[place], rows, rtol=1e-4, atol=1e-5), f"{place}: the values differ")


def some_windows(shape, windows, tokens, seed=5):
    return numpy.random.default_rng(seed).integers(0, shape["V"], (windows, tokens), dtype="int32")


# The cases on the forward pass.


def forward_as_the_reference():
    # The third shape's epsilon and rotary angles are large enough that an error in their use shows.
    for shape in (TINY, DENSE, {**TINY, "eps": 0.25, "theta": 100.0}):
        weights, windows = RandomWeights(shape, 1), some_windows(shape, 3, 7)
        expected = reference_of_all(weights, shape, windows)
        for chunk in (1, 2, 3, 100):
            check_same_sites(run_forward(weights, shape, windows, chunk=chunk), expected)
        check_same_sites(run_forward(weights, shape, windows), expected)


def forward_query_factor():
    check(float_forward.QUERY_FACTOR_PERIOD == 8192 and float_forward.QUERY_FACTOR_SLOPE == 0.1,
          "the query factor's constants are not 8,192 and 0.1")  # fmt: skip
    # The factor is 1 up to position 8,190, so the case shortens its period.
    weights, windows = RandomWeights(TINY, 2), some_windows(TINY, 2, 8)
    float_forward.QUERY_FACTOR_PERIOD = 3
    try:
        seen = run_forward(weights, TINY, windows)
    finally:
        float_forward.QUERY_FACTOR_PERIOD = 8192
    check_same_sites(seen, reference_of_all(weights, TINY, windows, period=3))
    check(not numpy.allclose(seen["q", 3], reference_of_all(weights, TINY, windows)["q", 3], rtol=1e-4, atol=1e-5),
          "negative control: the shorter period changes nothing")  # fmt: skip


def forward_one_window():
    weights, windows = RandomWeights(TINY, 3), some_windows(TINY, 1, 6)
    one, two = run_forward(weights, TINY, windows[0]), run_forward(weights, TINY, torch.from_numpy(windows).long())
    check_same_sites(one, two)  # a window as an array of one dimension and as an array of two dimensions
    check_same_sites(run_forward(weights, TINY, [3]), reference_of_all(weights, TINY, [[3]]))  # one token


def forward_router_tie():
    weights, windows = RandomWeights(TINY, 4), some_windows(TINY, 2, 5)
    for layer in TINY["moe"]:
        weights.arrays[f"blk.{layer}.ffn_gate_inp.weight"][:] = 0  # every logit is 0
    check_same_sites(run_forward(weights, TINY, windows), reference_of_all(weights, TINY, windows))
    check({expert for name, expert in weights.asked if name.endswith("_exps.weight")} == {0},
          "among equal logits another expert than the first was chosen")  # fmt: skip


def forward_reads_what_it_says():
    for shape in (TINY, DENSE):
        stated = float_forward.tensor_shapes(shape)
        check({**stated, "output.weight": (shape["V"], shape["d"])} == tensor_shapes(shape),
              "float_forward.tensor_shapes does not give the model's tensors without output.weight")  # fmt: skip
        weights = RandomWeights(shape, 1)
        run_forward(weights, shape, some_windows(shape, 5, 9), chunk=2)
        check({name for name, _ in weights.asked} == set(stated), "the pass does not read what tensor_shapes says")
        check(len(set(weights.asked)) == len(weights.asked), "the pass read a tensor twice")
        check(all((expert is not None) == name.endswith("_exps.weight") for name, expert in weights.asked),
              "a tensor of routed experts was read whole")  # fmt: skip
        layers = [int(name.split(".")[1]) for name, _ in weights.asked if name.startswith("blk.")]
        check(layers == sorted(layers), "the pass went back to an earlier layer's weights")


def forward_device():
    # Without a CUDA device the part on cuda is skipped without a message.
    weights, windows = RandomWeights(TINY, 1), some_windows(TINY, 3, 7)
    check(inspect.signature(float_forward.forward).parameters["chunk"].default == 4, "the default of chunk is not 4")
    run_forward(weights, TINY, windows, device="cpu:0")
    check(weights.devices == {"cpu:0"}, f"with the device cpu:0 the weights were asked for on {weights.devices}")
    if torch.cuda.is_available():
        weights = RandomWeights(TINY, 1)
        check_same_sites(run_forward(weights, TINY, windows, device="cuda"), reference_of_all(weights, TINY, windows))
        check(weights.devices == {"cuda"}, f"with the device cuda the weights were asked for on {weights.devices}")


def forward_rejections():
    weights = RandomWeights(TINY, 1)
    faulty = {
        "an id equal to V": [[0, TINY["V"]]],
        "a negative id": [[0, -1]],
        "ids that are floating-point numbers": [[0.0, 1.0]],
        "ids that are truth values": [[True, False]],
        "no token": numpy.zeros((2, 0), dtype="int32"),
        "three dimensions": numpy.zeros((1, 2, 2), dtype="int32"),
        "a window of 8,193 tokens": numpy.zeros((1, 8193), dtype="int32"),
    }
    for what, tokens in faulty.items():
        try:
            run_forward(weights, TINY, tokens)
        except float_forward.ForwardError:
            continue
        check(False, f"{what}: no ForwardError")
    check(tuple(float_forward.as_windows(numpy.zeros((1, 8192), dtype="uint8"), TINY).shape) == (1, 8192),
          "a window of 8,192 tokens is not accepted")  # fmt: skip
    check(float_forward.as_windows(numpy.asfortranarray(some_windows(TINY, 3, 7)), TINY).is_contiguous(),
          "ids in Fortran order keep their memory order, which can change the last bits of the pass")  # fmt: skip


# The cases on calibrate.py. Each gets a new folder.


def write_model(folder, weights, shape, layers=None, double=()):
    """Write the weights as a one-file GGUF model with the metadata keys of gguf_weights.SHAPE_KEYS.

    `layers` replaces the number of layers in the metadata. The tensors named in `double` are written
    as F64, a type that gguf_weights.py does not read, and all others as F32.
    """
    os.mkdir(folder)
    writer = gguf.GGUFWriter(os.path.join(folder, "tiny.gguf"), "llama4")
    writer.add_uint32("llama4.embedding_length", shape["d"])
    writer.add_uint32("llama4.block_count", layers or shape["L"])
    writer.add_uint32("llama4.attention.key_length", shape["dh"])
    writer.add_float32("llama4.attention.layer_norm_rms_epsilon", shape["eps"])
    writer.add_float32("llama4.rope.freq_base", shape["theta"])
    for name in weights.names():
        writer.add_tensor(name, weights.arrays[name].astype("float64" if name in double else "float32"))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


EXPONENT, EMBEDDING_BITS = 12, 9  # `write_sums`: a matrix's default exponent, the embedding's bits


def write_sums(path, shape, e_res=11, other={}, without=()):
    """Write the sums file (the matrices' exponents and bit widths) of a model of this shape.

    e_res is the embedding's exponent, `other` gives single tensors another exponent, and tensors
    named in `without` get no line. calibrate.py does not read the SHA-256 sums, so they are zeros.
    """
    lines = []
    for name, dims in sorted(tensor_shapes(shape).items()):
        if len(dims) > 1 and name not in without:
            exponent, bits = other.get(name, EXPONENT), 13
            if name == "token_embd.weight":
                exponent, bits = e_res, EMBEDDING_BITS
            lines.append(f"{name} {math.prod(dims)} {exponent} {bits} {'0' * 64}")
    with open(path, "w") as file:
        file.write("".join(line + "\n" for line in lines + [f"END {len(lines)}"]))


def inputs(folder, shape=TINY):
    """Write a model, its sums file and a token file into the folder.

    Return the weights, the windows, calibrate.py's options.
    """
    weights, windows = RandomWeights(shape, 7), some_windows(shape, 3, 9)
    write_model(f"{folder}/model", weights, shape)
    write_sums(f"{folder}/wsums.txt", shape)
    numpy.save(f"{folder}/tokens.npy", windows)
    options = {"--gguf": f"{folder}/model", "--wsums": f"{folder}/wsums.txt", "--tokens": f"{folder}/tokens.npy",
               "--out": f"{folder}/out"}  # fmt: skip
    return weights, windows, options


def words_of(options):
    return [str(word) for pair in options.items() for word in pair]


def call(options):
    """Call calibrate.main() in this process.

    Return the exit status, standard output and standard error.
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = calibrate.main(options if isinstance(options, list) else words_of(options))
        except SystemExit as ending:
            status = ending.code
    return status, out.getvalue(), err.getvalue()


def check_done(options, what, status=DONE):
    """Check a call's exit status and that standard error is empty. Return standard output."""
    got, out, err = call(options)
    check(got == status and not err, f"{what}: status {got}, not {status}\n{out}{err}")
    return out


def read_outputs(folder):
    with open(f"{folder}/maxima.json") as maxima, open(f"{folder}/words.json") as words:
        return json.load(maxima), json.load(words)


def check_same_outputs(one, two, what):
    """Check that two folders have the same scales (words.json) and maxima (maxima.json).

    The maxima (largest absolute activation per site and layer) are compared up to floating-point
    rounding, because a linear-algebra library may sum in another order in a second call.
    """
    (maxima, words), (other_maxima, other_words) = read_outputs(one), read_outputs(two)
    check(words == other_words, f"{what}: words.json differs")
    check(set(maxima) == set(other_maxima), f"{what}: the sites differ")
    for site, of_site in maxima.items():
        check(set(of_site) == set(other_maxima[site]), f"{what}: the layers of {site} differ")
        for layer, value in of_site.items():
            check(math.isclose(value, other_maxima[site][layer], rel_tol=1e-4), f"{what}: {site}, {layer} differs")


def check_rejected(options, what, says="", after_the_first_line=False):
    """Check a rejection: status 2, one line on standard error containing `says`, no folder --out.

    Standard output must be empty, or non-empty if after_the_first_line: calibrate.py finds such a
    fault only after it has printed its first line.
    """
    status, out, err = call(options)
    check(status == FAULT, f"{what}: status {status}, not {FAULT}\n{out}{err}")
    check(err.startswith("calibrate.py: ") and err.count("\n") == 1, f"{what}: standard error is not one line\n{err}")
    check(says in err, f"{what}: the message lacks {says!r}\n{err}")
    check(bool(out) == after_the_first_line, f"{what}: standard output is {out!r}")
    check(not os.path.exists(options["--out"]), f"{what}: the folder --out was made")


# calibrate.py as a child process: its exit status, its two output files, what it prints.
def program_on_the_tiny_model(folder):
    weights, windows, options = inputs(folder)
    command = [sys.executable, "-B", os.path.join(SCALES, "calibrate.py")] + words_of(options)
    done = subprocess.run(command, capture_output=True, text=True, cwd=folder)
    check(done.returncode == DONE and not done.stderr, f"status {done.returncode}\n{done.stdout}{done.stderr}")
    check(sorted(os.listdir(options["--out"])) == ["maxima.json", "words.json"], "other files than the two")
    for name in os.listdir(options["--out"]):
        check(stat.S_IMODE(os.stat(f"{options['--out']}/{name}").st_mode) == 0o600, f"{name} does not have mode 600")
    maxima, words = read_outputs(options["--out"])

    # maxima.json against reference(). At q, calibrate.py stores the maximum divided by sqrt(dh).
    expected = {}
    for (site, layer), rows in reference_of_all(weights, TINY, windows).items():
        if site in ("n1", "n2", "q", "k", "v", "u", "uR", "nF"):
            expected.setdefault(site, {})[str(layer)] = abs(rows).max() / (math.sqrt(TINY["dh"]) if site == "q" else 1)
    check(set(maxima) == set(expected), f"the sites are {sorted(maxima)}")
    for site, of_site in expected.items():
        check(set(maxima[site]) == set(of_site), f"{site}: the layers are {sorted(maxima[site])}")
        for layer, value in of_site.items():
            check(math.isclose(maxima[site][layer], value, rel_tol=1e-4), f"the maximum at {site}, {layer} differs")

    # words.json: "KL" has each layer's scales, and "W" the values that all layers share. They must
    # be the fit of maxima.json, with eps rounded to float32, as the GGUF file stores it.
    check(set(words) == {"KL", "W", "ew", "emb_bits"}, f"the members of words.json are {sorted(words)}")
    matrices = sorted(name for name, dims in tensor_shapes(TINY).items() if len(dims) > 1)
    matrices.remove("token_embd.weight")
    check(words["ew"] == {name: EXPONENT for name in matrices}, '"ew" is not the sums file\'s exponents')
    check(words["emb_bits"] == EMBEDDING_BITS, '"emb_bits" is not the width in the sums file')
    exponents = {"output": EXPONENT}
    for name in matrices:
        if name.startswith("blk."):
            exponents.setdefault(name.split(".")[2], {})[name.split(".")[1]] = EXPONENT
    as_in_the_file = {**TINY, "eps": float(numpy.float32(TINY["eps"]))}
    fitted, _ = fit_scales.fit(as_in_the_file, maxima, exponents)
    check(words["KL"] == fitted["KL"] and words["W"] == fitted["W"], "the scales are not the fit of maxima.json")

    # calibrate.py is documented to print no weight, activation or fitted scale. Only the maxima are
    # searched for in its output, by the first six characters of their repr().
    for of_site in maxima.values():
        for value in of_site.values():
            check(repr(value)[:6] not in done.stdout, f"the program printed a maximum\n{done.stdout}")


def program_forms_of_the_call(folder):
    """The token file's two forms and integer types, --chunk, --device, argparse's option forms."""
    _, windows, options = inputs(folder)
    out = check_done(options, "the plain call").splitlines()
    first = "shards: 1, layers: 4, layers with experts: 2, width: 32, windows: 3, tokens per window: 9, device: cpu"
    check(out[0] == first, f"the first line is {out[0]!r}")
    check(out[1].startswith("wrote 27 maxima "), f"the closing line is {out[1]!r}")  # 6 L + layers with experts + 1
    windows.astype("<i4").tofile(f"{folder}/tokens.raw")
    with open(f"{folder}/tokens.more", "wb") as file:
        file.write(windows.astype("<i4").tobytes() + b"\x01\x02\x03")
    numpy.save(f"{folder}/wide.npy", windows.astype(">i8"))
    numpy.save(f"{folder}/narrow.npy", windows.astype("uint8"))
    numpy.save(f"{folder}/columns.npy", numpy.asfortranarray(windows))
    same = {
        "a raw file": {"--tokens": f"{folder}/tokens.raw", "--window": 9},
        "a raw file with three bytes more": {"--tokens": f"{folder}/tokens.more", "--window": 9},
        "a .npy file with --window": {"--window": 5},
        "big-endian 64-bit ids": {"--tokens": f"{folder}/wide.npy"},
        "unsigned 8-bit ids": {"--tokens": f"{folder}/narrow.npy"},
        "an array in Fortran order": {"--tokens": f"{folder}/columns.npy"},
        "--device cpu, --eres 11, --chunk 4": {"--device": "cpu", "--eres": 11, "--chunk": 4},
        "--chunk 1": {"--chunk": 1},
        "--chunk 2": {"--chunk": 2},
        "--chunk 100": {"--chunk": 100},
    }
    for number, (what, more) in enumerate(same.items()):
        check_done({**options, **more, "--out": f"{folder}/same{number}"}, what)
        check_same_outputs(options["--out"], f"{folder}/same{number}", what)
    passed_on, forward = {}, float_forward.forward

    def recording(*arguments, **more):
        passed_on.update(more)
        return forward(*arguments, **more)

    float_forward.forward = recording
    try:
        check_done({**options, "--out": f"{folder}/defaults"}, "the plain call")
        check(passed_on == {"device": "cpu", "chunk": 4}, f"without --device and --chunk the pass got {passed_on}")
        check_done({**options, "--device": "cpu:0", "--chunk": 2, "--out": f"{folder}/passed"}, "--device cpu:0")
        check(passed_on == {"device": "cpu:0", "chunk": 2}, f"with --device cpu:0, --chunk 2 the pass got {passed_on}")
    finally:
        float_forward.forward = forward
    short = [f"--gguf={options['--gguf']}", "--ws", options["--wsums"], "--tok", options["--tokens"], "--eres=11"]
    check_done(short + ["--out", f"{folder}/short"], "options with an equals sign and shortened options")
    check_same_outputs(options["--out"], f"{folder}/short", "options with an equals sign and shortened options")
    status, out, err = call(["--help"])
    check(status == DONE and "--chunk" in out and not err, f"--help: status {status}\n{out}{err}")


# --eres must equal e_res, the residual stream's exponent, at which the sums file was made. The
# file states it as the embedding's exponent.
def program_other_e_res(folder):
    _, _, options = inputs(folder)
    check_rejected({**options, "--eres": 10}, "a sums file at 11 and --eres 10", "--eres 11 here")
    write_sums(options["--wsums"], TINY, e_res=10)
    check_rejected(options, "a sums file at 10 and no --eres", "--eres 10 here")
    check_done({**options, "--eres": 10}, "a sums file at 10 and --eres 10")
    check(read_outputs(options["--out"])[1]["W"]["e_res"] == 10, "words.json does not state e_res = 10")


def program_inadmissible_scale(folder):
    _, _, options = inputs(folder)
    write_sums(options["--wsums"], TINY, other={"blk.2.attn_v.weight": -40})
    out = check_done(options, "an exponent of -40", INADMISSIBLE)
    check("layer 2" in out and "kv" in out, f"the place is not printed\n{out}")
    check(out.splitlines()[-1].startswith("words_to_policy.py will reject"), f"the last line is not the warning\n{out}")
    check(read_outputs(options["--out"])[1]["KL"][2]["kv"] < 1, "words.json does not hold the inadmissible scale")


# calibrate.py must overwrite nothing: if --out has maxima.json or words.json (or a dangling
# symbolic link at words.json), the call fails, and the existing files stay as they were.
# A new --out must get mode 700. If other accounts can write in an existing --out (mode 777 here),
# the call must fail and write nothing into it, and its message must not advise chmod, which would
# leave any links that they have already put there. A new --out below that folder must fail too.
def program_output_folder(folder):
    _, _, options = inputs(folder)
    os.mkdir(options["--out"], 0o700)
    with open(f"{options['--out']}/other", "w") as file:
        file.write("kept")
    check_done(options, "a folder that holds another file")
    before = read_outputs(options["--out"])
    for name, other in (("maxima.json", "words.json"), ("words.json", "maxima.json")):
        os.rename(f"{options['--out']}/{other}", f"{folder}/aside")
        status, out, err = call(options)
        check(status == FAULT and name in err and not out, f"{name} exists: status {status}\n{out}{err}")
        check(sorted(os.listdir(options["--out"])) == sorted([name, "other"]), f"{name} exists: the run wrote a file")
        os.rename(f"{folder}/aside", f"{options['--out']}/{other}")
    check(read_outputs(options["--out"]) == before, "an existing file was changed")
    os.mkdir(f"{folder}/link")
    os.symlink(f"{folder}/absent", f"{folder}/link/words.json")
    status, out, err = call({**options, "--out": f"{folder}/link"})
    check(status == FAULT and not out, f"a dangling link at words.json: status {status}\n{out}{err}")
    check_done({**options, "--out": f"{folder}/a/b/c"}, "a folder whose parents do not exist")
    check(os.stat(f"{folder}/a/b/c").st_mode & 0o777 == 0o700, "the new folder does not have mode 700")
    os.mkdir(f"{folder}/open")
    os.chmod(f"{folder}/open", 0o777)
    status, out, err = call({**options, "--out": f"{folder}/open"})
    check(status == FAULT and "can write in it" in err and not out, f"an open folder: status {status}\n{out}{err}")
    check("chmod" not in err, "an open folder: the refusal advises to close a folder in which links may lie already")
    check(os.listdir(f"{folder}/open") == [], "an open folder: the run wrote into it")
    status, out, err = call({**options, "--out": f"{folder}/open/new"})
    check(status == FAULT and "lies below" in err and not out, f"below an open folder: status {status}\n{out}{err}")
    check(os.listdir(f"{folder}/open") == [], "below an open folder: the run created something there")
    here = os.getcwd()
    os.mkdir(f"{folder}/here")
    os.chdir(f"{folder}/here")
    try:
        check_rejected({**options, "--out": ""}, "--out with the empty word", "--out is empty")
        check(os.listdir(os.curdir) == [], "--out with the empty word: the run wrote into the working folder")
    finally:
        os.chdir(here)
    for what, path in (("a file", options["--tokens"]), ("a path below a file", options["--tokens"] + "/x/y")):
        status, out, err = call({**options, "--out": path})
        check(status == FAULT and err.count("\n") == 1 and not out, f"--out is {what}: status {status}\n{out}{err}")


def program_faulty_tokens(folder):
    _, windows, options = inputs(folder)
    raw = windows.astype("<i4").tobytes()

    def rejected(what, content, name="faulty.npy", **more):
        with open(f"{folder}/{name}", "wb") as file:
            if isinstance(content, bytes):
                file.write(content)
            else:
                numpy.save(file, content)
        check_rejected({**options, "--tokens": f"{folder}/{name}", **more}, what, name)

    rejected("an id equal to V", numpy.full((1, 4), TINY["V"], dtype="int32"))
    rejected("a negative id", numpy.full((1, 4), -1, dtype="int64"))
    rejected("an id of 2^63", numpy.full((1, 4), 2**63, dtype="uint64"))
    rejected("a window of 8,193 tokens", numpy.zeros((1, 8193), dtype="int32"))
    rejected("an array of one dimension", windows[0])
    rejected("an array of three dimensions", windows[None])
    rejected("an array of floating-point numbers", windows.astype("float32"))
    rejected("an array of truth values", windows > 3)
    rejected("an array of objects", windows.astype(object))
    rejected("an array without windows", windows[:0])
    rejected("an array without tokens", windows[:, :0])
    rejected("raw ids under a name that ends in .npy", raw)
    rejected("an empty file under a name that ends in .npy", b"")
    rejected("the start of a zip archive under a name that ends in .npy", b"PK\x03\x04" + raw)
    rejected("a raw file without --window", raw, "faulty.raw")
    rejected("a .npy file under a name that ends in .NPY", windows, "faulty.NPY")
    rejected("--window 0", raw, "faulty.raw", **{"--window": 0})
    rejected("--window -3", raw, "faulty.raw", **{"--window": -3})
    rejected("a raw file that is not whole windows", raw[:-4], "faulty.raw", **{"--window": 9})
    rejected("an empty raw file", b"", "faulty.raw", **{"--window": 9})
    rejected("a raw file with a negative id", b"\xff\xff\xff\xff", "faulty.raw", **{"--window": 1})
    check_rejected({**options, "--tokens": f"{folder}/absent.npy"}, "no token file", "absent.npy")
    check_rejected({**options, "--tokens": f"{folder}/absent", "--window": 9}, "no raw token file", "absent")


def program_faulty_sums(folder):
    _, _, options = inputs(folder)
    faulty = {**options, "--wsums": f"{folder}/faulty.txt"}

    # `how` gets the lines of a valid sums file: lines[-2] is "END <count>", and lines[-1] is empty.
    def change(how):
        write_sums(faulty["--wsums"], TINY)
        with open(faulty["--wsums"]) as file:
            lines = file.read().split("\n")
        with open(faulty["--wsums"], "w") as file:
            file.write("\n".join(how(lines)))

    rejected = {
        "an empty file": lambda lines: [],
        "no line END": lambda lines: lines[:-2] + [""],
        "a line less than END counts": lambda lines: lines[1:],
        "words after the number of END": lambda lines: lines[:-2] + [lines[-2] + " x", ""],
        "a line of four fields": lambda lines: [lines[0].rsplit(" ", 1)[0]] + lines[1:],
        "two blanks between fields": lambda lines: [lines[0].replace(" ", "  ", 1)] + lines[1:],
        "an exponent that is not a number": lambda lines: [lines[0].replace(f" {EXPONENT} ", " x ")] + lines[1:],
        "an exponent with a plus sign": lambda lines: [lines[0].replace(f" {EXPONENT} ", f" +{EXPONENT} ")] + lines[1:],
        "a negative number of bits": lambda lines: [lines[0].replace(" 13 ", " -13 ")] + lines[1:],
        "a name twice": lambda lines: [lines[1]] + lines[1:],
    }
    for what, how in rejected.items():
        change(how)
        check_rejected(faulty, f"a sums file with {what}", "faulty.txt")
    write_sums(faulty["--wsums"], TINY, without=("token_embd.weight",))
    check_rejected(faulty, "a sums file without the embedding", "token_embd.weight")
    check_rejected({**options, "--wsums": f"{folder}/absent"}, "no sums file", "absent")
    # Text that the maker of an input chooses: a repeated name is not printed, and a path is printed with escapes.
    change(lambda lines: ["\x1b[2Jforged " + lines[0].split(" ", 1)[1]] * 2 + lines[2:])
    check_rejected(faulty, "a sums file with a hostile name twice", "tensor line 2 of the sums file")
    check("forged" not in call(faulty)[2], "a name from the sums file is printed")
    odd = {**options, "--wsums": f"{folder}/\x1b[2J\nabsent"}
    check_rejected(odd, "no sums file, and control characters in its path", "\\x1b[2J\\nabsent")
    # calibrate.py looks up the other matrices' lines only after it has printed its first line.
    for name in ("blk.1.ffn_down_exps.weight", "blk.2.ffn_up.weight", "output.weight"):
        write_sums(faulty["--wsums"], TINY, without=(name,))
        check_rejected(faulty, f"a sums file without {name}", name, after_the_first_line=True)

    check_done(options, "the sums file as it is")
    accepted = {
        "empty lines": lambda lines: ["", lines[0], ""] + lines[1:],
        "no final line feed": lambda lines: lines[:-1],
        "the lines in another order": lambda lines: lines[-3::-1] + lines[-2:],
        "other second and fifth fields": lambda lines: [lines[0].split(" ")[0] + f" x {EXPONENT} 13 y"] + lines[1:],
    }
    for number, (what, how) in enumerate(accepted.items()):
        change(how)
        check_done({**faulty, "--out": f"{folder}/out{number}"}, f"a sums file with {what}")
        check_same_outputs(options["--out"], f"{folder}/out{number}", f"a sums file with {what}")
    # A line for a tensor that the model does not have goes into "ew".
    change(lambda lines: ["blk.9.attn_q.weight 1 -7 13 0"] + lines[:-2] + [f"END {len(lines) - 1}", ""])
    check_done({**faulty, "--out": f"{folder}/more"}, "a sums file with a line more")
    check(read_outputs(f"{folder}/more")[1]["ew"]["blk.9.attn_q.weight"] == -7, '"ew" lacks the further line')


def program_faulty_models(folder):
    weights, _, options = inputs(folder)

    def model(name, change, **how):
        changed = RandomWeights(TINY, 7)
        change(changed.arrays)
        write_model(f"{folder}/{name}", changed, TINY, **how)
        return {**options, "--gguf": f"{folder}/{name}"}

    check_rejected({**options, "--gguf": f"{folder}/absent"}, "no folder", "absent")
    check_rejected({**options, "--gguf": options["--tokens"]}, "--gguf names a file")
    os.mkdir(f"{folder}/empty")
    check_rejected({**options, "--gguf": f"{folder}/empty"}, "a folder without GGUF files", "empty")
    os.mkdir(f"{folder}/damaged")
    with open(f"{options['--gguf']}/tiny.gguf", "rb") as whole, open(f"{folder}/damaged/tiny.gguf", "wb") as cut:
        cut.write(whole.read(100))
    check_rejected({**options, "--gguf": f"{folder}/damaged"}, "a GGUF file cut in its metadata")
    os.mkdir(f"{folder}/dense")
    write_model(f"{folder}/dense/model", RandomWeights(DENSE, 7), DENSE)
    check_rejected({**options, "--gguf": f"{folder}/dense/model"}, "a model without experts", "blk.1.ffn_gate")

    for name in ("blk.2.attn_norm.weight", "blk.3.ffn_up_exps.weight", "blk.2.ffn_down.weight", "output_norm.weight"):
        check_rejected(model(f"without.{name}", lambda arrays: arrays.pop(name)), f"a model without {name}", name)

    def narrower(arrays):
        arrays["blk.2.attn_v.weight"] = arrays["blk.2.attn_v.weight"][:-1]

    check_rejected(model("narrower", narrower), "a matrix with a row less", "blk.2.attn_v.weight")
    for what, shape in (("3 key heads for 4 query heads", {**TINY, "G": 3}), ("heads of 7 entries", {**TINY, "dh": 7})):
        os.mkdir(f"{folder}/{what}")
        write_model(f"{folder}/{what}/model", RandomWeights(shape, 7), shape)
        check_rejected({**options, "--gguf": f"{folder}/{what}/model"}, what, "query heads")

    # The remaining faults show only during or after the forward pass, after the first printed line.
    check_rejected(model("double", lambda arrays: None, double=("blk.2.attn_v.weight",)), "a tensor of the type F64",
                   "blk.2.attn_v.weight", after_the_first_line=True)  # fmt: skip
    for value, says in ((math.inf, "blk.1.ffn_down_shexp.weight"), (math.nan, "blk.1.ffn_down_shexp.weight"),
                        (3e38, "the site n1 of layer 2 is not finite")):  # fmt: skip

        def one_row(arrays):
            arrays["blk.1.ffn_down_shexp.weight"][3] = value  # at 3e38 the weights are finite and the stream is not

        check_rejected(model(f"value.{value}", one_row), f"weights of {value}", says, after_the_first_line=True)

    def no_values(arrays):
        arrays["blk.0.attn_v.weight"][:] = 0

    check_rejected(model("zero", no_values), "a matrix of zeros", "the site v of layer 0", after_the_first_line=True)


def program_order_of_the_faults(folder):
    """Check that of two faults calibrate.py reports the one that comes first in `faults`.

    Two faults in the same option cannot be combined, so such pairs are left out.
    """
    _, _, options = inputs(folder)
    os.mkdir(f"{folder}/full")
    os.mkdir(f"{folder}/empty")
    with open(f"{folder}/full/words.json", "w"):
        pass
    write_sums(f"{folder}/at10.txt", TINY, e_res=10)
    lacking = RandomWeights(TINY, 7)
    del lacking.arrays["blk.2.attn_norm.weight"]
    write_model(f"{folder}/lacking", lacking, TINY)
    faults = [
        ({"--eres": 25}, "--eres"),
        ({"--out": f"{folder}/full"}, "words.json exists"),
        ({"--gguf": f"{folder}/empty"}, ".gguf"),
        ({"--wsums": f"{folder}/absent.txt"}, "absent.txt"),
        ({"--wsums": f"{folder}/at10.txt"}, "--eres 10 here"),
        ({"--tokens": f"{folder}/absent.npy"}, "absent.npy"),
        ({"--gguf": f"{folder}/lacking"}, "blk.2.attn_norm.weight"),
    ]
    for first, (one, says) in enumerate(faults):
        for two, _ in faults[first + 1 :]:
            if set(one) != set(two):
                status, out, err = call({**options, **two, **one})
                check(status == FAULT and says in err and not out, f"{one} with {two}: status {status}\n{out}{err}")


def program_wrong_calls(folder):
    _, _, options = inputs(folder)
    wrong = [words_of({name: value for name, value in options.items() if name != option}) for option in options]
    wrong += [words_of(options) + more for more in (["--window", "x"], ["--eg", "7"], ["more"], ["--eres", "x"],
              ["--eres", "0"], ["--eres", "25"], ["--chunk", "0"], ["--chunk", "-1"], ["--chunk", "x"])]  # fmt: skip
    for words in wrong:
        status, out, err = call(words)
        check(status == FAULT and err.startswith("usage: ") and not out, f"{words}: status {status}\n{out}{err}")
        check(not os.path.exists(options["--out"]), f"{words}: the folder --out was made")
    check_rejected({**options, "--device": "nonsense"}, "a device that torch does not know", "nonsense")
    if not torch.cuda.is_available():
        check_rejected({**options, "--device": "cuda"}, "cuda on a machine without a GPU", "cuda")


def program_faults_of_its_own(folder):
    """A failed second write leaves no file; unexpected exceptions and failed imports give status 2.

    Python's own status for these two, 1, would mean that calibrate.py has written both files.
    """
    _, _, options = inputs(folder)
    write, forward = fit_scales.write_private_json, float_forward.forward

    def fails_at_the_second(path, value):
        if path.endswith("words.json"):
            raise OSError(28, "No space left on device")
        write(path, value)

    def fails(*arguments, **more):
        raise RuntimeError("planted\x1b[2J\nforged line")

    try:
        fit_scales.write_private_json = fails_at_the_second
        status, out, err = call(options)
        check(status == FAULT and "No space left" in err, f"a failed write: status {status}\n{out}{err}")
        check(os.listdir(options["--out"]) == [], "a failed write leaves a file")
        fit_scales.write_private_json = write
        float_forward.forward = fails
        status, out, err = call(options)
        check(status == FAULT and "planted" in err, f"an exception in the pass: status {status}\n{out}{err}")
        check(err.endswith("RuntimeError: planted\\x1b[2J\\nforged line\n"), f"not its text, escaped\n{err!a}")
    finally:
        fit_scales.write_private_json, float_forward.forward = write, forward

    os.mkdir(f"{folder}/fake")
    with open(f"{folder}/fake/gguf.py", "w") as file:
        file.write('raise ImportError("planted", name="gguf")\n')
    done = subprocess.run([sys.executable, "-B", os.path.join(SCALES, "calibrate.py")] + words_of(options), text=True,
                          capture_output=True, env={**os.environ, "PYTHONPATH": f"{folder}/fake"})  # fmt: skip
    check(done.returncode == FAULT and done.stderr.count("\n") == 1 and "gguf" in done.stderr and not done.stdout,
          f"a package that cannot be imported: status {done.returncode}\n{done.stdout}{done.stderr}")  # fmt: skip


def program_tensors_that_the_pass_does_not_read(folder):
    weights, _, options = inputs(folder)
    plain = check_done(options, "the plain call")
    check("does not read" not in plain, f"a line on tensors that the pass does not read\n{plain}")

    # Two tensors that float_forward.py does not read: a printed line on them, the same outputs.
    weights.arrays["rope_freqs.weight"] = numpy.full(TINY["dh"] // 2, 8, dtype="float32")
    weights.arrays["blk.0.attn_q_norm.weight"] = numpy.ones(TINY["dh"], dtype="float32")
    write_model(f"{folder}/model2", weights, TINY)
    out = check_done({**options, "--gguf": f"{folder}/model2", "--out": f"{folder}/out2"}, "two further tensors")
    check("does not read: 2, the first of them blk.0.attn_q_norm.weight" in out, f"no line on the two tensors\n{out}")
    check_same_outputs(options["--out"], f"{folder}/out2", "two further tensors")

    # The maker of a file chooses its names. A long one with control characters is printed escaped and cut.
    weights.arrays["\x1b[2J\r\nforged line" + "w" * 500] = numpy.ones(TINY["dh"], dtype="float32")
    write_model(f"{folder}/model4", weights, TINY)
    out = check_done({**options, "--gguf": f"{folder}/model4", "--out": f"{folder}/out4"}, "a hostile name")
    check("does not read: 3, the first of them \\x1b[2J\\r\\nforged line" + "w" * 63 + " ... (cut). If one" in out,
          f"the hostile name is not printed with escapes and cut to 80 characters\n{out!a}")  # fmt: skip
    check("\x1b" not in out and "\r" not in out and "\nforged" not in out, f"a control character is printed\n{out!a}")

    # With one layer in the metadata the model has no layer with experts, and so no site uR.
    write_model(f"{folder}/model3", RandomWeights(TINY, 7), TINY, layers=1)
    out = check_done({**options, "--gguf": f"{folder}/model3", "--out": f"{folder}/out3"}, "one layer")
    maxima, words = read_outputs(f"{folder}/out3")
    check(sorted(maxima) == ["k", "n1", "n2", "nF", "q", "u", "v"], f"one layer: the sites are {sorted(maxima)}")
    check(list(maxima["nF"]) == ["1"] and len(words["KL"]) == 1, "one layer: nF is not under 1, or KL has more layers")
    check(words["ew"] == read_outputs(options["--out"])[1]["ew"], 'one layer: "ew" is not the whole sums file')


def program_on_the_small_model(folder):
    """Fit the scales of the model in SMALL_MODEL and compare them with an example configuration's.

    The scales of the example, prover/examples/target_small_file_f2, were fitted to other tokens.
    With the tokens of this case all scales come out as the example's except kq, kk and sh (the
    shifts of queries, keys and attention scores), which follow the sizes of queries and keys.
    Other tokens can move other scales too.
    """
    model, example = os.environ["SMALL_MODEL"], os.path.join(TREE, "prover", "examples", "target_small_file_f2")
    windows = numpy.random.default_rng(7).integers(0, 2048, (2, 16), dtype="int32")
    numpy.save(f"{folder}/tokens.npy", windows)
    options = {"--gguf": model, "--wsums": f"{model}/wsums.txt", "--tokens": f"{folder}/tokens.npy", "--chunk": 1,
               "--out": f"{folder}/fit"}  # fmt: skip
    for command in (
        ["calibrate.py"] + words_of(options),
        ["words_to_policy.py", f"{example}/policy.txt", f"{folder}/fit/words.json", f"{model}/wsums.txt",
         f"{folder}/fit/policy.txt", f"{folder}/fit/KL.bin"],  # fmt: skip
    ):
        done = subprocess.run([sys.executable, "-B", os.path.join(SCALES, command[0])] + command[1:], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # fmt: skip
        check(done.returncode == DONE, f"{command[0]}: status {done.returncode}\n{done.stdout}")

    maxima, words = read_outputs(f"{folder}/fit")
    counts = {site: len(of_site) for site, of_site in maxima.items()}
    check(counts == {"n1": 4, "n2": 4, "q": 4, "k": 4, "v": 4, "u": 4, "uR": 2, "nF": 1}, f"the maxima are {counts}")
    with open(f"{model}/wsums.txt") as file:
        check(len(words["ew"]) == len(file.read().splitlines()) - 2, '"ew" does not have an entry for each matrix')

    def scales(path):
        with open(path, "rb") as file:
            data = file.read()
        return struct.unpack(f"<{len(data) // 4}I", data)

    fitted, given = scales(f"{folder}/fit/KL.bin"), scales(f"{example}/KL.bin")
    check(len(fitted) == len(given), f"KL.bin has {len(fitted)} words, and the example's has {len(given)}")
    # KL.bin (32-bit little-endian words) has the 13 scales listed in `names` for each of the four
    # layers in turn, then three more scales (kr, kgR, kuR) for each of the two layers with experts.
    names = "kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp".split()
    for at, (one, two) in enumerate(zip(fitted, given)):
        name = names[at % 13] if at < 4 * 13 else "kr kgR kuR".split()[(at - 4 * 13) % 3]
        check(one == two or name in ("kq", "kk", "sh"), f"word {at} of KL.bin ({name}) is not the example's")

    def lines(path):
        with open(path) as file:
            return dict(line.split(" ", 1) for line in file.read().splitlines())

    # Of policy.txt, the final norm's constants (MF, KF) and the norms' epsilons are compared. The
    # line s, the logits' exponent, is not: in the example it is not fitted to the model's weights.
    fitted, given = lines(f"{folder}/fit/policy.txt"), lines(f"{example}/policy.txt")
    for name in ("MF", "KF", "eps1_lo", "eps1_hi", "eps2_lo", "eps2_hi", "epsF_lo", "epsF_hi"):
        check(fitted[name] == given[name], f"the line {name} of policy.txt is not the example's")

    windows.astype("<i4").tofile(f"{folder}/tokens.raw")
    raw = {**options, "--tokens": f"{folder}/tokens.raw", "--window": 16, "--chunk": 2, "--out": f"{folder}/raw"}
    check_done(raw, "a raw token file and --chunk 2")
    check(read_outputs(f"{folder}/raw")[1] == words, "a raw token file and --chunk 2 give another words.json")
    check(call(options)[0] == FAULT, "existing output files are not a fault")
    check_rejected({**raw, "--window": 5, "--out": f"{folder}/new"}, "raw ids that are not whole windows")
    check_rejected({**options, "--eres": 10, "--out": f"{folder}/new"}, "--eres 10 on sums made at 11")


def main():
    cases = [forward_as_the_reference, forward_query_factor, forward_one_window, forward_router_tie,
             forward_reads_what_it_says, forward_device, forward_rejections, program_on_the_tiny_model,
             program_forms_of_the_call, program_other_e_res, program_inadmissible_scale, program_output_folder,
             program_faulty_tokens, program_faulty_sums, program_faulty_models, program_order_of_the_faults,
             program_wrong_calls, program_faults_of_its_own, program_tensors_that_the_pass_does_not_read]  # fmt: skip
    if "SMALL_MODEL" in os.environ:
        cases.append(program_on_the_small_model)
    above = calibrate.others_can_change(tempfile.gettempdir())
    if above:
        print(f"stood aside: below an open place: {above[0]} ({calibrate.whose(*above)}): this test's temporary "
              "folders lie there, and calibrate.py refuses every call. Run it below folders that only you or root can "
              "change (by hand: set TMPDIR to such a folder)")
        return 77
    failed = 0
    for case in cases:
        try:
            if case.__name__.startswith("program"):
                with tempfile.TemporaryDirectory() as folder:
                    case(folder)
            else:
                case()
            print(f"passed  {case.__name__}")
        except Exception as fault:
            failed += 1
            print(f"FAILED  {case.__name__}: {type(fault).__name__}: {fault}")
    if failed:
        print(f"FAILED: {failed} of {len(cases)} cases")
        return 1
    if "SMALL_MODEL" not in os.environ:
        print(f"in part: {len(cases)} cases passed; SMALL_MODEL is not set, so program_on_the_small_model was not run")
        return 77
    print(f"OK: {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
