#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Makes quantization scales coarser as the worst-case check asks, or compares it with calibration.

usage: worst_case.py <table file> <scales file> <new scales file>
       worst_case.py against <table file> <scales file> <maxima file>

The fit (calibrate.py, fit_scales.py) chooses the quantization scales so that the largest absolute
values that a calibration pass recorded just fit. The worst-case check, which the committer program
cp_commit makes before it commits to the weights, computes upper bounds on the value at each site,
meant to hold for every prompt, from the weights and the scales (stated and argued in
scales/worst-case-bounds.md, "The bounds of the worst-case check"). A site is a named point of the
forward pass whose value has a limit. If a bound is above its limit, the site is short, and
cp_commit exits with status 7 and makes no commitment. cp_commit -worst-case-table writes a table
with a line per site and layer, whose last field is the "bits short": the least b >= 0 with
bound <= limit * 2^b. scales/README.md, "Terms", defines limit, shift, exponent and lookup key.

  <table file>   the check's table, computed under the scales of <scales file> (this is not checked)
  <scales file>  the fitted quantization scales: the words.json of fit_scales.py or calibrate.py
  <maxima file>  the largest absolute values that calibration recorded: maxima.json of calibrate.py

The first form writes the scales, with short sites made coarser, to <new scales file>, which must
not exist. A shift takes a sum of products from the scale of its two factors to the scale of its
output (B. Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference", arXiv:1712.05877, section 2.2, equations 5 and 6). So raising a
shift by b makes its output b bits coarser. Lowering by b the shift or exponent of the operation
that reads this output keeps the model's function otherwise the same
(shown in scales/README.md, "Making a cell coarser").

The names below are those of words.json. kq, kk, kv, ko, ku and kd are the shifts of the query, key,
value, attention output, up and down projections, and kuR is the ku of a routed expert. The sites q,
k, v and up are the outputs of the projections of these names. Where two changes apply to the same
number, they add up.

  v       kv is raised by the bits that v is short, and ko is lowered by them.
  up      ku is raised by the bits that up is short, and kd is lowered by them. In a
          mixture-of-experts layer the shared and the routed expert have one kd, so ku and kuR are
          both raised by the larger shortfall of its two sites, "up, shared" and "up, routed".
  q, k    kq and kk are raised by the bits that q and k are short, so sh, the shift of the softmax
          table's lookup key (the integer that selects a row of the table), has to be lowered by
          their sum. The new file has the same sh in every layer: the smallest value that this gives
          in any of them. A layer for which it gives a larger value raises kq by the difference as
          well, which costs q some precision there.
  stream  e_res, the exponent of the residual stream (an integer n there stands for n * 2^-e_res),
          is lowered by the most bits that any of the stream's four sites is short. ko and kd are
          raised by as many in every layer. If e_res would fall below 0, the program exits with
          status 2. The norms' epsilons eps1, eps2 and epsF are computed again. emb_bits, the width
          of the embedding's integers, is lowered by as many bits as e_res, but not below 2. You
          must then create the sums file (wsums.txt) and fold.txt, which both state e_res, again at
          the new e_res (scales/README.md, "A real model", step 2). The new emb_bits is an estimate:
          words_to_policy.py, which writes the parameter file (policy.txt) and the quantization
          scales (KL.bin) from words.json, rejects it if the new sums file has another width.

The program cannot make a gate, a router, a norm's quotient or the output head's margin coarser. The rounded output of a
gate or a router is the lookup key of a public table whose keys have a fixed step, so a different shift would compute a
different function. For the other two, and for a site whose name it does not know, for example "down quotient", the
program has no rule. Nor does it change anything for an up site of the wrong kind for its layer: "up" in a layer that
has experts in words.json, or "up, shared" or "up, routed" in one that has none. If one of these sites is short, the
program reports it and prints which lines of the parameter file (policy.txt) would raise the limit of a gate or a
router. It still makes all other changes and writes the new file, and it exits with status 3.

Some of the lines that the program names (eg, smax) take the parameter file out of the standard
parameter form. That is the form of the public inputs without which no run of the verifier program
cp_verify counts as a verification (its conditions: `Served` in verifier/src/vf_common.h).

The first form does not test whether the new scales are admissible. Run words_to_policy.py next,
which rejects a scale outside its range, and then run the check again. Only the check decides
whether every bounded site fits.

The second form writes nothing. It needs the words.json of calibrate.py, whose member "ew" has the
exponent of each weight tensor but the embedding. At the sites q, k, v and up it compares the
table's bound with the maximum that calibration recorded, layer by layer. It skips a maximum that is
not above 0. For each kind of site it prints the least distance between the two, in bits, and the
layer. A bound is to hold for every prompt, so no maximum should be above its bound. A maximum
counts as above its bound if it is more than 0.01 bits above. If that happens, the three files do
not belong to one model and one fit, or the check is wrong.

The table and the JSON files say more about the weights than the published scales do, and so does
what this program prints. Keep a real model's copies with the weights. The new file is readable by
its owner only.

Exit status:
  0  the new file is written; or no maximum is above its bound
  1  "against": a maximum is above its bound
  2  the call or an input file is faulty, or the new file cannot be written; no new file is left
  3  the new file is written, but a site for which the program has changed nothing is short
"""

import collections
import json
import math
import os
import sys

DONE, ABOVE_BOUND, FAULT, STILL_SHORT = 0, 1, 2, 3  # exit statuses

# The table's site names, as lines() in prover/weights_commitment/worst_case.cuh sets them
# (listed in scales/worst-case-bounds.md, "The sites of the check and their limits").
STREAM_SITES = ("stream at norm 1", "stream at norm 2", "stream at the last norm", "last stream, a cell")
SHIFTED_SITES = ("q", "k", "v", "up", "up, shared", "up, routed")
# Other sites, which cannot be made coarser. Each has the lines, if any, to print on what would
# raise its limit: gcl 1 and rcl 1 clamp a gate's lookup key and a router's logit inside the proof, and eg is the
# exponent of that key (8 without the line). The effect of gcl 1, rcl 1 and eg is described in
# docs/model.md, "10 Parameters that change the function". The numbers of bits follow from the
# limits in lines().
GATE = (
    "adding the line gcl 1 to the parameter file, if it is not there, raises a gate's limit by 7 bits",
    "each step of the line eg below 8 raises a gate's limit by 1 bit (fit again with fit_scales.py --eg), "
    "but the parameter file is then outside the standard parameter form",
)
ROUTER = ("adding the line rcl 1 to the parameter file, if it is not there, raises a router's limit by almost 4 bits",)
FIXED_SITES = {
    "gate": GATE,
    "gate, shared": GATE,
    "gate, routed": GATE,
    "router": ROUTER,
    "norm 1 quotient": (),
    "norm 2 quotient": (),
    "last norm quotient": (),
    "head margin": (),
}
NO_BOUND = 99  # the "bits short" of a site for which the check has no bound

# What "against" compares, in the order of a layer's sites. Each row: the table's site, the site of
# maxima.json, the weight tensor of the matrix product, the constant M of the norm whose output the
# product reads, the product's right shift, and the exponent of the rotary table's scale where the
# product includes it, else 0.
COMPARED = (
    ("q", "q", "attn_q", "M1", "kq", 12),
    ("k", "k", "attn_k", "M1", "kk", 12),
    ("v", "v", "attn_v", "M1", "kv", 0),
)
COMPARED_DENSE = (("up", "u", "ffn_up", "M2", "ku", 0),)
COMPARED_EXPERTS = (
    ("up, shared", "u", "ffn_up_shexp", "M2", "ku", 0),
    ("up, routed", "uR", "ffn_up_exps", "M2", "kuR", 0),
)
# The room of a site and layer is log2(bound) - log2(maximum). A maximum is above its bound if the
# room is below this.
LEAST_ROOM = -0.01

# A parsed table (its lines: scales/worst-case-bounds.md, "The lines of the table").
# d: the model's width. sites: {(site, layer): Site}. what_if: [(smax, bits short)],
# how short the site "stream at the last norm" would be if the parameter file's line smax, which
# limits the SiLU table's output, had this value. The first entry is for the file's own value.
Table = collections.namedtuple("Table", "d sites what_if")
# A site line: site | name | layer | parameter | value | log2 bound | log2 limit | bits short.
# shift: the field "value" (at q, k, v and up, the right shift); log2_bound: log2 of the bound;
# both as text, since only against() parses them. short: bits short.
Site = collections.namedtuple("Site", "shift log2_bound short")


class Fault(Exception):
    """Ends the run with the status FAULT and this message on standard error."""


def read_text(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except OSError as why:
        raise Fault("cannot read %s: %s" % (path, why.strerror)) from None
    except UnicodeDecodeError:
        raise Fault("%s has a byte that is not ASCII" % path) from None


def read_json(path):
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError as why:
        raise Fault("%s is not JSON (line %d, column %d)" % (path, why.lineno, why.colno)) from None
    except (ValueError, RecursionError):
        raise Fault("%s has a number that is too long, or is nested too deeply" % path) from None


def read_table(path):
    """Parses the check's table. Raises Fault if it is malformed, incomplete or contradictory,
    or if a site has no bound. Lines that this program does not need are skipped."""
    lines = read_text(path).split("\n")
    if lines[0] != "worst-case 1":
        raise Fault('the first line of %s is not "worst-case 1": it is not a table of the worst-case check' % path)
    d = verdict = None
    sites, what_if = {}, []
    for number, line in enumerate(lines, 1):
        fields = [x.strip(" \t") for x in line.split("|")]
        try:
            if fields[0] == "site":
                _, site, layer, _, shift, log2_bound, _, short = fields
                place = (site, int(layer))
                if place in sites:
                    raise Fault('%s has two lines for the site "%s", layer %d' % ((path,) + place))
                sites[place] = Site(shift, log2_bound, int(short))
            elif fields[0] == "what if":
                what_if.append((int(fields[1].split()[0]), int(fields[3])))
            elif line.startswith("d "):
                d = int(line.split()[1])
            elif line.startswith("verdict "):
                verdict = line.split()[1]
        except (ValueError, IndexError):
            raise Fault("line %d of %s is not in the form that the worst-case check writes" % (number, path)) from None
    if d is None or d < 1:
        raise Fault('%s has no line "d <width>" with a width of at least 1' % path)
    if verdict not in ("fits", "short"):
        raise Fault('%s has no verdict "fits" or "short", so it is not complete' % path)
    if (verdict == "short") != any(x.short for x in sites.values()):
        raise Fault('the verdict of %s is "%s", which contradicts its site lines' % (path, verdict))
    for place, x in sites.items():
        if x.short == NO_BOUND:
            raise Fault(
                '%s cannot be used: the check found no bound for the site "%s", layer %d (the last field is %d), '
                "so the table does not say by how many bits the site is short" % ((path,) + place + (NO_BOUND,))
            )
    return Table(d, sites, what_if)


def member(thing, key, kind, where):
    """Returns thing[key], which must be of type `kind`. JSON's true and false are not integers."""
    value = thing.get(key) if isinstance(thing, dict) else None
    if not isinstance(value, kind) or isinstance(value, bool):
        raise Fault('%s has no %s "%s"' % (where, {int: "integer", list: "list", dict: "object"}[kind], key))
    return value


def has_experts(layer):
    """Whether a layer's object in words.json is that of a mixture-of-experts layer."""
    return isinstance(layer, dict) and "kuR" in layer


def count(n, what):
    return "%d %s%s" % (n, what, "" if n == 1 else "s")


def refit(words, table, path):
    """Changes the scales of `words` in place, by the rules in the module's docstring. The member
    "KL" of `words` has an object of scales per layer, and "W" has the whole model's values. path is
    the scales file's name in messages. Returns the lines to print about what has changed. If it
    raises Fault, `words` may be partly changed."""
    layers, model = member(words, "KL", list, path), member(words, "W", dict, path)
    every = range(len(layers))
    for site, layer in table.sites:
        if layer not in every and site not in STREAM_SITES and site not in FIXED_SITES:
            has = (site, layer, path, count(len(layers), "layer"))
            raise Fault('the table has the site "%s" in layer %d, but %s has %s' % has)

    def short(site, layer):
        return table.sites[site, layer].short if (site, layer) in table.sites else 0

    def scale(layer, key):
        return member(layers[layer], key, int, "layer %d of %s" % (layer, path))

    def up(layer):
        if has_experts(layers[layer]):
            return max(short("up, shared", layer), short("up, routed", layer))
        return short("up", layer)

    stream = max([0] + [x.short for (site, _), x in table.sites.items() if site in STREAM_SITES])
    sh = min((scale(l, "sh") - short("q", l) - short("k", l) for l in every), default=0)
    # The bits by which each layer's shift is raised. For kq: short("q", l), plus the difference
    # between the sh that layer l would get by itself and the common sh.
    raised = {
        "kq": [scale(l, "sh") - sh - short("k", l) for l in every],
        "kk": [short("k", l) for l in every],
        "kv": [short("v", l) for l in every],
        "ku": [up(l) for l in every],
    }
    for l in every:
        changes = {key: by_layer[l] for key, by_layer in raised.items()}
        changes.update({"ko": stream - raised["kv"][l], "kd": stream - raised["ku"][l]})
        if has_experts(layers[l]):
            changes["kuR"] = raised["ku"][l]
        for key, by in changes.items():
            layers[l][key] = scale(l, key) + by
        layers[l]["sh"] = sh
    said = []
    for site, key in (("q", "kq"), ("k", "kk"), ("v", "kv"), ("up", "ku")):
        changed = [by for by in raised[key] if by]
        if changed:
            numbers = (site, key, count(max(changed), "bit"), count(len(changed), "layer"))
            said.append("%s: %s is raised by up to %s, in %s" % numbers)
    if stream:
        e_res = member(model, "e_res", int, '"W" of ' + path) - stream
        if e_res < 0:
            raise Fault("the residual stream is short by so many bits that the e_res of %s would fall below 0" % path)
        model["e_res"] = e_res
        # round(d * eps * 4^e_res) in integer arithmetic, for the floating-point norms' eps = 10^-5
        # (defined in scales/README.md, "The epsilon of a norm").
        model["eps1"] = model["eps2"] = model["epsF"] = (2 * table.d * 4**e_res + 100000) // 200000
        if "emb_bits" in words:
            words["emb_bits"] = max(2, member(words, "emb_bits", int, path) - stream)
        bits = count(stream, "bit")
        said.append("residual stream: e_res is lowered by %s, and ko and kd are raised by it in every layer" % bits)
        said.append(
            "e_res has changed to %d, so wsums.txt and fold.txt must be made again; weight_sums does not overwrite a "
            "file, so make a new folder, closed to other accounts: mkdir -m 700 <new folder>" % e_res
        )
        said.append(
            "run weight_sums with --eres %d and --out <new folder>/wsums.txt on the model's .gguf files, "
            'and write <new folder>/fold.txt with the two lines "eres %d" and "qkrows asis"' % (e_res, e_res)
        )
        said.append(
            "the prover's programs expect the model's .gguf files themselves, not links to them, in that folder too; "
            'use that folder in place of the old one from here on (scales/README.md, "A real model", step 2)'
        )
        for smax, left in table.what_if[1:]:
            then = (path, smax, "be %s short" % count(left, "bit") if left else "fit")
            said.append("alternative: with the scales of %s and the line smax %d the residual stream would %s" % then)
        if table.what_if[1:]:
            said.append("with the line smax the parameter file is outside the standard parameter form")
    return said


def still_short(table, layers, path):
    """The lines to print about the short sites that refit() has left unchanged: those in neither
    STREAM_SITES nor SHIFTED_SITES, and an up site of the wrong kind for its layer. layers is the
    member "KL" of the scales, and path is as in refit()."""
    found = {}  # {(site, why nothing is changed): [bits short]}
    for (site, layer), x in table.sites.items():
        if site not in STREAM_SITES + SHIFTED_SITES:
            why = "this program cannot make it coarser"
        elif site.startswith("up") and has_experts(layers[layer]) == (site == "up"):
            why = "%s has %s there, so nothing is changed for it" % (path, "experts" if site == "up" else "no experts")
        else:
            continue
        if x.short:
            found.setdefault((site, why), []).append(x.short)
    said, hints = [], []
    for (site, why), short in sorted(found.items()):
        numbers = (site, count(max(short), "bit"), count(len(short), "layer"), why)
        said.append("%s: short by up to %s, in %s; %s" % numbers)
        hints += [x for x in FIXED_SITES.get(site, ()) if x not in hints]
    return said + hints


def write_new(path, words):
    """Writes words.json at a path where nothing exists, readable and writable by its owner only."""
    try:
        text = json.dumps(words)
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        made = os.fstat(handle)
    except ValueError:
        raise Fault("a number of the new scales is too long to write") from None
    except OSError as why:
        raise Fault("cannot create %s: %s" % (path, why.strerror)) from None
    try:
        with os.fdopen(handle, "w") as f:
            os.fchmod(handle, 0o600)  # whatever the umask is
            f.write(text)
    except OSError as why:
        remove_created(path, made)
        raise Fault("cannot write %s: %s" % (path, why.strerror)) from None


def remove_created(path, made):
    """Removes the file that this run created at path. `made` is what os.fstat() returned for it
    while it was open. Another account that can change a directory above the file can make the path
    lead to another file, so this opens the file's directory and removes the name only if it is
    still that file, following no link. Errors are ignored: the caller reports the first failure."""
    try:
        directory = os.open(os.path.dirname(path) or ".", getattr(os, "O_PATH", 0) | os.O_DIRECTORY)
        try:
            name = os.path.basename(path)
            if os.path.samestat(os.stat(name, dir_fd=directory, follow_symlinks=False), made):
                os.unlink(name, dir_fd=directory)
        finally:
            os.close(directory)
    except OSError:
        pass


def say(line):
    print("worst_case.py: " + line)


def first_form(table_path, words_path, new_path):
    """Writes new_path: words_path's scales, changed as the table asks. Returns the exit status."""
    table = read_table(table_path)
    words = read_json(words_path)
    changed = refit(words, table, words_path)
    stuck = still_short(table, words["KL"], words_path)
    write_new(new_path, words)
    for line in changed + stuck:
        say(line)
    if changed:
        say("%s is written; run words_to_policy.py on it, and then the check again" % new_path)
    else:
        say("nothing is changed: %s is written with the scales of %s" % (new_path, words_path))
    return STILL_SHORT if stuck else DONE


def against(table_path, words_path, maxima_path):
    """Compares the table's bounds with the calibration pass's maxima. Returns the exit status."""
    table = read_table(table_path)
    words, maxima = read_json(words_path), read_json(maxima_path)
    exponents = member(words, "ew", dict, words_path)
    needed = '"ew" of %s, which must be the words.json of calibrate.py,' % words_path
    # Half of log2(d), rounded down. So 2^t is the largest power of two that does not exceed the
    # width's square root.
    t = (table.d.bit_length() - 1) // 2
    least = {}  # {the table's site: (its least room, the layer)}, in the order in which the sites first have a room
    for l, layer in enumerate(member(words, "KL", list, words_path)):
        here = "layer %d of %s" % (l, words_path)
        rows = COMPARED + (COMPARED_EXPERTS if has_experts(layer) else COMPARED_DENSE)
        for site, seen, tensor, norm, shift, rotary in rows:
            # The table bounds the sum before the right shift under which the table was made. Both
            # sides are compared after the shift.
            try:
                bound = float(table.sites[site, l].log2_bound) - int(table.sites[site, l].shift)
            except (KeyError, ValueError, OverflowError):
                bound = math.nan
            if not math.isfinite(bound):
                raise Fault('%s has no usable shift or bound for the site "%s", layer %d' % (table_path, site, l))
            # A real value x at the site is the integer x 2^Es after the shift. The fit's formulas
            # give Es from the scales: M = 32 + Es(norm) + t, and shift = Es(norm) + e - Es, plus 12
            # for q and k, where e is the weight tensor's exponent (fit() in scales/fit_scales.py;
            # stated in scales/README.md, "The constants of a norm", and in
            # scales/README.md, "The shifts").
            e = member(exponents, "blk.%d.%s.weight" % (l, tensor), int, needed)
            es = (member(layer, norm, int, here) - 32 - t) + e + rotary - member(layer, shift, int, here)
            maximum = maxima.get(seen) if isinstance(maxima, dict) else None
            maximum = maximum.get(str(l)) if isinstance(maximum, dict) else None
            if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
                raise Fault('%s has no number for the site "%s", layer %d' % (maxima_path, seen, l))
            if not maximum > 0:  # 0, a negative number or a NaN
                continue
            try:
                room = bound - (math.log2(maximum) + es)
            except OverflowError:
                raise Fault("%s has a number that is too large" % here) from None
            if site not in least or room < least[site][0]:
                least[site] = (room, l)
    above = 0
    for site, (room, l) in least.items():
        if room < LEAST_ROOM:
            above += 1
            say("%s: calibration recorded a value that is %.2f bits above the bound (layer %d)" % (site, -room, l))
        else:
            say("%s: the bound is at least %.2f bits above the recorded maximum (least in layer %d)" % (site, room, l))
    if above:
        how_many = count(above, "kind")
        say("at %s of site calibration recorded a value above a bound that is meant to hold for every prompt"
            % how_many)
        say("the three files do not belong to one model and one fit, or the check is wrong")
    return ABOVE_BOUND if above else DONE


def main(argv):
    try:
        if len(argv) == 4:
            return first_form(*argv[1:])
        if len(argv) == 5 and argv[1] == "against":
            return against(*argv[2:])
    except Fault as fault:
        print("worst_case.py: %s" % fault, file=sys.stderr)
        return FAULT
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return FAULT


if __name__ == "__main__":
    sys.exit(main(sys.argv))
