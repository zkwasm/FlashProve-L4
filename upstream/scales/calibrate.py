#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Run the calibration pass of a model and fit its quantization scales.

usage: calibrate.py --gguf <folder> --wsums <wsums.txt> --tokens <file> --out <folder>
                    [--window <tokens per window>] [--device <torch device>] [--eres N] [--chunk N]

The calibration pass runs the float32 model (float_forward.py) over a calibration set and records
the largest absolute activation at each site in SITES (named points of the forward pass), layer by
layer. The fit (fit_scales.py) chooses the integer (fixed-point) model's quantization scales
from these maxima and from the weight tensors' exponents: an integer w with the exponent e
represents w / 2^e. To prepare a model for the prover, pass the output file words.json to
words_to_policy.py, which writes the published files (the parameter file policy.txt and the scales
file KL.bin). The sites and the fit are defined in scales/README.md, "The fit of the scales".

  --gguf    the folder with the model's shards (its GGUF files)
  --wsums   the sums file, a checksum file that the tools in weight_sums/ made from these shards. It
            gives the weight tensors' exponents and the width of the embedding's integers. Its
            checksums are not compared with the shards.
  --tokens  the calibration set: text, tokenized with the model's tokenizer and cut into windows of
            equal length, at most 8,192 tokens. Each window is evaluated as a separate prompt, so it
            should start with the start token (not checked). A file whose name ends in .npy contains
            an integer array of the shape (windows, tokens per window). Any other file is raw:
            32-bit little-endian ids, window after window.
  --out     the folder for the two output files. If it does not exist, it is created with mode 700.
            If it exists, it must belong to you, and neither its group nor others may have write
            permission on it. Every folder above it and every symbolic link on the path to it must
            belong to you, to root or, under sudo, to the account that ran sudo. A folder above it
            may be writable by its group or by others only if it has the sticky bit, as /tmp has.
            Otherwise another account could replace the two files. Missing folders above --out are
            also created, without write permission for their group or others. A run that fails
            after it has created folders leaves them in place.
  --window  tokens per window: required for a raw token file, ignored for a .npy file
  --device  the torch device that computes (default cpu)
  --eres N  e_res, the exponent of the residual stream and of the embedding, from 1 to 24 (default
            11). The sums file must have been made with the same N. At a model width d of 5,120,
            use N from 2 to 14: words_to_policy.py rejects the words.json of any other N and prints
            "w2p refuse arith-eps 3", although this program can exit with status 0.
            words_to_policy.py requires the integer model's norm epsilons eps1, eps2 and epsF to be
            round(d * 1e-5 * 4^N) and not 0. For N = 1 that number is 0. From N = 15 on the fit's
            value differs, because a shard stores 1e-5 as a float32 number.
  --chunk N how many windows go through a layer at a time (default 4). A smaller N saves memory for
            most activations. It changes the maxima only by floating-point rounding.

The program writes two JSON files into the folder --out, which must not contain them already:

  maxima.json  {site: {layer: largest absolute value}}. At the site q the value is divided by
               sqrt(dh), where dh is the head dimension.
  words.json   the fitted scales, as fit_scales.py writes them. "ew" gives the exponent of every
               tensor in the sums file except the embedding, and "emb_bits" the width of the
               embedding's integers.

Both files reveal more about the weights than the published files do. So they are created with mode
600, and you should store those of a real model where you store its weights. The program prints
names, counts and differences in bits, but no weight, no activation and no fitted scale. A
difference in bits between two scales ("made B bits coarser") is not in the published files, so
store the printed output of a run on a real model with the weights too.

Exit status: 0 if both files are written. 1 if both are written but a scale is outside the range
that fit_scales.py accepts; the program prints which one. (words_to_policy.py applies the verifier's
narrower ranges.) 2 for an invalid command line, an unusable input or --out folder, or a failed
import; then no file is written, and standard error gives the reason. A write error also gives 2,
and the files that were written are removed. An unexpected exception also gives 2, and one or both
files may then exist. If standard output is closed before the program ends, Python may end it with
status 120 after both files are written.

The method is static calibration by the largest absolute value seen on a calibration set:
H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical
Evaluation", arXiv:2004.09602, section 3.4.
"""

import argparse
import math
import os
import re
import stat
import sys
import traceback

# Exit status of a failed run, and argparse's for an invalid command line.
FAULT = 2

sys.dont_write_bytecode = True  # the program leaves no __pycache__ folder next to itself

try:
    import numpy
    import torch

    import fit_scales
    import float_forward
    import gguf_weights
except ImportError as fault:
    # An uncaught exception would give exit status 1, which here means that both files are written.
    print(f"calibrate.py: the Python package {fault.name} cannot be imported", file=sys.stderr)
    sys.exit(FAULT)

# The sites that the fit reads. record() in run() ignores the others that float_forward.forward()
# reports. n1, n2 and nF are the outputs of a layer's first norm, its second norm and the final
# norm, before the norm's gain vector; nF has one entry, keyed by the number of layers. q, k and v
# are the queries, keys and values. u is the up projection of a dense layer or of the shared expert,
# uR that of the routed expert.
SITES = ("n1", "n2", "q", "k", "v", "u", "uR", "nF")

# The layers' weight tensors whose exponents the fit reads. A tensor's name is
# blk.<layer>.<kind>.weight.
KINDS_IN_EVERY_LAYER = ("attn_q", "attn_k", "attn_v", "attn_output")
KINDS_IN_A_DENSE_LAYER = ("ffn_gate", "ffn_up", "ffn_down")
KINDS_IN_A_LAYER_WITH_EXPERTS = ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp", "ffn_gate_exps",
                                 "ffn_up_exps", "ffn_down_exps")  # fmt: skip
EMBEDDING = "token_embd.weight"


class Fault(Exception):
    """An error that makes the run exit with status FAULT. main() prints its text for the user."""


# The look that run_suite.py makes before it works below a folder, each part to the letter:
# the constants and functions below are run_suite.py's own. Copy this block whole into a script that
# makes a folder, or works below one, that it was given; change nothing in it.
#
# To use it, call others_can_change(<the existing parent of the folder you will make>, or <the folder itself if it exists>).
# It returns None, or (place, reason): the first place at or above it, from / down, on the typed path, on the resolved
# path and through every link on the way, that another account may be able to change, that does not exist, or that could not be
# examined (a failed look is never a pass). On a find, print "<script>: " + refusal(<what>, <path>, place, why) on standard error and
# exit with status 2. run_suite.py passes its option --i-trust-every-account-on-this-machine as refusal()'s last argument,
# trust, and so adds that option to the same sentence; for a place that does not exist or could not be examined it prints another
# sentence. The advice to chmod is given only to a caller who is not root and owns that open folder. Then use the resolved
# path of the parent from there on. (run_suite.py also refuses its build folder, named by the environment variable BUILD,
# if that has a default access control list; that check is not part of this block.)
# It needs "import os", "import re" and "import stat" in the script. It brings its own printable(); a script that has
# one of its own keeps this one all the same.

ACL = "has a default access control list"
OPEN = "can be written by its group or by other accounts and is not sticky"
THEIRS = "belongs to another account"
UNSEEN = "could not be examined"
GONE = "does not exist"
INVISIBLE = "\u2800\u3164\u115f\u1160\uffa0\U00013441\U00013442\U0001d159"


def printable(text):
    """Returns `text` with each character that str.isprintable() rejects, and each one in INVISIBLE,
    written as its Python escape (such as \\x1b), so that a test's output cannot move the cursor or
    pass for this program's own."""
    return "".join(c if c.isprintable() and c not in INVISIBLE else ascii(c)[1:-1] for c in text)


def trusted_owners():
    """Returns the accounts whose directories this run may build and work in: this one, root, and
    under sudo (as root, with SUDO_UID set) the account that ran sudo, so that under sudo that
    account's own directories, such as its home, are not reported as another account's."""
    owners = {0, os.geteuid()}
    if os.geteuid() == 0 and re.fullmatch(r"[0-9]+", os.environ.get("SUDO_UID", "")):  # anything else is not trusted
        owners.add(int(os.environ["SUDO_UID"]))
    return owners


def others_can_change(folder):
    """Returns (place, why) for the first directory or symbolic link at or above `folder`, on its
    path as typed, as the system reads it, or through the symbolic links on it, that another account
    could change, that does not exist or that could not be examined; else None. A place that another
    account could change belongs to an account that trusted_owners() does not name, or is a
    directory that its group or others can write in without the sticky bit. There another account
    could swap the directory below it, or point the link elsewhere."""
    def prefixes(path):
        parts = [p for p in path.split("/") if p]
        return ["/"] + ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]

    read = {}  # each symbolic link whose target looked_in() read and followed: its identity -> that target

    def identity(about):  # with the target kept beside it: a link swapped in may reuse the inode and even the change time
        return about.st_dev, about.st_ino, about.st_ctime_ns

    def looked_in(path):
        """Returns the directories that the system passes through to follow `path`, each symbolic
        link on the way, and the place where `path` leads. A link's target is followed from the
        directory that contains the link, so a link to a link elsewhere brings that directory in
        too. After 40 links, where the system gives up, the rest counts as plain names. Each link
        whose target it read goes into `read`, by its identity and with that target, if the same link
        stood before and after the read."""
        whole = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        places, names, at, links = [], [p for p in whole.split("/") if p], "/", 0
        while names:
            name = names.pop(0)
            if name == ".":
                continue
            if name == "..":  # the parent as the system takes it, not as typed
                at = os.path.dirname(at)
                continue
            places.append(at)
            here = os.path.join(at, name)
            try:
                first = os.lstat(here) if links < 40 else None
                target = os.readlink(here) if first is not None and stat.S_ISLNK(first.st_mode) else None
                if target is not None:
                    again = os.lstat(here)  # a link swapped during the read is left unread: the loop below refuses it
                    target = target if identity(again) == identity(first) else None
            except OSError:  # not there, or not examined here: the loop below decides
                target = None
            if target is not None:
                places.append(here)  # the link's owner can point it elsewhere
                read[identity(first)] = target
                links += 1
                names = [p for p in target.split("/") if p] + names
                at = "/" if target.startswith("/") else at
            else:
                at = here
        return places + [at]

    for place in prefixes(os.path.abspath(folder)) + prefixes(os.path.realpath(folder)) + looked_in(folder):
        try:
            about = os.lstat(place)
        except FileNotFoundError:  # a missing one is never a pass, even if the folder exists (as typed: a symbolic link, then "..")
            return place, GONE
        except OSError:  # nor is one that could not be examined
            return place, UNSEEN
        if about.st_uid not in trusted_owners():
            return place, THEIRS
        if stat.S_ISLNK(about.st_mode):  # only a link that the look read, still leading where it led then
            try:
                same = read.get(identity(about)) == os.readlink(place)
            except OSError:
                same = False
            if not same:
                return place, UNSEEN
        if stat.S_ISDIR(about.st_mode) and about.st_mode & 0o022 and not about.st_mode & stat.S_ISVTX:
            return place, OPEN
    return None


def refusal(what, path, place, why, trust=""):
    """The one sentence for a folder that is refused because place above it may be changed by another account: what it
    is (such as "work folder"), its path, place and why as others_can_change() gave them, and trust, the words that a
    caller who may say so adds (the runner: its option; a script without one: nothing)."""
    return ("the %s %s lies below %s, which another account may be able to change (%s): it could put another folder in its "
            "place and choose what runs as you. %same a new %s below folders that only you or root can change (such as "
            "one in your home folder, or in /tmp, whose sticky bit keeps others from replacing it)%s; nothing was run"
            % (what, printable(path), printable(place), whose(place, why), cure(place, why), what, trust))


def whose(place, why):
    """Why another account may be able to change place, in words, for the refusals of the work folder and of BUILD."""
    if why == THEIRS:
        return "a symbolic link that belongs to another account" if os.path.islink(place) else "owned by another account"
    if why == ACL:
        return "a default access control list"
    if why == UNSEEN:
        return "it could not be examined"
    if why == GONE:
        return "it does not exist"
    return "its group or others can write in it, and it has no sticky bit"


def cure(place, why):
    """The first words of what to do about place: chmod go-w, for a folder of the caller's own that is open by its mode
    alone (its group may be the caller's private group, as under a umask of 002). Never for root, for whom a shared disk's
    mount point is «its own»: obeyed, it would shut every other account out."""
    try:
        mine = why == OPEN and os.geteuid() != 0 and os.lstat(place).st_uid == os.geteuid()
    except OSError:
        mine = False
    return ("If %s is yours, close it with chmod go-w (a further such folder below it, if any, is named in turn); "
            "otherwise n" % printable(place) if mine else "N")


def one_line(text, most=2000):
    """Return text that may quote a path or an input file, for one printed line.

    It is cut after `most` characters, with a mark, and printable() escapes its control characters.
    """
    return printable(text[:most]) + (" ... (cut)" if len(text) > most else "")


def check_own_folder(folder):
    """Raise Fault unless the existing folder is the caller's and not writable by group or others.

    check_above() checks the folders above it.
    """
    state = os.stat(folder)
    if state.st_uid != os.geteuid():
        raise Fault(f"the folder {folder} does not belong to the account that runs this program, so its owner could "
                    "replace the two files. Name a new folder with --out.")
    if state.st_mode & 0o022:
        raise Fault(f"the group of the folder {folder} or other accounts can write in it, so they could replace the "
                    "two files, and may already have put links there. Name a new folder with --out.")


def check_above(folder, existing):
    """Raise Fault if another account could change a folder or symbolic link at or above `existing`.

    `existing` is `folder` or, if `folder` does not exist, the deepest existing folder above it.
    A place on the path that is missing or cannot be examined also raises Fault.
    """
    found = others_can_change(existing)
    if found:
        raise Fault(refusal("output folder", folder, *found))


def check_output_paths(folder, paths):
    """Raise Fault if the folder's name is empty, a path exists, or the folder cannot be made, written to or trusted."""
    if not folder:
        raise Fault("--out is empty. Name the folder for the two output files.")
    for path in paths:
        if os.path.lexists(path):
            raise Fault(f"{path} exists. Name another folder with --out.")
    existing = os.path.abspath(folder)
    while not os.path.lexists(existing):
        existing = os.path.dirname(existing)
    if not os.path.isdir(existing) or not os.access(existing, os.W_OK | os.X_OK):
        raise Fault(f"the folder {folder} cannot be made or written to, because {existing} is not a writable folder")
    if existing == os.path.abspath(folder):
        check_own_folder(folder)
    check_above(folder, existing)


def read_sums(path):
    """Return {tensor name: (exponent, bits)} from the sums file, in the file's order.

    The file has one line "<name> <elements> <exponent> <bits> <SHA-256>" per tensor and then the
    line "END <number of lines before it>" (defined in weight_sums/README.md, "The file's form").
    <bits> is the width of the tensor's integers. Empty lines are skipped, and the second and the
    fifth field are not read.
    """
    try:
        with open(path, encoding="ascii") as file:
            lines = [line for line in file.read().split("\n") if line]
    except OSError as fault:
        raise Fault(f"the sums file {path} cannot be read: {fault.strerror}") from fault
    except ValueError as fault:
        raise Fault(f"the sums file {path} is not ASCII text") from fault
    if not lines or not re.fullmatch("END [0-9]+", lines[-1]) or int(lines[-1][4:]) != len(lines) - 1:
        raise Fault(f'the sums file {path} does not end with the line "END <number of lines before it>"')
    sums = {}
    for number, line in enumerate(lines[:-1], 1):
        fields = line.split(" ")
        if len(fields) != 5 or not re.fullmatch("-?[0-9]+", fields[2]) or not re.fullmatch("[0-9]+", fields[3]):
            raise Fault(f'tensor line {number} of the sums file {path} does not have the form "<name> <elements> '
                        '<exponent> <bits> <SHA-256>"')  # fmt: skip
        if fields[0] in sums:
            raise Fault(f"tensor line {number} of the sums file {path} repeats the name of an earlier line")
        sums[fields[0]] = (int(fields[2]), int(fields[3]))
    return sums


def line_of(sums, name):
    """Return (exponent, bits) of the tensor `name`."""
    if name not in sums:
        raise Fault(f"the sums file has no line for the tensor {name}")
    return sums[name]


def exponents_for_the_fit(sums, shape):
    """Return the fit's input {kind: {layer: exponent}, "output": exponent}.

    `shape` comes from gguf_weights.shape_words(); its "moe" lists the layers with experts.
    """
    exponents = {}
    for layer in range(shape["L"]):
        rest = KINDS_IN_A_LAYER_WITH_EXPERTS if layer in shape["moe"] else KINDS_IN_A_DENSE_LAYER
        for kind in KINDS_IN_EVERY_LAYER + rest:
            exponents.setdefault(kind, {})[str(layer)] = line_of(sums, f"blk.{layer}.{kind}.weight")[0]
    exponents["output"] = line_of(sums, "output.weight")[0]
    return exponents


def read_tokens(path, window, shape):
    """Return the file's token ids as an int64 tensor of the shape (windows, tokens per window).

    float_forward.as_windows() rejects out-of-vocabulary ids and windows over 8,192 tokens.
    """
    try:
        if path.endswith(".npy"):
            with open(path, "rb") as file:
                tokens = numpy.lib.format.read_array(file, allow_pickle=False)
            if tokens.dtype.kind not in "iu" or tokens.ndim != 2:
                raise Fault(f"the token file {path} does not hold an integer array of two dimensions")
        else:
            if window < 1:
                raise Fault(f"the name of the token file {path} does not end in .npy, so the file is read as raw ids. "
                            "Give the number of tokens per window with --window.")  # fmt: skip
            tokens = numpy.fromfile(path, dtype="<i4")  # leaves out one to three bytes after the last whole id
            if tokens.size % window != 0:
                raise Fault(f"the token file {path} holds {tokens.size} ids, not a multiple of --window {window}")
            tokens = tokens.reshape(-1, window)
    except OSError as fault:
        raise Fault(f"the token file {path} cannot be read: {fault.strerror}") from fault
    except ValueError as fault:
        raise Fault(f"the token file {path} is not in the .npy format, although its name ends in .npy") from fault
    if tokens.size == 0:
        raise Fault(f"the token file {path} holds no token")
    try:
        return float_forward.as_windows(tokens.astype(numpy.int64), shape)
    except float_forward.ForwardError as fault:
        raise Fault(f"the token file {path} cannot be used: {fault}") from fault


def check_tensors(model, shape):
    """Raise Fault unless every tensor that forward() reads is present with the right dimensions.

    The heads must also fit float_forward.attend() (H query heads, a multiple of the G key and value
    heads) and float_forward.rotate() (dh even).
    """
    names = model.names()
    for name, dims in float_forward.tensor_shapes(shape).items():
        if name not in names:
            raise Fault(f"the model has no tensor {name}")
        in_file = model.info(name)[1][::-1]  # a file lists the dimensions in the reverse of torch's order
        if in_file != dims:
            raise Fault(f"the tensor {name} has the dimensions {in_file}, and the rest of the model asks for {dims}")
    if shape["H"] % shape["G"] != 0 or shape["dh"] % 2 != 0:
        raise Fault(f"the model has {shape['H']} query heads and {shape['G']} key and value heads of {shape['dh']} "
                    "entries. The first number must be a multiple of the second, and the third even.")  # fmt: skip


def run(call):
    """Run the calibration pass and the fit, write the two files, and return the exit status.

    Most invalid inputs are rejected before the calibration pass starts. A tensor's type and values
    are checked only when it is read or computed with, and fit_scales.fit() can reject maxima or
    exponents afterwards.
    """
    if call.out:  # one path for all checks and writes: the system reads "link/../out" otherwise than abspath() does
        call.out = os.path.abspath(call.out)
    maxima_path, words_path = os.path.join(call.out, "maxima.json"), os.path.join(call.out, "words.json")
    check_output_paths(call.out, (maxima_path, words_path))
    try:
        torch.empty(0, device=call.device)
    except (RuntimeError, AssertionError) as fault:
        raise Fault(f"torch cannot use the device {call.device}: {str(fault).splitlines()[0]}") from fault
    try:
        model = gguf_weights.open_model(call.gguf)
        shape = gguf_weights.shape_words(model)
    except gguf_weights.ModelError as fault:
        raise Fault(str(fault)) from fault
    sums = read_sums(call.wsums)
    made_at, embedding_bits = line_of(sums, EMBEDDING)
    if made_at != call.eres:
        raise Fault(f"the sums file was made with --eres {made_at}, and this run has --eres {call.eres}. Pass "
                    f"--eres {made_at} here (the option takes 1 to 24), or run weight_sums with --eres {call.eres} "
                    f"and pass the new file with --wsums. Give weight_sums a new --out path: it does not overwrite "
                    f"a file.")  # fmt: skip
    tokens = read_tokens(call.tokens, call.window, shape)
    check_tensors(model, shape)

    print(f"shards: {len({model.info(name)[2] for name in model.names()})}, layers: {shape['L']}, "
          f"layers with experts: {len(shape['moe'])}, width: {shape['d']}, windows: {tokens.shape[0]}, "
          f"tokens per window: {tokens.shape[1]}, device: {call.device}", flush=True)  # fmt: skip
    # output.weight is left out: the calibration pass ends before it, at the final norm.
    unread = sorted(set(model.names()) - set(float_forward.tensor_shapes(shape)) - {"output.weight"})
    if unread:
        unread[0] = one_line(unread[0], 80)  # the file's maker chose the name
        print(f"tensors of the model that the pass does not read: {len(unread)}, the first of them {unread[0]}. If one "
              "of them stands for a step of the model that the pass does not have, the scales do not fit the model.",
              flush=True)  # fmt: skip
    exponents = exponents_for_the_fit(sums, shape)

    largest = {site: {} for site in SITES}

    def place(site, layer):
        return f"the site {site}" + ("" if site == "nF" else f" of layer {layer}")

    def record(site, layer, tensor):
        if site in largest:
            seen = float(tensor.abs().max())  # a float32 number, which a Python float holds exactly
            if not math.isfinite(seen):
                raise Fault(f"an activation at {place(site, layer)} is not finite. A model file may be damaged.")
            largest[site][str(layer)] = max(seen, largest[site].get(str(layer), 0.0))

    try:
        float_forward.forward(model, shape, tokens, record, device=call.device, chunk=call.chunk)
    except gguf_weights.ModelError as fault:
        raise Fault(str(fault)) from fault
    for site, of_site in largest.items():
        for layer, seen in of_site.items():
            if seen == 0:
                raise Fault(f"every activation at {place(site, layer)} is zero, so no scale fits it")
    # The integer model's queries are q / sqrt(dh), because the conversion of the weights to
    # integers multiplies attention's factor 1 / sqrt(dh) into the query matrix
    # (`QUERY_CONSTANT` in weight_sums/weight_sums.py; its value is defined in
    # docs/model.md, "11.3 Which tensor is which matrix"; why the site q is divided:
    # scales/README.md, "The sites").
    largest["q"] = {layer: seen * shape["dh"] ** -0.5 for layer, seen in largest["q"].items()}
    maxima = {site: of_site for site, of_site in largest.items() if of_site}
    try:
        words, notes = fit_scales.fit(shape, maxima, exponents, e_res=call.eres)
    except fit_scales.InputFault as fault:
        raise Fault(str(fault)) from fault
    # words_to_policy.py compares "ew" with the sums file.
    words["ew"] = {name: exponent for name, (exponent, _) in sums.items() if name != EMBEDDING}
    words["emb_bits"] = embedding_bits

    written = []
    try:
        check_output_paths(call.out, (maxima_path, words_path))  # all checks again: hours may have gone by
        mask = os.umask(0o077)
        try:
            os.umask(mask | 0o022)  # neither group nor others get write permission on the parents that are created
            os.makedirs(os.path.dirname(call.out), exist_ok=True)
        finally:
            os.umask(mask)
        try:
            os.mkdir(call.out, 0o700)  # a new folder is closed to other users
        except FileExistsError:
            check_own_folder(call.out)  # the usual way if the folder existed before
        check_above(call.out, call.out)  # sees what another account put under a missing name in the last moment
        for path, content in ((maxima_path, maxima), (words_path, words)):
            fit_scales.write_private_json(path, content)
            written.append(path)
    except OSError as fault:
        for path in written:
            os.unlink(path)
        raise Fault(f"the two files cannot be written to the folder {call.out}: {fault.strerror}") from fault

    print(f"wrote {sum(len(of_site) for of_site in maxima.values())} maxima to {maxima_path} "
          f"and the scales of {shape['L']} layers to {words_path}")  # fmt: skip
    status = fit_scales.report(notes)
    if status != 0:
        print("words_to_policy.py will reject this words.json. Check the named layers' exponents in the sums file.")
    return status


def whole_number(least, most):
    """Return an argparse type function that accepts a whole number from `least` to `most`."""

    def convert(text):
        try:
            value = int(text)
        except ValueError:
            value = least - 1
        if not least <= value <= most:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number from {least} to {most}")
        return value

    return convert


def main(arguments):
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description="Run the calibration pass of a model and fit its quantization scales.",
        epilog="Exit status: 0 if both files are written. 1 if both are written but a scale is outside the range that "
        "fit_scales.py accepts; the program prints which one. 2 for an invalid command line, an unusable input or a "
        "Python package that cannot be imported; then no file is written, and standard error gives the reason. A write "
        "error also gives 2, and what was written is removed. An unexpected exception also gives 2, and one or both "
        "files may then exist. If standard output is closed before the program ends, Python may end it with status 120 "
        "after both files are written. The text at the top of the program describes the files.",
    )
    parser.add_argument("--gguf", required=True, metavar="FOLDER", help="the folder with the model's GGUF files")
    parser.add_argument("--wsums", required=True, metavar="FILE", help="the sums file (wsums.txt) of these GGUF files")
    parser.add_argument("--tokens", required=True, metavar="FILE", help="the calibration set, a .npy or raw file")
    parser.add_argument("--out", required=True, metavar="FOLDER", help="the folder for maxima.json and words.json")
    parser.add_argument("--window", type=int, default=0, metavar="N", help="tokens per window of a raw token file")
    parser.add_argument("--device", default="cpu", help="the torch device that computes (default cpu)")
    parser.add_argument("--eres", type=whole_number(1, 24), default=11, metavar="N",
                        help="the residual stream's exponent e_res, as for the sums file (default 11)")  # fmt: skip
    parser.add_argument("--chunk", type=whole_number(1, 2**31), default=4, metavar="N",
                        help="windows that go through a layer at a time (default 4)")  # fmt: skip
    call = parser.parse_args(arguments)
    try:
        return run(call)
    except Fault as fault:
        print(f"calibrate.py: {one_line(str(fault))}", file=sys.stderr)  # a message can quote a path
    except Exception as fault:
        # Uncaught, the exception would give exit status 1 (see the imports). Its text can quote an
        # input file, so it is printed like a Fault's, after the calls that led to it.
        sys.stderr.write("Traceback (most recent call last):\n" + "".join(traceback.format_tb(fault.__traceback__)))
        print(f"{type(fault).__name__}: {one_line(str(fault))}", file=sys.stderr)
    return FAULT


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
