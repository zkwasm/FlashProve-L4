#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Make the calibration set for Llama 4 Maverick's quantization scales from three published files.

usage: make_calibration_set.py <train.parquet> <llama2 tokenizer.json> <llama4 tokenizer.json> <out .npy> [--any-files]

The set is 32 windows of 1,024 token ids. Each window is the model's start token followed by 1,023
tokens of English prose from the training split of WikiText-2. scales/calibrate.py runs the
floating-point model over the windows to fit the scales. It also reads sets made in other ways, as
described in scales/README.md, "The calibration set". The seven-step rule in the docstring of
apply_rule() makes this set a fixed function of the three files in INPUTS, so anyone who has them
can regenerate it byte for byte. The expected SHA-256 sums were checked with the versions in
CHECKED_WITH: another version of Python or of the package tokenizers may give a different set, and a
different set can give different scales.

The program rejects input files other than those in INPUTS. After writing the set it compares two
SHA-256 sums with their expected values: that of the bytes that it wrote (it does not read the file
back) and that of the 128 drawn rows of step 3 (random slices of the corpus, 2,048 ids each). With
--any-files, which may appear anywhere among the paths, it applies the same rule to another parquet
file with a string column "text" and two other tokenizer files, and compares no sum. Nothing may
exist at <out .npy>: the program does not overwrite. The path should end in .npy, which is not
checked: scales/calibrate.py reads a file with any other ending as raw ids.

Steps 1 to 3 are the way in which published post-training quantization code samples its calibration text. Section 5 of
E. Frantar, S. Ashkboos, T. Hoefler and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", arXiv:2210.17323, describes the calibration data as "128 random 2048 token segments", there
of the C4 dataset. Steps 1 to 3 are after the WikiText-2 sampler in datautils.py of the GPTQ authors'
repository (github.com/IST-DASLab/gptq, get_wikitext2). The library Optimum of Hugging Face has such a sampler
(github.com/huggingface/optimum, version 1.18.1, optimum/gptq/data.py). Its function get_dataset (lines 221 to 245) is
described as "Get the dataset from the original paper of GPTQ", has the defaults of 128 samples, 2,048 ids and seed 0,
and calls random.seed(seed). Its function get_wikitext2 (lines 115 to 131) tokenizes the joined rows of this same split
once and draws each segment's start with random.randint(0, n - seqlen - 1). That version joins the first 1,000 rows
without a separator, after replacing each empty row with a space and a line feed. Step 1 joins all rows with two line
feeds and encodes them in one call, as Hugging Face's guide "Perplexity of fixed-length models" does with the test split
of the same corpus (Transformers documentation, huggingface.co/docs/transformers/perplexity).

Steps 4 and 5 carry the sample over from the Llama 2 tokenizer's ids to the model's own. Cutting the ids into
consecutive windows that each begin with the start token (step 6) is also how llama.cpp's tool imatrix reads its
calibration text, except that imatrix overwrites each window's first id. The source is llama.cpp
(github.com/ggml-org/llama.cpp), commit 794fe23f29fb40104975c91fe19f23798f7c726e: examples/imatrix/imatrix.cpp,
compute_imatrix, lines 470 and 511 to 513.

Exit status:
0  the file is written and, without --any-files, both sums are as expected
1  the file is written, but a sum is not as expected
2  a package cannot be imported, the program rejects its arguments or an input, or it cannot write
   the file: the run leaves no file. Also after an unexpected exception: the file may then exist.
"""

import hashlib
import io
import os
import random
import sys
import traceback

WRITTEN, DIFFERS, REJECTED = 0, 1, 2  # The exit statuses.

try:
    import numpy
    import pyarrow
    import pyarrow.parquet
    import tokenizers
except ImportError as e:
    # Uncaught, the exception would make Python exit with status 1, which here means a written file.
    print(f"make_calibration_set: cannot import a Python package that the program needs: {e}", file=sys.stderr)
    sys.exit(REJECTED)

USAGE = """\
usage: make_calibration_set.py <train.parquet> <llama2 tokenizer.json> <llama4 tokenizer.json> <out .npy> [--any-files]
Without --any-files only the three published files in the table INPUTS at the top of the program are accepted. \
With it, other files are accepted and no SHA-256 sum is compared."""

# The three input files, in the order of the command line. Each is published at
# https://huggingface.co/<repository>/resolve/<revision>/<path>, and "bytes" and "sha256" are that
# file's size and SHA-256 sum. The repositories of the two tokenizers are open only to users whose
# request for access the publisher has granted. The request includes accepting the model's licence.
INPUTS = (
    {
        "role": "parquet file",
        "what": "the training split of WikiText-2 (raw)",
        "repository": "datasets/Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "path": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "bytes": 6357543,
        "sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    },
    {
        "role": "first tokenizer file",
        "what": "the Llama 2 tokenizer",
        "repository": "meta-llama/Llama-2-7b-hf",
        "revision": "01c7f73d771dfac7d292323805ebc428287df4f9",
        "path": "tokenizer.json",
        "bytes": 1842767,
        "sha256": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    },
    {
        "role": "second tokenizer file",
        "what": "the Llama 4 tokenizer",
        "repository": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "revision": "73d14711bcc77c16df3470856949c3764056b617",
        "path": "tokenizer.json",
        "bytes": 27948578,
        "sha256": "172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4",
    },
)

# The sums that a run on the three files in INPUTS should give. No other data goes into them, so
# both are fingerprints of token ids of public text. SET_SHA256 is the SHA-256 sum of the output
# file. DRAWS_SHA256 is that of the 128 drawn rows of step 3 as a .npy file in the set's format,
# with shape (128, 2048). That file is only hashed, never written. If the set's sum differs, the
# drawn rows' sum shows whether the difference arises in steps 1 to 3 or in steps 4 to 7.
# No test in this repository
# reproduces either sum.
DRAWS_SHA256 = "8cd63f617d474124a50ee052eae2fd0be603ea45f191ec8ccb4f443f676fb58c"
SET_SHA256 = "a2ce04bd55b1b6fa2b9148744e087f5a2a484c1b3d525ea9fb2fc1ab8c6c2c02"
CHECKED_WITH = "Python 3.13.15, numpy 2.3.5, pyarrow 23.0.1, tokenizers 0.22.2"

# The constants of the rule (steps 3, 4 and 6 at apply_rule()).
SEED = 0
DRAWS = 128
DRAWS_USED = 48
DRAW_LENGTH = 2048
REMOVED_IDS = (1, 2)
START_ID = 200000
WINDOWS = 32
WINDOW_LENGTH = 1024


class Rejection(Exception):
    """A reason to exit with the status REJECTED. Its text is the one-line message for the user."""


def say(line, stream=None):
    """Print a line prefixed by the program's name, to standard output unless `stream` is given."""
    print("make_calibration_set: " + line, file=stream, flush=True)


def one_line(e):
    """Return an exception's message as one line.

    str() of a KeyError puts quotation marks around the message, so its argument is used instead.
    """
    return " ".join(str(e.args[0] if isinstance(e, KeyError) else e).split())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_input(path, named, any_files):
    """Return an input file's bytes and print their SHA-256 sum.

    Unless any_files is set, reject a file whose sum is not that of `named`, the file's entry in
    INPUTS. The program parses these same bytes, so the file checked is the file used.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise Rejection(f"cannot read {path}: {e.strerror}")
    found = sha256(data)
    say(f"{named['role']}: {path}, {len(data)} bytes, SHA-256 {found}")
    if not any_files and found != named["sha256"]:
        raise Rejection(
            f"{path} is not the expected {named['role']}, which is {named['what']}: {named['path']} in the repository "
            f"{named['repository']} on huggingface.co, revision {named['revision']}, {named['bytes']} bytes, "
            f"SHA-256 {named['sha256']}. To apply the rule to other files, add --any-files."
        )
    return data


def text_rows(path, data):
    """Return the column "text" of a parquet file, given as bytes, as a list of strings."""
    # ParquetFile.read with use_threads=False works in the calling thread.
    # pyarrow.parquet.read_table starts worker threads, and with pyarrow 23.0.1 a process that ends
    # soon after that call sometimes does not exit.
    try:
        table = pyarrow.parquet.ParquetFile(pyarrow.BufferReader(data)).read(columns=["text"], use_threads=False)
        rows = table.column("text").to_pylist()
    except (pyarrow.ArrowException, OSError, KeyError, UnicodeDecodeError) as e:
        raise Rejection(f'{path} is not a parquet file with a column "text" of strings: {one_line(e)}')
    if not all(isinstance(row, str) for row in rows):
        raise Rejection(f'the column "text" of {path} has an entry that is not a string (perhaps a null)')
    return rows


def ask_tokenizers(fault, function, *arguments, **options):
    """Call a function of the package tokenizers, and return its result.

    The package reports a failure with a plain Exception, which becomes a Rejection whose message
    begins with `fault`.
    """
    try:
        return function(*arguments, **options)
    except Exception as e:
        raise Rejection(f"{fault}: {one_line(e)}")


def load_tokenizer(path, data):
    """Return the tokenizer that a tokenizer.json file, given as bytes, describes."""
    fault = f"{path} is not a tokenizer file for the package tokenizers"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Rejection(f"{fault}: {one_line(e)}")
    return ask_tokenizers(fault, tokenizers.Tokenizer.from_str, text)


def npy_file(rows):
    """Return rows of ids as the bytes of a .npy file (format: step 7 at apply_rule())."""
    try:
        array = numpy.array(rows, dtype="<i4")
    except OverflowError:
        raise Rejection("a tokenizer gives an id above 2147483647, which the output file cannot store")
    buffer = io.BytesIO()
    numpy.lib.format.write_array(buffer, array, version=(1, 0))
    return buffer.getvalue()


def apply_rule(rows, llama2, llama4):
    """Apply steps 1 to 6 of the rule to the text rows.

    Return the drawn rows, the windows and a dict of five counts, which run() prints. The counts in
    parentheses below are those five for the three files in INPUTS. "Encode" and "decode" are
    methods of the class Tokenizer of the package tokenizers. The rule's numbers, among them the ids
    1, 2 and 200,000, are the same with --any-files, whatever those ids mean to other tokenizers.

    1. Text. Take the column "text" of the parquet file: all rows, in the file's order, including
       empty rows (36,718 rows). Join them with two line feeds as the separator.
    2. First tokenization. Encode the entire text in one call with the Llama 2 tokenizer, with
       add_special_tokens true. With the file in INPUTS this prepends the start token, id 1. Call
       the ids E and their number n (n = 2,874,559).
    3. Draws. Seed Python's random.Random with the integer 0 and call randint(0, n - 2048 - 1) 128
       times. A result i gives the drawn row E[i], .., E[i + 2047] of 2,048 ids. DRAWS_SHA256
       covers all 128 drawn rows. Only the first 48 go into the set.
    4. Prose. Concatenate the first 48 drawn rows, in the order of the draws, into one list of ids.
       Remove every id 1 and 2, the Llama 2 tokenizer's start and end tokens. Decode the list in one
       call with the Llama 2 tokenizer, with skip_special_tokens true (370,151 characters). The
       result is not a continuous passage, because each drawn row begins and ends wherever its
       random position falls. Text occurs in it twice where two drawn rows overlap, that is, start
       less than 2,048 ids apart (two pairs of the 48).
    5. Second tokenization. Encode the prose in one call with the Llama 4 tokenizer, with
       add_special_tokens false. Call the ids I (83,095 ids).
    6. Windows. Window w, for w = 0, .., 31, is the id 200,000, the model's start token, followed by
       the 1,023 ids I[1023 * w], .., I[1023 * w + 1022]. The rest of I is not used.
    7. File (npy_file(), called from run()). Write the windows as a .npy file of format version 1.0:
       an array of shape (32, 1024) and dtype "<i4" (little-endian 32-bit integers) in C order. The
       file has 131,200 bytes.
    """
    fault = "the first tokenizer (<llama2 tokenizer.json>) cannot encode the text"
    first_ids = ask_tokenizers(fault, llama2.encode, "\n\n".join(rows), add_special_tokens=True).ids
    if len(first_ids) <= DRAW_LENGTH:
        raise Rejection(
            f"too little text: the first tokenizer (<llama2 tokenizer.json>) gives {len(first_ids)} ids, "
            f"and the rule needs at least {DRAW_LENGTH + 1}"
        )
    # Python's documentation promises the same sequence in every version for random(), not for randint (The Python
    # Standard Library, "random", "Notes on Reproducibility"). DRAWS_SHA256 would show a change.
    generator = random.Random(SEED)
    starts = [generator.randint(0, len(first_ids) - DRAW_LENGTH - 1) for _ in range(DRAWS)]
    draws = [first_ids[start : start + DRAW_LENGTH] for start in starts]

    kept = [i for draw in draws[:DRAWS_USED] for i in draw if i not in REMOVED_IDS]
    fault = "the first tokenizer (<llama2 tokenizer.json>) cannot decode the drawn rows"
    prose = ask_tokenizers(fault, llama2.decode, kept, skip_special_tokens=True)
    fault = "the second tokenizer (<llama4 tokenizer.json>) cannot encode the prose"
    second_ids = ask_tokenizers(fault, llama4.encode, prose, add_special_tokens=False).ids
    step = WINDOW_LENGTH - 1
    if len(second_ids) < WINDOWS * step:
        raise Rejection(
            f"too little text: the second tokenizer (<llama4 tokenizer.json>) gives {len(second_ids)} ids "
            f"for the prose, and the rule needs at least {WINDOWS * step}"
        )
    windows = [[START_ID] + second_ids[step * w : step * (w + 1)] for w in range(WINDOWS)]

    used = sorted(starts[:DRAWS_USED])
    overlapping = sum(b - a < DRAW_LENGTH for at, a in enumerate(used) for b in used[at + 1 :])
    counts = {
        "text rows": len(rows),
        "first ids": len(first_ids),
        "overlapping pairs": overlapping,
        "characters": len(prose),
        "second ids": len(second_ids),
    }
    return draws, windows, counts


def write_new_file(path, data):
    """Create the file and write the bytes.

    Fail if anything exists at the path. If writing fails or is interrupted, remove the file.
    """
    try:
        f = open(path, "xb")
        made = os.fstat(f.fileno())
    except OSError as e:
        raise Rejection(f"cannot create {path}: {e.strerror}")
    try:
        with f:
            f.write(data)
    except OSError as e:
        remove_created(path, made)
        raise Rejection(f"cannot write {path}: {e.strerror}")
    except BaseException:  # For example, a KeyboardInterrupt (Ctrl-C).
        remove_created(path, made)
        raise


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


def run(paths, any_files):
    """Make the set from the four paths of the command line.

    Return the exit status, or raise a Rejection.
    """
    out = paths[3]
    if os.path.lexists(out):
        raise Rejection(
            f"{out} exists, and this program does not overwrite files. Give a path that does not exist yet."
        )
    folder = os.path.dirname(out) or "."
    if not os.path.isdir(folder):
        raise Rejection(f"cannot create {out}: {folder} is not an existing folder")
    # All three files are read and checked before the first is parsed.
    data = [read_input(path, named, any_files) for path, named in zip(paths, INPUTS)]
    draws, windows, counts = apply_rule(
        text_rows(paths[0], data[0]), load_tokenizer(paths[1], data[1]), load_tokenizer(paths[2], data[2])
    )
    draws_file, set_file = npy_file(draws), npy_file(windows)
    write_new_file(out, set_file)

    say(
        f"drew {DRAWS} rows of {DRAW_LENGTH} ids from the {counts['first ids']} ids that the first tokenizer gives for "
        f"the {counts['text rows']} rows of text. They are not written. As a .npy file they have "
        f"{len(draws_file)} bytes and SHA-256 {sha256(draws_file)}"
    )
    say(
        f"wrote {out}: shape ({WINDOWS}, {WINDOW_LENGTH}), {len(set_file)} bytes, SHA-256 {sha256(set_file)}. "
        f"The prose has {counts['characters']} characters, and the second tokenizer gives {counts['second ids']} ids "
        f"for it. {counts['overlapping pairs']} pairs of the first {DRAWS_USED} drawn rows overlap."
    )
    if any_files:
        return WRITTEN
    draws_differ, set_differs = sha256(draws_file) != DRAWS_SHA256, sha256(set_file) != SET_SHA256
    if draws_differ:
        say(f"the drawn rows are not as expected: their SHA-256 should be {DRAWS_SHA256}")
    else:
        say("the drawn rows are as expected")
    if set_differs:
        say(f"the set is not as expected: its SHA-256 should be {SET_SHA256}. The file is written all the same.")
    else:
        say("the set is as expected")
    if draws_differ or set_differs:
        say(
            f"the expected sums were checked with {CHECKED_WITH}. This run uses Python {sys.version.split()[0]}, "
            f"numpy {numpy.__version__}, pyarrow {pyarrow.__version__}, tokenizers {tokenizers.__version__}. "
            "If the versions differ, run the program again with the first ones and a new output path."
        )
        return DIFFERS
    return WRITTEN


def main(argv):
    paths = [argument for argument in argv if argument != "--any-files"]
    if len(paths) != 4:
        print(USAGE, file=sys.stderr)
        return REJECTED
    try:
        return run(paths, any_files=len(paths) < len(argv))
    except Rejection as rejection:
        say(str(rejection), sys.stderr)
    except Exception:
        # Uncaught, the exception would make Python exit with status 1 (DIFFERS).
        traceback.print_exc()
    return REJECTED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
