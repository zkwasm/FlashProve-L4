#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Checks that each copy of open_above(), others_can_change() and available_memory() is as agreed.

usage: python3 -B tests/folder_look_copies_test.py

Some scripts here refuse to work in a folder that another account could change. They make that
check with one of two routines, open_above() in shell or others_can_change() in Python. Some
example scripts also carry available_memory(), a shell function that finds how much memory they
may use. Each script carries its own copy of a routine, so that it can run alone. A copy that is
not exactly the agreed text may have drifted: a fix made in one copy and not in the others, or an
edit by hand. This test finds every copy and compares the SHA-256 sum of its text with the sum of
the agreed text.

How a copy's text is taken:
  - open_above(), in .sh and .bash files: the line "open_above() {" (at the start of a line), the
    comment lines directly above it, starting at the line SHELL_FIRST_LINE when that line is among
    them, and the lines below it up to the first line that is only "}", each followed by a line
    feed. Its sum must be SHELL_SUM.
  - others_can_change(), in .py files: the constants and functions of PYTHON_NAMES, each taken as
    written from the file's top level, then laid out as the agreed copy lays them out:
    PYTHON_HEADER, a blank line, the constants one a line, two blank lines, and the functions with
    two blank lines between them. Comments outside a part's own code (above it, between parts, or
    after its last statement) are not part of the sum; any change to the code of a part, comments
    inside a function included, is. Its sum must be PYTHON_SUM.
  - available_memory(), in .sh and .bash files: the line "available_memory() {" (at the start of a
    line) and the lines below it up to the first line that is only "}", each followed by a line
    feed; the comments above it are not part of the sum. Its sum must be MEMORY_SUM.

Also a failure: a file or folder that cannot be read; a shell file with more than one definition
of open_above() or of available_memory(), one in another form, or one with no line that is only
"}" after it; a Python file that cannot be parsed and names others_can_change; a Python file that
defines others_can_change other than as a function at its top level; a Python copy that lacks one
of PYTHON_NAMES, binds one of them more than once at its top level (also by an import, a class, a
loop or an assignment after the block), puts a decorator on one of them, or also imports
others_can_change; and «from <module> import others_can_change» where <module> is not a copy in the
same folder, named without dots but for one leading dot («from .spool import …»). An import from
such a copy is named ("imports its copy from <file>") and passes.

To change a routine on purpose, change every copy the same way, run this test, and set SHELL_SUM,
PYTHON_SUM or MEMORY_SUM to the sum that it then prints for every copy.

Not seen: a copy brought in by «from <module> import *», and a rebinding by :=, except ... as, match
or exec; in shell, a definition made through eval, in a file that the script sources, or after
«unset -f open_above» (or available_memory), and whether a copy stands inside an if or another
function.

COPIES names every copy by its file and its kind: "shell" for open_above(), "python" for
others_can_change(), "available_memory" for available_memory(); a file that holds two routines
has a line for each. A copy that COPIES lists and that the file does not hold fails ("restore the
copy, or, if it was removed on purpose, remove its line (path, kind) from COPIES"; not said of a
copy that has a finding of its own, such as two definitions), and so does a copy that a file holds
and COPIES does not list ("add the line (path, kind) to COPIES"), and a listed kind that is not the
kind of the copy found ("correct its kind in COPIES"). Whoever adds or removes a copy adds or
removes its line in COPIES in the same change.

The search starts at the folder above this file's folder, the tree's top, and walks every folder
below it. It opens only .sh, .bash and .py files, and never follows a symbolic link. It reads
files only: it runs nothing and writes nothing.

Each copy is printed as "path: the agreed text" or "path: differs from the agreed text (sha256
<sum>); ..." with what to do (for available_memory(), "path: available_memory(): the agreed text"
and so on), and each other finding as "path: <what is wrong>". The last line is
"folder_look_copies_test.py: N copies of open_above(), others_can_change() and available_memory(),
all unchanged", where N is the number of lines in COPIES, or "...: K copies of open_above(),
others_can_change() or available_memory() differ from the agreed text; N copies are listed in
COPIES; J other findings", where K counts every copy that differs, listed or not, and J every other
line that names a file and what is wrong.

Exit status: 0 if every copy is the agreed text and nothing else was found, 1 otherwise, and 2 if
the test is given an argument.
"""

import ast
import hashlib
import os
import re
import sys

ME = "folder_look_copies_test.py"
TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_ENDINGS = (".sh", ".bash")

SHELL_SUM = "e0de355bd18633461bbc223e9938a07c9c57b587ca37e84b087c42e6591db536"
SHELL_DEFINITION = "open_above() {"
SHELL_FIRST_LINE = "# <absolute path of a directory>: walks it from the top as the system does, through each symbolic"
# any line that defines open_above, in whatever form
SHELL_ANY_DEFINITION = re.compile(r"^\s*(?:function\s+open_above\b|open_above\s*\(\s*\))")
SHELL_FIX = "copy open_above() unchanged from a file that this test reports as the agreed text"

PYTHON_SUM = "074f31044e834d727586a54594240539bdb4428bb287a6afebb91e5e20f5f2ab"
PYTHON_CONSTANTS = ("ACL", "OPEN", "THEIRS", "UNSEEN", "GONE", "INVISIBLE")
PYTHON_FUNCTIONS = ("printable", "trusted_owners", "others_can_change", "refusal", "whose", "cure")
PYTHON_NAMES = PYTHON_CONSTANTS + PYTHON_FUNCTIONS
PYTHON_FIX = "copy others_can_change() and its block unchanged from run_suite.py"
# available_memory(), a shell function that some example scripts carry to find how much memory they may use
MEMORY_SUM = "8966795b8006bf382a0e10597520e33f7cf8851c4640b0aba1826e846ef17b73"
MEMORY_DEFINITION = "available_memory() {"
MEMORY_ANY_DEFINITION = re.compile(r"^\s*(?:function\s+available_memory\b|available_memory\s*\(\s*\))")
MEMORY_FIX = "copy available_memory() unchanged from a file that this test reports as the agreed text"
ROUTINES = {"shell": "open_above()", "python": "others_can_change()", "available_memory": "available_memory()"}
# Every copy, by its file and its kind: "shell" for open_above(), "python" for others_can_change(), "available_memory"
# for available_memory(). Whoever adds or removes a copy adds or removes its line here in the same change.
COPIES = (
    ("blocks/first_look.sh", "shell"),
    ("driver/spool.py", "python"),
    ("examples/maverick/run.sh", "shell"),
    ("examples/maverick/run.sh", "available_memory"),
    ("examples/quickstart-gpu.sh", "shell"),
    ("examples/quickstart-gpu.sh", "available_memory"),
    ("examples/rejected-proofs/try_altered_copies.py", "python"),
    ("integrations/check_proof.py", "python"),
    ("integrations/spool.py", "python"),
    ("integrations/vllm/integer_model/examples/try_limits.py", "python"),
    ("integrations/vllm/integer_model/integer_model_vllm/example.py", "python"),
    ("integrations/vllm/integer_model/tests/conftest.py", "python"),
    ("prover/build.sh", "shell"),
    ("prover/examples/run_target_example.sh", "shell"),
    ("prover/examples/run_target_example.sh", "available_memory"),
    ("prover/tests/unit.sh", "shell"),
    ("run_suite.py", "python"),
    ("scales/calibrate.py", "python"),
    ("tests/new/run_new.py", "python"),
    ("tests/new/second/run_honest.sh", "shell"),
    ("tools/accepts/accepts.sh", "shell"),
    ("tools/audit/audit.py", "python"),
    ("tools/audit/audit_batch.sh", "shell"),
    ("tools/audit/audit_commitment.sh", "shell"),
    ("tools/audit/play.sh", "shell"),
    ("tools/parser/parse_files.py", "python"),
    ("tools/parser/parse_files.sh", "shell"),
    ("toy/quick_start_test.py", "python"),
    ("verifier/build.sh", "shell"),
    ("verifier/tests/clang_build_test.sh", "shell"),
    ("verifier/tests/compiler_pass_control.sh", "shell"),
    ("weight_sums/build.sh", "shell"),
    ("weight_sums/tests/run_tests.sh", "shell"),
)
PYTHON_HEADER = '''# The look that run_suite.py makes before it works below a folder, each part to the letter:
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
'''


def sha256(text):
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def shell_copy(text):
    """(the copy's text, None) for a shell file that defines open_above() once, in the agreed form; (None, what is
    wrong) for one that defines it otherwise; (None, None) for one that does not define it."""
    lines = text.split("\n")
    defining = [n for n, line in enumerate(lines) if SHELL_ANY_DEFINITION.match(line)]
    if not defining:
        return None, None
    if len(defining) > 1:
        return None, "defines open_above() %d times (lines %s); keep only one, the agreed copy" % (
            len(defining), ", ".join(str(n + 1) for n in defining))
    at = defining[0]
    if lines[at] != SHELL_DEFINITION:
        return None, "defines open_above() in another form (line %d); %s" % (at + 1, SHELL_FIX)
    start = at
    while start > 0 and lines[start - 1].startswith("#"):
        start -= 1
    if SHELL_FIRST_LINE in lines[start:at]:
        start = lines.index(SHELL_FIRST_LINE, start, at)
    end = next((n for n in range(at + 1, len(lines)) if lines[n] == "}"), None)
    if end is None:
        return None, "defines open_above() with no line that is only \"}\" after it (line %d)" % (at + 1)
    return "".join(line + "\n" for line in lines[start : end + 1]), None


def memory_copy(text):
    """(the copy's text, None) for a shell file that defines available_memory() once, in the agreed form: its line
    MEMORY_DEFINITION and the lines below it up to the first line that is only "}", each followed by a line feed;
    (None, what is wrong) for one that defines it otherwise; (None, None) for one that does not define it."""
    lines = text.split("\n")
    defining = [n for n, line in enumerate(lines) if MEMORY_ANY_DEFINITION.match(line)]
    if not defining:
        return None, None
    if len(defining) > 1:
        return None, "defines available_memory() %d times (lines %s); keep only one, the agreed copy" % (
            len(defining), ", ".join(str(n + 1) for n in defining))
    at = defining[0]
    if lines[at] != MEMORY_DEFINITION:
        return None, "defines available_memory() in another form (line %d); %s" % (at + 1, MEMORY_FIX)
    end = next((n for n in range(at + 1, len(lines)) if lines[n] == "}"), None)
    if end is None:
        return None, "defines available_memory() with no line that is only \"}\" after it (line %d)" % (at + 1)
    return "".join(line + "\n" for line in lines[at : end + 1]), None


def module_bindings(body):
    """Every binding at a module's top level, as (name, node, line): definitions, classes, assignments, imports, and
    loop and with targets, also inside top-level if, try, for, while and with statements."""
    out = []

    def names(target):
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for e in target.elts for n in names(e)]
        if isinstance(target, ast.Starred):
            return names(target.value)
        return []

    todo = list(body)
    while todo:
        node = todo.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, node, node.lineno))
            continue
        if isinstance(node, ast.Assign):
            out += [(n, node, node.lineno) for t in node.targets for n in names(t)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            out += [(n, node, node.lineno) for n in names(node.target)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out += [((a.asname or a.name).split(".")[0], node, node.lineno) for a in node.names if a.name != "*"]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            out += [(n, node, node.lineno) for n in names(node.target)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            out += [(n, node, node.lineno) for item in node.items if item.optional_vars for n in names(item.optional_vars)]
        for field in ("body", "orelse", "finalbody"):
            todo += getattr(node, field, []) if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []
        for handler in getattr(node, "handlers", []):
            todo += handler.body
    return out


def python_copy(text):
    """(the copy's text, what is wrong, the modules it takes others_can_change from, as (module, line)) for a Python
    file; the copy's text is None for a file that holds no copy."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None, ("cannot be read as Python, and it names others_can_change" if "others_can_change" in text else None), []
    imports = [(("." * node.level) + (node.module or ""), node.lineno) for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and any(a.name == "others_can_change" for a in node.names)]
    top_def = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "others_can_change"]
    elsewhere = sorted(node.lineno for node in ast.walk(tree)
                       if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == "others_can_change"
                           or isinstance(node, ast.Name) and node.id == "others_can_change" and isinstance(node.ctx, ast.Store))
                       and node not in top_def)
    if not top_def:
        if elsewhere:
            return None, "defines others_can_change() other than as a function at its top level (line %s)" % (
                ", ".join(map(str, elsewhere))), []
        return None, None, imports
    bound = {}
    for name, node, line in module_bindings(tree.body):
        if name in PYTHON_NAMES:
            bound.setdefault(name, []).append((node, line))
    wrong = []
    missing = [name for name in PYTHON_NAMES if name not in bound]
    if missing:
        wrong.append("without the rest of its block (no %s)" % ", ".join(missing))
    for name in PYTHON_NAMES:
        if len(bound.get(name, [])) > 1:
            wrong.append("%s is bound %d times at the top level (lines %s)" % (
                name, len(bound[name]), ", ".join(str(line) for _, line in bound[name])))
    for name in PYTHON_FUNCTIONS:
        for node, line in bound.get(name, []):
            if not isinstance(node, ast.FunctionDef) or node not in tree.body:
                wrong.append("%s is not a plain function at the top level (line %d)" % (name, line))
            elif node.decorator_list:
                wrong.append("%s has a decorator (line %d)" % (name, line))
    for name in PYTHON_CONSTANTS:
        for node, line in bound.get(name, []):
            if not isinstance(node, ast.Assign) or node not in tree.body:
                wrong.append("%s is not a plain assignment at the top level (line %d)" % (name, line))
    inner = [line for line in elsewhere if line not in [line for _, line in bound.get("others_can_change", [])]]
    if inner:
        wrong.append("others_can_change is also defined inside something (line %s)" % ", ".join(map(str, inner)))
    if imports:
        wrong.append("it also imports others_can_change (line %s)" % ", ".join(str(line) for _, line in imports))
    if wrong:
        return None, "holds a copy of others_can_change() in another way: %s; %s" % ("; ".join(wrong), PYTHON_FIX), []
    part = {name: ast.get_source_segment(text, bound[name][0][0]) for name in PYTHON_NAMES}
    body = "\n".join(part[n] for n in PYTHON_CONSTANTS) + "\n\n\n" + "\n\n\n".join(part[n] for n in PYTHON_FUNCTIONS) + "\n"
    return PYTHON_HEADER + "\n" + body, None, []


def main(argv):
    if argv:
        print("usage: python3 -B tests/folder_look_copies_test.py    (it takes no arguments)")
        return 2
    # found: (path, kind) of each copy found; flagged: (path, kind) of each routine with a finding of its own
    found, differ, other, imports, flagged = [], 0, 0, [], set()

    def cannot(error):
        nonlocal other
        print("%s: cannot be read (%s)" % (os.path.relpath(error.filename or TOP, TOP), error.strerror or error))
        other += 1

    def report(rel, kind, copy, wrong, agreed, fix, label):
        nonlocal differ, other
        if wrong:
            print("%s: %s" % (rel, wrong))
            other += 1
            flagged.add((rel, kind))  # its own finding says what is wrong; COPIES does not add a second one
        elif copy is not None:
            found.append((rel, kind))
            if sha256(copy) == agreed:
                print("%s: %sthe agreed text" % (rel, label))
            else:
                print("%s: %sdiffers from the agreed text (sha256 %s); %s" % (rel, label, sha256(copy), fix))
                differ += 1

    for folder, folders, names in os.walk(TOP, onerror=cannot):
        folders[:] = sorted(d for d in folders if not os.path.islink(os.path.join(folder, d)))
        for name in sorted(names):
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, TOP)
            if not name.endswith(SHELL_ENDINGS + (".py",)) or os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    text = f.read().decode("utf-8", "surrogateescape")
            except OSError as error:
                print("%s: cannot be read (%s)" % (rel, error.strerror or error))
                other += 1
                continue
            if name.endswith(SHELL_ENDINGS):
                copy, wrong = shell_copy(text)
                report(rel, "shell", copy, wrong, SHELL_SUM, SHELL_FIX, "")
                copy, wrong = memory_copy(text)
                report(rel, "available_memory", copy, wrong, MEMORY_SUM, MEMORY_FIX, "available_memory(): ")
            else:
                copy, wrong, taken = python_copy(text)
                report(rel, "python", copy, wrong, PYTHON_SUM, PYTHON_FIX, "")
                imports += [(rel, module, line) for module, line in taken]
    for rel, module, line in imports:
        beside = module[1:] if module.startswith(".") and module.count(".") == 1 else module
        source = os.path.join(os.path.dirname(rel), beside + ".py") if beside and "." not in beside else None
        if source is not None and (source, "python") in found:
            print("%s: imports its copy from %s" % (rel, source))
        else:
            print("%s: takes others_can_change from %s (line %d), which is not a copy in its folder; %s"
                  % (rel, module or ".", line, PYTHON_FIX))
            other += 1
    listed = set(COPIES)
    for rel, kind in COPIES:
        if kind not in ROUTINES:
            print("%s: COPIES gives it the kind %s, which is none of %s; correct its kind in COPIES"
                  % (rel, kind, ", ".join(sorted(ROUTINES))))
            other += 1
            continue
        held = kind
        if kind in ("shell", "python"):
            held = "shell" if rel.endswith(SHELL_ENDINGS) else "python"
        if kind != held and (rel, held) in found:
            print("%s: COPIES lists its copy as %s, and the file holds a %s copy; correct its kind in COPIES"
                  % (rel, kind, held))
            other += 1
        elif (rel, kind) not in found and (rel, kind) not in flagged and (rel, held) not in flagged:
            print("%s: COPIES lists it as holding a copy of %s, and it holds none; restore the copy, or, if it was"
                  " removed on purpose, remove its line (\"%s\", \"%s\") from COPIES" % (rel, ROUTINES[kind], rel, kind))
            other += 1
    for rel, kind in found:
        # a file listed as shell where it holds python, or the other way round, has its finding above already
        either = kind in ("shell", "python") and ((rel, "shell") in listed or (rel, "python") in listed)
        # and a path that COPIES lists under an unknown kind has its finding above too
        unknown = any(path == rel and k not in ROUTINES for path, k in COPIES)
        if (rel, kind) not in listed and not either and not unknown:
            print("%s: holds a copy of %s, and COPIES does not list it; add the line (\"%s\", \"%s\") to COPIES"
                  % (rel, ROUTINES[kind], rel, kind))
            other += 1
    n = len(COPIES)
    if differ or other:
        print("%s: %d %s of open_above(), others_can_change() or available_memory() %s from the agreed text; %d copies are"
              " listed in COPIES; %d other %s" % (ME, differ, "copy" if differ == 1 else "copies",
                                                 "differs" if differ == 1 else "differ", n, other,
                                                 "finding" if other == 1 else "findings"))
        return 1
    print("%s: %d copies of open_above(), others_can_change() and available_memory(), all unchanged" % (ME, n))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
