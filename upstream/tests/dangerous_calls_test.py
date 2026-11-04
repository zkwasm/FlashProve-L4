#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Searches the text of this tree's scripts, sources and documents for calls that can signal or
delete the wrong target. A pass shows only that the test found none of the spellings that it looks
for: of five faulty forms, and of lines that end processes or delete recursively without a count and
a reason in a table. It does not show that every call is safe: see "Not detected" below.

usage: dangerous_calls_test.py [<tree>]

<tree> defaults to the parent of this file's directory; the table is the dangerous_calls.tsv
beside this file, read together with each further table of the same form beside it
(dangerous_calls.<word>.tsv whose first line is MARK), where one stands; a file listed in two
tables is a finding. (Identifiers and messages say `folder` for a directory.)
Nothing in the tree is run or written: the self-test uses a new temporary directory.
Files read: shell (.sh .bash .job .mk .yaml .yml, Makefile, and a file named CHEATS or CHEATS.more
if there is one: this repository has none), Python, C, C++ and CUDA sources, documents (.md .txt;
"page" in the code), and any other file whose "#!" line names a shell or Python. Not read: symbolic
links, files with this test's name or named dangerous_calls.tsv or dangerous_calls.<word>.tsv,
the directories that UNREAD names or BUILT matches, and what store() finds: a version-control store,
a Python virtual environment, and what a package's install leaves beside its pyproject.toml or
setup.py.
Files below a directory named `hooks` are counted against a table of their own (HOOKS).
Comments in code are skipped, but Python docstrings are read. In a document, only what a reader
might run is read: fenced blocks, indented blocks and text between backquotes.

Two types of finding:
  1. A faulty form (a key of HARD). In code, no table entry excuses it: fix the line. A document
     may quote one to describe it, and there it is counted like a line of type 2.
     - cd-pwd: any $(cd ... && pwd), or the same in backquotes, except an assignment
       x=$(cd ... && pwd) followed on its line by "||" or "&&". Also rejected: a further $( ) after
       it, whose status would replace the cd's; local, export, readonly, declare or typeset before
       x, whose own status hides the failure; and any $(cd ...; pwd). An empty x turns
       "rm -rf $x/tmp" into "rm -rf /tmp".
     - fork: in C, C++ or CUDA, a fork() unless its result is assigned to a variable that is
       compared with 0 (by <, >= or >) or with -1 on the same line or the next four. A failed fork()
       returns -1; kill(-1, ...) then signals every process of the user, and waitpid(-1, ...) waits
       for any child.
     - kill-all: a signal sent to -1 (every process of the user) or 0 (the caller's process group).
     - kill-from-file: a signal to a number read from a file in the same command. Nothing shows
       that the number still belongs to the process that wrote it.
     - rm-var-slash: a recursive rm of a path that begins with a variable and a slash ("$x/..." or
       ${x}/...). With "${x:?}/..." the shell stops if x is empty or unset (shellcheck SC2115).
  2. A line that ends processes or deletes recursively: kill, killpg, pidfd_send_signal,
     send_signal, pkill, killall, Python's .kill() and .terminate(), rm -r, rmtree, removedirs,
     remove_all, find -delete, and any use of $! (a stale $! names a process that may be gone and
     whose number may be reused). rmtree, .kill and .terminate also count where they are passed as
     a function, as to addCleanup or atexit. For each file, the table gives the number of such
     lines of each kind (a key of LISTED_KINDS; in a document, also of HARD) and the reason why
     they are safe. One line more or fewer fails the test.

A file that cannot be read is a finding too.
When nothing named lean stands at the tree's top (not a directory, a file or a broken link), the
table's lines for files below lean/ are not compared, and the closing line counts them. A tree
without lean/ therefore passes whatever those lines say. Any other missing file or top directory is
a finding.

Not detected, among other things:
  - On a counted line: a new target, or a second call (the count of lines stays). A call swapped
    for another of the same kind in the same file. A target that the table wrongly calls safe.
  - A call under another name, in a variable, assembled from pieces or split over lines by a
    backslash; a signal sent through syscall(); "/bin/kill -9 -1"; Python's os.fork().
  - kill-all: "kill -s KILL -1", "kill -9 -1 $x".
  - kill-from-file: a number read into a variable first (whether or not anything then checks whose
    number it is); one read by a program other than a bare cat (head, tail, sed, read, xargs,
    pkill -F, /bin/cat), by Python's f.read() or read_text(), or in C or C++.
  - cd-pwd: "x=$(cd a && pwd)/`foo` || exit 2"; a cd inside the cd's own argument.
  - rm-var-slash: "rm -rf ${W%/}/tmp".
  - fork: "> 0" counts as a comparison, although it does not tell -1 from 0.
  - rmtree or a kill method bound to a name first (x = rmtree, key=rmtree, x = p.kill, [p.kill]);
    rmtree passed under another module name (sh.rmtree without parentheses, after
    import shutil as sh).
  - A script that someone puts into an output directory that BUILT matches.
  - A script inside a directory that store() takes for a tool's own: one that holds wc.db, or HEAD
    and a directory objects/, or pyvenv.cfg, or a build/ or *.egg-info beside a pyproject.toml or
    setup.py. Such a directory is skipped with everything below it.
  - Not searched for at all: the removal of one file or one empty directory (rm without -r, unlink,
    unlinkat, rmdir, os.remove; the programs' own removal functions among them), renaming,
    emptying (truncate, ">"), and fixed paths in a directory that others share, such as /tmp.
  - In shell and Python files, text after " # " is taken for a comment even inside a string.
  - A .txt file that holds commands in plain lines is read as a document, whose plain lines are
    skipped, so such commands are not seen.
  - A table line for lean/ whose file has gone, in a tree that has no lean/ at all (see above).
Exit code 0: all as expected; table lines for lean/ passed over as above do not change this. 1: a
finding, and a printed line says which. 2: the self-test (SAMPLES and further cases), which runs
first, failed; the tree was not read.
"""

import os
import re
import sys
import tempfile

UNREAD = ("__pycache__",)  # skipped at any depth: bytecode; so is what store() below finds


def store(path):
    """True for a directory that a tool made and keeps, not a part of the sources: a version-control
    store (one that holds HEAD and objects/, for git, or wc.db, for another tool's working copy), a Python
    virtual environment (one that holds pyvenv.cfg), and what a Python package's install leaves beside
    its pyproject.toml or setup.py (build/ and a folder whose name ends in .egg-info)."""
    has = lambda name: os.path.exists(os.path.join(path, name))
    if (has("HEAD") and os.path.isdir(os.path.join(path, "objects"))) or has("wc.db") or has("pyvenv.cfg"):
        return True
    up = os.path.dirname(path)
    beside_a_package = os.path.isfile(os.path.join(up, "pyproject.toml")) or os.path.isfile(os.path.join(up, "setup.py"))
    return beside_a_package and (os.path.basename(path) == "build" or path.endswith(".egg-info"))


# The build scripts' default output directories, from the tree's top: prover/build (prover/build.sh)
# and bin (verifier/build.sh, weight_sums/build.sh). Also matched: either one with a suffix (bin-x).
BUILT = re.compile(r"^(?:bin|prover/build)(?:-[A-Za-z0-9]+)?$")
# A directory of this name contains test hooks (code compiled only into test programs) and their
# tests. Its files are listed in a table inside it, <that directory>/dangerous_calls.tsv, by their
# paths below it; no other table of the repository may name that
# directory's path. A directory of this name inside another uses the outermost one's table.
HOOKS = "hooks"
SHELL = (".sh", ".bash", ".job", ".mk", ".yaml", ".yml")
PYTHON = (".py",)
C = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".inc")
PAGE = (".md", ".txt")
THIS = os.path.basename(__file__)
# The cd-pwd form that passes: x=$(cd ... && pwd), with no further $( ) after it, then || or &&
CHECKED_CD = re.compile(
    r"(?<![\w-])\w+=\"?\$\(\s*cd\s((?!;\s*pwd\b).)*?&&\s*pwd(\s+-[LP])?\s*\)(\$\{[^}]*\}|\$\w+|[^\s;|&$])*\s*(\|\||&&)"
)

# Name -> pattern, searched line by line. HARD: the faulty forms. LISTED_KINDS: the counted lines.
HARD = {
    "cd-pwd": re.compile(r"(\$\(|\x60)\s*(\w+=\S*\s+)*cd\s.*?(&&|;)\s*pwd\b"),  # \x60: a backquote
    "fork": re.compile(r"(?<![\w.])fork\s*\(\s*\)"),
    "kill-all": re.compile(
        r"(?<![\w.-])(os\.)?kill(pg)?\s*\(\s*(-\s*1|0)\s*,"
        r"|(?<![\w./-])kill(\s+-[-\w]+)*\s+(--\s+)?(-1|0)(?![\w.$-])(?=[\"'\x60]|\s*($|[;&|)}#<>\\]|\d+>))"  # -1 or 0 as the last word
    ),
    "kill-from-file": re.compile(  # a number read back from a file, signalled on the same line: it may be another's by now
        r"(?<![\w./-])kill\b[^;&|]*(\$\(\s*(cat\s|<)|\x60\s*cat\s)|\bkill(pg)?\s*\(\s*int\s*\(\s*open\s*\("
    ),
    "rm-var-slash": re.compile(
        r"(?<![\w-])rm(\s+-[a-zA-Z-]+)*\s+(-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)(\s+-[a-zA-Z-]+)*\s+"
        r"([^;&|]*\s)?\"?(\$\w+\"?/|\$\{\w+\}\"?/)"
    ),
}
LISTED_KINDS = {
    "kill": re.compile(
        r"(?<![\w.-])(kill|killpg)\s*\(|(?<![\w./-])kill(\s+[-$\"'0-9%(]|\s*$|\s*[;|&)])|xargs\s+(-\S+\s+)*kill\b"
        r"|[\"']kill[\"']\s*,|\bos\.kill(pg)?\s*\(|\bpidfd_send_signal\s*\(|\.send_signal\s*\("
    ),
    "pkill": re.compile(r"\b(pkill|killall)\b"),
    "kill-method": re.compile(r"\.(kill|terminate)\s*\(\s*\)|\.(kill|terminate)\s*[,)]"),  # a call, or the method passed as an argument (addCleanup)
    "rm-r": re.compile(
        r"(?<![\w-])rm(\s+-[a-zA-Z-]+)*\s+(-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\b"
        r"|[\"']rm[\"']\s*,\s*([\"']-[a-zA-Z-]+[\"']\s*,\s*)*[\"'](-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)[\"']"
    ),
    "rmtree": re.compile(r"\brmtree\s*\(|\bremovedirs\s*\(|\bshutil\.rmtree\b(?!\s*\()|[(,]\s*rmtree\s*[,)]"),  # a call, or the function passed as an argument (addCleanup, atexit)
    "remove_all": re.compile(r"\bremove_all\s*\("),
    "find-delete": re.compile(r"\bfind\b.*\s-delete\b"),
    "bang-pid": re.compile(r"\$!|\$\{!\}"),
}
HERE = os.path.dirname(os.path.abspath(__file__))
# Columns, tab-separated: path from the tree's top, kind, number of lines, the reason they are
# safe; "#" begins a comment line.
TABLE = os.path.join(HERE, "dangerous_calls.tsv")
# Further tables of the same form may stand beside it (dangerous_calls.<word>.tsv, whose first line is MARK); each one found
# is read with it, and a file listed in two tables is a finding.
MARK = "# a further table for tests/dangerous_calls_test.py"


def first_line(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.readline().rstrip("\n")
    except (OSError, ValueError):
        return ""


MORE = sorted(os.path.join(HERE, n) for n in os.listdir(HERE) if n.startswith("dangerous_calls.") and n.endswith(".tsv") and n != "dangerous_calls.tsv" and first_line(os.path.join(HERE, n)) == MARK)


def listed(path=TABLE, shown=None):
    """Returns {file: {kind: (lines, reason)}} and a message for each line that is malformed or
    repeats a file and kind. shown: the table's name in those messages (default: its base name)."""
    out, wrong = {}, []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.rstrip("\n").split("\t")
            if len(cells) != 4 or cells[1] not in LISTED_KINDS and cells[1] not in HARD or not cells[2].isdigit() or not cells[3].strip():
                wrong.append("%s:%d: not «path, kind, lines, reason»: %s" % (shown or os.path.basename(path), n, line.strip()[:120]))
            elif cells[1] in out.get(cells[0], {}):
                wrong.append("%s:%d: %s %s stands twice" % (shown or os.path.basename(path), n, cells[0], cells[1]))
            else:
                out.setdefault(cells[0], {})[cells[1]] = (int(cells[2]), cells[3])
    return out, wrong


def tables():
    """Returns the tree's table merged with each further table beside it, a message for each malformed line and for
    each file listed in two tables, and, for each file of a further table, that table's name (used in findings)."""
    table, wrong = listed()
    homes = {}
    for extra in MORE:
        if not os.path.isfile(extra):
            continue
        more, also = listed(extra)
        wrong += also
        for path, kinds in more.items():
            if path in table:
                wrong.append("%s stands in both %s and %s" % (path, homes.get(path, os.path.basename(TABLE)), os.path.basename(extra)))
            else:
                table[path] = kinds
                homes[path] = os.path.basename(extra)
    return table, wrong, homes


# Returns how a file is read: "shell", "python", "c" or "page"; None if it is not read.
def kind_of(path):
    name = os.path.basename(path)
    if name.endswith(SHELL) or name in ("Makefile", "CHEATS", "CHEATS.more"):
        return "shell"
    for suffixes, kind in ((PYTHON, "python"), (C, "c"), (PAGE, "page")):
        if name.endswith(suffixes):
            return kind
    try:
        with open(path, "rb") as f:
            head = f.readline(200).decode("latin-1")
    except OSError:
        return "shell"  # findings() then reports that it cannot be read
    if head.startswith("#!"):
        return "python" if "python" in head else "shell" if re.search(r"\b(ba|da|z)?sh\b", head) else None
    return None


def c_code(text):
    """Returns the lines of C text without comments, line numbers unchanged. String, character and
    raw string literals are kept: a string may contain a command."""
    out, i, n, state = [], 0, len(text), None
    while i < n:
        ch = text[i]
        if state is None:
            if text.startswith("//", i) or text.startswith("/*", i):
                state, i = text[i + 1], i + 2  # "/" or "*"
                continue
            raw = re.match(r'R"([^(\s]*)\(', text[i:]) if text.startswith('R"', i) else None
            if raw:
                end = text.find(")" + raw.group(1) + '"', i)
                end = n if end < 0 else end + len(raw.group(1)) + 2
                out.append(text[i:end])
                i = end
                continue
            # A ' after a letter or digit is taken for a digit separator (1'000), not a literal.
            if ch == '"' or ch == "'" and not (i and text[i - 1].isalnum()):
                state = ch
            out.append(ch)
            i += 1
        elif state in "/*":
            if state == "*" and text.startswith("*/", i):
                state, i = None, i + 2
                continue
            if ch == "\n":
                out.append(ch)
                state = None if state == "/" else state
            i += 1
        else:
            if ch == "\\":
                out.append(text[i : i + 2])
                i += 2
                continue
            if ch == state or ch == "\n":
                state = None
            out.append(ch)
            i += 1
    return "".join(out).split("\n")


def code_of(lines, kind):
    """Returns the lines, reduced to what is searched (see the top of this file). In a document, an
    indented block (four spaces or a tab) must begin the file or follow a blank line, as in
    Markdown."""
    if kind == "c":
        return c_code("\n".join(lines))
    out, fence, indented, prev = [], None, False, ""
    for line in lines:
        if kind == "page":  # what a reader may run: fenced and indented blocks, and text between backquotes
            blank_before = not out or not out[-1].strip() and not prev.strip()
            prev = line
            mark = line.lstrip()[:3]
            if mark in ("```", "~~~") and fence in (None, mark):
                fence = None if fence else mark
                line = ""
            elif not fence and not (re.match(r"( {4}|\t)", line) and (blank_before or indented)):
                indented = False
                line = " ".join(re.findall(r"`([^`]+)`", line))
            elif not fence:
                indented = True
        elif kind in ("shell", "python"):
            line = "" if line.lstrip().startswith("#") and not line.startswith("#!") else re.sub(r"\s#\s.*", "", line)
        out.append(line)
    return out


def fork_unchecked(lines, i):
    """Whether lines[i] to lines[i + 4] never compare the result of the fork() on lines[i]."""
    m = re.search(r"(\w+)\s*=\s*(::)?fork\s*\(", lines[i])
    if not m:
        return True  # if (!fork()), or a result that is never named
    v = re.escape(m.group(1))
    near = "\n".join(lines[i : i + 5])
    looks = r"\b%s\s*(<|>=|>)\s*0\b|\b%s\s*[!=]=\s*-1\b|\b0\s*(>|<=|<)\s*%s\b|\(\s*%s\s*=\s*(::)?fork\s*\(\s*\)\s*\)\s*(<|>=)\s*0\b"
    return not re.search(looks % (v, v, v, v), near)


def findings(path):
    """Returns [(line, kind, hard, text)] for one file. line: from 1. kind: a key of HARD or
    LISTED_KINDS, or "unreadable" (with line 0). hard: no table entry excuses the finding."""
    kind = kind_of(path)
    if kind is None:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
    except OSError as e:
        return [(0, "unreadable", True, str(e))]
    lines = code_of(lines, kind)
    out = []
    for i, line in enumerate(lines):
        for name, rx in HARD.items():
            if not rx.search(line):
                continue
            if name == "cd-pwd":  # each x=$( ) needs its own "||" or "&&"; none after local etc.
                bad = re.search(r"(?<![\w-])(local|export|readonly|declare|typeset)\s[^;|&]*\$\(\s*cd\s", line) or len(
                    rx.findall(line)
                ) != len(CHECKED_CD.findall(line))
            elif name == "fork":  # not inside a C string: a message may name fork()
                bad = kind == "c" and rx.search(re.sub(r'"(\\.|[^"\\])*"', '""', line)) and fork_unchecked(lines, i)
            else:
                bad = True
            if bad:  # a document may quote a faulty form: there the table must count it
                out.append((i + 1, name, kind != "page", line.strip()))
        for name, rx in LISTED_KINDS.items():
            if name == "kill-method" and kind != "python":
                continue
            if rx.search(line):
                out.append((i + 1, name, False, line.strip()))
    return out


def walk(tree):
    """Yields (path, its name in its table, that table's directory) for every file that is read;
    the directory is None for the tree's own table."""
    for top, dirs, names in os.walk(tree):
        at = os.path.relpath(top, tree)
        dirs[:] = sorted(d for d in dirs if d not in UNREAD and not store(os.path.join(top, d)) and not BUILT.match(os.path.normpath(os.path.join(at, d))))
        parts = at.split(os.sep)
        inside = os.path.join(tree, *parts[: parts.index(HOOKS) + 1]) if HOOKS in parts else None
        for n in sorted(names):
            p = os.path.join(top, n)
            if n not in (THIS, os.path.basename(TABLE)) and not (n.startswith("dangerous_calls.") and n.endswith(".tsv")) and not os.path.islink(p):
                yield p, os.path.relpath(p, inside or tree), inside


LEAN = "lean"  # a tree may lack this top directory: see absent_lean()


def absent_lean(tree, table):
    """When nothing named LEAN stands at the tree's top (not even a file or a broken link), drops the
    lines for files below LEAN/ from a table from tables(). Returns (the rest of the table, the number
    of table lines dropped)."""
    if os.path.lexists(os.path.join(tree, LEAN)):
        return table, 0
    below = [path for path in table if path.startswith(LEAN + "/")]
    return {p: k for p, k in table.items() if p not in below}, sum(len(table[p]) for p in below)


def passed_over(n):
    """The closing line's tail when n table lines were passed over; "" for 0."""
    if not n:
        return ""
    return "; %d %s of the table %s not compared, because this tree has no %s/" % (
        n, "line" if n == 1 else "lines", "was" if n == 1 else "were", LEAN)


def judge(tree, table, homes=None):
    """Returns a message for each finding; none if all is as expected. table: the tree's table merged
    with the further tables, from tables(), less the lines that absent_lean() drops. The tables of
    HOOKS directories are read here. homes: for a file of a further table, that table's name, used in
    its findings."""
    said, counts, folders = [], {}, set()
    for path, rel, inside in walk(tree):
        if inside:
            folders.add(inside)
        for line, name, hard, text in findings(path):
            if hard:
                said.append("%s:%d: %s: %s" % (os.path.relpath(path, tree), line, name, text[:160]))
            else:
                counts.setdefault(inside, {}).setdefault(rel, {}).setdefault(name, []).append(line)
    tables = {None: table}
    for inside in sorted(folders):
        own = os.path.join(inside, os.path.basename(TABLE))
        if os.path.isfile(own):
            tables[inside], wrong = listed(own, os.path.relpath(own, tree))
            said += wrong
        else:
            tables[inside] = {}
            if counts.get(inside):
                said.append("%s: no table %s: its files are listed there" % (os.path.relpath(inside, tree), os.path.basename(TABLE)))
    for inside, of_it in tables.items():
        have_all = counts.get(inside, {})
        name_of = os.path.relpath(os.path.join(inside, os.path.basename(TABLE)), tree) if inside else os.path.basename(TABLE)
        for rel in sorted(set(have_all) | set(of_it)):
            have, want = have_all.get(rel, {}), of_it.get(rel, {})
            for name in sorted(set(have) | set(want)):
                n, m = len(have.get(name, [])), want.get(name, (0, ""))[0]
                if n != m:
                    said.append(
                        "%s: %d line(s) of the kind %s, %s says %d (lines %s)"
                        % (rel, n, name, (homes or {}).get(rel, name_of) if inside is None else name_of, m, ",".join(map(str, have.get(name, []))) or "none")
                    )
    return said


SAMPLES = [  # (file, text, findings wanted): faulty texts, and sound ones that give no faulty form
    ("a.sh", 'W=$(cd "$1" && pwd)\n', ["cd-pwd"]),
    ("b.sh", 'W=$(cd "$1" && pwd) || exit 2\n', []),
    ("c.sh", 'f() { local W=$(cd "$1" && pwd) || exit 2; }\n', ["cd-pwd"]),
    ("c2.sh", 'A=$(cd "$1" && pwd) B=$(cd "$2" && pwd) || exit 2\n', ["cd-pwd"]),
    ("c3.sh", 'V=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")\nT=$(cd "$1" && pwd)/x || exit 2\n', ["cd-pwd"]),
    ("c5.sh", 'V=$(cd "$(dirname "$2")" && pwd)/$(basename "$2") || exit 2\nV=$(cd "$(dirname "$2")" && pwd)/${2##*/} || exit 2\n', ["cd-pwd"]),
    ("c6.sh", 'W=$(cd "$(dirname "$0")/../.." && pwd)\nX=$(cd "$1"; pwd) || exit 2\nf() { local a=1 W=$(cd "$1" && pwd) || exit 2; }\n', ["cd-pwd", "cd-pwd", "cd-pwd"]),
    ("g3.sh", "trap 'kill 0' EXIT\nbash -c \"kill -9 -1\"\n", ["kill-all", "kill", "kill-all", "kill"]),
    ("g4.sh", "kill -KILL $(cat $J/pid.k)\nkill $(<$J/pid)\n", ["kill-from-file", "kill", "kill-from-file", "kill"]),
    ("g6.sh", 'p=$(cat $J/pid.k); kill -0 $bg && [ "$(cat /proc/$p/comm)" = cp_commit ] && kill -KILL "$p"\n', ["kill"]),
    ("g5.py", "os.kill(int(open(p).read()), 9)\n", ["kill-from-file", "kill"]),
    ("g2.sh", 'kill -1 "$pid"\nkill -9 -1 2>/dev/null\n', ["kill", "kill-all", "kill"]),
    ("s4.cpp", 'const char *s = "/tmp/*";\nint r = kill(-1, 9);\n', ["kill-all", "kill"]),
    ("c4.sh", 'A=$(cd "$1" && pwd) || exit 2; B="$(cd "$2" && pwd)" || exit 2\n', []),
    ("d.sh", 'rm -rf "$W/tmp"\n', ["rm-var-slash", "rm-r"]),
    ("e.sh", 'rm -rf "${W:?}/tmp"\n', ["rm-r"]),
    ("f.sh", "rm -f -r $W/x\n", ["rm-var-slash", "rm-r"]),
    ("g.sh", "kill -9 -1\n", ["kill-all", "kill"]),
    ("h.sh", "kill -- -1\nkill 0\n", ["kill-all", "kill", "kill-all", "kill"]),
    ("i.sh", 'kill -0 "$p"\n', ["kill"]),
    ("j.sh", 'sleep 9 &\np=$!\n# rm -rf "$W/tmp" in a comment\n', ["bang-pid"]),
    ("k.sh", "pkill -f \"$x\"\nfind \"$d\" -name '*.o' -delete\n", ["pkill", "find-delete"]),
    ("l.cpp", "pid_t child = fork();\nif (child == 0) _exit(0);\nkill(child, SIGKILL);\n", ["fork", "kill"]),
    ("m.cpp", "pid_t child = fork();\nif (child < 0) abort();\nwaitpid(child, nullptr, 0);\n", []),
    ("n.cpp", "if (!fork()) _exit(0);\n// kill(-1, SIGKILL) in a comment\n", ["fork"]),
    ("o.cpp", "int r = kill(-1, SIGKILL);\nstd::filesystem::remove_all(p);\n", ["kill-all", "kill", "remove_all"]),
    ("p.py", "import os, shutil\nos.kill(0, 9)\nshutil.rmtree(d)\nq.terminate()\n", ["kill-all", "kill", "rmtree", "kill-method"]),
    ("p3.py", "self.addCleanup(proc.kill)\natexit.register(p.terminate, )\n", ["kill-method", "kill-method"]),
    ("p2.py", "self.addCleanup(shutil.rmtree, d)\natexit.register(shutil.rmtree, p, True)\nfrom shutil import rmtree\natexit.register(rmtree, p)\n", ["rmtree", "rmtree", "rmtree"]),
    ("q.py", 'subprocess.run(["rm", "-rf", d])\nsubprocess.run(["kill", "-9", str(p)])\n', ["rm-r", "kill"]),
    ("r2.md", "Prose may say kill(-1, SIGKILL) or rm -rf $W/tmp.\n\x60\x60\x60\nkill -9 -1\n\x60\x60\x60\n", ["kill-all", "kill"]),
    ("r3.md", "A list:\n\n    rm -rf $W/tmp\n~~~\nkill 0\n~~~\n- an item\n    with kill -9 -1 said in prose\n", ["rm-var-slash", "rm-r", "kill-all", "kill"]),
    ("r.md", "Run `rm -rf $OUT/` when done.\n", ["rm-var-slash", "rm-r"]),
    ("s.cpp", "pid_t child = fork();\nfor (int i = 0; i >= 0; i--) {}\n", ["fork"]),
    ("s3.cpp", 'pid_t child = fork();\nif (child < 0) fprintf(stderr, "fork() failed");\nputs("a fork() here");\n', []),
    ("s2.cpp", "if ((child = fork()) < 0) abort();\nelse if (!child) _exit(0);\nif (!(other = fork())) _exit(0);\n", ["fork"]),
    ("t.cpp", "/* a block\n   fork() and kill(-1, SIGKILL) in prose */ int x;\nif (x < 0) { // kill(-1, 9) would end all\n}\n", []),
    ("u.sh", "echo done  # rm -rf \"$W/tmp\" is not done\nrm -rf \"$W/x\"  # but this is\n", ["rm-var-slash", "rm-r"]),
]


def teeth(folder):
    """Self-test in the empty directory `folder`. Returns one message per case that fails. Leaves
    files in `folder`: the caller removes it."""
    said = []
    for name, text, wanted in SAMPLES:
        p = os.path.join(folder, name)
        with open(p, "w") as f:
            f.write(text)
        got = sorted((k, h) for _, k, h, _ in findings(p))
        os.unlink(p)
        want = sorted((k, k in HARD and not name.endswith(PAGE)) for k in wanted)  # hard: a faulty form in code, not in a document
        if got != want:
            said.append("sample %s: found %s, wanted %s" % (name, got, want))
    # The table: a faulty form in code is reported whatever the table says, and its line is not
    # counted, so an entry for it is a wrong count too. wanted: the number of messages.
    with open(os.path.join(folder, "x.sh"), "w") as f:
        f.write('W=$(cd "$1" && pwd)\nrm -r "$tmp"\n')
    for listed, wanted in (
        ({"x.sh": {"cd-pwd": (1, "a reason"), "rm-r": (1, "a reason")}}, 2),  # the form, and its entry (no line counted)
        ({"x.sh": {"rm-r": (2, "a reason")}}, 2),  # the form, and one line fewer than listed
        ({"x.sh": {"cd-pwd": (1, "a reason")}}, 3),  # the form, its entry, and an rm -r line without an entry
    ):
        got = judge(folder, listed)
        if len(got) != wanted:
            said.append("the lists %s: %d finding(s), wanted %d: %s" % (listed, len(got), wanted, got))
    os.unlink(os.path.join(folder, "x.sh"))
    # Directories: HOOKS has a table of its own; what BUILT matches is not read, but a directory of
    # such a name deeper in the tree is.
    tree = os.path.join(folder, "w")
    for rel in ("a/hooks/y.sh", "bin/y.sh", "prover/build/y.sh", "a/bin/y.sh", "a/build/y.sh"):
        os.makedirs(os.path.dirname(os.path.join(tree, rel)), exist_ok=True)
        with open(os.path.join(tree, rel), "w") as f:
            f.write('rm -r "$tmp"\n')
    deeper = {"a/bin/y.sh": {"rm-r": (1, "a reason")}, "a/build/y.sh": {"rm-r": (1, "a reason")}}
    got = judge(tree, deeper)
    wanted = ["a/hooks: no table %s: its files are listed there" % os.path.basename(TABLE),
              "y.sh: 1 line(s) of the kind rm-r, a/hooks/%s says 0 (lines 1)" % os.path.basename(TABLE)]
    if sorted(got) != sorted(wanted):  # only the missing table and the line that it should count
        said.append("the folders, without hooks' table: found %s, wanted %s" % (got, wanted))
    with open(os.path.join(tree, "a/hooks/" + os.path.basename(TABLE)), "w") as f:
        f.write("y.sh\trm-r\t1\ta reason\n")
    got = judge(tree, deeper)
    if got:
        said.append("the folders, with hooks' table: %d finding(s), wanted 0: %s" % (len(got), got))
    # Folders that store() takes for a tool's own are not read: a virtual environment, a version-control
    # store (git's HEAD and objects/, or another tool's wc.db) and a package install's build/ or *.egg-info beside
    # its pyproject.toml or setup.py. A build/ with no package beside it and a .github/ are read.
    tree = os.path.join(folder, "s")
    for rel, text in (("venv/pyvenv.cfg", "home = /usr/bin\n"), ("venv/bin/y.sh", 'rm -r "$tmp"\n'),
                      ("vcs/HEAD", "ref\n"), ("vcs/objects/o.txt", ""), ("vcs/y.sh", 'rm -r "$tmp"\n'),
                      ("pkg/pyproject.toml", ""), ("pkg/build/y.sh", 'rm -r "$tmp"\n'),
                      ("wcopy/wc.db", ""), ("wcopy/y.sh", 'rm -r "$tmp"\n'),
                      ("pkg2/setup.py", ""), ("pkg2/x.egg-info/y.sh", 'rm -r "$tmp"\n'),
                      ("tools/build/y.sh", 'rm -r "$tmp"\n'), (".github/y.sh", 'rm -r "$tmp"\n')):
        os.makedirs(os.path.dirname(os.path.join(tree, rel)), exist_ok=True)
        with open(os.path.join(tree, rel), "w") as f:
            f.write(text)
    got = judge(tree, {})
    if sorted(g.split(":")[0] for g in got) != [".github/y.sh", "tools/build/y.sh"]:
        said.append("folders that a tool makes (virtual environment, version-control store, package build): found %s, wanted one finding each in .github/y.sh and tools/build/y.sh" % got)
    # absent_lean(): with no lean/ in the tree, table lines below lean/ are passed over and counted;
    # a wrong count on a present file, and every other missing file or top directory (one that is
    # absent, a file, a link that leads nowhere), are still findings.
    tree = os.path.join(folder, "t")
    os.makedirs(os.path.join(tree, "a", "sub"))
    with open(os.path.join(tree, "a/y.sh"), "w") as f:
        f.write('rm -r "$tmp"\n')
    open(os.path.join(tree, "file"), "w").close()
    os.symlink("nowhere", os.path.join(tree, "link"))
    one, two = {"rm-r": (1, "a reason")}, {"rm-r": (2, "a reason")}
    both = {"rm-r": (1, "a reason"), "rmtree": (1, "a reason")}
    for listed, wanted, count in (
        ({"lean/y.sh": one, "a/y.sh": one}, [], 1),
        ({"lean/y.sh": both, "a/y.sh": one}, [], 2),
        ({"lean/y.sh": one, "a/y.sh": two}, ["a/y.sh"], 1),
        ({"gone/y.sh": one, "a/y.sh": one, "a/sub/z.sh": one, "link/z.sh": one, "file/z.sh": one, "z.sh": one},
         ["a/sub/z.sh", "file/z.sh", "gone/y.sh", "link/z.sh", "z.sh"], 0),
        ({"gone/y.sh": both, "a/y.sh": one, "leaner/y.sh": one, "a/lean/y.sh": one},
         ["a/lean/y.sh", "gone/y.sh", "gone/y.sh", "leaner/y.sh"], 0),
    ):
        kept, n = absent_lean(tree, listed)
        got = judge(tree, kept)
        if sorted(g.split(":")[0] for g in got) != wanted or n != count:
            said.append("table lines when the tree has no lean/ (%s): found %s and %d passed over, wanted findings for %s and %d" % (listed, got, n, wanted, count))
    at = os.path.join(tree, LEAN)
    for kind in ("file", "link that leads nowhere", "folder"):  # any lean at the top: its lines are compared
        if kind == "file":
            open(at, "w").close()
        elif kind == "folder":
            os.mkdir(at)
        else:
            os.symlink("nowhere", at)
        kept, n = absent_lean(tree, {"lean/y.sh": one, "a/y.sh": one})
        got = judge(tree, kept)
        if n or [g.split(":")[0] for g in got] != ["lean/y.sh"]:
            said.append("with a %s named lean, its table lines must be compared: %d passed over, findings %s" % (kind, n, got))
        if kind != "folder":
            os.unlink(at)
    tails = (passed_over(0), passed_over(1), passed_over(2))
    if tails != ("", "; 1 line of the table was not compared, because this tree has no lean/",
                 "; 2 lines of the table were not compared, because this tree has no lean/"):
        said.append("the closing line's tail on table lines passed over reads %s" % (tails,))
    return said


def main():
    tree = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."))
    with tempfile.TemporaryDirectory(prefix="dangerous-calls-") as folder:
        wrong = teeth(folder)
    if wrong:
        print("\n".join(wrong))
        print("dangerous_calls_test.py: the search failed on %d of its %d samples" % (len(wrong), len(SAMPLES)))
        return 2
    table, said, homes = tables()
    table, n = absent_lean(tree, table)
    said += judge(tree, table, homes)
    print("\n".join(said + ["dangerous_calls_test.py: %d samples as expected, %d finding(s)%s" % (len(SAMPLES), len(said), passed_over(n))]))
    return 1 if said else 0


if __name__ == "__main__":
    sys.exit(main())
